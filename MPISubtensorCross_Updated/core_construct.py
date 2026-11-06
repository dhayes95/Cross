"""TT core construction by projection-enhanced interpolative decomposition (PEID).

Algorithms 3.1 (TT-PEID-Par) and 3.2 (TT-PEID-Seq) of "Systematic cross" and
their right-to-left variants, with the oversampling sets of the paper, on top
of the block-form entry access of tt_utils:

  peid_par_ltr_adap / peid_par_rtl_adap   Algorithm 3.1, serial
  peid_seq_ltr_adap / peid_seq_rtl_adap   Algorithm 3.2, serial
  peid_par_sliced                         Algorithm 3.1, distributed (MPI)

The serial routines keep the signatures of the collaborator's core_construct.py
(tensor_entry, FI, FJ, sample_row, sample_col, tensor_dim, adap_tol) and accept
either a plain entry function f(indices, dims) or a tt_utils.FunctionalTensor
(block or separable entry handle).  Right-to-left variants run the
left-to-right code on the reversed tensor.

What is different from the straightforward implementation:
  * every unfolding submatrix is fetched as a rectangle (row parts x column
    parts) through fetch_outer: a separable handle evaluates it in
    O(rows + cols) feature work plus one combine, a plain entry function sees
    chunked index arrays built once, without repeat/tile/column_stack passes;
  * the oblique-projection solves M^+ B (M is (r+p) x r, B has n r columns)
    use an LU solve when M is square and a Householder QR of M otherwise,
    instead of the SVD-based LAPACK gelsd that numpy.linalg.lstsq selects;
  * the rows of F_{<=j} in Algorithm 3.2 are formed for all oversampled
    multi-indices at once with batched matrix products;
  * the distributed Algorithm 3.1 splits every unfolding by its mode index
    (the TT core slab of those mode values is then local), keeps the TSQR Q
    factor distributed, hands the (r+p) x r boundary matrices between
    dimension groups, and with fewer ranks than dimensions gives each rank a
    run of consecutive dimensions (all QRs first, boundary exchange, solves).
"""
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import numpy as np
try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:                                   # pragma: no cover
    _threadpool_limits = None
try:
    from scipy.linalg import (solve_triangular as _solve_triangular, lstsq as _sp_lstsq,
                              cholesky as _cholesky, qr as _sp_qr)
except ImportError:                                   # pragma: no cover
    _solve_triangular = _sp_lstsq = _cholesky = _sp_qr = None
try:
    from mpi4py import MPI
except ImportError:                                   # pragma: no cover
    MPI = None

import tt_utils as _tt
from tt_utils import (FunctionalTensor, sample_nested, set_conversion, rank_cutoff,
                      cheap_sample, Large_ravel, Large_unravel)


# ---------------------------------------------------------------- data access
class _EntryData:
    """fetch_outer for a plain entry function f(indices, dims) -> values."""
    def __init__(self, fn, dims, chunk=1 << 20):
        self.fn, self.dims, self.chunk = fn, tuple(int(x) for x in dims), chunk

    def fetch_outer(self, rp, cp, k):
        rp = np.asarray(rp, dtype=np.int64); cp = np.asarray(cp, dtype=np.int64)
        nr, nc, d = len(rp), len(cp), len(self.dims)
        out = np.empty((nr, nc), dtype=np.float64)
        if nr == 0 or nc == 0:
            return out
        step = max(1, min(nr, self.chunk // nc))
        idx = np.empty((step, nc, d), dtype=np.int64)
        idx[:, :, k + 1:] = cp[None, :, :]
        for r0 in range(0, nr, step):
            r1 = min(nr, r0 + step); m = r1 - r0
            idx[:m, :, :k + 1] = rp[r0:r1, None, :]
            out[r0:r1] = np.asarray(self.fn(idx[:m].reshape(-1, d), self.dims), dtype=np.float64).reshape(m, nc)
        return out


class _Reversed:
    """The tensor with its dimensions reversed, through the original handle."""
    def __init__(self, data, d):
        self.data, self.d = data, d

    def fetch_outer(self, rp, cp, k):
        rp = np.asarray(rp, dtype=np.int64); cp = np.asarray(cp, dtype=np.int64)
        return np.ascontiguousarray(self.data.fetch_outer(cp[:, ::-1], rp[:, ::-1], self.d - 2 - k).T)


def _as_data(x, dims):
    if isinstance(x, FunctionalTensor) and x.sep is None and x.block_fn is None:
        # entry-by-entry FunctionalTensor: its fetch uses tt_utils' module-level
        # tensor_entry, which only tt_cross_os2_mpi2 sets
        if getattr(_tt, "tensor_entry", None) is None:
            raise ValueError("FunctionalTensor without a block or separable handle outside the cross: pass the entry function f(indices, dims) instead")
        return _EntryData(lambda idx, _dims: x.fetch(idx), dims)
    if hasattr(x, "fetch_outer"):
        return x
    return _EntryData(x, dims)


def _row_parts(prefixes, i_slice):
    """(prefix, i) for every prefix and i, prefix-major: row p*len(i_slice) + pos(i)."""
    P = np.asarray(prefixes, dtype=np.int64).reshape(len(prefixes), -1)
    m = len(i_slice)
    out = np.empty((len(P) * m, P.shape[1] + 1), dtype=np.int64)
    out[:, :-1] = np.repeat(P, m, axis=0)
    out[:, -1] = np.tile(np.asarray(i_slice, dtype=np.int64), len(P))
    return out


def _col_parts(i_slice, suffixes):
    """(i, suffix) for every i and suffix, i-major: column pos(i)*len(suffixes) + s."""
    S = np.asarray(suffixes, dtype=np.int64).reshape(len(suffixes), -1)
    m = len(i_slice)
    out = np.empty((m * len(S), S.shape[1] + 1), dtype=np.int64)
    out[:, 0] = np.repeat(np.asarray(i_slice, dtype=np.int64), len(S))
    out[:, 1:] = np.tile(S, (m, 1))
    return out


# ---------------------------------------------------------------- threads, tall-skinny QR
_NTHREADS = 1
_TSQR_BLOCK = 16384


def default_threads():
    """Threads per rank for the core construction: PEID_THREADS if set, else the
    physical cores this process is bound to (half the affinity set, SMT on)."""
    v = os.environ.get("PEID_THREADS")
    if v:
        return max(1, int(v))
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:                            # pragma: no cover
        ncpu = os.cpu_count() or 1
    return max(1, min(16, ncpu // 2))


def _threaded(fn):
    """Public-routine decorator: resolve nthreads, publish it to the TSQR and set
    the BLAS thread count for the duration of the call."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, nthreads=None, **kw):
        global _NTHREADS
        prev = _NTHREADS
        _NTHREADS = default_threads() if nthreads is None else max(1, int(nthreads))
        try:
            with _blas_threads(_NTHREADS):
                return fn(*args, nthreads=_NTHREADS, **kw)
        finally:
            _NTHREADS = prev
    return wrapper


def _blas_threads(k):
    """Context that sets the BLAS thread count to k (no-op without threadpoolctl)."""
    return _threadpool_limits(limits=int(k)) if _threadpool_limits is not None else nullcontext()


_FETCH_BLOCK = 4096


def _fetch(data, rp, cp, k):
    """data.fetch_outer(rp, cp, k), the rows split over _NTHREADS threads when
    there are enough of them.  numpy ufuncs and GEMMs in a separable handle
    release the GIL, so the handle's combine runs in parallel; a plain entry
    function gains only if it does too."""
    nr = len(rp)
    if _NTHREADS <= 1 or nr < 2 * _FETCH_BLOCK:
        return data.fetch_outer(rp, cp, k)
    edges = np.linspace(0, nr, _NTHREADS + 1).astype(int)
    out = np.empty((nr, len(cp)))
    extra = {}
    if isinstance(data, FunctionalTensor) and data.sep is not None:
        extra["fc"] = data.col_features(np.asarray(cp, dtype=np.int64))      # once, not per thread

    def part(i):
        out[edges[i]:edges[i + 1]] = data.fetch_outer(rp[edges[i]:edges[i + 1]], cp, k, **extra)
    with _blas_threads(1):
        with ThreadPoolExecutor(_NTHREADS) as ex:
            list(ex.map(part, range(_NTHREADS)))
    return out


def _tsqr_local(Y):
    """Householder QR of the tall Y by independent row blocks of _TSQR_BLOCK
    rows, then a QR of the stacked R factors.  The blocks stay in cache, which
    already beats LAPACK's panel factorisation single-threaded (1.45x on
    1.2M x 150), and numpy's QR releases the GIL, so with _NTHREADS > 1 the
    blocks run on Python threads over single-threaded BLAS (2.5x more at 4-8
    threads).  OpenBLAS' own threading does nothing for this shape."""
    m, n = Y.shape
    if m == 0:
        return np.empty((0, n)), np.zeros((n, n))
    if m <= 2 * _TSQR_BLOCK:
        return np.linalg.qr(Y, mode="reduced")
    starts = list(range(0, m, _TSQR_BLOCK))

    def fac(r0):
        return np.linalg.qr(Y[r0:r0 + _TSQR_BLOCK], mode="reduced")
    with _blas_threads(1):
        if _NTHREADS > 1:
            with ThreadPoolExecutor(_NTHREADS) as ex:
                res = list(ex.map(fac, starts))
        else:
            res = [fac(r0) for r0 in starts]
        q2, R = np.linalg.qr(np.vstack([r for _, r in res]), mode="reduced")
        Q = np.empty((m, n))
        offs = np.cumsum([0] + [q.shape[1] for q, _ in res])

        def mul(i):
            q = res[i][0]
            np.matmul(q, q2[offs[i]:offs[i + 1]], out=Q[starts[i]:starts[i] + q.shape[0]])
        if _NTHREADS > 1:
            with ThreadPoolExecutor(_NTHREADS) as ex:
                list(ex.map(mul, range(len(starts))))
        else:
            for i in range(len(starts)):
                mul(i)
    return Q, R


def _tsqr_dist(Y, sub):
    """TSQR over the ranks of `sub` (rows distributed): local _tsqr_local, the
    R factors stacked and factorised on the root, Q completed locally.
    Returns (Q_local, R) with R on every rank."""
    n = Y.shape[1]
    Q1, R1 = _tsqr_local(Y)
    Rs = sub.gather(R1, root=0)
    if sub.Get_rank() == 0:
        q2, R = np.linalg.qr(np.vstack(Rs), mode="reduced")
        offs = np.cumsum([0] + [r.shape[0] for r in Rs])
        blocks = [q2[offs[i]:offs[i + 1]] for i in range(len(Rs))]
    else:
        R, blocks = None, None
    Q2 = sub.scatter(blocks, root=0)
    R = sub.bcast(R, root=0)
    return Q1 @ Q2, R


# ---------------------------------------------------------------- linear algebra
CHOLQR_COND = 1e7          # Cholesky-QR2 below this condition number of Y, Householder above
_BASIS_PATHS = {"cholqr2": 0, "householder": 0}


def _gram_cond(G):
    ev = np.linalg.eigvalsh(G)
    return np.sqrt(ev[-1] / ev[0]) if ev[0] > 0 else np.inf


def _cholqr_pass(Y, G):
    """One Cholesky-QR pass: R = chol(G), Q = Y R^{-1} (rows of Y stay row-major)."""
    R = _cholesky(G, lower=False, check_finite=False)
    Q = _solve_triangular(R.T, Y.T, lower=True, check_finite=False).T if Y.shape[0] else Y.copy()
    return Q, R


def _truncate(R, rk, adap_tol):
    Ur, s, _ = np.linalg.svd(R)
    if adap_tol is not None:
        rk = rank_cutoff(s, adap_tol)
    return Ur[:, :max(1, min(int(rk), Ur.shape[1]))]


def _chol_qr(Y, gram):
    """Cholesky-QR2 of the tall Y (two Gram / Cholesky / triangular-solve
    passes, all BLAS3, orthogonal to machine precision when cond(Y) <
    CHOLQR_COND); gram(Z) returns Z^T Z (summed over ranks when distributed).
    The Gram eigenvalues certify the condition number for the price of one
    GEMM; returns None when they do not (caller uses Householder).

    With oversampling the extra columns are usually almost in the span of the
    pivot columns, so for a fast-decaying spectrum Y is numerically
    rank-deficient (cond 1e15 and beyond) and only Householder is reliable;
    shifted Cholesky-QR3 was tried and fails there too."""
    G = gram(Y)
    ev = np.linalg.eigvalsh(G)
    if not (ev[0] > 0 and np.sqrt(ev[-1] / ev[0]) < CHOLQR_COND):
        return None
    try:
        Q1, R1 = _cholqr_pass(Y, G)
        Q, R2 = _cholqr_pass(Q1, gram(Q1))
        return Q, R2 @ R1, "cholqr2"
    except np.linalg.LinAlgError:
        return None


def _basis(Y, rk, adap_tol):
    """Orthonormal basis of the dominant column space of Y (tall): Q U_R[:, :rk].
    Cholesky-based QR (all BLAS3, 2-3x faster single-threaded than LAPACK's
    tall-skinny Householder QR), Householder as the last resort."""
    out = _chol_qr(Y, lambda Z: Z.T @ Z) if (_cholesky is not None and Y.shape[0] >= Y.shape[1]) else None
    if out is None:
        q, r = _tsqr_local(Y)
        out = (q, r, "householder")
    Q, R, path = out
    _BASIS_PATHS[path] = _BASIS_PATHS.get(path, 0) + 1
    return Q @ _truncate(R, rk, adap_tol)


def _basis_dist(Y, sub, rk, adap_tol):
    """_basis for Y distributed by rows over `sub`; returns this rank's rows of
    the basis.  The Cholesky passes need one all-reduce of the (r+p)^2 Gram
    matrix each; the Householder fallback is the TSQR of tt_utils."""
    n = Y.shape[1]

    def gram(Z):
        G = np.empty((n, n)); sub.Allreduce(np.ascontiguousarray(Z.T @ Z), G, op=MPI.SUM); return G
    out = _chol_qr(Y, gram) if _cholesky is not None else None
    ok = sub.allreduce(out is not None, op=MPI.LAND)
    if not ok:
        Q, R = _tsqr_dist(Y, sub)
        out = (Q, R, "householder")
    Q, R, path = out
    _BASIS_PATHS[path] = _BASIS_PATHS.get(path, 0) + 1
    Ur = sub.bcast(_truncate(R, rk, adap_tol) if sub.Get_rank() == 0 else None, root=0)
    return Q @ Ur


def _pinv_solve(M, B):
    """M^+ B.  LU when M is square; QR (Householder, column rank checked) when
    M has more rows than columns; LAPACK gelsy otherwise or when R is singular."""
    M = np.asarray(M, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    m, n = M.shape
    if m == n:
        try:
            return np.linalg.solve(M, B)
        except np.linalg.LinAlgError:
            pass
    elif m > n and _solve_triangular is not None:
        q, r = np.linalg.qr(M, mode="reduced")
        dg = np.abs(np.diag(r))
        if dg.min() > 1e-13 * dg.max():
            return _solve_triangular(r, q.T @ B, lower=False, check_finite=False)
    if _sp_lstsq is not None:
        return _sp_lstsq(M, B, lapack_driver="gelsy", check_finite=False)[0]
    return np.linalg.lstsq(M, B, rcond=None)[0]


def _tt_rows(cores, idx):
    """Rows of the TT built from cores[0..j] at the multi-indices idx (R, j+1): (R, r_j)."""
    idx = np.asarray(idx, dtype=np.int64).reshape(len(idx), -1)
    v = cores[0][0, idx[:, 0], :]                                   # (R, r_0)
    for k in range(1, idx.shape[1]):
        v = np.matmul(v[:, None, :], cores[k][:, idx[:, k], :].transpose(1, 0, 2))[:, 0, :]
    return v


# ---------------------------------------------------------------- index sets
def _want(pivots, target_rank):
    """Output rank of an unfolding: the pivot count, or target_rank when given
    (capped by the r+p available columns inside _truncate)."""
    return len(pivots) if target_rank is None else max(1, int(target_rank))


def _as_lists(FI, FJ):
    return ([[[int(v) for v in x] for x in lv] for lv in FI],
            [[[int(v) for v in x] for x in lv] for lv in FJ])


class _seeded:
    """Run the draws (tt_utils' cheap_sample uses the module-level random) from
    `seed` and hand the caller's random state back afterwards; seed=None draws
    from the caller's stream."""
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        if self.seed is not None:
            self.state = random.getstate(); random.seed(self.seed)

    def __exit__(self, *exc):
        if self.seed is not None:
            random.setstate(self.state)


def _nested_sets(FI, FJ, sample_row, sample_col, dims, seed=None):
    """Oversampled sets of Algorithm 3.1 (rows nested, columns uniform) as int
    arrays, with I[j] = positions of uFI[j] inside the row set uFI[j-1] x I_j."""
    FIl, FJl = _as_lists(FI, FJ)
    d = len(dims)
    sr = [0] * (d - 1) if sample_row is None else [int(x) for x in sample_row]
    sc = [0] * (d - 1) if sample_col is None else [int(x) for x in sample_col]
    with _seeded(seed):
        uFI, uFJ = sample_nested(FIl, FJl, sr, sc, list(dims), 0)
    I, _ = set_conversion(uFI, uFJ, list(dims))
    uFI = [np.asarray(x, dtype=np.int64).reshape(len(x), j + 1) for j, x in enumerate(uFI)]
    uFJ = [np.asarray(x, dtype=np.int64).reshape(len(x), d - j - 1) for j, x in enumerate(uFJ)]
    I = [np.asarray(x, dtype=np.int64) for x in I]
    return uFI, uFJ, I


def _uniform_extra(pivots, box, count):
    """count multi-indices drawn uniformly from the box (list of sizes), not in pivots."""
    box = [int(x) for x in box]
    unavail = [Large_ravel(list(x), box) for x in pivots]
    picks = cheap_sample(0, math.prod(box), unavail, int(count))
    return [list(Large_unravel(j, box)) for j in picks]


def _independent_sets(FI, FJ, sample_row, sample_col, dims, seed=None):
    """Oversampled sets of Algorithm 3.2: rows and columns of every unfolding
    drawn independently and uniformly outside the pivots."""
    FIl, FJl = _as_lists(FI, FJ)
    d = len(dims)
    sr = [0] * (d - 1) if sample_row is None else [int(x) for x in sample_row]
    sc = [0] * (d - 1) if sample_col is None else [int(x) for x in sample_col]
    with _seeded(seed):
        uFI = [np.asarray(FIl[j] + _uniform_extra(FIl[j], dims[:j + 1], sr[j]), dtype=np.int64).reshape(-1, j + 1) for j in range(d - 1)]
        uFJ = [np.asarray(FJl[j] + _uniform_extra(FJl[j], dims[j + 1:], sc[j]), dtype=np.int64).reshape(-1, d - j - 1) for j in range(d - 1)]
    return uFI, uFJ


def _reverse_sets(FI, FJ, sample_row, sample_col, d):
    FIr = [[tuple(reversed(c)) for c in FJ[d - 2 - j]] for j in range(d - 1)]
    FJr = [[tuple(reversed(r)) for r in FI[d - 2 - j]] for j in range(d - 1)]
    sr = None if sample_col is None else list(sample_col)[::-1]
    sc = None if sample_row is None else list(sample_row)[::-1]
    return FIr, FJr, sr, sc


def _reverse_cores(cores):
    return [np.ascontiguousarray(c.transpose(2, 1, 0)) for c in cores[::-1]]


# ---------------------------------------------------------------- Algorithm 3.1, serial
@_threaded
def peid_par_ltr_adap(tensor_entry, FI, FJ, sample_row, sample_col, tensor_dim, adap_tol=None, seed=None, nthreads=None, target_rank=None):
    """TT-PEID-Par (Algorithm 3.1), left to right, one process."""
    dims = [int(x) for x in tensor_dim]; d = len(dims); n = d - 1
    data = _as_data(tensor_entry, dims)
    uFI, uFJ, I = _nested_sets(FI, FJ, sample_row, sample_col, dims, seed)
    cores, Uprev = [], None
    for j in range(n):
        prefixes = np.zeros((1, 0), dtype=np.int64) if j == 0 else uFI[j - 1]
        R = len(prefixes)
        Y = _fetch(data, _row_parts(prefixes, np.arange(dims[j])), uFJ[j], j)      # (R n_j, r+p)
        U = _basis(Y, _want(FI[j], target_rank), adap_tol)                              # (R n_j, rk)
        rk = U.shape[1]
        if j == 0:
            cores.append(U.reshape(1, dims[0], rk))
        else:
            M = Uprev[I[j - 1]]                                                         # (r+p, r_{j-1})
            cores.append(_pinv_solve(M, U.reshape(R, dims[j] * rk)).reshape(M.shape[1], dims[j], rk))
        Uprev = U
    X = _fetch(data, uFI[n - 1], np.arange(dims[n], dtype=np.int64)[:, None], n - 1)   # (r+p, n_{d-1})
    M = Uprev[I[n - 1]]
    cores.append(_pinv_solve(M, X).reshape(M.shape[1], dims[n], 1))
    return cores


@_threaded
def peid_par_rtl_adap(tensor_entry, FI, FJ, sample_row, sample_col, tensor_dim, adap_tol=None, seed=None, nthreads=None, target_rank=None):
    """TT-PEID-Par (Algorithm 3.1) run right to left: the left-to-right routine
    on the reversed tensor, cores transposed back."""
    dims = [int(x) for x in tensor_dim]; d = len(dims)
    data = _Reversed(_as_data(tensor_entry, dims), d)
    FIr, FJr, sr, sc = _reverse_sets(FI, FJ, sample_row, sample_col, d)
    return _reverse_cores(peid_par_ltr_adap(data, FIr, FJr, sr, sc, dims[::-1], adap_tol, seed, nthreads=nthreads, target_rank=target_rank))


# ---------------------------------------------------------------- Algorithm 3.2, serial
@_threaded
def peid_seq_ltr_adap(tensor_entry, FI, FJ, sample_row, sample_col, tensor_dim, adap_tol=None, seed=None, nthreads=None, target_rank=None):
    """TT-PEID-Seq (Algorithm 3.2), left to right, one process."""
    dims = [int(x) for x in tensor_dim]; d = len(dims); n = d - 1
    data = _as_data(tensor_entry, dims)
    uFI, uFJ = _independent_sets(FI, FJ, sample_row, sample_col, dims, seed)
    cores = []
    Y = _fetch(data, np.arange(dims[0], dtype=np.int64)[:, None], uFJ[0], 0)         # (n_0, r+p)
    for j in range(n):
        U = _basis(Y, _want(FI[j], target_rank), adap_tol)
        rk = U.shape[1]
        cores.append(U.reshape(-1, dims[j], rk))
        F = _tt_rows(cores, uFI[j])                                                     # (r+p, rk)
        if j < n - 1:
            X = _fetch(data, uFI[j], _col_parts(np.arange(dims[j + 1]), uFJ[j + 1]), j)   # (r+p, n_{j+1} (r+p))
            Y = _pinv_solve(F, X).reshape(rk * dims[j + 1], -1)
        else:
            X = _fetch(data, uFI[j], np.arange(dims[n], dtype=np.int64)[:, None], j)       # (r+p, n_{d-1})
            cores.append(_pinv_solve(F, X).reshape(rk, dims[n], 1))
    return cores


@_threaded
def peid_seq_rtl_adap(tensor_entry, FI, FJ, sample_row, sample_col, tensor_dim, adap_tol=None, seed=None, nthreads=None, target_rank=None):
    """TT-PEID-Seq (Algorithm 3.2) run right to left (reversed tensor)."""
    dims = [int(x) for x in tensor_dim]; d = len(dims)
    data = _Reversed(_as_data(tensor_entry, dims), d)
    FIr, FJr, sr, sc = _reverse_sets(FI, FJ, sample_row, sample_col, d)
    return _reverse_cores(peid_seq_ltr_adap(data, FIr, FJr, sr, sc, dims[::-1], adap_tol, seed, nthreads=nthreads, target_rank=target_rank))


def _contiguous_partition(work, P):
    """Split range(len(work)) into P contiguous runs minimising the largest run
    total (dynamic programme; d and P are small).  Runs are never empty."""
    d = len(work); P = min(P, d)
    pre = np.concatenate(([0.0], np.cumsum(np.asarray(work, dtype=float))))
    INF = float("inf")
    best = [[INF] * (d + 1) for _ in range(P + 1)]; cut = [[0] * (d + 1) for _ in range(P + 1)]
    best[0][0] = 0.0
    for k in range(1, P + 1):
        for j in range(k, d + 1):
            for i in range(k - 1, j):
                v = max(best[k - 1][i], pre[j] - pre[i])
                if v < best[k][j]:
                    best[k][j], cut[k][j] = v, i
    parts, j = [], d
    for k in range(P, 0, -1):
        i = cut[k][j]; parts.append(list(range(i, j))); j = i
    return parts[::-1]


# ---------------------------------------------------------------- Algorithm 3.1, distributed
def _activate(comm, max_peid_ranks):
    world_rank, world_size = comm.Get_rank(), comm.Get_size()
    peid_size = world_size if max_peid_ranks is None else max(1, min(int(max_peid_ranks), world_size))
    active = world_rank < peid_size
    pcomm = comm.Split(0 if active else MPI.UNDEFINED, world_rank)
    return pcomm, active, world_rank, world_size, peid_size


def _finish(comm, info, world_rank, world_size, peid_size, elapsed):
    info = comm.bcast(info if world_rank == 0 else None, root=0)
    info["local_time"] = elapsed
    info["peid_ranks"] = peid_size
    info["inactive_ranks"] = list(range(peid_size, world_size))
    return info


def _par_engine(data, dims, uFI, uFJ, I, ranks_tt, n_unf, tail, adap_tol, pcomm, world_rank):
    """Dimension-parallel engine of Algorithm 3.1 over the ranks of `pcomm`:
    bases of unfoldings 0..n_unf-1 (one rank group each, the group splitting
    the mode index; with fewer ranks than tasks, runs of consecutive tasks per
    rank), the oblique projections against the previous unfolding's boundary
    matrix M_j = U_j[I_j], and with tail="core" the last core of a
    d = n_unf+1 tensor.  With tail="boundary" the last unfolding's boundary
    matrix is returned instead (on the root of its group, None elsewhere).
    Returns (slabs {dim: (lo, hi, slab)}, M_last, stats)."""
    rank, size = pcomm.Get_rank(), pcomm.Get_size()
    ntask = n_unf + (1 if tail == "core" else 0)
    work = [(1 if j == 0 else len(uFI[j - 1])) * dims[j] * len(uFJ[j]) for j in range(n_unf)]
    if tail == "core":
        work.append(len(uFI[n_unf - 1]) * dims[n_unf])
    if size >= ntask:
        group_sizes = [1] * ntask
        extra = size - ntask
        if extra > 0:
            w = np.asarray(work, dtype=float); prop = w / w.sum()
            alloc = np.floor(prop * extra).astype(int)
            for idx in np.argsort(prop * extra - alloc)[::-1][:extra - int(alloc.sum())]:
                alloc[idx] += 1
            group_sizes = [1 + int(a) for a in alloc]
        groups, start = [], 0
        for g in group_sizes:
            groups.append(list(range(start, start + g))); start += g
        my_dims = [j for j, g in enumerate(groups) if rank in g]
        sub = pcomm.Split(my_dims[0], rank)
        mode = "sliced"
    else:
        parts = _contiguous_partition(work, size)
        groups = [None] * ntask
        for r_, a in enumerate(parts):
            for j in a:
                groups[j] = [r_]
        group_sizes = [1] * ntask
        my_dims = parts[rank]
        sub = MPI.COMM_SELF
        mode = "consecutive_dims"
    roots = [g[0] for g in groups]
    srank, ssize = sub.Get_rank(), sub.Get_size()
    t0 = MPI.Wtime()
    wait = 0.0
    # ---- phase 1: bases (independent across dimensions), boundary matrices sent on
    state = {}
    Mloc = {}                      # dim j -> M_j when its consumer is this rank
    M_last = None
    sends = []
    for j in my_dims:
        nk = dims[j]
        edges = np.linspace(0, nk, ssize + 1).astype(int)
        lo, hi = int(edges[srank]), int(edges[srank + 1])
        i_slice = np.arange(lo, hi); m = hi - lo
        if j < n_unf:
            prefixes = np.zeros((1, 0), dtype=np.int64) if j == 0 else uFI[j - 1]
            R = len(prefixes)
            Y = _fetch(data, _row_parts(prefixes, i_slice), uFJ[j], j) if m else np.empty((0, len(uFJ[j])))
            U = _basis_dist(Y, sub, ranks_tt[j], adap_tol) if ssize > 1 else _basis(Y, ranks_tt[j], adap_tol)
            rk = U.shape[1]
            # pivot rows I[j] of U_j (row p*nk + i): mine are those with i in my slice
            pr_, ik_ = np.divmod(I[j], nk)
            mine = np.flatnonzero((ik_ >= lo) & (ik_ < hi))
            own = (mine, U[pr_[mine] * m + (ik_[mine] - lo), :])
            parts_ = sub.gather(own, root=0) if ssize > 1 else [own]
            if srank == 0:
                Mj = np.empty((len(I[j]), rk))
                for idx, rows_ in parts_:
                    Mj[idx] = rows_
                if j + 1 >= ntask:
                    M_last = Mj
                elif roots[j + 1] == rank:
                    Mloc[j] = Mj
                else:
                    sends.append(pcomm.isend(Mj, dest=roots[j + 1], tag=2000 + j + 1))
            state[j] = (lo, hi, m, R, U)
        else:
            X = _fetch(data, uFI[n_unf - 1], i_slice[:, None], n_unf - 1) if m else np.empty((len(uFI[n_unf - 1]), 0))
            state[j] = (lo, hi, m, len(uFI[n_unf - 1]), X)
    # ---- phase 2: oblique projections against the previous dimension's boundary matrix
    slabs = {}
    for j in my_dims:
        lo, hi, m, R, A = state[j]
        if j == 0:
            slab = A.reshape(1, m, A.shape[1])
        else:
            if srank == 0:
                if (j - 1) in Mloc:
                    M = Mloc[j - 1]
                else:
                    tw = MPI.Wtime()
                    M = pcomm.recv(source=roots[j - 1], tag=2000 + j)
                    wait += MPI.Wtime() - tw
            else:
                M = None
            if ssize > 1:
                M = sub.bcast(M, root=0)
            if j < n_unf:
                rk = A.shape[1]
                slab = _pinv_solve(M, A.reshape(R, m * rk)).reshape(M.shape[1], m, rk)
            else:
                slab = _pinv_solve(M, A).reshape(M.shape[1], m, 1)
        slabs[j] = (lo, hi, slab)
    for s_ in sends:
        s_.wait()
    elapsed = MPI.Wtime() - t0
    tmax = sub.allreduce(elapsed, op=MPI.MAX) if ssize > 1 else elapsed
    stats = {"group_sizes": group_sizes, "groups": groups, "elapsed": elapsed, "wait": wait,
             "my_dims": my_dims, "mode": mode, "group_time": tmax, "root": srank == 0}
    if sub is not MPI.COMM_SELF:
        sub.Free()
    return slabs, M_last, stats


@_threaded
def peid_par_sliced(tensor_entry, FI, FJ, sample_row, sample_col, tensor_dim, comm, adap_tol=None,
                    max_peid_ranks=None, return_info=False, seed=None, nthreads=None, target_rank=None):
    """TT-PEID-Par (Algorithm 3.1) distributed over `comm`.

    Cores are returned distributed: on every rank a list of d entries, entry k
    being (lo, hi, slab) with slab = core_k[:, lo:hi, :] when this rank holds
    that slice of the mode index, else None (tt_utils.gather_tt_cores assembles
    them).  With P >= d ranks the dimensions get rank groups proportional to
    the entries they extract and each group splits its mode index; with P < d
    every rank takes a run of consecutive dimensions whole.  info holds the
    layout and timings (construction only, no gather).
    """
    dims = [int(x) for x in tensor_dim]; d = len(dims); n = d - 1
    pcomm, active, world_rank, world_size, peid_size = _activate(comm, max_peid_ranks)
    cores = [None] * d
    info = None
    elapsed = 0.0
    if active:
        rank = pcomm.Get_rank()
        data = _as_data(tensor_entry, dims)
        sets = _nested_sets(FI, FJ, sample_row, sample_col, dims, seed) if rank == 0 else None
        uFI, uFJ, I = pcomm.bcast(sets, root=0)
        slabs, _, st = _par_engine(data, dims, uFI, uFJ, I, [_want(x, target_rank) for x in FI], n, "core", adap_tol, pcomm, world_rank)
        elapsed = st["elapsed"]
        for j, v in slabs.items():
            cores[j] = v
        res = pcomm.gather([(j, st["group_time"], st["group_time"] - st["wait"]) for j in st["my_dims"]] if st["root"] else [], root=0)
        layout = pcomm.gather([(j, world_rank, v[0], v[1]) for j, v in slabs.items()], root=0)
        if rank == 0:
            ordered = sorted(x for part in res for x in part)
            flat = sorted(x for part in layout for x in part)
            info = {"core_group_sizes": st["group_sizes"], "core_groups": st["groups"],
                    "dimension_times": [x[1] for x in ordered],
                    "dimension_active_times": [x[2] for x in ordered],
                    "construct_time": max(x[1] for x in ordered),
                    "core_layout": [[(r_, lo_, hi_) for (j_, r_, lo_, hi_) in flat if j_ == k] for k in range(d)],
                    "execution_mode": st["mode"],
                    "oversampling": {"rows": [len(x) - len(y) for x, y in zip(uFI, FI)],
                                     "cols": [len(x) - len(y) for x, y in zip(uFJ, FJ)]}}
        pcomm.Free()
    info = _finish(comm, info, world_rank, world_size, peid_size, elapsed)
    return (cores, info) if return_info else cores


@_threaded
def peid_par2_sliced(tensor_entry, FI, FJ, sample_row, sample_col, tensor_dim, comm, adap_tol=None,
                     max_peid_ranks=None, return_info=False, seed=None, nthreads=None, target_rank=None):
    """TT-PEID-Par2 (Algorithm 3.3, two-sided) distributed over `comm`.

    The ranks split into three groups that run concurrently: a left group
    builds cores 0..nL-1 from column-space bases of unfoldings 0..nL-1 (the
    Algorithm 3.1 engine), a right group builds cores d-1..nL+1 from
    row-space bases of unfoldings d-2..nL (the same engine on the reversed
    tensor), and a middle group fetches
    X_{nL-1}(I_{<=nL-1} u ~I, I_nL x (J_{>nL} u ~J)) sliced by i_nL and applies
    the two boundary pseudo-inverses (lines 13-16 of the algorithm).  Needs
    d >= 3 and at least 3 ranks; otherwise falls back to peid_par_sliced.
    Cores come back distributed as in peid_par_sliced."""
    dims = [int(x) for x in tensor_dim]; d = len(dims)
    pcomm, active, world_rank, world_size, peid_size = _activate(comm, max_peid_ranks)
    if d < 3 or peid_size < 3:
        if active:
            pcomm.Free()
        return peid_par_sliced(tensor_entry, FI, FJ, sample_row, sample_col, tensor_dim, comm, adap_tol=adap_tol,
                               max_peid_ranks=max_peid_ranks, return_info=return_info, seed=seed, nthreads=nthreads, target_rank=target_rank)
    nL = d // 2; nR = d - 1 - nL; mid = nL
    cores = [None] * d
    info = None
    elapsed = 0.0
    if active:
        rank, size = pcomm.Get_rank(), pcomm.Get_size()
        data = _as_data(tensor_entry, dims)
        FIr, FJr, srr, scr = _reverse_sets(FI, FJ, sample_row, sample_col, d)
        if rank == 0:
            setsL = _nested_sets(FI, FJ, sample_row, sample_col, dims, seed)
            setsR = _nested_sets(FIr, FJr, srr, scr, dims[::-1], None if seed is None else seed + 1)
        else:
            setsL = setsR = None
        (uFI, uFJ, I), (uFIr, uFJr, Ir) = pcomm.bcast((setsL, setsR), root=0)
        # ---- ranks for the three sides, proportional to the entries each fetches
        wL = sum((1 if j == 0 else len(uFI[j - 1])) * dims[j] * len(uFJ[j]) for j in range(nL))
        wR = sum((1 if j == 0 else len(uFIr[j - 1])) * dims[::-1][j] * len(uFJr[j]) for j in range(nR))
        wM = 1.25 * len(uFI[nL - 1]) * dims[mid] * len(uFIr[nR - 1])      # two solves, no QR, but measured the laggard
        w = np.asarray([wL, wR, wM], dtype=float); prop = w / w.sum()
        alloc = np.floor(prop * (size - 3)).astype(int)
        for idx in np.argsort(prop * (size - 3) - alloc)[::-1][:size - 3 - int(alloc.sum())]:
            alloc[idx] += 1
        PL, PR, PM = (1 + int(a) for a in alloc)
        side = 0 if rank < PL else (1 if rank < PL + PR else 2)
        sub = pcomm.Split(side, rank)
        mid_root = PL + PR
        t0 = MPI.Wtime()
        slabs = {}
        wait = 0.0
        if side == 0:
            sl, M_L, _ = _par_engine(data, dims, uFI, uFJ, I, [_want(x, target_rank) for x in FI], nL, "boundary", adap_tol, sub, world_rank)
            slabs.update(sl)
            if M_L is not None:
                pcomm.send(M_L, dest=mid_root, tag=3001)
        elif side == 1:
            sl, M_R, _ = _par_engine(_Reversed(data, d), dims[::-1], uFIr, uFJr, Ir, [_want(x, target_rank) for x in FIr], nR, "boundary", adap_tol, sub, world_rank)
            for j, (lo, hi, slab) in sl.items():
                slabs[d - 1 - j] = (lo, hi, np.ascontiguousarray(slab.transpose(2, 1, 0)))
            if M_R is not None:
                pcomm.send(M_R, dest=mid_root, tag=3002)
        else:
            srank, ssize = sub.Get_rank(), sub.Get_size()
            edges = np.linspace(0, dims[mid], ssize + 1).astype(int)
            lo, hi = int(edges[srank]), int(edges[srank + 1]); m = hi - lo
            cols_mid = np.ascontiguousarray(uFIr[nR - 1][:, ::-1])                  # suffixes over dims nL+1..d-1
            G = _fetch(data, uFI[nL - 1], _col_parts(np.arange(lo, hi), cols_mid), nL - 1) if m else np.empty((len(uFI[nL - 1]), 0))
            if srank == 0:
                tw = MPI.Wtime()
                M_L = pcomm.recv(source=MPI.ANY_SOURCE, tag=3001)
                M_R = pcomm.recv(source=MPI.ANY_SOURCE, tag=3002)
                wait = MPI.Wtime() - tw
            else:
                M_L = M_R = None
            if ssize > 1:
                M_L, M_R = sub.bcast((M_L, M_R), root=0)
            Z = _pinv_solve(M_L, G).reshape(-1, len(cols_mid))                       # (r_{nL-1} m, R_R)
            slab = _pinv_solve(M_R, Z.T).T.reshape(M_L.shape[1], m, M_R.shape[1])    # Z pinv(M_R)^T
            slabs[mid] = (lo, hi, slab)
        elapsed = MPI.Wtime() - t0
        for j, v in slabs.items():
            cores[j] = v
        side_t = np.empty(3); pcomm.Allreduce(np.asarray([elapsed if side == k else 0.0 for k in range(3)]), side_t, op=MPI.MAX)
        layout = pcomm.gather([(j, world_rank, v[0], v[1]) for j, v in slabs.items()], root=0)
        if rank == 0:
            flat = sorted(x for part in layout for x in part)
            info = {"core_group_sizes": None, "core_groups": None,          # per-side layout: see "sides"
                    "dimension_times": None, "dimension_active_times": None,
                    "side_times": {"left": float(side_t[0]), "right": float(side_t[1]), "middle": float(side_t[2])},
                    "sides": {"left_ranks": PL, "right_ranks": PR, "middle_ranks": PM, "left_unfoldings": nL, "right_unfoldings": nR, "middle_core": mid},
                    "construct_time": float(side_t.max()),
                    "core_layout": [[(r_, lo_, hi_) for (j_, r_, lo_, hi_) in flat if j_ == k] for k in range(d)],
                    "execution_mode": "two_sided",
                    "oversampling": {"rows_left": [len(x) - len(y) for x, y in zip(uFI, FI)][:nL],
                                     "cols_left": [len(x) - len(y) for x, y in zip(uFJ, FJ)][:nL],
                                     "rows_right": [len(x) - len(y) for x, y in zip(uFIr, FIr)][:nR],
                                     "cols_right": [len(x) - len(y) for x, y in zip(uFJr, FJr)][:nR]}}
        sub.Free()
        pcomm.Free()
    info = _finish(comm, info, world_rank, world_size, peid_size, elapsed)
    return (cores, info) if return_info else cores


# ---------------------------------------------------------------- Algorithm 3.2, distributed
def _tt_rows_dist(slabs, idx, pcomm, bounds):
    """Rows of the TT built from the distributed cores 0..j at the multi-indices
    idx (R, j+1).  Core k is held as slabs[k] = core_k[:, lo_k:hi_k, :] on every
    rank (bounds[k] = (lo_k, hi_k)); the slices at the requested mode values
    are gathered from their owners, one allgather per dimension."""
    idx = np.asarray(idx, dtype=np.int64).reshape(len(idx), -1)
    R = len(idx)
    v = None
    for k in range(idx.shape[1]):
        lo, hi = bounds[k]
        mk = idx[:, k]
        mine = np.flatnonzero((mk >= lo) & (mk < hi))
        parts = pcomm.allgather((mine, np.ascontiguousarray(slabs[k][:, mk[mine] - lo, :].transpose(1, 0, 2))))
        a, b = slabs[k].shape[0], slabs[k].shape[2]
        S = np.empty((R, a, b))
        for ids, arr in parts:
            if len(ids):
                S[ids] = arr
        v = S[:, 0, :] if k == 0 else np.matmul(v[:, None, :], S)[:, 0, :]
    return v


@_threaded
def peid_seq_sliced(tensor_entry, FI, FJ, sample_row, sample_col, tensor_dim, comm, adap_tol=None,
                    max_peid_ranks=None, return_info=False, seed=None, nthreads=None, target_rank=None, direction="ltr"):
    """TT-PEID-Seq (Algorithm 3.2) distributed over `comm`.

    The dimension loop is sequential; every step is split over all ranks by
    the mode index of the next dimension: each rank fetches its slab of
    X_j(uFI_j, (i_{j+1}, uFJ_{j+1})), applies F^+ to it (which yields exactly
    its rows of Y), joins the distributed QR, and keeps its slab of the next
    core.  F_{<=j} is assembled from core slices gathered from their owners.
    direction="rtl" runs the same on the reversed tensor.  Cores come back
    distributed as in peid_par_sliced."""
    dims = [int(x) for x in tensor_dim]; d = len(dims); n = d - 1
    if direction == "rtl":
        FIr, FJr, sr, sc = _reverse_sets(FI, FJ, sample_row, sample_col, d)
        out = peid_seq_sliced(_Reversed(_as_data(tensor_entry, dims), d), FIr, FJr, sr, sc, dims[::-1], comm,
                              adap_tol=adap_tol, max_peid_ranks=max_peid_ranks, return_info=True, seed=seed,
                              nthreads=nthreads, target_rank=target_rank)
        cores_r, info = out
        cores = [None if c is None else (c[0], c[1], np.ascontiguousarray(c[2].transpose(2, 1, 0))) for c in cores_r[::-1]]
        info["core_layout"] = info["core_layout"][::-1]
        info["execution_mode"] = "seq_sliced_rtl"
        return (cores, info) if return_info else cores
    pcomm, active, world_rank, world_size, peid_size = _activate(comm, max_peid_ranks)
    cores = [None] * d
    info = None
    elapsed = 0.0
    if active:
        rank, size = pcomm.Get_rank(), pcomm.Get_size()
        data = _as_data(tensor_entry, dims)
        sets = _independent_sets(FI, FJ, sample_row, sample_col, dims, seed) if rank == 0 else None
        uFI, uFJ = pcomm.bcast(sets, root=0)
        ranks_tt = [_want(x, target_rank) for x in FI]
        bounds = []
        for j in range(d):
            e = np.linspace(0, dims[j], size + 1).astype(int)
            bounds.append((int(e[rank]), int(e[rank + 1])))
        slabs = [None] * d
        step_times = []
        t0 = MPI.Wtime()
        lo, hi = bounds[0]
        Y = _fetch(data, np.arange(lo, hi, dtype=np.int64)[:, None], uFJ[0], 0)              # (m_0, r+p)
        rprev = 1
        for j in range(n):
            ts = MPI.Wtime()
            U = _basis_dist(Y, pcomm, ranks_tt[j], adap_tol) if size > 1 else _basis(Y, ranks_tt[j], adap_tol)
            rk = U.shape[1]
            lo, hi = bounds[j]
            slabs[j] = U.reshape(rprev, hi - lo, rk)                 # core_j[:, lo:hi, :]; explicit: the slice may be empty
            rprev = rk
            F = _tt_rows_dist(slabs, uFI[j], pcomm, bounds)                                    # (r+p, rk), replicated
            lo1, hi1 = bounds[j + 1]
            i_next = np.arange(lo1, hi1, dtype=np.int64); m = hi1 - lo1
            if j < n - 1:
                X = _fetch(data, uFI[j], _col_parts(i_next, uFJ[j + 1]), j) if m else np.empty((len(uFI[j]), 0))
                Y = _pinv_solve(F, X).reshape(rk * m, -1) if m else np.empty((0, len(uFJ[j + 1])))
            else:
                X = _fetch(data, uFI[j], i_next[:, None], j) if m else np.empty((len(uFI[j]), 0))
                slabs[n] = _pinv_solve(F, X).reshape(rk, m, 1)
            step_times.append(MPI.Wtime() - ts)
        elapsed = MPI.Wtime() - t0
        for j in range(d):
            cores[j] = (bounds[j][0], bounds[j][1], slabs[j])
        tmax = pcomm.allreduce(elapsed, op=MPI.MAX)
        steps = np.empty(len(step_times)); pcomm.Allreduce(np.asarray(step_times, dtype=np.float64), steps, op=MPI.MAX)
        layout = pcomm.gather([(j, world_rank, bounds[j][0], bounds[j][1]) for j in range(d)], root=0)
        if rank == 0:
            flat = sorted(x for part in layout for x in part)
            info = {"core_group_sizes": [size] * d, "core_groups": [list(range(size))] * d,
                    "dimension_times": [float(x) for x in steps], "dimension_active_times": None,
                    "construct_time": tmax,
                    "core_layout": [[(r_, lo_, hi_) for (j_, r_, lo_, hi_) in flat if j_ == k] for k in range(d)],
                    "execution_mode": "seq_sliced",
                    "oversampling": {"rows": [len(x) - len(y) for x, y in zip(uFI, FI)],
                                     "cols": [len(x) - len(y) for x, y in zip(uFJ, FJ)]}}
        pcomm.Free()
    info = _finish(comm, info, world_rank, world_size, peid_size, elapsed)
    return (cores, info) if return_info else cores


# ---------------------------------------------------------------- evaluation without gathering
def tt_eval_dist(cores_local, comm, indices, root=0):
    """Values of the distributed TT (the (lo, hi, slab) lists returned by the
    sliced routines) at the multi-indices `indices` (S, d), without gathering
    the cores: for each dimension the partial row vectors travel to the ranks
    that own the requested mode values (one alltoall per dimension, S r
    numbers in total), which multiply their slab in.  `indices` is read on
    `root`; the values (S,) are returned on root, None elsewhere.  Every rank
    of comm must call it."""
    rank, size = comm.Get_rank(), comm.Get_size()
    idx = comm.bcast(np.asarray(indices, dtype=np.int64) if rank == root else None, root=root)
    S, d = idx.shape
    # owner of every mode value of every dimension: from the layout each rank holds
    mine = [(k, e[0], e[1]) for k, e in enumerate(cores_local or []) if e is not None]
    layout = comm.allgather(mine)
    owner = np.full((d, S), -1, dtype=np.int64)
    for r_, part in enumerate(layout):
        for k, lo, hi in part:
            sel = (idx[:, k] >= lo) & (idx[:, k] < hi)
            owner[k, sel] = r_
    if (owner < 0).any():
        raise ValueError("tt_eval_dist: the distributed cores do not cover every requested mode value")
    # v: current partial rows, kept by the owner of the previous dimension's mode value
    pos = np.arange(S) if rank == root else np.empty(0, dtype=np.int64)       # sample ids this rank holds
    v = np.ones((S, 1)) if rank == root else np.empty((0, 1))
    for k in range(d):
        dest = owner[k, pos]
        send = [(pos[dest == r_], v[dest == r_]) for r_ in range(size)]
        recv = comm.alltoall(send)
        got = [(p_, x) for p_, x in recv if len(p_)]
        pos = np.concatenate([p_ for p_, _ in got]) if got else np.empty(0, dtype=np.int64)
        v = np.concatenate([x for _, x in got], axis=0) if got else np.empty((0, 1))
        if len(pos):
            lo, hi, slab = cores_local[k]
            v = np.matmul(v[:, None, :], slab[:, idx[pos, k] - lo, :].transpose(1, 0, 2))[:, 0, :]
    # back to root in sample order
    parts = comm.gather((pos, v[:, 0] if v.shape[1] else np.zeros(len(pos))), root=root)
    if rank != root:
        return None
    out = np.empty(S)
    for p_, x in parts:
        out[p_] = x
    return out


def sampled_error_dist(entry_fn, dims, cores_local, comm, samples=100000, seed=10, root=0):
    """Relative error at `samples` uniform random points, cores left distributed."""
    rank = comm.Get_rank()
    if rank == root:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, np.asarray(dims), size=(samples, len(dims)))
    else:
        idx = None
    approx = tt_eval_dist(cores_local, comm, idx, root=root)
    if rank != root:
        return None
    true = np.asarray(entry_fn(idx, dims), dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(true - approx) / np.linalg.norm(true))
