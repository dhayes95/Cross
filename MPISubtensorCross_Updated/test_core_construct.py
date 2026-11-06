"""Checks for core_construct (and the pieces of tt_utils it relies on).

    python test_core_construct.py            # one rank: serial variants, distributed ones degenerate
    srun -n 4 python test_core_construct.py  # any rank count: distributed variants, fewer-ranks-than-
                                             # dimensions path, zero-row ranks in the cross (P >= 4)

Every check is collective; rank 0 prints PASS/FAIL per check and the process
exits non-zero if any failed.  The test tensor is the Hilbert-like
1 / (1 + sum_k w_k i_k) with a separable handle, 5-D with mode size 16, which
the cross resolves to ~1e-9 at TT rank 12; errors are measured on 50 000
random entries.
"""
import os
import sys
import time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from mpi4py import MPI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tt_utils as tt
import core_construct as cc

comm = MPI.COMM_WORLD
rank, P = comm.Get_rank(), comm.Get_size()
FAILED = []


def _abort_on_error(exc_type, exc, tb):          # an exception on one rank must not hang the others
    import traceback
    traceback.print_exception(exc_type, exc, tb)
    sys.stdout.flush(); sys.stderr.flush()
    comm.Abort(1)


sys.excepthook = _abort_on_error


def check(name, ok, detail=""):
    ok_all = comm.allreduce(bool(ok), op=MPI.LAND)
    if rank == 0:
        print("%s  %s  %s" % ("PASS" if ok_all else "FAIL", name, detail), flush=True)
    if not ok_all:
        FAILED.append(name)


# ---------------------------------------------------------------- test tensor
D, N, R = 5, 16, 12
DIMS = [N] * D
W = 1.0 + 0.37 * np.arange(D)


def entry(indices, dims=None):
    idx = np.asarray(indices, dtype=np.float64)
    return 1.0 / (1.0 + idx @ W[:idx.shape[1]])


def make_sep(d):
    """separable handle of the test tensor for a d-dimensional instance"""
    def rowf(rp):
        return rp.astype(np.float64) @ W[:rp.shape[1]]

    def colf(cp):
        return cp.astype(np.float64) @ W[d - cp.shape[1]:d]

    def combine(fr, fc):
        return 1.0 / (1.0 + fr[:, None] + fc[None, :])
    return (rowf, colf, combine)


SEP = make_sep(D)
RNG = np.random.default_rng(3)
SAMPLES = RNG.integers(0, N, size=(50000, D))
TRUE = entry(SAMPLES)


def err_full(cores, dims=DIMS):
    """relative error of full cores on the samples (rank 0 value, broadcast)"""
    e = None
    if rank == 0:
        pts = SAMPLES[:, :len(dims)] % dims[0]
        true = TRUE if list(dims) == DIMS else entry(pts)
        e = float(np.linalg.norm(true - np.asarray(tt.Core_to_Tensor_Value(cores, pts))) / np.linalg.norm(true))
    return comm.bcast(e, root=0)


def run_cross(dims, max_rank, tol=1e-15, block_size=64, sample_row=None, sample_col=None, **kw):
    return tt.tt_cross_os2_mpi2(entry, dims, sample_row, sample_col, min_rank=[1] * (len(dims) - 1),
                                max_rank=[max_rank] * (len(dims) - 1), tol=tol, sampling_type="Nested", tol_type="Relative",
                                adap_toler=None, err_sample_size=100, return_info=True, process_grid=(1, 1), block_size=block_size,
                                max_peid_ranks=P, seed=11, tensor_block_fn=make_sep(len(dims)), gather_cores=True, search_mode="cooperative", **kw)


# ---------------------------------------------------------------- 0. helpers
check("contiguous partition balanced", cc._contiguous_partition([1, 30, 30, 30, 30, 1], 4) == [[0, 1], [2], [3], [4, 5]]
      and all(len(p_) for p_ in cc._contiguous_partition([5, 1, 1, 1, 9], 5)))
g = tt.Grow2D(0, 0, transposed=True); g.add_cols(np.zeros(0)); g.add_cols(np.zeros((0, 1)))
h = tt.Grow2D(0, 0); h.add_rows(np.zeros(0)); h.add_rows(np.zeros((0, 1)))
check("Grow2D accepts rows/cols on an empty rank", g.a.shape == (0, 2) and h.a.shape == (1, 0))   # a 1-D vector is one row
M = RNG.standard_normal((9, 9)); B = RNG.standard_normal((9, 7))
Mt = RNG.standard_normal((14, 9)); Bt = RNG.standard_normal((14, 7))
check("pinv solve: square = solve, tall = lstsq",
      np.allclose(cc._pinv_solve(M, B), np.linalg.solve(M, B)) and np.allclose(cc._pinv_solve(Mt, Bt), np.linalg.lstsq(Mt, Bt, rcond=None)[0]))
Y = RNG.standard_normal((40000, 23)) * np.logspace(0, -3, 23)
Q, Rr = cc._tsqr_local(Y)
check("blocked TSQR", np.linalg.norm(Q.T @ Q - np.eye(23)) < 1e-12 and np.linalg.norm(Q @ Rr - Y) / np.linalg.norm(Y) < 1e-13)
U = cc._basis(Y, 10, None); Yb = RNG.standard_normal((40000, 23)) * np.logspace(0, -14, 23); Ub = cc._basis(Yb, 10, None)
check("basis: Cholesky-QR2 and Householder paths orthonormal",
      np.linalg.norm(U.T @ U - np.eye(10)) < 1e-12 and np.linalg.norm(Ub.T @ Ub - np.eye(10)) < 1e-12
      and cc._BASIS_PATHS.get("cholqr2", 0) >= 1 and cc._BASIS_PATHS.get("householder", 0) >= 1)
Ua = cc._basis(Yb, 23, 1e-6)
check("basis: adaptive rank cut", 5 <= Ua.shape[1] <= 12, "rank %d" % Ua.shape[1])

# ---------------------------------------------------------------- 1. the cross and its pivots
cores0, FI, FJ, info = run_cross(DIMS, R)
e0 = err_full(cores0)
check("cross (cooperative search) to rank %d" % R, e0 < 1e-7, "error %.1e, ranks %s" % (e0, [len(x) for x in FI]))

# ---------------------------------------------------------------- 2. serial variants (rank 0)
for name in ("par_ltr", "par_rtl", "seq_ltr", "seq_rtl"):
    ok, detail = True, ""
    if rank == 0:
        fn = getattr(cc, "peid_%s_adap" % name)
        res = {}
        for p in (0, 4):
            for label, data in (("entry", entry), ("handle", tt.FunctionalTensor(DIMS, SEP))):
                cores = fn(data, FI, FJ, [2 * p] * (D - 1), [p] * (D - 1), DIMS, seed=5, nthreads=2)
                e = float(np.linalg.norm(TRUE - np.asarray(tt.Core_to_Tensor_Value(cores, SAMPLES))) / np.linalg.norm(TRUE))
                res[(p, label)] = (e, [c.shape for c in cores])
        shapes = res[(0, "entry")][1]
        ok = (all(v[1] == shapes for v in res.values()) and all(v[0] < 1e-7 for v in res.values())
              and shapes[0] == (1, N, R) and shapes[-1][2] == 1 and all(sh[2] == R for sh in shapes[:-1]))
        detail = " ".join("p=%d/%s %.1e" % (k[0], k[1], v[0]) for k, v in sorted(res.items()))
    check("serial %s: entry fn and handle, p = 0 and 4" % name, ok, detail)
if rank == 0:
    big = cc.peid_par_ltr_adap(entry, FI, FJ, [12] * (D - 1), [6] * (D - 1), DIMS, seed=5, target_rank=R + 4)
    ebig = float(np.linalg.norm(TRUE - np.asarray(tt.Core_to_Tensor_Value(big, SAMPLES))) / np.linalg.norm(TRUE))
    ada = cc.peid_par_ltr_adap(entry, FI, FJ, [6] * (D - 1), [6] * (D - 1), DIMS, adap_tol=1e-4, seed=5)
    tr_ok = all(c.shape[2] == R + 4 for c in big[:-1]) and ebig < 1e-7
    ad_ok = all(c.shape[2] < R for c in ada[:-1])
else:
    tr_ok = ad_ok = True; ebig = 0.0
check("serial: target_rank above the pivot count", tr_ok, "ranks %d error %.1e" % (R + 4, ebig))
check("serial: adaptive truncation lowers the ranks", ad_ok)

# ---------------------------------------------------------------- 3. distributed variants
data = tt.FunctionalTensor(DIMS, SEP)
for name, fn, kw in (("par_sliced", cc.peid_par_sliced, {}), ("seq_sliced ltr", cc.peid_seq_sliced, {}),
                     ("seq_sliced rtl", cc.peid_seq_sliced, {"direction": "rtl"}), ("par2_sliced", cc.peid_par2_sliced, {})):
    cores_d, inf = fn(data, FI, FJ, [8] * (D - 1), [4] * (D - 1), DIMS, comm, max_peid_ranks=P, return_info=True, seed=5, **kw)
    layout_ok = all(sorted((lo, hi) for (_, lo, hi) in inf["core_layout"][k]) == sorted(set((lo, hi) for (_, lo, hi) in inf["core_layout"][k]))
                    and sum(hi - lo for (_, lo, hi) in inf["core_layout"][k]) == N for k in range(D))
    held_ok = all((c is None) or (c[2].shape[1] == c[1] - c[0]) for c in cores_d)
    vals = cc.tt_eval_dist(cores_d, comm, SAMPLES[:3000])
    full = tt.gather_tt_cores(cores_d, comm, root=0)
    if rank == 0:
        e = float(np.linalg.norm(TRUE - np.asarray(tt.Core_to_Tensor_Value(full, SAMPLES))) / np.linalg.norm(TRUE))
        ev = float(np.max(np.abs(vals - np.asarray(tt.Core_to_Tensor_Value(full, SAMPLES[:3000])))))
    else:
        e, ev = 0.0, 0.0
    e, ev = comm.bcast((e, ev), root=0)
    check("distributed %s (P=%d, mode %s)" % (name, P, inf["execution_mode"]), layout_ok and held_ok and e < 1e-7 and ev < 1e-12,
          "error %.1e, eval-without-gather max diff %.1e" % (e, ev))
if P >= 2:
    cores_d, inf = cc.peid_par_sliced(data, FI, FJ, [8] * (D - 1), [4] * (D - 1), DIMS, comm, max_peid_ranks=min(P, 3), return_info=True, seed=5)
    full = tt.gather_tt_cores(cores_d, comm, root=0)
    e = err_full(full)
    check("fewer ranks than dimensions (3 ranks, 5 dims)", inf["execution_mode"] == "consecutive_dims" and e < 1e-7, "error %.1e" % e)

# ---------------------------------------------------------------- 3b. more ranks than the mode size (empty slices)
if P >= 2:
    dims_s = [max(2, P // 2)] * 4
    cs, FIs, FJs, _ = run_cross(dims_s, 6)
    es0 = err_full(cs, dims_s)                       # small ranks: judge PEID against the cross, not the exact tensor
    data_s = tt.FunctionalTensor(dims_s, make_sep(4))
    oks, dets = True, ["cross %.1e" % es0]
    for name, fn, kw in (("par_sliced", cc.peid_par_sliced, {}), ("seq_sliced", cc.peid_seq_sliced, {}), ("par2_sliced", cc.peid_par2_sliced, {})):
        cd, inf = fn(data_s, FIs, FJs, [4] * 3, [2] * 3, dims_s, comm, max_peid_ranks=P, return_info=True, seed=5, **kw)
        full = tt.gather_tt_cores(cd, comm, root=0)
        pts = SAMPLES[:500, :4] % dims_s[0]
        vals = cc.tt_eval_dist(cd, comm, pts)
        if rank == 0:
            e = float(np.linalg.norm(entry(pts) - np.asarray(tt.Core_to_Tensor_Value(full, pts))) / np.linalg.norm(entry(pts)))
            ev = float(np.max(np.abs(vals - np.asarray(tt.Core_to_Tensor_Value(full, pts)))))
            oks = oks and e < max(1e-7, 3 * es0) and ev < 1e-12; dets.append("%s %.1e" % (name, e))
    check("mode size %d on %d ranks (empty slices)" % (dims_s[0], P), oks, " ".join(dets))

# ---------------------------------------------------------------- 4. pipeline: oversampling and target rank through tt_cross_os2_mpi2
cores_p, _, _, info_p = run_cross(DIMS, 8, sample_col=[4] * (D - 1), peid_target_rank=R)
ep = err_full(cores_p)
check("pipeline: cross to rank 8, PEID with 4 column / 8 row oversamples to rank %d" % R,
      ep < 1e-6 and all(c.shape[2] == R for c in (cores_p[:-1] if rank == 0 else [])), "error %.1e (cross alone %.1e)" % (ep, e0))
if P >= 4:
    cores_z, _, _, _ = run_cross([20] * 4, R, block_size=64)
    ez = err_full(cores_z, [20] * 4)
    check("cross with ranks owning zero rows (block 64 > rows per process row)", ez < 1e-6, "error %.1e" % ez)

if rank == 0:
    print("\n%d checks failed" % len(FAILED) if FAILED else "\nall checks passed", flush=True)
sys.exit(1 if FAILED else 0)
