# Threaded-across-unfoldings MPI version; retains tt_cross_os2_mpi2 API.
# The local dense superblock is never allocated. Only residual cross factors
# and transient sampled rows/columns are retained.

"""DISTRIBUTED MEMORY PARALLEL ADAPTIVE TENSOR-TRAIN CROSS APPROXIMATION
Version 4 V2: MPI unfolding groups, each with pivot-order 2D block-cyclic layout.

    srun -n P python tt_cross_v4_V2.py err_samples trials dims ranks [pr,pc] [nb]

    err_samples  number of random entries used to sample the error
    trials       number of repetitions
    dims         tensor dimensions, comma separated
    ranks        TT core ranks, comma separated (one per unfolding)
    pr,pc        minimum process grid per unfolding; extra MPI ranks are
                 distributed across unfolding groups (default: 1,1).
                 Each group gets a near-square 2D factorisation.
    nb           block size of the cyclic distribution (default 1)

Environment: TT_SEED (seed), TT_PROFILE=1 (split idle wait from communication),
TT_PROFILE_RANKS, TT_PROFILE_OUT (per-rank CSV, same columns as Version 3).
The tensor is chosen with TENSOR_KIND below.  Helper functions (tensor
definitions, entry cache, core construction, error sampling) are those of
Version 3; the pivot-search phase is new and is described below the helpers.
"""

import numpy as np
import scipy as sp
import math
import random
import os
from concurrent.futures import ThreadPoolExecutor
from mpi4py import MPI



#TENSOR_KIND = "hilbert"     # CHANGE THIS: "hilbert" | "maxwell4" | "maxwell6"
#_TENSOR_DIMS = None
#_CALLABLE_TENSOR_ENTRY = None


def tensor_entry_core_construct(index, tensor_dim):
    """Adapter for core_construct's tensor_entry(index, tensor_dim) interface."""
    return tensor_entry(index)


def tensor_entry(index):
    """Evaluate one tensor entry or a batch of entries.
    """
    index = np.asarray(index)
    one_d = (index.ndim == 1)
    if one_d:
        index = index.reshape(1, -1)
    if index.ndim != 2:
        raise ValueError("index must have shape (d,) or (m, d)")

    if _CALLABLE_TENSOR_ENTRY is not None:
        out = np.asarray(_CALLABLE_TENSOR_ENTRY(index, _TENSOR_DIMS)).reshape(-1)
    elif TENSOR_KIND == "hilbert":
        out = 1.0 / (np.sum(index, axis=1) + 1.0)

    elif TENSOR_KIND == "maxwell4":
        d = _TENSOR_DIMS
        out = Maxwellian2d2v(index[:,0], index[:,1], index[:,2], index[:,3], d)

    elif TENSOR_KIND == "maxwell6":
        d = _TENSOR_DIMS
        out = Maxwellian3d3v(index[:,0], index[:,2], index[:,4],
                             index[:,1], index[:,3], index[:,5], d)
    else:
        raise ValueError("unknown TENSOR_KIND: %r" % (TENSOR_KIND,))

    return out[0] if one_d else out


def tensor_entries(tensor_entry, indices, bounds):
    """Evaluate bounds-shifted tensor entries in one NumPy batch.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim == 1:
        indices = indices.reshape(1, -1)

    bounds_array = np.asarray(bounds, dtype=np.int64)
    shifted_indices = indices + bounds_array[:, 0]
    return np.asarray(tensor_entry(shifted_indices))


def tensor_entries_glob(indices):
    """Evaluate a batch of global tensor multi-indices together."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim == 1:
        indices = indices.reshape(1, -1)
    return np.asarray(tensor_entry(indices))


def Maxwellian2d2v(x_ind,vx_ind,y_ind,vy_ind,dim):
    ax = -1/2
    bx = 1/2

    ay = -1/2
    by = 1/2
    avx = -3
    bvx = 3
    avy = -3
    bvy = 3
    x = ax + (x_ind/(dim[0]-1))*(bx-ax)
    y = ay + (y_ind/(dim[2]-1))*(by-ay)
    vx = avx + (vx_ind/(dim[1]-1))*(bvx - avx)
    vy = avy + (vy_ind/(dim[3]-1))*(bvy - avy)
    rhox = 1+(7/8)*np.sin(2*np.pi*x)
    Tx = (1/2)+(2/5)*np.sin(2*np.pi*x)
    rhoy = 1+(7/8)*np.sin(2*np.pi*y)
    Ty = (1/2)+(2/5)*np.sin(2*np.pi*y)
    value = (((rhox)/(2*np.sqrt(2*np.pi*(Tx))))+((rhoy)/(2*np.sqrt(2*np.pi*Ty))))*(np.exp(-((np.abs(vx - 3/4)**2)/(2*Tx))-((np.abs(vy - 3/4)**2)/(2*Ty))) + np.exp(-((np.abs(vx + 3/4)**2)/(2*Tx))-((np.abs(vy + 3/4)**2)/(2*Ty))) )
    return value


def Maxwellian3d3v(x_ind,y_ind,z_ind,vx_ind,vy_ind,vz_ind,dim):
    ax = -1/2
    bx = 1/2

    ay = -1/2
    by = 1/2

    az = -1/2
    bz = 1/2
    avx = -3
    bvx = 3
    avy = -3
    bvy = 3

    avz = -3
    bvz = 3
    x = ax + (x_ind/(dim[0]-1))*(bx-ax)
    y = ay + (y_ind/(dim[2]-1))*(by-ay)
    z = az + (z_ind/(dim[4]-1))*(bz-az)
    vx = avx + (vx_ind/(dim[1]-1))*(bvx - avx)
    vy = avy + (vy_ind/(dim[3]-1))*(bvy - avy)
    vz = avz + (vz_ind/(dim[5]-1))*(bvz - avz)
    rhox = 1+(7/8)*np.sin(2*np.pi*x)
    Tx = (1/2)+(2/5)*np.sin(2*np.pi*x)
    rhoy = 1+(7/8)*np.sin(2*np.pi*y)
    Ty = (1/2)+(2/5)*np.sin(2*np.pi*y)

    rhoz = 1+(7/8)*np.sin(2*np.pi*z)
    Tz = (1/2)+(2/5)*np.sin(2*np.pi*z)
    value = (((rhox)/(2*np.sqrt(2*np.pi*(Tx))))+((rhoy)/(2*np.sqrt(2*np.pi*Ty)))+((rhoz)/(2*np.sqrt(2*np.pi*Tz))))*(np.exp(-((np.abs(vx - 3/4)**2)/(2*Tx))-((np.abs(vy - 3/4)**2)/(2*Ty))-((np.abs(vz - 3/4)**2)/(2*Tz))) + np.exp(-((np.abs(vx + 3/4)**2)/(2*Tx))-((np.abs(vy + 3/4)**2)/(2*Ty))-((np.abs(vz + 3/4)**2)/(2*Tz))) )
    return value


def Large_ravel(index,dim):
    # handling every value to a Python int on entry.

    idx = [int(x) for x in index]
    d   = [int(x) for x in dim]
    ravel_index = 0
    multiplier = 1
    for i in reversed(range(len(d))):
        ravel_index += idx[i] * multiplier
        multiplier *= d[i]
    return ravel_index


def Large_unravel(index,dim):
    # handling Python ints on entry.
    index = int(index)
    d = [int(x) for x in dim]
    unravel_index = []
    for i in range(len(d)):
        stride = math.prod(d[i+1:])
        q = index // stride
        unravel_index.append(q)
        index -= q * stride
    return tuple(unravel_index)


def _key(coord):
    # coord: a tuple/list/1-D array of ints -> one bytes key
    return np.asarray(coord, dtype=np.int64).tobytes()


def _keys_from_rows(arr2d):
    # arr2d: (N,d) int array -> list of N bytes keys (one per row)
    a = np.ascontiguousarray(np.asarray(arr2d, dtype=np.int64))
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel().tolist()


def _coord_from_key(bkey, d):
    # bytes key -> tuple coordinate (only needed on a miss, to call tensor_entry)
    return tuple(int(x) for x in np.frombuffer(bkey, dtype=np.int64))


class EntryCache:
    _H1 = np.uint64(0x9E3779B97F4A7C15)
    _H2 = np.uint64(0xC2B2AE3D27D4EB4F)
    _MAXLOAD = 0.6
    _LIMIT = 2**62

    def __init__(self, dims):
        self.dims = tuple(int(x) for x in dims)
        s = EntryCache._split(self.dims)
        if s is None:
            raise ValueError("block too large for a two-part key")
        groups = [self.dims] if s == len(self.dims) else [self.dims[:s], self.dims[s:]]
        self.split = s
        self.parts = len(groups)
        self.strides = [np.array([math.prod(g[i+1:]) for i in range(len(g))], dtype=np.int64) for g in groups]
        self.n = 0
        self._alloc(1 << 10)

    @staticmethod
    def _split(dims):
        # number of leading dims in the first key part (len(dims) = one part)
        if math.prod(dims) < EntryCache._LIMIT:
            return len(dims)
        best = None
        for s in range(1, len(dims)):
            a, b = math.prod(dims[:s]), math.prod(dims[s:])
            if a < EntryCache._LIMIT and b < EntryCache._LIMIT and (best is None or max(a, b) < best[0]):
                best = (max(a, b), s)
        return None if best is None else best[1]

    @staticmethod
    def fits(dims):
        return EntryCache._split(tuple(int(x) for x in dims)) is not None

    def _alloc(self, cap):
        self.cap = cap
        self.mask = cap - 1
        self.shift = np.uint64(64 - (cap.bit_length() - 1))
        self.keys = np.full((cap, self.parts), -1, dtype=np.int64)
        self.vals = np.empty(cap, dtype=np.float64)
        self.claim = np.empty(cap, dtype=np.int64)

    def linear(self, coords):
        coords = np.asarray(coords, dtype=np.int64).reshape(-1, len(self.dims))
        if self.parts == 1:
            return (coords @ self.strides[0])[:, None]
        s = self.split
        return np.column_stack((coords[:, :s] @ self.strides[0], coords[:, s:] @ self.strides[1]))

    def _slot0(self, K):
        h = K[:, 0].astype(np.uint64) * self._H1
        if self.parts == 2:
            h = (h ^ (K[:, 1].astype(np.uint64) * self._H2)) * self._H1
        return (h >> self.shift).astype(np.int64)

    def _same(self, p, K):
        if self.parts == 1:
            return self.keys[p, 0] == K[:, 0]
        return (self.keys[p, 0] == K[:, 0]) & (self.keys[p, 1] == K[:, 1])

    def _unique(self, K):
        # first-occurrence index and inverse of the distinct rows of K
        if self.parts == 1:
            _, first, inv = np.unique(K[:, 0], return_index=True, return_inverse=True)
        else:
            rows = np.ascontiguousarray(K).view(np.dtype((np.void, 16))).ravel()
            _, first, inv = np.unique(rows, return_index=True, return_inverse=True)
        return first, inv.reshape(-1)

    def _find(self, K):
        # slot holding each key, or -1
        pos = self._slot0(K)
        out = np.full(K.shape[0], -1, dtype=np.int64)
        todo = np.arange(K.shape[0])
        while todo.size:
            p = pos[todo]
            hit = self._same(p, K[todo])
            out[todo[hit]] = p[hit]
            cont = ~hit & (self.keys[p, 0] != -1)
            todo = todo[cont]
            pos[todo] = (p[cont] + 1) & self.mask
        return out

    def _insert(self, K, v):
        # K: keys known to be absent and distinct.  Returns the slot of each key.
        need = self.n + K.shape[0]
        if need > self._MAXLOAD*self.cap:
            cap = self.cap
            while need > 0.5*cap:
                cap *= 2
            used = self.keys[:, 0] != -1
            oK, ov = self.keys[used], self.vals[used]
            self._alloc(cap)
            self.n = 0
            self._insert(oK, ov)
        pos = self._slot0(K)
        todo = np.arange(K.shape[0])
        while todo.size:
            p = pos[todo]
            free = self.keys[p, 0] == -1
            cand, cp = todo[free], p[free]
            # several keys may race for one free slot: each writes its id to
            # the slot's claim, the id that landed takes the slot and the
            # others keep probing
            self.claim[cp] = cand
            won = self.claim[cp] == cand
            self.keys[cp[won]] = K[cand[won]]
            self.vals[cp[won]] = v[cand[won]]
            done = np.zeros(todo.size, dtype=bool)
            done[np.flatnonzero(free)[won]] = True
            todo = todo[~done]
            pos[todo] = (pos[todo] + 1) & self.mask
        self.n = need
        return pos

    def fetch(self, coords, evaluate):
        """Values at block-local coords (m,d); misses go to evaluate(miss_coords)."""
        coords = np.asarray(coords, dtype=np.int64).reshape(-1, len(self.dims))
        K = self.linear(coords)
        slot = self._find(K)
        miss = np.flatnonzero(slot < 0)
        if miss.size:
            # distinct missing keys, evaluated in first-occurrence order
            first, inv = self._unique(K[miss])
            order = np.argsort(first, kind='stable')
            new = miss[first[order]]
            vals = np.asarray(evaluate(coords[new]), dtype=np.float64).reshape(-1)
            if vals.size != new.size:
                raise ValueError("tensor_entries must return one value for each input multi-index")
            cap = self.cap
            uslot = np.empty(first.size, dtype=np.int64)
            uslot[order] = self._insert(K[new], vals)
            if self.cap != cap:
                hit = slot >= 0                 # the table was rehashed
                slot[hit] = self._find(K[hit])
            slot[miss] = uslot[inv]
        return self.vals[slot]

    def put_missing(self, coords, vals):
        """Insert (coords, vals) pairs whose key is not cached yet; first one wins."""
        vals = np.asarray(vals, dtype=np.float64).reshape(-1)
        K = self.linear(coords)
        miss = np.flatnonzero(self._find(K) < 0)
        if miss.size:
            first, _ = self._unique(K[miss])
            new = miss[np.sort(first)]
            self._insert(K[new], vals[new])

    # dict API for the scalar call sites, bytes keys as produced by _key()
    def _k1(self, bkey):
        return self.linear(np.frombuffer(bkey, dtype=np.int64))

    def __contains__(self, bkey):
        return self._find(self._k1(bkey))[0] >= 0

    def __getitem__(self, bkey):
        s = self._find(self._k1(bkey))[0]
        if s < 0:
            raise KeyError(bkey)
        return self.vals[s]

    def __setitem__(self, bkey, value):
        K = self._k1(bkey)
        s = self._find(K)[0]
        if s >= 0:
            self.vals[s] = value
        else:
            self._insert(K, np.array([value], dtype=np.float64))

    def __len__(self):
        return self.n


def _new_entry_cache(dims):
    # blocks too large even for a two-part key keep the bytes-keyed dict
    return EntryCache(dims) if EntryCache.fits(dims) else {}


def _batch_unravel(flat_indices, dims):
    """Convert many C-order flattened indices to multi-indices in one batch."""
    flat_indices = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
    dims = tuple(int(v) for v in dims)

    if len(dims) == 0:
        return np.empty((flat_indices.size, 0), dtype=np.int64)

    coords = np.unravel_index(flat_indices, dims, order='C')
    return np.column_stack(coords).astype(np.int64, copy=False)


def _unfold_pairs_to_tensor_indices(row_indices, col_indices, k, Xdim):
    """Batch-convert unfolding row/column positions to full tensor multi-indices."""
    rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    cols = np.asarray(col_indices, dtype=np.int64).reshape(-1)

    if rows.size != cols.size:
        raise ValueError("row_indices and col_indices must have the same length")

    row_multi = _batch_unravel(rows, Xdim[:k + 1])
    col_multi = _batch_unravel(cols, Xdim[k + 1:])
    return np.column_stack((row_multi, col_multi))


def _cached_core_entries_batch(tensor_entries, tensor_indices, X_dict):
    """Pull core tensor entries with batched evaluation of all cache misses.
    """
    tensor_indices = np.asarray(tensor_indices, dtype=np.int64)
    if tensor_indices.ndim == 1:
        tensor_indices = tensor_indices.reshape(1, -1)

    if isinstance(X_dict, EntryCache):
        # The cache only covers its own block; entries outside it are evaluated.
        inside = np.all(tensor_indices < np.asarray(X_dict.dims), axis=1)
        out = np.empty(tensor_indices.shape[0], dtype=np.float64)
        out[inside] = X_dict.fetch(tensor_indices[inside], tensor_entries)
        if not inside.all():
            out[~inside] = np.asarray(tensor_entries(tensor_indices[~inside])).reshape(-1)
        return out

    keys = _keys_from_rows(tensor_indices)
    key_to_row = {}
    for row_i, kk in enumerate(keys):
        if kk not in X_dict and kk not in key_to_row:
            key_to_row[kk] = row_i
    if key_to_row:
        missing_keys = list(key_to_row.keys())
        missing_indices = tensor_indices[list(key_to_row.values())]
        missing_values = np.asarray(tensor_entries(missing_indices)).reshape(-1)

        if missing_values.size != len(missing_keys):
            raise ValueError(
                "tensor_entries must return one value for each input multi-index"
            )

        X_dict.update(zip(missing_keys, missing_values))

    return np.fromiter(
        (X_dict[key] for key in keys),
        dtype=np.float64,
        count=len(keys),
    )


def Low_Access_Tk(tensor_entries, I, J, k, Xdim, X_dict, I_prev):
    """Construct Tk with all batchable core work vectorized.
    """
    I = np.asarray(I, dtype=np.int64)
    J = np.asarray(J, dtype=np.int64)

    if I.size == 0 or J.size == 0:
        raise ValueError("I and J must contain at least one pivot")

    if k == 0:
        indices = np.arange(Xdim[0], dtype=np.int64)
        setlen = indices.size
    else:
        I_prev = np.asarray(I_prev, dtype=np.int64)

        base_indices = (
            np.repeat(I_prev, Xdim[k]) * Xdim[k]
            + np.tile(np.arange(Xdim[k], dtype=np.int64), I_prev.size)
        )

        setlen = base_indices.size
        indices = np.concatenate((base_indices, I))

    pivot_matches = I[:, None] == indices[None, :]
    if not np.all(np.any(pivot_matches, axis=1)):
        raise ValueError("Every pivot row in I must be present in the Tk row set")
    pivot_positions = np.argmax(pivot_matches, axis=1)

    # Pull the whole first selected unfolding column as one batch.
    first_cols = np.full(indices.size, J[0], dtype=np.int64)
    first_tensor_indices = _unfold_pairs_to_tensor_indices(
        indices, first_cols, k, Xdim
    )
    Tk = _cached_core_entries_batch(
        tensor_entries, first_tensor_indices, X_dict
    ).reshape(-1, 1)

    delta = 1.0 / Tk[pivot_positions[0], 0]
    Tk *= delta

    for j in range(1, I.size):
        jj = J[j]

        cols = np.full(indices.size, jj, dtype=np.int64)
        tensor_indices = _unfold_pairs_to_tensor_indices(
            indices, cols, k, Xdim
        )
        temp = _cached_core_entries_batch(
            tensor_entries, tensor_indices, X_dict
        ).reshape(-1, 1)

        itemp_positions = pivot_positions[:j]
        ii_position = pivot_positions[j]

        Sk = Tk @ temp[itemp_positions, :] - temp

        delta1 = Sk[ii_position, 0]
        delta = 0.0 if delta1 == 0 else 1.0 / delta1

        a = Tk - delta * Sk * Tk[ii_position, :]
        b = delta * Sk
        Tk = np.concatenate((a, b), axis=1)

    return Tk[:setlen, :]


def Low_Access_Core_Extract_Update(tensor_entries, Xdim, X_dict, Ir, Ic):
    """Construct all TT cores with vectorized core-entry generation/pulling."""
    cores = []

    for i in range(len(Xdim)):
        if i == 0:
            Tk = Low_Access_Tk(
                tensor_entries, Ir[0], Ic[0], 0, Xdim, X_dict, [0]
            )
            cores.append(np.reshape(Tk, [1, Xdim[i], len(Ic[i])]))

        elif i == len(Xdim) - 1:
            # Vectorized replacement for the original j-by-k nested loops.
            row_prefixes = _batch_unravel(Ir[-1], Xdim[:-1])
            repeated_prefixes = np.repeat(
                row_prefixes, Xdim[-1], axis=0
            )
            last_mode_indices = np.tile(
                np.arange(Xdim[-1], dtype=np.int64), len(Ir[-1])
            )
            full_indices = np.column_stack(
                (repeated_prefixes, last_mode_indices)
            )

            values = _cached_core_entries_batch(
                tensor_entries, full_indices, X_dict
            )
            temp = values.reshape(len(Ir[-1]), Xdim[-1])
            cores.append(np.reshape(temp, [len(Ir[-1]), Xdim[-1], 1]))

        else:
            Tk = Low_Access_Tk(
                tensor_entries, Ir[i], Ic[i], i, Xdim, X_dict, Ir[i - 1]
            )
            cores.append(
                np.reshape(Tk, [len(Ir[i - 1]), Xdim[i], len(Ic[i])])
            )

    return cores, X_dict


def Core_to_Tensor_Value(cores,index):
    index_array = np.asarray(index, dtype=int)

    if index_array.ndim == 1:
        start = cores[0][:,index_array[0],:]
        for i in range(1,len(cores)):
            start = start@cores[i][:,index_array[i],:]
        return start[0][0]

    if index_array.ndim != 2:
        raise ValueError("index must be a 1D index or a 2D array of indices")

    if index_array.shape[1] != len(cores):
        raise ValueError("each index must have one entry per TT core")

    start = np.transpose(cores[0][:,index_array[:,0],:], (1,0,2))
    for i in range(1,len(cores)):
        next_core = np.transpose(cores[i][:,index_array[:,i],:], (1,0,2))
        start = np.einsum('bij,bjk->bik', start, next_core)

    return start[:,0,0]


# ==========================================================================
# Version 4: pivot-order 2D block-cyclic layout
# ==========================================================================
# The object worked on for unfolding k is the superblock
#
#     A_k = X( I_{k-1} x n_k ,  n_{k+1} x J_{k+1} ),
#
# a matrix whose rows are (prefix, i_k) pairs and whose columns are
# (i_{k+1}, suffix) pairs.  Row r = pidx*n_k + i_k, where pidx is the ORDINAL
# of the prefix in the pivot list I_{k-1}; column c = sidx*n_{k+1} + i_{k+1}.
# Rows and columns are dealt out block-cyclically over a p_r x p_c process grid
# by that index, so every process row holds the same number of rows of A_k no
# matter where in the tensor the pivots fall.  Each process keeps the dense
# tile A_k[rows, cols] it owns, plus the factors of the cross approximation
# restricted to its rows (U) and columns (V):  E_l = A - sum_m delta_m u_m v_m^T.
#
# Per pivot step: a tile-local rook search, one allgather of the candidates,
# then the pivot column segment is broadcast along the row group and the pivot
# row segment down the column group (the paper's communication pattern).  The
# new pivot's prefix extends the rows of A_{k+1}, its suffix the columns of
# A_{k-1}; extending the factors to the new rows needs the pivot columns at
# those rows, gathered inside the row group.
#
# The data layer is separate from the layout: FunctionalTensor evaluates on
# request; a stored, pre-partitioned tensor plugs in by implementing fetch()
# as a collective gather from the owners (not implemented in this prototype).
# ==========================================================================


class FunctionalTensor:
    """Entries are evaluated on request; any process can evaluate any entry.

    block_fn, if given, is the block form of the entry function:
    block_fn(rp, cp, k) -> (len(rp), len(cp)) values at every combination of
    the row parts rp (dims 0..k) and column parts cp (dims k+1..d-1).  A
    separable function can evaluate it in O(len(rp) + len(cp)) without ever
    forming the full multi-indices; without it the rectangle is expanded and
    evaluated entry by entry as before."""
    def __init__(self, dims, block_fn=None):
        self.dims = tuple(int(x) for x in dims)
        self.block_fn = block_fn
    def fetch(self, indices):
        """indices: (m, d) ints -> (m,) float64"""
        return np.asarray(tensor_entry(np.asarray(indices)), dtype=np.float64).reshape(-1)
    def fetch_outer(self, rp, cp, k, chunk=1 << 20):
        """Values of the rectangle rp x cp of unfolding k, shape (len(rp), len(cp))."""
        nr, nc, d = len(rp), len(cp), len(self.dims)
        if self.block_fn is not None:
            return np.asarray(self.block_fn(rp, cp, k), dtype=np.float64).reshape(nr, nc)
        out = np.empty((nr, nc), dtype=np.float64)
        step = max(1, chunk // max(nc, 1))
        for r0 in range(0, nr, step):
            r1 = min(nr, r0 + step)
            idx = np.empty((r1 - r0, nc, d), dtype=rp.dtype)
            idx[:, :, :k + 1] = rp[r0:r1, None, :]
            idx[:, :, k + 1:] = cp[None, :, :]
            out[r0:r1] = self.fetch(idx.reshape(-1, d)).reshape(r1 - r0, nc)
        return out


class StoredTensor:
    """Interface for a tensor that is already partitioned and stored.

    owner(indices) -> rank holding each entry;  fetch(indices) must return the
    values and is called collectively (same call sequence on every process),
    so it can be implemented as one alltoallv per round to the owners.
    """
    def __init__(self, dims):
        raise NotImplementedError("stored-tensor backend is not part of this prototype")


# ---- per-rank compute / communication / wait accounting (as in Version 3) ----
_PROFILE_SYNC = False
_PROF = {'piv': {'comm': 0.0, 'wait': 0.0, 'total': 0.0},
         'core': {'comm': 0.0, 'wait': 0.0, 'total': 0.0}}


def _prof_reset():
    for ph in _PROF:
        for key in _PROF[ph]:
            _PROF[ph][key] = 0.0


def _prof_comm(phase, comm, fn):
    if _PROFILE_SYNC:
        t = MPI.Wtime(); comm.Barrier(); _PROF[phase]['wait'] += MPI.Wtime() - t
    t = MPI.Wtime(); out = fn(); _PROF[phase]['comm'] += MPI.Wtime() - t
    return out


class Grow2D:
    """2D array that grows in both directions with amortised O(1) copying.
    .a is the live (n x m) view; the tile, U and V are appended to on every
    extension, and vstack/hstack would copy the whole tile each time."""
    def __init__(self, n=0, m=0):
        self.buf = np.zeros((n, m)); self.n = n; self.m = m
    @property
    def a(self):
        return self.buf[:self.n, :self.m]
    def set(self, X):
        self.buf = np.array(X, dtype=np.float64, copy=True); self.n, self.m = X.shape
    def _ensure(self, n, m):
        N, M = self.buf.shape
        if n > N or m > M:
            nb = np.zeros((max(n, 2 * N) if n > N else N, max(m, 2 * M) if m > M else M))
            nb[:self.n, :self.m] = self.a
            self.buf = nb
    def add_rows(self, X):
        X = np.asarray(X, dtype=np.float64).reshape(-1, self.m)
        self._ensure(self.n + X.shape[0], self.m)
        self.buf[self.n:self.n + X.shape[0], :self.m] = X; self.n += X.shape[0]
    def add_cols(self, X):
        X = np.asarray(X, dtype=np.float64).reshape(self.n, -1)
        self._ensure(self.n, self.m + X.shape[1])
        self.buf[:self.n, self.m:self.m + X.shape[1]] = X; self.m += X.shape[1]


class Unfolding:
    """Distributed state of the superblock A_k on this process."""

    def __init__(self, k, dims, pr, pc, my_pr, my_pc, nb):
        self.k = k
        self.d = len(dims)
        self.nk = int(dims[k])
        self.nk1 = int(dims[k + 1])
        self.pr, self.pc, self.my_pr, self.my_pc, self.nb = pr, pc, my_pr, my_pc, nb
        self.prefixes = []        # I_{k-1} in pivot order (tuples over dims 0..k-1)
        self.suffixes = []        # J_{k+1} in pivot order (tuples over dims k+2..d-1)
        self.rows = np.zeros(0, dtype=np.int64)   # global row indices held, ascending
        self.cols = np.zeros(0, dtype=np.int64)
        self.A = None                              # matrix-free: never store the dense tile
        self.U = Grow2D()                          # u_m restricted to rows
        self.V = Grow2D()                          # v_m restricted to cols
        self.npre_done = 0                         # prefixes whose rows are materialised
        self.nsuf_done = 0                         # suffixes whose columns are materialised
        self.dl = np.zeros(0)                      # deltas
        self.piv_rows = []                         # global row index of each pivot
        self.piv_cols = []
        self.Pm = np.zeros((0, 0))                 # Pm[m,l] = v_m(j_l), m\<l
        self.Qm = np.zeros((0, 0))                 # Qm[m,l] = u_m(i_l), m\<l
        self.fetched = 0                           # entries evaluated/fetched
        # tensor-coordinate parts of the local rows (prefix, i_k) and columns
        # (i_k+1, suffix), kept so a scan builds its index array by one copy
        self.rowc = np.zeros((0, k + 1), dtype=np.int64)
        self.colc = np.zeros((0, self.d - k - 1), dtype=np.int64)
        # rows / columns not yet used as pivots, kept up to date instead of
        # recomputed with np.isin at every search
        self.avail_r = np.zeros(0, dtype=bool)
        self.avail_c = np.zeros(0, dtype=bool)

    # ---- layout ----
    def nrows_glob(self): return len(self.prefixes) * self.nk
    def ncols_glob(self): return len(self.suffixes) * self.nk1
    def owner_row(self, r): return (r // self.nb) % self.pr
    def owner_col(self, c): return (c // self.nb) % self.pc
    def my_new_rows(self, pidx):
        r = np.arange(pidx * self.nk, (pidx + 1) * self.nk, dtype=np.int64)
        return r[self.owner_row(r) == self.my_pr]
    def my_new_cols(self, sidx):
        c = np.arange(sidx * self.nk1, (sidx + 1) * self.nk1, dtype=np.int64)
        return c[self.owner_col(c) == self.my_pc]
    def lr(self, r): return np.searchsorted(self.rows, r)
    def lc(self, c): return np.searchsorted(self.cols, c)

    # ---- data ----
    def coords(self, rows, cols):
        """Tensor multi-indices of A_k[rows, cols], row-major, shape (len(rows)*len(cols), d)."""
        k, d = self.k, self.d
        pidx, ik = np.divmod(np.asarray(rows, dtype=np.int64), self.nk)
        sidx, ik1 = np.divmod(np.asarray(cols, dtype=np.int64), self.nk1)
        out = np.empty((len(rows), len(cols), d), dtype=np.int64)
        if k > 0:
            Pre = np.asarray(self.prefixes, dtype=np.int64).reshape(len(self.prefixes), k)
            out[:, :, :k] = Pre[pidx][:, None, :]
        out[:, :, k] = ik[:, None]
        out[:, :, k + 1] = ik1[None, :]
        if k + 2 < d:
            Suf = np.asarray(self.suffixes, dtype=np.int64).reshape(len(self.suffixes), d - k - 2)
            out[:, :, k + 2:] = Suf[sidx][None, :, :]
        return out.reshape(-1, d)

    def row_part(self, rows):
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        pidx, ik = np.divmod(rows, self.nk)
        out = np.empty((len(rows), self.k + 1), dtype=np.int64)
        if self.k > 0:
            out[:, :self.k] = np.asarray(self.prefixes, dtype=np.int64).reshape(-1, self.k)[pidx]
        out[:, self.k] = ik
        return out

    def col_part(self, cols):
        cols = np.asarray(cols, dtype=np.int64).reshape(-1)
        sidx, ik1 = np.divmod(cols, self.nk1)
        out = np.empty((len(cols), self.d - self.k - 1), dtype=np.int64)
        out[:, 0] = ik1
        if self.k + 2 < self.d:
            out[:, 1:] = np.asarray(self.suffixes, dtype=np.int64).reshape(-1, self.d - self.k - 2)[sidx]
        return out

    def fetch_parts(self, rp, cp, chunk=1 << 20):
        """Entries at every (row, column) pair of the coordinate parts rp (nr, k+1)
        and cp (nc, d-k-1), through the data layer (block form if it has one)."""
        nr, nc = len(rp), len(cp)
        if nr == 0 or nc == 0:
            return np.empty((nr, nc), dtype=np.float64)
        out = self.data.fetch_outer(rp, cp, self.k, chunk)
        self.fetched += out.size
        return out

    def fetch_block(self, data, rows, cols, chunk=65536):
        """Evaluate a requested rectangle, never the whole local superblock.

        The returned rectangle is transient. `chunk` bounds the number of
        tensor indices in each evaluation batch, not the output size.
        """
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        cols = np.asarray(cols, dtype=np.int64).reshape(-1)
        out = np.empty((len(rows), len(cols)), dtype=np.float64)
        if not len(rows) or not len(cols):
            return out
        # Chunk in BOTH dimensions so even a single long row is bounded.
        colstep = min(len(cols), chunk)
        for c0 in range(0, len(cols), colstep):
            cc = cols[c0:c0 + colstep]
            rowstep = max(1, chunk // len(cc))
            for r0 in range(0, len(rows), rowstep):
                rr = rows[r0:r0 + rowstep]
                out[r0:r0+len(rr), c0:c0+len(cc)] = (
                    data.fetch(self.coords(rr, cc)).reshape(len(rr), len(cc))
                )
        self.fetched += out.size
        return out

    def fetch_pairs(self, lrs, lcs):
        """Evaluate corresponding local (row,column) pairs in batches."""
        lrs = np.asarray(lrs, dtype=np.int64).reshape(-1)
        lcs = np.asarray(lcs, dtype=np.int64).reshape(-1)
        if len(lrs) != len(lcs):
            raise ValueError("pair index lengths differ")
        out = np.empty(len(lrs), dtype=np.float64)
        for start in range(0, len(lrs), 65536):
            end = min(start + 65536, len(lrs))
            rr = self.rows[lrs[start:end]]
            cc = self.cols[lcs[start:end]]
            pidx, ik = np.divmod(rr, self.nk)
            sidx, ik1 = np.divmod(cc, self.nk1)
            coords = np.empty((len(rr), self.d), dtype=np.int64)
            if self.k:
                pre = np.asarray(self.prefixes, dtype=np.int64).reshape(-1, self.k)
                coords[:, :self.k] = pre[pidx]
            coords[:, self.k] = ik
            coords[:, self.k + 1] = ik1
            if self.k + 2 < self.d:
                suf = np.asarray(self.suffixes, dtype=np.int64).reshape(-1, self.d-self.k-2)
                coords[:, self.k+2:] = suf[sidx]
            out[start:end] = self.data.fetch(coords)
        self.fetched += len(out)
        return out

    # Matrix-free residual E = A - U diag(dl) V.
    def ecol(self, lc):
        values = self.fetch_parts(self.rowc, self.colc[lc:lc+1]).ravel()
        return values - self.U.a @ (self.dl * self.V.a[:, lc])

    def erow(self, lr):
        values = self.fetch_parts(self.rowc[lr:lr+1], self.colc).ravel()
        return values - (self.dl * self.U.a[lr, :]) @ self.V.a

    def epairs(self, lrs, lcs):
        values = self.fetch_pairs(lrs, lcs)
        return values - np.einsum('im,m,mi->i',
                                  self.U.a[lrs], self.dl, self.V.a[:, lcs])


def local_search(unf, rng, sample_size=10, maxiter=10):
    """Greedy rook search on this process's tile, excluding pivot rows/columns.
    Same scheme as Version 3's Low_Access_MPIGreedy_Search_PivV2, run on the
    cyclic tile instead of a subtensor.  Returns (|E|, E, global row, global col)."""
    avail_r = np.flatnonzero(unf.avail_r)
    avail_c = np.flatnonzero(unf.avail_c)
    if avail_r.size == 0 or avail_c.size == 0:
        return (0.0, 0.0, -1, -1)
    s = min(sample_size, avail_r.size, avail_c.size)
    a = avail_r[rng.sample(range(avail_r.size), s)]
    b = avail_c[rng.sample(range(avail_c.size), s)]
    E = np.abs(unf.epairs(a, b))
    j = b[int(np.argmax(E))]
    err1 = unf.ecol(j)[avail_r]
    
    si = int(np.argmax(np.abs(err1)))
    for _ in range(maxiter):
        si = int(np.argmax(np.abs(err1)))
        i = avail_r[si]

        err2 = unf.erow(i)[avail_c]
        #print(f"MPI rank {MPI.COMM_WORLD.Get_rank()}, unfolding {unf.k}, erow shape: {row.shape}", flush=True)
       
        sj = int(np.argmax(np.abs(err2)))
        j = avail_c[sj]
        #print(f"MPI rank {MPI.COMM_WORLD.Get_rank()}, unfolding {unf.k}, ecol shape: {col.shape}", flush=True)
        
        err1 = unf.ecol(j)[avail_r]

        if (
            np.abs(err1[si]) >= np.abs(err1).max()
            and np.abs(err2[sj]) >= np.abs(err2).max()
        ):
            break

    # Ensure the returned row maximizes the final residual column
    si = int(np.argmax(np.abs(err1)))
    i = avail_r[si]
    val = float(err1[si])

    return (
        abs(val),
        val,
        int(unf.rows[i]),
        int(unf.cols[j])
    )


def kickstart(data, dims, rng, sample_size=10, iters=10):
    """Starting pivot: best of random samples, then alternating argmax along fibers."""
    idx = np.array([[rng.randint(0, n - 1) for n in dims] for _ in range(sample_size)], dtype=np.int64)
    vals = data.fetch(idx)
    x = idx[int(np.argmax(np.abs(vals)))]
    best = float(vals[int(np.argmax(np.abs(vals)))])
    for _ in range(iters):
        for i in range(len(dims)):
            fiber = np.tile(x, (dims[i], 1))
            fiber[:, i] = np.arange(dims[i])
            v = data.fetch(fiber)
            t = int(np.argmax(np.abs(v)))
            x = fiber[t]; best = float(v[t])
    return tuple(int(t) for t in x), best


def make_group_layout(P, ngroups, base_pr, base_pc):
    """Allocate at least base_pr*base_pc ranks to each unfolding.

    Remaining ranks are distributed round-robin. Each group's actual 2D
    grid is chosen as a factorisation close to the requested aspect ratio.
    """
    minimum = base_pr * base_pc
    if base_pr < 1 or base_pc < 1 or P < ngroups * minimum:
        raise ValueError("need at least {} ranks ({} per unfolding)".format(
            ngroups * minimum, minimum))
    counts = [minimum] * ngroups
    for extra in range(P - ngroups * minimum):
        counts[extra % ngroups] += 1

    grids = []
    target = base_pr / base_pc
    for count in counts:
        if count == minimum:
            grids.append((base_pr, base_pc))
            continue
        choices = [(a, count // a) for a in range(1, count + 1) if count % a == 0]
        grids.append(min(choices, key=lambda ab: (
            abs(math.log(ab[0] / ab[1])),
            abs(math.log((ab[0] / ab[1]) / target)))))
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)
    return counts, grids, starts


class PivotOrderCross:
    def __init__(self, dims, pr, pc, nb, data, world, seed, tol=0.0, nthreads=1, tol_type="Relative"):
        self.dims = [int(x) for x in dims]
        self.tol = tol
        self.tol_type = tol_type
        # Threads for the per-unfolding local work of a round (fetch, factor
        # recurrences, rook search).  MPI is only ever called from the main
        # thread.  Results are identical for any thread count.
        self.pool = ThreadPoolExecutor(max_workers=nthreads) if nthreads > 1 else None
        self.d = len(dims)
        self.pr, self.pc, self.nb = pr, pc, nb
        self.world = world
        self.rank = world.Get_rank()
        self.my_pr, self.my_pc = divmod(self.rank, pc)
        # rank inside row_comm == my_pc, inside col_comm == my_pr
        self.row_comm = world.Split(self.my_pr, self.my_pc)
        self.col_comm = world.Split(self.my_pc, self.my_pr)
        self.data = data
        self.seed = seed
        self.unfs = [Unfolding(k, self.dims, pr, pc, self.my_pr, self.my_pc, nb) for k in range(self.d - 1)]
        self.Ipiv = [[] for _ in range(self.d - 1)]   # pivot multi-indices over dims 0..k
        self.Jpiv = [[] for _ in range(self.d - 1)]   # over dims k+1..d-1
        self.rngs = [random.Random(None if seed is None else seed + 1000003 * k + self.rank)
                     for k in range(self.d - 1)]

    # ---- pivot bookkeeping ----
    def add_prefix(self, unf, prefix):
        """A new pivot of unfolding k-1 adds n_k rows to A_k (materialised lazily)."""
        unf.prefixes.append(tuple(prefix))

    def add_suffix(self, unf, suffix):
        """A new pivot of unfolding k+1 adds n_{k+1} columns to A_k (materialised lazily)."""
        unf.suffixes.append(tuple(suffix))

    # ---- pivot bookkeeping ----
    def _apply_pivot(self, unf, i, j, e, u, v, q, p):
        """Record pivot (global row i, global col j, residual e) of unfolding unf
        with its column segment u = E(rows, j), row segment v = E(i, cols), and
        the entries of the earlier factors at the pivot, q = U[i,:], p = V[:,j]."""
        l = len(unf.dl)
        Pm = np.zeros((l + 1, l + 1)); Pm[:l, :l] = unf.Pm; Pm[:l, l] = p
        Qm = np.zeros((l + 1, l + 1)); Qm[:l, :l] = unf.Qm; Qm[:l, l] = q
        unf.Pm, unf.Qm = Pm, Qm
        unf.U.add_cols(u[:, None])
        unf.V.add_rows(v[None, :])
        unf.dl = np.append(unf.dl, 1.0 / e)
        unf.piv_rows.append(int(i)); unf.piv_cols.append(int(j))
        if unf.owner_row(i) == self.my_pr:
            unf.avail_r[unf.lr(i)] = False
        if unf.owner_col(j) == self.my_pc:
            unf.avail_c[unf.lc(j)] = False

    def _map(self, fn, items):
        """fn over items, in order; threaded when a pool exists and there is more than one item."""
        items = list(items)
        if self.pool is not None and len(items) > 1:
            return list(self.pool.map(fn, items))
        return [fn(x) for x in items]

    def exchange_segments(self, want):
        """Pivot segments for every unfolding k in want = {k: (i, j)}, with one
        allgather per group for all unfoldings together.

        The column segment E(rows, j) is computed by the process column holding
        j and travels along the ROW group, carrying V[:, j] with it; the row
        segment E(i, cols) is computed by the process row holding i and travels
        down the COLUMN group, carrying U[i, :].  Every row has one member of
        the owning column and every column one member of the owning row, so
        all four pieces reach every process."""
        us, vs = {}, {}
        for k, (i, j) in want.items():
            unf = self.unfs[k]
            if unf.owner_col(j) == self.my_pc:
                lc = unf.lc(j); us[k] = (unf.ecol(lc), unf.V.a[:, lc].copy())
            if unf.owner_row(i) == self.my_pr:
                lr = unf.lr(i); vs[k] = (unf.erow(lr), unf.U.a[lr, :].copy())
        allu = _prof_comm('piv', self.row_comm, lambda: self.row_comm.allgather(us))
        allv = _prof_comm('piv', self.col_comm, lambda: self.col_comm.allgather(vs))
        U, V = {}, {}
        for dct in allu:
            U.update(dct)
        for dct in allv:
            V.update(dct)
        return U, V

    def sync(self, unf):
        """Extend distributed cross factors without materializing A.

        When columns grow, only existing pivot rows at those columns are
        needed to extend V. When rows grow, only existing pivot columns
        at those rows are needed to extend U. These are the same cross
        samples used by the earlier Ej/D/Ei implementation.
        """
        if unf.nsuf_done < len(unf.suffixes):
            cols_new = np.concatenate([
                unf.my_new_cols(sidx)
                for sidx in range(unf.nsuf_done, len(unf.suffixes))
            ])
            unf.nsuf_done = len(unf.suffixes)
            if cols_new.size:
                l = len(unf.dl)
                own = [(m, im) for m, im in enumerate(unf.piv_rows) if unf.owner_row(im) == self.my_pr]
                mine = []
                if own:   # all my pivot rows at the new columns, one batch
                    cp_new = unf.col_part(cols_new)
                    vals = unf.fetch_parts(unf.row_part([im for _, im in own]), cp_new)
                    mine = [(m, vals[t]) for t, (m, _) in enumerate(own)]
                allm = _prof_comm('piv', self.col_comm,
                                  lambda: self.col_comm.allgather(mine))
                Ap = np.empty((l, cols_new.size))
                for chunk in allm:
                    for m, rowv in chunk:
                        Ap[m, :] = rowv
                Vnew = np.empty((l, cols_new.size))
                for m in range(l):
                    Vnew[m, :] = Ap[m, :] - (
                        (unf.dl[:m] * unf.Qm[:m, m]) @ Vnew[:m, :]
                    )
                unf.cols = np.concatenate((unf.cols, cols_new))
                unf.colc = np.concatenate((unf.colc, cp_new if own else unf.col_part(cols_new)))
                unf.avail_c = np.concatenate((unf.avail_c, np.ones(cols_new.size, dtype=bool)))
                unf.V.add_cols(Vnew)

        if unf.npre_done < len(unf.prefixes):
            rows_new = np.concatenate([
                unf.my_new_rows(pidx)
                for pidx in range(unf.npre_done, len(unf.prefixes))
            ])
            unf.npre_done = len(unf.prefixes)
            if rows_new.size:
                l = len(unf.dl)
                own = [(m, jm) for m, jm in enumerate(unf.piv_cols) if unf.owner_col(jm) == self.my_pc]
                mine = []
                rp_new = unf.row_part(rows_new)
                if own:   # the new rows at all my pivot columns, one batch
                    vals = unf.fetch_parts(rp_new, unf.col_part([jm for _, jm in own]))
                    mine = [(m, vals[:, t]) for t, (m, _) in enumerate(own)]
                allm = _prof_comm('piv', self.row_comm,
                                  lambda: self.row_comm.allgather(mine))
                Ap = np.empty((rows_new.size, l))
                for chunk in allm:
                    for m, colv in chunk:
                        Ap[:, m] = colv
                Unew = np.empty((rows_new.size, l))
                for m in range(l):
                    Unew[:, m] = Ap[:, m] - (
                        Unew[:, :m] @ (unf.dl[:m] * unf.Pm[:m, m])
                    )
                unf.rows = np.concatenate((unf.rows, rows_new))
                unf.rowc = np.concatenate((unf.rowc, rp_new))
                unf.avail_r = np.concatenate((unf.avail_r, np.ones(rows_new.size, dtype=bool)))
                unf.U.add_rows(Unew)

    def sync_all(self, ks):
        """Synchronize unfoldings in identical order on every rank.

        MPI collectives remain on the main thread. Only independent local
        searches are threaded (via self._map in round()).
        """
        for k in ks:
            self.sync(self.unfs[k])

    # ---- phases ----
    def initialise(self):
        rng = random.Random(None if self.seed is None else self.seed + self.rank)
        x, val = kickstart(self.data, self.dims, rng)
        cand = _prof_comm('piv', self.world, lambda: self.world.allgather((abs(val), self.rank, x, val)))
        best = max(cand, key=lambda t: (t[0], -t[1]))
        x, e0 = best[2], best[3]
        for k, unf in enumerate(self.unfs):
            unf.prefixes = [tuple(x[:k])]
            unf.suffixes = [tuple(x[k + 2:])]
            unf.rows = unf.my_new_rows(0)
            unf.cols = unf.my_new_cols(0)
            unf.data = self.data  # Matrix-free: do not materialize the tile
            unf.rowc = unf.row_part(unf.rows); unf.colc = unf.col_part(unf.cols)
            unf.avail_r = np.ones(len(unf.rows), dtype=bool); unf.avail_c = np.ones(len(unf.cols), dtype=bool)
            unf.U = Grow2D(len(unf.rows), 0); unf.V = Grow2D(0, len(unf.cols))
            unf.npre_done = unf.nsuf_done = 1
        Useg, Vseg = self.exchange_segments({k: (x[k], x[k + 1]) for k in range(self.d - 1)})
        for k, unf in enumerate(self.unfs):
            self._apply_pivot(unf, x[k], x[k + 1], e0, Useg[k][0], Vseg[k][0], Vseg[k][1], Useg[k][1])
            self.Ipiv[k].append(tuple(x[:k + 1])); self.Jpiv[k].append(tuple(x[k + 1:]))

    def round(self, active):
        """One round of dimension-parallel pivot selection: every unfolding in
        `active` is searched on the state at the start of the round, the
        candidates of all unfoldings go into one allgather, and all new pivot
        segments are exchanged together.  Returns the unfoldings that gained
        a pivot."""
        self.sync_all(active)
        searchable = [k for k in active
                      if self.unfs[k].nrows_glob() > len(self.unfs[k].dl) and self.unfs[k].ncols_glob() > len(self.unfs[k].dl)]
        found = self._map(lambda k: local_search(self.unfs[k], self.rngs[k]), searchable)
        cands = dict(zip(searchable, found))
        allc = _prof_comm('piv', self.world, lambda: self.world.allgather(cands))
        winners = {}
        for k in active:
            best = None
            for dct in allc:                    # ties go to the lowest rank
                c = dct.get(k)
                if c is not None and (best is None or c[0] > best[0]):
                    best = c
            if best is None:
                continue
            aval, val, i, j = best
            threshold = (self.tol * abs(1.0 / self.unfs[k].dl[0])
                         if self.tol_type == "Relative" else self.tol)
            #if self.rank == 0:
            #    print(
            #        f"k={k}, current_rank={len(self.unfs[k].dl)}, "
            #        f"pivot={aval:.6e}, threshold={threshold:.6e}, "
            #        f"accepted={aval > threshold}"
            #    )
            if aval == 0.0 or aval <= threshold:
                # nothing left to gain: a further pivot would make A(I,J)
                # numerically singular and the core meaningless
                continue
            winners[k] = (i, j, val)
        Useg, Vseg = self.exchange_segments({k: (w[0], w[1]) for k, w in winners.items()})
        for k, (i, j, val) in winners.items():
            unf = self.unfs[k]
            self._apply_pivot(unf, i, j, val, Useg[k][0], Vseg[k][0], Vseg[k][1], Useg[k][1])
            pidx, ik = divmod(i, unf.nk); sidx, ik1 = divmod(j, unf.nk1)
            prefix = unf.prefixes[pidx] + (ik,)
            suffix = (ik1,) + unf.suffixes[sidx]
            self.Ipiv[k].append(prefix); self.Jpiv[k].append(suffix)
            if k + 1 <= self.d - 2:
                self.add_prefix(self.unfs[k + 1], prefix)
            if k - 1 >= 0:
                self.add_suffix(self.unfs[k - 1], suffix)
        return list(winners)

    def run(self, rs):
        self.initialise()

        active = set(range(len(rs)))

        for _ in range(max(rs) - 1):

            searchable = sorted(
                k for k in active if rs[k] > 1
            )

            if not searchable:
                break

            winners = self.round(searchable)

            # Remove unfoldings that did not accept a pivot
            active.intersection_update(winners)

            # Decrease remaining rank budgets
            for k in searchable:
                rs[k] -= 1

            if not active:
                break



    # TT core construction is performed by core_construct on MPI_COMM_WORLD.

    # ---- phases ----
    def tile_area(self):
        return sum(len(u.rows) * len(u.cols) for u in self.unfs)


def tt_cross_os2_mpi2(
    tensor_entry_fn,
    tensor_dim,
    sample_row=None,
    sample_col=None,
    min_rank=None,
    max_rank=None,
    tol=None,
    sampling_type=None,
    tol_type="Relative",
    adap_toler=None,
    process_grid=(1, 1),
    block_size=1,
    number_trial=1,
    err_sample_size=0,
    seed=None,
    return_info=False,
    pivot_ranks=None,
    max_peid_ranks=None,
    nthreads=None,
    tensor_block_fn=None,
):
    """
    Collective MPI TT-Cross: all pivot ranks share every unfolding.
    Threads run independent local unfolding searches; MPI calls stay on main thread.

    pivot_ranks:
        None: use all MPI ranks for pivot searching.
        int:  use only the first pivot_ranks MPI ranks for
              pivot searching. All MPI ranks still participate
              in parallel TT core construction.

    Returns (cores, FI, FJ) on world rank 0.
    Other ranks return (None, None, None).

    With return_info=True, appends an info dictionary.
    """

    if len(tensor_dim) < 2:
        raise ValueError("tensor_dim must have at least two modes")

    if max_rank is None or len(max_rank) != len(tensor_dim) - 1:
        raise ValueError("max_rank must contain d-1 ranks")

    if number_trial < 1:
        raise ValueError("number_trial must be positive")

    if seed is None:
        value = os.environ.get("TT_SEED")
        seed = int(value) if value not in (None, "") else None

    global _TENSOR_DIMS, _CALLABLE_TENSOR_ENTRY
    global _PROFILE_SYNC, _PROF_MAXRANKS, _PROF_OUT

    dims = list(map(int, tensor_dim))
    rs = list(map(int, max_rank))

    _TENSOR_DIMS = dims
    _CALLABLE_TENSOR_ENTRY = tensor_entry_fn

    world = MPI.COMM_WORLD
    P = world.Get_size()
    world_rank = world.Get_rank()

    ngroups = len(dims) - 1
    pivot_size = P if pivot_ranks is None else int(pivot_ranks)
    if pivot_size < 1 or pivot_size > P:
        raise ValueError("pivot_ranks must be between 1 and MPI world size")
    active_pivot = world_rank < pivot_size
    pivot_comm = world.Split(0 if active_pivot else MPI.UNDEFINED, world_rank)

    # One shared 2D MPI grid for ALL unfoldings (not one group per unfolding).
    # process_grid must be a single pair, or None for a near-square grid.
    if process_grid is None or process_grid == (1, 1):
        pr = max(q for q in range(1, math.isqrt(pivot_size) + 1)
                 if pivot_size % q == 0)
        pc = pivot_size // pr
    else:
        if len(process_grid) != 2 or any(isinstance(v, (list, tuple)) for v in process_grid):
            raise ValueError("threaded dimension mode requires one (pr,pc) MPI grid")
        pr, pc = map(int, process_grid)
    if pr < 1 or pc < 1 or pr * pc != pivot_size:
        raise ValueError(f"process_grid {pr}x{pc} must use all {pivot_size} pivot ranks")
    nb = int(block_size)
    if nb < 1:
        raise ValueError("block_size must be positive")
    if nthreads is None:
        nthreads = int(os.environ.get("TT_THREADS", "1"))
    nthreads = int(nthreads)
    if nthreads < 1:
        raise ValueError("nthreads must be positive")
    _SEED = seed
    _PROFILE_SYNC = os.environ.get("TT_PROFILE") not in (None, "", "0")
    _PROF_MAXRANKS = int(os.environ.get("TT_PROFILE_RANKS", "16"))
    _PROF_OUT = os.environ.get("TT_PROFILE_OUT")
    data = FunctionalTensor(dims, tensor_block_fn)
    dimension_times = None
    core_info = None
    cores = None
    I_row_unrav = None
    I_col_unrav = None

    # -------------------------------------------------------
    # Trials
    # -------------------------------------------------------

    for ll in range(number_trial):

        _prof_reset()

        world.Barrier()

        # ---------------------------------------------------
        # Pivot search: active ranks only
        # ---------------------------------------------------

        t0 = MPI.Wtime()

        if active_pivot:
            cross = PivotOrderCross(dims, pr, pc, nb, data, pivot_comm,
                                    _SEED, tol=0.0 if tol is None else tol,
                                    nthreads=nthreads, tol_type=tol_type)
            try:
                cross.run(rs)
            finally:
                if cross.pool is not None:
                    cross.pool.shutdown(wait=True)
            local_dimension_times = None
        else:
            cross = None
        # All dimensions are processed by all pivot ranks; no separate groups.
        dimension_times = None

        # Synchronize before recording the pivot phase end.
        world.Barrier()

        _PROF["piv"]["total"] = MPI.Wtime() - t0



        # ---------------------------------------------------
        # MPI-parallel TT core construction: ALL ranks
        # ---------------------------------------------------

        tc0 = MPI.Wtime()

        if world_rank == 0:
            I_row_unrav = [[tuple(map(int, idx)) for idx in piv]
                            for piv in cross.Ipiv]
            I_col_unrav = [[tuple(map(int, idx)) for idx in piv]
                            for piv in cross.Jpiv]
        else:
            I_row_unrav = None
            I_col_unrav = None
        I_row_unrav = world.bcast(I_row_unrav, root=0)
        I_col_unrav = world.bcast(I_col_unrav, root=0)

        sample_row_test = (
            [0] * ngroups
            if sample_row is None
            else list(map(int, sample_row))
        )

        sample_col_test = (
            [0] * ngroups
            if sample_col is None
            else list(map(int, sample_col))
        )

        # PEID uses the full MPI communicator.
        cores, core_info = (
            peid_par_ltr_adap_parallel(
                tensor_entry_core_construct,
                I_row_unrav,
                I_col_unrav,
                sample_row_test,
                sample_col_test,
                dims,
                comm=world,
                adap_tol=adap_toler,
                return_info=True,
                max_peid_ranks=max_peid_ranks,
            )
        )

        _PROF["core"]["total"] = (
            MPI.Wtime() - tc0
        )

        core_wall = world.allreduce(
            _PROF["core"]["total"],
            op=MPI.MAX
        )

    
        # ---------------------------------------------------
        # Per-rank profiling
        # ---------------------------------------------------

        if active_pivot:
            area = cross.tile_area()
            fetched = sum(unf.fetched for unf in cross.unfs)
        else:
            area = 0
            fetched = 0

        rows = world.gather(
            (
                world_rank,
                _PROF["piv"]["total"],
                _PROF["piv"]["comm"],
                _PROF["piv"]["wait"],
                _PROF["core"]["total"],
                _PROF["core"]["comm"],
                _PROF["core"]["wait"],
                area,
                fetched
            ),
            root=0
        )

        if world_rank == 0:

            prof = [
                {
                    "rank": r[0],
                    "piv_total": r[1],
                    "piv_comm": r[2],
                    "piv_wait": r[3],
                    "piv_compute": max(
                        r[1] - r[2] - r[3], 0.0
                    ),
                    "core_total": r[4],
                    "core_comm": r[5],
                    "core_wait": r[6],
                    "core_compute": max(
                        r[4] - r[5] - r[6], 0.0
                    ),
                    "area": r[7],
                    "fetched": r[8]
                }
                for r in rows
            ]

            if _PROF_OUT:

                cols = [
                    "rank",
                    "piv_total",
                    "piv_compute",
                    "piv_comm",
                    "piv_wait",
                    "core_total",
                    "core_compute",
                    "core_comm",
                    "core_wait"
                ]

                with open(_PROF_OUT, "w") as fh:

                    fh.write(
                        ",".join(cols) + "\n"
                    )

                    for r in prof:
                        fh.write(
                            ",".join(
                                str(r[c]) for c in cols
                            ) + "\n"
                        )

    # -------------------------------------------------------
    # Return information
    # -------------------------------------------------------

    info = {
        "pivot_search": {
            "architecture": "all-pivot-ranks-per-unfolding; threads-across-unfoldings",
            "group_sizes": [pivot_size] * ngroups,
            "grids": [(pr, pc)] * ngroups,
            "groups": [list(range(pivot_size)) for _ in range(ngroups)],
            "block_size": nb,
            "pivot_ranks": pivot_size,
            "inactive_ranks": list(range(pivot_size, P)),
            "threads_per_rank": nthreads,
            "dimension_times": dimension_times,
        },
        "core_construction": core_info,
        "timings": {
            "pivot_search": _PROF["piv"]["total"],
            "core_construction": core_wall,
        },
    }
    

    result = (
        (cores, I_row_unrav, I_col_unrav)
        if world_rank == 0
        else (None, None, None)
    )

    # Release the pivot communicator.
    if active_pivot:
        pivot_comm.Free()

    return result + (info,) if return_info else result

def peid_par_ltr_adap(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,adap_tol = None):

    trunc_ranks = [len(i) for i in FI]
    cores = []
    uj = []
    uj_tilde = []

    n = len(trunc_ranks)

    FI = [[[int(i) for i in FI[j][k]] for k in range(len(FI[j]))] for j in range(len(FI))]
    FJ = [[[int(i) for i in FJ[j][k]] for k in range(len(FJ[j]))] for j in range(len(FJ))]

    uFI,uFJ = sample_nested(FI,FJ,sample_row,sample_col,tensor_dim,0)
    uK,uL = set_conversion(uFI, uFJ, tensor_dim)

    I = uK
    J = uL

    uFI = [np.array(i) for i in uFI]
    uFJ = [np.array(i) for i in uFJ]

    for i in range(n+1):
        if i == n:
            mat = unfolding_submatrix(tensor_entry, uFI, uFJ, i-1, tensor_dim, 0)
            A = np.linalg.lstsq(uj[-1][I[i-1],:],mat,rcond = None)[0]
            cores.append(np.reshape(A,[trunc_ranks[i-1],tensor_dim[i],1]))
        elif i == 0:
            mat = unfolding_submatrix(tensor_entry, uFI, uFJ, i, tensor_dim, 1)
            q,r = sp.linalg.qr(mat,mode = "economic")
            U,s_values,_ = np.linalg.svd(r)
            if adap_tol is not None:
                cutoff = rank_cutoff(s_values, adap_tol)
                trunc_ranks[i] = cutoff
            u = q@U
            uj.append(u[:,:trunc_ranks[i]])
            cores.append(np.reshape(uj[0],[1,tensor_dim[i],trunc_ranks[i]]))
        else:
            mat = unfolding_submatrix(tensor_entry, uFI, uFJ, i, tensor_dim, 1)
            q,r = sp.linalg.qr(mat,mode = "economic")
            U,s_values,_ = np.linalg.svd(r)
            if adap_tol is not None:
                cutoff = rank_cutoff(s_values, adap_tol)
                trunc_ranks[i] = cutoff
            u = q@U
            uj.append(u[:,:trunc_ranks[i]])

            uj_tilde = np.reshape(uj[i],[len(uFI[i-1]),tensor_dim[i]*trunc_ranks[i]])
            A = np.linalg.lstsq(uj[i-1][I[i-1],:],uj_tilde,rcond = None)[0]
            cores.append(np.reshape(A,[trunc_ranks[i-1],tensor_dim[i],trunc_ranks[i]]))

    return cores

def peid_par_ltr_adap_parallel(
    tensor_entry, FI, FJ, sample_row, sample_col,
    tensor_dim, comm=None, adap_tol=None,
    print_grid=False, return_info=False,
    max_peid_ranks=None
):
    """
    Construct TT cores using MPI.
    max_peid_ranks:
        None: use all available MPI ranks.
        int:  use at most this many ranks for PEID.
    With return_info=True, return (cores, info).
    Cores are returned only on world rank 0.
    Timing and group metadata are returned on every rank.
    """
    import time
    d = len(tensor_dim)
    # --------------------------------------------------
    # Serial execution
    # --------------------------------------------------
    if MPI is None and comm is None:
        t0 = time.perf_counter()
        cores = peid_par_ltr_adap(
            tensor_entry, FI, FJ,
            sample_row, sample_col,
            tensor_dim, adap_tol=adap_tol
        )
        elapsed = time.perf_counter() - t0
        info = {
            "core_group_sizes": [1] * d,
            "core_groups": [[0] for _ in range(d)],
            "dimension_times": None,
            "dimension_active_times": None,
            "total_time": elapsed,
            "peid_ranks": 1,
            "inactive_ranks": []
        }
        return (cores, info) if return_info else cores
    if comm is None:
        comm = MPI.COMM_WORLD
    world_comm = comm
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()
    # --------------------------------------------------
    # Select number of ranks for PEID
    # --------------------------------------------------
    if max_peid_ranks is None:
        peid_size = world_size
    else:
        peid_size = min(
            int(max_peid_ranks), world_size
        )
    if peid_size < 1:
        raise ValueError("max_peid_ranks must allow at least one MPI rank")
    if peid_size < d:
        # Fallback: execute the existing sequential PEID algorithm on world
        # rank 0. All ranks participate in the metadata broadcast below.
        # This avoids communicator deadlocks and preserves the serial method.
        cores = None
        info = None
        if world_rank == 0:
            t0 = MPI.Wtime()
            cores = peid_par_ltr_adap(
                tensor_entry, FI, FJ, sample_row, sample_col,
                tensor_dim, adap_tol=adap_tol
            )
            elapsed = MPI.Wtime() - t0
            info = {
                "core_group_sizes": [1] * d,
                "core_groups": [[0] for _ in range(d)],
                "dimension_times": None,
                "dimension_active_times": None,
                "total_time": elapsed,
                "execution_mode": "serial_fallback",
            }
            if print_grid:
                print("PEID: fewer than d ranks; using serial core construction on rank 0", flush=True)
        info = world_comm.bcast(info, root=0)
        info["peid_ranks"] = 1
        info["inactive_ranks"] = list(range(1, world_size))
        return (cores, info) if return_info else cores
    active_peid = world_rank < peid_size
    peid_comm = world_comm.Split(
        color=0 if active_peid else MPI.UNDEFINED,
        key=world_rank
    )
    cores = None
    info = None
    # --------------------------------------------------
    # Active PEID ranks only
    # --------------------------------------------------
    if active_peid:
        comm = peid_comm
        rank = comm.Get_rank()
        size = comm.Get_size()
        # ----------------------------------------------
        # One-dimensional case
        # ----------------------------------------------
        if d == 1:
            t0 = MPI.Wtime()
            if rank == 0:
                cores = peid_par_ltr_adap(
                    tensor_entry, FI, FJ,
                    sample_row, sample_col,
                    tensor_dim, adap_tol=adap_tol
                )
            elapsed = MPI.Wtime() - t0
            elapsed = comm.allreduce(
                elapsed, op=MPI.MAX
            )
            info = {
                "core_group_sizes": [1],
                "core_groups": [[0]],
                "dimension_times": [elapsed],
                "dimension_active_times": [elapsed]
            }
        else:
            trunc_ranks = [len(i) for i in FI]
            n = len(trunc_ranks)
            # ------------------------------------------
            # Sampling and index conversion
            # ------------------------------------------
            if rank == 0:
                FI = [
                    [
                        [int(i) for i in FI[j][k]]
                        for k in range(len(FI[j]))
                    ]
                    for j in range(len(FI))
                ]
                FJ = [
                    [
                        [int(i) for i in FJ[j][k]]
                        for k in range(len(FJ[j]))
                    ]
                    for j in range(len(FJ))
                ]
                
                uFI, uFJ = sample_nested(
                    FI, FJ, sample_row, sample_col,
                    tensor_dim, 0
                )
                
                uK, uL = set_conversion(
                    uFI, uFJ, tensor_dim
                )
                I = uK
                J = uL
                uFI = [np.array(i) for i in uFI]
                uFJ = [np.array(i) for i in uFJ]
            else:
                uFI = None
                uFJ = None
                I = None
                J = None
            uFI = comm.bcast(uFI, root=0)
            uFJ = comm.bcast(uFJ, root=0)
            I = comm.bcast(I, root=0)
            J = comm.bcast(J, root=0)
            # ------------------------------------------
            # Determine QR matrix sizes
            # ------------------------------------------
            qr_shapes = []
            qr_weights = []
            for i in range(n):
                row_count = (
                    tensor_dim[0]
                    if i == 0
                    else len(uFI[i-1]) * tensor_dim[i]
                )
                col_count = len(uFJ[i])
                qr_shapes.append(
                    (row_count, col_count)
                )
                qr_weights.append(row_count)
            # ------------------------------------------
            # Allocate PEID ranks to dimensions
            # ------------------------------------------
            group_sizes = [1 for _ in range(d)]
            extra_ranks = size - d
            if extra_ranks > 0:
                weights = np.asarray(
                    qr_weights, dtype=float
                )
                if np.allclose(weights.sum(), 0.0):
                    proportions = (
                        np.ones(n, dtype=float) / n
                    )
                else:
                    proportions = (
                        weights / weights.sum()
                    )
                raw_extra = proportions * extra_ranks
                extra_alloc = np.floor(
                    raw_extra
                ).astype(int)
                remaining = (
                    extra_ranks - int(np.sum(extra_alloc))
                )
                if remaining > 0:
                    for idx in np.argsort(
                        raw_extra - extra_alloc
                    )[::-1][:remaining]:
                        extra_alloc[idx] += 1
                for i in range(n):
                    group_sizes[i] += int(extra_alloc[i])
            core_groups = []
            start = 0
            for group_size in group_sizes:
                stop = start + group_size
                core_groups.append(
                    list(range(start, stop))
                )
                start = stop
            info = {
                "core_group_sizes": group_sizes,
                "core_groups": core_groups
            }
            if print_grid and rank == 0:
                print(
                    "PEID core groups:",
                    core_groups,
                    flush=True
                )
            # ------------------------------------------
            # Create per-core communicator
            # ------------------------------------------
            core_index = next(
                i for i, ranks in enumerate(core_groups)
                if rank in ranks
            )
            group_roots = [
                ranks[0] for ranks in core_groups
            ]
            subcomm = comm.Split(
                color=core_index,
                key=rank
            )
            subrank = subcomm.Get_rank()
            # ------------------------------------------
            # Per-dimension timing
            # ------------------------------------------
            core_start = MPI.Wtime()
            recv_wait = 0.0
            core = None
            # ------------------------------------------
            # Last TT core
            # ------------------------------------------
            if core_index == n:
                if subrank == 0:
                    t_recv = MPI.Wtime()
                    uj_prev = comm.recv(
                        source=group_roots[core_index-1],
                        tag=1000+core_index
                    )
                    recv_wait += MPI.Wtime() - t_recv

                    mat = unfolding_submatrix(
                        tensor_entry,
                        uFI,
                        uFJ,
                        core_index-1,
                        tensor_dim,
                        0
                    )
                    
                    A = np.linalg.lstsq(
                        uj_prev[I[core_index-1], :],
                        mat,
                        rcond=None
                    )[0]
                    
                    core = np.reshape(
                        A,
                        [
                            uj_prev.shape[1],
                            tensor_dim[core_index],
                            1
                        ]
                    )

            # ------------------------------------------
            # First and interior TT cores
            # ------------------------------------------
            else:
                if core_index == 0:
                    gap = np.linspace(
                        0,
                        tensor_dim[core_index],
                        subcomm.Get_size()+1
                    ).astype(int)
                else:
                    gap = np.linspace(
                        0,
                        len(uFI[core_index-1]),
                        subcomm.Get_size()+1
                    ).astype(int)
                
                mat_local = (
                    unfolding_submatrix_slice(
                        tensor_entry,
                        range(
                            gap[subrank],
                            gap[subrank+1]
                        ),
                        uFI,
                        uFJ,
                        core_index,
                        tensor_dim,
                        1,
                        "row"
                    )
                )
                
                q, r = qr_tsqr_local(
                    mat_local,
                    comm=subcomm,
                    gather_q=True
                )
                
                if subrank == 0:
                    U, s_values, _ = np.linalg.svd(r)
                    if adap_tol is not None:
                        cutoff = rank_cutoff(
                            s_values, adap_tol
                        )
                        trunc_ranks[core_index] = cutoff
                    u = q @ U
                    uj = u[:, :trunc_ranks[core_index]]
                    # Send U_j before constructing this TT core.
                    # Nonblocking communication avoids serializing solves.
                    send_request = comm.isend(
                        uj,
                        dest=group_roots[core_index+1],
                        tag=1000+core_index+1
                    )
                    if core_index == 0:
                        core = np.reshape(
                            uj,
                            [
                                1,
                                tensor_dim[core_index],
                                uj.shape[1]
                            ]
                        )
                    else:
                        t_recv = MPI.Wtime()
                        uj_prev = comm.recv(
                            source=group_roots[core_index-1],
                            tag=1000+core_index
                        )
                        recv_wait += (
                            MPI.Wtime() - t_recv
                        )
                        uj_tilde = np.reshape(
                            uj,
                            [
                                len(uFI[core_index-1]),
                                tensor_dim[core_index]
                                * uj.shape[1]
                            ]
                        )
                        A = np.linalg.lstsq(
                            uj_prev[I[core_index-1], :],
                            uj_tilde,
                            rcond=None
                        )[0]
                        core = np.reshape(
                            A,
                            [
                                uj_prev.shape[1],
                                tensor_dim[core_index],
                                uj.shape[1]
                            ]
                        )
                    # Ensure the outgoing U_j transfer completes.
                    send_request.wait()
            # ------------------------------------------
            # Collect per-dimension timing
            # ------------------------------------------
            core_elapsed_local = (
                MPI.Wtime() - core_start
            )
            core_active_local = (
                core_elapsed_local - recv_wait
            )
            core_elapsed = subcomm.allreduce(
                core_elapsed_local,
                op=MPI.MAX
            )
            core_active = subcomm.allreduce(
                core_active_local,
                op=MPI.MAX
            )
            # ------------------------------------------
            # Gather cores and timing information
            # ------------------------------------------
            core_result = (
                core_index,
                core,
                core_elapsed,
                core_active
            ) if subrank == 0 else None
            core_results = comm.gather(
                core_result,
                root=0
            )
            dimension_times = None
            dimension_active_times = None
            if rank == 0:
                ordered = sorted(
                    (
                        result
                        for result in core_results
                        if result is not None
                    ),
                    key=lambda x: x[0]
                )
                cores = [
                    item[1] for item in ordered
                ]
                dimension_times = [
                    item[2] for item in ordered
                ]
                dimension_active_times = [
                    item[3] for item in ordered
                ]
            dimension_times = comm.bcast(
                dimension_times,
                root=0
            )
            dimension_active_times = comm.bcast(
                dimension_active_times,
                root=0
            )
            info["dimension_times"] = (
                dimension_times
            )
            info["dimension_active_times"] = (
                dimension_active_times
            )
            subcomm.Free()
        peid_comm.Free()
    # --------------------------------------------------
    # Return metadata to all original MPI ranks
    # --------------------------------------------------
    # Rank 0 belongs to the active PEID communicator.
    # Only metadata is broadcast; TT cores remain
    # on world rank 0.
    info = world_comm.bcast(
        info if world_rank == 0 else None,
        root=0
    )
    info["peid_ranks"] = peid_size
    info["inactive_ranks"] = list(
        range(peid_size, world_size)
    )
    return (cores, info) if return_info else cores



def rank_cutoff(s_values, adap_tol):
    if len(s_values) == 0:
        return 1
    leading = s_values[0]
    if np.isclose(leading, 0.0):
        return 1
    norm_s = np.asarray(s_values) / leading
    return max(1, int(np.count_nonzero(norm_s >= adap_tol)))

def qr_tsqr_local(local_data, comm=None, gather_q=True):
    if MPI is None and comm is None:
        return np.linalg.qr(local_data, mode="reduced")

    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    local_data = np.asarray(local_data)
    if local_data.ndim != 2:
        raise ValueError("qr_tsqr_local expects a two-dimensional array")

    _, n = local_data.shape
    q1, r1 = np.linalg.qr(local_data, mode="reduced")

    r_rows = comm.gather(r1.shape[0], root=0)
    r_blocks = comm.gather(r1, root=0)

    if rank == 0:
        r0 = np.vstack(r_blocks) if r_blocks else np.empty((0, n), dtype=local_data.dtype)
        q2, r = np.linalg.qr(r0, mode="reduced")
        q2_starts = np.cumsum([0] + r_rows)
        q2_blocks = [
            q2[q2_starts[i]:q2_starts[i + 1], :]
            for i in range(size)
        ]
    else:
        r = None
        q2_blocks = None

    q2_local = comm.scatter(q2_blocks, root=0)
    q_local = q1 @ q2_local

    if not gather_q:
        r = comm.bcast(r, root=0)
        return q_local, r

    q_blocks = comm.gather(q_local, root=0)
    if rank == 0:
        return np.vstack(q_blocks), r
    return None, None

def unfolding_submatrix_slice(tensor_entry,group,uFI,uFJ,k,tensor_dim,axis,part):
    
    if axis == 0:
        if part == "row":

            if k==0:
                fst_indt = np.repeat(uFI[0][group],tensor_dim[k+1]*len(uFJ[k+1]))
                mid_indt = np.tile(np.arange(tensor_dim[k+1]),len(uFJ[k+1])*len(uFI[k][group]))
                lst_indt = np.tile(np.repeat(uFJ[k+1],tensor_dim[k+1],axis = 0),(len(uFI[k][group]),1))
                lists = [fst_indt,mid_indt,lst_indt]
                
                ind = np.column_stack(lists)

                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[0][group]),tensor_dim[1]*len(uFJ[k+1])])
            elif k==len(tensor_dim)-2:
                fst_indt = np.repeat(uFI[k][group],tensor_dim[k+1],axis = 0)
                lst_indt = np.tile(np.arange(tensor_dim[k+1]),len(uFI[k][group]))

                lists = [fst_indt,lst_indt]
                ind = np.column_stack(lists)
                
                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[k][group]),tensor_dim[k+1]])
                
            else:
                fst_indt = np.repeat(uFI[k][group],tensor_dim[k+1]*len(uFJ[k+1]),axis = 0)
                mid_indt = np.tile(np.arange(tensor_dim[k+1]),len(uFJ[k+1])*len(uFI[k][group]))
                lst_indt = np.tile(np.repeat(uFJ[k+1],tensor_dim[k+1],axis = 0),(len(uFI[k][group]),1))
                lists = [fst_indt,mid_indt,lst_indt]

                ind = np.column_stack(lists)

                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[k][group]),tensor_dim[k+1]*len(uFJ[k+1])])
        elif part == "col":


            if k==0:
                fst_indt = np.repeat(uFI[0],tensor_dim[k+1]*len(uFJ[k+1][group]))
                mid_indt = np.tile(np.arange(tensor_dim[k+1]),len(uFJ[k+1][group])*len(uFI[k]))
                lst_indt = np.tile(np.repeat(uFJ[k+1][group],tensor_dim[k+1],axis = 0),(len(uFI[k]),1))
                lists = [fst_indt,mid_indt,lst_indt]
                
                ind = np.column_stack(lists)

                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[0]),tensor_dim[1]*len(uFJ[k+1][group])])
            elif k==len(tensor_dim)-2:
                fst_indt = np.repeat(uFI[k],len(group),axis = 0)
                lst_indt = np.tile(np.arange(group),len(uFI[k]))

                lists = [fst_indt,lst_indt]
                ind = np.column_stack(lists)
                
                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[k]),len(group)])
                
            else:
                fst_indt = np.repeat(uFI[k],tensor_dim[k+1]*len(uFJ[k+1][group]),axis = 0)
                mid_indt = np.tile(np.arange(tensor_dim[k+1]),len(uFJ[k+1][group])*len(uFI[k]))
                lst_indt = np.tile(np.repeat(uFJ[k+1][group],tensor_dim[k+1],axis = 0),(len(uFI[k]),1))
                lists = [fst_indt,mid_indt,lst_indt]

                ind = np.column_stack(lists)

                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[k]),tensor_dim[k+1]*len(uFJ[k+1][group])])

    elif axis == 1:
        if part == "row":
            if k==0:
                
                fst_ind = np.repeat(group,len(uFJ[k]),axis = 0)
                lst_ind = np.tile(uFJ[k],(len(group),1))
                lists = [fst_ind,lst_ind]
                ind = np.column_stack(lists)
                
                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(group),len(uFJ[k])])
                
            elif k == len(tensor_dim)-2:
                
                fst_ind = np.repeat(uFI[k-1][group],len(uFJ[k])*tensor_dim[k],axis = 0)
                mid_ind = np.tile(np.repeat(np.arange(tensor_dim[k]),len(uFJ[k])),len(uFI[k-1][group]))
                lst_ind = np.tile(uFJ[k],(len(uFI[k-1][group])*tensor_dim[k],1))
                
                lists = [fst_ind,mid_ind,lst_ind]

                ind = np.column_stack(lists)

                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[k-1][group])*tensor_dim[k],len(uFJ[k])])

            else:
                
                fst_ind = np.repeat(uFI[k-1][group],len(uFJ[k])*tensor_dim[k],axis = 0)
                mid_ind = np.tile(np.repeat(np.arange(tensor_dim[k]),len(uFJ[k])),len(uFI[k-1][group]))
                lst_ind = np.tile(uFJ[k],(len(uFI[k-1][group])*tensor_dim[k],1))
                
                lists = [fst_ind,mid_ind,lst_ind]

                ind = np.column_stack(lists)

                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[k-1][group])*tensor_dim[k],len(uFJ[k])]) 
        elif part == "col":
            if k==0:
                
                fst_ind = np.repeat(np.arange(tensor_dim[k]),len(uFJ[k][group]),axis = 0)
                lst_ind = np.tile(uFJ[k][group],(tensor_dim[k],1))
                lists = [fst_ind,lst_ind]
                ind = np.column_stack(lists)
                
                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[tensor_dim[k],len(uFJ[k][group])])
                
            elif k == len(tensor_dim)-2:

                fst_ind = np.repeat(uFI[k-1],len(uFJ[k][group])*tensor_dim[k],axis = 0)
                mid_ind = np.tile(np.repeat(np.arange(tensor_dim[k]),len(uFJ[k][group])),len(uFI[k-1]))
                lst_ind = np.tile(uFJ[k][group],(len(uFI[k-1])*tensor_dim[k],1))
                
                lists = [fst_ind,mid_ind,lst_ind]

                ind = np.column_stack(lists)

                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[k-1])*tensor_dim[k],len(uFJ[k][group])])

            else:

                fst_ind = np.repeat(uFI[k-1],len(uFJ[k][group])*tensor_dim[k],axis = 0)
                mid_ind = np.tile(np.repeat(np.arange(tensor_dim[k]),len(uFJ[k][group])),len(uFI[k-1]))
                lst_ind = np.tile(uFJ[k][group],(len(uFI[k-1])*tensor_dim[k],1))
                
                lists = [fst_ind,mid_ind,lst_ind]

                ind = np.column_stack(lists)

                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[k-1])*tensor_dim[k],len(uFJ[k][group])]) 
    
    return submatrix

def unfolding_submatrix(tensor_entry,uFI,uFJ,k,tensor_dim,axis,full = True):
    if axis == 0:
        if full:
            if k==0:
            
                fst_indt = np.repeat(uFI[0],tensor_dim[k+1]*len(uFJ[k+1]))
                mid_indt = np.tile(np.arange(tensor_dim[k+1]),len(uFJ[k+1])*len(uFI[k]))
                lst_indt = np.tile(np.repeat(uFJ[k+1],tensor_dim[k+1],axis = 0),(len(uFI[k]),1))
                lists = [fst_indt,mid_indt,lst_indt]
                
                ind = np.column_stack(lists)

                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[0]),tensor_dim[1]*len(uFJ[k+1])])
            elif k==len(tensor_dim)-2:

                fst_indt = np.repeat(uFI[k],tensor_dim[k+1],axis = 0)
                lst_indt = np.tile(np.arange(tensor_dim[k+1]),len(uFI[k]))


                lists = [fst_indt,lst_indt]
                ind = np.column_stack(lists)

                
                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[k]),tensor_dim[k+1]])
                
            else:

                fst_indt = np.repeat(uFI[k],tensor_dim[k+1]*len(uFJ[k+1]),axis = 0)
                mid_indt = np.tile(np.arange(tensor_dim[k+1]),len(uFJ[k+1])*len(uFI[k]))
                lst_indt = np.tile(np.repeat(uFJ[k+1],tensor_dim[k+1],axis = 0),(len(uFI[k]),1))
                lists = [fst_indt,mid_indt,lst_indt]


                ind = np.column_stack(lists)

                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[k]),tensor_dim[k+1]*len(uFJ[k+1])])

        else:
            
            if k==len(tensor_dim)-2:

                fst_indt = np.repeat(uFI[k],tensor_dim[k+1],axis = 0)
                lst_indt = np.tile(np.arange(tensor_dim[k+1]),len(uFI[k]))


                lists = [fst_indt,lst_indt]
                ind = np.column_stack(lists)

                
                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[k]),tensor_dim[k+1]])
                
            else:

                fst_indt = np.repeat(uFI[k],tensor_dim[k+1],axis = 0)
                mid_indt = np.tile(np.arange(tensor_dim[k+1]),len(uFI[k]))
                lst_indt = np.tile(np.repeat([uFJ[k+1][-1]],tensor_dim[k+1],axis = 0),(len(uFI[k]),1))
                lists = [fst_indt,mid_indt,lst_indt]


                ind = np.column_stack(lists)

                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[k]),tensor_dim[k+1]])

    elif axis == 1:
        if full:
            if k==0:
                
                fst_ind = np.repeat(np.arange(tensor_dim[k]),len(uFJ[k]),axis = 0)
                lst_ind = np.tile(uFJ[k],(tensor_dim[k],1))
                lists = [fst_ind,lst_ind]
                ind = np.column_stack(lists)
                
                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[tensor_dim[k],len(uFJ[k])])
                
            elif k == len(tensor_dim)-2:

                fst_ind = np.repeat(uFI[k-1],len(uFJ[k])*tensor_dim[k],axis = 0)
                mid_ind = np.tile(np.repeat(np.arange(tensor_dim[k]),len(uFJ[k])),len(uFI[k-1]))
                lst_ind = np.tile(uFJ[k],(len(uFI[k-1])*tensor_dim[k],1))
                
                lists = [fst_ind,mid_ind,lst_ind]

                ind = np.column_stack(lists)

                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[k-1])*tensor_dim[k],len(uFJ[k])])

            else:

                fst_ind = np.repeat(uFI[k-1],len(uFJ[k])*tensor_dim[k],axis = 0)
                mid_ind = np.tile(np.repeat(np.arange(tensor_dim[k]),len(uFJ[k])),len(uFI[k-1]))
                lst_ind = np.tile(uFJ[k],(len(uFI[k-1])*tensor_dim[k],1))
                
                lists = [fst_ind,mid_ind,lst_ind]

                ind = np.column_stack(lists)

                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[k-1])*tensor_dim[k],len(uFJ[k])]) 
        else:
            if k==0:
                
                fst_ind = np.repeat(np.arange(tensor_dim[k]),len(uFJ[k]),axis = 0)
                lst_ind = np.tile(uFJ[k],(tensor_dim[k],1))
                lists = [fst_ind,lst_ind]
                ind = np.column_stack(lists)
                
                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[tensor_dim[k],len(uFJ[k])])

            else:

                fst_ind = np.repeat([uFI[k-1][-1]],len(uFJ[k])*tensor_dim[k],axis = 0)
                mid_ind = np.repeat(np.arange(tensor_dim[k]),len(uFJ[k]))
                lst_ind = np.tile(uFJ[k],(tensor_dim[k],1))
                
                lists = [fst_ind,mid_ind,lst_ind]
                
                ind = np.column_stack(lists)

                submatrix = np.reshape(tensor_entry(ind,tensor_dim),[tensor_dim[k],len(uFJ[k])]) 
    return submatrix

def set_conversion(FI,FJ,dim):
    I = [[] for _ in  range(len(FI))]
    J = [[] for _ in range(len(FJ))]

    for i in range(len(FI)):
        if i==0:
            I[0] = [x for y in FI[0] for x in y]

        else:
            for j in FI[i]:
                for k in range(len(FI[i-1])):
                    if FI[i-1][k] == j[:i]:
                        I[i].append(Large_ravel([k,j[-1]],[len(FI[i-1]),dim[i]]))

    for i in range(len(FJ)-1,-1,-1):
        
        if i==len(FJ)-1:
            J[i] = [x for y in FJ[i] for x in y]
        
        else:
            for j in FJ[i]:
                for k in range(len(FJ[i+1])):
                    if FJ[i+1][k]==j[1:]:
                        J[i].append(Large_ravel([k,j[0]],[len(FJ[i+1]),dim[i+1]]))
                        break


    return I,J

def sample_nested(row,col,sample_row,sample_col,tensor_dim,axis = 0):

    FI= [[] for _ in range(len(row))]
    FJ = [[] for _ in range(len(col))]
    if axis == 0:
        for i in range(len(tensor_dim)-1):
            FJ[i] = list(col[i])
            unavail = [Large_ravel(j,tensor_dim[i+1:]) for j in col[i]]
            selected = cheap_sample(0,math.prod(tensor_dim[i+1:]),unavail,sample_col[i])
            for j in selected:
                FJ[i].append(list(Large_unravel(j,tensor_dim[i+1:])))


            if i==0:
                FI[i] = list(row[i])
                unavail = [row[i][j][-1] for j in range(len(row[i]))]
                selected = cheap_sample(0,tensor_dim[0],unavail,sample_row[i])
                for j in selected:
                    FI[i].append([j])
            else:
                FI[i] = list(row[i])
                unavail = []
                for j in row[i]:
                    count = 0
                    for k in range(len(row[i-1])):
                        if row[i-1][k]==j[:i]:
                            unavail.append(Large_ravel([k,j[-1]],[len(FI[i-1]),tensor_dim[i]]))
                            break
                        else:
                            k+=1            
                locs = cheap_sample(0,len(FI[i-1])*tensor_dim[i],unavail,sample_row[i])
                unrav = [Large_unravel(j,[len(FI[i-1]),tensor_dim[i]]) for j in locs]
                for j in unrav:
                    FI[i].append(FI[i-1][j[0]]+[j[1]])
    else:
        for i in range(len(tensor_dim)-1):
            FI[i]  = list(row[i])
            unavail = [Large_ravel(j,tensor_dim[:i+1]) for j in row[i]]
            selected = cheap_sample(0,math.prod(tensor_dim[:i+1]),unavail,sample_row[i])
            for j in selected:
                FI[i].append(list(Large_unravel(j,tensor_dim[:i+1])))

        for i in range(len(col)-1,-1,-1):
            FJ[i] = list(col[i])

            if i==len(col)-1:
                unavail = [col[i][j][0] for j in range(len(col[i]))]
                selected = cheap_sample(0,tensor_dim[-1],unavail,sample_col[i])
                for j in selected:
                    FJ[i].append([j])
            else:
                unavail = []
                for j in col[i]:
                    for k in range(len(col[i+1])):
                        if col[i+1][k]==j[i+1:]:
                            unavail.append(Large_ravel([j[0],k],[tensor_dim[i+1],len(FJ[i+1])]))
                            break
                locs = cheap_sample(0,len(FJ[i+1])*tensor_dim[i+1],unavail,sample_col[i])
                unrav = [Large_unravel(j,[tensor_dim[i+1],len(FJ[i+1])]) for j in locs]
                for j in unrav:
                    FJ[i].append([j[0]] + FJ[i+1][j[1]])

    return FI,FJ

def cheap_sample(low, high, unavailable, sample_size):
    unavailable = set(unavailable)

    num_samples = min(
        sample_size,
        (high - low) - len(unavailable)
    )

    samples = []
    selected = set(unavailable)

    while len(samples) < num_samples:
        b = random.randrange(low, high)

        if b not in selected:
            selected.add(b)
            samples.append(b)

    return samples

def Large_ravel(index, dim):
    ravel_index = 0
    multiplier = 1
    for i in reversed(range(len(dim))):
        ravel_index += index[i] * multiplier
        multiplier *= dim[i]
    return ravel_index


def Large_unravel(index, dim):
    unravel_index = [0] * len(dim)
    strides = [1] * len(dim)
    
    # compute strides
    for i in reversed(range(len(dim)-1)):
        strides[i] = strides[i+1] * dim[i+1]

    # unravel
    for i in range(len(dim)):
        unravel_index[i] = index // strides[i]
        index %= strides[i]

    return tuple(unravel_index)

def sampled_error(tensor_entry,dim,samples,*args,relative=True,seed=10):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, np.asarray(dim), size=(samples, len(dim)))
    true_values = np.asarray(tensor_entry(indices,dim))
    norm = np.sum(true_values**2)

    errors = []
    for cores in args:
        approx_values = np.asarray(Core_to_Tensor_Value(cores,indices))
        error = np.sum((true_values - approx_values)**2)
        if relative:
            errors.append(np.sqrt(error) / np.sqrt(norm))
        else:
            errors.append(np.sqrt(error))
    return errors
