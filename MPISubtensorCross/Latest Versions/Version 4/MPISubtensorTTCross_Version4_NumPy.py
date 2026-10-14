"""DISTRIBUTED MEMORY PARALLEL ADAPTIVE TENSOR-TRAIN CROSS APPROXIMATION
Version 4 (prototype): pivot-order 2D block-cyclic layout.

    srun -n P python MPISubtensorTTCross_Version4_NumPy.py err_samples trials dims ranks [pr,pc] [nb]

    err_samples  number of random entries used to sample the error
    trials       number of repetitions
    dims         tensor dimensions, comma separated
    ranks        TT core ranks, comma separated (one per unfolding)
    pr,pc        process grid, pr*pc must equal the number of ranks launched
                 (default: the most square factorisation)
    nb           block size of the cyclic distribution (default 1)

Environment: TT_SEED (seed), TT_PROFILE=1 (split idle wait from communication),
TT_PROFILE_RANKS, TT_PROFILE_OUT (per-rank CSV, same columns as Version 3).
The tensor is chosen with TENSOR_KIND below.  Helper functions (tensor
definitions, entry cache, core construction, error sampling) are those of
Version 3; the pivot-search phase is new and is described below the helpers.
"""
import numpy as np
import math
import time
import random
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from mpi4py import MPI

TENSOR_KIND = "hilbert"     # CHANGE THIS: "hilbert" | "maxwell4" | "maxwell6"
_TENSOR_DIMS = None          # filled in after argv parsing

def tensor_entry(index):
    """Evaluate one tensor entry or a batch of entries.
    """
    index = np.asarray(index)
    one_d = (index.ndim == 1)
    if one_d:
        index = index.reshape(1, -1)
    if index.ndim != 2:
        raise ValueError("index must have shape (d,) or (m, d)")

    if TENSOR_KIND == "hilbert":
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
    
    start = cores[0][:,index[0],:]
    for i in range(1,len(cores)):
        start = start@cores[i][:,index[i],:]
    value = start[0][0]
    return value


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
    """Entries are evaluated on request; any process can evaluate any entry."""
    def __init__(self, dims):
        self.dims = tuple(int(x) for x in dims)
    def fetch(self, indices):
        """indices: (m, d) int64 -> (m,) float64"""
        return np.asarray(tensor_entry(np.asarray(indices, dtype=np.int64)), dtype=np.float64).reshape(-1)


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
        self.A = Grow2D()                          # tile A_k[rows, cols]
        self.U = Grow2D()                          # u_m restricted to rows
        self.V = Grow2D()                          # v_m restricted to cols
        self.npre_done = 0                         # prefixes whose rows are materialised
        self.nsuf_done = 0                         # suffixes whose columns are materialised
        self.dl = np.zeros(0)                      # deltas
        self.piv_rows = []                         # global row index of each pivot
        self.piv_cols = []
        self.Pm = np.zeros((0, 0))                 # Pm[m,l] = v_m(j_l), m<l
        self.Qm = np.zeros((0, 0))                 # Qm[m,l] = u_m(i_l), m<l
        self.fetched = 0                           # entries evaluated/fetched

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

    def fetch_block(self, data, rows, cols, chunk=4000000):
        """A_k[rows, cols] from the data layer, in bounded-size chunks."""
        out = np.empty((len(rows), len(cols)))
        if len(rows) == 0 or len(cols) == 0:
            return out
        step = max(1, chunk // len(cols))
        for a in range(0, len(rows), step):
            rr = rows[a:a + step]
            out[a:a + len(rr)] = data.fetch(self.coords(rr, cols)).reshape(len(rr), len(cols))
        self.fetched += out.size
        return out

    # ---- residual E_l = A - sum_m delta_m u_m v_m^T on this tile ----
    def ecol(self, lc):
        return self.A.a[:, lc] - self.U.a @ (self.dl * self.V.a[:, lc])
    def erow(self, lr):
        return self.A.a[lr, :] - (self.dl * self.U.a[lr, :]) @ self.V.a
    def epairs(self, lrs, lcs):
        return self.A.a[lrs, lcs] - np.einsum('im,m,mi->i', self.U.a[lrs], self.dl, self.V.a[:, lcs])


def local_search(unf, rng, sample_size=10, maxiter=10):
    """Greedy rook search on this process's tile, excluding pivot rows/columns.
    Same scheme as Version 3's Low_Access_MPIGreedy_Search_PivV2, run on the
    cyclic tile instead of a subtensor.  Returns (|E|, E, global row, global col)."""
    avail_r = np.flatnonzero(~np.isin(unf.rows, unf.piv_rows))
    avail_c = np.flatnonzero(~np.isin(unf.cols, unf.piv_cols))
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
        sj = int(np.argmax(np.abs(err2)))
        j = avail_c[sj]
        err1 = unf.ecol(j)[avail_r]
        if np.abs(err1[si]) >= np.abs(err1).max() and np.abs(err2[sj]) >= np.abs(err2).max():
            break
    val = float(err1[si])
    return (abs(val), val, int(unf.rows[i]), int(unf.cols[j]))


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


def tt_values(cores, idx):
    """TT values at a batch of multi-indices idx (n, d)."""
    idx = np.asarray(idx, dtype=np.int64)
    v = cores[0][0, idx[:, 0], :]                               # (n, r_1)
    for k in range(1, len(cores)):
        v = np.einsum('nr,rns->ns', v, cores[k][:, idx[:, k], :])
    return v[:, 0]


def sampled_error(cores, dims, nsamples, world, seed, reps=10):
    """Relative error on nsamples random entries, repeated reps times, the
    samples split over all ranks (Version 3 draws them serially on rank 0)."""
    P, rank = world.Get_size(), world.Get_rank()
    rng = np.random.default_rng(None if seed is None else seed + 7919 + rank)
    n_loc = nsamples // P + (1 if rank < nsamples % P else 0)
    part = np.zeros((reps, 2))
    for t in range(reps):
        idx = np.stack([rng.integers(0, n, n_loc) for n in dims], 1)
        if n_loc:
            a = np.asarray(tensor_entries_glob(idx)).reshape(-1)
            b = tt_values(cores, idx)
            part[t] = [np.sum((a - b) ** 2), np.sum(a ** 2)]
    tot = _prof_comm('core', world, lambda: world.allreduce(part, op=MPI.SUM))
    return np.sqrt(tot[:, 0] / tot[:, 1])


class PivotOrderCross:
    def __init__(self, dims, pr, pc, nb, data, world, seed, tol=0.0, nthreads=1):
        self.dims = [int(x) for x in dims]
        self.tol = tol                 # stop an unfolding when |pivot residual| < tol*|first residual|
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

    def sync_all(self, ks):
        """Bring the tiles of unfoldings ks up to date with their pending
        prefixes and suffixes, with one column-group and one row-group
        allgather for all of them.

        Lazy as in Version 3: an unfolding that is never swept again stops
        growing.  Columns first (their factors need the pivot rows, all of
        which exist), then the new rows over all columns (their factors need
        the pivot columns, all of which exist).  Whether anything is pending
        is the same for every member of a group, so the collectives match.
        The fetches and recurrences of different unfoldings are independent
        and run on the thread pool when one is configured."""
        def col_fetch(k):
            unf = self.unfs[k]
            if unf.nsuf_done >= len(unf.suffixes):
                return None
            cols_new = np.concatenate([unf.my_new_cols(si) for si in range(unf.nsuf_done, len(unf.suffixes))])
            unf.nsuf_done = len(unf.suffixes)
            if cols_new.size == 0:
                return None
            Anew = unf.fetch_block(self.data, unf.rows, cols_new)
            mine = [(m, Anew[unf.lr(im), :]) for m, im in enumerate(unf.piv_rows) if unf.owner_row(im) == self.my_pr]
            return (k, cols_new, Anew, mine)

        def col_apply(item):
            k, cols_new, Anew, _ = item
            unf = self.unfs[k]; l = len(unf.dl)
            Ap = np.empty((l, cols_new.size))
            for dct in allm:
                for m, rowv in dct.get(k, ()):
                    Ap[m, :] = rowv
            Vnew = np.empty((l, cols_new.size))
            for m in range(l):
                Vnew[m, :] = Ap[m, :] - (unf.dl[:m] * unf.Qm[:m, m]) @ Vnew[:m, :]
            unf.cols = np.concatenate((unf.cols, cols_new))
            unf.A.add_cols(Anew)
            unf.V.add_cols(Vnew)

        def row_fetch(k):
            unf = self.unfs[k]
            if unf.npre_done >= len(unf.prefixes):
                return None
            rows_new = np.concatenate([unf.my_new_rows(p) for p in range(unf.npre_done, len(unf.prefixes))])
            unf.npre_done = len(unf.prefixes)
            if rows_new.size == 0:
                return None
            Anew = unf.fetch_block(self.data, rows_new, unf.cols)
            mine = [(m, Anew[:, unf.lc(jm)]) for m, jm in enumerate(unf.piv_cols) if unf.owner_col(jm) == self.my_pc]
            return (k, rows_new, Anew, mine)

        def row_apply(item):
            k, rows_new, Anew, _ = item
            unf = self.unfs[k]; l = len(unf.dl)
            Ap = np.empty((rows_new.size, l))
            for dct in allm:
                for m, colv in dct.get(k, ()):
                    Ap[:, m] = colv
            Unew = np.empty((rows_new.size, l))
            for m in range(l):
                Unew[:, m] = Ap[:, m] - Unew[:, :m] @ (unf.dl[:m] * unf.Pm[:m, m])
            unf.rows = np.concatenate((unf.rows, rows_new))
            unf.A.add_rows(Anew)
            unf.U.add_rows(Unew)

        pend = [x for x in self._map(col_fetch, ks) if x is not None]
        if pend:
            allm = _prof_comm('piv', self.col_comm, lambda: self.col_comm.allgather({x[0]: x[3] for x in pend}))
            self._map(col_apply, pend)
        pend = [x for x in self._map(row_fetch, ks) if x is not None]
        if pend:
            allm = _prof_comm('piv', self.row_comm, lambda: self.row_comm.allgather({x[0]: x[3] for x in pend}))
            self._map(row_apply, pend)

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
            unf.A.set(unf.fetch_block(self.data, unf.rows, unf.cols))
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
            if aval == 0.0 or aval < self.tol * abs(1.0 / self.unfs[k].dl[0]):
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
        rs = list(rs)
        self.rounds = 0
        for _ in range(max(rs) - 1):
            active = [k for k in range(len(rs)) if rs[k] > 1]
            for k in active:
                rs[k] -= 1
            self.round(active)
            self.rounds += 1

    # ---- core construction ----
    def finish_rows_for_core(self, unf):
        """Extend U to the rows still pending from the lazy sync.  Only the
        pivot columns at those rows are needed, not the whole strip; the tile A
        is left as it is (nothing reads it after this point)."""
        if unf.npre_done >= len(unf.prefixes):
            return
        rows_new = np.concatenate([unf.my_new_rows(pidx) for pidx in range(unf.npre_done, len(unf.prefixes))])
        unf.npre_done = len(unf.prefixes)
        if rows_new.size == 0:                   # identical for the whole row group
            return
        l = len(unf.dl)
        mycols = [(m, jm) for m, jm in enumerate(unf.piv_cols) if unf.owner_col(jm) == self.my_pc]
        mine = []
        if mycols:
            Ablk = unf.fetch_block(self.data, rows_new, np.array([jm for _, jm in mycols], dtype=np.int64))
            mine = [(m, Ablk[:, t]) for t, (m, _) in enumerate(mycols)]
        allm = _prof_comm('core', self.row_comm, lambda: self.row_comm.allgather(mine))
        Ap = np.empty((rows_new.size, l))
        for chunk in allm:
            for m, colv in chunk:
                Ap[:, m] = colv
        Unew = np.empty((rows_new.size, l))
        for m in range(l):
            Unew[:, m] = Ap[:, m] - Unew[:, :m] @ (unf.dl[:m] * unf.Pm[:m, m])
        unf.rows = np.concatenate((unf.rows, rows_new))
        unf.U.add_rows(Unew)

    def pivot_matrix(self, unf):
        """U_k(I_k, :): lower triangular; Qm below the diagonal, pivot residuals on it."""
        l = len(unf.dl)
        T = np.zeros((l, l))
        for a in range(l):
            T[a, :a] = unf.Qm[:a, a]
            T[a, a] = 1.0 / unf.dl[a]
        return T

    def build_cores(self):
        """TT cores W_k = U_k U_k(I_k,:)^{-1}, assembled on rank 0 (None elsewhere).

        Every process row holds U_k for its rows, so the solve is local; the
        p_c processes of a row split its rows between them.  The last core is
        A_{d-2}(I_{d-2}, :), the pivot rows of the last tile."""
        d = self.d
        cores = [None] * d
        for k, unf in enumerate(self.unfs):
            self.finish_rows_for_core(unf)
            l = len(unf.dl)
            T = self.pivot_matrix(unf)
            sl = slice(self.my_pc, None, self.pc)
            rows_sl = unf.rows[sl]
            Wsl = np.linalg.solve(T.T, unf.U.a[sl].T).T if rows_sl.size else np.zeros((0, l))
            parts = _prof_comm('core', self.world, lambda: self.world.gather((rows_sl, Wsl), root=0))
            if self.rank == 0:
                W = np.empty((unf.nrows_glob(), l))
                for r_, w_ in parts:
                    W[r_] = w_
                cores[k] = W.reshape(len(unf.prefixes), unf.nk, l)
        unf = self.unfs[-1]
        l = len(unf.dl)
        mine = [(m, unf.cols, unf.A.a[unf.lr(im), :].copy())
                for m, im in enumerate(unf.piv_rows) if unf.owner_row(im) == self.my_pr]
        parts = _prof_comm('core', self.world, lambda: self.world.gather(mine, root=0))
        if self.rank == 0:
            C = np.empty((l, unf.nk1))
            for chunk in parts:
                for m, cols, vals in chunk:
                    C[m, cols] = vals
            cores[d - 1] = C.reshape(l, unf.nk1, 1)
        return cores if self.rank == 0 else None

    # ---- phases ----
    def tile_area(self):
        return sum(len(u.rows) * len(u.cols) for u in self.unfs)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("usage: python MPISubtensorTTCross_Version4_NumPy.py err_samples trials dims ranks [pr,pc] [nb]")
        sys.exit(1)
    err_sample_size = int(float(sys.argv[1]))
    number_trial = int(sys.argv[2])
    dims = list(map(int, sys.argv[3].split(",")))
    _TENSOR_DIMS = list(dims)
    rs = list(map(int, sys.argv[4].split(",")))
    world = MPI.COMM_WORLD
    P = world.Get_size(); world_rank = world.Get_rank()
    if len(sys.argv) > 5:
        pr, pc = map(int, sys.argv[5].split(","))
    else:
        pr = max(q for q in range(1, int(math.isqrt(P)) + 1) if P % q == 0)
        pc = P // pr
    nb = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    _SEED = os.environ.get("TT_SEED")
    _SEED = None if _SEED in (None, "") else int(_SEED)
    _PROFILE_SYNC = os.environ.get("TT_PROFILE") not in (None, "", "0")
    _PROF_MAXRANKS = int(os.environ.get("TT_PROFILE_RANKS", "16"))
    _PROF_OUT = os.environ.get("TT_PROFILE_OUT")
    _TOL = float(os.environ.get("TT_TOL", "0"))
    _THREADS = int(os.environ.get("TT_THREADS", "1"))

    reason = None
    if pr * pc != P:
        reason = "process grid {}x{} needs {} ranks but {} were launched".format(pr, pc, pr * pc, P)
    elif len(rs) != len(dims) - 1:
        reason = "need one TT rank per unfolding ({})".format(len(dims) - 1)
    if reason is not None:
        if world_rank == 0:
            print(reason)
        sys.exit(0)

    data = FunctionalTensor(dims)
    for ll in range(number_trial):
        _prof_reset()
        if world_rank == 0:
            print('starting run', ll)
        world.Barrier()
        t0 = MPI.Wtime()
        cross = PivotOrderCross(dims, pr, pc, nb, data, world, _SEED, _TOL, _THREADS)
        cross.run(rs)
        t1 = MPI.Wtime()
        _PROF['piv']['total'] = t1 - t0

        area = cross.tile_area()
        core_ranks = [1] + [len(u.dl) for u in cross.unfs] + [1]

        # ---- core construction: W_k = U_k U_k(I_k,:)^{-1} from the stored factors ----
        tc0 = MPI.Wtime()
        cores = cross.build_cores()
        cores = _prof_comm('core', world, lambda: world.bcast(cores, root=0))
        err_meas = sampled_error(cores, dims, err_sample_size, world, _SEED)
        _PROF['core']['total'] = MPI.Wtime() - tc0

        if world_rank == 0 and os.environ.get("TT_CHECK_CORES") not in (None, "", "0"):
            # compare with the serial Version 3 routine on the same pivot sets
            I_row_global = [[Large_ravel(ip, dims[:k + 1]) for ip in cross.Ipiv[k]] for k in range(len(dims) - 1)]
            I_col_global = [[Large_ravel(jp, dims[k + 1:]) for jp in cross.Jpiv[k]] for k in range(len(dims) - 1)]
            ref, _ = Low_Access_Core_Extract_Update(tensor_entries_glob, dims, _new_entry_cache(dims), I_row_global, I_col_global)
            dev = [float(np.max(np.abs(c - r)) / max(np.max(np.abs(r)), 1e-300)) for c, r in zip(cores, ref)]
            print('CHECK cores vs serial Version 3 routine: max relative deviation per core', ' '.join('%.1e' % x for x in dev))

        # ---- per-rank report ----
        fetched = sum(u.fetched for u in cross.unfs)
        rows = world.gather((world_rank, _PROF['piv']['total'], _PROF['piv']['comm'], _PROF['piv']['wait'],
                             _PROF['core']['total'], _PROF['core']['comm'], _PROF['core']['wait'], area, fetched), root=0)
        if world_rank == 0:
            prof = [{'rank': r[0], 'piv_total': r[1], 'piv_comm': r[2], 'piv_wait': r[3],
                     'piv_compute': max(r[1] - r[2] - r[3], 0.0),
                     'core_total': r[4], 'core_comm': r[5], 'core_wait': r[6],
                     'core_compute': max(r[4] - r[5] - r[6], 0.0), 'area': r[7], 'fetched': r[8]} for r in rows]
            if _PROF_OUT:
                cols = ['rank', 'piv_total', 'piv_compute', 'piv_comm', 'piv_wait',
                        'core_total', 'core_compute', 'core_comm', 'core_wait']
                with open(_PROF_OUT, 'w') as fh:
                    fh.write(','.join(cols) + '\n')
                    for r in prof:
                        fh.write(','.join(str(r[c]) for c in cols) + '\n')
            comp = np.array([r['piv_compute'] for r in prof]); cm = np.array([r['piv_comm'] for r in prof])
            wt = np.array([r['piv_wait'] for r in prof]); ar = np.array([r['area'] for r in prof], dtype=float)
            wall = max(r['piv_total'] for r in prof)
            print('*' * 65)
            print('Percent of time for full pivot searching   :', np.round(wall, 4), 'seconds')
            print('-' * 65)
            ct = np.array([r['core_total'] for r in prof]); cc = np.array([r['core_compute'] for r in prof])
            print('Percent of time for core construction      :', np.round(ct.max(), 4), 'seconds  (cores + error sampling, all ranks)')
            print('   [per-rank] core compute max={:.4f}s  avg={:.4f}s  comm max={:.4f}s'.format(
                  cc.max(), cc.mean(), max(r['core_comm'] for r in prof)))
            print('-' * 65)
            print('Sampled error                              :', "{0:.3E}".format(np.min(err_meas)))
            print('-' * 65)
            print('Percent of full tensor accessed            :', "{0:.3E}".format(100 * sum(r['fetched'] for r in prof) / math.prod(dims)), '%')
            print('-' * 65)
            print('Tile area per rank (sum over unfoldings)   : max={:.3E}  avg={:.3E}  min={:.3E}  imbalance(max/avg)={:.3f}'.format(
                  ar.max(), ar.mean(), ar.min(), ar.max() / ar.mean() if ar.mean() > 0 else 0.0))
            print('-' * 65)
            print('Per-rank time breakdown over {} ranks  (wait split: {})'.format(P, 'ON' if _PROFILE_SYNC else 'OFF - folded into comm, set TT_PROFILE=1 to split'))
            print('  PIVOT SEARCH : wall={:.4f}s'.format(wall))
            for name, v in (('compute', comp), ('comm', cm), ('wait', wt)):
                if name == 'wait' and not _PROFILE_SYNC:
                    continue
                print('    {:<9} max={:>9.4f}s (rank {:>4d})  min={:>9.4f}s  avg={:>9.4f}s  max/avg={:6.3f}'.format(
                      name, v.max(), int(np.argmax(v)), v.min(), v.mean(), v.max() / v.mean() if v.mean() > 0 else 0.0))
            print('    -> shares of the phase wall: compute {:.1f}%  comm {:.1f}%  wait {:.1f}%'.format(
                  100 * comp.mean() / wall, 100 * cm.mean() / wall, 100 * wt.mean() / wall))
            if P <= _PROF_MAXRANKS:
                print('   rank |  piv_compute   piv_comm   piv_wait |   tile area')
                for r in prof:
                    print('   {:>4d} | {:>12.4f} {:>10.4f} {:>10.4f} | {:>11.3E}'.format(
                          r['rank'], r['piv_compute'], r['piv_comm'], r['piv_wait'], r['area']))
            if _PROF_OUT:
                print('   per-rank CSV written to', _PROF_OUT)
            print('-' * 65)
            print('Core ranks', core_ranks)
            print('-' * 65)
            for k, u in enumerate(cross.unfs):
                print('Unfolding {}: A_k is {} x {} ({} prefixes, {} suffixes); rank 0 tile {} x {}'.format(
                      k, u.nrows_glob(), u.ncols_glob(), len(u.prefixes), len(u.suffixes), len(u.rows), len(u.cols)))
            print('-' * 65)
            print('Layout: pivot-order 2D block-cyclic, process grid {}x{} ({} ranks), block size {}, TT_TOL={:g}'.format(pr, pc, P, nb, _TOL))
            print('Dimension-parallel rounds:', cross.rounds, ' (all active unfoldings searched per round, collectives fused; TT_THREADS={}, OMP_NUM_THREADS={})'.format(_THREADS, os.environ.get('OMP_NUM_THREADS', 'unset')))
            print('Pivot residuals |e_0| -> |e_last| per unfolding:', ' '.join('{:.1e}->{:.1e}'.format(abs(1/u.dl[0]), abs(1/u.dl[-1])) for u in cross.unfs))
            print('*' * 65)
            print('Size', dims)
            print('*' * 65)
