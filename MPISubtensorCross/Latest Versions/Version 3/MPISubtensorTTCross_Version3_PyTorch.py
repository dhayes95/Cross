import numpy as np
import math
import time
import itertools
import random
import decimal
import os
from decimal import Decimal

import torch
from mpi4py import MPI
import sys
decimal.getcontext().prec=10000



####################################################################
################Settings for results in paper#######################
####################################################################
# - 3d Hilbert: uncomment lines 29 and 40
# - 6d Hilbert: uncomment lines 32 and 40
# - 4D Maxwellian uncomment lines 33 and 41
# - 6D Maxwellian uncomment lines 34 and 42


def Local_subtensor_construction(bounds,sub_dim): # did not use

    #Computes the local subtensors:
    # inputs do not need to change, only change the function below in line 13 and 19.
    # subtensor = np.fromfunction(lambda i0,i1,i2: (1)/(1+i0+bounds[0][0]+i1+bounds[1][0]+i2+bounds[2][0]),sub_dim)
    #subtensor = np.fromfunction(lambda i0,i1,i2,i3: (1)/(1+i0+bounds[0][0]+i1+bounds[1][0]+i2+bounds[2][0]+i3+bounds[3][0]),sub_dim)
    #subtensor = np.fromfunction(lambda i0,i1,i2,i3,i4: (1)/(1+i0+bounds[0][0]+i1+bounds[1][0]+i2+bounds[2][0]+i3+bounds[3][0]+i4+bounds[4][0]),sub_dim)
    #subtensor = np.fromfunction(lambda i0,i1,i2,i3,i4,i5: (1)/(1+i0+bounds[0][0]+i1+bounds[1][0]+i2+bounds[2][0]+i3+bounds[3][0]+i4+bounds[4][0]+i5+bounds[5][0]),sub_dim)
    #subtensor = np.fromfunction(lambda i0,i1,i2,i3: Maxwellian2d2v(i0+bounds[0][0],i1+bounds[1][0],i2+bounds[2][0],i3+bounds[3][0],dims),sub_dim)
    #subtensor = np.fromfunction(lambda i0,i1,i2,i3,i4,i5: Maxwellian3d3v(i0+bounds[0][0],i1+bounds[1][0],i2+bounds[2][0],i3+bounds[3][0],i4+bounds[4][0],i5+bounds[5][0],dims),sub_dim)
    
    return subtensor

# ==========================================================================
# Set 'TENSOR_KIND' to choose which tensor to approximate.
# ==========================================================================
TENSOR_KIND = "hilbert"     # CHANGE THIS: "hilbert" | "maxwell4" | "maxwell6"
_TENSOR_DIMS = None          # filled in after argv parsing

# ==========================================================================
# PyTorch CPU configuration.
# ==========================================================================
TORCH_DTYPE = torch.float64
TORCH_DEVICE = torch.device("cpu")


def _configure_torch_device():
    """Configure PyTorch for CPU execution."""
    thread_count = os.environ.get("TT_TORCH_NUM_THREADS")
    if thread_count is not None:
        torch.set_num_threads(max(1, int(thread_count)))
    return torch.device("cpu")


def _torch_sync():
    """No device synchronization"""
    return None

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


def Unfolding(X_ten,n): # did not use
    """
    Parameters
    ----------
    X_ten : Tensor to unfold
    n : dimension to unfold along

    Returns
    -------
    X_mat : Unfolding Matrix

    """
    dim = X_ten.shape
    
    new_dim = [math.prod(dim[:n+1]),math.prod(dim[n+1::])]
    
    X_mat = np.reshape(X_ten,new_dim)
    
    return X_mat

def Subtensor_Split(X,partitions):
    """

    Parameters
    ----------
    X : Tensor
    partitions : Number of partitions in either all dimensions, or list of number per dimension

    Returns
    -------
    Subtensor : List of subtensors
    index_shifts : list of corresponding index shifts for each subtensor compared to whole tensor

    """

        
    dim = X.shape
    Subtensor = []
        
    selections = [list(Large_unravel(i,partitions)) for i in range(math.prod(partitions))]
    single_indexing = [np.array_split(np.arange(dim[i]),partitions[i]) for i in range(len(partitions))]
    
    indexing = [[single_indexing[i][selections[j][i]] for i in range(len(partitions))] for j in range(len(selections))]
    
    Bounds = [[[indexing[j][i][0],indexing[j][i][-1]+1] for i in range(len(partitions))] for j in range(len(selections))]
       
    Shift_index = [[indexing[j][i][0] for i in range(len(partitions))] for j in range(len(selections))]
    
    Subtensor = np.array_split(X,partitions[0],axis = 0)
    
    for i in range(1,len(partitions)):
        Subtensor = [m for b in Subtensor for m in np.array_split(b,partitions[i],axis = i)]
    
    return Subtensor, Shift_index, Bounds

def Subtensor_Comm(selection,shifting,k):
    """
    

    Parameters
    ----------
    X : Tensor
    partitions : Number of partitions in either all dimensions, or list of number per dimension

    Returns
    -------
    Subtensor : List of subtensors
    index_shifts : list of corresponding index shifts for each subtensor compared to whole tensor

        

    """
    
    rows = []
    cols = []
    selected_shift = shifting[selection]
    
    for i in range(len(shifting)):
        if shifting[i][:k+1]==selected_shift[:k+1]:
            rows.append(i)
            
        if shifting[i][k+1:]==selected_shift[k+1:]:
            cols.append(i)
    
    
    return rows,cols

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

# ---- dict-key encoding -------------------------------------------
# X_dict is keyed by a per-coordinate BYTES string instead of a Python tuple.

def _pivot_values(I_arr, j_fixed, I_l, J_u, k, X_dim, X_dict, bounds, is_col=True):
    """fills only cache misses (decoding bytes -> coords), returns values array."""
    I_arr = np.asarray(I_arr, dtype=np.int64)
    if is_col:
        coords = Super_to_Global_coords(I_arr, np.full(I_arr.shape, j_fixed, dtype=np.int64), I_l, J_u, k, X_dim)
    else:
        coords = Super_to_Global_coords(np.full(I_arr.shape, j_fixed, dtype=np.int64), I_arr, I_l, J_u, k, X_dim)
    return _cache_fetch(X_dict, coords, bounds)

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
# --------------------------------------------------------------------------

# ---- vectorised entry cache ----------------------------------------------
# The bytes-keyed dict costs ~150 bytes and one Python-level operation per
# cached entry, which made cache lookups, not tensor evaluation, the dominant
# cost of superblock construction.  EntryCache holds the same entries in an
# open-addressing hash table keyed by the block-local linear index, so lookups
# and inserts are whole-array NumPy operations at ~32 bytes per entry.
# Blocks too large for one int64 index (common in 6D/8D) use a two-part key:
# the linear index of the leading and of the trailing dimensions.
# Semantics are unchanged: a cached value is never recomputed, and misses are
# evaluated in one batch in first-occurrence order, exactly as before, so the
# results are bit-identical.  The dict API used by the remaining scalar call
# sites (bytes keys) is kept.

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


def _cache_fetch(X_dict, coords, bounds):
    """Cached values of block-local entries coords (m,d), evaluating misses in one batch."""
    if isinstance(X_dict, EntryCache):
        return X_dict.fetch(coords, lambda c: tensor_entries(tensor_entry, c, bounds))
    return _cached_entries_from_keys(tensor_entries, _keys_from_rows(coords), bounds, X_dict, np.shape(coords)[1])
# --------------------------------------------------------------------------

def Full_To_Super(ind_row,ind_col,k,dim):
    Super_ind_row = []
    Super_ind_col = []
    
    row_dim = dim[:k+1]
    col_dim = dim[k+1::]
    
    for i in range(len(ind_row)):
        Super_ind_row.append(Large_unravel(ind_row[i],row_dim)[-1])
        Super_ind_col.append(Large_unravel(ind_col[i],col_dim)[0])
    
    
    
    return Super_ind_row,Super_ind_col

def Sub_to_Unfold(sub_index,subdim,fulldim,shifting,k):
    
    full_sub = Large_unravel(Large_ravel(sub_index,[math.prod(subdim[:k+1]),math.prod(subdim[k+1:])]),subdim)
    
    full_full = [full_sub[i]+shifting[i] for i in range(len(fulldim))]
    
    full_index = Large_unravel(Large_ravel(full_full,fulldim),[math.prod(fulldim[:k+1]),math.prod(fulldim[k+1:])])
    
    return full_index


def MPICrossInterpSingleItemSuperPiv(A,Atilde,Ir,Ic,FIr,FIc,k,Xsize): # did not use
    flag =0
    
    if k==0:
        #Ii,Jj,p = Greedy_Pivot_Search_Piv(A,Ir[0],Ic[0])
        Ii,Jj,p,ps = MPIGreedy_Search_Piv(A, Atilde, Ir[0], Ic[0])
        if Ii[-1] not in Ir[0] or Jj[-1] not in Ic[0]:
            Ir[0] = np.append(Ir[0],Ii[-1]).astype(int)
            Ic[0] = np.append(Ic[0],Jj[-1]).astype(int)
            #print(Ir[0],Ic[0])
            if len(Ic[1])>0:
                fi,fj = Matrix_Super_Index_Conversion(Ir[0][-1], Ic[0][-1], [0], FIc[1], 0, Xsize)
            else:
                fi,fj = Matrix_Super_Index_Conversion(Ir[0][-1], Ic[0][-1], [0], [0], 0, Xsize)
            FIr[0] = np.append(FIr[0],fi).astype(int)
            FIc[0] = np.append(FIc[0],fj).astype(int)
        else:
            flag = 1
            
        
    elif k==len(Xsize)-2:
        #Ii,Jj,p = Greedy_Pivot_Search_Piv(A,Ir[k],Ic[k]) 
        Ii,Jj,p,ps = MPIGreedy_Search_Piv(A, Atilde, Ir[k], Ic[k])
        if Ii[-1] not in Ir[k] or Jj[-1] not in Ic[k]:
            Ir[k] = np.append(Ir[k],Ii[-1]).astype(int)
            Ic[k] = np.append(Ic[k],Jj[-1]).astype(int)
            if len(Ir[-2])>0:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], FIr[-2], [0], k, Xsize)
            else:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], [0], [0], k, Xsize)
            FIr[k] = np.append(FIr[k],fi).astype(int)
            FIc[k] = np.append(FIc[k],fj).astype(int)
        else:
            flag = 1
        
        
    else:
        #Ii,Jj,p = Greedy_Pivot_Search_Piv(A,Ir[k],Ic[k])
        Ii,Jj,p,ps = MPIGreedy_Search_Piv(A, Atilde, Ir[k], Ic[k])
        if Ii[-1] not in Ir[k] or Jj[-1] not in Ic[k]:
            Ir[k] = np.append(Ir[k],Ii[-1]).astype(int)
            Ic[k] = np.append(Ic[k],Jj[-1]).astype(int)
            #print(Ir[k][-1], Ic[k][-1], FIr[k-1], FIc[k+1])
            if len(Ir[k-1])>0 and len(Ic[k+1])>0:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], FIr[k-1], FIc[k+1], k, Xsize)
            elif len(Ir[k-1])>0 and len(Ic[k+1])==0:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], FIr[k-1], [0], k, Xsize)
            elif len(Ir[k-1])==0 and len(Ic[k+1])>0:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], [0], FIc[k+1], k, Xsize)
            else:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], [0], [0], k, Xsize)
            #print(Ir[k][-1],Ic[k][-1],Ir[k-1],Ic[k+1],k,X.shape,fi,fj)
            FIr[k] = np.append(FIr[k],fi).astype(int)
            FIc[k] = np.append(FIc[k],fj).astype(int)
        else:
            flag = 1
    
    
    #if flag==1:
    #    print("Duplicate occurred. Specified rank may not be obtained")
        
    
        
    return Ir,Ic,FIr,FIc,p,ps

def MPIGreedy_Search_Piv(A,Atilde,I,J,sample_size=4,maxiter=10,tol=0): # did not use
    """
    Input:
        A - Matrix 
        I - Row indices
        J - Column indices
        sample_size - Number of random samples
        maxiter - maximum iteration count 
        tol - tolerance for CUR recursive construction

    Output:
        I_new - Enriched row indices
        J_new - Enriched column indices
    
    """
    dim = A.shape
    

    #Check if I and J are empty
    #This if-else statement takes care of empty indices, and if non-empty then 
    #only search along unslected rows and columns
    if len(I)==dim[0] and len(J)==dim[1]:
        I_new = I
        J_new = J
        p = 0
        p_sign = 1
    else:
        if len(I)==dim[0] and len(J)<dim[1]:
            J_out = np.delete(np.arange(dim[1]),J)
            #ind_i,ind_j = Large_unravel(np.argmax(np.abs((A - Atilde)[:,J_out])),[dim[0],len(J_out)])
            ind_j = J_out[0]
            j_star = J_out[ind_j]
            I_new = I
            J_new = np.append(J,j_star)
            p = 0
            #p = (A - Atilde)[ind_i,ind_j]
            p_sign = 1


        elif len(J)==dim[1] and len(I)<dim[0]:
            I_out = np.delete(np.arange(dim[0]),I)
            #ind_i,ind_j  =Large_unravel(np.argmax(np.abs((A - Atilde)[I_out,:])),[len(I_out),dim[1]])
            ind_i = I_out[0]
            i_star = I_out[ind_i]
            I_new = np.append(I,i_star)
            J_new = J
            #p = (A - Atilde)[ind_i,ind_j]
            p = 0
            p_sign = 1


        else:
            #Set up counter for iterations    
            iteration_count = 0
            
            Asub = np.delete(np.delete(A,I,0),J,1)
            Atildesub = np.delete(np.delete(Atilde,I,0),J,1)
            
            Row_track = np.delete(np.arange(dim[0]),I)
            Col_track = np.delete(np.arange(dim[1]),J)
            
            subdim = Asub.shape
            
            sample_number = np.min([sample_size,dim[0]-len(I),dim[1]-len(J)])
            
            Irandom = random.sample(list(np.arange(subdim[0])),sample_number)
            Jrandom = random.sample(list(np.arange(subdim[1])),sample_number)

            #Initial pivot in skeleton error
            E = np.abs(Asub-Atildesub)
            i_star,j_star = Large_unravel(np.argmax(E[Irandom,:][:,Jrandom]),[len(Irandom),len(Jrandom)])

            #Main while loop
            while iteration_count < maxiter:
        
                i_star = np.argmax(E[:,j_star])
                j_star = np.argmax(E[i_star,:])
        
                #Check rook condition
                if all([E[i_star,j_star]>E[m,j_star] for m in range(subdim[0])]) \
                    and all([E[i_star,j_star]>E[i_star,n] for n in range(subdim[1])]):
                    break
                iteration_count+=1

            if E[i_star,j_star]>tol:
                I_new = np.append(I,Row_track[i_star]).astype(int)
                J_new = np.append(J,Col_track[j_star]).astype(int)
                p = E[i_star,j_star]
                p_sign = np.sign(Asub[i_star,j_star] - Atildesub[i_star,j_star])
            else:
                i_star,j_star = Large_unravel(np.argmax(E),E.shape)
                if E[i_star,j_star]>tol:
                    I_new = np.append(I,Row_track[i_star]).astype(int)
                    J_new = np.append(J,Col_track[j_star]).astype(int)
                    p = E[i_star,j_star]
                    p_sign = np.sign(Asub[i_star,j_star] - Atildesub[i_star,j_star])
                else:
                    I_new = I
                    J_new = J
                    p=0
                    p_sign = 1
            
    return I_new,J_new,p,p_sign


def Low_access_MPICrossInterpSingleItemSuperPiv(tensor_entries,Ej,Ei,Delt,bounds,X_dict,Ir,Ic,FIr,FIc,k,Xsize): # did not use
    flag =0
    
    if k==0:
        #Ii,Jj,p = Greedy_Pivot_Search_Piv(A,Ir[0],Ic[0])
        Ii,Jj,p,ps,X_dict = Low_Access_MPIGreedy_Search_Piv(tensor_entries,Ej,Ei,Delt, Ir[k], Ic[k],[0],FIc[1],k,Xsize,bounds,X_dict)
        if Ii[-1] not in Ir[0] or Jj[-1] not in Ic[0]:
            Ir[0] = np.append(Ir[0],Ii[-1]).astype(int)
            Ic[0] = np.append(Ic[0],Jj[-1]).astype(int)
            #print(Ir[0],Ic[0])
            if len(Ic[1])>0:
                fi,fj = Matrix_Super_Index_Conversion(Ir[0][-1], Ic[0][-1], [0], FIc[1], 0, Xsize)
            else:
                fi,fj = Matrix_Super_Index_Conversion(Ir[0][-1], Ic[0][-1], [0], [0], 0, Xsize)
            FIr[0] = np.append(FIr[0],fi).astype(int)
            FIc[0] = np.append(FIc[0],fj).astype(int)
        else:
            flag = 1
            
        
    elif k==len(Xsize)-2:
        #Ii,Jj,p = Greedy_Pivot_Search_Piv(A,Ir[k],Ic[k]) 
        Ii,Jj,p,ps,X_dict = Low_Access_MPIGreedy_Search_Piv(tensor_entries,Ej,Ei,Delt, Ir[k], Ic[k],FIr[-2],[0],k,Xsize,bounds,X_dict)
        if Ii[-1] not in Ir[k] or Jj[-1] not in Ic[k]:
            Ir[k] = np.append(Ir[k],Ii[-1]).astype(int)
            Ic[k] = np.append(Ic[k],Jj[-1]).astype(int)
            if len(Ir[-2])>0:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], FIr[-2], [0], k, Xsize)
            else:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], [0], [0], k, Xsize)
            FIr[k] = np.append(FIr[k],fi).astype(int)
            FIc[k] = np.append(FIc[k],fj).astype(int)
        else:
            flag = 1
        
        
    else:
        #Ii,Jj,p = Greedy_Pivot_Search_Piv(A,Ir[k],Ic[k])
        Ii,Jj,p,ps,X_dict = Low_Access_MPIGreedy_Search_Piv(tensor_entries,Ej,Ei,Delt, Ir[k], Ic[k],FIr[k-1],FIc[k+1],k,Xsize,bounds,X_dict)
        if Ii[-1] not in Ir[k] or Jj[-1] not in Ic[k]:
            Ir[k] = np.append(Ir[k],Ii[-1]).astype(int)
            Ic[k] = np.append(Ic[k],Jj[-1]).astype(int)
            #print(Ir[k][-1], Ic[k][-1], FIr[k-1], FIc[k+1])
            if len(Ir[k-1])>0 and len(Ic[k+1])>0:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], FIr[k-1], FIc[k+1], k, Xsize)
            elif len(Ir[k-1])>0 and len(Ic[k+1])==0:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], FIr[k-1], [0], k, Xsize)
            elif len(Ir[k-1])==0 and len(Ic[k+1])>0:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], [0], FIc[k+1], k, Xsize)
            else:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], [0], [0], k, Xsize)
            #print(Ir[k][-1],Ic[k][-1],Ir[k-1],Ic[k+1],k,X.shape,fi,fj)
            FIr[k] = np.append(FIr[k],fi).astype(int)
            FIc[k] = np.append(FIc[k],fj).astype(int)
        else:
            flag = 1
    
    
    #if flag==1:
    #    print("Duplicate occurred. Specified rank may not be obtained")
        
    
        
    return Ir,Ic,FIr,FIc,p,ps,X_dict


def Low_access_MPICrossInterpSingleItemSuperPivV2(tensor_entries,At,bounds,X_dict,Ir,Ic,FIr,FIc,k,Xsize):
    flag =0
    
    if k==0:
        #Ii,Jj,p = Greedy_Pivot_Search_Piv(A,Ir[0],Ic[0])
        Ii,Jj,p,ps,X_dict = Low_Access_MPIGreedy_Search_PivV2(tensor_entries,At, Ir[k], Ic[k],[0],FIc[1],k,Xsize,bounds,X_dict)
        if Ii[-1] not in Ir[0] and Jj[-1] not in Ic[0]:
            Ir[0] = np.append(Ir[0],Ii[-1]).astype(int)
            Ic[0] = np.append(Ic[0],Jj[-1]).astype(int)
            #print(Ir[0],Ic[0])
            if len(Ic[1])>0:
                fi,fj = Matrix_Super_Index_Conversion(Ir[0][-1], Ic[0][-1], [0], FIc[1], 0, Xsize)
            else:
                fi,fj = Matrix_Super_Index_Conversion(Ir[0][-1], Ic[0][-1], [0], [0], 0, Xsize)
            FIr[0] = np.append(FIr[0],fi).astype(int)
            FIc[0] = np.append(FIc[0],fj).astype(int)
        else:
            flag = 1
            
        
    elif k==len(Xsize)-2:
        #Ii,Jj,p = Greedy_Pivot_Search_Piv(A,Ir[k],Ic[k]) 
        Ii,Jj,p,ps,X_dict = Low_Access_MPIGreedy_Search_PivV2(tensor_entries,At, Ir[k], Ic[k],FIr[-2],[0],k,Xsize,bounds,X_dict)
        if Ii[-1] not in Ir[k] and Jj[-1] not in Ic[k]:
            Ir[k] = np.append(Ir[k],Ii[-1]).astype(int)
            Ic[k] = np.append(Ic[k],Jj[-1]).astype(int)
            if len(Ir[-2])>0:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], FIr[-2], [0], k, Xsize)
            else:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], [0], [0], k, Xsize)
            FIr[k] = np.append(FIr[k],fi).astype(int)
            FIc[k] = np.append(FIc[k],fj).astype(int)
        else:
            flag = 1
        
        
    else:
        #Ii,Jj,p = Greedy_Pivot_Search_Piv(A,Ir[k],Ic[k])
        Ii,Jj,p,ps,X_dict = Low_Access_MPIGreedy_Search_PivV2(tensor_entries,At, Ir[k], Ic[k],FIr[k-1],FIc[k+1],k,Xsize,bounds,X_dict)
        if Ii[-1] not in Ir[k] and Jj[-1] not in Ic[k]:
            Ir[k] = np.append(Ir[k],Ii[-1]).astype(int)
            Ic[k] = np.append(Ic[k],Jj[-1]).astype(int)
            #print(Ir[k][-1], Ic[k][-1], FIr[k-1], FIc[k+1])
            if len(Ir[k-1])>0 and len(Ic[k+1])>0:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], FIr[k-1], FIc[k+1], k, Xsize)
            elif len(Ir[k-1])>0 and len(Ic[k+1])==0:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], FIr[k-1], [0], k, Xsize)
            elif len(Ir[k-1])==0 and len(Ic[k+1])>0:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], [0], FIc[k+1], k, Xsize)
            else:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], [0], [0], k, Xsize)
            #print(Ir[k][-1],Ic[k][-1],Ir[k-1],Ic[k+1],k,X.shape,fi,fj)
            FIr[k] = np.append(FIr[k],fi).astype(int)
            FIc[k] = np.append(FIc[k],fj).astype(int)
        else:
            flag = 1
    
    
    #if flag==1:
    #    print("Duplicate occurred. Specified rank may not be obtained")
        
    
        
    return Ir,Ic,FIr,FIc,p,ps,X_dict

def Low_Access_MPIGreedy_Search_Piv(tensor_entries,Ej,Ei,Delt,I,J,I_l,J_u,k,X_dim,bounds,X_dict,sample_size=10,maxiter=10,tol=0): # did not use
    """
    Input:
        A - Matrix 
        I - Row indices
        J - Column indices
        sample_size - Number of random samples
        maxiter - maximum iteration count 
        tol - tolerance for CUR recursive construction

    Output:
        I_new - Enriched row indices
        J_new - Enriched column indices
    
    """
    dim = [Ej.shape[0],Ei.shape[1]]
    

    #Check if I and J are empty
    #This if-else statement takes care of empty indices, and if non-empty then 
    #only search along unslected rows and columns
    if len(I)==dim[0] and len(J)==dim[1]:
        I_new = I
        J_new = J
        p = 0
        p_sign = 1
    else:
        if len(I)==dim[0] and len(J)<dim[1]:
            J_out = np.delete(np.arange(dim[1]),J)
            #ind_i,ind_j = Large_unravel(np.argmax(np.abs((A - Atilde)[:,J_out])),[dim[0],len(J_out)])
            ind_j = J_out[0]
            j_star = J_out[ind_j]
            I_new = I
            J_new = np.append(J,j_star)
            #p = (A - Atilde)[ind_i,ind_j]
            p = 0
            p_sign = 1


        elif len(J)==dim[1] and len(I)<dim[0]:
            I_out = np.delete(np.arange(dim[0]),I)
            #ind_i,ind_j  =Large_unravel(np.argmax(np.abs((A - Atilde)[I_out,:])),[len(I_out),dim[1]])
            ind_i = I_out[0]
            i_star = I_out[ind_i]
            I_new = np.append(I,i_star)
            J_new = J
            #p = (A - Atilde)[ind_i,ind_j]
            p = 0
            p_sign = 1


        else:
            #Set up counter for iterations    
            iteration_count = 0
            
            #Asub = np.delete(np.delete(A,I,0),J,1)
            #Atildesub = np.delete(np.delete(Atilde,I,0),J,1)
            
            I_avail = np.delete(np.arange(dim[0]),I)
            J_avail = np.delete(np.arange(dim[1]),J)
            
            #subdim = Asub.shape
            
            sample_number = np.min([sample_size,dim[0]-len(I),dim[1]-len(J)])
            
            a = random.sample(list(np.arange(Ej.shape[0])),sample_number)
            b = random.sample(list(np.arange(Ei.shape[1])),sample_number)

            rand_inds = [[a[i],b[i]] for i in range(sample_number)]
            rand_as_glob = [Super_to_Global(rand_inds[i],I_l,J_u,k,X_dim) for i in range(sample_number)]

            true_vals = []
            for indi in rand_as_glob:
                if indi not in X_dict:
                    X_dict[indi] = tensor_entries(tensor_entry,[_coord_from_key(indi,len(X_dim))],bounds)[0]
                true_vals.append(X_dict[indi])
            
            approx_vals = [(Ej[[rand_inds[i][0]],:]@np.diag(Delt)@Ei[:,[rand_inds[i][1]]])[0][0] for i in range(sample_number)]
                
            
            #Initial pivot in skeleton error
            #E = np.abs(Asub-Atildesub)
            E = np.abs(np.asarray(true_vals) - np.asarray(approx_vals))
            
            i_star,j_star = rand_inds[np.argmax(E)]

            c_inds = [Super_to_Global([i,j_star],I_l,J_u,k,X_dim) for i in I_avail]
            true_vals = []
            for indi in c_inds:
                if indi not in X_dict:
                    X_dict[indi] = tensor_entries(tensor_entry,[_coord_from_key(indi,len(X_dim))],bounds)[0]
                true_vals.append(X_dict[indi])
            approx_vals = Ej[I_avail,:]@np.diag(Delt)@Ei[:,j_star]
            
            err_1 = np.asarray(true_vals) - np.asarray(approx_vals)


            #Main while loop
            while iteration_count < maxiter:
        
                sel_i = np.argmax(np.abs(err_1))
                i_star = I_avail[np.argmax(np.abs(err_1))]

                r_inds = [Super_to_Global([i_star,j],I_l,J_u,k,X_dim) for j in J_avail]
                true_vals = []
                for indi in r_inds:
                    if indi not in X_dict:
                        X_dict[indi] = tensor_entries(tensor_entry,[_coord_from_key(indi,len(X_dim))],bounds)[0]
                    true_vals.append(X_dict[indi])
                approx_vals = Ej[i_star,:]@np.diag(Delt)@Ei[:,J_avail]
                err_2 = np.asarray(true_vals) - np.asarray(approx_vals)
                
                sel_j = np.argmax(np.abs(err_2))
                j_star = J_avail[np.argmax(np.abs(err_2))]

                c_inds = [Super_to_Global([i,j_star],I_l,J_u,k,X_dim) for i in I_avail]
                true_vals = []
                for indi in c_inds:
                    if indi not in X_dict:
                        X_dict[indi] = tensor_entries(tensor_entry,[_coord_from_key(indi,len(X_dim))],bounds)[0]
                    true_vals.append(X_dict[indi])
                approx_vals = Ej[I_avail,:]@np.diag(Delt)@Ei[:,j_star]
                err_1 = np.asarray(true_vals) - np.asarray(approx_vals)

                #print(['err_1',err_1,len(I_avail),'err_2',err_2,len(J_avail)])
                #Check rook condition
                if np.all(np.abs(err_1[sel_i]) >= np.abs(err_1)) and np.all(np.abs(err_2[sel_j]) >= np.abs(err_2)):
                    
                    break
                iteration_count+=1

            if np.abs(err_1[sel_i])>tol:
                I_new = np.append(I,i_star).astype(int)
                J_new = np.append(J,j_star).astype(int)
                p = np.abs(err_1[sel_i])
                p_sign = np.sign(err_1[sel_i])
            else:
                I_new = I
                J_new = J
                p=0
                p_sign = 1
        
    return I_new,J_new,p,p_sign,X_dict


def Low_Access_MPIGreedy_Search_PivV2(tensor_entries,At,I,J,I_l,J_u,k,X_dim,bounds,X_dict,sample_size=10,maxiter=10,tol=0):
    """
    Input:
        A - Matrix 
        I - Row indices
        J - Column indices
        sample_size - Number of random samples
        maxiter - maximum iteration count 
        tol - tolerance for CUR recursive construction

    Output:
        I_new - Enriched row indices
        J_new - Enriched column indices
    
    """
    dim = At.shape
    

    #Check if I and J are empty
    #This if-else statement takes care of empty indices, and if non-empty then 
    #only search along unslected rows and columns
    if len(I)==dim[0] and len(J)==dim[1]:
        I_new = I
        J_new = J
        p = 0
        p_sign = 1
    else:
        if len(I)==dim[0] and len(J)<dim[1]:
            J_out = np.delete(np.arange(dim[1]),J)
            #ind_i,ind_j = Large_unravel(np.argmax(np.abs((A - Atilde)[:,J_out])),[dim[0],len(J_out)])
            ind_j = J_out[0]
            j_star = J_out[ind_j]
            I_new = I
            J_new = np.append(J,j_star)
            #p = (A - Atilde)[ind_i,ind_j]
            p = 0
            p_sign = 1


        elif len(J)==dim[1] and len(I)<dim[0]:
            I_out = np.delete(np.arange(dim[0]),I)
            #ind_i,ind_j  =Large_unravel(np.argmax(np.abs((A - Atilde)[I_out,:])),[len(I_out),dim[1]])
            ind_i = I_out[0]
            i_star = I_out[ind_i]
            I_new = np.append(I,i_star)
            J_new = J
            #p = (A - Atilde)[ind_i,ind_j]
            p = 0
            p_sign = 1


        else:
            #Set up counter for iterations    
            iteration_count = 0
            
            #Asub = np.delete(np.delete(A,I,0),J,1)
            #Atildesub = np.delete(np.delete(Atilde,I,0),J,1)
            
            I_avail = np.delete(np.arange(dim[0]),I)
            J_avail = np.delete(np.arange(dim[1]),J)
            
            #subdim = Asub.shape
            
            sample_number = np.min([sample_size,dim[0]-len(I),dim[1]-len(J)])
            
            #range() draws the same sample as the materialised list did
            a = random.sample(range(At.shape[0]),sample_number)
            b = random.sample(range(At.shape[1]),sample_number)

            rand_inds = [[a[i],b[i]] for i in range(sample_number)]
            _rc = Super_to_Global_coords(np.asarray(a,dtype=np.int64), np.asarray(b,dtype=np.int64), I_l, J_u, k, X_dim)
            true_vals = _cache_fetch(X_dict, _rc, bounds)
            
            approx_vals = At[np.asarray(a,dtype=np.int64), np.asarray(b,dtype=np.int64)]
                
            
            #Initial pivot in skeleton error
            #E = np.abs(Asub-Atildesub)
            E = np.abs(np.asarray(true_vals) - np.asarray(approx_vals))
            
            i_star,j_star = rand_inds[np.argmax(E)]

            true_vals = _pivot_values(I_avail, j_star, I_l, J_u, k, X_dim, X_dict, bounds, is_col=True)
            approx_vals = At[I_avail,j_star]
            
            err_1 = np.asarray(true_vals) - np.asarray(approx_vals)


            #Main while loop
            while iteration_count < maxiter:
        
                sel_i = np.argmax(np.abs(err_1))
                i_star = I_avail[np.argmax(np.abs(err_1))]

                true_vals = _pivot_values(J_avail, i_star, I_l, J_u, k, X_dim, X_dict, bounds, is_col=False)
                approx_vals = At[i_star,J_avail]
                err_2 = np.asarray(true_vals) - np.asarray(approx_vals)
                
                sel_j = np.argmax(np.abs(err_2))
                j_star = J_avail[np.argmax(np.abs(err_2))]

                true_vals = _pivot_values(I_avail, j_star, I_l, J_u, k, X_dim, X_dict, bounds, is_col=True)
                approx_vals = At[I_avail,j_star]
                err_1 = np.asarray(true_vals) - np.asarray(approx_vals)

                #print(['err_1',err_1,len(I_avail),'err_2',err_2,len(J_avail)])
                #Check rook condition
                if np.all(np.abs(err_1[sel_i]) >= np.abs(err_1)) and np.all(np.abs(err_2[sel_j]) >= np.abs(err_2)):
                    
                    break
                iteration_count+=1

            if np.abs(err_1[sel_i])>tol:
                I_new = np.append(I,i_star).astype(int)
                J_new = np.append(J,j_star).astype(int)
                p = np.abs(err_1[sel_i])
                p_sign = np.sign(err_1[sel_i])
            else:
                #I_new = I
                #J_new = J
                I_out = np.delete(np.arange(dim[0]),I)
                #ind_i,ind_j  =Large_unravel(np.argmax(np.abs((A - Atilde)[I_out,:])),[len(I_out),dim[1]])
                ind_i = I_out[0]
                #i_star = I_out[ind_i]
                I_new = np.append(I,i_star)
                J_out = np.delete(np.arange(dim[1]),J)
                #ind_i,ind_j = Large_unravel(np.argmax(np.abs((A - Atilde)[:,J_out])),[dim[0],len(J_out)])
                ind_j = J_out[0]
                #j_star = J_out[ind_j]
                #I_new = I
                J_new = np.append(J,j_star)
                p=0
                p_sign = 1
        
    return I_new,J_new,p,p_sign,X_dict


def Matrix_Super_Index_Conversion(ind_row,ind_col,I,J,k,dim):
    dim_matrix = [dim[k]*len(I),dim[k+1]*len(J)]
    index_shift = np.reshape(np.reshape(np.arange(dim_matrix[1]),[dim[k+1],len(J)]).T,[1,dim[k+1]*len(J)])
    
    fix_i,fix_j = Super_to_Unfold_Index(ind_row, index_shift[0][ind_col], I, J, k, dim)
    
    return int(fix_i),int(fix_j)


def Super_to_Unfold_Index(ind_row,ind_col,I,J,k,dim):
    r"""
    

    Parameters
    ----------
    ind_row : row index in superblock
    ind_col : column index in superblock
    I : row index set of lower dimension I_\leq{k-1}
    J : column index set of higher dimension I_>{k+1}
    k : dimension of first full index in superblock X(I_\leq{k-1}i_k,i_{k+1}I_>{k+1})
    dim : Dimension of full tensor

    Returns
    -------
    Row_index : Adjusted row index for full k-unfolding of X
    Col_index : Adjusted column index for full k-unfolding of X

    """
    
    if k==0:
        Row_index = ind_row
        Col_index = J[ind_col%len(J)] + math.floor(ind_col/len(J))*math.prod(dim[k+2::])
    elif k==len(dim)-2:
        Row_index = dim[k]*I[math.floor(ind_row/dim[k])] + ind_row%dim[k]
        Col_index = ind_col
    else:
        Row_index = dim[k]*I[math.floor(ind_row/dim[k])] + ind_row%dim[k]
        Col_index = int(J[ind_col%len(J)]) + math.floor(ind_col/len(J))*math.prod(dim[k+2::])
    
    return Row_index, Col_index


def Subtensor_Processor_Build(partitions,dim):
    selections = [list(Large_unravel(i,partitions)) for i in range(math.prod(partitions))]
    single_indexing = [np.array_split(np.arange(dim[i]),partitions[i]) for i in range(len(partitions))]
    
    indexing = [[single_indexing[i][selections[j][i]] for i in range(len(partitions))] for j in range(len(selections))]
    
    Bounds = [[[indexing[j][i][0],indexing[j][i][-1]+1] for i in range(len(partitions))] for j in range(len(selections))]
    
    #Subtensor = [X[np.ix_(*indexing[i])] for i in range(len(selections))]
       
    Shift_index = [[indexing[j][i][0] for i in range(len(partitions))] for j in range(len(selections))]
    return Shift_index,Bounds

def MPI_Tk_Index(full_index,shifting,bounds,k,tensor_dim): # did not use
    
    unfold_dim = [math.prod(tensor_dim[:k+1]),math.prod(tensor_dim[k+1:])]
    
    tensor_index = Large_unravel(Large_ravel(full_index,unfold_dim), tensor_dim)
    rank_index_search = [[(tensor_index[i]-shifting[j][i])//(bounds[j][i][1]-bounds[j][i][0]) for i in range(len(tensor_dim))] for j in range(len(shifting))]
    
    rank_index = np.argmin(sum(np.abs(rank_index_search),axis = 1))
    
    rank_dim = [bounds[rank_index][i][1]-bounds[rank_index][i][0] for i in range(len(tensor_dim))]
    sub_unfold_dim = [math.prod(rank_dim[:k+1]),math.prod(rank_dim[k+1:])]
    
    row_index,col_index = Large_unravel(Large_ravel([tensor_index[i] - shifting[rank_index][i] for i in range(len(tensor_dim))],rank_dim),sub_unfold_dim)
    row,col = Subtensor_Comm(rank_index, shifting, k)
    
    row_loc = [row_index,row]
    col_loc = [col_index,col]
    
    return rank_index,row_loc,col_loc


def MPI_Recursive_Tk(col,I,tol = 0): # did not use 
    Tk = np.reshape(col[:,0]*(1/col[I[0],0]),[col.shape[0],1])
    
    for i in range(1,len(I)):
        Sk = np.reshape(Tk@col[I[:i],i],[col.shape[0],1]) - col[:,[i]]
        
        deltainv = Sk[I[i],0]
        if np.abs(deltainv)>tol:
            delta = 1/deltainv
        else:
            delta = 0
        Tk = np.concatenate((Tk - delta*Sk@Tk[[I[i]],:],delta*Sk),axis = 1)
    
    return Tk

def Core_to_Tensor_Value(cores,index): # does not get used
    
    start = cores[0][:,index[0],:]
    for i in range(1,len(cores)):
        start = start@cores[i][:,index[i],:]
    value = start[0][0]
    return value

def Super_to_Global(super_index,I,J,k,dim):
    
    new_dim = [math.prod(dim[:k+1]),math.prod(dim[k+1::])]
    
    intermed = Large_ravel(Matrix_Super_Index_Conversion(super_index[0], super_index[1], I, J, k, dim),new_dim)
        
    global_index = Large_unravel(intermed,dim)
    return _key(global_index)

def Super_to_Global_batch(r, c, I, J, k, dim):
    # Same as Super_to_Global_coords, returned as bytes keys.
    return _keys_from_rows(Super_to_Global_coords(r, c, I, J, k, dim))

def Super_to_Global_coords(r, c, I, J, k, dim):
    # Vectorized batched form of Super_to_Global: maps arrays of superblock
    d = len(dim)
    lenJ = len(J)
    r = np.asarray(r, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    NC = dim[k+1]*lenJ
    perm = np.arange(NC).reshape(dim[k+1], lenJ).T.reshape(-1)   # the index_shift column permutation
    cc = perm[c]
    Iarr = np.asarray(I, dtype=np.int64)
    Jarr = np.asarray(J, dtype=np.int64)
    if k == 0:
        fix_i = r.copy()
    else:
        fix_i = dim[k]*Iarr[r // dim[k]] + (r % dim[k])
    if k == d-2:
        fix_j = cc.copy()
    else:
        tail = math.prod(dim[k+2:])
        fix_j = Jarr[cc % lenJ] + (cc // lenJ) * tail
 
    g = np.empty((fix_i.shape[0], d), dtype=np.int64)
    if k == d - 2:
        last = dim[d-1]
        hi = fix_i + fix_j // last    
        lo = fix_j % last
    else:
        hi = fix_i
        lo = fix_j
    rem = hi
    for i in range(k + 1):
        stride = math.prod(dim[i+1:k+1])   
        q = rem // stride
        g[:, i] = q
        rem = rem - q * stride
    rem = lo
    for i in range(k + 1, d):
        stride = math.prod(dim[i+1:])      
        q = rem // stride
        g[:, i] = q
        rem = rem - q * stride
    return g

def better_kickstart(tensor_entries,bounds,X_dict,X_dim,maxiter = 10,sample_size = 10):

    index = [tuple([random.randint(0,X_dim[j]-1) for j in range(len(X_dim))]) for _ in range(sample_size)]
    values = tensor_entries(tensor_entry,index,bounds)
    if isinstance(X_dict, EntryCache):
        X_dict.put_missing(index, values)
    else:
        for i in range(len(index)):
            _kk = _key(index[i])
            if _kk not in X_dict:
                X_dict[_kk] = values[i]
    start_index = index[np.argmax(np.abs(values))]

    count = 0
      
    #Each fiber is fetched as one batch; the entry functions are element-wise,
    #so this gives the same values as evaluating the entries one at a time.
    for _ in range(10):
        for i in range(len(X_dim)):
            index = np.tile(np.asarray(start_index, dtype=np.int64), (X_dim[i], 1))
            index[:, i] = np.arange(X_dim[i])
            values = _cache_fetch(X_dict, index, bounds)
            start_index = tuple(int(x) for x in index[np.argmax(np.abs(values))])
        
        


    return start_index,np.max(np.abs(values)),X_dict

def Lower_access_Superblock(tensor_entries,bounds,I_inp,J_inp,row_list,col_list,k,X_dict,X_dim): # did not use

    if k==0:
        I = [0]
        J = J_inp
    elif k==len(X_dim)-2:
        I = I_inp
        J = [0]
    else:
        I = I_inp
        J = J_inp

    #Build indices
    index_set = []
    for i0 in I:
        
        low = Large_unravel(i0,X_dim[:k])
        for i1 in range(X_dim[k]):
            for i2 in range(X_dim[k+1]):
                for i3 in J:
                    high = Large_unravel(i3,X_dim[k+2:])
                    index = _key(tuple(low)+tuple([i1,i2])+tuple(high))
                    
                    index_set.append(index)
        #print(index)
    tensor_values = []
    for indi in index_set:
        if indi not in X_dict:
            X_dict[indi] = tensor_entries(tensor_entry,[_coord_from_key(indi,len(X_dim))],bounds)[0]
        tensor_values.append(X_dict[indi])
    Superblock = np.reshape(np.transpose(np.reshape(tensor_values,[len(I),X_dim[k],X_dim[k+1],len(J)]),[0,1,3,2]),[X_dim[k]*len(I),len(J)*X_dim[k+1]])
    
    Super_rows = Superblock[row_list,:]
    Super_cols = Superblock[:,col_list]

    return Super_rows,Super_cols,X_dict

def _cached_entries_from_keys(tensor_entries, keys, bounds, X_dict, d):
    """Return cached entries and batch-evaluate misses. Keys are BYTES"""
    keys = list(keys)
    missing_keys = list(dict.fromkeys(key for key in keys if key not in X_dict))

    if missing_keys:
        missing_indices = np.frombuffer(b''.join(missing_keys), dtype=np.int64).reshape(-1, d)
        missing_values = np.asarray(
            tensor_entries(tensor_entry, missing_indices, bounds)
        ).reshape(-1)

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

_LASB = {'index': 0.0, 'member': 0.0, 'eval': 0.0, 'assemble': 0.0}   # finer split inside entry-pulling

# ==========================================================================
# Per-rank compute / communication / wait accounting
# ==========================================================================
# Every collective is split in two.  A Barrier goes first and its cost is
# charged to 'wait': that is time this rank sat idle because some other rank
# had not reached the collective yet, which is exactly the load imbalance.
# The collective itself follows and is charged to 'comm', the actual transfer.
# Whatever is left of the phase is local work, reported as 'compute'.
#
# The extra Barriers are not free, so the split only happens when TT_PROFILE is
# set.  With it unset the Barriers are skipped and idle time is folded into
# 'comm', which is what a normal timing run wants.
_PROFILE_SYNC = False
_PROF = {'piv':  {'comm': 0.0, 'wait': 0.0, 'total': 0.0},
         'core': {'comm': 0.0, 'wait': 0.0, 'total': 0.0}}


def _prof_reset():
    for _ph in _PROF.values():
        for _k in _ph:
            _ph[_k] = 0.0


def _prof_comm(phase, comm, fn):
    """Run one collective, separating idle time from transfer time."""
    if _PROFILE_SYNC:
        _t0 = MPI.Wtime()
        comm.Barrier()
        _t1 = MPI.Wtime()
        _PROF[phase]['wait'] += _t1 - _t0
    else:
        _t1 = MPI.Wtime()
    _r = fn()
    _PROF[phase]['comm'] += MPI.Wtime() - _t1
    return _r


def _prof_wait(phase, comm):
    """A Barrier that exists in its own right: pure idle time."""
    _t0 = MPI.Wtime()
    comm.Barrier()
    _PROF[phase]['wait'] += MPI.Wtime() - _t0


def _prof_stats(rows, key):
    """max / min / mean of one column, plus the rank holding the max."""
    vals = [r[key] for r in rows]
    hi = max(vals)
    return hi, min(vals), sum(vals) / len(vals), rows[vals.index(hi)]['rank']


def _prof_line(label, rows, key):
    hi, lo, avg, who = _prof_stats(rows, key)
    imb = (hi / avg) if avg > 0 else 0.0
    return '    {:<9} max={:8.4f}s (rank {:>4})  min={:8.4f}s  avg={:8.4f}s  max/avg={:6.3f}'.format(
        label, hi, who, lo, avg, imb)
def Lower_access_Superblock_test(tensor_entries,bounds,I_inp,J_inp,row_list,col_list,k,X_dict,X_dim):
    """Build selected superblock rows/columns with batched index generation.
    """
    if k==0:
        I = [0]
        J = J_inp
    elif k==len(X_dim)-2:
        I = I_inp
        J = [0]
    else:
        I = I_inp
        J = J_inp

    # Selected superblock rows: build every requested cell position in a batch.
    _t0 = MPI.Wtime()
    _NC = len(J)*X_dim[k+1]
    _rr = np.repeat(np.asarray(row_list,dtype=np.int64), _NC)
    _cc = np.tile(np.arange(_NC,dtype=np.int64), len(row_list))
    col_inds = Super_to_Global_coords(_rr,_cc,I,J,k,X_dim)
    _LASB['index'] += MPI.Wtime() - _t0

    _t0 = MPI.Wtime()
    tensor_values = _cache_fetch(X_dict, col_inds, bounds)
    _LASB['eval'] += MPI.Wtime() - _t0

    _t0 = MPI.Wtime()
    Super_rows = np.reshape(
        tensor_values,
        [len(row_list),len(J)*X_dim[k+1]],
    )
    _LASB['assemble'] += MPI.Wtime() - _t0

    # Selected superblock columns: build every requested cell position in a batch.
    _t0 = MPI.Wtime()
    _NR = len(I)*X_dim[k]
    _rr = np.repeat(np.arange(_NR,dtype=np.int64), len(col_list))
    _cc = np.tile(np.asarray(col_list,dtype=np.int64), _NR)
    row_inds = Super_to_Global_coords(_rr,_cc,I,J,k,X_dim)
    _LASB['index'] += MPI.Wtime() - _t0

    _t0 = MPI.Wtime()
    tensor_values = _cache_fetch(X_dict, row_inds, bounds)
    _LASB['eval'] += MPI.Wtime() - _t0

    _t0 = MPI.Wtime()
    Super_cols = np.reshape(
        tensor_values,
        [len(I)*X_dim[k],len(col_list)],
    )
    _LASB['assemble'] += MPI.Wtime() - _t0

    return Super_rows,Super_cols,X_dict


def Lower_access_Superblock_row(tensor_entries,bounds,I_inp,J_inp,row_list,active_J,k,X_dict,X_dim): # did not use
    
    if k==0:
        I = [0]
        J = J_inp
    elif k==len(X_dim)-2:
        I = I_inp
        J = [0]
    else:
        I = I_inp
        J = J_inp
    
    tensor_values = []
    col_inds = []
    for i0 in row_list:
        for i1 in range(np.min(active_J)*X_dim[k+1],len(J)*X_dim[k+1]):
            col_inds.append(Super_to_Global([i0,i1],I,J,k,X_dim))
    for indi in col_inds:
        if indi not in X_dict:
            X_dict[indi] = tensor_entries(tensor_entry,[_coord_from_key(indi,len(X_dim))],bounds)[0]
        tensor_values.append(X_dict[indi])
    Super_rows = np.reshape(tensor_values,[len(row_list),len(J)*X_dim[k+1] - np.min(active_J)*X_dim[k+1]])

    

    return Super_rows,X_dict

def Lower_access_Superblock_col(tensor_entries,bounds,I_inp,J_inp,col_list,active_I,k,X_dict,X_dim): # did not use
    
    if k==0:
        I = [0]
        J = J_inp
    elif k==len(X_dim)-2:
        I = I_inp
        J = [0]
    else:
        I = I_inp
        J = J_inp

    tensor_values = []
    row_inds = []
    for i0 in range(np.min(active_I)*X_dim[k],len(I)*X_dim[k]):
        for i1 in col_list:
            row_inds.append(Super_to_Global([i0,i1],I,J,k,X_dim))
    for indi in row_inds:
        if indi not in X_dict:
            X_dict[indi] = tensor_entries(tensor_entry,[_coord_from_key(indi,len(X_dim))],bounds)[0]
        tensor_values.append(X_dict[indi])
    Super_cols = np.reshape(tensor_values,[len(I)*X_dim[k] - np.min(active_I)*X_dim[k],len(col_list)])

    return Super_cols,X_dict

def Lower_access_Superblock_subblock(tensor_entries,bounds,I_inp,J_inp,k,X_dict,X_dim):
    if k==0:
        I = [0]
        J = J_inp
    elif k==len(X_dim)-2:
        I = I_inp
        J = [0]
    else:
        I = I_inp
        J = J_inp

    _NR = len(I)*X_dim[k]
    _NC = len(J)*X_dim[k+1]

    _rr = np.repeat(np.arange(_NR,dtype=np.int64), _NC)
    _cc = np.tile(np.arange(_NC,dtype=np.int64), _NR)
    cell_indices = Super_to_Global_coords(_rr,_cc,I,J,k,X_dim)

    tensor_values = _cache_fetch(X_dict, cell_indices, bounds)
    Super_block = np.reshape(tensor_values,[_NR,_NC])

    return Super_block,X_dict



def Low_access_Superblock(tensor_entries,bounds,I_inp,J_inp,k,X_dict,X_dim): # did not use

    if k==0:
        I = [0]
        J = J_inp
    elif k==len(X_dim)-2:
        I = I_inp
        J = [0]
    else:
        I = I_inp
        J = J_inp

    #Build indices
    index_set = []
    for i0 in I:
        
        low = Large_unravel(i0,X_dim[:k])
        for i1 in range(X_dim[k]):
            for i2 in range(X_dim[k+1]):
                for i3 in J:
                    high = Large_unravel(i3,X_dim[k+2:])
                    index = _key(tuple(low)+tuple([i1,i2])+tuple(high))
                    
                    index_set.append(index)
        #print(index)
    tensor_values = []
    for indi in index_set:
        if indi not in X_dict:
            X_dict[indi] = tensor_entries(tensor_entry,[_coord_from_key(indi,len(X_dim))],bounds)[0]
        tensor_values.append(X_dict[indi])
    Superblock = np.reshape(np.transpose(np.reshape(tensor_values,[len(I),X_dim[k],X_dim[k+1],len(J)]),[0,1,3,2]),[X_dim[k]*len(I),len(J)*X_dim[k+1]])

    return Superblock,X_dict

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


def Core_to_Tensor_Values_Torch(torch_cores, indices):
    """Evaluate many TT entries together with batched PyTorch contractions.

    Parameters
    ----------
    torch_cores : list[torch.Tensor]
        TT cores with shapes (r_{k-1}, n_k, r_k), already on one device.
    indices : array-like, shape (batch_size, number_of_modes)
        Tensor multi-indices to evaluate.
    """
    if len(torch_cores) == 0:
        raise ValueError("torch_cores must contain at least one TT core")

    device = torch_cores[0].device
    index_tensor = torch.as_tensor(
        indices, dtype=torch.long, device=device
    )
    if index_tensor.ndim == 1:
        index_tensor = index_tensor.reshape(1, -1)
    if index_tensor.ndim != 2 or index_tensor.shape[1] != len(torch_cores):
        raise ValueError(
            "indices must have shape (batch_size, number_of_cores)"
        )

    # Selecting mode indices gives (r_left, batch, r_right).  
    state = torch_cores[0][:, index_tensor[:, 0], :].permute(1, 0, 2)

    for mode in range(1, len(torch_cores)):
        selected_core = torch_cores[mode][
            :, index_tensor[:, mode], :
        ].permute(1, 0, 2)
        state = torch.bmm(state, selected_core)

    return state[:, 0, 0]



# ==========================================================================
# 2D block-cyclic layout
# ==========================================================================
# The tensor is cut into a Cartesian grid of `block_grid` subtensors (blocks).
# Blocks are handed to MPI processes arranged on a Cartesian `proc_grid` using
# the standard cyclic rule, applied independently in every dimension:
#
#     block  (b_0, ..., b_{d-1})  ->  process  (b_0 % p_0, ..., b_{d-1} % p_{d-1})
#
# For the k-th unfolding this is exactly a 2D block-cyclic distribution: the
# unfolding's row blocks are indexed by (b_0..b_k) and its column blocks by
# (b_{k+1}..b_{d-1}), so the process grid it sees is
# prod(p[:k+1]) x prod(p[k+1:]) and consecutive block rows/columns land on
# different process rows/columns.  Setting block_grid == proc_grid gives one
# block per process, i.e. the original layout.

def Block_Cyclic_Build(block_grid, proc_grid, dim, mapping='cyclic'):
    """Cut `dim` into `block_grid` blocks and map them onto `proc_grid`.

    Parameters
    ----------
    block_grid : number of blocks per dimension
    proc_grid  : number of MPI processes per dimension (prod == active ranks)
    dim        : full tensor dimensions
    mapping    : 'cyclic' (default) deals blocks out round robin in every
                 dimension; 'block' hands each process one contiguous chunk of
                 blocks per dimension.  'block' exists so the two layouts can be
                 compared at an identical block grid - it reproduces the coarse
                 behaviour of the original code at a finer partition.

    Returns
    -------
    Shift_index : per block, the index shift of its first entry
    Bounds      : per block, [lo,hi) per dimension
    owner       : per block, the rank that owns it
    rank_blocks : per rank, the ascending list of blocks it owns
    local_pos   : per block, its position inside its owner's rank_blocks list
    coords      : per block, its coordinates in the block grid
    """
    d = len(block_grid)
    n_blocks = math.prod(block_grid)
    n_procs = math.prod(proc_grid)

    coords = [list(Large_unravel(b, block_grid)) for b in range(n_blocks)]
    single_indexing = [np.array_split(np.arange(dim[i]), block_grid[i]) for i in range(d)]
    indexing = [[single_indexing[i][coords[b][i]] for i in range(d)] for b in range(n_blocks)]

    Bounds = [[[int(indexing[b][i][0]), int(indexing[b][i][-1]) + 1] for i in range(d)]
              for b in range(n_blocks)]
    Shift_index = [[int(indexing[b][i][0]) for i in range(d)] for b in range(n_blocks)]

    if mapping == 'block':
        chunk = [-(-block_grid[i] // proc_grid[i]) for i in range(d)]
        owner = [Large_ravel([min(coords[b][i] // chunk[i], proc_grid[i] - 1) for i in range(d)],
                             proc_grid) for b in range(n_blocks)]
    else:
        owner = [Large_ravel([coords[b][i] % proc_grid[i] for i in range(d)], proc_grid)
                 for b in range(n_blocks)]

    rank_blocks = [[] for _ in range(n_procs)]
    for b in range(n_blocks):
        rank_blocks[owner[b]].append(b)

    local_pos = [0] * n_blocks
    for r in range(n_procs):
        for pos, b in enumerate(rank_blocks[r]):
            local_pos[b] = pos

    return Shift_index, Bounds, owner, rank_blocks, local_pos, coords


def Unfolding_Group_Ids(coords, block_grid):
    """Row/column group id of every block, for every unfolding k.

    Two blocks share a row group for unfolding k when their first k+1 block
    coordinates agree (they cover the same rows of the unfolding), and a column
    group when their last d-k-1 coordinates agree.  This replaces the O(#blocks)
    scan in Subtensor_Comm with an O(1) integer comparison.
    """
    d = len(block_grid)
    ROW_GID = []
    COL_GID = []
    for k in range(d - 1):
        ROW_GID.append([Large_ravel(bc[:k + 1], block_grid[:k + 1]) for bc in coords])
        COL_GID.append([Large_ravel(bc[k + 1:], block_grid[k + 1:]) for bc in coords])
    return ROW_GID, COL_GID


def Block_Cyclic_Report(block_grid, proc_grid, rank_blocks, mapping='cyclic'):
    """Human readable summary of how many blocks each process received."""
    counts = [len(x) for x in rank_blocks]
    return {
        'mapping': mapping,
        'block_grid': list(block_grid),
        'proc_grid': list(proc_grid),
        'n_blocks': math.prod(block_grid),
        'n_procs': math.prod(proc_grid),
        'blocks_per_proc_min': int(np.min(counts)),
        'blocks_per_proc_max': int(np.max(counts)),
        'blocks_per_proc_avg': float(np.mean(counts)),
    }

if __name__ == "__main__":
    

    t1_track = 0
    t2_track = 0
    t3_track = 0


    core_times_list = []
    pivot_times_list = []

    #Collect the inputs
    dims = list(map(int,sys.argv[3].split(",")))     
    _TENSOR_DIMS = list(dims)  
    partitions = [1 for _ in range(len(dims))]      
    err_sample_size = int(float(sys.argv[1]))        
    number_trial = int(sys.argv[2])                  

    #Input 5: the BLOCK grid - how many subtensors the tensor is cut into.
    block_grid = list(map(int,sys.argv[5].split(",")))

    #Input 6 (optional): the MPI PROCESS grid.  Blocks are dealt out to this
    #grid cyclically in every dimension, so each unfolding sees a 2D
    #block-cyclic layout.  Leaving it off makes the process grid equal to the
    #block grid, i.e. one block per process, which is the original behaviour.
    if len(sys.argv)>6:
        proc_grid = list(map(int,sys.argv[6].split(",")))
    else:
        proc_grid = list(block_grid)

    #Optional deterministic seeding, so two layouts can be compared exactly.
    _SEED = os.environ.get("TT_SEED")
    #Block -> process mapping: 'cyclic' (default) or 'block'.
    _MAP = os.environ.get("TT_MAP","cyclic")
    #Separate wait from communication by adding a Barrier before every
    #collective.  Costs extra synchronisation, so it is opt in.
    _PROFILE_SYNC = os.environ.get("TT_PROFILE") not in (None,"","0")
    #Per-rank table is printed for small runs; raise or lower with TT_PROFILE_RANKS.
    _PROF_MAXRANKS = int(os.environ.get("TT_PROFILE_RANKS","16"))
    #Optional CSV with one row per rank, for large runs.
    _PROF_OUT = os.environ.get("TT_PROFILE_OUT")

    _world_size_probe = MPI.COMM_WORLD.Get_size()

    reason = None
    if len(block_grid)!=len(dims) or len(proc_grid)!=len(dims):
        reason = 'Block grid and process grid must have one entry per tensor dimension'
    elif np.min([dims[i]//block_grid[i] for i in range(len(dims))])<2:
        reason = 'Not valid partition, reduce grid size'
    elif np.min([block_grid[i]-proc_grid[i] for i in range(len(dims))])<0:
        reason = 'Block grid must be at least the process grid in every dimension'
    elif math.prod(proc_grid)>_world_size_probe:
        reason = 'Process grid needs {} ranks but only {} were launched'.format(
                 math.prod(proc_grid),_world_size_probe)

    if reason is not None:
        if MPI.COMM_WORLD.Get_rank()==0:
            print(reason)
        pp_f = []
    else:
        pp_f = [proc_grid]

    if len(pp_f)==0:
        pass
    
    else:
        for processor_partition in pp_f:
            
            rs = list(map(int,sys.argv[4].split(",")))  
            world_comm = MPI.COMM_WORLD
            world_rank = MPI.COMM_WORLD.Get_rank()
            world_size = MPI.COMM_WORLD.Get_size()  

            TORCH_DEVICE = _configure_torch_device()
            if world_rank == 0:
                print(
                    "PyTorch numerical kernels use CPU. Device:",
                    TORCH_DEVICE,
                    "dtype:",
                    TORCH_DTYPE,
                )

            if _SEED is not None:
                random.seed(int(_SEED)+world_rank)

            
            if world_rank < math.prod(processor_partition):
                color = 10
                key = world_rank
            else:
                color = 20
                key = world_rank

            comm = MPI.COMM_WORLD.Split(color,key)
            rank = comm.Get_rank()
            size = comm.Get_size()

            #Cut the tensor into block_grid subtensors and deal them out
            #cyclically over processor_partition.
            (shifting_p, bounds_p, block_owner, rank_blocks,
             block_local_pos, block_coords) = Block_Cyclic_Build(block_grid,processor_partition,dims,_MAP)
            n_blocks = len(shifting_p)
            ROW_GID, COL_GID = Unfolding_Group_Ids(block_coords,block_grid)
            layout_report = Block_Cyclic_Report(block_grid,processor_partition,rank_blocks,_MAP)

            #Row and column communicators for every unfolding, set up once.
            #Blocks that share a row group of unfolding k (same first k+1 block
            #coordinates) live on processes with the same first k+1 grid
            #coordinates, because both mappings are per-dimension products; so
            #one Split per unfolding gives the row (and column) groups of the
            #paper as communicators.  The cross-vector exchange then stays
            #inside the group instead of going to every rank.
            if color==10:
                _pcoord = Large_unravel(rank,processor_partition)
                row_comms = [comm.Split(Large_ravel(_pcoord[:k+1],processor_partition[:k+1]),rank)
                             for k in range(len(dims)-1)]
                col_comms = [comm.Split(Large_ravel(_pcoord[k+1:],processor_partition[k+1:]),rank)
                             for k in range(len(dims)-1)]

            if color==10:
                #sweeps is used to track the number of ranks for each dimension connection
                sweeps = []
                for i in range(np.max(rs)):
                    for j in range(len(rs)):
                        if rs[j]>1:
                            sweeps.append(j)
                            rs[j]-=1

                send_data = None
                
                #Every process now owns a LIST of subtensors instead of one.
                my_blocks = rank_blocks[rank]
                nloc = len(my_blocks)
                sub_dim = [[bounds_p[b][i][1]-bounds_p[b][i][0] for i in range(len(dims))]
                           for b in my_blocks]
            piv_times = []
            core_times = []
            # --- superblock-size report defaults (used on rank 0 at print time) ---
            sb_avg_peak = 0.0
            sb_max_peak = 0
            sb_avg_total = 0.0
            sb_busiest_shape = (0, 0)
            sb_nactive = 0
            sb_nactive_blocks = 0
            sb_imbalance = 0.0
            sb_total_min = 0.0
            sb_total_max = 0.0
            for ll in range(number_trial):
                _prof_reset()
                if world_rank==0:
                    print('starting run',ll)
                if color==10:    
                    X_dict = [_new_entry_cache(sub_dim[bl]) for bl in range(nloc)]
                    IndiI_dict = {}
                    deltas = []
                    search_time = 0
                    avg_piv_times = 0
                    avg_core_times = 0
                    #Set up necessary memory for indices
                    start_time = MPI.Wtime()
                    if rank==0:
                        
                        I_row_global = [[] for i in range(len(dims)-1)]
                        I_col_global = [[] for i in range(len(dims)-1)]
                        FI_row_global = [[] for i in range(len(dims)-1)]
                        FI_col_global = [[] for i in range(len(dims)-1)]
                    #Per-block index sets: one entry per LOCAL block.
                    I_row_local = [[[] for _ in range(len(dims)-1)] for _ in range(nloc)]
                    I_col_local = [[[] for _ in range(len(dims)-1)] for _ in range(nloc)]
                    FI_row_local = [[[] for _ in range(len(dims)-1)] for _ in range(nloc)]
                    FI_col_local = [[[] for _ in range(len(dims)-1)] for _ in range(nloc)]

                    Selection_row_local = [[[0] for _ in range(len(dims)-1)] for _ in range(nloc)]
                    Selection_col_local = [[[0] for _ in range(len(dims)-1)] for _ in range(nloc)]
                    
                    #With TT_SEED set, every BLOCK gets its own random stream keyed
                    #by its global block id.  The pivots a block finds then depend
                    #only on the block grid, never on which process happens to own
                    #it, which is what makes two process grids directly comparable.
                    if _SEED is not None:
                        rng_state = [random.Random(int(_SEED)+my_blocks[bl]+100003*ll).getstate()
                                     for bl in range(nloc)]
                    else:
                        rng_state = None

                    p1 = [[] for _ in range(nloc)]
                    p =[[] for _ in range(nloc)]
                    piv_vals = [[] for _ in range(nloc)]
                    local_piv_value = []
                    local_piv_loc = []
                    
                    #------------------------------------------------------------------------#
                    ############################ INITIAL SEARCH ##############################
                    ################ Can switch initial search from argmax ###################
                    ####### Most efficient way would be a random sample from unfolding #######
                    #------------------------------------------------------------------------#
                    
                    for bl in range(nloc):
                        b = my_blocks[bl]
                        sd = sub_dim[bl]
                        for i in range(len(dims)-1):
                            if i==0:
                                if rng_state is not None:
                                    _saved = random.getstate(); random.setstate(rng_state[bl])
                                full_start,vals,X_dict[bl] = better_kickstart(tensor_entries,bounds_p[b],X_dict[bl],sd)
                                if rng_state is not None:
                                    rng_state[bl] = random.getstate(); random.setstate(_saved)
                                i_new,j_new = Large_unravel(Large_ravel(full_start,sd),[math.prod(sd[:i+1]),math.prod(sd[i+1:])])
                            
                            else:
                                i_new,j_new = Large_unravel(Large_ravel([FI_row_local[bl][0][0],FI_col_local[bl][0][0]],[sd[0],math.prod(sd[1:])]),[math.prod(sd[:i+1]),math.prod(sd[i+1:])])
                                vals = piv_vals[bl][0]
                            piv_vals[bl].append(vals)
                            FI_row_local[bl][i].append(i_new)
                            FI_col_local[bl][i].append(j_new)
                            I_row_local[bl][i] = Full_To_Super(FI_row_local[bl][i],FI_col_local[bl][i],i,sd)[0]
                            I_col_local[bl][i] = Full_To_Super(FI_row_local[bl][i],FI_col_local[bl][i],i,sd)[1]
                
                    
                    #Pivot selection is now a reduction over BLOCKS, not over ranks.
                    _gathered = _prof_comm('piv',comm,lambda: comm.allgather([(my_blocks[bl],piv_vals[bl]) for bl in range(nloc)]))
                    local_pivot_value = [None]*n_blocks
                    for _chunk in _gathered:
                        for _b,_v in _chunk:
                            local_pivot_value[_b] = _v
                    
                    selection = [[int(np.argmax(np.abs([local_pivot_value[i][k] for i in range(n_blocks)])))] for k in range(len(dims)-1)]
                    
                    #Communication to correct index sets for influenced blocks
                    for k in range(len(dims)-1):
                        s = selection[k][0]

                        info_index = None
                        if rank==block_owner[s]:
                            sb = block_local_pos[s]
                            info_index = [I_row_local[sb][k][0],I_col_local[sb][k][0],FI_row_local[sb][k][0],FI_col_local[sb][k][0],sub_dim[sb]]
                        info_index = _prof_comm('piv',comm,lambda: comm.bcast(info_index,root = block_owner[s]))

                        for bl in range(nloc):
                            b = my_blocks[bl]
                            if b==s:
                                continue
                            in_row = (ROW_GID[k][b]==ROW_GID[k][s])
                            in_col = (COL_GID[k][b]==COL_GID[k][s])
                            if not in_row and len(I_row_local[bl][k])>0:
                                I_row_local[bl][k] = np.delete(I_row_local[bl][k],[-1])
                                FI_row_local[bl][k] = np.delete(FI_row_local[bl][k],[-1])
                                Selection_row_local[bl][k] = np.delete(Selection_row_local[bl][k],[-1])
                            elif in_row:
                                I_row_local[bl][k][-1] = info_index[0]
                                FI_row_local[bl][k][-1] = info_index[2]
                            if not in_col and len(I_col_local[bl][k])>0:
                                I_col_local[bl][k] = np.delete(I_col_local[bl][k],[-1])
                                FI_col_local[bl][k] = np.delete(FI_col_local[bl][k],[-1])
                                Selection_col_local[bl][k] = np.delete(Selection_col_local[bl][k],[-1])
                            elif in_col:
                                I_col_local[bl][k][-1] = info_index[1]
                                FI_col_local[bl][k][-1] = info_index[3]

                        if rank==0:
                            
                            adj_ind = Sub_to_Unfold([info_index[2],info_index[3]],info_index[4],dims,shifting_p[s],k)
                            I_row_global[k].append(adj_ind[0])
                            I_col_global[k].append(adj_ind[1])
                    
                    t1 = MPI.Wtime()
                    t1_track= t1 - start_time

                    update_time = 0
                    internal_tt = 0
                    #Per-block superblock caches.
                    old_rows = [[None for _ in range(len(dims)-1)] for _ in range(nloc)]
                    old_cols = [[None for _ in range(len(dims)-1)] for _ in range(nloc)]
                    sup_store = [[[] for _ in range(len(dims)-1)] for _ in range(nloc)]
                    counter = [1 for _ in range(len(dims)-1)]
                    tracker = 0
                    # --- per-process superblock-size instrumentation (summed over local blocks) ---
                    superblock_peak_area = 0      # largest local superblock this process searched
                    superblock_total_area = 0     # cumulative area searched (work proxy)
                    superblock_peak_shape = (0, 0)
                    superblock_build_count = 0
                    # --- log len(I), len(J) for ALL unfoldings, every sweep, every rank ---
                 
                    VERBOSE_SWEEP_LOG = False
                    track_sweep = [0]*(len(dims)-1)
                    for j in sweeps:

                        ##################### DETERMINE THE ACTIVE BLOCKS #####################
                        ############## LOCAL SUPERBLOCKS AND APPROXIMATIONS ARE FORMED/ALLOCATED #############
                        ut1 = MPI.Wtime()
                        super_rows = [None]*nloc
                        super_cols = [None]*nloc
                        suptilde = [None]*nloc
                        # The full local cross approximation of every block stays
                        # on one PyTorch device; MPI still ships NumPy arrays, so
                        # only the communicated rows/columns move back to CPU.
                        super_rows_t = [None]*nloc
                        super_cols_t = [None]*nloc
                        suptilde_t = [None]*nloc
                        my_active = []
                        my_search = []

                        for bl in range(nloc):
                            b = my_blocks[bl]
                            sd = sub_dim[bl]
                            active = 0
                            search = 0
                            if j==0:
                                if len(I_col_local[bl][1])>0:
                                    active = 1
                                    if sd[j]>len(I_row_local[bl][j]) and len(I_col_local[bl][j+1])*sd[j+1]>len(I_col_local[bl][j]):
                                        search = 1
                            elif j==len(dims)-2:
                                if len(I_row_local[bl][j-1])>0:
                                    active = 1
                                    if len(I_row_local[bl][j-1])*sd[j]>len(I_row_local[bl][j]) and sd[j+1]>len(I_col_local[bl][j]):
                                        search = 1
                            else:
                                if len(I_row_local[bl][j-1])>0 and len(I_col_local[bl][j+1])>0:
                                    active=1  
                                    if len(I_row_local[bl][j-1])*sd[j]>len(I_row_local[bl][j]) and len(I_col_local[bl][j+1])*sd[j+1]>len(I_col_local[bl][j]):
                                        search = 1

                            if active==0:
                                continue
                            my_active.append(b)
                            if search==1:
                                my_search.append(b)

                            if j==0:
                                if old_cols[bl][j] is None:
                                    c1p,X_dict[bl] = Lower_access_Superblock_subblock(tensor_entries,bounds_p[b],[0],FI_col_local[bl][1],0,X_dict[bl],sd)
                                    sup_store[bl][j] = c1p
                                else:
                                    col_meas = len(FI_col_local[bl][j+1]) - old_cols[bl][j]
                                    if col_meas>0:
                                        c2p,X_dict[bl] = Lower_access_Superblock_subblock(tensor_entries,bounds_p[b],[0],FI_col_local[bl][1][-col_meas:],0,X_dict[bl],sd)
                                        c1p = np.concatenate((sup_store[bl][j],c2p),axis = 1)
                                        sup_store[bl][j] = c1p
                                    else:
                                        c1p = sup_store[bl][j]
                                old_cols[bl][j]= len(FI_col_local[bl][j+1])
                            elif j==len(dims)-2:
                                if old_rows[bl][j] is None:
                                    c1p,X_dict[bl] = Lower_access_Superblock_subblock(tensor_entries,bounds_p[b],FI_row_local[bl][j-1],[0],j,X_dict[bl],sd)
                                    sup_store[bl][j] = c1p
                                else:
                                    row_meas = len(FI_row_local[bl][j-1]) - old_rows[bl][j]
                                    if row_meas>0:
                                        c2p,X_dict[bl] = Lower_access_Superblock_subblock(tensor_entries,bounds_p[b],FI_row_local[bl][j-1][-row_meas:],[0],j,X_dict[bl],sd)
                                        c1p = np.concatenate((sup_store[bl][j],c2p),axis = 0)
                                        sup_store[bl][j] = c1p
                                    else:
                                        c1p = sup_store[bl][j]
                                old_rows[bl][j]= len(FI_row_local[bl][j-1])
                            else:
                                if old_rows[bl][j] is None and old_cols[bl][j] is None:
                                    c1p,X_dict[bl] = Lower_access_Superblock_subblock(tensor_entries,bounds_p[b],FI_row_local[bl][j-1],FI_col_local[bl][j+1],j,X_dict[bl],sd)
                                    sup_store[bl][j] = c1p
                                else:
                                    row_meas = len(FI_row_local[bl][j-1]) - old_rows[bl][j]
                                    col_meas = len(FI_col_local[bl][j+1]) - old_cols[bl][j]
                                    if col_meas>0 and row_meas>0:
                                        c2p,X_dict[bl] = Lower_access_Superblock_subblock(tensor_entries,bounds_p[b],FI_row_local[bl][j-1][-row_meas:],FI_col_local[bl][j+1][:-col_meas],j,X_dict[bl],sd)
                                        c3p,X_dict[bl] = Lower_access_Superblock_subblock(tensor_entries,bounds_p[b],FI_row_local[bl][j-1],FI_col_local[bl][j+1][-col_meas:],j,X_dict[bl],sd)
                                        c1p = np.concatenate((np.concatenate((sup_store[bl][j],c2p),axis = 0),c3p),axis = 1)
                                        sup_store[bl][j] = c1p
                                    elif col_meas>0 and row_meas==0:
                                        c2p,X_dict[bl] = Lower_access_Superblock_subblock(tensor_entries,bounds_p[b],FI_row_local[bl][j-1],FI_col_local[bl][j+1][-col_meas:],j,X_dict[bl],sd)
                                        c1p = np.concatenate((sup_store[bl][j],c2p),axis = 1)
                                        sup_store[bl][j] = c1p
                                    elif col_meas ==0 and row_meas>0:
                                        c3p,X_dict[bl] = Lower_access_Superblock_subblock(tensor_entries,bounds_p[b],FI_row_local[bl][j-1][-row_meas:],FI_col_local[bl][j+1],j,X_dict[bl],sd)
                                        c1p = np.concatenate((sup_store[bl][j],c3p),axis = 0)
                                        sup_store[bl][j] = c1p
                                    else:
                                        c1p = sup_store[bl][j]
                                old_rows[bl][j] = len(FI_row_local[bl][j-1])
                                old_cols[bl][j]= len(FI_col_local[bl][j+1])

                            super_rows[bl] = c1p[I_row_local[bl][j],:]
                            super_cols[bl] = c1p[:,I_col_local[bl][j]]
                            suptilde[bl] = np.zeros((super_cols[bl].shape[0],super_rows[bl].shape[1]))
                            super_rows_t[bl] = torch.as_tensor(
                                np.ascontiguousarray(super_rows[bl]),
                                dtype=TORCH_DTYPE,
                                device=TORCH_DEVICE,
                            )
                            super_cols_t[bl] = torch.as_tensor(
                                np.ascontiguousarray(super_cols[bl]),
                                dtype=TORCH_DTYPE,
                                device=TORCH_DEVICE,
                            )
                            suptilde_t[bl] = torch.zeros(
                                suptilde[bl].shape,
                                dtype=TORCH_DTYPE,
                                device=TORCH_DEVICE,
                            )

                        #One collective instead of a gather/scatter pair over ranks.
                        _as = _prof_comm('piv',comm,lambda: comm.allgather((my_active,my_search)))
                        active_blocks = sorted([b for _x in _as for b in _x[0]])
                        search_blocks = sorted([b for _x in _as for b in _x[1]])
                        active_set = set(active_blocks)
                        search_set = set(search_blocks)
                        active_ranks = sorted(set([block_owner[b] for b in active_blocks]))
                        search_ranks = sorted(set([block_owner[b] for b in search_blocks]))
                        ut2 = MPI.Wtime()

                        # --- record this process's local superblock size for this sweep ---
                        for bl in range(nloc):
                            if suptilde[bl] is None:
                                continue
                            __sb_r, __sb_c = suptilde[bl].shape
                            __sb_area = int(__sb_r) * int(__sb_c)
                            superblock_total_area += __sb_area
                            if __sb_area > superblock_peak_area:
                                superblock_peak_area = __sb_area
                                superblock_peak_shape = (int(__sb_r), int(__sb_c))
                            superblock_build_count += 1
                        # --- log len(I) and len(J) for EVERY unfolding, every sweep, every block ---
                        track_sweep[j] += 1
                        if VERBOSE_SWEEP_LOG:
                            _rows = []
                            for bl in range(nloc):
                                b = my_blocks[bl]
                                _li = len(FI_row_local[bl][j-1]) if j > 0 else 0
                                _lj = len(FI_col_local[bl][j+1]) if j < len(dims) - 2 else 0
                                _rows.append((b, rank, 1 if b in active_set else 0, _li, _lj))
                            _info = comm.gather(_rows, root=0)
                            if rank == 0:
                                print('--- unfolding k={}  |  sweep {} for this dim ---'.format(j, track_sweep[j]))
                                for _b, _r, _act, _ii, _jj in sorted([x for y in _info for x in y]):
                                    print('    block {:>4} (rank {:>3}) | active={} | len(I)={:>4} | len(J)={:>4}'.format(_b, _r, _act, _ii, _jj))
                        ################################################################################################
                        ############################ Build local cross approximations ##################################
                        ################################################################################################
                        
                        t2 = MPI.Wtime()
                        t2_track=t2-t1
                        #Latest cross vectors, one per local block.
                        Ei_cur = [None]*nloc
                        Ej_cur = [None]*nloc
                        Delt = []

                        for l in range(len(selection[j])):
                            s = selection[j][l]

                            #The block holding the pivot computes delta once.
                            delta = None
                            if rank==block_owner[s]:
                                sb = block_local_pos[s]
                                local_row_position = list(Selection_row_local[sb][j]).index(l)
                                index_row = I_row_local[sb][j][local_row_position]
                                index_col = I_col_local[sb][j][list(Selection_col_local[sb][j]).index(l)]
                                c2 = float(suptilde_t[sb][index_row,index_col].item()) if l>0 else 0.0
                                pivot_residual = float(super_rows_t[sb][local_row_position,index_col].item()) - c2
                                delta = 1.0/pivot_residual if pivot_residual!=0.0 else 0.0
                            delta = _prof_comm('piv',comm,lambda: comm.bcast(delta,root = block_owner[s]))
                            Delt.append(delta)

                            #EXTRACT ROWS AND COLS THAT LIVE ON THIS PROCESS.
                            #A row vector is needed by every block in the same
                            #block COLUMN, a column vector by every block in the
                            #same block ROW, so index the payload by group id.
                            ei_send = {}
                            ej_send = {}
                            ei_local = {}
                            ej_local = {}
                            for bl in range(nloc):
                                b = my_blocks[bl]
                                if b not in active_set:
                                    continue
                                if l in Selection_row_local[bl][j]:
                                    index_row = list(Selection_row_local[bl][j]).index(l)
                                    new_ei_t = super_rows_t[bl][index_row:index_row+1,:]
                                    if l>0:
                                        global_row = I_row_local[bl][j][index_row]
                                        new_ei_t = new_ei_t - suptilde_t[bl][global_row:global_row+1,:]
                                    Ei_cur[bl] = new_ei_t
                                    ei_local[COL_GID[j][b]] = new_ei_t
                                    ei_send[COL_GID[j][b]] = new_ei_t.detach().cpu().numpy()
                                if l in Selection_col_local[bl][j]:
                                    index_col = list(Selection_col_local[bl][j]).index(l)
                                    new_ej_t = super_cols_t[bl][:,index_col:index_col+1]
                                    if l>0:
                                        global_col = I_col_local[bl][j][index_col]
                                        new_ej_t = new_ej_t - suptilde_t[bl][:,global_col:global_col+1]
                                    Ej_cur[bl] = new_ej_t
                                    ej_local[ROW_GID[j][b]] = new_ej_t
                                    ej_send[ROW_GID[j][b]] = new_ej_t.detach().cpu().numpy()

                            #A superblock row is needed by the blocks of the same
                            #block column, which live in this process's column
                            #group; a superblock column by the row group.
                            _exi = _prof_comm('piv',col_comms[j],lambda: col_comms[j].allgather(ei_send))
                            _exj = _prof_comm('piv',row_comms[j],lambda: row_comms[j].allgather(ej_send))
                            ei_all = {}
                            ej_all = {}
                            for _e in _exi:
                                ei_all.update(_e)
                            for _f in _exj:
                                ej_all.update(_f)

                            for bl in range(nloc):
                                b = my_blocks[bl]
                                if b not in active_set:
                                    continue
                                _g = COL_GID[j][b]
                                if _g in ei_local:
                                    Ei_cur[bl] = ei_local[_g]
                                elif _g in ei_all:
                                    Ei_cur[bl] = torch.as_tensor(
                                        np.ascontiguousarray(ei_all[_g]),
                                        dtype=TORCH_DTYPE,
                                        device=TORCH_DEVICE,
                                    )
                                _g = ROW_GID[j][b]
                                if _g in ej_local:
                                    Ej_cur[bl] = ej_local[_g]
                                elif _g in ej_all:
                                    Ej_cur[bl] = torch.as_tensor(
                                        np.ascontiguousarray(ej_all[_g]),
                                        dtype=TORCH_DTYPE,
                                        device=TORCH_DEVICE,
                                    )
                                if Ei_cur[bl] is not None and Ej_cur[bl] is not None:
                                    # addmm_ performs suptilde += delta * column @ row.
                                    suptilde_t[bl].addmm_(
                                        Ej_cur[bl],
                                        Ei_cur[bl],
                                        beta=1.0,
                                        alpha=float(delta),
                                    )

                        for bl in range(nloc):
                            if suptilde_t[bl] is not None:
                                _torch_sync()
                                suptilde[bl] = suptilde_t[bl].detach().cpu().numpy()

                        ut3 = MPI.Wtime()
                        internal_tt += ut2-ut1
                        update_time+=ut3 - ut2        
                        #Run a pivot search over all searching blocks
                        piv_local = []
                        for bl in range(nloc):
                            b = my_blocks[bl]
                            if b not in search_set:
                                continue
                            if rng_state is not None:
                                _saved = random.getstate(); random.setstate(rng_state[bl])
                            I_row_local[bl],I_col_local[bl],FI_row_local[bl],FI_col_local[bl],p_,ps_,X_dict[bl] = Low_access_MPICrossInterpSingleItemSuperPivV2(tensor_entries,suptilde[bl],bounds_p[b],X_dict[bl],I_row_local[bl],I_col_local[bl],FI_row_local[bl],FI_col_local[bl],j,sub_dim[bl])
                            if rng_state is not None:
                                rng_state[bl] = random.getstate(); random.setstate(_saved)
                            
                            Selection_row_local[bl][j] = np.append(Selection_row_local[bl][j],counter[j])
                            Selection_col_local[bl][j] = np.append(Selection_col_local[bl][j],counter[j])
                            piv_local.append((b,p_*ps_))
                            
                        #Gather the pivot values for selection
                        _pv = _prof_comm('piv',comm,lambda: comm.allgather(piv_local))
                        piv_by_block = {}
                        for _chunk in _pv:
                            for _b,_v in _chunk:
                                piv_by_block[_b] = _v

                        if len(search_blocks)==0:
                            #No block can extend its index sets any further.
                            _prof_wait('piv',comm)
                            continue

                        selection[j].append(search_blocks[int(np.argmax(np.abs([piv_by_block[_b] for _b in search_blocks])))])
                        s = selection[j][-1]
                        
                        #Send out information required to alter the index sets      
                        info_index = None
                        if rank==block_owner[s]:
                            sb = block_local_pos[s]
                            info_index = [I_row_local[sb][j],I_col_local[sb][j],FI_row_local[sb][j],FI_col_local[sb][j],sub_dim[sb]]
                        info_index = _prof_comm('piv',comm,lambda: comm.bcast(info_index,root = block_owner[s]))

                        for bl in range(nloc):
                            b = my_blocks[bl]
                            if b==s:
                                continue
                            in_row = (ROW_GID[j][b]==ROW_GID[j][s])
                            in_col = (COL_GID[j][b]==COL_GID[j][s])
                            searched = b in search_set
                            if in_row and searched:
                                I_row_local[bl][j][-1] = info_index[0][-1]
                                FI_row_local[bl][j][-1] = info_index[2][-1]
                            elif in_row and not searched:
                                I_row_local[bl][j] = info_index[0]
                                FI_row_local[bl][j] = info_index[2]
                                Selection_row_local[bl][j] = np.append(Selection_row_local[bl][j],counter[j])
                            elif (not in_row) and searched:
                                I_row_local[bl][j] = np.delete(I_row_local[bl][j],[-1])
                                FI_row_local[bl][j] = np.delete(FI_row_local[bl][j],[-1])
                                Selection_row_local[bl][j] = np.delete(Selection_row_local[bl][j],[-1])
                            if in_col and searched:
                                I_col_local[bl][j][-1] = info_index[1][-1]
                                FI_col_local[bl][j][-1] = info_index[3][-1]
                            elif in_col and not searched:
                                I_col_local[bl][j] = info_index[1]
                                FI_col_local[bl][j] = info_index[3]
                                Selection_col_local[bl][j] = np.append(Selection_col_local[bl][j],counter[j])
                            elif (not in_col) and searched:
                                I_col_local[bl][j] = np.delete(I_col_local[bl][j],[-1])
                                FI_col_local[bl][j] = np.delete(FI_col_local[bl][j],[-1])
                                Selection_col_local[bl][j] = np.delete(Selection_col_local[bl][j],[-1])
                        
                        counter[j] = counter[j]+1 
                        #Adjust global indices  
                        if rank==0:
                            adj_ind = Sub_to_Unfold([info_index[2][-1],info_index[3][-1]],info_index[4],dims,shifting_p[s],j)
                            I_row_global[j].append(adj_ind[0])
                            I_col_global[j].append(adj_ind[1])
                        _prof_wait('piv',comm)
                    
                    intermediate_time = MPI.Wtime()
                    _PROF['piv']['total'] = intermediate_time - start_time
                    t3_track = intermediate_time - t2
                    #Set up items needed to construct global cores
                    if rank==0:
                        col_tk = [[] for _ in range(len(dims)-1)]
                        cols = [[] for _ in range(len(dims)-1)]
                        core_ranks=[1 for _ in range(len(dims)+1)]
                        for i in range(len(dims)-1):
                            core_ranks[i+1] = len(set(I_row_global[i]))
                        _pt = MPI.Wtime()
                        for i in range(1,size):
                            comm.send([I_row_global,I_col_global],dest = i,tag = 909)
                        _PROF['core']['comm'] += MPI.Wtime() - _pt

                    if rank!=0:
                        _pt = MPI.Wtime()
                        index_recv = comm.recv(source = 0,tag = 909)
                        _PROF['core']['comm'] += MPI.Wtime() - _pt
                        I_row_global = index_recv[0]
                        I_col_global = index_recv[1]
                    internal_time = 0  

                    if rank!=0:
                        core_ranks = None
                    core_ranks = _prof_comm('core',comm,lambda: comm.bcast(core_ranks,root = 0))
                    
                    #This form will try to best distribute ranks based off size
                    
                    weights = [dims[0]]
                    for i in range(1,len(dims)-1):
                        weights.append(dims[i]*core_ranks[i])
                    dispatch = []
                    for i in range(len(dims)-1):
                        val = (weights[i]/sum(weights))*world_size
                        if val<1:
                            dispatch.append(1)

                        else:
                            dispatch.append(np.round(val).astype(int))
                    
                    #This form will do a strict processor grid allocation
                    
                    if world_size>=len(dims)-1:
                        dispatch = np.array_split(np.linspace(0,np.max([np.min([world_size,len(dims)-1]),math.prod(processor_partition)])-1,np.max([np.min([world_size,len(dims)-1]),math.prod(processor_partition)])).astype(int),len(dims)-1)
                        dispatch = [len(dispatch[i]) for i in range(len(dispatch))]
                        
                        rank_assignment = [np.linspace(0,dispatch[i]-1,dispatch[i]).astype(int) for i in range(len(dims)-1)]
                        
                        for i in range(1,len(rank_assignment)):
                            rank_assignment[i]+=np.max(rank_assignment[i-1])+1
                        
                    else:
                        dispatch = [1 for i in range(len(dims)-1)]
                        ph = list(np.round(np.linspace(0,world_size-1,len(dims)-1)).astype(int))
                        rank_assignment = [[ph[i]] for i in range(len(ph))]
                    
                if color!=10:
                    dispatch = None
                    rank_assignment = None
                
                dispatch = _prof_comm('core',world_comm,lambda: world_comm.bcast(dispatch,root = 0))
                rank_assignment = _prof_comm('core',world_comm,lambda: world_comm.bcast(rank_assignment,root = 0))

                # --- per-dimension core-construction timing (this rank) ---
                core_time_by_dim = {}   # dim -> seconds this rank spent building that core
                extract_time_by_dim = {}  # dim -> seconds this rank spent in the extraction/setup phase
                pull_time_by_dim = {}     # dim -> seconds this rank spent ONLY pulling tensor entries (no comm)
                _LASB['index'] = 0.0; _LASB['member'] = 0.0; _LASB['eval'] = 0.0; _LASB['assemble'] = 0.0   # reset finer split

                if world_rank==0:
                    temp = []
                
                


                if world_rank<math.prod(processor_partition):

                    _ex_t0 = None
                    _ex_prev = None
                    for j in range(len(dims)):
                        _ex_now = MPI.Wtime()
                        if _ex_prev is not None:
                            extract_time_by_dim[_ex_prev] = extract_time_by_dim.get(_ex_prev, 0.0) + (_ex_now - _ex_t0)
                        _ex_t0 = _ex_now
                        _ex_prev = j

                        my_records = []       # (block id, cols, I_row, Selection_row, I_col)
                        my_active = []
                        for bl in range(nloc):
                            b = my_blocks[bl]
                            sd = sub_dim[bl]
                            active = 0
                            if j==0:
                                if len(I_col_local[bl][1])>0:
                                    active = 1
                            elif j>=len(dims)-2:
                                if len(I_row_local[bl][j-1])>0:
                                    active = 1
                            else:
                                if len(I_row_local[bl][j-1])>0 and len(I_col_local[bl][j+1])>0:
                                    active=1  
                            if active==0:
                                continue
                            my_active.append(b)

                            _pull_t0 = MPI.Wtime()
                            if j==0:
                                super_rows_b,cols_b,X_dict[bl] = Lower_access_Superblock_test(tensor_entries,bounds_p[b],[0],FI_col_local[bl][1],I_row_local[bl][0],I_col_local[bl][0],0,X_dict[bl],sd)
                                rec = (b,cols_b,I_row_local[bl][j],Selection_row_local[bl][j],I_col_local[bl][j])
                            elif j==len(dims)-1:
                                super_rows_b,cols_b,X_dict[bl] = Lower_access_Superblock_test(tensor_entries,bounds_p[b],FI_row_local[bl][j-2],[0],I_row_local[bl][j-1],I_col_local[bl][j-1],j-1,X_dict[bl],sd)
                                final_core = cols_b[I_row_local[bl][j-1],:]
                                rec = (b,cols_b,I_row_local[bl][-1],Selection_row_local[bl][-1],I_col_local[bl][-1])
                            elif j==len(dims)-2:
                                super_rows_b,cols_b,X_dict[bl] = Lower_access_Superblock_test(tensor_entries,bounds_p[b],FI_row_local[bl][j-1],[0],I_row_local[bl][j],I_col_local[bl][j],j,X_dict[bl],sd)
                                rec = (b,cols_b,I_row_local[bl][j],Selection_row_local[bl][j],I_col_local[bl][j])
                            else:
                                super_rows_b,cols_b,X_dict[bl] = Lower_access_Superblock_test(tensor_entries,bounds_p[b],FI_row_local[bl][j-1],FI_col_local[bl][j+1],I_row_local[bl][j],I_col_local[bl][j],j,X_dict[bl],sd)
                                rec = (b,cols_b,I_row_local[bl][j],Selection_row_local[bl][j],I_col_local[bl][j])
                            pull_time_by_dim[j] = pull_time_by_dim.get(j, 0.0) + (MPI.Wtime() - _pull_t0)
                            my_records.append(rec)

                        if j<len(dims)-1:
                            test = _prof_comm('core',comm,lambda: comm.gather(my_records,root = 0))
                            if world_rank==0:
                                recs = {}
                                for _chunk in test:
                                    for _rec in _chunk:
                                        recs[_rec[0]] = _rec

                                #Group the active blocks into the block ROWS of
                                #this unfolding.  Ordering by group id (and by
                                #block id inside a group) makes the assembled
                                #matrix independent of which rank owns what.
                                groups = {}
                                for _b in sorted(recs):
                                    groups.setdefault(ROW_GID[j][_b],[]).append(_b)
                                g_sorted = sorted(groups)

                                row_indx = []
                                row_selx = []
                                _offset = 0
                                for _g in g_sorted:
                                    _b0 = groups[_g][0]
                                    if len(recs[_b0][2])!=0:
                                        row_indx.extend([int(x)+_offset for x in recs[_b0][2]])
                                        row_selx.extend([int(x) for x in recs[_b0][3]])
                                    _offset += recs[_b0][1].shape[0]

                                pls = np.block([[np.block([recs[_b][1] for _b in groups[_g]])] for _g in g_sorted])
                                
                                col_split,shift,bounds_core = Subtensor_Split(pls,[dispatch[j],1])
                                local_ranks = []
                                loc_row_indx = []
                                for i in row_indx:
                                    for l in range(len(col_split)):
                                        if i>= bounds_core[l][0][0] and i< bounds_core[l][0][1]:
                                            local_ranks.append(rank_assignment[j][l])
                                            loc_row_indx.append(i - bounds_core[l][0][0])
                                for i in range(len(rank_assignment[j])):
                                    temp.append([col_split[i],row_selx,local_ranks,loc_row_indx,rank_assignment[j],rank_assignment[j][i],j])
                    if _ex_prev is not None:
                        extract_time_by_dim[_ex_prev] = extract_time_by_dim.get(_ex_prev, 0.0) + (MPI.Wtime() - _ex_t0)
                temp_cols = []                  
                
                if world_rank==0:
                    
                    _pt = MPI.Wtime()
                    for i in range(len(temp)):
                        if temp[i][5]!=0:
                            world_comm.send(temp[i],dest = temp[i][5],tag = 999)
                        else:
                            temp_cols.append(temp[i])
                    _PROF['core']['comm'] += MPI.Wtime() - _pt
                            
                if world_rank!=0 and world_rank<=np.max([x for y in rank_assignment for x in y]):#sum(dispatch):
                    
                    ph = [x for y in rank_assignment for x in y]
                    _pt = MPI.Wtime()
                    for _ in range(ph.count(world_rank)):
                        temp_cols.append(world_comm.recv(source = 0,tag = 999))
                    _PROF['core']['comm'] += MPI.Wtime() - _pt
                    
                
                if world_rank<=np.max([x for y in rank_assignment for x in y]):#sum(dispatch):
                    
                    ########local_cols[4] give processor allocation per core (who can communicate with who for Tk construction)
                    ########local_cols[3] gives local row indices for selection
                    ########local_cols[2] gives rank in which each row is located
                    ########local_cols[1] gives the order of selection of indices in local_cols[3]
                    for local_cols in temp_cols:
                        _cc_t0 = MPI.Wtime()
                        _cc_dim = local_cols[6]
                        #Fix ordering issue
                        new_order = list(np.linspace(0,len(local_cols[1])-1,len(local_cols[1])).astype(int))#[local_cols[1].index(i) for i in range(len(local_cols[1]))]
                        
                        #Kick start
                        #First give all the necessary submatrix A(I,J) 
                        indicator = [local_cols[3][i] for i in range(len(local_cols[2])) if local_cols[2][new_order[i]]==world_rank]
                        loc_submatrix=np.zeros((len(local_cols[3]),local_cols[0].shape[1]))
                        placement = [i for i in range(len(local_cols[2])) if local_cols[2][new_order[i]]==world_rank]
                        loc_submatrix[placement,:] = local_cols[0][indicator,:]
                        
                        #tt1 = MPI.Wtime()
                        _pt = MPI.Wtime()
                        _t34 = []   # isend so the recv loop is always reached
                        for i in local_cols[4]:
                            if i!=world_rank:
                                _t34.append(world_comm.isend([local_cols[0][indicator,:],placement],dest = i,tag = 34))
                        for i in local_cols[4]:
                            if i!=world_rank:
                                placehold = world_comm.recv(source = i,tag = 34)
                                loc_submatrix[placehold[1],:] = placehold[0]
                        if _t34:
                            MPI.Request.waitall(_t34)
                        if world_rank == local_cols[2][new_order[0]]:
                            scale_value = 1/local_cols[0][local_cols[3][new_order[0]],0]
                            for i in local_cols[4]:
                                if i!=world_rank:
                                    world_comm.send(scale_value,dest = i,tag = 24)
                        if world_rank!=local_cols[2][new_order[0]] and world_rank in local_cols[4]:
                            scale_value = world_comm.recv(source = local_cols[2][new_order[0]],tag = 24)
                        _PROF['core']['comm'] += MPI.Wtime() - _pt
                        # keeping the distributed Tk numerical kernel
                        # in PyTorch and preallocate all columns.  This removes the
                        # repeated np.concatenate allocation from every rank step.
                        local_values_t = torch.as_tensor(
                            np.ascontiguousarray(local_cols[0]),
                            dtype=TORCH_DTYPE,
                            device=TORCH_DEVICE,
                        )
                        loc_submatrix_t = torch.as_tensor(
                            np.ascontiguousarray(loc_submatrix),
                            dtype=TORCH_DTYPE,
                            device=TORCH_DEVICE,
                        )
                        tk_rank = len(local_cols[2])
                        Tk_local_t = torch.empty(
                            (local_values_t.shape[0],tk_rank),
                            dtype=TORCH_DTYPE,
                            device=TORCH_DEVICE,
                        )
                        Tk_local_t[:,0:1].copy_(
                            float(scale_value) * local_values_t[:,0:1]
                        )
                        

                        #Finish process
                        #First build Sk
                        tt2 = MPI.Wtime()
                        for j in range(1,tk_rank):
                            Tk_current_t = Tk_local_t[:,:j]
                            Sk_t = (
                                Tk_current_t @ loc_submatrix_t[:j,j:j+1]
                                - local_values_t[:,j:j+1]
                            )
                            _pt = MPI.Wtime()
                            if world_rank == local_cols[2][new_order[j]]:
                                pivot_row = local_cols[3][new_order[j]]
                                scale_valueinv = float(Sk_t[pivot_row,0].item())
                                scale_value = -1.0/scale_valueinv if scale_valueinv!=0.0 else 0.0
                                ext_t = Tk_current_t[pivot_row:pivot_row+1,:].clone()
                                ext_send = ext_t.detach().cpu().numpy()
                                for i in local_cols[4]:
                                    if i!=world_rank:
                                        world_comm.send(scale_value,dest = i,tag = 24)
                                        world_comm.send(ext_send,dest = i,tag = 24)
                            if world_rank!=local_cols[2][new_order[j]] and world_rank in local_cols[4]:
                                scale_value = world_comm.recv(source = local_cols[2][new_order[j]],tag = 24)
                                ext_recv = world_comm.recv(source = local_cols[2][new_order[j]],tag = 24)
                                ext_t = torch.as_tensor(
                                    np.ascontiguousarray(ext_recv),
                                    dtype=TORCH_DTYPE,
                                    device=TORCH_DEVICE,
                                )
                            _PROF['core']['comm'] += MPI.Wtime() - _pt

                            # Existing columns: 
                            Tk_current_t.addmm_(
                                Sk_t,
                                ext_t,
                                beta=1.0,
                                alpha=float(scale_value),
                            )
                            # New column: 
                            Tk_local_t[:,j:j+1].copy_(Sk_t)
                            Tk_local_t[:,j:j+1].mul_(-float(scale_value))

                        _torch_sync()
                        core_time_by_dim[_cc_dim] = core_time_by_dim.get(_cc_dim, 0.0) + (MPI.Wtime() - _cc_t0)
                    
                final_time = MPI.Wtime()
                try:
                    _PROF['core']['total'] = final_time - intermediate_time
                except NameError:
                    _PROF['core']['total'] = 0.0
                # --- gather per-dimension core-construction times across all world ranks ---
                _cc_all = world_comm.gather((world_rank, core_time_by_dim), root=0)
                if world_rank == 0:
                    _cc_perdim = {}      # dim -> list of (rank, seconds)
                    _cc_perrank = {}     # rank -> total seconds in construction
                    for _wr, _d in _cc_all:
                        if _d:
                            _cc_perrank[_wr] = float(sum(_d.values()))
                            for _dim, _t in _d.items():
                                _cc_perdim.setdefault(_dim, []).append((_wr, float(_t)))
                    cc_perdim_report = _cc_perdim
                    cc_perrank_report = _cc_perrank
                    cc_serial_work = float(sum(_cc_perrank.values())) if _cc_perrank else 0.0
                    cc_critical_path = float(max(_cc_perrank.values())) if _cc_perrank else 0.0
                # --- gather extraction-phase per-dimension times across all world ranks ---
                _ex_all = world_comm.gather((world_rank, extract_time_by_dim), root=0)
                if world_rank == 0:
                    _ex_perdim_total = {}   # dim -> total seconds over all ranks
                    _ex_perdim_max = {}     # dim -> max seconds on any single rank (wall-clock contribution)
                    for _wr, _d in _ex_all:
                        if _d:
                            for _dim, _t in _d.items():
                                _ex_perdim_total[_dim] = _ex_perdim_total.get(_dim, 0.0) + float(_t)
                                _ex_perdim_max[_dim] = max(_ex_perdim_max.get(_dim, 0.0), float(_t))
                    ex_perdim_total_report = _ex_perdim_total
                    ex_perdim_max_report = _ex_perdim_max
                    ex_total_all = float(sum(_ex_perdim_total.values())) if _ex_perdim_total else 0.0
                    ex_walltime = float(sum(_ex_perdim_max.values())) if _ex_perdim_max else 0.0
                # --- gather entry-pull-only (no comm) per-dimension times across all world ranks ---
                _pull_all = world_comm.gather((world_rank, pull_time_by_dim), root=0)
                if world_rank == 0:
                    _pull_perdim_total = {}
                    _pull_perdim_max = {}
                    for _wr, _d in _pull_all:
                        if _d:
                            for _dim, _t in _d.items():
                                _pull_perdim_total[_dim] = _pull_perdim_total.get(_dim, 0.0) + float(_t)
                                _pull_perdim_max[_dim] = max(_pull_perdim_max.get(_dim, 0.0), float(_t))
                    pull_perdim_total_report = _pull_perdim_total
                    pull_perdim_max_report = _pull_perdim_max
                    pull_total_all = float(sum(_pull_perdim_total.values())) if _pull_perdim_total else 0.0
                    pull_walltime = float(sum(_pull_perdim_max.values())) if _pull_perdim_max else 0.0
                # --- gather the finer eval/assemble/index split across all world ranks ---
                _lasb_all = world_comm.gather((world_rank, _LASB['index'], _LASB['member'], _LASB['eval'], _LASB['assemble']), root=0)
                if world_rank == 0:
                    _idx = [t[1] for t in _lasb_all if (t[1]+t[2]+t[3]+t[4]) > 0]
                    _mem = [t[2] for t in _lasb_all if (t[1]+t[2]+t[3]+t[4]) > 0]
                    _evl = [t[3] for t in _lasb_all if (t[1]+t[2]+t[3]+t[4]) > 0]
                    _asm = [t[4] for t in _lasb_all if (t[1]+t[2]+t[3]+t[4]) > 0]
                    lasb_idx_total = float(sum(_idx)) if _idx else 0.0
                    lasb_idx_max   = float(max(_idx)) if _idx else 0.0
                    lasb_mem_total = float(sum(_mem)) if _mem else 0.0
                    lasb_mem_max   = float(max(_mem)) if _mem else 0.0
                    lasb_evl_total = float(sum(_evl)) if _evl else 0.0
                    lasb_evl_max   = float(max(_evl)) if _evl else 0.0
                    lasb_asm_total = float(sum(_asm)) if _asm else 0.0
                    lasb_asm_max   = float(max(_asm)) if _asm else 0.0
                # --- EVERY rank computes its own phase elapsed time---
                try:
                    _my_piv  = float(intermediate_time - start_time)
                    _my_core = float(final_time - intermediate_time)
                except NameError:
                    _my_piv = -1.0; _my_core = -1.0
                # --- per-rank compute / communication / wait, both phases ---
                # compute = phase wall time minus everything spent inside MPI.
                # wait is only separated from comm when TT_PROFILE is set; with
                # it unset every collective's idle time shows up under comm.
                _prof_all = world_comm.gather(
                    (world_rank,
                     _PROF['piv']['total'], _PROF['piv']['comm'], _PROF['piv']['wait'],
                     _PROF['core']['total'], _PROF['core']['comm'], _PROF['core']['wait']),
                    root=0)
                if world_rank == 0:
                    prof_rows = []
                    for _wr, _pt, _pc, _pw, _ct, _cc2, _cw in _prof_all:
                        if _pt <= 0.0 and _ct <= 0.0:
                            continue      # rank sat out this partition entirely
                        prof_rows.append({
                            'rank': _wr,
                            'piv_total': _pt, 'piv_comm': _pc, 'piv_wait': _pw,
                            'piv_compute': max(_pt - _pc - _pw, 0.0),
                            'core_total': _ct, 'core_comm': _cc2, 'core_wait': _cw,
                            'core_compute': max(_ct - _cc2 - _cw, 0.0),
                        })
                    if _PROF_OUT and prof_rows:
                        _cols = ['rank', 'piv_total', 'piv_compute', 'piv_comm', 'piv_wait',
                                 'core_total', 'core_compute', 'core_comm', 'core_wait']
                        with open(_PROF_OUT, 'w') as _fh:
                            _fh.write(','.join(_cols) + '\n')
                            for _r in prof_rows:
                                _fh.write(','.join(str(_r[_c]) for _c in _cols) + '\n')

                _phase_all = world_comm.gather((world_rank, _my_piv, _my_core), root=0)
                if world_rank == 0:
                    _pa = [t for t in _phase_all if t[1] >= 0.0]
                    if _pa:
                        _pv = [t[1] for t in _pa]; _cv = [t[2] for t in _pa]
                        phase_n          = len(_pa)
                        phase_piv_max    = float(max(_pv)); phase_piv_min = float(min(_pv))
                        phase_piv_avg    = float(sum(_pv))/len(_pv)
                        phase_piv_argmax = int(_pa[_pv.index(max(_pv))][0])
                        phase_core_max   = float(max(_cv)); phase_core_min = float(min(_cv))
                        phase_core_avg   = float(sum(_cv))/len(_cv)
                        phase_core_argmax= int(_pa[_cv.index(max(_cv))][0])
                        phase_rank0_piv  = float(_phase_all[0][1]); phase_rank0_core = float(_phase_all[0][2])
                    else:
                        phase_n = 0
                        phase_piv_max=phase_piv_min=phase_piv_avg=phase_rank0_piv=0.0
                        phase_core_max=phase_core_min=phase_core_avg=phase_rank0_core=0.0
                        phase_piv_argmax=phase_core_argmax=-1
                if world_rank==0:
                    piv_times.append(intermediate_time - start_time)
                    core_times.append(final_time - intermediate_time)
                    core_times_list.append(np.min(core_times))
                    pivot_times_list.append(np.min(piv_times))
                # --- gather superblock-size stats across active processes (outside timed regions) ---
                if color==10:
                    sb_peak_all = comm.gather(superblock_peak_area, root=0)
                    sb_total_all = comm.gather(superblock_total_area, root=0)
                    sb_shape_all = comm.gather(superblock_peak_shape, root=0)
                    sb_count_all = comm.gather(superblock_build_count, root=0)
                    sb_block_all = comm.gather(len(my_blocks), root=0)
                    if rank == 0:
                        _act = [i for i in range(len(sb_count_all)) if sb_count_all[i] > 0]
                        _peaks = [sb_peak_all[i] for i in _act]
                        _totals = [sb_total_all[i] for i in _act]
                        if len(_peaks) > 0:
                            sb_avg_peak = float(np.mean(_peaks))
                            sb_max_peak = int(np.max(_peaks))
                            sb_avg_total = float(np.mean(_totals))
                            sb_busiest_shape = sb_shape_all[_act[int(np.argmax(_peaks))]]
                            sb_nactive = len(_act)
                            # load balance of the pivot-search phase: total superblock
                            # area searched per process.  imbalance = max/avg, so 1.0 is
                            # perfect and larger means one process is holding up the rest.
                            sb_total_min = float(np.min(_totals))
                            sb_total_max = float(np.max(_totals))
                            sb_imbalance = (sb_total_max/sb_avg_total) if sb_avg_total > 0 else 0.0
                            sb_nactive_blocks = int(sum(sb_block_all))
            total = 0
            largest = 0
            obje = None
            local_vars = list(locals().items())
            for var, obj in local_vars:
                if sys.getsizeof(obj)>largest:
                    largest = sys.getsizeof(obj)
                    obje = var
                    
                total+= sys.getsizeof(obj)
            total = comm.gather(total,root = 0)
            largest = comm.gather(largest,root = 0)
            if 'X_dict' in locals():
                access = sum(len(_x) for _x in X_dict)
            else:
                access = None

            access = world_comm.gather(access,root = 0)
            if world_rank==0:
                # rank 0 owns block 0, whose local coordinates coincide with the
                # global ones, so its cache can be reused for the serial extraction.
                cores,_core_dict = Low_Access_Core_Extract_Update(tensor_entries_glob, dims, X_dict[0], I_row_global, I_col_global)
                
                # Converting the final TT cores once and evaluating all
                # sampled indices in a batch instead of one Python
                # matrix-chain loop per tensor entry.
                torch_cores = [
                    torch.as_tensor(
                        np.ascontiguousarray(core),
                        dtype=TORCH_DTYPE,
                        device=TORCH_DEVICE,
                    )
                    for core in cores
                ]

                err_meas = []
                error_rng = np.random.default_rng(
                    int(_SEED)+7919 if _SEED is not None else None)
                with torch.inference_mode():
                    for _ in range(10):
                        sample_indices = np.column_stack(
                            [
                                error_rng.integers(
                                    0,
                                    dims[mode],
                                    size=err_sample_size,
                                    dtype=np.int64,
                                )
                                for mode in range(len(dims))
                            ]
                        )

                        true_values = np.asarray(
                            tensor_entries_glob(sample_indices),
                            dtype=np.float64,
                        ).reshape(-1)
                        approx_values_t = Core_to_Tensor_Values_Torch(
                            torch_cores,
                            sample_indices,
                        )
                        true_values_t = torch.as_tensor(
                            true_values,
                            dtype=TORCH_DTYPE,
                            device=TORCH_DEVICE,
                        )

                        sample_error_t = torch.linalg.vector_norm(
                            true_values_t - approx_values_t
                        )
                        sample_norm_t = torch.linalg.vector_norm(true_values_t)
                        relative_error = sample_error_t / sample_norm_t
                        err_meas.append(float(relative_error.item()))
                    
                print('*'*65)
                print('Percent of time for full pivot searching   :',np.round(np.min(piv_times),4),'seconds')
                print('   [per-rank] pivot  max={:.4f}s (rank {})  min={:.4f}s  avg={:.4f}s  rank0={:.4f}s  spread(max-min)={:.4f}s'.format(
                      phase_piv_max, phase_piv_argmax, phase_piv_min, phase_piv_avg, phase_rank0_piv, phase_piv_max-phase_piv_min))
                print('-'*65)
                print('Percent of time for core construction      :',np.round(np.min(core_times),4),'seconds')
                print('   [per-rank] core   max={:.4f}s (rank {})  min={:.4f}s  avg={:.4f}s  rank0={:.4f}s  spread(max-min)={:.4f}s'.format(
                      phase_core_max, phase_core_argmax, phase_core_min, phase_core_avg, phase_rank0_core, phase_core_max-phase_core_min))
                # print('-'*65)
                # print('Extraction phase per CORE (Tensor-entry pulling + communication) (last trial):')
                # for _dim in sorted(ex_perdim_total_report.keys()):
                #     print('    core j={} | total over ranks = {:.4f}s | max on one rank = {:.4f}s'.format(
                #         _dim, ex_perdim_total_report[_dim], ex_perdim_max_report.get(_dim, 0.0)))
                # print('    extraction total work (sum over ranks)        : {:.4f}s'.format(ex_total_all))
                # print('    extraction wall-clock estimate (sum of maxes) : {:.4f}s'.format(ex_walltime))
                # print('-'*65)
                # print('Only Tensor-entry pulling, no communication (last trial):')
                # for _dim in sorted(pull_perdim_total_report.keys()):
                #     print('    core j={} | total over ranks = {:.4f}s | max on one rank = {:.4f}s'.format(
                #         _dim, pull_perdim_total_report[_dim], pull_perdim_max_report.get(_dim, 0.0)))
                # print('    entry-pull total work (sum over ranks)         : {:.4f}s'.format(pull_total_all))
                # print('    entry-pull wall-clock estimate (sum of maxes)  : {:.4f}s'.format(pull_walltime))
                # print('-'*65)
                # print('Inside entry-pulling: index-building vs dict-membership vs entry-evaluation vs A_k matrix assembly (last trial):')
                # print('    index-building (Super_to_Global) | total over ranks = {:.4f}s | max on one rank = {:.4f}s'.format(lasb_idx_total, lasb_idx_max))
                # print('    dict-membership (not in X_dict + reads) | total = {:.4f}s | max on one rank = {:.4f}s'.format(lasb_mem_total, lasb_mem_max))
                # print('    entry-evaluation (tensor_entry, on misses) | total = {:.4f}s | max on one rank = {:.4f}s'.format(lasb_evl_total, lasb_evl_max))
                # print('    A_k matrix assembly (np.reshape) | total = {:.4f}s | max on one rank = {:.4f}s'.format(lasb_asm_total, lasb_asm_max))
                # print('-'*65)
                # print('Communication = extraction - Tensor Entry pulling  : {:.4f}s (wall-clock est.)'.format(ex_walltime - pull_walltime))
                # print('-'*65)
                # print('Core construction (build step) per UNFOLDING (last trial):')
                # for _dim in sorted(cc_perdim_report.keys()):
                #     _entries = sorted(cc_perdim_report[_dim])
                #     _tag = ', '.join(['rank {} = {:.4f}s'.format(_r, _t) for _r, _t in _entries])
                #     print('    unfolding k={} | built by: {}'.format(_dim, _tag))
                # print('    sum over ranks (serial work)  : {:.4f}s'.format(cc_serial_work))
                # print('    max over ranks: {:.4f}s'.format(cc_critical_path))
                print('-'*65)
                print('Sampled error                              :',"{0:.3E}".format(np.min(err_meas)))
                print('-'*65)
                print('Total storage needed                       :',"{0:.3E}".format((1e-9)*sum(total)),'GB')
                print('-'*65)
                print('Largest local storage needed               :',"{0:.3E}".format((1e-9)*np.max(total)),'GB')
                print('-'*65)
                print('Percent of full tensor accessed            :',"{0:.3E}".format(100*sum(filter(None,access))/math.prod(dims)),'%')
                print('-'*65)
                print('Avg superblock size over the MPI ranks     :',"{0:.3E}".format(sb_avg_peak),'entries')
                print('-'*65)
                print('Active procs in pivot search               :',sb_nactive)
                print('-'*65)
                print('Pivot-search load balance (superblock area) : max={:.3E}  avg={:.3E}  min={:.3E}  imbalance(max/avg)={:.3f}'.format(
                      sb_total_max, sb_avg_total, sb_total_min, sb_imbalance))
                print('-'*65)
                if prof_rows:
                    print('Per-rank time breakdown over {} ranks  (wait split: {})'.format(
                          len(prof_rows), 'ON' if _PROFILE_SYNC else 'OFF - folded into comm, set TT_PROFILE=1 to split'))
                    for _ph, _tag in (('piv', 'PIVOT SEARCH'), ('core', 'CORE CONSTRUCTION')):
                        _tot_hi, _, _tot_avg, _ = _prof_stats(prof_rows, _ph+'_total')
                        print('  {} : wall={:.4f}s'.format(_tag, _tot_hi))
                        print(_prof_line('compute', prof_rows, _ph+'_compute'))
                        print(_prof_line('comm', prof_rows, _ph+'_comm'))
                        if _PROFILE_SYNC:
                            print(_prof_line('wait', prof_rows, _ph+'_wait'))
                        _c_hi, _c_lo, _c_avg, _c_who = _prof_stats(prof_rows, _ph+'_compute')
                        _w_hi, _w_lo, _w_avg, _ = _prof_stats(prof_rows, _ph+'_wait')
                        _m_hi, _, _m_avg, _ = _prof_stats(prof_rows, _ph+'_comm')
                        print('    -> busiest rank {} does {:.4f}s of local work; the average rank does {:.4f}s'.format(
                              _c_who, _c_hi, _c_avg))
                        if _tot_hi > 0:
                            print('    -> shares of the phase wall: compute {:.1f}%  comm {:.1f}%{}'.format(
                                  100*_c_avg/_tot_hi, 100*_m_avg/_tot_hi,
                                  '  wait {:.1f}%'.format(100*_w_avg/_tot_hi) if _PROFILE_SYNC else ''))
                        if _PROFILE_SYNC and _c_avg > 0:
                            print('    -> imbalance cost: avg idle {:.4f}s = {:.1f}% of avg local work'.format(
                                  _w_avg, 100*_w_avg/_c_avg))
                    if len(prof_rows) <= _PROF_MAXRANKS:
                        print('   rank |  piv_compute   piv_comm   piv_wait |  core_compute  core_comm  core_wait')
                        for _r in prof_rows:
                            print('   {:>4} | {:12.4f} {:10.4f} {:10.4f} | {:13.4f} {:10.4f} {:10.4f}'.format(
                                  _r['rank'], _r['piv_compute'], _r['piv_comm'], _r['piv_wait'],
                                  _r['core_compute'], _r['core_comm'], _r['core_wait']))
                    elif not _PROF_OUT:
                        print('   ({} ranks; set TT_PROFILE_RANKS higher for a per-rank table, or TT_PROFILE_OUT=file.csv)'.format(len(prof_rows)))
                    if _PROF_OUT:
                        print('   per-rank CSV written to', _PROF_OUT)
                print('-'*65)
                print('Core ranks',core_ranks)
                print('-'*65)
                print('Block grid       ', layout_report['block_grid'],
                      '->', layout_report['n_blocks'], 'subtensors')
                print('Processor Partition', layout_report['proc_grid'],
                      '->', layout_report['n_procs'], 'ranks;',
                      layout_report['mapping'], 'mapping;',
                      'blocks/rank min={} max={} avg={:.2f}'.format(
                          layout_report['blocks_per_proc_min'],
                          layout_report['blocks_per_proc_max'],
                          layout_report['blocks_per_proc_avg']))
                print('*'*65)
                print('Size', dims)
                print('*'*65)
  
            
                
                
                

            
