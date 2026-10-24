import os

#This forces numpy to not thread, comment out for standard numpy
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI
import torch as tn
import tt_utils

def tensor_entry(indices,dim):
    
    def normalize_indices(indices):
        if isinstance(indices[0], tuple):
            if type(indices[0][0]) == tn.Tensor:
                grid_shape = tuple(indices[0][0].shape)
                inds = np.stack([idx.detach().cpu().numpy() for idx in indices[0]], axis=-1)
                return inds.reshape(-1, len(dim)), grid_shape, True

            if isinstance(indices[0][0], np.ndarray):
                grid_shape = indices[0][0].shape
                inds = np.stack(indices[0], axis=-1)
                return inds.reshape(-1, len(dim)), grid_shape, False

        if type(indices[0][0]) == tn.Tensor:
            return np.column_stack(indices).T, None, True

        return np.array(indices), None, False

    inds, grid_shape, use_torch = normalize_indices(indices)

    if tensor_def == "Matern":

        index_tuple = np.array([np.cos((inds[:,i]+1/2)*np.pi/(dim[i])) for i in range(len(dim))]).T
        index_x = ((3-2)/2)*index_tuple[:,:len(dim)//2] + ((3+2)/2)
        index_y = ((1-0)/2)*index_tuple[:,len(dim)//2:] + ((1+0)/2)
        norm = np.linalg.norm(index_x - index_y,axis = 1)
        val = (1.0 + np.sqrt(5.0)*norm + (5/3)*norm**2) * np.exp(-np.sqrt(5)*norm)
        if grid_shape is not None:
            val = val.reshape(grid_shape)
        if use_torch:
            value = tn.from_numpy(val).float()
        else:
            value = val
    elif tensor_def == "ThinPlateSpline":
        index_tuple = np.array([np.cos((inds[:,i]+1/2)*np.pi/(dim[i])) for i in range(len(dim))]).T
        index_x = ((3-2)/2)*index_tuple[:,:len(dim)//2] + ((3+2)/2)
        index_y = ((1-0)/2)*index_tuple[:,len(dim)//2:] + ((1+0)/2)
        norm = np.linalg.norm(index_x - index_y,axis = 1)
        val = np.log(norm**2)*norm**2
        if grid_shape is not None:
            val = val.reshape(grid_shape)
        if use_torch:
            value = tn.from_numpy(val).float()
        else:
            value = val
    elif tensor_def == "2d2v":
        ax,ay,bx,by = -1/2,-1/2,1/2,1/2
        avx,avy,bvx,bvy = -3,-3,3,3
    
        x = ax + (inds[:,0]/(dim[0]-1))*(bx-ax)
        y = ay + (inds[:,2]/(dim[2]-1))*(by-ay)
        vx = avx + (inds[:,1]/(dim[1]-1))*(bvx - avx)
        vy = avy + (inds[:,3]/(dim[3]-1))*(bvy - avy)
    
        rhox = 1+(7/8)*np.sin(2*np.pi*x)
        Tx = (1/2)+(2/5)*np.sin(2*np.pi*x)
        rhoy = 1+(7/8)*np.sin(2*np.pi*y)
        Ty = (1/2)+(2/5)*np.sin(2*np.pi*y)
    
        val = (((rhox)/(2*np.sqrt(2*np.pi*(Tx))))+((rhoy)/(2*np.sqrt(2*np.pi*Ty))))*(np.exp(-((np.abs(vx - 3/4)**2)/(2*Tx))-((np.abs(vy - 3/4)**2)/(2*Ty))) + np.exp(-((np.abs(vx + 3/4)**2)/(2*Tx))-((np.abs(vy + 3/4)**2)/(2*Ty))) )
        if grid_shape is not None:
            val = val.reshape(grid_shape)
        if use_torch:
            value = tn.from_numpy(val).float()
        else:
            value = val
    elif tensor_def == "3d3v":
        ax,ay,az,bx,by,bz = -1/2,-1/2,-1/2,1/2,1/2,1/2
        avx,avy,avz,bvx,bvy,bvz = -3,-3,-3,3,3,3
       
        x = ax + (inds[:,0]/(dim[0]-1))*(bx-ax)
        y = ay + (inds[:,2]/(dim[2]-1))*(by-ay)
        z = az + (inds[:,4]/(dim[4]-1))*(bz-az)
    
        vx = avx + (inds[:,1]/(dim[1]-1))*(bvx - avx)
        vy = avy + (inds[:,3]/(dim[3]-1))*(bvy - avy)
        vz = avz + (inds[:,5]/(dim[5]-1))*(bvz - avz)
    
        rhox = 1+(7/8)*np.sin(2*np.pi*x)
        Tx = (1/2)+(2/5)*np.sin(2*np.pi*(x))
    
        rhoy = 1+(7/8)*np.sin(2*np.pi*y)
        Ty = (1/2)+(2/5)*np.sin(2*np.pi*(y))

        rhoz = 1+(7/8)*np.sin(2*np.pi*z)
        Tz = (1/2)+(2/5)*np.sin(2*np.pi*(z))
    
        val = 1+(((rhox)/(2*np.sqrt((2*np.pi*(Tx))**3)))+((rhoy)/(2*np.sqrt((2*np.pi*Ty)**3)))+((rhoz)/(2*np.sqrt((2*np.pi*Tz)**3))))*(np.exp(-((np.abs(vx - 3/4)**2)/(2*Tx))-((np.abs(vy - 3/4)**2)/(2*Ty))-((np.abs(vz - 3/4)**2)/(2*Tz))) + np.exp(-((np.abs(vx + 3/4)**2)/(2*Tx))-((np.abs(vy + 3/4)**2)/(2*Ty))-((np.abs(vz + 3/4)**2)/(2*Tz))) )
        if grid_shape is not None:
            val = val.reshape(grid_shape)
        if use_torch:
            value = tn.from_numpy(val).float()
        else:
            value = val
    elif tensor_def == "Hilbert":
        #value = [1/(sum(indices[j])+1) for j in range(len(indices))]
        val = 1/(np.sum(inds,axis = 1)+1)
        if grid_shape is not None:
            val = val.reshape(grid_shape)
        if use_torch:
            value = tn.from_numpy(val).float()
        else:
            value = val
    elif tensor_def == "Hilbert_Python":
        value = [1.0/(sum(i)+1) for i in inds]
        
            
    
    else:
        raise ValueError(f"Unknown tensor_def: {tensor_def}")
    
    return value


def tensor_block(rp, cp, k):
    """Block form of the Hilbert entry for tt_cross_os2_mpi2(tensor_block_fn=...):
    values at every combination of row parts rp (dims 0..k) and column parts
    cp (dims k+1..d-1).  1/(1 + sum of all coordinates) splits into a row sum
    and a column sum, so a whole scan costs O(len(rp) + len(cp)) and no full
    multi-index is ever formed.  Leave tensor_block_fn out for a tensor without
    such structure; the generic path then evaluates entry by entry as before."""
    return 1.0 / (1.0 + rp.sum(1, dtype=np.int64)[:, None] + cp.sum(1, dtype=np.int64)[None, :])


if __name__ == "__main__":
    import sys

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # MPI process count supplied by the Slurm script
    mpin = int(sys.argv[1])

    if size != mpin:
        if rank == 0:
            print(
                f"ERROR: Expected {mpin} MPI ranks, "
                f"but received {size}",
                flush=True
            )
        comm.Abort(1)

    tensor_def = "Hilbert"

    tensor_dim = [2000]*8
    sample_row = [0]*(len(tensor_dim)-1)
    sample_col = [0]*(len(tensor_dim)-1)
    min_ranks = [1]*(len(tensor_dim)-1)
    max_ranks = [15]*(len(tensor_dim)-1)
    tolerance = 1e-15
    random_seed = 2026

    num_unfoldings = len(tensor_dim) - 1

    nthreads = min(3, num_unfoldings)  # Threads per MPI rank, across unfoldings


    # Synchronize before timing
    comm.Barrier()
    time_start = MPI.Wtime()
    #partial(tensor_entry_logistic, Xs=Xs, YtX=YtX, grids_1d=grids_local)
    cores, FI, FJ, info = tt_utils.tt_cross_os2_mpi2(
        tensor_entry,
        tensor_dim,
        sample_row,
        sample_col,
        min_rank=min_ranks,
        max_rank=max_ranks,
        tol=tolerance,
        sampling_type="Nested",
        tol_type="Relative",
        adap_toler=None,
        err_sample_size=1000,
        return_info=True,
        process_grid=(1,1),  # Near-square 2D grid using ALL MPI ranks
        block_size=64,
        pivot_ranks=None,  # Every MPI rank participates in every unfolding
        nthreads=nthreads,
        max_peid_ranks=mpin,
        tensor_block_fn=tensor_block if tensor_def == "Hilbert" else None,
    )

    comm.Barrier()
    time_end = MPI.Wtime()

    if rank == 0:

        ranks = [
            cores[i].shape[2]
            for i in range(len(cores)-1)
        ]

        err = tt_utils.sampled_error(
            tensor_entry,
            tensor_dim,
            100000,
            cores
        )
        

        print('*'*80)
        print(f'{tensor_def} tensor, dimension {len(tensor_dim)}')
        print('*'*80)
        print(f'Pivot-search architecture:     {info["pivot_search"]["architecture"]}')
        print(f'Threads per MPI rank:          {info["pivot_search"]["threads_per_rank"]}')
        print(f'MPI ranks per unfolding:       {info["pivot_search"]["pivot_ranks"]}')
        print(f'Superblock grid sizes:        {info["pivot_search"]["grids"]}')
        print(f'TT ranks:                     {ranks}')
        print(f'Pivot search time: {info["timings"]["pivot_search"]:.6f} s')
        print(f'PEID time:         {info["timings"]["core_construction"]:.6f} s')
        print('-'*80)
        core_times = info["core_construction"].get("dimension_times")
        if core_times is not None:
            print(f'Core times per dimension:     {np.round(core_times, 6)}')
        core_groups = info["core_construction"].get("core_group_sizes")
        if core_groups is not None:
            print(f'Core construction work count: {core_groups}')
            
        print('-'*80)
        print(f'Total time:                   {time_end - time_start}')
        print(f'Error:                        {err[0]:.4e}')
        print('*'*80, flush=True)
       
    
