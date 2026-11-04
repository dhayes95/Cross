import numpy as np
import scipy as sp
import math
import random
try:
    from mpi4py import MPI
except Exception:
    MPI = None


def rank_cutoff(s_values, adap_tol):
    if len(s_values) == 0:
        return 1
    leading = s_values[0]
    if np.isclose(leading, 0.0):
        return 1
    norm_s = np.asarray(s_values) / leading
    return max(1, int(np.count_nonzero(norm_s >= adap_tol)))


def peid_par_rtl_adap(tensor_entry,FI,FJ,sample_row,sample_col,tensor_dim,adap_tol = None):


    trunc_ranks = [len(i) for i in FI]
    cores = []
    uj = []
    uj_tilde = []

    cores_adap = []
    uj_adap = []
    uj_tilde_adap = []
    trunc_ranks_adap = []

    n = len(trunc_ranks)
    superblock_time = 0

    uFI,uFJ = sample_nested(FI,FJ,sample_row,sample_col,tensor_dim,1)
    uK,uL = set_conversion(uFI,uFJ, tensor_dim)
    I = uK
    J = uL

    uFI = [np.array(i) for i in uFI]
    uFJ = [np.array(i) for i in uFJ]

    for i in range(n,-1,-1):
        if i == n:  
            mat = unfolding_submatrix(tensor_entry, uFI, uFJ, i-1, tensor_dim, 0).T
            
            q,r = sp.linalg.qr(mat,mode = "economic")
            U,s_values,_ = np.linalg.svd(r)

            if adap_tol is not None:
                cutoff = rank_cutoff(s_values, adap_tol)
                trunc_ranks[i-1] = cutoff
            
            u = q@U
            uj.append(u[:,:trunc_ranks[i-1]])
            cores.append(np.reshape(uj[-1],[1,tensor_dim[i],trunc_ranks[i-1]]))

        elif i==0:

            mat = unfolding_submatrix(tensor_entry, uFI, uFJ, 0, tensor_dim, 1).T
            A = sp.linalg.lstsq(uj[-1][J[0],:],mat,lapack_driver="gelsy")[0]
            cores.append(np.reshape(A,[trunc_ranks[i],tensor_dim[i],1]))

        else:
            
            mat = unfolding_submatrix(tensor_entry, uFI, uFJ, i-1, tensor_dim, 0).T
            q,r = sp.linalg.qr(mat,mode = "economic")
           
            U,s_values,_ = np.linalg.svd(r)
            if adap_tol is not None:
                cutoff = rank_cutoff(s_values, adap_tol)
                trunc_ranks[i-1] = cutoff
            
            u = q@U
            
            uj.append(u[:,:trunc_ranks[i-1]])

            uj_tilde = np.reshape(uj[-1],[len(uFJ[i]),tensor_dim[i]*trunc_ranks[i-1]])

            A = sp.linalg.lstsq(uj[-2][J[i],:],uj_tilde,lapack_driver="gelsy")[0]
            cores.append(np.reshape(A,[trunc_ranks[i],tensor_dim[i],trunc_ranks[i-1]]))


    core_list = [np.transpose(cores[i],[2,1,0]) for i in range(len(tensor_dim)-1,-1,-1)]
    return core_list


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


def peid_seq_ltr_adap(tensor_entry,FIt,FJt,sample_col,sample_row,tensor_dim,adap_tol = None):
   
    truncation_rank = [len(i) for i in FIt]

    cores = []
    cores_adap = []
    trunc_ranks_adap = []

    K_list = []
    L_list = []
    
    
    for i in range(len(tensor_dim)-1):
        L_list.append(tuple_sample([0 for _ in range(len(tensor_dim[i+1:]))],[j-1 for j in tensor_dim[i+1:]],FJt[i],sample_col[i],tensor_dim[i+1:]))
    for i in range(len(tensor_dim)-1):
        K_list.append(tuple_sample([0 for _ in range(len(tensor_dim[:i+1]))],[j-1 for j in tensor_dim[:i+1]],FIt[i],sample_row[i],tensor_dim[:i+1]))

    uFJ = [FJt[i]+L_list[i] for i in range(len(FJt))]
    uFI = [FIt[i] + K_list[i] for i in range(len(FIt))]
    
    for i in range(len(tensor_dim)):
        if i==0:
            
            B_test = unfolding_submatrix(tensor_entry,uFI,uFJ, i, tensor_dim, 1)

            q,r = sp.linalg.qr(B_test,mode = "economic")
            U,s_values,_ = np.linalg.svd(r)
            if adap_tol is not None:
                cutoff = rank_cutoff(s_values, adap_tol)
                truncation_rank[i] = cutoff
            u = q@U
            #fixed_rank = _bounded_rank(truncation_rank[i], u.shape[1])
            cores.append(np.reshape(u[:,:truncation_rank[i]],[1,tensor_dim[i],truncation_rank[i]]))

        elif i==len(tensor_dim)-1:

            A_test = unfolding_submatrix(tensor_entry, uFI,uFJ, i-1, tensor_dim, 0)
            F = np.zeros((len(uFI[i-1]),cores[-1].shape[-1]))
            for ir in range(len(uFI[i-1])):
                row  = cores[0][:,uFI[i-1][ir][0],:]
                for j in range(1,i):
                    row = row@cores[j][:,uFI[i-1][ir][j],:]
                F[ir,:] = row[0]
            
            B = np.linalg.lstsq(F,A_test,rcond = None)[0]
            cores.append(np.reshape(B,[cores[-1].shape[-1],tensor_dim[i],1]))

        else:
            
            fst_ind = np.repeat(uFI[i-1],len(uFJ[i])*tensor_dim[i],axis = 0)
            mid_ind = np.tile(np.repeat(np.arange(tensor_dim[i]),len(uFJ[i])),len(uFI[i-1]))
            lst_ind = np.tile(uFJ[i],(len(uFI[i-1])*tensor_dim[i],1))
            ind = np.column_stack([fst_ind,mid_ind,lst_ind])
            t = np.reshape(tensor_entry(ind,tensor_dim),[len(uFI[i-1]),tensor_dim[i]*len(uFJ[i])])

            F = np.zeros((len(uFI[i-1]),cores[-1].shape[-1]))
            for ir in range(len(uFI[i-1])):
                row  = cores[0][:,uFI[i-1][ir][0],:]
                for j in range(1,i):
                    row = row@cores[j][:,uFI[i-1][ir][j],:]
                F[ir,:] = row[0]
            
            B = np.reshape(np.linalg.lstsq(F,t,rcond = None)[0],[cores[-1].shape[-1]*tensor_dim[i],len(uFJ[i])])

            q,r = sp.linalg.qr(B,mode = "economic")
            U,s_values,_ = np.linalg.svd(r)
            if adap_tol is not None:
                cutoff = rank_cutoff(s_values, adap_tol)
                truncation_rank[i] = cutoff
            u = q@U
            #fixed_rank = _bounded_rank(truncation_rank[i], u.shape[1])
            cores.append(np.reshape(u[:,:truncation_rank[i]],[cores[-1].shape[-1],tensor_dim[i],truncation_rank[i]]))

    
    return cores


def peid_seq_rtl_adap(tensor_entry,FIt,FJt,sample_col,sample_row,tensor_dim,adap_tol = None):

    trunc_ranks = [len(i) for i in FIt]

    K_list = []
    L_list = []

    for i in range(len(tensor_dim)-1):
        L_list.append(tuple_sample([0 for _ in range(len(tensor_dim[i+1:]))],[j-1 for j in tensor_dim[i+1:]],FJt[i],sample_col[i],tensor_dim[i+1:]))
    for i in range(len(tensor_dim)-1):
        K_list.append(tuple_sample([0 for _ in range(len(tensor_dim[:i+1]))],[j-1 for j in tensor_dim[:i+1]],FIt[i],sample_row[i],tensor_dim[:i+1]))

    uFJ = [FJt[i]+L_list[i] for i in range(len(FJt))]
    uFI = [FIt[i] + K_list[i] for i in range(len(FIt))]

    core_rev_t = []
    for i in range(len(tensor_dim)-1,-1,-1):
        if i == len(tensor_dim)-1:
            mat = unfolding_submatrix(tensor_entry, uFI, uFJ, i-1, tensor_dim, 0).T
            Q,r = sp.linalg.qr(mat,mode = "economic")
            U,s_values,_ = np.linalg.svd(r)
            if adap_tol is not None:
                cutoff = rank_cutoff(s_values, adap_tol)
                trunc_ranks[i-1] = cutoff
            q = Q@U
            core_rev_t.append(np.reshape(q[:,:trunc_ranks[i-1]].T,[trunc_ranks[i-1],tensor_dim[i],1]))
        else:
            F = np.zeros((core_rev_t[0].shape[0],len(uFJ[i])))
            for j in range(len(uFJ[i])):
                ind = list(uFJ[i][j])
                col = core_rev_t[0][:,ind[0],:]
                for l in range(1,len(core_rev_t)):
                    col = col@core_rev_t[l][:,ind[l],:]
                F[:,[j]] = col

            mat = unfolding_submatrix(tensor_entry, uFI, uFJ, i, tensor_dim, 1)
            Ft = np.ascontiguousarray(F.T)
            matt = np.ascontiguousarray(mat.T)
            y = np.linalg.lstsq(Ft,matt,rcond = None)[0].T

            if i > 0:
                y = np.reshape(y,[len(uFI[i-1]),tensor_dim[i]*core_rev_t[0].shape[0]]).T
                Q,r = sp.linalg.qr(y,mode = "economic")
                U,s_values,_ = np.linalg.svd(r)
                if adap_tol is not None:
                    cutoff = rank_cutoff(s_values, adap_tol)
                    trunc_ranks[i-1] = cutoff
                q = Q@U
                core_rev_t.insert(0,np.reshape(q[:,:trunc_ranks[i-1]].T,[trunc_ranks[i-1],tensor_dim[i],core_rev_t[0].shape[0]]))
            else:
                core_rev_t.insert(0,np.reshape(y,[1,tensor_dim[0],core_rev_t[0].shape[0]]))

    return core_rev_t


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


def peid_par_ltr_adap_parallel(tensor_entry, FI, FJ, sample_row, sample_col, tensor_dim, comm=None, adap_tol=None, print_grid=False, return_info=False, max_peid_ranks=None):
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
    # Serial execution
    if MPI is None and comm is None:
        t0 = time.perf_counter()
        cores = peid_par_ltr_adap(tensor_entry, FI, FJ, sample_row, sample_col, tensor_dim, adap_tol=adap_tol)
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
    # Select number of ranks for PEID
    if max_peid_ranks is None:
        peid_size = world_size
    else:
        peid_size = min(int(max_peid_ranks), world_size)
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
            cores = peid_par_ltr_adap(tensor_entry, FI, FJ, sample_row, sample_col, tensor_dim, adap_tol=adap_tol)
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
    peid_comm = world_comm.Split(color=0 if active_peid else MPI.UNDEFINED, key=world_rank)
    cores = None
    info = None
    # Active PEID ranks only
    if active_peid:
        comm = peid_comm
        rank = comm.Get_rank()
        size = comm.Get_size()
        # One-dimensional case
        if d == 1:
            t0 = MPI.Wtime()
            if rank == 0:
                cores = peid_par_ltr_adap(tensor_entry, FI, FJ, sample_row, sample_col, tensor_dim, adap_tol=adap_tol)
            elapsed = MPI.Wtime() - t0
            elapsed = comm.allreduce(elapsed, op=MPI.MAX)
            info = { "core_group_sizes": [1], "core_groups": [[0]], "dimension_times": [elapsed], "dimension_active_times": [elapsed] }
        else:
            trunc_ranks = [len(i) for i in FI]
            n = len(trunc_ranks)
            # Sampling and index conversion
            if rank == 0:
                FI = [ [ [int(i) for i in FI[j][k]] for k in range(len(FI[j])) ] for j in range(len(FI)) ]
                FJ = [ [ [int(i) for i in FJ[j][k]] for k in range(len(FJ[j])) ] for j in range(len(FJ)) ]
                
                uFI, uFJ = sample_nested(FI, FJ, sample_row, sample_col, tensor_dim, 0)
                
                uK, uL = set_conversion(uFI, uFJ, tensor_dim)
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
            # Determine QR matrix sizes
            qr_shapes = []
            qr_weights = []
            for i in range(n):
                row_count = (tensor_dim[0] if i == 0 else len(uFI[i-1]) * tensor_dim[i])
                col_count = len(uFJ[i])
                qr_shapes.append((row_count, col_count))
                qr_weights.append(row_count)
            # Allocate PEID ranks to dimensions
            group_sizes = [1 for _ in range(d)]
            extra_ranks = size - d
            if extra_ranks > 0:
                weights = np.asarray(qr_weights, dtype=float)
                if np.allclose(weights.sum(), 0.0):
                    proportions = (np.ones(n, dtype=float) / n)
                else:
                    proportions = (weights / weights.sum())
                raw_extra = proportions * extra_ranks
                extra_alloc = np.floor(raw_extra).astype(int)
                remaining = (extra_ranks - int(np.sum(extra_alloc)))
                if remaining > 0:
                    for idx in np.argsort(raw_extra - extra_alloc)[::-1][:remaining]:
                        extra_alloc[idx] += 1
                for i in range(n):
                    group_sizes[i] += int(extra_alloc[i])
            core_groups = []
            start = 0
            for group_size in group_sizes:
                stop = start + group_size
                core_groups.append(list(range(start, stop)))
                start = stop
            info = { "core_group_sizes": group_sizes, "core_groups": core_groups }
            if print_grid and rank == 0:
                print("PEID core groups:", core_groups, flush=True)
            # Create per-core communicator
            core_index = next(i for i, ranks in enumerate(core_groups) if rank in ranks)
            group_roots = [ ranks[0] for ranks in core_groups ]
            subcomm = comm.Split(color=core_index, key=rank)
            subrank = subcomm.Get_rank()
            # Per-dimension timing
            core_start = MPI.Wtime()
            recv_wait = 0.0
            core = None
            # Last TT core
            if core_index == n:
                if subrank == 0:
                    t_recv = MPI.Wtime()
                    uj_prev = comm.recv(source=group_roots[core_index-1], tag=1000+core_index)
                    recv_wait += MPI.Wtime() - t_recv

                    mat = unfolding_submatrix(tensor_entry, uFI, uFJ, core_index-1, tensor_dim, 0)
                    
                    A = np.linalg.lstsq(uj_prev[I[core_index-1], :], mat, rcond=None)[0]
                    
                    core = np.reshape(A, [ uj_prev.shape[1], tensor_dim[core_index], 1 ])

            # First and interior TT cores
            else:
                if core_index == 0:
                    gap = np.linspace(0, tensor_dim[core_index], subcomm.Get_size()+1).astype(int)
                else:
                    gap = np.linspace(0, len(uFI[core_index-1]), subcomm.Get_size()+1).astype(int)
                
                mat_local = (unfolding_submatrix_slice(tensor_entry, range(gap[subrank], gap[subrank+1]), uFI, uFJ, core_index, tensor_dim, 1, "row"))
                
                q, r = qr_tsqr_local(mat_local, comm=subcomm, gather_q=True)
                
                if subrank == 0:
                    U, s_values, _ = np.linalg.svd(r)
                    if adap_tol is not None:
                        cutoff = rank_cutoff(s_values, adap_tol)
                        trunc_ranks[core_index] = cutoff
                    u = q @ U
                    uj = u[:, :trunc_ranks[core_index]]
                    # Send U_j before constructing this TT core.
                    # Nonblocking communication avoids serializing solves.
                    send_request = comm.isend(uj, dest=group_roots[core_index+1], tag=1000+core_index+1)
                    if core_index == 0:
                        core = np.reshape(uj, [ 1, tensor_dim[core_index], uj.shape[1] ])
                    else:
                        t_recv = MPI.Wtime()
                        uj_prev = comm.recv(source=group_roots[core_index-1], tag=1000+core_index)
                        recv_wait += (MPI.Wtime() - t_recv)
                        uj_tilde = np.reshape(uj, [ len(uFI[core_index-1]), tensor_dim[core_index] * uj.shape[1] ])
                        A = np.linalg.lstsq(uj_prev[I[core_index-1], :], uj_tilde, rcond=None)[0]
                        core = np.reshape(A, [ uj_prev.shape[1], tensor_dim[core_index], uj.shape[1] ])
                    # Ensure the outgoing U_j transfer completes.
                    send_request.wait()
            # Collect per-dimension timing
            core_elapsed_local = (MPI.Wtime() - core_start)
            core_active_local = (core_elapsed_local - recv_wait)
            core_elapsed = subcomm.allreduce(core_elapsed_local, op=MPI.MAX)
            core_active = subcomm.allreduce(core_active_local, op=MPI.MAX)
            # Gather cores and timing information
            core_result = (core_index, core, core_elapsed, core_active) if subrank == 0 else None
            core_results = comm.gather(core_result, root=0)
            dimension_times = None
            dimension_active_times = None
            if rank == 0:
                ordered = sorted((result for result in core_results if result is not None), key=lambda x: x[0])
                cores = [ item[1] for item in ordered ]
                dimension_times = [ item[2] for item in ordered ]
                dimension_active_times = [ item[3] for item in ordered ]
            dimension_times = comm.bcast(dimension_times, root=0)
            dimension_active_times = comm.bcast(dimension_active_times, root=0)
            info["dimension_times"] = (dimension_times)
            info["dimension_active_times"] = (dimension_active_times)
            subcomm.Free()
        peid_comm.Free()
    # Return metadata to all original MPI ranks
    # Rank 0 belongs to the active PEID communicator.
    # Only metadata is broadcast; TT cores remain
    # on world rank 0.
    info = world_comm.bcast(info if world_rank == 0 else None, root=0)
    info["peid_ranks"] = peid_size
    info["inactive_ranks"] = list(range(peid_size, world_size))
    return (cores, info) if return_info else cores

"""
**********************************************************************
Functions from here on are just to make the file self contained
**********************************************************************
"""

def tuple_sample(low,high,unavail,sample_size,dim):

    high = [int(i) for i in high]
    high_end = Large_ravel(high,dim)
    low_end = Large_ravel(low,dim)
    rav_unavail = [Large_ravel(i,dim) for i in unavail]
    samples = cheap_sample(low_end, high_end,rav_unavail, sample_size)
    sample_tuples = [list(Large_unravel(i,dim)) for i in samples]
    
    return sample_tuples

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


# -----------------------------------------------------------------------------
# Local helper functions (no project-specific module imports required)
# -----------------------------------------------------------------------------

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
