import os

#This forces numpy to not thread, comment out for standard numpy
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI
import torch as tn
import tt_utils_virtual as tt_utils

import time
from functools import partial

from scipy.stats import multivariate_normal, norm
from scipy.optimize import minimize

def data_generation_logistic_federated(random_seed, d, K, lower_bound, upper_bound,
                                       binary=False, mixed=False):
    """
    生成 n=2000 个样本,随机分配到 K 个 site。
    binary=True:  全部 covariate 二值化
    mixed=True:   前一半 covariate 二值化,后一半保持连续(优先级高于 binary)
    都为 False:    全部连续
    """
    n = 2000
    true_beta = np.tile([-0.5, 0.5], int(np.ceil((d - 1) / 2)))[:d - 1]
    true_beta = np.concatenate(([-1], true_beta))

    Sigma_mat = np.full((d - 1, d - 1), 0.3)
    np.fill_diagonal(Sigma_mat, 1.0)

    np.random.seed(random_seed)
    # 先按多元正态生成 d-1 个 covariate
    Xs = multivariate_normal.rvs(mean=np.zeros(d - 1), cov=Sigma_mat, size=n)

    if mixed:
        # 前一半二值化(copula step),后一半保留为连续
        n_bin = (d - 1) // 2
        p = np.full(n_bin, 0.5)
        U = norm.cdf(Xs[:, :n_bin])
        Xs_bin = (U < p).astype(float)
        Xs = np.column_stack((Xs_bin, Xs[:, n_bin:]))
    elif binary:
        p = np.full(d - 1, 0.5)
        U = norm.cdf(Xs)
        Xs = (U < p).astype(float)
    # else: 保持连续

    Xs = np.column_stack((np.ones(n), Xs))
    prob_vec = np.exp(Xs @ true_beta) / (1 + np.exp(Xs @ true_beta))
    Ys = np.random.binomial(n=1, p=prob_vec, size=n)

    site_assignment = np.random.randint(0, K, size=n)
    sites = []
    for k in range(K):
        mask = site_assignment == k
        sites.append({'Xs': Xs[mask], 'Ys': Ys[mask], 'n_k': mask.sum()})

    n_test = 1000
    test_points = np.random.uniform(low=lower_bound, high=upper_bound, size=(n_test, d))
    return sites, test_points, n

def neg_loglike_site(beta, Xs_k, Ys_k):
    """Site 本地的 loss 之 mean(在 site 自己的样本上求平均)。"""
    temp_prod = Xs_k @ beta
    return (np.log1p(np.exp(temp_prod)) - Ys_k * temp_prod).mean()

def global_loss(beta, sites, total_n):
    """
    Server 把各 site 的 mean loss 按样本量加权,得到全局 mean loss:
        global_mean = sum_k (n_k / total_n) * local_mean_k
                    = sum_k local_sum_k / total_n
    与"先 sum 再除 total_n"在数学上完全等价。
    """
    return sum(s['n_k'] * neg_loglike_site(beta, s['Xs'], s['Ys']) for s in sites) / total_n

def federated_pooled_estimator(sites, total_n, initial, bounds):
    """Federated pooled MLE:每个 beta 查询时,server 让每个 site 算一下再加起来。"""
    res = minimize(
        fun=lambda beta: global_loss(beta, sites, total_n),
        x0=initial,
        method="L-BFGS-B",
        bounds=bounds)
    return res.x, res

def tensor_entry_logistic(indices, dim, Xs, YtX, grids_1d, chunk_size=4096):
    idx = np.asarray(indices, dtype=int)

    if idx.ndim == 1:
        idx = idx.reshape(1, -1)

    d = len(dim)
    num_entries = len(idx)

    vals = np.empty(num_entries, dtype=np.float64)

    for start in range(0, num_entries, chunk_size):
        end = min(start + chunk_size, num_entries)

        idx_chunk = idx[start:end]

        beta_mat = np.column_stack([
            grids_1d[j][idx_chunk[:, j]]
            for j in range(d)
        ])

        # Shape: (number of observations, chunk size)
        temp_prod = Xs @ beta_mat.T

        # Compute log(1 + exp(x)) in place
        np.logaddexp(0.0, temp_prod, out=temp_prod)

        vals[start:end] = (
            temp_prod.mean(axis=0)
            - (YtX @ beta_mat.T) / Xs.shape[0]
        )

    return vals

def neg_loglike_logistic(beta, Xs, Ys):
    temp_prod = Xs @ beta
    loss_vec = np.log1p(np.exp(temp_prod)) - Ys * temp_prod
    return loss_vec.mean()

def s_from_x(x, LL, UU):
    return (2.0 * x - (UU + LL)) / (UU - LL)

def orthonormal_legendre_basis(s, poly_order):
    s = np.asarray(s, dtype=float)
    P = np.empty(s.shape + (poly_order + 1,), dtype=float)
    P[..., 0] = 1.0
    P[..., 1] = s
    for n in range(1, poly_order):
        P[..., n + 1] = ((2 * n + 1) * s * P[..., n] - n * P[..., n - 1]) / (n + 1)
    scales = np.sqrt((2 * np.arange(poly_order + 1) + 1) / 2.0)
    return P * scales

def psi_at_scalar_t(t, LL, UU, poly_order):
    s_t = s_from_x(t, LL, UU)
    return orthonormal_legendre_basis(s_t, poly_order)

def edges_equal(a, b, n_intervals):
    return np.linspace(float(a), float(b), int(n_intervals) + 1)

def build_cache_for_coord(coord, a, b, coord_spec, coord_cache,
                          default_n_intervals, default_poly_order, default_gauss):
    spec = coord_spec.get(coord, {})

    # 1. edges
    if "edges" in spec and spec["edges"] is not None:
        edges = np.asarray(spec["edges"], dtype=float)
    else:
        n_intervals = int(spec.get("n_intervals", default_n_intervals))
        edges = edges_equal(a, b, n_intervals)
    n_intervals = edges.size - 1

    # 2. polynomial order
    poly_order = int(spec.get("poly_order", default_poly_order))

    # 3. gauss points per interval
    g = spec.get("gauss", default_gauss)
    if isinstance(g, int):
        gauss_list = [int(g)] * n_intervals
    else:
        gauss_list = list(g)

    # 4. build quadrature and projection info
    quads = []
    all_pts = []
    for i, m_i in enumerate(gauss_list):
        s_nodes, w_nodes = np.polynomial.legendre.leggauss(m_i)
        psi_nodes = orthonormal_legendre_basis(s_nodes, poly_order)
        proj = w_nodes[:, None] * psi_nodes
        quads.append({
            "nodes_s": s_nodes,
            "weights": w_nodes,
            "psi_nodes": psi_nodes,
            "proj": proj,
        })
        a_i, b_i = edges[i], edges[i + 1]
        x_nodes = 0.5 * (b_i - a_i) * s_nodes + 0.5 * (b_i + a_i)
        all_pts.append(x_nodes)

    grid_1d = np.concatenate(all_pts)

    # 5. prefix sums for indexing
    offsets = np.zeros(n_intervals + 1, dtype=int)
    offsets[1:] = np.cumsum(gauss_list)

    # 存入 cache
    coord_cache[coord] = {
        "edges": edges,
        "offsets": offsets,
        "quads": quads,
        "gauss_list": gauss_list,
        "poly_order": poly_order,
        "grid_1d": grid_1d,
        "n_points": int(np.sum(gauss_list)),
    }

def spectral_projection_local_matrix(t, coord, cores, coord_cache):
    cache = coord_cache[coord]
    edges = cache["edges"]
    offsets = cache["offsets"]
    quads = cache["quads"]
    poly_order = cache["poly_order"]

    # find interval once
    idx = int(np.searchsorted(edges, t, side="right") - 1)
    idx = max(0, min(idx, len(edges) - 2))
    LL, UU = edges[idx], edges[idx + 1]

    # slice all ranks at once: (rL, m, rR)
    start = int(offsets[idx])
    stop = int(offsets[idx + 1])
    block = cores[coord][:, start:stop, :]  # (r_left, m_nodes_in_interval, r_right)

    # interval-specific projection matrix: (m, poly_order)
    proj = quads[idx]["proj"]

    # coefficients: (r_left, poly_order, r_right)
    # multiply along the "m" (nodes) dimension
    coeffs = np.einsum('lmr,mp->lpr', block, proj, optimize=True)

    # basis at scalar t on [LL, UU]: (poly_order,)
    psi_t = psi_at_scalar_t(t, LL, UU, poly_order)

    # evaluate: contract over poly_order -> (r_left, r_right)
    values = np.einsum('lpr,p->lr', coeffs, psi_t, optimize=True)
    return values

def TT_approx_local(t, cores, my_rank, coord_cache):
    la = np.concatenate(([1], np.asarray(my_rank, dtype=int), [1]))
    d = len(my_rank) + 1

    # first core: (1, r1) -> row vector
    M0 = spectral_projection_local_matrix(t[0], 0, cores, coord_cache)
    res = M0[0, :]  # shape (r1,)

    # remaining cores: multiply (ri, ri+1) matrices sequentially
    for i in range(1, d):
        Mi = spectral_projection_local_matrix(t[i], i, cores, coord_cache)  # (r_i, r_{i+1})
        res = res @ Mi  # keeps collapsing to length r_{i+1}

    return np.float64(res.ravel()[0])

def TT_approx_local_federated(t, site_cores_list, site_ranks_list, site_weights, coord_cache):
    """
    Local-grid 版的 federated 函数评估。
    site_ranks_list: list of K rank vectors,每个对应一个 site 的 TT。
    """
    return sum(
        w * TT_approx_local(t, cores, ranks, coord_cache)
        for w, cores, ranks in zip(site_weights, site_cores_list, site_ranks_list)
    )

def rel_err(est, ref, eps=0):
    """Norm-based relative error, safe when some component of ref is ~0."""
    return np.linalg.norm(est - ref) / (np.linalg.norm(ref) + eps)

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
        val = 1/(np.sum(inds,axis = 1)+1)
        if grid_shape is not None:
            val = val.reshape(grid_shape)
        if use_torch:
            value = tn.from_numpy(val).float()
        else:
            value = val
    elif tensor_def == "Hilbert_Python":
        value = []
        for i in inds:
            value.append(1.0/(sum(i)+1))
            time.sleep(0.0001)
        
            
    
    else:
        raise ValueError(f"Unknown tensor_def: {tensor_def}")
    
    return value



def build_grids(d, lower_bound, upper_bound, n_intervals, gauss, poly_order=3):
    """Construct the logistic-regression parameter grids."""
    
    coord_cache = {}
    for c in range(d):
        build_cache_for_coord(
            c, float(lower_bound[c]), float(upper_bound[c]),
            {}, coord_cache, n_intervals, poly_order, gauss,
        )
    return coord_cache, [coord_cache[c]["grid_1d"] for c in range(d)]


def setup_logistic(comm, rank, dims, seed, n_intervals=20, gauss=4, sites_count=1):
    """Replicate the original single-site federated logistic experiment."""
    
    d = len(dims)
    if any(n != n_intervals * gauss for n in dims):
        raise ValueError("Logistic tensor mode sizes must equal n_intervals * gauss")
    lower = np.full(d, -5.0)
    upper = np.full(d, 5.0)
    coord_cache, grids = build_grids(d, lower, upper, n_intervals, gauss)

    if rank == 0:
        sites, _, total_n = data_generation_logistic_federated(
            random_seed=seed, d=d, K=sites_count,
            lower_bound=lower, upper_bound=upper, binary=True,
        )
        weights = [site["n_k"] / total_n for site in sites]
        initial = [0] * d
        bounds = list(zip(lower, upper))
        pooled_est, pooled_res = federated_pooled_estimator(
            sites, total_n, initial, bounds,
        )
        payload = (sites, weights, initial, bounds, pooled_est, pooled_res)
    else:
        payload = None

    sites, weights, initial, bounds, pooled_est, pooled_res = comm.bcast(payload, root=0)
    # The original script uses K=1, so the sole site's objective is the tensor.
    if sites_count != 1:
        raise ValueError("This driver currently supports only one logistic site (K=1)")
    Xs, Ys = sites[0]["Xs"], sites[0]["Ys"]
    entry = partial(tensor_entry_logistic, Xs=Xs, YtX=Ys @ Xs, grids_1d=grids)
    return entry, (sites, weights, initial, bounds, pooled_est, pooled_res, coord_cache)


def report_logistic(cores, context):
    """Compare the pooled and TT-based logistic estimators on rank zero."""
    from scipy.optimize import minimize
    
    sites, weights, initial, bounds, pooled_est, pooled_res, coord_cache = context
    ranks = [core.shape[2] for core in cores[:-1]]
    exact_at_gold = sum(
        w * neg_loglike_logistic(pooled_est, site["Xs"], site["Ys"])
        for w, site in zip(weights, sites)
    )
    tt_at_gold = TT_approx_local_federated(
        pooled_est, [cores], [ranks], weights, coord_cache,
    )
    start = MPI.Wtime()
    tt_res = minimize(
        fun=TT_approx_local_federated,
        x0=np.asarray(initial, dtype=float),
        args=([cores], [ranks], weights, coord_cache),
        method="L-BFGS-B", bounds=bounds,
    )
    elapsed = MPI.Wtime() - start
    print('-' * 80)
    print(f"Exact objective at gold:       {exact_at_gold:.12e}")
    print(f"TT objective at gold:          {tt_at_gold:.12e}")
    print(f"Gold-standard estimator:       {np.array2string(np.asarray(pooled_est), precision=6)}")
    print(f"TT-based estimator:            {np.array2string(tt_res.x, precision=6)}")
    print(f"Gold-standard objective:       {pooled_res.fun:.8e}")
    print(f"TT objective at TT estimate:    {tt_res.fun:.8e}")
    print(f"Estimator relative error:      {100.0 * rel_err(tt_res.x, pooled_est):.6e} %")
    print(f"TT minimization time:          {elapsed:.6f} s")
    print(f"Gold optimizer success:        {pooled_res.success}")
    print(f"TT optimizer success:          {tt_res.success}")


if __name__ == "__main__":
    import sys

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    mpin = int(sys.argv[1])
    if size != mpin:
        if rank == 0:
            print(f"ERROR: Expected {mpin} MPI ranks, but received {size}", flush=True)
        comm.Abort(1)

    # Choose "Logistic" or one of the original analytic tensor definitions.
    tensor_def = "Logistic"
    tensor_dim = [80] * 6
    sample_row = [0] * (len(tensor_dim) - 1)
    sample_col = [0] * (len(tensor_dim) - 1)
    min_ranks = [1] * (len(tensor_dim) - 1)
    max_ranks = [50] * (len(tensor_dim) - 1)
    tolerance = 1e-5
    random_seed = 2026
    nthreads = 1

    logistic_context = None
    if tensor_def == "Logistic":
        # Each mode has 4 Gaussian nodes per interval; e.g. 80 points -> 20 intervals.
        if len(set(tensor_dim)) != 1 or tensor_dim[0] % 4:
            raise ValueError("Logistic requires equal mode sizes divisible by 4")
        entry_fn, logistic_context = setup_logistic(
            comm, rank, tensor_dim, random_seed,
            n_intervals=tensor_dim[0] // 4, gauss=4, sites_count=1,
        )
    else:
        entry_fn = tensor_entry

    comm.Barrier()
    time_start = MPI.Wtime()
    cores, FI, FJ, info = tt_utils.tt_cross_os2_mpi2(
        entry_fn,
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
        process_grid=None,
        block_size=1,
        pivot_ranks=None,
        nthreads=nthreads,
        max_peid_ranks=mpin,
    )
    comm.Barrier()
    time_end = MPI.Wtime()

    if rank == 0:
        ranks = [core.shape[2] for core in cores[:-1]]
        err = tt_utils.sampled_error(entry_fn, tensor_dim, 100000, cores)
        print('*' * 80)
        print(f'{tensor_def} tensor, dimension {len(tensor_dim)}')
        print('*' * 80)
        print(f'Pivot-search architecture:     {info["pivot_search"]["architecture"]}')
        print(f'Threads per MPI rank:          {info["pivot_search"]["threads_per_rank"]}')
        print(f'MPI ranks per unfolding:       {info["pivot_search"]["pivot_ranks"]}')
        print(f'Superblock grid sizes:         {info["pivot_search"]["grids"]}')
        print(f'TT ranks:                     {ranks}')
        print(f'Pivot search time:            {info["timings"]["pivot_search"]:.6f} s')
        print(f'PEID time:                    {info["timings"]["core_construction"]:.6f} s')
        print('-' * 80)
        core_times = info["core_construction"].get("dimension_times")
        if core_times is not None:
            print(f'Core times per dimension:     {np.round(core_times, 6)}')
        core_groups = info["core_construction"].get("core_group_sizes")
        if core_groups is not None:
            print(f'Core construction work count: {core_groups}')
        print('-' * 80)
        print(f'Total time:                   {time_end - time_start}')
        print(f'Error:                        {err[0]:.4e}')
        if logistic_context is not None:
            report_logistic(cores, logistic_context)
        print('*' * 80, flush=True)
