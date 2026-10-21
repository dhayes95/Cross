import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-numpy_thread",
    type=lambda x: x.lower() in ("true", "1", "yes"),
    default=True
)

args, remaining = parser.parse_known_args()

if not args.numpy_thread:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["BLIS_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI
import torch as tn
import tt_utils
import time


def tensor_entry_numpy(indices,dim):
    
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

    val = 1/(np.sum(inds,axis = 1)+1)
    if grid_shape is not None:
        val = val.reshape(grid_shape)
    if use_torch:
        value = tn.from_numpy(val).float()
    else:
        value = val

    
    return value

def tensor_entry_python(indices,dim):
    value = [1.0/(sum(i)+1) for i in indices]
    return value



if __name__ == "__main__":
    tensor_dim = [1000]*4

    ind_num = [1,10,100,1000,10000,100000,1000000,10000000]
    python_times = []
    numpy_times = []
    for indn in ind_num:
        sample_indices = np.random.randint(0,1000,size = (indn,4))
        t_start1 = time.perf_counter()
        vals = tensor_entry_python(sample_indices,tensor_dim)
        t_mid1 = time.perf_counter()
        val2 = tensor_entry_numpy(sample_indices,tensor_dim)
        t_end1 = time.perf_counter()
        python_times.append(t_mid1 - t_start1)
        numpy_times.append(t_end1 - t_mid1)
        if indn==1:
            order_p = np.nan
            order_n = np.nan
        else:
            order_p = np.log(python_times[-1]/python_times[-2])/np.log(10)
            order_n = np.log(numpy_times[-1]/numpy_times[-2])/np.log(10)

        print(f"Numpy time: {(t_end1 - t_mid1):2e} order: {order_n:2e}. Pythonic time: {(t_mid1 - t_start1):2e} order: {order_p:2e}. For {indn} indices")


