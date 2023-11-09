#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:28:21 2023

@author: dphayes
"""



from IndexingFunctions import *
from FunctionsMatrixLevel import *
from FunctionsTensorLevel import *
from TensorFunctionBankDan import *
import time



if __name__ =="__main__":
    
    
    #Tensor input
    X = np.random.rand(72,72,72,72,72)
    print("Tensor built")
    
    
    #Compute row and column indices both in parallel(dimension) and serial
    t1 = time.perf_counter()
    par_I_row,par_I_col = Dimension_Parallel_TT_Cross_Super(X, [5,5,5,5])
    t2 = time.perf_counter()
    print("Parallel time: {}".format(t2-t1))
    t3 = time.perf_counter()
    ser_I_row,ser_I_col = Greedy_TT_Cross_Approx(X, [5,5,5,5])
    t4 = time.perf_counter()
    print("Serial time: {}".format(t4-t3))
    
    """
    #Construct cores
    par_cores = Recursive_Core_Extract(X, par_I_row, par_I_col)
    ser_cores = Recursive_Core_Extract(X, ser_I_row, ser_I_col)
    
    #Build approximations
    par_X_approx = Core_to_Tensor(par_cores)
    ser_X_approx = Core_to_Tensor(ser_cores)
    
    #Print errors
    print("Parallel error: {}".format(np.linalg.norm(X - par_X_approx)/np.linalg.norm(X)))
    print("Serial error: {}:".format(np.linalg.norm(X - ser_X_approx)/np.linalg.norm(X)))
    """