#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:05:02 2023

@author: dphayes
"""

import numpy as np
from multiprocessing import Pool
from joblib import Parallel, delayed 
from FunctionsMatrixLevel import *
from IndexingFunctions import *



def Unfolding(X_ten,n):
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
    
    new_dim = [np.prod(dim[:n+1]).astype(int),np.prod(dim[n+1::]).astype(int)]
    
    X_mat = np.reshape(X_ten,new_dim)
    
    return X_mat


def KSuperblock(X_ten,I,J,k):
    dim = np.shape(X_ten)
    I_dim = dim[0:k]
    J_dim = dim[k+2::]
    
    if k>0 and k<len(dim)-2:
        X_super = np.zeros([len(I),dim[k],dim[k+1],len(J)])
        
        
        for dim_i in range(len(I)):
            for dim_j in range(len(J)):
                I_lexo = np.unravel_index(I[dim_i],I_dim)
                J_lexo = np.unravel_index(J[dim_j],J_dim)
                
                for ind_1 in range(dim[k]):
                    for ind_2 in range(dim[k+1]):
                        full_index = I_lexo+tuple([ind_1]) + tuple([ind_2])+J_lexo
                        
                        X_super[dim_i,ind_1,ind_2,dim_j] = X_ten[full_index]
    elif k==0:
        X_super = np.zeros([1,dim[k],dim[k+1],len(J)])
        for dim_j in range(len(J)):
            J_lexo = np.unravel_index(J[dim_j],J_dim)
            for ind_1 in range(dim[k]):
                for ind_2 in range(dim[k+1]):
                    full_index = tuple([ind_1,ind_2])+J_lexo
                    X_super[0,ind_1,ind_2,dim_j] = X_ten[full_index]
    else:
        X_super = np.zeros([len(I),dim[k],dim[k+1],1])
        for dim_i in range(len(I)):
            I_lexo = np.unravel_index(I[dim_i],I_dim)
            for ind_1 in range(dim[k]):
                for ind_2 in range(dim[k+1]):
                    full_index = I_lexo + tuple([ind_1,ind_2])
                    X_super[dim_i,ind_1,ind_2,0] = X_ten[full_index]
        
    return X_super



def MatrixSuperblock(X_ten,I,J,k):
    
    if len(I)==0:
        I = [0]
    if len(J)==0:
        J = [0]
    
    
    X_super = KSuperblock(X_ten, I, J, k)
    
    Matrix_Superblock = Unfolding(np.transpose(X_super,[0,1,3,2]), 1)
        
    return Matrix_Superblock


def MatrixSuperblock_new(X_ten,I,J,k):
    """
    Combine KSuperblock and MatrixSuperblock
    Directly generate the superblock instead of a 4D tensor and then flatten
    Avoid using nested for loops, use vectorized operations, such as taking submatrices

    For example, when 0 < k < len(dim)-2
    X_mat = Unfolding(X_ten, k-1)
    Matrix_Superblock = X_mat(dim[k]*prod(dim[0:k])+I, dim[k+1]*prod(dim[k+2::])+J)
    It'll even better if we don't need to build the unfolding first, but directly take subtensors, but I don't know how that works with Python tuples
    """

def CrossInterpSingleItemSuper(A,Ir,Ic,FIr,FIc,k,Xsize):
    flag =0
    
    if k==0:
        Ii,Jj = Greedy_Pivot_Search(A,Ir[0],Ic[0])
        if Ii[-1] not in Ir[0] or Jj[-1] not in Ic[0]:
            Ir[0] = np.append(Ir[0],Ii[-1])
            Ic[0] = np.append(Ic[0],Jj[-1])
            fi,fj = Matrix_Super_Index_Conversion(Ir[0][-1], Ic[0][-1], [0], Ic[1], 0, Xsize)
            FIr[0] = np.append(FIr[0],fi)
            FIc[0] = np.append(FIc[0],fj)
        else:
            flag = 1
            
        
    elif k==len(Ir)-1:
        Ii,Jj = Greedy_Pivot_Search(A,Ir[k],Ic[k]) 
        if Ii[-1] not in Ir[k] or Jj[-1] not in Ic[k]:
            Ir[k] = np.append(Ir[k],Ii[-1])
            Ic[k] = np.append(Ic[k],Jj[-1])
            fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], Ir[-2], [0], k, Xsize)
            FIr[k] = np.append(FIr[k],fi)
            FIc[k] = np.append(FIc[k],fj)
        else:
            flag = 1
        
        
    else:
        Ii,Jj = Greedy_Pivot_Search(A,Ir[k],Ic[k])
        if Ii[-1] not in Ir[k] or Jj[-1] not in Ic[k]:
            Ir[k] = np.append(Ir[k],Ii[-1])
            Ic[k] = np.append(Ic[k],Jj[-1])
            fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], Ir[k-1], Ic[k+1], k, Xsize)
            #print(Ir[k][-1],Ic[k][-1],Ir[k-1],Ic[k+1],k,X.shape,fi,fj)
            FIr[k] = np.append(FIr[k],fi)
            FIc[k] = np.append(FIc[k],fj)
        else:
            flag = 1
    
    
    if flag==1:
        print("Duplicate occurred. Specified rank may not be obtained")
        
    
        
    return Ir,Ic,FIr,FIc



def Greedy_TT_Cross_Approx_Super(X,rs):
    
    Ir = list(np.zeros([len(X.shape)-1,1],dtype = int))
    Ic = list(np.zeros([len(X.shape)-1,1],dtype = int))
    
    FIr = list(np.zeros([len(X.shape)-1,1],dtype = int))
    FIc = list(np.zeros([len(X.shape)-1,1],dtype = int))
    

    for k in range(np.max(rs)-1):
        if k%2==0:
            for j in range(len(rs)):
                if rs[j]>1:
                    if j==0:
                        A = MatrixSuperblock(X, [0], Ic[1], 0)
                    elif j==len(rs)-1:
                        A = MatrixSuperblock(X, Ir[-2], [0], j)  
                    else:
                        A = MatrixSuperblock(X, Ir[j-1], Ic[j+1], j)
                        
                    Ir,Ic,FIr,FIc = CrossInterpSingleItemSuper(A, Ir, Ic, FIr,FIc, j,X.shape)
                    rs[j]-=1
                             
        else:
            for j in range(len(rs)-1,-1,-1):
                if rs[j]>1:
                    if j==0:
                        A = MatrixSuperblock(X, [0], Ic[1], 0)
                    elif j==len(rs)-1:
                        A = MatrixSuperblock(X, Ir[-2], [0], j)  
                    else:
                        A = MatrixSuperblock(X, Ir[j-1], Ic[j+1], j)
                    Ir,Ic,FIr,FIc = CrossInterpSingleItemSuper(A, Ir, Ic, FIr,FIc, j,X.shape)
                    rs[j]-=1


    return FIr,FIc


def Dimension_Parallel_TT_Cross_Super(X,rs):
    dim = X.shape 
    
    
    Ir = [[0] for i in range(len(rs))]
    Ic = [[0] for i in range(len(rs))]
    FIr = [[0] for i in range(len(rs))]
    FIc = [[0] for i in range(len(rs))]
    
    for j in range(np.max(rs)-1):
        returns = []
        iters = []
        for i in range(len(dim)-1):
            if rs[i]>1:
                if i==0:
                    A = MatrixSuperblock(X, [0], Ic[1], 0)
                elif i==len(rs)-1:
                    A = MatrixSuperblock(X, Ir[-2], [0], i)            
                else:
                    A = MatrixSuperblock(X, Ir[i-1], Ic[i+1], i)
                iters.append((A,Ir,Ic,FIr,FIc,i,dim))
        
        with Pool() as pool:
            for items in pool.starmap(CrossInterpSingleItemSuper,iters):
                returns.append(items)
        pool.close()
        pool.join()
        
        rs = [rs[i]-1 for i in range(len(dim)-1)]
        for i in range(len(iters)):
            Ir[iters[i][-2]] = returns[i][0][iters[i][-2]]
            Ic[iters[i][-2]] = returns[i][1][iters[i][-2]]
            FIr[iters[i][-2]] = returns[i][2][iters[i][-2]]
            FIc[iters[i][-2]] = returns[i][3][iters[i][-2]]
    
    return FIr,FIc




def Recursive_Core_Extract(X,Ir,Ic):
    dim = np.shape(X)
    if len(Ir)!=len(Ic):
        raise  ValueError("The row index and column index sets are not compatible length")
        
    cores = list(np.zeros(len(dim)))
    
    for i in range(len(dim)):
        
        if i==0:
            A = Unfolding(X,0)
            Tk = recursive_Tk(A, Ir[0], Ic[0])
            cores[i] = np.reshape(Tk,[1,dim[i],len(Ic[i])])
        elif i==len(dim)-1:
            
            cores[i] = np.zeros([len(Ir[i-1]),dim[-1],1])
            for j in range(dim[i]):
                for k in range(len(Ir[i-1])):
                    p = np.unravel_index(Ir[i-1][k], dim[:i])

                    cores[i][k,j,0] = X[tuple(p)+tuple([j])]
        else:
            cores[i]=np.zeros([len(Ir[i-1]),dim[i],len(Ic[i])])
            A = Unfolding(X,i)
            ind = []

            for j in range(len(Ir[i-1])):
                p = np.unravel_index(Ir[i-1][j],dim[:i])
                for k in range(dim[i]):
                    ind = np.append(ind,np.ravel_multi_index(tuple(p)+tuple([k]),dim[:i+1]))
            ind = ind.astype(int)
            Tk = recursive_Tk(A,Ir[i],Ic[i])
            cores[i] = np.reshape(Tk[ind,:],[len(Ir[i-1]),dim[i],len(Ic[i])])

    return cores


def Core_to_Tensor(cores):
    """
    

    Parameters
    ----------
    cores : a list of 3D tensors cores for the decomposition

    Returns
    -------
    X_approx : Tensor defined by the input cores
        

    """
    
    
    num_cores = len(cores)
    dim = [cores[i].shape[1] for i in range(num_cores)]
    #Get first set up to iterate through the rest of the dimensions
    X_approx = np.tensordot(cores[0],cores[1],axes = ((2),(0)))
    
    for i in range(2,num_cores):
        X_approx = np.tensordot(X_approx,cores[i],axes = ((i+1),(0)))
        
    X_approx = np.reshape(X_approx,tuple(dim))    
    return X_approx
