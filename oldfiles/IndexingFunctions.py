#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:05:08 2023

@author: dphayes
"""

import numpy as np
import math




def Super_to_Unfold_Index(ind_row,ind_col,I,J,k,dim):
    """
    

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
        Col_index = J[ind_col%len(J)] + math.floor(ind_col/len(J))*np.prod(dim[k+2::])
    elif k==len(dim)-2:
        Row_index = dim[k]*I[math.floor(ind_row/dim[k])] + ind_row%dim[k]
        Col_index = ind_col
    else:
        Row_index = dim[k]*I[math.floor(ind_row/dim[k])] + ind_row%dim[k]
        Col_index = J[ind_col%len(J)] + math.floor(ind_col/len(J))*np.prod(dim[k+2::])
    
    return Row_index, Col_index


def Matrix_Super_Index_Conversion(ind_row,ind_col,I,J,k,dim):
    dim_matrix = [dim[k]*len(I),dim[k+1]*len(J)]
    index_shift = np.reshape(np.reshape(np.arange(dim_matrix[1]),[dim[k+1],len(J)]).T,[1,dim[k+1]*len(J)])
    
    fix_i,fix_j = Super_to_Unfold_Index(ind_row, index_shift[0][ind_col], I, J, k, dim)
    
    return fix_i,fix_j



def SubSuper_to_Full(subindex,I,J,shifting,dimsub,dimfull,k):
    """
    

    Parameters
    ----------
    subindex : Index in subtensor superblock
    I : row index set of lower dimension I_\leq{k-1} of subtensor
    J : column index set of higher dimension I_>{k+1} of subtensor
    shifting : shifting of subtensor in full tensor
    dimsub : dimension of subtensor
    dimfull : dimension of full tensor
    k : dimension of unfolding

    Returns
    -------
    Row_index: adjusted row index for full k-unfolding of X
    Col_index: adjusted column index for full k-unfolding of X

    """
    
    Subunfold = Super_to_Unfold_Index(subindex[0], subindex[1], I, J, k, dimsub)
    
    subfull = np.unravel_index(np.ravel_multi_index(Subunfold, [np.prod(dimsub[:k+1]),np.prod(dimsub[k+1::])]),dimsub)
    
    Shiftedfull = list(np.array(subfull)+np.array(shifting))
    
    Row_index,Col_index = np.unravel_index(np.ravel_multi_index(Shiftedfull,dimfull),[np.prod(dimfull[:k+1]),np.prod(dimfull[k+1::])])
    
    return Row_index, Col_index


