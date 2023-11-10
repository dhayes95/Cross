#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:17:59 2023

@author: dphayes
"""

import numpy as np
import random




def recursive_Tk(A,I,J):
    """
    

    Parameters
    ----------
    A : Matrix
    I : row indices
    J : column indices

    Returns
    -------
    Tk : A[:,J]@np.linalg.pinv(A[I,J])

    Note: This function is used for the TT cross core construction
    """
    dim = A.shape
    Ilen = len(I)
    
    
    #Initalize with exact first
    Itemp = I[:1]
    Jtemp = J[:1]

    Tk = A[:,Jtemp]@np.linalg.pinv(A[Itemp,:][:,Jtemp])

    for i in range(1,Ilen):
        Itemp = I[:i]
        Jtemp = J[:i]
        
        
        
        ii = I[i]
        jj = J[i]
        
        
        
        Sk = np.reshape(Tk,[dim[0],i])@A[Itemp,:][:,jj] - A[:,jj]
        e1 = np.zeros([1,dim[0]])
        e1[0,ii]=1
        
        
        delta1 = (e1@Sk)
        if delta1==0:
            delta = 0
        else:
            delta = 1.0/delta1

        a = np.reshape(Tk,[dim[0],i]) - delta*np.outer(np.reshape(Sk,[dim[0],1]),np.reshape(Tk[ii,:],[1,Tk.shape[1]]))
        
        b = np.reshape(delta*Sk,[dim[0],1])

            
        Tk = np.concatenate((a,b),axis=1)

    
    return Tk


def Recursive_CUR(A,I,J,tol):
    """
    

    Parameters
    ----------
    A : Matrix
    I : row indices
    J : column indices
    tol : tolerance for delta parameter

    Returns
    -------
    Atilde : CUR

    """
    
    dim = A.shape

    Atilde = np.zeros(dim)

    for k in range(len(I)):
        E = A - Atilde
        deltainv = E[I[k],J[k]]
        
        
        if np.abs(deltainv) <= tol:
            break
        else: 
            Atilde = Atilde + (1.0/deltainv)*np.outer(E[:,J[k]],E[I[k],:])
           

    return Atilde




def Greedy_Pivot_Search(A,I,J,sample_size=3,maxiter=10,tol=1e-16):
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
    if len(I)==0 and len(J)==0:
        Asub = A
        Atildesub = np.zeros(dim)
        subdim = dim
        Irandom = random.sample(list(np.arange(dim[0])),sample_size)
        Jrandom = random.sample(list(np.arange(dim[1])),sample_size)
    else:
        #If non-empty then create Atilde recursively
        Atilde = Recursive_CUR(A,I,J,tol)
        
        #Only sample unused indices
        Asub = np.delete(np.delete(A,I,0),J,1)
        Atildesub = np.delete(np.delete(Atilde,I,0),J,1)
        subdim = Asub.shape

        #Get adjusted sample size
        sample_number = np.min([sample_size,subdim[0],subdim[1]])
        
        #Generate random skeleton to begin search
        Irandom = random.sample(list(np.arange(subdim[0])),sample_number)
        Jrandom = random.sample(list(np.arange(subdim[1])),sample_number)
     
    #Needed for tracking true index. Not needed if you search over whole matrix
    Row_track = np.delete(np.arange(dim[0]),I,0)
    Col_track = np.delete(np.arange(dim[1]),J,0)

    #Set up counter for iterations    
    iteration_count = 0
    
    #Initial pivot in skeleton error
    E = np.abs(Asub-Atildesub)
    i_star,j_star = np.unravel_index(np.argmax(E[Irandom,Jrandom]),subdim)

    #Main while loop
    while iteration_count < maxiter:

        i_star = np.argmax(E[:,j_star])
        j_star = np.argmax(E[i_star,:])

        #Check rook condition
        if all([E[i_star,j_star]>E[m,j_star] for m in range(subdim[0])]) \
            and all([E[i_star,j_star]>E[i_star,n] for n in range(subdim[1])]):
            break
        iteration_count+=1
    
    
    I_new = np.append(I,Row_track[i_star]).astype(int)
    J_new = np.append(J,Col_track[j_star]).astype(int)


    
    return I_new,J_new
