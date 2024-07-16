import numpy as np
import time
from MatrixFunctionsNew import *
from IndexingFunctions import *
import concurrent.futures



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



def MatrixSuperblockUpdate(X_ten,I_inp,J_inp,k):
    dim = X_ten.shape
    I_dim = dim[0:k]
    J_dim = dim[k+2::]
    inds = []
    indssub = []
    
    time1 = time.time()
    if k==0:
        I = [0]
        J = J_inp
    elif k==len(dim)-2:
        I = I_inp
        J = [0]
    else:
        I = I_inp
        J = J_inp
    
    
    if k==0:
        size = [dim[0],dim[1]*len(J)]
    elif k==len(dim)-1:
        size = [len(I)*dim[-2],dim[-1]]
    else:
        size = [len(I)*dim[k],dim[k+1]*len(J)]
        
    reshaping = np.arange(len(dim))
    reshaping = np.append(np.delete(reshaping,[k+1]),k+1)
        
    Matrix_Superblock = np.zeros(size)
    
    for dims in range(len(I_dim)):
        inds.append([])
        indssub.append([])
        
    inds.append(np.arange(dim[k]))
    inds.append(np.arange(dim[k+1]))
    indssub.append(np.arange(dim[k]))
    indssub.append(np.arange(dim[k+1]))
    
    for dims in range(len(J_dim)):
        inds.append([])
        indssub.append([])
        

    
    for dim_i in range(len(I)):
        I_lexo = np.unravel_index(I[dim_i],I_dim)
        for dims in range(len(I_dim)):

            inds[dims].append(int(I_lexo[dims]))
            indssub[dims].append(int(dim_i))
            
    for dim_j in range(len(J)):
        J_lexo = np.unravel_index(J[dim_j],J_dim)

        for dims in range(-1,-len(J_dim)-1,-1):

            inds[dims].append(int(J_lexo[dims]))
            indssub[dims].append(int(dim_j))
    
    
    for dim_i in range(len(I)):
        for dim_j in range(len(J)):
            slicing = [[inds[i][dim_i]] if i<k else [inds[i][dim_j]] if i>=k+2 else np.arange(dim[i]) for i in range(len(dim))]
            Matrix_Superblock[dim_i*dim[k]:(dim_i+1)*dim[k],dim_j*dim[k+1]:(dim_j+1)*dim[k+1]] = np.reshape(X_ten[np.ix_(*slicing)],[dim[k],dim[k+1]])

    
    return Matrix_Superblock
            


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

def CrossInterpSingleItemSuperPiv(A,Ir,Ic,FIr,FIc,k,Xsize):
    flag =0
    
    if k==0:
        Ii,Jj,p = Greedy_Pivot_Search_Piv(A,Ir[0],Ic[0])
        if Ii[-1] not in Ir[0] or Jj[-1] not in Ic[0]:
            Ir[0] = np.append(Ir[0],Ii[-1]).astype(int)
            Ic[0] = np.append(Ic[0],Jj[-1]).astype(int)
            #print(Ir[0],Ic[0])
            if len(Ic[1])>0:
                fi,fj = Matrix_Super_Index_Conversion(Ir[0][-1], Ic[0][-1], [0], Ic[1], 0, Xsize)
            else:
                fi,fj = Matrix_Super_Index_Conversion(Ir[0][-1], Ic[0][-1], [0], [0], 0, Xsize)
            FIr[0] = np.append(FIr[0],fi).astype(int)
            FIc[0] = np.append(FIc[0],fj).astype(int)
        else:
            flag = 1
            
        
    elif k==len(Xsize)-2:
        Ii,Jj,p = Greedy_Pivot_Search_Piv(A,Ir[k],Ic[k]) 
        if Ii[-1] not in Ir[k] or Jj[-1] not in Ic[k]:
            Ir[k] = np.append(Ir[k],Ii[-1]).astype(int)
            Ic[k] = np.append(Ic[k],Jj[-1]).astype(int)
            if len(Ir[-2])>0:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], Ir[-2], [0], k, Xsize)
            else:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], [0], [0], k, Xsize)
            FIr[k] = np.append(FIr[k],fi).astype(int)
            FIc[k] = np.append(FIc[k],fj).astype(int)
        else:
            flag = 1
        
        
    else:
        Ii,Jj,p = Greedy_Pivot_Search_Piv(A,Ir[k],Ic[k])
        if Ii[-1] not in Ir[k] or Jj[-1] not in Ic[k]:
            Ir[k] = np.append(Ir[k],Ii[-1]).astype(int)
            Ic[k] = np.append(Ic[k],Jj[-1]).astype(int)
            if len(Ir[k-1])>0 and len(Ic[k+1])>0:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], Ir[k-1], Ic[k+1], k, Xsize)
            elif len(Ir[k-1])>0 and len(Ic[k+1])==0:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], Ir[k-1], [0], k, Xsize)
            elif len(Ir[k-1])==0 and len(Ic[k+1])>0:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], [0], Ic[k+1], k, Xsize)
            else:
                fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], [0], [0], k, Xsize)
            #print(Ir[k][-1],Ic[k][-1],Ir[k-1],Ic[k+1],k,X.shape,fi,fj)
            FIr[k] = np.append(FIr[k],fi).astype(int)
            FIc[k] = np.append(FIc[k],fj).astype(int)
        else:
            flag = 1
    
    
    if flag==1:
        print("Duplicate occurred. Specified rank may not be obtained")
        
    
        
    return Ir,Ic,FIr,FIc,p


def CrossInterpSingleItemSuper_Update(A,Atilde,Ir,Ic,FIr,FIc,k,Xsize):
    flag =0
    
    if k==0:
        Ii,Jj = Greedy_Pivot_Search_update(A,Atilde,Ir[0],Ic[0])
        if Ii[-1] not in Ir[0] or Jj[-1] not in Ic[0]:
            Ir[0] = np.append(Ir[0],Ii[-1])
            Ic[0] = np.append(Ic[0],Jj[-1])
            fi,fj = Matrix_Super_Index_Conversion(Ir[0][-1], Ic[0][-1], [0], Ic[1], 0, Xsize)
            FIr[0] = np.append(FIr[0],fi)
            FIc[0] = np.append(FIc[0],fj)
        else:
            flag = 1
            
        
    elif k==len(Ir)-1:
        Ii,Jj = Greedy_Pivot_Search_update(A,Atilde,Ir[k],Ic[k]) 
        if Ii[-1] not in Ir[k] or Jj[-1] not in Ic[k]:
            Ir[k] = np.append(Ir[k],Ii[-1])
            Ic[k] = np.append(Ic[k],Jj[-1])
            fi,fj = Matrix_Super_Index_Conversion(Ir[k][-1], Ic[k][-1], Ir[-2], [0], k, Xsize)
            FIr[k] = np.append(FIr[k],fi)
            FIc[k] = np.append(FIc[k],fj)
        else:
            flag = 1
        
        
    else:
        Ii,Jj = Greedy_Pivot_Search_update(A,Atilde,Ir[k],Ic[k])
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



def Greedy_TT_Cross_Approx_Super_Update(X,rs):
    
    
    FIr = [[np.unravel_index(np.argmax(Unfolding(X,i)),Unfolding(X,i).shape)[0]] for i in range(len(X.shape)-1)]
    FIc = [[np.unravel_index(np.argmax(Unfolding(X,i)),Unfolding(X,i).shape)[1]] for i in range(len(X.shape)-1)]
    
    Ir = [Full_To_Super(FIr[i], FIc[i], i, X.shape)[0] for i in range(len(X.shape)-1)]
    Ic = [Full_To_Super(FIr[i], FIc[i], i, X.shape)[1] for i in range(len(X.shape)-1)]
    
        
    
    for k in range(np.max(rs)-1):
        #ttt1 = time.time()
        if k%2==0:
            for j in range(len(rs)):
                if rs[j]>1:
                    
                    if j==0:
                        A = MatrixSuperblockUpdate(X, [0], Ic[1], 0)

                    elif j==len(rs)-1:

                        A = MatrixSuperblockUpdate(X, Ir[-2], [0], j)  


                    else:

                        A = MatrixSuperblockUpdate(X, Ir[j-1], Ic[j+1], j)
   
                    Ir,Ic,FIr,FIc,p = CrossInterpSingleItemSuperPiv(A, Ir, Ic, FIr,FIc, j,X.shape)
                    rs[j]-=1

                    
                              
        else:
            for j in range(len(rs)-1,-1,-1):
                if rs[j]>1:
                    if j==0:
                        A = MatrixSuperblockUpdate(X, [0], Ic[1], 0)

                    elif j==len(rs)-1:
                        A = MatrixSuperblockUpdate(X, Ir[-2], [0], j) 

                    else:
                        A = MatrixSuperblockUpdate(X, Ir[j-1], Ic[j+1], j)

                        
                    Ir,Ic,FIr,FIc,p = CrossInterpSingleItemSuperPiv(A, Ir, Ic, FIr,FIc, j,X.shape)
                    rs[j]-=1

    return FIr,FIc

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


def Greedy_TT_Cross_Single_Super(X,Ir,Ic,FIr,FIc):
    dim = X.shape
    piv = []

    
    for j in range(len(dim)-1):
        maxdim = [np.prod(dim[:j+1]),np.prod(dim[j+1:])]
        if len(Ir[j])==maxdim[0] or len(Ic[j])==maxdim[1]:
            piv.append(0)
        else:
            if j==0:
                if len(Ic[1])>0:
                    A = MatrixSuperblockUpdate(X, [0], Ic[1], 0)
                else:
                    A = MatrixSuperblockUpdate(X, [0], [0], 0)
            elif j==len(dim)-2:
                if len(Ir[-2])>0:
                    A = MatrixSuperblockUpdate(X, Ir[-2], [0], j)  
                else:
                    A = MatrixSuperblockUpdate(X, [0], [0], j)
            else:
                if len(Ir[j-1])>0 and Ic[j+1]>0:
                    A = MatrixSuperblockUpdate(X, Ir[j-1], Ic[j+1], j)
                elif len(Ir[j-1])>0 and len(Ic[j+1])==0:
                    A = MatrixSuperblockUpdate(X, Ir[j-1], [0], j)
                elif len(Ir[j-1])==0 and len(Ic[j+1])>0:
                    A = MatrixSuperblockUpdate(X, [0], Ic[j+1], j)
                else:
                    A = MatrixSuperblockUpdate(X, [0], [0], j)
                            
            Ir,Ic,FIr,FIc,p = CrossInterpSingleItemSuperPiv(A, Ir, Ic, FIr,FIc, j,X.shape)
            piv.append(p)           
                             


    return Ir,Ic,FIr,FIc,piv

def Subhelp(ins):
    
    internalstart=time.time()
    Ir,Ic,FIr,FIc,p = Greedy_TT_Cross_Single_Super(ins[0], ins[1], ins[2], ins[3], ins[4])
    internalend = time.time()
    return Ir,Ic,FIr,FIc,internalend-internalstart,ins[5],p

def Subtensor_Parallel_TT_Cross(X,rs,partitions):
    
    dimX = X.shape
    nS = np.prod(partitions)
    Ir_loc = [[[] for j in range(len(dimX)-1)] for i in range(nS)]
    Ic_loc = [[[] for j in range(len(dimX)-1)]for i in range(nS)]
    FIr_loc = [[[] for j in range(len(dimX)-1)] for i in range(nS)]
    FIc_loc = [[[] for j in range(len(dimX)-1)]for i in range(nS)]
    FIr = [[] for j in range(len(dimX)-1)]
    FIc = [[] for j in range(len(dimX)-1)]
    Xsubs,shifting,bounds = Subtensor_Split(X, partitions)

    for _ in range(rs):
        
        iters = []
        results = []
        pivs = []
        pivsel = []
        rowc = []
        colc = []

        for i in range(nS):
            iters.append((Xsubs[i],Ir_loc[i],Ic_loc[i],FIr_loc[i],FIc_loc[i],i))
        t2 = time.time()
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            info = executor.map(Subhelp,iters)
            
            for item in info:
                results.append(item)
                pivs.append(item[-1])


        pivs = np.reshape(pivs,[nS,len(dimX)-1]) 
        t3 = time.time()
        for j in range(len(dimX)-1):       
            pivsel.append(results[np.argmax(pivs[:,j])][-2])
            rr,cc = Subtensor_Comm(pivsel[j], shifting, j)
            rowc.append(rr)
            colc.append(cc)
        
        

        for j in range(len(dimX)-1):
            subind = [results[pivsel[j]][0][j][-1],results[pivsel[j]][1][j][-1]]
            if j==0:
                fix_ind = SuperSubblock_to_Full(subind,[0],results[pivsel[j]][1][j+1],shifting[pivsel[j]],Xsubs[pivsel[j]].shape,dimX,j)
            elif j==len(dimX)-2:
                fix_ind = SuperSubblock_to_Full(subind,results[pivsel[j]][0][j-1],[0],shifting[pivsel[j]],Xsubs[pivsel[j]].shape,dimX,j)
            else:
                fix_ind = SuperSubblock_to_Full(subind,results[pivsel[j]][0][j-1],results[pivsel[j]][1][j+1],shifting[pivsel[j]],Xsubs[pivsel[j]].shape,dimX,j)

            FIr[j].append(fix_ind[0])
            FIc[j].append(fix_ind[1])

        for j in range(nS):
            for i in range(len(dimX)-1):
                if j in rowc[i]:
                    Ir_loc[j][i] =np.append(Ir_loc[j][i],results[pivsel[i]][0][i][-1]).astype(int)
                    FIr_loc[j][i] = np.append(FIr_loc[j][i],results[pivsel[i]][2][i][-1]).astype(int)

                if j in colc[i]:
                    Ic_loc[j][i] = np.append(Ic_loc[j][i],results[pivsel[i]][1][i][-1]).astype(int)
                    FIc_loc[j][i] = np.append(FIc_loc[j][i],results[pivsel[i]][3][i][-1]).astype(int)
 
        
        
    
    
    return FIr,FIc



def helper(ins):
    
    Ir,Ic,FIr,FIc = CrossInterpSingleItemSuper(ins[0], ins[1], ins[2], ins[3], ins[4], ins[5], ins[6])
    
    return Ir,Ic,FIr,FIc

def Dimension_Parallel_TT_Cross_Super_Update(X,rs):
    dim = X.shape 
    
    
    Ir = [[0] for i in range(len(rs))]
    Ic = [[0] for i in range(len(rs))]
    FIr = [[0] for i in range(len(rs))]
    FIc = [[0] for i in range(len(rs))]
    
    for j in range(np.max(rs)-1):
        returns = []
        iters = []
        dims = []
        #t1=time.time()
        As = []
        #print(Ir,Ic)
        for i in range(len(dim)-1):
            if rs[i]>1:
                timet1 = time.perf_counter()
                if i==0:

                    A = MatrixSuperblockUpdate(X, [0], Ic[1], 0)

                elif i==len(rs)-1:

                    A = MatrixSuperblockUpdate(X, Ir[-2], [0], i) 

                else:
                    A = MatrixSuperblockUpdate(X, Ir[i-1], Ic[i+1], i)

            
                iters.append((A,Ir,Ic,FIr,FIc,i,dim))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            info = executor.map(helper,iters)
            
            for item in info:
                returns.append(item)

        
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
        
    selections = [list(np.unravel_index(i,partitions)) for i in range(np.prod(partitions))]
    single_indexing = [np.array_split(np.arange(dim[i]),partitions[i]) for i in range(len(partitions))]
    
    indexing = [[single_indexing[i][selections[j][i]] for i in range(len(partitions))] for j in range(len(selections))]
    
    Bounds = [[[indexing[j][i][0],indexing[j][i][-1]+1] for i in range(len(partitions))] for j in range(len(selections))]
    
    Subtensor = [X[np.ix_(*indexing[i])] for i in range(len(selections))]
       
    Shift_index = [[indexing[j][i][0] for i in range(len(partitions))] for j in range(len(selections))]
    
    
    
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


