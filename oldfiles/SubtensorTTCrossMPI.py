import numpy as np
import math
import time
import random
from mpi4py import MPI
import sys


def Local_subtensor_construction(bounds,sub_dim):

    #Computes the local subtensors:
    # inputs do not need to change, only change the function below in line 13 and 19.
    subtensor = np.fromfunction(lambda i0,i1,i2: (1)/(1+i0+bounds[0][0]+i1+bounds[1][0]+i2+bounds[2][0]),sub_dim)
    
    return subtensor

def tensor_entry(index):

    value = 1/(sum(index)+1)

    return value


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
        
    selections = [list(np.unravel_index(i,partitions)) for i in range(np.prod(partitions))]
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
    
    ravel_index = sum([int(sum([int(index[i])*int(math.prod(dim[i+1:])) for i in range(len(dim)-1)])),index[-1]])
    
    return ravel_index

def Large_unravel(index,dim):
    
    unravel_index = []
    
    for i in range(len(dim)):
        unravel_index.append(index//math.prod(dim[i+1:]))
        index -= int(unravel_index[-1])*int(math.prod(dim[i+1:]))
    return tuple(unravel_index)

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


def MPICrossInterpSingleItemSuperPiv(A,Atilde,Ir,Ic,FIr,FIc,k,Xsize):
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

def MPIGreedy_Search_Piv(A,Atilde,I,J,sample_size=4,maxiter=10,tol=0):
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
            ind_i,ind_j = Large_unravel(np.argmax(np.abs((A - Atilde)[:,J_out])),[dim[0],len(J_out)])
            j_star = J_out[ind_j]
            I_new = I
            J_new = np.append(J,j_star)
            p = (A - Atilde)[ind_i,ind_j]
            p_sign = 1


        elif len(J)==dim[1] and len(I)<dim[0]:
            I_out = np.delete(np.arange(dim[0]),I)
            ind_i,ind_j  =Large_unravel(np.argmax(np.abs((A - Atilde)[I_out,:])),[len(I_out),dim[1]])
            i_star = I_out[ind_i]
            I_new = np.append(I,i_star)
            J_new = J
            p = (A - Atilde)[ind_i,ind_j]
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


def Matrix_Super_Index_Conversion(ind_row,ind_col,I,J,k,dim):
    dim_matrix = [dim[k]*len(I),dim[k+1]*len(J)]
    index_shift = np.reshape(np.reshape(np.arange(dim_matrix[1]),[dim[k+1],len(J)]).T,[1,dim[k+1]*len(J)])
    
    fix_i,fix_j = Super_to_Unfold_Index(ind_row, index_shift[0][ind_col], I, J, k, dim)
    
    return int(fix_i),int(fix_j)


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
        Col_index = J[ind_col%len(J)] + math.floor(ind_col/len(J))*math.prod(dim[k+2::])
    elif k==len(dim)-2:
        Row_index = dim[k]*I[math.floor(ind_row/dim[k])] + ind_row%dim[k]
        Col_index = ind_col
    else:
        Row_index = dim[k]*I[math.floor(ind_row/dim[k])] + ind_row%dim[k]
        Col_index = int(J[ind_col%len(J)]) + math.floor(ind_col/len(J))*math.prod(dim[k+2::])
    
    return Row_index, Col_index


def Subtensor_Processor_Build(partitions,dim):
    selections = [list(np.unravel_index(i,partitions)) for i in range(math.prod(partitions))]
    single_indexing = [np.array_split(np.arange(dim[i]),partitions[i]) for i in range(len(partitions))]
    
    indexing = [[single_indexing[i][selections[j][i]] for i in range(len(partitions))] for j in range(len(selections))]
    
    Bounds = [[[indexing[j][i][0],indexing[j][i][-1]+1] for i in range(len(partitions))] for j in range(len(selections))]
    
    #Subtensor = [X[np.ix_(*indexing[i])] for i in range(len(selections))]
       
    Shift_index = [[indexing[j][i][0] for i in range(len(partitions))] for j in range(len(selections))]
    return Shift_index,Bounds

def MPI_Tk_Index(full_index,shifting,bounds,k,tensor_dim):
    
    unfold_dim = [math.prod(tensor_dim[:k+1]),math.prod(tensor_dim[k+1:])]
    
    tensor_index = Large_unravel(Large_ravel(full_index,unfold_dim), tensor_dim)
    rank_index_search = [[(tensor_index[i]-shifting[j][i])//(bounds[j][i][1]-bounds[j][i][0]) for i in range(len(tensor_dim))] for j in range(len(shifting))]
    
    rank_index = np.argmin(np.sum(np.abs(rank_index_search),axis = 1))
    
    rank_dim = [bounds[rank_index][i][1]-bounds[rank_index][i][0] for i in range(len(tensor_dim))]
    sub_unfold_dim = [math.prod(rank_dim[:k+1]),math.prod(rank_dim[k+1:])]
    
    row_index,col_index = Large_unravel(Large_ravel([tensor_index[i] - shifting[rank_index][i] for i in range(len(tensor_dim))],rank_dim),sub_unfold_dim)
    row,col = Subtensor_Comm(rank_index, shifting, k)
    
    row_loc = [row_index,row]
    col_loc = [col_index,col]
    
    return rank_index,row_loc,col_loc


def MPI_Recursive_Tk(col,I,tol = 0):
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

def Core_to_Tensor_Value(cores,index):
    
    start = cores[0][:,index[0],:]
    for i in range(1,len(cores)):
        start = start@cores[i][:,index[i],:]
    value = start[0][0]
    return value



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if __name__ == "__main__":

    #Collect the inputs
    dims = list(map(int,sys.argv[1].split(",")))
    rs = list(map(int,sys.argv[2].split(",")))
    processor_partition = list(map(int,sys.argv[3].split(",")))
    err_sample_size = int(sys.argv[4])
    partitions = [1 for _ in range(len(dims))]

    #sweeps is used to track the number of ranks for each dimension connection
    sweeps = []
    for i in range(np.max(rs)):
        for j in range(len(rs)):
            if rs[j]>1:
                sweeps.append(j)
                rs[j]-=1

    send_data = None
    
    #Build the subtensors on each rank
    shifting_p,bounds_p = Subtensor_Processor_Build(processor_partition,dims)
    sub_dim = [bounds_p[rank][i][1]-bounds_p[rank][i][0] for i in range(len(bounds_p[rank]))]

    test_local_sub = Local_subtensor_construction(bounds_p[rank],sub_dim)
    Xsubs_local_test,shifting_local_test,bounds_local_test = Subtensor_Split(test_local_sub,partitions)

    local_data = []
    comm_times = 0
    search_times = 0
    #local_approx = []
    for i in range(len(Xsubs_local_test)):
        local_data.append([Xsubs_local_test[i],i])

    #Set up necessary memory for indices
    if rank==0:
        start_time = MPI.Wtime()
        I_row_global = [[] for i in range(len(dims)-1)]
        I_col_global = [[] for i in range(len(dims)-1)]
    
    
    I_row_local = [[[] for _ in range(len(dims)-1)] for _ in range(len(local_data))]
    I_col_local = [[[] for _ in range(len(dims)-1)] for _ in range(len(local_data))]
    FI_row_local = [[[] for _ in range(len(dims)-1)] for _ in range(len(local_data))]
    FI_col_local = [[[] for _ in range(len(dims)-1)] for _ in range(len(local_data))]

    Selection_row_local = [[[0] for _ in range(len(dims)-1)] for _ in range(len(local_data))]
    Selection_col_local = [[[0] for _ in range(len(dims)-1)] for _ in range(len(local_data))]
    
    p1 = [[] for _ in range(len(local_data))]
    p =[[] for _ in range(len(local_data))]
    piv_vals = []
    local_piv_value = []
    local_piv_loc = []
    

    #------------------------------------------------------------------------#
    ############################ INITIAL SEARCH ##############################
    ################ Can switch initial search from argmax ###################
    ####### Most efficient way would be a random sample from unfolding #######
    #------------------------------------------------------------------------#
    

    for i in range(len(I_row_local[0])):
        FI_row_local[0][i] = [np.unravel_index(np.argmax(Unfolding(local_data[0][0],i)),Unfolding(local_data[0][0],i).shape)[0]]
        FI_col_local[0][i] = [np.unravel_index(np.argmax(Unfolding(local_data[0][0],i)),Unfolding(local_data[0][0],i).shape)[1]]
        I_row_local[0][i] = Full_To_Super(FI_row_local[0][i],FI_col_local[0][i],i,local_data[0][0].shape)[0]
        I_col_local[0][i] = Full_To_Super(FI_row_local[0][i],FI_col_local[0][i],i,local_data[0][0].shape)[1]

    piv_vals = [Unfolding(local_data[0][0],i)[FI_row_local[0][i][0],FI_col_local[0][i][0]] for i in range(len(I_row_local[0]))]
    local_pivot_values = comm.gather(piv_vals,root=0)

    

    #Select largest pivot
    if rank==0:
        selection = [[np.argmax(np.abs([local_pivot_values[i][k] for i in range(len(local_pivot_values))]))] for k in range(len(dims)-1)]
        for l in range(1,size):
            comm.send(selection,dest = l)

    if rank!=0:
        selection = comm.recv(source=0)
    
    #Communication to correct index sets for influenced ranks
    for k in range(len(dims)-1):
        
        if rank==selection[k][0]:
            info_index = [I_row_local[0][k][0],I_col_local[0][k][0],FI_row_local[0][k][0],FI_col_local[0][k][0],sub_dim] 
            for l in range(size):
                if l!= selection[k][0]:
                    comm.send(info_index,dest = l,tag = 1)
        if rank!=selection[k][0]:
            info_index = comm.recv(source = selection[k][0],tag = 1)
        

        row,col = Subtensor_Comm(selection[k][0],shifting_p,k)
        if rank!=selection[k][0]:
            if rank not in row and len(I_row_local[0][k])>0:
                I_row_local[0][k] = np.delete(I_row_local[0][k],[-1])
                FI_row_local[0][k] = np.delete(FI_row_local[0][k],[-1])
                Selection_row_local[0][k] = np.delete(Selection_row_local[0][k],[-1])
            elif rank in row:
                I_row_local[0][k][-1] = info_index[0]
                FI_row_local[0][k][-1] = info_index[2]
            if rank not in col and len(I_col_local[0][k])>0:
                I_col_local[0][k] = np.delete(I_col_local[0][k],[-1])
                FI_col_local[0][k] = np.delete(FI_col_local[0][k],[-1])
                Selection_col_local[0][k] = np.delete(Selection_col_local[0][k],[-1])
            elif rank in col:
                I_col_local[0][k][-1] = info_index[1]
                FI_col_local[0][k][-1] = info_index[3]
  
        if rank==0:
            adj_ind = Sub_to_Unfold([info_index[2],info_index[3]],info_index[4],dims,shifting_p[selection[k][0]],k)
            I_row_global[k].append(adj_ind[0])
            I_col_global[k].append(adj_ind[1])

    
    counter = [1 for _ in range(len(dims)-1)]
    for j in sweeps:
        
        active_ranks = []  
        active = 0

        ##################### LINES 641 - 665 DETERMINE THE ACTIVE RANKS #####################
        ############## LOCAL SUPERBLOCKS AND APPROXIMATIONS ARE FORMED/ALLOCATED #############

        if j==0:
            if len(I_col_local[0][1])>0:
                active = 1
        elif j==len(dims)-2:
            if len(I_row_local[0][j-1])>0:
                active = 1
        else:
            if len(I_row_local[0][j-1])>0 and len(I_col_local[0][j+1])>0:
                active=1       
        if j==0 and active==1:
            sups = MatrixSuperblockUpdate(local_data[0][0],[0],FI_col_local[0][1],0)       
            suptilde = np.zeros(sups.shape)
        elif j==len(dims)-2 and active==1:
            sups = MatrixSuperblockUpdate(local_data[0][0],FI_row_local[0][j-1],[0],j)       
            suptilde = np.zeros(sups.shape)      
        elif j>0 and j<len(dims)-2 and active==1:
            sups = MatrixSuperblockUpdate(local_data[0][0],FI_row_local[0][j-1],FI_col_local[0][j+1],j)       
            suptilde = np.zeros(sups.shape)
        active = comm.gather([rank,active],root=0)
        if rank==0:
            active_ranks = [active[i][0] for i in range(len(active)) if active[i][1]==1]
            for i in range(1,size):
                comm.send(active_ranks,dest = i)
        if rank!=0:
            active_ranks = comm.recv(source = 0)

        ################################################################################################
        #################### Lines 677 - 722 Build local cross approximations ########################## 
        ################################################################################################
            
         
        for l in range(len(selection[j])):
            #comm.Barrier()
            t1 = MPI.Wtime()
            if rank in active_ranks:
            
                #EXTRACT ROW AND COL THAT ARE ON THEIR LOCAL RANK                
                row_vecs = []
                col_vecs = []
                
                row,col = Subtensor_Comm(rank,shifting_p,j)
                row_send = [col[i] for i in range(len(col)) if col[i] in active_ranks and col[i]!=rank]
                col_send = [row[i] for i in range(len(row)) if row[i] in active_ranks and row[i]!=rank]

                if rank==selection[j][l]:
                    index_row = I_row_local[0][j][list(Selection_row_local[0][j]).index(l)]
                    index_col = I_col_local[0][j][list(Selection_col_local[0][j]).index(l)]
                    delta = 1/((sups - suptilde)[index_row,index_col])
                    for i in active_ranks:
                        if rank!=i:
                            comm.send(delta,dest = i,tag = 666)
                if rank!=selection[j][l]:
                    delta = comm.recv(source = selection[j][l],tag = 666)
                
                if l in Selection_row_local[0][j]:
                    index_row = list(Selection_row_local[0][j]).index(l)
                    row_vecs.append([(sups-suptilde)[I_row_local[0][j][index_row],:]])
                for i in row_send:
                    if len(row_vecs)>0:
                        comm.send(row_vecs[-1],dest = i,tag = 1234)
                    else:
                        comm.send(None,dest = i,tag = 1234)

                if l in Selection_col_local[0][j]:
                    index_col = list(Selection_col_local[0][j]).index(l)
                    col_vecs.append([(sups-suptilde)[:,I_col_local[0][j][index_col]]])
                for i in col_send:
                    if len(col_vecs)>0:
                        comm.send(col_vecs[-1],dest = i,tag = 4321)
                    else:
                        comm.send(None,dest = i,tag = 4321)

                for i in row_send:
                    row_recv = comm.recv(source = i,tag = 1234)
                    if row_recv is not None:
                        row_vecs.append(row_recv)

                for i in col_send:
                    col_recv = comm.recv(source = i,tag = 4321)
                    if col_recv is not None:
                        col_vecs.append(col_recv)
                comm_times+=MPI.Wtime() - t1
                suptilde += delta*np.outer(col_vecs,row_vecs)
                
                    
        #Run a pivot search over all active ranks
        if rank in active_ranks:
            t2 = MPI.Wtime()
            I_row_local[0],I_col_local[0],FI_row_local[0],FI_col_local[0],p,ps=MPICrossInterpSingleItemSuperPiv(sups,suptilde,I_row_local[0],I_col_local[0],FI_row_local[0],FI_col_local[0],j,local_data[0][0].shape)    
            search_times+=MPI.Wtime() - t2
            Selection_row_local[0][j] = np.append(Selection_row_local[0][j],counter[j])
            Selection_col_local[0][j] = np.append(Selection_col_local[0][j],counter[j])
            piv_vals = [p*ps]

        #Gather the pivot values for selection
        local_piv_value = comm.gather(piv_vals,root = 0)

        if rank==0:
            active_piv_value = []
            for i in active_ranks:
                active_piv_value.append(local_piv_value[i])
            selection[j].append(active_ranks[np.argmax(np.abs([active_piv_value[i] for i in range(len(active_piv_value))]))])
            row,col = Subtensor_Comm(selection[j][-1],shifting_p,j)

            for i in range(1,size):
                comm.send(selection,dest=i)
        
        if rank!=0:
            selection = comm.recv(source = 0)
            row,col = Subtensor_Comm(selection[j][-1],shifting_p,j)

        #Send out information required to alter the index sets      
        if rank==selection[j][-1]:
            info_index = [I_row_local[0][j],I_col_local[0][j],FI_row_local[0][j],FI_col_local[0][j],local_data[0][0].shape]
            for l in range(size):
                if l!= selection[j][-1]:
                    comm.send(info_index,dest = l,tag = 1)

        if rank!=selection[j][-1]:
            info_index = comm.recv(source = selection[j][-1],tag = 1)

        if rank!=selection[j][-1]:
                if rank in row and rank in active_ranks:
                    I_row_local[0][j][-1] = info_index[0][-1]
                    FI_row_local[0][j][-1] = info_index[2][-1]
                elif rank in row and rank not in active_ranks:
                    I_row_local[0][j] = info_index[0]
                    FI_row_local[0][j] = info_index[2]
                    Selection_row_local[0][j] = np.append(Selection_row_local[0][j],counter[j])
                elif rank not in row and rank in active_ranks:
                    I_row_local[0][j] = np.delete(I_row_local[0][j],[-1])
                    FI_row_local[0][j] = np.delete(FI_row_local[0][j],[-1])
                    Selection_row_local[0][j] = np.delete(Selection_row_local[0][j],[-1])
                if rank in col and rank in active_ranks:
                    I_col_local[0][j][-1] = info_index[1][-1]
                    FI_col_local[0][j][-1] = info_index[3][-1]
                elif rank in col and rank not in active_ranks:
                    I_col_local[0][j] = info_index[1]
                    FI_col_local[0][j] = info_index[3]
                    Selection_col_local[0][j] = np.append(Selection_col_local[0][j],counter[j])
                elif rank not in col and rank in active_ranks:
                    I_col_local[0][j] = np.delete(I_col_local[0][j],[-1])
                    FI_col_local[0][j] = np.delete(FI_col_local[0][j],[-1])
                    Selection_col_local[0][j] = np.delete(Selection_col_local[0][j],[-1])

        counter[j] = counter[j]+1 
        #Adjust global indices    
        if rank==0:
                adj_ind = Sub_to_Unfold([info_index[2][-1],info_index[3][-1]],info_index[4],dims,shifting_p[selection[j][-1]],j)
                I_row_global[j].append(adj_ind[0])
                I_col_global[j].append(adj_ind[1])

    #Set up items needed to construct global cores
    if rank==0:
        col_tk = [[] for _ in range(len(dims)-1)]
        cols = [[] for _ in range(len(dims)-1)]
        core_ranks=[1 for _ in range(len(dims)+1)]
        for i in range(len(dims)-1):
            core_ranks[i+1] = len(I_row_global[i])
        for i in range(1,size):
            comm.send([I_row_global,I_col_global],dest = i,tag = 909)

    if rank!=0:
        index_recv = comm.recv(source = 0,tag = 909)
        I_row_global = index_recv[0]
        I_col_global = index_recv[1]
    internal_time = 0  

    core_start_time = MPI.Wtime()

    for j in range(len(dims)-1):
        A = Unfolding(local_data[0][0],j)

        #Extract needed columns locally to send to rank zero for core construction
        for i in range(len(selection[j])):
            col_tk = [[] for _ in range(len(dims))]
            t1 = MPI.Wtime()
            sel_index,row_info,col_info = MPI_Tk_Index([I_row_global[j][i],I_col_global[j][i]],shifting_p,bounds_p,j,dims)           
            
            if rank in col_info[1]:
                if rank!=0:
                    comm.send([rank,A[:,col_info[0]],local_data[0][0].shape],dest=0,tag = 809)
                else:
                    col_tk[j].append([rank,A[:,col_info[0]],local_data[0][0].shape])
        
            if rank==0:
                for l in col_info[1]:
                    if l!=0:
                        col_tk[j].append(comm.recv(source = l,tag = 809))  
            
            
            if rank==0:
                grouping = np.flip(processor_partition[1:j+1])
                inter = [np.reshape(col_tk[j][l][1],col_tk[j][l][2][:j+1]) for l in range(len(col_tk[j]))]
                placeholder = []

                for n in range(len(grouping)): 
                    for m in range(0,len(inter),grouping[n]):
                        placeholder.append(inter[m:m+grouping[n]])
                    inter = placeholder
                    placeholder = []

                sub_block = np.block(inter)

                cols[j].append(sub_block.reshape(1,-1)[0])
            #comm_times+=MPI.Wtime() - t1
        

             
     
    if rank==0:
        #Construction of cores 1 through d-1
        cores = []
        for j in range(len(dims)-1):
            cols[j] = np.array(cols[j]).T
            Tk = MPI_Recursive_Tk(cols[j],I_row_global[j])

            if j == 0:
                cores.append( np.reshape(Tk,[1,dims[j],len(I_col_global[j])]))

            else:
                Tk = np.reshape(Tk,[math.prod(dims[:j]),dims[j],len(I_col_global[j])])
                cores.append(Tk[np.ix_(I_row_global[j-1])])
        
    #Construction of final core which comes from row vectors
    A = Unfolding(local_data[0][0],len(dims)-2)
    rows = [[] for _ in range(len(I_row_global[-1]))]
    for i in range(len(I_row_global[-1])):          
        row_info = MPI_Tk_Index([I_row_global[-1][i],0],shifting_p,bounds_p,len(dims)-2,dims)[1]

        if rank in row_info[1]:
            if rank!=0:
                comm.send(A[row_info[0],:],dest = 0,tag = 1001)
            else:
                rows[i].append(A[row_info[0],:])
        if rank==0:
            for l in row_info[1]:
                if l!=0:
                    rows[i].append(comm.recv(source = l,tag = 1001))
        
        
    commun_times = comm.gather(comm_times,root = 0)
    searching_times = comm.gather(search_times,root = 0) 

    if rank==0:
        #Build final core   
        for i in range(len(rows)):
            rows[i] = np.array([x for y in rows[i] for x in y])
        cores.append(np.reshape(rows,[len(I_row_global[-1]),dims[-1],1]))
        final_time = MPI.Wtime()

        #Compute sampled errors
        sample_error = 0
        for _ in range(err_sample_size):
            index = tuple([random.randint(0,dims[n]-1) for n in range(len(dims))])            
            sample_error+=np.abs(Core_to_Tensor_Value(cores,index) - tensor_entry(index))**2/np.abs(tensor_entry(index))**2

        
        print('-'*55)
        print('Time to run full algorithm            :',np.round(final_time - start_time,4),'seconds')
        print('-'*55)
        print('Percent of time for communication     :',np.round(100*comm_times/(final_time-start_time),4),'%')
        print('-'*55)
        print('Percent of time for pivot searching   :',np.round(100*search_times/(final_time - start_time),4),'%')
        print('-'*55)
        print('Percent of time for core construction :',np.round(100*(final_time - core_start_time)/(final_time - start_time),4),'%')
        print('-'*55)
        print('Unaccounted time percentage           :', np.round(100 - 100*(comm_times + search_times + (final_time - core_start_time))/(final_time-start_time)),'%')
        print('-'*55)
        print('Sampled Relative Error                : {:.3e}'.format(np.sqrt(sample_error)))
        print('-'*55)
        print('Acheived ranks:',core_ranks)   
        print('-'*55)

        
