#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import random
from mpi4py import MPI
import matplotlib.pyplot as plt
import sys
import itertools

def Recursive_CUR(A,Iin,Jin,tol):
    """
    Parameters
    ----------
    A : Matrix
    Iin : row indices
    Jin : column indices
    tol : tolerance for delta parameter

    Returns
    -------
    Atilde : CUR
    
    Update reason: This can now deal with index sets of different sizes. 
    """
    

    dim = A.shape
    if len(Iin)==len(Jin):
        Atilde = np.zeros(dim)
        I = Iin
        J = Jin
    else:
        if len(Iin)<len(Jin):
            n = len(Jin)-len(Iin)
            if len(Iin)>0:
                Atilde = A[:,Jin[:n]]@np.linalg.pinv(A[:,Jin[:n]][[Iin[0]],:])@A[[Iin[0]],:]
                I = np.delete(Iin,[0])
                J = np.delete(Jin,np.arange(n))
            else:
                Atilde = A[:,Jin[:n]]@np.linalg.pinv(A[:,Jin[:n]])@A
                I = Iin
                J = np.delete(Jin,np.arange(n))
        else:
            n = len(Iin)-len(Jin)
            if len(Jin)>0:
                Atilde = A[:,[Jin[0]]]@np.linalg.pinv(A[Iin[:n],:][:,[Jin[0]]])@A[Iin[:n],:]
                J = np.delete(Jin,[0])
                I = np.delete(Iin,np.arange(n)) 
            else:
                Atilde = A@np.linalg.pinv(A[Iin[:n],:])@A[Iin[:n],:]
                J = Jin
                I = np.delete(Iin,np.arange(n)) 
    
    for k in range(np.min([len(I),len(J)])):
        E = A - Atilde
        deltainv = E[I[k],J[k]]
         
         
        if np.abs(deltainv) <= tol:
            break
        else: 
            Atilde = Atilde + (1.0/deltainv)*np.outer(E[:,J[k]],E[I[k],:])

    return Atilde

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
            ind_i,ind_j = np.unravel_index(np.argmax(np.abs((A - Atilde)[:,J_out])),[dim[0],len(J_out)])
            j_star = J_out[ind_j]
            
            I_new = I
            J_new = np.append(J,j_star)
            p = (A - Atilde)[ind_i,ind_j]
            p_sign = 1
        elif len(J)==dim[1] and len(I)<dim[0]:
            I_out = np.delete(np.arange(dim[0]),I)
            ind_i,ind_j  =np.unravel_index(np.argmax(np.abs((A - Atilde)[I_out,:])),[len(I_out),dim[1]])
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
            i_star,j_star = np.unravel_index(np.argmax(E[Irandom,:][:,Jrandom]),[len(Irandom),len(Jrandom)])

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
                I_new = I
                J_new = J
                p=0
                p_sign = 1
            


    
    return I_new,J_new,p,p_sign



def Vector_Comm_Paths(current,selection,shifting,partitions,send_locs,rank):
    """
    

    Parameters
    ----------
    selection : Submatrix with selected pivot
    shifting : Shifting information from Subtensor_Split

    Returns
    -------
    row_paths : first column gives origin second column gives destination
    col_paths : first column gives origin second column gives destination

    """
    
    identifier = shifting[selection]
    
    row_origin = np.reshape(list([i for i in range(len(shifting)) if shifting[i][0]==identifier[0]])*(partitions[0]-1),[(partitions[0]-1)*(partitions[1]),1])
    
    row_path = np.concatenate((row_origin,np.reshape([i for i in range(len(shifting)) if i not in row_origin],[(partitions[0]-1)*(partitions[1]),1])),axis = 1)
     
    row_comm = []
    
    for i in range(len(row_path)):
        if row_path[i,0]==current and send_locs[row_path[i,1]]!=rank:
            row_comm.append([row_path[i,1],row_path[i,0]])
            #row_comm.append(row_path[i,1])
 
    
    col_origin = np.reshape(np.repeat([i for i in range(len(shifting)) if shifting[i][1]==identifier[1]],partitions[1]-1),[(partitions[0])*(partitions[1]-1),1])
    
    col_path = np.concatenate((col_origin,np.reshape([i for i in range(len(shifting)) if i not in col_origin],[(partitions[0])*(partitions[1]-1),1])),axis = 1)
      
    col_comm = []
    for i in range(len(col_path)):
        if col_path[i,0]==current and send_locs[col_path[i,1]]!=rank:
            col_comm.append([col_path[i,1],col_path[i,0]])
            #col_comm.append(col_path[i,1])
 
    
    return row_comm, col_comm

def Vector_Comm_Recv(current,selection,shifting,partitions,send_locs,rank):
    
    identifier = shifting[selection]
    
    row_origin = np.reshape(list([i for i in range(len(shifting)) if shifting[i][0]==identifier[0]])*(partitions[0]-1),[(partitions[0]-1)*(partitions[1]),1])
    
    row_path = np.concatenate((row_origin,np.reshape([i for i in range(len(shifting)) if i not in row_origin],[(partitions[0]-1)*(partitions[1]),1])),axis = 1)

    
    col_origin = np.reshape(np.repeat([i for i in range(len(shifting)) if shifting[i][1]==identifier[1]],partitions[1]-1),[(partitions[0])*(partitions[1]-1),1])
    
    col_path = np.concatenate((col_origin,np.reshape([i for i in range(len(shifting)) if i not in col_origin],[(partitions[0])*(partitions[1]-1),1])),axis = 1)
      
    row_recv = []
    col_recv = []
    
    for i in range(len(row_path)):
        if row_path[i,1]==current and send_locs[row_path[i,0]]!=rank:
            row_recv.append(row_path[i,0])
            #row_recv.append(row_path[i,1])
            
    for i in range(len(col_path)):
        if col_path[i,1]==current and send_locs[col_path[i,0]]!=rank:
            col_recv.append(col_path[i,0])
            #col_recv.append(col_path[i,1])
    
    
    return row_recv,col_recv


def Local_Vector_Comm_Recv(current,selection,shifting,partitions,send_locs,rank):
    
    identifier = shifting[selection]
    
    row_origin = np.reshape(list([i for i in range(len(shifting)) if shifting[i][0]==identifier[0]])*(partitions[0]-1),[(partitions[0]-1)*(partitions[1]),1])
    
    row_path = np.concatenate((row_origin,np.reshape([i for i in range(len(shifting)) if i not in row_origin],[(partitions[0]-1)*(partitions[1]),1])),axis = 1)

    
    col_origin = np.reshape(np.repeat([i for i in range(len(shifting)) if shifting[i][1]==identifier[1]],partitions[1]-1),[(partitions[0])*(partitions[1]-1),1])
    
    col_path = np.concatenate((col_origin,np.reshape([i for i in range(len(shifting)) if i not in col_origin],[(partitions[0])*(partitions[1]-1),1])),axis = 1)
    
    row_recv = []
    col_recv = []
    
    for i in range(len(row_path)):
        #print(i,row_path[i,1],send_locs[row_path[i,0]])
        if row_path[i,1]==current and send_locs[row_path[i,0]]==rank:
            row_recv.append(row_path[i,0])
            #row_recv.append(row_path[i,1])
            
    for i in range(len(col_path)):
        if col_path[i,1]==current and send_locs[col_path[i,0]]==rank:
            col_recv.append(col_path[i,0])
            #col_recv.append(col_path[i,1])
    
    
    return row_recv,col_recv

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
    
    #Subtensor = [X[np.ix_(*indexing[i])] for i in range(len(selections))]
       
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

if __name__ == "__main__":
    
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    mpi_times = []
    mpi_errors = []
    full_times = []
    

    m1 = int(sys.argv[1])
    m2 = int(sys.argv[2])
    part1 = int(sys.argv[4])
    part2 = int(sys.argv[5])
    error_cal = sys.argv[6]
    full_cal = sys.argv[7]
    
    r_size = np.min([int(sys.argv[3]),m1,m2])
    
    A = np.fromfunction(lambda i0,i1: 1.0/(i0+i1+1),(m1,m2))
    partitions = (part1,part2)
    Atilde = np.zeros(A.shape)
    Asubs,shifting,bounds = Subtensor_Split(A, partitions)
    
    tstart = MPI.Wtime()
    I_global = []
    J_global = []
    local_data = []
    rank_count = [int(np.floor(np.prod(partitions)/size)+1) if i<(np.prod(partitions)%size) else int(np.floor(np.prod(partitions)/size)) for i in range(size)]
    send_locs = [i for i in range(size) for j in range(rank_count[i])]
    if rank == 0:
        data = [[Asubs[i],i] for i in range(len(Asubs))]        


    row_comm_info = []
    col_comm_info = []

    
    #Send out all the local submatrices to specified ranks
    if rank==0:
        for i in range(len(send_locs)):
            if send_locs[i]!=0 :
                comm.send(data[i],dest=send_locs[i])             
            else:
                local_data.append(data[i])
    if rank!=0:
        for _ in range(rank_count[rank]):
            local_data.append( comm.recv(source = 0))
    
        
    #Each rank must have a corresponding approximation matrix for the iterative construction
    local_approx = []
    for i in range(len(local_data)):
        local_approx.append([np.zeros(local_data[i][0].shape),local_data[i][1]])

    #Every rank will have local I,J indices as well as pivot lists p and pivot signs s
    I = [[] for _ in range(len(local_data))]
    J = [[] for _ in range(len(local_data))]
    p = [[] for _ in range(len(local_data))]
    s = [[] for _ in range(len(local_data))]
    
    
    #Run loop to get rank r_size
    for _ in range(r_size): 
        #Search for local pivot values over all ranks
        for i in range(len(local_data)):   
            I[i],J[i],p[i],s[i] = MPIGreedy_Search_Piv(local_data[i][0],local_approx[i][0], I[i], J[i])

        #Each rank will do a local pivot selection as more than one submatrix can populate one rank        
        local_pivsel = np.argmax(p)
        local_sign = s[local_pivsel]
        local_loc = local_data[local_pivsel][1]
        local_pivval = p[local_pivsel]
        local_interal_sel = comm.gather(local_pivsel,root=0)

        
        #Gather necessary information to master rank to make index alterations
        I = comm.gather(I,root=0)
        J = comm.gather(J,root=0)
        signs = comm.gather(local_sign,root=0)
        piv_val = comm.gather(local_pivval,root=0)
        piv_loc = comm.gather(local_loc,root=0)
        locations = comm.gather(local_loc,root=0)

        #On master rank we make global pivot selection, and fix local indices        
        if rank==0:
            g_pivloc = locations[np.argmax(piv_val)]
            g_pivval = piv_val[np.argmax(piv_val)]
            g_sign = signs[np.argmax(piv_val)]
            index_loc = np.argmax(piv_val)
            rowc,colc = Subtensor_Comm(g_pivloc, shifting, 0)   
            I_temp = I[index_loc][local_interal_sel[index_loc]][-1]
            J_temp = J[index_loc][local_interal_sel[index_loc]][-1]
            

            #Add to global index values
            I_global.append(I_temp + shifting[g_pivloc][0])
            J_global.append(J_temp + shifting[g_pivloc][1])
            
            
            #Deleting unnecessary index values from local indices, and fixing necessary ones
            full_indexing_count = [np.arange(i) for i in rank_count]
            bookkeep = [i for j in full_indexing_count for i in j]
            #print(I,J)
            for i in range(len(send_locs)):
                if i not in rowc:
                    I[send_locs[i]][bookkeep[i]] = np.delete(I[send_locs[i]][bookkeep[i]],[-1])
                else:
                    I[send_locs[i]][bookkeep[i]][-1] = I_temp
                    
                if i not in colc:
                    J[send_locs[i]][bookkeep[i]] = np.delete(J[send_locs[i]][bookkeep[i]],[-1])
                else:
                    J[send_locs[i]][bookkeep[i]][-1] = J_temp
                    
                    
            #Send out global information to each rank so that communication patterns can be built
            for i in range(1,size):
                comm.send(g_pivloc,dest = i,tag=11)
                comm.send(I_temp,dest = i,tag=22)
                comm.send(J_temp,dest = i,tag=33)
                comm.send(g_sign,dest = i,tag=34)
                comm.send(g_pivval,dest = i,tag=35)
                comm.send(bookkeep,dest = i,tag=100)
        
       
        
        if rank!=0:
            g_pivloc = comm.recv(source = 0,tag=11)
            I_temp = comm.recv(source=0,tag = 22)
            J_temp = comm.recv(source=0,tag = 33)
            g_sign = comm.recv(source=0,tag = 34)
            g_pivval = comm.recv(source=0,tag = 35)
            bookkeep = comm.recv(source=0,tag = 100)
        
        
        comm.Barrier()

        Ei = []
        Ej = []
        

        
        for i in range(len(local_data)): 
            tempr,tempc = Vector_Comm_Paths(local_data[i][-1], g_pivloc, shifting, partitions,send_locs,rank)
    
            if len(tempr)>0:
                for j in range(len(tempr)):
                    comm.send([(local_data[i][0]-local_approx[i][0])[np.ix_([I_temp])],tempr[j][0]],dest=send_locs[tempr[j][0]],tag=44)
                    
                    
        comm.Barrier()
        
        for i in range(len(local_data)):
            recr,recc = Vector_Comm_Recv(local_data[i][-1], g_pivloc, shifting, partitions,send_locs,rank)
            if len(recr)>0:
                Ei.append(comm.recv(source = send_locs[recr[0]],tag=44))
    
            else:             
                lrecr,lrecc = Local_Vector_Comm_Recv(local_data[i][-1], g_pivloc, shifting, partitions,send_locs,rank)

                if len(lrecr)>0:                 
                    Ei.append([(local_data[bookkeep[lrecr[0]]][0]-local_approx[bookkeep[lrecr[0]]][0])[np.ix_([I_temp])],local_data[i][-1]])
                else:
                    Ei.append([(local_data[i][0]-local_approx[i][0])[np.ix_([I_temp])],local_data[i][-1]])

        comm.Barrier()
        
        
        
        
        for i in range(len(local_data)):      
            tempr,tempc = Vector_Comm_Paths(local_data[i][-1], g_pivloc, shifting, partitions,send_locs,rank)

            if len(tempc)>0:
                for j in range(len(tempc)):
                    comm.send([np.transpose(np.transpose(local_data[i][0]-local_approx[i][0])[np.ix_([J_temp])]),tempc[j][0]],dest=send_locs[tempc[j][0]],tag=55)
                    
            
        
        for i in range(len(local_data)):
            recr,recc = Vector_Comm_Recv(local_data[i][-1], g_pivloc, shifting, partitions,send_locs,rank)

            if len(recc)>0:  
                Ej.append(comm.recv(source = send_locs[recc[0]],tag=55))
    
            else:               
                lrecr,lrecc = Local_Vector_Comm_Recv(local_data[i][-1], g_pivloc, shifting, partitions,send_locs,rank)
    
                if len(lrecc)>0:                  
                    Ej.append([np.transpose(np.transpose(local_data[bookkeep[lrecc[0]]][0]-local_approx[bookkeep[lrecc[0]]][0])[np.ix_([J_temp])]),local_data[i][-1]])
                else:
                    Ej.append([np.transpose(np.transpose(local_data[i][0] - local_approx[i][0])[np.ix_([J_temp])]),local_data[i][-1]])
    
        Ei.sort(key = lambda alpha: alpha[-1]) 
        Ej.sort(key = lambda alpha: alpha[-1])           
          
        comm.Barrier()
        for i in range(len(local_data)):
            local_approx[i][0] += ((g_sign*(1.0/g_pivval))*(np.outer(Ej[i][0],Ei[i][0])))

        
        I = comm.scatter(I,root=0)
        J = comm.scatter(J,root=0)
        p = [[] for _ in range(len(local_data))]
        comm.Barrier()
        
        
        if rank==0:
            if error_cal=='True':
                Atildetest = Recursive_CUR(A, I_global, J_global, 0)
                mpi_errors.append(np.linalg.norm(A - Atildetest)/np.linalg.norm(A))
                
    
        

        
    tend = MPI.Wtime()
    
    if rank==0:
        if full_cal=='True':
            tsub1 = MPI.Wtime()
            I_full = []
            J_full = []
            Atilde = np.zeros((m1,m2))
            for _ in range(r_size):
                I_full,J_full,pp,ss = MPIGreedy_Search_Piv(A, Atilde, I_full, J_full)
                Atilde = Recursive_CUR(A, I_full, J_full, 0)
            
            tsub2 = MPI.Wtime()
            full_times.append(tsub2-tsub1)

    
    
    if rank==0:
        mpi_times.append(tend-tstart)

       

if rank==0:        
    print('time taken MPI :', mpi_times,'seconds')
    if full_cal=='True':
        print('time taken full:', full_times,'seconds')
    print('Ranks used:', size)
    print('Partitions:', part1,'x',part2)
    if error_cal == 'True':
        print('        Error      ')
        print('-------------------')
        for i in range(len(mpi_errors)):
            print(mpi_errors[i])


    
    
    

    
    
    
    
    
    
    