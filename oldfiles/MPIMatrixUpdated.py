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

def Processor_Comm_Path(proc,selection_proc,selection_loc,shifting_proc,shifting_loc,k):
        
    proc_row,proc_col = Subtensor_Comm(selection_proc,shifting_proc,k)
    loc_row,loc_col = Subtensor_Comm(selection_loc,shifting_loc,k)

    rows_local = []
    cols_local = []

    full_rows_send = []
    full_cols_send = []

    full_rows_recv = []
    full_cols_recv = []

    #Information sending locations
    if proc in proc_row:
        ph,rows = Subtensor_Comm(proc,shifting_proc,k)
        rows_proc = [p for p in rows if p!=proc]
    else:
        rows_proc = []

    if proc in proc_col:
        cols,ph = Subtensor_Comm(proc,shifting_proc,k)
        cols_proc = [p for p in cols if p!=proc]
    else:
        cols_proc = []
    
    rows_loc = []
    for i in range(len(shifting_loc)):
        if i in loc_row:
            ph,rows = Subtensor_Comm(i,shifting_loc,k)
            rows_loc.append([i,[p for p in rows]])
            

    full_rows_send = [[row,rows_loc] for row in rows_proc]
    rows_local = [rows_loc[i][0] for i in range(len(rows_loc)) if len(rows_proc)>0]

    cols_loc = []
    for i in range(len(shifting_loc)):
        if i in loc_col:
            cols,ph = Subtensor_Comm(i,shifting_loc,k)
            cols_loc.append([i,[p for p in cols]])
            cols_local.append(i)

    full_cols_send = [[col,cols_loc] for col in cols_proc]
    cols_local = [cols_loc[i][0] for i in range(len(cols_loc)) if len(cols_proc)>0]      

    #Information receiving locations
    if proc not in proc_col:
        
        cols,ph = Subtensor_Comm(proc,shifting_proc,k)
        full_cols_recv= [i for i in proc_col if i in cols]
    else:
        full_cols_recv = []
    
    if proc not in proc_row:
        
        ph,rows = Subtensor_Comm(proc,shifting_proc,k)
        full_rows_recv= [i for i in proc_row if i in rows]
    else:
        full_rows_recv = []

    




    return full_rows_send,full_cols_send,full_rows_recv,full_cols_recv,rows_local,cols_local


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    m1 = int(sys.argv[1])
    m2 = int(sys.argv[2])
    p1_grid = int(sys.argv[3])
    p2_grid = int(sys.argv[4])
    internal_p1 = int(sys.argv[5])
    internal_p2 = int(sys.argv[6])
    time_coll = []
    time_coll2 = []
    if rank==0:
        if p1_grid*p2_grid != size and p1_grid!=p2_grid:
            print('Currently only working for processor grid uniform and same size submatrices')
            exit()
    
    
    send_data = None
    shifting_p = None
    if rank==0:
        B = np.fromfunction(lambda i0,i1: 1.0/(i0+i1+1),(m1,m2))
        #B = np.random.rand(m1,m2)
        #B = np.reshape(np.arange(m1*m2),[m1,m2]).astype(float)

        I_global = []
        J_global = []
    #
    t4p = MPI.Wtime()
    if rank==0:
        Bsubs,shifting_p,bounds_p = Subtensor_Split(B,[p1_grid,p2_grid])
        raveled = [np.ravel(Bsub) for Bsub in Bsubs]
        send_data = np.concatenate(raveled)
 
        for i in range(1,size):
            comm.send(shifting_p,dest = i)
    if rank!=0:
        shifting_p = comm.recv(source=0)
      
    t4 = MPI.Wtime()
    recvbuf = np.empty((int(m1/p1_grid), int(m2/p2_grid)), dtype='float')

    comm.Scatterv(send_data, recvbuf, root=0)
        
    Bsubs_local,shifting_local,bounds_local = Subtensor_Split(recvbuf,[internal_p1,internal_p2])
    comm.Barrier()

        
    local_data = []
    local_approx = []
    for i in range(len(Bsubs_local)):
        local_data.append([Bsubs_local[i],i])
        local_approx.append(np.zeros(Bsubs_local[i].shape))

        
    comm.Barrier()
    comm_dict = {}
    for j in range(len(local_data)):
        for i in range(size):
            row_send,col_send,row_recv,col_recv,p1,p2 = Processor_Comm_Path(rank,i,j,shifting_p,shifting_local,0)
            comm_dict[(i,j)] = [row_send,col_send,row_recv,col_recv,p1,p2]
    
    #if rank==0:
    #    print(comm_dict)


    I_local = [[] for _ in range(len(local_data))]
    J_local = [[] for _ in range(len(local_data))]
    piv_local = [[] for _ in range(len(local_data))]
    piv_sign_local = [[] for _ in range(len(local_data))]

    for _ in range(20):

        for i in range(len(local_data)):
            I_local[i],J_local[i],piv_local[i],piv_sign_local[i] = MPIGreedy_Search_Piv(local_data[i][0],local_approx[i],I_local[i],J_local[i])


        local_piv_selection = np.argmax(piv_local)
        local_piv_value = piv_sign_local[local_piv_selection]*piv_local[local_piv_selection]


        #All gather for pivot values
        global_piv = comm.gather(local_piv_value,root = 0)
        local_indices = comm.gather([I_local[local_piv_selection],J_local[local_piv_selection]],root = 0)
        global_piv_local_location = comm.gather(local_piv_selection,root=0)



        if rank==0:

            global_piv_selection = np.argmax(np.abs(global_piv))
            global_piv_value = global_piv[global_piv_selection]
            submatrix_location = global_piv_local_location[global_piv_selection]

            
            global_piv_info = [global_piv_selection,submatrix_location,local_indices[global_piv_selection][0][-1],local_indices[global_piv_selection][1][-1],global_piv_value]

            I_global.append(global_piv_info[2] + shifting_local[submatrix_location][0]+shifting_p[global_piv_selection][0])
            J_global.append(global_piv_info[3] + shifting_local[submatrix_location][1]+shifting_p[global_piv_selection][1])



            for i in range(1,size):
                comm.send(global_piv_info,dest = i)


        if rank!=0:
            global_piv_info = comm.recv(source = 0)



       
        #row_send,col_send,row_recv,col_recv,p1,p2 = Processor_Comm_Path(rank,global_piv_info[0],global_piv_info[1],shifting_p,shifting_local,0)
        [row_send,col_send,row_recv,col_recv,p1,p2] = comm_dict[(global_piv_info[0],global_piv_info[1])]
        """
        if (global_piv_info[0],global_piv_info[1]) in comm_dict:
            [row_send,col_send,row_recv,col_recv,p1,p2] = comm_dict[(global_piv_info[0],global_piv_info[1])]
        else:
            for j in range(len(local_data)):
                for i in range(size):
                    row_send,col_send,row_recv,col_recv,p1,p2 = Processor_Comm_Path(rank,i,j,shifting_p,shifting_local,0)
                    comm_dict[(i,j)] = [row_send,col_send,row_recv,col_recv,p1,p2]
        """
        for i in range(len(local_data)):
            
            
            if i not in p1:
                if len(I_local[i])>0:
                    I_local[i] = np.delete(I_local[i],[-1])
            else:
                if len(I_local[i])>0:
                    I_local[i][-1] = global_piv_info[2]
                else:
                    np.append(I_local[i],global_piv_info[2])
            if i not in p2:
                if len(J_local[i])>0:
                    J_local[i] = np.delete(J_local[i],[-1])
            else:
                if len(J_local[i])>0:
                    J_local[i][-1] = global_piv_info[3]
                else:
                    np.append(J_local[i],global_piv_info[3])
 

        comm.Barrier()


        

        for i in range(len(row_send)):
            row_info = []
            
            for j in range(len(row_send[i][1])):               
                row_info.append([row_send[i][1][j][1],(local_data[row_send[i][1][j][0]][0]-local_approx[row_send[i][1][j][0]])[np.ix_([global_piv_info[2]])]])
            #Get non-blocking send added
            comm.send(row_info,dest=row_send[i][0],tag = 2)

                
        for i in range(len(row_recv)):
            row_info = comm.recv(source = row_recv[i],tag = 2)

        for i in range(len(col_send)):
            col_info = []
            for j in range(len(col_send[i][1])):
                col_info.append([col_send[i][1][j][1],np.transpose((local_data[col_send[i][1][j][0]][0] - local_approx[col_send[i][1][j][0]]))[np.ix_([global_piv_info[3]])].T])
            #Get non-blocking send added
            comm.send(col_info,dest=col_send[i][0],tag = 1)

                
        for i in range(len(col_recv)):
            col_info = comm.recv(source = col_recv[i],tag = 1)
        
        comm.Barrier()  

        for i in range(len(local_data)):
            for j in range(len(row_info)):
                if i in row_info[j][0]:
                    local_row_vectors=row_info[j][1]

            for j in range(len(col_info)):
                if i in col_info[j][0]:
                    local_col_vectors=col_info[j][1]

            local_approx[i]+=(1/global_piv_info[-1])*np.outer(local_col_vectors,local_row_vectors)

    
            
        

    comm.Barrier()
    t5 = MPI.Wtime()
    time_coll2.append(t5-t4p)


    #if rank==0:
    #    print(I_global)
    #    for i in range(len(I_global)):
    #        Atilde = Recursive_CUR(B,I_global[:i+1],J_global[:i+1],0)
    #        print(np.linalg.norm(B - Atilde)/np.linalg.norm(B))

    if rank==0:        
        print('time taken MPI :', t5-t4p,'seconds')
        print('Ranks used:', size)
        print('Partitions:', p1_grid*internal_p1,'x',p2_grid*internal_p2)
    
