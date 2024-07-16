import numpy as np
import random
import threading




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
    # Tk = A[:,J[0]]/A[I[0],J[0]];

    for i in range(1,Ilen):
        Itemp = I[:i]
        Jtemp = J[:i]
              
        ii = I[i]
        jj = J[i]
               
        Sk = np.reshape(Tk,[dim[0],i])@A[Itemp,:][:,jj] - A[:,jj]
        e1 = np.zeros([1,dim[0]])
        e1[0,ii]=1

        # Sk = Tk@A[Itemp,:][:,jj] - A[:,jj]
        
        
        delta1 = (e1@Sk)
        if delta1==0:
            delta = 0
        else:
            delta = 1.0/delta1

        a = np.reshape(Tk,[dim[0],i]) - delta*np.outer(np.reshape(Sk,[dim[0],1]),np.reshape(Tk[ii,:],[1,Tk.shape[1]]))
        b = np.reshape(delta*Sk,[dim[0],1])       
        Tk = np.concatenate((a,b),axis=1)

        # if Sk[ii] == 0:
        #     delta = 0
        # else:
        #     delta = 1.0/Sk[ii]

        # Tk = np.concatenate((Tk - delta*Sk@Tk[ii,:], delta*Sk), axis=1)
    
    return Tk

def recursive_Tk_update(A,Tk_old,I,J):
    """
    Parameters
    ----------
    A : Matrix
    Tk_old : Tk generated at the previous step
    I : row indices currently selected
    J : column indices currently selected

    Returns
    -------
    Tk : A[:,[J,jj]]@np.linalg.pinv(A[[I,ii],[J,jj]]), new Tk matrix

    Note: This function is used for the TT cross core construction
    """
    Ilen = len(I)
    Itemp = I[:Ilen-1]
    ii = I[Ilen-1]
    jj = J[Ilen-1]
    
    #print(Itemp)
    
    Sk = np.reshape(Tk_old@A[Itemp,:][:,jj] - A[:,jj],[A.shape[0],1])
    print(Sk.shape)
    print(Tk_old[ii,:].shape)
    if Sk[ii] == 0:
        delta = 0
    else:
        delta = 1.0/Sk[ii]
    
    Tk = np.concatenate((Tk_old - delta*Sk@np.reshape(Tk_old[ii,:],[1,Ilen-1]), delta*Sk), axis=1)
    
    return Tk

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


def Recursive_CUR_update(A_ii,A_jj,Atilde,ii,jj,tol):
    """
    Parameters
    ----------
    A : Matrix
    Atilde_old : approximation at the previous iteration
    ii : newly selected row pivot
    jj : newly selected column pivot
    tol : tolerance for delta parameter

    Returns
    -------
    Atilde : CUR
    """
    deltainv = A_ii[jj] - Atilde[ii,jj]
    if np.abs(deltainv) <= tol:
        return Atilde
    else: 
        Atilde = Atilde +(1.0/deltainv)*np.outer(A_jj-Atilde[:,jj],A_ii-Atilde[ii,:])

    return Atilde




def Greedy_Pivot_Search_Piv(A,I,J,sample_size=1000,maxiter=100,tol=0):
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
        sample_number = np.min([sample_size,subdim[0],subdim[1]])
        Irandom = random.sample(list(np.arange(dim[0])),sample_number)
        Jrandom = random.sample(list(np.arange(dim[1])),sample_number)
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
    
    
    if E[i_star,j_star]>tol:
        I_new = np.append(I,Row_track[i_star]).astype(int)
        J_new = np.append(J,Col_track[j_star]).astype(int)
    else:
        I_new = I
        J_new = J
    
    #print(E[i_star,j_star])

    
    return I_new,J_new,E[i_star,j_star]











