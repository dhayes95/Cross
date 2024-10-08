# DISTRIBUTED MEMORY PARALLEL ADAPTIVE TENSOR-TRAIN CROSS APPROXIMATION - Authors: Tianyi Shi, Daniel Hayes, Jingmei Qiu

The file included in this folder titled "MPISubtensorTTCross.py" is a self contained files that will automaticall load numpy, math, and mpi4py. All other functions used are included before the main body of the file begins.

# Description of the code

The provided python code is split into two main parts:
* 1 - The custom built functions which go from lines 25 - 1370
* 2 - The implementation of the algorithm which goes from lines 1374 - 2053

Functions in part 1 include items such as index specific functions, greedy pivot search functions, and functions pertaining to actions at the tensor level such as entrywise computation, unfoldings, and superblock constructions.

Items in part two follow the structure outlined in the paper where lines 1374 - 1813 do the computation of all pivot selections across all dimensions, and the remainder of the code implements core construction. 

# Running the code

In order to run the code, it can be run through command line arguments, or sbatch files with the following input setup. 
* 1 - Select the desired tensor definition in the .py file. Instructions are included at the start of the .py file.
* 2 - Input the desired parameters in the command line. This file uses the sys.argv() input method and required 7 distinct inputs.
  * Input 0: Number of MPI ranks to be used
  * Input 1: Number of indices in which we will sample error at
  * Input 2: Number of test runs that the algorithm will be tested on
  * Input 3: Tensor dimension as a comma separated seqence of numbers, e.g., 50,35,40,26 would correspond to a 4D tensor of size [50,35,40,26]
  * Input 4: Internal core ranks as a comma separated sequence of numbers, e.g., 15,17,9 would correspond to a TT-Cross approximation of a 4D tensor with core ranks (1,15,17,9,1)
  * Input 5: List of required partition structure as a comma separated sequence of numbers, e.g., 1,2,1,1 would partition the second dimension into two intervals.
 
Below is an example of a the setup to run a 4D Maxwellian tensor of size [800,400,800,400] with TT core ranks (1,10,5,20,1) on a partition of (1,8,8,1) over 10 test runs with 100000 error samples
* Uncomment lines 33 and 41 of "MPISubtensorTTCross.py"
* execute the command mpirun -n 64 MPISubtensorTTCross.py 100000 10 800,400,800,400 10,5,20 1,8,8,1 







