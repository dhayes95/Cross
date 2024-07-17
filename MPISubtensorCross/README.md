# Distributed memory subtensor parallel TT-Cross

The file included in this folder titled "MPISubtensorTTCross.py" is a self contained files that will automaticall load numpy, math, and mpi4py. All other functions used are included before the main body of the file begins.

# Description of the code

The provided python code is split into two main parts:
* 1 - The custom built functions which go from lines 25 - 1370
* 2 - The implementation of the algorithm which goes from lines 1374 - 2088

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
  * Input 5: List of required partition structure as a comma separated sequence of numbers, e.g., 1,2,1,1 would require any partition to be a refinement of the partition 1,2,1,1
  * Input 6: List(s) of unordered partition values as a comma separated sequence of numbers, e.g. 1,1,2,2 1,1,1,4 would run all combinations of partitions in which either two     dimensions are partitioned into two intervals, or one dimension is partitioned into 4 intervals provided they are a refinement of Input 5. Note: if a specific partition is to be run, set Input 5 and Input 6 to the same desired partition list.






