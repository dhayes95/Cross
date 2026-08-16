# Version 1.

*** NEW ADDITION *** 

Vectorization, Key-byte representation (NumPy)

Result: the optimized code is ~6.1x faster for Pivot Search, and ~34x faster for core construction compared to [the previous version](https://github.com/dhayes95/Cross/blob/main/MPISubtensorCross/MPISubtensorTTCross.py)

# Version 2.

*** NEW ADDITION *** 

PyTorch on top of Vectorization, Key-byte representation, and NumPy

Result: For MPI Rank 1, the optimized code is ~1.3x faster for Pivot Search, and ~1.09x faster for core construction compared to Version 1
