# Version 1.

*** NEW ADDITION *** 

Vectorization, Key-byte representation (NumPy)

Result: the optimized code is ~6.1x faster for Pivot Search, and ~34x faster for core construction compared to [the previous version](https://github.com/dhayes95/Cross/blob/main/MPISubtensorCross/MPISubtensorTTCross.py)

# Version 2.

*** NEW ADDITION *** 

PyTorch on top of Vectorization, Key-byte representation, and NumPy

Result: For MPI Rank 1, the optimized code is ~1.3x faster for Pivot Search, and ~1.09x faster for core construction compared to Version 1

# Version 3.

*** NEW ADDITION ***

2D block-cyclic process grid layout, on top of both Version 1 (NumPy) and Version 2 (PyTorch)

The subtensor partition is no longer tied to the MPI rank count. A separate block grid
and process grid are given, and blocks are dealt out to the process grid cyclically in
every dimension, so each unfolding sees a 2D block-cyclic distribution.

Result: any partition can be run on any number of ranks. Setting the process grid equal
to the block grid reproduces Versions 1 and 2 exactly, and the approximation depends
only on the block grid - the same partition gives bit-identical pivots on 1, 4, 8 or 32
ranks. On a 240^3 Hilbert tensor at 4 ranks, a 16-subtensor partition is ~2.7x more
accurate than the 4-subtensor partition that 4 ranks previously forced, at the same cost.

Also adds a per-rank compute / communication / wait breakdown (`TT_PROFILE=1`), which
separates load imbalance from transfer cost by putting a Barrier in front of every
collective. On a small local test ~40% of the pivot-search phase turned out to be idle
wait and only ~5% transfer, with local compute spread 2x across ranks - all of which the
previous total-phase-time spread metric hid.

Later additions to Version 3 (same files): a vectorised entry cache and lazily
evaluated cross approximation (2-7x faster, 2-3x less memory, bit-identical results),
and per-unfolding row/column communicators so the cross-vector exchange stays inside
the paper's row and column groups instead of an allgather over every rank.

# Version 4.

*** NEW ADDITION ***

Pivot-order 2D block-cyclic layout (prototype, NumPy only)

A Perlmutter study of Version 3 showed that any layout hashing tensor coordinates to
processes leaves ranks idle 70-97% of the pivot-search phase, because the pivots - and
hence the superblock rows - cluster in a slab of the tensor. Version 4 instead deals the
rows and columns of each unfolding's superblock out block-cyclically by their position
in the pivot list, which is balanced by construction (as in block-cyclic LU) while
keeping the same process grid and row/column communicators.

Result: on one node with 64 ranks, 20-55x faster pivot search than Version 3 at
comparable sampled error, with the busiest rank within 2-10% of the average and memory
spread evenly (no hot rank). The approximation now depends on the process grid rather
than on a block grid. Core construction is still serial on rank 0 in this prototype.
