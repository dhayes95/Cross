# DISTRIBUTED MEMORY PARALLEL ADAPTIVE TENSOR-TRAIN CROSS APPROXIMATION - Authors: Tianyi Shi, Daniel Hayes, Jingmei Qiu

Version 3 adds a **2D block-cyclic process grid layout**. Two files are provided,
`MPISubtensorTTCross_Version3_NumPy.py` and `MPISubtensorTTCross_Version3_PyTorch.py`;
they are the Version 1 and Version 2 codes with the new layout applied. Both are
self-contained and load their dependencies automatically.

# What changed

In Versions 1 and 2 the subtensor partition and the MPI process count were the same
number: a `(2,2,1)` grid meant 4 subtensors *and* 4 ranks, and each rank owned exactly
one subtensor. The partition is an algorithmic knob (it decides which pivots the
adaptive cross can see), so tying it to the machine allocation meant you could not, for
example, run a 64-subtensor partition on 4 ranks.

Version 3 separates the two:

* the **block grid** says how many subtensors the tensor is cut into;
* the **process grid** says how those subtensors are spread over MPI ranks.

Blocks are dealt out to the process grid cyclically, independently in each dimension:

```
block (b_0, ..., b_{d-1})  ->  process (b_0 % p_0, ..., b_{d-1} % p_{d-1})
```

For the k-th unfolding this is exactly a 2D block-cyclic distribution: the unfolding's
row blocks are indexed by `(b_0..b_k)` and its column blocks by `(b_{k+1}..b_{d-1})`,
so the unfolding sees a `prod(p[:k+1]) x prod(p[k+1:])` process grid with neighbouring
block rows and block columns landing on different processes. With a `(64,64)` block
grid and a `(2,2)` process grid this is the classical 2D block-cyclic matrix layout.

Every process now holds a *list* of subtensors instead of one, and all per-subtensor
state (index sets, superblocks, entry caches, cross approximations) is kept per block.
Pivot selection is a reduction over blocks rather than over ranks.

Two properties hold by construction and are verified:

1. **Setting the process grid equal to the block grid reproduces Versions 1 and 2
   exactly** - same pivots, same cores, same error.
2. **The result depends only on the block grid, never on the process grid.** The same
   64-block partition gives bit-identical pivots on 1, 4, 8 and 32 ranks. The process
   grid is a placement decision, not part of the algorithm.

Two smaller changes came with the refactor:

* Row/column group membership is an O(1) integer comparison instead of the O(#blocks)
  scan in `Subtensor_Comm`, and the cross-vector exchange uses one collective per step
  instead of O(p) point-to-point pairs per rank.
* The order in which blocks are assembled during core construction is now fixed by
  block id. It previously came out of a Python `set()`, so it varied between runs.

# Running the code

Same as before, with one new optional argument.

* 1 - Select the desired tensor. Set `TENSOR_KIND` near the top of the .py file to
  `"hilbert"`, `"maxwell4"` or `"maxwell6"`.
* 2 - Input the parameters on the command line. This file uses `sys.argv()` and takes
  5 required inputs and 1 optional one.
  * Input 1: Number of indices at which we sample error
  * Input 2: Number of test runs
  * Input 3: Tensor dimension, comma separated, e.g. `50,35,40,26`
  * Input 4: Internal core ranks, comma separated, e.g. `15,17,9`
  * Input 5: **Block grid** - how many subtensors per dimension, e.g. `8,8,1,1`
  * Input 6 (optional): **Process grid** - how many MPI ranks per dimension, e.g.
    `2,2,1,1`. If omitted it defaults to the block grid, which is the Version 1/2
    behaviour of one subtensor per rank.

Constraints: `prod(process grid)` must not exceed the number of launched ranks, the
block grid must be at least the process grid in every dimension, and every dimension
must hold at least 2 indices per block.

Run a 4D Maxwellian of size `[800,400,800,400]` with TT core ranks `(1,10,5,20,1)`,
cut into `8*8 = 64` subtensors but placed on only 16 ranks:

```
mpirun -n 16 python3 MPISubtensorTTCross_Version3_NumPy.py 100000 10 800,400,800,400 10,5,20 1,8,8,1 1,4,4,1
```

Leaving off the last argument reproduces the old run exactly:

```
mpirun -n 64 python3 MPISubtensorTTCross_Version3_NumPy.py 100000 10 800,400,800,400 10,5,20 1,8,8,1
```

# Environment variables

| variable | meaning |
| --- | --- |
| `TT_SEED` | Seed the run. Each *block* gets its own random stream keyed by its global block id, so two process grids over the same block grid produce identical output. Unset (the default) leaves the original unseeded behaviour. |
| `TT_MAP` | `cyclic` (default) or `block`. `block` gives each process one contiguous chunk of blocks per dimension instead of dealing them out round robin. It exists so the two placements can be compared at an identical block grid; both give the same numerical result. |
| `TT_PROFILE` | Set to 1 to split idle wait from communication in the per-rank report (see below). Adds one `Barrier` per collective, so leave it off for pure timing runs. |
| `TT_PROFILE_RANKS` | Print the per-rank table only when the run has at most this many ranks. Default 16; set 0 to always suppress it. |
| `TT_PROFILE_OUT` | Write one CSV row per rank to this path. Use this instead of the table at scale. |
| `TT_TORCH_NUM_THREADS` | PyTorch file only, unchanged from Version 2. |

# Measuring load balance

Total phase time per rank hides imbalance: an under-loaded rank does not finish
early, it finishes at the same moment as everyone else with the difference spent
blocked inside MPI. The report therefore splits each rank's phase time three ways.

* **compute** - local work: superblock construction, tensor-entry evaluation, the
  rank-one cross updates, the greedy pivot search, index bookkeeping.
* **comm** - time inside MPI collectives and point-to-point calls, measured *after*
  every rank has arrived, so it is the transfer cost.
* **wait** - time sitting in the `Barrier` placed in front of each collective, i.e.
  time idle because another rank had not arrived yet. **This is the load imbalance.**

The split needs `TT_PROFILE=1`; the extra Barriers are not free, so with it unset the
Barriers are skipped and idle time is folded into `comm`. A `Barrier` has its own
log(P) latency, which at large rank counts adds a small floor to the reported wait.

Output looks like:

```
Per-rank time breakdown over 8 ranks  (wait split: ON)
  PIVOT SEARCH : wall=0.3669s
    compute   max=  0.2533s (rank    0)  min=  0.1245s  avg=  0.1904s  max/avg= 1.331
    comm      max=  0.0200s (rank    5)  min=  0.0188s  avg=  0.0192s  max/avg= 1.041
    wait      max=  0.2229s (rank    7)  min=  0.0845s  avg=  0.1524s  max/avg= 1.462
    -> busiest rank 0 does 0.2533s of local work; the average rank does 0.1904s
    -> shares of the phase wall: compute 51.9%  comm 5.2%  wait 41.5%
    -> imbalance cost: avg idle 0.1524s = 80.0% of avg local work
   rank |  piv_compute   piv_comm   piv_wait |  core_compute  core_comm  core_wait
      0 |       0.2533     0.0189     0.0845 |        0.0126     0.0022     0.0000
      ...
```

At scale, write a CSV instead of the table and summarise it afterwards:

```
TT_PROFILE=1 TT_PROFILE_RANKS=0 TT_PROFILE_OUT=results/run.csv srun -n 1024 python3 ...
python3 profile_summary.py results/*.csv
```

`profile_summary.py` prints percentiles across ranks, names the busiest and idlest
rank, and reports aggregate idle rank-seconds as a fraction of the allocation held.
`perlmutter_scaling.sh` is a Slurm template that sweeps the block grid and the
placement rule at a fixed process grid, then repeats a fixed block grid at several
process counts; it writes one CSV per configuration. Its account, queue and module
lines need to be set for your allocation before submitting.

Comparing two runs is what the instrumentation is for. Because the approximation
depends only on the block grid, a change in `TT_MAP` or in the process grid at a fixed
block grid changes nothing numerically, so every difference in the profile is parallel
behaviour alone.

# Measured behaviour

**These numbers come from a 10-core laptop and are indicative only.** The problem sizes
that fit here are far too small to say anything about the layout at scale; they exist
to show the instrumentation works and that the invariants hold.

Refining the block grid at a fixed rank count is now possible and does change the
approximation. On 240x240x240 Hilbert with TT core ranks `(1,16,16,1)` on 4 ranks
(`2,2,1` process grid):

| block grid | subtensors | sampled error | pivot time |
| --- | --- | --- | --- |
| 2,2,1 | 4 | 7.00e-09 | 0.72 s |
| 4,4,1 | 16 | 2.56e-09 | 0.70 s |
| 6,6,1 | 36 | 6.72e-09 | 0.83 s |
| 8,8,1 | 64 | 6.72e-09 | 1.06 s |
| 12,12,1 | 144 | 6.15e-09 | 1.54 s |

The `4,4,1` partition is 2.7x more accurate than `2,2,1` at the same cost on the same
4 ranks; before Version 3 it would have needed 16 ranks. There is a sweet spot rather
than a monotone trend - past it the per-block Python overhead starts to dominate.

What the compute/comm/wait split shows locally, on 240^3 with `(1,12,12,1)` over 8
ranks, block grid `4,4,2`:

* roughly **40% of the pivot-search phase is idle wait**, and only ~5% is transfer.
  The bottleneck is imbalance, not communication.
* local compute spreads about **2x between the busiest and idlest rank**
  (max/avg ~ 1.33), which the old total-phase-time spread completely hid - that
  metric read under 3% because the idle time absorbs the difference.
* **cyclic versus block placement came out within run-to-run noise here.** A single
  cold trial favoured cyclic by ~25%, but with three trials the two are
  indistinguishable (wait_avg 0.130 s vs 0.136 s, compute max/avg 1.32 vs 1.37).
  At this size the per-step work is too small relative to Python and MPI overhead for
  placement to matter, so this needs re-running at scale before drawing a conclusion.

Total arithmetic work stays close to invariant under refinement: a block's superblock
shrinks about as fast as the block count grows, which the reported average superblock
area confirms.
