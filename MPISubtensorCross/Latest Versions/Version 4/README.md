# DISTRIBUTED MEMORY PARALLEL ADAPTIVE TENSOR-TRAIN CROSS APPROXIMATION - Authors: Tianyi Shi, Daniel Hayes, Jingmei Qiu

Version 4 is a **prototype of a pivot-order 2D block-cyclic layout** for the
pivot-search phase. One file is provided, `MPISubtensorTTCross_Version4_NumPy.py`.
It is self-contained; the tensor definitions, entry cache, core construction and
error sampling are those of Version 3, so its numbers are directly comparable.

# Why

The Version 3 study on Perlmutter showed that the subtensor layout leaves ranks idle
for 70-97% of the pivot-search phase. The cause is not a few hot blocks but a *slab*:
for unfolding k the superblock of a block has `|I_{k-1} ∩ block| · n_k` rows, and the
pivot prefixes `I_{k-1}` cluster where the function has its features (the origin for
Hilbert, the velocity peaks for the Maxwellian). Any layout that hashes *tensor
coordinates* to processes - cyclic, contiguous or any other per-dimension bucketing -
maps that slab onto a slice of the ranks, and the best such bucketing was measured to
be no better than cyclic. Fixed row/column communicators for all unfoldings force the
map to be per-dimension, so inside the subtensor layout there is no further freedom.

# What changed

The object worked on for unfolding k is the superblock

```
A_k = X( I_{k-1} x n_k ,  n_{k+1} x J_{k+1} )
```

whose rows are `(prefix, i_k)` pairs and whose columns are `(i_{k+1}, suffix)` pairs.
Version 4 numbers the rows by the **ordinal of the prefix in the pivot list**,
`r = pidx*n_k + i_k`, and the columns by the ordinal of the suffix,
`c = sidx*n_{k+1} + i_{k+1}`, and deals rows and columns out block-cyclically over a
`p_r x p_c` process grid by those numbers. The r-th pivot prefix found goes to process
row `(r*n_k/nb) mod p_r` wherever it lies in the tensor, so every process row holds the
same number of rows of `A_k` by construction - the same reason block-cyclic LU is
balanced regardless of where its pivots are. Memory is balanced the same way.

Same machine, same communicators, same collective pattern as the subtensor layout; the
one difference is hashing pivot ordinals instead of tensor coordinates.

Each process keeps the dense tile `A_k[rows, cols]` it owns and the factors of the cross
approximation restricted to its rows and columns,

```
E_l = A - sum_{m<l} delta_m u_m v_m^T ,   U = [u_m(rows)],  V = [v_m(cols)]
```

plus two small `l x l` matrices `Pm[m,l] = v_m(j_l)` and `Qm[m,l] = u_m(i_l)` that let
the factors be extended to new rows and columns without recomputation.

## One pivot step on unfolding k

1. Every process runs the greedy rook search of Version 3 on its own tile (random
   probes, then alternating full-column and full-row scans over the tile, up to 10
   rounds), excluding rows and columns that are already pivots.
2. `Allgather` of the candidates `(|E|, E, row, col)` over all processes; the largest
   wins, ties go to the lowest rank.
3. The pivot column segment `E_l(rows, j*)` is computed by the process column holding
   `j*` and **broadcast along the row group**; the pivot row segment by the process row
   holding `i*` and **broadcast down the column group**. The pivot owner broadcasts the
   `2l` scalars `U[i*, :l]`, `V[:l, j*]` that fill `Pm` and `Qm`.
4. The new prefix `(prefix, i_k)` is recorded as `n_{k+1}` pending rows of `A_{k+1}` and
   the new suffix `(i_{k+1}, suffix)` as `n_k` pending columns of `A_{k-1}`. Nothing is
   fetched yet.

Before an unfolding is swept, its tile is brought up to date (`sync`): each process
fetches its share of the pending columns and rows, the pivot rows at the new columns are
gathered **inside the column group**, the pivot columns at the new rows **inside the row
group** (each process contributes the ones it holds), and the factors follow from
`v_m(j) = A(i_m,j) - sum_{m'<m} delta_{m'} Qm[m',m] v_{m'}(j)` and
`u_m(i) = A(i,j_m) - sum_{m'<m} delta_{m'} u_{m'}(i) Pm[m',m]`. The extension is lazy
for the same reason it is in Version 3: an unfolding that has reached its rank is never
swept again, so its superblock stops growing. (An eager version fetched 12x more
entries on the 6D Maxwellian, whose ranks are very unequal.)

So per pivot: one all-process allgather of 4 scalars, one all-process broadcast of `2l`
scalars, one row-group broadcast of a column segment, one column-group broadcast of a
row segment; and per sweep at most one row-group and one column-group allgather of `l`
short vectors. This is the communication pattern analysed in the paper, with the
subtensor tile sides replaced by `|I_{k-1}| n_k / p_r` and `n_{k+1} |J_{k+1}| / p_c`.

## Core construction

The TT core of unfolding k is the interpolation matrix `W_k = A_k(:,J_k) A_k(I_k,J_k)^{-1}`
(rows `(prefix, i_k)`, reshaped to `r_{k-1} x n_k x r_k`), which Versions 1-3 compute by
re-pulling the pivot columns and running the column-by-column recurrence `Low_Access_Tk`.
With the factors kept here it has a closed form: since the cross approximation is exact
on the pivot rows and columns,

```
W_k = U_k · U_k(I_k, :)^{-1}
```

and `U_k(I_k, :)` is the small lower-triangular matrix whose strict lower part is `Qm`
and whose diagonal holds the pivot residuals - both already on every rank. Each process
therefore computes its rows of every core with one `r x r` triangular solve on its slice
of `U_k` (the `p_c` processes of a row split the rows), with no tensor evaluation and no
communication beyond the final gather to rank 0. Rows still pending from the lazy sync
are finished first using only the pivot columns at those rows (`|pending| x r_k`
entries). The last core is `A_{d-2}(I_{d-2}, :)`, the pivot rows of the last tile.
Error sampling is split over the ranks as well.

`TT_CHECK_CORES=1` makes rank 0 also run the Version 3 serial routine on the same pivot
sets and print the relative deviation per core; it is `1e-14` for the Hilbert tensors and
`cond(A(I,J))·eps` in general.

**`TT_TOL`.** Stop adding pivots to an unfolding once `|pivot residual| < TT_TOL · |first
residual|` (default 0, i.e. the Version 3 behaviour of stopping only at an exactly zero
residual). This matters when the requested rank exceeds the numerical rank of an
unfolding: on the 6D Maxwellian the rank-5 unfoldings have numerical rank 4, the fifth
residual is `1e-15`, `cond(A(I,J)) = 1e16`, and the core is then meaningless in both
Version 3 and Version 4 (both violate `W(I,:) = Id` by ~10%; the TT error stays small only
because the garbage lies in a direction that cancels). With `TT_TOL=1e-13` those
unfoldings stop at rank 4 and all cores agree with the serial routine to `1e-14`.
`Core ranks` always reports the ranks actually reached.

## Data layer

The layout is separate from where the tensor lives. `FunctionalTensor.fetch(indices)`
evaluates entries on request, which is what every version so far assumes. A tensor that
is already partitioned and stored plugs in through the `StoredTensor` interface:
`fetch` is called collectively with the same sequence on every process, so it can be one
`alltoallv` per round to the owners. **The stored backend is not implemented in this
prototype**; the kickstart and the error sampling would also have to go through it.

# Running the code

```
srun -n P python MPISubtensorTTCross_Version4_NumPy.py err_samples trials dims ranks [pr,pc] [nb]
```

* Inputs 1-4 are as in Versions 1-3: error samples, trials, tensor dimensions, TT core
  ranks (one per unfolding).
* Input 5 (optional): process grid `pr,pc`; `pr*pc` must equal the number of launched
  ranks. Default: the most square factorisation.
* Input 6 (optional): block size `nb` of the cyclic distribution, default 1.

Environment variables `TT_SEED`, `TT_PROFILE`, `TT_PROFILE_RANKS`, `TT_PROFILE_OUT` have
the Version 3 meanings; the CSV has the same columns, so `profile_summary.py` from
Version 3 reads it. New here: `TT_TOL` (relative stopping tolerance on the pivot
residual, default 0) and `TT_CHECK_CORES=1` (compare the cores with the serial Version 3
routine), both described under *Core construction*. The report adds the tile area per rank (sum over unfoldings) and
its max/avg, which is the layout's own balance metric, and the global size of every
`A_k` at the end.

# What is different from Version 3, deliberately

* **The result depends on the process grid, not on a block grid.** Each process
  searches its own cyclic tile, so the set of candidates - and therefore the pivots -
  changes with `p_r x p_c`. A distributed rook search (argmax along full rows and
  columns through the group communicators) would make the result independent of
  everything, at about ten small group allreduces per pivot; it is not in this
  prototype.
* The starting pivot is found globally (every rank samples and climbs, the best wins)
  instead of per block.
* If the largest remaining residual is exactly zero (or below `TT_TOL` relative to the
  first one) the unfolding stops growing instead of adding a zero pivot; `Core ranks`
  reports the ranks actually reached.
* Core construction uses the closed form above instead of the Version 3 recurrence and
  grids; the error samples are drawn per rank, so the error estimate is not the same
  random draw as Version 3's.
* There is no PyTorch file yet.

# Memory

Per process, per unfolding, one dense tile of about
`(|I_{k-1}| n_k / p_r) x (n_{k+1} |J_{k+1}| / p_c)` doubles plus the factors. The whole
superblock `A_k` is therefore held once across the machine, evenly; the hot-rank memory
of the subtensor layout (one rank holding the origin block) does not occur.

# Measured behaviour

Single Perlmutter CPU node, 64 ranks, `TT_SEED=1`, `TT_PROFILE=1`. Version 3 numbers are
the optimised Version 3 of the same day (with the row/column communicators), at its
one-block-per-rank layout and, where it was run, at its best refined block grid.
"compute max/avg" is the busiest rank's local work over the average; 1.0 is perfect.

| case (TT ranks) | Version 3: pivot time, error, hot-rank memory, compute max/avg | Version 4 (8x8 grid) |
| --- | --- | --- |
| Hilbert 512^4 (20,20,20) | 6.1 s, 7.35e-10, 0.9 GB, 4.3 | **0.25 s**, 7.26e-10, 0.12 GB, 1.1 |
| Hilbert 1024^4 (20,20,20) | 22.0 s / 14.9 s (16,16,8,8 blocks), 3.2e-8 / 3.0e-8, 3.4 GB, 4.9 / 2.7 | **0.56 s**, 1.6e-8, 0.24 GB, 1.02 |
| Hilbert 2048^4 (20,20,20) | 84.7 s / 54.8 s, 3.3e-8 / 2.5e-8, 11.3 GB / 7.2 GB | **2.7 s**, 2.4e-8, 0.63 GB, 1.04 |
| Maxwellian 6D (1600,800,...) (10,5,30,5,20) | 8.3 s / 11.5 s (2,4,8,8,4,2 blocks), 1.0e-5 / 7.7e-6, 0.6 GB, 3.0 | **0.47 s**, 8.2e-6, 0.11 GB, 1.03 |
| Maxwellian 6D (3200,1600,...) | 40.3 s / 75.3 s, 7.9e-6 / 7.8e-6, 1.8 GB | **2.0 s**, 8.3e-6, 0.25 GB, 1.02 |
| Hilbert 8D 256^8 (15,17,18,18,18,17,15) | 13.0 s, 7.9e-8, 1.7 GB, 10.8 | **0.23 s**, 7.0e-8, 0.14 GB, 1.02 |

The sampled errors are within the run-to-run spread of Version 3 over its block grids, so
the accuracy of the pivot-order tiling is comparable; it is not identical, and it varies
with the process grid (see above).

Strong scaling of the 6D Maxwellian (1600,800,...) on one node:

| ranks | pivot time | speedup | compute max/avg | share of wall: compute / comm / wait |
| --- | --- | --- | --- | --- |
| 1 | 14.4 s | 1 | 1.000 | 99.9% / 0.1% / 0% |
| 4 (2x2) | 3.49 s | 4.1 | 1.003 | 98.3% / 0.5% / 1.2% |
| 16 (4x4) | 1.07 s | 13.5 | 1.024 | 93.7% / 1.8% / 4.4% |
| 64 (8x8) | 0.47 s | 30.5 | 1.025 | 87.0% / 6.1% / 6.9% |

Idle wait is 5-14% of the pivot phase in every 64-rank run, against 68-97% for the
subtensor layout in the study that motivated this version; the remaining loss at 64 ranks
on these small problems is collective latency, not imbalance. The number of entries
fetched does not depend on the process grid.

Core construction plus error sampling (20000 samples x 10) with the closed-form solve
takes 0.06-0.19 s in all of these runs on 4-64 ranks, against 1.8-9 s for the rank-0
serial routine it replaced. The cores agree with that routine to `cond(A(I,J))·eps`:
`1e-8` for Hilbert 2048^4 and 8D, `1e-6`-`1e-5` for Hilbert 512^4 (pivot residuals
reach `1e-10`), `1e-14` for the 6D Maxwellian with `TT_TOL=1e-13`.
