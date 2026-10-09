#!/bin/bash
# Template Slurm job for a Perlmutter CPU run.  It sweeps the block grid at a
# fixed process grid and writes one per-rank CSV per configuration, so the
# compute / comm / wait split can be compared across layouts afterwards with
#
#     python3 profile_summary.py results/*.csv
#
# CHECK BEFORE SUBMITTING: the account, the queue, and the module lines below
# depend on your allocation and on whichever Python provides mpi4py in your
# environment.  Nothing here is auto-detected.

#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --job-name=ttcross-blockcyclic
#SBATCH --output=ttcross-%j.out

module load python                 # must provide mpi4py (and torch for the PyTorch file)

CODE=MPISubtensorTTCross_Version3_NumPy.py
DIMS=1024,1024,1024,1024
RANKS=20,30,20                     # TT core ranks
ERRSAMPLES=100000
TRIALS=3                           # first trial warms caches; the report covers the last one

PROCGRID=16,16,1,1                 # 256 ranks = 4 nodes x 64
NTASKS=256

mkdir -p results

# One MPI thread per rank; let Slurm handle the placement.
export OMP_NUM_THREADS=1
export TT_TORCH_NUM_THREADS=1
export TT_SEED=1                   # per-block streams: identical math across layouts
export TT_PROFILE=1                # split idle wait from communication
export TT_PROFILE_RANKS=0          # suppress the per-rank table, use the CSV instead

# Sweep the block grid at a fixed process grid, and the placement rule at a
# fixed block grid.  BLOCKS == PROCGRID reproduces Version 1/2 exactly and is
# the baseline every other row should be compared against.
for CFG in "16,16,1,1 cyclic" \
           "32,32,2,2 cyclic" \
           "32,32,2,2 block" \
           "64,64,4,4 cyclic" \
           "64,64,4,4 block" ; do
  set -- $CFG
  BLOCKS=$1
  MAP=$2
  TAG="b${BLOCKS//,/x}_p${PROCGRID//,/x}_${MAP}"
  echo "=== $TAG ==="
  TT_MAP=$MAP TT_PROFILE_OUT=results/${TAG}.csv \
    srun -n $NTASKS python3 $CODE $ERRSAMPLES $TRIALS $DIMS $RANKS $BLOCKS $PROCGRID
done

# Strong scaling at a fixed block grid: the approximation is identical at every
# process count, so any difference in the numbers is pure parallel behaviour.
BLOCKS=64,64,4,4
for PG in "8,8,1,1 64" "16,16,1,1 256" "16,16,2,2 1024" ; do
  set -- $PG
  P=$1
  N=$2
  [ $N -gt $SLURM_NTASKS ] && continue
  TAG="scale_p${P//,/x}"
  echo "=== $TAG ==="
  TT_PROFILE_OUT=results/${TAG}.csv \
    srun -n $N python3 $CODE $ERRSAMPLES $TRIALS $DIMS $RANKS $BLOCKS $P
done
