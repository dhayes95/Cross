#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --job-name=openmpi_job_test
#SBATCH --partition=standard
#SBATCH --mem-per-cpu 15G
#SBATCH --time=10:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --export=NONE
#SBATCH --mail-user=dphayes@udel.edu
#SBATCH --mail-type=END
UD_QUIET_JOB_SETUP=YES

vpkg_require openmpi/4.1.0 python

source ./subten-mpi-env/bin/activate

mpirun python MPIOptimized.py 50,50,50,50,50 10,10,10,10 1,2,4,2,2 100