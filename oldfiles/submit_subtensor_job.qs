#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --job-name=Subtensor_Job
#SBATCH --partition=standard
#SBATCH --mem-per-cpu 15G
#SBATCH --time=10:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --export=NONE

vpkg_require openmpi/4.1.0 python

source ./subten-mpi-env/bin/activate

mpirun python SubtensorTTCrossMPI.py 50,50,50 10,10 2,2,2 100
