#!/bin/sh

#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=0
#SBATCH --account=Education-EEMCS-Courses-WI4450
#SBATCH --reservation=subnumatest

# run CG for a n^3 Poisson problem
n=600
#export OMP_PROC_BIND=close
#export OMP_PLACES=threads
#echo $threads
#srun likwid-pin -c M0:1-$threads ../main_cg_poisson.x $n

export OMP_NUM_THREADS=$1
export OMP_PROC_BIND=close
export OMP_PLACES=threads
echo $OMP_NUM_THREADS
srun ../main_cg_poisson.x $n
