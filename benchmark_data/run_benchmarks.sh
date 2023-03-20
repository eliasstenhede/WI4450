#!/bin/sh

benchmarks=("store_avx" "ddot_avx" "daxpy_avx" "copy_avx")
threadsl=(48 32 24 20 16 12 10 8 6 4 2 1)

for threads in "${threadsl[@]}"; do
  for benchmark in "${benchmarks[@]}"; do
		sbatch ./run_likwid_benchmarks.slurm $threads $benchmark
  done
	sbatch ./run_cg_poisson.slurm $threads
done
