#!/bin/bash
#SBATCH --job-name=tpdt_n15_check
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source slurm/env.sh

# your old runs: same N, M, lambda, t_max, now 2000 trajectories per sector
time python run_point.py --N 15 --M 200 --lam 0.53 --n-traj 2000 --workers $SLURM_CPUS_PER_TASK --outdir results/n15_check
time python run_point.py --N 15 --M 210 --lam 0.54 --n-traj 2000 --workers $SLURM_CPUS_PER_TASK --outdir results/n15_check
time python run_point.py --N 15 --M 200 --lam 0.60 --n-traj 2000 --workers $SLURM_CPUS_PER_TASK --outdir results/n15_check

# lambda = 0.6 again with a larger cutoff: was M = 200 big enough?
time python run_point.py --N 15 --M 320 --lam 0.60 --n-traj 2000 --workers $SLURM_CPUS_PER_TASK --outdir results/n15_check
