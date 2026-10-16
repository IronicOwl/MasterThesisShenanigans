#!/bin/bash
#SBATCH --job-name=tpdt_stability
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source slurm/env.sh
# same N, same initial ensemble (uniform below n = 200), only the Fock cutoff M changes
for lam in 0.55 0.60; do
    for M in 200 260 320 400; do
        time python run_point.py --N 15 --M $M --lam $lam --init-nmax 200 --n-traj 2000 \
             --workers $SLURM_CPUS_PER_TASK --outdir results/stability
    done
done
python stability_report.py
