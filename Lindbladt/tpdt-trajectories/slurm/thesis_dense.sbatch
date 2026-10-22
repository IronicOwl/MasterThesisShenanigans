#!/bin/bash
#SBATCH --job-name=tpdt_dense
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source slurm/env.sh
N=${N:-15}
SEED=${SEED:-1}
NMAX=$((5 * N))      # same starting ensemble as the main sweep -> same results folder
# extra points around the transition; the 0.01 steps already exist
LAMS=${LAMS:-"0.5025 0.505 0.5075 0.5125 0.515 0.5175 0.5225 0.525 0.5275 0.5325 0.535 0.5375"}

for lam in $LAMS; do
    time python run_point.py --N $N --M auto --lam $lam --init-nmax $NMAX --seed $SEED \
         --n-traj 2000 --t-max 70 --workers $SLURM_CPUS_PER_TASK --outdir results/thesis
done
