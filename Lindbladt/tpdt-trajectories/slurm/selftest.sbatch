#!/bin/bash
#SBATCH --job-name=tpdt_selftest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source slurm/env.sh
python -c "import qutip; print('qutip', qutip.__version__)"
time python tests/selftest.py
time python run_point.py --N 6 --M 80 --lam 0.53 --n-traj 100 \
     --workers $SLURM_CPUS_PER_TASK --outdir results/try
