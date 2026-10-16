#!/bin/bash
#SBATCH --job-name=ns_compass
#SBATCH --cpus-per-task=8          # parallel symmetry blocks (one sparse LU each)
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=ns_compass_%j.log

# Load the same Python environment you use for sweep.py, e.g.
# module load python
# source ~/venvs/qutip/bin/activate

export OMP_NUM_THREADS=1           # one BLAS thread per process (also set in the script)

# Choose what to run:
#   sbatch run_ns_compass.slurm                              -> thesis parameters
#   sbatch --export=ALL,PRESET=scaling run_ns_compass.slurm  -> finite-size scaling study
PRESET=${PRESET:-thesis}

if [ "$PRESET" = "thesis" ]; then
    # lambda=0.6, kappa2=0.05 (thesis values).  N=8 needs ~40-75 GB per block,
    # so the script runs only 1-2 blocks at a time there (about 6 h in total).
    python ns_compass.py --N 2 3 4 5 6 7 8 --lam 0.6 --kappa2 0.05 \
        --workers ${SLURM_CPUS_PER_TASK} --mem-budget 110
elif [ "$PRESET" = "scaling" ]; then
    # lambda=1.0, kappa2=0.5: the number k* of collective spin flips needed to
    # switch superradiance off grows from 1 to 4 over this range of N (see PDF).
    python ns_compass.py --N 2 3 4 5 6 7 8 9 10 11 12 13 14 --lam 1.0 --kappa2 0.5 \
        --workers ${SLURM_CPUS_PER_TASK} --mem-budget 110
fi

# Re-make figures and the summary table from the saved .npz files only:
# python ns_compass.py --plot-only --lam 0.6 --kappa2 0.05
