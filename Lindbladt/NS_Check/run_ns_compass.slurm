#!/bin/bash
#SBATCH --job-name=ns_compass
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # parallel symmetry blocks (the script uses at most 6)
#SBATCH --mem-per-cpu=16G          # 8 x 16 GB = 128 GB; the script plans with 110 GB
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Submit from "master thesis shenanigans" (where .venv lives):
#   (test first: see PRESET=test below)
#   sbatch Lindbladt/NS_Check/run_ns_compass.slurm                              -> thesis parameters
#   sbatch --export=ALL,PRESET=scaling Lindbladt/NS_Check/run_ns_compass.slurm  -> scaling study
# The .out/.err logs land in the folder you submit from; the results go to
# Lindbladt/NS_Check/ns_results/.

module load stack/2025-06 gcc/8.5.0
module load python/3.11.9
cd "$SLURM_SUBMIT_DIR" || exit 1
source .venv/bin/activate
cd Lindbladt/NS_Check || { echo "code folder not found"; exit 1; }

export OMP_NUM_THREADS=1           # one BLAS thread per process (also set in the script)

PRESET=${PRESET:-thesis}

if [ "$PRESET" = "test" ]; then
    # quick check of the installation (a few minutes); see the PDF, Tables 2-3, for
    # the expected numbers.  Submit with smaller resources, e.g.
    #   sbatch --export=ALL,PRESET=test --cpus-per-task=2 --mem-per-cpu=4G --time=00:15:00 \
    #          Lindbladt/NS_Check/run_ns_compass.slurm
    time python ns_compass.py --N 2 3 --lam 0.6 --kappa2 0.05 \
        --workers ${SLURM_CPUS_PER_TASK} --mem-budget 6
elif [ "$PRESET" = "thesis" ]; then
    # lambda=0.6, kappa2=0.05 (thesis values).  N=8 needs ~40-75 GB per block,
    # so the script runs only 1-2 blocks at a time there (about 6 h in total).
    time python ns_compass.py --N 2 3 4 5 6 7 8 --lam 0.6 --kappa2 0.05 \
        --workers ${SLURM_CPUS_PER_TASK} --mem-budget 110
elif [ "$PRESET" = "scaling" ]; then
    # lambda=1.0, kappa2=0.5: the number k* of collective spin flips needed to
    # switch superradiance off grows from 1 to 4 over this range of N (see PDF).
    time python ns_compass.py --N 2 3 4 5 6 7 8 9 10 11 12 13 14 --lam 1.0 --kappa2 0.5 \
        --workers ${SLURM_CPUS_PER_TASK} --mem-budget 110
fi

# Re-make figures and the summary table from the saved .npz files only:
# python ns_compass.py --plot-only --lam 0.6 --kappa2 0.05
