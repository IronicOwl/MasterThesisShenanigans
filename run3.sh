#!/bin/bash
#SBATCH --job-name=gap_sweep_N10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err


module load stack/2025-06 gcc/8.5.0
module load python/3.11.9
source .venv/bin/activate

time python Lindbladt/test_sweep.py