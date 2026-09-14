#!/bin/bash
#SBATCH --job-name=lindblad_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8G
#SBATCH --time=05:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load stack/2025-06 gcc/8.5.0
module load python/3.11.9
source .venv/bin/activate

python Lindbladt/test.py

