#!/bin/bash
#SBATCH --job-name=gap_traceless
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G               # 8 x 15 GB = 120 GB; the script reads this and picks its worker count
#SBATCH --time=12:00:00
#SBATCH --output=gap_traceless_%j.log
##SBATCH --partition=CHANGE_ME          # remove one '#' if your cluster needs a partition
#
# Gap closure in the six traceless sectors for N = 5, 6, 7 over lambda = 0.48 ... 0.58.
# Results: Gap_Closure_Traceless/ (data/*.npz, run_info.json, two PNG figures).
# Expected run time ~2 h (N=7 dominates).  For N = 6, 7, 8 use "--Ns 6 7 8" and
# set --time=24:00:00 above (N=8 alone takes roughly 8 h with 2 parallel workers).

# --- activate the Python environment you use for QuTiP (needs numpy, scipy, matplotlib) ---
# module load python
# source ~/path/to/venv/bin/activate

module load stack/2025-06 gcc/8.5.0
module load python/3.11.9
source /cluster/home/egoldhahn/masterarbeit/MasterThesisShenanigans/.venv/bin/activate

python sweep_traceless.py --Ns 5 6 7 --lam-min 0.48 --lam-max 0.58 --n-steps 33
