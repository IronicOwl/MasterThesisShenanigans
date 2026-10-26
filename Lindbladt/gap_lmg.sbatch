#!/bin/bash
#SBATCH --job-name=gap_lmg
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G                # 16 x 8 GB = 128 GB; the script reads this and picks its worker count
#SBATCH --time=12:00:00
#SBATCH --output=gap_lmg_%j.log
##SBATCH --partition=CHANGE_ME          # remove one '#' if your cluster needs a partition
#
# Two-photon Dicke model + spin-spin term -(chi/N) Jx^2 + collective spin decay sqrt(gamma/N) J_-
# (gamma = 0.1, kappa2 = 0.5, wc = wa = 1): gap closure in the symmetry sectors.
# Put sweep_traceless_lmg.py in the folder you submit from (next to sweep_traceless.py).
#
# run 1, the cascade:  (chi, lam) = (0.10, 0.47) -> (0.70, 0.47) -> (0.70, 0.80)
#     16 points while the spins order (2 -> 4), 34 points (step 0.01) while lambda goes up
#     through the lobe transition (~0.53) deep into the 8 phase; N = 4 6 8 10 12;
#     traceless sectors + E_++_1, E_--_1 (--with-trace shows where the system switches).
#     -> Gap_Closure_LMG/        expected ~1.5 h (N = 12 dominates, 6 parallel workers)
# run 2, the 8 phase at larger N:  lambda = 0.55 ... 0.80 at chi = 0.7, 11 points,
#     N = 6 8 10 12 14 (four traceless sectors; the E_-+ ones are filled in by conjugation).
#     -> Gap_Closure_LMG_S/      expected ~2 h (N = 14 dominates, 3 parallel workers)
# Times are estimates from single evaluations on a comparable CPU (N=10: ~100 s, N=14: ~20 min).
# Each N is saved as soon as it is finished; resubmitting the same job skips finished N.
# The figure to look at is zero_mode_distance_log.png (smallest |eigenvalue| per sector).
#
# N = 16 needs ~45 min and up to ~60 GB per evaluation: submit separately with
#     python sweep_traceless_lmg.py --Ns 16 --path 0.70,0.55 0.70,0.80 --n-steps 11 --outdir Gap_Closure_LMG_S
# and --time=24:00:00 (it is added to the same folder and figure).

module load stack/2025-06 gcc/8.5.0
module load python/3.11.9
source /cluster/home/egoldhahn/masterarbeit/MasterThesisShenanigans/.venv/bin/activate
export PYTHONUNBUFFERED=1

# block construction vs full Liouvillian, symmetry checks (a few seconds); stop if it fails
python sweep_traceless_lmg.py --selfcheck || exit 1

python sweep_traceless_lmg.py --Ns 4 6 8 10 12 \
    --path 0.10,0.47 0.70,0.47 0.70,0.80 --n-steps 16 34 --with-trace \
    --outdir Gap_Closure_LMG

python sweep_traceless_lmg.py --Ns 6 8 10 12 14 \
    --path 0.70,0.55 0.70,0.80 --n-steps 11 \
    --outdir Gap_Closure_LMG_S
