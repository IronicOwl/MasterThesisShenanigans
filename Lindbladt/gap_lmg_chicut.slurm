#!/bin/bash
#SBATCH --job-name=gap_lmg_chi
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G                # 16 x 8 GB = 128 GB; the script reads this and picks its worker count
#SBATCH --time=12:00:00
#SBATCH --output=gap_lmg_chi_%j.log
##SBATCH --partition=CHANGE_ME          # remove one '#' if your cluster needs a partition
#
# Single-parameter test of the 2 -> 4 -> 8 cascade: chi alone, at fixed lambda = 0.55
# (kappa2 = 0.5, gamma = 0.1, wc = wa = 1).  Mean field along this line:
#     spins order (2 -> 4) at chi = 0.261, lobes appear at chi = 0.446, Q unstable (-> 8) at chi = 0.614.
# Put the new sweep_traceless_lmg.py in the folder you submit from (your old jobs run unchanged with it).
#
# run B, the verdict: N = 8 ... 22 at five points (chi = 0.15 N, 0.40 Q, 0.80 / 1.00 / 1.20 S),
#     sectors E_++_-1 (weak breaking) and E_+-_i (strong breaking) only; run A has E_--_-1 and
#     E_+-_-i up to N = 16.                                 -> Gap_Closure_LMG_chicut_bigN/   ~1.5 h
# run A, the sweep: 29 chi points (dense around chi_c and the lobe transition), N = 4 ... 16,
#     all sectors incl. E_++_1, E_--_1.                    -> Gap_Closure_LMG_chicut/        ~1.5 h
# Times: the script's cost model (N=10: ~100 s at block size 78408), before cutoff retries.
# Every finished evaluation is saved immediately (data/points_*/): if the job hits the time limit
# or a worker is killed (exit code 3), resubmit unchanged; nothing finished is computed twice.
#
# What decides it: gap_scaling.png and summary.txt in Gap_Closure_LMG_chicut_bigN/
#   2 -> 4: at chi = 0.40 (Q), E_++_-1 shrinks with N (filled markers, no '~' in summary.txt),
#           while E_+-_i stays at ~0.4-0.5 and oscillates (open markers, '~').
#   4 -> 8: at chi = 0.80-1.20 (S), E_+-_i must stop oscillating (markers turn filled) and fall
#           off faster with N.  Sandbox check at chi = 1.0, N = 4/6/8/10: 0.40 0.34 0.31 0.28, still
#           oscillating.  If it still oscillates at N = 22, the 8 is not visible on this cut at
#           reachable N.
#
# Optional, if N = 22 is not conclusive: N = 24 needs up to ~65 GB and ~45 min per evaluation, one
# at a time (~4.5 h).  Submit separately; it lands in the same folder and figure:
#     python sweep_traceless_lmg.py --lam 0.55 --chis 0.15 0.40 0.80 1.00 1.20 --Ns 24 \
#         --sectors E_++_-1 E_+-_i --outdir Gap_Closure_LMG_chicut_bigN

module load stack/2025-06 gcc/8.5.0
module load python/3.11.9
source /cluster/home/egoldhahn/masterarbeit/MasterThesisShenanigans/.venv/bin/activate
export PYTHONUNBUFFERED=1

# block construction vs full Liouvillian, symmetry checks (a few seconds); stop if it fails
python sweep_traceless_lmg_modified.py --selfcheck || exit 1

LAM=0.55

# run B: N scaling at five points
python sweep_traceless_lmg_modified.py --lam $LAM --chis 0.15 0.40 0.80 1.00 1.20 \
    --Ns 8 10 12 14 16 18 20 22 --sectors E_++_-1 E_+-_i \
    --outdir Gap_Closure_LMG_chicut_bigN

# run A: the sweep
python sweep_traceless_lmg_modified.py --lam $LAM \
    --chis 0.10 0.15 0.20 0.23 0.25 0.26 0.27 0.28 0.30 0.33 0.36 0.40 0.44 0.47 0.50 \
           0.53 0.56 0.59 0.62 0.66 0.70 0.75 0.80 0.90 1.00 1.10 1.20 1.30 1.40 \
    --Ns 4 6 8 10 12 14 16 --with-trace \
    --scaling-chis 0.15 0.40 0.53 0.80 1.00 1.20 \
    --outdir Gap_Closure_LMG_chicut
