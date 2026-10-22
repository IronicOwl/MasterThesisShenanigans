#!/bin/bash
#SBATCH --job-name=gapscan
#SBATCH --array=3-9
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G 
#SBATCH --time=24:00:00
#SBATCH --output=gapscan_N%a.log
# One array task per N.  Afterwards:  python gap_scan.py analyze
# Memory per worker is set by the sparse LU.  The first log line of each run prints
# its size; budget ~2.5x that per worker.  Estimates for kappa2 = 0.05 near lambda* ~ 0.55:
#   N=6: 1.6 GB LU,  ~40 s per lambda      N=8: ~11 GB LU, ~8 min per lambda
#   N=7: ~6 GB LU,   ~3 min per lambda     N=9: ~23 GB LU, ~20-25 min per lambda
N=$SLURM_ARRAY_TASK_ID
case $N in
  3|4|5) ARGS="--workers 8" ;;                                          # default window 0.51-0.60
  6)     ARGS="--workers 8 --lam-lo 0.535 --lam-hi 0.575 --ngrid 8" ;;
  7)     ARGS="--workers 6 --lam-lo 0.540 --lam-hi 0.570 --ngrid 6" ;;
  8)     ARGS="--workers 3 --lam-lo 0.545 --lam-hi 0.568 --ngrid 6 --zoom-points 3 --no-convcheck" ;;
  9)     ARGS="--workers 1 --lam-lo 0.548 --lam-hi 0.566 --ngrid 5 --zoom-points 4 --no-convcheck" ;;
esac
python gap_scan.py compute --N "$N" $ARGS
