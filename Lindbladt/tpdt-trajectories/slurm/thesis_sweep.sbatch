#!/bin/bash
#SBATCH --job-name=tpdt_thesis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source slurm/env.sh
N=${N:-15}            # spins; submit with  N=20 sbatch ...  for another size
SEED=${SEED:-1}       # SEED=2 sbatch ...  adds 2000 more trajectories per point
NMAX=$((5 * N))       # every lambda starts from the same states: uniform below n = 5N

for lam in 0.49 0.50 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.60; do
    time python run_point.py --N $N --M auto --lam $lam --init-nmax $NMAX --seed $SEED \
         --n-traj 2000 --t-max 70 --workers $SLURM_CPUS_PER_TASK --outdir results/thesis
done

# thesis figures for this N
SYS=results/thesis/N${N}_k0.05_wc1_wa1_t70_Tinf-n${NMAX}
python plot_sweep.py curve $SYS --out figures/n_vs_lambda_N${N}.pdf --lam-c 0.5
python plot_sweep.py grid  $SYS --lam 0.50 0.52 0.55 0.60 --out figures/wigner_N${N}.pdf
grep -h WARNING logs/tpdt_thesis_${SLURM_JOB_ID}.out || echo "no warnings"
