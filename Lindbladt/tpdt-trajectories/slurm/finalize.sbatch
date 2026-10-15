#!/bin/bash
# Merges all seeds/chunks, computes Wigner functions + P(n) and plots for every
# point under $RESULTS that changed since its last summary. Submitted by
# slurm/submit.sh with a dependency on the trajectory array.
#SBATCH --job-name=tpdt_finalize
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
: "${RESULTS:?RESULTS not set}"
source slurm/env.sh
python finalize.py "${RESULTS}"
