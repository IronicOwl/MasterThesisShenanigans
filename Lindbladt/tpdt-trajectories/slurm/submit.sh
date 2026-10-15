#!/bin/bash
# Submit a whole sweep:   bash slurm/submit.sh sweeps/example_highres.json
#
#  1. expands the sweep file into jobs/<tag>.txt (one task per line)
#  2. submits a job array over those tasks (at most max_parallel at once)
#  3. submits a finalize job that runs after the array has ended
set -euo pipefail
SWEEP=${1:?usage: bash slurm/submit.sh sweeps/<sweep>.json}
cd "$(dirname "$0")/.."
source slurm/env.sh

JOBFILE=$(python make_jobs.py "${SWEEP}" --print-jobfile)
TAG=$(basename "${JOBFILE}" .txt)
NTASKS=$(grep -c . "${JOBFILE}")
SB_ARGS=$(cat "jobs/${TAG}.sbatch_args")
MAXPAR=$(cat "jobs/${TAG}.max_parallel")
OUTDIR=$(cat "jobs/${TAG}.outdir")
mkdir -p logs

# shellcheck disable=SC2086
ARRAY_ID=$(sbatch --parsable ${SB_ARGS} --array=1-${NTASKS}%${MAXPAR} \
           --export=ALL,JOBFILE="${JOBFILE}" slurm/run_array.sbatch)
FIN_ID=$(sbatch --parsable --job-name="tpdt_${TAG}_finalize" \
         --dependency=afterany:${ARRAY_ID} --export=ALL,RESULTS="${OUTDIR}" \
         slurm/finalize.sbatch)

echo "sweep ${TAG}: array ${ARRAY_ID} (${NTASKS} tasks, max ${MAXPAR} at once), finalize ${FIN_ID}"
echo "results -> ${OUTDIR}   logs -> logs/"
echo "monitor:  squeue -u \$USER    |   tail -f logs/tpdt_${TAG}_${ARRAY_ID}_1.out"
