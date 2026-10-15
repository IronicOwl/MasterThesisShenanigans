#!/bin/bash
# Re-run selected tasks of a sweep (e.g. after a time-out or node failure).
# They resume from their checkpoints; finished sectors are skipped.
#
#   bash slurm/resubmit.sh <tag> 3,7,12-15 [extra sbatch options, e.g. --time=48:00:00]
set -euo pipefail
TAG=${1:?usage: bash slurm/resubmit.sh <tag> <array spec> [sbatch options]}
SPEC=${2:?give the array indices, e.g. 3,7,12-15}
shift 2
cd "$(dirname "$0")/.."
JOBFILE="jobs/${TAG}.txt"
[[ -f "${JOBFILE}" ]] || { echo "${JOBFILE} not found"; exit 1; }
SB_ARGS=$(cat "jobs/${TAG}.sbatch_args")
OUTDIR=$(cat "jobs/${TAG}.outdir")
mkdir -p logs
# shellcheck disable=SC2086
ARRAY_ID=$(sbatch --parsable ${SB_ARGS} "$@" --array="${SPEC}" \
           --export=ALL,JOBFILE="${JOBFILE}" slurm/run_array.sbatch)
FIN_ID=$(sbatch --parsable --job-name="tpdt_${TAG}_finalize" \
         --dependency=afterany:${ARRAY_ID} --export=ALL,RESULTS="${OUTDIR}" \
         slurm/finalize.sbatch)
echo "resubmitted ${TAG} tasks ${SPEC}: array ${ARRAY_ID}, finalize ${FIN_ID}"
