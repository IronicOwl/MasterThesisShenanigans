#!/bin/bash
# One array task = one line of the job file = one run_point.py call.
# Resources come from the sweep file via slurm/submit.sh (command-line options
# override the defaults below). Add --partition / --account in the sweep's
# "slurm": {"extra": [...]} if your cluster needs them.
#SBATCH --job-name=tpdt
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --signal=B:USR1@300

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
: "${JOBFILE:?JOBFILE not set - submit with slurm/submit.sh}"
source slurm/env.sh

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${JOBFILE}")
if [[ -z "${LINE}" ]]; then
    echo "no line ${SLURM_ARRAY_TASK_ID} in ${JOBFILE}"; exit 1
fi
eval "set -- ${LINE}"

echo "job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname) with ${SLURM_CPUS_PER_TASK} cpus"
echo "run_point.py $*"

# Checkpoints are written every --checkpoint-min minutes; if the job hits its
# time limit, simply resubmit the same task (slurm/resubmit.sh) and it resumes.
python run_point.py "$@" --workers "${SLURM_CPUS_PER_TASK}" &
PID=$!
trap 'echo "time limit approaching - last checkpoint is kept; resubmit to resume"; kill -TERM ${PID}' USR1
wait ${PID}
