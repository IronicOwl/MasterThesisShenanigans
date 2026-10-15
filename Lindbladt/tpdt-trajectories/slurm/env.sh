# Sourced by every job script. Same modules and venv as your other jobs.
case $- in *u*) _tpdt_u=1;; *) _tpdt_u=0;; esac
set +u      # module/venv scripts don't always tolerate 'set -u'

module load stack/2025-06 gcc/8.5.0
module load python/3.11.9

# code folder = parent of slurm/; the venv sits two levels above it
_tpdt_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${TPDT_VENV:-$_tpdt_root/../../.venv}/bin/activate"

if [ "$_tpdt_u" = 1 ]; then set -u; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
