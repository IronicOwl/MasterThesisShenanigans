# Sourced by every SLURM job and by slurm/submit.sh. EDIT THIS for your cluster.
#
# Load a Python module (if your cluster uses modules) and activate the virtual
# environment that has qutip, numpy, scipy and matplotlib installed, e.g.
#
#   module load stack/2024-06 python/3.11.6        # example; see `module avail python`
#   python -m venv $HOME/venvs/tpdt && source $HOME/venvs/tpdt/bin/activate
#   pip install -r requirements.txt
#
# module load <python module>
source "${TPDT_VENV:-$HOME/venvs/tpdt}/bin/activate"

# one BLAS/OpenMP thread per process (parallelism comes from the worker pool)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
