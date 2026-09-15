import os
import numpy as np
import qutip as qt
from scipy.sparse.linalg import eigs
from multiprocessing import Pool, cpu_count
import matplotlib
matplotlib.use("Agg")  # safe for headless/cluster use
import matplotlib.pyplot as plt

# --- 1. System Parameters & Sweep Setup ---
N = 4
M = 90
wc = 1.0
wa = 1.0
kappa2 = 0.05

lam_min = 0.45
lam_max = 0.65
n_steps = 20
lam_vals = np.linspace(lam_min, lam_max, n_steps)

target_sector = "E_+-_i"


# --- 2. Worker-local globals ---
# These get built ONCE per worker process (not once per lambda),
# via the Pool initializer below.
_g = {}


def _init_worker(N, M, wc, wa, kappa2, target_sector):
    """Runs once in each worker process to set up shared operators/masks."""
    j_spin = N / 2.0
    a = qt.tensor(qt.destroy(M), qt.qeye(N + 1))
    num_op = qt.tensor(qt.num(M), qt.qeye(N + 1))
    Jx = qt.tensor(qt.qeye(M), qt.jmat(j_spin, 'x'))
    Jz = qt.tensor(qt.qeye(M), qt.jmat(j_spin, 'z'))
    L2 = np.sqrt(kappa2 / N) * a**2

    diag_num = num_op.diag()
    diag_Jz = Jz.diag()

    diag_Pi_2 = np.exp(1j * np.pi * diag_num)
    diag_Pi_4 = np.exp(1j * np.pi * (diag_num / 2.0 + diag_Jz))

    Pi_2 = qt.qdiags(diag_Pi_2, 0, dims=num_op.dims)
    Pi_4 = qt.qdiags(diag_Pi_4, 0, dims=num_op.dims)

    diag_P2_L = np.real(qt.spre(Pi_2).diag())
    diag_P2_R = np.real(qt.spost(Pi_2).diag())

    diag_P4_L = qt.spre(Pi_4).diag()
    diag_P4_R = qt.spost(Pi_4.dag()).diag()
    diag_Z4_weak = diag_P4_L * diag_P4_R

    E_plus_plus   = (diag_P2_L > 0) & (diag_P2_R > 0)
    E_minus_minus = (diag_P2_L < 0) & (diag_P2_R < 0)
    E_plus_minus  = (diag_P2_L > 0) & (diag_P2_R < 0)
    E_minus_plus  = (diag_P2_L < 0) & (diag_P2_R > 0)

    tol = 1e-8
    E_1  = np.isclose(diag_Z4_weak, 1, atol=tol)
    E_m1 = np.isclose(diag_Z4_weak, -1, atol=tol)
    E_i  = np.isclose(diag_Z4_weak, 1j, atol=tol)
    E_mi = np.isclose(diag_Z4_weak, -1j, atol=tol)

    sectors = {
        "E_++_1":  E_plus_plus & E_1,
        "E_++_-1": E_plus_plus & E_m1,
        "E_--_1":  E_minus_minus & E_1,
        "E_--_-1": E_minus_minus & E_m1,
        "E_+-_i":  E_plus_minus & E_i,
        "E_+-_-i": E_plus_minus & E_mi,
        "E_-+_i":  E_minus_plus & E_i,
        "E_-+_-i": E_minus_plus & E_mi,
    }

    _g["num_op"] = num_op
    _g["Jx"] = Jx
    _g["Jz"] = Jz
    _g["a"] = a
    _g["L2"] = L2
    _g["mask"] = sectors[target_sector]
    _g["N"] = N
    _g["wc"] = wc
    _g["wa"] = wa


def _compute_eigenvalue(lam):
    """Runs for each lambda value, using this worker's pre-built globals."""
    num_op = _g["num_op"]
    Jx = _g["Jx"]
    Jz = _g["Jz"]
    a = _g["a"]
    L2 = _g["L2"]
    mask = _g["mask"]
    N = _g["N"]
    wc = _g["wc"]
    wa = _g["wa"]

    H = wc * num_op + (wa / 2.0) * Jz + (lam / N) * Jx * (a.dag()**2 + a**2)
    L = qt.liouvillian(H, [L2])

    try:
        L_sparse = L.data.as_scipy()
    except AttributeError:
        L_sparse = L.data

    L_block = L_sparse[mask, :][:, mask]

    evals, _ = eigs(L_block, k=1, which='LR')
    max_real_eval = evals[0]

    print(f"Lam: {lam:.3f} | Eigenvalue: {np.real(max_real_eval):.3e}", flush=True)
    return max_real_eval


if __name__ == "__main__":
    # Respect Slurm's CPU allocation if present, else fall back to all cores
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
    print(f"Starting sweep for sector: {target_sector} using {n_workers} workers")

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(N, M, wc, wa, kappa2, target_sector),
    ) as pool:
        # map preserves input order, so results line up with lam_vals
        tracked_evals = pool.map(_compute_eigenvalue, lam_vals)

    tracked_evals = np.array(tracked_evals)

    # --- Save Data ---
    filename = f'eigenvalues_{target_sector}_N{N}_M{M}.npz'
    np.savez(filename, lam_vals=lam_vals, evals=tracked_evals)
    print(f"\nData saved to {filename}")

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(lam_vals, -np.real(tracked_evals), 'b.-', markersize=8)
    plt.xlabel('Coupling Strength $\\lambda$', fontsize=12)
    plt.ylabel(r'$-\mathrm{Re}(\lambda_{eval})$', fontsize=12)
    plt.title(f'Dissipative Gap for Sector {target_sector} (N={N}, M={M})', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'gap_{target_sector}_N{N}_M{M}.png', dpi=150)
    print("Plot saved.")