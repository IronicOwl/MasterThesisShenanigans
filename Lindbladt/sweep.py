import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
import qutip as qt
from scipy.sparse.linalg import eigs, ArpackNoConvergence
from multiprocessing import Pool, cpu_count
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Parameters ---
N = 5
M = 130
wc = 1.0
wa = 1.0
kappa2 = 0.05
target_sector = "E_--_1"
core_nr = 8

lam_min = 0.45
lam_max = 0.58
n_steps = 32
lam_vals = np.linspace(lam_min, lam_max, n_steps)

sigma = -1e-6 + 1e-8j  # fixed small offset, avoids landing exactly on zero

# --- Worker-local globals, built once per worker process ---
_g = {}

def _init_worker(N, M, wc, wa, kappa2, target_sector):
    print(f"[worker {os.getpid()}] initializing...", flush=True)

    j_spin = N / 2.0
    a = qt.tensor(qt.destroy(M), qt.qeye(N + 1))
    num_op = qt.tensor(qt.num(M), qt.qeye(N + 1))

    # Standard (unscaled) spin operators — used ONLY for building the
    # symmetry masks (Pi_2, Pi_4), since the rotation identity
    # Pi_4 Jx Pi_4^dagger = -Jx requires the standard commutator [Jz,Jx]=iJy.
    Jx_std = qt.tensor(qt.qeye(M), qt.jmat(j_spin, 'x'))
    Jz_std = qt.tensor(qt.qeye(M), qt.jmat(j_spin, 'z'))

    # Paper-convention (2x scaled) spin operators — used ONLY in the
    # Hamiltonian, to match the paper's stated coefficients (wa/2, lam/N).
    Jx_H = 2 * Jx_std
    Jz_H = 2 * Jz_std

    L2 = np.sqrt(kappa2 / N) * a**2

    # --- Masks built from the STANDARD (unscaled) Jz ---
    diag_num = num_op.diag()
    diag_Jz = Jz_std.diag()
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
        "E_++_1":  E_plus_plus & E_1, "E_++_-1": E_plus_plus & E_m1,
        "E_--_1":  E_minus_minus & E_1, "E_--_-1": E_minus_minus & E_m1,
        "E_+-_i":  E_plus_minus & E_i, "E_+-_-i": E_plus_minus & E_mi,
        "E_-+_i":  E_minus_plus & E_i, "E_-+_-i": E_minus_plus & E_mi,
    }

    _g["num_op"] = num_op
    _g["Jx"] = Jx_H       # scaled version goes into the Hamiltonian
    _g["Jz"] = Jz_H       # scaled version goes into the Hamiltonian
    _g["a"] = a
    _g["L2"] = L2
    _g["mask"] = sectors[target_sector]
    _g["N"] = N
    _g["wc"] = wc
    _g["wa"] = wa

    print(f"[worker {os.getpid()}] ready. Block size: {np.sum(sectors[target_sector])}", flush=True)


def _compute_eigenvalues(lam):
    t0 = time.time()
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

    try:
        evals, _ = eigs(L_block, k=4, sigma=sigma, which='LM',
                         maxiter=5000, tol=1e-10)
        order = np.argsort(-np.real(evals))
        evals_sorted = evals[order]
        elapsed = time.time() - t0
        print(f"Lam={lam:.4f} | converged in {elapsed:.1f}s | "
              f"leading eval={evals_sorted[0]:.6e}", flush=True)
        return lam, evals_sorted

    except ArpackNoConvergence as e:
        elapsed = time.time() - t0
        print(f"Lam={lam:.4f} | DID NOT CONVERGE after {elapsed:.1f}s | "
              f"partial={e.eigenvalues}", flush=True)
        partial = np.array(e.eigenvalues)
        padded = np.full(4, np.nan, dtype=complex)
        padded[:len(partial)] = partial
        return lam, padded

    except RuntimeError as e:
        elapsed = time.time() - t0
        print(f"Lam={lam:.4f} | FACTORIZATION ERROR after {elapsed:.1f}s | "
              f"{e}", flush=True)
        return lam, np.full(4, np.nan, dtype=complex)


if __name__ == "__main__":
    n_workers = min(int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count())), core_nr)
    print(f"Starting sweep for sector: {target_sector} using {n_workers} workers")
    print(f"N={N}, M={M}, lambda range=[{lam_min}, {lam_max}], steps={n_steps}\n")

    t_start = time.time()
    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(N, M, wc, wa, kappa2, target_sector),
    ) as pool:
        results = pool.map(_compute_eigenvalues, lam_vals)

    total_time = time.time() - t_start
    print(f"\nTotal sweep time: {total_time:.1f}s")

    lam_out = np.array([r[0] for r in results])
    all_evals = np.array([r[1] for r in results])
    leading_evals = all_evals[:, 0]

    outdir = os.path.join('Plots', 'Gap_Closure', 'Plot_data')
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.join(outdir, f'eigenvalues_{target_sector}_N{N}_M{M}.npz')
    np.savez(filename, lam_vals=lam_out, leading_evals=leading_evals,
             all_evals=all_evals)
    print(f"Data saved to {filename}")
    
    outdir2 = os.path.join('Plots', 'Gap_Closure')
    os.makedirs(outdir2, exist_ok=True)
    plotname = os.path.join(outdir2, f'gap_{target_sector}_N{N}_M{M}.png')

    plt.figure(figsize=(8, 5))
    plt.plot(lam_out, -np.real(leading_evals), 'b.-', markersize=8)
    plt.axvline(0.51, color='gray', linestyle='--', alpha=0.6,
                label='Mean-field $\\lambda_c$')
    plt.xlabel('Coupling Strength $\\lambda$', fontsize=12)
    plt.ylabel(r'$-\mathrm{Re}(\lambda_{eval})$', fontsize=12)
    plt.title(f'Dissipative Gap for Sector {target_sector} (N={N}, M={M})', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plotname, dpi=150)
    print("Plot saved.")