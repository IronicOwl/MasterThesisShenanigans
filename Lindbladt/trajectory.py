"""
Parity-resolved quantum-trajectory steady state of the two-photon Dicke model
with pure two-photon loss, parallelised over trajectories for SLURM.

    H  = wc a^dag a + (wa/2) Jz + (lam/N) Jx (a^dag^2 + a^2)      (J = 2 * std. spin ops)
    L2 = sqrt(kappa2/N) a^2

Initial states are drawn from the infinite-temperature ensemble restricted to
even (or odd) photon number, each trajectory is evolved to t_max, and
rho_ss ~ (1/n_traj) sum |psi(t_max)><psi(t_max)|.

Outputs (per parity):
    <outdir>/Plot_Data/TPDT_N.._M.._l.._t.._n.._{E|O}        cavity rho (qt.qsave, .qu)
    <outdir>/Plot_Data/TPDT_..._{E|O}_full                   full cavity+spin rho (optional)
    <outdir>/Plot_Data/TPDT_..._{E|O}.npz                    P(n), Wigner grid, parameters
    <outdir>/wigner_TPDT_..._{E|O}.png
    <outdir>/photon_dist_TPDT_..._{E|O or EO}.png

Example:
    python qt_parity_sampling.py --parity both --lam 0.53 --n_traj 3000
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import math
import time
import numpy as np
import qutip as qt
from multiprocessing import Pool, cpu_count
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUTIP5 = int(qt.__version__.split(".")[0]) >= 5


# ----------------------------------------------------------------------------
# Command-line parameters (defaults = values from the notebook)
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, default=15, help="number of spins")
    p.add_argument("--M", type=int, default=120, help="Fock space truncation")
    p.add_argument("--wc", type=float, default=1.0)
    p.add_argument("--wa", type=float, default=1.0)
    p.add_argument("--kappa2", type=float, default=0.05)
    p.add_argument("--lam", type=float, default=0.53)
    p.add_argument("--t_max", type=float, default=70.0)
    p.add_argument("--n_traj", type=int, default=3000,
                   help="trajectories PER parity sector")
    p.add_argument("--parity", choices=["even", "odd", "both"], default="both")
    p.add_argument("--seed", type=int, default=12345, help="master RNG seed")
    p.add_argument("--max_workers", type=int, default=64,
                   help="upper bound on worker processes")
    p.add_argument("--chunks_per_worker", type=int, default=4,
                   help="tasks per worker (load balancing)")
    p.add_argument("--x_max", type=float, default=15.0)
    p.add_argument("--wigner_points", type=int, default=200)
    p.add_argument("--no_full_rho", action="store_true",
                   help="do not save the full (cavity x spin) density matrix")
    p.add_argument("--outdir", default=os.path.join("Plots", "Two_photon_dicke_2p_jump"))
    return p.parse_args()


# ----------------------------------------------------------------------------
# Worker-local globals, built once per worker process
# ----------------------------------------------------------------------------
_g = {}


def _init_worker(N, M, wc, wa, kappa2, t_max):
    dim_spin = N + 1
    j = N / 2.0
    a = qt.tensor(qt.destroy(M), qt.qeye(dim_spin))
    Jx = 2 * qt.tensor(qt.qeye(M), qt.jmat(j, "x"))
    Jz = 2 * qt.tensor(qt.qeye(M), qt.jmat(j, "z"))

    _g.update(N=N, M=M, dim_spin=dim_spin, wc=wc, wa=wa, a=a, Jx=Jx, Jz=Jz,
              c_ops=[np.sqrt(kappa2 / N) * a**2], tlist=[0.0, t_max],
              solver=None, lam=None)
    print(f"[worker {os.getpid()}] ready, Hilbert dim = {M * dim_spin}", flush=True)


def _get_solver(lam):
    """Build H (and, for QuTiP 5, a reusable MCSolver) for this lambda, cached per worker."""
    if _g["lam"] == lam:
        return
    N, a, Jx, Jz = _g["N"], _g["a"], _g["Jx"], _g["Jz"]
    H = (_g["wc"] * a.dag() * a + (_g["wa"] / 2.0) * Jz
         + (lam / N) * Jx * (a.dag()**2 + a**2))
    _g["H"] = H
    _g["lam"] = lam
    if QUTIP5:
        opts = {"store_states": False, "store_final_state": True,
                "keep_runs_results": True, "progress_bar": False, "map": "serial"}
        _g["solver"] = qt.MCSolver(H, _g["c_ops"], options=opts)


def _final_state_array(res):
    """Return the final state of a single-trajectory result as a numpy vector or matrix."""
    obj = None
    runs = getattr(res, "runs_final_states", None)          # QuTiP 5, keep_runs_results
    if runs is not None and len(runs) > 0:
        obj = runs[0]
    if obj is None:
        obj = getattr(res, "final_state", None)             # QuTiP 4/5 averaged final state
    if obj is None:
        obj = res.states                                    # fallback: last stored state
        while not isinstance(obj, qt.Qobj):
            obj = obj[-1]
    arr = obj.full()
    return arr.ravel() if obj.isket else arr


def _run_one(psi0, seed):
    if QUTIP5:
        res = _g["solver"].run(psi0, _g["tlist"], ntraj=1, seeds=seed)
    else:
        np.random.seed(seed % (2**32))
        opts = qt.Options(store_states=False, store_final_state=True)
        opts.seeds = [seed % (2**31)]
        res = qt.mcsolve(_g["H"], psi0, _g["tlist"], _g["c_ops"], [], ntraj=1,
                         options=opts, progress_bar=None, map_func=qt.serial_map)
    return _final_state_array(res)


def _run_chunk(task):
    """Run a chunk of trajectories and return the (unnormalised) sum of |psi><psi|."""
    lam, parity, n_in_chunk, seedseq, chunk_id = task
    _get_solver(lam)
    M, dim_spin = _g["M"], _g["dim_spin"]
    rng = np.random.default_rng(seedseq)
    offset = 0 if parity == "even" else 1
    n_pairs = len(range(offset, M, 2))       # admissible Fock states of this parity

    dim = M * dim_spin
    acc = np.zeros((dim, dim), dtype=complex)
    t0 = time.time()
    for _ in range(n_in_chunk):
        n_cav = offset + 2 * rng.integers(0, n_pairs)
        n_spin = rng.integers(0, dim_spin)
        psi0 = qt.tensor(qt.basis(M, int(n_cav)), qt.basis(dim_spin, int(n_spin)))
        st = _run_one(psi0, int(rng.integers(0, 2**31 - 1)))
        if st.ndim == 1:
            st = st / np.linalg.norm(st)
            acc += np.outer(st, st.conj())
        else:
            acc += st / np.trace(st).real
    print(f"[worker {os.getpid()}] {parity} chunk {chunk_id}: {n_in_chunk} traj "
          f"in {time.time() - t0:.1f}s", flush=True)
    return parity, n_in_chunk, acc


# ----------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------
def cavity_reduced(rho_full, M, dim_spin):
    r = rho_full.reshape(M, dim_spin, M, dim_spin)
    return np.einsum("ijkj->ik", r)


def plot_wigner(rho_cav, xvec, title, fname):
    W = qt.wigner(rho_cav, xvec, xvec)
    w_max = np.max(np.abs(W)) or 1.0
    fig, ax = plt.subplots(figsize=(6, 6))
    mesh = ax.pcolormesh(xvec, xvec, W, cmap="RdBu", shading="auto",
                         vmin=-w_max, vmax=w_max)
    ax.set_xlabel("X")
    ax.set_ylabel("P")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="Wigner Function $W(X,P)$")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return W


def plot_photon_dists(P_dict, M, lam, fname):
    labels = list(P_dict.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5), squeeze=False)
    for ax, label in zip(axes[0], labels):
        P_n = P_dict[label]
        ax.bar(range(M), P_n[:M], color="cyan", alpha=0.7, label=f"{label} sampling")
        ax.set_xlabel("Fock State (n)")
        ax.set_ylabel("Probability P(n)")
        ax.set_title(rf"Photon Distribution at $\lambda={lam}$ for {label} simulation")
        ax.set_xlim(0, M)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    parities = ["even", "odd"] if args.parity == "both" else [args.parity]
    n_workers = min(int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count())),
                    args.max_workers)

    lam_str = f"{args.lam:g}".replace(".", "")
    tmax_str = f"{args.t_max:g}".replace(".", "p")
    base = f"TPDT_N{args.N}_M{args.M}_l{lam_str}_t{tmax_str}_n{args.n_traj}"
    datadir = os.path.join(args.outdir, "Plot_Data")
    os.makedirs(datadir, exist_ok=True)

    print(f"QuTiP {qt.__version__} | workers: {n_workers} | parities: {parities}")
    print(f"N={args.N}, M={args.M}, lam={args.lam}, kappa2={args.kappa2}, "
          f"t_max={args.t_max}, n_traj/parity={args.n_traj}\n", flush=True)

    # Split trajectories into chunks with independent RNG streams
    n_chunks = max(1, min(args.n_traj, args.chunks_per_worker * n_workers))
    sizes = [args.n_traj // n_chunks + (1 if i < args.n_traj % n_chunks else 0)
             for i in range(n_chunks)]
    master = np.random.SeedSequence(args.seed)
    tasks = []
    for parity, ss_par in zip(parities, master.spawn(len(parities))):
        for cid, (sz, ss) in enumerate(zip(sizes, ss_par.spawn(n_chunks))):
            if sz > 0:
                tasks.append((args.lam, parity, sz, ss, cid))

    dim = args.M * (args.N + 1)
    rho_sum = {p: np.zeros((dim, dim), dtype=complex) for p in parities}
    count = {p: 0 for p in parities}

    t_start = time.time()
    with Pool(processes=n_workers, initializer=_init_worker,
              initargs=(args.N, args.M, args.wc, args.wa, args.kappa2, args.t_max)) as pool:
        for parity, n_done, acc in pool.imap_unordered(_run_chunk, tasks):
            rho_sum[parity] += acc
            count[parity] += n_done
            print(f"--> {parity}: {count[parity]}/{args.n_traj} trajectories "
                  f"({time.time() - t_start:.0f}s elapsed)", flush=True)
    print(f"\nTotal trajectory time: {time.time() - t_start:.1f}s", flush=True)

    # Post-processing
    xvec = np.linspace(-args.x_max, args.x_max, args.wigner_points)
    dims_full = [[args.M, args.N + 1], [args.M, args.N + 1]]
    P_all = {}
    for parity in parities:
        suffix = "E" if parity == "even" else "O"
        name = f"{base}_{suffix}"

        rho_full = rho_sum[parity] / count[parity]
        rho_full = 0.5 * (rho_full + rho_full.conj().T)       # enforce hermiticity
        rho_cav_np = cavity_reduced(rho_full, args.M, args.N + 1)
        rho_cav = qt.Qobj(rho_cav_np)
        P_n = np.real(np.diag(rho_cav_np))
        n_mean = float(np.sum(np.arange(args.M) * P_n))
        print(f"[{parity}] Tr(rho)={np.trace(rho_full).real:.6f}, <n>={n_mean:.3f}, "
              f"<n>/N={n_mean / args.N:.4f}, P(M-1)={P_n[-1]:.2e}", flush=True)

        # Same format as in the notebook: reduced cavity density matrix via qsave
        qt.qsave(rho_cav, os.path.join(datadir, name))
        if not args.no_full_rho:
            qt.qsave(qt.Qobj(rho_full, dims=dims_full), os.path.join(datadir, name + "_full"))

        W = plot_wigner(rho_cav, xvec,
                        rf"QT Wigner Function (N={args.N}, $\lambda$={args.lam}) "
                        f"{parity.capitalize()} Sampling",
                        os.path.join(args.outdir, f"wigner_{name}.png"))

        np.savez(os.path.join(datadir, name + ".npz"),
                 P_n=P_n, xvec=xvec, W=W, rho_cavity=rho_cav_np,
                 N=args.N, M=args.M, wc=args.wc, wa=args.wa, kappa2=args.kappa2,
                 lam=args.lam, t_max=args.t_max, n_traj=count[parity],
                 seed=args.seed, parity=parity)
        P_all[parity] = P_n

    suffix_all = "".join("E" if p == "even" else "O" for p in parities)
    plot_photon_dists(P_all, args.M, args.lam,
                      os.path.join(args.outdir, f"photon_dist_{base}_{suffix_all}.png"))
    print(f"All data saved to {datadir}, plots to {args.outdir}")