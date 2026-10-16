"""
Parallel quantum-trajectory runner for one parity sector.

Every trajectory starts in a basis state |n> (x) |m> of the chosen sector,
drawn from the infinite-temperature state of that sector (or from the vacuum),
and is evolved with qutip.MCSolver to t_max. Because the initial states are
sampled uniformly, the average over trajectories of |psi(t)><psi(t)| is an
unbiased estimate of exp(L t) rho_0 with rho_0 the maximally mixed state of the
sector, which is exactly what the original notebook computed.

Nothing of size dim^2 is ever stored: each worker only accumulates
  * the reduced cavity state   sum_traj Tr_spin |psi><psi|   (M/2 x M/2)
  * the reduced spin state     sum_traj Tr_cav  |psi><psi|   ((N+1) x (N+1))
  * <n>(t), <Jz>(t) and their squares on the output time grid
  * per-trajectory final values of <n>, <Jz>, <a^4>, <Jx a^2> and the initial state
which are exactly the quantities needed for Wigner functions, P(n) and
statistical error bars, and which can be summed across runs with different seeds.

The main process writes a checkpoint (raw_<sector>_seed<seed>.npz) every few
minutes; re-running the same command resumes from it.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np

from .config import RunConfig, keys_equal
from .model import SECTORS, build_sector_model, sector_basis_ket

class Interrupted(Exception):
    """Raised by the SIGTERM handler of run_point.py (SLURM time limit)."""


# ---------------------------------------------------------------------------
# seeds and initial states
# ---------------------------------------------------------------------------

def _root_seed(cfg: RunConfig, sector: str) -> np.random.SeedSequence:
    return np.random.SeedSequence([int(cfg.seed), SECTORS.index(sector), 2412_14271])


def initial_state_indices(cfg: RunConfig, sector: str, fock: np.ndarray,
                          jz_diag: np.ndarray, rng: np.random.Generator):
    """Return arrays (k, s): cavity sector index and spin index of every
    trajectory's initial basis state."""
    n_spin = jz_diag.size
    if cfg.init == "vacuum":
        k = np.zeros(cfg.n_traj, dtype=np.int64)                  # |0> or |1>
        s = np.full(cfg.n_traj, int(np.argmin(jz_diag)), np.int64)  # all spins down
        return k, s

    nmax = cfg.init_nmax or cfg.M
    k_allowed = np.nonzero(fock < nmax)[0]
    D = k_allowed.size * n_spin
    if cfg.sampling == "stratified":
        reps = math.ceil(cfg.n_traj / D)
        flat = np.concatenate([rng.permutation(D) for _ in range(reps)])[: cfg.n_traj]
    else:
        flat = rng.integers(0, D, size=cfg.n_traj)
    return k_allowed[flat // n_spin], flat % n_spin


def auto_batch_size(n_traj: int, workers: int) -> int:
    # ~6 tasks per worker for load balancing, at most 25 trajectories per task
    return int(max(1, min(25, n_traj // max(1, 6 * workers))))


# ---------------------------------------------------------------------------
# worker side
# ---------------------------------------------------------------------------
_W: dict = {}


def _worker_init(cfg_dict: dict, sector: str):
    import qutip as qt  # imported here so every worker builds its own solver

    cfg = RunConfig.from_dict(cfg_dict)
    model = build_sector_model(cfg.N, cfg.M, cfg.lam, cfg.kappa2, cfg.wc, cfg.wa, sector)
    options = {
        "method": cfg.method,
        "atol": cfg.atol,
        "rtol": cfg.rtol,
        "nsteps": cfg.nsteps,
        "progress_bar": False,
        "norm_t_tol": 1e-9,
    }
    solver = qt.MCSolver(model.H, model.c_ops, options=options)
    _W.clear()
    _W.update(cfg=cfg, model=model, solver=solver,
              tlist=np.linspace(0.0, cfg.t_max, cfg.n_t))


def _observables(psi_arr: np.ndarray, fock: np.ndarray, jz_diag: np.ndarray):
    prob = np.abs(psi_arr) ** 2
    return float(prob.sum(axis=1) @ fock), float(prob.sum(axis=0) @ jz_diag)


def _run_batch(task):
    """task = (batch_id, k_indices, s_indices, int_seeds)."""
    batch_id, ks, ss, seeds = task
    model, solver, tlist = _W["model"], _W["solver"], _W["tlist"]
    fock, jz_diag = model.fock, model.jz_diag
    Ms, Ns, nt = model.dim_cav, model.dim_spin, tlist.size
    a2v = model.a2_sector

    rho_c = np.zeros((Ms, Ms), dtype=complex)
    rho_s = np.zeros((Ns, Ns), dtype=complex)
    n_t = np.zeros(nt); n_t2 = np.zeros(nt)
    jz_t = np.zeros(nt); jz_t2 = np.zeros(nt)
    fin = {key: [] for key in _FIN_KEYS}
    a4v = a2v[:-1] * a2v[1:]          # <f_i| a^4 |f_{i+2}> on the sector

    t0 = time.time()
    for k, s, seed in zip(ks, ss, seeds):
        psi0 = sector_basis_ket(model, k, s)
        solver.start(psi0, tlist[0], seed=int(seed))
        n_tr = np.empty(nt); jz_tr = np.empty(nt)
        n_tr[0], jz_tr[0] = fock[k], jz_diag[s]
        psi_arr = None
        for i in range(1, nt):
            psi = solver.step(tlist[i])
            # ket dims [[Ms, Ns], [1, 1]]: cavity index is the slow index
            psi_arr = psi.full().reshape(Ms, Ns)
            n_tr[i], jz_tr[i] = _observables(psi_arr, fock, jz_diag)

        rho_c += psi_arr @ psi_arr.conj().T
        rs = psi_arr.T @ psi_arr.conj()
        rho_s += rs
        n_t += n_tr; n_t2 += n_tr ** 2
        jz_t += jz_tr; jz_t2 += jz_tr ** 2
        fin["n"].append(n_tr[-1])
        fin["jz"].append(jz_tr[-1])
        # <a^2> and <Jx> vanish identically on every trajectory (each trajectory stays in
        # an eigenspace of the Z4 generator Pi_4, which flips the sign of both), while
        # <a^4> and <Jx a^2> are Z4-invariant, i.e. equal for all four lobes. They separate
        # the vacuum basin (~0) from the lobes, their phases give the lobe angle theta
        # (4 theta, mod pi/2) and the spin-lobe locking (2 theta + pi if Jx < 0 on the lobe).
        fin["a4"].append(complex(np.sum(psi_arr[:-2].conj() * a4v[:, None] * psi_arr[2:])))
        fin["jxa2"].append(complex(np.sum(psi_arr[:-1].conj() * a2v[:, None]
                                          * (psi_arr[1:] @ model.jx_spin.T))))
        fin["init_n"].append(int(fock[k]))
        fin["init_s"].append(int(s))

    return batch_id, {
        "rho_c": rho_c, "rho_s": rho_s,
        "n_t": n_t, "n_t2": n_t2, "jz_t": jz_t, "jz_t2": jz_t2,
        "fin": {key: np.asarray(v) for key, v in fin.items()},
        "count": len(ks), "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# raw (checkpoint) files
# ---------------------------------------------------------------------------
_FIN_KEYS = ("n", "jz", "a4", "jxa2", "init_n", "init_s")
_FIN_DTYPES = {"n": float, "jz": float, "a4": complex, "jxa2": complex,
               "init_n": np.int64, "init_s": np.int64}


def raw_path(point_dir: Path, sector: str, seed: int) -> Path:
    return Path(point_dir) / f"raw_{sector}_seed{seed}.npz"


def _empty_state(cfg: RunConfig, sector: str, Ms: int, Ns: int, batch_size: int) -> dict:
    return {
        "meta": {}, "batch_size": batch_size, "batches_done": [], "n_done": 0,
        "rho_c": np.zeros((Ms, Ms), complex), "rho_s": np.zeros((Ns, Ns), complex),
        "n_t": np.zeros(cfg.n_t), "n_t2": np.zeros(cfg.n_t),
        "jz_t": np.zeros(cfg.n_t), "jz_t2": np.zeros(cfg.n_t),
        "fin": {k: np.zeros(0, dtype=_FIN_DTYPES[k]) for k in _FIN_KEYS},
        "compute_seconds": 0.0, "wall_seconds": 0.0,
    }


def load_raw(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        st = {
            "meta": json.loads(str(z["meta_json"])),
            "batches_done": [int(b) for b in z["batches_done"]],
            "n_done": int(z["n_done"]),
            "rho_c": z["rho_c_sum"], "rho_s": z["rho_s_sum"],
            "n_t": z["n_t_sum"], "n_t2": z["n_t_sqsum"],
            "jz_t": z["jz_t_sum"], "jz_t2": z["jz_t_sqsum"],
            "tlist": z["tlist"], "fock": z["fock"],
            "fin": {k: z[f"final_{k}"] for k in _FIN_KEYS},
        }
    st["batch_size"] = int(st["meta"]["batch_size"])
    st["compute_seconds"] = float(st["meta"].get("compute_seconds", 0.0))
    st["wall_seconds"] = float(st["meta"].get("wall_seconds", 0.0))
    return st


def save_raw(path: Path, st: dict, cfg: RunConfig, sector: str, fock, tlist, workers: int):
    import qutip as qt

    meta = {
        "physics_key": cfg.physics_key(),
        "config": cfg.to_dict(),
        "sector": sector, "seed": int(cfg.seed),
        "n_traj_target": int(cfg.n_traj), "n_done": int(st["n_done"]),
        "batch_size": int(st["batch_size"]),
        "compute_seconds": float(st["compute_seconds"]),
        "wall_seconds": float(st["wall_seconds"]),
        "workers": int(workers),
        "qutip_version": qt.__version__, "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    arrays = {
        "meta_json": np.array(json.dumps(meta)),
        "batches_done": np.asarray(sorted(st["batches_done"]), dtype=np.int64),
        "n_done": np.int64(st["n_done"]),
        "rho_c_sum": st["rho_c"], "rho_s_sum": st["rho_s"],
        "n_t_sum": st["n_t"], "n_t_sqsum": st["n_t2"],
        "jz_t_sum": st["jz_t"], "jz_t_sqsum": st["jz_t2"],
        "tlist": np.asarray(tlist), "fock": np.asarray(fock),
    }
    for k in _FIN_KEYS:
        arrays[f"final_{k}"] = np.asarray(st["fin"][k], dtype=_FIN_DTYPES[k])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _accumulate(st: dict, batch_id: int, res: dict):
    for key in ("rho_c", "rho_s", "n_t", "n_t2", "jz_t", "jz_t2"):
        st[key] = st[key] + res[key]
    for key in _FIN_KEYS:
        st["fin"][key] = np.concatenate([st["fin"][key], res["fin"][key].astype(_FIN_DTYPES[key])])
    st["batches_done"].append(int(batch_id))
    st["n_done"] += int(res["count"])
    st["compute_seconds"] += float(res["seconds"])


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def run_sector(cfg: RunConfig, sector: str, point_dir: Path, workers: int = 1,
               checkpoint_seconds: float = 600.0, log=print) -> Path:
    """Run (or resume) cfg.n_traj trajectories in one parity sector and write
    raw_<sector>_seed<seed>.npz into point_dir. Returns the path."""
    point_dir = Path(point_dir)
    point_dir.mkdir(parents=True, exist_ok=True)
    path = raw_path(point_dir, sector, cfg.seed)

    # small model in the main process: dimensions + initial-state list only
    model = build_sector_model(cfg.N, cfg.M, cfg.lam, cfg.kappa2, cfg.wc, cfg.wa, sector)
    Ms, Ns = model.dim_cav, model.dim_spin
    tlist = np.linspace(0.0, cfg.t_max, cfg.n_t)

    if path.exists():
        st = load_raw(path)
        if not keys_equal(st["meta"]["physics_key"], cfg.physics_key()):
            raise RuntimeError(
                f"{path} exists but was produced with different parameters:\n"
                f"  file:   {st['meta']['physics_key']}\n  now:    {cfg.physics_key()}\n"
                "Use another --outdir or --seed.")
        if st["meta"]["n_traj_target"] != cfg.n_traj:
            raise RuntimeError(
                f"{path} was started with n_traj={st['meta']['n_traj_target']} but now "
                f"n_traj={cfg.n_traj}. To ADD trajectories, run again with a new --seed; "
                "finalize merges all seeds automatically.")
        batch_size = st["batch_size"]
        log(f"[{sector}] resuming {path.name}: {st['n_done']}/{cfg.n_traj} trajectories done")
    else:
        batch_size = cfg.batch_size or auto_batch_size(cfg.n_traj, workers)
        st = _empty_state(cfg, sector, Ms, Ns, batch_size)

    root = _root_seed(cfg, sector)
    init_rng = np.random.default_rng(root.spawn(1)[0])
    ks, ss = initial_state_indices(cfg, sector, model.fock, model.jz_diag, init_rng)
    traj_seeds = root.spawn(1)[0].generate_state(cfg.n_traj, dtype=np.uint32)

    n_batches = math.ceil(cfg.n_traj / batch_size)
    done = set(st["batches_done"])
    tasks = []
    for b in range(n_batches):
        if b in done:
            continue
        sl = slice(b * batch_size, min((b + 1) * batch_size, cfg.n_traj))
        tasks.append((b, ks[sl], ss[sl], traj_seeds[sl]))

    if not tasks:
        log(f"[{sector}] {path.name} already complete ({st['n_done']} trajectories)")
        return path

    workers = max(1, min(int(workers), len(tasks)))
    log(f"[{sector}] N={cfg.N} M={cfg.M} lam={cfg.lam:g} | sector dim = {Ms} x {Ns} = {Ms * Ns}"
        f" | {cfg.n_traj - st['n_done']} trajectories in {len(tasks)} batches of {batch_size}"
        f" on {workers} worker(s)")

    t_start = time.time()
    wall0 = st["wall_seconds"]
    last_ckpt = t_start
    last_log = 0.0
    n_at_start = st["n_done"]

    def handle(batch_id, res):
        nonlocal last_ckpt, last_log
        _accumulate(st, batch_id, res)
        now = time.time()
        st["wall_seconds"] = wall0 + (now - t_start)
        n_new = st["n_done"]
        if now - last_log > 30.0 or n_new >= cfg.n_traj:   # log at most every 30 s
            last_log = now
            rate = (n_new - n_at_start) / max(now - t_start, 1e-9)       # trajectories per second
            eta = (cfg.n_traj - n_new) / max(rate, 1e-9)
            log(f"[{sector}] {n_new}/{cfg.n_traj} done | {res['seconds'] / max(res['count'], 1):.2f}"
                f" s/traj per core | elapsed {now - t_start:7.0f} s | ETA {eta:7.0f} s")
        if now - last_ckpt > checkpoint_seconds:
            save_raw(path, st, cfg, sector, model.fock, tlist, workers)
            last_ckpt = now

    try:
        if workers == 1:
            _worker_init(cfg.to_dict(), sector)
            for task in tasks:
                handle(*_run_batch(task))
        else:
            # 'spawn' is robust on clusters (no forked BLAS/OpenMP state); each
            # worker imports qutip and builds its MCSolver once.
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=workers, initializer=_worker_init,
                          initargs=(cfg.to_dict(), sector)) as pool:
                for batch_id, res in pool.imap_unordered(_run_batch, tasks, chunksize=1):
                    handle(batch_id, res)
    except BaseException:
        st["wall_seconds"] = wall0 + (time.time() - t_start)
        save_raw(path, st, cfg, sector, model.fock, tlist, workers)
        log(f"[{sector}] interrupted: checkpoint with {st['n_done']} trajectories saved to {path};"
            " run the same command again to resume")
        raise

    st["wall_seconds"] = wall0 + (time.time() - t_start)
    save_raw(path, st, cfg, sector, model.fock, tlist, workers)
    log(f"[{sector}] finished: {st['n_done']} trajectories, wall {time.time() - t_start:.0f} s -> {path}")
    return path
