#!/usr/bin/env python
"""
Run one parameter point of the two-photon Dicke model (two-photon loss only):
quantum trajectories in the even and/or odd photon-parity sector, then merge,
compute Wigner functions + P(n) and plot.

Example (one node, 32 cores):
    python run_point.py --N 15 --M 160 --lam 0.53 --n-traj 2000 --workers 32

Re-running the same command resumes from the checkpoint. Running again with a
different --seed ADDS trajectories; finalize merges all seeds of a point.
"""
import os

# one BLAS/OpenMP thread per worker process; must be set before numpy is imported
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import signal
import sys
import time
from pathlib import Path

from tpdt.config import INIT_CHOICES, SAMPLING_CHOICES, RunConfig, auto_cutoff
from tpdt.model import SECTORS


def default_workers() -> int:
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        if os.environ.get(var):
            return int(os.environ[var])
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("model")
    g.add_argument("--N", type=int, required=True, help="number of spins")
    g.add_argument("--M", required=True,
                   help="Fock cutoff (photon numbers 0..M-1), or 'auto' (estimate from N, lambda, kappa2)")
    g.add_argument("--lam", type=float, required=True, help="coupling lambda")
    g.add_argument("--kappa2", type=float, default=0.05)
    g.add_argument("--wc", type=float, default=1.0)
    g.add_argument("--wa", type=float, default=1.0)
    g = p.add_argument_group("evolution / ensemble")
    g.add_argument("--t-max", type=float, default=70.0)
    g.add_argument("--n-t", type=int, default=141, help="output time points for <n>(t), <Jz>(t)")
    g.add_argument("--n-traj", type=int, default=1000, help="trajectories per sector for this seed")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--sectors", nargs="+", choices=SECTORS, default=list(SECTORS))
    g.add_argument("--init", choices=INIT_CHOICES, default="infinite_temperature")
    g.add_argument("--init-nmax", type=int, default=None,
                   help="sample initial Fock states only below this photon number (default: M)")
    g.add_argument("--sampling", choices=SAMPLING_CHOICES, default="stratified")
    g = p.add_argument_group("integrator")
    g.add_argument("--method", default="vern7", help="qutip ODE method (vern7, adams, bdf, ...)")
    g.add_argument("--atol", type=float, default=1e-8)
    g.add_argument("--rtol", type=float, default=1e-6)
    g.add_argument("--nsteps", type=int, default=1_000_000)
    g = p.add_argument_group("execution / output")
    g.add_argument("--workers", type=int, default=None, help="processes (default: SLURM_CPUS_PER_TASK)")
    g.add_argument("--batch-size", type=int, default=None)
    g.add_argument("--checkpoint-min", type=float, default=10.0, help="checkpoint interval in minutes")
    g.add_argument("--outdir", default="results")
    g.add_argument("--no-finalize", action="store_true",
                   help="only run trajectories (use when several jobs work on the same point)")
    g.add_argument("--wigner-grid", type=int, default=201)
    g.add_argument("--wigner-xmax", type=float, default=None, help="default: automatic from P(n)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if str(args.M).lower() == "auto":
        M = auto_cutoff(args.N, args.lam, args.kappa2, args.wc)
    else:
        M = int(args.M)
    cfg = RunConfig(
        N=args.N, M=M, lam=args.lam, kappa2=args.kappa2, wc=args.wc, wa=args.wa,
        t_max=args.t_max, n_t=args.n_t, n_traj=args.n_traj, seed=args.seed,
        init=args.init, init_nmax=args.init_nmax, sampling=args.sampling,
        method=args.method, atol=args.atol, rtol=args.rtol, nsteps=args.nsteps,
        batch_size=args.batch_size,
    )
    workers = args.workers or default_workers()
    point_dir = cfg.point_dir(args.outdir)

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    import qutip
    log(f"qutip {qutip.__version__} | point dir {point_dir} | {workers} workers")

    from tpdt.trajectories import Interrupted, run_sector

    def on_sigterm(signum, frame):   # SLURM time limit / scancel: save, then stop
        raise Interrupted()

    signal.signal(signal.SIGTERM, on_sigterm)
    try:
        for sector in args.sectors:
            run_sector(cfg, sector, point_dir, workers=workers,
                       checkpoint_seconds=60.0 * args.checkpoint_min, log=log)
    except (Interrupted, KeyboardInterrupt):
        log("stopped before completion; finalize skipped")
        return 143

    if not args.no_finalize:
        from tpdt.analysis import finalize_point
        finalize_point(point_dir, n_grid=args.wigner_grid, x_max=args.wigner_xmax, log=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
