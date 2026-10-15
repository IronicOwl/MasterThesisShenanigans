#!/usr/bin/env python
"""
Merge all raw trajectory files (all seeds / chunks) below a results directory,
recompute summaries, Wigner functions and plots.

    python finalize.py results/                     # every point that changed
    python finalize.py results/N15_M160_* --force   # recompute everything there
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
from pathlib import Path

from tpdt.analysis import find_point_dirs, finalize_point


def is_stale(point_dir: Path) -> bool:
    summ = point_dir / "summary.json"
    if not summ.exists() or not (point_dir / "wigner_pn.png").exists():
        return True
    t_summary = summ.stat().st_mtime
    return any(f.stat().st_mtime > t_summary for f in point_dir.glob("raw_*_seed*.npz"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="results directories (searched recursively)")
    p.add_argument("--force", action="store_true", help="recompute even if up to date")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--wigner-grid", type=int, default=201)
    p.add_argument("--wigner-xmax", type=float, default=None)
    args = p.parse_args(argv)

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    dirs = sorted({d for path in args.paths for d in find_point_dirs(Path(path))})
    todo = [d for d in dirs if args.force or is_stale(d)]
    log(f"{len(dirs)} point directories found, {len(todo)} to finalize")
    n_fail = 0
    for d in todo:
        try:
            finalize_point(d, n_grid=args.wigner_grid, x_max=args.wigner_xmax,
                           make_plots=not args.no_plots, log=log)
        except Exception as exc:  # keep going, report at the end
            n_fail += 1
            log(f"ERROR in {d}: {exc!r}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
