#!/usr/bin/env python
"""
Sweep-level figures from finalized results.

Wigner grid (rows = selected lambdas, columns = even/odd), like the 3x2 figure
of the original notebook:
    python plot_sweep.py grid results/highres_v1/N30_k0.05_wc1_wa1_t70_Tinf \
        --lam 0.51 0.53 0.6 --out figures/wigner_grid_N30.pdf --xmax 20

<n>/N versus lambda for several system sizes and both sectors:
    python plot_sweep.py curve results/highres_v1/N15_*_Tinf results/highres_v1/N30_*_Tinf \
        --out figures/n_vs_lambda.pdf --lam-c 0.512
"""
import argparse
import json
import sys
from pathlib import Path

from tpdt.analysis import find_point_dirs
from tpdt.plotting import plot_n_vs_lambda, plot_wigner_grid


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grid", help="Wigner functions, rows = lambda, cols = sector")
    g.add_argument("system_dir")
    g.add_argument("--lam", type=float, nargs="+", required=True)
    g.add_argument("--out", default="figures/wigner_grid.pdf")
    g.add_argument("--xmax", type=float, default=None,
                   help="common phase-space window (recomputes W); default: stored per point")
    g.add_argument("--grid", type=int, default=201)
    g.add_argument("--title", default=None)

    c = sub.add_parser("curve", help="<n>/N vs lambda")
    c.add_argument("system_dirs", nargs="+")
    c.add_argument("--out", default="figures/n_vs_lambda.pdf")
    c.add_argument("--lam-c", type=float, default=None, help="draw a vertical line at lambda_c")

    args = p.parse_args(argv)

    if args.cmd == "grid":
        by_lam, M_of = {}, {}
        for d in find_point_dirs(Path(args.system_dir)):
            f = d / "summary.json"
            if f.exists():
                key = json.loads(f.read_text())["physics_key"]
                lam = round(key["lam"], 6)
                if lam not in by_lam or key["M"] > M_of[lam]:   # largest cutoff wins
                    by_lam[lam], M_of[lam] = d, key["M"]
        chosen = []
        for lam in args.lam:
            best = min(by_lam, key=lambda x: abs(x - lam)) if by_lam else None
            if best is None or abs(best - lam) > 1e-6:
                sys.exit(f"no finalized point with lambda = {lam} in {args.system_dir} "
                         f"(available: {sorted(by_lam)})")
            chosen.append(by_lam[best])
        title = args.title or f"QT Wigner functions: {Path(args.system_dir).name}"
        plot_wigner_grid(chosen, Path(args.out), x_max=args.xmax, n_grid=args.grid, title=title)
    else:
        plot_n_vs_lambda([Path(s) for s in args.system_dirs], Path(args.out), lam_c=args.lam_c)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
