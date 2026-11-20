#!/usr/bin/env python3
"""
replot_trace_gap.py -- redraw Figure A.4 of the thesis (gap closure in the trace-carrying sector
of the two-photon Dicke model) from the results of gap_scan.py, in the notation of the thesis.

Reads <outdir>/gapscan_<sector>_N*.json written by "gap_scan.py compute" and draws panel (a) of
"gap_scan.py analyze" on its own: for every N the gap of the vacuum <-> superradiant switching
mode against lambda, its minimum as a star (the parabola fit of gap_scan.py), and the mean-field
lambda_c as a dashed line.

Labels as in the thesis:
  * sector E^0_00 (E_++_1 in gap_scan.py; E^0_11 with --sector --), labelled by the photon
    parity for every N, as in Figure A.3;
  * y axis "dissipative gap -Re lambda_1", x axis "coupling lambda", legend N = ..., mean-field
    lambda_c.
gap_scan.py plots the switching mode. Where another mode is slower, that mode is lambda_1, so
the script reports how often this happens; --show-all adds the slowest mode of any kind as
dotted lines, as in the original panel (a).

    python replot_trace_gap.py                      # results in gap_results/, sector ++
    python replot_trace_gap.py my_results --no-title
    python replot_trace_gap.py --show-all

Writes <outdir>/gap_trace_thesis_<sector>.{pdf,png}; gap_scan.py is not imported or changed.
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = {"++": r"$E^{0}_{00}$", "--": r"$E^{0}_{11}$"}


def load(outdir, sector):
    files = sorted(glob.glob(os.path.join(outdir, f"gapscan_{sector}_N*.json")))
    if not files:
        raise SystemExit(f"no gapscan_{sector}_N*.json in {outdir}")
    return sorted((json.load(open(f)) for f in files), key=lambda d: d["summary"]["N"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", nargs="?", default="gap_results",
                    help="output folder of gap_scan.py (default gap_results)")
    ap.add_argument("--sector", choices=("++", "--"), default="++")
    ap.add_argument("--no-title", action="store_true", help="no 'Sector E^0_00' above the plot")
    ap.add_argument("--show-all", action="store_true",
                    help="also the slowest mode of any kind (dotted), as in gap_scan.py")
    ap.add_argument("--out", default=None, help="file name without extension "
                    "(default <outdir>/gap_trace_thesis_<sector>)")
    args = ap.parse_args()

    data = load(args.outdir, args.sector)
    lam_c = data[0]["summary"]["lam_c_mf"]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(data)))
    other_slower = []
    for d, col in zip(data, cols):
        ev = sorted(d["evaluations"], key=lambda r: r["lam"])
        lam = np.array([r["lam"] for r in ev])
        gsw = np.array([r["gap_sw"] for r in ev], float)
        gall = np.array([r["gap_all"] for r in ev], float)
        N = d["summary"]["N"]
        ax.semilogy(lam, gsw, "o-", ms=3.5, color=col, label=rf"$N = {N}$")
        if args.show_all:
            ax.semilogy(lam, gall, ":", color=col, lw=1)
        ax.plot(d["summary"]["lam_star"], d["summary"]["gap_min"], "*", color=col, ms=12, mec="k")
        ok = np.isfinite(gsw) & np.isfinite(gall)
        n_other = int(np.sum(gsw[ok] > gall[ok] * (1 + 1e-6)))
        if n_other:
            other_slower.append(f"N={N}: {n_other} of {ok.sum()} points")
    ax.axvline(lam_c, color="gray", ls="--", lw=1, label=r"mean-field $\lambda_c$")
    ax.set_xlabel(r"coupling $\lambda$")
    ax.set_ylabel(r"dissipative gap $-\mathrm{Re}\,\lambda_1$")
    if not args.no_title:
        ax.set_title("Sector " + LABEL[args.sector])
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = args.out or os.path.join(args.outdir, f"gap_trace_thesis_{args.sector}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
        print("wrote", f"{out}.{ext}")
    if other_slower:
        print("note: at these points another mode is slower than the switching mode, so there "
              "-Re lambda_1 is below the plotted curve (see --show-all): " + "; ".join(other_slower))
    else:
        print("the switching mode is the slowest mode at every point, i.e. the curves are -Re lambda_1")


if __name__ == "__main__":
    main()