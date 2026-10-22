"""
plot_ns_test.py
===============

Stand-alone version of panel (a) of fig2: the noiseless-subsystem test of
qubit A versus N, made from the .npz files that ns_compass.py saved.  Sized
for a thesis page (15 cm wide) and written as PDF (vector) and PNG.

Run it from the folder that contains ns_results/, e.g.

    python plot_ns_test.py --lam 0.6 --kappa2 0.05                 # thesis run
    python plot_ns_test.py --lam 0.6 --kappa2 0.05 --wa 0.25       # low-leak run
    python plot_ns_test.py --lam 0.6 --kappa2 0.05 --nmin 3 --series dom_eo dom_oo full_eo inc eps

Output: ns_results/ns_test_qubitA_<parameters>.pdf and .png (or --out NAME).
Needs only numpy and matplotlib (no QuTiP).
"""

import os
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

OUTDIR = "ns_results"
WC_DEFAULT, WA_DEFAULT = 1.0, 1.0      # same defaults as ns_compass.py

# Curves that can be drawn: short name -> (key in the .npz file, legend label, style).
# Same curves as panel (a) of fig2 of ns_compass.py.
SERIES = {
    "full_eo":  ("cf_A_full_Dt_ee_eo",  r"full metastable state: $D_t(z_{ee},z_{eo})$", dict(marker="s", ls="-")),
    "full_oo":  ("cf_A_full_Dt_ee_oo",  r"full metastable state: $D_t(z_{ee},z_{oo})$", dict(marker="o", ls="-")),
    "dom_eo":   ("cf_A_dom_Dt_ee_eo",   r"dominant aux. states: $D_t(z_{ee},z_{eo})$",  dict(marker="^", ls="-")),
    "dom_oo":   ("cf_A_dom_Dt_ee_oo",   r"dominant aux. states: $D_t(z_{ee},z_{oo})$",  dict(marker="D", ls="-")),
    "basin_eo": ("cf_A_basin_Dt_ee_eo", r"committed basin: $D_t(z_{ee},z_{eo})$",       dict(marker="*", ls="-")),
    "inc":      ("cf_A_w_inc",          "incoherent weight",                            dict(marker="v", ls="--")),
    "eps":      ("cf_eps",              r"$\varepsilon$ = slow rate / next rate",       dict(marker="P", ls="--")),
    "old":      ("ns_A_main_Dt_ee_eo",  r"Fock cut (first method): $D_t(z_{ee},z_{eo})$", dict(marker="x", ls=":")),
}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def run_tag(lam, kappa2, wc=WC_DEFAULT, wa=WA_DEFAULT):
    """Parameter part of the result file names (same rule as ns_compass.py)."""
    tag = f"lam{lam:.3f}_k2{kappa2:.3f}"
    if wc != WC_DEFAULT:
        tag += f"_wc{wc:.3f}"
    if wa != WA_DEFAULT:
        tag += f"_wa{wa:.3f}"
    return tag


def load_runs(tag, nmin, nmax):
    """All saved results with this parameter tag, sorted by N, restricted to nmin..nmax."""
    files = glob.glob(os.path.join(OUTDIR, f"ns_N*_{tag}.npz"))
    data = [dict(np.load(f, allow_pickle=True)) for f in files]
    data = [d for d in data if nmin <= int(d["N"]) <= nmax]
    return sorted(data, key=lambda d: int(d["N"]))


def main():
    """Command line: choose the run (lam, kappa2, wc, wa), the curves and the N range."""
    p = argparse.ArgumentParser(description="Noiseless-subsystem test of qubit A versus N, as a single figure.")
    p.add_argument("--lam", type=float, required=True)
    p.add_argument("--kappa2", type=float, required=True)
    p.add_argument("--wc", type=float, default=WC_DEFAULT)
    p.add_argument("--wa", type=float, default=WA_DEFAULT)
    p.add_argument("--series", nargs="+", default=list(SERIES), choices=list(SERIES),
                   help="curves to draw, in this order (default: all, as in fig2)")
    p.add_argument("--nmin", type=int, default=2, help="smallest N to show")
    p.add_argument("--nmax", type=int, default=99, help="largest N to show")
    p.add_argument("--title", default=None, help="title above the plot ('' for none)")
    p.add_argument("--width-cm", type=float, default=15.0, help="figure width (cm)")
    p.add_argument("--height-cm", type=float, default=11.0, help="figure height incl. legend (cm)")
    p.add_argument("--out", default=None, help="output name without extension")
    args = p.parse_args()

    tag = run_tag(args.lam, args.kappa2, args.wc, args.wa)
    data = load_runs(tag, args.nmin, args.nmax)
    if not data:
        raise SystemExit(f"no files {OUTDIR}/ns_N*_{tag}.npz found (run from the folder that contains {OUTDIR}/)")
    Ns = [int(d["N"]) for d in data]

    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "font.size": 10, "legend.fontsize": 8.5, "axes.titlesize": 10,
                         "lines.linewidth": 1.4, "lines.markersize": 5})
    fig, ax = plt.subplots(figsize=(args.width_cm / 2.54, args.height_cm / 2.54))
    drawn = 0
    for name in args.series:
        key, label, style = SERIES[name]
        y = np.array([abs(float(d[key])) if key in d else np.nan for d in data])
        if np.all(np.isnan(y)):
            print(f"  skipping '{name}': not in these result files (older script version?)")
            continue
        col = "0.45" if name == "old" else COLORS[list(SERIES).index(name)]
        ax.semilogy(Ns, y, color=col, label=label, **style)
        drawn += 1
    if drawn == 0:
        raise SystemExit("none of the requested curves is in the result files")

    ax.set_xlabel(r"$N$")
    ax.set_ylabel("trace distance / weight / ratio")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, which="major", color="0.88", lw=0.6)
    ax.set_axisbelow(True)
    if args.title is None:
        title = rf"qubit A, $\lambda={args.lam:g}$, $\kappa_2={args.kappa2:g}$"
        if args.wc != WC_DEFAULT:
            title += rf", $\omega_c={args.wc:g}$"
        if args.wa != WA_DEFAULT:
            title += rf", $\omega_a={args.wa:g}$"
    else:
        title = args.title
    if title:
        ax.set_title(title)
    # legend below the axes, so it never covers data points
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False,
              handlelength=2.6, columnspacing=1.2)
    fig.tight_layout()

    out = args.out or os.path.join(OUTDIR, f"ns_test_qubitA_{tag}")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {out}.pdf and {out}.png  (N = {Ns})")


if __name__ == "__main__":
    main()