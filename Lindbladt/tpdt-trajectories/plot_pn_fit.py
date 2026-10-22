#!/usr/bin/env python
"""
Presentation figure: full photon-number distribution P(n) of one parameter point
(vacuum part included) with the Gaussian fit of the superradiant lobe on top,
even and odd sector side by side. Uses the same lobe fit as fit_lobes.py.

    python plot_pn_fit.py results/thesis/N15_k0.05_wc1_wa1_t70_Tinf-n75/lam0.5400_M200 \
        --out figures/pn_fit_lam054            [--log] [--xmax 220] [--transparent]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fit_lobes import analyse, gauss

SECTORS = {"even": (0, "#2a78d6", r"even sector ($\Pi_2=+1$)"),
           "odd": (1, "#eb6834", r"odd sector ($\Pi_2=-1$)")}
INK, INK2 = "#0b0b0b", "#52514e"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("point_dir")
    ap.add_argument("--out", default=None, help="default: figures/pn_fit_lam<lambda>")
    ap.add_argument("--log", action="store_true", help="logarithmic P(n) axis")
    ap.add_argument("--xmax", type=float, default=None, help="default: lobe centre + 5 widths")
    ap.add_argument("--transparent", action="store_true", help="transparent background (slides)")
    args = ap.parse_args(argv)

    d = Path(args.point_dir)
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "legend.fontsize": 11.5})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    xmax, lam, N, ymax = 0.0, None, None, 0.0
    for ax, (sec, (parity, col, title)) in zip(axes, SECTORS.items()):
        f = d / f"summary_{sec}.npz"
        if not f.exists():
            ax.set_visible(False)
            continue
        with np.load(f) as z:
            P, N, lam, final_n = z["P_n"], int(z["N"]), float(z["lam"]), z["final_n"]
        n = np.arange(parity, P.size, 2)
        p = np.clip(P[n], 0.0, None)
        r = analyse(P, parity, N, final_n)
        ax.bar(n, p, width=1.6, color=col, alpha=0.85, linewidth=0,
               label=f"trajectories, {sec} ({final_n.size})")
        ymax = max(ymax, p.max())
        if r["method"] == "P(n)":
            x = np.linspace(max(r["n_split"], r["mu"] - 5 * r["sigma"], 0), r["mu"] + 5 * r["sigma"], 800)
            g = gauss(x, *r["_fit"])
            ax.plot(x, g, color=INK, lw=2.2, label="Gaussian fit to the lobe")
            ax.text(0.5, 0.97, rf"$\mu/N = {r['mu'] / N:.2f}$,   $\sigma/N = {r['sigma'] / N:.2f}$,   "
                    rf"lobe weight ${r['weight']:.2f}$", transform=ax.transAxes, ha="center", va="top",
                    fontsize=12, color=INK2)
            xmax = max(xmax, r["mu"] + 5 * r["sigma"])
        else:
            print(f"{sec}: no Gaussian fit at this point ({r.get('reason', r['method'])})")
            xmax = max(xmax, n[np.nonzero(np.cumsum(p) > 0.999 * p.sum())[0][0]])
        ax.set_title(title, fontsize=13, color=INK)
        ax.set_xlabel(r"photon number $n$")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    xmax = args.xmax or xmax
    for ax in axes:
        ax.set_xlim(-2, xmax)
        if args.log:
            ax.set_yscale("log")
            ax.set_ylim(1e-6, 40 * ymax)
        else:
            ax.set_ylim(0, 1.22 * ymax)
    axes[0].set_ylabel(r"$P(n)$")
    fig.suptitle(rf"$\lambda = {lam:g}$,  $N = {N}$", fontsize=14, color=INK)
    fig.tight_layout()
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    order = sorted(range(len(labels)), key=lambda i: labels[i].startswith("Gaussian"))
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=len(labels), frameon=False)
    out = Path(args.out or f"figures/pn_fit_lam{int(round(lam * 1000)):04d}")
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 220})):
        fig.savefig(out.with_suffix(ext), bbox_inches="tight", transparent=args.transparent, **kw)
    print(f"wrote {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
