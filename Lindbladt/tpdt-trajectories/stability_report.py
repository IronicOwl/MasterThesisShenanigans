import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tpdt.analysis import load_summary

OUT = Path("results/stability")
pts = defaultdict(list)                      # lambda -> [(M, point_dir, report)]
for js in sorted(OUT.glob("*/lam*_M*/summary.json")):
    rep = json.loads(js.read_text())
    k = rep["physics_key"]
    pts[round(k["lam"], 4)].append((k["M"], js.parent, rep))

lams = sorted(pts)
if not lams:
    raise SystemExit(f"no finished points under {OUT}")
fig, axes = plt.subplots(len(lams), 2, figsize=(11, 3.6 * len(lams)), squeeze=False)
for row, lam in enumerate(lams):
    runs = sorted(pts[lam], key=lambda x: x[0])
    N = runs[0][2]["physics_key"]["N"]
    print(f"\nlambda = {lam}, N = {N}, initial states uniform below n = {runs[0][2]['physics_key']['init_nmax']}")
    print(f"{'M':>5} | {'even <n>/N':>16} | {'odd <n>/N':>16} | P(n >= 0.9 M)")
    for M, d, rep in runs:
        se = rep["sectors"]
        cells = [f"{se[s]['n_over_N']:7.3f} +- {se[s]['n_over_N_se']:.3f}" if s in se else f"{'-':>16}"
                 for s in ("even", "odd")]
        tail = max(se[s]["tail_weight"] for s in se)
        print(f"{M:5d} | {cells[0]:>16} | {cells[1]:>16} | {tail:.1e}")
    if len(runs) >= 2:
        (M1, _, r1), (M2, _, r2) = runs[-2], runs[-1]
        for s in ("even", "odd"):
            if s in r1["sectors"] and s in r2["sectors"]:
                a, b = r1["sectors"][s], r2["sectors"][s]
                z = (b["n_over_N"] - a["n_over_N"]) / np.hypot(a["n_over_N_se"], b["n_over_N_se"])
                print(f"  {s}: M {M1} -> {M2}: change of <n>/N = {z:+.1f} standard errors"
                      f" ({'consistent with no change' if abs(z) < 3 else 'SIGNIFICANT change'})")
    colors = plt.cm.Blues(np.linspace(0.4, 0.95, len(runs)))
    for col, sec in enumerate(("even", "odd")):
        ax = axes[row, col]
        for c, (M, d, rep) in zip(colors, runs):
            try:
                P = load_summary(d, sec)["P_n"]
            except FileNotFoundError:
                continue
            n = np.arange(P.size)
            m = (n % 2 == (0 if sec == "even" else 1)) & (P > 0)
            ax.semilogy(n[m], P[m], color=c, lw=1.8, label=f"M = {M}")
            ax.axvline(M, color=c, ls=":", lw=1.2)
        ax.set_ylim(1e-10, 1)
        ax.set_xlabel("photon number n")
        ax.set_ylabel("P(n)")
        ax.set_title(f"lambda = {lam}, {sec} sector (dotted: cutoff M)", fontsize=10)
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(frameon=False, fontsize=9)
fig.suptitle(f"Fock-cutoff convergence, N = {N}, same initial ensemble for all M")
fig.tight_layout()
fig.savefig(OUT / "stability.png", dpi=150)
fig.savefig(OUT / "stability.pdf")
print(f"\nfigure: {OUT / 'stability.png'}")
