"""Gap vs coupling for every N in gap_results/ (one figure, nothing else).
Usage:  python plot_gap.py        -> sector E_++_1
        python plot_gap.py --     -> sector E_--_1
"""
import sys, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sector = sys.argv[1] if len(sys.argv) > 1 else "++"
runs = sorted((json.load(open(f)) for f in glob.glob(f"gap_results/gapscan_{sector}_N*.json")),
              key=lambda d: d["summary"]["N"])

fig, ax = plt.subplots(figsize=(7, 4.5))
for d, c in zip(runs, plt.cm.viridis(np.linspace(0.05, 0.9, len(runs)))):
    ev = sorted(d["evaluations"], key=lambda r: r["lam"])
    s = d["summary"]
    ax.semilogy([r["lam"] for r in ev], [r["gap_sw"] for r in ev], "o-", ms=3.5, color=c, label=f"N = {s['N']}")
    ax.plot(s["lam_star"], s["gap_min"], "*", ms=12, color=c, mec="k")      # fitted minimum
ax.axvline(runs[0]["summary"]["lam_c_mf"], color="gray", ls="--", lw=1, label=r"mean-field $\lambda_c$")
ax.set_xlabel(r"coupling $\lambda$")
ax.set_ylabel(r"dissipative gap  $-\mathrm{Re}\,\lambda_1$")
ax.set_title(f"trace-1 sector E_{sector}_1")
ax.legend(fontsize=8)
fig.tight_layout()
out = f"gap_results/gap_vs_coupling_{sector}.png"
fig.savefig(out, dpi=200)
print("saved", out)