import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from wigner import wigner

INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
                     "mathtext.fontset": "cm", "font.family": "serif"})
d = np.load("data_nested.npz")
alpha = np.sqrt(int(d["Nsc"]))*complex(d["alpha_t"])
lobes = [(1j)**g*alpha for g in range(4)]
xv = np.linspace(-8, 8, 281)
cols = [("coh", r"coherent state $|\alpha\rangle$"), ("catA", r"cat on one axis"), ("four", r"four-legged cat $|\pi_0\rangle$")]
fig, axs = plt.subplots(2, 3, figsize=(6.3, 4.35), gridspec_kw={"wspace": 0.08, "hspace": 0.12})
labels = iter("abcdef")
for row, key in enumerate(["init", "long"]):
    for c, (name, title) in enumerate(cols):
        ax = axs[row, c]; W = wigner(d[f"{key}_{name}"], xv, xv); vmax = np.abs(W).max()
        ax.pcolormesh(xv, xv, W, cmap="RdBu", norm=TwoSlopeNorm(0, -vmax, vmax), shading="auto", rasterized=True)
        for b in lobes:
            ax.plot(np.sqrt(2)*b.real, np.sqrt(2)*b.imag, "+", ms=5, mew=0.9, color=INK)
        ax.set_aspect("equal"); ax.set_xticks([-5, 0, 5]); ax.set_yticks([-5, 0, 5])
        if row == 0: ax.set_title(title, fontsize=8.5, color=INK, pad=4); ax.set_xticklabels([])
        else: ax.set_xlabel(r"$x$", labelpad=1)
        if c == 0: ax.set_ylabel(("initial\n" if row == 0 else "long time\n") + r"$p$", labelpad=1)
        else: ax.set_yticklabels([])
        ax.text(0.04, 0.89, f"({next(labels)})", transform=ax.transAxes, fontsize=9, color=INK,
                bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.8))
fig.savefig("fig_nested_46.pdf", bbox_inches="tight"); fig.savefig("fig_nested_46.png", dpi=200, bbox_inches="tight")
print("ok, |alpha|^2 =", abs(alpha)**2)
