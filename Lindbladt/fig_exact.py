import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from wigner import wigner

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = {2: "#86b6ef", 4: "#2a78d6", 8: "#104281"}     # one blue hue, light -> dark with size
ORANGE = "#eb6834"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
                     "mathtext.fontset": "cm", "font.family": "serif"})

d = np.load("data.npz")
r = d["ratios"]
w, k1, k2t, Kt = 1.0, 1.0, 1.0, 0.5
lamc = float(d["lamc"])

def alpha_mf(lam, Nsc):
    k2, K = k2t/Nsc, Kt/Nsc
    A2 = K**2 + k2**2; B = 2*(w*K + k1*k2/2); C = w**2 + k1**2/4 - 4*lam**2
    n = (-B + np.sqrt(B**2 - 4*A2*C))/(2*A2)
    e2 = (1j*(k1/2 + k2*n) - (w + K*n))/(2*lam)
    return np.sqrt(n)*np.exp(-0.5j*np.angle(e2))

fig = plt.figure(figsize=(6.3, 2.45))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.42)
ax = fig.add_subplot(gs[0])
for N in (2, 4, 8):
    ax.semilogy(r, d[f"g1_N{N}"], color=SEQ[N], lw=2, label=rf"$\mathcal{{N}}={N}$")
    ax.semilogy(r, d[f"g0_N{N}"], color=SEQ[N], lw=1.2, ls=(0, (3, 2)))
ax.axvline(1, color=GRID, lw=0.8, zorder=0)
ax.text(1.36, 3.3, r"$E^0$ (dashed)", color=INK, fontsize=8)
ax.text(1.5, 0.47, r"$E^1$ (solid)", color=INK, fontsize=8)
ax.set_xlabel(r"$\lambda/\lambda_c$"); ax.set_ylabel(r"$-\mathrm{Re}\,\lambda_1$ per sector")
ax.set_xlim(r[0], r[-1]); ax.set_ylim(8e-4, 6)
ax.legend(frameon=False, fontsize=7.5, loc="lower left", handlelength=1.6)
ax.text(0.03, 0.93, "(a)", transform=ax.transAxes, fontsize=9, color=INK)
for s in ["top", "right"]: ax.spines[s].set_visible(False)

xv = np.linspace(-6.5, 6.5, 261)
a0 = alpha_mf(float(d["r"])*lamc, int(d["Nsc"]))
for j, (key, lab, title) in enumerate([("rho0", "(b)", r"$W[\hat\rho_{\rm ss}]$, $E^0$"),
                                        ("rho1", "(c)", r"$W[\hat\rho_1]$, $E^1$")]):
    rho = d[key]
    W = wigner(rho, xv, xv)
    if key == "rho1":   # orient so that the +alpha lobe is positive
        ix = np.argmin(abs(xv - np.sqrt(2)*a0.real)); iy = np.argmin(abs(xv - np.sqrt(2)*a0.imag))
        if W[iy, ix] < 0: W = -W
    axw = fig.add_subplot(gs[j+1])
    vmax = np.abs(W).max()
    im = axw.pcolormesh(xv, xv, W, cmap="RdBu", norm=TwoSlopeNorm(0, -vmax, vmax), shading="auto", rasterized=True)
    for s in (+1, -1):
        axw.plot(np.sqrt(2)*(s*a0).real, np.sqrt(2)*(s*a0).imag, "+", ms=7, mew=1.2, color=INK)
    axw.set_aspect("equal"); axw.set_xlabel(r"$x$"); axw.set_ylabel(r"$p$", labelpad=1)
    axw.set_xticks([-5, 0, 5]); axw.set_yticks([-5, 0, 5])
    axw.text(0.04, 0.9, lab, transform=axw.transAxes, fontsize=9, color=INK)
    axw.set_title(title, fontsize=8.5, color=INK, pad=4)
fig.savefig("fig_kerr_exact.pdf", bbox_inches="tight"); fig.savefig("fig_kerr_exact.png", dpi=200, bbox_inches="tight")
print("mean-field alpha at N=8, 2 lam_c:", a0, "|alpha|^2 =", abs(a0)**2)
print("Tr rho1 =", np.trace(d['rho1']).real, " lam0, lam1 =", d['lam0'], d['lam1'])
