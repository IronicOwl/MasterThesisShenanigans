import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from wigner import wigner, coherent_dm
from scipy.special import gammaln

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = {2: "#86b6ef", 4: "#2a78d6", 8: "#104281"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
                     "mathtext.fontset": "cm", "font.family": "serif"})
d = np.load("data_strong.npz"); r = d["ratios"]
w, k1, k2t, Kt = 1.0, 0.0, 1.0, 0.5; lamc = float(d["lamc"])
Nsc, D = int(d["Nsc"]), int(d["D"])

def alpha_mf(lam, N):
    k2, K = k2t/N, Kt/N
    A2 = K**2 + k2**2; B = 2*(w*K + k1*k2/2); C = w**2 + k1**2/4 - 4*lam**2
    n = (-B + np.sqrt(B**2 - 4*A2*C))/(2*A2)
    e2 = (1j*(k1/2 + k2*n) - (w + K*n))/(2*lam)
    return np.sqrt(n)*np.exp(-0.5j*np.angle(e2))

a0 = alpha_mf(float(d["r"])*lamc, Nsc)
# --- reconstruct rho_SB = rho_00 + rho_01 + rho_10 + rho_11 from the exact zero modes
r00, r11, X = d["rho00"], d["rho11"], d["X01"]
X = X/(2*np.linalg.svd(X, compute_uv=False).sum())        # trace norm 1/2
coh = coherent_dm(a0, D)
ov = np.trace(coh @ X)                                     # <alpha|X|alpha>
X = X*np.exp(-1j*np.angle(ov))                              # gauge: <alpha|X|alpha> real positive
rho_sb = 0.5*r00 + 0.5*r11 + X + X.conj().T
ev = np.linalg.eigvalsh(0.5*(rho_sb + rho_sb.conj().T))
F = np.real(np.trace(coh @ rho_sb))
print(f"reconstructed: trace={np.trace(rho_sb).real:.4f} min eig={ev.min():.2e} purity={np.real(np.trace(rho_sb@rho_sb)):.3f} <alpha|rho|alpha>={F:.3f}")

fig, axs = plt.subplots(1, 4, figsize=(6.5, 1.95), gridspec_kw={"wspace": 0.1})
xv = np.linspace(-6.5, 6.5, 261)
panels = [(axs[0], r00, "(a)", r"$W[2\hat\rho_{00}]$"), (axs[1], r11, "(b)", r"$W[2\hat\rho_{11}]$"),
          (axs[2], 0.5*(r00 + r11), "(c)", r"$W[\hat\rho_{00}+\hat\rho_{11}]$"),
          (axs[3], rho_sb, "(d)", r"$W[\hat\rho_{\mathrm{SB}}]$")]
for axw, rho, lab, title in panels:
    W = wigner(rho, xv, xv); vmax = np.abs(W).max()
    axw.pcolormesh(xv, xv, W, cmap="RdBu", norm=TwoSlopeNorm(0, -vmax, vmax), shading="auto", rasterized=True)
    for s in (+1, -1):
        axw.plot(np.sqrt(2)*(s*a0).real, np.sqrt(2)*(s*a0).imag, "+", ms=6, mew=1.0, color=INK)
    axw.set_aspect("equal"); axw.set_xticks([-5, 0, 5]); axw.set_yticks([-5, 0, 5])
    axw.set_xlabel(r"$x$", labelpad=1)
    if axw is axs[0]: axw.set_ylabel(r"$p$", labelpad=1)
    else: axw.set_yticklabels([])
    axw.set_title(title, fontsize=8.5, color=INK, pad=4)
    axw.text(0.04, 0.88, lab, transform=axw.transAxes, fontsize=9, color=INK)
fig.savefig("fig_kerr_strong.pdf", bbox_inches="tight"); fig.savefig("fig_kerr_strong.png", dpi=200, bbox_inches="tight")
print("alpha_mf =", a0, " |alpha|^2 =", abs(a0)**2, " lam01 =", d["lam01"])
