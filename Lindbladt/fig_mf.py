import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
                     "mathtext.fontset": "cm", "font.family": "serif"})

w, k1, k2 = 1.0, 1.0, 1.0           # scaled: kappa_2 = k2/N, K = Kt/N
lamc = 0.5*np.sqrt(w**2 + k1**2/4)

def ntilde(lam, K):
    A2 = K**2 + k2**2; B = 2*(w*K + k1*k2/2); C = w**2 + k1**2/4 - 4*lam**2
    disc = B**2 - 4*A2*C
    n = (-B + np.sqrt(np.maximum(disc, 0)))/(2*A2)
    return np.where(C < 0, n, np.nan)

def alpha_t(lam, K):
    n = ntilde(lam, K)
    e2 = (1j*(k1/2 + k2*n) - (w + K*n))/(2*lam)
    return np.sqrt(n)*np.exp(-0.5j*np.angle(e2))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.7), gridspec_kw={"width_ratios": [1.25, 1]})

r = np.linspace(0.0, 2.0, 400); lam = r*lamc
ax1.plot(r[r <= 1], 0*r[r <= 1], color=INK, lw=1.6)
ax1.plot(r[r >= 1], 0*r[r >= 1], color=INK, lw=1.2, ls=(0, (3, 2)))
for K, col, lab in [(0.0, BLUE, r"$\tilde K=0$"), (0.5, ORANGE, r"$\tilde K=0.5$")]:
    n = ntilde(lam, K)
    ax1.plot(r, n, color=col, lw=2, label=lab)
    i = -1
    ax1.text(r[i]+0.03, n[i], lab, color=INK, va="center", fontsize=8)
ax1.axvline(1, color=GRID, lw=0.8, zorder=0)
ax1.set_xlim(0, 2.35); ax1.set_ylim(-0.08, 1.75)
ax1.set_xlabel(r"$\lambda/\lambda_c$"); ax1.set_ylabel(r"$|\alpha|^2/\mathcal{N}$")
ax1.text(0.04, 1.62, "(a)", fontsize=9, color=INK)
ax1.text(0.3, 0.06, "vacuum stable", fontsize=7.5, color=INK2)
ax1.text(1.25, 0.06, "vacuum unstable", fontsize=7.5, color=INK2)
for s in ["top", "right"]: ax1.spines[s].set_visible(False)

# phase portrait, K~ = 0.5, lambda = 2 lambda_c
K, lam0 = 0.5, 2*lamc
x = np.linspace(-2, 2, 200); X, Y = np.meshgrid(x, x); Z = X + 1j*Y
F = -(1j*w + k1/2)*Z - 2j*lam0*np.conj(Z) - (1j*K + k2)*np.abs(Z)**2*Z
ax2.streamplot(X, Y, F.real, F.imag, color="#9b9a95", density=1.1, linewidth=0.6, arrowsize=0.7)
a0 = alpha_t(lam0, K)
for s in (+1, -1):
    ax2.plot((s*a0).real, (s*a0).imag, "o", ms=7, color=ORANGE, mec="white", mew=1.2, zorder=5)
ax2.plot(0, 0, "o", ms=7, mfc="white", mec=INK, mew=1.2, zorder=5)
ax2.annotate(r"$+\tilde\alpha$", ((a0).real, (a0).imag), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
ax2.annotate(r"$-\tilde\alpha$", ((-a0).real, (-a0).imag), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
ax2.set_xlim(-2, 2); ax2.set_ylim(-2, 2); ax2.set_aspect("equal")
ax2.set_xlabel(r"$\mathrm{Re}\,\tilde\alpha$"); ax2.set_ylabel(r"$\mathrm{Im}\,\tilde\alpha$")
ax2.text(-1.88, 1.66, "(b)", fontsize=9, color=INK, bbox=dict(fc="white", ec="none", pad=1.5), zorder=6)
fig.tight_layout()
fig.savefig("fig_kerr_meanfield.pdf"); fig.savefig("fig_kerr_meanfield.png", dpi=200)
print("alpha~(2 lam_c, K=0.5) =", a0, " n~ =", abs(a0)**2)
