"""Fig. 4.3: linear stability of the mean-field branches of the two-photon Dicke model (kappa_1 = 0).
All axes are linear.

(a) Photon number per spin of the two superradiant branches, Eq. (4.22), together with the stability
    threshold n_th of Eq. (4.31). The upper branch lies above the threshold (stable), the lower branch
    below it (unstable); all three meet at lambda_c (saddle-node). The vacuum n = 0 is stable for all lambda.
(b) Real parts of the eigenvalues, i.e. of the roots of the characteristic polynomial, Eq. (4.29),
        (mu + 2 k2 n)^2 (mu^2 + wa^2 / jz^2) - 4 wa^2 wc^2 = 0 .
    Upper branch: its two real eigenvalues (both negative). Lower branch: its positive real eigenvalue.
    The slow real eigenvalue of the two branches passes through zero at lambda_c (saddle-node).
    Not drawn: the complex pair of the upper branch, the spin precession (Im mu ~ wa/|jz|), whose real
    part is negative but only of order -10^-4 (checked below and printed when the script runs).

The script also checks that the roots of Eq. (4.29) agree with the eigenvalues of the Jacobian (4.27)
and that n_- < n_th < n_+ for every lambda > lambda_c.

Run:  python fig_mf_stability.py   ->   fig_mf_stability.pdf and fig_mf_stability.png
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# ----------------------------------------------------------------------------- settings
WA, WC, K2 = 1.0, 1.0, 0.05            # omega_a, omega_c, kappa_2
LAM_MAX = 0.65                         # right end of the lambda axis
LAM_LEFT = 0.48                        # left end of the lambda axis
NPTS = 800
OUT = "fig_mf_stability"

INK, SOFT, GRID = "#0b0b0b", "#6b6a65", "#e4e3dd"
C_SLOW, C_FAST = "#2a78d6", "#9a9893"   # slow real mode, second real mode (upper branch)

LAM_C = 0.5 * np.sqrt(WC**2 + WA * K2)


# ----------------------------------------------------------------------------- mean field
def branch(lam, sign):
    """Photon number n, |j_x| and j_z of the upper (sign=+1) or lower (sign=-1) branch."""
    s = 4 * lam**2
    disc = np.maximum((s - WC**2) ** 2 - (WA * K2) ** 2, 0.0)
    n2 = (WC**2 * (2 * s * (s - WC**2) - WA**2 * K2**2 + sign * 2 * s * np.sqrt(disc))
          / (K2**2 * (4 * s * WC**2 + WA**2 * K2**2)))
    n = np.sqrt(n2)
    jx = np.sqrt(WC**2 + K2**2 * n**2) / (2 * lam)
    jz = -WA * jx**2 / (2 * WC * n)
    return n, jx, jz


def n_threshold(lam):
    """Stability threshold, Eq. (4.31): stable if n > n_th."""
    return np.sqrt(WA * WC**2 / (K2 * (8 * lam**2 - WA * K2)))


def char_roots(n, jz):
    """Roots of (mu + a)^2 (mu^2 + w^2) - g^2, Eq. (4.29)."""
    a, w, g = 2 * K2 * n, WA / abs(jz), 2 * WA * WC
    p = np.polymul(np.polymul([1, a], [1, a]), [1, 0, w**2])
    p[-1] -= g**2
    return np.roots(p)


def jacobian(lam, n, jx_abs, jz):
    """Jacobian (4.27) in (x, y, j_x, j_y) at the k = 0 lobe (j_x < 0)."""
    th = 0.5 * np.arctan(K2 * n / WC)
    x, y = np.sqrt(n) * np.cos(th), np.sqrt(n) * np.sin(th)
    jx = -jx_abs
    return np.array([
        [-K2 * (3 * x**2 + y**2), WC - 2 * lam * jx - 2 * K2 * x * y, -2 * lam * y, 0],
        [-(WC + 2 * lam * jx) - 2 * K2 * x * y, -K2 * (x**2 + 3 * y**2), -2 * lam * x, 0],
        [0, 0, 0, -WA],
        [-8 * lam * jz * x, 8 * lam * jz * y, WA / jz**2, 0]])


def modes(roots):
    """Split the four roots into (fast real, slow real, complex pair)."""
    real = np.sort(roots[np.abs(roots.imag) < 1e-6].real)
    cplx = roots[np.abs(roots.imag) >= 1e-6]
    assert real.size == 2 and cplx.size == 2, roots
    return real[0], real[1], cplx[0]


# ----------------------------------------------------------------------------- data
t = np.linspace(1e-3, 1, NPTS)
lam = LAM_C + (LAM_MAX - LAM_C) * t**2          # dense near lambda_c, where the branches merge
n_up, jx_up, jz_up = branch(lam, +1)
n_lo, jx_lo, jz_lo = branch(lam, -1)
n_th = n_threshold(lam)
assert np.all(n_lo < n_th) and np.all(n_th < n_up), "threshold not between the branches"

up = np.array([modes(char_roots(n, jz)) for n, jz in zip(n_up, jz_up)])
lo = np.array([modes(char_roots(n, jz)) for n, jz in zip(n_lo, jz_lo)])
assert np.all(up[:, 1].real < 0) and np.all(up[:, 2].real < 0), "upper branch not stable"
assert np.all(lo[:, 1].real > 0), "lower branch not unstable"

# cross-check: roots of (4.29) = eigenvalues of the Jacobian (4.27)
dev = 0.0
for l in (0.52, 0.535, 0.7, LAM_MAX):
    for sgn in (+1, -1):
        n, jx, jz = branch(l, sgn)
        ev = np.sort_complex(np.linalg.eigvals(jacobian(l, n, jx, jz)))
        rt = np.sort_complex(char_roots(n, jz))
        dev = max(dev, np.max(np.abs(ev - rt)))
print(f"lambda_c = {LAM_C:.4f};  max |eig(J) - roots of (4.29)| = {dev:.1e}")
i = np.argmin(abs(lam - 0.535))
print(f"lambda = {lam[i]:.3f}:  upper Re mu = {up[i, 0].real:.4f}, {up[i, 1].real:.4f}, "
      f"{up[i, 2].real:.2e} +- {abs(up[i, 2].imag):.2f}i;  lower growth rate = {lo[i, 1].real:.4f}")

# ----------------------------------------------------------------------------- figure
print(f"upper branch, spin precession: Re mu between {up[:, 2].real.min():.1e} and {up[:, 2].real.max():.1e}")
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "legend.frameon": False})
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.3, 2.7), gridspec_kw={"wspace": 0.3})
fig.subplots_adjust(left=0.085, right=0.975, bottom=0.16, top=0.91)

# (a) branches and threshold
ax_a.plot(lam, n_up, color=INK, lw=1.4, label="upper branch (stable)")
ax_a.plot(lam, n_lo, color=INK, lw=1.4, ls=(0, (5, 2.5)), label="lower branch (unstable)")
ax_a.plot(lam, n_th, color=SOFT, lw=1.0, ls=(0, (6, 2, 1.5, 2)), label=r"threshold $n_{\rm th}$")
ax_a.plot([LAM_LEFT, LAM_MAX], [0, 0], color=INK, lw=1.4)            # vacuum, stable for all lambda
ax_a.text(LAM_C - 0.002, 0.45 * n_up.max(), rf"$\lambda_c={LAM_C:.3f}$", color=SOFT, fontsize=7.5,
          rotation=90, ha="right", va="center")
ax_a.text(LAM_MAX - 0.003, -0.3, "vacuum (stable)", color=SOFT, fontsize=7, ha="right", va="top")
ax_a.set_ylim(-1.6, None)
ax_a.set_ylabel(r"$n=\langle a^\dagger a\rangle/N$")
ax_a.legend(loc="upper left", bbox_to_anchor=(0.02, 1.0), fontsize=7.5, handlelength=2.6)

# (b) real eigenvalues: upper branch (negative), lower branch (positive)
ax_b.axhline(0, color=SOFT, lw=0.6)
ax_b.plot(lam, up[:, 0].real, color=C_FAST, lw=1.0)                 # second real mode, upper branch
ax_b.plot(lam, up[:, 1].real, color=C_SLOW, lw=1.4)
ax_b.plot(lam, lo[:, 1].real, color=INK, lw=1.4, ls=(0, (5, 2.5)))
x_lab = LAM_C + 0.62 * (LAM_MAX - LAM_C)
ax_b.text(x_lab, 0.45, "lower branch", fontsize=7, ha="center", va="center")
ax_b.text(x_lab, -0.55, "upper branch", fontsize=7, ha="center", va="center")
ax_b.text(LAM_MAX - 0.002, up[-1, 0].real - 0.12, "upper, 2nd real mode", fontsize=6.5, ha="right",
          va="top", color=SOFT)
ax_b.set_ylim(up[:, 0].real.min() - 0.45, None)
ax_b.set_ylabel(r"$\mathrm{Re}\,\mu$")

for ax, lab in ((ax_a, "(a)"), (ax_b, "(b)")):
    ax.set_xlim(LAM_LEFT, LAM_MAX)
    ax.set_xlabel(r"$\lambda$")
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.axvline(LAM_C, color=SOFT, lw=0.7, ls=":")
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.text(0.0, 1.03, lab, transform=ax.transAxes, fontweight="bold", va="bottom", ha="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=300)
print(f"saved {OUT}.pdf and {OUT}.png")