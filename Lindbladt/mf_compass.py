"""Mean-field prediction for the parity-resolved steady states (Fig. mf_wigner of the mean field section;
same layout and colour scale as the trajectory Wigner figure, Fig. 4.1 a,b).

Each parity sector: equal mixture of the cat states on the two axes,
    rho_even = (|C+^A><C+^A| + |C+^B><C+^B|)/2 ,  rho_odd = (|C-^A><C-^A| + |C-^B><C-^B|)/2 ,
with |C+-^P> ~ |alpha_P> +- |-alpha_P>, alpha_A = sqrt(N n) e^{i theta0}, alpha_B = i alpha_A
(the spin is traced out; it is identical within a pair, so the cat coherences survive the trace).
Wigner function of |a><b| for coherent states, with gamma = (X + iP)/sqrt(2):
    W_ab(gamma) = (1/pi) <b|a> exp[-2 (gamma - a)(gamma* - b*)],   normalised as  int W dX dP = 1.

Run:  python fig_mf_wigner.py   ->   fig_mf_wigner.pdf and fig_mf_wigner.png
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ----------------------------------------------------------------------------- settings
WA, WC, K2 = 1.0, 1.0, 0.05    # omega_a, omega_c, kappa_2
LAM, N = 0.535, 15             # coupling and number of spins
VMAX = 0.035                   # colour scale +-VMAX (same as the trajectory figure)
L, NPTS = 22, 901              # plot range |X|, |P| <= L and grid points
INSET = 1.0                    # inset range |X|, |P| <= INSET, on the scale +-1/pi
OUT = "fig_mf_wigner"          # output file name without extension
TITLE_EVEN = r"even sector ($\Pi_2=+1$)"   # Pi_2 = photon parity e^{i pi a^dag a}; note that
TITLE_ODD = r"odd sector ($\Pi_2=-1$)"     # Pi_4^2 = (-1)^N e^{i pi a^dag a} flips the sign for odd N

# ----------------------------------------------------------------------------- mean-field upper branch
s = 4 * LAM**2
disc = (s - WC**2) ** 2 - (WA * K2) ** 2
n2 = WC**2 * (2 * s * (s - WC**2) - WA**2 * K2**2 + 2 * s * np.sqrt(disc)) / (K2**2 * (4 * s * WC**2 + WA**2 * K2**2))
n = np.sqrt(n2)
theta0 = 0.5 * np.arctan(K2 * n / WC)
amp = np.sqrt(N * n)
alphaA = amp * np.exp(1j * theta0)


def W_cat(G, alpha, sign):
    """Wigner function of the normalised cat |alpha> + sign |-alpha>."""
    norm = 1.0 / (2 * (1 + sign * np.exp(-2 * abs(alpha) ** 2)))
    diag = (np.exp(-2 * abs(G - alpha) ** 2) + np.exp(-2 * abs(G + alpha) ** 2)) / np.pi
    cross = 2 * np.exp(-2 * abs(G) ** 2) * np.cos(4 * np.imag(G * np.conj(alpha))) / np.pi
    return norm * (diag + sign * cross)


def W_sector(X, P, sign):
    G = (X + 1j * P) / np.sqrt(2)
    return 0.5 * (W_cat(G, alphaA, sign) + W_cat(G, 1j * alphaA, sign))


xs = np.linspace(-L, L, NPTS)
X, P = np.meshgrid(xs, xs)
zs = np.linspace(-INSET, INSET, 401)
Xz, Pz = np.meshgrid(zs, zs)

fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.0))
for ax, sign, lab, title in [(axs[0], +1, "(a)", TITLE_EVEN), (axs[1], -1, "(b)", TITLE_ODD)]:
    W = W_sector(X, P, sign)
    im = ax.imshow(W, origin="lower", extent=[-L, L, -L, L], cmap="RdBu", vmin=-VMAX, vmax=VMAX,
                   interpolation="bilinear")
    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"$P$")
    ax.set_title(title, fontsize=11)
    ax.text(-0.93 * L, 0.89 * L, lab, fontweight="bold", fontsize=11, va="top",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    cb = fig.colorbar(im, ax=ax, extend="both", fraction=0.046, pad=0.04)
    cb.set_label(r"$W(X,P)$")
    # inset: interference pattern at the origin on its own colour scale
    axin = inset_axes(ax, width="34%", height="34%", loc="lower right", borderpad=0.6)
    Wz = W_sector(Xz, Pz, sign)
    axin.imshow(Wz, origin="lower", extent=[-INSET, INSET, -INSET, INSET], cmap="RdBu",
                vmin=-1 / np.pi, vmax=1 / np.pi, interpolation="bilinear")
    axin.set_xticks([])
    axin.set_yticks([])
    for spine in axin.spines.values():
        spine.set_linewidth(0.8)
with warnings.catch_warnings():          # the insets make tight_layout warn; harmless
    warnings.simplefilter("ignore")
    fig.tight_layout()
fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=160)
print(f"n = {n:.4f}, theta0 = {np.degrees(theta0):.2f} deg, |alpha| = {amp:.3f}, "
      f"radius sqrt(2Nn) = {np.sqrt(2) * amp:.3f}")
print(f"saved {OUT}.pdf and {OUT}.png")