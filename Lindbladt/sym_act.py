"""Sketch of the symmetry action on the four superradiant lobes of the two-photon Dicke model.

(a) Z4:  Pi_4 = exp[i pi (a^dag a + J_z)/2] rotates the cavity phase space by pi/2 and flips J_x.
(b) Z2:  Pi_4^2 = (-1)^N exp(i pi a^dag a) rotates the cavity phase space by pi; the spin is unchanged.

Each lobe is a photon coherent state times a collective spin coherent state,
    |alpha>|<=>,  |i alpha>|=>>,  |-alpha>|<=>,  |-i alpha>|=>>,
where <= (=>) is the spin coherent state with j_x < 0 (j_x > 0); j_z < 0 for all four.

Lobe radius and angle are the mean-field values for omega_a = omega_c = 1, kappa_2 = 0.05,
lambda = 0.535, N = 15, drawn in the quadratures a = (x + i p)/sqrt(2). The lobe width is schematic.
Convention: J_a = sum_i sigma_a^i.

Run:  python fig_symmetry_sketch.py   ->   tpdm_symmetry_sketch.pdf and .png
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

# ----------------------------------------------------------------------------- settings to edit
OUT = "tpdm_symmetry_sketch"     # output file name without extension
LOBE_ANGLE = None                # None: mean-field angle theta_0; or an angle in rad, e.g. 0.0
PHOTON = [r"|\alpha\rangle", r"|i\alpha\rangle", r"|{-\alpha}\rangle", r"|{-i\alpha}\rangle"]
SPIN = {-1: r"|\!\Leftarrow\!\rangle", +1: r"|\!\Rightarrow\!\rangle"}   # j_x < 0, j_x > 0
TITLE_Z4 = r"(a) $\mathbb{Z}_4$:  $\Pi_4=e^{i\pi(a^{\!\dagger}\!a+J_z)/2}$"
NOTE_Z4 = r"rotation by $\pi/2$, spin flips $J_x\to{-J_x}$"
TITLE_Z2 = r"(b) $\mathbb{Z}_2$:  $\Pi_4^2=(-1)^N e^{i\pi a^{\!\dagger}\!a}$"
NOTE_Z2 = r"rotation by $\pi$, spin unchanged"

# Computer Modern labels. With a LaTeX installation you can set "text.usetex": True instead
# (then add \usepackage{amssymb} to "text.latex.preamble" for \mathbb).
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "axes.formatter.use_mathtext": True,
    "mathtext.fontset": "cm",
    "font.size": 9,
    "pdf.fonttype": 42,
})

# ----------------------------------------------------------------------------- mean-field lobes
wa = wc = 1.0
k2, lam, N = 0.05, 0.535, 15

s = 4 * lam**2                                   # quadratic for n^2, stable (upper) root
A_q = k2**2 * (4 * s * wc**2 + wa**2 * k2**2)
B_q = wc**2 * (4 * s * wc**2 + 2 * wa**2 * k2**2 - 4 * s**2)
C_q = wa**2 * wc**4
n = np.sqrt((-B_q + np.sqrt(B_q**2 - 4 * A_q * C_q)) / (2 * A_q))
theta0 = 0.5 * np.arctan(k2 * n / wc) if LOBE_ANGLE is None else LOBE_ANGLE
R = np.sqrt(2 * N * n)                           # lobe radius in (x, p)

# ----------------------------------------------------------------------------- style
C_A, C_B = "#128a60", "#6250d6"    # pair {|alpha>,|-alpha>} (j_x<0) and {|i alpha>,|-i alpha>} (j_x>0)
INK, SOFT, GRID = "#0b0b0b", "#5f5e5a", "#c3c2b7"
SIG = 1.7                          # schematic lobe width (a coherent state has 1/sqrt(2))
GAP = 3.0                          # arrows stop this far from a lobe centre
L = R + 15                         # half width of each panel
H = R + 10                         # half height of each panel
FIG_W, FIG_H = 6.3, 3.45           # figure size in inches (6.3 in = typical thesis text width)
AX_BOX = (0.10, 0.76)              # bottom and height of the two panels, in figure units
PT = min(0.5 * FIG_W / (2 * L), AX_BOX[1] * FIG_H / (2 * H)) * 72   # points per data unit


def lobe(k):
    ang = theta0 + k * np.pi / 2
    return R * np.cos(ang), R * np.sin(ang), ang


def head(ax, p_from, p_to, color, lw):
    """Solid arrowhead at p_to, pointing from p_from to p_to."""
    ax.add_patch(FancyArrowPatch(p_from, p_to, arrowstyle="-|>", mutation_scale=8, lw=lw,
                                 color=color, shrinkA=0, shrinkB=0, zorder=5))


def arc(ax, r, a0, a1, color, lw=0.9):
    """Circular arc around the origin from angle a0 to a1 with an arrowhead at a1."""
    t = np.linspace(a0, a1, 200)
    x, y = r * np.cos(t), r * np.sin(t)
    ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", zorder=4)
    head(ax, (x[-8], y[-8]), (x[-1], y[-1]), color, lw)


def swoop(ax, k_from, k_to, color, rad=0.42, lw=1.0):
    """Curved double arrow between two lobes, bending counter-clockwise around the origin."""
    p0, p1 = lobe(k_from)[:2], lobe(k_to)[:2]
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="<|-|>",
                                 mutation_scale=8, lw=lw, color=color, shrinkA=GAP * PT,
                                 shrinkB=GAP * PT, zorder=4))


def draw_base(ax):
    """Axes, the four lobes and their labels."""
    ax.plot([-L + 1, L - 1], [0, 0], color=GRID, lw=0.5, zorder=1)
    ax.plot([0, 0], [-H + 1, H - 1.6], color=GRID, lw=0.5, zorder=1)
    ax.text(L - 1, -0.9, r"$x$", ha="right", va="top", color=SOFT)
    ax.text(0.9, H - 0.4, r"$p$", ha="left", va="top", color=SOFT)

    xs = np.linspace(-L, L, 700)
    X, P = np.meshgrid(xs, xs)
    for k in range(4):
        cx, cp, ang = lobe(k)
        col = C_A if k % 2 == 0 else C_B
        g = np.exp(-((X - cx) ** 2 + (P - cp) ** 2) / (2 * SIG**2))
        cmap = mpl.colors.LinearSegmentedColormap.from_list("lobe", ["#ffffff00", col])
        ax.contourf(X, P, np.ma.masked_less(g, 0.05), levels=np.linspace(0.05, 1, 9),
                    cmap=cmap, zorder=3)

        spin = -1 if k % 2 == 0 else +1
        c, sn = np.cos(ang), np.sin(ang)
        if abs(c) >= abs(sn):
            ha, va = ("left" if c > 0 else "right"), "center"
        else:
            ha, va = "center", ("bottom" if sn > 0 else "top")
        rl = R + 3.3
        ax.text(rl * c, rl * sn, f"${PHOTON[k]}\\,{SPIN[spin]}$", ha=ha, va=va,
                fontsize=9.5, color=INK, zorder=6)

    ax.set_xlim(-L, L)
    ax.set_ylim(-H, H)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_z4(ax):
    draw_base(ax)
    d = (GAP + 0.6) / R                              # angular gap around each lobe
    for k in range(4):                               # Pi_4: lobe k -> lobe k+1
        arc(ax, R, theta0 + k * np.pi / 2 + d, theta0 + (k + 1) * np.pi / 2 - d, INK)
    am = theta0 + np.pi / 4
    ax.text((R + 2.4) * np.cos(am), (R + 2.4) * np.sin(am), r"$\Pi_4$", ha="center",
            va="center", fontsize=9.5, color=INK)
    # angle marker: pi/2 between neighbouring lobes
    for k in (0, 1):
        ang = theta0 + k * np.pi / 2
        ax.plot([0, (R - GAP) * np.cos(ang)], [0, (R - GAP) * np.sin(ang)], color=SOFT,
                lw=0.6, ls=(0, (1.5, 2)), zorder=2)
    arc(ax, 4.2, theta0 + 0.06, theta0 + np.pi / 2 - 0.06, SOFT, lw=0.8)
    ax.text(6.8 * np.cos(am), 6.8 * np.sin(am), r"$\pi/2$", ha="center", va="center",
            fontsize=8.5, color=SOFT)


def panel_z2(ax):
    draw_base(ax)
    swoop(ax, 0, 2, C_A)                             # Pi_4^2: |alpha> <-> |-alpha>
    swoop(ax, 1, 3, C_B)                             #         |i alpha> <-> |-i alpha>
    # angle marker: pi between opposite lobes
    ax.plot([-4.0 * np.cos(theta0), 4.0 * np.cos(theta0)],
            [-4.0 * np.sin(theta0), 4.0 * np.sin(theta0)], color=SOFT, lw=0.6,
            ls=(0, (1.5, 2)), zorder=2)
    arc(ax, 2.8, theta0 + 0.08, theta0 + np.pi - 0.08, SOFT, lw=0.8)
    am = theta0 + 0.95
    ax.text(4.6 * np.cos(am), 4.6 * np.sin(am), r"$\pi$", ha="center", va="center",
            fontsize=8.5, color=SOFT)


fig = plt.figure(figsize=(FIG_W, FIG_H))
ax1 = fig.add_axes([0.0, AX_BOX[0], 0.5, AX_BOX[1]])
ax2 = fig.add_axes([0.5, AX_BOX[0], 0.5, AX_BOX[1]])
panel_z4(ax1)
panel_z2(ax2)

for x0, title, note in ((0.02, TITLE_Z4, NOTE_Z4), (0.52, TITLE_Z2, NOTE_Z2)):
    fig.text(x0, 0.975, title, ha="left", va="top", fontsize=9.5, color=INK)
    fig.text(x0 + 0.033, 0.91, note, ha="left", va="top", fontsize=8.5, color=SOFT)

handles = [
    Line2D([], [], ls="", marker="o", ms=7, mfc=C_A, mec="none",
           label=rf"$\{{{PHOTON[0]},{PHOTON[2]}\}}\otimes{SPIN[-1]}$, $j_x<0$"),
    Line2D([], [], ls="", marker="o", ms=7, mfc=C_B, mec="none",
           label=rf"$\{{{PHOTON[1]},{PHOTON[3]}\}}\otimes{SPIN[+1]}$, $j_x>0$"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5,
           handletextpad=0.4, columnspacing=2.5, bbox_to_anchor=(0.5, 0.0))

fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=300)
print(f"n = {n:.3f}, lobe radius sqrt(2Nn) = {R:.2f}, lobe angle = {np.degrees(theta0):.1f} deg")
print(f"saved {OUT}.pdf and {OUT}.png")