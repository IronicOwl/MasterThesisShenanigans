#!/usr/bin/env python3
"""
phase_diagram_248.py -- mean-field phase diagram of the 2 -> 4 -> 8 cascade in the dissipative
two-photon Dicke model with an LMG term and collective spin decay.

    H  = wc a^+a + (wa/2) Jz + (lam/N) Jx (a^2 + a^+2) - (chi/N) Jx^2
    L1 = sqrt(kappa2/N) a^2 ,   L2 = sqrt(gamma/N) J_-              (J = 2 x spin N/2)

Stable mean-field states and the Liouvillian zero modes they carry for N -> infinity:
    N   alpha = 0, spins down                  2   nothing broken
    Q   alpha = 0, spins ordered (+-jx)        4   weak Z4 -> strong Z2
    S   four lobes, alpha != 0                 8   both broken
Critical lines (derivation in mean_field_section.tex):
    chi_c      = wa/4 + gamma^2/wa                         N unstable, Q appears (continuous)
    lam_p(chi) = wc / (2 |jx^Q|)                           Q parametrically unstable
    lam_S(chi) = min over zeta of lam(zeta) on S branch    lobes appear at a fold (first order)
Regions: 2 (N), 2|8 (N+S), 4 (Q), 4|8 (Q+S), 8 (S). Valid for chi >= 0 and 0 <= gamma < wa/2.

Each stable state's stability region is drawn as a light wash of its colour (N grey, Q blue,
S orange), so coexistence regions show up where two washes overlap.

Outputs in --outdir:
    phase_diagram.{pdf,png}         (chi, lam) plane, one panel per kappa2
    cuts.{pdf,png}                  chi-cut through 2->4, soft cavity mode of Q,
                                    lam-cuts through 4->8 (one per kappa2)
    critical_lines_kappa2_<k>.csv   chi, lam_S, lam_p: the data behind the diagram

Usage:
    python phase_diagram_248.py                                 # kappa2 = 0.05 and 0.5, gamma = 0.1
    python phase_diagram_248.py --kappa2 0.5 --gamma 0.2 --chi-max 1.5 --lam-max 1.0
    python phase_diagram_248.py --check                         # verify the regions by direct
                                                                # stability analysis on a grid
    python phase_diagram_248.py --serif                         # serif fonts for a LaTeX thesis
"""
import argparse
import os

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import brentq, minimize_scalar

# colours: blue/orange validated as a categorical pair on white (CVD-safe); grey = symmetric phase
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#898781"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
WASH = 0.20
PHASE_NAME = {"2": "N", "2|8": "N + S", "4": "Q", "4|8": "Q + S", "8": "S"}


# =====================================================================================  model
class Model:
    """Mean-field fixed points, critical lines and stability at fixed wc, wa, gamma, kappa2."""

    def __init__(self, wc=1.0, wa=1.0, gamma=0.1, kappa2=0.5):
        if not (wc > 0 and wa > 0 and kappa2 > 0):
            raise SystemExit("need wc, wa, kappa2 > 0")
        if not 0 <= gamma < wa / 2:
            raise SystemExit("this script assumes 0 <= gamma < wa/2; for gamma >= wa/2 the spin "
                             "transition is first order and N, Q coexist (see mean_field_section.tex)")
        self.wc, self.wa, self.g, self.k2 = wc, wa, gamma, kappa2
        self.eta = 2 * gamma / wa
        self.chi_c = wa / 4 + gamma ** 2 / wa

    # --- spin geometry at a fixed point: jy = eta jz jx, |j| = 1
    def B(self, jz, chi):
        return self.wa * (1 + self.eta ** 2 * jz ** 2) + 4 * chi * jz

    def jx2(self, jz):
        return (1 - jz ** 2) / (1 + self.eta ** 2 * jz ** 2)

    # --- Q and its parametric threshold
    def q_jz(self, chi):
        """jz of the Q state; nan where Q does not exist (chi <= chi_c)."""
        chi = np.asarray(chi, float)
        root = np.sqrt(np.maximum(chi ** 2 - self.g ** 2, 0.0))
        out = np.where(chi > self.chi_c, -self.wa / (2 * (chi + root)), np.nan)
        return out if out.ndim else float(out)

    def q_jx(self, chi):
        return np.sqrt(self.jx2(self.q_jz(chi)))

    def lam_p(self, chi):
        """Q is unstable for lam > lam_p; nan where Q does not exist."""
        return self.wc / (2 * self.q_jx(chi))

    # --- S branch, parametrised by zeta = -jz in (0, zeta_max)
    def zeta_max(self, chi):
        return 1.0 if chi <= self.chi_c else -float(self.q_jz(chi))

    def branch(self, zeta, chi):
        """lam(zeta), r^2(zeta), jx^2(zeta): every zeta is an exact fixed point at lam(zeta)."""
        jx2 = self.jx2(-zeta)
        r2 = jx2 * self.B(-zeta, chi) / (2 * self.wc * zeta)
        lam = np.sqrt(self.wc ** 2 + self.k2 ** 2 * r2 ** 2) / (2 * np.sqrt(jx2))
        return lam, r2, jx2

    def fold(self, chi, n=3000):
        """(zeta_S, lam_S): minimum of lam(zeta), where the lobes appear."""
        zm = self.zeta_max(chi)
        z = zm * np.linspace(1e-6, 1 - 1e-9, n)
        i = int(np.argmin(self.branch(z, chi)[0]))
        r = minimize_scalar(lambda t: self.branch(t, chi)[0], method="bounded",
                            bounds=(z[max(i - 1, 0)], z[min(i + 1, n - 1)]), options=dict(xatol=1e-13))
        return r.x, r.fun

    def lam_S(self, chi):
        return self.fold(chi)[1]

    def s_states(self, chi, lam):
        """S states at (chi, lam), one copy each: list of (zeta, r^2, 'bright' | 'dim')."""
        zS, lS = self.fold(chi)
        if lam < lS:
            return []
        f = lambda z: self.branch(z, chi)[0] - lam
        zm = self.zeta_max(chi)
        out = []
        zb = brentq(f, 1e-12 * zm, zS, xtol=1e-15)
        out.append((zb, self.branch(zb, chi)[1], "bright"))
        hi = zm * (1 - 1e-12)
        if lam > lS and f(hi) > 0:                     # no dim state once Q is unstable
            zd = brentq(f, zS, hi, xtol=1e-15)
            out.append((zd, self.branch(zd, chi)[1], "dim"))
        return out

    def region(self, chi, lam):
        """Region label from the critical lines."""
        lS = self.lam_S(chi)
        if chi < self.chi_c:
            return "2" if lam < lS else "2|8"
        if lam < lS:
            return "4"
        return "4|8" if lam < self.lam_p(chi) else "8"

    # --- equations of motion and stability (used by --check)
    def rhs(self, v, chi, lam):
        x, y, jx, jy, jz = v
        n, R = x * x + y * y, x * x - y * y
        return np.array([(self.wc - 2 * lam * jx) * y - self.k2 * n * x,
                         -(self.wc + 2 * lam * jx) * x - self.k2 * n * y,
                         -self.wa * jy + 2 * self.g * jz * jx,
                         self.wa * jx + 4 * chi * jx * jz - 4 * lam * jz * R + 2 * self.g * jz * jy,
                         -4 * chi * jx * jy + 4 * lam * jy * R - 2 * self.g * (jx * jx + jy * jy)])

    def jacobian(self, v, chi, lam):
        x, y, jx, jy, jz = v
        wc, wa, g, k2 = self.wc, self.wa, self.g, self.k2
        R = x * x - y * y
        return np.array([
            [-k2 * (3 * x * x + y * y), wc - 2 * lam * jx - 2 * k2 * x * y, -2 * lam * y, 0, 0],
            [-(wc + 2 * lam * jx) - 2 * k2 * x * y, -k2 * (x * x + 3 * y * y), -2 * lam * x, 0, 0],
            [0, 0, 2 * g * jz, -wa, 2 * g * jx],
            [-8 * lam * jz * x, 8 * lam * jz * y, wa + 4 * chi * jz, 2 * g * jz, 4 * chi * jx - 4 * lam * R + 2 * g * jy],
            [8 * lam * jy * x, -8 * lam * jy * y, -4 * chi * jy - 4 * g * jx, -4 * chi * jx + 4 * lam * R - 4 * g * jy, 0]])

    def tangent_eigs(self, v, chi, lam):
        """The four physical eigenvalues: Jacobian on cavity plane + tangent plane of the sphere."""
        jx, jy, jz = v[2:]
        th, ph = np.arccos(np.clip(jz, -1, 1)), np.arctan2(jy, jx)
        P = np.zeros((5, 4))
        P[0, 0] = P[1, 1] = 1.0
        P[2:, 2] = [np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)]
        P[2:, 3] = [-np.sin(ph), np.cos(ph), 0.0]
        return np.linalg.eigvals(P.T @ self.jacobian(v, chi, lam) @ P)

    def classify_direct(self, chi, lam, tol=1e-9):
        """Region label from the stable fixed points found directly (neutral counts as stable).
        S states come from F(jz) = 0 at fixed lam, independent of the zeta parametrisation."""
        stable = set()
        vN = np.array([0, 0, 0, 0, -1.0])
        if self.tangent_eigs(vN, chi, lam).real.max() <= tol:
            stable.add("N")
        if chi > self.chi_c:
            jz = self.q_jz(chi)
            jx = np.sqrt(self.jx2(jz))
            vQ = np.array([0, 0, jx, self.eta * jz * jx, jz])
            assert np.abs(self.rhs(vQ, chi, lam)).max() < 1e-12
            if self.tangent_eigs(vQ, chi, lam).real.max() <= tol:
                stable.add("Q")
        w = (self.wc / (2 * lam)) ** 2
        if w < 1:
            zlim = np.sqrt((1 - w) / (1 + self.eta ** 2 * w))      # (2 lam jx)^2 > wc^2 inside
            F = lambda z: (self.jx2(-z) * self.B(-z, chi) - 2 * self.wc * z
                           * np.sqrt(np.maximum(4 * lam ** 2 * self.jx2(-z) - self.wc ** 2, 0)) / self.k2)
            z = zlim * np.linspace(1e-7, 1 - 1e-10, 4000)
            Fz = F(z)
            for i in np.nonzero(np.sign(Fz[:-1]) != np.sign(Fz[1:]))[0]:
                zr = brentq(F, z[i], z[i + 1], xtol=1e-15)
                jz, jx = -zr, np.sqrt(self.jx2(-zr))
                r2 = np.sqrt(4 * lam ** 2 * jx ** 2 - self.wc ** 2) / self.k2
                al = np.sqrt(r2 * (-self.wc - 1j * self.k2 * r2) / (2 * lam * jx))
                v = np.array([al.real, al.imag, jx, self.eta * jz * jx, jz])
                if np.abs(self.rhs(v, chi, lam)).max() > 1e-8:
                    continue
                if self.tangent_eigs(v, chi, lam).real.max() <= tol:
                    stable.add("S")
        code = {"N": "2", "Q": "4", "S": "8"}
        return "|".join(code[s] for s in ("N", "Q", "S") if s in stable) or "none"


# =====================================================================================  data
def critical_lines(m, chi_min, chi_max, n=900):
    """chi grids left/right of chi_c with lam_S and lam_p (lam_p = inf at chi_c)."""
    out = {}
    if chi_min < m.chi_c:
        cl = np.linspace(chi_min, min(m.chi_c, chi_max), max(int(n * (min(m.chi_c, chi_max) - chi_min)
                                                                  / (chi_max - chi_min)), 50))
        out["left"] = (cl, np.array([m.lam_S(c) for c in cl]))
    if chi_max > m.chi_c:
        lo = max(m.chi_c, chi_min)
        cr = np.linspace(lo, chi_max, max(int(n * (chi_max - lo) / (chi_max - chi_min)), 50))
        lp = np.array([m.lam_p(c) if c > m.chi_c else np.inf for c in cr])
        out["right"] = (cr, np.array([m.lam_S(c) for c in cr]), lp)
    return out


def save_csv(m, lines, path):
    rows = []
    if "left" in lines:
        rows += [(c, s, np.nan) for c, s in zip(*lines["left"])]
    if "right" in lines:
        rows += [(c, s, p) for c, s, p in zip(*lines["right"])]
    hdr = (f"wc={m.wc:g} wa={m.wa:g} gamma={m.g:g} kappa2={m.k2:g} chi_c={m.chi_c:.6g}\n"
           "chi,lam_S,lam_p   (lam_p = nan where Q does not exist, inf at chi_c)")
    np.savetxt(path, np.array(rows), delimiter=",", header=hdr, fmt="%.8g")


# =====================================================================================  plotting
def setup_style(serif):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rc = {"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
          "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
          "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
          "xtick.major.size": 3, "ytick.major.size": 3, "axes.labelcolor": INK, "text.color": INK,
          "axes.titlecolor": INK, "legend.frameon": False, "savefig.bbox": "tight",
          "savefig.pad_inches": 0.03, "lines.solid_capstyle": "round", "lines.dash_capstyle": "round",
          "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "figure.facecolor": "white"}
    if serif:
        rc.update({"font.family": "serif", "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
                   "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})
    plt.rcParams.update(rc)
    return plt


def label_regions(fig, ax, regions, xlim, ylim, avoid=None, n=320):
    """Put each region's label where a box of the label's size fits inside the region (and off the
    cut lines), at the most interior such point. Falls back to the bare number for thin regions."""
    from scipy.ndimage import minimum_filter
    fig.canvas.draw()
    bb = ax.get_window_extent()
    W, H = bb.width * 72 / fig.dpi, bb.height * 72 / fig.dpi           # axes size in points
    X, Y = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    free = np.ones_like(X, bool) if avoid is None else ~avoid(X, Y, W, H)
    for key, inside in regions.items():
        M = inside(X, Y) & free
        for w_pt, h_pt, full in ((30, 23, True), (19, 13, False)):    # label box incl. padding
            hx, hy = int(np.ceil(w_pt / 2 / W * (n - 1))), int(np.ceil(h_pt / 2 / H * (n - 1)))
            E = minimum_filter(M.astype(np.uint8), size=(2 * hy + 1, 2 * hx + 1), mode="constant", cval=0)
            if E.any():
                D = distance_transform_edt(np.pad(E.astype(bool), 1))[1:-1, 1:-1]
                iy, ix = np.unravel_index(np.argmax(D), D.shape)
                xy = (ix / (n - 1), iy / (n - 1))
                kw = dict(xycoords="axes fraction", textcoords="offset points", ha="center", va="center")
                if full:
                    ax.annotate(key, xy, xytext=(0, 5), fontsize=10, fontweight="bold", color=INK, **kw)
                    ax.annotate(PHASE_NAME[key], xy, xytext=(0, -6.5), fontsize=7.5, color=INK2, **kw)
                else:
                    ax.annotate(key, xy, xytext=(0, 0), fontsize=8.5, fontweight="bold", color=INK, **kw)
                break


def draw_phase_diagram(plt, models, a):
    from matplotlib.lines import Line2D
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 2.95), sharey=True, squeeze=False)
    xlim, ylim = (a.chi_min, a.chi_max), (a.lam_min, a.lam_max)
    all_lines, label_jobs = [], []
    for k, (ax, m) in enumerate(zip(axes[0], models)):
        L = critical_lines(m, *xlim)
        all_lines.append(L)
        if "left" in L:                                # N stable everywhere left of chi_c
            cl, sl = L["left"]
            ax.fill_between(cl, ylim[0], ylim[1], color=GREY, alpha=WASH, lw=0)
            ax.fill_between(cl, np.clip(sl, *ylim), ylim[1], color=ORANGE, alpha=WASH, lw=0)
            ax.plot(cl, sl, color=ORANGE, lw=1.6)
        if "right" in L:                               # Q stable below lam_p, S above lam_S
            cr, sr, pr = L["right"]
            ax.fill_between(cr, ylim[0], np.clip(pr, *ylim), color=BLUE, alpha=WASH, lw=0)
            ax.fill_between(cr, np.clip(sr, *ylim), ylim[1], color=ORANGE, alpha=WASH, lw=0)
            ax.plot(cr, sr, color=ORANGE, lw=1.6)
            ax.plot(cr[1:], pr[1:], color=BLUE, lw=1.6)
        if xlim[0] < m.chi_c < xlim[1]:
            ax.axvline(m.chi_c, color=INK2, lw=1.2)

        chis = np.concatenate([L[s][0] for s in ("left", "right") if s in L])
        lamS = np.concatenate([L[s][1] for s in ("left", "right") if s in L])
        # region masks for the labels (all loop variables bound now, not at call time)
        S_at = lambda X, cs=chis, ls=lamS: np.interp(X, cs, ls)
        if "right" in L:
            cr, _, pr = L["right"]
            P_at = lambda X, cr=cr[1:], pr=np.minimum(pr[1:], 1e6): np.interp(X, cr, pr, left=1e6)
        else:
            P_at = lambda X: np.full_like(X, 1e6)
        c = m.chi_c
        label_jobs.append((ax, {
            "2": lambda X, Y, c=c, S=S_at: (X < c) & (Y < S(X)),
            "2|8": lambda X, Y, c=c, S=S_at: (X < c) & (Y > S(X)),
            "4": lambda X, Y, c=c, S=S_at: (X > c) & (Y < S(X)),
            "4|8": lambda X, Y, c=c, S=S_at, P=P_at: (X > c) & (Y > S(X)) & (Y < P(X)),
            "8": lambda X, Y, c=c, P=P_at: (X > c) & (Y > P(X))}))
        if not a.no_cuts:
            ax.axhline(a.cut_lam, color=INK2, lw=0.7, ls=(0, (1, 2)))
            ax.axvline(a.cut_chi, color=INK2, lw=0.7, ls=(0, (1, 2)))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"spin–spin coupling $\chi$")
        ax.set_title((f"({'abcdefgh'[k]})  " if n > 1 else "") + rf"$\kappa_2 = {m.k2:g}$", loc="left")
        for s in ax.spines.values():
            s.set_color(AXIS)
    axes[0, 0].set_ylabel(r"two-photon coupling $\lambda$")
    m0 = models[0]
    handles = [Line2D([], [], color=INK2, lw=1.2, label=r"$\chi_c$: spins order (2$\to$4, continuous)"),
               Line2D([], [], color=ORANGE, lw=1.6, label=r"$\lambda_S$: lobes appear (fold)"),
               Line2D([], [], color=BLUE, lw=1.6, label=r"$\lambda_p$: Q unstable, $2\lambda|j_x|=\omega_c$")]
    if not a.no_cuts:
        handles.append(Line2D([], [], color=INK2, lw=0.7, ls=(0, (1, 2)),
                              label=rf"cuts: $\lambda={a.cut_lam:g}$, $\chi={a.cut_chi:g}$"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=2 if n == 1 else 4, bbox_to_anchor=(0.5, 0.0),
                     handlelength=1.8, columnspacing=1.4)
    fig.canvas.draw()                                  # reserve exactly the legend's height below the panels
    h = leg.get_window_extent(fig.canvas.get_renderer()).height / fig.bbox.height
    fig.tight_layout(rect=(0, h + 0.015, 1, 1))
    if not a.no_title:
        put_title(fig, axes[0], rf"$\omega_c={m0.wc:g}$, $\omega_a={m0.wa:g}$, $\gamma={m0.g:g}$, $N\to\infty$;"
                  + ("\n" if n == 1 else "  ") + "labels: number of zero modes and stable states")

    def near_cuts(X, Y, W, H):                         # 3 pt either side of the dotted cut lines
        if a.no_cuts:
            return np.zeros_like(X, bool)
        dx, dy = 3 / W * (xlim[1] - xlim[0]), 3 / H * (ylim[1] - ylim[0])
        return (abs(X - a.cut_chi) < dx) | (abs(Y - a.cut_lam) < dy)
    for ax, regions in label_jobs:
        label_regions(fig, ax, regions, xlim, ylim, avoid=near_cuts)
    return fig, all_lines


def put_title(fig, axes, text):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    top = max(ax.get_tightbbox(r).y1 for ax in axes if ax.get_visible()) / fig.bbox.height
    fig.text(0.5, top + 5 / (fig.bbox.height * 72 / fig.dpi), text, ha="center", va="bottom",
             fontsize=8, color=INK2)


def style_cut_axis(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)


def draw_cuts(plt, models, a):
    from matplotlib.lines import Line2D
    m0 = models[0]
    npan = 2 + len(models)
    ncol = 2
    nrow = int(np.ceil(npan / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4, 2.45 * nrow), squeeze=False)
    axes = axes.ravel()
    for ax in axes[npan:]:
        ax.set_visible(False)
    letters = iter("abcdefghij")
    solid, dash = dict(lw=1.6), dict(lw=1.1, ls=(0, (3, 2)))

    # (a) chi-cut through 2 -> 4 at lam < wc/2: no lobes possible, independent of kappa2
    ax = axes[0]
    style_cut_axis(ax)
    chis = np.linspace(a.chi_min, a.chi_max, 800)
    c = m0.chi_c
    ax.plot(chis[chis <= c], 0 * chis[chis <= c], color=GREY, **solid)
    ax.plot(chis[chis >= c], 0 * chis[chis >= c], color=GREY, **dash)
    cq = chis[chis > c]
    jx = m0.q_jx(cq)
    stab = a.cut_lam < m0.wc / (2 * jx)
    for sgn in (1, -1):
        ax.plot(np.r_[c, cq[stab]], sgn * np.r_[0, jx[stab]], color=BLUE, **solid)
        if (~stab).any():
            ax.plot(cq[~stab], sgn * jx[~stab], color=BLUE, **dash)
    ax.axvline(c, color=INK2, lw=0.8)
    ax.text(c, 1.02, rf" $\chi_c={c:.3g}$", color=INK2, fontsize=7.5, ha="left", va="bottom")
    ax.set_xlim(a.chi_min, a.chi_max)
    ax.set_ylim(-1.1, 1.18)
    ax.set_xlabel(r"spin–spin coupling $\chi$")
    ax.set_ylabel(r"$j_x=\langle J_x\rangle/N$")
    ax.set_title(f"({next(letters)})  2$\\to$4 at $\\lambda={a.cut_lam:g}$", loc="left")
    ax.legend(handles=[Line2D([], [], color=GREY, **solid, label="N"),
                       Line2D([], [], color=BLUE, **solid, label=r"Q ($\pm j_x$)"),
                       Line2D([], [], color=INK2, **dash, label="unstable")], loc="center right",
              bbox_to_anchor=(1.0, 0.64), handlelength=1.6)

    # (b) soft cavity mode of Q: omega' = sqrt(wc^2 - (2 lam jx)^2), closes at lam_p
    ax = axes[1]
    style_cut_axis(ax)
    lo, hi = a.cut_lam_range
    lams = np.linspace(lo, hi, 800)
    letter = next(letters)
    if a.cut_chi > m0.chi_c:
        jxq = m0.q_jx(a.cut_chi)
        lp = m0.lam_p(a.cut_chi)
        w = np.sqrt(np.maximum(m0.wc ** 2 - (2 * lams * jxq) ** 2, 0))
        ax.plot(lams[lams <= lp], w[lams <= lp], color=BLUE, **solid)
        ax.axvspan(lo, lp, color=BLUE, alpha=WASH, lw=0)
        ax.axvspan(lp, hi, color=ORANGE, alpha=WASH, lw=0)
        ax.axvline(lp, color=BLUE, lw=0.8)
        ax.text(lp, 0.9 * m0.wc, rf" $\lambda_p={lp:.4g}$", color=INK2, fontsize=7.5, ha="left", va="center")
        ax.axhline(m0.wc, color=GREY, lw=0.8)
        ax.text(lo, m0.wc, r" N: $\omega_c$", color=INK2, fontsize=7.5, va="bottom")
        ax.text(0.5 * (lp + hi), 0.5 * m0.wc, "Q unstable\n(lobes)", color=INK2, fontsize=7.5,
                ha="center", va="center")
        ax.set_ylim(0, 1.2 * m0.wc)
    else:
        ax.text(0.5, 0.5, rf"no Q at $\chi={a.cut_chi:g}\leq\chi_c$", transform=ax.transAxes,
                ha="center", va="center", color=INK2)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"two-photon coupling $\lambda$")
    ax.set_ylabel(r"cavity frequency in Q, $\omega'$")
    ax.set_title(f"({letter})  soft mode of Q at $\\chi={a.cut_chi:g}$", loc="left")

    # (c, d, ...) lam-cuts through 4 -> 8, one per kappa2
    for k, m in enumerate(models):
        ax = axes[2 + k]
        style_cut_axis(ax)
        chi = a.cut_chi
        base, base_col = ("Q", BLUE) if chi > m.chi_c else ("N", GREY)
        lp = m.lam_p(chi) if base == "Q" else np.inf
        zS, lS = m.fold(chi)
        zm = m.zeta_max(chi)
        z = zm * np.concatenate([np.logspace(-7, -1, 600, endpoint=False), np.linspace(0.1, 1 - 1e-9, 1500)])
        lam, r2, _ = m.branch(z, chi)
        keep = lam <= hi * 1.05
        bright, dim = keep & (z <= zS), keep & (z >= zS)
        ax.axvspan(lo, min(lp, hi), color=base_col, alpha=WASH, lw=0)
        ax.axvspan(lS, hi, color=ORANGE, alpha=WASH, lw=0)
        ax.plot([lo, min(lp, hi)], [0, 0], color=base_col, **solid)
        if lp < hi:
            ax.plot([lp, hi], [0, 0], color=base_col, **dash)
        ax.plot(lam[bright], r2[bright], color=ORANGE, **solid)
        ax.plot(lam[dim], r2[dim], color=ORANGE, **dash)
        ymax = 1.08 * r2[bright][lam[bright] <= hi].max()
        ax.axvline(lS, color=ORANGE, lw=0.8)
        ax.annotate(rf"$\lambda_S={lS:.4g}$", (lS, 0.72 * ymax), xytext=(-2.5, 0), textcoords="offset points",
                    rotation=90, ha="right", va="center", color=INK2, fontsize=7.5)
        rS = m.branch(zS, chi)[1]
        arrow = dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=7, shrinkA=0, shrinkB=0)
        ax.annotate("", xy=(lS, 0), xytext=(lS, rS), arrowprops=arrow)            # S -> base at lam_S
        if lp < hi:
            ax.axvline(lp, color=BLUE, lw=0.8)
            ax.annotate(rf"$\lambda_p={lp:.4g}$", (lp, 0.72 * ymax), xytext=(2.5, 0), textcoords="offset points",
                        rotation=90, ha="left", va="center", color=INK2, fontsize=7.5)
            r2_jump = m.s_states(chi, lp)[0][1]
            ax.annotate("", xy=(lp, r2_jump), xytext=(lp, 0), arrowprops=arrow)   # Q -> S at lam_p
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.04 * ymax, ymax)
        ax.set_xlabel(r"two-photon coupling $\lambda$")
        ax.set_ylabel(r"photons per spin $n/N=|\alpha|^2$")
        ax.set_title(f"({next(letters)})  4$\\to$8 at $\\chi={chi:g}$, $\\kappa_2={m.k2:g}$", loc="left")
        ax.legend(handles=[Line2D([], [], color=base_col, **solid, label=f"{base} ($\\alpha=0$)"),
                           Line2D([], [], color=ORANGE, **solid, label="S, bright lobes"),
                           Line2D([], [], color=ORANGE, **dash, label="S, dim lobes (unstable)")],
                  loc="lower right", bbox_to_anchor=(1.0, 0.06), handlelength=1.6)
    fig.tight_layout()
    if not a.no_title:
        put_title(fig, axes, rf"$\omega_c={m0.wc:g}$, $\omega_a={m0.wa:g}$, $\gamma={m0.g:g}$, $N\to\infty$;"
                  "  solid: stable, dashed: unstable")
    return fig


# =====================================================================================  check
def check_regions(models, a, n=41, margin=0.004):
    """Classify a grid by direct stability analysis and compare with the critical lines."""
    bad_total = 0
    for m in models:
        counts, bad = {}, []
        for chi in np.linspace(a.chi_min, a.chi_max, n):
            lS = m.lam_S(chi)
            lp = m.lam_p(chi) if chi > m.chi_c else np.inf
            for lam in np.linspace(a.lam_min, a.lam_max, n):
                if abs(chi - m.chi_c) < margin or abs(lam - lS) < margin or abs(lam - lp) < margin:
                    continue                            # too close to a boundary to classify reliably
                want, got = m.region(chi, lam), m.classify_direct(chi, lam)
                counts[want] = counts.get(want, 0) + 1
                if want != got:
                    bad.append((round(chi, 4), round(lam, 4), want, got))
        bad_total += len(bad)
        print(f"check kappa2={m.k2:g}: {sum(counts.values())} grid points "
              f"({', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))}), {len(bad)} mismatches"
              + (f"; first: {bad[:5]}" if bad else ""), flush=True)
    return bad_total


# =====================================================================================  main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kappa2", type=float, nargs="+", default=[0.05, 0.5], help="one panel per value")
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--wc", type=float, default=1.0)
    ap.add_argument("--wa", type=float, default=1.0)
    ap.add_argument("--chi-min", type=float, default=0.0)
    ap.add_argument("--chi-max", type=float, default=1.0)
    ap.add_argument("--lam-min", type=float, default=0.3)
    ap.add_argument("--lam-max", type=float, default=0.8)
    ap.add_argument("--cut-lam", type=float, default=0.47, help="lambda of the 2->4 cut (must be < wc/2)")
    ap.add_argument("--cut-chi", type=float, default=0.7, help="chi of the 4->8 cuts")
    ap.add_argument("--cut-lam-range", type=float, nargs=2, default=[0.45, 0.8])
    ap.add_argument("--no-cuts", action="store_true", help="only the phase diagram")
    ap.add_argument("--check", action="store_true", help="verify the regions on a grid by direct stability analysis")
    ap.add_argument("--serif", action="store_true", help="serif (Computer Modern-like) fonts")
    ap.add_argument("--no-title", action="store_true", help="omit the parameter line above the panels")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--outdir", default="phase_diagram_248")
    a = ap.parse_args()
    if a.chi_min < 0 or a.chi_max <= a.chi_min or a.lam_max <= a.lam_min:
        raise SystemExit("need 0 <= chi-min < chi-max and lam-min < lam-max")
    if not a.no_cuts and a.cut_lam >= a.wc / 2:
        raise SystemExit("--cut-lam must be below wc/2 so that the 2->4 cut has no lobes")

    models = [Model(a.wc, a.wa, a.gamma, k) for k in a.kappa2]
    os.makedirs(a.outdir, exist_ok=True)
    m0 = models[0]
    print(f"chi_c = wa/4 + gamma^2/wa = {m0.chi_c:.6g}")
    if a.cut_chi > m0.chi_c:
        print(f"chi = {a.cut_chi:g}: Q has jz = {m0.q_jz(a.cut_chi):.4f}, |jx| = {m0.q_jx(a.cut_chi):.4f},"
              f" lam_p = {m0.lam_p(a.cut_chi):.4f} (any kappa2)")
    for m in models:
        zS, lS = m.fold(a.cut_chi)
        msg = f"kappa2 = {m.k2:g}: lam_S(chi={a.cut_chi:g}) = {lS:.4f} (n/N = {m.branch(zS, a.cut_chi)[1]:.3f} at the fold)"
        if a.cut_chi > m.chi_c:
            lp = m.lam_p(a.cut_chi)
            msg += f", Q+S coexist for {lS:.4f} < lam < {lp:.4f}, jump at lam_p to n/N = {m.s_states(a.cut_chi, lp)[0][1]:.3f}"
        print(msg)

    plt = setup_style(a.serif)
    fig, lines = draw_phase_diagram(plt, models, a)
    for fmt in a.formats:
        fig.savefig(os.path.join(a.outdir, f"phase_diagram.{fmt}"), dpi=a.dpi)
    plt.close(fig)
    for m, L in zip(models, lines):
        save_csv(m, L, os.path.join(a.outdir, f"critical_lines_kappa2_{m.k2:g}.csv"))
    if not a.no_cuts:
        fig = draw_cuts(plt, models, a)
        for fmt in a.formats:
            fig.savefig(os.path.join(a.outdir, f"cuts.{fmt}"), dpi=a.dpi)
        plt.close(fig)
    print(f"figures and data -> {a.outdir}/")
    if a.check and check_regions(models, a):
        raise SystemExit("CHECK FAILED: regions from the critical lines disagree with direct stability analysis")


if __name__ == "__main__":
    main()