#!/usr/bin/env python
"""
wigner_pn figure of one parameter point, with a zoom into the centre of phase space.

The figure is the one finalize.py writes for every point (Wigner function on top, P(n)
below, even and odd sector side by side), drawn with the same tpdt.plotting functions from
the stored summary files. Each Wigner panel additionally gets an inset showing the centre.

The inset is recomputed from the stored cavity density matrix on a fine grid. The fringes
between opposite lobes have period pi / x_lobe (about 0.2 for lobes at |X| ~ 15), which is
about the spacing of the stored 201-point grid, so magnifying the stored grid would only
show aliasing.

    python plot_wigner_zoom.py results/thesis/N15_k0.05_wc1_wa1_t70_Tinf-n75/lam0.5350_M200
    python plot_wigner_zoom.py <point_dir> --zoom 3 --zoom-grid 601 --loc "lower right" \\
        --out figures/wigner_zoom_lam0535

Options
  --zoom         half-width of the zoomed window in X and P (default 2.5)
  --zoom-grid    grid points per axis in the inset (default 401)
  --loc          inset corner: auto (default: an empty corner, never the one with the
                 panel letter), "upper right", "lower right", "lower left", "upper left"
  --size         inset size as a fraction of the panel (default: the largest of
                 0.38, 0.36, ..., 0.26 that leaves the lobes uncovered)
  --inset-clim   colour scale of the inset: auto (default, its own symmetric scale),
                 main (same as the big panel, the centre then saturates) or a number
  --main-grid    also recompute the big panels on a finer grid (default: the stored grid,
                 identical to wigner_pn.png)
  --no-indicator no frame and connector lines around the zoomed region in the big panel

Writes <out>.png and <out>.pdf (default: <point_dir>/wigner_pn_zoom). Needs qutip, i.e. the
same environment as finalize.py. Run it from the repository root (it imports tpdt).
"""
import argparse
import time
from pathlib import Path

import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle

from tpdt.analysis import compute_wigner
from tpdt.model import SECTORS, embed_cavity_rho
from tpdt.plotting import (INK, SECTOR_LABEL, _load, _panel_label, _title, draw_pn,
                           draw_wigner, plt, wigner_color_limit)

CORNERS = ("upper right", "lower right", "lower left", "upper left")   # preference order
SIGN = {"upper right": (1, 1), "lower right": (1, -1), "lower left": (-1, -1),
        "upper left": (-1, 1)}
PAD = 0.025          # gap between inset and panel edge, as a fraction of the panel
SIZES = (0.38, 0.36, 0.34, 0.32, 0.30, 0.28, 0.26)    # tried in this order without --size
HIDE_OK = 0.003      # largest acceptable fraction of a panel's |W| weight under the inset


def corner_bounds(loc, size, pad=PAD):
    """Inset position [x0, y0, width, height] in panel (axes) coordinates."""
    vert, horiz = loc.split()
    x0 = 1.0 - pad - size if horiz == "right" else pad
    y0 = 1.0 - pad - size if vert == "upper" else pad
    return [x0, y0, size, size]


def covered_weight(xvec, W, bounds):
    """Fraction of sum |W| of the big panel that an inset at `bounds` would hide."""
    lo, span = xvec[0], xvec[-1] - xvec[0]
    x0, y0, w, h = bounds
    in_x = (xvec >= lo + x0 * span) & (xvec <= lo + (x0 + w) * span)
    in_y = (xvec >= lo + y0 * span) & (xvec <= lo + (y0 + h) * span)
    a = np.abs(W)
    return float(a[np.ix_(in_y, in_x)].sum() / a.sum())       # W[p index, x index]


def choose_inset(main, sectors, loc="auto", size=None):
    """Corner and size of the inset: the first corner (preference order, never the one with
    the panel letter unless asked for) and the largest size for which the inset hides at most
    HIDE_OK of the Wigner weight in every panel. Returns (corner, size, hidden fraction)."""
    corners = CORNERS[:3] if loc == "auto" else (loc,)
    sizes = SIZES if size is None else (size,)
    for sz in sizes:
        hidden = {c: max(covered_weight(*main[s], corner_bounds(c, sz)) for s in sectors)
                  for c in corners}
        ok = [c for c in corners if hidden[c] <= HIDE_OK]
        if ok:
            return ok[0], sz, hidden[ok[0]]
    c = min(corners, key=hidden.get)          # nothing is free: least covered, smallest size
    return c, sizes[-1], hidden[c]


def lobe_radius(xvec, W, r_min=4.0):
    """Distance of the strongest Wigner peak outside the centre (the lobes) from the origin."""
    X, P = np.meshgrid(xvec, xvec)
    r = np.hypot(X, P)
    a = np.where(r > r_min, np.abs(W), 0.0)
    i = np.unravel_index(np.argmax(a), a.shape)
    return float(r[i]) if a[i] > 0 else float("nan")


def add_zoom(ax, xz, Wz, loc, size, clim, indicate=True):
    """Inset with the zoomed Wigner function in corner `loc` of panel `ax`, plus a frame
    around the zoomed region and one connector line along the (empty) diagonal."""
    axins = ax.inset_axes(corner_bounds(loc, size))
    axins.pcolormesh(xz, xz, Wz, cmap="RdBu", shading="auto", vmin=-clim, vmax=clim,
                     rasterized=True)
    axins.set_xlim(xz[0], xz[-1])
    axins.set_ylim(xz[0], xz[-1])
    axins.set_aspect("equal", adjustable="box")
    t = np.floor(0.9 * xz[-1]) if xz[-1] >= 1.2 else round(0.8 * xz[-1], 1)
    axins.set_xticks([-t, 0, t])
    axins.set_yticks([-t, 0, t])
    axins.tick_params(direction="in", length=2.5, width=0.6, color=INK,
                      labelbottom=False, labelleft=False, top=True, right=True)
    for side in axins.spines.values():          # tpdt.plotting hides top/right spines
        side.set_visible(True)
        side.set_color(INK)
        side.set_linewidth(0.8)
    box = dict(facecolor="white", alpha=0.85, edgecolor="none", pad=0.8)
    axins.text(0.5, 0.97, rf"$|W|\leq{clim:.2f}$", transform=axins.transAxes, ha="center",
               va="top", fontsize=6.5, color=INK, bbox=box)
    axins.text(0.5, 0.03, rf"$|X|,|P|\leq{xz[-1]:g}$", transform=axins.transAxes,
               ha="center", va="bottom", fontsize=6.5, color=INK, bbox=box)
    if indicate:
        z = xz[-1]
        style = dict(color=INK, linewidth=0.8, alpha=0.8, zorder=3)
        ax.add_patch(Rectangle((-z, -z), 2 * z, 2 * z, fill=False, **style))
        sx, sy = SIGN[loc]          # from the frame corner facing the inset to the inset corner
        ax.add_artist(ConnectionPatch(xyA=(sx * z, sy * z), coordsA="data", axesA=ax,
                                      xyB=((1 - sx) / 2, (1 - sy) / 2), coordsB="axes fraction",
                                      axesB=axins, **style))
    return axins


def make_figure(point_dir, out, zoom=2.5, zoom_grid=401, loc="auto", size=None,
                inset_clim="auto", main_grid=None, indicate=True, log=print):
    point_dir = Path(point_dir)
    rep, data = _load(point_dir)
    key = rep["physics_key"]
    M = int(key["M"])
    sectors = [s for s in SECTORS if s in data]
    x_max = float(rep.get("wigner", {}).get("x_max", data[sectors[0]]["xvec"][-1]))

    main, zoomed = {}, {}
    for s in sectors:
        d = data[s]
        rho = embed_cavity_rho(d["rho_sector"], d["fock"], M)
        t0 = time.time()
        main[s] = compute_wigner(rho, x_max, main_grid) if main_grid else (d["xvec"], d["W"])
        zoomed[s] = compute_wigner(rho, zoom, zoom_grid)
        log(f"  {s}: Wigner function recomputed in {time.time() - t0:.1f} s")

    loc, size, hidden = choose_inset(main, sectors, loc, size)
    if hidden > HIDE_OK:
        log(f"  warning: the inset covers {100 * hidden:.1f} % of the Wigner weight; "
            f"try another --loc or a smaller --size")

    # same layout as tpdt.plotting.plot_point (wigner_pn.png), plus the insets
    fig, axes = plt.subplots(2, len(sectors), figsize=(4.6 * len(sectors), 8.4),
                             squeeze=False, constrained_layout=True)
    letters = iter("abcdefgh")
    for j, s in enumerate(sectors):
        ax = axes[0, j]
        xvec, W = main[s]
        draw_wigner(ax, xvec, W, fig=fig)
        ax.set_title(SECTOR_LABEL[s])
        _panel_label(ax, next(letters))
        xz, Wz = zoomed[s]
        if inset_clim == "auto":
            clim = float(np.max(np.abs(Wz)))
        elif inset_clim == "main":
            clim = wigner_color_limit(W)
        else:
            clim = float(inset_clim)
        add_zoom(ax, xz, Wz, loc, size, clim, indicate)
    for j, s in enumerate(sectors):
        ax = axes[1, j]
        draw_pn(ax, data[s]["P_n"], s, rep["sectors"][s], M)
        _panel_label(ax, next(letters))
    fig.suptitle("Two-photon Dicke model, two-photon loss only\n" + _title(key, rep), fontsize=11)

    out = Path(out)
    if out.suffix in (".png", ".pdf"):
        out = out.with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        f = Path(f"{out}.{ext}")          # not with_suffix: names like lam0.535 contain dots
        fig.savefig(f)
        written.append(str(f))
    plt.close(fig)

    xv = main[sectors[0]][0]
    r_lobe = np.nanmean([lobe_radius(*main[s]) for s in sectors])
    log(f"  inset: {loc}, size {size:.2f}, window |X|,|P| <= {zoom:g}, "
        f"grid spacing {2 * zoom / (zoom_grid - 1):.4f}"
        f" (big panels: {xv[1] - xv[0]:.3f}); lobes at r = {r_lobe:.1f} -> fringe period "
        f"pi/r = {np.pi / r_lobe:.3f}")
    log("  wrote " + ", ".join(written))
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("point_dirs", nargs="+", help="point directories (with summary.json)")
    ap.add_argument("--out", default=None, help="output path without extension "
                    "(default: <point_dir>/wigner_pn_zoom); only with a single point")
    ap.add_argument("--zoom", type=float, default=2.5)
    ap.add_argument("--zoom-grid", type=int, default=401)
    ap.add_argument("--loc", default="auto", choices=("auto",) + CORNERS)
    ap.add_argument("--size", type=float, default=None)
    ap.add_argument("--inset-clim", default="auto")
    ap.add_argument("--main-grid", type=int, default=None)
    ap.add_argument("--no-indicator", action="store_true")
    args = ap.parse_args(argv)
    if args.out and len(args.point_dirs) > 1:
        ap.error("--out only works with a single point directory")
    if args.inset_clim not in ("auto", "main"):
        float(args.inset_clim)           # fail early on a typo
    for pd in args.point_dirs:
        print(pd, flush=True)
        make_figure(pd, args.out or Path(pd) / "wigner_pn_zoom", zoom=args.zoom,
                    zoom_grid=args.zoom_grid, loc=args.loc, size=args.size,
                    inset_clim=args.inset_clim, main_grid=args.main_grid,
                    indicate=not args.no_indicator)


if __name__ == "__main__":
    main()