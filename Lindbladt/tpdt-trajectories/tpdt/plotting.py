"""Figures for one parameter point and for whole sweeps (matplotlib, Agg)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .model import SECTORS

# categorical colours (fixed order: even first, odd second)
SECTOR_COLOR = {"even": "#2a78d6", "odd": "#eb6834"}
SECTOR_LABEL = {"even": r"even sector ($\Pi_2=+1$)", "odd": r"odd sector ($\Pi_2=-1$)"}
INK = "#1f1f1e"
MUTED = "#6b6a63"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "figure.dpi": 100,
})


def _load(point_dir: Path):
    rep = json.loads((Path(point_dir) / "summary.json").read_text())
    data = {}
    for s in SECTORS:
        f = Path(point_dir) / f"summary_{s}.npz"
        if f.exists():
            with np.load(f, allow_pickle=False) as z:
                data[s] = {k: z[k] for k in z.files}
    return rep, data


def _panel_label(ax, text):
    ax.text(0.03, 0.96, f"({text})", transform=ax.transAxes, ha="left", va="top",
            fontsize=11, fontweight="bold", color=INK,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))


def _title(key, rep):
    n_tr = "/".join(str(rep["sectors"][s]["n_traj"]) for s in SECTORS if s in rep["sectors"])
    return (rf"$N={key['N']}$, $M={key['M']}$, $\lambda={key['lam']:g}$, "
            rf"$\kappa_2={key['kappa2']:g}$, $\omega_c={key['wc']:g}$, $\omega_a={key['wa']:g}$, "
            rf"$t={key['t_max']:g}$, $N_\mathrm{{traj}}={n_tr}$")


def wigner_color_limit(W, clip_percentile=99.8):
    """Symmetric colour limit. In a single parity sector W(0,0) = +-1/pi exactly
    (W(0) = sum_n (-1)^n P(n) / pi), a few-pixel spike that would otherwise wash out
    the superradiant lobes. The limit is therefore taken at a high percentile of |W|;
    saturated values are marked by the colour-bar arrows. clip_percentile=100: no clipping."""
    a = np.abs(np.asarray(W))
    lim = float(np.percentile(a, clip_percentile)) if clip_percentile < 100 else float(a.max())
    return lim if lim > 0 else float(a.max() or 1.0)


def draw_wigner(ax, xvec, W, cbar=True, fig=None, label=r"$W(X,P)$", clip_percentile=99.8):
    lim = wigner_color_limit(W, clip_percentile)
    mesh = ax.pcolormesh(xvec, xvec, W, cmap="RdBu", shading="auto",
                         vmin=-lim, vmax=lim, rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"$P$")
    if cbar:
        extend = "both" if float(np.max(np.abs(W))) > lim * 1.0001 else "neither"
        cb = (fig or ax.figure).colorbar(mesh, ax=ax, fraction=0.046, pad=0.03, extend=extend)
        cb.set_label(label)
        cb.outline.set_visible(False)
    return mesh


def _sci(x):
    """1.2e-03 -> LaTeX 1.2\\times10^{-3}"""
    if x == 0 or not np.isfinite(x):
        return f"{x:g}"
    e = int(np.floor(np.log10(abs(x))))
    return rf"{x / 10 ** e:.1f}\times10^{{{e}}}"


def draw_pn(ax, P_n, sector, sc, M):
    n = np.arange(P_n.size)
    sel = P_n > 0
    ax.bar(n[sel], P_n[sel], width=1.6, color=SECTOR_COLOR[sector], linewidth=0)
    ax.axvspan(sc["tail_start"], M, color=MUTED, alpha=0.12, linewidth=0)
    ax.axvline(M, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlim(-1, M + 2)
    ax.set_xlabel(r"photon number $n$")
    ax.set_ylabel(r"$P(n)$")
    txt = (rf"$\langle n\rangle/N = {sc['n_over_N']:.3f} \pm {sc['n_over_N_se']:.3f}$" + "\n"
           + rf"$P(n\geq{sc['tail_start']}) = {_sci(sc['tail_weight'])}$")
    if not sc["truncation_ok"]:
        txt += "\n" + r"$\bf{warning:}$ truncation, increase $M$"
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color=INK, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
    ax.text(M, ax.get_ylim()[1] * 0.02, r" $M$", color=MUTED, ha="left", va="bottom", fontsize=9)


def plot_point(point_dir: Path):
    """wigner_pn.png/.pdf: Wigner function (top) and P(n) (bottom) per sector."""
    point_dir = Path(point_dir)
    rep, data = _load(point_dir)
    key = rep["physics_key"]
    sectors = [s for s in SECTORS if s in data]
    fig, axes = plt.subplots(2, len(sectors), figsize=(4.6 * len(sectors), 8.4),
                             squeeze=False, constrained_layout=True)
    letters = iter("abcdefgh")
    for j, s in enumerate(sectors):
        d, sc = data[s], rep["sectors"][s]
        ax = axes[0, j]
        draw_wigner(ax, d["xvec"], d["W"], fig=fig)
        ax.set_title(SECTOR_LABEL[s])
        _panel_label(ax, next(letters))
    for j, s in enumerate(sectors):
        ax = axes[1, j]
        draw_pn(ax, data[s]["P_n"], s, rep["sectors"][s], key["M"])
        _panel_label(ax, next(letters))
    fig.suptitle("Two-photon Dicke model, two-photon loss only\n" + _title(key, rep), fontsize=11)
    for ext in ("png", "pdf"):
        fig.savefig(point_dir / f"wigner_pn.{ext}")
    plt.close(fig)


def plot_diagnostics(point_dir: Path):
    """diagnostics.png: is t_max long enough, is M large enough, and where did the
    individual trajectories end up (vacuum basin vs. superradiant lobes, lobe angle)?"""
    point_dir = Path(point_dir)
    rep, data = _load(point_dir)
    key = rep["physics_key"]
    N, M = key["N"], key["M"]
    sectors = [s for s in SECTORS if s in data]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
    (a1, a2, a3), (a4, a5, a6) = axes

    n_max = max(float(np.max(data[s]["final_n"])) for s in sectors) / N
    bins = np.linspace(0, max(n_max, 1e-3) * 1.02, 41)
    for s in sectors:
        d, c = data[s], SECTOR_COLOR[s]
        t = d["tlist"]
        a1.plot(t, d["n_t_mean"] / N, color=c, lw=2, label=SECTOR_LABEL[s])
        a1.fill_between(t, (d["n_t_mean"] - 2 * d["n_t_se"]) / N,
                        (d["n_t_mean"] + 2 * d["n_t_se"]) / N, color=c, alpha=0.2, lw=0)
        a2.plot(t, d["jz_t_mean"] / N, color=c, lw=2, label=SECTOR_LABEL[s])
        a2.fill_between(t, (d["jz_t_mean"] - 2 * d["jz_t_se"]) / N,
                        (d["jz_t_mean"] + 2 * d["jz_t_se"]) / N, color=c, alpha=0.2, lw=0)
        P = d["P_n"]
        sel = P > 0
        a3.semilogy(np.arange(M)[sel], P[sel], color=c, lw=2, marker="o", ms=2.5,
                    label=SECTOR_LABEL[s])
        a4.hist(d["final_n"] / N, bins=bins, color=c, alpha=0.55, label=SECTOR_LABEL[s])
        z = d["final_a4"] / N ** 2
        a5.scatter(z.real, z.imag, s=10, color=c, alpha=0.45, linewidths=0, label=SECTOR_LABEL[s])
        a6.scatter(d["init_n"] / N, d["final_n"] / N, s=10, color=c, alpha=0.45, linewidths=0,
                   label=SECTOR_LABEL[s])

    a1.set(xlabel=r"$t$", ylabel=r"$\langle a^\dagger a\rangle(t)/N$",
           title=r"relaxation of $\langle n\rangle$ (band: $\pm 2$ s.e.)")
    a2.set(xlabel=r"$t$", ylabel=r"$\langle J_z\rangle(t)/N$", title="spin polarisation")
    a3.axvline(M, color=MUTED, linestyle="--", lw=1)
    a3.axvspan(int(np.ceil(0.9 * M)), M, color=MUTED, alpha=0.12, lw=0)
    floor = min(float(data[s]["P_n"][data[s]["P_n"] > 0].min()) for s in sectors)
    a3.set(xlabel=r"photon number $n$", ylabel=r"$P(n)$", title=r"$P(n)$, log scale (shaded: top 10 % of $M$)",
           xlim=(-1, M + 2), ylim=(max(floor, 1e-14) / 3, 1.5))
    a4.set(xlabel=r"per-trajectory $\langle n\rangle/N$ at $t_{\max}$", ylabel="trajectories",
           title="where trajectories end (vacuum vs. lobes)")
    lim = max(1e-6, max(float(np.max(np.abs(data[s]["final_a4"]))) for s in sectors) / N ** 2) * 1.08
    a5.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel=r"Re$\langle a^4\rangle/N^2$",
           ylabel=r"Im$\langle a^4\rangle/N^2$",
           title=r"per-trajectory $\langle a^4\rangle$: phase $=4\theta_{\rm lobe}$")
    a5.set_aspect("equal", adjustable="box")
    for ax in (a5,):
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.axvline(0, color=MUTED, lw=0.6)
    a6.set(xlabel=r"initial photon number $/N$", ylabel=r"final $\langle n\rangle/N$",
           title="initial state vs. outcome")
    for ax, lab in zip(axes.ravel(), "abcdef"):
        ax.legend(frameon=False, fontsize=8, loc="best")
        _panel_label(ax, lab)
    lines = [f"{s}: drift over last 20% of t = {rep['sectors'][s]['relax_drift_sigma']:.1f} s.e., "
             f"tail weight = {rep['sectors'][s]['tail_weight']:.1e}" for s in sectors]
    fig.suptitle("Diagnostics — " + _title(key, rep) + "\n" + " | ".join(lines), fontsize=10)
    fig.savefig(point_dir / "diagnostics.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# sweep-level figures
# ---------------------------------------------------------------------------

def plot_wigner_grid(point_dirs: list[Path], out: Path, x_max: float | None = None,
                     n_grid: int = 201, title: str | None = None):
    """Rows: parameter points (e.g. lambda values); columns: even / odd sector.
    Same layout as the 3x2 thesis figure in the original notebook."""
    from .analysis import compute_wigner, load_summary
    from .model import embed_cavity_rho

    rows = len(point_dirs)
    fig, axes = plt.subplots(rows, 2, figsize=(8.5, 3.6 * rows + 0.6), squeeze=False,
                             constrained_layout=True)
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for i, pd in enumerate(point_dirs):
        for j, s in enumerate(SECTORS):
            ax = axes[i, j]
            d = load_summary(pd, s)
            if x_max is None:
                xvec, W = d["xvec"], d["W"]
            else:
                rho = embed_cavity_rho(d["rho_sector"], d["fock"], int(d["M"]))
                xvec, W = compute_wigner(rho, x_max, n_grid)
            draw_wigner(ax, xvec, W, fig=fig)
            t = rf"$\lambda = {d['physics_key']['lam']:g}$  ($M={d['physics_key']['M']}$)"
            if i == 0:
                t = f"{s.upper()} SECTOR\n" + t
            ax.set_title(t, fontweight="bold")
            _panel_label(ax, next(letters))
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_n_vs_lambda(system_dirs: list[Path], out: Path, lam_c: float | None = None):
    """<n>/N vs lambda for every system (N, M) directory and both sectors.
    Open markers: truncation or relaxation warning at that point."""
    from .analysis import find_point_dirs

    fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for i, sd in enumerate(system_dirs):
        rows = []
        for pd in find_point_dirs(sd):
            f = pd / "summary.json"
            if f.exists():
                rows.append(json.loads(f.read_text()))
        if not rows:
            continue
        by_lam = {}       # several cutoffs at one lambda (convergence checks): keep the largest M
        for r in rows:
            lam_r = round(r["physics_key"]["lam"], 6)
            if lam_r not in by_lam or r["physics_key"]["M"] > by_lam[lam_r]["physics_key"]["M"]:
                by_lam[lam_r] = r
        rows = [by_lam[x] for x in sorted(by_lam)]
        key = rows[0]["physics_key"]
        for s in SECTORS:
            pts = [(r["physics_key"]["lam"], r["sectors"][s]) for r in rows if s in r["sectors"]]
            if not pts:
                continue
            lam = np.array([p[0] for p in pts])
            y = np.array([p[1]["n_over_N"] for p in pts])
            e = np.array([p[1]["n_over_N_se"] for p in pts])
            ok = np.array([p[1]["truncation_ok"] and p[1]["relaxation_ok"] for p in pts])
            c = SECTOR_COLOR[s]
            m = markers[i % len(markers)]
            ax.plot(lam, y, color=c, lw=1.5, alpha=0.8,
                    label=rf"$N={key['N']}$, {s}")
            ax.errorbar(lam[ok], y[ok], yerr=2 * e[ok], fmt=m, color=c, ms=6, capsize=2, lw=1)
            if (~ok).any():
                ax.errorbar(lam[~ok], y[~ok], yerr=2 * e[~ok], fmt=m, mfc="white", mec=c,
                            color=c, ms=6, capsize=2, lw=1)
    if lam_c is not None:
        ax.axvline(lam_c, color=MUTED, linestyle="--", lw=1)
        ax.text(lam_c, ax.get_ylim()[1], r" $\lambda_c$", color=MUTED, va="top")
    ax.set(xlabel=r"$\lambda/\omega_c$", ylabel=r"$\langle a^\dagger a\rangle/N$",
           title=r"photon number at $t_{\max}$ (error bars: $\pm 2$ s.e.; open: warning)")
    ax.legend(frameon=False, fontsize=8)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
