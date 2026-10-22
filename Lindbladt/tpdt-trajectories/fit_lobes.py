#!/usr/bin/env python
"""
Superradiant lobe analysis: separate the vacuum part of P(n) from the
superradiant lobe and measure the lobe's photon number, e.g. to compare with
mean-field / cumulant predictions.

    python fit_lobes.py results/thesis/N10_* results/thesis/N15_* results/thesis/N20_* \
        --out figures/lobe_fit --mf mf.csv

For every lambda point and sector:
  split   n_split = minimum of P(n) between the vacuum peak and the lobe
  weight  w = P(n > n_split): probability of the superradiant phase at t_max
  moment  <n>_lobe = sum_{n > n_split} n P(n) / w
  gauss   centre mu and width sigma of a Gaussian fitted to the lobe core
          (points with P > 5 % of the lobe maximum)
  traj    mean of the per-trajectory <n>(t_max) over the trajectories that
          ended above n_split, with its statistical standard error
If the lobe overlaps the vacuum in P(n) (typical close to the transition), the split
is taken from the per-trajectory <n>(t_max) instead, which separates the two basins
much more sharply; those points get the trajectory mean but no Gaussian centre.
Points without any separated lobe are listed with the reason; --dip / --nmin relax
the detection.

Writes  <out>.csv               all numbers (per N, lambda, sector)
        <out>.pdf / .png        lobe photon number per spin and lobe weight vs lambda
        <out>_fits_N<N>.png     P(n) with split and Gaussian, to check every fit by eye
--mf    optional CSV with columns  lam,n_over_N  (your mean-field / cumulant curve),
        drawn as a dashed line in the first panel.
Needs only numpy, scipy and matplotlib (no qutip), so it also runs on a laptop.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

SECTORS = {"even": 0, "odd": 1}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]  # one per N, fixed order
INK, INK2 = "#0b0b0b", "#52514e"


def gauss(n, A, mu, s):
    return A * np.exp(-0.5 * ((n - mu) / s) ** 2)


def _find_lobe(x, y, N, dip, nmin, min_weight):
    """(peak index, dip index) of the lobe in a distribution y(x), or (None, best ratio).
    Lobe = highest local maximum at x >= nmin*N with a dip at least `dip` times lower
    before it; the vacuum part only decreases, so it never produces such a maximum."""
    ys = np.convolve(y, np.ones(3) / 3, mode="same")
    cands, best = [], 0.0
    for i in range(1, ys.size - 1):
        if ys[i] >= ys[i - 1] and ys[i] >= ys[i + 1] and x[i] >= nmin * N and ys[i] > 0:
            j = int(np.argmin(ys[:i]))
            if y[j + 1:].sum() <= min_weight * y.sum():
                continue                                    # a few stray points, not a lobe
            ratio = ys[i] / max(ys[j], 1e-300)
            best = max(best, min(ratio, 1e9))
            if ratio > dip:
                cands.append((ys[i], i, j))
    if not cands:
        return None, best
    return max(cands)[1:], best


def analyse(P, parity, N, final_n, dip=3.0, nmin=0.5):
    """Lobe quantities for one sector. method = 'P(n)' (Gaussian fit), 'trajectories'
    (split only from the per-trajectory <n>, when lobe and vacuum overlap in P(n)) or None."""
    n = np.arange(parity, P.size, 2)
    p = np.clip(P[n], 0.0, None)
    found, r_p = _find_lobe(n, p, N, dip, nmin, 1e-3)
    if found is not None:
        ipk, isp = found
        nl, pl = n[isp + 1:], p[isp + 1:]
        w = pl.sum()
        mean = (nl * pl).sum() / w
        std = np.sqrt(((nl - mean) ** 2 * pl).sum() / w)
        core = pl > 0.05 * pl.max()
        popt, pcov = curve_fit(gauss, nl[core], pl[core], p0=[pl.max(), mean, std], maxfev=20000)
        split = n[isp]
        res = {"method": "P(n)", "n_split": float(split), "weight": float(w),
               "mean": float(mean), "std": float(std), "mu": float(popt[1]),
               "mu_err": float(np.sqrt(abs(pcov[1, 1]))), "sigma": float(abs(popt[2])),
               "_fit": popt}
    else:
        # lobe and vacuum overlap in P(n): try the per-trajectory <n>(t_max), which is far
        # sharper because each trajectory sits in one basin
        edges = np.linspace(0.0, final_n.max() * 1.0001 + 1e-9, 41)
        counts, _ = np.histogram(final_n, edges)
        centres = 0.5 * (edges[1:] + edges[:-1])
        found_t, r_t = _find_lobe(centres, counts.astype(float), N, dip, nmin, 0.05)
        if found_t is None:
            desc = lambda r: "no lobe peak" if r == 0 else f"peak/dip only {r:.1f}"
            return {"method": None, "reason": f"no separated lobe (P(n): {desc(r_p)}; "
                    f"trajectories: {desc(r_t)}; --dip is {dip:g})"}
        split = edges[found_t[1] + 1]
        res = {"method": "trajectories", "n_split": float(split),
               "weight": float((final_n > split).mean()), "_fit": None}
    sel = final_n > res["n_split"]
    k = int(sel.sum())
    res.update({"traj_mean": float(final_n[sel].mean()) if k else np.nan,
                "traj_se": float(final_n[sel].std(ddof=1) / np.sqrt(k)) if k > 1 else np.nan,
                "traj_frac": float(sel.mean()), "n_traj_lobe": k})
    return res


def mf_branches(lam, wc, wa, k2):
    """Cumulant / mean-field steady state of the thesis (Sec. 4.6, Eq. 4.54):
    (16 l^2 wc^2 k2^2 + k2^4 wa^2) x^2 + (16 l^2 wc^4 + 2 wa^2 k2^2 wc^2 - 64 l^4 wc^2) x
      + wa^2 wc^4 = 0,  x = (<n>/N)^2.  Returns (stable, unstable) <n>/N, NaN where absent."""
    lam = np.asarray(lam, float)
    A = 16 * lam**2 * wc**2 * k2**2 + k2**4 * wa**2
    B = 16 * lam**2 * wc**4 + 2 * wa**2 * k2**2 * wc**2 - 64 * lam**4 * wc**2
    C = wa**2 * wc**4
    disc = B**2 - 4 * A * C
    disc = np.where(np.abs(disc) < 1e-12 * B**2, 0.0, disc)
    with np.errstate(invalid="ignore"):
        xp = (-B + np.sqrt(disc)) / (2 * A)
        xm = (-B - np.sqrt(disc)) / (2 * A)
        return np.sqrt(np.where(xp > 0, xp, np.nan)), np.sqrt(np.where(xm > 0, xm, np.nan))


def lambda_c(wc, wa, k2):
    """Eq. 4.55 of the thesis."""
    return 0.5 * np.sqrt(wc**2 + wa * k2)


def load_points(system_dir):
    """{lam: point_dir} for one system folder; largest cutoff wins if a lambda was run twice."""
    best = {}
    for d in sorted(Path(system_dir).glob("lam*_M*")):
        f = d / "summary_even.npz"
        if not f.exists():
            f = d / "summary_odd.npz"
        if not f.exists():
            continue
        with np.load(f) as z:
            lam, M = round(float(z["lam"]), 6), int(z["M"])
        if lam not in best or M > best[lam][1]:
            best[lam] = (d, M)
    return {lam: d for lam, (d, _) in sorted(best.items())}


def _combine(vals, errs, wts):
    vals, errs, wts = map(np.asarray, (vals, errs, wts))
    w = wts / wts.sum()
    return float((w * vals).sum()), float(np.sqrt(((w * errs) ** 2).sum()))


def plot_mf_comparison(rows, params, out):
    if len(params) != 1:
        print(f"warning: data with different (wc, wa, kappa2) {sorted(params)}; mean field uses the first")
    wc, wa, k2 = sorted(params)[0]
    lc = lambda_c(wc, wa, k2)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    lam_all = sorted({r["lam"] for r in rows})
    lo, hi = min(lam_all), max(lam_all)
    pad = 0.02 * (hi - lo)

    grid = np.linspace(lc, hi + pad, 600)
    st, un = mf_branches(grid, wc, wa, k2)
    ax.plot([lo - pad, hi + pad], [0, 0], color=INK2, lw=1.0, zorder=1, label="mean field: normal phase")
    ax.plot(grid, st, color=INK, lw=2.0, zorder=2, label="mean field: stable branch")
    ax.plot(grid, un, color=INK2, lw=1.5, ls="--", zorder=2, label="mean field: unstable branch")
    ax.axvline(lc, color=INK2, lw=1.0, ls=":", zorder=1)
    ax.text(lc, 1.0, rf" $\lambda_c = {lc:.3f}$", transform=ax.get_xaxis_transform(),
            va="top", ha="left", fontsize=9, color=INK2)

    for c, N in zip(COLORS, sorted({r["N"] for r in rows})):
        xs, ys, es = [], [], []
        first = min((r["lam"] for r in rows if r["N"] == N and r["method"] != "none"), default=np.inf)
        for lam in sorted({r["lam"] for r in rows if r["N"] == N}):
            rs = [r for r in rows if r["N"] == N and r["lam"] == lam]
            lob = [r for r in rs if r["method"] != "none"]
            if lam < first:
                v, e = _combine([r["all_mean_over_N"] for r in rs], [r["all_se_over_N"] for r in rs],
                                [1.0] * len(rs))
            elif lob:
                v, e = _combine([r["mu_over_N"] if r["method"] == "P(n)" else r["traj_mean_over_N"]
                                 for r in lob], [r["traj_se_over_N"] for r in lob],
                                [r["n_traj_lobe"] for r in lob])
            else:
                continue
            xs.append(lam); ys.append(v); es.append(e)
        ax.errorbar(xs, ys, yerr=es, fmt="o", ms=5.5, color=c, mec="white", mew=0.8,
                    capsize=0, elinewidth=1.2, zorder=3, label=f"trajectories, N = {N}")

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\langle \hat a^\dagger \hat a\rangle / N$")
    ax.grid(alpha=0.2, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("system_dirs", nargs="+")
    ap.add_argument("--out", default="figures/lobe_fit")
    ap.add_argument("--mf", default=None, help="CSV with columns lam,n_over_N")
    ap.add_argument("--dip", type=float, default=3.0,
                    help="lobe peak must be this many times higher than the dip before it (default 3)")
    ap.add_argument("--nmin", type=float, default=0.5,
                    help="lobe peak must lie above n = nmin*N (default 0.5)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, by_N, curves, params = [], {}, {}, set()
    for sd in args.system_dirs:
        for lam, d in load_points(sd).items():
            for sec, parity in SECTORS.items():
                f = d / f"summary_{sec}.npz"
                if not f.exists():
                    continue
                with np.load(f) as z:
                    P, N, M, final_n = z["P_n"], int(z["N"]), int(z["M"]), z["final_n"]
                key = json.loads((d / "summary.json").read_text())["physics_key"]
                params.add((key["wc"], key["wa"], key["kappa2"]))
                r = analyse(P, parity, N, final_n, dip=args.dip, nmin=args.nmin)
                curves.setdefault(N, {}).setdefault(lam, {})[sec] = (P, parity, r)
                row = {"N": N, "lam": lam, "M": M, "sector": sec, "method": r["method"] or "none"}
                # vacuum basin = trajectories below the split (all of them if there is no lobe)
                v = final_n[final_n <= r["n_split"]] if r["method"] else final_n
                row.update({"all_mean_over_N": final_n.mean() / N,
                            "all_se_over_N": final_n.std(ddof=1) / np.sqrt(final_n.size) / N,
                            "vac_frac": v.size / final_n.size,
                            "vac_mean_over_N": v.mean() / N if v.size else np.nan,
                            "vac_se_over_N": v.std(ddof=1) / np.sqrt(v.size) / N if v.size > 1 else np.nan})
                if r["method"] is None:
                    row["reason"] = r["reason"]
                else:
                    row.update({"n_split": r["n_split"], "weight": r["weight"],
                                "traj_mean_over_N": r["traj_mean"] / N, "traj_se_over_N": r["traj_se"] / N,
                                "traj_frac": r["traj_frac"], "n_traj_lobe": r["n_traj_lobe"]})
                    if r["method"] == "P(n)":
                        row.update({"mu_over_N": r["mu"] / N, "mu_err_over_N": r["mu_err"] / N,
                                    "sigma_over_N": r["sigma"] / N,
                                    "moment_over_N": r["mean"] / N, "std_over_N": r["std"] / N})
                    by_N.setdefault(N, []).append(row)
                rows.append(row)

    if not rows:
        raise SystemExit("no finalized points found")
    keys = ["N", "lam", "M", "sector", "method", "n_split", "weight", "mu_over_N", "mu_err_over_N",
            "sigma_over_N", "moment_over_N", "std_over_N", "traj_mean_over_N", "traj_se_over_N",
            "traj_frac", "n_traj_lobe", "vac_frac", "vac_mean_over_N", "vac_se_over_N",
            "all_mean_over_N", "all_se_over_N", "reason"]
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, restval="")
        wr.writeheader()
        for row in rows:
            wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})

    # --- summary table on screen -------------------------------------------------
    print(f"{'N':>3} {'lam':>6} {'sec':>4} | {'gauss mu/N':>16} | {'traj mean/N':>16} | {'weight':>6} | method")
    for row in rows:
        head = f"{row['N']:3d} {row['lam']:6.3f} {row['sector']:>4} | "
        if row["method"] == "none":
            print(head + row["reason"])
            continue
        g = (f"{row['mu_over_N']:7.3f} +- {row['mu_err_over_N']:.3f}" if "mu_over_N" in row
             else f"{'-':>16}")
        print(head + f"{g} | {row['traj_mean_over_N']:7.3f} +- {row['traj_se_over_N']:.3f} | "
              f"{row['weight']:6.3f} | {row['method']}")

    # --- lobe photon number and weight vs lambda ---------------------------------
    # circles: Gaussian centre from P(n); squares: mean over lobe trajectories, used where
    # the lobe overlaps the vacuum in P(n); filled = even, open = odd
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    any_traj = False
    for c, N in zip(COLORS, sorted(by_N)):
        for sec, face in (("even", c), ("odd", "none")):
            rs = sorted((r for r in by_N[N] if r["sector"] == sec), key=lambda r: r["lam"])
            g = [r for r in rs if r["method"] == "P(n)"]
            t = [r for r in rs if r["method"] == "trajectories"]
            a1.errorbar([r["lam"] for r in g], [r["mu_over_N"] for r in g],
                        yerr=[r["mu_err_over_N"] for r in g], fmt="o", ms=6, mfc=face, mec=c,
                        color=c, lw=1.2, capsize=2, label=f"N = {N}, {sec}")
            if t:
                any_traj = True
                a1.errorbar([r["lam"] for r in t], [r["traj_mean_over_N"] for r in t],
                            yerr=[r["traj_se_over_N"] for r in t], fmt="s", ms=6, mfc=face,
                            mec=c, color=c, lw=1.2, capsize=2)
            a2.plot([r["lam"] for r in rs], [r["weight"] for r in rs], "o-", ms=6, mfc=face,
                    mec=c, color=c, lw=1.5, label=f"N = {N}, {sec}")
    if any_traj:
        a1.plot([], [], "s", ms=6, mfc="none", mec=INK2, label="trajectory mean (lobe overlaps vacuum)")
    if args.mf and not Path(args.mf).exists():
        print(f"warning: {args.mf} not found, plotting without the mean-field curve")
    elif args.mf:
        mf = np.genfromtxt(args.mf, delimiter=",", names=True)
        a1.plot(mf["lam"], mf["n_over_N"], "--", color=INK2, lw=1.8, label="mean field / cumulant")
    a1.set_xlabel(r"$\lambda$"); a1.set_ylabel(r"lobe photon number per spin $\mu/N$")
    a1.set_title("photon number of the superradiant lobe", fontsize=11, color=INK)
    a2.set_xlabel(r"$\lambda$"); a2.set_ylabel(r"lobe weight")
    a2.set_ylim(0, 1.05)
    a2.set_title(r"population of the superradiant phase at $t_{\max}$", fontsize=11, color=INK)
    for a in (a1, a2):
        a.grid(alpha=0.25, lw=0.6)
        a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)

    # --- transition: vacuum branch and superradiant lobe vs lambda ---------------
    # triangles: mean <n>/N of the trajectories in the vacuum basin (all trajectories where
    # there is no lobe); circles: Gaussian centre of the lobe; squares: mean of the lobe
    # trajectories where the lobe overlaps the vacuum in P(n). A branch is drawn where it
    # holds at least 5 % of the trajectories. filled = even, open = odd.
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for c, N in zip(COLORS, sorted({r["N"] for r in rows})):
        for sec, face in (("even", c), ("odd", "none")):
            rs = sorted((r for r in rows if r["N"] == N and r["sector"] == sec), key=lambda r: r["lam"])
            vac = [r for r in rs if r["vac_frac"] >= 0.05]
            ax.errorbar([r["lam"] for r in vac], [r["vac_mean_over_N"] for r in vac],
                        yerr=[r["vac_se_over_N"] for r in vac], fmt="v-", ms=6, mfc=face, mec=c,
                        color=c, lw=1, alpha=0.9, capsize=2)
            lob = [r for r in rs if r["method"] in ("P(n)", "trajectories")]
            ax.plot([r["lam"] for r in lob],
                    [r["mu_over_N"] if r["method"] == "P(n)" else r["traj_mean_over_N"] for r in lob],
                    "-", color=c, lw=1, alpha=0.9)
            for meth, mk, key, ekey in (("P(n)", "o", "mu_over_N", "mu_err_over_N"),
                                        ("trajectories", "s", "traj_mean_over_N", "traj_se_over_N")):
                pts = [r for r in lob if r["method"] == meth]
                ax.errorbar([r["lam"] for r in pts], [r[key] for r in pts],
                            yerr=[r[ekey] for r in pts], fmt=mk, ms=6, mfc=face, mec=c,
                            color=c, capsize=2)
    if args.mf and Path(args.mf).exists():
        ax.plot(mf["lam"], mf["n_over_N"], "--", color=INK2, lw=1.8)
    handles = [Line2D([], [], color=c, lw=2, label=f"N = {N}")
               for c, N in zip(COLORS, sorted({r["N"] for r in rows}))]
    handles += [Line2D([], [], ls="", marker="o", color=INK2, label="lobe: Gaussian centre"),
                Line2D([], [], ls="", marker="s", color=INK2, label="lobe: trajectory mean (overlaps vacuum)"),
                Line2D([], [], ls="", marker="v", color=INK2, label="vacuum basin: trajectory mean"),
                Line2D([], [], ls="", marker="o", mfc="none", color=INK2, label="open = odd, filled = even")]
    if args.mf and Path(args.mf).exists():
        handles.append(Line2D([], [], ls="--", color=INK2, lw=1.8, label="mean field / cumulant"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"$\langle n\rangle / N$ at $t_{\max}$")
    ax.set_title("superradiant transition: vacuum branch and lobe", fontsize=11, color=INK)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(f"{out}_transition.pdf")
    fig.savefig(f"{out}_transition.png", dpi=150)
    plt.close(fig)

    # --- clean comparison with mean field ---------------------------------------
    # one point per N and lambda, even and odd combined: below the transition the mean
    # <n>/N of all trajectories, from the first lambda with a superradiant lobe on the lobe
    # position (Gaussian centre; mean of the lobe trajectories where the lobe still overlaps
    # the vacuum). Error bars: statistical standard error from the trajectories.
    plot_mf_comparison(rows, params, Path(f"{out}_mf"))

    # --- check figures: full P(n) (log), split and Gaussian for every lambda -----
    for N, per_lam in curves.items():
        lams = sorted(per_lam)
        ncol = 4
        nrow = int(np.ceil(len(lams) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow), squeeze=False)
        for ax, lam in zip(axes.flat, lams):
            tags = []
            for sec, col in (("even", COLORS[0]), ("odd", COLORS[1])):
                if sec not in per_lam[lam]:
                    continue
                P, parity, r = per_lam[lam][sec]
                n = np.arange(parity, P.size, 2)
                m = P[n] > 0
                ax.semilogy(n[m], P[n][m], ".", ms=2.5, color=col, label=sec)
                if r["method"] is not None:
                    ax.axvline(r["n_split"], color=col, ls=":", lw=1)
                if r.get("_fit") is not None:
                    x = np.linspace(r["n_split"], n.max(), 400)
                    yv = gauss(x, *r["_fit"])
                    ax.semilogy(x[yv > 1e-12], yv[yv > 1e-12], "-", lw=1.2, color=col)
                tags.append({"P(n)": "fit", "trajectories": "traj", None: "none"}[r["method"]])
            ax.set_ylim(1e-8, 1)
            ax.set_title(rf"$\lambda$ = {lam:g}  ({'/'.join(tags)})", fontsize=9)
            ax.tick_params(labelsize=7)
        for ax in list(axes.flat)[len(lams):]:
            ax.axis("off")
        axes[0, 0].legend(frameon=False, fontsize=7)
        fig.suptitle(f"N = {N}: P(n) (dots), split (dotted), Gaussian fit (lines); "
                     "title: method even/odd", fontsize=10)
        fig.tight_layout()
        fig.savefig(f"{out}_fits_N{N}.png", dpi=130)
        plt.close(fig)

    print(f"wrote {out}_mf.pdf")
    print(f"\nwrote {out.with_suffix('.csv')}, {out.with_suffix('.pdf')}, {out}_transition.pdf, "
          f"{out}_fits_N*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())