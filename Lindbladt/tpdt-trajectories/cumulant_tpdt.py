#!/usr/bin/env python
"""
Second-order cumulant theory of the two-photon Dicke model with two-photon loss only
(H = wc a+a + wa/2 Jz + lam/N Jx (a+^2 + a^2),  L = sqrt(k2/N) a^2,  J = 2 jmat),
in the form of Shah et al.: photon moments <a>, <a^2>, <a+a> to second order
(Gaussian closure), spins at mean field. All variables are intensive:
    al = <a>/sqrt(N),  A = <a^2>/N,  n = <a+a>/N,  j = <J>/N  (|j| = 1).

Two ways to close the equations:
  * <a> free ("symmetry broken"): each superradiant lobe is a displaced Gaussian.
    In the thermodynamic limit this reduces exactly to the thesis equations
    4.35-4.37 and the quartic 4.54 (stable + unstable branch from lambda_c).
    At finite N the 1/N terms (vacuum fluctuations) add the leading finite-size
    correction to the lobe position and predict the lobe width.
  * <a> = 0 imposed ("symmetric Gaussian"): the four-lobe state is replaced by one
    Gaussian centred at the origin, which triples the two-photon loss
    (<a+^2 a^2> = 3n^2 instead of n^2). Gives a second branch with n ~ n_SB/3 that
    appears only at lambda_G = sqrt(wc^2 + 3 wa k2)/2.

The equations were checked against the exact Lindbladian (QuTiP): for a product of
a Gaussian photon state and a spin coherent state they agree to machine precision.

    python cumulant_tpdt.py --csv figures/lobe_fit.csv --out figures/cumulant_vs_data
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import root

WC, WA, K2 = 1.0, 1.0, 0.05
INK, INK2, BLUE = "#0b0b0b", "#52514e", "#2a78d6"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def rhs(y, lam, N=np.inf, wc=WC, wa=WA, k2=K2):
    """d/dt of y = [Re al, Im al, Re A, Im A, n, jx, jy, jz]."""
    al, A, n, jx, jy, jz = y[0] + 1j * y[1], y[2] + 1j * y[3], y[4], y[5], y[6], y[7]
    iN = 0.0 if np.isinf(N) else 1.0 / N
    dal = -1j * wc * al - 2j * lam * jx * np.conj(al) - k2 * (2 * n * al + A * np.conj(al) - 2 * abs(al) ** 2 * al)
    dA = -2j * wc * A - 4j * lam * jx * (n + 0.5 * iN) - k2 * (6 * n * A - 4 * abs(al) ** 2 * al ** 2 + A * iN)
    dn = -4 * lam * jx * A.imag - 2 * k2 * (abs(A) ** 2 + 2 * n ** 2 - 2 * abs(al) ** 4)
    return np.array([dal.real, dal.imag, dA.real, dA.imag, dn,
                     -wa * jy, wa * jx - 4 * lam * jz * A.real, 4 * lam * jy * A.real])


def growth(y, lam, N):
    """Largest real part of the linearised dynamics (spin-length mode removed)."""
    h = 1e-7
    J = np.column_stack([(rhs(y + h * e, lam, N) - rhs(y - h * e, lam, N)) / (2 * h) for e in np.eye(8)])
    e = np.zeros(8); e[5:] = y[5:] / np.linalg.norm(y[5:])
    Q, _ = np.linalg.qr(np.column_stack([e, np.eye(8)]))
    T = Q[:, 1:8]
    return float(np.max(np.linalg.eigvals(T.T @ J @ T).real))


def lambda_c(wc=WC, wa=WA, k2=K2, symmetric=False):
    """Onset of the symmetry-broken branch (thesis Eq. 4.55) or of the <a>=0 branch."""
    return 0.5 * np.sqrt(wc ** 2 + (3 if symmetric else 1) * wa * k2)


def _td_guess(n, symmetric):
    """Thermodynamic-limit solution with photon number n (analytic), as start for Newton."""
    k = (3 if symmetric else 1) * K2
    c1 = (WC ** 2 + k ** 2 * n ** 2) / 4                 # jx^2 = c1 / lam^2
    c2 = (WA * c1 / (2 * WC * n)) ** 2                  # jz^2 = c2 / lam^4
    u = (-c1 + np.sqrt(c1 ** 2 + 4 * c2)) / (2 * c2)    # u = 1/lam^2 from jx^2 + jz^2 = 1
    lam = 1 / np.sqrt(u)
    jx = -np.sqrt(c1 * u); jz = -np.sqrt(max(1 - jx ** 2, 0.0))
    if symmetric:
        A = -2j * lam * jx * n / (1j * WC + 3 * K2 * n)
        return np.r_[A.real, A.imag, lam, jx, 0.0, jz]
    e2 = (-1j * WC - K2 * n) / (2j * lam * jx)          # exp(-2i theta)
    th = -0.5 * np.angle(e2)
    al = np.sqrt(n) * np.exp(1j * th)
    return np.r_[al.real, al.imag, (al ** 2).real, (al ** 2).imag, lam, jx, 0.0, jz]


def branch(N=np.inf, symmetric=False, n_grid=None):
    """Steady-state branch parametrised by n = <a+a>/N, through its fold:
    returns arrays lam, n, width (sigma_n/N of the Gaussian lobe), growth rate, solution."""
    if n_grid is None:
        top = 6.0 if symmetric else 16.0
        n_grid = np.concatenate([np.linspace(top, 1.5, 150), np.linspace(1.5, 0.2, 100)[1:]])

    def eqs(z, n):
        if symmetric:                                  # z = [Re A, Im A, lam, jx, jy, jz], al = 0
            y = np.r_[0.0, 0.0, z[0], z[1], n, z[3:6]]
            r = rhs(y, z[2], N)
            return np.r_[r[2:7], z[3] ** 2 + z[4] ** 2 + z[5] ** 2 - 1]
        y = np.r_[z[0:4], n, z[5:8]]                   # z = [Re al, Im al, Re A, Im A, lam, jx, jy, jz]
        r = rhs(y, z[4], N)
        return np.r_[r[0:7], z[5] ** 2 + z[6] ** 2 + z[7] ** 2 - 1]

    z = _td_guess(n_grid[0], symmetric)
    out = []
    for n in n_grid:
        s = root(eqs, z, args=(n,), method="hybr", tol=1e-13)
        if not s.success or np.max(abs(eqs(s.x, n))) > 1e-9:
            continue
        z = s.x
        if symmetric:
            y, lam = np.r_[0.0, 0.0, z[0], z[1], n, z[3:6]], z[2]
        else:
            y, lam = np.r_[z[0:4], n, z[5:8]], z[4]
        al, A = y[0] + 1j * y[1], y[2] + 1j * y[3]
        iN = 0.0 if np.isinf(N) else 1.0 / N
        width = np.sqrt(max(abs(A) ** 2 + n ** 2 - 2 * abs(al) ** 4 + n * iN, 0.0))
        out.append((lam, n, width, growth(y, lam, N)))
    return np.array(out)


def load_data(path):
    """Per (N, lambda): lobe centre, its error, lobe width, and the all-trajectory mean."""
    rows = list(csv.DictReader(open(path)))
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else np.nan
    data = {}
    for r in rows:
        d = data.setdefault(int(r["N"]), {}).setdefault(float(r["lam"]), {"lobe": [], "all": []})
        d["all"].append((f(r, "all_mean_over_N"), f(r, "all_se_over_N")))
        if r["method"] in ("P(n)", "trajectories"):
            mu = f(r, "mu_over_N") if r["method"] == "P(n)" else f(r, "traj_mean_over_N")
            d["lobe"].append((mu, f(r, "traj_se_over_N"), f(r, "sigma_over_N"), int(float(r["n_traj_lobe"]))))
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="figures/lobe_fit.csv", help="output of fit_lobes.py")
    ap.add_argument("--out", default="figures/cumulant_vs_data")
    ap.add_argument("--lam-max", type=float, default=None)
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_data(args.csv)
    Ns = sorted(data)
    lam_all = sorted({l for N in Ns for l in data[N]})
    lo, hi = min(lam_all) - 0.003, (args.lam_max or max(lam_all)) + 0.003

    TD, SYM = branch(np.inf), branch(np.inf, symmetric=True)
    fin = {N: branch(N) for N in Ns}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # table: theory at the measured lambdas + data
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "lam", "data_lobe_over_N", "data_err", "data_sigma_over_N",
                    "SB_TD_over_N", "SB_N_over_N", "SB_N_sigma_over_N", "sym_TD_over_N"])
        for N in Ns:
            for lam in sorted(data[N]):
                lob = data[N][lam]["lobe"]
                if not lob:
                    continue
                wts = np.array([x[3] for x in lob], float)
                mu = np.average([x[0] for x in lob], weights=wts)
                err = np.sqrt(np.sum((wts / wts.sum() * np.array([x[1] for x in lob])) ** 2))
                sigs = [x[2] for x in lob if not np.isnan(x[2])]
                sig = np.mean(sigs) if sigs else np.nan
                def on(B, stable=True):
                    m = (B[:, 3] < 1e-9) if stable else np.ones(len(B), bool)
                    Bs = B[m][np.argsort(B[m][:, 0])]
                    ok = (Bs[:, 0].min() <= lam <= Bs[:, 0].max())
                    return (np.interp(lam, Bs[:, 0], Bs[:, 1]), np.interp(lam, Bs[:, 0], Bs[:, 2])) if ok else (np.nan, np.nan)
                upTD = TD[TD[:, 1] >= TD[np.argmin(TD[:, 0]), 1]]
                upN = fin[N][fin[N][:, 1] >= fin[N][np.argmin(fin[N][:, 0]), 1]]
                upS = SYM[SYM[:, 1] >= SYM[np.argmin(SYM[:, 0]), 1]] if len(SYM) else SYM
                w.writerow([N, lam, f"{mu:.5f}", f"{err:.5f}", f"{sig:.5f}",
                            f"{on(upTD, False)[0]:.5f}", f"{on(upN, False)[0]:.5f}",
                            f"{on(upN, False)[1]:.5f}", f"{on(upS, False)[0]:.5f}"])

    # figure
    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], hspace=0.35, wspace=0.25)
    ax, axw, axd = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

    def draw(a, B, color, lw, label_s=None, label_u=None, ls_u="--", col=1, unstable=True):
        st = B[:, 3] < 1e-9
        for mask, ls, lab in ((st, "-", label_s), (~st, ls_u, label_u))[:2 if unstable else 1]:
            Bm = np.where(mask[:, None], B, np.nan)
            a.plot(Bm[:, 0], Bm[:, col], ls=ls, color=color, lw=lw, label=lab)

    ax.plot([lo, hi], [0, 0], color=INK2, lw=1.0)
    draw(ax, TD, INK, 2.0, r"$N\to\infty$, $\langle a\rangle\neq0$ (= Eq. 4.54): stable", "unstable")
    if len(SYM):
        ax.plot(SYM[:, 0], SYM[:, 1], ls=":", color=INK2, lw=1.8,
                label=r"$N\to\infty$, $\langle a\rangle = 0$ imposed")
    for c, N in zip(COLORS, Ns):
        draw(ax, fin[N], c, 1.6, rf"$N = {N}$, $\langle a\rangle\neq0$ with $1/N$ terms", unstable=False)
        lams = sorted(data[N])
        first = min([l for l in lams if data[N][l]["lobe"]], default=np.inf)
        xs, ys, es = [], [], []
        for l in lams:
            lob = data[N][l]["lobe"]
            if lob:
                wts = np.array([x[3] for x in lob], float)
                xs.append(l); ys.append(np.average([x[0] for x in lob], weights=wts))
                es.append(np.sqrt(np.sum((wts / wts.sum() * np.array([x[1] for x in lob])) ** 2)))
        ax.errorbar(xs, ys, yerr=es, fmt="o", ms=5.5, color=c, mec="white", mew=0.8, zorder=4,
                    label=f"trajectories N = {N}: lobe")
        below = [l for l in lams if l < first]
        ax.plot(below, [np.mean([v for v, _ in data[N][l]["all"]]) for l in below], "o", ms=5.5,
                mfc="white", mec=c, mew=1.2, zorder=4, label=f"trajectories N = {N}: below threshold" + r" ($t=t_{\max}$)")
        # width and deviation panels
        wx = [l for l in lams if any(not np.isnan(x[2]) for x in data[N][l]["lobe"])]
        axw.plot(wx, [np.nanmean([x[2] for x in data[N][l]["lobe"]]) for l in wx], "o", ms=5, color=c,
                 mec="white", mew=0.8, zorder=4, label=f"trajectories N = {N}")
        B = fin[N][fin[N][:, 3] < 1e-9]; B = B[np.argsort(B[:, 0])]
        axw.plot(B[:, 0], B[:, 2], color=c, lw=1.6, label=rf"$\langle a\rangle\neq0$ Gaussian, $N={N}$")
        axw.plot(B[:, 0], np.sqrt(B[:, 1] / N), color=INK2, lw=1.2, ls="--",
                 label=r"coherent state $\sqrt{\langle n\rangle}/N$" if N == Ns[0] else None)
        upTD = TD[TD[:, 3] < 1e-9]; upTD = upTD[np.argsort(upTD[:, 0])]
        gx = [l for l in xs if l >= B[:, 0].min()]
        gy = [ys[xs.index(l)] for l in gx]
        axd.plot(gx, 100 * (np.array(gy) / np.interp(gx, upTD[:, 0], upTD[:, 1]) - 1), "s-", ms=4.5,
                 color=INK2, lw=1, label=r"vs $N\to\infty$ (Eq. 4.54)" if N == Ns[0] else None)
        axd.plot(gx, 100 * (np.array(gy) / np.interp(gx, B[:, 0], B[:, 1]) - 1), "o-", ms=4.5, color=c,
                 lw=1, label=rf"vs $\langle a\rangle\neq0$, $N={N}$")
    lc = lambda_c()
    ax.axvline(lc, color=INK2, lw=1, ls=":")
    ax.text(lc, 0.99, rf" $\lambda_c={lc:.3f}$", transform=ax.get_xaxis_transform(), va="top", fontsize=9, color=INK2)
    ax.set_xlim(lo, hi); ax.set_ylim(-0.5, None)
    ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"$\langle a^\dagger a\rangle/N$")
    leg = ax.legend(frameon=True, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    leg.get_frame().set(facecolor="white", edgecolor="none", alpha=0.95)
    axw.set_xlim(lo, hi); axw.set_xlabel(r"$\lambda$"); axw.set_ylabel(r"lobe width $\sigma_n/N$")
    axw.legend(frameon=False, fontsize=7, loc="upper right")
    axd.axhline(0, color=INK2, lw=0.8)
    axd.set_xlim(lo, hi); axd.set_xlabel(r"$\lambda$"); axd.set_ylabel("data / theory − 1  [%]")
    axd.legend(frameon=False, fontsize=7, loc="lower right")
    for a in (ax, axw, axd):
        a.grid(alpha=0.2, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    for a, t in ((ax, "(a)"), (axw, "(b)"), (axd, "(c)")):
        a.text(0.0, 1.02, t, transform=a.transAxes, fontsize=10, fontweight="bold")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out.with_suffix('.pdf')}, {out.with_suffix('.png')}, {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()