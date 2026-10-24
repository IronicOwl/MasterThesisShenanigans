#!/usr/bin/env python3
"""
scaling_lmg.py -- finite-size scaling of the slowest mode per sector, from the output folders
of sweep_traceless_lmg.py (data/*.npz).  Light: runs in seconds on a login node.

    python scaling_lmg.py Gap_Closure_LMG Gap_Closure_LMG_S
    python scaling_lmg.py Gap_Closure_LMG Gap_Closure_LMG_S --points 0.1,0.47 0.7,0.47 0.7,0.8

For each (chi, lam) point it prints the slowest eigenvalue (smallest |lambda|) of every sector
and N, and in the lobed phase fits ln|lambda| against N (exponential, rate c per spin) and
against ln N (power law, exponent p).  Local log-log slopes d ln|lambda| / d ln N are printed
too: constant for a power law, growing ~ -c N for an exponential.
Writes scaling_lmg.png:
  (a) |lambda| vs N at the deepest point, semi-log (exponential = straight line)
  (b) the same on log-log axes (power law = straight line)
  (c) N * (-Re lambda) of the least-damped E_+-_i mode (the one behind the -Re dip) along the
      path of the first folder; flat in N means its damping vanishes like 1/N
  (d) |Im lambda| of that mode, with the mean-field even/odd splitting w' in the Q phase
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import sweep_traceless_lmg as S          # for the mean-field lines; optional
except Exception:
    S = None


def load(dirs):
    rows = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "data", "gap_*_N*.npz"))):
            z = np.load(f, allow_pickle=True)
            rows.setdefault((str(z["sector"]), int(z["N"])), []).append(dict(
                chi=z["chi_vals"], lam=z["lam_vals"], u=z["u_vals"], ev=z["all_evals"],
                path=np.asarray(z["path"], float), src=d,
                model=dict(wc=float(z["wc"]), wa=float(z["wa"]), kappa2=float(z["kappa2"]),
                           gamma=float(z["gamma"]))))
    if not rows:
        raise SystemExit("no data/*.npz found in " + ", ".join(dirs))
    return rows


def slowest(ev):
    e = ev[np.isfinite(ev)]
    return e[np.argmin(np.abs(e))] if e.size else np.nan


def least_damped(ev):
    e = ev[np.isfinite(ev)]
    return e[np.argmax(e.real)] if e.size else np.nan


def at_point(rows, sector, chi, lam, tol=2e-3):
    """{N: slowest eigenvalue} at (chi, lam), taken from the first folder that has the point."""
    out = {}
    for (s, N), recs in rows.items():
        if s != sector:
            continue
        for r in recs:
            d = np.hypot(r["chi"] - chi, r["lam"] - lam)
            i = int(np.argmin(d))
            if d[i] < tol:
                out[N] = slowest(r["ev"][i])
                break
    return dict(sorted(out.items()))


def fits(Ns, g):
    Ns, lg = np.asarray(Ns, float), np.log(np.asarray(g, float))
    pe, pp = np.polyfit(Ns, lg, 1), np.polyfit(np.log(Ns), lg, 1)
    rms = lambda p, x: float(np.sqrt(np.mean((np.polyval(p, x) - lg) ** 2)))
    return dict(c=-pe[0], rms_exp=rms(pe, Ns), p=-pp[0], rms_pow=rms(pp, np.log(Ns)),
                local=np.diff(lg) / np.diff(np.log(Ns)), pe=pe, pp=pp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--points", nargs="+",
                    default=["0.1,0.47", "0.7,0.47", "0.7,0.65", "0.7,0.7", "0.7,0.75", "0.7,0.8"],
                    help="(chi, lam) points; the last one is used for panels (a, b)")
    ap.add_argument("--out", default="scaling_lmg.png")
    a = ap.parse_args()
    rows = load(a.dirs)
    model = next(iter(rows.values()))[0]["model"]
    sectors = [s for s in ("E_++_-1", "E_--_-1", "E_+-_i", "E_+-_-i", "E_++_1", "E_--_1")
               if any(k[0] == s for k in rows)]
    pts = [tuple(float(x) for x in p.split(",")) for p in a.points]
    phase = {}
    for chi, lam in pts:
        st = S.mf_phase(chi, lam, model) if S else None
        phase[(chi, lam)] = ("?" if st is None else "S" if st["S"] and not st["Q"] else
                             "Q|S" if st["S"] else "Q" if st["Q"] else "N")
        print(f"\n=== (chi, lam) = ({chi:g}, {lam:g})   mean-field phase: {phase[(chi, lam)]}")
        for s in sectors:
            d = at_point(rows, s, chi, lam)
            if not d:
                continue
            print(f"  {s:8s} " + "  ".join(f"N={N}: {abs(e):.3e} ({e.real:+.2e}{e.imag:+.2e}i)"
                                           for N, e in d.items()))
            if phase[(chi, lam)] == "S" and s in ("E_++_-1", "E_--_-1", "E_+-_i", "E_+-_-i") and len(d) >= 3:
                f = fits(list(d), [abs(e) for e in d.values()])
                print(f"           exp: c = {f['c']:.3f}/spin (rms {f['rms_exp']:.3f})   "
                      f"power: p = {f['p']:.2f} (rms {f['rms_pow']:.3f})   local slopes "
                      + " ".join(f"{x:+.2f}" for x in f["local"]))
        if phase[(chi, lam)] == "S":
            c, w = at_point(rows, "E_+-_i", chi, lam), at_point(rows, "E_++_-1", chi, lam)
            print("  ratio |E_+-_i| / |E_++_-1|: " + "  ".join(
                f"N={N}: {abs(c[N]) / abs(w[N]):.2f}" for N in c if N in w))

    # ---------------------------------------------------------------- figure
    BLUE, ORANGE, INK2 = "#2a78d6", "#eb6834", "#52514e"
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.2))
    chi, lam = pts[-1]
    for s, col, ls in (("E_+-_i", BLUE, "-"), ("E_++_-1", ORANGE, "-"),
                       ("E_+-_-i", BLUE, ":"), ("E_--_-1", ORANGE, ":")):
        d = at_point(rows, s, chi, lam)
        if len(d) < 2:
            continue
        Ns, g = np.array(list(d), float), np.array([abs(e) for e in d.values()])
        f = fits(Ns, g)
        x = np.linspace(Ns.min(), Ns.max(), 100)
        ax[0, 0].semilogy(Ns, g, "o" + ls, color=col, label=f"{s}: c = {f['c']:.2f}")
        ax[0, 0].semilogy(x, np.exp(np.polyval(f["pe"], x)), color=col, lw=0.8, alpha=0.6)
        ax[0, 1].loglog(Ns, g, "o" + ls, color=col, label=f"{s}: p = {f['p']:.2f}")
        ax[0, 1].loglog(x, np.exp(np.polyval(f["pp"], np.log(x))), color=col, lw=0.8, alpha=0.6)
    for k, t in ((0, "exponential test (straight = exponential)"), (1, "power-law test (straight = power law)")):
        ax[0, k].set_xlabel("N")
        ax[0, k].set_ylabel(r"smallest $|\lambda|$")
        ax[0, k].set_title(rf"({'ab'[k]}) $(\chi,\lambda)=({chi:g},{lam:g})$: {t}", loc="left", fontsize=10.5)
        ax[0, k].legend(fontsize=8.5, frameon=False)
    first = [(N, r) for (s, N), recs in rows.items() if s == "E_+-_i" for r in recs if r["src"] == a.dirs[0]]
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, max(1, len(first))))
    for (N, r), c in zip(sorted(first, key=lambda t: t[0]), cols):
        e = np.array([least_damped(v) for v in r["ev"]])
        ax[1, 0].plot(r["u"], -N * e.real, "-", color=c, lw=1.5, label=f"N = {N}")
        ax[1, 1].plot(r["u"], np.abs(e.imag), "-", color=c, lw=1.5, label=f"N = {N}")
    if first and S:
        r = first[0][1]
        wp = np.full(r["u"].shape, np.nan)
        for i, (c_, l_) in enumerate(zip(r["chi"], r["lam"])):
            q = S.mf_Q(c_, model)
            if q is not None and 2 * l_ * q[2] < abs(model["wc"]):
                wp[i] = np.sqrt(model["wc"] ** 2 - (2 * l_ * q[2]) ** 2)
        ax[1, 1].plot(r["u"], wp, "k--", lw=1.4, label=r"MF $\omega'$ (Q)")
        V = r["path"]
        for k in (0, 1):
            ax[1, k].set_xticks(np.arange(len(V)))
            ax[1, k].set_xticklabels([f"({c:g}, {l:g})" for c, l in V], fontsize=9)
            for xv in range(1, len(V) - 1):
                ax[1, k].axvline(xv, color="k", ls=":", lw=0.8)
    ax[1, 0].set_ylabel(r"$N\times(-\mathrm{Re}\,\lambda)$, least-damped $E_{+-,i}$ mode")
    ax[1, 0].set_title("(c) damping of the parity-coherence mode (flat = 1/N)", loc="left", fontsize=10.5)
    ax[1, 1].set_ylabel(r"$|\mathrm{Im}\,\lambda|$, least-damped $E_{+-,i}$ mode")
    ax[1, 1].set_title("(d) its frequency (N-independent = not a zero mode)", loc="left", fontsize=10.5)
    for k in (0, 1):
        ax[1, k].set_xlabel(r"path through $(\chi,\lambda)$")
        ax[1, k].legend(fontsize=8.5, frameon=False)
    for axx in ax.flat:
        for s_ in ("top", "right"):
            axx.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=170)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()