#!/usr/bin/env python3
"""
replot_traceless.py -- redraw the gap-closure figure of sweep_traceless.py (Figure A.3 of the
thesis) from its saved data, in the notation of the thesis.

Sector labels E^w_{lr} as in Eq. (5.4): l, r = photon-number parity of ket and bra
(0 = even, 1 = odd), w = weak charge, Pi_4 rho Pi_4^dagger = i^w rho with
Pi_4 = exp(i pi (a^dagger a + J_z) / 2), the same operator sweep_traceless.py uses. Translation:
    E_++_-1 -> E^2_00     E_--_-1 -> E^2_11
    E_+-_i  -> E^1_01     E_+-_-i -> E^3_01     E_-+_i -> E^1_10     E_-+_-i -> E^3_10
(For odd N the spin eigenvalue n in Eq. (5.4) is odd, so (n + m) mod 2 would swap the strong
labels 0 and 1; the labels here are the photon parities for every N, which is Eq. (5.4) for
even N.)

Everything else is the original figure: 3 x 2 panels, viridis colours per N, open circles for
points whose Fock-cutoff tail exceeds tail_tol, dashed line at the mean-field lambda_c. The
title on top is gone (the parameters belong in the caption), and the axes read "coupling
lambda" and "dissipative gap -Re lambda_1" as in Figure A.4.

    python replot_traceless.py                          # data in Gap_Closure_Traceless/
    python replot_traceless.py my_outdir --scale lin    # only the linear version
    python replot_traceless.py --no-conj-note           # panel titles without "(= complex ...)"

Writes <outdir>/gap_closure_traceless_thesis_{log,lin}.{pdf,png}. Reads data/gap_*_N*.npz and
run_info.json (for omega_c, omega_a, kappa_2 and tail_tol); needs only numpy and matplotlib.
sweep_traceless.py is neither imported nor changed.
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# script label -> (weak charge w, strong labels lr) of E^w_{lr}
THESIS = {"E_++_1": (0, "00"), "E_--_1": (0, "11"), "E_++_-1": (2, "00"), "E_--_-1": (2, "11"),
          "E_+-_i": (1, "01"), "E_+-_-i": (3, "01"), "E_-+_i": (1, "10"), "E_-+_-i": (3, "10")}
LAYOUT = [["E_++_-1", "E_--_-1"], ["E_+-_i", "E_-+_-i"], ["E_+-_-i", "E_-+_i"]]
CONJ = {"E_-+_-i": "E_+-_i", "E_-+_i": "E_+-_-i"}      # spectrum = complex conjugate of


def label(s):
    w, lr = THESIS[s]
    return rf"$E^{{{w}}}_{{{lr}}}$"


def load(outdir):
    files = glob.glob(os.path.join(outdir, "data", "gap_*_N*.npz"))
    if not files:
        raise SystemExit(f"no data/gap_*_N*.npz in {outdir}")
    data = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        data[(str(d["sector"]), int(d["N"]))] = d
    return data


def parameters(outdir, args):
    """omega_c, omega_a, kappa_2, tail_tol of the run (run_info.json), unless given here."""
    p = dict(wc=1.0, wa=1.0, kappa2=0.05, tail_tol=1e-7)          # sweep_traceless defaults
    f = os.path.join(outdir, "run_info.json")
    if os.path.exists(f):
        saved = json.load(open(f)).get("params", {})
        p.update({k: float(saved[k]) for k in p if saved.get(k) is not None})
    for k in p:
        if getattr(args, k) is not None:
            p[k] = getattr(args, k)
    return p


def plot(data, p, scale, out, conj_note=True):
    Ns = sorted({N for (_, N) in data})
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(Ns)))
    lam_c = 0.5 * np.sqrt(p["wc"] ** 2 + p["kappa2"] * p["wa"])     # as in sweep_traceless.py
    fig, axes = plt.subplots(3, 2, figsize=(10, 11), sharex=True)
    for i, row in enumerate(LAYOUT):
        for j, s in enumerate(row):
            ax = axes[i, j]
            for N, c in zip(Ns, cols):
                if (s, N) not in data:
                    continue
                d = data[(s, N)]
                lam, gap, tail = d["lam_vals"], d["gap"], d["tail"]
                ax.plot(lam, gap, "-", color=c, lw=1.5, label=rf"$N = {N}$")
                ok = tail <= p["tail_tol"]
                ax.plot(lam[ok], gap[ok], "o", color=c, ms=3.5)
                ax.plot(lam[~ok], gap[~ok], "o", mfc="white", mec=c, ms=4.5)
            ax.axvline(lam_c, color="gray", ls="--", lw=1)
            if scale == "log":
                ax.set_yscale("log")
            title = "Sector " + label(s)
            if conj_note and s in CONJ:
                title += "   (= complex conjugate of " + label(CONJ[s]) + ")"
            ax.set_title(title, fontsize=11)
            ax.grid(True, which="both", ls="--", alpha=0.35)
            if j == 0:
                ax.set_ylabel(r"dissipative gap $-\mathrm{Re}\,\lambda_1$")
            if i == 2:
                ax.set_xlabel(r"coupling $\lambda$")
    axes[0, 0].plot([], [], "--", color="gray", lw=1, label=r"mean-field $\lambda_c$")
    axes[0, 0].legend(fontsize=9)
    fig.tight_layout()
    written = []
    for ext in ("pdf", "png"):
        fn = f"{out}_{scale}.{ext}"
        fig.savefig(fn, dpi=200)
        written.append(fn)
    plt.close(fig)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", nargs="?", default="Gap_Closure_Traceless",
                    help="output folder of sweep_traceless.py (default Gap_Closure_Traceless)")
    ap.add_argument("--scale", choices=("log", "lin", "both"), default="both")
    ap.add_argument("--out", default=None,
                    help="file name without _log/_lin and extension "
                         "(default <outdir>/gap_closure_traceless_thesis)")
    ap.add_argument("--no-conj-note", action="store_true",
                    help="no '(= complex conjugate of ...)' in the right-hand panel titles")
    ap.add_argument("--wc", type=float, default=None)
    ap.add_argument("--wa", type=float, default=None)
    ap.add_argument("--kappa2", type=float, default=None)
    ap.add_argument("--tail-tol", type=float, default=None)
    args = ap.parse_args()
    data = load(args.outdir)
    p = parameters(args.outdir, args)
    out = args.out or os.path.join(args.outdir, "gap_closure_traceless_thesis")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for scale in (("log", "lin") if args.scale == "both" else (args.scale,)):
        for fn in plot(data, p, scale, out, conj_note=not args.no_conj_note):
            print("wrote", fn)


if __name__ == "__main__":
    main()