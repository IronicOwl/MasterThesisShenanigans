#!/usr/bin/env python3
"""
gap_scan.py -- finite-size scaling of the first-order dissipative gap in the
two-photon Dicke model with two-photon loss only, in a trace-1 symmetry sector.

Model (same conventions as your QuTiP code / Shah et al.):
    H = wc a^+a + (wa/2) J_z + (lam/N) J_x (a^2 + a^+2),   J = 2 S (Pauli sums)
    L = sqrt(kappa2/N) a^2
Sector: one photon-parity diagonal block (E_++ or E_--, strong Z2) with
weak-Z4 eigenvalue +1, i.e. the sector holding that block's steady state.

Per lambda: the Liouvillian is built directly inside the parity block (only
even or only odd Fock states), factorised once with a nested-dissection
ordering, and the slowest eigenpairs are found by shift-invert Arnoldi.  The
steady state is identified as the only eigenmatrix with non-zero trace; the
vacuum <-> superradiant switching mode is identified from its eigenmatrix
(positive part vacuum-like, negative part superradiant, or vice versa).
The minimum of the switching gap over lambda is located by a parallel zoom
and a parabola fit of ln(gap), and the analysis step fits its N-dependence.

Workflow (e.g. one SLURM array task per N, then one analysis run):
    python gap_scan.py selfcheck                 # sector construction vs full space
    python gap_scan.py compute --N 6 --workers 8
    python gap_scan.py compute --N 7 --workers 4
    python gap_scan.py analyze                   # fits + plots from all saved N
Memory: each worker holds one sparse LU.  The first evaluation prints its size;
peak use is roughly 2.5x that.  Choose --workers accordingly.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import time
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, eigs, LinearOperator, ArpackNoConvergence


# --------------------------------------------------------------------------- mean field
def lam_c(wc, wa, k2):
    """Mean-field saddle-node where the superradiant branch appears."""
    return 0.5 * np.sqrt(wc**2 + k2 * wa)


def mf_r2(lam, wc, wa, k2):
    """|alpha|^2 / N on the stable superradiant branch (None below lam_c)."""
    g = 2.0 * lam                     # (lam/N) J_x = (2 lam/N) S_x
    disc = (g**2 - wc**2) ** 2 - (k2 * wa) ** 2
    if g**2 <= wc**2 or disc < 0:
        return None
    v = (2 * wc**2 / k2**2) * ((g**2 - wc**2) + np.sqrt(disc))
    return v / (2 * wc * np.sqrt(v + wa**2))


def photon_cutoff(N, lam, p):
    """Fock cutoff M (full-space convention: n = 0..M-1).  Default: mfac x the
    mean-field superradiant photon number + moff, rounded up to a multiple of 8.
    mfac = 2.2 gave ~0.1 % accuracy of the gap at N = 5."""
    if p["mfixed"]:
        return int(p["mfixed"])
    lc = lam_c(p["wc"], p["wa"], p["kappa2"])
    r2 = mf_r2(max(lam, lc * (1 + 1e-9)), p["wc"], p["wa"], p["kappa2"])
    M = p["mfac"] * N * r2 + p["moff"]
    return int(np.ceil(M / 8.0) * 8)


# --------------------------------------------------------------------------- sector Liouvillian
def nd_perm(coords, leaf=64):
    """Geometric nested dissection.  Every Liouvillian term moves each coordinate
    (n_index, s, n'_index, s') by at most one step, so single layers separate."""
    order = []

    def rec(sub):
        if len(sub) == 0:
            return
        c = coords[sub]
        lo, hi = c.min(0), c.max(0)
        d = int(np.argmax(hi - lo))
        if len(sub) <= leaf or hi[d] - lo[d] < 2:
            order.append(sub)
            return
        mid = (lo[d] + hi[d]) // 2
        rec(sub[c[:, d] < mid])
        rec(sub[c[:, d] > mid])
        order.append(sub[c[:, d] == mid])

    rec(np.arange(len(coords)))
    return np.concatenate(order)


class Sector:
    """L(lam) = L0 + lam * L1 restricted to (parity block, weak-Z4 = +1),
    stored in nested-dissection order."""

    def __init__(self, N, M, parity, wc, wa, k2):
        j = N / 2.0
        ms = np.arange(-j, j + 1)
        ds = N + 1
        ns = np.arange(parity, M, 2)          # only one photon parity
        dn = len(ns)
        A2 = sp.diags([np.sqrt(ns[1:] * (ns[1:] - 1.0))], [1], shape=(dn, dn), format="csr")
        Splus = sp.diags([np.sqrt(j * (j + 1) - ms[:-1] * (ms[:-1] + 1))], [-1],
                         shape=(ds, ds), format="csr")
        Sx = 0.5 * (Splus + Splus.T)
        Sz = sp.diags(ms, format="csr")
        Ip, Is = sp.identity(dn, format="csr"), sp.identity(ds, format="csr")
        H0 = wc * sp.kron(sp.diags(ns.astype(float)), Is) + wa * sp.kron(Ip, Sz)  # (wa/2) J_z = wa S_z
        H1 = (2.0 / N) * sp.kron(A2 + A2.T, Sx)                                     # (1/N) J_x (a^2 + a^+2)
        Lj = np.sqrt(k2 / N) * sp.kron(A2, Is)
        D = dn * ds
        I = sp.identity(D, format="csr")
        LdL = (Lj.conj().T @ Lj).tocsr()
        # column stacking (as QuTiP): vec(A rho B) = (B^T kron A) vec(rho)
        L0 = (-1j * (sp.kron(I, H0) - sp.kron(H0.T, I)) + sp.kron(Lj.conj(), Lj)
              - 0.5 * sp.kron(I, LdL) - 0.5 * sp.kron(LdL.T, I)).tocsr()
        L1 = (-1j * (sp.kron(I, H1) - sp.kron(H1.T, I))).tocsr()
        # weak Z4 (Pi = exp[i pi (n/2 + S_z)]) eigenvalue +1 inside the parity block
        cls = (np.repeat((ns - parity) // 2, ds) + np.tile(np.arange(ds), dn)) % 2
        rowv = np.tile(np.arange(D), D)
        colv = np.repeat(np.arange(D), D)
        keep = np.flatnonzero(cls[rowv] == cls[colv])
        row, col = keep % D, keep // D
        coords = np.stack([row // ds, row % ds, col // ds, col % ds], axis=1)
        sel = keep[nd_perm(coords)]
        self.L0 = L0[sel][:, sel].tocsc()
        self.L1 = L1[sel][:, sel].tocsc()
        self.sel, self.D = sel, D
        self.nvec = np.repeat(ns.astype(float), ds)       # photon number of basis states

    def to_rho(self, v):
        full = np.zeros(self.D * self.D, dtype=complex)
        full[self.sel] = v
        return full.reshape(self.D, self.D).T              # rho[row, col]


_CACHE = OrderedDict()


def get_sector(N, M, p):
    key = (N, M, p["parity"], p["wc"], p["wa"], p["kappa2"])
    if key not in _CACHE:
        _CACHE[key] = Sector(N, M, p["parity"], p["wc"], p["wa"], p["kappa2"])
        while len(_CACHE) > 2:
            _CACHE.popitem(last=False)
    return _CACHE[key]


def photon_split(r, nvec):
    """<a^+a> of the normalised positive and negative parts of an eigenmatrix."""
    h, a = r + r.conj().T, 1j * (r - r.conj().T)
    r = h if np.linalg.norm(h) >= np.linalg.norm(a) else a
    w, U = np.linalg.eigh(r)
    out = []
    for s in (w > 0, w < 0):
        wt = np.abs(w[s])
        diagP = np.einsum("ij,j,ij->i", U[:, s], wt, U[:, s].conj()).real
        out.append(float(np.sum(nvec * diagP) / np.sum(wt)))
    return out


def evaluate(task):
    N, lam, M, p = task
    t0 = time.time()
    S = get_sector(N, M, p)
    Lp = (S.L0 + lam * S.L1).tocsc()
    K = Lp.shape[0]
    t1 = time.time()
    lu = splu((Lp - p["sigma"] * sp.identity(K, format="csc")).tocsc(), permc_spec="NATURAL",
              diag_pivot_thresh=0.1, options=dict(SymmetricMode=True))
    t_lu = time.time() - t1
    nnz = lu.L.nnz + lu.U.nnz
    OPinv = LinearOperator((K, K), matvec=lambda b: lu.solve(np.asarray(b, dtype=complex)),
                           dtype=complex)
    try:
        vals, vecs = eigs(Lp, k=p["k"], sigma=p["sigma"], OPinv=OPinv, which="LM",
                          tol=p["tol"], maxiter=20000)
    except ArpackNoConvergence as e:
        vals, vecs = e.eigenvalues, e.eigenvectors
    del lu
    rhos = [S.to_rho(vecs[:, i]) for i in range(len(vals))]
    trw = [abs(np.trace(r)) / np.linalg.norm(r) for r in rhos]
    iss = int(np.argmax(trw))                               # only trace-carrying mode
    rss = 0.5 * (rhos[iss] + rhos[iss].conj().T)
    rss /= np.trace(rss).real
    n_ss = float(np.real(np.sum(S.nvec * np.diag(rss))))
    modes = []
    for i in np.argsort(-vals.real):
        if i == iss:
            continue
        m = dict(re=float(vals[i].real), im=float(vals[i].imag), switch=False)
        if abs(vals[i].imag) <= 1e-6 * abs(vals[i].real) + 1e-12:
            npos, nneg = photon_split(rhos[i], S.nvec)
            hi, lo = max(npos, nneg), min(npos, nneg)
            m.update(n_pos=npos, n_neg=nneg,
                     switch=bool(hi > max(2.0, 1.0 * N) and lo < 0.25 * hi))
        modes.append(m)
    sw = [m for m in modes if m["switch"]]
    out = dict(lam=float(lam), M=int(M), K=int(K), nnz_lu=int(nnz), t_lu=t_lu,
               t_total=time.time() - t0, ss_eval=float(abs(vals[iss])), n_ss=n_ss,
               n_ss_per_spin=n_ss / N, gap_all=-modes[0]["re"] if modes else np.nan,
               gap_sw=-sw[0]["re"] if sw else np.nan,
               sw_n=(sw[0]["n_pos"], sw[0]["n_neg"]) if sw else None, modes=modes)
    print(f"  N={N} lam={lam:.5f} M={M} K={K} LU={nnz:.2e} nnz ({nnz * 16 / 1e9:.2f} GB, {t_lu:.0f}s) "
          f"gap_sw={out['gap_sw']:.4e} gap_all={out['gap_all']:.4e} <n>/N={n_ss / N:.3f} "
          f"[{out['t_total']:.0f}s]", flush=True)
    return out


# --------------------------------------------------------------------------- compute driver
def fit_minimum(res):
    lam = np.array([r["lam"] for r in res])
    g = np.array([r["gap_sw"] for r in res])
    ok = np.isfinite(g)
    lam, g = lam[ok], g[ok]
    o = np.argsort(lam)
    lam, g = lam[o], g[o]
    ib = int(np.argmin(g))
    edge = ib == 0 or ib == len(lam) - 1
    idx = np.sort(np.argsort(np.abs(lam - lam[ib]))[:5])
    lam_star, gmin, method = lam[ib], g[ib], "grid"
    if len(idx) >= 3 and not edge:
        x = lam[idx] - lam[ib]
        c2, c1, c0 = np.polyfit(x, np.log(g[idx]), 2)
        if c2 > 0 and abs(c1 / (2 * c2)) <= 0.5 * (x.max() - x.min()):
            lam_star = lam[ib] - c1 / (2 * c2)
            gmin = float(np.exp(c0 - c1**2 / (4 * c2)))
            method = "parabola in ln(gap)"
    return float(lam_star), float(gmin), bool(edge), method


def compute(a):
    p = dict(parity=0 if a.sector == "++" else 1, wc=a.wc, wa=a.wa, kappa2=a.kappa2,
             mfac=a.mfac, moff=a.moff, mfixed=a.mfixed, sigma=a.sigma, k=a.k, tol=a.tol)
    N = a.N
    lc = lam_c(a.wc, a.wa, a.kappa2)
    grid = np.array(a.lams) if a.lams else np.linspace(a.lam_lo, a.lam_hi, a.ngrid)
    zp = a.zoom_points or max(a.workers, 4)
    print(f"N={N} sector E_{a.sector}_1, mean-field lam_c={lc:.4f}, coarse grid {grid.min():.3f}..{grid.max():.3f} "
          f"({len(grid)} pts), {a.zoom_rounds} zoom rounds x {zp} pts, {a.workers} workers", flush=True)

    def batch(pool, lams, pp=p):
        return pool.map(evaluate, [(N, float(l), photon_cutoff(N, l, pp), pp) for l in lams], chunksize=1)

    t0 = time.time()
    with Pool(a.workers) as pool:
        res = batch(pool, grid)
        for _ in range(a.zoom_rounds):
            lam = np.array(sorted(r["lam"] for r in res if np.isfinite(r["gap_sw"])))
            if len(lam) == 0:
                print("no switching mode found on the grid -- widen the lambda window", flush=True)
                break
            best = min((r for r in res if np.isfinite(r["gap_sw"])), key=lambda r: r["gap_sw"])["lam"]
            ib = int(np.argmin(np.abs(lam - best)))
            if ib in (0, len(lam) - 1):
                print("WARNING: minimum at the edge of the scanned window -- extend --lam-lo/--lam-hi", flush=True)
            lo, hi = lam[max(ib - 1, 0)], lam[min(ib + 1, len(lam) - 1)]
            allx = np.array([r["lam"] for r in res])
            new = [x for x in np.linspace(lo, hi, zp + 2)[1:-1] if np.min(np.abs(allx - x)) > 1e-7]
            res += batch(pool, new)
        lam_star, gmin, edge, method = fit_minimum(res)
        conv = None
        if a.convcheck and not edge:
            M0 = photon_cutoff(N, lam_star, p)
            M1 = int(np.ceil(1.25 * M0 / 8.0) * 8)
            pp = dict(p, mfixed=M1)
            r0, r1 = batch(pool, [lam_star], p)[0], batch(pool, [lam_star], pp)[0]
            conv = dict(M0=M0, M1=M1, gap0=r0["gap_sw"], gap1=r1["gap_sw"],
                        rel_change=abs(r1["gap_sw"] - r0["gap_sw"]) / r1["gap_sw"])
            print(f"cutoff check at lam*: M={M0} -> {M1}: relative change of gap {conv['rel_change']:.2e}", flush=True)
    summary = dict(N=N, sector=a.sector, lam_star=lam_star, gap_min=gmin, edge=edge, method=method,
                   lam_c_mf=lc, convcheck=conv, wall_time=time.time() - t0,
                   params=dict(vars(a), func=None))
    os.makedirs(a.outdir, exist_ok=True)
    fn = os.path.join(a.outdir, f"gapscan_{a.sector}_N{N}.json")
    json.dump(dict(summary=summary, evaluations=sorted(res, key=lambda r: r["lam"])), open(fn, "w"), indent=1)
    print(f"N={N}: lam*={lam_star:.5f}  gap_min={gmin:.4e}  ({method}{', AT EDGE' if edge else ''})  "
          f"-> {fn}  [{summary['wall_time']:.0f}s]", flush=True)


# --------------------------------------------------------------------------- analysis
def linfit(x, y):
    """Least squares y = c0 + c1 x; returns coefficients, standard errors, rms residual."""
    X = np.vstack([np.ones_like(x), x]).T
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ c
    dof = max(len(x) - 2, 1)
    cov = np.linalg.inv(X.T @ X) * (r @ r) / dof
    return c, np.sqrt(np.diag(cov)), float(np.sqrt(np.mean(r**2)))


def analyze(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(os.path.join(a.outdir, f"gapscan_{a.sector}_N*.json")))
    data = sorted([json.load(open(f)) for f in files], key=lambda d: d["summary"]["N"])
    if not data:
        raise SystemExit("no results found in " + a.outdir)
    S = [d["summary"] for d in data]
    Ns = np.array([s["N"] for s in S], float)
    ls = np.array([s["lam_star"] for s in S])
    gm = np.array([s["gap_min"] for s in S])
    use = np.array([not s["edge"] for s in S])
    print(f"\nSector E_{a.sector}_1   (mean-field lam_c = {S[0]['lam_c_mf']:.4f})")
    print("   N    lam*(N)     gap_min      ln gap_min   cutoff-check")
    for s in S:
        cc = f"{s['convcheck']['rel_change']:.1e}" if s.get("convcheck") else "-"
        print(f"  {s['N']:2d}   {s['lam_star']:.5f}   {s['gap_min']:.4e}   {np.log(s['gap_min']):8.3f}"
              f"     {cc}{'   (minimum at window edge: excluded)' if s['edge'] else ''}")
    x, y = Ns[use], np.log(gm[use])
    fits = {}
    if use.sum() >= 2:
        c, e, rms = linfit(x, y)
        fits["exp"] = (c, e, rms)
        print(f"\nexponential  gap_min = A exp(-W N):  W = {-c[1]:.4f} +- {e[1]:.4f} per spin,"
              f"  A = {np.exp(c[0]):.3e},  rms(ln) = {rms:.3f}")
        c, e, rms = linfit(np.log(x), y)
        fits["pow"] = (c, e, rms)
        print(f"power law    gap_min = B N^-z:       z = {-c[1]:.3f} +- {e[1]:.3f},"
              f"            B = {np.exp(c[0]):.3e},  rms(ln) = {rms:.3f}")
    if use.sum() >= 4:
        X = np.vstack([np.ones_like(x), x, np.log(x)]).T
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        rms = float(np.sqrt(np.mean((y - X @ c) ** 2)))
        print(f"exp x power  gap_min = A exp(-W N) N^-c:  W = {-c[1]:.4f}, c = {-c[2]:.3f}, rms(ln) = {rms:.3f}")
    if use.sum() >= 3:
        c, e, rms = linfit(1.0 / x, ls[use])
        fits["lam"] = (c, e, rms)
        print(f"lam*(N) = lam*_inf + b/N:    lam*_inf = {c[0]:.4f} +- {e[0]:.4f},  b = {c[1]:.3f},  rms = {rms:.1e}")
        c, e, rms = linfit(1.0 / x**2, ls[use])
        fits["lam2"] = (c, e, rms)
        print(f"lam*(N) = lam*_inf + b/N^2:  lam*_inf = {c[0]:.4f} +- {e[0]:.4f},  b = {c[1]:.3f},  rms = {rms:.1e}")
        print("  (the spread between these two extrapolations is a fair estimate of their uncertainty)")

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(data)))
    for d, col in zip(data, cols):
        ev = d["evaluations"]
        lam = np.array([r["lam"] for r in ev])
        gsw = np.array([r["gap_sw"] for r in ev])
        gall = np.array([r["gap_all"] for r in ev])
        N = d["summary"]["N"]
        ax[0, 0].semilogy(lam, gsw, "o-", ms=3.5, color=col, label=f"N={N}")
        ax[0, 0].semilogy(lam, gall, ":", color=col, lw=1)
        ax[0, 0].plot(d["summary"]["lam_star"], d["summary"]["gap_min"], "*", color=col, ms=12, mec="k")
        ax[1, 1].plot(lam, [r["n_ss_per_spin"] for r in ev], "o-", ms=3, color=col, label=f"N={N}")
    for axx in (ax[0, 0], ax[1, 1]):
        axx.axvline(S[0]["lam_c_mf"], color="gray", ls="--", lw=1)
        axx.set_xlabel(r"$\lambda$")
        axx.legend(fontsize=8)
    ax[0, 0].set_ylabel(r"switching gap $-\mathrm{Re}\,\lambda_{sw}$  (dotted: slowest mode of any kind)")
    ax[0, 0].set_title(f"(a) trace-1 sector E_{a.sector}_1: gap vs coupling")
    ax[1, 1].set_ylabel(r"steady state $\langle a^\dagger a\rangle / N$")
    ax[1, 1].set_title("(d) steady-state photon number")
    ax[0, 1].semilogy(Ns[use], gm[use], "ko", ms=7, label="minimum of the switching gap")
    if (~use).any():
        ax[0, 1].semilogy(Ns[~use], gm[~use], "kx", ms=8, label="minimum at window edge (excluded)")
    if "exp" in fits:
        xx = np.linspace(Ns.min() - 0.5, Ns.max() + 1, 100)
        c, e, rms = fits["exp"]
        ax[0, 1].semilogy(xx, np.exp(c[0] + c[1] * xx), "C3-",
                          label=fr"$A e^{{-WN}}$, W={-c[1]:.3f}$\pm${e[1]:.3f} (rms {rms:.2f})")
        c, e, rms = fits["pow"]
        ax[0, 1].semilogy(xx, np.exp(c[0]) * xx ** c[1], "C0--",
                          label=fr"$B N^{{-z}}$, z={-c[1]:.2f} (rms {rms:.2f})")
    ax[0, 1].set_xlabel("N")
    ax[0, 1].set_ylabel(r"$\min_\lambda$ switching gap")
    ax[0, 1].set_title("(b) closing of the gap at the transition")
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(1.0 / Ns[use], ls[use], "ko", ms=7)
    if "lam" in fits:
        xx = np.linspace(0, 1.05 / Ns[use].min(), 50)
        c, e, rms = fits["lam"]
        ax[1, 0].plot(xx, c[0] + c[1] * xx, "C3-", label=fr"$a+b/N$: $\lambda^*_\infty$={c[0]:.4f}$\pm${e[0]:.4f}")
        c, e, rms = fits["lam2"]
        ax[1, 0].plot(xx, c[0] + c[1] * xx**2, "C0--", label=fr"$a+b/N^2$: $\lambda^*_\infty$={c[0]:.4f}$\pm${e[0]:.4f}")
        ax[1, 0].legend(fontsize=8)
    ax[1, 0].axhline(S[0]["lam_c_mf"], color="gray", ls="--", lw=1)
    ax[1, 0].set_xlabel("1/N")
    ax[1, 0].set_ylabel(r"$\lambda^*(N)$ (position of the minimum)")
    ax[1, 0].set_title("(c) drift of the transition point")
    fig.tight_layout()
    fn = os.path.join(a.outdir, f"gap_scaling_{a.sector}.png")
    fig.savefig(fn, dpi=150)
    print("\nplot ->", fn)

    # Effective closing exponent W_eff(lam) = -d ln(gap)/dN between consecutive N,
    # on the lambda range covered by both scans.  Exponential closing over the whole
    # coexistence window (not only at lam*) is what metastability predicts.
    curves = []
    for d in data:
        ev = [r for r in d["evaluations"] if np.isfinite(r["gap_sw"])]
        lam = np.array([r["lam"] for r in ev])
        o = np.argsort(lam)
        curves.append((d["summary"]["N"], lam[o], np.log(np.array([r["gap_sw"] for r in ev])[o])))
    if len(curves) >= 2:
        fig2, ax2 = plt.subplots(figsize=(6.5, 4.3))
        for (N1, l1, g1), (N2, l2, g2), col in zip(curves[:-1], curves[1:], cols[1:]):
            lo, hi = max(l1.min(), l2.min()), min(l1.max(), l2.max())
            if hi <= lo:
                continue
            ll = np.linspace(lo, hi, 200)
            W = -(np.interp(ll, l2, g2) - np.interp(ll, l1, g1)) / (N2 - N1)
            ax2.plot(ll, W, color=col, label=f"N={N1}->{N2}")
        for s in S:
            ax2.axvline(s["lam_star"], color="k", lw=0.6, alpha=0.4)
        ax2.axvline(S[0]["lam_c_mf"], color="gray", ls="--", lw=1)
        ax2.set_xlabel(r"$\lambda$")
        ax2.set_ylabel(r"$W_{\rm eff}(\lambda) = -\Delta \ln(\mathrm{gap}) / \Delta N$")
        ax2.set_title("effective closing exponent (thin lines: lambda*(N))")
        ax2.legend(fontsize=8)
        fig2.tight_layout()
        fn2 = os.path.join(a.outdir, f"gap_closing_rate_{a.sector}.png")
        fig2.savefig(fn2, dpi=150)
        print("plot ->", fn2)


# --------------------------------------------------------------------------- self-check
def selfcheck(a):
    """Compare the reduced sector construction with a full-space Liouvillian built
    exactly like the QuTiP script (column stacking, spin basis +j..-j, masks from
    exp(i pi n) and exp(i pi (n/2 + Jz)))."""
    rng = [(3, 24, 0.55, 0), (4, 20, 0.53, 1), (5, 20, 0.54, 0)]
    for N, M, lam, parity in rng:
        S = Sector(N, M, parity, a.wc, a.wa, a.kappa2)
        ev_r = np.linalg.eigvals((S.L0 + lam * S.L1).toarray())
        j, ds = N / 2.0, N + 1
        m = np.arange(j, -j - 1, -1)
        Jp = sp.diags([np.sqrt(j * (j + 1) - m[1:] * (m[1:] + 1))], [1], shape=(ds, ds))
        Jx, Jz = (Jp + Jp.T) / 2, sp.diags(m)
        a1 = sp.diags([np.sqrt(np.arange(1, M))], [1], shape=(M, M))
        aa = sp.kron(a1, sp.identity(ds))
        num = sp.kron(sp.diags(np.arange(M) * 1.0), sp.identity(ds))
        JxH, JzH = 2 * sp.kron(sp.identity(M), Jx), 2 * sp.kron(sp.identity(M), Jz)
        H = (a.wc * num + (a.wa / 2) * JzH + (lam / N) * JxH @ (aa.T @ aa.T + aa @ aa)).tocsr()
        L2 = (np.sqrt(a.kappa2 / N) * aa @ aa).tocsr()
        D = M * ds
        I = sp.identity(D)
        LdL = L2.conj().T @ L2
        Lf = (-1j * (sp.kron(I, H) - sp.kron(H.T, I)) + sp.kron(L2.conj(), L2)
              - 0.5 * sp.kron(I, LdL) - 0.5 * sp.kron(LdL.T, I)).tocsr()
        dn_ = np.repeat(np.arange(M), ds)
        dz = np.tile(m, M)
        P2 = np.exp(1j * np.pi * dn_)
        P4 = np.exp(1j * np.pi * (dn_ / 2 + dz))
        rows, cols = np.tile(np.arange(D), D), np.repeat(np.arange(D), D)
        sgn = 1 if parity == 0 else -1
        mask = ((sgn * P2[rows].real > 0) & (sgn * P2[cols].real > 0)
                & np.isclose(P4[rows] * np.conj(P4[cols]), 1, atol=1e-8))
        ev_f = np.linalg.eigvals(Lf[mask][:, mask].toarray())
        dev = max(np.min(np.abs(ev_f - z)) for z in ev_r)
        dev2 = max(np.min(np.abs(ev_r - z)) for z in ev_f)
        print(f"N={N} M={M} lam={lam} sector E_{'++' if parity == 0 else '--'}_1: "
              f"dim {len(ev_r)} vs {len(ev_f)}, max eigenvalue mismatch {max(dev, dev2):.2e}")


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["compute", "analyze", "selfcheck"])
    ap.add_argument("--N", type=int, help="number of spins (compute)")
    ap.add_argument("--sector", choices=["++", "--"], default="++", help="photon-parity block (default ++)")
    ap.add_argument("--wc", type=float, default=1.0)
    ap.add_argument("--wa", type=float, default=1.0)
    ap.add_argument("--kappa2", type=float, default=0.05)
    ap.add_argument("--lam-lo", type=float, default=0.51)
    ap.add_argument("--lam-hi", type=float, default=0.60)
    ap.add_argument("--ngrid", type=int, default=10, help="coarse grid points")
    ap.add_argument("--lams", type=float, nargs="*", help="explicit coarse grid (overrides lo/hi/ngrid)")
    ap.add_argument("--zoom-rounds", type=int, default=2)
    ap.add_argument("--zoom-points", type=int, default=0, help="points per zoom round (default = workers, >=4)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--mfac", type=float, default=2.2, help="cutoff = mfac * N * r2_MF(lam) + moff")
    ap.add_argument("--moff", type=float, default=16)
    ap.add_argument("--mfixed", type=int, default=0, help="fixed Fock cutoff M (overrides mfac/moff)")
    ap.add_argument("--sigma", type=float, default=1e-8, help="shift (keep below the smallest expected gap)")
    ap.add_argument("--k", type=int, default=6, help="eigenpairs per lambda")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--no-convcheck", dest="convcheck", action="store_false",
                    help="skip the cutoff check (re-run at lam* with 1.25x M)")
    ap.add_argument("--outdir", default="gap_results")
    a = ap.parse_args()
    if a.mode == "compute":
        if a.N is None:
            ap.error("--N is required for compute")
        compute(a)
    elif a.mode == "analyze":
        analyze(a)
    else:
        selfcheck(a)


if __name__ == "__main__":
    main()