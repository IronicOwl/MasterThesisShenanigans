#!/usr/bin/env python3
"""
sweep_traceless.py -- gap closure in the six traceless symmetry sectors of the
two-photon Dicke model (two-photon loss only), for several N in one run.

Model and sectors exactly as in sweep.py:
    H  = wc a^+a + (wa/2) Jz + (lam/N) Jx (a^2 + a^+2)        (J = 2 x standard spin)
    L2 = sqrt(kappa2/N) a^2
    E_<pL pR>_<z>: photon parity of ket / bra (strong Z2) and eigenvalue z of
    rho -> Pi rho Pi^+ with Pi = exp(i pi (n/2 + Jz_std))   (weak Z4)
    traceless sectors: E_++_-1, E_--_-1, E_+-_i, E_+-_-i, E_-+_i, E_-+_-i
    "gap" = -Re of the eigenvalue with the largest real part found near sigma ~ 0.

What changed compared with sweep.py (speed / memory / accuracy):
  * every sector block is built directly from the parity-restricted Fock spaces,
    never from the full Liouvillian;
  * the sparse LU uses a nested-dissection ordering (several times faster, less memory);
  * the Fock cutoff grows with lambda: M = mfac * (mean-field photon number) + moff,
    so points at large lambda stay converged.  Each point stores the weight of the
    leading eigenmatrix in the top Fock states ('tail'); points with tail > --tail-tol
    are drawn as open circles;
  * E_-+_-i and E_-+_i are the Hermitian conjugates of E_+-_i and E_+-_-i
    (rho -> rho^+ commutes with the Liouvillian), so their spectra are exactly the
    complex conjugates.  They are filled in from those (--no-mirror computes them).

Output folder (default Gap_Closure_Traceless/):
    data/gap_<sector>_N<N>.npz   lam_vals, gap, leading_evals, all_evals, M_vals, tail, ...
    run_info.json                parameters, timings
    gap_closure_traceless_log.png, gap_closure_traceless_lin.png   (3 rows x 2 columns)

Usage:
    python sweep_traceless.py                      # N = 5, 6, 7 over 0.48 ... 0.58
    python sweep_traceless.py --Ns 4 5 6 --lam-max 0.56
    python sweep_traceless.py --plot-only          # redraw from the saved data
    python sweep_traceless.py --selfcheck          # compare with the full-space build of sweep.py
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

SECTORS = ["E_++_-1", "E_--_-1", "E_+-_i", "E_+-_-i", "E_-+_i", "E_-+_-i"]
MIRROR = {"E_-+_-i": "E_+-_i", "E_-+_i": "E_+-_-i"}           # sector : computed partner
LAYOUT = [["E_++_-1", "E_--_-1"], ["E_+-_i", "E_-+_-i"], ["E_+-_-i", "E_-+_i"]]
ZVAL = {"1": 1.0, "-1": -1.0, "i": 1j, "-i": -1j}
TEX = {"1": "1", "-1": "-1", "i": "i", "-i": "-i"}


def parse_sector(s):
    _, par, z = s.split("_")
    return (0 if par[0] == "+" else 1), (0 if par[1] == "+" else 1), ZVAL[z]


def sector_label(s):
    _, par, z = s.split("_")
    return rf"$E_{{{par},\,{TEX[z]}}}$"


# ------------------------------------------------------------------ mean field / cutoff
def lam_c(wc, wa, k2):
    """Mean-field coupling where the superradiant branch appears."""
    return 0.5 * np.sqrt(wc**2 + k2 * wa)


def mf_r2(lam, wc, wa, k2):
    """Mean-field superradiant photon number per spin (None below lam_c)."""
    g = 2.0 * lam
    disc = (g**2 - wc**2) ** 2 - (k2 * wa) ** 2
    if g**2 <= wc**2 or disc < 0:
        return None
    v = (2 * wc**2 / k2**2) * ((g**2 - wc**2) + np.sqrt(disc))
    return v / (2 * wc * np.sqrt(v + wa**2))


def ceil8(x):
    return int(np.ceil(x / 8.0) * 8)


def cutoff(N, lam, o):
    """Starting Fock cutoff M (n = 0 .. M-1), multiple of 8.  Near and below lam_c the
    slow modes already involve superradiant-like states, so the photon number used is
    never below its mean-field value at lam_c + 0.02."""
    if o["M_fixed"]:
        return int(o["M_fixed"])
    lc = lam_c(o["wc"], o["wa"], o["kappa2"])
    r2 = mf_r2(max(lam, lc + 0.02), o["wc"], o["wa"], o["kappa2"])
    return max(ceil8(o["mfac"] * N * r2 + o["moff"]), o["M_min"])


def est_peak_gb(N, M):
    """Conservative peak memory of one evaluation: LU of 1.6 GB measured at block size
    1.0e5 (N=6, M=128), growing like K^1.5, times 2.5 for SuperLU work space."""
    K = (M / 2 * (N + 1)) ** 2 / 2
    return 2.5 * 1.6 * (K / 100352.0) ** 1.5


# ------------------------------------------------------------------ sector Liouvillian
def nd_perm(coords, leaf=64):
    """Geometric nested dissection; every Liouvillian term moves each coordinate by <= 1."""
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


def parity_space(N, M, parity, wc, wa, k2):
    """Operators on the Fock states of one photon parity (x spin j = N/2)."""
    j = N / 2.0
    ms = np.arange(-j, j + 1)
    ds = N + 1
    ns = np.arange(parity, M, 2)
    dn = len(ns)
    A2 = sp.diags([np.sqrt(ns[1:] * (ns[1:] - 1.0))], [1], shape=(dn, dn), format="csr")
    Splus = sp.diags([np.sqrt(j * (j + 1) - ms[:-1] * (ms[:-1] + 1))], [-1],
                     shape=(ds, ds), format="csr")
    Sx = 0.5 * (Splus + Splus.T)
    Ip, Is = sp.identity(dn, format="csr"), sp.identity(ds, format="csr")
    H0 = (wc * sp.kron(sp.diags(ns.astype(float)), Is) + wa * sp.kron(Ip, sp.diags(ms))).tocsr()
    H1 = ((2.0 / N) * sp.kron(A2 + A2.T, Sx)).tocsr()          # (1/N) Jx (a^2 + a^+2)
    Lj = (np.sqrt(k2 / N) * sp.kron(A2, Is)).tocsr()
    nn = np.repeat(ns, ds).astype(float)
    mm = np.tile(ms, dn)
    return dict(H0=H0, H1=H1, Lj=Lj, LdL=(Lj.conj().T @ Lj).tocsr(), D=dn * ds, dn=dn,
                phase=np.exp(1j * np.pi * (nn / 2 + mm)),
                nidx=np.repeat(np.arange(dn), ds), sidx=np.tile(np.arange(ds), dn))


class SectorBlock:
    """L(lam) = L0 + lam * L1 on the operators |ket><bra| of one sector, in ND order."""

    def __init__(self, N, M, sector, wc, wa, k2):
        pr, pc, z = parse_sector(sector)
        R = parity_space(N, M, pr, wc, wa, k2)
        C = R if pc == pr else parity_space(N, M, pc, wc, wa, k2)
        Dr, Dc = R["D"], C["D"]
        Ir, Ic = sp.identity(Dr, format="csr"), sp.identity(Dc, format="csr")
        # column stacking (as QuTiP): vec(A rho B) = (B^T kron A) vec(rho), index = col*Dr + row
        L0 = (-1j * (sp.kron(Ic, R["H0"]) - sp.kron(C["H0"].T, Ir))
              + sp.kron(C["Lj"].conj(), R["Lj"])
              - 0.5 * sp.kron(Ic, R["LdL"]) - 0.5 * sp.kron(C["LdL"].T, Ir)).tocsr()
        L1 = (-1j * (sp.kron(Ic, R["H1"]) - sp.kron(C["H1"].T, Ir))).tocsr()
        rowv = np.tile(np.arange(Dr), Dc)
        colv = np.repeat(np.arange(Dc), Dr)
        keep = np.flatnonzero(np.isclose(R["phase"][rowv] * np.conj(C["phase"][colv]), z, atol=1e-8))
        coords = np.stack([R["nidx"][rowv[keep]], R["sidx"][rowv[keep]],
                           C["nidx"][colv[keep]], C["sidx"][colv[keep]]], axis=1)
        sel = keep[nd_perm(coords)]
        self.L0 = L0[sel][:, sel].tocsc()
        self.L1 = L1[sel][:, sel].tocsc()
        rs, cs = rowv[sel], colv[sel]
        self.edge = (R["nidx"][rs] >= R["dn"] - 3) | (C["nidx"][cs] >= C["dn"] - 3)
        self.K = len(sel)


_CACHE = OrderedDict()


def get_block(N, M, sector, o):
    key = (N, M, sector)
    if key not in _CACHE:
        _CACHE[key] = SectorBlock(N, M, sector, o["wc"], o["wa"], o["kappa2"])
        while len(_CACHE) > 2:
            _CACHE.popitem(last=False)
    return _CACHE[key]


def solve_point(N, sector, lam, M, o):
    """Slowest eigenvalues of one sector at one lambda and cutoff M."""
    B = get_block(N, M, sector, o)
    Lp = (B.L0 + lam * B.L1).tocsc()
    K = Lp.shape[0]
    ev = np.full(o["k"], np.nan, dtype=complex)
    try:
        lu = splu((Lp - o["sigma"] * sp.identity(K, format="csc")).tocsc(), permc_spec="NATURAL",
                  diag_pivot_thresh=0.1, options=dict(SymmetricMode=True))
    except RuntimeError as e:
        print(f"  N={N} {sector} lam={lam:.5f} M={M}: FACTORIZATION ERROR {e}", flush=True)
        return ev, np.nan, K, 0
    nnz = lu.L.nnz + lu.U.nnz
    OPinv = LinearOperator((K, K), matvec=lambda b: lu.solve(np.asarray(b, dtype=complex)),
                           dtype=complex)
    try:
        vals, vecs = eigs(Lp, k=o["k"], sigma=o["sigma"], OPinv=OPinv, which="LM",
                          tol=o["tol"], maxiter=20000)
    except ArpackNoConvergence as e:
        vals, vecs = e.eigenvalues, e.eigenvectors
    del lu
    order = np.argsort(-vals.real)
    vals, vecs = vals[order], vecs[:, order]
    tail = np.nan
    if len(vals):
        v = vecs[:, 0]
        tail = float(np.sum(np.abs(v[B.edge]) ** 2) / np.sum(np.abs(v) ** 2))
    ev[:len(vals)] = vals
    return ev, tail, K, nnz


def evaluate(task):
    """One (N, sector, lambda) point.  If the leading eigenmatrix has too much weight in the
    top Fock states, the cutoff is raised (x1.3) and the point recomputed, up to M_cap."""
    N, sector, lam, M, M_cap, o = task
    t0 = time.time()
    while True:
        ev, tail, K, nnz = solve_point(N, sector, lam, M, o)
        if tail <= o["tail_tol"] or M >= M_cap or not np.isfinite(tail):
            break
        M = min(M_cap, ceil8(1.3 * M))
    secs = time.time() - t0
    print(f"  N={N} {sector:8s} lam={lam:.5f} M={M:3d} K={K:7d} LU={nnz * 16 / 1e9:6.2f} GB "
          f"gap={-ev[0].real:.4e} (Im {ev[0].imag:+.2e}) tail={tail:.1e} [{secs:.0f}s]", flush=True)
    return dict(N=N, sector=sector, lam=lam, M=M, K=K, evals=ev, tail=tail, nnz=nnz, secs=secs)


# ------------------------------------------------------------------ driver
def slurm_mem_gb():
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if os.environ.get("SLURM_MEM_PER_CPU"):
        return float(os.environ["SLURM_MEM_PER_CPU"]) * cpus / 1024.0
    if os.environ.get("SLURM_MEM_PER_NODE"):
        return float(os.environ["SLURM_MEM_PER_NODE"]) / 1024.0
    return None


def save_sector(path, rs, extra=None):
    rs = sorted(rs, key=lambda r: r["lam"])
    ev = np.array([r["evals"] for r in rs])
    np.savez(path, lam_vals=np.array([r["lam"] for r in rs]), gap=-ev[:, 0].real,
             leading_evals=ev[:, 0], all_evals=ev, M_vals=np.array([r["M"] for r in rs]),
             K_vals=np.array([r["K"] for r in rs]), tail=np.array([r["tail"] for r in rs]),
             lu_nnz=np.array([r["nnz"] for r in rs]), seconds=np.array([r["secs"] for r in rs]),
             **(extra or {}))


def run(a):
    o = dict(wc=a.wc, wa=a.wa, kappa2=a.kappa2, mfac=a.mfac, moff=a.moff, M_min=a.M_min,
             M_fixed=a.M_fixed, sigma=a.sigma, k=a.k, tol=a.tol, tail_tol=a.tail_tol)
    datadir = os.path.join(a.outdir, "data")
    os.makedirs(datadir, exist_ok=True)
    lam_vals = np.linspace(a.lam_min, a.lam_max, a.n_steps)
    todo = SECTORS if a.no_mirror else [s for s in SECTORS if s not in MIRROR]
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    mem = slurm_mem_gb() or a.mem_gb
    info = dict(params=vars(a), lam_c_mf=lam_c(a.wc, a.wa, a.kappa2), runs=[])
    print(f"lambda {a.lam_min}..{a.lam_max} ({a.n_steps} pts), N = {a.Ns}, sectors computed: {todo}, "
          f"cpus={cpus}, memory budget={mem:.0f} GB, mean-field lam_c={info['lam_c_mf']:.4f}", flush=True)
    for N in sorted(a.Ns):
        Ms = [cutoff(N, l, o) for l in lam_vals]
        est = est_peak_gb(N, max(Ms))
        workers = a.workers or max(1, min(cpus, int(0.85 * mem / est)))
        print(f"\nN={N}: cutoff M from {min(Ms)} to {max(Ms)}, ~{est:.1f} GB peak per evaluation "
              f"-> {workers} parallel workers", flush=True)
        M_cap = max(Ms) if not a.M_fixed else a.M_fixed      # retries never exceed the memory estimate
        tasks = [(N, s, float(l), M, M_cap, o) for s in todo for l, M in zip(lam_vals, Ms)]
        tasks.sort(key=lambda t: -t[3])                     # largest cutoff first
        t0 = time.time()
        with Pool(workers) as pool:
            res = pool.map(evaluate, tasks, chunksize=1)
        for s in todo:
            save_sector(os.path.join(datadir, f"gap_{s}_N{N}.npz"), [r for r in res if r["sector"] == s],
                        dict(N=N, sector=s, mirrored_from=""))
        if not a.no_mirror:
            for s, src in MIRROR.items():
                rs = [dict(r, evals=np.conj(r["evals"])) for r in res if r["sector"] == src]
                save_sector(os.path.join(datadir, f"gap_{s}_N{N}.npz"), rs,
                            dict(N=N, sector=s, mirrored_from=src))
        bad = [(r["sector"], round(r["lam"], 4)) for r in res if not (r["tail"] <= a.tail_tol)]
        info["runs"].append(dict(N=N, workers=workers, M_min=min(Ms), M_max=max(Ms),
                                 seconds=time.time() - t0, points_above_tail_tol=bad))
        print(f"N={N} done in {time.time() - t0:.0f}s; points with cutoff tail > {a.tail_tol:g}: "
              f"{len(bad)}", flush=True)
        json.dump(info, open(os.path.join(a.outdir, "run_info.json"), "w"), indent=1, default=str)
    plot(a)


# ------------------------------------------------------------------ plotting
def plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datadir = os.path.join(a.outdir, "data")
    files = glob.glob(os.path.join(datadir, "gap_*_N*.npz"))
    if not files:
        raise SystemExit(f"no data in {datadir}")
    data = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        data[(str(d["sector"]), int(d["N"]))] = d
    Ns = sorted({N for (_, N) in data})
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(Ns)))
    lc = lam_c(a.wc, a.wa, a.kappa2)
    for scale in ("log", "lin"):
        fig, axes = plt.subplots(3, 2, figsize=(10, 11), sharex=True)
        for i, row in enumerate(LAYOUT):
            for j, s in enumerate(row):
                ax = axes[i, j]
                for N, c in zip(Ns, cols):
                    if (s, N) not in data:
                        continue
                    d = data[(s, N)]
                    lam, gap, tail = d["lam_vals"], d["gap"], d["tail"]
                    ax.plot(lam, gap, "-", color=c, lw=1.5, label=f"N = {N}")
                    ok = tail <= a.tail_tol
                    ax.plot(lam[ok], gap[ok], "o", color=c, ms=3.5)
                    ax.plot(lam[~ok], gap[~ok], "o", mfc="white", mec=c, ms=4.5)
                ax.axvline(lc, color="gray", ls="--", lw=1)
                if scale == "log":
                    ax.set_yscale("log")
                title = "Sector " + sector_label(s)
                if s in MIRROR and any(str(data[k]["mirrored_from"]) for k in data if k[0] == s):
                    title += "   (= complex conjugate of " + sector_label(MIRROR[s]) + ")"
                ax.set_title(title, fontsize=11)
                ax.grid(True, which="both", ls="--", alpha=0.35)
                if j == 0:
                    ax.set_ylabel(r"$-\mathrm{Re}(\lambda_{eval})$")
                if i == 2:
                    ax.set_xlabel(r"Coupling strength $\lambda$")
        axes[0, 0].plot([], [], "--", color="gray", lw=1, label=r"mean-field $\lambda_c$")
        axes[0, 0].legend(fontsize=9)
        fig.suptitle(rf"Dissipative gap in the traceless sectors  ($\omega_c$={a.wc:g}, $\omega_a$={a.wa:g}, "
                     rf"$\kappa_2$={a.kappa2:g})", fontsize=12)
        fig.tight_layout()
        fn = os.path.join(a.outdir, f"gap_closure_traceless_{scale}.png")
        fig.savefig(fn, dpi=200)
        plt.close(fig)
        print("plot ->", fn, flush=True)


# ------------------------------------------------------------------ self-check
def selfcheck(a):
    """All 8 sectors of the direct block construction vs the full-space Liouvillian built
    exactly like sweep.py (column stacking, spin basis +j..-j, masks from Pi_2, Pi_4)."""
    allsec = ["E_++_1", "E_++_-1", "E_--_1", "E_--_-1", "E_+-_i", "E_+-_-i", "E_-+_i", "E_-+_-i"]
    for N, M, lam in [(3, 16, 0.55), (4, 14, 0.53)]:
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
        P2 = np.exp(1j * np.pi * dn_).real
        P4 = np.exp(1j * np.pi * (dn_ / 2 + dz))
        rows, cols = np.tile(np.arange(D), D), np.repeat(np.arange(D), D)
        spectra = {}
        for s in allsec:
            pr, pc, z = parse_sector(s)
            mask = (((P2[rows] > 0) if pr == 0 else (P2[rows] < 0))
                    & ((P2[cols] > 0) if pc == 0 else (P2[cols] < 0))
                    & np.isclose(P4[rows] * np.conj(P4[cols]), z, atol=1e-8))
            ev_f = np.linalg.eigvals(Lf[mask][:, mask].toarray())
            B = SectorBlock(N, M, s, a.wc, a.wa, a.kappa2)
            ev_b = np.linalg.eigvals((B.L0 + lam * B.L1).toarray())
            spectra[s] = ev_b
            dev = max(max(np.min(np.abs(ev_f - x)) for x in ev_b), max(np.min(np.abs(ev_b - x)) for x in ev_f))
            print(f"N={N} M={M} lam={lam} {s:8s} dim {len(ev_b):5d} vs {len(ev_f):5d}  "
                  f"max eigenvalue mismatch {dev:.1e}", flush=True)
        for s, src in MIRROR.items():
            dev = max(np.min(np.abs(spectra[s] - np.conj(x))) for x in spectra[src])
            print(f"   mirror check: spectrum({s}) = conj(spectrum({src})) up to {dev:.1e}", flush=True)


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Ns", type=int, nargs="+", default=[5, 6, 7])
    ap.add_argument("--lam-min", type=float, default=0.48)
    ap.add_argument("--lam-max", type=float, default=0.58)
    ap.add_argument("--n-steps", type=int, default=33)
    ap.add_argument("--wc", type=float, default=1.0)
    ap.add_argument("--wa", type=float, default=1.0)
    ap.add_argument("--kappa2", type=float, default=0.05)
    ap.add_argument("--mfac", type=float, default=2.2, help="cutoff = mfac * N * r2_MF(lam) + moff")
    ap.add_argument("--moff", type=float, default=16)
    ap.add_argument("--M-min", type=int, default=40)
    ap.add_argument("--M-fixed", type=int, default=0, help="use one fixed cutoff instead")
    ap.add_argument("--k", type=int, default=6, help="eigenvalues per point")
    ap.add_argument("--sigma", type=float, default=1e-8)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--workers", type=int, default=0, help="0 = choose from memory budget")
    ap.add_argument("--mem-gb", type=float, default=100.0, help="memory budget if not under SLURM")
    ap.add_argument("--tail-tol", type=float, default=1e-7,
                    help="weight of the eigenmatrix in the top Fock states above which a point is "
                         "drawn open (1e-6 ~ few %% error in the gap)")
    ap.add_argument("--no-mirror", action="store_true", help="compute E_-+ sectors explicitly")
    ap.add_argument("--outdir", default="Gap_Closure_Traceless")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck(a)
    elif a.plot_only:
        plot(a)
    else:
        run(a)


if __name__ == "__main__":
    main()
