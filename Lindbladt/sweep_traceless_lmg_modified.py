#!/usr/bin/env python3
"""
sweep_traceless_lmg.py -- gap closure in the symmetry sectors of the two-photon Dicke model
extended by a spin-spin (LMG) term and collective spin decay, along a path in the
(chi, lambda) plane, for several N in one run.  Built on sweep_traceless.py.

    H   = wc a^+a + (wa/2) Jz + (lam/N) Jx (a^2 + a^+2) - (chi/N) Jx^2      (J = 2 x standard spin)
    L_2 = sqrt(kappa2/N) a^2 ,     L_s = sqrt(gamma/N) J_-                   (J_- = 2 x standard j_-)

chi = gamma = 0 is exactly the model of sweep_traceless.py.

Symmetry (unchanged): Pi_2 = exp(i pi n) commutes with H, L_2 and L_s (strong Z2);
Pi = exp(i pi (n/2 + Jz_std)) leaves H invariant and maps L_2 -> -L_2, L_s -> -L_s (weak Z4,
Pi^2 = +-Pi_2).  The sectors E_<pL pR>_<z> are the same as before; --selfcheck shows that the
full Liouvillian has no matrix element between different sectors and that the block spectra
agree with the full-space build.

What to expect (mean field, mf_nested.py; gamma = 0.1 gives chi_c = wa/4 + gamma^2/wa = 0.26)
along the default path  (chi, lam): (0.10, 0.47) -> (0.70, 0.47) -> (0.70, 0.80):
  first leg, chi up at lam = 0.47 < wc/2 (no lobes possible):
      E_++_-1 and E_--_-1 close at chi_c (spins order, weak Z4 -> Z2, 2 -> 4 zero modes);
      the coherence blocks stay open (slowest mode oscillates at w' = sqrt(wc^2 - (2 lam jx)^2))
  second leg, lam up at chi = 0.7: lobes appear at lam ~ 0.53, Q is unstable from
      lam_p = wc/(2 jx) ~ 0.537: E_+- and E_-+ close as well (4 -> 8 zero modes).
The mean-field transitions along the path are drawn as vertical lines.

Why kappa2 = 0.5 by default (not the thesis value 0.05): with collective spin decay the lobes
must not be too bright.  At kappa2 = 0.05 the lobed phase holds n/N ~ 10-17 photons, their torque
pins the spins on the equator (jz ~ -0.03), where sqrt(gamma/N) J_- adds noise but almost no
damping (damping ~ 2 gamma |jz|).  The spins are then hot and the 8-fold manifold is not
protected at reachable N: E_++_-1 and E_+- stay at ~0.6 gamma, independent of N (checked
N = 3..6, also at chi = 0).  At kappa2 = 0.5 the lobes hold n/N ~ 1-1.6, jz ~ -0.19 and the
gaps close with N; the small photon number also keeps M ~ 40-64, so N = 10-12 is cheap.
Spot checks at chi = 0.7, gamma = 0.1, kappa2 = 0.5 (slowest |eigenvalue|, N = 4/6/8/10/12):
  lam = 0.47 (Q):  E_++_-1  0.048 0.030 0.025 0.014 0.0077     E_+-_i  0.60 0.52 0.54 0.54 0.53
  lam = 0.62 (S):  E_++_-1  0.051 0.040 0.031 0.018           E_+-_i  0.25 0.15 0.11 0.082
  lam = 0.75 (S):  E_++_-1  0.058 0.044 0.028 0.017           E_+-_i  0.087 0.061 0.048 0.035
  chi = 0.1 (N):   E_++_-1  0.58 (N=6) 0.60 (N=10)             E_+-_i  0.28 0.29
Cost (one evaluation in the lobed phase at lam = 0.75-0.8, single core): N=10 ~100 s and 2.2 GB
of LU, N=12 ~7 min, N=14 ~20 min, N=16 ~45 min; the Q leg is far cheaper (M = 40).
Finished N are skipped when the job is resubmitted with the same path (--overwrite recomputes).

Two gap measures are stored for every point:
    gap     = -Re of the eigenvalue with the largest real part found near sigma ~ 0   (as before)
    absgap  = smallest |eigenvalue| found = distance of the slowest mode from zero
Below threshold the 1/N-scaled two-photon loss makes -Re small (~ kappa2/N) in every sector,
while modes that are not zero modes keep an O(1) imaginary part (spin precession, photon
coherences, the even/odd splitting w'); absgap separates "zero mode" from "slow but oscillating".
With --with-trace the trace sectors E_++_1, E_--_1 are added (gap after removing the steady
state); a small value there signals first-order switching inside the bistable region.

Output folder (default Gap_Closure_LMG/):
    data/gap_<sector>_N<N>.npz   u_vals, chi_vals, lam_vals, gap, absgap, leading_evals,
                                 all_evals, M_vals, tail, path, model parameters
    run_info.json
    gap_closure_{log,lin}.png    -Re of the slowest mode (as in sweep_traceless.py)
    zero_mode_distance_log.png   min |eigenvalue|

Usage:
    python sweep_traceless_lmg.py --selfcheck
    python sweep_traceless_lmg.py                                       # N = 4..12, default path
    python sweep_traceless_lmg.py --Ns 6 8 10 12 14 --path 0.7,0.55 0.7,0.80 --n-steps 11 \
           --outdir Gap_Closure_LMG_S                                    # lambda sweep in the 8 phase
    python sweep_traceless_lmg.py --path 0,0.47 0.8,0.47 --n-steps 33    # chi sweep at lam = 0.47
    python sweep_traceless_lmg.py --plot-only
    python sweep_traceless_lmg.py --lam 0.55 --chis 0.10 0.20 0.26 0.30 0.40 0.50 0.60 0.80 1.00 1.20 \
           --Ns 4 6 8 10 12 14 16 --outdir Gap_Closure_LMG_chicut         # chi alone, fixed lambda
    python sweep_traceless_lmg.py --lam 0.55 --chis 0.15 0.40 0.80 1.00 1.20 --Ns 8 10 12 14 16 18 20 \
           --sectors E_++_-1 E_+-_i --outdir Gap_Closure_LMG_chicut_bigN    # N scaling at a few points

Single-parameter cascade (chi alone at fixed lambda, --lam with --chis):
With wc/2 < lam < lam_S(chi_c) (lam_S(chi_c) = 0.573 at kappa2 = 0.5) a sweep of chi alone goes
2 -> 4 -> 8: chi orders the spins at chi_c, then keeps raising |jx| until the cavity drive
2 lam |jx| crosses wc and lobes form.  Mean field at lam = 0.55, kappa2 = 0.5, gamma = 0.1:
spins order at chi = 0.261, lobes appear at chi = 0.446, Q unstable (-> 8) at chi = 0.614.
The drive can never exceed 2 lam, so the lobed phase stays only slightly above threshold and the
lobes are dim (n/N ~ 0.8); the parity-coherence block closes slowly.  Spot check (smallest |ev|
of E_+-_i at chi = 1.0, lam = 0.5495, N = 4/6/8/10): 0.40 0.34 0.31 0.28, slowest mode still
oscillating (Im 0.33 -> 0.18), so N ~ 20 is needed to decide.  Dim lobes keep the cutoff small
(M ~ 40-64), so N = 20-24 is affordable at a few points.
--chis takes any spacing (put points where the transitions are); --sectors restricts the work
(--sectors overrides --with-trace; E_-+ sectors are still filled in by conjugation of their
computed E_+- partner).

Bookkeeping:
    data/points_<hash>/          one file per finished evaluation (N, sector, chi, lam), written as
                                 soon as it is done: a job killed by the time limit loses nothing,
                                 resubmitting resumes, and adding points or N reuses what exists
                                 (--overwrite recomputes).  <hash> encodes the model and cutoff
                                 settings, so other settings never mix.
    gap_scaling.png              smallest |eigenvalue| against N at selected points (log-log and
                                 log-lin); filled markers: the mode is not oscillating
                                 (|Im| <= 0.3 |ev|), open markers: it is
    summary.txt                  the same numbers as text, with the log-log slope of the last
                                 two N and the mean-field phase of each point
    --scaling-chis               chi values for those two (nearest computed points; default: all
                                 points if there are at most 6, else 6 spread along the sweep)
If a worker process is killed (usually for exceeding its memory), the run stops with exit code 3
instead of hanging; finished evaluations are kept, resubmit with a smaller --workers.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import hashlib
import json
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq
from scipy.sparse.linalg import splu, eigs, LinearOperator, ArpackNoConvergence

TRACELESS = ["E_++_-1", "E_--_-1", "E_+-_i", "E_+-_-i", "E_-+_i", "E_-+_-i"]
TRACE = ["E_++_1", "E_--_1"]
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


# ------------------------------------------------------------------ mean field (N -> infinity)
# alpha = <a>/sqrt(N) (coherent cavity), j = <J>/N on the unit sphere; same as mf_nested.py
def mf_rhs(y, chi, lam, o):
    wc, wa, k2, g = o["wc"], o["wa"], o["kappa2"], o["gamma"]
    al = y[0] + 1j * y[1]
    jx, jy, jz = y[2], y[3], y[4]
    da = -1j * wc * al - 2j * lam * jx * np.conj(al) - k2 * abs(al) ** 2 * al
    s = 2.0 * (al * al).real                                   # <a^2 + a^+2>/N
    return np.array([da.real, da.imag,
                     -wa * jy + 2 * g * jz * jx,
                     wa * jx - 2 * lam * jz * s + 4 * chi * jx * jz + 2 * g * jz * jy,
                     2 * lam * jy * s - 4 * chi * jx * jy - 2 * g * (jx * jx + jy * jy)])


def mf_stable(y, chi, lam, o, tol=1e-7, h=1e-6):
    """Linear stability on the tangent space of the Bloch sphere (neutral counts as stable)."""
    J = np.empty((5, 5))
    for i in range(5):
        e = np.zeros(5)
        e[i] = h
        J[:, i] = (mf_rhs(y + e, chi, lam, o) - mf_rhs(y - e, chi, lam, o)) / (2 * h)
    t = np.linalg.svd((y[2:] / np.linalg.norm(y[2:]))[None, :])[2][1:]
    B = np.zeros((5, 4))
    B[0, 0] = B[1, 1] = 1.0
    B[2:, 2:] = t.T
    return np.linalg.eigvals(B.T @ J @ B).real.max() <= tol


def chi_c(o):
    return o["wa"] / 4 + o["gamma"] ** 2 / o["wa"]


def mf_Q(chi, o):
    """alpha = 0, spins ordered by the LMG term (+jx copy; Pi maps it to -jx). None below chi_c."""
    g, wa = o["gamma"], o["wa"]
    if chi <= chi_c(o):
        return None
    jz = -wa / (2 * (chi + np.sqrt(chi ** 2 - g ** 2)))
    c = 2 * g * jz / wa
    jx = np.sqrt((1 - jz ** 2) / (1 + c * c))
    return np.array([0.0, 0.0, jx, c * jx, jz])


def mf_S(chi, lam, o, n=1500):
    """Fixed points with <a> != 0 (jx > 0 copy)."""
    wc, wa, k2, g = o["wc"], o["wa"], o["kappa2"], o["gamma"]
    if 2 * lam <= abs(wc):                    # lobes need |2 lam jx| > |wc| and |jx| <= 1
        return []
    c2 = (2 * g / wa) ** 2
    w = (wc / (2 * lam)) ** 2
    zmax = np.sqrt((1 - w) / (1 + w * c2))

    def F(jz):                                # djy = 0 after eliminating alpha and jy
        jx2 = (1 - jz ** 2) / (1 + c2 * jz ** 2)
        r2 = np.sqrt(np.maximum(4 * lam ** 2 * jx2 - wc ** 2, 0.0)) / k2
        return wa * jx2 + 2 * wc * r2 * jz + 4 * chi * jx2 * jz + wa * c2 * jz ** 2 * jx2

    z = -zmax * np.linspace(1e-9, 1 - 1e-9, n)
    f = F(z)
    out = []
    for i in np.nonzero(np.sign(f[:-1]) != np.sign(f[1:]))[0]:
        jz = brentq(F, z[i], z[i + 1], xtol=1e-15)
        jx2 = (1 - jz ** 2) / (1 + c2 * jz ** 2)
        jx = np.sqrt(jx2)
        r2 = np.sqrt(4 * lam ** 2 * jx2 - wc ** 2) / k2
        al = np.sqrt(r2 * (-wc - 1j * k2 * r2) / (2 * lam * jx))
        y = np.array([al.real, al.imag, jx, 2 * g * jz * jx / wa, jz])
        if np.abs(mf_rhs(y, chi, lam, o)).max() < 1e-7:
            out.append(y)
    return out


def mf_phase(chi, lam, o):
    """Stable mean-field states at (chi, lam) and the photon number per spin of the lobes."""
    q = mf_Q(chi, o)
    S = [y for y in mf_S(chi, lam, o) if mf_stable(y, chi, lam, o)]
    return dict(N=mf_stable(np.array([0, 0, 0, 0, -1.0]), chi, lam, o),
                Q=q is not None and mf_stable(q, chi, lam, o),
                S=len(S) > 0, nS=max((y[0] ** 2 + y[1] ** 2 for y in S), default=np.nan))


def mf_lam_S(chi, o, hi=1.0):
    """Lowest lambda with a stable lobed state at this chi (bisection)."""
    lo = abs(o["wc"]) / 2
    if not mf_phase(chi, hi, o)["S"]:
        return np.nan
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if mf_phase(chi, mid, o)["S"] else (mid, hi)
    return hi


def mf_markers(V, o, n=150):
    """Mean-field transitions along the path: list of (u, kind); leg k covers u in [k, k+1]."""
    u = np.linspace(0, len(V) - 1, n * (len(V) - 1) + 1)
    chi = np.interp(u, np.arange(len(V)), V[:, 0])
    lam = np.interp(u, np.arange(len(V)), V[:, 1])
    st = [mf_phase(c, l, o) for c, l in zip(chi, lam)]
    out = []
    for key, on, off in (("Q", "spins order (weak Z4 breaks)", "Q unstable"),
                         ("S", "lobes appear", "lobes disappear")):
        f = np.array([s[key] for s in st])
        for i in np.nonzero(f[1:] != f[:-1])[0]:
            out.append((0.5 * (u[i] + u[i + 1]), on if f[i + 1] else off))
    return out


def mf_label(chi, lam, o):
    """Stable mean-field phases at (chi, lam): 'N', 'Q', 'S', or 'N|S', 'Q|S' where they coexist."""
    st = mf_phase(chi, lam, o)
    base = [k for k in ("N", "Q") if st[k]]
    return "|".join(base + (["S"] if st["S"] else [])) or "?"


# ------------------------------------------------------------------ cutoff / memory
def ceil8(x):
    return int(np.ceil(x / 8.0) * 8)


def cutoff(N, chi, lam, o):
    """Starting Fock cutoff (multiple of 8).  Where lobes exist or are about to (lam less than
    0.03 below their mean-field onset) it follows their photon number, taken at
    max(lam, onset + 0.02); in the N and Q phases the photon number is O(1) and M_min is used
    (the tail check raises it if needed)."""
    if o["M_fixed"]:
        return int(o["M_fixed"])
    r2 = lobe_photons(chi, lam, o)
    if r2 is None or not np.isfinite(r2):
        return o["M_min"]
    return max(ceil8(o["mfac"] * N * r2 + o["moff"]), o["M_min"])


_LOBES = {}


def lobe_photons(chi, lam, o):
    """n/N of the lobes used by cutoff() at (chi, lam), None where there are none and none are about
    to appear.  Independent of N, so it is computed once per point."""
    key = (round(chi, 12), round(lam, 12), o["wc"], o["wa"], o["kappa2"], o["gamma"])
    if key not in _LOBES:
        ls = mf_lam_S(chi, o)
        _LOBES[key] = (None if not np.isfinite(ls) or lam < ls - 0.03
                       else mf_phase(chi, max(lam, ls + 0.02), o)["nS"])
    return _LOBES[key]


def est_peak_gb(N, M):
    """Peak memory of one evaluation: LU of 2.2 GB measured at block size K = 78408 (N=10, M=72,
    this model; ~100 s), growing like K^1.5, times 2.5 for SuperLU work space as in
    sweep_traceless.py.  Run time grows roughly like K^2."""
    K = (M / 2 * (N + 1)) ** 2 / 2
    return 2.5 * 2.2 * (K / 78408.0) ** 1.5


# ------------------------------------------------------------------ sector Liouvillian
def nd_perm(coords, reach, leaf=64):
    """Geometric nested dissection; a Liouvillian term moves coordinate d by at most reach[d]
    (photon indices by 1, spin indices by 2 because of Jx^2), so separators are reach[d] wide."""
    order = []

    def rec(sub):
        if len(sub) == 0:
            return
        c = coords[sub]
        lo, hi = c.min(0), c.max(0)
        d = int(np.argmax((hi - lo) / reach))
        if len(sub) <= leaf or hi[d] - lo[d] < reach[d] + 1:
            order.append(sub)
            return
        mid = (lo[d] + hi[d]) // 2
        rec(sub[c[:, d] < mid])
        rec(sub[c[:, d] >= mid + reach[d]])
        order.append(sub[(c[:, d] >= mid) & (c[:, d] < mid + reach[d])])

    rec(np.arange(len(coords)))
    return np.concatenate(order)


def parity_space(N, M, parity, o):
    """Operators on the Fock states of one photon parity (x spin j = N/2, m ascending)."""
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
    H0 = (o["wc"] * sp.kron(sp.diags(ns.astype(float)), Is) + o["wa"] * sp.kron(Ip, sp.diags(ms))).tocsr()
    H1 = ((2.0 / N) * sp.kron(A2 + A2.T, Sx)).tocsr()          # x lam: (1/N) Jx (a^2 + a^+2)
    H2 = (-(4.0 / N) * sp.kron(Ip, Sx @ Sx)).tocsr()            # x chi: -(1/N) Jx^2
    jumps = [np.sqrt(o["kappa2"] / N) * sp.kron(A2, Is)]
    if o["gamma"] > 0:
        jumps.append(np.sqrt(o["gamma"] / N) * 2.0 * sp.kron(Ip, Splus.T))   # sqrt(gamma/N) J_-
    jumps = [(L.tocsr(), (L.conj().T @ L).tocsr()) for L in jumps]
    nn = np.repeat(ns, ds).astype(float)
    mm = np.tile(ms, dn)
    return dict(H0=H0, H1=H1, H2=H2, jumps=jumps, D=dn * ds, dn=dn,
                phase=np.exp(1j * np.pi * (nn / 2 + mm)),
                nidx=np.repeat(np.arange(dn), ds), sidx=np.tile(np.arange(ds), dn))


class SectorBlock:
    """L(chi, lam) = L0 + lam * L1 + chi * L2 on the operators |ket><bra| of one sector."""

    def __init__(self, N, M, sector, o):
        pr, pc, z = parse_sector(sector)
        R = parity_space(N, M, pr, o)
        C = R if pc == pr else parity_space(N, M, pc, o)
        Dr, Dc = R["D"], C["D"]
        Ir, Ic = sp.identity(Dr, format="csr"), sp.identity(Dc, format="csr")
        # column stacking (as QuTiP): vec(A rho B) = (B^T kron A) vec(rho), index = col*Dr + row
        comm = lambda X: (-1j * (sp.kron(Ic, R[X]) - sp.kron(C[X].T, Ir))).tocsr()
        L0 = comm("H0")
        for (LR, LdLR), (LC, LdLC) in zip(R["jumps"], C["jumps"]):
            L0 = L0 + sp.kron(LC.conj(), LR) - 0.5 * sp.kron(Ic, LdLR) - 0.5 * sp.kron(LdLC.T, Ir)
        L0, L1, L2 = L0.tocsr(), comm("H1"), comm("H2")
        rowv = np.tile(np.arange(Dr), Dc)
        colv = np.repeat(np.arange(Dc), Dr)
        keep = np.flatnonzero(np.isclose(R["phase"][rowv] * np.conj(C["phase"][colv]), z, atol=1e-8))
        coords = np.stack([R["nidx"][rowv[keep]], R["sidx"][rowv[keep]],
                           C["nidx"][colv[keep]], C["sidx"][colv[keep]]], axis=1)
        sel = keep[nd_perm(coords, np.array([1, 2, 1, 2]))]
        self.L0 = L0[sel][:, sel].tocsc()
        self.L1 = L1[sel][:, sel].tocsc()
        self.L2 = L2[sel][:, sel].tocsc()
        rs, cs = rowv[sel], colv[sel]
        self.edge = (R["nidx"][rs] >= R["dn"] - 3) | (C["nidx"][cs] >= C["dn"] - 3)
        self.K = len(sel)
        self.trace = (pr == pc and z == 1.0)


_CACHE = OrderedDict()


def get_block(N, M, sector, o):
    key = (N, M, sector)
    if key not in _CACHE:
        _CACHE[key] = SectorBlock(N, M, sector, o)
        while len(_CACHE) > 2:
            _CACHE.popitem(last=False)
    return _CACHE[key]


def solve_point(N, sector, chi, lam, M, o):
    """Slowest eigenvalues of one sector at one (chi, lam) and cutoff M."""
    B = get_block(N, M, sector, o)
    Lp = (B.L0 + lam * B.L1 + chi * B.L2).tocsc()
    K = Lp.shape[0]
    ev = np.full(o["k"], np.nan, dtype=complex)
    try:
        lu = splu((Lp - o["sigma"] * sp.identity(K, format="csc")).tocsc(), permc_spec="NATURAL",
                  diag_pivot_thresh=0.1, options=dict(SymmetricMode=True))
    except RuntimeError as e:
        print(f"  N={N} {sector} chi={chi:.4f} lam={lam:.4f} M={M}: FACTORIZATION ERROR {e}", flush=True)
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
    if B.trace and len(vals):                 # drop the steady state (exact zero, trace preservation)
        i0 = int(np.argmin(np.abs(vals)))
        vals, vecs = np.delete(vals, i0), np.delete(vecs, i0, axis=1)
    order = np.argsort(-vals.real)
    vals, vecs = vals[order], vecs[:, order]
    tail = np.nan
    if len(vals):
        idx = {0, int(np.argmin(np.abs(vals)))}           # leading and closest-to-zero mode
        tail = max(float(np.sum(np.abs(vecs[B.edge, i]) ** 2) / np.sum(np.abs(vecs[:, i]) ** 2))
                   for i in idx)
    ev[:len(vals)] = vals
    return ev, tail, K, nnz


def evaluate(task):
    """One (N, sector, path point).  If the slow eigenmatrices have too much weight in the top
    Fock states, the cutoff is raised (x1.3) and the point recomputed, up to M_cap."""
    N, sector, u, chi, lam, M, M_cap, o = task
    t0 = time.time()
    while True:
        ev, tail, K, nnz = solve_point(N, sector, chi, lam, M, o)
        if tail <= o["tail_tol"] or M >= M_cap or not np.isfinite(tail):
            break
        M = min(M_cap, ceil8(1.3 * M))
    secs = time.time() - t0
    absgap = float(np.nanmin(np.abs(ev))) if np.isfinite(ev).any() else np.nan
    print(f"  N={N} {sector:8s} chi={chi:.4f} lam={lam:.4f} M={M:3d} K={K:7d} "
          f"LU={nnz * 16 / 1e9:6.2f} GB gap={-ev[0].real:.3e} (Im {ev[0].imag:+.2e}) "
          f"min|ev|={absgap:.3e} tail={tail:.1e} [{secs:.0f}s]", flush=True)
    return dict(N=N, sector=sector, u=u, chi=chi, lam=lam, M=M, K=K, evals=ev, tail=tail,
                nnz=nnz, secs=secs)


# ------------------------------------------------------------------ driver
def model_opts(a):
    return dict(wc=a.wc, wa=a.wa, kappa2=a.kappa2, gamma=a.gamma, mfac=a.mfac, moff=a.moff,
                M_min=a.M_min, M_fixed=a.M_fixed, sigma=a.sigma, k=a.k, tol=a.tol,
                tail_tol=a.tail_tol)


def parse_path(items):
    V = np.array([[float(x) for x in s.split(",")] for s in items])
    if V.ndim != 2 or V.shape[1] != 2 or len(V) < 2:
        raise SystemExit("--path needs at least two corners written as chi,lam")
    return V


def path_points(V, n_steps):
    """n_steps[k] points on leg k (a single value is used for every leg; corners shared).
    Returns u (leg k covers [k, k+1]), chi, lam."""
    legs = len(V) - 1
    steps = list(n_steps) * legs if len(n_steps) == 1 else list(n_steps)
    if len(steps) != legs:
        raise SystemExit(f"--n-steps needs 1 or {legs} values for a path with {legs} legs")
    u = np.concatenate([k + np.linspace(0, 1, steps[k])[(1 if k else 0):] for k in range(legs)])
    X = np.arange(len(V))
    return u, np.interp(u, X, V[:, 0]), np.interp(u, X, V[:, 1])


def sweep_points(a):
    """Path corners V and points (u, chi, lam).  With --chis: chi alone at fixed --lam, any spacing,
    u = position between the smallest and the largest chi.  Otherwise --path with --n-steps."""
    if a.chis:
        if a.lam is None:
            raise SystemExit("--chis needs --lam (the fixed lambda of the chi sweep)")
        chis = np.unique(np.asarray(a.chis, float))
        if len(chis) < 2:
            raise SystemExit("--chis needs at least two different values")
        V = np.array([[chis[0], a.lam], [chis[-1], a.lam]])
        return V, (chis - chis[0]) / (chis[-1] - chis[0]), chis, np.full(len(chis), float(a.lam))
    V = parse_path(a.path)
    u, chis, lams = path_points(V, a.n_steps)
    return V, u, chis, lams


def sectors_to_compute(a):
    """(sectors to diagonalise, {E_-+ sector: computed E_+- partner filled in by conjugation})."""
    if a.sectors:
        bad = [s for s in a.sectors if s not in TRACELESS + TRACE]
        if bad:
            raise SystemExit(f"unknown sector(s) {bad}; choose from {TRACELESS + TRACE}")
        todo = list(dict.fromkeys(a.sectors))
    else:
        todo = TRACELESS + (TRACE if a.with_trace else [])
    if a.no_mirror:
        return todo, {}
    mirrored = {s: src for s, src in MIRROR.items() if src in todo}
    return [s for s in todo if s not in mirrored], mirrored


def same_run(files, V, chis, lams, o):
    """True if all output files exist and were made for the same path, points and model."""
    try:
        if not all(os.path.exists(f) for f in files):
            return False
        d = np.load(files[0])
        return (d["path"].shape == V.shape and np.allclose(d["path"], V)
                and len(d["chi_vals"]) == len(chis) and np.allclose(d["chi_vals"], chis)
                and np.allclose(d["lam_vals"], lams)
                and all(np.isclose(float(d[key]), o[key]) for key in ("wc", "wa", "kappa2", "gamma")))
    except Exception:
        return False


POINT_KEYS = ("wc", "wa", "kappa2", "gamma", "mfac", "moff", "M_min", "M_fixed", "sigma", "k", "tol",
              "tail_tol")


def point_dir(outdir, o):
    """Folder of single evaluations; the name encodes everything that changes their result."""
    h = hashlib.md5(json.dumps({k: o[k] for k in POINT_KEYS}, sort_keys=True).encode()).hexdigest()[:8]
    return os.path.join(outdir, "data", f"points_{h}")


def point_file(pdir, N, sector, chi, lam):
    return os.path.join(pdir, f"N{N}_{sector}_chi{chi:.6f}_lam{lam:.6f}.npz")


def evaluate_and_store(job):
    """evaluate() in a worker, saved to its point file as soon as it is done (atomic rename)."""
    task, fn = job
    r = evaluate(task)
    tmp = fn[:-4] + ".part.npz"
    np.savez(tmp, **{key: np.asarray(val) for key, val in r.items()})
    os.replace(tmp, fn)
    return r


def load_point(fn, u):
    d = np.load(fn)
    r = dict(N=int(d["N"]), sector=str(d["sector"]), u=float(u), chi=float(d["chi"]), lam=float(d["lam"]),
             M=int(d["M"]), K=int(d["K"]), evals=np.array(d["evals"]), tail=float(d["tail"]),
             nnz=int(d["nnz"]), secs=float(d["secs"]))
    return r


def slurm_mem_gb():
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if os.environ.get("SLURM_MEM_PER_CPU"):
        return float(os.environ["SLURM_MEM_PER_CPU"]) * cpus / 1024.0
    if os.environ.get("SLURM_MEM_PER_NODE"):
        return float(os.environ["SLURM_MEM_PER_NODE"]) / 1024.0
    return None


def save_sector(path, rs, extra=None):
    rs = sorted(rs, key=lambda r: r["u"])
    ev = np.array([r["evals"] for r in rs])
    with np.errstate(all="ignore"):
        absgap = np.array([np.nanmin(np.abs(e)) if np.isfinite(e).any() else np.nan for e in ev])
    np.savez(path, u_vals=np.array([r["u"] for r in rs]), chi_vals=np.array([r["chi"] for r in rs]),
             lam_vals=np.array([r["lam"] for r in rs]), gap=-ev[:, 0].real, absgap=absgap,
             leading_evals=ev[:, 0], all_evals=ev, M_vals=np.array([r["M"] for r in rs]),
             K_vals=np.array([r["K"] for r in rs]), tail=np.array([r["tail"] for r in rs]),
             lu_nnz=np.array([r["nnz"] for r in rs]), seconds=np.array([r["secs"] for r in rs]),
             **(extra or {}))


def run(a):
    o = model_opts(a)
    datadir = os.path.join(a.outdir, "data")
    pdir = point_dir(a.outdir, o)
    os.makedirs(pdir, exist_ok=True)
    V, u, chis, lams = sweep_points(a)
    todo, mirrored = sectors_to_compute(a)
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    mem = slurm_mem_gb() or a.mem_gb
    info = dict(params=vars(a), model=o, chi_c_mf=chi_c(o), path=V.tolist(),
                points=dict(chi=chis.tolist(), lam=lams.tolist()), sectors=todo, mirrored=mirrored,
                mf_markers=[(float(x), k) for x, k in mf_markers(V, o)], runs=[])
    print(f"path {' -> '.join(f'({c:g}, {l:g})' for c, l in V)} ({len(u)} pts), N = {a.Ns}, "
          f"gamma = {a.gamma:g}, kappa2 = {a.kappa2:g}, sectors computed: {todo}"
          f"{f' (+ {list(mirrored)} by conjugation)' if mirrored else ''}, cpus={cpus}, "
          f"memory budget={mem:.0f} GB, mean-field chi_c={info['chi_c_mf']:.4f}", flush=True)
    for x, k in info["mf_markers"]:
        c, l = np.interp(x, np.arange(len(V)), V[:, 0]), np.interp(x, np.arange(len(V)), V[:, 1])
        print(f"  mean field: {k} at (chi, lam) = ({c:.3f}, {l:.3f})", flush=True)
    print("  points (mean-field phase): " + "  ".join(f"({c:.3f}, {l:.3f}) {mf_label(c, l, o)}"
                                                       for c, l in zip(chis, lams)), flush=True)
    for N in sorted(a.Ns):
        outs = [os.path.join(datadir, f"gap_{s}_N{N}.npz") for s in todo + list(mirrored)]
        if not a.overwrite and same_run(outs, V, chis, lams, o):
            print(f"\nN={N}: finished data for these points already in {datadir}, skipped "
                  f"(--overwrite recomputes)", flush=True)
            continue
        Ms = [cutoff(N, c, l, o) for c, l in zip(chis, lams)]
        M_cap = a.M_fixed or max(max(Ms), ceil8(1.5 * a.M_min))   # retries stay within the memory estimate
        est = est_peak_gb(N, M_cap)
        workers = a.workers or max(1, min(cpus, int(0.85 * mem / est)))
        res, jobs = [], []
        for s in todo:
            for x, c, l, M in zip(u, chis, lams, Ms):
                fn = point_file(pdir, N, s, c, l)
                if not a.overwrite and os.path.exists(fn):
                    try:
                        res.append(load_point(fn, x))
                        continue
                    except Exception:
                        pass                                # unreadable: compute it again
                jobs.append(((N, s, float(x), float(c), float(l), M, M_cap, o), fn))
        jobs.sort(key=lambda job: -job[0][5])               # largest cutoff first
        workers = min(workers, max(1, len(jobs)))
        print(f"\nN={N}: cutoff M from {min(Ms)} to {max(Ms)} (retries up to {M_cap}), ~{est:.1f} GB peak "
              f"per evaluation -> {workers} parallel workers; {len(res)} evaluations from earlier runs, "
              f"{len(jobs)} to compute", flush=True)
        t0 = time.time()
        if jobs:
            try:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    for f in as_completed([ex.submit(evaluate_and_store, job) for job in jobs]):
                        res.append(f.result())
            except BrokenProcessPool:
                print(f"\nN={N}: a worker process died, most likely killed for exceeding its memory "
                      f"(estimate {est:.1f} GB per evaluation).  Finished evaluations are kept in {pdir}; "
                      f"resubmit with a smaller --workers (e.g. --workers {max(1, workers // 2)}).",
                      flush=True)
                info["runs"].append(dict(N=N, workers=workers, aborted="worker died"))
                json.dump(info, open(os.path.join(a.outdir, "run_info.json"), "w"), indent=1, default=str)
                sys.exit(3)
        extra = dict(N=N, path=V, wc=a.wc, wa=a.wa, kappa2=a.kappa2, gamma=a.gamma)
        for s in todo:
            save_sector(os.path.join(datadir, f"gap_{s}_N{N}.npz"), [r for r in res if r["sector"] == s],
                        dict(extra, sector=s, mirrored_from=""))
        for s, src in mirrored.items():
            rs = [dict(r, evals=np.conj(r["evals"])) for r in res if r["sector"] == src]
            save_sector(os.path.join(datadir, f"gap_{s}_N{N}.npz"), rs,
                        dict(extra, sector=s, mirrored_from=src))
        bad = [(r["sector"], round(r["chi"], 4), round(r["lam"], 4)) for r in res
               if not (r["tail"] <= a.tail_tol)]
        info["runs"].append(dict(N=N, workers=workers, M_min=min(Ms), M_max=max(Ms), computed=len(jobs),
                                 seconds=time.time() - t0, points_above_tail_tol=bad))
        print(f"N={N} done in {time.time() - t0:.0f}s; points with cutoff tail > {a.tail_tol:g}: "
              f"{len(bad)}", flush=True)
        json.dump(info, open(os.path.join(a.outdir, "run_info.json"), "w"), indent=1, default=str)
    plot(a)


# ------------------------------------------------------------------ plotting
MSTY = {"spins order (weak Z4 breaks)": ("#52514e", "--"), "lobes appear": ("#eb6834", "--"),
        "Q unstable": ("#2a78d6", "--"), "lobes disappear": ("#eb6834", ":")}


def plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datadir = os.path.join(a.outdir, "data")
    files = sorted(glob.glob(os.path.join(datadir, "gap_*_N*.npz")))
    if not files:
        raise SystemExit(f"no data in {datadir}")
    data = {}
    for f in files:
        d = dict(np.load(f, allow_pickle=True))
        data[(str(d["sector"]), int(d["N"]))] = d
    Ns = sorted({N for (_, N) in data})
    have = {s for (s, _) in data}
    layout = [row for row in LAYOUT + [TRACE] if any(s in have for s in row)]
    d0 = next(iter(data.values()))
    V = np.asarray(d0["path"], float)
    o = dict(wc=float(d0["wc"]), wa=float(d0["wa"]), kappa2=float(d0["kappa2"]), gamma=float(d0["gamma"]))
    marks = mf_markers(V, o)
    X = np.arange(len(V))
    ticks = None
    if len(V) == 2 and V[0, 0] == V[1, 0]:
        xof = lambda uu: np.interp(uu, X, V[:, 1])
        xlabel = rf"$\lambda$   ($\chi = {V[0, 0]:g}$)"
    elif len(V) == 2 and V[0, 1] == V[1, 1]:
        xof = lambda uu: np.interp(uu, X, V[:, 0])
        xlabel = rf"$\chi$   ($\lambda = {V[0, 1]:g}$)"
    else:
        xof = lambda uu: np.asarray(uu, float)
        xlabel = r"path through $(\chi, \lambda)$"
        ticks = (X, [f"({c:g}, {l:g})" for c, l in V])
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(Ns)))
    path_txt = " $\\to$ ".join(f"({c:g}, {l:g})" for c, l in V)
    for key, ylab, scales, stem in (
            ("gap", r"$-\mathrm{Re}\,\lambda$ (slowest mode)", ("log", "lin"), "gap_closure"),
            ("absgap", r"$\min|\lambda|$ (distance from zero)", ("log",), "zero_mode_distance")):
        for scale in scales:
            fig, axes = plt.subplots(len(layout), 2, figsize=(10.5, 3.5 * len(layout) + 0.8),
                                     sharex=True, squeeze=False)
            for i, row in enumerate(layout):
                for jj, s in enumerate(row):
                    ax = axes[i, jj]
                    if s not in have:
                        ax.set_visible(False)
                        continue
                    for N, c in zip(Ns, cols):
                        if (s, N) not in data:
                            continue
                        d = data[(s, N)]
                        x, y, tail = xof(d["u_vals"]), d[key], d["tail"]
                        ok = tail <= a.tail_tol
                        ax.plot(x, y, "-", color=c, lw=1.5, label=f"N = {N}")
                        ax.plot(x[ok], y[ok], "o", color=c, ms=3.2)
                        ax.plot(x[~ok], y[~ok], "o", mfc="white", mec=c, ms=4.2)
                    for uu, kind in marks:
                        col, ls = MSTY.get(kind, ("gray", ":"))
                        ax.axvline(float(xof(uu)), color=col, ls=ls, lw=1.1)
                    if ticks is not None:
                        for xv in ticks[0][1:-1]:
                            ax.axvline(xv, color="black", ls=":", lw=0.8)
                    if scale == "log":
                        ax.set_yscale("log")
                    title = "Sector " + sector_label(s)
                    if s in MIRROR and str(data.get((s, Ns[0]), {}).get("mirrored_from", "")):
                        title += "   (= c.c. of " + sector_label(MIRROR[s]) + ")"
                    ax.set_title(title, fontsize=11)
                    ax.grid(True, which="both", ls="--", alpha=0.35)
                    if jj == 0:
                        ax.set_ylabel(ylab)
                    if i == len(layout) - 1:
                        ax.set_xlabel(xlabel)
                        if ticks is not None:
                            ax.set_xticks(ticks[0])
                            ax.set_xticklabels(ticks[1], fontsize=9)
            ax0 = next(ax for ax in axes.flat if ax.get_visible())
            for kind in dict.fromkeys(k for _, k in marks):
                col, ls = MSTY.get(kind, ("gray", ":"))
                ax0.plot([], [], ls=ls, color=col, lw=1.1, label="MF: " + kind)
            ax0.legend(fontsize=8.5, loc="best", framealpha=0.9)
            fig.suptitle(rf"$-(\chi/N)J_x^2$, $\gamma$={o['gamma']:g}, $\omega_c$={o['wc']:g}, "
                         rf"$\omega_a$={o['wa']:g}, $\kappa_2$={o['kappa2']:g};  path {path_txt}", fontsize=11.5)
            fig.tight_layout()
            fn = os.path.join(a.outdir, f"{stem}_{scale}.png")
            fig.savefig(fn, dpi=200)
            plt.close(fig)
            print("plot ->", fn, flush=True)
    plot_scaling(a, data, o)


def plot_scaling(a, data, o):
    """Smallest |eigenvalue| against N at selected points: gap_scaling.png and summary.txt.
    Filled markers / no '~': the mode does not oscillate (|Im| <= 0.3 |ev|)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    Ns = sorted({N for (_, N) in data})
    secs = [s for s in TRACELESS + TRACE
            if any(key[0] == s and not str(d.get("mirrored_from", "")) for key, d in data.items())]
    if not secs:
        return
    ref = max(data.values(), key=lambda d: len(d["u_vals"]))
    pts = np.stack([ref["chi_vals"], ref["lam_vals"]], axis=1)
    if a.scaling_chis:
        idx = list(dict.fromkeys(int(np.argmin(np.abs(pts[:, 0] - c))) for c in a.scaling_chis))
    elif len(pts) <= 6:
        idx = list(range(len(pts)))
    else:
        idx = list(np.unique(np.round(np.linspace(0, len(pts) - 1, 6)).astype(int)))
    pts = pts[idx]
    labels = [mf_label(c, l, o) for c, l in pts]
    cols = plt.cm.plasma(np.linspace(0.0, 0.85, len(pts)))
    lines = [f"smallest |eigenvalue| per sector against N   (gamma={o['gamma']:g}, kappa2={o['kappa2']:g}, "
             f"wc={o['wc']:g}, wa={o['wa']:g})",
             f"  '~' the mode oscillates (|Im| > 0.3 |ev|), '*' cutoff tail above {a.tail_tol:g}; "
             f"[phase] = stable mean-field states; slope = d ln|ev| / d ln N over the last two N"]
    fig, axes = plt.subplots(len(secs), 2, figsize=(11, 3.4 * len(secs) + 0.9), squeeze=False)
    for i, s in enumerate(secs):
        lines.append(f"\n{s}")
        firsts = []
        for (c, l), lab, col in zip(pts, labels, cols):
            ser = []
            for N in Ns:
                d = data.get((s, N))
                if d is None:
                    continue
                hit = np.flatnonzero(np.isclose(d["chi_vals"], c, atol=1e-9)
                                     & np.isclose(d["lam_vals"], l, atol=1e-9))
                if not len(hit) or not np.isfinite(d["all_evals"][hit[0]]).any():
                    continue
                e = d["all_evals"][hit[0]]
                k = int(np.nanargmin(np.abs(e)))
                ser.append((N, abs(e[k]), abs(e[k].imag), float(d["tail"][hit[0]])))
            if not ser:
                continue
            n_, g_, im_, t_ = map(np.array, zip(*ser))
            osc = im_ > 0.3 * g_
            firsts.append(g_[0] * n_[0])
            for ax in axes[i]:
                ax.plot(n_, g_, "-", color=col, lw=1.4, label=f"$\\chi$={c:g}, $\\lambda$={l:g} [{lab}]")
                ax.plot(n_[~osc], g_[~osc], "o", color=col, ms=5)
                ax.plot(n_[osc], g_[osc], "o", mfc="white", mec=col, ms=5.5)
            slope = np.log(g_[-1] / g_[-2]) / np.log(n_[-1] / n_[-2]) if len(n_) > 1 else np.nan
            vals = "  ".join(f"N={int(n)}: {g:.3e}{'~' if w else ' '}{'*' if not (t <= a.tail_tol) else ' '}"
                             for n, g, w, t in zip(n_, g_, osc, t_))
            lines.append(f"  chi={c:.4f} lam={l:.4f} [{lab:>3s}]  {vals}  slope {slope:+.2f}")
        if firsts:
            nn = np.array(Ns, float)
            axes[i, 0].plot(nn, np.exp(np.mean(np.log(firsts))) / nn, ":", color="gray", lw=1.1,
                            label=r"$\propto 1/N$")
        for jj, ax in enumerate(axes[i]):
            ax.set_yscale("log")
            if jj == 0:
                ax.set_xscale("log")
                ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticks(Ns)
            ax.set_xticklabels([str(N) for N in Ns])
            ax.set_title(f"Sector {sector_label(s)}   ({'log-log' if jj == 0 else 'log-lin'})", fontsize=11)
            ax.grid(True, which="both", ls="--", alpha=0.35)
            ax.set_xlabel("N")
        axes[i, 0].set_ylabel(r"smallest $|\lambda|$")
        if i == 0:
            axes[i, 0].legend(fontsize=8, loc="best", framealpha=0.9)
    fig.suptitle(rf"$\gamma$={o['gamma']:g}, $\kappa_2$={o['kappa2']:g}, $\omega_c$={o['wc']:g}, "
                 rf"$\omega_a$={o['wa']:g}.  Filled: slowest mode not oscillating ($|\mathrm{{Im}}| \leq 0.3|\lambda|$), "
                 "open: oscillating", fontsize=10.5)
    fig.tight_layout()
    fn = os.path.join(a.outdir, "gap_scaling.png")
    fig.savefig(fn, dpi=200)
    plt.close(fig)
    print("plot ->", fn, flush=True)
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(a.outdir, "summary.txt"), "w") as fh:
        fh.write(txt)
    print("\n" + txt, flush=True)


# ------------------------------------------------------------------ self-check
def selfcheck(a):
    """All 8 sectors of the block construction vs the full-space Liouvillian (spin basis +j..-j,
    masks from Pi_2, Pi_4), including the chi and gamma terms; also checks that the full
    Liouvillian has no matrix element between different sectors."""
    o = model_opts(a)
    allsec = ["E_++_1", "E_++_-1", "E_--_1", "E_--_-1", "E_+-_i", "E_+-_-i", "E_-+_i", "E_-+_-i"]
    worst = 0.0
    for N, M, chi, lam in [(3, 16, 0.5, 0.55), (4, 14, 0.7, 0.50)]:
        j, ds = N / 2.0, N + 1
        m = np.arange(j, -j - 1, -1)
        Jp = sp.diags([np.sqrt(j * (j + 1) - m[1:] * (m[1:] + 1))], [1], shape=(ds, ds))
        Jx, Jz, Jm = (Jp + Jp.T) / 2, sp.diags(m), Jp.T
        a1 = sp.diags([np.sqrt(np.arange(1, M))], [1], shape=(M, M))
        aa = sp.kron(a1, sp.identity(ds))
        IM = sp.identity(M)
        num = sp.kron(sp.diags(np.arange(M) * 1.0), sp.identity(ds))
        JxH, JzH, JmH = 2 * sp.kron(IM, Jx), 2 * sp.kron(IM, Jz), 2 * sp.kron(IM, Jm)
        H = (o["wc"] * num + (o["wa"] / 2) * JzH + (lam / N) * JxH @ (aa.T @ aa.T + aa @ aa)
             - (chi / N) * JxH @ JxH).tocsr()
        Ls = [(np.sqrt(o["kappa2"] / N) * aa @ aa).tocsr(), (np.sqrt(o["gamma"] / N) * JmH).tocsr()]
        D = M * ds
        I = sp.identity(D)
        Lf = -1j * (sp.kron(I, H) - sp.kron(H.T, I))
        for L in Ls:
            LdL = L.conj().T @ L
            Lf = Lf + sp.kron(L.conj(), L) - 0.5 * sp.kron(I, LdL) - 0.5 * sp.kron(LdL.T, I)
        Lf = Lf.tocsr()
        dn_ = np.repeat(np.arange(M), ds)
        dz = np.tile(m, M)
        P2 = np.exp(1j * np.pi * dn_).real
        P4 = np.exp(1j * np.pi * (dn_ / 2 + dz))
        P2o, P4o = sp.diags(P2), sp.diags(P4)
        nrm = lambda X: abs(X).max() if X.nnz else 0.0
        sym = [nrm(P2o @ H - H @ P2o), max(nrm(P2o @ L - L @ P2o) for L in Ls),
               nrm(P4o @ H @ P4o.conj() - H), max(nrm(P4o @ L @ P4o.conj() + L) for L in Ls)]
        worst = max(worst, *sym)
        print(f"N={N} M={M} chi={chi} lam={lam} gamma={o['gamma']} kappa2={o['kappa2']}:  |[Pi2,H]|={sym[0]:.1e}  "
              f"|[Pi2,L]|={sym[1]:.1e}  |Pi4 H Pi4^+ - H|={sym[2]:.1e}  |Pi4 L Pi4^+ + L|={sym[3]:.1e}", flush=True)
        rows, cols = np.tile(np.arange(D), D), np.repeat(np.arange(D), D)
        spectra = {}
        for s in allsec:
            pr, pc, z = parse_sector(s)
            mask = (((P2[rows] > 0) if pr == 0 else (P2[rows] < 0))
                    & ((P2[cols] > 0) if pc == 0 else (P2[cols] < 0))
                    & np.isclose(P4[rows] * np.conj(P4[cols]), z, atol=1e-8))
            leak = nrm(Lf[~mask][:, mask])
            ev_f = np.linalg.eigvals(Lf[mask][:, mask].toarray())
            B = SectorBlock(N, M, s, o)
            ev_b = np.linalg.eigvals((B.L0 + lam * B.L1 + chi * B.L2).toarray())
            spectra[s] = ev_b
            dev = max(max(np.min(np.abs(ev_f - x)) for x in ev_b), max(np.min(np.abs(ev_b - x)) for x in ev_f))
            worst = max(worst, dev, leak, 0.0 if len(ev_b) == len(ev_f) else np.inf)
            print(f"   {s:8s} dim {len(ev_b):5d} vs {len(ev_f):5d}  max eigenvalue mismatch {dev:.1e}  "
                  f"coupling into other sectors {leak:.1e}", flush=True)
        for s, src in MIRROR.items():
            dev = max(np.min(np.abs(spectra[s] - np.conj(x))) for x in spectra[src])
            worst = max(worst, dev)
            print(f"   mirror check: spectrum({s}) = conj(spectrum({src})) up to {dev:.1e}", flush=True)
    if worst > 1e-8:
        raise SystemExit(f"SELFCHECK FAILED (worst deviation {worst:.1e})")
    print(f"selfcheck passed (worst deviation {worst:.1e})", flush=True)


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Ns", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    ap.add_argument("--path", nargs="+", default=["0.10,0.47", "0.70,0.47", "0.70,0.80"],
                    help="corners chi,lam of a piecewise straight path")
    ap.add_argument("--n-steps", type=int, nargs="+", default=[16, 34],
                    help="points per leg, one value for all legs or one per leg (corners shared)")
    ap.add_argument("--lam", type=float, default=None, help="fixed lambda of a chi sweep (with --chis)")
    ap.add_argument("--chis", type=float, nargs="+", default=None,
                    help="chi values of a sweep at fixed --lam, any spacing (replaces --path / --n-steps)")
    ap.add_argument("--sectors", nargs="+", default=None,
                    help="compute only these sectors (default: the traceless ones, plus E_++_1 and E_--_1 "
                         "with --with-trace); E_-+ sectors are still filled in by conjugation")
    ap.add_argument("--scaling-chis", type=float, nargs="+", default=None,
                    help="chi values for gap_scaling.png and summary.txt (nearest computed points)")
    ap.add_argument("--gamma", type=float, default=0.1, help="collective spin decay sqrt(gamma/N) J_-")
    ap.add_argument("--wc", type=float, default=1.0)
    ap.add_argument("--wa", type=float, default=1.0)
    ap.add_argument("--kappa2", type=float, default=0.5, help="two-photon loss (thesis: 0.05, see docstring)")
    ap.add_argument("--mfac", type=float, default=2.2, help="cutoff = mfac * N * (n/N of the lobes) + moff")
    ap.add_argument("--moff", type=float, default=16)
    ap.add_argument("--M-min", type=int, default=40, help="cutoff where there are no lobes")
    ap.add_argument("--M-fixed", type=int, default=0, help="use one fixed cutoff instead")
    ap.add_argument("--k", type=int, default=6, help="eigenvalues per point")
    ap.add_argument("--sigma", type=float, default=1e-8)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--workers", type=int, default=0, help="0 = choose from memory budget")
    ap.add_argument("--mem-gb", type=float, default=100.0, help="memory budget if not under SLURM")
    ap.add_argument("--tail-tol", type=float, default=1e-7,
                    help="weight of the slow eigenmatrices in the top Fock states above which a point "
                         "is drawn open")
    ap.add_argument("--with-trace", action="store_true", help="also E_++_1 and E_--_1 (steady state removed)")
    ap.add_argument("--no-mirror", action="store_true", help="compute E_-+ sectors explicitly")
    ap.add_argument("--outdir", default="Gap_Closure_LMG")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute N whose finished data for the same path already exist")
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