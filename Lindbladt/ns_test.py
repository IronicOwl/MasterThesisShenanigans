#!/usr/bin/env python3
"""
ns_test.py -- noiseless-subsystem (NS) test for the two-photon Dicke model with
two-photon loss (strong Z2 x weak Z4), following Lieu et al., PRL 125, 240405,
Supplemental Material Sec. 2.

Stages
  solve    For every N and each of the 8 symmetry blocks of L: shift-invert ARPACK
           for the eigenoperators closest to zero.  Parallel over (N, block); every
           block is checkpointed to disk and skipped on restart.
  analyze  Per N (parallel over N):
             1. vacuum / SR separation in E_++_1 and E_--_1 (Rayleigh-Ritz on n)
             2. gauge fixing of all eigenoperators
             3. recombination of weak-Z4 partner blocks into the two compass-qubit
                branches R (<Jx> > 0) and I (<Jx> < 0)
             4. z-matrices and D_t(z++, z--), D_t(z++, z+-), D_t(z++, z-+)
  report   Summary tables, plots, summary.npz

Usage
  python ns_test.py                           # solve + analyze, default N list
  python ns_test.py --N 4 6 8 10 --lam 0.55
  python ns_test.py --stage solve             # e.g. on a big-memory node
  python ns_test.py --stage analyze           # reuse saved blocks

What each symmetry block should contain in the SR phase (thermodynamic limit).
Let |e>, |o> be the even / odd cat of the branch with <Jx> > 0 ("R") and
|e'> = P4|e>, |o'> = P4|o> their weak-Z4 images ("I"):
  E_++_1   vacuum_even  and  (|e><e| + |e'><e'|)/2          <- 2D slow space
  E_++_-1  (|e><e| - |e'><e'|)/2                             <- traceless
  E_--_1   vacuum_odd   and  (|o><o| + |o'><o'|)/2
  E_--_-1  (|o><o| - |o'><o'|)/2
  E_+-_i, E_+-_-i   (|e><o| -/+ i |e'><o'|)/2  (sign set by phase conventions)
  E_-+_i, E_-+_-i   Hermitian conjugates of the E_+- blocks
Sum / difference of weak-Z4 partners isolates one branch:
  R_even = |e><e|  (from the ++ pair),  R_odd = |o><o|  (from the -- pair),
  X_R    = |e><o|  (from the +- pair).
The NS statement is  R_even = |+><+| (x) z,  R_odd = |-><-| (x) z,  X_R = |+><-| (x) z
with one common z, which is what the trace distances test.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import gc
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import cpu_count

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, splu, LinearOperator, ArpackNoConvergence
import qutip as qt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

# =============================================================================
# Parameters
# =============================================================================
WC, WA, KAPPA2 = 1.0, 1.0, 0.05
LAM = 0.55                # must lie above lambda_c = sqrt(wc^2 + wa*kappa2)/2 ~ 0.512
N_LIST = [4, 6, 8, 10]

# Fock cutoff: M = M_FACTOR * N * n_SR(lam) + M_PAD, rounded up to an even number.
# n_SR is the intensive SR photon number from Eq. (46) of your notes (kappa1 = 0).
# The report prints the SR weight in the top 10% of Fock levels; raise M if it
# is not negligible.
M_FACTOR, M_PAD, M_MIN = 1.8, 30, 40
M_OVERRIDE = {}           # e.g. {10: 220}

SIGMA = -1e-5             # shift-invert target; never exactly 0 (L is singular there)
TOL, MAXITER = 1e-10, 5000
K_TRACE, K_OTHER = 4, 3   # eigenpairs requested in trace-carrying / other blocks
# Step 1 is only meaningful if the vacuum is metastable, i.e. the second eigenvalue of
# E_++_1 / E_--_1 is well separated from the third.  "auto": un-mix if
# |lam_3|/|lam_2| >= SEP_MIN, otherwise the steady state is already the SR state.
UNMIX_MODE = "auto"       # "auto" | "always" | "never"
SEP_MIN = 10.0
# SR state entering the branch reconstruction: "unmix" = step 1 output from the E_1
# blocks (your proposal); "jordan" = |S_-1| from the traceless E_-1 blocks, which never
# contain the vacuum.  Both converge to the same state; the report always prints their
# distance ("Jordan" column).
SR_SOURCE = "unmix"
PERMC_SPEC = "COLAMD"     # SuperLU column ordering; MMD_* orderings blew up memory at N=5 in tests
# The (-+) blocks are exact Hermitian conjugates of the (+-) blocks (L preserves
# Hermiticity).  False: obtain them by conjugation (25% cheaper).  True: solve them
# independently as a numerical cross-check (reported as "(-+) pair").
SOLVE_MIRROR = False
CORE_NR = 8
OUTROOT = os.path.join("Plots", "NS_test")

# =============================================================================
# Symmetry sectors
# =============================================================================
# name: (left photon parity, right photon parity, w), weak-Z4 eigenvalue = i**w.
# Parity 0 = even ('+'), 1 = odd ('-').  A matrix unit |n,m><n',m'| carries
# Pi4 (.) Pi4^dag eigenvalue exp(i pi ((n-n')/2 + (m-m'))) = i**w with
# w = (n - n') + (2m - 2m')  mod 4   (integer arithmetic, no float tolerance).
SECTORS = {
    "E_++_1":  (0, 0, 0), "E_++_-1": (0, 0, 2),
    "E_--_1":  (1, 1, 0), "E_--_-1": (1, 1, 2),
    "E_+-_i":  (0, 1, 1), "E_+-_-i": (0, 1, 3),
    "E_-+_i":  (1, 0, 1), "E_-+_-i": (1, 0, 3),
}
TRACE_SECTORS = ("E_++_1", "E_--_1")
# Hermiticity preservation of L maps E_{+-} ∩ E_w onto E_{-+} ∩ E_{w*}.
MIRROR = {"E_-+_i": "E_+-_-i", "E_-+_-i": "E_+-_i"}


# =============================================================================
# Model
# =============================================================================
def n_sr_intensive(lam, wc, wa, k2):
    """Intensive SR photon number <n>/N, Eq. (46) of the notes (kappa1 = 0)."""
    A = 16 * lam**2 * wc**2 * k2**2 + k2**4 * wa**2
    B = 16 * lam**2 * wc**4 + 2 * wa**2 * k2**2 * wc**2 - 64 * lam**4 * wc**2
    C = wa**2 * wc**4
    disc = B**2 - 4 * A * C
    if disc < 0 or B > 0:
        return np.nan
    return float(np.sqrt((-B + np.sqrt(disc)) / (2 * A)))


def choose_M(N, P):
    M = M_OVERRIDE.get(N, int(np.ceil(M_FACTOR * N * P["n_int"] + M_PAD)))
    M = max(M, M_MIN)
    return M + (M % 2)          # even M -> even and odd blocks have equal size


def basis_labels(N, M):
    """Photon number n and 2m for every index of tensor(Fock(M), spin(N/2))."""
    twom = np.rint(2 * np.real(np.asarray(qt.jmat(N / 2, "z").diag()))).astype(np.int64)
    return np.repeat(np.arange(M), N + 1), np.tile(twom, M)


def sector_index(N, M, sec):
    """Positions of `sec` in QuTiP's column-stacked vec(rho), k = i + D*j."""
    pL, pR, w = SECTORS[sec]
    n, tm = basis_labels(N, M)
    mask = ((n[:, None] % 2 == pL) & (n[None, :] % 2 == pR)
            & (((n[:, None] - n[None, :]) + (tm[:, None] - tm[None, :])) % 4 == w))
    return np.flatnonzero(mask.ravel(order="F"))


def build_liouvillian(N, M, P):
    """Same H and jump operator as sweep.py; returns a scipy CSR matrix."""
    j = N / 2.0
    a = qt.tensor(qt.destroy(M), qt.qeye(N + 1))
    num = qt.tensor(qt.num(M), qt.qeye(N + 1))
    Jx = 2 * qt.tensor(qt.qeye(M), qt.jmat(j, "x"))     # paper convention
    Jz = 2 * qt.tensor(qt.qeye(M), qt.jmat(j, "z"))
    H = P["wc"] * num + (P["wa"] / 2.0) * Jz + (P["lam"] / N) * Jx * (a.dag()**2 + a**2)
    L2 = np.sqrt(P["k2"] / N) * a**2
    L = qt.liouvillian(H, [L2])
    data = L.data.as_scipy() if hasattr(L.data, "as_scipy") else L.data
    return sp.csr_matrix(data)


def check_block_structure(P, N=2, M=10):
    """L must map every sector into itself.  Verifies that the masks match the
    vectorisation convention of the installed QuTiP version."""
    L = build_liouvillian(N, M, P).tocsc()
    worst = 0.0
    for sec in SECTORS:
        idx = sector_index(N, M, sec)
        out = np.setdiff1d(np.arange(L.shape[0]), idx)
        leak = abs(L[out][:, idx]).max() if len(out) else 0.0
        worst = max(worst, leak / abs(L).max())
    if worst > 1e-12:
        raise RuntimeError(f"Sector masks do not block-diagonalise L (leak {worst:.2e}). "
                           "Check QuTiP's vectorisation convention.")
    return worst


# =============================================================================
# Paths
# =============================================================================
def run_dir(P):
    return os.path.join(P["outroot"],
                        f"lam{P['lam']:g}_wc{P['wc']:g}_wa{P['wa']:g}_k2{P['k2']:g}")


def raw_path(P, N, M, sec):
    return os.path.join(run_dir(P), "raw", f"N{N}_M{M}_{sec}.npz")


# =============================================================================
# Stage 1: eigensolves (one task per (N, sector))
# =============================================================================
def solve_sector(task):
    N, M, sec, P = task
    out = raw_path(P, N, M, sec)
    if os.path.exists(out):
        return f"[skip]    N={N:3d} M={M:4d} {sec:8s} checkpoint exists"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    L = build_liouvillian(N, M, P)
    idx = sector_index(N, M, sec)
    Lb = L[idx][:, idx].tocsc()
    del L
    gc.collect()
    t_build = time.time() - t0

    nb = Lb.shape[0]
    nkeep = 2 if sec in TRACE_SECTORS else 1
    k = min(K_TRACE if sec in TRACE_SECTORS else K_OTHER, nb - 2)
    err, fill_gb = "", np.nan
    for sig in (P["sigma"], 10.0 * P["sigma"]):
        try:
            # Explicit LU so the fill memory can be reported; this is the memory wall.
            lu = splu((Lb - sig * sp.identity(nb, dtype=complex, format="csc")).tocsc(),
                      permc_spec=P["permc"])
            fill_gb = (lu.L.nnz + lu.U.nnz) * 20 / 1e9      # complex128 + int32 index
            OPinv = LinearOperator((nb, nb), matvec=lu.solve, dtype=complex)
            vals, vecs = eigs(Lb, k=k, sigma=sig, OPinv=OPinv, which="LM",
                              tol=P["tol"], maxiter=P["maxiter"])
            status = "ok"
            break
        except ArpackNoConvergence as e:
            err = f"ARPACK: {len(e.eigenvalues)}/{k} converged"
            if len(e.eigenvalues) >= nkeep:
                vals, vecs, status = e.eigenvalues, e.eigenvectors, "partial"
                break
        except RuntimeError as e:
            err = f"factorization error: {e}"
        finally:
            lu = OPinv = None
            gc.collect()
    else:
        return f"[FAIL]    N={N:3d} M={M:4d} {sec:8s} {err}"

    order = np.argsort(np.abs(vals))
    vals, vecs = vals[order], vecs[:, order]
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, vals=vals, vecs=vecs[:, :nkeep], status=status, block_dim=nb,
                 fill_gb=fill_gb)
    os.replace(tmp, out)
    ev = "  ".join(f"{v.real:+.2e}{v.imag:+.1e}j" for v in vals[:nkeep + 1])
    return (f"[{status:7s}] N={N:3d} M={M:4d} {sec:8s} dim={nb:>9d} LU {fill_gb:6.2f} GB "
            f"build {t_build:5.1f}s total {time.time() - t0:7.1f}s | {ev}")


# =============================================================================
# Linear-algebra helpers (all matrices are dense parity blocks, Dh x Dh)
# =============================================================================
def herm(A):
    return 0.5 * (A + A.conj().T)


def antiherm_frac(A):
    return np.linalg.norm(A - A.conj().T) / (2 * np.linalg.norm(A))


def trnorm(A):
    return np.linalg.svd(A, compute_uv=False).sum()


def trnorm_h(A):
    return np.abs(np.linalg.eigvalsh(A)).sum()


def dtrace(A, B):
    """D_t = 1/2 ||A - B||_1 (sum of singular values; valid for non-Hermitian z+-)."""
    return 0.5 * trnorm(A - B)


def herm_gauge(r):
    """Remove the arbitrary eigs phase of an eigenoperator with r^dag ∝ r.
    If r^dag = e^{i phi} r then Tr(r r) = e^{-i phi} ||r||^2 and e^{i phi/2} r is Hermitian.
    Returns (Hermitian r, residual anti-Hermitian fraction, |Tr r^2|/||r||^2)."""
    p = np.sum(r * r.T)                       # Tr(r r)
    q = abs(p) / np.vdot(r, r).real           # 1 iff r is Hermitian up to a phase
    h = r * np.sqrt(np.conj(p) / abs(p))
    return herm(h), antiherm_frac(h), q


def expect_jx(rho, jx, N):
    """Tr(Jx_std rho) for a parity block (photon-major, spin-minor ordering)."""
    d = N + 1
    r4 = rho.reshape(rho.shape[0] // d, d, rho.shape[1] // d, d)
    return float(np.real(np.einsum("ab,nbna->", jx, r4)))


def photon_dist(rho, N):
    return np.real(np.diag(rho)).reshape(-1, N + 1).sum(axis=1)


def split_trace_span(v1, v2):
    """Any basis of the 2D slow space -> (Hermitian trace-1 element, Hermitian traceless
    element).  Robust when the switching rate is below numerical precision and eigs
    returns arbitrary mixtures of the steady state and the switching mode."""
    t = np.array([np.trace(v1), np.trace(v2)])
    ra = np.conj(t[0]) * v1 + np.conj(t[1]) * v2          # maximal-trace direction
    rb = t[1] * v1 - t[0] * v2                            # exactly traceless
    rho = ra / np.trace(ra)
    ah = antiherm_frac(rho)
    r2, ah2, q2 = herm_gauge(rb)
    return herm(rho), r2, max(ah, ah2), q2


def unmix(rho_ss, r2, nvec):
    """Step 1.  Rayleigh-Ritz of N(rho) = {n, rho}/2 in span{rho_ss, r2} with the
    Hilbert-Schmidt metric: solve N v = mu G v (2x2).  For HS-orthogonal vacuum and SR
    components the Ritz vectors are exactly those two states.
    Returns (sr, vac, mu, cond(G)), both states trace-normalised."""
    basis = (rho_ss, r2 / np.linalg.norm(r2))
    G = np.array([[np.real(np.sum(a * b.T)) for b in basis] for a in basis])
    Nn = np.array([[np.real(np.einsum("ij,j,ji->", a, nvec, b)) for b in basis]
                   for a in basis])
    mu, V = sla.eigh(Nn, G)
    vac, sr = [herm(V[0, k] * basis[0] + V[1, k] * basis[1]) for k in range(2)]
    return sr / np.trace(sr).real, vac / np.trace(vac).real, mu, np.linalg.cond(G)


def combine_offdiag(r_i, r_mi, Fl, Fr):
    """Step 3, off-diagonal.  r_{-i} = a(X + iY), r_i = b(X - iY) up to unknown complex
    a, b, with X the coherence of the wanted branch and Y the other branch.  The ratio
    a/b is physical (it decides which branch the sum isolates), so it cannot be fixed
    element-wise.  Filtering with the branch's own diagonal states Fl, Fr kills Y
    (disjoint support), and the least-squares ratio of the filtered blocks gives a/b."""
    T_i, T_mi = Fl @ r_i @ Fr, Fl @ r_mi @ Fr
    c = np.vdot(T_i, T_mi) / np.vdot(T_i, T_i)
    resid = np.linalg.norm(T_mi - c * T_i) / np.linalg.norm(T_mi)
    X = r_mi + c * r_i
    return X / trnorm(X), c, resid


def z_analysis(Be, Bo, Xpm, Xmp):
    """Step 4.  U = diag(U+, U-) from the eigenvectors of the diagonal branch blocks
    (sorted descending).  The column phases of U- are gauge: they are fixed so that
    diag(U+^dag X U-) is real and positive -- without this D_t(z++, z+-) cannot
    converge.  The overall phase of the (-+) block is fixed by Tr z-+ > 0."""
    ze, Ue = np.linalg.eigh(Be)
    zo, Uo = np.linalg.eigh(Bo)
    ze, Ue, zo, Uo = ze[::-1], Ue[:, ::-1], zo[::-1], Uo[:, ::-1]
    d = np.diag(Ue.conj().T @ Xpm @ Uo)
    Uo = Uo * np.exp(-1j * np.angle(d))[None, :]
    z_pp = Ue.conj().T @ Be @ Ue
    z_mm = Uo.conj().T @ Bo @ Uo
    z_pm = Ue.conj().T @ Xpm @ Uo
    z_mp = Uo.conj().T @ Xmp @ Ue
    z_mp = z_mp * np.exp(-1j * np.angle(np.trace(z_mp)))
    dpm = np.diag(z_pm)
    return dict(
        ze=ze, zo=zo, z_pm_diag=dpm.real,
        D_pp_mm=dtrace(z_pp, z_mm), D_pp_pm=dtrace(z_pp, z_pm), D_pp_mp=dtrace(z_pp, z_mp),
        purity=float(np.sum(ze**2) / np.sum(ze)**2),
        offdiag_frac_pm=float(np.sqrt(max(np.linalg.norm(z_pm)**2 - np.sum(np.abs(dpm)**2), 0))
                              / np.linalg.norm(z_pm)),
        min_eig_branch=float(min(ze.min(), zo.min())),
    )


# =============================================================================
# Stage 2: analysis of one system size
# =============================================================================
def load_sector(P, N, M, sec, pos_of, Dh):
    path = raw_path(P, N, M, sec)
    if not os.path.exists(path) and sec in MIRROR:
        blocks, vals, _ = load_sector(P, N, M, MIRROR[sec], pos_of, Dh)
        return [b.conj().T for b in blocks], np.conj(vals), True
    dat = np.load(path)
    D = M * (N + 1)
    idx = sector_index(N, M, sec)
    rows, cols = pos_of[idx % D], pos_of[idx // D]
    blocks = []
    for c in range(dat["vecs"].shape[1]):
        b = np.zeros((Dh, Dh), dtype=complex)
        b[rows, cols] = dat["vecs"][:, c]
        blocks.append(b)
    return blocks, dat["vals"], False


def analyze_N(task):
    N, M, P = task
    t0 = time.time()
    D, Dh = M * (N + 1), M * (N + 1) // 2
    n_of, _ = basis_labels(N, M)
    ev, od = np.flatnonzero(n_of % 2 == 0), np.flatnonzero(n_of % 2 == 1)
    pos_of = np.empty(D, dtype=np.int64)
    pos_of[ev], pos_of[od] = np.arange(Dh), np.arange(Dh)
    nvec = {"E_++_1": n_of[ev].astype(float), "E_--_1": n_of[od].astype(float)}
    jx = np.asarray(qt.jmat(N / 2, "x").full())
    res = dict(N=N, M=M, mirrored=False)

    blk, vals = {}, {}
    for sec in SECTORS:
        blk[sec], vals[sec], mirrored = load_sector(P, N, M, sec, pos_of, Dh)
        if mirrored:
            res["mirrored"] = True
    res["vals"] = vals

    # ---- Step 1: vacuum / SR un-mixing in the trace-carrying blocks ----------
    trs = {}
    for sec in TRACE_SECTORS:
        rho_ss, r2, ah, q2 = split_trace_span(blk[sec][0], blk[sec][1])
        lam = np.abs(vals[sec])
        sep = float(lam[2] / lam[1]) if len(lam) > 2 else np.inf
        do_unmix = (P["unmix"] == "always") or (P["unmix"] == "auto" and sep >= P["sep_min"])
        if do_unmix:
            sr, vac, mu, condG = unmix(rho_ss, r2, nvec[sec])
        else:   # no metastable vacuum: the unique slow state is the SR state itself
            sr, vac, mu, condG = rho_ss, np.zeros_like(rho_ss), np.full(2, np.nan), np.nan
        p_sr = photon_dist(sr, N)
        trs[sec] = dict(
            ss=rho_ss, sr=sr, vac=vac, ritz=mu, condG=condG, unmixed=do_unmix,
            n_sr=float(np.real(np.diag(sr)) @ nvec[sec]),
            P_low_sr=float(p_sr[0]),
            neg_sr=float(0.5 * (trnorm_h(sr) - 1.0)),
            tail_sr=float(p_sr[nvec[sec][::N + 1] >= 0.9 * M].sum()),
            antiherm=ah, herm_quality=q2,
            lam_switch=complex(vals[sec][1]),
            separation=sep,
        )

    # ---- Step 2: gauge-fix the traceless weak-(-1) blocks ---------------------
    # Trace normalisation is impossible here (E_-1 is traceless).  Use trace norm 1,
    # which is exact when the two branches have disjoint support, and fix the sign
    # by the branch label <Jx> > 0 (Jx is odd under Pi4, so it lives in E_-1).
    wm = {}
    for sec in ("E_++_-1", "E_--_-1"):
        h, ah, q = herm_gauge(blk[sec][0])
        h = h / trnorm_h(h)
        s = expect_jx(h, jx, N)
        if s < 0:
            h, s = -h, -s
        lam = np.abs(vals[sec])
        wm[sec] = dict(h=h, jx_over_j=s / (N / 2), antiherm=ah, herm_quality=q,
                       lam=complex(vals[sec][0]),
                       separation=float(lam[1] / lam[0]) if len(lam) > 1 else np.nan)

    # Cross-check of step 1: if the branches have disjoint support,
    # |S_-1| = |e><e|/2 + |e'><e'|/2 must equal the un-mixed SR state.
    jordan = {}
    for sec_m, sec_1 in (("E_++_-1", "E_++_1"), ("E_--_-1", "E_--_1")):
        w, V = np.linalg.eigh(wm[sec_m]["h"])
        jordan[sec_1] = 0.5 * trnorm_h((V * np.abs(w)) @ V.conj().T - trs[sec_1]["sr"])

    # ---- Step 3: compass-qubit branches ---------------------------------------
    if P["sr_source"] == "jordan":
        # Alternative: the E_-1 blocks contain no normal-phase component, so for disjoint
        # branches |S_-1| = (|e><e| + |e'><e'|)/2 is a vacuum-free SR state by construction.
        Se, So = [(V * np.abs(w)) @ V.conj().T
                  for w, V in (np.linalg.eigh(wm[s]["h"]) for s in ("E_++_-1", "E_--_-1"))]
    else:
        Se, So = trs["E_++_1"]["sr"], trs["E_--_1"]["sr"]
    Me, Mo = wm["E_++_-1"]["h"], wm["E_--_-1"]["h"]
    branches = {"R": (Se + Me, So + Mo), "I": (Se - Me, So - Mo)}

    # ---- Steps 3b + 4: coherences, z-matrices, trace distances -----------------
    # Branch I is the exact Pi4 image of branch R (the weak blocks are exact symmetry
    # sectors), so its z-distances coincide with R's; it is kept as a code-path check.
    zres = {}
    for b, (Be, Bo) in branches.items():
        Xpm, c_pm, f_pm = combine_offdiag(blk["E_+-_i"][0], blk["E_+-_-i"][0], Be, Bo)
        Xmp, c_mp, f_mp = combine_offdiag(blk["E_-+_i"][0], blk["E_-+_-i"][0], Bo, Be)
        z = z_analysis(Be, Bo, Xpm, Xmp)
        rho_q = herm(0.5 * np.block([[Be, Xpm], [Xpm.conj().T, Bo]]))
        ov = np.vdot(Xpm.conj().T, Xmp)
        z.update(
            min_eig_qubit=float(np.linalg.eigvalsh(rho_q).min()),
            pair_check=dtrace(Xmp, np.exp(1j * np.angle(ov)) * Xpm.conj().T),
            filter_resid=float(max(f_pm, f_mp)),
        )
        zres[b] = z

    off_lam = {sec: complex(vals[sec][0]) for sec in SECTORS if sec[2] != sec[3]}
    off_sep = {sec: float(abs(vals[sec][1]) / abs(vals[sec][0]))
               for sec in off_lam if len(vals[sec]) > 1}

    res.update(
        trace={s: {k: v for k, v in d.items() if k not in ("ss", "sr", "vac")}
               for s, d in trs.items()},
        weak_minus={s: {k: v for k, v in d.items() if k != "h"} for s, d in wm.items()},
        jordan=jordan, z=zres, off_lam=off_lam, off_sep=off_sep,
        pn={s: dict(n=nvec[s][::N + 1], ss=photon_dist(trs[s]["ss"], N),
                    sr=photon_dist(trs[s]["sr"], N), vac=photon_dist(trs[s]["vac"], N))
            for s in TRACE_SECTORS},
        time=time.time() - t0,
    )
    return res


# =============================================================================
# Stage 3: report
# =============================================================================
def report(results, P):
    rd = run_dir(P)
    os.makedirs(rd, exist_ok=True)
    results = sorted(results, key=lambda r: r["N"])
    Ns = np.array([r["N"] for r in results])

    print("\n" + "=" * 112)
    print("Step 1 diagnostics (even sector E_++_1 | odd sector E_--_1)")
    print(f"{'N':>3} {'M':>4} {'mode':>11} {'<n>_SR':>15} {'N*n_an':>7} {'P_SR(lowest n)':>19} "
          f"{'neg_SR':>17} {'|lam_2|':>19} {'sep':>11} {'tail':>8} {'Jordan':>17} {'<Jx>/j':>7}")
    mode = {True: "unmix", False: "raw"}
    for r in results:
        e, o = r["trace"]["E_++_1"], r["trace"]["E_--_1"]
        print(f"{r['N']:3d} {r['M']:4d} {mode[e['unmixed']]:>5}|{mode[o['unmixed']]:<5} "
              f"{e['n_sr']:7.1f}|{o['n_sr']:7.1f} "
              f"{r['N'] * P['n_int']:7.1f} {e['P_low_sr']:9.1e}|{o['P_low_sr']:9.1e} "
              f"{e['neg_sr']:8.1e}|{o['neg_sr']:8.1e} "
              f"{abs(e['lam_switch']):9.2e}|{abs(o['lam_switch']):9.2e} "
              f"{e['separation']:5.0e}|{o['separation']:5.0e} "
              f"{max(e['tail_sr'], o['tail_sr']):8.1e} "
              f"{r['jordan']['E_++_1']:8.1e}|{r['jordan']['E_--_1']:8.1e} "
              f"{r['weak_minus']['E_++_-1']['jx_over_j']:7.3f}")

    print("\nNoiseless-subsystem test (branch R)")
    print(f"{'N':>3} {'D(z++,z--)':>11} {'D(z++,z+-)':>11} {'D(z++,z-+)':>11} {'offdiag z+-':>12} "
          f"{'purity z':>9} {'min eig qubit':>14} {'max|lam_off|':>13} {'filter':>9} {'(-+) pair':>10}")
    for r in results:
        R = r["z"]["R"]
        lo = max(abs(v) for v in r["off_lam"].values())
        print(f"{r['N']:3d} {R['D_pp_mm']:11.2e} {R['D_pp_pm']:11.2e} {R['D_pp_mp']:11.2e} "
              f"{R['offdiag_frac_pm']:12.2e} {R['purity']:9.4f} {R['min_eig_qubit']:14.2e} "
              f"{lo:13.2e} {R['filter_resid']:9.1e} "
              + (f"{'conj.':>10}" if r["mirrored"] else f"{R['pair_check']:10.1e}"))
    warn = []
    for r in results:
        for sec, t in r["trace"].items():
            if t["n_sr"] < 0.5 * r["N"] * P["n_int"]:
                warn.append(f"N={r['N']} {sec}: <n>_SR={t['n_sr']:.1f} is far below the analytic "
                            "value -- the slow state is probably not the SR state")
        tail = max(t["tail_sr"] for t in r["trace"].values())
        if tail > 1e-6:
            warn.append(f"N={r['N']}: SR weight {tail:.1e} in the top 10% of Fock levels -- "
                        "distances much below this are cutoff-limited; increase M to go lower")
    for w in warn:
        print("  ! " + w)
    print("=" * 112)
    print("mode: 'unmix' = vacuum metastable, Rayleigh-Ritz separation applied; 'raw' = |lam_3|/|lam_2|\n"
          "< SEP_MIN, so the steady state is already the SR state.  P_SR(lowest n), neg_SR -> 0 if the\n"
          "vacuum is removed and the state is positive.  Jordan = D_t(|S_-1|, SR) -> 0 for disjoint\n"
          "branches.  offdiag z+- = Frobenius weight of z+- off its diagonal (tensor-product test).\n"
          "filter = residual of the off-diagonal branch filter.  (-+) pair = D_t between the (-+)\n"
          "coherence and the dagger of the (+-) one (only if SOLVE_MIRROR; 'conj.' = taken as dagger).  Branch I is the\n"
          "exact Pi4 image of R and gives identical distances (saved in summary.npz).")

    # ---- summary.npz ----
    flat = dict(N=Ns, M=np.array([r["M"] for r in results]), lam=P["lam"], n_int=P["n_int"])
    for b in ("R", "I"):
        for key in ("D_pp_mm", "D_pp_pm", "D_pp_mp", "purity", "min_eig_qubit",
                    "offdiag_frac_pm", "pair_check", "filter_resid"):
            flat[f"{key}_{b}"] = np.array([r["z"][b][key] for r in results])
    for sec in TRACE_SECTORS:
        for key in ("n_sr", "P_low_sr", "neg_sr", "tail_sr", "separation"):
            flat[f"{key}_{sec}"] = np.array([r["trace"][sec][key] for r in results])
        flat[f"lam_switch_{sec}"] = np.array([r["trace"][sec]["lam_switch"] for r in results])
    for sec in ("E_++_-1", "E_--_-1"):
        flat[f"lam_{sec}"] = np.array([r["weak_minus"][sec]["lam"] for r in results])
    for sec in results[0]["off_lam"]:
        flat[f"lam_{sec}"] = np.array([r["off_lam"][sec] for r in results])
    for r in results:
        for sec in SECTORS:
            flat[f"N{r['N']}_vals_{sec}"] = r["vals"][sec]
        for b in ("R", "I"):
            flat[f"N{r['N']}_ze_{b}"] = r["z"][b]["ze"]
            flat[f"N{r['N']}_zo_{b}"] = r["z"][b]["zo"]
            flat[f"N{r['N']}_zpm_{b}"] = r["z"][b]["z_pm_diag"]
        for sec in TRACE_SECTORS:
            for key, arr in r["pn"][sec].items():
                flat[f"N{r['N']}_pn_{sec}_{key}"] = arr
    np.savez(os.path.join(rd, "summary.npz"), **flat)

    # ---- P(n) verification plots (step 1) ----
    for r in results:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
        for row, sec in enumerate(TRACE_SECTORS):
            d, tr = r["pn"][sec], r["trace"][sec]
            for col, logy in enumerate((False, True)):
                ax = axes[row, col]
                for key, ls, col, lab in (("ss", "-", "C0", "raw steady state"),
                                          ("vac", "--", "C1", "vacuum component"),
                                          ("sr", "-", "C2", "isolated SR state")):
                    if key == "vac" and not tr["unmixed"]:
                        continue
                    y = np.clip(np.abs(d[key]), 1e-18, None) if logy else d[key]
                    ax.plot(d["n"], y, ls, color=col, marker=".", ms=3, lw=1, label=lab)
                ax.axvline(r["N"] * P["n_int"], color="gray", ls=":", lw=1,
                           label=r"analytic $N\,n_{SR}$")
                if logy:
                    ax.set_yscale("log")
                    ax.set_ylim(1e-16, 1.5)
                ax.set_xlabel("Fock state n")
                ax.set_ylabel("|P(n)|" if logy else "P(n)")
                ax.grid(alpha=0.3)
                ax.set_title(f"{sec} [{'un-mixed' if tr['unmixed'] else 'raw, no metastable vacuum'}]"
                             f": <n>_SR = {tr['n_sr']:.1f}, "
                             f"P_SR(n={int(d['n'][0])}) = {tr['P_low_sr']:.1e}", fontsize=9)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"Vacuum / SR separation, N={r['N']}, M={r['M']}, lambda={P['lam']}")
        fig.tight_layout()
        fig.savefig(os.path.join(rd, f"pn_SR_N{r['N']}_M{r['M']}.png"), dpi=150)
        plt.close(fig)

    # ---- convergence plot (Lieu SM Fig. S2 analogue) ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for key, mk, lab in (("D_pp_mm", "o", r"$D_t(z_{++},z_{--})$"),
                         ("D_pp_pm", "s", r"$D_t(z_{++},z_{+-})$"),
                         ("D_pp_mp", "^", r"$D_t(z_{++},z_{-+})$")):
        ax.semilogy(Ns, [r["z"]["R"][key] for r in results], "-" + mk, label=lab)
    ax.semilogy(Ns, [r["z"]["R"]["offdiag_frac_pm"] for r in results], ":d", color="gray",
                label=r"off-diagonal weight of $z_{+-}$")
    ax.set_xlabel("N")
    ax.set_title("z-matrix trace distances (branch R)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for sec in ("E_++_-1", "E_--_-1"):
        ax.semilogy(Ns, [abs(r["weak_minus"][sec]["lam"]) for r in results], "-o", label=sec)
    for sec in results[0]["off_lam"]:
        ax.semilogy(Ns, [abs(r["off_lam"][sec]) for r in results], "-s", ms=4, label=sec)
    for sec in TRACE_SECTORS:
        ax.semilogy(Ns, [abs(r["trace"][sec]["lam_switch"]) for r in results], ":x",
                    label=f"{sec} (switching)")
    ax.set_xlabel("N")
    ax.set_title(r"$|\Lambda|$ of slowest mode per block")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    ax.plot(Ns, [r["z"]["R"]["purity"] for r in results], "-o", label=r"Tr $z_{++}^2$")
    ax.set_xlabel("N")
    ax.set_title("purity of z (1 = DFS, < 1 = NS)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    for ax in axes:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.suptitle(f"lambda={P['lam']}, kappa2={P['k2']}, wc={P['wc']}, wa={P['wa']}")
    fig.tight_layout()
    fig.savefig(os.path.join(rd, "ns_convergence.png"), dpi=150)
    plt.close(fig)

    # ---- z spectra (Lieu SM Fig. S2(c) analogue) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for r in results:
        z = r["z"]["R"]["ze"][:40]
        axes[0].semilogy(np.arange(len(z)), np.clip(z, 1e-18, None), "-o", ms=3, label=f"N={r['N']}")
    axes[0].set_xlabel("i")
    axes[0].set_title(r"$(z_{++})_{ii}$, branch R")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    r = results[-1]
    for key, mk, lab in (("ze", "o", r"$z_{++}$"), ("zo", "s", r"$z_{--}$"),
                         ("z_pm_diag", "^", r"$|z_{+-}|$ diag")):
        y = np.clip(np.abs(r["z"]["R"][key][:40]), 1e-18, None)
        axes[1].semilogy(np.arange(len(y)), y, "-" + mk, ms=4, mfc="none", label=lab)
    axes[1].set_xlabel("i")
    axes[1].set_title(f"diagonals, N={r['N']}, branch R")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(rd, "z_spectra.png"), dpi=150)
    plt.close(fig)
    print(f"\nPlots and summary.npz written to {rd}")


# =============================================================================
# Main
# =============================================================================
def make_executor(n, fresh_workers=False):
    """ProcessPoolExecutor rather than multiprocessing.Pool: if the OOM killer takes out
    a worker, Pool hangs forever, whereas the executor raises BrokenProcessPool.
    fresh_workers: one task per worker process so factorisation memory is returned
    to the OS (Python >= 3.11; silently ignored on older versions)."""
    if fresh_workers:
        try:
            return ProcessPoolExecutor(max_workers=n, max_tasks_per_child=1)
        except TypeError:
            pass
    return ProcessPoolExecutor(max_workers=n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--N", type=int, nargs="+", default=N_LIST)
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--stage", choices=("all", "solve", "analyze"), default="all")
    ap.add_argument("--outroot", default=OUTROOT)
    ap.add_argument("--unmix", choices=("auto", "always", "never"), default=UNMIX_MODE)
    ap.add_argument("--sr-source", choices=("unmix", "jordan"), default=SR_SOURCE,
                    help="SR state used to build the branches (see SR_SOURCE)")
    args = ap.parse_args()

    P = dict(lam=args.lam, wc=WC, wa=WA, k2=KAPPA2, sigma=SIGMA, tol=TOL,
             maxiter=MAXITER, outroot=args.outroot, unmix=args.unmix, sep_min=SEP_MIN,
             permc=PERMC_SPEC, sr_source=args.sr_source)
    P["n_int"] = n_sr_intensive(P["lam"], P["wc"], P["wa"], P["k2"])
    if not np.isfinite(P["n_int"]):
        raise SystemExit(f"lambda={P['lam']} is below the SR threshold "
                         f"{0.5 * np.sqrt(WC**2 + WA * KAPPA2):.4f}")
    n_workers = args.workers or min(int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count())),
                                    CORE_NR)
    Ns = sorted(set(args.N))
    Ms = {N: choose_M(N, P) for N in Ns}

    print(f"lambda={P['lam']}  kappa2={P['k2']}  wc={P['wc']}  wa={P['wa']}  "
          f"n_SR/N (Eq. 46) = {P['n_int']:.3f}   workers={n_workers}")
    print(f"block structure check: max leak {check_block_structure(P):.1e}")
    for N in Ns:
        D = Ms[N] * (N + 1)
        print(f"  N={N:3d}  M={Ms[N]:4d}  D={D:6d}  D^2={D * D:>11,d}  "
              f"block ~{D * D // 8:>10,d}  N*n_SR={N * P['n_int']:.1f}")

    if args.stage in ("all", "solve"):
        secs = [s for s in SECTORS if SOLVE_MIRROR or s not in MIRROR]
        tasks = [(N, Ms[N], s, P) for N in Ns for s in secs]
        tasks.sort(key=lambda t: -t[1] * (t[0] + 1))       # largest blocks first
        t0 = time.time()
        with make_executor(n_workers, fresh_workers=True) as ex:
            futs = [ex.submit(solve_sector, t) for t in tasks]
            try:
                for f in as_completed(futs):
                    print(f.result(), flush=True)
            except BrokenProcessPool:
                raise SystemExit(
                    "A worker died, most likely killed for running out of memory during the "
                    "LU factorisation.  Finished blocks are checkpointed; rerun with fewer "
                    "--workers or more memory per task.")
        print(f"solve stage: {time.time() - t0:.1f}s")

    if args.stage in ("all", "analyze"):
        ready = []
        for N in Ns:
            needed = [s for s in SECTORS if not (s in MIRROR and not os.path.exists(
                raw_path(P, N, Ms[N], s)))]
            missing = [s for s in needed if not os.path.exists(raw_path(P, N, Ms[N], s))]
            if missing:
                print(f"N={N}: missing blocks {missing}, skipped")
            else:
                ready.append((N, Ms[N], P))
        if not ready:
            raise SystemExit("nothing to analyse")
        t0 = time.time()
        with make_executor(min(n_workers, len(ready))) as ex:
            results = list(ex.map(analyze_N, ready))
        print(f"analysis stage: {time.time() - t0:.1f}s")
        report(results, P)


if __name__ == "__main__":
    main()
