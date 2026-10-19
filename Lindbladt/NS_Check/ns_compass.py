"""
ns_compass.py
=============

Numerical test of whether the symmetry-broken (superradiant, SR) phase of the
dissipative two-photon Dicke model hosts a NOISELESS SUBSYSTEM (NS) in the
sense of Lieu et al., PRL 125, 240405 (2020), Supplemental Material Sec. 2,
and how the two "compass" qubits of the nested Z4^W > Z2^S symmetry relate.

Model (same conventions as sweep.py / thesis Eq. 4.9-4.10):
    H   = wc a^dag a + (wa/2) Jz + (lam/N) Jx (a^dag^2 + a^2),
          Jx, Jz = 2 x standard spin-N/2 operators   ([Jx, Jy] = 2i Jz)
    L2  = sqrt(kappa2/N) a^2                       (two-photon loss only)

Symmetries:  Pi_2 = exp(i pi a^dag a)               -> strong Z2 (photon parity)
             Pi_4 = exp(i pi (a^dag a/2 + Jz_std))   -> weak  Z4
The Liouvillian splits into 8 blocks E_{lr}^{w}:
    l, r  = photon parity of ket (left) / bra (right)          ("+" even, "-" odd)
    w     = weak Z4 label  pi4(ket) * conj(pi4(bra))            (+-1 or +-i)

What the script does, for every system size N in a list:
  1. builds the Liouvillian with QuTiP and cuts it into the 8 symmetry blocks;
  2. in every block computes the few slowest right AND left eigenmodes
     (shift-invert around 0, sparse LU with a nested-dissection ordering);
  3. CUT-FREE NS TEST (primary result, section 4b): builds the operators of
     each compass qubit by projecting qubit-like states onto the slow
     manifold with the left eigenvectors, P(X) = sum_k r_k <l_k, X> (Lieu et
     al.'s quench protocol evaluated exactly).  The vacuum is removed by the
     dynamics itself (slow vacuum<->SR switching mode), not by a Fock cut.
     The Lieu test is then reported (a) for the full metastable qubit state,
     (b) for its dominant auxiliary states, (c) for its committed basin, with
     the incoherent weight and eps = slow rate / next rate.  For (b) the
     coherence block z_eo is also broken down (diagonal mismatch, largest
     off-diagonal element) and D_t(z_ee,z_eo) is divided by the decay rate
     of the logical coherence (a decaying mode cannot be exactly NS-shaped);
  4. OLD METHOD, kept for comparison: removes the vacuum by projecting onto
     n >= n_cut in Fock space and superimposes the weak-Z4 partner blocks
     (+1 with -1, +i with -i) by a support condition; Lieu test on that;
  5. quantifies how the two qubits relate (direct sum vs. coherent, overlap,
     Z4 map, which-qubit classical bit, decoherence of inter-qubit coherence).

Results go into ONE flat folder (OUTDIR): one .npz per N, a text summary and
a handful of figures.  File names carry lam and kappa2, plus wc and wa when
these differ from the defaults (e.g. '..._wa0.250' for the low-leak control
run with --wa 0.25).  Run `python ns_compass.py --help` for options.
The accompanying PDF explains the method in detail.
"""

# ---------------------------------------------------------------------------
# Thread settings: SuperLU and ARPACK are single-threaded; we parallelise over
# symmetry blocks with processes, so BLAS must not spawn extra threads.
# (Same as in sweep.py; must happen before numpy is imported.)
# ---------------------------------------------------------------------------
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import sys
import glob
import json
import time
import argparse
import multiprocessing as mp
import resource
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, eigs, LinearOperator, ArpackNoConvergence
import qutip as qt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ===========================================================================
# PARAMETERS  (defaults; most can be overridden from the command line)
# ===========================================================================
N_LIST = [2, 3, 4, 5, 6, 7, 8]  # number of spins; the "thermodynamic" scan
LAM = 0.6                       # coupling, must lie in the superradiant phase
KAPPA2 = 0.05                   # two-photon loss rate (thesis value)
WC = 1.0                        # cavity frequency
WA = 1.0                        # atomic frequency

# Fock cutoff.  The SR lobe sits at n_MF = N*sqrt(4 lam^2 - wc^2)/kappa2
# (thesis Eq. 4.51 with |<Jx>| -> 1) and is ~1.5x wider than a coherent state.
# M = n_MF + M_SIGMAS*sqrt(n_MF) + M_EXTRA keeps the slow modes' weight at
# the cutoff below ~1e-9 (checked by the script, see 'tail weight').
M_SIGMAS = 10.0
M_EXTRA = 6

N_MODES = 6                     # eigenpairs computed per block
SIGMA = -1e-7 + 1e-9j           # shift for shift-invert (just next to 0)
ARPACK_TOL = 1e-12
MAX_WORKERS = 6                 # parallel blocks (each holds one sparse LU)
MEM_BUDGET_GB = 110             # memory the parallel blocks may use together
COMPUTE_CONJUGATE_BLOCKS = False  # E_-+ blocks are the Hermitian conjugates
                                  # of E_+- blocks; set True to compute them
                                  # anyway as a numerical cross-check.
CUT_SHIFT = 4                   # vacuum cut is varied by +-CUT_SHIFT photons
Q_THRESHOLD = 1e-3              # q-test uses auxiliary states with z_k > this
OUTDIR = "ns_results"

ALL_BLOCKS = ["E_++_1", "E_++_-1", "E_--_1", "E_--_-1",
              "E_+-_i", "E_+-_-i", "E_-+_i", "E_-+_-i"]
CONJUGATE_OF = {"E_-+_-i": "E_+-_i", "E_-+_i": "E_+-_-i"}   # (E_+-^w)^dag


def auto_cutoff(N, lam, kappa2, wc):
    """Fock cutoff M(N) from the mean-field photon number (see PARAMETERS)."""
    if 4 * lam**2 > wc**2:
        n_mf = N * np.sqrt(4 * lam**2 - wc**2) / kappa2
    else:                                   # not in the SR regime: fallback
        n_mf = 4.0 * N
    M = int(np.ceil(n_mf + M_SIGMAS * np.sqrt(n_mf) + M_EXTRA))
    return M + (M % 2)                      # even M: equal even/odd sectors


# ===========================================================================
# 1. MODEL AND SYMMETRY BLOCKS
# ===========================================================================
def build_model(N, M, lam, kappa2, wc, wa):
    """
    Build H, the jump operator and the Liouvillian with QuTiP, exactly as in
    sweep.py, and return everything the analysis needs as scipy matrices.

    Hilbert space ordering is QuTiP's tensor(photon, spin):
        h = n*(N+1) + s ,   spin index s = 0..N  <->  m = N/2 - s.
    Vectorisation (QuTiP, column stacking): rho[row, col] <-> k = row + D*col.
    """
    j = N / 2.0
    a = qt.tensor(qt.destroy(M), qt.qeye(N + 1))
    num = qt.tensor(qt.num(M), qt.qeye(N + 1))
    Jx_std = qt.tensor(qt.qeye(M), qt.jmat(j, "x"))   # standard spin ops
    Jz_std = qt.tensor(qt.qeye(M), qt.jmat(j, "z"))
    Jx_H, Jz_H = 2 * Jx_std, 2 * Jz_std                 # paper convention

    H = wc * num + (wa / 2.0) * Jz_H + (lam / N) * Jx_H * (a.dag()**2 + a**2)
    L2 = np.sqrt(kappa2 / N) * a**2
    L = qt.liouvillian(H, [L2])

    def csr(op):                        # QuTiP object -> scipy CSR matrix
        try:
            return sp.csr_matrix(op.data.as_scipy())      # QuTiP 5
        except AttributeError:
            return sp.csr_matrix(op.data)                 # QuTiP 4

    n_lab = np.rint(np.real(num.diag())).astype(int)      # photon number of h
    m_lab = np.real(Jz_std.diag())                        # spin m of h
    return dict(N=N, M=M, D=M * (N + 1), lam=lam, kappa2=kappa2, wc=wc, wa=wa,
                L=csr(L), H=csr(H), c=csr(L2), a=csr(a), Jx=csr(Jx_std),
                n=n_lab, m=m_lab, s=np.rint(j - m_lab).astype(int))


def symmetry_labels(model):
    """
    Z2 parities and Z4 eigenvalue of every Hilbert-space basis state:
        pi2(h) = (-1)^n ,   pi4(h) = exp(i pi (n/2 + m)).
    (Identical to the diagonals of Pi_2 and Pi_4 built in sweep.py.)
    """
    n, m = model["n"], model["m"]
    return (-1.0) ** n, np.exp(1j * np.pi * (n / 2.0 + m))


def block_indices(model):
    """
    Vectorised indices k = row + D*col belonging to each of the 8 blocks.
    Same labelling as sweep.py:  E_{lr}_{w},  w = pi4(row)*conj(pi4(col)).
    Returns {name: sorted index array}.
    """
    D = model["D"]
    pi2, pi4 = symmetry_labels(model)
    row = np.tile(np.arange(D), D)            # k = row + D*col
    col = np.repeat(np.arange(D), D)
    l, r = pi2[row], pi2[col]
    w = pi4[row] * np.conj(pi4[col])
    out = {}
    for name in ALL_BLOCKS:
        _, par, wl = name.split("_")
        sl = 1 if par[0] == "+" else -1
        sr = 1 if par[1] == "+" else -1
        wv = {"1": 1, "-1": -1, "i": 1j, "-i": -1j}[wl]
        mask = (l == sl) & (r == sr) & np.isclose(w, wv, atol=1e-8)
        out[name] = np.nonzero(mask)[0]
    assert sum(len(v) for v in out.values()) == D * D, "blocks do not cover space"
    return out


def check_block_structure(model, blocks):
    """Sanity check: the Liouvillian has NO matrix elements between blocks."""
    D = model["D"]
    lab = np.empty(D * D, dtype=np.int8)
    for i, name in enumerate(ALL_BLOCKS):
        lab[blocks[name]] = i
    Lc = model["L"].tocoo()
    return int(np.count_nonzero(lab[Lc.row] != lab[Lc.col]))


# ===========================================================================
# 2. SLOW EIGENMODES OF ONE BLOCK (runs in a worker process)
# ===========================================================================
def nested_dissection_order(coords, steps, leaf=64):
    """
    Fill-reducing ordering for the sparse LU.  Every vectorised index of a
    block is a point (n_ket, s_ket, n_bra, s_bra) of a 4D lattice and the
    Liouvillian only couples nearest neighbours (n changes by 2, s by 1).
    Recursive bisection: split along the longest lattice direction at its
    median; order [left half, right half, separating plane].  Separator-last
    orderings keep the LU fill ~10x smaller than SuperLU's default COLAMD for
    this problem (COLAMD runs out of memory already at N=6).
    """
    out = []

    def rec(idx):                       # order idx: [left part, right part, plane]
        if len(idx) <= leaf:
            out.append(idx)
            return
        c = coords[idx]
        extent = (c.max(0) - c.min(0)) / steps
        ax = int(np.argmax(extent))
        if extent[ax] < 1:
            out.append(idx)
            return
        vals = np.unique(c[:, ax])
        med = vals[len(vals) // 2]
        rec(idx[c[:, ax] < med])
        rec(idx[c[:, ax] > med])
        out.append(idx[c[:, ax] == med])

    rec(np.arange(len(coords)))
    return np.concatenate(out)


_SHARED = {}    # filled by the master process before the workers are forked


def slow_modes_of_block(name):
    """
    Worker: the N_MODES eigenvalues of block `name` closest to zero, with
    right eigenvectors (L r = lam r) and left eigenvectors (L^dag l = lam* l),
    via shift-invert Arnoldi.  ONE sparse LU of (L_block - sigma) serves both:
    right modes use lu.solve(x), left modes lu.solve(x, trans='H').
    """
    t0 = time.time()
    Lb, coords = _SHARED["blocks"][name]
    k, sigma, tol = _SHARED["k"], _SHARED["sigma"], _SHARED["tol"]
    n = Lb.shape[0]
    perm = nested_dissection_order(coords, np.array([2, 1, 2, 1]))
    A = (Lb - sigma * sp.identity(n, dtype=complex, format="csr"))[perm][:, perm]
    A = A.tocsc()
    lu = splu(A, permc_spec="NATURAL", diag_pivot_thresh=0.1,
              options=dict(SymmetricMode=True))
    del A
    t_lu = time.time() - t0
    nnz_lu = lu.L.nnz + lu.U.nnz

    def solve(x, trans):               # (L_block - sigma)^{-1} x, or ^{-H} x
        y = np.empty(n, dtype=complex)
        y[perm] = lu.solve(np.ascontiguousarray(x[perm]), trans=trans)
        return y

    def arnoldi(trans):                # ARPACK on the inverse -> slowest modes
        op = LinearOperator((n, n), dtype=complex,
                            matvec=lambda x: solve(x, trans))
        try:
            nu, V = eigs(op, k=k, which="LM", tol=tol, maxiter=20000)
            ok = True
        except ArpackNoConvergence as e:
            nu, V, ok = e.eigenvalues, e.eigenvectors, False
        if trans == "N":
            lam = sigma + 1.0 / nu
        else:                                   # (L - s)^H w = (1/nu) w
            lam = np.conj(np.conj(sigma) + 1.0 / nu)
        order = np.argsort(np.abs(lam))
        return lam[order], V[:, order], ok

    lamR, R, okR = arnoldi("N")
    lamL, W, okL = arnoldi("H")
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print(f"    [{name:8s}] dim {n:>8d}  LU {nnz_lu/1e6:7.1f}M nnz "
          f"({t_lu:6.1f}s)  total {time.time()-t0:6.1f}s  peak mem {peak_gb:5.1f} GB  "
          f"slowest: {lamR[0].real:+.2e}{lamR[0].imag:+.2e}j"
          + ("" if okR and okL else "  ARPACK NOT CONVERGED"), flush=True)
    return dict(name=name, lam=lamR, R=R, lamL=lamL, L=W, converged=okR and okL,
                t_lu=t_lu, t_total=time.time() - t0, nnz_lu=nnz_lu,
                dim=n, nnz=Lb.nnz, peak_gb=peak_gb)


def estimated_block_memory_gb(dim):
    """
    Rough peak memory of one worker (sparse LU + Arnoldi), calibrated on test
    runs with the nested-dissection ordering:  ~4 GB at block dimension 8e4,
    growing like dim^1.65.  Used only to choose how many blocks run at once.
    """
    return 4.0 * (dim / 8.0e4) ** 1.65


def compute_all_blocks(model, blocks, n_workers, mem_budget_gb=None):
    """
    Extract the blocks, then solve them in parallel.  The blocks are put in a
    module-level dict BEFORE the worker processes are forked, so the workers
    read them from shared memory instead of receiving pickled copies.
    """
    D = model["D"]
    names = [b for b in ALL_BLOCKS
             if COMPUTE_CONJUGATE_BLOCKS or b not in CONJUGATE_OF]
    _SHARED.clear()
    _SHARED.update(k=N_MODES, sigma=SIGMA, tol=ARPACK_TOL, blocks={})
    for b in names:
        idx = blocks[b]
        row, col = idx % D, idx // D
        coords = np.stack([model["n"][row], model["s"][row],
                           model["n"][col], model["s"][col]], axis=1)
        _SHARED["blocks"][b] = (model["L"][idx][:, idx].tocsr(), coords)
    model["L"] = None           # full Liouvillian no longer needed: free it

    dim = max(len(blocks[b]) for b in names)
    est = estimated_block_memory_gb(dim)
    nw = min(n_workers, len(names))
    if mem_budget_gb:
        nw = max(1, min(nw, int(mem_budget_gb // est)))
    print(f"    {len(names)} blocks of dim ~{dim}, estimated {est:.1f} GB each "
          f"-> {nw} in parallel", flush=True)
    try:
        with ProcessPoolExecutor(max_workers=nw,
                                 mp_context=mp.get_context("fork")) as ex:
            results = list(ex.map(slow_modes_of_block, names))
    except BrokenProcessPool:
        sys.exit("\nERROR: a worker process died, most likely killed for using too "
                 "much memory.\nRun with fewer parallel blocks (--workers or a smaller "
                 "--mem-budget) or request more memory.")
    _SHARED.clear()
    res = {r["name"]: r for r in results}
    # E_-+ blocks from Hermitian conjugation:  L(X^dag) = L(X)^dag
    for b, src in CONJUGATE_OF.items():
        if b in res:
            continue
        res[b] = dict(name=b, conjugate_of=src, lam=np.conj(res[src]["lam"]),
                      lamL=np.conj(res[src]["lamL"]), dim=res[src]["dim"])
    return res


# ===========================================================================
# 3. SMALL HELPERS FOR THE ANALYSIS
# ===========================================================================
def vec_to_op(idx, v, D):
    """Block eigenvector (values on indices idx) -> D x D operator."""
    full = np.zeros(D * D, dtype=complex)
    full[idx] = v
    return full.reshape(D, D, order="F")        # column stacking


def hermitian_with_phase(X):
    """
    Eigenvectors are defined up to a complex factor.  Operators of the
    self-conjugate blocks (E_++, E_--) are Hermitian up to that factor:
    remove the phase (via the trace, or via Tr(X X) if X is traceless) and
    return the Hermitian part together with the relative anti-Hermitian
    residual (should be ~1e-13).
    """
    tr = np.trace(X)
    if abs(tr) > 1e-6 * np.linalg.norm(X):
        Y = X * np.exp(-1j * np.angle(tr))
    else:
        Y = X * np.exp(-0.5j * np.angle(np.sum(X * X.T)))   # Tr(X X)
    res = np.linalg.norm(Y - Y.conj().T) / np.linalg.norm(Y)
    return 0.5 * (Y + Y.conj().T), res


def trace_distance(A, B):
    """D_t(A,B) = (1/2) Tr|A-B| (also for rectangular/non-Hermitian A-B)."""
    return 0.5 * np.sum(np.linalg.svd(A - B, compute_uv=False))


def eig_desc(X):
    """Eigen-decomposition of a Hermitian matrix, eigenvalues descending."""
    w, V = np.linalg.eigh(X)
    return w[::-1], V[:, ::-1]


# ===========================================================================
# 4. THE ANALYSIS FOR ONE N
# ===========================================================================
def find_superradiant_basin(model, blocks, res):
    """
    The steady state is bistable: superradiant (SR) lobes at n ~ n_MF and a
    normal-phase (vacuum) component at small n.  The two are separated in
    photon number by a macroscopic distance (~N).  We locate
      * the SR peak from the slowest mode of E_++_-1 (the A/B population
        imbalance, which only exists in the SR phase), and
      * the valley of the full steady state P(n) between n=0 and that peak.
    Everything with n < n_cut is the vacuum basin and is projected out.
    """
    D, M, nl = model["D"], model["M"], model["n"]
    out = {}
    # photon distribution of the exact steady state (even + odd sector)
    P = np.zeros(M)
    for b in ("E_++_1", "E_--_1"):
        idx, v = blocks[b], res[b]["R"][:, 0]
        row, col = idx % D, idx // D
        d = row == col
        diag = v[d] * np.exp(-1j * np.angle(np.sum(v[d])))   # remove global phase
        p = np.bincount(nl[row[d]], weights=np.real(diag), minlength=M)
        P += 0.5 * p / p.sum()
    # weight profile of the SR-only mode
    idx, v = blocks["E_++_-1"], res["E_++_-1"]["R"][:, 0]
    prof = np.bincount(nl[idx % D], weights=np.abs(v)**2, minlength=M)
    n_peak = int(np.argmax(prof))
    # combine neighbouring even/odd photon numbers (P alternates by parity)
    m2 = M // 2
    Ppair = P[0:2 * m2:2] + P[1:2 * m2:2]
    ivalley = int(np.argmin(Ppair[: n_peak // 2 + 1]))
    n_cut = 2 * ivalley
    out.update(P_ss=P, profile_SR=prof, n_peak=n_peak, n_cut=n_cut,
               valley_ratio=Ppair[ivalley] / Ppair[min(n_peak // 2, m2 - 1)],
               vacuum_weight=P[:n_cut].sum(),
               P0=P[0])
    # Fock-cutoff check: weight of the slow modes in the top 4 Fock states
    tails = []
    for b in res:
        if "R" not in res[b]:
            continue
        idx, v = blocks[b], res[b]["R"][:, 0]
        w = np.bincount(nl[idx % D], weights=np.abs(v)**2, minlength=M)
        tails.append(w[-4:].sum() / w.max())
    out["tail_weight"] = float(max(tails))
    return out


def select_sr_modes(model, blocks, res, n_cut):
    """
    Pick, in every block, the eigenmode that belongs to the SR steady-state
    manifold.  For each computed mode we evaluate (without forming dense
    matrices)
        frac  = ||P_SR r P_SR|| / ||r||      (fraction living in the SR basin)
        trSR  = |Tr(P_SR r)| / ||r||          (SR population it carries)
    * E_++_1, E_--_1:  the mode with the largest trSR.  This is the exact
      steady state (lam=0) when the SR phase dominates it, otherwise the
      vacuum<->SR switching mode; after the projection both give the SR state.
      Vacuum modes and SR-internal relaxation modes have trSR ~ 0.  A second
      mode that also carries SR population is the vacuum<->SR switching mode.
    * all other blocks: the slowest mode with frac > 0.5.
    'lam_second' is the next-slowest eigenvalue of the block (of any kind);
    it measures how well the selected mode is separated from the rest.
    """
    D, nl = model["D"], model["n"]
    nan = complex(np.nan, np.nan)
    sel = {}
    for b in res:
        if "R" not in res[b]:
            continue
        idx = blocks[b]
        row, col = idx % D, idx // D
        insr = (nl[row] >= n_cut) & (nl[col] >= n_cut)
        diag_sr = insr & (row == col)
        frac, trsr = [], []
        for kk in range(res[b]["R"].shape[1]):
            v = res[b]["R"][:, kk]
            nv = np.linalg.norm(v)
            frac.append(np.linalg.norm(v[insr]) / nv)
            trsr.append(abs(np.sum(v[diag_sr])) / nv)
        frac, trsr = np.array(frac), np.array(trsr)
        lam = res[b]["lam"]
        lam_switch = nan
        if b in ("E_++_1", "E_--_1"):
            k = int(np.argmax(trsr))
            sw = [q for q in range(len(frac)) if q != k and trsr[q] > 0.1 * trsr[k]]
            if sw:
                lam_switch = lam[sw[0]]
        else:
            cand = [q for q in range(len(frac)) if frac[q] > 0.5]
            k = cand[0] if cand else 0
        rest = [q for q in range(len(frac)) if q != k]
        sel[b] = dict(k=k, lam=lam[k], frac=frac, trsr=trsr,
                      lam_second=(lam[rest[0]] if rest else nan),
                      lam_switch=lam_switch)
    # E_-+ blocks: inherit from their Hermitian-conjugate partner
    for b, src in CONJUGATE_OF.items():
        if b not in sel:
            s = sel[src]
            sel[b] = dict(k=s["k"], lam=np.conj(s["lam"]), frac=s["frac"],
                          trsr=s["trsr"], lam_second=np.conj(s["lam_second"]),
                          lam_switch=nan)
    return sel


def sr_operator(model, blocks, res, sel, b, n_cut, left=False):
    """Selected (right or left) eigenmode of block b as an operator, projected
    onto the SR basin n >= n_cut (vacuum basin removed)."""
    D, nl = model["D"], model["n"]
    k = sel[b]["k"]
    if left:   # left eigenvector belonging to the same eigenvalue
        k = int(np.argmin(np.abs(res[b]["lamL"] - res[b]["lam"][k])))
        X = vec_to_op(blocks[b], res[b]["L"][:, k], D)
    else:
        X = vec_to_op(blocks[b], res[b]["R"][:, k], D)
    keep = nl >= n_cut
    return X * keep[:, None] * keep[None, :]


def split_into_qubits(Sd, Td, Cp, Cm, Jx_e, Jx_o):
    """
    Superimpose the weak-Z4 partner blocks to isolate the two compass qubits.

    Expected structure (thesis Sec. 4.5.1, '|q1>,|q3>' = pair A, '|q2>,|q4>'
    = pair B):
        S_mu  (E_mumu^{+1}) = (rho_A^{mumu} + rho_B^{mumu}) / 2
        T_mu  (E_mumu^{-1}) ~  rho_A^{mumu} - rho_B^{mumu}
        C_+   (E_+-^{+i})   ~  rho_A^{+-}  - i rho_B^{+-}
        C_-   (E_+-^{-i})   ~  rho_A^{+-}  + i rho_B^{+-}
    The supports of A and B are read off from the sign of T_mu: its positive
    / negative eigenspaces span qubit A / qubit B (labelled so that A has
    <Jx> > 0, i.e. the spins point along +x; B along -x).  The superposition
    coefficients are then fixed by demanding that the qubit-A operators have
    NO weight on qubit B's support:
        rho_A^{mumu} ~ S_mu + c_mu T_mu ,   c_mu = -Tr(P_B S)/Tr(P_B T)
        rho_A^{+-}   ~ C_+ + r C_- ,         r = -<B_-, B_+>/<B_-, B_->
    (B_+- = P_B C_+- P_B), and analogously for qubit B.
    Inputs Sd, Td: dicts {'e','o'} of Hermitian matrices on the even/odd
    subspace; Cp, Cm: even x odd matrices.
    """
    out = dict(P={}, rho={}, diag={})
    for mu in ("e", "o"):
        S, T = Sd[mu], Td[mu]
        Jx = Jx_e if mu == "e" else Jx_o
        w, V = np.linalg.eigh(T)
        eps = 1e-10 * np.max(np.abs(w))
        Vp, Vm = V[:, w > eps], V[:, w < -eps]
        Pp, Pm = Vp @ Vp.conj().T, Vm @ Vm.conj().T
        if np.real(np.trace(Jx @ Pp @ S @ Pp)) < np.real(np.trace(Jx @ Pm @ S @ Pm)):
            Pp, Pm, T, Vp, Vm = Pm, Pp, -T, Vm, Vp  # A := spins along +x
        out["P"][("A", mu)], out["P"][("B", mu)] = Pp, Pm
        out.setdefault("V", {})
        out["V"][("A", mu)], out["V"][("B", mu)] = Vp, Vm
        cA = -np.real(np.trace(Pm @ S)) / np.real(np.trace(Pm @ T))
        cB = -np.real(np.trace(Pp @ S)) / np.real(np.trace(Pp @ T))
        rA, rB = S + cA * T, S + cB * T
        out["rho"][("A", mu)] = rA / np.real(np.trace(rA))
        out["rho"][("B", mu)] = rB / np.real(np.trace(rB))
        nS = np.linalg.norm(S)
        out["diag"][mu] = dict(
            weight_A=np.real(np.trace(Pp @ S)), weight_B=np.real(np.trace(Pm @ S)),
            # A-B coherence still present in the E^{+1} steady state:
            ab_coherence=np.linalg.norm(Pp @ S @ Pm) / nS,
            # is the E^{-1} mode the population imbalance of the SAME states?
            shape_mismatch=trace_distance(
                Pp @ T @ Pp / np.real(np.trace(Pp @ T)),
                Pp @ S @ Pp / np.real(np.trace(Pp @ S))),
            coef=cA)
    PAe, PAo = out["P"][("A", "e")], out["P"][("A", "o")]
    PBe, PBo = out["P"][("B", "e")], out["P"][("B", "o")]
    for q, (Pe, Po), (Qe, Qo) in (("A", (PAe, PAo), (PBe, PBo)),
                                  ("B", (PBe, PBo), (PAe, PAo))):
        Xp, Xm = Qe @ Cp @ Qo, Qe @ Cm @ Qo         # parts on the OTHER qubit
        r = -np.vdot(Xm, Xp) / np.vdot(Xm, Xm)
        out["rho"][(q, "eo")] = Cp + r * Cm
        out["diag"]["eo_" + q] = dict(
            cancel_residual=np.linalg.norm(Xp + r * Xm) / np.linalg.norm(Xp),
            coef=r)
    # cross terms (coherence between A and B) inside the coherence modes
    out["diag"]["eo_cross"] = max(
        np.linalg.norm(PAe @ C @ PBo) / np.linalg.norm(C) for C in (Cp, Cm))
    return out


def lieu_ns_test(rho_ee, rho_oo, rho_eo, left_eo=None):
    """
    Noiseless-subsystem test of Lieu et al. (SM Sec. 2) for ONE qubit.
    If the steady states of the qubit are  r_{mu nu} = |mu><nu| (x) M  then
        rho_ee = W_e M W_e^dag,  rho_oo = W_o M W_o^dag,  rho_eo = W_e M W_o^dag.
    Diagonalise rho_ee = U_e z_ee U_e^dag and rho_oo = U_o z_oo U_o^dag and
    transform the coherence: z_eo = U_e^dag rho_eo U_o.  The phases of the
    columns of U_o are a gauge freedom; they are fixed so that diag(z_eo) is
    real and positive, and z_eo is normalised to unit trace.
    NS  <=>  z_ee = z_oo = z_eo (diagonal) ;  DFS if additionally pure.
    The conserved-quantity test (Lieu SM Sec. 4): the left eigenoperator of
    the coherence mode, transformed the same way and normalised so that
    Tr(q^dag z_eo) = 1, must be the identity on the code (q -> 1).
    """
    z_ee, U_e = eig_desc(rho_ee)
    z_oo, U_o = eig_desc(rho_oo)
    Z = U_e.conj().T @ rho_eo @ U_o
    phase = np.exp(-1j * np.angle(np.diag(Z)))
    Z = Z * phase[None, :]
    Z = Z / np.trace(Z)
    K = int(min(np.sum(z_ee > 1e-12), np.sum(z_oo > 1e-12)))   # resolved part
    out = dict(
        z_ee=z_ee[:K].real, z_oo=z_oo[:K].real, z_eo_diag=np.diag(Z)[:K],
        Dt_ee_oo=0.5 * np.sum(np.abs(z_ee[:K] - z_oo[:K])),
        Dt_ee_eo=trace_distance(np.diag(z_ee[:K]), Z[:K, :K]),
        offdiag_eo=np.max(np.abs(Z[:K, :K] - np.diag(np.diag(Z[:K, :K])))),
        purity=float(np.sum(z_ee[:K] ** 2)), rank=K,
        U_e_top=U_e[:, 0], U_o_top=U_o[:, 0])
    if left_eo is not None:
        Q = U_e.conj().T @ left_eo @ (U_o * phase[None, :])
        Q = Q / np.conj(np.trace(Q.conj().T @ Z))
        Kq = int(min(np.sum(z_ee > Q_THRESHOLD), 20))
        out.update(q_matrix=Q[:12, :12], q_error=float(
            np.max(np.abs(Q[:Kq, :Kq] - np.eye(Kq)))), q_K=Kq)
    return out


def core_states(split, q, mu):
    """
    Dominant auxiliary states of qubit q (A or B) in parity sector mu (e/o):
    the eigenvectors with weight > Q_THRESHOLD of rho_q restricted to the
    qubit's own support, P_q rho_q P_q.  The restriction makes the states of
    A and B exactly orthogonal.  (rho_q itself can contain a small component
    that A and B share, e.g. low-photon states just above the vacuum cut
    where the lobes of A and B are not yet distinguishable; without the
    restriction that component would appear in both cores and fake A-B
    cross terms.)  Used for the which-qubit and A-B coherence diagnostics.
    """
    P = split["P"][(q, mu)]
    rho = P @ split["rho"][(q, mu)] @ P
    z, U = eig_desc(rho / np.real(np.trace(rho)))
    return U[:, z > Q_THRESHOLD]


def spin_resolved_memory(model, l_even, n_peak):
    """
    WHERE is the which-qubit information lost?  l_even is the conserved
    quantity (left eigenvector) of the E_++^{-1} mode on the even subspace:
    +1 on qubit A, -1 on qubit B, and |l| < 1 on states from which the system
    returns to A or B at random.  We take its diagonal in the basis
    |n> (x) |J_x = m_x>  at the superradiant peak photon number n_peak and
    normalise it to 1 at the fully polarised spin m_x = +N/2.
    Mean field: superradiance needs 2*lam*|<J_x>|/(N/2) > wc, i.e. it
    collapses after  k* = floor((N/2)(1 - wc/(2 lam))) + 1  collective spin
    flips.  If l(m_x) drops to ~0 already for m_x = N/2 - 1, a single spin
    flip erases the which-qubit memory (small-N regime, see the PDF).
    """
    N = model["N"]
    mx, U = np.linalg.eigh(qt.jmat(N / 2.0, "x").full())
    order = np.argsort(-mx)
    mx, U = mx[order], U[:, order]
    Me = l_even.shape[0] // (N + 1)               # number of even photon numbers
    row = min(n_peak // 2, Me - 1)                # even photon number n_peak
    block = l_even.reshape(Me, N + 1, Me, N + 1)[row, :, row, :]
    diag = np.einsum("ji,jk,ki->i", U.conj(), block, U)   # <m_x| l |m_x>
    return mx, np.real(diag / diag[0])


def predicted_spin_flips(N, lam, wc):
    """k* = number of collective spin flips that push |<J_x>|/j below the
    mean-field threshold wc/(2 lam) (where superradiance collapses)."""
    j = N / 2.0
    return int(np.floor(j * (1.0 - wc / (2.0 * lam)) + 1e-9)) + 1


def dephasing_rate(c, u, v):
    """
    Instantaneous decay rate of the coherence X = |u><v| (u, v normalised):
        Gamma = -Re Tr(X^dag L(X)) / Tr(X^dag X)
              = (<c^dag c>_u + <c^dag c>_v)/2 - Re(<c>_u <c>_v^*).
    The Hamiltonian only contributes an imaginary part.  Gamma is small when
    the environment cannot distinguish u from v (same <a^2>), large otherwise.
    """
    cu, cv = c @ u, c @ v
    return float(0.5 * (np.vdot(cu, cu).real + np.vdot(cv, cv).real)
                 - np.real(np.vdot(u, cu) * np.conj(np.vdot(v, cv))))


# ===========================================================================
# 4b. CUT-FREE ISOLATION OF THE QUBITS: PROJECTION ONTO THE SLOW MANIFOLD
# ===========================================================================
# The Fock-space cut n >= n_cut (above) removes the low-photon part of the
# slow modes by hand.  The functions below avoid any cut: they build the
# qubit-A operators from the dynamics itself, with the LEFT eigenvectors.
SWITCH_FRACTION = 0.2   # E_mumu^{+1} modes slower than SWITCH_FRACTION x (next
                        # rate) are kept as vacuum<->SR switching modes
COMMITTOR_MIN = 0.8     # "committed basin" of a qubit: states whose committor
                        # (normalised which-qubit observable) exceeds this


def paired_left(res, b, k):
    """Left eigenvector belonging to right eigenvector k of block b."""
    j = int(np.argmin(np.abs(res[b]["lamL"] - res[b]["lam"][k])))
    return res[b]["L"][:, j]


class SlowProjector:
    """
    Projection onto (part of) the slow manifold with right AND left modes,
        P(X) = sum_k r_k <l_k, X> / <l_k, r_k>,     <A, B> = Tr(A^dag B),
    over the chosen modes sel = {block: [k, ...]}.  P(rho0) is the state that
    rho0 has evolved into once the fast modes have decayed but before the
    chosen slow modes decay: Lieu et al.'s quench protocol, evaluated exactly
    instead of by time evolution.  No cut and no free parameter enter.
    """

    def __init__(self, D, blocks, res, sel):   # store (indices, r_k, l_k/<l_k,r_k>*)
        self.D, self.terms = D, []
        for b, ks in sel.items():
            for k in ks:
                r = res[b]["R"][:, k]
                w = paired_left(res, b, k)
                self.terms.append((blocks[b], r, w / np.conj(np.vdot(w, r))))

    def __call__(self, X):                      # P(X) for a D x D operator X
        x = X.reshape(-1, order="F")
        out = np.zeros(self.D * self.D, dtype=complex)
        for idx, r, w in self.terms:
            out[idx] += r * np.vdot(w, x[idx])
        return out.reshape(self.D, self.D, order="F")


def metastability_ratio(res):
    """
    eps = (slowest rate of the symmetry-breaking blocks) / (next rate in those
    blocks).  Perturbatively the slow modes deviate from exactly steady
    (noiseless-subsystem) form by O(eps), so eps sets the floor of the NS test.
    """
    sb = ["E_++_-1", "E_--_-1", "E_+-_i", "E_+-_-i"]
    slow = max(abs(res[b]["lam"][0]) for b in sb)
    nxt = min(abs(res[b]["lam"][1]) for b in sb)
    return slow / nxt, slow, nxt


def which_qubit_observable(model, blocks, res, b, sub):
    """
    The conserved quantity of the E_mumu^{-1} mode as a Hermitian observable
    l_T on the parity subspace `sub`: P(rho) contains r_T * Tr(l_T rho), with
    r_T Hermitian and oriented so that Tr(Jx r_T) > 0.  l_T ~ +1 on qubit A,
    -1 on qubit B, ~0 on states that end up in A or B at random.
    """
    D = model["D"]
    rH, _ = hermitian_with_phase(vec_to_op(blocks[b], res[b]["R"][:, 0], D))
    if np.real(np.trace(model["Jx"] @ rH)) < 0:
        rH = -rH
    w = vec_to_op(blocks[b], paired_left(res, b, 0), D)
    w = w / np.conj(np.vdot(w.reshape(-1), rH.reshape(-1)))     # <w, r_T> = 1
    lT = w.conj().T[sub]                                        # Tr(lT rho) = <w, rho>
    return 0.5 * (lT + lT.conj().T)


def spectral_qubit_states(model, blocks, res, qubit="A"):
    """
    Cut-free operators of one compass qubit:
      psi_e : extreme metastable state of the qubit in the even sector, i.e. the
              eigenvector of the which-qubit observable l_T with the largest
              (A) or smallest (B) eigenvalue -- the state most certain to stay
              in that qubit;
      psi_o : a|psi_e>/norm, its odd partner with the SAME auxiliary state
              (a is the logical X of a cat qubit);
      rho_mu,nu = P(|psi_mu><psi_nu|) with P the slow-manifold projection over
              the slowest mode of every block, plus slow vacuum<->SR switching
              modes of E_mumu^{+1} where they exist (then the vacuum is removed
              by the dynamics' own definition, "SR-conditioned").
    """
    D, nl = model["D"], model["n"]
    even, odd = (nl % 2 == 0), (nl % 2 == 1)
    ee, oo, eo = np.ix_(even, even), np.ix_(odd, odd), np.ix_(even, odd)
    _, _, nxt = metastability_ratio(res)
    sw = {b: [k for k in range(1, len(res[b]["lam"]))
              if abs(res[b]["lam"][k]) < SWITCH_FRACTION * nxt]
          for b in ("E_++_1", "E_--_1")}
    sel = {"E_++_1": [0] + sw["E_++_1"], "E_--_1": [0] + sw["E_--_1"],
           "E_++_-1": [0], "E_--_-1": [0], "E_+-_i": [0], "E_+-_-i": [0]}
    P = SlowProjector(D, blocks, res, sel)
    lT = which_qubit_observable(model, blocks, res, "E_++_-1", ee)
    w, V = np.linalg.eigh(lT)
    # committed basin of this qubit in both parity sectors: eigenvectors of the
    # which-qubit observable with committor (l_T normalised to +1 on this qubit)
    # above COMMITTOR_MIN -- a dynamical replacement for the photon-number cut
    w_o, V_o = np.linalg.eigh(which_qubit_observable(model, blocks, res, "E_--_-1", oo))
    basin = {}
    for mu, wv, Vv in (("e", w, V), ("o", w_o, V_o)):
        q = wv / (wv.max() if qubit == "A" else wv.min())       # committor, 1 on this qubit
        basin[mu] = Vv[:, q > COMMITTOR_MIN]
    psi_e = np.zeros(D, dtype=complex)
    psi_e[even] = V[:, -1] if qubit == "A" else V[:, 0]
    psi_o = model["a"] @ psi_e
    psi_o /= np.linalg.norm(psi_o)
    rho_ee = P(np.outer(psi_e, psi_e.conj()))[ee]
    rho_oo = P(np.outer(psi_o, psi_o.conj()))[oo]
    rho_eo = P(np.outer(psi_e, psi_o.conj()))[eo]
    neg = min(np.linalg.eigvalsh(0.5 * (r + r.conj().T)).min() / np.real(np.trace(r))
              for r in (rho_ee, rho_oo))
    rho_ee = 0.5 * (rho_ee + rho_ee.conj().T) / np.real(np.trace(rho_ee))
    rho_oo = 0.5 * (rho_oo + rho_oo.conj().T) / np.real(np.trace(rho_oo))
    l_eo = vec_to_op(blocks["E_+-_i"], paired_left(res, "E_+-_i", 0), D)[eo]
    return dict(rho_ee=rho_ee, rho_oo=rho_oo, rho_eo=rho_eo, l_eo=l_eo, basin=basin,
                negativity=float(neg), lT_extreme=float(w[-1] if qubit == "A" else w[0]),
                switch={b: [abs(res[b]["lam"][k]) for k in ks] for b, ks in sw.items()})


def committed_basin_ns_test(st):
    """
    NS test of the cut-free qubit states restricted to the qubit's committed
    basin (committor > COMMITTOR_MIN in both parity sectors), i.e. with the
    uncommitted spin-flip excursion states removed -- by the dynamics' own
    criterion, not by photon number.  Returns distances and the weight kept.
    """
    Pe = st["basin"]["e"] @ st["basin"]["e"].conj().T
    Po = st["basin"]["o"] @ st["basin"]["o"].conj().T
    re, ro, reo = Pe @ st["rho_ee"] @ Pe, Po @ st["rho_oo"] @ Po, Pe @ st["rho_eo"] @ Po
    kept = float(np.real(np.trace(re)))
    r = lieu_ns_test(re / np.real(np.trace(re)), ro / np.real(np.trace(ro)), reo)
    return dict(Dt_ee_oo=float(r["Dt_ee_oo"]), Dt_ee_eo=float(r["Dt_ee_eo"]), kept=kept)


def aux_resolved_ns_test(st, n_even):
    """
    Lieu NS test of the cut-free qubit states, resolved by auxiliary state.
    Besides the usual numbers (full test) it reports
      dom : the test restricted to the dominant auxiliary states z_k > Q_THRESHOLD
            (renormalised) -- the noiseless-subsystem part of the qubit;
      w_inc : weight of auxiliary states that carry less than half of the
            logical coherence an NS requires (Re z_eo,kk / z_k < 1/2) -- the
            incoherent admixture (low-photon / spin-flip excursion states);
      per aux state: weight z_k, mean photon number n_k, coherence ratio.
    """
    full = lieu_ns_test(st["rho_ee"], st["rho_oo"], st["rho_eo"], left_eo=st["l_eo"])
    ze, Ue = eig_desc(st["rho_ee"])
    zo, Uo = eig_desc(st["rho_oo"])
    Z = Ue.conj().T @ st["rho_eo"] @ Uo
    Z = Z * np.exp(-1j * np.angle(np.diag(Z)))[None, :]
    Z = Z / np.trace(Z)
    K = int(min(np.sum(ze > 1e-12), np.sum(zo > 1e-12), 400))
    coh = np.real(np.diag(Z)[:K]) / ze[:K]
    n_k = np.array([np.sum(n_even * np.abs(Ue[:, k]) ** 2) for k in range(K)])
    Kd = max(1, int(np.sum(ze > Q_THRESHOLD)))
    a_, b_ = ze[:Kd] / ze[:Kd].sum(), zo[:Kd] / zo[:Kd].sum()
    Zd = Z[:Kd, :Kd] / np.trace(Z[:Kd, :Kd])
    return dict(full=full, dom_K=Kd, dom_Dt_ee_oo=float(0.5 * np.abs(a_ - b_).sum()),
                dom_Dt_ee_eo=float(trace_distance(np.diag(a_), Zd)),
                # the coherence block itself, on the dominant states: compared with
                # the odd populations, and split into its diagonal mismatch and its
                # largest off-diagonal element (both must vanish for an NS)
                dom_Dt_oo_eo=float(trace_distance(np.diag(b_), Zd)),
                dom_eo_diag=float(0.5 * np.sum(np.abs(a_ - np.real(np.diag(Zd))))),
                dom_eo_offdiag=float(np.max(np.abs(Zd - np.diag(np.diag(Zd))))) if Kd > 1 else 0.0,
                w_inc=float(ze[:K][coh < 0.5].sum()),
                aux_z=ze[:K].real, aux_n=n_k, aux_coh=coh)


def cut_free_analysis(model, blocks, res):
    """Cut-free NS test for qubits A and B plus the Z4 map between them."""
    eps, slow, nxt = metastability_ratio(res)
    even = model["n"] % 2 == 0
    out = dict(eps=eps, slow=slow, next=nxt)
    states = {}
    for q in ("A", "B"):
        states[q] = spectral_qubit_states(model, blocks, res, q)
        out[q] = aux_resolved_ns_test(states[q], model["n"][even])
        out[q].update(negativity=states[q]["negativity"], lT_extreme=states[q]["lT_extreme"])
        out[q].update({f"basin_{k}": v for k, v in committed_basin_ns_test(states[q]).items()})
    out["switch"] = states["A"]["switch"]
    _, pi4 = symmetry_labels(model)
    p = pi4[even]
    out["Z4map"] = float(trace_distance(p[:, None] * states["A"]["rho_ee"] * np.conj(p)[None, :],
                                        states["B"]["rho_ee"]))
    Jx = model["Jx"].toarray()[np.ix_(even, even)]
    out["Jx_A"] = float(np.real(np.trace(Jx @ states["A"]["rho_ee"])))
    return out


def analyse(model, blocks, res, n_cut=None, verbose=True):
    """
    Steps 3-6 for one N, given the slow modes `res` of all blocks:
    vacuum cut -> mode selection -> qubit separation -> Lieu NS test for
    qubits A and B -> relation diagnostics.  n_cut overrides the automatic
    vacuum cut (used for the sensitivity scan).  Returns a dict of results.
    On the main call (n_cut=None) the cut-free NS test (section 4b) is run
    as well; it is the primary result, the cut-based test a comparison.
    """
    D, nl = model["D"], model["n"]
    even, odd = (nl % 2 == 0), (nl % 2 == 1)
    basin = find_superradiant_basin(model, blocks, res)
    main_call = n_cut is None
    if n_cut is None:
        n_cut = basin["n_cut"]
    sel = select_sr_modes(model, blocks, res, n_cut)

    # --- SR-projected slowest modes, restricted to their parity subspaces
    ee, oo, eo = np.ix_(even, even), np.ix_(odd, odd), np.ix_(even, odd)
    herm_res = {}
    Sd, Td = {}, {}
    for mu, bp, bm, sub in (("e", "E_++_1", "E_++_-1", ee),
                            ("o", "E_--_1", "E_--_-1", oo)):
        S, herm_res[bp] = hermitian_with_phase(
            sr_operator(model, blocks, res, sel, bp, n_cut))
        T, herm_res[bm] = hermitian_with_phase(
            sr_operator(model, blocks, res, sel, bm, n_cut))
        Sd[mu] = S[sub] / np.real(np.trace(S))
        Td[mu] = T[sub]
    Cp = sr_operator(model, blocks, res, sel, "E_+-_i", n_cut)[eo]
    Cm = sr_operator(model, blocks, res, sel, "E_+-_-i", n_cut)[eo]
    Lp = sr_operator(model, blocks, res, sel, "E_+-_i", n_cut, left=True)[eo]
    Lm = sr_operator(model, blocks, res, sel, "E_+-_-i", n_cut, left=True)[eo]

    Jx = model["Jx"].toarray()
    split = split_into_qubits(Sd, Td, Cp, Cm, Jx[ee], Jx[oo])

    # --- Lieu NS test for both qubits.
    # 'main' : coherence = superposition of the two E_+- blocks (see
    #          split_into_qubits); conserved quantity from the E_+-^{+i} block.
    # 'check': coherence and conserved quantity taken from the E_+-^{-i} block
    #          alone (projected on the qubit's support) -> independent estimate.
    ns = {}
    for q in ("A", "B"):
        Pe, Po = split["P"][(q, "e")], split["P"][(q, "o")]
        ns[(q, "main")] = lieu_ns_test(split["rho"][(q, "e")], split["rho"][(q, "o")],
                                       split["rho"][(q, "eo")], left_eo=Pe @ Lp @ Po)
        ns[(q, "check")] = lieu_ns_test(split["rho"][(q, "e")], split["rho"][(q, "o")],
                                        Pe @ Cm @ Po, left_eo=Pe @ Lm @ Po)

    # --- how do the two qubits relate?
    a2 = (model["a"] @ model["a"]).toarray()
    rel = {}
    for mu, sub in (("e", ee), ("o", oo)):
        for q in ("A", "B"):
            rho = split["rho"][(q, mu)]
            rel[f"Jx_{q}_{mu}"] = float(np.real(np.trace(Jx[sub] @ rho)))
            rel[f"a2_{q}_{mu}"] = complex(np.trace(a2[sub] @ rho))
    # Z4 map: Pi_4 rho_A Pi_4^dag should equal rho_B (diagonal unitary)
    _, pi4 = symmetry_labels(model)
    for mu, msk in (("e", even), ("o", odd)):
        p = pi4[msk]
        mapped = p[:, None] * split["rho"][("A", mu)] * np.conj(p)[None, :]
        rel[f"Z4map_{mu}"] = trace_distance(mapped, split["rho"][("B", mu)])
    # conserved quantity of the E^{-1} mode = which-qubit observable (+1 on A,
    # -1 on B): its left eigenvector in the dominant states of A and of B
    for mu, bm, sub in (("e", "E_++_-1", ee), ("o", "E_--_-1", oo)):
        lT = sr_operator(model, blocks, res, sel, bm, n_cut, left=True)[sub]
        UA, UB = core_states(split, "A", mu), core_states(split, "B", mu)
        K = int(min(UA.shape[1], UB.shape[1], 20))
        qA = UA[:, :K].conj().T @ lT @ UA[:, :K]
        qB = UB[:, :K].conj().T @ lT @ UB[:, :K]
        s = np.mean(np.diag(qA))                     # normalise: +1 on A
        rel[f"which_qubit_err_{mu}"] = float(max(
            np.max(np.abs(qA / s - np.eye(K))), np.max(np.abs(qB / s + np.eye(K)))))
        if mu == "e":
            rel["spin_mx"], rel["spin_memory"] = spin_resolved_memory(
                model, lT, basin["n_peak"])
    # Does ANY computed slow mode carry coherence BETWEEN the qubits?  A single
    # 4-level noiseless subsystem would have such modes (|e_A><e_B| lives in
    # E_++^{+-1}, |e_A><o_B| in E_+-^{+-i}); the direct sum of two qubits has none.
    # We look at every computed mode inside the SR "core" (the auxiliary states
    # of the two qubits with weight > 1e-3, see core_states) and ask whether
    # it is mostly A-B coherence there.
    keep = nl >= n_cut
    core = {(q, mu): core_states(split, q, mu) for q in ("A", "B") for mu in ("e", "o")}
    ab_rates = []
    for b, (l, r) in (("E_++_1", ("e", "e")), ("E_++_-1", ("e", "e")),
                      ("E_--_1", ("o", "o")), ("E_--_-1", ("o", "o")),
                      ("E_+-_i", ("e", "o")), ("E_+-_-i", ("e", "o"))):
        sub = np.ix_(even if l == "e" else odd, even if r == "e" else odd)
        for kk in range(res[b]["R"].shape[1]):
            X = vec_to_op(blocks[b], res[b]["R"][:, kk], D)
            X = (X * keep[:, None] * keep[None, :])[sub]
            blk = {(p, q): core[(p, l)].conj().T @ X @ core[(q, r)]
                   for p in ("A", "B") for q in ("A", "B")}
            inside = np.sqrt(sum(np.linalg.norm(v)**2 for v in blk.values()))
            cross = np.sqrt(np.linalg.norm(blk[("A", "B")])**2
                            + np.linalg.norm(blk[("B", "A")])**2)
            if inside > 0.5 * np.linalg.norm(X) and cross > 0.5 * inside:
                ab_rates.append(abs(res[b]["lam"][kk]))
    lam_max = min(max(abs(res[b]["lam"])) for b in res if "R" in res[b])
    rel["ab_mode_rate"] = min(ab_rates) if ab_rates else np.nan
    rel["ab_mode_bound"] = lam_max          # if none found: rate > this bound

    # instantaneous decay of populations / intra- / inter-qubit coherences,
    # evaluated in the dominant auxiliary state of each qubit
    c = model["c"]

    def embed(vec, msk):               # parity-subspace vector -> full space
        u = np.zeros(D, dtype=complex)
        u[msk] = vec
        return u
    eA = embed(ns[("A", "main")]["U_e_top"], even)
    oA = embed(ns[("A", "main")]["U_o_top"], odd)
    eB = embed(ns[("B", "main")]["U_e_top"], even)
    oB = embed(ns[("B", "main")]["U_o_top"], odd)
    rel["Gamma_pop"] = dephasing_rate(c, eA, eA)        # |e_A><e_A|
    rel["Gamma_intra"] = dephasing_rate(c, eA, oA)      # |e_A><o_A|
    rel["Gamma_inter_ee"] = dephasing_rate(c, eA, eB)   # |e_A><e_B|
    rel["Gamma_inter_eo"] = dephasing_rate(c, eA, oB)   # |e_A><o_B|
    rel["overlap_AB_e"] = float(abs(np.vdot(eA, eB)))

    out = dict(N=model["N"], M=model["M"], D=D, lam=model["lam"],
               kappa2=model["kappa2"], wc=model["wc"], wa=model["wa"],
               n_cut_used=n_cut, sel=sel, basin=basin, herm_res=herm_res,
               split_diag=split["diag"], ns=ns, rel=rel)
    if main_call:                          # primary, cut-free NS test (section 4b)
        out["cutfree"] = cut_free_analysis(model, blocks, res)
    if verbose:
        report(out, res)
    return out


# ===========================================================================
# 5. REPORTING, SAVING, PLOTTING
# ===========================================================================
def report(out, res):
    """Human-readable summary of one N (also written to the summary file)."""
    b, sel, rel, sd = out["basin"], out["sel"], out["rel"], out["split_diag"]
    L = []
    freq = "".join(f"  {k}={out[k]}" for k, dflt in (("wc", WC), ("wa", WA)) if out[k] != dflt)
    L.append(f"\n=== N={out['N']}  M={out['M']}  lam={out['lam']}  "
             f"kappa2={out['kappa2']}{freq}  (Hilbert dim {out['D']}) ===")
    L.append(f"Fock cutoff check: slow-mode weight in top 4 Fock states = "
             f"{b['tail_weight']:.1e}" + ("   <-- INCREASE M" if b["tail_weight"] > 1e-7 else ""))
    L.append(f"Vacuum isolation: SR peak n={b['n_peak']}, cut n_cut={out['n_cut_used']}, "
             f"P(valley)/P(peak)={b['valley_ratio']:.1e}, steady-state weight "
             f"below cut={b['vacuum_weight']:.2e}, P(0)={b['P0']:.1e}")
    L.append("Selected superradiant mode per block (eigenvalue | |2nd-slowest eigenvalue| of the block):")
    for blk in ALL_BLOCKS:
        s = sel[blk]
        extra = (f"   vacuum<->SR switching mode {s['lam_switch'].real:+.2e}"
                 if np.isfinite(s["lam_switch"]) else "")
        L.append(f"   {blk:8s} mode {s['k']}: {s['lam'].real:+.3e}{s['lam'].imag:+.2e}j"
                 f"   SR frac {s['frac'][s['k']]:.4f}   | 2nd {abs(s['lam_second']):.2e}"
                 + extra)
    for mu in ("e", "o"):
        d = sd[mu]
        L.append(f"[{mu}] weights A/B = {d['weight_A']:.4f}/{d['weight_B']:.4f}, "
                 f"A-B coherence in steady state = {d['ab_coherence']:.1e}, "
                 f"E^-1 vs E^+1 shape mismatch = {d['shape_mismatch']:.1e}")
    L.append(f"coherence modes: cancellation residual A {sd['eo_A']['cancel_residual']:.1e}, "
             f"B {sd['eo_B']['cancel_residual']:.1e}; A-B cross terms {sd['eo_cross']:.1e}")
    if "cutfree" in out:
        cf = out["cutfree"]
        # decay rate of the logical coherence (mean of the two coherence modes), in
        # units of omega_c: the coherence block of a mode that decays at this rate
        # cannot be exactly NS-shaped, so D_t(z_ee,z_eo) is compared with it
        lam_coh = 0.5 * sum(abs(sel[bb]["lam"].real) for bb in ("E_+-_i", "E_+-_-i")) / out["wc"]
        L.append(f"CUT-FREE NS test (projection onto the slow manifold, no Fock cut):  "
                 f"eps = slow rate / next rate = {cf['slow']:.2e}/{cf['next']:.2e} = {cf['eps']:.2e}   "
                 f"switching modes kept: e {np.round(cf['switch']['E_++_1'], 5)}, "
                 f"o {np.round(cf['switch']['E_--_1'], 5)}")
        for q in ("A", "B"):
            c, f = cf[q], cf[q]["full"]
            L.append(f"   qubit {q}, full metastable state: D_t(z_ee,z_oo)={f['Dt_ee_oo']:.2e}  "
                     f"D_t(z_ee,z_eo)={f['Dt_ee_eo']:.2e}  q-test err={f['q_error']:.1e} (top {f['q_K']})  "
                     f"purity={f['purity']:.4f}  most negative eigenvalue={c['negativity']:.1e}")
            L.append(f"   qubit {q}, dominant aux states (z>{Q_THRESHOLD:g}, {c['dom_K']} states): "
                     f"D_t(z_ee,z_oo)={c['dom_Dt_ee_oo']:.2e}  D_t(z_ee,z_eo)={c['dom_Dt_ee_eo']:.2e}   "
                     f"incoherent weight={c['w_inc']:.2e}")
            L.append(f"      coherence block on these states: D_t(z_oo,z_eo)={c['dom_Dt_oo_eo']:.2e}  "
                     f"diagonal mismatch={c['dom_eo_diag']:.2e}  largest off-diagonal={c['dom_eo_offdiag']:.2e}   "
                     f"D_t(z_ee,z_eo)/(|Re lam_coh|/wc) = {c['dom_Dt_ee_eo'] / lam_coh:.2f}")
            L.append(f"   qubit {q}, committed basin (committor>{COMMITTOR_MIN:g}, weight kept "
                     f"{c['basin_kept']:.5f}): D_t(z_ee,z_oo)={c['basin_Dt_ee_oo']:.2e}  "
                     f"D_t(z_ee,z_eo)={c['basin_Dt_ee_eo']:.2e}")
        L.append(f"   Z4 map Pi4 rho_A Pi4^dag vs rho_B: {cf['Z4map']:.1e}   <Jx>_A = {cf['Jx_A']:+.3f}")
    L.append(f"NS test WITH FOCK CUT n >= {out['n_cut_used']} (old method; lobe only):")
    for q in ("A", "B"):
        n1, n2 = out["ns"][(q, "main")], out["ns"][(q, "check")]
        L.append(f"   qubit {q}: D_t(z_ee,z_oo)={n1['Dt_ee_oo']:.2e}  "
                 f"D_t(z_ee,z_eo)={n1['Dt_ee_eo']:.2e} [{n2['Dt_ee_eo']:.2e}]  "
                 f"purity(z)={n1['purity']:.4f}  q-test err={n1['q_error']:.1e} "
                 f"(top {n1['q_K']})  z[:4]={np.round(n1['z_ee'][:4], 4)}")
    L.append(f"Relation: <Jx>_A={rel['Jx_A_e']:+.3f} <Jx>_B={rel['Jx_B_e']:+.3f}  "
             f"<a^2>_A={rel['a2_A_e']:.2f} <a^2>_B={rel['a2_B_e']:.2f}  "
             f"Z4 map residual={max(rel['Z4map_e'], rel['Z4map_o']):.1e}  "
             f"which-qubit err={max(rel['which_qubit_err_e'], rel['which_qubit_err_o']):.1e}")
    if np.isfinite(rel["ab_mode_rate"]):
        L.append(f"          slowest mode carrying A-B coherence: |lam| = {rel['ab_mode_rate']:.2e}"
                 f"  ({rel['ab_mode_rate'] / abs(sel['E_++_-1']['lam']):.0f} x the E_++_-1 rate)")
    else:
        L.append(f"          no computed slow mode carries A-B coherence "
                 f"(such a mode would have |lam| > {rel['ab_mode_bound']:.2e})")
    L.append(f"          which-qubit memory l(m_x) at the SR peak, m_x = N/2, N/2-1, ...: "
             + " ".join(f"{v:+.3f}" for v in rel["spin_memory"][: min(6, len(rel['spin_memory']))])
             + f"   (mean-field: SR collapses after k* = "
             f"{predicted_spin_flips(out['N'], out['lam'], out['wc'])} spin flips)")
    L.append(f"          decay rates: pop {rel['Gamma_pop']:.3e}  intra-qubit coh "
             f"{rel['Gamma_intra']:.3e}  inter-qubit coh {rel['Gamma_inter_ee']:.3e} / "
             f"{rel['Gamma_inter_eo']:.3e}")
    txt = "\n".join(L)
    print(txt, flush=True)
    out["report"] = txt


def run_tag(lam, kappa2, wc=WC, wa=WA):
    """Parameter part of every output file name, e.g. 'lam0.600_k20.050'.
    omega_c and omega_a are appended only when they differ from the defaults
    (e.g. '..._wa0.250' for the low-leak control run), so such runs do not
    overwrite the default results and old result files keep their names."""
    tag = f"lam{lam:.3f}_k2{kappa2:.3f}"
    if wc != WC:
        tag += f"_wc{wc:.3f}"
    if wa != WA:
        tag += f"_wa{wa:.3f}"
    return tag


def result_path(N, lam, kappa2, wc=WC, wa=WA):
    """File name of the saved results for one N (all in the flat OUTDIR)."""
    return os.path.join(OUTDIR, f"ns_N{N}_{run_tag(lam, kappa2, wc, wa)}.npz")


def save(out, res, timing):
    """Flatten the nested result dict into one .npz file per N."""
    f = {}
    for k in ("N", "M", "D", "lam", "kappa2", "wc", "wa", "n_cut_used", "report"):
        f[k] = out[k]
    for k, v in out["basin"].items():
        f["basin_" + k] = v
    for blk in ALL_BLOCKS:
        f[f"evals_{blk}"] = res[blk]["lam"]
        s = out["sel"][blk]
        f[f"sel_{blk}"] = s["lam"]
        f[f"second_{blk}"] = s["lam_second"]
        f[f"switch_{blk}"] = s["lam_switch"]
        f[f"srfrac_{blk}"] = s["frac"]
    for mu, d in out["split_diag"].items():
        if isinstance(d, dict):
            for k, v in d.items():
                f[f"split_{mu}_{k}"] = v
        else:
            f[f"split_{mu}"] = d
    for (q, tag), d in out["ns"].items():
        for k, v in d.items():
            if k.startswith("U_"):
                continue
            f[f"ns_{q}_{tag}_{k}"] = v
    for k, v in out["rel"].items():
        f["rel_" + k] = v
    for k, v in out.get("cut_scan", {}).items():
        f["cutscan_" + k] = v
    cf = out.get("cutfree")
    if cf is not None:                     # cut-free NS test -> keys cf_*
        for k in ("eps", "slow", "next", "Z4map", "Jx_A"):
            f[f"cf_{k}"] = cf[k]
        for b, v in cf["switch"].items():
            f[f"cf_switch_{b}"] = np.array(v)
        for q in ("A", "B"):
            for k, v in cf[q].items():
                if k == "full":
                    for kk, vv in v.items():
                        if not kk.startswith("U_"):
                            f[f"cf_{q}_full_{kk}"] = vv
                else:
                    f[f"cf_{q}_{k}"] = v
    f["timing"] = json.dumps(timing)
    np.savez(result_path(out["N"], out["lam"], out["kappa2"], out["wc"], out["wa"]), **f)


BLOCK_LABEL = {b: (r"$E_{%s}^{%s}$" % (b.split("_")[1], b.split("_")[2]))
               for b in ALL_BLOCKS}
SB_BLOCKS = ["E_++_-1", "E_--_-1", "E_+-_i", "E_+-_-i"]   # symmetry-breaking modes


def make_plots(lam, kappa2, wc=WC, wa=WA):
    """Summary figures + table from all saved .npz files with these parameters."""
    tag = run_tag(lam, kappa2, wc, wa)
    files = sorted(glob.glob(os.path.join(OUTDIR, f"ns_N*_{tag}.npz")),
                   key=lambda f: int(os.path.basename(f).split("_")[1][1:]))
    if not files:
        print("no result files found for plotting")
        return
    data = [dict(np.load(f, allow_pickle=True)) for f in files]
    Ns = np.array([int(d["N"]) for d in data])
    title = (rf"$\lambda={lam}$, $\kappa_2={kappa2}$"
             + (rf", $\omega_c={wc}$" if wc != WC else "")
             + (rf", $\omega_a={wa}$" if wa != WA else ""))

    # --- Fig 1: spectrum.  Left: the slowest SR mode of the four blocks that
    # carry symmetry breaking (E_-+ are their conjugates; E^{+1} hold the
    # exact steady states).  Right: gap to the next SR mode in the same block.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for b, mk in zip(SB_BLOCKS, "os^v"):
        ax[0].semilogy(Ns, [abs(d[f"sel_{b}"].real) for d in data], mk + "-",
                       label=BLOCK_LABEL[b])
        ax[1].semilogy(Ns, [abs(d[f"second_{b}"]) for d in data], mk + "-",
                       label=BLOCK_LABEL[b])
    ax[0].set_xlabel("N"); ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set_ylabel(r"$|\mathrm{Re}\,\lambda|$")
    ax[0].set_title("slowest superradiant mode per block  (" + title + ")")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("N"); ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].set_ylabel(r"$|\lambda_2|$, second-slowest mode of the block")
    ax[1].set_title("only ONE slow mode per block")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"fig1_spectrum_{tag}.png"), dpi=150)
    plt.close(fig)

    # --- Fig 2: NS test.  (a) cut-free test vs N (full metastable qubit state,
    # its dominant auxiliary states, incoherent weight, eps), old Fock-cut test
    # for comparison; (b) which auxiliary states carry the logical coherence;
    # (c) conserved-quantity test.  Files from runs without the cut-free test
    # (older versions of this script) show only the old curve.
    def g(d, k):                       # value from a result file, NaN if absent
        return float(d[k]) if k in d else np.nan
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.4))
    series = [("cf_A_full_Dt_ee_eo", r"cut-free, full: $D_t(z_{ee},z_{eo})$", "s-"),
              ("cf_A_full_Dt_ee_oo", r"cut-free, full: $D_t(z_{ee},z_{oo})$", "o-"),
              ("cf_A_dom_Dt_ee_eo", r"cut-free, dominant aux states: $D_t(z_{ee},z_{eo})$", "^-"),
              ("cf_A_dom_Dt_ee_oo", r"cut-free, dominant aux states: $D_t(z_{ee},z_{oo})$", "D-"),
              ("cf_A_basin_Dt_ee_eo", r"cut-free, committed basin: $D_t(z_{ee},z_{eo})$", "*-"),
              ("cf_A_w_inc", "incoherent weight", "v--"),
              ("cf_eps", r"$\varepsilon$ = slow rate / next rate", "P--")]
    for key, lab, mk in series:
        ax[0].semilogy(Ns, [abs(g(d, key)) for d in data], mk, ms=5, label=lab)
    ax[0].semilogy(Ns, [g(d, "ns_A_main_Dt_ee_eo") for d in data], "x:", color="0.45",
                   label=r"old method (Fock cut): $D_t(z_{ee},z_{eo})$")
    ax[0].set_xlabel("N"); ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set_title("noiseless-subsystem test, qubit A  (" + title + ")")
    ax[0].legend(fontsize=7)
    dl = data[-1]
    if "cf_A_aux_z" in dl:
        z, n, c = dl["cf_A_aux_z"], dl["cf_A_aux_n"], np.clip(dl["cf_A_aux_coh"], 0, 1)
        sc = ax[1].scatter(n, z, c=c, cmap="Blues", vmin=-0.15, vmax=1, s=26,
                           edgecolors="0.3", linewidths=0.3)
        ax[1].set_yscale("log")
        ax[1].set_ylim(max(1e-12, z[z > 0].min() / 2), 2)
        cb = plt.colorbar(sc, ax=ax[1])
        cb.set_label(r"coherence carried, Re$(z_{eo})_{kk}/z_k$", fontsize=8)
        ax[1].set_xlabel(r"mean photon number of auxiliary state $k$")
        ax[1].set_ylabel(r"weight $z_k$")
        ax[1].set_title(f"auxiliary states of qubit A, N={int(dl['N'])} (cut-free)")
    Q = dl["cf_A_full_q_matrix"] if "cf_A_full_q_matrix" in dl else dl["ns_A_main_q_matrix"]
    zq = dl["cf_A_full_z_ee"][:Q.shape[0]] if "cf_A_full_z_ee" in dl else dl["ns_A_main_z_ee"][:Q.shape[0]]
    im = ax[2].imshow(np.log10(np.abs(Q - np.eye(Q.shape[0])) + 1e-16),
                      cmap="viridis", vmin=-8, vmax=0)
    ax[2].set_title(rf"$\log_{{10}}|q_{{eo}}-1|$, qubit A, N={int(dl['N'])}")
    ax[2].set_xlabel("k  (auxiliary state, ordered by weight $z_k$)")
    secax = [f"{z:.0e}" for z in zq]
    ax[2].set_yticks(range(len(secax)))
    ax[2].set_yticklabels(secax, fontsize=6)
    ax[2].set_ylabel("$z_k$", fontsize=8)
    plt.colorbar(im, ax=ax[2])
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"fig2_ns_test_{tag}.png"), dpi=150)
    plt.close(fig)

    # --- Fig 3: relation between the two noiseless subsystems
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogy(Ns, [d["split_e_ab_coherence"] for d in data], "o-",
                   label="A-B coherence left in the steady state")
    ax[0].semilogy(Ns, [d["split_eo_cross"] for d in data], "s-",
                   label="A-B cross terms in the coherence modes")
    ax[0].semilogy(Ns, [d["split_e_shape_mismatch"] for d in data], "^-",
                   label=r"$E^{-1}$ mode vs. $E^{+1}$ mode on one qubit")
    ax[0].semilogy(Ns, [d["rel_which_qubit_err_e"] for d in data], "v-",
                   label="which-qubit conserved quantity vs. $\\pm 1$")
    ax[0].set_xlabel("N"); ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set_title("direct-sum structure of the two qubits")
    ax[0].legend(fontsize=8)
    ax[1].semilogy(Ns, [d["rel_Gamma_pop"] for d in data], "o-", label="population")
    ax[1].semilogy(Ns, [d["rel_Gamma_intra"] for d in data], "s--",
                   label="coherence inside one qubit")
    ax[1].semilogy(Ns, [d["rel_Gamma_inter_ee"] for d in data], "^-",
                   label="coherence between the qubits")
    ax[1].set_xlabel("N"); ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].set_ylabel(r"instantaneous decay rate $\Gamma$")
    ax[1].set_title(r"$a^2$ distinguishes A from B, not even from odd")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"fig3_relation_{tag}.png"), dpi=150)
    plt.close(fig)

    # --- Fig 4: photon distribution of the exact steady state and the cut
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for d in data:
        P = np.abs(d["basin_P_ss"])
        n = np.arange(len(P))
        l, = ax.semilogy(n, P + 1e-300, label=f"N={int(d['N'])}")
        ax.axvline(int(d["n_cut_used"]), color=l.get_color(), ls=":", lw=1)
    ax.set_ylim(1e-14, 1)
    ax.set_xlabel("photon number n")
    ax.set_ylabel("P(n) of the exact steady state")
    ax.set_title("vacuum / superradiant bistability (dotted: cut)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"fig4_photon_distribution_{tag}.png"), dpi=150)
    plt.close(fig)

    # --- Fig 5: where the which-qubit memory is lost (collective spin flips)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for d in data:
        N = int(d["N"])
        prof = d["rel_spin_memory"]
        jx = d["rel_spin_mx"] / (N / 2.0)           # spin polarisation m_x/(N/2)
        ax.plot(jx, prof, "o-", label=f"N={N}")
    jstar = float(data[0]["wc"]) / (2.0 * float(data[0]["lam"]))
    for x in (jstar, -jstar):
        ax.axvline(x, color="k", ls="--", lw=1)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"spin polarisation $m_x/(N/2)$")
    ax.set_ylabel(r"which-qubit memory $\ell(m_x)$")
    ax.set_title(r"memory vs. spin polarisation (dashed: $\pm\omega_c/2\lambda$)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"fig5_spin_memory_{tag}.png"), dpi=150)
    plt.close(fig)

    # --- text table (cut-free test first; 'cut' columns = old Fock-cut method)
    # D/lam_c: dominant-state D_t(z_ee,z_eo) over the coherence decay rate in units of omega_c
    lines = ["N   M    k*  eps       |Re lam| E++-1  |Re lam| E+-i  "
             "cut-free: D_t(ee,oo) D_t(ee,eo)  dominant: D_t(ee,oo) D_t(ee,eo)  incoh.w.  q-err    D/lam_c | "
             "cut: n_cut D_t(ee,oo) D_t(ee,eo)  AB-coh   Gam_inter/Gam_intra"]
    for d in data:
        N = int(d["N"])
        lam_c = 0.5 * (abs(d["sel_E_+-_i"].real) + abs(d["sel_E_+-_-i"].real)) / float(d["wc"])
        lines.append(
            f"{N:<3d} {int(d['M']):<4d} {predicted_spin_flips(N, float(d['lam']), float(d['wc'])):<3d} "
            f"{g(d, 'cf_eps'):.2e}  {abs(d['sel_E_++_-1'].real):.2e}       {abs(d['sel_E_+-_i'].real):.2e}      "
            f"{g(d, 'cf_A_full_Dt_ee_oo'):.2e}  {g(d, 'cf_A_full_Dt_ee_eo'):.2e}            "
            f"{g(d, 'cf_A_dom_Dt_ee_oo'):.2e}  {g(d, 'cf_A_dom_Dt_ee_eo'):.2e}  "
            f"{g(d, 'cf_A_w_inc'):.2e}  {g(d, 'cf_A_full_q_error'):.1e}  {g(d, 'cf_A_dom_Dt_ee_eo') / lam_c:6.2f} | "
            f"{int(d['n_cut_used']):<5d} {float(d['ns_A_main_Dt_ee_oo']):.2e}  {float(d['ns_A_main_Dt_ee_eo']):.2e}    "
            f"{float(d['split_e_ab_coherence']):.1e}  "
            f"{float(d['rel_Gamma_inter_ee'])/max(float(d['rel_Gamma_intra']), 1e-300):.2e}")
    table = "\n".join(lines)
    with open(os.path.join(OUTDIR, f"summary_table_{tag}.txt"), "w") as fh:
        fh.write(table + "\n")
    print("\n" + table)


# ===========================================================================
# 6. MAIN
# ===========================================================================
def run_one_N(N, M, lam, kappa2, wc, wa, n_workers, mem_budget_gb):
    """Everything for one system size: build, solve the blocks, analyse,
    check the sensitivity to the vacuum cut, save, append to the summary."""
    t0 = time.time()
    print(f"\n##### N={N}: building Liouvillian (M={M}, Hilbert dim "
          f"{M*(N+1)}, Liouville dim {(M*(N+1))**2}) #####", flush=True)
    model = build_model(N, M, lam, kappa2, wc, wa)
    blocks = block_indices(model)
    n_cross = check_block_structure(model, blocks)
    print(f"    block structure check: {n_cross} matrix elements between "
          f"blocks (must be 0)", flush=True)
    t_build = time.time() - t0
    res = compute_all_blocks(model, blocks, n_workers, mem_budget_gb)
    t_eig = time.time() - t0 - t_build
    out = analyse(model, blocks, res)

    # robustness of the vacuum isolation: move the cut by +-CUT_SHIFT photons
    # (inside the valley the NS distances must not depend on where we cut)
    cuts, dt_oo, dt_eo = [], [], []
    for dn in (-CUT_SHIFT, 0, CUT_SHIFT):
        nc = max(out["n_cut_used"] + dn, 0)
        if dn != 0 and (nc >= out["basin"]["n_peak"] or nc in cuts):
            continue
        alt = out if dn == 0 else analyse(model, blocks, res, n_cut=nc, verbose=False)
        cuts.append(nc)
        dt_oo.append(alt["ns"][("A", "main")]["Dt_ee_oo"])
        dt_eo.append(alt["ns"][("A", "main")]["Dt_ee_eo"])
    spread = max(max(dt_oo) / min(dt_oo), max(dt_eo) / min(dt_eo))
    out["cut_scan"] = dict(cuts=np.array(cuts), Dt_ee_oo=np.array(dt_oo),
                           Dt_ee_eo=np.array(dt_eo), spread=spread)
    fmt = lambda xs: "[" + ", ".join(f"{x:.2e}" for x in xs) + "]"
    line = (f"    vacuum-cut sensitivity: cuts {cuts} -> D_t(z_ee,z_oo) "
            f"{fmt(dt_oo)}, D_t(z_ee,z_eo) {fmt(dt_eo)}  (max/min = {spread:.2f})")
    print(line, flush=True)
    timing = dict(build_s=t_build, eig_s=t_eig, total_s=time.time() - t0,
                  lu_nnz={b: int(res[b].get("nnz_lu", 0)) for b in res},
                  peak_gb={b: float(res[b].get("peak_gb", 0.0)) for b in res},
                  converged={b: bool(res[b].get("converged", True)) for b in res})
    out["report"] += "\n" + line + f"\n    time {timing['total_s']:.0f}s"
    save(out, res, timing)
    with open(os.path.join(OUTDIR, f"summary_{run_tag(lam, kappa2, wc, wa)}.txt"), "a") as fh:
        fh.write(out["report"] + "\n")
    print(f"    saved {result_path(N, lam, kappa2, wc, wa)}  ({timing['total_s']:.0f}s)", flush=True)


def main():
    """Command-line entry point (see --help); loops over N, then plots."""
    p = argparse.ArgumentParser(description="Noiseless-subsystem test of the superradiant phase of the dissipative two-photon Dicke model.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, nargs="+", default=N_LIST, help="system sizes")
    p.add_argument("--lam", type=float, default=LAM)
    p.add_argument("--kappa2", type=float, default=KAPPA2)
    p.add_argument("--wc", type=float, default=WC)
    p.add_argument("--wa", type=float, default=WA)
    p.add_argument("--M", type=int, nargs="+", default=None,
                   help="Fock cutoff(s); one value per N, or omit for the automatic rule")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel blocks (default: min(SLURM_CPUS_PER_TASK, MAX_WORKERS))")
    p.add_argument("--mem-budget", type=float, default=MEM_BUDGET_GB,
                   help="memory (GB) the parallel blocks may use together; the number "
                        "of simultaneous blocks is reduced to fit (default %(default)s)")
    p.add_argument("--plot-only", action="store_true",
                   help="only (re)make the summary figures/table from saved results")
    args = p.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    if not args.plot_only:
        n_workers = args.workers or min(int(os.environ.get("SLURM_CPUS_PER_TASK",
                                                           mp.cpu_count())), MAX_WORKERS)
        print(f"qutip {qt.__version__}, numpy {np.__version__}, workers={n_workers}")
        for i, N in enumerate(args.N):
            if args.M is None:
                M = auto_cutoff(N, args.lam, args.kappa2, args.wc)
            else:
                M = args.M[i] if len(args.M) == len(args.N) else args.M[0]
            try:
                run_one_N(N, M, args.lam, args.kappa2, args.wc, args.wa, n_workers,
                          args.mem_budget)
            except Exception:                      # keep going with the next N
                import traceback
                traceback.print_exc()
                print(f"!!! analysis for N={N} failed (see traceback); continuing",
                      flush=True)
    make_plots(args.lam, args.kappa2, args.wc, args.wa)


if __name__ == "__main__":
    main()
