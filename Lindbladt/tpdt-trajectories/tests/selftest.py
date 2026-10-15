#!/usr/bin/env python
"""
Self-test: run this once on the cluster (login node is fine, ~1-2 min) before
submitting a sweep.

  1. The parity-sector operators equal the full-space operators (as in the
     original notebook) projected onto the sector -- exactly.
  2. mesolve in the sector reproduces mesolve in the full space, i.e. the
     even/odd restriction loses nothing.
  3. The trajectory average of the parallel runner agrees with the exact
     mesolve evolution of the infinite-temperature initial state within
     statistical error, in both sectors, for serial and parallel execution.
  4. run_point.py runs end to end and writes all output files.

Usage:  python tests/selftest.py           (from the repository root)
"""
import os
import sys
import tempfile
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import qutip as qt

from tpdt.config import RunConfig
from tpdt.model import build_full_model, build_sector_model, sector_fock_numbers
from tpdt.trajectories import load_raw, raw_path, run_sector

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILED.append(name)


def projector(N, M, sector):
    """Isometry from the sector space into the full cavity x spin space."""
    fock = sector_fock_numbers(M, sector)
    P = np.zeros((M, fock.size))
    P[fock, np.arange(fock.size)] = 1.0
    return np.kron(P, np.eye(N + 1))


def test_operators(N=3, M=14, lam=0.7, kappa2=0.05):
    H, c_ops, a = build_full_model(N, M, lam, kappa2)
    for sector in ("even", "odd"):
        sm = build_sector_model(N, M, lam, kappa2, sector=sector)
        V = projector(N, M, sector)
        dH = np.abs(V.T @ H.full() @ V - sm.H.full()).max()
        dL = np.abs(V.T @ c_ops[0].full() @ V - sm.c_ops[0].full()).max()
        # the full operators must not couple the sector to the other parity
        Q = np.eye(V.shape[0]) - V @ V.T
        leak = max(np.abs(Q @ H.full() @ V).max(), np.abs(Q @ c_ops[0].full() @ V).max())
        check(f"operators [{sector}]", max(dH, dL, leak) < 1e-12,
              f"(max |dH| = {dH:.1e}, |dL| = {dL:.1e}, parity leak = {leak:.1e})")


def test_mesolve_equivalence(N=2, M=16, lam=0.8, kappa2=0.05, t=4.0):
    H, c_ops, a = build_full_model(N, M, lam, kappa2)
    tl = np.linspace(0, t, 5)
    opts = {"atol": 1e-10, "rtol": 1e-8, "progress_bar": False}
    for sector in ("even", "odd"):
        sm = build_sector_model(N, M, lam, kappa2, sector=sector)
        V = projector(N, M, sector)
        D = V.shape[1]
        rho0_s = qt.Qobj(np.eye(D) / D, dims=sm.H.dims)
        rho0_f = qt.Qobj(V @ rho0_s.full() @ V.T, dims=H.dims)
        r_s = qt.mesolve(sm.H, rho0_s, tl, sm.c_ops, options=opts).states[-1]
        r_f = qt.mesolve(H, rho0_f, tl, c_ops, options=opts).states[-1]
        diff = np.abs(V.T @ r_f.full() @ V - r_s.full()).max()
        outside = 1.0 - np.trace(V.T @ r_f.full() @ V).real
        check(f"mesolve sector == full space [{sector}]", diff < 1e-6 and abs(outside) < 1e-8,
              f"(max diff = {diff:.1e}, weight outside sector = {outside:.1e})")


def exact_reference(cfg, sector):
    sm = build_sector_model(cfg.N, cfg.M, cfg.lam, cfg.kappa2, cfg.wc, cfg.wa, sector)
    D = sm.dim_cav * sm.dim_spin
    rho0 = qt.Qobj(np.eye(D) / D, dims=sm.H.dims)
    tl = np.linspace(0, cfg.t_max, cfg.n_t)
    res = qt.mesolve(sm.H, rho0, tl, sm.c_ops, e_ops=[sm.n_op, sm.Jz],
                     options={"atol": 1e-10, "rtol": 1e-8, "progress_bar": False,
                              "store_final_state": True})
    rho_c = res.final_state.ptrace(0).full()
    a4 = qt.expect(sm.a2_op * sm.a2_op, res.final_state)
    jxa2 = qt.expect(sm.Jx * sm.a2_op, res.final_state)
    return res.expect[0], res.expect[1], rho_c, (a4, jxa2)


def test_trajectories(tmp, workers):
    cfg = RunConfig(N=2, M=16, lam=0.8, kappa2=0.05, t_max=4.0, n_t=9,
                    n_traj=600, seed=7)
    for sector in ("even", "odd"):
        pdir = Path(tmp) / f"w{workers}" / sector
        run_sector(cfg, sector, pdir, workers=workers, log=lambda m: None)
        st = load_raw(raw_path(pdir, sector, cfg.seed))
        n = st["n_done"]
        n_ex, jz_ex, rho_ex, (a4_ex, jxa2_ex) = exact_reference(cfg, sector)
        n_mean = st["n_t"] / n
        n_se = np.sqrt(np.maximum(st["n_t2"] / n - n_mean ** 2, 0) / (n - 1))
        jz_mean = st["jz_t"] / n
        jz_se = np.sqrt(np.maximum(st["jz_t2"] / n - jz_mean ** 2, 0) / (n - 1))
        zn = np.max(np.abs(n_mean[1:] - n_ex[1:]) / n_se[1:])
        zj = np.max(np.abs(jz_mean[1:] - jz_ex[1:]) / np.maximum(jz_se[1:], 1e-12))
        za4 = 0.0
        for vals, ex in ((st["fin"]["a4"], a4_ex), (st["fin"]["jxa2"], jxa2_ex)):
            for part in (np.real, np.imag):
                v = part(vals)
                za4 = max(za4, abs(v.mean() - part(ex)) / (v.std(ddof=1) / np.sqrt(n) + 1e-12))
        rho_tr = st["rho_c"] / n
        tdist = 0.5 * np.abs(np.linalg.eigvalsh(rho_tr - rho_ex)).sum()
        # max over 8 time points of |z|: < 4.5 fails with prob ~ 5e-5 if correct
        check(f"trajectories vs mesolve [{sector}, workers={workers}]",
              zn < 4.5 and zj < 4.5 and za4 < 4.5 and tdist < 0.15,
              f"(max |z|: <n> {zn:.2f}, <Jz> {zj:.2f}, <a^4>,<Jx a^2> {za4:.2f}; trace distance rho_cav = {tdist:.3f}"
              f" with {n} trajectories)")


def test_end_to_end(tmp):
    import subprocess
    root = Path(__file__).resolve().parents[1]
    out = Path(tmp) / "e2e"
    cmd = [sys.executable, str(root / "run_point.py"), "--N", "2", "--M", "16", "--lam", "0.8",
           "--t-max", "3", "--n-t", "7", "--n-traj", "16", "--workers", "2", "--outdir", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    pdir = next(out.glob("*/lam*"), None)
    expected = ["raw_even_seed1.npz", "raw_odd_seed1.npz", "summary_even.npz", "summary_odd.npz",
                "summary.json", "wigner_pn.png", "wigner_pn.pdf", "diagnostics.png"]
    missing = [f for f in expected if pdir is None or not (pdir / f).exists()]
    check("run_point.py end to end", r.returncode == 0 and not missing,
          f"(missing: {missing})" if missing else "")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-4000:])


if __name__ == "__main__":
    print(f"qutip {qt.__version__}, numpy {np.__version__}")
    test_operators()
    test_mesolve_equivalence()
    with tempfile.TemporaryDirectory() as tmp:
        test_trajectories(tmp, workers=1)
        test_trajectories(tmp, workers=2)
        test_end_to_end(tmp)
    print("\nALL TESTS PASSED" if not FAILED else f"\n{len(FAILED)} TEST(S) FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
