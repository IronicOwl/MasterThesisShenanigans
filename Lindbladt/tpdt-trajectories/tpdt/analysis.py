"""
Merge raw trajectory files (any number of seeds) of one parameter point and
derive everything that goes into the thesis: the parity-resolved cavity
density matrix, P(n), the Wigner function, spin observables, error bars and
convergence diagnostics.

Files written into each point directory
---------------------------------------
summary_<sector>.npz   arrays (cavity rho on the sector, P(n), Wigner, traces, ...)
summary.json           scalar results + diagnostics for both sectors (git-diffable)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import keys_equal
from .model import SECTORS, embed_cavity_rho
from .trajectories import load_raw

TAIL_FRACTION = 0.9        # truncation check: weight in the top 10 % of the Fock space
TAIL_WARN = 1e-3
DRIFT_WARN_SIGMA = 3.0


# ---------------------------------------------------------------------------
# merging
# ---------------------------------------------------------------------------

def raw_files(point_dir: Path, sector: str) -> list[Path]:
    return sorted(Path(point_dir).glob(f"raw_{sector}_seed*.npz"))


def merge_raws(point_dir: Path, sector: str) -> dict | None:
    files = raw_files(point_dir, sector)
    if not files:
        return None
    merged = None
    for f in files:
        st = load_raw(f)
        if st["n_done"] == 0:
            continue
        if merged is None:
            merged = st
            merged["seeds"] = [st["meta"]["seed"]]
            merged["n_traj_per_seed"] = [st["n_done"]]
            continue
        if not keys_equal(st["meta"]["physics_key"], merged["meta"]["physics_key"]):
            raise RuntimeError(f"{f.name} has different physics parameters than "
                               f"{files[0].name}; refusing to merge.")
        for key in ("rho_c", "rho_s", "n_t", "n_t2", "jz_t", "jz_t2"):
            merged[key] = merged[key] + st[key]
        for key in merged["fin"]:
            merged["fin"][key] = np.concatenate([merged["fin"][key], st["fin"][key]])
        merged["n_done"] += st["n_done"]
        merged["compute_seconds"] += st["compute_seconds"]
        merged["seeds"].append(st["meta"]["seed"])
        merged["n_traj_per_seed"].append(st["n_done"])
    return merged


# ---------------------------------------------------------------------------
# derived quantities
# ---------------------------------------------------------------------------

def _mean_se(s1: np.ndarray, s2: np.ndarray, n: int):
    mean = s1 / n
    var = np.maximum(s2 / n - mean ** 2, 0.0)
    se = np.sqrt(var / max(n - 1, 1))
    return mean, se


def wigner_extent(P_n: np.ndarray, quantile: float = 0.999) -> float:
    """Half-width of the phase-space window (QuTiP convention g = sqrt(2), so a
    coherent state |alpha> sits at x = sqrt(2) Re(alpha))."""
    cdf = np.cumsum(P_n)
    n_hi = int(np.searchsorted(cdf, quantile * cdf[-1]))
    return float(np.ceil(np.sqrt(2.0 * max(n_hi, 1)) + 3.0))


def sector_summary(merged: dict) -> dict:
    key = merged["meta"]["physics_key"]
    N, M = key["N"], key["M"]
    n = merged["n_done"]
    fock = np.asarray(merged["fock"])

    rho_c = merged["rho_c"] / n
    tr = np.trace(rho_c).real
    rho_c = rho_c / tr
    rho_c = 0.5 * (rho_c + rho_c.conj().T)
    P_sector = np.clip(np.real(np.diag(rho_c)), 0.0, None)
    P_n = np.zeros(M)
    P_n[fock] = P_sector

    n_vals = np.arange(M)
    n_mean = float(P_n @ n_vals)
    n2_mean = float(P_n @ n_vals ** 2)
    var_n = n2_mean - n_mean ** 2
    fin_n = merged["fin"]["n"]
    n_se = float(np.std(fin_n, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

    tail_start = int(np.ceil(TAIL_FRACTION * M))
    tail_weight = float(P_n[tail_start:].sum())

    # spin
    rho_s = merged["rho_s"] / n
    rho_s = rho_s / np.trace(rho_s).real
    jz_diag = N - 2.0 * np.arange(N + 1)          # eigenvalues of 2*jmat('z'), qutip order
    jz_mean = float(np.real(np.diag(rho_s)) @ jz_diag)
    jx = _jx_matrix(N)
    jx2_mean = float(np.real(np.trace(rho_s @ jx @ jx)))

    # time traces and convergence of the relaxation
    tlist = np.asarray(merged["tlist"])
    n_t_mean, n_t_se = _mean_se(merged["n_t"], merged["n_t2"], n)
    jz_t_mean, jz_t_se = _mean_se(merged["jz_t"], merged["jz_t2"], n)
    i_ref = int(0.8 * (tlist.size - 1))
    drift = float(n_t_mean[-1] - n_t_mean[i_ref])
    drift_sigma = float(abs(drift) / max(np.hypot(n_t_se[-1], n_t_se[i_ref]), 1e-12))

    a4 = merged["fin"]["a4"]
    return {
        "arrays": {
            "rho_sector": rho_c, "fock": fock, "P_n": P_n,
            "rho_spin": rho_s,
            "tlist": tlist, "n_t_mean": n_t_mean, "n_t_se": n_t_se,
            "jz_t_mean": jz_t_mean, "jz_t_se": jz_t_se,
            "final_n": fin_n, "final_jz": merged["fin"]["jz"], "final_jxa2": merged["fin"]["jxa2"],
            "final_a4": a4, "init_n": merged["fin"]["init_n"], "init_s": merged["fin"]["init_s"],
        },
        "scalars": {
            "n_traj": int(n),
            "seeds": [int(s) for s in merged["seeds"]],
            "n_traj_per_seed": [int(c) for c in merged["n_traj_per_seed"]],
            "trace_before_renorm": float(tr),
            "n_mean": n_mean, "n_mean_se": n_se,
            "n_over_N": n_mean / N, "n_over_N_se": n_se / N,
            "var_n": float(var_n),
            "mandel_Q": float((var_n - n_mean) / n_mean) if n_mean > 0 else float("nan"),
            "Jz_over_N": jz_mean / N, "Jx2_over_N2": jx2_mean / N ** 2,
            # Tr(a^4 rho)/N^2 (= trajectory mean) and the mean modulus per trajectory
            "a4_over_N2": [float(np.mean(a4.real) / N ** 2), float(np.mean(a4.imag) / N ** 2)],
            "abs_a4_over_N2_traj_mean": float(np.mean(np.abs(a4)) / N ** 2),
            "Jx_a2_over_N2": [float(np.mean(merged["fin"]["jxa2"].real) / N ** 2),
                              float(np.mean(merged["fin"]["jxa2"].imag) / N ** 2)],
            "tail_start": tail_start, "tail_weight": tail_weight,
            "truncation_ok": bool(tail_weight < TAIL_WARN),
            "relax_drift_n_last20pct": drift, "relax_drift_sigma": drift_sigma,
            "relaxation_ok": bool(drift_sigma < DRIFT_WARN_SIGMA),
            "compute_core_hours": float(merged["compute_seconds"] / 3600.0),
        },
    }


def _jx_matrix(N: int) -> np.ndarray:
    import qutip as qt
    return 2.0 * qt.jmat(N / 2.0, "x").full()


def compute_wigner(rho_full: np.ndarray, x_max: float, n_grid: int):
    import qutip as qt
    xvec = np.linspace(-x_max, x_max, n_grid)
    W = qt.wigner(qt.Qobj(rho_full), xvec, xvec)
    return xvec, np.asarray(W)


# ---------------------------------------------------------------------------
# finalize one point directory
# ---------------------------------------------------------------------------

def finalize_point(point_dir: Path, n_grid: int = 201, x_max: float | None = None,
                   make_plots: bool = True, log=print) -> dict | None:
    point_dir = Path(point_dir)
    merged = {s: merge_raws(point_dir, s) for s in SECTORS}
    merged = {s: m for s, m in merged.items() if m is not None}
    if not merged:
        log(f"[finalize] {point_dir}: no trajectories yet")
        return None

    keys = [m["meta"]["physics_key"] for m in merged.values()]
    if len(keys) == 2 and not keys_equal(keys[0], keys[1]):
        raise RuntimeError(f"{point_dir}: even and odd runs have different parameters")
    key = keys[0]

    summaries = {s: sector_summary(m) for s, m in merged.items()}

    # common phase-space window for both sectors so the panels are comparable
    if x_max is None:
        x_max = max(wigner_extent(sm["arrays"]["P_n"]) for sm in summaries.values())
    for s, sm in summaries.items():
        rho_full = embed_cavity_rho(sm["arrays"]["rho_sector"], sm["arrays"]["fock"], key["M"])
        xvec, W = compute_wigner(rho_full, x_max, n_grid)
        sm["arrays"]["xvec"] = xvec
        sm["arrays"]["W"] = W.astype(np.float32)
        sm["scalars"]["wigner_min"] = float(W.min())
        sm["scalars"]["wigner_max"] = float(W.max())
        sm["scalars"]["wigner_negativity_volume"] = float(
            np.sum(np.abs(W) - W) / 2 * (xvec[1] - xvec[0]) ** 2)
        tmp = point_dir / f"summary_{s}.tmp.npz"
        np.savez_compressed(tmp, M=np.int64(key["M"]), N=np.int64(key["N"]),
                            lam=np.float64(key["lam"]), **sm["arrays"])
        tmp.replace(point_dir / f"summary_{s}.npz")

    warnings = []
    for s, sm in summaries.items():
        sc = sm["scalars"]
        if not sc["truncation_ok"]:
            warnings.append(f"{s}: P(n >= {sc['tail_start']}) = {sc['tail_weight']:.2e} "
                            f"> {TAIL_WARN:g} -> increase M")
        if not sc["relaxation_ok"]:
            warnings.append(f"{s}: <n>(t) still drifting over the last 20% of t_max "
                            f"({sc['relax_drift_sigma']:.1f} sigma) -> increase t_max")

    report = {
        "physics_key": key,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wigner": {"x_max": x_max, "n_grid": n_grid, "g": "sqrt(2) (qutip default)"},
        "sectors": {s: sm["scalars"] for s, sm in summaries.items()},
        "warnings": warnings,
    }
    tmp = point_dir / "summary.json.tmp"
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(point_dir / "summary.json")

    for w in warnings:
        log(f"[finalize] WARNING {point_dir.name}: {w}")
    log(f"[finalize] {point_dir}: " + ", ".join(
        f"{s}: {sm['scalars']['n_traj']} traj, <n>/N = {sm['scalars']['n_over_N']:.3f}"
        f" +- {sm['scalars']['n_over_N_se']:.3f}" for s, sm in summaries.items()))

    if make_plots:
        from .plotting import plot_diagnostics, plot_point
        plot_point(point_dir)
        plot_diagnostics(point_dir)
    return report


# ---------------------------------------------------------------------------
# loading (for notebooks / thesis figures)
# ---------------------------------------------------------------------------

def load_summary(point_dir: Path, sector: str) -> dict:
    """All arrays of summary_<sector>.npz as a dict (+ 'scalars' from summary.json)."""
    point_dir = Path(point_dir)
    with np.load(point_dir / f"summary_{sector}.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    rep = json.loads((point_dir / "summary.json").read_text())
    d["scalars"] = rep["sectors"][sector]
    d["physics_key"] = rep["physics_key"]
    return d


def load_cavity_state(point_dir: Path, sector: str):
    """Reduced cavity density matrix (full M x M Fock space) as a qutip.Qobj."""
    import qutip as qt
    d = load_summary(point_dir, sector)
    return qt.Qobj(embed_cavity_rho(d["rho_sector"], d["fock"], int(d["M"])))


def find_point_dirs(root: Path) -> list[Path]:
    """Every directory below root that contains raw or summary files."""
    root = Path(root)
    dirs = {p.parent for p in root.rglob("raw_*_seed*.npz")}
    dirs |= {p.parent for p in root.rglob("summary.json")}
    return sorted(dirs)
