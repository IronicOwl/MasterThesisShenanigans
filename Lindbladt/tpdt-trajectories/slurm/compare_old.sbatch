#!/bin/bash
#SBATCH --job-name=tpdt_compare_old
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source slurm/env.sh
python - <<'PY'
import json, pickle
from pathlib import Path
import numpy as np

N = 15
OLD = Path.home() / "masterarbeit/MasterThesisShenanigans/Plots/Tpdt_SC/Plot_Data"
NEW = Path("results/n15_check/N15_k0.05_wc1_wa1_t70_Tinf")
POINTS = [(0.53, 200, "053"), (0.54, 210, "054"), (0.60, 200, "06"), (0.60, 320, None)]

def old_n_over_N(path, M):
    with open(path, "rb") as fh:
        rho = pickle.load(fh, encoding="latin1")
    r = rho.full() if hasattr(rho, "full") else np.asarray(rho)
    dims = rho.dims[0] if hasattr(rho, "dims") else None
    Ns = N + 1
    if r.shape[0] == M:                          # already the cavity state
        rc = r
    elif dims is not None and list(dims) == [Ns, M]:   # spin (x) cavity
        rc = np.einsum("sisj->ij", r.reshape(Ns, M, Ns, M))
    else:                                        # cavity (x) spin
        rc = np.einsum("isjs->ij", r.reshape(M, Ns, M, Ns))
    p = np.real(np.diag(rc))
    return (p @ np.arange(M)) / p.sum() / N

print(f"{'lambda':>6} {'M':>4} {'sector':>6} | {'old <n>/N (500 traj)':>20} | new <n>/N (2000 traj)")
for lam, M, tag in POINTS:
    js = NEW / f"lam{lam:.4f}_M{M}" / "summary.json"
    rep = json.loads(js.read_text()) if js.exists() else None
    for sec, s in (("even", "E"), ("odd", "O")):
        old = "-"
        if tag is not None:
            f = OLD / f"TPDT_N15_M{M}_l{tag}_t70_n500_{s}_full.qu"
            old = f"{old_n_over_N(f, M):.3f}" if f.exists() else "file not found"
        new = "not finished yet"
        if rep and sec in rep["sectors"]:
            sc = rep["sectors"][sec]
            new = f"{sc['n_over_N']:.3f} +- {sc['n_over_N_se']:.3f}"
            if not sc["truncation_ok"]:
                new += f"   (truncated: P(n>={sc['tail_start']}) = {sc['tail_weight']:.1e})"
        print(f"{lam:6.2f} {M:4d} {sec:>6} | {old:>20} | {new}")
PY
