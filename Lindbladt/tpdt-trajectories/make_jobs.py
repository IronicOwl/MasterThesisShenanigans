#!/usr/bin/env python
"""
Expand a sweep definition (JSON) into a SLURM job list: one line of
run_point.py arguments per array task.

    python make_jobs.py sweeps/example.json
        -> jobs/<tag>.txt          one task per line
        -> jobs/<tag>.sbatch_args  resources for sbatch (read by slurm/submit.sh)

Sweep file keys
---------------
tag        name of the sweep (used for job files and as results subfolder)
outdir     results root (default "results"); data go to <outdir>/<tag>/...
common     run_point.py options shared by all tasks, e.g.
           {"kappa2": 0.05, "t_max": 70, "n_traj": 2000}
systems    list of system sizes, each {"N": .., "M": .., optional overrides};
           "M": "auto" picks the cutoff per lambda from the superradiant estimate
           (tpdt.config.auto_cutoff; tune with "M_factor", default 1.6)
lam        list of couplings; or {"start":..,"stop":..,"num":..} (inclusive linspace)
chunks     split each point into this many independent jobs with different
           seeds (n_traj is per chunk!). Default 1.
split_sectors  true: separate tasks for even and odd (default false)
seed       base seed (default 1); chunk c uses seed + c
slurm      {"cpus_per_task": 32, "time": "24:00:00", "mem_per_cpu": "1G",
            "max_parallel": 20, "extra": ["--partition=...", "--account=..."]}
"""
import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np

from tpdt.config import auto_cutoff

ALLOWED = {"N", "M", "lam", "kappa2", "wc", "wa", "t_max", "n_t", "n_traj", "seed", "init",
           "init_nmax", "sampling", "method", "atol", "rtol", "nsteps", "batch_size",
           "checkpoint_min", "wigner_grid", "wigner_xmax"}


def lam_values(spec):
    if isinstance(spec, dict):
        return [float(x) for x in np.round(np.linspace(spec["start"], spec["stop"], spec["num"]), 6)]
    return [float(x) for x in spec]


def to_args(opts: dict) -> list[str]:
    out = []
    for k, v in opts.items():
        if v is None:
            continue
        if k not in ALLOWED:
            raise ValueError(f"unknown option {k!r} in sweep file")
        out += [f"--{k.replace('_', '-')}", str(v)]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep", help="sweep JSON file")
    p.add_argument("--jobs-dir", default="jobs")
    p.add_argument("--print-jobfile", action="store_true", help="only print the job file path")
    args = p.parse_args(argv)

    sw = json.loads(Path(args.sweep).read_text())
    tag = sw["tag"]
    outdir = Path(sw.get("outdir", "results")) / tag
    common = dict(sw.get("common", {}))
    chunks = int(sw.get("chunks", 1))
    split = bool(sw.get("split_sectors", False))
    base_seed = int(sw.get("seed", common.pop("seed", 1)))
    lams = lam_values(sw["lam"])
    sector_sets = [["even"], ["odd"]] if split else [["even", "odd"]]
    needs_finalize_job = chunks > 1 or split

    lines = []
    for system in sw["systems"]:
        for lam in lams:
            for c in range(chunks):
                for sectors in sector_sets:
                    opts = {**common, **system, "lam": lam, "seed": base_seed + c}
                    factor = float(opts.pop("M_factor", 1.6))
                    if str(opts.get("M", "auto")).lower() == "auto":
                        opts["M"] = auto_cutoff(int(opts["N"]), lam, float(opts.get("kappa2", 0.05)),
                                                float(opts.get("wc", 1.0)), factor=factor)
                    a = to_args(opts) + ["--sectors", *sectors, "--outdir", str(outdir)]
                    if needs_finalize_job:
                        a.append("--no-finalize")
                    lines.append(shlex.join(a))

    jobs_dir = Path(args.jobs_dir)
    jobs_dir.mkdir(exist_ok=True)
    jobfile = jobs_dir / f"{tag}.txt"
    jobfile.write_text("\n".join(lines) + "\n")

    sl = sw.get("slurm", {})
    sb = [f"--job-name=tpdt_{tag}",
          f"--cpus-per-task={sl.get('cpus_per_task', 32)}",
          f"--time={sl.get('time', '24:00:00')}",
          f"--mem-per-cpu={sl.get('mem_per_cpu', '1G')}",
          *sl.get("extra", [])]
    (jobs_dir / f"{tag}.sbatch_args").write_text(" ".join(sb) + "\n")
    (jobs_dir / f"{tag}.max_parallel").write_text(f"{sl.get('max_parallel', 20)}\n")
    (jobs_dir / f"{tag}.outdir").write_text(f"{outdir}\n")

    if args.print_jobfile:
        print(jobfile)
    else:
        print(f"{len(lines)} tasks -> {jobfile}")
        print(f"results -> {outdir}")
        print(f"submit with: bash slurm/submit.sh {args.sweep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
