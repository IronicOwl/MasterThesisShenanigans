# Parity-resolved quantum trajectories for the two-photon Dicke model

Cluster code (SLURM + QuTiP 5) for the driven-dissipative two-photon Dicke model with
**two-photon loss only**,

$$H = \omega_c a^\dagger a + \tfrac{\omega_a}{2} J_z + \tfrac{\lambda}{N} J_x (a^{\dagger 2} + a^2), \qquad L_2 = \sqrt{\kappa_2/N}\, a^2 ,$$

with $J_{x,z} = 2\,\mathrm{jmat}(N/2)$ as in Shah et al. (arXiv:2412.14271) and the
original `Coherence_resolvedTpDt.ipynb`. For every parameter point it returns, separately
for the **even** and **odd** photon-parity sectors (strong $\mathbb Z_2$), the
coherence-resolved reduced cavity state, its Wigner function, $P(n)$, spin observables,
error bars and convergence diagnostics, plus figures. Everything is stored as small
`.npz`/`.json` files that can be committed to git.

## What changed compared to the notebook

| | notebook | here |
|---|---|---|
| state space | full $M \times (N+1)$ | one parity sector, $\lceil M/2\rceil \times (N+1)$ — **exact** (H and $L_2$ commute with $e^{i\pi a^\dagger a}$; checked in `tests/selftest.py`) |
| trajectories | Python loop, `mcsolve(ntraj=1)` per call | `MCSolver` built once per worker, `start`/`step`, process pool over all cores, SLURM array over parameter points |
| initial states | uniform over the sector basis (infinite temperature) | same (default), stratified so every basis state is used before any repeats; or `--init vacuum`; `--init-nmax` caps the photon number |
| output | `qsave` pickle of $\rho_\mathrm{cav}$, one plot | raw accumulators + summaries (`npz`, `json`), $\pm$ s.e., Wigner + $P(n)$ figure, diagnostics figure |
| robustness | — | checkpoint every 10 min, resume, merge any number of seeds, truncation and relaxation warnings |

Only quantities of size $(M/2)^2$ or $(N+1)^2$ are accumulated, never the full $\rho$, so memory
stays small even for $N\sim 50$, $M\sim 600$.

## Quick start on the cluster

```bash
git clone <your repo> && cd tpdt-trajectories
python -m venv ~/venvs/tpdt && source ~/venvs/tpdt/bin/activate
pip install -r requirements.txt
# edit slurm/env.sh: module load + path of the venv (TPDT_VENV)

python tests/selftest.py                    # ~1 min, must print ALL TESTS PASSED
bash slurm/submit.sh sweeps/test_small.json # tiny sweep, checks the SLURM chain
bash slurm/submit.sh sweeps/example_highres.json
```

`submit.sh` writes `jobs/<tag>.txt` (one `run_point.py` call per line), submits a job array over
it and a dependent `finalize` job. Logs go to `logs/`, results to `results/<tag>/`.

A single point without SLURM (any machine):

```bash
python run_point.py --N 15 --M 200 --lam 0.53 --n-traj 2000 --workers 32
```

## Sweep files

```json
{
  "tag": "highres_v1",
  "common":  {"kappa2": 0.05, "t_max": 70, "n_traj": 2000},
  "systems": [{"N": 15, "M": "auto"}, {"N": 25, "M": "auto"}],
  "lam":     [0.50, 0.51, 0.52, 0.53, 0.55, 0.60],
  "chunks": 1, "split_sectors": false,
  "slurm":   {"cpus_per_task": 32, "time": "24:00:00", "mem_per_cpu": "1G",
              "max_parallel": 20, "extra": ["--partition=normal"]}
}
```

* every key of `common`/`systems` is a `run_point.py` option (`t_max` → `--t-max`, ...);
  a system can override common options, e.g. `{"N": 40, "M": 520, "t_max": 120}`
* `"M": "auto"` chooses the Fock cutoff separately for every λ (see *Choosing M* below);
  `"M_factor"` scales it (default 1.6). `sweeps/cutoff_check.json` runs the same points with two
  cutoffs and a fixed initial ensemble to verify convergence in M.
* `lam` can also be `{"start": 0.5, "stop": 0.6, "num": 11}`
* `chunks: K` splits each point into K jobs with seeds `seed … seed+K-1` (`n_traj` is then
  per chunk); `split_sectors: true` runs even and odd as separate jobs. Both are only needed
  when one job per point would exceed the wall-time limit; the finalize job merges them.

## Outputs per point: `results/<tag>/N{N}_k{κ2}_wc{ωc}_wa{ωa}_t{t}_{init}/lam{λ}_M{M}/`

| file | content |
|---|---|
| `wigner_pn.png/.pdf` | (a,b) Wigner function even/odd, (c,d) $P(n)$ with cutoff, tail weight and $\langle n\rangle/N\pm$ s.e. |
| `diagnostics.png` | (a,b) $\langle n\rangle(t)/N$, $\langle J_z\rangle(t)/N$ with $\pm2$ s.e. bands — is $t_\mathrm{max}$ long enough? (c) $\log P(n)$ — is $M$ large enough? (d) histogram of the per-trajectory $\langle n\rangle/N$ at $t_\mathrm{max}$ (vacuum basin vs. lobes), (e) per-trajectory $\langle a^4\rangle/N^2$ (≈ 0: vacuum basin; lobes: phase $=4\theta$ of the lobe angle — one cluster, since all four lobes give the same value), (f) initial vs. final photon number |
| `summary.json` | parameters and scalar results per sector: `n_over_N ± se`, Mandel Q, `Jz_over_N`, `Jx2_over_N2`, $\mathrm{Tr}(a^4\rho)/N^2$, tail weight, relaxation drift, Wigner negativity volume, core-hours, warnings |
| `summary_<sector>.npz` | `rho_sector` (cavity state on the sector Fock states `fock`), `P_n`, `xvec`, `W`, `rho_spin`, time traces, per-trajectory final values and initial states |
| `raw_<sector>_seed<s>.npz` | checkpoint / raw accumulators of one run; merged by `finalize.py` |

Example (`python run_point.py --N 6 --M auto --lam 0.6 --n-traj 300 --workers 2`, ≈ 11 min on 2 cores):
[`docs/example_wigner_pn.png`](docs/example_wigner_pn.png), [`docs/example_diagnostics.png`](docs/example_diagnostics.png).

Loading in a notebook:

```python
from tpdt.analysis import load_cavity_state, load_summary
rho = load_cavity_state("results/highres_v1/N15_k0.05_wc1_wa1_t70_Tinf/lam0.5300_M180", "even")  # qutip.Qobj, M x M
d   = load_summary(".../lam0.5300_M180", "odd")   # dict: P_n, W, xvec, rho_sector, scalars, ...
```

Thesis figures across a sweep:

```bash
python plot_sweep.py grid  results/highres_v1/N20_k0.05_wc1_wa1_t70_Tinf --lam 0.51 0.53 0.6 --xmax 40 --out figures/grid_N20.pdf
python plot_sweep.py curve results/highres_v1/N*_Tinf --out figures/n_vs_lambda.pdf
```

## More trajectories, resuming, re-plotting

* **Time-out / node failure:** `bash slurm/resubmit.sh <tag> 4,9` — tasks resume from their last
  checkpoint (finished sectors are skipped). Shortly before the time limit SLURM sends a signal and
  the run saves a checkpoint before exiting.
* **More statistics:** run the same point again with a new `--seed` (or bump `"seed"` in the sweep
  file and resubmit). `finalize.py` merges all seeds of a point; runs with different physics
  parameters are never merged (it raises instead).
* **Re-plot / re-merge:** `python finalize.py results/<tag>` (only points with new raw data;
  `--force` for all, `--wigner-xmax 20` for a fixed window).

## Choosing M, t_max and the number of trajectories

* **Fock cutoff:** the superradiant lobes sit far out. With two-photon loss only, the steady-state
  relation $4\lambda^2\langle J_x\rangle^2=\kappa_2^2(\langle n\rangle/N)^2+\omega_c^2$ with $|\langle J_x\rangle|\le1$
  bounds the lobe at $\langle n\rangle/N \le \sqrt{4\lambda^2-\omega_c^2}/\kappa_2$, i.e. ≈ 7 at λ = 0.53, 9 at
  0.55, 13 at 0.60 for κ₂ = 0.05. For N = 15 that is n ≈ 140 at λ = 0.55 — beyond the old cutoff
  M = 120, which is probably what looked like a finite-size effect there. `M auto` uses
  $M = 1.6\,N\sqrt{4\lambda^2-\omega_c^2}/\kappa_2$ (at least max(60, 6N)). It is an estimate: always check
  `tail_weight` = $P(n \ge 0.9M)$ in `summary.json` / the $P(n)$ panel (warning above $10^{-3}$), and
  confirm a few points with a larger cutoff (`sweeps/cutoff_check.json`).
* **t_max:** near the transition relaxation slows down. If the diagnostics show $\langle n\rangle(t)$
  still drifting over the last 20 % of the window (`relax_drift_sigma` > 3), increase `t_max`.
* **Initial states (important for comparisons):** in the coexistence region the slowest Liouvillian
  mode can be far longer than `t_max`, so the result at `t_max` depends on how many trajectories start
  in the basin of the vacuum vs. the superradiant lobes. The infinite-temperature start depends on `M`
  (it samples $n < M$ uniformly), so when comparing different `M` or `N` keep the initial ensemble fixed
  with `--init-nmax` (e.g. proportional to $N$), or compare with `--init vacuum`.
* **Cost** (one core, `vern7`, $t_\mathrm{max}=70$, measured while developing):
  $N=15, M=120, \lambda=0.53$: ≈ 2.3 s per trajectory; $N=30, M=300$: ≈ 12 s; $N=8, M=160, \lambda=0.8$: ≈ 5 s.
  So 2000 trajectories in both sectors at $N=30, M=300$ take ≈ 13 core-hours (< 30 min on 32 cores).
  The cost grows faster than $M(N+1)$ because large photon numbers also force smaller ODE steps, so
  expect roughly $\propto M^2 N$; points with large λ and N (M ≳ 600) can take hours per point.
* **Statistics:** $\langle n\rangle$ s.e. is reported; Wigner fringes of the cat-like structure need a
  few thousand trajectories. Wigner grid: `--wigner-grid 201` points, window chosen automatically from
  $P(n)$ (QuTiP convention $g=\sqrt2$: a coherent state $|\alpha\rangle$ sits at $X=\sqrt2\,\mathrm{Re}\,\alpha$).
* **Wigner colour scale:** inside one parity sector $W(0,0)=\pm1/\pi$ exactly (it measures the parity), a
  few-pixel spike that would wash out the lobes. The colour limit is therefore the 99.8th percentile of
  $|W|$ and the colour bar gets arrows when values are clipped (`clip_percentile=100` in
  `tpdt.plotting.draw_wigner` turns this off). The stored `W` arrays are never clipped.
* **Per-trajectory symmetry:** every trajectory starts in an eigenstate of $\Pi_4=e^{i\pi(a^\dagger a/2+J_z)}$
  and stays in one (H conserves $\Pi_4$, a jump $a^2$ only flips its sign), so $\langle a^2\rangle=0$ and
  $\langle J_x\rangle=0$ on each trajectory: a single trajectory is $\mathbb Z_4$-symmetric and never picks one
  of the four lobes; the symmetry breaking lives in the coherences of the ensemble state. What is stored per
  trajectory (`final_a4`, `final_jxa2`) are the $\mathbb Z_4$-invariant quantities $\langle a^4\rangle$ (≈ 0 in the
  vacuum basin, phase $4\theta$ of the lobe angle on the lobes) and $\langle J_x a^2\rangle$ (phase $2\theta$, $+\pi$ if
  $J_x<0$ on the lobe at angle $\theta$: which spin orientation is locked to which lobe pair). Both are equal
  for all four lobes, so the scatter in `diagnostics.png` (e) is one cluster plus the vacuum.

## Git

Per point and sector the stored data are ~(M/2)² complex numbers plus a 201×201 float32 Wigner
grid: ≈ 1–2 MB per point for $M\le 500$. Commit `results/`, `sweeps/` and the code; `logs/` and `jobs/`
are ignored. For very large sweeps consider git LFS for `*.npz`
(`git lfs track "results/**/*.npz"`).

## Layout

```
run_point.py            one parameter point (trajectories → finalize)
finalize.py             merge seeds, Wigner, P(n), plots for a results tree
make_jobs.py            sweep JSON → jobs/<tag>.txt + sbatch options
plot_sweep.py           Wigner grid over λ, <n>/N vs λ for several N
tpdt/model.py           Hamiltonian and jump operator on one parity sector
tpdt/trajectories.py    sampling, worker pool, checkpoints
tpdt/analysis.py        merging, summaries, loading helpers
tpdt/plotting.py        figures
slurm/                  env.sh (edit), run_array.sbatch, finalize.sbatch, submit.sh, resubmit.sh
sweeps/                 example sweep files
tests/selftest.py       exactness of the sector restriction + trajectories vs mesolve
```

Requires Python ≥ 3.9 and QuTiP ≥ 5.0 (uses `MCSolver.start/step`). Single-photon loss
($\kappa_1>0$) is deliberately not supported: it couples the parity sectors.
