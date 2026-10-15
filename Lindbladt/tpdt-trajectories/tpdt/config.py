"""Run configuration, parameter identity and output-directory naming."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

INIT_CHOICES = ("infinite_temperature", "vacuum")
SAMPLING_CHOICES = ("stratified", "random")


@dataclass
class RunConfig:
    # model
    N: int
    M: int
    lam: float
    kappa2: float = 0.05
    wc: float = 1.0
    wa: float = 1.0
    # time evolution
    t_max: float = 70.0
    n_t: int = 141                      # output grid points in [0, t_max] (for <n>(t), <Jz>(t))
    # ensemble
    n_traj: int = 1000                  # trajectories per sector for THIS run (seed)
    seed: int = 1
    init: str = "infinite_temperature"  # or "vacuum"
    init_nmax: Optional[int] = None     # only sample initial Fock states n < init_nmax (None: all < M)
    sampling: str = "stratified"        # stratified: cycle through a random permutation of the basis
    # ODE integrator (QuTiP options)
    method: str = "vern7"                # qutip >= 5.0; "adams" also works
    atol: float = 1e-8
    rtol: float = 1e-6
    nsteps: int = 1_000_000
    # bookkeeping
    batch_size: Optional[int] = None    # trajectories per worker task (None: automatic)

    def __post_init__(self):
        if self.init not in INIT_CHOICES:
            raise ValueError(f"init must be one of {INIT_CHOICES}")
        if self.sampling not in SAMPLING_CHOICES:
            raise ValueError(f"sampling must be one of {SAMPLING_CHOICES}")
        if self.n_t < 2:
            raise ValueError("n_t must be >= 2")
        if self.init_nmax is not None and not (2 <= self.init_nmax <= self.M):
            raise ValueError("init_nmax must satisfy 2 <= init_nmax <= M")

    # --- identity -----------------------------------------------------------
    def physics_key(self) -> dict:
        """Everything that changes the ensemble-averaged state. Runs (different
        seeds) are only merged if these agree exactly."""
        return {
            "N": int(self.N), "M": int(self.M), "lam": float(self.lam),
            "kappa2": float(self.kappa2), "wc": float(self.wc), "wa": float(self.wa),
            "t_max": float(self.t_max), "n_t": int(self.n_t),
            "init": self.init,
            "init_nmax": None if self.init == "vacuum" else int(self.init_nmax or self.M),
            "method": self.method, "atol": float(self.atol), "rtol": float(self.rtol),
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**fields)

    # --- directories ---------------------------------------------------------
    def system_dirname(self) -> str:
        if self.init == "vacuum":
            init_tag = "vac"
        elif self.init_nmax is None or self.init_nmax == self.M:
            init_tag = "Tinf"
        else:
            init_tag = f"Tinf-n{self.init_nmax}"
        return (f"N{self.N}_k{self.kappa2:g}_wc{self.wc:g}_wa{self.wa:g}"
                f"_t{self.t_max:g}_{init_tag}")

    def point_dirname(self) -> str:
        # M is part of the point (not the system) so that the cutoff can grow with
        # lambda and convergence checks in M sit next to each other
        return f"lam{self.lam:.4f}_M{self.M}"

    def point_dir(self, outdir: str | Path) -> Path:
        return Path(outdir) / self.system_dirname() / self.point_dirname()


def keys_equal(a: dict, b: dict) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ---------------------------------------------------------------------------
# Fock cutoff estimate
# ---------------------------------------------------------------------------

def superradiant_n_over_N(lam: float, kappa2: float, wc: float = 1.0) -> float:
    """Estimate of <n>/N on the superradiant branch with two-photon loss only:
    the steady-state relation 4 lam^2 <Jx>^2 = kappa2^2 (<n>/N)^2 + wc^2 of the
    cumulant analysis with |<Jx>| <= 1 gives the upper bound
    sqrt(4 lam^2 - wc^2) / kappa2   (0 below lam = wc/2)."""
    x = 4.0 * lam ** 2 - wc ** 2
    return math.sqrt(x) / kappa2 if x > 0 else 0.0


def auto_cutoff(N: int, lam: float, kappa2: float, wc: float = 1.0,
                factor: float = 1.6, minimum: int = 60, step: int = 20) -> int:
    """Fock cutoff M = factor * N * superradiant_n_over_N, rounded up to a multiple
    of `step`, at least max(minimum, 6 N). Always confirm with the tail weight that
    finalize reports (P(n >= 0.9 M) < 1e-3)."""
    m = max(float(minimum), 6.0 * N, factor * N * superradiant_n_over_N(lam, kappa2, wc))
    return int(step * math.ceil(m / step))
