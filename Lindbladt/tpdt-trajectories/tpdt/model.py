"""
Two-photon Dicke model with pure two-photon loss, restricted to one cavity
parity sector.

    H   = wc a^dag a + (wa/2) Jz + (lam/N) Jx (a^dag^2 + a^2)
    L_2 = sqrt(kappa2/N) a^2

with Jx = 2 jmat(N/2,'x'), Jz = 2 jmat(N/2,'z') (eigenvalues -N..N), i.e. the
same conventions as Shah et al. (arXiv:2412.14271) and the original notebook.

Why the sector restriction is exact
-----------------------------------
Both H and L_2 commute with the photon parity Pi_2 = exp(i pi a^dag a) (strong
Z2). A trajectory started in the even (odd) photon-number subspace therefore
never leaves it, and the operators restricted to that subspace are exactly the
truncated full-space operators projected onto it. Simulating only the sector
halves the state dimension (M -> ~M/2) at no loss of accuracy. With single-photon
loss (kappa1 > 0) this would NOT hold, which is why kappa1 is not supported here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qutip as qt

SECTORS = ("even", "odd")
_PARITY = {"even": 0, "odd": 1}


def sector_fock_numbers(M: int, sector: str) -> np.ndarray:
    """Photon numbers n < M with n % 2 == parity(sector)."""
    if sector not in _PARITY:
        raise ValueError(f"sector must be one of {SECTORS}, got {sector!r}")
    return np.arange(_PARITY[sector], M, 2)


@dataclass
class SectorModel:
    """Operators of the model on (parity sector of the cavity) x (spin N/2)."""

    N: int
    M: int
    sector: str
    fock: np.ndarray          # photon numbers spanned by the sector
    H: qt.Qobj
    c_ops: list
    n_op: qt.Qobj             # a^dag a
    a2_op: qt.Qobj            # a^2
    Jx: qt.Qobj               # 2 jmat('x')
    Jz: qt.Qobj               # 2 jmat('z')
    jz_diag: np.ndarray       # eigenvalues of Jz in the spin basis order
    jx_spin: np.ndarray       # dense (N+1)x(N+1) matrix of 2 jmat('x')
    a2_sector: np.ndarray     # superdiagonal of a^2 in the sector basis

    @property
    def dim_cav(self) -> int:
        return int(self.fock.size)

    @property
    def dim_spin(self) -> int:
        return self.N + 1


def _to_csr(op: qt.Qobj) -> qt.Qobj:
    try:
        return op.to("CSR")
    except Exception:  # very old data layers: leave as is
        return op


def build_sector_model(N: int, M: int, lam: float, kappa2: float,
                       wc: float = 1.0, wa: float = 1.0,
                       sector: str = "even") -> SectorModel:
    if N < 1:
        raise ValueError("N must be >= 1")
    if M < 6:
        raise ValueError("M (Fock cutoff) must be >= 6")
    if kappa2 <= 0:
        raise ValueError("kappa2 must be > 0 (two-photon loss stabilises the model)")

    fock = sector_fock_numbers(M, sector)
    Ms = fock.size
    j = N / 2.0

    # cavity operators on the sector: n|f> = f|f>,  a^2|f> = sqrt(f(f-1))|f-2>
    n_c = qt.qdiags(fock.astype(float), 0)
    a2_vals = np.sqrt(fock[1:] * (fock[1:] - 1.0))
    a2_c = qt.qdiags(a2_vals, 1)

    I_c = qt.qeye(Ms)
    I_s = qt.qeye(N + 1)
    jx = 2.0 * qt.jmat(j, "x")
    jz = 2.0 * qt.jmat(j, "z")

    n_op = _to_csr(qt.tensor(n_c, I_s))
    a2_op = _to_csr(qt.tensor(a2_c, I_s))
    Jx = _to_csr(qt.tensor(I_c, jx))
    Jz = _to_csr(qt.tensor(I_c, jz))

    H = wc * n_op + 0.5 * wa * Jz + (lam / N) * Jx * (a2_op.dag() + a2_op)
    c_ops = [np.sqrt(kappa2 / N) * a2_op]

    return SectorModel(
        N=N, M=M, sector=sector, fock=fock,
        H=_to_csr(H), c_ops=[_to_csr(c) for c in c_ops],
        n_op=n_op, a2_op=a2_op, Jx=Jx, Jz=Jz,
        jz_diag=np.real(jz.diag()).astype(float),
        jx_spin=jx.full(),
        a2_sector=a2_vals,
    )


def build_full_model(N: int, M: int, lam: float, kappa2: float,
                     wc: float = 1.0, wa: float = 1.0):
    """Full (both parities) model exactly as in the original notebook.
    Only used for validation in tests/selftest.py."""
    j = N / 2.0
    a = qt.tensor(qt.destroy(M), qt.qeye(N + 1))
    Jx = qt.tensor(qt.qeye(M), 2.0 * qt.jmat(j, "x"))
    Jz = qt.tensor(qt.qeye(M), 2.0 * qt.jmat(j, "z"))
    H = wc * a.dag() * a + 0.5 * wa * Jz + (lam / N) * Jx * (a.dag() ** 2 + a ** 2)
    c_ops = [np.sqrt(kappa2 / N) * a ** 2]
    return H, c_ops, a


def sector_basis_ket(model: SectorModel, k: int, s: int) -> qt.Qobj:
    """|fock[k]> (x) |spin index s>  (spin index 0 = Jz=+N, index N = Jz=-N)."""
    return qt.tensor(qt.basis(model.dim_cav, int(k)), qt.basis(model.dim_spin, int(s)))


def embed_cavity_rho(rho_sector: np.ndarray, fock: np.ndarray, M: int) -> np.ndarray:
    """Embed a cavity density matrix given on the sector into the full M x M space."""
    rho = np.zeros((M, M), dtype=complex)
    rho[np.ix_(fock, fock)] = rho_sector
    return rho
