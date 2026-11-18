"""Second-order cumulant expansion of the two-photon Dicke model (kappa_1 = 0), following Shah et al.
with corrected prefactors: analytic steady state, lobe width and numerical linear stability.

Model:  H = wc a^dag a + (wa/2) Jz + (lam/N) Jx (a^dag^2 + a^2),   L = sqrt(k2/N) a^2,   J = sum sigma.
Notation as in the thesis: expectation values of the rescaled operators a/sqrt(N), J/N, i.e.
alpha = <a> = x + i y, a2 = <a^2>, n = <a^dag a>, j = <J> (spin-photon factorised, photon moments closed
with the Gaussian rules eq:3cumulant, eq:4cumulant). Equations of motion (eq:cum_eom_a ... eq:cum_eom_jz):
    d<a>/dt     = -i wc <a> - 2i lam jx <a>* - k2 (2 n <a> + <a^2> <a>* - 2|<a>|^2 <a>)
    d<a^2>/dt   = -2i wc <a^2> - 4i lam jx n - 2 k2 (3 n <a^2> - 2|<a>|^2 <a>^2) - (2i lam jx + k2 <a^2>)/N
    d<a^dag a>/dt = -4 lam jx Im<a^2> - 2 k2 (2 n^2 + |<a^2>|^2 - 2|<a>|^4)
    d jx/dt = -wa jy,   d jy/dt = wa jx - 4 lam jz Re<a^2>,   d jz/dt = 4 lam jy Re<a^2>

Analytic steady state on the superradiant branches, fluctuations n_f = n - |alpha|^2 and xi = <a^2> - alpha^2
(eq:cum_split), alpha = |alpha| e^{i theta0}:
    xi = (n_f + 1/(2N)) eta e^{2 i theta0},  eta = (i wc + 2k2 n_f)/(i wc + 2k2 n)           (eq:cum_xi)
    8 N k2^2 n_f (n^2 - n_f^2 - n_f/(2N)) = wc^2                                               (eq:cum_nf)
    lam^2 = [(wc^2 + 2k2^2 n_f(2n+n_f))^2 + wc^2 k2^2 n^2] (wc n + sqrt(wc^2 n^2 + wa^2))
            / (8 wc n (wc^2 + 4 k2^2 n_f^2))                                                   (eq:cum_lambda)
    jz = -wa jx^2/(2 wc n)                                                                     (eq:cum_ss_jz)
The script checks that this solves the equations of motion, then builds the Jacobian at the vacuum and
at both superradiant branches. It plots the real parts of the eigenvalues for the vacuum and the upper
branch; the lower branch is only checked to be unstable (one eigenvalue with positive real part).

Outputs:  fig_cum_stability.pdf/.png  ((a) vacuum, (b) upper branch),
          cumulant_curves.csv  (analytic n and lobe width sigma_n/N vs lambda, e.g. for Fig. mf_vs_qt).
Run:      python cumulant_tpdm.py            (a few seconds)
          python cumulant_tpdm.py --check    (also checks the equations of motion against the exact
                                              Lindbladian, see check_eom())
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar

# ----------------------------------------------------------------------------- settings
WA, WC, K2 = 1.0, 1.0, 0.05      # omega_a, omega_c, kappa_2
N = 15                           # number of spins (np.inf for the thermodynamic limit)
LAM_VAC = (0.3, 1.0)             # lambda range of panel (a), vacuum
LAM_SR = (0.505, 0.65)           # lambda range of panel (b), upper branch
NPTS = 400

INK, SOFT, GRID = "#0b0b0b", "#6b6a65", "#e4e3dd"
C1, C2 = "#2a78d6", "#eb6834"    # blue, orange


# ----------------------------------------------------------------------------- equations of motion
def rhs(v, lam, N=N):
    """Cumulant equations of motion in v = (x, y, Re<a^2>, Im<a^2>, n, jx, jy), alpha = <a> = x + i y;
    jz = -sqrt(1 - jx^2 - jy^2) eliminated as in the mean field Jacobian."""
    x, y, a2r, a2i, n, jx, jy = v
    jz = -np.sqrt(max(1.0 - jx**2 - jy**2, 0.0))
    alpha, a2 = x + 1j * y, a2r + 1j * a2i
    inv = 0.0 if np.isinf(N) else 1.0 / N
    d_alpha = (-1j * WC * alpha - 2j * lam * jx * np.conj(alpha)
               - K2 * (2 * n * alpha + a2 * np.conj(alpha) - 2 * abs(alpha)**2 * alpha))
    d_a2 = (-2j * WC * a2 - 4j * lam * jx * n - 2 * K2 * (3 * n * a2 - 2 * abs(alpha)**2 * alpha**2)
            - inv * (2j * lam * jx + K2 * a2))
    d_n = -4 * lam * jx * a2.imag - 2 * K2 * (2 * n**2 + abs(a2)**2 - 2 * abs(alpha)**4)
    return np.array([d_alpha.real, d_alpha.imag, d_a2.real, d_a2.imag, d_n,
                     -WA * jy, WA * jx - 4 * lam * jz * a2.real])


def jacobian(v, lam, N=N, h=1e-7):
    """Central-difference Jacobian of rhs at v."""
    v = np.asarray(v, float)
    return np.array([(rhs(v + h * e, lam, N) - rhs(v - h * e, lam, N)) / (2 * h) for e in np.eye(7)]).T


# ----------------------------------------------------------------------------- analytic steady state
def nf_of_n(n, N=N):
    """Fluctuation photon number n_f (intensive) from eq:cum_nf; smallest positive root below n, nan if none."""
    if np.isinf(N):
        return 0.0
    r = np.roots([8 * N * K2**2, 4 * K2**2, -8 * N * K2**2 * n**2, WC**2])
    r = np.sort(r[(abs(r.imag) < 1e-10) & (r.real > 0)].real)
    r = r[r < n]
    return r[0] if r.size else np.nan


def lam_of_n(n, N=N):
    """Coupling at which the branch has photon number n, eq:cum_lambda."""
    nf = nf_of_n(n, N)
    P = WC**2 + 2 * K2**2 * nf * (2 * n + nf)
    W = P**2 + WC**2 * K2**2 * n**2
    K = WC**2 + 4 * K2**2 * nf**2
    return np.sqrt(W * (WC * n + np.sqrt(WC**2 * n**2 + WA**2)) / (8 * WC * n * K))


def steady_state(n, N=N):
    """Full steady state (lam, n_f, v, eta) on the branch with photon number n (lobe with jx < 0)."""
    nf, lam = nf_of_n(n, N), lam_of_n(n, N)
    inv = 0.0 if np.isinf(N) else 1.0 / N
    P = WC**2 + 2 * K2**2 * nf * (2 * n + nf)
    K = WC**2 + 4 * K2**2 * nf**2
    jx = -np.sqrt((P**2 + WC**2 * K2**2 * n**2) / (4 * lam**2 * K))                  # eq:cum_ss_jx
    eta = (1j * WC + 2 * K2 * nf) / (1j * WC + 2 * K2 * n)                            # eq:cum_xi
    u = -2j * lam * jx / (1j * WC + K2 * (n + nf) + K2 * (nf + inv / 2) * eta)       # u = e^{2 i theta0}
    alpha = np.sqrt(n - nf) * np.exp(0.5j * np.angle(u))
    xi = (nf + inv / 2) * eta * u                                                     # eq:cum_xi
    a2 = alpha**2 + xi
    return lam, nf, np.array([alpha.real, alpha.imag, a2.real, a2.imag, n, jx, 0.0]), eta


def n_min(N=N):
    """Photon number at the fold, where lam(n) is minimal: lambda_c(N) = lam_of_n(n_min)."""
    lo = 2.06 if not np.isinf(N) else 0.5
    return minimize_scalar(lambda n: lam_of_n(n, N), bounds=(lo, 10), method="bounded",
                           options={"xatol": 1e-10}).x


def branch(lam, which, N=N):
    """Photon number of the upper or lower branch at coupling lam (nan if it does not exist)."""
    nc = n_min(N)
    if lam <= lam_of_n(nc, N):
        return np.nan
    f = lambda n: lam_of_n(n, N) - lam
    if which == "upper":
        return brentq(f, nc, 500)
    lo = 1e-4 if np.isinf(N) else 2.0
    while np.isnan(lam_of_n(lo, N)):            # finite N: eq:cum_nf needs a root n_f < n
        lo += 1e-4
    return brentq(f, lo, nc) if f(lo) > 0 else np.nan


def lobe_width(n, N=N):
    """sigma_n / N of the Gaussian lobe: eq:cum_var (exact for the Gaussian) and eq:cum_width (leading order)."""
    _, nf, v, eta = steady_state(n, N)
    alpha, a2 = v[0] + 1j * v[1], v[2] + 1j * v[3]
    xi = a2 - alpha**2
    var = (abs(alpha)**2 * (2 * nf + 1 / N) + 2 * (np.conj(alpha)**2 * xi).real
           + nf * (nf + 1 / N) + abs(xi)**2)                                          # eq:cum_var / N^2
    lead = n / N * (1 + WC**2 / (2 * K2**2 * n**2))                                   # eq:cum_width / N^2
    return np.sqrt(var), np.sqrt(lead)


def n_mf(lam):
    """Mean-field upper branch, eq. (4.22)."""
    s = 4 * lam**2
    disc = (s - WC**2)**2 - (WA * K2)**2
    if s <= WC**2 or disc < 0:
        return np.nan
    return np.sqrt(WC**2 * (2 * s * (s - WC**2) - WA**2 * K2**2 + 2 * s * np.sqrt(disc))
                   / (K2**2 * (4 * s * WC**2 + WA**2 * K2**2)))


# ----------------------------------------------------------------------------- check of the EoM
def check_eom(Nspin=3, M=30, lam=0.61, wa=0.7, wc=1.3, k2=0.37, seed=1):
    """d<O>/dt = Tr[O L(rho)] for a random state vs the analytic right-hand sides (before closure),
    ours and the prefactors printed in Shah et al. (their Eqs. 29, 31-33)."""
    rng = np.random.default_rng(seed)
    a1 = np.diag(np.sqrt(np.arange(1, M)), 1)
    s = Nspin / 2
    mm = np.arange(s, -s - 1, -1)
    Sp = np.zeros((Nspin + 1, Nspin + 1))
    for i, mi in enumerate(mm[1:], start=1):
        Sp[i - 1, i] = np.sqrt(s * (s + 1) - mi * (mi + 1))
    Ip, Is = np.eye(M), np.eye(Nspin + 1)
    a = np.kron(a1, Is)
    ad = a.conj().T
    Jx, Jy, Jz = (np.kron(Ip, J) for J in (Sp + Sp.T, -1j * (Sp - Sp.T), np.diag(2 * mm)))
    H = wc * ad @ a + wa / 2 * Jz + lam / Nspin * Jx @ (ad @ ad + a @ a)
    L = np.sqrt(k2 / Nspin) * a @ a
    d = M * (Nspin + 1)
    keep = np.kron((np.arange(M) <= 8).astype(float), np.ones(Nspin + 1)).astype(bool)
    X = rng.normal(size=(d, 4)) + 1j * rng.normal(size=(d, 4))
    X[~keep] = 0
    rho = X @ X.conj().T
    rho /= np.trace(rho)
    Lrho = -1j * (H @ rho - rho @ H) + L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
    ev = lambda O: np.trace(O @ rho)
    n = ad @ a
    a2, ad2 = a @ a, ad @ ad
    ours = {"a": -1j * wc * ev(a) - 2j * lam / Nspin * ev(Jx @ ad) - k2 / Nspin * ev(ad @ a2),
            "a^2": (-2j * wc * ev(a2) - 4j * lam / Nspin * ev(Jx @ n) - 2j * lam / Nspin * ev(Jx)
                    - k2 / Nspin * (ev(a2) + 2 * ev(ad @ a2 @ a))),
            "a^dag a": 2j * lam / Nspin * ev(Jx @ (a2 - ad2)) - 2 * k2 / Nspin * ev(ad2 @ a2),
            "Jx": -wa * ev(Jy), "Jy": wa * ev(Jx) - 2 * lam / Nspin * ev(Jz @ (a2 + ad2)),
            "Jz": 2 * lam / Nspin * ev(Jy @ (a2 + ad2))}
    shah = {"a^2": -2j * wc * ev(a2) - 4j * lam / Nspin * ev(Jx @ n) - 2 * k2 / Nspin * (ev(a2) + 2 * ev(ad @ a2 @ a)),
            "a^dag a": 2j * lam / Nspin * ev(Jx @ (a2 - ad2)) - k2 / Nspin * ev(ad2 @ a2),
            "Jx": -2 * wa * ev(Jy), "Jy": 2 * wa * ev(Jx) - 2 * lam / Nspin * ev(Jz @ (a2 + ad2))}
    ops = {"a": a, "a^2": a2, "a^dag a": n, "Jx": Jx, "Jy": Jy, "Jz": Jz}
    print("check of the equations of motion against Tr[O L(rho)]:")
    for k, O in ops.items():
        ex = np.trace(O @ Lrho)
        line = f"  d<{k}>/dt: ours off by {abs(ours[k] - ex):.1e}"
        if k in shah:
            line += f",  Shah et al. off by {abs(shah[k] - ex):.1e}"
        print(line)


# ----------------------------------------------------------------------------- run
if __name__ == "__main__":
    if "--check" in sys.argv:
        check_eom()

    nc = n_min(N)
    lam_c = lam_of_n(nc, N)
    lam_c_mf = 0.5 * np.sqrt(WC**2 + WA * K2)
    n_c_mf = np.sqrt(WA * WC**2 / (K2 * (2 * WC**2 + WA * K2)))
    lam_c_1st = np.sqrt(lam_c_mf**2 + WC * (WC * n_c_mf + np.sqrt(WC**2 * n_c_mf**2 + WA**2)) / (8 * N * n_c_mf**2))
    print(f"lambda_c: mean field {lam_c_mf:.4f};  cumulant N={N}: {lam_c:.4f} (n = {nc:.3f}), "
          f"first order in 1/N: {lam_c_1st:.4f}")

    # the analytic solution solves the equations of motion
    res = max(np.abs(rhs(steady_state(n)[2], lam_of_n(n))).max() for n in np.linspace(nc, 20, 25))
    print(f"max residual of the analytic steady state in the EoM: {res:.1e}")

    lam = 0.535
    n = branch(lam, "upper")
    _, nf, v, eta = steady_state(n)
    sq = (N * nf + 0.5) * (1 - abs(eta))
    print(f"lambda = {lam}: n = {n:.4f} (mean field {n_mf(lam):.4f}), N n_f = {N * nf:.3f} fluctuation photons, "
          f"sigma_n/N = {lobe_width(n)[0]:.3f} (coherent {np.sqrt(n / N):.3f}), "
          f"smallest quadrature variance {sq:.3f} (vacuum 0.5)")

    # ---------------------------------------------------- analytic curves for Fig. mf_vs_qt
    lams = np.linspace(lam_c + 1e-6, LAM_SR[1], 300)
    rows = []
    for l in lams:
        nu = branch(l, "upper")
        rows.append((l, nu, lobe_width(nu)[0], n_mf(l)))
    np.savetxt("cumulant_curves.csv", np.array(rows), delimiter=",", fmt="%.6f",
               header=f"lambda, n_cumulant(N={N}), sigma_n/N cumulant(N={N}), n_meanfield")
    print("saved cumulant_curves.csv")

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "legend.frameon": False})

    def style(ax, xlabel=r"$\lambda$"):
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlabel)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # ---------------------------------------------------- vacuum spectrum
    lv = np.linspace(*LAM_VAC, NPTS)
    vac = np.zeros(7)
    spec = np.array([np.sort_complex(np.linalg.eigvals(jacobian(vac, l))) for l in lv])
    assert np.all(spec.real < 1e-9), "vacuum linearly unstable in the plotted range"
    sub = np.ix_([2, 3, 5, 6], [2, 3, 5, 6])                         # <a^2> and spin block
    blk = np.array([np.linalg.eigvals(jacobian(vac, l)[sub]) for l in lv])
    sq_re = np.array([b[np.argmax(abs(b.imag))].real for b in blk])  # pair at |Im| ~ 2 wc: <a^2>
    sp_re = np.array([b[np.argmin(abs(b.imag))].real for b in blk])  # pair at |Im| ~ wa: spin

    # ---------------------------------------------------- superradiant branches
    def max_re(l, which, NN):
        nb = branch(l, which, NN)
        if np.isnan(nb):
            return np.nan
        _, _, vs, _ = steady_state(nb, NN)
        return np.linalg.eigvals(jacobian(vs, l, NN)).real.max()

    ls = np.linspace(*LAM_SR, NPTS)
    up = {NN: np.array([max_re(l, "upper", NN) for l in ls]) for NN in (N, np.inf)}
    lo = {NN: np.array([max_re(l, "lower", NN) for l in ls]) for NN in (N, np.inf)}   # checked, not plotted
    assert np.nanmax(up[N]) < 0 and np.nanmax(up[np.inf]) < 0, "upper branch unstable somewhere"
    assert np.nanmin(lo[N]) > 0 and np.nanmin(lo[np.inf]) > 0, "lower branch stable somewhere"

    # ---------------------------------------------------- figure: (a) vacuum, (b) upper branch
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.3, 2.7), gridspec_kw={"wspace": 0.3})
    fig.subplots_adjust(left=0.08, right=0.965, bottom=0.16, top=0.9)
    ax_a.plot(lv, 1e3 * spec.real.max(axis=1), color=INK, lw=1.6)
    ax_a.plot(lv, 1e3 * sp_re, color=C1, lw=1.3)
    ax_a.plot(lv, 1e3 * sq_re, color=C2, lw=1.3)
    ax_a.text(lv[-1], 0.12, r"$\max\,\mathrm{Re}\,\mu=0$: $\langle a\rangle$ ($\pm i\omega_c$), $\langle a^\dagger a\rangle$",
              ha="right", va="bottom", fontsize=7)
    ax_a.text(lv[-1], 1e3 * sp_re[-1] - 0.12, "spin precession (pair)", ha="right", va="top", fontsize=7)
    ax_a.text(lv[-1], 1e3 * sq_re[-1] + 0.12, r"$\langle a^2\rangle$ (pair)", ha="right", va="bottom", fontsize=7)
    ax_a.set_ylim(1e3 * sq_re.min() - 0.5, 0.7)
    ax_a.set_xlim(*LAM_VAC)
    ax_a.set_ylabel(r"$\mathrm{Re}\,\mu$  $(10^{-3})$")
    ax_b.plot(ls, 1e4 * up[np.inf], color=SOFT, lw=1.0, ls=(0, (5, 2.5)), label=r"$N\to\infty$")
    ax_b.plot(ls, 1e4 * up[N], color=INK, lw=1.4, label=rf"$N={N}$")
    ax_b.axhline(0, color=SOFT, lw=0.6)
    ax_b.axvline(lam_c, color=SOFT, lw=0.7, ls=":")
    ax_b.set_ylabel(r"$\max\,\mathrm{Re}\,\mu$  $(10^{-4})$")
    ax_b.set_ylim(np.nanmin(1e4 * up[N]) * 1.15, 0.6)
    ax_b.set_xlim(*LAM_SR)
    ax_b.legend(loc="lower right", fontsize=7.5)
    for ax, lab in ((ax_a, r"$\mathbf{(a)}$  vacuum, $N=%d$" % N), (ax_b, r"$\mathbf{(b)}$  upper branch")):
        ax.text(0.0, 1.03, lab, transform=ax.transAxes, va="bottom", ha="left")
        style(ax)
    fig.savefig("fig_cum_stability.pdf")
    fig.savefig("fig_cum_stability.png", dpi=300)
    i = np.nanargmax(np.where(np.isnan(lo[N]), -np.inf, ls))
    print(f"N={N}: lower branch unstable wherever it exists, up to lambda = {ls[i]:.4f}; "
          f"upper branch max Re mu between {np.nanmin(up[N]):.1e} and {np.nanmax(up[N]):.1e}")
    print("saved fig_cum_stability.pdf/.png")