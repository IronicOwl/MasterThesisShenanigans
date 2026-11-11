import numpy as np, time
from scipy.linalg import eig, eigvals
from exact46 import liouvillian46, sector_idx, SECTORS
from wigner import coherent_dm

w, k1, k6t, lt = 0.4, 0.0, 0.2, 1.0
def alpha_t():
    g = lambda n: (3*k6t*n**5)**2 + w**2 - 16*lt**2*n**2
    ns = np.linspace(0.05, 4, 400001); gv = g(ns); j = np.where(np.sign(gv[:-1]) != np.sign(gv[1:]))[0][-1]
    n = ns[j]; e4 = 1j*((3*k6t*n**5) + 1j*w)/(4*lt*n)
    return np.sqrt(n)*np.exp(-0.25j*np.angle(e4))
at = alpha_t()
out = {"alpha_t": at}
# spectra (modulus of the two smallest eigenvalues per sector) for several N
for Nsc in [4, 6, 8, 10, 12]:
    D = int(abs(at)**2*Nsc + 6*np.sqrt(abs(at)**2*Nsc) + 12)
    Lv, nn, mm = liouvillian46(D, w, k1, lt/Nsc, k6t/Nsc**5)
    rows = []
    for (i, l, r) in SECTORS:
        v = eigvals(Lv[np.ix_(*(2*[sector_idx(nn, mm, i, l, r)]))]); v = v[np.argsort(np.abs(v))]
        rows.append(v[:2])
    out[f"spec_N{Nsc}"] = np.array(rows); print(Nsc, D, np.round(np.abs(np.array(rows)), 5).tolist(), flush=True)

# long-time projection onto the slowest mode of every sector, N = 10
Nsc = 10; D = int(abs(at)**2*Nsc + 6*np.sqrt(abs(at)**2*Nsc) + 12)
Lv, nn, mm = liouvillian46(D, w, k1, lt/Nsc, k6t/Nsc**5)
modes = []
for (i, l, r) in SECTORS:
    idx = sector_idx(nn, mm, i, l, r)
    vals, vl, vr = eig(Lv[np.ix_(idx, idx)], left=True, right=True)
    j = np.argmin(np.abs(vals)); modes.append((idx, vals[j], vl[:, j], vr[:, j]))
alpha = np.sqrt(Nsc)*at
def ket(b):
    from scipy.special import gammaln
    n = np.arange(D); return np.exp(-abs(b)**2/2 + n*np.log(b+0j) - 0.5*gammaln(n+1))
k0, k1_, k2, k3 = [ket((1j)**g*alpha) for g in range(4)]
catA = k0 + k2; catA /= np.linalg.norm(catA)
four = k0 + k1_ + k2 + k3; four /= np.linalg.norm(four)
inits = {"coh": np.outer(k0, k0.conj()), "catA": np.outer(catA, catA.conj()), "four": np.outer(four, four.conj())}
for name, rho0 in inits.items():
    x = rho0.reshape(-1, order='F'); P = np.zeros(D*D, complex)
    for idx, lam, l, r in modes:
        P[idx] += (l.conj()@x[idx])/(l.conj()@r)*r
    P = P.reshape(D, D, order='F'); P = 0.5*(P + P.conj().T)
    out[f"init_{name}"] = rho0; out[f"long_{name}"] = P
    ev = np.linalg.eigvalsh(P)
    print(f"{name}: Tr={np.trace(P).real:.4f} min eig={ev.min():.2e} purity={np.trace(P@P).real:.3f} "
          f"overlap with initial={np.real(np.trace(rho0@P)):.3f}", flush=True)
# compare 'four' long-time state with Labay-Mora's mu_0 built from cats on both axes
catB = k1_ + k3; catB /= np.linalg.norm(catB)
mu0 = 0.5*(np.outer(catA, catA.conj()) + np.outer(catB, catB.conj()))
P = out["long_four"]; print("four -> mu0: <pi0|P|pi0> =", np.real(four.conj()@P@four),
      " trace distance to mu0 =", 0.5*np.abs(np.linalg.eigvalsh(P - mu0)).sum())
out.update(D=D, Nsc=Nsc, lam_slow=np.array([m[1] for m in modes]))
np.savez("data_nested.npz", **out); print("saved")
