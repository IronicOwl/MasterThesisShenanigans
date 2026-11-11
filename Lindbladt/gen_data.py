import numpy as np, time
from exact import liouvillian_sectors, sector_spectrum

w, k1, k2t, Kt = 1.0, 1.0, 1.0, 0.5
lamc = 0.5*np.sqrt(w**2 + k1**2/4)
ratios = np.linspace(0.6, 2.0, 22)
sizes = [(2, 24), (4, 34), (8, 48)]
out = {"ratios": ratios, "lamc": lamc}
for Nsc, D in sizes:
    g0, g1 = [], []
    t0 = time.time()
    for r in ratios:
        Lv, k, nn, mm = liouvillian_sectors(D, w, k1, k2t/Nsc, Kt/Nsc, r*lamc)
        v0, _, _ = sector_spectrum(Lv, k, 0, nev=3)
        v1, _, _ = sector_spectrum(Lv, k, 1, nev=3)
        g0.append(-v0[1].real)      # first non-zero eigenvalue in E^0
        g1.append(-v1[0].real)      # slowest eigenvalue in E^1
    out[f"g0_N{Nsc}"] = np.array(g0); out[f"g1_N{Nsc}"] = np.array(g1)
    print(Nsc, "done", f"{time.time()-t0:.0f}s", flush=True)

# eigenmatrices at N=8, lambda = 2 lambda_c
Nsc, D, r = 8, 48, 2.0
Lv, k, nn, mm = liouvillian_sectors(D, w, k1, k2t/Nsc, Kt/Nsc, r*lamc)
v0, vec0, idx0 = sector_spectrum(Lv, k, 0, nev=1)
v1, vec1, idx1 = sector_spectrum(Lv, k, 1, nev=1)
r0 = np.zeros(D*D, complex); r0[idx0] = vec0[:, 0]; r0 = r0.reshape(D, D, order='F')
r1 = np.zeros(D*D, complex); r1[idx1] = vec1[:, 0]; r1 = r1.reshape(D, D, order='F')
r0 = r0/np.trace(r0)
# make rho_1 Hermitian (fix global phase) and normalise to unit trace norm
ph = np.angle(r1[np.unravel_index(np.argmax(abs(r1)), r1.shape)])
r1 = r1*np.exp(-1j*ph)
r1 = 0.5*(r1 + r1.conj().T)
r1 = r1/np.abs(np.linalg.eigvalsh(r1)).sum()
out.update(rho0=r0, rho1=r1, lam0=v0[0], lam1=v1[0], D=D, Nsc=Nsc, r=r)
np.savez("data.npz", **out)
print("saved; eigenvalues", v0[0], v1[0], "hermiticity err", np.abs(r1 - r1.conj().T).max())
