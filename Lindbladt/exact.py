import numpy as np, time, sys
from scipy.linalg import eig

def build_ops(D):
    a = np.diag(np.sqrt(np.arange(1, D)), 1).astype(complex)
    return a

def liouvillian_sectors(D, w, k1, k2, K, lam):
    a = build_ops(D); ad = a.conj().T; I = np.eye(D)
    H = w*ad@a + 0.5*K*ad@ad@a@a + lam*(a@a + ad@ad)
    Ls = [np.sqrt(k1)*a, np.sqrt(k2)*a@a]
    # column stacking: vec(A X B) = (B^T kron A) vec(X)
    Lv = -1j*(np.kron(I, H) - np.kron(H.T, I))
    for L in Ls:
        LdL = L.conj().T@L
        Lv += np.kron(L.conj(), L) - 0.5*np.kron(I, LdL) - 0.5*np.kron(LdL.T, I)
    n = np.arange(D)
    # index of rho_{nm} is n + D*m
    nn = np.tile(n, D); mm = np.repeat(n, D)
    k = (nn - mm) % 2
    return Lv, k, nn, mm

def sector_spectrum(Lv, k, sector, nev=4):
    idx = np.where(k == sector)[0]
    Ls = Lv[np.ix_(idx, idx)]
    vals, vecs = eig(Ls)
    order = np.argsort(-vals.real)
    return vals[order][:nev], vecs[:, order][:, :nev], idx

if __name__ == "__main__":
    w, k1, k2t, Kt = 1.0, 1.0, 1.0, float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    lamc = 0.5*np.sqrt(w**2 + k1**2/4)
    for Nsc, D in [(2, 24), (4, 34), (8, 48)]:
        t0 = time.time()
        for r in [0.8, 1.2, 1.6, 2.0]:
            lam = r*lamc
            Lv, k, nn, mm = liouvillian_sectors(D, w, k1, k2t/Nsc, Kt/Nsc, lam)
            v0, vec0, idx0 = sector_spectrum(Lv, k, 0)
            v1, vec1, idx1 = sector_spectrum(Lv, k, 1)
            # steady state photon number and cutoff check
            rho = np.zeros(D*D, complex); rho[idx0] = vec0[:, 0]
            rho = rho.reshape(D, D, order='F'); rho /= np.trace(rho)
            pn = np.real(np.diag(rho))
            print(f"N={Nsc:2d} D={D} lam/lamc={r:.1f}: E0 {np.round(v0[:3],4)} | E1 {np.round(v1[:3],4)} | <n>={np.sum(np.arange(D)*pn):.2f} p(n>D-5)={pn[-5:].sum():.1e}")
        print(f"   ({time.time()-t0:.1f}s)")
