import numpy as np, time
from scipy.linalg import eig

SECTORS = [(0,0,0),(0,1,1),(2,0,0),(2,1,1),(1,0,1),(3,0,1),(1,1,0),(3,1,0)]   # (i, l, r): E^i_{lr}

def liouvillian46(D, w, k1, lam, k6):
    a = np.diag(np.sqrt(np.arange(1, D)), 1).astype(complex); ad = a.conj().T; I = np.eye(D)
    a4 = np.linalg.matrix_power(a, 4); a6 = np.linalg.matrix_power(a, 6)
    H = w*ad@a + lam*(a4 + a4.conj().T)
    Lv = -1j*(np.kron(I, H) - np.kron(H.T, I))
    for L in ([np.sqrt(k1)*a] if k1 > 0 else []) + [np.sqrt(k6)*a6]:
        LdL = L.conj().T@L
        Lv += np.kron(L.conj(), L) - 0.5*np.kron(I, LdL) - 0.5*np.kron(LdL.T, I)
    n = np.arange(D); nn = np.tile(n, D); mm = np.repeat(n, D)      # vec index n + D*m  <-> |n><m|
    return Lv, nn, mm

def sector_idx(nn, mm, i, l, r):
    return np.where((((nn - mm) % 4) == i) & ((nn % 2) == l) & ((mm % 2) == r))[0]

def slowest(Lv, idx, k=3, left=False):
    A = Lv[np.ix_(idx, idx)]
    if left:
        vals, vl, vr = eig(A, left=True, right=True)
    else:
        vals, vr = eig(A); vl = None
    o = np.argsort(-vals.real)
    return vals[o][:k], (vl[:, o][:, :k] if left else None), vr[:, o][:, :k]

if __name__ == "__main__":
    w, k1, lt, k6t = 0.4, 0.0, 0.18, 0.2
    for Nsc in [2, 4, 6, 8, 10, 12]:
        nbar = 1.0*Nsc; D = int(nbar + 6*np.sqrt(nbar) + 12)
        t0 = time.time()
        Lv, nn, mm = liouvillian46(D, w, k1, lt/Nsc, k6t/Nsc**5)
        line = []
        for (i, l, r) in SECTORS:
            v, _, _ = slowest(Lv, sector_idx(nn, mm, i, l, r))
            # in trace-carrying sectors report the first non-zero eigenvalue as well
            line.append(f"E{i}_{l}{r}:{np.round(v[0],5)}" + (f"/{np.round(v[1],4)}" if i == 0 else ""))
        print(f"N={Nsc:2d} D={D}  " + "  ".join(line), f"({time.time()-t0:.1f}s)", flush=True)
