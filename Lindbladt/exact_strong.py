import numpy as np
from scipy.linalg import eig
from exact import liouvillian_sectors

def strong_sector(Lv, nn, mm, l, r, nev=3):
    idx = np.where(((nn % 2) == l) & ((mm % 2) == r))[0]
    vals, vecs = eig(Lv[np.ix_(idx, idx)])
    o = np.argsort(-vals.real)
    return vals[o][:nev], vecs[:, o][:, :nev], idx

if __name__ == "__main__":
    import time
    w, k1, k2t, Kt = 1.0, 0.0, 1.0, 0.5
    lamc = abs(w)/2
    for Nsc, D in [(2, 24), (4, 34), (8, 48)]:
        t0 = time.time()
        for rr in [0.8, 1.2, 1.6, 2.0]:
            Lv, k, nn, mm = liouvillian_sectors(D, w, k1, k2t/Nsc, Kt/Nsc, rr*lamc)
            s = []
            for (l, r) in [(0,0),(1,1),(0,1),(1,0)]:
                v, _, _ = strong_sector(Lv, nn, mm, l, r)
                s.append(np.round(v[:2], 4))
            print(f"N={Nsc} lam/lamc={rr}: E00 {s[0]} | E11 {s[1]} | E01 {s[2]} | E10 {s[3]}")
        print(f"  {time.time()-t0:.1f}s")
    # exactly solvable point omega = 0
    for Nsc, D in [(2, 24), (4, 34)]:
        Lv, k, nn, mm = liouvillian_sectors(D, 0.0, 0.0, k2t/Nsc, Kt/Nsc, 0.6)
        v, _, _ = strong_sector(Lv, nn, mm, 0, 1)
        print(f"omega=0, N={Nsc}: E01 slowest {np.round(v[:2],8)}")
