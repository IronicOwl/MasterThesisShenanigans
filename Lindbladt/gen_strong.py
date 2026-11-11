import numpy as np, time
from exact import liouvillian_sectors
from exact_strong import strong_sector

w, k1, k2t, Kt = 1.0, 0.0, 1.0, 0.5
lamc = abs(w)/2
ratios = np.linspace(0.6, 2.0, 22)
out = {"ratios": ratios, "lamc": lamc}
for Nsc, D in [(2, 24), (4, 34), (8, 48)]:
    t0 = time.time(); g00, g11, l01 = [], [], []
    for rr in ratios:
        Lv, k, nn, mm = liouvillian_sectors(D, w, k1, k2t/Nsc, Kt/Nsc, rr*lamc)
        v00, _, _ = strong_sector(Lv, nn, mm, 0, 0)
        v11, _, _ = strong_sector(Lv, nn, mm, 1, 1)
        v01, _, _ = strong_sector(Lv, nn, mm, 0, 1)
        g00.append(v00[1]); g11.append(v11[1]); l01.append(v01[0])
    out[f"l00_N{Nsc}"] = np.array(g00); out[f"l11_N{Nsc}"] = np.array(g11); out[f"l01_N{Nsc}"] = np.array(l01)
    print(Nsc, "done", f"{time.time()-t0:.0f}s", flush=True)

Nsc, D, rr = 8, 48, 2.0
Lv, k, nn, mm = liouvillian_sectors(D, w, k1, k2t/Nsc, Kt/Nsc, rr*lamc)
mats = {}
for (l, r) in [(0, 0), (1, 1), (0, 1)]:
    v, vec, idx = strong_sector(Lv, nn, mm, l, r, nev=1)
    X = np.zeros(D*D, complex); X[idx] = vec[:, 0]; X = X.reshape(D, D, order='F')
    mats[f"{l}{r}"] = (X, v[0])
r00 = mats["00"][0]/np.trace(mats["00"][0]); r11 = mats["11"][0]/np.trace(mats["11"][0])
out.update(rho00=r00, rho11=r11, X01=mats["01"][0], lam01=mats["01"][1], D=D, Nsc=Nsc, r=rr)
np.savez("data_strong.npz", **out)
print("saved; purity rho00, rho11:", np.trace(r00@r00).real, np.trace(r11@r11).real, " lam01 =", mats["01"][1])
