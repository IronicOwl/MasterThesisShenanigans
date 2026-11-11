import numpy as np
from scipy.integrate import solve_ivp
rng = np.random.default_rng(1)

def f(a, w, k1, k2, K, lam):
    return -(1j*w + k1/2)*a - 2j*lam*np.conj(a) - (1j*K + k2)*abs(a)**2*a

def fixed_points(w, k1, k2, K, lam):
    # q(n) = (k1/2 + k2 n)^2 + (w + K n)^2 - 4 lam^2
    A2 = K**2 + k2**2; B = 2*(w*K + k1*k2/2); C = w**2 + k1**2/4 - 4*lam**2
    roots = np.roots([A2, B, C])
    out = []
    for n in roots:
        if abs(n.imag) < 1e-12 and n.real > 1e-12:
            n = n.real
            e2 = (1j*(k1/2 + k2*n) - (w + K*n))/(2*lam)   # = e^{-2 i phi}
            phi = -np.angle(e2)/2
            qp = 2*A2*n + B
            out.append((n, np.sqrt(n)*np.exp(1j*phi), qp))
    return out

def jac_eigs(a0, w, k1, k2, K, lam, h=1e-7):
    # real Jacobian of the flow in (Re a, Im a)
    def F(v):
        z = f(v[0]+1j*v[1], w, k1, k2, K, lam); return np.array([z.real, z.imag])
    v0 = np.array([a0.real, a0.imag]); J = np.zeros((2,2))
    for j in range(2):
        e = np.zeros(2); e[j] = h
        J[:, j] = (F(v0+e) - F(v0-e))/(2*h)
    return np.linalg.eigvals(J)

for (w,k1,k2,K) in [(1,1,1,0),(1,1,1,1),(1,1,0.3,0.5),(-2,0.5,0.1,1)]:
    lamc = 0.5*np.sqrt(w**2 + k1**2/4)
    print(f"\nparams w={w},k1={k1},k2={k2},K={K}; lam_c={lamc:.4f}; wK+k1k2/2={w*K+k1*k2/2:.3f}")
    for lam in [0.8*lamc, 0.95*lamc, 1.2*lamc, 2*lamc]:
        fps = fixed_points(w,k1,k2,K,lam)
        ev0 = jac_eigs(0j, w,k1,k2,K,lam)
        pred0 = [-k1/2 + np.sqrt(complex(4*lam**2 - w**2)), -k1/2 - np.sqrt(complex(4*lam**2 - w**2))]
        s = f"  lam/lamc={lam/lamc:.2f}: vacuum eig={np.round(ev0,4)} pred={np.round(pred0,4)}"
        print(s)
        for n,a0,qp in fps:
            for sgn in (+1,-1):
                res = abs(f(sgn*a0,w,k1,k2,K,lam))
                ev = jac_eigs(sgn*a0,w,k1,k2,K,lam)
                print(f"    n={n:.4f} sign={sgn:+d} |f|={res:.1e} q'={qp:+.3f} stable={np.all(ev.real<0)} eig={np.round(ev,3)}")
        # integrate from random ICs
        finals = []
        for _ in range(6):
            a_init = (rng.normal()+1j*rng.normal())*2
            sol = solve_ivp(lambda t,y: [f(y[0]+1j*y[1],w,k1,k2,K,lam).real, f(y[0]+1j*y[1],w,k1,k2,K,lam).imag],
                            [0,400],[a_init.real,a_init.imag],rtol=1e-9,atol=1e-11)
            finals.append(np.round(sol.y[0,-1]+1j*sol.y[1,-1],4))
        print("    ODE end points:", sorted(set(finals), key=lambda z:(z.real,z.imag)))
