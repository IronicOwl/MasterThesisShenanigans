import numpy as np
from scipy.integrate import solve_ivp
# scaled (4,6) model: H = w a^dag a + lam (a^4 + a^dag^4), L6 = sqrt(k6) a^6, L1 = sqrt(k1) a
# scaled variables: alpha = sqrt(N) at, lam = lt/N, k6 = k6t/N^5  -> N-independent equation
def f(at, w, k1, lt, k6t):
    return -(1j*w + k1/2)*at - 4j*lt*np.conj(at)**3 - 3*k6t*abs(at)**10*at

def fixed_points(w, k1, lt, k6t):
    # |(k1/2 + 3 k6t n^5) + i w| = 4 lt n
    g = lambda n: (k1/2 + 3*k6t*n**5)**2 + w**2 - 16*lt**2*n**2
    ns = np.linspace(1e-4, 5, 200001); gv = g(ns)
    roots = []
    for j in np.where(np.sign(gv[:-1]) != np.sign(gv[1:]))[0]:
        a, b = ns[j], ns[j+1]
        for _ in range(60):
            c = 0.5*(a+b)
            if np.sign(g(c)) == np.sign(g(a)): a = c
            else: b = c
        roots.append(0.5*(a+b))
    out = []
    for n in roots:
        e4 = 1j*((k1/2 + 3*k6t*n**5) + 1j*w)/(4*lt*n)       # = e^{-4 i phi}
        phi = -np.angle(e4)/4
        out.append((n, np.sqrt(n)*np.exp(1j*phi)))
    return out

def jac(a0, *p, h=1e-7):
    F = lambda v: (lambda z: np.array([z.real, z.imag]))(f(v[0]+1j*v[1], *p))
    v0 = np.array([a0.real, a0.imag]); J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2); e[j] = h; J[:, j] = (F(v0+e) - F(v0-e))/(2*h)
    return np.linalg.eigvals(J)

w, k1, k6t = 0.4, 0.0, 0.2
for lt in [0.12, 0.14, 0.18, 0.25]:
    fps = fixed_points(w, k1, lt, k6t)
    print(f"lt={lt}: ", [(round(n,4), np.round(a,4)) for n, a in fps])
    for n, a in fps:
        for k in range(4):
            z = (1j)**k*a
            print(f"     n={n:.4f} k={k} |f|={abs(f(z,w,k1,lt,k6t)):.1e} eig={np.round(jac(z,w,k1,lt,k6t),3)}")
# flow from random initial conditions at lt = 0.18
rng = np.random.default_rng(3); ends = []
for _ in range(12):
    z0 = (rng.normal() + 1j*rng.normal())*0.9
    sol = solve_ivp(lambda t, y: (lambda z: [z.real, z.imag])(f(y[0]+1j*y[1], w, k1, 0.18, k6t)), [0, 3000], [z0.real, z0.imag], rtol=1e-9, atol=1e-12)
    ends.append(np.round(sol.y[0, -1] + 1j*sol.y[1, -1], 3))
print("ODE end points (lt=0.18):", ends)
