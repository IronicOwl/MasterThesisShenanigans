import numpy as np
from scipy.special import eval_genlaguerre, gammaln

def wigner(rho, xvec, pvec):
    """Wigner function, convention a = (x + i p)/sqrt(2) (same as QuTiP default g=sqrt(2)).
    rho[m, n] = <m|rho|n>."""
    D = rho.shape[0]
    X, P = np.meshgrid(xvec, pvec)
    A = (X + 1j*P)/np.sqrt(2)
    B = 4*np.abs(A)**2
    W = np.zeros_like(X, dtype=float)
    for m in range(D):
        if abs(rho[m, m]) > 1e-15:
            W += np.real(rho[m, m]*(-1)**m*eval_genlaguerre(m, 0, B))
        for n in range(m+1, D):
            if abs(rho[m, n]) > 1e-15:
                pref = np.exp(0.5*(gammaln(m+1) - gammaln(n+1)))
                W += 2*np.real(rho[m, n]*(-1)**m*(2*A)**(n-m)*pref*eval_genlaguerre(m, n-m, B))
    return W*np.exp(-B/2)/np.pi

def coherent_dm(beta, D):
    n = np.arange(D)
    c = np.exp(-abs(beta)**2/2 + n*np.log(beta+0j) - 0.5*gammaln(n+1)) if beta != 0 else (n == 0).astype(complex)
    return np.outer(c, c.conj())

if __name__ == "__main__":
    xv = np.linspace(-6, 6, 241)
    beta = 2.0 + 1.0j
    rho = coherent_dm(beta, 48)
    W = wigner(rho, xv, xv)
    X, P = np.meshgrid(xv, xv)
    Wex = np.exp(-((X - np.sqrt(2)*beta.real)**2 + (P - np.sqrt(2)*beta.imag)**2))/np.pi
    dx = xv[1]-xv[0]
    print("trace(rho)", np.trace(rho).real, "int W", W.sum()*dx*dx, "max|W-Wex|", np.abs(W-Wex).max())
