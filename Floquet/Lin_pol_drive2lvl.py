import numpy as np
from numpy.linalg import eig
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Pauli matrices
sx = np.array([[0, 1],[1, 0]], dtype=complex)
sy = np.array([[0, -1j],[1j, 0]], dtype=complex)
sz = np.array([[1, 0],[0, -1]], dtype=complex)
I  = np.eye(2, dtype=complex)

def H(t, w0, A, w):
    return 0.5*w0*sz + A*np.cos(w*t)*sx

def calc_oneperiod_U(w0,w, A, steps=4000):
    U = I.copy()
    T = (2* np.pi)/w
    Delta_t = T/steps
    t = 0
    for i in range(steps):
        t = (i + 0.5)*Delta_t
        U = expm(-1j* H(t, w0, A, w)*Delta_t) @ U
    return U 

def get_eigenernergies(U, w):
    vals, vecs = eig(U)
    T = (2* np.pi)/w
    phases = np.angle(vals)           # in (-pi, pi]
    eps = -phases / T                 # quasienergies in (-w/2, w/2]
    # sort
    idx = np.argsort(eps)
    return eps[idx], vecs[:, idx]
    
def sweep_frequencies(w0=1,A=0.1, sweep_low=0.7,sweep_high=1.3, sweep_steps=5, steps=4000):
    frequencies = np.linspace(sweep_low, sweep_high, sweep_steps)
    print(np.shape(frequencies))
    eplus = []
    eminus = []
    for i in frequencies:
        U = calc_oneperiod_U(w0, i, A)
        epsilon_vals = get_eigenernergies(U,i)[0]
        eplus.append(epsilon_vals[1])
        eminus.append(epsilon_vals[0])

    return frequencies, eplus, eminus

def eigenphases(U):
    vals, _ = eig(U)
    phases = np.angle(vals)   # in (-pi, pi]
    phases.sort()
    return phases

def sweep_phases(w0=10, A=1, sweep_low=5, sweep_high=15, sweep_steps=400, steps=4000):
    frequencies = np.linspace(sweep_low, sweep_high, sweep_steps)
    th_plus = []
    th_minus = []
    for w in frequencies:
        U = calc_oneperiod_U(w0, w, A, steps=steps)
        th = eigenphases(U)
        th_minus.append(th[0])
        th_plus.append(th[1])
    return frequencies, np.array(th_plus), np.array(th_minus)


def phase_gap(U):
    vals, _ = eig(U)
    phases = np.angle(vals)          # each in (-pi, pi]
    # sort, but sorting alone can still swap branches; gap formula below is robust
    phases.sort()
    th_minus, th_plus = phases[0], phases[1]
    # circular (mod 2pi) phase difference:
    dth = np.angle(np.exp(1j*(th_plus - th_minus)))  # in (-pi, pi]
    return th_plus, th_minus, abs(dth)

def sweep_phase_gap(w0=10, A=1, sweep_low=5, sweep_high=15, sweep_steps=400, steps=4000):
    freqs = np.linspace(sweep_low, sweep_high, sweep_steps)
    gaps = []
    thp_list, thm_list = [], []
    for w in freqs:
        U = calc_oneperiod_U(w0, w, A, steps=steps)
        thp, thm, gap = phase_gap(U)
        thp_list.append(thp)
        thm_list.append(thm)
        gaps.append(gap)
    return freqs, np.array(thp_list), np.array(thm_list), np.array(gaps)


"""
w0 = 10
freqs, thp, thm, gap = sweep_phase_gap(w0=w0, A=1, sweep_low=5, sweep_high=15, sweep_steps=40)

plt.figure(figsize=(6,4))
plt.plot(freqs, gap)
plt.axvline(w0, linestyle='--', color='gray', alpha=0.5)
plt.xlabel(r'$\omega$')
plt.ylabel(r'phase gap $|\Delta\theta|$')
plt.title('Floquet phase gap (circular distance)')
plt.tight_layout()
plt.show()

"""

"""
w0 = 1.0
A  = 0.2
w  = 1.0
steps = 4000
U = calc_oneperiod_U(w0, A, w, steps)
print("unitarity error:", np.linalg.norm(U.conj().T @ U - I))
"""
def eig_phases_vecs(U):
    vals, vecs = eig(U)
    phases = np.angle(vals)
    return phases, vecs

def sweep_tracked_phases(w0=10, A=1, sweep_low=5, sweep_high=15, sweep_steps=400, steps=4000):
    freqs = np.linspace(sweep_low, sweep_high, sweep_steps)

    # first point: just pick an ordering
    U0 = calc_oneperiod_U(w0, freqs[0], A, steps=steps)
    phases0, vecs0 = eig_phases_vecs(U0)

    # pick "branch 0" and "branch 1" by sorting phases initially
    order = np.argsort(phases0)
    phases_prev = phases0[order]
    vecs_prev = vecs0[:, order]

    th0 = [phases_prev[0]]
    th1 = [phases_prev[1]]

    for w in freqs[1:]:
        U = calc_oneperiod_U(w0, w, A, steps=steps)
        phases, vecs = eig_phases_vecs(U)

        # compute overlaps with previous eigenvectors
        # overlap matrix O_ij = |<v_prev_i | v_new_j>|^2
        O = np.abs(vecs_prev.conj().T @ vecs)**2

        # choose assignment that maximizes total overlap
        # for 2x2, just compare the two possibilities
        if O[0,0] + O[1,1] >= O[0,1] + O[1,0]:
            assign = [0,1]
        else:
            assign = [1,0]

        phases_tr = phases[assign]
        vecs_tr   = vecs[:, assign]

        # OPTIONAL: unwrap phases to keep continuity (helpful for plots)
        # unwrap relative to previous by allowing ±2π shifts
        for i in [0,1]:
            candidates = phases_tr[i] + 2*np.pi*np.arange(-2,3)
            phases_tr[i] = candidates[np.argmin(np.abs(candidates - phases_prev[i]))]

        th0.append(phases_tr[0])
        th1.append(phases_tr[1])

        phases_prev = phases_tr
        vecs_prev   = vecs_tr

    th0 = np.array(th0)
    th1 = np.array(th1)

    # circular phase gap (always in [0, pi])
    gap = np.abs(np.angle(np.exp(1j*(th1 - th0))))
    return freqs, th0, th1, gap

w0 = 10
freqs, th0, th1, gap = sweep_tracked_phases(w0=w0, A=1, sweep_low=5, sweep_high=15, sweep_steps=400)

plt.figure(figsize=(6,4))
plt.plot(freqs, gap)
plt.axvline(w0, linestyle='--', color='gray', alpha=0.5)
plt.xlabel(r'$\omega$')
plt.ylabel(r'tracked phase gap $|\Delta\theta|$')
plt.title('Floquet phase gap (tracked branches)')
plt.tight_layout()
plt.show()

