"""Prototypes for Figs. 3.1 and 3.2, built from the SAME three Lindbladians.

Model: stabilised Eq. (3.31)-(3.32) with N = 4,
    H = Delta a^dag a + lambda (a^4 + a^dag^4),   jumps {a, a^4}, {a^2, a^4}, {a^4}
    -> M = gcd(4, k, 4) = 1, 2, 4: weak Z4, nested Z4 > Z2, strong Z4.
The four-photon drive breaks the accidental U(1) of the losses down to Z4, so every block is an
irreducible symmetry sector; the D[a^4] term makes all three models stable without changing M.
Detuning Delta != 0 lifts the decoherence-free subspace of the pure four-photon cat, so
dim ker L = 1, 2, 4 exactly (no extra zero modes in off-diagonal sectors).

Fig. 3.1: block structure of the Liouvillian (A: full 64x64 matrix, B: coarse-grained 16x16 sector map).
Fig. 3.2: the steady states, one per trace-carrying sector, with symmetry-forbidden elements in white.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})

N = 4
PARS = dict(Delta=0.2, lam=1.0, g4=0.125, gk=0.1)   # block patterns do not depend on these
INK, MUTED = "#0b0b0b", "#898781"
WEAK_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4"]   # one hue per weak label i
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CASES = [  # key, title, jump label, k (extra loss a^k; 0 = none), block key for Fig. 3.1
    ("weak", r"(a) weak $\mathbb{Z}_4^W$", r"$\{a,\,a^4\}$", 1, lambda i, l, q: i),
    ("nested", r"(b) nested $\mathbb{Z}_4^W\!\supset\mathbb{Z}_2^S$", r"$\{a^2,\,a^4\}$", 2, lambda i, l, q: (i, l)),
    ("strong", r"(c) strong $\mathbb{Z}_4^S$", r"$\{a^4\}$", 0, lambda i, l, q: (i, q)),
]
BLOCK_COUNT = {"weak": r"$N=4$", "nested": r"$NM=8$", "strong": r"$N^2=16$"}


def liouvillian(d, k, Delta, lam, g4, gk):
    """Row-major vectorisation: vec(|n><m|) = n*d + m and vec(A X B) = (A kron B^T) vec(X)."""
    a = np.diag(np.sqrt(np.arange(1, d)), 1); Id = np.eye(d)
    a4 = np.linalg.matrix_power(a, 4)
    H = Delta * a.T @ a + lam * (a4 + a4.T)
    jumps = [np.sqrt(g4) * a4] + ([np.sqrt(gk) * np.linalg.matrix_power(a, k)] if k else [])
    S = -1j * (np.kron(H, Id) - np.kron(Id, H.T))
    for L in jumps:
        LdL = L.T @ L
        S = S + np.kron(L, L) - 0.5 * np.kron(LdL, Id) - 0.5 * np.kron(Id, LdL.T)
    return S


def runs(keys):
    out, start = [], 0
    for j in range(1, len(keys) + 1):
        if j == len(keys) or keys[j] != keys[start]:
            out.append((start, j - start, keys[start])); start = j
    return out


# ============================ Fig. 3.1 ============================
d1 = 8
basis = [(n, m) for n in range(d1) for m in range(d1)]
def labels(n, m):
    q, qp = n % N, m % N
    return (q - qp) % N, q % 2, q            # (i, l, q); q' = q - i is then fixed
order = sorted(range(d1 * d1), key=lambda k: (labels(*basis[k]), basis[k]))
lab = [labels(*basis[k]) for k in order]


def draw_blocks(ax, title, jump, blocks, mat, keyfun, count, n_per_cell, tick_labels=None):
    size = mat.shape[0]
    for start, length, i in runs([x[0] for x in blocks]):
        ax.add_patch(Rectangle((start - .5, start - .5), length, length, fc=WEAK_COLORS[i],
                               alpha=0.10, lw=0, zorder=0))
        ax.text(start + length / 2 - .5, 1.015, rf"$E^{{{i}}}$", ha="center", va="bottom",
                fontsize=8, color=INK, transform=ax.get_xaxis_transform())
    rgba = np.ones((size, size, 4)); rgba[..., 3] = 0
    for r, c in zip(*np.nonzero(np.abs(mat) > 1e-12)):
        rgba[r, c] = matplotlib.colors.to_rgba(WEAK_COLORS[blocks[r][0]])
    ax.imshow(rgba, interpolation="nearest", zorder=1)
    for start, length, _ in runs([keyfun(*x) for x in blocks]):
        ax.add_patch(Rectangle((start - .5, start - .5), length, length, fill=False,
                               ec=INK, lw=0.8, zorder=2))
    ax.set_title(f"{title}\n{jump}:  {count} blocks", fontsize=8.5, pad=15)
    ax.set_xlim(-.5, size - .5); ax.set_ylim(size - .5, -.5)
    ax.set_xticks([])
    if tick_labels is None:
        ax.set_yticks([])
    else:
        ax.set_yticks(np.arange(n_per_cell / 2 - .5, size, n_per_cell), tick_labels,
                      fontsize=5.5, color=MUTED)
        ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_linewidth(0.5)


sec_runs = runs([(x[0], x[2]) for x in lab])
sector_names = [rf"${q}{(q - i) % N}$" for _, _, (i, q) in sec_runs]
sec_lab = [lab[start] for start, _, _ in sec_runs]
for version in ("A", "B"):
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.75), constrained_layout=True)
    for j, (ax, (key, title, jump, k, keyfun)) in enumerate(zip(axes, CASES)):
        P = liouvillian(d1, k, **PARS)[np.ix_(order, order)]
        if version == "B":   # coarse-grain: one cell per pair of strong sectors
            P = np.array([[np.linalg.norm(P[s0:s0 + ls, t0:t0 + lt]) for t0, lt, _ in sec_runs]
                          for s0, ls, _ in sec_runs])
        draw_blocks(ax, title, jump, lab if version == "A" else sec_lab, P, keyfun,
                    BLOCK_COUNT[key], 4 if version == "A" else 1,
                    sector_names if j == 0 else None)
    axes[0].set_ylabel(r"strong sector $(q,q')$", fontsize=7, color=MUTED)
    name = "fig31_A_matrix" if version == "A" else "fig31_B_sectors"
    for ext in ("png", "pdf"):
        fig.savefig(f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

# ============================ Fig. 3.2 ============================
d2, show = 40, 16                       # Fock cutoff for the solve, window shown
n_idx = np.arange(d2)
SECTORS = {   # trace-carrying sectors = conserved pieces; each holds exactly one steady state
    "weak":   [(r"$\rho_{\rm ss}$", lambda n, m: (n - m) % 4 == 0)],
    "nested": [(rf"$l={l}$ ({p})", (lambda l: lambda n, m: (n - m) % 4 == 0 and n % 2 == l)(l))
               for l, p in [(0, "even"), (1, "odd")]],
    "strong": [(rf"$q={q}$", (lambda q: lambda n, m: n % 4 == q and m % 4 == q)(q)) for q in range(4)],
}


def sector_steady_state(S, mask):
    idx = [n * d2 + m for n in range(d2) for m in range(d2) if mask(n, m)]
    B = S[np.ix_(idx, idx)].copy()
    rhs = np.zeros(len(idx), complex)
    B[0, :] = [1.0 if (j // d2) == (j % d2) else 0.0 for j in idx]   # trace = 1
    rhs[0] = 1.0
    rho = np.zeros(d2 * d2, complex); rho[idx] = np.linalg.solve(B, rhs)
    return rho.reshape(d2, d2)


cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
norm = PowerNorm(0.5, vmin=0, vmax=1)

# Transposed "staircase": one column per symmetry type, one panel per conserved sector.
SECTOR_NAMES = {"weak": [r"$E^0$"],
                "nested": [r"$E^0_{00}$ (even $n$)", r"$E^0_{11}$ (odd $n$)"],
                "strong": [rf"$E_{{{q}{q}}}$" for q in range(4)]}
HEADERS = {"weak":   "weak $\\mathbb{Z}_4^W$\n" + r"$\{a,\,a^4\}$: 1 steady state",
           "nested": "nested $\\mathbb{Z}_4^W\\!\\supset\\mathbb{Z}_2^S$\n" + r"$\{a^2,\,a^4\}$: 2 steady states",
           "strong": "strong $\\mathbb{Z}_4^S$\n" + r"$\{a^4\}$: 4 steady states"}
mosaic = [["weak0", "nested0", "strong0"],
          [".",     "nested1", "strong1"],
          ["key",   "key",     "strong2"],
          ["key",   "key",     "strong3"]]
fig, ax = plt.subplot_mosaic(mosaic, figsize=(6.3, 7.4), constrained_layout=True)
last = {"weak": 0, "nested": 1, "strong": 3}
for key, title, jump, k, _ in CASES:
    S = liouvillian(d2, k, **PARS)
    for j, (label, mask) in enumerate(SECTORS[key]):
        rho = sector_steady_state(S, mask)
        assert abs(np.trace(rho) - 1) < 1e-9 and np.linalg.eigvalsh((rho + rho.conj().T) / 2).min() > -1e-9
        a_ = ax[f"{key}{j}"]
        allowed = np.array([[mask(n, m) for m in range(show)] for n in range(show)])
        vals = np.abs(rho[:show, :show]); vals /= vals.max()
        img = np.ones((show, show, 4))                       # forbidden by symmetry -> white
        img[allowed] = cmap(norm(vals[allowed]))
        a_.imshow(img, interpolation="nearest", origin="lower")
        name = SECTOR_NAMES[key][j]
        a_.set_title((HEADERS[key] + "\n\n" + name) if j == 0 else name, fontsize=9, pad=4)
        a_.set_xticks([0, 4, 8, 12]); a_.set_yticks([0, 4, 8, 12])
        a_.tick_params(labelsize=8, length=2, color=MUTED, labelcolor=MUTED)
        a_.set_ylabel(r"$n$", fontsize=9, labelpad=1)
        if j == last[key]:
            a_.set_xlabel(r"$m$", fontsize=9, labelpad=1)
        for s in a_.spines.values():
            s.set_color(MUTED); s.set_linewidth(0.5)
        print(f"{key:7s} {name:28s} <n> = {np.real(np.trace(np.diag(n_idx) @ rho)):.2f}, "
              f"weight beyond n={show - 1}: {np.real(np.trace(rho[show:, show:])):.1e}")

k_ax = ax["key"]; k_ax.axis("off")
cb_ax = k_ax.inset_axes([0.10, 0.70, 0.80, 0.07])
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, orientation="horizontal")
cb.set_label(r"$|\langle n|\rho|m\rangle|\,/\,\max|\rho|$", fontsize=9)
cb.ax.tick_params(labelsize=8)
k_ax.add_patch(Rectangle((0.10, 0.36), 0.07, 0.09, fc="white", ec=MUTED, lw=0.6, transform=k_ax.transAxes))
k_ax.text(0.20, 0.405, "forbidden by the symmetry\n(exactly zero)", fontsize=9, va="center",
          transform=k_ax.transAxes, color=INK)
for ext in ("png", "pdf"):
    fig.savefig(f"fig32_steadystates.{ext}", dpi=220, bbox_inches="tight")