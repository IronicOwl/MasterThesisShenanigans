import numpy as np
import matplotlib.pyplot as plt


def initialize_random_complex_matrix(
    n: int = 15,
    seed: int | None = None,
    scale: float = 1.0
) -> np.ndarray:
    """
    Initialize an n x n complex matrix with real and imaginary parts
    uniformly distributed in [-1, 1].
    """
    rng = np.random.default_rng(seed)
    real_part = rng.uniform(-1.0, 1.0, size=(n, n))
    imag_part = rng.uniform(-1.0, 1.0, size=(n, n))
    return scale * (real_part + 1j * imag_part)


def split_diagonal_offdiagonal(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a matrix A into:
        D = diagonal part
        V = off-diagonal part
    so that A = D + V
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a square 2D array")

    D = np.diag(np.diag(matrix))
    V = matrix - D
    return D, V


def drop_small_offdiagonals(
    matrix: np.ndarray,
    original_matrix: np.ndarray,
    threshold: float = 0.01,
    atol: float = 0.0
) -> np.ndarray:
    """
    Zero out off-diagonal entries of `matrix` if

        |matrix[i,j]| < threshold * |original_matrix[i,j]|

    or, additionally, if

        |matrix[i,j]| < atol

    Diagonal entries are left unchanged.
    """
    if matrix.shape != original_matrix.shape:
        raise ValueError("matrix and original_matrix must have the same shape")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix and original_matrix must be square 2D arrays")

    result = matrix.copy()
    n = matrix.shape[0]

    for i in range(n):
        for j in range(n):
            if i != j:
                if (
                    np.abs(result[i, j]) < threshold * np.abs(original_matrix[i, j])
                    or np.abs(result[i, j]) < atol
                ):
                    result[i, j] = 0.0

    return result


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


def split_diagonal_offdiagonal(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    A = D + V
    D: diagonal part
    V: off-diagonal part
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a square 2D array")

    D = np.diag(np.diag(matrix))
    V = matrix - D
    return D, V


def eta_first(A: np.ndarray) -> np.ndarray:
    """
    First Rosso generator:
        eta_1 = [A^\dagger, V]
    """
    _, V = split_diagonal_offdiagonal(A)
    return commutator(A.conj().T, V)


def eta_second(A: np.ndarray) -> np.ndarray:
    """
    Second Rosso generator:
        eta_2 = [D^\dagger, V]
    """
    D, V = split_diagonal_offdiagonal(A)
    return commutator(D.conj().T, V)


def eta_third(A: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    """
    Third Rosso/White-like generator:
        eta_3[n,k] = V[n,k] / (D[n,n] - D[k,k])   if denominator != 0
                   = 0                            otherwise

    The diagonal is zero automatically because V has zero diagonal.
    """
    D, V = split_diagonal_offdiagonal(A)
    diag = np.diag(D)
    n = A.shape[0]

    eta = np.zeros_like(A, dtype=np.complex128)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            denom = diag[i] - diag[j]
            if np.abs(denom) > eps:
                eta[i, j] = V[i, j] / denom
            else:
                eta[i, j] = 0.0

    return eta


def get_eta(A: np.ndarray, generator: str = "first", eps: float = 1e-14) -> np.ndarray:
    """
    Dispatch function for generator choice.

    Allowed values:
        'first'  -> eta_1 = [A^\dagger, V]
        'second' -> eta_2 = [D^\dagger, V]
        'third'  -> eta_3[n,k] = V[n,k] / (D_nn - D_kk)
    """
    generator = generator.lower()

    if generator == "first":
        return eta_first(A)
    elif generator == "second":
        return eta_second(A)
    elif generator == "third":
        return eta_third(A, eps=eps)
    else:
        raise ValueError(
            "Unknown generator. Choose from {'first', 'second', 'third'}."
        )


def flow_rhs(A: np.ndarray, generator: str = "first", eps: float = 1e-14) -> np.ndarray:
    """
    Flow equation RHS:
        dA/dl = [eta(A), A]
    """
    eta = get_eta(A, generator=generator, eps=eps)
    return commutator(eta, A)


def rk4_step(f, y: np.ndarray, h: float, **f_kwargs) -> np.ndarray:
    k1 = f(y, **f_kwargs)
    k2 = f(y + 0.5 * h * k1, **f_kwargs)
    k3 = f(y + 0.5 * h * k2, **f_kwargs)
    k4 = f(y + h * k3, **f_kwargs)

    return y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def initialize_diagonal_tracker(n: int) -> dict:
    """
    Create a container to store the flow of diagonal entries.

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    dict
        Tracker dictionary for storing flow data.
    """
    return {
        "l_values": [],
        "diag_real": [[] for _ in range(n)],
        "diag_imag": [[] for _ in range(n)],
    }


def record_diagonal_step(tracker: dict, A: np.ndarray, l: float) -> None:
    """
    Record the diagonal entries of the current matrix A at flow parameter l.

    Parameters
    ----------
    tracker : dict
        Tracker created by initialize_diagonal_tracker.
    A : np.ndarray
        Current matrix.
    l : float
        Current flow parameter.
    """
    diag = np.diag(A)
    tracker["l_values"].append(l)

    for i, val in enumerate(diag):
        tracker["diag_real"][i].append(val.real)
        tracker["diag_imag"][i].append(val.imag)


def plot_diagonal_flow(tracker: dict,
                       figsize: tuple[float, float] = (8, 6),
                       show_legend: bool = False,
                       title_prefix: str = "Diagonal flow") -> None:
    """
    Plot the real and imaginary parts of the diagonal entries in two separate plots.

    Parameters
    ----------
    tracker : dict
        Tracker containing recorded diagonal flow data.
    figsize : tuple
        Figure size for each plot.
    show_legend : bool
        Whether to show a legend for each diagonal element.
    title_prefix : str
        Prefix for plot titles.
    """
    l_values = np.array(tracker["l_values"])
    diag_real = tracker["diag_real"]
    diag_imag = tracker["diag_imag"]

    # Plot real parts
    plt.figure(figsize=figsize)
    for i, y in enumerate(diag_real):
        label = rf"$\mathrm{{Re}}\,A_{{{i+1}{i+1}}}$" if show_legend else None
        plt.plot(l_values, y, label=label)
    plt.xlabel(r"$\ell$")
    plt.ylabel("Real part of diagonal entries")
    plt.title(f"{title_prefix}: real parts")
    if show_legend:
        plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

    # Plot imaginary parts
    plt.figure(figsize=figsize)
    for i, y in enumerate(diag_imag):
        label = rf"$\mathrm{{Im}}\,A_{{{i+1}{i+1}}}$" if show_legend else None
        plt.plot(l_values, y, label=label)
    plt.xlabel(r"$\ell$")
    plt.ylabel("Imaginary part of diagonal entries")
    plt.title(f"{title_prefix}: imaginary parts")
    if show_legend:
        plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

def solve_flow_rk4(
    A0: np.ndarray,
    dl: float = 1e-3,
    n_steps: int = 15000,
    threshold: float = 0.01,
    atol_drop: float = 0.0,
    generator: str = "first",
    eta_eps: float = 1e-14,
    track_diagonal: bool = False
):
    A = A0.astype(np.complex128).copy()
    A_initial = A.copy()

    tracker = initialize_diagonal_tracker(A.shape[0]) if track_diagonal else None
    if track_diagonal:
        record_diagonal_step(tracker, A, l=0.0)

    for step in range(n_steps):
        A = rk4_step(
            flow_rhs,
            A,
            dl,
            generator=generator,
            eps=eta_eps
        )

        A = drop_small_offdiagonals(
            A,
            A_initial,
            threshold=threshold,
            atol=atol_drop
        )

        if track_diagonal:
            current_l = (step + 1) * dl
            record_diagonal_step(tracker, A, l=current_l)

    return A, tracker


A0 = initialize_random_complex_matrix(n=15, seed=42)

A_first, tracker_first = solve_flow_rk4(
    A0,
    generator="first",
    track_diagonal=True
)

A_second, tracker_second = solve_flow_rk4(
    A0,
    generator="second",
    track_diagonal=True
)

A_third, tracker_third = solve_flow_rk4(
    A0,
    generator="third",
    track_diagonal=True
)
plot_diagonal_flow(tracker_first, title_prefix="First generator")
plot_diagonal_flow(tracker_second, title_prefix="Second generator")
plot_diagonal_flow(tracker_third, title_prefix="Third generator")