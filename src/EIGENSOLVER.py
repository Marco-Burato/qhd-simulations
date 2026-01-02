from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, diags, eye, kron, lil_matrix
from scipy.sparse.linalg import eigsh

from UTILITIES import ComputationalBasisIndex_to_SpatialCoordinates


def diagonalize(
    d: int,
    q: int,
    times: np.ndarray,
    a_fun: Callable[[float], float],
    b_fun: Callable[[float], float],
    n_states: int,
    f: Callable[[np.ndarray], np.ndarray],
    domain: tuple[float, float],
    bc: str = "periodic",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize the time-dependent Hamiltonian H(t) on a Cartesian grid.

    The Hamiltonian is discretized as:
        H(t) = 0.5 * a(t) * D2 + b(t) * V,
    where:
      - D2 is the discrete Laplacian (second-difference operator) on a uniform grid
        with boundary conditions controlled by `bc`;
      - V is a diagonal operator constructed from the potential function f(x).

    The routine uses sparse operators and ARPACK (`eigsh`) by default, and falls back
    to dense diagonalization (`eigh`) only for very small 1D problems.

    Parameters
    ----------
    d:
        Spatial dimension.
    q:
        Qubits per dimension. The number of grid points per dimension is N = 2**q.
    times:
        Time grid of shape (Nt,).
    a_fun, b_fun:
        Schedule functions a(t) and b(t).
    n_states:
        Number of lowest-energy eigenpairs to compute at each time.
    f:
        Potential function. It must accept an array of coordinates with shape (M, d)
        and return an array broadcastable to shape (M,).
    domain:
        Tuple (x_min, x_max) defining the spatial domain per dimension.
    bc:
        Boundary condition, either "periodic" or "dirichlet".

    Returns
    -------
    energies:
        Array of shape (Nt, n_states) containing the lowest eigenvalues at each time.
        If the Hilbert space is too small for `eigsh` to return `n_states` eigenpairs,
        remaining entries are filled with NaN.
    eigenvectors:
        Array of shape (Nt, M, n_states) containing the corresponding eigenvectors
        in the computational basis. If padding occurs, unused columns are set to zero.

    Notes
    -----
    - The full Hilbert space dimension is M = (2**q)**d.
    - For sparse diagonalization, we request the smallest-algebraic eigenvalues
      (which="SA") and explicitly sort them.
    - The code intentionally preserves the original numerical choices (tolerances,
      padding behavior, etc.) to avoid changing downstream results.
    """
    # --- Grid sizes ---
    N = 2**q
    M = N**d
    x_min, x_max = map(float, domain)
    L = x_max - x_min
    dx = L / N

    bc = bc.lower()
    if bc not in ("periodic", "dirichlet"):
        raise ValueError("bc must be 'periodic' or 'dirichlet'.")

    # --- 1D discrete Laplacian (sparse) ---
    T1 = lil_matrix((N, N), dtype=np.float64)
    T1.setdiag(2.0)
    T1.setdiag(-1.0, k=+1)
    T1.setdiag(-1.0, k=-1)

    if bc == "periodic":
        # Periodic wrap-around coupling between endpoints.
        T1[0, -1] = -1.0
        T1[-1, 0] = -1.0
    # Dirichlet: no wrap-around couplings.

    T1 = (1.0 / dx**2) * T1.tocsr()
    I1 = eye(N, format="csr", dtype=np.float64)

    def kron_dim_sparse(A: csr_matrix, dim_index: int) -> csr_matrix:
        """
        Embed a 1D operator A into dimension `dim_index` via Kronecker products.

        This constructs:
            I ⊗ ... ⊗ I ⊗ A ⊗ I ⊗ ... ⊗ I
        with the position of A controlled by dim_index.
        """
        out = None
        for jj in reversed(range(d)):
            factor = A if jj == dim_index else I1
            out = factor if out is None else kron(out, factor, format="csr")
        return out

    # Sum Kronecker-embedded 1D Laplacians to obtain the dD Laplacian.
    D2_sparse = None
    for kdim in range(d):
        term = kron_dim_sparse(T1, kdim)
        D2_sparse = term if D2_sparse is None else (D2_sparse + term)

    # --- Diagonal potential operator (sparse) ---
    coords = ComputationalBasisIndex_to_SpatialCoordinates(q, d, domain, bc)  # (M, d)
    Vdiag = np.asarray(f(coords), dtype=np.float64).reshape(M)
    V = diags(Vdiag, offsets=0, shape=(M, M), format="csr")

    Nt = len(times)
    energies = np.empty((Nt, n_states), dtype=np.float64)
    eigenvectors = np.empty((Nt, M, n_states), dtype=np.complex128)

    # Prefer sparse diagonalization except for very small 1D instances.
    use_sparse = True
    small_dense = (M <= 4096) and (d == 1)  # conservative threshold
    if small_dense:
        use_sparse = False

    for it, t in enumerate(times):
        a = a_fun(t)
        b = b_fun(t)

        if use_sparse:
            # Keep the original operator definition and scaling.
            H = 0.5 * a * D2_sparse + b * V  # CSR

            # ARPACK requires k < M; protect against degenerate tiny sizes.
            k_eff = min(n_states, max(1, M - 2)) if M > 2 else 1

            vals, vecs = eigsh(H, k=k_eff, which="SA", tol=1e-8)

            # Sort to ensure ascending energies.
            order = np.argsort(vals)
            vals = vals[order]
            vecs = vecs[:, order]

            if k_eff < n_states:
                # Pad outputs to keep the public API stable.
                energies[it, :k_eff] = vals
                energies[it, k_eff:] = np.nan
                eigenvectors[it, :, :k_eff] = vecs
                eigenvectors[it, :, k_eff:] = 0.0
            else:
                energies[it, :] = vals
                eigenvectors[it, :, :] = vecs

        else:
            # Dense path only for very small cases.
            H = (0.5 * a * D2_sparse + b * V).toarray()
            vals, vecs = eigh(H, subset_by_index=(0, n_states - 1), check_finite=False)
            energies[it, :] = vals
            eigenvectors[it, :, :] = vecs

    return energies, eigenvectors


def weights(autovettori, psi_num):
    """
    Compute projection weights of a state onto an eigenbasis over time.

    For each time step t and eigenstate k, this returns:
        w(t, k) = | <phi_k(t) | psi(t)> |^2

    Parameters
    ----------
    autovettori:
        Eigenvectors over time, shape (Nt, M, K). (Name kept for backward compatibility.)
    psi_num:
        Reference state trajectory, shape (Nt, M).

    Returns
    -------
    np.ndarray
        Weights array of shape (Nt, K), with non-negative real entries.

    Notes
    -----
    This routine preserves the original implementation, including:
    - explicit Python loops (no vectorization changes);
    - use of `np.vdot` and |.|^2 in float precision.
    """
    Nt, M, K = autovettori.shape
    weights = np.empty((Nt, K), dtype=float)

    for it in range(Nt):
        for k in range(K):
            ck = np.vdot(autovettori[it, :, k], psi_num[it, :])
            weights[it, k] = float(np.abs(ck) ** 2)

    return weights
