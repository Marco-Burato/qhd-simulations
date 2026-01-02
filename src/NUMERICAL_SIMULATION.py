from typing import Callable, Optional, Tuple

import numpy as np
from scipy.fft import dstn, idstn

from UTILITIES import ComputationalBasisIndex_to_SpatialCoordinates, make_snapshot_schedule


def numerical_simulation(
    d: int,
    q: int,
    T: float,
    dt: float,
    *,
    a_fun: Callable[[float], float],
    b_fun: Callable[[float], float],
    f: Callable[[np.ndarray], np.ndarray],
    domain: Tuple[float, float],
    order: int = 1,
    n_snapshots: int = 101,
    bc: str = "periodic",
    psi0_fun: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """
    Numerically integrate the Schrödinger-like evolution using Trotter–Suzuki splitting.

    The simulation evolves a wavefunction ψ on a regular Cartesian grid using a
    split-operator approach with a time-dependent Hamiltonian of the form
        H(t) = a(t) * T + b(t) * V,
    where:
      - T is the kinetic operator implemented in spectral space (FFT or DST depending on BC),
      - V is a diagonal potential operator in position space.

    Parameters
    ----------
    d, q:
        Spatial dimension and qubits per axis. The number of grid points per axis is N = 2**q.
    T, dt:
        Final time and time step.
    a_fun, b_fun:
        Scalar schedules a(t) and b(t).
    f:
        Potential function. It must accept coords of shape (M, d) and return values
        broadcastable to shape (M,).
    domain:
        Spatial domain (x_min, x_max) for each axis.
    order:
        Splitting order:
          - 1: Lie–Trotter (global error ~ O(T * dt))
          - 2m (m >= 1, even integer): Suzuki splitting of order 2m
            (global error ~ O(T * dt**(2m))).
        Must satisfy: order == 1 or (order is even and order >= 2).
    n_snapshots:
        Number of snapshots stored (including t=0 and t=T). Snapshot selection is
        handled by `make_snapshot_schedule` and does not constrain the integrator.
    bc:
        Boundary conditions: "periodic" or "dirichlet".
          - "periodic": FFT basis.
          - "dirichlet": sine basis (DST-I), enforcing ψ=0 at the domain boundary.
    psi0_fun:
        Callable building the initial state ψ0(coords). Required.

    Returns
    -------
    states:
        Complex array of shape (Ns, M) containing ψ at the requested snapshot times,
        where Ns is the number of stored snapshots and M = N**d.

    Notes
    -----
    This function intentionally preserves the original numerical choices:
      - midpoint evaluation in the order-1 update (as in the provided implementation),
      - renormalization at each step for numerical stability,
      - Suzuki coefficient recursion and optional normalization.
    """
    if psi0_fun is None:
        raise ValueError("psi0_fun must be provided to construct the initial state.")

    # ------------------------------------------------------------------
    # Spatial coordinates and potential (position space)
    # ------------------------------------------------------------------
    coords = ComputationalBasisIndex_to_SpatialCoordinates(q, d, domain, bc)  # (M, d)
    N = 2**q
    M = coords.shape[0]
    L = float(domain[1] - domain[0])

    Vvals = np.asarray(f(coords), dtype=np.float64).reshape(M)

    # ------------------------------------------------------------------
    # Kinetic operator in spectral space (depends on boundary conditions)
    # ------------------------------------------------------------------
    bc = bc.lower()
    if bc == "periodic":
        # Periodic modes: k_n = 2π n / L with n from FFT frequency convention.
        m = (np.fft.fftfreq(N) * N).astype(np.float64)  # 0, 1, ..., -2, -1
        k1 = (2.0 * np.pi / L) * m
        transform = lambda arr: np.fft.fftn(arr, norm="ortho")
        itransform = lambda arr: np.fft.ifftn(arr, norm="ortho")
    elif bc == "dirichlet":
        # Dirichlet modes: sine basis with k_n ≈ nπ/L, n=1,...,N (DST-I).
        m = np.arange(1, N + 1, dtype=np.float64)
        k1 = (np.pi / L) * m
        transform = lambda arr: dstn(arr, type=1, norm="ortho")
        itransform = lambda arr: idstn(arr, type=1, norm="ortho")
    else:
        raise ValueError("bc must be 'periodic' or 'dirichlet'.")

    # Build 0.5 * |k|^2 over the dD grid via broadcasting.
    half_k2_1d = 0.5 * (k1**2)  # shape (N,)
    k2 = 0.0
    for ax in range(d):
        shape = [1] * d
        shape[ax] = N
        k2 = k2 + half_k2_1d.reshape(shape)

    def _apply_T(psi_vec: np.ndarray, a_val: float, dtau: float) -> np.ndarray:
        """
        Apply the kinetic block exp(-i a(t) * dtau * T) in spectral space.

        Notes
        -----
        The function reshapes ψ to an (N,)*d array, applies the appropriate transform
        (FFT/DST) to reach spectral space, multiplies by the phase factor, and transforms
        back. The exact transform choices are determined by `bc`.
        """
        arr = psi_vec.reshape([N] * d)
        arr = itransform(arr)

        phase = np.exp(-1j * a_val * dtau * k2)
        arr *= phase

        arr = transform(arr)
        return arr.reshape(M)

    def _apply_V(psi_vec: np.ndarray, b_val: float, dtau: float, Vvals_local: np.ndarray) -> np.ndarray:
        """
        Apply the potential block exp(-i b(t) * dtau * V) in position space.
        """
        return psi_vec * np.exp(-1j * b_val * dtau * Vvals_local)

    def S2_step(psi_vec: np.ndarray, t0: float, dtau: float) -> Tuple[np.ndarray, float]:
        """
        Perform one symmetric Strang step on [t0, t0 + dtau].

        The implemented step is:
            e^{-i (dtau/2) a(tm) T} e^{-i dtau b(tm) V} e^{-i (dtau/2) a(tm) T},
        where tm = t0 + dtau/2 (midpoint time).
        """
        t_mid = t0 + 0.5 * dtau
        psi_vec = _apply_T(psi_vec, a_fun(t_mid), 0.5 * dtau)
        psi_vec = _apply_V(psi_vec, b_fun(t_mid), dtau, Vvals)
        psi_vec = _apply_T(psi_vec, a_fun(t_mid), 0.5 * dtau)
        t_next = t_mid + 0.5 * dtau
        return psi_vec, t_next

    def _suzuki_coeffs(m: int) -> np.ndarray:
        """
        Return coefficients {c_j} such that:
            S_{2m}(Δt) = ∏_j S_2(c_j Δt)

        using Suzuki's fractal recursion:
            S_{2m}(x) = S_{2m-2}(p_m x)^2 S_{2m-2}((1-4 p_m)x) S_{2m-2}(p_m x)^2
        with:
            p_m = 1 / (4 - 4^{1/(2m-1)}).

        Convention: m=1 corresponds to S_2 itself, hence {c_j} = {1}.
        """
        if m < 1:
            raise ValueError("m must be >= 1.")
        if m == 1:
            return np.array([1.0], dtype=float)

        coeffs_prev = _suzuki_coeffs(m - 1)
        p_m = 1.0 / (4.0 - 4.0 ** (1.0 / (2.0 * m - 1.0)))
        blocks = [p_m, p_m, 1.0 - 4.0 * p_m, p_m, p_m]

        coeffs_list = []
        for b in blocks:
            # Each S_{2m-2}(b x) expands into a product of S_2(c_prev x).
            coeffs_list.extend(b * coeffs_prev)

        coeffs = np.array(coeffs_list, dtype=float)

        # Optional numerical normalization (kept as in the original implementation).
        coeffs /= coeffs.sum()
        return coeffs

    # ------------------------------------------------------------------
    # Snapshot timeline
    # ------------------------------------------------------------------
    snap_idx, _, nsteps, Tsim = make_snapshot_schedule(T, dt, n_snapshots)
    snap_set = set(int(i) for i in np.asarray(snap_idx, int))

    # ------------------------------------------------------------------
    # Initial state (normalized)
    # ------------------------------------------------------------------
    psi = np.asarray(psi0_fun(coords), dtype=np.complex128).reshape(M)
    norm = np.linalg.norm(psi)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Initial state norm is zero or not finite.")
    psi = psi / norm

    states = []
    if 0 in snap_set:
        states.append(psi.copy())

    # ------------------------------------------------------------------
    # Time evolution via splitting
    # ------------------------------------------------------------------
    if order == 1:
        # Lie–Trotter (as implemented): V then T, with midpoint evaluation.
        for j in range(nsteps):
            t0 = j * dt
            psi = _apply_V(psi, b_fun(t0 + dt / 2.0), dt, Vvals)
            psi = _apply_T(psi, a_fun(t0 + dt / 2.0), dt)

            # Renormalize for numerical stability.
            psi /= np.linalg.norm(psi)

            if (j + 1) in snap_set:
                states.append(psi.copy())
    else:
        if order < 2 or (order % 2) != 0:
            raise ValueError("order must be 1 or an even integer >= 2.")

        m = order // 2
        coeffs = _suzuki_coeffs(m)

        for j in range(nsteps):
            t0 = j * dt
            t = t0
            for c in coeffs:
                dt_block = c * dt
                psi, t = S2_step(psi, t, dt_block)

            # Renormalize for numerical stability.
            psi /= np.linalg.norm(psi)

            if (j + 1) in snap_set:
                states.append(psi.copy())

    return np.stack(states, axis=0)
