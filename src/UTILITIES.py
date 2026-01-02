from __future__ import annotations

import warnings
from functools import wraps
from typing import Callable, Dict, Optional, Tuple

import numpy as np


# =============================================================================
# Snapshot schedule utilities
# =============================================================================

def make_snapshot_schedule(
    T: float,
    dt: float,
    n_snapshots: int = 101,
    *,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """
    Build a snapshot schedule for fixed-step time integration.

    The integrator advances with a constant step `dt`. Since `T/dt` may not be an
    integer, the simulation time is truncated to:

        nsteps = floor(T / dt)   (up to numerical tolerance)
        T_sim  = nsteps * dt     <= T

    Snapshot indices are chosen to (approximately) span `[0, T_sim]` with
    `n_snapshots` points.

    Parameters
    ----------
    T:
        Target final time (requested).
    dt:
        Fixed integrator time step.
    n_snapshots:
        Number of snapshots to store, including the initial state (t=0) and the
        last stored time (t=T_sim). Must be at least 2.
    tol:
        Absolute tolerance used when deciding whether `T/dt` is (numerically)
        an integer.

    Returns
    -------
    snap_idx:
        Integer step indices of snapshots in `[0, nsteps]`.
    snap_times:
        Snapshot times in physical units, `snap_idx * dt`.
    nsteps:
        Total number of integration steps to perform.
    T_sim:
        Effective simulated final time, `nsteps * dt` (guaranteed `<= T`).

    Warnings
    --------
    - If `T/dt` is not (close to) an integer, `T` is not exactly reachable with
      a fixed step; `T_sim` is used instead.
    - If `nsteps` is not a multiple of `n_snapshots - 1`, snapshots cannot be
      exactly equally spaced in time using integer step indices. In that case,
      rounded indices are used and may be non-uniform (and may contain duplicates
      before de-duplication).
    """
    if T <= 0.0 or dt <= 0.0:
        raise ValueError("T and dt must be > 0.")
    if n_snapshots < 2:
        raise ValueError("n_snapshots must be >= 2.")

    ratio = T / dt
    nsteps_floor = int(np.floor(ratio + tol))
    nsteps_round = int(np.rint(ratio))

    # 1) Reachability of T with constant dt
    if not np.isclose(ratio, nsteps_round, rtol=0.0, atol=tol):
        T_sim = nsteps_floor * dt
        warnings.warn(
            f"T/dt = {ratio:.16g} is not an integer: with constant dt you cannot "
            f"reach exactly T.\n"
            f"Using nsteps=floor(T/dt)={nsteps_floor}, hence "
            f"T_sim=nsteps*dt={T_sim:.16g} (delta={T - T_sim:.3e}).\n"
            f"If you compare against a reference at T, an O(dt) discrepancy may "
            f"appear for first-order methods.",
            RuntimeWarning,
            stacklevel=2,
        )
        nsteps = max(1, nsteps_floor)
    else:
        nsteps = max(1, nsteps_round)

    T_sim = nsteps * dt

    # 2) Exact equal spacing in time requires nsteps divisible by (n_snapshots - 1)
    denom = n_snapshots - 1
    if nsteps % denom != 0:
        k = nsteps / denom
        warnings.warn(
            f"nsteps={nsteps} is not a multiple of (n_snapshots-1)={denom}.\n"
            f"Exact equally-spaced snapshots in time are not possible with fixed dt "
            f"and integer step indices.\n"
            f"Rounded indices will be used; spacing may be non-uniform (k≈{k:.6g}).",
            RuntimeWarning,
            stacklevel=2,
        )
        snap_idx = np.rint(np.linspace(0, nsteps, n_snapshots)).astype(int)
        snap_idx = np.unique(np.clip(snap_idx, 0, nsteps))
    else:
        stride = nsteps // denom
        snap_idx = np.arange(0, nsteps + 1, stride, dtype=int)

    snap_times = snap_idx * dt
    return snap_idx, snap_times, nsteps, T_sim


# =============================================================================
# Coordinate mapping: computational basis -> spatial grid
# =============================================================================

def ComputationalBasisIndex_to_SpatialCoordinates(
    q: int,
    d: int,
    domain: Tuple[float, float],
    bc: str = "periodic",
) -> np.ndarray:
    """
    Map computational-basis indices |x⟩ to physical spatial coordinates in [a, b]^d.

    The grid has `N = 2**q` points per axis, and the total number of basis states is
    `M = 2**(d*q) = N**d`. Coordinates are returned in an array of shape (M, d).

    Parameters
    ----------
    q:
        Qubits per spatial axis. Grid points per axis: N = 2**q.
    d:
        Spatial dimension.
    domain:
        Spatial interval (a, b) applied to every axis.
    bc:
        Boundary condition convention used to choose grid points:
        - "periodic": uniform points on [a, b), i.e. x_j = a + (j/N)*(b-a)
        - "dirichlet": interior points only (walls at a and b), i.e.
          x_j = a + (j/(N+1))*(b-a), for j=1..N

    Returns
    -------
    coords:
        Array of shape (M, d) with spatial coordinates for each computational basis index.

    Notes
    -----
    This function assumes the bit layout where each axis consumes `q` bits, and the axis
    index `j` uses bits [j*q, (j+1)*q). This matches the original implementation and is
    relied upon by other modules.
    """
    a, b = domain
    L = b - a
    N = 2**q
    M = 2 ** (d * q)

    bc = bc.lower()
    if bc == "periodic":
        axis = np.arange(N, dtype=np.float64) / N
    elif bc == "dirichlet":
        axis = np.arange(1, N + 1, dtype=np.float64) / (N + 1)
    else:
        raise ValueError("bc must be 'periodic' or 'dirichlet'.")

    # Basis indices 0..M-1 as uint64 so bit ops are well-defined.
    basis = np.arange(M, dtype=np.uint64).reshape(M, 1)

    # Extract q-bit chunks per dimension.
    per_dim = []
    mask_q = (1 << q) - 1
    for j in range(d):
        idx_j = ((basis >> (j * q)) & mask_q).reshape(M)
        per_dim.append(idx_j)

    per_dim = np.stack(per_dim, axis=1)  # (M, d), integer indices on each axis
    return a + L * axis[per_dim]


# =============================================================================
# Benchmark potentials
# =============================================================================
# Convention:
#   Each benchmark returns (values, domain), where:
#     - values: np.ndarray of shape (M,), computed from coords (M, d)
#     - domain: (lo, hi) tuple, interpreted as [lo, hi]^d
#
# Important: Do not change numerical formulas here; only documentation/structure.
# =============================================================================

def ackley(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Ackley function (d-dimensional).

    Definition
    ----------
    f(x) = -20 exp(-0.2 sqrt((1/d) sum_i x_i^2))
           - exp((1/d) sum_i cos(2π x_i))
           + 20 + e

    Domain
    ------
    x_i in [-2.048, 2.048]
    """
    d = coords.shape[1]
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(coords**2, axis=1) / d))
    term2 = -np.exp(np.mean(np.cos(2 * np.pi * coords), axis=1))
    function = term1 + term2 + 20.0 + np.e
    domain = (-2.048, 2.048)
    return function, domain


def ackley2_2d(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Ackley N.2 (2D).

    Definition
    ----------
    f(x, y) = -200 exp(-0.2 sqrt(x^2 + y^2))

    Domain
    ------
    x, y in [-32, 32]
    """
    assert coords.shape[1] == 2, "This benchmark requires d=2."
    function = -200.0 * np.exp(-0.2 * np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2))
    domain = (-32.0, 32.0)
    return function, domain


def alpine1(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Alpine 1 (d-dimensional).

    Definition
    ----------
    f(x) = sum_i | x_i sin(x_i) + 0.1 x_i |

    Domain
    ------
    x_i in [-2, 2]
    """
    function = np.sum(np.abs(coords * np.sin(coords) + 0.1 * coords), axis=1)
    domain = (-2.0, 2.0)
    return function, domain


def alpine2(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Alpine 2 (d-dimensional).

    Definition
    ----------
    f(x) = prod_i sqrt(x_i) sin(x_i)

    Domain
    ------
    x_i in [0, 10]
    """
    function = np.prod(np.sqrt(coords) * np.sin(coords), axis=1)
    domain = (0.0, 10.0)
    return function, domain


def bohachevsky2_2d(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Bohachevsky #2 (2D).

    Definition
    ----------
    f(x, y) = x^2 + 2 y^2 - 0.3 cos(3πx) cos(4πy) + 0.3

    Domain
    ------
    x, y in [-2, 2]
    """
    assert coords.shape[1] == 2, "This benchmark requires d=2."
    function = (
        coords[:, 0] ** 2
        + 2.0 * coords[:, 1] ** 2
        - 0.3 * np.cos(3.0 * np.pi * coords[:, 0]) * np.cos(4.0 * np.pi * coords[:, 1])
        + 0.3
    )
    domain = (-2.0, 2.0)
    return function, domain


def camel3_2d(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Three-Hump Camel (2D).

    Definition
    ----------
    f(x, y) = 2x^2 - 1.05x^4 + x^6/6 + x y + y^2

    Domain
    ------
    x, y in [-4, 4]
    """
    assert coords.shape[1] == 2, "This benchmark requires d=2."
    x = coords[:, 0]
    y = coords[:, 1]
    function = 2.0 * x**2 - 1.05 * x**4 + (x**6) / 6.0 + x * y + y**2
    domain = (-4.0, 4.0)
    return function, domain


def csendes(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Csendes function (d-dimensional).

    Definition
    ----------
    f(x) = sum_i x_i^6 * (2 + sin(1/x_i)), with sin(1/0)=0 by convention.

    Domain
    ------
    x_i in [-1, 1]
    """
    inv = np.where(np.abs(coords) > 1e-12, 1.0 / coords, 0.0)
    function = np.sum((coords**6) * (2.0 + np.sin(inv)), axis=1)
    domain = (-1.0, 1.0)
    return function, domain


def deflected_corrugated_spring(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Deflected Corrugated Spring (d-dimensional).

    Definition
    ----------
    f(x) = 0.1 * sum_i x_i^2 - cos(5 * sqrt(sum_i x_i^2))

    Domain
    ------
    x_i in [-2, 2]
    """
    r2 = np.sum(coords**2, axis=1)
    function = 0.1 * r2 - np.cos(5.0 * np.sqrt(r2))
    domain = (-2.0, 2.0)
    return function, domain


def dropwave_2d(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Drop-Wave (2D).

    Definition
    ----------
    f(x, y) = - (1 + cos(12 r)) / (0.5 r^2 + 2),  r = sqrt(x^2 + y^2)

    Domain
    ------
    x, y in [-2, 2]
    """
    assert coords.shape[1] == 2, "This benchmark requires d=2."
    r2 = np.sum(coords**2, axis=1)
    r = np.sqrt(r2)
    function = -(1.0 + np.cos(12.0 * r)) / (0.5 * r2 + 2.0)
    domain = (-2.0, 2.0)
    return function, domain


def easom_2d(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Easom (2D).

    Definition
    ----------
    f(x, y) = -cos(x) cos(y) * exp(-(x-π)^2 - (y-π)^2)

    Domain
    ------
    x, y in [-2, 2]
    """
    assert coords.shape[1] == 2, "This benchmark requires d=2."
    x = coords[:, 0]
    y = coords[:, 1]
    function = -np.cos(x) * np.cos(y) * np.exp(-(x - np.pi) ** 2 - (y - np.pi) ** 2)
    domain = (-2.0, 2.0)
    return function, domain


def griewank(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Griewank function (d-dimensional).

    Definition
    ----------
    f(x) = 1 + (1/4000) sum_i x_i^2 - prod_i cos(x_i / sqrt(i))

    Domain
    ------
    x_i in [-10, 10]
    """
    i = np.arange(1, coords.shape[1] + 1, dtype=float)
    function = 1.0 + 0.00025 * np.sum(coords**2, axis=1) - np.prod(
        np.cos(coords / np.sqrt(i)), axis=1
    )
    domain = (-10.0, 10.0)
    return function, domain


def holder_table_2d(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Hölder Table (2D).

    Definition
    ----------
    f(x, y) = -| sin(x) cos(y) * exp(|1 - sqrt(x^2+y^2)/π|) |

    Domain
    ------
    x, y in [0, 10]
    """
    assert coords.shape[1] == 2, "This benchmark requires d=2."
    x = coords[:, 0]
    y = coords[:, 1]
    function = -np.abs(
        np.sin(x) * np.cos(y) * np.exp(np.abs(1.0 - np.sqrt(x**2 + y**2) / np.pi))
    )
    domain = (0.0, 10.0)
    return function, domain


def hosaki_2d(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Hosaki (2D).

    Definition
    ----------
    f(x, y) = (1 - 8x + 7x^2 - (7/3)x^3 + (1/4)x^4) * y^2 * exp(-y)

    Domain
    ------
    x, y in [0, 5]
    """
    assert coords.shape[1] == 2, "This benchmark requires d=2."
    x = coords[:, 0]
    y = coords[:, 1]
    poly = 1 - 8 * x + 7 * x**2 - (7.0 / 3.0) * x**3 + 0.25 * x**4
    function = poly * (y**2) * np.exp(-y)
    domain = (0.0, 5.0)
    return function, domain


def levy(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Lévy function (d-dimensional).

    Let w_i = 1 + (x_i - 1)/4.

    Definition
    ----------
    f(x) = sin^2(π w_1)
           + sum_{i=1}^{d-1} (w_i - 1)^2 [1 + 10 sin^2(π w_i + 1)]
           + (w_d - 1)^2 [1 + sin^2(2π w_d)]

    Domain
    ------
    x_i in [-10, 10]
    """
    w = 1 + (coords - 1) / 4.0
    term1 = np.sin(np.pi * w[:, 0]) ** 2
    if coords.shape[1] > 1:
        term2 = np.sum(
            (w[:, :-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:, :-1] + 1.0) ** 2),
            axis=1,
        )
    else:
        term2 = 0.0
    term3 = (w[:, -1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[:, -1]) ** 2)
    function = term1 + term2 + term3
    domain = (-10.0, 10.0)
    return function, domain


def levy13_2d(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Lévy N.13 (2D).

    Definition
    ----------
    f(x, y) = sin^2(3πx)
              + (x-1)^2 [1 + sin^2(3πy)]
              + (y-1)^2 [1 + sin^2(2πy)]

    Domain
    ------
    x, y in [-2, 2]
    """
    assert coords.shape[1] == 2, "This benchmark requires d=2."
    x = coords[:, 0]
    y = coords[:, 1]
    function = (
        np.sin(3 * np.pi * x) ** 2
        + (x - 1.0) ** 2 * (1.0 + np.sin(3 * np.pi * y) ** 2)
        + (y - 1.0) ** 2 * (1.0 + np.sin(2 * np.pi * y) ** 2)
    )
    domain = (-2.0, 2.0)
    return function, domain


def michalewicz(coords: np.ndarray, m: int = 10) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Michalewicz function (d-dimensional).

    Definition
    ----------
    f(x) = - sum_i sin(x_i) * [sin(i * x_i^2 / π)]^{2m}

    Domain
    ------
    x_i in [0, π]

    Notes
    -----
    Parameter `m` controls the steepness of the landscape (default 10).
    """
    i = np.arange(1, coords.shape[1] + 1, dtype=float)
    function = -np.sum(
        np.sin(coords) * (np.sin((i * (coords**2)) / np.pi) ** (2 * m)),
        axis=1,
    )
    domain = (0.0, float(np.pi))
    return function, domain


def rastrigin(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Rastrigin function (d-dimensional).

    Definition
    ----------
    f(x) = A d + sum_i (x_i^2 - A cos(2π x_i)), with A=10

    Domain
    ------
    x_i in [-2, 2]
    """
    d = coords.shape[1]
    A = 10.0
    function = A * d + np.sum(coords**2 - A * np.cos(2 * np.pi * coords), axis=1)
    domain = (-2.0, 2.0)
    return function, domain


def shubert_2d(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Shubert function (2D).

    Definition
    ----------
    f(x, y) = [ sum_{i=1}^5 i cos((i+1)x + i) ] * [ sum_{i=1}^5 i cos((i+1)y + i) ]

    Domain
    ------
    x, y in [-2, 2]
    """
    assert coords.shape[1] == 2, "This benchmark requires d=2."
    i = np.arange(1, 6, dtype=float).reshape(1, -1)  # shape (1, 5)
    sx = np.sum(i * np.cos((i + 1) * coords[:, 0].reshape(-1, 1) + i), axis=1)
    sy = np.sum(i * np.cos((i + 1) * coords[:, 1].reshape(-1, 1) + i), axis=1)
    function = sx * sy
    domain = (-2.0, 2.0)
    return function, domain


def styblinski_tang(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Styblinski–Tang function (d-dimensional).

    Definition
    ----------
    f(x) = 0.5 * sum_i (x_i^4 - 16 x_i^2 + 5 x_i)

    Domain
    ------
    x_i in [-5, 5]
    """
    function = 0.5 * np.sum(coords**4 - 16 * coords**2 + 5 * coords, axis=1)
    domain = (-5.0, 5.0)
    return function, domain


def sum_of_squares(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Sum of Squares / Axis-Parallel Hyper-Ellipsoid (d-dimensional).

    Definition
    ----------
    f(x) = sum_{i=1}^d i * x_i^2

    Domain
    ------
    x_i in [-5, 5]
    """
    i = np.arange(1, coords.shape[1] + 1, dtype=float)
    function = np.sum(i * (coords**2), axis=1)
    domain = (-5.0, 5.0)
    return function, domain


def xin_she_yang3(
    coords: np.ndarray, beta: float = 15.0, m: int = 3
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Xin-She Yang N.3 (d-dimensional).

    Definition
    ----------
    f(x) = exp(-sum_i (x_i/beta)^{2m})
           - 2 exp(-sum_i x_i^2) * prod_i cos^2(x_i)

    Domain
    ------
    x_i in [-20, 20]
    """
    function = np.exp(-np.sum((coords / beta) ** (2 * m), axis=1)) - 2.0 * np.exp(
        -np.sum(coords**2, axis=1)
    ) * np.prod(np.cos(coords) ** 2, axis=1)
    domain = (-20.0, 20.0)
    return function, domain


def asymmetric_double_well(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Asymmetric double well (d-dimensional).

    Definition
    ----------
    f(x) = sum_i [ (x_i^2 - 0.25)^2 - 0.1 x_i ]

    Domain
    ------
    x_i in [-1, 1]
    """
    function = np.sum((coords**2 - 0.5**2) ** 2 - 0.1 * coords, axis=1)
    domain = (-1.0, 1.0)
    return function, domain


def periodic_ripples(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Periodic Ripples (d-dimensional).

    Definition
    ----------
    f(x) = sum_i [ -1.5 cos(x_i - 0.5) + 0.4 cos(3 x_i + 0.4) + 0.2 cos(5 x_i - 0.9) ]

    Domain
    ------
    x_i in [-π, π]
    """
    function = np.sum(
        (-1.5 * np.cos(coords - 0.5))
        + (0.4 * np.cos(3.0 * coords + 0.4))
        + (0.2 * np.cos(5.0 * coords - 0.9)),
        axis=1,
    )
    domain = (-float(np.pi), float(np.pi))
    return function, domain


# Registry: string name -> (values, domain) benchmark callable
NAME_TO_POTENTIAL: Dict[str, Callable[[np.ndarray], tuple[np.ndarray, tuple[float, float]]]] = {
    "ackley": ackley,
    "ackley2_2d": ackley2_2d,
    "alpine1": alpine1,
    "alpine2": alpine2,
    "bohachevsky2_2d": bohachevsky2_2d,
    "camel3_2d": camel3_2d,
    "csendes": csendes,
    "deflected_corrugated_spring": deflected_corrugated_spring,
    "dropwave_2d": dropwave_2d,
    "easom_2d": easom_2d,
    "griewank": griewank,
    "holder_table_2d": holder_table_2d,
    "hosaki_2d": hosaki_2d,
    "levy": levy,
    "levy13_2d": levy13_2d,
    "michalewicz": michalewicz,
    "rastrigin": rastrigin,
    "shubert_2d": shubert_2d,
    "styblinski_tang": styblinski_tang,
    "sum_of_squares": sum_of_squares,
    "xin_she_yang3": xin_she_yang3,
    "asymmetric_double_well": asymmetric_double_well,
    "periodic_ripples": periodic_ripples,
}


# =============================================================================
# TeX formulas for potentials (used by the reporting layer)
# =============================================================================

NAME_TO_F_TEX: Dict[str, str] = {
    "ackley": (
        r"$f\left(\mathbf{x}\right)= -20 e^{-0.2 \sqrt{\frac{1}{d} \sum x_i^2}}"
        r" - e^{\sum \frac{\cos\left(2\pi x_i\right)}{d}} + 20 + e$"
    ),
    "ackley2_2d": r"$f\left(x,y\right) = -200 e^{-0.2 \sqrt{x^2 + y^2}}$",
    "alpine1": r"$f\left(\mathbf{x}\right) = \sum \left|x_i \sin\left(x_i\right) + 0.1\,x_i\right|$",
    "alpine2": r"$f\left(\mathbf{x}\right) = \prod_{i=1}^d \sqrt{x_i}\,\sin\left(x_i\right)$",
    "bohachevsky2_2d": (
        r"$f\left(x,y\right) = x^2 + 2 y^2"
        r" - 0.3 \cos\left(3\pi x\right)\,\cos\left(4\pi y\right) + 0.3$"
    ),
    "camel3_2d": r"$f\left(x,y\right) = 2 x^2 - 1.05 x^4 + \frac{x^6}{6} + x y + y^2$",
    "csendes": r"$f\left(\mathbf{x}\right) = \sum x_i^6 \left(2 + \sin\left(\frac{1}{x_i}\right)\right)$",
    "deflected_corrugated_spring": (
        r"$f\left(\mathbf{x}\right) = 0.1 \sum x_i^2 - \cos\left(5 \sqrt{\sum x_i^2}\right)$"
    ),
    "dropwave_2d": (
        r"$f\left(x,y\right) = -\frac{1 + \cos\left(12 \sqrt{x^2 + y^2}\right)}"
        r"{0.5 \left(x^2 + y^2\right) + 2}$"
    ),
    "easom_2d": (
        r"$f\left(x,y\right) = -\cos\left(x\right)\,\cos\left(y\right)"
        r" e^{-\left(x-\pi\right)^2 - \left(y-\pi\right)^2}$"
    ),
    "griewank": (
        r"$f\left(\mathbf{x}\right) = 1 + \frac{1}{4000} \sum x_i^2"
        r" - \prod_{i=1}^d \cos\left(\frac{x_i}{\sqrt{i}}\right)$"
    ),
    "holder_table_2d": (
        r"$f\left(x,y\right) = -\left|\sin\left(x\right)\,\cos\left(y\right)\,"
        r" e^{\left|1 - \frac{\sqrt{x^2 + y^2}}{\pi}\right|}\right|$"
    ),
    "hosaki_2d": (
        r"$f\left(x,y\right) = \left(1 - 8x + 7x^2 - \frac{7}{3} x^3 + \frac{1}{4} x^4\right)"
        r" y^2 e^{-y}$"
    ),
    "levy": (
        r"$f\left(\mathbf{x}\right) = \sin^2\left(\pi \left(1 + \frac{x_1 - 1}{4}\right)\right) +$" "\n"
        r"$+\sum_{i=1}^{d-1} \left(\frac{x_i - 1}{4}\right)^2"
        r" \left[1 + 10 \sin^2\left(\pi \left(1 + \frac{x_i - 1}{4}\right) + 1\right)\right] +$" "\n"
        r"$+ \left(\frac{x_d - 1}{4}\right)^2"
        r" \left[1 + \sin^2\left(2\pi \left(1 + \frac{x_d - 1}{4}\right)\right)\right]$"
    ),
    "levy13_2d": (
        r"$f\left(x,y\right) = \sin^2\left(3\pi x\right) +$" "\n"
        r"$ + \left(x-1\right)^2 \left[1 + \sin^2\left(3\pi y\right)\right]$" "\n"
        r"$ + \left(y-1\right)^2 \left[1 + \sin^2\left(2\pi y\right)\right]$"
    ),
    "michalewicz": (
        r"$f\left(\mathbf{x}\right) = -\sum \sin\left(x_i\right)\,"
        r"\left[\sin\left(\frac{i x_i^2}{\pi}\right)\right]^{20}$"
    ),
    "rastrigin": r"$f\left(\mathbf{x}\right) = 10d + \sum \left(x_i^2 - 10 \cos\left(2\pi x_i\right)\right)$",
    "shubert_2d": (
        r"$f\left(x,y\right) = \left[\sum_{i=1}^5 i \cos\left(\left(i+1\right) x + i\right)\right]"
        r"\,\left[\sum_{i=1}^5 i \cos\left(\left(i+1\right) y + i\right)\right]$"
    ),
    "styblinski_tang": r"$f\left(\mathbf{x}\right) = \frac{1}{2} \sum \left(x_i^4 - 16 x_i^2 + 5 x_i\right)$",
    "sum_of_squares": r"$f\left(\mathbf{x}\right) = \sum i\,x_i^2$",
    "xin_she_yang3": (
        r"$f\left(\mathbf{x}\right) = e^{-\sum \left(x_i/15\right)^{20}}"
        r" -2 e^{-\sum x_i^2} \prod_{i=1}^d \cos^2\left(x_i\right)$"
    ),
    "asymmetric_double_well": (
        r"$f\left(\mathbf{x}\right) = \sum \left(\left(x_i^2 - 0.25\right)^2 - 0.1\,x_i\right)$"
    ),
    "periodic_ripples": (
        r"$f\left(\mathbf{x}\right)=\sum_i\left[-1.5\cos\left(x_i-0.5\right)"
        r"+0.4\cos\left(3x_i+0.4\right)+0.2\cos\left(5x_i-0.9\right)\right]$"
    ),
}


# =============================================================================
# Potential resolution (string registry or custom callable)
# =============================================================================

PotentialFn = Callable[[np.ndarray], np.ndarray]
Resolved = Tuple[PotentialFn, Tuple[float, float], Optional[str]]


def _values_only(fn: Callable) -> PotentialFn:
    """
    Wrap a benchmark function so it returns only the potential values.

    The benchmark registry functions typically return `(values, domain)`. Downstream
    simulation code expects a callable that returns only `values`.
    """
    @wraps(fn)
    def g(coords: np.ndarray) -> np.ndarray:
        out = fn(coords)
        if isinstance(out, tuple) and len(out) == 2:
            return out[0]
        return out
    return g


def _extract_domain(fn: Callable, prefer_2d: bool) -> Tuple[float, float]:
    """
    Attempt to infer the domain of a benchmark by calling it with an empty input.

    Some benchmarks enforce dimensionality via asserts (e.g. d=2). This helper tries
    an empty array with a plausible dimension and extracts the `(lo, hi)` domain from
    the returned tuple `(values, domain)`.

    Parameters
    ----------
    fn:
        Benchmark callable.
    prefer_2d:
        If True, try d=2 first, then d=1; otherwise try d=1 first.

    Returns
    -------
    (lo, hi):
        Domain bounds. Falls back to (-1.0, 1.0) if inference fails.
    """
    for d_guess in ([2, 1] if prefer_2d else [1, 2]):
        try:
            dummy = np.empty((0, d_guess))
            out = fn(dummy)
            if isinstance(out, tuple) and len(out) == 2:
                dom = out[1]
                if isinstance(dom, (tuple, list)) and len(dom) == 2:
                    return float(dom[0]), float(dom[1])
        except Exception:
            continue

    return (-1.0, 1.0)


# Pre-compute domains for registered benchmarks (avoids repeated probing).
NAME_TO_DOMAIN: Dict[str, Tuple[float, float]] = {}
for _name, _fn in NAME_TO_POTENTIAL.items():
    NAME_TO_DOMAIN[_name] = _extract_domain(_fn, prefer_2d=_name.endswith("_2d"))


def resolve_potential(f_item) -> Resolved:
    """
    Resolve a benchmark potential from either a registry name or a callable.

    Parameters
    ----------
    f_item:
        Either:
        - str: key in `NAME_TO_POTENTIAL`, or
        - callable: custom potential function.

    Returns
    -------
    f_values_only:
        Callable `f(coords) -> values` (domain stripped off if the function returns it).
    domain:
        Tuple `(lo, hi)` used as the default spatial domain for the potential.
    f_tex:
        TeX string describing the potential (or None). For custom callables, you may
        set `f_item._f_tex = r"$f(x)=...$"` to propagate the formula to the report.

    Raises
    ------
    ValueError
        If `f_item` is neither a recognized string key nor a callable.
    """
    # Case 1: registry lookup by name
    if isinstance(f_item, str):
        if f_item not in NAME_TO_POTENTIAL:
            raise ValueError(f"Unrecognized potential: {f_item!r}")

        fn = NAME_TO_POTENTIAL[f_item]
        dom = NAME_TO_DOMAIN.get(
            f_item,
            _extract_domain(fn, prefer_2d=f_item.endswith("_2d")),
        )
        f_tex = NAME_TO_F_TEX.get(f_item)
        return _values_only(fn), dom, f_tex

    # Case 2: generic callable
    if callable(f_item):
        dom = _extract_domain(f_item, prefer_2d=False)
        f_tex = getattr(f_item, "_f_tex", None)
        if not isinstance(f_tex, str):
            f_tex = None
        return _values_only(f_item), dom, f_tex

    raise ValueError(f"Unrecognized potential: {f_item!r}")
