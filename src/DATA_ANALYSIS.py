"""
Utilities for analyzing adiabatic spectra, avoided crossings, and wavefunction observables.

This module contains helper routines used in the QHD/adiabatic-analysis pipeline:
- peak localization metrics (peak position, FWHM);
- formatting utilities for value ± error strings;
- lightweight heuristics (barrier height estimation, completion criteria);
- construction of discrete Laplacian/potential operators on a Cartesian grid;
- adiabatic-basis ODE integration and local window refinement around avoided crossings.

Notes
-----
- Several functions assume a uniform Cartesian grid induced by `q` qubits per dimension.
- Array shape conventions are documented in each function's docstring.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Callable, Dict, List

import numpy as np
from scipy.sparse import csr_matrix, diags, eye, kron, lil_matrix

from UTILITIES import ComputationalBasisIndex_to_SpatialCoordinates
from FUNCTION_ANALYSIS import find_local_minima_and_basins, hessian_eigvals_at_points
from EIGENSOLVER import diagonalize


# =============================================================================
# Peak metrics: peak location, peak value, and FWHM along one axis
# =============================================================================

def _fwhm_1d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the full width at half maximum (FWHM) of a unimodal peak y(x) on a 1D grid.

    Parameters
    ----------
    x:
        1D array of coordinates (monotonicity is not required; the routine uses indices).
    y:
        1D array of sampled values (same length as `x`).

    Returns
    -------
    float
        The FWHM. Returns NaN if inputs are invalid or the peak is not well-defined
        (e.g., non-finite maximum, non-positive maximum, or non-positive width).

    Notes
    -----
    This function uses simple linear interpolation at the half-maximum crossing points.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 3:
        return float("nan")

    idx_max = int(np.nanargmax(y))
    y_max = float(y[idx_max])
    if not np.isfinite(y_max) or y_max <= 0.0:
        return float("nan")

    half = 0.5 * y_max

    # --- Left half-maximum crossing ---
    i = idx_max
    while i > 0 and y[i] >= half:
        i -= 1

    if y[i] >= half:
        x_left = x[0]
    else:
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]
        if y1 == y0:
            x_left = x0
        else:
            frac = (half - y0) / (y1 - y0)
            x_left = x0 + frac * (x1 - x0)

    # --- Right half-maximum crossing ---
    i = idx_max
    n = x.size
    while i < n - 1 and y[i] >= half:
        i += 1

    # NOTE: retained as-is to preserve behavior (legacy clamp logic).
    if y[i] >= n:
        i = n - 1

    if y[i] >= half:
        x_right = x[-1]
    else:
        x0, x1 = x[i - 1], x[i]
        y0, y1 = y[i - 1], y[i]
        if y1 == y0:
            x_right = x1
        else:
            frac = (half - y0) / (y1 - y0)
            x_right = x0 + frac * (x1 - x0)

    width = float(x_right - x_left)
    if not np.isfinite(width) or width <= 0.0:
        return float("nan")
    return width


def series_max_probability(density: np.ndarray) -> np.ndarray:
    """
    Compute the time-series of the maximum probability density.

    Parameters
    ----------
    density:
        Array of shape (Nt, M) containing probability densities |psi(t)|^2 sampled
        over the spatial grid (flattened index of length M).

    Returns
    -------
    np.ndarray
        Array of shape (Nt,) containing max_x |psi(t, x)|^2 for each time.
    """
    return np.asarray(density).max(axis=1)


def peak_position_and_fwhm(
    coords: np.ndarray,
    density_final: np.ndarray,
    *,
    axis: int = 0,
    atol: float = 1e-12,
) -> tuple[np.ndarray, float, float]:
    """
    Extract peak position, peak value, and 1D FWHM along a chosen axis.

    Parameters
    ----------
    coords:
        Array of shape (M, d) with Cartesian grid coordinates (uniform grid assumed).
    density_final:
        Flattened density at the final time, array-like of shape (M,).
    axis:
        Axis along which the 1D line cut is taken (only used if d > 1).
    atol:
        Absolute tolerance for identifying points lying on the same line cut
        (matching all coordinates except the chosen axis).

    Returns
    -------
    (coord_peak, fwhm_axis, peak_val):
        coord_peak:
            Array of shape (d,) with the coordinates of the global maximum.
        fwhm_axis:
            FWHM computed on the 1D section along `axis` passing through the peak.
            Returns NaN if the line cut is ill-defined.
        peak_val:
            Maximum density value at the peak location.
    """
    coords = np.asarray(coords, float)
    dens = np.asarray(density_final, float).reshape(-1)

    if coords.ndim != 2:
        raise ValueError("coords must have shape (M, d).")
    M, d = coords.shape
    if dens.shape[0] != M:
        raise ValueError("density_final must have length M.")

    idx_peak = int(np.nanargmax(dens))
    coord_peak = coords[idx_peak].copy()
    peak_val = float(dens[idx_peak])

    # Build a 1D line cut through the peak.
    if d == 1:
        x_line = coords[:, 0]
        dens_line = dens
    else:
        mask = np.ones(M, dtype=bool)
        for j in range(d):
            if j == axis:
                continue
            mask &= np.isclose(coords[:, j], coord_peak[j], atol=atol)

        x_line = coords[mask, axis]
        dens_line = dens[mask]

    if x_line.size < 2:
        return coord_peak, float("nan"), peak_val

    # Sort the line cut by coordinate to make the FWHM meaningful.
    order = np.argsort(x_line)
    x_line = x_line[order]
    dens_line = dens_line[order]

    fwhm = _fwhm_1d(x_line, dens_line)
    return coord_peak, fwhm, peak_val


# =============================================================================
# Formatting helpers: value ± error
# =============================================================================

def _format_value_with_error_scientific(value: float, error: float) -> str:
    """
    Format (value ± error) using scientific notation suitable for LaTeX.

    Output format (approximately):
        $(v \\pm \\sigma)\\times 10^{n}$

    - The error is displayed with 2 significant digits.
    - The value is rounded to the same decimal place as the error.

    Parameters
    ----------
    value:
        Central estimate.
    error:
        Standard uncertainty (must be positive to use the specialized formatting).

    Returns
    -------
    str
        LaTeX string representing the formatted quantity.
    """
    if not (np.isfinite(value) and np.isfinite(error)) or error <= 0.0:
        return rf"${value:.3g} \pm {error:.2g}$"

    # Exponent chosen from the magnitude of the measurement (or error if value == 0).
    if value == 0.0:
        exp = int(math.floor(math.log10(abs(error))))
    else:
        exp = int(math.floor(math.log10(abs(value))))

    scale = 10.0 ** (-exp)
    v_scaled = value * scale
    e_scaled = error * scale

    e_abs = abs(e_scaled)
    exp_err_scaled = int(math.floor(math.log10(e_abs)))
    n_dec = max(0, 1 - exp_err_scaled)  # 2 significant digits for the error

    fmt = f"{{:.{n_dec}f}}"
    v_str = fmt.format(v_scaled)
    e_str = fmt.format(e_scaled)

    return rf"$({v_str} \pm {e_str})\times 10^{{{exp}}}$"


def _format_value_with_error_plain(value: float, error: float) -> str:
    """
    Format value ± error in decimal notation.

    - The error is displayed with 2 significant digits.
    - The value is rounded to the same least significant digit as the error.

    Parameters
    ----------
    value:
        Central estimate.
    error:
        Standard uncertainty.

    Returns
    -------
    str
        Human-readable string like "1.23 ± 0.04".
    """
    if not (np.isfinite(value) and np.isfinite(error)) or error <= 0.0:
        return f"{value:.3g}"

    e_abs = abs(error)
    if e_abs == 0.0:
        return f"{value:.3g}"

    exp_err = int(math.floor(math.log10(e_abs)))
    digits = 2  # two significant digits for the error

    # Rounding exponent: keep `digits` significant digits in the error.
    round_exp = exp_err - (digits - 1)
    factor = 10.0 ** round_exp
    e_rounded = round(error / factor) * factor

    # Number of decimal places needed to display e_rounded.
    n_dec = max(0, -round_exp)

    fmt = f"{{:.{n_dec}f}}"
    v_str = fmt.format(value)
    e_str = fmt.format(e_rounded)
    return f"{v_str} ± {e_str}"


def format_peak_position_vector(coord: np.ndarray, errors: np.ndarray) -> str:
    """
    Format a coordinate vector with component-wise uncertainties.

    Example output:
        "(x ± Δx, y ± Δy, ...)"  for d > 1
        "x ± Δx"                 for d = 1

    Each uncertainty is printed with 2 significant digits, and each coordinate is
    rounded to the same least significant digit as its corresponding uncertainty.

    Parameters
    ----------
    coord:
        Array of shape (d,) containing the coordinate vector.
    errors:
        Array of shape (d,) containing uncertainties for each component.

    Returns
    -------
    str
        Formatted string.
    """
    coord = np.asarray(coord, float).reshape(-1)
    errors = np.asarray(errors, float).reshape(-1)
    if coord.size != errors.size:
        raise ValueError("coord and errors must have the same length.")

    parts = [_format_value_with_error_plain(v, e) for v, e in zip(coord, errors)]
    if coord.size == 1:
        return parts[0]
    return "(" + ", ".join(parts) + ")"


# =============================================================================
# Landscape heuristics: barrier estimation and completion criterion
# =============================================================================

def _estimate_barrier_height(
    out_min: Dict[str, Any] | None,
    f_vals_flat: np.ndarray,
    d: int,
    N: int,
) -> float | None:
    """
    Estimate a barrier height ΔV = V_barrier - V_min from basin information.

    The barrier is approximated as the minimum potential value among the grid points
    adjacent (face-neighbors) to the union of global basins.

    Parameters
    ----------
    out_min:
        Output dictionary from `find_local_minima_and_basins` (or compatible).
        If None, the routine returns None.
    f_vals_flat:
        Flattened potential values on the grid, shape (M,).
    d:
        Spatial dimension (currently supports d=1 or d=2 for adjacency-based estimation).
    N:
        Grid points per dimension, N = 2**q.

    Returns
    -------
    float | None
        Positive barrier height if identified, otherwise None.
    """
    if out_min is None:
        return None

    v = np.asarray(f_vals_flat, float).reshape(-1)
    global_min_val = float(out_min.get("global_min_val", np.nan))
    if not np.isfinite(global_min_val):
        return None

    if d == 1:
        basin_1d = np.asarray(
            out_min.get("basin_1d", out_min.get("global_basin_idx_flat", [])),
            dtype=int,
        )
        if basin_1d.size == 0:
            return None

        mask = np.zeros(N, dtype=bool)
        mask[basin_1d] = True

        barrier_vals: list[float] = []
        for i in range(N):
            if mask[i]:
                continue
            if (i > 0 and mask[i - 1]) or (i < N - 1 and mask[i + 1]):
                barrier_vals.append(v[i])

        if not barrier_vals:
            return None

        barrier_val = float(np.min(barrier_vals))
        dv = barrier_val - global_min_val
        return dv if dv > 0.0 else None

    if d == 2:
        basin_mask_flat = np.asarray(out_min.get("global_basin_idx_flat", []), int)
        if basin_mask_flat.size == 0:
            return None

        mask = np.zeros(N * N, dtype=bool)
        mask[basin_mask_flat] = True
        mask2d = mask.reshape(N, N, order="C")
        f2d = v.reshape(N, N, order="C")

        barrier_vals = []
        for i in range(N):
            for j in range(N):
                if mask2d[i, j]:
                    continue
                neighbors = []
                if i > 0:
                    neighbors.append(mask2d[i - 1, j])
                if i < N - 1:
                    neighbors.append(mask2d[i + 1, j])
                if j > 0:
                    neighbors.append(mask2d[i, j - 1])
                if j < N - 1:
                    neighbors.append(mask2d[i, j + 1])
                if any(neighbors):
                    barrier_vals.append(f2d[i, j])

        if not barrier_vals:
            return None

        barrier_val = float(np.min(barrier_vals))
        dv = barrier_val - global_min_val
        return dv if dv > 0.0 else None

    return None


def _simulation_completed(
    meta: Dict[str, Any],
    out_min: Dict[str, Any] | None,
    f_vals_flat: np.ndarray,
    a_fun: Callable[[float], float],
    b_fun: Callable[[float], float],
) -> bool:
    """
    Heuristic completion criterion for the annealing/schedule.

    Implements:
    - If a barrier estimate exists:
        |b(T) * (V_barrier - V_min)|  >>  |a(T) / L^2|
    - Otherwise:
        |b(T)|  >>  |a(T)|

    Here ">>" is operationally interpreted as a factor of ~1e4.

    Parameters
    ----------
    meta:
        Metadata dictionary containing at least "T", "d", "q", and "domain".
    out_min:
        Output dictionary from minima/basin analysis, or None.
    f_vals_flat:
        Flattened potential values (used for barrier estimation).
    a_fun, b_fun:
        Schedule functions a(t), b(t).

    Returns
    -------
    bool
        True if the criterion is satisfied, False otherwise.
    """
    T = float(meta.get("T", 1.0))
    d = int(meta.get("d"))
    q = int(meta.get("q"))
    domain = tuple(meta.get("domain"))
    x_min, x_max = float(domain[0]), float(domain[1])
    N = 2**q
    L = (x_max - x_min)

    aT = abs(float(a_fun(T)))
    bT = abs(float(b_fun(T)))
    if bT == 0.0:
        return False

    ratio_required = 1e4

    dv = _estimate_barrier_height(out_min, f_vals_flat, d, N)
    if dv is not None:
        lhs = abs(bT * dv)
        rhs = abs(aT / L**2)
        if rhs == 0.0:
            return lhs > 0.0
        return lhs >= ratio_required * rhs

    if aT == 0.0:
        return bT > 0.0
    return bT >= ratio_required * aT


# =============================================================================
# Minima + barrier positions (1D helper)
# =============================================================================

def _extract_minima_and_barriers_1d(
    coords: np.ndarray,
    values: np.ndarray,
    out_min: Dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract local minima, global minima, and barrier (local maxima) positions in 1D.

    Parameters
    ----------
    coords:
        Coordinates of shape (M, 1).
    values:
        Potential values of shape (M,).
    out_min:
        Optional precomputed output from `find_local_minima_and_basins`.

    Returns
    -------
    (min_positions, global_min_positions, barrier_positions):
        min_positions:
            All local minima positions (x) found.
        global_min_positions:
            Positions of global minima (subset of `min_positions`).
        barrier_positions:
            Positions of local maxima between adjacent global minima.
    """
    coords = np.asarray(coords, float)
    values = np.asarray(values, float).reshape(-1)

    if coords.ndim != 2 or coords.shape[1] != 1:
        return np.array([]), np.array([]), np.array([])

    try:
        out = out_min if out_min is not None else find_local_minima_and_basins(coords, values)
    except Exception:
        return np.array([]), np.array([]), np.array([])

    mins = np.asarray(out.get("min_coords", np.empty((0, 1))))
    if mins.size == 0:
        return np.array([]), np.array([]), np.array([])

    x_minima = mins[:, 0]
    global_min_indices = np.asarray(
        out.get("global_min_indices", [out.get("global_idx", 0)]),
        dtype=int,
    )
    x_global = x_minima[global_min_indices]

    x = coords[:, 0]
    v = values

    # Map minima coordinates to nearest grid indices.
    min_idx = [int(np.argmin(np.abs(x - xm))) for xm in x_minima]
    global_min_idx = [min_idx[g] for g in global_min_indices]
    global_min_idx = sorted(set(global_min_idx))

    barrier_positions: list[float] = []
    for i in range(len(global_min_idx) - 1):
        i1 = global_min_idx[i]
        i2 = global_min_idx[i + 1]
        if i2 <= i1 + 1:
            continue
        j_local = i1 + int(np.argmax(v[i1 : i2 + 1]))
        barrier_positions.append(float(x[j_local]))

    return x_minima, x_global, np.asarray(barrier_positions, float)


# =============================================================================
# HO/free-particle toy model support
# =============================================================================

def _enumerate_ho_multi_indices(freq_vec: np.ndarray, n_levels: int, max_n: int | None = None) -> np.ndarray:
    """
    Enumerate multi-indices (n1, ..., nd) for a d-dimensional harmonic oscillator.

    The enumeration is ordered by energy proportional to:
        sum_j freq_j * (n_j + 1/2)

    Parameters
    ----------
    freq_vec:
        Frequencies (or frequency-like weights), shape (d,).
    n_levels:
        Number of multi-indices to return.
    max_n:
        Maximum quantum number per dimension in the enumeration grid.
        If None, a conservative default is used.

    Returns
    -------
    np.ndarray
        Array of shape (n_levels, d) of integer multi-indices.
    """
    freq = np.asarray(freq_vec, float)
    dim = freq.size

    if max_n is None:
        max_n = max(4, n_levels)

    combos: list[tuple[float, np.ndarray]] = []
    for idx in itertools.product(range(max_n + 1), repeat=dim):
        n = np.array(idx, dtype=int)
        score = float(np.sum(freq * (n + 0.5)))
        combos.append((score, n))

    combos.sort(key=lambda x: x[0])
    res = [n for _, n in combos[:n_levels]]
    return np.asarray(res, dtype=int)


def _enumerate_free_multi_indices(dim: int, n_free: int, max_n: int | None = None) -> np.ndarray:
    """
    Enumerate multi-indices for a d-dimensional free particle.

    The enumeration is ordered by:
        |n|^2 = sum_j n_j^2

    Parameters
    ----------
    dim:
        Spatial dimension d.
    n_free:
        Number of indices to return.
    max_n:
        Maximum integer per dimension to consider.

    Returns
    -------
    np.ndarray
        Array of shape (n_free, dim) of integer multi-indices.
    """
    if max_n is None:
        max_n = max(4, n_free)

    combos: list[tuple[float, np.ndarray]] = []
    for idx in itertools.product(range(max_n + 1), repeat=dim):
        n = np.array(idx, dtype=int)
        score = float(np.sum(n**2))
        combos.append((score, n))

    combos.sort(key=lambda x: x[0])
    res = [n for _, n in combos[:n_free]]
    return np.asarray(res, dtype=int)


def _compute_model_params_for_ho(
    coords: np.ndarray,
    f_vals_flat: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    *,
    max_wells: int = 2,
    minima_out: Dict[str, Any] | None = None,
    hess_eigs_all: np.ndarray | None = None,
) -> Dict[str, Any]:
    """
    Extract toy-model parameters (local HO wells + free-particle levels) from f(x).

    Outputs include:
    - the first `max_wells` minima (sorted by x in 1D, by depth in d>1);
    - Hessian eigenvalues at those minima (dimension-agnostic);
    - an estimated barrier value (uses analysis output when available, otherwise heuristic).

    Parameters
    ----------
    coords:
        Grid coordinates, shape (M, d) (or (M,) interpreted as 1D).
    f_vals_flat:
        Potential values on the grid (flattened), shape (M,).
    f:
        Callable potential function f(coords)->values used to compute Hessian eigenvalues.
    max_wells:
        Maximum number of minima ("wells") to keep.
    minima_out:
        Optional output from `find_local_minima_and_basins`.
    hess_eigs_all:
        Optional precomputed Hessian eigenvalues at all minima (aligned with minima_out).

    Returns
    -------
    dict
        Keys:
            dim, min_coords, min_vals, hess_eigs, barrier_val
    """
    coords = np.asarray(coords, float)
    f_vals_flat = np.asarray(f_vals_flat, float).reshape(-1)

    if coords.ndim != 2:
        coords = coords.reshape(-1, 1)
    _, dim = coords.shape

    out = minima_out if minima_out is not None else find_local_minima_and_basins(coords, f_vals_flat)

    min_coords = np.asarray(out.get("min_coords", []), float)
    min_vals = np.asarray(out.get("min_vals", []), float).reshape(-1)

    if min_coords.size == 0:
        # Fallback: use global minimum on the grid.
        idx = int(np.argmin(f_vals_flat))
        min_coords = coords[[idx]]
        min_vals = np.array([float(f_vals_flat[idx])])
        hess_all = hessian_eigvals_at_points(f, min_coords)
    else:
        if hess_eigs_all is None:
            hess_all = hessian_eigvals_at_points(f, min_coords)
        else:
            hess_all = np.asarray(hess_eigs_all, float)

    hess_all = np.asarray(hess_all, float)
    if hess_all.ndim == 1:
        hess_all = hess_all[:, None]  # (n_min, 1) for d=1

    if hess_all.shape[0] != min_coords.shape[0]:
        # If mismatch, recompute to ensure alignment.
        hess_all = np.asarray(hessian_eigvals_at_points(f, min_coords), float)
        if hess_all.ndim == 1:
            hess_all = hess_all[:, None]

    # Sorting:
    # - 1D: left-to-right
    # - d>1: by depth (min_vals)
    if dim == 1:
        order = np.argsort(min_coords.reshape(-1))
    else:
        order = np.argsort(min_vals)

    min_coords = min_coords[order]
    min_vals = min_vals[order]
    hess_all = hess_all[order]

    n_wells = min(int(max_wells), min_coords.shape[0])
    min_coords = min_coords[:n_wells]
    min_vals = min_vals[:n_wells]
    hess_eigs = hess_all[:n_wells]

    # Prefer barrier estimate from analysis output if available.
    barrier_val = out.get("barrier_val", None)
    if barrier_val is not None:
        try:
            barrier_val = float(barrier_val)
        except (TypeError, ValueError):
            barrier_val = None

    # Backward-compatible keys / alternative formats.
    if barrier_val is None or not np.isfinite(barrier_val):
        for key in (
            "barrier", "saddle_val", "saddle",
            "barrier_height", "barrier_vals", "saddle_vals",
        ):
            if key in out:
                val = out[key]
                if val is None:
                    continue
                if np.isscalar(val):
                    barrier_val = float(val)
                else:
                    arr = np.asarray(val, float)
                    if arr.size > 0:
                        barrier_val = float(np.max(arr))
                break

    # Final heuristic fallback.
    if barrier_val is None or not np.isfinite(barrier_val):
        if f_vals_flat.size > 10:
            high = float(np.percentile(f_vals_flat, 95))
        else:
            high = float(np.max(f_vals_flat))
        barrier_val = max(high, float(np.max(min_vals)))

    return dict(
        dim=dim,
        min_coords=min_coords,    # (n_wells, dim)
        min_vals=min_vals,        # (n_wells,)
        hess_eigs=hess_eigs,      # (n_wells, dim)
        barrier_val=barrier_val,  # scalar
    )


def _build_ho_and_free_energies(
    L: float,
    tgrid: np.ndarray,
    a_fun: Callable[[float], float],
    b_fun: Callable[[float], float],
    min_vals: np.ndarray,
    hess_eigs: np.ndarray,
    barrier_val: float,
    n_levels_ho: int = 6,
    bc: str = "periodic",
) -> Dict[str, Any]:
    """
    Build toy-model energies: local HO levels per well + free-particle levels.

    For each well i:
        E_{i,k}(t) = b(t) * V_min(i) + sqrt(|a(t) b(t)|) * S_k
    where S_k are HO multi-index energies constructed from Hessian eigenvalues.

    For the free-particle sector:
        E_free,n(t) ~ b(t) * base_level + coef(bc, L) * a(t) * |n|^2

    Parameters
    ----------
    L:
        Domain length (x_max - x_min).
    tgrid:
        Time grid, shape (Nt,).
    a_fun, b_fun:
        Schedule functions.
    min_vals:
        Well minima values, shape (n_wells,).
    hess_eigs:
        Hessian eigenvalues at minima, shape (n_wells, d) (or (n_wells,) for d=1).
    barrier_val:
        Representative barrier potential value (scalar).
    n_levels_ho:
        Number of HO levels per well.
    bc:
        "periodic" or "dirichlet". Affects the free-particle spectrum coefficient.

    Returns
    -------
    dict
        Keys:
            wells_E: list of (Nt, n_levels_ho) arrays, one per well
            free_E : (Nt, n_free) array
            Vbar_t : (Nt,) array
            bc     : normalized boundary condition label
    """
    tgrid = np.asarray(tgrid, float)
    Nt = tgrid.size

    a_vals = np.array([a_fun(float(t)) for t in tgrid], float)
    b_vals = np.array([b_fun(float(t)) for t in tgrid], float)

    # Hybrid factor controlling local HO energies.
    root = np.sqrt(np.abs(a_vals) * np.abs(b_vals))

    min_vals = np.asarray(min_vals, float).reshape(-1)
    hess_eigs = np.asarray(hess_eigs, float)
    if hess_eigs.ndim == 1:
        hess_eigs = hess_eigs[:, None]  # (n_wells, 1) for d=1

    n_wells, dim = hess_eigs.shape
    if min_vals.size != n_wells:
        raise ValueError("min_vals and hess_eigs have incompatible number of wells.")

    bc = bc.lower()
    if bc not in ("periodic", "dirichlet"):
        raise ValueError("bc must be 'periodic' or 'dirichlet'.")

    # --- Local HO levels per well ---
    wells_E: List[np.ndarray] = []
    n_levels_ho = int(n_levels_ho)

    for i in range(n_wells):
        Vmin = float(min_vals[i])
        eigs = np.array(hess_eigs[i], float)

        # Clean up potentially noisy eigenvalues.
        eigs[~np.isfinite(eigs)] = 0.0
        eigs[eigs < 0.0] = 0.0
        freq = np.sqrt(np.maximum(eigs, 0.0))

        if not np.any(freq > 0.0):
            # If Hessian is effectively zero, use a dummy isotropic frequency.
            freq[:] = 1.0

        n_multi = _enumerate_ho_multi_indices(freq, n_levels_ho)
        S = (n_multi + 0.5) @ freq  # (n_levels_ho,)

        E = np.empty((Nt, n_levels_ho), float)
        for k in range(n_levels_ho):
            E[:, k] = b_vals * Vmin + root * S[k]
        wells_E.append(E)

    # --- Free-particle levels in d dimensions ---
    if bc == "periodic":
        n_free = max(1, (n_levels_ho + 1) // 2)
    else:
        n_free = n_levels_ho

    n_multi_free = _enumerate_free_multi_indices(dim, n_free)  # (n_free, dim)

    # Reference energy level: blend minima and barrier (heuristic).
    base_level = (min_vals[0] + min_vals[1] + 2.0 * barrier_val) / 4.0

    # Boundary-condition-dependent prefactor:
    # Dirichlet:  (pi^2 / 2) * a(t) * |n|^2 / L^2
    # Periodic :  factor 4 larger than Dirichlet in this convention
    if bc == "dirichlet":
        coef = (math.pi**2) / (2.0 * L**2)
    else:
        coef = (4 * math.pi**2) / (2.0 * L**2)

    free_E = np.empty((Nt, n_free), float)
    for i in range(n_free):
        factor = float(np.sum(n_multi_free[i] ** 2))  # |n|^2
        free_E[:, i] = base_level * b_vals + coef * a_vals * factor

    Vbar_t = barrier_val * b_vals
    return dict(wells_E=wells_E, free_E=free_E, Vbar_t=Vbar_t, bc=bc)


# =============================================================================
# Eigenvector utilities: ensure shape, phase alignment, derivatives, integrals
# =============================================================================

def ensure_vecs_t_x_k(vecs: np.ndarray, n_states: int) -> np.ndarray:
    """
    Normalize eigenvector tensor shape to (Nt, Nx, K).

    Parameters
    ----------
    vecs:
        3D tensor containing eigenvectors, with one axis equal to n_states.
        Accepted permutations include:
          - (Nt, Nx, K)
          - (Nt, K, Nx)
          - (K, Nt, Nx)
          - any permutation containing n_states as one axis
    n_states:
        Target number of eigenstates K.

    Returns
    -------
    np.ndarray
        Tensor with shape (Nt, Nx, K).

    Raises
    ------
    ValueError
        If the tensor cannot be mapped to (Nt, Nx, K).
    """
    shp = vecs.shape
    if len(shp) != 3:
        raise ValueError(f"Expected a 3D tensor for eigenvectors; got shape={shp}")

    if shp[2] == n_states:  # (Nt, Nx, K)
        return vecs
    if shp[1] == n_states:  # (Nt, K, Nx)
        return np.transpose(vecs, (0, 2, 1))
    if shp[0] == n_states:  # (K, Nt, Nx)
        return np.transpose(vecs, (1, 2, 0))

    if n_states in shp:
        k_ax = shp.index(n_states)
        axes = [a for a in range(3) if a != k_ax] + [k_ax]
        out = np.transpose(vecs, axes)
        if out.shape[2] == n_states:
            return out

    raise ValueError(f"Cannot map eigenvectors to (Nt, Nx, K); shape={shp}")


def phase_align_no_perm(vecs_txk: np.ndarray) -> np.ndarray:
    """
    Stabilize global phases over time without permuting eigenvectors.

    For each k, impose that <psi_k(t-1) | psi_k(t)> is real and positive by applying
    a time-dependent global phase to psi_k(t).

    Parameters
    ----------
    vecs_txk:
        Eigenvectors with shape (Nt, Nx, K).

    Returns
    -------
    np.ndarray
        Phase-aligned eigenvectors with the same shape.
    """
    Nt, Nx, K = vecs_txk.shape
    aligned = np.empty_like(vecs_txk, dtype=complex)
    aligned[0] = vecs_txk[0]

    eps = 1e-14
    for t in range(1, Nt):
        prev = aligned[t - 1]      # (Nx, K)
        curr = vecs_txk[t]         # (Nx, K)
        fixed = np.empty_like(curr, dtype=complex)
        for k in range(K):
            s = np.vdot(prev[:, k], curr[:, k])
            phase = 0.0 if (np.abs(s) < eps) else -np.angle(s)
            fixed[:, k] = curr[:, k] * np.exp(1j * phase)
        aligned[t] = fixed
    return aligned


def time_derivative(arr_t_x_k: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Compute time derivative along axis=0 using `np.gradient` (edge_order=2).

    Parameters
    ----------
    arr_t_x_k:
        Array with time as the first axis (e.g., (Nt, Nx, K)).
    times:
        Time grid, shape (Nt,).

    Returns
    -------
    np.ndarray
        Time derivative with the same shape as `arr_t_x_k`.
    """
    return np.gradient(arr_t_x_k, times, axis=0, edge_order=2)


def cumulative_trapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Cumulative trapezoidal integral with out[0]=0 and len(out)=len(x)=len(y).

    Parameters
    ----------
    y:
        Samples of the integrand, shape (N,).
    x:
        Monotone grid, shape (N,).

    Returns
    -------
    np.ndarray
        Cumulative integral array, shape (N,).
    """
    out = np.zeros_like(x, dtype=float)
    if len(x) > 1:
        dx = np.diff(x)
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    return out


def contiguous_segment(mask: np.ndarray, idx_center: int) -> tuple[int, int]:
    """
    Return inclusive endpoints of the contiguous True-segment containing idx_center.

    Parameters
    ----------
    mask:
        Boolean array.
    idx_center:
        Index guaranteed (by caller) to be within the mask range.

    Returns
    -------
    (i0, i1):
        Inclusive endpoints of the connected segment.
    """
    i0 = i1 = idx_center
    N = mask.size
    while i0 - 1 >= 0 and mask[i0 - 1]:
        i0 -= 1
    while i1 + 1 < N and mask[i1 + 1]:
        i1 += 1
    return i0, i1


def _theta_trapz(times: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
    Compute theta(t) = ∫ E(t) dt for each level via trapezoidal rule.

    Parameters
    ----------
    times:
        Time grid, shape (Nt,).
    E:
        Energies, shape (Nt, K).

    Returns
    -------
    np.ndarray
        Theta values, shape (Nt, K), with theta[0]=0.
    """
    Nt, K = E.shape
    theta = np.zeros((Nt, K), dtype=np.float64)
    if Nt > 1:
        dtv = np.diff(times)
        theta[1:, :] = np.cumsum(0.5 * (E[1:, :] + E[:-1, :]) * dtv[:, None], axis=0)
    return theta


# =============================================================================
# Discrete operators: Laplacian and potential on the computational basis
# =============================================================================

def build_D2_V(d: int, q: int, domain, f, bc):
    """
    Build discrete Laplacian (D2) and diagonal potential operator (V) in the computational basis.

    The intended usage is for:
        Hdot(t) = 0.5 * a'(t) * D2 + b'(t) * V

    Parameters
    ----------
    d:
        Spatial dimension.
    q:
        Qubits per dimension (N = 2**q grid points per dimension).
    domain:
        Tuple (x_min, x_max).
    f:
        Potential function evaluated on coordinates: f(coords)->values.
    bc:
        Boundary condition label forwarded to the coordinate generator.

    Returns
    -------
    (D2, Vop):
        D2:
            Sparse CSR matrix of shape (M, M), where M=N**d.
        Vop:
            Sparse CSR diagonal matrix of shape (M, M).
    """
    N = 2**q
    M = N**d
    x_min, x_max = map(float, domain)
    L = x_max - x_min
    dx = L / N

    # 1D periodic second-difference operator.
    T1 = lil_matrix((N, N), dtype=np.float64)
    T1.setdiag(2.0)
    T1.setdiag(-1.0, k=+1)
    T1.setdiag(-1.0, k=-1)
    T1[0, -1] = -1.0
    T1[-1, 0] = -1.0
    T1 = (1.0 / dx**2) * T1.tocsr()

    I1 = eye(N, format="csr", dtype=np.float64)

    def kron_dim(A: csr_matrix, dim_index: int) -> csr_matrix:
        out = None
        for jj in reversed(range(d)):
            factor = A if jj == dim_index else I1
            out = factor if out is None else kron(out, factor, format="csr")
        return out

    # Sum over dimensions to get the full dD Laplacian.
    D2 = None
    for kdim in range(d):
        term = kron_dim(T1, kdim)
        D2 = term if D2 is None else (D2 + term)

    coords = ComputationalBasisIndex_to_SpatialCoordinates(q, d, domain, bc)
    Vdiag = np.asarray(f(coords), dtype=np.float64).reshape(M)
    Vop = diags(Vdiag, offsets=0, shape=(M, M), format="csr")
    return D2, Vop


# =============================================================================
# Adiabatic-basis ODE integration utilities
# =============================================================================

def integrate_c_global(
    times: np.ndarray,
    E: np.ndarray,
    U: np.ndarray,
    D2: csr_matrix,
    Vop: csr_matrix,
    a_fun,
    b_fun,
    K_use: int = 20,
) -> np.ndarray:
    """
    Integrate adiabatic-basis ODE on [0, T] using RK4 on the provided time grid.

    Uses:
        Hdot(t) = 0.5 * a'(t) * D2 + b'(t) * V
    and the standard adiabatic coupling:
        A_mn = <m|Hdot|n> / (E_n - E_m)   for n != m

    Parameters
    ----------
    times:
        Time grid, shape (Nt,).
    E:
        Energies, shape (Nt, K_all).
    U:
        Eigenvectors, shape (Nt, M, K_all), where M is Hilbert-space dimension.
    D2, Vop:
        Operators defining Hdot(t).
    a_fun, b_fun:
        Schedule functions.
    K_use:
        Number of eigenstates to include (truncation).

    Returns
    -------
    np.ndarray
        Coefficients c(t) in the adiabatic basis, shape (Nt, K), with c(0) = (1,0,...).
    """
    Nt, M, K_all = U.shape
    K = min(K_use, K_all)

    E = E[:, :K]
    U = U[:, :, :K]
    theta = _theta_trapz(times, E)

    a_vals = np.array([a_fun(float(t)) for t in times], float)
    b_vals = np.array([b_fun(float(t)) for t in times], float)
    adot = np.gradient(a_vals, times, edge_order=2)
    bdot = np.gradient(b_vals, times, edge_order=2)

    c = np.zeros((Nt, K), dtype=np.complex128)
    c[0, 0] = 1.0 + 0.0j
    eyeK = np.eye(K, dtype=bool)

    for t in range(Nt - 1):
        h = float(times[t + 1] - times[t])

        adot0 = adot[t]
        adot1 = adot[t + 1]
        bdot0 = bdot[t]
        bdot1 = bdot[t + 1]

        Hdot0 = 0.5 * adot0 * D2 + bdot0 * Vop
        Hdot1 = 0.5 * adot1 * D2 + bdot1 * Vop

        U0 = U[t]
        U1 = U[t + 1]
        Hsub0 = U0.conj().T @ (Hdot0.dot(U0))
        Hsub1 = U1.conj().T @ (Hdot1.dot(U1))

        gaps0 = (E[t][None, :] - E[t][:, None])
        gaps1 = (E[t + 1][None, :] - E[t + 1][:, None])

        # Robust small-gap tolerance based on spectral span.
        span0 = float(np.ptp(E[t])) if K > 0 else 1.0
        span1 = float(np.ptp(E[t + 1])) if K > 0 else 1.0
        tol0 = max(1e-12, 1e-9 * max(1.0, span0))
        tol1 = max(1e-12, 1e-9 * max(1.0, span1))

        A0 = np.zeros_like(Hsub0, dtype=np.complex128)
        A1 = np.zeros_like(Hsub1, dtype=np.complex128)
        mask0 = (~eyeK) & (np.abs(gaps0) >= tol0)
        mask1 = (~eyeK) & (np.abs(gaps1) >= tol1)
        A0[mask0] = Hsub0[mask0] / gaps0[mask0]
        A1[mask1] = Hsub1[mask1] / gaps1[mask1]

        phase0 = np.exp(1j * (theta[t][None, :] - theta[t][:, None]))
        phase1 = np.exp(1j * (theta[t + 1][None, :] - theta[t + 1][:, None]))
        M0 = A0 * phase0
        M1 = A1 * phase1
        Mh = 0.5 * (M0 + M1)

        # RK4 step.
        k1 = -M0 @ c[t]
        k2 = -Mh @ (c[t] + 0.5 * h * k1)
        k3 = -Mh @ (c[t] + 0.5 * h * k2)
        k4 = -M1 @ (c[t] + h * k3)
        c[t + 1] = c[t] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if not np.isfinite(c[t + 1]).all():
            c[t + 1] = np.nan_to_num(c[t + 1], nan=0.0, posinf=0.0, neginf=0.0)

    return c


def interp_c_at_time(times_full: np.ndarray, c_full: np.ndarray, t: float) -> np.ndarray:
    """
    Linearly interpolate complex coefficients c(t) on a given time grid.

    Parameters
    ----------
    times_full:
        Time grid, shape (Nt,).
    c_full:
        Coefficients, shape (Nt, K).
    t:
        Target time.

    Returns
    -------
    np.ndarray
        Interpolated coefficient vector, shape (K,).
    """
    if t <= times_full[0]:
        return c_full[0].copy()
    if t >= times_full[-1]:
        return c_full[-1].copy()

    idx = int(np.searchsorted(times_full, t) - 1)
    idx = max(0, min(idx, len(times_full) - 2))
    t0, t1 = times_full[idx], times_full[idx + 1]
    a = (t - t0) / (t1 - t0)
    return (1.0 - a) * c_full[idx] + a * c_full[idx + 1]


def integrate_window_with_init(
    times_win: np.ndarray,
    E_win: np.ndarray,
    U_win: np.ndarray,
    D2: csr_matrix,
    Vop: csr_matrix,
    a_fun,
    b_fun,
    c0: np.ndarray,
) -> np.ndarray:
    """
    RK4 integration on a sub-window [t1, t2] with initial condition c(t1)=c0.

    If c0 has fewer components than required, it is zero-padded; if it has more,
    it is truncated.

    Parameters
    ----------
    times_win:
        Window time grid, shape (Tn,).
    E_win:
        Energies over the window, shape (Tn, Kwin_total).
    U_win:
        Eigenvectors over the window, shape (Tn, M, Kwin_total).
    D2, Vop:
        Operators used to build Hdot(t).
    a_fun, b_fun:
        Schedule functions.
    c0:
        Initial coefficient vector at times_win[0], shape (K0,).

    Returns
    -------
    np.ndarray
        Window coefficients, shape (Tn, Kwin), where Kwin = U_win.shape[2].
    """
    Tn, M, Kwin = U_win.shape
    K0 = c0.size
    if K0 < Kwin:
        c_init = np.pad(c0, (0, Kwin - K0))
    else:
        c_init = c0[:Kwin]

    theta = _theta_trapz(times_win, E_win[:, :Kwin])
    eyeK = np.eye(Kwin, dtype=bool)

    a_vals = np.array([a_fun(float(t)) for t in times_win], float)
    b_vals = np.array([b_fun(float(t)) for t in times_win], float)
    adot = np.gradient(a_vals, times_win, edge_order=2)
    bdot = np.gradient(b_vals, times_win, edge_order=2)

    c = np.zeros((Tn, Kwin), dtype=np.complex128)
    c[0, :] = c_init

    for t in range(Tn - 1):
        h = float(times_win[t + 1] - times_win[t])

        adot0 = adot[t]
        adot1 = adot[t + 1]
        bdot0 = bdot[t]
        bdot1 = bdot[t + 1]

        Hdot0 = 0.5 * adot0 * D2 + bdot0 * Vop
        Hdot1 = 0.5 * adot1 * D2 + bdot1 * Vop

        U0 = U_win[t, :, :Kwin]
        U1 = U_win[t + 1, :, :Kwin]
        Hsub0 = U0.conj().T @ (Hdot0.dot(U0))
        Hsub1 = U1.conj().T @ (Hdot1.dot(U1))

        gaps0 = (E_win[t, :Kwin][None, :] - E_win[t, :Kwin][:, None])
        gaps1 = (E_win[t + 1, :Kwin][None, :] - E_win[t + 1, :Kwin][:, None])

        span0 = float(np.ptp(E_win[t, :Kwin])) if Kwin > 0 else 1.0
        span1 = float(np.ptp(E_win[t + 1, :Kwin])) if Kwin > 0 else 1.0
        tol0 = max(1e-12, 1e-9 * max(1.0, span0))
        tol1 = max(1e-12, 1e-9 * max(1.0, span1))

        A0 = np.zeros_like(Hsub0, dtype=np.complex128)
        A1 = np.zeros_like(Hsub1, dtype=np.complex128)
        mask0 = (~eyeK) & (np.abs(gaps0) >= tol0)
        mask1 = (~eyeK) & (np.abs(gaps1) >= tol1)
        A0[mask0] = Hsub0[mask0] / gaps0[mask0]
        A1[mask1] = Hsub1[mask1] / gaps1[mask1]

        phase0 = np.exp(1j * (theta[t][None, :] - theta[t][:, None]))
        phase1 = np.exp(1j * (theta[t + 1][None, :] - theta[t + 1][:, None]))
        M0 = A0 * phase0
        M1 = A1 * phase1
        Mh = 0.5 * (M0 + M1)

        k1 = -M0 @ c[t]
        k2 = -Mh @ (c[t] + 0.5 * h * k1)
        k3 = -Mh @ (c[t] + 0.5 * h * k2)
        k4 = -M1 @ (c[t] + h * k3)
        c[t + 1] = c[t] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if not np.isfinite(c[t + 1]).all():
            c[t + 1] = np.nan_to_num(c[t + 1], nan=0.0, posinf=0.0, neginf=0.0)

    return c


# =============================================================================
# Avoided crossing window refinement
# =============================================================================

def _clip_segment_to_width(
    times: np.ndarray,
    seg0: int,
    seg1: int,
    idx_center: int,
    max_width: float,
) -> tuple[int, int]:
    """
    Restrict a contiguous segment [seg0, seg1] to a subsegment with width <= max_width.

    Parameters
    ----------
    times:
        Time grid.
    seg0, seg1:
        Inclusive endpoints of the original segment.
    idx_center:
        Index around which the segment is centered.
    max_width:
        Maximum allowed time width.

    Returns
    -------
    (L, R):
        Inclusive endpoints of the clipped segment.
    """
    L = idx_center
    R = idx_center
    half = 0.5 * max_width

    while L - 1 >= seg0 and (times[idx_center] - times[L - 1] <= half):
        L -= 1
    while R + 1 <= seg1 and (times[R + 1] - times[idx_center] <= half):
        R += 1

    while times[R] - times[L] > max_width and R > L:
        if (times[idx_center] - times[L]) > (times[R] - times[idx_center]):
            L += 1
        else:
            R -= 1
    return L, R


def _segment_and_brackets(
    times: np.ndarray,
    dE: np.ndarray,
    level: float,
    idx_center: int,
) -> tuple[int, int, float, float]:
    """
    Identify the connected segment where dE <= level containing idx_center,
    and return the adjacent bracketing times just outside the segment.

    Parameters
    ----------
    times:
        Time grid, shape (Nt,).
    dE:
        Gap curve, shape (Nt,).
    level:
        Threshold level for "small gap" region.
    idx_center:
        Index inside the small-gap region.

    Returns
    -------
    (seg0, seg1, t1p, t2p):
        seg0, seg1:
            Inclusive segment endpoints.
        t1p, t2p:
            Times just outside the segment (or segment endpoints at boundaries).
    """
    mask = (dE <= level)
    seg0, seg1 = contiguous_segment(mask, idx_center)
    t1p = float(times[seg0 - 1]) if seg0 > 0 else float(times[seg0])
    t2p = float(times[seg1 + 1]) if seg1 < len(times) - 1 else float(times[seg1])
    return seg0, seg1, t1p, t2p


def _find_avoided_crossing_index(
    times: np.ndarray,
    Ei: np.ndarray,
    Ej: np.ndarray,
    edge_frac: float = 0.1,
    max_rel_width: float = 0.3,
    allowed_mask: np.ndarray | None = None,
) -> int:
    """
    Identify a "physical" avoided crossing index for two energy levels Ei(t), Ej(t).

    The selection enforces:
    - exclude an `edge_frac` portion of the time interval at both ends;
    - optionally restrict to `allowed_mask=True` points;
    - consider only local minima of ΔE(t) = |Ej - Ei|;
    - require opposite slopes at the candidate time (Ei' * Ej' < 0);
    - require the small-gap region to be localized (not too wide).

    Parameters
    ----------
    times:
        Time grid, shape (Nt,).
    Ei, Ej:
        Energy curves, shape (Nt,).
    edge_frac:
        Fraction of the time interval excluded at each edge (0 < edge_frac < 0.5).
    max_rel_width:
        Maximum allowed relative width (fraction of total span) of the small-gap region.
    allowed_mask:
        Optional boolean mask restricting admissible points (same shape as times).

    Returns
    -------
    int
        Index of the selected avoided crossing.
    """
    times = np.asarray(times, float)
    Ei = np.asarray(Ei, float)
    Ej = np.asarray(Ej, float)

    Nt = times.size
    if Nt < 3:
        return int(np.argmin(np.abs(Ej - Ei)))

    dE = np.abs(Ej - Ei)

    # 1) Exclude edges.
    t0, t1 = float(times[0]), float(times[-1])
    span = t1 - t0
    if span <= 0.0:
        core_mask = np.ones(Nt, dtype=bool)
    else:
        if 0.0 < edge_frac < 0.5:
            tL = t0 + edge_frac * span
            tR = t1 - edge_frac * span
            core_mask = (times >= tL) & (times <= tR)
        else:
            core_mask = np.ones(Nt, dtype=bool)

    # Additional admissibility mask (e.g., t >= t_min_allowed).
    if allowed_mask is not None:
        allowed_mask = np.asarray(allowed_mask, bool)
        if allowed_mask.shape != core_mask.shape:
            raise ValueError("allowed_mask must have the same shape as times.")
        core_mask &= allowed_mask

    # 2) Derivatives of levels and (approximate) curvature of the gap.
    dEi = np.gradient(Ei, times, edge_order=2)
    dEj = np.gradient(Ej, times, edge_order=2)
    d2dE = np.gradient(np.gradient(dE, times, edge_order=2), times, edge_order=2)

    candidates: list[int] = []
    for k in range(1, Nt - 1):
        if not core_mask[k]:
            continue
        if not (np.isfinite(dE[k]) and np.isfinite(dEi[k]) and np.isfinite(dEj[k])):
            continue

        # Local minimum of ΔE.
        if not (dE[k] <= dE[k - 1] and dE[k] <= dE[k + 1]):
            continue

        # Positive curvature to avoid flat plateaus.
        if not np.isfinite(d2dE[k]) or d2dE[k] <= 0.0:
            continue

        # Opposite slopes (approach + repel).
        if dEi[k] * dEj[k] >= 0.0:
            continue

        # 3) Localized small-gap region width constraint.
        level = 1.5 * dE[k]
        mask_small = (dE <= level)
        seg0, seg1 = contiguous_segment(mask_small, k)
        width = float(times[seg1] - times[seg0])

        if span > 0.0 and width > max_rel_width * span:
            continue

        candidates.append(k)

    if not candidates:
        # Fallback: global minimum gap over the admissible region.
        mask = core_mask & np.isfinite(dE)
        if np.any(mask):
            idxs = np.where(mask)[0]
            return int(idxs[np.argmin(dE[idxs])])

        finite = np.isfinite(dE)
        if np.any(finite):
            idxs = np.where(finite)[0]
            return int(idxs[np.argmin(dE[idxs])])

        return int(np.nanargmin(dE))

    return int(min(candidates, key=lambda kk: dE[kk]))


EDGE_FRAC: float = 0.15
REFINED_PTS: int = 100
MIN_POINTS_IN_WINDOW: int = 50
MAX_REF_ITERS: int = 10
LEVEL_FACTOR: float = 3.0
MAX_INIT_SPAN: float = 1.0


def refine_pair_window_and_analyze(
    i: int,
    j: int,
    *,
    d: int,
    q: int,
    times_coarse: np.ndarray,
    E_coarse: np.ndarray,
    V_coarse_aligned: np.ndarray,
    f,
    domain,
    n_states: int,
    xgrid: np.ndarray,
    edge_frac: float = EDGE_FRAC,
    refined_pts: int = REFINED_PTS,
    min_points_in_window: int = MIN_POINTS_IN_WINDOW,
    max_iters: int = MAX_REF_ITERS,
    times_full: np.ndarray | None = None,
    c_global: np.ndarray | None = None,
    D2_op: csr_matrix | None = None,
    V_op: csr_matrix | None = None,
    a_fun=None,
    b_fun=None,
    t_min_allowed: float | None = None,
) -> Dict[str, Any]:
    """
    Iteratively refine (zoom) the time window around an avoided crossing between levels i and j.

    High-level algorithm
    --------------------
    1) Compute ΔE(t) on the coarse grid.
    2) Exclude an `edge_frac` portion at both ends from the crossing search.
    3) Identify an initial small-gap segment where ΔE(t) <= LEVEL_FACTOR * ΔE*.
    4) Iteratively zoom by diagonalizing on a refined time grid over a bracket [t1', t2'].
    5) Stop once the small-gap segment has at least `min_points_in_window` points or
       after `max_iters` iterations.

    Optional features
    -----------------
    If `times_full`, `c_global`, `D2_op`, and `V_op` are provided, a local adiabatic-basis
    integration is performed in the refined window to estimate |c_i(t)|^2 and |c_j(t)|^2.

    Parameters
    ----------
    i, j:
        Indices of the two levels.
    d, q:
        Spatial dimension and qubits per dimension.
    times_coarse:
        Coarse time grid, shape (Nt,).
    E_coarse:
        Coarse energies, shape (Nt, K).
    V_coarse_aligned:
        Coarse eigenvectors (phase-aligned), kept for API compatibility.
        (Not used directly by the refinement; retained as in the original code.)
    f, domain:
        Potential and domain passed to the eigensolver.
    n_states:
        Number of eigenstates returned by diagonalization.
    xgrid:
        Spatial grid (kept for API compatibility in this routine).
    edge_frac, refined_pts, min_points_in_window, max_iters:
        Refinement controls.
    times_full, c_global, D2_op, V_op:
        Optional global integration data to seed a local integration.
    a_fun, b_fun:
        Schedule functions (required).
    t_min_allowed:
        Optional constraint to disallow searching before a given time.

    Returns
    -------
    Dict[str, Any]
        Payload containing selected window arrays and diagnostics:
        - times_sel, psi_i_sel, psi_j_sel
        - dE_sel, dH_sel, B_sel
        - t_star, DeltaE_star
        - max_dH, area_B
        - t1, t2, t1p, t2p, pts_under
        - ci2_sel, cj2_sel (if local integration was performed)
    """
    if a_fun is None or b_fun is None:
        raise ValueError("refine_pair_window_and_analyze requires non-null a_fun and b_fun.")

    times = np.asarray(times_coarse, float)
    dE_coarse = np.asarray(E_coarse[:, j] - E_coarse[:, i], float)
    Nt = times.size

    # --- Define the "core" time window excluding edges. ---
    if Nt > 1 and 0.0 < edge_frac < 0.5:
        span_t = float(times[-1] - times[0])
        if span_t > 0.0:
            t_core_min = float(times[0] + edge_frac * span_t)
            t_core_max = float(times[-1] - edge_frac * span_t)
            core_mask_all = (times >= t_core_min) & (times <= t_core_max)
        else:
            t_core_min = float(times[0])
            t_core_max = float(times[-1])
            core_mask_all = np.ones_like(times, dtype=bool)
    else:
        t_core_min = float(times[0])
        t_core_max = float(times[-1])
        core_mask_all = np.ones_like(times, dtype=bool)

    # --- Additional constraint: do not search before t_min_allowed. ---
    if t_min_allowed is not None:
        t_min_allowed = float(t_min_allowed)
        t_core_min = max(t_core_min, t_min_allowed)
        core_mask_all &= (times >= t_min_allowed)

    # --- Safety: ensure we have finite values in the admissible region. ---
    finite_mask_all = np.isfinite(dE_coarse)
    finite_mask = finite_mask_all & core_mask_all

    if not np.any(finite_mask):
        if t_min_allowed is not None:
            finite_mask = finite_mask_all & (times >= t_min_allowed)
        if not np.any(finite_mask):
            finite_mask = finite_mask_all

    if not np.any(finite_mask):
        raise ValueError(f"dE_coarse is all NaN/inf for pair ({i}, {j}).")

    Ei_coarse = E_coarse[:, i]
    Ej_coarse = E_coarse[:, j]

    idx_star = _find_avoided_crossing_index(
        times,
        Ei_coarse,
        Ej_coarse,
        edge_frac=edge_frac,
        max_rel_width=0.3,
        allowed_mask=finite_mask,
    )

    DeltaE_star = float(dE_coarse[idx_star])
    level = LEVEL_FACTOR * DeltaE_star

    seg0, seg1 = contiguous_segment(dE_coarse <= level, idx_star)
    if times[seg1] - times[seg0] > MAX_INIT_SPAN:
        seg0, seg1 = _clip_segment_to_width(times, seg0, seg1, idx_star, MAX_INIT_SPAN)

    t1 = float(times[seg0])
    t2 = float(times[seg1])
    t1p = float(times[seg0 - 1]) if seg0 > 0 else t1
    t2p = float(times[seg1 + 1]) if seg1 < Nt - 1 else t2

    last_payload: Dict[str, Any] | None = None
    it = 0

    while True:
        # --- Choose refinement window [tL, tR]. ---
        tL, tR = (t1p, t2p) if t2p > t1p else (t1, t2)
        times_ref = np.linspace(tL, tR, refined_pts, endpoint=True)

        # --- Refined diagonalization. ---
        E_ref, V_ref = diagonalize(
            d=d, q=q, times=times_ref,
            a_fun=a_fun, b_fun=b_fun,
            n_states=n_states, f=f, domain=domain
        )
        V_ref = ensure_vecs_t_x_k(V_ref, n_states=n_states).astype(complex, copy=False)
        V_ref = phase_align_no_perm(V_ref)
        dV_ref = time_derivative(V_ref, times_ref)

        dE_ref = E_ref[:, j] - E_ref[:, i]

        # --- Find refined minimum gap, restricted to core window when possible. ---
        core_ref_mask = (times_ref >= t_core_min) & (times_ref <= t_core_max)
        finite_ref = np.isfinite(dE_ref)
        valid_ref = core_ref_mask & finite_ref

        if np.any(valid_ref):
            dE_tmp = np.where(valid_ref, dE_ref, np.inf)
            idx_star_ref = int(np.argmin(dE_tmp))
        else:
            idx_star_ref = int(np.nanargmin(dE_ref))

        t_star_ref = float(times_ref[idx_star_ref])
        DeltaE_star = float(dE_ref[idx_star_ref])
        level = LEVEL_FACTOR * DeltaE_star

        seg0_r, seg1_r, t1p_new, t2p_new = _segment_and_brackets(
            times_ref, dE_ref, level, idx_star_ref
        )
        pts_under = int(seg1_r - seg0_r + 1)

        # Update [t1, t2] and brackets [t1', t2'].
        t1 = float(times_ref[seg0_r])
        t2 = float(times_ref[seg1_r])
        t1p, t2p = t1p_new, t2p_new

        # Slice selection for "small gap" region.
        sl = slice(seg0_r, seg1_r + 1)
        t_sel = times_ref[sl]
        psi_i_s = V_ref[sl, :, i]
        psi_j_s = V_ref[sl, :, j]

        # Coupling terms.
        dC_ref = np.einsum("tx,tx->t", np.conjugate(V_ref[:, :, i]), dV_ref[:, :, j], optimize=True)
        dE_ref_flat = dE_ref
        dH_ref = dE_ref_flat * dC_ref
        phi_ref = -cumulative_trapz(dE_ref_flat, times_ref)

        den_eps = 1e-12 * (1.0 + np.nanmax(np.abs(dE_ref_flat)))
        B_ref_full = np.empty_like(dC_ref, dtype=complex)
        mask = np.abs(dE_ref_flat) > den_eps
        B_ref_full[mask] = np.exp(1j * phi_ref[mask]) * (dH_ref[mask] / dE_ref_flat[mask])
        B_ref_full[~mask] = np.exp(1j * phi_ref[~mask]) * dC_ref[~mask]

        dE_s = dE_ref_flat[sl]
        dH_s = dH_ref[sl]
        B_s = B_ref_full[sl]

        # --- Optional local integration on [t1, t2] if sufficient data is available. ---
        ci2_sel = None
        cj2_sel = None
        if (
            times_full is not None
            and c_global is not None
            and D2_op is not None
            and V_op is not None
            and t_sel.size >= 3  # required by np.gradient(edge_order=2)
        ):
            c_init = interp_c_at_time(times_full, c_global, t1)
            U_win = V_ref[sl, :, :]
            E_win = E_ref[sl, :]
            c_win = integrate_window_with_init(t_sel, E_win, U_win, D2_op, V_op, a_fun, b_fun, c_init)
            ci2_sel = np.abs(c_win[:, i])**2
            cj2_sel = np.abs(c_win[:, j])**2

        max_dH_val = np.nanmax(np.abs(dH_s)) if dH_s.size else np.nan
        area_B_val = np.abs(np.trapezoid(B_s, t_sel)) if B_s.size else np.nan

        last_payload = dict(
            times_sel=t_sel,
            psi_i_sel=psi_i_s,
            psi_j_sel=psi_j_s,
            dE_sel=dE_s,
            dH_sel=dH_s,
            B_sel=B_s,
            t_star=t_star_ref,
            DeltaE_star=DeltaE_star,
            max_dH=max_dH_val,
            area_B=area_B_val,
            t1=t1,
            t2=t2,
            t1p=t1p,
            t2p=t2p,
            pts_under=pts_under,
            ci2_sel=ci2_sel,
            cj2_sel=cj2_sel,
        )

        if pts_under >= min_points_in_window or it >= max_iters:
            return last_payload

        it += 1


# =============================================================================
# Diagnostics: fidelity, basin probability, stabilization time
# =============================================================================

def fidelity_over_time(psi_qhd: np.ndarray, psi_num: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Compute fidelity over time between two state trajectories.

    Fidelity definition:
        F(t) = |<psi_qhd(t) | psi_num(t)>| / (||psi_qhd(t)|| * ||psi_num(t)||)

    Parameters
    ----------
    psi_qhd, psi_num:
        State trajectories with the same shape (Nt, M) (or compatible 2D shapes).
    times:
        Time grid, shape (Nt,). Used only for consistency checks.

    Returns
    -------
    np.ndarray
        Fidelity array, shape (Nt,).
    """
    assert psi_qhd.shape == psi_num.shape
    Nt = psi_qhd.shape[0]
    assert len(times) == Nt

    fidelity = np.empty(Nt, dtype=float)
    for s in range(Nt):
        denom = np.linalg.norm(psi_qhd[s]) * np.linalg.norm(psi_num[s])
        fidelity[s] = float(np.abs(np.vdot(psi_qhd[s], psi_num[s]) / denom))
    return fidelity


def probability_in_secondary_basins(density: np.ndarray, global_basin_idx: int) -> np.ndarray:
    """
    Estimate the probability mass outside the (assumed) global basin.

    Parameters
    ----------
    density:
        Probability mass array, shape (Nt, ...). This routine assumes the second axis
        indexes basin membership / indicator; it preserves the original behavior:
            1 - density[:, global_basin_idx]
    global_basin_idx:
        Index identifying the global basin component.

    Returns
    -------
    np.ndarray
        Estimated secondary-basin probability, shape (Nt,).
    """
    return 1 - density[:, global_basin_idx]


def detect_stabilization_time(
    curve: np.ndarray,
    times: np.ndarray,
    *,
    tol_rel: float = 0.02,
    min_tail_frac: float = 0.2,
) -> float | None:
    """
    Detect a stabilization time for a time series using tail MAD (robust) criterion.

    The curve is considered stabilized if, on the last p points where:
        p = max(ceil(min_tail_frac * n), 8),
    the relative MAD satisfies:
        MAD / |median| < tol_rel.

    If stabilized, the routine returns the earliest time t_i such that all subsequent
    points remain within 3*MAD of the tail median.

    Parameters
    ----------
    curve:
        Time series, shape (n,).
    times:
        Time grid, shape (n,).
    tol_rel:
        Relative MAD threshold.
    min_tail_frac:
        Fraction of points used in the tail window.

    Returns
    -------
    float | None
        Detected stabilization time, or None if no stabilization is found.
    """
    curve = np.asarray(curve, dtype=float).reshape(-1)
    times = np.asarray(times, dtype=float).reshape(-1)
    n = len(curve)

    if n < 8:
        return None

    p = max(int(np.ceil(min_tail_frac * n)), 8)
    tail = curve[-p:]
    L = np.median(tail)
    mad = np.median(np.abs(tail - L)) + 1e-16

    if mad / (np.abs(L) + 1e-16) > tol_rel:
        return None

    thr = 3.0 * mad
    for i in range(n - p + 1):
        if np.all(np.abs(curve[i:] - L) <= thr):
            return float(times[i])
    return None
