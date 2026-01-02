from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def _levels_and_N(coords: np.ndarray) -> Tuple[List[np.ndarray], int]:
    """
    Infer grid levels and per-axis grid size from a regular Cartesian grid.

    Parameters
    ----------
    coords:
        Array of shape (M, d) containing grid coordinates (Cartesian product grid).

    Returns
    -------
    levels:
        List of length d. Each entry is the sorted unique coordinate values along that axis.
    N:
        Number of grid points per axis (assumed equal along all axes).

    Raises
    ------
    ValueError
        If axes have different cardinalities, or if M != N**d.
    """
    coords = np.asarray(coords, float)
    M, d = coords.shape

    levels = [np.unique(coords[:, j]) for j in range(d)]
    Ns = [len(lev) for lev in levels]
    if len(set(Ns)) != 1:
        raise ValueError(f"Per-axis grid sizes are not equal: {Ns}")

    N = Ns[0]
    if N**d != M:
        raise ValueError(f"Number of points M={M} is not compatible with N**d={N**d}.")
    return levels, N


def _to_level_idx(col: np.ndarray, lev: np.ndarray) -> np.ndarray:
    """
    Map coordinate values to nearest level indices for one axis.

    This is robust to tiny floating-point discrepancies by snapping each coordinate
    to the closest entry in `lev`.

    Parameters
    ----------
    col:
        Coordinate column of shape (M,).
    lev:
        Sorted unique levels for that axis.

    Returns
    -------
    np.ndarray
        Integer indices in [0, len(lev)-1], shape (M,).
    """
    pos = np.searchsorted(lev, col)
    pos = np.clip(pos, 0, len(lev) - 1)

    left = np.maximum(pos - 1, 0)
    choose_left = np.abs(col - lev[left]) < np.abs(col - lev[pos])
    return np.where(choose_left, left, pos).astype(np.int64)


def find_local_minima_and_basins(
    coords: np.ndarray,
    values: np.ndarray,
    rtol_plateau: float = 1e-12,
    atol_plateau: float = 1e-15,
) -> Dict[str, Any]:
    """
    Find local minima and (steepest-descent) attraction basins on a regular dD grid.

    The method:
      1) snaps coordinates to a regular Cartesian grid and reshapes `values` in a
         consistent C-order flattening;
      2) performs a single-step steepest descent to the best face-neighbor, then
         applies path compression to identify sinks (fixed points);
      3) merges sinks that form a plateau (equal value within tolerances);
      4) labels each grid point by its destination plateau label.

    Compatibility notes
    -------------------
    The return dictionary keeps backward-compatible keys for d=1 and d=2:
      - d=1: `basin_1d` contains flat indices of the global basin.
      - d=2: `basin_mask_2d` is a boolean mask of shape (N, N).
    For d >= 3, `basin_mask_nd` is always provided with shape (N,)*d.

    Parameters
    ----------
    coords:
        Grid coordinates of shape (M, d), assumed to form a full Cartesian product grid.
    values:
        Scalar field sampled on the grid, of shape (M,) or broadcastable to (M,).
    rtol_plateau, atol_plateau:
        Relative/absolute tolerances used when merging sink plateaux.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing minima information, global basin indices, and masks.
        The structure is intentionally kept stable for existing downstream code.

    Raises
    ------
    ValueError
        If inputs have inconsistent shapes or the grid is not regular.
    """
    coords = np.asarray(coords, float)
    v = np.asarray(values, float).reshape(-1)

    if coords.ndim != 2:
        raise ValueError("coords must have shape (M, d).")
    M, d = coords.shape
    if v.shape[0] != M:
        raise ValueError("`values` must have length M = coords.shape[0].")
    if d < 1:
        raise ValueError("d must be >= 1.")

    levels, N = _levels_and_N(coords)

    # Map coordinates to integer grid indices per axis (0..N-1 for each axis).
    grid_idx = np.stack([_to_level_idx(coords[:, j], levels[j]) for j in range(d)], axis=1)

    # Flatten indices consistently with an (N,)*d grid in C-order.
    # Keep d=1 and d=2 conventions as in previous versions, generalize for d>=3.
    if d == 1:
        flat_idx = grid_idx[:, 0]
    elif d == 2:
        ix = grid_idx[:, 0]  # x-index
        iy = grid_idx[:, 1]  # y-index
        # Use (iy, ix) so that reshaping to (N, N) yields Z[y, x] in C-order.
        flat_idx = np.ravel_multi_index((iy, ix), (N, N), order="C")
    else:
        flat_idx = np.ravel_multi_index(tuple(grid_idx.T), (N,) * d, order="C")

    # Inverse mapping to reorder values to grid-flattened order.
    inv = np.empty(M, dtype=np.int64)
    inv[flat_idx] = np.arange(M)
    v_flat = v[inv]

    # Mapping from flat-grid index back to original point index in `coords`.
    point_from_flat = np.empty(M, dtype=np.int64)
    point_from_flat[flat_idx] = np.arange(M)

    # Build face-neighborhood (2d neighbors per point) in flattened grid indexing.
    multi = np.stack(np.unravel_index(np.arange(M), (N,) * d, order="C"), axis=1)
    neighs: List[np.ndarray] = []

    for ax in range(d):
        # -1 along axis ax
        m = multi.copy()
        mask = m[:, ax] > 0
        mminus = np.full(M, -1, np.int64)
        m[mask, ax] -= 1
        mminus[mask] = np.ravel_multi_index(tuple(m[mask].T), (N,) * d, order="C")
        neighs.append(mminus)

        # +1 along axis ax
        m = multi.copy()
        mask = m[:, ax] < N - 1
        mplus = np.full(M, -1, np.int64)
        m[mask, ax] += 1
        mplus[mask] = np.ravel_multi_index(tuple(m[mask].T), (N,) * d, order="C")
        neighs.append(mplus)

    neighs = np.stack(neighs, axis=1)  # (M, 2d), -1 indicates "no neighbor" at boundary.

    # One-step steepest descent to the best face-neighbor (then path compression).
    next_idx = np.arange(M, dtype=np.int64)
    best = v_flat.copy()
    for k in range(neighs.shape[1]):
        n = neighs[:, k]
        ok = n >= 0
        better = ok & (v_flat[n] < best)
        next_idx[better] = n[better]
        best[better] = v_flat[n[better]]

    # Path compression: next_idx[i] becomes the sink reached by repeated descent.
    changed = True
    while changed:
        new_next = next_idx[next_idx]
        changed = np.any(new_next != next_idx)
        next_idx = new_next

    sinks = next_idx == np.arange(M, dtype=np.int64)

    def face_neighbors(u: int) -> List[int]:
        """
        Return valid face neighbors of a flattened grid index u.

        This is used only during plateau merging (connecting sinks with equal values).
        """
        res: List[int] = []
        mu = multi[u]
        for ax in range(d):
            if mu[ax] > 0:
                m = mu.copy()
                m[ax] -= 1
                res.append(np.ravel_multi_index(tuple(m), (N,) * d, order="C"))
            if mu[ax] < N - 1:
                m = mu.copy()
                m[ax] += 1
                res.append(np.ravel_multi_index(tuple(m), (N,) * d, order="C"))
        return res

    # Merge sinks that form a plateau (equal value within tolerances).
    sink_label = -np.ones(M, dtype=np.int64)
    K = 0
    for s in np.where(sinks)[0]:
        if sink_label[s] != -1:
            continue

        v0 = v_flat[s]
        stack = [s]
        sink_label[s] = K

        while stack:
            u = stack.pop()
            for nb in face_neighbors(u):
                if (not sinks[nb]) or (sink_label[nb] != -1):
                    continue
                if np.isclose(
                    v_flat[nb],
                    v0,
                    rtol=rtol_plateau,
                    atol=atol_plateau * max(1.0, abs(v0)),
                ):
                    sink_label[nb] = K
                    stack.append(nb)

        K += 1

    # Destination plateau label for each point.
    labels_flat = sink_label[next_idx]

    # One representative sink per plateau label.
    minima_flat = np.zeros(K, dtype=np.int64)
    for k in range(K):
        minima_flat[k] = np.where(sinks & (sink_label == k))[0][0]

    minima_vals = v_flat[minima_flat]
    minima_coords = coords[point_from_flat[minima_flat]]

    # Identify all global minima within tolerance.
    global_min_val = float(minima_vals.min())
    atol = max(atol_plateau, 1e-15) * max(1.0, abs(global_min_val))
    rtol = max(rtol_plateau, 1e-12)
    is_global = np.isclose(minima_vals, global_min_val, rtol=rtol, atol=atol)

    global_min_indices = np.where(is_global)[0]
    global_idx = int(global_min_indices[0])  # backward-compatible single index

    # Barrier estimate only for d=1, as in older behavior.
    barrier_val: float | None = None
    if d == 1 and minima_vals.size >= 2:
        non_global = np.where(~is_global)[0]
        if non_global.size > 0:
            # "Primary" secondary minimum = lowest non-global minimum.
            sec_idx = int(non_global[np.argmin(minima_vals[non_global])])

            # Flat grid indices of the two minima on the 1D grid.
            ig_flat = int(minima_flat[global_idx])
            is_flat = int(minima_flat[sec_idx])
            i0, i1 = sorted((ig_flat, is_flat))

            if i1 > i0:
                segment = v_flat[i0 : i1 + 1]
                barrier_val = float(np.max(segment))
            else:
                barrier_val = float(global_min_val)

    # Union of attraction basins of all global minima plateaux.
    basin_mask_flat = np.isin(labels_flat, global_min_indices)
    global_basin_idx_flat = np.where(basin_mask_flat)[0]
    global_basin_idx_orig = point_from_flat[global_basin_idx_flat]

    # Visualization masks (preserve old keys where applicable).
    basin_1d = global_basin_idx_flat if d == 1 else None
    basin_mask_2d = basin_mask_flat.reshape(N, N, order="C") if d == 2 else None
    basin_mask_nd = basin_mask_flat.reshape((N,) * d, order="C")

    out: Dict[str, Any] = dict(
        levels=levels,
        N=N,
        dim=d,
        min_coords=minima_coords,
        min_vals=minima_vals,
        global_idx=global_idx,
        global_min_indices=global_min_indices.astype(int),
        global_min_coord=minima_coords[global_idx],
        global_min_val=global_min_val,
        global_basin_idx_flat=global_basin_idx_flat,
        basin_1d=basin_1d,
        basin_mask_2d=basin_mask_2d,
        basin_mask_nd=basin_mask_nd,
        global_basin_idx_orig=global_basin_idx_orig,
        barrier_val=barrier_val,
    )
    return out


def hessian_eigvals_at_points(
    f: callable,
    points: np.ndarray,
    h: np.ndarray | float | None = None,
) -> np.ndarray:
    """
    Numerically estimate Hessian eigenvalues at given points via centered differences.

    This implementation supports only d=1 or d=2 and follows a standard finite
    difference stencil:
      - diagonal entries: second centered difference per axis;
      - off-diagonal (d=2): mixed derivative via the 4-corner stencil.

    Parameters
    ----------
    f:
        Callable accepting an array of shape (P, d) and returning values of shape (P,).
    points:
        Evaluation points of shape (K, d) (or (d,) for a single point).
    h:
        Step size for finite differences. Can be:
          - None: choose an adaptive step per dimension;
          - float: same step for all dimensions;
          - array-like: broadcastable to (K, d).

    Returns
    -------
    np.ndarray
        Array of shape (K, d) containing the Hessian eigenvalues at each point,
        sorted in ascending order for each point.

    Raises
    ------
    ValueError
        If d is not 1 or 2.
    """
    pts = np.atleast_2d(np.asarray(points, float))
    K, d = pts.shape
    if d not in (1, 2):
        raise ValueError("hessian_eigvals_at_points supports only d=1 or d=2.")

    # Step size per dimension.
    if h is None:
        # Scale step with typical magnitude of points (or 1.0 near zero).
        base = np.maximum(np.abs(pts).mean(axis=0), 1.0)
        h = 1e-3 * base

    h = np.broadcast_to(np.asarray(h, float).reshape(1, -1), (K, d))
    eigvals = np.zeros((K, d), float)

    for k in range(K):
        p = pts[k]
        hk = h[k]

        def eval_batch(XX):
            return np.asarray(f(np.asarray(XX, float))).reshape(-1)

        f0 = eval_batch([p])[0]

        # Build Hessian via finite differences.
        H = np.zeros((d, d), float)

        # Diagonal terms.
        for i in range(d):
            e = np.zeros(d)
            e[i] = 1.0
            fp = eval_batch([p + hk[i] * e])[0]
            fm = eval_batch([p - hk[i] * e])[0]
            H[i, i] = (fp - 2.0 * f0 + fm) / (hk[i] ** 2)

        # Off-diagonal (mixed) term for d=2.
        if d == 2:
            e1 = np.array([1.0, 0.0])
            e2 = np.array([0.0, 1.0])
            fpp = eval_batch([p + hk[0] * e1 + hk[1] * e2])[0]
            fpm = eval_batch([p + hk[0] * e1 - hk[1] * e2])[0]
            fmp = eval_batch([p - hk[0] * e1 + hk[1] * e2])[0]
            fmm = eval_batch([p - hk[0] * e1 - hk[1] * e2])[0]
            Hij = (fpp - fpm - fmp + fmm) / (4.0 * hk[0] * hk[1])
            H[0, 1] = H[1, 0] = Hij

        # Symmetrize and compute eigenvalues.
        H = 0.5 * (H + H.T)
        w = np.linalg.eigvalsh(H)

        # Numeric stabilization: clamp tiny magnitudes to zero, and cap negatives to -TOL.
        TOL = 1e-6
        w[np.abs(w) < TOL] = 0.0
        w[w < 0.0] = np.maximum(w[w < 0.0], -TOL)

        eigvals[k, :] = np.sort(w)

    return eigvals
