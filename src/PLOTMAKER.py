from typing import Any, Callable, Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

from FUNCTION_ANALYSIS import _to_level_idx, find_local_minima_and_basins

# Matplotlib text settings:
# - Keep math parsing enabled (for labels like r"$a(t)$")
# - Do not force LaTeX rendering (portable on systems without TeX)
mpl.rcParams["text.parse_math"] = True
mpl.rcParams["text.usetex"] = False


# =============================================================================
# Common rendering helpers (report/figure panels)
# =============================================================================

def render_params_box(ax, meta: Dict[str, Any]) -> None:
    """
    Render a small, text-only panel summarizing the simulation parameters.

    Parameters
    ----------
    ax:
        Matplotlib axis where the text box is drawn.
    meta:
        Metadata dictionary produced by the simulation runner/reporting pipeline.
        Expected keys include: d, q, domain, T, dt, bc, order, initialization.
    """
    ax.axis("off")

    d = int(meta.get("d"))
    q = int(meta.get("q"))
    N = 2**q

    domain = tuple(meta.get("domain"))
    x_min, x_max = float(domain[0]), float(domain[1])
    L = x_max - x_min
    dx = L / N

    init = meta.get("initialization", "uniform")
    init_txt = "uniforme" if str(init).lower() == "uniform" else str(init)

    order = meta.get("order", None)
    if order is None:
        order_txt = "—"
    else:
        try:
            o = int(order)
            if o == 1:
                order_txt = "1° ordine"
            elif o == 2:
                order_txt = "2° ordine"
            elif o == 4:
                order_txt = "4° ordine"
            else:
                order_txt = f"{o}° ordine"
        except Exception:
            order_txt = str(order)

    bc = meta.get("bc", "periodiche")

    lines = [
        rf"$d = {d}$",
        rf"$q = {q}, \ N = 2^q = {N}$",
        rf"$\Delta x = {dx:.2g}$",
        rf"$T = {float(meta.get('T')):.2g}, \ \Delta t = {float(meta.get('dt')):.2g}$",
        f"Boundary conditions: {bc}",
        f"Ordine simulazione numerica: {order_txt}",
        f"Inizializzazione: {init_txt}",
    ]

    y = 1.0
    dy = 0.12
    for line in lines:
        ax.text(
            0.0,
            y,
            line,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )
        y -= dy


def _schedule_formula_lines(meta: Dict[str, Any]) -> List[str]:
    """
    Extract schedule LaTeX strings from metadata and format them for display.

    Parameters
    ----------
    meta:
        Metadata dictionary expected to contain schedule LaTeX strings:
        `schedule_a_tex` and `schedule_b_tex`.

    Returns
    -------
    lines:
        A list of two strings suitable for rendering in a text axis.
    """
    a_label = meta.get("schedule_a_tex")
    b_label = meta.get("schedule_b_tex")
    return [fr"$a(t) =$ {a_label}", fr"$b(t) =$ {b_label}"]


def render_schedule_panel(
    ax_text,
    ax_plot,
    meta: Dict[str, Any],
    a_fun: Callable[[float], float],
    b_fun: Callable[[float], float],
) -> None:
    """
    Render a schedule panel: LaTeX formulas on the left and time series on the right.

    Parameters
    ----------
    ax_text, ax_plot:
        Matplotlib axes for text and plot, respectively.
    meta:
        Metadata dictionary; must contain `T` and schedule LaTeX strings.
    a_fun, b_fun:
        Callables implementing schedules a(t) and b(t).
    """
    # --- Text: formulas ---
    ax_text.axis("off")
    sched_lines = _schedule_formula_lines(meta)

    y = 1.0
    dy = 0.25
    for line in sched_lines:
        ax_text.text(
            0.0,
            y,
            line,
            transform=ax_text.transAxes,
            va="top",
            ha="left",
            fontsize=12.0,
        )
        y -= dy

    # --- Plot: a(t), b(t) curves ---
    T = float(meta.get("T", 1.0))
    ts = np.linspace(0.0, T, 400)
    a_vals = np.array([float(a_fun(t)) for t in ts])
    b_vals = np.array([float(b_fun(t)) for t in ts])

    ax_plot.plot(ts, a_vals, label=r"$a(t)$", lw=1.5)
    ax_plot.plot(ts, b_vals, label=r"$b(t)$", lw=1.5)
    ax_plot.set_xlabel("t")
    ax_plot.set_title("Schedule temporali $a(t)$, $b(t)$")
    ax_plot.grid(True, alpha=0.3)
    ax_plot.legend(fontsize=8)


def render_function_info(ax, meta: Dict[str, Any], f: Callable[[np.ndarray], np.ndarray]) -> None:
    """
    Render a text-only panel describing the benchmark function and its domain.

    Parameters
    ----------
    ax:
        Matplotlib axis where the text is drawn.
    meta:
        Metadata dictionary, expected keys: d, domain, f, f_tex.
    f:
        Potential callable used in the simulation (used only for a fallback name).
    """
    ax.axis("off")
    d = int(meta.get("d"))
    domain = tuple(meta.get("domain"))
    fname = meta.get("f", getattr(f, "__name__", "f"))

    y = 1.0
    ax.text(0.0, y, rf"Funzione: {fname}", transform=ax.transAxes, ha="left", va="top")
    y -= 0.15

    ax.text(
        0.0,
        y,
        rf"Dominio: $[{domain[0]}, {domain[1]}]^{{d}}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    y -= 0.15

    f_tex = meta.get("f_tex")
    if f_tex:
        ax.text(
            0.0,
            y,
            f_tex,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12.0,
            multialignment="left",
        )


def render_function_plot(ax, coords: np.ndarray, f_vals_flat: np.ndarray, meta: Dict[str, Any]) -> None:
    """
    Dispatch function plotting based on dimension (d=1, d=2).

    For d>2, no plot is produced (text placeholder).

    Parameters
    ----------
    ax:
        Matplotlib axis to draw on.
    coords:
        Grid coordinates of shape (M, d).
    f_vals_flat:
        Potential values on the grid, flattened to shape (M,).
    meta:
        Metadata dictionary containing `d`.
    """
    dim = int(meta.get("d"))
    if dim == 1:
        plot_1dfunction(coords, f_vals_flat, ax=ax, show=False)
        ax.set_title("f(x) con minimi e bacino globale")
    elif dim == 2:
        plot_2dfunction(coords, f_vals_flat, ax=ax, show=False)
        ax.set_title("f(x,y) con minimi e bacino globale")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Nessun grafico per d > 2", ha="center", va="center", fontsize=9)


def render_table(ax, data: List[List[Any]], col_labels: List[str]) -> None:
    """
    Render a small table (up to 5 rows) inside an axis.

    Parameters
    ----------
    ax:
        Matplotlib axis where the table is drawn.
    data:
        Row-major data for the table.
    col_labels:
        Column labels.
    """
    ax.axis("off")
    if not data:
        ax.text(0.5, 0.5, "N/D", ha="center", va="center")
        return

    ncols = len(col_labels)
    # Column widths tuned to reduce text overflow in typical report layouts.
    if ncols == 2:
        col_widths = [0.65, 0.35]
    elif ncols == 3:
        col_widths = [0.48, 0.22, 0.30]
    else:
        col_widths = [1.0 / max(ncols, 1)] * ncols

    table = ax.table(
        cellText=data[:5],
        colLabels=col_labels,
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.0)
    table.scale(1.0, 1.25)

    try:
        table.auto_set_column_width(col=list(range(ncols)))
    except Exception:
        # Some Matplotlib backends may not support this; ignore for portability.
        pass


# =============================================================================
# Plot utilities
# =============================================================================

def plot_pair_heatmaps_shared_cb(
    psi_i: np.ndarray,
    psi_j: np.ndarray,
    times: np.ndarray,
    xgrid: np.ndarray,
    ax_left,
    ax_right,
    *,
    title_i: str = "",
    title_j: str = "",
):
    """
    Plot two side-by-side heatmaps |psi_i|^2 and |psi_j|^2 with a shared colorbar.

    Parameters
    ----------
    psi_i, psi_j:
        Complex arrays representing the two time-dependent states. Expected shape is
        (Nt, Nx) for 1D time-space heatmaps.
    times:
        Time array of shape (Nt,).
    xgrid:
        Spatial grid of shape (Nx,).
    ax_left, ax_right:
        Matplotlib axes where the two heatmaps are drawn.
    title_i, title_j:
        Titles for the left and right panels.

    Returns
    -------
    imL, imR, cbar:
        The two AxesImage objects and the shared colorbar.
    """
    dens_i = np.abs(psi_i) ** 2
    dens_j = np.abs(psi_j) ** 2

    vmax = float(np.nanmax([np.nanmax(dens_i), np.nanmax(dens_j)]))
    norm = PowerNorm(gamma=0.3, vmin=0.0, vmax=vmax)
    extent = [xgrid.min(), xgrid.max(), times.min(), times.max()]

    imL = ax_left.imshow(
        dens_i,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
        norm=norm,
        cmap="inferno",
    )
    imR = ax_right.imshow(
        dens_j,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
        norm=norm,
        cmap="inferno",
    )

    ax_left.set_xlabel("x")
    ax_left.set_ylabel("t")
    ax_left.set_title(title_i)

    ax_right.set_xlabel("x")
    ax_right.set_ylabel("t")
    ax_right.set_title(title_j)

    cbar = ax_left.figure.colorbar(imL, ax=[ax_left, ax_right], fraction=0.046, pad=0.04)
    return imL, imR, cbar



def plot_weights_heatmap(
    weights: np.ndarray,
    times: np.ndarray,
    *,
    ax=None,
    title: str = "",
    gamma: float = 0.3,
    cmap: str = "inferno",
):
    """
    Plot a heatmap of weights w_k(t) over the first n_states eigenstates.

    Parameters
    ----------
    weights:
        Array of shape (Nt, n_states).
    times:
        Array of shape (Nt,).
    ax:
        Optional Matplotlib axis. If None, a new (fig, ax) is created.
    title:
        Plot title.
    gamma:
        Gamma parameter for PowerNorm (gamma < 1 enhances low values).
    cmap:
        Matplotlib colormap name.

    Returns
    -------
    fig, ax, im:
        Matplotlib figure, axis, and AxesImage.
    """
    weights = np.asarray(weights)
    times = np.asarray(times)
    if weights.ndim != 2 or weights.shape[0] != times.shape[0]:
        raise ValueError("weights must have shape (Nt, n_states) and len(times) must equal Nt.")

    Z = weights.T  # rows are state index (k), columns are time (t)
    n_states = Z.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
    else:
        fig = ax.figure

    vmax = float(np.nanmax(Z)) if np.isfinite(np.nanmax(Z)) else 1.0
    tmin = float(times.min())
    tmax = float(times.max())

    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[tmin, tmax, -0.5, n_states - 0.5],
        interpolation="nearest",
        norm=PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax),
        cmap=cmap,
    )

    ax.set_title(title)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$k$")

    ax.set_ylim(-0.5, n_states - 0.5)
    ax.set_yticks(np.arange(n_states))
    ax.set_yticklabels([str(k) for k in range(n_states)])

    return fig, ax, im


def plot_weights_sum(
    weights: np.ndarray,
    times: np.ndarray,
    *,
    ax=None,
    title: str = "Somma pesi sui primi n_states",
):
    """
    Plot the total weight Σ_k |c_k(t)|^2 over the retained eigenstates.

    Parameters
    ----------
    weights:
        Array of shape (Nt, n_states).
    times:
        Array of shape (Nt,).
    ax:
        Optional Matplotlib axis. If None, a new (fig, ax) is created.
    title:
        Plot title.

    Returns
    -------
    fig, ax:
        Matplotlib figure and axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
    else:
        fig = ax.figure

    sum_w = weights.sum(axis=1)
    ax.plot(times, sum_w, lw=1.6)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.grid(True, alpha=0.25)
    return fig, ax


def plot_1dfunction(coords, values, *, ax=None, show=True):
    """
    Plot a 1D function f(x) and annotate minima and the global basin.

    The plot includes:
      - global basin boundaries (vertical dashed gray lines near boundary maxima),
      - local minima (vertical yellow lines),
      - global minimum (vertical green line).

    This function is compatible with `find_local_minima_and_basins`, which returns
    the union of basins of global minima.

    Parameters
    ----------
    coords:
        Coordinates array of shape (M, 1).
    values:
        Function values array of shape (M,).
    ax:
        Optional Matplotlib axis. If None, a new (fig, ax) is created.
    show:
        If True and the figure was created internally, call plt.show().

    Returns
    -------
    fig, ax:
        Matplotlib figure and axis.
    """
    coords = np.asarray(coords, float)
    values = np.asarray(values, float).reshape(-1)

    if coords.ndim != 2 or coords.shape[1] != 1:
        raise ValueError("coords must have shape (M, 1) for a 1D plot.")
    if values.shape[0] != coords.shape[0]:
        raise ValueError("values must have the same length as coords.")

    out = find_local_minima_and_basins(coords, values)
    x_levels = np.asarray(out["levels"][0])
    N = int(out["N"])

    # Reconstruct values on the canonical grid order and use the grid levels as x.
    grid_idx = _to_level_idx(coords[:, 0], x_levels)  # (M,)
    inv = np.empty(coords.shape[0], dtype=np.int64)
    inv[grid_idx] = np.arange(coords.shape[0])

    x = x_levels  # (N,)
    y = values[inv]  # (N,)

    # Build a boolean mask for the global basin in 1D.
    mask = np.zeros(N, dtype=bool)
    if out.get("basin_1d") is not None:
        mask[np.asarray(out["basin_1d"], dtype=int)] = True
    else:
        mask[np.asarray(out["global_basin_idx_flat"], dtype=int)] = True

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        created = True
    else:
        fig = ax.figure

    ax.plot(x, y, linewidth=1.6)

    # Basin boundaries: where the mask changes value.
    boundaries = np.flatnonzero(mask[:-1] != mask[1:])
    for b in boundaries:
        # Find a nearby local maximum to place a visually meaningful boundary marker.
        w0 = max(0, b - 1)
        w1 = min(N - 1, b + 2)
        m_local = w0 + int(np.argmax(y[w0 : w1 + 1]))
        ax.axvline(x[m_local], color="gray", linestyle="--", linewidth=1.4, zorder=3)

    # Local minima detection directly on y (kept to preserve line-annotation behavior).
    local_min_idx = []
    for i in range(N):
        lval = y[i - 1] if i > 0 else np.inf
        rval = y[i + 1] if i < N - 1 else np.inf
        if y[i] <= lval and y[i] <= rval:
            local_min_idx.append(i)

    # Thin contiguous plateaus by keeping the leftmost representative.
    filtered_min_idx = []
    for idx in local_min_idx:
        if (
            filtered_min_idx
            and idx == filtered_min_idx[-1] + 1
            and np.isclose(y[idx], y[idx - 1])
        ):
            continue
        filtered_min_idx.append(idx)

    # Global minimum marker.
    xg = float(np.atleast_1d(out["global_min_coord"]).reshape(-1)[0])
    g_idx = int(np.argmin(np.abs(x - xg)))
    ax.axvline(
        x[g_idx],
        color="green",
        linestyle="-",
        linewidth=0.9,
        zorder=4,
        label="Minimo globale",
    )

    # Other local minima markers.
    for i in filtered_min_idx:
        if i == g_idx:
            continue
        ax.axvline(x[i], color="yellow", linestyle="-", linewidth=0.6, zorder=4)

    ax.set_xlabel("x")
    ax.grid(True, alpha=0.25)

    if show and created:
        plt.show()

    return fig, ax


def plot_2dfunction(coords, values, *, ax=None, show=True):
    """
    Plot a 2D heatmap of f(x, y) with basin contours and minima markers.

    The plot includes:
      - heatmap of f on the grid,
      - contour of the union of global-minima basins (dashed gray line),
      - global minima (red-edged markers),
      - local minima (white-edged markers).

    Compatible with the updated `find_local_minima_and_basins`.

    Parameters
    ----------
    coords:
        Coordinates array of shape (M, 2).
    values:
        Function values array of shape (M,).
    ax:
        Optional Matplotlib axis. If None, a new (fig, ax) is created.
    show:
        If True and the figure was created internally, call plt.show().

    Returns
    -------
    fig, ax:
        Matplotlib figure and axis.
    """
    coords = np.asarray(coords, float)
    values = np.asarray(values, float).reshape(-1)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (M, 2) for a 2D plot.")
    if values.shape[0] != coords.shape[0]:
        raise ValueError("values must have the same length as coords.")

    out = find_local_minima_and_basins(coords, values)
    levels = out["levels"]
    N = int(out["N"])
    xlev, ylev = np.asarray(levels[0]), np.asarray(levels[1])

    # Reconstruct Z (N×N) in canonical grid order (C-order).
    ix = _to_level_idx(coords[:, 0], xlev)
    iy = _to_level_idx(coords[:, 1], ylev)
    flat_idx = np.ravel_multi_index((iy, ix), (N, N), order="C")

    inv = np.empty(coords.shape[0], dtype=np.int64)
    inv[flat_idx] = np.arange(coords.shape[0])
    Z = values[inv].reshape(N, N, order="C")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
        created_fig = True
    else:
        fig = ax.figure

    im = ax.imshow(
        Z,
        origin="lower",
        extent=(xlev.min(), xlev.max(), ylev.min(), ylev.max()),
        aspect="equal",
        interpolation="nearest",
    )

    # Contour of the union of basins of global minima.
    basin_mask_2d = np.asarray(out.get("basin_mask_2d", np.array([])), dtype=bool)
    if basin_mask_2d.size == N * N:
        ax.contour(
            xlev,
            ylev,
            basin_mask_2d.astype(int),
            levels=[0.5],
            colors="gray",
            linestyles="dashed",
            linewidths=0.9,
        )

    # Minima markers: global minima and local minima.
    mins = np.asarray(out.get("min_coords", np.empty((0, 2))))
    if mins.size:
        gidxs = np.asarray(out.get("global_min_indices", [out.get("global_idx", 0)]), dtype=int)

        ax.scatter(
            mins[gidxs, 0],
            mins[gidxs, 1],
            s=20,
            facecolors=None,
            edgecolors="red",
            linewidths=1.2,
            zorder=5,
            label="Minimi globali",
        )

        mask_loc = np.ones(len(mins), dtype=bool)
        mask_loc[gidxs] = False
        if np.any(mask_loc):
            ax.scatter(
                mins[mask_loc, 0],
                mins[mask_loc, 1],
                s=15,
                facecolors=None,
                edgecolors="white",
                linewidths=1.0,
                zorder=4,
                label="Minimi locali",
            )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)

    if show and created_fig:
        plt.show()

    return fig, ax


def plot_density_timespace_1d(
    psi: np.ndarray,
    times: np.ndarray,
    xgrid: np.ndarray,
    *,
    ax=None,
    title: str = "",
    min_positions: Optional[np.ndarray] = None,
    global_min_positions: Optional[np.ndarray] = None,
):
    """
    Plot a time-space heatmap of |psi(x, t)|^2 for 1D simulations.

    A PowerNorm (gamma < 1) is used to emphasize low-probability regions.

    Parameters
    ----------
    psi:
        Complex array of shape (Nt, Nx).
    times:
        Array of shape (Nt,).
    xgrid:
        Array of shape (Nx,).
    ax:
        Optional Matplotlib axis. If None, a new (fig, ax) is created.
    title:
        Plot title.
    min_positions, global_min_positions:
        Optional arrays of x-positions where vertical dashed lines are drawn.
        Global minima positions are highlighted with a different color.

    Returns
    -------
    fig, ax:
        Matplotlib figure and axis.
    """
    dens = np.abs(psi) ** 2

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    else:
        fig = ax.figure

    norm = PowerNorm(gamma=0.3, vmin=0.0, vmax=float(np.nanmax(dens)))
    im = ax.imshow(
        dens,
        origin="lower",
        aspect="auto",
        extent=[xgrid.min(), xgrid.max(), times.min(), times.max()],
        interpolation="nearest",
        norm=norm,
        cmap="inferno",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(title)

    # Optional dashed vertical lines at minima positions.
    if min_positions is not None:
        mins = np.atleast_1d(min_positions).astype(float)
        gmins = (
            np.atleast_1d(global_min_positions).astype(float)
            if global_min_positions is not None
            else np.array([])
        )
        for x in mins:
            color = "yellow"
            if gmins.size and np.any(np.isclose(x, gmins, atol=1e-12)):
                color = "green"
            ax.axvline(float(x), color=color, linestyle="--", linewidth=0.6, alpha=0.8)

    fig.colorbar(im, ax=ax)
    return fig, ax


def plot_density_final_2d(
    psi_final_flat: np.ndarray,
    q: int,
    domain: tuple[float, float],
    *,
    ax=None,
    title: str = "",
):
    """
    Plot a 2D heatmap of the final probability density |psi(x, y)|^2.

    Parameters
    ----------
    psi_final_flat:
        Final state vector flattened to shape (N*N,).
    q:
        Qubits per axis (N = 2**q).
    domain:
        Spatial domain (x_min, x_max), used for axis extents.
    ax:
        Optional Matplotlib axis. If None, a new (fig, ax) is created.
    title:
        Plot title.

    Returns
    -------
    fig, ax:
        Matplotlib figure and axis.
    """
    N = 2**q
    dens = (np.abs(psi_final_flat) ** 2).reshape(N, N, order="F")
    x_min, x_max = domain

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.8, 4.4))
    else:
        fig = ax.figure

    norm = PowerNorm(gamma=0.3, vmin=0.0, vmax=float(np.nanmax(dens)))
    im = ax.imshow(
        dens,
        origin="lower",
        extent=[x_min, x_max, x_min, x_max],
        interpolation="nearest",
        aspect="equal",
        norm=norm,
        cmap="inferno",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    return fig, ax


def _plot_3dfunction(
    coords: np.ndarray,
    values_flat: np.ndarray,
    *,
    ax=None,
    title: str = "",
):
    """
    Plot a 3D surface for a 2D function f(x, y), reconstructed on the canonical grid.

    This helper reconstructs the grid using the level information produced by
    `find_local_minima_and_basins`, then draws a surface plot. It does not save to disk.

    Parameters
    ----------
    coords:
        Coordinates array of shape (M, 2).
    values_flat:
        Flattened values array of shape (M,).
    ax:
        Optional 3D axis. If None, a new 3D axis is created.
    title:
        Optional title.

    Returns
    -------
    fig, ax:
        Matplotlib figure and 3D axis.
    """
    out_levels = find_local_minima_and_basins(coords, values_flat)
    xlev = np.asarray(out_levels["levels"][0])
    ylev = np.asarray(out_levels["levels"][1])
    Nlev = int(out_levels["N"])

    def _to_idx(col, lev):
        pos = np.searchsorted(lev, col)
        pos = np.clip(pos, 0, len(lev) - 1)
        left = np.maximum(pos - 1, 0)
        choose_left = np.abs(col - lev[left]) < np.abs(col - lev[pos])
        return np.where(choose_left, left, pos).astype(np.int64)

    ix = _to_idx(coords[:, 0], xlev)
    iy = _to_idx(coords[:, 1], ylev)

    # Keep mapping logic intact.
    flat_idx = np.ravel_multi_index((ix, iy), (Nlev, Nlev), order="C")
    inv = np.empty(coords.shape[0], dtype=np.int64)
    inv[flat_idx] = np.arange(coords.shape[0])

    Z = values_flat[inv].reshape(Nlev, Nlev, order="C")
    X, Y = np.meshgrid(xlev, ylev, indexing="xy")

    created = False
    if ax is None:
        fig = plt.figure(figsize=(5.0, 4.4))
        ax = fig.add_subplot(111, projection="3d")
        created = True
    else:
        fig = ax.figure

    stride = max(Nlev // 64, 1)
    ax.plot_surface(X, Y, Z, rstride=stride, cstride=stride, linewidth=0, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    if title:
        ax.set_title(title)

    return fig, ax
