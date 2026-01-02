"""
Report generation utilities for the QHD / numerical simulation pipeline.

This module assembles plots and quantitative summaries into a single PDF report.
The numerical states and ancillary data are produced elsewhere; here we focus on
presentation:

- simulation metadata (grid, time step, boundary conditions),
- potential inspection (minima, basins, barrier estimates),
- wavefunction diagnostics (density heatmaps, spectral weights),
- optional "probability transfer" diagnostics.

Notes
-----
`matplotlib.use("Agg")` is kept in its original position to preserve behavior.
In general, backends should be selected before importing `pyplot`.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import PowerNorm
from typing import Callable, Tuple, Dict, Any, List
from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams["text.parse_math"] = True
mpl.rcParams["text.usetex"] = False



from UTILITIES import ComputationalBasisIndex_to_SpatialCoordinates
from SAVES import save_report_pdf
from FUNCTION_ANALYSIS import find_local_minima_and_basins, hessian_eigvals_at_points, _to_level_idx
from EIGENSOLVER import diagonalize, weights
from PLOTMAKER import (
    render_params_box,
    render_schedule_panel,
    render_function_info,
    render_function_plot,
    render_table,
    plot_density_timespace_1d,
    plot_weights_heatmap,
)

from DATA_ANALYSIS import (
    _simulation_completed,
    _extract_minima_and_barriers_1d,
    _compute_model_params_for_ho,
    _build_ho_and_free_energies,
    ensure_vecs_t_x_k,
    phase_align_no_perm,
    build_D2_V,
    integrate_c_global,
    refine_pair_window_and_analyze,
    series_max_probability,
    peak_position_and_fwhm,
    format_peak_position_vector,
)


def report_page_parametri(
    meta: Dict[str, Any],
    f: Callable[[np.ndarray], np.ndarray],
    a_fun: Callable[[float], float],
    b_fun: Callable[[float], float],
    coords: np.ndarray,
    f_vals_flat: np.ndarray,
    minima_table: List[List[str]],
) -> plt.Figure:
    """
    Create the "Overview / Parameters" page.
    
    The page includes:
    - a compact parameter box (d, q, grid spacing, dt, T, boundary conditions, etc.),
    - a schedule panel with the analytic forms and plots of a(t), b(t),
    - a brief potential summary table (minima and barrier metrics).
    
    Parameters
    ----------
    meta:
        Simulation metadata dictionary (see the batch runner and `numerical_simulation`).
    f:
        Potential function used in the run (callable accepting coords (M,d)).
    a_fun, b_fun:
        Time schedules used by the integrator.
    coords:
        Spatial coordinates array (M,d).
    f_vals_flat:
        Potential values evaluated on `coords`, flattened (M,).
    minima_table:
        Table rows pre-formatted for `matplotlib.table`.
    """

    fig1 = plt.figure(figsize=(10.2, 7.2), dpi=300)
    fig1.suptitle("Parametri", fontsize=12, y=0.98)

    gs_outer = GridSpec(
        nrows=1, ncols=2, width_ratios=[1.0, 1.0],
        wspace=0.25, figure=fig1
    )

    # Left column
    gs_left = GridSpecFromSubplotSpec(
        nrows=3, ncols=1,
        height_ratios=[0.35, 0.25, 0.40],
        hspace=0.15,
        subplot_spec=gs_outer[0, 0],
    )

    ax_params = fig1.add_subplot(gs_left[0, 0])
    render_params_box(ax_params, meta)

    ax_sched_text = fig1.add_subplot(gs_left[1, 0])
    ax_sched_plot = fig1.add_subplot(gs_left[2, 0])
    render_schedule_panel(ax_sched_text, ax_sched_plot, meta, a_fun, b_fun)

    # Right column
    gs_right = GridSpecFromSubplotSpec(
        nrows=3, ncols=1,
        height_ratios=[0.25, 0.35, 0.40],
        hspace=0.15,
        subplot_spec=gs_outer[0, 1],
    )

    ax_finfo = fig1.add_subplot(gs_right[0, 0])
    render_function_info(ax_finfo, meta, f)

    ax_minima = fig1.add_subplot(gs_right[1, 0])
    render_table(ax_minima, minima_table,
                  [r"$x_{\min}$", r"$f(x_{\min})$", r"$\omega^2_i$"])

    ax_fx = fig1.add_subplot(gs_right[2, 0])
    render_function_plot(ax_fx, coords, f_vals_flat, meta)

    return fig1


def report_page_risultati(
    d: int,
    q: int,
    domain: Tuple[float, float],
    meta: Dict[str, Any],
    f: Callable[[np.ndarray], np.ndarray],
    coords: np.ndarray,
    f_vals_flat: np.ndarray,
    out_min: Dict[str, Any] | None,
    basins_info: Dict[str, Any] | None,
    times: np.ndarray,
    psi_num: np.ndarray,
    a_fun: Callable[[float], float],
    b_fun: Callable[[float], float],
) -> plt.Figure:
    """
    Create the "Results" page.
    
    Depending on `meta["d"]`, this page renders:
    - the potential (1D/2D) with local minima and the global basin overlay,
    - a time-space density heatmap |psi(x,t)|^2 (1D only),
    - final density snapshots (1D/2D), plus optional markers for minima positions.
    
    All plotting helpers are delegated to `PLOTMAKER` to keep the report driver thin.
    
    Parameters
    ----------
    psi_num:
        Saved states array of shape (K, M), where K = number of snapshots.
    times:
        Snapshot times, shape (K,).
    xgrid:
        1D grid used for plotting; for d>1 only the first coordinate axis is used.
    meta:
        Simulation metadata dictionary.
    coords:
        Spatial coordinates array (M,d).
    f_vals_flat:
        Potential values evaluated on coords, flattened (M,).
    out_min:
        Output dictionary from `find_local_minima_and_basins`.
    """

    fig2 = plt.figure(figsize=(10.2, 7.2), dpi=300)
    fig2.suptitle("Risultati", fontsize=12, y=0.98)

    gs_outer = GridSpec(nrows=2, ncols=1, height_ratios=[1.5, 1.0],
                        hspace=0.25, figure=fig2)

    d_ = d
    dens_num = np.abs(psi_num)**2
    # valore minimo globale di f (f(x*))
    f_min = float(np.nanmin(f_vals_flat))

    # --- Riga 1: visualizzazione densità ---
    if d_ == 1:
        ax_top = fig2.add_subplot(gs_outer[0, 0])
        min_positions = None
        global_min_positions = None
        if out_min is not None:
            mins = np.asarray(out_min.get("min_coords", []))
            if mins.size:
                min_positions = mins[:, 0]
                gidxs = np.asarray(
                    out_min.get("global_min_indices",
                                [int(out_min.get("global_idx", 0))]),
                    dtype=int,
                )
                global_min_positions = mins[gidxs, 0]
        plot_density_timespace_1d(
            psi_num, times, coords.reshape(-1),
            ax=ax_top,
            title=r"$|\psi(x,t)|^2$",
            min_positions=min_positions,
            global_min_positions=global_min_positions,
        )

    elif d_ == 2:
        gs_top = GridSpecFromSubplotSpec(
            nrows=1, ncols=3, wspace=0.25,
            subplot_spec=gs_outer[0, 0]
        )
        ax1 = fig2.add_subplot(gs_top[0, 0])
        ax2 = fig2.add_subplot(gs_top[0, 1])
        ax3 = fig2.add_subplot(gs_top[0, 2])
        axes = [ax1, ax2, ax3]

        Ttot = float(meta.get("T", times[-1]))
        target_times = [Ttot/10.0, Ttot/3.0, Ttot]
        idxs = [int(np.argmin(np.abs(times - tt))) for tt in target_times]

        N = 2**q
        a_dom, b_dom = domain
        dens_list: List[np.ndarray] = []

        # Prova a riordinare |psi|^2 sulla stessa griglia usata per f(x,y)
        # così che l'orientamento coincida con plot_2dfunction/find_local_minima_and_basins.
        if out_min is not None and "levels" in out_min:
            try:
                levels = out_min["levels"]
                Nlev = int(out_min.get("N", N))
                xlev = np.asarray(levels[0], float)
                ylev = np.asarray(levels[1], float)

                coords_arr = np.asarray(coords, float)
                M = coords_arr.shape[0]
                if coords_arr.ndim != 2 or M != dens_num.shape[1]:
                    raise ValueError("Shape mismatch between coords e psi_num.")

                # stessi indici di livello usati in find_local_minima_and_basins
                grid_idx = np.stack(
                    [
                        _to_level_idx(coords_arr[:, 0], xlev),
                        _to_level_idx(coords_arr[:, 1], ylev),
                    ],
                    axis=1,
                )
                ix = grid_idx[:, 0]
                iy = grid_idx[:, 1]
                flat_idx = np.ravel_multi_index((iy, ix), (Nlev, Nlev), order="C")
                inv = np.empty(M, dtype=np.int64)
                inv[flat_idx] = np.arange(M)

                for idx_t in idxs:
                    dens_flat = dens_num[idx_t]
                    dens = dens_flat[inv].reshape(Nlev, Nlev, order="C")
                    dens_list.append(dens)

                a_dom, b_dom = float(xlev.min()), float(xlev.max())
            except Exception:
                dens_list = []

        # Fallback: reshaping semplice se l'allineamento fine fallisce
        if not dens_list:
            for idx_t in idxs:
                dens = dens_num[idx_t].reshape(N, N, order="F")
                dens_list.append(dens)

        vmax = max(float(np.nanmax(d)) for d in dens_list)
        norm = PowerNorm(gamma=0.3, vmin=0.0, vmax=vmax)


        mins = np.asarray(out_min.get("min_coords", [])) if out_min is not None else np.empty((0, 2))
        gidxs = np.asarray(
            out_min.get("global_min_indices", [int(out_min.get("global_idx", 0))]),
            dtype=int,
        ) if out_min is not None and mins.size else np.array([], int)

        for ax, dens, tt in zip(axes, dens_list, target_times):
            im = ax.imshow(
                dens, origin="lower",
                extent=[a_dom, b_dom, a_dom, b_dom],
                interpolation="nearest", aspect="equal",
                norm=norm, cmap="inferno"
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"|ψ(x,y)|², t={tt:.3g}")


        cbar = fig2.colorbar(im, ax=axes, fraction=0.046, pad=0.04)

    else:
        ax_top = fig2.add_subplot(gs_outer[0, 0])
        ax_top.axis("off")
        ax_top.text(0.5, 0.5,
                    "Visualizzazione densità non implementata per d > 2.",
                    ha="center", va="center")

    # --- Riga 2: curve + tabella ---
    gs_bottom = GridSpecFromSubplotSpec(
        nrows=1, ncols=2,
        width_ratios=[1.4, 1.0],
        wspace=0.30,
        subplot_spec=gs_outer[1, 0]
    )
    ax_curves = fig2.add_subplot(gs_bottom[0, 0])
    ax_table  = fig2.add_subplot(gs_bottom[0, 1])

    maxprob = series_max_probability(dens_num)
    ax_curves.plot(times, maxprob, label=r"$\max_{\mathbf{x}} |\psi(\mathbf{x},t)|^2$", lw=1.6)

    success_rate = None
    if basins_info is not None:
        gb = np.asarray(basins_info.get("global_basin_idx_orig", []), dtype=int)
        if gb.size > 0:
            p_global = dens_num[:, gb].sum(axis=1)
            success_rate = p_global
            ax_curves.plot(times, success_rate, label=r"$P_{suc}$", lw=1.6)

    exp_normalized = (dens_num @ f_vals_flat - f_min) / (np.mean(f_vals_flat) - f_min + 1e-30)
    ax_curves.plot(
        times,
        exp_normalized,
        label=r"$\mathbb{E}$",
        lw=1.6,
    )



    ax_curves.set_xlabel("t")
    ax_curves.legend(frameon=False)
    ax_curves.grid(True, alpha=0.25)

    dens_final_flat = dens_num[-1].reshape(-1)
    # prima chiamata: individua il massimo globale (coord_peak)
    coord_peak, _, peak_val = peak_position_and_fwhm(
        coords, dens_final_flat, axis=0
    )

    # errore sistematico: metà passo di griglia (uguale in tutte le direzioni)
    a_dom, b_dom = domain
    N = 2**q
    dx = (b_dom - a_dom) / N
    err_sys = 0.5 * dx

    # errore "statistico": metà FWHM lungo ciascun asse
    d = coords.shape[1]
    fwhm_axes = np.empty(d, float)
    for ax in range(d):
        _, fwhm_ax, _ = peak_position_and_fwhm(
            coords, dens_final_flat, axis=ax
        )
        fwhm_axes[ax] = fwhm_ax

    err_tot = np.empty(d, float)
    for ax in range(d):
        fwhm_ax = fwhm_axes[ax]
        err_sto = 0.0 if not np.isfinite(fwhm_ax) else 0.5 * float(fwhm_ax)
        err_tot[ax] = max(err_sys, err_sto)

    # stringa fisicamente corretta: per ogni componente x_j si scrive
    # x_j ± Δx_j, con Δx_j a 2 cifre significative.
    peak_pos_str = format_peak_position_vector(coord_peak, err_tot)



    final_success_rate = float("nan")
    if success_rate is not None:
        # if _simulation_completed(meta, out_min, f_vals_flat, a_fun, b_fun):
            final_success_rate = float(success_rate[-1])

    exp_normalized_final = float(exp_normalized[-1])

    rows = [
        [r"$\mathbf{x}_\text{picco}$", peak_pos_str],
        [r"$P_\text{suc}$",
         "nan" if final_success_rate is None or not np.isfinite(final_success_rate) else f"{final_success_rate:.3g}"],
        [r"$\mathbb{E}$", f"{exp_normalized_final:.3g}"],
    ]
    render_table(ax_table, rows, [r"Metrica ($t=T$)", "Valore"])

    return fig2


def report_page_componenti_spettrali(
    times: np.ndarray,
    weights: np.ndarray,
) -> plt.Figure:
    """
    Create the "Spectral components" page.
    
    This page visualizes the instantaneous spectral decomposition in a fixed eigenbasis:
    - heatmap of weights w_k(t) = |<phi_k(t) | psi(t)>|^2,
    - sum of weights over the retained eigenstates (sanity check).
    
    Parameters
    ----------
    weights_arr:
        Array (K, n_states) of spectral weights at each snapshot.
    times:
        Snapshot times, shape (K,).
    meta:
        Simulation metadata dictionary.
    title_suffix:
        Optional string appended to panel titles (e.g., "numerical" vs "QHD").
    """

    fig3 = plt.figure(figsize=(10.2, 7.2), dpi=300)
    fig3.suptitle("Componenti Spettrali", fontsize=12, y=0.98)

    gs3 = GridSpec(
        nrows=2, ncols=2,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.6, 0.8],
        hspace=0.35, wspace=0.35,
        figure=fig3,
    )

    Nt, n_states_eff = weights.shape
    n5 = min(5, n_states_eff)
    n10 = min(10, n_states_eff)

    ax_lines = fig3.add_subplot(gs3[0, 0])
    for k in range(n5):
        ax_lines.plot(times, weights[:, k], lw=1.4, label=rf"$|c_{{{k}}}(t)|^2$")
    ax_lines.set_xlabel("t")
    ax_lines.grid(True, alpha=0.25)
    ax_lines.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=False,
    )

    ax_heat = fig3.add_subplot(gs3[1, 0])
    _, _, im_w = plot_weights_heatmap(
        weights[:, :n10],
        times,
        ax=ax_heat,
        title=r"$|c_k(t)|^2$",
        gamma=0.3,
        cmap="inferno",
    )
    cbar = fig3.colorbar(im_w, ax=ax_heat, pad=0.01, fraction=0.048)

    ax_sum_t = fig3.add_subplot(gs3[:, 1])
    sum10 = np.sum(weights[:, :n10], axis=1)
    ax_sum_t.plot(sum10, times, lw=1.8, label=r"$\sum_{k=0}^{9}|c_k(t)|^2$")
    ax_sum_t.set_ylim(float(times.min()), float(times.max()))
    ax_sum_t.set_xlim(1.0, 0.0)
    ax_sum_t.set_ylabel("t")
    ax_sum_t.legend(frameon=False)
    ax_sum_t.grid(True, alpha=0.25)

    return fig3


def report_page_autostati(
    d: int,
    times: np.ndarray,
    xgrid: np.ndarray,
    coords: np.ndarray,
    f_vals_flat: np.ndarray,
    eigenvectors: np.ndarray,
    out_min: Dict[str, Any] | None,
) -> plt.Figure | None:
    """
    Create the "Eigenstates and weights" page.
    
    The page shows:
    - paired heatmaps of the densities of the first two eigenstates over time,
    - spectral weight heatmaps and their sums.
    
    This page is intended for diagnostic purposes and may be computationally heavy
    because it depends on diagonalization over snapshot times.
    
    Parameters
    ----------
    eigvecs:
        Eigenvectors array (K, M, n_states) returned by `diagonalize`.
    eigvals:
        Eigenvalues array (K, n_states) returned by `diagonalize`.
    psi_num:
        Numerical states (K, M) used to compute overlaps with eigenvectors.
    times, xgrid:
        Snapshot times and 1D plotting grid.
    meta:
        Simulation metadata dictionary.
    """

    if d != 1:
        return None

    Nt, M, K = eigenvectors.shape
    W = min(6, K)
    if W == 0:
        return None

    dens_all = np.abs(eigenvectors[:, :, :W])**2
    vmax = float(np.nanmax(dens_all)) if np.isfinite(np.nanmax(dens_all)) else 1.0
    norm = PowerNorm(gamma=0.3, vmin=0.0, vmax=vmax)

    min_positions, global_min_positions, barrier_positions = _extract_minima_and_barriers_1d(
        coords, f_vals_flat, out_min=out_min
    )

    fig4 = plt.figure(figsize=(10.2, 7.2), dpi=300)
    fig4.suptitle("Autostati", fontsize=12, y=0.98)

    gs4 = GridSpec(
        nrows=2, ncols=3,
        hspace=0.25, wspace=0.25,
        figure=fig4,
    )

    axes = []
    im = None
    for k in range(W):
        r = k // 3
        c = k % 3
        ax = fig4.add_subplot(gs4[r, c])
        axes.append(ax)

        dens_k = dens_all[:, :, k]
        im = ax.imshow(
            dens_k,
            origin="lower",
            aspect="auto",
            extent=[xgrid.min(), xgrid.max(), times.min(), times.max()],
            interpolation="nearest",
            norm=norm,
            cmap="inferno",
        )

        if min_positions is not None and min_positions.size:
            mins = np.atleast_1d(min_positions)
            gmins = np.atleast_1d(global_min_positions) if global_min_positions is not None else np.array([])
            for xm in mins:
                col = "yellow"
                if gmins.size and np.any(np.isclose(xm, gmins, atol=1e-12)):
                    col = "green"
                ax.axvline(
                    float(xm),
                    color=col,
                    linestyle="--",
                    linewidth=0.7,
                    alpha=0.9,
                )

        if barrier_positions is not None and barrier_positions.size:
            for xb in np.atleast_1d(barrier_positions):
                ax.axvline(
                    float(xb),
                    color="gray",
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.9,
                )

        ax.set_title(rf"Autostato {k}", fontsize=9)

        if r == 1:
            ax.set_xlabel("x")
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel("t")
        else:
            ax.set_yticklabels([])

    if im is not None:
        cbar = fig4.colorbar(
            im,
            ax=axes,
            orientation="horizontal",
            fraction=0.03,
            aspect=50,
            pad=0.12,
        )
        cbar.set_label("Densità di probabilità")

    return fig4


def report_page_modellino(
    times: np.ndarray,
    energies: np.ndarray,
    model_params: Dict[str, Any],
    meta: Dict[str, Any],
    a_fun: Callable[[float], float],
    b_fun: Callable[[float], float],
) -> plt.Figure:
    """
    Create the "Toy model" comparison page.
    
    This page compares the full simulation against a reduced ("toy") model, when
    available. The reduced model is computed by helper routines in `DATA_ANALYSIS`
    and is meant to provide a lightweight diagnostic for probability flow.
    
    Parameters
    ----------
    psi_num, times, xgrid, meta:
        Same as in `report()`.
    model_params:
        Parameters of the toy model (as returned by `DATA_ANALYSIS`).
    toy_model_out:
        Output structure from the toy model routine, including probabilities vs time.
    """

    domain = meta["domain"]
    L = np.abs(domain[0]-domain[1])
    n_ho = min(7, energies.shape[1]) if energies.ndim == 2 else 7
    model_E = _build_ho_and_free_energies(
        L,
        times,
        a_fun,
        b_fun,
        min_vals=model_params["min_vals"],
        hess_eigs=model_params["hess_eigs"],
        barrier_val=model_params["barrier_val"],
        n_levels_ho=n_ho,
        bc=meta["bc"],
    )

    wells_E: List[np.ndarray] = model_E["wells_E"]
    free_E: np.ndarray = model_E["free_E"]
    Vbar_t: np.ndarray = model_E["Vbar_t"]

    fig5 = plt.figure(figsize=(10.6, 6.8), dpi=300)
    ax5 = fig5.add_subplot(111)

    max_n = max(
        [E.shape[1] for E in wells_E] + [free_E.shape[1]]
    )
    cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(max_n)]

    ls_map: Dict[str, str] = {"free": "-"}
    lw_map: Dict[str, float] = {"free": 1.1}
    styles_wells = [":", "--", "-."]
    for i in range(len(wells_E)):
        key = f"well_{i}"
        ls_map[key] = styles_wells[i % len(styles_wells)]
        lw_map[key] = 1.35

    vb = Vbar_t
    vb09 = 0.8 * vb
    vb10 = vb
    vb11 = 1.2 * vb

    # --- maschera di "esistenza" delle curve (faint + full) ---
    def _visible_mask(label_cls: str, y: np.ndarray) -> np.ndarray:
        finite = np.isfinite(y)
        if label_cls == "free":
            # visibile dove la curva è almeno lievemente trasparente o piena
            return finite & (y >= vb10)
        else:
            # pozzi: visibili sotto la barriera (faint o piena)
            return finite & (y <= vb10)

    # --- raccolta energie di intersezione tra curve del modellino ---
    def _collect_intersections(
        y1: np.ndarray,
        y2: np.ndarray,
        vis_common: np.ndarray,
    ) -> list[float]:
        E_ints: list[float] = []
        diff = y1 - y2
        for k in range(len(times) - 1):
            if not (vis_common[k] and vis_common[k+1]):
                continue
            d0 = diff[k]
            d1 = diff[k+1]
            if not (np.isfinite(d0) and np.isfinite(d1)):
                continue
            if d0 == 0.0:
                E_ints.append(float(y1[k]))
            elif d1 == 0.0:
                E_ints.append(float(y1[k+1]))
            elif d0 * d1 < 0.0:
                t0, t1 = float(times[k]), float(times[k+1])
                if t1 == t0:
                    continue
                a = d0 / (d0 - d1)
                y_int = float(y1[k] + a * (y1[k+1] - y1[k]))
                E_ints.append(y_int)
        return E_ints

    # Lista delle curve del modellino (solo free + HO, niente energie esatte)
    curves: list[tuple[str, int, np.ndarray]] = []
    for n in range(free_E.shape[1]):
        curves.append(("free", n, free_E[:, n]))
    for iw, Ew in enumerate(wells_E):
        for n in range(Ew.shape[1]):
            curves.append((f"well_{iw}", n, Ew[:, n]))

    intersections_E: list[float] = []
    for idx1 in range(len(curves)):
        label1, _, y1 = curves[idx1]
        vis1 = _visible_mask(label1, y1)
        if not np.any(vis1):
            continue
        for idx2 in range(idx1 + 1, len(curves)):
            label2, _, y2 = curves[idx2]
            vis2 = _visible_mask(label2, y2)
            vis_common = vis1 & vis2
            if not np.any(vis_common):
                continue
            intersections_E.extend(_collect_intersections(y1, y2, vis_common))

    # --- plotting delle famiglie (come prima) ---
    def _plot_family(ax, label_cls: str, Y: np.ndarray) -> None:
        for n in range(Y.shape[1]):
            y = Y[:, n]
            finite = np.isfinite(y)

            if label_cls == "free":
                m_vis = (y >= vb10) & finite
            else:
                m_vis = (y <= vb10) & finite

            y_vis = np.ma.masked_where(~m_vis, y)

            color = colors[n % len(colors)]
            ls = ls_map[label_cls]
            lw = lw_map[label_cls]

            # 1) linea faint continua su tutta la regione visibile
            ax.plot(
                times, y_vis,
                linestyle=ls, linewidth=lw,
                color=color, alpha=0.35, zorder=1,
            )

            # 2) sovrappongo solo le parti "full" con alpha più alto
            if label_cls == "free":
                m_full = (y > vb11) & finite
            else:
                m_full = (y < vb09) & finite

            y_full = np.ma.masked_where(~m_full, y)

            ax.plot(
                times, y_full,
                linestyle=ls, linewidth=lw,
                color=color, alpha=0.95, zorder=2,
            )


    _plot_family(ax5, "free", free_E)
    for i in range(len(wells_E)):
        _plot_family(ax5, f"well_{i}", wells_E[i])

    arrays = [free_E] + wells_E

    # energie esatte (grigie) – come prima
    if energies.ndim == 2:
        K_plot = min(6, energies.shape[1])
        for k in range(K_plot):
            ax5.plot(times, energies[:, k],
                     lw=1.2, color="gray", alpha=0.9, zorder=0)
        arrays.append(energies[:, :K_plot])

    # --- nuova logica di zoom verticale ---
    def _set_ylim_from_global(arrays_list: list[np.ndarray]) -> None:
        all_vals = np.concatenate([np.ravel(A) for A in arrays_list])
        finite = all_vals[np.isfinite(all_vals)]
        if finite.size > 0:
            e_min = np.percentile(finite, 2.0)
            e_max = np.percentile(finite, 98.0)
            if not np.isclose(e_min, e_max):
                margin = 0.1 * (e_max - e_min)
                ax5.set_ylim(e_min - margin, e_max + margin)

    ints = np.asarray(intersections_E, float)
    ints = ints[np.isfinite(ints)]

    if ints.size >= 1:
        e_int_min = float(np.min(ints))
        e_int_max = float(np.max(ints))
        if e_int_max > e_int_min:
            span_int = e_int_max - e_int_min
            center   = 0.5 * (e_int_max + e_int_min)
            full_span = 4.0 * span_int
            margin = 0.05 * full_span
            y_low  = center - 0.5 * full_span - margin
            y_high = center + 0.5 * full_span + margin
            ax5.set_ylim(y_low, y_high)
        else:
            # intersezioni quasi degeneri → fallback globale
            _set_ylim_from_global(arrays)
    else:
        # nessuna intersezione nel dominio di visibilità → fallback globale
        _set_ylim_from_global(arrays)

    ax5.set_xlabel("t")
    ax5.set_ylabel("E")
    ax5.grid(True, alpha=0.25)
    ax5.set_title("Energie Modello vs Energie Esatte.")

    handles = [
        Line2D([0], [0], color="black", linestyle=ls_map["free"],
               label="Particella libera"),
    ]
    for i in range(len(wells_E)):
        key = f"well_{i}"
        handles.append(
            Line2D([0], [0], color="black", linestyle=ls_map[key],
                   label=f"Buca {i}")
        )
    ax5.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)

    return fig5


def report_page_modellino_exact_only(
    times: np.ndarray,
    energies: np.ndarray,
) -> plt.Figure:
    """
    Create the "Toy model (exact-only)" page.
    
    This variant renders only the "exact toy model" curves without overlaying the
    numerical/QHD results. It is mainly used for debugging the reduced model itself.
    
    Parameters are the same as `report_page_modellino`.
    """

    fig = plt.figure(figsize=(8.0, 5.0), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    fig.suptitle("Spettro esatto (scala log)", fontsize=12)

    times = np.asarray(times, float)
    E = np.asarray(energies, float)

    if E.ndim == 1:
        ax.plot(times, E, lw=1.0)
    elif E.ndim == 2:
        K = E.shape[1]
        for k in range(K):
            ax.plot(times, E[:, k], lw=1.0)
    else:
        raise ValueError("`energies` deve essere un array 1D o 2D (Nt, K).")

    ax.set_xlabel("t")
    ax.set_ylabel("Energia")

    # Scala logaritmica robusta anche in presenza di energie <= 0:
    # usa una scala "symlog" per evitare problemi numerici.
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.grid(True, alpha=0.3)

    return fig


def make_probability_transfer_pages(
    d: int,
    q: int,
    times: np.ndarray,
    a_fun,
    b_fun,
    f,
    domain,
    n_states: int,
    bc: str = "periodic",
    energies: np.ndarray | None = None,
    eigenvectors: np.ndarray | None = None,
) -> tuple[list[plt.Figure], dict]:

    """
    Generate the "Probability transfer" pages.
    
    The probability-transfer analysis and its corresponding figures are produced in
    `PROBABILITY_TRANSFER`. This helper merely wraps that pipeline and formats its
    outputs for the PDF report.
    
    Parameters
    ----------
    psi_num:
        Saved states array (K, M).
    times:
        Snapshot times (K,).
    xgrid:
        1D grid for plotting.
    meta:
        Simulation metadata dictionary.
    coords:
        Spatial coordinates array (M,d).
    f_vals_flat:
        Potential values on the grid (M,).
    out_min:
        Output dictionary from `find_local_minima_and_basins`.
    """

    # In d > 1 non mostriamo le heatmap spaziali, ma solo i grafici 1D.
    show_heatmaps = (d == 1)
    if n_states < 2:
        raise ValueError("Servono almeno 2 autostati per formare coppie adiacenti.")
    
    figs: list[plt.Figure] = []

    energies = np.asarray(energies, float)
    vecs = np.asarray(eigenvectors, complex)
    if energies.shape[1] < n_states or vecs.shape[2] < n_states:
        raise ValueError("Spettri passati a make_probability_transfer_pages hanno meno stati di n_states.")

    vecs_txk = ensure_vecs_t_x_k(vecs, n_states=n_states).astype(complex, copy=False)
    vecs_aligned = phase_align_no_perm(vecs_txk)


    coords = ComputationalBasisIndex_to_SpatialCoordinates(q, d, domain, bc)
    xgrid = coords.reshape(-1) if coords.ndim == 1 else coords[:, 0]

    t_min_for_pairs: float | None = None

    K = n_states
    pair_ids   = np.arange(K - 1)
    tstars_ref = np.full(K - 1, np.nan)
    DeltaEs_ref= np.full(K - 1, np.nan)
    max_dH_ref = np.full(K - 1, np.nan)
    area_B_ref = np.full(K - 1, np.nan)
    ratio_sched_ref = np.full(K - 1, np.nan)
    max_dH_over_DE2_ref = np.full(K - 1, np.nan)

    D2_op, V_op = build_D2_V(d, q, domain, f, bc)
    K_use = min(20, n_states)
    c_global = integrate_c_global(times, energies, vecs_aligned, D2_op, V_op, a_fun, b_fun, K_use=K_use)

    for m in range(K - 1):
        i, j = m, m + 1

        res = refine_pair_window_and_analyze(
            i, j,
            d=d, q=q,
            times_coarse=times, E_coarse=energies, V_coarse_aligned=vecs_aligned,
            f=f, domain=domain, n_states=n_states, xgrid=xgrid,
            edge_frac = 0.05,
            times_full=times, c_global=c_global, D2_op=D2_op, V_op=V_op,
            a_fun=a_fun, b_fun=b_fun,
            t_min_allowed=t_min_for_pairs,
        )


        t_sel   = res["times_sel"]
        psi_i_s = res["psi_i_sel"]
        psi_j_s = res["psi_j_sel"]
        dE_s    = res["dE_sel"]
        dH_s    = res["dH_sel"]
        B_s     = res["B_sel"]
        t_star  = res["t_star"]
        DeltaE_ = res["DeltaE_star"]
        tstars_ref[m]  = t_star

        # aggiorna il vincolo per la coppia successiva:
        # i crossing successivi verranno cercati solo per t >= t_star corrente.
        if np.isfinite(t_star):
            if t_min_for_pairs is None:
                t_min_for_pairs = 0.8 * float(t_star)
            else:
                t_min_for_pairs = 0.8 * max(t_min_for_pairs, float(t_star))
        # -----------------------------
        # FIGURA PER LA COPPIA (i, j)
        # -----------------------------
        fig = plt.figure(figsize=(11.7, 8.3), dpi=300, constrained_layout=True)

        # Griglia ESTERNA:
        # - riga 0: tutto il contenuto (heatmap + 3 grafici a destra)
        # - riga 1: riga sottile per la colorbar orizzontale
        width_ratios = [1.0, 1.3] if show_heatmaps else [0.05, 1.0]
        gs_main = fig.add_gridspec(
            nrows=2, ncols=2,
            height_ratios=[1.0, 0.02],      # riga bassa molto sottile
            width_ratios=width_ratios,
        )

        # --- COLONNA SINISTRA (riga alta): 2 heatmap UNA SOPRA L'ALTRA ---
        gs_left = gs_main[0, 0].subgridspec(
            nrows=2, ncols=1,
            hspace=0.0                      # attaccate verticalmente
        )
        ax_hi = fig.add_subplot(gs_left[0, 0])
        ax_hj = fig.add_subplot(gs_left[1, 0])

        # --- COLONNA DESTRA (riga alta): 3 grafici con STESSA ALTEZZA ---
        gs_right = gs_main[0, 1].subgridspec(
            nrows=3, ncols=1,
            hspace=0.15                     # spazio tra i tre pannelli
        )
        ax_gap_full = fig.add_subplot(gs_right[0, 0])  # ΔE(t) globale
        ax_mix      = fig.add_subplot(gs_right[1, 0])  # ΔE + |<m|Ḣ|n>|
        ax_term     = fig.add_subplot(gs_right[2, 0])  # |B| + |c_i|^2,|c_j|^2

        # --- Riga bassa: colorbar orizzontale sotto le 2 heatmap ---
        ax_cb = fig.add_subplot(gs_main[1, 0])
        # la cella [1,1] non ci serve
        ax_dummy = fig.add_subplot(gs_main[1, 1])
        ax_dummy.axis("off")

        if not show_heatmaps:
            ax_hi.set_visible(False)
            ax_hj.set_visible(False)
            ax_cb.set_visible(False)
            ax_dummy.set_visible(False)

        # =========================
        if show_heatmaps:
            #      HEATMAP SINISTRA
            # =========================
            dens_i = np.abs(psi_i_s)**2
            dens_j = np.abs(psi_j_s)**2
            vmax = float(np.nanmax([np.nanmax(dens_i), np.nanmax(dens_j)]))
            norm = PowerNorm(gamma=0.3, vmin=0.0, vmax=vmax)
            extent = [xgrid.min(), xgrid.max(), t_sel.min(), t_sel.max()]

            im_i = ax_hi.imshow(
                dens_i,
                origin="lower", aspect="auto", extent=extent,
                interpolation="nearest", norm=norm, cmap="inferno",
            )
            im_j = ax_hj.imshow(
                dens_j,
                origin="lower", aspect="auto", extent=extent,
                interpolation="nearest", norm=norm, cmap="inferno",
            )

            # Heatmap superiore: niente label/tick sull'asse x
            ax_hi.set_xlabel("")
            ax_hi.set_xticklabels([])
            ax_hi.set_xticks([])
            ax_hi.set_ylabel("t")
            ax_hi.set_title(rf"$|\langle x|{i}_t\rangle|^2$")

            # Heatmap inferiore: asse x completo
            ax_hj.set_xlabel("x")
            ax_hj.set_ylabel("t")
            ax_hj.set_title(rf"$|\langle x|{j}_t\rangle|^2$")

            # Colorbar orizzontale SOTTILE, unica per entrambe
            cbar = fig.colorbar(im_i, cax=ax_cb, orientation="horizontal")
            cbar.ax.tick_params(labelsize=6)

            # =========================
        #    COLONNA DESTRA
        # =========================

        # ΔE(t) su T intero, scala semilog
        dE_full = np.abs(energies[:, j] - energies[:, i])
        dE_plot = np.clip(dE_full, 1e-15, None)

        color_dE = "C0"
        color_dH = "C1"

        ax_gap_full.semilogy(times, dE_plot, lw=1.6, color=color_dE, label=(rf"$E_{{{j}}}-E_{{{i}}}$"))
        ax_gap_full.axvline(t_star, ls="--", lw=1.0, color="gray", alpha=0.6)
        ax_gap_full.set_xlabel("t")
        ax_gap_full.grid(True, which="both", alpha=0.3)
        ax_gap_full.legend(frameon=False)

        # finestra raffinata: ΔE + |<m|Ḣ|n>|
        ax_dE = ax_mix
        ax_dE.plot(t_sel, dE_s, label=rf"$E_{{{j}}}-E_{{{i}}}$", lw=1.8, color=color_dE)
        ax_dE.set_xlabel("t")
        ax_dE.grid(True, alpha=0.25)
        ax_dE.axvline(t_star, ls="--", lw=1.0, color="gray", alpha=0.6)
        ax_dE.tick_params(axis="y", colors=color_dE)
        ax_dE.spines["left"].set_color(color_dE)

        ax_dH = ax_dE.twinx()
        ax_dH.plot(
            t_sel,
            np.abs(dH_s),
            label=rf"$\left|\langle e_{{{i}}}|\dot \hat H|e_{{{j}}}\rangle\right|$",
            lw=1.2,
            ls="--",
            color=color_dH,
        )
        ax_dH.tick_params(axis="y", colors=color_dH)
        ax_dH.spines["right"].set_color(color_dH)

        lines1, labels1 = ax_dE.get_legend_handles_labels()
        lines2, labels2 = ax_dH.get_legend_handles_labels()
        ax_dE.legend(lines1 + lines2, labels1 + labels2,
                     frameon=False)

        # pannello in basso: |B| + |c_i|^2, |c_j|^2
        color_B  = "C2"
        color_ci = "C3"
        color_cj = "C3"

        ax_term.plot(
            t_sel,
            np.abs(B_s),
            lw=1.4,
            label=r"$|a_2|$",
            color=color_B,
        )
        ax_term.set_xlabel("t")
        ax_term.grid(True, alpha=0.25)
        ax_term.tick_params(axis="y", colors=color_B)
        ax_term.spines["left"].set_color(color_B)

        ax_amp = ax_term.twinx()
        if res["ci2_sel"] is not None and res["cj2_sel"] is not None:
            ax_amp.plot(
                t_sel,
                res["ci2_sel"],
                ls="--",
                lw=1.2,
                label=fr"$|c_{{{i}}}|^2$",
                color=color_ci,
            )
            ax_amp.plot(
                t_sel,
                res["cj2_sel"],
                ls=":",
                lw=1.2,
                label=fr"$|c_{{{j}}}|^2$",
                color=color_cj,
            )
            ax_amp.tick_params(axis="y", colors=color_ci)
            ax_amp.spines["right"].set_color(color_ci)

            ylim = max(np.max(res["ci2_sel"]), np.max(res["cj2_sel"]))
            ax_amp.set_ylim(0.0, ylim)

            h1, l1 = ax_term.get_legend_handles_labels()
            h2, l2 = ax_amp.get_legend_handles_labels()
            ax_term.legend(h1 + h2, l1 + l2, frameon=False)
        else:
            ax_amp.set_yticklabels([])
            ax_amp.set_ylabel("")




        figs.append(fig)

        tstars_ref[m]  = t_star
        DeltaEs_ref[m] = DeltaE_
        max_dH_ref[m]  = res["max_dH"]
        area_B_ref[m]  = res["area_B"]

        a_star = float(a_fun(t_star))
        b_star = float(b_fun(t_star))
        ratio_sched_ref[m] = (b_star / a_star) if a_star != 0.0 else np.nan


        if np.isfinite(DeltaE_) and DeltaE_ != 0.0:
            max_dH_over_DE2_ref[m] = res["max_dH"] / (DeltaE_**2)
        else:
            max_dH_over_DE2_ref[m] = np.nan

    figR = plt.figure(figsize=(11.7, 8.3), dpi=300, constrained_layout=True)
    gsR = figR.add_gridspec(3, 2, height_ratios=[2,2,1], width_ratios=[1,1])

    ax1 = figR.add_subplot(gsR[0, 0])
    ax1.plot(pair_ids, tstars_ref, marker='o')
    ax1.set_title(r"Tempo di quasi-crossing $t^*$ per coppia")
    ax1.grid(True, alpha=0.3)

    ax2 = figR.add_subplot(gsR[0, 1])
    ax2.plot(pair_ids, DeltaEs_ref, marker='o')
    ax2.set_title(r"Minima distanza energetica $|\Delta E^*|$ per coppia")
    ax2.grid(True, alpha=0.3)

    ax3 = figR.add_subplot(gsR[1, 0])
    ax3.plot(pair_ids, max_dH_ref, marker='o')
    ax3.set_title(r"$\max_t |\langle m|\dot{H}|m+1\rangle|$ per coppia")
    ax3.grid(True, alpha=0.3)

    ax4 = figR.add_subplot(gsR[1, 1])
    ax4.plot(pair_ids, area_B_ref, marker='o', color="C0", label=r"$A_2$")
    ax4.tick_params(axis = 'y', labelcolor ="C0")
    ax4.grid(True, alpha=0.3)

    ax4b = ax4.twinx()
    ax4b.plot(
        pair_ids,
        max_dH_over_DE2_ref,
        marker='s',
        linestyle="--",
        color="C3",
        label=r"$A_1$"
    )
    ax4b.set_yscale('log')
    ax4b.tick_params(axis = 'y', labelcolor ="C3")

    h1, l1 = ax4.get_legend_handles_labels()
    h2, l2 = ax4b.get_legend_handles_labels()
    ax4.legend(h1 + h2, l1 + l2, frameon=False)
    ax4.set_title("Metriche di adiabaticità per coppia")

    table_rows = []
    for m in range(K - 1):
        pair_label = f"{m}→{m+1}"
        tstar_str  = f"{tstars_ref[m]:.6g}" if np.isfinite(tstars_ref[m]) else "nan"
        ratio_str  = f"{ratio_sched_ref[m]:.6g}" if np.isfinite(ratio_sched_ref[m]) else "nan"
        table_rows.append([pair_label, tstar_str, ratio_str])
    
    axTbl = figR.add_subplot(gsR[2, :])
    axTbl.axis("off")
    the_table = axTbl.table(
        cellText=table_rows,
        colLabels=["Coppia (m→n)", r"$t^*$", r"$b(t^*)/a(t^*)$"],
        loc="center"
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.0, 1.3)

    figs.append(figR)


    summary = {
        "pair_ids":        pair_ids,
        "t_star":          tstars_ref,
        "DeltaE_star":     DeltaEs_ref,
        "max_dH":          max_dH_ref,
        "area_B":          area_B_ref,
        "ratio_sched":     ratio_sched_ref,
        "max_dH_over_DE2": max_dH_over_DE2_ref,
    }
    return figs, summary


# -------------------------------------------------------------------
# Funzione principale report()
# -------------------------------------------------------------------

def report(
    psi_num: np.ndarray,
    times: np.ndarray,
    xgrid: np.ndarray,
    meta: Dict[str, Any],
    f: Callable[[np.ndarray], np.ndarray],
    a_fun: Callable[[float], float] | None = None,
    b_fun: Callable[[float], float] | None = None,
    pages: List[int] | None = None,
):
    """
    Build and save a multi-page PDF report for a single simulation run.
    
    This is the top-level entry point called by the batch runner. It orchestrates:
    - potential analysis (minima, basins, Hessian eigenvalues),
    - optional diagonalization and spectral diagnostics,
    - optional probability-transfer diagnostics,
    - PDF assembly and file naming via `save_report_pdf`.
    
    Parameters
    ----------
    psi_num:
        Saved simulation states of shape (K, M).
    times:
        Snapshot times, shape (K,).
    xgrid:
        1D grid used for plotting (shape (M,) for d=1; for d>1, first axis).
    meta:
        Metadata dictionary describing the run configuration.
    f:
        Potential function values evaluator (callable: coords (M,d) -> (M,)).
    a_fun, b_fun:
        Schedule callables. If omitted, the function expects `meta` to contain
        LaTeX descriptions only; some pages will be skipped.
    pages:
        Optional list of page indices to include (1-based), used for debugging.
    
    Returns
    -------
    None
        The report is written to disk by `save_report_pdf`.
    """

    d = int(meta["d"])
    q = int(meta["q"])
    domain: Tuple[float, float] = tuple(meta["domain"])
    bc = str(meta.get("bc", "periodic"))

    if a_fun is None:
        a_fun = lambda t: 1.0
    if b_fun is None:
        b_fun = lambda t: 1.0

    pages_set = set(pages) if pages is not None else None

    def _want(page_num: int) -> bool:
        return (pages_set is None) or (page_num in pages_set)

    # --- oggetti comuni (calcolati una volta) ---
    coords = ComputationalBasisIndex_to_SpatialCoordinates(q, d, domain, bc)
    f_vals_flat = np.asarray(f(coords), dtype=float).reshape(-1)


    # --- analisi dei minimi e bacini ---
    try:
        out_min = find_local_minima_and_basins(coords, f_vals_flat)
        min_coords = np.asarray(out_min.get("min_coords", []))
        min_vals   = np.asarray(out_min.get("min_vals",   []))
        n_minima_total = int(len(min_vals))

        if min_coords.size > 0:
            try:
                hess_all = np.asarray(
                    hessian_eigvals_at_points(f, min_coords),
                    float,
                )
            except Exception:
                hess_all = None

            # Indici dei minimi globali e secondari
            gidxs = np.asarray(
                out_min.get(
                    "global_min_indices",
                    [int(out_min.get("global_idx", 0))],
                ),
                dtype=int,
            )
            all_idx = np.arange(len(min_vals), dtype=int)

            # garantisce che gli indici globali siano unici
            gidxs = np.unique(gidxs)
            mask_non_global = ~np.isin(all_idx, gidxs)
            sec_idx = all_idx[mask_non_global]

            # ordina i minimi secondari per profondità (valore di f crescente)
            if sec_idx.size:
                sec_order = sec_idx[np.argsort(min_vals[sec_idx])]
                full_order = np.concatenate([gidxs, sec_order])
            else:
                full_order = gidxs

            # tieni solo i primi 8 (la tabella ne mostra comunque al massimo 8)
            order = full_order[:8]

            sel_coords = min_coords[order]
            sel_vals   = min_vals[order]
            if hess_all is not None:
                hess_sel = hess_all[order]
            else:
                hess_sel = np.full((len(order), 1), np.nan, float)

            gset = set(gidxs.tolist())
            minima_table: List[List[str]] = []
            for k, idx in enumerate(order):
                coord_str = ", ".join(
                    f"{c:.3g}" for c in np.atleast_1d(sel_coords[k])
                )
                val_str   = f"{float(sel_vals[k]):.3g}"
                eigs_k    = np.atleast_1d(hess_sel[k])
                # se non abbiamo Hessiana affidabile o è ~0, salta la riga
                if np.all(~np.isfinite(eigs_k)) or np.all(np.abs(eigs_k) < 1e-4):
                    continue
                eig_str   = " / ".join(f"{ev:.3g}" for ev in eigs_k)
                tag = " (global)" if idx in gset else ""
                minima_table.append([coord_str, val_str, eig_str + tag])

            basins_info = {
                "global_basin_idx_flat": np.asarray(
                    out_min.get("global_basin_idx_flat", []),
                    dtype=int,
                ),
                "global_basin_idx_orig": np.asarray(
                    out_min.get("global_basin_idx_orig", []),
                    dtype=int,
                ),
            }

        else:
            minima_table = []
            basins_info = None
            out_min = None
            hess_all = None
            n_minima_total = 0
    except Exception:
        minima_table = []
        basins_info = None
        out_min = None
        hess_all = None
        n_minima_total = 0

    model_params = _compute_model_params_for_ho(
        coords, f_vals_flat, f,
        max_wells=2,
        minima_out=out_min,
        hess_eigs_all=hess_all,
    )

    # --- spettri + autostati globali (una sola diagonalizzazione per tutto il report) ---
    include_probability_transfer = bool(meta.get("include_probability_transfer", True))
    need_spectral = _want(3) or (_want(4) and d == 1) or _want(5)

    # per pag. 3–5
    energies: np.ndarray | None = None
    eigenvectors: np.ndarray | None = None
    components: np.ndarray | None = None


    need_eigensystem = need_spectral or include_probability_transfer

    if need_eigensystem:
        n_states = 10
        energies, eigenvectors = diagonalize(
            d=d,
            q=q,
            times=times,
            a_fun=a_fun,
            b_fun=b_fun,
            n_states=n_states,
            f=f,
            domain=domain,
            bc=bc,
        )
        components = weights(eigenvectors, psi_num)


    figs: list[plt.Figure] = []

    # Page 1
    if _want(1):
        fig1 = report_page_parametri(
            meta=meta,
            f=f,
            a_fun=a_fun,
            b_fun=b_fun,
            coords=coords,
            f_vals_flat=f_vals_flat,
            minima_table=minima_table,
        )
        figs.append(fig1)

    # Page 2
    if _want(2):
        fig2 = report_page_risultati(
            d=d,
            q=q,
            domain=domain,
            meta=meta,
            f=f,
            coords=coords,
            f_vals_flat=f_vals_flat,
            out_min=out_min,
            basins_info=basins_info,
            times=times,
            psi_num=psi_num,
            a_fun=a_fun,
            b_fun=b_fun,
        )
        figs.append(fig2)

    # Page 3
    if _want(3) and components is not None:
        fig3 = report_page_componenti_spettrali(
            times=times,
            weights=weights,
        )
        figs.append(fig3)

    # Page 4 (1D only)
    if _want(4) and d == 1 and eigenvectors is not None:
        fig4 = report_page_autostati(
            d=d,
            times=times,
            xgrid=xgrid,
            coords=coords,
            f_vals_flat=f_vals_flat,
            eigenvectors=eigenvectors,
            out_min=out_min,
        )
        if fig4 is not None:
            figs.append(fig4)

    # Page 5
    if _want(5) and energies is not None:
        try:
            if n_minima_total == 2:
                fig5 = report_page_modellino(
                    times=times,
                    energies=energies,
                    model_params=model_params,
                    meta=meta,
                    a_fun=a_fun,
                    b_fun=b_fun,
                )
            else:
                fig5 = report_page_modellino_exact_only(
                    times=times,
                    energies=energies,
                )
            figs.append(fig5)
        except Exception as exc:
            print(f"[WARN] Page 5 (modellino HO/libero) saltata: {exc}")

    # Blocchi probability_transfer
    if (
        include_probability_transfer
        and n_states > 1
        and energies is not None
        and eigenvectors is not None
    ):

        figs_PT, summary_PT = make_probability_transfer_pages(
            d=d,
            q=q,
            times=times,
            a_fun=a_fun,
            b_fun=b_fun,
            f=f,
            domain=domain,
            n_states=n_states,
            bc=bc,
            energies=energies,
            eigenvectors=eigenvectors,
        )

        figs.extend(figs_PT)

    path = save_report_pdf(figs, meta, times=times)
    for ffig in figs:
        plt.close(ffig)
    return path
