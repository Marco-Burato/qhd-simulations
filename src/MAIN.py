"""
Batch runner for QHD / numerical simulations on benchmark potentials.

Key features
------------
- Explicit parameter specification (d, q, T, dt, integrator order, benchmark potential).
- Time schedules defined once as mathematical functions: a(t), b(t).
- Initial state defined once as a mathematical function of the spatial coordinates.
- LaTeX formulas for schedules are stored in `meta["schedule_a_tex"]` and
  `meta["schedule_b_tex"]` and used by the reporting pipeline (overview page).
- Batch generation via `itertools.product`.

Notes
-----
This module is intentionally "thin": it wires together the existing simulation engine
(`numerical_simulation`), potential-resolution utilities, snapshot schedules, optional
data persistence, and the reporting pipeline. Numerical behavior is preserved.
"""

from __future__ import annotations

import itertools as it
from typing import Any, Callable, Dict, Iterable

import numpy as np

from NUMERICAL_SIMULATION import numerical_simulation
from REPORT import report
from SAVES import data_save
from UTILITIES import (
    ComputationalBasisIndex_to_SpatialCoordinates,
    make_snapshot_schedule,
    resolve_potential,
)

# ======================================================================
# Global schedules
# ======================================================================


def build_schedule_callables() -> tuple[
    Callable[[float], float],
    Callable[[float], float],
    str,
    str,
]:
    """
    Construct the schedule functions a(t), b(t) and their LaTeX strings.

    Returns
    -------
    a_fun, b_fun:
        Callables implementing the schedules.
    a_tex, b_tex:
        LaTeX strings used for reporting/documentation.
    """

    def a_fun(t: float) -> float:
        return 1.0 * (2.0 / (t**1.0 + 0.001))

    def b_fun(t: float) -> float:
        return 1.0 * (2.0 * t**1.0)

    a_tex = r"$\frac{2}{t + 0.001}$"
    b_tex = r"$2 t$"
    return a_fun, b_fun, a_tex, b_tex


def build_initial_state_callables() -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    """
    Construct the initial state ψ0 and its LaTeX representation.

    Returns
    -------
    psi0_fun:
        Callable mapping coords with shape (M, d) to a complex state vector of shape (M,).
    psi0_tex:
        LaTeX string describing the initialization.

    Notes
    -----
    The current initialization is uniform (constant amplitude) over the grid.
    The intermediate variable `L` is kept as in the original code for full
    behavioral equivalence, even though it is not used downstream.
    """
    def psi0_fun(coords: np.ndarray) -> np.ndarray:
        x = coords[:, 0]
        L = (x[1] - x[0]) * len(x) - x[0]  # computed but not used (kept intentionally)
        psi = 0 * x + 1.0
        psi = psi.astype(np.complex128)
        return psi

    psi0_tex = r"$\propto 1$"
    return psi0_fun, psi0_tex


# ======================================================================
# Single run
# ======================================================================


def run_single_simulation(
    f_item,  # benchmark name (str) or callable
    *,
    d: int,
    q: int,
    T: float,
    dt: float,
    order: int,
    bc: str,
    n_snapshots: int = 101,
    save: bool,
) -> None:
    """
    Run one simulation instance, optionally save raw outputs, and generate a report.

    Parameters
    ----------
    f_item:
        Benchmark potential identifier:
          - string key understood by `resolve_potential`, or
          - a callable potential (if supported by `resolve_potential`).
    d:
        Spatial dimension.
    q:
        Qubits per axis. The number of grid points per axis is N = 2**q.
    T:
        Final simulation time.
    dt:
        Integrator time step.
    order:
        Integrator order forwarded to `numerical_simulation` (e.g., 1, 2, 4, ...).
    bc:
        Boundary conditions ("periodic" or "dirichlet").
    n_snapshots:
        Number of snapshot times used for saving/plotting and reporting. This does not
        alter the integrator step size; it only selects which times are recorded.
    save:
        If True, persist raw arrays and metadata via `data_save`.
    """
    # --- Resolve potential function, domain, and LaTeX description ---
    f_values, domain, f_tex = resolve_potential(f_item)

    if isinstance(f_item, str):
        f_name = f_item
    else:
        f_name = getattr(f_item, "__name__", "custom_potential")

    d = int(d)
    q = int(q)
    T = float(T)
    dt = float(dt)

    # --- Spatial coordinates of the computational basis ---
    coords = ComputationalBasisIndex_to_SpatialCoordinates(q, d, domain, bc)
    xgrid = coords.reshape(-1) if coords.ndim == 1 else coords[:, 0]

    # --- Schedule functions + LaTeX metadata ---
    a_fun, b_fun, a_tex, b_tex = build_schedule_callables()

    # --- Initial state function + LaTeX metadata ---
    psi0_fun, psi0_tex = build_initial_state_callables()

    # --- Snapshot schedule (used for reporting and optional saving) ---
    snap_idx, snap_times, nsteps, _ = make_snapshot_schedule(T, dt, n_snapshots=n_snapshots)

    # --- Metadata payload for reproducibility and reporting ---
    meta: Dict[str, Any] = {
        "d": d,
        "q": q,
        "T": T,
        "dt": dt,
        "order": order,
        "f": f_name,
        "domain": domain,
        "f_tex": f_tex,
        "initialization": psi0_tex,
        "bc": bc,
        "n_snapshots": int(n_snapshots),
        "nsteps": int(nsteps),
        "schedule_a_tex": a_tex,
        "schedule_b_tex": b_tex,
        "include_probability_transfer": False,
    }

    # --- Numerical simulation ---
    psi_num = numerical_simulation(
        d=d,
        q=q,
        T=T,
        dt=dt,
        a_fun=a_fun,
        b_fun=b_fun,
        f=f_values,
        domain=domain,
        order=order,
        n_snapshots=n_snapshots,
        bc=bc,
        psi0_fun=psi0_fun,
    )

    # ==================================================================
    # Raw data saving + report generation
    # ==================================================================

    if save:
        data_save(psi_num=psi_num, times=snap_times, xgrid=xgrid, meta=meta)

    report(
        psi_num=psi_num,
        times=snap_times,
        xgrid=xgrid,
        meta=meta,
        f=f_values,
        a_fun=a_fun,
        b_fun=b_fun,
    )


# ======================================================================
# Batch runner
# ======================================================================


def run_batch_simulations(
    *,
    d_list: Iterable[int],
    q_list: Iterable[int],
    T_list: Iterable[float],
    dt_list: Iterable[float],
    order_list: Iterable[int],
    bc_list: Iterable[str],
    f_list: Iterable,  # list of benchmark names (str) or callables
    n_snapshots: int = 101,
    save: bool,
) -> None:
    """
    Run a batch of simulations over the Cartesian product of parameter lists.

    Parameters
    ----------
    d_list, q_list, T_list, dt_list, order_list, bc_list:
        Iterables of parameter values.
    f_list:
        Iterable of benchmark identifiers (strings) or callables.
    n_snapshots:
        Number of snapshots per run (forwarded to `run_single_simulation`).
    save:
        If True, save raw outputs for each run via `data_save`.
    """
    for (d, q, T, dt, order, bc, f_item) in it.product(
        d_list,
        q_list,
        T_list,
        dt_list,
        order_list,
        bc_list,
        f_list,
    ):
        print(
            f"\n=== Simulation: f={f_item}, d={d}, q={q}, "
            f"T={T}, dt={dt}, order={order}, bc={bc} ==="
        )
        run_single_simulation(
            f_item=f_item,
            d=int(d),
            q=int(q),
            T=float(T),
            dt=float(dt),
            order=int(order),
            bc=str(bc),
            n_snapshots=int(n_snapshots),
            save=save,
        )


if __name__ == "__main__":
    # Default batch configuration (edit as needed).
    d_list = [1]
    q_list = [8]
    T_list = [100.0]
    dt_list = [1e-3]
    order_list = [1]
    bc_list = ["periodic"]  # "periodic" or "dirichlet"
    f_list = ["periodic_ripples"]
    save = False

    run_batch_simulations(
        d_list=d_list,
        q_list=q_list,
        T_list=T_list,
        dt_list=dt_list,
        order_list=order_list,
        bc_list=bc_list,
        f_list=f_list,
        n_snapshots=101,
        save=save,
    )
