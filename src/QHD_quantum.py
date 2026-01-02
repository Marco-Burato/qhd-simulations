"""Quantum Hamiltonian Descent (QHD) — research reference implementation.

This script collects a self-contained set of routines to simulate Quantum
Hamiltonian Descent on a discretized spatial grid. It provides:

- an **offline** mode, where the objective f(x) is precomputed once and then used
  as a diagonal potential during time evolution;
- an **online** mode, where f(x) is evaluated by a quantum subroutine at each
  time step (CPTP map on the simulation register);
- a **stochastic online** mode based on mid-circuit measurements (MCM) and reset.

The configuration block at the top of the file is intended as the single entry
point for reproducible experiments.

Dependencies: project-local modules `UTILITIES` and `FUNCTION_ANALYSIS`.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Literal, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import pennylane as qml
from time import perf_counter

from UTILITIES import (
    ComputationalBasisIndex_to_SpatialCoordinates,
    make_snapshot_schedule,
)
from FUNCTION_ANALYSIS import find_local_minima_and_basins


# ============================================================
# ============================================================
# 0. TOP-LEVEL CONFIGURATION
# ============================================================

RUN_OFFLINE: bool = True
RUN_ONLINE: bool = True                 # online (CPTP map on S, uses default.mixed)
RUN_ONLINE_STOCH: bool = False           # stochastic online (mid-circuit measurement + reset)


@dataclass(frozen=True)
class OfflinePrecomputeConfig:
    """Configuration for the offline precomputation estimator.

    Parameters
    ----------
    estimator
        - "expval": analytic expectation values (shots=None).
        - "shots": shot-based estimator (finite shots) to emulate sampling noise.
    shots
        Number of shots *per circuit evaluation* when estimator="shots".
    seed
        Optional RNG seed forwarded to the PennyLane device when estimator="shots".
    """
    estimator: Literal["expval", "shots"] = "shots"
    shots: int = 1000
    seed: Optional[int] = 0


# Controls how f(x_r) is estimated during the offline precomputation.
OFFLINE_PRECOMPUTE: OfflinePrecomputeConfig = OfflinePrecomputeConfig(
    estimator="shots",
    shots=11,
    seed=0,
)


STOCH_SHOTS: int = 100            # shots per time step in stochastic online mode
STOCH_SEED: Optional[int] = 0     # RNG seed for stochastic online mode

DEVICE: str = "lightning.qubit"

# Spatial dimension (number of coordinates of the domain)
D: int = 1

# Spatial domain [a, b]
DOMAIN: Tuple[float, float] = (0.0, 2.0 * np.pi)

# Qubits per spatial dimension (N = 2**Q_PER_DIM grid points per axis)
Q_PER_DIM: int = 6

# Number of qubits in the measurement register M (used to define the quantum objective f(x))
N_MEAS: int = 1

# Total simulation time and time step
T_TOTAL: float = 8.0
DT: float = 0.001

# Number of snapshots to store (including t=0 and t=T)
N_SNAPSHOTS: int = 51
if T_TOTAL / DT % (N_SNAPSHOTS - 1) != 0:
    raise ValueError("(T_TOTAL/DT) must be a multiple of (N_SNAPSHOTS-1).")
CHUNK_SIZE = int(T_TOTAL / DT / (N_SNAPSHOTS-1))
# Boundary conditions
BOUNDARY_CONDITIONS: str = "periodic"       # Dirichlet BC are not implemented yet

# Time-splitting order (Trotter/Suzuki)
SPLIT_ORDER: int = 1                        # higher-order splittings are not implemented yet

# Time-dependent schedules a(t) and b(t)
def A_SCHEDULE(t: float) -> float:
    """Coefficient a(t) multiplying the kinetic term in the QHD Hamiltonian."""
    return 2.0 / (t**3 + 0.001)

def B_SCHEDULE(t: float) -> float:
    """Coefficient b(t) multiplying the objective/potential term in the QHD Hamiltonian."""
    return 2.0 * t**3


# --- Measurement-state preparation and observable defining f(x) ---
def prepare_measure_state(sim_wires: list[int], meas_wires: list[int]) -> None:
    """Prepare the measurement register M conditioned on the simulation register S.

    This unitary implements the map |x⟩_S |0⟩_M → |x⟩_S |ψ(x)⟩_M. The state
    |ψ(x)⟩_M is then measured to define the quantum objective value f(x).

    Notes
    -----
    The default implementation below is a minimal example (a single controlled
    rotation). Replace it with the task-specific W_SM used in your experiments.
    """ 
    delta = DOMAIN[1] - DOMAIN[0]
    for j, w in enumerate(sim_wires): 
        angle = delta / (2 ** (j + 1)) 
        qml.CRY(angle, wires=[w, meas_wires[0]])


def observable_factory(meas_wires: list[int]) -> qml.operation.Operator:
    """Return the observable O_M measured on the register M to obtain f(x).

    The returned operator must act only on `meas_wires`.
    """
    return qml.PauliZ(meas_wires[0])


def f_exact(coords: np.ndarray) -> np.ndarray:
    """
    Exact classical expression for f(x).
    
    This function is used as a reference to:
    - identify the basin of attraction of the global minimum;
    - compute ⟨f⟩(t) and the convergence coefficient E(t).
    """
    coords = np.asarray(coords, float)
    assert coords.ndim == 2, "coords deve essere (M, d)."
    x1 = coords[:, 0]
    return np.cos(x1)


# --- Initial state preparation for the simulation register ---
def prepare_initial_state(sim_wires: list[int]) -> None:
    """Prepare the initial state of the simulation register S."""
    for w in sim_wires:
        qml.Hadamard(wires=w)


# ============================================================
# 1. PARTE COMUNE: ANALISI, PLOT, ECC.
# ============================================================


def prepare_initial_state_vector(
    d: int,
    q: int,
    prepare_initial_state: Callable[[list[int]], None],
) -> np.ndarray:
    """
    Build the initial state vector |ψ₀⟩ on the simulation register S.
    
    Parameters
    ----------
    d, q
        Spatial dimension and number of qubits per spatial dimension.
    prepare_initial_state
        State-preparation routine with signature `prepare_initial_state(sim_wires)`.
    
    Returns
    -------
    psi0
        Complex state vector of shape (2**(d*q),) representing the initial state of S.
    """
    n_sim = d * q
    sim_wires = list(range(n_sim))
    dev_init = qml.device(DEVICE, wires=n_sim, shots=None)

    @qml.qnode(dev_init, interface="numpy", diff_method=None)
    def _prep():
        prepare_initial_state(sim_wires)
        return qml.state()

    psi0 = _prep()
    return np.asarray(psi0, dtype=complex)


def precompute_kinetic_Ldiag(q: int, domain: Tuple[float, float]) -> np.ndarray:
    """
    Precompute the 1D kinetic spectrum for a periodic grid.
    
    Returns the diagonal entries L(k) = 1/2 k^2 with centered momentum indices.
    The result is reusable for each spatial axis.
    """
    N = 2**q
    L = float(domain[1] - domain[0])
    r = np.arange(N, dtype=int)
    m = r - (N // 2)
    k = (2.0 * np.pi / L) * m
    return 0.5 * (k**2)


def apply_kinetic_block(
    *,
    dt: float,
    a_t: float,
    dim_wires: list[list[int]],
    L_diag: np.ndarray,
) -> None:
    """Apply the kinetic block U_K on S (QFT + diagonal phase in momentum space)."""
    phase_K = np.exp(-1j * dt * a_t * L_diag).astype(np.complex128)
    for wj in dim_wires:
        qml.PauliZ(wires=wj[-1])
        qml.adjoint(qml.QFT)(wires=wj)
        qml.DiagonalQubitUnitary(phase_K, wires=wj)
        qml.QFT(wires=wj)
        qml.PauliZ(wires=wj[-1])


def find_global_basin_indices(
    coords: np.ndarray,
    f_vals_exact: np.ndarray,
) -> np.ndarray:
    """
    Identify the basin of the global minimum.
    
    Uses `FUNCTION_ANALYSIS.find_local_minima_and_basins` to locate basins and
    returns the indices (with respect to `coords`) belonging to the basin of the
    global minimum.
    """
    info = find_local_minima_and_basins(coords, f_vals_exact)
    basin_idx = info["global_basin_idx_orig"]
    return np.asarray(basin_idx, dtype=int)


def compute_success_probability(
    probs: np.ndarray,
    basin_idx: np.ndarray,
) -> np.ndarray:
    """Compute Psuc(t) = ∑_{x in global basin} p(x, t)."""
    probs = np.asarray(probs, float)
    basin_idx = np.asarray(basin_idx, int)
    return np.sum(probs[:, basin_idx], axis=1)


def compute_E_coefficient(
    probs: np.ndarray,
    f_vals_exact: np.ndarray,
) -> np.ndarray:
    """
    Compute the convergence coefficient E(t):
    
        E(t) = (⟨f⟩(t) - f_min) / (⟨f⟩(0) - f_min),
    
    where ⟨f⟩(t) = ∑_x f(x) p(x, t) and f_min is the global minimum of f(x).
    """
    probs = np.asarray(probs, float)
    f_vals_exact = np.asarray(f_vals_exact, float)
    f_min = np.min(f_vals_exact)

    exp_f0 = probs[0] @ f_vals_exact
    denom = exp_f0 - f_min
    if np.isclose(denom, 0.0):
        return np.zeros_like(probs[:, 0])

    exp_f = probs @ f_vals_exact  # (K,)
    E = (exp_f - f_min) / denom
    return E


def kolmogorov_fidelity(p: np.ndarray, q: np.ndarray) -> float:
    """
    Kolmogorov fidelity between two probability distributions.
    
    Given p and q, the Kolmogorov fidelity is defined as:
        F_K(p, q) = (∑_i √(p_i q_i))^2
    """
    p = np.asarray(p, float)
    q = np.asarray(q, float)

    if p.shape != q.shape:
        raise ValueError(
            f"kolmogorov_fidelity: forme diverse p.shape={p.shape}, q.shape={q.shape}."
        )

    # Correzione numerica: tronca sotto zero ed assicura normalizzazione.
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)

    sp = p.sum()
    sq = q.sum()
    if sp <= 0.0 or sq <= 0.0:
        raise ValueError("kolmogorov_fidelity: somma di probabilità nulla o negativa.")

    p /= sp
    q /= sq

    tv_distance = 0.5 * np.abs(p - q).sum()
    F = 1.0 - tv_distance
    return float(F)


def compute_kolmogorov_fidelity_vs_time(
    probs_off: np.ndarray,
    probs_on: np.ndarray,
) -> np.ndarray:
    """
    Compute Kolmogorov fidelity vs time with respect to a reference distribution.
    
    Returns an array F_K(t_k) computed at the snapshot times.
    """
    probs_off = np.asarray(probs_off, float)
    probs_on = np.asarray(probs_on, float)

    if probs_off.shape != probs_on.shape:
        raise ValueError(
            "compute_kolmogorov_fidelity_vs_time: "
            f"forme diverse probs_off.shape={probs_off.shape}, "
            f"probs_on.shape={probs_on.shape}."
        )

    K, M = probs_off.shape
    F = np.empty(K, dtype=float)
    for k in range(K):
        F[k] = kolmogorov_fidelity(probs_off[k], probs_on[k])
    return F


def plot_kolmogorov_fidelity(
    times: np.ndarray,
    F: np.ndarray,
) -> plt.Figure:
    """Plot Kolmogorov fidelity as a function of time."""
    times = np.asarray(times, float)
    F = np.asarray(F, float)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(times, F, label=r"$F$")
    ax.set_xlabel("t")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig("Fidelity_OnOff.pdf", format='pdf', bbox_inches='tight')
    return fig


def plot_Psuc_and_E(
    times: np.ndarray,
    Psuc_off: np.ndarray,
    E_off: np.ndarray,
    Psuc_on: np.ndarray,
    E_on: np.ndarray,
) -> plt.Figure:
    """Plot Psuc(t) and E(t) on a shared time axis."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(times, Psuc_off, label="Offline")
    ax[0].plot(times, Psuc_on, label="Online", linestyle="--")
    ax[0].set_xlabel("t", loc = 'right')
    ax[0].set_title(r"$P_{suc}$")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(times, E_off, label="Offline")
    ax[1].plot(times, E_on, label="Online", linestyle="--")
    ax[1].set_xlabel("t", loc = 'right')
    ax[1].set_title(r"$\mathbb{E}$")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig("Risultati_QHD.pdf", format='pdf', bbox_inches='tight')
    return fig


def plot_Psuc_and_E_single(times, Psuc, E, *, label: str) -> plt.Figure:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(times, Psuc, label=label)
    ax[0].set_xlabel("t", loc="right")
    ax[0].set_title(r"$P_{suc}$")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(times, E, label=label)
    ax[1].set_xlabel("t", loc="right")
    ax[1].set_title(r"$\mathbb{E}$")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"Risultati_QHD_{label}.pdf", format="pdf", bbox_inches="tight")
    return fig


def plot_probability_heatmap(
    times: np.ndarray,
    coords: np.ndarray,
    probs: np.ndarray,
    *,
    title: str = r"$|\psi(x,t)|^2$",
    filename: str ="Heatmap.pdf"
) -> plt.Figure:
    """
    Plot a heatmap of the probability density |ψ(x, t)|² vs time and space.
    
    Parameters
    ----------
    times
        Snapshot times, shape (K,).
    coords
        Spatial coordinates, shape (M, d). If d>1, probabilities are marginalized
        over all other dimensions and the heatmap shows the marginal over the first
        coordinate.
    probs
        Probabilities |ψ(x_r, t_k)|², shape (K, M).
    """
    coords = np.asarray(coords, float)
    xgrid = coords[:, 0]
    probs = np.asarray(probs, float)

    vmax = float(np.nanmax(probs))
    norm = PowerNorm(gamma=0.3, vmin=0.0, vmax=vmax)
    extent = [xgrid.min(), xgrid.max(), times.min(), times.max()]

    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)

    if coords.shape[0] != probs.shape[1]:
        raise ValueError("coords e probs non sono compatibili (dimensione M diversa).")

    x = coords[:, 0]

    extent = [x.min(), x.max(), times.min(), times.max()]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(probs, origin="lower", aspect="auto", extent=extent,
        interpolation="nearest", norm=norm, cmap="inferno")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    fig.tight_layout()

    fig.savefig(filename, format='pdf', bbox_inches='tight')
    return fig


def plot_heatmaps(
    probs_off: np.ndarray,
    probs_on: np.ndarray,
    times: np.ndarray,
    coords: np.ndarray,
):  
    """Convenience wrapper to generate one or more probability heatmaps from stored results."""

    coords = np.asarray(coords, float)
    xgrid = coords[:, 0]
    probs_on = np.asarray(probs_on, float)
    probs_off = np.asarray(probs_off, float)

    vmax = float(np.nanmax([np.nanmax(probs_off), np.nanmax(probs_on)]))
    norm = PowerNorm(gamma=0.3, vmin=0.0, vmax=vmax)
    extent = [xgrid.min(), xgrid.max(), times.min(), times.max()]

    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    
    fig.subplots_adjust(bottom=0.2)

    imL = ax_left.imshow(probs_off, origin="lower", aspect="auto", extent=extent,
        interpolation="nearest", norm=norm, cmap="inferno")
    
    ax_left.set_xlabel("x")
    ax_left.set_ylabel("t")
    ax_left.set_title("Densità Offline")

    imR = ax_right.imshow(probs_on, origin="lower", aspect="auto", extent=extent,
        interpolation="nearest", norm=norm, cmap="inferno")
    ax_right.set_xlabel("x")
    ax_right.set_ylabel("t")
    ax_right.set_title("Densità Online")

    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.015]) 
    
    cbar = fig.colorbar(imL, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=8)

    fig.savefig("Confronto_heatmaps.pdf", format='pdf', bbox_inches='tight')

    return fig

def plot_Psuc_and_E_multi(
    times: np.ndarray,
    series: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    filename: str = "Risultati_QHD_MULTI.pdf",
) -> plt.Figure:
    """Plot Psuc(t) and E(t) for multiple simulation modes on shared axes."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    for label, Psuc, E in series:
        ax[0].plot(times, Psuc, label=label)
        ax[1].plot(times, E, label=label)

    ax[0].set_xlabel("t", loc="right")
    ax[0].set_title(r"$P_{suc}$")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].set_xlabel("t", loc="right")
    ax[1].set_title(r"$\mathbb{E}$")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(filename, format="pdf", bbox_inches="tight")
    return fig


def plot_heatmaps_multi(
    probs_list: list[np.ndarray],
    titles: list[str],
    times: np.ndarray,
    coords: np.ndarray,
    *,
    filename: str = "Confronto_heatmaps_MULTI.pdf",
) -> plt.Figure:
    """Plot side-by-side heatmaps (typically N=2 or N=3) with a shared colorbar."""
    if len(probs_list) != len(titles):
        raise ValueError("plot_heatmaps_multi: probs_list e titles devono avere stessa lunghezza.")

    coords = np.asarray(coords, float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)

    xgrid = coords[:, 0]
    vmax = float(np.nanmax([np.nanmax(p) for p in probs_list]))
    norm = PowerNorm(gamma=0.3, vmin=0.0, vmax=vmax)
    extent = [xgrid.min(), xgrid.max(), times.min(), times.max()]

    n = len(probs_list)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1:
        axes = [axes]

    fig.subplots_adjust(bottom=0.2)

    ims = []
    for ax, probs, title in zip(axes, probs_list, titles):
        im = ax.imshow(
            probs, origin="lower", aspect="auto", extent=extent,
            interpolation="nearest", norm=norm, cmap="inferno"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)
        ims.append(im)

    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.015])
    cbar = fig.colorbar(ims[0], cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(filename, format="pdf", bbox_inches="tight")
    return fig


def plot_kolmogorov_fidelity_multi(
    times: np.ndarray,
    curves: list[tuple[str, np.ndarray]],
    *,
    filename: str = "Fidelity_MULTI.pdf",
) -> plt.Figure:
    """Plot Kolmogorov fidelity curves for multiple simulation modes."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for label, F in curves:
        ax.plot(times, F, label=label)

    ax.set_xlabel("t")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, format="pdf", bbox_inches="tight")
    return fig


def save_probability_data(
    filename: str,
    times: np.ndarray,
    coords: np.ndarray,
    probs: np.ndarray,
    *,
    extra_metadata: Dict[str, Any] | None = None,
) -> None:
    """
    Save probability data to a compressed NumPy .npz file.
    
    Parameters
    ----------
    filename
        Output filename (typically ending with .npz).
    times
        Snapshot times, shape (K,).
    coords
        Spatial coordinates associated with computational basis states |x_r>, shape (M, d).
    probs
        Probability density |ψ(x_r, t_k)|², shape (K, M).
    extra_metadata
        Optional dictionary of additional metadata saved in the .npz container.
    """
    times = np.asarray(times, float)
    coords = np.asarray(coords, float)
    probs = np.asarray(probs, float)

    data: Dict[str, Any] = {
        "times": times,
        "coords": coords,
        "probs": probs,
    }

    if extra_metadata is not None:
        # Aggiunge eventuali metadati; verranno salvati come scalari/array in npz.
        data.update(extra_metadata)

    np.savez(filename, **data)


# ============================================================
# 2. OFFLINE PROCEDURE: PRECOMPUTATION AND QHD
# ============================================================

def _int_to_bits(r: int, width: int) -> np.ndarray:
    """Convert an integer into a bit array of a fixed width (MSB → LSB)."""
    # MSB -> LSB, coerente con np.binary_repr(..., width=width)
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint64)
    return ((np.uint64(r) >> shifts) & 1).astype(np.int8)

def precompute_f_offline(
    d: int,
    q: int,
    coords: np.ndarray,
    n_meas: int,
    prepare_measure_state: Callable[[list[int], list[int]], None],
    observable_factory: Callable[[list[int]], qml.operation.Operator],
    *,
    device: str = DEVICE,
    precompute_cfg: OfflinePrecomputeConfig = OFFLINE_PRECOMPUTE,
) -> np.ndarray:
    """Precompute the objective values f(x_r) for all computational basis points.

    The function evaluates the objective circuit once per computational basis
    state |x_r⟩ of the simulation register S and stores the resulting scalar
    f(x_r). The estimator can be configured to use analytic expectation values
    (precompute_cfg.estimator="expval", shots=None) or a shot-based estimator
    (precompute_cfg.estimator="shots") to emulate sampling noise.

    Parameters
    ----------
    d, q
        Spatial dimension and number of qubits per spatial dimension.
    coords
        Coordinates with shape (2**(d*q), d), ordered consistently with the
        computational basis indexing.
    n_meas
        Number of qubits in the measurement register M.
    prepare_measure_state
        Callable implementing W_SM.
    observable_factory
        Callable returning the observable O_M to be measured on M.
    device
        PennyLane device name (e.g., "lightning.qubit").
    precompute_cfg
        Controls whether to use analytic expectation values or finite shots.

    Returns
    -------
    f_values
        Array of shape (2**(d*q),) containing f(x_r).
    """

    coords = np.asarray(coords, float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)

    M, d_inferred = coords.shape
    if d_inferred != d:
        raise ValueError("coords has an inconsistent second dimension with respect to d.")

    n_sim = d * q
    dim = 2 ** n_sim
    if M != dim:
        raise ValueError(f"Expected M = 2**(d*q). Found M={M}, expected {dim}.")

    shots = None if precompute_cfg.estimator == "expval" else int(precompute_cfg.shots)
    device_kwargs: Dict[str, Any] = {}
    if shots is not None and precompute_cfg.seed is not None:
        device_kwargs["seed"] = int(precompute_cfg.seed)
    dev = qml.device(device, wires=n_sim + n_meas, shots=shots, **device_kwargs)

    sim_wires = list(range(n_sim))
    meas_wires = list(range(n_sim, n_sim + n_meas))
    obs = observable_factory(meas_wires)

    @qml.qnode(dev, interface="numpy", diff_method=None)
    def f_qnode(basis_state: np.ndarray):
        qml.BasisState(basis_state, wires=sim_wires)
        prepare_measure_state(sim_wires, meas_wires)
        return qml.expval(obs)

    f_vals = np.empty(M, dtype=np.float64)
    for r in range(M):
        basis_state = _int_to_bits(r, n_sim)
        f_vals[r] = float(f_qnode(basis_state))

    return f_vals

def qhd_time_evolution_offline(
    d: int,
    q: int,
    T: float,
    dt: float,
    *,
    a_fun: Callable[[float], float],
    b_fun: Callable[[float], float],
    f_values: np.ndarray,
    domain: Tuple[float, float],
    n_snapshots: int = 101,
    bc: str = "periodic",
    ord: int = 1,
    prepare_initial_state: Callable[[list[int]], None],
):
    """Offline QHD evolution using a precomputed objective f(x_r).

    The state is evolved as a pure state on the simulation register S.
    At each time step, the potential term is implemented as a diagonal
    unitary exp(-i dt * b(t) * f(x_r)) using the precomputed array `f_values`.
    The kinetic term is applied via a QFT-based block on each spatial axis.

    Notes
    -----
    Currently limited to periodic boundary conditions (bc="periodic") and
    first-order splitting (ord=1).
    """

    if bc != "periodic":
        raise NotImplementedError
    if ord != 1:
        raise NotImplementedError

    n = d * q
    N = 2 ** q
    dim = 2 ** n

    nsteps = int(round(T / dt))
    if (n_snapshots - 1) <= 0:
        raise ValueError
    if nsteps % (n_snapshots - 1) != 0:
        raise ValueError(
            "With equally spaced snapshots, nsteps must be a multiple of (n_snapshots-1). "
            f"Qui nsteps={nsteps}, n_snapshots={n_snapshots}."
        )

    steps_per_snap = nsteps // (n_snapshots - 1)
    snap_times = np.linspace(0.0, T, n_snapshots)

    L_diag = precompute_kinetic_Ldiag(q, domain)

    wires = list(range(n))
    dim_wires = [wires[j * q:(j + 1) * q] for j in range(d)]

    f_values = np.asarray(f_values, dtype=np.float64).reshape(dim)

    dev = qml.device(DEVICE, wires=n, shots=None)

    # QNode for the initial state
    @qml.qnode(dev, interface="numpy", diff_method=None)
    def init_state_qnode():
        prepare_initial_state(wires)
        return qml.state()

    # "chunk" QNode: evolves `steps_per_snap` steps starting from a provided input state
    @qml.qnode(dev, interface="numpy", diff_method=None)
    def evolve_chunk(state_in: np.ndarray, t0: float):
        # Set the chunk's input state
        # (QubitStateVector spesso è supportato bene su lightning.qubit; StatePrep è fallback)
        if hasattr(qml, "QubitStateVector"):
            qml.QubitStateVector(state_in, wires=wires)
        else:
            qml.StatePrep(state_in, wires=wires)

        for j in range(steps_per_snap):
            t = t0 + j * dt

            # Potenziale
            phase_V = np.exp(-1j * dt * b_fun(t) * f_values).astype(np.complex128)
            qml.DiagonalQubitUnitary(phase_V, wires=wires)

            # Cinetica su ciascun asse
            apply_kinetic_block(dt=dt, a_t=a_fun(t), dim_wires=dim_wires, L_diag=L_diag)

        return qml.state()

    # Loop esterno: solo N_SNAPSHOT stati
    states = np.empty((n_snapshots, dim), dtype=np.complex128)
    psi = init_state_qnode()
    states[0] = psi

    t0 = 0.0
    for k in range(1, n_snapshots):
        psi = evolve_chunk(psi, t0)
        states[k] = psi
        t0 += steps_per_snap * dt

    # coords: usa la tua funzione
    coords = ComputationalBasisIndex_to_SpatialCoordinates(q, d, domain, BOUNDARY_CONDITIONS)

    return states, snap_times, coords


def probabilities_from_pure_states(states: np.ndarray) -> np.ndarray:
    """Convert a batch of pure states |ψ(t_k)⟩ into probabilities in the computational basis."""
    states = np.asarray(states, complex)
    probs = np.abs(states) ** 2
    return probs


# ============================================================
# 3. PROCEDURA ONLINE: BLOCCO W, POTENZIALE E QHD
# ============================================================


def potential_block_online_unitary(
    dt: float,
    b_t: float,
    sim_wires: list[int],
    meas_wires: list[int],
    *,
    observable_factory,
) -> None:
    """Apply the online potential block U_V for a single time step.

    The block is implemented as:
        U_V = W_SM^† · exp(-i φ O_M) · W_SM,
    where φ = dt * b(t) and O_M is provided by `observable_factory`.
    """
    phi = dt * b_t

    # W_SM: |x>_S |0>_M -> |x>_S |ω(x)>_M
    prepare_measure_state(sim_wires, meas_wires)

    # U_V = exp(-i phi O_M) con O_M preso dalla factory
    O_M = observable_factory(meas_wires)
    qml.exp(O_M, coeff=-1j * phi)

    # W_SM^\dagger
    qml.adjoint(prepare_measure_state)(sim_wires, meas_wires)


def qhd_time_evolution_online(
    d: int,
    q: int,
    T: float,
    dt: float,
    *,
    a_fun: Callable[[float], float],
    b_fun: Callable[[float], float],
    domain: Tuple[float, float],
    n_snapshots: int = 101,
    bc: str = "periodic",
    ord: int = 1,
    n_meas: int = N_MEAS,
    prepare_initial_state: Callable[[list[int]], None] = prepare_initial_state,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Online QHD evolution as an iterated CPTP map on S.
    
    The reduced state on the simulation register S evolves according to:
    
        ρ_S^{j+1} = Tr_M[ U_step(j) ( ρ_S^j ⊗ |0…0⟩⟨0…0|_M ) U_step(j)† ]
    
    with
    
        U_step(j) = U_K(dt, a(t_j)) · U_V(dt, b(t_j)),
    
    where:
    - U_V = W_SM† · exp(-i dt b(t_j) O_M) · W_SM   (see `potential_block_online_unitary`);
    - U_K is the kinetic block (QFT + diagonal phase in momentum space), acting only on S.
    
    Implementation notes
    --------------------
    - Device: `default.mixed` on (n_sim + n_meas) qubits (S + M).
    - A single QNode implements one step. It takes as input ρ_SM = ρ_S ⊗ |0…0⟩⟨0…0|_M
      and returns the reduced density matrix on S.
    
    Parameters
    ----------
    d, q, T, dt, a_fun, b_fun, domain, n_snapshots, bc, ord, n_meas, prepare_initial_state
        As in the offline version, with current limitations bc="periodic" and ord=1.
        `n_meas` is the number of qubits in the measurement register M.
    
    Returns
    -------
    rhos
        Density matrices on S at snapshot times, shape (K, M, M), where M = 2**(d*q).
    snap_times
        Snapshot times, shape (K,).
    coords
        Spatial coordinates, shape (M, d).
    """
    if bc != "periodic":
        raise NotImplementedError("Per ora è implementato solo bc='periodic'.")
    if ord != 1:
        raise NotImplementedError("Per ora è implementato solo ord=1 (Lie–Trotter).")

    n_sim = d * q
    dim = 2**n_sim
    N = 2**q

    # Griglia e lunghezza dominio
    coords = ComputationalBasisIndex_to_SpatialCoordinates(q, d, domain, BOUNDARY_CONDITIONS)
    L = domain[1] - domain[0]

    # Definizione dei registri
    sim_wires = list(range(n_sim))                 # simulation register S
    meas_wires = list(range(n_sim, n_sim + n_meas))  # measurement register M
    all_wires = sim_wires + meas_wires

    dim_S = 2**n_sim
    dim_M = 2**n_meas
    dim_tot = dim_S * dim_M

    rho_SM_init = np.zeros((dim_tot, dim_tot), dtype=np.complex128)

    # Blocchi di qubit per le singole dimensioni (uno per ogni asse)
    dim_wires = [sim_wires[j * q:(j + 1) * q] for j in range(d)]

    # Spettro cinetico monodimensionale riutilizzato per ciascun asse
    L_diag = precompute_kinetic_Ldiag(q, domain)

    # Scheduling degli snapshot
    snap_idx, snap_times, nsteps, _ = make_snapshot_schedule(T, dt, n_snapshots)
    snap_idx = np.array(sorted(snap_idx), dtype=int)

    # chunk size costante (in step)
    if len(snap_idx) < 2:
        # n_snapshots=1: solo t=0
        CHUNK_SIZE = 0
    else:
        gaps = np.diff(snap_idx)
        CHUNK_SIZE = int(gaps[0])
        if not np.all(gaps == CHUNK_SIZE):
            raise ValueError("Chunk non uniformi: qui assumiamo tutti i chunk uguali.")

    # Device mixed-state su S⊗M
    dev = qml.device("default.mixed", wires=n_sim + n_meas)

    def reset_meas_to_zero(meas_wires):
        for w in meas_wires:
            qml.ResetError(1.0, 0.0, wires=w)

    @qml.qnode(dev, interface="numpy", diff_method=None)
    def evolve_chunk_qnode(rho_SM_init: np.ndarray, t0: float) -> np.ndarray:
        qml.QubitDensityMatrix(rho_SM_init, wires=all_wires)

        for j in range(CHUNK_SIZE):
            t = t0 + j * dt

            potential_block_online_unitary(
                dt, b_fun(t), sim_wires, meas_wires,
                observable_factory=observable_factory
            )

            reset_meas_to_zero(meas_wires)

            apply_kinetic_block(dt=dt, a_t=a_fun(t), dim_wires=dim_wires, L_diag=L_diag)

        return qml.density_matrix(wires=sim_wires)

    # Initial state on S
    psi0 = prepare_initial_state_vector(d, q, prepare_initial_state)
    rho_S = np.outer(psi0, psi0.conj())

    rho_list = [rho_S.copy()]

    for k in range(len(snap_idx) - 1):
        t0 = snap_idx[k] * dt  # evita drift numerico

        rho_SM_init.fill(0.0)
        rho_SM_init[0::dim_M, 0::dim_M] = rho_S

        rho_S = evolve_chunk_qnode(rho_SM_init, t0)
        rho_list.append(rho_S.copy())



    rhos = np.stack(rho_list, axis=0)
    return rhos, snap_times, coords



def probabilities_from_density_matrices(rhos: np.ndarray) -> np.ndarray:
    """
    Extract computational-basis probabilities from density matrices.
    
    Parameters
    ----------
    rhos
        Reduced density matrices on S at snapshot times, shape (K, M, M).
    
    Returns
    -------
    probs
        Probabilities p(x_r, t_k) = rho[x_r, x_r], shape (K, M).
    """
    rhos = np.asarray(rhos, complex)
    probs = np.real(np.diagonal(rhos, axis1=1, axis2=2))
    return probs



def mid_measure_and_reset_register(meas_wires: list[int]) -> None:
    """
    Measure M in the computational basis and reset each qubit to |0⟩.
    
    This operation induces shot-by-shot collapse and re-initializes M for the next step.
    """
    # Importante: effettuare esplicitamente la mid-circuit measurement.
    # Outcomes are not needed unless you implement feed-forward.
    for w in meas_wires:
        _ = qml.measure(w, reset=True)


_MCM_SUPPORT_CACHE: Dict[str, bool] = {}


def assert_mcm_supported(device_name: str) -> str:
    """
    Check at runtime whether the device supports mid-circuit measurements with reset.
    
    If unsupported, returns "default.qubit" as a fallback device name.
    """
    if not device_name:
        return "default.qubit"

    # Correzione typo comune
    if device_name == "lightining.qubit":
        device_name = "lightning.qubit"

    # Cache hit
    if device_name in _MCM_SUPPORT_CACHE:
        if _MCM_SUPPORT_CACHE[device_name]:
            return device_name
        print(
            f"Il device '{device_name}' nella tua installazione non supporta "
            "mid-circuit measurement con reset=True. Usiamo 'default.qubit'."
        )
        return "default.qubit"

    # Cache miss: test runtime senza shots sul device
    try:
        dev = qml.device(device_name, wires=1)

        @qml.qnode(dev, diff_method=None)
        def _test():
            m = qml.measure(0, reset=True)
            return qml.sample(m)

        _test_10 = qml.set_shots(10)(_test)   # <-- shots sul QNode
        _ = _test_10()

        _MCM_SUPPORT_CACHE[device_name] = True
        return device_name

    except Exception:
        _MCM_SUPPORT_CACHE[device_name] = False
        print(
            f"Il device '{device_name}' non supporta mid-circuit measurement con reset=True "
            "(o non è disponibile nella tua installazione). Usiamo 'default.qubit'."
        )
        return "default.qubit"



def qhd_time_evolution_online_stochastic(
    d: int,
    q: int,
    T: float,
    dt: float,
    *,
    a_fun: Callable[[float], float],
    b_fun: Callable[[float], float],
    domain: Tuple[float, float],
    n_snapshots: int = 101,
    bc: str = "periodic",
    ord: int = 1,
    n_meas: int = N_MEAS,
    shots: int = 10_000,
    seed: Optional[int] = 0,
    prepare_initial_state: Callable[[list[int]], None] = prepare_initial_state,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stochastic online QHD evolution via mid-circuit measurement (MCM) and reset of register M.
    
    At each time step:
      1) apply the online potential block U_V on (S + M),
      2) measure M in the computational basis and reset it to |0…0⟩,
      3) apply the kinetic block U_K on S.
    
    Returns
    -------
    probs
        Computational-basis probabilities on S at snapshot times, shape (K, 2**(d*q)).
    snap_times
        Snapshot times, shape (K,).
    coords
        Spatial coordinates, shape (2**(d*q), d).
    """
    if bc != "periodic":
        raise NotImplementedError("Per ora è implementato solo bc='periodic'.")
    if ord != 1:
        raise NotImplementedError("Per ora è implementato solo ord=1 (Lie–Trotter).")

    # Verifica supporto mid-circuit measurement con reset
    device_name = assert_mcm_supported(DEVICE)

    # Coordinate griglia
    coords = ComputationalBasisIndex_to_SpatialCoordinates(
        q, d, domain, BOUNDARY_CONDITIONS
    )

    # Snapshot schedule
    snap_idx, snap_times, nsteps, _ = make_snapshot_schedule(T, dt, n_snapshots)

    n_sim = d * q
    dim_S = 2**n_sim
    K = len(snap_times)

    probs = np.zeros((K, dim_S), dtype=float)

    # t=0: esatto (senza rumore shot) usando lo statevector iniziale
    psi0 = prepare_initial_state_vector(d, q, prepare_initial_state)
    probs[0, :] = probabilities_from_pure_states(psi0)

    # Precompute strutture comuni
    sim_wires = list(range(n_sim))
    meas_wires = list(range(n_sim, n_sim + n_meas))
    dim_wires = [sim_wires[j * q:(j + 1) * q] for j in range(d)]
    L_diag = precompute_kinetic_Ldiag(q, domain)

    # Cache dei QNode per diversi n_steps (uno per snapshot tipicamente)
    qnode_cache: Dict[int, Callable[[], np.ndarray]] = {}

    def _build_qnode(n_steps_local: int) -> Callable[[], np.ndarray]:
        dev_kwargs = dict(wires=n_sim + n_meas)
        local_seed = None if seed is None else (seed + 1000 * int(n_steps_local))

        try:
            dev_kwargs["seed"] = local_seed
            dev = qml.device(device_name, **dev_kwargs)
        except TypeError:
            dev_kwargs.pop("seed", None)
            dev = qml.device(device_name, **dev_kwargs)

        @qml.qnode(dev, diff_method=None)
        def _qnode():
            prepare_initial_state(sim_wires)
            for j in range(n_steps_local):
                t = j * dt
                potential_block_online_unitary(
                    dt, b_fun(t), sim_wires, meas_wires,
                    observable_factory=observable_factory
                )
                mid_measure_and_reset_register(meas_wires)
                apply_kinetic_block(dt=dt, a_t=a_fun(t), dim_wires=dim_wires, L_diag=L_diag)
            return qml.probs(wires=sim_wires)

        _qnode = qml.set_shots(shots)(_qnode)
        return _qnode


    # Valuta gli snapshot >0
    for k in range(1, K):
        n_steps_k = int(snap_idx[k])

        if n_steps_k not in qnode_cache:
            qnode_cache[n_steps_k] = _build_qnode(n_steps_k)

        probs_k = qnode_cache[n_steps_k]()
        probs[k, :] = np.asarray(probs_k, dtype=float)

    return probs, snap_times, coords







if __name__ == "__main__":
    # --------------------------
    # Griglia e f(x) esatta (classica)
    # --------------------------
    coords = ComputationalBasisIndex_to_SpatialCoordinates(Q_PER_DIM, D, DOMAIN, BOUNDARY_CONDITIONS)
    f_vals_exact = f_exact(coords)

    # Global basin (via FUNCTION_ANALYSIS)
    global_basin_idx = find_global_basin_indices(coords, f_vals_exact)

    results: Dict[str, Dict[str, Any]] = {}  # mode -> dict(times, coords, probs, Psuc, E, elapsed)

    # --------------------------
    # OFFLINE
    # --------------------------
    if RUN_OFFLINE:
        print("Offline precomputation of f(x_r)...")
        t0 = perf_counter()
        f_vals_offline = precompute_f_offline(
            D,
            Q_PER_DIM,
            coords,
            N_MEAS,
            prepare_measure_state=prepare_measure_state,
            observable_factory=observable_factory,
            device=DEVICE,
            precompute_cfg=OFFLINE_PRECOMPUTE,
        )
        t_pre = perf_counter() - t0
        print(f"Offline precomputation: {t_pre:.3f} s")

        print("Running OFFLINE QHD simulation...")
        t0 = perf_counter()
        states_off, times_off, coords_off = qhd_time_evolution_offline(
            D,
            Q_PER_DIM,
            T_TOTAL,
            DT,
            a_fun=A_SCHEDULE,
            b_fun=B_SCHEDULE,
            f_values=f_vals_offline,
            domain=DOMAIN,
            n_snapshots=N_SNAPSHOTS,
            bc=BOUNDARY_CONDITIONS,
            ord=SPLIT_ORDER,
            prepare_initial_state=prepare_initial_state,
        )
        t_sim = perf_counter() - t0
        print(f"Offline simulation: {t_sim:.3f} s")

        probs_off = probabilities_from_pure_states(states_off)
        Psuc_off = compute_success_probability(probs_off, global_basin_idx)
        E_off = compute_E_coefficient(probs_off, f_vals_exact)

        save_probability_data(
            "probabilities_offline.npz",
            times_off,
            coords_off,
            probs_off,
            extra_metadata={
                "mode": "offline",
                "elapsed_precompute_s": float(t_pre),
                "elapsed_sim_s": float(t_sim),
                "D": D,
                "Q_PER_DIM": Q_PER_DIM,
                "N_MEAS": N_MEAS,
                "T_TOTAL": T_TOTAL,
                "DT": DT,
                "N_SNAPSHOTS": N_SNAPSHOTS,
                "BOUNDARY_CONDITIONS": BOUNDARY_CONDITIONS,
                "SPLIT_ORDER": SPLIT_ORDER,
            },
        )

        results["Offline"] = dict(
            times=times_off, coords=coords_off, probs=probs_off,
            Psuc=Psuc_off, E=E_off, elapsed=float(t_pre + t_sim)
        )

    # --------------------------
    # ONLINE "NORMALE" (CPTP, default.mixed)
    # --------------------------
    if RUN_ONLINE:
        print("Running ONLINE QHD (CPTP map, default.mixed)...")
        t0 = perf_counter()
        rhos_on, times_on, coords_on = qhd_time_evolution_online(
            D,
            Q_PER_DIM,
            T_TOTAL,
            DT,
            a_fun=A_SCHEDULE,
            b_fun=B_SCHEDULE,
            domain=DOMAIN,
            n_snapshots=N_SNAPSHOTS,
            bc=BOUNDARY_CONDITIONS,
            ord=SPLIT_ORDER,
            n_meas=N_MEAS,
            prepare_initial_state=prepare_initial_state,
        )
        t_sim = perf_counter() - t0
        print(f"Online simulation (CPTP): {t_sim:.3f} s")

        probs_on = probabilities_from_density_matrices(rhos_on)
        Psuc_on = compute_success_probability(probs_on, global_basin_idx)
        E_on = compute_E_coefficient(probs_on, f_vals_exact)

        save_probability_data(
            "probabilities_online.npz",
            times_on,
            coords_on,
            probs_on,
            extra_metadata={
                "mode": "online_cptp",
                "elapsed_sim_s": float(t_sim),
                "D": D,
                "Q_PER_DIM": Q_PER_DIM,
                "N_MEAS": N_MEAS,
                "T_TOTAL": T_TOTAL,
                "DT": DT,
                "N_SNAPSHOTS": N_SNAPSHOTS,
                "BOUNDARY_CONDITIONS": BOUNDARY_CONDITIONS,
                "SPLIT_ORDER": SPLIT_ORDER,
            },
        )

        results["Online (CPTP)"] = dict(
            times=times_on, coords=coords_on, probs=probs_on,
            Psuc=Psuc_on, E=E_on, elapsed=float(t_sim)
        )

    # --------------------------
    # STOCHASTIC ONLINE (MCM + reset, shot-based)
    # --------------------------
    if RUN_ONLINE_STOCH:
        print(f"Running STOCHASTIC ONLINE QHD (shots={STOCH_SHOTS}, device={DEVICE})...")
        t0 = perf_counter()
        probs_st, times_st, coords_st = qhd_time_evolution_online_stochastic(
            D,
            Q_PER_DIM,
            T_TOTAL,
            DT,
            a_fun=A_SCHEDULE,
            b_fun=B_SCHEDULE,
            domain=DOMAIN,
            n_snapshots=N_SNAPSHOTS,
            bc=BOUNDARY_CONDITIONS,
            ord=SPLIT_ORDER,
            n_meas=N_MEAS,
            shots=STOCH_SHOTS,
            seed=STOCH_SEED,
            prepare_initial_state=prepare_initial_state,
        )
        t_sim = perf_counter() - t0
        print(f"Stochastic online simulation: {t_sim:.3f} s")

        Psuc_st = compute_success_probability(probs_st, global_basin_idx)
        E_st = compute_E_coefficient(probs_st, f_vals_exact)

        save_probability_data(
            "probabilities_online_stochastic.npz",
            times_st,
            coords_st,
            probs_st,
            extra_metadata={
                "mode": "online_stochastic",
                "shots": int(STOCH_SHOTS),
                "device": str(DEVICE),
                "seed": -1 if STOCH_SEED is None else int(STOCH_SEED),
                "elapsed_sim_s": float(t_sim),
                "D": D,
                "Q_PER_DIM": Q_PER_DIM,
                "N_MEAS": N_MEAS,
                "T_TOTAL": T_TOTAL,
                "DT": DT,
                "N_SNAPSHOTS": N_SNAPSHOTS,
                "BOUNDARY_CONDITIONS": BOUNDARY_CONDITIONS,
                "SPLIT_ORDER": SPLIT_ORDER,
            },
        )

        results[f"Online (stoch, shots={STOCH_SHOTS})"] = dict(
            times=times_st, coords=coords_st, probs=probs_st,
            Psuc=Psuc_st, E=E_st, elapsed=float(t_sim)
        )

    # --------------------------
    # Summary tempi
    # --------------------------
    if len(results) == 0:
        raise RuntimeError("No simulation executed: enable at least one of RUN_OFFLINE / RUN_ONLINE / RUN_ONLINE_STOCH.")

    print("\n=== Riepilogo tempi (s) ===")
    for k, v in results.items():
        print(f"{k:>22s}: {v['elapsed']:.3f}")

    # --------------------------
    # Seleziona times/coords di riferimento (assumiamo siano coerenti)
    # --------------------------
    labels = list(results.keys())
    t_ref = results[labels[0]]["times"]
    c_ref = results[labels[0]]["coords"]

    for lab in labels[1:]:
        if not np.allclose(results[lab]["times"], t_ref):
            raise ValueError(f"Timeline diversa tra '{labels[0]}' e '{lab}'.")
        # coords: spesso identiche; se vuoi essere severo:
        # if not np.allclose(results[lab]["coords"], c_ref): raise ValueError(...)

    # --------------------------
    # Plot Psuc ed E: 1/2/3 curve
    # --------------------------
    series = [(lab, results[lab]["Psuc"], results[lab]["E"]) for lab in labels]
    if len(series) == 1:
        plot_Psuc_and_E_single(t_ref, series[0][1], series[0][2], label=series[0][0].replace(" ", "_"))
    elif len(series) == 2:
        # mantiene lo stile del tuo confronto a 2
        (l0, Ps0, E0), (l1, Ps1, E1) = series
        plot_Psuc_and_E(t_ref, Ps0, E0, Ps1, E1)
    else:
        plot_Psuc_and_E_multi(t_ref, series)

    # --------------------------
    # Heatmaps (solo D==1): 1/2/3 pannelli
    # --------------------------
    if D == 1:
        probs_list = [results[lab]["probs"] for lab in labels]
        titles = [lab for lab in labels]

        if len(probs_list) == 1:
            plot_probability_heatmap(t_ref, c_ref, probs_list[0], title=rf"$|\psi(x,t)|^2$ ({titles[0]})",
                                     filename=f"Heatmap_{titles[0].replace(' ','_')}.pdf")
        elif len(probs_list) == 2:
            plot_heatmaps(probs_list[0], probs_list[1], t_ref, c_ref)
        else:
            plot_heatmaps_multi(probs_list, titles, t_ref, c_ref)

    # --------------------------
    # Kolmogorov fidelity: confronto a 2 o a 3 (pairwise)
    # --------------------------
    if len(labels) == 2:
        p0 = results[labels[0]]["probs"]
        p1 = results[labels[1]]["probs"]
        F = compute_kolmogorov_fidelity_vs_time(p0, p1)
        plot_kolmogorov_fidelity(t_ref, F)
    elif len(labels) == 3:
        pA = results[labels[0]]["probs"]
        pB = results[labels[1]]["probs"]
        pC = results[labels[2]]["probs"]

        F_AB = compute_kolmogorov_fidelity_vs_time(pA, pB)
        F_AC = compute_kolmogorov_fidelity_vs_time(pA, pC)
        F_BC = compute_kolmogorov_fidelity_vs_time(pB, pC)

        plot_kolmogorov_fidelity_multi(
            t_ref,
            [
                (f"{labels[0]} vs {labels[1]}", F_AB),
                (f"{labels[0]} vs {labels[2]}", F_AC),
                (f"{labels[1]} vs {labels[2]}", F_BC),
            ],
        )



        # --- Preparazione stato di misura e osservabile per definire f(x) ---
        # def prepare_measure_state(sim_wires: list[int], meas_wires: list[int]) -> None: 
        #     """ Operatore W_SM che prepara |ω(x)> ≡ |ψ(x1,x2,x3)> sul registro di misura M, controllato dallo stato |x> del registro di simulazione S. Implementazione: |x>_S |0,0>_M → |x>_S |ψ(x1,x2,x3)>_M, dove: - x1 è codificato nel primo terzo dei qubit di S, - x2 nel secondo terzo, - x3 nel terzo terzo, usando rotazioni controllate CRX/CRY con pesi binari. La parte su M realizza il circuito [R_y(x3) ⊗ I] · CNOT(1,2) · [R_x(x1) ⊗ R_x(x2)] · CNOT(2,1) · [I ⊗ H] |0,0>. """ 
        #     if len(meas_wires) < 2: 
        #         raise ValueError("Servono almeno 2 qubit di misura per questo esempio (σ_z ⊗ σ_z).") 
        #     n_sim = len(sim_wires) 
        #     if n_sim % 3 != 0: 
        #         raise ValueError( f"prepare_measure_state: ci si aspetta che il numero di qubit di simulazione " f"sia multiplo di 3 (uno per ciascuna coordinata x1,x2,x3); ricevuto {n_sim}." ) 
        #     n_per_dim = n_sim // 3 
        #     register1 = sim_wires[0:n_per_dim]              # codifica x1 
        #     register2 = sim_wires[n_per_dim:2*n_per_dim]    # codifica x2 
        #     register3 = sim_wires[2*n_per_dim:3*n_per_dim]  # codifica x3 
        #
        #     # Intervallo del dominio (es. 2π)
        #     delta = DOMAIN[1] - DOMAIN[0]     
        #
        #     # Qubit di misura 
        #     q1 = meas_wires[0] # "qubit 1" 
        #     q2 = meas_wires[1] # "qubit 2" 
        #
        #     # ============================================================ # 
        #     # 1) [ I ⊗ H ] |0,0> 
        #     # ============================================================ 
        #
        #     qml.H(wires=q2) 
        #
        #     # ============================================================ 
        #     # 2) CNOT(2,1): controllo su q2, target q1 
        #     # (questa è la CNOT(2,1) nella notazione dell'espressione analitica) 
        #     # ============================================================ 
        #
        #     qml.CNOT(wires=[q2, q1]) 
        #
        #     # ============================================================ 
        #     # 3) [ R_x(x1) ⊗ R_x(x2) ] 
        #     # Realizzati come prodotti di CRX dal registro S 
        #     # ============================================================ 
        #
        #     # R_x(x1) su q1, con x1 codificato nei bit di register1 
        #     for j, w in enumerate(register1): 
        #         angle = delta / (2 ** (j + 1)) 
        #         qml.CRX(angle, wires=[w, q1]) 
        #     # R_x(x2) su q2, con x2 codificato nei bit di register2 
        #     for j, w in enumerate(register2): 
        #         angle = delta / (2 ** (j + 1)) 
        #         qml.CRX(angle, wires=[w, q2]) 
        #
        #     # ============================================================
        #     # 4) CNOT(1,2): controllo su q1, target q2 # ============================================================ 
        #
        #     qml.CNOT(wires=[q1, q2]) 
        #     
        #     # ============================================================ 
        #     # 5) [ R_y(x3) ⊗ I ] # R_y(x3) su q1, codificando x3 nei bit di register3 
        #     # ============================================================ 
        #
        #     for j, w in enumerate(register3): 
        #         angle = delta / (2 ** (j + 1)) 
        #         qml.CRY(angle, wires=[w, q1])
        #
        #
        # def observable_factory(meas_wires: list[int]) -> qml.operation.Operator:
        #     """
        #     Osservabile O_M misurata sul registro di misura per ottenere f(x).
        #
        #     Deve restituire un oggetto di tipo qml.operation.Operator (ad esempio
        #     un prodotto di Pauli su più qubit).
        #     """
        #     if len(meas_wires) < 2:
        #         raise ValueError("Servono almeno 2 qubit di misura.")
        #     return qml.PauliZ(meas_wires[0]) @ qml.PauliZ(meas_wires[1])
        #
        #
        # def f_exact(coords: np.ndarray) -> np.ndarray:
        #     """
        #     Espressione classica esatta di f(x), usata per:
        #     - definire il bacino globale;
        #     - calcolare ⟨f⟩(t) e il coefficiente E(t).
        #     """
        #     coords = np.asarray(coords, float)
        #     assert coords.ndim == 2, "coords deve essere (M, d)."
        #     x1 = coords[:, 0]
        #     x2 = coords[:, 1]
        #     x3 = coords[:, 2]
        #     return -np.cos(x1+x2) * np.sin(x3)
