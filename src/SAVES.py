from __future__ import annotations

import datetime as _dt
import gc
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Use a non-interactive backend for batch/report generation environments.
matplotlib.use("Agg")


def data_save(
    psi_num: np.ndarray,
    times: np.ndarray,
    xgrid: np.ndarray,
    meta: Dict[str, Any],
    psi_qhd: Optional[np.ndarray] = None,
    out_dir: Path | str = Path("./QHD_data") / "Raw_data",
    compressed: bool = False,
) -> Path:
    """
    Save raw simulation outputs to a single NPZ archive.

    This function is designed for reproducible batch runs:
    - Produces one `.npz` file per simulation (optional compression).
    - Stores: `psi_num`, `psi_qhd` (or empty), `times`, `xgrid`, and `meta_json`.
    - The filename encodes only the minimal identity fields: (d, q, T, dt, f).
      All other configuration is persisted inside `meta_json`.
    - Never overwrites: if the base name already exists, suffixes `-(1)`, `-(2)`, ...
      are appended.

    Parameters
    ----------
    psi_num:
        Numerical states array with shape (K, M), where K is the number of saved
        snapshots and M is the number of spatial grid points (M = N**d).
    times:
        Snapshot times array with shape (K,).
    xgrid:
        Spatial grid array with shape (M,). For d>1 this is typically the x-axis
        for 1D projections/plots; it is still saved as provided.
    meta:
        Metadata dictionary. Must contain at least: "d", "q", "T", "dt". The key
        "f" is used for filename generation if present.
    psi_qhd:
        Optional QHD states array with shape (K, M). If None, an empty array is saved.
    out_dir:
        Output directory (created if missing).
    compressed:
        If True, uses `np.savez_compressed`; otherwise uses `np.savez`.

    Returns
    -------
    Path
        Path to the saved `.npz` file.

    Raises
    ------
    ValueError
        If array shapes are inconsistent with each other.
    KeyError
        If required fields are missing from `meta`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _fmt_float(x: float) -> str:
        """Format floats compactly for filenames."""
        return f"{float(x):.6g}"

    def _sanitize_filename_token(s: str) -> str:
        """Restrict filename tokens to safe characters."""
        return re.sub(r"[^A-Za-z0-9_.-]+", "", s)

    # ---- Extract minimal identity fields from metadata ----
    d = int(meta["d"])
    q = int(meta["q"])
    T = float(meta["T"])
    dt = float(meta["dt"])
    f_name = meta.get("f", "")

    base_name = (
        f"d{d}_q{q}_T{_fmt_float(T)}_dt{_fmt_float(dt)}_"
        f"f-{_sanitize_filename_token(str(f_name))}"
    )

    # ---- Shape checks / normalization ----
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    xgrid = np.asarray(xgrid, dtype=np.float64).reshape(-1)
    psi_num = np.asarray(psi_num)

    if psi_num.ndim != 2:
        raise ValueError("psi_num must have shape (K, M).")
    K_num, M = psi_num.shape

    if K_num != len(times):
        raise ValueError(
            f"Shape mismatch: len(times)={len(times)} but psi_num.shape[0]={K_num}."
        )
    if len(xgrid) != M:
        raise ValueError(f"Shape mismatch: len(xgrid)={len(xgrid)} but M={M}.")

    has_qhd = psi_qhd is not None
    if has_qhd:
        psi_qhd = np.asarray(psi_qhd)
        if psi_qhd.shape != (len(times), M):
            raise ValueError(
                f"psi_qhd must have shape (K, M)=({len(times)}, {M}); got {psi_qhd.shape}."
            )
    else:
        psi_qhd = np.empty((0,), dtype=np.complex128)

    # Ensure contiguous storage (helps save speed and downstream memory mapping).
    if not psi_num.flags.c_contiguous:
        psi_num = np.ascontiguousarray(psi_num)
    if has_qhd and not psi_qhd.flags.c_contiguous:
        psi_qhd = np.ascontiguousarray(psi_qhd)

    # ---- Unique output path (no overwrite) ----
    ext = ".npz"
    path = out_dir / f"{base_name}{ext}"
    if path.exists():
        k = 1
        while (out_dir / f"{base_name}-({k}){ext}").exists():
            k += 1
        path = out_dir / f"{base_name}-({k}){ext}"

    # ---- Serialize metadata (kept fully inside the archive) ----
    meta_json = json.dumps(dict(meta), ensure_ascii=False)

    # ---- Save NPZ ----
    save_kwargs = dict(
        psi_num=psi_num,
        psi_qhd=psi_qhd,
        times=times,
        xgrid=xgrid,
        meta_json=np.array(meta_json),
    )
    if compressed:
        np.savez_compressed(path, **save_kwargs)
    else:
        np.savez(path, **save_kwargs)

    print(f"[data_save] Saved ({'compressed' if compressed else 'raw'}) -> {path}")
    return path


def _sanitize(txt: str) -> str:
    """
    Sanitize an arbitrary string into a filesystem-friendly token.

    Only allows [A-Za-z0-9_.-], replacing other characters with underscores and
    trimming leading/trailing underscores.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(txt)).strip("_")


def save_report_pdf(
    figures: Iterable[plt.Figure],
    meta: Dict[str, Any],
    out_dir: Path | str = Path("./QHD_DATA") / "reports",
    **kwargs,
) -> Path:
    """
    Save a collection of Matplotlib figures into a single multi-page PDF.

    Conventions
    -----------
    - Default directory: `./QHD_DATA/reports`
    - Filename encodes only: d, q, T, dt, and the function name f (to match the
      raw-data naming convention).
    - Never overwrites: adds suffixes `-(1)`, `-(2)`, ...
    - Uses a temporary file and `os.replace` for atomic output.

    Parameters
    ----------
    figures:
        An iterable of Matplotlib Figure objects. A single Figure is also accepted.
    meta:
        Metadata dictionary. Expected keys include: d, q, T, dt, and f (or f_name).
    out_dir:
        Output directory (created if missing).
    **kwargs:
        Reserved for forward compatibility (currently unused).

    Returns
    -------
    Path
        Path to the generated PDF.

    Notes
    -----
    Figures are closed after saving to release resources (important in batch runs).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = meta.get("d")
    q = meta.get("q")
    T = meta.get("T")
    dt = meta.get("dt")
    f_name = meta.get("f", meta.get("f_name", ""))

    base_name = f"d{d}_q{q}_T{T}_dt{dt}_f-{_sanitize(f_name)}"
    out_path = out_dir / f"{base_name}.pdf"
    tmp_path = out_dir / f"{base_name}.tmp.pdf"

    # No overwrite: add suffixes if needed.
    if out_path.exists():
        k = 1
        while (out_dir / f"{base_name}-({k}).pdf").exists():
            k += 1
        out_path = out_dir / f"{base_name}-({k}).pdf"
        tmp_path = out_path.with_suffix(".tmp.pdf")

    pdf: Optional[PdfPages] = None
    try:
        pdf = PdfPages(tmp_path)
        infod = pdf.infodict()
        infod["Title"] = f"QHD Report: {f_name}"
        infod["Author"] = "QHD Pipeline"
        infod["Subject"] = f"d={d}, q={q}, T={T}, dt={dt}"
        infod["CreationDate"] = _dt.datetime.now()

        if not isinstance(figures, (list, tuple)):
            figures = [figures]

        for fig in figures:
            pdf.savefig(fig)

    finally:
        if pdf is not None:
            pdf.close()

    os.replace(tmp_path, out_path)

    # Close figures to free memory in long batch runs.
    try:
        for fig in figures:
            plt.close(fig)
    except Exception:
        pass

    gc.collect()
    print(f"[save_report_pdf] Saved: {out_path}")
    return out_path
