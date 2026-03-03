"""
Plot interface motion (alpha=0.5 contour) from saved one-domain alpha snapshots.

This is intended to avoid re-running the expensive 2D one-domain simulation just
to regenerate the Fig.6a-style "interface motion" panel.

Inputs (in --results-dir)
-------------------------
- summary.json        (written by duddu2007_one_domain_growth_2d_fig6_example2.py)
- snaps_alpha.npz     (ditto; contains t_days and alpha snapshots)

Output
------
- <results-dir>/fig6a_interface_from_snaps.png  (default; configurable via --out)

Run (recommended)
----------------
conda run --no-capture-output -n fenicsx python \
  examples/biofilms/benchmarks/dadu/plot_one_domain_interface_from_snaps.py \
  --results-dir <OUTDIR>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _paper_times_days() -> np.ndarray:
    # Duddu et al. (2007) Fig.6 caption times (Example 2).
    return np.asarray(
        [0.0, 1.1, 2.7, 4.6, 6.6, 8.7, 10.7, 12.7, 14.7, 16.7, 18.6, 20.6, 22.5, 24.5, 26.5, 28.6],
        dtype=float,
    )


def _select_nearest_indices(t_snap: np.ndarray, t_targets: np.ndarray) -> list[int]:
    idx: list[int] = []
    for tt in np.asarray(t_targets, dtype=float).tolist():
        j = int(np.argmin(np.abs(t_snap - float(tt))))
        if j not in idx:
            idx.append(j)
    return idx


def _build_alpha_grid_map(*, L: float, H: float, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a robust DOF->(i,j) mapping for alpha on the structured grid.

    Returns
    -------
    xs : (nx+1,) sorted x-coordinates
    ys : (ny+1,) sorted y-coordinates
    ii : (ndof,) integer x-indices for each DOF
    jj : (ndof,) integer y-indices for each DOF
    """
    # Import locally: pycutfem pulls in optional deps not available in all python envs.
    from pycutfem.core.dofhandler import DofHandler
    from pycutfem.core.mesh import Mesh
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.utils.meshgen import structured_quad

    nodes, elems, _edges, corners = structured_quad(float(L), float(H), nx=int(nx), ny=int(ny), poly_order=2, offset=(0.0, 0.0))
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=2)
    me = MixedElement(mesh, field_specs={"alpha": 1})
    dh = DofHandler(me, method="cg")

    coords = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    xs = np.unique(np.round(coords[:, 0], decimals=14))
    ys = np.unique(np.round(coords[:, 1], decimals=14))
    if xs.size != int(nx) + 1 or ys.size != int(ny) + 1:
        raise ValueError(f"Unexpected grid shape from DOF coords: xs={xs.size} ys={ys.size} expected {(nx+1, ny+1)}")

    xs = np.sort(xs)
    ys = np.sort(ys)
    x_to_i = {float(x): i for i, x in enumerate(xs.tolist())}
    y_to_j = {float(y): j for j, y in enumerate(ys.tolist())}

    ii = np.empty((coords.shape[0],), dtype=int)
    jj = np.empty((coords.shape[0],), dtype=int)
    for k, (x, y) in enumerate(coords.tolist()):
        xi = float(round(float(x), 14))
        yi = float(round(float(y), 14))
        ii[k] = int(x_to_i[xi])
        jj[k] = int(y_to_j[yi])

    return xs, ys, ii, jj


def _alpha_grid_from_map(
    *, xs: np.ndarray, ys: np.ndarray, ii: np.ndarray, jj: np.ndarray, alpha_dof: np.ndarray
) -> np.ndarray:
    alpha_dof = np.asarray(alpha_dof, dtype=float).ravel()
    if alpha_dof.size != ii.size:
        raise ValueError(f"alpha DOF size mismatch: map={ii.size} values={alpha_dof.size}")
    A = np.empty((ys.size, xs.size), dtype=float)
    A[jj, ii] = alpha_dof
    return A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, required=True)
    ap.add_argument("--summary", type=str, default="summary.json")
    ap.add_argument("--snaps", type=str, default="snaps_alpha.npz")
    ap.add_argument("--out", type=str, default="", help="Default: <results-dir>/fig6a_interface_from_snaps.png")
    ap.add_argument("--alpha-half", type=float, default=0.5)
    ap.add_argument("--paper-times", action="store_true", help="Plot at Duddu(2007) Fig.6 caption times.")
    ap.add_argument(
        "--targets",
        type=str,
        default="",
        help="Comma-separated target times in days (select nearest stored snapshot per time).",
    )
    ap.add_argument("--stride", type=int, default=7, help="If not --paper-times: plot every N snapshots.")
    ap.add_argument("--max-curves", type=int, default=30, help="Hard limit on number of contours plotted.")
    ap.add_argument("--linewidth", type=float, default=1.0)
    ap.add_argument("--color-by-time", action="store_true", help="Color interface curves by time with a colormap.")
    ap.add_argument("--cmap", type=str, default="viridis", help="Colormap used when --color-by-time is set.")
    args = ap.parse_args()

    results_dir = Path(str(args.results_dir))
    summary_path = results_dir / str(args.summary)
    snaps_path = results_dir / str(args.snaps)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not snaps_path.exists():
        raise FileNotFoundError(snaps_path)

    summary = json.loads(summary_path.read_text())
    L = float(summary["L_mm"])
    H = float(summary["H_mm"])
    nx = int(summary["nx"])
    ny = int(summary["ny"])

    d = np.load(snaps_path)
    t_snap = np.asarray(d["t_days"], dtype=float).ravel()
    a_snap = np.asarray(d["alpha"], dtype=float)
    if a_snap.ndim != 2 or a_snap.shape[0] != t_snap.size:
        raise ValueError("Unexpected snaps_alpha.npz shapes.")

    if bool(args.paper_times):
        targets = _paper_times_days()
        idx = _select_nearest_indices(t_snap, targets)
    elif str(args.targets).strip():
        targets = np.asarray([float(s) for s in str(args.targets).split(",") if s.strip()], dtype=float)
        idx = _select_nearest_indices(t_snap, targets)
    else:
        stride = max(1, int(args.stride))
        idx = list(range(0, int(t_snap.size), stride))

    idx = idx[: max(1, int(args.max_curves))]

    outpng = Path(str(args.out)) if str(args.out).strip() else (results_dir / "fig6a_interface_from_snaps.png")
    outpng.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    fig, ax = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    ax.set_title("Interface motion (alpha=0.5)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_xlim(0.0, float(L))
    ax.set_ylim(0.0, float(H))
    ax.grid(True, alpha=0.2)

    level = float(args.alpha_half)
    xs, ys, ii, jj = _build_alpha_grid_map(L=L, H=H, nx=nx, ny=ny)
    if bool(args.color_by_time):
        times = np.asarray([float(t_snap[int(j)]) for j in idx], dtype=float)
        tmin = float(np.min(times)) if times.size else 0.0
        tmax = float(np.max(times)) if times.size else 1.0
        norm = plt.Normalize(vmin=tmin, vmax=tmax)
        cmap = plt.get_cmap(str(args.cmap))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.9)
        cbar.set_label("t (days)")
    for j in idx:
        A = _alpha_grid_from_map(xs=xs, ys=ys, ii=ii, jj=jj, alpha_dof=a_snap[int(j), :])
        if bool(args.color_by_time):
            c = cmap(norm(float(t_snap[int(j)])))
            ax.contour(xs, ys, A, levels=[level], colors=[c], linewidths=float(args.linewidth))
        else:
            ax.contour(xs, ys, A, levels=[level], colors="k", linewidths=float(args.linewidth))

    fig.savefig(outpng, dpi=200)
    plt.close(fig)
    print(f"- Wrote {outpng}")


if __name__ == "__main__":
    main()
