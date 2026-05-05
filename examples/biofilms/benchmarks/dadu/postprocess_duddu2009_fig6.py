"""
Postprocess helper for Duddu (2009) Fig. 6 style comparisons.

Reads the final VTK snapshot (solution_*.vtu) for each detachment model folder produced by
`duddu2009_detachment_2d_seq.py` and generates:

  - alpha contour plots with the alpha=0.5 interface overlaid
  - thickness profiles l(x) computed from the alpha=alpha_half interface
  - per-colony peak heights (printed to stdout)

All outputs are written under the provided --outdir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _read_final_vtu(model_dir: Path) -> Path:
    vtus = sorted(model_dir.glob("solution_*.vtu"))
    if not vtus:
        raise FileNotFoundError(f"No VTK files found in {model_dir}")
    return vtus[-1]


def _triangulation_from_quads(points_xy: np.ndarray, quads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_xy, dtype=float)
    q = np.asarray(quads, dtype=int)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Expected quad connectivity (n,4), got {q.shape}")
    tris = np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]).astype(int, copy=False)
    used = np.unique(tris.ravel())
    return used, tris


def _thickness_profile_nodal(
    *,
    points_xy: np.ndarray,
    alpha: np.ndarray,
    used_point_ids: np.ndarray,
    alpha_half: float,
    x_round: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_xy, dtype=float)[used_point_ids]
    a = np.asarray(alpha, dtype=float).ravel()[used_point_ids]
    mask = np.isfinite(a) & (a >= float(alpha_half))
    if not np.any(mask):
        return np.asarray([]), np.asarray([])
    x = np.round(pts[mask, 0], decimals=int(x_round))
    y = pts[mask, 1]
    x_levels = np.unique(x)
    l = np.zeros_like(x_levels, dtype=float)
    for i, xv in enumerate(x_levels):
        yy = y[x == xv]
        l[i] = float(np.max(yy)) if yy.size else 0.0
    return np.asarray(x_levels, dtype=float), np.asarray(l, dtype=float)


def _interface_vertices_from_contour_set(contour_set) -> np.ndarray:
    # TriContourSet uses linear interpolation on the triangulation; this gives sub-mesh
    # interface coordinates, avoiding the "node-quantized" thickness artifacts.
    try:
        segs = contour_set.allsegs[0]  # one contour level
    except Exception:
        segs = []
    if not segs:
        return np.zeros((0, 2), dtype=float)
    arrays: list[np.ndarray] = []
    for s in segs:
        a = np.asarray(s, dtype=float)
        if a.size == 0:
            continue
        arrays.append(a.reshape(-1, 2))
    if not arrays:
        return np.zeros((0, 2), dtype=float)
    verts = np.vstack(arrays)
    return np.asarray(verts, dtype=float)


def _thickness_profile_from_vertices(*, verts_xy: np.ndarray, x_round: int = 10) -> tuple[np.ndarray, np.ndarray]:
    verts = np.asarray(verts_xy, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2 or verts.size == 0:
        return np.asarray([]), np.asarray([])
    x = np.round(verts[:, 0], decimals=int(x_round))
    y = verts[:, 1]
    x_levels = np.unique(x)
    l = np.zeros_like(x_levels, dtype=float)
    for i, xv in enumerate(x_levels):
        yy = y[x == xv]
        l[i] = float(np.max(yy)) if yy.size else 0.0
    return np.asarray(x_levels, dtype=float), np.asarray(l, dtype=float)


def _colony_centers(*, L: float, r0: float, n_colonies: int) -> np.ndarray:
    margin = max(2.0 * float(r0), 1.0e-12)
    if float(L) <= 2.0 * margin:
        raise ValueError("Domain too small for requested colonies/radius.")
    return np.linspace(margin, float(L) - margin, int(n_colonies))


def main() -> None:
    ap = argparse.ArgumentParser(description="Postprocess Duddu2009 Fig. 6 runs (alpha contours + thickness profiles).")
    ap.add_argument("--outdir", type=str, required=True, help="Run output directory (contains model=*/ folders).")
    ap.add_argument("--models", type=str, default="shear,l2,poly", help="Comma list: shear,l2,poly,none or 'all'.")
    ap.add_argument("--alpha-half", type=float, default=0.5, help="Threshold used for thickness l(x).")
    ap.add_argument(
        "--thickness-mode",
        type=str,
        default="contour",
        choices=("contour", "nodal"),
        help="How to compute l(x): from alpha=alpha_half contour ('contour') or nodal alpha>=alpha_half ('nodal').",
    )
    ap.add_argument("--L", type=float, default=2.0, help="Domain length (mm), used for colony centers.")
    ap.add_argument("--r0", type=float, default=0.025, help="Initial colony radius (mm), used for colony centers.")
    ap.add_argument("--n-colonies", type=int, default=5, help="Number of initial colonies, used for colony centers.")
    ap.add_argument("--save-prefix", type=str, default="fig6", help="Prefix for saved images.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if not outdir.exists():
        raise SystemExit(f"Missing --outdir {outdir}")

    raw = str(args.models or "").strip().lower()
    if raw in {"all", "*"}:
        raw = "shear,l2,poly"
    models = [p.strip() for p in raw.split(",") if p.strip()]
    if not models:
        raise SystemExit("Empty --models list.")

    try:
        import meshio  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"meshio is required ({exc}).")

    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.tri import Triangulation  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"matplotlib is required ({exc}).")

    # Load meshes / profiles
    alpha_fields: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    l_profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    interface_verts: dict[str, np.ndarray] = {}

    for m in models:
        model_dir = outdir / f"model={m}"
        vtu = _read_final_vtu(model_dir)
        mesh = meshio.read(str(vtu))
        if "quad" not in getattr(mesh, "cells_dict", {}):
            raise SystemExit(f"{vtu} does not contain quad cells (have {list(mesh.cells_dict)})")
        if "alpha" not in getattr(mesh, "point_data", {}):
            raise SystemExit(f"{vtu} missing point_data['alpha'] (have {list(mesh.point_data)})")

        pts_xy = np.asarray(mesh.points[:, :2], dtype=float)
        quads = np.asarray(mesh.cells_dict["quad"], dtype=int)
        used, tris = _triangulation_from_quads(pts_xy, quads)
        alpha = np.asarray(mesh.point_data["alpha"], dtype=float).ravel()

        alpha_fields[m] = (pts_xy, tris, alpha, used)

    # Alpha/interface plots
    n = len(models)
    fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 2.4), sharex=True, sharey=True, constrained_layout=True)
    if n == 1:
        axs = [axs]
    levels = np.linspace(0.0, 1.0, 21)
    mappable = None
    for ax, m in zip(axs, models):
        pts_xy, tris, alpha, _used = alpha_fields[m]
        tri = Triangulation(pts_xy[:, 0], pts_xy[:, 1], tris)
        mappable = ax.tricontourf(tri, alpha, levels=levels, cmap="viridis")
        cs = ax.tricontour(tri, alpha, levels=[float(args.alpha_half)], colors="w", linewidths=1.0)
        interface_verts[m] = _interface_vertices_from_contour_set(cs)
        ax.set_title(m)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
    axs[0].set_ylabel("y (mm)")
    if mappable is not None:
        fig.colorbar(mappable, ax=axs, shrink=0.9, pad=0.02, label="alpha")
    save_alpha = outdir / f"{str(args.save_prefix)}_alpha_contours.png"
    fig.savefig(str(save_alpha), dpi=200)
    print(f"[ok] wrote {save_alpha}")

    # Interface-only plot (paper-style)
    figi, axsi = plt.subplots(1, n, figsize=(4.2 * n, 2.0), sharex=True, sharey=True, constrained_layout=True)
    if n == 1:
        axsi = [axsi]
    for ax, m in zip(axsi, models):
        pts_xy, tris, alpha, _used = alpha_fields[m]
        tri = Triangulation(pts_xy[:, 0], pts_xy[:, 1], tris)
        ax.tricontour(tri, alpha, levels=[float(args.alpha_half)], colors="k", linewidths=1.5)
        ax.set_title(m)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
    axsi[0].set_ylabel("y (mm)")
    save_int = outdir / f"{str(args.save_prefix)}_interface_only.png"
    figi.savefig(str(save_int), dpi=200)
    print(f"[ok] wrote {save_int}")

    # Thickness profiles (compute after contour extraction so we can use the interface vertices).
    for m in models:
        pts_xy, _tris, alpha, used = alpha_fields[m]
        verts = interface_verts.get(m, np.zeros((0, 2), dtype=float))
        if str(args.thickness_mode).strip().lower() == "contour" and verts.size:
            x_prof, l_prof = _thickness_profile_from_vertices(verts_xy=verts)
        else:
            x_prof, l_prof = _thickness_profile_nodal(
                points_xy=pts_xy, alpha=alpha, used_point_ids=used, alpha_half=float(args.alpha_half)
            )
        l_profiles[m] = (x_prof, l_prof)

    # Thickness profiles
    fig2, ax2 = plt.subplots(figsize=(7.0, 2.7), constrained_layout=True)
    for m in models:
        x_prof, l_prof = l_profiles[m]
        if x_prof.size == 0:
            continue
        ax2.plot(x_prof, l_prof, label=m)
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel(f"l(x) [alpha>={float(args.alpha_half):g}] (mm)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", frameon=False)
    save_l = outdir / f"{str(args.save_prefix)}_thickness_profiles.png"
    fig2.savefig(str(save_l), dpi=200)
    print(f"[ok] wrote {save_l}")

    # Per-colony peak heights (simple numeric summary)
    centers = _colony_centers(L=float(args.L), r0=float(args.r0), n_colonies=int(args.n_colonies))
    print("\nPer-colony peak heights at final time (mm):")
    print("  centers_x =", ", ".join(f"{c:.4g}" for c in centers))
    for m in models:
        verts = interface_verts.get(m, np.zeros((0, 2), dtype=float))
        peaks = []
        if str(args.thickness_mode).strip().lower() == "contour" and verts.size:
            x_v = np.asarray(verts[:, 0], dtype=float)
            y_v = np.asarray(verts[:, 1], dtype=float)
            for cx in centers:
                mask = np.abs(x_v - float(cx)) <= float(args.r0)
                peaks.append(float(np.max(y_v[mask])) if np.any(mask) else 0.0)
        else:
            x_prof, l_prof = l_profiles[m]
            for cx in centers:
                mask = np.abs(x_prof - cx) <= float(args.r0)
                peaks.append(float(np.max(l_prof[mask])) if np.any(mask) else 0.0)
        print(f"  {m:>6s}: " + ", ".join(f"{p:.4g}" for p in peaks))


if __name__ == "__main__":
    main()
