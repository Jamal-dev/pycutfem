"""
Duddu et al. (2007) growth-only model: 2D Example 2 (Fig. 6) using the one-domain model.

Goal
----
Reproduce the qualitative growth patterns of Duddu et al. (2007) Fig. 6 (Example 2)
with our *one-domain* diffuse-interface model (`examples/utils/biofilm/one_domain.py`)
in the growth-only limit (no detachment, no mechanics).

We follow a paper-like operator splitting per growth step:
  (1) quasi-steady substrate S  (nonlinear diffusion-reaction),
  (2) solve (p, vS) from the volume constraint + skeleton balance, given S,
  (3) advect the diffuse indicator alpha with vS.

Important modelling choices for the Duddu(2007)-limit
-----------------------------------------------------
- Growth source in the constraint:
    div(C v + B vS) = alpha * s_v, with v frozen to 0,
  where we choose   s_v = (1-phi_b) * divU_Duddu(S) so that div(vS) ≈ divU in the
  biofilm bulk (alpha≈1, phi≈phi_b).

- Substrate uptake:
  We map Duddu's Michaelis–Menten consumption µ_S(S) to the one-domain Monod sink
  by choosing mu_max so that RS(S) matches Duddu inside the biofilm.

- Moving substrate Dirichlet boundary:
  Duddu imposes S=Sbar on a line Γ_S^d located Ls above the current top-most biofilm
  point. We enforce this via a CutFEM penalty integral on a moving affine level set.

All outputs are written under examples/biofilms/benchmarks/dadu/results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.solvers.nonlinear_solver import (
    HAS_PETSC,
    LinearSolverParameters,
    NewtonParameters,
    NewtonSolver,
    PdasNewtonSolver,
    VIParameters,
)
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    dot,
    grad,
    inner,
)
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dInterface, dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad

from pycutfem.plotting.triangulate import triangulate_field

from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms


def _tag_rectangle_boundaries(mesh: Mesh, *, L: float, H: float, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - float(L)) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - float(H)) <= tol,
        }
    )


def _write_pvd(*, out_pvd: Path, datasets: list[tuple[float, str]]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for t, rel in datasets:
        lines.append(f'    <DataSet timestep="{float(t):.12e}" group="" part="0" file="{rel}"/>')
    lines += ["  </Collection>", "</VTKFile>", ""]
    out_pvd.write_text("\n".join(lines), encoding="utf-8")


def _parse_targets_csv(s: str) -> list[float]:
    if not str(s).strip():
        return []
    vals: list[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    out: list[float] = []
    for v in sorted(vals):
        if not out or abs(v - out[-1]) > 1.0e-12:
            out.append(float(v))
    return out


def _set_inactive_fields(dh: DofHandler, field_names: list[str]) -> None:
    inactive: set[int] = set()
    for fname in field_names:
        try:
            sl = np.asarray(dh.get_field_slice(fname), dtype=int).ravel()
        except Exception:
            continue
        inactive.update(int(i) for i in sl)
    dh.dof_tags = {"inactive": inactive}


def _as_float_time(fn):
    return lambda x, y, t: float(np.asarray(fn(np.asarray(x), np.asarray(y), float(t))).reshape(()))


def _smooth_step(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(np.asarray(z, dtype=float)))


def _alpha_union_semicircles(
    x,
    y,
    *,
    centers_x: list[float],
    radii: list[float],
    wall_y: float,
    eps: float,
) -> np.ndarray:
    """
    Union of semicircular colonies attached to a wall y=wall_y.

    Each colony is the upper half of a disk centered at (cx, wall_y).
    The indicator alpha uses a diffuse tanh profile with half-thickness ~eps.
    """
    eps = max(float(eps), 1.0e-12)
    xx, yy = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    a = np.zeros_like(xx, dtype=float)
    for cx, r in zip(centers_x, radii):
        d = np.sqrt((xx - float(cx)) ** 2 + (yy - float(wall_y)) ** 2)
        a_i = _smooth_step((float(r) - d) / (2.0 * eps))
        a = np.maximum(a, a_i)
    return np.clip(a, 0.0, 1.0)


def _height_profile_alpha_half(
    *,
    dh: DofHandler,
    alpha: Function,
    alpha_half: float = 0.5,
    x_round: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Thickness profile l(x) from nodal alpha by per-x-column crossing detection.

    For each x-column (grouped by rounding), we locate the highest y where alpha
    crosses alpha_half from above to below and linearly interpolate the crossing.
    """
    coords = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    vals = np.asarray(alpha.nodal_values, dtype=float).ravel()
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Unexpected alpha dof coordinates shape.")
    if vals.size != coords.shape[0]:
        raise ValueError("alpha.nodal_values size mismatch.")

    # group indices by x (rounded)
    x_keys = np.round(coords[:, 0], decimals=int(x_round))
    groups: dict[float, list[int]] = {}
    for i, k in enumerate(x_keys.tolist()):
        groups.setdefault(float(k), []).append(int(i))

    xs: list[float] = []
    ls: list[float] = []
    ah = float(alpha_half)
    for xk in sorted(groups.keys()):
        idx = np.asarray(groups[xk], dtype=int)
        yy = coords[idx, 1]
        aa = vals[idx]
        order = np.argsort(yy)
        yy = yy[order]
        aa = aa[order]

        # Find crossing alpha >= ah -> alpha < ah as y increases.
        y_int = float("nan")
        for j in range(int(yy.size) - 1):
            a0 = float(aa[j])
            a1 = float(aa[j + 1])
            if (a0 >= ah) and (a1 < ah) and (a1 != a0):
                t01 = (ah - a0) / (a1 - a0)
                y_int = float(yy[j] + t01 * (yy[j + 1] - yy[j]))
        if not math.isfinite(y_int):
            # Fallback: take max y where alpha >= ah (coarse).
            mask = aa >= ah
            if np.any(mask):
                y_int = float(np.max(yy[mask]))
            else:
                y_int = float(0.0)

        xs.append(float(xk))
        ls.append(float(max(0.0, y_int)))

    return np.asarray(xs, dtype=float), np.asarray(ls, dtype=float)


def _biofilm_top_y(*, dh: DofHandler, alpha: Function, alpha_half: float = 0.5) -> float:
    _, l = _height_profile_alpha_half(dh=dh, alpha=alpha, alpha_half=float(alpha_half))
    return float(np.max(l)) if l.size else 0.0


def _update_phi_from_alpha(
    *,
    phi: Function,
    alpha: Function,
    phi_b: float,
    mode: str = "mix",
    alpha0: float = 0.1,
    alpha_width: float = 0.05,
) -> None:
    """
    Set porosity field from the diffuse indicator.

    Modes
    -----
    - mode='const': constant biofilm porosity everywhere
        phi = phi_b
      This is useful when validating against sharp-interface models (like Duddu 2007)
      where the biofilm porosity is assumed uniform and the diffuse-interface weighting
      in B=alpha*(1-phi) would otherwise introduce an unintended alpha^2 factor when
      using mode='mix'.

    - mode='mix' (default): smooth mixture interpolation
        phi = (1-alpha)*1 + alpha*phi_b
      This is the most direct way to represent a frozen biofilm porosity and
      a pure-fluid value phi=1 outside.

    - mode='sharp': sharp-but-smooth mapping in alpha-space
        phi = 1 - (1-phi_b) * H_alpha(alpha),
      where H_alpha transitions from 0→1 around alpha≈alpha0 with width alpha_width.
      This makes phi reach ~phi_b already near the alpha=0.5 contour, which can
      reduce unintended extra localization from (1-phi)*alpha when phi is used as
      a coefficient (e.g. in substrate uptake).
    """
    a = np.asarray(alpha.nodal_values, dtype=float)
    mode_key = str(mode).strip().lower()
    if mode_key in {"const", "constant", "uniform"}:
        phi.nodal_values[:] = float(phi_b)
        return
    if mode_key in {"mix", "mixture"}:
        phi.nodal_values[:] = 1.0 - (1.0 - float(phi_b)) * a
        return
    if mode_key in {"sharp", "step", "threshold"}:
        w = _smooth_step((a - float(alpha0)) / (2.0 * max(float(alpha_width), 1.0e-12)))
        phi.nodal_values[:] = 1.0 - (1.0 - float(phi_b)) * w
        return
    raise ValueError(f"Unknown phi update mode {mode!r}. Use 'const', 'mix', or 'sharp'.")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    keys = list(rows[0].keys())

    def _py(val: object) -> object:
        try:
            if isinstance(val, np.generic):
                return val.item()
        except Exception:
            pass
        return val

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _py(r.get(k)) for k in keys})


def _write_snaps_npz(path: Path, snaps: list[tuple[float, np.ndarray]]) -> None:
    """
    Persist alpha snapshots so partial results survive non-graceful exits
    (e.g. PETSc abort / SIGTERM).

    The file contains:
      - t_days: (n,) float64
      - alpha:  (n, ndof) float32
    """
    if not snaps:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.asarray([float(tt) for tt, _ in snaps], dtype=float)
    a = np.stack([np.asarray(arr, dtype=np.float32) for _, arr in snaps], axis=0)
    np.savez_compressed(path, t_days=t, alpha=a)


def _reinitialize_alpha_from_signed_distance(
    *,
    dh: DofHandler,
    alpha: Function,
    nx: int,
    ny: int,
    L: float,
    H: float,
    eps_alpha: float,
    alpha_half: float = 0.5,
    refine: int = 1,
) -> None:
    """
    Reinitialize the diffuse indicator α to a tanh profile based on the signed distance
    to the current α=alpha_half contour.

    This mimics the level-set reinitialization used in Duddu (2007) but operates directly
    on the CG1 nodal α grid (uniform (nx+1)×(ny+1) here).
    """
    try:
        from scipy.ndimage import distance_transform_edt  # type: ignore
    except Exception as exc:  # noqa: PERF203
        raise RuntimeError(
            "SciPy is required for --alpha-reinit-every (missing scipy.ndimage.distance_transform_edt)."
        ) from exc

    nx = int(nx)
    ny = int(ny)
    refine = int(refine)
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1 for alpha reinitialization.")
    if refine < 1:
        raise ValueError("refine must be >= 1 for alpha reinitialization.")
    dx = float(L) / float(nx)
    dy = float(H) / float(ny)
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("L and H must be positive for alpha reinitialization.")
    eps = max(float(eps_alpha), 1.0e-12)

    coords = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    vals = np.asarray(alpha.nodal_values, dtype=float).ravel()
    if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] != vals.size:
        raise ValueError("Unexpected alpha dof coordinates/values shape.")

    ii = np.rint(coords[:, 0] / dx).astype(int)
    jj = np.rint(coords[:, 1] / dy).astype(int)
    ii = np.clip(ii, 0, nx)
    jj = np.clip(jj, 0, ny)

    grid_c = np.zeros((ny + 1, nx + 1), dtype=float)
    grid_c[jj, ii] = vals

    if refine == 1:
        grid = grid_c
        sampling = (dy, dx)
        jj_s = jj
        ii_s = ii
    else:
        # Build a finer auxiliary grid by bilinear interpolation so the signed-distance
        # interface can move sub-cell relative to the CG1 nodal grid (reduces reinit locking).
        nx_f = nx * refine
        ny_f = ny * refine
        x_c = np.linspace(0.0, float(L), nx + 1)
        y_c = np.linspace(0.0, float(H), ny + 1)
        x_f = np.linspace(0.0, float(L), nx_f + 1)
        y_f = np.linspace(0.0, float(H), ny_f + 1)

        # Interpolate in x for each coarse y-row.
        tmp = np.empty((ny + 1, nx_f + 1), dtype=float)
        for j in range(ny + 1):
            tmp[j, :] = np.interp(x_f, x_c, grid_c[j, :])
        # Interpolate in y for each fine x-column.
        grid = np.empty((ny_f + 1, nx_f + 1), dtype=float)
        for i in range(nx_f + 1):
            grid[:, i] = np.interp(y_f, y_c, tmp[:, i])

        sampling = (dy / float(refine), dx / float(refine))

        # Coarse node indices on the fine grid.
        jj_s = jj * refine
        ii_s = ii * refine
        jj_s = np.clip(jj_s, 0, ny_f)
        ii_s = np.clip(ii_s, 0, nx_f)

    inside = grid >= float(alpha_half)
    d_to_outside = distance_transform_edt(inside, sampling=sampling)
    d_to_inside = distance_transform_edt(~inside, sampling=sampling)
    sd = np.asarray(d_to_inside - d_to_outside, dtype=float)  # negative inside, positive outside

    a_new = 0.5 * (1.0 - np.tanh(sd / (2.0 * eps)))
    a_new = np.clip(a_new, 0.0, 1.0)
    alpha.nodal_values[:] = a_new[jj_s, ii_s]


def _alpha_sharpen_values(a: np.ndarray, *, power: float) -> np.ndarray:
    """
    Contrast-enhancing monotone map that preserves the α=0.5 contour:

      α ↦ α^p / (α^p + (1-α)^p),  p>0

    For p>1 this pushes values towards {0,1} and counteracts numerical diffusion.
    """
    p = float(power)
    if p <= 0.0:
        return np.asarray(a, dtype=float)
    aa = np.asarray(a, dtype=float)
    aa = np.clip(aa, 0.0, 1.0)
    eps = 1.0e-12
    aa = np.clip(aa, eps, 1.0 - eps)
    ap = aa**p
    bp = (1.0 - aa) ** p
    return ap / (ap + bp)


def _plot_interface_contours(
    *,
    tri,
    snaps: list[tuple[float, np.ndarray]],
    alpha_half: float,
    outpng: Path,
    L: float,
    H: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    for t, a in snaps:
        ax.tricontour(tri, np.asarray(a, dtype=float), levels=[float(alpha_half)], colors="k", linewidths=0.8, alpha=0.9)
    ax.set_xlim(0.0, float(L))
    ax.set_ylim(0.0, float(H))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Interface motion (alpha=0.5)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    outpng.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpng, dpi=200)
    plt.close(fig)


def _plot_scalar_field(
    *,
    tri,
    values: np.ndarray,
    title: str,
    outpng: Path,
    cmap: str,
    levels: int = 20,
    overlay_iface: tuple[object, np.ndarray, float] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    tcf = ax.tricontourf(tri, np.asarray(values, dtype=float), levels=int(levels), cmap=str(cmap))
    fig.colorbar(tcf, ax=ax, shrink=0.85)
    if overlay_iface is not None:
        tri_a, a_vals, alpha_half = overlay_iface
        ax.tricontour(tri_a, np.asarray(a_vals, dtype=float), levels=[float(alpha_half)], colors="k", linewidths=0.9)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(str(title))
    ax.set_aspect("equal")
    outpng.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpng, dpi=200)
    plt.close(fig)


@dataclass(frozen=True)
class Duddu2007Params:
    # active biomass fraction
    f_active: float = 0.5
    # densities (mgVS/mm^3)
    rho_x: float = 1.0250
    rho_w: float = 1.0125
    # yields (mgVS/mgO2)
    Y_xO: float = 0.583
    Y_wO: float = 0.215
    # kinetics
    qhat0: float = 8.0  # mgO2/(mgVS day)
    K0: float = 5.0e-7  # mgO2/mm^3
    b: float = 0.3  # 1/day
    f_D: float = 0.8
    g: float = 1.42  # mgO2/mgVS

    def monod(self, S):
        K0 = Constant(float(self.K0))
        return S / (S + K0)

    def divU(self, S):
        f = Constant(float(self.f_active))
        rho_x = Constant(float(self.rho_x))
        rho_w = Constant(float(self.rho_w))
        Y_xO = Constant(float(self.Y_xO))
        Y_wO = Constant(float(self.Y_wO))
        qhat0 = Constant(float(self.qhat0))
        b = Constant(float(self.b))
        f_D = Constant(float(self.f_D))
        mon = self.monod(S)
        rho_x_rate = (Y_xO * qhat0 - b) * mon
        rho_w_rate = (rho_x / rho_w) * ((Constant(1.0) - f_D) * b + Y_wO * qhat0) * mon
        return f * (rho_x_rate + rho_w_rate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="examples/biofilms/benchmarks/dadu/results/duddu2007_one_domain_fig6_example2")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--linear-solver", type=str, default="petsc", choices=("petsc", "scipy"))
    ap.add_argument("--q", type=int, default=4)
    ap.add_argument(
        "--include-fluid",
        action="store_true",
        help="Solve v along with (p,vS) in step (2). This carries the global volume-production outflow with v instead of forcing vS to extend through the pure fluid.",
    )

    # Geometry (mm)
    ap.add_argument("--L", type=float, default=0.5)
    ap.add_argument("--H", type=float, default=0.5)
    ap.add_argument("--nx", type=int, default=60, help="Quad divisions in x (mesh poly_order=2).")
    ap.add_argument("--ny", type=int, default=60, help="Quad divisions in y (mesh poly_order=2).")

    # Initial colonies
    ap.add_argument("--centers-x", type=str, default="0.05,0.25,0.45")
    ap.add_argument("--radii", type=str, default="0.01,0.02,0.01")
    ap.add_argument("--eps-alpha", type=float, default=0.01, help="Diffuse interface half-thickness (mm).")
    ap.add_argument("--phi-b", type=float, default=0.3, help="Frozen biofilm porosity.")
    ap.add_argument(
        "--phi-update",
        choices=("const", "mix", "sharp"),
        default="mix",
        help="How to update the frozen porosity proxy phi from alpha (see script docstring).",
    )
    ap.add_argument("--phi-alpha0", type=float, default=0.1, help="(sharp mode) alpha switch location.")
    ap.add_argument("--phi-alpha-width", type=float, default=0.05, help="(sharp mode) alpha transition width.")

    # Substrate BC (paper: moving line Ls above top-most biofilm point)
    ap.add_argument("--Ls", type=float, default=0.1)
    ap.add_argument("--Sbar", type=float, default=8.3e-6)
    ap.add_argument("--S-penalty", type=float, default=1.0e6)

    # Effective substrate diffusion (mm^2/day) in the one-domain surrogate.
    ap.add_argument("--D-S", type=float, default=138.5)
    ap.add_argument(
        "--mu-max-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the mapped Monod max rate mu_max (affects substrate uptake strength).",
    )

    # (p, vS) parameters (see duddu2007_one_domain_slab_speed.py for calibration)
    ap.add_argument("--kappa-inv", type=float, default=8.0)
    ap.add_argument("--mu-f", type=float, default=1.0)
    ap.add_argument(
        "--s-v-mode",
        type=str,
        default="auto",
        choices=("auto", "divU", "BdivU"),
        help="Volumetric source in the mass constraint. auto: BdivU for alpha_advect_with=vS, divU for alpha_advect_with=mix/mix_biofilm. divU: s_v=divU(S). BdivU: s_v=(1-phi)*divU(S) so alpha*s_v = B*divU with B=alpha*(1-phi).",
    )
    ap.add_argument("--gamma-vS", type=float, default=1.0)
    ap.add_argument("--vS-ext-mode", type=str, default="grad", choices=("l2", "grad"))
    ap.add_argument("--gamma-vS-pin", type=float, default=0.0)
    ap.add_argument(
        "--gamma-vS-pin-power",
        type=int,
        default=2,
        help="Exponent for the vS L2 pin weight in the fluid: (1-alpha)^m. Use m>2 to localize pinning away from the diffuse interface.",
    )
    ap.add_argument(
        "--vS-top-bc",
        type=str,
        default="pin",
        choices=("pin", "open"),
        help="(include-fluid only) Top boundary condition for vS_y. pin: enforce vS_y=0 on y=H (previous behavior). open: do not constrain vS_y on y=H (often closer to the Duddu(2007) velocity-potential extension).",
    )
    ap.add_argument("--gamma-p-out", type=float, default=1.0e6)
    ap.add_argument("--gamma-p-out-power", type=int, default=100)

    # Alpha regularization (optional)
    ap.add_argument("--D-alpha", type=float, default=0.0)
    ap.add_argument("--ac-M", type=float, default=0.0)
    ap.add_argument("--ac-gamma", type=float, default=0.0)
    ap.add_argument(
        "--ac-mobility",
        type=str,
        default="constant",
        choices=("constant", "degenerate"),
        help="Allen–Cahn mobility for alpha regularization. Use 'degenerate' to localize regularization to the diffuse interface.",
    )
    ap.add_argument(
        "--ac-mobility-floor",
        type=float,
        default=0.0,
        help="(degenerate mobility) Optional mobility floor m_floor in M(α)=M0(α(1-α)+m_floor).",
    )
    ap.add_argument(
        "--alpha-advect-with",
        type=str,
        default="vS",
        choices=("vS", "v", "mix", "mix_biofilm"),
        help="Velocity used for alpha advection: vS=skeleton, v=fluid, mix=C v + B vS (mixture/volume velocity), mix_biofilm=like mix but gate Cv to avoid advecting alpha through pure fluid.",
    )
    ap.add_argument(
        "--alpha-mix-gate-alpha0",
        type=float,
        default=0.1,
        help="(mix_biofilm) Gate cutoff alpha0 in g(alpha)=alpha^m/(alpha^m+alpha0^m). Use a smaller value (e.g. 0.01) to keep the advector close to 'mix' while still suppressing transport for tiny alpha in the bulk fluid.",
    )
    ap.add_argument(
        "--alpha-mix-gate-power",
        type=int,
        default=4,
        help="(mix_biofilm) Gate power m in g(alpha)=alpha^m/(alpha^m+alpha0^m). Larger m gives a sharper cutoff.",
    )
    ap.add_argument("--alpha-solver", choices=("newton", "pdas"), default="newton", help="Alpha update solver (linear advection is fastest with Newton+clip).")
    ap.add_argument("--alpha-supg", type=float, default=1.0, help="SUPG stabilization strength for the alpha advection solve (recommended for D_alpha=0).")
    ap.add_argument(
        "--alpha-cleanup-cut",
        type=float,
        default=0.0,
        help=(
            "Optional numerical cleanup after each alpha update: set alpha<cut -> 0 and alpha>(1-cut) -> 1. "
            "This mimics a mild reinitialization by removing diffuse tails that can seed spurious chimneys."
        ),
    )
    ap.add_argument(
        "--alpha-reinit-every",
        type=int,
        default=0,
        help=(
            "Optional reinitialization of alpha every N accepted steps using a signed-distance EDT on the nodal alpha grid. "
            "Use 1 to mimic the level-set reinitialization used in Duddu (2007). 0 disables."
        ),
    )
    ap.add_argument(
        "--alpha-reinit-refine",
        type=int,
        default=1,
        help=(
            "Refinement factor for the alpha reinitialization grid (>=1). "
            "Use 4–8 to reduce interface locking when --alpha-reinit-every is enabled."
        ),
    )
    ap.add_argument(
        "--alpha-sharpen-power",
        type=float,
        default=0.0,
        help=(
            "Optional post-update sharpening of alpha via a monotone map that preserves the alpha=0.5 contour. "
            "Use values like 2–6 to counteract numerical diffusion (0 disables)."
        ),
    )
    ap.add_argument(
        "--alpha-sharpen-every",
        type=int,
        default=1,
        help="Apply alpha sharpening every N accepted steps (default: 1).",
    )

    # Time stepping (days)
    ap.add_argument("--t-final", type=float, default=28.6)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument(
        "--cfl-dt",
        action="store_true",
        help=(
            "Limit the growth step by a CFL-like condition based on the current maximum "
            "advection speed and the mesh size: dt <= cfl*h/max|u|. This mimics the "
            "explicit level-set stability limit used by Duddu (2007) (Eq. (39))."
        ),
    )
    ap.add_argument("--cfl", type=float, default=0.8, help="CFL safety factor used with --cfl-dt (typical: 0.5–0.9).")
    ap.add_argument("--adaptive-dt", action="store_true", help="Retry failed steps with reduced dt (recommended for long runs).")
    ap.add_argument("--dt-min", type=float, default=1.0e-4)
    ap.add_argument("--dt-reduction-factor", type=float, default=0.5)
    ap.add_argument("--dt-steady", type=float, default=1.0e6, help="Large dt used to approximate quasi-steady S (days).")
    ap.add_argument("--snap-every", type=int, default=1, help="Record interface snapshot every N accepted steps.")
    ap.add_argument(
        "--flush-snaps-every",
        type=int,
        default=20,
        help="Write/update alpha snapshot NPZ every N accepted steps (0 disables).",
    )
    ap.add_argument(
        "--snaps-npz",
        type=str,
        default="",
        help="Optional path for the alpha snapshot NPZ (default: <outdir>/snaps_alpha.npz).",
    )
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument(
        "--write-every",
        type=int,
        default=1,
        help="Write y_top_timeseries.csv every N accepted steps (use 1 to keep partial results on interrupts).",
    )
    ap.add_argument("--skip-plots", action="store_true", help="Skip matplotlib plots (faster for parameter sweeps).")

    # Output control
    ap.add_argument(
        "--targets",
        type=str,
        default="",
        help=(
            "Optional comma-separated list of times (days). When provided, the time step is split "
            "so the solver lands exactly on these targets (in addition to t_final)."
        ),
    )
    ap.add_argument("--vtk-full", action="store_true", help="Write VTK snapshots containing (alpha, phi, S, p, vS).")
    ap.add_argument(
        "--vtk-full-every",
        type=int,
        default=0,
        help="Export full-field VTK every N accepted steps (0 disables periodic export).",
    )
    ap.add_argument(
        "--vtk-dir",
        type=str,
        default="",
        help="Optional VTK output directory (default: <outdir>/vtk_full).",
    )

    # Newton/VI
    ap.add_argument("--newton-tol", type=float, default=1.0e-10)
    ap.add_argument("--max-it", type=int, default=40)
    ap.add_argument(
        "--solver-print-level",
        type=int,
        default=0,
        help="Newton/PDAS verbosity: 0=quiet, 1=step summary, 2=per-Newton-iter (very noisy).",
    )
    ap.add_argument("--vi-c", type=float, default=0.0)
    ap.add_argument("--vi-active-tol", type=float, default=0.0)
    ap.add_argument(
        "--substrate-solver",
        choices=("pdas", "newton"),
        default="pdas",
        help="Solver for the quasi-steady substrate VI (pdas=enforce S>=0 via PDAS; newton=Newton + clip).",
    )
    ap.add_argument(
        "--substrate-advection",
        choices=("off", "lagged"),
        default="off",
        help=(
            "Whether to include the fluid-advection term div(C S v) in the substrate solve. "
            "Duddu (2007) neglects substrate advection; use 'off' to match the paper. "
            "'lagged' keeps the current behavior (use previous-step v in step (1))."
        ),
    )

    args = ap.parse_args()

    if bool(args.include_fluid) and str(args.alpha_advect_with).strip().lower().startswith("mix"):
        print(
            "[warn] include-fluid + alpha-advect-with=mix* advects α using the fluid velocity component; "
            "for the Duddu(2007) growth benchmark this can create an unphysical vertical 'chimney' due to "
            "top outflow. Recommended: --alpha-advect-with vS."
        )

    if str(args.linear_solver).lower() == "petsc" and not HAS_PETSC:
        raise RuntimeError("PETSc requested but not available in this environment.")

    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    L = float(args.L)
    H = float(args.H)
    nx = int(args.nx)
    ny = int(args.ny)
    qdeg = int(args.q)

    centers_x = [float(s.strip()) for s in str(args.centers_x).split(",") if s.strip()]
    radii = [float(s.strip()) for s in str(args.radii).split(",") if s.strip()]
    if len(centers_x) != len(radii):
        raise ValueError("--centers-x and --radii must have the same length.")

    eps_alpha = float(args.eps_alpha)
    phi_b = float(args.phi_b)
    if not (0.0 < phi_b < 1.0):
        raise ValueError("--phi-b must be in (0,1).")

    # --- mesh ---------------------------------------------------------------
    nodes, elems, _edges, corners = structured_quad(L, H, nx=nx, ny=ny, poly_order=2, offset=(0.0, 0.0))
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    _tag_rectangle_boundaries(mesh, L=L, H=H)

    # MixedElement space (keep full set; we will inactivate most fields).
    field_specs: dict[str, object] = {
        "v_x": 2,
        "v_y": 2,
        "p": 1,
        "vS_x": 2,
        "vS_y": 2,
        "u_x": 2,
        "u_y": 2,
        "phi": 1,
        "alpha": 1,
        "S": 1,
    }
    me = MixedElement(mesh, field_specs=field_specs)
    dh = DofHandler(me, method="cg")

    # Spaces / trial/test functions
    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    VS = FunctionSpace("VS", ["vS_x", "vS_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    dv = VectorTrialFunction(space=V, dof_handler=dh)
    dvS = VectorTrialFunction(space=VS, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    dphi = TrialFunction("phi", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    dS = TrialFunction("S", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    vS_test = VectorTestFunction(space=VS, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    S_test = TestFunction("S", dof_handler=dh)

    # Functions at k and n
    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    vS_k = VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    S_k = Function("S_k", "S", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    vS_n = VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    S_n = Function("S_n", "S", dof_handler=dh)

    # Initial conditions / frozen fields
    for vf in (v_k, vS_k, u_k, v_n, vS_n, u_n):
        vf.nodal_values[:] = 0.0
    for sf in (p_k, p_n):
        sf.nodal_values[:] = 0.0

    alpha_n.set_values_from_function(
        lambda x, y: _alpha_union_semicircles(x, y, centers_x=centers_x, radii=radii, wall_y=0.0, eps=eps_alpha)
    )
    alpha_k.nodal_values[:] = alpha_n.nodal_values[:]

    # Porosity: set to phi_b in biofilm (alpha≈1) and 1 in fluid (alpha≈0).
    _update_phi_from_alpha(
        phi=phi_n,
        alpha=alpha_n,
        phi_b=phi_b,
        mode=str(args.phi_update),
        alpha0=float(args.phi_alpha0),
        alpha_width=float(args.phi_alpha_width),
    )
    phi_k.nodal_values[:] = phi_n.nodal_values[:]

    Sbar = float(args.Sbar)
    S_n.set_values_from_function(lambda x, y: float(Sbar))
    S_k.nodal_values[:] = S_n.nodal_values[:]

    # Time constant used by one_domain forms (updated per block).
    dt_c = Constant(float(args.dt))
    dx_q = dx(metadata={"q": int(qdeg)})

    # Duddu(2007) kinetics mapping (consumption and divU source).
    kin = Duddu2007Params()

    one_m_phi_b = 1.0 - float(phi_b)
    if one_m_phi_b <= 0.0:
        raise ValueError("--phi-b must be < 1.")

    # Growth source for constraint.
    #
    # In Duddu(2007), div(U)=divU(S) holds in the biofilm, where U is the biofilm
    # advection velocity derived from the potential Φ. In the one-domain model we
    # enforce the volume constraint:
    #   div(F) = α s_v,   with F = C v + B vS.
    #
    # Depending on which velocity we use to advect α:
    # - alpha_advect_with="vS": we want div(vS)≈divU in the bulk (α≈1, φ≈φ_b), so we
    #   set s_v = (1-φ) divU(S) so that α s_v = B divU.
    # - alpha_advect_with="mix": we interpret F as the biofilm velocity, so we set
    #   s_v = divU(S) so that div(F)≈divU in the bulk.
    divU_k = kin.divU(S_k)
    sv_mode = str(args.s_v_mode).strip().lower()
    if sv_mode == "auto":
        adv_key = str(args.alpha_advect_with).strip().lower()
        sv_mode = "divu" if adv_key.startswith("mix") else "bdivu"
    if sv_mode in {"divu", "u"}:
        s_v = divU_k
    elif sv_mode in {"bdivu", "b*divu", "b"}:
        s_v = ((-phi_k) + Constant(1.0)) * divU_k
    else:
        raise ValueError(f"Unknown --s-v-mode {args.s_v_mode!r}. Use 'auto', 'divU', or 'BdivU'.")
    ds_v = Constant(0.0)

    # Substrate uptake mapping (see duddu2007_one_domain_slab_speed.py).
    desired_uptake = kin.f_active * (kin.qhat0 + kin.g * kin.f_D * kin.b)  # [mgO2/(mgVS day)]
    mu_max = float(desired_uptake) * float(kin.Y_xO) / float(one_m_phi_b)  # [1/day]
    mu_max *= float(args.mu_max_scale)

    forms = build_biofilm_one_domain_forms(
        v_k=v_k,
        p_k=p_k,
        vS_k=vS_k,
        u_k=u_k,
        phi_k=phi_k,
        alpha_k=alpha_k,
        S_k=S_k,
        v_n=v_n,
        p_n=p_n,
        vS_n=vS_n,
        u_n=u_n,
        phi_n=phi_n,
        alpha_n=alpha_n,
        S_n=S_n,
        dv=dv,
        dp=dp,
        dvS=dvS,
        du=du,
        dphi=dphi,
        dalpha=dalpha,
        dS=dS,
        v_test=v_test,
        q_test=q_test,
        vS_test=vS_test,
        u_test=u_test,
        phi_test=phi_test,
        alpha_test=alpha_test,
        S_test=S_test,
        dx=dx_q,
        dt=dt_c,
        theta=1.0,
        rho_f=Constant(0.0),
        mu_f=Constant(float(args.mu_f)),
        kappa_inv=Constant(float(args.kappa_inv)),
        mu_b_model="phi_mu",
        solid_model="linear",
        mu_s=Constant(1.0e-8),
        lambda_s=Constant(1.0e-8),
        solid_visco_eta=0.0,
        include_skeleton_acceleration=False,
        # freeze phi equation (phi is inactive)
        D_phi=0.0,
        gamma_phi=0.0,
        # alpha
        D_alpha=float(args.D_alpha),
        alpha_advect_with=str(args.alpha_advect_with),
        alpha_mix_gate_alpha0=float(args.alpha_mix_gate_alpha0),
        alpha_mix_gate_power=int(args.alpha_mix_gate_power),
        alpha_advection_form="advective",
        alpha_cahn_M=float(args.ac_M),
        alpha_cahn_gamma=float(args.ac_gamma),
        alpha_cahn_eps=float(eps_alpha),
        alpha_cahn_mobility=str(args.ac_mobility),
        alpha_cahn_mobility_floor=float(args.ac_mobility_floor),
        alpha_cahn_conservative=False,
        alpha_cahn_conservative_mode="eliminate",
        alpha_supg=float(args.alpha_supg),
        # substrate
        D_S=float(args.D_S),
        substrate_reaction_scheme="implicit",
        substrate_diffusion_scheme="implicit",
        mu_max=float(mu_max),
        K_S=float(kin.K0),
        k_g=0.0,
        k_d=0.0,
        Y=float(kin.Y_xO),
        rho_s_star=float(kin.rho_x),
        k_det=0.0,
        s_v=s_v,
        ds_v=ds_v,
        D_det_prev=Constant(0.0),
        # vS extension
        gamma_vS=float(args.gamma_vS),
        vS_extension_mode=str(args.vS_ext_mode),
        gamma_vS_pin=float(args.gamma_vS_pin),
        gamma_vS_pin_power=int(args.gamma_vS_pin_power),
    )

    # Moving substrate Dirichlet line via penalty on dInterface (CutFEM).
    Ls = float(args.Ls)
    if Ls < 0.0:
        raise ValueError("--Ls must be >= 0.")
    y_top0 = _biofilm_top_y(dh=dh, alpha=alpha_n, alpha_half=0.5)
    # IMPORTANT: avoid exact alignment of Γ_S^d with mesh edges/nodes (can make
    # the CutFEM interface measure empty / singular on structured meshes).
    y_D0 = float(min(float(H) - 1.0e-12, y_top0 + Ls + 1.0e-10))
    ls_Sd = AffineLevelSet(0.0, 1.0, -float(y_D0))
    dGammaS = dInterface(level_set=ls_Sd, metadata={"q": int(qdeg), "linear_interface": True})
    penS = Constant(float(args.S_penalty))
    Sbar_c = Constant(float(Sbar))
    h = CellDiameter()
    r_Spen = (penS / h) * (S_k - Sbar_c) * S_test * dGammaS
    a_Spen = (penS / h) * dS * S_test * dGammaS

    # (p,vS) p-out penalty (keeps p DOFs in the pure fluid well-posed).
    m_pow = int(max(1, int(args.gamma_p_out_power)))
    one_m_alpha = (-alpha_k) + Constant(1.0)
    w_p_out = one_m_alpha
    for _ in range(m_pow - 1):
        w_p_out = w_p_out * one_m_alpha
    gamma_p_out = Constant(float(args.gamma_p_out))
    r_p_out = gamma_p_out * w_p_out * p_k * q_test * dx_q
    a_p_out = gamma_p_out * w_p_out * dp * q_test * dx_q

    # Boundary conditions for (p, vS): normal components on outer boundary.
    bc_vx_left = BoundaryCondition("v_x", "dirichlet", "left", _as_float_time(lambda x, y, t: 0.0))
    bc_vx_left_h = BoundaryCondition("v_x", "dirichlet", "left", (lambda x, y: 0.0))
    bc_vx_right = BoundaryCondition("v_x", "dirichlet", "right", _as_float_time(lambda x, y, t: 0.0))
    bc_vx_right_h = BoundaryCondition("v_x", "dirichlet", "right", (lambda x, y: 0.0))
    bc_vy_bottom = BoundaryCondition("v_y", "dirichlet", "bottom", _as_float_time(lambda x, y, t: 0.0))
    bc_vy_bottom_h = BoundaryCondition("v_y", "dirichlet", "bottom", (lambda x, y: 0.0))

    bc_vSx_left = BoundaryCondition("vS_x", "dirichlet", "left", _as_float_time(lambda x, y, t: 0.0))
    bc_vSx_left_h = BoundaryCondition("vS_x", "dirichlet", "left", (lambda x, y: 0.0))
    bc_vSx_right = BoundaryCondition("vS_x", "dirichlet", "right", _as_float_time(lambda x, y, t: 0.0))
    bc_vSx_right_h = BoundaryCondition("vS_x", "dirichlet", "right", (lambda x, y: 0.0))
    bc_vSy_bottom = BoundaryCondition("vS_y", "dirichlet", "bottom", _as_float_time(lambda x, y, t: 0.0))
    bc_vSy_bottom_h = BoundaryCondition("vS_y", "dirichlet", "bottom", (lambda x, y: 0.0))
    bc_vSy_top = BoundaryCondition("vS_y", "dirichlet", "top", _as_float_time(lambda x, y, t: 0.0))
    bc_vSy_top_h = BoundaryCondition("vS_y", "dirichlet", "top", (lambda x, y: 0.0))
    # Pressure gauge
    bc_p_top = BoundaryCondition("p", "dirichlet", "top", _as_float_time(lambda x, y, t: 0.0))
    bc_p_top_h = BoundaryCondition("p", "dirichlet", "top", (lambda x, y: 0.0))

    # IMPORTANT:
    # - If we only solve (p,vS) with v≡0, the fixed-domain constraint requires an
    #   outflow in B vS; we therefore leave the top boundary "open" for vS.
    # - If we also solve v, we can carry the outflow with v (C v) and safely
    #   pin vS_y=0 on the top boundary, which avoids advecting alpha through the
    #   pure fluid.
    if bool(args.include_fluid):
        vS_top_bc = str(args.vS_top_bc).strip().lower()
        bcs_pvS = [
            bc_vx_left,
            bc_vx_right,
            bc_vy_bottom,
            bc_vSx_left,
            bc_vSx_right,
            bc_vSy_bottom,
            bc_p_top,
        ]
        bcs_pvS_homog = [
            bc_vx_left_h,
            bc_vx_right_h,
            bc_vy_bottom_h,
            bc_vSx_left_h,
            bc_vSx_right_h,
            bc_vSy_bottom_h,
            bc_p_top_h,
        ]
        if vS_top_bc == "pin":
            bcs_pvS.insert(-1, bc_vSy_top)
            bcs_pvS_homog.insert(-1, bc_vSy_top_h)
    else:
        bcs_pvS = [bc_vSx_left, bc_vSx_right, bc_vSy_bottom, bc_p_top]
        bcs_pvS_homog = [bc_vSx_left_h, bc_vSx_right_h, bc_vSy_bottom_h, bc_p_top_h]

    # --- solvers -----------------------------------------------------------
    newton_tol = float(args.newton_tol)
    max_it = int(args.max_it)
    print_level = int(max(0, int(args.solver_print_level)))
    lin_backend = str(args.linear_solver)
    backend = str(args.backend)

    # Substrate: only S active
    inactive_S = ["v_x", "v_y", "p", "vS_x", "vS_y", "u_x", "u_y", "phi", "alpha"]
    _set_inactive_fields(dh, inactive_S)
    substrate_solver_mode = str(args.substrate_solver).strip().lower()
    if substrate_solver_mode == "pdas":
        solver_S = PdasNewtonSolver(
            forms.r_substrate + r_Spen,
            forms.a_substrate + a_Spen,
            dof_handler=dh,
            mixed_element=me,
            bcs=[],
            bcs_homog=[],
            vi_params=VIParameters(
                c=float(args.vi_c),
                active_tol=float(args.vi_active_tol),
                project_initial_guess=True,
                project_each_iteration=True,
            ),
            newton_params=NewtonParameters(
                newton_tol=newton_tol,
                newton_rtol=0.0,
                max_newton_iter=max_it,
                print_level=print_level,
                # NOTE: line search can fail on larger meshes for this VI block (no
                # residual decrease even though the PDAS update is feasible). Full-step
                # PDAS is robust here because we enforce S>=0 explicitly.
                line_search=False,
                ls_mode="dealii",
            ),
            lin_params=LinearSolverParameters(backend=lin_backend),
            quad_order=qdeg,
            backend=backend,
        )
        solver_S.set_box_bounds(by_field={"S": (0.0, None)})
        substrate_needs_clip = False
    elif substrate_solver_mode == "newton":
        solver_S = NewtonSolver(
            forms.r_substrate + r_Spen,
            forms.a_substrate + a_Spen,
            dof_handler=dh,
            mixed_element=me,
            bcs=[],
            bcs_homog=[],
            newton_params=NewtonParameters(
                newton_tol=newton_tol,
                newton_rtol=0.0,
                max_newton_iter=max_it,
                print_level=print_level,
                line_search=True,
                ls_mode="dealii",
            ),
            lin_params=LinearSolverParameters(backend=lin_backend),
            quad_order=qdeg,
            backend=backend,
        )
        substrate_needs_clip = True
    else:
        raise ValueError(f"Unknown --substrate-solver {args.substrate_solver!r}.")

    # (p,vS) (or (v,p,vS)): keep only the intended PV unknowns active.
    inactive_pvS = ["u_x", "u_y", "phi", "alpha", "S"]
    if not bool(args.include_fluid):
        inactive_pvS += ["v_x", "v_y"]
    _set_inactive_fields(dh, inactive_pvS)
    if bool(args.include_fluid):
        r_pvS = forms.r_momentum + forms.r_mass + forms.r_skeleton + r_p_out
        a_pvS = forms.a_momentum + forms.a_mass + forms.a_skeleton + a_p_out
    else:
        r_pvS = forms.r_mass + forms.r_skeleton + r_p_out
        a_pvS = forms.a_mass + forms.a_skeleton + a_p_out
    solver_pvS = NewtonSolver(
        r_pvS,
        a_pvS,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs_pvS,
        bcs_homog=bcs_pvS_homog,
        newton_params=NewtonParameters(
            newton_tol=newton_tol,
            newton_rtol=0.0,
            max_newton_iter=max_it,
            print_level=print_level,
            line_search=False,
            ls_mode="dealii",
        ),
        lin_params=LinearSolverParameters(backend=lin_backend),
        quad_order=qdeg,
        backend=backend,
    )

    # Alpha: only alpha active
    inactive_alpha = ["v_x", "v_y", "p", "vS_x", "vS_y", "u_x", "u_y", "phi", "S"]
    _set_inactive_fields(dh, inactive_alpha)
    alpha_solver_key = str(args.alpha_solver).strip().lower()
    if alpha_solver_key == "pdas":
        solver_alpha = PdasNewtonSolver(
            forms.r_alpha,
            forms.a_alpha,
            dof_handler=dh,
            mixed_element=me,
            bcs=[],
            bcs_homog=[],
            vi_params=VIParameters(
                c=float(args.vi_c),
                active_tol=float(args.vi_active_tol),
                project_initial_guess=True,
                project_each_iteration=True,
            ),
            newton_params=NewtonParameters(
                newton_tol=newton_tol,
                newton_rtol=0.0,
                max_newton_iter=max_it,
                print_level=print_level,
                line_search=True,
                ls_mode="dealii",
            ),
            lin_params=LinearSolverParameters(backend=lin_backend),
            quad_order=qdeg,
            backend=backend,
        )
        solver_alpha.set_box_bounds(by_field={"alpha": (0.0, 1.0)})
        alpha_needs_clip = False
    else:
        # For the Duddu(2007) growth-only limit we typically keep ac_M=ac_gamma=D_alpha=0,
        # so the alpha equation is linear in alpha for frozen vS. A plain Newton solve
        # is significantly more robust than PDAS here; we then clip to [0,1].
        solver_alpha = NewtonSolver(
            forms.r_alpha,
            forms.a_alpha,
            dof_handler=dh,
            mixed_element=me,
            bcs=[],
            bcs_homog=[],
            newton_params=NewtonParameters(
                newton_tol=float(max(newton_tol, 1.0e-10)),
                newton_rtol=0.0,
                max_newton_iter=10,
                print_level=print_level,
                line_search=False,
                ls_mode="dealii",
            ),
            lin_params=LinearSolverParameters(backend=lin_backend),
            quad_order=qdeg,
            backend=backend,
        )
        alpha_needs_clip = True

    # --- time loop ---------------------------------------------------------
    functions = [v_k, p_k, vS_k, u_k, phi_k, alpha_k, S_k]
    prev_functions = [v_n, p_n, vS_n, u_n, phi_n, alpha_n, S_n]

    t_final = float(args.t_final)
    dt_growth = float(args.dt)
    dt_steady = float(args.dt_steady)
    if dt_growth <= 0.0:
        raise ValueError("--dt must be > 0.")
    if dt_steady <= 0.0:
        raise ValueError("--dt-steady must be > 0.")

    snaps: list[tuple[float, np.ndarray]] = [(0.0, np.asarray(alpha_n.nodal_values, dtype=float).copy())]
    y_top0 = _biofilm_top_y(dh=dh, alpha=alpha_n, alpha_half=0.5)
    y_D0 = float(min(float(H) - 1.0e-12, float(y_top0) + float(Ls) + 1.0e-10))
    a0 = np.asarray(alpha_n.nodal_values, dtype=float)
    rows: list[dict[str, object]] = [
        {
            "t_days": 0.0,
            "step": 0,
            "dt_days": 0.0,
            "y_top_mm": float(y_top0),
            "y_D_mm": float(y_D0),
            "alpha_min": float(np.min(a0)),
            "alpha_max": float(np.max(a0)),
        }
    ]
    snap_every = int(max(1, int(args.snap_every)))
    flush_snaps_every = int(max(0, int(args.flush_snaps_every)))
    snaps_npz = Path(str(args.snaps_npz)).expanduser() if str(args.snaps_npz).strip() else (outdir / "snaps_alpha.npz")
    progress_every = int(max(1, int(args.progress_every)))
    write_every = int(max(1, int(args.write_every)))

    # Optional target alignment + full-field VTK output.
    targets_all = [t for t in _parse_targets_csv(str(args.targets)) if t > 0.0]
    targets = [t for t in targets_all if t <= t_final + 1.0e-12]
    next_target_idx = 0

    vtk_full = bool(args.vtk_full)
    vtk_full_every = int(max(0, int(args.vtk_full_every)))
    vtk_dir = Path(str(args.vtk_dir)).expanduser() if str(args.vtk_dir).strip() else (outdir / "vtk_full")
    vtk_datasets: list[tuple[float, str]] = []
    vtk_step_no = 0
    if vtk_full:
        vtk_dir.mkdir(parents=True, exist_ok=True)

        def _write_full_vtk(t_days: float) -> None:
            nonlocal vtk_step_no, vtk_datasets
            out = vtk_dir / f"step={vtk_step_no:04d}.vtu"
            export_vtk(
                str(out),
                mesh,
                dh,
                {"alpha": alpha_n, "phi": phi_n, "S": S_n, "p": p_n, "vS": vS_n},
            )
            vtk_datasets.append((float(t_days), out.name))
            vtk_step_no += 1

        _write_full_vtk(0.0)

    t_n = 0.0
    step_no = 0
    dt_nom = float(dt_growth)
    dt_min = float(args.dt_min)
    red = float(args.dt_reduction_factor)
    use_cfl_dt = bool(args.cfl_dt)
    cfl = float(args.cfl)
    if use_cfl_dt and not (0.0 < cfl <= 1.0):
        raise ValueError("--cfl must be in (0,1] when --cfl-dt is enabled.")
    if dt_min <= 0.0:
        dt_min = 0.0
    if not (0.0 < red < 1.0):
        raise ValueError("--dt-reduction-factor must be in (0,1).")

    t0_wall = time.perf_counter()
    interrupted = False
    reinit_every = int(max(0, int(args.alpha_reinit_every)))
    reinit_refine = int(max(1, int(args.alpha_reinit_refine)))
    sharpen_power = float(args.alpha_sharpen_power)
    sharpen_every = int(max(1, int(args.alpha_sharpen_every)))
    try:
        while t_n < t_final - 1.0e-12:
            while next_target_idx < len(targets) and targets[next_target_idx] <= t_n + 1.0e-12:
                next_target_idx += 1

            step_no += 1
            dt_step = min(dt_nom, t_final - t_n)
            if next_target_idx < len(targets):
                dt_to_target = float(targets[next_target_idx] - t_n)
                if dt_to_target > 1.0e-12:
                    dt_step = min(dt_step, dt_to_target)

            # CFL-like dt restriction based on previous-step advection speed.
            # Duddu (2007) uses dt_max = min(dx,dy)/max(F); we approximate F using |u|
            # of the chosen alpha advector on the FE mesh.
            if use_cfl_dt:
                h_min = float(min(L / float(nx), H / float(ny)))
                max_speed = 0.0
                adv_key = str(args.alpha_advect_with).strip().lower()
                try:
                    if adv_key == "vs":
                        vx = np.asarray(vS_n.nodal_values_component(0), dtype=float)
                        vy = np.asarray(vS_n.nodal_values_component(1), dtype=float)
                        max_speed = float(np.max(np.sqrt(vx * vx + vy * vy)))
                    elif adv_key == "v":
                        vx = np.asarray(v_n.nodal_values_component(0), dtype=float)
                        vy = np.asarray(v_n.nodal_values_component(1), dtype=float)
                        max_speed = float(np.max(np.sqrt(vx * vx + vy * vy)))
                    else:
                        # For mix/mix_biofilm we would need nodal mixture velocity assembly;
                        # keep CFL restriction off for these modes.
                        max_speed = 0.0
                except Exception:
                    max_speed = 0.0
                if max_speed > 1.0e-14:
                    dt_cfl = float(cfl) * float(h_min) / float(max_speed)
                    dt_step = float(min(dt_step, dt_cfl))

            while True:
                if dt_min > 0.0 and dt_step < dt_min - 0.0:
                    raise RuntimeError(f"dt dropped below dt_min={dt_min:.3e} at step {step_no}.")

                # Predictor (k <- n)
                for f, f_prev in zip(functions, prev_functions):
                    f.nodal_values[:] = f_prev.nodal_values[:]

                # Update moving substrate boundary from the current interface (t_n).
                y_top = _biofilm_top_y(dh=dh, alpha=alpha_n, alpha_half=0.5)
                y_D = float(min(float(H) - 1.0e-12, y_top + Ls + 1.0e-10))
                ls_Sd.c = -float(y_D)

                try:
                    # (1) Quasi-steady substrate solve
                    dt_c.value = float(dt_steady)
                    # Duddu (2007) explicitly neglects advection of the substrate by the growth velocity.
                    # Our one-domain substrate equation includes `div(C S v)` by default; when `include-fluid`
                    # is enabled this term uses the previous-step fluid velocity as a lagged advector.
                    # To match Duddu's assumption, we optionally zero-out `v_k` during the substrate solve.
                    v_backup = None
                    if str(args.substrate_advection).strip().lower() == "off":
                        v_backup = np.asarray(v_k.nodal_values, dtype=float).copy()
                        v_k.nodal_values[:] = 0.0
                    solver_S._current_step_no = int(step_no)
                    solver_S._current_t = float(t_n)
                    solver_S._current_dt = float(dt_steady)
                    solver_S._newton_loop(functions, prev_functions, {"dt": dt_c}, bcs_now=[])
                    if substrate_needs_clip:
                        S_k.nodal_values[:] = np.maximum(np.asarray(S_k.nodal_values, dtype=float), 0.0)
                    if v_backup is not None:
                        v_k.nodal_values[:] = v_backup

                    # (2) Solve (p, vS) with S, alpha frozen
                    dt_c.value = float(dt_step)
                    solver_pvS._current_step_no = int(step_no)
                    solver_pvS._current_t = float(t_n)
                    solver_pvS._current_dt = float(dt_step)
                    bcs_now = solver_pvS._freeze_bcs(bcs_pvS, float(t_n))
                    dh.apply_bcs(bcs_now, *functions)
                    solver_pvS._newton_loop(functions, prev_functions, {"dt": dt_c}, bcs_now)

                    # Promote S and pvS fields as frozen coefficients for alpha solve.
                    S_n.nodal_values[:] = S_k.nodal_values[:]
                    p_n.nodal_values[:] = p_k.nodal_values[:]
                    v_n.nodal_values[:] = v_k.nodal_values[:]
                    vS_n.nodal_values[:] = vS_k.nodal_values[:]

                    # (3) Alpha update
                    solver_alpha._current_step_no = int(step_no)
                    solver_alpha._current_t = float(t_n)
                    solver_alpha._current_dt = float(dt_step)
                    solver_alpha._newton_loop(functions, prev_functions, {"dt": dt_c}, bcs_now=[])
                    if alpha_needs_clip:
                        alpha_k.nodal_values[:] = np.clip(np.asarray(alpha_k.nodal_values, dtype=float), 0.0, 1.0)
                    a_cut = float(args.alpha_cleanup_cut)
                    if a_cut > 0.0:
                        if a_cut >= 0.5:
                            raise ValueError("--alpha-cleanup-cut must be < 0.5 when enabled.")
                        a_vals = np.asarray(alpha_k.nodal_values, dtype=float)
                        a_vals[a_vals < a_cut] = 0.0
                        a_vals[a_vals > 1.0 - a_cut] = 1.0
                        alpha_k.nodal_values[:] = a_vals
                    if reinit_every > 0 and (int(step_no) % int(reinit_every) == 0):
                        _reinitialize_alpha_from_signed_distance(
                            dh=dh,
                            alpha=alpha_k,
                            nx=nx,
                            ny=ny,
                            L=L,
                            H=H,
                            eps_alpha=eps_alpha,
                            alpha_half=0.5,
                            refine=reinit_refine,
                        )
                    if sharpen_power > 0.0 and (int(step_no) % int(sharpen_every) == 0):
                        alpha_k.nodal_values[:] = _alpha_sharpen_values(np.asarray(alpha_k.nodal_values, dtype=float), power=sharpen_power)

                    # Update the frozen porosity proxy to remain consistent with alpha.
                    _update_phi_from_alpha(
                        phi=phi_k,
                        alpha=alpha_k,
                        phi_b=phi_b,
                        mode=str(args.phi_update),
                        alpha0=float(args.phi_alpha0),
                        alpha_width=float(args.phi_alpha_width),
                    )

                except Exception as exc:
                    if not bool(args.adaptive_dt):
                        raise
                    dt_step *= float(red)
                    msg = str(exc).strip()
                    tag = f"{type(exc).__name__}: {msg}" if msg else f"{type(exc).__name__}"
                    print(f"    Rejecting step {step_no}; reducing Δt → {dt_step:.3e} ({tag}) and retrying.")
                    continue

                # Accept step: promote k -> n (all fields)
                for f_prev, f in zip(prev_functions, functions):
                    f_prev.nodal_values[:] = f.nodal_values[:]

                t_n += float(dt_step)

                # Time series row + snapshots
                y_top_new = _biofilm_top_y(dh=dh, alpha=alpha_n, alpha_half=0.5)
                a_vals = np.asarray(alpha_n.nodal_values, dtype=float)
                rows.append(
                    {
                        "t_days": float(t_n),
                        "step": int(step_no),
                        "dt_days": float(dt_step),
                        "y_top_mm": float(y_top_new),
                        "y_D_mm": float(y_D),
                        "alpha_min": float(np.min(a_vals)),
                        "alpha_max": float(np.max(a_vals)),
                    }
                )
                if int(step_no) % snap_every == 0 or abs(t_n - t_final) <= 1.0e-12:
                    snaps.append((float(t_n), np.asarray(alpha_n.nodal_values, dtype=float).copy()))
                if flush_snaps_every > 0 and (int(step_no) % flush_snaps_every == 0 or abs(t_n - t_final) <= 1.0e-12):
                    _write_snaps_npz(snaps_npz, snaps)

                # Keep partial results so an interrupt still yields a usable CSV.
                if int(step_no) % write_every == 0 or abs(t_n - t_final) <= 1.0e-12:
                    _write_csv(outdir / "y_top_timeseries.csv", rows)

                if int(step_no) % progress_every == 0 or abs(t_n - t_final) <= 1.0e-12:
                    wall = time.perf_counter() - t0_wall
                    print(f"[progress] step={step_no:04d}  t={t_n:.3f} d  y_top={y_top_new:.4f} mm  (wall={wall:.1f}s)")

                if vtk_full:
                    hit_target = any(abs(t_n - tt) <= 1.0e-10 for tt in targets) or abs(t_n - t_final) <= 1.0e-12
                    periodic = vtk_full_every > 0 and (int(step_no) % int(vtk_full_every) == 0)
                    if hit_target or periodic:
                        _write_full_vtk(float(t_n))
                break
    except KeyboardInterrupt:
        interrupted = True
        print("[interrupt] Caught KeyboardInterrupt; writing partial outputs.")

    # --- outputs -----------------------------------------------------------
    if rows:
        _write_csv(outdir / "y_top_timeseries.csv", rows)
    if flush_snaps_every > 0:
        _write_snaps_npz(snaps_npz, snaps)

    if vtk_full and vtk_datasets:
        _write_pvd(out_pvd=vtk_dir / "series.pvd", datasets=vtk_datasets)
        print(f"[ok] wrote {vtk_dir/'series.pvd'} ({len(vtk_datasets)} dataset(s))")

    # Persist final scalar fields so Fig.6 panels can be regenerated even when
    # the run was executed with --skip-plots (useful for parameter sweeps).
    try:
        np.savez_compressed(
            outdir / "final_fields.npz",
            t_days=float(t_n),
            alpha=np.asarray(alpha_n.nodal_values, dtype=np.float32),
            S=np.asarray(S_n.nodal_values, dtype=np.float32),
            p=np.asarray(p_n.nodal_values, dtype=np.float32),
            vS=np.asarray(vS_n.nodal_values, dtype=np.float32),
        )
    except Exception as exc:
        print(f"[warn] Failed to write final_fields.npz: {type(exc).__name__}: {exc}")

    if not bool(args.skip_plots) and rows:
        tri_alpha = triangulate_field(mesh, dh, "alpha")
        tri_S = triangulate_field(mesh, dh, "S")
        tri_p = triangulate_field(mesh, dh, "p")

        _plot_interface_contours(
            tri=tri_alpha,
            snaps=snaps,
            alpha_half=0.5,
            outpng=outdir / "fig6a_interface.png",
            L=L,
            H=H,
        )
        _plot_scalar_field(
            tri=tri_S,
            values=np.asarray(S_n.nodal_values, dtype=float),
            title="S : substrate concentration (final)",
            outpng=outdir / "fig6b_S.png",
            cmap="gray_r",
            overlay_iface=(tri_alpha, np.asarray(alpha_n.nodal_values, dtype=float), 0.5),
        )
        _plot_scalar_field(
            tri=tri_p,
            values=np.asarray(p_n.nodal_values, dtype=float),
            title="p : potential/pressure surrogate (final)",
            outpng=outdir / "fig6c_Phi.png",
            cmap="gray",
            overlay_iface=(tri_alpha, np.asarray(alpha_n.nodal_values, dtype=float), 0.5),
        )

    summary = {
        "L_mm": L,
        "H_mm": H,
        "nx": nx,
        "ny": ny,
        "q": qdeg,
        "centers_x_mm": centers_x,
        "radii_mm": radii,
        "eps_alpha_mm": eps_alpha,
        "phi_b": phi_b,
        "phi_update": str(args.phi_update),
        "phi_alpha0": float(args.phi_alpha0),
        "phi_alpha_width": float(args.phi_alpha_width),
        "Ls_mm": Ls,
        "Sbar": Sbar,
        "S_penalty": float(args.S_penalty),
        "D_S_mm2_per_day": float(args.D_S),
        "include_fluid": bool(args.include_fluid),
        "substrate_advection": str(args.substrate_advection),
        "vS_top_bc": str(args.vS_top_bc),
        "kappa_inv": float(args.kappa_inv),
        "mu_f": float(args.mu_f),
        "s_v_mode": str(args.s_v_mode),
        "gamma_vS": float(args.gamma_vS),
        "gamma_vS_pin": float(args.gamma_vS_pin),
        "gamma_vS_pin_power": int(args.gamma_vS_pin_power),
        "vS_ext_mode": str(args.vS_ext_mode),
        "gamma_p_out": float(args.gamma_p_out),
        "gamma_p_out_power": int(args.gamma_p_out_power),
        "mu_max_1_per_day": float(mu_max),
        "mu_max_scale": float(args.mu_max_scale),
        "K_S": float(kin.K0),
        "Y": float(kin.Y_xO),
        "rho_s_star": float(kin.rho_x),
        "D_alpha": float(args.D_alpha),
        "ac_M": float(args.ac_M),
        "ac_gamma": float(args.ac_gamma),
        "ac_mobility": str(args.ac_mobility),
        "ac_mobility_floor": float(args.ac_mobility_floor),
        "alpha_advect_with": str(args.alpha_advect_with),
        "alpha_supg": float(args.alpha_supg),
        "alpha_cleanup_cut": float(args.alpha_cleanup_cut),
        "alpha_reinit_every": int(reinit_every),
        "alpha_reinit_refine": int(reinit_refine),
        "alpha_sharpen_power": float(sharpen_power),
        "alpha_sharpen_every": int(sharpen_every),
        "t_final_days": float(t_final),
        "dt_days": float(dt_growth),
        "steps": int(step_no),
        "y_top_final_mm": float(rows[-1]["y_top_mm"]) if rows else None,
        "interrupted": bool(interrupted),
        "targets_days": targets,
        "vtk_full": vtk_full,
        "vtk_full_every": vtk_full_every,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"- Wrote {outdir/'summary.json'}")


if __name__ == "__main__":
    main()
