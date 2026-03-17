#!/usr/bin/env python3
"""Paper 1 Benchmark 7: Seboldt et al. (2021) Example 2.

This driver adapts the reduced deformation-only one-domain model to the
Seboldt geometry:

  - fluid region (0,1) x (0,1),
  - poroelastic region (0,1) x (1,1.5),
  - parabolic inflow at the lower boundary,
  - pinned lateral displacement,
  - drained porous top boundary.

The primary quantitative target is the moving-domain linear profile family from
Figure 6 of the Seboldt paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional in lightweight envs
    matplotlib = None
    plt = None

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

for _k in (
    "NUMBA_DEBUG",
    "NUMBA_DUMP_BYTECODE",
    "NUMBA_DUMP_IR",
    "NUMBA_DUMP_SSA",
    "NUMBA_DEBUG_ARRAY_OPT",
):
    os.environ[_k] = "0"

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.solvers.nonlinear_solver import (
    LinearSolverParameters,
    NewtonParameters,
    NewtonSolver,
    PdasNewtonSolver,
    TimeStepperParameters,
    VIParameters,
)
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    HdivFunction,
    HdivTestFunction,
    HdivTrialFunction,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
)
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import ds, dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad
from pycutfem.utils.functionals import NamedFunctionalEvaluator

from examples.utils.biofilm.deformation_only import build_deformation_only_forms
from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms


@dataclass(frozen=True)
class ProfileMetrics:
    rmse: float
    linf: float
    amplitude: float
    rmse_over_amplitude: float
    linf_over_amplitude: float
    peak_amplitude: float
    peak_amplitude_ref: float
    peak_amplitude_relative_error: float
    peak_x: float
    peak_x_ref: float
    peak_x_error: float


@dataclass(frozen=True)
class CaseResult:
    kappa: float
    outdir: Path
    summary_row: dict[str, object]
    profile_x: np.ndarray
    profile_uy: np.ndarray
    moving_metrics: ProfileMetrics | None
    fixed_metrics: ProfileMetrics | None


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for item in str(raw).split(","):
        text = item.strip()
        if not text:
            continue
        out.append(float(text))
    if not out:
        raise ValueError("Expected at least one numeric value.")
    return out


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=str, default="out/benchmark7_seboldt")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument(
        "--nonlinear-solver",
        type=str,
        default="pdas",
        choices=("pdas", "newton"),
        help="Nonlinear solver for Benchmark 7. PDAS enforces box constraints on alpha/phi.",
    )
    ap.add_argument(
        "--linear-backend",
        type=str,
        default="scipy",
        choices=("scipy", "petsc", "pardiso"),
        help="Linear algebra backend used inside the Newton/PDAS solve.",
    )
    ap.add_argument("--poly-order", type=int, default=2, help="Primary polynomial order for velocity and solid fields.")
    ap.add_argument(
        "--fluid-space",
        type=str,
        default="cg",
        choices=("cg", "hdiv"),
        help="Fluid velocity space. 'hdiv' uses a single RT field for v.",
    )
    ap.add_argument(
        "--fluid-hdiv-order",
        type=int,
        default=0,
        help="RT order used when --fluid-space=hdiv.",
    )
    ap.add_argument("--pressure-order", type=int, default=None, help="Pressure order; defaults to poly_order-1.")
    ap.add_argument("--scalar-order", type=int, default=None, help="Alpha/mu/phi order; defaults to poly_order-1.")
    ap.add_argument("--nx", type=int, default=20, help="Cells in x. h=0.05 corresponds to nx=20.")
    ap.add_argument("--ny", type=int, default=30, help="Cells in y. h=0.05 on Ly=1.5 corresponds to ny=30.")
    ap.add_argument("--dt", type=float, default=1.0e-3)
    ap.add_argument("--t-final", type=float, default=3.0)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument(
        "--solid-model",
        type=str,
        default="linear",
        choices=("linear", "neo_hookean", "neo-hookean", "nh"),
        help="Skeleton constitutive model on the Eulerian reference-map formulation.",
    )
    ap.add_argument(
        "--enable-phi-evolution",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Switch Benchmark 7 from the reduced fixed-phi_b model to the full one-domain phi-transport model.",
    )
    ap.add_argument("--kappa-list", type=str, default="1e-3,1e-4,1e-5")
    ap.add_argument("--rho-f", type=float, default=1.0)
    ap.add_argument("--mu-f", type=float, default=0.035)
    ap.add_argument("--mu-b", type=float, default=0.035)
    ap.add_argument(
        "--mu-b-model",
        type=str,
        default="mu",
        choices=("mu", "phi_mu", "alpha_mu", "alpha_phi_mu"),
        help="Porous-region effective viscosity model inside the one-domain momentum block.",
    )
    ap.add_argument("--phi-b", type=float, default=0.5)
    ap.add_argument("--mu-s", type=float, default=1.67785e5)
    ap.add_argument("--lambda-s", type=float, default=8.22148e6)
    ap.add_argument("--solid-visco-eta", type=float, default=0.0)
    ap.add_argument("--gamma-div", type=float, default=0.0)
    ap.add_argument("--gamma-u", type=float, default=0.0, help="Reference-map extension penalty in the free-fluid region.")
    ap.add_argument(
        "--u-extension",
        type=str,
        default="l2",
        choices=("l2", "mass", "grad", "h1"),
        help="Extension mode for u outside the porous body.",
    )
    ap.add_argument("--gamma-u-pin", type=float, default=0.0, help="Optional pin for grad-mode u extension.")
    ap.add_argument("--u-cip", type=float, default=0.0, help="Interior-facet CIP stabilization for u transport.")
    ap.add_argument(
        "--u-cip-weight",
        type=str,
        default="fluid",
        choices=("fluid", "biofilm", "both"),
        help="Region weight for u-CIP stabilization.",
    )
    ap.add_argument("--vS-cip", type=float, default=0.0, help="Interior-facet CIP stabilization for vS.")
    ap.add_argument("--gamma-vS", type=float, default=None, help="Optional vS extension penalty; defaults to gamma_u.")
    ap.add_argument(
        "--vS-ext-mode",
        type=str,
        default=None,
        choices=("l2", "mass", "grad", "h1"),
        help="Optional vS extension mode; defaults to u_extension.",
    )
    ap.add_argument("--gamma-vS-pin", type=float, default=None, help="Optional pin for grad-mode vS extension.")
    ap.add_argument("--D-phi", type=float, default=0.0, help="Optional porosity diffusion coefficient used only for regularization tests.")
    ap.add_argument(
        "--phi-diffusion-weight",
        type=str,
        default="fluid",
        choices=("unity", "fluid", "biofilm"),
        help="Weight for the optional D_phi regularization term in the porosity equation.",
    )
    ap.add_argument("--gamma-phi", type=float, default=5.0, help="Penalty enforcing phi->1 in the free-fluid region.")
    ap.add_argument("--phi-supg", type=float, default=0.0, help="SUPG stabilization for phi advection in full-model mode.")
    ap.add_argument("--phi-cip", type=float, default=0.0, help="CIP stabilization for phi advection in full-model mode.")
    ap.add_argument("--M-alpha", type=float, default=1.0)
    ap.add_argument("--gamma-alpha", type=float, default=1.0)
    ap.add_argument(
        "--alpha-regularization",
        type=str,
        default="ch",
        choices=("none", "ch", "olsson_nt"),
        help="Geometric/interface regularization model for alpha; use 'none' for pure transport.",
    )
    ap.add_argument(
        "--alpha-reg-gamma",
        type=float,
        default=1.0,
        help="Strength of the conservative alpha interface-maintenance flux when alpha-regularization=olsson_nt.",
    )
    ap.add_argument(
        "--alpha-reg-eps-normal",
        type=float,
        default=None,
        help="Normal smoothing scale for alpha-regularization=olsson_nt; defaults to eps_alpha.",
    )
    ap.add_argument(
        "--alpha-reg-eps-tangent",
        type=float,
        default=None,
        help="Tangential smoothing scale for alpha-regularization=olsson_nt; defaults to 0.25*eps_alpha.",
    )
    ap.add_argument(
        "--alpha-reg-eta",
        type=float,
        default=1.0e-12,
        help="Small positive floor for the lagged alpha-normal in alpha-regularization=olsson_nt.",
    )
    ap.add_argument(
        "--alpha-advect-with",
        type=str,
        default="vS",
        choices=("vS", "v", "relative", "mix", "interface", "mix_biofilm"),
        help="Velocity used to advect the diffuse biofilm indicator alpha.",
    )
    ap.add_argument(
        "--alpha-advection-form",
        type=str,
        default="conservative",
        choices=("advective", "conservative", "conservative_weak", "interface_band_conservative"),
        help="Alpha transport form: advective strong form, conservative strong form, conservative weak/IBP form, or interface-band conservative transport.",
    )
    ap.add_argument("--eps-alpha", type=float, default=0.05)
    ap.add_argument(
        "--eps-alpha-over-h",
        type=float,
        default=None,
        help="If set, override eps_alpha with eps_alpha_over_h * h_char, h_char=max(Lx/nx,Ly/ny).",
    )
    ap.add_argument(
        "--kappa-inv-model",
        type=str,
        default="spatial",
        choices=("spatial", "constant", "const", "refmap", "reference-map", "reference_map", "eulerian", "eulerian_refmap"),
        help="Inverse-permeability frame/model. Use refmap/reference-map for the Eulerian push-forward tensor.",
    )
    ap.add_argument("--v-in", type=float, default=5.0)
    ap.add_argument("--t-ramp", type=float, default=0.0, help="Optional cosine ramp time for the inlet profile [s].")
    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--Ly", type=float, default=1.5)
    ap.add_argument("--y-interface", type=float, default=1.0)
    ap.add_argument(
        "--reg-rect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Localize u/vS extension and CIP regularization to a fixed rectangle centered at the interface.",
    )
    ap.add_argument("--reg-rect-center-x", type=float, default=None, help="Override x-center for the regularization rectangle.")
    ap.add_argument("--reg-rect-center-y", type=float, default=None, help="Override y-center for the regularization rectangle.")
    ap.add_argument("--reg-rect-half-width", type=float, default=None, help="Half-width of the regularization rectangle.")
    ap.add_argument("--reg-rect-half-height", type=float, default=None, help="Half-height of the regularization rectangle.")
    ap.add_argument("--y-profile", type=float, default=1.25)
    ap.add_argument("--profile-samples", type=int, default=201)
    ap.add_argument("--vtk-every", type=int, default=0)
    ap.add_argument("--newton-tol", type=float, default=1.0e-8)
    ap.add_argument("--newton-rtol", type=float, default=1.0e-8)
    ap.add_argument("--max-it", type=int, default=12)
    ap.add_argument(
        "--alpha-box-constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce 0<=alpha<=1 when the PDAS solver is used.",
    )
    ap.add_argument(
        "--phi-box-constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce 0<=phi<=1 when phi evolution is enabled and the PDAS solver is used.",
    )
    ap.add_argument(
        "--vi-c",
        type=float,
        default=1.0e4,
        help="PDAS complementarity scaling parameter for bounded fields; set 0 to re-enable auto-scaling.",
    )
    ap.add_argument("--vi-enter-tol", type=float, default=0.0, help="PDAS active-set entry hysteresis.")
    ap.add_argument("--vi-leave-tol", type=float, default=0.0, help="PDAS active-set release hysteresis.")
    ap.add_argument("--vi-persistence", type=int, default=0, help="PDAS active-set persistence.")
    ap.add_argument("--vi-lambda0", type=float, default=1.0e-4, help="Initial PDAS inactive-block regularization.")
    ap.add_argument("--vi-lambda-max", type=float, default=1.0e6, help="Maximum PDAS inactive-block regularization.")
    ap.add_argument("--vi-lambda-growth", type=float, default=5.0, help="PDAS inactive-block regularization growth.")
    ap.add_argument("--vi-lambda-decay", type=float, default=0.5, help="PDAS inactive-block regularization decay.")
    ap.add_argument(
        "--vi-active-soft-threshold",
        type=int,
        default=0,
        help="Trigger soft damping of active-set corrections when DeltaA exceeds this threshold.",
    )
    ap.add_argument(
        "--vi-active-soft-alpha",
        type=float,
        default=1.0,
        help="Soft damping factor for marginal active-set corrections.",
    )
    ap.add_argument(
        "--vi-active-strong-factor",
        type=float,
        default=5.0,
        help="Multiplier used to detect strongly active PDAS DOFs.",
    )
    ap.add_argument(
        "--vi-filter-max-delta-active",
        type=int,
        default=0,
        help="Optional PDAS line-search filter on active-set changes.",
    )
    ap.add_argument(
        "--vi-filter-max-residual-growth",
        type=float,
        default=1.25,
        help="PDAS line-search filter on inactive-block residual growth.",
    )
    ap.add_argument(
        "--vi-filter-max-gap-growth",
        type=float,
        default=1.25,
        help="PDAS line-search filter on active-set gap growth.",
    )
    ap.add_argument(
        "--vi-unconstrained-lm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable semismooth LM globalization on the PDAS-modified system.",
    )
    ap.add_argument("--vi-lm-lambda0", type=float, default=1.0e-4, help="Initial semismooth LM damping.")
    ap.add_argument("--vi-lm-lambda-max", type=float, default=1.0e6, help="Maximum semismooth LM damping.")
    ap.add_argument("--vi-lm-growth", type=float, default=5.0, help="Semismooth LM damping growth factor.")
    ap.add_argument("--vi-lm-decay", type=float, default=0.5, help="Semismooth LM damping decay factor.")
    ap.add_argument("--vi-lm-accept-ratio", type=float, default=1.0e-3, help="Semismooth LM accept threshold.")
    ap.add_argument("--vi-lm-good-ratio", type=float, default=5.0e-2, help="Semismooth LM good-step threshold.")
    ap.add_argument("--vi-lm-max-tries", type=int, default=6, help="Maximum semismooth LM trial solves per Newton step.")
    ap.add_argument(
        "--ls-mode",
        type=str,
        default="dealii",
        choices=("armijo", "dealii"),
        help="Newton line-search mode; dealii only requires residual decrease and is often cheaper/less strict.",
    )
    ap.add_argument(
        "--line-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Newton line search.",
    )
    ap.add_argument(
        "--predictor",
        type=str,
        default="prev",
        choices=("prev", "delta"),
        help="Initial guess for each new time step.",
    )
    ap.add_argument(
        "--predictor-damping",
        type=float,
        default=1.0,
        help="Damping applied to the delta predictor.",
    )
    ap.add_argument("--lin-tol", type=float, default=1.0e-10)
    ap.add_argument("--lin-maxit", type=int, default=20000)
    ap.add_argument("--dt-min", type=float, default=1.0e-4)
    ap.add_argument(
        "--dt-max",
        type=float,
        default=None,
        help="Optional cap for adaptive dt growth; defaults to the initial dt.",
    )
    ap.add_argument("--dt-reduction-factor", type=float, default=0.5)
    ap.add_argument(
        "--dt-increase-factor",
        type=float,
        default=1.1,
        help="Factor for adaptive dt growth after easy Newton steps; set to 1.0 to freeze dt growth.",
    )
    ap.add_argument(
        "--dt-iters-increase-threshold",
        type=int,
        default=8,
        help="Newton-iteration threshold for classifying an accepted step as easy.",
    )
    ap.add_argument(
        "--dt-easy-steps-before-increase",
        type=int,
        default=2,
        help="Number of consecutive easy steps required before increasing dt.",
    )
    ap.add_argument("--dt-iters-decrease-threshold", type=int, default=8)
    ap.add_argument("--dt-decrease-factor-slow", type=float, default=0.75)
    ap.add_argument("--dt-slow-steps-before-decrease", type=int, default=1)
    ap.add_argument("--no-dt-reduction", action="store_true", help="Disable adaptive dt reduction.")
    ap.add_argument(
        "--reference-csv",
        type=str,
        default=str(here / "reference_profiles_fig6.csv"),
        help="Figure 6 reference CSV produced by extract_seboldt_fig6_reference.py",
    )
    return ap.parse_args()


def _characteristic_h(*, Lx: float, Ly: float, nx: int, ny: int) -> float:
    return max(float(Lx) / float(nx), float(Ly) / float(ny))


def _resolved_orders(args: argparse.Namespace) -> tuple[int, int, int]:
    poly_order = int(args.poly_order)
    if poly_order < 2:
        raise ValueError("Benchmark 7 requires poly_order >= 2.")
    pressure_order = int(args.pressure_order) if args.pressure_order is not None else max(1, poly_order - 1)
    scalar_order = int(args.scalar_order) if args.scalar_order is not None else max(1, poly_order - 1)
    if pressure_order < 1 or scalar_order < 1:
        raise ValueError("pressure_order and scalar_order must be positive.")
    if pressure_order > poly_order:
        raise ValueError("pressure_order must not exceed poly_order.")
    return poly_order, pressure_order, scalar_order


def _effective_eps_alpha(args: argparse.Namespace) -> float:
    if args.eps_alpha_over_h is None:
        return float(args.eps_alpha)
    h_char = _characteristic_h(Lx=float(args.Lx), Ly=float(args.Ly), nx=int(args.nx), ny=int(args.ny))
    return float(args.eps_alpha_over_h) * h_char


def _tag_rectangle_boundaries(mesh: Mesh, *, Lx: float, Ly: float, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - float(Lx)) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - float(Ly)) <= tol,
        }
    )


def _as_float_time(fn):
    return lambda x, y, t: float(np.asarray(fn(np.asarray(x), np.asarray(y), float(t))).reshape(()))


def _alpha_equilibrium(y: np.ndarray, *, y_interface: float, eps_alpha: float) -> np.ndarray:
    yy = np.asarray(y, dtype=float)
    eps = max(float(eps_alpha), 1.0e-12)
    return 0.5 * (1.0 + np.tanh((yy - float(y_interface)) / (math.sqrt(2.0) * eps)))


def _bottom_inlet(x: np.ndarray, y: np.ndarray, t: float, *, v_in: float) -> np.ndarray:
    xx = np.asarray(x, dtype=float)
    return 4.0 * float(v_in) * xx * (1.0 - xx)


def _cosine_ramp_value(t_now: float, ramp_time: float) -> float:
    tr = float(ramp_time)
    if not np.isfinite(tr) or tr <= 0.0:
        return 1.0
    tt = max(0.0, float(t_now))
    if tt >= tr:
        return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * tt / max(1.0e-12, tr)))


def _eval_scalar_at_point(dh: DofHandler, mesh: Mesh, f_scalar: Function, point: tuple[float, float]) -> float:
    from pycutfem.fem import transform

    xy = np.asarray(point, dtype=float)
    for elem in mesh.elements_list:
        verts = mesh.nodes_x_y_pos[list(elem.nodes)]
        if not (
            verts[:, 0].min() - 1.0e-12 <= xy[0] <= verts[:, 0].max() + 1.0e-12
            and verts[:, 1].min() - 1.0e-12 <= xy[1] <= verts[:, 1].max() + 1.0e-12
        ):
            continue
        try:
            xi, eta = transform.inverse_mapping(mesh, elem.id, xy)
        except Exception:
            continue
        if not (-1.0001 <= xi <= 1.0001 and -1.0001 <= eta <= 1.0001):
            continue
        me = dh.mixed_element
        fld = f_scalar.field_name
        phi = me.basis(fld, float(xi), float(eta))[me.slice(fld)]
        gdofs = dh.element_maps[fld][elem.id]
        vals = f_scalar.get_nodal_values(gdofs)
        return float(phi @ vals)
    return float("nan")


def _eval_vector_at_point(dh: DofHandler, mesh: Mesh, f_vec: VectorFunction, point: tuple[float, float]) -> np.ndarray:
    return np.asarray([_eval_scalar_at_point(dh, mesh, comp, point) for comp in f_vec.components], dtype=float)


def _vector_component_values(f_vec: VectorFunction, idx: int) -> np.ndarray:
    arr = np.asarray(getattr(f_vec, "nodal_values", np.asarray([])), dtype=float)
    if arr.ndim == 2 and arr.shape[1] > int(idx):
        return np.asarray(arr[:, int(idx)], dtype=float)
    comps = getattr(f_vec, "components", None)
    if comps is None or len(comps) <= int(idx):
        raise IndexError(f"VectorFunction does not expose component {idx}.")
    return np.asarray(comps[int(idx)].nodal_values, dtype=float)


def _configure_regularization_mask(
    *,
    problem: dict[str, object],
    enabled: bool,
    Lx: float,
    Ly: float,
    y_interface: float,
    center_x: float | None,
    center_y: float | None,
    half_width: float | None,
    half_height: float | None,
) -> dict[str, float]:
    meta = {
        "enabled": float(1.0 if bool(enabled) else 0.0),
        "center_x": float("nan"),
        "center_y": float("nan"),
        "half_width": float("nan"),
        "half_height": float("nan"),
        "fraction": float("nan"),
    }
    problem["reg_weight"] = None
    if not bool(enabled):
        return meta

    cx = 0.5 * float(Lx) if center_x is None else float(center_x)
    cy = float(y_interface) if center_y is None else float(center_y)
    hw = 0.5 * float(Lx) if half_width is None else float(half_width)
    hh = 0.5 * max(float(Ly) - float(y_interface), 1.0e-12) if half_height is None else float(half_height)
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(hw) and math.isfinite(hh)):
        raise ValueError("Regularization rectangle parameters must be finite.")
    if hw <= 0.0 or hh <= 0.0:
        raise ValueError("Regularization rectangle half-width and half-height must be positive.")

    coords = np.asarray(problem["dh"].get_dof_coords("alpha"), dtype=float)
    inside = (
        (coords[:, 0] >= (cx - hw))
        & (coords[:, 0] <= (cx + hw))
        & (coords[:, 1] >= (cy - hh))
        & (coords[:, 1] <= (cy + hh))
    )
    if not np.any(inside):
        raise RuntimeError("Regularization rectangle keeps zero alpha DOFs.")

    reg_weight = Function("reg_weight", "alpha", dof_handler=problem["dh"])
    reg_weight.nodal_values[:] = 0.0
    reg_weight.nodal_values[np.asarray(inside, dtype=bool)] = 1.0
    problem["reg_weight"] = reg_weight
    meta.update(
        {
            "center_x": float(cx),
            "center_y": float(cy),
            "half_width": float(hw),
            "half_height": float(hh),
            "fraction": float(np.mean(np.asarray(inside, dtype=float))),
        }
    )
    return meta


def _create_problem(
    *,
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    poly_order: int,
    pressure_order: int,
    scalar_order: int,
    fluid_space: str = "cg",
    fluid_hdiv_order: int = 0,
    enable_phi_evolution: bool,
) -> dict[str, object]:
    fluid_space_key = str(fluid_space).strip().lower()
    if fluid_space_key not in {"cg", "hdiv"}:
        raise ValueError(f"Unsupported fluid_space={fluid_space!r}.")

    nodes, elems, _, corners = structured_quad(float(Lx), float(Ly), nx=int(nx), ny=int(ny), poly_order=int(poly_order))
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=int(poly_order),
    )
    _tag_rectangle_boundaries(mesh, Lx=float(Lx), Ly=float(Ly))

    field_specs = {
        "p": int(pressure_order),
        "vS_x": int(poly_order),
        "vS_y": int(poly_order),
        "u_x": int(poly_order),
        "u_y": int(poly_order),
        "alpha": int(scalar_order),
        "mu_alpha": int(scalar_order),
    }
    if fluid_space_key == "cg":
        field_specs = {"v_x": int(poly_order), "v_y": int(poly_order), **field_specs}
    else:
        field_specs = {"v": ("RT", int(fluid_hdiv_order)), **field_specs}
    if bool(enable_phi_evolution):
        field_specs["phi"] = int(scalar_order)
        field_specs["S"] = int(scalar_order)
    me = MixedElement(mesh, field_specs=field_specs)
    dh = DofHandler(me, method="cg")

    VS = FunctionSpace("VS", ["vS_x", "vS_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    if fluid_space_key == "cg":
        V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
        dv = VectorTrialFunction(space=V, dof_handler=dh)
        v_test = VectorTestFunction(space=V, dof_handler=dh)
        v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
        v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    else:
        dv = HdivTrialFunction("v")
        v_test = HdivTestFunction("v")
        v_k = HdivFunction("v_k", "v", dof_handler=dh)
        v_n = HdivFunction("v_n", "v", dof_handler=dh)

    problem: dict[str, object] = {
        "mesh": mesh,
        "me": me,
        "dh": dh,
        "fluid_space": fluid_space_key,
        "fluid_hdiv_order": int(fluid_hdiv_order),
        "dv": dv,
        "dvS": VectorTrialFunction(space=VS, dof_handler=dh),
        "du": VectorTrialFunction(space=U, dof_handler=dh),
        "dp": TrialFunction("p", dof_handler=dh),
        "dalpha": TrialFunction("alpha", dof_handler=dh),
        "dmu": TrialFunction("mu_alpha", dof_handler=dh),
        "v_test": v_test,
        "vS_test": VectorTestFunction(space=VS, dof_handler=dh),
        "u_test": VectorTestFunction(space=U, dof_handler=dh),
        "q_test": TestFunction("p", dof_handler=dh),
        "alpha_test": TestFunction("alpha", dof_handler=dh),
        "mu_test": TestFunction("mu_alpha", dof_handler=dh),
        "v_k": v_k,
        "p_k": Function("p_k", "p", dof_handler=dh),
        "vS_k": VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh),
        "u_k": VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh),
        "alpha_k": Function("alpha_k", "alpha", dof_handler=dh),
        "mu_k": Function("mu_k", "mu_alpha", dof_handler=dh),
        "v_n": v_n,
        "p_n": Function("p_n", "p", dof_handler=dh),
        "vS_n": VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh),
        "u_n": VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh),
        "alpha_n": Function("alpha_n", "alpha", dof_handler=dh),
        "mu_n": Function("mu_n", "mu_alpha", dof_handler=dh),
    }
    if bool(enable_phi_evolution):
        problem.update(
            {
                "dphi": TrialFunction("phi", dof_handler=dh),
                "dS": TrialFunction("S", dof_handler=dh),
                "phi_test": TestFunction("phi", dof_handler=dh),
                "S_test": TestFunction("S", dof_handler=dh),
                "phi_k": Function("phi_k", "phi", dof_handler=dh),
                "S_k": Function("S_k", "S", dof_handler=dh),
                "phi_n": Function("phi_n", "phi", dof_handler=dh),
                "S_n": Function("S_n", "S", dof_handler=dh),
                "reg_weight": None,
            }
        )
    else:
        problem.update(
            {
                "dphi": None,
                "dS": None,
                "phi_test": None,
                "S_test": None,
                "phi_k": None,
                "S_k": None,
                "phi_n": None,
                "S_n": None,
                "reg_weight": None,
            }
        )
    for key in ("v_k", "vS_k", "u_k", "v_n", "vS_n", "u_n"):
        problem[key].nodal_values[:] = 0.0
    for key in ("p_k", "p_n", "mu_k", "mu_n", "phi_k", "phi_n", "S_k", "S_n"):
        if problem.get(key) is not None:
            problem[key].nodal_values[:] = 0.0
    return problem


def _build_forms(
    problem: dict[str, object],
    *,
    qdeg: int,
    dt_c,
    theta: float,
    rho_f: float,
    mu_f: float,
    mu_b: float,
    mu_b_model: str,
    kappa_inv: float,
    mu_s: float,
    lambda_s: float,
    phi_b: float,
    M_alpha: float,
    gamma_alpha: float,
    eps_alpha: float,
    solid_visco_eta: float,
    gamma_div: float,
    gamma_u: float,
    u_extension_mode: str,
    gamma_u_pin: float,
    u_cip: float,
    u_cip_weight: str,
    vS_cip: float,
    gamma_vS: float | None,
    vS_extension_mode: str | None,
    gamma_vS_pin: float | None,
    D_phi: float,
    phi_diffusion_weight: str,
    gamma_phi: float,
    phi_supg: float,
    phi_cip: float,
    alpha_regularization: str,
    alpha_reg_gamma: float,
    alpha_reg_eps_normal: float,
    alpha_reg_eps_tangent: float,
    alpha_reg_eta: float,
    alpha_advect_with: str,
    alpha_advection_form: str,
    solid_model: str,
    kappa_inv_model: str,
    enable_phi_evolution: bool,
):
    solid_model_key = str(solid_model).strip().lower().replace("-", "_")
    common_kwargs = {
        "dt": dt_c,
        "theta": float(theta),
        "rho_f": Constant(float(rho_f)),
        "mu_f": Constant(float(mu_f)),
        "mu_b": Constant(float(mu_b)),
        "kappa_inv": Constant(float(kappa_inv)),
        "mu_s": Constant(float(mu_s)),
        "lambda_s": Constant(float(lambda_s)),
        "solid_visco_eta": float(solid_visco_eta),
        "gamma_div": float(gamma_div),
        "solid_model": solid_model_key,
        "kappa_inv_model": str(kappa_inv_model),
    }
    one_domain_kwargs = {
        **common_kwargs,
        "gamma_u": float(gamma_u),
        "u_extension_mode": str(u_extension_mode),
        "gamma_u_pin": float(gamma_u_pin),
    }
    if not bool(enable_phi_evolution):
        return build_deformation_only_forms(
            v_k=problem["v_k"],
            p_k=problem["p_k"],
            vS_k=problem["vS_k"],
            u_k=problem["u_k"],
            alpha_k=problem["alpha_k"],
            mu_alpha_k=problem["mu_k"],
            v_n=problem["v_n"],
            p_n=problem["p_n"],
            vS_n=problem["vS_n"],
            u_n=problem["u_n"],
            alpha_n=problem["alpha_n"],
            mu_alpha_n=problem["mu_n"],
            dv=problem["dv"],
            dp=problem["dp"],
            dvS=problem["dvS"],
            du=problem["du"],
            dalpha=problem["dalpha"],
            dmu_alpha=problem["dmu"],
            v_test=problem["v_test"],
            q_test=problem["q_test"],
            vS_test=problem["vS_test"],
            u_test=problem["u_test"],
            alpha_test=problem["alpha_test"],
            mu_alpha_test=problem["mu_test"],
            dx=dx(metadata={"q": int(qdeg)}),
            phi_b=float(phi_b),
            M_alpha=float(M_alpha),
            gamma_alpha=float(gamma_alpha),
            eps_alpha=float(eps_alpha),
            **common_kwargs,
        )
    return build_biofilm_one_domain_forms(
        v_k=problem["v_k"],
        p_k=problem["p_k"],
        vS_k=problem["vS_k"],
        u_k=problem["u_k"],
        phi_k=problem["phi_k"],
        alpha_k=problem["alpha_k"],
        mu_alpha_k=problem["mu_k"],
        S_k=problem["S_k"],
        v_n=problem["v_n"],
        p_n=problem["p_n"],
        vS_n=problem["vS_n"],
        u_n=problem["u_n"],
        phi_n=problem["phi_n"],
        alpha_n=problem["alpha_n"],
        mu_alpha_n=problem["mu_n"],
        S_n=problem["S_n"],
        dv=problem["dv"],
        dp=problem["dp"],
        dvS=problem["dvS"],
        du=problem["du"],
        dphi=problem["dphi"],
        dalpha=problem["dalpha"],
        dmu_alpha=problem["dmu"],
        dS=problem["dS"],
        v_test=problem["v_test"],
        q_test=problem["q_test"],
        vS_test=problem["vS_test"],
        u_test=problem["u_test"],
        phi_test=problem["phi_test"],
        alpha_test=problem["alpha_test"],
        mu_alpha_test=problem["mu_test"],
        S_test=problem["S_test"],
        dx=dx(metadata={"q": int(qdeg)}),
        ds_cip=ds(metadata={"q": int(qdeg)}),
        mu_b_model=str(mu_b_model),
        D_phi=float(D_phi),
        phi_diffusion_weight=str(phi_diffusion_weight),
        gamma_phi=float(gamma_phi),
        phi_supg=float(phi_supg),
        phi_cip=float(phi_cip),
        regularization_weight=problem.get("reg_weight"),
        u_cip=float(u_cip),
        u_cip_weight=str(u_cip_weight),
        vS_cip=float(vS_cip),
        gamma_vS=gamma_vS,
        vS_extension_mode=vS_extension_mode,
        gamma_vS_pin=gamma_vS_pin,
        D_alpha=0.0,
        alpha_interface_reg=(
            "olsson_nt"
            if str(alpha_regularization).strip().lower() == "olsson_nt"
            else "none"
        ),
        alpha_interface_reg_gamma=float(alpha_reg_gamma),
        alpha_interface_reg_eps_normal=float(alpha_reg_eps_normal),
        alpha_interface_reg_eps_tangent=float(alpha_reg_eps_tangent),
        alpha_interface_reg_eta=float(alpha_reg_eta),
        alpha_mu_aux_pin=1.0,
        alpha_advect_with=str(alpha_advect_with),
        alpha_advection_form=str(alpha_advection_form),
        alpha_ch_M=float(M_alpha if str(alpha_regularization).strip().lower() == "ch" else 0.0),
        alpha_ch_gamma=float(gamma_alpha if str(alpha_regularization).strip().lower() == "ch" else 0.0),
        alpha_ch_eps=float(eps_alpha),
        D_S=0.0,
        mu_max=0.0,
        K_S=1.0,
        k_g=0.0,
        k_d=0.0,
        Y=1.0,
        rho_s_star=1.0,
        k_det=0.0,
        **one_domain_kwargs,
    )


def _build_bcs(
    *,
    y_interface: float,
    eps_alpha: float,
    v_in: float,
    t_ramp: float,
) -> list[BoundaryCondition]:
    alpha_bc = lambda x, y, t: float(_alpha_equilibrium(y, y_interface=float(y_interface), eps_alpha=float(eps_alpha)).reshape(()))
    zero = lambda x, y, t: 0.0
    inflow_y = lambda x, y, t: float(
        _cosine_ramp_value(float(t), float(t_ramp)) * _bottom_inlet(x, y, t, v_in=float(v_in)).reshape(())
    )

    bcs: list[BoundaryCondition] = []
    for tag in ("left", "right"):
        bcs.extend(
            [
                BoundaryCondition("v_x", "dirichlet", tag, _as_float_time(zero)),
                BoundaryCondition("v_y", "dirichlet", tag, _as_float_time(zero)),
                BoundaryCondition("vS_x", "dirichlet", tag, _as_float_time(zero)),
                BoundaryCondition("vS_y", "dirichlet", tag, _as_float_time(zero)),
                BoundaryCondition("u_x", "dirichlet", tag, _as_float_time(zero)),
                BoundaryCondition("u_y", "dirichlet", tag, _as_float_time(zero)),
                BoundaryCondition("alpha", "dirichlet", tag, _as_float_time(alpha_bc)),
                BoundaryCondition("mu_alpha", "dirichlet", tag, _as_float_time(zero)),
            ]
        )

    bcs.extend(
        [
            BoundaryCondition("v_x", "dirichlet", "bottom", _as_float_time(zero)),
            BoundaryCondition("v_y", "dirichlet", "bottom", _as_float_time(inflow_y)),
            BoundaryCondition("vS_x", "dirichlet", "bottom", _as_float_time(zero)),
            BoundaryCondition("vS_y", "dirichlet", "bottom", _as_float_time(zero)),
            BoundaryCondition("u_x", "dirichlet", "bottom", _as_float_time(zero)),
            BoundaryCondition("u_y", "dirichlet", "bottom", _as_float_time(zero)),
            BoundaryCondition("alpha", "dirichlet", "bottom", _as_float_time(alpha_bc)),
            BoundaryCondition("mu_alpha", "dirichlet", "bottom", _as_float_time(zero)),
        ]
    )
    bcs.extend(
        [
            BoundaryCondition("p", "dirichlet", "top", _as_float_time(zero)),
            BoundaryCondition("alpha", "dirichlet", "top", _as_float_time(alpha_bc)),
            BoundaryCondition("mu_alpha", "dirichlet", "top", _as_float_time(zero)),
        ]
    )
    return bcs


def _sample_profile(
    *,
    problem: dict[str, object],
    Lx: float,
    y_profile: float,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, float(Lx), int(n_samples), dtype=float)
    u_y = np.asarray(
        [
            _eval_vector_at_point(problem["dh"], problem["mesh"], problem["u_k"], (float(xx), float(y_profile)))[1]
            for xx in x
        ],
        dtype=float,
    )
    return x, u_y


def _write_timeseries_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_profile_csv(path: Path, *, x: np.ndarray, u_y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "u_y"])
        for x_i, u_i in zip(np.asarray(x, dtype=float), np.asarray(u_y, dtype=float)):
            writer.writerow([f"{float(x_i):.16e}", f"{float(u_i):.16e}"])


def _build_alpha_diagnostics(problem: dict[str, object], *, quad_order: int, backend: str) -> NamedFunctionalEvaluator:
    alpha_k = problem["alpha_k"]
    one = Constant(1.0)
    forms = {
        "alpha_area": alpha_k * dx(metadata={"q": int(quad_order)}),
        "alpha_band": (Constant(4.0) * alpha_k * (one - alpha_k)) * dx(metadata={"q": int(quad_order)}),
    }
    return NamedFunctionalEvaluator(
        forms=forms,
        dof_handler=problem["dh"],
        mixed_element=problem["me"],
        backend=str(backend),
        quad_order=int(quad_order),
    )


def _load_reference_curve(
    *,
    reference_csv: Path,
    kappa: float,
    curve_label: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not reference_csv.exists():
        return None
    rows = []
    with reference_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("curve_label", "")).strip() != str(curve_label):
                continue
            try:
                kappa_i = float(row.get("kappa", "nan"))
            except Exception:
                continue
            if not math.isfinite(kappa_i) or abs(kappa_i - float(kappa)) > 1.0e-12 * max(1.0, abs(float(kappa))):
                continue
            rows.append((float(row["x"]), float(row["eta_y"])))
    if not rows:
        return None
    arr = np.asarray(sorted(rows, key=lambda pair: pair[0]), dtype=float)
    return arr[:, 0], arr[:, 1]


def _compute_profile_metrics(
    *,
    x_num: np.ndarray,
    y_num: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
) -> ProfileMetrics:
    x_common = np.asarray(x_num, dtype=float)
    y_num_i = np.asarray(y_num, dtype=float)
    y_ref_i = np.interp(x_common, np.asarray(x_ref, dtype=float), np.asarray(y_ref, dtype=float))
    diff = y_num_i - y_ref_i
    rmse = float(np.sqrt(np.mean(diff * diff)))
    linf = float(np.max(np.abs(diff)))
    amplitude = float(max(np.max(y_ref_i) - np.min(y_ref_i), 1.0e-14))
    peak_idx = int(np.argmax(y_num_i))
    peak_ref_idx = int(np.argmax(y_ref_i))
    peak_amp = float(np.max(y_num_i))
    peak_amp_ref = float(np.max(y_ref_i))
    peak_rel = abs(peak_amp - peak_amp_ref) / max(abs(peak_amp_ref), 1.0e-14)
    peak_x = float(x_common[peak_idx])
    peak_x_ref = float(x_common[peak_ref_idx])
    return ProfileMetrics(
        rmse=rmse,
        linf=linf,
        amplitude=amplitude,
        rmse_over_amplitude=rmse / amplitude,
        linf_over_amplitude=linf / amplitude,
        peak_amplitude=peak_amp,
        peak_amplitude_ref=peak_amp_ref,
        peak_amplitude_relative_error=peak_rel,
        peak_x=peak_x,
        peak_x_ref=peak_x_ref,
        peak_x_error=abs(peak_x - peak_x_ref),
    )


def _write_case_plot(
    path: Path,
    *,
    kappa: float,
    x_num: np.ndarray,
    y_num: np.ndarray,
    moving_ref: tuple[np.ndarray, np.ndarray] | None,
    fixed_ref: tuple[np.ndarray, np.ndarray] | None,
    nonlinear_ref: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.5), constrained_layout=True)
    ax.plot(x_num, y_num, color="tab:green", lw=2.2, label="one-domain")
    if moving_ref is not None:
        ax.plot(moving_ref[0], moving_ref[1], color="#149dff", lw=2.0, label="Seboldt moving linear")
    if fixed_ref is not None:
        ax.plot(fixed_ref[0], fixed_ref[1], color="red", lw=1.8, ls="--", label="Seboldt fixed linear")
    if nonlinear_ref is not None:
        ax.plot(nonlinear_ref[0], nonlinear_ref[1], color="#b748ff", lw=1.8, label="Seboldt moving nonlinear")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$u_y(x, y=1.25)$")
    ax.set_title(rf"$\kappa={float(kappa):.0e}I$")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_combined_profiles_plot(path: Path, results: list[CaseResult], *, reference_csv: Path) -> None:
    if plt is None or not results:
        return
    fig, axes = plt.subplots(1, len(results), figsize=(4.8 * len(results), 3.6), constrained_layout=True)
    if len(results) == 1:
        axes = [axes]
    for ax, result in zip(axes, results):
        ax.plot(result.profile_x, result.profile_uy, color="tab:green", lw=2.2, label="one-domain")
        moving = _load_reference_curve(reference_csv=reference_csv, kappa=result.kappa, curve_label="partitioned_moving_linear")
        fixed = _load_reference_curve(reference_csv=reference_csv, kappa=result.kappa, curve_label="partitioned_fixed_linear")
        nonlinear = _load_reference_curve(reference_csv=reference_csv, kappa=result.kappa, curve_label="partitioned_moving_nonlinear")
        if moving is not None:
            ax.plot(moving[0], moving[1], color="#149dff", lw=2.0, label="moving linear")
        if fixed is not None:
            ax.plot(fixed[0], fixed[1], color="red", lw=1.8, ls="--", label="fixed linear")
        if nonlinear is not None:
            ax.plot(nonlinear[0], nonlinear[1], color="#b748ff", lw=1.8, label="moving nonlinear")
        ax.set_title(rf"$\kappa={result.kappa:.0e}I$")
        ax.set_xlabel("x")
        if ax is axes[0]:
            ax.set_ylabel(r"$u_y(x, y=1.25)$")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _run_case(args: argparse.Namespace, *, kappa: float, outdir: Path) -> CaseResult:
    if str(args.fluid_space).strip().lower() == "hdiv":
        raise NotImplementedError(
            "Benchmark 7 solve mode is not yet wired for H(div) fluid boundary conditions. "
            "Use the comparison harness to assemble the H(div) forms/Jacobian."
        )

    poly_order, pressure_order, scalar_order = _resolved_orders(args)
    qdeg = max(6, 2 * int(poly_order) + 2)
    eps_alpha_eff = _effective_eps_alpha(args)
    alpha_reg_eps_normal = float(args.alpha_reg_eps_normal) if args.alpha_reg_eps_normal is not None else float(eps_alpha_eff)
    alpha_reg_eps_tangent = float(args.alpha_reg_eps_tangent) if args.alpha_reg_eps_tangent is not None else float(0.25 * eps_alpha_eff)
    h_char = _characteristic_h(Lx=float(args.Lx), Ly=float(args.Ly), nx=int(args.nx), ny=int(args.ny))
    problem = _create_problem(
        Lx=float(args.Lx),
        Ly=float(args.Ly),
        nx=int(args.nx),
        ny=int(args.ny),
        poly_order=int(poly_order),
        pressure_order=int(pressure_order),
        scalar_order=int(scalar_order),
        fluid_space=str(args.fluid_space),
        fluid_hdiv_order=int(args.fluid_hdiv_order),
        enable_phi_evolution=bool(args.enable_phi_evolution),
    )
    alpha_init = lambda x, y: _alpha_equilibrium(y, y_interface=float(args.y_interface), eps_alpha=float(eps_alpha_eff))
    reg_mask_meta = _configure_regularization_mask(
        problem=problem,
        enabled=bool(args.reg_rect),
        Lx=float(args.Lx),
        Ly=float(args.Ly),
        y_interface=float(args.y_interface),
        center_x=args.reg_rect_center_x,
        center_y=args.reg_rect_center_y,
        half_width=args.reg_rect_half_width,
        half_height=args.reg_rect_half_height,
    )

    problem["alpha_n"].set_values_from_function(lambda x, y: float(alpha_init(x, y)))
    problem["alpha_k"].nodal_values[:] = problem["alpha_n"].nodal_values[:]
    if problem["phi_n"] is not None:
        phi_init = np.clip(
            1.0 - (1.0 - float(args.phi_b)) * np.asarray(problem["alpha_n"].nodal_values, dtype=float),
            0.0,
            1.0,
        )
        problem["phi_n"].nodal_values[:] = phi_init
        problem["phi_k"].nodal_values[:] = phi_init
        a0 = np.asarray(problem["alpha_n"].nodal_values, dtype=float)
        Wp = 2.0 * a0 * (1.0 - a0) * (1.0 - 2.0 * a0)
        problem["mu_n"].nodal_values[:] = float(args.gamma_alpha / max(float(eps_alpha_eff), 1.0e-12)) * Wp
        problem["mu_k"].nodal_values[:] = problem["mu_n"].nodal_values[:]

    dt_c = Constant(float(args.dt))
    forms = _build_forms(
        problem,
        qdeg=qdeg,
        dt_c=dt_c,
        theta=float(args.theta),
        rho_f=float(args.rho_f),
        mu_f=float(args.mu_f),
        mu_b=float(args.mu_b),
        mu_b_model=str(args.mu_b_model),
        kappa_inv=1.0 / float(kappa),
        mu_s=float(args.mu_s),
        lambda_s=float(args.lambda_s),
        phi_b=float(args.phi_b),
        M_alpha=float(args.M_alpha),
        gamma_alpha=float(args.gamma_alpha),
        eps_alpha=float(eps_alpha_eff),
        solid_visco_eta=float(args.solid_visco_eta),
        gamma_div=float(args.gamma_div),
        gamma_u=float(args.gamma_u),
        u_extension_mode=str(args.u_extension),
        gamma_u_pin=float(args.gamma_u_pin),
        u_cip=float(args.u_cip),
        u_cip_weight=str(args.u_cip_weight),
        vS_cip=float(args.vS_cip),
        gamma_vS=args.gamma_vS,
        vS_extension_mode=args.vS_ext_mode,
        gamma_vS_pin=args.gamma_vS_pin,
        D_phi=float(args.D_phi),
        phi_diffusion_weight=str(args.phi_diffusion_weight),
        gamma_phi=float(args.gamma_phi),
        phi_supg=float(args.phi_supg),
        phi_cip=float(args.phi_cip),
        alpha_regularization=str(args.alpha_regularization),
        alpha_reg_gamma=float(args.alpha_reg_gamma),
        alpha_reg_eps_normal=float(alpha_reg_eps_normal),
        alpha_reg_eps_tangent=float(alpha_reg_eps_tangent),
        alpha_reg_eta=float(args.alpha_reg_eta),
        alpha_advect_with=str(args.alpha_advect_with),
        alpha_advection_form=str(args.alpha_advection_form),
        solid_model=str(args.solid_model),
        kappa_inv_model=str(args.kappa_inv_model),
        enable_phi_evolution=bool(args.enable_phi_evolution),
    )

    bcs = _build_bcs(
        y_interface=float(args.y_interface),
        eps_alpha=float(eps_alpha_eff),
        v_in=float(args.v_in),
        t_ramp=float(args.t_ramp),
    )
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, (lambda x, y: 0.0)) for b in bcs]

    outdir.mkdir(parents=True, exist_ok=True)
    vtk_dir = outdir / "vtk"
    dt_c = Constant(float(args.dt))
    timeseries_rows: list[dict[str, float]] = []
    alpha_diagnostics = _build_alpha_diagnostics(problem, quad_order=int(qdeg), backend=str(args.backend))
    alpha_coeffs = {problem["alpha_k"].name: problem["alpha_k"]}
    alpha_diag0 = alpha_diagnostics.evaluate(alpha_coeffs)
    alpha_area0 = float(alpha_diag0.get("alpha_area", float("nan")))
    alpha_band0 = float(alpha_diag0.get("alpha_band", float("nan")))

    def _record_step(functions) -> None:
        step_no = int(getattr(solver, "_current_step_no", len(timeseries_rows) + 1))
        t_now = float(getattr(solver, "_current_t", 0.0) + getattr(solver, "_current_dt", float(args.dt)))
        alpha_diag = alpha_diagnostics.evaluate(alpha_coeffs)
        alpha_area = float(alpha_diag.get("alpha_area", float("nan")))
        alpha_band = float(alpha_diag.get("alpha_band", float("nan")))
        row = {
            "step": float(step_no),
            "t": float(t_now),
            "v_max": float(np.max(np.abs(problem["v_k"].nodal_values))),
            "p_max": float(np.max(np.abs(problem["p_k"].nodal_values))),
            "vS_max": float(np.max(np.abs(problem["vS_k"].nodal_values))),
            "u_y_max": float(np.max(_vector_component_values(problem["u_k"], 1))),
            "u_y_min": float(np.min(_vector_component_values(problem["u_k"], 1))),
            "alpha_min": float(np.min(problem["alpha_k"].nodal_values)),
            "alpha_max": float(np.max(problem["alpha_k"].nodal_values)),
            "alpha_area": alpha_area,
            "alpha_area_rel_drift": float((alpha_area - alpha_area0) / max(abs(alpha_area0), 1.0e-30)),
            "alpha_band": alpha_band,
            "alpha_band_rel_drift": float((alpha_band - alpha_band0) / max(abs(alpha_band0), 1.0e-30)),
        }
        if problem["phi_k"] is not None:
            row["phi_min"] = float(np.min(problem["phi_k"].nodal_values))
            row["phi_max"] = float(np.max(problem["phi_k"].nodal_values))
        timeseries_rows.append(row)
        if int(args.vtk_every) > 0 and (step_no % int(args.vtk_every) == 0):
            vtk_dir.mkdir(parents=True, exist_ok=True)
            vtk_functions = {
                "v": problem["v_k"],
                "p": problem["p_k"],
                "vS": problem["vS_k"],
                "u": problem["u_k"],
                "alpha": problem["alpha_k"],
                "mu_alpha": problem["mu_k"],
            }
            if problem["phi_k"] is not None:
                vtk_functions["phi"] = problem["phi_k"]
                vtk_functions["S"] = problem["S_k"]
            if problem.get("reg_weight") is not None:
                vtk_functions["reg_weight"] = problem["reg_weight"]
            export_vtk(
                str(vtk_dir / f"step={step_no:05d}.vtu"),
                mesh=problem["mesh"],
                dof_handler=problem["dh"],
                functions=vtk_functions,
            )

    newton_params = NewtonParameters(
        newton_tol=float(args.newton_tol),
        newton_rtol=float(args.newton_rtol),
        max_newton_iter=int(args.max_it),
        ls_mode=str(args.ls_mode),
    )
    newton_params.line_search = bool(args.line_search)

    common_solver_kwargs = dict(
        dof_handler=problem["dh"],
        mixed_element=problem["me"],
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=newton_params,
        lin_params=LinearSolverParameters(
            backend=str(args.linear_backend),
            tol=float(args.lin_tol),
            maxit=int(args.lin_maxit),
        ),
        quad_order=int(qdeg),
        backend=str(args.backend),
        postproc_timeloop_cb=_record_step,
    )
    solver_key = str(args.nonlinear_solver).strip().lower()
    if solver_key == "pdas":
        solver = PdasNewtonSolver(
            forms.residual_form,
            forms.jacobian_form,
            vi_params=VIParameters(
                c=float(args.vi_c),
                enter_tol=float(args.vi_enter_tol),
                leave_tol=float(args.vi_leave_tol),
                active_set_persistence=int(args.vi_persistence),
                project_initial_guess=True,
                project_each_iteration=True,
                inactive_reg_lambda0=float(args.vi_lambda0),
                inactive_reg_lambda_max=float(args.vi_lambda_max),
                inactive_reg_growth=float(args.vi_lambda_growth),
                inactive_reg_decay=float(args.vi_lambda_decay),
                active_step_delta_active_trigger=int(args.vi_active_soft_threshold),
                active_step_soft_alpha=float(args.vi_active_soft_alpha),
                active_step_strong_factor=float(args.vi_active_strong_factor),
                filter_max_delta_active=int(args.vi_filter_max_delta_active),
                filter_max_residual_growth=float(args.vi_filter_max_residual_growth),
                filter_max_gap_growth=float(args.vi_filter_max_gap_growth),
                unconstrained_lm=bool(args.vi_unconstrained_lm),
                unconstrained_lm_lambda0=float(args.vi_lm_lambda0),
                unconstrained_lm_lambda_max=float(args.vi_lm_lambda_max),
                unconstrained_lm_growth=float(args.vi_lm_growth),
                unconstrained_lm_decay=float(args.vi_lm_decay),
                unconstrained_lm_accept_ratio=float(args.vi_lm_accept_ratio),
                unconstrained_lm_good_ratio=float(args.vi_lm_good_ratio),
                unconstrained_lm_max_tries=int(args.vi_lm_max_tries),
            ),
            **common_solver_kwargs,
        )
    else:
        solver = NewtonSolver(
            forms.residual_form,
            forms.jacobian_form,
            **common_solver_kwargs,
        )

    if solver_key == "pdas" and hasattr(solver, "set_box_bounds"):
        bounds_by_field: dict[str, tuple[float | None, float | None]] = {}
        if bool(args.alpha_box_constraints):
            bounds_by_field["alpha"] = (0.0, 1.0)
        if bool(args.enable_phi_evolution) and problem["phi_k"] is not None and bool(args.phi_box_constraints):
            bounds_by_field["phi"] = (0.0, 1.0)
        if bounds_by_field:
            solver.set_box_bounds(by_field=bounds_by_field)

    def _on_dt_change(new_dt: float) -> None:
        dt_c.value = float(new_dt)

    current_functions = [
        problem["v_k"],
        problem["p_k"],
        problem["vS_k"],
        problem["u_k"],
        problem["alpha_k"],
        problem["mu_k"],
    ]
    previous_functions = [
        problem["v_n"],
        problem["p_n"],
        problem["vS_n"],
        problem["u_n"],
        problem["alpha_n"],
        problem["mu_n"],
    ]
    if problem["phi_k"] is not None:
        current_functions.extend([problem["phi_k"], problem["S_k"]])
        previous_functions.extend([problem["phi_n"], problem["S_n"]])
    aux_solver_functions: dict[str, object] = {"dt": dt_c}
    if problem.get("reg_weight") is not None:
        aux_solver_functions["reg_weight"] = problem["reg_weight"]
    solve_error: str = ""
    t_start = time.perf_counter()
    try:
        solver.solve_time_interval(
            functions=current_functions,
            prev_functions=previous_functions,
            aux_functions=aux_solver_functions,
            time_params=TimeStepperParameters(
                dt=float(args.dt),
                final_time=float(args.t_final),
                max_steps=int(1.0e9),
                theta=float(args.theta),
                t0=0.0,
                stop_on_steady=False,
                on_dt_change=_on_dt_change,
                allow_dt_reduction=not bool(args.no_dt_reduction),
                dt_min=float(args.dt_min),
                dt_max=(None if args.dt_max is None else float(args.dt_max)),
                dt_reduction_factor=float(args.dt_reduction_factor),
                dt_increase_factor=float(args.dt_increase_factor),
                dt_iters_increase_threshold=int(args.dt_iters_increase_threshold),
                dt_easy_steps_before_increase=int(args.dt_easy_steps_before_increase),
                dt_iters_decrease_threshold=int(args.dt_iters_decrease_threshold),
                dt_decrease_factor_slow=float(args.dt_decrease_factor_slow),
                dt_slow_steps_before_decrease=int(args.dt_slow_steps_before_decrease),
                predictor=str(args.predictor),
                predictor_damping=float(args.predictor_damping),
                predictor_clip_01=bool(args.enable_phi_evolution),
            ),
        )
    except Exception as exc:
        solve_error = str(exc)
        print(f"[warn] solve terminated early: {solve_error}", flush=True)
        for f, f_prev in zip(current_functions, previous_functions):
            f.nodal_values[:] = f_prev.nodal_values[:]
    solve_seconds = time.perf_counter() - t_start
    alpha_diag_final = alpha_diagnostics.evaluate(alpha_coeffs)
    alpha_area_final = float(alpha_diag_final.get("alpha_area", float("nan")))
    alpha_band_final = float(alpha_diag_final.get("alpha_band", float("nan")))

    profile_x, profile_uy = _sample_profile(
        problem=problem,
        Lx=float(args.Lx),
        y_profile=float(args.y_profile),
        n_samples=int(args.profile_samples),
    )
    _write_profile_csv(outdir / "profile_final.csv", x=profile_x, u_y=profile_uy)
    _write_timeseries_csv(outdir / "timeseries.csv", timeseries_rows)

    vtk_final = {
        "v": problem["v_k"],
        "p": problem["p_k"],
        "vS": problem["vS_k"],
        "u": problem["u_k"],
        "alpha": problem["alpha_k"],
        "mu_alpha": problem["mu_k"],
    }
    if problem["phi_k"] is not None:
        vtk_final["phi"] = problem["phi_k"]
        vtk_final["S"] = problem["S_k"]
    if problem.get("reg_weight") is not None:
        vtk_final["reg_weight"] = problem["reg_weight"]
    export_vtk(str(outdir / "final_state.vtu"), mesh=problem["mesh"], dof_handler=problem["dh"], functions=vtk_final)

    reference_csv = Path(args.reference_csv).resolve()
    moving_ref = _load_reference_curve(reference_csv=reference_csv, kappa=float(kappa), curve_label="partitioned_moving_linear")
    fixed_ref = _load_reference_curve(reference_csv=reference_csv, kappa=float(kappa), curve_label="partitioned_fixed_linear")
    nonlinear_ref = _load_reference_curve(reference_csv=reference_csv, kappa=float(kappa), curve_label="partitioned_moving_nonlinear")
    moving_metrics = (
        _compute_profile_metrics(x_num=profile_x, y_num=profile_uy, x_ref=moving_ref[0], y_ref=moving_ref[1])
        if moving_ref is not None
        else None
    )
    fixed_metrics = (
        _compute_profile_metrics(x_num=profile_x, y_num=profile_uy, x_ref=fixed_ref[0], y_ref=fixed_ref[1])
        if fixed_ref is not None
        else None
    )

    vi_c_effective = float(getattr(getattr(solver, "vi_params", None), "c", float(args.vi_c)) or float(args.vi_c))
    vi_c_field_current = getattr(solver, "_vi_c_field_current", {}) if solver_key == "pdas" else {}

    summary_row: dict[str, object] = {
        "kappa": float(kappa),
        "kappa_inv": float(1.0 / float(kappa)),
        "nx": float(args.nx),
        "ny": float(args.ny),
        "dt": float(args.dt),
        "theta": float(args.theta),
        "t_final": float(args.t_final),
        "nonlinear_solver": str(solver_key),
        "alpha_box_constraints": float(1.0 if bool(args.alpha_box_constraints) else 0.0),
        "phi_box_constraints": float(1.0 if bool(args.enable_phi_evolution and args.phi_box_constraints) else 0.0),
        "vi_c": float(args.vi_c),
        "vi_c_effective": float(vi_c_effective),
        "vi_c_fields": json.dumps(vi_c_field_current, sort_keys=True),
        "vi_unconstrained_lm": float(1.0 if bool(args.vi_unconstrained_lm) else 0.0),
        "poly_order": float(poly_order),
        "pressure_order": float(pressure_order),
        "scalar_order": float(scalar_order),
        "solid_model": str(args.solid_model),
        "mu_b_model": str(args.mu_b_model),
        "kappa_inv_model": str(args.kappa_inv_model),
        "phi_evolution": float(1.0 if args.enable_phi_evolution else 0.0),
        "alpha_regularization": str(args.alpha_regularization),
        "alpha_reg_gamma": float(args.alpha_reg_gamma),
        "alpha_reg_eps_normal": float(alpha_reg_eps_normal),
        "alpha_reg_eps_tangent": float(alpha_reg_eps_tangent),
        "alpha_reg_eta": float(args.alpha_reg_eta),
        "alpha_advect_with": str(args.alpha_advect_with),
        "alpha_advection_form": str(args.alpha_advection_form),
        "D_phi": float(args.D_phi),
        "phi_diffusion_weight": str(args.phi_diffusion_weight),
        "t_ramp": float(args.t_ramp),
        "h_char": float(h_char),
        "eps_alpha": float(eps_alpha_eff),
        "eps_alpha_over_h": float(eps_alpha_eff / max(h_char, 1.0e-14)),
        "reg_rect": float(reg_mask_meta["enabled"]),
        "reg_rect_center_x": float(reg_mask_meta["center_x"]),
        "reg_rect_center_y": float(reg_mask_meta["center_y"]),
        "reg_rect_half_width": float(reg_mask_meta["half_width"]),
        "reg_rect_half_height": float(reg_mask_meta["half_height"]),
        "reg_rect_fraction": float(reg_mask_meta["fraction"]),
        "ls_mode": str(args.ls_mode),
        "line_search": float(1.0 if args.line_search else 0.0),
        "predictor": str(args.predictor),
        "predictor_damping": float(args.predictor_damping),
        "solve_seconds": float(solve_seconds),
        "steps_recorded": float(len(timeseries_rows)),
        "alpha_area0": float(alpha_area0),
        "alpha_area_final": float(alpha_area_final),
        "alpha_area_rel_drift": float((alpha_area_final - alpha_area0) / max(abs(alpha_area0), 1.0e-30)),
        "alpha_band0": float(alpha_band0),
        "alpha_band_final": float(alpha_band_final),
        "alpha_band_rel_drift": float((alpha_band_final - alpha_band0) / max(abs(alpha_band0), 1.0e-30)),
        "u_y_max": float(np.max(profile_uy)),
        "u_y_min": float(np.min(profile_uy)),
        "u_y_peak_x": float(profile_x[int(np.argmax(profile_uy))]),
        "solve_completed": float(0.0 if solve_error else 1.0),
    }
    if problem["phi_k"] is not None:
        summary_row.update(
            {
                "phi_min": float(np.min(problem["phi_k"].nodal_values)),
                "phi_max": float(np.max(problem["phi_k"].nodal_values)),
                "phi_mean": float(np.mean(problem["phi_k"].nodal_values)),
            }
        )
    if moving_metrics is not None:
        summary_row.update(
            {
                "rmse_to_moving_linear": float(moving_metrics.rmse),
                "linf_to_moving_linear": float(moving_metrics.linf),
                "rmse_over_amp_moving_linear": float(moving_metrics.rmse_over_amplitude),
                "linf_over_amp_moving_linear": float(moving_metrics.linf_over_amplitude),
                "peak_amp_relerr_moving_linear": float(moving_metrics.peak_amplitude_relative_error),
                "peak_x_error_moving_linear": float(moving_metrics.peak_x_error),
            }
        )
    if fixed_metrics is not None:
        summary_row.update(
            {
                "rmse_to_fixed_linear": float(fixed_metrics.rmse),
                "rmse_over_amp_fixed_linear": float(fixed_metrics.rmse_over_amplitude),
                "closer_to_moving_than_fixed": float(
                    1.0 if (moving_metrics is not None and moving_metrics.rmse <= fixed_metrics.rmse) else 0.0
                ),
            }
        )

    summary_payload = {
        "case": summary_row,
        "solve_error": solve_error,
        "moving_linear_metrics": None if moving_metrics is None else moving_metrics.__dict__,
        "fixed_linear_metrics": None if fixed_metrics is None else fixed_metrics.__dict__,
        "reference_csv": str(reference_csv) if reference_csv.exists() else "",
        "profile_csv": str(outdir / "profile_final.csv"),
        "timeseries_csv": str(outdir / "timeseries.csv"),
        "vtk_final": str(outdir / "final_state.vtu"),
    }
    (outdir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _write_case_plot(
        outdir / "profile_compare.png",
        kappa=float(kappa),
        x_num=profile_x,
        y_num=profile_uy,
        moving_ref=moving_ref,
        fixed_ref=fixed_ref,
        nonlinear_ref=nonlinear_ref,
    )
    return CaseResult(
        kappa=float(kappa),
        outdir=outdir,
        summary_row=summary_row,
        profile_x=profile_x,
        profile_uy=profile_uy,
        moving_metrics=moving_metrics,
        fixed_metrics=fixed_metrics,
    )


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir).resolve()
    kappas = _parse_float_list(args.kappa_list)
    results: list[CaseResult] = []
    for kappa in kappas:
        case_id = f"kappa_{kappa:.0e}".replace("+0", "").replace("-0", "-")
        case_outdir = outdir / case_id
        print(f"[run] kappa={kappa:.6e} -> {case_outdir}", flush=True)
        result = _run_case(args, kappa=float(kappa), outdir=case_outdir)
        results.append(result)

    summary_csv = outdir / "benchmark7_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    if results:
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].summary_row.keys()))
            writer.writeheader()
            for result in results:
                writer.writerow(result.summary_row)

    combined = {
        "cases": [result.summary_row for result in results],
        "reference_csv": str(Path(args.reference_csv).resolve()),
    }
    (outdir / "benchmark7_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    _write_combined_profiles_plot(
        outdir / "benchmark7_seboldt_profiles.png",
        results,
        reference_csv=Path(args.reference_csv).resolve(),
    )
    print(f"[done] wrote {summary_csv}")
    print(f"[done] wrote {outdir / 'benchmark7_summary.json'}")
    if (outdir / "benchmark7_seboldt_profiles.png").exists():
        print(f"[done] wrote {outdir / 'benchmark7_seboldt_profiles.png'}")


if __name__ == "__main__":
    main()
