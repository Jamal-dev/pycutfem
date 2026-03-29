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
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from types import SimpleNamespace

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
    InteriorPointNewtonSolver,
    LinearEqualityConstraint,
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
    exp,
    tanh,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.helpers import analyze_active_dofs
from pycutfem.ufl.measures import dS, ds, dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad
from pycutfem.utils.functionals import NamedFunctionalEvaluator
from pycutfem.utils.mpi import barrier, get_mpi_context

from examples.utils.biofilm.deformation_only import build_deformation_only_forms
from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms


MPI_CTX = get_mpi_context()


def _mpi_io_root() -> bool:
    return (not MPI_CTX.enabled) or MPI_CTX.is_root


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


def _named_constant(name: str, value, *, dim: int | None = None) -> Constant:
    const = value if isinstance(value, Constant) else Constant(value, dim=dim)
    setattr(const, "_jit_name", str(name))
    return const


def _pc_path_tangent_euler_step(
    *,
    x_red: np.ndarray,
    jacobian_red,
    dH_dlambda_red: np.ndarray,
    delta_lambda: float,
    solve_linear_system,
) -> tuple[np.ndarray, np.ndarray]:
    x_now = np.asarray(x_red, dtype=float).ravel()
    dH = np.asarray(dH_dlambda_red, dtype=float).ravel()
    z_dot = np.asarray(
        solve_linear_system(jacobian_red, -dH),
        dtype=float,
    ).ravel()
    if x_now.shape != z_dot.shape:
        raise ValueError(
            f"Tangent solve returned shape {z_dot.shape}, expected {x_now.shape}."
        )
    x_pred = x_now + float(delta_lambda) * z_dot
    return np.asarray(x_pred, dtype=float), np.asarray(z_dot, dtype=float)


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
        "--cpp-fuse-integrals",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "For backend=cpp, fuse same-measure integrals during kernel compilation. "
            "Defaults to on for Benchmark 7 unless PYCUTFEM_CPP_FUSE_INTEGRALS is already set."
        ),
    )
    ap.add_argument(
        "--nonlinear-solver",
        type=str,
        default="pdas",
        choices=("pdas", "ipm", "newton"),
        help="Nonlinear solver for Benchmark 7. PDAS and IPM enforce box constraints on alpha/phi.",
    )
    ap.add_argument(
        "--linear-backend",
        type=str,
        default="scipy",
        choices=("scipy", "petsc", "pardiso"),
        help="Linear algebra backend used inside the Newton/PDAS solve.",
    )
    ap.add_argument(
        "--petsc-distributed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use PETSc COMM_WORLD distributed linear solves when --linear-backend=petsc under mpirun.",
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
    ap.add_argument(
        "--hdiv-tangential-dirichlet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For fluid-space=hdiv, enforce tangential no-slip weakly on velocity-Dirichlet boundaries.",
    )
    ap.add_argument(
        "--hdiv-tangential-gamma",
        type=float,
        default=20.0,
        help="Penalty/Nitsche parameter for H(div) tangential Dirichlet enforcement.",
    )
    ap.add_argument(
        "--hdiv-tangential-method",
        type=str,
        default="penalty",
        choices=("penalty", "nitsche"),
        help="Weak tangential Dirichlet method used for fluid-space=hdiv.",
    )
    ap.add_argument("--pressure-order", type=int, default=None, help="Pressure order; defaults to poly_order-1.")
    ap.add_argument("--scalar-order", type=int, default=None, help="Alpha/mu/phi order; defaults to poly_order-1.")
    ap.add_argument(
        "--quad-order",
        type=int,
        default=None,
        help="Quadrature order override. Defaults to max(6, 2*poly_order + 2).",
    )
    ap.add_argument("--nx", type=int, default=20, help="Cells in x. h=0.05 corresponds to nx=20.")
    ap.add_argument("--ny", type=int, default=30, help="Cells in y. h=0.05 on Ly=1.5 corresponds to ny=30.")
    ap.add_argument("--dt", type=float, default=1.0e-3)
    ap.add_argument("--t-final", type=float, default=3.0)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument(
        "--include-skeleton-acceleration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include Eulerian skeleton acceleration in the full one-domain Benchmark 7 path.",
    )
    ap.add_argument(
        "--rho-s0-tilde",
        type=float,
        default=1.1,
        help="Reference solid density used in the Eulerian skeleton acceleration term.",
    )
    ap.add_argument(
        "--skeleton-inertia-convection",
        type=str,
        default="full",
        choices=("lagged", "full"),
        help="Treatment of the convective part of the Eulerian skeleton inertia.",
    )
    ap.add_argument(
        "--fluid-convection",
        type=str,
        default="full",
        choices=("full", "lagged", "imex", "off"),
        help="Treatment of the convective part of the one-domain fluid momentum block.",
    )
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
        help="Use the full one-domain phi-transport model. Disable only for reduced fixed-phi_b debugging runs.",
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
    ap.add_argument("--phi-b", type=float, default=0.18)
    ap.add_argument("--mu-s", type=float, default=1.67785e5)
    ap.add_argument("--lambda-s", type=float, default=8.22148e6)
    ap.add_argument("--solid-visco-eta", type=float, default=0.0)
    ap.add_argument("--gamma-div", type=float, default=0.0)
    ap.add_argument(
        "--condition-balanced-auto-gamma-div",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For mechanics_nondim_mode=condition_balanced, promote the consistent mixture-divergence "
            "AL term to a permeability-aware default when gamma_div is left at zero."
        ),
    )
    ap.add_argument(
        "--mechanics-nondim-mode",
        type=str,
        default="condition_balanced",
        choices=("legacy", "stress_balance", "condition_balanced"),
        help=(
            "Formulation-level momentum nondimensionalization. "
            "'stress_balance' scales the fluid and skeleton momentum equations by "
            "their constitutive stress scales so the monolithic Jacobian is less "
            "sensitive to mu_s/lambda_s. "
            "'condition_balanced' keeps the same consistent weak form as "
            "'stress_balance' but evaluates the reduced Newton/PDAS system in a "
            "fixed normalized coordinate basis for (p, vS) so the raw operator "
            "is balanced against the Darcy, skeleton, and kinematic scales."
        ),
    )
    ap.add_argument(
        "--solid-volumetric-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use a mixed total-pressure / pore-pressure split for the linear solid: "
            "the skeleton momentum keeps only the deviatoric elastic stress while a "
            "separate scalar pi_s represents the normalized solid total pressure."
        ),
    )
    ap.add_argument(
        "--solid-volumetric-penalty",
        type=float,
        default=1.0,
        help=(
            "Small fluid-side penalty for the mixed volumetric-stress variable pi_s. "
            "Used only when --solid-volumetric-split is enabled."
        ),
    )
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
    ap.add_argument(
        "--phi-cip",
        type=float,
        default=0.0,
        help=(
            "Facet-CIP stabilization for the legacy phi equation. "
            "Not supported for support_physics='internal_conversion'."
        ),
    )
    ap.add_argument("--alpha-supg", type=float, default=0.0, help="SUPG stabilization for alpha transport in full-model mode.")
    ap.add_argument("--alpha-cip", type=float, default=0.0, help="Facet-CIP stabilization for alpha transport in full-model mode.")
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
        default="biofilm_volume",
        choices=("vS", "v", "biofilm_volume", "relative", "mix", "interface", "mix_biofilm"),
        help=(
            "Velocity used to advect the diffuse biofilm indicator alpha. "
            "'biofilm_volume' is the support-preserving choice when alpha tracks the conserved biofilm support."
        ),
    )
    ap.add_argument(
        "--alpha-advection-form",
        type=str,
        default="conservative_weak",
        choices=("advective", "conservative", "conservative_weak", "interface_band_conservative"),
        help="Alpha transport form: advective strong form, conservative strong form, conservative weak/IBP form, or interface-band conservative transport.",
    )
    ap.add_argument(
        "--alpha-from-refmap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Do not solve the alpha transport PDE. Instead, treat alpha as a dependent field "
            "recomputed from the Eulerian reference map u via alpha(x,t)=alpha0(x-u(x,t)) "
            "while keeping phi enabled and solved monolithically."
        ),
    )
    ap.add_argument(
        "--support-physics",
        type=str,
        default="internal_conversion",
        choices=("legacy_exchange", "internal_conversion"),
        help=(
            "Biofilm support model. 'internal_conversion' preserves total alpha and evolves phi through "
            "the conservative solid-volume balance B=alpha(1-phi)."
        ),
    )
    ap.add_argument(
        "--alpha-bc-mode",
        type=str,
        default="auto",
        choices=("auto", "equilibrium", "natural"),
        help=(
            "Boundary treatment for alpha and mu_alpha. "
            "'auto' uses natural no-flux boundaries for the physical benchmark runs."
        ),
    )
    ap.add_argument(
        "--alpha-solid-dirichlet-sides",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Impose alpha=1 on the left/right walls. When latent bounded transport is active, "
            "the matching alpha_latent Dirichlet value is applied through inv_map(alpha)."
        ),
    )
    ap.add_argument(
        "--alpha-solid-dirichlet-bottom",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Impose alpha=1 on the bottom wall. When latent bounded transport is active, "
            "the matching alpha_latent Dirichlet value is applied through inv_map(alpha)."
        ),
    )
    ap.add_argument(
        "--solid-bc-mode",
        type=str,
        default="lateral_clamped",
        choices=("base_only", "wall_normal", "lateral_clamped"),
        help=(
            "Structure displacement / skeleton-velocity boundary treatment. "
            "'base_only' constrains u and vS only at the bottom support; "
            "'wall_normal' enforces no normal solid motion on the left/right walls "
            "(vS_x = 0 and u_x = 0 on this rectangular domain) while keeping tangential slip free; "
            "'lateral_clamped' also clamps the left/right structure sides and matches the pinned lateral displacement "
            "described for the Seboldt benchmark."
        ),
    )
    ap.add_argument(
        "--alpha-biot",
        type=float,
        default=None,
        help=(
            "Optional Biot-Willis coefficient used in the skeleton pressure term. "
            "Set to 1.0 to recover the Seboldt linear-Biot div(eta) coefficient."
        ),
    )
    ap.add_argument(
        "--skeleton-pressure-mode",
        type=str,
        default="whole_domain",
        choices=("whole_domain", "seboldt"),
        help=(
            "Skeleton pressure coupling model. "
            "'whole_domain' keeps the diffuse one-domain split -(p, div(B eta)); "
            "'seboldt' uses the sharp Biot term -(alpha_biot * alpha * p, div(eta))."
        ),
    )
    ap.add_argument("--eps-alpha", type=float, default=0.05)
    ap.add_argument(
        "--eps-alpha-over-h",
        type=float,
        default=0.6,
        help="If set, override eps_alpha with eps_alpha_over_h * h_char, h_char=max(Lx/nx,Ly/ny).",
    )
    ap.add_argument(
        "--kappa-inv-model",
        type=str,
        default="refmap",
        choices=("spatial", "constant", "const", "refmap", "reference-map", "reference_map", "eulerian", "eulerian_refmap"),
        help="Inverse-permeability frame/model. Use refmap/reference-map for the Eulerian push-forward tensor.",
    )
    ap.add_argument(
        "--drag-formulation",
        type=str,
        default="mixed_lm",
        choices=("direct", "mixed_lm"),
        help=(
            "Drag coupling formulation. "
            "'mixed_lm' introduces an auxiliary interaction-force field lambda_drag so the large "
            "Darcy drag enters as a saddle-point constraint instead of a direct penalty term."
        ),
    )
    ap.add_argument(
        "--alpha-mass-constraint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Add a global scalar Lagrange-multiplier row enforcing "
            "int_Omega alpha^{n+1} dx = int_Omega alpha^n dx in the direct-alpha monolithic solve."
        ),
    )
    ap.add_argument(
        "--top-drainage-transport",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For the full evolving-phi branch, keep the top-boundary transport fluxes in the conservative alpha/B "
            "balances so the drained porous top is not artificially closed."
        ),
    )
    ap.add_argument("--v-in", type=float, default=5.0)
    ap.add_argument("--t-ramp", type=float, default=0.0, help="Optional cosine ramp time for the inlet profile [s].")
    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--Ly", type=float, default=1.5)
    ap.add_argument("--y-interface", type=float, default=1.0)
    ap.add_argument(
        "--condition-balanced-solid-cut-fix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optional debugging aid for mechanics_nondim_mode=condition_balanced: "
            "deactivate solid u/vS DOFs above --solid-dof-y-cut. Disabled by default "
            "because Seboldt's porous block lies above y_interface, so using y_interface "
            "as an implicit cutoff would remove the entire deforming solid."
        ),
    )
    ap.add_argument(
        "--solid-dof-y-cut",
        type=float,
        default=None,
        help=(
            "Deactivate u/vS DOFs with y strictly above this cutoff. "
            "No cutoff is applied unless this is set explicitly."
        ),
    )
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
    ap.add_argument("--newton-tol", type=float, default=1.0e-6)
    ap.add_argument("--newton-rtol", type=float, default=1.0e-6)
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
        "--phi-box-alpha-threshold",
        type=float,
        default=5.0e-2,
        help=(
            "When support_physics=internal_conversion, only enforce phi box bounds on DOFs whose matched alpha "
            "value exceeds this threshold; outside the biofilm support phi remains an unconstrained extension field."
        ),
    )
    ap.add_argument(
        "--logistic-bounded-transform",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Solve selected bounded transport fields in logit coordinates with the plain Newton solver. "
            "This keeps the physical residual assembly unchanged, removes PDAS box handling for those fields, "
            "and applies the Jacobian chain rule in solver coordinates."
        ),
    )
    ap.add_argument(
        "--logistic-bounded-fields",
        type=str,
        default="alpha,phi",
        help="Comma-separated fields solved in logit/sigmoid coordinates when --logistic-bounded-transform is enabled.",
    )
    ap.add_argument(
        "--logistic-bounded-eps",
        type=float,
        default=1.0e-8,
        help="Open-interval clipping used by the logit transform to avoid exactly hitting 0 or 1.",
    )
    ap.add_argument(
        "--latent-bounded-transport",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Introduce latent transport unknowns and algebraic relations alpha=map(alpha_latent) "
            "and phi=map(phi_latent). The benchmark then uses plain monolithic Newton without PDAS "
            "or solver-coordinate transform hacks."
        ),
    )
    ap.add_argument(
        "--latent-bounded-fields",
        type=str,
        default="alpha,phi",
        help=(
            "Comma-separated bounded transport fields promoted to latent variables. "
            "In 'transformed' mode both alpha and phi are solved through their latent coordinates; "
            "in 'embedded' mode phi is filtered out because the extra embedded phi/phi_latent coupling is unstable."
        ),
    )
    ap.add_argument(
        "--latent-bounded-map",
        type=str,
        default="sigmoid",
        choices=("sigmoid", "tanh", "algebraic"),
        help="Bounded latent map used for alpha/phi when --latent-bounded-transport is enabled.",
    )
    ap.add_argument(
        "--latent-bounded-formulation",
        type=str,
        default="embedded",
        choices=("embedded", "transformed"),
        help=(
            "Latent bounded formulation: 'embedded' keeps the extra algebraic rows alpha-map(z)=0, "
            "while 'transformed' substitutes alpha=map(z) directly into the PDE and solves only for z."
        ),
    )
    ap.add_argument(
        "--latent-bounded-eps",
        type=float,
        default=1.0e-8,
        help="Open-interval clipping used to initialize latent bounded transport variables from physical values.",
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
        "--vi-variable-column-scaling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply a field-wise right scaling to the reduced VI Newton step. "
            "This is a direct-solve block update preconditioner, not a Krylov linear preconditioner."
        ),
    )
    ap.add_argument(
        "--vi-variable-column-scaling-fields",
        type=str,
        default="v_x,v_y,p,vS_x,vS_y",
        help=(
            "Comma-separated reduced fields that participate in the VI right-scaling layer. "
            "Typical flow/skeleton block: v_x,v_y,p,vS_x,vS_y."
        ),
    )
    ap.add_argument(
        "--pressure-mean-constraint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a clean mixed-element mean-pressure constraint using an auxiliary scalar field p_mean "
            "instead of pinning p on the top boundary."
        ),
    )
    ap.add_argument(
        "--pressure-mean-gauge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a weighted zero-mean gauge term on the fluid pressure block of the monolithic Newton system. "
            "This is a rank-one residual/Jacobian augmentation used to test whether a pressure-gauge deficiency "
            "is limiting the linear solve."
        ),
    )
    ap.add_argument(
        "--pressure-mean-gauge-strength",
        type=float,
        default=1.0,
        help="Strength of the weighted pressure mean-value gauge term.",
    )
    ap.add_argument(
        "--latent-block-preconditioner",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For the latent bounded-transport formulation, keep the staged flow/transport/solid block solve "
            "available as a nonlinear preconditioner / initial-guess builder instead of disabling it."
        ),
    )
    ap.add_argument(
        "--predictor-corrector-startup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable the step-1 predictor-corrector startup guess builder: P0 frozen-transport mechanics, "
            "P1 alpha-released monolithic predictor, then the exact monolithic corrector used only to construct "
            "the first restart state."
        ),
    )
    ap.add_argument(
        "--pc-p0-outer-it",
        type=int,
        default=2,
        help="Number of frozen-transport outer sweeps used in predictor stage P0.",
    )
    ap.add_argument(
        "--pc-p1-max-it",
        type=int,
        default=12,
        help="Maximum Newton iterations used in predictor stage P1.",
    )
    ap.add_argument(
        "--pc-p2-max-it",
        type=int,
        default=12,
        help="Maximum Newton iterations used in predictor stage P2.",
    )
    ap.add_argument(
        "--pc-exact-probe-max-it",
        type=int,
        default=1,
        help=(
            "Before stage P2, try this many exact monolithic Newton iterations from the current P1 basin. "
            "If the exact branch already decreases |R_raw|_inf materially, skip P2."
        ),
    )
    ap.add_argument(
        "--pc-p2-lambda-steps",
        type=int,
        default=4,
        help="Number of continuation substeps used in predictor stage P2 between the lagged model and the exact model.",
    )
    ap.add_argument(
        "--pc-p2-lambda-growth",
        type=float,
        default=1.5,
        help="Growth factor for adaptive P2 continuation steps after a successful lambda stage.",
    )
    ap.add_argument(
        "--pc-p2-lambda-shrink",
        type=float,
        default=0.5,
        help="Shrink factor for adaptive P2 continuation steps after a failed lambda stage.",
    )
    ap.add_argument(
        "--pc-p2-lambda-min-step",
        type=float,
        default=1.0e-3,
        help="Minimum adaptive P2 continuation step size before the homotopy stops.",
    )
    ap.add_argument(
        "--pc-p2-max-substeps",
        type=int,
        default=32,
        help="Maximum number of adaptive P2 continuation substeps attempted before handing over to the exact corrector.",
    )
    ap.add_argument(
        "--pc-p2-fluid-convection",
        type=str,
        default="lagged",
        choices=("full", "lagged", "imex", "off"),
        help="Fluid convection model used in full-field predictor stage P2.",
    )
    ap.add_argument(
        "--pc-p2-skeleton-inertia-convection",
        type=str,
        default="lagged",
        choices=("lagged", "full"),
        help="Skeleton inertia / convection model used in full-field predictor stage P2.",
    )
    ap.add_argument(
        "--pc-p2-easy-dt-divisor",
        type=float,
        default=100.0,
        help=(
            "Pseudo-time-step divisor used for the easy P2 homotopy model G. "
            "The easy model uses dt_easy = dt / this divisor."
        ),
    )
    ap.add_argument(
        "--pc-p2-staggered-anchor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before the full P2 homotopy, build an anchor state with one staggered micro-step "
            "using dt/pc_p2_easy_dt_divisor."
        ),
    )
    ap.add_argument(
        "--pc-p2-staggered-anchor-outer-it",
        type=int,
        default=1,
        help="Number of staggered outer sweeps used to build the P2 micro-step anchor.",
    )
    ap.add_argument(
        "--pc-projection-trials",
        type=int,
        default=5,
        help="Number of geometric segment-projection trials used after predictor stage P1.",
    )
    ap.add_argument(
        "--pc-p2-projection-trials",
        type=int,
        default=12,
        help="Number of geometric segment-projection trials used inside predictor stage P2.",
    )
    ap.add_argument(
        "--pc-p2-max-exact-worsen-rel",
        type=float,
        default=5.0e-2,
        help=(
            "Maximum relative worsening in the exact raw residual allowed during intermediate P2 "
            "homotopy stages, measured against the P2 entry state."
        ),
    )
    ap.add_argument(
        "--pc-min-rel-improve",
        type=float,
        default=1.0e-2,
        help="Minimum relative improvement in the exact monolithic raw residual required to keep a predictor stage.",
    )
    ap.add_argument(
        "--pc-energy-mass-weight",
        type=float,
        default=1.0,
        help="Weight of the relative alpha-mass defect in the predictor-corrector energy.",
    )
    ap.add_argument(
        "--pc-alpha-mass-tol",
        type=float,
        default=1.0e-10,
        help="Maximum acceptable relative alpha-mass defect after predictor return mapping.",
    )
    ap.add_argument(
        "--pc-alpha-return-map-max-it",
        type=int,
        default=64,
        help="Maximum bisection iterations used by the alpha-mass return mapping on predictor states.",
    )
    ap.add_argument(
        "--pc-min-abs-decrease",
        type=float,
        default=1.0e-10,
        help="Minimum exact |R_raw|_inf decrease required before a predictor/P2 stage counts as real progress.",
    )
    ap.add_argument(
        "--pc-p2-reentry-max-retries",
        type=int,
        default=0,
        help=(
            "Maximum number of legacy P2 re-entry retries allowed after the exact corrector stalls on the same step. "
            "Default 0 disables P2 re-entry so the monolithic SQP/globalization recovery path remains primary."
        ),
    )
    ap.add_argument(
        "--pc-p2-reentry-lambda0",
        type=float,
        default=0.0,
        help=(
            "First relaxed lambda used when the exact corrector re-enters P2 from lambda=1. "
            "Non-positive values fall back to the same cautious initial P2 lambda step used by startup."
        ),
    )
    ap.add_argument(
        "--pc-p2-reentry-lambda-min",
        type=float,
        default=1.0e-3,
        help="Smallest relaxed lambda used by P2 re-entry before giving up.",
    )
    ap.add_argument(
        "--pc-prebuild-p2-kernels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prebuild and cache the P2 blended solver once so later startup/re-entry passes reuse the same kernels.",
    )
    ap.add_argument(
        "--newton-equation-row-scaling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply PDAS-style field-wise row scaling to the unconstrained Newton reduced linear system.",
    )
    ap.add_argument(
        "--newton-variable-column-scaling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply PDAS-style field-wise column scaling to the unconstrained Newton reduced linear system.",
    )
    ap.add_argument(
        "--newton-variable-column-scaling-fields",
        type=str,
        default="",
        help=(
            "Comma-separated fields included in the Newton column scaling layer. "
            "Leave empty to scale every active field in the reduced system."
        ),
    )
    ap.add_argument(
        "--newton-reduced-scaling-mode",
        type=str,
        default="field",
        choices=("field", "ruiz"),
        help=(
            "Reduced linear-system scaling used by unconstrained Newton. "
            "'field' keeps the old field-median row/column scaling. "
            "'ruiz' applies full reduced-system Ruiz equilibration."
        ),
    )
    ap.add_argument(
        "--newton-ruiz-iters",
        type=int,
        default=6,
        help="Number of Ruiz equilibration sweeps used when --newton-reduced-scaling-mode=ruiz.",
    )
    ap.add_argument(
        "--newton-pressure-schur-solve",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use an exact reduced pressure-block Schur solve on the unconstrained Newton linear system. "
            "This targets the weak p/vS saddle block directly."
        ),
    )
    ap.add_argument(
        "--newton-pressure-schur-fields",
        type=str,
        default="p,p_mean",
        help="Comma-separated reduced fields treated as the pressure block in the Newton Schur solve.",
    )
    ap.add_argument(
        "--newton-pressure-schur-diag-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use only the diagonal of the reduced Schur complement instead of the full dense block.",
    )
    ap.add_argument(
        "--newton-pressure-schur-shift-rel",
        type=float,
        default=1.0e-12,
        help="Relative diagonal shift used to stabilize the reduced dense Schur complement.",
    )
    ap.add_argument(
        "--newton-pressure-schur-scale-mode",
        type=str,
        default="none",
        choices=("none", "constant", "drag", "inv_drag"),
        help=(
            "Pressure-block normalization used inside the reduced Newton Schur solve. "
            "'drag' uses mu_f / kappa so the Schur block is normalized with the Darcy drag scale."
        ),
    )
    ap.add_argument(
        "--newton-pressure-schur-scale-value",
        type=float,
        default=1.0,
        help=(
            "Constant scale used when --newton-pressure-schur-scale-mode=constant. "
            "Ignored by the drag-based modes."
        ),
    )
    ap.add_argument(
        "--newton-pressure-schur-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print reduced Schur block statistics during Newton linear solves.",
    )
    ap.add_argument(
        "--vi-ptc-recovery",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable identified-manifold pseudo-transient continuation on a selected reduced mechanics block.",
    )
    ap.add_argument(
        "--vi-ptc-fields",
        type=str,
        default="v_x,v_y,p,vS_x,vS_y,u_x,u_y",
        help="Comma-separated reduced fields included in the PTC identified-manifold block.",
    )
    ap.add_argument("--vi-ptc-sigma0", type=float, default=1.0e-2, help="Initial PTC diagonal weight.")
    ap.add_argument("--vi-ptc-sigma-max", type=float, default=1.0e8, help="Maximum PTC diagonal weight.")
    ap.add_argument("--vi-ptc-growth", type=float, default=5.0, help="PTC diagonal growth factor after a stall.")
    ap.add_argument("--vi-ptc-decay", type=float, default=0.5, help="PTC diagonal decay factor after a good step.")
    ap.add_argument(
        "--newton-ptc-operator-mode",
        type=str,
        default="row_normalized",
        choices=("row_normalized", "sym", "diag"),
        help="Operator used for Newton-side PTC regularization on the latent/unconstrained branch.",
    )
    ap.add_argument(
        "--newton-ptc-late-fields",
        type=str,
        default="",
        help=(
            "Optional late-phase Newton PTC field mask. When non-empty and the residual falls below "
            "--newton-ptc-late-switch-residual, the Newton solver switches from --vi-ptc-fields "
            "to this field list."
        ),
    )
    ap.add_argument(
        "--newton-ptc-late-switch-residual",
        type=float,
        default=0.0,
        help="Residual threshold for switching Newton PTC from the base field mask to the late-phase field mask.",
    )
    ap.add_argument(
        "--newton-ptc-late-operator-mode",
        type=str,
        default="",
        choices=("", "row_normalized", "sym", "diag"),
        help=(
            "Optional late-phase Newton PTC operator. Empty keeps --newton-ptc-operator-mode after the field-mask switch."
        ),
    )
    ap.add_argument(
        "--vi-ptc-ginf-trigger",
        type=float,
        default=5.0e-2,
        help="Minimum |G|_inf required before arming the identified-manifold PTC phase.",
    )
    ap.add_argument(
        "--vi-ptc-ginf-max",
        type=float,
        default=2.0e-1,
        help="Maximum |G|_inf for forcing the identified-window PTC phase.",
    )
    ap.add_argument(
        "--vi-anderson-acceleration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Anderson acceleration on the guarded/proximal/PTC local fixed-point map.",
    )
    ap.add_argument("--vi-anderson-history", type=int, default=3, help="Anderson history length.")
    ap.add_argument(
        "--vi-anderson-regularization",
        type=float,
        default=1.0e-10,
        help="Tikhonov regularization used in the Anderson least-squares system.",
    )
    ap.add_argument(
        "--vi-anderson-damping",
        type=float,
        default=0.85,
        help="Damping applied to Anderson mixed candidates.",
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
    ap.add_argument("--vi-ipm-mu0", type=float, default=1.0e-2, help="Initial barrier parameter for the interior-point constrained solver.")
    ap.add_argument("--vi-ipm-mu-min", type=float, default=1.0e-10, help="Terminal barrier parameter for the interior-point constrained solver.")
    ap.add_argument("--vi-ipm-mu-decay", type=float, default=0.2, help="Barrier decay factor between interior-point homotopy stages.")
    ap.add_argument("--vi-ipm-max-barrier-steps", type=int, default=12, help="Maximum number of interior-point barrier stages per nonlinear solve.")
    ap.add_argument("--vi-ipm-fraction-to-boundary", type=float, default=0.995, help="Fraction-to-the-boundary safeguard used by the interior-point solver.")
    ap.add_argument("--vi-ipm-armijo-c1", type=float, default=1.0e-4, help="Armijo decrease constant for the interior-point residual line search.")
    ap.add_argument("--vi-ipm-step-reduction", type=float, default=0.5, help="Backtracking factor used by the interior-point line search.")
    ap.add_argument("--vi-ipm-step-min", type=float, default=1.0e-10, help="Minimum admissible backtracking step for the interior-point solver.")
    ap.add_argument("--vi-ipm-initial-push", type=float, default=1.0e-8, help="Strict-interior push used when initializing the interior-point slacks.")
    ap.add_argument("--vi-ipm-stage-tol-factor", type=float, default=0.25, help="Barrier-stage stopping tolerance factor relative to the current mu.")
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
        "--newton-globalization",
        type=str,
        default="line_search",
        choices=("line_search", "trust_region", "line_search_then_trust"),
        help=(
            "Step globalization for the unconstrained Newton path. "
            "'line_search_then_trust' first tries the current line search and falls back to dogleg trust-region."
        ),
    )
    ap.add_argument("--trust-radius-init", type=float, default=1.0, help="Initial trust-region radius in the reduced trust metric.")
    ap.add_argument("--trust-radius-max", type=float, default=1.0e3, help="Maximum trust-region radius.")
    ap.add_argument("--trust-max-it", type=int, default=8, help="Maximum trust-region trial contractions per Newton step.")
    ap.add_argument("--trust-eta-accept", type=float, default=1.0e-4, help="Trust-region accept threshold on rho.")
    ap.add_argument("--trust-eta-contract", type=float, default=2.5e-1, help="Trust-region shrink threshold on rho.")
    ap.add_argument("--trust-eta-expand", type=float, default=7.5e-1, help="Trust-region expand threshold on rho.")
    ap.add_argument("--trust-shrink", type=float, default=2.5e-1, help="Trust-region radius shrink factor.")
    ap.add_argument("--trust-expand", type=float, default=2.0, help="Trust-region radius expansion factor.")
    ap.add_argument("--trust-min-radius", type=float, default=1.0e-10, help="Minimum trust-region radius.")
    ap.add_argument(
        "--trust-min-abs-residual-drop",
        type=float,
        default=1.0e-10,
        help="Minimum accepted |R|_inf decrease required by the exact-corrector trust-region branch.",
    )
    ap.add_argument(
        "--trust-min-rel-residual-drop",
        type=float,
        default=1.0e-2,
        help="Relative |R|_inf decrease floor used by the exact-corrector trust-region branch.",
    )
    ap.add_argument(
        "--newton-stall-window",
        type=int,
        default=8,
        help="Residual-history window used to detect an exact-corrector Newton stall before handing control back to P2.",
    )
    ap.add_argument(
        "--newton-stall-min-abs-residual-drop",
        type=float,
        default=1.0e-10,
        help="Minimum absolute |R|_inf decrease required across the Newton stall window.",
    )
    ap.add_argument(
        "--newton-stall-min-rel-residual-drop",
        type=float,
        default=1.0e-2,
        help="Minimum relative |R|_inf decrease required across the Newton stall window.",
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
    ap.add_argument(
        "--startup-bootstrap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "On the first failed step, solve one simpler startup problem to build a better "
            "initial guess before reducing dt."
        ),
    )
    ap.add_argument(
        "--startup-bootstrap-fluid-convection",
        type=str,
        default="off",
        choices=("full", "lagged", "imex", "off"),
        help="Fluid convection model used in the simpler startup bootstrap problem.",
    )
    ap.add_argument(
        "--startup-bootstrap-include-skeleton-acceleration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include skeleton acceleration in the simpler startup bootstrap problem.",
    )
    ap.add_argument(
        "--startup-bootstrap-max-it",
        type=int,
        default=20,
        help="Maximum nonlinear iterations used for the simpler startup bootstrap problem.",
    )
    ap.add_argument(
        "--startup-monolithic-max-it",
        type=int,
        default=None,
        help=(
            "Optional Newton-iteration budget for the first full monolithic step after a startup "
            "guess is applied. Defaults to a benchmark-local boosted budget."
        ),
    )
    ap.add_argument(
        "--startup-stage-relaxed-ginf",
        type=float,
        default=None,
        help=(
            "Relaxed semismooth accept threshold used only by the startup stage solvers when a "
            "line search stalls. Defaults to max(1e3*newton_tol, 1e-6) instead of the old hardcoded 5e-2."
        ),
    )
    ap.add_argument(
        "--startup-staggered-outer-it",
        type=int,
        default=3,
        help=(
            "Number of fluid/solid outer sweeps used to build the first-step staggered startup "
            "guess before retrying the monolithic solve."
        ),
    )
    ap.add_argument(
        "--startup-preload-prev-blend",
        type=float,
        default=1.0,
        help=(
            "After a staggered startup/preload, keep the transport fields but blend the "
            "remaining fields back toward the previous accepted state before the "
            "monolithic restart. 1.0 keeps the staged state; 0.0 reuses the previous state."
        ),
    )
    ap.add_argument(
        "--startup-transport-solver",
        type=str,
        default="auto",
        choices=("auto", "newton", "pdas", "ipm"),
        help=(
            "Nonlinear solver used only for the staggered startup transport stage. "
            "'auto' follows the monolithic solver; 'pdas' is useful when the main solve uses ipm "
            "but the startup transport preload should use a more robust active-set predictor."
        ),
    )
    ap.add_argument(
        "--later-step-staggered-outer-it",
        type=int,
        default=1,
        help=(
            "Number of fluid/solid outer sweeps used to replace an aggressive later-step delta "
            "predictor with a staggered initial guess."
        ),
    )
    ap.add_argument(
        "--stall-frozen-transport-restart",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When the full PDAS/VI solve stalls with a nearly fixed transport active set, "
            "freeze alpha/phi and run a reduced flow/solid restart as a nonlinear preconditioner "
            "before cutting dt."
        ),
    )
    ap.add_argument(
        "--stall-frozen-transport-outer-it",
        type=int,
        default=1,
        help="Number of frozen-transport flow/solid outer sweeps used in the nonlinear-preconditioner restart.",
    )
    ap.add_argument(
        "--stall-frozen-transport-max-delta-active",
        type=int,
        default=2,
        help="Maximum active-set change allowed before the frozen-transport restart is considered in a stalled VI solve.",
    )
    ap.add_argument(
        "--stall-frozen-transport-max-gap",
        type=float,
        default=1.0e-6,
        help="Maximum active-gap infinity norm for arming the frozen-transport nonlinear restart.",
    )
    ap.add_argument(
        "--stall-frozen-transport-max-eq",
        type=float,
        default=1.0e-8,
        help="Maximum equality infinity norm for arming the frozen-transport nonlinear restart.",
    )
    ap.add_argument(
        "--stall-frozen-transport-min-ginf",
        type=float,
        default=1.0e-2,
        help="Minimum VI infinity residual required before attempting the frozen-transport nonlinear restart.",
    )
    ap.add_argument(
        "--stall-frozen-transport-min-rel-improve",
        type=float,
        default=5.0e-2,
        help=(
            "Minimum relative decrease in the full raw monolithic residual required to keep a "
            "frozen-transport nonlinear restart."
        ),
    )
    ap.add_argument(
        "--delta-predictor-reset-threshold",
        type=float,
        default=100.0,
        help=(
            "If the previous accepted step had ΔU_step∞ above this threshold, replace the next "
            "step's default predictor with a staggered fluid/transport/solid guess."
        ),
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


def _configure_benchmark7_cpp_fuse_integrals(*, backend: str, enabled: bool | None) -> str | None:
    backend_key = str(backend).strip().lower()
    if backend_key not in {"cpp", "c++"}:
        return os.environ.get("PYCUTFEM_CPP_FUSE_INTEGRALS")
    prev = os.environ.get("PYCUTFEM_CPP_FUSE_INTEGRALS")
    if enabled is None:
        os.environ.setdefault("PYCUTFEM_CPP_FUSE_INTEGRALS", "1")
    else:
        os.environ["PYCUTFEM_CPP_FUSE_INTEGRALS"] = "1" if bool(enabled) else "0"
    return prev


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


def _full_top_drainage_transport_enabled(*, enable_phi_evolution: bool, top_drainage_transport: bool) -> bool:
    return bool(enable_phi_evolution) and bool(top_drainage_transport)


def _predictor_corrector_startup_enabled(
    args: argparse.Namespace,
    problem: dict | None = None,
) -> bool:
    if not bool(getattr(args, "predictor_corrector_startup", False)):
        return False
    if not bool(getattr(args, "enable_phi_evolution", False)):
        return False
    if problem is not None and problem.get("phi_k") is None:
        return False
    return True


def _startup_stage_solver_kind(
    *,
    main_solver_kind: str,
    active_fields: list[str],
    stage_name: str | None = None,
    transport_solver_kind_override: str | None = None,
) -> str:
    override_key = str(transport_solver_kind_override or "auto").strip().lower()
    if str(stage_name or "").strip().lower() == "transport" and override_key in {"newton", "pdas", "ipm"}:
        return override_key
    solver_key = str(main_solver_kind).strip().lower()
    active = {str(name) for name in list(active_fields or [])}
    if solver_key in {"pdas", "ipm"} and bool(active & {"alpha", "phi"}):
        return solver_key
    return "newton"


def _startup_monolithic_max_it(args: argparse.Namespace) -> int:
    raw = getattr(args, "startup_monolithic_max_it", None)
    if raw is not None:
        return max(1, int(raw))
    base = max(1, int(getattr(args, "max_it", 1)))
    return max(base, 24, 2 * base)


def _startup_stage_relaxed_accept_ginf(args: argparse.Namespace) -> float:
    raw = getattr(args, "startup_stage_relaxed_ginf", None)
    if raw is not None:
        try:
            val = float(raw)
        except Exception:
            val = float("nan")
        if np.isfinite(val) and val > 0.0:
            return float(val)
    tol = max(float(getattr(args, "newton_tol", 1.0e-8) or 1.0e-8), 1.0e-16)
    return max(1.0e3 * tol, 1.0e-6)


def _pc_p2_lambda_schedule(n_steps: int, *, include_zero: bool = False) -> list[float]:
    n = max(1, int(n_steps))
    out: list[float] = [0.0] if bool(include_zero) else []
    if n <= 1:
        out.append(1.0)
        return out
    for i in range(1, n + 1):
        out.append(float(i * i) / float(n * n))
    return out


def _pc_p2_easy_dt_value(args: argparse.Namespace, dt_now: float) -> float:
    dt_val = max(abs(float(dt_now)), 1.0e-12)
    divisor = max(1.0, float(getattr(args, "pc_p2_easy_dt_divisor", 100.0) or 100.0))
    return float(dt_val / divisor)


def _pc_fluid_convection_selectors(mode: str) -> dict[str, float]:
    key = str(mode).strip().lower()
    if key == "explicit":
        key = "imex"
    if key not in {"full", "lagged", "imex", "off"}:
        raise ValueError(
            f"Unsupported predictor-corrector fluid_convection={mode!r}. "
            "Use 'full', 'lagged', 'imex', or 'off'."
        )
    return {
        "full": 1.0 if key == "full" else 0.0,
        "lagged": 1.0 if key == "lagged" else 0.0,
        "imex": 1.0 if key == "imex" else 0.0,
    }


def _pc_skeleton_inertia_selectors(mode: str) -> dict[str, float]:
    key = str(mode).strip().lower()
    if key in {"conservative", "nonlinear"}:
        key = "full"
    if key in {"picard", "semi", "semi_implicit", "linear"}:
        key = "lagged"
    if key not in {"full", "lagged"}:
        raise ValueError(
            f"Unsupported predictor-corrector skeleton_inertia_convection={mode!r}. "
            "Use 'full' or 'lagged'."
        )
    return {
        "full": 1.0 if key == "full" else 0.0,
        "lagged": 1.0 if key == "lagged" else 0.0,
    }


def _pc_required_drop(
    raw_before: float,
    *,
    min_abs_decrease: float = 1.0e-10,
    min_rel_improve: float = 0.0,
) -> float:
    abs_floor = max(0.0, float(min_abs_decrease))
    rel_floor = max(0.0, float(min_rel_improve))
    if not np.isfinite(float(raw_before)) or float(raw_before) <= 0.0:
        return float(abs_floor)
    return float(max(abs_floor, rel_floor * float(raw_before)))


def _pc_progress(
    before_stats: dict[str, float],
    after_stats: dict[str, float],
    *,
    key: str = "raw_inf",
    min_abs_decrease: float = 1.0e-10,
    min_rel_improve: float = 0.0,
) -> dict[str, float | bool]:
    raw_before = float(before_stats.get(key, float("nan")))
    raw_after = float(after_stats.get(key, float("nan")))
    drop = float(raw_before - raw_after) if np.isfinite(raw_before) and np.isfinite(raw_after) else float("nan")
    required = _pc_required_drop(
        raw_before,
        min_abs_decrease=float(min_abs_decrease),
        min_rel_improve=float(min_rel_improve),
    )
    improved = bool(
        np.isfinite(raw_before)
        and np.isfinite(raw_after)
        and raw_after < raw_before
        and np.isfinite(drop)
        and drop >= required
    )
    return {
        "key": str(key),
        "before": float(raw_before),
        "after": float(raw_after),
        "improved": bool(improved),
        "drop": float(drop),
        "required": float(required),
    }


def _pc_should_keep_lambda_stage(
    *,
    lam: float,
    before_stats: dict[str, float],
    after_stats: dict[str, float],
    exact_reference_stats: dict[str, float] | None = None,
    alpha_mass_ok: bool,
    min_abs_decrease: float = 1.0e-10,
    min_rel_improve: float = 0.0,
    max_exact_worsen_rel: float = 5.0e-2,
    homotopy_tol: float = 1.0e-6,
) -> tuple[bool, dict[str, object]]:
    exact_progress = _pc_progress(
        before_stats,
        after_stats,
        key="raw_inf",
        min_abs_decrease=float(min_abs_decrease),
        min_rel_improve=float(min_rel_improve),
    )
    homotopy_progress = _pc_progress(
        before_stats,
        after_stats,
        key="homotopy_raw_inf",
        min_abs_decrease=float(min_abs_decrease),
        min_rel_improve=float(min_rel_improve),
    )
    homotopy_after = float(after_stats.get("homotopy_raw_inf", after_stats.get("raw_inf", float("nan"))))
    homotopy_converged = bool(np.isfinite(homotopy_after) and homotopy_after <= max(float(homotopy_tol), 1.0e-16))
    ref_stats = before_stats if exact_reference_stats is None else exact_reference_stats
    exact_ref = float(ref_stats.get("raw_inf", float("nan")))
    exact_after = float(after_stats.get("raw_inf", float("nan")))
    rel_worsen_cap = max(0.0, float(max_exact_worsen_rel))
    exact_guard_limit = (
        float(exact_ref) * (1.0 + rel_worsen_cap)
        if np.isfinite(exact_ref)
        else float("nan")
    )
    exact_within_guard = bool(
        np.isfinite(exact_after)
        and (
            not np.isfinite(exact_guard_limit)
            or exact_after <= exact_guard_limit
        )
    )
    if float(lam) >= 1.0 - 1.0e-14:
        keep = bool(alpha_mass_ok) and bool(exact_progress["improved"] or homotopy_converged) and bool(exact_within_guard)
    else:
        keep = bool(alpha_mass_ok) and bool(homotopy_progress["improved"] or homotopy_converged) and bool(exact_within_guard)
    return bool(keep), {
        "lambda": float(lam),
        "exact": dict(exact_progress),
        "homotopy": dict(homotopy_progress),
        "homotopy_converged": bool(homotopy_converged),
        "alpha_mass_ok": bool(alpha_mass_ok),
        "exact_reference": float(exact_ref),
        "exact_guard_limit": float(exact_guard_limit),
        "exact_within_guard": bool(exact_within_guard),
    }


def _pc_should_prefer_exact_probe(
    *,
    before_stats: dict[str, float],
    after_stats: dict[str, float],
    alpha_mass_ok: bool,
    min_abs_decrease: float = 1.0e-10,
    min_rel_improve: float = 0.0,
    strong_min_abs_decrease: float = 0.0,
    strong_min_rel_improve: float = 0.0,
) -> tuple[bool, dict[str, object]]:
    exact_progress = _pc_progress(
        before_stats,
        after_stats,
        key="raw_inf",
        min_abs_decrease=float(min_abs_decrease),
        min_rel_improve=float(min_rel_improve),
    )
    strong_exact_progress = _pc_progress(
        before_stats,
        after_stats,
        key="raw_inf",
        min_abs_decrease=max(float(min_abs_decrease), float(strong_min_abs_decrease)),
        min_rel_improve=max(float(min_rel_improve), float(strong_min_rel_improve)),
    )
    require_strong_progress = bool(
        float(strong_min_abs_decrease) > 0.0 or float(strong_min_rel_improve) > 0.0
    )
    keep = bool(alpha_mass_ok) and bool(exact_progress["improved"])
    if require_strong_progress:
        keep = bool(keep) and bool(strong_exact_progress["improved"])
    return bool(keep), {
        "exact": dict(exact_progress),
        "strong_exact": dict(strong_exact_progress),
        "alpha_mass_ok": bool(alpha_mass_ok),
    }


def _parse_csv_fields(raw: str | None) -> tuple[str, ...]:
    parts = []
    for tok in str(raw or "").replace(";", ",").split(","):
        item = tok.strip()
        if item:
            parts.append(item)
    return tuple(parts)


def _effective_logistic_bounded_fields(args: argparse.Namespace) -> tuple[str, ...]:
    requested = _parse_csv_fields(getattr(args, "logistic_bounded_fields", ""))
    keep: list[str] = []
    for name in requested:
        key = str(name).strip()
        if not key or key in keep:
            continue
        if key == "alpha" and bool(getattr(args, "alpha_from_refmap", False)):
            continue
        if key == "phi" and not bool(getattr(args, "enable_phi_evolution", False)):
            continue
        keep.append(key)
    return tuple(keep)


def _effective_latent_bounded_fields(args: argparse.Namespace, *, enable_phi_evolution: bool) -> tuple[str, ...]:
    requested = _parse_csv_fields(getattr(args, "latent_bounded_fields", ""))
    formulation = _latent_bounded_formulation_key(args)
    active: list[str] = []
    for name in requested:
        key = str(name).strip()
        if not key or key in active:
            continue
        if key == "alpha":
            if bool(getattr(args, "alpha_from_refmap", False)):
                continue
            active.append("alpha")
        elif key == "phi" and bool(enable_phi_evolution):
            if formulation != "transformed":
                print(
                    "[info] disabling embedded latent phi: the current embedded phi/phi_latent pair creates "
                    "a near-null algebraic mode near convergence. Keeping phi as a direct field."
                )
                continue
            active.append("phi")
    return tuple(active)


def _logistic_refmap_phi_only_mode(args: argparse.Namespace) -> bool:
    if not bool(getattr(args, "logistic_bounded_transform", False)):
        return False
    if not bool(getattr(args, "alpha_from_refmap", False)):
        return False
    return _effective_logistic_bounded_fields(args) == ("phi",)


def _condition_balanced_field_scales(
    *,
    mechanics_nondim_mode: str,
    drag_formulation: str = "direct",
    dt,
    mu_f: float,
    kappa_inv: float,
    mu_s: float,
    lambda_s: float,
    rho_s0_tilde: float,
    dim: int,
) -> dict[str, float]:
    key = str(mechanics_nondim_mode).strip().lower()
    if key != "condition_balanced":
        return {}
    drag_key = str(drag_formulation or "direct").strip().lower().replace("-", "_")
    if drag_key == "mixed_lm":
        # The mixed drag block already turns the Brinkman penalty into a balanced
        # saddle-point system on the assembled operator. Applying the legacy
        # extra coordinate scaling on top of that distorts the lambda-coupled
        # rows and reintroduces an artificial kappa trend in the reduced solve.
        return {}
    dt_val = max(abs(float(dt)), 1.0e-30)
    solid_ref = max(1.0, abs(float(2.0 * mu_s + float(dim) * lambda_s)))
    darcy_ref = max(abs(float(mu_f)) * max(abs(float(kappa_inv)), 1.0e-30), 1.0e-30)
    rho_s_ref = max(abs(float(rho_s0_tilde)), 1.0e-30)
    vS_ref = dt_val * math.sqrt(solid_ref / rho_s_ref)
    return {
        "p": float(darcy_ref),
        "p_mean": float(darcy_ref),
        "vS_x": float(vS_ref),
        "vS_y": float(vS_ref),
    }


def _condition_balanced_kinematic_setup(
    *,
    mechanics_nondim_mode: str,
    mu_f: float,
    kappa_inv: float,
    gamma_u: float,
    u_extension_mode: str,
    gamma_u_pin: float,
    gamma_vS: float | None,
    vS_extension_mode: str | None,
    gamma_vS_pin: float | None,
) -> dict[str, float | str | None]:
    key = str(mechanics_nondim_mode).strip().lower()
    gamma_u_eff = float(gamma_u)
    u_extension_mode_eff = str(u_extension_mode)
    gamma_u_pin_eff = float(gamma_u_pin)
    gamma_vS_eff = None if gamma_vS is None else float(gamma_vS)
    vS_extension_mode_eff = None if vS_extension_mode is None else str(vS_extension_mode)
    gamma_vS_pin_eff = None if gamma_vS_pin is None else float(gamma_vS_pin)
    kinematics_scale = 1.0
    if key != "condition_balanced":
        return {
            "gamma_u": float(gamma_u_eff),
            "u_extension_mode": str(u_extension_mode_eff),
            "gamma_u_pin": float(gamma_u_pin_eff),
            "gamma_vS": gamma_vS_eff,
            "vS_extension_mode": vS_extension_mode_eff,
            "gamma_vS_pin": gamma_vS_pin_eff,
            "kinematics_scale": float(kinematics_scale),
        }

    auto_u_grad_extension = (
        gamma_u_eff == 0.0
        and gamma_u_pin_eff == 0.0
        and str(u_extension_mode_eff).strip().lower() in {"l2", "mass"}
    )
    if auto_u_grad_extension:
        # In the free-fluid region the reference map is only an extension field, so
        # use H1 diffusion there by default and add a tiny pin to remove the rigid
        # translation nullspace of the pure grad seminorm.
        gamma_u_eff = 1.0
        u_extension_mode_eff = "grad"
        gamma_u_pin_eff = 1.0e-6

    if gamma_vS_eff is None and vS_extension_mode_eff is None and gamma_vS_pin_eff is None and auto_u_grad_extension:
        gamma_vS_eff = gamma_u_eff
        vS_extension_mode_eff = "grad"
        gamma_vS_pin_eff = gamma_u_pin_eff

    # Keep the transport-style kinematic block on an O(1) equation scale.
    #
    # A permeability-aware row factor here reintroduces an artificial kappa
    # dependence even on common-mode diagnostics where the drag block should
    # cancel (for example after restricting to v = vS). The solver-side field
    # normalization remains the correct place to balance the Darcy scale.
    kinematics_scale = 1.0
    return {
        "gamma_u": float(gamma_u_eff),
        "u_extension_mode": str(u_extension_mode_eff),
        "gamma_u_pin": float(gamma_u_pin_eff),
        "gamma_vS": gamma_vS_eff,
        "vS_extension_mode": vS_extension_mode_eff,
        "gamma_vS_pin": gamma_vS_pin_eff,
        "kinematics_scale": float(kinematics_scale),
    }


def _condition_balanced_volume_setup(
    *,
    mechanics_nondim_mode: str,
    mu_f: float,
    kappa_inv: float,
    gamma_div: float,
    auto_gamma_div: bool,
) -> dict[str, float]:
    key = str(mechanics_nondim_mode).strip().lower()
    darcy_ref = max(abs(float(mu_f)) * max(abs(float(kappa_inv)), 1.0e-30), 1.0e-30)
    gamma_div_eff = float(gamma_div)
    if key == "condition_balanced" and bool(auto_gamma_div) and gamma_div_eff == 0.0:
        # Pressure is normalized on the Darcy scale p ~ mu_f / kappa, so the
        # default AL weight uses the reciprocal scale when the user leaves the
        # formulation term unset.
        gamma_div_eff = 1.0 / darcy_ref
    return {
        "gamma_div": float(gamma_div_eff),
        "darcy_ref": float(darcy_ref),
    }


def _condition_balanced_solid_cutoff_y(
    *,
    mechanics_nondim_mode: str,
    y_interface: float,
    solid_dof_y_cut: float | None,
    condition_balanced_solid_cut_fix: bool,
) -> float | None:
    key = str(mechanics_nondim_mode).strip().lower()
    if key != "condition_balanced" or not bool(condition_balanced_solid_cut_fix):
        return None
    if solid_dof_y_cut is not None:
        return float(solid_dof_y_cut)
    return None


def _function_global_values(func, *, total_dofs: int) -> np.ndarray:
    values = np.zeros((int(total_dofs),), dtype=float)
    if func is None:
        return values
    g = np.asarray(getattr(func, "_g_dofs", np.array([], dtype=int)), dtype=int).ravel()
    nodal = np.asarray(getattr(func, "nodal_values", np.array([], dtype=float)), dtype=float).ravel()
    if g.size != nodal.size:
        raise ValueError(
            f"Function '{getattr(func, 'name', '<unnamed>')}' has {int(nodal.size)} nodal values but "
            f"{int(g.size)} global DOF ids."
        )
    if g.size:
        values[g] = nodal
    return values


def _inactive_solid_alpha_phase(
    problem: dict[str, object],
    *,
    reference_y: float,
    alpha_state_key: str = "alpha_k",
) -> str:
    cached = str(problem.get("_inactive_solid_alpha_phase", "") or "").strip().lower()
    if cached in {"high", "low"}:
        return cached
    alpha_func = problem.get(alpha_state_key) or problem.get("alpha_k") or problem.get("alpha_n")
    if alpha_func is None:
        return "high"
    dh = problem["dh"]
    xy = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    alpha_vals = np.asarray(getattr(alpha_func, "nodal_values", np.array([], dtype=float)), dtype=float).ravel()
    if xy.ndim != 2 or xy.shape[0] != alpha_vals.size or xy.shape[1] < 2:
        return "high"
    tol = 1.0e-12 * max(1.0, abs(float(reference_y)))
    probe = np.asarray(alpha_vals[xy[:, 1] > float(reference_y) + tol], dtype=float)
    if probe.size == 0:
        probe = np.asarray(alpha_vals[xy[:, 1] >= float(reference_y) - tol], dtype=float)
    if probe.size == 0:
        probe = np.asarray(alpha_vals, dtype=float)
    phase = "high" if float(np.mean(probe)) >= 0.5 else "low"
    problem["_inactive_solid_alpha_phase"] = phase
    return phase


def _inactive_solid_element_mask_from_alpha(
    problem: dict[str, object],
    *,
    alpha_state_key: str = "alpha_k",
    reference_y: float,
    band_halfwidth: float,
) -> tuple[np.ndarray, str, float, float]:
    dh = problem["dh"]
    mesh = problem["mesh"]
    n_elem = int(getattr(mesh, "n_elements", len(getattr(mesh, "elements_list", []))))
    elem_mask = np.zeros((n_elem,), dtype=bool)
    alpha_func = problem.get(alpha_state_key) or problem.get("alpha_k") or problem.get("alpha_n")
    if alpha_func is None or "alpha" not in getattr(dh, "field_names", ()):
        return elem_mask, "high", 0.0, 1.0
    alpha_global = _function_global_values(alpha_func, total_dofs=int(dh.total_dofs))
    phase = _inactive_solid_alpha_phase(
        problem,
        reference_y=float(reference_y),
        alpha_state_key=alpha_state_key,
    )
    halfwidth = min(max(float(band_halfwidth), 0.0), 0.49)
    lo_thr = 0.5 - halfwidth
    hi_thr = 0.5 + halfwidth
    elem_maps = list(getattr(dh, "element_maps", {}).get("alpha", []) or [])
    for eid, gds in enumerate(elem_maps):
        if eid >= n_elem:
            break
        g_arr = np.asarray(gds, dtype=int).ravel()
        if g_arr.size == 0:
            continue
        nodal = np.asarray(alpha_global[g_arr], dtype=float)
        if not np.all(np.isfinite(nodal)):
            continue
        if phase == "high":
            elem_mask[eid] = bool(np.all(nodal >= hi_thr))
        else:
            elem_mask[eid] = bool(np.all(nodal <= lo_thr))
    return elem_mask, phase, float(lo_thr), float(hi_thr)


def _tag_inactive_solid_dofs_outside_interface_band(
    problem: dict[str, object],
    *,
    reference_y: float | None,
    alpha_state_key: str = "alpha_k",
    band_halfwidth: float | None = None,
) -> dict[str, int]:
    if reference_y is None:
        return {}
    ref_y = float(reference_y)
    if not np.isfinite(ref_y):
        return {}
    dh = problem["dh"]
    prev_tagged = set(int(d) for d in list(problem.get("_inactive_solid_tagged_dofs", set()) or set()))
    inactive_base = set(int(d) for d in list(getattr(dh, "dof_tags", {}).get("inactive", set()) or set()))
    inactive_base.difference_update(prev_tagged)
    dh.dof_tags["inactive"] = inactive_base
    halfwidth = (
        float(problem.get("_inactive_solid_alpha_band_halfwidth", 0.25))
        if band_halfwidth is None
        else float(band_halfwidth)
    )
    elem_mask, phase, lo_thr, hi_thr = _inactive_solid_element_mask_from_alpha(
        problem,
        alpha_state_key=alpha_state_key,
        reference_y=ref_y,
        band_halfwidth=halfwidth,
    )
    counts: dict[str, int] = {}
    selected_all: set[int] = set()
    solid_fields = ["u_x", "u_y", "vS_x", "vS_y"]
    if problem.get("lambda_drag_k") is not None:
        solid_fields.extend(["lambda_drag_x", "lambda_drag_y"])
    for field in solid_fields:
        if field not in getattr(dh, "field_names", ()):
            counts[field] = 0
            continue
        selected = dh.tag_dofs_from_element_bitset("inactive", field, elem_mask, strict=True)
        selected_set = set(int(g) for g in selected)
        counts[field] = int(len(selected_set))
        selected_all.update(selected_set)
    problem["_inactive_solid_reference_y"] = float(ref_y)
    problem["_inactive_solid_alpha_phase"] = str(phase)
    problem["_inactive_solid_alpha_band_halfwidth"] = float(halfwidth)
    problem["_inactive_solid_alpha_threshold_low"] = float(lo_thr)
    problem["_inactive_solid_alpha_threshold_high"] = float(hi_thr)
    problem["_inactive_solid_element_count"] = int(np.count_nonzero(elem_mask))
    problem["_inactive_solid_counts"] = dict(counts)
    problem["_inactive_solid_tagged_dofs"] = set(selected_all)
    return counts


def _tag_inactive_solid_dofs_above_y(problem: dict[str, object], *, y_cut: float | None) -> dict[str, int]:
    if y_cut is None:
        return {}
    y_cut_val = float(y_cut)
    if not np.isfinite(y_cut_val):
        return {}
    dh = problem["dh"]
    inactive = set(int(d) for d in list(getattr(dh, "dof_tags", {}).get("inactive", set()) or set()))
    tol = 1.0e-12 * max(1.0, abs(y_cut_val))
    counts: dict[str, int] = {}
    tagged: set[int] = set()
    solid_fields = ["u_x", "u_y", "vS_x", "vS_y"]
    if problem.get("lambda_drag_k") is not None:
        solid_fields.extend(["lambda_drag_x", "lambda_drag_y"])
    for field in solid_fields:
        sl = np.asarray(dh.get_field_slice(field), dtype=int)
        xy = np.asarray(dh.get_dof_coords(field), dtype=float)
        if sl.size == 0 or xy.size == 0:
            counts[field] = 0
            continue
        mask = np.asarray(xy[:, 1] > y_cut_val + tol, dtype=bool)
        selected = np.asarray(sl[mask], dtype=int)
        counts[field] = int(selected.size)
        field_selected = set(int(g) for g in selected.tolist())
        inactive.update(field_selected)
        tagged.update(field_selected)
    dh.dof_tags["inactive"] = inactive
    problem["_inactive_solid_y_cut"] = float(y_cut_val)
    problem["_inactive_solid_counts"] = dict(counts)
    problem["_inactive_solid_tagged_dofs"] = set(tagged)
    problem["_inactive_solid_alpha_phase"] = "static_y_cut"
    problem["_inactive_solid_alpha_band_halfwidth"] = float("nan")
    problem["_inactive_solid_reference_y"] = None
    return counts


def _tag_inactive_fields(problem: dict[str, object], *field_names: str) -> dict[str, int]:
    dh = problem["dh"]
    inactive = set(int(d) for d in list(getattr(dh, "dof_tags", {}).get("inactive", set()) or set()))
    counts: dict[str, int] = {}
    for field in field_names:
        name = str(field).strip()
        if not name or name not in getattr(dh, "field_names", ()):
            counts[name] = 0
            continue
        sl = np.asarray(dh.get_field_slice(name), dtype=int).ravel()
        counts[name] = int(sl.size)
        inactive.update(int(g) for g in sl.tolist())
    dh.dof_tags["inactive"] = inactive
    return counts


def _solver_requested_active_dofs(target_solver, requested_fields: tuple[str, ...] | None) -> np.ndarray:
    if requested_fields:
        full_ids: list[int] = []
        for field in requested_fields:
            full_ids.extend(np.asarray(target_solver.dh.get_field_slice(field), dtype=int).ravel().tolist())
        return np.asarray(sorted(set(int(g) for g in full_ids)), dtype=int)
    bcs_for_active = target_solver.bcs_homog if getattr(target_solver, "bcs_homog", None) else target_solver.bcs
    active_by_restr, has_restriction = analyze_active_dofs(
        target_solver.equation,
        target_solver.dh,
        target_solver.me,
        bcs_for_active,
        verbose=False,
    )
    if has_restriction:
        return np.asarray(sorted(set(int(g) for g in active_by_restr)), dtype=int)
    return np.arange(int(target_solver.dh.total_dofs), dtype=int)


def _refresh_solver_inactive_solid_interface_band(
    *,
    problem: dict[str, object],
    target_solver,
    reference_y: float | None = None,
    alpha_state_key: str = "alpha_k",
) -> dict[str, int]:
    counts = _tag_inactive_solid_dofs_outside_interface_band(
        problem,
        reference_y=(
            problem.get("_inactive_solid_reference_y", None)
            if reference_y is None
            else reference_y
        ),
        alpha_state_key=alpha_state_key,
    )
    signature = tuple(sorted(int(d) for d in list(problem.get("_inactive_solid_tagged_dofs", set()) or set())))
    if signature != getattr(target_solver, "_benchmark7_inactive_solid_signature", None):
        requested_fields = tuple(
            str(name)
            for name in tuple(getattr(target_solver, "_benchmark7_requested_active_fields", tuple()) or tuple())
            if str(name).strip()
        ) or None
        target_solver.set_active_dofs(_solver_requested_active_dofs(target_solver, requested_fields))
        target_solver._benchmark7_inactive_solid_signature = signature
    return counts


def _set_solver_active_fields_with_tracking(target_solver, active_fields) -> np.ndarray:
    fields = tuple(str(name) for name in list(active_fields or []) if str(name).strip())
    target_solver._benchmark7_requested_active_fields = fields
    return target_solver.set_active_fields(fields)


def _bind_solver_inactive_solid_interface_retagging(
    *,
    problem: dict[str, object],
    target_solver,
) -> None:
    reference_y = problem.get("_inactive_solid_reference_y", None)
    target_solver._benchmark7_requested_active_fields = tuple(
        str(name)
        for name in tuple(getattr(target_solver, "_benchmark7_requested_active_fields", tuple()) or tuple())
        if str(name).strip()
    )
    target_solver._benchmark7_inactive_solid_signature = tuple(
        sorted(int(d) for d in list(problem.get("_inactive_solid_tagged_dofs", set()) or set()))
    )
    if reference_y is None:
        return
    base_pre_cb = getattr(target_solver, "pre_cb", None)

    def _wrapped_pre_cb(funcs) -> None:
        if callable(base_pre_cb):
            base_pre_cb(funcs)
        _refresh_solver_inactive_solid_interface_band(
            problem=problem,
            target_solver=target_solver,
            reference_y=reference_y,
            alpha_state_key="alpha_k",
        )

    target_solver.pre_cb = _wrapped_pre_cb


def _full_field_scale_vector(dof_handler: DofHandler, field_scales: dict[str, float] | None) -> np.ndarray:
    scale = np.ones((int(dof_handler.total_dofs),), dtype=float)
    for fld, raw_val in dict(field_scales or {}).items():
        name = str(fld).strip()
        if not name:
            continue
        try:
            val = float(raw_val)
        except Exception:
            continue
        if not np.isfinite(val) or val <= 0.0:
            continue
        try:
            sl = np.asarray(dof_handler.get_field_slice(name), dtype=int)
        except Exception:
            continue
        if sl.size:
            scale[sl] = val
    return np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)


def _reduced_field_scale_vector(problem: dict[str, object], field_scales: dict[str, float] | None) -> np.ndarray:
    full = _full_field_scale_vector(problem["dh"], field_scales)
    active = np.asarray(problem.get("_reduced_active_dofs", ()), dtype=int)
    if active.size == 0:
        active = np.asarray(problem["dh"].free_dofs, dtype=int)
    return np.asarray(full[active], dtype=float).ravel()


def _uses_extrapolative_predictor(predictor: str) -> bool:
    return str(predictor).strip().lower() in {"delta", "increment", "extrap", "extrapolate"}


def _should_use_staggered_predictor_after_large_step(
    *,
    step_no: int,
    last_step_no: int | None,
    last_step_delta_inf: float | None,
    threshold: float,
) -> bool:
    if int(step_no) <= 1:
        return False
    if last_step_no is None or int(last_step_no) != int(step_no) - 1:
        return False
    if threshold <= 0.0:
        return False
    if last_step_delta_inf is None:
        return False
    try:
        delta_val = float(last_step_delta_inf)
    except Exception:
        return False
    if not np.isfinite(delta_val):
        return False
    return delta_val >= float(threshold)


def _should_use_frozen_transport_restart(
    *,
    enable_phi_evolution: bool,
    step_no: int,
    startup_guess_applied_step_no: int | None,
    metrics: dict[str, float] | None,
    max_delta_active: int,
    max_gap: float,
    max_eq: float,
    min_ginf: float,
) -> bool:
    if not bool(enable_phi_evolution):
        return False
    if int(step_no) <= 0:
        return False
    if int(step_no) == 1 and startup_guess_applied_step_no != int(step_no):
        return False
    data = dict(metrics or {})
    if not data:
        return False
    try:
        ginf = float(data.get("G_inf", float("nan")))
        gap = float(data.get("active_gap_inf", float("nan")))
        eq = float(data.get("equality_inf", float("nan")))
        delta_active = int(round(float(data.get("delta_active", float("inf")))))
    except Exception:
        return False
    if not (np.isfinite(ginf) and np.isfinite(gap) and np.isfinite(eq)):
        return False
    if ginf < float(min_ginf):
        return False
    if gap > float(max_gap):
        return False
    if eq > float(max_eq):
        return False
    if abs(int(delta_active)) > int(max_delta_active):
        return False
    return True


def _benchmark7_requires_constrained_solver(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "latent_bounded_transport", False)):
        return False
    if bool(getattr(args, "logistic_bounded_transform", False)):
        return False
    alpha_bc_mode_key = str(getattr(args, "alpha_bc_mode", "natural")).strip().lower()
    if alpha_bc_mode_key == "auto":
        alpha_bc_mode_key = "natural"
    if (
        not bool(getattr(args, "alpha_from_refmap", False))
        and bool(getattr(args, "alpha_box_constraints", False))
        and alpha_bc_mode_key == "natural"
    ):
        return True
    if bool(getattr(args, "enable_phi_evolution", False)) and bool(getattr(args, "phi_box_constraints", False)):
        return True
    return False


def _normalize_benchmark7_solver_choice(args: argparse.Namespace) -> argparse.Namespace:
    solver_key = str(getattr(args, "nonlinear_solver", "pdas")).strip().lower()
    if bool(getattr(args, "alpha_from_refmap", False)):
        if bool(getattr(args, "alpha_mass_constraint", False)):
            print(
                "[info] disabling the exact alpha-mass equality because --alpha-from-refmap does not solve alpha as an independent unknown."
            )
            args.alpha_mass_constraint = False
    if bool(getattr(args, "logistic_bounded_transform", False)):
        requested_logistic_fields = _parse_csv_fields(getattr(args, "logistic_bounded_fields", ""))
        effective_logistic_fields = _effective_logistic_bounded_fields(args)
        dropped_logistic_fields = tuple(
            name for name in requested_logistic_fields if str(name).strip() not in set(effective_logistic_fields)
        )
        if dropped_logistic_fields:
            print(
                "[info] filtering solver-coordinate logistic fields "
                f"{dropped_logistic_fields}; effective bounded Newton fields are {effective_logistic_fields or ('<none>',)}."
            )
        args.logistic_bounded_fields = ",".join(effective_logistic_fields)
        if not effective_logistic_fields:
            print(
                "[info] disabling solver-coordinate logistic transform because no active bounded fields remain "
                "after the benchmark-local filtering."
            )
            args.logistic_bounded_transform = False
    if _predictor_corrector_startup_enabled(args) and not bool(getattr(args, "startup_bootstrap", False)):
        print(
            "[info] enabling startup bootstrap because --predictor-corrector-startup uses the startup initial-guess hook."
        )
        args.startup_bootstrap = True
    if bool(getattr(args, "latent_bounded_transport", False)):
        effective_latent_fields = _effective_latent_bounded_fields(
            args,
            enable_phi_evolution=bool(getattr(args, "enable_phi_evolution", False)),
        )
        requested_latent_fields = _parse_csv_fields(getattr(args, "latent_bounded_fields", ""))
        dropped_latent_fields = tuple(
            name for name in requested_latent_fields if str(name).strip() not in set(effective_latent_fields)
        )
        if dropped_latent_fields:
            print(
                "[info] filtering latent bounded fields "
                f"{dropped_latent_fields}; effective latent fields are {effective_latent_fields or ('<none>',)}."
            )
        args.latent_bounded_fields = ",".join(effective_latent_fields)
        if not effective_latent_fields:
            print(
                "[info] disabling latent bounded transport because no active latent fields remain "
                "after the benchmark-local filtering."
            )
            args.latent_bounded_transport = False
            solver_key = str(getattr(args, "nonlinear_solver", "pdas")).strip().lower()
        else:
            if solver_key != "newton":
                print(
                    "[info] forcing nonlinear-solver=newton because --latent-bounded-transport "
                    "uses the unconstrained monolithic Newton path."
                )
                args.nonlinear_solver = "newton"
            if (
                bool(getattr(args, "startup_bootstrap", False))
                and not bool(getattr(args, "latent_block_preconditioner", False))
                and not _predictor_corrector_startup_enabled(args)
            ):
                print(
                    "[info] disabling startup bootstrap for the latent bounded-transport experiment so step 1 "
                    "starts from the raw monolithic initial guess."
                )
                args.startup_bootstrap = False
            if bool(getattr(args, "stall_frozen_transport_restart", False)) and not bool(getattr(args, "latent_block_preconditioner", False)):
                print(
                    "[info] disabling frozen-transport restart for the latent bounded-transport experiment."
                )
                args.stall_frozen_transport_restart = False
            globalization = str(getattr(args, "newton_globalization", "line_search") or "line_search").strip().lower()
            if _predictor_corrector_startup_enabled(args) and globalization == "line_search":
                print("[info] using trust-region globalization for the predictor-corrector exact solves.")
                args.newton_globalization = "trust_region"
            return args
    if bool(getattr(args, "latent_bounded_transport", False)):
        if bool(getattr(args, "logistic_bounded_transform", False)):
            print(
                "[info] disabling solver-coordinate logistic transform because --latent-bounded-transport "
                "moves the bounded mapping into the formulation itself."
            )
            args.logistic_bounded_transform = False
    if bool(getattr(args, "logistic_bounded_transform", False)):
        if solver_key != "newton":
            print(
                "[info] forcing nonlinear-solver=newton because --logistic-bounded-transform "
                "uses the unconstrained monolithic Newton path."
            )
            args.nonlinear_solver = "newton"
        if bool(getattr(args, "startup_bootstrap", False)):
            if _logistic_refmap_phi_only_mode(args):
                print(
                    "[info] keeping startup bootstrap enabled for the logistic-transform experiment because "
                    "--alpha-from-refmap leaves only phi in solver coordinates."
                )
                probe_budget = max(0, int(getattr(args, "pc_exact_probe_max_it", 1) or 0))
                if _predictor_corrector_startup_enabled(args) and probe_budget < 2:
                    print(
                        "[info] raising pc-exact-probe-max-it to 2 so the kept G-anchor state gets a "
                        "meaningful exact monolithic probe before the frozen-transport predictor."
                    )
                    args.pc_exact_probe_max_it = 2
                globalization = str(getattr(args, "newton_globalization", "line_search") or "line_search").strip().lower()
                if _predictor_corrector_startup_enabled(args) and globalization == "line_search":
                    print(
                        "[info] using line-search-then-trust globalization for the refmap phi-only logistic startup branch "
                        "so the exact corrector can keep a good Armijo step but still compare against trust region "
                        "when line search only finds a best-effort decrease."
                    )
                    args.newton_globalization = "line_search_then_trust"
            else:
                print(
                    "[info] disabling startup bootstrap for the logistic-transform experiment so step 1 "
                    "starts from the raw monolithic initial guess."
                )
                args.startup_bootstrap = False
        if bool(getattr(args, "stall_frozen_transport_restart", False)):
            print(
                "[info] disabling frozen-transport restart for the logistic-transform experiment."
            )
            args.stall_frozen_transport_restart = False
        return args
    if solver_key == "newton" and _benchmark7_requires_constrained_solver(args):
        print(
            "[info] promoting nonlinear-solver=newton to pdas because the Benchmark 7 configuration "
            "requests constrained alpha/phi transport. The unconstrained Newton path does not enforce "
            "the alpha/phi box bounds or the alpha-mass projection."
        )
        args.nonlinear_solver = "pdas"
    return args


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


def _as_bc_time(fn):
    def _wrapped(x, y, t):
        value = np.asarray(fn(np.asarray(x), np.asarray(y), float(t)), dtype=float)
        if value.ndim == 0:
            return float(value.reshape(()))
        return value.copy()

    return _wrapped


def _alpha_equilibrium(y: np.ndarray, *, y_interface: float, eps_alpha: float) -> np.ndarray:
    yy = np.asarray(y, dtype=float)
    eps = max(float(eps_alpha), 1.0e-12)
    return 0.5 * (1.0 + np.tanh((yy - float(y_interface)) / (math.sqrt(2.0) * eps)))


def _latent_bounded_fields(args: argparse.Namespace, *, enable_phi_evolution: bool) -> tuple[str, ...]:
    return _effective_latent_bounded_fields(args, enable_phi_evolution=enable_phi_evolution)


def _latent_bounded_map_key(obj) -> str:
    raw = getattr(obj, "latent_bounded_map", None) if not isinstance(obj, dict) else obj.get("latent_bounded_map", None)
    return str(raw or "sigmoid").strip().lower()


def _latent_bounded_formulation_key(obj) -> str:
    raw = getattr(obj, "latent_bounded_formulation", None) if not isinstance(obj, dict) else obj.get("latent_bounded_formulation", None)
    return str(raw or "embedded").strip().lower()


def _latent_map_expr(z, *, map_kind: str):
    key = str(map_kind).strip().lower()
    one = Constant(1.0)
    if key == "algebraic":
        return Constant(0.5) * (one + z / ((one + z * z) ** Constant(0.5)))
    if key == "tanh":
        return Constant(0.5) * (one + tanh(z))
    if key != "sigmoid":
        raise ValueError(f"Unsupported latent bounded map '{map_kind}'.")
    exp_neg = exp(-z)
    return one / (one + exp_neg)


def _latent_map_prime_expr(z, *, map_kind: str):
    key = str(map_kind).strip().lower()
    if key == "algebraic":
        return Constant(0.5) * ((Constant(1.0) + z * z) ** Constant(-1.5))
    if key == "tanh":
        th = tanh(z)
        return Constant(0.5) * (Constant(1.0) - th * th)
    sig = _latent_map_expr(z, map_kind=key)
    return sig * (Constant(1.0) - sig)


def _latent_inverse_array(values, *, eps: float, map_kind: str) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), float(eps), 1.0 - float(eps))
    key = str(map_kind).strip().lower()
    if key == "algebraic":
        numer = (2.0 * clipped) - 1.0
        denom = 2.0 * np.sqrt(np.maximum(clipped * (1.0 - clipped), float(np.finfo(float).tiny)))
        return numer / denom
    if key == "tanh":
        return np.arctanh((2.0 * clipped) - 1.0)
    if key != "sigmoid":
        raise ValueError(f"Unsupported latent bounded map '{map_kind}'.")
    return np.log(clipped) - np.log1p(-clipped)


def _latent_forward_array(values, *, map_kind: str) -> np.ndarray:
    zz = np.asarray(values, dtype=float)
    out = np.empty_like(zz, dtype=float)
    key = str(map_kind).strip().lower()
    if key == "algebraic":
        out[:] = 0.5 * (1.0 + (zz / np.sqrt(1.0 + zz * zz)))
        return out
    if key == "tanh":
        out[:] = 0.5 * (1.0 + np.tanh(zz))
        return out
    if key != "sigmoid":
        raise ValueError(f"Unsupported latent bounded map '{map_kind}'.")
    pos = zz >= 0.0
    if np.any(pos):
        out[pos] = 1.0 / (1.0 + np.exp(-zz[pos]))
    if np.any(~pos):
        ez = np.exp(zz[~pos])
        out[~pos] = ez / (1.0 + ez)
    return out


def _latent_mass_return_shift(
    latent_values,
    *,
    local_weights,
    target_mass: float,
    map_kind: str,
    free_mask=None,
    tol_rel: float = 1.0e-10,
    max_it: int = 64,
    bracket_growth: float = 2.0,
    max_bracket_it: int = 32,
) -> dict[str, object]:
    z0 = np.asarray(latent_values, dtype=float).ravel()
    weights = np.asarray(local_weights, dtype=float).ravel()
    if z0.size != weights.size:
        raise ValueError(
            f"latent mass return shift size mismatch: latent has {int(z0.size)} entries, "
            f"weights have {int(weights.size)}."
        )
    if free_mask is None:
        free = np.ones(z0.size, dtype=bool)
    else:
        free = np.asarray(free_mask, dtype=bool).ravel()
        if free.size != z0.size:
            raise ValueError(
                f"latent mass return shift free-mask size mismatch: mask has {int(free.size)} entries, "
                f"latent has {int(z0.size)}."
            )
    target = float(target_mass)
    tol_abs = max(float(tol_rel), 0.0) * max(abs(target), 1.0)
    growth = max(float(bracket_growth), 1.01)
    max_bracket = max(1, int(max_bracket_it))
    max_iter = max(1, int(max_it))

    best_shift = 0.0
    best_mass = float(weights @ _latent_forward_array(z0, map_kind=map_kind))
    best_defect = best_mass - target
    best_values = z0.copy()

    def _evaluate(shift: float):
        nonlocal best_shift, best_mass, best_defect, best_values
        z = z0.copy()
        if np.any(free):
            z[free] = z0[free] + float(shift)
        mass = float(weights @ _latent_forward_array(z, map_kind=map_kind))
        defect = mass - target
        if abs(defect) < abs(best_defect):
            best_shift = float(shift)
            best_mass = float(mass)
            best_defect = float(defect)
            best_values = z.copy()
        return float(mass), float(defect), z

    mass0, defect0, z_at_zero = _evaluate(0.0)
    if (not np.any(free)) or abs(defect0) <= tol_abs:
        return {
            "shift": 0.0,
            "mass": float(mass0),
            "target_mass": float(target),
            "defect": float(defect0),
            "relative_defect": float(defect0 / max(abs(target), 1.0e-30)),
            "iterations": 0,
            "bracketed": True,
            "converged": bool(abs(defect0) <= tol_abs),
            "latent_values": z_at_zero.copy(),
        }

    bracketed = False
    lo_shift = 0.0
    hi_shift = 0.0
    lo_defect = defect0
    hi_defect = defect0
    bracket_iters = 0
    trial_shift = 1.0 if defect0 < 0.0 else -1.0
    while bracket_iters < max_bracket:
        _, trial_defect, _ = _evaluate(trial_shift)
        bracket_iters += 1
        if defect0 < 0.0:
            lo_shift, lo_defect = 0.0, defect0
            hi_shift, hi_defect = float(trial_shift), float(trial_defect)
            if hi_defect >= 0.0:
                bracketed = True
                break
        else:
            lo_shift, lo_defect = float(trial_shift), float(trial_defect)
            hi_shift, hi_defect = 0.0, defect0
            if lo_defect <= 0.0:
                bracketed = True
                break
        trial_shift *= growth

    if bracketed:
        for it in range(max_iter):
            mid_shift = 0.5 * (lo_shift + hi_shift)
            _, mid_defect, mid_values = _evaluate(mid_shift)
            if abs(mid_defect) <= tol_abs:
                return {
                    "shift": float(mid_shift),
                    "mass": float(target + mid_defect),
                    "target_mass": float(target),
                    "defect": float(mid_defect),
                    "relative_defect": float(mid_defect / max(abs(target), 1.0e-30)),
                    "iterations": int(bracket_iters + it + 1),
                    "bracketed": True,
                    "converged": True,
                    "latent_values": mid_values.copy(),
                }
            if mid_defect < 0.0:
                lo_shift, lo_defect = float(mid_shift), float(mid_defect)
            else:
                hi_shift, hi_defect = float(mid_shift), float(mid_defect)

    return {
        "shift": float(best_shift),
        "mass": float(best_mass),
        "target_mass": float(target),
        "defect": float(best_defect),
        "relative_defect": float(best_defect / max(abs(target), 1.0e-30)),
        "iterations": int(bracket_iters + (max_iter if bracketed else 0)),
        "bracketed": bool(bracketed),
        "converged": bool(abs(best_defect) <= tol_abs),
        "latent_values": best_values.copy(),
    }


def _latent_inverse_value_callback(value_cb, *, eps: float, map_kind: str):
    def _wrapped(x, y, t):
        raw = float(value_cb(x, y, t))
        return float(_latent_inverse_array(np.asarray([raw], dtype=float), eps=float(eps), map_kind=str(map_kind))[0])

    return _wrapped


def _sync_latent_bounded_problem_fields(
    *,
    problem: dict[str, object],
    funcs=None,
    find_named_function=None,
) -> None:
    latent_fields = tuple(problem.get("latent_bounded_fields", tuple()) or tuple())
    if not latent_fields:
        return
    map_kind = _latent_bounded_map_key(problem)

    def _pick(template):
        if funcs is None or find_named_function is None:
            return template
        return find_named_function(funcs, template)

    for base in ("alpha", "phi"):
        if base not in latent_fields:
            continue
        cur = problem.get(f"{base}_k")
        lat = problem.get(f"{base}_latent_k")
        prev = problem.get(f"{base}_n")
        lat_prev = problem.get(f"{base}_latent_n")
        if cur is not None and lat is not None:
            cur_obj = _pick(cur)
            lat_obj = _pick(lat)
            cur_obj.nodal_values[:] = _latent_forward_array(lat_obj.nodal_values, map_kind=map_kind)
        if prev is not None and lat_prev is not None:
            prev_obj = _pick(prev)
            lat_prev_obj = _pick(lat_prev)
            prev_obj.nodal_values[:] = _latent_forward_array(lat_prev_obj.nodal_values, map_kind=map_kind)


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


def _build_transport_measures(
    *,
    problem: dict[str, object],
    qdeg: int,
    enable_phi_evolution: bool,
    top_drainage_transport: bool,
    support_physics: str,
):
    if not _full_top_drainage_transport_enabled(
        enable_phi_evolution=bool(enable_phi_evolution),
        top_drainage_transport=bool(top_drainage_transport),
    ):
        return None, None
    support_key = str(support_physics).strip().lower()
    if support_key == "internal_conversion":
        # In the conserved-support model, top drainage releases pore fluid
        # through the fluid pressure/outflow closure only. The support balance
        # for alpha and the conservative solid-volume balance for B remain
        # closed at the outer boundary, so we must not open alpha/B transport
        # fluxes here.
        return None, None
    ds_top = dS(defined_on=problem["mesh"].edge_bitset("top"), metadata={"q": int(qdeg)})
    return ds_top, ds_top


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
    latent_bounded_transport: bool = False,
    latent_bounded_fields: tuple[str, ...] | None = None,
    latent_bounded_map: str = "sigmoid",
    latent_bounded_formulation: str = "embedded",
    alpha_mass_constraint: bool = False,
    pressure_mean_constraint: bool = False,
    solid_volumetric_split: bool = False,
    drag_formulation: str = "direct",
) -> dict[str, object]:
    fluid_space_key = str(fluid_space).strip().lower()
    if fluid_space_key not in {"cg", "hdiv"}:
        raise ValueError(f"Unsupported fluid_space={fluid_space!r}.")
    latent_field_set = {
        str(name).strip()
        for name in (
            tuple(latent_bounded_fields)
            if latent_bounded_fields is not None
            else (("alpha", "phi") if bool(latent_bounded_transport) else tuple())
        )
        if str(name).strip()
    }

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
        **({"p_mean": ":number:"} if bool(pressure_mean_constraint) else {}),
        **({"alpha_mass_lm": ":number:"} if bool(alpha_mass_constraint) and not bool(latent_bounded_transport) else {}),
        **({"pi_s": int(pressure_order)} if bool(solid_volumetric_split) else {}),
        "vS_x": int(poly_order),
        "vS_y": int(poly_order),
        "u_x": int(poly_order),
        "u_y": int(poly_order),
        **(
            {"lambda_drag_x": int(poly_order), "lambda_drag_y": int(poly_order)}
            if str(drag_formulation).strip().lower().replace("-", "_") == "mixed_lm"
            else {}
        ),
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
    if bool(latent_bounded_transport):
        if "alpha" in latent_field_set:
            field_specs["alpha_latent"] = int(scalar_order)
        if bool(enable_phi_evolution) and "phi" in latent_field_set:
            field_specs["phi_latent"] = int(scalar_order)
    me = MixedElement(mesh, field_specs=field_specs)
    dh = DofHandler(me, method="cg")

    VS = FunctionSpace("VS", ["vS_x", "vS_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)
    lambda_drag_space = None
    if "lambda_drag_x" in field_specs:
        lambda_drag_space = FunctionSpace("LambdaDrag", ["lambda_drag_x", "lambda_drag_y"], dim=1)

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
        "dlambda_drag": (None if lambda_drag_space is None else VectorTrialFunction(space=lambda_drag_space, dof_handler=dh)),
        "dp": TrialFunction("p", dof_handler=dh),
        "dp_mean": (TrialFunction("p_mean", dof_handler=dh) if bool(pressure_mean_constraint) else None),
        "dalpha_mass_lm": (TrialFunction("alpha_mass_lm", dof_handler=dh) if "alpha_mass_lm" in field_specs else None),
        "dpi_s": (TrialFunction("pi_s", dof_handler=dh) if bool(solid_volumetric_split) else None),
        "dalpha": TrialFunction("alpha", dof_handler=dh),
        "dmu": TrialFunction("mu_alpha", dof_handler=dh),
        "v_test": v_test,
        "vS_test": VectorTestFunction(space=VS, dof_handler=dh),
        "u_test": VectorTestFunction(space=U, dof_handler=dh),
        "lambda_drag_test": (None if lambda_drag_space is None else VectorTestFunction(space=lambda_drag_space, dof_handler=dh)),
        "q_test": TestFunction("p", dof_handler=dh),
        "p_mean_test": (TestFunction("p_mean", dof_handler=dh) if bool(pressure_mean_constraint) else None),
        "alpha_mass_lm_test": (TestFunction("alpha_mass_lm", dof_handler=dh) if "alpha_mass_lm" in field_specs else None),
        "pi_s_test": (TestFunction("pi_s", dof_handler=dh) if bool(solid_volumetric_split) else None),
        "alpha_test": TestFunction("alpha", dof_handler=dh),
        "mu_test": TestFunction("mu_alpha", dof_handler=dh),
        "v_k": v_k,
        "p_k": Function("p_k", "p", dof_handler=dh),
        "p_mean_k": (Function("p_mean_k", "p_mean", dof_handler=dh) if bool(pressure_mean_constraint) else None),
        "alpha_mass_lm_k": (Function("alpha_mass_lm_k", "alpha_mass_lm", dof_handler=dh) if "alpha_mass_lm" in field_specs else None),
        "pi_s_k": (Function("pi_s_k", "pi_s", dof_handler=dh) if bool(solid_volumetric_split) else None),
        "vS_k": VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh),
        "u_k": VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh),
        "lambda_drag_k": (
            None
            if lambda_drag_space is None
            else VectorFunction("lambda_drag_k", ["lambda_drag_x", "lambda_drag_y"], dof_handler=dh)
        ),
        "alpha_k": Function("alpha_k", "alpha", dof_handler=dh),
        "mu_k": Function("mu_k", "mu_alpha", dof_handler=dh),
        "v_n": v_n,
        "p_n": Function("p_n", "p", dof_handler=dh),
        "p_mean_n": (Function("p_mean_n", "p_mean", dof_handler=dh) if bool(pressure_mean_constraint) else None),
        "alpha_mass_lm_n": (Function("alpha_mass_lm_n", "alpha_mass_lm", dof_handler=dh) if "alpha_mass_lm" in field_specs else None),
        "pi_s_n": (Function("pi_s_n", "pi_s", dof_handler=dh) if bool(solid_volumetric_split) else None),
        "vS_n": VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh),
        "u_n": VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh),
        "lambda_drag_n": (
            None
            if lambda_drag_space is None
            else VectorFunction("lambda_drag_n", ["lambda_drag_x", "lambda_drag_y"], dof_handler=dh)
        ),
        "alpha_n": Function("alpha_n", "alpha", dof_handler=dh),
        "mu_n": Function("mu_n", "mu_alpha", dof_handler=dh),
        "latent_bounded_transport": bool(latent_bounded_transport),
        "latent_bounded_fields": tuple(latent_field_set),
        "latent_bounded_map": str(latent_bounded_map),
        "latent_bounded_formulation": str(latent_bounded_formulation),
        "alpha_mass_constraint": bool(bool(alpha_mass_constraint) and not bool(latent_bounded_transport)),
        "pressure_mean_constraint": bool(pressure_mean_constraint),
        "solid_volumetric_split": bool(solid_volumetric_split),
        "drag_formulation": str(drag_formulation),
        "_alpha_mass_constraint_residual_form": None,
        "_alpha_mass_constraint_jacobian_form": None,
        "_pressure_mean_residual_form": None,
        "_pressure_mean_jacobian_form": None,
    }
    if bool(latent_bounded_transport) and "alpha" in latent_field_set:
        problem.update(
            {
                "dalpha_latent": TrialFunction("alpha_latent", dof_handler=dh),
                "alpha_latent_test": TestFunction("alpha_latent", dof_handler=dh),
                "alpha_latent_k": Function("alpha_latent_k", "alpha_latent", dof_handler=dh),
                "alpha_latent_n": Function("alpha_latent_n", "alpha_latent", dof_handler=dh),
            }
        )
    else:
        problem.update(
            {
                "dalpha_latent": None,
                "alpha_latent_test": None,
                "alpha_latent_k": None,
                "alpha_latent_n": None,
            }
        )
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
        if bool(latent_bounded_transport) and "phi" in latent_field_set:
            problem.update(
                {
                    "dphi_latent": TrialFunction("phi_latent", dof_handler=dh),
                    "phi_latent_test": TestFunction("phi_latent", dof_handler=dh),
                    "phi_latent_k": Function("phi_latent_k", "phi_latent", dof_handler=dh),
                    "phi_latent_n": Function("phi_latent_n", "phi_latent", dof_handler=dh),
                }
            )
        else:
            problem.update(
                {
                    "dphi_latent": None,
                    "phi_latent_test": None,
                    "phi_latent_k": None,
                    "phi_latent_n": None,
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
                "dphi_latent": None,
                "phi_latent_test": None,
                "phi_latent_k": None,
                "phi_latent_n": None,
            }
        )
    for key in ("v_k", "vS_k", "u_k", "lambda_drag_k", "v_n", "vS_n", "u_n", "lambda_drag_n"):
        if problem.get(key) is not None:
            problem[key].nodal_values[:] = 0.0
    for key in (
        "p_k",
        "p_n",
        "p_mean_k",
        "p_mean_n",
        "pi_s_k",
        "pi_s_n",
        "mu_k",
        "mu_n",
        "phi_k",
        "phi_n",
        "S_k",
        "S_n",
        "alpha_latent_k",
        "alpha_latent_n",
        "phi_latent_k",
        "phi_latent_n",
    ):
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
    mechanics_nondim_mode: str = "legacy",
    solid_volumetric_split: bool = False,
    solid_volumetric_penalty: float = 1.0,
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
    alpha_supg: float,
    alpha_cip: float,
    v_supg: float = 0.0,
    v_supg_mode: str = "streamline",
    v_supg_c_nu: float = 4.0,
    u_supg: float = 0.0,
    v_cip: float = 0.0,
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
    include_skeleton_acceleration: bool,
    rho_s0_tilde: float,
    skeleton_inertia_convection: str,
    ds_hdiv_tangential=None,
    hdiv_tangential_gamma: float = 20.0,
    hdiv_tangential_method: str = "penalty",
    fluid_convection: str = "full",
    fluid_convection_full_weight=None,
    fluid_convection_lagged_weight=None,
    fluid_convection_imex_weight=None,
    support_physics: str = "legacy_exchange",
    ds_alpha_transport=None,
    ds_B_transport=None,
    skeleton_pressure_mode: str = "whole_domain",
    alpha_biot: float | None = None,
    g_t_k=None,
    g_t_n=None,
    traction_weight_k=None,
    traction_weight_n=None,
    drag_formulation: str = "direct",
    skeleton_acceleration_weight=None,
    skeleton_inertia_full_weight=None,
    skeleton_inertia_lagged_weight=None,
):
    solid_model_key = str(solid_model).strip().lower().replace("-", "_")
    latent_fields = tuple(problem.get("latent_bounded_fields", tuple()) or tuple())
    latent_map_kind = _latent_bounded_map_key(problem)
    latent_formulation_key = _latent_bounded_formulation_key(problem)
    use_alpha_latent = (
        bool(problem.get("latent_bounded_transport", False))
        and "alpha" in latent_fields
        and problem.get("alpha_latent_k") is not None
    )
    use_phi_latent = (
        bool(problem.get("latent_bounded_transport", False))
        and "phi" in latent_fields
        and problem.get("phi_latent_k") is not None
    )
    alpha_k_eff = problem["alpha_k"]
    alpha_n_eff = problem["alpha_n"]
    dalpha_eff = problem["dalpha"]
    alpha_transport_test = problem["alpha_test"]
    if use_alpha_latent and problem.get("alpha_latent_test") is not None:
        alpha_transport_test = problem["alpha_latent_test"]
        if latent_formulation_key == "transformed":
            alpha_k_eff = _latent_map_expr(problem["alpha_latent_k"], map_kind=latent_map_kind)
            alpha_n_eff = _latent_map_expr(problem["alpha_latent_n"], map_kind=latent_map_kind)
            dalpha_eff = _latent_map_prime_expr(problem["alpha_latent_k"], map_kind=latent_map_kind) * problem["dalpha_latent"]
    phi_k_eff = problem["phi_k"]
    phi_n_eff = problem["phi_n"]
    dphi_eff = problem["dphi"]
    phi_transport_test = problem["phi_test"]
    if use_phi_latent and problem.get("phi_latent_test") is not None:
        phi_transport_test = problem["phi_latent_test"]
        if latent_formulation_key == "transformed":
            phi_k_eff = _latent_map_expr(problem["phi_latent_k"], map_kind=latent_map_kind)
            phi_n_eff = _latent_map_expr(problem["phi_latent_n"], map_kind=latent_map_kind)
            dphi_eff = _latent_map_prime_expr(problem["phi_latent_k"], map_kind=latent_map_kind) * problem["dphi_latent"]
    theta_c = _named_constant("b7_theta", float(theta))
    rho_f_c = _named_constant("b7_rho_f", float(rho_f))
    mu_f_c = _named_constant("b7_mu_f", float(mu_f))
    mu_b_c = _named_constant("b7_mu_b", float(mu_b))
    kappa_inv_c = _named_constant("b7_kappa_inv", float(kappa_inv))
    mu_s_c = _named_constant("b7_mu_s", float(mu_s))
    lambda_s_c = _named_constant("b7_lambda_s", float(lambda_s))
    rho_s0_tilde_c = _named_constant("b7_rho_s0_tilde", float(rho_s0_tilde))
    common_kwargs = {
        "dt": dt_c,
        "theta": theta_c,
        "rho_f": rho_f_c,
        "mu_f": mu_f_c,
        "mu_b": mu_b_c,
        "kappa_inv": kappa_inv_c,
        "mu_s": mu_s_c,
        "lambda_s": lambda_s_c,
        "solid_visco_eta": float(solid_visco_eta),
        "gamma_div": float(gamma_div),
        "solid_model": solid_model_key,
        "kappa_inv_model": str(kappa_inv_model),
        "drag_formulation": str(drag_formulation),
        "include_skeleton_acceleration": bool(include_skeleton_acceleration),
        "rho_s0_tilde": rho_s0_tilde_c,
        "skeleton_inertia_convection": str(skeleton_inertia_convection),
    }
    mechanics_nondim_key = str(mechanics_nondim_mode).strip().lower()
    if mechanics_nondim_key not in {"legacy", "stress_balance", "condition_balanced"}:
        raise ValueError(
            f"Unsupported mechanics_nondim_mode={mechanics_nondim_mode!r}. "
            "Use 'legacy', 'stress_balance', or 'condition_balanced'."
        )
    fluid_momentum_scale = None
    skeleton_momentum_scale = None
    kinematics_scale = None
    pressure_block_lift_scale = 0.0
    condition_balance_field_scales: dict[str, float] = {}
    dim = int(problem["mesh"].dim) if getattr(problem.get("mesh"), "dim", None) is not None else 2
    if mechanics_nondim_key in {"stress_balance", "condition_balanced"}:
        fluid_ref = max(1.0, abs(float(mu_f)), abs(float(mu_b)))
        solid_ref = max(1.0, abs(float(2.0 * mu_s + float(dim) * lambda_s)))
        fluid_momentum_scale = 1.0 / fluid_ref
        skeleton_momentum_scale = 1.0 / solid_ref
        # Stress-balance mode keeps the transport-style u equation on an O(1) row
        # scale. Condition-balanced mode may override this below with a
        # permeability-aware row scale.
        kinematics_scale = 1.0
        if mechanics_nondim_key == "condition_balanced":
            condition_balance_field_scales = _condition_balanced_field_scales(
                mechanics_nondim_mode=mechanics_nondim_key,
                drag_formulation=str(drag_formulation),
                dt=dt_c,
                mu_f=float(mu_f),
                kappa_inv=float(kappa_inv),
                mu_s=float(mu_s),
                lambda_s=float(lambda_s),
                rho_s0_tilde=float(rho_s0_tilde),
                dim=int(dim),
            )
    kinematic_setup = _condition_balanced_kinematic_setup(
        mechanics_nondim_mode=mechanics_nondim_key,
        mu_f=float(mu_f),
        kappa_inv=float(kappa_inv),
        gamma_u=float(gamma_u),
        u_extension_mode=str(u_extension_mode),
        gamma_u_pin=float(gamma_u_pin),
        gamma_vS=gamma_vS,
        vS_extension_mode=vS_extension_mode,
        gamma_vS_pin=gamma_vS_pin,
    )
    gamma_u = float(kinematic_setup["gamma_u"])
    u_extension_mode = str(kinematic_setup["u_extension_mode"])
    gamma_u_pin = float(kinematic_setup["gamma_u_pin"])
    gamma_vS = None if kinematic_setup["gamma_vS"] is None else float(kinematic_setup["gamma_vS"])
    vS_extension_mode = (
        None if kinematic_setup["vS_extension_mode"] is None else str(kinematic_setup["vS_extension_mode"])
    )
    gamma_vS_pin = None if kinematic_setup["gamma_vS_pin"] is None else float(kinematic_setup["gamma_vS_pin"])
    if mechanics_nondim_key == "condition_balanced":
        kinematics_scale = float(kinematic_setup["kinematics_scale"])
    problem["_condition_balanced_field_scales"] = dict(condition_balance_field_scales)
    one_domain_kwargs = {
        **common_kwargs,
        "gamma_u": _named_constant("b7_gamma_u", float(gamma_u)),
        "u_extension_mode": str(u_extension_mode),
        "gamma_u_pin": _named_constant("b7_gamma_u_pin", float(gamma_u_pin)),
        "fluid_convection": str(fluid_convection),
        "fluid_convection_full_weight": fluid_convection_full_weight,
        "fluid_convection_lagged_weight": fluid_convection_lagged_weight,
        "fluid_convection_imex_weight": fluid_convection_imex_weight,
        "kinematics_scale": kinematics_scale,
        "fluid_momentum_scale": fluid_momentum_scale,
        "skeleton_momentum_scale": skeleton_momentum_scale,
        "pi_s_k": problem.get("pi_s_k"),
        "pi_s_n": problem.get("pi_s_n"),
        "dpi_s": problem.get("dpi_s"),
        "pi_s_test": problem.get("pi_s_test"),
        "solid_volumetric_split": bool(solid_volumetric_split),
        "solid_volumetric_penalty": float(solid_volumetric_penalty),
        "pressure_block_lift_scale": float(pressure_block_lift_scale),
        "skeleton_acceleration_weight": skeleton_acceleration_weight,
        "skeleton_inertia_full_weight": skeleton_inertia_full_weight,
        "skeleton_inertia_lagged_weight": skeleton_inertia_lagged_weight,
    }
    if not bool(enable_phi_evolution):
        forms = build_deformation_only_forms(
            v_k=problem["v_k"],
            p_k=problem["p_k"],
            vS_k=problem["vS_k"],
            lambda_drag_k=problem.get("lambda_drag_k"),
            u_k=problem["u_k"],
            alpha_k=alpha_k_eff,
            mu_alpha_k=problem["mu_k"],
            v_n=problem["v_n"],
            p_n=problem["p_n"],
            vS_n=problem["vS_n"],
            lambda_drag_n=problem.get("lambda_drag_n"),
            u_n=problem["u_n"],
            alpha_n=alpha_n_eff,
            mu_alpha_n=problem["mu_n"],
            dv=problem["dv"],
            dp=problem["dp"],
            dpi_s=problem.get("dpi_s"),
            dvS=problem["dvS"],
            dlambda_drag=problem.get("dlambda_drag"),
            du=problem["du"],
            dalpha=dalpha_eff,
            dmu_alpha=problem["dmu"],
            v_test=problem["v_test"],
            q_test=problem["q_test"],
            pi_s_test=problem.get("pi_s_test"),
            vS_test=problem["vS_test"],
            lambda_drag_test=problem.get("lambda_drag_test"),
            u_test=problem["u_test"],
            alpha_test=alpha_transport_test,
            mu_alpha_test=problem["mu_test"],
            dx=dx(metadata={"q": int(qdeg)}),
            phi_b=_named_constant("b7_phi_b", float(phi_b)),
            M_alpha=_named_constant("b7_M_alpha", float(M_alpha)),
            gamma_alpha=_named_constant("b7_gamma_alpha", float(gamma_alpha)),
            eps_alpha=_named_constant("b7_eps_alpha", float(eps_alpha)),
            support_physics=str(support_physics),
            alpha_advect_with=str(alpha_advect_with),
            alpha_advection_form=str(alpha_advection_form),
            fluid_convection=str(fluid_convection),
            g_t_k=g_t_k,
            g_t_n=g_t_n,
            traction_weight_k=traction_weight_k,
            traction_weight_n=traction_weight_n,
            pi_s_k=problem.get("pi_s_k"),
            pi_s_n=problem.get("pi_s_n"),
            solid_volumetric_split=bool(solid_volumetric_split),
            solid_volumetric_penalty=float(solid_volumetric_penalty),
            pressure_block_lift_scale=float(pressure_block_lift_scale),
            skeleton_pressure_mode=str(skeleton_pressure_mode),
            alpha_biot=alpha_biot,
            **common_kwargs,
        )
    else:
        forms = build_biofilm_one_domain_forms(
            v_k=problem["v_k"],
            p_k=problem["p_k"],
            vS_k=problem["vS_k"],
            lambda_drag_k=problem.get("lambda_drag_k"),
            u_k=problem["u_k"],
            phi_k=phi_k_eff,
            alpha_k=alpha_k_eff,
            mu_alpha_k=problem["mu_k"],
            S_k=problem["S_k"],
            v_n=problem["v_n"],
            p_n=problem["p_n"],
            vS_n=problem["vS_n"],
            lambda_drag_n=problem.get("lambda_drag_n"),
            u_n=problem["u_n"],
            phi_n=phi_n_eff,
            alpha_n=alpha_n_eff,
            mu_alpha_n=problem["mu_n"],
            S_n=problem["S_n"],
            dv=problem["dv"],
            dp=problem["dp"],
            dvS=problem["dvS"],
            dlambda_drag=problem.get("dlambda_drag"),
            du=problem["du"],
            dphi=dphi_eff,
            dalpha=dalpha_eff,
            dmu_alpha=problem["dmu"],
            dS=problem["dS"],
            v_test=problem["v_test"],
            q_test=problem["q_test"],
            vS_test=problem["vS_test"],
            lambda_drag_test=problem.get("lambda_drag_test"),
            u_test=problem["u_test"],
            phi_test=phi_transport_test,
            alpha_test=alpha_transport_test,
            mu_alpha_test=problem["mu_test"],
            S_test=problem["S_test"],
            dx=dx(metadata={"q": int(qdeg)}),
            ds_cip=ds(metadata={"q": int(qdeg)}),
            mu_b_model=str(mu_b_model),
            D_phi=_named_constant("b7_D_phi", float(D_phi)),
            phi_diffusion_weight=str(phi_diffusion_weight),
            gamma_phi=_named_constant("b7_gamma_phi", float(gamma_phi)),
            phi_supg=float(phi_supg),
            phi_cip=float(phi_cip),
            alpha_supg=float(alpha_supg),
            alpha_cip=float(alpha_cip),
            v_supg=float(v_supg),
            v_supg_mode=str(v_supg_mode),
            v_supg_c_nu=float(v_supg_c_nu),
            u_supg=float(u_supg),
            v_cip=float(v_cip),
            regularization_weight=problem.get("reg_weight"),
            u_cip=float(u_cip),
            u_cip_weight=str(u_cip_weight),
            vS_cip=float(vS_cip),
            gamma_vS=(
                None
                if gamma_vS is None
                else _named_constant("b7_gamma_vS", float(gamma_vS))
            ),
            vS_extension_mode=vS_extension_mode,
            gamma_vS_pin=(
                None
                if gamma_vS_pin is None
                else _named_constant("b7_gamma_vS_pin", float(gamma_vS_pin))
            ),
            ds_hdiv_tangential=ds_hdiv_tangential,
            hdiv_tangential_gamma=float(hdiv_tangential_gamma),
            hdiv_tangential_method=str(hdiv_tangential_method),
            D_alpha=0.0,
            alpha_interface_reg=(
                "olsson_nt"
                if str(alpha_regularization).strip().lower() == "olsson_nt"
                else "none"
            ),
            alpha_interface_reg_gamma=_named_constant("b7_alpha_reg_gamma", float(alpha_reg_gamma)),
            alpha_interface_reg_eps_normal=_named_constant("b7_alpha_reg_eps_normal", float(alpha_reg_eps_normal)),
            alpha_interface_reg_eps_tangent=_named_constant("b7_alpha_reg_eps_tangent", float(alpha_reg_eps_tangent)),
            alpha_interface_reg_eta=_named_constant("b7_alpha_reg_eta", float(alpha_reg_eta)),
            alpha_mu_aux_pin=1.0,
            alpha_advect_with=str(alpha_advect_with),
            alpha_advection_form=str(alpha_advection_form),
            support_physics=str(support_physics),
            ds_alpha_transport=ds_alpha_transport,
            ds_B_transport=ds_B_transport,
            g_t_k=g_t_k,
            g_t_n=g_t_n,
            traction_weight_k=traction_weight_k,
            traction_weight_n=traction_weight_n,
            skeleton_pressure_mode=str(skeleton_pressure_mode),
            alpha_biot=alpha_biot,
            alpha_ch_M=_named_constant(
                "b7_alpha_ch_M",
                float(M_alpha if str(alpha_regularization).strip().lower() == "ch" else 0.0),
            ),
            alpha_ch_gamma=_named_constant(
                "b7_alpha_ch_gamma",
                float(gamma_alpha if str(alpha_regularization).strip().lower() == "ch" else 0.0),
            ),
            alpha_ch_eps=_named_constant("b7_alpha_ch_eps", float(eps_alpha)),
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

    residual_form = forms.residual_form
    jacobian_form = forms.jacobian_form
    dx_q = dx(metadata={"q": int(qdeg)})
    problem["_alpha_mass_constraint_residual_form"] = None
    problem["_alpha_mass_constraint_jacobian_form"] = None
    problem["_pressure_mean_residual_form"] = None
    problem["_pressure_mean_jacobian_form"] = None
    if use_alpha_latent and latent_formulation_key != "transformed":
        alpha_sig_k = _latent_map_expr(problem["alpha_latent_k"], map_kind=latent_map_kind)
        r_alpha_embed = (problem["alpha_k"] - alpha_sig_k) * problem["alpha_test"] * dx_q
        a_alpha_embed = (problem["dalpha"] - (_latent_map_prime_expr(problem["alpha_latent_k"], map_kind=latent_map_kind) * problem["dalpha_latent"])) * problem["alpha_test"] * dx_q
        residual_form = residual_form + r_alpha_embed
        jacobian_form = jacobian_form + a_alpha_embed
        forms = replace(
            forms,
            residual_form=residual_form,
            jacobian_form=jacobian_form,
            r_alpha=(forms.r_alpha + r_alpha_embed),
            a_alpha=((forms.a_alpha + a_alpha_embed) if forms.a_alpha is not None else a_alpha_embed),
        )
    if use_phi_latent and bool(enable_phi_evolution) and latent_formulation_key != "transformed":
        phi_sig_k = _latent_map_expr(problem["phi_latent_k"], map_kind=latent_map_kind)
        r_phi_embed = (problem["phi_k"] - phi_sig_k) * problem["phi_test"] * dx_q
        a_phi_embed = (problem["dphi"] - (_latent_map_prime_expr(problem["phi_latent_k"], map_kind=latent_map_kind) * problem["dphi_latent"])) * problem["phi_test"] * dx_q
        residual_form = forms.residual_form + r_phi_embed
        jacobian_form = forms.jacobian_form + a_phi_embed
        forms = replace(
            forms,
            residual_form=residual_form,
            jacobian_form=jacobian_form,
            r_phi=(forms.r_phi + r_phi_embed),
            a_phi=((forms.a_phi + a_phi_embed) if forms.a_phi is not None else a_phi_embed),
        )
    if (
        bool(problem.get("alpha_mass_constraint", False))
        and problem.get("alpha_mass_lm_k") is not None
        and problem.get("alpha_mass_lm_test") is not None
        and problem.get("dalpha_mass_lm") is not None
    ):
        r_alpha_mass = (
            problem["alpha_mass_lm_k"] * problem["alpha_test"]
            + (alpha_k_eff - alpha_n_eff) * problem["alpha_mass_lm_test"]
        ) * dx_q
        a_alpha_mass = (
            problem["dalpha_mass_lm"] * problem["alpha_test"]
            + dalpha_eff * problem["alpha_mass_lm_test"]
        ) * dx_q
        residual_form = residual_form + r_alpha_mass
        jacobian_form = jacobian_form + a_alpha_mass
        problem["_alpha_mass_constraint_residual_form"] = r_alpha_mass
        problem["_alpha_mass_constraint_jacobian_form"] = a_alpha_mass
        forms = replace(
            forms,
            residual_form=residual_form,
            jacobian_form=jacobian_form,
        )
    if (
        bool(problem.get("pressure_mean_constraint", False))
        and problem.get("p_mean_k") is not None
        and problem.get("p_mean_test") is not None
        and problem.get("dp_mean") is not None
    ):
        r_pressure_mean = (
            problem["p_mean_k"] * problem["q_test"] + problem["p_k"] * problem["p_mean_test"]
        ) * dx_q
        a_pressure_mean = (
            problem["dp_mean"] * problem["q_test"] + problem["dp"] * problem["p_mean_test"]
        ) * dx_q
        residual_form = residual_form + r_pressure_mean
        jacobian_form = jacobian_form + a_pressure_mean
        problem["_pressure_mean_residual_form"] = r_pressure_mean
        problem["_pressure_mean_jacobian_form"] = a_pressure_mean
        forms = replace(
            forms,
            residual_form=residual_form,
            jacobian_form=jacobian_form,
        )
    return forms


def _build_bcs(
    *,
    fluid_space: str,
    enable_phi_evolution: bool,
    y_interface: float,
    eps_alpha: float,
    v_in: float,
    t_ramp: float,
    alpha_bc_mode: str,
    alpha_solid_dirichlet_sides: bool = False,
    alpha_solid_dirichlet_bottom: bool = False,
    solid_bc_mode: str = "lateral_clamped",
    latent_bounded_fields: tuple[str, ...] = tuple(),
    latent_bounded_eps: float = 1.0e-8,
    latent_bounded_map: str = "sigmoid",
    pressure_mean_constraint: bool = False,
) -> list[BoundaryCondition]:
    alpha_bc = lambda x, y, t: float(_alpha_equilibrium(y, y_interface=float(y_interface), eps_alpha=float(eps_alpha)).reshape(()))
    alpha_one = lambda x, y, t: 1.0
    zero = lambda x, y, t: 0.0
    inflow_y = lambda x, y, t: float(
        _cosine_ramp_value(float(t), float(t_ramp)) * _bottom_inlet(x, y, t, v_in=float(v_in)).reshape(())
    )
    alpha_bc_mode_key = str(alpha_bc_mode).strip().lower()
    if alpha_bc_mode_key == "auto":
        alpha_bc_mode_key = "natural"
    if alpha_bc_mode_key not in {"equilibrium", "natural"}:
        raise ValueError(f"Unsupported alpha_bc_mode={alpha_bc_mode!r}. Use 'auto', 'equilibrium', or 'natural'.")
    solid_bc_mode_key = str(solid_bc_mode).strip().lower()
    if solid_bc_mode_key not in {"base_only", "wall_normal", "lateral_clamped"}:
        raise ValueError(
            f"Unsupported solid_bc_mode={solid_bc_mode!r}. Use 'base_only', 'wall_normal', or 'lateral_clamped'."
        )

    bcs: list[BoundaryCondition] = []
    for tag in ("left", "right"):
        if str(fluid_space).strip().lower() == "hdiv":
            bcs.append(BoundaryCondition("v", "dirichlet", tag, _as_float_time(zero)))
        else:
            bcs.extend(
                [
                    BoundaryCondition("v_x", "dirichlet", tag, _as_float_time(zero)),
                    BoundaryCondition("v_y", "dirichlet", tag, _as_float_time(zero)),
                ]
            )
        if solid_bc_mode_key == "wall_normal":
            bcs.extend(
                [
                    BoundaryCondition("vS_x", "dirichlet", tag, _as_float_time(zero)),
                    BoundaryCondition("u_x", "dirichlet", tag, _as_float_time(zero)),
                ]
            )
        elif solid_bc_mode_key == "lateral_clamped":
            bcs.extend(
                [
                    BoundaryCondition("vS_x", "dirichlet", tag, _as_float_time(zero)),
                    BoundaryCondition("vS_y", "dirichlet", tag, _as_float_time(zero)),
                    BoundaryCondition("u_x", "dirichlet", tag, _as_float_time(zero)),
                    BoundaryCondition("u_y", "dirichlet", tag, _as_float_time(zero)),
                ]
            )
        if alpha_bc_mode_key == "equilibrium":
            bcs.extend(
                [
                    BoundaryCondition("alpha", "dirichlet", tag, _as_float_time(alpha_bc)),
                    BoundaryCondition("mu_alpha", "dirichlet", tag, _as_float_time(zero)),
                ]
            )
        if bool(alpha_solid_dirichlet_sides):
            bcs.append(BoundaryCondition("alpha", "dirichlet", tag, _as_float_time(alpha_one)))

    if str(fluid_space).strip().lower() == "hdiv":
        inflow_vec = lambda x, y, t: np.asarray([0.0, inflow_y(x, y, t)], dtype=float)
        bcs.append(BoundaryCondition("v", "dirichlet", "bottom", _as_bc_time(inflow_vec)))
    else:
        bcs.extend(
            [
                BoundaryCondition("v_x", "dirichlet", "bottom", _as_float_time(zero)),
                BoundaryCondition("v_y", "dirichlet", "bottom", _as_float_time(inflow_y)),
            ]
        )
    bcs.extend(
        [
            BoundaryCondition("vS_x", "dirichlet", "bottom", _as_float_time(zero)),
            BoundaryCondition("vS_y", "dirichlet", "bottom", _as_float_time(zero)),
            BoundaryCondition("u_x", "dirichlet", "bottom", _as_float_time(zero)),
            BoundaryCondition("u_y", "dirichlet", "bottom", _as_float_time(zero)),
        ]
    )
    if alpha_bc_mode_key == "equilibrium":
        bcs.extend(
            [
                BoundaryCondition("alpha", "dirichlet", "bottom", _as_float_time(alpha_bc)),
                BoundaryCondition("mu_alpha", "dirichlet", "bottom", _as_float_time(zero)),
            ]
        )
    if bool(alpha_solid_dirichlet_bottom):
        bcs.append(BoundaryCondition("alpha", "dirichlet", "bottom", _as_float_time(alpha_one)))
    if not bool(pressure_mean_constraint):
        bcs.extend(
            [
                BoundaryCondition("p", "dirichlet", "top", _as_float_time(zero)),
            ]
        )
    if alpha_bc_mode_key == "equilibrium":
        bcs.extend(
            [
                BoundaryCondition("alpha", "dirichlet", "top", _as_float_time(alpha_bc)),
                BoundaryCondition("mu_alpha", "dirichlet", "top", _as_float_time(zero)),
            ]
        )
    latent_field_set = {str(name).strip() for name in tuple(latent_bounded_fields or tuple()) if str(name).strip()}
    if latent_field_set:
        latent_bcs: list[BoundaryCondition] = []
        for bc in bcs:
            if bc.method != "dirichlet":
                continue
            latent_field = None
            if bc.field == "alpha" and "alpha" in latent_field_set:
                latent_field = "alpha_latent"
            elif bc.field == "phi" and "phi" in latent_field_set:
                latent_field = "phi_latent"
            if latent_field is None:
                continue
            latent_bcs.append(
                BoundaryCondition(
                    latent_field,
                    bc.method,
                    bc.domain_tag,
                    _latent_inverse_value_callback(
                        bc.value,
                        eps=float(latent_bounded_eps),
                        map_kind=str(latent_bounded_map),
                    ),
                )
            )
        bcs.extend(latent_bcs)
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


def _assemble_field_integral_weights(
    problem: dict[str, object],
    *,
    test_function,
    quad_order: int,
    backend: str,
) -> np.ndarray:
    _, vec = assemble_form(
        Equation(None, test_function * dx(metadata={"q": int(quad_order)})),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=int(quad_order),
        backend=str(backend),
    )
    return np.asarray(vec, dtype=float).ravel()


def _function_to_full_vector(dof_handler: DofHandler, func: Function | VectorFunction | HdivFunction) -> np.ndarray:
    full = np.zeros(int(dof_handler.total_dofs), dtype=float)
    g = np.asarray(getattr(func, "_g_dofs", np.array([], dtype=int)), dtype=int).ravel()
    vals = np.asarray(getattr(func, "nodal_values", np.array([], dtype=float)), dtype=float).ravel()
    if g.size != vals.size:
        raise ValueError(
            f"Function '{getattr(func, 'name', '<unnamed>')}' has nodal_values size {int(vals.size)} "
            f"but _g_dofs size {int(g.size)}."
        )
    if g.size:
        full[g] = vals
    return full


def _field_coordinate_permutation(
    problem: dict[str, object],
    *,
    source_field: str,
    target_field: str,
    digits: int = 12,
) -> np.ndarray:
    cache = problem.setdefault("_field_coordinate_permutations", {})
    cache_key = (str(source_field), str(target_field), int(digits))
    cached = cache.get(cache_key)
    if isinstance(cached, np.ndarray):
        return np.asarray(cached, dtype=int)

    dof_handler = problem["dh"]
    source_xy = np.asarray(dof_handler.get_dof_coords(str(source_field)), dtype=float)
    target_xy = np.asarray(dof_handler.get_dof_coords(str(target_field)), dtype=float)
    if source_xy.shape == target_xy.shape and np.allclose(source_xy, target_xy, atol=1.0e-14, rtol=0.0):
        perm = np.arange(source_xy.shape[0], dtype=int)
        cache[cache_key] = perm.copy()
        return perm

    key_to_source: dict[tuple[float, float], int] = {}
    for idx, xy in enumerate(source_xy):
        key = (round(float(xy[0]), int(digits)), round(float(xy[1]), int(digits)))
        if key in key_to_source:
            raise ValueError(
                f"Cannot build coordinate permutation from {source_field!r} to {target_field!r}: "
                f"duplicate source coordinate {key!r}."
            )
        key_to_source[key] = int(idx)

    perm = np.full(target_xy.shape[0], -1, dtype=int)
    missing: list[tuple[float, float]] = []
    for idx, xy in enumerate(target_xy):
        key = (round(float(xy[0]), int(digits)), round(float(xy[1]), int(digits)))
        src_idx = key_to_source.get(key)
        if src_idx is None:
            missing.append(key)
            continue
        perm[idx] = int(src_idx)
    if np.any(perm < 0):
        example = missing[0] if missing else ("?", "?")
        raise ValueError(
            f"Cannot build coordinate permutation from {source_field!r} to {target_field!r}: "
            f"{int(np.count_nonzero(perm < 0))} target DOFs could not be matched (e.g. {example!r})."
        )

    cache[cache_key] = perm.copy()
    return perm


def _apply_field_box_bounds(
    lower_full: np.ndarray,
    upper_full: np.ndarray,
    *,
    dof_handler: DofHandler,
    field_name: str,
    lo: float | None,
    hi: float | None,
    local_mask: np.ndarray | None = None,
) -> None:
    field_dofs = np.asarray(dof_handler.get_field_slice(str(field_name)), dtype=int).ravel()
    if local_mask is None:
        target = field_dofs
    else:
        mask = np.asarray(local_mask, dtype=bool).ravel()
        if mask.size != field_dofs.size:
            raise ValueError(
                f"Local mask for field {field_name!r} has size {int(mask.size)}, expected {int(field_dofs.size)}."
            )
        target = field_dofs[mask]
    if lo is not None and target.size:
        lower_full[target] = float(lo)
    if hi is not None and target.size:
        upper_full[target] = float(hi)


def _build_support_aware_phi_box_bounds(
    problem: dict[str, object],
    *,
    alpha_func: Function,
    alpha_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dof_handler = problem["dh"]
    total_dofs = int(dof_handler.total_dofs)
    lower_full = np.full(total_dofs, -np.inf, dtype=float)
    upper_full = np.full(total_dofs, np.inf, dtype=float)
    phi_dofs = np.asarray(dof_handler.get_field_slice("phi"), dtype=int).ravel()
    phi_to_alpha = _field_coordinate_permutation(problem, source_field="alpha", target_field="phi")
    alpha_values = np.asarray(getattr(alpha_func, "nodal_values", np.array([], dtype=float)), dtype=float).ravel()
    if alpha_values.size != int(phi_to_alpha.max(initial=-1) + 1):
        alpha_size_expected = int(np.asarray(dof_handler.get_dof_coords("alpha"), dtype=float).shape[0])
        if alpha_values.size != alpha_size_expected:
            raise ValueError(
                f"alpha_func has {int(alpha_values.size)} nodal values, expected {alpha_size_expected} for support-aware phi bounds."
            )
    support_mask = np.asarray(alpha_values[phi_to_alpha] > float(alpha_threshold), dtype=bool)
    _apply_field_box_bounds(
        lower_full,
        upper_full,
        dof_handler=dof_handler,
        field_name="phi",
        lo=0.0,
        hi=1.0,
        local_mask=support_mask,
    )
    return lower_full, upper_full, support_mask


def _apply_open_top_global_phi_cleanup(
    *,
    args: argparse.Namespace,
    problem: dict[str, object],
    funcs,
    find_named_function,
) -> None:
    if bool(problem.get("latent_bounded_transport", False)):
        return
    if not bool(args.enable_phi_evolution) or problem["phi_k"] is None or not bool(args.phi_box_constraints):
        return
    if not _full_top_drainage_transport_enabled(
        enable_phi_evolution=bool(args.enable_phi_evolution),
        top_drainage_transport=bool(args.top_drainage_transport),
    ):
        return
    phi_cur = find_named_function(funcs, problem["phi_k"])
    phi_vals = np.asarray(getattr(phi_cur, "nodal_values", np.array([], dtype=float)), dtype=float)
    if phi_vals.size:
        np.clip(phi_vals, 0.0, 1.0, out=phi_vals)


def _build_vi_linear_equalities(
    *,
    args: argparse.Namespace,
    problem: dict[str, object],
    qdeg: int,
    alpha_bc_mode_key: str,
    find_named_function,
) -> list[LinearEqualityConstraint]:
    if not (bool(args.alpha_box_constraints) and str(alpha_bc_mode_key) == "natural"):
        return []

    alpha_weights_full = _assemble_field_integral_weights(
        problem,
        test_function=problem["alpha_test"],
        quad_order=int(qdeg),
        backend=str(args.backend),
    )

    def _alpha_mass_target(*, prev_funcs, **_kwargs) -> float:
        alpha_prev = find_named_function(prev_funcs, problem["alpha_n"])
        return float(alpha_weights_full @ _function_to_full_vector(problem["dh"], alpha_prev))

    equalities: list[LinearEqualityConstraint] = []
    if (
        not bool(problem.get("alpha_mass_constraint", False))
        and not bool(getattr(args, "alpha_from_refmap", False))
    ):
        equalities.append(
            LinearEqualityConstraint(
                name="alpha_mass",
                weights_full=alpha_weights_full,
                target_callback=_alpha_mass_target,
                field_name="alpha",
                project_feasible=True,
            )
        )

    if (
        bool(args.enable_phi_evolution)
        and problem["phi_k"] is not None
        and bool(args.phi_box_constraints)
        and not _full_top_drainage_transport_enabled(
            enable_phi_evolution=bool(args.enable_phi_evolution),
            top_drainage_transport=bool(args.top_drainage_transport),
        )
    ):
        def _phi_biofilm_fluid_mass_weights(*, funcs, **_kwargs) -> np.ndarray:
            alpha_cur = find_named_function(funcs, problem["alpha_k"])
            _, vec = assemble_form(
                Equation(None, alpha_cur * problem["phi_test"] * dx(metadata={"q": int(qdeg)})),
                dof_handler=problem["dh"],
                bcs=[],
                quad_order=int(qdeg),
                backend=str(args.backend),
            )
            return np.asarray(vec, dtype=float).ravel()

        def _phi_biofilm_fluid_mass_target(*, prev_funcs, **_kwargs) -> float:
            alpha_prev = find_named_function(prev_funcs, problem["alpha_n"])
            phi_prev = find_named_function(prev_funcs, problem["phi_n"])
            _, weights_prev = assemble_form(
                Equation(None, alpha_prev * problem["phi_test"] * dx(metadata={"q": int(qdeg)})),
                dof_handler=problem["dh"],
                bcs=[],
                quad_order=int(qdeg),
                backend=str(args.backend),
            )
            return float(np.asarray(weights_prev, dtype=float).ravel() @ _function_to_full_vector(problem["dh"], phi_prev))

        equalities.append(
            LinearEqualityConstraint(
                name="phi_biofilm_fluid_mass",
                weights_callback=_phi_biofilm_fluid_mass_weights,
                target_callback=_phi_biofilm_fluid_mass_target,
                field_name="phi",
                project_feasible=True,
            )
        )
    return equalities


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
    io_root = _mpi_io_root()
    poly_order, pressure_order, scalar_order = _resolved_orders(args)
    qdeg = int(args.quad_order) if args.quad_order is not None else max(6, 2 * int(poly_order) + 2)
    if qdeg < 1:
        raise ValueError("quad_order must be positive.")
    eps_alpha_eff = _effective_eps_alpha(args)
    if args.eps_alpha_over_h is not None:
        print(
            "[info] using mesh-scaled interface width: "
            f"eps_alpha={float(eps_alpha_eff):.3e} from eps_alpha_over_h={float(args.eps_alpha_over_h):.3e}."
        )
    alpha_reg_eps_normal = float(args.alpha_reg_eps_normal) if args.alpha_reg_eps_normal is not None else float(eps_alpha_eff)
    alpha_reg_eps_tangent = float(args.alpha_reg_eps_tangent) if args.alpha_reg_eps_tangent is not None else float(0.25 * eps_alpha_eff)
    h_char = _characteristic_h(Lx=float(args.Lx), Ly=float(args.Ly), nx=int(args.nx), ny=int(args.ny))
    mechanics_nondim_key = str(args.mechanics_nondim_mode).strip().lower()
    volume_setup = _condition_balanced_volume_setup(
        mechanics_nondim_mode=mechanics_nondim_key,
        mu_f=float(args.mu_f),
        kappa_inv=float(1.0 / float(kappa)),
        gamma_div=float(args.gamma_div),
        auto_gamma_div=bool(getattr(args, "condition_balanced_auto_gamma_div", True)),
    )
    effective_gamma_div = float(volume_setup["gamma_div"])
    solid_dof_y_cut = _condition_balanced_solid_cutoff_y(
        mechanics_nondim_mode=mechanics_nondim_key,
        y_interface=float(args.y_interface),
        solid_dof_y_cut=args.solid_dof_y_cut,
        condition_balanced_solid_cut_fix=bool(getattr(args, "condition_balanced_solid_cut_fix", True)),
    )
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
        latent_bounded_transport=bool(args.latent_bounded_transport),
        latent_bounded_fields=_latent_bounded_fields(
            args,
            enable_phi_evolution=bool(args.enable_phi_evolution),
        ),
        latent_bounded_map=str(args.latent_bounded_map),
        latent_bounded_formulation=str(args.latent_bounded_formulation),
        alpha_mass_constraint=bool(args.alpha_mass_constraint),
        pressure_mean_constraint=bool(args.pressure_mean_constraint),
        solid_volumetric_split=bool(args.solid_volumetric_split),
        drag_formulation=str(args.drag_formulation),
    )
    if (
        mechanics_nondim_key == "condition_balanced"
        and bool(getattr(args, "condition_balanced_auto_gamma_div", True))
        and float(args.gamma_div) == 0.0
    ):
        print(
            "[setup] using condition-balanced volume AL "
            f"gamma_div={float(effective_gamma_div):.6g} "
            f"(Darcy scale mu_f*kappa_inv={float(volume_setup['darcy_ref']):.6g})."
        )
    problem["_effective_gamma_div"] = float(effective_gamma_div)
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

    hdiv_tangential_bc_measure = None
    if str(args.fluid_space).strip().lower() == "hdiv" and bool(args.hdiv_tangential_dirichlet):
        hdiv_tangential_bc_measure = dS(
            defined_on=(
                problem["mesh"].edge_bitset("left")
                | problem["mesh"].edge_bitset("right")
                | problem["mesh"].edge_bitset("bottom")
            ),
            metadata={"q": int(qdeg)},
        )
    ds_alpha_transport, ds_B_transport = _build_transport_measures(
        problem=problem,
        qdeg=int(qdeg),
        enable_phi_evolution=bool(args.enable_phi_evolution),
        top_drainage_transport=bool(args.top_drainage_transport),
        support_physics=str(args.support_physics),
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
    if bool(problem.get("latent_bounded_transport", False)):
        latent_eps = float(getattr(args, "latent_bounded_eps", 1.0e-8))
        latent_map_kind = _latent_bounded_map_key(problem)
        if "alpha" in tuple(problem.get("latent_bounded_fields", tuple()) or tuple()):
            alpha_latent_init = _latent_inverse_array(
                problem["alpha_n"].nodal_values,
                eps=latent_eps,
                map_kind=latent_map_kind,
            )
            problem["alpha_latent_n"].nodal_values[:] = alpha_latent_init
            problem["alpha_latent_k"].nodal_values[:] = alpha_latent_init
        if (
            "phi" in tuple(problem.get("latent_bounded_fields", tuple()) or tuple())
            and problem.get("phi_latent_n") is not None
            and problem.get("phi_n") is not None
        ):
            phi_latent_init = _latent_inverse_array(
                problem["phi_n"].nodal_values,
                eps=latent_eps,
                map_kind=latent_map_kind,
            )
            problem["phi_latent_n"].nodal_values[:] = phi_latent_init
            problem["phi_latent_k"].nodal_values[:] = phi_latent_init
        _sync_latent_bounded_problem_fields(problem=problem)
    if problem.get("alpha_mass_lm_n") is not None:
        problem["alpha_mass_lm_n"].nodal_values[:] = 0.0
        problem["alpha_mass_lm_k"].nodal_values[:] = 0.0
    alpha_from_refmap_enabled = bool(getattr(args, "alpha_from_refmap", False))
    alpha_from_refmap_perm = None
    alpha_xy = None
    if alpha_from_refmap_enabled:
        alpha_from_refmap_perm = _field_coordinate_permutation(
            problem,
            source_field="u_x",
            target_field="alpha",
        )
        alpha_xy = np.asarray(problem["dh"].get_dof_coords("alpha"), dtype=float)

        def _alpha_from_refmap_values(displacement) -> np.ndarray:
            ux_vals = _vector_component_values(displacement, 0)
            uy_vals = _vector_component_values(displacement, 1)
            u_at_alpha = np.column_stack(
                [
                    np.asarray(ux_vals[alpha_from_refmap_perm], dtype=float),
                    np.asarray(uy_vals[alpha_from_refmap_perm], dtype=float),
                ]
            )
            chi = np.asarray(alpha_xy, dtype=float) - u_at_alpha
            alpha_vals = np.clip(
                np.asarray(
                    _alpha_equilibrium(
                        chi[:, 1],
                        y_interface=float(args.y_interface),
                        eps_alpha=float(eps_alpha_eff),
                    ),
                    dtype=float,
                ).ravel(),
                0.0,
                1.0,
            )
            return alpha_vals

        def _sync_alpha_from_refmap(*, sync_previous: bool = False) -> None:
            problem["alpha_k"].nodal_values[:] = _alpha_from_refmap_values(problem["u_k"])
            if problem.get("mu_k") is not None:
                problem["mu_k"].nodal_values[:] = 0.0
            if problem.get("alpha_mass_lm_k") is not None:
                problem["alpha_mass_lm_k"].nodal_values[:] = 0.0
            if sync_previous:
                problem["alpha_n"].nodal_values[:] = _alpha_from_refmap_values(problem["u_n"])
                if problem.get("mu_n") is not None:
                    problem["mu_n"].nodal_values[:] = 0.0
                if problem.get("alpha_mass_lm_n") is not None:
                    problem["alpha_mass_lm_n"].nodal_values[:] = 0.0

        _sync_alpha_from_refmap(sync_previous=True)
        inactive_alpha_counts = _tag_inactive_fields(
            problem,
            "alpha",
            "mu_alpha",
            "alpha_mass_lm",
        )
        print(
            "[setup] enabling alpha-from-refmap mode: alpha and its auxiliary transport unknowns are frozen "
            f"and alpha is rebuilt from u before assembly (inactive counts={inactive_alpha_counts})."
        )
    else:
        def _sync_alpha_from_refmap() -> None:
            return
    if solid_dof_y_cut is not None:
        solid_inactive_counts = _tag_inactive_solid_dofs_above_y(
            problem,
            y_cut=solid_dof_y_cut,
        )
        if solid_inactive_counts:
            dropped = int(sum(int(v) for v in solid_inactive_counts.values()))
            print(
                "[setup] deactivating solid DOFs above the condition-balanced cutoff "
                f"y={float(solid_dof_y_cut):.6g}: "
                f"{dropped} total "
                f"(u_x={solid_inactive_counts.get('u_x', 0)}, "
                f"u_y={solid_inactive_counts.get('u_y', 0)}, "
                f"vS_x={solid_inactive_counts.get('vS_x', 0)}, "
                f"vS_y={solid_inactive_counts.get('vS_y', 0)})."
            )
    else:
        problem["_inactive_solid_reference_y"] = None
        problem["_inactive_solid_alpha_band_halfwidth"] = float("nan")
        solid_inactive_counts = {}

    dt_c = _named_constant("b7_dt", float(args.dt))
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
        gamma_div=float(effective_gamma_div),
        mechanics_nondim_mode=str(args.mechanics_nondim_mode),
        solid_volumetric_split=bool(args.solid_volumetric_split),
        solid_volumetric_penalty=float(args.solid_volumetric_penalty),
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
        alpha_supg=float(args.alpha_supg),
        alpha_cip=float(args.alpha_cip),
        alpha_regularization=str(args.alpha_regularization),
        alpha_reg_gamma=float(args.alpha_reg_gamma),
        alpha_reg_eps_normal=float(alpha_reg_eps_normal),
        alpha_reg_eps_tangent=float(alpha_reg_eps_tangent),
        alpha_reg_eta=float(args.alpha_reg_eta),
        alpha_advect_with=str(args.alpha_advect_with),
        alpha_advection_form=str(args.alpha_advection_form),
        support_physics=str(args.support_physics),
        solid_model=str(args.solid_model),
        kappa_inv_model=str(args.kappa_inv_model),
        drag_formulation=str(args.drag_formulation),
        fluid_convection=str(args.fluid_convection),
        skeleton_pressure_mode=str(args.skeleton_pressure_mode),
        alpha_biot=(None if args.alpha_biot is None else float(args.alpha_biot)),
        enable_phi_evolution=bool(args.enable_phi_evolution),
        include_skeleton_acceleration=bool(args.include_skeleton_acceleration),
        rho_s0_tilde=float(args.rho_s0_tilde),
        skeleton_inertia_convection=str(args.skeleton_inertia_convection),
        ds_hdiv_tangential=hdiv_tangential_bc_measure,
        ds_alpha_transport=ds_alpha_transport,
        ds_B_transport=ds_B_transport,
        hdiv_tangential_gamma=float(args.hdiv_tangential_gamma),
        hdiv_tangential_method=str(args.hdiv_tangential_method),
    )

    bcs = _build_bcs(
        fluid_space=str(args.fluid_space),
        enable_phi_evolution=bool(args.enable_phi_evolution),
        y_interface=float(args.y_interface),
        eps_alpha=float(eps_alpha_eff),
        v_in=float(args.v_in),
        t_ramp=float(args.t_ramp),
        alpha_bc_mode=str(args.alpha_bc_mode),
        alpha_solid_dirichlet_sides=bool(args.alpha_solid_dirichlet_sides),
        alpha_solid_dirichlet_bottom=bool(args.alpha_solid_dirichlet_bottom),
        solid_bc_mode=str(args.solid_bc_mode),
        latent_bounded_fields=tuple(problem.get("latent_bounded_fields", tuple()) or tuple()),
        latent_bounded_eps=float(getattr(args, "latent_bounded_eps", 1.0e-8)),
        latent_bounded_map=str(getattr(args, "latent_bounded_map", "sigmoid")),
        pressure_mean_constraint=bool(args.pressure_mean_constraint),
    )
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, (lambda x, y: 0.0)) for b in bcs]

    if io_root:
        outdir.mkdir(parents=True, exist_ok=True)
    barrier(MPI_CTX)
    vtk_dir = outdir / "vtk"
    dt_c = _named_constant("b7_dt", float(args.dt))
    timeseries_rows: list[dict[str, float]] = []
    alpha_diagnostics = _build_alpha_diagnostics(problem, quad_order=int(qdeg), backend=str(args.backend))
    alpha_coeffs = {problem["alpha_k"].name: problem["alpha_k"]}
    alpha_diag0 = alpha_diagnostics.evaluate(alpha_coeffs)
    alpha_area0 = float(alpha_diag0.get("alpha_area", float("nan")))
    alpha_band0 = float(alpha_diag0.get("alpha_band", float("nan")))

    base_solver_max_newton_iter = int(args.max_it)
    base_solver_relaxed_accept_ginf = float(
        getattr(getattr(solver, "vi_params", None), "relaxed_filter_accept_ginf", 0.0) or 0.0
    ) if "solver" in locals() else 0.0
    base_vi_c = float(getattr(getattr(solver, "vi_params", None), "c", float(args.vi_c)) or float(args.vi_c)) if "solver" in locals() else float(args.vi_c)
    base_vi_c_by_field = dict(getattr(getattr(solver, "vi_params", None), "c_by_field", {}) or {}) if "solver" in locals() else {}
    base_vi_enter_tol = float(getattr(getattr(solver, "vi_params", None), "enter_tol", float(args.vi_enter_tol)) or float(args.vi_enter_tol)) if "solver" in locals() else float(args.vi_enter_tol)
    base_vi_leave_tol = float(getattr(getattr(solver, "vi_params", None), "leave_tol", float(args.vi_leave_tol)) or float(args.vi_leave_tol)) if "solver" in locals() else float(args.vi_leave_tol)
    base_vi_persistence = int(getattr(getattr(solver, "vi_params", None), "active_set_persistence", int(args.vi_persistence)) or int(args.vi_persistence)) if "solver" in locals() else int(args.vi_persistence)
    base_vi_active_soft_threshold = int(getattr(getattr(solver, "vi_params", None), "active_step_delta_active_trigger", int(args.vi_active_soft_threshold)) or int(args.vi_active_soft_threshold)) if "solver" in locals() else int(args.vi_active_soft_threshold)
    base_vi_active_soft_alpha = float(getattr(getattr(solver, "vi_params", None), "active_step_soft_alpha", float(args.vi_active_soft_alpha)) or float(args.vi_active_soft_alpha)) if "solver" in locals() else float(args.vi_active_soft_alpha)
    base_vi_eq_active_step_threshold = float(
        getattr(getattr(solver, "vi_params", None), "equality_active_step_ginf_threshold", 5.0e-2) or 5.0e-2
    ) if "solver" in locals() else 5.0e-2
    base_vi_nonmono_window = int(
        getattr(getattr(solver, "vi_params", None), "line_search_nonmonotone_window", 0) or 0
    ) if "solver" in locals() else 0
    base_vi_nonmono_stable_iters = int(
        getattr(getattr(solver, "vi_params", None), "line_search_nonmonotone_active_stable_iters", 0) or 0
    ) if "solver" in locals() else 0
    base_vi_nonmono_ginf_trigger = float(
        getattr(getattr(solver, "vi_params", None), "line_search_nonmonotone_ginf_trigger", 0.0) or 0.0
    ) if "solver" in locals() else 0.0
    base_vi_nonmono_gap_ratio = float(
        getattr(getattr(solver, "vi_params", None), "line_search_nonmonotone_gap_ratio", 1.0) or 1.0
    ) if "solver" in locals() else 1.0
    base_vi_nonmono_eq_abs = float(
        getattr(getattr(solver, "vi_params", None), "line_search_nonmonotone_eq_abs", 1.0e-10) or 1.0e-10
    ) if "solver" in locals() else 1.0e-10
    base_vi_nonmono_disable_filter = bool(
        getattr(getattr(solver, "vi_params", None), "line_search_nonmonotone_disable_filter", False)
    ) if "solver" in locals() else False
    base_vi_accept_best_filtered_descent = bool(
        getattr(getattr(solver, "vi_params", None), "accept_best_filtered_descent", False)
    ) if "solver" in locals() else False
    base_vi_lm_max_tries = int(
        getattr(getattr(solver, "vi_params", None), "unconstrained_lm_max_tries", int(args.vi_lm_max_tries))
        or int(args.vi_lm_max_tries)
    ) if "solver" in locals() else int(args.vi_lm_max_tries)
    base_vi_lm_lambda_max = float(
        getattr(getattr(solver, "vi_params", None), "unconstrained_lm_lambda_max", float(args.vi_lm_lambda_max))
        or float(args.vi_lm_lambda_max)
    ) if "solver" in locals() else float(args.vi_lm_lambda_max)
    base_vi_affine_cycle_fallback = bool(
        getattr(getattr(solver, "vi_params", None), "affine_cycle_fallback", False)
    ) if "solver" in locals() else False
    base_vi_affine_identified_acceleration = bool(
        getattr(getattr(solver, "vi_params", None), "affine_identified_acceleration", False)
    ) if "solver" in locals() else False
    base_vi_affine_identified_stable_iters = int(
        getattr(getattr(solver, "vi_params", None), "affine_identified_stable_iters", 2) or 2
    ) if "solver" in locals() else 2
    base_vi_affine_identified_ginf_trigger = float(
        getattr(getattr(solver, "vi_params", None), "affine_identified_ginf_trigger", 5.0e-2) or 5.0e-2
    ) if "solver" in locals() else 5.0e-2
    base_vi_working_set_guard_after_affine = int(
        getattr(getattr(solver, "vi_params", None), "working_set_guard_after_affine", 0) or 0
    ) if "solver" in locals() else 0
    base_vi_field_proximal_recovery = bool(
        getattr(getattr(solver, "vi_params", None), "field_proximal_recovery", False)
    ) if "solver" in locals() else False
    base_vi_field_proximal_recovery_fields = tuple(
        getattr(getattr(solver, "vi_params", None), "field_proximal_recovery_fields", ()) or ()
    ) if "solver" in locals() else tuple()
    base_vi_field_proximal_recovery_lambda0 = float(
        getattr(getattr(solver, "vi_params", None), "field_proximal_recovery_lambda0", 1.0e-3) or 1.0e-3
    ) if "solver" in locals() else 1.0e-3
    base_vi_field_proximal_recovery_lambda_max = float(
        getattr(getattr(solver, "vi_params", None), "field_proximal_recovery_lambda_max", 1.0e6) or 1.0e6
    ) if "solver" in locals() else 1.0e6
    base_vi_field_proximal_recovery_growth = float(
        getattr(getattr(solver, "vi_params", None), "field_proximal_recovery_growth", 10.0) or 10.0
    ) if "solver" in locals() else 10.0
    base_vi_field_proximal_recovery_max_tries = int(
        getattr(getattr(solver, "vi_params", None), "field_proximal_recovery_max_tries", 6) or 6
    ) if "solver" in locals() else 6
    base_vi_field_proximal_recovery_stable_iters = int(
        getattr(getattr(solver, "vi_params", None), "field_proximal_recovery_stable_iters", 1) or 1
    ) if "solver" in locals() else 1
    base_vi_field_proximal_recovery_ginf_trigger = float(
        getattr(getattr(solver, "vi_params", None), "field_proximal_recovery_ginf_trigger", 5.0e-2) or 5.0e-2
    ) if "solver" in locals() else 5.0e-2
    base_vi_field_proximal_recovery_gap_ratio = float(
        getattr(getattr(solver, "vi_params", None), "field_proximal_recovery_gap_ratio", 1.0) or 1.0
    ) if "solver" in locals() else 1.0
    base_vi_field_proximal_recovery_eq_abs = float(
        getattr(getattr(solver, "vi_params", None), "field_proximal_recovery_eq_abs", 1.0e-10) or 1.0e-10
    ) if "solver" in locals() else 1.0e-10
    base_vi_field_proximal_recovery_identified_window = bool(
        getattr(getattr(solver, "vi_params", None), "field_proximal_recovery_identified_window", False)
    ) if "solver" in locals() else False
    base_vi_field_proximal_recovery_ginf_max = float(
        getattr(getattr(solver, "vi_params", None), "field_proximal_recovery_ginf_max", 0.0) or 0.0
    ) if "solver" in locals() else 0.0
    base_vi_ptc_recovery = bool(
        getattr(getattr(solver, "vi_params", None), "ptc_recovery", bool(args.vi_ptc_recovery))
    ) if "solver" in locals() else bool(args.vi_ptc_recovery)
    base_vi_ptc_fields = tuple(
        getattr(getattr(solver, "vi_params", None), "ptc_fields", _parse_csv_fields(args.vi_ptc_fields)) or ()
    ) if "solver" in locals() else _parse_csv_fields(args.vi_ptc_fields)
    base_vi_ptc_sigma0 = float(
        getattr(getattr(solver, "vi_params", None), "ptc_sigma0", float(args.vi_ptc_sigma0)) or float(args.vi_ptc_sigma0)
    ) if "solver" in locals() else float(args.vi_ptc_sigma0)
    base_vi_ptc_sigma_max = float(
        getattr(getattr(solver, "vi_params", None), "ptc_sigma_max", float(args.vi_ptc_sigma_max)) or float(args.vi_ptc_sigma_max)
    ) if "solver" in locals() else float(args.vi_ptc_sigma_max)
    base_vi_ptc_growth = float(
        getattr(getattr(solver, "vi_params", None), "ptc_growth", float(args.vi_ptc_growth)) or float(args.vi_ptc_growth)
    ) if "solver" in locals() else float(args.vi_ptc_growth)
    base_vi_ptc_decay = float(
        getattr(getattr(solver, "vi_params", None), "ptc_decay", float(args.vi_ptc_decay)) or float(args.vi_ptc_decay)
    ) if "solver" in locals() else float(args.vi_ptc_decay)
    base_vi_ptc_stable_iters = int(
        getattr(getattr(solver, "vi_params", None), "ptc_stable_iters", 1) or 1
    ) if "solver" in locals() else 1
    base_vi_ptc_ginf_trigger = float(
        getattr(getattr(solver, "vi_params", None), "ptc_ginf_trigger", float(args.vi_ptc_ginf_trigger)) or float(args.vi_ptc_ginf_trigger)
    ) if "solver" in locals() else float(args.vi_ptc_ginf_trigger)
    base_vi_ptc_gap_ratio = float(
        getattr(getattr(solver, "vi_params", None), "ptc_gap_ratio", 1.0) or 1.0
    ) if "solver" in locals() else 1.0
    base_vi_ptc_eq_abs = float(
        getattr(getattr(solver, "vi_params", None), "ptc_eq_abs", 1.0e-10) or 1.0e-10
    ) if "solver" in locals() else 1.0e-10
    base_vi_ptc_ginf_max = float(
        getattr(getattr(solver, "vi_params", None), "ptc_ginf_max", float(args.vi_ptc_ginf_max)) or float(args.vi_ptc_ginf_max)
    ) if "solver" in locals() else float(args.vi_ptc_ginf_max)
    base_vi_ptc_identified_window = bool(
        getattr(getattr(solver, "vi_params", None), "ptc_identified_window", bool(args.vi_ptc_recovery))
    ) if "solver" in locals() else bool(args.vi_ptc_recovery)
    base_vi_ptc_freeze_complement = bool(
        getattr(getattr(solver, "vi_params", None), "ptc_freeze_complement", True)
    ) if "solver" in locals() else True
    base_vi_anderson_acceleration = bool(
        getattr(getattr(solver, "vi_params", None), "anderson_acceleration", bool(args.vi_anderson_acceleration))
    ) if "solver" in locals() else bool(args.vi_anderson_acceleration)
    base_vi_anderson_history = int(
        getattr(getattr(solver, "vi_params", None), "anderson_history", int(args.vi_anderson_history)) or int(args.vi_anderson_history)
    ) if "solver" in locals() else int(args.vi_anderson_history)
    base_vi_anderson_regularization = float(
        getattr(getattr(solver, "vi_params", None), "anderson_regularization", float(args.vi_anderson_regularization)) or float(args.vi_anderson_regularization)
    ) if "solver" in locals() else float(args.vi_anderson_regularization)
    base_vi_anderson_damping = float(
        getattr(getattr(solver, "vi_params", None), "anderson_damping", float(args.vi_anderson_damping)) or float(args.vi_anderson_damping)
    ) if "solver" in locals() else float(args.vi_anderson_damping)
    base_vi_anderson_stable_iters = int(
        getattr(getattr(solver, "vi_params", None), "anderson_stable_iters", 1) or 1
    ) if "solver" in locals() else 1
    base_vi_anderson_ginf_trigger = float(
        getattr(getattr(solver, "vi_params", None), "anderson_ginf_trigger", 5.0e-2) or 5.0e-2
    ) if "solver" in locals() else 5.0e-2
    base_vi_anderson_gap_ratio = float(
        getattr(getattr(solver, "vi_params", None), "anderson_gap_ratio", 1.0) or 1.0
    ) if "solver" in locals() else 1.0
    base_vi_anderson_eq_abs = float(
        getattr(getattr(solver, "vi_params", None), "anderson_eq_abs", 1.0e-10) or 1.0e-10
    ) if "solver" in locals() else 1.0e-10
    base_vi_anderson_ginf_max = float(
        getattr(getattr(solver, "vi_params", None), "anderson_ginf_max", 0.0) or 0.0
    ) if "solver" in locals() else 0.0
    startup_monolithic_budget = int(_startup_monolithic_max_it(args))

    def _restore_base_monolithic_controls() -> None:
        if int(getattr(solver.np, "max_newton_iter", base_solver_max_newton_iter)) != int(base_solver_max_newton_iter):
            solver.np.max_newton_iter = int(base_solver_max_newton_iter)
        if getattr(solver, "vi_params", None) is not None:
            relaxed_now = float(getattr(solver.vi_params, "relaxed_filter_accept_ginf", 0.0) or 0.0)
            if relaxed_now != float(base_solver_relaxed_accept_ginf):
                solver.vi_params.relaxed_filter_accept_ginf = float(base_solver_relaxed_accept_ginf)
            solver.vi_params.c = float(base_vi_c)
            solver.vi_params.c_by_field = dict(base_vi_c_by_field)
            solver.vi_params.enter_tol = float(base_vi_enter_tol)
            solver.vi_params.leave_tol = float(base_vi_leave_tol)
            solver.vi_params.active_set_persistence = int(base_vi_persistence)
            solver.vi_params.active_step_delta_active_trigger = int(base_vi_active_soft_threshold)
            solver.vi_params.active_step_soft_alpha = float(base_vi_active_soft_alpha)
            solver.vi_params.equality_active_step_ginf_threshold = float(base_vi_eq_active_step_threshold)
            solver.vi_params.line_search_nonmonotone_window = int(base_vi_nonmono_window)
            solver.vi_params.line_search_nonmonotone_active_stable_iters = int(base_vi_nonmono_stable_iters)
            solver.vi_params.line_search_nonmonotone_ginf_trigger = float(base_vi_nonmono_ginf_trigger)
            solver.vi_params.line_search_nonmonotone_gap_ratio = float(base_vi_nonmono_gap_ratio)
            solver.vi_params.line_search_nonmonotone_eq_abs = float(base_vi_nonmono_eq_abs)
            solver.vi_params.line_search_nonmonotone_disable_filter = bool(base_vi_nonmono_disable_filter)
            solver.vi_params.accept_best_filtered_descent = bool(base_vi_accept_best_filtered_descent)
            solver.vi_params.unconstrained_lm_max_tries = int(base_vi_lm_max_tries)
            solver.vi_params.unconstrained_lm_lambda_max = float(base_vi_lm_lambda_max)
            solver.vi_params.affine_cycle_fallback = bool(base_vi_affine_cycle_fallback)
            solver.vi_params.affine_identified_acceleration = bool(base_vi_affine_identified_acceleration)
            solver.vi_params.affine_identified_stable_iters = int(base_vi_affine_identified_stable_iters)
            solver.vi_params.affine_identified_ginf_trigger = float(base_vi_affine_identified_ginf_trigger)
            solver.vi_params.working_set_guard_after_affine = int(base_vi_working_set_guard_after_affine)
            solver.vi_params.field_proximal_recovery = bool(base_vi_field_proximal_recovery)
            solver.vi_params.field_proximal_recovery_fields = tuple(base_vi_field_proximal_recovery_fields)
            solver.vi_params.field_proximal_recovery_lambda0 = float(base_vi_field_proximal_recovery_lambda0)
            solver.vi_params.field_proximal_recovery_lambda_max = float(base_vi_field_proximal_recovery_lambda_max)
            solver.vi_params.field_proximal_recovery_growth = float(base_vi_field_proximal_recovery_growth)
            solver.vi_params.field_proximal_recovery_max_tries = int(base_vi_field_proximal_recovery_max_tries)
            solver.vi_params.field_proximal_recovery_stable_iters = int(base_vi_field_proximal_recovery_stable_iters)
            solver.vi_params.field_proximal_recovery_ginf_trigger = float(base_vi_field_proximal_recovery_ginf_trigger)
            solver.vi_params.field_proximal_recovery_gap_ratio = float(base_vi_field_proximal_recovery_gap_ratio)
            solver.vi_params.field_proximal_recovery_eq_abs = float(base_vi_field_proximal_recovery_eq_abs)
            solver.vi_params.ptc_recovery = bool(base_vi_ptc_recovery)
            solver.vi_params.ptc_fields = tuple(base_vi_ptc_fields)
            solver.vi_params.ptc_sigma0 = float(base_vi_ptc_sigma0)
            solver.vi_params.ptc_sigma_max = float(base_vi_ptc_sigma_max)
            solver.vi_params.ptc_growth = float(base_vi_ptc_growth)
            solver.vi_params.ptc_decay = float(base_vi_ptc_decay)
            solver.vi_params.ptc_stable_iters = int(base_vi_ptc_stable_iters)
            solver.vi_params.ptc_ginf_trigger = float(base_vi_ptc_ginf_trigger)
            solver.vi_params.ptc_gap_ratio = float(base_vi_ptc_gap_ratio)
            solver.vi_params.ptc_eq_abs = float(base_vi_ptc_eq_abs)
            solver.vi_params.ptc_ginf_max = float(base_vi_ptc_ginf_max)
            solver.vi_params.ptc_identified_window = bool(base_vi_ptc_identified_window)
            solver.vi_params.ptc_freeze_complement = bool(base_vi_ptc_freeze_complement)
            solver.vi_params.anderson_acceleration = bool(base_vi_anderson_acceleration)
            solver.vi_params.anderson_history = int(base_vi_anderson_history)
            solver.vi_params.anderson_regularization = float(base_vi_anderson_regularization)
            solver.vi_params.anderson_damping = float(base_vi_anderson_damping)
            solver.vi_params.anderson_stable_iters = int(base_vi_anderson_stable_iters)
            solver.vi_params.anderson_ginf_trigger = float(base_vi_anderson_ginf_trigger)
            solver.vi_params.anderson_gap_ratio = float(base_vi_anderson_gap_ratio)
            solver.vi_params.anderson_eq_abs = float(base_vi_anderson_eq_abs)
            solver.vi_params.anderson_ginf_max = float(base_vi_anderson_ginf_max)

    def _record_step(functions) -> None:
        step_no = int(getattr(solver, "_current_step_no", len(timeseries_rows) + 1))
        t_now = float(getattr(solver, "_current_t", 0.0) + getattr(solver, "_current_dt", float(args.dt)))
        delta_inf_raw = getattr(solver, "_last_accepted_step_delta_inf", None)
        if delta_inf_raw is None:
            delta_inf_raw = getattr(solver, "_last_nonlinear_update_inf", None)
        delta_inf = None
        if delta_inf_raw is not None:
            try:
                delta_inf_candidate = float(delta_inf_raw)
                if np.isfinite(delta_inf_candidate):
                    delta_inf = delta_inf_candidate
            except Exception:
                delta_inf = None
        startup_retry_state["last_accepted_step_no"] = int(step_no)
        startup_retry_state["last_accepted_delta_inf"] = delta_inf
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
        if io_root and int(args.vtk_every) > 0 and (step_no % int(args.vtk_every) == 0):
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
        _restore_base_monolithic_controls()

    solver_key = str(args.nonlinear_solver).strip().lower()
    alpha_bc_mode_key = str(args.alpha_bc_mode).strip().lower()
    if alpha_bc_mode_key == "auto":
        alpha_bc_mode_key = "natural"
    def _find_named_function(funcs_in, template):
        for f in list(funcs_in or []):
            if str(getattr(f, "name", "")) == str(getattr(template, "name", "")):
                return f
        return template

    def _latent_transformed_active_fields(fields_in) -> list[str]:
        fields = []
        latent_fields = set(problem.get("latent_bounded_fields", tuple()) or tuple())
        transformed = (
            bool(problem.get("latent_bounded_transport", False))
            and _latent_bounded_formulation_key(problem) == "transformed"
        )
        for name in list(fields_in or []):
            key = str(name).strip()
            if transformed and key == "alpha" and "alpha" in latent_fields:
                key = "alpha_latent"
            elif transformed and key == "phi" and "phi" in latent_fields:
                key = "phi_latent"
            if key and key not in fields:
                fields.append(key)
        return fields

    def _latent_transformed_solver_fields() -> list[str]:
        fields = list(getattr(problem["dh"], "field_names", []) or [])
        latent_fields = set(problem.get("latent_bounded_fields", tuple()) or tuple())
        if not (
            bool(problem.get("latent_bounded_transport", False))
            and _latent_bounded_formulation_key(problem) == "transformed"
        ):
            return [str(name) for name in fields]
        out = []
        for name in fields:
            key = str(name)
            if key == "alpha" and "alpha" in latent_fields:
                continue
            if key == "phi" and "phi" in latent_fields:
                continue
            out.append(key)
        return out

    def _sync_latent_transformed_fields(funcs_in) -> None:
        if not (
            bool(problem.get("latent_bounded_transport", False))
            and _latent_bounded_formulation_key(problem) == "transformed"
        ):
            return
        _sync_latent_bounded_problem_fields(
            problem=problem,
            funcs=funcs_in,
            find_named_function=_find_named_function,
        )

    def _configure_vi_box_bounds(target_solver, *, funcs_in=None) -> None:
        if solver_key not in {"pdas", "ipm"}:
            return
        if hasattr(target_solver, "set_box_bounds"):
            lower_full = np.full(int(problem["dh"].total_dofs), -np.inf, dtype=float)
            upper_full = np.full(int(problem["dh"].total_dofs), np.inf, dtype=float)
            any_box_bounds = False
            if bool(args.alpha_box_constraints) and not bool(getattr(args, "alpha_from_refmap", False)):
                _apply_field_box_bounds(
                    lower_full,
                    upper_full,
                    dof_handler=problem["dh"],
                    field_name="alpha",
                    lo=0.0,
                    hi=1.0,
                )
                any_box_bounds = True
            if bool(args.enable_phi_evolution) and problem["phi_k"] is not None and bool(args.phi_box_constraints):
                support_key = str(args.support_physics).strip().lower()
                if support_key == "internal_conversion":
                    freeze_bounds = bool(getattr(target_solver, "_benchmark7_freeze_support_phi_bounds", False))
                    phi_bounds_cache = None
                    phi_bounds_step = getattr(target_solver, "_benchmark7_support_phi_bounds_step", None)
                    current_step = int(getattr(target_solver, "_current_step_no", -1))
                    if freeze_bounds and int(phi_bounds_step if phi_bounds_step is not None else -2) == int(current_step):
                        phi_bounds_cache = getattr(target_solver, "_benchmark7_support_phi_bounds_cache", None)
                    if phi_bounds_cache is None:
                        alpha_ref = problem["alpha_k"]
                        if funcs_in is not None:
                            alpha_ref = _find_named_function(funcs_in, problem["alpha_k"])
                        phi_lower, phi_upper, _ = _build_support_aware_phi_box_bounds(
                            problem,
                            alpha_func=alpha_ref,
                            alpha_threshold=float(args.phi_box_alpha_threshold),
                        )
                        phi_bounds_cache = (np.asarray(phi_lower, dtype=float).copy(), np.asarray(phi_upper, dtype=float).copy())
                        if freeze_bounds:
                            target_solver._benchmark7_support_phi_bounds_step = int(current_step)
                            target_solver._benchmark7_support_phi_bounds_cache = (
                                phi_bounds_cache[0].copy(),
                                phi_bounds_cache[1].copy(),
                            )
                    phi_lower, phi_upper = phi_bounds_cache
                    lower_full = np.maximum(lower_full, phi_lower)
                    upper_full = np.minimum(upper_full, phi_upper)
                else:
                    _apply_field_box_bounds(
                        lower_full,
                        upper_full,
                        dof_handler=problem["dh"],
                        field_name="phi",
                        lo=0.0,
                        hi=1.0,
                    )
                any_box_bounds = True
            if any_box_bounds:
                target_solver.set_box_bounds(lower=lower_full, upper=upper_full)

    def _configure_vi_constraints(target_solver, *, funcs_in=None) -> None:
        _configure_vi_box_bounds(target_solver, funcs_in=funcs_in)
        if not hasattr(target_solver, "set_linear_equalities"):
            return
        equalities = _build_vi_linear_equalities(
            args=args,
            problem=problem,
            qdeg=int(qdeg),
            alpha_bc_mode_key=str(alpha_bc_mode_key),
            find_named_function=_find_named_function,
        )
        if equalities:
            target_solver.set_linear_equalities(equalities)

    def _make_solver(
        forms_obj,
        *,
        postproc_cb=None,
        max_newton_iter: int | None = None,
        accept_factor: float = 10.0,
        relaxed_accept_ginf: float | None = None,
        solver_kind: str | None = None,
        bcs_in=None,
        bcs_homog_in=None,
        freeze_support_phi_bounds: bool = False,
    ):
        solver_kind_key = str(solver_key if solver_kind is None else solver_kind).strip().lower()
        newton_params = NewtonParameters(
            newton_tol=float(args.newton_tol),
            newton_rtol=float(args.newton_rtol),
            max_newton_iter=int(args.max_it) if max_newton_iter is None else int(max_newton_iter),
            ls_mode=str(args.ls_mode),
            accept_nonconverged_atol_factor=float(accept_factor),
            globalization=str(getattr(args, "newton_globalization", "line_search")),
            tr_max_iter=int(getattr(args, "trust_max_it", 8)),
            tr_radius_init=float(getattr(args, "trust_radius_init", 1.0)),
            tr_radius_max=float(getattr(args, "trust_radius_max", 1.0e3)),
            tr_eta_accept=float(getattr(args, "trust_eta_accept", 1.0e-4)),
            tr_eta_contract=float(getattr(args, "trust_eta_contract", 2.5e-1)),
            tr_eta_expand=float(getattr(args, "trust_eta_expand", 7.5e-1)),
            tr_shrink=float(getattr(args, "trust_shrink", 2.5e-1)),
            tr_expand=float(getattr(args, "trust_expand", 2.0)),
            tr_min_radius=float(getattr(args, "trust_min_radius", 1.0e-10)),
            tr_min_abs_decrease_inf=float(getattr(args, "trust_min_abs_residual_drop", 0.0)),
            tr_min_rel_decrease_inf=float(getattr(args, "trust_min_rel_residual_drop", 0.0)),
            stall_window=int(getattr(args, "newton_stall_window", 0)),
            stall_min_abs_decrease_inf=float(getattr(args, "newton_stall_min_abs_residual_drop", 0.0)),
            stall_min_rel_decrease_inf=float(getattr(args, "newton_stall_min_rel_residual_drop", 0.0)),
            ptc_recovery=bool(args.vi_ptc_recovery),
            ptc_fields=_parse_csv_fields(args.vi_ptc_fields),
            ptc_sigma0=float(args.vi_ptc_sigma0),
            ptc_sigma_max=float(args.vi_ptc_sigma_max),
            ptc_growth=float(args.vi_ptc_growth),
            ptc_decay=float(args.vi_ptc_decay),
            ptc_freeze_complement=False,
            ptc_operator_mode=str(getattr(args, "newton_ptc_operator_mode", "row_normalized")),
            ptc_late_fields=_parse_csv_fields(getattr(args, "newton_ptc_late_fields", "")),
            ptc_late_switch_residual=float(getattr(args, "newton_ptc_late_switch_residual", 0.0)),
            ptc_late_operator_mode=str(getattr(args, "newton_ptc_late_operator_mode", "")),
        )
        newton_params.line_search = bool(args.line_search)
        common_solver_kwargs = dict(
            dof_handler=problem["dh"],
            mixed_element=problem["me"],
            bcs=(bcs if bcs_in is None else bcs_in),
            bcs_homog=(bcs_homog if bcs_homog_in is None else bcs_homog_in),
            newton_params=newton_params,
            lin_params=LinearSolverParameters(
                backend=str(args.linear_backend),
                tol=float(args.lin_tol),
                maxit=int(args.lin_maxit),
                distributed=bool(
                    MPI_CTX.enabled
                    and bool(getattr(args, "petsc_distributed", True))
                    and str(args.linear_backend).strip().lower() == "petsc"
                ),
            ),
            quad_order=int(qdeg),
            backend=str(args.backend),
            postproc_timeloop_cb=postproc_cb,
        )
        if solver_kind_key in {"pdas", "ipm"}:
            solver_cls = PdasNewtonSolver if solver_kind_key == "pdas" else InteriorPointNewtonSolver
            target_solver = solver_cls(
                forms_obj.residual_form,
                forms_obj.jacobian_form,
                vi_params=VIParameters(
                    c=float(args.vi_c),
                    enter_tol=float(args.vi_enter_tol),
                    leave_tol=float(args.vi_leave_tol),
                    active_set_persistence=int(args.vi_persistence),
                    project_initial_guess=True,
                    project_each_iteration=False,
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
                    bound_step_limit=True,
                    bound_step_tau=1.0,
                    bound_blocking_activate=True,
                    bound_blocking_trigger_alpha=0.95,
                    bound_blocking_max_iter=16,
                    relaxed_filter_accept_ginf=(
                        float(relaxed_accept_ginf)
                        if relaxed_accept_ginf is not None
                        else (0.0 if solver_kind_key == "ipm" else 1.0e-2)
                    ),
                    relaxed_filter_merit_growth=1.05,
                    unconstrained_lm=bool(args.vi_unconstrained_lm),
                    unconstrained_lm_lambda0=float(args.vi_lm_lambda0),
                    unconstrained_lm_lambda_max=float(args.vi_lm_lambda_max),
                    unconstrained_lm_growth=float(args.vi_lm_growth),
                    unconstrained_lm_decay=float(args.vi_lm_decay),
                    unconstrained_lm_accept_ratio=float(args.vi_lm_accept_ratio),
                    unconstrained_lm_good_ratio=float(args.vi_lm_good_ratio),
                    unconstrained_lm_max_tries=int(args.vi_lm_max_tries),
                    line_search_nonmonotone_window=0,
                    line_search_nonmonotone_active_stable_iters=0,
                    line_search_nonmonotone_ginf_trigger=0.0,
                    line_search_nonmonotone_gap_ratio=1.0,
                    line_search_nonmonotone_eq_abs=1.0e-10,
                    line_search_nonmonotone_disable_filter=False,
                    affine_identified_acceleration=False,
                    equation_row_scaling=True,
                    variable_column_scaling=bool(args.vi_variable_column_scaling),
                    variable_column_scaling_fields=_parse_csv_fields(args.vi_variable_column_scaling_fields),
                    ptc_recovery=bool(args.vi_ptc_recovery),
                    ptc_fields=_parse_csv_fields(args.vi_ptc_fields),
                    ptc_sigma0=float(args.vi_ptc_sigma0),
                    ptc_sigma_max=float(args.vi_ptc_sigma_max),
                    ptc_growth=float(args.vi_ptc_growth),
                    ptc_decay=float(args.vi_ptc_decay),
                    ptc_ginf_trigger=float(args.vi_ptc_ginf_trigger),
                    ptc_ginf_max=float(args.vi_ptc_ginf_max),
                    anderson_acceleration=bool(args.vi_anderson_acceleration),
                    anderson_history=int(args.vi_anderson_history),
                    anderson_regularization=float(args.vi_anderson_regularization),
                    anderson_damping=float(args.vi_anderson_damping),
                    interior_point_mu0=float(getattr(args, "vi_ipm_mu0", 1.0e-2)),
                    interior_point_mu_min=float(getattr(args, "vi_ipm_mu_min", 1.0e-10)),
                    interior_point_mu_decay=float(getattr(args, "vi_ipm_mu_decay", 0.2)),
                    interior_point_max_barrier_steps=int(getattr(args, "vi_ipm_max_barrier_steps", 12)),
                    interior_point_fraction_to_boundary=float(getattr(args, "vi_ipm_fraction_to_boundary", 0.995)),
                    interior_point_armijo_c1=float(getattr(args, "vi_ipm_armijo_c1", 1.0e-4)),
                    interior_point_step_reduction=float(getattr(args, "vi_ipm_step_reduction", 0.5)),
                    interior_point_step_min=float(getattr(args, "vi_ipm_step_min", 1.0e-10)),
                    interior_point_initial_push=float(getattr(args, "vi_ipm_initial_push", 1.0e-8)),
                    interior_point_stage_tol_factor=float(getattr(args, "vi_ipm_stage_tol_factor", 0.25)),
                ),
                **common_solver_kwargs,
            )
            target_solver._benchmark7_freeze_support_phi_bounds = bool(freeze_support_phi_bounds)
            target_solver._benchmark7_support_phi_bounds_step = None
            target_solver._benchmark7_support_phi_bounds_generation = 0
            target_solver._benchmark7_support_phi_bounds_cache = None

            def _reset_benchmark7_vi_bounds_freeze() -> None:
                target_solver._benchmark7_support_phi_bounds_step = None
                target_solver._benchmark7_support_phi_bounds_cache = None
                target_solver._benchmark7_support_phi_bounds_generation = int(
                    getattr(target_solver, "_benchmark7_support_phi_bounds_generation", 0)
                ) + 1

            target_solver._reset_benchmark7_vi_bounds_freeze = _reset_benchmark7_vi_bounds_freeze

            def _refresh_vi_constraints(funcs) -> None:
                _sync_latent_transformed_fields(funcs)
                _configure_vi_box_bounds(target_solver, funcs_in=funcs)

            target_solver.pre_cb = _refresh_vi_constraints
        else:
            target_solver = NewtonSolver(
                forms_obj.residual_form,
                forms_obj.jacobian_form,
                **common_solver_kwargs,
            )
            auto_equilibrate_newton = (
                bool(problem.get("latent_bounded_transport", False))
                and bool(getattr(args, "pressure_mean_constraint", False))
            )
            newton_row_scaling = bool(getattr(args, "newton_equation_row_scaling", False))
            newton_col_scaling = bool(getattr(args, "newton_variable_column_scaling", False))
            newton_scaling_mode = str(getattr(args, "newton_reduced_scaling_mode", "field") or "field")
            newton_ruiz_iters = int(getattr(args, "newton_ruiz_iters", 6) or 6)
            if auto_equilibrate_newton and (not newton_row_scaling) and (not newton_col_scaling):
                newton_row_scaling = True
                newton_col_scaling = True
                if str(newton_scaling_mode).strip().lower() == "field":
                    newton_scaling_mode = "ruiz"
                print(
                    "    [solver] auto-enabling reduced Newton equilibration for the latent monolithic branch "
                    f"(row=1, col=1, mode={newton_scaling_mode}, ruiz_iters={int(newton_ruiz_iters)})."
                )
            target_solver.set_reduced_system_scaling(
                equation_row_scaling=bool(newton_row_scaling),
                variable_column_scaling=bool(newton_col_scaling),
                variable_column_scaling_fields=_parse_csv_fields(
                    getattr(args, "newton_variable_column_scaling_fields", "")
                ),
                mode=str(newton_scaling_mode),
                ruiz_iters=int(newton_ruiz_iters),
            )
            if bool(newton_row_scaling) or bool(newton_col_scaling):
                print(
                    "    [solver] enabling Newton reduced-system scaling "
                    f"(row={int(bool(newton_row_scaling))}, "
                    f"col={int(bool(newton_col_scaling))}, "
                    f"mode={str(newton_scaling_mode)}, "
                    f"ruiz_iters={int(newton_ruiz_iters)}, "
                    f"fields={_parse_csv_fields(getattr(args, 'newton_variable_column_scaling_fields', '')) or ('<all>',)})."
                )
            if bool(getattr(args, "newton_pressure_schur_solve", False)):
                schur_fields = _parse_csv_fields(getattr(args, "newton_pressure_schur_fields", "")) or ("p", "p_mean")
                schur_scale_mode = str(getattr(args, "newton_pressure_schur_scale_mode", "none") or "none").strip().lower()
                schur_scale_value = float(getattr(args, "newton_pressure_schur_scale_value", 1.0) or 1.0)
                if schur_scale_mode in {"drag", "inv_drag"}:
                    schur_scale_value = max(float(getattr(args, "mu_f", 1.0)) / max(float(kappa), 1.0e-30), 1.0e-30)
                target_solver.set_reduced_schur_preconditioner(
                    enabled=True,
                    pressure_fields=schur_fields,
                    diag_only=bool(getattr(args, "newton_pressure_schur_diag_only", False)),
                    shift_rel=float(getattr(args, "newton_pressure_schur_shift_rel", 1.0e-12)),
                    pressure_scale_mode=schur_scale_mode,
                    pressure_scale_value=schur_scale_value,
                    trace=bool(getattr(args, "newton_pressure_schur_trace", False)),
                )
                print(
                    "    [solver] enabling reduced Newton pressure-Schur solve "
                    f"(fields={tuple(schur_fields)}, diag_only={int(bool(getattr(args, 'newton_pressure_schur_diag_only', False)))}, "
                    f"shift_rel={float(getattr(args, 'newton_pressure_schur_shift_rel', 1.0e-12)):.1e}, "
                    f"scale_mode={schur_scale_mode}, scale_value={schur_scale_value:.3e})."
                )
            if bool(getattr(args, "pressure_mean_gauge", False)):
                if bool(getattr(args, "pressure_mean_constraint", False)):
                    print("    [solver] skipping weighted pressure mean-value gauge because the clean p_mean constraint is active.")
                else:
                    p_weights_full = _assemble_field_integral_weights(
                        problem,
                        test_function=problem["q_test"],
                        quad_order=int(qdeg),
                        backend=str(args.backend),
                    )
                    target_solver.set_mean_value_gauge(
                        "p",
                        full_weights=p_weights_full,
                        coeff_name="p_k",
                        strength=float(args.pressure_mean_gauge_strength),
                    )
                    print(
                        "    [solver] enabling weighted pressure mean-value gauge on field 'p' "
                        f"with strength={float(args.pressure_mean_gauge_strength):.3e}."
                    )
            if bool(getattr(args, "logistic_bounded_transform", False)):
                requested_logistic_fields = _effective_logistic_bounded_fields(args)
                available_fields = {str(name) for name in getattr(problem["dh"], "field_names", [])}
                logistic_fields = [name for name in requested_logistic_fields if name in available_fields]
                skipped_logistic_fields = [name for name in requested_logistic_fields if name not in available_fields]
                if skipped_logistic_fields:
                    print(
                        "    [solver] skipping unavailable logit-coordinate fields "
                        f"{tuple(skipped_logistic_fields)}; available={tuple(sorted(available_fields))}."
                    )
                if not logistic_fields:
                    raise RuntimeError(
                        "The logistic-transform experiment requested bounded fields "
                        f"{tuple(requested_logistic_fields)}, but none are present in the active mixed space."
                    )
                target_solver.set_logistic_transform_fields(
                    logistic_fields,
                    eps=float(args.logistic_bounded_eps),
                )
                print(
                    "    [solver] using logit-coordinate Newton for bounded fields "
                    f"{tuple(logistic_fields)} with eps={float(args.logistic_bounded_eps):.1e}."
                )
            if (
                bool(problem.get("latent_bounded_transport", False))
                and _latent_bounded_formulation_key(problem) == "transformed"
            ):
                target_solver.pre_cb = _sync_latent_transformed_fields
        if alpha_from_refmap_enabled:
            prev_pre_cb = getattr(target_solver, "pre_cb", None)

            def _refmap_pre_cb(funcs) -> None:
                _sync_alpha_from_refmap()
                if callable(prev_pre_cb):
                    prev_pre_cb(funcs)

            target_solver.pre_cb = _refmap_pre_cb
        _bind_solver_inactive_solid_interface_retagging(
            problem=problem,
            target_solver=target_solver,
        )
        condition_balance_field_scales = dict(problem.get("_condition_balanced_field_scales", {}) or {})
        mixed_drag_condition_balanced = (
            str(args.mechanics_nondim_mode).strip().lower() == "condition_balanced"
            and str(args.drag_formulation).strip().lower().replace("-", "_") == "mixed_lm"
        )
        if condition_balance_field_scales:
            target_solver.set_manual_reduced_system_scaling(
                equation_row_field_scales=condition_balance_field_scales,
                variable_column_field_scales=condition_balance_field_scales,
            )
            target_solver.set_reduced_system_scaling(
                equation_row_scaling=True,
                variable_column_scaling=True,
                variable_column_scaling_fields=tuple(sorted(condition_balance_field_scales)),
                mode=str(getattr(target_solver, "_reduced_system_scaling_mode", "field") or "field"),
                ruiz_iters=int(getattr(target_solver, "_reduced_scaling_ruiz_iters", 6) or 6),
            )
            if hasattr(target_solver, "vi_params") and getattr(target_solver, "vi_params", None) is not None:
                target_solver.vi_params.equation_row_scaling = True
                target_solver.vi_params.variable_column_scaling = True
            print(
                "    [solver] enabling fixed condition-balanced coordinate scales "
                f"{condition_balance_field_scales}."
            )
        elif mixed_drag_condition_balanced:
            ruiz_iters = max(int(getattr(target_solver, "_reduced_scaling_ruiz_iters", 8) or 8), 8)
            target_solver.set_reduced_system_scaling(
                equation_row_scaling=True,
                variable_column_scaling=True,
                mode="ruiz",
                ruiz_iters=ruiz_iters,
            )
            if hasattr(target_solver, "vi_params") and getattr(target_solver, "vi_params", None) is not None:
                target_solver.vi_params.equation_row_scaling = True
                target_solver.vi_params.variable_column_scaling = True
            print(
                "    [solver] enabling Ruiz reduced-system scaling for the "
                "condition-balanced mixed-drag branch."
            )
        def _postprocess_physical_bounds(funcs) -> None:
            _sync_latent_transformed_fields(funcs)
            _apply_open_top_global_phi_cleanup(
                args=args,
                problem=problem,
                funcs=funcs,
                find_named_function=_find_named_function,
            )
            if alpha_from_refmap_enabled:
                _sync_alpha_from_refmap()
            if getattr(target_solver, "constraints", None) is not None:
                target_solver._enforce_constraints_on_functions(funcs)
            reset_bounds_cb = getattr(target_solver, "_reset_benchmark7_vi_bounds_freeze", None)
            if callable(reset_bounds_cb):
                reset_bounds_cb()

        target_solver.post_accept_cb = _postprocess_physical_bounds
        if alpha_from_refmap_enabled:
            prev_preassemble_cb = getattr(target_solver, "preassemble_cb", None)

            def _refmap_preassemble(coeffs) -> None:
                if callable(prev_preassemble_cb):
                    prev_preassemble_cb(coeffs)
                _sync_alpha_from_refmap()

            target_solver.preassemble_cb = _refmap_preassemble
        _configure_vi_constraints(target_solver)
        return target_solver

    main_accept_factor = 10.0
    if _logistic_refmap_phi_only_mode(args) and _predictor_corrector_startup_enabled(args):
        main_accept_factor = max(float(main_accept_factor), 150.0)
        print(
            "    [solver] allowing nonconverged exact accepts up to "
            f"{float(main_accept_factor):.0f}·atol on the refmap phi-only logistic startup branch."
        )

    solver = _make_solver(
        forms,
        postproc_cb=_record_step,
        max_newton_iter=int(args.max_it),
        accept_factor=float(main_accept_factor),
        freeze_support_phi_bounds=True,
    )
    if (
        bool(problem.get("latent_bounded_transport", False))
        and _latent_bounded_formulation_key(problem) == "transformed"
    ):
        transformed_fields = _latent_transformed_solver_fields()
        _set_solver_active_fields_with_tracking(solver, transformed_fields)
        print(
            "    [solver] using transformed latent bounded formulation with active fields "
            f"{tuple(transformed_fields)} and map='{_latent_bounded_map_key(problem)}'."
        )
    base_solver_relaxed_accept_ginf = float(
        getattr(getattr(solver, "vi_params", None), "relaxed_filter_accept_ginf", 0.0) or 0.0
    )

    def _arm_startup_monolithic_budget(*, step_no: int, reason: str) -> None:
        if int(step_no) != 1:
            return
        boosted = int(startup_monolithic_budget)
        current_budget = int(getattr(solver.np, "max_newton_iter", base_solver_max_newton_iter) or base_solver_max_newton_iter)
        if boosted > int(base_solver_max_newton_iter) and current_budget != boosted:
            solver.np.max_newton_iter = boosted
            print(
                f"    [startup] extending first-step monolithic Newton budget to {boosted} iterations "
                f"after {reason}."
            )
        if getattr(solver, "vi_params", None) is not None:
            is_ipm_solver = isinstance(solver, InteriorPointNewtonSolver)
            if not is_ipm_solver:
                relaxed_boost = max(float(base_solver_relaxed_accept_ginf), 2.0e-2)
                relaxed_now = float(getattr(solver.vi_params, "relaxed_filter_accept_ginf", 0.0) or 0.0)
                if relaxed_now < relaxed_boost:
                    solver.vi_params.relaxed_filter_accept_ginf = float(relaxed_boost)
                    print(
                        "    [startup] relaxing first-step semismooth accept threshold to "
                        f"|G|_∞<={relaxed_boost:.3e} after {reason}."
                    )
            # First-step-only PDAS tuning for the stiff monolithic restart:
            # keep the model fixed, but reduce active-set chatter and let the
            # bounded fields choose field-wise c from the current Jacobian.
            solver.vi_params.c = 0.0
            solver.vi_params.c_by_field = {}
            solver.vi_params.enter_tol = max(float(base_vi_enter_tol), 1.0e-6)
            solver.vi_params.leave_tol = max(float(base_vi_leave_tol), float(solver.vi_params.enter_tol))
            solver.vi_params.active_set_persistence = max(int(base_vi_persistence), 1)
            solver.vi_params.active_step_delta_active_trigger = max(int(base_vi_active_soft_threshold), 12)
            solver.vi_params.active_step_soft_alpha = min(float(base_vi_active_soft_alpha), 0.35)
            solver.vi_params.equality_active_step_ginf_threshold = float("inf")
            solver.vi_params.affine_cycle_fallback = True
            solver.vi_params.affine_identified_acceleration = False
            # Globalization from the literature: once the imported startup active
            # set is already stable and the complementarity/equality parts are
            # small, switch from a strictly monotone merit test to a short
            # nonmonotone Armijo window so the local semismooth-Newton regime is
            # not strangled by tiny backtracking steps.
            solver.vi_params.line_search_nonmonotone_window = max(int(base_vi_nonmono_window), 5)
            solver.vi_params.line_search_nonmonotone_active_stable_iters = max(
                int(base_vi_nonmono_stable_iters), 1
            )
            solver.vi_params.line_search_nonmonotone_ginf_trigger = max(
                float(base_vi_nonmono_ginf_trigger), 3.0e-1
            )
            solver.vi_params.line_search_nonmonotone_gap_ratio = max(
                float(base_vi_nonmono_gap_ratio), 1.0
            )
            solver.vi_params.line_search_nonmonotone_eq_abs = max(
                float(base_vi_nonmono_eq_abs), 1.0e-10
            )
            solver.vi_params.line_search_nonmonotone_disable_filter = True
            solver.vi_params.accept_best_filtered_descent = True
            # The corrected Benchmark 7 path still needs a wider LM trust-region
            # window on the first monolithic step: several kappa=1e-3/1e-4
            # failures only recover once the PDAS LM branch is allowed to grow
            # past the default six shifts.
            solver.vi_params.unconstrained_lm_max_tries = max(int(base_vi_lm_max_tries), 12)
            solver.vi_params.unconstrained_lm_lambda_max = max(float(base_vi_lm_lambda_max), 1.0e10)
            if is_ipm_solver:
                # The interior-point path benefits from the larger first-step LM
                # window, but the semismooth rescue package below is counter-
                # productive here: it pre-arms mechanics-only PTC/prox phases
                # that were designed for PDAS working-set recovery and can
                # override user choices such as --no-vi-ptc-recovery.
                solver.vi_params.ptc_recovery = bool(base_vi_ptc_recovery)
                solver.vi_params.ptc_fields = tuple(base_vi_ptc_fields)
                solver.vi_params.ptc_sigma0 = float(base_vi_ptc_sigma0)
                solver.vi_params.ptc_sigma_max = float(base_vi_ptc_sigma_max)
                solver.vi_params.ptc_growth = float(base_vi_ptc_growth)
                solver.vi_params.ptc_decay = float(base_vi_ptc_decay)
                solver.vi_params.ptc_stable_iters = int(base_vi_ptc_stable_iters)
                solver.vi_params.ptc_ginf_trigger = float(base_vi_ptc_ginf_trigger)
                solver.vi_params.ptc_gap_ratio = float(base_vi_ptc_gap_ratio)
                solver.vi_params.ptc_eq_abs = float(base_vi_ptc_eq_abs)
                solver.vi_params.ptc_ginf_max = float(base_vi_ptc_ginf_max)
                solver.vi_params.ptc_identified_window = bool(base_vi_ptc_identified_window)
                solver.vi_params.ptc_freeze_complement = bool(base_vi_ptc_freeze_complement)
                solver.vi_params.field_proximal_recovery = bool(base_vi_field_proximal_recovery)
                solver.vi_params.field_proximal_recovery_fields = tuple(base_vi_field_proximal_recovery_fields)
                solver.vi_params.field_proximal_recovery_lambda0 = float(base_vi_field_proximal_recovery_lambda0)
                solver.vi_params.field_proximal_recovery_lambda_max = float(base_vi_field_proximal_recovery_lambda_max)
                solver.vi_params.field_proximal_recovery_growth = float(base_vi_field_proximal_recovery_growth)
                solver.vi_params.field_proximal_recovery_max_tries = int(base_vi_field_proximal_recovery_max_tries)
                solver.vi_params.field_proximal_recovery_stable_iters = int(base_vi_field_proximal_recovery_stable_iters)
                solver.vi_params.field_proximal_recovery_ginf_trigger = float(base_vi_field_proximal_recovery_ginf_trigger)
                solver.vi_params.field_proximal_recovery_gap_ratio = float(base_vi_field_proximal_recovery_gap_ratio)
                solver.vi_params.field_proximal_recovery_eq_abs = float(base_vi_field_proximal_recovery_eq_abs)
                solver.vi_params.field_proximal_recovery_identified_window = bool(base_vi_field_proximal_recovery_identified_window)
                solver.vi_params.field_proximal_recovery_ginf_max = float(base_vi_field_proximal_recovery_ginf_max)
                solver.vi_params.anderson_acceleration = bool(base_vi_anderson_acceleration)
                solver.vi_params.anderson_history = int(base_vi_anderson_history)
                solver.vi_params.anderson_regularization = float(base_vi_anderson_regularization)
                solver.vi_params.anderson_damping = float(base_vi_anderson_damping)
                solver.vi_params.anderson_stable_iters = int(base_vi_anderson_stable_iters)
                solver.vi_params.anderson_ginf_trigger = float(base_vi_anderson_ginf_trigger)
                solver.vi_params.anderson_gap_ratio = float(base_vi_anderson_gap_ratio)
                solver.vi_params.anderson_eq_abs = float(base_vi_anderson_eq_abs)
                solver.vi_params.anderson_ginf_max = float(base_vi_anderson_ginf_max)
                solver._vi_force_initial_field_prox_lambda = 0.0
                solver._vi_force_initial_ptc_sigma = 0.0
                return
            # The staggered startup already imports a constrained transport
            # active set into the first monolithic retry. Requiring two extra
            # stable semismooth iterations and |G|<=5e-2 before trying the
            # local affine solve is too strict for backend-sensitive cases:
            # cpp/scipy and cpp/pardiso can fail at the second monolithic
            # iteration with |G| still around 6e-2..8e-2, so the intended
            # identified-active-set acceleration never fires. Relax the trigger
            # only for this first-step restarted solve.
            solver.vi_params.affine_identified_stable_iters = 1
            solver.vi_params.affine_identified_ginf_trigger = max(
                float(base_vi_affine_identified_ginf_trigger),
                1.5,
            )
            solver.vi_params.line_search_nonmonotone_ginf_trigger = max(
                float(solver.vi_params.line_search_nonmonotone_ginf_trigger),
                1.5,
            )
            # SQP-style working-set continuation: after an accepted local
            # affine rescue, keep that rescued active set fixed for a couple of
            # outer VI iterations so the monolithic restart settles before
            # allowing fresh active-set entries.
            solver.vi_params.working_set_guard_after_affine = max(
                int(base_vi_working_set_guard_after_affine),
                2,
            )
            # Stabilized SQP / proximal-point rescue on the coupled flow/
            # pressure/skeleton block. This only arms once the transport
            # active set is already stable, so it acts as a local trust-region
            # step on the hard inactive block rather than another transport
            # globalization tweak.
            solver.vi_params.field_proximal_recovery = True
            solver.vi_params.field_proximal_recovery_fields = ("v_x", "v_y", "p", "vS_x", "vS_y")
            solver.vi_params.field_proximal_recovery_lambda0 = max(
                float(base_vi_field_proximal_recovery_lambda0),
                1.0e-2,
            )
            solver.vi_params.field_proximal_recovery_lambda_max = max(
                float(base_vi_field_proximal_recovery_lambda_max),
                1.0e8,
            )
            solver.vi_params.field_proximal_recovery_growth = max(
                float(base_vi_field_proximal_recovery_growth),
                5.0,
            )
            solver.vi_params.field_proximal_recovery_max_tries = max(
                int(base_vi_field_proximal_recovery_max_tries),
                6,
            )
            solver.vi_params.field_proximal_recovery_stable_iters = 0
            prox_ginf_trigger = float(base_vi_field_proximal_recovery_ginf_trigger)
            if not np.isfinite(prox_ginf_trigger) or prox_ginf_trigger <= 0.0:
                prox_ginf_trigger = 5.0e-2
            solver.vi_params.field_proximal_recovery_ginf_trigger = min(
                prox_ginf_trigger,
                4.0e-2,
            )
            solver.vi_params.field_proximal_recovery_gap_ratio = max(
                float(base_vi_field_proximal_recovery_gap_ratio),
                1.0,
            )
            solver.vi_params.field_proximal_recovery_eq_abs = max(
                float(base_vi_field_proximal_recovery_eq_abs),
                1.0e-10,
            )
            solver.vi_params.field_proximal_recovery_identified_window = True
            solver.vi_params.field_proximal_recovery_ginf_max = 2.0e-1
            # True identified-manifold local solve on the mechanics block:
            # freeze transport, regularize the mechanics saddle block with a
            # pseudo-time term, and let Anderson accelerate the resulting fixed
            # local map once the active set is stable.
            solver.vi_params.ptc_recovery = True
            solver.vi_params.ptc_fields = ("v_x", "v_y", "p", "vS_x", "vS_y", "u_x", "u_y")
            solver.vi_params.ptc_sigma0 = max(float(base_vi_ptc_sigma0), 5.0e-2)
            solver.vi_params.ptc_sigma_max = max(float(base_vi_ptc_sigma_max), 1.0e8)
            solver.vi_params.ptc_growth = max(float(base_vi_ptc_growth), 5.0)
            solver.vi_params.ptc_decay = min(max(float(base_vi_ptc_decay), 0.0), 0.5)
            solver.vi_params.ptc_stable_iters = 0
            ptc_ginf_trigger = float(base_vi_ptc_ginf_trigger)
            if not np.isfinite(ptc_ginf_trigger) or ptc_ginf_trigger <= 0.0:
                ptc_ginf_trigger = 5.0e-2
            solver.vi_params.ptc_ginf_trigger = min(ptc_ginf_trigger, 4.0e-2)
            solver.vi_params.ptc_gap_ratio = max(float(base_vi_ptc_gap_ratio), 1.0)
            solver.vi_params.ptc_eq_abs = max(float(base_vi_ptc_eq_abs), 1.0e-10)
            solver.vi_params.ptc_identified_window = True
            solver.vi_params.ptc_ginf_max = max(float(base_vi_ptc_ginf_max), 2.0e-1)
            solver.vi_params.ptc_freeze_complement = True
            solver.vi_params.anderson_acceleration = True
            solver.vi_params.anderson_history = max(int(base_vi_anderson_history), 3)
            solver.vi_params.anderson_regularization = max(
                float(base_vi_anderson_regularization),
                1.0e-10,
            )
            solver._vi_force_initial_field_prox_lambda = float(
                solver.vi_params.field_proximal_recovery_lambda0
            )
            solver._vi_force_initial_ptc_sigma = float(
                solver.vi_params.ptc_sigma0
            )
            solver.vi_params.anderson_damping = min(
                max(float(base_vi_anderson_damping), 0.25),
                0.9,
            )
            solver.vi_params.anderson_stable_iters = 1
            solver.vi_params.anderson_ginf_trigger = max(
                float(base_vi_anderson_ginf_trigger),
                4.0e-2,
            )
            solver.vi_params.anderson_gap_ratio = max(float(base_vi_anderson_gap_ratio), 1.0)
            solver.vi_params.anderson_eq_abs = max(float(base_vi_anderson_eq_abs), 1.0e-10)
            solver.vi_params.anderson_ginf_max = max(float(base_vi_anderson_ginf_max), 2.0e-1)

    def _on_dt_change(new_dt: float) -> None:
        dt_c.value = float(new_dt)
        try:
            reset_bounds_cb = getattr(solver, "_reset_benchmark7_vi_bounds_freeze", None)
        except Exception:
            reset_bounds_cb = None
        if callable(reset_bounds_cb):
            reset_bounds_cb()
        # A reduced dt changes the transient operator materially. Reusing the
        # previous step-scoped staged guess at a new dt can poison the retry,
        # especially on step 1 where the staggered preload is the main
        # globalization device. Clear the per-step "already applied" latches so
        # the next attempt rebuilds a fresh staged initial guess for the new dt.
        startup_retry_state["startup_guess_applied_step_no"] = None
        startup_retry_state["later_step_stage_guess_applied_step_no"] = None
        startup_retry_state["bootstrap_attempts"] = 0
        startup_retry_state["post_guess_retry_attempts"] = 0
        startup_retry_state["near_converged_retry_attempts"] = 0
        startup_retry_state["later_step_stage_retry_attempts"] = 0
        startup_retry_state["frozen_transport_retry_attempts"] = 0
        startup_retry_state["pc_p2_reentry_retry_attempts"] = 0

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
    if problem.get("p_mean_k") is not None:
        current_functions.insert(2, problem["p_mean_k"])
        previous_functions.insert(2, problem["p_mean_n"])
    if problem.get("alpha_mass_lm_k") is not None:
        current_functions.insert(3, problem["alpha_mass_lm_k"])
        previous_functions.insert(3, problem["alpha_mass_lm_n"])
    if problem.get("pi_s_k") is not None:
        current_functions.insert(4, problem["pi_s_k"])
        previous_functions.insert(4, problem["pi_s_n"])
    if problem.get("lambda_drag_k") is not None:
        drag_insert = current_functions.index(problem["alpha_k"])
        prev_drag_insert = previous_functions.index(problem["alpha_n"])
        current_functions.insert(drag_insert, problem["lambda_drag_k"])
        previous_functions.insert(prev_drag_insert, problem["lambda_drag_n"])
    if problem.get("alpha_latent_k") is not None:
        current_functions.append(problem["alpha_latent_k"])
        previous_functions.append(problem["alpha_latent_n"])
    if problem["phi_k"] is not None:
        current_functions.extend([problem["phi_k"], problem["S_k"]])
        previous_functions.extend([problem["phi_n"], problem["S_n"]])
    if problem.get("phi_latent_k") is not None:
        current_functions.append(problem["phi_latent_k"])
        previous_functions.append(problem["phi_latent_n"])
    aux_solver_functions: dict[str, object] = {"dt": dt_c}
    if problem.get("reg_weight") is not None:
        aux_solver_functions["reg_weight"] = problem["reg_weight"]
    alpha_weights_full_pc = _assemble_field_integral_weights(
        problem,
        test_function=problem["alpha_test"],
        quad_order=int(qdeg),
        backend=str(args.backend),
    )

    startup_retry_state = {
        "step_no": None,
        "bootstrap_attempts": 0,
        "startup_guess_applied_step_no": None,
        "later_step_stage_guess_applied_step_no": None,
        "post_guess_retry_attempts": 0,
        "near_converged_retry_attempts": 0,
        "later_step_stage_retry_attempts": 0,
        "frozen_transport_retry_attempts": 0,
        "pc_p2_reentry_retry_attempts": 0,
        "pc_last_successful_lambda": None,
        "last_accepted_step_no": None,
        "last_accepted_delta_inf": None,
    }
    bootstrap_solver_cache: dict[str, object] = {}
    startup_stage_solver_cache: dict[str, object] = {}
    startup_stage_relaxed_ginf = float(_startup_stage_relaxed_accept_ginf(args))

    def _copy_prev_into_current(funcs, prev_funcs) -> None:
        for f, f_prev in zip(list(funcs or []), list(prev_funcs or [])):
            f.nodal_values[:] = f_prev.nodal_values[:]

    def _snapshot_function_values(funcs):
        return tuple(np.asarray(f.nodal_values, dtype=float).copy() for f in list(funcs or []))

    def _restore_function_values(funcs, snapshot) -> None:
        if snapshot is None:
            return
        for f, values in zip(list(funcs or []), list(snapshot or [])):
            f.nodal_values[:] = np.asarray(values, dtype=float)

    def _current_alpha_mass_relative_defect(*, funcs, prev_funcs) -> float:
        try:
            alpha_cur = _find_named_function(funcs, problem["alpha_k"])
            alpha_prev = _find_named_function(prev_funcs, problem["alpha_n"])
            mass_cur = float(alpha_weights_full_pc @ _function_to_full_vector(problem["dh"], alpha_cur))
            mass_prev = float(alpha_weights_full_pc @ _function_to_full_vector(problem["dh"], alpha_prev))
            return float((mass_cur - mass_prev) / max(abs(mass_prev), 1.0e-30))
        except Exception:
            return float("nan")

    def _predictor_alpha_mass_ok(stats: dict[str, float]) -> bool:
        if not (
            bool(problem.get("latent_bounded_transport", False))
            and "alpha" in tuple(problem.get("latent_bounded_fields", tuple()) or tuple())
            and problem.get("alpha_latent_k") is not None
        ):
            return True
        mass_rel = float(stats.get("alpha_mass_rel", float("nan")))
        if not np.isfinite(mass_rel):
            return False
        return bool(abs(mass_rel) <= max(float(getattr(args, "pc_alpha_mass_tol", 1.0e-10) or 1.0e-10), 0.0))

    def _predictor_alpha_mass_target(*, prev_funcs) -> float:
        alpha_prev = _find_named_function(prev_funcs, problem["alpha_n"])
        return float(alpha_weights_full_pc @ _function_to_full_vector(problem["dh"], alpha_prev))

    def _predictor_alpha_return_setup() -> dict[str, np.ndarray] | None:
        if not (
            bool(problem.get("latent_bounded_transport", False))
            and "alpha" in tuple(problem.get("latent_bounded_fields", tuple()) or tuple())
            and problem.get("alpha_latent_k") is not None
        ):
            return None
        cache = problem.setdefault("_pc_alpha_return_map_setup", {})
        cached = cache.get("default")
        if isinstance(cached, dict):
            return cached
        alpha_template = problem.get("alpha_k")
        alpha_latent_template = problem.get("alpha_latent_k")
        if alpha_template is None or alpha_latent_template is None:
            return None
        alpha_g = np.asarray(getattr(alpha_template, "_g_dofs", np.array([], dtype=int)), dtype=int).ravel()
        latent_g = np.asarray(getattr(alpha_latent_template, "_g_dofs", np.array([], dtype=int)), dtype=int).ravel()
        if alpha_g.size == 0 or latent_g.size == 0:
            return None
        perm_target_to_source = _field_coordinate_permutation(
            problem,
            source_field="alpha_latent",
            target_field="alpha",
        )
        if int(perm_target_to_source.size) != int(alpha_g.size):
            raise ValueError(
                "Alpha predictor return map permutation size mismatch: "
                f"target permutation has {int(perm_target_to_source.size)} entries, "
                f"alpha field has {int(alpha_g.size)} DOFs."
            )
        weights_target = np.asarray(alpha_weights_full_pc[alpha_g], dtype=float).ravel()
        weights_source = np.zeros(int(latent_g.size), dtype=float)
        np.add.at(weights_source, perm_target_to_source, weights_target)
        setup = {
            "alpha_g": np.asarray(alpha_g, dtype=int).copy(),
            "latent_g": np.asarray(latent_g, dtype=int).copy(),
            "weights_source": np.asarray(weights_source, dtype=float).copy(),
        }
        cache["default"] = setup
        return setup

    def _apply_predictor_alpha_mass_return_map(
        *,
        funcs,
        prev_funcs,
        bcs_now,
        stage_label: str,
        verbose: bool = False,
    ) -> dict[str, object]:
        setup = _predictor_alpha_return_setup()
        if setup is None:
            return {
                "applied": False,
                "converged": True,
                "shift": 0.0,
                "alpha_mass_rel": _current_alpha_mass_relative_defect(funcs=funcs, prev_funcs=prev_funcs),
            }
        alpha_latent = _find_named_function(funcs, problem["alpha_latent_k"])
        if alpha_latent is None:
            return {
                "applied": False,
                "converged": True,
                "shift": 0.0,
                "alpha_mass_rel": _current_alpha_mass_relative_defect(funcs=funcs, prev_funcs=prev_funcs),
            }
        latent_vals = np.asarray(alpha_latent.nodal_values, dtype=float).ravel()
        latent_g = np.asarray(setup["latent_g"], dtype=int).ravel()
        if latent_vals.size != latent_g.size:
            raise ValueError(
                f"Predictor alpha return map size mismatch: alpha_latent has {int(latent_vals.size)} values, "
                f"but cached latent_g has {int(latent_g.size)}."
            )
        blocked = set(int(d) for d in list(problem["dh"].dof_tags.get("inactive", set()) or set()))
        alpha_bcs = _filter_bcs_by_fields(bcs_now, ("alpha", "alpha_latent"))
        try:
            blocked.update(int(d) for d in dict(problem["dh"].get_dirichlet_data(alpha_bcs)).keys())
        except Exception:
            pass
        if blocked:
            blocked_arr = np.fromiter(sorted(blocked), dtype=int)
            free_mask = ~np.isin(latent_g, blocked_arr)
        else:
            free_mask = np.ones(int(latent_g.size), dtype=bool)
        rm = _latent_mass_return_shift(
            latent_vals,
            local_weights=np.asarray(setup["weights_source"], dtype=float),
            target_mass=float(_predictor_alpha_mass_target(prev_funcs=prev_funcs)),
            map_kind=_latent_bounded_map_key(problem),
            free_mask=free_mask,
            tol_rel=max(float(getattr(args, "pc_alpha_mass_tol", 1.0e-10) or 1.0e-10), 0.0),
            max_it=max(1, int(getattr(args, "pc_alpha_return_map_max_it", 64))),
        )
        alpha_latent.nodal_values[:] = np.asarray(rm["latent_values"], dtype=float).ravel()
        _sync_latent_bounded_problem_fields(problem=problem, funcs=funcs, find_named_function=_find_named_function)
        problem["dh"].apply_bcs(bcs_now, *funcs)
        if getattr(solver, "constraints", None) is not None:
            solver._enforce_constraints_on_functions(funcs)
        _sync_latent_bounded_problem_fields(problem=problem, funcs=funcs, find_named_function=_find_named_function)
        alpha_mass_rel = _current_alpha_mass_relative_defect(funcs=funcs, prev_funcs=prev_funcs)
        result = {
            "applied": True,
            "converged": bool(rm.get("converged", False)),
            "bracketed": bool(rm.get("bracketed", False)),
            "shift": float(rm.get("shift", 0.0)),
            "defect": float(rm.get("defect", float("nan"))),
            "alpha_mass_rel": float(alpha_mass_rel),
        }
        if bool(verbose) and stage_label:
            print(
                f"    [pc] {stage_label} alpha return map shift={float(result['shift']):.3e}, "
                f"alpha_mass_rel={float(result['alpha_mass_rel']):.3e}, "
                f"converged={int(bool(result['converged']))}."
            )
        return result

    def _snapshot_to_vector(snapshot) -> np.ndarray:
        parts = [np.asarray(values, dtype=float).ravel() for values in list(snapshot or [])]
        if not parts:
            return np.zeros((0,), dtype=float)
        return np.hstack(parts)

    def _vector_to_snapshot(vec: np.ndarray, template_snapshot) -> tuple[np.ndarray, ...]:
        vec = np.asarray(vec, dtype=float).ravel()
        out: list[np.ndarray] = []
        pos = 0
        for values in list(template_snapshot or []):
            arr = np.asarray(values, dtype=float)
            n = int(arr.size)
            out.append(np.asarray(vec[pos:pos + n], dtype=float).reshape(arr.shape).copy())
            pos += n
        if pos != int(vec.size):
            raise ValueError("Snapshot vector size does not match the template snapshot.")
        return tuple(out)

    def _anderson_mix_fixed_point(pairs) -> np.ndarray | None:
        items = list(pairs or [])
        if len(items) < 2:
            return None
        G = np.column_stack([np.asarray(item["g"], dtype=float).ravel() for item in items])
        F = np.column_stack(
            [
                np.asarray(item["g"], dtype=float).ravel() - np.asarray(item["x"], dtype=float).ravel()
                for item in items
            ]
        )
        if not np.all(np.isfinite(G)) or not np.all(np.isfinite(F)):
            return None
        reg = max(
            float(getattr(getattr(solver, "vi_params", None), "anderson_regularization", float(args.vi_anderson_regularization)) or float(args.vi_anderson_regularization)),
            0.0,
        )
        m = int(F.shape[1])
        kkt = np.zeros((m + 1, m + 1), dtype=float)
        kkt[:m, :m] = np.asarray(F.T @ F, dtype=float)
        if reg > 0.0:
            kkt[:m, :m] += reg * np.eye(m, dtype=float)
        kkt[:m, m] = 1.0
        kkt[m, :m] = 1.0
        rhs = np.zeros((m + 1,), dtype=float)
        rhs[m] = 1.0
        try:
            sol = np.linalg.solve(kkt, rhs)
        except np.linalg.LinAlgError:
            return None
        alpha = np.asarray(sol[:m], dtype=float)
        if not np.all(np.isfinite(alpha)):
            return None
        z_mix = np.asarray(G @ alpha, dtype=float).ravel()
        z_curr = np.asarray(items[-1]["x"], dtype=float).ravel()
        damping = min(
            max(
                float(getattr(getattr(solver, "vi_params", None), "anderson_damping", float(args.vi_anderson_damping)) or float(args.vi_anderson_damping)),
                0.0,
            ),
            1.0,
        )
        if damping < 1.0:
            z_mix = z_curr + damping * (z_mix - z_curr)
        if not np.all(np.isfinite(z_mix)):
            return None
        return z_mix

    def _relax_staggered_preload_toward_previous(
        funcs,
        prev_funcs,
        *,
        blend: float,
        keep_fields: list[str] | tuple[str, ...],
    ) -> bool:
        try:
            theta = float(blend)
        except Exception:
            theta = 1.0
        theta = min(max(theta, 0.0), 1.0)
        if theta >= 1.0 - 1.0e-14:
            return False
        keep = {str(name) for name in list(keep_fields or [])}
        changed = False
        for f, f_prev in zip(list(funcs or []), list(prev_funcs or [])):
            field_name = str(getattr(f, "field_name", "") or "")
            if field_name in keep:
                continue
            cur = np.asarray(f.nodal_values, dtype=float)
            prev = np.asarray(f_prev.nodal_values, dtype=float)
            changed = changed or bool(np.any(np.abs(cur - prev) > 1.0e-14))
            f.nodal_values[:] = prev + theta * (cur - prev)
        return bool(changed)

    def _filter_bcs_by_fields(bcs_in, field_names) -> list[BoundaryCondition]:
        allowed = {str(name) for name in list(field_names or [])}
        return [bc for bc in list(bcs_in or []) if str(getattr(bc, "field", "")) in allowed]

    def _startup_flow_stage_fields() -> list[str]:
        if str(problem["fluid_space"]).strip().lower() == "hdiv":
            fields = ["v", "p"]
        else:
            fields = ["v_x", "v_y", "p"]
        if problem.get("p_mean_k") is not None:
            fields.append("p_mean")
        return fields

    def _startup_transport_stage_fields() -> list[str]:
        fields = ["alpha", "mu_alpha"]
        if problem.get("alpha_mass_lm_k") is not None:
            fields.append("alpha_mass_lm")
        if problem.get("alpha_latent_k") is not None:
            fields.append("alpha_latent")
        if problem["phi_k"] is not None:
            fields.append("phi")
            if problem.get("phi_latent_k") is not None:
                fields.append("phi_latent")
            if problem.get("S_k") is not None:
                fields.append("S")
        return _latent_transformed_active_fields(fields)

    def _startup_fluid_stage_fields() -> list[str]:
        fields = _startup_flow_stage_fields()
        fields.extend(_startup_transport_stage_fields())
        return _latent_transformed_active_fields(fields)

    def _startup_solid_stage_fields() -> list[str]:
        fields = ["vS_x", "vS_y", "u_x", "u_y"]
        if problem.get("lambda_drag_k") is not None:
            fields.extend(["lambda_drag_x", "lambda_drag_y"])
        if problem.get("pi_s_k") is not None:
            fields.append("pi_s")
        return _latent_transformed_active_fields(fields)

    def _startup_predictor_p1_fields() -> list[str]:
        fields: list[str] = []
        for name in _startup_flow_stage_fields() + _startup_solid_stage_fields():
            if name not in fields:
                fields.append(name)
        field_keys = {
            "alpha": "alpha_k",
            "mu_alpha": "mu_k",
            "alpha_mass_lm": "alpha_mass_lm_k",
            "alpha_latent": "alpha_latent_k",
        }
        for name, key in field_keys.items():
            if problem.get(key) is not None and name not in fields:
                fields.append(name)
        return _latent_transformed_active_fields(fields)

    def _startup_predictor_p2_fields() -> list[str]:
        fields: list[str] = []
        ordered_keys = (
            ("v_x", "v_k"),
            ("v_y", "v_k"),
            ("p", "p_k"),
            ("p_mean", "p_mean_k"),
            ("alpha_mass_lm", "alpha_mass_lm_k"),
            ("pi_s", "pi_s_k"),
            ("vS_x", "vS_k"),
            ("vS_y", "vS_k"),
            ("u_x", "u_k"),
            ("u_y", "u_k"),
            ("lambda_drag_x", "lambda_drag_k"),
            ("lambda_drag_y", "lambda_drag_k"),
            ("alpha", "alpha_k"),
            ("mu_alpha", "mu_k"),
            ("alpha_latent", "alpha_latent_k"),
            ("phi", "phi_k"),
            ("phi_latent", "phi_latent_k"),
            ("S", "S_k"),
        )
        for name, key in ordered_keys:
            if problem.get(key) is not None and name not in fields:
                fields.append(name)
        return _latent_transformed_active_fields(fields)

    def _forms_support_startup_staggered(forms_obj) -> bool:
        required = ("r_momentum", "r_mass", "r_skeleton", "r_kinematics", "a_momentum", "a_mass", "a_skeleton", "a_kinematics")
        return all(getattr(forms_obj, name, None) is not None for name in required)

    def _sum_stage_forms(*parts):
        kept = [part for part in parts if part is not None]
        if not kept:
            raise ValueError("Stage form assembly received no residual/Jacobian terms.")
        total = kept[0]
        for part in kept[1:]:
            total = total + part
        return total

    startup_stage_accept_factor = 0.0
    if _logistic_refmap_phi_only_mode(args) and _predictor_corrector_startup_enabled(args):
        startup_stage_accept_factor = max(float(startup_stage_accept_factor), 150.0)

    def _make_startup_stage_solver(
        *,
        stage_name: str,
        residual_form,
        jacobian_form,
        active_fields: list[str],
        max_newton_iter_override: int | None = None,
    ):
        cache_key = f"{stage_name}:{','.join(active_fields)}"
        target_solver = startup_stage_solver_cache.get(cache_key)
        if target_solver is not None:
            return target_solver
        stage_bcs = _filter_bcs_by_fields(bcs, active_fields)
        stage_bcs_homog = _filter_bcs_by_fields(bcs_homog, active_fields)
        stage_solver_kind = _startup_stage_solver_kind(
            main_solver_kind=str(solver_key),
            active_fields=active_fields,
            stage_name=stage_name,
            transport_solver_kind_override=str(getattr(args, "startup_transport_solver", "auto")),
        )
        stage_relaxed_accept_ginf = float(startup_stage_relaxed_ginf)
        if stage_name == "transport" and stage_solver_kind in {"pdas", "ipm"}:
            # The staged transport solve is only a predictor for the monolithic
            # retry. If it already reaches a tightly identified bounded state
            # with |G| around 1e-4..1e-3, we should keep that state and export
            # its active-set information instead of throwing it away because a
            # final local line search could not reduce it further.
            stage_relaxed_accept_ginf = max(stage_relaxed_accept_ginf, 1.0e-3)
        target_solver = _make_solver(
            SimpleNamespace(residual_form=residual_form, jacobian_form=jacobian_form),
            postproc_cb=None,
            max_newton_iter=(
                int(args.startup_bootstrap_max_it)
                if max_newton_iter_override is None
                else int(max_newton_iter_override)
            ),
            # Startup stage solves only need to produce a materially better guess
            # for the subsequent monolithic retry. Requiring full Newton
            # convergence here is too strict and can discard useful transport
            # updates (especially the alpha/phi block) after they already
            # reduced the dominant first-step residual by orders of magnitude.
            # Keep this relaxed, but cap it tightly enough that obviously
            # nonconverged PDAS states (|G|=O(1..10)) are not accepted as
            # startup guesses in the stiff low-kappa cases.
            accept_factor=float(startup_stage_accept_factor),
            relaxed_accept_ginf=float(stage_relaxed_accept_ginf),
            solver_kind=stage_solver_kind,
            bcs_in=stage_bcs,
            bcs_homog_in=stage_bcs_homog,
        )
        _set_solver_active_fields_with_tracking(target_solver, active_fields)
        if hasattr(target_solver, "np"):
            target_solver.np.stall_window = 0
            target_solver.np.stall_min_abs_decrease_inf = 0.0
            target_solver.np.stall_min_rel_decrease_inf = 0.0
            target_solver.np.tr_min_abs_decrease_inf = 0.0
            target_solver.np.tr_min_rel_decrease_inf = 0.0
        startup_stage_solver_cache[cache_key] = target_solver
        return target_solver

    def _get_startup_staggered_solvers():
        flow_solver = startup_stage_solver_cache.get("flow")
        transport_solver = startup_stage_solver_cache.get("transport")
        solid_solver = startup_stage_solver_cache.get("solid")
        if flow_solver is not None and transport_solver is not None and solid_solver is not None:
            return flow_solver, transport_solver, solid_solver
        if not _forms_support_startup_staggered(forms):
            return None
        flow_fields = _startup_flow_stage_fields()
        transport_fields = _startup_transport_stage_fields()
        solid_fields = _startup_solid_stage_fields()
        flow_solver = _make_startup_stage_solver(
            stage_name="flow",
            residual_form=_sum_stage_forms(
                forms.r_momentum,
                forms.r_mass,
                problem.get("_pressure_mean_residual_form"),
            ),
            jacobian_form=_sum_stage_forms(
                forms.a_momentum,
                forms.a_mass,
                problem.get("_pressure_mean_jacobian_form"),
            ),
            active_fields=flow_fields,
        )
        transport_solver = _make_startup_stage_solver(
            stage_name="transport",
            residual_form=_sum_stage_forms(
                getattr(forms, "r_alpha", None),
                getattr(forms, "r_mu_alpha", None),
                problem.get("_alpha_mass_constraint_residual_form"),
                getattr(forms, "r_phi", None),
                getattr(forms, "r_substrate", None),
            ),
            jacobian_form=_sum_stage_forms(
                getattr(forms, "a_alpha", None),
                getattr(forms, "a_mu_alpha", None),
                problem.get("_alpha_mass_constraint_jacobian_form"),
                getattr(forms, "a_phi", None),
                getattr(forms, "a_substrate", None),
            ),
            active_fields=transport_fields,
        )
        solid_solver = _make_startup_stage_solver(
            stage_name="solid",
            residual_form=forms.r_skeleton + forms.r_kinematics,
            jacobian_form=forms.a_skeleton + forms.a_kinematics,
            active_fields=solid_fields,
        )
        startup_stage_solver_cache["flow"] = flow_solver
        startup_stage_solver_cache["transport"] = transport_solver
        startup_stage_solver_cache["solid"] = solid_solver
        return flow_solver, transport_solver, solid_solver

    def _run_startup_staggered_guess(
        *,
        funcs,
        prev_funcs,
        aux_funcs,
        bcs_now,
        step_no: int,
        t_fail: float,
        dt_fail: float,
        outer_it_override: int | None = None,
    ):
        staged = _get_startup_staggered_solvers()
        if staged is None:
            raise RuntimeError("staggered startup solver is unavailable for the current formulation.")
        flow_solver, transport_solver, solid_solver = staged
        flow_fields = _startup_flow_stage_fields()
        transport_fields = _startup_transport_stage_fields()
        solid_fields = _startup_solid_stage_fields()
        flow_bcs_now = _filter_bcs_by_fields(bcs_now, flow_fields)
        transport_bcs_now = _filter_bcs_by_fields(bcs_now, transport_fields)
        solid_bcs_now = _filter_bcs_by_fields(bcs_now, solid_fields)
        _copy_prev_into_current(funcs, prev_funcs)
        for stage_solver in (flow_solver, transport_solver, solid_solver):
            stage_solver._current_t = float(t_fail)
            stage_solver._current_dt = float(dt_fail)
            stage_solver._current_step_no = int(step_no)
        accept_key = "PYCUTFEM_NEWTON_MAXITER_ACCEPT_FACTOR"
        prev_accept = os.environ.get(accept_key)
        os.environ[accept_key] = "0"
        if outer_it_override is None:
            n_outer = max(1, int(getattr(args, "startup_staggered_outer_it", 1)))
        else:
            n_outer = max(1, int(outer_it_override))
        flow_hist: list[int] = []
        transport_hist: list[int] = []
        solid_hist: list[int] = []
        transport_vi_state = None
        last_good_snapshot = _snapshot_function_values(funcs)
        best_snapshot = last_good_snapshot
        best_exact_norm = _current_monolithic_raw_residual_inf(
            funcs=funcs,
            prev_funcs=prev_funcs,
            aux_funcs=aux_funcs,
            bcs_now=bcs_now,
        )
        completed_outer = 0
        partial_reason: str | None = None
        try:
            for outer_it in range(n_outer):
                try:
                    _set_solver_newton_trace_context(
                        flow_solver,
                        label=f"startup-staggered:flow[o{int(outer_it) + 1}]",
                        global_bcs_now=bcs_now,
                    )
                    problem["dh"].apply_bcs(flow_bcs_now, *funcs)
                    _, flow_converged, flow_iters = flow_solver._newton_loop(
                        funcs,
                        prev_funcs,
                        aux_funcs,
                        flow_bcs_now,
                    )
                    if not bool(flow_converged):
                        raise RuntimeError(
                            f"staggered startup flow solve did not converge on outer sweep {int(outer_it) + 1}"
                        )
                    flow_hist.append(int(flow_iters))
                    _set_solver_newton_trace_context(
                        transport_solver,
                        label=f"startup-staggered:transport[o{int(outer_it) + 1}]",
                        global_bcs_now=bcs_now,
                    )
                    problem["dh"].apply_bcs(transport_bcs_now, *funcs)
                    _, transport_converged, transport_iters = transport_solver._newton_loop(
                        funcs,
                        prev_funcs,
                        aux_funcs,
                        transport_bcs_now,
                    )
                    if not bool(transport_converged):
                        raise RuntimeError(
                            f"staggered startup transport solve did not converge on outer sweep {int(outer_it) + 1}"
                        )
                    transport_hist.append(int(transport_iters))
                    export_vi_state = getattr(transport_solver, "export_vi_state", None)
                    if callable(export_vi_state):
                        transport_vi_state = export_vi_state()
                    _set_solver_newton_trace_context(
                        solid_solver,
                        label=f"startup-staggered:solid[o{int(outer_it) + 1}]",
                        global_bcs_now=bcs_now,
                    )
                    problem["dh"].apply_bcs(solid_bcs_now, *funcs)
                    _, solid_converged, solid_iters = solid_solver._newton_loop(
                        funcs,
                        prev_funcs,
                        aux_funcs,
                        solid_bcs_now,
                    )
                    if not bool(solid_converged):
                        raise RuntimeError(
                            f"staggered startup solid solve did not converge on outer sweep {int(outer_it) + 1}"
                        )
                    solid_hist.append(int(solid_iters))
                except Exception as stage_exc:
                    candidate_snapshot = _snapshot_function_values(funcs)
                    candidate_norm = _current_monolithic_raw_residual_inf(
                        funcs=funcs,
                        prev_funcs=prev_funcs,
                        aux_funcs=aux_funcs,
                        bcs_now=bcs_now,
                    )
                    if (
                        np.isfinite(candidate_norm)
                        and (not np.isfinite(best_exact_norm) or float(candidate_norm) < float(best_exact_norm))
                    ):
                        best_exact_norm = float(candidate_norm)
                        best_snapshot = candidate_snapshot
                        partial_reason = str(stage_exc)
                        _restore_function_values(funcs, best_snapshot)
                        break
                    _restore_function_values(funcs, last_good_snapshot)
                    if completed_outer > 0:
                        partial_reason = str(stage_exc)
                        break
                    raise
                completed_outer += 1
                last_good_snapshot = _snapshot_function_values(funcs)
                best_snapshot = last_good_snapshot
                best_exact_norm = _current_monolithic_raw_residual_inf(
                    funcs=funcs,
                    prev_funcs=prev_funcs,
                    aux_funcs=aux_funcs,
                    bcs_now=bcs_now,
                )
        finally:
            if prev_accept is None:
                os.environ.pop(accept_key, None)
            else:
                os.environ[accept_key] = prev_accept
        _restore_function_values(funcs, best_snapshot)
        preload_relaxed = _relax_staggered_preload_toward_previous(
            funcs,
            prev_funcs,
            blend=float(getattr(args, "startup_preload_prev_blend", 1.0)),
            keep_fields=_startup_transport_stage_fields(),
        )
        if preload_relaxed:
            print(
                "    [startup] relaxed staged preload toward the previous accepted state "
                f"with blend={float(getattr(args, 'startup_preload_prev_blend', 1.0)):.3f} "
                "(transport fields kept from the staged solve)."
            )
        import_vi_state = getattr(solver, "import_vi_state", None)
        if completed_outer > 0 and callable(import_vi_state) and transport_vi_state is not None:
            import_vi_state(transport_vi_state, force_once=True)
        reset_bounds_cb = getattr(solver, "_reset_benchmark7_vi_bounds_freeze", None)
        if callable(reset_bounds_cb):
            reset_bounds_cb()
        return {
            "outer_it": int(completed_outer),
            "requested_outer_it": int(n_outer),
            "flow_iters_total": int(sum(flow_hist)),
            "transport_iters_total": int(sum(transport_hist)),
            "solid_iters_total": int(sum(solid_hist)),
            "flow_iters_hist": tuple(int(v) for v in flow_hist),
            "transport_iters_hist": tuple(int(v) for v in transport_hist),
            "solid_iters_hist": tuple(int(v) for v in solid_hist),
            "partial_success": bool(partial_reason is not None),
            "partial_reason": partial_reason,
        }

    def _current_monolithic_raw_residual_inf(*, funcs, prev_funcs, aux_funcs, bcs_now) -> float:
        try:
            problem["dh"].apply_bcs(bcs_now, *funcs)
            if getattr(solver, "constraints", None) is not None:
                solver._enforce_constraints_on_functions(funcs)
            current = {f.name: f for f in funcs}
            current.update({f.name: f for f in prev_funcs})
            if aux_funcs:
                current.update(aux_funcs)
            _, r_red = solver._assemble_system_reduced(current, need_matrix=False)
            return float(np.linalg.norm(np.asarray(r_red, dtype=float), ord=np.inf))
        except Exception:
            return float("nan")

    def _set_solver_newton_trace_context(
        target_solver,
        *,
        label,
        global_bcs_now=None,
        global_label: str = "|R_raw|_∞",
    ) -> None:
        if target_solver is None:
            return
        target_solver._newton_trace_label = label

        def _global_residual(funcs_now, prev_funcs_now, aux_funcs_now, bcs_stage_now):
            return _current_monolithic_raw_residual_inf(
                funcs=funcs_now,
                prev_funcs=prev_funcs_now,
                aux_funcs=aux_funcs_now,
                bcs_now=(bcs_stage_now if global_bcs_now is None else global_bcs_now),
            )

        target_solver._newton_trace_global_residual_cb = _global_residual
        target_solver._newton_trace_global_residual_label = str(global_label)

    def _current_stage_raw_residual_inf(*, stage_solver, funcs, prev_funcs, aux_funcs, bcs_now) -> float:
        if stage_solver is None:
            return _current_monolithic_raw_residual_inf(
                funcs=funcs,
                prev_funcs=prev_funcs,
                aux_funcs=aux_funcs,
                bcs_now=bcs_now,
            )
        try:
            problem["dh"].apply_bcs(bcs_now, *funcs)
            if getattr(stage_solver, "constraints", None) is not None:
                stage_solver._enforce_constraints_on_functions(funcs)
            current = {f.name: f for f in funcs}
            current.update({f.name: f for f in prev_funcs})
            if aux_funcs:
                current.update(aux_funcs)
            _, r_red = stage_solver._assemble_system_reduced(current, need_matrix=False)
            return float(np.linalg.norm(np.asarray(r_red, dtype=float), ord=np.inf))
        except Exception:
            return float("nan")

    def _current_predictor_corrector_energy(
        *,
        funcs,
        prev_funcs,
        aux_funcs,
        bcs_now,
        stage_solver=None,
        stage_bcs_now=None,
    ) -> dict[str, float]:
        exact_raw_inf = _current_monolithic_raw_residual_inf(
            funcs=funcs,
            prev_funcs=prev_funcs,
            aux_funcs=aux_funcs,
            bcs_now=bcs_now,
        )
        homotopy_raw_inf = _current_stage_raw_residual_inf(
            stage_solver=stage_solver,
            funcs=funcs,
            prev_funcs=prev_funcs,
            aux_funcs=aux_funcs,
            bcs_now=(bcs_now if stage_bcs_now is None else stage_bcs_now),
        )
        alpha_mass_rel = _current_alpha_mass_relative_defect(funcs=funcs, prev_funcs=prev_funcs)
        mass_weight = max(0.0, float(getattr(args, "pc_energy_mass_weight", 1.0) or 0.0))
        exact_energy = float("nan")
        if np.isfinite(exact_raw_inf):
            exact_energy = float(exact_raw_inf * exact_raw_inf)
            if np.isfinite(alpha_mass_rel) and mass_weight > 0.0:
                exact_energy += float(mass_weight) * float(alpha_mass_rel * alpha_mass_rel)
        homotopy_energy = float("nan")
        if np.isfinite(homotopy_raw_inf):
            homotopy_energy = float(homotopy_raw_inf * homotopy_raw_inf)
            if np.isfinite(alpha_mass_rel) and mass_weight > 0.0:
                homotopy_energy += float(mass_weight) * float(alpha_mass_rel * alpha_mass_rel)
        return {
            "raw_inf": float(exact_raw_inf),
            "exact_raw_inf": float(exact_raw_inf),
            "alpha_mass_rel": float(alpha_mass_rel),
            "energy": float(exact_energy),
            "exact_energy": float(exact_energy),
            "homotopy_raw_inf": float(homotopy_raw_inf),
            "homotopy_energy": float(homotopy_energy),
        }

    def _pc_required_exact_drop(raw_before: float) -> float:
        return _pc_required_drop(
            raw_before,
            min_abs_decrease=float(getattr(args, "pc_min_abs_decrease", 1.0e-10) or 0.0),
            min_rel_improve=float(getattr(args, "pc_min_rel_improve", 0.0) or 0.0),
        )

    def _pc_exact_progress(before_stats: dict[str, float], after_stats: dict[str, float]) -> dict[str, float | bool]:
        return _pc_progress(
            before_stats,
            after_stats,
            key="raw_inf",
            min_abs_decrease=float(getattr(args, "pc_min_abs_decrease", 1.0e-10) or 0.0),
            min_rel_improve=float(getattr(args, "pc_min_rel_improve", 0.0) or 0.0),
        )

    def _pc_homotopy_progress(before_stats: dict[str, float], after_stats: dict[str, float]) -> dict[str, float | bool]:
        return _pc_progress(
            before_stats,
            after_stats,
            key="homotopy_raw_inf",
            min_abs_decrease=float(getattr(args, "pc_min_abs_decrease", 1.0e-10) or 0.0),
            min_rel_improve=float(getattr(args, "pc_min_rel_improve", 0.0) or 0.0),
        )

    def _blend_snapshots(snapshot_a, snapshot_b, theta: float) -> tuple[np.ndarray, ...]:
        blend = float(theta)
        return tuple(
            np.asarray(a, dtype=float) + blend * (np.asarray(b, dtype=float) - np.asarray(a, dtype=float))
            for a, b in zip(list(snapshot_a or []), list(snapshot_b or []))
        )

    def _predictor_p2_lambda_schedule() -> list[float]:
        # Start cautiously: the first positive continuation jump should be much
        # smaller than the final jump so P2 can accept partial homotopy progress
        # before the exact corrector takes over.
        return _pc_p2_lambda_schedule(
            max(1, int(getattr(args, "pc_p2_lambda_steps", 4))),
            include_zero=False,
        )

    def _predictor_p2_initial_lambda_step() -> float:
        schedule = _predictor_p2_lambda_schedule()
        if not schedule:
            return 1.0
        return max(1.0e-8, float(schedule[0]))

    def _project_predictor_segment(
        *,
        funcs,
        prev_funcs,
        aux_funcs,
        bcs_now,
        anchor_snapshot,
        trial_snapshot,
        stage_label: str,
        stage_solver=None,
        stage_bcs_now=None,
        primary_key: str = "raw_inf",
    ) -> dict[str, object]:
        max_trials = max(1, int(getattr(args, "pc_projection_trials", 5)))
        if str(stage_label).startswith("P2"):
            max_trials = max(max_trials, int(getattr(args, "pc_p2_projection_trials", 12)))
        primary_energy_key = "homotopy_energy" if str(primary_key) == "homotopy_raw_inf" else "energy"
        thetas = [1.0]
        theta = 1.0
        for _ in range(max_trials - 1):
            theta *= 0.5
            thetas.append(theta)
        if not any(abs(v) <= 1.0e-15 for v in thetas):
            thetas.append(0.0)
        best_snapshot = anchor_snapshot
        best_stats = None
        best_theta = 0.0
        best_mass_ok = False
        for theta in thetas:
            cand_snapshot = _blend_snapshots(anchor_snapshot, trial_snapshot, theta)
            _restore_function_values(funcs, cand_snapshot)
            _apply_predictor_alpha_mass_return_map(
                funcs=funcs,
                prev_funcs=prev_funcs,
                bcs_now=bcs_now,
                stage_label=f"{stage_label}[theta={float(theta):.3f}]",
                verbose=False,
            )
            cand_snapshot = _snapshot_function_values(funcs)
            stats = _current_predictor_corrector_energy(
                funcs=funcs,
                prev_funcs=prev_funcs,
                aux_funcs=aux_funcs,
                bcs_now=bcs_now,
                stage_solver=stage_solver,
                stage_bcs_now=stage_bcs_now,
            )
            cand_mass_ok = _predictor_alpha_mass_ok(stats)
            if best_stats is None:
                best_snapshot = cand_snapshot
                best_stats = dict(stats)
                best_theta = float(theta)
                best_mass_ok = bool(cand_mass_ok)
                continue
            cand_energy = float(stats.get(primary_energy_key, float("nan")))
            best_energy = float(best_stats.get(primary_energy_key, float("nan")))
            cand_raw = float(stats.get(primary_key, float("nan")))
            best_raw = float(best_stats.get(primary_key, float("nan")))
            raw_tol = 1.0e-15 * max(1.0, abs(cand_raw), abs(best_raw))
            cand_exact = float(stats.get("raw_inf", float("nan")))
            best_exact = float(best_stats.get("raw_inf", float("nan")))
            exact_tol = 1.0e-15 * max(1.0, abs(cand_exact), abs(best_exact))
            if (
                (cand_mass_ok and not best_mass_ok)
                or (
                    cand_mass_ok == best_mass_ok
                    and np.isfinite(cand_raw)
                    and not np.isfinite(best_raw)
                )
                or (
                    cand_mass_ok == best_mass_ok
                    and np.isfinite(cand_raw)
                    and np.isfinite(best_raw)
                    and cand_raw < best_raw - raw_tol
                )
                or (
                    cand_mass_ok == best_mass_ok
                    and np.isfinite(cand_raw)
                    and np.isfinite(best_raw)
                    and abs(cand_raw - best_raw) <= raw_tol
                    and np.isfinite(cand_exact)
                    and not np.isfinite(best_exact)
                )
                or (
                    cand_mass_ok == best_mass_ok
                    and np.isfinite(cand_raw)
                    and np.isfinite(best_raw)
                    and abs(cand_raw - best_raw) <= raw_tol
                    and np.isfinite(cand_exact)
                    and np.isfinite(best_exact)
                    and cand_exact < best_exact - exact_tol
                )
                or (
                    cand_mass_ok == best_mass_ok
                    and np.isfinite(cand_raw)
                    and np.isfinite(best_raw)
                    and abs(cand_raw - best_raw) <= raw_tol
                    and np.isfinite(cand_exact)
                    and np.isfinite(best_exact)
                    and abs(cand_exact - best_exact) <= exact_tol
                    and np.isfinite(cand_energy)
                    and not np.isfinite(best_energy)
                )
                or (
                    cand_mass_ok == best_mass_ok
                    and np.isfinite(cand_raw)
                    and np.isfinite(best_raw)
                    and abs(cand_raw - best_raw) <= raw_tol
                    and np.isfinite(cand_exact)
                    and np.isfinite(best_exact)
                    and abs(cand_exact - best_exact) <= exact_tol
                    and np.isfinite(cand_energy)
                    and np.isfinite(best_energy)
                    and cand_energy < best_energy
                )
                or (
                    cand_mass_ok == best_mass_ok
                    and not np.isfinite(cand_raw)
                    and not np.isfinite(best_raw)
                    and np.isfinite(cand_energy)
                    and not np.isfinite(best_energy)
                )
                or (
                    cand_mass_ok == best_mass_ok
                    and not np.isfinite(cand_raw)
                    and not np.isfinite(best_raw)
                    and np.isfinite(cand_energy)
                    and np.isfinite(best_energy)
                    and cand_energy < best_energy
                )
            ):
                best_snapshot = cand_snapshot
                best_stats = dict(stats)
                best_theta = float(theta)
                best_mass_ok = bool(cand_mass_ok)
        _restore_function_values(funcs, best_snapshot)
        if best_stats is None:
            best_stats = {
                "raw_inf": float("nan"),
                "alpha_mass_rel": float("nan"),
                "energy": float("nan"),
            }
        print(
            f"    [pc] {stage_label} segment projection kept theta={best_theta:.3f} "
            f"with |R_raw|_∞={float(best_stats['raw_inf']):.3e}, "
            f"|H|_∞={float(best_stats.get('homotopy_raw_inf', best_stats['raw_inf'])):.3e}, "
            f"alpha_mass_rel={float(best_stats['alpha_mass_rel']):.3e}, "
            f"E_exact={float(best_stats['energy']):.3e}, "
            f"E_h={float(best_stats.get('homotopy_energy', best_stats['energy'])):.3e}."
        )
        return {
            "theta": float(best_theta),
            "stats": dict(best_stats),
        }

    def _stage_reduced_iterate(stage_solver, funcs) -> np.ndarray:
        pack_cb = getattr(stage_solver, "_pack_reduced_iterate", None)
        if callable(pack_cb):
            return np.asarray(pack_cb(funcs), dtype=float).ravel()
        current_cb = getattr(stage_solver, "_current_reduced_iterate", None)
        if callable(current_cb):
            return np.asarray(current_cb(funcs), dtype=float).ravel()
        gather_cb = getattr(stage_solver, "_gather_full_iterate", None)
        restrictor = getattr(stage_solver, "restrictor", None)
        if callable(gather_cb) and restrictor is not None and hasattr(restrictor, "restrict_vec"):
            x_full = np.asarray(gather_cb(funcs), dtype=float).ravel()
            return np.asarray(restrictor.restrict_vec(x_full), dtype=float).ravel()
        raise AttributeError(
            f"{type(stage_solver).__name__} does not provide a reduced-iterate packer."
        )

    def _write_stage_reduced_iterate(
        *,
        stage_solver,
        funcs,
        prev_funcs,
        aux_funcs,
        bcs_now,
        x_target_red,
        project_to_bounds: bool = True,
    ) -> None:
        x_curr_red = _stage_reduced_iterate(stage_solver, funcs)
        x_target = np.asarray(x_target_red, dtype=float).ravel()
        if x_curr_red.shape != x_target.shape:
            raise ValueError(
                f"Reduced iterate shape mismatch: current={x_curr_red.shape}, target={x_target.shape}."
            )
        d_red = x_target - x_curr_red
        if np.any(d_red):
            d_full = stage_solver.restrictor.expand_vec(d_red)
            stage_solver.dh.add_to_functions(d_full, funcs)
            stage_solver.dh.apply_bcs(bcs_now, *funcs)
            if getattr(stage_solver, "constraints", None) is not None:
                stage_solver._enforce_constraints_on_functions(funcs)
        if not bool(project_to_bounds) or not hasattr(stage_solver, "_project_funcs_to_bounds"):
            return
        try:
            lo_red, hi_red = stage_solver._bounds_reduced()
        except Exception:
            return
        eq_prepare_callback = None
        if hasattr(stage_solver, "_vi_prepare_linear_equalities"):
            eq_prepare_callback = lambda: stage_solver._vi_prepare_linear_equalities(
                funcs,
                prev_funcs,
                aux_funcs,
                bcs_now,
            )
        stage_solver._project_funcs_to_bounds(
            funcs,
            bcs_now,
            lo_red,
            hi_red,
            eq_prepare_callback=eq_prepare_callback,
        )

    def _run_predictor_corrector_p2_path_predictor(
        *,
        funcs,
        prev_funcs,
        aux_funcs,
        bcs_now,
        p2_solver,
        p2_bcs_now,
        p2_lambda,
        lam_from: float,
        lam_to: float,
        stage_label: str,
    ) -> dict[str, object] | None:
        lam0 = float(lam_from)
        lam1 = float(lam_to)
        if p2_lambda is None or not np.isfinite(lam0) or not np.isfinite(lam1):
            return None
        delta_lambda = float(lam1 - lam0)
        if abs(delta_lambda) <= 1.0e-15:
            return None
        predictor_snapshot = _snapshot_function_values(funcs)
        try:
            p2_lambda.value = float(lam0)
            problem["dh"].apply_bcs(p2_bcs_now, *funcs)
            if getattr(p2_solver, "constraints", None) is not None:
                p2_solver._enforce_constraints_on_functions(funcs)
            current = {f.name: f for f in funcs}
            current.update({f.name: f for f in prev_funcs})
            if aux_funcs:
                current.update(aux_funcs)
            x_curr_red = _stage_reduced_iterate(p2_solver, funcs)
            A_red, _ = p2_solver._assemble_system_reduced(current, need_matrix=True)
            p2_lambda.value = 0.0
            _, r_easy_red = p2_solver._assemble_system_reduced(current, need_matrix=False)
            p2_lambda.value = 1.0
            _, r_exact_red = p2_solver._assemble_system_reduced(current, need_matrix=False)
            dH_dlambda_red = (
                np.asarray(r_exact_red, dtype=float).ravel()
                - np.asarray(r_easy_red, dtype=float).ravel()
            )
            easy_inf = float(np.linalg.norm(np.asarray(r_easy_red, dtype=float).ravel(), ord=np.inf))
            exact_inf = float(np.linalg.norm(np.asarray(r_exact_red, dtype=float).ravel(), ord=np.inf))
            forcing_inf = float(np.linalg.norm(dH_dlambda_red, ord=np.inf))
            forcing_rel = float(forcing_inf / max(1.0, easy_inf, exact_inf))
            p2_lambda.value = float(lam0)
            x_pred_red, z_dot_red = _pc_path_tangent_euler_step(
                x_red=x_curr_red,
                jacobian_red=A_red,
                dH_dlambda_red=dH_dlambda_red,
                delta_lambda=delta_lambda,
                solve_linear_system=lambda A, rhs: p2_solver._solve_linear_system_with_context(
                    A,
                    np.asarray(rhs, dtype=float).ravel(),
                    context="homotopy_predictor",
                ),
            )
        except Exception as exc:
            _restore_function_values(funcs, predictor_snapshot)
            p2_lambda.value = float(lam1)
            print(
                f"    [pc] {stage_label} tangent predictor failed; falling back to the anchor state: {exc}"
            )
            return {
                "used": False,
                "exception": str(exc),
            }

        candidate_alphas = [1.0, 0.5, 0.25, 0.1, 0.0]
        best_snapshot = predictor_snapshot
        best_stats = None
        best_alpha = 0.0
        best_mass_ok = False
        raw_step = np.asarray(x_pred_red, dtype=float) - np.asarray(x_curr_red, dtype=float)
        for alpha in candidate_alphas:
            cand_red = np.asarray(x_curr_red, dtype=float) + float(alpha) * raw_step
            _restore_function_values(funcs, predictor_snapshot)
            p2_lambda.value = float(lam1)
            try:
                _write_stage_reduced_iterate(
                    stage_solver=p2_solver,
                    funcs=funcs,
                    prev_funcs=prev_funcs,
                    aux_funcs=aux_funcs,
                    bcs_now=p2_bcs_now,
                    x_target_red=cand_red,
                    project_to_bounds=True,
                )
                cand_snapshot = _snapshot_function_values(funcs)
                cand_stats = _current_predictor_corrector_energy(
                    funcs=funcs,
                    prev_funcs=prev_funcs,
                    aux_funcs=aux_funcs,
                    bcs_now=bcs_now,
                    stage_solver=p2_solver,
                    stage_bcs_now=p2_bcs_now,
                )
                cand_mass_ok = _predictor_alpha_mass_ok(cand_stats)
            except Exception:
                continue
            if best_stats is None:
                best_snapshot = cand_snapshot
                best_stats = dict(cand_stats)
                best_alpha = float(alpha)
                best_mass_ok = bool(cand_mass_ok)
                continue
            cand_h = float(cand_stats.get("homotopy_raw_inf", float("nan")))
            best_h = float(best_stats.get("homotopy_raw_inf", float("nan")))
            cand_r = float(cand_stats.get("raw_inf", float("nan")))
            best_r = float(best_stats.get("raw_inf", float("nan")))
            if (
                (cand_mass_ok and not best_mass_ok)
                or (
                    cand_mass_ok == best_mass_ok
                    and np.isfinite(cand_h)
                    and not np.isfinite(best_h)
                )
                or (
                    cand_mass_ok == best_mass_ok
                    and np.isfinite(cand_h)
                    and np.isfinite(best_h)
                    and cand_h < best_h - 1.0e-15 * max(1.0, abs(cand_h), abs(best_h))
                )
                or (
                    cand_mass_ok == best_mass_ok
                    and np.isfinite(cand_h)
                    and np.isfinite(best_h)
                    and abs(cand_h - best_h) <= 1.0e-15 * max(1.0, abs(cand_h), abs(best_h))
                    and np.isfinite(cand_r)
                    and (not np.isfinite(best_r) or cand_r < best_r)
                )
            ):
                best_snapshot = cand_snapshot
                best_stats = dict(cand_stats)
                best_alpha = float(alpha)
                best_mass_ok = bool(cand_mass_ok)
        _restore_function_values(funcs, best_snapshot)
        p2_lambda.value = float(lam1)
        if best_stats is None:
            return {
                "used": False,
                "exception": "no_finite_candidate",
            }
        print(
            f"    [pc] {stage_label} tangent predictor solved H_z z_dot + H_lambda = 0 "
            f"for delta_lambda={delta_lambda:.3f} and kept alpha={best_alpha:.3f} "
            f"with |dH/dlambda|_∞={forcing_inf:.3e} (rel={forcing_rel:.3e}), "
            f"with |H|_∞={float(best_stats['homotopy_raw_inf']):.3e}, "
            f"|R_raw|_∞={float(best_stats['raw_inf']):.3e}, "
            f"alpha_mass_rel={float(best_stats['alpha_mass_rel']):.3e}."
        )
        if forcing_rel <= 1.0e-8:
            print(
                f"    [pc] {stage_label} warning: easy and exact residuals are nearly identical at this anchor, "
                "so the continuation ODE has almost no forcing."
            )
        return {
            "used": bool(best_alpha > 0.0),
            "alpha": float(best_alpha),
            "stats": dict(best_stats),
            "state": _snapshot_function_values(funcs),
            "tangent_inf": float(np.linalg.norm(np.asarray(z_dot_red, dtype=float), ord=np.inf)),
            "delta_lambda": float(delta_lambda),
            "forcing_inf": float(forcing_inf),
            "forcing_rel": float(forcing_rel),
            "exception": None,
        }

    def _get_predictor_corrector_p1_solver():
        cache_key = "predictor_corrector_p1"
        target_solver = startup_stage_solver_cache.get(cache_key)
        if target_solver is not None:
            return target_solver
        p1_fields = _startup_predictor_p1_fields()
        p1_residual = _sum_stage_forms(
            forms.r_momentum,
            forms.r_mass,
            problem.get("_pressure_mean_residual_form"),
            forms.r_skeleton,
            forms.r_kinematics,
            getattr(forms, "r_alpha", None),
            getattr(forms, "r_mu_alpha", None),
            problem.get("_alpha_mass_constraint_residual_form"),
        )
        p1_jacobian = _sum_stage_forms(
            forms.a_momentum,
            forms.a_mass,
            problem.get("_pressure_mean_jacobian_form"),
            forms.a_skeleton,
            forms.a_kinematics,
            getattr(forms, "a_alpha", None),
            getattr(forms, "a_mu_alpha", None),
            problem.get("_alpha_mass_constraint_jacobian_form"),
        )
        target_solver = _make_startup_stage_solver(
            stage_name="predictor_p1",
            residual_form=p1_residual,
            jacobian_form=p1_jacobian,
            active_fields=p1_fields,
            max_newton_iter_override=max(1, int(getattr(args, "pc_p1_max_it", 10))),
        )
        startup_stage_solver_cache[cache_key] = target_solver
        return target_solver

    def _get_predictor_corrector_p2_solver():
        cache_key = "predictor_corrector_p2"
        target_solver = startup_stage_solver_cache.get(cache_key)
        if target_solver is not None:
            return target_solver
        saved_alpha_mass_res = problem.get("_alpha_mass_constraint_residual_form")
        saved_alpha_mass_jac = problem.get("_alpha_mass_constraint_jacobian_form")
        saved_pm_res = problem.get("_pressure_mean_residual_form")
        saved_pm_jac = problem.get("_pressure_mean_jacobian_form")
        p2_lambda = _named_constant("pc_p2_lambda", 0.0)
        one_minus_p2_lambda = _named_constant("pc_p2_one", 1.0) - p2_lambda
        p2_easy_fluid_full = _named_constant("pc_p2_easy_fluid_full", 0.0)
        p2_easy_fluid_lagged = _named_constant("pc_p2_easy_fluid_lagged", 0.0)
        p2_easy_fluid_imex = _named_constant("pc_p2_easy_fluid_imex", 0.0)
        p2_exact_fluid_full = _named_constant("pc_p2_exact_fluid_full", 0.0)
        p2_exact_fluid_lagged = _named_constant("pc_p2_exact_fluid_lagged", 0.0)
        p2_exact_fluid_imex = _named_constant("pc_p2_exact_fluid_imex", 0.0)
        p2_easy_skeleton_accel = _named_constant("pc_p2_easy_skeleton_accel", 0.0)
        p2_exact_skeleton_accel = _named_constant(
            "pc_p2_exact_skeleton_accel",
            1.0 if bool(args.include_skeleton_acceleration) else 0.0,
        )
        p2_easy_skeleton_full = _named_constant("pc_p2_easy_skeleton_full", 0.0)
        p2_easy_skeleton_lagged = _named_constant("pc_p2_easy_skeleton_lagged", 0.0)
        p2_exact_skeleton_full = _named_constant("pc_p2_exact_skeleton_full", 0.0)
        p2_exact_skeleton_lagged = _named_constant("pc_p2_exact_skeleton_lagged", 0.0)
        p2_forms = _build_forms(
            problem,
            qdeg=int(qdeg),
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
            gamma_div=float(effective_gamma_div),
            mechanics_nondim_mode=str(args.mechanics_nondim_mode),
            solid_volumetric_split=bool(args.solid_volumetric_split),
            solid_volumetric_penalty=float(args.solid_volumetric_penalty),
            gamma_u=float(args.gamma_u),
            u_extension_mode=str(args.u_extension),
            gamma_u_pin=float(args.gamma_u_pin),
            u_cip=float(args.u_cip),
            u_cip_weight=str(args.u_cip_weight),
            vS_cip=float(args.vS_cip),
            gamma_vS=(None if args.gamma_vS is None else float(args.gamma_vS)),
            vS_extension_mode=args.vS_ext_mode,
            gamma_vS_pin=(None if args.gamma_vS_pin is None else float(args.gamma_vS_pin)),
            D_phi=float(args.D_phi),
            phi_diffusion_weight=str(args.phi_diffusion_weight),
            gamma_phi=float(args.gamma_phi),
            phi_supg=float(args.phi_supg),
            phi_cip=float(args.phi_cip),
            alpha_supg=float(args.alpha_supg),
            alpha_cip=float(args.alpha_cip),
            alpha_regularization=str(args.alpha_regularization),
            alpha_reg_gamma=float(args.alpha_reg_gamma),
            alpha_reg_eps_normal=float(alpha_reg_eps_normal),
            alpha_reg_eps_tangent=float(alpha_reg_eps_tangent),
            alpha_reg_eta=float(args.alpha_reg_eta),
            alpha_advect_with=str(args.alpha_advect_with),
            alpha_advection_form=str(args.alpha_advection_form),
            support_physics=str(args.support_physics),
            solid_model=str(args.solid_model),
            kappa_inv_model=str(args.kappa_inv_model),
            drag_formulation=str(args.drag_formulation),
            fluid_convection=str(getattr(args, "pc_p2_fluid_convection", "lagged")),
            fluid_convection_full_weight=(
                one_minus_p2_lambda * p2_easy_fluid_full
                + p2_lambda * p2_exact_fluid_full
            ),
            fluid_convection_lagged_weight=(
                one_minus_p2_lambda * p2_easy_fluid_lagged
                + p2_lambda * p2_exact_fluid_lagged
            ),
            fluid_convection_imex_weight=(
                one_minus_p2_lambda * p2_easy_fluid_imex
                + p2_lambda * p2_exact_fluid_imex
            ),
            enable_phi_evolution=bool(args.enable_phi_evolution),
            include_skeleton_acceleration=bool(args.include_skeleton_acceleration),
            skeleton_acceleration_weight=(
                one_minus_p2_lambda * p2_easy_skeleton_accel
                + p2_lambda * p2_exact_skeleton_accel
            ),
            rho_s0_tilde=float(args.rho_s0_tilde),
            skeleton_inertia_convection=str(getattr(args, "pc_p2_skeleton_inertia_convection", "lagged")),
            skeleton_inertia_full_weight=(
                one_minus_p2_lambda * p2_easy_skeleton_full
                + p2_lambda * p2_exact_skeleton_full
            ),
            skeleton_inertia_lagged_weight=(
                one_minus_p2_lambda * p2_easy_skeleton_lagged
                + p2_lambda * p2_exact_skeleton_lagged
            ),
            ds_hdiv_tangential=hdiv_tangential_bc_measure,
            ds_alpha_transport=ds_alpha_transport,
            ds_B_transport=ds_B_transport,
            hdiv_tangential_gamma=float(args.hdiv_tangential_gamma),
            hdiv_tangential_method=str(args.hdiv_tangential_method),
        )
        problem["_alpha_mass_constraint_residual_form"] = saved_alpha_mass_res
        problem["_alpha_mass_constraint_jacobian_form"] = saved_alpha_mass_jac
        problem["_pressure_mean_residual_form"] = saved_pm_res
        problem["_pressure_mean_jacobian_form"] = saved_pm_jac
        p2_fields = _startup_predictor_p2_fields()
        target_solver = _make_startup_stage_solver(
            stage_name="predictor_p2",
            residual_form=p2_forms.residual_form,
            jacobian_form=p2_forms.jacobian_form,
            active_fields=p2_fields,
            max_newton_iter_override=max(1, int(getattr(args, "pc_p2_max_it", 8))),
        )
        if hasattr(target_solver, "np"):
            globalization_mode = str(getattr(target_solver.np, "globalization", "line_search") or "line_search").strip().lower()
            if globalization_mode in {"trust_region", "line_search_then_trust"}:
                target_solver.np.tr_radius_init = max(float(getattr(target_solver.np, "tr_radius_init", 1.0)), 1.0)
                target_solver.np.tr_radius_max = max(float(getattr(target_solver.np, "tr_radius_max", 1.0e3)), 1.0e3)
            else:
                target_solver.np.tr_radius_init = min(float(getattr(target_solver.np, "tr_radius_init", 1.0)), 2.5e-1)
                target_solver.np.tr_radius_max = min(float(getattr(target_solver.np, "tr_radius_max", 1.0e3)), 2.0)
        target_solver._pc_p2_lambda = p2_lambda
        target_solver._pc_p2_mode_constants = {
            "easy_fluid_full": p2_easy_fluid_full,
            "easy_fluid_lagged": p2_easy_fluid_lagged,
            "easy_fluid_imex": p2_easy_fluid_imex,
            "exact_fluid_full": p2_exact_fluid_full,
            "exact_fluid_lagged": p2_exact_fluid_lagged,
            "exact_fluid_imex": p2_exact_fluid_imex,
            "easy_skeleton_accel": p2_easy_skeleton_accel,
            "exact_skeleton_accel": p2_exact_skeleton_accel,
            "easy_skeleton_full": p2_easy_skeleton_full,
            "easy_skeleton_lagged": p2_easy_skeleton_lagged,
            "exact_skeleton_full": p2_exact_skeleton_full,
            "exact_skeleton_lagged": p2_exact_skeleton_lagged,
        }
        startup_stage_solver_cache[cache_key] = target_solver
        return target_solver

    def _configure_predictor_corrector_p2_solver(
        *,
        p2_solver,
        dt_now: float,
    ) -> dict[str, float | str]:
        mode_constants = getattr(p2_solver, "_pc_p2_mode_constants", None)
        if isinstance(mode_constants, dict):
            easy_fluid = _pc_fluid_convection_selectors(str(getattr(args, "startup_bootstrap_fluid_convection", "off")))
            exact_fluid = _pc_fluid_convection_selectors(str(getattr(args, "pc_p2_fluid_convection", "lagged")))
            easy_skeleton = _pc_skeleton_inertia_selectors("lagged")
            exact_skeleton = _pc_skeleton_inertia_selectors(str(getattr(args, "pc_p2_skeleton_inertia_convection", "lagged")))
            mode_constants["easy_fluid_full"].value = float(easy_fluid["full"])
            mode_constants["easy_fluid_lagged"].value = float(easy_fluid["lagged"])
            mode_constants["easy_fluid_imex"].value = float(easy_fluid["imex"])
            mode_constants["exact_fluid_full"].value = float(exact_fluid["full"])
            mode_constants["exact_fluid_lagged"].value = float(exact_fluid["lagged"])
            mode_constants["exact_fluid_imex"].value = float(exact_fluid["imex"])
            mode_constants["easy_skeleton_accel"].value = float(
                1.0 if bool(getattr(args, "startup_bootstrap_include_skeleton_acceleration", False)) else 0.0
            )
            mode_constants["exact_skeleton_accel"].value = float(
                1.0 if bool(args.include_skeleton_acceleration) else 0.0
            )
            mode_constants["easy_skeleton_full"].value = float(easy_skeleton["full"])
            mode_constants["easy_skeleton_lagged"].value = float(easy_skeleton["lagged"])
            mode_constants["exact_skeleton_full"].value = float(exact_skeleton["full"])
            mode_constants["exact_skeleton_lagged"].value = float(exact_skeleton["lagged"])
        return {
            "anchor_dt": float(_pc_p2_easy_dt_value(args, float(dt_now))),
            "fluid_convection": str(getattr(args, "startup_bootstrap_fluid_convection", "off")),
            "include_skeleton_acceleration": float(
                1.0 if bool(getattr(args, "startup_bootstrap_include_skeleton_acceleration", False)) else 0.0
            ),
        }

    def _prime_predictor_corrector_p2_solver():
        if not (
            bool(getattr(args, "pc_prebuild_p2_kernels", True))
            and bool(args.enable_phi_evolution)
            and problem.get("phi_k") is not None
        ):
            return None
        target_solver = _get_predictor_corrector_p2_solver()
        if not bool(startup_stage_solver_cache.get("_predictor_corrector_p2_prebuilt", False)):
            startup_stage_solver_cache["_predictor_corrector_p2_prebuilt"] = True
            print("    [pc] P2 globalization solver cached and kernels prebuilt for reuse.")
        return target_solver

    def _run_predictor_corrector_exact_probe(
        *,
        funcs,
        prev_funcs,
        aux_funcs,
        bcs_now,
        before_snapshot,
        before_stats: dict[str, float],
        stage_label: str = "P-exact",
    ) -> dict[str, object]:
        probe_budget = max(0, int(getattr(args, "pc_exact_probe_max_it", 1) or 0))
        if probe_budget <= 0:
            return {
                "used": False,
                "keep": False,
                "snapshot": before_snapshot,
                "after": dict(before_stats),
                "iters": 0,
                "converged": False,
                "exception": None,
                "decision": {
                    "exact": _pc_exact_progress(before_stats, before_stats),
                    "alpha_mass_ok": bool(_predictor_alpha_mass_ok(before_stats)),
                },
            }
        budget_saved = int(
            getattr(getattr(solver, "np", None), "max_newton_iter", base_solver_max_newton_iter)
            or base_solver_max_newton_iter
        )
        trace_label_saved = getattr(solver, "_newton_trace_label", None)
        trace_cb_saved = getattr(solver, "_newton_trace_global_residual_cb", None)
        trace_global_label_saved = getattr(solver, "_newton_trace_global_residual_label", None)
        _restore_function_values(funcs, before_snapshot)
        problem["dh"].apply_bcs(bcs_now, *funcs)
        probe_exception = None
        strong_probe_abs = 0.0
        strong_probe_rel = 0.0
        if _logistic_refmap_phi_only_mode(args) and str(stage_label).strip().lower() == "g-exact":
            # On the refmap phi-only branch the G-anchor can already be much
            # better than the raw initial state, so a tiny exact decrease from
            # that basin is not enough reason to skip P0/P1/P2 entirely.
            strong_probe_rel = 0.25
        try:
            if hasattr(solver, "np"):
                solver.np.max_newton_iter = int(probe_budget)
            _set_solver_newton_trace_context(
                solver,
                label=f"pc:{str(stage_label)}",
            )
            try:
                _, probe_converged, probe_iters = solver._newton_loop(
                    funcs,
                    prev_funcs,
                    aux_funcs,
                    bcs_now,
                )
            except Exception as exc:
                probe_exception = exc
                probe_converged = False
                probe_iters = max(1, int(probe_budget))
            probe_after = _current_predictor_corrector_energy(
                funcs=funcs,
                prev_funcs=prev_funcs,
                aux_funcs=aux_funcs,
                bcs_now=bcs_now,
            )
            probe_snapshot = _snapshot_function_values(funcs)
            probe_keep, probe_decision = _pc_should_prefer_exact_probe(
                before_stats=before_stats,
                after_stats=probe_after,
                alpha_mass_ok=_predictor_alpha_mass_ok(probe_after),
                min_abs_decrease=float(getattr(args, "pc_min_abs_decrease", 1.0e-10) or 0.0),
                min_rel_improve=float(getattr(args, "pc_min_rel_improve", 0.0) or 0.0),
                strong_min_abs_decrease=float(strong_probe_abs),
                strong_min_rel_improve=float(strong_probe_rel),
            )
        finally:
            if hasattr(solver, "np"):
                solver.np.max_newton_iter = int(budget_saved)
            solver._newton_trace_label = trace_label_saved
            solver._newton_trace_global_residual_cb = trace_cb_saved
            solver._newton_trace_global_residual_label = trace_global_label_saved
        if not bool(probe_keep):
            _restore_function_values(funcs, before_snapshot)
            probe_snapshot = before_snapshot
            probe_after = dict(before_stats)
        exact_progress = dict(probe_decision.get("exact", {}))
        print(
            f"    [pc] {stage_label} monolithic probe "
            f"|R_raw|_∞ {float(before_stats['raw_inf']):.3e} -> {float(probe_after['raw_inf']):.3e}, "
            f"alpha_mass_rel {float(probe_after['alpha_mass_rel']):.3e}, "
            f"R_drop={float(exact_progress.get('drop', float('nan'))):.3e}, "
            f"R_req={float(exact_progress.get('required', float('nan'))):.3e}, "
            f"strong_req={float(dict(probe_decision.get('strong_exact', {})).get('required', float('nan'))):.3e}, "
            f"iters={int(probe_iters)}, converged={int(bool(probe_converged))}, kept={int(bool(probe_keep))}."
        )
        if probe_exception is not None:
            print(f"    [pc] {stage_label} monolithic probe stopped before full convergence: {probe_exception}")
        return {
            "used": True,
            "keep": bool(probe_keep),
            "snapshot": probe_snapshot,
            "after": dict(probe_after),
            "iters": int(probe_iters),
            "converged": bool(probe_converged),
            "exception": (None if probe_exception is None else str(probe_exception)),
            "decision": dict(probe_decision),
        }

    def _run_predictor_corrector_p2_staggered_anchor(
        *,
        funcs,
        prev_funcs,
        aux_funcs,
        bcs_now,
        step_no: int,
        t_fail: float,
        dt_fail: float,
        best_snapshot,
        best_stats,
    ) -> dict[str, object] | None:
        if not bool(getattr(args, "pc_p2_staggered_anchor", True)):
            return None
        dt_micro = _pc_p2_easy_dt_value(args, float(dt_fail))
        if not np.isfinite(dt_micro) or dt_micro <= 0.0:
            return None
        print(
            "    [pc] building G-anchor with a staggered micro-step "
            f"(dt={float(dt_micro):.3e}, outer={int(max(1, int(getattr(args, 'pc_p2_staggered_anchor_outer_it', 1))))})."
        )
        snapshot_in = _snapshot_function_values(funcs)
        dt_saved = float(dt_c.value)
        try:
            dt_c.value = float(dt_micro)
            startup_stats = _run_startup_staggered_guess(
                funcs=funcs,
                prev_funcs=prev_funcs,
                aux_funcs=aux_funcs,
                bcs_now=bcs_now,
                step_no=int(step_no),
                t_fail=float(t_fail),
                dt_fail=float(dt_micro),
                outer_it_override=max(1, int(getattr(args, "pc_p2_staggered_anchor_outer_it", 1))),
            )
        except Exception as exc:
            _restore_function_values(funcs, snapshot_in)
            print(f"    [pc] G-anchor staggered micro-step failed at dt={float(dt_micro):.3e}: {exc}")
            return {
                "used": False,
                "dt_micro": float(dt_micro),
                "exception": str(exc),
            }
        finally:
            dt_c.value = float(dt_saved)
        staged_snapshot = _snapshot_function_values(funcs)
        staged_stats = _current_predictor_corrector_energy(
            funcs=funcs,
            prev_funcs=prev_funcs,
            aux_funcs=aux_funcs,
            bcs_now=bcs_now,
        )
        progress = _pc_exact_progress(best_stats, staged_stats)
        keep = bool(progress["improved"]) and _predictor_alpha_mass_ok(staged_stats)
        if not keep:
            _restore_function_values(funcs, snapshot_in)
            staged_snapshot = snapshot_in
            staged_stats = dict(best_stats)
        print(
            "    [pc] G-anchor staggered micro-step "
            f"(dt={float(dt_micro):.3e}) |R_raw|_∞ {float(best_stats['raw_inf']):.3e} "
            f"-> {float(staged_stats['raw_inf']):.3e}, "
            f"alpha_mass_rel={float(staged_stats['alpha_mass_rel']):.3e}, "
            f"kept={int(bool(keep))}, sweeps={int(startup_stats.get('outer_it', 0) if 'startup_stats' in locals() else 0)}."
        )
        return {
            "used": bool(keep),
            "dt_micro": float(dt_micro),
            "startup_stats": dict(startup_stats),
            "snapshot": staged_snapshot,
            "stats": dict(staged_stats),
            "exception": None,
        }

    def _run_predictor_corrector_p2_lambda_attempt(
        *,
        funcs,
        prev_funcs,
        aux_funcs,
        bcs_now,
        p2_solver,
        p2_bcs_now,
        p2_lambda,
        lam_from: float | None,
        lam: float,
        stage_label: str,
        anchor_snapshot,
        exact_reference_stats: dict[str, float] | None = None,
    ) -> dict[str, object]:
        predictor_info = None
        lam_from_eff = None if lam_from is None else float(lam_from)
        if lam_from_eff is not None and abs(float(lam) - lam_from_eff) > 1.0e-15:
            predictor_info = _run_predictor_corrector_p2_path_predictor(
                funcs=funcs,
                prev_funcs=prev_funcs,
                aux_funcs=aux_funcs,
                bcs_now=bcs_now,
                p2_solver=p2_solver,
                p2_bcs_now=p2_bcs_now,
                p2_lambda=p2_lambda,
                lam_from=float(lam_from_eff),
                lam_to=float(lam),
                stage_label=stage_label,
            )
        if p2_lambda is not None:
            p2_lambda.value = float(lam)
        lam_before_snapshot = _snapshot_function_values(funcs)
        lam_before = _current_predictor_corrector_energy(
            funcs=funcs,
            prev_funcs=prev_funcs,
            aux_funcs=aux_funcs,
            bcs_now=bcs_now,
            stage_solver=p2_solver,
            stage_bcs_now=p2_bcs_now,
        )
        print(
            f"    [pc] {stage_label} solve at lambda={float(lam):.3f} "
            f"from |H|_∞={float(lam_before['homotopy_raw_inf']):.3e}, "
            f"|R_raw|_∞={float(lam_before['raw_inf']):.3e}."
        )
        lam_converged = False
        lam_exception = None
        lam_iters = 0
        try:
            _set_solver_newton_trace_context(
                p2_solver,
                label=f"pc:{str(stage_label)}(lambda={float(lam):.3f})",
                global_bcs_now=bcs_now,
            )
            _, lam_converged, lam_iters = p2_solver._newton_loop(
                funcs,
                prev_funcs,
                aux_funcs,
                p2_bcs_now,
            )
        except Exception as exc:
            lam_exception = exc
            lam_iters = max(1, int(getattr(args, "pc_p2_max_it", 8)))
        lam_return = _apply_predictor_alpha_mass_return_map(
            funcs=funcs,
            prev_funcs=prev_funcs,
            bcs_now=bcs_now,
            stage_label=f"{stage_label}(lambda={float(lam):.3f})",
            verbose=True,
        )
        lam_trial_snapshot = _snapshot_function_values(funcs)
        lam_trial = _current_predictor_corrector_energy(
            funcs=funcs,
            prev_funcs=prev_funcs,
            aux_funcs=aux_funcs,
            bcs_now=bcs_now,
            stage_solver=p2_solver,
            stage_bcs_now=p2_bcs_now,
        )
        _project_predictor_segment(
            funcs=funcs,
            prev_funcs=prev_funcs,
            aux_funcs=aux_funcs,
            bcs_now=bcs_now,
            anchor_snapshot=anchor_snapshot,
            trial_snapshot=lam_trial_snapshot,
            stage_label=f"{stage_label}(lambda={float(lam):.3f})",
            stage_solver=p2_solver,
            stage_bcs_now=p2_bcs_now,
            primary_key="homotopy_raw_inf",
        )
        lam_projected_snapshot = _snapshot_function_values(funcs)
        lam_projected = _current_predictor_corrector_energy(
            funcs=funcs,
            prev_funcs=prev_funcs,
            aux_funcs=aux_funcs,
            bcs_now=bcs_now,
            stage_solver=p2_solver,
            stage_bcs_now=p2_bcs_now,
        )
        lam_keep, lam_decision = _pc_should_keep_lambda_stage(
            lam=float(lam),
            before_stats=lam_before,
            after_stats=lam_projected,
            exact_reference_stats=exact_reference_stats,
            alpha_mass_ok=_predictor_alpha_mass_ok(lam_projected),
            min_abs_decrease=float(getattr(args, "pc_min_abs_decrease", 1.0e-10) or 0.0),
            min_rel_improve=float(getattr(args, "pc_min_rel_improve", 0.0) or 0.0),
            max_exact_worsen_rel=float(getattr(args, "pc_p2_max_exact_worsen_rel", 5.0e-2) or 0.0),
            homotopy_tol=max(float(getattr(args, "newton_tol", 1.0e-6) or 1.0e-6), 1.0e-12),
        )
        exact_progress = dict(lam_decision.get("exact", {}))
        homotopy_progress = dict(lam_decision.get("homotopy", {}))
        print(
            f"    [pc] {stage_label} lambda result "
            f"|H|_∞ {float(lam_before['homotopy_raw_inf']):.3e} -> {float(lam_projected['homotopy_raw_inf']):.3e}, "
            f"|R_raw|_∞ {float(lam_before['raw_inf']):.3e} -> {float(lam_projected['raw_inf']):.3e}, "
            f"alpha_mass_rel={float(lam_projected['alpha_mass_rel']):.3e}, "
            f"H_drop={float(homotopy_progress.get('drop', float('nan'))):.3e}, "
            f"H_req={float(homotopy_progress.get('required', float('nan'))):.3e}, "
            f"R_drop={float(exact_progress.get('drop', float('nan'))):.3e}, "
            f"R_req={float(exact_progress.get('required', float('nan'))):.3e}, "
            f"R_ref={float(lam_decision.get('exact_reference', float('nan'))):.3e}, "
            f"R_cap={float(lam_decision.get('exact_guard_limit', float('nan'))):.3e}, "
            f"iters={int(lam_iters)}, converged={int(bool(lam_converged))}, kept={int(bool(lam_keep))}."
        )
        if not lam_keep:
            _restore_function_values(funcs, anchor_snapshot)
        return {
            "before": dict(lam_before),
            "trial_after": dict(lam_trial),
            "projected_after": dict(lam_projected),
            "projected_snapshot": lam_projected_snapshot,
            "keep": bool(lam_keep),
            "iters": int(lam_iters),
            "converged": bool(lam_converged),
            "exception": lam_exception,
            "return_map": dict(lam_return),
            "progress": dict(homotopy_progress),
            "decision": dict(lam_decision),
            "predictor": predictor_info,
        }

    def _attempt_predictor_corrector_p2_reentry(
        *,
        funcs,
        prev_funcs,
        aux_funcs,
        bcs_now,
        step_no: int,
        t_fail: float,
        dt_fail: float,
        reason: str,
    ) -> dict[str, object] | None:
        if not _predictor_corrector_startup_enabled(args, problem):
            return None
        p2_solver = _prime_predictor_corrector_p2_solver()
        if p2_solver is None:
            p2_solver = _get_predictor_corrector_p2_solver()
        _configure_predictor_corrector_p2_solver(
            p2_solver=p2_solver,
            dt_now=float(dt_fail),
        )
        p2_lambda = getattr(p2_solver, "_pc_p2_lambda", None)
        p2_fields = _startup_predictor_p2_fields()
        p2_bcs_now = _filter_bcs_by_fields(bcs_now, p2_fields)
        p2_solver._current_t = float(t_fail)
        p2_solver._current_dt = float(dt_fail)
        p2_solver._current_step_no = int(step_no)
        base_snapshot = _snapshot_function_values(funcs)
        base_stats = _current_predictor_corrector_energy(
            funcs=funcs,
            prev_funcs=prev_funcs,
            aux_funcs=aux_funcs,
            bcs_now=bcs_now,
        )
        exact_reference_stats = dict(base_stats)
        anchor_attempt = _run_predictor_corrector_p2_lambda_attempt(
            funcs=funcs,
            prev_funcs=prev_funcs,
            aux_funcs=aux_funcs,
            bcs_now=bcs_now,
            p2_solver=p2_solver,
            p2_bcs_now=p2_bcs_now,
            p2_lambda=p2_lambda,
            lam_from=None,
            lam=0.0,
            stage_label="P2-reentry-anchor",
            anchor_snapshot=base_snapshot,
            exact_reference_stats=exact_reference_stats,
        )
        anchor_kept = bool(anchor_attempt["keep"])
        if anchor_kept:
            base_snapshot = anchor_attempt["projected_snapshot"]
            base_stats = dict(anchor_attempt["projected_after"])
        lam0_raw = float(getattr(args, "pc_p2_reentry_lambda0", 0.0) or 0.0)
        if not np.isfinite(lam0_raw) or lam0_raw <= 0.0:
            lam0_raw = float(_predictor_p2_initial_lambda_step())
        lam0_cfg = min(max(lam0_raw, 1.0e-8), 1.0 - 1.0e-8)
        lam_hint = startup_retry_state.get("pc_last_successful_lambda", None)
        try:
            lam_hint_val = float(lam_hint) if lam_hint is not None else float("nan")
        except Exception:
            lam_hint_val = float("nan")
        if np.isfinite(lam_hint_val) and lam_hint_val > 0.0:
            lam0_cfg = min(lam0_cfg, max(lam_hint_val, 1.0e-8))
        lam = float(lam0_cfg)
        lam_min = min(max(float(getattr(args, "pc_p2_reentry_lambda_min", 1.0e-3) or 1.0e-3), 1.0e-8), lam)
        attempt_no = 0
        while lam >= lam_min - 1.0e-15:
            attempt_no += 1
            _restore_function_values(funcs, base_snapshot)
            attempt = _run_predictor_corrector_p2_lambda_attempt(
                funcs=funcs,
                prev_funcs=prev_funcs,
                aux_funcs=aux_funcs,
                bcs_now=bcs_now,
                p2_solver=p2_solver,
                p2_bcs_now=p2_bcs_now,
                p2_lambda=p2_lambda,
                lam_from=(0.0 if anchor_kept else None),
                lam=float(lam),
                stage_label="P2-reentry",
                anchor_snapshot=base_snapshot,
                exact_reference_stats=exact_reference_stats,
            )
            if bool(attempt["keep"]):
                if p2_lambda is not None:
                    p2_lambda.value = 1.0
                startup_retry_state["pc_last_successful_lambda"] = float(lam)
                improved_stats = dict(attempt["projected_after"])
                print(
                    "    [pc] exact-corrector stall handed control back to P2 and recovered a new homotopy state "
                    f"({reason}; lambda={float(lam):.3f}, |H|_∞ {float(base_stats['homotopy_raw_inf']):.3e} -> "
                    f"{float(improved_stats['homotopy_raw_inf']):.3e}, |R_raw|_∞ {float(base_stats['raw_inf']):.3e} -> "
                    f"{float(improved_stats['raw_inf']):.3e})."
                )
                return {
                    "mode": "reentry",
                    "reason": str(reason),
                    "lambda": float(lam),
                    "attempts": int(attempt_no),
                    "before": dict(base_stats),
                    "after": dict(improved_stats),
                    "iters": int(attempt["iters"]),
                    "converged": bool(attempt["converged"]),
                    "exception": (None if attempt["exception"] is None else str(attempt["exception"])),
                }
            lam *= 0.5
        if anchor_kept:
            if p2_lambda is not None:
                p2_lambda.value = 1.0
            _restore_function_values(funcs, base_snapshot)
            print(
                "    [pc] exact-corrector stall re-entered P2 and kept the lambda=0 anchor state "
                f"({reason}; |H|_∞={float(base_stats['homotopy_raw_inf']):.3e}, |R_raw|_∞={float(base_stats['raw_inf']):.3e})."
            )
            return {
                "mode": "reentry_anchor",
                "reason": str(reason),
                "lambda": 0.0,
                "attempts": int(attempt_no),
                "before": dict(base_stats),
                "after": dict(base_stats),
                "iters": int(anchor_attempt["iters"]),
                "converged": bool(anchor_attempt["converged"]),
                "exception": (None if anchor_attempt["exception"] is None else str(anchor_attempt["exception"])),
            }
        if p2_lambda is not None:
            p2_lambda.value = 1.0
        _restore_function_values(funcs, base_snapshot)
        print(
            "    [pc] exact-corrector stall re-entered P2 but no lambda stage produced a meaningful homotopy improvement; "
            f"reason={reason}."
        )
        return None

    def _run_predictor_corrector_startup_guess(
        *,
        funcs,
        prev_funcs,
        aux_funcs,
        bcs_now,
        step_no: int,
        t_fail: float,
        dt_fail: float,
    ) -> dict[str, object]:
        if int(step_no) != 1:
            raise RuntimeError("predictor-corrector startup is only defined for step 1.")
        _copy_prev_into_current(funcs, prev_funcs)
        initial_snapshot = _snapshot_function_values(funcs)
        initial_stats = _current_predictor_corrector_energy(
            funcs=funcs,
            prev_funcs=prev_funcs,
            aux_funcs=aux_funcs,
            bcs_now=bcs_now,
        )
        best_snapshot = initial_snapshot
        best_stats = dict(initial_stats)
        stage_log: list[dict[str, object]] = []
        if bool(args.enable_phi_evolution) and problem.get("phi_k") is not None:
            anchor_info = _run_predictor_corrector_p2_staggered_anchor(
                funcs=funcs,
                prev_funcs=prev_funcs,
                aux_funcs=aux_funcs,
                bcs_now=bcs_now,
                step_no=int(step_no),
                t_fail=float(t_fail),
                dt_fail=float(dt_fail),
                best_snapshot=best_snapshot,
                best_stats=best_stats,
            )
            if anchor_info is not None:
                if bool(anchor_info.get("used", False)):
                    best_snapshot = anchor_info["snapshot"]
                    best_stats = dict(anchor_info["stats"])
                else:
                    _restore_function_values(funcs, best_snapshot)
                stage_log.append(
                    {
                        "stage": "G0",
                        "before": dict(initial_stats),
                        "after": dict(anchor_info.get("stats", best_stats)),
                        "kept": bool(anchor_info.get("used", False)),
                        "detail": dict(anchor_info.get("startup_stats", {})),
                        "dt_micro": float(anchor_info.get("dt_micro", float("nan"))),
                        "exception": anchor_info.get("exception", None),
                    }
                )
                if bool(anchor_info.get("used", False)):
                    _restore_function_values(funcs, best_snapshot)
                    g_exact_before_snapshot = _snapshot_function_values(funcs)
                    g_exact_before = dict(best_stats)
                    g_exact_probe = _run_predictor_corrector_exact_probe(
                        funcs=funcs,
                        prev_funcs=prev_funcs,
                        aux_funcs=aux_funcs,
                        bcs_now=bcs_now,
                        before_snapshot=g_exact_before_snapshot,
                        before_stats=g_exact_before,
                        stage_label="G-exact",
                    )
                    stage_log.append(
                        {
                            "stage": "G-exact",
                            "before": dict(g_exact_before),
                            "after": dict(g_exact_probe.get("after", g_exact_before)),
                            "converged": bool(g_exact_probe.get("converged", False)),
                            "iters": int(g_exact_probe.get("iters", 0)),
                            "exception": g_exact_probe.get("exception", None),
                            "kept": bool(g_exact_probe.get("keep", False)),
                            "decision": dict(g_exact_probe.get("decision", {})),
                        }
                    )
                    if bool(g_exact_probe.get("keep", False)):
                        best_snapshot = g_exact_probe["snapshot"]
                        best_stats = dict(g_exact_probe["after"])
                        _restore_function_values(funcs, best_snapshot)
                        print(
                            "    [pc] G-exact already reduced the exact monolithic residual from the kept "
                            "G-anchor basin; skipping P0/P1/P2 continuation on this startup pass."
                        )
                        return {
                            "initial": dict(initial_stats),
                            "final": dict(best_stats),
                            "stages": stage_log,
                        }
                    _restore_function_values(funcs, best_snapshot)
                    if _logistic_refmap_phi_only_mode(args):
                        print(
                            "    [pc] keeping the best kept G-anchor basin on the refmap phi-only logistic "
                            "branch and continuing into P0/P1/P2 because G-exact did not improve enough to "
                            "replace the continuation stages."
                        )

        p0_before = dict(best_stats)
        p0_snapshot_in = _snapshot_function_values(funcs)
        p0_exception = None
        try:
            p0_stats = _run_frozen_transport_restart(
                funcs=funcs,
                prev_funcs=prev_funcs,
                aux_funcs=aux_funcs,
                bcs_now=bcs_now,
                step_no=int(step_no),
                t_fail=float(t_fail),
                dt_fail=float(dt_fail),
                outer_it_override=max(1, int(getattr(args, "pc_p0_outer_it", 2))),
            )
            p0_return = _apply_predictor_alpha_mass_return_map(
                funcs=funcs,
                prev_funcs=prev_funcs,
                bcs_now=bcs_now,
                stage_label="P0",
                verbose=True,
            )
            p0_after = _current_predictor_corrector_energy(
                funcs=funcs,
                prev_funcs=prev_funcs,
                aux_funcs=aux_funcs,
                bcs_now=bcs_now,
            )
            p0_progress = _pc_exact_progress(p0_before, p0_after)
            if bool(p0_progress["improved"]) and _predictor_alpha_mass_ok(p0_after):
                best_snapshot = _snapshot_function_values(funcs)
                best_stats = dict(p0_after)
                kept_p0 = True
            else:
                _restore_function_values(funcs, p0_snapshot_in)
                kept_p0 = False
        except Exception as exc:
            p0_exception = exc
            _restore_function_values(funcs, p0_snapshot_in)
            p0_stats = {
                "outer_it": 0,
                "requested_outer_it": max(1, int(getattr(args, "pc_p0_outer_it", 2))),
                "flow_iters_total": 0,
                "solid_iters_total": 0,
                "flow_iters_hist": tuple(),
                "solid_iters_hist": tuple(),
                "partial_success": False,
                "partial_reason": str(exc),
                "residual_before": float(p0_before["raw_inf"]),
                "residual_after": float(p0_before["raw_inf"]),
            }
            p0_return = {
                "applied": False,
                "reason": f"P0 predictor failed before return mapping: {exc}",
            }
            p0_after = dict(p0_before)
            kept_p0 = False
        stage_log.append(
            {
                "stage": "P0",
                "before": dict(p0_before),
                "after": dict(p0_after),
                "kept": bool(kept_p0),
                "detail": dict(p0_stats),
                "return_map": dict(p0_return),
                "exception": (None if p0_exception is None else str(p0_exception)),
            }
        )
        print(
            "    [pc] P0 frozen-transport predictor "
            f"|R_raw|_∞ {float(p0_before['raw_inf']):.3e} -> {float(p0_after['raw_inf']):.3e}, "
            f"alpha_mass_rel {float(p0_after['alpha_mass_rel']):.3e}, kept={int(bool(kept_p0))}."
        )
        if p0_exception is not None:
            print(f"    [pc] P0 predictor stopped before full convergence: {p0_exception}")

        _restore_function_values(funcs, best_snapshot)
        p1_solver = _get_predictor_corrector_p1_solver()
        p1_fields = _startup_predictor_p1_fields()
        p1_bcs_now = _filter_bcs_by_fields(bcs_now, p1_fields)
        p1_solver._current_t = float(t_fail)
        p1_solver._current_dt = float(dt_fail)
        p1_solver._current_step_no = int(step_no)
        p1_before_snapshot = _snapshot_function_values(funcs)
        p1_before = dict(best_stats)
        problem["dh"].apply_bcs(p1_bcs_now, *funcs)
        p1_exception = None
        try:
            _set_solver_newton_trace_context(
                p1_solver,
                label="pc:P1",
                global_bcs_now=bcs_now,
            )
            _, p1_converged, p1_iters = p1_solver._newton_loop(
                funcs,
                prev_funcs,
                aux_funcs,
                p1_bcs_now,
            )
        except Exception as exc:
            p1_exception = exc
            p1_converged = False
            p1_iters = max(1, int(getattr(args, "pc_p1_max_it", 10)))
        p1_return = _apply_predictor_alpha_mass_return_map(
            funcs=funcs,
            prev_funcs=prev_funcs,
            bcs_now=bcs_now,
            stage_label="P1",
            verbose=True,
        )
        p1_trial_snapshot = _snapshot_function_values(funcs)
        p1_after = _current_predictor_corrector_energy(
            funcs=funcs,
            prev_funcs=prev_funcs,
            aux_funcs=aux_funcs,
            bcs_now=bcs_now,
        )
        _project_predictor_segment(
            funcs=funcs,
            prev_funcs=prev_funcs,
            aux_funcs=aux_funcs,
            bcs_now=bcs_now,
            anchor_snapshot=p1_before_snapshot,
            trial_snapshot=p1_trial_snapshot,
            stage_label="P1",
        )
        p1_projected_snapshot = _snapshot_function_values(funcs)
        p1_projected = _current_predictor_corrector_energy(
            funcs=funcs,
            prev_funcs=prev_funcs,
            aux_funcs=aux_funcs,
            bcs_now=bcs_now,
        )
        p1_progress = _pc_exact_progress(p1_before, p1_projected)
        keep_p1 = (
            bool(p1_progress["improved"])
            and _predictor_alpha_mass_ok(p1_projected)
        )
        if keep_p1:
            best_snapshot = p1_projected_snapshot
            best_stats = dict(p1_projected)
        else:
            _restore_function_values(funcs, p1_before_snapshot)
        stage_log.append(
            {
                "stage": "P1",
                "before": dict(p1_before),
                "trial_after": dict(p1_after),
                "projected_after": dict(p1_projected),
                "converged": bool(p1_converged),
                "iters": int(p1_iters),
                "exception": (None if p1_exception is None else str(p1_exception)),
                "kept": bool(keep_p1),
                "return_map": dict(p1_return),
            }
        )
        print(
            "    [pc] P1 alpha-released predictor "
            f"|R_raw|_∞ {float(p1_before['raw_inf']):.3e} -> {float(p1_projected['raw_inf']):.3e}, "
            f"alpha_mass_rel {float(p1_projected['alpha_mass_rel']):.3e}, "
            f"iters={int(p1_iters)}, converged={int(bool(p1_converged))}, kept={int(bool(keep_p1))}."
        )
        if p1_exception is not None:
            print(f"    [pc] P1 predictor stopped before full convergence: {p1_exception}")

        if bool(args.enable_phi_evolution) and problem.get("phi_k") is not None:
            _restore_function_values(funcs, best_snapshot)
            p_exact_before_snapshot = _snapshot_function_values(funcs)
            p_exact_before = dict(best_stats)
            exact_probe = _run_predictor_corrector_exact_probe(
                funcs=funcs,
                prev_funcs=prev_funcs,
                aux_funcs=aux_funcs,
                bcs_now=bcs_now,
                before_snapshot=p_exact_before_snapshot,
                before_stats=p_exact_before,
                stage_label="P-exact",
            )
            stage_log.append(
                {
                    "stage": "P-exact",
                    "before": dict(p_exact_before),
                    "after": dict(exact_probe.get("after", p_exact_before)),
                    "converged": bool(exact_probe.get("converged", False)),
                    "iters": int(exact_probe.get("iters", 0)),
                    "exception": exact_probe.get("exception", None),
                    "kept": bool(exact_probe.get("keep", False)),
                    "decision": dict(exact_probe.get("decision", {})),
                }
            )
            if bool(exact_probe.get("keep", False)):
                best_snapshot = exact_probe["snapshot"]
                best_stats = dict(exact_probe["after"])
                _restore_function_values(funcs, best_snapshot)
                print(
                    "    [pc] P-exact already reduced the exact monolithic residual from the P1 basin; "
                    "skipping P2 continuation on this startup pass."
                )
                return {
                    "initial": dict(initial_stats),
                    "final": dict(best_stats),
                    "stages": stage_log,
                }
            _restore_function_values(funcs, best_snapshot)
            _prime_predictor_corrector_p2_solver()
            p2_solver = _get_predictor_corrector_p2_solver()
            p2_easy_info = _configure_predictor_corrector_p2_solver(
                p2_solver=p2_solver,
                dt_now=float(dt_fail),
            )
            p2_lambda = getattr(p2_solver, "_pc_p2_lambda", None)
            p2_fields = _startup_predictor_p2_fields()
            p2_bcs_now = _filter_bcs_by_fields(bcs_now, p2_fields)
            p2_solver._current_t = float(t_fail)
            p2_solver._current_dt = float(dt_fail)
            p2_solver._current_step_no = int(step_no)
            p2_before_snapshot = _snapshot_function_values(funcs)
            p2_before = dict(best_stats)
            problem["dh"].apply_bcs(p2_bcs_now, *funcs)
            p2_exception = None
            p2_converged = True
            p2_iters = 0
            anchor_kept = False
            completed_lambdas = 0
            p2_best_snapshot = p2_before_snapshot
            p2_best_stats = dict(p2_before)
            p2_reference_stats = dict(p2_before)
            p2_lambda_attempts: list[float] = []
            p2_lambda_kept: list[float] = []
            lam_curr = 0.0
            anchor_attempt = _run_predictor_corrector_p2_lambda_attempt(
                funcs=funcs,
                prev_funcs=prev_funcs,
                aux_funcs=aux_funcs,
                bcs_now=bcs_now,
                p2_solver=p2_solver,
                p2_bcs_now=p2_bcs_now,
                p2_lambda=p2_lambda,
                lam_from=0.0,
                lam=0.0,
                stage_label="P2-anchor",
                anchor_snapshot=p2_before_snapshot,
                exact_reference_stats=p2_reference_stats,
            )
            p2_iters += int(anchor_attempt["iters"])
            if bool(anchor_attempt["keep"]):
                anchor_kept = True
                p2_best_snapshot = anchor_attempt["projected_snapshot"]
                p2_best_stats = dict(anchor_attempt["projected_after"])
                _restore_function_values(funcs, p2_best_snapshot)
            else:
                _restore_function_values(funcs, p2_before_snapshot)
            lam_step = float(_predictor_p2_initial_lambda_step())
            lam_step = min(max(lam_step, 1.0e-8), 1.0)
            lam_growth = max(1.0, float(getattr(args, "pc_p2_lambda_growth", 1.5) or 1.5))
            lam_shrink = min(0.99, max(1.0e-3, float(getattr(args, "pc_p2_lambda_shrink", 0.5) or 0.5)))
            lam_min_step = max(1.0e-8, float(getattr(args, "pc_p2_lambda_min_step", 1.0e-3) or 1.0e-3))
            lam_max_substeps = max(1, int(getattr(args, "pc_p2_max_substeps", 32)))
            while lam_curr < 1.0 - 1.0e-14 and completed_lambdas < lam_max_substeps:
                lam = min(1.0, lam_curr + lam_step)
                p2_lambda_attempts.append(float(lam))
                attempt = _run_predictor_corrector_p2_lambda_attempt(
                    funcs=funcs,
                    prev_funcs=prev_funcs,
                    aux_funcs=aux_funcs,
                    bcs_now=bcs_now,
                    p2_solver=p2_solver,
                    p2_bcs_now=p2_bcs_now,
                    p2_lambda=p2_lambda,
                    lam_from=float(lam_curr),
                    lam=float(lam),
                    stage_label="P2",
                    anchor_snapshot=p2_best_snapshot,
                    exact_reference_stats=p2_reference_stats,
                )
                p2_iters += int(attempt["iters"])
                lam_keep = bool(attempt["keep"])
                if lam_keep:
                    p2_best_snapshot = attempt["projected_snapshot"]
                    p2_best_stats = dict(attempt["projected_after"])
                    p2_lambda_kept.append(float(lam))
                    startup_retry_state["pc_last_successful_lambda"] = float(lam)
                    lam_curr = float(lam)
                    completed_lambdas += 1
                    if attempt["exception"] is not None:
                        p2_exception = attempt["exception"]
                        p2_converged = False
                        print(
                            f"    [pc] P2 lambda={float(lam):.3f} improved the homotopy state but stopped early; "
                            "continuing the homotopy from the improved state."
                        )
                        lam_step = min(lam_step, max(1.0 - lam_curr, lam_min_step))
                    elif not bool(attempt["converged"]):
                        p2_exception = RuntimeError(f"P2 lambda={float(lam):.3f} stopped before convergence.")
                        p2_converged = False
                        print(
                            f"    [pc] P2 lambda={float(lam):.3f} improved the homotopy state without full convergence; "
                            "continuing the homotopy from the improved state."
                        )
                        lam_step = min(lam_step, max(1.0 - lam_curr, lam_min_step))
                    else:
                        lam_step = min(max(lam_step * lam_growth, lam_min_step), max(1.0 - lam_curr, lam_min_step))
                    if lam_curr >= 1.0 - 1.0e-14:
                        break
                    continue
                next_step = float(lam_step * lam_shrink)
                if not np.isfinite(next_step) or next_step < lam_min_step:
                    p2_exception = attempt["exception"] if attempt["exception"] is not None else RuntimeError(
                        f"P2 lambda={float(lam):.3f} did not improve the homotopy residual."
                    )
                    p2_converged = False
                    break
                lam_step = min(next_step, max(1.0 - lam_curr, lam_min_step))
                p2_exception = attempt["exception"] if attempt["exception"] is not None else RuntimeError(
                    f"P2 lambda={float(lam):.3f} did not improve the homotopy residual; shrinking the homotopy step."
                )
            if p2_lambda is not None:
                p2_lambda.value = 1.0
            _restore_function_values(funcs, p2_best_snapshot)
            p2_return = _apply_predictor_alpha_mass_return_map(
                funcs=funcs,
                prev_funcs=prev_funcs,
                bcs_now=bcs_now,
                stage_label="P2-final",
                verbose=True,
            )
            p2_trial_snapshot = _snapshot_function_values(funcs)
            p2_after = _current_predictor_corrector_energy(
                funcs=funcs,
                prev_funcs=prev_funcs,
                aux_funcs=aux_funcs,
                bcs_now=bcs_now,
            )
            p2_projected_snapshot = p2_trial_snapshot
            p2_projected = dict(p2_after)
            p2_progress = _pc_exact_progress(p2_before, p2_projected)
            p2_exact_cap = float(p2_reference_stats["raw_inf"]) * (
                1.0 + max(0.0, float(getattr(args, "pc_p2_max_exact_worsen_rel", 5.0e-2) or 0.0))
            )
            p2_stage_advanced = bool(anchor_kept or completed_lambdas > 0)
            keep_p2 = bool(_predictor_alpha_mass_ok(p2_projected)) and bool(
                (p2_stage_advanced or p2_progress["improved"])
                and np.isfinite(float(p2_projected["raw_inf"]))
                and float(p2_projected["raw_inf"]) <= float(p2_exact_cap)
            )
            if keep_p2:
                best_snapshot = p2_projected_snapshot
                best_stats = dict(p2_projected)
            else:
                _restore_function_values(funcs, p2_before_snapshot)
            stage_log.append(
                {
                    "stage": "P2",
                    "before": dict(p2_before),
                    "trial_after": dict(p2_after),
                    "projected_after": dict(p2_projected),
                    "converged": bool(p2_converged),
                    "iters": int(p2_iters),
                    "exception": (None if p2_exception is None else str(p2_exception)),
                    "kept": bool(keep_p2),
                    "anchor_kept": bool(anchor_kept),
                    "completed_lambdas": int(completed_lambdas),
                    "lambda_attempts": tuple(float(v) for v in p2_lambda_attempts),
                    "lambda_kept": tuple(float(v) for v in p2_lambda_kept),
                    "anchor_dt": float(p2_easy_info["anchor_dt"]),
                    "easy_fluid_convection": str(p2_easy_info["fluid_convection"]),
                    "easy_include_skeleton_acceleration": float(p2_easy_info["include_skeleton_acceleration"]),
                    "fluid_convection": str(getattr(args, "pc_p2_fluid_convection", "lagged")),
                    "skeleton_inertia_convection": str(getattr(args, "pc_p2_skeleton_inertia_convection", "lagged")),
                    "return_map": dict(p2_return),
                }
            )
            print(
                "    [pc] P2 full-field continuation predictor "
                f"|R_raw|_∞ {float(p2_before['raw_inf']):.3e} -> {float(p2_projected['raw_inf']):.3e}, "
                f"|H|_∞ {float(p2_before.get('homotopy_raw_inf', p2_before['raw_inf'])):.3e} -> "
                f"{float(p2_projected.get('homotopy_raw_inf', p2_projected['raw_inf'])):.3e}, "
                f"alpha_mass_rel {float(p2_projected['alpha_mass_rel']):.3e}, "
                f"iters={int(p2_iters)}, converged={int(bool(p2_converged))}, kept={int(bool(keep_p2))}, "
                f"anchor_kept={int(bool(anchor_kept))}, "
                f"lambdas_kept={int(completed_lambdas)}, "
                f"G0_dt={float(p2_easy_info['anchor_dt']):.3e}, "
                f"G_fluid={str(p2_easy_info['fluid_convection'])}, "
                f"G_skel_accel={int(bool(p2_easy_info['include_skeleton_acceleration']))}, "
                f"F_fluid={str(getattr(args, 'pc_p2_fluid_convection', 'lagged'))}, "
                f"F_skeleton={str(getattr(args, 'pc_p2_skeleton_inertia_convection', 'lagged'))}."
            )
            if p2_exception is not None:
                print(f"    [pc] P2 predictor stopped before full convergence: {p2_exception}")

        _restore_function_values(funcs, best_snapshot)
        return {
            "initial": dict(initial_stats),
            "final": dict(best_stats),
            "stages": stage_log,
        }

    def _run_frozen_transport_restart(
        *,
        funcs,
        prev_funcs,
        aux_funcs,
        bcs_now,
        step_no: int,
        t_fail: float,
        dt_fail: float,
        outer_it_override: int | None = None,
    ):
        staged = _get_startup_staggered_solvers()
        if staged is None:
            raise RuntimeError("frozen-transport restart is unavailable for the current formulation.")
        flow_solver, _, solid_solver = staged
        flow_fields = _startup_flow_stage_fields()
        solid_fields = _startup_solid_stage_fields()
        flow_bcs_now = _filter_bcs_by_fields(bcs_now, flow_fields)
        solid_bcs_now = _filter_bcs_by_fields(bcs_now, solid_fields)
        for stage_solver in (flow_solver, solid_solver):
            stage_solver._current_t = float(t_fail)
            stage_solver._current_dt = float(dt_fail)
            stage_solver._current_step_no = int(step_no)
        accept_key = "PYCUTFEM_NEWTON_MAXITER_ACCEPT_FACTOR"
        prev_accept = os.environ.get(accept_key)
        os.environ[accept_key] = "0"
        if outer_it_override is None:
            n_outer = max(1, int(getattr(args, "stall_frozen_transport_outer_it", 1)))
        else:
            n_outer = max(1, int(outer_it_override))
        flow_hist: list[int] = []
        solid_hist: list[int] = []
        before_norm = _current_monolithic_raw_residual_inf(
            funcs=funcs,
            prev_funcs=prev_funcs,
            aux_funcs=aux_funcs,
            bcs_now=bcs_now,
        )
        aa_enabled = bool(
            getattr(getattr(solver, "vi_params", None), "anderson_acceleration", bool(args.vi_anderson_acceleration))
        )
        aa_history_len = max(
            2,
            int(getattr(getattr(solver, "vi_params", None), "anderson_history", int(args.vi_anderson_history)) or int(args.vi_anderson_history)),
        )
        if aa_enabled and n_outer < 2:
            n_outer = 2
        aa_pairs: list[dict[str, np.ndarray]] = []
        best_norm = float(before_norm)
        best_snapshot = _snapshot_function_values(funcs)
        vi_state_before = None
        export_vi_state = getattr(solver, "export_vi_state", None)
        if callable(export_vi_state):
            vi_state_before = export_vi_state()
        completed_outer = 0
        partial_reason: str | None = None
        try:
            for outer_it in range(n_outer):
                input_snapshot = _snapshot_function_values(funcs)
                try:
                    _set_solver_newton_trace_context(
                        flow_solver,
                        label=f"frozen-transport:flow[o{int(outer_it) + 1}]",
                        global_bcs_now=bcs_now,
                    )
                    problem["dh"].apply_bcs(flow_bcs_now, *funcs)
                    _, flow_converged, flow_iters = flow_solver._newton_loop(
                        funcs,
                        prev_funcs,
                        aux_funcs,
                        flow_bcs_now,
                    )
                    if not bool(flow_converged):
                        raise RuntimeError(
                            f"frozen-transport flow solve did not converge on outer sweep {int(outer_it) + 1}"
                        )
                    flow_hist.append(int(flow_iters))
                    _set_solver_newton_trace_context(
                        solid_solver,
                        label=f"frozen-transport:solid[o{int(outer_it) + 1}]",
                        global_bcs_now=bcs_now,
                    )
                    problem["dh"].apply_bcs(solid_bcs_now, *funcs)
                    _, solid_converged, solid_iters = solid_solver._newton_loop(
                        funcs,
                        prev_funcs,
                        aux_funcs,
                        solid_bcs_now,
                    )
                    if not bool(solid_converged):
                        raise RuntimeError(
                            f"frozen-transport solid solve did not converge on outer sweep {int(outer_it) + 1}"
                        )
                    solid_hist.append(int(solid_iters))
                except Exception as stage_exc:
                    candidate_snapshot = _snapshot_function_values(funcs)
                    candidate_norm = _current_monolithic_raw_residual_inf(
                        funcs=funcs,
                        prev_funcs=prev_funcs,
                        aux_funcs=aux_funcs,
                        bcs_now=bcs_now,
                    )
                    if (
                        np.isfinite(candidate_norm)
                        and (not np.isfinite(best_norm) or float(candidate_norm) < float(best_norm))
                    ):
                        best_norm = float(candidate_norm)
                        best_snapshot = candidate_snapshot
                        partial_reason = str(stage_exc)
                        _restore_function_values(funcs, best_snapshot)
                        break
                    _restore_function_values(funcs, best_snapshot)
                    if completed_outer > 0:
                        partial_reason = str(stage_exc)
                        break
                    raise
                completed_outer += 1
                kept_snapshot = _snapshot_function_values(funcs)
                kept_norm = _current_monolithic_raw_residual_inf(
                    funcs=funcs,
                    prev_funcs=prev_funcs,
                    aux_funcs=aux_funcs,
                    bcs_now=bcs_now,
                )
                plain_norm = float(kept_norm)
                if aa_enabled:
                    x_in = _snapshot_to_vector(input_snapshot)
                    g_out = _snapshot_to_vector(kept_snapshot)
                    if x_in.shape == g_out.shape and x_in.size > 0 and np.all(np.isfinite(x_in)) and np.all(np.isfinite(g_out)):
                        aa_pairs.append({"x": x_in.copy(), "g": g_out.copy()})
                        if len(aa_pairs) > aa_history_len:
                            aa_pairs = aa_pairs[-aa_history_len:]
                        z_mix = _anderson_mix_fixed_point(aa_pairs)
                        if z_mix is not None and int(z_mix.size) == int(g_out.size):
                            trial_snapshot = kept_snapshot
                            try:
                                mix_snapshot = _vector_to_snapshot(z_mix, trial_snapshot)
                                _restore_function_values(funcs, mix_snapshot)
                                mix_norm = _current_monolithic_raw_residual_inf(
                                    funcs=funcs,
                                    prev_funcs=prev_funcs,
                                    aux_funcs=aux_funcs,
                                    bcs_now=bcs_now,
                                )
                                if np.isfinite(mix_norm) and (
                                    not np.isfinite(kept_norm) or float(mix_norm) < float(kept_norm)
                                ):
                                    kept_snapshot = mix_snapshot
                                    kept_norm = float(mix_norm)
                                    aa_pairs[-1] = {"x": x_in.copy(), "g": _snapshot_to_vector(kept_snapshot)}
                                    print(
                                        "    [restart-aa] Anderson-accelerated frozen-transport sweep improved "
                                        f"|R_raw|_∞ from {float(plain_norm):.3e} "
                                        f"to {float(kept_norm):.3e} on outer sweep {int(outer_it) + 1}."
                                    )
                                else:
                                    _restore_function_values(funcs, trial_snapshot)
                            except Exception:
                                _restore_function_values(funcs, trial_snapshot)
                _restore_function_values(funcs, kept_snapshot)
                if not np.isfinite(best_norm) or (
                    np.isfinite(kept_norm) and (not np.isfinite(best_norm) or kept_norm < best_norm)
                ):
                    best_norm = float(kept_norm)
                    best_snapshot = _snapshot_function_values(funcs)
        finally:
            if prev_accept is None:
                os.environ.pop(accept_key, None)
            else:
                os.environ[accept_key] = prev_accept
        _restore_function_values(funcs, best_snapshot)
        import_vi_state = getattr(solver, "import_vi_state", None)
        if callable(import_vi_state) and vi_state_before is not None:
            import_vi_state(vi_state_before, force_once=True)
        return {
            "outer_it": int(completed_outer),
            "requested_outer_it": int(n_outer),
            "flow_iters_total": int(sum(flow_hist)),
            "solid_iters_total": int(sum(solid_hist)),
            "flow_iters_hist": tuple(int(v) for v in flow_hist),
            "solid_iters_hist": tuple(int(v) for v in solid_hist),
            "partial_success": bool(partial_reason is not None),
            "partial_reason": partial_reason,
            "residual_before": float(before_norm),
            "residual_after": float(best_norm),
        }

    def _run_step1_mechanics_recovery(
        *,
        funcs,
        prev_funcs,
        aux_funcs,
        bcs_now,
        step_no: int,
        t_fail: float,
        dt_fail: float,
        reason: str,
    ):
        if int(step_no) != 1 or not bool(args.enable_phi_evolution):
            return None
        base_outer = max(1, int(getattr(args, "stall_frozen_transport_outer_it", 1)))
        outer_it = max(base_outer, 3)
        max_cycles = 3
        target_raw = 5.0e-2
        min_rel_improve = max(
            1.0e-3,
            0.25 * float(getattr(args, "stall_frozen_transport_min_rel_improve", 0.0) or 0.0),
        )
        best_snapshot = _snapshot_function_values(funcs)
        best_norm = _current_monolithic_raw_residual_inf(
            funcs=funcs,
            prev_funcs=prev_funcs,
            aux_funcs=aux_funcs,
            bcs_now=bcs_now,
        )
        completed_cycles = 0
        last_stats: dict[str, object] | None = None
        while completed_cycles < max_cycles and np.isfinite(best_norm) and float(best_norm) > target_raw:
            cycle_in_snapshot = _snapshot_function_values(funcs)
            cycle_before = float(best_norm)
            stats = _run_frozen_transport_restart(
                funcs=funcs,
                prev_funcs=prev_funcs,
                aux_funcs=aux_funcs,
                bcs_now=bcs_now,
                step_no=int(step_no),
                t_fail=float(t_fail),
                dt_fail=float(dt_fail),
                outer_it_override=int(outer_it),
            )
            cycle_after = float(stats.get("residual_after", float("nan")))
            rel_improve = (
                (cycle_before - cycle_after) / cycle_before
                if np.isfinite(cycle_before) and cycle_before > 0.0 and np.isfinite(cycle_after)
                else float("nan")
            )
            if np.isfinite(cycle_after) and cycle_after < cycle_before and (
                cycle_before <= 0.0 or rel_improve >= min_rel_improve
            ):
                best_norm = float(cycle_after)
                best_snapshot = _snapshot_function_values(funcs)
                completed_cycles += 1
                last_stats = dict(stats)
                print(
                    f"    [startup] step-1 mechanics recovery cycle {completed_cycles}/{max_cycles} "
                    f"after {reason} improved |R_raw|_∞ {cycle_before:.3e} -> {cycle_after:.3e} "
                    f"using {int(outer_it)} frozen-transport sweep(s)."
                )
                outer_it = min(max(outer_it + 1, 3), 6)
                if best_norm <= target_raw:
                    break
            else:
                _restore_function_values(funcs, cycle_in_snapshot)
                break
        _restore_function_values(funcs, best_snapshot)
        if completed_cycles <= 0:
            return None
        return {
            "cycles": int(completed_cycles),
            "residual_after": float(best_norm),
            "last_stats": last_stats,
        }

    def _format_startup_stats(startup_stats: dict[str, object]) -> str:
        outer_done = int(startup_stats.get("outer_it", 0) or 0)
        outer_req = int(startup_stats.get("requested_outer_it", outer_done) or outer_done)
        summary = (
            f"({outer_done} sweep(s), flow {int(startup_stats['flow_iters_total'])} it "
            f"{list(startup_stats['flow_iters_hist'])}, transport {int(startup_stats['transport_iters_total'])} it "
            f"{list(startup_stats['transport_iters_hist'])}, solid {int(startup_stats['solid_iters_total'])} it "
            f"{list(startup_stats['solid_iters_hist'])}"
        )
        if bool(startup_stats.get("partial_success", False)):
            reason = str(startup_stats.get("partial_reason", "") or "").strip()
            if reason:
                summary += f"; accepted best completed sweep after later startup failure: {reason}"
            else:
                summary += "; accepted best completed sweep after later startup failure"
        elif outer_done < outer_req:
            summary += f"; requested {outer_req} sweep(s)"
        summary += ")"
        return summary

    def _format_frozen_transport_stats(stats: dict[str, object]) -> str:
        outer_done = int(stats.get("outer_it", 0) or 0)
        outer_req = int(stats.get("requested_outer_it", outer_done) or outer_done)
        before = float(stats.get("residual_before", float("nan")))
        after = float(stats.get("residual_after", float("nan")))
        summary = (
            f"({outer_done} sweep(s), flow {int(stats['flow_iters_total'])} it {list(stats['flow_iters_hist'])}, "
            f"solid {int(stats['solid_iters_total'])} it {list(stats['solid_iters_hist'])}, "
            f"|R_raw|_∞ {before:.3e} -> {after:.3e}"
        )
        if bool(stats.get("partial_success", False)):
            reason = str(stats.get("partial_reason", "") or "").strip()
            if reason:
                summary += f"; kept best completed sweep after later reduced-stage failure: {reason}"
        elif outer_done < outer_req:
            summary += f"; requested {outer_req} sweep(s)"
        summary += ")"
        return summary

    def _attempt_frozen_transport_retry(
        *,
        funcs,
        prev_funcs,
        aux_funcs,
        bcs_now,
        step_no: int,
        t_fail: float,
        dt_fail: float,
        message: str,
    ):
        startup_retry_state["frozen_transport_retry_attempts"] = int(
            startup_retry_state.get("frozen_transport_retry_attempts", 0)
        ) + 1
        print(message)
        snapshot_before = _snapshot_function_values(funcs)
        before_try = _current_monolithic_raw_residual_inf(
            funcs=funcs,
            prev_funcs=prev_funcs,
            aux_funcs=aux_funcs,
            bcs_now=bcs_now,
        )
        try:
            if int(step_no) == 1:
                recovery_stats = _run_step1_mechanics_recovery(
                    funcs=funcs,
                    prev_funcs=prev_funcs,
                    aux_funcs=aux_funcs,
                    bcs_now=bcs_now,
                    step_no=int(step_no),
                    t_fail=float(t_fail),
                    dt_fail=float(dt_fail),
                    reason="retry",
                )
                if recovery_stats is not None:
                    after_norm = float(recovery_stats.get("residual_after", float("nan")))
                    before_norm = float(before_try)
                    restart_stats = dict((recovery_stats.get("last_stats", {}) or {}))
                    restart_stats["residual_before"] = float(before_norm)
                    restart_stats["residual_after"] = float(after_norm)
                else:
                    restart_stats = _run_frozen_transport_restart(
                        funcs=funcs,
                        prev_funcs=prev_funcs,
                        aux_funcs=aux_funcs,
                        bcs_now=bcs_now,
                        step_no=int(step_no),
                        t_fail=float(t_fail),
                        dt_fail=float(dt_fail),
                    )
                    before_norm = float(restart_stats.get("residual_before", float("nan")))
                    after_norm = float(restart_stats.get("residual_after", float("nan")))
            else:
                restart_stats = _run_frozen_transport_restart(
                    funcs=funcs,
                    prev_funcs=prev_funcs,
                    aux_funcs=aux_funcs,
                    bcs_now=bcs_now,
                    step_no=int(step_no),
                    t_fail=float(t_fail),
                    dt_fail=float(dt_fail),
                )
                before_norm = float(restart_stats.get("residual_before", float("nan")))
                after_norm = float(restart_stats.get("residual_after", float("nan")))
            min_rel_improve = max(
                0.0,
                float(getattr(args, "stall_frozen_transport_min_rel_improve", 0.0) or 0.0),
            )
            rel_improve = (
                ((before_norm - after_norm) / before_norm)
                if np.isfinite(before_norm) and before_norm > 0.0 and np.isfinite(after_norm)
                else float("nan")
            )
            improved = (
                np.isfinite(before_norm)
                and np.isfinite(after_norm)
                and after_norm < before_norm
                and (before_norm <= 0.0 or rel_improve >= min_rel_improve)
            )
            if improved:
                print(
                    "    [retry] frozen-transport nonlinear restart improved the full raw residual "
                    f"{_format_frozen_transport_stats(restart_stats)}; retrying the same full step."
                )
                solver._ls_alpha_prev = 1.0
                reset_bounds_cb = getattr(solver, "_reset_benchmark7_vi_bounds_freeze", None)
                if callable(reset_bounds_cb):
                    reset_bounds_cb()
                return "retry_keep_guess"
            _restore_function_values(funcs, snapshot_before)
            print(
                "    [retry] frozen-transport nonlinear restart did not improve the full raw residual "
                f"enough {_format_frozen_transport_stats(restart_stats)}; keeping the pre-restart state."
            )
        except Exception as restart_exc:
            _restore_function_values(funcs, snapshot_before)
            print(f"    [retry] frozen-transport nonlinear restart failed: {restart_exc}")
        return None

    def _should_use_later_step_staggered_guess(*, step_no: int) -> bool:
        return _should_use_staggered_predictor_after_large_step(
            step_no=int(step_no),
            last_step_no=startup_retry_state.get("last_accepted_step_no", None),
            last_step_delta_inf=startup_retry_state.get("last_accepted_delta_inf", None),
            threshold=float(args.delta_predictor_reset_threshold),
        )

    def _get_startup_bootstrap_solver():
        target_solver = bootstrap_solver_cache.get("solver")
        if target_solver is not None:
            return target_solver
        bootstrap_forms = _build_forms(
            problem,
            qdeg=int(qdeg),
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
            gamma_div=float(effective_gamma_div),
            mechanics_nondim_mode=str(args.mechanics_nondim_mode),
            solid_volumetric_split=bool(args.solid_volumetric_split),
            solid_volumetric_penalty=float(args.solid_volumetric_penalty),
            gamma_u=float(args.gamma_u),
            u_extension_mode=str(args.u_extension),
            gamma_u_pin=float(args.gamma_u_pin),
            u_cip=float(args.u_cip),
            u_cip_weight=str(args.u_cip_weight),
            vS_cip=float(args.vS_cip),
            gamma_vS=(None if args.gamma_vS is None else float(args.gamma_vS)),
            vS_extension_mode=args.vS_ext_mode,
            gamma_vS_pin=(None if args.gamma_vS_pin is None else float(args.gamma_vS_pin)),
            D_phi=float(args.D_phi),
            phi_diffusion_weight=str(args.phi_diffusion_weight),
            gamma_phi=float(args.gamma_phi),
            phi_supg=float(args.phi_supg),
            phi_cip=float(args.phi_cip),
            alpha_supg=float(args.alpha_supg),
            alpha_cip=float(args.alpha_cip),
            alpha_regularization=str(args.alpha_regularization),
            alpha_reg_gamma=float(args.alpha_reg_gamma),
            alpha_reg_eps_normal=float(alpha_reg_eps_normal),
            alpha_reg_eps_tangent=float(alpha_reg_eps_tangent),
            alpha_reg_eta=float(args.alpha_reg_eta),
            alpha_advect_with=str(args.alpha_advect_with),
            alpha_advection_form=str(args.alpha_advection_form),
            support_physics=str(args.support_physics),
            solid_model=str(args.solid_model),
            kappa_inv_model=str(args.kappa_inv_model),
            drag_formulation=str(args.drag_formulation),
            fluid_convection=str(args.startup_bootstrap_fluid_convection),
            enable_phi_evolution=bool(args.enable_phi_evolution),
            include_skeleton_acceleration=bool(args.startup_bootstrap_include_skeleton_acceleration),
            rho_s0_tilde=float(args.rho_s0_tilde),
            skeleton_inertia_convection=str(args.skeleton_inertia_convection),
            ds_hdiv_tangential=hdiv_tangential_bc_measure,
            ds_alpha_transport=ds_alpha_transport,
            ds_B_transport=ds_B_transport,
            hdiv_tangential_gamma=float(args.hdiv_tangential_gamma),
            hdiv_tangential_method=str(args.hdiv_tangential_method),
        )
        target_solver = _make_solver(
            bootstrap_forms,
            postproc_cb=None,
            max_newton_iter=int(args.startup_bootstrap_max_it),
            accept_factor=float(startup_stage_accept_factor),
        )
        bootstrap_solver_cache["solver"] = target_solver
        return target_solver

    def _on_step_failure(**info):
        if not bool(args.startup_bootstrap):
            return None
        try:
            step_no = int(info.get("step_no", info.get("global_step_no", -1)))
        except Exception:
            step_no = -1
        try:
            t_fail = float(info.get("t", 0.0))
        except Exception:
            t_fail = 0.0
        try:
            dt_fail = float(info.get("dt", float(args.dt)))
        except Exception:
            dt_fail = float(args.dt)
        last_accepted = bool(getattr(solver, "_last_nonlinear_accepted", False))
        if startup_retry_state.get("step_no", None) != step_no:
            startup_retry_state["step_no"] = int(step_no)
            startup_retry_state["bootstrap_attempts"] = 0
            startup_retry_state["post_guess_retry_attempts"] = 0
            startup_retry_state["near_converged_retry_attempts"] = 0
            startup_retry_state["later_step_stage_retry_attempts"] = 0
            startup_retry_state["frozen_transport_retry_attempts"] = 0
            startup_retry_state["pc_p2_reentry_retry_attempts"] = 0
        last_norm = getattr(solver, "_last_nonlinear_norm", None)
        last_label = str(getattr(solver, "_last_nonlinear_norm_label", "") or "")
        exact_pc_failure = bool(
            solver_key == "newton"
            and last_label == "|R|_∞"
            and _predictor_corrector_startup_enabled(args, problem)
        )
        if (
            last_label == "|G|_∞"
            and last_norm is not None
            and np.isfinite(float(last_norm))
            and float(last_norm) <= 1.0e-3
            and last_accepted
        ):
            if int(startup_retry_state.get("near_converged_retry_attempts", 0)) < 2:
                startup_retry_state["near_converged_retry_attempts"] = int(
                    startup_retry_state.get("near_converged_retry_attempts", 0)
                ) + 1
                solver._ls_alpha_prev = 1.0
                if int(step_no) == 1:
                    _arm_startup_monolithic_budget(
                        step_no=int(step_no),
                        reason=(
                            "near-converged semismooth retry "
                            f"#{int(startup_retry_state['near_converged_retry_attempts'])}"
                        ),
                    )
                print(
                    "    [retry] semismooth norm is already small "
                    f"(|G|_∞={float(last_norm):.3e}); retrying the same step from the current iterate."
                )
                reset_bounds_cb = getattr(solver, "_reset_benchmark7_vi_bounds_freeze", None)
                if callable(reset_bounds_cb):
                    reset_bounds_cb()
                return "retry_keep_guess"
        if exact_pc_failure:
            startup_guess_applied_step_no = startup_retry_state.get("startup_guess_applied_step_no", None)
            p2_reentry_allowed = bool(
                int(step_no) > 1 or startup_guess_applied_step_no == int(step_no)
            )
            max_reentries = max(0, int(getattr(args, "pc_p2_reentry_max_retries", 0) or 0))
            if p2_reentry_allowed and max_reentries <= 0:
                print(
                    "    [retry] exact corrector stalled; skipping legacy P2 re-entry and keeping the "
                    "identified-manifold monolithic recovery path primary."
                )
            if (
                p2_reentry_allowed
                and int(startup_retry_state.get("pc_p2_reentry_retry_attempts", 0)) < max_reentries
            ):
                funcs = list(info.get("functions", []) or [])
                prev_funcs = list(info.get("prev_functions", []) or [])
                bcs_now = info.get("bcs", [])
                aux_funcs = info.get("aux_functions", aux_solver_functions)
                retry_no = int(startup_retry_state.get("pc_p2_reentry_retry_attempts", 0)) + 1
                startup_retry_state["pc_p2_reentry_retry_attempts"] = int(retry_no)
                exc_obj = info.get("exception", None)
                exc_text = str(exc_obj).strip() if exc_obj is not None else ""
                reason_parts = []
                if exc_text:
                    reason_parts.append(exc_text)
                if last_norm is not None:
                    try:
                        last_norm_val = float(last_norm)
                    except Exception:
                        last_norm_val = float("nan")
                    if np.isfinite(last_norm_val):
                        reason_parts.append(f"|R|_∞={last_norm_val:.3e}")
                reason = "; ".join(reason_parts) or "Newton exact corrector stalled"
                reentry = _attempt_predictor_corrector_p2_reentry(
                    funcs=funcs,
                    prev_funcs=prev_funcs,
                    aux_funcs=aux_funcs,
                    bcs_now=bcs_now,
                    step_no=int(step_no),
                    t_fail=float(t_fail),
                    dt_fail=float(dt_fail),
                    reason=str(reason),
                )
                if reentry is not None:
                    solver._ls_alpha_prev = 1.0
                    if int(step_no) == 1:
                        _arm_startup_monolithic_budget(
                            step_no=int(step_no),
                            reason=f"P2 re-entry retry #{int(retry_no)}",
                        )
                    reset_bounds_cb = getattr(solver, "_reset_benchmark7_vi_bounds_freeze", None)
                    if callable(reset_bounds_cb):
                        reset_bounds_cb()
                    return "retry_keep_guess"
        if bool(getattr(args, "stall_frozen_transport_restart", True)) and int(
            startup_retry_state.get("frozen_transport_retry_attempts", 0)
        ) < 1:
            vi_metrics = dict(getattr(solver, "_vi_last_metrics", {}) or {})
            startup_guess_applied_step_no = startup_retry_state.get("startup_guess_applied_step_no", None)
            if _should_use_frozen_transport_restart(
                enable_phi_evolution=bool(args.enable_phi_evolution),
                step_no=int(step_no),
                startup_guess_applied_step_no=(
                    None if startup_guess_applied_step_no is None else int(startup_guess_applied_step_no)
                ),
                metrics=vi_metrics,
                max_delta_active=int(args.stall_frozen_transport_max_delta_active),
                max_gap=float(args.stall_frozen_transport_max_gap),
                max_eq=float(args.stall_frozen_transport_max_eq),
                min_ginf=float(args.stall_frozen_transport_min_ginf),
            ):
                funcs = list(info.get("functions", []) or [])
                prev_funcs = list(info.get("prev_functions", []) or [])
                bcs_now = info.get("bcs", [])
                aux_funcs = info.get("aux_functions", aux_solver_functions)
                retry_action = _attempt_frozen_transport_retry(
                    funcs=funcs,
                    prev_funcs=prev_funcs,
                    aux_funcs=aux_funcs,
                    bcs_now=bcs_now,
                    step_no=int(step_no),
                    t_fail=float(t_fail),
                    dt_fail=float(dt_fail),
                    message=(
                        "    [retry] VI stalled with a nearly fixed transport active set; running a "
                        "frozen-transport flow/solid restart as a nonlinear preconditioner."
                    ),
                )
                if retry_action is not None:
                    return retry_action
        if (
            bool(getattr(args, "stall_frozen_transport_restart", True))
            and int(step_no) == 1
            and startup_retry_state.get("startup_guess_applied_step_no", None) == int(step_no)
            and not last_accepted
            and int(startup_retry_state.get("frozen_transport_retry_attempts", 0)) < 1
        ):
            vi_metrics = dict(getattr(solver, "_vi_last_metrics", {}) or {})
            try:
                ginf = float(vi_metrics.get("G_inf", float("nan")))
                gap = float(vi_metrics.get("active_gap_inf", float("nan")))
                eq = float(vi_metrics.get("equality_inf", float("nan")))
            except Exception:
                ginf = float("nan")
                gap = float("nan")
                eq = float("nan")
            gap_cap = max(float(getattr(args, "stall_frozen_transport_max_gap", 0.0) or 0.0), 1.0e-4)
            eq_cap = max(float(getattr(args, "stall_frozen_transport_max_eq", 0.0) or 0.0), 1.0e-8)
            ginf_min = min(float(getattr(args, "stall_frozen_transport_min_ginf", 0.0) or 0.0), 5.0e-2)
            if (
                np.isfinite(ginf)
                and np.isfinite(gap)
                and np.isfinite(eq)
                and ginf >= ginf_min
                and ginf <= 5.0e-1
                and gap <= gap_cap
                and eq <= eq_cap
            ):
                funcs = list(info.get("functions", []) or [])
                prev_funcs = list(info.get("prev_functions", []) or [])
                bcs_now = info.get("bcs", [])
                aux_funcs = info.get("aux_functions", aux_solver_functions)
                retry_action = _attempt_frozen_transport_retry(
                    funcs=funcs,
                    prev_funcs=prev_funcs,
                    aux_funcs=aux_funcs,
                    bcs_now=bcs_now,
                    step_no=int(step_no),
                    t_fail=float(t_fail),
                    dt_fail=float(dt_fail),
                    message=(
                        "    [retry] first-step monolithic restart stalled after the staged startup guess; "
                        "running the frozen-transport flow/solid nonlinear preconditioner before reducing dt."
                    ),
                )
                if retry_action is not None:
                    return retry_action
        if int(step_no) > 1 and int(startup_retry_state.get("later_step_stage_retry_attempts", 0)) < 1:
            funcs = list(info.get("functions", []) or [])
            prev_funcs = list(info.get("prev_functions", []) or [])
            bcs_now = info.get("bcs", [])
            aux_funcs = info.get("aux_functions", aux_solver_functions)
            startup_retry_state["later_step_stage_retry_attempts"] = int(
                startup_retry_state.get("later_step_stage_retry_attempts", 0)
            ) + 1
            print(
                "    [retry] later-step monolithic solve stalled; rebuilding the same step with a "
                "staggered fluid/solid refresh from the previous accepted state."
            )
            try:
                startup_stats = _run_startup_staggered_guess(
                    funcs=funcs,
                    prev_funcs=prev_funcs,
                    aux_funcs=aux_funcs,
                    bcs_now=bcs_now,
                    step_no=int(step_no),
                    t_fail=float(t_fail),
                    dt_fail=float(dt_fail),
                    outer_it_override=int(args.later_step_staggered_outer_it),
                )
                print(
                    "    [retry] later-step staggered refresh converged "
                    f"({int(startup_stats['outer_it'])} sweep(s), flow {int(startup_stats['flow_iters_total'])} it "
                    f"{list(startup_stats['flow_iters_hist'])}, transport {int(startup_stats['transport_iters_total'])} it "
                    f"{list(startup_stats['transport_iters_hist'])}, solid {int(startup_stats['solid_iters_total'])} it "
                    f"{list(startup_stats['solid_iters_hist'])}); retrying the same full step with the staged state."
                )
                solver._ls_alpha_prev = 1.0
                reset_bounds_cb = getattr(solver, "_reset_benchmark7_vi_bounds_freeze", None)
                if callable(reset_bounds_cb):
                    reset_bounds_cb()
                return "retry_keep_guess"
            except Exception as stage_exc:
                _copy_prev_into_current(funcs, prev_funcs)
                print(f"    [retry] later-step staggered refresh failed: {stage_exc}")
        if int(step_no) != 1 or abs(float(t_fail)) > 1.0e-14:
            return None
        if startup_retry_state.get("startup_guess_applied_step_no", None) == int(step_no):
            if exact_pc_failure:
                return None
            if not last_accepted:
                return None
            if int(startup_retry_state.get("post_guess_retry_attempts", 0)) >= 2:
                return None
            startup_retry_state["post_guess_retry_attempts"] = int(startup_retry_state.get("post_guess_retry_attempts", 0)) + 1
            solver._ls_alpha_prev = 1.0
            _arm_startup_monolithic_budget(
                step_no=int(step_no),
                reason=f"post-startup same-step retry #{int(startup_retry_state['post_guess_retry_attempts'])}",
            )
            print(
                "    [retry] first-step monolithic solve stalled after the startup guess; "
                "retrying the same full step from the current iterate."
            )
            reset_bounds_cb = getattr(solver, "_reset_benchmark7_vi_bounds_freeze", None)
            if callable(reset_bounds_cb):
                reset_bounds_cb()
            return "retry_keep_guess"
        if int(startup_retry_state.get("bootstrap_attempts", 0)) >= 1:
            return None

        funcs = list(info.get("functions", []) or [])
        prev_funcs = list(info.get("prev_functions", []) or [])
        bcs_now = info.get("bcs", [])
        aux_funcs = info.get("aux_functions", aux_solver_functions)
        startup_retry_state["bootstrap_attempts"] = int(startup_retry_state.get("bootstrap_attempts", 0)) + 1
        use_pc_startup = _predictor_corrector_startup_enabled(args, problem)
        if use_pc_startup:
            print(
                "    [retry] first-step solve stalled; rebuilding the same step with the predictor-corrector "
                f"startup path (P0 frozen-transport {int(max(1, int(getattr(args, 'pc_p0_outer_it', 2))))} sweep(s), "
                f"P1 alpha-released predictor max {int(max(1, int(getattr(args, 'pc_p1_max_it', 10))))} Newton steps, "
                f"P2 full-field continuation max {int(max(1, int(getattr(args, 'pc_p2_max_it', 8))))} Newton steps)."
            )
        else:
            print(
                "    [retry] first-step solve stalled; building a staggered startup initial guess "
                f"(flow with rigid solid, then alpha/phi transport, then solid with frozen fluid traction; "
                f"{int(max(1, int(args.startup_staggered_outer_it)))} outer sweep(s))."
            )
        try:
            if use_pc_startup:
                pc_stats = _run_predictor_corrector_startup_guess(
                    funcs=funcs,
                    prev_funcs=prev_funcs,
                    aux_funcs=aux_funcs,
                    bcs_now=bcs_now,
                    step_no=int(step_no),
                    t_fail=float(t_fail),
                    dt_fail=float(dt_fail),
                )
                print(
                    "    [retry] predictor-corrector startup completed "
                    f"(|R_raw|_∞ {float(pc_stats['initial']['raw_inf']):.3e} -> {float(pc_stats['final']['raw_inf']):.3e}); "
                    "retrying the same full step with the predictor state."
                )
                _arm_startup_monolithic_budget(step_no=int(step_no), reason="predictor-corrector startup retry")
            else:
                startup_stats = _run_startup_staggered_guess(
                    funcs=funcs,
                    prev_funcs=prev_funcs,
                    aux_funcs=aux_funcs,
                    bcs_now=bcs_now,
                    step_no=int(step_no),
                    t_fail=float(t_fail),
                    dt_fail=float(dt_fail),
                )
                print(
                    "    [retry] staggered startup converged "
                    f"{_format_startup_stats(startup_stats)}; "
                    "retrying the same full step with the staged state."
                )
                _arm_startup_monolithic_budget(step_no=int(step_no), reason="staggered startup retry")
            reset_bounds_cb = getattr(solver, "_reset_benchmark7_vi_bounds_freeze", None)
            if callable(reset_bounds_cb):
                reset_bounds_cb()
            return "retry_keep_guess"
        except Exception as bootstrap_exc:
            _copy_prev_into_current(funcs, prev_funcs)
            print(f"    [retry] staggered startup failed: {bootstrap_exc}")
            if not bool(args.enable_phi_evolution):
                try:
                    bootstrap_solver = _get_startup_bootstrap_solver()
                    bootstrap_solver._current_t = float(t_fail)
                    bootstrap_solver._current_dt = float(dt_fail)
                    bootstrap_solver._current_step_no = int(step_no)
                    _set_solver_newton_trace_context(
                        bootstrap_solver,
                        label="startup-bootstrap",
                    )
                    _, bootstrap_converged, bootstrap_iters = bootstrap_solver._newton_loop(
                        funcs,
                        prev_funcs,
                        aux_funcs,
                        bcs_now,
                    )
                    if not bool(bootstrap_converged):
                        raise RuntimeError("startup bootstrap solve did not converge")
                    print(
                        f"    [retry] legacy startup bootstrap converged in {int(bootstrap_iters)} iterations; "
                        "retrying the same full step with the bootstrap state."
                    )
                    _arm_startup_monolithic_budget(step_no=int(step_no), reason="legacy startup bootstrap")
                    reset_bounds_cb = getattr(solver, "_reset_benchmark7_vi_bounds_freeze", None)
                    if callable(reset_bounds_cb):
                        reset_bounds_cb()
                    return "retry_keep_guess"
                except Exception as legacy_exc:
                    _copy_prev_into_current(funcs, prev_funcs)
                    print(f"    [retry] legacy startup bootstrap failed: {legacy_exc}")
            return None

    def _step_initial_guess_callback(**info):
        if not bool(args.startup_bootstrap):
            _restore_base_monolithic_controls()
            return None
        try:
            step_no = int(info.get("step_no", -1))
        except Exception:
            step_no = -1
        try:
            t_now = float(info.get("t", 0.0))
        except Exception:
            t_now = 0.0
        if int(step_no) != 1 or abs(float(t_now)) > 1.0e-14:
            _restore_base_monolithic_controls()
            if (
                _should_use_later_step_staggered_guess(step_no=int(step_no))
                and startup_retry_state.get("later_step_stage_guess_applied_step_no", None) != int(step_no)
            ):
                funcs = list(info.get("functions", []) or [])
                prev_funcs = list(info.get("prev_functions", []) or [])
                bcs_now = info.get("bcs", [])
                aux_funcs = info.get("aux_functions", aux_solver_functions)
                dt_now = float(info.get("dt", float(args.dt)))
                last_delta_inf = startup_retry_state.get("last_accepted_delta_inf", None)
                print(
                    "    [startup] previous accepted step was large "
                    f"(ΔU_step∞={float(last_delta_inf):.3e}); replacing the later-step delta predictor "
                    f"with a staggered flow/transport/solid guess ({int(max(1, int(args.later_step_staggered_outer_it)))} outer sweep(s))."
                )
                startup_stats = _run_startup_staggered_guess(
                    funcs=funcs,
                    prev_funcs=prev_funcs,
                    aux_funcs=aux_funcs,
                    bcs_now=bcs_now,
                    step_no=int(step_no),
                    t_fail=float(t_now),
                    dt_fail=float(dt_now),
                    outer_it_override=int(args.later_step_staggered_outer_it),
                )
                startup_retry_state["later_step_stage_guess_applied_step_no"] = int(step_no)
                solver._ls_alpha_prev = 1.0
                reset_bounds_cb = getattr(solver, "_reset_benchmark7_vi_bounds_freeze", None)
                if callable(reset_bounds_cb):
                    reset_bounds_cb()
                print(
                    "    [startup] later-step staggered guess converged "
                    f"{_format_startup_stats(startup_stats)}."
                )
            return None
        if startup_retry_state.get("startup_guess_applied_step_no", None) == int(step_no):
            return None
        funcs = list(info.get("functions", []) or [])
        prev_funcs = list(info.get("prev_functions", []) or [])
        bcs_now = info.get("bcs", [])
        aux_funcs = info.get("aux_functions", aux_solver_functions)
        dt_now = float(info.get("dt", float(args.dt)))
        if _predictor_corrector_startup_enabled(args, problem):
            print(
                "    [startup] building first-step predictor-corrector initial guess "
                f"(P0 frozen-transport with {int(max(1, int(getattr(args, 'pc_p0_outer_it', 2))))} sweep(s), "
                f"P1 alpha-released predictor with max {int(max(1, int(getattr(args, 'pc_p1_max_it', 10))))} Newton steps, "
                f"P2 full-field continuation with max {int(max(1, int(getattr(args, 'pc_p2_max_it', 8))))} Newton steps)."
            )
            pc_stats = _run_predictor_corrector_startup_guess(
                funcs=funcs,
                prev_funcs=prev_funcs,
                aux_funcs=aux_funcs,
                bcs_now=bcs_now,
                step_no=int(step_no),
                t_fail=float(t_now),
                dt_fail=float(dt_now),
            )
            print(
                "    [startup] predictor-corrector initial guess completed "
                f"(|R_raw|_∞ {float(pc_stats['initial']['raw_inf']):.3e} -> {float(pc_stats['final']['raw_inf']):.3e}, "
                f"alpha_mass_rel={float(pc_stats['final']['alpha_mass_rel']):.3e})."
            )
        else:
            print(
                "    [startup] building first-step staggered initial guess "
                f"({int(max(1, int(args.startup_staggered_outer_it)))} outer sweep(s)) before the monolithic solve."
            )
            startup_stats = _run_startup_staggered_guess(
                funcs=funcs,
                prev_funcs=prev_funcs,
                aux_funcs=aux_funcs,
                bcs_now=bcs_now,
                step_no=int(step_no),
                t_fail=float(t_now),
                dt_fail=float(dt_now),
            )
            if bool(getattr(args, "stall_frozen_transport_restart", True)) and bool(args.enable_phi_evolution):
                staged_snapshot = _snapshot_function_values(funcs)
                try:
                    recovery_stats = _run_step1_mechanics_recovery(
                        funcs=funcs,
                        prev_funcs=prev_funcs,
                        aux_funcs=aux_funcs,
                        bcs_now=bcs_now,
                        step_no=int(step_no),
                        t_fail=float(t_now),
                        dt_fail=float(dt_now),
                        reason="staggered startup initial guess",
                    )
                    if recovery_stats is not None:
                        restart_stats = dict((recovery_stats.get("last_stats", {}) or {}))
                        before_norm = float(
                            restart_stats.get("residual_before", float("nan"))
                        )
                        after_norm = float(recovery_stats.get("residual_after", float("nan")))
                        restart_stats["residual_after"] = float(after_norm)
                        improved = np.isfinite(after_norm)
                    else:
                        restart_stats = _run_frozen_transport_restart(
                            funcs=funcs,
                            prev_funcs=prev_funcs,
                            aux_funcs=aux_funcs,
                            bcs_now=bcs_now,
                            step_no=int(step_no),
                            t_fail=float(t_now),
                            dt_fail=float(dt_now),
                            outer_it_override=max(1, int(getattr(args, "stall_frozen_transport_outer_it", 1))),
                        )
                        before_norm = float(restart_stats.get("residual_before", float("nan")))
                        after_norm = float(restart_stats.get("residual_after", float("nan")))
                        min_rel_improve = max(
                            0.0,
                            float(getattr(args, "stall_frozen_transport_min_rel_improve", 0.0) or 0.0),
                        )
                        rel_improve = (
                            ((before_norm - after_norm) / before_norm)
                            if np.isfinite(before_norm) and before_norm > 0.0 and np.isfinite(after_norm)
                            else float("nan")
                        )
                        improved = (
                            np.isfinite(before_norm)
                            and np.isfinite(after_norm)
                            and after_norm < before_norm
                            and (before_norm <= 0.0 or rel_improve >= min_rel_improve)
                        )
                    if improved:
                        print(
                            "    [startup] frozen-transport nonlinear preconditioner improved the staged monolithic residual "
                            f"{_format_frozen_transport_stats(restart_stats)}; keeping the preconditioned staged state."
                        )
                    else:
                        _restore_function_values(funcs, staged_snapshot)
                        print(
                            "    [startup] frozen-transport nonlinear preconditioner did not improve the staged monolithic residual "
                            f"enough {_format_frozen_transport_stats(restart_stats)}; keeping the original staged state."
                        )
                except Exception as restart_exc:
                    _restore_function_values(funcs, staged_snapshot)
                    print(f"    [startup] frozen-transport nonlinear preconditioner failed: {restart_exc}")
        startup_retry_state["startup_guess_applied_step_no"] = int(step_no)
        _arm_startup_monolithic_budget(step_no=int(step_no), reason="staggered startup initial guess")
        reset_bounds_cb = getattr(solver, "_reset_benchmark7_vi_bounds_freeze", None)
        if callable(reset_bounds_cb):
            reset_bounds_cb()
        print(
            (
                "    [startup] predictor-corrector initial guess armed for the exact monolithic corrector."
                if _predictor_corrector_startup_enabled(args, problem)
                else "    [startup] staggered initial guess converged "
                f"{_format_startup_stats(startup_stats)}."
            )
        )
        return None

    solve_error: str = ""
    t_start = time.perf_counter()

    def _post_step_refiner(step, bcs_now, functions, prev_functions) -> None:
        if alpha_from_refmap_enabled:
            _sync_alpha_from_refmap()

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
                on_step_failure=_on_step_failure,
                step_initial_guess_callback=_step_initial_guess_callback,
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
                predictor_clip_01=False,
            ),
            post_step_refiner=_post_step_refiner,
        )
    except Exception as exc:
        solve_error = str(exc)
        print(f"[warn] solve terminated early: {solve_error}", flush=True)
        for f, f_prev in zip(current_functions, previous_functions):
            f.nodal_values[:] = f_prev.nodal_values[:]
    solve_seconds = time.perf_counter() - t_start
    final_raw_residual_inf = _current_monolithic_raw_residual_inf(
        funcs=current_functions,
        prev_funcs=previous_functions,
        aux_funcs=aux_solver_functions,
        bcs_now=bcs,
    )
    final_residual_source = "assembled_raw_residual"
    if not np.isfinite(final_raw_residual_inf):
        fallback_norm = float(getattr(solver, "_last_nonlinear_norm", float("nan")) or float("nan"))
        if np.isfinite(fallback_norm):
            final_raw_residual_inf = float(fallback_norm)
            final_residual_source = str(
                getattr(solver, "_last_nonlinear_norm_label", "solver_last_nonlinear_norm") or "solver_last_nonlinear_norm"
            )
    alpha_diag_final = alpha_diagnostics.evaluate(alpha_coeffs)
    alpha_area_final = float(alpha_diag_final.get("alpha_area", float("nan")))
    alpha_band_final = float(alpha_diag_final.get("alpha_band", float("nan")))

    profile_x, profile_uy = _sample_profile(
        problem=problem,
        Lx=float(args.Lx),
        y_profile=float(args.y_profile),
        n_samples=int(args.profile_samples),
    )
    if io_root:
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
    if io_root:
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
        "latent_bounded_transport": float(1.0 if bool(args.latent_bounded_transport) else 0.0),
        "latent_bounded_fields": str(args.latent_bounded_fields),
        "latent_bounded_eps": float(args.latent_bounded_eps),
        "logistic_bounded_transform": float(1.0 if bool(args.logistic_bounded_transform) else 0.0),
        "logistic_bounded_fields": str(args.logistic_bounded_fields),
        "logistic_bounded_eps": float(args.logistic_bounded_eps),
        "alpha_from_refmap": float(1.0 if bool(getattr(args, "alpha_from_refmap", False)) else 0.0),
        "alpha_mass_constraint": float(1.0 if bool(getattr(args, "alpha_mass_constraint", False)) else 0.0),
        "alpha_box_constraints": float(1.0 if bool(args.alpha_box_constraints) else 0.0),
        "phi_box_constraints": float(1.0 if bool(args.enable_phi_evolution and args.phi_box_constraints) else 0.0),
        "vi_c": float(args.vi_c),
        "vi_c_effective": float(vi_c_effective),
        "vi_c_fields": json.dumps(vi_c_field_current, sort_keys=True),
        "vi_unconstrained_lm": float(1.0 if bool(args.vi_unconstrained_lm) else 0.0),
        "poly_order": float(poly_order),
        "pressure_order": float(pressure_order),
        "scalar_order": float(scalar_order),
        "fluid_space": str(args.fluid_space),
        "fluid_hdiv_order": float(args.fluid_hdiv_order),
        "solid_model": str(args.solid_model),
        "mu_b_model": str(args.mu_b_model),
        "kappa_inv_model": str(args.kappa_inv_model),
        "drag_formulation": str(args.drag_formulation),
        "mechanics_nondim_mode": str(args.mechanics_nondim_mode),
        "gamma_div_input": float(args.gamma_div),
        "gamma_div_effective": float(effective_gamma_div),
        "condition_balanced_auto_gamma_div": float(
            1.0 if bool(getattr(args, "condition_balanced_auto_gamma_div", True)) else 0.0
        ),
        "condition_balanced_solid_cut_fix": float(
            1.0 if bool(getattr(args, "condition_balanced_solid_cut_fix", True)) else 0.0
        ),
        "solid_dof_y_cut": ("none" if solid_dof_y_cut is None else float(solid_dof_y_cut)),
        "inactive_solid_mode": str(problem.get("_inactive_solid_alpha_phase", "none")),
        "inactive_solid_dofs_above_cut": float(sum(int(v) for v in solid_inactive_counts.values())),
        "inactive_solid_dofs_outside_interface_band": 0.0,
        "inactive_solid_dof_counts": json.dumps(solid_inactive_counts, sort_keys=True),
        "inactive_solid_alpha_phase": str(problem.get("_inactive_solid_alpha_phase", "none")),
        "inactive_solid_alpha_band_halfwidth": float(problem.get("_inactive_solid_alpha_band_halfwidth", float("nan"))),
        "fluid_convection": str(args.fluid_convection),
        "phi_evolution": float(1.0 if args.enable_phi_evolution else 0.0),
        "skeleton_acceleration": float(1.0 if args.include_skeleton_acceleration else 0.0),
        "rho_s0_tilde": float(args.rho_s0_tilde),
        "skeleton_inertia_convection": str(args.skeleton_inertia_convection),
        "startup_bootstrap": float(1.0 if bool(args.startup_bootstrap) else 0.0),
        "startup_bootstrap_fluid_convection": str(args.startup_bootstrap_fluid_convection),
        "startup_bootstrap_skeleton_acceleration": float(
            1.0 if bool(args.startup_bootstrap_include_skeleton_acceleration) else 0.0
        ),
        "startup_bootstrap_max_it": float(args.startup_bootstrap_max_it),
        "startup_staggered_outer_it": float(args.startup_staggered_outer_it),
        "alpha_regularization": str(args.alpha_regularization),
        "alpha_reg_gamma": float(args.alpha_reg_gamma),
        "alpha_reg_eps_normal": float(alpha_reg_eps_normal),
        "alpha_reg_eps_tangent": float(alpha_reg_eps_tangent),
        "alpha_reg_eta": float(args.alpha_reg_eta),
        "alpha_advect_with": str(args.alpha_advect_with),
        "alpha_advection_form": str(args.alpha_advection_form),
        "alpha_supg": float(args.alpha_supg),
        "alpha_cip": float(args.alpha_cip),
        "support_physics": str(args.support_physics),
        "solid_bc_mode": str(args.solid_bc_mode),
        "skeleton_pressure_mode": str(args.skeleton_pressure_mode),
        "alpha_biot": (float(args.alpha_biot) if args.alpha_biot is not None else "none"),
        "top_drainage_transport": float(
            1.0
            if _full_top_drainage_transport_enabled(
                enable_phi_evolution=bool(args.enable_phi_evolution),
                top_drainage_transport=bool(args.top_drainage_transport),
            )
            else 0.0
        ),
        "alpha_bc_mode": ("natural" if str(args.alpha_bc_mode).strip().lower() == "auto" else str(args.alpha_bc_mode)),
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
        "final_raw_residual_inf": float(final_raw_residual_inf),
        "final_residual_source": str(final_residual_source),
        "alpha_area0": float(alpha_area0),
        "alpha_area_final": float(alpha_area_final),
        "alpha_area_rel_drift": float((alpha_area_final - alpha_area0) / max(abs(alpha_area0), 1.0e-30)),
        "alpha_band0": float(alpha_band0),
        "alpha_band_final": float(alpha_band_final),
        "alpha_band_rel_drift": float((alpha_band_final - alpha_band0) / max(abs(alpha_band0), 1.0e-30)),
        "alpha_min": float(np.min(problem["alpha_k"].nodal_values)),
        "alpha_max": float(np.max(problem["alpha_k"].nodal_values)),
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
    if io_root:
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
    barrier(MPI_CTX)
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
    args = _normalize_benchmark7_solver_choice(args)
    prev_cpp_fuse = _configure_benchmark7_cpp_fuse_integrals(
        backend=str(args.backend),
        enabled=getattr(args, "cpp_fuse_integrals", None),
    )
    if (
        str(getattr(args, "backend", "")).strip().lower() in {"cpp", "c++"}
        and MPI_CTX.is_root
    ):
        cpp_fuse_value = os.environ.get("PYCUTFEM_CPP_FUSE_INTEGRALS", prev_cpp_fuse)
        if cpp_fuse_value is not None:
            print(
                f"[setup] Benchmark 7 cpp integral fusion: PYCUTFEM_CPP_FUSE_INTEGRALS={cpp_fuse_value}",
                flush=True,
            )
    if (
        MPI_CTX.enabled
        and bool(getattr(args, "petsc_distributed", True))
        and str(getattr(args, "linear_backend", "")).strip().lower() == "petsc"
        and MPI_CTX.is_root
    ):
        print(
            f"[mpi] COMM_WORLD size={MPI_CTX.size}; enabling collective PETSc linear solves for Benchmark 7.",
            flush=True,
        )
    if bool(getattr(args, "newton_pressure_schur_solve", False)) and bool(getattr(args, "solid_volumetric_split", False)):
        print(
            "[solver] disabling solid_volumetric_split for the reduced Newton pressure-Schur solve, "
            "because the current pi_s branch destroys the useful pressure block.",
            flush=True,
        )
        args.solid_volumetric_split = False
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
    if _mpi_io_root():
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
    barrier(MPI_CTX)


if __name__ == "__main__":
    main()
