#!/usr/bin/env python
# coding: utf-8
"""
Monolithic CutFEM setup for the Turek–Hron FSI-2/FSI-3 benchmarks.

- Geometry: channel with a *rigid* circular hole; the elastic beam is described
  by a level-set that is advected with the solid displacement.
- Mechanics: solid stress in a fully Eulerian frame (Cauchy stress, advective transport),
  optionally linearized for robustness (set USE_LINEAR_SOLID=0 for full nonlinear StVK),
  incompressible Navier–Stokes for the fluid.
- The beam level set is updated every time step and the mesh is reclassified so
  curved deformations of the beam are captured.
- A small finite-difference Jacobian check is run to validate the assembled
  Jacobian against the residual.

Example:
  PENALTY_VAL=1e-3 PENALTY_GRAD=1e-3 PYCUTFEM_JIT_BACKEND=cpp FD_BACKEND=python \
    conda run --no-capture-output -n fenicsx python -u examples/turek_fsi_fully_eulerian/turek_fsi_fully_eulerian.py \
    --no-refine-initial
  PENALTY_VAL=1e-3 PENALTY_GRAD=1e-3 PYCUTFEM_JIT_BACKEND=cpp \
    ALLOW_DT_REDUCTION=1 REFINE_INITIAL=0 LEVELSET_UPDATE_TOL=1e-8 \
    conda run --no-capture-output -n fenicsx python -u examples/turek_fsi_fully_eulerian/turek_fsi_fully_eulerian.py \
    --run-fd-check --run-fd-terms --fd-term stab --fd-skip-full --no-run-time-stepping

Long-run validation (FSI-2, `dFacetPatch`, cpp backend):
  PYCUTFEM_JIT_BACKEND=cpp PYCUTFEM_UNIFIED_PRECOMPUTE=1 USE_FACET_PATCH_GHOST=1 \
    PENALTY_VAL=1e-3 PENALTY_GRAD=1e-3 BETA_PENALTY=20 \
    python -u examples/turek_fsi_fully_eulerian/turek_fsi_fully_eulerian.py \
      --turek-case fsi2 --mesh-backend structured --mesh-size 0.05 \
      --dt 0.005 --final-time 3.0 \
      --newton-tol 1e-6 --newton-rtol 0 --max-newton-iter 10 \
      --obs-every 20 --no-save-vtk --no-run-fd-check --no-run-fd-terms
  For nonlinear StVK solid, add: `USE_LINEAR_SOLID=0` (more expensive).


"""
from __future__ import annotations

import math
import os
import sys
import argparse
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from functools import lru_cache
from typing import Callable, Dict, Iterable, Sequence

try:
    import gmsh  # type: ignore
except Exception:
    gmsh = None

import numba
import numpy as np

from pycutfem.core.geometry import hansbo_cut_ratio
from pycutfem.core.levelset import BeamLevelSet, LevelSetGridFunction
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.bitset import BitSet
from pycutfem.utils.fsi_fully_eulerian import (
    nudge_levelset_zeros as _nudge_levelset_zeros,
    refresh_sliver_weights,
)
from pycutfem.utils.gmsh_loader import mesh_from_gmsh
from pycutfem.utils.ogrid_meshgen import circular_hole_ogrid
from pycutfem.utils.refinement import TensorRefiner
from pycutfem.fem.reference import get_reference

from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    FacetNormal,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    Pos,
    Neg,
    ElementWiseConstant,
    restrict,
    grad,
    inner,
    dot,
    div,
    jump,
    Identity,
    Hessian,
    det,
    inv,
    trace,
    Jump,
)
from pycutfem.ufl.measures import dx, ds, dGhost, dInterface, dFacetPatch
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.io.vtk import export_vtk
from pycutfem.io.visualization import plot_mesh_2
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.helpers import analyze_active_dofs
from pycutfem.core.topology import Node
from pycutfem.fem import transform
from pycutfem.solvers.nonlinear_solver import (
    NewtonParameters,
    NewtonSolver,
    PetscSnesNewtonSolver,
    LinearSolverParameters,
    TimeStepperParameters,
    _ActiveReducer,
)

# -----------------------------------------------------------------------------
# Numba configuration
# -----------------------------------------------------------------------------
try:
    num_cores = os.cpu_count()
    numba.set_num_threads(num_cores)
    print(f"Numba threads: {numba.get_num_threads()}")
except Exception:
    print("Numba not configured; continuing without thread pinning.")

_t0_global = time.perf_counter()
def _log_step(msg: str) -> None:
    t = time.perf_counter() - _t0_global
    print(f"[t={t:7.3f}s] {msg}")

# -----------------------------------------------------------------------------
# CLI / environment options
# -----------------------------------------------------------------------------
def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw not in ("0", "false", "False", "no", "No")


def _env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return float(raw)


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return int(raw)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Monolithic CutFEM Turek–Hron FSI-2/FSI-3 (fully Eulerian solid)."
    )
    turek_case_default = os.getenv("TUREK_CASE", "fsi2").strip().lower()
    if not turek_case_default:
        turek_case_default = "fsi2"
    parser.add_argument(
        "--turek-case",
        choices=("fsi1", "fsi2", "fsi3"),
        default=turek_case_default,
        help="Turek–Hron preset: fsi1=steady, fsi2=periodic, fsi3=chaotic. Env: TUREK_CASE.",
    )
    parser.add_argument("--u-mean", type=float, default=None, help="Mean inflow velocity U_mean. Env: U_MEAN.")
    parser.add_argument("--rho-s", type=float, default=None, help="Solid density rho_s. Env: RHO_S.")
    parser.add_argument("--dt", type=float, default=None, help="Time step size. Env: DT.")
    parser.add_argument(
        "--theta",
        type=float,
        default=None,
        help="Theta scheme parameter (1=BE, 0.5=CN). Env: THETA.",
    )
    parser.add_argument(
        "--poly-order",
        type=int,
        default=int(os.getenv("POLY_ORDER", "2")),
        help="Polynomial order for primary fields. Env: POLY_ORDER.",
    )
    parser.add_argument(
        "--taylor-hood",
        dest="taylor_hood",
        action="store_true",
        help="Use Taylor–Hood pressure (p=k-1). Env: PYCUTFEM_TAYLOR_HOOD.",
    )
    parser.add_argument(
        "--no-taylor-hood",
        dest="taylor_hood",
        action="store_false",
        help="Use equal-order pressure (p=k). Env: PYCUTFEM_TAYLOR_HOOD.",
    )
    parser.add_argument(
        "--mesh-size",
        type=float,
        default=float(os.getenv("MESH_SIZE", "0.025")),
        help="Target mesh size for structured O-grid. Env: MESH_SIZE.",
    )
    parser.add_argument(
        "--mesh-diagnostics-enabled",
        type=bool,
        default=_env_bool("MESH_DIAGNOSTICS_ENABLED", False),
        help="Enable mesh diagnostics output. Env: MESH_DIAGNOSTICS_ENABLED.",
    ) 
    mesh_backend_default = os.getenv("MESH_BACKEND", "gmsh").strip().lower()
    if not mesh_backend_default:
        mesh_backend_default = "gmsh"
    parser.add_argument(
        "--mesh-backend",
        choices=("gmsh", "structured"),
        default=mesh_backend_default,
        help="Mesh generator: gmsh or structured. Env: MESH_BACKEND.",
    )
    parser.add_argument("--mesh-file", type=Path, default=None, help="Optional path to reuse/store the gmsh .msh file.")
    parser.add_argument("--rebuild-mesh", action="store_true", help="Force rebuilding the gmsh mesh instead of reusing an existing file.")
    parser.add_argument("--view-gmsh", action="store_true", help="Preview the gmsh model before meshing.")
    parser.add_argument(
        "--no-refine-initial",
        dest="refine_initial",
        action="store_true",
        help="Skip anisotropic refinement around the beam in the initial mesh. Env: REFINE_INITIAL.",
    )
    parser.add_argument(
        "--refine-initial",
        dest="refine_initial",
        action="store_false",
        help="Apply anisotropic refinement around the beam in the initial mesh. Env: REFINE_INITIAL.",
    )
    fd_backend_default = os.getenv("FD_BACKEND", "jit").strip().lower()
    if not fd_backend_default:
        fd_backend_default = "jit"
    parser.add_argument(
        "--fd-backend",
        type=str,
        default=fd_backend_default,
        choices=["jit", "python"],
        help="Backend for FD Jacobian checks. Env: FD_BACKEND.",
    )
    jit_backend_env = os.getenv("PYCUTFEM_JIT_BACKEND", "").strip().lower()
    jit_backend_default = "cpp" if jit_backend_env in {"cpp", "c++"} else "numba"
    parser.add_argument(
        "--jit-backend",
        choices=("numba", "cpp"),
        default=jit_backend_default,
        help="JIT backend selection. Env: PYCUTFEM_JIT_BACKEND.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getenv("OUTPUT_DIR", "turek_results_fsi_ii_eulerian"),
        help="Directory for VTK output. Env: OUTPUT_DIR.",
    )
    parser.add_argument("--save-vtk", dest="save_vtk", action="store_true", help="Enable VTK output. Env: SAVE_VTK.")
    parser.add_argument("--no-save-vtk", dest="save_vtk", action="store_false", help="Disable VTK output. Env: SAVE_VTK.")
    parser.add_argument(
        "--compute-observables",
        dest="compute_observables",
        action="store_true",
        help="Compute drag/lift/Δp each step (python backend). Env: COMPUTE_OBSERVABLES.",
    )
    parser.add_argument(
        "--no-compute-observables",
        dest="compute_observables",
        action="store_false",
        help="Disable observables computation. Env: COMPUTE_OBSERVABLES.",
    )
    parser.add_argument(
        "--obs-every",
        type=int,
        default=int(os.getenv("OBS_EVERY", "1")),
        help="Compute observables every N steps when enabled (0 disables). Env: OBS_EVERY.",
    )
    parser.add_argument(
        "--vtk-interface-clamp",
        type=float,
        default=float(os.getenv("VTK_INTERFACE_CLAMP", "0.0")),
        help="Clamp |value| at interface nodes in VTK output (0 disables). Env: VTK_INTERFACE_CLAMP.",
    )
    parser.add_argument("--run-fd-check", dest="run_fd_check", action="store_true", help="Run finite-difference Jacobian check. Env: RUN_FD_CHECK.")
    parser.add_argument("--no-run-fd-check", dest="run_fd_check", action="store_false", help="Skip finite-difference check. Env: RUN_FD_CHECK.")
    parser.add_argument("--run-fd-terms", dest="run_fd_terms", action="store_true", help="Run per-term FD checks. Env: RUN_FD_TERMS.")
    parser.add_argument("--no-run-fd-terms", dest="run_fd_terms", action="store_false", help="Skip per-term FD checks. Env: RUN_FD_TERMS.")
    parser.add_argument("--run-time-stepping", dest="run_time_stepping", action="store_true", help="Run transient solve. Env: RUN_TIME_STEPPING.")
    parser.add_argument("--no-run-time-stepping", dest="run_time_stepping", action="store_false", help="Skip transient solve. Env: RUN_TIME_STEPPING.")
    parser.add_argument("--plot-mesh", dest="plot_mesh", action="store_true", help="Save mesh/ghost/level-set plot each step. Env: PLOT_MESH.")
    parser.add_argument("--no-plot-mesh", dest="plot_mesh", action="store_false", help="Disable mesh plotting. Env: PLOT_MESH.")
    parser.add_argument(
        "--plot-mesh-every",
        type=int,
        default=int(os.getenv("PLOT_MESH_EVERY", "1")),
        help="Plot mesh every N steps when enabled. Env: PLOT_MESH_EVERY.",
    )
    parser.add_argument("--interactive-plot", dest="interactive_plot", action="store_true", help="Show interactive mesh plot. Env: INTERACTIVE_PLOT.")
    parser.add_argument("--no-interactive-plot", dest="interactive_plot", action="store_false", help="Disable interactive mesh plot. Env: INTERACTIVE_PLOT.")
    parser.add_argument("--plot-levelset", dest="plot_levelset", action="store_true", help="Overlay level-set zero contour. Env: PLOT_LEVELSET.")
    parser.add_argument("--no-plot-levelset", dest="plot_levelset", action="store_false", help="Hide level-set in mesh plot. Env: PLOT_LEVELSET.")
    parser.add_argument("--plot-interface-points", dest="plot_interface_points", action="store_true", help="Overlay interface points/segments. Env: PLOT_INTERFACE_POINTS.")
    parser.add_argument("--no-plot-interface-points", dest="plot_interface_points", action="store_false", help="Hide interface points. Env: PLOT_INTERFACE_POINTS.")
    parser.add_argument("--plot-show", dest="plot_show", action="store_true", help="Call plt.show() for mesh plots. Env: PLOT_SHOW.")
    parser.add_argument("--no-plot-show", dest="plot_show", action="store_false", help="Do not show mesh plots. Env: PLOT_SHOW.")
    parser.add_argument(
        "--plot-resolution",
        type=int,
        default=int(os.getenv("PLOT_RESOLUTION", "120")),
        help="Grid resolution for level-set contour. Env: PLOT_RESOLUTION.",
    )
    parser.add_argument("--plot-only", dest="plot_only", action="store_true", help="Stop after plotting the initial mesh. Env: PLOT_ONLY.")
    parser.add_argument("--no-plot-only", dest="plot_only", action="store_false", help="Run full setup after plotting. Env: PLOT_ONLY.")
    parser.add_argument("--force-full-setup", dest="force_full_setup", action="store_true", help="Build full solver even when plotting. Env: FORCE_FULL_SETUP.")
    parser.add_argument("--no-force-full-setup", dest="force_full_setup", action="store_false", help="Allow early exit when plotting only. Env: FORCE_FULL_SETUP.")
    parser.add_argument("--beam-shift-x", type=float, default=_env_float("BEAM_SHIFT_X") or 0.0, help="Shift beam root in x. Env: BEAM_SHIFT_X.")
    parser.add_argument("--beam-root-inset", type=float, default=None, help="Inset for curved beam root. Env: BEAM_ROOT_INSET.")
    parser.add_argument("--beam-root-dof-tol", type=float, default=None, help="Beam root DOF locator tolerance. Env: BEAM_ROOT_DOF_TOL.")
    parser.add_argument("--pin-pressure", dest="pin_pressure", action="store_true", help="Pin one pressure DOF. Env: PIN_PRESSURE.")
    parser.add_argument("--no-pin-pressure", dest="pin_pressure", action="store_false", help="Disable pressure pinning. Env: PIN_PRESSURE.")
    parser.add_argument(
        "--solid-cut-drop",
        type=float,
        default=float(os.getenv("SOLID_CUT_DROP", "1e-3")),
        help="Drop solid DOFs for tiny cuts. Env: SOLID_CUT_DROP.",
    )
    parser.add_argument("--use-aligned-interface", dest="use_aligned_interface", action="store_true", help="Keep aligned interface edges. Env: USE_ALIGNED_INTERFACE.")
    parser.add_argument("--no-use-aligned-interface", dest="use_aligned_interface", action="store_false", help="Disable aligned interface edges. Env: USE_ALIGNED_INTERFACE.")
    parser.add_argument("--solid-advect-lagged", dest="solid_advect_lagged", action="store_true", help="Lag solid advection velocity. Env: SOLID_ADVECT_LAGGED.")
    parser.add_argument("--no-solid-advect-lagged", dest="solid_advect_lagged", action="store_false", help="Use current solid velocity in advection. Env: SOLID_ADVECT_LAGGED.")
    parser.add_argument("--use-restricted-forms", dest="use_restricted_forms", action="store_true", help="Restrict forms to active subdomains. Env: USE_RESTRICTED_FORMS.")
    parser.add_argument("--no-use-restricted-forms", dest="use_restricted_forms", action="store_false", help="Disable restricted forms. Env: USE_RESTRICTED_FORMS.")
    parser.add_argument("--use-linear-solid", dest="use_linear_solid", action="store_true", help="Use linearized solid stress. Env: USE_LINEAR_SOLID.")
    parser.add_argument("--no-use-linear-solid", dest="use_linear_solid", action="store_false", help="Use nonlinear StVK solid. Env: USE_LINEAR_SOLID.")
    parser.add_argument(
        "--levelset-zero-eps",
        type=float,
        default=None,
        help="Interface edge tolerance (zero band). Env: LEVELSET_ZERO_EPS.",
    )
    parser.add_argument(
        "--levelset-nudge-eps",
        type=float,
        default=None,
        help="Nudge epsilon for level-set nodes. Env: LEVELSET_NUDGE_EPS.",
    )
    levelset_side_default = os.getenv("LEVELSET_ZERO_SIDE", "neg").strip().lower()
    if not levelset_side_default:
        levelset_side_default = "neg"
    parser.add_argument("--levelset-zero-side", choices=("neg", "pos"), default=levelset_side_default, help="Preferred level-set sign. Env: LEVELSET_ZERO_SIDE.")
    parser.add_argument(
        "--levelset-update-tol",
        type=float,
        default=None,
        help="Skip level-set refresh if max |Δφ| <= tol. Env: LEVELSET_UPDATE_TOL.",
    )
    parser.add_argument("--fd-timing", dest="fd_timing", action="store_true", help="Print FD timing diagnostics. Env: FD_TIMING.")
    parser.add_argument("--no-fd-timing", dest="fd_timing", action="store_false", help="Disable FD timing diagnostics. Env: FD_TIMING.")
    parser.add_argument("--fd-jac-cols-only", dest="fd_jac_cols_only", action="store_true", help="Only build FD Jacobian columns. Env: FD_JAC_COLS_ONLY.")
    parser.add_argument("--no-fd-jac-cols-only", dest="fd_jac_cols_only", action="store_false", help="Build full FD Jacobian. Env: FD_JAC_COLS_ONLY.")
    fd_debug_top_default = _env_int("FD_DEBUG_TOP")
    parser.add_argument("--fd-debug-col", type=str, default=os.getenv("FD_DEBUG_COL", ""), help="Debug FD column index. Env: FD_DEBUG_COL.")
    parser.add_argument(
        "--fd-debug-top",
        type=int,
        default=fd_debug_top_default if fd_debug_top_default is not None else 5,
        help="Number of largest FD errors to report. Env: FD_DEBUG_TOP.",
    )
    s_nitsche_env = os.getenv("S_NITSCHE_VALUE")
    if s_nitsche_env is None or not s_nitsche_env.strip():
        s_nitsche_env = os.getenv("S_NITSCHE")
    if s_nitsche_env is None or not s_nitsche_env.strip():
        # Default to *incomplete* Nitsche for robustness in the fully Eulerian cut setting.
        # Symmetric Nitsche (s=+1) can be re-enabled via S_NITSCHE_VALUE=1.
        s_nitsche_env = "0.0"
    parser.add_argument("--penalty-val", type=float, default=float(os.getenv("PENALTY_VAL", "0.0")), help="Stabilization penalty value. Env: PENALTY_VAL.")
    parser.add_argument("--penalty-grad", type=float, default=float(os.getenv("PENALTY_GRAD", "0.0")), help="Stabilization penalty gradient. Env: PENALTY_GRAD.")
    parser.add_argument("--solid-reg-eps", type=float, default=float(os.getenv("SOLID_REG_EPS", "1e-6")), help="Solid regularization weight. Env: SOLID_REG_EPS.")
    parser.add_argument("--fd-reuse-kernels", dest="fd_reuse_kernels", action="store_true", help="Reuse FD JIT kernels. Env: FD_REUSE_KERNELS.")
    parser.add_argument("--no-fd-reuse-kernels", dest="fd_reuse_kernels", action="store_false", help="Do not reuse FD kernels. Env: FD_REUSE_KERNELS.")
    parser.add_argument("--fd-skip-full", dest="fd_skip_full", action="store_true", help="Skip full-form FD check. Env: FD_SKIP_FULL.")
    parser.add_argument("--no-fd-skip-full", dest="fd_skip_full", action="store_false", help="Run full-form FD check. Env: FD_SKIP_FULL.")
    parser.add_argument("--fd-term-split", dest="fd_term_split", action="store_true", help="Split FD checks by term. Env: FD_TERM_SPLIT.")
    parser.add_argument("--no-fd-term-split", dest="fd_term_split", action="store_false", help="Disable term-split FD checks. Env: FD_TERM_SPLIT.")
    parser.add_argument("--fd-interface-split", dest="fd_interface_split", action="store_true", help="Split interface FD terms. Env: FD_INTERFACE_SPLIT.")
    parser.add_argument("--no-fd-interface-split", dest="fd_interface_split", action="store_false", help="Disable interface FD split. Env: FD_INTERFACE_SPLIT.")
    parser.add_argument("--fd-term", type=str, default=os.getenv("FD_TERM", ""), help="Filter FD term name. Env: FD_TERM.")
    max_steps_default = _env_int("MAX_STEPS")
    parser.add_argument("--max-steps", type=int, default=max_steps_default if max_steps_default is not None else 50, help="Maximum time steps. Env: MAX_STEPS.")
    final_time_default = _env_float("FINAL_TIME")
    parser.add_argument(
        "--final-time",
        type=float,
        default=final_time_default,
        help="Stop when physical time reaches this value (overrides max_steps-derived default). Env: FINAL_TIME.",
    )

    restart_dir_default = os.getenv("RESTART_DIR", "").strip()
    parser.add_argument(
        "--restart-dir",
        type=Path,
        default=Path(restart_dir_default) if restart_dir_default else None,
        help="Base directory containing levelset_dumps/ and state_dumps/ for restart. Env: RESTART_DIR.",
    )
    parser.add_argument(
        "--restart-step",
        type=int,
        default=int(os.getenv("RESTART_STEP", "-1") or "-1"),
        help="Dump index to restart from (0-based). Env: RESTART_STEP.",
    )
    restart_tag_default = os.getenv("RESTART_TAG", "step").strip().lower() or "step"
    parser.add_argument(
        "--restart-tag",
        choices=("step", "fail"),
        default=restart_tag_default,
        help="Which dump tag to load: step (accepted) or fail (Newton failure). Env: RESTART_TAG.",
    )
    parser.add_argument(
        "--restart-reset-counters",
        action="store_true",
        default=_env_bool("RESTART_RESET_COUNTERS", False),
        help=(
            "Reset step counters for logs/dumps after restart (physical time t is still taken from the checkpoint). "
            "Env: RESTART_RESET_COUNTERS."
        ),
    )
    parser.add_argument("--allow-dt-reduction", dest="allow_dt_reduction", action="store_true", help="Allow adaptive dt reduction. Env: ALLOW_DT_REDUCTION.")
    parser.add_argument("--no-allow-dt-reduction", dest="allow_dt_reduction", action="store_false", help="Disable adaptive dt reduction. Env: ALLOW_DT_REDUCTION.")
    parser.add_argument(
        "--dt-reduction-factor",
        type=float,
        default=float(os.getenv("DT_REDUCTION_FACTOR", "0.5")),
        help="Factor for dt reduction. Env: DT_REDUCTION_FACTOR.",
    )
    parser.add_argument(
        "--dt-reduction-threshold",
        type=float,
        default=float(os.getenv("DT_REDUCTION_THRESHOLD", "5.0")),
        help="ΔU growth threshold for dt reduction. Env: DT_REDUCTION_THRESHOLD.",
    )
    parser.add_argument(
        "--newton-tol",
        type=float,
        default=float(os.getenv("NEWTON_TOL", "1e-8")),
        help="Newton residual tolerance. Env: NEWTON_TOL.",
    )
    parser.add_argument(
        "--newton-rtol",
        type=float,
        default=float(os.getenv("NEWTON_RTOL", "1e-8")),
        help="Newton relative residual tolerance (‖R‖∞ ≤ rtol·‖R0‖∞). Env: NEWTON_RTOL.",
    )
    parser.add_argument("--max-newton-iter", type=int, default=int(os.getenv("MAX_NEWTON_ITER", "50")), help="Maximum Newton iterations. Env: MAX_NEWTON_ITER.")
    nonlinear_solver_default = os.getenv("NONLINEAR_SOLVER", "newton").strip().lower()
    if not nonlinear_solver_default:
        nonlinear_solver_default = "newton"
    parser.add_argument(
        "--nonlinear-solver",
        choices=("newton", "snes"),
        default=nonlinear_solver_default,
        help="Nonlinear solver: Python Newton or PETSc SNES. Env: NONLINEAR_SOLVER.",
    )
    linear_solver_default = os.getenv("LINEAR_SOLVER", "scipy").strip().lower()
    if not linear_solver_default:
        linear_solver_default = "scipy"
    parser.add_argument(
        "--linear-solver",
        choices=("scipy", "petsc"),
        default=linear_solver_default,
        help="Linear solver backend for --nonlinear-solver=newton. Env: LINEAR_SOLVER.",
    )
    ls_mode_default = os.getenv("LS_MODE", "armijo").strip().lower()
    if not ls_mode_default:
        ls_mode_default = "armijo"
    parser.add_argument("--ls-mode", type=str, default=ls_mode_default, choices=("armijo", "dealii"), help="Line-search mode. Env: LS_MODE.")
    parser.add_argument("--ls-max-iter", type=int, default=int(os.getenv("LS_MAX_ITER", "12")), help="Line-search max iterations. Env: LS_MAX_ITER.")
    parser.add_argument(
        "--s-nitsche",
        "--s-nitsche-value",
        "--s-intsche",
        dest="s_nitsche_value",
        type=float,
        default=float(s_nitsche_env),
        help=(
            "Nitsche value (1 = symmetric, 0 = incomplete, -1 = skew-symmetric). "
            "Env: S_NITSCHE_VALUE or S_NITSCHE."
        ),
    )
    parser.set_defaults(
        save_vtk=_env_bool("SAVE_VTK", True),
        compute_observables=_env_bool("COMPUTE_OBSERVABLES", True),
        run_fd_check=_env_bool("RUN_FD_CHECK", False),
        run_fd_terms=_env_bool("RUN_FD_TERMS", False),
        run_time_stepping=_env_bool("RUN_TIME_STEPPING", True),
        plot_mesh=_env_bool("PLOT_MESH", False),
        interactive_plot=_env_bool("INTERACTIVE_PLOT", False),
        plot_levelset=_env_bool("PLOT_LEVELSET", True),
        plot_interface_points=_env_bool("PLOT_INTERFACE_POINTS", True),
        plot_show=_env_bool("PLOT_SHOW", False),
        plot_only=_env_bool("PLOT_ONLY", False),
        force_full_setup=_env_bool("FORCE_FULL_SETUP", False),
        refine_initial=_env_bool("REFINE_INITIAL", True),
        taylor_hood=_env_bool("PYCUTFEM_TAYLOR_HOOD", False),
        pin_pressure=_env_bool("PIN_PRESSURE", True),
        use_aligned_interface=_env_bool("USE_ALIGNED_INTERFACE", True),
        solid_advect_lagged=_env_bool("SOLID_ADVECT_LAGGED", True),
        use_restricted_forms=_env_bool("USE_RESTRICTED_FORMS", True),
        use_linear_solid=_env_bool("USE_LINEAR_SOLID", True),
        fd_timing=_env_bool("FD_TIMING", False),
        fd_jac_cols_only=_env_bool("FD_JAC_COLS_ONLY", True),
        fd_reuse_kernels=_env_bool("FD_REUSE_KERNELS", True),
        fd_skip_full=_env_bool("FD_SKIP_FULL", False),
        fd_term_split=_env_bool("FD_TERM_SPLIT", False),
        fd_interface_split=_env_bool("FD_INTERFACE_SPLIT", False),
        allow_dt_reduction=_env_bool("ALLOW_DT_REDUCTION", False),
    )
    args = parser.parse_args()

    case_label_map = {"fsi1": "FSI-1", "fsi2": "FSI-2", "fsi3": "FSI-3"}
    case_defaults = {
        "fsi1": {"u_mean": 0.2, "rho_s": 1.0e3, "dt": 1.0, "theta": 1.0},
        "fsi2": {"u_mean": 1.0, "rho_s": 1.0e4, "dt": 0.005, "theta": 0.5},
        "fsi3": {"u_mean": 2.0, "rho_s": 1.0e4, "dt": 0.005, "theta": 0.5},
    }
    preset = case_defaults.get(str(args.turek_case), case_defaults["fsi2"])
    env_u_mean = _env_float("U_MEAN")
    env_rho_s = _env_float("RHO_S")
    env_dt = _env_float("DT")
    env_theta = _env_float("THETA")
    if args.u_mean is None and env_u_mean is not None:
        args.u_mean = env_u_mean
    if args.rho_s is None and env_rho_s is not None:
        args.rho_s = env_rho_s
    if args.dt is None and env_dt is not None:
        args.dt = env_dt
    if args.theta is None and env_theta is not None:
        args.theta = env_theta
    if args.u_mean is None:
        args.u_mean = float(preset["u_mean"])
    if args.rho_s is None:
        args.rho_s = float(preset["rho_s"])
    if args.dt is None:
        args.dt = float(preset["dt"])
    if args.theta is None:
        args.theta = float(preset["theta"])
    args.case_label = case_label_map.get(str(args.turek_case), str(args.turek_case))
    if not math.isclose(float(args.dt), float(preset["dt"])):
        print(
            f"[warn] dt={float(args.dt):g} differs from {float(preset['dt']):g} for {args.case_label}; "
            "large dt can suppress the expected oscillatory response."
        )
    if not math.isclose(float(args.theta), float(preset["theta"])):
        print(
            f"[warn] theta={float(args.theta):g} differs from {float(preset['theta']):g} for {args.case_label}; "
            "theta>0.5 adds numerical damping."
        )
    if args.beam_root_inset is None:
        env_root_inset = _env_float("BEAM_ROOT_INSET")
        args.beam_root_inset = env_root_inset if env_root_inset is not None else max(5.0e-4, 0.04 * args.mesh_size)
    if args.beam_root_dof_tol is None:
        env_root_tol = _env_float("BEAM_ROOT_DOF_TOL")
        args.beam_root_dof_tol = env_root_tol if env_root_tol is not None else max(0.2 * args.mesh_size, 1.0e-3)
    if args.levelset_zero_eps is None:
        env_zero = _env_float("LEVELSET_ZERO_EPS")
        args.levelset_zero_eps = env_zero if env_zero is not None else max(1.0e-10, 1.0e-8 * args.mesh_size)
    if args.levelset_nudge_eps is None:
        env_nudge = _env_float("LEVELSET_NUDGE_EPS")
        args.levelset_nudge_eps = env_nudge
    if args.levelset_update_tol is None:
        env_update_tol = _env_float("LEVELSET_UPDATE_TOL")
        args.levelset_update_tol = (
            env_update_tol if env_update_tol is not None else max(1.0e-10, 1.0e-8 * args.mesh_size)
        )
    if "--no-refine-initial" in sys.argv:
        args.refine_initial = False
    if "--refine-initial" in sys.argv:
        args.refine_initial = True
    return args

ARGS = _parse_args()


def _apply_env_overrides(args: argparse.Namespace) -> None:
    def _set_env(name: str, value) -> None:
        if value is None:
            return
        os.environ[name] = str(value)

    def _set_env_bool(name: str, value: bool) -> None:
        os.environ[name] = "1" if value else "0"

    _set_env("DT", args.dt)
    _set_env("POLY_ORDER", args.poly_order)
    _set_env_bool("PYCUTFEM_TAYLOR_HOOD", args.taylor_hood)
    _set_env("MESH_SIZE", args.mesh_size)
    _set_env("MESH_BACKEND", args.mesh_backend)
    _set_env("FD_BACKEND", args.fd_backend)
    _set_env("NEWTON_RTOL", args.newton_rtol)
    _set_env("OUTPUT_DIR", args.output_dir)
    _set_env("PLOT_MESH_EVERY", args.plot_mesh_every)
    _set_env("PLOT_RESOLUTION", args.plot_resolution)
    _set_env_bool("SAVE_VTK", args.save_vtk)
    _set_env_bool("COMPUTE_OBSERVABLES", args.compute_observables)
    _set_env("OBS_EVERY", args.obs_every)
    _set_env_bool("RUN_FD_CHECK", args.run_fd_check)
    _set_env_bool("RUN_FD_TERMS", args.run_fd_terms)
    _set_env_bool("RUN_TIME_STEPPING", args.run_time_stepping)
    _set_env_bool("PLOT_MESH", args.plot_mesh)
    _set_env_bool("INTERACTIVE_PLOT", args.interactive_plot)
    _set_env_bool("PLOT_LEVELSET", args.plot_levelset)
    _set_env_bool("PLOT_INTERFACE_POINTS", args.plot_interface_points)
    _set_env_bool("PLOT_SHOW", args.plot_show)
    _set_env_bool("PLOT_ONLY", args.plot_only)
    _set_env_bool("FORCE_FULL_SETUP", args.force_full_setup)
    _set_env_bool("REFINE_INITIAL", args.refine_initial)
    _set_env("BEAM_SHIFT_X", args.beam_shift_x)
    _set_env("BEAM_ROOT_INSET", args.beam_root_inset)
    _set_env("BEAM_ROOT_DOF_TOL", args.beam_root_dof_tol)
    _set_env_bool("PIN_PRESSURE", args.pin_pressure)
    _set_env("SOLID_CUT_DROP", args.solid_cut_drop)
    _set_env_bool("USE_ALIGNED_INTERFACE", args.use_aligned_interface)
    _set_env_bool("SOLID_ADVECT_LAGGED", args.solid_advect_lagged)
    _set_env_bool("USE_RESTRICTED_FORMS", args.use_restricted_forms)
    _set_env_bool("USE_LINEAR_SOLID", args.use_linear_solid)
    _set_env("LEVELSET_ZERO_EPS", args.levelset_zero_eps)
    _set_env("LEVELSET_NUDGE_EPS", args.levelset_nudge_eps)
    _set_env("LEVELSET_ZERO_SIDE", args.levelset_zero_side)
    _set_env("LEVELSET_UPDATE_TOL", args.levelset_update_tol)
    _set_env_bool("FD_TIMING", args.fd_timing)
    _set_env_bool("FD_JAC_COLS_ONLY", args.fd_jac_cols_only)
    _set_env("FD_DEBUG_COL", args.fd_debug_col)
    _set_env("FD_DEBUG_TOP", args.fd_debug_top)
    _set_env("S_NITSCHE", args.s_nitsche_value)
    _set_env("S_NITSCHE_VALUE", args.s_nitsche_value)
    _set_env("PENALTY_VAL", args.penalty_val)
    _set_env("PENALTY_GRAD", args.penalty_grad)
    _set_env("SOLID_REG_EPS", args.solid_reg_eps)
    _set_env_bool("FD_REUSE_KERNELS", args.fd_reuse_kernels)
    _set_env_bool("FD_SKIP_FULL", args.fd_skip_full)
    _set_env_bool("FD_TERM_SPLIT", args.fd_term_split)
    _set_env_bool("FD_INTERFACE_SPLIT", args.fd_interface_split)
    _set_env("FD_TERM", args.fd_term)
    _set_env("MAX_STEPS", args.max_steps)
    _set_env("FINAL_TIME", args.final_time)
    _set_env_bool("ALLOW_DT_REDUCTION", args.allow_dt_reduction)
    _set_env("DT_REDUCTION_FACTOR", args.dt_reduction_factor)
    _set_env("DT_REDUCTION_THRESHOLD", args.dt_reduction_threshold)
    _set_env("NEWTON_TOL", args.newton_tol)
    _set_env("MAX_NEWTON_ITER", args.max_newton_iter)
    _set_env("NONLINEAR_SOLVER", args.nonlinear_solver)
    _set_env("LINEAR_SOLVER", args.linear_solver)
    _set_env("LS_MODE", args.ls_mode)
    _set_env("LS_MAX_ITER", args.ls_max_iter)
    _set_env("U_MEAN", args.u_mean)
    _set_env("RHO_S", args.rho_s)
    _set_env("THETA", args.theta)
    _set_env("TUREK_CASE", args.turek_case)
    _set_env("VTK_INTERFACE_CLAMP", args.vtk_interface_clamp)
    if args.jit_backend:
        os.environ["PYCUTFEM_JIT_BACKEND"] = str(args.jit_backend)


_apply_env_overrides(ARGS)

# -----------------------------------------------------------------------------
# Problem parameters (Turek–Hron FSI-2)
# -----------------------------------------------------------------------------
H = 0.41
L = 2.2
RADIUS = 0.05
CENTER = (0.2, 0.2)

RHO_F = 1.0e3
MU_F = 1.0
U_MEAN = float(ARGS.u_mean)
U_MAX = 1.5 * U_MEAN

NU_S = 0.4
MU_S = 0.5e6
E_S = 2.0 * MU_S * (1.0 + NU_S)
RHO_S = float(ARGS.rho_s)
MU_S = E_S / (2.0 * (1.0 + NU_S))
LAMBDA_S = E_S * NU_S / ((1.0 + NU_S) * (1.0 - 2.0 * NU_S))

BEAM_LENGTH = 0.35 
BEAM_HEIGHT = 0.02
BEAM_CENTER = (CENTER[0] + RADIUS + 0.5 * BEAM_LENGTH, CENTER[1])
BEAM_SHIFT_X = float(os.getenv("BEAM_SHIFT_X", "0.0"))
BEAM_REF_CENTER = (BEAM_CENTER[0] - BEAM_SHIFT_X, BEAM_CENTER[1])
BEAM_REF_LENGTH = BEAM_LENGTH + 2.0 * BEAM_SHIFT_X
BEAM_REF_HEIGHT = BEAM_HEIGHT
POINT_B = (0.15, 0.2)
POINT_A_INITIAL = (0.6, 0.2) # Point A will change while Point B is fixed

BETA_PENALTY = float(os.getenv("BETA_PENALTY", "20.0")) * MU_F
DT = float(ARGS.dt)
POLY_ORDER = int(ARGS.poly_order)
USE_TAYLOR_HOOD = os.getenv("PYCUTFEM_TAYLOR_HOOD", "1") not in ("0", "false", "False")
REDUCE_ORDER_SOLID = os.getenv("REDUCE_ORDER_SOLID", "1") not in ("0", "false", "False")
PRESSURE_ORDER = (POLY_ORDER - 1) if USE_TAYLOR_HOOD else POLY_ORDER
SOLID_ORDER = POLY_ORDER - 1 if REDUCE_ORDER_SOLID else POLY_ORDER
if SOLID_ORDER < 1:
    SOLID_ORDER = 1
if PRESSURE_ORDER < 1:
    PRESSURE_ORDER = 1
    if USE_TAYLOR_HOOD:
        print("[element] Requested Taylor–Hood but POLY_ORDER<2; falling back to equal-order pressure.")
MESH_SIZE = float(ARGS.mesh_size)
FD_BACKEND = ARGS.fd_backend
CASE_LABEL = str(getattr(ARGS, "case_label", "FSI-2"))
BEAM_ROOT_TOL = float(max(1.0e-6, 1.0e-3 * MESH_SIZE))
BEAM_ROOT_BIAS = float(max(1.0e-8, 1.0e-4 * MESH_SIZE))
BEAM_ROOT_INSET = float(os.getenv("BEAM_ROOT_INSET", str(max(5.0e-4, 0.04 * MESH_SIZE))))
BEAM_ROOT_DOF_TOL = float(os.getenv("BEAM_ROOT_DOF_TOL", str(max(0.2 * MESH_SIZE, 1.0e-3))))
PIN_PRESSURE = os.getenv("PIN_PRESSURE", "1") not in ("0", "false", "False")
SOLID_CUT_DROP = float(os.getenv("SOLID_CUT_DROP", "0.0"))
USE_ALIGNED_INTERFACE = os.getenv("USE_ALIGNED_INTERFACE", "1") not in ("0", "false", "False")
SOLID_ADVECT_LAGGED = os.getenv("SOLID_ADVECT_LAGGED", "1") not in ("0", "false", "False")
USE_RESTRICTED_FORMS = os.getenv("USE_RESTRICTED_FORMS", "1") not in ("0", "false", "False")
USE_LINEAR_SOLID = os.getenv("USE_LINEAR_SOLID", "1") not in ("0", "false", "False")
USE_LINEAR_INTERFACE = os.getenv("PYCUTFEM_LINEAR_INTERFACE", "0") not in ("0", "false", "False")
USE_SKEW_CONVECTION = os.getenv("PYCUTFEM_SKEW_CONVECTION", "0") not in ("0", "false", "False")
SOLID_SKEW_ADD_DIV = os.getenv("PYCUTFEM_SOLID_SKEW_ADD_DIV", "0") not in ("0", "false", "False")
USE_DT_SCALED_NITSCHE_PENALTY = os.getenv("PYCUTFEM_NITSCHE_PENALTY_DT", "0") not in ("0", "false", "False")
SOLID_SYM_NITSCHE_LAGGED = os.getenv("SOLID_SYM_NITSCHE_LAGGED", "1") not in ("0", "false", "False")
SOLID_VEL_GHOST_DT_SCALE = os.getenv("PYCUTFEM_SOLID_VEL_GHOST_DT_SCALE", "1") not in ("0", "false", "False")
SOLID_VEL_GHOST_MASS = float(os.getenv("PYCUTFEM_SOLID_VEL_GHOST_MASS", "0.0"))
FLUID_VEL_GHOST_INERTIA = os.getenv("PYCUTFEM_FLUID_VEL_GHOST_INERTIA", "0") not in ("0", "false", "False")
SOLID_KINEMATIC_STAB = float(os.getenv("SOLID_KINEMATIC_STAB", "0.0"))
SOLID_KINEMATIC_STAB_EXP = float(os.getenv("SOLID_KINEMATIC_STAB_EXP", "2.0"))
PYCUTFEM_INTERFACE_PENALTY_SLIVER_WEIGHT = os.getenv("PYCUTFEM_INTERFACE_PENALTY_SLIVER_WEIGHT", "0") not in (
    "0",
    "false",
    "False",
)
LEVELSET_ZERO_EPS = float(os.getenv("LEVELSET_ZERO_EPS", str(max(1.0e-10, 1.0e-8 * MESH_SIZE))))
LEVELSET_EDGE_TOL = float(os.getenv("LEVELSET_EDGE_TOL", str(LEVELSET_ZERO_EPS)))
LEVELSET_ZERO_SIDE = os.getenv("LEVELSET_ZERO_SIDE", "neg").strip().lower()
LEVELSET_PREFER_NEGATIVE = LEVELSET_ZERO_SIDE != "pos"
LEVELSET_UPDATE_TOL = float(os.getenv("LEVELSET_UPDATE_TOL", str(max(1.0e-10, 1.0e-8 * MESH_SIZE))))
LEVELSET_EDGE_TOL = max(LEVELSET_EDGE_TOL, LEVELSET_UPDATE_TOL)
_snap_env = os.getenv("LEVELSET_SNAP_EPS", "").strip()
if _snap_env:
    LEVELSET_SNAP_EPS = float(_snap_env)
else:
    LEVELSET_SNAP_EPS = max(1.0e-3 * MESH_SIZE, 10.0 * LEVELSET_EDGE_TOL) if USE_ALIGNED_INTERFACE else 0.0
_nudge_env = os.getenv("LEVELSET_NUDGE_EPS", "").strip()
if _nudge_env:
    LEVELSET_NUDGE_EPS = float(_nudge_env)
else:
    LEVELSET_NUDGE_EPS = 0.0 if USE_ALIGNED_INTERFACE else LEVELSET_EDGE_TOL
if USE_ALIGNED_INTERFACE and LEVELSET_NUDGE_EPS > 0.0:
    print("[interface] LEVELSET_NUDGE_EPS>0 with USE_ALIGNED_INTERFACE=1 will shift aligned edges.")

# Sliver stabilization weights for near-aligned cuts
SLIVER_THETA0 = float(os.getenv("SLIVER_THETA0", "0.05"))
SLIVER_P = float(os.getenv("SLIVER_P", "1.0"))
SLIVER_WMAX = float(os.getenv("SLIVER_WMAX", "1000.0"))
SLIVER_THETAMIN = float(os.getenv("SLIVER_THETAMIN", "1e-6"))
SLIVER_SMOOTH = float(os.getenv("SLIVER_SMOOTH", "0.3"))

# Optional CIP stabilization (ghost-edge interior penalty) to damp advection-driven peaks.
# CIP-like stabilization (ghost-edge gradient-jump penalty) scaled by rho*h*|u|.
# Default is enabled at a mild level for robustness of the fully Eulerian FSI-2 run.
PYCUTFEM_CIP_BETA_FLUID = float(os.getenv("PYCUTFEM_CIP_BETA_FLUID", "0.1"))
PYCUTFEM_CIP_BETA_SOLID = float(os.getenv("PYCUTFEM_CIP_BETA_SOLID", "0.1"))
PYCUTFEM_CIP_BETA_DISP = float(os.getenv("PYCUTFEM_CIP_BETA_DISP", "0.1"))
PYCUTFEM_CIP_LAGGED = os.getenv("PYCUTFEM_CIP_LAGGED", "1") not in ("0", "false", "False")
PYCUTFEM_CIP_U_EPS = float(os.getenv("PYCUTFEM_CIP_U_EPS", "1e-12"))

# Optional fluid mass-jump ghost penalty (helps control near-nullspace modes in extreme slivers).
PYCUTFEM_FLUID_VEL_GHOST_MASS = float(os.getenv("PYCUTFEM_FLUID_VEL_GHOST_MASS", "0.0"))

# Optional sliver-robust cut-cell mass regularization:
# adds (ρ/dt) * (1/θ) * ⟨u, v⟩ on the *cut* subdomain so tiny cut volumes cannot
# carry arbitrarily large values with negligible cost.
PYCUTFEM_SLIVER_MASS_FLUID = float(os.getenv("PYCUTFEM_SLIVER_MASS_FLUID", "0.0"))
PYCUTFEM_SLIVER_MASS_SOLID = float(os.getenv("PYCUTFEM_SLIVER_MASS_SOLID", "0.0"))

# Optional: pressure stabilization on the fluid ghost-edge set (Richter-style).
# This targets equal-order (Q2/Q2) pressure robustness near the moving interface.
# Set one or both to a small value (e.g. 0.1–10) to enable:
# - PYCUTFEM_PRESSURE_STAB_JUMP:   h^3 [∇p]·[∇q]  on ghost edges (jump form)
# - PYCUTFEM_PRESSURE_STAB_AVG:    h^3 {∇p·∇q}    on ghost edges (average form)
PYCUTFEM_PRESSURE_STAB_JUMP = float(os.getenv("PYCUTFEM_PRESSURE_STAB_JUMP", "0.0"))
PYCUTFEM_PRESSURE_STAB_AVG = float(os.getenv("PYCUTFEM_PRESSURE_STAB_AVG", "0.0"))

# -----------------------------------------------------------------------------
# Mesh and boundary helpers
# -----------------------------------------------------------------------------


class BeamArcRootLevelSet:
    """
    Beam level set with a curved root that follows the cylinder arc, so the
    beam attaches without leaving a vertical gap at x=beam_x0.
    """

    def __init__(
        self,
        *,
        beam_center: tuple[float, float],
        beam_length: float,
        beam_height: float,
        cyl_center: tuple[float, float],
        cyl_radius: float,
        root_inset: float,
        root_bias: float,
        root_tol: float,
    ):
        self.cx = float(beam_center[0])
        self.cy = float(beam_center[1])
        self.hx = 0.5 * float(beam_length)
        self.hy = 0.5 * float(beam_height)
        self._beam_x1 = self.cx + self.hx
        self._beam_y0 = self.cy - self.hy
        self._beam_y1 = self.cy + self.hy
        self._cyl_center = np.asarray(cyl_center, dtype=float)
        self._cyl_radius = float(cyl_radius)
        self._root_inset = float(root_inset)
        self._root_bias = float(root_bias)
        self._root_tol = float(root_tol)
        self.cache_token = (
            "beam_arc_root",
            float(beam_center[0]),
            float(beam_center[1]),
            float(beam_length),
            float(beam_height),
            float(cyl_radius),
            float(root_inset),
            float(root_bias),
            float(root_tol),
        )

    def _x_arc(self, y: np.ndarray) -> np.ndarray:
        dy = y - self._cyl_center[1]
        rad2 = self._cyl_radius * self._cyl_radius
        inside = np.maximum(rad2 - dy * dy, 0.0)
        return self._cyl_center[0] + np.sqrt(inside) - self._root_inset

    def __call__(self, x):
        x = np.asarray(x, float)
        x_coord = x[..., 0]
        y_coord = x[..., 1]

        x_arc = self._x_arc(y_coord)
        phi_left = x_arc - x_coord
        phi_right = x_coord - self._beam_x1
        phi_top = y_coord - self._beam_y1
        phi_bottom = self._beam_y0 - y_coord

        phi = np.max(np.stack((phi_left, phi_right, phi_top, phi_bottom), axis=-1), axis=-1)
        if self._root_bias > 0.0:
            if x.ndim == 1:
                on_root = (
                    (np.abs(phi_left) <= self._root_tol)
                    and (self._beam_y0 - self._root_tol <= y_coord <= self._beam_y1 + self._root_tol)
                    and (x_coord >= self._cyl_center[0])
                )
                if on_root:
                    phi = min(float(phi), -self._root_bias)
            else:
                on_root = (
                    (np.abs(phi_left) <= self._root_tol)
                    & (y_coord >= self._beam_y0 - self._root_tol)
                    & (y_coord <= self._beam_y1 + self._root_tol)
                    & (x_coord >= self._cyl_center[0])
                )
                phi = np.where(on_root, np.minimum(phi, -self._root_bias), phi)
        return phi

    def gradient(self, x):
        x = np.asarray(x, float)
        x_coord = x[..., 0]
        y_coord = x[..., 1]

        x_arc = self._x_arc(y_coord)
        phi_left = x_arc - x_coord
        phi_right = x_coord - self._beam_x1
        phi_top = y_coord - self._beam_y1
        phi_bottom = self._beam_y0 - y_coord
        phis = np.stack((phi_left, phi_right, phi_top, phi_bottom), axis=-1)
        idx = np.argmax(phis, axis=-1)

        dy = y_coord - self._cyl_center[1]
        rad2 = self._cyl_radius * self._cyl_radius
        denom = np.sqrt(np.maximum(rad2 - dy * dy, 1.0e-18))
        dx_arc_dy = -dy / denom

        grad_left = np.stack((-np.ones_like(x_coord), dx_arc_dy), axis=-1)
        grad_right = np.stack((np.ones_like(x_coord), np.zeros_like(x_coord)), axis=-1)
        grad_top = np.stack((np.zeros_like(x_coord), np.ones_like(x_coord)), axis=-1)
        grad_bottom = np.stack((np.zeros_like(x_coord), -np.ones_like(x_coord)), axis=-1)
        grads = np.stack((grad_left, grad_right, grad_top, grad_bottom), axis=-2)
        if x.ndim == 1:
            return grads[int(idx)]
        grad = np.take_along_axis(grads, idx[..., None, None], axis=-2).squeeze(-2)
        return grad


def _count_segments(width: float, mesh_size: float, *, min_cells: int = 1) -> int:
    if width <= 1.0e-12:
        return 0
    return max(min_cells, int(math.ceil(width / mesh_size)))


def _nodes_for_length(length: float, mesh_size: float, *, min_nodes: int = 3) -> int:
    segments = max(1, int(round(length / mesh_size)))
    return max(min_nodes, segments + 1)


def build_blocked_gmsh_mesh(
    path: Path,
    mesh_size: float,
    poly_order: int,
    *,
    view: bool = False,
    beam_center: tuple[float, float] | None = None,
    beam_length: float | None = None,
    beam_height: float | None = None,
) -> None:
    """
    Build a blocked, beam-aligned O-grid mesh with gmsh.
    """
    if gmsh is None:
        raise RuntimeError("Gmsh backend requested but the gmsh Python module is not available.")

    if beam_center is None:
        beam_center = BEAM_CENTER
    if beam_length is None:
        beam_length = BEAM_LENGTH
    if beam_height is None:
        beam_height = BEAM_HEIGHT

    gmsh.initialize()
    try:
        gmsh.model.add("turek_fsi_blocked")
        occ = gmsh.model.occ

        # Helper registries ---------------------------------------------------
        def _point_key(x: float, y: float) -> tuple[float, float]:
            return (round(float(x), 12), round(float(y), 12))

        point_lookup: dict[tuple[float, float], int] = {}
        point_coords: dict[int, tuple[float, float]] = {}

        def add_point(x: float, y: float) -> int:
            key = _point_key(x, y)
            if key in point_lookup:
                return point_lookup[key]
            tag = occ.addPoint(x, y, 0.0)
            point_lookup[key] = tag
            point_coords[tag] = (float(x), float(y))
            return tag

        line_lookup: dict[tuple[int, int], int] = {}
        line_lengths: dict[int, float] = {}
        line_target_nodes: dict[int, int] = {}
        line_meta: dict[int, tuple[int, int]] = {}

        def register_line(tag: int, start: int, end: int, length: float) -> None:
            line_lookup[(start, end)] = tag
            line_lengths[tag] = length

        def oriented_line(start: int, end: int) -> int:
            tag = line_lookup.get((start, end))
            if tag is not None:
                return tag
            tag = line_lookup.get((end, start))
            if tag is None:
                raise KeyError(f"No curve between points {start} and {end}.")
            return -tag

        boundary_edges: dict[str, list[int]] = {"inlet": [], "outlet": [], "walls": [], "cylinder": []}

        def add_line(start: int, end: int, boundary: str | None = None) -> int:
            tag = occ.addLine(start, end)
            x0, y0 = point_coords[start]
            x1, y1 = point_coords[end]
            length = math.hypot(x1 - x0, y1 - y0)
            register_line(tag, start, end, length)
            line_meta[tag] = (start, end)
            if boundary:
                boundary_edges[boundary].append(tag)
            line_target_nodes[tag] = line_target_nodes.get(tag, _nodes_for_length(length, mesh_size))
            return tag

        def _coord_index(seq: Sequence[float], value: float) -> int:
            for i, v in enumerate(seq):
                if abs(v - value) <= 1.0e-12:
                    return i
            raise ValueError(f"{value} not found in coordinate list.")

        def _arc_length(a_tag: int, b_tag: int) -> float:
            xa, ya = point_coords[a_tag]
            xb, yb = point_coords[b_tag]
            ang_a = math.atan2(ya - CENTER[1], xa - CENTER[0])
            ang_b = math.atan2(yb - CENTER[1], xb - CENTER[0])
            delta = ang_b - ang_a
            if delta <= 0.0:
                delta += 2.0 * math.pi
            return RADIUS * delta

        # Beam-aware square around the cylinder --------------------------------
        beam_x0 = beam_center[0] - 0.5 * beam_length
        beam_x1 = beam_center[0] + 0.5 * beam_length
        beam_y0 = beam_center[1] - 0.5 * beam_height
        beam_y1 = beam_center[1] + 0.5 * beam_height

        pad = max(0.6 * mesh_size, 0.008)
        hx = max(RADIUS + pad, min(CENTER[0] - pad, L - CENTER[0] - pad, 0.35))
        hy = max(RADIUS + pad, min(CENTER[1] - pad, H - CENTER[1] - pad, 0.35))
        if hx <= RADIUS or hy <= RADIUS:
            raise RuntimeError("Beam-aware O-grid collapsed; increase mesh size or adjust parameters.")

        square_left = CENTER[0] - hx
        square_right = CENTER[0] + hx
        square_bottom = CENTER[1] - hy
        square_top = CENTER[1] + hy

        beam_nodes = max(4, _nodes_for_length(beam_height, mesh_size, min_nodes=4))

        x_coords = sorted({0.0, square_left, square_right, beam_x0, beam_x1, L})
        y_coords = sorted({0.0, square_bottom, beam_y0, beam_center[1], beam_y1, square_top, H})

        x_interval_nodes = []
        for ix in range(len(x_coords) - 1):
            length = x_coords[ix + 1] - x_coords[ix]
            x_interval_nodes.append(_nodes_for_length(length, mesh_size))

        y_interval_nodes = []
        for iy in range(len(y_coords) - 1):
            length = y_coords[iy + 1] - y_coords[iy]
            count = _nodes_for_length(length, mesh_size)
            if y_coords[iy] >= square_bottom - 1.0e-12 and y_coords[iy + 1] <= square_top + 1.0e-12:
                count = max(count, beam_nodes)
            y_interval_nodes.append(count)

        grid_points: dict[tuple[int, int], int] = {}
        for ix, x in enumerate(x_coords):
            for iy, y in enumerate(y_coords):
                grid_points[(ix, iy)] = add_point(x, y)

        def _inside_inner(x_mid: float, y_mid: float) -> bool:
            return (square_left < x_mid < square_right) and (square_bottom < y_mid < square_top)

        horizontal_lines: dict[tuple[int, int], int] = {}
        for iy, y in enumerate(y_coords):
            for ix in range(len(x_coords) - 1):
                x0, x1 = x_coords[ix], x_coords[ix + 1]
                if _inside_inner(0.5 * (x0 + x1), y):
                    continue
                boundary = None
                if abs(y - 0.0) <= 1.0e-12 or abs(y - H) <= 1.0e-12:
                    boundary = "walls"
                tag = add_line(grid_points[(ix, iy)], grid_points[(ix + 1, iy)], boundary=boundary)
                horizontal_lines[(iy, ix)] = tag
                line_target_nodes[tag] = max(line_target_nodes.get(tag, 0), x_interval_nodes[ix])

        vertical_lines: dict[tuple[int, int], int] = {}
        for ix, x in enumerate(x_coords):
            for iy in range(len(y_coords) - 1):
                y0, y1 = y_coords[iy], y_coords[iy + 1]
                if _inside_inner(x, 0.5 * (y0 + y1)):
                    continue
                boundary = None
                if abs(x - 0.0) <= 1.0e-12:
                    boundary = "inlet"
                elif abs(x - L) <= 1.0e-12:
                    boundary = "outlet"
                tag = add_line(grid_points[(ix, iy)], grid_points[(ix, iy + 1)], boundary=boundary)
                vertical_lines[(ix, iy)] = tag
                line_target_nodes[tag] = max(line_target_nodes.get(tag, 0), y_interval_nodes[iy])

        fluid_surfaces: list[int] = []
        surface_loops: list[list[int]] = []
        for ix in range(len(x_coords) - 1):
            for iy in range(len(y_coords) - 1):
                xm = 0.5 * (x_coords[ix] + x_coords[ix + 1])
                ym = 0.5 * (y_coords[iy] + y_coords[iy + 1])
                if _inside_inner(xm, ym):
                    continue
                loop = [
                    horizontal_lines[(iy, ix)],
                    vertical_lines[(ix + 1, iy)],
                    -horizontal_lines[(iy + 1, ix)],
                    -vertical_lines[(ix, iy)],
                ]
                cloop = occ.addCurveLoop(loop)
                fluid_surfaces.append(occ.addPlaneSurface([cloop]))
                surface_loops.append(loop)

        # Beam-aware O-grid ring around the circle -----------------------------
        x_right_idx = _coord_index(x_coords, square_right)
        x_left_idx = _coord_index(x_coords, square_left)
        y_bot_idx = _coord_index(y_coords, square_bottom)
        y_top_idx = _coord_index(y_coords, square_top)
        y_beam_bot_idx = _coord_index(y_coords, beam_y0)
        y_beam_top_idx = _coord_index(y_coords, beam_y1)
        y_mid_idx = _coord_index(y_coords, CENTER[1])

        # Build square boundary points aligned with grid intersections to avoid T-junctions.
        y_seq = [y_bot_idx, y_beam_bot_idx, y_mid_idx, y_beam_top_idx, y_top_idx]
        top_x_indices = sorted(
            {idx for idx, xv in enumerate(x_coords) if square_left - 1e-12 <= xv <= square_right + 1e-12},
            reverse=True,
        )

        def _append_unique(coords_list: list[tuple[int, int]], coord: tuple[int, int]) -> None:
            if not coords_list or coords_list[-1] != coord:
                coords_list.append(coord)

        square_point_coords: list[tuple[int, int]] = []
        # right edge bottom→top
        for yi in y_seq:
            _append_unique(square_point_coords, (x_right_idx, yi))
        # top edge right→left
        for idx in top_x_indices[1:]:
            _append_unique(square_point_coords, (idx, y_top_idx))
        # left edge top→bottom (skip repeating top)
        for yi in reversed(y_seq[:-1]):
            _append_unique(square_point_coords, (x_left_idx, yi))
        # bottom edge left→right (skip left corner)
        bot_x_indices = sorted({idx for idx in top_x_indices}, reverse=False)
        for idx in bot_x_indices[1:]:
            _append_unique(square_point_coords, (idx, y_bot_idx))
        # close by omitting duplicate of starting point
        if square_point_coords and square_point_coords[-1] == square_point_coords[0]:
            square_point_coords.pop()

        square_points = [grid_points[(ix, iy)] for ix, iy in square_point_coords]

        center_pt = add_point(CENTER[0], CENTER[1])

        def _circle_point_through_square(nid_square: int) -> int:
            xs, ys = point_coords[nid_square]
            vec = np.array([xs - CENTER[0], ys - CENTER[1]], float)
            rlen = float(np.hypot(vec[0], vec[1]))
            if rlen <= 1.0e-14:
                raise RuntimeError("Square point coincides with circle center.")
            scale = RADIUS / rlen
            xc = CENTER[0] + scale * vec[0]
            yc = CENTER[1] + scale * vec[1]
            return add_point(xc, yc)

        circle_points = [_circle_point_through_square(pid) for pid in square_points]

        arc_lines: list[int] = []
        for i in range(len(circle_points)):
            start = circle_points[i]
            end = circle_points[(i + 1) % len(circle_points)]
            tag = occ.addCircleArc(start, center_pt, end)
            length = _arc_length(start, end)
            register_line(tag, start, end, length)
            line_target_nodes[tag] = max(line_target_nodes.get(tag, 0), _nodes_for_length(length, mesh_size))
            boundary_edges["cylinder"].append(tag)
            arc_lines.append(tag)

        square_segments: list[int] = []
        def _segment_target_nodes(a_tag: int, b_tag: int) -> int:
            xa, ya = point_coords[a_tag]
            xb, yb = point_coords[b_tag]
            if abs(ya - yb) <= 1.0e-12:
                ix0 = _coord_index(x_coords, min(xa, xb))
                return x_interval_nodes[ix0]
            if abs(xa - xb) <= 1.0e-12:
                iy0 = _coord_index(y_coords, min(ya, yb))
                return y_interval_nodes[iy0]
            return _nodes_for_length(math.hypot(xb - xa, yb - ya), mesh_size)

        for i in range(len(square_points)):
            a = square_points[i]
            b = square_points[(i + 1) % len(square_points)]
            try:
                seg = oriented_line(a, b)
            except KeyError:
                seg = add_line(a, b)
            square_segments.append(seg)
            seg_tag = abs(seg)
            line_target_nodes[seg_tag] = max(line_target_nodes.get(seg_tag, 0), _segment_target_nodes(a, b))

        radial_nodes = max(beam_nodes, _nodes_for_length(max(hx, hy) - RADIUS, mesh_size, min_nodes=4))
        radial_lines: list[int] = []
        for i, (c_pt, s_pt) in enumerate(zip(circle_points, square_points)):
            tag = add_line(c_pt, s_pt)
            default_nodes = max(radial_nodes, beam_nodes)
            line_target_nodes[tag] = max(default_nodes, line_target_nodes.get(tag, default_nodes))
            radial_lines.append(tag)

        for i, arc in enumerate(arc_lines):
            seg_tag = abs(square_segments[i])
            a = square_points[i]
            b = square_points[(i + 1) % len(square_points)]
            seg_nodes = _segment_target_nodes(a, b)
            arc_nodes = _nodes_for_length(line_lengths[arc], mesh_size)
            pair_count = max(seg_nodes, arc_nodes)
            xa, ya = point_coords[a]
            xb, yb = point_coords[b]
            if abs(ya - yb) <= 1.0e-12:
                ix0 = _coord_index(x_coords, min(xa, xb))
                x_interval_nodes[ix0] = max(x_interval_nodes[ix0], pair_count)
                pair_count = max(pair_count, x_interval_nodes[ix0])
            elif abs(xa - xb) <= 1.0e-12:
                iy0 = _coord_index(y_coords, min(ya, yb))
                y_interval_nodes[iy0] = max(y_interval_nodes[iy0], pair_count)
                pair_count = max(pair_count, y_interval_nodes[iy0])
            line_target_nodes[arc] = pair_count
            line_target_nodes[seg_tag] = pair_count

            ra = radial_lines[i]
            rb = radial_lines[(i + 1) % len(radial_lines)]
            radial_count = max(
                pair_count,
                line_target_nodes.get(ra, radial_nodes),
                line_target_nodes.get(rb, radial_nodes),
                radial_nodes,
            )
            line_target_nodes[ra] = radial_count
            line_target_nodes[rb] = radial_count

        for (iy, ix), tag in horizontal_lines.items():
            line_target_nodes[tag] = max(line_target_nodes.get(tag, 0), x_interval_nodes[ix])
        for (ix, iy), tag in vertical_lines.items():
            line_target_nodes[tag] = max(line_target_nodes.get(tag, 0), y_interval_nodes[iy])
        for i, seg in enumerate(square_segments):
            a = square_points[i]
            b = square_points[(i + 1) % len(square_points)]
            xa, ya = point_coords[a]
            xb, yb = point_coords[b]
            if abs(ya - yb) <= 1.0e-12:
                ix0 = _coord_index(x_coords, min(xa, xb))
                target = x_interval_nodes[ix0]
            elif abs(xa - xb) <= 1.0e-12:
                iy0 = _coord_index(y_coords, min(ya, yb))
                target = y_interval_nodes[iy0]
            else:
                target = line_target_nodes.get(abs(seg), _segment_target_nodes(a, b))
            line_target_nodes[abs(seg)] = max(line_target_nodes.get(abs(seg), 0), target)
            line_target_nodes[arc_lines[i]] = max(line_target_nodes.get(arc_lines[i], 0), target)

        for i in range(len(arc_lines)):
            pair = max(line_target_nodes.get(arc_lines[i], 0), line_target_nodes.get(abs(square_segments[i]), 0))
            line_target_nodes[arc_lines[i]] = pair
            line_target_nodes[abs(square_segments[i])] = pair
            rad_pair = max(
                line_target_nodes.get(radial_lines[i], 0),
                line_target_nodes.get(radial_lines[(i + 1) % len(radial_lines)], 0),
            )
            line_target_nodes[radial_lines[i]] = rad_pair
            line_target_nodes[radial_lines[(i + 1) % len(radial_lines)]] = rad_pair

        for i in range(len(circle_points)):
            next_i = (i + 1) % len(circle_points)
            loop = [
                arc_lines[i],
                radial_lines[next_i],
                -square_segments[i],
                -radial_lines[i],
            ]
            cloop = occ.addCurveLoop(loop)
            fluid_surfaces.append(occ.addPlaneSurface([cloop]))
            surface_loops.append(loop)

        for tag, loop in zip(fluid_surfaces, surface_loops):
            if len(loop) != 4:
                continue
            a = abs(loop[0])
            b = abs(loop[1])
            c = abs(loop[2])
            d = abs(loop[3])
            na = line_target_nodes.get(a)
            nc = line_target_nodes.get(c)
            nb = line_target_nodes.get(b)
            nd = line_target_nodes.get(d)
            if na != nc or nb != nd:
                def _edge_info(edge_tag: int) -> str:
                    pts = line_meta.get(abs(edge_tag))
                    if pts:
                        p0, p1 = pts
                        x0, y0 = point_coords[p0]
                        x1, y1 = point_coords[p1]
                        return f"{abs(edge_tag)}:({x0:.4f},{y0:.4f})->({x1:.4f},{y1:.4f})"
                    return str(abs(edge_tag))
                raise RuntimeError(
                    f"Transfinite mismatch on surface {tag}: "
                    f"edges {[ _edge_info(e) for e in loop ]} have node counts {(na, nc, nb, nd)}"
                )

        occ.synchronize()
        gmsh.model.mesh.setCompound(2, fluid_surfaces)

        gmsh.model.addPhysicalGroup(2, fluid_surfaces, tag=1)
        gmsh.model.setPhysicalName(2, 1, "fluid")
        boundary_tag_hints = {"inlet": 11, "outlet": 12, "walls": 13, "cylinder": 14}
        for name, tag_hint in boundary_tag_hints.items():
            edges = sorted(set(boundary_edges[name]))
            if not edges:
                continue
            tag = gmsh.model.addPhysicalGroup(1, edges, tag=tag_hint)
            gmsh.model.setPhysicalName(1, tag, name)

        radial_set = set(radial_lines)
        for tag, length in line_lengths.items():
            target_nodes = int(line_target_nodes.get(tag, _nodes_for_length(length, mesh_size)))
            progression = 1.0
            if tag in radial_set:
                progression = 1.12
            gmsh.model.mesh.setTransfiniteCurve(tag, target_nodes, "Progression", progression)

        for surf in fluid_surfaces:
            gmsh.model.mesh.setTransfiniteSurface(surf)
            gmsh.model.mesh.setRecombine(2, surf)

        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(int(poly_order))

        path.parent.mkdir(parents=True, exist_ok=True)
        if view:
            try:
                gmsh.fltk.initialize()
                gmsh.fltk.run()
            except Exception:
                print("Gmsh GUI not available; skipping mesh preview.")
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def build_structured_channel_mesh(mesh_size: float, poly_order: int) -> Mesh:
    """
    Structured O-grid mesh of the channel with a circular hole.
    """
    # Keep a thin gap between the circle and the inner box even for coarse meshes.
    base_margin = max(2.5 * mesh_size, 0.015)
    min_half_cap = min(CENTER[0], L - CENTER[0], CENTER[1], H - CENTER[1])
    min_ring = max(0.5 * mesh_size, 0.005)
    max_margin = min_half_cap - (RADIUS + min_ring)
    if max_margin <= 0.0:
        raise RuntimeError("O-grid collapsed: circle too close to boundary; reduce radius or move center.")
    margin = min(base_margin, max_margin)

    half_x_cap = min(CENTER[0], L - CENTER[0]) - margin
    half_y_cap = min(CENTER[1], H - CENTER[1]) - margin
    hx = half_x_cap
    hy = half_y_cap
    ring_thickness = min(hx, hy) - RADIUS
    if ring_thickness <= 0.0:
        raise RuntimeError("Ring thickness must be positive for the structured mesh.")

    x_inner_left = CENTER[0] - hx
    x_inner_right = CENTER[0] + hx
    y_inner_bottom = CENTER[1] - hy
    y_inner_top = CENTER[1] + hy

    nx_left = _count_segments(x_inner_left - 0.0, mesh_size, min_cells=1)
    nx_right = _count_segments(L - x_inner_right, mesh_size, min_cells=1)
    nx_mid = _count_segments(x_inner_right - x_inner_left, mesh_size, min_cells=4)
    if nx_mid % 2:
        nx_mid += 1

    ny_bottom = _count_segments(y_inner_bottom - 0.0, mesh_size, min_cells=1)
    ny_top = _count_segments(H - y_inner_top, mesh_size, min_cells=1)
    ny_mid = _count_segments(y_inner_top - y_inner_bottom, mesh_size, min_cells=4)
    if ny_mid % 2:
        ny_mid += 1

    n_radial_layers = max(2, _count_segments(ring_thickness, mesh_size, min_cells=2))

    nodes, elements, edges, corners = circular_hole_ogrid(
        L,
        H,
        circle_center=CENTER,
        circle_radius=RADIUS,
        ring_thickness=ring_thickness,
        n_radial_layers=n_radial_layers,
        nx_outer=(nx_left, nx_mid, nx_right),
        ny_outer=(ny_bottom, ny_mid, ny_top),
        poly_order=poly_order,
        outer_box_half_lengths=(hx, hy),
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elements,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )
    tag_channel_boundaries(mesh, mesh_size)
    return mesh


def tag_channel_boundaries(mesh: Mesh, mesh_size: float) -> None:
    """
    Tag inlet, outlet, walls, and the rigid cylinder boundary.
    """
    tol = 1.0e-9
    circle_tol = max(0.25 * mesh_size, 1.0e-4)
    rect_locators = {
        "inlet": lambda x, y: abs(x - 0.0) <= tol,
        "outlet": lambda x, y: abs(x - L) <= tol,
        "walls": lambda x, y: abs(y - 0.0) <= tol or abs(y - H) <= tol,
    }
    mesh.tag_boundary_edges(rect_locators)

    def on_circle(x: float, y: float) -> bool:
        return abs(math.hypot(x - CENTER[0], y - CENTER[1]) - RADIUS) <= circle_tol

    for edge in mesh.edges_list:
        if edge.right is not None:
            continue
        mpx, mpy = mesh.nodes_x_y_pos[list(edge.nodes)].mean(axis=0)
        if on_circle(mpx, mpy):
            edge.tag = "cylinder"
    # cache
    cyl_mask = np.fromiter((getattr(e, "tag", "") == "cylinder" for e in mesh.edges_list), bool)
    mesh._edge_bitsets = getattr(mesh, "_edge_bitsets", {})
    mesh._edge_bitsets["cylinder"] = BitSet(cyl_mask)
    loc_map = getattr(mesh, "_boundary_locators", {})
    loc_map["cylinder"] = on_circle
    mesh._boundary_locators = loc_map


def _beam_root_locator(
    mesh: Mesh,
    beam_center: tuple[float, float],
    beam_length: float,
    beam_height: float,
    *,
    tol: float = 0.0,
):
    cx, cy = CENTER
    r2 = RADIUS * RADIUS
    beam_x0 = beam_center[0] - 0.5 * beam_length
    beam_y0 = beam_center[1] - 0.5 * beam_height
    beam_y1 = beam_center[1] + 0.5 * beam_height

    coords = np.asarray(getattr(mesh, "nodes_x_y_pos", []), float)
    xmin_raw, ymin_raw, xmax_raw, ymax_raw = 0.0, 0.0, L, H
    if coords.size:
        xmin_raw, ymin_raw = coords.min(axis=0)
        xmax_raw, ymax_raw = coords.max(axis=0)
    span = float(max(xmax_raw - xmin_raw, ymax_raw - ymin_raw, 1.0))
    tol_loc = max(tol, 1.0e-4 * span)
    tol_root = max(tol_loc, 4.5e-3 * span)
    tol_y = max(tol_loc, 8e-4 * span)
    tol_x_root = max(tol_loc, 1.5e-3 * span)

    def on_beam_root(x: float, y: float) -> bool:
        on_vertical = abs(x - beam_x0) < tol_x_root and (beam_y0 - tol_y) <= y <= (beam_y1 + tol_y)
        on_arc = (
            abs((x - cx) ** 2 + (y - cy) ** 2 - r2) < tol_root
            and (beam_y0 - tol_y) <= y <= (beam_y1 + tol_y)
            and x >= cx
        )
        return on_vertical or on_arc

    return on_beam_root


def _tag_beam_root_from_cylinder(
    dh: DofHandler,
    mesh: Mesh,
    locator,
    fields: Sequence[str],
    *,
    tag: str = "beam_root",
) -> None:
    loc_map = getattr(mesh, "_boundary_locators", {})
    loc_map[tag] = locator
    mesh._boundary_locators = loc_map

    edge_mask = np.zeros(len(mesh.edges_list), dtype=bool)
    try:
        candidates = mesh.edge_bitset("cylinder").to_indices()
    except Exception:
        candidates = np.arange(len(mesh.edges_list))
    for eid in candidates:
        try:
            e_obj = mesh.edge(int(eid))
        except Exception:
            continue
        if (e_obj.left is not None) and (e_obj.right is not None):
            continue
        mpx, mpy = mesh.nodes_x_y_pos[list(e_obj.nodes)].mean(axis=0)
        if locator(float(mpx), float(mpy)):
            edge_mask[int(eid)] = True

    if edge_mask.any():
        mesh._edge_bitsets = getattr(mesh, "_edge_bitsets", {})
        mesh._edge_bitsets[tag] = BitSet(edge_mask)
    else:
        return

    node_coords = mesh.nodes_x_y_pos
    for field in fields:
        node2dof = dh.dof_map.get(field, {})
        ids: set[int] = set()
        for eid in np.flatnonzero(edge_mask):
            e_obj = mesh.edge(int(eid))
            nodes = e_obj.all_nodes if getattr(e_obj, "all_nodes", ()) else e_obj.nodes
            for nid in nodes:
                x, y = node_coords[int(nid)]
                if locator(float(x), float(y)):
                    gd = node2dof.get(int(nid))
                    if gd is not None:
                        ids.add(int(gd))
        if ids:
            dh.dof_tags.setdefault(tag, set()).update(ids)


def _tag_beam_root_from_levelset(
    dh: DofHandler,
    beam_ls: "BeamArcRootLevelSet",
    fields: Sequence[str],
    *,
    tag: str = "beam_root",
    tol: float | None = None,
) -> int:
    if not hasattr(beam_ls, "_x_arc"):
        return 0
    if not fields:
        return 0
    _ = dh.get_field_slice(fields[0])
    coords = getattr(dh, "_dof_coords", None)
    if coords is None or not len(coords):
        return 0
    tol_x = float(BEAM_ROOT_DOF_TOL if tol is None else tol)
    tol_y = max(0.5 * tol_x, 1.0e-4)
    y = coords[:, 1]
    x_arc = beam_ls._x_arc(y)
    on_root = (
        (np.abs(coords[:, 0] - x_arc) <= tol_x)
        & (y >= beam_ls._beam_y0 - tol_y)
        & (y <= beam_ls._beam_y1 + tol_y)
        & (coords[:, 0] >= beam_ls._cyl_center[0] - tol_x)
    )
    added = 0
    for field in fields:
        ids = np.asarray(dh.get_field_slice(field), dtype=int)
        sel = ids[on_root[ids]]
        if sel.size:
            dh.dof_tags.setdefault(tag, set()).update(map(int, sel))
            added += int(sel.size)
    return added


def _load_gmsh_mesh(
    mesh_size: float,
    poly_order: int,
    *,
    mesh_file: Path | None,
    rebuild: bool,
    view: bool,
    beam_center: tuple[float, float] | None = None,
    beam_length: float | None = None,
    beam_height: float | None = None,
) -> tuple[Mesh, Path | None]:
    if gmsh is None:
        raise RuntimeError("Gmsh backend requested but gmsh is not available.")
    mesh_path = mesh_file.expanduser().resolve() if mesh_file is not None else None
    if mesh_path is not None:
        if rebuild or not mesh_path.exists():
            print(f"Generating gmsh blocked mesh at {mesh_path} (h={mesh_size}, Q{poly_order})")
            build_blocked_gmsh_mesh(
                mesh_path,
                mesh_size,
                poly_order,
                view=view,
                beam_center=beam_center,
                beam_length=beam_length,
                beam_height=beam_height,
            )
        else:
            print(f"Reusing gmsh mesh at {mesh_path}")
        return mesh_from_gmsh(mesh_path), mesh_path

    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "turek_fsi_block.msh"
        print(f"Generating temporary gmsh blocked mesh (h={mesh_size}, Q{poly_order})")
        build_blocked_gmsh_mesh(
            tmp_path,
            mesh_size,
            poly_order,
            view=view,
            beam_center=beam_center,
            beam_length=beam_length,
            beam_height=beam_height,
        )
        mesh = mesh_from_gmsh(tmp_path)
    return mesh, None


def build_channel_mesh(
    mesh_size: float,
    poly_order: int,
    *,
    beam_center: tuple[float, float] | None = None,
    beam_length: float | None = None,
    beam_height: float | None = None,
) -> Mesh:
    """
    Select mesh backend (gmsh blocked grid or legacy structured O-grid) and
    validate the resulting mesh for Q{poly_order}.
    """
    backend = getattr(ARGS, "mesh_backend", "gmsh")
    if backend == "gmsh":
        mesh, _ = _load_gmsh_mesh(
            mesh_size,
            poly_order,
            mesh_file=getattr(ARGS, "mesh_file", None),
            rebuild=bool(getattr(ARGS, "rebuild_mesh", False)),
            view=bool(getattr(ARGS, "view_gmsh", False)),
            beam_center=beam_center,
            beam_length=beam_length,
            beam_height=beam_height,
        )
    else:
        mesh = build_structured_channel_mesh(mesh_size, poly_order)
    if mesh.element_type != "quad":
        raise RuntimeError(f"Expected a quadrilateral mesh, got {mesh.element_type}.")
    if int(mesh.poly_order) != int(poly_order):
        raise RuntimeError(f"Gmsh mesh order {mesh.poly_order} does not match requested Q{poly_order}.")
    tag_channel_boundaries(mesh, mesh_size)
    return mesh


# -----------------------------------------------------------------------------
# Localized asymmetric refinement around the beam (produces hanging nodes)
# -----------------------------------------------------------------------------
def _quad_corner_indices(p: int) -> tuple[int, int, int, int]:
    """Return (bl, br, tr, tl) local indices in lattice order (eta outer, xi inner)."""
    n = p + 1
    bl = 0
    br = p
    tr = p * n + p
    tl = p * n
    return bl, br, tr, tl


def _refine_element_quads(mesh: Mesh, eid: int, orientation: str, nodes, node_lookup) -> tuple[list[list[int]], list[list[int]]]:
    """
    Split one quad element into 2 children (vertical or horizontal).
    Returns (child_connectivity, child_corners).
    """
    p = mesh.poly_order
    t = np.linspace(-1.0, 1.0, p + 1)
    parent_conn = mesh.elements_connectivity[eid]

    def _parent_node(xi_p: float, eta_p: float) -> int | None:
        ix = np.where(np.isclose(t, xi_p, atol=1e-12))[0]
        iy = np.where(np.isclose(t, eta_p, atol=1e-12))[0]
        if ix.size and iy.size:
            idx = int(iy[0] * (p + 1) + ix[0])
            return int(parent_conn[idx])
        return None

    def _get_node(xi_p: float, eta_p: float) -> int:
        # Reuse parent nodes when possible; otherwise create/reuse global by coordinate.
        nid = _parent_node(xi_p, eta_p)
        if nid is not None:
            return nid
        x_phys = transform.x_mapping(mesh, eid, (float(xi_p), float(eta_p)))
        key = (float(round(x_phys[0], 14)), float(round(x_phys[1], 14)))
        nid = node_lookup.get(key)
        if nid is not None:
            return nid
        nid = len(nodes)
        node_lookup[key] = nid
        nodes.append(Node(nid, float(x_phys[0]), float(x_phys[1])))
        return nid

    def _child(refine_mode: str):
        conn = []
        xi_child = t
        eta_child = t
        for eta in eta_child:
            for xi in xi_child:
                if refine_mode == "left":
                    xi_p = 0.5 * (xi - 1.0)
                    eta_p = eta
                elif refine_mode == "right":
                    xi_p = 0.5 * (xi + 1.0)
                    eta_p = eta
                elif refine_mode == "bottom":
                    xi_p = xi
                    eta_p = 0.5 * (eta - 1.0)
                elif refine_mode == "top":
                    xi_p = xi
                    eta_p = 0.5 * (eta + 1.0)
                else:
                    raise ValueError(refine_mode)
                conn.append(_get_node(xi_p, eta_p))
        bl, br, tr, tl = _quad_corner_indices(p)
        corners = [conn[bl], conn[br], conn[tr], conn[tl]]
        return conn, corners

    if orientation == "vertical":
        c1, corners1 = _child("left")
        c2, corners2 = _child("right")
        return [c1, c2], [corners1, corners2]
    else:
        c1, corners1 = _child("bottom")
        c2, corners2 = _child("top")
        return [c1, c2], [corners1, corners2]


def asymmetric_refine_around_beam(mesh: Mesh, beam_ls: BeamLevelSet, levels: int = 2, band: float | None = None, splits_per_mark: int = 2) -> Mesh:
    """
    Refine quads touching the beam level set with orientation bias:
    left half → vertical split, right half → horizontal split.
    Produces hanging nodes that are handled by the constraint layer.
    """
    if mesh.element_type != "quad":
        return mesh

    band = band or max(3.0 * beam_ls.hy, 4.0 * MESH_SIZE)
    center_x = float(getattr(beam_ls, "cx", BEAM_CENTER[0]))
    beam_xmin = float(beam_ls.cx - beam_ls.hx)
    beam_xmax = float(beam_ls.cx + beam_ls.hx)
    beam_ymin = float(beam_ls.cy - beam_ls.hy)
    beam_ymax = float(beam_ls.cy + beam_ls.hy)

    marked = set()
    t0_mark = time.perf_counter()
    for elem in mesh.elements_list:
        corners = mesh.nodes_x_y_pos[list(elem.corner_nodes)]
        phi_corner = beam_ls(corners)
        hits_phi = np.any(phi_corner <= 0.0) or np.any(np.abs(phi_corner) <= band)
        ex_min, ey_min = corners.min(axis=0)
        ex_max, ey_max = corners.max(axis=0)
        hits_bbox = (ex_min <= beam_xmax + band and ex_max >= beam_xmin - band and ey_min <= beam_ymax + band and ey_max >= beam_ymin - band)
        if hits_phi or hits_bbox:
            marked.add(elem.id)
    print(f"[refine_beam] mark phase: {len(marked)} elems in {time.perf_counter()-t0_mark:.3f}s")
    for _ in range(max(0, levels - 1)):
        new = set()
        for eid in marked:
            for nb in mesh._neighbors[eid]:
                if nb is not None:
                    new.add(int(nb))
        marked |= new
    print(f"[refine_beam] neighbor expansion to {len(marked)} elems (levels={levels})")

    if not marked:
        return mesh

    # Cap splits_per_mark to avoid runaway refinement
    splits_per_mark = max(1, min(int(splits_per_mark), 4))

    nodes = list(mesh.nodes_list)
    node_lookup = {(round(nd.x, 14), round(nd.y, 14)): int(nd.id) for nd in nodes}
    new_elems = []
    new_corners = []

    t0_split = time.perf_counter()
    for eid, elem in enumerate(mesh.elements_list):
        if eid not in marked:
            new_elems.append(list(mesh.elements_connectivity[eid]))
            new_corners.append(list(mesh.corner_connectivity[eid]))
            continue
        cx, _ = elem.centroid()
        orient_primary = "vertical" if cx <= center_x else "horizontal"
        conns_lvl1, corners_lvl1 = _refine_element_quads(mesh, eid, orient_primary, nodes, node_lookup)

        if splits_per_mark <= 1:
            new_elems.extend(conns_lvl1)
            new_corners.extend(corners_lvl1)
            continue

        current = list(zip(conns_lvl1, corners_lvl1))
        for _ in range(1, splits_per_mark):
            next_level = []
            orient_secondary = "horizontal" if orient_primary == "vertical" else "vertical"
            bl, br, tr, tl = _quad_corner_indices(mesh.poly_order)
            for conn_child, _ in current:
                child_corners = [conn_child[bl], conn_child[br], conn_child[tr], conn_child[tl]]
                tmp_mesh = Mesh(
                    nodes=nodes,
                    element_connectivity=np.asarray([conn_child], dtype=int),
                    elements_corner_nodes=np.asarray([child_corners], dtype=int),
                    element_type="quad",
                    poly_order=mesh.poly_order,
                )
                tmp_conn, tmp_corners = _refine_element_quads(tmp_mesh, 0, orient_secondary, nodes, node_lookup)
                next_level.extend(zip(tmp_conn, tmp_corners))
            current = next_level
        for conn_child, corners_child in current:
            new_elems.append(conn_child)
            new_corners.append(corners_child)
    print(f"[refine_beam] split phase produced {len(new_elems)} elems, {len(nodes)} nodes in {time.perf_counter()-t0_split:.3f}s")

    new_mesh = Mesh(
        nodes=nodes,
        element_connectivity=np.asarray(new_elems, dtype=int),
        elements_corner_nodes=np.asarray(new_corners, dtype=int),
        element_type="quad",
        poly_order=mesh.poly_order,
    )
    tag_channel_boundaries(new_mesh, MESH_SIZE)
    print(f"[refine_beam] marked {len(marked)} elems → {len(new_elems)} elements, {len(nodes)} nodes (band={band:.4f})")
    return new_mesh


# -----------------------------------------------------------------------------
# Optimized asymmetric refinement (precomputed grids, no temporary Mesh)
# -----------------------------------------------------------------------------
def _precompute_child_parametric_fast(p: int) -> dict[str, np.ndarray]:
    t = np.linspace(-1.0, 1.0, p + 1)
    grids = {}
    for ref in ("left", "right", "bottom", "top"):
        xi_list = []
        eta_list = []
        for eta in t:
            for xi in t:
                if ref == "left":
                    xi_p = 0.5 * (xi - 1.0)
                    eta_p = eta
                elif ref == "right":
                    xi_p = 0.5 * (xi + 1.0)
                    eta_p = eta
                elif ref == "bottom":
                    xi_p = xi
                    eta_p = 0.5 * (eta - 1.0)
                elif ref == "top":
                    xi_p = xi
                    eta_p = 0.5 * (eta + 1.0)
                xi_list.append(xi_p)
                eta_list.append(eta_p)
        grids[ref] = np.column_stack([np.asarray(xi_list), np.asarray(eta_list)])
    return grids


def _apply_sequence_to_grid(base_grid: np.ndarray, sequence: list[str]) -> np.ndarray:
    xi_eta = base_grid.copy()
    for op in sequence:
        if op == "left":
            xi_eta[:, 0] = 0.5 * (xi_eta[:, 0] - 1.0)
        elif op == "right":
            xi_eta[:, 0] = 0.5 * (xi_eta[:, 0] + 1.0)
        elif op == "bottom":
            xi_eta[:, 1] = 0.5 * (xi_eta[:, 1] - 1.0)
        elif op == "top":
            xi_eta[:, 1] = 0.5 * (xi_eta[:, 1] + 1.0)
    return xi_eta


def _generate_sequences(primary_orientation: str, depth: int) -> list[list[str]]:
    if primary_orientation == "vertical":
        seqs = [["left"], ["right"]]
    else:
        seqs = [["bottom"], ["top"]]
    orient = primary_orientation
    for _ in range(1, depth):
        orient = "horizontal" if orient == "vertical" else "vertical"
        ops = ["bottom", "top"] if orient == "horizontal" else ["left", "right"]
        seqs = [s + [op] for s in seqs for op in ops]
    return seqs


def _parent_parametric_grid(p: int) -> np.ndarray:
    """Full parent reference grid (no split applied)."""
    t = np.linspace(-1.0, 1.0, p + 1)
    xi, eta = np.meshgrid(t, t, indexing="xy")
    return np.column_stack([xi.ravel(), eta.ravel()])


def _grid_key(grid: np.ndarray, ndp: int = 12) -> tuple[tuple[float, float], ...]:
    """Hashable, rounded representation of a parametric grid."""
    return tuple((float(np.round(pt[0], ndp)), float(np.round(pt[1], ndp))) for pt in np.asarray(grid))


@lru_cache(maxsize=None)
def _shape_table_for_grid(element_type: str, poly_order: int, grid_key: tuple[tuple[float, float], ...]) -> np.ndarray:
    """
    Tabulate reference shape functions for all points in `grid_key`.
    Cached so repeated refinement stages reuse the same tables.
    """
    ref = get_reference(element_type, poly_order)
    n_pts = len(grid_key)
    n_loc = len(ref.shape(0.0, 0.0))
    tab = np.empty((n_pts, n_loc), dtype=float)
    for i, (xi, eta) in enumerate(grid_key):
        tab[i, :] = ref.shape(float(xi), float(eta))
    return tab


def _map_grid_to_physical(mesh: Mesh, eid: int, grid: np.ndarray) -> np.ndarray:
    """
    Fast batched parent→physical mapping for all points in `grid` on element `eid`.
    Uses cached reference shape tables to avoid thousands of transform.x_mapping calls.
    """
    key = _grid_key(grid)
    Ntab = _shape_table_for_grid(mesh.element_type, mesh.poly_order, key)
    coords = mesh.nodes_x_y_pos[mesh.elements_connectivity[eid]]
    return Ntab @ coords


def _on_parent_edge(pt: np.ndarray, corners_xy: np.ndarray, tol: float = 1.0e-12) -> bool:
    """Detect whether point lies on any parent edge (straight edges only)."""
    edge_pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
    for i0, i1 in edge_pairs:
        a = corners_xy[i0]
        b = corners_xy[i1]
        ab = b - a
        L2 = float(np.dot(ab, ab))
        if L2 <= tol:
            continue
        cross = abs((pt[0] - a[0]) * ab[1] - (pt[1] - a[1]) * ab[0])
        if cross > tol * max(1.0, math.sqrt(L2)):
            continue
        s = np.dot(pt - a, ab) / L2
        if -tol <= s <= 1.0 + tol:
                return True
    return False


def _refine_element_with_sequences_fast(mesh: Mesh, eid: int, sequences: list[list[str]], nodes, node_lookup, base_grid: np.ndarray, xi_to_idx, bl, br, tr, tl) -> tuple[list[list[int]], list[list[int]]]:
    conns_out: list[list[int]] = []
    corners_out: list[list[int]] = []

    # Cache parametric grids per sequence for this element to avoid repeated transforms.
    seq_cache: dict[tuple[str, ...], np.ndarray] = {}

    for seq in sequences:
        key = tuple(seq)
        grid = seq_cache.get(key)
        if grid is None:
            grid = _apply_sequence_to_grid(base_grid, seq)
            seq_cache[key] = grid
        phys_pts = _map_grid_to_physical(mesh, eid, grid)
        conn = []
        for (xi, eta), p_phys in zip(grid, phys_pts):
            xi_p = float(np.round(xi, 12))
            eta_p = float(np.round(eta, 12))
            key_pe = (xi_p, eta_p)
            idx = xi_to_idx.get(key_pe)
            if idx is not None:
                conn.append(int(mesh.elements_connectivity[eid][idx]))
                continue
            key = (float(round(p_phys[0], 14)), float(round(p_phys[1], 14)))
            nid = node_lookup.get(key)
            if nid is None:
                nid = len(nodes)
                node_lookup[key] = nid
                nodes.append(Node(nid, float(p_phys[0]), float(p_phys[1])))
            conn.append(int(nid))
        corners = [conn[bl], conn[br], conn[tr], conn[tl]]
        conns_out.append(conn)
        corners_out.append(corners)
    return conns_out, corners_out


def asymmetric_refine_around_beam_fast(mesh: Mesh, beam_ls: BeamLevelSet, levels: int = 2, band: float | None = None, splits_per_mark: int = 2) -> Mesh:
    """Optimized refinement around the beam using precomputed param grids."""
    if mesh.element_type != "quad":
        return mesh
    band = band or max(4.0 * beam_ls.hy, 1.5 * MESH_SIZE)
    center_x = float(getattr(beam_ls, "cx", BEAM_CENTER[0]))
    base_splits = max(1, int(splits_per_mark))
    refiner = TensorRefiner()
    bbox_hint = (beam_ls.cx - beam_ls.hx, beam_ls.cx + beam_ls.hx, beam_ls.cy - beam_ls.hy, beam_ls.cy + beam_ls.hy)

    marked = refiner.mark_near_levelset(mesh, beam_ls, band=band, levels=levels, bbox_hint=bbox_hint)
    print(f"[refine_beam_fast] total marked {len(marked)} elems (band={band:.4f}, levels={levels})")
    if not marked:
        return mesh

    nodes = list(mesh.nodes_list)
    node_lookup = {(round(nd.x, 14), round(nd.y, 14)): int(nd.id) for nd in nodes}
    new_elems: list[list[int]] = []
    new_corners: list[list[int]] = []

    p = mesh.poly_order
    t = np.linspace(-1.0, 1.0, p + 1)
    xi_to_idx_template = {(float(xi), float(eta)): int(j * (p + 1) + i) for j, eta in enumerate(t) for i, xi in enumerate(t)}
    bl, br, tr, tl = _quad_corner_indices(p)
    base_grid_parent = _parent_parametric_grid(p)

    t0_split = time.perf_counter()
    for eid, elem in enumerate(mesh.elements_list):
        if eid not in marked:
            new_elems.append(list(mesh.elements_connectivity[eid]))
            new_corners.append(list(mesh.corner_connectivity[eid]))
            continue
        cx, _ = elem.centroid()
        corners = mesh.nodes_x_y_pos[list(elem.corner_nodes)]
        h_y = float(corners[:, 1].max() - corners[:, 1].min())
        h_x = float(corners[:, 0].max() - corners[:, 0].min())

        # target thickness: resolve beam height by at least ~3 layers
        target_h = max(beam_ls.hy / 3.0, 0.25 * MESH_SIZE)

        # Bias orientation when elements are too tall relative to beam thickness.
        if h_y > 1.1 * target_h:
            primary = "horizontal"
        else:
            primary = "vertical" if cx <= center_x else "horizontal"

        ratio = max(h_y / max(target_h, 1e-14), 1.0)
        needed_horiz = int(math.ceil(math.log(ratio, 2))) if ratio > 1.0 else 0
        # horizontals: left of center → occur on even splits; right → on odd splits
        if primary == "vertical":
            min_splits = 2 * needed_horiz if needed_horiz > 0 else base_splits
        else:
            min_splits = max(base_splits, max(1, 2 * needed_horiz - 1) if needed_horiz > 0 else base_splits)
        splits = max(base_splits, min_splits)
        splits = min(splits, 7)  # allow more refinement across beam thickness
        sequences = _generate_sequences(primary, splits)
        conns, corners = _refine_element_with_sequences_fast(
            mesh, eid, sequences, nodes, node_lookup, base_grid_parent, xi_to_idx_template, bl, br, tr, tl
        )
        new_elems.extend(conns)
        new_corners.extend(corners)
    print(f"[refine_beam_fast] split produced {len(new_elems)} elems, {len(nodes)} nodes in {time.perf_counter()-t0_split:.3f}s")

    new_mesh = Mesh(
        nodes=nodes,
        element_connectivity=np.asarray(new_elems, dtype=int),
        elements_corner_nodes=np.asarray(new_corners, dtype=int),
        element_type="quad",
        poly_order=mesh.poly_order,
    )
    tag_channel_boundaries(new_mesh, MESH_SIZE)
    print(f"[refine_beam_fast] final mesh: {len(new_mesh.elements_list)} elems, {len(new_mesh.nodes_list)} nodes")
    return new_mesh


def refine_beam_anisotropic(mesh: Mesh, beam_ls: BeamLevelSet, *, levels: int = 2, target_h: float | None = None) -> Mesh:
    """
    Anisotropic refinement around the beam using a balanced tensor-product split
    (2^nx × 2^ny children per refined parent). We plan nx, ny from element
    sizes relative to the beam height and a small x-band, then enforce a 2:1
    balance so neighbouring sides differ by at most one level.
    """
    if mesh.element_type != "quad":
        return mesh

    refiner = TensorRefiner(max_ratio=2.0, max_ref=6)
    target_h = target_h or float(getattr(beam_ls, "hy", BEAM_HEIGHT * 0.5))
    band = max(2.5 * target_h, 1.2 * getattr(beam_ls, "hy", target_h))
    bbox_hint = (beam_ls.cx - beam_ls.hx, beam_ls.cx + beam_ls.hx, beam_ls.cy - beam_ls.hy, beam_ls.cy + beam_ls.hy)

    marked = refiner.mark_near_levelset(mesh, beam_ls, band=band, levels=max(levels, 1), bbox_hint=bbox_hint)
    if not marked:
        return mesh

    beam_xmin = float(beam_ls.cx - beam_ls.hx)
    beam_xmax = float(beam_ls.cx + beam_ls.hx)
    rx_plan, ry_plan = refiner.plan_tensor_levels(
        mesh,
        marked,
        target_h=target_h,
        span_x=(beam_xmin, beam_xmax),
        target_x=max(target_h, 0.4 * getattr(beam_ls, "hy", target_h)),
        span_x_halo=0.35 * band,
    )
    rx_bal, ry_bal = refiner.balance_levels(mesh, rx_plan, ry_plan)
    active = np.nonzero((rx_bal > 0) | (ry_bal > 0))[0]
    if active.size == 0:
        return mesh

    t0 = time.perf_counter()
    mesh_cur = refiner.refine(mesh, rx_bal, ry_bal)
    tag_channel_boundaries(mesh_cur, MESH_SIZE)
    elapsed = time.perf_counter() - t0
    diag = mesh_topology_diagnostics(mesh_cur)
    print(
        f"[refine_beam_aniso] refined {len(active)} parents → {len(mesh_cur.elements_list)} elems in {elapsed:.3f}s "
        f"(missing_side={diag['missing_side']} ownerless_edges={diag['ownerless_edges']} "
        f"degenerate_shared={diag['degenerate_shared']} t_ratio={diag['t_ratio_violation']})"
    )

    if diag["missing_side"] > 0 or diag["ownerless_edges"] > 0:
        repaired = repair_degenerate_edges(mesh_cur)
        diag = mesh_topology_diagnostics(mesh_cur)
        print(
            f"[refine_beam_aniso] repaired {repaired} edges → "
            f"missing_side={diag['missing_side']} ownerless_edges={diag['ownerless_edges']} "
            f"degenerate_shared={diag['degenerate_shared']} t_ratio={diag['t_ratio_violation']}"
        )

    # Safety net for rare fragmented sides (should be rare after balancing)
    mesh_cur = fix_fragmented_sides(mesh_cur, max_segments=2, max_iters=1)
    cov = coverage_diagnostics(mesh_cur, n_samples=80)
    if cov["gaps_x"]:
        msg = ", ".join(f"{x:.4f}" for x in cov["gaps_x"][:8])
        if len(cov["gaps_x"]) > 8:
            msg += " ..."
        print(f"[refine_beam_aniso] coverage gaps at x≈ {msg}")
    center_gaps = inside_centerline_gaps(mesh_cur, x_end=0.6, n_samples=120, require_tags=False)
    if center_gaps:
        msg = ", ".join(f"{x:.4f}" for x in center_gaps[:8])
        if len(center_gaps) > 8:
            msg += " ..."
        print(f"[refine_beam_aniso] missing inside/cut cells along beam centreline at x≈ {msg}")

    # Enforce inside coverage explicitly if columns are missing.
    mesh_cur = _enforce_inside_columns(mesh_cur, beam_ls, target_h, refiner=refiner, attempts=2)
    return mesh_cur


def _enforce_inside_columns(mesh: Mesh, beam_ls: BeamLevelSet, target_h: float, *, refiner: TensorRefiner, attempts: int = 1) -> Mesh:
    """
    If the beam interior is not fully covered by inside elements, refine
    elements whose centroid lies in the beam box until a full inside column exists.
    """
    mesh_cur = mesh
    for _ in range(attempts):
        cov_inside = beam_inside_coverage(mesh_cur, beam_ls, nx=160, ny=10, inside_only=True)
        missing_x = cov_inside["missing_x"][:24]  # limit scope to reduce blow-up
        if not missing_x:
            break

        y0 = beam_ls.cy - beam_ls.hy - 1e-10
        y1 = beam_ls.cy + beam_ls.hy + 1e-10
        mesh_cur = refiner.ensure_column_coverage(mesh_cur, beam_ls, target_h, missing_x=missing_x, y_interval=(y0, y1))
        tag_channel_boundaries(mesh_cur, MESH_SIZE)
        diag = mesh_topology_diagnostics(mesh_cur)
        if diag["degenerate_shared"] > 0:
            repaired = repair_degenerate_edges(mesh_cur)
            diag = mesh_topology_diagnostics(mesh_cur)
        mesh_cur = fix_fragmented_sides(mesh_cur, max_segments=2, max_iters=1)
        print(
            f"[inside-fix] refined {len(missing_x)} columns for inside coverage → "
            f"{len(mesh_cur.elements_list)} elems (t_ratio={diag['t_ratio_violation']} deg_shared={diag['degenerate_shared']})"
        )
    return mesh_cur


# -----------------------------------------------------------------------------
# Post-refinement clean-up: fix highly fragmented sides (4:1, …)
# -----------------------------------------------------------------------------
def _refine_tjunctions_by_side_global(mesh_in: Mesh, max_segments: int = 2) -> Mesh:
    """
    Refine coarse neighbours to remove >2:1 chains along entire sides.
    Standalone version so we can reuse the logic after the main refinement loop.
    """
    if mesh_in.element_type != "quad":
        return mesh_in

    nodes = list(mesh_in.nodes_list)
    node_lookup = {(round(nd.x, 14), round(nd.y, 14)): int(nd.id) for nd in nodes}
    hanging_nodes: set[int] = set(getattr(mesh_in, "hanging_nodes", []))
    new_elems: list[list[int]] = []
    new_corners: list[list[int]] = []
    parent_to_children: dict[int, list[int]] = {}

    orient_map: dict[int, set[str]] = {}
    for elem in mesh_in.elements_list:
        for lid, side_edges in enumerate(elem.edges_by_side):
            if not side_edges:
                continue
            n_seg = len(side_edges)
            if n_seg <= max_segments:
                continue
            # side index convention: 0 bottom, 1 right, 2 top, 3 left
            if lid in (0, 2):
                orient = "vertical"   # split in x
            else:
                orient = "horizontal" # split in y
            orient_map.setdefault(elem.id, set()).add(orient)

    if not orient_map:
        return mesh_in

    p = mesh_in.poly_order
    t = np.linspace(-1.0, 1.0, p + 1)
    xi_to_idx_template = {(float(xi), float(eta)): int(j * (p + 1) + i) for j, eta in enumerate(t) for i, xi in enumerate(t)}
    bl, br, tr, tl = _quad_corner_indices(p)
    base_grid_parent = _parent_parametric_grid(p)

    for eid, elem in enumerate(mesh_in.elements_list):
        requested = orient_map.get(eid)
        if not requested:
            new_elems.append(list(mesh_in.elements_connectivity[eid]))
            new_corners.append(list(mesh_in.corner_connectivity[eid]))
            parent_to_children[eid] = [len(new_elems) - 1]
            continue

        primary = "vertical" if "vertical" in requested else "horizontal"
        worst_ratio = max(len(side) for side in elem.edges_by_side if side)
        depth_needed = int(math.ceil(math.log(max(worst_ratio / max_segments, 1.0), 2.0)))
        depth = max(1, min(depth_needed, 5))
        sequences = _generate_sequences(primary, depth)

        corners_xy = mesh_in.nodes_x_y_pos[list(elem.corner_nodes)]
        before_nodes = len(nodes)
        conns, corners = _refine_element_with_sequences_fast(
            mesh_in,
            eid,
            sequences,
            nodes,
            node_lookup,
            base_grid_parent,
            xi_to_idx_template,
            bl,
            br,
            tr,
            tl,
        )

        new_node_ids = range(before_nodes, len(nodes))
        for nid in new_node_ids:
            pt = np.array([nodes[nid].x, nodes[nid].y], float)
            if _on_parent_edge(pt, corners_xy):
                hanging_nodes.add(nid)

        idx_children = []
        for conn_child, corners_child in zip(conns, corners):
            idx_children.append(len(new_elems))
            new_elems.append(conn_child)
            new_corners.append(corners_child)
        parent_to_children[eid] = idx_children

    refined = Mesh(
        nodes=nodes,
        element_connectivity=np.asarray(new_elems, dtype=int),
        elements_corner_nodes=np.asarray(new_corners, dtype=int),
        element_type="quad",
        poly_order=mesh_in.poly_order,
    )
    refined.hanging_nodes = sorted(hanging_nodes)
    refined.parent_to_children = parent_to_children
    tag_channel_boundaries(refined, MESH_SIZE)
    return refined


def _refine_tjunctions_edge_global(mesh_in: Mesh, max_ratio: float = 2.0) -> Mesh:
    """
    Refine coarse neighbours on edges that violate the 2:1 T-junction rule
    (based on node counts on each side of the shared edge).
    """
    if mesh_in.element_type != "quad":
        return mesh_in
    nodes = list(mesh_in.nodes_list)
    node_lookup = {(round(nd.x, 14), round(nd.y, 14)): int(nd.id) for nd in nodes}
    hanging_nodes: set[int] = set(getattr(mesh_in, "hanging_nodes", []))
    new_elems: list[list[int]] = []
    new_corners: list[list[int]] = []
    parent_to_children: dict[int, list[int]] = {}

    orient_map: dict[int, set[str]] = {}
    for e in mesh_in.edges_list:
        if e.left is None or e.right is None:
            continue
        l_cnt, r_cnt, shared_cnt = mesh_in._edge_owner_counts(e)
        fine = max(l_cnt, r_cnt)
        coarse = min(l_cnt, r_cnt)
        needs_refine = fine > max_ratio * max(coarse, 1) or shared_cnt < 1
        if not needs_refine:
            continue
        targets = []
        if shared_cnt < 1:
            targets.extend([e.left, e.right])
        else:
            targets.append(e.left if l_cnt < r_cnt else e.right)
        p0, p1 = mesh_in.nodes_x_y_pos[list(e.nodes)]
        dx, dy = p1 - p0
        orient = "vertical" if abs(dx) >= abs(dy) else "horizontal"
        for coarse_eid in targets:
            if coarse_eid is None:
                continue
            orient_map.setdefault(int(coarse_eid), set()).add(orient)

    if not orient_map:
        return mesh_in

    p = mesh_in.poly_order
    t = np.linspace(-1.0, 1.0, p + 1)
    xi_to_idx_template = {(float(xi), float(eta)): int(j * (p + 1) + i) for j, eta in enumerate(t) for i, xi in enumerate(t)}
    bl, br, tr, tl = _quad_corner_indices(p)
    base_grid_parent = _parent_parametric_grid(p)

    for eid, elem in enumerate(mesh_in.elements_list):
        orient_set = orient_map.get(eid)
        if not orient_set:
            new_elems.append(list(mesh_in.elements_connectivity[eid]))
            new_corners.append(list(mesh_in.corner_connectivity[eid]))
            parent_to_children[eid] = [len(new_elems) - 1]
            continue

        primary = "vertical" if "vertical" in orient_set else "horizontal"
        depth = 2 if len(orient_set) > 1 else 1
        sequences = _generate_sequences(primary, depth)

        corners_xy = mesh_in.nodes_x_y_pos[list(elem.corner_nodes)]
        before_nodes = len(nodes)
        conns, corners = _refine_element_with_sequences_fast(
            mesh_in, eid, sequences, nodes, node_lookup, base_grid_parent, xi_to_idx_template, bl, br, tr, tl
        )
        new_node_ids = range(before_nodes, len(nodes))
        for nid in new_node_ids:
            pt = np.array([nodes[nid].x, nodes[nid].y], float)
            if _on_parent_edge(pt, corners_xy):
                hanging_nodes.add(nid)

        idx_children = []
        for conn_child, corners_child in zip(conns, corners):
            idx_children.append(len(new_elems))
            new_elems.append(conn_child)
            new_corners.append(corners_child)
        parent_to_children[eid] = idx_children

    refined = Mesh(
        nodes=nodes,
        element_connectivity=np.asarray(new_elems, dtype=int),
        elements_corner_nodes=np.asarray(new_corners, dtype=int),
        element_type="quad",
        poly_order=mesh_in.poly_order,
    )
    refined.hanging_nodes = sorted(hanging_nodes)
    refined.parent_to_children = parent_to_children
    tag_channel_boundaries(refined, MESH_SIZE)
    return refined


def _refine_side_balance_global(mesh_in: Mesh, max_ratio: float = 2.0) -> Mesh:
    """
    Refine the coarse neighbour along any shared side where the number of edge
    fragments differs by more than `max_ratio` (targets the coarse side of 4:1 gaps).
    """
    if mesh_in.element_type != "quad":
        return mesh_in

    nodes = list(mesh_in.nodes_list)
    node_lookup = {(round(nd.x, 14), round(nd.y, 14)): int(nd.id) for nd in nodes}
    hanging_nodes: set[int] = set(getattr(mesh_in, "hanging_nodes", []))
    new_elems: list[list[int]] = []
    new_corners: list[list[int]] = []
    parent_to_children: dict[int, list[int]] = {}

    orient_map: dict[int, set[str]] = {}
    for elem in mesh_in.elements_list:
        for lid, side_edges in enumerate(elem.edges_by_side):
            if not side_edges:
                continue
            nb = elem.neighbors.get(lid)
            if nb is None:
                continue
            nb_elem = mesh_in.elements_list[int(nb)]
            edge_gid = side_edges[0]
            nb_lid = nb_elem.edge_gid_to_local.get(edge_gid)
            if nb_lid is None:
                continue
            len_this = len(side_edges)
            len_nb = len(nb_elem.edges_by_side[int(nb_lid)])
            fine = max(len_this, len_nb)
            coarse = min(len_this, len_nb)
            if coarse * max_ratio >= fine or coarse == 0:
                continue
            target = elem.id if len_this < len_nb else nb_elem.id
            orient = "vertical" if lid in (0, 2) else "horizontal"
            orient_map.setdefault(int(target), set()).add(orient)

    if not orient_map:
        return mesh_in

    p = mesh_in.poly_order
    t = np.linspace(-1.0, 1.0, p + 1)
    xi_to_idx_template = {(float(xi), float(eta)): int(j * (p + 1) + i) for j, eta in enumerate(t) for i, xi in enumerate(t)}
    bl, br, tr, tl = _quad_corner_indices(p)
    base_grid_parent = _parent_parametric_grid(p)

    for eid, elem in enumerate(mesh_in.elements_list):
        requested = orient_map.get(eid)
        if not requested:
            new_elems.append(list(mesh_in.elements_connectivity[eid]))
            new_corners.append(list(mesh_in.corner_connectivity[eid]))
            parent_to_children[eid] = [len(new_elems) - 1]
            continue

        primary = "vertical" if "vertical" in requested else "horizontal"
        depth = 1 if len(requested) == 1 else 2
        sequences = _generate_sequences(primary, depth)

        corners_xy = mesh_in.nodes_x_y_pos[list(elem.corner_nodes)]
        before_nodes = len(nodes)
        conns, corners = _refine_element_with_sequences_fast(
            mesh_in,
            eid,
            sequences,
            nodes,
            node_lookup,
            base_grid_parent,
            xi_to_idx_template,
            bl,
            br,
            tr,
            tl,
        )
        new_node_ids = range(before_nodes, len(nodes))
        for nid in new_node_ids:
            pt = np.array([nodes[nid].x, nodes[nid].y], float)
            if _on_parent_edge(pt, corners_xy):
                hanging_nodes.add(nid)

        idx_children = []
        for conn_child, corners_child in zip(conns, corners):
            idx_children.append(len(new_elems))
            new_elems.append(conn_child)
            new_corners.append(corners_child)
        parent_to_children[eid] = idx_children

    refined = Mesh(
        nodes=nodes,
        element_connectivity=np.asarray(new_elems, dtype=int),
        elements_corner_nodes=np.asarray(new_corners, dtype=int),
        element_type="quad",
        poly_order=mesh_in.poly_order,
    )
    refined.hanging_nodes = sorted(hanging_nodes)
    refined.parent_to_children = parent_to_children
    tag_channel_boundaries(refined, MESH_SIZE)
    return refined


def fix_fragmented_sides(mesh: Mesh, max_segments: int = 2, max_iters: int = 2) -> Mesh:
    """
    Run a few global passes that split coarse elements until no side has more than
    `max_segments` sub-edges (removes lingering 4:1 T-junctions).
    """
    mesh_cur = mesh
    for it in range(max_iters):
        tj = mesh_cur.count_tjunction_violations(max_ratio=max_segments)
        if tj["count"] == 0 or tj.get("worst_ratio", 0.0) <= max_segments:
            break
        mesh_cur = _refine_side_balance_global(mesh_cur, max_ratio=max_segments)
        mesh_cur = _refine_tjunctions_edge_global(mesh_cur, max_ratio=max_segments)
        mesh_cur = _refine_tjunctions_by_side_global(mesh_cur, max_segments=max_segments)
        tj_next = mesh_cur.count_tjunction_violations(max_ratio=max_segments)
        print(
            f"[tjunction-fix] iter {it + 1}: elems={len(mesh_cur.elements_list)} "
            f"remaining={tj_next['count']} worst_ratio={tj_next.get('worst_ratio', 0.0):.2f}"
        )
        if tj_next["count"] == 0:
            break
    return mesh_cur


# -----------------------------------------------------------------------------
# Level-set update utilities
# -----------------------------------------------------------------------------


def update_beam_levelset_from_displacement(
    ls_beam: LevelSetGridFunction,
    disp_vec: VectorFunction,
    beam_ref_ls: Callable[[np.ndarray], float],
    *,
    nudge_eps: float = 0.0,
    snap_eps: float = 0.0,
    edge_tol: float | None = None,
    prefer_negative: bool = True,
    update_tol: float = 0.0,
) -> bool:
    """
    Advect the beam LevelSetGridFunction with the current solid displacement:
    φ^{k+1}(x_node) = φ_ref(x_node - d^{k+1}(x_node)).
    Returns True if the level set was updated beyond the tolerance.
    """
    dh_ls = ls_beam.dh
    mesh = dh_ls.mixed_element.mesh
    disp_dh = disp_vec._dof_handler

    gphi = np.asarray(dh_ls.get_field_slice(ls_beam.field), dtype=int)
    node_ids = np.array([dh_ls._dof_to_node_map[int(gd)][1] for gd in gphi], dtype=int)

    disp_x = disp_vec.components[0]
    disp_y = disp_vec.components[1]
    phi_vals = ls_beam.nodal_values()
    phi_new_vals = np.empty_like(phi_vals)

    n_nodes = mesh.nodes_x_y_pos.shape[0]
    phi_by_node = np.full(n_nodes, np.nan, dtype=float)
    phi_by_node[node_ids] = phi_vals
    me_disp = disp_dh.mixed_element
    fld_x = disp_x.field_name
    fld_y = disp_y.field_name
    cache = getattr(ls_beam, "_disp_eval_cache", None)
    if (
        cache is None
        or cache.get("mesh_id") != id(mesh)
        or cache.get("n_nodes") != n_nodes
        or cache.get("n_elems") != len(mesh.elements_list)
        or cache.get("fld_x") != fld_x
        or cache.get("fld_y") != fld_y
    ):
        li_x = np.full(n_nodes, -1, dtype=int)
        li_y = np.full(n_nodes, -1, dtype=int)

        dof_map_x = disp_dh.dof_map.get(fld_x, {})
        dof_map_y = disp_dh.dof_map.get(fld_y, {})
        g2l = getattr(disp_vec, "_g2l", {}) or {}

        for nid, gd in dof_map_x.items():
            li = g2l.get(int(gd))
            if li is not None:
                li_x[int(nid)] = int(li)
        for nid, gd in dof_map_y.items():
            li = g2l.get(int(gd))
            if li is not None:
                li_y[int(nid)] = int(li)

        cache = {
            "mesh_id": id(mesh),
            "n_nodes": n_nodes,
            "n_elems": len(mesh.elements_list),
            "fld_x": fld_x,
            "fld_y": fld_y,
            "li_x": li_x,
            "li_y": li_y,
        }
        setattr(ls_beam, "_disp_eval_cache", cache)
    li_x = cache["li_x"]
    li_y = cache["li_y"]
    try:
        inside_eids = mesh.element_bitset("inside").to_indices()
    except Exception:
        inside_eids = np.array([], dtype=int)
    try:
        cut_eids = mesh.element_bitset("cut").to_indices()
    except Exception:
        cut_eids = np.array([], dtype=int)
    if inside_eids.size or cut_eids.size:
        active_eids = np.unique(np.concatenate([inside_eids, cut_eids]).astype(int, copy=False))
        active_nodes = np.unique(mesh.elements_connectivity[active_eids].ravel())
        node_active = np.zeros(n_nodes, dtype=bool)
        node_active[active_nodes.astype(int, copy=False)] = True
    else:
        node_active = np.zeros(n_nodes, dtype=bool)

    disp_eval_tags = {"inside", "cut"}
    phi_disp_cutoff = float(edge_tol) if edge_tol is not None else 0.0
    for gd_phi, nid in zip(gphi, node_ids):
        x, y = mesh.nodes_x_y_pos[int(nid)]
        ux = 0.0
        uy = 0.0
        phi_n = phi_by_node[int(nid)]
        # The solid displacement is only meaningful on the negative side of the
        # interface. Values on the positive side can be unconstrained and may
        # blow up, which would corrupt the level-set update.
        if node_active[int(nid)] and np.isfinite(phi_n) and phi_n <= phi_disp_cutoff:
            lix = int(li_x[int(nid)])
            liy = int(li_y[int(nid)])
            if lix >= 0 and liy >= 0:
                ux = float(disp_vec.nodal_values[lix])
                uy = float(disp_vec.nodal_values[liy])
            else:
                disp_val = _eval_vector_at_point(
                    disp_dh, mesh, disp_vec, (float(x), float(y)), elem_tags=disp_eval_tags
                )
                ux = float(disp_val[0])
                uy = float(disp_val[1])

        X_ref = np.array([x - ux, y - uy], float)
        phi_new = beam_ref_ls(X_ref)

        li = ls_beam._g2l[int(gd_phi)]
        phi_new = float(phi_new)
        phi_new_vals[int(li)] = phi_new
    if nudge_eps > 0.0:
        nudge_mask = np.abs(phi_new_vals) <= nudge_eps
        if np.any(nudge_mask):
            phi_new_vals[nudge_mask] = -nudge_eps if prefer_negative else nudge_eps

    if snap_eps > 0.0:
        snap_mask = np.abs(phi_new_vals) <= snap_eps
        if np.any(snap_mask):
            phi_new_vals[snap_mask] = 0.0

    max_diff = float(np.max(np.abs(phi_new_vals - phi_vals))) if phi_vals.size else 0.0
    phi_diff = np.abs(phi_new_vals - phi_vals) if phi_vals.size else np.array([], dtype=float)
    if max_diff <= update_tol:
        ls_beam._last_update_stats = {
            "changed": False,
            "max_diff": max_diff,
            "max_diff_cut": 0.0,
            "cut_elems": 0,
            "interface_edges": 0,
        }
        return False

    phi_vals[:] = phi_new_vals
    tol_commit = edge_tol if edge_tol is not None else LEVELSET_EDGE_TOL
    ls_beam.commit(tol=tol_commit)
    if edge_tol is not None:
        mesh.classify_edges(ls_beam, tol=edge_tol)
        mesh.build_interface_segments(ls_beam, tol=edge_tol)

    cut_ids = mesh.element_bitset("cut").to_indices()
    interface_edges = mesh.edge_bitset("interface").to_indices()
    max_diff_cut = 0.0
    if cut_ids.size and phi_diff.size:
        diff_by_node = np.full(n_nodes, np.nan, dtype=float)
        diff_by_node[node_ids] = phi_diff
        cut_nodes = np.unique(np.concatenate([mesh.elements_connectivity[int(eid)] for eid in cut_ids]))
        if cut_nodes.size:
            cut_vals = diff_by_node[cut_nodes]
            if np.any(np.isfinite(cut_vals)):
                max_diff_cut = float(np.nanmax(cut_vals))
    ls_beam._last_update_stats = {
        "changed": True,
        "max_diff": max_diff,
        "max_diff_cut": max_diff_cut,
        "cut_elems": int(cut_ids.size),
        "interface_edges": int(interface_edges.size),
    }
    return True


def _zero_inactive_side_values(
    *,
    ls_beam: LevelSetGridFunction,
    dof_handler: DofHandler,
    mesh: Mesh,
    uf: list[VectorFunction],
    pf: list[Function],
    us: list[VectorFunction],
    disp: list[VectorFunction],
    tol: float,
) -> None:
    """
    Enforce the physical domain extension by zero:
    - Fluid unknowns (u,p) are meaningful only on the '+' side (phi>0).
    - Solid unknowns (vs,d) are meaningful only on the '-' side (phi<=0).

    This prevents unconstrained/inactive DOFs on the opposite side from
    drifting to huge values (which can poison debugging and the level-set update).
    """
    cache = getattr(ls_beam, "_side_zero_cache", None)
    if (
        cache is None
        or cache.get("mesh_id") != id(mesh)
        or cache.get("n_nodes") != int(mesh.nodes_x_y_pos.shape[0])
    ):
        field_node_ids: dict[str, np.ndarray] = {}
        for fld in [
            "u_pos_x",
            "u_pos_y",
            "p_pos_",
            "vs_neg_x",
            "vs_neg_y",
            "d_neg_x",
            "d_neg_y",
        ]:
            gdofs = np.asarray(dof_handler.get_field_slice(fld), dtype=int)
            node_ids = np.array([dof_handler._dof_to_node_map[int(gd)][1] for gd in gdofs], dtype=int)
            field_node_ids[fld] = node_ids
        cache = {
            "mesh_id": id(mesh),
            "n_nodes": int(mesh.nodes_x_y_pos.shape[0]),
            "field_node_ids": field_node_ids,
            "vec_ranges": {},
            "phi_token": None,
            "phi_tol": None,
            "solid_nodes": None,
            "fluid_nodes": None,
        }
        setattr(ls_beam, "_side_zero_cache", cache)

    field_node_ids = cache["field_node_ids"]
    vec_ranges = cache["vec_ranges"]
    tol = float(tol)
    phi_token = getattr(ls_beam, "cache_token", None)
    if cache.get("phi_token") != phi_token or cache.get("phi_tol") != tol:
        phi_nodes = ls_beam.evaluate_on_nodes(mesh)
        solid_nodes = np.asarray(phi_nodes <= tol, dtype=bool)
        cache["phi_token"] = phi_token
        cache["phi_tol"] = tol
        cache["solid_nodes"] = solid_nodes
        cache["fluid_nodes"] = ~solid_nodes
    solid_nodes = np.asarray(cache.get("solid_nodes"), dtype=bool)
    fluid_nodes = np.asarray(cache.get("fluid_nodes"), dtype=bool)

    def _field_ranges_for(vf: VectorFunction) -> dict[str, tuple[int, int]]:
        key = tuple(vf.field_names)
        ranges = vec_ranges.get(key)
        if ranges is not None:
            return ranges
        off = 0
        out: dict[str, tuple[int, int]] = {}
        for fld in vf.field_names:
            n = int(np.asarray(dof_handler.get_field_slice(fld), dtype=int).size)
            out[fld] = (off, off + n)
            off += n
        vec_ranges[key] = out
        return out

    def _zero_vec(vf: VectorFunction, fld: str, node_mask: np.ndarray) -> None:
        start, stop = _field_ranges_for(vf)[fld]
        seg = vf.nodal_values[start:stop]
        ids = field_node_ids[fld]
        m = node_mask[ids]
        if np.any(m):
            seg[m] = 0.0

    def _zero_scalar(f: Function, fld: str, node_mask: np.ndarray) -> None:
        ids = field_node_ids[fld]
        m = node_mask[ids]
        if np.any(m):
            f.nodal_values[m] = 0.0

    # Fluid: zero on solid side
    for vf in uf:
        _zero_vec(vf, "u_pos_x", solid_nodes)
        _zero_vec(vf, "u_pos_y", solid_nodes)
    for f in pf:
        _zero_scalar(f, "p_pos_", solid_nodes)

    # Solid: zero on fluid side
    for vf in us:
        _zero_vec(vf, "vs_neg_x", fluid_nodes)
        _zero_vec(vf, "vs_neg_y", fluid_nodes)
    for vf in disp:
        _zero_vec(vf, "d_neg_x", fluid_nodes)
        _zero_vec(vf, "d_neg_y", fluid_nodes)


def _stitch_fluid_velocity_to_solid_near_interface(
    *,
    mesh: Mesh,
    dof_handler: DofHandler,
    ls_beam: LevelSetGridFunction,
    uf: list[VectorFunction],
    us: list[VectorFunction],
    band: float,
    only_cut_nodes: bool = True,
) -> int:
    """
    Initialize the *fluid* velocity extension near Γ by copying the solid velocity
    extension at the same nodes. This improves Newton robustness after sudden
    re-tagging (cut ↔ interface set changes) by reducing the Nitsche penalty jump
    in the initial guess.

    Note: this is a *guess* conditioning step; the converged solution is still
    determined by the monolithic Newton solve.
    """
    band = float(band)
    if band <= 0.0:
        return 0

    cache = getattr(ls_beam, "_ifc_stitch_cache", None)
    if cache is None or cache.get("mesh_id") != id(mesh):
        n_nodes = int(mesh.nodes_x_y_pos.shape[0])
        field_to_node = {}
        field_to_gdofs = {}
        for fld in ("u_pos_x", "u_pos_y", "vs_neg_x", "vs_neg_y"):
            gdofs = np.asarray(dof_handler.get_field_slice(fld), dtype=int)
            node_ids = np.array([dof_handler._dof_to_node_map[int(gd)][1] for gd in gdofs], dtype=int)
            field_to_node[fld] = node_ids
            field_to_gdofs[fld] = gdofs

        # node -> solid gdof maps (fast lookup by node id)
        vsx_at_node = -np.ones(n_nodes, dtype=int)
        vsy_at_node = -np.ones(n_nodes, dtype=int)
        vsx_at_node[field_to_node["vs_neg_x"]] = field_to_gdofs["vs_neg_x"]
        vsy_at_node[field_to_node["vs_neg_y"]] = field_to_gdofs["vs_neg_y"]

        cache = {
            "mesh_id": id(mesh),
            "n_nodes": n_nodes,
            "field_to_node": field_to_node,
            "field_to_gdofs": field_to_gdofs,
            "vsx_at_node": vsx_at_node,
            "vsy_at_node": vsy_at_node,
            "cut_nodes_mask": None,
        }
        setattr(ls_beam, "_ifc_stitch_cache", cache)

    n_nodes = int(cache["n_nodes"])
    try:
        phi_nodes = np.asarray(ls_beam.evaluate_on_nodes(mesh), dtype=float)
    except Exception:
        try:
            phi_nodes = np.asarray(ls_beam.nodal_values(), dtype=float)
        except Exception:
            phi_nodes = np.zeros(n_nodes, dtype=float)
    if int(phi_nodes.size) != n_nodes:
        phi_nodes = np.resize(phi_nodes, n_nodes)
    near = np.abs(phi_nodes) <= band

    if only_cut_nodes:
        cut_nodes_mask = cache.get("cut_nodes_mask")
        if cut_nodes_mask is None:
            cut_eids = np.nonzero(mesh.element_bitset("cut").mask)[0]
            cut_nodes_mask = np.zeros(n_nodes, dtype=bool)
            if cut_eids.size:
                cut_nodes_mask[np.asarray(mesh.elements_connectivity[cut_eids].ravel(), dtype=int)] = True
            cache["cut_nodes_mask"] = cut_nodes_mask
        near = near & np.asarray(cut_nodes_mask, dtype=bool)

    if not np.any(near):
        return 0

    field_to_node = cache["field_to_node"]
    field_to_gdofs = cache["field_to_gdofs"]
    vsx_at_node = np.asarray(cache["vsx_at_node"], dtype=int)
    vsy_at_node = np.asarray(cache["vsy_at_node"], dtype=int)

    # Select u-pos gdofs by node mask, then map to corresponding solid gdofs.
    u_nodes_x = np.asarray(field_to_node["u_pos_x"], dtype=int)
    u_nodes_y = np.asarray(field_to_node["u_pos_y"], dtype=int)
    u_gdofs_x = np.asarray(field_to_gdofs["u_pos_x"], dtype=int)
    u_gdofs_y = np.asarray(field_to_gdofs["u_pos_y"], dtype=int)

    sel_x = np.nonzero(near[u_nodes_x])[0]
    sel_y = np.nonzero(near[u_nodes_y])[0]
    if sel_x.size == 0 and sel_y.size == 0:
        return 0

    n_stitched = 0
    for uf_v in uf:
        for us_v in us:
            # X component
            if sel_x.size:
                nodes = u_nodes_x[sel_x]
                u_gd = u_gdofs_x[sel_x]
                s_gd = vsx_at_node[nodes]
                ok = s_gd >= 0
                if np.any(ok):
                    vals = us_v.get_nodal_values(s_gd[ok])
                    uf_v.set_nodal_values(u_gd[ok], vals)
                    n_stitched = max(n_stitched, int(u_gd[ok].size))
            # Y component
            if sel_y.size:
                nodes = u_nodes_y[sel_y]
                u_gd = u_gdofs_y[sel_y]
                s_gd = vsy_at_node[nodes]
                ok = s_gd >= 0
                if np.any(ok):
                    vals = us_v.get_nodal_values(s_gd[ok])
                    uf_v.set_nodal_values(u_gd[ok], vals)
                    n_stitched = max(n_stitched, int(u_gd[ok].size))
            # Only need one solid vector for values (same field slices).
            break
    return int(n_stitched)


def _reextend_wrong_side_by_nearest(
    *,
    mesh: Mesh,
    dof_handler: DofHandler,
    ls_beam: LevelSetGridFunction,
    uf: list[VectorFunction],
    pf: list[Function],
    us: list[VectorFunction],
    disp: list[VectorFunction],
    tol: float,
    only_cut_nodes: bool = True,
    trace: bool = False,
) -> dict[str, int]:
    """
    Re-initialize extension values on the *wrong* side of the level set by copying
    from the nearest node on the correct side (within the cut band, by default).

    This is a robustness tool for sudden re-tagging events: it prevents stale
    (and potentially huge) extension values from poisoning the interface Nitsche
    penalty when the cut/interface sets change.
    """
    from scipy.spatial import cKDTree  # local import to keep base deps light

    tol = float(tol)
    n_nodes = int(mesh.nodes_x_y_pos.shape[0])
    try:
        phi = np.asarray(ls_beam.evaluate_on_nodes(mesh), dtype=float)
    except Exception:
        try:
            phi = np.asarray(ls_beam.nodal_values(), dtype=float)
        except Exception:
            phi = np.zeros(n_nodes, dtype=float)
    if int(phi.size) != n_nodes:
        phi = np.resize(phi, n_nodes)

    cut_mask = np.ones(n_nodes, dtype=bool)
    if only_cut_nodes:
        cut_eids = np.nonzero(mesh.element_bitset("cut").mask)[0]
        cut_mask = np.zeros(n_nodes, dtype=bool)
        if cut_eids.size:
            cut_mask[np.asarray(mesh.elements_connectivity[cut_eids].ravel(), dtype=int)] = True

    coords = np.asarray(mesh.nodes_x_y_pos, dtype=float)
    # Fluid fields live on '+' (phi > tol)
    fluid_ok = (phi > tol) & cut_mask
    fluid_bad = (phi <= tol) & cut_mask
    # Solid fields live on '-' (phi <= tol)
    solid_ok = (phi <= tol) & cut_mask
    solid_bad = (phi > tol) & cut_mask

    stats = {"u_pos": 0, "p_pos": 0, "vs_neg": 0, "d_neg": 0}

    def _node_to_gdof(field: str) -> np.ndarray:
        gdofs = np.asarray(dof_handler.get_field_slice(field), dtype=int)
        node_ids = np.array([dof_handler._dof_to_node_map[int(gd)][1] for gd in gdofs], dtype=int)
        out = -np.ones(n_nodes, dtype=int)
        out[node_ids] = gdofs
        return out

    # Precompute nearest-neighbour maps once (fluid->fluid, solid->solid)
    def _nearest(src_mask: np.ndarray, dst_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        src = np.nonzero(src_mask)[0]
        dst = np.nonzero(dst_mask)[0]
        if src.size == 0 or dst.size == 0:
            return src, np.empty(0, dtype=int)
        tree = cKDTree(coords[dst])
        _, j = tree.query(coords[src], k=1)
        return src, dst[np.asarray(j, dtype=int)]

    fluid_src, fluid_nn = _nearest(fluid_bad, fluid_ok)
    solid_src, solid_nn = _nearest(solid_bad, solid_ok)

    if trace:
        print(
            f"[reextend] cut_nodes={int(cut_mask.sum())} "
            f"fluid_bad={int(fluid_src.size)} solid_bad={int(solid_src.size)} tol={tol:.3e}"
        )

    if fluid_src.size:
        u_px = _node_to_gdof("u_pos_x")
        u_py = _node_to_gdof("u_pos_y")
        p_p = _node_to_gdof("p_pos_")
        gd_u_x = u_px[fluid_src]
        gd_u_y = u_py[fluid_src]
        gd_p = p_p[fluid_src]
        gd_u_x_nn = u_px[fluid_nn]
        gd_u_y_nn = u_py[fluid_nn]
        gd_p_nn = p_p[fluid_nn]
        ok_x = (gd_u_x >= 0) & (gd_u_x_nn >= 0)
        ok_y = (gd_u_y >= 0) & (gd_u_y_nn >= 0)
        ok_p = (gd_p >= 0) & (gd_p_nn >= 0)
        for vf in uf:
            if np.any(ok_x):
                vf.set_nodal_values(gd_u_x[ok_x], vf.get_nodal_values(gd_u_x_nn[ok_x]))
                stats["u_pos"] = max(stats["u_pos"], int(gd_u_x[ok_x].size))
            if np.any(ok_y):
                vf.set_nodal_values(gd_u_y[ok_y], vf.get_nodal_values(gd_u_y_nn[ok_y]))
                stats["u_pos"] = max(stats["u_pos"], int(gd_u_y[ok_y].size))
        for f in pf:
            if np.any(ok_p):
                # Scalar Function has same API as VectorFunction for set_nodal_values/get_nodal_values.
                f.set_nodal_values(gd_p[ok_p], f.get_nodal_values(gd_p_nn[ok_p]))
                stats["p_pos"] = max(stats["p_pos"], int(gd_p[ok_p].size))

    if solid_src.size:
        vsx = _node_to_gdof("vs_neg_x")
        vsy = _node_to_gdof("vs_neg_y")
        dx_ = _node_to_gdof("d_neg_x")
        dy_ = _node_to_gdof("d_neg_y")
        gd_vx = vsx[solid_src]
        gd_vy = vsy[solid_src]
        gd_dx = dx_[solid_src]
        gd_dy = dy_[solid_src]
        gd_vx_nn = vsx[solid_nn]
        gd_vy_nn = vsy[solid_nn]
        gd_dx_nn = dx_[solid_nn]
        gd_dy_nn = dy_[solid_nn]
        ok_vx = (gd_vx >= 0) & (gd_vx_nn >= 0)
        ok_vy = (gd_vy >= 0) & (gd_vy_nn >= 0)
        ok_dx = (gd_dx >= 0) & (gd_dx_nn >= 0)
        ok_dy = (gd_dy >= 0) & (gd_dy_nn >= 0)
        for vf in us:
            if np.any(ok_vx):
                vf.set_nodal_values(gd_vx[ok_vx], vf.get_nodal_values(gd_vx_nn[ok_vx]))
                stats["vs_neg"] = max(stats["vs_neg"], int(gd_vx[ok_vx].size))
            if np.any(ok_vy):
                vf.set_nodal_values(gd_vy[ok_vy], vf.get_nodal_values(gd_vy_nn[ok_vy]))
                stats["vs_neg"] = max(stats["vs_neg"], int(gd_vy[ok_vy].size))
        for vf in disp:
            if np.any(ok_dx):
                vf.set_nodal_values(gd_dx[ok_dx], vf.get_nodal_values(gd_dx_nn[ok_dx]))
                stats["d_neg"] = max(stats["d_neg"], int(gd_dx[ok_dx].size))
            if np.any(ok_dy):
                vf.set_nodal_values(gd_dy[ok_dy], vf.get_nodal_values(gd_dy_nn[ok_dy]))
                stats["d_neg"] = max(stats["d_neg"], int(gd_dy[ok_dy].size))

    return stats


def _copy_bitset(bs: BitSet) -> BitSet:
    return BitSet(np.array(bs.mask, dtype=bool))

def _ghost_band_edges(mesh: Mesh, seed: BitSet, *, layers: int = 2) -> BitSet:
    layers = max(int(layers), 0)
    if seed.cardinality() == 0:
        return BitSet(np.zeros(len(mesh.edges_list), dtype=bool))

    seed_ids = seed.to_indices()
    band: set[int] = set(map(int, seed_ids))
    frontier: set[int] = set(band)
    nbs = mesh.neighbors()

    for _ in range(layers):
        if not frontier:
            break
        nxt: set[int] = set()
        for eid in frontier:
            for nb in nbs[int(eid)]:
                nb = int(nb)
                if nb not in band:
                    nxt.add(nb)
        band.update(nxt)
        frontier = nxt

    mask = np.zeros(len(mesh.edges_list), dtype=bool)
    for e in mesh.edges_list:
        if e.right is None:
            continue
        if int(e.left) in band and int(e.right) in band:
            mask[int(e.gid)] = True

    interface_bs = mesh.edge_bitset("interface")
    return BitSet(mask) - interface_bs


def _fluid_quad_edge_mask(mesh: Mesh) -> np.ndarray:
    """
    Edges between two *outside* (quadrilateral) elements (Richter: E_h^0 subset).
    Used for pressure stabilization on the uncut fluid mesh.
    """
    mask = np.zeros(len(mesh.edges_list), dtype=bool)
    for edge in mesh.edges_list:
        if edge.right is None:
            continue
        lt = getattr(mesh.elements_list[int(edge.left)], "tag", "")
        rt = getattr(mesh.elements_list[int(edge.right)], "tag", "")
        if lt == "outside" and rt == "outside":
            gid = int(getattr(edge, "gid", getattr(edge, "id", 0)))
            if 0 <= gid < mask.size:
                mask[gid] = True
    # Safety: exclude aligned interface edges (should already be absent here).
    try:
        iface = np.asarray(mesh.edge_bitset("interface").mask, dtype=bool)
        if iface.shape == mask.shape:
            mask &= ~iface
    except Exception:
        pass
    return mask

def _disable_aligned_interface_edges(mesh: Mesh) -> int:
    """Disable aligned-interface edge tags to force cut-cell interface integration."""
    interface_edges = [e for e in mesh.edges_list if getattr(e, "tag", "") == "interface"]
    if not interface_edges:
        return 0
    for e in interface_edges:
        e.tag = ""
    mesh.rebuild_edge_bitsets()
    return len(interface_edges)

def _aligned_cut_mask(mesh: Mesh) -> np.ndarray:
    """Flag cut elements whose interface lies exactly on an element edge."""
    try:
        interface_edges = mesh.edge_bitset("interface").to_indices()
    except Exception:
        interface_edges = np.array([], dtype=int)
    if interface_edges.size == 0:
        return np.zeros(len(mesh.elements_list), dtype=bool)
    interface_set = set(map(int, interface_edges))
    aligned = np.zeros(len(mesh.elements_list), dtype=bool)
    for elem in mesh.elements_list:
        if getattr(elem, "tag", "") != "cut":
            continue
        for side_edges in elem.edges_by_side:
            for gid in side_edges:
                if int(gid) in interface_set:
                    aligned[int(elem.id)] = True
                    break
            if aligned[int(elem.id)]:
                break
    return aligned


def make_domain_sets(mesh: Mesh) -> Dict[str, BitSet]:
    fluid = mesh.element_bitset("outside")
    solid = mesh.element_bitset("inside")
    cut = mesh.element_bitset("cut")
    if USE_ALIGNED_INTERFACE and mesh.edge_bitset("interface").cardinality() > 0:
        aligned_cut = _aligned_cut_mask(mesh)
        cut_interface = BitSet(cut.mask & ~aligned_cut)
    else:
        cut_interface = _copy_bitset(cut)
    # These are not interface integrals; they are volume integrals
    fluid_ifc = fluid | cut
    solid_ifc = solid | cut
    has_pos = fluid | cut
    has_neg = solid | cut

    interface_bs = mesh.edge_bitset("interface")
    ghost_pos = mesh.edge_bitset("ghost_pos") - interface_bs
    ghost_neg = mesh.edge_bitset("ghost_neg") - interface_bs
    ghost_both = mesh.edge_bitset("ghost_both") - interface_bs
    ghost_all = (ghost_pos | ghost_neg | ghost_both) - interface_bs

    solid_ghost = ghost_neg | ghost_both
    fluid_ghost = ghost_pos | ghost_both

    # Sliver/corner robustness:
    # When one side has no fully-inside/outside elements, stabilize on an element
    # band around cut cells (includes interior edges beyond immediate cut-neighbors).
    if cut.cardinality() > 0:
        if fluid.cardinality() == 0 or solid.cardinality() == 0:
            band_ghost = _ghost_band_edges(mesh, cut, layers=2)
            if band_ghost.cardinality() > 0:
                fluid_ghost = _copy_bitset(band_ghost)
                solid_ghost = _copy_bitset(band_ghost)
        else:
            if fluid_ghost.cardinality() == 0 and ghost_all.cardinality() > 0:
                fluid_ghost = _copy_bitset(ghost_all)
            if solid_ghost.cardinality() == 0 and ghost_all.cardinality() > 0:
                solid_ghost = _copy_bitset(ghost_all)
    fluid_quad_edges = BitSet(_fluid_quad_edge_mask(mesh))
    return {
        "fluid_domain": _copy_bitset(fluid),
        "solid_domain": _copy_bitset(solid),
        "cut_domain": _copy_bitset(cut),
        "cut_interface": _copy_bitset(cut_interface),
        "fluid_interface": _copy_bitset(fluid_ifc),
        "solid_interface": _copy_bitset(solid_ifc),
        "has_pos": _copy_bitset(has_pos),
        "has_neg": _copy_bitset(has_neg),
        "solid_ghost": _copy_bitset(solid_ghost),
        "fluid_ghost": _copy_bitset(fluid_ghost),
        "fluid_quad_edges": _copy_bitset(fluid_quad_edges),
    }


def _update_bs(target: BitSet, new_mask: np.ndarray) -> None:
    target.mask[...] = new_mask.astype(bool)


def refresh_domains(mesh: Mesh, domains: Dict[str, BitSet]) -> None:
    fluid = mesh.element_bitset("outside")
    solid = mesh.element_bitset("inside")
    cut = mesh.element_bitset("cut")
    has_fluid = fluid | cut
    has_solid = solid | cut
    if USE_ALIGNED_INTERFACE and mesh.edge_bitset("interface").cardinality() > 0:
        aligned_cut = _aligned_cut_mask(mesh)
        cut_interface = cut.mask & ~aligned_cut
    else:
        cut_interface = cut.mask
    _update_bs(domains["fluid_domain"], fluid.mask)
    _update_bs(domains["solid_domain"], solid.mask)
    _update_bs(domains["cut_domain"], cut.mask)
    _update_bs(domains["cut_interface"], cut_interface)
    _update_bs(domains["fluid_interface"], has_fluid.mask)
    _update_bs(domains["solid_interface"], has_solid.mask)
    _update_bs(domains["has_pos"], fluid.mask | cut.mask)
    _update_bs(domains["has_neg"], solid.mask | cut.mask)
    interface_bs = mesh.edge_bitset("interface")
    ghost_pos = mesh.edge_bitset("ghost_pos") - interface_bs
    ghost_neg = mesh.edge_bitset("ghost_neg") - interface_bs
    ghost_both = mesh.edge_bitset("ghost_both") - interface_bs
    ghost_all = (ghost_pos | ghost_neg | ghost_both) - interface_bs

    solid_ghost = ghost_neg | ghost_both
    fluid_ghost = ghost_pos | ghost_both
    if cut.cardinality() > 0:
        if fluid.cardinality() == 0 or solid.cardinality() == 0:
            band_ghost = _ghost_band_edges(mesh, mesh.element_bitset("cut"), layers=2)
            if band_ghost.cardinality() > 0:
                fluid_ghost = _copy_bitset(band_ghost)
                solid_ghost = _copy_bitset(band_ghost)
        else:
            if fluid_ghost.cardinality() == 0 and ghost_all.cardinality() > 0:
                fluid_ghost = _copy_bitset(ghost_all)
            if solid_ghost.cardinality() == 0 and ghost_all.cardinality() > 0:
                solid_ghost = _copy_bitset(ghost_all)
    _update_bs(domains["solid_ghost"], solid_ghost.mask)
    _update_bs(domains["fluid_ghost"], fluid_ghost.mask)
    if "fluid_quad_edges" in domains:
        _update_bs(domains["fluid_quad_edges"], _fluid_quad_edge_mask(mesh))


def refresh_hansbo_kappa(
    mesh: Mesh,
    level_set,
    theta_pos_vals: np.ndarray,
    theta_neg_vals: np.ndarray,
    theta_pos_raw_vals: np.ndarray | None = None,
    theta_neg_raw_vals: np.ndarray | None = None,
    theta_min: float = 1.0e-12,
) -> None:
    # IMPORTANT:
    # - `theta_*_vals` are clipped (>=theta_min) for stable kappa/weight computations.
    # - `theta_*_raw_vals` retain the true (possibly ~0) cut ratio, which is used for
    #   sliver DOF dropping decisions (SOLID_CUT_DROP) to avoid masking tiny slivers
    #   by the clipping floor.
    thp_raw = hansbo_cut_ratio(mesh, level_set, side="+")
    thn_raw = hansbo_cut_ratio(mesh, level_set, side="-")
    if theta_pos_raw_vals is not None:
        theta_pos_raw_vals[:] = thp_raw
        thp_raw = theta_pos_raw_vals
    if theta_neg_raw_vals is not None:
        theta_neg_raw_vals[:] = thn_raw
        thn_raw = theta_neg_raw_vals
    theta_pos_vals[:] = np.clip(thp_raw, theta_min, 1.0)
    theta_neg_vals[:] = np.clip(thn_raw, theta_min, 1.0)


def log_sliver_stats(
    mesh: Mesh,
    theta_pos_vals: np.ndarray,
    theta_neg_vals: np.ndarray,
    w_pos_vals: np.ndarray,
    w_neg_vals: np.ndarray,
    *,
    theta0: float,
) -> None:
    cut_ids = mesh.element_bitset("cut").to_indices()
    if cut_ids.size == 0:
        print("[sliver] no cut elements (weights stay at 1.0)")
        return
    thp = theta_pos_vals[cut_ids]
    thn = theta_neg_vals[cut_ids]
    wp = w_pos_vals[cut_ids]
    wn = w_neg_vals[cut_ids]
    print(
        "[sliver] min θ+ = {:.3e}  min θ- = {:.3e}  max w+ = {:.3e}  max w- = {:.3e}  "
        "#(θ+<θ0) = {}  #(θ-<θ0) = {}".format(
            float(thp.min()),
            float(thn.min()),
            float(wp.max()),
            float(wn.max()),
            int((thp < theta0).sum()),
            int((thn < theta0).sum()),
        )
    )


def mesh_topology_diagnostics(mesh: Mesh) -> dict[str, int]:
    """
    Quick consistency checks:
      - Every element side has at least one edge (no gaps).
      - Every interior edge has both owners.
      - Edge owners share nodes (edge not degenerate). Allow single-node
        intersections to support T-junction/hanging configurations.
      - Optional 2:1 balance sanity: flag edges where one side has many more
        subdivisions than its neighbour (T-junction ratio > 2).
    Returns counts of issues to guide refinement/debugging.
    """
    missing_side = 0
    boundary_edges = 0
    ownerless_edges = 0
    degenerate_shared = 0
    t_ratio_violation = 0
    t_junctions = 0
    for elem in mesh.elements_list:
        for side_edges in elem.edges_by_side:
            if not side_edges:
                missing_side += 1
        for gid in elem.edges:
            if gid == -1:
                continue
            e = mesh.edges_list[int(gid)]
            if e.right is None:
                boundary_edges += 1
    for e in mesh.edges_list:
        if e.right is None:
            continue
        if e.left is None:
            ownerless_edges += 1
            continue
        left_nodes = set(e.left_nodes)
        right_nodes = set(e.right_nodes)
        shared = left_nodes.intersection(right_nodes)
        if len(shared) < 1:
            degenerate_shared += 1
        if len(left_nodes) != len(right_nodes):
            t_junctions += 1
        l_count = max(len(left_nodes), 1)
        r_count = max(len(right_nodes), 1)
        coarse = min(l_count, r_count)
        fine = max(l_count, r_count)
        if coarse > 0 and fine > 2 * coarse:
            t_ratio_violation += 1
    return {
        "missing_side": int(missing_side),
        "boundary_edges": int(boundary_edges),
        "ownerless_edges": int(ownerless_edges),
        "degenerate_shared": int(degenerate_shared),
        "t_junctions": int(t_junctions),
        "t_ratio_violation": int(t_ratio_violation),
    }


def coverage_diagnostics(mesh: Mesh, n_samples: int = 40, tol: float = 1.0e-6) -> dict[str, list[float]]:
    """
    Sample vertical lines across the domain and report x-locations
    where the union of element y-intervals leaves a gap (potential holes).
    Ignores the rigid cylinder hole.
    """
    xs = np.linspace(mesh.nodes_x_y_pos[:, 0].min(), mesh.nodes_x_y_pos[:, 0].max(), n_samples)
    gaps = []
    for x0 in xs:
        # Skip the circle hole on purpose
        if abs(x0 - CENTER[0]) <= RADIUS + tol:
            continue
        intervals = []
        for e in mesh.elements_list:
            cn = mesh.nodes_x_y_pos[list(e.corner_nodes)]
            xmin, xmax = cn[:, 0].min(), cn[:, 0].max()
            if xmin - tol <= x0 <= xmax + tol:
                ymin, ymax = cn[:, 1].min(), cn[:, 1].max()
                intervals.append((ymin, ymax))
        if not intervals:
            gaps.append(float(x0))
            continue
        intervals.sort()
        y0 = intervals[0][0]
        y1 = intervals[0][1]
        for a, b in intervals[1:]:
            if a > y1 + tol:
                gaps.append(float(x0))
                break
            y1 = max(y1, b)
        if y0 > mesh.nodes_x_y_pos[:, 1].min() + tol or y1 < mesh.nodes_x_y_pos[:, 1].max() - tol:
            gaps.append(float(x0))
    return {"gaps_x": gaps}


def inside_centerline_gaps(mesh: Mesh, x_end: float = 0.6, n_samples: int = 80, tol: float = 1.0e-6, require_tags: bool = True) -> list[float]:
    """
    Sample points along the beam centreline (y = BEAM_CENTER[1]) and detect
    x-locations without an inside/cut element covering the point.
    Useful to catch missing inside elements near the beam attachment.
    When `require_tags` is False, any element covering the query point counts
    (used before the mesh has been classified).
    """
    xs = np.linspace(CENTER[0] + RADIUS, x_end, n_samples)
    y0 = BEAM_CENTER[1]
    gaps: list[float] = []
    for x0 in xs:
        found = False
        for e in mesh.elements_list:
            if require_tags:
                tag = getattr(e, "tag", "")
                if tag not in ("inside", "cut"):
                    continue
            cn = mesh.nodes_x_y_pos[list(e.corner_nodes)]
            if x0 < cn[:, 0].min() - tol or x0 > cn[:, 0].max() + tol:
                continue
            if y0 < cn[:, 1].min() - tol or y0 > cn[:, 1].max() + tol:
                continue
            try:
                xi, eta = transform.inverse_mapping(mesh, e.id, (x0, y0))
            except Exception:
                continue
            if -1.001 <= xi <= 1.001 and -1.001 <= eta <= 1.001:
                found = True
                break
        if not found:
            gaps.append(float(x0))
    return gaps


def interface_approx_error(mesh: Mesh, beam_ls: BeamLevelSet) -> dict[str, float]:
    """
    Measure how well the discrete interface points approximate the beam level set.
    Returns max/mean |φ| evaluated at interface points (0 is exact).
    """
    errs: list[float] = []
    n_pts = 0
    edge_ids = mesh.edge_bitset("interface").to_indices()
    if edge_ids.size:
        for e in edge_ids:
            edge = mesh.edge(int(e))
            pts: list[tuple[float, float]] = []
            for owner in (edge.left, edge.right):
                if owner is None:
                    continue
                el = mesh.elements_list[int(owner)]
                if getattr(el, "interface_pts", None):
                    pts.extend(el.interface_pts)
            if not pts:
                mid = mesh.nodes_x_y_pos[list(edge.nodes)].mean(axis=0)
                pts.append((float(mid[0]), float(mid[1])))
            for pt in pts:
                errs.append(abs(float(beam_ls(np.asarray(pt, float)))))
                n_pts += 1
    else:
        for elem in mesh.elements_list:
            if getattr(elem, "tag", "") != "cut":
                continue
            for pt in getattr(elem, "interface_pts", []):
                errs.append(abs(float(beam_ls(np.asarray(pt, float)))))
                n_pts += 1
    if not errs:
        return {"max_abs_phi": 0.0, "mean_abs_phi": 0.0, "n_pts": 0}
    errs_arr = np.asarray(errs, float)
    return {
        "max_abs_phi": float(errs_arr.max()),
        "mean_abs_phi": float(errs_arr.mean()),
        "n_pts": int(n_pts),
    }


def beam_inside_coverage(mesh: Mesh, beam_ls: BeamLevelSet, nx: int = 120, ny: int = 8, tol: float = 1.0e-8, *, inside_only: bool = False) -> dict[str, object]:
    """
    Sample the beam rectangle and report coverage by inside/cut elements.
    Returns coverage fraction and x-locations missing coverage.
    Set inside_only=True to require elements tagged strictly 'inside'.
    """
    x0 = beam_ls.cx - beam_ls.hx
    x1 = beam_ls.cx + beam_ls.hx
    y0 = beam_ls.cy - beam_ls.hy
    y1 = beam_ls.cy + beam_ls.hy
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    total = 0
    covered = 0
    missing_x: list[float] = []
    for x in xs:
        column_missing = False
        for y in ys:
            total += 1
            found = False
            for e in mesh.elements_list:
                tag = getattr(e, "tag", "")
                if inside_only:
                    if tag != "inside":
                        continue
                else:
                    if tag not in ("inside", "cut"):
                        continue
                cn = mesh.nodes_x_y_pos[list(e.corner_nodes)]
                if x < cn[:, 0].min() - tol or x > cn[:, 0].max() + tol:
                    continue
                if y < cn[:, 1].min() - tol or y > cn[:, 1].max() + tol:
                    continue
                try:
                    xi, eta = transform.inverse_mapping(mesh, e.id, (x, y))
                except Exception:
                    continue
                if -1.01 <= xi <= 1.01 and -1.01 <= eta <= 1.01:
                    found = True
                    break
            if found:
                covered += 1
            else:
                column_missing = True
        if column_missing:
            missing_x.append(float(x))
    frac = covered / max(total, 1)
    return {"coverage": frac, "missing_x": missing_x, "samples": total}

def repair_degenerate_edges(mesh: Mesh, tol: float = 1.0e-10) -> int:
    """
    For interior edges whose owners share no common node, rebuild the node
    lists along the edge using geometric collinearity so hanging nodes are
    consistently shared. Returns number of repaired edges.
    """
    repaired = 0
    pts = mesh.nodes_x_y_pos
    for e in mesh.edges_list:
        if e.left is None or e.right is None:
            continue
        shared = set(e.left_nodes).intersection(set(e.right_nodes))
        # Allow single-node (T-junction/hanging) intersections; only repair
        # when owners have no common node.
        if len(shared) >= 1:
            continue
        n0, n1 = e.nodes
        p0, p1 = pts[n0], pts[n1]
        d = p1 - p0
        L2 = np.dot(d, d)
        if L2 < tol:
            continue
        # Collect candidate nodes from both owners that lie on the segment.
        cand_ids = set(mesh.elements_list[e.left].nodes) | set(mesh.elements_list[e.right].nodes)
        on_edge = []
        for nid in cand_ids:
            q = pts[int(nid)]
            cross = abs((q[0] - p0[0]) * d[1] - (q[1] - p0[1]) * d[0])
            if cross > np.sqrt(L2) * tol:
                continue
            t = np.dot(q - p0, d) / L2
            if -tol <= t <= 1.0 + tol:
                on_edge.append((t, int(nid)))
        on_edge.sort()
        seq = [nid for _, nid in on_edge]
        if len(seq) < 2:
            seq = [int(n0), int(n1)]
        e.all_nodes = tuple(seq)
        e.left_nodes = tuple(n for n in seq if n in mesh.elements_list[e.left].nodes)
        e.right_nodes = tuple(n for n in seq if n in mesh.elements_list[e.right].nodes)
        repaired += 1
    return repaired

def retag_inactive(
    dh: DofHandler,
    *,
    theta_neg: np.ndarray | None = None,
    solid_cut_drop: float = 0.0,
) -> None:
    dh.dof_tags["inactive"] = set()
    dh.tag_dofs_from_element_bitset("inactive", "u_pos_x", "inside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "u_pos_y", "inside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "p_pos_", "inside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "vs_neg_x", "outside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "vs_neg_y", "outside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "d_neg_x", "outside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "d_neg_y", "outside", strict=True)
    if theta_neg is not None and solid_cut_drop > 0.0:
        cut_mask = mesh.element_bitset("cut").mask
        bad = cut_mask & (theta_neg < solid_cut_drop)
        if np.any(bad):
            for field in ("vs_neg_x", "vs_neg_y", "d_neg_x", "d_neg_y"):
                dh.tag_dofs_from_element_bitset("inactive", field, bad, strict=False)


def recompute_active_dofs(solver: NewtonSolver, bcs_active: Sequence[BoundaryCondition]) -> bool:
    dh = solver.dh
    constraints = getattr(solver, "constraints", None)
    ndof_eff = int(constraints.n_master) if constraints is not None else int(dh.total_dofs)
    old_active = np.asarray(getattr(solver, "active_dofs", np.empty(0, dtype=int)), dtype=int)
    active_by_restr, has_restriction = analyze_active_dofs(solver.equation, dh, solver.me, bcs_active)
    bc_dofs_full = set(dh.get_dirichlet_data(bcs_active).keys())
    inactive_full = set(dh.dof_tags.get("inactive", set()))
    inactive_free_full = inactive_full - bc_dofs_full

    if constraints is None:
        candidate = set(active_by_restr) if has_restriction else set(range(ndof_eff))
        free = sorted((candidate - bc_dofs_full) - inactive_free_full)
        new_active = np.asarray(free, dtype=int)
    else:
        candidate_master = (
            constraints.to_master_set(active_by_restr) if has_restriction else set(range(ndof_eff))
        )
        bc_master = constraints.to_master_set(bc_dofs_full)
        inactive_master = constraints.to_master_set(inactive_free_full)
        free = sorted((candidate_master - bc_master) - inactive_master)
        new_active = np.asarray(free, dtype=int)
    dec_mask = getattr(solver, "_decoupled_full_mask", None)
    if isinstance(dec_mask, np.ndarray) and dec_mask.dtype == bool and int(dec_mask.size) == int(ndof_eff):
        if new_active.size:
            new_active = new_active[~dec_mask[new_active]]
    if old_active.size == new_active.size and np.array_equal(old_active, new_active):
        return False
    solver.active_dofs = new_active
    solver.full_to_red = -np.ones(ndof_eff, dtype=int)
    solver.full_to_red[solver.active_dofs] = np.arange(len(solver.active_dofs), dtype=int)
    solver.red_to_full = solver.active_dofs
    solver.use_reduced = len(solver.active_dofs) < ndof_eff
    solver.restrictor = _ActiveReducer(dh, solver.active_dofs, constraint=constraints)
    solver._pattern_stale = True
    return True


# -----------------------------------------------------------------------------
# Robustness: initialize newly-active DOFs after geometry changes
# -----------------------------------------------------------------------------


def extend_newly_active_dofs_nearest(
    *,
    dh: DofHandler,
    newly_active: np.ndarray,
    active_old: np.ndarray,
    active_new: np.ndarray,
    field_to_current,
    field_to_prev,
    k: int = 4,
    trace: bool = False,
) -> None:
    """
    When the interface moves, some DOFs transition from inactive→active and were
    not solved for previously. If left at stale/zero values, the time-derivative
    terms can create large residual jumps and Newton can fail.

    This routine initializes those newly-active DOFs by a nearest-neighbor (or
    k-NN inverse-distance weighted) extension from DOFs that remained active.
    """
    newly_active = np.asarray(newly_active, dtype=int).ravel()
    if newly_active.size == 0:
        return

    # Restrict sources to DOFs that are active *now* and were already active before.
    active_old = np.asarray(active_old, dtype=int).ravel()
    active_new = np.asarray(active_new, dtype=int).ravel()
    stable_active = np.setdiff1d(active_new, newly_active, assume_unique=False)
    if stable_active.size == 0:
        return

    try:
        dh._ensure_dof_coords()  # internal but reliable; coords are geometry-independent
        coords = np.asarray(dh._dof_coords, dtype=float)
    except Exception:
        return

    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:
        cKDTree = None  # noqa: N806

    if cKDTree is None:
        return

    field_names = sorted(set(field_to_current.keys()) & set(field_to_prev.keys()))
    if not field_names:
        return

    # Precompute per-field DOF slices and per-carrier offsets (VectorFunction layout is concatenated slices).
    field_slices: dict[str, np.ndarray] = {
        f: np.asarray(dh.get_field_slice(f), dtype=int) for f in field_names
    }

    def _carrier_field_view(carrier, field: str) -> np.ndarray | None:
        if carrier is None:
            return None
        if hasattr(carrier, "field_name"):
            if getattr(carrier, "field_name", None) != field:
                return None
            return carrier.nodal_values
        if hasattr(carrier, "field_names"):
            names = list(getattr(carrier, "field_names", []))
            if field not in names:
                return None
            start = 0
            for nm in names:
                n = int(field_slices[nm].size)
                if nm == field:
                    return carrier.nodal_values[start : start + n]
                start += n
        return None

    for field in field_names:
        fs = field_slices[field]
        # Newly active DOFs for this field (global ids).
        new_f = np.intersect1d(newly_active, fs, assume_unique=False)
        if new_f.size == 0:
            continue

        # Source DOFs: stable active DOFs for this field.
        src_f = np.intersect1d(stable_active, fs, assume_unique=False)
        if src_f.size == 0:
            if trace:
                print(f"[extend] field={field}: no stable active source DOFs; skipping {new_f.size} targets")
            continue

        cur_car = field_to_current[field]
        prev_car = field_to_prev[field]
        cur_view = _carrier_field_view(cur_car, field)
        prev_view = _carrier_field_view(prev_car, field)
        if cur_view is None or prev_view is None:
            continue

        # Map src/target global DOFs to local indices in the field slice.
        src_pos = np.searchsorted(fs, src_f)
        src_vals = np.asarray(cur_view[src_pos], dtype=float)

        # Build k-NN extension
        k_eff = int(max(1, min(int(k), int(src_f.size))))
        tree = cKDTree(coords[src_f])
        dist, idx = tree.query(coords[new_f], k=k_eff)
        if k_eff == 1:
            new_vals = src_vals[np.asarray(idx, dtype=int)]
        else:
            idx = np.asarray(idx, dtype=int)
            dist = np.asarray(dist, dtype=float)
            w = 1.0 / np.maximum(dist, 1.0e-12)
            wsum = np.sum(w, axis=1)
            new_vals = np.sum(w * src_vals[idx], axis=1) / np.maximum(wsum, 1.0e-30)

        tgt_pos = np.searchsorted(fs, new_f)
        cur_view[tgt_pos] = new_vals
        prev_view[tgt_pos] = new_vals

        if trace:
            maxd = float(np.max(dist)) if np.size(dist) else 0.0
            print(f"[extend] field={field}: filled {new_f.size} DOFs from {src_f.size} sources (k={k_eff}, max_dist={maxd:.3e})")


# -----------------------------------------------------------------------------
# Finite-difference Jacobian check
# -----------------------------------------------------------------------------


def select_fd_dofs(
    dh: DofHandler,
    fields_to_probe: Dict[str, int],
    elem_tag: str = "cut",
    *,
    bc_dofs: set[int] | None = None,
    inactive: set[int] | None = None,
    max_elems: int = 50,
) -> np.ndarray:
    selected: list[int] = []
    seen = set()
    elems = dh.element_bitset(elem_tag).to_indices()
    if elems.size == 0:
        return np.array([], dtype=int)
    remaining = dict(fields_to_probe)
    for eid in elems[:max_elems]:
        done = True
        for field, count in list(remaining.items()):
            if count <= 0:
                continue
            done = False
            if field not in dh.field_names:
                remaining[field] = 0
                continue
            try:
                local = dh.element_dofs(field, int(eid))
            except Exception:
                continue
            for gdof in local:
                gdof_i = int(gdof)
                if gdof_i < 0:
                    continue
                if bc_dofs and gdof_i in bc_dofs:
                    continue
                if inactive and gdof_i in inactive:
                    continue
                if gdof_i in seen:
                    continue
                selected.append(gdof_i)
                seen.add(gdof_i)
                remaining[field] = remaining[field] - 1
                if remaining[field] <= 0:
                    break
        if done or all(val <= 0 for val in remaining.values()):
            break
    return np.array(sorted(set(selected)), dtype=int)


def select_fd_probes(
    dh: DofHandler,
    probes_by_tag: Dict[str, Dict[str, int]],
    *,
    bc_dofs: set[int] | None = None,
    inactive: set[int] | None = None,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    all_dofs: list[int] = []
    by_tag: Dict[str, np.ndarray] = {}
    for tag, fields in probes_by_tag.items():
        dofs = select_fd_dofs(
            dh,
            fields,
            elem_tag=tag,
            bc_dofs=bc_dofs,
            inactive=inactive,
        )
        by_tag[tag] = dofs
        all_dofs.extend(dofs.tolist())
    return np.array(sorted(set(all_dofs)), dtype=int), by_tag


def finite_difference_check(
    jac_form,
    res_form,
    dh: DofHandler,
    bcs: Sequence[BoundaryCondition],
    functions: Dict[str, VectorFunction | Function],
    probe_dofs: Iterable[int],
    eps: float = 1.0e-6,
    *,
    bcs_elim: Sequence[BoundaryCondition] | None = None,
    coeffs: Dict[str, VectorFunction | Function] | None = None,
    backend: str | None = None,
    apply_bcs_each: bool = False,
    label: str | None = None,
    kernel_pool: Dict[str, Dict[int, object]] | None = None,
) -> None:
    backend = backend or FD_BACKEND
    debug_timing = os.getenv("FD_TIMING", "").lower() in {"1", "true", "yes"}
    cols_only = os.getenv("FD_JAC_COLS_ONLY", "1") != "0"
    timings: Dict[str, float] = {}
    t_total = time.perf_counter() if debug_timing else None

    def _t_add(key: str, dt: float) -> None:
        timings[key] = timings.get(key, 0.0) + dt

    if jac_form is None or res_form is None:
        print("Skipping FD check: Jacobian or residual form missing.")
        return
    eq_jac = Equation(jac_form, None)
    eq_res = Equation(None, res_form)
    bc_values = list(bcs) if bcs else []
    bc_elim = list(bcs_elim) if bcs_elim else []
    uniq_funcs = []
    seen = set()
    for func in functions.values():
        if id(func) in seen:
            continue
        seen.add(id(func))
        uniq_funcs.append(func)
    if bc_values:
        dh.apply_bcs(bc_values, *uniq_funcs)

    probe_list = list(probe_dofs)
    if debug_timing:
        lbl = f" ({label})" if label else ""
        print(
            f"[FD timing{lbl}] start backend={backend} fast_jit={backend=='jit' and coeffs is not None} "
            f"probes={len(probe_list)} eps={eps:.1e} cols_only={cols_only}"
        )

    def _form_integrals(form):
        if form is None:
            return []
        ints = getattr(form, "integrals", None)
        return list(ints) if ints is not None else [form]

    def _select_kernels_from_pool(form, pool, tag):
        if not pool:
            return None
        ints = _form_integrals(form)
        missing = 0
        selected = []
        for intg in ints:
            ker_list = pool.get(id(intg))
            if ker_list is None:
                missing += 1
            else:
                if isinstance(ker_list, list):
                    selected.extend(ker_list)
                else:
                    selected.append(ker_list)
        if missing:
            if debug_timing:
                print(
                    f"[FD timing] kernel pool missing {missing}/{len(ints)} integrals for {tag}; "
                    "falling back to compile."
                )
            return None
        return selected

    def _apply_dirichlet(K, R, bcs_apply):
        if not bcs_apply:
            return
        data = dh.get_dirichlet_data(bcs_apply)
        if not data:
            return
        rows = np.fromiter(data.keys(), dtype=int)
        vals = np.fromiter(data.values(), dtype=float)
        if K is not None:
            if R is not None and np.any(vals):
                bc_vec = np.zeros(R.size)
                bc_vec[rows] = vals
                R -= K @ bc_vec
            K_lil = K.tolil()
            K_lil[rows, :] = 0
            K_lil[:, rows] = 0
            K_lil[rows, rows] = 1.0
            K[:] = K_lil
        if R is not None:
            R[rows] = vals

    # Fast JIT path: precompile kernels and reuse static geometry for FD sweeps.
    use_fast_jit = backend == "jit" and coeffs is not None
    base_cols = None
    if use_fast_jit:
        import scipy.sparse as sp
        from pycutfem.jit import compile_multi
        from pycutfem.ufl.helpers_jit import _scatter_element_contribs

        ndof = dh.total_dofs
        kernels_K = None
        kernels_R = None
        if kernel_pool:
            kernels_K = _select_kernels_from_pool(jac_form, kernel_pool.get("jac", {}), "jac")
            kernels_R = _select_kernels_from_pool(res_form, kernel_pool.get("res", {}), "res")
            if debug_timing and kernels_K is not None and kernels_R is not None:
                print("[FD timing] reusing kernel pool (no compile).")
        if kernels_K is None:
            if debug_timing:
                print("[FD timing] compiling jac kernels...")
            t0 = time.perf_counter() if debug_timing else None
            kernels_K = compile_multi(
                jac_form,
                dof_handler=dh,
                mixed_element=dh.mixed_element,
                backend=backend,
            )
            if debug_timing and t0 is not None:
                dt = time.perf_counter() - t0
                _t_add("compile_jac", dt)
                print(f"[FD timing] compiled jac kernels in {dt:.3f}s (n={len(kernels_K)})")
        if kernels_R is None:
            if debug_timing:
                print("[FD timing] compiling res kernels...")
            t0 = time.perf_counter() if debug_timing else None
            kernels_R = compile_multi(
                res_form,
                dof_handler=dh,
                mixed_element=dh.mixed_element,
                backend=backend,
            )
            if debug_timing and t0 is not None:
                dt = time.perf_counter() - t0
                _t_add("compile_res", dt)
                print(f"[FD timing] compiled res kernels in {dt:.3f}s (n={len(kernels_R)})")

        def _slice_static_args(static_args, idx):
            n_total = int(np.asarray(static_args.get("eids")).shape[0])
            sliced = {}
            for key, val in static_args.items():
                if isinstance(val, np.ndarray) and val.ndim >= 1 and int(val.shape[0]) == n_total:
                    sliced[key] = val[idx]
                else:
                    sliced[key] = val
            return sliced

        def assemble_K_cols(probes):
            probe_set = {int(p) for p in probes}
            probe_arr = np.fromiter(probe_set, dtype=int)
            jac_cols = {int(p): np.zeros(ndof) for p in probes}
            domain_times: Dict[str, float] = {}
            for ker in kernels_K:
                gdofs_map = ker.static_args["gdofs_map"]
                if gdofs_map.size == 0:
                    continue
                mask = np.any(np.isin(gdofs_map, probe_arr), axis=1)
                if not np.any(mask):
                    continue
                idx = np.where(mask)[0]
                static_args = _slice_static_args(ker.static_args, idx)
                t0 = time.perf_counter() if debug_timing else None
                Kloc, _, _ = ker.runner(coeffs, static_args)
                if debug_timing and t0 is not None:
                    dt = time.perf_counter() - t0
                    domain_times[ker.domain] = domain_times.get(ker.domain, 0.0) + dt
                gdofs_sub = static_args["gdofs_map"]
                for e in range(gdofs_sub.shape[0]):
                    gdofs = gdofs_sub[e]
                    valid_rows = gdofs >= 0
                    if not np.any(valid_rows):
                        continue
                    row_idx = np.where(valid_rows)[0]
                    rows = gdofs[valid_rows]
                    Kloc_e = Kloc[e]
                    cols = [j for j, g in enumerate(gdofs) if g in probe_set]
                    for j in cols:
                        gcol = int(gdofs[j])
                        if gcol < 0:
                            continue
                        jac_cols[gcol][rows] += Kloc_e[row_idx, j]
            if debug_timing:
                for dom, dt in domain_times.items():
                    _t_add(f"assemble_K:{dom}", dt)
            if bc_elim:
                data = dh.get_dirichlet_data(bc_elim)
                if data:
                    bc_rows = np.fromiter(data.keys(), dtype=int)
                    for gcol in jac_cols:
                        col = jac_cols[gcol]
                        col[bc_rows] = 0.0
                        if gcol in data:
                            col[:] = 0.0
                            col[gcol] = 1.0
            return jac_cols

        def assemble_K():
            K = sp.lil_matrix((ndof, ndof))
            domain_times: Dict[str, float] = {}
            for ker in kernels_K:
                t0 = time.perf_counter() if debug_timing else None
                Kloc, _, _ = ker.exec(coeffs)
                _scatter_element_contribs(
                    K_elem=Kloc,
                    F_elem=None,
                    J_elem=None,
                    element_ids=ker.static_args["eids"],
                    gdofs_map=ker.static_args["gdofs_map"],
                    matvec=K,
                    ctx={"rhs": False, "add": True},
                    integrand=ker,
                    hook=None,
                )
                if debug_timing and t0 is not None:
                    dt = time.perf_counter() - t0
                    domain_times[ker.domain] = domain_times.get(ker.domain, 0.0) + dt
            if debug_timing:
                for dom, dt in domain_times.items():
                    _t_add(f"assemble_K:{dom}", dt)
            if bc_elim:
                _apply_dirichlet(K, None, bc_elim)
            return K.tocsr()

        def assemble_R():
            R = np.zeros(ndof)
            domain_times: Dict[str, float] = {}
            for ker in kernels_R:
                t0 = time.perf_counter() if debug_timing else None
                _, Floc, _ = ker.exec(coeffs)
                _scatter_element_contribs(
                    K_elem=None,
                    F_elem=Floc,
                    J_elem=None,
                    element_ids=ker.static_args["eids"],
                    gdofs_map=ker.static_args["gdofs_map"],
                    matvec=R,
                    ctx={"rhs": True, "add": True},
                    integrand=ker,
                    hook=None,
                )
                if debug_timing and t0 is not None:
                    dt = time.perf_counter() - t0
                    domain_times[ker.domain] = domain_times.get(ker.domain, 0.0) + dt
            if debug_timing:
                for dom, dt in domain_times.items():
                    _t_add(f"assemble_R:{dom}", dt)
            if bc_elim:
                _apply_dirichlet(None, R, bc_elim)
            return R

        base_K = None
        base_cols = None
        t0 = time.perf_counter() if debug_timing else None
        if cols_only:
            base_cols = assemble_K_cols(probe_list)
        else:
            base_K = assemble_K()
        if debug_timing and t0 is not None:
            dt = time.perf_counter() - t0
            _t_add("assemble_base_K", dt)
            print(f"[FD timing] assembled base K in {dt:.3f}s")
    else:
        compiler = FormCompiler(dh, backend=backend)
        t0 = time.perf_counter() if debug_timing else None
        base_K, _ = compiler.assemble(eq_jac, bcs=bc_elim or None)
        if debug_timing and t0 is not None:
            dt = time.perf_counter() - t0
            _t_add("assemble_base_K", dt)
            print(f"[FD timing] assembled base K in {dt:.3f}s")
    if base_K is None:
        if base_cols is None:
            print("Skipping FD check: Jacobian or residual form missing.")
            return

    def perturb(field: str, gdof: int, new_value: float) -> float:
        func = functions[field]
        old = func.get_nodal_values(np.array([gdof], dtype=int))[0]
        func.set_nodal_values(np.array([gdof], dtype=int), np.array([new_value], dtype=float))
        return old

    rows = []
    debug_col = os.getenv("FD_DEBUG_COL", "").strip()
    debug_top = int(os.getenv("FD_DEBUG_TOP", "5") or "5")
    bc_src = bc_values if bc_values else bc_elim
    bc_dofs = set(dh.get_dirichlet_data(bc_src).keys()) if bc_src else set()
    inactive = set(dh.dof_tags.get("inactive", set()))
    active_mask = np.ones(dh.total_dofs, dtype=bool)
    if bc_dofs:
        active_mask[np.fromiter(bc_dofs, dtype=int)] = False
    if inactive:
        active_mask[np.fromiter(inactive, dtype=int)] = False
    for gdof in probe_list:
        field, _ = dh._dof_to_node_map[int(gdof)]
        if field not in functions:
            continue
        if int(gdof) in bc_dofs:
            continue
        if int(gdof) in inactive:
            continue
        old_val = functions[field].get_nodal_values(np.array([gdof], dtype=int))[0]
        perturb(field, int(gdof), old_val + eps)
        if apply_bcs_each and bc_values:
            dh.apply_bcs(bc_values, *uniq_funcs)
        t0 = time.perf_counter() if debug_timing else None
        if use_fast_jit:
            R_plus = assemble_R()
        else:
            _, R_plus = compiler.assemble(eq_res, bcs=bc_elim or None)
        if debug_timing and t0 is not None:
            _t_add("assemble_R_plus", time.perf_counter() - t0)
        perturb(field, int(gdof), old_val - eps)
        if apply_bcs_each and bc_values:
            dh.apply_bcs(bc_values, *uniq_funcs)
        t0 = time.perf_counter() if debug_timing else None
        if use_fast_jit:
            R_minus = assemble_R()
        else:
            _, R_minus = compiler.assemble(eq_res, bcs=bc_elim or None)
        if debug_timing and t0 is not None:
            _t_add("assemble_R_minus", time.perf_counter() - t0)
        perturb(field, int(gdof), old_val)
        if apply_bcs_each and bc_values:
            dh.apply_bcs(bc_values, *uniq_funcs)
        fd_col = (R_plus - R_minus) / (2 * eps)
        if base_K is not None:
            jac_col = base_K[:, int(gdof)].toarray().ravel()
        else:
            jac_col = base_cols.get(int(gdof), np.zeros(dh.total_dofs))
        err_vec = fd_col - jac_col
        err = np.linalg.norm(err_vec[active_mask], ord=np.inf)
        mag = np.linalg.norm(jac_col[active_mask], ord=np.inf)
        rel = err / (mag + 1.0e-14)
        rows.append((gdof, field, err, mag, rel))
        if debug_col and int(debug_col) == int(gdof):
            abs_err = np.abs(err_vec)
            top_n = min(debug_top, abs_err.size)
            top_idx = np.argpartition(abs_err, -top_n)[-top_n:]
            top_idx = top_idx[np.argsort(abs_err[top_idx])[::-1]]
            print(f"[FD debug] gdof={gdof} field={field} top_diff_idx={top_idx.tolist()}")
            for idx in top_idx:
                row_field, _ = dh._dof_to_node_map[int(idx)]
                row_active = bool(active_mask[int(idx)])
                print(
                    f"[FD debug] row={int(idx)} field={row_field} active={row_active} "
                    f"fd={fd_col[int(idx)]:.3e} jac={jac_col[int(idx)]:.3e} diff={err_vec[int(idx)]:.3e}"
                )
        if debug_timing and t_total is not None:
            print(
                f"[FD timing] dof={gdof} field={field} total_elapsed={time.perf_counter() - t_total:.3f}s"
            )
    print("Finite-difference Jacobian check (gdof, field, err, |J|, rel):")
    for gd, fld, err, mag, rel in rows:
        print(f"  {gd:5d}  {fld:10s}  err={err:9.3e}  |J|={mag:9.3e}  rel={rel:9.3e}")
    if debug_timing and t_total is not None:
        total = time.perf_counter() - t_total
        lbl = f" ({label})" if label else ""
        print(f"[FD timing{lbl}] total {total:.3f}s")
        for key in sorted(timings):
            print(f"[FD timing]   {key} {timings[key]:.3f}s")


def detect_mesh_pathologies(mesh: Mesh, tol: float = 1e-10):
    """
    Detects degenerate features, zombie entities, and internal topological cracks.
    """
    pathologies = {
        "zero_length_edges": [],
        "zombie_edges": [],  # Edges with NO owners
        "internal_cracks": [], # Edges with 1 owner that are NOT on the boundary
        "orphan_nodes": [], # Nodes not used by any element
    }

    # --- 1. Detect Zero-Length Edges ---
    # Calculate lengths of all edges
    nodes_all = mesh.nodes_x_y_pos
    span = float(np.ptp(nodes_all, axis=0).max() or 1.0)
    tol_len = max(tol, 1e-9 * span)
    edge_nodes = np.array([list(e.nodes) for e in mesh.edges_list])
    p0 = nodes_all[edge_nodes[:, 0]]
    p1 = nodes_all[edge_nodes[:, 1]]
    lengths = np.linalg.norm(p1 - p0, axis=1)
    
    bad_len_indices = np.where(lengths < tol_len)[0]
    pathologies["zero_length_edges"] = bad_len_indices.tolist()

    # --- 2. Detect Zombie Edges (No Owners) ---
    # An edge should have at least a 'left' or 'right' element.
    # If both are None, it's a ghost object floating in memory.
    zombies = []
    internal_cracks = []
    
    # Pre-calculate domain bounding box to help identify 'internal' cracks
    xmin, ymin = nodes_all.min(axis=0)
    xmax, ymax = nodes_all.max(axis=0)
    
    for e in mesh.edges_list:
        if e.left is None and e.right is None:
            zombies.append(e.gid)
            continue
            
        # --- 3. Detect Internal Cracks ---
        # If an edge has only 1 owner, it MUST be on the physical boundary.
        # If it is geometrically inside the domain, it is a topological crack.
        if (e.left is None) ^ (e.right is None): # XOR: exactly one is None
            # Check midpoint
            mid = (nodes_all[e.nodes[0]] + nodes_all[e.nodes[1]]) * 0.5
            
            # Simple bounding box check (strict interior)
            # (A robust check would use your boundary_locators)
            on_boundary = (
                abs(mid[0] - xmin) < 1e-4 or abs(mid[0] - xmax) < 1e-4 or
                abs(mid[1] - ymin) < 1e-4 or abs(mid[1] - ymax) < 1e-4
            )
            
            # Check cylinder boundary (specific to Turek FSI)
            # Center=(0.2, 0.2), Radius=0.05
            dx = mid[0] - 0.2
            dy = mid[1] - 0.2
            dist_cyl = math.sqrt(dx*dx + dy*dy)
            on_cylinder = abs(dist_cyl - 0.05) < 1e-3
            
            if not (on_boundary or on_cylinder):
                internal_cracks.append(e.gid)

    pathologies["zombie_edges"] = zombies
    pathologies["internal_cracks"] = internal_cracks
    if internal_cracks:
        crack_lengths = lengths[np.array(internal_cracks, dtype=int)]
        print(
            f"  crack length stats: min={crack_lengths.min():.3e}, "
            f"median={np.median(crack_lengths):.3e}, max={crack_lengths.max():.3e}"
        )
        sample = internal_cracks[:5]
        for gid in sample:
            e = mesh.edges_list[int(gid)]
            mid = nodes_all[list(e.nodes)].mean(axis=0)
            print(
                f"    crack edge {gid}: left={e.left} right={e.right} "
                f"mid=({mid[0]:.4f},{mid[1]:.4f}) len={np.linalg.norm(nodes_all[list(e.nodes)][1]-nodes_all[list(e.nodes)][0]):.3e}"
            )

    # --- 4. Detect Orphan Nodes ---
    # Nodes that exist in the list but are not referenced by any active element
    active_nodes = set()
    for el in mesh.elements_list:
        active_nodes.update(el.nodes)
        
    all_node_ids = set(n.id for n in mesh.nodes_list)
    orphans = list(all_node_ids - active_nodes)
    pathologies["orphan_nodes"] = orphans

    # --- Report ---
    print("\n--- Mesh Pathology Report ---")
    print(f"Zero-length edges: {len(pathologies['zero_length_edges'])}")
    if pathologies['zero_length_edges']:
        print(f"  -> IDs: {pathologies['zero_length_edges'][:10]}...")
        
    print(f"Zombie edges (no owner): {len(pathologies['zombie_edges'])}")
    
    print(f"Internal Cracks (1 owner, inside domain): {len(pathologies['internal_cracks'])}")
    if pathologies['internal_cracks']:
        print(f"  -> IDs: {pathologies['internal_cracks'][:10]}...")
        
    print(f"Orphan Nodes (unused): {len(pathologies['orphan_nodes'])}")
    
    return pathologies

# -----------------------------------------------------------------------------
# Main setup
# -----------------------------------------------------------------------------
print(f"--- Setting up the Turek–Hron {CASE_LABEL} benchmark ---")
re_mean = RHO_F * U_MEAN * (2 * RADIUS) / MU_F
print(
    f"Benchmark preset: {CASE_LABEL} | U_mean={U_MEAN:g}, rho_s={RHO_S:g}, "
    f"dt={DT:g}, theta={float(ARGS.theta):g} | Re_mean≈{re_mean:.1f}"
)
_log_step("start setup")

# Beam level set (reference configuration)
beam_ref_center = BEAM_REF_CENTER
beam_ref_length = BEAM_REF_LENGTH
beam_ref_height = BEAM_REF_HEIGHT

# Mesh with rigid hole
mesh = build_channel_mesh(
    MESH_SIZE,
    POLY_ORDER,
    beam_center=beam_ref_center,
    beam_length=beam_ref_length,
    beam_height=beam_ref_height,
)
_log_step(f"built base mesh ({ARGS.mesh_backend})")

beam_ref_ls = BeamArcRootLevelSet(
    beam_center=beam_ref_center,
    beam_length=beam_ref_length,
    beam_height=beam_ref_height,
    cyl_center=CENTER,
    cyl_radius=RADIUS,
    root_inset=BEAM_ROOT_INSET,
    root_bias=BEAM_ROOT_BIAS,
    root_tol=BEAM_ROOT_TOL,
)
if ARGS.refine_initial:
    mesh = refine_beam_anisotropic(mesh, beam_ref_ls, levels=2, target_h=0.5 * BEAM_HEIGHT)
    _log_step("refined mesh around beam")
    # Final clean-up: eliminate lingering 4:1 T-junctions before classification/plots
    mesh = fix_fragmented_sides(mesh, max_segments=2, max_iters=2)
else:
    _log_step("skipped anisotropic beam refinement (refine_initial=0)")

# Use higher order for φ_beam to reduce geometric distortion of the zero set
ls_me = MixedElement(mesh, field_specs={"phi_beam": POLY_ORDER})
ls_dh = DofHandler(ls_me, method="cg")
ls_beam = LevelSetGridFunction(ls_dh, field="phi_beam")
ls_beam.interpolate(lambda x, y: beam_ref_ls(np.array([x, y])))
ls_beam.commit(tol=LEVELSET_EDGE_TOL)
nudged = _nudge_levelset_zeros(
    ls_beam,
    LEVELSET_NUDGE_EPS,
    prefer_negative=LEVELSET_PREFER_NEGATIVE,
    commit=False,
)
if nudged:
    print(
        f"[levelset] nudged {nudged} near-zero nodes by {LEVELSET_NUDGE_EPS:.2e} "
        f"towards {'-' if LEVELSET_PREFER_NEGATIVE else '+'}"
    )
# Classify mesh against beam level set
mesh.classify_elements(ls_beam, tol=LEVELSET_EDGE_TOL)
mesh.classify_edges(ls_beam, tol=LEVELSET_EDGE_TOL)
mesh.build_interface_segments(ls_beam, tol=LEVELSET_EDGE_TOL)
_log_step("interpolated/committed level set")
if not USE_ALIGNED_INTERFACE:
    disabled = _disable_aligned_interface_edges(mesh)
    if disabled:
        print(f"[interface] disabled {disabled} aligned interface edges; using cut-cell interface integrals only.")
    if mesh.edge_bitset("interface").cardinality() > 0:
        print("[warn] Interface edges are still tagged; aligned-interface assembly may be active.")

domains = make_domain_sets(mesh)
inside_elems = mesh.element_bitset("inside").to_indices()
if inside_elems.size:
    x_in = mesh.nodes_x_y_pos[np.unique(np.concatenate([mesh.corner_connectivity[i] for i in inside_elems]))][:, 0]
    print(f"Inside elements: {inside_elems.size}, x-span=({x_in.min():.3f},{x_in.max():.3f})")
else:
    print("Inside elements: 0")
cut_elems = mesh.element_bitset("cut").to_indices()
if cut_elems.size:
    x_cut = mesh.nodes_x_y_pos[np.unique(np.concatenate([mesh.corner_connectivity[i] for i in cut_elems]))][:, 0]
    print(f"Cut elements: {cut_elems.size}, x-span=({x_cut.min():.3f},{x_cut.max():.3f})")
if ARGS.mesh_diagnostics_enabled:
    centerline_gaps = inside_centerline_gaps(mesh, x_end=0.6, n_samples=160)
    if centerline_gaps:
        msg = ", ".join(f"{x:.4f}" for x in centerline_gaps[:8])
        if len(centerline_gaps) > 8:
            msg += " ..."
        print(f"[inside-check] Missing inside/cut coverage along beam centreline at x≈ {msg}")
    iface_err = interface_approx_error(mesh, beam_ref_ls)
    print(
        f"Interface approximation: max|phi|={iface_err['max_abs_phi']:.3e}, "
        f"mean|phi|={iface_err['mean_abs_phi']:.3e} over {iface_err['n_pts']} pts"
    )
    tj_bad = mesh.count_tjunction_violations()
    worst = tj_bad.get("worst_ratio", 0.0)
    print(f"T-junction violations (>2:1): {tj_bad['count']} (worst ratio={worst:.2f})")
    coverage = beam_inside_coverage(mesh, beam_ref_ls, nx=160, ny=12)
    coverage_inside = beam_inside_coverage(mesh, beam_ref_ls, nx=160, ny=12, inside_only=True)
    if coverage["missing_x"]:
        msg = ", ".join(f"{x:.4f}" for x in coverage["missing_x"][:8])
        if len(coverage["missing_x"]) > 8:
            msg += " ..."
        print(f"[inside-coverage] coverage={coverage['coverage']:.3f}, missing columns at x≈ {msg}")
    else:
        print(f"[inside-coverage] coverage={coverage['coverage']:.3f} (beam fully covered by inside/cut)")
    if coverage_inside["missing_x"]:
        msg = ", ".join(f"{x:.4f}" for x in coverage_inside["missing_x"][:8])
        if len(coverage_inside["missing_x"]) > 8:
            msg += " ..."
        print(f"[inside-only] coverage={coverage_inside['coverage']:.3f}, missing inside columns at x≈ {msg}")
    else:
        print(f"[inside-only] coverage={coverage_inside['coverage']:.3f} (beam fully covered by inside)")
print(
    f"Ghost edges: total={mesh.edge_bitset('ghost').cardinality()}, "
    f"pos={mesh.edge_bitset('ghost_pos').cardinality()}, "
    f"neg={mesh.edge_bitset('ghost_neg').cardinality()}, "
    f"both={mesh.edge_bitset('ghost_both').cardinality()}, "
    f"fluid_ghost(defined_on)={domains['fluid_ghost'].cardinality()}, "
    f"solid_ghost(defined_on)={domains['solid_ghost'].cardinality()}"
)
_log_step("built domain sets / ghost counts")

# Prepare output dir before any early exits
output_dir = ARGS.output_dir
os.makedirs(output_dir, exist_ok=True)

# Fast path: only produce an initial plot and exit (skip JIT/solver setup).
quick_plot_only = (
    ARGS.plot_mesh
    and (ARGS.plot_only or (not ARGS.run_time_stepping and not ARGS.run_fd_check and not ARGS.run_fd_terms))
    and not ARGS.force_full_setup
)

# After your refinement loop:
if ARGS.mesh_diagnostics_enabled:
    diagnoisis = detect_mesh_pathologies(mesh)

    # If internal cracks > 0, do NOT proceed to solver. 
    # The mesh is topologically broken.
    if len(diagnoisis["internal_cracks"]) > 0:
        print("CRITICAL: Mesh has internal cracks. Visualizing crack locations...")
        print(diagnoisis)

    PATHOLOGY_EDGES: list[int] = []
    for _key in ("internal_cracks", "zombie_edges", "zero_length_edges"):
        PATHOLOGY_EDGES.extend(diagnoisis.get(_key, []))
    PATHOLOGY_EDGES = sorted(set(PATHOLOGY_EDGES))
if quick_plot_only:
    import matplotlib.pyplot as plt
    from pycutfem.io.visualization import plot_mesh_2

    fig, ax = plt.subplots(figsize=(10, 8))
    level_set_for_plot = beam_ref_ls if ARGS.plot_levelset else None
    plot_mesh_2(
        mesh,
        level_set=level_set_for_plot,
        plot_nodes=True,
        plot_edges=True,
        elem_tags=True,
        edge_colors=True,
        plot_interface=bool(ARGS.plot_interface_points),
        show=False,
        ax=ax,
        resolution=max(20, int(ARGS.plot_resolution)),
        highlight_edges=PATHOLOGY_EDGES,
    )
    ax.set_title("Initial mesh (plot-only)")
    fname = os.path.join(output_dir, f"mesh_{0:04d}.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    diag = mesh_topology_diagnostics(mesh)
    print(
        f"[mesh-check] missing_side={diag['missing_side']} "
        f"ownerless_edges={diag['ownerless_edges']} "
        f"degenerate_shared={diag['degenerate_shared']}"
    )
    if diag["degenerate_shared"] > 0:
        repaired = repair_degenerate_edges(mesh)
        diag = mesh_topology_diagnostics(mesh)
        print(
            f"[mesh-repair] repaired={repaired} → "
            f"missing_side={diag['missing_side']} "
            f"ownerless_edges={diag['ownerless_edges']} "
            f"degenerate_shared={diag['degenerate_shared']}"
        )
    cov = coverage_diagnostics(mesh, n_samples=60)
    if cov["gaps_x"]:
        print(f"[coverage] gaps at x≈ {', '.join(f'{x:.4f}' for x in cov['gaps_x'][:10])}" + (" ..." if len(cov['gaps_x']) > 10 else ""))
    else:
        print("[coverage] no vertical gaps detected in sampled lines")
    print(f"[plot-only] saved {fname} and exiting (skipped solver setup)")
    if ARGS.plot_show or ARGS.interactive_plot:
        plt.show()
    else:
        plt.close(fig)
    sys.exit(0)

# Mixed element for fluid/solid unknowns
# Default is Taylor hood. Set PYCUTFEM_TAYLOR_HOOD=1 for P^k/P^(k-1).
mixed_element = MixedElement(
    mesh,
    field_specs={
        "u_pos_x": POLY_ORDER,
        "u_pos_y": POLY_ORDER,
        "p_pos_": PRESSURE_ORDER,
        "vs_neg_x": SOLID_ORDER ,
        "vs_neg_y": SOLID_ORDER ,
        "d_neg_x":  SOLID_ORDER ,
        "d_neg_y":  SOLID_ORDER ,
    },
)
dof_handler = DofHandler(mixed_element, method="cg")

# Boundary conditions
def parabolic_inflow(x, y, t=None):
    """
    Parabolic inflow ramped in time:
        v_in(t,0,y) = v_base(y) * 0.5*(1 - cos(pi/2 * t))  for t < 2
                    = v_base(y)                             otherwise
    """
    v_base = 1.5 * 4 * U_MEAN * y * (H - y) / (H**2)
    if t is None:
        return v_base
    if t < 2.0:
        return v_base * 0.5 * (1.0 - math.cos(0.5 * math.pi * t))
    return v_base


bcs: list[BoundaryCondition] = [
    BoundaryCondition("u_pos_x", "dirichlet", "inlet", parabolic_inflow),
    BoundaryCondition("u_pos_y", "dirichlet", "inlet", lambda x, y: 0.0),
    BoundaryCondition("u_pos_x", "dirichlet", "walls", lambda x, y: 0.0),
    BoundaryCondition("u_pos_y", "dirichlet", "walls", lambda x, y: 0.0),
    BoundaryCondition("u_pos_x", "dirichlet", "cylinder", lambda x, y: 0.0),
    BoundaryCondition("u_pos_y", "dirichlet", "cylinder", lambda x, y: 0.0),
]

# Pressure pin to remove nullspace
if PIN_PRESSURE:
    pin_tag = "pressure_pin"
    dof_handler.tag_dof_by_locator(
        pin_tag,
        "p_pos_",
        locator=lambda x, y: abs(x - L) <= 1.0e-9 and abs(y - 0.5 * H) <= 1.0e-3,
        find_first=True,
    )
    bcs.append(BoundaryCondition("p_pos_", "dirichlet", pin_tag, lambda x, y: 0.0))

# Clamp beam at the circle interface
beam_clamp_tag = "beam_root"
beam_root_locator = _beam_root_locator(mesh, beam_ref_center, beam_ref_length, beam_ref_height)
beam_root_fields = ["vs_neg_x", "vs_neg_y", "d_neg_x", "d_neg_y"]
_tag_beam_root_from_cylinder(dof_handler, mesh, beam_root_locator, beam_root_fields, tag=beam_clamp_tag)
_tag_beam_root_from_levelset(dof_handler, beam_ref_ls, beam_root_fields, tag=beam_clamp_tag)
beam_root_dofs = len(dof_handler.dof_tags.get(beam_clamp_tag, set()))
if beam_root_dofs == 0:
    print("[warn] Beam root DOF tagging found no DOFs; check BEAM_ROOT_DOF_TOL or beam geometry.")
else:
    print(f"Beam root DOFs tagged: {beam_root_dofs}")
for field in beam_root_fields:
    bcs.append(BoundaryCondition(field, "dirichlet", beam_clamp_tag, lambda x, y: 0.0))

bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]

print(f"Interface edges: {mesh.edge_bitset('interface').cardinality()}")
print(f"Cut elements:    {mesh.element_bitset('cut').cardinality()}")

# -----------------------------------------------------------------------------
# Function spaces and unknowns
# -----------------------------------------------------------------------------
velocity_fluid_space = FunctionSpace(name="velocity_fluid", field_names=["u_pos_x", "u_pos_y"], dim=1, side="+")
pressure_fluid_space = FunctionSpace(name="pressure_fluid", field_names=["p_pos_"], dim=0, side="+")
velocity_solid_space = FunctionSpace(name="velocity_solid", field_names=["vs_neg_x", "vs_neg_y"], dim=1, side="-")
displacement_space = FunctionSpace(name="displacement", field_names=["d_neg_x", "d_neg_y"], dim=1, side="-")

du_f = VectorTrialFunction(space=velocity_fluid_space, dof_handler=dof_handler)
dp_f = TrialFunction(name="trial_pressure_fluid", field_name="p_pos_", dof_handler=dof_handler, side="+")
du_s = VectorTrialFunction(space=velocity_solid_space, dof_handler=dof_handler)
ddisp_s = VectorTrialFunction(space=displacement_space, dof_handler=dof_handler)
test_vel_f = VectorTestFunction(space=velocity_fluid_space, dof_handler=dof_handler)
test_q_f = TestFunction(name="test_pressure_fluid", field_name="p_pos_", dof_handler=dof_handler, side="+")
test_vel_s = VectorTestFunction(space=velocity_solid_space, dof_handler=dof_handler)
test_disp_s = VectorTestFunction(space=displacement_space, dof_handler=dof_handler)

uf_k = VectorFunction(name="u_f_k", field_names=["u_pos_x", "u_pos_y"], dof_handler=dof_handler, side="+")
pf_k = Function(name="p_f_k", field_name="p_pos_", dof_handler=dof_handler, side="+")
uf_n = VectorFunction(name="u_f_n", field_names=["u_pos_x", "u_pos_y"], dof_handler=dof_handler, side="+")
pf_n = Function(name="p_f_n", field_name="p_pos_", dof_handler=dof_handler, side="+")
us_k = VectorFunction(name="u_s_k", field_names=["vs_neg_x", "vs_neg_y"], dof_handler=dof_handler, side="-")
us_n = VectorFunction(name="u_s_n", field_names=["vs_neg_x", "vs_neg_y"], dof_handler=dof_handler, side="-")
disp_k = VectorFunction(name="disp_k", field_names=["d_neg_x", "d_neg_y"], dof_handler=dof_handler, side="-")
disp_n = VectorFunction(name="disp_n", field_names=["d_neg_x", "d_neg_y"], dof_handler=dof_handler, side="-")

for func in [uf_k, pf_k, uf_n, pf_n, us_k, us_n, disp_k, disp_n]:
    func.nodal_values.fill(0.0)

# IMPORTANT: time-dependent BCs must be frozen; otherwise `parabolic_inflow(x,y,t=None)`
# is evaluated with t=None (i.e. *full* inflow, no ramp), which corrupts restart dumps
# and can destabilize early transient steps.
bcs_t0 = NewtonSolver._freeze_bcs(bcs, 0.0)
dof_handler.apply_bcs(bcs_t0, uf_k, pf_k, us_k, disp_k)
dof_handler.apply_bcs(bcs_t0, uf_n, pf_n, us_n, disp_n)

# -----------------------------------------------------------------------------
# Measures and stabilization weights
# -----------------------------------------------------------------------------
qvol = 6
dx_fluid = dx(
    defined_on=domains["fluid_interface"],
    level_set=ls_beam,
    metadata={"q": qvol, "side": "+"},
)
# Solid integrals are evaluated in the current (Eulerian) frame because the
# level set is updated with the present displacement.
dx_solid = dx(
    defined_on=domains["solid_interface"],
    level_set=ls_beam,
    metadata={"q": qvol, "side": "-"},
)
dx_fluid_cut = dx(
    defined_on=domains["cut_domain"],
    level_set=ls_beam,
    metadata={"q": qvol, "side": "+"},
)
dx_solid_cut = dx(
    defined_on=domains["cut_domain"],
    level_set=ls_beam,
    metadata={"q": qvol, "side": "-"},
)
dΓ = dInterface(
    defined_on=domains["cut_interface"],
    level_set=ls_beam,
    metadata={"q": qvol + 2, "derivs": {(0, 0), (0, 1), (1, 0)}, "linear_interface": USE_LINEAR_INTERFACE},
)

use_facet_patch_ghost = _env_bool(
    "USE_FACET_PATCH_GHOST", str(getattr(dof_handler, "method", "")).lower() == "cg"
)
ghost_measure = dFacetPatch if use_facet_patch_ghost else dGhost

dG_fluid = ghost_measure(
    defined_on=domains["fluid_ghost"],
    level_set=ls_beam,
    metadata={"q": qvol, "derivs": {(0, 1), (1, 0)}},
)
dG_solid = ghost_measure(
    defined_on=domains["solid_ghost"],
    level_set=ls_beam,
    metadata={"q": qvol, "derivs": {(0, 1), (1, 0)}},
)
dS_fluid_quad = ds(
    defined_on=domains["fluid_quad_edges"],
    level_set=ls_beam,
    metadata={"q": qvol, "derivs": {(0, 1), (1, 0)}},
)

cell_h = CellDiameter()
beta_N = Constant(BETA_PENALTY * POLY_ORDER * (POLY_ORDER + 1))

# Cut-ratio floor used for:
# - sliver weights (refresh_sliver_weights)
# - kappa weights in Nitsche averages
#
# Using the same floor as the sliver logic avoids artificially capping the
# stabilization strength on extreme slivers (raw θ ≪ 1e-3) while still
# preventing divide-by-zero.
theta_min = max(float(SLIVER_THETAMIN), 1.0e-12)
theta_pos_raw_vals = hansbo_cut_ratio(mesh, ls_beam, side="+")
theta_neg_raw_vals = hansbo_cut_ratio(mesh, ls_beam, side="-")
theta_pos_vals = np.clip(theta_pos_raw_vals, theta_min, 1.0)
theta_neg_vals = np.clip(theta_neg_raw_vals, theta_min, 1.0)
w_pos_vals = np.ones_like(theta_pos_vals)
w_neg_vals = np.ones_like(theta_neg_vals)
refresh_sliver_weights(
    mesh,
    theta_pos_vals,
    theta_neg_vals,
    w_pos_vals,
    w_neg_vals,
    theta0=SLIVER_THETA0,
    p=SLIVER_P,
    wmax=SLIVER_WMAX,
    thetamin=SLIVER_THETAMIN,
    smooth=SLIVER_SMOOTH,
)

theta_pos_cell = ElementWiseConstant(theta_pos_vals)
theta_neg_cell = ElementWiseConstant(theta_neg_vals)
theta_sum = Pos(theta_pos_cell) + Neg(theta_neg_cell) + Constant(1.0e-12)
kappa_pos = Pos(theta_pos_cell) / theta_sum
kappa_neg = Neg(theta_neg_cell) / theta_sum

w_pos_cell = ElementWiseConstant(w_pos_vals)
w_neg_cell = ElementWiseConstant(w_neg_vals)
w_gp_fluid = Constant(0.5) * (Pos(w_pos_cell) + Neg(w_pos_cell))
w_gp_solid = Constant(0.5) * (Pos(w_neg_cell) + Neg(w_neg_cell))
w_gp_ifc = Constant(0.5) * (w_gp_fluid + w_gp_solid)
log_sliver_stats(
    mesh,
    theta_pos_vals,
    theta_neg_vals,
    w_pos_vals,
    w_neg_vals,
    theta0=SLIVER_THETA0,
)
retag_inactive(dof_handler, theta_neg=theta_neg_raw_vals, solid_cut_drop=SOLID_CUT_DROP)

use_restricted_forms = USE_RESTRICTED_FORMS
if use_restricted_forms:
    du_f_R = restrict(du_f, domains["has_pos"])
    dp_f_R = restrict(dp_f, domains["has_pos"])
    test_vel_f_R = restrict(test_vel_f, domains["has_pos"])
    test_q_f_R = restrict(test_q_f, domains["has_pos"])
    uf_k_R = restrict(uf_k, domains["has_pos"])
    uf_n_R = restrict(uf_n, domains["has_pos"])
    pf_k_R = restrict(pf_k, domains["has_pos"])
    pf_n_R = restrict(pf_n, domains["has_pos"])

    du_s_R = restrict(du_s, domains["has_neg"])
    ddisp_s_R = restrict(ddisp_s, domains["has_neg"])
    test_vel_s_R = restrict(test_vel_s, domains["has_neg"])
    test_disp_s_R = restrict(test_disp_s, domains["has_neg"])
    us_k_R = restrict(us_k, domains["has_neg"])
    us_n_R = restrict(us_n, domains["has_neg"])
    disp_k_R = restrict(disp_k, domains["has_neg"])
    disp_n_R = restrict(disp_n, domains["has_neg"])
else:
    du_f_R = du_f
    dp_f_R = dp_f
    test_vel_f_R = test_vel_f
    test_q_f_R = test_q_f
    uf_k_R = uf_k
    uf_n_R = uf_n
    pf_k_R = pf_k
    pf_n_R = pf_n
    du_s_R = du_s
    ddisp_s_R = ddisp_s
    test_vel_s_R = test_vel_s
    test_disp_s_R = test_disp_s
    us_k_R = us_k
    us_n_R = us_n
    disp_k_R = disp_k
    disp_n_R = disp_n

I2 = Identity(2)
n = FacetNormal()

# -----------------------------------------------------------------------------
# Constitutive helpers
# -----------------------------------------------------------------------------
def epsilon_f(u):
    return 0.5 * (grad(u) + grad(u).T)


def epsilon_s_linear_L(disp, disp_k):
    return 0.5 * (grad(disp) + grad(disp).T + dot(grad(disp).T, grad(disp_k)) + dot(grad(disp_k).T, grad(disp)))


def epsilon_s_linear_R(disp_k):
    return 0.5 * (grad(disp_k) + grad(disp_k).T + dot(grad(disp_k).T, grad(disp_k)))


def sigma_s_linear_weak_linear_a(ddisp, disp_k, grad_v_test):
    eps = epsilon_s_linear_L(ddisp, disp_k)
    return 2.0 * Constant(MU_S) * inner(eps, grad_v_test) + Constant(LAMBDA_S) * trace(eps) * trace(grad_v_test)


def sigma_s_linear_weak_nonlinear_residual(disp_k, grad_v_test):
    eps = epsilon_s_linear_R(disp_k)
    return 2.0 * Constant(MU_S) * inner(eps, grad_v_test) + Constant(LAMBDA_S) * trace(eps) * trace(grad_v_test)


def traction_fluid_primal(u_vec, p_scal):
    return 2.0 * Constant(MU_F) * dot(epsilon_f(u_vec), n) - p_scal * n


def traction_fluid_adjoint(v_vec, q_scal):
    return 2.0 * Constant(MU_F) * dot(epsilon_f(v_vec), n) + q_scal * n

def F_of(d):
    # Eulerian displacement: X = x - d(x) ⇒ F = (I - ∇d)^{-1}
    return inv(I2 - grad(d))


def C_of(F):
    return dot(F.T, F)


def E_of(F):
    return 0.5 * (C_of(F) - I2)


def S_stvk(E):
    return Constant(LAMBDA_S) * trace(E) * I2 + Constant(2.0 * MU_S) * E


def sigma_s_linear(d):
    eps = 0.5 * (grad(d) + grad(d).T)
    return Constant(2.0 * MU_S) * eps + Constant(LAMBDA_S) * trace(eps) * I2


def dsigma_s_linear(delta_d):
    eps = 0.5 * (grad(delta_d) + grad(delta_d).T)
    return Constant(2.0 * MU_S) * eps + Constant(LAMBDA_S) * trace(eps) * I2


def sigma_s_nonlinear(d):
    if USE_LINEAR_SOLID:
        return sigma_s_linear(d)
    F = F_of(d)
    E = E_of(F)
    S = S_stvk(E)
    J = det(F)
    return (1.0 / J) * dot(dot(F, S), F.T)

def _interface_length(mesh: Mesh) -> float:
    """Total length of edges tagged as interface."""
    length = 0.0
    for gid in mesh.edge_bitset("interface").to_indices():
        e = mesh.edge(int(gid))
        p0, p1 = mesh.nodes_x_y_pos[list(e.nodes)]
        length += float(np.linalg.norm(p1 - p0))
    return length


def _eval_scalar_at_point(
    dh: DofHandler,
    mesh: Mesh,
    f_scalar: Function,
    point: tuple[float, float],
    *,
    elem_tags: set[str] | None = None,
) -> float:
    """
    Evaluate a scalar Function at a physical point using element search
    and basis evaluation (robust to mixed-order layouts).
    """
    xy = np.asarray(point, float)
    for e in mesh.elements_list:
        if elem_tags is not None and e.tag not in elem_tags:
            continue
        verts = mesh.nodes_x_y_pos[list(e.nodes)]
        if not (verts[:, 0].min() - 1e-12 <= xy[0] <= verts[:, 0].max() + 1e-12 and
                verts[:, 1].min() - 1e-12 <= xy[1] <= verts[:, 1].max() + 1e-12):
            continue
        try:
            xi, eta = transform.inverse_mapping(mesh, e.id, xy)
        except Exception:
            continue
        if not (-1.0001 <= xi <= 1.0001 and -1.0001 <= eta <= 1.0001):
            continue
        me = dh.mixed_element
        fld = f_scalar.field_name
        phi = me.basis(fld, float(xi), float(eta))[me.slice(fld)]
        gdofs = dh.element_maps[fld][e.id]
        vals = f_scalar.get_nodal_values(gdofs)
        return float(phi @ vals)
    return float("nan")


def _eval_vector_at_point(
    dh: DofHandler,
    mesh: Mesh,
    f_vec: VectorFunction,
    point: tuple[float, float],
    *,
    elem_tags: set[str] | None = None,
) -> np.ndarray:
    """Evaluate a 2D VectorFunction at a physical point."""
    vals = [_eval_scalar_at_point(dh, mesh, comp, point, elem_tags=elem_tags) for comp in f_vec.components]
    return np.asarray(vals, dtype=float)

# Cache for the previous tip position (important for Eulerian/ref-map tracking)
_TIP_POS_CACHE: np.ndarray | None = None

def _material_point_position_ref_map(
    dh: DofHandler,
    mesh: Mesh,
    disp: VectorFunction,
    X_ref: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    max_iter: int = 20,
    tol: float = 1e-12,
    damping: float = 1.0,
) -> np.ndarray:
    """Track a solid material point in the Eulerian *reference-map* formulation.

    In this code, the solid displacement is Eulerian: X(x,t) = x - d(x,t).
    A material point with reference coordinate X_ref therefore satisfies
        X(x,t) = X_ref  ⇔  x - d(x,t) = X_ref  ⇔  x = X_ref + d(x,t).

    We solve this fixed-point relation by iteration, starting from x0 (typically the
    previous time-step tip position).
    """

    X_ref = np.asarray(X_ref, dtype=float)
    x = np.asarray(X_ref if x0 is None else x0, dtype=float).copy()

    # Evaluate displacement from the solid side (inside/cut) to avoid picking a fluid cell
    # when the query point is on/near the interface.
    solid_tags = {"inside", "cut"}

    for _ in range(int(max_iter)):
        d = _eval_vector_at_point(dh, mesh, disp, (float(x[0]), float(x[1])), elem_tags=solid_tags)
        if not np.all(np.isfinite(d)):
            # Fallback: try evaluating at the reference point.
            d = _eval_vector_at_point(dh, mesh, disp, (float(X_ref[0]), float(X_ref[1])), elem_tags=solid_tags)
        x_new = X_ref + d
        if damping != 1.0:
            x_new = x + float(damping) * (x_new - x)
        if float(np.linalg.norm(x_new - x)) < tol:
            x = x_new
            break
        x = x_new
    return x


def _tip_position(dh: DofHandler, mesh: Mesh, disp: VectorFunction, ref_tip: np.ndarray) -> np.ndarray:
    """Current position of the beam tip (Eulerian / reference-map consistent).

    NOTE: This project uses the Eulerian reference-map displacement d(x) with
        X(x) = x - d(x)
    (see also the level-set update φ(x) = φ_ref(x - d(x)) and F = (I - ∇d)^{-1}).
    Therefore, the standard Lagrangian formula X_ref + u(X_ref) is *not* valid.
    """

    global _TIP_POS_CACHE
    x0 = _TIP_POS_CACHE
    x_tip = _material_point_position_ref_map(dh, mesh, disp, ref_tip, x0=x0)
    _TIP_POS_CACHE = x_tip.copy()
    return x_tip


def dsigma_s(d_ref, delta_d):
    if USE_LINEAR_SOLID:
        return dsigma_s_linear(delta_d)
    Fk = F_of(d_ref)
    Ek = E_of(Fk)
    Sk = S_stvk(Ek)
    # δF = F * ∇(δd) * F  (since F = (I - ∇d)^{-1})
    dF = dot(Fk, dot(grad(delta_d), Fk))
    dE = 0.5 * (dot(dF.T, Fk) + dot(Fk.T, dF))
    dS = Constant(LAMBDA_S) * trace(dE) * I2 + Constant(2.0 * MU_S) * dE
    Jk = det(Fk)
    dJ = Jk * trace(dot(grad(delta_d), Fk))
    term = dot(dF, dot(Sk, Fk.T)) + dot(Fk, dot(dS, Fk.T)) + dot(Fk, dot(Sk, dF.T))
    return (1.0 / Jk) * term - (dJ / Jk) * sigma_s_nonlinear(d_ref)


def traction_solid_R(d):
    return dot(sigma_s_nonlinear(d), n)


def traction_solid_L(delta_d, d_ref):
    return dot(dsigma_s(d_ref, delta_d), n)


def delta_E_GreenLagrange(w, u_ref):
    F_ref = F_of(u_ref)
    grad_w = grad(w)
    return Constant(0.5) * (dot(grad_w.T, F_ref) + dot(F_ref.T, grad_w))


def grad_inner_jump(u, v):
    a = dot(jump(grad(u)), n)
    b = dot(jump(grad(v)), n)
    return inner(a, b)


# -----------------------------------------------------------------------------
# Weak forms
# -----------------------------------------------------------------------------
dt = Constant(DT)
dt._jit_name = "dt"
theta = Constant(float(ARGS.theta))
rho_f_const = Constant(RHO_F)
rho_s_const = Constant(RHO_S)
mu_f_const = Constant(MU_F)
mu_s_const = Constant(MU_S)
lambda_s_const = Constant(LAMBDA_S)

du_f_ifc = du_f_R
dp_f_ifc = dp_f_R
test_vel_f_ifc = test_vel_f_R
test_q_f_ifc = test_q_f_R
uf_k_ifc = uf_k_R
pf_k_ifc = pf_k_R
du_s_ifc = du_s_R
ddisp_s_ifc = ddisp_s_R
test_vel_s_ifc = test_vel_s_R
us_k_ifc = us_k_R
disp_k_ifc = disp_k_R

jump_vel_trial = Pos(du_f_ifc) - Neg(du_s_ifc)
jump_vel_test = Pos(test_vel_f_ifc) - Neg(test_vel_s_ifc)
jump_vel_res = Pos(uf_k_ifc) - Neg(us_k_ifc)
jump_test_f = Pos(test_vel_f_ifc)
jump_test_s = Neg(test_vel_s_ifc)

solid_adv_vel = us_n_R if SOLID_ADVECT_LAGGED else us_k_R

avg_flux_fluid_trial = kappa_pos * traction_fluid_primal(Pos(du_f_ifc), Pos(dp_f_ifc))
avg_flux_fluid_test =  kappa_pos * traction_fluid_adjoint(Pos(test_vel_f_ifc), Pos(test_q_f_ifc))
avg_flux_fluid_res =   kappa_pos * traction_fluid_primal(Pos(uf_k_ifc), Pos(pf_k_ifc))

# Sign convention (single normal `n` used on both sides):
# - `n` points from solid (−) → fluid (+).
# - With outward normals n_f = −n (fluid) and n_s = +n (solid), the combined
#   IBP boundary contribution on Γ for the volume terms used below is
#       B_Γ(v_f,v_s) = ∫_Γ (σ_f n_f)·v_f + (σ_s n_s)·v_s
#                  = ∫_Γ (−t_f·v_f + t_s·v_s),
#   where we define tractions using the same `n`:  t_f = σ_f n,  t_s = σ_s n.
# - Under the physical dynamic condition t_f = t_s = t this becomes
#       B_Γ = −∫_Γ t·(v_f − v_s) = −∫_Γ t·jump(v),
#   where jump(v) := v_f − v_s.
# - A consistent Nitsche term must therefore be
#       N_Γ(u;v) = +∫_Γ {t(u)}·jump(v),
#   with the *sum* average {t} = κ⁺ t_f + κ⁻ t_s (κ⁺+κ⁻=1) so that on the exact
#   interface condition {t}=t and B_Γ + N_Γ = 0 for arbitrary discontinuous tests.
# Verified by `tests/ufl/test_fsi_eulerian_interface_traction.py` which assembles
# `B_Γ + R_int` and checks it is ≈0 iff the manufactured fields satisfy t_f=t_s.
avg_flux_solid_trial = kappa_neg * traction_solid_L(Neg(ddisp_s_ifc), Neg(disp_k_ifc))
_sym_ref = disp_n if (SOLID_SYM_NITSCHE_LAGGED and not USE_LINEAR_SOLID) else disp_k_ifc
avg_flux_solid_test =  kappa_neg * traction_solid_L(Neg(test_vel_s_ifc), Neg(_sym_ref))
avg_flux_solid_res =   kappa_neg * traction_solid_R(Neg(disp_k_ifc))

s_nitsche_value = float(ARGS.s_nitsche_value)
s_nitsche = Constant(s_nitsche_value)   # 1 = symmetric, 0 = incomplete, -1 = skew-symmetric

# With jump_test_f = v_f and jump_test_s = v_s (note `v_s` already carries the Neg())
# the consistency term +∫ {t}·jump(v) expands to:
#   +∫ {t}·v_f − ∫ {t}·v_s,
# which is implemented below as "+ dot(avg_flux, jump_test_f) - dot(avg_flux, jump_test_s)".
J_int_fluid = (dot(avg_flux_fluid_trial, jump_test_f)) * dΓ - (dot(avg_flux_fluid_trial, jump_test_s)) * dΓ
R_int_fluid = (dot(avg_flux_fluid_res, jump_test_f)) * dΓ   - (dot(avg_flux_fluid_res, jump_test_s)) * dΓ
J_int_solid = (dot(avg_flux_solid_trial, jump_test_f)) * dΓ - (dot(avg_flux_solid_trial, jump_test_s)) * dΓ
R_int_solid = (dot(avg_flux_solid_res, jump_test_f)) * dΓ   - (dot(avg_flux_solid_res, jump_test_s)) * dΓ
nitsche_pen_scale = mu_f_const / cell_h
if USE_DT_SCALED_NITSCHE_PENALTY:
    nitsche_pen_scale = nitsche_pen_scale + (rho_f_const * cell_h / dt)
pen_weight = w_gp_ifc if PYCUTFEM_INTERFACE_PENALTY_SLIVER_WEIGHT else Constant(1.0)
J_int_pen = beta_N * nitsche_pen_scale * pen_weight * dot(jump_vel_trial, jump_vel_test) * dΓ
R_int_pen = beta_N * nitsche_pen_scale * pen_weight * dot(jump_vel_res, jump_vel_test) * dΓ

J_int = J_int_fluid + J_int_solid + J_int_pen
R_int = R_int_fluid + R_int_solid + R_int_pen

J_int_sym_fluid = None
J_int_sym_solid = None
R_int_sym_fluid = None
R_int_sym_solid = None
if s_nitsche_value != 0.0:
    J_int_sym_fluid = (s_nitsche * dot(avg_flux_fluid_test, jump_vel_trial)) * dΓ
    J_int_sym_solid = (s_nitsche * dot(avg_flux_solid_test, jump_vel_trial)) * dΓ
    R_int_sym_fluid = (s_nitsche * dot(avg_flux_fluid_test, jump_vel_res)) * dΓ
    R_int_sym_solid = (s_nitsche * dot(avg_flux_solid_test, jump_vel_res)) * dΓ
    J_int = J_int + J_int_sym_fluid + J_int_sym_solid
    R_int = R_int + R_int_sym_fluid + R_int_sym_solid



def _conv_adv(u, v):
    return dot(dot(grad(u), u), v)


def _conv_adv_jac(u_ref, du, v):
    return dot(dot(grad(u_ref), du), v) + dot(dot(grad(du), u_ref), v)


def _conv_skew(u, v):
    return Constant(0.5) * (dot(dot(grad(u), u), v) - dot(dot(grad(v), u), u))


def _conv_skew_jac(u_ref, du, v):
    return Constant(0.5) * (
        dot(dot(grad(u_ref), du), v)
        + dot(dot(grad(du), u_ref), v)
        - dot(dot(grad(v), du), u_ref)
        - dot(dot(grad(v), u_ref), du)
    )


if USE_SKEW_CONVECTION:
    conv_f_jac = _conv_skew_jac(uf_k_R, du_f_R, test_vel_f_R)
    conv_f_k = _conv_skew(uf_k_R, test_vel_f_R)
    conv_f_n = _conv_skew(uf_n_R, test_vel_f_R)
else:
    conv_f_jac = _conv_adv_jac(uf_k_R, du_f_R, test_vel_f_R)
    conv_f_k = _conv_adv(uf_k_R, test_vel_f_R)
    conv_f_n = _conv_adv(uf_n_R, test_vel_f_R)

a_vol_f = (
    rho_f_const / dt * dot(du_f_R, test_vel_f_R)
    + theta * rho_f_const * conv_f_jac
    + theta * mu_f_const * inner(grad(du_f_R), grad(test_vel_f_R))
    - dp_f_R * div(test_vel_f_R)
    + test_q_f_R * div(du_f_R)
) * dx_fluid

r_vol_f = (
    rho_f_const * dot(uf_k_R - uf_n_R, test_vel_f_R) / dt
    + theta * rho_f_const * conv_f_k
    + (1 - theta) * rho_f_const * conv_f_n
    + theta * mu_f_const * inner(grad(uf_k_R), grad(test_vel_f_R))
    + (1 - theta) * mu_f_const * inner(grad(uf_n_R), grad(test_vel_f_R))
    - pf_k_R * div(test_vel_f_R)
    + test_q_f_R * div(uf_k_R)
) * dx_fluid

sigma_s_k = sigma_s_nonlinear(disp_k_R)  # Cauchy stress in current frame
sigma_s_n = sigma_s_nonlinear(disp_n_R)
dsigma_s_k = dsigma_s(disp_k_R, ddisp_s_R)

a_half = Constant(0.5)

def _advect(w, u, v):
    return dot(dot(grad(u), w), v)


def _advect_jac_w_fixed(w, du, v):
    return dot(dot(grad(du), w), v)


def _advect_skew(w, u, v):
    return a_half * (dot(dot(grad(u), w), v) - dot(dot(grad(v), w), u))


def _advect_skew_jac_w_fixed(w, du, v):
    return a_half * (dot(dot(grad(du), w), v) - dot(dot(grad(v), w), du))


def _advect_skew_div(w, u, v):
    return a_half * (dot(dot(grad(u), w), v) - dot(dot(grad(v), w), u) + div(w) * dot(u, v))


def _advect_skew_div_jac_w_fixed(w, du, v):
    return a_half * (dot(dot(grad(du), w), v) - dot(dot(grad(v), w), du) + div(w) * dot(du, v))


if USE_SKEW_CONVECTION:
    if SOLID_ADVECT_LAGGED:
        if SOLID_SKEW_ADD_DIV:
            adv_s_jac = _advect_skew_div_jac_w_fixed(solid_adv_vel, du_s_R, test_vel_s_R)
            adv_s_k = _advect_skew_div(solid_adv_vel, us_k_R, test_vel_s_R)
            adv_s_n = _advect_skew_div(us_n_R, us_n_R, test_vel_s_R)
        else:
            adv_s_jac = _advect_skew_jac_w_fixed(solid_adv_vel, du_s_R, test_vel_s_R)
            adv_s_k = _advect_skew(solid_adv_vel, us_k_R, test_vel_s_R)
            adv_s_n = _advect_skew(us_n_R, us_n_R, test_vel_s_R)
    else:
        adv_s_jac = _conv_skew_jac(us_k_R, du_s_R, test_vel_s_R)
        adv_s_k = _conv_skew(us_k_R, test_vel_s_R)
        adv_s_n = _conv_skew(us_n_R, test_vel_s_R)
else:
    if SOLID_ADVECT_LAGGED:
        adv_s_jac = _advect_jac_w_fixed(solid_adv_vel, du_s_R, test_vel_s_R)
        adv_s_k = _advect(solid_adv_vel, us_k_R, test_vel_s_R)
        adv_s_n = _advect(us_n_R, us_n_R, test_vel_s_R)
    else:
        adv_s_jac = dot(dot(grad(us_k_R), du_s_R), test_vel_s_R) + dot(dot(grad(du_s_R), us_k_R), test_vel_s_R)
        adv_s_k = _conv_adv(us_k_R, test_vel_s_R)
        adv_s_n = _conv_adv(us_n_R, test_vel_s_R)

a_vol_s = (
    rho_s_const * dot(du_s_R, test_vel_s_R) / dt
    + theta * inner(dsigma_s_k, grad(test_vel_s_R))
    + (rho_s_const * theta * adv_s_jac)
) * dx_solid
r_vol_s = (
    rho_s_const * dot(us_k_R - us_n_R, test_vel_s_R) / dt
    + theta * inner(sigma_s_k, grad(test_vel_s_R))
    + (1 - theta) * inner(sigma_s_n, grad(test_vel_s_R))
    + rho_s_const * (theta * adv_s_k + (1 - theta) * adv_s_n)
) * dx_solid



if USE_SKEW_CONVECTION and SOLID_ADVECT_LAGGED:
    if SOLID_SKEW_ADD_DIV:
        adv_disp_jac = _advect_skew_div_jac_w_fixed(solid_adv_vel, ddisp_s_R, test_disp_s_R)
        adv_disp_k = _advect_skew_div(solid_adv_vel, disp_k_R, test_disp_s_R)
        adv_disp_n = _advect_skew_div(us_n_R, disp_n_R, test_disp_s_R)
    else:
        adv_disp_jac = _advect_skew_jac_w_fixed(solid_adv_vel, ddisp_s_R, test_disp_s_R)
        adv_disp_k = _advect_skew(solid_adv_vel, disp_k_R, test_disp_s_R)
        adv_disp_n = _advect_skew(us_n_R, disp_n_R, test_disp_s_R)
else:
    adv_disp_jac = (
        dot(dot(grad(ddisp_s_R), solid_adv_vel), test_disp_s_R)
        if SOLID_ADVECT_LAGGED
        else (
            dot(dot(grad(ddisp_s_R), us_k_R), test_disp_s_R)
            + dot(dot(grad(disp_k_R), du_s_R), test_disp_s_R)
        )
    )
    adv_disp_k = dot(dot(grad(disp_k_R), solid_adv_vel), test_disp_s_R)
    adv_disp_n = dot(dot(grad(disp_n_R), us_n_R), test_disp_s_R)

a_svc = (
    dot(ddisp_s_R, test_disp_s_R) / dt
    - theta * dot(du_s_R, test_disp_s_R)
    + theta * adv_disp_jac
) * dx_solid
if SOLID_KINEMATIC_STAB > 0.0:
    alpha_s = Constant(float(SOLID_KINEMATIC_STAB))
    s_exp = float(SOLID_KINEMATIC_STAB_EXP)
    a_svc = a_svc + (-alpha_s * cell_h**s_exp * inner(grad(du_s_R), grad(test_disp_s_R))) * dx_solid
# Kinematic constraint with advected displacement in Eulerian frame
r_svc = (
    dot(disp_k_R - disp_n_R, test_disp_s_R) / dt
    - theta * dot(us_k_R, test_disp_s_R)
    - (1 - theta) * dot(us_n_R, test_disp_s_R)
    + theta * adv_disp_k
    + (1 - theta) * adv_disp_n
) * dx_solid
if SOLID_KINEMATIC_STAB > 0.0:
    r_svc = r_svc + (-alpha_s * cell_h**s_exp * inner(grad(us_k_R), grad(test_disp_s_R))) * dx_solid

penalty_val = float(os.getenv("PENALTY_VAL", "0.0"))
penalty_grad = float(os.getenv("PENALTY_GRAD", "0.0"))
# Keep the scalar Constants around so we can scale them (e.g. on Newton failure)
# without regenerating forms/kernels.
gamma_v_f0 = Constant(penalty_val * POLY_ORDER**2)
gamma_p_f0 = Constant(penalty_val * PRESSURE_ORDER)
gamma_v_s0 = Constant(penalty_val * POLY_ORDER**2)
gamma_disp_s0 = Constant(penalty_grad * POLY_ORDER**2)
gamma_v_f_mass0 = Constant(float(PYCUTFEM_FLUID_VEL_GHOST_MASS))
gamma_v_s_mass0 = Constant(float(SOLID_VEL_GHOST_MASS))
gamma_sliver_mass_f0 = Constant(float(PYCUTFEM_SLIVER_MASS_FLUID))
gamma_sliver_mass_s0 = Constant(float(PYCUTFEM_SLIVER_MASS_SOLID))
gamma_v_f = gamma_v_f0 * w_gp_fluid
gamma_p_f = gamma_p_f0 * w_gp_fluid
gamma_v_s = gamma_v_s0 * w_gp_solid
gamma_disp_s = gamma_disp_s0 * w_gp_solid
gamma_v_s_mass = gamma_v_s_mass0
gamma_v_f_mass = gamma_v_f_mass0
gamma_sliver_mass_f = gamma_sliver_mass_f0
gamma_sliver_mass_s = gamma_sliver_mass_s0
gamma_p_stab_jump0 = Constant(PYCUTFEM_PRESSURE_STAB_JUMP)
gamma_p_stab_avg0 = Constant(PYCUTFEM_PRESSURE_STAB_AVG)
gamma_p_stab_jump = gamma_p_stab_jump0 * w_gp_fluid
gamma_p_stab_avg = gamma_p_stab_avg0 * w_gp_fluid
solid_reg_eps = Constant(float(os.getenv("SOLID_REG_EPS", "1e-6")))


def g_v_f(gamma, phi_1, phi_2):
    return gamma * (cell_h * grad_inner_jump(phi_1, phi_2))


def g_v_f_inertia(gamma, phi_1, phi_2):
    return gamma * ((cell_h**3.0 / dt) * grad_inner_jump(phi_1, phi_2))


def g_v_f_mass(gamma, phi_1, phi_2):
    return gamma * (cell_h * inner(jump(phi_1), jump(phi_2)))


def g_p(gamma, phi_1, phi_2):
    return gamma * (cell_h**3.0 * grad_inner_jump(phi_1, phi_2))


def grad_inner_jump_full(u, v):
    return inner(jump(grad(u)), jump(grad(v)))


def grad_inner_avg(u, v):
    # Average of a scalar quantity across the edge: 0.5*(pos + neg).
    return Constant(0.5) * (Pos(inner(grad(u), grad(v))) + Neg(inner(grad(u), grad(v))))


def g_v_s(gamma, phi_1, phi_2):
    scale = cell_h**3.0
    if SOLID_VEL_GHOST_DT_SCALE:
        scale = scale / dt
    return gamma * (scale * grad_inner_jump(phi_1, phi_2))


def g_v_s_mass(gamma, phi_1, phi_2):
    return gamma * (cell_h * inner(jump(phi_1), jump(phi_2)))


def g_disp_s(gamma, phi_1, phi_2):
    return gamma * (cell_h * grad_inner_jump(phi_1, phi_2))


a_stab_fluid = (
    Constant(2.0) * mu_f_const * g_v_f(gamma_v_f, du_f_R, test_vel_f_R)
    + (rho_f_const / dt) * g_v_f_mass(gamma_v_f_mass, du_f_R, test_vel_f_R)
    + g_p(gamma_p_f, dp_f_R, test_q_f_R)
)
r_stab_fluid = (
    Constant(2.0) * mu_f_const * g_v_f(gamma_v_f, uf_k_R, test_vel_f_R)
    + (rho_f_const / dt) * g_v_f_mass(gamma_v_f_mass, uf_k_R, test_vel_f_R)
    + g_p(gamma_p_f, pf_k_R, test_q_f_R)
)
if FLUID_VEL_GHOST_INERTIA:
    a_stab_fluid = a_stab_fluid + rho_f_const * g_v_f_inertia(gamma_v_f, du_f_R, test_vel_f_R)
    r_stab_fluid = r_stab_fluid + rho_f_const * g_v_f_inertia(gamma_v_f, uf_k_R, test_vel_f_R)

a_stab_solid = (
    rho_s_const * g_v_s(gamma_v_s, du_s_R, test_vel_s_R)
    + (rho_s_const / dt) * g_v_s_mass(gamma_v_s_mass, du_s_R, test_vel_s_R)
    + Constant(2.0) * mu_s_const * g_disp_s(gamma_disp_s, ddisp_s_R, test_disp_s_R)
)
r_stab_solid = (
    rho_s_const * g_v_s(gamma_v_s, us_k_R, test_vel_s_R)
    + (rho_s_const / dt) * g_v_s_mass(gamma_v_s_mass, us_k_R, test_vel_s_R)
    + Constant(2.0) * mu_s_const * g_disp_s(gamma_disp_s, disp_k_R, test_disp_s_R)
)

a_stab = a_stab_fluid * dG_fluid + a_stab_solid * dG_solid
r_stab = r_stab_fluid * dG_fluid + r_stab_solid * dG_solid

a_p_stab = Constant(0.0) * dx_fluid
r_p_stab = Constant(0.0) * dx_fluid
if PYCUTFEM_PRESSURE_STAB_JUMP > 0.0:
    a_p_stab = a_p_stab + (gamma_p_stab_jump * (cell_h**3.0 * grad_inner_jump_full(dp_f, test_q_f))) * dS_fluid_quad
    r_p_stab = r_p_stab + (gamma_p_stab_jump * (cell_h**3.0 * grad_inner_jump_full(pf_k, test_q_f))) * dS_fluid_quad
if PYCUTFEM_PRESSURE_STAB_AVG > 0.0:
    a_p_stab = a_p_stab + (gamma_p_stab_avg * (cell_h**3.0 * grad_inner_avg(dp_f, test_q_f))) * dG_fluid
    r_p_stab = r_p_stab + (gamma_p_stab_avg * (cell_h**3.0 * grad_inner_avg(pf_k, test_q_f))) * dG_fluid

a_cip = Constant(0.0) * dx_fluid
r_cip = Constant(0.0) * dx_fluid
if (PYCUTFEM_CIP_BETA_FLUID > 0.0) or (PYCUTFEM_CIP_BETA_SOLID > 0.0) or (PYCUTFEM_CIP_BETA_DISP > 0.0):
    cip_eps = Constant(PYCUTFEM_CIP_U_EPS)
    u_cip_f = uf_n_R if PYCUTFEM_CIP_LAGGED else uf_k_R
    u_cip_s = us_n_R if PYCUTFEM_CIP_LAGGED else us_k_R
    u_mag_f = (dot(u_cip_f, u_cip_f) + cip_eps) ** 0.5
    u_mag_s = (dot(u_cip_s, u_cip_s) + cip_eps) ** 0.5
    tau_cip_f = Constant(PYCUTFEM_CIP_BETA_FLUID) * rho_f_const * cell_h * u_mag_f
    tau_cip_s = Constant(PYCUTFEM_CIP_BETA_SOLID) * rho_s_const * cell_h * u_mag_s
    tau_cip_d = Constant(PYCUTFEM_CIP_BETA_DISP) * rho_s_const * cell_h * u_mag_s
    a_cip = (
        tau_cip_f * grad_inner_jump(du_f_R, test_vel_f_R) * dG_fluid
        + tau_cip_s * grad_inner_jump(du_s_R, test_vel_s_R) * dG_solid
        + tau_cip_d * grad_inner_jump(ddisp_s_R, test_disp_s_R) * dG_solid
    )
    r_cip = (
        tau_cip_f * grad_inner_jump(uf_k_R, test_vel_f_R) * dG_fluid
        + tau_cip_s * grad_inner_jump(us_k_R, test_vel_s_R) * dG_solid
        + tau_cip_d * grad_inner_jump(disp_k_R, test_disp_s_R) * dG_solid
    )

a_reg = solid_reg_eps * (dot(du_s_R, test_vel_s_R) + dot(ddisp_s_R, test_disp_s_R)) * dx_solid
r_reg = solid_reg_eps * (dot(us_k_R, test_vel_s_R) + dot(disp_k_R, test_disp_s_R)) * dx_solid

j_inv_theta_f = Constant(1.0) / (theta_pos_cell + Constant(1.0e-12))
j_inv_theta_s = Constant(1.0) / (theta_neg_cell + Constant(1.0e-12))
a_sliver_mass = (
    gamma_sliver_mass_f * (rho_f_const / dt) * j_inv_theta_f * dot(du_f_R, test_vel_f_R) * dx_fluid_cut
    + gamma_sliver_mass_s * (rho_s_const / dt) * j_inv_theta_s * dot(du_s_R, test_vel_s_R) * dx_solid_cut
)
r_sliver_mass = (
    gamma_sliver_mass_f * (rho_f_const / dt) * j_inv_theta_f * dot(uf_k_R, test_vel_f_R) * dx_fluid_cut
    + gamma_sliver_mass_s * (rho_s_const / dt) * j_inv_theta_s * dot(us_k_R, test_vel_s_R) * dx_solid_cut
)

jacobian_form = a_vol_f + J_int + a_vol_s + a_stab + a_p_stab + a_cip + a_svc + a_reg + a_sliver_mass
residual_form = r_vol_f + R_int + r_vol_s + r_stab + r_p_stab + r_cip + r_svc + r_reg + r_sliver_mass

# -----------------------------------------------------------------------------
# Interface residual diagnostics (optional)
# -----------------------------------------------------------------------------
def _trace_interface_residuals(label: str = "ifc", *, bcs_apply=None) -> None:
    if os.getenv("PYCUTFEM_INTERFACE_RESIDUAL_TRACE", "").lower() not in {"1", "true", "yes"}:
        return
    backend = os.getenv("PYCUTFEM_INTERFACE_RESIDUAL_TRACE_BACKEND", "jit").strip() or "jit"
    seed_fields = os.getenv("PYCUTFEM_INTERFACE_RESIDUAL_SEED", "").lower() in {"1", "true", "yes"}
    bcs_apply = bcs if bcs_apply is None else bcs_apply
    terms = {
        "R_int": R_int,
        "R_int_fluid": R_int_fluid,
        "R_int_solid": R_int_solid,
        "R_int_pen": R_int_pen,
        "R_int_sym_fluid": R_int_sym_fluid,
        "R_int_sym_solid": R_int_sym_solid,
    }
    solid_vel = np.concatenate(
        [
            np.asarray(dof_handler.get_field_slice("vs_neg_x"), dtype=int),
            np.asarray(dof_handler.get_field_slice("vs_neg_y"), dtype=int),
        ]
    )
    fluid_vel = np.concatenate(
        [
            np.asarray(dof_handler.get_field_slice("u_pos_x"), dtype=int),
            np.asarray(dof_handler.get_field_slice("u_pos_y"), dtype=int),
        ]
    )
    saved = None
    if seed_fields:
        seed_pressure = os.getenv("PYCUTFEM_INTERFACE_RESIDUAL_SEED_PRESSURE", "1").lower() not in {"0", "false", "no"}
        saved = (
            uf_k.nodal_values.copy(),
            pf_k.nodal_values.copy(),
        )
        u_coords = dof_handler.get_dof_coords("u_pos_x")
        u_vals = np.array([parabolic_inflow(float(x), float(y), t=2.0) for x, y in u_coords], dtype=float)
        u_dofs = np.asarray(dof_handler.get_field_slice("u_pos_x"), dtype=int)
        v_dofs = np.asarray(dof_handler.get_field_slice("u_pos_y"), dtype=int)
        uf_k.set_nodal_values(u_dofs, u_vals)
        uf_k.set_nodal_values(v_dofs, np.zeros_like(v_dofs, dtype=float))
        if seed_pressure:
            p_coords = dof_handler.get_dof_coords("p_pos_")
            pf_k.nodal_values[:] = -np.asarray(p_coords[:, 0], dtype=float)
        else:
            pf_k.nodal_values.fill(0.0)
        dof_handler.apply_bcs(bcs, uf_k, pf_k)
        print(f"[{label}] seeded uf_k/pf_k for interface residual trace")

    bc_dofs: set[int] = set()
    try:
        bc_dofs = {int(k) for k in dof_handler.get_dirichlet_data(bcs_apply).keys()}
    except Exception:
        bc_dofs = set()
    inactive_dofs = {int(k) for k in getattr(dof_handler, "dof_tags", {}).get("inactive", set())}
    active_dofs: set[int] = set()
    try:
        solver_obj = globals().get("solver", None)
        active_raw = getattr(solver_obj, "active_dofs", None) if solver_obj is not None else None
        if active_raw is None:
            active_dofs = set()
        else:
            active_dofs = {int(k) for k in np.asarray(active_raw, dtype=int).ravel()}
    except Exception:
        active_dofs = set()

    print(f"[{label}] interface residual trace backend={backend}")
    for name, term in terms.items():
        if term is None:
            continue
        _, R = assemble_form(Equation(None, term), dof_handler=dof_handler, bcs=[], backend=backend)
        if R is None:
            continue
        solid_norm = float(np.linalg.norm(R[solid_vel], ord=np.inf)) if solid_vel.size else 0.0
        fluid_norm = float(np.linalg.norm(R[fluid_vel], ord=np.inf)) if fluid_vel.size else 0.0
        print(f"[{label}] {name}: |R|_solid_vel_inf={solid_norm:.3e} |R|_fluid_vel_inf={fluid_norm:.3e}")

        if os.getenv("PYCUTFEM_INTERFACE_RESIDUAL_TRACE_WORST", "").lower() in {"1", "true", "yes"}:
            def _dump_worst(tag: str, subset: np.ndarray) -> None:
                if subset.size == 0:
                    return
                vals = np.asarray(R[subset], dtype=float)
                idx = int(np.argmax(np.abs(vals)))
                gdof = int(subset[idx])
                val = float(vals[idx])
                field = None
                try:
                    field = getattr(dof_handler, "_dof_to_node_map", {}).get(gdof, (None, None))[0]
                except Exception:
                    field = None
                status = "active" if gdof in active_dofs else ("inactive" if gdof in inactive_dofs else "other")
                bc = "bc" if gdof in bc_dofs else "free"
                red = None
                try:
                    solver_obj = globals().get("solver", None)
                    if solver_obj is not None and hasattr(solver_obj, "full_to_red"):
                        red = int(np.asarray(getattr(solver_obj, "full_to_red"))[gdof])
                except Exception:
                    red = None
                fmsg = f", field={field}" if field is not None else ""
                rmsg = f", red={red}" if red is not None else ""
                print(f"[{label}] {name}: worst_{tag} gdof={gdof} val={val:+.3e} ({status},{bc}{fmsg}{rmsg})")

            _dump_worst("solid", solid_vel)
            _dump_worst("fluid", fluid_vel)

    if os.getenv("PYCUTFEM_INTERFACE_RESIDUAL_TRACE_FIELDS", "").lower() in {"1", "true", "yes"}:
        u_pos_sq = dot(Pos(uf_k_ifc), Pos(uf_k_ifc)) * dΓ
        jump_sq = dot(jump_vel_res, jump_vel_res) * dΓ
        hooks = {
            u_pos_sq.integrand: {"name": "u_pos_sq"},
            jump_sq.integrand: {"name": "jump_sq"},
        }
        res = assemble_form(
            Equation(None, u_pos_sq + jump_sq), dof_handler=dof_handler, bcs=[], backend=backend, assembler_hooks=hooks
        )
        u_pos_val = float(np.asarray(res.get("u_pos_sq", 0.0)).reshape(-1)[0])
        jump_val = float(np.asarray(res.get("jump_sq", 0.0)).reshape(-1)[0])
        print(f"[{label}] ⟨|u_pos|^2⟩_Γ={u_pos_val:.3e} ⟨|jump|^2⟩_Γ={jump_val:.3e}")

    if saved is not None:
        uf_k.nodal_values[:] = saved[0]
        pf_k.nodal_values[:] = saved[1]
        dof_handler.apply_bcs(bcs, uf_k, pf_k)

# Optional: trace interface residual contributions at the initial state.
_trace_interface_residuals("ifc_init", bcs_apply=bcs_t0)

# -----------------------------------------------------------------------------
# Peclet number diagnostics (optional)
# -----------------------------------------------------------------------------
_PECLET_FORM_CACHE: dict[tuple[str, int, bool], dict[str, object]] = {}


def _peclet_diagnostics(
    label: str = "pe",
    *,
    step: int | None = None,
    t: float | None = None,
    dt: float | None = None,
    backend: str | None = None,
) -> None:
    if os.getenv("PYCUTFEM_PECLET_TRACE", "").lower() not in {"1", "true", "yes"}:
        return
    mode = os.getenv("PYCUTFEM_PECLET_MODE", "nodal").strip().lower()
    backend = (backend or os.getenv("PYCUTFEM_PECLET_BACKEND", "")).strip() or assembly_backend
    try:
        p = int(os.getenv("PYCUTFEM_PECLET_P", "8") or "8")
    except Exception:
        p = 8
    p = int(max(2, min(p, 24)))
    try:
        eps = float(os.getenv("PYCUTFEM_PECLET_EPS", "1e-12") or "1e-12")
    except Exception:
        eps = 1.0e-12
    eps = float(max(eps, 1.0e-30))
    lagged = os.getenv("PYCUTFEM_PECLET_LAGGED", "0").lower() in {"1", "true", "yes"}

    if mode not in {"integral", "assemble", "ufl"}:
        cache = getattr(_peclet_diagnostics, "_nodal_cache", None)
        if cache is None:
            try:
                ux_gdofs = np.asarray(dof_handler.get_field_slice("u_pos_x"), dtype=int)
                uy_gdofs = np.asarray(dof_handler.get_field_slice("u_pos_y"), dtype=int)
                ux_nodes = np.array([dof_handler._dof_to_node_map[int(gd)][1] for gd in ux_gdofs], dtype=int)
                uy_nodes = np.array([dof_handler._dof_to_node_map[int(gd)][1] for gd in uy_gdofs], dtype=int)
            except Exception:
                ux_gdofs = np.empty(0, dtype=int)
                uy_gdofs = np.empty(0, dtype=int)
                ux_nodes = np.empty(0, dtype=int)
                uy_nodes = np.empty(0, dtype=int)
            cache = {
                "ux_gdofs": ux_gdofs,
                "uy_gdofs": uy_gdofs,
                "ux_nodes": ux_nodes,
                "uy_nodes": uy_nodes,
            }
            setattr(_peclet_diagnostics, "_nodal_cache", cache)

        try:
            h_diag = float(os.getenv("PYCUTFEM_PECLET_H", str(MESH_SIZE)) or str(MESH_SIZE))
        except Exception:
            h_diag = float(MESH_SIZE)
        pe_scale = float(RHO_F) * float(h_diag) / (2.0 * float(MU_F) + 1.0e-30)

        u_ref_vals = np.asarray((uf_n if lagged else uf_k).nodal_values, dtype=float)
        n_ux = int(np.asarray(cache["ux_gdofs"], dtype=int).size)
        n_uy = int(np.asarray(cache["uy_gdofs"], dtype=int).size)
        ux_vals = u_ref_vals[:n_ux] if n_ux else np.empty(0, dtype=float)
        uy_vals = u_ref_vals[n_ux : n_ux + n_uy] if n_uy else np.empty(0, dtype=float)

        n_nodes = int(getattr(mesh.nodes_x_y_pos, "shape", (0,))[0])
        ux_node = np.zeros(n_nodes, dtype=float)
        uy_node = np.zeros(n_nodes, dtype=float)
        sol = globals().get("solver", None)
        active_full = np.asarray(getattr(sol, "active_dofs", np.empty(0, dtype=int)), dtype=int) if sol is not None else np.empty(0, dtype=int)
        if n_ux:
            ux_nodes = np.asarray(cache["ux_nodes"], dtype=int)
            ux_gdofs = np.asarray(cache["ux_gdofs"], dtype=int)
            if active_full.size:
                ux_mask = np.isin(ux_gdofs, active_full, assume_unique=False)
                ux_node[ux_nodes[ux_mask]] = ux_vals[ux_mask]
            else:
                ux_node[ux_nodes] = ux_vals
        if n_uy:
            uy_nodes = np.asarray(cache["uy_nodes"], dtype=int)
            uy_gdofs = np.asarray(cache["uy_gdofs"], dtype=int)
            if active_full.size:
                uy_mask = np.isin(uy_gdofs, active_full, assume_unique=False)
                uy_node[uy_nodes[uy_mask]] = uy_vals[uy_mask]
            else:
                uy_node[uy_nodes] = uy_vals

        u_mag = np.sqrt(ux_node * ux_node + uy_node * uy_node + float(eps))
        pe_nodes = pe_scale * u_mag

        cut_eids = np.nonzero(mesh.element_bitset("cut").mask)[0]
        cut_nodes_mask = np.zeros(n_nodes, dtype=bool)
        if cut_eids.size:
            cut_nodes_mask[np.asarray(mesh.elements_connectivity[cut_eids].ravel(), dtype=int)] = True
        try:
            phi = np.asarray(ls_beam.nodal_values(), dtype=float)
        except Exception:
            phi = np.asarray(getattr(ls_beam, "nodal_values", np.zeros(n_nodes, dtype=float)), dtype=float)
        if int(phi.size) != int(n_nodes):
            phi = np.resize(phi, n_nodes)
        fluid_nodes_mask = phi >= 0.0
        cut_nodes_mask = cut_nodes_mask & fluid_nodes_mask

        def _stats(mask: np.ndarray) -> tuple[float, float]:
            vals = np.asarray(pe_nodes[mask], dtype=float)
            if vals.size == 0:
                return float("nan"), float("nan")
            mean = float(np.mean(vals))
            pnorm = float(np.mean(vals**p) ** (1.0 / p))
            return mean, pnorm

        pe_mean_all, pe_pnorm_all = _stats(fluid_nodes_mask)
        pe_mean_cut, pe_pnorm_cut = _stats(cut_nodes_mask)
        u_max_all = float(np.max(u_mag[fluid_nodes_mask])) if np.any(fluid_nodes_mask) else float("nan")
        u_max_cut = float(np.max(u_mag[cut_nodes_mask])) if np.any(cut_nodes_mask) else float("nan")
        pe_max_all = float(np.max(pe_nodes[fluid_nodes_mask])) if np.any(fluid_nodes_mask) else float("nan")
        pe_max_cut = float(np.max(pe_nodes[cut_nodes_mask])) if np.any(cut_nodes_mask) else float("nan")
        loc_trace = os.getenv("PYCUTFEM_PECLET_TRACE_LOC", "").lower() in {"1", "true", "yes"}
        loc_msg = ""
        if loc_trace and np.any(fluid_nodes_mask):
            idxs = np.nonzero(fluid_nodes_mask)[0]
            i_max = int(idxs[int(np.argmax(u_mag[idxs]))])
            try:
                x_max, y_max = map(float, mesh.nodes_x_y_pos[i_max])
            except Exception:
                x_max, y_max = float("nan"), float("nan")
            phi_max = float(phi[i_max]) if int(phi.size) > i_max else float("nan")
            in_cut = bool(cut_nodes_mask[i_max]) if int(cut_nodes_mask.size) > i_max else False
            loc_msg = f" max@node={i_max} (x={x_max:.3f},y={y_max:.3f},phi={phi_max:.2e},cut={int(in_cut)})"

        step_s = f"{int(step):04d}" if step is not None else "????"
        t_s = f"{float(t):.6f}" if t is not None else "nan"
        dt_s = f"{float(dt):.3e}" if dt is not None else "nan"
        print(
            f"[{label}] step={step_s} t={t_s} dt={dt_s} "
            f"Pe(mean={pe_mean_all:.3e}, p{p}={pe_pnorm_all:.3e}) "
            f"Pe_cut(mean={pe_mean_cut:.3e}, p{p}={pe_pnorm_cut:.3e}) "
            f"|u|max={u_max_all:.3e} Pe_max≈{pe_max_all:.3e} "
            f"|u|max_cut={u_max_cut:.3e} Pe_cut_max≈{pe_max_cut:.3e} "
            f"(mode={mode}){loc_msg}"
        )
        return

    cache_key = (str(backend), int(p), bool(lagged))
    cache = _PECLET_FORM_CACHE.get(cache_key)
    if cache is None:
        eps0 = Constant(float(eps))
        # IMPORTANT: use distinct Constant objects per integral so the
        # `assembler_hooks` lookup (by integrand equality) can distinguish
        # integrals that share the same algebraic integrand but live on
        # different measures (dx_fluid vs dx_fluid_cut).
        one_area_all = Constant(1.0)
        one_area_cut = Constant(1.0)
        one_pe_all = Constant(1.0)
        one_pe_cut = Constant(1.0)
        one_pe_p_all = Constant(1.0)
        one_pe_p_cut = Constant(1.0)
        two = Constant(2.0)
        tiny = Constant(1.0e-30)
        u_ref = uf_n_R if lagged else uf_k_R
        u_mag = (dot(u_ref, u_ref) + eps0) ** 0.5
        pe = (rho_f_const * u_mag * cell_h) / (two * mu_f_const + tiny)

        area_all = one_area_all * dx_fluid
        area_cut = one_area_cut * dx_fluid_cut
        pe_all = (one_pe_all * pe) * dx_fluid
        pe_cut = (one_pe_cut * pe) * dx_fluid_cut
        pe_p_all = (one_pe_p_all * (pe**int(p))) * dx_fluid
        pe_p_cut = (one_pe_p_cut * (pe**int(p))) * dx_fluid_cut

        hooks = {
            area_all.integrand: {"name": "area_all"},
            area_cut.integrand: {"name": "area_cut"},
            pe_all.integrand: {"name": "pe_int_all"},
            pe_cut.integrand: {"name": "pe_int_cut"},
            pe_p_all.integrand: {"name": "pe_p_int_all"},
            pe_p_cut.integrand: {"name": "pe_p_int_cut"},
        }
        equation = Equation(None, area_all + area_cut + pe_all + pe_cut + pe_p_all + pe_p_cut)
        cache = {"equation": equation, "hooks": hooks, "eps": eps0}
        _PECLET_FORM_CACHE[cache_key] = cache
    else:
        eps0 = cache.get("eps")
        if hasattr(eps0, "value"):
            eps0.value = float(eps)

    res = assemble_form(
        cache["equation"],
        dof_handler=dof_handler,
        bcs=[],
        backend=backend,
        assembler_hooks=cache["hooks"],
    )

    def _scalar(name: str) -> float:
        try:
            return float(np.asarray(res.get(name, 0.0)).reshape(-1)[0])
        except Exception:
            try:
                return float(res.get(name, 0.0))
            except Exception:
                return 0.0

    A_all = max(_scalar("area_all"), 0.0)
    A_cut = max(_scalar("area_cut"), 0.0)
    pe_int_all = _scalar("pe_int_all")
    pe_int_cut = _scalar("pe_int_cut")
    pe_p_int_all = max(_scalar("pe_p_int_all"), 0.0)
    pe_p_int_cut = max(_scalar("pe_p_int_cut"), 0.0)

    pe_mean_all = pe_int_all / A_all if A_all > 0.0 else float("nan")
    pe_mean_cut = pe_int_cut / A_cut if A_cut > 0.0 else float("nan")
    pe_pnorm_all = (pe_p_int_all / A_all) ** (1.0 / p) if A_all > 0.0 else float("nan")
    pe_pnorm_cut = (pe_p_int_cut / A_cut) ** (1.0 / p) if A_cut > 0.0 else float("nan")

    step_s = f"{int(step)}" if step is not None else "?"
    t_s = f"{float(t):.6f}" if t is not None else "?"
    dt_s = f"{float(dt):.3e}" if dt is not None else "?"
    print(
        f"[{label}] step={step_s} t={t_s} dt={dt_s} "
        f"Pe(mean={pe_mean_all:.3e}, p{p}={pe_pnorm_all:.3e}) "
        f"Pe_cut(mean={pe_mean_cut:.3e}, p{p}={pe_pnorm_cut:.3e}) "
        f"(backend={backend}, lagged={int(lagged)})"
    )

# ----------------------------------------------------------------------------- 
# Diagnostics: tip displacement, drag/lift (avg traction), pressure drop, VTK
# -----------------------------------------------------------------------------
REF_TIP = np.array([CENTER[0] + RADIUS + BEAM_LENGTH, CENTER[1]], dtype=float)
PROBE_B = np.array([CENTER[0] - 0.05, CENTER[1]], dtype=float)
FLOW_PROBE_TOL = 1.0e-10
FLOW_PROBES = [
    ("inlet_mid", (0.05, 0.5 * H)),
    ("above_beam", (BEAM_REF_CENTER[0], BEAM_REF_CENTER[1] + 0.12)),
    ("below_beam", (BEAM_REF_CENTER[0], BEAM_REF_CENTER[1] - 0.12)),
    ("mid_channel", (1.1, 0.5 * H)),
    ("outlet_mid", (L - 0.2, 0.5 * H)),
]
obs_history = {"time": [], "tip": [], "drag": [], "lift": [], "dp": []}
output_dir = ARGS.output_dir
os.makedirs(output_dir, exist_ok=True)
SAVE_VTK = bool(ARGS.save_vtk)
COMPUTE_OBSERVABLES = bool(getattr(ARGS, "compute_observables", True))
try:
    OBS_EVERY = int(getattr(ARGS, "obs_every", 1))
except Exception:
    OBS_EVERY = 1
PLOT_MESH_EVERY = max(1, int(ARGS.plot_mesh_every))
VTK_INTERFACE_CLAMP = float(os.getenv("VTK_INTERFACE_CLAMP", "0.0"))
VTK_CLAMP_BAND = max(1.5 * MESH_SIZE, 5.0 * LEVELSET_EDGE_TOL)

# -----------------------------------------------------------------------------
# Debug: dump level-set DOF coordinates + nodal values per step
# -----------------------------------------------------------------------------
DUMP_LEVELSET = os.getenv("PYCUTFEM_DUMP_LEVELSET", "").lower() in {"1", "true", "yes"}
DUMP_LEVELSET_EVERY = max(1, int(os.getenv("PYCUTFEM_DUMP_LEVELSET_EVERY", "1") or "1"))
LEVELSET_DUMP_DIR = (
    os.getenv("PYCUTFEM_LEVELSET_DUMP_DIR", "").strip()
    or os.getenv("PYCUTFEM_DUMP_LEVELSET_DIR", "").strip()
    or os.path.join(output_dir, "levelset_dumps")
)
LEVELSET_DUMP_MESH_FILE = os.path.join(LEVELSET_DUMP_DIR, "mesh.npz")

# Optional: dump full solution vectors for offline Newton replay/debugging.
# Use sparingly: these are large (~O(10^5–10^6) DOFs).
DUMP_STATE = os.getenv("PYCUTFEM_DUMP_STATE", "").lower() in {"1", "true", "yes"}
try:
    DUMP_STATE_EVERY = int(os.getenv("PYCUTFEM_DUMP_STATE_EVERY", "0") or "0")
except Exception:
    DUMP_STATE_EVERY = 0
STATE_DUMP_DIR = os.getenv("PYCUTFEM_STATE_DUMP_DIR", "").strip() or os.path.join(output_dir, "state_dumps")


def _dump_levelset_mesh_once() -> None:
    if not DUMP_LEVELSET:
        return
    os.makedirs(LEVELSET_DUMP_DIR, exist_ok=True)
    if os.path.exists(LEVELSET_DUMP_MESH_FILE):
        return
    try:
        phi_coords = np.asarray(ls_dh.get_dof_coords("phi_beam"), dtype=float)
    except Exception:
        phi_coords = np.empty((0, 2), dtype=float)
    # Some meshes (e.g. refined structured) do not carry an explicit `edges_connectivity`
    # array. For replay/debug, we only need edge endpoints; Mesh will recover
    # intermediate/hanging nodes along the geometric edge.
    try:
        edges_conn = getattr(mesh, "edges_connectivity", None)
        if edges_conn is None:
            edges_conn = np.asarray([tuple(map(int, e.nodes)) for e in getattr(mesh, "edges_list", [])], dtype=np.int64)
        else:
            edges_conn = np.asarray(edges_conn, dtype=np.int64)
    except Exception:
        edges_conn = np.empty((0, 2), dtype=np.int64)
    try:
        corners_conn = getattr(mesh, "corner_connectivity", None)
        if corners_conn is None:
            corners_conn = np.asarray([tuple(map(int, el.corner_nodes)) for el in getattr(mesh, "elements_list", [])], dtype=np.int64)
        else:
            corners_conn = np.asarray(corners_conn, dtype=np.int64)
    except Exception:
        corners_conn = np.empty((0, 0), dtype=np.int64)
    np.savez_compressed(
        LEVELSET_DUMP_MESH_FILE,
        element_type=str(getattr(mesh, "element_type", "")),
        poly_order=int(getattr(mesh, "poly_order", 1)),
        nodes=np.asarray(mesh.nodes_x_y_pos, dtype=float),
        elements=np.asarray(mesh.elements_connectivity, dtype=np.int64),
        edges=edges_conn,
        corners=corners_conn,
        phi_field="phi_beam",
        phi_poly_order=int(POLY_ORDER),
        phi_dof_coords=phi_coords,
    )
    print(f"[dump] wrote {LEVELSET_DUMP_MESH_FILE}")


def _dump_levelset_step(step_no: int, t_curr: float, *, tag: str = "step") -> None:
    if not DUMP_LEVELSET:
        return
    if (int(step_no) % int(DUMP_LEVELSET_EVERY)) != 0:
        return
    _dump_levelset_mesh_once()
    os.makedirs(LEVELSET_DUMP_DIR, exist_ok=True)
    try:
        phi = np.asarray(ls_beam.nodal_values(), dtype=float).copy()
    except Exception:
        phi = np.asarray(ls_beam.nodal_values, dtype=float).copy()
    stats = getattr(ls_beam, "_last_update_stats", None)
    payload = {
        "step": int(step_no),
        "t": float(t_curr),
        "dt": float(getattr(dt, "value", DT)),
        "phi": phi,
        "cut_elems": int(mesh.element_bitset("cut").cardinality()),
        "interface_edges": int(mesh.edge_bitset("interface").cardinality()),
    }
    if isinstance(stats, dict):
        for k in ("max_diff", "max_diff_cut", "cut_elems", "interface_edges"):
            if k in stats:
                payload[f"ls_{k}"] = float(stats[k]) if "diff" in k else int(stats[k])
    fname = os.path.join(LEVELSET_DUMP_DIR, f"phi_{tag}_{int(step_no):04d}.npz")
    np.savez_compressed(fname, **payload)
    print(f"[dump] wrote {fname}")


def _dump_state_step(
    step_no: int,
    t_curr: float,
    *,
    tag: str,
    funcs: list | None = None,
    prev_funcs: list | None = None,
) -> None:
    if not DUMP_STATE:
        return
    if tag == "step" and int(DUMP_STATE_EVERY) > 0 and (int(step_no) % int(DUMP_STATE_EVERY)) != 0:
        return
    os.makedirs(STATE_DUMP_DIR, exist_ok=True)
    payload: dict[str, object] = {
        "step": int(step_no),
        "t": float(t_curr),
        "dt": float(getattr(dt, "value", DT)),
        "mesh_n_nodes": int(getattr(mesh.nodes_x_y_pos, "shape", (0,))[0]),
        "mesh_n_elements": int(getattr(mesh, "n_elements", len(getattr(mesh, "elements_list", [])))),
        "mesh_n_edges": int(len(getattr(mesh, "edges_list", []))),
        "mesh_element_type": str(getattr(mesh, "element_type", "")),
        "mesh_poly_order": int(getattr(mesh, "poly_order", 1)),
        "dof_total": int(getattr(dof_handler, "total_dofs", -1)),
        "phi_field": str(getattr(ls_beam, "field", "")),
    }
    # Make state dumps restartable without requiring a separate level-set dump:
    # persist the current φ nodal values + basic tag counts alongside U^n/U^{n-1}.
    try:
        phi_vals = np.asarray(ls_beam.nodal_values(), dtype=float).copy()
    except Exception:
        phi_vals = np.asarray(getattr(ls_beam, "nodal_values", np.array([])), dtype=float).copy()
    payload["phi"] = phi_vals
    try:
        payload["cut_elems"] = int(mesh.element_bitset("cut").cardinality())
        payload["interface_edges"] = int(mesh.edge_bitset("interface").cardinality())
    except Exception:
        payload["cut_elems"] = -1
        payload["interface_edges"] = -1
    if funcs is not None:
        for f in funcs:
            name = getattr(f, "name", None)
            if not name:
                continue
            payload[f"{name}_k"] = np.asarray(getattr(f, "nodal_values", np.array([])), dtype=float).copy()
    if prev_funcs is not None:
        for f in prev_funcs:
            name = getattr(f, "name", None)
            if not name:
                continue
            payload[f"{name}_n"] = np.asarray(getattr(f, "nodal_values", np.array([])), dtype=float).copy()
    fname = os.path.join(STATE_DUMP_DIR, f"state_{tag}_{int(step_no):04d}.npz")
    np.savez_compressed(fname, **payload)
    print(f"[dump] wrote {fname}")


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    with np.load(str(path)) as data:
        return {k: data[k] for k in data.files}


def _load_restart_payload(*, restart_dir: Path, step_no: int, tag: str) -> dict[str, object]:
    restart_dir = Path(restart_dir)
    if not restart_dir.exists():
        raise FileNotFoundError(str(restart_dir))
    ls_dir_raw = os.getenv("RESTART_LEVELSET_DIR", "").strip()
    st_dir_raw = os.getenv("RESTART_STATE_DIR", "").strip()
    levelset_dir = Path(ls_dir_raw) if ls_dir_raw else restart_dir / "levelset_dumps"
    state_dir = Path(st_dir_raw) if st_dir_raw else restart_dir / "state_dumps"

    phi_path = levelset_dir / f"phi_{str(tag)}_{int(step_no):04d}.npz"
    state_path = state_dir / f"state_{str(tag)}_{int(step_no):04d}.npz"

    state_npz = _load_npz_dict(state_path)
    phi_npz: dict[str, np.ndarray] = {}
    try:
        phi_npz = _load_npz_dict(phi_path)
    except FileNotFoundError:
        phi_npz = {}

    t_val = float(state_npz.get("t", phi_npz.get("t", 0.0)))
    dt_val = float(state_npz.get("dt", phi_npz.get("dt", DT)))
    step_val = int(state_npz.get("step", phi_npz.get("step", step_no)))
    phi_arr = phi_npz.get("phi", state_npz.get("phi", np.array([])))
    phi_arr = np.asarray(phi_arr, dtype=float)
    if phi_arr.size == 0:
        raise FileNotFoundError(str(phi_path))

    out: dict[str, object] = {
        "restart_dir": str(restart_dir),
        "levelset_dir": str(levelset_dir),
        "state_dir": str(state_dir),
        "phi_path": str(phi_path),
        "state_path": str(state_path),
        "step": step_val,
        "t": t_val,
        "dt": dt_val,
        "phi": phi_arr,
        "state": state_npz,
    }
    return out


def _is_fluid_point(point: tuple[float, float]) -> bool:
    x, y = point
    if (x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2 <= RADIUS ** 2:
        return False
    try:
        phi = float(ls_beam(np.array([x, y], dtype=float)))
    except Exception:
        phi = float(beam_ref_ls(np.array([x, y], dtype=float)))
    return phi > 0.0


def _probe_flow_velocity(step_idx: int) -> None:
    mags: list[float] = []
    entries: list[str] = []
    for label, pt in FLOW_PROBES:
        if not _is_fluid_point(pt):
            entries.append(f"{label}=skip")
            continue
        vel = _eval_vector_at_point(dof_handler, mesh, uf_k, pt)
        if not np.all(np.isfinite(vel)):
            entries.append(f"{label}=nan")
            continue
        mag = float(np.linalg.norm(vel))
        mags.append(mag)
        entries.append(f"{label}=({vel[0]:.2e},{vel[1]:.2e})|u|={mag:.2e}")
    if entries:
        print(f"[probe {step_idx:04d}] " + " ".join(entries))
    if mags and all(m <= FLOW_PROBE_TOL for m in mags):
        print(
            f"[probe {step_idx:04d}] WARNING: flow velocity ~0 at all probe points "
            f"(max|u|={max(mags):.2e})."
        )

def _save_vtk(step_idx: int) -> None:
    if not SAVE_VTK:
        return
    fname = os.path.join(output_dir, f"solution_{step_idx:04d}.vtu")
    phi_nodes = ls_beam.evaluate_on_nodes(mesh)
    if VTK_INTERFACE_CLAMP > 0.0:
        num_nodes = len(mesh.nodes_list)

        def _scalar_to_nodes(f_scalar: Function) -> np.ndarray:
            scal = np.zeros(num_nodes)
            for gdof, lidx in f_scalar._g2l.items():
                _, nid = dof_handler._dof_to_node_map.get(gdof, (None, None))
                if nid is None:
                    continue
                scal[int(nid)] = f_scalar.nodal_values[int(lidx)]
            return scal

        def _vector_to_nodes(f_vec: VectorFunction) -> np.ndarray:
            vec = np.zeros((num_nodes, 2))
            for gdof, lidx in f_vec._g2l.items():
                fld, nid = dof_handler._dof_to_node_map.get(gdof, (None, None))
                if nid is None or fld not in f_vec.field_names:
                    continue
                comp = f_vec.field_names.index(fld)
                vec[int(nid), comp] = f_vec.nodal_values[int(lidx)]
            return vec

        
        mask = np.isfinite(phi_nodes) & (np.abs(phi_nodes) <= VTK_CLAMP_BAND)

        uf_nodes = _vector_to_nodes(uf_k)
        us_nodes = _vector_to_nodes(us_k)
        disp_nodes = _vector_to_nodes(disp_k)
        pf_nodes = _scalar_to_nodes(pf_k)

        if np.any(mask):
            uf_nodes[mask] = np.clip(uf_nodes[mask], -VTK_INTERFACE_CLAMP, VTK_INTERFACE_CLAMP)
            us_nodes[mask] = np.clip(us_nodes[mask], -VTK_INTERFACE_CLAMP, VTK_INTERFACE_CLAMP)
            disp_nodes[mask] = np.clip(disp_nodes[mask], -VTK_INTERFACE_CLAMP, VTK_INTERFACE_CLAMP)
            pf_nodes[mask] = np.clip(pf_nodes[mask], -VTK_INTERFACE_CLAMP, VTK_INTERFACE_CLAMP)

        export_vtk(
            filename=fname,
            mesh=mesh,
            dof_handler=dof_handler,
            functions={
                "uf": uf_nodes,
                "pf": pf_nodes,
                "us": us_nodes,
                "disp": disp_nodes,
                # Beam level-set (negative inside solid, zero on interface)
                "phi_beam": phi_nodes
            },
        )
        return
    export_vtk(
        filename=fname,
        mesh=mesh,
        dof_handler=dof_handler,
        functions={
            "uf": uf_k,
            "pf": pf_k,
            "us": us_k,
            "disp": disp_k,
            # Beam level-set (negative inside solid, zero on interface)
            "phi_beam": phi_nodes
        },
    )


def _plot_mesh(step_idx: int, title: str = "Mesh / Ghost / Level-set") -> None:
    if not ARGS.plot_mesh:
        return
    if step_idx % PLOT_MESH_EVERY != 0:
        return
    import matplotlib.pyplot as plt
    from matplotlib.widgets import CheckButtons
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot with the analytic reference LS to avoid FE warping in the visualization.
    level_set_for_plot = beam_ref_ls if ARGS.plot_levelset else None
    plot_mesh_2(
        mesh,
        level_set=level_set_for_plot,
        plot_nodes=True,
        plot_edges=True,
        elem_tags=True,
        edge_colors=True,
        plot_interface=False,  # handled below for toggling
        show=False,
        ax=ax,
        resolution=max(20, int(ARGS.plot_resolution)),
        highlight_edges=PATHOLOGY_EDGES,
    )
    interface_artists = []
    if ARGS.plot_interface_points:
        segments = []
        pts_all = []
        for elem in mesh.elements_list:
            pts = getattr(elem, "interface_pts", [])
            if len(pts) == 2:
                segments.append(np.asarray(pts, float))
            pts_all.extend(pts)
        if segments:
            interface_artists.append(
                ax.add_collection(LineCollection(segments, colors="magenta", linewidths=1.5, zorder=6, label="Interface"))
            )
        if pts_all:
            pts_np = np.asarray(pts_all, float)
            interface_artists.append(ax.plot(pts_np[:, 0], pts_np[:, 1], "o", color="cyan", markersize=6, markeredgecolor="black", label="Interface pts")[0])

    levelset_artists = []
    if level_set_for_plot is not None:
        xmin, ymin = mesh.nodes_x_y_pos.min(axis=0)
        xmax, ymax = mesh.nodes_x_y_pos.max(axis=0)
        padding = (xmax - xmin) * 0.05
        res = max(20, int(ARGS.plot_resolution))
        gx, gy = np.meshgrid(
            np.linspace(xmin - padding, xmax + padding, res),
            np.linspace(ymin - padding, ymax + padding, res),
        )
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        vals = np.apply_along_axis(level_set_for_plot, 1, pts).reshape(gx.shape)
        cs = ax.contour(gx, gy, vals, levels=[0.0], colors="green", linewidths=1.5, zorder=5)
        collections = getattr(cs, "collections", None)
        if collections:
            levelset_artists = list(collections)
        else:
            levelset_artists = [cs]

    toggles = []
    labels = []
    artists = []
    if levelset_artists:
        labels.append("Level set")
        artists.append(levelset_artists)
        toggles.append(True)
    if interface_artists:
        labels.append("Interface pts")
        artists.append(interface_artists)
        toggles.append(True)

    if ARGS.interactive_plot and labels:
        rax = fig.add_axes([0.82, 0.4, 0.15, 0.15])
        check = CheckButtons(rax, labels, toggles)

        def func(label):
            for lab, arts in zip(labels, artists):
                if lab == label:
                    new_vis = not arts[0].get_visible()
                    for art in arts:
                        art.set_visible(new_vis)
            fig.canvas.draw_idle()

        check.on_clicked(func)

    ax.set_title(f"{title} (step {step_idx})")
    fname = os.path.join(output_dir, f"mesh_{step_idx:04d}.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    if ARGS.plot_show or ARGS.interactive_plot:
        import matplotlib
        plt.show()
        # block = bool(ARGS.interactive_plot)
        # backend = matplotlib.get_backend().lower()
        # if "agg" in backend and block:
        #     print("Non-interactive backend detected; skipping blocking plt.show(). Image saved instead.")
        #     block = False
        # plt.show(block=block)
        # if not block:
        #     plt.pause(0.1)
        #     plt.close(fig)
    else:
        plt.close(fig)

# Plot initial mesh/level set before any expensive steps (FD checks/time stepping)
_plot_mesh(step_idx=0, title="Initial mesh")
if ARGS.mesh_diagnostics_enabled:
    diag = mesh_topology_diagnostics(mesh)
    print(f"[mesh-check] missing_side={diag['missing_side']} ownerless_edges={diag['ownerless_edges']} degenerate_shared={diag['degenerate_shared']}")
    if diag["degenerate_shared"] > 0:
        repaired = repair_degenerate_edges(mesh)
        diag = mesh_topology_diagnostics(mesh)
        print(f"[mesh-repair] repaired={repaired} → missing_side={diag['missing_side']} ownerless_edges={diag['ownerless_edges']} degenerate_shared={diag['degenerate_shared']}")
    cov = coverage_diagnostics(mesh, n_samples=60)
    if cov["gaps_x"]:
        print(f"[coverage] gaps at x≈ {', '.join(f'{x:.4f}' for x in cov['gaps_x'][:10])}" + (" ..." if len(cov['gaps_x']) > 10 else ""))
    else:
        print("[coverage] no vertical gaps detected in sampled lines")

def _compute_observables(step_idx: int, t_curr: float) -> None:
    dGamma_obs = dInterface(
        defined_on=domains["cut_interface"],
        level_set=ls_beam,
        metadata={"q": qvol + 2, "derivs": {(0, 0), (1, 0), (0, 1)}, "linear_interface": USE_LINEAR_INTERFACE},
    )
    ex = Constant(np.array([1.0, 0.0]), dim=1)
    ey = Constant(np.array([0.0, 1.0]), dim=1)
    t_fluid = traction_fluid_primal(Pos(uf_k), Pos(pf_k))
    t_solid = traction_solid_R(Neg(disp_k))
    t_avg = Constant(0.5) * (t_fluid + t_solid)
    drag_int = dot(t_avg, ex) * dGamma_obs
    lift_int = dot(t_avg, ey) * dGamma_obs
    hooks = {drag_int.integrand: {"name": "FD"}, lift_int.integrand: {"name": "FL"}}
    res = assemble_form(
        Equation(None, drag_int + lift_int),
        dof_handler=dof_handler,
        bcs=[],
        assembler_hooks=hooks,
        backend="python",
    )
    F_D = float(np.asarray(res.get("FD", 0.0)).reshape(-1)[0])
    F_L = float(np.asarray(res.get("FL", 0.0)).reshape(-1)[0])
    tip_pos = _tip_position(dof_handler, mesh, disp_k, REF_TIP)
    tip_disp = tip_pos - REF_TIP
    pA = _eval_scalar_at_point(dof_handler, mesh, pf_k, tuple(tip_pos))
    pB = _eval_scalar_at_point(dof_handler, mesh, pf_k, tuple(PROBE_B))
    dp = pB - pA
    obs_history["time"].append(t_curr)
    obs_history["drag"].append(F_D)
    obs_history["lift"].append(F_L)
    obs_history["dp"].append(dp)
    obs_history["tip"].append(tip_pos.tolist())
    print(
        f"[obs {step_idx:04d}] t={t_curr:.3f}  FD={F_D:.4e}  FL={F_L:.4e}  Δp={dp:.4e}  "
        f"tip=({tip_pos[0]:.5f},{tip_pos[1]:.5f})  "
        f"tip_disp=({tip_disp[0]:.5e},{tip_disp[1]:.5e})  |Γ|≈{_interface_length(mesh):.5f}"
    )
    _save_vtk(step_idx)

# -----------------------------------------------------------------------------
# Finite-difference alignment check
# -----------------------------------------------------------------------------
if ARGS.run_fd_check:
    fd_fields = {
        "u_pos_x": uf_k,
        "u_pos_y": uf_k,
        "p_pos_": pf_k,
        "vs_neg_x": us_k,
        "vs_neg_y": us_k,
        "d_neg_x": disp_k,
        "d_neg_y": disp_k,
    }
    fd_coeffs = {f.name: f for f in (uf_k, pf_k, uf_n, pf_n, us_k, us_n, disp_k, disp_n)}
    fd_coeffs["dt"] = dt
    fd_kernel_pool = None
    if FD_BACKEND == "jit" and os.getenv("FD_REUSE_KERNELS", "1") != "0":
        from pycutfem.jit import compile_multi

        def _pool_from_form(form, kernels):
            pool: Dict[int, list] = {}
            for ker in kernels:
                key = getattr(ker, "integral_id", None)
                if key is None:
                    continue
                pool.setdefault(int(key), []).append(ker)
            if pool:
                return pool
            ints = getattr(form, "integrals", None)
            integrals = list(ints) if ints is not None else [form]
            return {id(intg): [ker] for intg, ker in zip(integrals, kernels)}

        debug_timing = os.getenv("FD_TIMING", "").lower() in {"1", "true", "yes"}
        if debug_timing:
            print("[FD timing] precompiling full-form kernels for reuse...")
        t0 = time.perf_counter() if debug_timing else None
        full_kernels_K = compile_multi(
            jacobian_form,
            dof_handler=dof_handler,
            mixed_element=dof_handler.mixed_element,
            backend=FD_BACKEND,
        )
        if debug_timing and t0 is not None:
            print(f"[FD timing] full jac kernels compiled in {time.perf_counter() - t0:.3f}s")
        t0 = time.perf_counter() if debug_timing else None
        full_kernels_R = compile_multi(
            residual_form,
            dof_handler=dof_handler,
            mixed_element=dof_handler.mixed_element,
            backend=FD_BACKEND,
        )
        if debug_timing and t0 is not None:
            print(f"[FD timing] full res kernels compiled in {time.perf_counter() - t0:.3f}s")
        fd_kernel_pool = {
            "jac": _pool_from_form(jacobian_form, full_kernels_K),
            "res": _pool_from_form(residual_form, full_kernels_R),
        }
    bc_probe_src = bcs if bcs else bcs_homog
    bc_probe_dofs = set(dof_handler.get_dirichlet_data(bc_probe_src).keys()) if bc_probe_src else set()
    inactive_probe = set(dof_handler.dof_tags.get("inactive", set()))
    probe_sets = {
        "cut": {
            "u_pos_x": 2,
            "u_pos_y": 2,
            "p_pos_": 2,
            "vs_neg_x": 2,
            "vs_neg_y": 2,
            "d_neg_x": 2,
            "d_neg_y": 2,
        },
        "outside": {"u_pos_x": 2, "u_pos_y": 2, "p_pos_": 2},
        "inside": {"vs_neg_x": 2, "vs_neg_y": 2, "d_neg_x": 2, "d_neg_y": 2},
    }
    probe, probe_by_tag = select_fd_probes(
        dof_handler,
        probe_sets,
        bc_dofs=bc_probe_dofs,
        inactive=inactive_probe,
    )
    tag_labels = {"cut": "cut", "outside": "fluid(outside)", "inside": "solid(inside)"}
    for tag, dofs in probe_by_tag.items():
        label = tag_labels.get(tag, tag)
        print(f"[FD probe] {label}: {len(dofs)} dofs")
    if probe.size == 0:
        print("[FD probe] warning: no active DOFs selected for FD checks.")
    if os.getenv("FD_SKIP_FULL", "0") == "0":
        finite_difference_check(
            jacobian_form,
            residual_form,
            dof_handler,
            bcs,
            fd_fields,
            probe,
            eps=1.0e-7,
            bcs_elim=bcs_homog,
            coeffs=fd_coeffs,
            label="full",
            kernel_pool=fd_kernel_pool,
        )
    if ARGS.run_fd_terms:
        split_terms = os.getenv("FD_TERM_SPLIT", "0") != "0"
        split_interface = os.getenv("FD_INTERFACE_SPLIT", "0") != "0"
        if split_terms:
            term_blocks = {
                "fluid_vol": (a_vol_f, r_vol_f),
                "solid_vol": (a_vol_s, r_vol_s),
                "solid_vel_constraint": (a_svc, r_svc),
                "interface": (J_int, R_int),
                "cip": (a_cip, r_cip),
                "stab_fluid_vel": (
                    (Constant(2.0) * mu_f_const * g_v_f(gamma_v_f, du_f_R, test_vel_f_R)) * dG_fluid,
                    (Constant(2.0) * mu_f_const * g_v_f(gamma_v_f, uf_k_R, test_vel_f_R)) * dG_fluid,
                ),
                "stab_fluid_p": (
                    g_p(gamma_p_f, dp_f_R, test_q_f_R) * dG_fluid,
                    g_p(gamma_p_f, pf_k_R, test_q_f_R) * dG_fluid,
                ),
                "pressure_stab": (a_p_stab, r_p_stab),
                "stab_solid_vel": (
                    (rho_s_const * g_v_s(gamma_v_s, du_s_R, test_vel_s_R)) * dG_solid,
                    (rho_s_const * g_v_s(gamma_v_s, us_k_R, test_vel_s_R)) * dG_solid,
                ),
                "stab_solid_disp": (
                    (Constant(2.0) * mu_s_const * g_disp_s(gamma_disp_s, ddisp_s_R, test_disp_s_R)) * dG_solid,
                    (Constant(2.0) * mu_s_const * g_disp_s(gamma_disp_s, disp_k_R, test_disp_s_R)) * dG_solid,
                ),
            }
        else:
            term_blocks = {
                "fluid_vol": (a_vol_f, r_vol_f),
                "solid_vol": (a_vol_s, r_vol_s),
                "solid_vel_constraint": (a_svc, r_svc),
                "interface": (J_int, R_int),
                "stab": (a_stab, r_stab),
                "pressure_stab": (a_p_stab, r_p_stab),
                "cip": (a_cip, r_cip),
                "reg": (a_reg, r_reg),
            }
        if split_interface:
            term_blocks.pop("interface", None)
            term_blocks.update(
                {
                    "interface_fluid": (J_int_fluid, R_int_fluid),
                    "interface_solid": (J_int_solid, R_int_solid),
                    "interface_penalty": (J_int_pen, R_int_pen),
                }
            )
            if s_nitsche_value != 0.0:
                term_blocks.update(
                    {
                        "interface_sym_fluid": (J_int_sym_fluid, R_int_sym_fluid),
                        "interface_sym_solid": (J_int_sym_solid, R_int_sym_solid),
                    }
                )
        term_filter = os.getenv("FD_TERM", "").strip()
        if term_filter:
            term_blocks = {k: v for k, v in term_blocks.items() if term_filter in k}
            if not term_blocks:
                print(f"[FD term] no terms match filter '{term_filter}'")
        for name, (jf, rf) in term_blocks.items():
            print(f"\n[FD term] {name}")
            finite_difference_check(
                jf,
                rf,
                dof_handler,
                bcs,
                fd_fields,
                probe,
                eps=1.0e-7,
                bcs_elim=bcs_homog,
                coeffs=fd_coeffs,
                label=name,
                kernel_pool=fd_kernel_pool,
            )

# -----------------------------------------------------------------------------
# Time stepping
# -----------------------------------------------------------------------------
if ARGS.run_time_stepping:
    max_steps = int(os.getenv("MAX_STEPS", "50"))
    dt_adapt_on_success = _env_bool("DT_ADAPT_ON_SUCCESS", False)
    dt_increase_factor = float(os.getenv("DT_INCREASE_FACTOR", "1.1")) if dt_adapt_on_success else 1.0
    dt_decrease_factor_slow = float(os.getenv("DT_DECREASE_FACTOR_SLOW", "0.9")) if dt_adapt_on_success else 1.0

    def _on_dt_change(new_dt: float) -> None:
        try:
            dt.value = float(new_dt)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Optional restart: load level set + solution snapshots from a prior run.
    # -------------------------------------------------------------------------
    restart_payload = None
    restart_t0 = 0.0
    restart_step0 = 0
    restart_step_idx0 = 0
    dt0 = float(DT)

    restart_dir = getattr(ARGS, "restart_dir", None)
    restart_step = int(getattr(ARGS, "restart_step", -1))
    restart_tag = str(getattr(ARGS, "restart_tag", "step"))
    restart_reset = bool(getattr(ARGS, "restart_reset_counters", False))

    if restart_dir is not None and restart_step >= 0:
        restart_payload = _load_restart_payload(restart_dir=Path(restart_dir), step_no=restart_step, tag=restart_tag)
        dt0 = float(restart_payload.get("dt", DT))
        restart_t0 = float(restart_payload.get("t", 0.0))
        loaded_step = int(restart_payload.get("step", restart_step))

        try:
            dt.value = float(dt0)
        except Exception:
            pass

        phi_arr = np.asarray(restart_payload.get("phi", np.array([])), dtype=float)
        if phi_arr.size:
            ls_beam.set_from_array(phi_arr)
            ls_beam.commit(tol=LEVELSET_EDGE_TOL)

            refresh_domains(mesh, domains)
            refresh_hansbo_kappa(
                mesh,
                ls_beam,
                theta_pos_vals,
                theta_neg_vals,
                theta_pos_raw_vals=theta_pos_raw_vals,
                theta_neg_raw_vals=theta_neg_raw_vals,
                theta_min=theta_min,
            )
            refresh_sliver_weights(
                mesh,
                theta_pos_vals,
                theta_neg_vals,
                w_pos_vals,
                w_neg_vals,
                theta0=SLIVER_THETA0,
                p=SLIVER_P,
                wmax=SLIVER_WMAX,
                thetamin=SLIVER_THETAMIN,
                smooth=SLIVER_SMOOTH,
            )
            log_sliver_stats(mesh, theta_pos_vals, theta_neg_vals, w_pos_vals, w_neg_vals, theta0=SLIVER_THETA0)
            retag_inactive(dof_handler, theta_neg=theta_neg_raw_vals, solid_cut_drop=SOLID_CUT_DROP)

        state_npz = restart_payload.get("state", {})

        def _restore_function(f, arr, *, label: str) -> None:
            vec = np.asarray(arr, dtype=float).ravel()
            target = getattr(f, "nodal_values", np.array([]))
            if target.shape == vec.shape:
                f.nodal_values[:] = vec
                return

            # Backward/forward compatible restart for pressure when switching between
            # equal-order (Q2) and Taylor–Hood (Q1) pressure. We map between:
            # - Q1 pressure vector: values on unique corner nodes (len == n_vertices)
            # - Q2 pressure vector: values on all mesh nodes (len == n_nodes)
            if getattr(f, "field_name", "") == "p_pos_":
                try:
                    n_nodes = int(np.asarray(mesh.nodes_x_y_pos).shape[0])
                except Exception:
                    n_nodes = int(vec.size)
                corner_conn = getattr(mesh, "corner_connectivity", None)
                if corner_conn is not None:
                    vertex_ids = np.unique(np.asarray(corner_conn, dtype=int).ravel())
                    n_vertices = int(vertex_ids.size)
                    if int(vec.size) == n_vertices and int(target.size) == n_nodes:
                        full = np.full((n_nodes,), np.nan, dtype=float)
                        full[vertex_ids] = vec.astype(float, copy=False)

                        # Mid-edge nodes (when present): average endpoints
                        edges_conn = getattr(mesh, "edges_connectivity", None)
                        if edges_conn is not None:
                            for nodes in np.asarray(edges_conn, dtype=int):
                                if nodes.size < 3:
                                    continue
                                a = int(nodes[0])
                                b = int(nodes[-1])
                                va = full[a]
                                vb = full[b]
                                if not (np.isfinite(va) and np.isfinite(vb)):
                                    continue
                                mids = nodes[1:-1]
                                if mids.size:
                                    full[mids] = 0.5 * (va + vb)

                        # Remaining nodes (e.g. element centers): average of element corners
                        try:
                            elems = np.asarray(mesh.elements_connectivity, dtype=int)
                            corners = np.asarray(mesh.corner_connectivity, dtype=int)
                            for eid in range(int(elems.shape[0])):
                                cn = corners[eid]
                                if cn.size == 0:
                                    continue
                                val = float(np.nanmean(full[cn]))
                                if not np.isfinite(val):
                                    continue
                                for nid in elems[eid]:
                                    nid = int(nid)
                                    if 0 <= nid < n_nodes and not np.isfinite(full[nid]):
                                        full[nid] = val
                        except Exception:
                            pass

                        full = np.nan_to_num(full, nan=0.0)
                        f.nodal_values[:] = full
                        print(f"[restart] mapped {label} from Q1({n_vertices}) → Q2({n_nodes}) pressure.")
                        return
                    if int(vec.size) == n_nodes and int(target.size) == n_vertices:
                        f.nodal_values[:] = np.asarray(vec, dtype=float)[vertex_ids]
                        print(f"[restart] mapped {label} from Q2({n_nodes}) → Q1({n_vertices}) pressure.")
                        return

            raise ValueError(
                f"Restart mismatch for {label}: got {vec.shape}, expected {target.shape}."
            )

        # Load current iterate and previous snapshot when available.
        field_pairs = [(uf_k, uf_n), (pf_k, pf_n), (us_k, us_n), (disp_k, disp_n)]
        loaded_prev: set[str] = set()
        for f_cur, f_prev in field_pairs:
            name_k = getattr(f_cur, "name", "") or ""
            name_n = getattr(f_prev, "name", "") or ""
            key_k = f"{name_k}_k" if name_k else ""
            key_n = f"{name_n}_n" if name_n else ""
            if key_k and key_k in state_npz:
                _restore_function(f_cur, state_npz[key_k], label=key_k)
            if key_n and key_n in state_npz:
                _restore_function(f_prev, state_npz[key_n], label=key_n)
                loaded_prev.add(name_n)

        # If prev snapshots are missing (common for accepted-step dumps), mirror the current state.
        for f_cur, f_prev in field_pairs:
            name_n = getattr(f_prev, "name", "") or ""
            if name_n and name_n not in loaded_prev:
                f_prev.nodal_values[:] = f_cur.nodal_values[:]

        bcs_restart = NewtonSolver._freeze_bcs(bcs, float(restart_t0))
        dof_handler.apply_bcs(bcs_restart, uf_k, pf_k, us_k, disp_k)
        dof_handler.apply_bcs(bcs_restart, uf_n, pf_n, us_n, disp_n)
        if os.getenv("PYCUTFEM_ZERO_INACTIVE_SIDES", "0").lower() in {"1", "true", "yes"}:
            _zero_inactive_side_values(
                ls_beam=ls_beam,
                dof_handler=dof_handler,
                mesh=mesh,
                uf=[uf_k, uf_n],
                pf=[pf_k, pf_n],
                us=[us_k, us_n],
                disp=[disp_k, disp_n],
                tol=LEVELSET_EDGE_TOL,
            )
        if os.getenv("PYCUTFEM_REEXTEND_WRONG_SIDE", "0").lower() in {"1", "true", "yes"}:
            stats = _reextend_wrong_side_by_nearest(
                mesh=mesh,
                dof_handler=dof_handler,
                ls_beam=ls_beam,
                uf=[uf_k, uf_n],
                pf=[pf_k, pf_n],
                us=[us_k, us_n],
                disp=[disp_k, disp_n],
                tol=LEVELSET_EDGE_TOL,
                only_cut_nodes=True,
                trace=os.getenv("PYCUTFEM_REEXTEND_TRACE", "").lower() in {"1", "true", "yes"},
            )
            if any(int(v) for v in stats.values()):
                print(
                    "[reextend] wrong-side copied by nearest: "
                    + ", ".join(f"{k}={int(v)}" for k, v in stats.items() if int(v))
                )
                dof_handler.apply_bcs(bcs_restart, uf_k, pf_k, us_k, disp_k, uf_n, pf_n, us_n, disp_n)
        if os.getenv("PYCUTFEM_STITCH_INTERFACE_VELOCITY", "0").lower() in {"1", "true", "yes"}:
            try:
                band = float(
                    os.getenv("PYCUTFEM_STITCH_INTERFACE_BAND", str(max(0.5 * MESH_SIZE, 2.0 * LEVELSET_EDGE_TOL)))
                )
            except Exception:
                band = float(max(0.5 * MESH_SIZE, 2.0 * LEVELSET_EDGE_TOL))
            n0 = _stitch_fluid_velocity_to_solid_near_interface(
                mesh=mesh,
                dof_handler=dof_handler,
                ls_beam=ls_beam,
                uf=[uf_k, uf_n],
                us=[us_k, us_n],
                band=band,
                only_cut_nodes=True,
            )
            if n0:
                print(f"[stitch] copied solid→fluid velocity at {n0} cut-band nodes (band={band:.3e})")
                dof_handler.apply_bcs(bcs_restart, uf_k, pf_k, us_k, disp_k, uf_n, pf_n, us_n, disp_n)

        # Step offsets:
        # - tag="step": dumps are 0-based indices of completed steps, so next step is (idx+2).
        # - tag="fail": dumps are tied to the failing step number (1-based).
        if restart_tag == "step":
            completed = int(loaded_step) + 1
            restart_step0 = 0 if restart_reset else completed
            restart_step_idx0 = 0 if restart_reset else completed
        else:
            fail_step_no = int(loaded_step)
            completed = max(0, fail_step_no - 1)
            restart_step0 = 0 if restart_reset else max(0, fail_step_no - 1)
            restart_step_idx0 = 0 if restart_reset else completed

        if restart_reset:
            print(
                f"[restart] reset counters; starting new run at physical t0={restart_t0:.6f} (dt={dt0:.3e})."
            )
        else:
            print(
                f"[restart] loaded step={loaded_step} tag={restart_tag} → t0={restart_t0:.6f} dt={dt0:.3e} "
                f"(solver step0={restart_step0}, dump step_idx0={restart_step_idx0})."
            )

    # If a final time is requested, ensure max_steps is large enough to reach it.
    final_time_target = getattr(ARGS, "final_time", None)
    if final_time_target is not None:
        remaining = max(0.0, float(final_time_target) - float(restart_t0))
        needed_steps = int(math.ceil(remaining / max(float(dt0), 1.0e-16)))
        if needed_steps > int(max_steps):
            print(f"[time] max_steps {max_steps} → {needed_steps} to reach final_time={float(final_time_target):.6f}.")
            max_steps = int(needed_steps)

    time_params = TimeStepperParameters(
        dt=dt0,
        max_steps=max_steps,
        stop_on_steady=True,
        steady_tol=1e-6,
        t0=float(restart_t0),
        step0=int(restart_step0),
        final_time=float(ARGS.final_time) if getattr(ARGS, "final_time", None) is not None else None,
        theta=theta.value,
        allow_dt_reduction=bool(getattr(ARGS, "allow_dt_reduction", False)),
        dt_reduction_factor=float(getattr(ARGS, "dt_reduction_factor", 0.5)),
        dt_reduction_threshold=float(getattr(ARGS, "dt_reduction_threshold", 5.0)),
        dt_min=float(os.getenv("DT_MIN", "1e-4")),
        dt_increase_factor=dt_increase_factor,
        dt_decrease_factor_slow=dt_decrease_factor_slow,
        on_dt_change=_on_dt_change,
    )
    newton_tol = float(os.getenv("NEWTON_TOL", "1e-8"))
    newton_rtol = float(os.getenv("NEWTON_RTOL", "1e-8"))
    max_newton_iter = int(os.getenv("MAX_NEWTON_ITER", "20"))
    ls_mode = os.getenv("LS_MODE", "armijo")
    ls_max_iter = int(os.getenv("LS_MAX_ITER", "12"))
    newton_params = NewtonParameters(
        newton_tol=newton_tol,
        newton_rtol=newton_rtol,
        max_newton_iter=max_newton_iter,
        line_search=True,
        ls_mode=ls_mode,
        ls_max_iter=ls_max_iter,
    )
    nonlinear_solver = getattr(ARGS, "nonlinear_solver", "newton")
    jit_backend_choice = str(getattr(ARGS, "jit_backend", "numba") or "numba").strip().lower()
    assembly_backend = "cpp" if jit_backend_choice in {"cpp", "c++"} else "jit"
    if nonlinear_solver == "snes":
        solver = PetscSnesNewtonSolver(
            residual_form,
            jacobian_form,
            dof_handler=dof_handler,
            mixed_element=mixed_element,
            bcs=bcs,
            bcs_homog=bcs_homog,
            newton_params=newton_params,
            backend=assembly_backend,
        )
    else:
        solver = NewtonSolver(
            residual_form,
            jacobian_form,
            dof_handler=dof_handler,
            mixed_element=mixed_element,
            bcs=bcs,
            bcs_homog=bcs_homog,
            newton_params=newton_params,
            lin_params=LinearSolverParameters(backend=str(getattr(ARGS, "linear_solver", "scipy"))),
            backend=assembly_backend,
        )

    if os.getenv("PYCUTFEM_ACTIVE_DOFS_TRACE", "").lower() in {"1", "true", "yes"}:
        active = np.asarray(solver.active_dofs, dtype=int)
        constraints = getattr(solver, "constraints", None)
        if constraints is not None:
            try:
                active = np.asarray(constraints.master_ids, dtype=int)[active]
            except Exception:
                pass
        counts = []
        for fld in getattr(dof_handler, "field_names", []):
            try:
                sl = np.asarray(dof_handler.get_field_slice(fld), dtype=int)
            except Exception:
                continue
            if sl.size == 0:
                continue
            n_active = np.intersect1d(active, sl).size
            counts.append(f"{fld}:{n_active}")
        if counts:
            print("[active] DOFs by field: " + ", ".join(counts))

    # Keep unknowns on their physical side: prevent inactive-side DOFs from drifting.
    zero_inactive_every_newton = os.getenv("PYCUTFEM_ZERO_INACTIVE_SIDES", "0").lower() in {"1", "true", "yes"}
    if zero_inactive_every_newton:
        def _post_newton_zero_inactive(funcs_now):
            try:
                uf_now, pf_now, us_now, disp_now = funcs_now
            except Exception:
                return
            _zero_inactive_side_values(
                ls_beam=ls_beam,
                dof_handler=dof_handler,
                mesh=mesh,
                uf=[uf_now],
                pf=[pf_now],
                us=[us_now],
                disp=[disp_now],
                tol=LEVELSET_EDGE_TOL,
            )

        solver.post_cb = _post_newton_zero_inactive

    # -------------------------------------------------------------------------
    # Debug hooks: penalty retry + failure dumps (no adaptive dt ping-pong)
    # -------------------------------------------------------------------------
    base_beta_N = float(getattr(beta_N, "value", beta_N))
    base_gamma_v_f0 = float(getattr(gamma_v_f0, "value", gamma_v_f0))
    base_gamma_p_f0 = float(getattr(gamma_p_f0, "value", gamma_p_f0))
    base_gamma_v_s0 = float(getattr(gamma_v_s0, "value", gamma_v_s0))
    base_gamma_disp_s0 = float(getattr(gamma_disp_s0, "value", gamma_disp_s0))
    base_gamma_v_f_mass0 = float(getattr(gamma_v_f_mass0, "value", gamma_v_f_mass0))
    base_gamma_v_s_mass0 = float(getattr(gamma_v_s_mass0, "value", gamma_v_s_mass0))
    base_gamma_sliver_mass_f0 = float(getattr(gamma_sliver_mass_f0, "value", gamma_sliver_mass_f0))
    base_gamma_sliver_mass_s0 = float(getattr(gamma_sliver_mass_s0, "value", gamma_sliver_mass_s0))
    base_gamma_p_stab_jump0 = float(getattr(gamma_p_stab_jump0, "value", gamma_p_stab_jump0))
    base_gamma_p_stab_avg0 = float(getattr(gamma_p_stab_avg0, "value", gamma_p_stab_avg0))

    def _set_penalty_scale(scale: float) -> None:
        s = float(scale)
        beta_N.value = base_beta_N * s
        gamma_v_f0.value = base_gamma_v_f0 * s
        gamma_p_f0.value = base_gamma_p_f0 * s
        gamma_v_s0.value = base_gamma_v_s0 * s
        gamma_disp_s0.value = base_gamma_disp_s0 * s
        gamma_v_f_mass0.value = base_gamma_v_f_mass0 * s
        gamma_v_s_mass0.value = base_gamma_v_s_mass0 * s
        gamma_sliver_mass_f0.value = base_gamma_sliver_mass_f0 * s
        gamma_sliver_mass_s0.value = base_gamma_sliver_mass_s0 * s
        gamma_p_stab_jump0.value = base_gamma_p_stab_jump0 * s
        gamma_p_stab_avg0.value = base_gamma_p_stab_avg0 * s

    retry_penalties = os.getenv("PYCUTFEM_RETRY_PENALTIES", "").lower() in {"1", "true", "yes"}
    retry_scales_raw = os.getenv("PYCUTFEM_RETRY_PENALTY_SCALES", "1.0,2.0,5.0,10.0")
    retry_scales: list[float] = []
    if retry_penalties:
        for tok in retry_scales_raw.replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                retry_scales.append(float(tok))
            except Exception:
                pass
    retry_state = {"step": None, "idx": 0}

    def _on_step_failure(**ctx):
        step = int(ctx.get("step_no", ctx.get("step", 0)))
        t_fail = float(ctx.get("t", 0.0))
        dt_fail = float(ctx.get("dt", DT))
        exc = ctx.get("exception", None)
        print(f"[fail] step={step} t={t_fail:.6f} dt={dt_fail:.3e} exc={exc}")

        _peclet_diagnostics(
            f"pe_fail_step{step:04d}",
            step=step,
            t=t_fail,
            dt=dt_fail,
        )

        # Dump current level set state (and mesh once) for offline replay.
        _dump_levelset_step(step, t_fail, tag="fail")
        _dump_state_step(
            step,
            t_fail,
            tag="fail",
            funcs=list(ctx.get("functions", []) or []),
            prev_funcs=list(ctx.get("prev_functions", []) or []),
        )

        # Optional: term-by-term interface residual traces.
        _trace_interface_residuals(f"ifc_fail_step{step:04d}", bcs_apply=ctx.get("bcs", bcs))

        if not retry_penalties or not retry_scales:
            return False

        if retry_state["step"] != step:
            retry_state["step"] = step
            retry_state["idx"] = 0

        idx = int(retry_state["idx"])
        if idx >= len(retry_scales):
            print("[retry] penalty scales exhausted; not retrying.")
            return False

        scale = float(retry_scales[idx])
        retry_state["idx"] = idx + 1
        _set_penalty_scale(scale)
        print(
            "[retry] scaling penalties by {:.3g} and retrying Newton at same dt "
            "(set PYCUTFEM_ABORT_ON_DT_REDUCTION=1 to stop on dt reduction).".format(scale)
        )
        # Keep the last iterate as the initial guess (more informative for debugging).
        return "retry_keep_guess"

    time_params.on_step_failure = _on_step_failure

    # Optional: run FD Jacobian check at a specific time step (1-based).
    # Enable with: FD_STEP=2 (uses existing FD_* env vars for backend/term filters).
    fd_step_target = int(os.getenv("FD_STEP", "0") or "0")
    fd_step_ran: set[int] = set()
    if fd_step_target > 0:
        fd_fields_step = {
            "u_pos_x": uf_k,
            "u_pos_y": uf_k,
            "p_pos_": pf_k,
            "vs_neg_x": us_k,
            "vs_neg_y": us_k,
            "d_neg_x": disp_k,
            "d_neg_y": disp_k,
        }
        fd_coeffs_step = {f.name: f for f in (uf_k, pf_k, uf_n, pf_n, us_k, us_n, disp_k, disp_n)}
        fd_coeffs_step["dt"] = dt
        bc_probe_dofs_step = set(dof_handler.get_dirichlet_data(bcs_homog or bcs).keys())
        inactive_probe_step = set(dof_handler.dof_tags.get("inactive", set()))
        probe_sets_step = {
            "cut": {
                "u_pos_x": 6,
                "u_pos_y": 6,
                "p_pos_": 6,
                "vs_neg_x": 4,
                "vs_neg_y": 4,
                "d_neg_x": 4,
                "d_neg_y": 4,
            },
            "outside": {"u_pos_x": 2, "u_pos_y": 2, "p_pos_": 2},
            "inside": {"vs_neg_x": 2, "vs_neg_y": 2, "d_neg_x": 2, "d_neg_y": 2},
        }
        probe_step, _ = select_fd_probes(
            dof_handler,
            probe_sets_step,
            bc_dofs=bc_probe_dofs_step,
            inactive=inactive_probe_step,
        )
        fd_step_only = os.getenv("FD_STEP_ONLY", "").strip()
        if fd_step_only:
            try:
                probe_step = np.array([int(fd_step_only)], dtype=int)
            except Exception:
                pass

        fd_kernel_pool_step = None
        if FD_BACKEND == "jit" and os.getenv("FD_REUSE_KERNELS", "1") != "0":
            def _pool_from_kernels(kernels):
                pool: Dict[int, list] = {}
                for ker in kernels:
                    key = getattr(ker, "integral_id", None)
                    if key is None:
                        continue
                    pool.setdefault(int(key), []).append(ker)
                return pool if pool else None

            pool_jac = _pool_from_kernels(getattr(solver, "kernels_K", []))
            pool_res = _pool_from_kernels(getattr(solver, "kernels_F", []))
            if pool_jac is not None and pool_res is not None:
                fd_kernel_pool_step = {"jac": pool_jac, "res": pool_res}

        prev_pre_cb = getattr(solver, "pre_cb", None)

        def _fd_pre_cb(funcs):
            if callable(prev_pre_cb):
                prev_pre_cb(funcs)
            step_no = step_idx[0] + 1  # step_idx counts completed steps
            if step_no != fd_step_target or step_no in fd_step_ran:
                return
            bcs_now = getattr(solver, "_current_bcs", None) or bcs
            print(f"[FD step] running FD check at step {step_no} (backend={FD_BACKEND})")
            finite_difference_check(
                jacobian_form,
                residual_form,
                dof_handler,
                bcs_now,
                fd_fields_step,
                probe_step,
                eps=1.0e-7,
                bcs_elim=bcs_homog,
                coeffs=fd_coeffs_step,
                label=f"step{step_no}",
                kernel_pool=fd_kernel_pool_step,
            )
            fd_step_ran.add(step_no)

        solver.pre_cb = _fd_pre_cb

    # Optional: print Peclet diagnostics once per (global) time step (at Newton it=0).
    if os.getenv("PYCUTFEM_PECLET_TRACE", "").lower() in {"1", "true", "yes"}:
        pe_state = {"step": None}
        prev_pre_cb = getattr(solver, "pre_cb", None)

        def _pe_pre_cb(_funcs):  # noqa: ANN001
            step_no = int(getattr(solver, "_current_step_no", -1))
            if pe_state["step"] == step_no:
                return
            pe_state["step"] = step_no
            _peclet_diagnostics(
                f"pe_step{step_no:04d}",
                step=step_no,
                t=float(getattr(solver, "_current_t", 0.0)),
                dt=float(getattr(solver, "_current_dt", DT)),
            )

        if prev_pre_cb is None:
            solver.pre_cb = _pe_pre_cb
        else:
            def _combined_pre_cb(funcs):  # noqa: ANN001
                _pe_pre_cb(funcs)
                prev_pre_cb(funcs)

            solver.pre_cb = _combined_pre_cb

    # Optional: cheap per-Newton-iteration Peclet estimate (DOF-wise |u|_inf scaling).
    # This is useful because the step-level Pe diagnostic is evaluated at Newton it=0,
    # while some failures only appear after Newton updates blow up the velocity field.
    if os.getenv("PYCUTFEM_PECLET_TRACE_NEWTON", "").lower() in {"1", "true", "yes"}:
        try:
            ux_gdofs = np.asarray(dof_handler.get_field_slice("u_pos_x"), dtype=int)
            uy_gdofs = np.asarray(dof_handler.get_field_slice("u_pos_y"), dtype=int)
            ux_nodes = np.array([dof_handler._dof_to_node_map[int(gd)][1] for gd in ux_gdofs], dtype=int)
            uy_nodes = np.array([dof_handler._dof_to_node_map[int(gd)][1] for gd in uy_gdofs], dtype=int)
        except Exception:
            ux_gdofs = np.empty(0, dtype=int)
            uy_gdofs = np.empty(0, dtype=int)
            ux_nodes = np.empty(0, dtype=int)
            uy_nodes = np.empty(0, dtype=int)

        try:
            h_diag = float(os.getenv("PYCUTFEM_PECLET_H", str(MESH_SIZE)) or str(MESH_SIZE))
        except Exception:
            h_diag = float(MESH_SIZE)
        pe_scale = float(RHO_F) * float(h_diag) / (2.0 * float(MU_F) + 1.0e-30)

        pe_newton_state: dict[str, object] = {"step": None, "it": 0, "mask_x": None, "mask_y": None}
        prev_post_cb = getattr(solver, "post_cb", None)

        def _pe_post_cb(funcs):  # noqa: ANN001
            if callable(prev_post_cb):
                prev_post_cb(funcs)
            try:
                uf_now = funcs[0]
            except Exception:
                return
            step_no = int(getattr(solver, "_current_step_no", -1))
            t_now = float(getattr(solver, "_current_t", 0.0))
            dt_now = float(getattr(solver, "_current_dt", DT))
            if pe_newton_state["step"] != step_no:
                pe_newton_state["step"] = step_no
                pe_newton_state["it"] = 0
                try:
                    cut_eids = mesh.element_bitset("cut").to_indices()
                    if cut_eids.size:
                        cut_nodes = np.unique(mesh.elements_connectivity[cut_eids].ravel())
                        node_in_cut = np.zeros(int(mesh.nodes_x_y_pos.shape[0]), dtype=bool)
                        node_in_cut[cut_nodes.astype(int, copy=False)] = True
                        pe_newton_state["mask_x"] = node_in_cut[ux_nodes] if ux_nodes.size else None
                        pe_newton_state["mask_y"] = node_in_cut[uy_nodes] if uy_nodes.size else None
                    else:
                        pe_newton_state["mask_x"] = None
                        pe_newton_state["mask_y"] = None
                except Exception:
                    pe_newton_state["mask_x"] = None
                    pe_newton_state["mask_y"] = None

            pe_newton_state["it"] = int(pe_newton_state.get("it", 0)) + 1
            it_no = int(pe_newton_state["it"])

            try:
                v = np.asarray(getattr(uf_now, "nodal_values", np.array([])), dtype=float).ravel()
            except Exception:
                return
            nux = int(ux_gdofs.size)
            nuy = int(uy_gdofs.size)
            if nux <= 0 or nuy <= 0 or v.size < (nux + nuy):
                return
            ux = v[:nux]
            uy = v[nux : nux + nuy]
            u_inf = float(max(np.max(np.abs(ux)), np.max(np.abs(uy))))
            pe_inf = pe_scale * u_inf

            u_inf_cut = float("nan")
            try:
                mx = pe_newton_state.get("mask_x", None)
                my = pe_newton_state.get("mask_y", None)
                if mx is not None and my is not None:
                    mx = np.asarray(mx, dtype=bool)
                    my = np.asarray(my, dtype=bool)
                    if mx.size == ux.size and my.size == uy.size and (np.any(mx) or np.any(my)):
                        ux_inf = float(np.max(np.abs(ux[mx]))) if np.any(mx) else 0.0
                        uy_inf = float(np.max(np.abs(uy[my]))) if np.any(my) else 0.0
                        u_inf_cut = float(max(ux_inf, uy_inf))
            except Exception:
                u_inf_cut = float("nan")

            pe_cut = pe_scale * float(u_inf_cut) if np.isfinite(u_inf_cut) else float("nan")
            print(
                f"[pe_newton] step={step_no} it={it_no} t={t_now:.6f} dt={dt_now:.3e} "
                f"|u|_inf={u_inf:.3e} Pe_inf≈{pe_inf:.3e} "
                f"|u|_inf_cut={u_inf_cut:.3e} Pe_cut≈{pe_cut:.3e}"
            )

        solver.post_cb = _pe_post_cb

    step_idx = [int(restart_step_idx0)]  # mutable so the closure can update
    t_accum = [float(restart_t0)]

    def post_step_cb(funcs):
        dt_step = float(dt.value) if hasattr(dt, "value") else float(DT)
        t_curr = float(t_accum[0]) + dt_step
        old_active = np.asarray(getattr(solver, "active_dofs", []), dtype=int).copy()
        ls_changed = update_beam_levelset_from_displacement(
            ls_beam,
            disp_k,
            beam_ref_ls,
            nudge_eps=LEVELSET_NUDGE_EPS,
            snap_eps=LEVELSET_SNAP_EPS,
            edge_tol=LEVELSET_EDGE_TOL,
            prefer_negative=LEVELSET_PREFER_NEGATIVE,
            update_tol=LEVELSET_UPDATE_TOL,
        )
        if ls_changed:
            refresh_domains(mesh, domains)
            refresh_hansbo_kappa(
                mesh,
                ls_beam,
                theta_pos_vals,
                theta_neg_vals,
                theta_pos_raw_vals=theta_pos_raw_vals,
                theta_neg_raw_vals=theta_neg_raw_vals,
                theta_min=theta_min,
            )
            refresh_sliver_weights(
                mesh,
                theta_pos_vals,
                theta_neg_vals,
                w_pos_vals,
                w_neg_vals,
                theta0=SLIVER_THETA0,
                p=SLIVER_P,
                wmax=SLIVER_WMAX,
                thetamin=SLIVER_THETAMIN,
                smooth=SLIVER_SMOOTH,
            )
            log_sliver_stats(
                mesh,
                theta_pos_vals,
                theta_neg_vals,
                w_pos_vals,
                w_neg_vals,
                theta0=SLIVER_THETA0,
            )
            ls_stats = getattr(ls_beam, "_last_update_stats", None)
            if isinstance(ls_stats, dict) and ls_stats.get("changed", False):
                print(
                    "[levelset] max|Δφ|={:.3e} max|Δφ|_cut={:.3e} cut_elems={} interface_edges={}".format(
                        float(ls_stats.get("max_diff", 0.0)),
                        float(ls_stats.get("max_diff_cut", 0.0)),
                        int(ls_stats.get("cut_elems", 0)),
                        int(ls_stats.get("interface_edges", 0)),
                    )
                )
            retag_inactive(dof_handler, theta_neg=theta_neg_raw_vals, solid_cut_drop=SOLID_CUT_DROP)
            solver.refresh_levelset_kernels(ls_beam)
        # Re-apply time-dependent BCs at the *current* physical time (t_curr) so
        # post-step diagnostics and restart dumps remain consistent.
        bcs_curr = NewtonSolver._freeze_bcs(bcs, t_curr)
        dof_handler.apply_bcs(bcs_curr, uf_k, pf_k, us_k, disp_k, uf_n, pf_n, us_n, disp_n)
        if os.getenv("PYCUTFEM_ZERO_INACTIVE_SIDES", "0").lower() in {"1", "true", "yes"}:
            _zero_inactive_side_values(
                ls_beam=ls_beam,
                dof_handler=dof_handler,
                mesh=mesh,
                uf=[uf_k, uf_n],
                pf=[pf_k, pf_n],
                us=[us_k, us_n],
                disp=[disp_k, disp_n],
                tol=LEVELSET_EDGE_TOL,
            )
        if ls_changed:
            active_changed = recompute_active_dofs(solver, solver.bcs_homog if solver.bcs_homog else solver.bcs)
            extend_enabled = os.getenv("PYCUTFEM_EXTEND_NEWLY_ACTIVE_DOFS", "1").lower() in {"1", "true", "yes"}
            if extend_enabled and active_changed:
                constraints = getattr(solver, "constraints", None)
                new_active = np.asarray(getattr(solver, "active_dofs", []), dtype=int)
                if constraints is not None:
                    try:
                        old_full = np.asarray(constraints.master_ids, dtype=int)[np.asarray(old_active, dtype=int)]
                        new_full = np.asarray(constraints.master_ids, dtype=int)[np.asarray(new_active, dtype=int)]
                    except Exception:
                        old_full = np.asarray(old_active, dtype=int)
                        new_full = np.asarray(new_active, dtype=int)
                else:
                    old_full = np.asarray(old_active, dtype=int)
                    new_full = np.asarray(new_active, dtype=int)
                newly_active = np.setdiff1d(new_full, old_full, assume_unique=False)
                if newly_active.size:
                    extend_newly_active_dofs_nearest(
                        dh=dof_handler,
                        newly_active=newly_active,
                        active_old=old_full,
                        active_new=new_full,
                        field_to_current={
                            "u_pos_x": uf_k,
                            "u_pos_y": uf_k,
                            "p_pos_": pf_k,
                            "vs_neg_x": us_k,
                            "vs_neg_y": us_k,
                            "d_neg_x": disp_k,
                            "d_neg_y": disp_k,
                        },
                        field_to_prev={
                            "u_pos_x": uf_n,
                            "u_pos_y": uf_n,
                            "p_pos_": pf_n,
                            "vs_neg_x": us_n,
                            "vs_neg_y": us_n,
                            "d_neg_x": disp_n,
                            "d_neg_y": disp_n,
                        },
                        k=int(os.getenv("PYCUTFEM_EXTEND_NEWLY_ACTIVE_K", "4") or "4"),
                        trace=os.getenv("PYCUTFEM_EXTEND_NEWLY_ACTIVE_TRACE", "").lower()
	                        in {"1", "true", "yes"},
	                    )
            reextend_enabled = os.getenv("PYCUTFEM_REEXTEND_WRONG_SIDE", "0").lower() in {"1", "true", "yes"}
            if reextend_enabled:
                stats = _reextend_wrong_side_by_nearest(
                    mesh=mesh,
                    dof_handler=dof_handler,
                    ls_beam=ls_beam,
                    uf=[uf_k, uf_n],
                    pf=[pf_k, pf_n],
                    us=[us_k, us_n],
                    disp=[disp_k, disp_n],
                    tol=LEVELSET_EDGE_TOL,
                    only_cut_nodes=True,
                    trace=os.getenv("PYCUTFEM_REEXTEND_TRACE", "").lower() in {"1", "true", "yes"},
                )
                if any(int(v) for v in stats.values()):
                    print(
                        "[reextend] wrong-side copied by nearest: "
                        + ", ".join(f"{k}={int(v)}" for k, v in stats.items() if int(v))
                    )
                    dof_handler.apply_bcs(bcs_curr, uf_k, pf_k, us_k, disp_k, uf_n, pf_n, us_n, disp_n)
            stitch_enabled = os.getenv("PYCUTFEM_STITCH_INTERFACE_VELOCITY", "0").lower() in {"1", "true", "yes"}
            if stitch_enabled:
                try:
                    band = float(
                        os.getenv("PYCUTFEM_STITCH_INTERFACE_BAND", str(max(0.5 * MESH_SIZE, 2.0 * LEVELSET_EDGE_TOL)))
                    )
                except Exception:
                    band = float(max(0.5 * MESH_SIZE, 2.0 * LEVELSET_EDGE_TOL))
                n0 = _stitch_fluid_velocity_to_solid_near_interface(
                    mesh=mesh,
                    dof_handler=dof_handler,
                    ls_beam=ls_beam,
                    uf=[uf_k, uf_n],
                    us=[us_k, us_n],
                    band=band,
                    only_cut_nodes=True,
                )
                if n0:
                    print(f"[stitch] copied solid→fluid velocity at {n0} cut-band nodes (band={band:.3e})")
                    dof_handler.apply_bcs(bcs_curr, uf_k, pf_k, us_k, disp_k, uf_n, pf_n, us_n, disp_n)
        t_accum[0] = t_curr
        _dump_levelset_step(step_idx[0], t_curr, tag="step")
        _dump_state_step(
            step_idx[0],
            t_curr,
            tag="step",
            funcs=[uf_k, pf_k, us_k, disp_k],
            prev_funcs=[uf_n, pf_n, us_n, disp_n],
        )
        if COMPUTE_OBSERVABLES and int(OBS_EVERY) > 0 and (int(step_idx[0]) % int(OBS_EVERY)) == 0:
            _compute_observables(step_idx[0], t_curr)
        _probe_flow_velocity(step_idx[0])
        _plot_mesh(step_idx[0], title="Mesh / level-set")
        step_idx[0] += 1

    solver.post_timeloop_cb = post_step_cb

    solver.solve_time_interval(
        functions=[uf_k, pf_k, us_k, disp_k],
        prev_functions=[uf_n, pf_n, us_n, disp_n],
        time_params=time_params,
        aux_functions={"dt": dt},
    )
