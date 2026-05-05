#!/usr/bin/env python
# coding: utf-8

"""
CutFEM cylinder benchmark (DFG / Schäfer–Turek).

This script supports the DFG variants via `--benchmark`:
- `--benchmark 2d-1`: steady Navier–Stokes (Re=20)
- `--benchmark 2d-2`: unsteady, constant inflow (Re=100)
- `--benchmark 2d-3`: unsteady, time-dependent inflow (Re=100)

The inflow is selected with `--inflow`:
- `--inflow constant`: constant parabolic profile
- `--inflow dfg`: time-dependent ramp (FeatFlow-style): `U(t)=0.2+0.8*sin(pi t/8)`

Recommended run commands (fastest: compiled C++ backend)
--------------------------------------------------------
Run in the `xfemcustom` conda environment:

DFG 2D-1 (steady, Re=20):
  conda run --no-capture-output -n xfemcustom python examples/turek_cylinder/turek_benchmark.py \\
    --benchmark 2d-1 --backend cpp --with-deformation --init stokes \\
    --fe-order 2 --beta0 100 --ghost-measure patch --gamma-gp 0 \\
    --force-eval surface --disable-mass --theta 1 --max-steps 1 --vtk-every 0 --level 4

Constant inflow (good for debug / term-by-term parity):
  conda run --no-capture-output -n xfemcustom python examples/turek_cylinder/turek_benchmark.py \\
    --benchmark 2d-2 --backend cpp --with-deformation --inflow constant --init stokes \\
    --fe-order 2 --beta0 40 --ghost-measure patch --gamma-gp 1e-2 \\
    --dt 0.01 --theta 0.5 --max-steps 600 --vtk-every 20 --level 3 --newton-tol 1e-8

DFG time-dependent inflow (target for the benchmark 2D-3 setup; runs to t=8):
  conda run --no-capture-output -n xfemcustom python examples/turek_cylinder/turek_benchmark.py \\
    --benchmark 2d-3 --backend cpp --with-deformation --inflow dfg --init stokes \\
    --fe-order 2 --beta0 40 --ghost-measure patch --gamma-gp 1e-2 \\
    --dt 0.005 --theta 0.5 --max-steps 1600 --vtk-every 20 --level 3

Mesh level sweep (levels 1..6)
------------------------------
Each `--level k` uses a reproducible (nx, ny, refine-level) preset and writes to
a dedicated output directory (see below). Example sweep:

  for lv in 1 2 3 4 5 6; do
    conda run --no-capture-output -n xfemcustom python examples/turek_cylinder/turek_benchmark.py \\
      --benchmark 2d-3 --backend cpp --with-deformation --inflow dfg --init stokes \\
      --fe-order 2 --beta0 40 --ghost-measure patch --gamma-gp 1e-2 \\
      --dt 0.005 --theta 0.5 --max-steps 1600 --vtk-every 50 --level ${lv}
  done

Outputs / VTK
-------------
- Output directory:
  - without `--level`: `examples/turek_cylinder/turek_results/`
  - with `--level k`:  `examples/turek_cylinder/turek_results_lv{k}/`
- `functionals.csv` is written at the end of the run (Cd/Cl/dp evaluated at `t_{n+theta}`).
- Enable VTU output via `--vtk-every N`:
  - `--vtk-every 0`: disable VTU output
  - `--vtk-every 1`: write every step
  - `--vtk-every 20`: write every 20 steps (recommended for long runs)
  VTU files are written as `solution_####.vtu` inside the output folder.

Comparing to bundled FeatFlow reference
---------------------------------------
After a run, compare against the reference data (same `--level`):
  conda run --no-capture-output -n xfemcustom python examples/turek_cylinder/compare_featflow.py \\
    --level 3 --sim examples/turek_cylinder/turek_results_lv3/functionals.csv
"""

# # Test case 2D-2 (unsteady)

# In[1]:


import numpy as np
import time
import scipy.sparse.linalg as sp_la
# Matplotlib is imported lazily (only when plots are requested)
import numba
import os
import argparse
import json
import sys

# --- Numba configuration ---
try:
    num_cores = os.cpu_count()
    numba.set_num_threads(num_cores)
    print(f"Numba is set to use {numba.get_num_threads()} threads.")
except (ImportError, AttributeError):
    print("Numba not found or configured. Running in pure Python mode.")

# --- Core pycutfem imports ---
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.levelset import CircleLevelSet, LevelSetMeshAdaptation
from pycutfem.utils.domain_manager import get_domain_bitset

# --- UFL-like imports ---
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.expressions import (
    TrialFunction, TestFunction, VectorTrialFunction, VectorTestFunction,
    Function, VectorFunction, Constant, grad, inner, dot, div, jump, avg, FacetNormal, CellDiameter
)
from pycutfem.ufl.measures import dx, dS, dGhost, dFacetPatch, dInterface
from pycutfem.ufl.forms import BoundaryCondition, Equation
from pycutfem.solvers.nonlinear_solver import NewtonSolver, NewtonParameters, TimeStepperParameters
from pycutfem.ufl.compilers import FormCompiler
from pathlib import Path


# In[2]:


# ============================================================================
#    1. BENCHMARK PROBLEM SETUP
# ============================================================================
print("--- Setting up the Turek benchmark (DFG) for flow around a cylinder ---")

parser = argparse.ArgumentParser(description="CutFEM Turek benchmark (DFG cylinder)")
parser.add_argument(
    "--benchmark",
    choices=("2d-1", "2d-2", "2d-3"),
    default="2d-2",
    help="DFG benchmark variant: 2d-1 (steady, Re=20), 2d-2 (unsteady constant inflow, Re=100), 2d-3 (unsteady ramp inflow, Re=100).",
)
parser.add_argument("--with-deformation", action="store_true", help="enable isoparametric deformation for the cut geometry")
parser.add_argument("--no-deformation", action="store_true", help="disable the deformation even if enabled elsewhere")
parser.add_argument(
    "--backend",
    choices=("python", "jit", "cpp"),
    default="cpp",
    help="form assembly backend ('cpp' uses compiled C++ kernels; 'jit' uses cached JIT kernels)",
)
parser.add_argument(
    "--fe-order",
    type=int,
    default=2,
    help="velocity FE polynomial order (Qk)",
)
parser.add_argument(
    "--beta0",
    type=float,
    default=40.0,
    help="Base Nitsche penalty parameter (scaled internally by p^2 and μ/h, ρ h/dt).",
)
parser.add_argument(
    "--p-order",
    type=int,
    default=None,
    help="pressure FE polynomial order (defaults to fe_order-1; set equal to fe_order for equal-order test)",
)
parser.add_argument(
    "--zero-inside-vel",
    action="store_true",
    help="clamp velocity DOFs whose coordinates lie inside the cylinder to 0 (experimental)",
)
parser.add_argument("--plot", action="store_true", help="enable diagnostic plots (slower startup)")
parser.add_argument("--clear-jit-cache", action="store_true", help="clear cached JIT kernels before running")
parser.add_argument("--ghost-measure", choices=("edge", "patch"), default="patch",
                     help="ghost stabilization integration: 'edge' uses dGhost, 'patch' uses dFacetPatch")
parser.add_argument("--gamma-gp", type=float, default=1.0e-2,
                    help="ghost penalty coefficient (matches NGSolve reference default 1e-2)")
parser.add_argument(
    "--gamma-gp-p",
    type=float,
    default=None,
    help="pressure ghost penalty coefficient (defaults to --gamma-gp)",
)
parser.add_argument(
    "--gamma-gp-hess",
    type=float,
    default=0.0,
    help=(
        "velocity Hessian ghost-penalty coefficient (j=2 term). "
        "For robustness this term is integrated on facets (dGhost) even when "
        "`--ghost-measure=patch`; 0 disables the term."
    ),
)
parser.add_argument("--force-eval", choices=("surface", "babuska", "both"), default="both",
                    help="drag/lift evaluation method: surface (interface), babuska (volume), or both")
parser.add_argument("--dt", type=float, default=0.1, help="time step size")
parser.add_argument("--max-steps", type=int, default=200, help="maximum time steps")
parser.add_argument("--theta", type=float, default=0.560, help="theta scheme parameter (0.5 = CN)")
parser.add_argument("--disable-mass", action="store_true", help="disable transient mass term (rho/dt (u^{n+1}-u^n, v))")
parser.add_argument("--disable-convection", action="store_true", help="disable convection term (rho (u·∇u, v))")
parser.add_argument(
    "--convection-form",
    choices=("standard", "skew"),
    default="standard",
    help="convection discretization: standard (u·∇u) or skew-symmetric (energy-conserving)",
)
parser.add_argument("--nx", type=int, default=65, help="background mesh elements in x")
parser.add_argument("--ny", type=int, default=50, help="background mesh elements in y")
parser.add_argument("--refine-level", type=int, default=1, help="local refinement levels around the level set")
parser.add_argument(
    "--level",
    type=int,
    default=None,
    choices=range(1, 7),
    help="Mesh level preset (1..6). Overrides --nx/--ny/--refine-level with a reproducible refinement ladder.",
)
parser.add_argument("--inflow", choices=("constant", "dfg"), default="constant",
                    help="inflow profile: 'dfg' uses the FeatFlow ramp 0.2 + 0.8*sin(pi t/8)")
parser.add_argument(
    "--init",
    choices=("zero", "stokes"),
    default="zero",
    help="initial condition: 'zero' uses u=0,p=0 at t=0; 'stokes' computes a steady Stokes solve before time stepping",
)
parser.add_argument("--newton-tol", type=float, default=1e-5, help="Newton residual infinity-norm tolerance")
parser.add_argument("--max-newton-iter", type=int, default=40, help="maximum Newton iterations per step")
parser.add_argument("--ls-mode", choices=("armijo", "dealii"), default="dealii",
                    help="line search mode (dealii is cheaper/less strict than armijo)")
parser.add_argument("--vtk-every", type=int, default=1,
                    help="write VTU output every N steps (0 disables VTU output)")
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help=(
        "Output directory for CSV/VTU. If relative, it is resolved relative to "
        "`examples/turek_cylinder/`. Defaults to `turek_results[_lv{level}]`."
    ),
)
parser.add_argument(
    "--diag-interface",
    action="store_true",
    help="compute extra interface diagnostics (slip norms + Nitsche penalty force components)",
)
parser.add_argument(
    "--diag-kernels",
    action="store_true",
    help="print JIT kernel inventory (domains + entity counts) for debugging assembly",
)

args, _ = parser.parse_known_args()
if args.with_deformation and args.no_deformation:
    raise SystemExit("Choose at most one of --with-deformation or --no-deformation")

def _argv_has(flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:])

with_deformation = args.with_deformation and not args.no_deformation
print(f"with_deformation = {with_deformation}")
print(f"backend = {args.backend}")
benchmark = str(args.benchmark)
print(f"benchmark = {benchmark}")

def _maybe_apply_tuned_params(args) -> None:
    """
    If `examples/turek_cylinder/tuned_params.json` exists, load a matching entry
    and override stabilization defaults *only* when the user did not pass the
    corresponding CLI flag.
    """
    tuned_path = Path(__file__).resolve().parent / "tuned_params.json"
    if not tuned_path.is_file():
        return
    try:
        data = json.loads(tuned_path.read_text())
    except Exception as exc:
        print(f"[tuned params] failed to read {tuned_path}: {exc}")
        return
    entries = data.get("entries", []) if isinstance(data, dict) else []
    if not isinstance(entries, list) or not entries:
        return

    fe_order_eff = int(args.fe_order)
    p_order_eff = int(args.p_order) if args.p_order is not None else fe_order_eff - 1
    level_eff = int(args.level) if args.level is not None else None
    dt_eff = float(args.dt)

    match = None
    for e in entries:
        if not isinstance(e, dict):
            continue
        if str(e.get("benchmark", "")) != str(args.benchmark):
            continue
        if e.get("level", None) != level_eff:
            continue
        if str(e.get("ghost_measure", e.get("ghost-measure", ""))) != str(args.ghost_measure):
            continue
        if bool(e.get("with_deformation", False)) != bool(with_deformation):
            continue
        if int(e.get("fe_order", fe_order_eff)) != fe_order_eff:
            continue
        if int(e.get("p_order", p_order_eff)) != p_order_eff:
            continue
        # dt matching is intentionally tolerant: allow slight float formatting differences.
        try:
            dt_entry = float(e.get("dt", dt_eff))
        except Exception:
            dt_entry = dt_eff
        if abs(dt_entry - dt_eff) > 1e-12:
            continue
        match = e
        break

    if match is None:
        return

    overrides = {}
    if not _argv_has("--beta0") and "beta0" in match:
        overrides["beta0"] = float(match["beta0"])
        args.beta0 = float(match["beta0"])
    if not _argv_has("--gamma-gp") and "gamma_gp" in match:
        overrides["gamma-gp"] = float(match["gamma_gp"])
        args.gamma_gp = float(match["gamma_gp"])
    if (not _argv_has("--gamma-gp-p")) and ("gamma_gp_p" in match) and (match.get("gamma_gp_p") is not None):
        overrides["gamma-gp-p"] = float(match["gamma_gp_p"])
        args.gamma_gp_p = float(match["gamma_gp_p"])
    if not _argv_has("--gamma-gp-hess") and "gamma_gp_hess" in match:
        overrides["gamma-gp-hess"] = float(match["gamma_gp_hess"])
        args.gamma_gp_hess = float(match["gamma_gp_hess"])

    if overrides:
        print(f"[tuned params] applied from {tuned_path}: {overrides}")

_maybe_apply_tuned_params(args)

# Optional plotting setup (imports deferred to save startup time when disabled)
ENABLE_PLOTS = bool(args.plot)
if ENABLE_PLOTS:
    import matplotlib.pyplot as plt

backend = args.backend
ghost_measure = args.ghost_measure
force_eval = args.force_eval
dt_val = float(args.dt)
max_time_steps = int(args.max_steps)
theta_val = float(args.theta)
NX = int(args.nx)
NY = int(args.ny)
max_refine_level = int(args.refine_level)
level = int(args.level) if args.level is not None else None
if level is not None:
    # A lightweight preset to make "level sweeps" convenient and to avoid
    # overwriting output folders. This is *not* a guarantee of exact FeatFlow
    # mesh equivalence; it is a reproducible refinement ladder for comparisons.
    LEVEL_PRESETS: dict[int, tuple[int, int, int]] = {
        1: (33, 25, 0),
        2: (49, 38, 0),
        3: (65, 50, 1),
        4: (97, 75, 1),
        5: (129, 100, 2),
        6: (193, 150, 2),
    }
    NX, NY, max_refine_level = LEVEL_PRESETS[level]
    print(f"[mesh preset] --level {level} -> nx={NX}, ny={NY}, refine-level={max_refine_level}")
inflow_mode = args.inflow
init_mode = str(args.init)
newton_tol = float(args.newton_tol)
max_newton_iter = int(args.max_newton_iter)
ls_mode = str(args.ls_mode)
vtk_every = int(args.vtk_every)
diag_interface = bool(args.diag_interface)
diag_kernels = bool(args.diag_kernels)
gamma_gp_p_val = float(args.gamma_gp_p) if args.gamma_gp_p is not None else float(args.gamma_gp)
fe_order = int(args.fe_order)
if fe_order < 1:
    raise SystemExit("--fe-order must be >= 1")
p_order = int(args.p_order) if args.p_order is not None else fe_order - 1
if p_order < 0:
    raise SystemExit("--p-order must be >= 0")
zero_inside_vel = bool(args.zero_inside_vel)
beta0_cli = float(args.beta0)
disable_mass = bool(args.disable_mass)
disable_convection = bool(args.disable_convection)
convection_form = str(args.convection_form)

if benchmark == "2d-1":
    # DFG 2D-1 is a stationary (steady) Navier–Stokes solve at Re=20.
    # Apply steady defaults unless the user explicitly overrides them.
    # Also, disable ghost-penalty stabilization by default for accuracy on 2D-1
    # (it acts like extra dissipation and biases Cd/dp unless very small).
    if not _argv_has("--gamma-gp"):
        args.gamma_gp = 0.0
        if args.gamma_gp_p is None and not _argv_has("--gamma-gp-p"):
            gamma_gp_p_val = 0.0
    if not _argv_has("--disable-mass"):
        disable_mass = True
    if not _argv_has("--theta"):
        theta_val = 1.0
    if not _argv_has("--max-steps"):
        max_time_steps = 1
    if not _argv_has("--dt"):
        dt_val = 0.5
    if not _argv_has("--init"):
        init_mode = "stokes"
    if not _argv_has("--inflow"):
        inflow_mode = "constant"

if args.clear_jit_cache:
    import shutil
    cache_root = os.path.expanduser("~/.cache/pycutfem_jit")
    if os.path.isdir(cache_root):
        print("Clearing cached JIT kernels …")
        shutil.rmtree(cache_root)

# --- Geometry and Fluid Properties ---
H = 0.41  # Channel height
L = 2.2   # Channel length
D = 0.1   # Cylinder diameter
c_x, c_y = 0.2, 0.2  # Cylinder center
rho = 1.0  # Density
mu = 1e-3  # Viscosity
# DFG/FeatFlow conventions:
# - 2d-1 (Re=20):  U_mean=0.2, U_max=0.3, nu=1e-3  -> Re = U_mean*D/nu = 20
# - 2d-2/2d-3 (Re=100): U_mean=1.0, U_max=1.5, nu=1e-3 -> Re = 100
if benchmark == "2d-1":
    U_mean = 0.2
    U_max = 0.3
else:
    U_mean = 1.0
    U_max = 1.5
Re = rho * U_mean * D / mu
geom_order = 2 if with_deformation else 1
print(f"Reynolds number (Re): {Re:.2f}")
print(f"Geometry order={geom_order}, FE order={fe_order}, p_order={p_order}")


# In[3]:


# from pycutfem.utils.adaptive_mesh import structured_quad_levelset_adaptive
from pycutfem.utils.adaptive_mesh_ls_numba import structured_quad_levelset_adaptive
# --- Mesh ---
# Background mesh resolution
dt = Constant(dt_val)
theta = Constant(theta_val) # Crank-Nicolson
analytic_level_set = CircleLevelSet(center=(c_x, c_y), radius=D/2.0 )
# IMPORTANT (DFG cylinder / do-nothing outlet):
# We do NOT impose a pressure pin or a mean-pressure constraint here.
# The natural "do-nothing" outlet condition fixes the pressure constant; adding
# an extra constraint shifts the traction and corrupts Cd/Cl.
# (Using booleans is cleaner, but 1 or 0 works too)
with_interface_theta_traction = False
with_interface_theta_pressure = False
with_interface_theta_penalty = False
# h  = 0.5*(L/NX + H/NY)


# nodes, elems, _, corners = structured_quad(L, H, nx=NX, ny=NY, poly_order=poly_order)

nodes, elems, edges, corners = structured_quad_levelset_adaptive(
        Lx=L, Ly=H, nx=NX, ny=NY, poly_order=geom_order,
        level_set=CircleLevelSet(center=(c_x, c_y), radius=(D/2.0+0.2*D/2.0) ),
        max_refine_level=max_refine_level)          # add a single halo, nothing else
mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=geom_order)

deformation = None
level_set = analytic_level_set
volume_quadrature = 2 * fe_order + (4 if with_deformation else 2)

if with_deformation:
    adapter = LevelSetMeshAdaptation(mesh, order=max(2, geom_order), threshold=1.0, max_steps=6)
    deformation = adapter.calc_deformation(analytic_level_set, q_vol=volume_quadrature)
    level_set = adapter.lset_p1

# ============================================================================
#    2. BOUNDARY CONDITIONS
# ============================================================================

# --- Tag Boundaries ---

bc_tags = {
    'inlet':  lambda x, y: np.isclose(x, 0),
    'outlet': lambda x, y: np.isclose(x, L),
    'walls':  lambda x, y: np.isclose(y, 0) | np.isclose(y, H),
}



# --- Define Parabolic Inflow Profile ---
def parabolic_inflow(x, y, t=0.0):
    amp = 1.0
    if inflow_mode == "dfg":
        # FeatFlow reference ramp:
        #   amp(t) = 0.2 + 0.8 sin(pi t / 8)
        # so that amp(0)=0.2, amp(4)=1.0, amp(8)=0.2.
        amp = 0.2 + 0.8 * np.sin(np.pi * float(t) / 8.0)
    # Use U_max in the standard DFG profile u(y) = 4 U_max y (H-y) / H^2.
    return 4.0 * (U_max * amp) * y * (H - y) / (H**2)

# --- Define Boundary Conditions List ---
bcs = [
    BoundaryCondition('ux', 'dirichlet', 'inlet', parabolic_inflow),
    BoundaryCondition('uy', 'dirichlet', 'inlet', lambda x, y: 0.0),
    BoundaryCondition('ux', 'dirichlet', 'walls', lambda x, y: 0.0),
    BoundaryCondition('uy', 'dirichlet', 'walls', lambda x, y: 0.0),
    # No-slip on the cylinder is handled by the CutFEM formulation
    # "Do-nothing" at the outlet is the natural BC
]

# Homogeneous BCs for Jacobian assembly
bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]



# In[4]:


# --- Level Set for the Cylinder Obstacle ---
mesh.classify_elements(level_set)
mesh.classify_edges(level_set)
mesh.build_interface_segments(level_set=level_set)
mesh.tag_boundary_edges(bc_tags)

# --- Define Domains with BitSets ---
fluid_domain = get_domain_bitset(mesh, "element", "outside")
rigid_domain = get_domain_bitset(mesh, "element", "inside")
cut_domain = get_domain_bitset(mesh, "element", "cut")
ghost_edges = mesh.edge_bitset('ghost')
physical_domain = fluid_domain | cut_domain

# --- Finite Element Space and DofHandler ---
# Taylor-Hood elements (Q2 for velocity, Q1 for pressure)
mixed_element = MixedElement(
    mesh,
    field_specs={"ux": fe_order, "uy": fe_order, "p": p_order},
)
dof_handler = DofHandler(mixed_element, method='cg')
# dof_handler.info()
print(f"Number of inside elements: {rigid_domain.cardinality()}")
print(f"Number of outside elements: {fluid_domain.cardinality()}")
print(f"Number of cut elements: {cut_domain.cardinality()}")
print(f"Number of interface edges: {mesh.edge_bitset('interface').cardinality()}")
print(f"Number of ghost edges: {mesh.edge_bitset('ghost').cardinality()}")
print(f"Number of cut elements: {cut_domain.cardinality()}")
print(f"Number of pos ghost edges: {mesh.edge_bitset('ghost_pos').cardinality()}")
print(f"Number of neg ghost edges: {mesh.edge_bitset('ghost_neg').cardinality()}")
print(f"Number of ghost edges (both): {mesh.edge_bitset('ghost_both').cardinality()}")


# In[5]:


# # 1. Define the target point.
# # target_point = np.array([1.5, 0.99 * H])
# target_point = np.array([c_x, c_y])

# # 2. Get all node IDs that have a pressure DOF associated with them.
# p_dofs = dof_handler.get_field_slice('p')
# p_node_ids = np.array([dof_handler._dof_to_node_map[dof][1] for dof in p_dofs])

# # 3. Get the coordinates of ONLY these pressure-carrying nodes.
# p_node_coords = mesh.nodes_x_y_pos[p_node_ids]

# # 4. Find the node closest to the target point WITHIN this restricted set.
# distances = np.linalg.norm(p_node_coords - target_point, axis=1)
# local_index = np.argmin(distances)

# # 5. Get the global ID and actual coordinates of that specific pressure node.
# closest_p_node_id = p_node_ids[local_index]
# actual_pin_coords = mesh.nodes_x_y_pos[closest_p_node_id]
# print(f"Pinning pressure at the node closest to {target_point}, found at {actual_pin_coords}")


# In[6]:


# dof_handler.tag_dof_by_locator(
#     'p_pin', 'p',
#     locator=lambda x, y: np.isclose(x, actual_pin_coords[0]) and np.isclose(y, actual_pin_coords[1]),
#     find_first=True
# )
# bcs.append(BoundaryCondition('p', 'dirichlet', 'p_pin', lambda x, y: 0.0))
# bcs_homog.append(BoundaryCondition('p', 'dirichlet', 'p_pin', lambda x, y: 0.0))
# Tag velocity DOFs inside the cylinder (same tag name for both fields is OK)
dof_handler.tag_dofs_from_element_bitset("inactive", "ux", "inside", strict=True)
dof_handler.tag_dofs_from_element_bitset("inactive", "uy", "inside", strict=True)
dof_handler.tag_dofs_from_element_bitset("inactive", "p", "inside", strict=True)

if zero_inside_vel:
    inside_vel_dofs: set[int] = set()
    tol_inside = 1.0e-12
    for fld in ("ux", "uy"):
        coords = np.asarray(dof_handler.get_dof_coords(fld), dtype=float)
        dofs = np.asarray(dof_handler.get_field_slice(fld), dtype=int)
        phi = np.asarray(analytic_level_set(coords), dtype=float)
        mask = phi < -tol_inside
        inside_vel_dofs.update(int(gd) for gd in dofs[mask])
    dof_handler.dof_tags.setdefault("inside_vel", set()).update(inside_vel_dofs)
    print(
        f"[exp] zero_inside_vel: clamping {len(inside_vel_dofs)} velocity DOFs with phi<0."
    )
    bc_inside_ux = BoundaryCondition("ux", "dirichlet", "inside_vel", lambda x, y: 0.0)
    bc_inside_uy = BoundaryCondition("uy", "dirichlet", "inside_vel", lambda x, y: 0.0)
    bcs.extend([bc_inside_ux, bc_inside_uy])
    bcs_homog.extend([bc_inside_ux, bc_inside_uy])


# In[7]:


for name, bitset in mesh._edge_bitsets.items():
    print(f"Edge bitset '{name}': {bitset.cardinality()}")


# In[ ]:





# In[8]:


if ENABLE_PLOTS:
    from pycutfem.io.visualization import plot_mesh_2
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_mesh_2(
        mesh,
        ax=ax,
        level_set=level_set,
        show=True,
        plot_nodes=False,
        elem_tags=False,
        edge_colors=True,
        plot_interface=False,
        resolution=200,
    )


# In[9]:


# ============================================================================
#    3. UFL FORMULATION WITH GHOST PENALTY
# ============================================================================
print("\n--- Defining the UFL weak form for Navier-Stokes with ghost penalty ---")

# --- Function Spaces and Functions ---
velocity_space = FunctionSpace(name="velocity", field_names=['ux', 'uy'],dim=1, side='+')
pressure_space = FunctionSpace(name="pressure", field_names=['p'], dim=0, side = '+')

# Trial and Test functions
du = VectorTrialFunction(space=velocity_space, dof_handler=dof_handler, side = '+')
dp = TrialFunction(name='trial_pressure', field_name='p', dof_handler=dof_handler, side = '+')
v = VectorTestFunction(space=velocity_space, dof_handler=dof_handler, side = '+')
q = TestFunction(name='test_pressure', field_name='p', dof_handler=dof_handler, side = '+')

# Solution functions at current (k) and previous (n) time steps
u_k = VectorFunction(name="u_k", field_names=['ux', 'uy'], dof_handler=dof_handler, side = '+')
p_k = Function(name="p_k", field_name='p', dof_handler=dof_handler, side = '+')
u_n = VectorFunction(name="u_n", field_names=['ux', 'uy'], dof_handler=dof_handler, side = '+')
p_n = Function(name="p_n", field_name='p', dof_handler=dof_handler, side = '+')

# Postprocessing helper fields (not part of the solve):
# - u_theta/p_theta store the θ-time state u_{n+θ}, p_{n+θ}
# - u_dt stores (u_{n+1} - u_n)/dt, which approximates ∂_t u at t_{n+θ} for θ-schemes.
u_theta = VectorFunction(name="u_theta", field_names=["ux", "uy"], dof_handler=dof_handler, side="+")
p_theta = Function(name="p_theta", field_name="p", dof_handler=dof_handler, side="+")
u_dt = VectorFunction(name="u_dt", field_names=["ux", "uy"], dof_handler=dof_handler, side="+")

# --- Parameters ---

mu_const = Constant(mu)
rho_const = Constant(rho)

u_k.nodal_values.fill(0.0)
p_k.nodal_values.fill(0.0)
u_n.nodal_values.fill(0.0)
p_n.nodal_values.fill(0.0)
u_theta.nodal_values[:] = u_k.nodal_values[:]
p_theta.nodal_values[:] = p_k.nodal_values[:]
u_dt.nodal_values.fill(0.0)


# In[10]:


if ENABLE_PLOTS:
    u_n.plot()


# In[11]:


print(len(dof_handler.get_dirichlet_data(bcs)))


# In[12]:




# In[13]:


from pycutfem.ufl.expressions import Derivative, FacetNormal, restrict, Hessian, Pos
from pycutfem.core.geometry import hansbo_cut_ratio
from pycutfem.ufl.expressions import ElementWiseConstant
from pycutfem.ufl.expressions import Pos, Neg, Grad, Dot, FacetNormal, Constant


# NOTE: On level-set (dInterface) integrals, `FacetNormal()` is oriented with the
# level-set gradient: n = ∇φ/‖∇φ‖, i.e. from Ω⁻ (φ<0, cylinder) to Ω⁺ (φ>0, fluid).
# The fluid outward normal on the cylinder boundary is therefore n_fluid = -n.
n = FacetNormal()
n_fluid = -n


def grad_inner_jump(u, v):
    """⟨∂ₙu, ∂ₙv⟩  (scalar or 2‑D vector)."""
    a = dot(jump(grad(u)), n)
    b = dot(jump(grad(v)), n)
    return inner(a, b)


def hessian_inner(u, v):
    return inner(Hessian(u), Hessian(v)) 


def hdotn(expr):
    """Convenience: (Hessian(expr)) · n  (vector in R^2)."""
    n = FacetNormal()
    return dot(Hessian(expr), n)
def nHn(expr,n):
    """Convenience:  n · (Hessian(expr)) · n   (vector)."""
    return dot(dot(Hessian(expr), n),n)


def hess_inner_jump(u, v):
    """
    Ghost penalty building block: ⟨ [∂²ₙ u], [∂²ₙ v] ⟩ on a facet.

    Uses the directional second derivative along the facet normal:
      ∂²ₙ u = nᵀ (H u) n.
    """
    # IMPORTANT: contract the Hessian with the normal on the *derivative indices*,
    # i.e. use right-right contractions (H · n) · n, not n · (H · n).
    a = dot(dot(jump(Hessian(u)), n), n)
    b = dot(dot(jump(Hessian(v)), n), n)
    return inner(a, b)



ghost_edges_used = mesh.edge_bitset('ghost_pos') | mesh.edge_bitset('interface') #| mesh.edge_bitset('ghost_both')

# ---------------------------------------------------------------------------
# DFG do-nothing outlet (Γ_out)
# ---------------------------------------------------------------------------
# Benchmark definition (with outward normal η):
#   ν ∂_η u - p η = 0
#
# Our viscous volume form uses the symmetric interior bilinear form 2ν ε(u):ε(v),
# whose natural traction corresponds to (ν(∇u+∇uᵀ) - pI)·η. To recover the DFG
# outflow based on ν∇u - pI while keeping the symmetric interior form, we add the
# transpose correction on Γ_out:
#   + ⟨ ν (∇uᵀ·η), v ⟩_{Γ_out}.
outlet_edges = mesh.edge_bitset("outlet")
if outlet_edges.cardinality() == 0:
    raise RuntimeError("Outlet boundary tag not found on the background mesh.")
outlet_quadrature = max(8, int(volume_quadrature))
dS_outlet = dS(defined_on=outlet_edges, metadata={"q": outlet_quadrature}, deformation=deformation)
n_out = FacetNormal()
dx_phys  = dx(
    defined_on=physical_domain,
    level_set=level_set,
    metadata={"q": volume_quadrature, "side": "+"},
    deformation=deformation,
)
dΓ        = dInterface(
    defined_on=mesh.element_bitset('cut'),
    level_set=level_set,
    metadata={"q": volume_quadrature, 'derivs': {(0, 0), (0, 1), (1, 0)}},
    deformation=deformation,
)
# Ghost stabilization facets:
# Ghost penalty region:
# Use the full cut-neighborhood (including cut↔inside facets). This is the standard
# CutFEM choice: the ghost penalty extends stability/coercivity from the physical
# domain Ω⁺ into a narrow band of "ghost" cells so that small-cut configurations
# do not destroy conditioning or the pressure inf–sup stability.
ghost_edges_vel = mesh.edge_bitset("ghost")
ghost_edges_pres = mesh.edge_bitset("ghost")
# Request only the derivative orders actually needed by the active stabilization:
# - value/grad jumps: (0,0),(1,0),(0,1)
# - Hessian jump penalty: additionally (2,0),(1,1),(0,2)
ghost_derivs = {(0, 0), (1, 0), (0, 1)}
if float(args.gamma_gp_hess) != 0.0:
    ghost_derivs |= {(2, 0), (1, 1), (0, 2)}
if ghost_measure == "patch":
    dG_vel = dFacetPatch(
        defined_on=ghost_edges_vel,
        level_set=analytic_level_set,
        metadata={"q": volume_quadrature + 1, "derivs": ghost_derivs},
        deformation=deformation,
    )
    dG_pres = dFacetPatch(
        defined_on=ghost_edges_pres,
        level_set=analytic_level_set,
        metadata={"q": volume_quadrature + 1, "derivs": ghost_derivs},
        deformation=deformation,
    )
    # Hessian ghost-penalty is numerically more robust as a *facet* integral:
    # on a two-cell patch the neighbor-side evaluation happens via polynomial
    # extension (mapped points outside the reference cell), which can make
    # second-derivative jumps overly stiff and harm Newton robustness.
    dG_vel_hess = dG_vel
    if float(args.gamma_gp_hess) != 0.0:
        dG_vel_hess = dGhost(
            defined_on=ghost_edges_vel,
            level_set=analytic_level_set,
            metadata={"q": volume_quadrature + 1, "derivs": ghost_derivs},
            deformation=deformation,
        )
else:
    dG_vel = dGhost(
        defined_on=ghost_edges_vel,
        level_set=analytic_level_set,
        metadata={"q": volume_quadrature + 1, "derivs": ghost_derivs},
        deformation=deformation,
    )
    dG_pres = dGhost(
        defined_on=ghost_edges_pres,
        level_set=analytic_level_set,
        metadata={"q": volume_quadrature + 1, "derivs": ghost_derivs},
        deformation=deformation,
    )
    dG_vel_hess = dG_vel

u_trial_phys = restrict(du, physical_domain)
v_test_phys = restrict(v, physical_domain)
p_trial_phys = restrict(dp, physical_domain)
q_test_phys = restrict(q, physical_domain)
u_k_phys = restrict(u_k, physical_domain)
p_k_phys = restrict(p_k, physical_domain)
u_n_phys = restrict(u_n, physical_domain)
p_n_phys = restrict(p_n, physical_domain)

cell_h  = CellDiameter() # length‑scale per element
beta_N  = Constant(20.0 * fe_order**2)      # Nitsche penalty (tweak)
def scaled_penalty_interface(penalty, poly_order=fe_order,
                             side='+'):
    # 1) Hansbo factor — this is a *numpy array*, one value per element
    beta0_val  = penalty * poly_order**2
    theta_min  = 1.0e-3
    hansbo_plus = hansbo_cut_ratio(mesh, level_set, side=side)    # -> np.ndarray, shape (n_elem,)
    hansbo_plus = np.clip(hansbo_plus, theta_min, 1.0)
    alpha = 0.5
    beta_hansbo_arr = beta0_val * hansbo_plus**(-alpha)
    β_visc = ElementWiseConstant(beta_hansbo_arr) * (mu_const / cell_h)
    β_iner = Constant(0.0)
    if not disable_mass:
        β_iner = beta0_val * (rho_const * cell_h / dt)        # no θ-scaling here

    # 3) Final penalty (symbolic EWC × expression)
    return β_visc + β_iner
β = scaled_penalty_interface(beta0_cli, side='+')  # Nitsche penalty

def epsilon(u):
    "Symmetric gradient."
    return 0.5 * (grad(u) + grad(u).T)

def _advect_standard(w, u, v):
    return dot(dot(grad(u), w), v)

def _advect_skew(w, u, v):
    # Standard skew-symmetric convection form (energy-conserving for v=u):
    #   1/2 * ( (w·∇u, v) - (w·∇v, u) )
    # For the incompressible Navier–Stokes nonlinearity we use w=u.
    return 0.5 * (dot(dot(grad(u), w), v) - dot(dot(grad(v), w), u))



def sigma_dot_n_v(u_vec, p_scal, v_test, normal):
    """
    Expanded form of (σ(u, p) · n) without using the '@' operator.

        σ(u, p)·n = μ (∇u + ∇uᵀ)·n  −  p n
    """
    # first term: μ (∇u)·n
    a = dot(grad(u_vec), normal)
    # second term: μ (∇uᵀ)·n
    b = dot(grad(u_vec).T, normal)
    # combine and subtract pressure part
    return mu_const * dot((a + b), v_test) - p_scal * dot(v_test, normal)

# Split σ(u,p)n·v into viscous/pressure parts so we can (optionally) time-weight
# them differently during debugging.
def viscous_dot_n_v(u_vec, v_test, normal):
    a = dot(grad(u_vec), normal)
    b = dot(grad(u_vec).T, normal)
    return mu_const * dot((a + b), v_test)

def pressure_dot_n_v(p_scal, v_test, normal):
    # ( -p I normal ) · v = -(p normal)·v = -p (v·normal)
    return -p_scal * dot(v_test, normal)

# --- Jacobian contribution on Γsolid --------------------------------
    # - sigma_dot_n_v(Pos(u_trial_phys), Pos(p_trial_phys), Pos(v_test_phys), n_f)
    # - sigma_dot_n_v(Pos(v_test_phys), Pos(q_test_phys), Pos(u_trial_phys), n_f)



# --- 2. Define the time-stepping factors based on these options ---
# 'k' is the implicit factor (for step k, used in J_int and R_int)
# 'n' is the explicit factor (for step n, used in R_int only)

k_factor_T = theta if with_interface_theta_traction else Constant(1.0)
n_factor_T = (1.0 - theta) if with_interface_theta_traction else Constant(0.0)

k_factor_P = theta if with_interface_theta_pressure else Constant(1.0)
n_factor_P = (1.0 - theta) if with_interface_theta_pressure else Constant(0.0)

k_factor_B = theta if with_interface_theta_penalty else Constant(1.0)  # B for Beta (penalty)
n_factor_B = (1.0 - theta) if with_interface_theta_penalty else Constant(0.0)

# --- 3. Define the Jacobian (J_int) ---
# J_int is the derivative of R_int w.r.t. the k-th step solution.
#
# We impose u=0 on Γ (cylinder) via symmetric Nitsche with the *fluid outward*
# normal `n_fluid`:
#
#   -⟨σ(u,p) n_fluid, v⟩_Γ  -⟨σ(v,q) n_fluid, u⟩_Γ  +⟨β u, v⟩_Γ
#
# with σ(u,p)=μ(∇u+∇uᵀ)−pI (consistent with the volume term 2μ ε(u)).
# On Γ (interface), use raw fields with Pos-traces; `restrict(...)` can mask
# DOFs near the cut and has caused under-enforcement of no-slip in practice.
u_trial_if = Pos(du)
v_test_if = Pos(v)
p_trial_if = Pos(dp)
q_test_if = Pos(q)
u_k_if = Pos(u_k)
p_k_if = Pos(p_k)
u_n_if = Pos(u_n)
p_n_if = Pos(p_n)

J_int = (
    -k_factor_T * viscous_dot_n_v(u_trial_if, v_test_if, n_fluid)
    -k_factor_T * viscous_dot_n_v(v_test_if, u_trial_if, n_fluid)
    -k_factor_P * pressure_dot_n_v(p_trial_if, v_test_if, n_fluid)
    -k_factor_P * pressure_dot_n_v(q_test_if, u_trial_if, n_fluid)
    + k_factor_B * β * dot(u_trial_if, v_test_if)
) * dΓ

def is_zero(a):
    if isinstance(a, Constant):
        return a.value == 0.0
    else:
        return a == 0.0


#
R_int = (
    -k_factor_T * viscous_dot_n_v(u_k_if, v_test_if, n_fluid)
    -k_factor_T * viscous_dot_n_v(v_test_if, u_k_if, n_fluid)
    -k_factor_P * pressure_dot_n_v(p_k_if, v_test_if, n_fluid)
    -k_factor_P * pressure_dot_n_v(q_test_if, u_k_if, n_fluid)
    + k_factor_B * β * dot(u_k_if, v_test_if)
)
if not is_zero(n_factor_T):
    R_int +=  (
        -n_factor_T * viscous_dot_n_v(u_n_if, v_test_if, n_fluid)
        -n_factor_T * viscous_dot_n_v(v_test_if, u_n_if, n_fluid)
    )
if not is_zero(n_factor_P):
    R_int += (
        -n_factor_P * pressure_dot_n_v(p_n_if, v_test_if, n_fluid)
        -n_factor_P * pressure_dot_n_v(q_test_if, u_n_if, n_fluid)
    )
if not is_zero(n_factor_B):
    R_int +=  (
         n_factor_B * β * dot(u_n_if, v_test_if)
    )
R_int = R_int * dΓ

# volume ------------------------------------------------------------

a_vol = (
    (0.0 if disable_mass else 1.0) * rho_const * dot(u_trial_phys, v_test_phys) / dt
    + (0.0 if disable_convection else 1.0)
    * theta
    * rho_const
    * (
        (
            dot(dot(grad(u_k_phys), u_trial_phys), v_test_phys)
            + dot(dot(grad(u_trial_phys), u_k_phys), v_test_phys)
        )
        if convection_form == "standard"
        else 0.5
        * (
            (
                dot(dot(grad(u_k_phys), u_trial_phys), v_test_phys)
                + dot(dot(grad(u_trial_phys), u_k_phys), v_test_phys)
            )
            - (
                dot(dot(grad(v_test_phys), u_trial_phys), u_k_phys)
                + dot(dot(grad(v_test_phys), u_k_phys), u_trial_phys)
            )
        )
    )
    + 2 * theta * mu_const * inner(epsilon(u_trial_phys), epsilon(v_test_phys))
    - p_trial_phys * div(v_test_phys)
    + q_test_phys * div(u_trial_phys)
) * dx_phys

r_vol = (
    (0.0 if disable_mass else 1.0) * rho_const * dot(u_k_phys - u_n_phys, v_test_phys) / dt
    + (0.0 if disable_convection else 1.0)
    * rho_const
    * (
        theta
        * (
            dot(dot(grad(u_k_phys), u_k_phys), v_test_phys)
            if convection_form == "standard"
            else _advect_skew(u_k_phys, u_k_phys, v_test_phys)
        )
        + (1 - theta)
        * (
            dot(dot(grad(u_n_phys), u_n_phys), v_test_phys)
            if convection_form == "standard"
            else _advect_skew(u_n_phys, u_n_phys, v_test_phys)
        )
    )
    + 2 * theta * mu_const * inner(epsilon(u_k_phys), epsilon(v_test_phys))
    + 2 * (1 - theta) * mu_const * inner(epsilon(u_n_phys), epsilon(v_test_phys))
    - p_k_phys * div(v_test_phys)
    + q_test_phys * div(u_k_phys)
) * dx_phys

# Do-nothing outlet correction term (DFG): +ν (∇uᵀ·η, v)_Γout.
#
# NOTE: the outlet is a boundary of the *background* domain (far away from the
# level set). Using unrestricted test/trial functions here avoids pulling CutFEM
# restriction masks into the boundary kernel, while still assembling the correct
# contribution on Γ_out.
outflow_jac = theta * mu_const * dot(dot(grad(du).T, n_out), v) * dS_outlet
outflow_res = mu_const * (
    theta * dot(dot(grad(u_k).T, n_out), v)
    + (1 - theta) * dot(dot(grad(u_n).T, n_out), v)
) * dS_outlet
a_vol += outflow_jac
r_vol += outflow_res

# ghost stabilisation (add exactly as in your Poisson tests) --------
gamma_gp_u = Constant(float(args.gamma_gp))
gamma_gp_p = Constant(float(gamma_gp_p_val))
gamma_gp_hess_u = Constant(float(args.gamma_gp_hess))

# Ghost penalty scaling depends on the integration domain:
# - `dGhost`: facet integral (|F|~h)  → use facet-style scaling (μ/h, μ h, ...)
# - `dFacetPatch`: two-cell *volume* patch (|P|~h^2) → use patch-style scaling
#   (matches NGSolve SymbolicFacetPatchBFI with skeleton=False):
#       γ μ / h^2 ⟨[u],[v]⟩_P  +  (γ/μ) ⟨[p],[q]⟩_P
if ghost_measure == "patch":
    R_stab = (gamma_gp_u * mu_const / (cell_h * cell_h) * dot(jump(u_k_phys), jump(v_test_phys))) * dG_vel
    J_stab_lin = (gamma_gp_u * mu_const / (cell_h * cell_h) * dot(jump(u_trial_phys), jump(v_test_phys))) * dG_vel
    if float(args.gamma_gp_hess) != 0.0:
        # Integrate the j=2 ghost penalty on *facets* even in patch mode.
        gamma_v_hess = gamma_gp_hess_u * mu_const
        R_stab += (gamma_v_hess * (cell_h**3.0) / 4.0 * hess_inner_jump(u_k_phys, v_test_phys)) * dG_vel_hess
        J_stab_lin += (gamma_v_hess * (cell_h**3.0) / 4.0 * hess_inner_jump(u_trial_phys, v_test_phys)) * dG_vel_hess
    R_stab += (gamma_gp_p / mu_const) * (jump(p_k_phys) * jump(q_test_phys)) * dG_pres
    J_stab_lin += (gamma_gp_p / mu_const) * (jump(p_trial_phys) * jump(q_test_phys)) * dG_pres
else:
    gamma_v = gamma_gp_u * mu_const
    gamma_v_grad = gamma_gp_u * mu_const
    gamma_v_hess = gamma_gp_hess_u * mu_const
    # For facet-based pressure stabilization, scale ~ h/μ (value-jump) for Stokes-like regimes.
    gamma_p = Constant(float(gamma_gp_p_val))

    R_stab = (
        gamma_v / cell_h * dot(jump(u_k_phys), jump(v_test_phys))
        + gamma_v_grad * cell_h * grad_inner_jump(u_k_phys, v_test_phys)
    ) * dG_vel
    J_stab_lin = (
        gamma_v / cell_h * dot(jump(u_trial_phys), jump(v_test_phys))
        + gamma_v_grad * cell_h * grad_inner_jump(u_trial_phys, v_test_phys)
    ) * dG_vel
    if float(args.gamma_gp_hess) != 0.0:
        # Facet Hessian ghost penalty for Q2/Q3 (normal second derivative jump).
        R_stab += (gamma_v_hess * (cell_h**3.0) / 4.0 * hess_inner_jump(u_k_phys, v_test_phys)) * dG_vel
        J_stab_lin += (gamma_v_hess * (cell_h**3.0) / 4.0 * hess_inner_jump(u_trial_phys, v_test_phys)) * dG_vel
    R_stab += (gamma_p * cell_h / mu_const) * (jump(p_k_phys) * jump(q_test_phys)) * dG_pres
    J_stab_lin += (gamma_p * cell_h / mu_const) * (jump(p_trial_phys) * jump(q_test_phys)) * dG_pres
# complete Jacobian and residual -----------------------------------
jacobian_form  = a_vol + J_int + J_stab_lin
residual_form  = r_vol + R_int + R_stab
# residual_form  = dot(  Constant(np.array([0.0, 0.0]),dim=1), v) * dx
# jacobian_form  = stab_lin
# residual_form  = stab





# In[14]:


# from pycutfem.ufl.forms import assemble_form
# K,F=assemble_form(jacobian_form==-residual_form, dof_handler=dof_handler, bcs=bcs_homog)
# print(np.linalg.norm(F, ord=np.inf))


# In[15]:


# get_ipython().system('rm -rf ~/.cache/pycutfem_jit/*')


# In[16]:


from pycutfem.io.vtk import export_vtk
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.fem import transform

out_dir_name = f"turek_results_lv{level}" if level is not None else "turek_results"
if args.output_dir:
    output_dir = Path(os.path.expanduser(str(args.output_dir)))
    if not output_dir.is_absolute():
        output_dir = (Path(__file__).resolve().parent / output_dir).resolve()
else:
    output_dir = Path(__file__).resolve().parent / out_dir_name
output_dir.mkdir(parents=True, exist_ok=True)
histories = {}  # Store histories for CD, CL, Δp

# ---------------------------------------------------------------------------
# Probe points (FeatFlow reference)
# ---------------------------------------------------------------------------
# FeatFlow point-values use (x,y) = (0.15, 0.199999) and (0.25, 0.199999),
# i.e. a tiny offset from the exact cylinder surface.
PROBE_A = np.asarray([0.15, 0.199999], dtype=float)  # upstream
PROBE_B = np.asarray([0.25, 0.199999], dtype=float)  # downstream
STAGNATION_POINT = PROBE_A.copy()


def _locate_point(mesh: Mesh, xy: np.ndarray) -> tuple[int, float, float] | None:
    """Return (eid, xi, eta) for a physical point, or None if not found."""
    xy = np.asarray(xy, dtype=float).reshape(2,)
    x, y = float(xy[0]), float(xy[1])
    ref_geom = transform.get_reference(mesh.element_type, int(getattr(mesh, "poly_order", 1)))

    def _inverse_map_deformed(eid: int, xy_target: np.ndarray, xi0: tuple[float, float]) -> tuple[float, float] | None:
        conn = np.asarray(mesh.elements_connectivity[int(eid)], dtype=int)
        coords_geom = np.asarray(mesh.nodes_x_y_pos[conn], dtype=float)
        if deformation is not None:
            coords_def = coords_geom + np.asarray(deformation.node_displacements[conn], dtype=float)
        else:
            coords_def = coords_geom

        xi_eta = np.asarray([float(xi0[0]), float(xi0[1])], dtype=float)
        best = xi_eta.copy()
        best_r = float("inf")
        for _ in range(40):
            xi, eta = float(xi_eta[0]), float(xi_eta[1])
            N = np.asarray(ref_geom.shape(xi, eta), dtype=float).ravel()
            dN = np.asarray(ref_geom.grad(xi, eta), dtype=float)
            X = N @ coords_def
            rhs = np.asarray(xy_target, dtype=float).reshape(2,) - np.asarray(X, dtype=float).reshape(2,)
            rnorm = float(np.linalg.norm(rhs))
            if rnorm < best_r:
                best_r = rnorm
                best[:] = xi_eta
            if rnorm < 1.0e-12:
                break
            J = coords_def.T @ dN
            try:
                delta = np.linalg.solve(J, rhs)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(J, rhs, rcond=None)[0]
            delta = np.asarray(delta, dtype=float).reshape(2,)
            dnorm = float(np.linalg.norm(delta))
            if not np.isfinite(dnorm):
                break
            if dnorm > 1.0 and dnorm > 0.0:
                delta *= 1.0 / dnorm
            xi_eta_next = xi_eta + delta
            if not np.isfinite(xi_eta_next).all():
                break
            # Clamp to a small extension band; probes should lie in the deformed cell.
            xi_eta_next = np.clip(xi_eta_next, -1.5, 1.5)
            if float(np.linalg.norm(xi_eta_next - xi_eta)) < 1.0e-12:
                xi_eta = xi_eta_next
                break
            xi_eta = xi_eta_next
        xi, eta = float(best[0]), float(best[1])
        if -1.00001 <= xi <= 1.00001 and -1.00001 <= eta <= 1.00001:
            return xi, eta
        return None

    for e in mesh.elements_list:
        # Probe values are defined on the *fluid* side only (Ω⁺ = outside/cut).
        # Avoid picking an "inside" element whose bounding box may still contain
        # the point (common near the cylinder on refined meshes).
        if getattr(e, "tag", None) == "inside":
            continue
        # quick bbox filter (use all geometry nodes for curved/high-order elements)
        node_ids = getattr(e, "nodes", None) or getattr(e, "node_ids", None) or []
        v = mesh.nodes_x_y_pos[list(node_ids)]
        if deformation is not None:
            v = v + np.asarray(deformation.node_displacements[list(node_ids)], dtype=float)
        if not (v[:, 0].min() <= x <= v[:, 0].max() and v[:, 1].min() <= y <= v[:, 1].max()):
            continue
        try:
            xi_guess, eta_guess = transform.inverse_mapping(mesh, int(e.id), xy)
        except Exception:
            continue
        if deformation is not None:
            xi_eta = _inverse_map_deformed(int(e.id), xy, (float(xi_guess), float(eta_guess)))
            if xi_eta is None:
                continue
            xi, eta = xi_eta
        else:
            xi, eta = float(xi_guess), float(eta_guess)
        if -1.00001 <= float(xi) <= 1.00001 and -1.00001 <= float(eta) <= 1.00001:
            return int(e.id), float(xi), float(eta)
    return None


def _make_probe_ctx(field: str, xy: np.ndarray) -> dict | None:
    loc = _locate_point(mesh, xy)
    if loc is None:
        return None
    eid, xi, eta = loc
    me = dof_handler.mixed_element
    phi = np.asarray(me.basis(field, xi, eta)[me.slice(field)], dtype=float)
    gdofs = np.asarray(dof_handler.element_maps[field][eid], dtype=int)
    return {"eid": eid, "xi": xi, "eta": eta, "phi": phi, "gdofs": gdofs}


def _eval_probe_scalar(f_scalar: Function, ctx: dict | None) -> float:
    if ctx is None:
        return float("nan")
    vals = np.asarray(f_scalar.get_nodal_values(ctx["gdofs"]), dtype=float)
    return float(ctx["phi"] @ vals)


_PROBE_CTX = {
    "pA": _make_probe_ctx("p", PROBE_A),
    "pB": _make_probe_ctx("p", PROBE_B),
    "ux_stag": _make_probe_ctx("ux", STAGNATION_POINT),
    "uy_stag": _make_probe_ctx("uy", STAGNATION_POINT),
}
# --- Traction helper on Γ: (σ(u,p)·n)·v_dir  -------------------------------
# DFG benchmark convention uses the (non-symmetric) stress:
#   σ := ν ∇u - p I
def traction_dot_dir(u_vec, p_scal, v_dir, side="+"):  # side: "+" (pos) or "-" (neg)
    if side == "+":
        du = Pos(Grad(u_vec))  # use POS trace of ∇u
        p  = Pos(p_scal)       # use POS trace of p
    else:
        du = Neg(Grad(u_vec))
        p  = Neg(p_scal)

    a = Dot(du, n)        # (∇u|side)·n
    t = mu * a - p * n    # σ(u,p)·n
    return Dot(t, v_dir)



def _build_babuska_miller_weights() -> tuple[VectorFunction, VectorFunction]:
    """
    Build (phi_drag, phi_lift) for Babuska–Miller force evaluation by solving a
    harmonic extension ψ with:
      ψ = 1 on Γ (cylinder interface, via Nitsche)
      ψ = 0 on outer boundaries (Dirichlet)
    and setting:
      phi_drag = (ψ, 0), phi_lift = (0, ψ)
    """
    print("Computing Babuska–Miller weight function ψ ...")

    psi_trial = TrialFunction(name="psi_trial", field_name="ux", dof_handler=dof_handler, side="+")
    psi_test = TestFunction(name="psi_test", field_name="ux", dof_handler=dof_handler, side="+")
    psi_trial_phys = restrict(psi_trial, physical_domain)
    psi_test_phys = restrict(psi_test, physical_domain)

    g_one = Constant(1.0)
    beta_psi = Constant(50.0 * fe_order**2)

    a_psi = inner(grad(psi_trial_phys), grad(psi_test_phys)) * dx_phys
    a_psi += (
        -dot(grad(Pos(psi_trial_phys)), n_fluid) * Pos(psi_test_phys)
        -dot(grad(Pos(psi_test_phys)), n_fluid) * Pos(psi_trial_phys)
        + (beta_psi / cell_h) * Pos(psi_trial_phys) * Pos(psi_test_phys)
    ) * dΓ
    L_psi = (
        -dot(grad(Pos(psi_test_phys)), n_fluid) * g_one
        + (beta_psi / cell_h) * g_one * Pos(psi_test_phys)
    ) * dΓ

    psi_bcs = [
        BoundaryCondition("ux", "dirichlet", "inlet", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "walls", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "outlet", lambda x, y: 0.0),
    ]

    K, F = assemble_form(Equation(a_psi, L_psi), dof_handler=dof_handler, bcs=psi_bcs, backend=backend)
    K = K.tocsr()
    F = np.asarray(F, dtype=float)

    ux_slice = np.asarray(dof_handler.get_field_slice("ux"), dtype=int)
    inactive = set(getattr(dof_handler, "dof_tags", {}).get("inactive", set()))
    active_mask = ~np.isin(ux_slice, np.asarray(sorted(inactive), dtype=int))
    ux_active = ux_slice[active_mask]

    Kux = K[ux_active, :][:, ux_active]
    Fux = F[ux_active]
    psi_active = sp_la.spsolve(Kux, Fux)

    psi_vals = np.zeros(len(ux_slice), dtype=float)
    psi_vals[active_mask] = np.asarray(psi_active, dtype=float)

    uy_slice = np.asarray(dof_handler.get_field_slice("uy"), dtype=int)
    if uy_slice.size != ux_slice.size:
        raise RuntimeError("Expected identical velocity component DOF counts for phi construction.")

    phi_drag = VectorFunction(name="phi_drag", field_names=["ux", "uy"], dof_handler=dof_handler, side="+")
    phi_lift = VectorFunction(name="phi_lift", field_names=["ux", "uy"], dof_handler=dof_handler, side="+")
    phi_drag.set_component_values(0, psi_vals)
    phi_drag.set_component_values(1, np.zeros_like(psi_vals))
    phi_lift.set_component_values(0, np.zeros_like(psi_vals))
    phi_lift.set_component_values(1, psi_vals)

    return phi_drag, phi_lift


def _init_force_functionals() -> None:
    """Pre-build force integrals + hooks once (avoid JIT recompilation each step)."""
    global SURF_DRAG_INT, SURF_LIFT_INT, SURF_DRAG_HOOK, SURF_LIFT_HOOK
    global SURF_PINT_INT, SURF_PINT_HOOK
    global SURF_PEN_DRAG_INT, SURF_PEN_LIFT_INT, SURF_PEN_DRAG_HOOK, SURF_PEN_LIFT_HOOK
    global SURF_U2_INT, SURF_UN2_INT, SURF_UT2_INT, SURF_U2_HOOK, SURF_UN2_HOOK, SURF_UT2_HOOK
    global IFACE_LEN, IFACE_NX, IFACE_NY, DΓ_FORCE
    global BM_DRAG_INT, BM_LIFT_INT, BM_DRAG_HOOK, BM_LIFT_HOOK
    global phi_drag, phi_lift

    # FeatFlow logs functionals at the θ-time (t_{n+θ}). We keep separate
    # postprocessing fields (`u_theta`, `p_theta`, `u_dt`) that are updated
    # each time step from (u_k,p_k) and (u_n,p_n) before evaluating integrals.

    if force_eval in {"surface", "both"}:
        dΓ_force = dInterface(defined_on=cut_domain, level_set=level_set, metadata={"q": 11}, deformation=deformation)
        DΓ_FORCE = dΓ_force
        e_x = Constant(np.array([1.0, 0.0]), dim=1)
        e_y = Constant(np.array([0.0, 1.0]), dim=1)
        # Convention: report the force on the cylinder. With φ<0 inside the cylinder,
        # dInterface uses n = ∇φ/‖∇φ‖ which points from cylinder → fluid, i.e. the
        # outward normal of the cylinder boundary. Hence, no extra sign flip.
        # Physical traction on Γ (force on the cylinder): σ(u,p)·n with n pointing
        # cylinder→fluid (our dInterface convention for φ<0 inside).
        SURF_DRAG_INT = traction_dot_dir(u_theta, p_theta, e_x) * dΓ_force
        SURF_LIFT_INT = traction_dot_dir(u_theta, p_theta, e_y) * dΓ_force
        SURF_DRAG_HOOK = {SURF_DRAG_INT.integrand: {"name": "FD_surf"}}
        SURF_LIFT_HOOK = {SURF_LIFT_INT.integrand: {"name": "FL_surf"}}

        if diag_interface:
            # Nitsche penalty contribution to the boundary reaction:
            # t_num = σ(u,p)·n + β u  (force on the cylinder; n points cylinder→fluid).
            SURF_PEN_DRAG_INT = (β * dot(Pos(u_theta), e_x)) * dΓ_force
            SURF_PEN_LIFT_INT = (β * dot(Pos(u_theta), e_y)) * dΓ_force
            SURF_PEN_DRAG_HOOK = {SURF_PEN_DRAG_INT.integrand: {"name": "FD_pen"}}
            SURF_PEN_LIFT_HOOK = {SURF_PEN_LIFT_INT.integrand: {"name": "FL_pen"}}

            # Interface slip diagnostics: RMS(|u|), RMS(|u·n|), RMS(|u·t|).
            u_if = Pos(u_theta)
            u2 = dot(u_if, u_if)
            un = dot(u_if, n)
            un2 = un * un
            ut2 = u2 - un2
            SURF_U2_INT = u2 * dΓ_force
            SURF_UN2_INT = un2 * dΓ_force
            SURF_UT2_INT = ut2 * dΓ_force
            SURF_U2_HOOK = {SURF_U2_INT.integrand: {"name": "u2"}}
            SURF_UN2_HOOK = {SURF_UN2_INT.integrand: {"name": "un2"}}
            SURF_UT2_HOOK = {SURF_UT2_INT.integrand: {"name": "ut2"}}
        # Pressure integral on Γ (for postprocessed gauge correction).
        SURF_PINT_INT = Pos(p_theta) * dΓ_force
        SURF_PINT_HOOK = {SURF_PINT_INT.integrand: {"name": "P_int"}}

        # Geometry-only diagnostics: length(Γ) and ∫_Γ n ds (should be ~0 for a closed curve).
        one = Constant(1.0)
        LEN_INT = one * dΓ_force
        NX_INT = n[0] * dΓ_force
        NY_INT = n[1] * dΓ_force
        IFACE_LEN = float(
            np.asarray(
                assemble_form(
                    Equation(None, LEN_INT),
                    dof_handler=dof_handler,
                    bcs=[],
                    assembler_hooks={LEN_INT.integrand: {"name": "len"}},
                    backend=backend,
                )["len"],
                dtype=float,
            ).reshape(-1)[0]
        )
        IFACE_NX = float(
            np.asarray(
                assemble_form(
                    Equation(None, NX_INT),
                    dof_handler=dof_handler,
                    bcs=[],
                    assembler_hooks={NX_INT.integrand: {"name": "nx"}},
                    backend=backend,
                )["nx"],
                dtype=float,
            ).reshape(-1)[0]
        )
        IFACE_NY = float(
            np.asarray(
                assemble_form(
                    Equation(None, NY_INT),
                    dof_handler=dof_handler,
                    bcs=[],
                    assembler_hooks={NY_INT.integrand: {"name": "ny"}},
                    backend=backend,
                )["ny"],
                dtype=float,
            ).reshape(-1)[0]
        )
        print(f"[geom] |Γ|≈{IFACE_LEN:.6e}  ∫n ds≈({IFACE_NX:.3e}, {IFACE_NY:.3e})")

    if force_eval in {"babuska", "both"}:
        phi_drag, phi_lift = _build_babuska_miller_weights()
        phi_drag_phys = restrict(phi_drag, physical_domain)
        phi_lift_phys = restrict(phi_lift, physical_domain)
        u_theta_phys = restrict(u_theta, physical_domain)
        p_theta_phys = restrict(p_theta, physical_domain)
        u_dt_phys = restrict(u_dt, physical_domain)

        # The Babuška–Miller volume identity yields the boundary traction with the
        # *fluid outward* normal. We postprocess with a sign flip so that we report
        # the force on the cylinder (FeatFlow convention).
        BM_DRAG_INT = (
            mu_const * inner(grad(u_theta_phys), grad(phi_drag_phys))
            - p_theta_phys * div(phi_drag_phys)
            + rho_const
            * inner(u_dt_phys + dot(grad(u_theta_phys), u_theta_phys), phi_drag_phys)
        ) * dx_phys
        BM_LIFT_INT = (
            mu_const * inner(grad(u_theta_phys), grad(phi_lift_phys))
            - p_theta_phys * div(phi_lift_phys)
            + rho_const
            * inner(u_dt_phys + dot(grad(u_theta_phys), u_theta_phys), phi_lift_phys)
        ) * dx_phys
        BM_DRAG_HOOK = {BM_DRAG_INT.integrand: {"name": "FD_bm"}}
        BM_LIFT_HOOK = {BM_LIFT_INT.integrand: {"name": "FL_bm"}}


def save_solution(funcs, prev_funcs=None, *, step_idx: int):
    """Export + compute CD/CL via surface and/or Babuska–Miller volume evaluation."""
    u_k_func = funcs[0]
    p_k_func = funcs[1]

    # Update θ-time postprocessing fields (used by surface + Babuška–Miller integrals).
    theta_num = float(theta.value)
    if prev_funcs is None:
        u_n_func = u_k_func
        p_n_func = p_k_func
    else:
        u_n_func = prev_funcs[0]
        p_n_func = prev_funcs[1]

    dt_num = float(dt.value)
    u_theta.nodal_values[:] = theta_num * u_k_func.nodal_values + (1.0 - theta_num) * u_n_func.nodal_values
    p_theta.nodal_values[:] = theta_num * p_k_func.nodal_values + (1.0 - theta_num) * p_n_func.nodal_values
    if dt_num > 0.0:
        u_dt.nodal_values[:] = (u_k_func.nodal_values - u_n_func.nodal_values) / dt_num
    else:
        u_dt.nodal_values.fill(0.0)

    # -----------------------------------------------------------------------
    # Pressure gauge handling
    # -----------------------------------------------------------------------
    # Pressure gauge diagnostics:
    # With the do-nothing outlet, the pressure constant is already fixed by the
    # boundary condition (and shifting it would change the physical traction at
    # the outlet). We therefore do NOT modify the pressure field here; we only
    # report the mean pressure on Γ as a diagnostic.
    p_mean = 0.0
    if force_eval in {"surface", "both"} and "SURF_PINT_INT" in globals():
        if "IFACE_LEN" not in globals() or abs(float(globals()["IFACE_LEN"])) < 1e-30:
            raise RuntimeError("Interface length IFACE_LEN not initialized.")
        p_int = assemble_form(
            Equation(None, globals()["SURF_PINT_INT"]),
            dof_handler=dof_handler,
            bcs=[],
            assembler_hooks=globals()["SURF_PINT_HOOK"],
            backend=backend,
        )["P_int"]
        p_mean = float(np.asarray(p_int, dtype=float).reshape(-1)[0]) / float(globals()["IFACE_LEN"])

    # ------------------ VTK output (as you already have) --------------------
    if vtk_every > 0 and (step_idx % vtk_every == 0):
        filename = output_dir / f"solution_{step_idx:04d}.vtu"
        export_vtk(
            filename=str(filename),
            mesh=mesh,
            dof_handler=dof_handler,
            functions={"velocity": u_k_func, "pressure": p_k_func},
        )

    coeff = 2.0 / (rho * (U_mean**2) * D)

    # ------------------ Interface (surface) integrals -----------------------
    F_D_surf = None
    F_L_surf = None
    C_D_surf = None
    C_L_surf = None
    F_D_pen = None
    F_L_pen = None
    C_D_nitsche = None
    C_L_nitsche = None
    u_rms = None
    un_rms = None
    ut_rms = None
    if force_eval in {"surface", "both"}:
        if "SURF_DRAG_INT" not in globals() or "SURF_LIFT_INT" not in globals():
            raise RuntimeError("Surface force integrals not initialized (SURF_* missing).")
        res_Fd = assemble_form(
            Equation(None, globals()["SURF_DRAG_INT"]),
            dof_handler=dof_handler,
            bcs=[],
            assembler_hooks=globals()["SURF_DRAG_HOOK"],
            backend=backend,
        )
        res_Fl = assemble_form(
            Equation(None, globals()["SURF_LIFT_INT"]),
            dof_handler=dof_handler,
            bcs=[],
            assembler_hooks=globals()["SURF_LIFT_HOOK"],
            backend=backend,
        )

        F_D_surf = float(np.asarray(res_Fd["FD_surf"], dtype=float).reshape(-1)[0])
        F_L_surf = float(np.asarray(res_Fl["FL_surf"], dtype=float).reshape(-1)[0])
        # Report gauge-corrected forces (p_mean removed); this is already baked
        # into (u_k, p_k) above. Log the applied p_mean for diagnostics.
        C_D_surf = coeff * F_D_surf
        C_L_surf = coeff * F_L_surf

        if diag_interface:
            if "SURF_PEN_DRAG_INT" not in globals() or "SURF_PEN_LIFT_INT" not in globals():
                raise RuntimeError("diag_interface requested but penalty force integrals are missing.")
            res_pen_d = assemble_form(
                Equation(None, globals()["SURF_PEN_DRAG_INT"]),
                dof_handler=dof_handler,
                bcs=[],
                assembler_hooks=globals()["SURF_PEN_DRAG_HOOK"],
                backend=backend,
            )
            res_pen_l = assemble_form(
                Equation(None, globals()["SURF_PEN_LIFT_INT"]),
                dof_handler=dof_handler,
                bcs=[],
                assembler_hooks=globals()["SURF_PEN_LIFT_HOOK"],
                backend=backend,
            )
            F_D_pen = float(np.asarray(res_pen_d["FD_pen"], dtype=float).reshape(-1)[0])
            F_L_pen = float(np.asarray(res_pen_l["FL_pen"], dtype=float).reshape(-1)[0])
            C_D_nitsche = coeff * (F_D_surf + F_D_pen)
            C_L_nitsche = coeff * (F_L_surf + F_L_pen)

            if "SURF_U2_INT" not in globals() or "SURF_UN2_INT" not in globals() or "SURF_UT2_INT" not in globals():
                raise RuntimeError("diag_interface requested but slip integrals are missing.")
            if "IFACE_LEN" not in globals() or abs(float(globals()["IFACE_LEN"])) < 1e-30:
                raise RuntimeError("Interface length IFACE_LEN not initialized.")
            u2 = assemble_form(
                Equation(None, globals()["SURF_U2_INT"]),
                dof_handler=dof_handler,
                bcs=[],
                assembler_hooks=globals()["SURF_U2_HOOK"],
                backend=backend,
            )["u2"]
            un2 = assemble_form(
                Equation(None, globals()["SURF_UN2_INT"]),
                dof_handler=dof_handler,
                bcs=[],
                assembler_hooks=globals()["SURF_UN2_HOOK"],
                backend=backend,
            )["un2"]
            ut2 = assemble_form(
                Equation(None, globals()["SURF_UT2_INT"]),
                dof_handler=dof_handler,
                bcs=[],
                assembler_hooks=globals()["SURF_UT2_HOOK"],
                backend=backend,
            )["ut2"]
            iface_len = float(globals()["IFACE_LEN"])
            u2_val = float(np.asarray(u2, dtype=float).reshape(-1)[0])
            un2_val = float(np.asarray(un2, dtype=float).reshape(-1)[0])
            ut2_val = float(np.asarray(ut2, dtype=float).reshape(-1)[0])
            u_rms = float(np.sqrt(max(u2_val, 0.0) / iface_len))
            un_rms = float(np.sqrt(max(un2_val, 0.0) / iface_len))
            ut_rms = float(np.sqrt(max(ut2_val, 0.0) / iface_len))

    # ------------------ Babuska–Miller (volume) integrals -------------------
    F_D_bm = None
    F_L_bm = None
    C_D_bm = None
    C_L_bm = None
    if force_eval in {"babuska", "both"}:
        if "BM_DRAG_INT" not in globals() or "BM_LIFT_INT" not in globals():
            raise RuntimeError("Babuska–Miller weights not initialized (phi_drag/phi_lift missing).")
        res_bm_d = assemble_form(
            Equation(None, globals()["BM_DRAG_INT"]),
            dof_handler=dof_handler,
            bcs=[],
            assembler_hooks=globals()["BM_DRAG_HOOK"],
            backend=backend,
        )
        res_bm_l = assemble_form(
            Equation(None, globals()["BM_LIFT_INT"]),
            dof_handler=dof_handler,
            bcs=[],
            assembler_hooks=globals()["BM_LIFT_HOOK"],
            backend=backend,
        )
        # Convert "force on the fluid" (fluid outward normal) → "force on the cylinder".
        F_D_bm = -float(np.asarray(res_bm_d["FD_bm"], dtype=float).reshape(-1)[0])
        F_L_bm = -float(np.asarray(res_bm_l["FL_bm"], dtype=float).reshape(-1)[0])
        C_D_bm = coeff * F_D_bm
        C_L_bm = coeff * F_L_bm

    # ------------------ FeatFlow probes at t_{n+θ} --------------------------
    pA_k = _eval_probe_scalar(p_k_func, _PROBE_CTX["pA"])
    pB_k = _eval_probe_scalar(p_k_func, _PROBE_CTX["pB"])
    pA_n = _eval_probe_scalar(p_n_func, _PROBE_CTX["pA"])
    pB_n = _eval_probe_scalar(p_n_func, _PROBE_CTX["pB"])
    pA = theta_num * pA_k + (1.0 - theta_num) * pA_n
    pB = theta_num * pB_k + (1.0 - theta_num) * pB_n
    dp = pA - pB

    ux_k = _eval_probe_scalar(u_k_func.components[0], _PROBE_CTX["ux_stag"])
    uy_k = _eval_probe_scalar(u_k_func.components[1], _PROBE_CTX["uy_stag"])
    ux_n = _eval_probe_scalar(u_n_func.components[0], _PROBE_CTX["ux_stag"])
    uy_n = _eval_probe_scalar(u_n_func.components[1], _PROBE_CTX["uy_stag"])
    ux_stag = theta_num * ux_k + (1.0 - theta_num) * ux_n
    uy_stag = theta_num * uy_k + (1.0 - theta_num) * uy_n

    # ------------------ Log / store ----------------------------------------
    msg = f"[step {step_idx+1:4d}]  Δp={dp:.6f}  u_stag=({ux_stag:.4e},{uy_stag:.4e})"
    if F_D_surf is not None and F_L_surf is not None:
        msg += f"  (surf) FD={F_D_surf:.6e} FL={F_L_surf:.6e} CD={C_D_surf:.6f} CL={C_L_surf:.6f}"
        if diag_interface and F_D_pen is not None and F_L_pen is not None:
            msg += f"  +βu: FD={F_D_pen:.3e} FL={F_L_pen:.3e}  CD*={C_D_nitsche:.6f} CL*={C_L_nitsche:.6f}"
        if diag_interface and u_rms is not None:
            msg += f"  slip_rms=(|u|={u_rms:.2e}, |u·n|={un_rms:.2e}, |u·t|={ut_rms:.2e})"
    if F_D_bm is not None and F_L_bm is not None:
        msg += f"  (bm) FD={F_D_bm:.6e} FL={F_L_bm:.6e} CD={C_D_bm:.6f} CL={C_L_bm:.6f}"
    print(msg)
    if ENABLE_PLOTS and step_idx % 2 == 0:
        u_k_func.plot(field='ux',
                      title=f"Velocity Ux at step {step_idx+1}",
                      xlabel='X-Axis', ylabel='Y-Axis',
                      levels=100, cmap='jet',
                      mask = fluid_domain,)

    # (Optional) append to global histories for later plotting
    if C_D_surf is not None and C_L_surf is not None:
        histories.setdefault("cd_surf", []).append(C_D_surf)
        histories.setdefault("cl_surf", []).append(C_L_surf)
        histories.setdefault("drag_surf", []).append(F_D_surf)
        histories.setdefault("lift_surf", []).append(F_L_surf)
        if diag_interface and C_D_nitsche is not None and C_L_nitsche is not None:
            histories.setdefault("drag_pen", []).append(F_D_pen)
            histories.setdefault("lift_pen", []).append(F_L_pen)
            histories.setdefault("cd_nitsche", []).append(C_D_nitsche)
            histories.setdefault("cl_nitsche", []).append(C_L_nitsche)
            histories.setdefault("u_rms_gamma", []).append(u_rms)
            histories.setdefault("un_rms_gamma", []).append(un_rms)
            histories.setdefault("ut_rms_gamma", []).append(ut_rms)
    if C_D_bm is not None and C_L_bm is not None:
        histories.setdefault("cd_bm", []).append(C_D_bm)
        histories.setdefault("cl_bm", []).append(C_L_bm)
        histories.setdefault("drag_bm", []).append(F_D_bm)
        histories.setdefault("lift_bm", []).append(F_L_bm)
    histories.setdefault("pA", []).append(pA)
    histories.setdefault("pB", []).append(pB)
    histories.setdefault("dp", []).append(dp)
    histories.setdefault("p_mean", []).append(p_mean)
    histories.setdefault("ux_stag", []).append(ux_stag)
    histories.setdefault("uy_stag", []).append(uy_stag)
    # FeatFlow reference files record functionals at the θ-time (t_{n+θ}).
    t_out = (int(step_idx) + float(theta.value)) * float(dt.value)
    histories.setdefault("time", []).append(float(t_out))

# Pre-build force integrals (surface and/or Babuska–Miller) once.
_init_force_functionals()



# In[ ]:


from pycutfem.solvers.nonlinear_solver import (NewtonSolver, 
                                               NewtonParameters, 
                                               TimeStepperParameters,) 
                                            #    AdamNewtonSolver,
                                            #    PetscSnesNewtonSolver)
# from pycutfem.solvers.aainhb_solver import AAINHBSolver           # or get_solver("aainhb")
import time

# build residual_form, jacobian_form, dof_handler, mixed_element, bcs, bcs_homog …
time_params = TimeStepperParameters(dt=dt.value,max_steps=max_time_steps
                                    ,stop_on_steady=True, 
                                    steady_tol=1e-6, theta= theta.value)
dirichlet_dofs = set(dof_handler.get_dirichlet_data(bcs).keys())  # bcs = your Dirichlet BCs

t0 = time.time()

solver = NewtonSolver(
    residual_form, jacobian_form,
    dof_handler=dof_handler,
    mixed_element=mixed_element,
    bcs=bcs, bcs_homog=bcs_homog,
    newton_params=NewtonParameters(
        newton_tol=newton_tol,
        max_newton_iter=max_newton_iter,
        line_search=True,
        ls_mode=ls_mode,
    ),
    postproc_timeloop_cb=None,  # use post_step_refiner so u_n is still previous when evaluating Babuska–Miller
    backend=backend,
    deformation=deformation,
    # preproc_cb=post_cb,  # Optional: peak filter callback
)
t1 = time.time()
print(f"Solver setup time: {t1 - t0:.2f} seconds")

if init_mode == "stokes":
    print("[init] Solving steady Stokes problem for initial condition …")
    bcs_init = NewtonSolver._freeze_bcs(bcs, 0.0)
    dof_handler.apply_bcs(bcs_init, u_n, p_n)
    dof_handler.apply_bcs(bcs_init, u_k, p_k)

    rho_saved = float(rho_const.value)
    theta_saved = float(theta.value)
    rho_const.value = 0.0
    theta.value = 1.0
    try:
        _delta, _converged, _iters = solver._newton_loop([u_k, p_k], [u_n, p_n], None, bcs_init)
    finally:
        rho_const.value = rho_saved
        theta.value = theta_saved
    if not bool(_converged):
        raise RuntimeError("[init] steady Stokes solve did not converge")
    u_n.nodal_values[:] = u_k.nodal_values[:]
    p_n.nodal_values[:] = p_k.nodal_values[:]
    print(f"[init] steady Stokes converged in {_iters} Newton iterations")
elif init_mode == "zero":
    # Start from rest (u=0,p=0) at t=0. Time-dependent Dirichlet values are
    # imposed by the time stepper at t_{n+θ}.
    u_n.nodal_values.fill(0.0)
    p_n.nodal_values.fill(0.0)
    u_k.nodal_values.fill(0.0)
    p_k.nodal_values.fill(0.0)
else:
    raise RuntimeError(f"Unknown --init mode: {init_mode!r}")

if diag_kernels and backend == "jit":
    def _kernel_stats(klist):
        stats = []
        for ker in klist:
            dom = getattr(ker, "domain", "unknown")
            eids = getattr(ker, "eids", None)
            if eids is None and isinstance(getattr(ker, "static_args", None), dict):
                eids = ker.static_args.get("eids", None)
            try:
                nent = int(np.asarray(eids).shape[0]) if eids is not None else -1
            except Exception:
                nent = -1
            gdofs = None
            if isinstance(getattr(ker, "static_args", None), dict):
                gdofs = ker.static_args.get("gdofs_map", None)
            if isinstance(gdofs, np.ndarray) and gdofs.size:
                gmin = int(np.min(gdofs))
                gmax = int(np.max(gdofs))
                nneg = int(np.sum(gdofs < 0))
            else:
                gmin, gmax, nneg = -1, -1, -1
            stats.append((dom, nent, gmin, gmax, nneg))
        return stats

    print("[kernels] Jacobian kernels (domain, n_entities):")
    for dom, nent, gmin, gmax, nneg in _kernel_stats(getattr(solver, "kernels_K", [])):
        extra = f" gdofs[min,max]=({gmin},{gmax}) n<0={nneg}"
        print(f"  - {dom:14s} n={nent}{extra}")
    print("[kernels] Residual kernels (domain, n_entities):")
    for dom, nent, gmin, gmax, nneg in _kernel_stats(getattr(solver, "kernels_F", [])):
        extra = f" gdofs[min,max]=({gmin},{gmax}) n<0={nneg}"
        print(f"  - {dom:14s} n={nent}{extra}")

    # Preflight: assemble the residual at the very first time-dependent BC time.
    # This helps catch cases where the JIT path accidentally ignores inhomogeneous
    # Dirichlet data under deformation.
    try:
        t_bc0 = float(theta.value) * float(dt.value)
        bcs0 = solver._freeze_bcs(bcs, t_bc0)
        _funs0 = [u_k, p_k]
        _prev0 = [u_n, p_n]
        snap = [f.nodal_values.copy() for f in _funs0] + [f.nodal_values.copy() for f in _prev0]
        # Predictor (as in the time loop): current = previous, then apply BCs at t_bc0.
        for f, f_prev in zip(_funs0, _prev0):
            f.nodal_values[:] = f_prev.nodal_values[:]
        dof_handler.apply_bcs(bcs0, *_funs0)
        current0 = {f.name: f for f in _funs0}
        current0.update({f.name: f for f in _prev0})
        _, R0 = solver._assemble_system_reduced(current0, need_matrix=False)
        print(f"[kernels] preflight ‖R_red‖∞ = {float(np.linalg.norm(R0, ord=np.inf)):.3e} at t_bc={t_bc0:.3e}")
        print("[kernels] preflight per-kernel max(|Floc|):")
        for ker in getattr(solver, "kernels_F", []):
            gdofs = None
            if isinstance(getattr(ker, "static_args", None), dict):
                gdofs = ker.static_args.get("gdofs_map", None)
            if isinstance(gdofs, np.ndarray) and gdofs.shape[0] == 0:
                continue
            _, Floc, _ = ker.exec(current0)
            maxabs = float(np.max(np.abs(Floc))) if isinstance(Floc, np.ndarray) and Floc.size else 0.0
            dom = getattr(ker, "domain", "unknown")
            print(f"  - {dom:14s} max|Floc|={maxabs:.3e}")

        # Also check Jacobian kernel magnitudes on the facet-patch/ghost domains.
        # This helps diagnose cases where Kloc is accidentally all-zero under deformation.
        print("[kernels] preflight (selected) per-kernel max(|Kloc|):")
        for ker in getattr(solver, "kernels_K", []):
            dom = getattr(ker, "domain", "unknown")
            if dom not in {"facet_patch", "ghost_edge"}:
                continue
            gdofs = None
            if isinstance(getattr(ker, "static_args", None), dict):
                gdofs = ker.static_args.get("gdofs_map", None)
            if isinstance(gdofs, np.ndarray) and gdofs.shape[0] == 0:
                continue
            Kloc, _, _ = ker.exec(current0)
            maxabsK = float(np.max(np.abs(Kloc))) if isinstance(Kloc, np.ndarray) and Kloc.size else 0.0
            print(f"  - {dom:14s} max|Kloc|={maxabsK:.3e}")

        if ghost_measure == "patch":
            # Inspect facet-patch geometry conditioning (deformation can lead to
            # very small |detJ| if the inverse map diverges).
            try:
                geo_fp = dof_handler.precompute_facet_patch_factors(
                    facet_ids=ghost_edges_vel,
                    qdeg=int(volume_quadrature + 1),
                    level_set=analytic_level_set,
                    derivs={(0, 0), (1, 0), (0, 1)},
                    reuse=True,
                    allow_interface=False,
                    deformation=deformation,
                )
                det = np.asarray(geo_fp.get("detJ", np.empty((0, 0))), dtype=float)
                inv = np.asarray(geo_fp.get("J_inv", np.empty((0, 0, 0, 0))), dtype=float)
                inv_pos = np.asarray(geo_fp.get("J_inv_pos", np.empty((0, 0, 0, 0))), dtype=float)
                inv_neg = np.asarray(geo_fp.get("J_inv_neg", np.empty((0, 0, 0, 0))), dtype=float)
                if det.size:
                    print(f"[facet_patch] min|detJ|={float(np.min(np.abs(det))):.3e}  max|detJ|={float(np.max(np.abs(det))):.3e}")
                if inv.size:
                    print(f"[facet_patch] max|J_inv|={float(np.max(np.abs(inv))):.3e}")
                if inv_pos.size:
                    print(f"[facet_patch] max|J_inv_pos|={float(np.max(np.abs(inv_pos))):.3e}")
                if inv_neg.size:
                    print(f"[facet_patch] max|J_inv_neg|={float(np.max(np.abs(inv_neg))):.3e}")
            except Exception as _exc:
                print(f"[facet_patch] geom stats failed: {_exc}")
    finally:
        # Restore state before actual solve_time_interval.
        if 'snap' in locals():
            for f, buf in zip(_funs0 + _prev0, snap):
                f.nodal_values[:] = buf
# Unconstrained Newton (fast, robust)
# solver = PetscSnesNewtonSolver(
#     residual_form, jacobian_form,
#     dof_handler=dof_handler,
#     mixed_element=mixed_element,
#     bcs=bcs, bcs_homog=bcs_homog,
#     newton_params=NewtonParameters(newton_tol=1e-6, line_search=False, max_newton_iter=60), # Line search handled by PETSc
#     postproc_timeloop_cb=save_solution,
#     petsc_options={
#         # --- SNES (Nonlinear Solver) Options ---
#         "snes_type": "newtonls",
#         "snes_linesearch_type": "bt",
#         "snes_converged_reason": None,
#         "snes_monitor": None,

#         # --- KSP (Linear Solver) and PC (Preconditioner) Options ---
#         "ksp_type": "gmres",  # A flexible iterative solver is best for fieldsplit
#         "pc_type": "fieldsplit",
#         "pc_fieldsplit_type": "schur",

#         # Configure the Schur factorization
#         # 'full' is most robust: S = C - B*inv(A)*B.T
#         "pc_fieldsplit_schur_fact_type": "full",

#         # --- Sub-solvers for the Velocity block (u) ---
#         # We'll use a direct solver (MUMPS) for the momentum block inv(A)
#         # This is often referred to as "ideal" Schur complement.
#         "fieldsplit_u_ksp_type": "preonly",
#         "fieldsplit_u_pc_type": "lu",
#         "fieldsplit_u_pc_factor_mat_solver_type": "mumps",

#         # --- Sub-solvers for the Pressure block (the Schur complement S) ---
#         # The solve on S is often approximate. We use GMRES + a simple PC.
#         "fieldsplit_p_ksp_type": "gmres",
#         "fieldsplit_p_pc_type": "hypre",   # Use Hypre's BoomerAMG 
#         "fieldsplit_p_ksp_rtol": 1e-5,
#     },
# )

# 2. Define the field splitting for the Schur complement
# This tells the solver how to partition the system.
# solver.set_schur_fieldsplit(
#     split_map={
#         'u': ['ux', 'uy'],  # The 'A' block in the matrix
#         'p': ['p'],         # The 'C' block (and where S operates)
#     }
# )

# VI solver (semismooth; we’ll run it for only 1–2 iterations per step)
# vi = PetscSnesNewtonSolver(
#     residual_form, jacobian_form,
#     dof_handler=dof_handler, mixed_element=mixed_element,
#     bcs=bcs, bcs_homog=bcs_homog,
#     newton_params=NewtonParameters(newton_tol=1e-6, line_search=True),
#     postproc_timeloop_cb=save_solution,
#     petsc_options={
#         "snes_type": "vinewtonssls",      # semismooth VI is robust near kinks
#         "snes_linesearch_type": "bt",
#         "snes_vi_monitor": None,          # optional: see active set
#         "ksp_type": "preonly", "pc_type": "lu",
#         "pc_factor_mat_solver_type": "mumps",
#     },
# )
# # Cap |u| only in a narrow band around the interface on φ>=0 (fluid) side
# Ucap = 4.0 * U_mean        # pick something physically reasonable
# vi.set_vi_on_interface_band(
#     level_set,
#     fields=("ux", "uy"),
#     side="+",
#     band_width=0.5,               # ~ one h around interface
#     bounds_by_field={"ux": (-Ucap, Ucap), "uy": (-Ucap, Ucap)},
# )
# primary unknowns
functions = [u_k, p_k]
prev_functions = [u_n, p_n]
# solver = AdamNewtonSolver(
#     residual_form, jacobian_form,
#     dof_handler=dof_handler,
#     mixed_element=mixed_element,
#     bcs=bcs, bcs_homog=bcs_homog,
#     newton_params=NewtonParameters(newton_tol=1e-6)
# )
# solver = AAINHBSolver(
#     residual_form, jacobian_form,
#     dof_handler=dof_handler,
#     mixed_element=mixed_element,
#     bcs=bcs, bcs_homog=bcs_homog,
#     newton_params=NewtonParameters(newton_tol=1e-6),
# )
# from petsc4py import PETSc
def vi_clip(step, bcs_now, funs, prev_funs):
    if step < 1:
        return

    vi._ensure_snes(len(vi.active_dofs), PETSc.COMM_WORLD)

    # ---- project current state into the box and write back to fields ----
    # assemble the reduced current guess from the fields
    x0_full = np.hstack([f.nodal_values for f in funs]).copy()
    x0_red  = x0_full[vi.active_dofs].copy()

    if getattr(vi, "_XL", None) is not None and getattr(vi, "_XU", None) is not None:
        lo = vi._XL.getArray(readonly=True)
        hi = vi._XU.getArray(readonly=True)
        x0_red = np.minimum(np.maximum(x0_red, lo), hi)

        # write the projected state back to the fields (absolute assign)
        new_full = x0_full
        new_full[vi.active_dofs] = x0_red
        for f in funs:
            g = f._g_dofs
            f.set_nodal_values(g, new_full[g])
        vi.dh.apply_bcs(bcs_now, *funs)

    # ---- short VI solve (1–2 steps is usually enough) ----
    old = vi._snes.getTolerances()
    vi._snes.setTolerances(rtol=0.0, atol=vi.np.newton_tol, max_it=2)
    _ = vi._newton_loop(funs, prev_funs, aux_funcs=None, bcs_now=bcs_now)
    vi._snes.setTolerances(*old)


t2 = time.time()



def plotting():
    t3 = time.time()
    print(f"Total solve time: {t3 - t2:.2f} seconds")
    print(f"Total setup + solve time: {t3 - t0:.2f} seconds")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    offset = 5
    t_hist = histories.get("time", [])
    if not t_hist:
        print("No history data collected; skipping plots.")
        return

    # --- Row 0: Coefficients ---
    # Subplot (0, 0) for Drag Coefficient
    if "cd_surf" in histories:
        axes[0, 0].plot(t_hist[offset:], histories["cd_surf"][offset:], label="Cd (surf)", color="blue")
    if "cd_bm" in histories:
        axes[0, 0].plot(t_hist[offset:], histories["cd_bm"][offset:], label="Cd (bm)", color="tab:cyan")
    axes[0, 0].set_ylabel("Drag Coefficient (Cd)")
    axes[0, 0].grid(True, linestyle=":", linewidth=0.5)
    axes[0, 0].set_title("Drag Coefficient over Time")
    axes[0, 0].legend()

    # Subplot (0, 1) for Lift Coefficient
    if "cl_surf" in histories:
        axes[0, 1].plot(t_hist[offset:], histories["cl_surf"][offset:], label="Cl (surf)", color="green")
    if "cl_bm" in histories:
        axes[0, 1].plot(t_hist[offset:], histories["cl_bm"][offset:], label="Cl (bm)", color="tab:olive")
    axes[0, 1].set_ylabel("Lift Coefficient (Cl)")
    axes[0, 1].grid(True, linestyle=":", linewidth=0.5)
    axes[0, 1].set_title("Lift Coefficient over Time")
    axes[0, 1].legend()

    # --- Row 1: Forces ---
    # Subplot (1, 0) for Drag Force
    if "drag_surf" in histories:
        axes[1, 0].plot(t_hist[offset:], histories["drag_surf"][offset:], label="Drag (surf)", color="red")
    if "drag_bm" in histories:
        axes[1, 0].plot(t_hist[offset:], histories["drag_bm"][offset:], label="Drag (bm)", color="tab:pink")
    axes[1, 0].set_ylabel("Drag Force")
    axes[1, 0].grid(True, linestyle=":", linewidth=0.5)
    axes[1, 0].set_title("Drag Force over Time")
    axes[1, 0].legend()

    # Subplot (1, 1) for Lift Force
    if "lift_surf" in histories:
        axes[1, 1].plot(t_hist[offset:], histories["lift_surf"][offset:], label="Lift (surf)", color="purple")
    if "lift_bm" in histories:
        axes[1, 1].plot(t_hist[offset:], histories["lift_bm"][offset:], label="Lift (bm)", color="tab:brown")
    axes[1, 1].set_ylabel("Lift Force")
    axes[1, 1].grid(True, linestyle=":", linewidth=0.5)
    axes[1, 1].set_title("Lift Force over Time")
    axes[1, 1].legend()

    # --- Row 2: Pressure Drop ---
    # Subplot (2, 0) for Pressure Drop
    if "dp" in histories:
        axes[2, 0].plot(t_hist[offset:], histories["dp"][offset:], label="Δp", color="orange")
    axes[2, 0].set_xlabel("Time")
    axes[2, 0].set_ylabel("Pressure Drop (Δp)")
    axes[2, 0].grid(True, linestyle=":", linewidth=0.5)
    axes[2, 0].set_title("Pressure Drop over Time")
    axes[2, 0].legend()

    # Subplot (2, 1) is unused, so we turn it off
    fig.delaxes(axes[2, 1])

    # Add a main title to the entire figure
    fig.suptitle("Flow Diagnostics for Turek Benchmark", fontsize=16)

    # Adjust layout to prevent titles and labels from overlapping
    fig.tight_layout(rect=[0, 0, 1, 0.96]) # rect leaves space for suptitle
    plt.savefig((Path(output_dir)/Path("turek_results.png")))
    if ENABLE_PLOTS:
        plt.show()
    else:
        plt.close(fig)


    # In[ ]:


    if ENABLE_PLOTS:
        u_n.plot(kind="contour",mask =fluid_domain,
                title="Turek-Schafer",
                xlabel='X-Axis', ylabel='Y-Axis',
                levels=100, cmap='jet')


    # In[ ]:


    if ENABLE_PLOTS:
        p_n.plot(
                title="Pressure",mask =fluid_domain)


def _write_functionals_csv(out_dir: Path, hist: dict) -> Path:
    """Write a single CSV file with the key benchmark functionals."""
    import csv
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "functionals.csv"

    n = int(len(hist.get("time", [])))
    if n == 0:
        return path

    def series(key: str) -> list[float]:
        vals = hist.get(key)
        if vals is None:
            return [float("nan")] * n
        if len(vals) != n:
            # keep the file rectangular; truncate/pad with NaN
            v = list(vals)[:n]
            if len(v) < n:
                v.extend([float("nan")] * (n - len(v)))
            return v
        return list(vals)

    rows = zip(
        range(n),
        series("time"),
        series("drag_surf"),
        series("lift_surf"),
        series("cd_surf"),
        series("cl_surf"),
        series("drag_pen"),
        series("lift_pen"),
        series("cd_nitsche"),
        series("cl_nitsche"),
        series("u_rms_gamma"),
        series("un_rms_gamma"),
        series("ut_rms_gamma"),
        series("drag_bm"),
        series("lift_bm"),
        series("cd_bm"),
        series("cl_bm"),
        series("pA"),
        series("pB"),
        series("dp"),
        series("p_mean"),
        series("ux_stag"),
        series("uy_stag"),
    )

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "step",
                "time",
                "Fd_surf",
                "Fl_surf",
                "Cd_surf",
                "Cl_surf",
                "Fd_pen",
                "Fl_pen",
                "Cd_nitsche",
                "Cl_nitsche",
                "u_rms_gamma",
                "un_rms_gamma",
                "ut_rms_gamma",
                "Fd_bm",
                "Fl_bm",
                "Cd_bm",
                "Cl_bm",
                "pA",
                "pB",
                "dp",
                "p_mean",
                "ux_stag",
                "uy_stag",
            ]
        )
        for row in rows:
            w.writerow(row)
    return path


def _finalize_outputs(status: str) -> None:
    try:
        csv_path = _write_functionals_csv(output_dir, histories)
        print(f"[turek] wrote {csv_path}")
    except Exception as exc:
        print(f"[turek] failed to write functionals.csv: {exc}")
    if benchmark == "2d-1":
        # DFG 2D-1 reference values (high-order spectral methods, see FeatFlow page).
        ref_cd = 5.57953523384
        ref_cl = 0.010618948146
        ref_dp = 0.11752016697
        try:
            cd = float(histories.get("cd_surf", [float("nan")])[-1])
            cl = float(histories.get("cl_surf", [float("nan")])[-1])
            dp = float(histories.get("dp", [float("nan")])[-1])
            print(f"[DFG 2D-1 ref]   Cd={ref_cd:.12f}  Cl={ref_cl:.12f}  dp={ref_dp:.12f}")
            print(f"[DFG 2D-1 last]  Cd={cd:.12f}  Cl={cl:.12f}  dp={dp:.12f}")
            print(f"[DFG 2D-1 |err|] Cd={abs(cd-ref_cd):.3e}  Cl={abs(cl-ref_cl):.3e}  dp={abs(dp-ref_dp):.3e}")
        except Exception as exc:
            print(f"[DFG 2D-1] failed to summarize reference errors: {exc}")
    try:
        plotting()
    except Exception as exc:
        print(f"[turek] plotting failed: {exc}")

try:
    def post_step(step, bcs_now, funs, prev_funs):
        save_solution(funs, prev_funs, step_idx=step)

    solver.solve_time_interval(functions=functions,
                           prev_functions= prev_functions,
                           time_params=time_params,
                           post_step_refiner=post_step,
                           )
    print("Simulation run successfully ...")
    _finalize_outputs("ok")
except Exception as e:
    import traceback
    print("Solver failed:", e)
    traceback.print_exc()
    _finalize_outputs("failed")
    
