#!/usr/bin/env python
"""
MMS: extreme cut-cell sliver sweep for fully-Eulerian FSI.

Purpose
-------
Provide a *reproducible* and *systematic* way to stress the unfitted FSI solver under
near-vanishing cut fractions (θ ≪ 1) while keeping a known exact solution, so we can
separate:
  - robustness/conditioning issues (Newton/linear solve breakdown),
  - accuracy impact of sliver stabilization.

We reuse the validated shear MMS from `fsi_mms_shear_flap.py` and move a straight-line
interface each time step to hit a prescribed list of target solid cut fractions.

Interface geometry
------------------
Level set (fluid "+", solid "-"):
  φ(x,y;c) = a x + b y - c

We pick a mesh vertex (xv,yv) and set c so that φ(xv,yv) = -δ (δ>0 small), which
generates a tiny *solid* sliver in one of the cut elements. We then correct δ by a few
multiplicative updates so that the *measured* min θ- matches the requested target.

Sliver-mass stabilization and “auto-scale”
------------------------------------------
We add the cut-cell sliver-mass term (fluid/solid):
  (ρ/dt) * (γ/θ) * ⟨u, v⟩  on the *cut subdomain*.

This script deliberately supports two θ modes:
  - **clipped θ** (`--theta-floor`, default 1e-6): mimics production runs where θ is
    floored to avoid division-by-zero and extreme coefficients.
  - **auto sliver-mass** (`--auto-sliver-mass`): uses *raw* Hansbo θ in the sliver-mass
    denominator so the stabilization stays effective even when θ_true < theta-floor.

    Optionally (if `--auto-sliver-mass-theta0 > 0`), also scale γ based on the measured
    min(θ) on cut elements:

        γ := γ0 * min(max_scale, max(1, θ0 / minθ))

Typical runs
------------
Reproduce breakdown (clipped θ; should fail for θ targets below `--theta-floor`):

  python -u examples/turek_fsi_fully_eulerian/fsi_mms_extreme_sliver_sweep.py \
    --nx 40 --ny 20 --poly-order-u 2 --poly-order-p 1 --poly-order-d 1 --dt 0.01 \
    --sliver-theta-list "1e-4,1e-6,1e-8,1e-10,1e-12" --sliver-mass-solid 1.0 \
    --use-hansbo-kappa

Fix via auto-scaled sliver mass (should run through the sweep and report errors):

  python -u examples/turek_fsi_fully_eulerian/fsi_mms_extreme_sliver_sweep.py \
    --nx 40 --ny 20 --poly-order-u 2 --poly-order-p 1 --poly-order-d 1 --dt 0.01 \
    --sliver-theta-list "1e-4,1e-6,1e-8,1e-10,1e-12" --sliver-mass-solid 1.0 \
    --use-hansbo-kappa --auto-sliver-mass

If you want to isolate *mass / cut-conditioning* effects (no stresses), use:
  --mms-profile const
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.geometry import hansbo_cut_ratio
from pycutfem.core.levelset import LevelSetGridFunction
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters, _ActiveReducer
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    ElementWiseConstant,
    Function,
    Neg,
    Pos,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    dot,
    inner,
    jump,
    restrict,
)
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.helpers import analyze_active_dofs
from pycutfem.ufl.measures import dFacetPatch, dGhost, dInterface, dx
from examples.utils.fsi.fully_eulerian import (
    build_fsi_eulerian_forms,
    make_domain_sets,
    refresh_domain_sets,
    retag_inactive,
)
from pycutfem.utils.meshgen import structured_quad


def _parse_float_list(s: str) -> List[float]:
    if not s:
        return []
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _tag_rect_boundaries(mesh: Mesh, *, x0: float, x1: float, y0: float, y1: float, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - x0) <= tol,
            "right": lambda x, y: abs(x - x1) <= tol,
            "bottom": lambda x, y: abs(y - y0) <= tol,
            "top": lambda x, y: abs(y - y1) <= tol,
        }
    )


def _recompute_active_dofs(solver: NewtonSolver, *, bcs_active: List[BoundaryCondition]) -> bool:
    dh = solver.dh
    constraints = getattr(solver, "constraints", None)
    ndof_eff = int(constraints.n_master) if constraints is not None else int(dh.total_dofs)
    old_active = np.asarray(getattr(solver, "active_dofs", np.empty(0, dtype=int)), dtype=int)

    active_by_restr, has_restriction = analyze_active_dofs(solver.equation, dh, solver.me, bcs_active, verbose=False)
    bc_dofs_full = set(dh.get_dirichlet_data(bcs_active).keys())
    inactive_full = set(getattr(dh, "dof_tags", {}).get("inactive", set()))
    inactive_free_full = inactive_full - bc_dofs_full

    if constraints is None:
        candidate = set(active_by_restr) if has_restriction else set(range(ndof_eff))
        free = sorted((candidate - bc_dofs_full) - inactive_free_full)
        new_active = np.asarray(free, dtype=int)
    else:
        candidate_master = constraints.to_master_set(active_by_restr) if has_restriction else set(range(ndof_eff))
        bc_master = constraints.to_master_set(bc_dofs_full)
        inactive_master = constraints.to_master_set(inactive_free_full)
        free = sorted((candidate_master - bc_master) - inactive_master)
        new_active = np.asarray(free, dtype=int)

    if old_active.size == new_active.size and np.array_equal(old_active, new_active):
        return False

    solver.active_dofs = new_active
    solver.full_to_red = -np.ones(ndof_eff, dtype=int)
    solver.full_to_red[new_active] = np.arange(len(new_active), dtype=int)
    solver.red_to_full = new_active
    solver.use_reduced = len(new_active) < ndof_eff
    solver.restrictor = _ActiveReducer(dh, new_active, constraint=constraints)
    solver._pattern_stale = True
    return True


@dataclass
class SliverSweepState:
    step_idx: int
    theta_targets: List[float]
    records: List[dict]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MMS shear case with moving interface to generate extreme cut slivers"
    )

    # Mesh / FE
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--q", type=int, default=6, help="Quadrature order")
    parser.add_argument("--poly-order-u", type=int, default=2)
    parser.add_argument("--poly-order-p", type=int, default=1)
    parser.add_argument(
        "--poly-order-d",
        type=int,
        default=1,
        help="Solid displacement polynomial order (match Turek: DS=1)",
    )
    parser.add_argument(
        "--poly-order-ls",
        type=int,
        default=1,
        help="Level-set polynomial order (P1 is enough for an affine line)",
    )
    parser.add_argument(
        "--use-facet-patch-ghost",
        action="store_true",
        default=True,
        help="Use facet-patch ghost measure (recommended for CG; default)",
    )
    parser.add_argument(
        "--no-facet-patch-ghost",
        action="store_false",
        dest="use_facet_patch_ghost",
        help="Use standard ghost-edge measure",
    )

    # Time
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--sliver-theta-list",
        type=str,
        default="1e-4,1e-6,1e-8,1e-10,1e-12",
        help="Comma-separated list of target min(theta_neg) values, one per timestep",
    )

    # Interface line φ = a x + b y - c
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument(
        "--grazing-i",
        type=int,
        default=None,
        help="Vertex i-index (0..nx) near which the interface grazes; default nx//2",
    )
    parser.add_argument(
        "--grazing-j",
        type=int,
        default=None,
        help="Vertex j-index (0..ny) near which the interface grazes; default ny//2",
    )
    parser.add_argument(
        "--ls-tol",
        type=float,
        default=1.0e-12,
        help="Tolerance for level-set commit/classification",
    )

    # Sliver handling knobs
    parser.add_argument(
        "--theta-floor",
        type=float,
        default=1.0e-6,
        help="Floor for θ used in kappa/sliver-mass expressions",
    )
    parser.add_argument(
        "--use-hansbo-kappa",
        action="store_true",
        default=False,
        help="Use Hansbo kappa weights based on θ instead of constant 0.5",
    )
    parser.add_argument(
        "--no-hansbo-kappa",
        action="store_false",
        dest="use_hansbo_kappa",
        help="Use kappa_pos=kappa_neg=0.5",
    )
    parser.add_argument(
        "--sliver-mass-fluid",
        type=float,
        default=float(os.getenv("PYCUTFEM_SLIVER_MASS_FLUID", "1.0")),
        help="γ for fluid sliver-mass stabilization (default from env or 1.0)",
    )
    parser.add_argument(
        "--sliver-mass-solid",
        type=float,
        default=float(os.getenv("PYCUTFEM_SLIVER_MASS_SOLID", "1.0")),
        help="γ for solid sliver-mass stabilization (default from env or 1.0)",
    )
    parser.add_argument(
        "--auto-sliver-mass",
        action="store_true",
        default=False,
        help="Auto-scale sliver-mass γ using min θ and use raw θ in the denominator",
    )
    parser.add_argument(
        "--auto-sliver-mass-theta0",
        type=float,
        default=0.0,
        help="Optional reference θ0 for extra γ scaling (0 disables): γ *= max(1, θ0/minθ)",
    )
    parser.add_argument(
        "--auto-sliver-mass-max-scale",
        type=float,
        default=1.0e6,
        help="Max multiplicative scale applied to γ when auto-scaling",
    )

    # Physics
    parser.add_argument("--rho-f", type=float, default=1.0)
    parser.add_argument("--rho-s", type=float, default=1.0)
    parser.add_argument("--mu-f", type=float, default=1.0)
    parser.add_argument("--mu-s", type=float, default=1.0)
    parser.add_argument("--beta-n", type=float, default=20.0)
    parser.add_argument("--A", type=float, default=1.0)
    parser.add_argument(
        "--mms-profile",
        choices=("const", "linear"),
        default="linear",
        help=(
            "Manufactured solution spatial profile ψ(x). "
            "'const' makes stresses zero (mass-only; best to isolate sliver-mass effects). "
            "'linear' adds a constant shear stress (more stringent, but can be dominated by interface-traction effects)."
        ),
    )
    parser.add_argument(
        "--gamma-v",
        type=float,
        default=0.1,
        help="Ghost-penalty strength (shared across fluid/solid velocity in build_fsi_eulerian_forms)",
    )
    parser.add_argument("--gamma-p", type=float, default=0.0, help="Fluid pressure ghost-penalty strength")
    parser.add_argument("--gamma-v-grad", type=float, default=0.1, help="Solid displacement ghost-penalty strength")
    parser.add_argument(
        "--solid-vel-ghost-mass",
        type=float,
        default=1.0,
        help="Extra solid velocity mass-jump ghost penalty (controls constant modes on ghost edges)",
    )
    parser.add_argument(
        "--fluid-vel-ghost-mass",
        type=float,
        default=0.0,
        help="Extra fluid velocity mass-jump ghost penalty (controls constant modes on ghost edges)",
    )

    # Solver
    parser.add_argument("--backend", choices=("python", "jit", "cpp"), default="jit")
    parser.add_argument("--newton-tol", type=float, default=1.0e-12)
    parser.add_argument("--max-newton-iter", type=int, default=12)
    parser.add_argument("--line-search", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument(
        "--debug-max-error",
        action="store_true",
        default=False,
        help="Print the location/DOF id of the max |error| for each reported field",
    )
    parser.add_argument(
        "--check-residual-only",
        action="store_true",
        default=False,
        help="Assemble and report the residual at the exact fields for the first step, then exit",
    )

    args = parser.parse_args()

    theta_targets = _parse_float_list(args.sliver_theta_list)
    if not theta_targets:
        raise ValueError("--sliver-theta-list produced an empty list")

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------
    x0, x1 = -1.0, 1.0
    y0, y1 = -0.5, 0.5
    Lx, Ly = x1 - x0, y1 - y0

    nodes, elems, edges, corners = structured_quad(
        Lx,
        Ly,
        nx=int(args.nx),
        ny=int(args.ny),
        poly_order=int(args.poly_order_u),
        offset=(x0, y0),
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=int(args.poly_order_u),
    )
    _tag_rect_boundaries(mesh, x0=x0, x1=x1, y0=y0, y1=y1)

    hx = float(Lx) / float(args.nx)
    hy = float(Ly) / float(args.ny)

    gi = int(args.grazing_i) if args.grazing_i is not None else int(args.nx // 2)
    gj = int(args.grazing_j) if args.grazing_j is not None else int(args.ny // 2)
    if not (0 <= gi <= int(args.nx) and 0 <= gj <= int(args.ny)):
        raise ValueError("grazing indices out of range")
    xv = float(x0 + gi * hx)
    yv = float(y0 + gj * hy)
    a_n = float(args.a)
    b_n = float(args.b)
    c_base = a_n * xv + b_n * yv

    # ------------------------------------------------------------------
    # Moving level set (grid function)
    # ------------------------------------------------------------------
    ls_me = MixedElement(mesh, field_specs={"phi": int(args.poly_order_ls)})
    ls_dh = DofHandler(ls_me, method="cg")
    level_set = LevelSetGridFunction(ls_dh, field="phi")
    ls_coords = np.asarray(ls_dh.get_dof_coords("phi"), dtype=float)

    domains = None  # will be created after the first commit

    # ------------------------------------------------------------------
    # MMS data (shear flap; global smooth extension)
    # ------------------------------------------------------------------
    mu_f = float(args.mu_f)
    mu_s = float(args.mu_s)
    alpha = mu_s / mu_f if mu_f != 0.0 else 1.0
    A = float(args.A)

    if args.mms_profile == "const":
        # Stress-free (∇u=0, ∇d=0): isolates mass/cut-conditioning effects.
        def psi(x):
            return 0.0 * x + 1.0

        def psi_dd(x):
            return 0.0 * x
    else:
        # Constant-shear (linear) profile; representable with DS=1.
        def psi(x):
            return x

        def psi_dd(x):
            return 0.0 * x

    def w_exact(x, t):
        return A * np.exp(alpha * t) * psi(x)

    def u_exact_y(x, t):
        return alpha * w_exact(x, t)

    def u_exact_y_xx(x, t):
        return alpha * A * np.exp(alpha * t) * psi_dd(x)

    def w_exact_xx(x, t):
        return A * np.exp(alpha * t) * psi_dd(x)

    time_state = {"t_prev": 0.0, "t_curr": 0.0, "dt": 0.0}

    def f_f_y_discrete(x):
        t0 = float(time_state["t_prev"])
        t1 = float(time_state["t_curr"])
        dt_step = float(time_state["dt"])
        u0 = u_exact_y(x, t0)
        u1 = u_exact_y(x, t1)
        return (u1 - u0) / max(dt_step, 1.0e-16) - mu_f * u_exact_y_xx(x, t1)

    def f_s_y_discrete(x):
        t0 = float(time_state["t_prev"])
        t1 = float(time_state["t_curr"])
        dt_step = float(time_state["dt"])
        u0 = u_exact_y(x, t0)
        u1 = u_exact_y(x, t1)
        return (u1 - u0) / max(dt_step, 1.0e-16) - mu_s * w_exact_xx(x, t1)

    def g_disp_y_discrete(x):
        t0 = float(time_state["t_prev"])
        t1 = float(time_state["t_curr"])
        dt_step = float(time_state["dt"])
        d0 = w_exact(x, t0)
        d1 = w_exact(x, t1)
        u1 = u_exact_y(x, t1)
        return (d1 - d0) / max(dt_step, 1.0e-16) - u1

    f_f = Analytic(lambda x, y: np.stack((0.0 * x, f_f_y_discrete(x)), axis=-1), degree=4)
    f_s = Analytic(lambda x, y: np.stack((0.0 * x, f_s_y_discrete(x)), axis=-1), degree=4)
    g_d = Analytic(lambda x, y: np.stack((0.0 * x, g_disp_y_discrete(x)), axis=-1), degree=4)

    # ------------------------------------------------------------------
    # Helpers: level set commit + θ stats + (delta -> target theta) solve
    # ------------------------------------------------------------------
    def _commit_delta(delta: float) -> None:
        c_val = float(c_base + float(delta))
        phi = a_n * ls_coords[:, 0] + b_n * ls_coords[:, 1] - c_val
        level_set.set_from_array(phi)
        level_set.commit(tol=float(args.ls_tol))
        nonlocal domains
        if domains is not None:
            refresh_domain_sets(mesh, domains)

    def _theta_stats() -> Tuple[np.ndarray, np.ndarray, float, float]:
        thp = hansbo_cut_ratio(mesh, level_set, side="+")
        thn = hansbo_cut_ratio(mesh, level_set, side="-")
        cut_mask = mesh.element_bitset("cut").mask
        if np.any(cut_mask):
            mn_p = float(np.min(thp[cut_mask]))
            mn_n = float(np.min(thn[cut_mask]))
        else:
            mn_p = 1.0
            mn_n = 1.0
        return thp, thn, mn_p, mn_n

    def _delta_for_target(theta_target: float) -> dict:
        theta_target = float(theta_target)
        if theta_target <= 0.0:
            raise ValueError("theta_target must be positive")

        # Initial guess (structured quad + diagonal cut corner scaling).
        delta = float(np.sqrt(max(2.0 * theta_target * hx * hy, 0.0)))
        info: dict = {"theta_target": theta_target, "delta_init": delta}

        for it in range(8):
            _commit_delta(delta)
            thp_raw, thn_raw, mn_p, mn_n = _theta_stats()
            info.update(
                {
                    "iters": it + 1,
                    "delta": float(delta),
                    "theta_min_pos": float(mn_p),
                    "theta_min_neg": float(mn_n),
                }
            )
            if mn_n <= 0.0:
                ratio = 10.0
            else:
                ratio = theta_target / mn_n
            if 0.85 <= ratio <= 1.18:
                break
            delta *= float(np.sqrt(max(ratio, 1.0e-16)))

        return info

    # Commit the first target before building measures/forms.
    first = _delta_for_target(theta_targets[0])
    if args.verbose:
        print(
            "[sliver] init: target={theta_target:.1e}  minθ-={theta_min_neg:.3e}  minθ+={theta_min_pos:.3e}  δ={delta:.3e}".format(
                **first
            )
        )
        try:
            n_cut = int(mesh.element_bitset("cut").cardinality())
            n_with_seg = 0
            n_seg = 0
            for e in mesh.elements_list:
                if getattr(e, "tag", "") != "cut":
                    continue
                segs = getattr(e, "interface_segments", None) or []
                if segs:
                    n_with_seg += 1
                    n_seg += len(segs)
            print(
                f"[geom] cut_elems={n_cut} cut_with_segs={n_with_seg} n_interface_segs={n_seg} "
                f"interface_edges={mesh.edge_bitset('interface').cardinality()} "
                f"ghost_pos={mesh.edge_bitset('ghost_pos').cardinality()} "
                f"ghost_neg={mesh.edge_bitset('ghost_neg').cardinality()} "
                f"ghost_both={mesh.edge_bitset('ghost_both').cardinality()}"
            )
        except Exception:
            pass

    domains = make_domain_sets(mesh, use_aligned_interface=False)

    # Allocate θ arrays (raw + clipped) for kappa and sliver mass.
    n_elem = len(mesh.elements_list)
    theta_pos_raw_vals = np.ones(n_elem, dtype=float)
    theta_neg_raw_vals = np.ones(n_elem, dtype=float)
    theta_pos_vals = np.ones(n_elem, dtype=float)
    theta_neg_vals = np.ones(n_elem, dtype=float)

    def _refresh_theta_arrays() -> Tuple[float, float]:
        thp_raw, thn_raw, mn_p, mn_n = _theta_stats()
        theta_pos_raw_vals[:] = thp_raw
        theta_neg_raw_vals[:] = thn_raw
        floor = float(args.theta_floor)
        theta_pos_vals[:] = np.clip(thp_raw, floor, 1.0)
        theta_neg_vals[:] = np.clip(thn_raw, floor, 1.0)
        return float(mn_p), float(mn_n)

    _refresh_theta_arrays()

    # ------------------------------------------------------------------
    # Unknowns (matching build_fsi_eulerian_forms conventions)
    # ------------------------------------------------------------------
    poly_order_u = int(args.poly_order_u)
    poly_order_p = int(args.poly_order_p)
    poly_order_d = int(args.poly_order_d)

    me = MixedElement(
        mesh,
        field_specs={
            "u_pos_x": poly_order_u,
            "u_pos_y": poly_order_u,
            "p_pos_": poly_order_p,
            "vs_neg_x": poly_order_u,
            "vs_neg_y": poly_order_u,
            "d_neg_x": poly_order_d,
            "d_neg_y": poly_order_d,
        },
    )
    dh = DofHandler(me, method="cg")
    # Retag inactive DOFs based on the current geometry. For higher-order CG, we
    # keep a thin band of neighbor elements so ghost-penalty normal derivatives
    # remain well-defined.
    retag_inactive(dh, mesh)

    velocity_fluid = FunctionSpace(name="velocity_fluid", field_names=["u_pos_x", "u_pos_y"], dim=1, side="+")
    pressure_fluid = FunctionSpace(name="pressure_fluid", field_names=["p_pos_"], dim=0, side="+")
    velocity_solid = FunctionSpace(name="velocity_solid", field_names=["vs_neg_x", "vs_neg_y"], dim=1, side="-")
    displacement_solid = FunctionSpace(name="displacement", field_names=["d_neg_x", "d_neg_y"], dim=1, side="-")

    du_f = VectorTrialFunction(space=velocity_fluid, dof_handler=dh)
    dp_f = TrialFunction(name="trial_pressure_fluid", field_name="p_pos_", dof_handler=dh, side="+")
    du_s = VectorTrialFunction(space=velocity_solid, dof_handler=dh)
    ddisp_s = VectorTrialFunction(space=displacement_solid, dof_handler=dh)

    v_f = VectorTestFunction(space=velocity_fluid, dof_handler=dh)
    q_f = TestFunction(name="test_pressure_fluid", field_name="p_pos_", dof_handler=dh, side="+")
    v_s = VectorTestFunction(space=velocity_solid, dof_handler=dh)
    w_s = VectorTestFunction(space=displacement_solid, dof_handler=dh)

    uf_k = VectorFunction(name="u_f_k", field_names=["u_pos_x", "u_pos_y"], dof_handler=dh, side="+")
    pf_k = Function(name="p_f_k", field_name="p_pos_", dof_handler=dh, side="+")
    uf_n = VectorFunction(name="u_f_n", field_names=["u_pos_x", "u_pos_y"], dof_handler=dh, side="+")
    pf_n = Function(name="p_f_n", field_name="p_pos_", dof_handler=dh, side="+")
    us_k = VectorFunction(name="u_s_k", field_names=["vs_neg_x", "vs_neg_y"], dof_handler=dh, side="-")
    us_n = VectorFunction(name="u_s_n", field_names=["vs_neg_x", "vs_neg_y"], dof_handler=dh, side="-")
    disp_k = VectorFunction(name="disp_k", field_names=["d_neg_x", "d_neg_y"], dof_handler=dh, side="-")
    disp_n = VectorFunction(name="disp_n", field_names=["d_neg_x", "d_neg_y"], dof_handler=dh, side="-")

    for f in (uf_k, pf_k, uf_n, pf_n, us_k, us_n, disp_k, disp_n):
        f.nodal_values.fill(0.0)

    # ------------------------------------------------------------------
    # Measures
    # ------------------------------------------------------------------
    qvol = int(args.q)
    dx_fluid = dx(defined_on=domains["fluid_interface"], level_set=level_set, metadata={"q": qvol, "side": "+"})
    dx_solid = dx(defined_on=domains["solid_interface"], level_set=level_set, metadata={"q": qvol, "side": "-"})
    dx_fluid_cut = dx(defined_on=domains["cut_domain"], level_set=level_set, metadata={"q": qvol, "side": "+"})
    dx_solid_cut = dx(defined_on=domains["cut_domain"], level_set=level_set, metadata={"q": qvol, "side": "-"})
    dGamma = dInterface(
        defined_on=domains["cut_interface"],
        level_set=level_set,
        metadata={"q": qvol + 2, "derivs": {(0, 0), (0, 1), (1, 0)}},
    )
    # Use one ghost-edge measure that supports both value- and grad-based integrals.
    ghost_measure = dFacetPatch if bool(getattr(args, "use_facet_patch_ghost", True)) else dGhost
    dG_fluid = ghost_measure(
        defined_on=domains["fluid_ghost"],
        level_set=level_set,
        metadata={"q": qvol, "derivs": {(0, 0), (0, 1), (1, 0)}},
    )
    dG_solid = ghost_measure(
        defined_on=domains["solid_ghost"],
        level_set=level_set,
        metadata={"q": qvol, "derivs": {(0, 0), (0, 1), (1, 0)}},
    )

    # ---------------------------------------------------------------------
    # Hansbo kappa weights based on θ (optional)
    # ---------------------------------------------------------------------
    theta_pos_cell = ElementWiseConstant(theta_pos_vals)
    theta_neg_cell = ElementWiseConstant(theta_neg_vals)
    theta_pos_raw_cell = ElementWiseConstant(theta_pos_raw_vals)
    theta_neg_raw_cell = ElementWiseConstant(theta_neg_raw_vals)

    if args.use_hansbo_kappa:
        theta_sum = Pos(theta_pos_cell) + Neg(theta_neg_cell) + Constant(1.0e-12)
        kappa_pos = Pos(theta_pos_cell) / theta_sum
        kappa_neg = Neg(theta_neg_cell) / theta_sum
    else:
        kappa_pos = Constant(0.5)
        kappa_neg = Constant(0.5)

    # ------------------------------------------------------------------
    # Forms
    # ------------------------------------------------------------------
    dt = Constant(float(args.dt))
    dt._jit_name = "dt"
    theta = Constant(1.0)

    rho_f_c = Constant(float(args.rho_f))
    rho_s_c = Constant(float(args.rho_s))
    mu_f_c = Constant(float(args.mu_f))
    mu_s_c = Constant(float(args.mu_s))

    forms = build_fsi_eulerian_forms(
        du_f=du_f,
        dp_f=dp_f,
        du_s=du_s,
        ddisp_s=ddisp_s,
        test_vel_f=v_f,
        test_q_f=q_f,
        test_vel_s=v_s,
        test_disp_s=w_s,
        uf_k=uf_k,
        pf_k=pf_k,
        uf_n=uf_n,
        pf_n=pf_n,
        us_k=us_k,
        us_n=us_n,
        disp_k=disp_k,
        disp_n=disp_n,
        dx_fluid=dx_fluid,
        dx_solid=dx_solid,
        dGamma=dGamma,
        dG_fluid=dG_fluid,
        dG_solid=dG_solid,
        kappa_pos=kappa_pos,
        kappa_neg=kappa_neg,
        cell_h=CellDiameter(),
        beta_N=Constant(float(args.beta_n)),
        rho_f=rho_f_c,
        rho_s=rho_s_c,
        mu_f=mu_f_c,
        mu_s=mu_s_c,
        lambda_s=Constant(0.0),
        dt=dt,
        theta=theta,
        gamma_v=Constant(float(args.gamma_v)),
        gamma_p=Constant(float(args.gamma_p)),
        gamma_v_grad=Constant(float(args.gamma_v_grad)),
        svc_scale=rho_s_c / dt,
        solid_reg_eps=Constant(1.0e-6),
        use_linear_solid=True,
        solid_advect_lagged=True,
        s_nitsche_value=0.0,
    )

    # ---------------------------------------------------------------------
    # Sliver-mass stabilization (cut elements only)
    # ---------------------------------------------------------------------
    gamma_sliver_mass_f_base = float(args.sliver_mass_fluid)
    gamma_sliver_mass_s_base = float(args.sliver_mass_solid)
    gamma_sliver_mass_f = Constant(gamma_sliver_mass_f_base)
    gamma_sliver_mass_s = Constant(gamma_sliver_mass_s_base)

    inv_theta_f = Constant(1.0) / ((theta_pos_raw_cell if args.auto_sliver_mass else theta_pos_cell) + Constant(1.0e-16))
    inv_theta_s = Constant(1.0) / ((theta_neg_raw_cell if args.auto_sliver_mass else theta_neg_cell) + Constant(1.0e-16))

    def _update_auto_sliver_mass_gamma(*, theta_min_pos: float, theta_min_neg: float) -> None:
        if not bool(args.auto_sliver_mass):
            return
        theta0 = float(args.auto_sliver_mass_theta0)
        max_scale = float(args.auto_sliver_mass_max_scale)
        theta_min_pos = max(float(theta_min_pos), 1.0e-300)
        theta_min_neg = max(float(theta_min_neg), 1.0e-300)
        scale_f = min(max_scale, max(1.0, theta0 / theta_min_pos))
        scale_s = min(max_scale, max(1.0, theta0 / theta_min_neg))
        gamma_sliver_mass_f.value = gamma_sliver_mass_f_base * scale_f
        gamma_sliver_mass_s.value = gamma_sliver_mass_s_base * scale_s
        if args.verbose:
            print(
                f"[sliver-mass] auto γ: fluid={float(gamma_sliver_mass_f.value):.3e} "
                f"solid={float(gamma_sliver_mass_s.value):.3e} "
                f"(scale_f={scale_f:.3e}, scale_s={scale_s:.3e}, θ0={theta0:.1e})"
            )

    # Initialize γ based on the current cut state (before the first solve).
    mn_pos, mn_neg = _refresh_theta_arrays()
    _update_auto_sliver_mass_gamma(theta_min_pos=mn_pos, theta_min_neg=mn_neg)

    a_sliver_mass = (
        gamma_sliver_mass_f * (Constant(float(args.rho_f)) / dt) * inv_theta_f * dot(du_f, v_f) * dx_fluid_cut
        + gamma_sliver_mass_s * (Constant(float(args.rho_s)) / dt) * inv_theta_s * dot(du_s, v_s) * dx_solid_cut
    )
    # IMPORTANT: make the sliver-mass term *time-consistent* by stabilizing the
    # inertia (u^{n+1} - u^n)/dt on cut elements. Using u^{n+1} alone acts like a
    # geometry-dependent damping term and can destroy MMS accuracy.
    r_sliver_mass = (
        gamma_sliver_mass_f
        * (Constant(float(args.rho_f)) / dt)
        * inv_theta_f
        * dot(uf_k - uf_n, v_f)
        * dx_fluid_cut
        + gamma_sliver_mass_s
        * (Constant(float(args.rho_s)) / dt)
        * inv_theta_s
        * dot(us_k - us_n, v_s)
        * dx_solid_cut
    )

    residual_form = (
        forms.residual_form
        - dot(f_f, v_f) * dx_fluid
        - dot(f_s, v_s) * dx_solid
        - dot(g_d, w_s) * dx_solid
        + r_sliver_mass
    )
    jacobian_form = forms.jacobian_form + a_sliver_mass

    # Extra mass-jump ghost penalty (mirrors the optional knobs in the Turek script).
    # This targets constant (zero-gradient) modes that are not controlled by grad-jump
    # ghost penalties on extreme slivers.
    if float(args.solid_vel_ghost_mass) > 0.0:
        gamma_sv_mass = Constant(float(args.solid_vel_ghost_mass))
        jacobian_form = jacobian_form + (Constant(float(args.rho_s)) / dt) * gamma_sv_mass * (
            CellDiameter() * inner(jump(du_s), jump(v_s))
        ) * dG_solid
        residual_form = residual_form + (Constant(float(args.rho_s)) / dt) * gamma_sv_mass * (
            CellDiameter() * inner(jump(us_k), jump(v_s))
        ) * dG_solid
    if float(args.fluid_vel_ghost_mass) > 0.0:
        gamma_fv_mass = Constant(float(args.fluid_vel_ghost_mass))
        jacobian_form = jacobian_form + (Constant(float(args.rho_f)) / dt) * gamma_fv_mass * (
            CellDiameter() * inner(jump(du_f), jump(v_f))
        ) * dG_fluid
        residual_form = residual_form + (Constant(float(args.rho_f)) / dt) * gamma_fv_mass * (
            CellDiameter() * inner(jump(uf_k), jump(v_f))
        ) * dG_fluid

    # ------------------------------------------------------------------
    # Time-dependent BCs (applied at t_{n+1} because theta=1)
    # ------------------------------------------------------------------
    def u_x_bc(x, y, t):
        return 0.0

    def u_y_bc(x, y, t):
        return float(u_exact_y(np.asarray(x), float(t)))

    def p_bc(x, y, t):
        return 0.0

    def d_x_bc(x, y, t):
        return 0.0

    def d_y_bc(x, y, t):
        return float(w_exact(np.asarray(x), float(t)))

    bcs: List[BoundaryCondition] = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("u_pos_x", "dirichlet", tag, u_x_bc),
                BoundaryCondition("u_pos_y", "dirichlet", tag, u_y_bc),
                BoundaryCondition("p_pos_", "dirichlet", tag, p_bc),
                BoundaryCondition("vs_neg_x", "dirichlet", tag, u_x_bc),
                BoundaryCondition("vs_neg_y", "dirichlet", tag, u_y_bc),
                BoundaryCondition("d_neg_x", "dirichlet", tag, d_x_bc),
                BoundaryCondition("d_neg_y", "dirichlet", tag, d_y_bc),
            ]
        )
    bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]

    # ------------------------------------------------------------------
    # Newton solver
    # ------------------------------------------------------------------
    solver = None  # assigned below; used in closure

    def _pre_cb(_funcs):
        assert solver is not None
        t0 = float(getattr(solver, "_current_t", 0.0))
        dt_step = float(getattr(solver, "_current_dt", float(dt.value)))
        time_state["t_prev"] = t0
        time_state["t_curr"] = t0 + dt_step
        time_state["dt"] = dt_step

    solver = NewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(
            newton_tol=float(args.newton_tol),
            max_newton_iter=int(args.max_newton_iter),
            line_search=bool(args.line_search),
        ),
        quad_order=int(args.q),
        backend=str(args.backend),
        preproc_cb=_pre_cb,
    )

    if bool(args.check_residual_only):
        t_prev = 0.0
        t_curr = float(dt.value)
        time_state["t_prev"] = t_prev
        time_state["t_curr"] = t_curr
        time_state["dt"] = float(dt.value)

        # Fill exact fields at {t_n, t_{n+1}}.
        def _set_exact(vecfun, field: str, f):
            gd = np.asarray(dh.get_field_slice(field), dtype=int)
            xy = dh.get_dof_coords(field)
            vecfun.set_nodal_values(gd, np.asarray(f(xy[:, 0], t_curr), dtype=float))

        def _set_exact_prev(vecfun, field: str, f):
            gd = np.asarray(dh.get_field_slice(field), dtype=int)
            xy = dh.get_dof_coords(field)
            vecfun.set_nodal_values(gd, np.asarray(f(xy[:, 0], t_prev), dtype=float))

        # Fluid
        _set_exact(uf_k, "u_pos_x", lambda x, t: 0.0 * x)
        _set_exact(uf_k, "u_pos_y", u_exact_y)
        _set_exact_prev(uf_n, "u_pos_x", lambda x, t: 0.0 * x)
        _set_exact_prev(uf_n, "u_pos_y", u_exact_y)
        pf_k.nodal_values.fill(0.0)
        pf_n.nodal_values.fill(0.0)

        # Solid
        _set_exact(us_k, "vs_neg_x", lambda x, t: 0.0 * x)
        _set_exact(us_k, "vs_neg_y", u_exact_y)
        _set_exact_prev(us_n, "vs_neg_x", lambda x, t: 0.0 * x)
        _set_exact_prev(us_n, "vs_neg_y", u_exact_y)
        _set_exact(disp_k, "d_neg_x", lambda x, t: 0.0 * x)
        _set_exact(disp_k, "d_neg_y", w_exact)
        _set_exact_prev(disp_n, "d_neg_x", lambda x, t: 0.0 * x)
        _set_exact_prev(disp_n, "d_neg_y", w_exact)

        bcs_now = NewtonSolver._freeze_bcs(bcs, t_curr)
        dh.apply_bcs(bcs_now, uf_k, pf_k, us_k, disp_k)

        coeffs = {f.name: f for f in [uf_k, pf_k, us_k, disp_k]}
        coeffs.update({f.name: f for f in [uf_n, pf_n, us_n, disp_n]})
        coeffs.update({"dt": dt})

        _, R_red = solver._assemble_system_reduced(coeffs, need_matrix=False)
        ndof_eff = getattr(solver.restrictor, "full_size", dh.total_dofs)
        R_full = np.zeros(int(ndof_eff), dtype=float)
        R_full[np.asarray(solver.active_dofs, dtype=int)] = np.asarray(R_red, dtype=float)
        print(f"[check] |R|_∞(exact) = {float(np.linalg.norm(R_full, ord=np.inf)):.3e}")
        for fld in dh.field_names:
            sl = dh.get_field_slice(fld)
            if sl is None or len(sl) == 0:
                continue
            print(f"[check] {fld:8s}: |R|_∞ = {float(np.linalg.norm(R_full[sl], ord=np.inf)):.3e}")
        return

    # ICs at t0
    t0 = 0.0
    gd = np.asarray(dh.get_field_slice("u_pos_x"), dtype=int)
    uf_n.set_nodal_values(gd, np.zeros_like(gd, dtype=float))
    uf_k.set_nodal_values(gd, np.zeros_like(gd, dtype=float))
    gd = np.asarray(dh.get_field_slice("u_pos_y"), dtype=int)
    xy = dh.get_dof_coords("u_pos_y")
    uf_n.set_nodal_values(gd, u_exact_y(xy[:, 0], t0))
    uf_k.set_nodal_values(gd, u_exact_y(xy[:, 0], t0))

    gd = np.asarray(dh.get_field_slice("vs_neg_x"), dtype=int)
    us_n.set_nodal_values(gd, np.zeros_like(gd, dtype=float))
    us_k.set_nodal_values(gd, np.zeros_like(gd, dtype=float))
    gd = np.asarray(dh.get_field_slice("vs_neg_y"), dtype=int)
    xy = dh.get_dof_coords("vs_neg_y")
    us_n.set_nodal_values(gd, u_exact_y(xy[:, 0], t0))
    us_k.set_nodal_values(gd, u_exact_y(xy[:, 0], t0))

    gd = np.asarray(dh.get_field_slice("d_neg_x"), dtype=int)
    disp_n.set_nodal_values(gd, np.zeros_like(gd, dtype=float))
    disp_k.set_nodal_values(gd, np.zeros_like(gd, dtype=float))
    gd = np.asarray(dh.get_field_slice("d_neg_y"), dtype=int)
    xy = dh.get_dof_coords("d_neg_y")
    disp_n.set_nodal_values(gd, w_exact(xy[:, 0], t0))
    disp_k.set_nodal_values(gd, w_exact(xy[:, 0], t0))

    pf_n.nodal_values.fill(0.0)
    pf_k.nodal_values.fill(0.0)

    state = SliverSweepState(step_idx=0, theta_targets=theta_targets, records=[])

    def _max_active_err(
        field: str,
        num_local: np.ndarray,
        exact_local: np.ndarray,
        *,
        physical_side: str | None = None,
    ) -> float:
        """
        Infinity-norm error on active DOFs.

        Note: In unfitted CG, some *active* DOFs in cut cells can be located on the
        opposite side of the interface (they belong to the polynomial extension).
        For MMS accuracy, the most meaningful metric is therefore the error on DOFs
        whose coordinates lie in the *physical* domain for that field (φ>0 for fluid,
        φ<0 for solid). Set `physical_side` to '+' or '-' to enable this filter.
        """
        inactive = set(getattr(dh, "dof_tags", {}).get("inactive", set()))
        gd = np.asarray(dh.get_field_slice(field), dtype=int)
        if gd.size == 0:
            return 0.0
        mask = np.array([int(d) not in inactive for d in gd], dtype=bool)
        if physical_side in ("+", "-"):
            xy = dh.get_dof_coords(field)
            phi = np.asarray(level_set(xy), dtype=float).ravel()
            if physical_side == "+":
                mask = mask & (phi > 0.0)
            else:
                mask = mask & (phi < 0.0)
        if not np.any(mask):
            return 0.0
        return float(np.max(np.abs(num_local[mask] - exact_local[mask])))

    def _max_active_err_info(
        field: str,
        num_local: np.ndarray,
        exact_local: np.ndarray,
        *,
        physical_side: str | None = None,
    ) -> Tuple[float, int, Tuple[float, float], float, float, float]:
        inactive = set(getattr(dh, "dof_tags", {}).get("inactive", set()))
        gd = np.asarray(dh.get_field_slice(field), dtype=int)
        if gd.size == 0:
            return 0.0, -1, (float("nan"), float("nan")), float("nan"), float("nan"), float("nan")
        mask = np.array([int(d) not in inactive for d in gd], dtype=bool)
        xy = dh.get_dof_coords(field)
        phi = np.asarray(level_set(xy), dtype=float).ravel()
        if physical_side in ("+", "-"):
            mask = mask & ((phi > 0.0) if physical_side == "+" else (phi < 0.0))
        if not np.any(mask):
            return 0.0, -1, (float("nan"), float("nan")), float("nan"), float("nan"), float("nan")
        err = np.abs(num_local - exact_local)
        err_masked = np.where(mask, err, -1.0)
        i_loc = int(np.argmax(err_masked))
        return (
            float(err[i_loc]),
            int(gd[i_loc]),
            (float(xy[i_loc, 0]), float(xy[i_loc, 1])),
            float(phi[i_loc]),
            float(num_local[i_loc]),
            float(exact_local[i_loc]),
        )

    def _record_step(t_curr: float) -> None:
        _refresh_theta_arrays()
        thp_raw, thn_raw, mn_p, mn_n = _theta_stats()

        # u_f_y
        xy = dh.get_dof_coords("u_pos_y")
        u_exact = u_exact_y(xy[:, 0], t_curr)
        e_u = _max_active_err("u_pos_y", uf_k.components[1].nodal_values, u_exact, physical_side="+")

        # u_s_y
        xy = dh.get_dof_coords("vs_neg_y")
        us_exact = u_exact_y(xy[:, 0], t_curr)
        e_us = _max_active_err("vs_neg_y", us_k.components[1].nodal_values, us_exact, physical_side="-")

        # d_s_y
        xy = dh.get_dof_coords("d_neg_y")
        d_exact = w_exact(xy[:, 0], t_curr)
        e_d = _max_active_err("d_neg_y", disp_k.components[1].nodal_values, d_exact, physical_side="-")

        rec = {
            "step": int(state.step_idx),
            "t": float(t_curr),
            "theta_target": float(state.theta_targets[state.step_idx]),
            "theta_min_neg": float(mn_n),
            "theta_min_pos": float(mn_p),
            "err_u_inf": float(e_u),
            "err_us_inf": float(e_us),
            "err_d_inf": float(e_d),
            "gamma_solid": float(getattr(gamma_sliver_mass_s, "value", gamma_sliver_mass_s)),
        }
        state.records.append(rec)
        if args.verbose:
            print(
                "[mms] step={step:02d} t={t:.3e} targetθ-={theta_target:.1e} minθ-={theta_min_neg:.2e} "
                "|e_u|∞(phys)={err_u_inf:.2e} |e_us|∞(phys)={err_us_inf:.2e} |e_d|∞(phys)={err_d_inf:.2e}".format(**rec)
            )
        if args.debug_max_error:
            eu, gd_u, xy_u, phi_u, u_num, u_ex = _max_active_err_info(
                "u_pos_y", uf_k.components[1].nodal_values, u_exact, physical_side="+"
            )
            eus, gd_us, xy_us, phi_us, us_num, us_ex = _max_active_err_info(
                "vs_neg_y", us_k.components[1].nodal_values, us_exact, physical_side="-"
            )
            ed, gd_d, xy_d, phi_d, d_num, d_ex = _max_active_err_info(
                "d_neg_y", disp_k.components[1].nodal_values, d_exact, physical_side="-"
            )
            print(
                f"[mms-debug] max e_u @ gdof={gd_u} x={xy_u[0]:+.3e} y={xy_u[1]:+.3e} phi={phi_u:+.3e}  "
                f"u={u_num:+.6e} u_ex={u_ex:+.6e} e={eu:.3e}"
            )
            print(
                f"[mms-debug] max e_us @ gdof={gd_us} x={xy_us[0]:+.3e} y={xy_us[1]:+.3e} phi={phi_us:+.3e}  "
                f"us={us_num:+.6e} us_ex={us_ex:+.6e} e={eus:.3e}"
            )
            print(
                f"[mms-debug] max e_d @ gdof={gd_d} x={xy_d[0]:+.3e} y={xy_d[1]:+.3e} phi={phi_d:+.3e}  "
                f"d={d_num:+.6e} d_ex={d_ex:+.6e} e={ed:.3e}"
            )

    def _set_exact_on_newly_active(t_curr: float) -> None:
        inactive = set(getattr(dh, "dof_tags", {}).get("inactive", set()))

        def _fill(field: str, fun, target_fun):
            gd = np.asarray(dh.get_field_slice(field), dtype=int)
            if gd.size == 0:
                return
            mask = np.array([int(d) not in inactive for d in gd], dtype=bool)
            if not np.any(mask):
                return
            xy = dh.get_dof_coords(field)
            vals = fun(xy[:, 0], t_curr) if callable(fun) else np.zeros_like(gd, dtype=float)
            # Update only active DOFs (leaving inactive stale is OK).
            target_fun.set_nodal_values(gd[mask], np.asarray(vals, dtype=float)[mask])

        _fill("u_pos_x", lambda x, t: 0.0 * x, uf_k)
        _fill("u_pos_y", u_exact_y, uf_k)
        _fill("vs_neg_x", lambda x, t: 0.0 * x, us_k)
        _fill("vs_neg_y", u_exact_y, us_k)
        _fill("d_neg_x", lambda x, t: 0.0 * x, disp_k)
        _fill("d_neg_y", w_exact, disp_k)
        pf_k.nodal_values.fill(0.0)

        # Keep prev vectors consistent (time loop already promoted current->prev)
        uf_n.nodal_values[:] = uf_k.nodal_values[:]
        pf_n.nodal_values[:] = pf_k.nodal_values[:]
        us_n.nodal_values[:] = us_k.nodal_values[:]
        disp_n.nodal_values[:] = disp_k.nodal_values[:]

    def _post_step_cb(_funcs):
        # End-of-step hook (called after a successful time step).
        dt_step = float(getattr(solver, "_current_dt", float(dt.value)))
        t0 = float(getattr(solver, "_current_t", 0.0))
        t_curr = t0 + dt_step

        _record_step(t_curr)
        state.step_idx += 1

        if state.step_idx >= len(state.theta_targets):
            return

        # Move interface for the next target θ-.
        next_info = _delta_for_target(state.theta_targets[state.step_idx])
        mn_pos, mn_neg = _refresh_theta_arrays()
        _update_auto_sliver_mass_gamma(theta_min_pos=mn_pos, theta_min_neg=mn_neg)
        retag_inactive(dh, mesh)

        # Update solver active DOFs for the new geometry.
        bcs_now = NewtonSolver._freeze_bcs(bcs, t_curr)
        active_changed = _recompute_active_dofs(solver, bcs_active=bcs_now if bcs_now is not None else bcs_homog)
        if active_changed and args.verbose:
            print(f"[sliver] active DOFs changed: n_active={len(solver.active_dofs)}")

        # Initialize newly-active DOFs to the exact solution to avoid time-derivative spikes.
        _set_exact_on_newly_active(t_curr)

        if args.verbose:
            print(
                "[sliver] next: target={theta_target:.1e}  minθ-={theta_min_neg:.3e}  minθ+={theta_min_pos:.3e}  δ={delta:.3e}".format(
                    **next_info
                )
            )

        solver.refresh_levelset_kernels(level_set)

    solver.post_timeloop_cb = _post_step_cb

    # ---------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------
    def _on_dt_change(dt_new: float) -> None:
        dt.value = float(dt_new)

    dt0 = float(dt.value)
    time_params = TimeStepperParameters(
        dt=dt0,
        max_steps=len(theta_targets),
        final_time=dt0 * len(theta_targets),
        theta=float(theta.value),
        stop_on_steady=False,
        on_dt_change=_on_dt_change,
    )

    try:
        solver.solve_time_interval(
            functions=[uf_k, pf_k, us_k, disp_k],
            prev_functions=[uf_n, pf_n, us_n, disp_n],
            aux_functions={"dt": dt},
            time_params=time_params,
        )
    except Exception as exc:  # noqa: BLE001
        print("\n[FAIL] solver raised:", repr(exc))
        if state.records:
            print("[FAIL] last completed step:", state.records[-1]["step"])
        raise
    finally:
        if state.records:
            print("\nSummary (per step):")
            print("step   t        targetθ-    minθ-      |e_u|∞(phys)  |e_us|∞(phys)  |e_d|∞(phys)")
            for r in state.records:
                print(
                    "{step:4d}  {t:7.3e}  {theta_target:9.1e}  {theta_min_neg:9.2e}  {err_u_inf:12.2e}  {err_us_inf:13.2e}  {err_d_inf:12.2e}".format(
                        **r
                    )
                )


if __name__ == "__main__":
    main()
