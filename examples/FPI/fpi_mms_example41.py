import argparse
import logging
import math
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

# Must be set before importing pyplot (headless-safe).
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pycutfem.core.geometry import hansbo_cut_ratio
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import (
    AffineLevelSet,
    MaxLevelSet,
    MinLevelSet,
    RotatedBoxLevelSet,
    ScaledLevelSet,
)
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    ElementWiseConstant,
    FacetNormal,
    Function,
    Pos,
    Neg,
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
from pycutfem.io.visualization import plot_mesh_2
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dx
from pycutfem.utils.fpi_fully_eulerian import build_fpi_eulerian_forms
from pycutfem.utils.fpi_mms_example41 import build_example41_mms
from pycutfem.utils.fsi_fully_eulerian import build_measures, make_domain_sets, refresh_sliver_weights
from pycutfem.utils.meshgen import structured_quad


def _tag_background_boundaries(mesh: Mesh, *, x0: float, x1: float, y0: float, y1: float, fluid_ls, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            # Γ_F,D: matching (mesh boundary) → Dirichlet.
            "inlet": lambda x, y: (abs(x - x0) <= tol) and (float(fluid_ls(np.array([x, y], dtype=float))) >= -1.0e-12),
            # fallback tags (not strictly needed for this example)
            "left": lambda x, y: abs(x - x0) <= tol,
            "right": lambda x, y: abs(x - x1) <= tol,
            "bottom": lambda x, y: abs(y - y0) <= tol,
            "top": lambda x, y: abs(y - y1) <= tol,
        }
    )


def _tag_paper_outer_boundaries(mesh: Mesh, *, cut_ls, tol: float = 1.0e-12) -> None:
    """Tag outer boundary edges of the rotated mesh, truncated by x>=x0.

    The physical domain is the rotated outer square with the left corner
    removed by the vertical cut x=x0. The cut itself is handled by `dInterface`
    (Neumann), while the remaining mesh boundary is Dirichlet.
    """
    mesh.tag_boundary_edges(
        {
            # Boundary-edge tags are midpoint-based: this is only used as a
            # convenience and can be replaced by DOF-based tagging via --inlet-bc dofs.
            "outer_dirichlet": lambda x, y: float(cut_ls(np.array([x, y], dtype=float))) <= tol,
            "outer_removed": lambda x, y: float(cut_ls(np.array([x, y], dtype=float))) > tol,
        }
    )


def _kinv_matrix(case: str, *, K: float) -> np.ndarray:
    case = str(case).strip().lower()
    if case in {"iso", "identity", "i"}:
        return (1.0 / float(K)) * np.eye(2, dtype=float)
    if case in {"aniso", "anisotropic", "a"}:
        base = np.array([[2.0, 0.3], [0.1, 1.5]], dtype=float)
        return (1.0 / float(K)) * base
    raise ValueError(f"Unknown K_inv case {case!r}")


def _build_problem(
    *,
    nx: int,
    poly_order: int,
    qdeg: int,
    dt_val: float,
    kinv_case: str,
    mesh_layout: str = "paper",
    ghost: str = "edge",
    ghost_weights: str = "none",
    cut_drop_fluid: float = 0.0,
    cut_drop_poro: float = 0.0,
):
    mesh_layout = str(mesh_layout).strip().lower()
    if mesh_layout not in {"legacy", "paper"}:
        raise ValueError("mesh_layout must be one of {'legacy','paper'}")

    # Example 4.1 geometry parameters (paper sec. 4.1)
    x0 = -0.45  # vertical truncation position (Δx in the paper)
    poro_ls = RotatedBoxLevelSet(center=(0.0, 0.0), hx=0.25, hy=0.25, angle=math.pi / 6.0)  # Ω^P (rotated 30°)
    fluid_sq_ls = RotatedBoxLevelSet(center=(0.0, 0.0), hx=0.5, hy=0.5, angle=math.pi / 4.0)  # outer square (rotated 45°)

    if mesh_layout == "legacy":
        # Legacy background mesh: axis-aligned square that *contains* the rotated outer square.
        y0 = -0.75
        L = 1.5
        x1 = x0 + L
        y1 = y0 + L

        nodes, elems, edges, corners = structured_quad(L, L, nx=nx, ny=nx, poly_order=poly_order, offset=(x0, y0))
        mesh = Mesh(
            nodes=nodes,
            element_connectivity=elems,
            edges_connectivity=edges,
            elements_corner_nodes=corners,
            element_type="quad",
            poly_order=poly_order,
        )

        cut_ls = AffineLevelSet(-1.0, 0.0, x0)  # φ = x0 - x  (negative for x > x0)

        # Ω^B = outer square ∩ {x >= x0} (negative inside) → flip sign so Ω^B is positive.
        outer_std = MaxLevelSet(fluid_sq_ls, cut_ls)
        outer_pos = ScaledLevelSet(-1.0, outer_std)

        # Fluid domain Ω^F = Ω^B \\ Ω^P : positive where (outside poro) ∩ (inside Ω^B)
        fluid_ls = MinLevelSet(poro_ls, outer_pos)

        # Dirichlet on the (matching) cut boundary x=x0; Neumann traction on the outer square.
        _tag_background_boundaries(mesh, x0=x0, x1=x1, y0=y0, y1=y1, fluid_ls=fluid_ls)
        dirichlet_edge_tag = "inlet"
        dirichlet_dof_tag = "inlet_dofs"
        neumann_ls = fluid_sq_ls
        neumann_name = "outer"
    else:
        # Paper mesh layout (Fig. 3): rotated elements on the outer square (size 1, rotated by 45°).
        # The left corner is removed by the vertical cut x=x0.
        L = 1.0
        x1 = float("nan")
        y0 = float("nan")
        y1 = float("nan")

        nodes, elems, edges, corners = structured_quad(
            L,
            L,
            nx=nx,
            ny=nx,
            poly_order=poly_order,
            offset=(-0.5, -0.5),
            rotation=math.pi / 4.0,
            rotation_center=(0.0, 0.0),
        )
        mesh = Mesh(
            nodes=nodes,
            element_connectivity=elems,
            edges_connectivity=edges,
            elements_corner_nodes=corners,
            element_type="quad",
            poly_order=poly_order,
        )

        # Vertical truncation: `cut_ls` is negative on the physical side {x >= x0}.
        cut_ls = AffineLevelSet(-1.0, 0.0, x0)
        cut_pos = ScaledLevelSet(-1.0, cut_ls)  # positive on {x >= x0}

        # Fluid domain Ω^F = {x>=x0} \\ Ω^P (outer boundary is mesh boundary; avoid extra cut cells there).
        outer_pos = None
        fluid_ls = MinLevelSet(poro_ls, cut_pos)

        _tag_paper_outer_boundaries(mesh, cut_ls=cut_ls)
        dirichlet_edge_tag = "outer_dirichlet"
        dirichlet_dof_tag = "outer_dofs"
        neumann_ls = cut_ls
        neumann_name = "inlet"

    ghost = str(ghost).strip().lower()
    if ghost not in {"edge", "patch"}:
        raise ValueError("ghost must be 'edge' or 'patch'.")
    use_patch_ghost = ghost == "patch"

    cut_drop_fluid = float(cut_drop_fluid)
    cut_drop_poro = float(cut_drop_poro)

    # --- Build fluid-domain measures (Ω^F) ---
    mesh.classify_elements(fluid_ls)
    mesh.classify_edges(fluid_ls)
    domains_fluid = make_domain_sets(mesh, use_aligned_interface=False)

    # ------------------------------------------------------------------
    # Optional Hansbo cut-ratio weights for ghost penalties
    # ------------------------------------------------------------------
    ghost_weights = str(ghost_weights).strip().lower()
    if ghost_weights not in {"none", "hansbo"}:
        raise ValueError("ghost_weights must be one of {'none','hansbo'}")

    sliver_theta0 = float(os.getenv("SLIVER_THETA0", "0.05"))
    sliver_p = float(os.getenv("SLIVER_P", "1.0"))
    sliver_wmax = float(os.getenv("SLIVER_WMAX", "1000.0"))
    sliver_thetamin = float(os.getenv("SLIVER_THETAMIN", "1e-6"))
    sliver_theta_floor = max(sliver_thetamin, 1.0e-12)

    theta_pos_raw_f = hansbo_cut_ratio(mesh, fluid_ls, side="+")
    theta_neg_raw_f = hansbo_cut_ratio(mesh, fluid_ls, side="-")
    theta_pos_f = np.clip(theta_pos_raw_f, sliver_theta_floor, 1.0)
    theta_neg_f = np.clip(theta_neg_raw_f, sliver_theta_floor, 1.0)
    w_sliver_fluid = None
    if ghost_weights == "hansbo":
        w_pos_f = np.ones_like(theta_pos_f)
        w_neg_f = np.ones_like(theta_neg_f)
        refresh_sliver_weights(
            mesh,
            theta_pos_f,
            theta_neg_f,
            w_pos_f,
            w_neg_f,
            theta0=sliver_theta0,
            p=sliver_p,
            wmax=sliver_wmax,
            thetamin=sliver_thetamin,
            smooth=1.0,
        )
        w_sliver_fluid = ElementWiseConstant(w_pos_f)
    bad_fluid_mask = None
    if cut_drop_fluid > 0.0:
        cut_mask = np.asarray(mesh.element_bitset("cut").mask, dtype=bool)
        theta = hansbo_cut_ratio(mesh, fluid_ls, side="+")
        bad_fluid_mask = cut_mask & (theta < cut_drop_fluid)
    dx_f, _dx_inactive, _dGamma_f, dG_f, _dG_inactive = build_measures(
        mesh, fluid_ls, domains_fluid, qvol=qdeg, use_facet_patch_ghost=use_patch_ghost
    )

    me = MixedElement(
        mesh,
        field_specs={
            "v_pos_x": poly_order,
            "v_pos_y": poly_order,
            "p_pos_": poly_order,
            "v_neg_x": poly_order,
            "v_neg_y": poly_order,
            "u_neg_x": poly_order,
            "u_neg_y": poly_order,
            "p_neg_": poly_order,
        },
    )
    dh = DofHandler(me, method="cg")

    # Tag inactive DOFs to avoid singular rows/cols (fluid outside Ω^F, poro outside Ω^P).
    dh.dof_tags["inactive"] = set()
    for fld in ("v_pos_x", "v_pos_y", "p_pos_"):
        dh.tag_dofs_from_element_bitset("inactive", fld, "inside", strict=True)
    if bad_fluid_mask is not None and np.any(bad_fluid_mask):
        for fld in ("v_pos_x", "v_pos_y", "p_pos_"):
            dh.tag_dofs_from_element_bitset("inactive", fld, bad_fluid_mask, strict=False)

    # --- Build poro-domain + interface measures (Ω^P and Γ^{FP}) ---
    mesh.classify_elements(poro_ls)
    mesh.classify_edges(poro_ls)
    mesh.build_interface_segments(poro_ls)
    domains_poro = make_domain_sets(mesh, use_aligned_interface=False)

    theta_pos_raw_p = hansbo_cut_ratio(mesh, poro_ls, side="+")
    theta_neg_raw_p = hansbo_cut_ratio(mesh, poro_ls, side="-")
    theta_pos_p = np.clip(theta_pos_raw_p, sliver_theta_floor, 1.0)
    theta_neg_p = np.clip(theta_neg_raw_p, sliver_theta_floor, 1.0)
    w_sliver_poro = None
    if ghost_weights == "hansbo":
        w_pos_p = np.ones_like(theta_pos_p)
        w_neg_p = np.ones_like(theta_neg_p)
        refresh_sliver_weights(
            mesh,
            theta_pos_p,
            theta_neg_p,
            w_pos_p,
            w_neg_p,
            theta0=sliver_theta0,
            p=sliver_p,
            wmax=sliver_wmax,
            thetamin=sliver_thetamin,
            smooth=1.0,
        )
        # Poro domain is on the negative side of `poro_ls`.
        w_sliver_poro = ElementWiseConstant(w_neg_p)
    bad_poro_mask = None
    if cut_drop_poro > 0.0:
        cut_mask = np.asarray(mesh.element_bitset("cut").mask, dtype=bool)
        theta = hansbo_cut_ratio(mesh, poro_ls, side="-")
        bad_poro_mask = cut_mask & (theta < cut_drop_poro)
    _dx_out, dx_p, dGamma, _dG_out, dG_p = build_measures(
        mesh, poro_ls, domains_poro, qvol=qdeg, use_facet_patch_ghost=use_patch_ghost
    )

    # Inactive DOFs for poro fields: elements outside Ω^P.
    for fld in ("v_neg_x", "v_neg_y", "u_neg_x", "u_neg_y", "p_neg_"):
        dh.tag_dofs_from_element_bitset("inactive", fld, "outside", strict=True)
    if bad_poro_mask is not None and np.any(bad_poro_mask):
        for fld in ("v_neg_x", "v_neg_y", "u_neg_x", "u_neg_y", "p_neg_"):
            dh.tag_dofs_from_element_bitset("inactive", fld, bad_poro_mask, strict=False)

    # Interface mesh size h_Γ (paper: |K|/|Γ∩K| on cut cells of Γ^{FP}).
    areas = np.asarray(getattr(mesh, "areas_list", None), dtype=float)
    if areas.size != len(mesh.elements_list):
        areas = np.asarray(mesh.areas(), dtype=float)
    h_gamma_vals = np.sqrt(np.maximum(areas, 0.0))
    cut_ids = mesh.element_bitset("cut").to_indices()
    for eid in cut_ids.tolist():
        segs = getattr(mesh.elements_list[int(eid)], "interface_segments", []) or []
        seg_len = 0.0
        for seg in segs:
            p0 = np.asarray(seg[0], dtype=float)
            p1 = np.asarray(seg[1], dtype=float)
            seg_len += float(np.linalg.norm(p1 - p0))
        if seg_len > 1.0e-14:
            h_gamma_vals[int(eid)] = float(areas[int(eid)]) / seg_len
    h_gamma = ElementWiseConstant(h_gamma_vals)

    # --- Neumann traction boundary measure (non-matching dInterface) ---
    # For the legacy layout: traction is imposed on the rotated outer-square boundary.
    # For the paper layout: traction is imposed on the vertical cut x=x0 ("inlet").
    mesh.classify_elements(neumann_ls)
    mesh.classify_edges(neumann_ls)
    mesh.build_interface_segments(neumann_ls)
    domains_neumann = make_domain_sets(mesh, use_aligned_interface=False)
    _dx_out_b, _dx_in_b, dGamma_neumann, _dG_out_b, _dG_in_b = build_measures(mesh, neumann_ls, domains_neumann, qvol=qdeg)

    # Tag inlet DOFs directly (robust even when the Γ_F,D segment does not align
    # with full boundary edges on coarse meshes). We constrain only DOFs whose
    # coordinates lie on x=x0 *and* inside the rotated outer square.
    def _inlet_dof_locator(x: float, y: float) -> bool:
        if abs(float(x) - float(x0)) > 1.0e-10:
            return False
        try:
            phi = float(fluid_sq_ls(np.array([float(x), float(y)], dtype=float)))
        except Exception:
            phi = float(fluid_sq_ls([float(x), float(y)]))
        return phi <= 1.0e-12

    if mesh_layout == "legacy":
        dh.tag_dofs_by_locator_map({dirichlet_dof_tag: _inlet_dof_locator}, fields=["v_pos_x", "v_pos_y"])
    else:
        # Tag Dirichlet DOFs on the *rotated* outer boundary, excluding the removed
        # corner (x < x0). For the unit outer square rotated by 45°, the boundary is
        # |x|+|y|=sqrt(2)/2.
        r = math.sqrt(2.0) / 2.0

        def _outer_dof_locator(x: float, y: float) -> bool:
            x = float(x)
            y = float(y)
            if x < float(x0) - 1.0e-12:
                return False
            return abs(abs(x) + abs(y) - r) <= 5.0e-10

        dh.tag_dofs_by_locator_map({dirichlet_dof_tag: _outer_dof_locator}, fields=["v_pos_x", "v_pos_y"])

    Vf = FunctionSpace(name="Vf", field_names=["v_pos_x", "v_pos_y"], dim=1, side="+")
    Vp = FunctionSpace(name="Vp", field_names=["v_neg_x", "v_neg_y"], dim=1, side="-")
    Up = FunctionSpace(name="Up", field_names=["u_neg_x", "u_neg_y"], dim=1, side="-")

    dvF = VectorTrialFunction(space=Vf, dof_handler=dh)
    dpF = TrialFunction(name="dpF", field_name="p_pos_", dof_handler=dh, side="+")
    dvP = VectorTrialFunction(space=Vp, dof_handler=dh)
    duP = VectorTrialFunction(space=Up, dof_handler=dh)
    dpP = TrialFunction(name="dpP", field_name="p_neg_", dof_handler=dh, side="-")

    vF_test = VectorTestFunction(space=Vf, dof_handler=dh)
    qF_test = TestFunction(name="qF", field_name="p_pos_", dof_handler=dh, side="+")
    vP_test = VectorTestFunction(space=Vp, dof_handler=dh)
    uP_test = VectorTestFunction(space=Up, dof_handler=dh)
    qP_test = TestFunction(name="qP", field_name="p_neg_", dof_handler=dh, side="-")

    vF_k = VectorFunction(name="vF_k", field_names=["v_pos_x", "v_pos_y"], dof_handler=dh, side="+")
    pF_k = Function(name="pF_k", field_name="p_pos_", dof_handler=dh, side="+")
    vP_k = VectorFunction(name="vP_k", field_names=["v_neg_x", "v_neg_y"], dof_handler=dh, side="-")
    uP_k = VectorFunction(name="uP_k", field_names=["u_neg_x", "u_neg_y"], dof_handler=dh, side="-")
    uP_nm1 = VectorFunction(name="uP_nm1", field_names=["u_neg_x", "u_neg_y"], dof_handler=dh, side="-")
    pP_k = Function(name="pP_k", field_name="p_neg_", dof_handler=dh, side="-")

    vF_n = VectorFunction(name="vF_n", field_names=["v_pos_x", "v_pos_y"], dof_handler=dh, side="+")
    pF_n = Function(name="pF_n", field_name="p_pos_", dof_handler=dh, side="+")
    vP_n = VectorFunction(name="vP_n", field_names=["v_neg_x", "v_neg_y"], dof_handler=dh, side="-")
    uP_n = VectorFunction(name="uP_n", field_names=["u_neg_x", "u_neg_y"], dof_handler=dh, side="-")
    pP_n = Function(name="pP_n", field_name="p_neg_", dof_handler=dh, side="-")

    for f in (vF_k, pF_k, vP_k, uP_k, uP_nm1, pP_k, vF_n, pF_n, vP_n, uP_n, pP_n):
        f.nodal_values.fill(0.0)

    has_fluid = domains_fluid["has_pos"]
    has_poro = domains_poro["has_neg"]

    return dict(
        mesh_layout=mesh_layout,
        mesh=mesh,
        poro_ls=poro_ls,
        fluid_ls=fluid_ls,
        fluid_sq_ls=fluid_sq_ls,
        outer_ls=outer_pos,
        dirichlet_edge_tag=dirichlet_edge_tag,
        dirichlet_dof_tag=dirichlet_dof_tag,
        neumann_name=neumann_name,
        neumann_ls=neumann_ls,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        domains_fluid=domains_fluid,
        domains_poro=domains_poro,
        domains_neumann=domains_neumann,
        dh=dh,
        me=me,
        dx_f=dx_f,
        dx_p=dx_p,
        dGamma=dGamma,
        dG_f=dG_f,
        dG_p=dG_p,
        dGamma_neumann=dGamma_neumann,
        h_gamma=h_gamma,
        w_sliver_fluid=w_sliver_fluid,
        w_sliver_poro=w_sliver_poro,
        h=L / float(nx),
        ghost=ghost,
        vF_k=vF_k,
        pF_k=pF_k,
        vP_k=vP_k,
        uP_k=uP_k,
        uP_nm1=uP_nm1,
        pP_k=pP_k,
        vF_n=vF_n,
        pF_n=pF_n,
        vP_n=vP_n,
        uP_n=uP_n,
        pP_n=pP_n,
        vF_kR=restrict(vF_k, has_fluid),
        pF_kR=restrict(pF_k, has_fluid),
        vF_nR=restrict(vF_n, has_fluid),
        pF_nR=restrict(pF_n, has_fluid),
        vP_kR=restrict(vP_k, has_poro),
        uP_kR=restrict(uP_k, has_poro),
        uP_nm1R=restrict(uP_nm1, has_poro),
        pP_kR=restrict(pP_k, has_poro),
        vP_nR=restrict(vP_n, has_poro),
        uP_nR=restrict(uP_n, has_poro),
        pP_nR=restrict(pP_n, has_poro),
        dvF_R=restrict(dvF, has_fluid),
        dpF_R=restrict(dpF, has_fluid),
        dvP_R=restrict(dvP, has_poro),
        duP_R=restrict(duP, has_poro),
        dpP_R=restrict(dpP, has_poro),
        vF_test=vF_test,
        vF_testR=restrict(vF_test, has_fluid),
        qF_testR=restrict(qF_test, has_fluid),
        vP_testR=restrict(vP_test, has_poro),
        uP_testR=restrict(uP_test, has_poro),
        qP_testR=restrict(qP_test, has_poro),
        dt=Constant(float(dt_val)),
        kinv_case=kinv_case,
    )


def _print_bc_diagnostics(*, prob: dict, inlet_bc: str) -> None:
    mesh: Mesh = prob["mesh"]
    dh: DofHandler = prob["dh"]
    mesh_layout = str(prob.get("mesh_layout", "legacy")).strip().lower()

    inlet_bc = str(inlet_bc).strip().lower()
    if inlet_bc not in {"dofs", "edges", "auto"}:
        raise ValueError("inlet_bc must be one of {'dofs','edges','auto'}")

    edge_tag = str(prob.get("dirichlet_edge_tag", "inlet"))
    dof_tag = str(prob.get("dirichlet_dof_tag", "inlet_dofs"))

    if inlet_bc == "edges":
        inlet_tag = edge_tag
    elif inlet_bc == "auto":
        try:
            inlet_tag = edge_tag if mesh.edge_bitset(edge_tag).cardinality() > 0 else dof_tag
        except Exception:
            inlet_tag = dof_tag
    else:
        inlet_tag = dof_tag

    bcs = [
        BoundaryCondition("v_pos_x", "dirichlet", inlet_tag, 0.0),
        BoundaryCondition("v_pos_y", "dirichlet", inlet_tag, 0.0),
    ]
    bc_data = dh.get_dirichlet_data(bcs)
    bc_dofs = set(int(k) for k in bc_data.keys())
    inactive = set(getattr(dh, "dof_tags", {}).get("inactive", set()))
    inlet_dof_tag = set(getattr(dh, "dof_tags", {}).get(dof_tag, set()))

    inlet_edges = float("nan")
    try:
        inlet_edges = float(mesh.edge_bitset(edge_tag).cardinality())
    except Exception:
        inlet_edges = float("nan")
    removed_edges = float("nan")
    try:
        removed_edges = float(mesh.edge_bitset("outer_removed").cardinality())
    except Exception:
        removed_edges = float("nan")

    bc_arr = np.fromiter(bc_dofs, dtype=int, count=len(bc_dofs)) if bc_dofs else np.empty((0,), dtype=int)
    inlet_arr = (
        np.fromiter(inlet_dof_tag, dtype=int, count=len(inlet_dof_tag)) if inlet_dof_tag else np.empty((0,), dtype=int)
    )

    x0 = float(prob.get("x0", float("nan")))
    print(f"BC diagnostics: layout={mesh_layout}  inlet_bc={inlet_bc}  dirichlet_tag={inlet_tag}")
    print(f"  edges(tagged)={inlet_edges:.0f}  outer_removed={removed_edges:.0f}")

    max_dx = 0.0
    for field in ("v_pos_x", "v_pos_y"):
        ids = np.asarray(dh.get_field_slice(field), dtype=int)
        coords = dh.get_dof_coords(field)

        tag_mask = np.isin(ids, inlet_arr) if inlet_arr.size else np.zeros(ids.shape, dtype=bool)
        bc_mask = np.isin(ids, bc_arr) if bc_arr.size else np.zeros(ids.shape, dtype=bool)
        if mesh_layout == "legacy" and np.isfinite(x0):
            geom_mask = np.abs(coords[:, 0] - x0) <= 1.0e-10
        else:
            r = math.sqrt(2.0) / 2.0
            geom_mask = np.abs(np.abs(coords[:, 0]) + np.abs(coords[:, 1]) - r) <= 5.0e-10

        n_tag = int(np.count_nonzero(tag_mask))
        n_bc = int(np.count_nonzero(bc_mask))
        n_geom = int(np.count_nonzero(geom_mask))
        if n_bc:
            bc_ids = set(int(v) for v in ids[bc_mask].tolist())
            n_bc_active = int(len(bc_ids - inactive))
            ys = np.asarray(coords[bc_mask, 1], dtype=float)
            yr_s = f"[{float(ys.min()):+.3e}, {float(ys.max()):+.3e}]"
            if np.isfinite(x0):
                xs = np.asarray(coords[bc_mask, 0], dtype=float)
                if mesh_layout == "legacy":
                    max_dx = max(max_dx, float(np.max(np.abs(xs - x0))))
        else:
            n_bc_active = 0
            yr_s = "n/a"
        geom_lbl = "dofs_on_x0" if mesh_layout == "legacy" else "dofs_on_outer"
        print(f"  {field}: {geom_lbl}={n_geom:4d}  tag={n_tag:4d}  dirichlet(bc)={n_bc:4d}  active={n_bc_active:4d}  y_range={yr_s}")

    if bc_dofs and np.isfinite(x0) and mesh_layout == "legacy":
        print(f"  max|x_dof - x0| over Dirichlet dofs = {max_dx:.3e} (x0={x0:+.3e})")


def _plot_example41_mesh(*, prob: dict, inlet_bc: str, out_path: Path) -> None:
    mesh: Mesh = prob["mesh"]
    dh: DofHandler = prob["dh"]
    mesh_layout = str(prob.get("mesh_layout", "legacy")).strip().lower()

    # Important: `_build_problem(...)` re-classifies the mesh multiple times
    # (fluid_ls, poro_ls, neumann_ls). For mesh diagnostics we want to see the
    # *fluid domain* classification (multi-level-set fluid_ls), so ensure the
    # tags and interface segments shown in this plot correspond to `fluid_ls`.
    try:
        fluid_ls = prob.get("fluid_ls", None)
        if fluid_ls is not None:
            mesh.classify_elements(fluid_ls)
            mesh.classify_edges(fluid_ls)
            mesh.build_interface_segments(fluid_ls)
    except Exception:
        pass

    inlet_bc = str(inlet_bc).strip().lower()
    if inlet_bc not in {"dofs", "edges", "auto"}:
        raise ValueError("inlet_bc must be one of {'dofs','edges','auto'}")

    edge_tag = str(prob.get("dirichlet_edge_tag", "inlet"))
    dof_tag = str(prob.get("dirichlet_dof_tag", "inlet_dofs"))

    if inlet_bc == "edges":
        inlet_tag = edge_tag
    elif inlet_bc == "auto":
        try:
            inlet_tag = edge_tag if mesh.edge_bitset(edge_tag).cardinality() > 0 else dof_tag
        except Exception:
            inlet_tag = dof_tag
    else:
        inlet_tag = dof_tag

    x0 = float(prob["x0"])
    y_max = (math.sqrt(2.0) / 2.0) + x0
    y_min_mesh = float(np.min(mesh.nodes_x_y_pos[:, 1]))
    y_max_mesh = float(np.max(mesh.nodes_x_y_pos[:, 1]))

    fig, ax = plt.subplots(figsize=(10, 9))
    # Background mesh: show element/edge classifications.
    plot_mesh_2(
        mesh,
        ax=ax,
        show=False,
        elem_tags=True,
        plot_interface=True,
        edge_colors=True,
        plot_nodes=True,
        fluid_solid_overlay=False,
    )
    # Keep plot_mesh_2 legend entries (elem/edge tags) and extend it with the
    # Dirichlet-DOF markers below.
    base_leg = ax.get_legend()
    base_handles: list[object] = []
    base_labels: list[str] = []
    if base_leg is not None:
        base_handles = list(getattr(base_leg, "legend_handles", [])) or list(getattr(base_leg, "legendHandles", []))
        base_labels = [str(t.get_text()) for t in base_leg.get_texts()]
        base_leg.remove()

    # Draw the rotated boxes *exactly* (avoid tricontour artifacts when φ=0 at nodes).
    def _plot_rotated_box(ls, *, color: str, lw: float, zorder: int) -> None:
        if ls is None:
            return
        try:
            cx, cy = float(ls.center[0]), float(ls.center[1])
            hx, hy = float(ls.hx), float(ls.hy)
            ang = float(ls.angle)
        except Exception:
            return
        ca, sa = math.cos(ang), math.sin(ang)
        R = np.array([[ca, -sa], [sa, ca]], dtype=float)
        corners = np.array(
            [
                [-hx, -hy],
                [hx, -hy],
                [hx, hy],
                [-hx, hy],
                [-hx, -hy],
            ],
            dtype=float,
        )
        pts = np.array([cx, cy], dtype=float) + corners @ R.T
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=float(lw), zorder=int(zorder))

    _plot_rotated_box(prob.get("fluid_sq_ls"), color="green", lw=2.5, zorder=6)
    _plot_rotated_box(prob.get("poro_ls"), color="purple", lw=2.5, zorder=6)

    # Highlight the vertical cut line and the (geometric) Γ^{F,N} segment on it.
    ax.plot([x0, x0], [y_min_mesh, y_max_mesh], linestyle="--", color="gray", linewidth=1.5, zorder=3)
    ax.plot([x0, x0], [-y_max, y_max], linestyle="-", color="red", linewidth=4.0, zorder=7)
    ax.plot([x0, x0], [-y_max, y_max], "o", color="red", markersize=5.5, markeredgecolor="white", zorder=8)

    # Highlight relevant tagged boundary edges for context.
    if mesh_layout == "legacy":
        tags = (("left", "dodgerblue", 3.0), ("inlet", "crimson", 3.0))
    else:
        tags = (("outer_dirichlet", "crimson", 3.0), ("outer_removed", "gray", 2.5))
    for tag, col, lw in tags:
        try:
            mask = np.asarray(mesh.edge_bitset(tag).mask, dtype=bool)
        except Exception:
            continue
        ids = np.flatnonzero(mask)
        for gid in ids.tolist():
            e = mesh.edges_list[int(gid)]
            nids = list(e.all_nodes) if getattr(e, "all_nodes", ()) else list(e.nodes)
            pts = mesh.nodes_x_y_pos[nids]
            ax.plot(pts[:, 0], pts[:, 1], "-", color=col, linewidth=lw, zorder=5)

    # Plot DOFs on x=x0 (legacy) and the ones constrained by inlet_tag.
    bcs = [
        BoundaryCondition("v_pos_x", "dirichlet", inlet_tag, 0.0),
        BoundaryCondition("v_pos_y", "dirichlet", inlet_tag, 0.0),
    ]
    bc_data = dh.get_dirichlet_data(bcs)
    bc_dofs = set(int(k) for k in bc_data.keys())
    inactive = set(getattr(dh, "dof_tags", {}).get("inactive", set()))

    fld = "v_pos_x"  # same coordinates as v_pos_y
    f_ids = np.asarray(dh.get_field_slice(fld), dtype=int)
    f_coords = dh.get_dof_coords(fld)
    x0_mask = np.abs(f_coords[:, 0] - x0) <= 1.0e-10
    bc_mask = np.isin(f_ids, np.fromiter(bc_dofs, dtype=int, count=len(bc_dofs))) if bc_dofs else np.zeros(f_ids.shape, bool)
    inact_mask = np.isin(f_ids, np.fromiter(inactive, dtype=int, count=len(inactive))) if inactive else np.zeros(f_ids.shape, bool)

    if mesh_layout == "legacy" and np.any(x0_mask):
        ax.plot(
            f_coords[x0_mask & ~inact_mask, 0],
            f_coords[x0_mask & ~inact_mask, 1],
            "o",
            color="black",
            markersize=6.0,
            zorder=9,
            label="Active DOFs on x=x0",
        )
        if np.any(x0_mask & inact_mask):
            ax.plot(
                f_coords[x0_mask & inact_mask, 0],
                f_coords[x0_mask & inact_mask, 1],
                "x",
                color="gray",
                markersize=7.0,
                zorder=9,
                label="Inactive DOFs on x=x0",
            )
    if np.any(bc_mask):
        (h_dirichlet,) = ax.plot(
            f_coords[bc_mask, 0],
            f_coords[bc_mask, 1],
            "o",
            color="gold",
            markersize=10.0,
            markeredgecolor="black",
            zorder=10,
            label="Dirichlet DOFs",
        )
    else:
        h_dirichlet = None

    L = 1.5 if mesh_layout == "legacy" else 1.0
    nx_guess = int(round(float(L) / float(prob["h"]))) if float(prob.get("h", 0.0) or 0.0) > 0.0 else -1
    ax.set_title(f"Example 4.1 mesh (layout={mesh_layout}, nx={nx_guess}, p={mesh.poly_order}) | dirichlet_tag={inlet_tag}")
    # Merge legends: plot_mesh_2 (elem/edge tags) + Dirichlet DOFs
    handles: list[object] = []
    labels: list[str] = []

    def _add(h, lbl: str) -> None:
        if h is None:
            return
        lbl = str(lbl)
        if not lbl or lbl.startswith("_"):
            return
        if lbl in labels:
            return
        handles.append(h)
        labels.append(lbl)

    for h, lbl in zip(base_handles, base_labels):
        _add(h, lbl)
    _add(h_dirichlet, "Dirichlet DOFs")

    if handles:
        ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _approx_vinf(mms, *, bbox: tuple[float, float, float, float], n: int = 41) -> float:
    x0, x1, y0, y1 = (float(v) for v in bbox)
    xs = np.linspace(x0, x1, int(n))
    ys = np.linspace(y0, y1, int(n))
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    v = np.asarray(mms.vF_k(X, Y))
    return float(np.max(np.abs(v)))


def _interface_errors(
    *,
    dh: DofHandler,
    dGamma,
    vF_k,
    vP_k,
    uP_k,
    uP_n,
    dt: Constant,
    porosity: Constant,
    beta_BJ: Constant,
    vF_A,
    vP_A,
    uP_Ak,
    uP_An,
    quad_order: int,
    backend: str,
) -> dict[str, float]:
    n = FacetNormal()

    u_dot = (uP_k - uP_n) / dt
    u_dot_A = (uP_Ak - uP_An) / dt

    # Kinematic parts of (16) and (17)
    kin_n_num = Pos(vF_k) - Neg(u_dot) - porosity * (Neg(vP_k) - Neg(u_dot))
    kin_t_num = Pos(vF_k) - Neg(u_dot) - porosity * beta_BJ * (Neg(vP_k) - Neg(u_dot))

    kin_n_A = vF_A - u_dot_A - porosity * (vP_A - u_dot_A)
    kin_t_A = vF_A - u_dot_A - porosity * beta_BJ * (vP_A - u_dot_A)

    diff_n = kin_n_num - kin_n_A
    diff_t = kin_t_num - kin_t_A

    proj_n = dot(diff_n, n) * n
    proj_t = diff_t - dot(diff_t, n) * n

    En2 = inner(proj_n, proj_n)
    Et2 = inner(proj_t, proj_t)

    hooks = {En2: {"name": "En2"}, Et2: {"name": "Et2"}}
    scalars = assemble_form(
        Equation(None, En2 * dGamma + Et2 * dGamma),
        dof_handler=dh,
        bcs=[],
        quad_order=int(quad_order),
        assembler_hooks=hooks,
        backend=backend,
    )
    en2 = float(np.asarray(scalars["En2"]).reshape(()))
    et2 = float(np.asarray(scalars["Et2"]).reshape(()))
    en = math.sqrt(en2)
    et = math.sqrt(et2)
    return {"En": en, "Et": et}


def _run_one(
    *,
    nx: int,
    poly_order: int,
    qdeg: int,
    qerr: int | None,
    dt_val: float,
    t_end: float,
    backend: str,
    kinv_case: str,
    interface: str,
    mesh_layout: str,
    gamma_inv: float,
    zeta: float,
    init_guess: str,
    ghost: str,
    ghost_weights: str,
    cut_drop_fluid: float,
    cut_drop_poro: float,
    sliver_mass_fluid: float = 0.0,
    sliver_mass_poro: float = 0.0,
    sliver_mass_skeleton: float = 0.0,
    sliver_theta_eps: float = 1.0e-16,
    ghost_mass_fluid: float = 0.0,
    ghost_mass_poro: float = 0.0,
    inlet_bc: str = "dofs",
    show_cut_ratios: bool = False,
    check_exact_residual: bool = False,
    use_interface_terms: bool = True,
    use_stabilization: bool = True,
):
    prob = _build_problem(
        nx=nx,
        poly_order=poly_order,
        qdeg=qdeg,
        dt_val=dt_val,
        kinv_case=kinv_case,
        mesh_layout=mesh_layout,
        ghost=ghost,
        ghost_weights=ghost_weights,
        cut_drop_fluid=cut_drop_fluid,
        cut_drop_poro=cut_drop_poro,
    )

    interface = str(interface).strip().lower()
    if interface == "bj":
        beta_bj_val = 1.0
    elif interface == "bjs":
        beta_bj_val = 0.0
    else:
        raise ValueError(f"Unknown interface variant {interface!r}; expected 'bj' or 'bjs'.")

    t_prev = float(t_end) - float(dt_val)
    if t_prev < 0.0:
        t_prev = 0.0
    poro_ls = prob["poro_ls"]
    if isinstance(poro_ls, RotatedBoxLevelSet):
        interface_name = "rotated_box"
        interface_params = (
            float(poro_ls.center[0]),
            float(poro_ls.center[1]),
            float(poro_ls.hx),
            float(poro_ls.hy),
            float(poro_ls.angle),
        )
    else:
        interface_name = "vertical"
        interface_params = ()

    mms = build_example41_mms(
        dt_val=dt_val,
        kinv_case=kinv_case,
        t_prev=t_prev,
        beta_BJ=beta_bj_val,
        interface=interface_name,
        interface_params=interface_params,
    )
    ana_deg = max(10, qdeg + 4)

    # Parameters (match MMS builder)
    rho_f = Constant(1.0)
    mu_f = Constant(1.0)
    rho_s0_tilde = Constant(1.0)
    porosity = Constant(0.5)
    K = 0.10
    K_inv = Constant(_kinv_matrix(kinv_case, K=K).tolist(), dim=2)
    E = 1000.0
    nu = 0.30
    mu_s = E / (2.0 * (1.0 + nu))
    c_nh = Constant(mu_s / 2.0)
    # Paper (eq. 36): plane strain / 3D parameter.
    beta_nh = Constant(nu / (1.0 - 2.0 * nu))
    beta_BJ = Constant(beta_bj_val)
    kappa = Constant(math.sqrt(K) / (1.0 * float(mu_f.value) * math.sqrt(float(porosity.value))))

    xy = np.asarray(prob["mesh"].nodes_x_y_pos, dtype=float)
    bbox = (float(xy[:, 0].min()), float(xy[:, 0].max()), float(xy[:, 1].min()), float(xy[:, 1].max()))
    v_inf = _approx_vinf(mms, bbox=bbox)
    # Paper uses the *inverse* penalty parameters (gamma_n^{-1}, gamma_t^{-1}).
    # The forms take gamma_n, gamma_t themselves (small numbers in the paper's convention).
    gamma_inv_val = float(gamma_inv)
    if gamma_inv_val <= 0.0:
        raise ValueError("gamma_inv must be positive.")
    gamma = 1.0 / gamma_inv_val

    # Manufactured interface data: build from analytic fields and multiply by the
    # *discrete* interface normal (paper Remark 8, num_example.tex).
    n = FacetNormal()  # (-) -> (+) (poro -> fluid)
    nF = Constant(-1.0) * n  # fluid outward
    vF_A = Analytic(lambda x, y: mms.vF_k(x, y), degree=ana_deg)
    vP_A = Analytic(lambda x, y: mms.vP_k(x, y), degree=ana_deg)
    uP_Ak = Analytic(lambda x, y: mms.uP_k(x, y), degree=ana_deg)
    uP_An = Analytic(lambda x, y: mms.uP_n(x, y), degree=ana_deg)
    pP_A = Analytic(lambda x, y: mms.pP_k(x, y), degree=ana_deg)
    sigmaF_A = Analytic(lambda x, y: mms.sigmaF_k(x, y), degree=ana_deg)
    sigmaP_A = Analytic(lambda x, y: mms.sigmaP_k(x, y), degree=ana_deg)

    u_dot_A = (uP_Ak - uP_An) / prob["dt"]
    tractionF_A = dot(sigmaF_A, nF)

    g_sigma = dot(sigmaF_A - sigmaP_A, nF)
    g_sigma_n = dot(nF, tractionF_A) + pP_A
    g_n = vF_A - u_dot_A - porosity * (vP_A - u_dot_A)
    g_t = vF_A - u_dot_A - beta_BJ * porosity * (vP_A - u_dot_A) + kappa * tractionF_A

    h_gamma = prob["h_gamma"]

    forms = build_fpi_eulerian_forms(
        vF_k=prob["vF_kR"],
        pF_k=prob["pF_kR"],
        vP_k=prob["vP_kR"],
        uP_k=prob["uP_kR"],
        pP_k=prob["pP_kR"],
        vF_n=prob["vF_nR"],
        pF_n=prob["pF_nR"],
        vP_n=prob["vP_nR"],
        uP_n=prob["uP_nR"],
        pP_n=prob["pP_nR"],
        uP_nm1=prob["uP_nm1R"],
        dvF=prob["dvF_R"],
        dpF=prob["dpF_R"],
        dvP=prob["dvP_R"],
        duP=prob["duP_R"],
        dpP=prob["dpP_R"],
        vF_test=prob["vF_testR"],
        qF_test=prob["qF_testR"],
        vP_test=prob["vP_testR"],
        uP_test=prob["uP_testR"],
        qP_test=prob["qP_testR"],
        dx_f=prob["dx_f"],
        dx_p=prob["dx_p"],
        dGamma=prob["dGamma"],
        dG_f=prob["dG_f"],
        dG_p=prob["dG_p"],
        level_set=prob["poro_ls"],
        level_set_f=prob["fluid_ls"],
        level_set_p=prob["poro_ls"],
        dt=prob["dt"],
        theta=1.0,
        rho_f=rho_f,
        mu_f=mu_f,
        rho_s0_tilde=rho_s0_tilde,
        porosity=porosity,
        K_inv=K_inv,
        c_nh=c_nh,
        beta_nh=beta_nh,
        beta_BJ=beta_BJ,
        kappa=kappa,
        gamma_n=Constant(gamma),
        gamma_t=Constant(gamma),
        zeta=float(zeta),
        # Paper stabilization constants are applied when `use_paper_stabilization=True`.
        gamma_F_p=0.0,
        gamma_F_gp=0.0,
        gamma_P_p=0.0,
        gamma_P_gp=0.0,
        g_sigma=g_sigma,
        g_sigma_n=g_sigma_n,
        g_n=g_n,
        g_t=g_t,
        use_paper_phi_gamma=True,
        use_paper_stabilization=True,
        poly_order=poly_order,
        vF_inf=v_inf,
        c_v_gamma=1.0 / 6.0,
        c_t_gamma=1.0 / 12.0,
        h_gamma=h_gamma,
        w_ghost_f=prob["w_sliver_fluid"],
        w_ghost_p=prob["w_sliver_poro"],
        use_interface_terms=bool(use_interface_terms),
        use_stabilization=bool(use_stabilization),
    )

    fF = Analytic(lambda x, y: mms.fF(x, y), degree=ana_deg)
    fD = Analytic(lambda x, y: mms.fD(x, y), degree=ana_deg)
    fS = Analytic(lambda x, y: mms.fS(x, y), degree=ana_deg)
    f_mass = Analytic(lambda x, y: mms.f_mass(x, y), degree=ana_deg)

    residual_form = (
        forms.residual_form
        - dot(fF, prob["vF_testR"]) * prob["dx_f"]
        - dot(fD, prob["vP_testR"]) * prob["dx_p"]
        - dot(fS, prob["uP_testR"]) * prob["dx_p"]
        - f_mass * prob["qP_testR"] * prob["dx_p"]
    )

    # ------------------------------------------------------------------
    # Fluid boundary conditions (paper sec. 4.1)
    #  - Γ_F,D: essential BC on a *matching* mesh boundary (outer square, truncated).
    #  - Γ_F,N: traction BC on a *non-matching* boundary treated via dInterface.
    # ------------------------------------------------------------------
    inlet_bc = str(inlet_bc).strip().lower()
    if inlet_bc not in {"dofs", "edges", "auto"}:
        raise ValueError("inlet_bc must be one of {'dofs','edges','auto'}")
    edge_tag = str(prob.get("dirichlet_edge_tag", "inlet"))
    dof_tag = str(prob.get("dirichlet_dof_tag", "inlet_dofs"))
    if inlet_bc == "edges":
        inlet_tag = edge_tag
    elif inlet_bc == "auto":
        try:
            inlet_tag = edge_tag if prob["mesh"].edge_bitset(edge_tag).cardinality() > 0 else dof_tag
        except Exception:
            inlet_tag = dof_tag
    else:
        inlet_tag = dof_tag

    bcs = [
        BoundaryCondition("v_pos_x", "dirichlet", inlet_tag, mms.vF_x),
        BoundaryCondition("v_pos_y", "dirichlet", inlet_tag, mms.vF_y),
    ]
    bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]

    # Neumann traction on Γ_F,N: add -∫ (σ^F_A n^F) · w_F ds on the *fluid* side.
    # Our `neumann_ls` is chosen such that the physical fluid side is the negative
    # side (Ω⁻), hence use `Neg(...)` consistently.
    nF_neu = FacetNormal()
    traction_neu = dot(sigmaF_A, nF_neu)
    residual_form = residual_form - inner(traction_neu, Neg(prob["vF_testR"])) * prob["dGamma_neumann"]

    # ------------------------------------------------------------------
    # Sliver robustness: cut-cell sliver-mass stabilization (optional)
    # ------------------------------------------------------------------
    jacobian_form = forms.jacobian_form
    theta_eps_c = Constant(float(sliver_theta_eps))

    if float(sliver_mass_fluid) != 0.0:
        gamma_sm_f = Constant(float(sliver_mass_fluid))
        theta_f = np.clip(hansbo_cut_ratio(prob["mesh"], prob["fluid_ls"], side="+"), 0.0, 1.0)
        inv_theta_f = Constant(1.0) / (ElementWiseConstant(theta_f) + theta_eps_c)
        dx_f_cut = dx(
            defined_on=prob["domains_fluid"]["cut_domain"],
            level_set=prob["fluid_ls"],
            metadata={"q": int(qdeg), "side": "+"},
        )
        jacobian_form = jacobian_form + gamma_sm_f * (rho_f / prob["dt"]) * inv_theta_f * dot(prob["dvF_R"], prob["vF_testR"]) * dx_f_cut
        residual_form = residual_form + gamma_sm_f * (rho_f / prob["dt"]) * inv_theta_f * dot(
            prob["vF_kR"] - prob["vF_nR"], prob["vF_testR"]
        ) * dx_f_cut

    if float(sliver_mass_poro) != 0.0 or float(sliver_mass_skeleton) != 0.0:
        theta_p = np.clip(hansbo_cut_ratio(prob["mesh"], prob["poro_ls"], side="-"), 0.0, 1.0)
        inv_theta_p = Constant(1.0) / (ElementWiseConstant(theta_p) + theta_eps_c)
        dx_p_cut = dx(
            defined_on=prob["domains_poro"]["cut_domain"],
            level_set=prob["poro_ls"],
            metadata={"q": int(qdeg), "side": "-"},
        )

        if float(sliver_mass_poro) != 0.0:
            gamma_sm_p = Constant(float(sliver_mass_poro))
            jacobian_form = jacobian_form + gamma_sm_p * (rho_f / prob["dt"]) * inv_theta_p * dot(prob["dvP_R"], prob["vP_testR"]) * dx_p_cut
            residual_form = residual_form + gamma_sm_p * (rho_f / prob["dt"]) * inv_theta_p * dot(
                prob["vP_kR"] - prob["vP_nR"], prob["vP_testR"]
            ) * dx_p_cut

        if float(sliver_mass_skeleton) != 0.0:
            gamma_sm_u = Constant(float(sliver_mass_skeleton))
            dt = prob["dt"]
            jacobian_form = jacobian_form + gamma_sm_u * (rho_s0_tilde / (dt * dt)) * inv_theta_p * dot(prob["duP_R"], prob["uP_testR"]) * dx_p_cut
            residual_form = residual_form + gamma_sm_u * (rho_s0_tilde / (dt * dt)) * inv_theta_p * dot(
                prob["uP_kR"] - Constant(2.0) * prob["uP_nR"] + prob["uP_nm1R"], prob["uP_testR"]
            ) * dx_p_cut

    # Ghost-mass stabilization on cut-adjacent facets (optional): controls constant
    # modes that grad-jump ghost penalties do not see on extreme slivers.
    if float(ghost_mass_fluid) != 0.0:
        gamma_gm_f = Constant(float(ghost_mass_fluid))
        jacobian_form = jacobian_form + (rho_f / prob["dt"]) * gamma_gm_f * CellDiameter() * inner(
            jump(prob["dvF_R"]), jump(prob["vF_testR"])
        ) * prob["dG_f"]
        residual_form = residual_form + (rho_f / prob["dt"]) * gamma_gm_f * CellDiameter() * inner(
            jump(prob["vF_kR"]), jump(prob["vF_testR"])
        ) * prob["dG_f"]

    if float(ghost_mass_poro) != 0.0:
        gamma_gm_p = Constant(float(ghost_mass_poro))
        jacobian_form = jacobian_form + (rho_f / prob["dt"]) * gamma_gm_p * CellDiameter() * inner(
            jump(prob["dvP_R"]), jump(prob["vP_testR"])
        ) * prob["dG_p"]
        residual_form = residual_form + (rho_f / prob["dt"]) * gamma_gm_p * CellDiameter() * inner(
            jump(prob["vP_kR"]), jump(prob["vP_testR"])
        ) * prob["dG_p"]

    solver = NewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=prob["dh"],
        mixed_element=prob["me"],
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=1.0e-10, max_newton_iter=30),
        quad_order=qdeg,
        backend=backend,
    )

    # Initial conditions at t=t_prev and initial guess
    prob["vF_n"].set_values_from_function(lambda x, y: mms.vF_n(x, y))
    prob["vP_n"].set_values_from_function(lambda x, y: mms.vP_n(x, y))
    prob["uP_n"].set_values_from_function(lambda x, y: mms.uP_n(x, y))
    prob["pF_n"].set_values_from_function(lambda x, y: mms.pF_n(x, y))
    prob["pP_n"].set_values_from_function(lambda x, y: mms.pP_n(x, y))
    if mms.uP_nm1 is not None:
        prob["uP_nm1"].set_values_from_function(lambda x, y: mms.uP_nm1(x, y))
    else:
        prob["uP_nm1"].nodal_values.fill(0.0)

    init_guess = "exact" if bool(check_exact_residual) else str(init_guess).strip().lower()
    if init_guess == "exact":
        prob["vF_k"].set_values_from_function(lambda x, y: mms.vF_k(x, y))
        prob["vP_k"].set_values_from_function(lambda x, y: mms.vP_k(x, y))
        prob["uP_k"].set_values_from_function(lambda x, y: mms.uP_k(x, y))
        prob["pF_k"].set_values_from_function(lambda x, y: mms.pF_k(x, y))
        prob["pP_k"].set_values_from_function(lambda x, y: mms.pP_k(x, y))
    else:
        prob["vF_k"].nodal_values[:] = prob["vF_n"].nodal_values
        prob["vP_k"].nodal_values[:] = prob["vP_n"].nodal_values
        prob["uP_k"].nodal_values[:] = prob["uP_n"].nodal_values
        prob["pF_k"].nodal_values[:] = prob["pF_n"].nodal_values
        prob["pP_k"].nodal_values[:] = prob["pP_n"].nodal_values

    # Keep an untouched snapshot of u_n for post-solve interface error evaluation.
    # `solve_time_interval` promotes current -> previous at the end of the step.
    uP_n_snapshot = VectorFunction(
        name="uP_n_snapshot", field_names=["u_neg_x", "u_neg_y"], dof_handler=prob["dh"], side="-"
    )
    uP_n_snapshot.nodal_values[:] = prob["uP_n"].nodal_values[:]
    uP_n_snapshotR = restrict(uP_n_snapshot, prob["domains_poro"]["has_neg"])

    if not bool(check_exact_residual):
        solver.solve_time_interval(
            functions=[prob["vF_k"], prob["pF_k"], prob["vP_k"], prob["uP_k"], prob["pP_k"]],
            prev_functions=[prob["vF_n"], prob["pF_n"], prob["vP_n"], prob["uP_n"], prob["pP_n"]],
            aux_functions={"dt": prob["dt"], "uP_nm1": prob["uP_nm1"]},
            time_params=TimeStepperParameters(dt=dt_val, final_time=dt_val, max_steps=1),
        )
    else:
        # Assemble the residual on the manufactured solution without solving.
        def _assemble_vec(expr):
            eq = Equation(None, expr)
            _, vec = assemble_form(eq, dof_handler=prob["dh"], bcs=[], quad_order=int(qdeg), backend=backend)
            return np.asarray(vec, dtype=float)

        # Component-wise residual (for debugging): volume, interface, stabilization, boundary traction.
        res_fluid = forms.r_fluid - dot(fF, prob["vF_testR"]) * prob["dx_f"]
        res_poro = (
            forms.r_poro
            - dot(fD, prob["vP_testR"]) * prob["dx_p"]
            - dot(fS, prob["uP_testR"]) * prob["dx_p"]
            - f_mass * prob["qP_testR"] * prob["dx_p"]
        )
        res_ifc = forms.interface.residual if forms.interface is not None else Constant(0.0) * prob["qF_testR"] * prob["dGamma"]
        res_cip = forms.r_cip
        res_gp = forms.r_gp
        res_stab = forms.r_stab
        res_bdry = -inner(traction_neu, Neg(prob["vF_testR"])) * prob["dGamma_neumann"]

        R_fluid = _assemble_vec(res_fluid)
        R_poro = _assemble_vec(res_poro)
        R_ifc = _assemble_vec(res_ifc)
        R_cip = _assemble_vec(res_cip)
        R_gp = _assemble_vec(res_gp)
        R_stab = _assemble_vec(res_stab)
        R_bdry = _assemble_vec(res_bdry)
        R = R_fluid + R_poro + R_ifc + R_stab + R_bdry
        bc_dofs = set(prob["dh"].get_dirichlet_data(bcs).keys())
        inactive = set(getattr(prob["dh"], "dof_tags", {}).get("inactive", set()))
        free = np.asarray(sorted((set(range(prob["dh"].total_dofs)) - bc_dofs) - inactive), dtype=int)
        def _inf(v):
            return float(np.linalg.norm(v[free], ord=np.inf)) if free.size else 0.0

        return dict(
            h=float(prob["h"]),
            residual_inf=_inf(R),
            residual_inf_fluid=_inf(R_fluid),
            residual_inf_poro=_inf(R_poro),
            residual_inf_ifc=_inf(R_ifc),
            residual_inf_cip=_inf(R_cip),
            residual_inf_gp=_inf(R_gp),
            residual_inf_stab=_inf(R_stab),
            residual_inf_bdry=_inf(R_bdry),
        )

    dh: DofHandler = prob["dh"]
    fluid_ls = prob["fluid_ls"]
    poro_ls = prob["poro_ls"]

    theta_min_fluid = float("nan")
    theta_min_poro = float("nan")
    theta_zeros_fluid = 0
    theta_zeros_poro = 0
    if show_cut_ratios:
        try:
            prob["mesh"].classify_elements(fluid_ls)
            cut_ids = prob["mesh"].element_bitset("cut").to_indices()
            if cut_ids.size:
                theta = hansbo_cut_ratio(prob["mesh"], fluid_ls, side="+")
                theta_cut = np.asarray(theta, dtype=float)[cut_ids]
                theta_zeros_fluid = int(np.sum(theta_cut <= 0.0))
                # Use the smallest *strictly positive* cut ratio for diagnostics:
                # for convex level sets, Γ can "graze" an element (two intersections
                # on one edge) without contributing any volume on this side.
                theta_pos = theta_cut[np.isfinite(theta_cut) & (theta_cut > 0.0) & (theta_cut < 1.0)]
                theta_min_fluid = float(theta_pos.min()) if theta_pos.size else float("nan")
        except Exception:
            theta_min_fluid = float("nan")
        try:
            prob["mesh"].classify_elements(poro_ls)
            cut_ids = prob["mesh"].element_bitset("cut").to_indices()
            if cut_ids.size:
                theta = hansbo_cut_ratio(prob["mesh"], poro_ls, side="-")
                theta_cut = np.asarray(theta, dtype=float)[cut_ids]
                theta_zeros_poro = int(np.sum(theta_cut <= 0.0))
                theta_pos = theta_cut[np.isfinite(theta_cut) & (theta_cut > 0.0) & (theta_cut < 1.0)]
                theta_min_poro = float(theta_pos.min()) if theta_pos.size else float("nan")
        except Exception:
            theta_min_poro = float("nan")

    qerr_eff = int(qerr) if qerr is not None else max(int(qdeg), 10)

    err_vF = dh.l2_error_on_side(
        functions=prob["vF_k"],
        exact={"v_pos_x": mms.vF_x, "v_pos_y": mms.vF_y},
        fields=["v_pos_x", "v_pos_y"],
        level_set=fluid_ls,
        side="+",
        quad_order=qerr_eff,
        relative=False,
    )
    err_pF = dh.l2_error_on_side(
        functions=prob["pF_k"],
        exact={"p_pos_": mms.pF_s},
        fields=["p_pos_"],
        level_set=fluid_ls,
        side="+",
        quad_order=qerr_eff,
        relative=False,
    )
    err_vP = dh.l2_error_on_side(
        functions=prob["vP_k"],
        exact={"v_neg_x": mms.vP_x, "v_neg_y": mms.vP_y},
        fields=["v_neg_x", "v_neg_y"],
        level_set=poro_ls,
        side="-",
        quad_order=qerr_eff,
        relative=False,
    )
    err_uP = dh.l2_error_on_side(
        functions=prob["uP_k"],
        exact={"u_neg_x": mms.uP_x, "u_neg_y": mms.uP_y},
        fields=["u_neg_x", "u_neg_y"],
        level_set=poro_ls,
        side="-",
        quad_order=qerr_eff,
        relative=False,
    )
    err_pP = dh.l2_error_on_side(
        functions=prob["pP_k"],
        exact={"p_neg_": mms.pP_s},
        fields=["p_neg_"],
        level_set=poro_ls,
        side="-",
        quad_order=qerr_eff,
        relative=False,
    )

    # H1-seminorm errors (paper plots ∥∇v^F-∇v^F_A∥ and ∥∇u^P-∇u^P_A∥).
    err_gvF = dh.h1_error_vector_on_side(
        prob["vF_k"],
        mms.grad_vF_k,
        fluid_ls,
        side="+",
        relative=False,
        quad_order=qerr_eff,
    )
    err_guP = dh.h1_error_vector_on_side(
        prob["uP_k"],
        mms.grad_uP_k,
        poro_ls,
        side="-",
        relative=False,
        quad_order=qerr_eff,
    )

    # Interface kinematic errors (paper eq. (45))
    vF_A = Analytic(lambda x, y: mms.vF_k(x, y), degree=ana_deg)
    vP_A = Analytic(lambda x, y: mms.vP_k(x, y), degree=ana_deg)
    uP_Ak = Analytic(lambda x, y: mms.uP_k(x, y), degree=ana_deg)
    uP_An = Analytic(lambda x, y: mms.uP_n(x, y), degree=ana_deg)
    ifc = _interface_errors(
        dh=dh,
        dGamma=prob["dGamma"],
        vF_k=prob["vF_kR"],
        vP_k=prob["vP_kR"],
        uP_k=prob["uP_kR"],
        uP_n=uP_n_snapshotR,
        dt=prob["dt"],
        porosity=porosity,
        beta_BJ=beta_BJ,
        vF_A=vF_A,
        vP_A=vP_A,
        uP_Ak=uP_Ak,
        uP_An=uP_An,
        quad_order=qerr_eff,
        backend=backend,
    )

    return dict(
        h=float(prob["h"]),
        theta_min_fluid=theta_min_fluid,
        theta_min_poro=theta_min_poro,
        theta_zeros_fluid=theta_zeros_fluid,
        theta_zeros_poro=theta_zeros_poro,
        err_vF=err_vF,
        err_pF=err_pF,
        err_vP=err_vP,
        err_uP=err_uP,
        err_pP=err_pP,
        err_gvF=err_gvF,
        err_guP=err_guP,
        En=float(ifc["En"]),
        Et=float(ifc["Et"]),
    )


def main():
    # Reduce FormCompiler info logs.
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("pycutfem.ufl.compilers").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="jit", choices=["python", "jit", "cpp"])
    parser.add_argument("--kinv", type=str, default="iso", choices=["iso", "aniso"])
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument(
        "--dt-fine-levels",
        type=int,
        default=0,
        help="In convergence mode: use a smaller dt on the finest N levels (paper: 2).",
    )
    parser.add_argument(
        "--dt-fine-factor",
        type=float,
        default=0.5,
        help="In convergence mode: dt_fine = dt * factor (paper: 0.5).",
    )
    parser.add_argument("--t-end", type=float, default=0.1)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument(
        "--q-error",
        type=int,
        default=0,
        help="Quadrature order for error norms (0 -> max(q, 10)).",
    )
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument(
        "--mesh-layout",
        type=str,
        default="paper",
        choices=["paper", "legacy"],
        help="Geometry/layout: 'paper' uses a rotated outer mesh (Fig. 3); 'legacy' uses an axis-aligned background mesh.",
    )
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--levels", type=int, default=5, help="Number of h-refinement levels (convergence mode).")
    parser.add_argument("--interface", type=str, default="bj", choices=["bj", "bjs", "both"])
    parser.add_argument(
        "--ghost",
        type=str,
        default="edge",
        choices=["edge", "patch"],
        help="Ghost-penalty integration: edge-based (default) or facet-patch.",
    )
    parser.add_argument(
        "--ghost-weights",
        type=str,
        default="none",
        choices=["none", "hansbo"],
        help="Optional Hansbo cut-ratio weights for ghost penalties (default off).",
    )
    parser.add_argument(
        "--cut-drop-fluid",
        type=float,
        default=0.0,
        help="Drop (tag inactive) fluid DOFs on cut cells with θ_+ < this threshold (sliver robustness).",
    )
    parser.add_argument(
        "--cut-drop-poro",
        type=float,
        default=0.0,
        help="Drop (tag inactive) poro DOFs on cut cells with θ_- < this threshold (sliver robustness).",
    )
    parser.add_argument(
        "--sliver-mass-fluid",
        type=float,
        default=float(os.getenv("PYCUTFEM_SLIVER_MASS_FLUID", "0.0")),
        help="γ for fluid sliver-mass stabilization on cut cells (0 disables).",
    )
    parser.add_argument(
        "--sliver-mass-poro",
        type=float,
        default=float(os.getenv("PYCUTFEM_SLIVER_MASS_PORO", "0.0")),
        help="γ for porous-fluid sliver-mass stabilization on cut cells (0 disables).",
    )
    parser.add_argument(
        "--sliver-mass-skeleton",
        type=float,
        default=float(os.getenv("PYCUTFEM_SLIVER_MASS_SKELETON", "0.0")),
        help="γ for skeleton sliver-mass stabilization on cut cells (0 disables).",
    )
    parser.add_argument(
        "--sliver-theta-eps",
        type=float,
        default=1.0e-16,
        help="ε added to θ in sliver-mass denominators to avoid division by zero.",
    )
    parser.add_argument(
        "--ghost-mass-fluid",
        type=float,
        default=float(os.getenv("PYCUTFEM_GHOST_MASS_FLUID", "0.0")),
        help="γ for fluid velocity ghost-mass stabilization (jump-mass on ghost facets; 0 disables).",
    )
    parser.add_argument(
        "--ghost-mass-poro",
        type=float,
        default=float(os.getenv("PYCUTFEM_GHOST_MASS_PORO", "0.0")),
        help="γ for porous velocity ghost-mass stabilization (jump-mass on ghost facets; 0 disables).",
    )
    parser.add_argument("--show-cut-ratios", action="store_true", help="Print min Hansbo cut ratios per mesh level.")
    parser.add_argument("--gamma-inv", type=float, default=45.0, help="Use gamma_n^{-1}=gamma_t^{-1}=gamma_inv (paper: 45).")
    parser.add_argument("--zeta", type=float, default=-1.0, help="Adjoint term symmetry: +1 (consistent) or -1 (inconsistent).")
    parser.add_argument(
        "--no-interface-terms",
        action="store_false",
        dest="use_interface_terms",
        help="Disable Nitsche interface coupling terms (debug only).",
    )
    parser.add_argument(
        "--no-stabilization",
        action="store_false",
        dest="use_stabilization",
        help="Disable CIP/ghost stabilization terms (debug only).",
    )
    parser.add_argument(
        "--inlet-bc",
        type=str,
        default="dofs",
        choices=["dofs", "edges", "auto"],
        help="How to impose Γ_F,D Dirichlet: DOF-tag ('dofs'), boundary-edge tag ('edges'), or auto fallback.",
    )
    parser.add_argument(
        "--init",
        type=str,
        default="prev",
        choices=["prev", "exact"],
        help="Newton initial guess: previous time state ('prev') or exact MMS at t=t_end ('exact').",
    )
    parser.add_argument("--outdir", type=str, default="examples/FPI/_mms_example41_plots")
    parser.add_argument("--convergence", action="store_true", help="Run an h-refinement study (prints tables + saves plots).")
    parser.add_argument(
        "--nx-list",
        type=str,
        default="",
        help="In convergence mode: comma-separated nx list (overrides --nx/--levels).",
    )
    parser.add_argument(
        "--paper-h-range",
        action="store_true",
        help="In convergence mode: use the paper's h-range [0.25, 0.00390625] with 12 points (paper layout: nx=[4,6,8,12,16,24,32,48,64,96,128,256]; legacy: nx=[6,8,12,16,24,32,48,64,96,128,192,384]).",
    )
    parser.add_argument(
        "--check-exact-residual",
        action="store_true",
        help="Assemble the residual at the manufactured solution (no solve) and print |R|_inf on free DOFs.",
    )
    parser.add_argument(
        "--check-bcs",
        action="store_true",
        help="Print inlet boundary tags/Dirichlet DOF counts (no solve/assembly).",
    )
    parser.add_argument(
        "--plot-mesh",
        action="store_true",
        help="Save a mesh plot (uses plot_mesh_2) with inlet segment and Dirichlet DOFs highlighted.",
    )
    args = parser.parse_args()

    if args.plot_mesh:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        prob = _build_problem(
            nx=int(args.nx),
            poly_order=int(args.p),
            qdeg=int(args.q),
            dt_val=float(args.dt),
            kinv_case=args.kinv,
            mesh_layout=args.mesh_layout,
            ghost=args.ghost,
            ghost_weights=args.ghost_weights,
            cut_drop_fluid=args.cut_drop_fluid,
            cut_drop_poro=args.cut_drop_poro,
        )
        out_path = outdir / f"mesh_example41_{args.mesh_layout}_nx{int(args.nx)}_p{int(args.p)}.png"
        _plot_example41_mesh(prob=prob, inlet_bc=args.inlet_bc, out_path=out_path)
        print(f"Saved mesh plot to {out_path}")
        return

    if args.check_bcs:
        def _parse_nx_list(spec: str) -> list[int]:
            items = [s.strip() for s in str(spec).split(",") if s.strip()]
            out: list[int] = []
            for s in items:
                try:
                    out.append(int(s))
                except Exception as e:
                    raise ValueError(f"Invalid --nx-list entry {s!r}") from e
            if any(n <= 0 for n in out):
                raise ValueError(f"--nx-list must contain positive integers; got {out}")
            return out

        if args.convergence:
            if bool(getattr(args, "paper_h_range", False)):
                nx_list = (
                    [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256]
                    if str(args.mesh_layout).strip().lower() == "paper"
                    else [6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384]
                )
            else:
                nx_list = _parse_nx_list(getattr(args, "nx_list", "") or "")
                if not nx_list:
                    levels = max(1, int(args.levels))
                    nx_list = [int(args.nx) * (2**i) for i in range(levels)]
        else:
            nx_list = [int(args.nx)]

        print(
            f"\nFPI Example 4.1 BC check | layout={args.mesh_layout} | backend={args.backend} | p={args.p} | inlet_bc={args.inlet_bc} | K_inv={args.kinv}"
        )
        for nx in nx_list:
            prob = _build_problem(
                nx=int(nx),
                poly_order=int(args.p),
                qdeg=int(args.q),
                dt_val=float(args.dt),
                kinv_case=args.kinv,
                mesh_layout=args.mesh_layout,
                ghost=args.ghost,
                ghost_weights=args.ghost_weights,
                cut_drop_fluid=args.cut_drop_fluid,
                cut_drop_poro=args.cut_drop_poro,
            )
            print(f"\n  nx={nx:4d}  h={float(prob['h']):.3e}")
            _print_bc_diagnostics(prob=prob, inlet_bc=args.inlet_bc)
        return

    if args.check_exact_residual:
        out = _run_one(
            nx=args.nx,
            poly_order=args.p,
            qdeg=args.q,
            qerr=(None if int(args.q_error) <= 0 else int(args.q_error)),
            dt_val=args.dt,
            t_end=args.t_end,
            backend=args.backend,
            kinv_case=args.kinv,
            interface=("bj" if args.interface == "both" else args.interface),
            mesh_layout=args.mesh_layout,
            gamma_inv=args.gamma_inv,
            zeta=args.zeta,
            init_guess="exact",
            ghost=args.ghost,
            ghost_weights=args.ghost_weights,
            cut_drop_fluid=args.cut_drop_fluid,
            cut_drop_poro=args.cut_drop_poro,
            sliver_mass_fluid=args.sliver_mass_fluid,
            sliver_mass_poro=args.sliver_mass_poro,
            sliver_mass_skeleton=args.sliver_mass_skeleton,
            sliver_theta_eps=args.sliver_theta_eps,
            ghost_mass_fluid=args.ghost_mass_fluid,
            ghost_mass_poro=args.ghost_mass_poro,
            inlet_bc=args.inlet_bc,
            show_cut_ratios=args.show_cut_ratios,
            check_exact_residual=True,
            use_interface_terms=args.use_interface_terms,
            use_stabilization=args.use_stabilization,
        )
        print(f"h={out['h']:.3e}  |R|_inf(free)={out['residual_inf']:.3e}")
        print(
            "  components:"
            f"  fluid={out.get('residual_inf_fluid', float('nan')):.3e}"
            f"  poro={out.get('residual_inf_poro', float('nan')):.3e}"
            f"  ifc={out.get('residual_inf_ifc', float('nan')):.3e}"
            f"  cip={out.get('residual_inf_cip', float('nan')):.3e}"
            f"  gp={out.get('residual_inf_gp', float('nan')):.3e}"
            f"  stab={out.get('residual_inf_stab', float('nan')):.3e}"
            f"  bdry={out.get('residual_inf_bdry', float('nan')):.3e}"
        )
        return

    if not args.convergence:
        out = _run_one(
            nx=args.nx,
            poly_order=args.p,
            qdeg=args.q,
            qerr=(None if int(args.q_error) <= 0 else int(args.q_error)),
            dt_val=args.dt,
            t_end=args.t_end,
            backend=args.backend,
            kinv_case=args.kinv,
            interface=("bj" if args.interface == "both" else args.interface),
            mesh_layout=args.mesh_layout,
            gamma_inv=args.gamma_inv,
            zeta=args.zeta,
            init_guess=args.init,
            ghost=args.ghost,
            ghost_weights=args.ghost_weights,
            cut_drop_fluid=args.cut_drop_fluid,
            cut_drop_poro=args.cut_drop_poro,
            sliver_mass_fluid=args.sliver_mass_fluid,
            sliver_mass_poro=args.sliver_mass_poro,
            sliver_mass_skeleton=args.sliver_mass_skeleton,
            sliver_theta_eps=args.sliver_theta_eps,
            ghost_mass_fluid=args.ghost_mass_fluid,
            ghost_mass_poro=args.ghost_mass_poro,
            inlet_bc=args.inlet_bc,
            show_cut_ratios=args.show_cut_ratios,
            use_interface_terms=args.use_interface_terms,
            use_stabilization=args.use_stabilization,
        )
        print(
            f"h={out['h']:.3e}  |e(vF)|={out['err_vF']:.3e}  |e(pF)|={out['err_pF']:.3e}  "
            f"|e(vP)|={out['err_vP']:.3e}  |e(uP)|={out['err_uP']:.3e}  |e(pP)|={out['err_pP']:.3e}  "
            f"|e(grad vF)|={out['err_gvF']:.3e}  |e(grad uP)|={out['err_guP']:.3e}  "
            f"En={out['En']:.3e}  Et={out['Et']:.3e}"
        )
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def _parse_nx_list(spec: str) -> list[int]:
        items = [s.strip() for s in str(spec).split(",") if s.strip()]
        out: list[int] = []
        for s in items:
            try:
                out.append(int(s))
            except Exception as e:
                raise ValueError(f"Invalid --nx-list entry {s!r}") from e
        if not out:
            return []
        if any(n <= 0 for n in out):
            raise ValueError(f"--nx-list must contain positive integers; got {out}")
        return out

    if bool(args.paper_h_range):
        nx_list = (
            [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256]
            if str(args.mesh_layout).strip().lower() == "paper"
            else [6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384]
        )
    else:
        nx_list = _parse_nx_list(getattr(args, "nx_list", "") or "")
        if not nx_list:
            levels = max(2, int(args.levels))
            nx_list = [int(args.nx) * (2**i) for i in range(levels)]

    levels = len(nx_list)

    def _dt_for_level(i: int) -> float:
        fine_levels = max(0, int(args.dt_fine_levels))
        if fine_levels <= 0:
            return float(args.dt)
        if i >= levels - fine_levels:
            return float(args.dt) * float(args.dt_fine_factor)
        return float(args.dt)

    def _run_convergence(interface: str):
        rows: list[dict[str, float]] = []
        for i, nx in enumerate(nx_list):
            out = _run_one(
                nx=nx,
                poly_order=args.p,
                qdeg=args.q,
                qerr=(None if int(args.q_error) <= 0 else int(args.q_error)),
                dt_val=_dt_for_level(i),
                t_end=args.t_end,
                backend=args.backend,
                kinv_case=args.kinv,
                interface=interface,
                mesh_layout=args.mesh_layout,
                gamma_inv=args.gamma_inv,
                zeta=args.zeta,
                init_guess=args.init,
                ghost=args.ghost,
                ghost_weights=args.ghost_weights,
                cut_drop_fluid=args.cut_drop_fluid,
                cut_drop_poro=args.cut_drop_poro,
                sliver_mass_fluid=args.sliver_mass_fluid,
                sliver_mass_poro=args.sliver_mass_poro,
                sliver_mass_skeleton=args.sliver_mass_skeleton,
                sliver_theta_eps=args.sliver_theta_eps,
                ghost_mass_fluid=args.ghost_mass_fluid,
                ghost_mass_poro=args.ghost_mass_poro,
                inlet_bc=args.inlet_bc,
                show_cut_ratios=args.show_cut_ratios,
                use_interface_terms=args.use_interface_terms,
                use_stabilization=args.use_stabilization,
            )
            out["nx"] = int(nx)
            out["dt"] = float(_dt_for_level(i))
            rows.append(out)

        print(
            f"\nFPI Example 4.1 (BE) | layout={args.mesh_layout} | backend={args.backend} | p={args.p} | K_inv={args.kinv} | interface={interface}"
        )

        def _eoc(prev_h: float, curr_h: float, prev_err: float, curr_err: float) -> float:
            if not (prev_h > 0.0 and curr_h > 0.0):
                return float("nan")
            if not (prev_err > 0.0 and curr_err > 0.0):
                return float("nan")
            return float(math.log(prev_err / curr_err) / math.log(prev_h / curr_h))

        def _fmt(val):
            return "   - " if not np.isfinite(val) else f"{val:6.2f}"

        # Domain L2 errors (Fig. 5: vF, pF, vP, pP, uP)
        print("\n  Domain L2 errors")
        print(
            f"{'h':>8}  {'e(vF)':>12} {'eoc':>6}  {'e(pF)':>12} {'eoc':>6}  "
            f"{'e(vP)':>12} {'eoc':>6}  {'e(pP)':>12} {'eoc':>6}  {'e(uP)':>12} {'eoc':>6}"
        )
        for i, r in enumerate(rows):
            if i == 0:
                eoc_vf = eoc_pf = eoc_vp = eoc_pp = eoc_up = float("nan")
            else:
                prev = rows[i - 1]
                eoc_vf = _eoc(prev["h"], r["h"], prev["err_vF"], r["err_vF"])
                eoc_pf = _eoc(prev["h"], r["h"], prev["err_pF"], r["err_pF"])
                eoc_vp = _eoc(prev["h"], r["h"], prev["err_vP"], r["err_vP"])
                eoc_pp = _eoc(prev["h"], r["h"], prev["err_pP"], r["err_pP"])
                eoc_up = _eoc(prev["h"], r["h"], prev["err_uP"], r["err_uP"])
            print(
                f"{r['h']:8.3e}  {r['err_vF']:12.3e} {_fmt(eoc_vf)}  {r['err_pF']:12.3e} {_fmt(eoc_pf)}  "
                f"{r['err_vP']:12.3e} {_fmt(eoc_vp)}  {r['err_pP']:12.3e} {_fmt(eoc_pp)}  {r['err_uP']:12.3e} {_fmt(eoc_up)}"
            )

        # Grad and kinematic interface errors (Fig. 5: grad(vF), grad(uP), En, Et)
        print("\n  Gradient / kinematic interface errors")
        print(
            f"{'h':>8}  {'e(grad vF)':>12} {'eoc':>6}  {'e(grad uP)':>12} {'eoc':>6}  "
            f"{'En':>12} {'eoc':>6}  {'Et':>12} {'eoc':>6}"
        )
        for i, r in enumerate(rows):
            if i == 0:
                eoc_gvf = eoc_gup = eoc_en = eoc_et = float("nan")
            else:
                prev = rows[i - 1]
                eoc_gvf = _eoc(prev["h"], r["h"], prev["err_gvF"], r["err_gvF"])
                eoc_gup = _eoc(prev["h"], r["h"], prev["err_guP"], r["err_guP"])
                eoc_en = _eoc(prev["h"], r["h"], prev["En"], r["En"])
                eoc_et = _eoc(prev["h"], r["h"], prev["Et"], r["Et"])
            print(
                f"{r['h']:8.3e}  {r['err_gvF']:12.3e} {_fmt(eoc_gvf)}  {r['err_guP']:12.3e} {_fmt(eoc_gup)}  "
                f"{r['En']:12.3e} {_fmt(eoc_en)}  {r['Et']:12.3e} {_fmt(eoc_et)}"
            )

        if args.show_cut_ratios:
            print("\n  Cut-cell diagnostics (min Hansbo θ on cut cells)")
            print(f"{'h':>8}  {'theta_min(fluid)':>16}  {'#θ=0(F)':>8}  {'theta_min(poro)':>14}  {'#θ=0(P)':>8}")
            for r in rows:
                tf = r.get("theta_min_fluid", float("nan"))
                tp = r.get("theta_min_poro", float("nan"))
                zf = int(r.get("theta_zeros_fluid", 0) or 0)
                zp = int(r.get("theta_zeros_poro", 0) or 0)
                print(f"{r['h']:8.3e}  {tf:16.3e}  {zf:8d}  {tp:14.3e}  {zp:8d}")

        # ------------------------------------------------------------------
        # Pandas tables (saved by default in convergence mode)
        # ------------------------------------------------------------------
        df_all = pd.DataFrame(rows)
        df_all = df_all.sort_values("h", ascending=False, ignore_index=True)

        def _add_eoc(df: pd.DataFrame, key: str, out_key: str) -> None:
            hs = df["h"].to_numpy(dtype=float)
            ys = df[key].to_numpy(dtype=float)
            eoc = np.full_like(ys, np.nan, dtype=float)
            for j in range(1, ys.size):
                eoc[j] = _eoc(float(hs[j - 1]), float(hs[j]), float(ys[j - 1]), float(ys[j]))
            df[out_key] = eoc

        for k in ("err_vF", "err_pF", "err_vP", "err_pP", "err_uP", "err_gvF", "err_guP", "En", "Et"):
            _add_eoc(df_all, k, f"eoc_{k}")

        df_domain = df_all[
            [
                "nx",
                "h",
                "dt",
                "err_vF",
                "eoc_err_vF",
                "err_pF",
                "eoc_err_pF",
                "err_vP",
                "eoc_err_vP",
                "err_pP",
                "eoc_err_pP",
                "err_uP",
                "eoc_err_uP",
            ]
        ].copy()
        df_grad = df_all[
            [
                "nx",
                "h",
                "dt",
                "err_gvF",
                "eoc_err_gvF",
                "err_guP",
                "eoc_err_guP",
                "En",
                "eoc_En",
                "Et",
                "eoc_Et",
            ]
        ].copy()

        print("\n  Domain L2 errors (pandas)")
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
            print(df_domain)
        print("\n  LaTeX (domain) (copy/paste):\n")
        latex_domain = df_domain.to_latex(index=False, float_format="%.3e", na_rep="-")
        print(latex_domain)

        print("\n  Gradient / kinematic interface errors (pandas)")
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
            print(df_grad)
        print("\n  LaTeX (grad/interface) (copy/paste):\n")
        latex_grad = df_grad.to_latex(index=False, float_format="%.3e", na_rep="-")
        print(latex_grad)

        stem = (
            f"fpi_example41_convergence_layout={args.mesh_layout}_backend={args.backend}_p={int(args.p)}_kinv={args.kinv}_iface={interface}"
        )
        out_csv_domain = outdir / f"{stem}_domain.csv"
        out_csv_grad = outdir / f"{stem}_grad.csv"
        out_tex_domain = outdir / f"{stem}_domain.tex"
        out_tex_grad = outdir / f"{stem}_grad.tex"
        df_domain.to_csv(out_csv_domain, index=False)
        df_grad.to_csv(out_csv_grad, index=False)
        out_tex_domain.write_text(latex_domain, encoding="utf-8")
        out_tex_grad.write_text(latex_grad, encoding="utf-8")
        print(f"\nSaved: {out_csv_domain}")
        print(f"Saved: {out_csv_grad}")
        print(f"Saved: {out_tex_domain}")
        print(f"Saved: {out_tex_grad}")
        return rows

    rows_by_iface: dict[str, list[dict[str, float]]] = {}
    if args.interface in {"bj", "bjs"}:
        rows_by_iface[args.interface] = _run_convergence(args.interface)
    else:
        rows_by_iface["bj"] = _run_convergence("bj")
        rows_by_iface["bjs"] = _run_convergence("bjs")

    plot_specs = [
        ("err_vF", "e(vF)"),
        ("err_pF", "e(pF)"),
        ("err_vP", "e(vP)"),
        ("err_pP", "e(pP)"),
        ("err_uP", "e(uP)"),
        ("err_gvF", "e(grad vF)"),
        ("err_guP", "e(grad uP)"),
        ("En", "En"),
        ("Et", "Et"),
    ]

    def _plot_metric(metric_key: str, y_label: str) -> None:
        plt.figure(figsize=(6.0, 4.0))
        colors = {"bj": "C0", "bjs": "C1"}
        hs_by_iface: dict[str, np.ndarray] = {}
        ys_by_iface: dict[str, np.ndarray] = {}
        for iface, rows in rows_by_iface.items():
            hs = np.array([r["h"] for r in rows], dtype=float)
            ys = np.array([r[metric_key] for r in rows], dtype=float)
            hs_by_iface[iface] = hs
            ys_by_iface[iface] = ys
            plt.loglog(hs, ys, "o-", label=iface.upper(), color=colors.get(iface, None))

        def _first_positive(hs: np.ndarray, ys: np.ndarray) -> tuple[float, float] | None:
            for h, y in zip(hs.tolist(), ys.tolist()):
                if not (np.isfinite(h) and np.isfinite(y)):
                    continue
                if h > 0.0 and y > 0.0:
                    return float(h), float(y)
            return None

        def _add_ref_line(*, ref_h: np.ndarray, slope: float, h0: float, y0: float, label: str, style: str, lw: float) -> None:
            if not (h0 > 0.0 and y0 > 0.0):
                return
            y_ref = y0 * (ref_h / h0) ** float(slope)
            plt.loglog(ref_h, y_ref, style, color="0.35", alpha=0.7, lw=lw, label=label)

        # Reference slope overlays: show h, h^{3/2}, h^2 on every plot.
        # Paper notes: grad errors ~ h, and Et differs between BJ (~h^{3/2}) and BJS (~h^2).
        if hs_by_iface:
            hs_cat = np.concatenate(list(hs_by_iface.values()))
            ref_h = np.array([float(np.min(hs_cat)), float(np.max(hs_cat))], dtype=float)
            ref_h.sort()

            anchor_default = None
            for iface, hs in hs_by_iface.items():
                ys = ys_by_iface[iface]
                anchor_default = _first_positive(hs, ys)
                if anchor_default is not None:
                    break

            anchor_bj = None
            anchor_bjs = None
            if "bj" in hs_by_iface:
                anchor_bj = _first_positive(hs_by_iface["bj"], ys_by_iface["bj"])
            if "bjs" in hs_by_iface:
                anchor_bjs = _first_positive(hs_by_iface["bjs"], ys_by_iface["bjs"])

            highlight_slope = 1.0 if metric_key in {"err_gvF", "err_guP"} else 2.0
            slopes = [
                (1.0, "h^1 ref", "--"),
                (1.5, "h^{3/2} ref", "-."),
                (2.0, "h^2 ref", ":"),
            ]

            for slope, label, style in slopes:
                anchor = anchor_default
                if metric_key == "Et" and anchor_bj is not None and anchor_bjs is not None:
                    if abs(slope - 1.5) < 1.0e-12:
                        anchor = anchor_bj
                    elif abs(slope - 2.0) < 1.0e-12:
                        anchor = anchor_bjs
                    else:
                        anchor = anchor_bj
                if anchor is None:
                    continue
                h0, y0 = anchor
                lw = 2.0 if abs(slope - highlight_slope) < 1.0e-12 else 1.25
                _add_ref_line(ref_h=ref_h, slope=slope, h0=h0, y0=y0, label=label, style=style, lw=lw)

        plt.gca().invert_xaxis()
        plt.xlabel("h")
        plt.ylabel(y_label)
        plt.title(f"FPI Example 4.1 {y_label} | layout={args.mesh_layout} | backend={args.backend} | p={args.p} | K_inv={args.kinv}")
        plt.grid(True, which="both", ls=":")
        plt.legend()
        outpath = outdir / f"example41_{metric_key}_layout-{args.mesh_layout}_backend-{args.backend}_p{args.p}_kinv-{args.kinv}_iface-{args.interface}.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        print(f"Saved log-log plot: {outpath}")

    print("\nSaving convergence plots...")
    for key, ylabel in plot_specs:
        _plot_metric(key, ylabel)


if __name__ == "__main__":
    main()
