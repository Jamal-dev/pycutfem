import argparse
import logging
import math
from pathlib import Path

import matplotlib
import numpy as np

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
    restrict,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.utils.fpi_fully_eulerian import build_fpi_eulerian_forms
from pycutfem.utils.fpi_mms_example41 import build_example41_mms
from pycutfem.utils.fsi_fully_eulerian import build_measures, make_domain_sets
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
    ghost: str = "edge",
    cut_drop_fluid: float = 0.0,
    cut_drop_poro: float = 0.0,
):
    # Background mesh (paper Fig. 3): square discretization with matching Dirichlet
    # boundary on the *vertical cut* at x = Δx = -0.45.
    x0 = -0.45
    y0 = -0.75
    L = 1.5
    x1 = x0 + L
    y1 = y0 + L

    nodes, elems, edges, corners = structured_quad(
        L, L, nx=nx, ny=nx, poly_order=poly_order, offset=(x0, y0)
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )
    # Level sets for Example 4.1 (paper sec. 4.1)
    poro_ls = RotatedBoxLevelSet(center=(0.0, 0.0), hx=0.25, hy=0.25, angle=math.pi / 6.0)  # Ω^P
    fluid_sq_ls = RotatedBoxLevelSet(center=(0.0, 0.0), hx=0.5, hy=0.5, angle=math.pi / 4.0)  # base square
    cut_ls = AffineLevelSet(-1.0, 0.0, x0)  # φ = x0 - x  (negative for x > x0)

    # Ω^F support square with vertical cut: Ω^B = square ∩ {x >= x0} (negative inside)
    outer_std = MaxLevelSet(fluid_sq_ls, cut_ls)
    # Flip sign so Ω^B is on the positive side (needed for measures with side='+').
    outer_pos = ScaledLevelSet(-1.0, outer_std)

    # Fluid domain Ω^F = Ω^B \ Ω^P : positive where (outside poro) ∩ (inside Ω^B)
    fluid_ls = MinLevelSet(poro_ls, outer_pos)

    # Tag matching boundary Γ_F,D ⊂ {x=x0}∩∂Ω^F as "inlet"
    _tag_background_boundaries(mesh, x0=x0, x1=x1, y0=y0, y1=y1, fluid_ls=fluid_ls)

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

    # --- Outer (non-matching) fluid boundary Γ_F,N for traction BCs ---
    # This must only include the rotated outer-square boundary, *not* the matching
    # inlet cut {x=x0}. Build it from `fluid_sq_ls` (paper sec. 4.1).
    mesh.classify_elements(fluid_sq_ls)
    mesh.classify_edges(fluid_sq_ls)
    mesh.build_interface_segments(fluid_sq_ls)
    domains_outer = make_domain_sets(mesh, use_aligned_interface=False)
    _dx_out_b, _dx_in_b, dGamma_outer, _dG_out_b, _dG_in_b = build_measures(mesh, fluid_sq_ls, domains_outer, qvol=qdeg)

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

    dh.tag_dofs_by_locator_map({"inlet_dofs": _inlet_dof_locator}, fields=["v_pos_x", "v_pos_y"])

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
        mesh=mesh,
        poro_ls=poro_ls,
        fluid_ls=fluid_ls,
        outer_ls=outer_pos,
        domains_fluid=domains_fluid,
        domains_poro=domains_poro,
        domains_outer=domains_outer,
        dh=dh,
        me=me,
        dx_f=dx_f,
        dx_p=dx_p,
        dGamma=dGamma,
        dG_f=dG_f,
        dG_p=dG_p,
        dGamma_outer=dGamma_outer,
        h_gamma=h_gamma,
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
    dt_val: float,
    t_end: float,
    backend: str,
    kinv_case: str,
    interface: str,
    gamma_inv: float,
    zeta: float,
    init_guess: str,
    ghost: str,
    cut_drop_fluid: float,
    cut_drop_poro: float,
    show_cut_ratios: bool = False,
):
    prob = _build_problem(
        nx=nx,
        poly_order=poly_order,
        qdeg=qdeg,
        dt_val=dt_val,
        kinv_case=kinv_case,
        ghost=ghost,
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
        use_interface_terms=True,
        use_stabilization=True,
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
    #  - Γ_F,D (matching): prescribe analytic v^F on the vertical cut boundary x=x0.
    #  - Γ_F,N (non-matching): prescribe analytic traction σ^F_A · n^F on the cut boundary
    #    of the rotated outer square.
    # ------------------------------------------------------------------
    bcs = [
        BoundaryCondition("v_pos_x", "dirichlet", "inlet_dofs", mms.vF_x),
        BoundaryCondition("v_pos_y", "dirichlet", "inlet_dofs", mms.vF_y),
    ]
    bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]

    # Neumann traction on Γ_F,N: add -∫ (σ^F_A n^F) · w_F ds.
    # Here `dGamma_outer` is built from `fluid_sq_ls` with negative inside the
    # square, so FacetNormal() points outward.
    nF_outer = FacetNormal()
    traction_outer = dot(sigmaF_A, nF_outer)
    # `dGamma_outer` is built from `fluid_sq_ls`, which is negative inside the
    # outer square (Ω⁻) and positive outside (Ω⁺). The physical Neumann boundary
    # Γ_F,N lies on the *inside* (fluid) side, hence use `Neg(...)`.
    residual_form = residual_form - inner(traction_outer, Neg(prob["vF_testR"])) * prob["dGamma_outer"]

    solver = NewtonSolver(
        residual_form,
        forms.jacobian_form,
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

    init_guess = str(init_guess).strip().lower()
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

    solver.solve_time_interval(
        functions=[prob["vF_k"], prob["pF_k"], prob["vP_k"], prob["uP_k"], prob["pP_k"]],
        prev_functions=[prob["vF_n"], prob["pF_n"], prob["vP_n"], prob["uP_n"], prob["pP_n"]],
        aux_functions={"dt": prob["dt"], "uP_nm1": prob["uP_nm1"]},
        time_params=TimeStepperParameters(dt=dt_val, final_time=dt_val, max_steps=1),
    )

    dh: DofHandler = prob["dh"]
    fluid_ls = prob["fluid_ls"]
    poro_ls = prob["poro_ls"]

    theta_min_fluid = float("nan")
    theta_min_poro = float("nan")
    if show_cut_ratios:
        try:
            prob["mesh"].classify_elements(fluid_ls)
            cut_ids = prob["mesh"].element_bitset("cut").to_indices()
            if cut_ids.size:
                theta = hansbo_cut_ratio(prob["mesh"], fluid_ls, side="+")
                theta_min_fluid = float(np.asarray(theta, dtype=float)[cut_ids].min())
        except Exception:
            theta_min_fluid = float("nan")
        try:
            prob["mesh"].classify_elements(poro_ls)
            cut_ids = prob["mesh"].element_bitset("cut").to_indices()
            if cut_ids.size:
                theta = hansbo_cut_ratio(prob["mesh"], poro_ls, side="-")
                theta_min_poro = float(np.asarray(theta, dtype=float)[cut_ids].min())
        except Exception:
            theta_min_poro = float("nan")

    err_vF = dh.l2_error_on_side(
        functions=prob["vF_k"],
        exact={"v_pos_x": mms.vF_x, "v_pos_y": mms.vF_y},
        fields=["v_pos_x", "v_pos_y"],
        level_set=fluid_ls,
        side="+",
        quad_order=qdeg,
        relative=False,
    )
    err_pF = dh.l2_error_on_side(
        functions=prob["pF_k"],
        exact={"p_pos_": mms.pF_s},
        fields=["p_pos_"],
        level_set=fluid_ls,
        side="+",
        quad_order=qdeg,
        relative=False,
    )
    err_vP = dh.l2_error_on_side(
        functions=prob["vP_k"],
        exact={"v_neg_x": mms.vP_x, "v_neg_y": mms.vP_y},
        fields=["v_neg_x", "v_neg_y"],
        level_set=poro_ls,
        side="-",
        quad_order=qdeg,
        relative=False,
    )
    err_uP = dh.l2_error_on_side(
        functions=prob["uP_k"],
        exact={"u_neg_x": mms.uP_x, "u_neg_y": mms.uP_y},
        fields=["u_neg_x", "u_neg_y"],
        level_set=poro_ls,
        side="-",
        quad_order=qdeg,
        relative=False,
    )
    err_pP = dh.l2_error_on_side(
        functions=prob["pP_k"],
        exact={"p_neg_": mms.pP_s},
        fields=["p_neg_"],
        level_set=poro_ls,
        side="-",
        quad_order=qdeg,
        relative=False,
    )

    # H1-seminorm errors (paper plots ∥∇v^F-∇v^F_A∥ and ∥∇u^P-∇u^P_A∥).
    err_gvF = dh.h1_error_vector_on_side(
        prob["vF_k"],
        mms.grad_vF_k,
        fluid_ls,
        side="+",
        relative=False,
        quad_order=qdeg,
    )
    err_guP = dh.h1_error_vector_on_side(
        prob["uP_k"],
        mms.grad_uP_k,
        poro_ls,
        side="-",
        relative=False,
        quad_order=qdeg,
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
        quad_order=qdeg,
        backend=backend,
    )

    return dict(
        h=float(prob["h"]),
        theta_min_fluid=theta_min_fluid,
        theta_min_poro=theta_min_poro,
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
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--nx", type=int, default=6)
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
    parser.add_argument("--show-cut-ratios", action="store_true", help="Print min Hansbo cut ratios per mesh level.")
    parser.add_argument("--gamma-inv", type=float, default=45.0, help="Use gamma_n^{-1}=gamma_t^{-1}=gamma_inv (paper: 45).")
    parser.add_argument("--zeta", type=float, default=-1.0, help="Adjoint term symmetry: +1 (consistent) or -1 (inconsistent).")
    parser.add_argument(
        "--init",
        type=str,
        default="prev",
        choices=["prev", "exact"],
        help="Newton initial guess: previous time state ('prev') or exact MMS at t=t_end ('exact').",
    )
    parser.add_argument("--outdir", type=str, default="examples/FPI/_mms_example41_plots")
    parser.add_argument("--convergence", action="store_true", help="Run a 2-level h-refinement study.")
    args = parser.parse_args()

    if not args.convergence:
        out = _run_one(
            nx=args.nx,
            poly_order=args.p,
            qdeg=args.q,
            dt_val=args.dt,
            t_end=args.t_end,
            backend=args.backend,
            kinv_case=args.kinv,
            interface=("bj" if args.interface == "both" else args.interface),
            gamma_inv=args.gamma_inv,
            zeta=args.zeta,
            init_guess=args.init,
            ghost=args.ghost,
            cut_drop_fluid=args.cut_drop_fluid,
            cut_drop_poro=args.cut_drop_poro,
            show_cut_ratios=args.show_cut_ratios,
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

    levels = max(2, int(args.levels))
    nx_list = [int(args.nx) * (2**i) for i in range(levels)]

    def _dt_for_level(i: int) -> float:
        fine_levels = max(0, int(args.dt_fine_levels))
        if fine_levels <= 0:
            return float(args.dt)
        if i >= levels - fine_levels:
            return float(args.dt) * float(args.dt_fine_factor)
        return float(args.dt)

    def _run_convergence(interface: str):
        rows = [
            _run_one(
                nx=nx,
                poly_order=args.p,
                qdeg=args.q,
                dt_val=_dt_for_level(i),
                t_end=args.t_end,
                backend=args.backend,
                kinv_case=args.kinv,
                interface=interface,
                gamma_inv=args.gamma_inv,
                zeta=args.zeta,
                init_guess=args.init,
                ghost=args.ghost,
                cut_drop_fluid=args.cut_drop_fluid,
                cut_drop_poro=args.cut_drop_poro,
                show_cut_ratios=args.show_cut_ratios,
            )
            for i, nx in enumerate(nx_list)
        ]

        print(f"\nFPI Example 4.1 (BE) | backend={args.backend} | p={args.p} | K_inv={args.kinv} | interface={interface}")

        def _eoc(pe, ce):
            if pe <= 0.0 or ce <= 0.0:
                return float("nan")
            return math.log(pe / ce, 2.0)

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
                eoc_vf = _eoc(prev["err_vF"], r["err_vF"])
                eoc_pf = _eoc(prev["err_pF"], r["err_pF"])
                eoc_vp = _eoc(prev["err_vP"], r["err_vP"])
                eoc_pp = _eoc(prev["err_pP"], r["err_pP"])
                eoc_up = _eoc(prev["err_uP"], r["err_uP"])
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
                eoc_gvf = _eoc(prev["err_gvF"], r["err_gvF"])
                eoc_gup = _eoc(prev["err_guP"], r["err_guP"])
                eoc_en = _eoc(prev["En"], r["En"])
                eoc_et = _eoc(prev["Et"], r["Et"])
            print(
                f"{r['h']:8.3e}  {r['err_gvF']:12.3e} {_fmt(eoc_gvf)}  {r['err_guP']:12.3e} {_fmt(eoc_gup)}  "
                f"{r['En']:12.3e} {_fmt(eoc_en)}  {r['Et']:12.3e} {_fmt(eoc_et)}"
            )

        if args.show_cut_ratios:
            print("\n  Cut-cell diagnostics (min Hansbo θ on cut cells)")
            print(f"{'h':>8}  {'theta_min(fluid)':>16}  {'theta_min(poro)':>14}")
            for r in rows:
                tf = r.get("theta_min_fluid", float("nan"))
                tp = r.get("theta_min_poro", float("nan"))
                print(f"{r['h']:8.3e}  {tf:16.3e}  {tp:14.3e}")
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
        plt.title(f"FPI Example 4.1 {y_label} | backend={args.backend} | p={args.p} | K_inv={args.kinv}")
        plt.grid(True, which="both", ls=":")
        plt.legend()
        outpath = outdir / f"example41_{metric_key}_backend-{args.backend}_p{args.p}_kinv-{args.kinv}_iface-{args.interface}.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        print(f"Saved log-log plot: {outpath}")

    print("\nSaving convergence plots...")
    for key, ylabel in plot_specs:
        _plot_metric(key, ylabel)


if __name__ == "__main__":
    main()
