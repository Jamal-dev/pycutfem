import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import LevelSetGridFunction
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import CellDiameter, Constant, Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction, restrict
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.utils.fsi_fully_eulerian import build_fsi_eulerian_forms, build_measures, make_domain_sets, refresh_domain_sets, retag_inactive
from pycutfem.utils.meshgen import structured_quad


def _tag_rect_boundaries(mesh: Mesh, *, x0: float, x1: float, y0: float, y1: float, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - x0) <= tol,
            "right": lambda x, y: abs(x - x1) <= tol,
            "bottom": lambda x, y: abs(y - y0) <= tol,
            "top": lambda x, y: abs(y - y1) <= tol,
        }
    )


def _set_const_vector(dh: DofHandler, vf: VectorFunction, *, field_x: str, field_y: str, ux: float, uy: float) -> None:
    gd = np.asarray(dh.get_field_slice(field_x), dtype=int)
    vf.set_nodal_values(gd, np.full_like(gd, float(ux), dtype=float))
    gd = np.asarray(dh.get_field_slice(field_y), dtype=int)
    vf.set_nodal_values(gd, np.full_like(gd, float(uy), dtype=float))


def _free_dofs(dh: DofHandler, bcs: list[BoundaryCondition]) -> np.ndarray:
    dirichlet = dh.get_dirichlet_data(bcs) or {}
    bc = set(map(int, dirichlet.keys()))
    inactive = set(getattr(dh, "dof_tags", {}).get("inactive", set()))
    blocked = bc | inactive
    return np.array([i for i in range(int(dh.total_dofs)) if i not in blocked], dtype=int)


def test_fsi_eulerian_mms_rigid_translation_moving_interface_residual_is_small() -> None:
    # Mesh / geometry
    x0, x1 = -1.0, 1.0
    y0, y1 = -0.5, 0.5
    Lx, Ly = x1 - x0, y1 - y0
    nx, ny = 6, 3

    nodes, elems, edges, corners = structured_quad(Lx, Ly, nx=nx, ny=ny, poly_order=1, offset=(x0, y0))
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    _tag_rect_boundaries(mesh, x0=x0, x1=x1, y0=y0, y1=y1)

    # Level set: moving vertical interface x = c(t)
    ls_me = MixedElement(mesh, field_specs={"phi": 1})
    ls_dh = DofHandler(ls_me, method="cg")
    level_set = LevelSetGridFunction(ls_dh, field="phi")
    ls_coords = np.asarray(ls_dh.get_dof_coords("phi"), dtype=float)

    def _commit_interface(c: float) -> None:
        phi = ls_coords[:, 0] - float(c)
        level_set.set_from_array(phi)
        level_set.commit(tol=1.0e-12)

    _commit_interface(0.0)
    domains = make_domain_sets(mesh, use_aligned_interface=False)
    dx_fluid, dx_solid, dGamma, dG_fluid, dG_solid = build_measures(mesh, level_set, domains, qvol=3)

    # Unknowns (matching build_fsi_eulerian_forms conventions)
    me = MixedElement(
        mesh,
        field_specs={
            "u_pos_x": 1,
            "u_pos_y": 1,
            "p_pos_": 1,
            "vs_neg_x": 1,
            "vs_neg_y": 1,
            "d_neg_x": 1,
            "d_neg_y": 1,
        },
    )
    dh = DofHandler(me, method="cg")

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

    dt_val = 0.05
    dt = Constant(dt_val)
    theta = Constant(1.0)
    rho_f = Constant(1.0)
    rho_s = Constant(1.0)

    # Restrict to active side sets so the residual is evaluated on the physical dofs.
    R = domains
    forms = build_fsi_eulerian_forms(
        du_f=restrict(du_f, R["has_pos"]),
        dp_f=restrict(dp_f, R["has_pos"]),
        du_s=restrict(du_s, R["has_neg"]),
        ddisp_s=restrict(ddisp_s, R["has_neg"]),
        test_vel_f=restrict(v_f, R["has_pos"]),
        test_q_f=restrict(q_f, R["has_pos"]),
        test_vel_s=restrict(v_s, R["has_neg"]),
        test_disp_s=restrict(w_s, R["has_neg"]),
        uf_k=restrict(uf_k, R["has_pos"]),
        pf_k=restrict(pf_k, R["has_pos"]),
        uf_n=restrict(uf_n, R["has_pos"]),
        pf_n=restrict(pf_n, R["has_pos"]),
        us_k=restrict(us_k, R["has_neg"]),
        us_n=restrict(us_n, R["has_neg"]),
        disp_k=restrict(disp_k, R["has_neg"]),
        disp_n=restrict(disp_n, R["has_neg"]),
        dx_fluid=dx_fluid,
        dx_solid=dx_solid,
        dGamma=dGamma,
        dG_fluid=dG_fluid,
        dG_solid=dG_solid,
        kappa_pos=Constant(0.5),
        kappa_neg=Constant(0.5),
        cell_h=CellDiameter(),
        beta_N=Constant(20.0),
        rho_f=rho_f,
        rho_s=rho_s,
        mu_f=Constant(1.0),
        mu_s=Constant(1.0),
        lambda_s=Constant(0.0),
        dt=dt,
        theta=theta,
        gamma_v=Constant(0.0),
        gamma_p=Constant(0.0),
        gamma_v_grad=Constant(0.0),
        svc_scale=rho_s / dt,
        solid_reg_eps=Constant(0.0),
        use_linear_solid=True,
        solid_advect_lagged=True,
        s_nitsche_value=0.0,
    )

    # Move interface (simulate one time-step motion)
    _commit_interface(0.13)
    refresh_domain_sets(mesh, domains)
    retag_inactive(dh, mesh)

    # Constant translation exact fields
    ux, uy = 0.2, -0.15
    t0, t1 = 0.0, float(dt_val)

    _set_const_vector(dh, uf_n, field_x="u_pos_x", field_y="u_pos_y", ux=ux, uy=uy)
    _set_const_vector(dh, uf_k, field_x="u_pos_x", field_y="u_pos_y", ux=ux, uy=uy)
    _set_const_vector(dh, us_n, field_x="vs_neg_x", field_y="vs_neg_y", ux=ux, uy=uy)
    _set_const_vector(dh, us_k, field_x="vs_neg_x", field_y="vs_neg_y", ux=ux, uy=uy)
    _set_const_vector(dh, disp_n, field_x="d_neg_x", field_y="d_neg_y", ux=ux * t0, uy=uy * t0)
    _set_const_vector(dh, disp_k, field_x="d_neg_x", field_y="d_neg_y", ux=ux * t1, uy=uy * t1)

    pf_n.nodal_values.fill(0.0)
    pf_k.nodal_values.fill(0.0)

    bcs: list[BoundaryCondition] = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("u_pos_x", "dirichlet", tag, float(ux)),
                BoundaryCondition("u_pos_y", "dirichlet", tag, float(uy)),
                BoundaryCondition("vs_neg_x", "dirichlet", tag, float(ux)),
                BoundaryCondition("vs_neg_y", "dirichlet", tag, float(uy)),
                BoundaryCondition("d_neg_x", "dirichlet", tag, float(ux * t1)),
                BoundaryCondition("d_neg_y", "dirichlet", tag, float(uy * t1)),
            ]
        )

    _, F = assemble_form(Equation(None, forms.residual_form), dof_handler=dh, bcs=[], backend="python")
    free = _free_dofs(dh, bcs)
    res_inf = float(np.linalg.norm(F[free], ord=np.inf)) if free.size else 0.0
    assert res_inf < 1.0e-12
