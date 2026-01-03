import numpy as np
import pytest

from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.meshgen import structured_quad
from pycutfem.utils.fsi_fully_eulerian import (
    build_fsi_eulerian_forms,
    build_measures,
    hansbo_kappa,
    make_domain_sets,
    retag_inactive,
)
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    ElementWiseConstant,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    Pos,
    Neg,
    restrict,
)
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.compilers import FormCompiler


BACKENDS = ("python", "jit", "cpp")


def _build_context(poly_order: int = 2, *, nx: int = 2, ny: int = 2, levelset_c: float = 0.48):
    L = 2.0
    H = 2.0
    nodes, elems, edges, corners = structured_quad(
        L, H, nx=nx, ny=ny, poly_order=poly_order, offset=(-L / 2.0, -H / 2.0)
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )

    level_set = AffineLevelSet(a=1.0, b=0.0, c=levelset_c)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)

    domains = make_domain_sets(mesh, use_aligned_interface=False)
    dx_fluid, dx_solid, dGamma, dG_fluid, dG_solid = build_measures(
        mesh, level_set, domains, qvol=3
    )

    me = MixedElement(
        mesh,
        field_specs={
            "u_pos_x": poly_order,
            "u_pos_y": poly_order,
            "p_pos_": poly_order - 1,
            "vs_neg_x": poly_order - 1,
            "vs_neg_y": poly_order - 1,
            "d_neg_x": poly_order - 1,
            "d_neg_y": poly_order - 1,
        },
    )
    dh = DofHandler(me, method="cg")

    velocity_fluid_space = FunctionSpace(name="velocity_fluid", field_names=["u_pos_x", "u_pos_y"], dim=1, side="+")
    pressure_fluid_space = FunctionSpace(name="pressure_fluid", field_names=["p_pos_"], dim=0, side="+")
    velocity_solid_space = FunctionSpace(name="velocity_solid", field_names=["vs_neg_x", "vs_neg_y"], dim=1, side="-")
    displacement_space = FunctionSpace(name="displacement", field_names=["d_neg_x", "d_neg_y"], dim=1, side="-")

    du_f = VectorTrialFunction(space=velocity_fluid_space, dof_handler=dh)
    dp_f = TrialFunction(name="trial_pressure_fluid", field_name="p_pos_", dof_handler=dh, side="+")
    du_s = VectorTrialFunction(space=velocity_solid_space, dof_handler=dh)
    ddisp_s = VectorTrialFunction(space=displacement_space, dof_handler=dh)
    test_vel_f = VectorTestFunction(space=velocity_fluid_space, dof_handler=dh)
    test_q_f = TestFunction(name="test_pressure_fluid", field_name="p_pos_", dof_handler=dh, side="+")
    test_vel_s = VectorTestFunction(space=velocity_solid_space, dof_handler=dh)
    test_disp_s = VectorTestFunction(space=displacement_space, dof_handler=dh)

    uf_k = VectorFunction(name="u_f_k", field_names=["u_pos_x", "u_pos_y"], dof_handler=dh, side="+")
    pf_k = Function(name="p_f_k", field_name="p_pos_", dof_handler=dh, side="+")
    uf_n = VectorFunction(name="u_f_n", field_names=["u_pos_x", "u_pos_y"], dof_handler=dh, side="+")
    pf_n = Function(name="p_f_n", field_name="p_pos_", dof_handler=dh, side="+")
    us_k = VectorFunction(name="u_s_k", field_names=["vs_neg_x", "vs_neg_y"], dof_handler=dh, side="-")
    us_n = VectorFunction(name="u_s_n", field_names=["vs_neg_x", "vs_neg_y"], dof_handler=dh, side="-")
    disp_k = VectorFunction(name="disp_k", field_names=["d_neg_x", "d_neg_y"], dof_handler=dh, side="-")
    disp_n = VectorFunction(name="disp_n", field_names=["d_neg_x", "d_neg_y"], dof_handler=dh, side="-")

    for func in (uf_k, pf_k, uf_n, pf_n, us_k, us_n, disp_k, disp_n):
        func.nodal_values.fill(0.0)

    theta_pos_vals, theta_neg_vals = hansbo_kappa(mesh, level_set, theta_min=1.0e-3)
    kappa_pos = Pos(ElementWiseConstant(theta_pos_vals))
    kappa_neg = Neg(ElementWiseConstant(theta_neg_vals))

    retag_inactive(dh, mesh, theta_neg=theta_neg_vals, solid_cut_drop=0.0)

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

    rho_f = Constant(1.0e3)
    rho_s = Constant(1.0e3)
    mu_f = Constant(1.0)
    mu_s = Constant(1.0)
    lambda_s = Constant(1.0)
    beta_N = Constant(90.0 * 1.0 * poly_order * (poly_order + 1))
    dt = Constant(1.0e-3)
    theta = Constant(1.0)

    penalty_val = 1.0e-3
    penalty_grad = 1.0e-3
    gamma_v = Constant(penalty_val * poly_order**2)
    gamma_v_grad = Constant(penalty_grad * poly_order**2)
    gamma_p = Constant(penalty_val * poly_order)
    solid_reg_eps = Constant(1.0e-6)

    cell_h = CellDiameter()

    forms = build_fsi_eulerian_forms(
        du_f=du_f_R,
        dp_f=dp_f_R,
        du_s=du_s_R,
        ddisp_s=ddisp_s_R,
        test_vel_f=test_vel_f_R,
        test_q_f=test_q_f_R,
        test_vel_s=test_vel_s_R,
        test_disp_s=test_disp_s_R,
        uf_k=uf_k_R,
        pf_k=pf_k_R,
        uf_n=uf_n_R,
        pf_n=pf_n_R,
        us_k=us_k_R,
        us_n=us_n_R,
        disp_k=disp_k_R,
        disp_n=disp_n_R,
        dx_fluid=dx_fluid,
        dx_solid=dx_solid,
        dGamma=dGamma,
        dG_fluid=dG_fluid,
        dG_solid=dG_solid,
        kappa_pos=kappa_pos,
        kappa_neg=kappa_neg,
        cell_h=cell_h,
        beta_N=beta_N,
        rho_f=rho_f,
        rho_s=rho_s,
        mu_f=mu_f,
        mu_s=mu_s,
        lambda_s=lambda_s,
        dt=dt,
        theta=theta,
        gamma_v=gamma_v,
        gamma_p=gamma_p,
        gamma_v_grad=gamma_v_grad,
        solid_reg_eps=solid_reg_eps,
        use_linear_solid=True,
        solid_advect_lagged=True,
        s_nitsche_value=1.0,
    )

    fd_fields = {
        "u_pos_x": uf_k,
        "u_pos_y": uf_k,
        "p_pos_": pf_k,
        "vs_neg_x": us_k,
        "vs_neg_y": us_k,
        "d_neg_x": disp_k,
        "d_neg_y": disp_k,
    }

    return mesh, dh, forms, fd_fields


def _select_fd_dofs(dh: DofHandler, fields_to_probe: dict[str, int], elem_tag: str = "cut") -> np.ndarray:
    selected: list[int] = []
    elems = dh.element_bitset(elem_tag).to_indices()
    probe_eid = int(elems[0]) if len(elems) else 0
    for field, count in fields_to_probe.items():
        try:
            local = dh.element_dofs(field, probe_eid)
        except Exception:
            local = []
        selected.extend(list(local[:count]))
    return np.array(sorted(set(selected)), dtype=int)


def _fd_check(
    jac_form,
    res_form,
    dh: DofHandler,
    bcs,
    functions: dict[str, Function | VectorFunction],
    probe_dofs: np.ndarray,
    *,
    compiler: FormCompiler,
    eps: float = 1.0e-7,
):
    eq_jac = Equation(jac_form, None)
    eq_res = Equation(None, res_form)
    base_K, _ = compiler.assemble(eq_jac, bcs=bcs)
    if base_K is None:
        raise AssertionError("Jacobian or residual assembly returned None.")

    bc_dofs = set(dh.get_dirichlet_data(bcs).keys()) if bcs else set()
    inactive = set(dh.dof_tags.get("inactive", set()))
    active_mask = np.ones(dh.total_dofs, dtype=bool)
    if bc_dofs:
        active_mask[np.fromiter(bc_dofs, dtype=int)] = False
    if inactive:
        active_mask[np.fromiter(inactive, dtype=int)] = False

    direction = np.zeros(dh.total_dofs, dtype=float)
    field_dofs: dict[str, np.ndarray] = {}
    for gdof in probe_dofs:
        gdof_i = int(gdof)
        if gdof_i in bc_dofs or gdof_i in inactive:
            continue
        field, _ = dh._dof_to_node_map[gdof_i]
        if field not in functions:
            continue
        direction[gdof_i] = 1.0
        field_dofs.setdefault(field, []).append(gdof_i)

    if not np.any(direction):
        raise AssertionError("Directional FD probe is empty after filtering DOFs.")

    base_vals: dict[str, np.ndarray] = {}
    for field, dofs in field_dofs.items():
        dof_arr = np.asarray(dofs, dtype=int)
        base_vals[field] = functions[field].get_nodal_values(dof_arr)

    def _set_vals(sign: float):
        for field, dofs in field_dofs.items():
            dof_arr = np.asarray(dofs, dtype=int)
            values = base_vals[field] + sign * eps
            functions[field].set_nodal_values(dof_arr, values)

    _set_vals(+1.0)
    _, R_plus = compiler.assemble(eq_res, bcs=bcs)
    _set_vals(-1.0)
    _, R_minus = compiler.assemble(eq_res, bcs=bcs)
    _set_vals(0.0)

    fd_vec = (R_plus - R_minus) / (2 * eps)
    jac_vec = base_K.dot(direction)
    err_vec = fd_vec - jac_vec
    max_abs = float(np.linalg.norm(err_vec[active_mask], ord=np.inf))
    mag = float(np.linalg.norm(jac_vec[active_mask], ord=np.inf))
    max_rel = max_abs / (mag + 1.0e-14) if mag > 0.0 else 0.0

    return max_abs, max_rel


@pytest.mark.parametrize("backend", BACKENDS)
def test_fsi_eulerian_fd_small_mesh(backend):
    _mesh, dh, forms, fd_fields = _build_context(poly_order=1)
    probe = _select_fd_dofs(dh, {"u_pos_x": 1, "vs_neg_x": 1}, elem_tag="cut")
    compiler = FormCompiler(dh, backend=backend)

    term_blocks = {
        "interface_pen": (forms.J_int_pen, forms.R_int_pen),
    }
    if backend == "python":
        term_blocks.update(
            {
                "volume": (forms.a_vol_f + forms.a_vol_s, forms.r_vol_f + forms.r_vol_s),
                "interface": (forms.J_int, forms.R_int),
                "stab": (forms.a_stab, forms.r_stab),
            }
        )

    for name, (jf, rf) in term_blocks.items():
        abs_err, rel_err = _fd_check(
            jf,
            rf,
            dh,
            [],
            fd_fields,
            probe,
            compiler=compiler,
            eps=1.0e-7,
        )
        assert np.isfinite(abs_err), f"{backend} produced non-finite abs error for {name}"
        assert np.isfinite(rel_err), f"{backend} produced non-finite rel error for {name}"
        assert abs_err < 1.0e-6, f"{backend} abs error too large for {name}: {abs_err:.3e}"
        assert rel_err < 1.0e-5, f"{backend} rel error too large for {name}: {rel_err:.3e}"


@pytest.mark.parametrize("backend", ("jit", "cpp"))
def test_fsi_eulerian_fd_stab_ghost_both(backend):
    mesh, dh, forms, fd_fields = _build_context(
        poly_order=2, nx=4, ny=4, levelset_c=0.0
    )
    if mesh.edge_bitset("ghost_both").cardinality() == 0:
        pytest.skip("No ghost_both edges found; adjust mesh or level set.")
    probe = _select_fd_dofs(dh, {"u_pos_x": 1, "vs_neg_x": 1}, elem_tag="cut")
    compiler = FormCompiler(dh, backend=backend)

    abs_err, rel_err = _fd_check(
        forms.a_stab,
        forms.r_stab,
        dh,
        [],
        fd_fields,
        probe,
        compiler=compiler,
        eps=1.0e-7,
    )

    assert np.isfinite(abs_err), f"{backend} produced non-finite abs error for stab (ghost_both)"
    assert np.isfinite(rel_err), f"{backend} produced non-finite rel error for stab (ghost_both)"
    assert abs_err < 1.0e-6, f"{backend} abs error too large for stab (ghost_both): {abs_err:.3e}"
    assert rel_err < 1.0e-5, f"{backend} rel error too large for stab (ghost_both): {rel_err:.3e}"
