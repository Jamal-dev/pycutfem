import math

import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    restrict,
)
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.utils.fpi_interface_eulerian import build_fpi_interface_forms
from pycutfem.utils.fsi_fully_eulerian import build_measures, make_domain_sets
from pycutfem.utils.meshgen import structured_quad


def _selected_backends(default: str = "jit") -> list[str]:
    import os

    raw = os.environ.get("PYCUTFEM_TEST_BACKENDS", os.environ.get("BACKEND", default))
    raw = str(raw).strip().lower()
    if raw in {"", "default"}:
        raw = default
    if raw in {"all", "*"}:
        return ["python", "jit", "cpp"]
    return [b.strip() for b in raw.split(",") if b.strip()]


def _set_constant_vector(vec: VectorFunction, value: tuple[float, float]) -> None:
    vec.set_values_from_function(lambda x, y: np.array([float(value[0]), float(value[1])]))


def _set_constant_scalar(f: Function, value: float) -> None:
    f.nodal_values.fill(float(value))


def _pick_cut_element(mesh: Mesh) -> int:
    cut_ids = mesh.element_bitset("cut").to_indices()
    if cut_ids.size:
        return int(cut_ids[0])
    return 0


def _select_fd_dofs(dh: DofHandler, fields_to_probe: dict[str, int], *, eid: int) -> np.ndarray:
    selected: list[int] = []
    for field, count in fields_to_probe.items():
        try:
            local = dh.element_dofs(field, int(eid))
        except Exception:
            local = []
        selected.extend(list(local[: int(count)]))
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
        raise AssertionError("Jacobian assembly returned None.")

    direction = np.zeros(dh.total_dofs, dtype=float)
    field_dofs: dict[str, list[int]] = {}
    for gdof in probe_dofs:
        gdof_i = int(gdof)
        field, _ = dh._dof_to_node_map[gdof_i]
        if field not in functions:
            continue
        direction[gdof_i] = 1.0
        field_dofs.setdefault(field, []).append(gdof_i)

    if not np.any(direction):
        raise AssertionError("Directional FD probe is empty.")

    base_vals: dict[str, np.ndarray] = {}
    for field, dofs in field_dofs.items():
        dof_arr = np.asarray(dofs, dtype=int)
        base_vals[field] = functions[field].get_nodal_values(dof_arr)

    def _set_vals(sign: float):
        for field, dofs in field_dofs.items():
            dof_arr = np.asarray(dofs, dtype=int)
            functions[field].set_nodal_values(dof_arr, base_vals[field] + sign * eps)

    _set_vals(+1.0)
    _, R_plus = compiler.assemble(eq_res, bcs=bcs)
    _set_vals(-1.0)
    _, R_minus = compiler.assemble(eq_res, bcs=bcs)
    _set_vals(0.0)

    fd_vec = (R_plus - R_minus) / (2 * eps)
    jac_vec = base_K.dot(direction)
    err_vec = fd_vec - jac_vec
    max_abs = float(np.linalg.norm(err_vec, ord=np.inf))
    mag = float(np.linalg.norm(jac_vec, ord=np.inf))
    max_rel = max_abs / (mag + 1.0e-14) if mag > 0.0 else 0.0
    return max_abs, max_rel


def _build_interface_problem(*, poly_order: int = 1):
    # Square [-1,1]^2 with a non-aligned vertical interface x=0:
    # pick nx=3 so x=0 cuts the middle column but still leaves pure elements.
    nodes, elems, edges, corners = structured_quad(2.0, 2.0, nx=3, ny=3, poly_order=poly_order, offset=(-1.0, -1.0))
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )

    # Interface x = 0 with Ω⁻={x<0} (poro) and Ω⁺={x>0} (fluid)
    level_set = AffineLevelSet(a=1.0, b=0.0, c=0.0)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)

    domains = make_domain_sets(mesh, use_aligned_interface=False)
    _, _, dGamma, _, _ = build_measures(mesh, level_set, domains, qvol=5)

    assert domains["fluid_domain"].cardinality() > 0
    assert domains["solid_domain"].cardinality() > 0
    assert domains["cut_interface"].cardinality() > 0

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
        },
    )
    dh = DofHandler(me, method="cg")

    V_f = FunctionSpace(name="vF", field_names=["v_pos_x", "v_pos_y"], dim=1, side="+")
    V_p = FunctionSpace(name="vP", field_names=["v_neg_x", "v_neg_y"], dim=1, side="-")
    U_p = FunctionSpace(name="uP", field_names=["u_neg_x", "u_neg_y"], dim=1, side="-")

    dvF = VectorTrialFunction(space=V_f, dof_handler=dh)
    dpF = TrialFunction(name="dpF", field_name="p_pos_", dof_handler=dh, side="+")
    dvP = VectorTrialFunction(space=V_p, dof_handler=dh)
    duP = VectorTrialFunction(space=U_p, dof_handler=dh)

    dvF_test = VectorTestFunction(space=V_f, dof_handler=dh)
    dpF_test = TestFunction(name="dpF_test", field_name="p_pos_", dof_handler=dh, side="+")
    dvP_test = VectorTestFunction(space=V_p, dof_handler=dh)
    duP_test = VectorTestFunction(space=U_p, dof_handler=dh)

    vF = VectorFunction(name="vF", field_names=["v_pos_x", "v_pos_y"], dof_handler=dh, side="+")
    pF = Function(name="pF", field_name="p_pos_", dof_handler=dh, side="+")
    vP = VectorFunction(name="vP", field_names=["v_neg_x", "v_neg_y"], dof_handler=dh, side="-")
    uP = VectorFunction(name="uP", field_names=["u_neg_x", "u_neg_y"], dof_handler=dh, side="-")
    uP_n = VectorFunction(name="uP_n", field_names=["u_neg_x", "u_neg_y"], dof_handler=dh, side="-")

    for f in (vF, pF, vP, uP, uP_n):
        f.nodal_values.fill(0.0)

    # Restrict fields to the active side sets (CutFEM assembly convention).
    vF_R = restrict(vF, domains["has_pos"])
    pF_R = restrict(pF, domains["has_pos"])
    dvF_R = restrict(dvF, domains["has_pos"])
    dpF_R = restrict(dpF, domains["has_pos"])
    dvF_test_R = restrict(dvF_test, domains["has_pos"])
    dpF_test_R = restrict(dpF_test, domains["has_pos"])

    vP_R = restrict(vP, domains["has_neg"])
    uP_R = restrict(uP, domains["has_neg"])
    uP_n_R = restrict(uP_n, domains["has_neg"])
    dvP_R = restrict(dvP, domains["has_neg"])
    duP_R = restrict(duP, domains["has_neg"])
    dvP_test_R = restrict(dvP_test, domains["has_neg"])
    duP_test_R = restrict(duP_test, domains["has_neg"])

    return dict(
        mesh=mesh,
        level_set=level_set,
        domains=domains,
        dGamma=dGamma,
        dh=dh,
        vF=vF_R,
        pF=pF_R,
        vP=vP_R,
        uP=uP_R,
        uP_n=uP_n_R,
        dvF=dvF_R,
        dpF=dpF_R,
        dvP=dvP_R,
        duP=duP_R,
        dvF_test=dvF_test_R,
        dpF_test=dpF_test_R,
        dvP_test=dvP_test_R,
        duP_test=duP_test_R,
        # also keep original (unrestricted) for nodal value assignment
        vF_full=vF,
        pF_full=pF,
        vP_full=vP,
        uP_full=uP,
        uP_n_full=uP_n,
    )


@pytest.mark.parametrize("backend", _selected_backends(default="python"))
def test_fpi_interface_case_a_traction_free_residual_and_fd(backend):
    prob = _build_interface_problem(poly_order=1)
    dh = prob["dh"]

    # -------------------------------------------------------------------------
    # CASE A: traction-free, manufactured kinematic data -> residual ~ 0
    # -------------------------------------------------------------------------
    vF_val = (1.0, 0.5)
    uPdot_val = (0.2, -0.3)  # with dt=1 and u_n=0 => u_dot = u_k
    vP_val = (0.8, 0.9)
    pF_val = 0.0

    _set_constant_vector(prob["vF_full"], vF_val)
    _set_constant_scalar(prob["pF_full"], pF_val)
    _set_constant_vector(prob["vP_full"], vP_val)
    _set_constant_vector(prob["uP_full"], uPdot_val)
    prob["uP_n_full"].nodal_values.fill(0.0)

    porosity = 0.6
    beta_BJ = 1.0
    kin_vec = np.array(vF_val) - np.array(uPdot_val) - porosity * (np.array(vP_val) - np.array(uPdot_val))
    slip_vec = np.array(vF_val) - np.array(uPdot_val) - beta_BJ * porosity * (np.array(vP_val) - np.array(uPdot_val))

    g_sigma = Constant((0.0, 0.0), dim=1)
    g_sigma_n = Constant(0.0, dim=0)
    g_n = Constant((float(kin_vec[0]), 0.0), dim=1)
    g_t = Constant((0.0, float(slip_vec[1])), dim=1)

    forms = build_fpi_interface_forms(
        vF_k=prob["vF"],
        pF_k=prob["pF"],
        vP_k=prob["vP"],
        uP_k=prob["uP"],
        uP_n=prob["uP_n"],
        dvF=prob["dvF"],
        dpF=prob["dpF"],
        dvP=prob["dvP"],
        duP=prob["duP"],
        dvF_test=prob["dvF_test"],
        dpF_test=prob["dpF_test"],
        dvP_test=prob["dvP_test"],
        duP_test=prob["duP_test"],
        dGamma=prob["dGamma"],
        mu_f=Constant(1.0),
        porosity=Constant(porosity),
        beta_BJ=Constant(beta_BJ),
        kappa=Constant(1.0),
        gamma_n=Constant(10.0),
        gamma_t=Constant(10.0),
        phi_gamma_F=Constant(1.0),
        h_gamma=CellDiameter(),
        zeta=Constant(1.0),
        dt=Constant(1.0),
        g_sigma=g_sigma,
        g_sigma_n=g_sigma_n,
        g_n=g_n,
        g_t=g_t,
    )

    qdeg = 6
    compiler = FormCompiler(dh, quadrature_order=qdeg, backend=backend)
    _, R = compiler.assemble(Equation(None, forms.residual), bcs=[])

    def inf_norm(field: str) -> float:
        sl = np.asarray(dh.get_field_slice(field), dtype=int)
        return float(np.linalg.norm(R[sl], ord=np.inf)) if sl.size else 0.0

    # Expect near-zero residual for all active interface fields.
    for fld in ("v_pos_x", "v_pos_y", "p_pos_", "v_neg_x", "v_neg_y", "u_neg_x", "u_neg_y"):
        assert inf_norm(fld) < 1.0e-10

    # FD consistency on a few cut-element DOFs
    eid = _pick_cut_element(prob["mesh"])
    probe = _select_fd_dofs(
        dh,
        {"v_pos_x": 2, "p_pos_": 2, "v_neg_x": 2, "u_neg_x": 2},
        eid=eid,
    )
    functions = {
        "v_pos_x": prob["vF_full"],
        "v_pos_y": prob["vF_full"],
        "p_pos_": prob["pF_full"],
        "v_neg_x": prob["vP_full"],
        "v_neg_y": prob["vP_full"],
        "u_neg_x": prob["uP_full"],
        "u_neg_y": prob["uP_full"],
    }
    abs_err, rel_err = _fd_check(
        forms.jacobian,
        forms.residual,
        dh,
        [],
        functions,
        probe,
        compiler=compiler,
        eps=1.0e-7,
    )
    assert math.isfinite(abs_err) and math.isfinite(rel_err)
    assert abs_err < 1.0e-6
    assert rel_err < 1.0e-5


@pytest.mark.parametrize("backend", _selected_backends(default="python"))
def test_fpi_interface_case_b_normal_traction_sanity(backend):
    prob = _build_interface_problem(poly_order=1)
    dh = prob["dh"]

    # -------------------------------------------------------------------------
    # CASE B: nonzero normal traction; cancel porous-side traction with g_sigma*
    # -------------------------------------------------------------------------
    vF_val = (0.0, 0.0)
    uPdot_val = (0.0, 0.0)
    vP_val = (0.0, 0.0)
    pF_val = 1.0

    _set_constant_vector(prob["vF_full"], vF_val)
    _set_constant_scalar(prob["pF_full"], pF_val)
    _set_constant_vector(prob["vP_full"], vP_val)
    _set_constant_vector(prob["uP_full"], uPdot_val)
    prob["uP_n_full"].nodal_values.fill(0.0)

    # With our normal convention (FacetNormal is poro->fluid and nF=-n),
    # σ=-pI gives traction σ·nF = +p*n (along +x). Choose g_sigma to match
    # and g_sigma_n = -p to cancel the porous-pressure boundary replacement.
    g_sigma = Constant((1.0, 0.0), dim=1)
    g_sigma_n = Constant(-1.0, dim=0)
    g_n = Constant((0.0, 0.0), dim=1)
    g_t = Constant((0.0, 0.0), dim=1)

    forms = build_fpi_interface_forms(
        vF_k=prob["vF"],
        pF_k=prob["pF"],
        vP_k=prob["vP"],
        uP_k=prob["uP"],
        uP_n=prob["uP_n"],
        dvF=prob["dvF"],
        dpF=prob["dpF"],
        dvP=prob["dvP"],
        duP=prob["duP"],
        dvF_test=prob["dvF_test"],
        dpF_test=prob["dpF_test"],
        dvP_test=prob["dvP_test"],
        duP_test=prob["duP_test"],
        dGamma=prob["dGamma"],
        mu_f=Constant(1.0),
        porosity=Constant(0.6),
        beta_BJ=Constant(1.0),
        kappa=Constant(1.0),
        gamma_n=Constant(10.0),
        gamma_t=Constant(10.0),
        phi_gamma_F=Constant(1.0),
        h_gamma=CellDiameter(),
        zeta=Constant(1.0),
        dt=Constant(1.0),
        g_sigma=g_sigma,
        g_sigma_n=g_sigma_n,
        g_n=g_n,
        g_t=g_t,
    )

    qdeg = 6
    compiler = FormCompiler(dh, quadrature_order=qdeg, backend=backend)
    _, R = compiler.assemble(Equation(None, forms.residual), bcs=[])

    def inf_norm(field: str) -> float:
        sl = np.asarray(dh.get_field_slice(field), dtype=int)
        return float(np.linalg.norm(R[sl], ord=np.inf)) if sl.size else 0.0

    porous_solid = max(inf_norm("v_neg_x"), inf_norm("v_neg_y"), inf_norm("u_neg_x"), inf_norm("u_neg_y"))
    fluid = max(inf_norm("v_pos_x"), inf_norm("v_pos_y"), inf_norm("p_pos_"))

    assert porous_solid < 1.0e-10
    assert fluid > 1.0e-8

