import math
import os

import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    dot,
    restrict,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.utils.fpi_fully_eulerian import build_fpi_eulerian_forms
from pycutfem.utils.fpi_mms_example41 import build_example41_mms
from pycutfem.utils.fsi_fully_eulerian import build_measures, make_domain_sets
from pycutfem.utils.meshgen import structured_quad


def _selected_backends(*, default: str = "jit") -> list[str]:
    import os

    spec = (os.environ.get("PYCUTFEM_TEST_BACKENDS") or os.environ.get("BACKEND") or default).strip()
    if not spec:
        return [default]
    if spec.lower() in {"all", "*"}:
        return ["python", "jit", "cpp"]
    backends = [b.strip() for b in spec.split(",") if b.strip()]
    valid = {"python", "jit", "cpp"}
    unknown = [b for b in backends if b not in valid]
    if unknown:
        raise ValueError(f"Unknown backend(s) {unknown}; valid={sorted(valid)}")
    return backends


def _tag_square_boundaries(mesh: Mesh, *, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x + 1.0) <= tol,
            "right": lambda x, y: abs(x - 1.0) <= tol,
            "bottom": lambda x, y: abs(y + 1.0) <= tol,
            "top": lambda x, y: abs(y - 1.0) <= tol,
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


def _pick_cut_element(mesh: Mesh) -> int:
    cut_ids = mesh.element_bitset("cut").to_indices()
    return int(cut_ids[0]) if cut_ids.size else 0


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
    functions_by_field: dict[str, Function | VectorFunction],
    probe_dofs: np.ndarray,
    *,
    compiler: FormCompiler,
    eps: float = 1.0e-7,
):
    eq_jac = Equation(jac_form, None)
    eq_res = Equation(None, res_form)
    base_K, _ = compiler.assemble(eq_jac, bcs=[])
    if base_K is None:
        raise AssertionError("Jacobian assembly returned None.")

    direction = np.zeros(dh.total_dofs, dtype=float)
    field_dofs: dict[str, list[int]] = {}
    for gdof in probe_dofs:
        gdof_i = int(gdof)
        field, _ = dh._dof_to_node_map[gdof_i]
        if field not in functions_by_field:
            continue
        direction[gdof_i] = 1.0
        field_dofs.setdefault(field, []).append(gdof_i)

    if not np.any(direction):
        raise AssertionError("Directional FD probe is empty.")

    base_vals: dict[str, np.ndarray] = {}
    for field, dofs in field_dofs.items():
        dof_arr = np.asarray(dofs, dtype=int)
        base_vals[field] = functions_by_field[field].get_nodal_values(dof_arr)

    def _set_vals(sign: float):
        for field, dofs in field_dofs.items():
            dof_arr = np.asarray(dofs, dtype=int)
            functions_by_field[field].set_nodal_values(dof_arr, base_vals[field] + sign * eps)

    _set_vals(+1.0)
    _, R_plus = compiler.assemble(eq_res, bcs=[])
    _set_vals(-1.0)
    _, R_minus = compiler.assemble(eq_res, bcs=[])
    _set_vals(0.0)

    fd_vec = (R_plus - R_minus) / (2 * eps)
    jac_vec = base_K.dot(direction)
    err_vec = fd_vec - jac_vec
    max_abs = float(np.linalg.norm(err_vec, ord=np.inf))
    mag = float(np.linalg.norm(jac_vec, ord=np.inf))
    max_rel = max_abs / (mag + 1.0e-14) if mag > 0.0 else 0.0
    return max_abs, max_rel


def _build_problem(*, nx: int, poly_order: int, dt_val: float, qdeg: int, kinv_case: str):
    # Mesh on [-1,1]^2 and a slightly shifted vertical interface x = -c0
    nodes, elems, edges, corners = structured_quad(2.0, 2.0, nx=nx, ny=nx, poly_order=poly_order, offset=(-1.0, -1.0))
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )
    _tag_square_boundaries(mesh)

    c0 = 0.13  # avoid aligned-interface degeneracy for nx=2^k refinements
    level_set = AffineLevelSet(a=1.0, b=0.0, c=c0)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)

    domains = make_domain_sets(mesh, use_aligned_interface=False)
    dx_f, dx_p, dGamma, dG_f, dG_p = build_measures(mesh, level_set, domains, qvol=qdeg)

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
            "p_neg_": poly_order,
        },
    )
    dh = DofHandler(me, method="cg")

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
    pP_k = Function(name="pP_k", field_name="p_neg_", dof_handler=dh, side="-")

    vF_n = VectorFunction(name="vF_n", field_names=["v_pos_x", "v_pos_y"], dof_handler=dh, side="+")
    pF_n = Function(name="pF_n", field_name="p_pos_", dof_handler=dh, side="+")
    vP_n = VectorFunction(name="vP_n", field_names=["v_neg_x", "v_neg_y"], dof_handler=dh, side="-")
    uP_n = VectorFunction(name="uP_n", field_names=["u_neg_x", "u_neg_y"], dof_handler=dh, side="-")
    pP_n = Function(name="pP_n", field_name="p_neg_", dof_handler=dh, side="-")

    for f in (vF_k, pF_k, vP_k, uP_k, pP_k, vF_n, pF_n, vP_n, uP_n, pP_n):
        f.nodal_values.fill(0.0)

    # Restrict to active element sets (CutFEM convention)
    has_pos = domains["has_pos"]
    has_neg = domains["has_neg"]

    prob = dict(
        mesh=mesh,
        level_set=level_set,
        domains=domains,
        dh=dh,
        me=me,
        dx_f=dx_f,
        dx_p=dx_p,
        dGamma=dGamma,
        dG_f=dG_f,
        dG_p=dG_p,
        vF_k=vF_k,
        pF_k=pF_k,
        vP_k=vP_k,
        uP_k=uP_k,
        pP_k=pP_k,
        vF_n=vF_n,
        pF_n=pF_n,
        vP_n=vP_n,
        uP_n=uP_n,
        pP_n=pP_n,
        vF_kR=restrict(vF_k, has_pos),
        pF_kR=restrict(pF_k, has_pos),
        vF_nR=restrict(vF_n, has_pos),
        pF_nR=restrict(pF_n, has_pos),
        vP_kR=restrict(vP_k, has_neg),
        uP_kR=restrict(uP_k, has_neg),
        pP_kR=restrict(pP_k, has_neg),
        vP_nR=restrict(vP_n, has_neg),
        uP_nR=restrict(uP_n, has_neg),
        pP_nR=restrict(pP_n, has_neg),
        dvF_R=restrict(dvF, has_pos),
        dpF_R=restrict(dpF, has_pos),
        dvP_R=restrict(dvP, has_neg),
        duP_R=restrict(duP, has_neg),
        dpP_R=restrict(dpP, has_neg),
        vF_testR=restrict(vF_test, has_pos),
        qF_testR=restrict(qF_test, has_pos),
        vP_testR=restrict(vP_test, has_neg),
        uP_testR=restrict(uP_test, has_neg),
        qP_testR=restrict(qP_test, has_neg),
        dt=Constant(float(dt_val)),
        kinv_case=kinv_case,
    )

    return prob


@pytest.mark.parametrize("backend", _selected_backends(default="python"))
@pytest.mark.parametrize("kinv_case", ["iso", "aniso"])
def test_fpi_example41_fd_consistency(backend: str, kinv_case: str):
    dt_val = 0.05
    poly_order = 1
    qdeg = 4 if backend == "python" else 6
    prob = _build_problem(nx=6, poly_order=poly_order, dt_val=dt_val, qdeg=qdeg, kinv_case=kinv_case)

    mms = build_example41_mms(dt_val=dt_val, kinv_case=kinv_case)
    t0, t1 = 0.0, dt_val
    ana_deg = max(8, qdeg + 2)

    # Put a non-trivial state into the dofs: exact BE data (t0 -> t1).
    prob["vF_n"].set_values_from_function(lambda x, y: mms.vF_n(x, y))
    prob["vP_n"].set_values_from_function(lambda x, y: mms.vP_n(x, y))
    prob["uP_n"].set_values_from_function(lambda x, y: mms.uP_n(x, y))
    prob["pF_n"].set_values_from_function(lambda x, y: mms.pF_n(x, y))
    prob["pP_n"].set_values_from_function(lambda x, y: mms.pP_n(x, y))

    prob["vF_k"].set_values_from_function(lambda x, y: mms.vF_k(x, y))
    prob["vP_k"].set_values_from_function(lambda x, y: mms.vP_k(x, y))
    prob["uP_k"].set_values_from_function(lambda x, y: mms.uP_k(x, y))
    prob["pF_k"].set_values_from_function(lambda x, y: mms.pF_k(x, y))
    prob["pP_k"].set_values_from_function(lambda x, y: mms.pP_k(x, y))

    # Material / coupling parameters
    rho_f = Constant(1.0)
    mu_f = Constant(1.0)
    rho_s0_tilde = Constant(1.0)
    porosity = Constant(0.5)
    K = 0.10
    K_inv = Constant(_kinv_matrix(kinv_case, K=K).tolist(), dim=2)
    K_inv._jit_name = "K_inv"
    E = 1000.0
    nu = 0.30
    mu_s = E / (2.0 * (1.0 + nu))
    c_nh = Constant(mu_s / 2.0)
    beta_nh = Constant(nu / (1.0 - nu))
    beta_BJ = Constant(1.0)
    kappa = Constant(math.sqrt(K) / (1.0 * float(mu_f.value) * math.sqrt(float(porosity.value))))

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
        level_set=prob["level_set"],
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
        gamma_n=Constant(10.0),
        gamma_t=Constant(10.0),
        zeta=1.0,
        gamma_F_p=10.0,
        gamma_F_gp=10.0,
        gamma_P_p=10.0,
        gamma_P_gp=10.0,
        g_sigma=Analytic(lambda x, y: mms.g_sigma(x, y), degree=ana_deg),
        g_sigma_n=Analytic(lambda x, y: mms.g_sigma_n(x, y), degree=ana_deg),
        g_n=Analytic(lambda x, y: mms.g_n(x, y), degree=ana_deg),
        g_t=Analytic(lambda x, y: mms.g_t(x, y), degree=ana_deg),
        use_interface_terms=True,
        use_stabilization=True,
    )

    # Add volume body forces (MMS)
    fF = Analytic(lambda x, y: mms.fF(x, y), degree=ana_deg)
    fD = Analytic(lambda x, y: mms.fD(x, y), degree=ana_deg)
    fS = Analytic(lambda x, y: mms.fS(x, y), degree=ana_deg)

    res = forms.residual_form - dot(fF, prob["vF_testR"]) * prob["dx_f"] - dot(fD, prob["vP_testR"]) * prob["dx_p"] - dot(fS, prob["uP_testR"]) * prob["dx_p"]

    compiler = FormCompiler(prob["dh"], qdeg, backend=backend)

    functions_by_field = {
        "v_pos_x": prob["vF_k"],
        "v_pos_y": prob["vF_k"],
        "p_pos_": prob["pF_k"],
        "v_neg_x": prob["vP_k"],
        "v_neg_y": prob["vP_k"],
        "u_neg_x": prob["uP_k"],
        "u_neg_y": prob["uP_k"],
        "p_neg_": prob["pP_k"],
    }

    eid = _pick_cut_element(prob["mesh"])
    probe = _select_fd_dofs(
        prob["dh"],
        {
            "v_pos_x": 3,
            "v_pos_y": 3,
            "p_pos_": 2,
            "v_neg_x": 3,
            "v_neg_y": 3,
            "u_neg_x": 3,
            "u_neg_y": 3,
            "p_neg_": 2,
        },
        eid=eid,
    )

    max_abs, max_rel = _fd_check(forms.jacobian_form, res, prob["dh"], functions_by_field, probe, compiler=compiler, eps=1.0e-7)
    print(f"[FPI Example 4.1 FD] backend={backend} K_inv={kinv_case}  abs={max_abs:.3e} rel={max_rel:.3e}")
    assert max_rel < (5.0e-5 if backend == "python" else 5.0e-6)


@pytest.mark.parametrize("backend", _selected_backends(default="jit"))
@pytest.mark.parametrize("kinv_case", ["iso", "aniso"])
@pytest.mark.parametrize("poly_order", [1, 2])
def test_fpi_example41_one_step_convergence(backend: str, kinv_case: str, poly_order: int):
    if str(os.environ.get("PYCUTFEM_RUN_SLOW", "")).strip().lower() not in {"1", "true", "yes"}:
        pytest.skip("Slow MMS solve; set PYCUTFEM_RUN_SLOW=1 to enable.")
    dt_val = 0.05
    qdeg = max(4, 2 * poly_order + 2) if backend != "python" else max(3, 2 * poly_order + 1)

    # Two mesh levels; keep python backend smaller.
    if backend == "python":
        nx_list = [2, 4]
    else:
        nx_list = [2, 4] if poly_order >= 2 else [4, 8]

    mms = build_example41_mms(dt_val=dt_val, kinv_case=kinv_case)
    ana_deg = max(10, qdeg + 4)

    errs = []
    hs = []

    for nx in nx_list:
        prob = _build_problem(nx=nx, poly_order=poly_order, dt_val=dt_val, qdeg=qdeg, kinv_case=kinv_case)

        # Parameters (match MMS builder)
        rho_f = Constant(1.0)
        mu_f = Constant(1.0)
        rho_s0_tilde = Constant(1.0)
        porosity = Constant(0.5)
        K = 0.10
        K_inv = Constant(_kinv_matrix(kinv_case, K=K).tolist(), dim=2)
        K_inv._jit_name = "K_inv"
        E = 1000.0
        nu = 0.30
        mu_s = E / (2.0 * (1.0 + nu))
        c_nh = Constant(mu_s / 2.0)
        beta_nh = Constant(nu / (1.0 - nu))
        beta_BJ = Constant(1.0)
        kappa = Constant(math.sqrt(K) / (1.0 * float(mu_f.value) * math.sqrt(float(porosity.value))))

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
            level_set=prob["level_set"],
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
            gamma_n=Constant(10.0),
            gamma_t=Constant(10.0),
            zeta=1.0,
            gamma_F_p=0.0,
            gamma_F_gp=10.0,
            gamma_P_p=0.0,
            gamma_P_gp=10.0,
            g_sigma=Analytic(lambda x, y: mms.g_sigma(x, y), degree=ana_deg),
            g_sigma_n=Analytic(lambda x, y: mms.g_sigma_n(x, y), degree=ana_deg),
            g_n=Analytic(lambda x, y: mms.g_n(x, y), degree=ana_deg),
            g_t=Analytic(lambda x, y: mms.g_t(x, y), degree=ana_deg),
            use_interface_terms=True,
            use_stabilization=True,
        )

        fF = Analytic(lambda x, y: mms.fF(x, y), degree=ana_deg)
        fD = Analytic(lambda x, y: mms.fD(x, y), degree=ana_deg)
        fS = Analytic(lambda x, y: mms.fS(x, y), degree=ana_deg)

        residual_form = (
            forms.residual_form
            - dot(fF, prob["vF_testR"]) * prob["dx_f"]
            - dot(fD, prob["vP_testR"]) * prob["dx_p"]
            - dot(fS, prob["uP_testR"]) * prob["dx_p"]
        )

        # Dirichlet BCs: impose the analytic state at t=t_{n+1}=dt (including pressure to fix the constant mode)
        bcs = []
        for tag in ("left", "right", "bottom", "top"):
            bcs.extend(
                [
                    BoundaryCondition("v_pos_x", "dirichlet", tag, mms.vF_x),
                    BoundaryCondition("v_pos_y", "dirichlet", tag, mms.vF_y),
                    BoundaryCondition("p_pos_", "dirichlet", tag, mms.pF_s),
                    BoundaryCondition("v_neg_x", "dirichlet", tag, mms.vP_x),
                    BoundaryCondition("v_neg_y", "dirichlet", tag, mms.vP_y),
                    BoundaryCondition("u_neg_x", "dirichlet", tag, mms.uP_x),
                    BoundaryCondition("u_neg_y", "dirichlet", tag, mms.uP_y),
                    BoundaryCondition("p_neg_", "dirichlet", tag, mms.pP_s),
                ]
            )
        bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]

        solver = NewtonSolver(
            residual_form,
            forms.jacobian_form,
            dof_handler=prob["dh"],
            mixed_element=prob["me"],
            bcs=bcs,
            bcs_homog=bcs_homog,
            newton_params=NewtonParameters(newton_tol=1.0e-10, max_newton_iter=12),
            quad_order=qdeg,
            backend=backend,
        )

        # Initial conditions at t_n=0 and initial guess
        prob["vF_n"].set_values_from_function(lambda x, y: mms.vF_n(x, y))
        prob["vP_n"].set_values_from_function(lambda x, y: mms.vP_n(x, y))
        prob["uP_n"].set_values_from_function(lambda x, y: mms.uP_n(x, y))
        prob["pF_n"].set_values_from_function(lambda x, y: mms.pF_n(x, y))
        prob["pP_n"].set_values_from_function(lambda x, y: mms.pP_n(x, y))

        prob["vF_k"].nodal_values[:] = prob["vF_n"].nodal_values
        prob["vP_k"].nodal_values[:] = prob["vP_n"].nodal_values
        prob["uP_k"].nodal_values[:] = prob["uP_n"].nodal_values
        prob["pF_k"].nodal_values[:] = prob["pF_n"].nodal_values
        prob["pP_k"].nodal_values[:] = prob["pP_n"].nodal_values

        solver.solve_time_interval(
            functions=[prob["vF_k"], prob["pF_k"], prob["vP_k"], prob["uP_k"], prob["pP_k"]],
            prev_functions=[prob["vF_n"], prob["pF_n"], prob["vP_n"], prob["uP_n"], prob["pP_n"]],
            aux_functions={"dt": prob["dt"]},
            time_params=TimeStepperParameters(dt=dt_val, final_time=dt_val, max_steps=1),
        )

        # Relative L2 errors on each side
        dh: DofHandler = prob["dh"]
        level_set = prob["level_set"]

        err_vF = dh.l2_error_on_side(
            functions=prob["vF_k"],
            exact={"v_pos_x": mms.vF_x, "v_pos_y": mms.vF_y},
            fields=["v_pos_x", "v_pos_y"],
            level_set=level_set,
            side="+",
            quad_order=qdeg,
            relative=True,
        )
        err_pF = dh.l2_error_on_side(
            functions=prob["pF_k"],
            exact={"p_pos_": mms.pF_s},
            fields=["p_pos_"],
            level_set=level_set,
            side="+",
            quad_order=qdeg,
            relative=True,
        )
        err_vP = dh.l2_error_on_side(
            functions=prob["vP_k"],
            exact={"v_neg_x": mms.vP_x, "v_neg_y": mms.vP_y},
            fields=["v_neg_x", "v_neg_y"],
            level_set=level_set,
            side="-",
            quad_order=qdeg,
            relative=True,
        )
        err_uP = dh.l2_error_on_side(
            functions=prob["uP_k"],
            exact={"u_neg_x": mms.uP_x, "u_neg_y": mms.uP_y},
            fields=["u_neg_x", "u_neg_y"],
            level_set=level_set,
            side="-",
            quad_order=qdeg,
            relative=True,
        )
        err_pP = dh.l2_error_on_side(
            functions=prob["pP_k"],
            exact={"p_neg_": mms.pP_s},
            fields=["p_neg_"],
            level_set=level_set,
            side="-",
            quad_order=qdeg,
            relative=True,
        )

        errs.append((err_vF, err_pF, err_vP, err_uP, err_pP))
        hs.append(2.0 / float(nx))

    # Print convergence table (requested with `pytest -s`)
    def _eoc(prev: float, curr: float) -> float:
        if prev <= 0.0 or curr <= 0.0:
            return float("nan")
        return math.log(prev / curr, 2.0)

    def _fmt_eoc(val: float) -> str:
        return "   - " if not np.isfinite(val) else f"{val:6.2f}"

    title = f"FPI Example 4.1 | backend={backend} | p={poly_order} | K_inv={kinv_case}"
    print("\n" + title)
    print(
        f"{'h':>8}  {'|e(vF)|':>12} {'eoc':>6}  {'|e(pF)|':>12} {'eoc':>6}  {'|e(vP)|':>12} {'eoc':>6}  {'|e(uP)|':>12} {'eoc':>6}  {'|e(pP)|':>12} {'eoc':>6}"
    )
    for i, (h, (e1, e2, e3, e4, e5)) in enumerate(zip(hs, errs, strict=True)):
        if i == 0:
            eoc1 = eoc2 = eoc3 = eoc4 = eoc5 = float("nan")
        else:
            p1, p2, p3, p4, p5 = errs[i - 1]
            eoc1 = _eoc(p1, e1)
            eoc2 = _eoc(p2, e2)
            eoc3 = _eoc(p3, e3)
            eoc4 = _eoc(p4, e4)
            eoc5 = _eoc(p5, e5)
        print(
            f"{h:8.3e}  {e1:12.3e} {_fmt_eoc(eoc1)}  {e2:12.3e} {_fmt_eoc(eoc2)}  {e3:12.3e} {_fmt_eoc(eoc3)}  {e4:12.3e} {_fmt_eoc(eoc4)}  {e5:12.3e} {_fmt_eoc(eoc5)}"
        )

    # Basic sanity: overall error should decrease under refinement.
    coarse = np.array(errs[0], dtype=float)
    fine = np.array(errs[-1], dtype=float)
    assert np.all(np.isfinite(coarse)) and np.all(np.isfinite(fine))
    assert np.linalg.norm(fine) <= 1.05 * np.linalg.norm(coarse) + 1.0e-14
