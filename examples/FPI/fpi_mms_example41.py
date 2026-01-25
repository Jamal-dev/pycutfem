import argparse
import math

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction, dot, restrict
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.utils.fpi_fully_eulerian import build_fpi_eulerian_forms
from pycutfem.utils.fpi_mms_example41 import build_example41_mms
from pycutfem.utils.fsi_fully_eulerian import build_measures, make_domain_sets
from pycutfem.utils.meshgen import structured_quad


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


def _build_problem(*, nx: int, poly_order: int, qdeg: int, dt_val: float, kinv_case: str):
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

    # Slightly shifted vertical interface x=-c0 to avoid aligned-interface degeneracy.
    c0 = 0.13
    level_set = AffineLevelSet(a=1.0, b=0.0, c=c0)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)

    domains = make_domain_sets(mesh, use_aligned_interface=False)
    dx_f, dx_p, dGamma, dG_f, dG_p = build_measures(mesh, level_set, domains, qvol=qdeg)

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

    has_pos = domains["has_pos"]
    has_neg = domains["has_neg"]

    return dict(
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


def _run_one(*, nx: int, poly_order: int, qdeg: int, dt_val: float, backend: str, kinv_case: str):
    prob = _build_problem(nx=nx, poly_order=poly_order, qdeg=qdeg, dt_val=dt_val, kinv_case=kinv_case)
    mms = build_example41_mms(dt_val=dt_val, kinv_case=kinv_case)
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

    # Dirichlet BCs at t_{n+1}=dt (including pressure to fix the constant mode)
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

    return dict(h=2.0 / float(nx), err_vF=err_vF, err_pF=err_pF, err_vP=err_vP, err_uP=err_uP, err_pP=err_pP)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="jit", choices=["python", "jit", "cpp"])
    parser.add_argument("--kinv", type=str, default="iso", choices=["iso", "aniso"])
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--convergence", action="store_true", help="Run a 2-level h-refinement study.")
    args = parser.parse_args()

    if not args.convergence:
        out = _run_one(nx=args.nx, poly_order=args.p, qdeg=args.q, dt_val=args.dt, backend=args.backend, kinv_case=args.kinv)
        print(
            f"h={out['h']:.3e}  |e(vF)|={out['err_vF']:.3e}  |e(pF)|={out['err_pF']:.3e}  "
            f"|e(vP)|={out['err_vP']:.3e}  |e(uP)|={out['err_uP']:.3e}  |e(pP)|={out['err_pP']:.3e}"
        )
        return

    nx_list = [args.nx, 2 * args.nx]
    rows = [
        _run_one(nx=nx, poly_order=args.p, qdeg=args.q, dt_val=args.dt, backend=args.backend, kinv_case=args.kinv)
        for nx in nx_list
    ]
    print(f"\nFPI Example 4.1 (BE) | backend={args.backend} | p={args.p} | K_inv={args.kinv}")
    print(
        f"{'h':>8}  {'|e(vF)|':>12} {'eoc':>6}  {'|e(pF)|':>12} {'eoc':>6}  {'|e(vP)|':>12} {'eoc':>6}  {'|e(uP)|':>12} {'eoc':>6}  {'|e(pP)|':>12} {'eoc':>6}"
    )
    for i, r in enumerate(rows):
        if i == 0:
            eocs = [float("nan")] * 5
        else:
            prev = rows[i - 1]
            def _eoc(pe, ce):
                if pe <= 0.0 or ce <= 0.0:
                    return float("nan")
                return math.log(pe / ce, 2.0)
            eocs = [
                _eoc(prev["err_vF"], r["err_vF"]),
                _eoc(prev["err_pF"], r["err_pF"]),
                _eoc(prev["err_vP"], r["err_vP"]),
                _eoc(prev["err_uP"], r["err_uP"]),
                _eoc(prev["err_pP"], r["err_pP"]),
            ]
        def _fmt(val):
            return "   - " if not np.isfinite(val) else f"{val:6.2f}"
        print(
            f"{r['h']:8.3e}  {r['err_vF']:12.3e} {_fmt(eocs[0])}  {r['err_pF']:12.3e} {_fmt(eocs[1])}  "
            f"{r['err_vP']:12.3e} {_fmt(eocs[2])}  {r['err_uP']:12.3e} {_fmt(eocs[3])}  {r['err_pP']:12.3e} {_fmt(eocs[4])}"
        )


if __name__ == "__main__":
    main()
