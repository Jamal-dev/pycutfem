import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.measures import dx
from examples.utils.biofilm.mms_one_domain import build_biofilm_one_domain_mms_affine
from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms
from pycutfem.utils.meshgen import structured_quad


BACKENDS = ("python", "cpp")


def _tag_unit_square_boundaries(mesh: Mesh, *, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - 1.0) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - 1.0) <= tol,
        }
    )


def _set_field(dh: DofHandler, field: str, values: np.ndarray, *, func) -> None:
    gdofs = np.asarray(dh.get_field_slice(field), dtype=int)
    func.set_nodal_values(gdofs, np.asarray(values, dtype=float))


def _residual_inf(dh: DofHandler, res_form, bcs, *, backend: str, quad_order: int) -> float:
    _, F = assemble_form(Equation(None, res_form), dof_handler=dh, bcs=[], quad_order=int(quad_order), backend=backend)
    F = np.asarray(F, dtype=float)

    dirichlet = dh.get_dirichlet_data(bcs) or {}
    bc_rows = np.fromiter(dirichlet.keys(), dtype=int) if dirichlet else np.zeros((0,), dtype=int)
    inactive = set(getattr(dh, "dof_tags", {}).get("inactive", set()))

    mask = np.ones(dh.total_dofs, dtype=bool)
    if bc_rows.size:
        mask[bc_rows] = False
    if inactive:
        mask[np.fromiter(inactive, dtype=int)] = False
    if not np.any(mask):
        return 0.0
    return float(np.linalg.norm(F[mask], ord=np.inf))


def _build_problem(*, nx: int, ny: int, q: int):
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    _tag_unit_square_boundaries(mesh)

    me = MixedElement(
        mesh,
        field_specs={
            "v_x": 2,
            "v_y": 2,
            "p": 1,
            "u_x": 2,
            "u_y": 2,
            "phi": 1,
            "alpha": 1,
            "S": 1,
        },
    )
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    dv = VectorTrialFunction(space=V, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    dphi = TrialFunction("phi", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    dS = TrialFunction("S", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    S_test = TestFunction("S", dof_handler=dh)

    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    S_k = Function("S_k", "S", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    S_n = Function("S_n", "S", dof_handler=dh)

    return {
        "mesh": mesh,
        "dh": dh,
        "me": me,
        "q": int(q),
        "dv": dv,
        "dp": dp,
        "du": du,
        "dphi": dphi,
        "dalpha": dalpha,
        "dS": dS,
        "v_test": v_test,
        "q_test": q_test,
        "u_test": u_test,
        "phi_test": phi_test,
        "alpha_test": alpha_test,
        "S_test": S_test,
        "v_k": v_k,
        "p_k": p_k,
        "u_k": u_k,
        "phi_k": phi_k,
        "alpha_k": alpha_k,
        "S_k": S_k,
        "v_n": v_n,
        "p_n": p_n,
        "u_n": u_n,
        "phi_n": phi_n,
        "alpha_n": alpha_n,
        "S_n": S_n,
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_biofilm_one_domain_mms_residual_zero_affine(backend):
    dt_val = 0.1
    prob = _build_problem(nx=2, ny=2, q=6)
    dh: DofHandler = prob["dh"]

    mms = build_biofilm_one_domain_mms_affine(dt_val=dt_val)

    # Populate time levels from the MMS.
    for field, fn in (("v_x", mms.v_x), ("v_y", mms.v_y), ("u_x", mms.u_x), ("u_y", mms.u_y)):
        xy = dh.get_dof_coords(field)
        vals = fn(xy[:, 0], xy[:, 1])
        if field.startswith("v_"):
            comp = 0 if field.endswith("x") else 1
            _set_field(dh, field, vals, func=prob["v_k"].components[comp])
        else:
            comp = 0 if field.endswith("x") else 1
            _set_field(dh, field, vals, func=prob["u_k"].components[comp])

    for fld, fnk, fnn, fk, fn in (
        ("p", mms.p_k, mms.p_n, prob["p_k"], prob["p_n"]),
        ("phi", mms.phi_k, mms.phi_n, prob["phi_k"], prob["phi_n"]),
        ("alpha", mms.alpha_k, mms.alpha_n, prob["alpha_k"], prob["alpha_n"]),
        ("S", mms.S_k, mms.S_n, prob["S_k"], prob["S_n"]),
    ):
        xy = dh.get_dof_coords(fld)
        _set_field(dh, fld, fnk(xy[:, 0], xy[:, 1]), func=fk)
        _set_field(dh, fld, fnn(xy[:, 0], xy[:, 1]), func=fn)

    # v_n and u_n are zero in this MMS.
    prob["v_n"].nodal_values.fill(0.0)
    prob["u_n"].nodal_values.fill(0.0)

    # Forcing terms (strong-form BE step).
    f_v = Analytic(lambda x, y: mms.f_v(x, y), degree=2)
    f_u = Analytic(lambda x, y: mms.f_u(x, y), degree=2)
    s_v = Analytic(lambda x, y: mms.s_v(x, y), degree=2)
    f_phi = Analytic(lambda x, y: mms.f_phi(x, y), degree=2)
    f_alpha = Analytic(lambda x, y: mms.f_alpha(x, y), degree=2)
    f_S = Analytic(lambda x, y: mms.f_S(x, y), degree=2)

    forms = build_biofilm_one_domain_forms(
        v_k=prob["v_k"],
        p_k=prob["p_k"],
        u_k=prob["u_k"],
        phi_k=prob["phi_k"],
        alpha_k=prob["alpha_k"],
        S_k=prob["S_k"],
        v_n=prob["v_n"],
        p_n=prob["p_n"],
        u_n=prob["u_n"],
        phi_n=prob["phi_n"],
        alpha_n=prob["alpha_n"],
        S_n=prob["S_n"],
        dv=prob["dv"],
        dp=prob["dp"],
        du=prob["du"],
        dphi=prob["dphi"],
        dalpha=prob["dalpha"],
        dS=prob["dS"],
        v_test=prob["v_test"],
        q_test=prob["q_test"],
        u_test=prob["u_test"],
        phi_test=prob["phi_test"],
        alpha_test=prob["alpha_test"],
        S_test=prob["S_test"],
        dx=dx(metadata={"q": int(prob["q"])}),
        dt=Constant(dt_val),
        theta=1.0,
        rho_f=Constant(1.0),
        mu_f=Constant(1.0e-2),
        kappa_inv=Constant(10.0),
        mu_s=Constant(1.0),
        lambda_s=Constant(1.0),
        D_phi=0.1,
        gamma_phi=1.0,
        D_alpha=0.1,
        D_S=0.1,
        mu_max=0.4,
        K_S=0.3,
        k_g=0.5,
        k_d=0.1,
        Y=0.8,
        k_det=0.2,
        f_v=f_v,
        f_u=f_u,
        s_v=s_v,
        f_phi=f_phi,
        f_alpha=f_alpha,
        f_S=f_S,
    )

    # Dirichlet BCs at t_{n+1} (masked from the residual check).
    def _as_float(fn):
        return lambda x, y: float(np.asarray(fn(np.asarray(x), np.asarray(y))).reshape(()))

    bcs = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("v_x", "dirichlet", tag, _as_float(mms.v_x)),
                BoundaryCondition("v_y", "dirichlet", tag, _as_float(mms.v_y)),
                BoundaryCondition("u_x", "dirichlet", tag, _as_float(mms.u_x)),
                BoundaryCondition("u_y", "dirichlet", tag, _as_float(mms.u_y)),
                BoundaryCondition("phi", "dirichlet", tag, _as_float(mms.phi)),
                BoundaryCondition("alpha", "dirichlet", tag, _as_float(mms.alpha)),
                BoundaryCondition("S", "dirichlet", tag, _as_float(mms.S)),
            ]
        )

    res = _residual_inf(dh, forms.residual_form, bcs, backend=backend, quad_order=int(prob["q"]))
    assert res < 1.0e-9

