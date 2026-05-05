import pytest

# This repo's biofilm model depends on optional scientific-stack packages (e.g. sympy)
# that may not be installed in minimal Python environments. Skip this unit test if
# those deps are unavailable so `pytest` remains usable outside the FEniCSx env.
pytest.importorskip("sympy")
pytest.importorskip("pycutfem")


def _build_minimal_one_domain_inputs():
    import numpy as np

    from pycutfem.core.dofhandler import DofHandler
    from pycutfem.core.mesh import Mesh
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction
    from pycutfem.ufl.measures import dx
    from pycutfem.ufl.spaces import FunctionSpace
    from pycutfem.utils.meshgen import structured_quad

    L = 1.0
    H = 1.0
    nodes, elems, _edges, corners = structured_quad(L, H, nx=1, ny=1, poly_order=2, offset=(0.0, 0.0))
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=2)

    field_specs = {
        "v_x": 2,
        "v_y": 2,
        "p": 1,
        "vS_x": 2,
        "vS_y": 2,
        "u_x": 2,
        "u_y": 2,
        "phi": 1,
        "alpha": 1,
        "S": 1,
    }
    me = MixedElement(mesh, field_specs=field_specs)
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    VS = FunctionSpace("VS", ["vS_x", "vS_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    dv = VectorTrialFunction(space=V, dof_handler=dh)
    dvS = VectorTrialFunction(space=VS, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    dphi = TrialFunction("phi", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    dS = TrialFunction("S", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    vS_test = VectorTestFunction(space=VS, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    S_test = TestFunction("S", dof_handler=dh)

    def _vf(name, fields):
        f = VectorFunction(name, fields, dof_handler=dh)
        f.nodal_values[:] = 0.0
        return f

    def _sf(name, field):
        f = Function(name, field, dof_handler=dh)
        f.nodal_values[:] = 0.0
        return f

    v_k = _vf("v_k", ["v_x", "v_y"])
    p_k = _sf("p_k", "p")
    vS_k = _vf("vS_k", ["vS_x", "vS_y"])
    u_k = _vf("u_k", ["u_x", "u_y"])
    phi_k = _sf("phi_k", "phi")
    alpha_k = _sf("alpha_k", "alpha")
    S_k = _sf("S_k", "S")

    v_n = _vf("v_n", ["v_x", "v_y"])
    p_n = _sf("p_n", "p")
    vS_n = _vf("vS_n", ["vS_x", "vS_y"])
    u_n = _vf("u_n", ["u_x", "u_y"])
    phi_n = _sf("phi_n", "phi")
    alpha_n = _sf("alpha_n", "alpha")
    S_n = _sf("S_n", "S")

    return dict(
        # unknowns at k/n
        v_k=v_k,
        p_k=p_k,
        vS_k=vS_k,
        u_k=u_k,
        phi_k=phi_k,
        alpha_k=alpha_k,
        S_k=S_k,
        v_n=v_n,
        p_n=p_n,
        vS_n=vS_n,
        u_n=u_n,
        phi_n=phi_n,
        alpha_n=alpha_n,
        S_n=S_n,
        # trials
        dv=dv,
        dp=dp,
        dvS=dvS,
        du=du,
        dphi=dphi,
        dalpha=dalpha,
        dS=dS,
        # tests
        v_test=v_test,
        q_test=q_test,
        vS_test=vS_test,
        u_test=u_test,
        phi_test=phi_test,
        alpha_test=alpha_test,
        S_test=S_test,
        # measure + dt
        dx=dx(metadata={"q": 1}),
        dt=Constant(1.0),
        # physical parameters (minimal)
        rho_f=Constant(0.0),
        mu_f=Constant(1.0),
        kappa_inv=Constant(1.0),
        mu_s=Constant(1.0e-8),
        lambda_s=Constant(1.0e-8),
        dim=2,
    )


def test_alpha_cahn_deg_mobility_floor_validation():
    from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms

    base = _build_minimal_one_domain_inputs()

    # Positive floor should be accepted.
    build_biofilm_one_domain_forms(
        **base,
        theta=1.0,
        alpha_cahn_M=1.0,
        alpha_cahn_gamma=1.0,
        alpha_cahn_eps=0.1,
        alpha_cahn_mobility="degenerate",
        alpha_cahn_mobility_floor=0.05,
    )

    # Negative floor must be rejected.
    with pytest.raises(ValueError):
        build_biofilm_one_domain_forms(
            **base,
            theta=1.0,
            alpha_cahn_M=1.0,
            alpha_cahn_gamma=1.0,
            alpha_cahn_eps=0.1,
            alpha_cahn_mobility="degenerate",
            alpha_cahn_mobility_floor=-0.01,
        )
