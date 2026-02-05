import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dS, dx
from pycutfem.utils.biofilm_one_domain import build_biofilm_one_domain_forms
from pycutfem.utils.meshgen import structured_quad


def _tag_unit_square_boundaries(mesh: Mesh, *, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - 1.0) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - 1.0) <= tol,
        }
    )


def test_damage_degrades_wall_adhesion_traction():
    """
    If bulk damage is enabled, the wall adhesion traction in the skeleton block is scaled by
    g_stiff(d)=(1-d)^2+κ. This test checks that the assembled *adhesion contribution* to the
    u-residual is reduced by ~κ when d goes from 0 -> 1.
    """
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=2)
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
            "d": 1,
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
    dd = TrialFunction("d", dof_handler=dh)
    dS_trial = TrialFunction("S", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    d_test = TestFunction("d", dof_handler=dh)
    S_test = TestFunction("S", dof_handler=dh)

    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    d_k = Function("d_k", "d", dof_handler=dh)
    S_k = Function("S_k", "S", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    d_n = Function("d_n", "d", dof_handler=dh)
    S_n = Function("S_n", "S", dof_handler=dh)

    # Simple nonzero displacement so the wall traction is nonzero (bottom y=0 must have u≠0).
    u_k.set_values_from_function(lambda x, y: np.array([0.0, 0.1], dtype=float))
    u_n.set_values_from_function(lambda x, y: np.array([0.0, 0.0], dtype=float))

    # Keep everything else simple.
    v_k.set_values_from_function(lambda x, y: np.array([0.0, 0.0], dtype=float))
    v_n.set_values_from_function(lambda x, y: np.array([0.0, 0.0], dtype=float))
    p_k.set_values_from_function(lambda x, y: 0.0)
    p_n.set_values_from_function(lambda x, y: 0.0)
    phi_k.set_values_from_function(lambda x, y: 0.5)
    phi_n.set_values_from_function(lambda x, y: 0.5)
    alpha_k.set_values_from_function(lambda x, y: 1.0)
    alpha_n.set_values_from_function(lambda x, y: 1.0)
    S_k.set_values_from_function(lambda x, y: 0.0)
    S_n.set_values_from_function(lambda x, y: 0.0)

    ds_bottom = dS(defined_on=mesh.edge_bitset("bottom"), metadata={"q": 4})
    dx_q = dx(metadata={"q": 4})

    kappa_stiff = 1.0e-3

    def _assemble_u_residual_with_damage(d_val: float, *, with_adhesion: bool) -> np.ndarray:
        d_k.set_values_from_function(lambda x, y: float(d_val))
        d_n.set_values_from_function(lambda x, y: float(d_val))

        forms = build_biofilm_one_domain_forms(
            v_k=v_k,
            p_k=p_k,
            u_k=u_k,
            phi_k=phi_k,
            alpha_k=alpha_k,
            d_k=d_k,
            S_k=S_k,
            v_n=v_n,
            p_n=p_n,
            u_n=u_n,
            phi_n=phi_n,
            alpha_n=alpha_n,
            d_n=d_n,
            S_n=S_n,
            dv=dv,
            dp=dp,
            du=du,
            dphi=dphi,
            dalpha=dalpha,
            dd=dd,
            dS=dS_trial,
            v_test=v_test,
            q_test=q_test,
            u_test=u_test,
            phi_test=phi_test,
            alpha_test=alpha_test,
            d_test=d_test,
            S_test=S_test,
            dx=dx_q,
            dt=Constant(1.0),
            theta=1.0,
            rho_f=Constant(1.0),
            mu_f=Constant(1.0),
            kappa_inv=Constant(1.0),
            mu_s=Constant(1.0),
            lambda_s=Constant(1.0),
            # Keep transport/growth inactive.
            D_phi=0.0,
            gamma_phi=0.0,
            D_alpha=0.0,
            D_S=0.0,
            # Enable damage coupling only through g_stiff/g_perm.
            damage_k=0.0,
            damage_sigma_cr=0.0,
            damage_m=1.0,
            damage_D=0.0,
            damage_gamma_out=0.0,
            damage_kappa_stiff=kappa_stiff,
            damage_kappa_perm=1.0,
            # Wall adhesion (spring only).
            ds_adh=ds_bottom if with_adhesion else None,
            adhesion_k_n=1.0 if with_adhesion else 0.0,
            adhesion_k_t=0.0,
            adhesion_gamma_n=0.0,
            adhesion_gamma_t=0.0,
            adhesion_a_prev=Constant(1.0),
        )
        _, R = assemble_form(Equation(None, forms.residual_form), dof_handler=dh, bcs=[], quad_order=4, backend="python")
        R = np.asarray(R, dtype=float)
        u_sl = np.asarray(dh.get_field_slice("u_x") + dh.get_field_slice("u_y"), dtype=int)
        return R[u_sl]

    # Adhesion contribution is the difference (with - without ds_adh).
    R_u_d0_on = _assemble_u_residual_with_damage(0.0, with_adhesion=True)
    R_u_d0_off = _assemble_u_residual_with_damage(0.0, with_adhesion=False)
    adh_d0 = R_u_d0_on - R_u_d0_off

    R_u_d1_on = _assemble_u_residual_with_damage(1.0, with_adhesion=True)
    R_u_d1_off = _assemble_u_residual_with_damage(1.0, with_adhesion=False)
    adh_d1 = R_u_d1_on - R_u_d1_off

    n0 = float(np.linalg.norm(adh_d0))
    n1 = float(np.linalg.norm(adh_d1))
    assert n0 > 0.0
    ratio = n1 / n0
    expected = kappa_stiff / (1.0 + kappa_stiff)
    assert ratio > 0.2 * expected
    assert ratio < 5.0 * expected
