import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.nonmatching import (
    build_nonmatching_interface,
    poisson_flux_mismatch_L2,
    scalar_H1_semi_error,
    scalar_L2_error,
    scalar_jump_L2,
)
from pycutfem.nonmatching.diagnostics import stokes_traction_mismatch_L2, stokes_velocity_jump_L2
from pycutfem.nonmatching.mortar import assemble_mortar_saddle_matrix, assemble_poisson_mortar_coupling
from pycutfem.nonmatching.nitsche import (
    assemble_poisson_nitsche_interface_matrix,
    assemble_stokes_nitsche_interface_matrix,
)
from pycutfem.nonmatching.system import apply_dirichlet_data, coupled_dirichlet_data
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import (
    Constant,
    TestFunction as UFLTestFunction,
    TrialFunction,
    VectorTestFunction,
    VectorTrialFunction,
    div,
    dot,
    grad,
    inner,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad, structured_triangles


def _make_submesh(*, element: str, poly_order: int, nx: int, ny: int, offset_x: float) -> Mesh:
    if element == "quad":
        nodes, elems, edges, corners = structured_quad(
            0.5, 1.0, nx=nx, ny=ny, poly_order=poly_order, offset=(offset_x, 0.0)
        )
        return Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=poly_order)
    if element == "tri":
        nodes, elems, edges, corners = structured_triangles(
            0.5, 1.0, nx_quads=nx, ny_quads=ny, poly_order=poly_order, offset=(offset_x, 0.0)
        )
        return Mesh(nodes, elems, edges, corners, element_type="tri", poly_order=poly_order)
    raise ValueError("element must be 'quad' or 'tri'")


def _solve_poisson_nitsche(
    *,
    element: str,
    degree: int,
    ny_neg: int,
    ny_pos: int,
    k_neg: float,
    k_pos: float,
    gamma: float,
    u_neg,
    grad_u_neg,
    u_pos,
    grad_u_pos,
    f,
    backend_volume: str = "python",
    backend_interface: str = "python",
):
    nx_neg = max(2, int(round(0.5 * ny_neg)))
    nx_pos = max(2, int(round(0.5 * ny_pos)))
    mesh_neg = _make_submesh(element=element, poly_order=degree, nx=nx_neg, ny=ny_neg, offset_x=0.0)
    mesh_pos = _make_submesh(element=element, poly_order=degree, nx=nx_pos, ny=ny_pos, offset_x=0.5)

    mesh_neg.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})
    mesh_pos.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})

    dh_neg = DofHandler(MixedElement(mesh_neg, {"u": int(degree)}), method="cg")
    dh_pos = DofHandler(MixedElement(mesh_pos, {"u": int(degree)}), method="cg")

    f_expr = Analytic(f, degree=max(4, 2 * degree + 2))

    def assemble_side(dh: DofHandler, k: float):
        u = TrialFunction(name="u_trial", field_name="u", dof_handler=dh)
        v = UFLTestFunction(name="v_test", field_name="u", dof_handler=dh)
        a = Constant(float(k)) * inner(grad(u), grad(v)) * dx()
        L = f_expr * v * dx()
        return assemble_form(
            Equation(a, L), dof_handler=dh, bcs=[], quad_order=int(2 * degree + 2), backend=backend_volume
        )

    K_pos, F_pos = assemble_side(dh_pos, float(k_pos))
    K_neg, F_neg = assemble_side(dh_neg, float(k_neg))

    interface = build_nonmatching_interface(mesh_neg=mesh_neg, mesh_pos=mesh_pos)
    K_if = assemble_poisson_nitsche_interface_matrix(
        interface=interface,
        dh_neg=dh_neg,
        dh_pos=dh_pos,
        k_neg=float(k_neg),
        k_pos=float(k_pos),
        gamma=float(gamma),
        backend=backend_interface,
    )

    n_pos = int(dh_pos.total_dofs)
    K = sp.block_diag([K_pos.tocsr(), K_neg.tocsr()], format="csr") + K_if
    F = np.concatenate([np.asarray(F_pos, float), np.asarray(F_neg, float)])

    bcs_pos = [BoundaryCondition("u", "dirichlet", "boundary", lambda xx, yy: float(u_pos(xx, yy)))]
    bcs_neg = [BoundaryCondition("u", "dirichlet", "boundary", lambda xx, yy: float(u_neg(xx, yy)))]
    bc_data = coupled_dirichlet_data(dh_pos=dh_pos, bcs_pos=bcs_pos, dh_neg=dh_neg, bcs_neg=bcs_neg, neg_offset=n_pos)
    K_bc, F_bc = apply_dirichlet_data(K, F, bc_data)

    sol = spla.spsolve(K_bc.tocsc(), F_bc)
    U_pos = sol[:n_pos]
    U_neg = sol[n_pos:]
    return U_pos, U_neg, dh_pos, dh_neg, interface


def _solve_poisson_mortar(
    *,
    element: str,
    degree: int,
    ny_neg: int,
    ny_pos: int,
    k_neg: float,
    k_pos: float,
    u_neg,
    grad_u_neg,
    u_pos,
    grad_u_pos,
    f,
    backend_volume: str = "python",
    backend_interface: str = "python",
):
    nx_neg = max(2, int(round(0.5 * ny_neg)))
    nx_pos = max(2, int(round(0.5 * ny_pos)))
    mesh_neg = _make_submesh(element=element, poly_order=degree, nx=nx_neg, ny=ny_neg, offset_x=0.0)
    mesh_pos = _make_submesh(element=element, poly_order=degree, nx=nx_pos, ny=ny_pos, offset_x=0.5)

    mesh_neg.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})
    mesh_pos.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})

    dh_neg = DofHandler(MixedElement(mesh_neg, {"u": int(degree)}), method="cg")
    dh_pos = DofHandler(MixedElement(mesh_pos, {"u": int(degree)}), method="cg")

    f_expr = Analytic(f, degree=max(4, 2 * degree + 2))

    def assemble_side(dh: DofHandler, k: float):
        u = TrialFunction(name="u_trial", field_name="u", dof_handler=dh)
        v = UFLTestFunction(name="v_test", field_name="u", dof_handler=dh)
        a = Constant(float(k)) * inner(grad(u), grad(v)) * dx()
        L = f_expr * v * dx()
        return assemble_form(
            Equation(a, L), dof_handler=dh, bcs=[], quad_order=int(2 * degree + 2), backend=backend_volume
        )

    K_pos, F_pos = assemble_side(dh_pos, float(k_pos))
    K_neg, F_neg = assemble_side(dh_neg, float(k_neg))

    interface = build_nonmatching_interface(mesh_neg=mesh_neg, mesh_pos=mesh_pos)
    coupling = assemble_poisson_mortar_coupling(interface=interface, dh_neg=dh_neg, dh_pos=dh_pos, backend=backend_interface)
    K = assemble_mortar_saddle_matrix(K_pos=K_pos, K_neg=K_neg, coupling=coupling)

    n_pos = int(dh_pos.total_dofs)
    n_neg = int(dh_neg.total_dofs)
    n_lam = int(coupling.n_lambda)
    F = np.concatenate([np.asarray(F_pos, float), np.asarray(F_neg, float), np.zeros(n_lam, dtype=float)])

    bcs_pos = [BoundaryCondition("u", "dirichlet", "boundary", lambda xx, yy: float(u_pos(xx, yy)))]
    bcs_neg = [BoundaryCondition("u", "dirichlet", "boundary", lambda xx, yy: float(u_neg(xx, yy)))]
    bc_data = coupled_dirichlet_data(dh_pos=dh_pos, bcs_pos=bcs_pos, dh_neg=dh_neg, bcs_neg=bcs_neg, neg_offset=n_pos)
    K_bc, F_bc = apply_dirichlet_data(K, F, bc_data)

    sol = spla.spsolve(K_bc.tocsc(), F_bc)
    U_pos = sol[:n_pos]
    U_neg = sol[n_pos : n_pos + n_neg]
    return U_pos, U_neg, dh_pos, dh_neg, interface


@pytest.mark.parametrize("element", ["quad", "tri"])
def test_nonmatching_poisson_patch_constant_nitsche(element):
    u_exact = lambda x, y: 1.0
    grad_u = lambda x, y: (0.0, 0.0)
    f = lambda xv, yv: 0.0 * xv

    U_pos, U_neg, dh_pos, dh_neg, interface = _solve_poisson_nitsche(
        element=element,
        degree=1,
        ny_neg=6,
        ny_pos=7,
        k_neg=1.0,
        k_pos=1.0,
        gamma=20.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend_volume="python",
        backend_interface="python",
    )

    err = max(
        float(np.max(np.abs(U_pos - 1.0))),
        float(np.max(np.abs(U_neg - 1.0))),
    )
    assert err < 1.0e-11
    assert scalar_jump_L2(interface=interface, dh_neg=dh_neg, u_neg=U_neg, dh_pos=dh_pos, u_pos=U_pos) < 1.0e-10


@pytest.mark.parametrize("element", ["quad", "tri"])
def test_nonmatching_poisson_patch_linear_nitsche(element):
    u_exact = lambda x, y: float(x + 2.0 * y - 0.3)
    grad_u = lambda x, y: (1.0, 2.0)
    f = lambda xv, yv: 0.0 * xv

    U_pos, U_neg, dh_pos, dh_neg, interface = _solve_poisson_nitsche(
        element=element,
        degree=1,
        ny_neg=6,
        ny_pos=9,
        k_neg=1.0,
        k_pos=1.0,
        gamma=20.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend_volume="python",
        backend_interface="python",
    )

    errL2_pos = scalar_L2_error(dh=dh_pos, uh=U_pos, u_exact=u_exact, field="u")
    errL2_neg = scalar_L2_error(dh=dh_neg, uh=U_neg, u_exact=u_exact, field="u")
    errL2 = float(np.sqrt(errL2_pos**2 + errL2_neg**2))
    assert errL2 < 1.0e-11
    assert scalar_jump_L2(interface=interface, dh_neg=dh_neg, u_neg=U_neg, dh_pos=dh_pos, u_pos=U_pos) < 1.0e-10


@pytest.mark.parametrize("element", ["quad", "tri"])
def test_nonmatching_poisson_mms_kjump_convergence_nitsche(element):
    k_neg = 1.0
    k_pos = 10.0

    u_neg = lambda x, y: (x - 0.5) * np.sin(np.pi * y)
    u_pos = lambda x, y: 0.1 * (x - 0.5) * np.sin(np.pi * y)
    grad_u_neg = lambda x, y: (np.sin(np.pi * y), (x - 0.5) * np.pi * np.cos(np.pi * y))
    grad_u_pos = lambda x, y: (0.1 * np.sin(np.pi * y), 0.1 * (x - 0.5) * np.pi * np.cos(np.pi * y))
    f = lambda xv, yv: (np.pi**2) * (xv - 0.5) * np.sin(np.pi * yv)

    errs = []
    hs = []
    for ny in (6, 12):
        U_pos, U_neg, dh_pos, dh_neg, interface = _solve_poisson_nitsche(
            element=element,
            degree=1,
            ny_neg=ny,
            ny_pos=ny + 1,
            k_neg=k_neg,
            k_pos=k_pos,
            gamma=20.0,
            u_neg=u_neg,
            grad_u_neg=grad_u_neg,
            u_pos=u_pos,
            grad_u_pos=grad_u_pos,
            f=f,
            backend_volume="python",
            backend_interface="python",
        )

        errL2_pos = scalar_L2_error(dh=dh_pos, uh=U_pos, u_exact=u_pos, field="u")
        errL2_neg = scalar_L2_error(dh=dh_neg, uh=U_neg, u_exact=u_neg, field="u")
        errH1_pos = scalar_H1_semi_error(dh=dh_pos, uh=U_pos, grad_u_exact=grad_u_pos, field="u")
        errH1_neg = scalar_H1_semi_error(dh=dh_neg, uh=U_neg, grad_u_exact=grad_u_neg, field="u")
        errs.append((float(np.sqrt(errL2_pos**2 + errL2_neg**2)), float(np.sqrt(errH1_pos**2 + errH1_neg**2))))
        hs.append(float(max(np.sqrt(dh_pos.mixed_element.mesh.areas_list).max(), np.sqrt(dh_neg.mixed_element.mesh.areas_list).max())))

        # sanity: interface jumps decay
        assert poisson_flux_mismatch_L2(interface=interface, dh_neg=dh_neg, u_neg=U_neg, dh_pos=dh_pos, u_pos=U_pos, k_neg=k_neg, k_pos=k_pos) < 1.0

    (eL2_0, eH1_0), (eL2_1, eH1_1) = errs
    h0, h1 = hs
    rate_L2 = np.log(eL2_0 / eL2_1) / np.log(h0 / h1)
    rate_H1 = np.log(eH1_0 / eH1_1) / np.log(h0 / h1)
    assert rate_L2 > 1.8
    assert rate_H1 > 0.9


def test_nonmatching_poisson_mortar_patch_linear_quad():
    u_exact = lambda x, y: float(x + 2.0 * y - 0.3)
    grad_u = lambda x, y: (1.0, 2.0)
    f = lambda xv, yv: 0.0 * xv

    U_pos, U_neg, dh_pos, dh_neg, interface = _solve_poisson_mortar(
        element="quad",
        degree=1,
        ny_neg=6,
        ny_pos=9,
        k_neg=1.0,
        k_pos=1.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend_volume="python",
        backend_interface="python",
    )

    errL2_pos = scalar_L2_error(dh=dh_pos, uh=U_pos, u_exact=u_exact, field="u")
    errL2_neg = scalar_L2_error(dh=dh_neg, uh=U_neg, u_exact=u_exact, field="u")
    errL2 = float(np.sqrt(errL2_pos**2 + errL2_neg**2))
    assert errL2 < 1.0e-11
    assert scalar_jump_L2(interface=interface, dh_neg=dh_neg, u_neg=U_neg, dh_pos=dh_pos, u_pos=U_pos) < 1.0e-10


def test_nonmatching_stokes_mms_convergence_quad():
    mu = 1.0
    ux = lambda x, y: np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    uy = lambda x, y: -np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    p_exact = lambda x, y: np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    f1 = lambda x, y: 2 * mu * np.pi**3 * np.sin(np.pi * x) * np.cos(np.pi * y) + 2 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
    f2 = lambda x, y: -2 * mu * np.pi**3 * np.cos(np.pi * x) * np.sin(np.pi * y) + 2 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    f_vec = Analytic(lambda xv, yv: np.stack([f1(xv, yv), f2(xv, yv)], axis=-1), dim=1, degree=8)

    vel_space = FunctionSpace("V", ["ux", "uy"], dim=1)
    u = VectorTrialFunction(vel_space)
    v = VectorTestFunction(vel_space)
    p = TrialFunction("p")
    q = UFLTestFunction("p")

    def eps(w):
        return 0.5 * (grad(w) + grad(w).T)

    a = (Constant(2.0 * mu) * inner(eps(u), eps(v)) - p * div(v) + q * div(u)) * dx()
    L = dot(f_vec, v) * dx()

    errs = []
    hs = []
    for ny in (4, 8):
        nx_neg = max(2, int(round(0.5 * ny)))
        nx_pos = max(2, int(round(0.5 * (ny + 1))))
        mesh_neg = _make_submesh(element="quad", poly_order=2, nx=nx_neg, ny=ny, offset_x=0.0)
        mesh_pos = _make_submesh(element="quad", poly_order=2, nx=nx_pos, ny=ny + 1, offset_x=0.5)
        mesh_neg.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})
        mesh_pos.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})

        fields = {"ux": 2, "uy": 2, "p": 1}
        dh_neg = DofHandler(MixedElement(mesh_neg, fields), method="cg")
        dh_pos = DofHandler(MixedElement(mesh_pos, fields), method="cg")
        dh_neg.tag_dof_by_locator("pressure_pin", "p", lambda xx, yy: abs(xx - 0.0) < 1e-12 and abs(yy - 0.0) < 1e-12)
        dh_pos.tag_dof_by_locator("pressure_pin", "p", lambda xx, yy: abs(xx - 1.0) < 1e-12 and abs(yy - 0.0) < 1e-12)

        K_pos, F_pos = assemble_form(Equation(a, L), dof_handler=dh_pos, bcs=[], quad_order=8, backend="python")
        K_neg, F_neg = assemble_form(Equation(a, L), dof_handler=dh_neg, bcs=[], quad_order=8, backend="python")

        interface = build_nonmatching_interface(mesh_neg=mesh_neg, mesh_pos=mesh_pos)
        K_if = assemble_stokes_nitsche_interface_matrix(
            interface=interface, dh_neg=dh_neg, dh_pos=dh_pos, mu_neg=mu, mu_pos=mu, gamma=40.0, backend="python"
        )

        n_pos = int(dh_pos.total_dofs)
        n_neg = int(dh_neg.total_dofs)
        K = sp.block_diag([K_pos.tocsr(), K_neg.tocsr()], format="csr") + K_if
        F = np.concatenate([np.asarray(F_pos, float), np.asarray(F_neg, float)])

        bcs_pos = [
            BoundaryCondition("ux", "dirichlet", "boundary", lambda xx, yy: float(ux(xx, yy))),
            BoundaryCondition("uy", "dirichlet", "boundary", lambda xx, yy: float(uy(xx, yy))),
            BoundaryCondition("p", "dirichlet", "pressure_pin", lambda xx, yy: float(p_exact(xx, yy))),
        ]
        bcs_neg = [
            BoundaryCondition("ux", "dirichlet", "boundary", lambda xx, yy: float(ux(xx, yy))),
            BoundaryCondition("uy", "dirichlet", "boundary", lambda xx, yy: float(uy(xx, yy))),
            BoundaryCondition("p", "dirichlet", "pressure_pin", lambda xx, yy: float(p_exact(xx, yy))),
        ]
        bc_data = coupled_dirichlet_data(dh_pos=dh_pos, bcs_pos=bcs_pos, dh_neg=dh_neg, bcs_neg=bcs_neg, neg_offset=n_pos)
        K_bc, F_bc = apply_dirichlet_data(K, F, bc_data)

        sol = spla.spsolve(K_bc.tocsc(), F_bc)
        U_pos = sol[:n_pos]
        U_neg = sol[n_pos : n_pos + n_neg]

        err_u = np.sqrt(
            scalar_L2_error(dh=dh_pos, uh=U_pos, u_exact=ux, field="ux") ** 2
            + scalar_L2_error(dh=dh_pos, uh=U_pos, u_exact=uy, field="uy") ** 2
            + scalar_L2_error(dh=dh_neg, uh=U_neg, u_exact=ux, field="ux") ** 2
            + scalar_L2_error(dh=dh_neg, uh=U_neg, u_exact=uy, field="uy") ** 2
        )
        err_p = np.sqrt(
            scalar_L2_error(dh=dh_pos, uh=U_pos, u_exact=p_exact, field="p") ** 2
            + scalar_L2_error(dh=dh_neg, uh=U_neg, u_exact=p_exact, field="p") ** 2
        )
        errs.append((float(err_u), float(err_p)))
        hs.append(float(max(np.sqrt(mesh_pos.areas_list).max(), np.sqrt(mesh_neg.areas_list).max())))

        # interface diagnostics (sanity)
        assert stokes_velocity_jump_L2(interface=interface, dh_neg=dh_neg, U_neg=U_neg, dh_pos=dh_pos, U_pos=U_pos) < 1.0
        assert stokes_traction_mismatch_L2(interface=interface, dh_neg=dh_neg, U_neg=U_neg, dh_pos=dh_pos, U_pos=U_pos, mu_neg=mu, mu_pos=mu) < 10.0

    (eu0, ep0), (eu1, ep1) = errs
    h0, h1 = hs
    rate_u = np.log(eu0 / eu1) / np.log(h0 / h1)
    rate_p = np.log(ep0 / ep1) / np.log(h0 / h1)
    assert rate_u > 2.5
    assert rate_p > 1.5


def _have_numba() -> bool:
    try:
        import numba  # noqa: F401
    except Exception:
        return False
    return True


def _have_pybind11() -> bool:
    try:
        import pybind11  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _have_numba(), reason="numba not available")
def test_nonmatching_backend_parity_poisson_nitsche_jit():
    u_exact = lambda x, y: float(x + 2.0 * y - 0.3)
    grad_u = lambda x, y: (1.0, 2.0)
    f = lambda xv, yv: 0.0 * xv

    U_pos_py, U_neg_py, dh_pos, dh_neg, interface = _solve_poisson_nitsche(
        element="quad",
        degree=1,
        ny_neg=4,
        ny_pos=5,
        k_neg=1.0,
        k_pos=1.0,
        gamma=20.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend_volume="python",
        backend_interface="python",
    )
    U_pos_jit, U_neg_jit, *_ = _solve_poisson_nitsche(
        element="quad",
        degree=1,
        ny_neg=4,
        ny_pos=5,
        k_neg=1.0,
        k_pos=1.0,
        gamma=20.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend_volume="python",
        backend_interface="jit",
    )

    assert np.max(np.abs(U_pos_py - U_pos_jit)) < 1.0e-12
    assert np.max(np.abs(U_neg_py - U_neg_jit)) < 1.0e-12


@pytest.mark.skipif(not _have_pybind11(), reason="pybind11 not available")
def test_nonmatching_backend_parity_poisson_nitsche_cpp():
    u_exact = lambda x, y: float(x + 2.0 * y - 0.3)
    grad_u = lambda x, y: (1.0, 2.0)
    f = lambda xv, yv: 0.0 * xv

    U_pos_py, U_neg_py, dh_pos, dh_neg, interface = _solve_poisson_nitsche(
        element="quad",
        degree=1,
        ny_neg=4,
        ny_pos=5,
        k_neg=1.0,
        k_pos=1.0,
        gamma=20.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend_volume="python",
        backend_interface="python",
    )
    U_pos_cpp, U_neg_cpp, *_ = _solve_poisson_nitsche(
        element="quad",
        degree=1,
        ny_neg=4,
        ny_pos=5,
        k_neg=1.0,
        k_pos=1.0,
        gamma=20.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend_volume="python",
        backend_interface="cpp",
    )

    assert np.max(np.abs(U_pos_py - U_pos_cpp)) < 1.0e-12
    assert np.max(np.abs(U_neg_py - U_neg_cpp)) < 1.0e-12


@pytest.mark.skipif(not _have_numba(), reason="numba not available")
def test_nonmatching_backend_parity_poisson_mortar_jit():
    u_exact = lambda x, y: float(x + 2.0 * y - 0.3)
    grad_u = lambda x, y: (1.0, 2.0)
    f = lambda xv, yv: 0.0 * xv

    U_pos_py, U_neg_py, *_ = _solve_poisson_mortar(
        element="quad",
        degree=1,
        ny_neg=4,
        ny_pos=5,
        k_neg=1.0,
        k_pos=1.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend_volume="python",
        backend_interface="python",
    )
    U_pos_jit, U_neg_jit, *_ = _solve_poisson_mortar(
        element="quad",
        degree=1,
        ny_neg=4,
        ny_pos=5,
        k_neg=1.0,
        k_pos=1.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend_volume="python",
        backend_interface="jit",
    )
    assert np.max(np.abs(U_pos_py - U_pos_jit)) < 1.0e-12
    assert np.max(np.abs(U_neg_py - U_neg_jit)) < 1.0e-12


@pytest.mark.skipif(not _have_pybind11(), reason="pybind11 not available")
def test_nonmatching_backend_parity_poisson_mortar_cpp():
    u_exact = lambda x, y: float(x + 2.0 * y - 0.3)
    grad_u = lambda x, y: (1.0, 2.0)
    f = lambda xv, yv: 0.0 * xv

    U_pos_py, U_neg_py, *_ = _solve_poisson_mortar(
        element="quad",
        degree=1,
        ny_neg=4,
        ny_pos=5,
        k_neg=1.0,
        k_pos=1.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend_volume="python",
        backend_interface="python",
    )
    U_pos_cpp, U_neg_cpp, *_ = _solve_poisson_mortar(
        element="quad",
        degree=1,
        ny_neg=4,
        ny_pos=5,
        k_neg=1.0,
        k_pos=1.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend_volume="python",
        backend_interface="cpp",
    )
    assert np.max(np.abs(U_pos_py - U_pos_cpp)) < 1.0e-12
    assert np.max(np.abs(U_neg_py - U_neg_cpp)) < 1.0e-12


@pytest.mark.skipif(not _have_numba(), reason="numba not available")
def test_nonmatching_backend_parity_stokes_nitsche_matrix_jit():
    nodes, elems, edges, corners = structured_quad(0.5, 1.0, nx=1, ny=2, poly_order=2, offset=(0.0, 0.0))
    mesh_neg = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=2)
    nodes, elems, edges, corners = structured_quad(0.5, 1.0, nx=1, ny=3, poly_order=2, offset=(0.5, 0.0))
    mesh_pos = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=2)
    mesh_neg.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})
    mesh_pos.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})

    fields = {"ux": 2, "uy": 2, "p": 1}
    dh_neg = DofHandler(MixedElement(mesh_neg, fields), method="cg")
    dh_pos = DofHandler(MixedElement(mesh_pos, fields), method="cg")
    interface = build_nonmatching_interface(mesh_neg=mesh_neg, mesh_pos=mesh_pos)

    K_py = assemble_stokes_nitsche_interface_matrix(
        interface=interface, dh_neg=dh_neg, dh_pos=dh_pos, mu_neg=1.0, mu_pos=1.0, backend="python"
    )
    K_jit = assemble_stokes_nitsche_interface_matrix(
        interface=interface, dh_neg=dh_neg, dh_pos=dh_pos, mu_neg=1.0, mu_pos=1.0, backend="jit"
    )
    diff = (K_py - K_jit).tocoo()
    assert diff.data.size == 0 or float(np.max(np.abs(diff.data))) < 1.0e-12


@pytest.mark.skipif(not _have_pybind11(), reason="pybind11 not available")
def test_nonmatching_backend_parity_stokes_nitsche_matrix_cpp():
    nodes, elems, edges, corners = structured_quad(0.5, 1.0, nx=1, ny=2, poly_order=2, offset=(0.0, 0.0))
    mesh_neg = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=2)
    nodes, elems, edges, corners = structured_quad(0.5, 1.0, nx=1, ny=3, poly_order=2, offset=(0.5, 0.0))
    mesh_pos = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=2)
    mesh_neg.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})
    mesh_pos.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})

    fields = {"ux": 2, "uy": 2, "p": 1}
    dh_neg = DofHandler(MixedElement(mesh_neg, fields), method="cg")
    dh_pos = DofHandler(MixedElement(mesh_pos, fields), method="cg")
    interface = build_nonmatching_interface(mesh_neg=mesh_neg, mesh_pos=mesh_pos)

    K_py = assemble_stokes_nitsche_interface_matrix(
        interface=interface, dh_neg=dh_neg, dh_pos=dh_pos, mu_neg=1.0, mu_pos=1.0, backend="python"
    )
    K_cpp = assemble_stokes_nitsche_interface_matrix(
        interface=interface, dh_neg=dh_neg, dh_pos=dh_pos, mu_neg=1.0, mu_pos=1.0, backend="cpp"
    )
    diff = (K_py - K_cpp).tocoo()
    assert diff.data.size == 0 or float(np.max(np.abs(diff.data))) < 1.0e-12
