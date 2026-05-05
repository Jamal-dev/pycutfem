import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Function, VectorFunction
from pycutfem.ufl.measures import dS, dx
from examples.utils.biofilm.adhesion import (
    solid_von_mises_mass_lumped_in_domain,
    solid_von_mises_mass_lumped_on_boundary_linear,
    update_adhesion_integrity_field_on_boundary_von_mises,
)
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


def _build_wall_failure_problem(*, nx: int = 2, ny: int = 2):
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
            "u_x": 2,
            "u_y": 2,
            "alpha": 1,
            "a": 1,
        },
    )
    dh = DofHandler(me, method="cg")

    u = VectorFunction("u", ["u_x", "u_y"], dof_handler=dh)
    alpha = Function("alpha", "alpha", dof_handler=dh)
    a_field = Function("a_field", "a", dof_handler=dh)

    ds_bottom = dS(defined_on=mesh.edge_bitset("bottom"), metadata={"q": 3})
    return dh, ds_bottom, u, alpha, a_field


def test_von_mises_wall_update_binary_failure_sets_a_to_zero_on_active_wall_dofs():
    dh, ds_bottom, u, alpha, a_field = _build_wall_failure_problem(nx=2, ny=2)

    # Uniform strain => nonzero von Mises stress.
    u.set_values_from_function(lambda x, y: np.array([float(x), 0.0], dtype=float))
    alpha.set_values_from_function(lambda x, y: 1.0)
    a_field.set_values_from_function(lambda x, y: 1.0)

    sigma, w = solid_von_mises_mass_lumped_on_boundary_linear(
        dof_handler=dh,
        field="a",
        u=u,
        alpha=alpha,
        ds_wall=ds_bottom,
        mu_s=1.0,
        lambda_s=1.0,
        backend="python",
        quad_order=3,
    )
    assert float(sigma.max()) > 0.0

    update_adhesion_integrity_field_on_boundary_von_mises(
        dof_handler=dh,
        a_field=a_field,
        dt=0.1,
        u=u,
        alpha=alpha,
        ds_wall=ds_bottom,
        mu_s=1.0,
        lambda_s=1.0,
        k_break=0.0,  # binary mode
        sigma_cr=1.0,  # below sigma_vm
        m=1.0,
        backend="python",
        quad_order=3,
    )

    active = w > 1.0e-14
    assert int(np.count_nonzero(active)) > 0
    assert np.allclose(np.asarray(a_field.nodal_values, float)[active], 0.0)


def test_von_mises_wall_update_rate_law_degrades_a_but_keeps_it_positive():
    dh, ds_bottom, u, alpha, a_field = _build_wall_failure_problem(nx=2, ny=2)

    u.set_values_from_function(lambda x, y: np.array([float(x), 0.0], dtype=float))
    alpha.set_values_from_function(lambda x, y: 1.0)
    a_field.set_values_from_function(lambda x, y: 1.0)

    sigma, w = solid_von_mises_mass_lumped_on_boundary_linear(
        dof_handler=dh,
        field="a",
        u=u,
        alpha=alpha,
        ds_wall=ds_bottom,
        mu_s=1.0,
        lambda_s=1.0,
        backend="python",
        quad_order=3,
    )
    active = w > 1.0e-14
    assert int(np.count_nonzero(active)) > 0

    update_adhesion_integrity_field_on_boundary_von_mises(
        dof_handler=dh,
        a_field=a_field,
        dt=0.1,
        u=u,
        alpha=alpha,
        ds_wall=ds_bottom,
        mu_s=1.0,
        lambda_s=1.0,
        k_break=10.0,  # rate mode
        sigma_cr=1.0,
        m=1.0,
        backend="python",
        quad_order=3,
    )

    a_vals = np.asarray(a_field.nodal_values, float)[active]
    assert float(a_vals.min()) > 0.0
    assert float(a_vals.max()) < 1.0


def test_von_mises_wall_update_a_snap_sets_tiny_a_to_zero():
    dh, ds_bottom, u, alpha, a_field = _build_wall_failure_problem(nx=2, ny=2)

    # Zero stress (sigma=0), but active wall DOFs exist (alpha=1 on the wall).
    u.set_values_from_function(lambda x, y: np.array([0.0, 0.0], dtype=float))
    alpha.set_values_from_function(lambda x, y: 1.0)
    a_field.set_values_from_function(lambda x, y: 0.04)

    sigma, w = solid_von_mises_mass_lumped_on_boundary_linear(
        dof_handler=dh,
        field="a",
        u=u,
        alpha=alpha,
        ds_wall=ds_bottom,
        mu_s=1.0,
        lambda_s=1.0,
        backend="python",
        quad_order=3,
    )
    assert float(sigma.max()) == 0.0
    active = w > 1.0e-14
    assert int(np.count_nonzero(active)) > 0

    update_adhesion_integrity_field_on_boundary_von_mises(
        dof_handler=dh,
        a_field=a_field,
        dt=0.1,
        u=u,
        alpha=alpha,
        ds_wall=ds_bottom,
        mu_s=1.0,
        lambda_s=1.0,
        k_break=0.0,  # binary mode (no stress-based failure here)
        sigma_cr=1.0e9,
        m=1.0,
        a_snap=0.05,
        backend="python",
        quad_order=3,
    )

    a_vals = np.asarray(a_field.nodal_values, float)[active]
    assert np.allclose(a_vals, 0.0)


def test_von_mises_domain_mass_lumped_projection_returns_constant_stress_for_linear_u():
    dh, _ds_bottom, u, alpha, _a_field = _build_wall_failure_problem(nx=2, ny=2)

    # Uniform strain => constant stress => constant von Mises stress.
    u.set_values_from_function(lambda x, y: np.array([float(x), 0.0], dtype=float))
    alpha.set_values_from_function(lambda x, y: 1.0)

    sigma, w = solid_von_mises_mass_lumped_in_domain(
        dof_handler=dh,
        field="u_x",
        u=u,
        alpha=alpha,
        dx_domain=dx(metadata={"q": 6}),
        mu_s=1.0,
        lambda_s=1.0,
        solid_model="linear",
        backend="python",
        quad_order=6,
    )
    active = w > 1.0e-14
    assert int(np.count_nonzero(active)) > 0
    assert np.allclose(sigma[active], np.sqrt(3.0), rtol=1.0e-12, atol=1.0e-12)
