import os
import numpy as np
import scipy.sparse.linalg as spla

from pycutfem.core.geometry import hansbo_cut_ratio
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.fsi_fully_eulerian import make_domain_sets, refresh_sliver_weights
from pycutfem.utils.meshgen import structured_quad

from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    ElementWiseConstant,
    FacetNormal,
    Neg,
    Pos,
    TestFunction,
    TrialFunction,
    dot,
    grad,
    inner,
    jump,
)
from pycutfem.ufl.measures import dx, dGhost
from pycutfem.ufl.forms import BoundaryCondition, assemble_form


def _build_scalar_system(mesh, dh, u, v, bcs, *, level_set, backend="python", edge_tol=1.0e-12):
    # Refresh geometry for this level set
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set, tol=edge_tol)
    mesh.build_interface_segments(level_set)

    # Refresh inactive DOF tags (IMPORTANT in moving-interface problems)
    dh.dof_tags["inactive"] = set()
    inactive = mesh.element_bitset("inside")
    dh.tag_dofs_from_element_bitset("inactive", "u_pos_", inactive, strict=True)

    # Domain sets / ghosts
    domains = make_domain_sets(mesh, use_aligned_interface=True)

    dx_fluid = dx(
        defined_on=domains["fluid_interface"],
        level_set=level_set,
        metadata={"q": 6, "side": "+"},
    )
    dG = dGhost(
        defined_on=domains["fluid_ghost"],
        level_set=level_set,
        metadata={"q": 6, "derivs": {(0, 1), (1, 0)}},
    )

    h = CellDiameter()
    n = FacetNormal()

    def grad_inner_jump(phi_1, phi_2):
        a = dot(jump(grad(phi_1)), n)
        b = dot(jump(grad(phi_2)), n)
        return inner(a, b)

    theta_pos_vals = np.clip(hansbo_cut_ratio(mesh, level_set, side="+"), 1.0e-8, 1.0)
    theta_neg_vals = np.clip(hansbo_cut_ratio(mesh, level_set, side="-"), 1.0e-8, 1.0)
    w_pos_vals = np.ones_like(theta_pos_vals)
    w_neg_vals = np.ones_like(theta_neg_vals)
    refresh_sliver_weights(
        mesh,
        theta_pos_vals,
        theta_neg_vals,
        w_pos_vals,
        w_neg_vals,
        theta0=0.05,
        p=1.0,
        wmax=1000.0,
        thetamin=1.0e-6,
        smooth=1.0,
    )
    w_pos_cell = ElementWiseConstant(w_pos_vals)
    w_gp = Constant(0.5) * (Pos(w_pos_cell) + Neg(w_pos_cell))

    gamma = Constant(0.1) * w_gp
    a = inner(grad(u), grad(v)) * dx_fluid + gamma * h * grad_inner_jump(u, v) * dG

    f = Constant(1.0)
    L = f * v * dx_fluid

    K, F = assemble_form(a == L, dh, bcs=bcs, quad_order=6, backend=backend)
    return K, F


def test_sliver_handling_corner_and_edge_cases():
    backend = os.getenv("SLIVER_SCALAR_BACKEND", "python").strip().lower()
    nodes, elems, _, corners = structured_quad(
        2.0, 2.0, nx=2, ny=2, poly_order=2, offset=(-1.0, -1.0)
    )
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=2)

    me = MixedElement(mesh, field_specs={"u_pos_": 2})
    dh = DofHandler(me, method="cg")

    space = FunctionSpace("u", ["u_pos_"], dim=1, side="+")
    u = TrialFunction(name="u_trial", field_name="u_pos_", dof_handler=dh, side="+")
    v = TestFunction(name="u_test", field_name="u_pos_", dof_handler=dh, side="+")

    # Outer boundary Dirichlet (keeps system well-posed)
    walls = {
        "left": lambda x, y: np.isclose(x, -1.0),
        "right": lambda x, y: np.isclose(x, 1.0),
        "bottom": lambda x, y: np.isclose(y, -1.0),
        "top": lambda x, y: np.isclose(y, 1.0),
    }
    mesh.tag_boundary_edges(walls)

    bcs = [
        *[BoundaryCondition("u_pos_", "dirichlet", side, lambda x, y: 0.0) for side in walls],
        BoundaryCondition("u_pos_", "dirichlet", "inactive", lambda x, y: 0.0),
    ]

    eps_list = [1.0e-1, 1.0e-2, 1.0e-4, 1.0e-6]

    for eps in eps_list:
        level_set = AffineLevelSet(a=1.0, b=1.0, c=2.0 - eps)
        K, F = _build_scalar_system(mesh, dh, u, v, bcs, level_set=level_set, backend=backend, edge_tol=1e-12)
        sol = spla.spsolve(K, F)
        assert np.all(np.isfinite(sol))

    for eps in eps_list:
        level_set = AffineLevelSet(a=1.0, b=1.0, c=-2.0 + eps)
        K, F = _build_scalar_system(mesh, dh, u, v, bcs, level_set=level_set, backend=backend, edge_tol=1e-12)
        sol = spla.spsolve(K, F)
        assert np.all(np.isfinite(sol))

    level_set = AffineLevelSet(a=1.0, b=1.0, c=2.0)
    K, F = _build_scalar_system(mesh, dh, u, v, bcs, level_set=level_set, backend=backend, edge_tol=1e-12)
    sol = spla.spsolve(K, F)
    assert np.all(np.isfinite(sol))

    level_set = AffineLevelSet(a=1.0, b=0.0, c=0.0)
    K, F = _build_scalar_system(mesh, dh, u, v, bcs, level_set=level_set, backend=backend, edge_tol=1e-14)
    sol = spla.spsolve(K, F)
    assert np.all(np.isfinite(sol))

    for eps in [1e-2, 1e-4, 1e-6]:
        level_set = AffineLevelSet(a=1.0, b=0.0, c=-eps)
        K, F = _build_scalar_system(mesh, dh, u, v, bcs, level_set=level_set, backend=backend, edge_tol=1e-14)
        sol = spla.spsolve(K, F)
        assert np.all(np.isfinite(sol))
