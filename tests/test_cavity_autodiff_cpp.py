import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl import linearize_form
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    div,
    dot,
    grad,
    inner,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _have_pybind11() -> bool:
    try:
        import pybind11  # noqa: F401
    except Exception:
        return False
    return True


def _build_cavity_mesh(mesh_kind: str, *, nx: int, ny: int, tmp_path) -> Mesh:
    if mesh_kind == "structured":
        nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=1)
        return Mesh(
            nodes=nodes,
            element_connectivity=elems,
            elements_corner_nodes=corners,
            element_type="quad",
            poly_order=1,
        )

    if mesh_kind == "gmsh":
        pytest.importorskip("gmsh")
        from examples.gmsh_cavity_mesh import build_caity_quad_mesh
        from pycutfem.utils.gmsh_loader import mesh_from_gmsh

        mesh_path = tmp_path / f"cavity_{nx}x{ny}_q1.msh"
        build_caity_quad_mesh(mesh_path, nx=nx, ny=ny, element_order=1)
        return mesh_from_gmsh(mesh_path, apply_boundary_tags=True)

    raise ValueError(f"Unsupported mesh_kind={mesh_kind!r}")


def _build_boundary_conditions(dof_handler: DofHandler, mesh: Mesh, *, prefer_gmsh_tags: bool) -> tuple[list[BoundaryCondition], list[BoundaryCondition]]:
    geom_tol = 1.0e-6
    locator_tags = {
        "bottom_wall": lambda x, y, tol=geom_tol: np.isclose(y, 0.0, atol=tol),
        "left_wall": lambda x, y, tol=geom_tol: np.isclose(x, 0.0, atol=tol),
        "right_wall": lambda x, y, tol=geom_tol: np.isclose(x, 1.0, atol=tol),
        "top_lid": lambda x, y, tol=geom_tol: np.isclose(y, 1.0, atol=tol),
    }
    if prefer_gmsh_tags:
        gmsh_tags = set()
        for edge in mesh.edges_list:
            if edge.right is not None or not edge.tag:
                continue
            gmsh_tags.update(tag.strip() for tag in edge.tag.split(",") if tag)
        missing = [tag for tag in locator_tags if tag not in gmsh_tags]
        if missing:
            mesh.tag_boundary_edges(locator_tags)
    else:
        mesh.tag_boundary_edges(locator_tags)

    dof_handler.tag_dof_by_locator(
        tag="pressure_pin_point",
        field="p",
        locator=lambda x, y, tol=geom_tol: np.isclose(x, 0.0, atol=tol) and np.isclose(y, 0.0, atol=tol),
        find_first=True,
    )

    bcs = [
        BoundaryCondition("ux", "dirichlet", "bottom_wall", lambda x, y: 0.0),
        BoundaryCondition("uy", "dirichlet", "bottom_wall", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "left_wall", lambda x, y: 0.0),
        BoundaryCondition("uy", "dirichlet", "left_wall", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "right_wall", lambda x, y: 0.0),
        BoundaryCondition("uy", "dirichlet", "right_wall", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "top_lid", lambda x, y: 1.0),
        BoundaryCondition("uy", "dirichlet", "top_lid", lambda x, y: 0.0),
        BoundaryCondition("p", "dirichlet", "pressure_pin_point", lambda x, y: 0.0),
    ]
    bcs_homog = [
        BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0)
        for bc in bcs
    ]
    return bcs, bcs_homog


def _build_cavity_forms(dof_handler: DofHandler):
    velocity_space = FunctionSpace("velocity", ["ux", "uy"])
    pressure_space = FunctionSpace("pressure", ["p"])

    du = VectorTrialFunction(velocity_space, dof_handler=dof_handler)
    dp = TrialFunction(pressure_space, dof_handler=dof_handler)
    v = VectorTestFunction(velocity_space, dof_handler=dof_handler)
    q = TestFunction(pressure_space, dof_handler=dof_handler)

    u_k = VectorFunction(name="u_k", field_names=["ux", "uy"], dof_handler=dof_handler)
    u_n = VectorFunction(name="u_n", field_names=["ux", "uy"], dof_handler=dof_handler)
    p_k = Function(name="p_k", field_name="p", dof_handler=dof_handler)
    p_n = Function(name="p_n", field_name="p", dof_handler=dof_handler)

    rho = Constant(1.0)
    dt = Constant(0.1)
    theta = Constant(1.0)
    mu = Constant(0.01)
    dx_q = dx(metadata={"q": 4})

    jacobian_form = (
        rho * dot(du, v) / dt
        + theta * rho * dot(dot(grad(u_k), du), v)
        + theta * rho * dot(dot(grad(du), u_k), v)
        + theta * mu * inner(grad(du), grad(v))
        - dp * div(v)
        + q * div(du)
    ) * dx_q

    residual_form = (
        rho * dot(u_k - u_n, v) / dt
        + theta * rho * dot(dot(grad(u_k), u_k), v)
        + (1.0 - theta) * rho * dot(dot(grad(u_n), u_n), v)
        + theta * mu * inner(grad(u_k), grad(v))
        + (1.0 - theta) * mu * inner(grad(u_n), grad(v))
        - p_k * div(v)
        + q * div(u_k)
    ) * dx_q

    auto_jacobian = linearize_form(residual_form, [u_k, p_k], [du, dp])
    return jacobian_form, auto_jacobian, u_k, u_n, p_k, p_n


def _initialize_cavity_state(
    dof_handler: DofHandler,
    *,
    bcs: list[BoundaryCondition],
    u_k: VectorFunction,
    u_n: VectorFunction,
    p_k: Function,
    p_n: Function,
) -> None:
    u_k.set_values_from_function(
        lambda x, y: np.array(
            [
                y + 0.05 * x * (1.0 - x) * y * (1.0 - y),
                0.03 * x * (1.0 - x) * y * (1.0 - y),
            ]
        )
    )
    u_n.set_values_from_function(
        lambda x, y: np.array(
            [
                0.85 * y + 0.02 * x * (1.0 - x) * y * (1.0 - y),
                -0.01 * x * (1.0 - x) * y * (1.0 - y),
            ]
        )
    )
    p_k.set_values_from_function(lambda x, y: 0.20 + 0.10 * x - 0.07 * y)
    p_n.set_values_from_function(lambda x, y: 0.05 + 0.04 * x - 0.03 * y)

    dof_handler.apply_bcs(bcs, u_k, p_k)
    dof_handler.apply_bcs(bcs, u_n, p_n)


def _assemble_dense_cpp(dof_handler: DofHandler, form, *, bcs: list[BoundaryCondition]) -> np.ndarray:
    matrix, _ = assemble_form(
        Equation(form, None),
        dof_handler=dof_handler,
        bcs=bcs,
        backend="cpp",
    )
    return matrix.toarray()


@pytest.mark.skipif(not _have_pybind11(), reason="pybind11 not available")
@pytest.mark.parametrize(
    ("mesh_kind", "nx", "ny"),
    [
        ("structured", 2, 3),
        ("gmsh", 2, 3),
    ],
    ids=["structured", "gmsh"],
)
def test_cavity_manual_vs_autodiff_jacobian_cpp(mesh_kind: str, nx: int, ny: int, tmp_path) -> None:
    mesh = _build_cavity_mesh(mesh_kind, nx=nx, ny=ny, tmp_path=tmp_path)
    mixed_element = MixedElement(mesh, field_specs={"ux": 2, "uy": 2, "p": 1})
    dof_handler = DofHandler(mixed_element, method="cg")
    bcs, bcs_homog = _build_boundary_conditions(
        dof_handler,
        mesh,
        prefer_gmsh_tags=(mesh_kind == "gmsh"),
    )

    manual_jacobian, auto_jacobian, u_k, u_n, p_k, p_n = _build_cavity_forms(dof_handler)
    _initialize_cavity_state(
        dof_handler,
        bcs=bcs,
        u_k=u_k,
        u_n=u_n,
        p_k=p_k,
        p_n=p_n,
    )

    manual_cpp = _assemble_dense_cpp(dof_handler, manual_jacobian, bcs=bcs_homog)
    auto_cpp = _assemble_dense_cpp(dof_handler, auto_jacobian, bcs=bcs_homog)

    np.testing.assert_allclose(auto_cpp, manual_cpp, rtol=1.0e-10, atol=1.0e-10)
