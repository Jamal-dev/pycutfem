import numpy as np
import pytest

from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import FacetNormal, Function, dot, grad
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dS
from pycutfem.utils.meshgen import structured_quad


def _unit_square_mesh():
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    mesh.tag_boundary_edges(
        {
            "inlet": lambda x, y: np.isclose(x, 0.0),
            "outlet": lambda x, y: np.isclose(x, 1.0),
            "walls": lambda x, y: np.isclose(y, 0.0) | np.isclose(y, 1.0),
        }
    )
    bdry = mesh.edge_bitset("inlet") | mesh.edge_bitset("outlet") | mesh.edge_bitset("walls")
    assert bdry.cardinality() > 0
    return mesh, bdry


def _assemble_scalar(expr, dh: DofHandler, backend: str) -> float:
    res = assemble_form(
        Equation(None, expr),
        dof_handler=dh,
        bcs=[],
        assembler_hooks={expr.integrand: {"name": "val"}},
        backend=backend,
    )["val"]
    return float(np.asarray(res, dtype=float).reshape(-1)[0])


@pytest.mark.parametrize("backend", ["python", "jit", "cpp"])
def test_boundary_edge_functional_integrates_p_exactly(backend: str):
    mesh, bdry = _unit_square_mesh()
    me = MixedElement(mesh, field_specs={"p": 1})
    dh = DofHandler(me, method="cg")

    p = Function(name="p", field_name="p", dof_handler=dh, side="+")
    coords = np.asarray(dh.get_dof_coords("p"), dtype=float)
    # Exact Q1 representation of p(x,y)=x+y on the unit square.
    p.nodal_values[:] = coords[:, 0] + coords[:, 1]

    integral = p * dS(defined_on=bdry, metadata={"q": 4})

    try:
        val = _assemble_scalar(integral, dh, backend)
    except Exception as exc:
        # Allow environments without a working C++ toolchain to run the rest of the tests.
        if backend == "cpp":
            pytest.skip(f"cpp backend unavailable: {exc}")
        raise

    assert np.isclose(val, 4.0, atol=1e-12)


@pytest.mark.parametrize("backend", ["python", "jit", "cpp"])
def test_boundary_edge_functional_flux_of_grad_is_zero(backend: str):
    mesh, bdry = _unit_square_mesh()
    me = MixedElement(mesh, field_specs={"p": 1})
    dh = DofHandler(me, method="cg")

    p = Function(name="p", field_name="p", dof_handler=dh, side="+")
    coords = np.asarray(dh.get_dof_coords("p"), dtype=float)
    p.nodal_values[:] = coords[:, 0] + coords[:, 1]

    n = FacetNormal()
    flux = dot(grad(p), n) * dS(defined_on=bdry, metadata={"q": 4})

    try:
        val = _assemble_scalar(flux, dh, backend)
    except Exception as exc:
        if backend == "cpp":
            pytest.skip(f"cpp backend unavailable: {exc}")
        raise

    assert abs(val) < 1.0e-12
