import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.meshgen import structured_quad
from pycutfem.utils.refinement import TensorRefiner


def _make_hanging_mesh(poly_order: int, *, axis: str) -> Mesh:
    """
    Build a small quad mesh with a single nonconforming interface.

    axis="y": refine left element in y → hanging nodes along a vertical interface
    axis="x": refine bottom element in x → hanging nodes along a horizontal interface
    """
    if axis == "y":
        nodes, elems, _edges, corners = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=poly_order)
    elif axis == "x":
        nodes, elems, _edges, corners = structured_quad(1.0, 1.0, nx=1, ny=2, poly_order=poly_order)
    else:
        raise ValueError(f"unknown axis={axis!r}")

    mesh0 = Mesh(
        nodes,
        element_connectivity=elems,
        edges_connectivity=None,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )

    refiner = TensorRefiner(max_ref=3)
    rx = np.zeros((len(mesh0.elements_list),), dtype=int)
    ry = np.zeros_like(rx)
    if axis == "y":
        ry[0] = 1
    else:
        rx[0] = 1
    return refiner.refine(mesh0, rx, ry)


def _poly_exact(x: float, y: float, *, p: int, axis: str) -> float:
    # Make the polynomial vary primarily along the direction where hanging nodes appear.
    if axis == "y":
        return y**p + 0.123 * x
    return x**p + 0.123 * y


@pytest.mark.parametrize("axis", ("x", "y"))
@pytest.mark.parametrize("poly_order", (1, 2, 3))
def test_hanging_constraint_prolong_reproduces_polynomial(axis, poly_order):
    mesh = _make_hanging_mesh(poly_order, axis=axis)
    me = MixedElement(mesh, field_specs={"u": poly_order})
    dh = DofHandler(me, method="cg")

    constraints = dh.build_hanging_node_constraints()
    assert constraints is not None, "expected hanging-node constraints"
    assert constraints.slaves.size > 0, "expected at least one slave DOF"

    xy = dh.get_dof_coords("u")
    u_full_exact = np.asarray([_poly_exact(float(x), float(y), p=poly_order, axis=axis) for x, y in xy], dtype=float)
    u_master = u_full_exact[constraints.master_ids]

    u_full = constraints.prolong(u_master)
    assert np.allclose(u_full, u_full_exact, atol=5e-11, rtol=5e-11)

    # Basic sanity: each constrained row is an interpolation (weights sum to 1).
    row_sums = np.asarray(constraints.E.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 1.0, atol=5e-12, rtol=0.0)
