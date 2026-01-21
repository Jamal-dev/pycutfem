import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem import transform
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.integration.quadrature import line_quadrature
from pycutfem.utils.meshgen import structured_quad
from pycutfem.utils.refinement import TensorRefiner


def _make_single_hanging_interface_mesh() -> Mesh:
    nodes, elems, _edges, corners = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=1)
    mesh0 = Mesh(
        nodes,
        element_connectivity=elems,
        edges_connectivity=None,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    refiner = TensorRefiner(max_ref=3)
    rx = np.zeros((len(mesh0.elements_list),), dtype=int)
    ry = np.zeros_like(rx)
    ry[0] = 1  # refine left element in y → hanging edge on the interior interface
    return refiner.refine(mesh0, rx, ry)


@pytest.mark.parametrize("k", (0, 1))
def test_hdiv_hanging_edge_constraints_enforce_flux_continuity(k):
    mesh = _make_single_hanging_interface_mesh()
    me = MixedElement(mesh, {"u": ("RT", int(k))})
    dh = DofHandler(me, method="cg")

    constraints = dh.build_hanging_node_constraints()
    assert constraints is not None
    assert constraints.slaves.size > 0

    rng = np.random.default_rng(0)
    u_master = rng.standard_normal(int(constraints.n_master))
    U = np.asarray(constraints.prolong(u_master), dtype=float)
    assert U.shape == (dh.total_dofs,)

    def eval_u_on_elem(eid: int, xi: float, eta: float) -> np.ndarray:
        loc = np.asarray(dh.element_maps["u"][int(eid)], dtype=int)
        sgn = np.asarray(dh.element_signs["u"][int(eid)], dtype=float)
        u_loc = U[loc] * sgn
        V = np.asarray(me.tabulate_value("u", float(xi), float(eta), element_id=int(eid)), dtype=float)
        return np.asarray(u_loc @ V, dtype=float)

    # Check normal-flux continuity on every interior mesh edge segment.
    for e in mesh.edges_list:
        if e.left is None or e.right is None:
            continue
        eL = int(e.left)
        eR = int(e.right)
        p0 = mesh.nodes_x_y_pos[int(e.nodes[0])]
        p1 = mesh.nodes_x_y_pos[int(e.nodes[1])]
        qpts, _wts = line_quadrature(p0, p1, order=8)
        n = np.asarray(e.normal, dtype=float)

        for xq in qpts:
            xiL, etaL = transform.inverse_mapping(mesh, eL, np.asarray(xq, float))
            xiR, etaR = transform.inverse_mapping(mesh, eR, np.asarray(xq, float))
            uL = eval_u_on_elem(eL, float(xiL), float(etaL))
            uR = eval_u_on_elem(eR, float(xiR), float(etaR))
            jump = float(uL @ n + uR @ (-n))
            assert abs(jump) < 1.0e-9

