import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.utils.meshgen import structured_quad
from pycutfem.utils.refinement import TensorRefiner


def _make_quad_mesh(nx: int = 1, ny: int = 1, p: int = 1) -> Mesh:
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=p)
    return Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=p)


def _assert_hanging_nodes_on_parent_edges(mesh: Mesh, tol: float = 1.0e-10) -> None:
    assert hasattr(mesh, "parent_corner_coords"), "refined mesh must expose parent_corner_coords for diagnostics"
    corners = mesh.parent_corner_coords
    pts = mesh.nodes_x_y_pos
    for nid in getattr(mesh, "hanging_nodes", []):
        x, y = pts[int(nid)]
        found = False
        for box in corners:
            xmin, xmax = box[:, 0].min(), box[:, 0].max()
            ymin, ymax = box[:, 1].min(), box[:, 1].max()
            if x < xmin - tol or x > xmax + tol or y < ymin - tol or y > ymax + tol:
                continue
            on_edge = (
                abs(x - xmin) <= tol
                or abs(x - xmax) <= tol
                or abs(y - ymin) <= tol
                or abs(y - ymax) <= tol
            )
            if on_edge:
                found = True
                break
        assert found, f"hanging node {nid} not located on any parent edge"


def test_vertical_split_hanging_nodes_flagged():
    mesh = _make_quad_mesh()
    refiner = TensorRefiner(max_ref=3)
    rx = np.array([1], dtype=int)
    ry = np.array([0], dtype=int)
    refined = refiner.refine(mesh, rx, ry)

    assert len(refined.hanging_nodes) == 2  # midpoints on top/bottom edges of the split parent
    _assert_hanging_nodes_on_parent_edges(refined)
    for e in refined.edges_list:
        if e.right is None:
            continue
        assert e.left is not None, "interior edges must have both owners"


def test_balance_prevents_tjunction_blowup():
    mesh = _make_quad_mesh(nx=2, ny=1, p=1)
    refiner = TensorRefiner(max_ref=4)
    rx = np.array([2, 0], dtype=int)
    ry = np.zeros_like(rx)
    rx_bal, ry_bal = refiner.balance_levels(mesh, rx, ry)
    refined = refiner.refine(mesh, rx_bal, ry_bal)

    tj = refined.count_tjunction_violations(max_ratio=2.0)
    assert tj["count"] == 0
    _assert_hanging_nodes_on_parent_edges(refined)


def test_circle_levelset_refines_without_ownerless_edges():
    mesh = _make_quad_mesh(nx=2, ny=2, p=2)

    def circle_phi(xy):
        x = xy[..., 0]
        y = xy[..., 1]
        return np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2) - 0.3

    refiner = TensorRefiner(max_ref=3)
    marked = refiner.mark_near_levelset(mesh, circle_phi, band=0.15, levels=2)
    rx_plan, ry_plan = refiner.plan_tensor_levels(mesh, marked, target_h=0.15)
    rx_bal, ry_bal = refiner.balance_levels(mesh, rx_plan, ry_plan)
    refined = refiner.refine(mesh, rx_bal, ry_bal)

    _assert_hanging_nodes_on_parent_edges(refined)
    for e in refined.edges_list:
        if e.right is None:
            continue
        assert e.left is not None
