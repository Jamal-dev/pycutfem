import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.utils.refinement import TensorRefiner


def test_lie_bbox_refinement_one_level_produces_hanging_nodes() -> None:
    # Import lazily to avoid paying the cost for tests that don't need this benchmark module.
    from examples.biofilms.benchmarks.lie.lie_synthetic_deformation_one_domain import _channel_minus_block_mesh

    L = 15.0e-3
    H = 10.0e-3
    block_w = 1.0e-3
    block_h = 3.0e-3
    block_xc = 0.5 * L
    block_x0 = block_xc - 0.5 * block_w
    block_x1 = block_xc + 0.5 * block_w

    nodes, elems, corners = _channel_minus_block_mesh(
        L=L,
        H=H,
        block_x0=block_x0,
        block_x1=block_x1,
        block_h=block_h,
        nx_left=10,
        nx_mid=6,
        nx_right=10,
        ny_bottom=4,
        ny_top=6,
        poly_order=2,
    )
    mesh0 = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=2)

    # Refine a small rectangle near the block-top attachment region (not the whole mesh).
    x0_ref = block_x0 - 0.75e-3
    x1_ref = block_x1 + 1.25e-3
    y0_ref = block_h - 0.25e-3
    y1_ref = block_h + 2.50e-3

    corners_all = np.asarray(mesh0.nodes_x_y_pos[np.asarray(mesh0.corner_connectivity, dtype=int)], dtype=float)
    ex_min = corners_all[..., 0].min(axis=1)
    ex_max = corners_all[..., 0].max(axis=1)
    ey_min = corners_all[..., 1].min(axis=1)
    ey_max = corners_all[..., 1].max(axis=1)
    marked = np.nonzero((ex_max >= x0_ref) & (ex_min <= x1_ref) & (ey_max >= y0_ref) & (ey_min <= y1_ref))[0]
    assert int(marked.size) > 0
    assert int(marked.size) < int(len(mesh0.elements_list))

    rx = np.zeros(len(mesh0.elements_list), dtype=int)
    ry = np.zeros(len(mesh0.elements_list), dtype=int)
    rx[marked] = 1
    ry[marked] = 1

    refiner = TensorRefiner(max_ratio=2.0, max_ref=1)
    mesh1 = refiner.refine(mesh0, rx, ry)

    assert len(mesh1.elements_list) > len(mesh0.elements_list)
    hanging = getattr(mesh1, "hanging_nodes", []) or []
    assert len(hanging) > 0
    assert max(hanging) < len(mesh1.nodes_list)
