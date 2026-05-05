import numpy as np

from examples.poroelasticity.consolidation_pycutfem import _build_p2_tri_mesh, _generate_points, _structured_cells


def test_poroelasticity_pressure_load_boundary_split_length():
    L = 20.0
    H = 10.0
    nx = 31
    ny = 14
    pressure_region = 5.0

    points = _generate_points(L=L, H=H, nx=nx, ny=ny)
    cells = _structured_cells(nx=nx, ny=ny)
    mesh = _build_p2_tri_mesh(points, cells)

    top_tol = 0.5
    mesh.tag_boundary_edges(
        {
            "bottom": lambda x, y: np.isclose(y, 0.0),
            "left": lambda x, y: np.isclose(x, 0.0),
            "right": lambda x, y: np.isclose(x, L),
            "top_drained": lambda x, y: (abs(y - H) <= top_tol) and (x > pressure_region),
            # Important: strict split (x < pressure_region) so the edge whose midpoint is
            # exactly at x=pressure_region is not tagged as traction.
            "pressure_load": lambda x, y: (abs(y - H) <= top_tol) and (x < pressure_region),
        }
    )

    def edge_length(edge) -> float:
        n0 = int(edge.nodes[0])
        n1 = int(edge.nodes[-1])
        x0, y0 = mesh.nodes_x_y_pos[n0]
        x1, y1 = mesh.nodes_x_y_pos[n1]
        return float(np.hypot(x1 - x0, y1 - y0))

    pl_len = 0.0
    drained_len = 0.0
    top_len = 0.0
    for e in mesh.edges_list:
        if e.right is not None:
            continue
        n0 = int(e.nodes[0])
        n1 = int(e.nodes[-1])
        y0 = float(mesh.nodes_x_y_pos[n0][1])
        y1 = float(mesh.nodes_x_y_pos[n1][1])
        if abs(y0 - H) > top_tol or abs(y1 - H) > top_tol:
            continue
        L_e = edge_length(e)
        top_len += L_e
        if e.tag == "pressure_load":
            pl_len += L_e
        elif e.tag == "top_drained":
            drained_len += L_e

    dx = L / (nx - 1)
    # With the strict split, the top edge whose midpoint is at x=pressure_region stays untagged.
    expected_pl = 7.0 * dx
    expected_drained = 22.0 * dx
    expected_top = 30.0 * dx

    assert abs(pl_len - expected_pl) < 1.0e-12
    assert abs(drained_len - expected_drained) < 1.0e-12
    assert abs(top_len - expected_top) < 1.0e-12

