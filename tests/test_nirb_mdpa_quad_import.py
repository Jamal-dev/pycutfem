from pathlib import Path

import numpy as np

from examples.NIRB.double_flap_reference import MDPAMesh
from examples.NIRB.run_example2_local import _mdpa_mesh_to_pycutfem


def test_mdpa_quad_import_rotates_distorted_quad_to_bottom_left_start() -> None:
    # Kratos can store Q4 connectivity in a rotated perimeter order. The
    # pycutfem Q1 transform expects the canonical geometry order
    #   corner_connectivity = [bl, br, tr, tl]
    # and the row-major lattice order
    #   elements_connectivity = [bl, br, tl, tr].
    # This distorted hotspot-like quad has a slightly tilted bottom edge, so a
    # plain "min y" start choice would incorrectly start at bottom-right.
    mdpa = MDPAMesh(
        path=Path("/tmp/distorted_quad.mdpa"),
        element_block="TotalLagrangianElement2D4N",
        condition_blocks=(),
        nodes={
            1: (1.52, 0.28),        # top-right
            2: (1.515, 0.28),       # top-left
            3: (1.51495687, 0.27458757),  # bottom-left
            4: (1.52, 0.2744),      # bottom-right
        },
        elements={1: (1, 2, 3, 4)},
        conditions={},
        submodelparts={},
    )

    mesh = _mdpa_mesh_to_pycutfem(
        mdpa=mdpa,
        domain_tag="solid",
        boundary_condition_tags={},
    )

    old_ids = np.asarray(mesh._mdpa_new_to_old_node, dtype=int)
    imported_corners = old_ids[np.asarray(mesh.corner_connectivity[0], dtype=int)]
    imported_lattice = old_ids[np.asarray(mesh.elements_connectivity[0], dtype=int)]

    assert imported_corners.tolist() == [3, 4, 1, 2]
    assert imported_lattice.tolist() == [3, 4, 2, 1]
