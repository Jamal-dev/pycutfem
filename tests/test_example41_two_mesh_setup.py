import numpy as np

from examples.FPI.fpi_mms_example41_two_mesh_setup import build_two_meshes


def test_example41_two_mesh_setup_builds_and_has_interface_segments():
    prob = build_two_meshes(nx_f=8, nx_p=4, poly_order=1, x0=-0.45)
    mesh_f = prob["mesh_f"]
    mesh_p = prob["mesh_p"]

    assert mesh_f.n_elements == 8 * 8
    assert mesh_p.n_elements == 4 * 4

    # There should be at least some cut elements and interface segments.
    n_cut = sum(1 for e in mesh_f.elements_list if getattr(e, "tag", "") == "cut")
    assert n_cut > 0

    n_seg = 0
    for e in mesh_f.elements_list:
        segs = getattr(e, "interface_segments", None)
        if segs:
            n_seg += len(segs)
    assert n_seg > 0

    inside = np.asarray(prob["inside_poro_nodes"], dtype=int)
    assert inside.size > 0

