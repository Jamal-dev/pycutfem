import numpy as np
import pytest

from examples.biofilms.benchmarks.blauert.blauert_biofilm_deformation_one_domain import refine_around_biofilm_bbox
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.meshgen import structured_quad


def _build_refined_q2_rt_dofhandler():
    L = 5.5e-3
    H = 1.0e-3
    nodes, elems, _edges, corners = structured_quad(L, H, nx=12, ny=4, poly_order=2, offset=(0.0, 0.0))
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=np.asarray(elems, dtype=int),
        elements_corner_nodes=np.asarray(corners, dtype=int),
        element_type="quad",
        poly_order=2,
    )
    poly = np.asarray(
        [
            [0.999e-3, 0.0],
            [2.5e-3, 0.0],
            [2.5e-3, 0.4325e-3],
            [0.999e-3, 0.4325e-3],
            [0.999e-3, 0.0],
        ],
        dtype=float,
    )
    mesh = refine_around_biofilm_bbox(mesh, poly=poly, band=(L / 12.0) * 2.0, expand_layers=0, L=L, H=H)
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= 1.0e-12,
            "right": lambda x, y: abs(x - L) <= 1.0e-12,
            "bottom": lambda x, y: abs(y - 0.0) <= 1.0e-12,
            "top": lambda x, y: abs(y - H) <= 1.0e-12,
        }
    )
    dh = DofHandler(MixedElement(mesh, field_specs={"v": ("RT", 0), "p": ("DG", 0)}), method="cg")
    return L, H, mesh, dh


def test_hdiv_edge_trace_projection_handles_high_order_boundary_segments():
    L, H, mesh, dh = _build_refined_q2_rt_dofhandler()

    # All tagged boundary segments must map back to the parent RT edge entity.
    for tag in ("left", "right", "bottom", "top"):
        for gid in mesh.edge_bitset(tag).to_indices():
            dofs = dh.edge_trace_dofs("v", int(gid))
            assert len(dofs) == 1

    # Constant normal flux should integrate over the full physical boundary length,
    # not just the last high-order subsegment encountered.
    left_flux = dh.project_hdiv_boundary_flux("v", "left", 1.0)
    right_flux = dh.project_hdiv_boundary_flux("v", "right", 1.0)
    bottom_flux = dh.project_hdiv_boundary_flux("v", "bottom", 1.0)
    top_flux = dh.project_hdiv_boundary_flux("v", "top", 1.0)

    assert sum(left_flux.values()) == pytest.approx(H)
    assert sum(right_flux.values()) == pytest.approx(H)
    assert sum(bottom_flux.values()) == pytest.approx(L)
    assert sum(top_flux.values()) == pytest.approx(L)
