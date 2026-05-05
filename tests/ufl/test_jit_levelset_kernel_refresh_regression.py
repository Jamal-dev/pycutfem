import numpy as np
import pytest

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.ufl.expressions import Function, TestFunction
from pycutfem.ufl.measures import dx, dInterface
from pycutfem.ufl.forms import Equation
from pycutfem.jit import compile_multi


@pytest.mark.parametrize("order", ["volume_then_interface", "interface_then_volume"])
def test_compile_multi_refresh_binds_per_integral_locals(order):
    """
    Regression test for a late-binding closure bug in jit.compile_multi builders.

    The bug manifested when an Equation had multiple integrals: builder callables
    captured loop variables (active_cols/qdeg/param_order/ir/integrand) by
    reference. On kernel.refresh(), earlier kernels would rebuild static args
    using the *last* integral's settings, causing shape mismatches or silently
    corrupted kernels (and Newton failures after level-set refresh).

    This test creates two integrals that activate different fields and reverses
    their order to ensure both volume and interface builders are exercised as
    the "non-last" integral.
    """
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )

    # Tilted line: true cut elements + interface segments in each cut element
    ls0 = AffineLevelSet(a=1.0, b=0.2, c=-0.55)
    ls1 = AffineLevelSet(a=1.0, b=0.2, c=-0.53)  # small shift; cut set unchanged

    mesh.classify_elements(ls0)
    mesh.classify_edges(ls0)
    mesh.build_interface_segments(ls0)

    # Two fields with different local DOF counts (Q2 vs Q1)
    me = MixedElement(mesh, field_specs={"u_pos_x": 2, "p_pos_": 1})
    dh = DofHandler(me, method="cg")

    u = Function("u_pos_x", field_name="u_pos_x", dof_handler=dh)
    v = TestFunction("u_pos_x", "u_pos_x", dh)
    p = Function("p_pos_", field_name="p_pos_", dof_handler=dh)
    q = TestFunction("p_pos_", "p_pos_", dh)

    a_vol = u * v * dx(level_set=ls0, metadata={"q": 2, "side": "+"})
    a_ifc = p * q * dInterface(level_set=ls0, metadata={"q": 2})
    if order == "volume_then_interface":
        eq = Equation(a_vol + a_ifc, None)
        target_domain = "volume"
        target_width = me.component_dof_slices["u_pos_x"].stop - me.component_dof_slices["u_pos_x"].start
    else:
        eq = Equation(a_ifc + a_vol, None)
        target_domain = "interface"
        target_width = me.component_dof_slices["p_pos_"].stop - me.component_dof_slices["p_pos_"].start

    kernels = compile_multi(eq, dof_handler=dh, mixed_element=me, backend="jit")
    targets = [
        k
        for k in kernels
        if getattr(k, "builder", None) is not None and getattr(k, "domain", None) == target_domain
    ]
    assert targets, f"expected at least one '{target_domain}' kernel with a refresh builder"

    for ker in targets:
        gdofs = np.asarray(ker.static_args.get("gdofs_map"))
        assert gdofs.ndim == 2
        assert gdofs.shape[1] == target_width

    # refresh after the compile_multi loop has finished (where the closure bug used to bite)
    mesh.classify_elements(ls1)
    mesh.classify_edges(ls1)
    mesh.build_interface_segments(ls1)
    for ker in targets:
        assert ker.refresh(ls1)
        gdofs = np.asarray(ker.static_args.get("gdofs_map"))
        assert gdofs.ndim == 2
        assert gdofs.shape[1] == target_width


def test_merge_static_arrays_pads_quadrature_axis_and_scalars():
    """
    Unit regression test for _merge_static_arrays:
    - allow old/new per-element arrays to have different nQ (pad to max)
    - handle scalar per-element arrays (ndim==1) without 0-d assignment errors
    """
    from pycutfem.jit import _merge_static_arrays

    target = np.asarray([10, 20, 30], dtype=np.int32)
    old = {
        "eids": target.copy(),
        "qw": np.zeros((3, 50), dtype=float),
        "phis": np.zeros((3, 50), dtype=float),
        "h_arr": np.asarray([1.0, 2.0, 3.0], dtype=float),
    }
    new = {
        "eids": np.asarray([20], dtype=np.int32),
        "qw": np.ones((1, 25), dtype=float),
        "phis": 2.0 * np.ones((1, 25), dtype=float),
        "h_arr": np.asarray([99.0], dtype=float),
    }
    merged = _merge_static_arrays(target, old, new)
    assert np.array_equal(np.asarray(merged["eids"], dtype=np.int32), target)
    assert merged["qw"].shape == (3, 50)
    assert merged["phis"].shape == (3, 50)
    assert merged["h_arr"].shape == (3,)

    # Only eid=20 should be overwritten (index 1), with padding to 50 QPs
    assert np.allclose(merged["qw"][0], 0.0)
    assert np.allclose(merged["qw"][2], 0.0)
    assert np.allclose(merged["qw"][1, :25], 1.0)
    assert np.allclose(merged["qw"][1, 25:], 0.0)
    assert np.allclose(merged["phis"][1, :25], 2.0)
    assert np.allclose(merged["phis"][1, 25:], 0.0)
    assert np.allclose(merged["h_arr"], np.asarray([1.0, 99.0, 3.0], dtype=float))

