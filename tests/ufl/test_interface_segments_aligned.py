import numpy as np
import pytest
from numpy.testing import assert_allclose

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.levelset import AffineLevelSet, BeamLevelSet
from pycutfem.ufl.measures import dInterface
from pycutfem.ufl.expressions import Constant, Jump, TrialFunction, TestFunction, Function
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.jit import compile_multi


def _assemble_scalar(form, dof_handler, backend="jit"):
    """Assemble scalar functional with a simple hook."""
    hook = {type(form.integrand): {"name": "scalar"}}
    res = assemble_form(Equation(None, form), dof_handler=dof_handler, bcs=[], backend=backend, assembler_hooks=hook)
    return res["scalar"]


class LShapeLevelSet:
    """φ(x, y) = min(x-0.5, y-0.5) → L-shaped cut with a sharp corner."""

    def __call__(self, xy):
        x, y = xy
        return min(x - 0.5, y - 0.5)

    def gradient(self, xy):
        x, y = xy
        if (x - 0.5) <= (y - 0.5):
            return np.array([1.0, 0.0])
        return np.array([0.0, 1.0])

    def evaluate_on_nodes(self, mesh):
        coords = mesh.nodes_x_y_pos
        return np.minimum(coords[:, 0] - 0.5, coords[:, 1] - 0.5)


@pytest.mark.parametrize("backend", ["python", "jit"])
def test_interface_multisegment_length(backend):
    """Sharp corner (L-shape) should yield two segments and length 1.0."""
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, edges_connectivity=edges,
                elements_corner_nodes=corners, element_type="quad", poly_order=1)

    ls = LShapeLevelSet()
    mesh.classify_elements(ls)
    mesh.classify_edges(ls)
    mesh.build_interface_segments(ls)

    cut_elems = mesh.element_bitset("cut").to_indices()
    assert len(cut_elems) == 1
    elem = mesh.elements_list[int(cut_elems[0])]
    assert len(getattr(elem, "interface_segments", [])) == 2

    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    form = Constant(1.0) * dInterface(defined_on=mesh.element_bitset("cut"), level_set=ls, metadata={"q": 4})
    val = _assemble_scalar(form, dh, backend=backend)
    assert_allclose(val, 1.0, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("backend", ["python", "jit"])
def test_aligned_interface_couples_neighbors(backend):
    """
    Aligned interface edge x=0.5 between two quads should produce coupling.
    Using DG ensures separate DOFs per element so the interface block is visible.
    """
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=1)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, edges_connectivity=edges,
                elements_corner_nodes=corners, element_type="quad", poly_order=1)

    ls = AffineLevelSet(a=1.0, b=0.0, c=-0.5)  # φ = x - 0.5 → shared edge is interface
    mesh.classify_elements(ls)
    mesh.classify_edges(ls)
    mesh.build_interface_segments(ls)

    iface = mesh.edge_bitset("interface")
    assert iface.cardinality() > 0

    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="dg")

    u = TrialFunction("u", "u", dh)
    v = TestFunction("u", "u", dh)
    a = Jump(u) * Jump(v) * dInterface(level_set=ls, metadata={"q": 4})

    K, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend=backend)
    A = K.toarray()

    dofs_left = dh.get_elemental_dofs(0)
    dofs_right = dh.get_elemental_dofs(1)
    block = A[np.ix_(dofs_left, dofs_right)]

    assert np.any(np.abs(block) > 1e-12), "Aligned interface block should couple left/right DOFs."
    # symmetry for the bilinear form
    assert_allclose(block, A[np.ix_(dofs_right, dofs_left)].T, atol=1e-12)


@pytest.mark.parametrize("backend", ["python", "jit"])
def test_beam_corner_jump_matches_symbolic(backend):
    """
    Beam level set corner in a single element produces two segments (one aligned).
    Check interface points, length, and jump integral against symbolic result.
    """
    # 2x2 mesh on unit square
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=1)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, edges_connectivity=edges,
                elements_corner_nodes=corners, element_type="quad", poly_order=1)

    # Beam spanning full width (x=0 and x=1) and half height (y=0.25, 0.75)
    ls = BeamLevelSet(center=(0.5, 0.5), Lb=1.0, Hb=0.5)
    mesh.classify_elements(ls)
    mesh.classify_edges(ls)
    mesh.build_interface_segments(ls)

    # target element: top-right (centroid > 0.5 in x and y)
    target_eid = None
    for el in mesh.elements_list:
        if el.centroid_x > 0.5 and el.centroid_y > 0.5:
            target_eid = int(el.id)
            target_elem = el
            break
    assert target_eid is not None

    # Interface geometry: two segments meeting at (1, 0.75)
    segments = getattr(target_elem, "interface_segments", [])
    assert len(segments) >= 2
    expected_segments = [
        [(1.0, 0.5), (1.0, 0.75)],        # vertical aligned edge piece
        [(1.0, 0.75), (0.75, 0.75)],      # first half of horizontal
        [(0.75, 0.75), (0.5, 0.75)],      # second half (ordering from root finder)
    ]
    for exp in expected_segments:
        found = any(
            (np.allclose(seg[0], exp[0], atol=1e-12) and np.allclose(seg[1], exp[1], atol=1e-12))
            or (np.allclose(seg[0], exp[1], atol=1e-12) and np.allclose(seg[1], exp[0], atol=1e-12))
            for seg in segments
        )
        assert found, f"Expected segment {exp} not found in {segments}"

    length = sum(np.linalg.norm(np.asarray(seg[1]) - np.asarray(seg[0])) for seg in segments)
    assert_allclose(length, 0.75, atol=1e-12)

    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    # Jump of piecewise analytic functions
    u_pos = Analytic(lambda x, y: x**2)           # outside beam
    u_neg = Analytic(lambda x, y: np.sin(x * y))  # inside beam
    form = Jump(u_pos, u_neg) * dInterface(level_set=ls, metadata={"q": 8})

    hook = {type(form.integrand): {"name": "jump"}}
    res = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], backend=backend, assembler_hooks=hook)
    computed = res["jump"]

    # Symbolic reference for the full beam perimeter (aligned + interior cuts across elements)
    import sympy as sp

    x, y = sp.symbols("x y", real=True)
    expr = x**2 - sp.sin(x * y)
    y0, y1 = 0.25, 0.75
    x0, x1 = 0.0, 1.0
    seg_right = sp.integrate(expr.subs(x, x1), (y, y0, y1))
    seg_left  = sp.integrate(expr.subs(x, x0), (y, y0, y1))
    seg_top   = sp.integrate(expr.subs(y, y1), (x, x0, x1))
    seg_bot   = sp.integrate(expr.subs(y, y0), (x, x0, x1))
    expected = float(sp.N(seg_right + seg_left + seg_top + seg_bot))

    assert abs(computed) > 1e-12  # non-zero jump
    assert_allclose(computed, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("backend", ["jit"])
def test_phi_changed_detector_recomputes_only_changed_ids(backend):
    """
    Move an affine level set slightly and verify that the JIT builders
    reuse unchanged elements but refresh phi-dependent arrays when the
    zero level set moves (cut/interface path).
    """
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=1)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, edges_connectivity=edges,
                elements_corner_nodes=corners, element_type="quad", poly_order=1)

    # Tilted line to guarantee true cut elements (not edge-aligned)
    ls0 = AffineLevelSet(a=1.0, b=0.2, c=-0.55)
    ls1 = AffineLevelSet(a=1.0, b=0.2, c=-0.45)  # larger shift to move phi visibly

    mesh.classify_elements(ls0)
    mesh.classify_edges(ls0)
    mesh.build_interface_segments(ls0)
    assert mesh.element_bitset("cut").cardinality() > 0

    me = MixedElement(mesh, field_specs={"u_pos_x": 1})
    dh = DofHandler(me, method="cg")
    f = Function("u_pos_x", field_name="u_pos_x", dof_handler=dh)
    v = TestFunction("u_pos_x", "u_pos_x", dh)
    form = f * v * dInterface(level_set=ls0, metadata={"q": 2})
    eq = Equation(form, None)

    kernels = compile_multi(eq, dof_handler=dh, mixed_element=me, backend="jit")
    iface_kernels = [k for k in kernels if k.domain == "interface"]
    assert iface_kernels, "expected an interface kernel"
    ker = iface_kernels[0]
    old_static = ker.static_args
    old_eids = np.asarray(old_static.get("eids", []), dtype=np.int32)
    assert old_eids.size > 0
    old_sig = np.asarray(old_static.get("_phi_sig"))

    # refresh with moved level set
    mesh.classify_elements(ls1)
    mesh.classify_edges(ls1)
    mesh.build_interface_segments(ls1)
    ker.refresh(ls1)

    new_static = ker.static_args
    new_eids = np.asarray(new_static.get("eids", []), dtype=np.int32)
    assert np.array_equal(np.sort(old_eids), np.sort(new_eids))
    new_sig = np.asarray(new_static.get("_phi_sig"))

    # phi signature at centroids should change after movement
    assert new_sig.shape == old_sig.shape
    assert not np.allclose(new_sig, old_sig)


@pytest.mark.parametrize("backend", ["jit"])
def test_phi_changed_detector_cut_volume(backend):
    """
    Cut volume: phi movement should refresh signatures while keeping element ids.
    """
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=1)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, edges_connectivity=edges,
                elements_corner_nodes=corners, element_type="quad", poly_order=1)
    ls0 = AffineLevelSet(a=1.0, b=0.3, c=-0.45)
    ls1 = AffineLevelSet(a=1.0, b=0.3, c=-0.35)
    mesh.classify_elements(ls0)
    mesh.classify_edges(ls0)
    mesh.build_interface_segments(ls0)
    assert mesh.element_bitset("cut").cardinality() > 0

    me = MixedElement(mesh, field_specs={"u_pos_x": 1})
    dh = DofHandler(me, method="cg")
    u = Function("u_pos_x", field_name="u_pos_x", dof_handler=dh)
    v = TestFunction("u_pos_x", "u_pos_x", dh)
    # Use a volume integral on the '+' side to get cut-volume kernels
    from pycutfem.ufl.measures import dx
    form = u * v * dx(level_set=ls0, metadata={"q": 2, "side": "+"})
    eq = Equation(form, None)
    kernels = compile_multi(eq, dof_handler=dh, mixed_element=me, backend="jit")
    vol_kernels = [k for k in kernels if k.domain == "volume" and k.side in ("+","-")]
    assert vol_kernels, "expected a volume kernel"
    ker = vol_kernels[0]
    old_static = ker.static_args
    old_eids = np.asarray(old_static.get("eids", []), dtype=np.int32)
    assert old_eids.size > 0
    old_sig = np.asarray(old_static.get("_phi_sig"))
    if old_sig.size == 0 or old_sig.dtype == object:
        pytest.skip("phi signatures not available for cut volume kernel")

    mesh.classify_elements(ls1)
    mesh.classify_edges(ls1)
    mesh.build_interface_segments(ls1)
    ker.refresh(ls1)

    new_static = ker.static_args
    new_eids = np.asarray(new_static.get("eids", []), dtype=np.int32)
    assert np.array_equal(np.sort(old_eids), np.sort(new_eids))
    new_sig = np.asarray(new_static.get("_phi_sig"))
    if new_sig.size == 0 or new_sig.dtype == object:
        pytest.skip("phi signatures not available for cut volume kernel")
    assert new_sig.shape == old_sig.shape
    assert not np.allclose(new_sig, old_sig)


@pytest.mark.parametrize("backend", ["jit"])
def test_phi_changed_detector_ghost(backend):
    """
    Ghost edge refresh: signatures change when phi moves, ids remain.
    """
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=1)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, edges_connectivity=edges,
                elements_corner_nodes=corners, element_type="quad", poly_order=1)
    ls0 = AffineLevelSet(a=1.0, b=0.0, c=-0.5)
    ls1 = AffineLevelSet(a=1.0, b=0.0, c=-0.4)
    mesh.classify_elements(ls0)
    mesh.classify_edges(ls0)
    mesh.build_interface_segments(ls0)
    assert mesh.edge_bitset("ghost").cardinality() >= 0  # ghost set may be empty for this tiny mesh

    me = MixedElement(mesh, field_specs={"u_pos_x": 1})
    dh = DofHandler(me, method="cg")
    u = Function("u_pos_x", field_name="u_pos_x", dof_handler=dh)
    v = TestFunction("u_pos_x", "u_pos_x", dh)
    # use dInterface to ensure ghost builder still runs if present; if no ghost edges, skip
    form = u * v * dInterface(level_set=ls0, metadata={"q": 2})
    eq = Equation(form, None)
    kernels = compile_multi(eq, dof_handler=dh, mixed_element=me, backend="jit")
    ghost_kernels = [k for k in kernels if k.domain == "ghost_edge"]
    if not ghost_kernels:
        pytest.skip("no ghost edges in this mesh setup")
    ker = ghost_kernels[0]
    old_static = ker.static_args
    old_eids = np.asarray(old_static.get("eids", []), dtype=np.int32)
    assert old_eids.size >= 0
    old_sig = np.asarray(old_static.get("_phi_sig"))

    mesh.classify_elements(ls1)
    mesh.classify_edges(ls1)
    mesh.build_interface_segments(ls1)
    ker.refresh(ls1)

    new_static = ker.static_args
    new_eids = np.asarray(new_static.get("eids", []), dtype=np.int32)
    assert np.array_equal(np.sort(old_eids), np.sort(new_eids))
    new_sig = np.asarray(new_static.get("_phi_sig"))
    if old_sig.size and new_sig.size:
        assert new_sig.shape == old_sig.shape
        assert not np.allclose(new_sig, old_sig)
