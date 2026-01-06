import numpy as np
import pytest
from numpy.testing import assert_allclose

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.levelset import AffineLevelSet, BeamLevelSet
from pycutfem.core.sideconvention import SIDE
from pycutfem.ufl.measures import dInterface
from pycutfem.ufl.expressions import (
    Constant,
    Jump,
    TrialFunction,
    TestFunction,
    Function,
    Pos,
    Neg,
    grad,
    dot,
    FacetNormal,
    CellDiameter,
)
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


def _aligned_mesh(poly_order: int = 1):
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=poly_order)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )
    ls = AffineLevelSet(a=1.0, b=0.0, c=-0.5)  # φ = x - 0.5 (aligned interior edge)
    mesh.classify_elements(ls)
    mesh.classify_edges(ls)
    return mesh, ls


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
def test_aligned_rectangle_interface_perimeter(backend):
    """
    Rectangle aligned with mesh lines should be handled purely by aligned-interface assembly.
    """
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=4, ny=4, poly_order=1)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, edges_connectivity=edges,
                elements_corner_nodes=corners, element_type="quad", poly_order=1)

    # Beam edges at x/y = 0.25, 0.75 align with the grid.
    ls = BeamLevelSet(center=(0.5, 0.5), Lb=0.5, Hb=0.5)
    mesh.classify_elements(ls)
    mesh.classify_edges(ls)
    mesh.build_interface_segments(ls)

    assert mesh.edge_bitset("interface").cardinality() > 0
    assert mesh.element_bitset("cut").cardinality() == 0

    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    form = Constant(1.0) * dInterface(level_set=ls, metadata={"q": 4})
    val = _assemble_scalar(form, dh, backend=backend)
    assert_allclose(val, 2.0, rtol=1e-6, atol=1e-6)


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
    # Keep the cut element set unchanged (still crosses x=0.5 somewhere in y∈[0,1])
    ls1 = AffineLevelSet(a=1.0, b=0.2, c=-0.53)

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
    cut_ids = np.asarray(mesh.element_bitset("cut").to_indices(), dtype=np.int32)
    assert cut_ids.size > 0

    me = MixedElement(mesh, field_specs={"u_pos_x": 1})
    dh = DofHandler(me, method="cg")
    u = Function("u_pos_x", field_name="u_pos_x", dof_handler=dh)
    v = TestFunction("u_pos_x", "u_pos_x", dh)
    # Use a volume integral on the '+' side to get cut-volume kernels
    from pycutfem.ufl.measures import dx
    form = u * v * dx(level_set=ls0, metadata={"q": 2, "side": "+"})
    eq = Equation(form, None)
    kernels = compile_multi(eq, dof_handler=dh, mixed_element=me, backend="jit")
    vol_kernels = [k for k in kernels if k.domain == "volume" and k.side in ("+", "-")]
    assert vol_kernels, "expected a volume kernel"
    ker = None
    for k in vol_kernels:
        eids_arr = np.asarray(k.static_args.get("eids", []), dtype=np.int32)
        if np.array_equal(np.sort(eids_arr), np.sort(cut_ids)) or k.static_args.get("_phi_sig") is not None:
            ker = k
            break
    assert ker is not None, "expected a cut-volume kernel with phi signatures"
    old_static = ker.static_args
    old_eids = np.asarray(old_static.get("eids", []), dtype=np.int32)
    assert np.array_equal(np.sort(old_eids), np.sort(cut_ids))
    old_sig = np.asarray(old_static.get("_phi_sig"), dtype=float)
    assert old_sig.size > 0

    mesh.classify_elements(ls1)
    mesh.classify_edges(ls1)
    mesh.build_interface_segments(ls1)
    ker.refresh(ls1)

    new_static = ker.static_args
    new_eids = np.asarray(new_static.get("eids", []), dtype=np.int32)
    assert np.array_equal(np.sort(old_eids), np.sort(new_eids))
    new_sig = np.asarray(new_static.get("_phi_sig"), dtype=float)
    assert new_sig.shape == old_sig.shape
    assert not np.allclose(new_sig, old_sig)


def test_aligned_interface_tagging_no_cut():
    mesh, ls = _aligned_mesh()
    cut = mesh.element_bitset("cut")
    inside = mesh.element_bitset("inside")
    outside = mesh.element_bitset("outside")
    iface = mesh.edge_bitset("interface")
    ghost = mesh.edge_bitset("ghost")

    assert cut.cardinality() == 0
    assert inside.cardinality() == 1
    assert outside.cardinality() == 1
    assert iface.cardinality() == 1
    assert ghost.cardinality() == 0

    tol = 10.0 * SIDE.tol
    for gid in iface.to_indices():
        e = mesh.edge(int(gid))
        node_ids = e.all_nodes if e.all_nodes else e.nodes
        coords = mesh.nodes_x_y_pos[list(node_ids)]
        assert np.allclose(coords[:, 0], 0.5, atol=1e-12)
        phi_vals = np.array([ls(np.asarray(pt, float)) for pt in coords], dtype=float)
        assert np.all(np.abs(phi_vals) <= tol)


@pytest.mark.parametrize("backend", ["jit"])
def test_compile_multi_aligned_interface_edges(backend):
    mesh, ls = _aligned_mesh()
    mesh.build_interface_segments(ls)
    iface_edges = np.asarray(mesh.edge_bitset("interface").to_indices(), dtype=np.int32)
    assert iface_edges.size > 0

    me = MixedElement(mesh, field_specs={"u_pos_x": 1})
    dh = DofHandler(me, method="cg")
    u = Function("u_pos_x", field_name="u_pos_x", dof_handler=dh)
    v = TestFunction("u_pos_x", "u_pos_x", dh)
    form = u * v * dInterface(level_set=ls, metadata={"q": 2})
    eq = Equation(form, None)

    kernels = compile_multi(eq, dof_handler=dh, mixed_element=me, backend=backend)
    iface_kernels = [k for k in kernels if k.domain == "interface"]
    assert iface_kernels, "expected interface kernels for aligned edges"

    edge_kernels = [k for k in iface_kernels if k.static_args.get("entity_kind") == "edge"]
    assert edge_kernels, "expected aligned-edge interface kernel"
    edge_ids = np.unique(np.concatenate([np.asarray(k.static_args.get("eids", []), dtype=np.int32) for k in edge_kernels]))
    assert np.array_equal(np.sort(edge_ids), np.sort(iface_edges))

    cut_kernels = [k for k in iface_kernels if k.static_args.get("entity_kind") == "element"]
    assert cut_kernels, "expected cut-interface kernel"
    for k in cut_kernels:
        eids = np.asarray(k.static_args.get("eids", []), dtype=np.int32)
        assert eids.size == 0


def test_aligned_interface_tagging_high_order_nodes():
    for poly_order in (2, 3):
        mesh, ls = _aligned_mesh(poly_order=poly_order)
        iface = mesh.edge_bitset("interface")
        assert iface.cardinality() == 1

        tol = 10.0 * SIDE.tol
        for gid in iface.to_indices():
            e = mesh.edge(int(gid))
            assert e.all_nodes, "expected edge.all_nodes for high-order edges"
            assert len(e.all_nodes) > len(e.nodes)
            coords = mesh.nodes_x_y_pos[list(e.all_nodes)]
            phi_vals = np.array([ls(np.asarray(pt, float)) for pt in coords], dtype=float)
            assert np.all(np.abs(phi_vals) <= tol)


@pytest.mark.parametrize("backend", ["python", "jit", "cpp"])
def test_aligned_interface_length_single_count(backend):
    mesh, ls = _aligned_mesh()
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")
    form = Constant(1.0) * dInterface(level_set=ls, metadata={"q": 4})
    val = _assemble_scalar(form, dh, backend=backend)
    assert_allclose(val, 1.0, rtol=1e-10, atol=1e-10)


def test_aligned_interface_quadrature_geometry():
    mesh, ls = _aligned_mesh()
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")
    iface = mesh.edge_bitset("interface")
    assert iface.cardinality() > 0

    geo = dh.precompute_ghost_factors(
        ghost_edge_ids=iface,
        qdeg=4,
        level_set=ls,
        derivs={(0, 0)},
        allow_interface=True,
        reuse=False,
    )

    qp = geo["qp_phys"]
    qw = geo["qw"]
    normals = geo["normals"]
    tol = 1.0e-6
    for ei in range(qp.shape[0]):
        xq = qp[ei, :, 0]
        assert np.allclose(xq, 0.5, atol=tol)
        phi_vals = np.array([ls(np.asarray(pt, float)) for pt in qp[ei]], dtype=float)
        assert np.all(np.abs(phi_vals) <= tol)
        edge_len = float(np.sum(qw[ei]))
        assert_allclose(edge_len, 1.0, rtol=1e-8, atol=1e-8)
        nrm = normals[ei]
        nmag = np.linalg.norm(nrm, axis=1)
        assert np.allclose(nmag, 1.0, atol=1e-8)
        assert np.all(nrm[:, 0] > 0.99)


def test_aligned_interface_restriction_masks():
    mesh, ls = _aligned_mesh()
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="dg")
    iface = mesh.edge_bitset("interface")
    assert iface.cardinality() > 0

    geo = dh.precompute_ghost_factors(
        ghost_edge_ids=iface,
        qdeg=2,
        level_set=ls,
        derivs={(0, 0)},
        allow_interface=True,
        reuse=False,
    )

    coords = dh.get_all_dof_coords()
    pos_masks = geo["restrict_mask_u_pos"]
    neg_masks = geo["restrict_mask_u_neg"]
    pos_maps = geo["pos_map_u"]
    neg_maps = geo["neg_map_u"]
    gdofs_map = geo["gdofs_map"]
    tol = 10.0 * SIDE.tol

    for ei in range(pos_masks.shape[0]):
        union_size = int(max(pos_maps[ei].max(), neg_maps[ei].max()) + 1)
        union_gdofs = gdofs_map[ei][:union_size]

        pos_union = np.unique(pos_maps[ei][pos_maps[ei] >= 0])
        pos_gdofs = union_gdofs[pos_union]
        pos_phi = np.array([ls(np.asarray(coords[int(d)], float)) for d in pos_gdofs], dtype=float)
        pos_iface = np.abs(pos_phi) <= tol
        assert np.any(pos_iface)
        assert np.all(pos_masks[ei][pos_union][pos_iface] > 0.5)

        neg_union = np.unique(neg_maps[ei][neg_maps[ei] >= 0])
        neg_gdofs = union_gdofs[neg_union]
        neg_phi = np.array([ls(np.asarray(coords[int(d)], float)) for d in neg_gdofs], dtype=float)
        neg_iface = np.abs(neg_phi) <= tol
        assert np.any(neg_iface)
        assert np.all(neg_masks[ei][neg_union][neg_iface] > 0.5)


@pytest.mark.parametrize("backend", ["python", "jit", "cpp"])
def test_aligned_interface_penalty_block_exact(backend):
    mesh, ls = _aligned_mesh()
    assert mesh.element_bitset("cut").cardinality() == 0

    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="dg")
    u = TrialFunction("u", "u", dh)
    v = TestFunction("u", "u", dh)
    form = Jump(u) * Jump(v) * dInterface(level_set=ls, metadata={"q": 4})
    K, _ = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], backend=backend)
    A = K.toarray()

    coords = dh.get_all_dof_coords()
    e0, e1 = mesh.elements_list[0], mesh.elements_list[1]
    left_eid = 0 if e0.centroid_x < e1.centroid_x else 1
    right_eid = 1 - left_eid
    dofs_left = dh.element_maps["u"][left_eid]
    dofs_right = dh.element_maps["u"][right_eid]
    tol = 1.0e-12
    iface_left = [d for d in dofs_left if abs(coords[int(d)][0] - 0.5) <= 10.0 * tol]
    iface_right = [d for d in dofs_right if abs(coords[int(d)][0] - 0.5) <= 10.0 * tol]
    assert len(iface_left) == 2
    assert len(iface_right) == 2
    iface_left = sorted(iface_left, key=lambda d: coords[int(d)][1])
    iface_right = sorted(iface_right, key=lambda d: coords[int(d)][1])

    idx = iface_left + iface_right
    block = A[np.ix_(idx, idx)]
    M = np.array([[1.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 3.0]])
    expected = np.block([[M, -M], [-M, M]])
    tol_mat = 1.0e-10 if backend == "python" else 1.0e-8
    assert_allclose(block, expected, rtol=tol_mat, atol=tol_mat)


def _aligned_interface_jump_energy(shift: float, backend: str) -> float:
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    ls = AffineLevelSet(a=1.0, b=0.0, c=-(0.5 + shift))
    mesh.classify_elements(ls)
    mesh.classify_edges(ls)
    mesh.build_interface_segments(ls)

    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="dg")

    f_pos_a = Analytic(lambda x, y: x)
    f_pos_b = Analytic(lambda x, y: x)
    f_neg_a = Analytic(lambda x, y: 0.0)
    f_neg_b = Analytic(lambda x, y: 0.0)
    jump_a = Pos(f_pos_a) - Neg(f_neg_a)
    jump_b = Pos(f_pos_b) - Neg(f_neg_b)
    form = jump_a * jump_b * dInterface(level_set=ls, metadata={"q": 4})
    hook = {type(form.integrand): {"name": "E"}}
    res = assemble_form(Equation(None, form), dof_handler=dh, bcs=[], backend=backend, assembler_hooks=hook)
    return float(np.asarray(res["E"]).item())


@pytest.mark.parametrize("backend", ["python", "jit", "cpp"])
def test_aligned_interface_cut_limit_continuity(backend):
    val_aligned = _aligned_interface_jump_energy(0.0, backend)
    val_large = _aligned_interface_jump_energy(1.0e-2, backend)
    val_small = _aligned_interface_jump_energy(2.0e-3, backend)

    diff_large = abs(val_large - val_aligned)
    diff_small = abs(val_small - val_aligned)
    assert diff_small < diff_large
    assert diff_small < 0.5 * diff_large


@pytest.mark.parametrize("backend", ["python", "jit", "cpp"])
def test_aligned_interface_nitsche_symmetry(backend):
    mesh, ls = _aligned_mesh()
    assert mesh.element_bitset("cut").cardinality() == 0

    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="dg")

    u = TrialFunction("u", "u", dh)
    v = TestFunction("u", "u", dh)
    n = FacetNormal()
    h = CellDiameter()

    jump_u = Pos(u) - Neg(u)
    jump_v = Pos(v) - Neg(v)
    avg_flux_u = 0.5 * (dot(grad(Pos(u)), n) + dot(grad(Neg(u)), n))
    avg_flux_v = 0.5 * (dot(grad(Pos(v)), n) + dot(grad(Neg(v)), n))
    beta = Constant(20.0)

    form = (avg_flux_u * jump_v + avg_flux_v * jump_u + (beta / h) * jump_u * jump_v) * dInterface(
        level_set=ls, metadata={"q": 4}
    )
    K, _ = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], backend=backend)
    A = K.toarray()
    assert_allclose(A, A.T, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("backend", ["jit"])
def test_phi_changed_detector_ghost(backend):
    """
    Ghost edge refresh: signatures change when phi moves, ids remain.
    """
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=1)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, edges_connectivity=edges,
                elements_corner_nodes=corners, element_type="quad", poly_order=1)
    ls0 = AffineLevelSet(a=1.0, b=0.2, c=-0.45)
    ls1 = AffineLevelSet(a=1.0, b=0.2, c=-0.35)
    mesh.classify_elements(ls0)
    mesh.classify_edges(ls0)
    mesh.build_interface_segments(ls0)
    ghost_ids = np.asarray(mesh.edge_bitset("ghost").to_indices(), dtype=np.int32)
    assert ghost_ids.size > 0, "expected at least one ghost edge in this setup"

    me = MixedElement(mesh, field_specs={"u_pos_x": 1})
    dh = DofHandler(me, method="cg")
    u = Function("u_pos_x", field_name="u_pos_x", dof_handler=dh)
    v = TestFunction("u_pos_x", "u_pos_x", dh)
    from pycutfem.ufl.measures import dGhost
    form = u * v * dGhost(level_set=ls0, metadata={"q": 2})
    eq = Equation(form, None)
    kernels = compile_multi(eq, dof_handler=dh, mixed_element=me, backend="jit")
    ghost_kernels = [k for k in kernels if k.domain == "ghost_edge"]
    assert ghost_kernels, "expected ghost-edge kernel"
    ker = ghost_kernels[0]
    old_static = ker.static_args
    old_eids = np.asarray(old_static.get("eids", []), dtype=np.int32)
    assert np.array_equal(np.sort(old_eids), np.sort(ghost_ids))
    old_sig = np.asarray(old_static.get("_phi_sig"), dtype=float)
    assert old_sig.size > 0

    mesh.classify_elements(ls1)
    mesh.classify_edges(ls1)
    mesh.build_interface_segments(ls1)
    ker.refresh(ls1)

    new_static = ker.static_args
    new_eids = np.asarray(new_static.get("eids", []), dtype=np.int32)
    assert np.array_equal(np.sort(old_eids), np.sort(new_eids))
    new_sig = np.asarray(new_static.get("_phi_sig"), dtype=float)
    assert new_sig.shape == old_sig.shape
    assert not np.allclose(new_sig, old_sig)
