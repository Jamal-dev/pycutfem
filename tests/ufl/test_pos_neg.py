import numpy as np
from collections import defaultdict
import pytest

# --- Core imports from the project ---
from pycutfem.core.mesh import Mesh
from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.helpers import HelpersFieldAware as _hfa

# --- UFL / Compiler imports ---
from pycutfem.ufl.expressions import (
    TrialFunction, TestFunction, Function,
    Pos, Neg, Jump
)
from pycutfem.ufl.compilers import FormCompiler


# ------------------------------ Test helpers ------------------------------

def _ghost_ctx(dh, fields, edge, pos_eid, neg_eid, level_set):
    """
    Build a context for a *ghost edge* quadrature point that matches the compiler’s
    own ghost-edge assembly:
      - global_dofs = union(pos_eid, neg_eid)
      - pos_map / neg_map = scatter maps from each element’s union-DOFs -> global_dofs
      - per-field maps: local(field) -> global_dofs
    """
    mesh = dh.mixed_element.mesh

    # midpoint from endpoint node coordinates
    xq = mesh.nodes_x_y_pos[list(edge.nodes)].mean(axis=0)

    # element-union DOFs per side
    pos_dofs = dh.get_elemental_dofs(pos_eid)
    neg_dofs = dh.get_elemental_dofs(neg_eid)

    # union across neighbors (sorted + unique is fine for tests)
    global_dofs = np.unique(np.concatenate([pos_dofs, neg_dofs]))

    # side-wide scatter maps (lengths: len(pos_dofs) / len(neg_dofs))
    pos_map = np.searchsorted(global_dofs, pos_dofs)
    neg_map = np.searchsorted(global_dofs, neg_dofs)

    # per-field padding maps (length = local DOFs of each field)
    pos_map_by_field, neg_map_by_field = _hfa.build_field_union_maps(
        dh, fields, pos_eid, neg_eid, global_dofs
    )

    # per-QP, per-side basis cache like the compiler uses
    basis_values = {"+": defaultdict(dict), "-": defaultdict(dict)}

    return {
        "pos_eid": pos_eid,
        "neg_eid": neg_eid,
        "x_phys": xq,
        "phi_val": float(level_set(xq)),   # the compiler sets this too
        "basis_values": basis_values,

        # union-of-neighbors layout + maps
        "global_dofs": global_dofs,
        "pos_map": pos_map,
        "neg_map": neg_map,
        "pos_map_by_field": pos_map_by_field,
        "neg_map_by_field": neg_map_by_field,
    }, xq


def _interface_ctx(dh, fields, eid, p0, p1, level_set):
    """
    Build a context for an *interface point inside a cut element* that matches the
    compiler’s interface assembly:
      - global_dofs = element DOFs (same owner element on both sides)
      - pos/neg maps are identical identities
      - per-field maps built against `global_dofs`
    """
    xq = 0.5 * (np.asarray(p0) + np.asarray(p1))
    global_dofs = dh.get_elemental_dofs(eid)

    # identity side-wide maps in the element layout
    pos_map = np.arange(len(global_dofs), dtype=int)
    neg_map = pos_map

    pos_map_by_field, neg_map_by_field = _hfa.build_field_union_maps(
        dh, fields, eid, eid, global_dofs
    )

    # per-QP, per-side cache; normal is sometimes used downstream
    g = level_set.gradient(xq)
    nrm = g / (np.linalg.norm(g) + 1e-30)
    basis_values = {"+": defaultdict(dict), "-": defaultdict(dict)}

    return {
        "eid": eid,
        "is_interface": True,
        "pos_eid": eid,
        "neg_eid": eid,
        "x_phys": xq,
        "phi_val": 0.0,                 # on Γ the compiler sets ~0
        "normal": nrm,
        "basis_values": basis_values,

        # element layout + maps
        "global_dofs": global_dofs,
        "pos_map": pos_map,
        "neg_map": neg_map,
        "pos_map_by_field": pos_map_by_field,
        "neg_map_by_field": neg_map_by_field,
    }, xq


def _eval_scalar(fc, expr):
    """Collapse VecOpInfo/function results at a single QP to a scalar."""
    out = fc._visit(expr)
    data = getattr(out, "data", out)
    return float(np.asarray(data).sum())


# ------------------------------ Fixtures ------------------------------

@pytest.fixture(scope="module")
def setup_multifield_environment():
    """
    Multi-field mesh with '+' and '−' fields for a generic (u,p) pair.
    Also provides one interior ghost edge and one cut element interface.
    """
    poly_order = 2
    L, H = 2.0, 1.0
    center = (L/2, H/2)
    radius = 0.4

    nodes, elem_conn, _, corner_nodes = structured_quad(
        L, H, nx=4, ny=2, poly_order=poly_order
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elem_conn,
        elements_corner_nodes=corner_nodes,
        element_type="quad",
        poly_order=poly_order
    )
    level_set = CircleLevelSet(center=center, radius=radius)

    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)

    # two-field-per-side layout (u,p on + and −)
    me = MixedElement(mesh, field_specs={
        'u_pos': poly_order, 'p_pos': poly_order - 1,
        'u_neg': poly_order, 'p_neg': poly_order - 1
    })
    dh = DofHandler(me, method="cg")
    fields = list(me.field_names)

    # pick a valid interior ghost edge (has left and right)
    ghost_indices = mesh.edge_bitset('ghost').to_indices()
    ghost_edge_data = None
    for edge_gid in ghost_indices:
        edge = mesh.edge(edge_gid)
        if edge.left is not None and edge.right is not None:
            phi_left  = level_set(mesh.elements_list[edge.left].centroid())
            phi_right = level_set(mesh.elements_list[edge.right].centroid())
            pos_eid = edge.left if phi_left >= phi_right else edge.right
            neg_eid = edge.right if phi_left >= phi_right else edge.left
            ghost_edge_data = {"edge": edge, "pos_eid": pos_eid, "neg_eid": neg_eid}
            break
    if ghost_edge_data is None:
        pytest.fail("No suitable ghost edge found.")

    # pick a cut element with exactly one segment
    interface_data = None
    for el in mesh.elements_list:
        if el.tag == 'cut' and el.interface_pts and len(el.interface_pts) == 2:
            interface_data = {"eid": el.id, "interface_pts": el.interface_pts}
            break
    if interface_data is None:
        pytest.fail("No suitable cut element with an interface found.")

    return {
        "mesh": mesh,
        "level_set": level_set,
        "me": me,
        "dh": dh,
        "fields": fields,
        "ghost_edge_data": ghost_edge_data,
        "interface_data": interface_data
    }


# ------------------------------ Ghost-edge tests ------------------------------

def test_ghost_pos_and_neg_on_own_fields(setup_multifield_environment):
    """
    On a ghost edge:
      Pos(u_pos)(xq) == u_pos(xq on + owner),
      Neg(u_neg)(xq) == u_neg(xq on − owner).
    (No expectation that Pos(u_neg) or Neg(u_pos) be zero.)
    """
    dh = setup_multifield_environment["dh"]
    fields = setup_multifield_environment["fields"]
    level_set = setup_multifield_environment["level_set"]
    edge   = setup_multifield_environment["ghost_edge_data"]["edge"]
    pos_eid= setup_multifield_environment["ghost_edge_data"]["pos_eid"]
    neg_eid= setup_multifield_environment["ghost_edge_data"]["neg_eid"]

    # simple linear polynomials (exactly represented by Q2)
    u_pos = Function("u_pos_func", "u_pos", dh); u_pos.set_values_from_function(lambda x,y: x + 2*y)
    u_neg = Function("u_neg_func", "u_neg", dh); u_neg.set_values_from_function(lambda x,y: 3*x - y)

    fc = FormCompiler(dh, backend="python")
    ctx, xq = _ghost_ctx(dh, fields, edge, pos_eid, neg_eid, level_set)
    fc.ctx.update(ctx)

    # Pos on '+' field
    got = _eval_scalar(fc, Pos(u_pos))
    exp = xq[0] + 2*xq[1]
    assert np.isclose(got, exp, rtol=1e-3), f"Pos(u_pos) got {got}, expected {exp}"

    # Neg on '−' field
    got = _eval_scalar(fc, Neg(u_neg))
    exp = 3*xq[0] - xq[1]
    assert np.isclose(got, exp, rtol=1e-3), f"Neg(u_neg) got {got}, expected {exp}"


def test_ghost_jump_mixed_and_same_field_sanity(setup_multifield_environment):
    """
    On a ghost edge:
      Jump(u_pos, u_neg) == u_pos(+) − u_neg(−),
      Jump(u_pos, u_pos) ≈ 0 and Jump(u_neg, u_neg) ≈ 0 (orientation sanity).
    """
    dh = setup_multifield_environment["dh"]
    fields = setup_multifield_environment["fields"]
    level_set = setup_multifield_environment["level_set"]
    edge   = setup_multifield_environment["ghost_edge_data"]["edge"]
    pos_eid= setup_multifield_environment["ghost_edge_data"]["pos_eid"]
    neg_eid= setup_multifield_environment["ghost_edge_data"]["neg_eid"]

    u_pos = Function("u_pos_func", "u_pos", dh); u_pos.set_values_from_function(lambda x,y: x + 2*y)
    u_neg = Function("u_neg_func", "u_neg", dh); u_neg.set_values_from_function(lambda x,y: 3*x - y)

    fc = FormCompiler(dh, backend="python")
    ctx, xq = _ghost_ctx(dh, fields, edge, pos_eid, neg_eid, level_set)
    fc.ctx.update(ctx)

    # Mixed jump
    got = _eval_scalar(fc, Jump(u_pos, u_neg))
    exp = (xq[0] + 2*xq[1]) - (3*xq[0] - xq[1])
    assert np.isclose(got, exp, rtol=1e-3), f"Jump(u_pos,u_neg) got {got}, expected {exp}"

    # Same-field jumps ~ 0
    assert np.isclose(_eval_scalar(fc, Jump(u_pos, u_pos)), 0.0, atol=1e-12)
    assert np.isclose(_eval_scalar(fc, Jump(u_neg, u_neg)), 0.0, atol=1e-12)


# ------------------------------ Interface tests ------------------------------

def test_interface_pos_and_neg_on_own_fields(setup_multifield_environment):
    """
    On the interface midpoint inside a cut element:
      Pos(u_pos) and Neg(u_neg) evaluate to the expected values.
    """
    dh   = setup_multifield_environment["dh"]
    fs   = setup_multifield_environment["fields"]
    eid  = setup_multifield_environment["interface_data"]["eid"]
    p0,p1= setup_multifield_environment["interface_data"]["interface_pts"]
    phi  = setup_multifield_environment["level_set"]

    u_pos = Function("u_pos_func", "u_pos", dh); u_pos.set_values_from_function(lambda x,y: x + 2*y)
    u_neg = Function("u_neg_func", "u_neg", dh); u_neg.set_values_from_function(lambda x,y: 3*x - y)

    fc = FormCompiler(dh, backend="python")
    ctx, xq = _interface_ctx(dh, fs, eid, p0, p1, phi)
    fc.ctx.update(ctx)

    got = _eval_scalar(fc, Pos(u_pos)); exp = xq[0] + 2*xq[1]
    assert np.isclose(got, exp, rtol=1e-3)

    got = _eval_scalar(fc, Neg(u_neg)); exp = 3*xq[0] - xq[1]
    assert np.isclose(got, exp, rtol=1e-3)


def test_interface_jump_mixed_and_same_field_sanity(setup_multifield_environment):
    """
    On the interface midpoint inside a cut element:
      Jump(u_pos, u_neg) == u_pos − u_neg and same-field jumps vanish.
    """
    dh   = setup_multifield_environment["dh"]
    fs   = setup_multifield_environment["fields"]
    eid  = setup_multifield_environment["interface_data"]["eid"]
    p0,p1= setup_multifield_environment["interface_data"]["interface_pts"]
    phi  = setup_multifield_environment["level_set"]

    u_pos = Function("u_pos_func", "u_pos", dh); u_pos.set_values_from_function(lambda x,y: x + 2*y)
    u_neg = Function("u_neg_func", "u_neg", dh); u_neg.set_values_from_function(lambda x,y: 3*x - y)

    fc = FormCompiler(dh, backend="python")
    ctx, xq = _interface_ctx(dh, fs, eid, p0, p1, phi)
    fc.ctx.update(ctx)

    got = _eval_scalar(fc, Jump(u_pos, u_neg))
    exp = (xq[0] + 2*xq[1]) - (3*xq[0] - xq[1])
    assert np.isclose(got, exp, rtol=1e-3)

    assert np.isclose(_eval_scalar(fc, Jump(u_pos, u_pos)), 0.0, atol=1e-12)
    assert np.isclose(_eval_scalar(fc, Jump(u_neg, u_neg)), 0.0, atol=1e-12)
