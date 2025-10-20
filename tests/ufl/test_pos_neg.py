import numpy as np
from collections import defaultdict
import pytest
from scipy.sparse.linalg import norm as spnorm


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
    Pos, Neg, Jump, ElementWiseConstant, FacetNormal, Constant,
    dot, inner, grad
)
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.measures import *
from pycutfem.ufl.forms import Equation, assemble_form
import os



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
        "is_ghost": True,
        "is_interface": False,

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

@pytest.fixture(scope="module")
def mini_interface_setup():
    poly = 2
    L = H = 2.0
    nx = ny = 6
    nodes, elems, _, corners = structured_quad(L,H,nx=nx,ny=ny,poly_order=poly,offset=[-L/2,-H/2])
    mesh = Mesh(nodes, element_connectivity=elems, 
                elements_corner_nodes=corners, poly_order=poly, element_type='quad')

    # circle centered at origin, radius so both +/− present & several cut cells
    phi = CircleLevelSet(center=(0.0,0.0), radius=0.65)

    # classify + build Γ
    mesh.classify_elements(phi)
    mesh.classify_edges(phi)
    mesh.build_interface_segments(phi)

    # element sets
    inside  = mesh.element_bitset("inside")   # φ < 0
    outside = mesh.element_bitset("outside")  # φ > 0
    cut     = mesh.element_bitset("cut")

    # ghost sets (keep them available even if we don't use them first)
    g_pos  = mesh.edge_bitset("ghost_pos")
    g_neg  = mesh.edge_bitset("ghost_neg")
    g_both = mesh.edge_bitset("ghost_both")
    g_if   = mesh.edge_bitset("interface")
    ghost_pos = g_pos | g_both | g_if
    ghost_neg = g_neg | g_both | g_if

    me = MixedElement(mesh, field_specs={'u_outside': poly, 'u_inside': poly})
    dh = DofHandler(me, method='cg')

    dx_pos = dx(defined_on=outside | cut, level_set=phi, metadata={'side': '+', 'q': poly+2})
    dx_neg = dx(defined_on=inside  | cut, level_set=phi, metadata={'side': '-', 'q': poly+2})
    dGamma = dInterface(defined_on=cut, level_set=phi, metadata={'q': poly+2})
    dGpos  = dGhost(defined_on=ghost_pos, level_set=phi, metadata={'q': poly+2, 'derivs': {(1,0),(0,1)}})
    dGneg  = dGhost(defined_on=ghost_neg, level_set=phi, metadata={'q': poly+2, 'derivs': {(1,0),(0,1)}})

    # trial/test
    u_pos, v_pos = TrialFunction('u_outside', dh, side = '+'), TestFunction('u_outside', dh, side = '+')
    u_neg, v_neg = TrialFunction('u_inside',  dh, side = '-'), TestFunction('u_inside',  dh, side = '-')

    # normals & constants
    n = FacetNormal()
    one = Constant(1.0)
    alpha_pos = Constant(1.0)
    alpha_neg = Constant(20.0)

    # simple kappas = 1 (avoid Hansbo scaling in the parity test)
    kappa_p = Pos(ElementWiseConstant(np.ones(mesh.num_elements())))
    kappa_m = Neg(ElementWiseConstant(np.ones(mesh.num_elements())))

    return dict(
        mesh=mesh, dh=dh, phi=phi,
        dx_pos=dx_pos, dx_neg=dx_neg, dGamma=dGamma, dGpos=dGpos, dGneg=dGneg,
        u_pos=u_pos, v_pos=v_pos, u_neg=u_neg, v_neg=v_neg,
        n=n, one=one, alpha_pos=alpha_pos, alpha_neg=alpha_neg,
        kappa_p=kappa_p, kappa_m=kappa_m
    )

def _assemble_K(a, dh, backend):
    K, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend=backend)
    return K.tocsr()


def _compare_K(Kjit, Kpy, label, rtol=5e-12, atol=5e-13):
    D = (Kjit - Kpy)
    num = spnorm(D)
    den = max(1.0, spnorm(Kjit))
    rel = num / den
    assert rel < rtol or num < atol, f"{label}: JIT vs PY mismatch — rel={rel:.3e}, abs={num:.3e}"


@pytest.mark.parametrize("term", [
    # --- volume terms (sanity) ---
    "vol_pos", "vol_neg",

    # --- interface penalty (no cells/ghost in play) ---
    "iface_penalty",

    # --- interface fluxes, each component separately and combined ---
    "iface_flux_pos_only",
    "iface_flux_neg_only",
    "iface_flux_both",

    # (optional) ghost grad-jump stabilizations — uncomment once volume/interface pass
    "ghost_penalty_pos",
    "ghost_penalty_neg",
    "ghost_pos_gradjump",
    "ghost_neg_gradjump",
])
def test_jit_python_parity(mini_interface_setup, term):
    s = mini_interface_setup
    u_pos, v_pos = s["u_pos"], s["v_pos"]
    u_neg, v_neg = s["u_neg"], s["v_neg"]
    dx_pos, dx_neg, dGamma = s["dx_pos"], s["dx_neg"], s["dGamma"]
    n = s["n"]; alpha_pos, alpha_neg = s["alpha_pos"], s["alpha_neg"]
    kappa_p, kappa_m = s["kappa_p"], s["kappa_m"]

    jump_u = Jump(u_pos, u_neg)
    jump_v = Jump(v_pos, v_neg)
    # build each bilinear form in isolation
    if term == "vol_pos":
        a = inner(alpha_pos * grad(u_pos), grad(v_pos)) * dx_pos
    elif term == "vol_neg":
        a = inner(alpha_neg * grad(u_neg), grad(v_neg)) * dx_neg
    elif term == "iface_penalty":
        a = Constant(10.0) * jump_u * jump_v * dGamma   # pure penalty with simple constant
    elif term == "iface_flux_pos_only":
        jump_v = Jump(v_pos, v_neg)
        a = (- kappa_p * alpha_pos * dot(grad(u_pos), n)) * jump_v * dGamma
    elif term == "iface_flux_neg_only":
        jump_v = Jump(v_pos, v_neg)
        a = (- kappa_m * alpha_neg * dot(grad(u_neg), n)) * jump_v * dGamma
    elif term == "iface_flux_both":
        jump_u = Jump(u_pos, u_neg)
        jump_v = Jump(v_pos, v_neg)
        avg_flux_u = (- kappa_p * alpha_pos * dot(grad(u_pos), n)
                      - kappa_m * alpha_neg * dot(grad(u_neg), n))
        a = (avg_flux_u * jump_v + (- kappa_p * alpha_pos * dot(grad(v_pos), n)
             - kappa_m * alpha_neg * dot(grad(v_neg), n)) * jump_u) * dGamma
    elif term == "ghost_penalty_pos":
        a = Constant(0.5) * jump_u * jump_v * s["dGpos"]
    elif term == "ghost_penalty_neg":
        a = Constant(0.5) * jump_u * jump_v * s["dGneg"]
    elif term == "ghost_pos_gradjump":
        a = Constant(0.5) * inner(Jump(grad(u_pos), grad(u_neg)), Jump(grad(v_pos), grad(v_neg))) * s["dGpos"]
    elif term == "ghost_neg_gradjump":
        a = Constant(0.5) * inner(Jump(grad(u_pos), grad(u_neg)), Jump(grad(v_pos), grad(v_neg))) * s["dGneg"]
    else:
        raise ValueError(term)

    Kj = _assemble_K(a, s["dh"], backend="jit")
    Kp = _assemble_K(a, s["dh"], backend="python")
    _compare_K(Kj, Kp, term)


# ------------------------------ Ghost-edge tests ------------------------------

@pytest.mark.parametrize("backend", ["python", "jit"])
def test_ghost_pos_and_neg_on_own_fields(setup_multifield_environment, backend):
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

    fc = FormCompiler(dh, backend=backend)
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


@pytest.mark.parametrize("backend", ["python", "jit"])
def test_ghost_jump_mixed_and_same_field_sanity(setup_multifield_environment, backend):
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

    fc = FormCompiler(dh, backend=backend)
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
