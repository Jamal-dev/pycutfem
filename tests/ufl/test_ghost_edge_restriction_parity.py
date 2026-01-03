import numpy as np
import pytest

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.expressions import (
    Function,
    TrialFunction,
    TestFunction,
    Jump,
    Pos,
    Neg,
    restrict,
    inner,
    grad,
)
from pycutfem.ufl.measures import dGhost
from pycutfem.ufl.forms import Equation, assemble_form


BACKENDS = ("python", "jit", "cpp")


def _build_mesh():
    nodes, elements, edges, corners = structured_quad(
        Lx=1.0,
        Ly=1.0,
        nx=6,
        ny=4,
        poly_order=1,
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elements,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    level_set = AffineLevelSet(a=1.0, b=0.0, c=-0.45)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    return mesh, level_set


def _assemble_jump(dh, expr, measure, backend: str) -> float:
    hooks = {type(expr): {"name": "I"}}
    res = assemble_form(
        Equation(None, expr * measure),
        dof_handler=dh,
        bcs=[],
        assembler_hooks=hooks,
        backend=backend,
    )
    return float(res["I"])


def _assemble_matrix(dh, form, backend: str):
    K, _ = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], backend=backend)
    return K.tocsr() if hasattr(K, "tocsr") else np.asarray(K)


def _assemble_vector(dh, form, backend: str) -> np.ndarray:
    _, F = assemble_form(Equation(None, form), dof_handler=dh, bcs=[], backend=backend)
    return np.asarray(F, dtype=float).ravel()


def _expected_jump_constant(
    dh,
    ghost_edges,
    level_set,
    qdeg,
    pos_dom,
    neg_dom,
    pos_val,
    neg_val,
    *,
    allow_equal: bool = False,
):
    geo = dh.precompute_ghost_factors(
        ghost_edge_ids=ghost_edges,
        qdeg=qdeg,
        level_set=level_set,
        derivs=set(),
        reuse=False,
    )
    qw = np.asarray(geo["qw"], dtype=float)
    pos_ids = np.asarray(geo["owner_pos_id"], dtype=int)
    neg_ids = np.asarray(geo["owner_neg_id"], dtype=int)

    pos_flag = np.zeros_like(pos_ids, dtype=float)
    neg_flag = np.zeros_like(neg_ids, dtype=float)
    if pos_ids.size:
        valid = pos_ids >= 0
        pos_flag[valid] = pos_dom.mask[pos_ids[valid]]
    if neg_ids.size:
        valid = neg_ids >= 0
        neg_flag[valid] = neg_dom.mask[neg_ids[valid]]

    if not allow_equal and not np.any(pos_flag != neg_flag):
        raise AssertionError(
            "Expected differing pos/neg restriction flags on ghost edges; adjust mesh/level set."
        )

    return float(np.sum(qw * (pos_val * pos_flag[:, None] - neg_val * neg_flag[:, None])))


@pytest.fixture(scope="module")
def ghost_setup():
    mesh, level_set = _build_mesh()
    ghost = mesh.edge_bitset("ghost")
    assert ghost.cardinality() > 0, "Expected ghost edges for restriction test."

    inside = mesh.element_bitset("inside")
    outside = mesh.element_bitset("outside")

    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")
    return dh, level_set, ghost, inside, outside


@pytest.mark.parametrize("backend", BACKENDS)
def test_ghost_jump_inside_outside_expected(ghost_setup, backend):
    dh, level_set, ghost, inside, outside = ghost_setup

    qdeg = 4
    val_outside = 2.0
    val_inside = 1.0

    u_out = Function(name="u_out", field_name="u", dof_handler=dh)
    u_in = Function(name="u_in", field_name="u", dof_handler=dh)
    u_out.nodal_values[:] = val_outside
    u_in.nodal_values[:] = val_inside

    expected = _expected_jump_constant(
        dh,
        ghost,
        level_set,
        qdeg,
        outside,
        inside,
        val_outside,
        val_inside,
    )
    assert abs(expected) > 1e-12, "Expected non-zero jump integral for inside/outside constants."

    jump_expr = Jump(
        restrict(Pos(u_out), outside),
        restrict(Neg(u_in), inside),
    )
    measure = dGhost(
        defined_on=ghost,
        level_set=level_set,
        metadata={"q": qdeg},
    )
    val = _assemble_jump(dh, jump_expr, measure, backend)
    assert np.isfinite(val), f"{backend} returned non-finite jump integral."
    assert abs(val - expected) < 1e-9, f"{backend} mismatch on ghost restriction: {val} vs {expected}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_ghost_jump_zero_when_equal(ghost_setup, backend):
    dh, level_set, ghost, inside, outside = ghost_setup
    mesh = dh.mixed_element.mesh
    full = inside | outside | mesh.element_bitset("cut")

    qdeg = 4
    val_outside = 3.0
    val_inside = 3.0

    u_out = Function(name="u_out", field_name="u", dof_handler=dh)
    u_in = Function(name="u_in", field_name="u", dof_handler=dh)
    u_out.nodal_values[:] = val_outside
    u_in.nodal_values[:] = val_inside

    expected = _expected_jump_constant(
        dh,
        ghost,
        level_set,
        qdeg,
        full,
        full,
        val_outside,
        val_inside,
        allow_equal=True,
    )
    assert abs(expected) < 1e-12, "Expected zero jump integral when inside/outside values match."

    jump_expr = Jump(
        restrict(Pos(u_out), full),
        restrict(Neg(u_in), full),
    )
    measure = dGhost(
        defined_on=ghost,
        level_set=level_set,
        metadata={"q": qdeg},
    )
    val = _assemble_jump(dh, jump_expr, measure, backend)
    assert np.isfinite(val), f"{backend} returned non-finite jump integral."
    assert abs(val) < 1e-9, f"{backend} expected ~0 jump integral, got {val}"


def test_ghost_grad_restriction_parity_matrix_and_rhs(ghost_setup):
    dh, level_set, ghost, inside, outside = ghost_setup
    mesh = dh.mixed_element.mesh

    ghost_neg = mesh.edge_bitset("ghost_neg") | mesh.edge_bitset("ghost_both")
    if ghost_neg.cardinality() == 0:
        pytest.skip("No negative-side ghost edges found for restriction parity test.")

    cut = mesh.element_bitset("cut")
    has_neg = inside | cut

    u = TrialFunction(name="u", field_name="u", dof_handler=dh)
    v = TestFunction(name="v", field_name="u", dof_handler=dh)
    u_k = Function(name="u_k", field_name="u", dof_handler=dh)
    u_k.set_values_from_function(lambda x, y: x + 0.25 * y)

    u_R = restrict(Neg(u), has_neg)
    v_R = restrict(Neg(v), has_neg)
    u_k_R = restrict(Neg(u_k), has_neg)

    qdeg = 4
    dG = dGhost(
        defined_on=ghost_neg,
        level_set=level_set,
        metadata={"q": qdeg, "derivs": {(1, 0), (0, 1)}},
    )

    a = inner(grad(u_R), grad(v_R)) * dG
    L = inner(grad(u_k_R), grad(v_R)) * dG

    K_py = _assemble_matrix(dh, a, "python")
    F_py = _assemble_vector(dh, L, "python")

    k_abs = np.max(np.abs(K_py.toarray())) if hasattr(K_py, "toarray") else np.max(np.abs(K_py))
    assert k_abs > 1e-12, "Reference matrix unexpectedly zero."
    assert np.max(np.abs(F_py)) > 1e-12, "Reference RHS unexpectedly zero."

    for backend in ("jit", "cpp"):
        K_b = _assemble_matrix(dh, a, backend)
        F_b = _assemble_vector(dh, L, backend)

        K_diff = K_b - K_py
        if hasattr(K_diff, "data"):
            k_err = float(np.max(np.abs(K_diff.data))) if K_diff.data.size else 0.0
        else:
            k_err = float(np.max(np.abs(K_diff)))

        f_err = float(np.max(np.abs(F_b - F_py)))

        assert k_err < 1e-9, f"{backend} mismatch in restricted ghost grad matrix: {k_err}"
        assert f_err < 1e-9, f"{backend} mismatch in restricted ghost grad RHS: {f_err}"
