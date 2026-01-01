import numpy as np
import pytest

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import AffineLevelSet, LevelSetDeformation
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import (
    Function,
    Inner,
    Jump,
    Neg,
    Pos,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
)
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dGhost

BACKENDS = ("python", "jit", "cpp")


def build_mesh(nx: int = 8, ny: int = 4, poly_order: int = 2):
    nodes, elements_connectivity, edge_connectivity, corner_nodes = structured_quad(
        Lx=2.0, Ly=1.0, nx=nx, ny=ny, poly_order=poly_order
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elements_connectivity,
        edges_connectivity=edge_connectivity,
        elements_corner_nodes=corner_nodes,
        element_type="quad",
        poly_order=poly_order,
    )
    level_set = AffineLevelSet(a=1.0, b=0.0, c=-1.0)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    return mesh, level_set


def build_deformation(mesh, scale: float = 0.075):
    coords = mesh.nodes_x_y_pos
    disp = np.zeros_like(coords)
    disp[:, 0] = scale * np.sin(np.pi * coords[:, 1])
    disp[:, 1] = 0.5 * scale * np.cos(np.pi * coords[:, 0])
    return LevelSetDeformation(mesh, disp)


def reference_jump_integral(
    dh: DofHandler,
    level_set,
    deformation,
    func_pos,
    func_neg,
    qdeg: int,
) -> float:
    mesh = dh.mixed_element.mesh
    ghost = mesh.edge_bitset("ghost")
    geo = dh.precompute_ghost_factors(
        ghost_edge_ids=ghost,
        qdeg=qdeg,
        level_set=level_set,
        derivs=set(),
        reuse=False,
        need_hess=False,
        need_o3=False,
        need_o4=False,
        deformation=deformation,
    )
    qp = geo["qp_phys"]
    qw = geo["qw"]
    xq = qp[..., 0]
    yq = qp[..., 1]
    jump_vals = func_pos(xq, yq) - func_neg(xq, yq)
    return float(np.sum(qw * jump_vals))


def assemble_jump_integral(dh, jump_expr, measure, backend: str) -> float:
    hooks = {type(jump_expr): {"name": "I"}}
    res = assemble_form(
        Equation(None, jump_expr * measure),
        dof_handler=dh,
        bcs=[],
        assembler_hooks=hooks,
        backend=backend,
    )
    return float(res["I"])


def assemble_system(eq: Equation, dh: DofHandler, backend: str, *, tol: float = 1e-14):
    K, F = assemble_form(eq, dof_handler=dh, bcs=[], backend=backend)
    ndofs = dh.total_dofs
    if K is None:
        mat = np.zeros((ndofs, ndofs))
    else:
        mat = K.toarray()
    if F is None:
        vec = np.zeros(ndofs)
    else:
        vec = np.asarray(F, dtype=float).copy()
    if tol > 0.0:
        mat[np.abs(mat) <= tol] = 0.0
        vec[np.abs(vec) <= tol] = 0.0
    return mat, vec


def assert_backends_agree(eq: Equation, dh: DofHandler, *, tol: float = 1e-10):
    K_ref, F_ref = assemble_system(eq, dh, "python")
    for backend in ("jit", "cpp"):
        K_b, F_b = assemble_system(eq, dh, backend)
        diff_K = float(np.max(np.abs(K_b - K_ref)))
        diff_F = float(np.max(np.abs(F_b - F_ref)))
        assert diff_K < tol, f"{backend} matrix mismatch: {diff_K}"
        assert diff_F < tol, f"{backend} rhs mismatch: {diff_F}"


@pytest.fixture(scope="module")
def ghost_setup():
    mesh, level_set = build_mesh()
    deformation = build_deformation(mesh)
    ghost = mesh.edge_bitset("ghost")
    assert ghost.cardinality() > 0
    me = MixedElement(mesh, field_specs={"u": mesh.poly_order})
    dh = DofHandler(me, method="cg")
    return dh, mesh, level_set, deformation, ghost


def test_jump_matrix_agrees_scalar(ghost_setup):
    dh, mesh, level_set, deformation, ghost = ghost_setup
    qdeg = 8
    measure = dGhost(
        defined_on=ghost,
        level_set=level_set,
        deformation=deformation,
        metadata={"q": qdeg},
    )

    u = TrialFunction(field_name="u", name="u_trial", dof_handler=dh)
    v = TestFunction(field_name="u", name="v_test", dof_handler=dh)
    a = Jump(u) * Jump(v) * measure

    f_coeff = Function(name="f_rhs", field_name="u", dof_handler=dh)
    f_coeff.set_values_from_function(lambda x, y: np.sin(np.pi * x) * np.cos(np.pi * y))
    L = Jump(f_coeff) * Jump(v) * measure

    assert_backends_agree(Equation(a, L), dh, tol=1e-10)


def test_jump_matrix_agrees_vector(ghost_setup):
    _dh, mesh, level_set, deformation, ghost = ghost_setup
    qdeg = 8
    measure = dGhost(
        defined_on=ghost,
        level_set=level_set,
        deformation=deformation,
        metadata={"q": qdeg},
    )

    me_vec = MixedElement(mesh, field_specs={"ux": mesh.poly_order, "uy": mesh.poly_order})
    dh_vec = DofHandler(me_vec, method="cg")
    space = FunctionSpace("vel", ["ux", "uy"])

    U = VectorTrialFunction(space, dof_handler=dh_vec)
    V = VectorTestFunction(space, dof_handler=dh_vec)
    a = Inner(Jump(U), Jump(V)) * measure

    G = VectorFunction(name="g_rhs", field_names=["ux", "uy"], dof_handler=dh_vec)
    G.components[0].set_values_from_function(lambda x, y: x + 0.25 * y)
    G.components[1].set_values_from_function(lambda x, y: np.sin(np.pi * x) + y**2)
    L = Inner(Jump(G), Jump(V)) * measure

    assert_backends_agree(Equation(a, L), dh_vec, tol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_jump_zero_matches_reference(ghost_setup, backend):
    dh, mesh, level_set, deformation, ghost = ghost_setup
    qdeg = 12
    measure = dGhost(
        defined_on=ghost,
        level_set=level_set,
        deformation=deformation,
        metadata={"q": qdeg},
    )

    def f_pos(xv, yv):
        return np.sin(2.0 * np.pi * xv) + 0.1 * yv**2 - 0.25

    analytic = Analytic(lambda xx, yy: f_pos(xx, yy), degree=6)
    jump_expr = Jump(analytic)

    expected = 0.0
    val = assemble_jump_integral(dh, jump_expr, measure, backend)
    assert abs(val - expected) < 1e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test_jump_nonzero_matches_reference(ghost_setup, backend):
    dh, mesh, level_set, deformation, ghost = ghost_setup
    qdeg = 24
    measure = dGhost(
        defined_on=ghost,
        level_set=level_set,
        deformation=deformation,
        metadata={"q": qdeg},
    )

    def f_pos(xv, yv):
        return np.sin(2.5 * np.pi * xv) + 0.35 * yv**2 + 0.2

    def f_neg(xv, yv):
        return 0.4 * xv**2 - 0.3 * np.cos(np.pi * yv)

    expr_pos = Pos(Analytic(lambda xx, yy: f_pos(xx, yy), degree=6))
    expr_neg = Neg(Analytic(lambda xx, yy: f_neg(xx, yy), degree=6))
    jump_expr = Jump(expr_pos, expr_neg)

    ref = reference_jump_integral(
        dh, level_set, deformation, f_pos, f_neg, qdeg=48
    )
    val = assemble_jump_integral(dh, jump_expr, measure, backend)
    assert abs(val - ref) < 1e-8
