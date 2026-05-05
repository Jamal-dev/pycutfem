import os
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.nonmatching import (
    apply_dirichlet_data,
    assemble_nonmatching_interface_form,
    build_composite_mesh,
    build_nonmatching_interface,
    coupled_dirichlet_data,
    lift_nonmatching_interface_to_composite,
)
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    FacetNormal,
    Function,
    Identity,
    Neg,
    Pos,
    TestFunction,
    TrialFunction,
    VectorTestFunction,
    VectorTrialFunction,
    div,
    dot,
    grad,
    inner,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.measures import dNonmatchingInterface, dx
from pycutfem.utils.bitset import BitSet
from pycutfem.utils.meshgen import structured_quad, structured_triangles


def _have_numba() -> bool:
    try:
        import numba  # noqa: F401
    except Exception:
        return False
    return True


def _have_pybind11() -> bool:
    try:
        import pybind11  # noqa: F401
    except Exception:
        return False
    return True


def _get_convergence_backend() -> str:
    backend = os.getenv("CONVERGENCE_TABLE_BACKEND", "python").strip().lower() or "python"
    allowed = {"python", "jit", "cpp"}
    if backend not in allowed:
        raise ValueError(f"Invalid CONVERGENCE_TABLE_BACKEND={backend!r}; expected one of {sorted(allowed)}")
    return backend


def _skip_if_backend_unavailable(backend: str) -> None:
    if backend == "jit" and not _have_numba():
        pytest.skip("numba not available (CONVERGENCE_TABLE_BACKEND=jit)")
    if backend == "cpp" and not _have_pybind11():
        pytest.skip("pybind11 not available (CONVERGENCE_TABLE_BACKEND=cpp)")


def _print_table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    col_widths = [len(c) for c in columns]
    for row in rows:
        for j, val in enumerate(row):
            col_widths[j] = max(col_widths[j], len(val))

    print("")
    print(title)
    print("  ".join(columns[j].ljust(col_widths[j]) for j in range(len(columns))))
    print("  ".join("-" * col_widths[j] for j in range(len(columns))))
    for row in rows:
        print("  ".join(row[j].ljust(col_widths[j]) for j in range(len(columns))))


def _make_submesh(*, element: str, poly_order: int, nx: int, ny: int, offset_x: float) -> Mesh:
    if element == "quad":
        nodes, elems, edges, corners = structured_quad(
            0.5, 1.0, nx=nx, ny=ny, poly_order=poly_order, offset=(offset_x, 0.0)
        )
        return Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=poly_order)
    if element == "tri":
        nodes, elems, edges, corners = structured_triangles(
            0.5, 1.0, nx_quads=nx, ny_quads=ny, poly_order=poly_order, offset=(offset_x, 0.0)
        )
        return Mesh(nodes, elems, edges, corners, element_type="tri", poly_order=poly_order)
    raise ValueError("element must be 'quad' or 'tri'")


def _component_bitsets(mesh: Mesh, *, pos_eids: np.ndarray, neg_eids: np.ndarray) -> tuple[BitSet, BitSet]:
    nE = int(getattr(mesh, "n_elements", len(mesh.elements_list)))
    m_pos = np.zeros(nE, dtype=bool)
    m_neg = np.zeros(nE, dtype=bool)
    m_pos[np.asarray(pos_eids, dtype=int)] = True
    m_neg[np.asarray(neg_eids, dtype=int)] = True
    return BitSet(m_pos), BitSet(m_neg)


def _solve_poisson_nitsche_ufl(
    *,
    element: str,
    degree: int,
    ny_neg: int,
    ny_pos: int,
    k_neg: float,
    k_pos: float,
    gamma: float,
    u_neg,
    grad_u_neg,
    u_pos,
    grad_u_pos,
    f,
    backend: str = "python",
):
    nx_neg = max(2, int(round(0.5 * ny_neg)))
    nx_pos = max(2, int(round(0.5 * ny_pos)))
    mesh_neg = _make_submesh(element=element, poly_order=degree, nx=nx_neg, ny=ny_neg, offset_x=0.0)
    mesh_pos = _make_submesh(element=element, poly_order=degree, nx=nx_pos, ny=ny_pos, offset_x=0.5)

    mesh_neg.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})
    mesh_pos.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})

    interface = build_nonmatching_interface(mesh_neg=mesh_neg, mesh_pos=mesh_pos)
    mapping = build_composite_mesh(mesh_pos=mesh_pos, mesh_neg=mesh_neg, order="pos_neg")
    mesh = mapping.mesh
    interface_c = lift_nonmatching_interface_to_composite(interface=interface, mapping=mapping)

    mesh.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})
    for eid in mapping.pos_elem_ids:
        mesh.elements_list[int(eid)].tag = "pos"
    for eid in mapping.neg_elem_ids:
        mesh.elements_list[int(eid)].tag = "neg"

    dh = DofHandler(MixedElement(mesh, {"u": int(degree)}), method="cg")

    q = int(2 * degree + 2)
    bs_pos, bs_neg = _component_bitsets(mesh, pos_eids=mapping.pos_elem_ids, neg_eids=mapping.neg_elem_ids)
    dx_pos = dx(defined_on=bs_pos, metadata={"q": q})
    dx_neg = dx(defined_on=bs_neg, metadata={"q": q})
    dGamma = dNonmatchingInterface(metadata={"q": q, "interface": interface_c})

    u = TrialFunction("u", dof_handler=dh)
    v = TestFunction("u", dof_handler=dh)
    n = FacetNormal()
    h = CellDiameter()

    denom = float(k_pos) + float(k_neg)
    if denom <= 0.0:
        raise ValueError("k_pos + k_neg must be > 0.")
    kappa_pos = Constant(float(k_neg) / denom)
    kappa_neg = Constant(float(k_pos) / denom)

    f_expr = Analytic(f, degree=max(4, q + 2))

    a = Constant(float(k_pos)) * inner(grad(u), grad(v)) * dx_pos
    a += Constant(float(k_neg)) * inner(grad(u), grad(v)) * dx_neg

    flux_u_pos = -Constant(float(k_pos)) * dot(grad(Pos(u)), n)
    flux_u_neg = -Constant(float(k_neg)) * dot(grad(Neg(u)), n)
    flux_v_pos = -Constant(float(k_pos)) * dot(grad(Pos(v)), n)
    flux_v_neg = -Constant(float(k_neg)) * dot(grad(Neg(v)), n)

    avg_flux_u = kappa_pos * flux_u_pos + kappa_neg * flux_u_neg
    avg_flux_v = kappa_pos * flux_v_pos + kappa_neg * flux_v_neg
    jump_u = Neg(u) - Pos(u)
    jump_v = Neg(v) - Pos(v)

    stab = Constant(float(gamma) * (float(k_pos) + float(k_neg))) / h
    a += (avg_flux_u * jump_v + avg_flux_v * jump_u + stab * jump_u * jump_v) * dGamma

    L = f_expr * v * dx_pos + f_expr * v * dx_neg

    def bc_value(xx, yy):
        xx = float(xx)
        yy = float(yy)
        return float(u_pos(xx, yy) if xx > 0.5 else u_neg(xx, yy))

    bc = BoundaryCondition("u", "dirichlet", "boundary", bc_value)
    K, F = assemble_form(Equation(a, L), dof_handler=dh, bcs=[bc], backend=backend)
    sol = spla.spsolve(K.tocsc(), F)
    return np.asarray(sol, float), dh, interface_c, mapping


def _solve_poisson_mortar_ufl(
    *,
    element: str,
    degree: int,
    ny_neg: int,
    ny_pos: int,
    k_neg: float,
    k_pos: float,
    u_neg,
    grad_u_neg,
    u_pos,
    grad_u_pos,
    f,
    backend: str = "python",
):
    nx_neg = max(2, int(round(0.5 * ny_neg)))
    nx_pos = max(2, int(round(0.5 * ny_pos)))
    mesh_neg = _make_submesh(element=element, poly_order=degree, nx=nx_neg, ny=ny_neg, offset_x=0.0)
    mesh_pos = _make_submesh(element=element, poly_order=degree, nx=nx_pos, ny=ny_pos, offset_x=0.5)

    mesh_neg.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})
    mesh_pos.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})

    interface = build_nonmatching_interface(mesh_neg=mesh_neg, mesh_pos=mesh_pos)
    mapping = build_composite_mesh(mesh_pos=mesh_pos, mesh_neg=mesh_neg, order="pos_neg")
    mesh = mapping.mesh
    interface_c = lift_nonmatching_interface_to_composite(interface=interface, mapping=mapping)

    mesh.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})
    for eid in mapping.pos_elem_ids:
        mesh.elements_list[int(eid)].tag = "pos"
    for eid in mapping.neg_elem_ids:
        mesh.elements_list[int(eid)].tag = "neg"

    # Mixed u + lambda system (lambda lives on the master-side interface nodes).
    dh = DofHandler(MixedElement(mesh, {"u": int(degree), "lam": 1}), method="cg")

    q = int(2 * degree + 2)
    bs_pos, bs_neg = _component_bitsets(mesh, pos_eids=mapping.pos_elem_ids, neg_eids=mapping.neg_elem_ids)
    dx_pos = dx(defined_on=bs_pos, metadata={"q": q})
    dx_neg = dx(defined_on=bs_neg, metadata={"q": q})
    dGamma = dNonmatchingInterface(metadata={"q": q, "interface": interface_c})

    u = TrialFunction("u", dof_handler=dh)
    v = TestFunction("u", dof_handler=dh)
    lam = TrialFunction("lam", dof_handler=dh)
    mu = TestFunction("lam", dof_handler=dh)

    f_expr = Analytic(f, degree=max(4, q + 2))

    # Master side selection by number of interface nodes (node-based P1 multipliers).
    def _edge_nodes(edge_ids: np.ndarray) -> np.ndarray:
        nodes: set[int] = set()
        for gid in np.unique(np.asarray(edge_ids, dtype=int)):
            e = mesh.edge(int(gid))
            nodes.update(int(n) for n in e.nodes)
        node_ids = np.fromiter(nodes, dtype=int)
        d0 = np.asarray(interface_c.P1[0] - interface_c.P0[0], dtype=float)
        t = d0 / max(float(np.linalg.norm(d0)), 1.0e-16)
        s = mesh.nodes_x_y_pos[node_ids] @ t
        return node_ids[np.argsort(s)]

    pos_nodes = _edge_nodes(interface_c.pos_edge_ids)
    neg_nodes = _edge_nodes(interface_c.neg_edge_ids)
    master = "neg" if neg_nodes.size <= pos_nodes.size else "pos"
    master_nodes = neg_nodes if master == "neg" else pos_nodes

    # Constrain all lambda DOFs except those on master interface nodes.
    dh.dof_tags = getattr(dh, "dof_tags", {}) or {}
    all_lam = set(int(i) for i in np.asarray(dh.get_field_slice("lam"), dtype=int).ravel())
    free_lam = set()
    for nid in np.asarray(master_nodes, dtype=int).ravel():
        gd = dh.dof_map.get("lam", {}).get(int(nid))
        if gd is not None:
            free_lam.add(int(gd))
    dh.dof_tags["lam_fixed"] = all_lam - free_lam

    bc_lam = BoundaryCondition("lam", "dirichlet", "lam_fixed", 0.0)

    def bc_u_value(xx, yy):
        xx = float(xx)
        yy = float(yy)
        return float(u_pos(xx, yy) if xx > 0.5 else u_neg(xx, yy))

    bc_u = BoundaryCondition("u", "dirichlet", "boundary", bc_u_value)

    if master == "neg":
        lam_here = Neg(lam)
        mu_here = Neg(mu)
    else:
        lam_here = Pos(lam)
        mu_here = Pos(mu)

    jump_u = Neg(u) - Pos(u)
    jump_v = Neg(v) - Pos(v)

    a = Constant(float(k_pos)) * inner(grad(u), grad(v)) * dx_pos
    a += Constant(float(k_neg)) * inner(grad(u), grad(v)) * dx_neg
    a += (lam_here * jump_v + mu_here * jump_u) * dGamma

    L = f_expr * v * dx_pos + f_expr * v * dx_neg

    K, F = assemble_form(Equation(a, L), dof_handler=dh, bcs=[bc_u, bc_lam], backend=backend)
    sol = spla.spsolve(K.tocsc(), F)
    return np.asarray(sol, float), dh, interface_c, mapping


def _solve_poisson_nitsche_multimesh_ufl(
    *,
    degree: int,
    ny_neg: int,
    ny_pos: int,
    k_neg: float,
    k_pos: float,
    gamma: float,
    u_neg,
    grad_u_neg,
    u_pos,
    grad_u_pos,
    f,
    backend: str = "python",
):
    """
    Poisson Nitsche coupling on a *nonmatching* interface with *different*
    element types across the interface (neg=tri, pos=quad).
    """
    nx_neg = max(2, int(round(0.5 * ny_neg)))
    nx_pos = max(2, int(round(0.5 * ny_pos)))
    mesh_neg = _make_submesh(element="tri", poly_order=degree, nx=nx_neg, ny=ny_neg, offset_x=0.0)
    mesh_pos = _make_submesh(element="quad", poly_order=degree, nx=nx_pos, ny=ny_pos, offset_x=0.5)

    mesh_neg.tag_boundary_edges(
        {
            "interface": lambda xx, yy: abs(xx - 0.5) < 1e-12,
            "boundary": lambda xx, yy: abs(xx - 0.5) >= 1e-12,
        }
    )
    mesh_pos.tag_boundary_edges(
        {
            "interface": lambda xx, yy: abs(xx - 0.5) < 1e-12,
            "boundary": lambda xx, yy: abs(xx - 0.5) >= 1e-12,
        }
    )

    interface = build_nonmatching_interface(mesh_neg=mesh_neg, mesh_pos=mesh_pos)

    dh_pos = DofHandler(MixedElement(mesh_pos, {"u": int(degree)}), method="cg")
    dh_neg = DofHandler(MixedElement(mesh_neg, {"u": int(degree)}), method="cg")

    q = int(2 * degree + 2)
    dx_pos = dx(metadata={"q": q})
    dx_neg = dx(metadata={"q": q})

    # Volume terms (assembled per mesh)
    up = TrialFunction("u", dof_handler=dh_pos)
    vp = TestFunction("u", dof_handler=dh_pos)
    un = TrialFunction("u", dof_handler=dh_neg)
    vn = TestFunction("u", dof_handler=dh_neg)

    f_expr_pos = Analytic(f, degree=max(4, q + 2))
    f_expr_neg = Analytic(f, degree=max(4, q + 2))

    a_pos = Constant(float(k_pos)) * inner(grad(up), grad(vp)) * dx_pos
    a_neg = Constant(float(k_neg)) * inner(grad(un), grad(vn)) * dx_neg
    L_pos = f_expr_pos * vp * dx_pos
    L_neg = f_expr_neg * vn * dx_neg

    # Dirichlet BC on the *outer* boundary (not on the interface)
    bc_pos = BoundaryCondition("u", "dirichlet", "boundary", lambda xx, yy: float(u_pos(float(xx), float(yy))))
    bc_neg = BoundaryCondition("u", "dirichlet", "boundary", lambda xx, yy: float(u_neg(float(xx), float(yy))))

    Kp, Fp = assemble_form(Equation(a_pos, L_pos), dof_handler=dh_pos, bcs=[], backend=backend)
    Kn, Fn = assemble_form(Equation(a_neg, L_neg), dof_handler=dh_neg, bcs=[], backend=backend)

    n_pos = int(dh_pos.total_dofs)
    n_neg = int(dh_neg.total_dofs)
    n_total = n_pos + n_neg

    K = sp.lil_matrix((n_total, n_total), dtype=float)
    K[:n_pos, :n_pos] = Kp
    K[n_pos:, n_pos:] = Kn
    F = np.zeros(n_total, dtype=float)
    F[:n_pos] = np.asarray(Fp, dtype=float)
    F[n_pos:] = np.asarray(Fn, dtype=float)

    # Interface terms (UFL, multimesh)
    u = TrialFunction("u", dof_handler=dh_pos)
    v = TestFunction("u", dof_handler=dh_pos)
    n = FacetNormal()
    h = CellDiameter()

    denom = float(k_pos) + float(k_neg)
    kappa_pos = Constant(float(k_neg) / denom)
    kappa_neg = Constant(float(k_pos) / denom)

    flux_u_pos = -Constant(float(k_pos)) * dot(grad(Pos(u)), n)
    flux_u_neg = -Constant(float(k_neg)) * dot(grad(Neg(u)), n)
    flux_v_pos = -Constant(float(k_pos)) * dot(grad(Pos(v)), n)
    flux_v_neg = -Constant(float(k_neg)) * dot(grad(Neg(v)), n)

    avg_flux_u = kappa_pos * flux_u_pos + kappa_neg * flux_u_neg
    avg_flux_v = kappa_pos * flux_v_pos + kappa_neg * flux_v_neg
    jump_u = Neg(u) - Pos(u)
    jump_v = Neg(v) - Pos(v)
    stab = Constant(float(gamma) * (float(k_pos) + float(k_neg))) / h
    a_if = avg_flux_u * jump_v + avg_flux_v * jump_u + stab * jump_u * jump_v

    K_if, _ = assemble_nonmatching_interface_form(
        a_if,
        interface=interface,
        dh_pos=dh_pos,
        dh_neg=dh_neg,
        quad_order=q,
        backend=backend,
        ordering="pos_neg",
    )
    K += K_if.tolil()

    bc_data = coupled_dirichlet_data(dh_pos=dh_pos, bcs_pos=[bc_pos], dh_neg=dh_neg, bcs_neg=[bc_neg], neg_offset=n_pos)
    K, F = apply_dirichlet_data(K.tocsr(), F, bc_data)
    sol = spla.spsolve(K.tocsc(), F)

    return np.asarray(sol, float), dh_pos, dh_neg, interface


@pytest.mark.parametrize("element", ["quad", "tri"])
def test_nonmatching_poisson_patch_constant_nitsche(element):
    u_exact = lambda x, y: 1.0
    grad_u = lambda x, y: (0.0, 0.0)
    f = lambda xv, yv: 0.0 * xv

    sol, dh, interface, _ = _solve_poisson_nitsche_ufl(
        element=element,
        degree=1,
        ny_neg=6,
        ny_pos=7,
        k_neg=1.0,
        k_pos=1.0,
        gamma=20.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend="python",
    )

    u_slice = np.asarray(dh.get_field_slice("u"), dtype=int)
    assert float(np.max(np.abs(sol[u_slice] - 1.0))) < 1.0e-11

    uh = Function(name="uh", field_name="u", dof_handler=dh)
    uh.set_nodal_values(u_slice, sol[u_slice])
    jump2 = (Neg(uh) - Pos(uh)) * (Neg(uh) - Pos(uh))
    res = assemble_form(
        Equation(jump2 * dNonmatchingInterface(metadata={"q": 6, "interface": interface}), None),
        dof_handler=dh,
        assembler_hooks={jump2: {"name": "jump2"}},
        backend="python",
    )
    assert float(np.sqrt(np.asarray(res["jump2"]).ravel()[0])) < 1.0e-10


@pytest.mark.parametrize("element", ["quad", "tri"])
def test_nonmatching_poisson_patch_linear_nitsche(element):
    u_exact = lambda x, y: float(x + 2.0 * y - 0.3)
    grad_u = lambda x, y: (1.0, 2.0)
    f = lambda xv, yv: 0.0 * xv

    sol, dh, interface, mapping = _solve_poisson_nitsche_ufl(
        element=element,
        degree=1,
        ny_neg=6,
        ny_pos=9,
        k_neg=1.0,
        k_pos=1.0,
        gamma=20.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend="python",
    )

    uh = Function(name="uh", field_name="u", dof_handler=dh)
    u_slice = np.asarray(dh.get_field_slice("u"), dtype=int)
    uh.set_nodal_values(u_slice, sol[u_slice])

    q = 8
    bs_pos, bs_neg = _component_bitsets(mapping.mesh, pos_eids=mapping.pos_elem_ids, neg_eids=mapping.neg_elem_ids)
    dx_pos = dx(defined_on=bs_pos, metadata={"q": q})
    dx_neg = dx(defined_on=bs_neg, metadata={"q": q})
    u_ex = Analytic(lambda xv, yv: u_exact(xv, yv), degree=q + 2)
    err2 = (uh - u_ex) * (uh - u_ex)
    res = assemble_form(
        Equation(err2 * dx_pos + err2 * dx_neg, None),
        dof_handler=dh,
        assembler_hooks={err2: {"name": "err2"}},
        backend="python",
    )
    assert float(np.sqrt(np.asarray(res["err2"]).ravel()[0])) < 1.0e-11

    jump2 = (Neg(uh) - Pos(uh)) * (Neg(uh) - Pos(uh))
    res = assemble_form(
        Equation(jump2 * dNonmatchingInterface(metadata={"q": q, "interface": interface}), None),
        dof_handler=dh,
        assembler_hooks={jump2: {"name": "jump2"}},
        backend="python",
    )
    assert float(np.sqrt(np.asarray(res["jump2"]).ravel()[0])) < 1.0e-10


@pytest.mark.parametrize("element", ["quad", "tri"])
def test_nonmatching_poisson_mms_kjump_convergence_nitsche(element):
    backend = _get_convergence_backend()
    _skip_if_backend_unavailable(backend)

    k_neg = 1.0
    k_pos = 10.0

    u_neg = lambda x, y: (x - 0.5) * np.sin(np.pi * y)
    u_pos = lambda x, y: 0.1 * (x - 0.5) * np.sin(np.pi * y)
    grad_u_neg = lambda x, y: (np.sin(np.pi * y), (x - 0.5) * np.pi * np.cos(np.pi * y))
    grad_u_pos = lambda x, y: (0.1 * np.sin(np.pi * y), 0.1 * (x - 0.5) * np.pi * np.cos(np.pi * y))
    f = lambda xv, yv: (np.pi**2) * (xv - 0.5) * np.sin(np.pi * yv)

    errs = []
    hs = []
    flux_mis_norms = []
    for ny in (6, 12):
        sol, dh, interface, mapping = _solve_poisson_nitsche_ufl(
            element=element,
            degree=1,
            ny_neg=ny,
            ny_pos=ny + 1,
            k_neg=k_neg,
            k_pos=k_pos,
            gamma=20.0,
            u_neg=u_neg,
            grad_u_neg=grad_u_neg,
            u_pos=u_pos,
            grad_u_pos=grad_u_pos,
            f=f,
            backend=backend,
        )

        uh = Function(name="uh", field_name="u", dof_handler=dh)
        u_slice = np.asarray(dh.get_field_slice("u"), dtype=int)
        uh.set_nodal_values(u_slice, sol[u_slice])

        q = 10
        bs_pos, bs_neg = _component_bitsets(mapping.mesh, pos_eids=mapping.pos_elem_ids, neg_eids=mapping.neg_elem_ids)
        dx_pos = dx(defined_on=bs_pos, metadata={"q": q})
        dx_neg = dx(defined_on=bs_neg, metadata={"q": q})

        u_ex_pos = Analytic(lambda xv, yv: u_pos(xv, yv), degree=q + 2)
        u_ex_neg = Analytic(lambda xv, yv: u_neg(xv, yv), degree=q + 2)
        g_ex_pos = Analytic(lambda xv, yv: np.stack(grad_u_pos(xv, yv), axis=-1), degree=q + 2)
        g_ex_neg = Analytic(lambda xv, yv: np.stack(grad_u_neg(xv, yv), axis=-1), degree=q + 2)

        err2_pos = (uh - u_ex_pos) * (uh - u_ex_pos)
        err2_neg = (uh - u_ex_neg) * (uh - u_ex_neg)
        res = assemble_form(
            Equation(err2_pos * dx_pos + err2_neg * dx_neg, None),
            dof_handler=dh,
            assembler_hooks={err2_pos: {"name": "err2"}, err2_neg: {"name": "err2"}},
            backend=backend,
        )
        eL2 = float(np.sqrt(np.asarray(res["err2"]).ravel()[0]))

        h12_pos = inner(grad(uh) - g_ex_pos, grad(uh) - g_ex_pos)
        h12_neg = inner(grad(uh) - g_ex_neg, grad(uh) - g_ex_neg)
        res = assemble_form(
            Equation(h12_pos * dx_pos + h12_neg * dx_neg, None),
            dof_handler=dh,
            assembler_hooks={h12_pos: {"name": "h12"}, h12_neg: {"name": "h12"}},
            backend=backend,
        )
        eH1 = float(np.sqrt(np.asarray(res["h12"]).ravel()[0]))
        errs.append((eL2, eH1))
        hs.append(float(np.sqrt(np.asarray(mapping.mesh.areas_list, float)).max()))

        # sanity: flux mismatch is bounded
        n = FacetNormal()
        flux_mis = Constant(float(k_neg)) * dot(grad(Neg(uh)), n) - Constant(float(k_pos)) * dot(grad(Pos(uh)), n)
        mis2 = flux_mis * flux_mis
        res = assemble_form(
            Equation(mis2 * dNonmatchingInterface(metadata={"q": q, "interface": interface}), None),
            dof_handler=dh,
            assembler_hooks={mis2: {"name": "mis2"}},
            backend=backend,
        )
        flux_mis_norm = float(np.sqrt(np.asarray(res["mis2"]).ravel()[0]))
        flux_mis_norms.append(flux_mis_norm)
        assert flux_mis_norm < 10.0

    (eL2_0, eH1_0), (eL2_1, eH1_1) = errs
    h0, h1 = hs
    rate_L2 = np.log(eL2_0 / eL2_1) / np.log(h0 / h1)
    rate_H1 = np.log(eH1_0 / eH1_1) / np.log(h0 / h1)
    _print_table(
        f"[nonmatching] Poisson MMS (Nitsche) element={element} backend={backend}",
        ["ny", "h", "L2", "H1", "rate_L2", "rate_H1", "flux_mis"],
        [
            [f"{6:d}", f"{hs[0]:.3e}", f"{errs[0][0]:.3e}", f"{errs[0][1]:.3e}", "", "", f"{flux_mis_norms[0]:.3e}"],
            [
                f"{12:d}",
                f"{hs[1]:.3e}",
                f"{errs[1][0]:.3e}",
                f"{errs[1][1]:.3e}",
                f"{rate_L2:.2f}",
                f"{rate_H1:.2f}",
                f"{flux_mis_norms[1]:.3e}",
            ],
        ],
    )
    assert rate_L2 > 1.8
    assert rate_H1 > 0.9


def test_nonmatching_poisson_mms_kjump_convergence_nitsche_mixed_element_types():
    backend = _get_convergence_backend()
    _skip_if_backend_unavailable(backend)

    k_neg = 1.0
    k_pos = 10.0

    u_neg = lambda x, y: (x - 0.5) * np.sin(np.pi * y)
    u_pos = lambda x, y: 0.1 * (x - 0.5) * np.sin(np.pi * y)
    grad_u_neg = lambda x, y: (np.sin(np.pi * y), (x - 0.5) * np.pi * np.cos(np.pi * y))
    grad_u_pos = lambda x, y: (0.1 * np.sin(np.pi * y), 0.1 * (x - 0.5) * np.pi * np.cos(np.pi * y))
    f = lambda xv, yv: (np.pi**2) * (xv - 0.5) * np.sin(np.pi * yv)

    errs = []
    hs = []
    for ny in (6, 12):
        sol, dh_pos, dh_neg, interface = _solve_poisson_nitsche_multimesh_ufl(
            degree=1,
            ny_neg=ny,
            ny_pos=ny + 1,
            k_neg=k_neg,
            k_pos=k_pos,
            gamma=20.0,
            u_neg=u_neg,
            grad_u_neg=grad_u_neg,
            u_pos=u_pos,
            grad_u_pos=grad_u_pos,
            f=f,
            backend=backend,
        )

        n_pos = int(dh_pos.total_dofs)
        sol_pos = np.asarray(sol[:n_pos], dtype=float)
        sol_neg = np.asarray(sol[n_pos:], dtype=float)

        uh_pos = Function(name="uh_pos", field_name="u", dof_handler=dh_pos)
        uh_neg = Function(name="uh_neg", field_name="u", dof_handler=dh_neg)
        slp = np.asarray(dh_pos.get_field_slice("u"), dtype=int)
        sln = np.asarray(dh_neg.get_field_slice("u"), dtype=int)
        uh_pos.set_nodal_values(slp, sol_pos[slp])
        uh_neg.set_nodal_values(sln, sol_neg[sln])

        q = 10
        dxp = dx(metadata={"q": q})
        dxn = dx(metadata={"q": q})

        u_ex_pos = Analytic(lambda xv, yv: u_pos(xv, yv), degree=q + 2)
        u_ex_neg = Analytic(lambda xv, yv: u_neg(xv, yv), degree=q + 2)
        g_ex_pos = Analytic(lambda xv, yv: np.stack(grad_u_pos(xv, yv), axis=-1), degree=q + 2)
        g_ex_neg = Analytic(lambda xv, yv: np.stack(grad_u_neg(xv, yv), axis=-1), degree=q + 2)

        err2_p = (uh_pos - u_ex_pos) * (uh_pos - u_ex_pos)
        err2_n = (uh_neg - u_ex_neg) * (uh_neg - u_ex_neg)
        res_p = assemble_form(
            Equation(err2_p * dxp, None),
            dof_handler=dh_pos,
            assembler_hooks={err2_p: {"name": "err2"}},
            backend=backend,
        )
        res_n = assemble_form(
            Equation(err2_n * dxn, None),
            dof_handler=dh_neg,
            assembler_hooks={err2_n: {"name": "err2"}},
            backend=backend,
        )
        eL2 = float(np.sqrt(float(np.asarray(res_p["err2"]).ravel()[0]) + float(np.asarray(res_n["err2"]).ravel()[0])))

        h12_p = inner(grad(uh_pos) - g_ex_pos, grad(uh_pos) - g_ex_pos)
        h12_n = inner(grad(uh_neg) - g_ex_neg, grad(uh_neg) - g_ex_neg)
        res_p = assemble_form(
            Equation(h12_p * dxp, None),
            dof_handler=dh_pos,
            assembler_hooks={h12_p: {"name": "h12"}},
            backend=backend,
        )
        res_n = assemble_form(
            Equation(h12_n * dxn, None),
            dof_handler=dh_neg,
            assembler_hooks={h12_n: {"name": "h12"}},
            backend=backend,
        )
        eH1 = float(np.sqrt(float(np.asarray(res_p["h12"]).ravel()[0]) + float(np.asarray(res_n["h12"]).ravel()[0])))

        errs.append((eL2, eH1))
        h_pos = float(np.sqrt(np.asarray(dh_pos.mixed_element.mesh.areas_list, float)).max())
        h_neg = float(np.sqrt(np.asarray(dh_neg.mixed_element.mesh.areas_list, float)).max())
        hs.append(max(h_pos, h_neg))

    (eL2_0, eH1_0), (eL2_1, eH1_1) = errs
    h0, h1 = hs
    rate_L2 = np.log(eL2_0 / eL2_1) / np.log(h0 / h1)
    rate_H1 = np.log(eH1_0 / eH1_1) / np.log(h0 / h1)
    _print_table(
        f"[nonmatching] Poisson MMS (Nitsche) mixed tri/quad backend={backend}",
        ["ny", "h", "L2", "H1", "rate_L2", "rate_H1"],
        [
            [f"{6:d}", f"{hs[0]:.3e}", f"{errs[0][0]:.3e}", f"{errs[0][1]:.3e}", "", ""],
            [f"{12:d}", f"{hs[1]:.3e}", f"{errs[1][0]:.3e}", f"{errs[1][1]:.3e}", f"{rate_L2:.2f}", f"{rate_H1:.2f}"],
        ],
    )
    assert rate_L2 > 1.8
    assert rate_H1 > 0.9


@pytest.mark.parametrize("element", ["quad", "tri"])
def test_nonmatching_poisson_mortar_patch_linear(element):
    u_exact = lambda x, y: float(x + 2.0 * y - 0.3)
    grad_u = lambda x, y: (1.0, 2.0)
    f = lambda xv, yv: 0.0 * xv

    sol, dh, interface, mapping = _solve_poisson_mortar_ufl(
        element=element,
        degree=1,
        ny_neg=6,
        ny_pos=9,
        k_neg=1.0,
        k_pos=1.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend="python",
    )

    u_slice = np.asarray(dh.get_field_slice("u"), dtype=int)
    uh = Function(name="uh", field_name="u", dof_handler=dh)
    uh.set_nodal_values(u_slice, sol[u_slice])

    q = 8
    bs_pos, bs_neg = _component_bitsets(mapping.mesh, pos_eids=mapping.pos_elem_ids, neg_eids=mapping.neg_elem_ids)
    dx_pos = dx(defined_on=bs_pos, metadata={"q": q})
    dx_neg = dx(defined_on=bs_neg, metadata={"q": q})
    u_ex = Analytic(lambda xv, yv: u_exact(xv, yv), degree=q + 2)
    err2 = (uh - u_ex) * (uh - u_ex)
    res = assemble_form(
        Equation(err2 * dx_pos + err2 * dx_neg, None),
        dof_handler=dh,
        assembler_hooks={err2: {"name": "err2"}},
        backend="python",
    )
    assert float(np.sqrt(np.asarray(res["err2"]).ravel()[0])) < 1.0e-11

    jump2 = (Neg(uh) - Pos(uh)) * (Neg(uh) - Pos(uh))
    res = assemble_form(
        Equation(jump2 * dNonmatchingInterface(metadata={"q": q, "interface": interface}), None),
        dof_handler=dh,
        assembler_hooks={jump2: {"name": "jump2"}},
        backend="python",
    )
    assert float(np.sqrt(np.asarray(res["jump2"]).ravel()[0])) < 1.0e-10


def test_nonmatching_stokes_mms_convergence_quad():
    backend = _get_convergence_backend()
    _skip_if_backend_unavailable(backend)

    mu = 1.0
    ux = lambda x, y: np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    uy = lambda x, y: -np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    p_exact = lambda x, y: np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    f1 = (
        lambda x, y: 2 * mu * np.pi**3 * np.sin(np.pi * x) * np.cos(np.pi * y)
        + 2 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
    )
    f2 = (
        lambda x, y: -2 * mu * np.pi**3 * np.cos(np.pi * x) * np.sin(np.pi * y)
        + 2 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    )
    f_vec = Analytic(lambda xv, yv: np.stack([f1(xv, yv), f2(xv, yv)], axis=-1), dim=1, degree=10)

    vel_space = FunctionSpace("V", ["ux", "uy"], dim=1)
    u = VectorTrialFunction(vel_space)
    v = VectorTestFunction(vel_space)
    p = TrialFunction("p")
    q = TestFunction("p")

    def eps(w):
        return 0.5 * (grad(w) + grad(w).T)

    errs = []
    hs = []
    for ny in (4, 8):
        nx_neg = max(2, int(round(0.5 * ny)))
        nx_pos = max(2, int(round(0.5 * (ny + 1))))
        mesh_neg = _make_submesh(element="quad", poly_order=2, nx=nx_neg, ny=ny, offset_x=0.0)
        mesh_pos = _make_submesh(element="quad", poly_order=2, nx=nx_pos, ny=ny + 1, offset_x=0.5)

        mesh_neg.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})
        mesh_pos.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})

        interface = build_nonmatching_interface(mesh_neg=mesh_neg, mesh_pos=mesh_pos)
        mapping = build_composite_mesh(mesh_pos=mesh_pos, mesh_neg=mesh_neg, order="pos_neg")
        mesh = mapping.mesh
        interface_c = lift_nonmatching_interface_to_composite(interface=interface, mapping=mapping)

        mesh.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})
        for eid in mapping.pos_elem_ids:
            mesh.elements_list[int(eid)].tag = "pos"
        for eid in mapping.neg_elem_ids:
            mesh.elements_list[int(eid)].tag = "neg"

        fields = {"ux": 2, "uy": 2, "p": 1}
        dh = DofHandler(MixedElement(mesh, fields), method="cg")

        qvol = 10
        bs_pos, bs_neg = _component_bitsets(mesh, pos_eids=mapping.pos_elem_ids, neg_eids=mapping.neg_elem_ids)
        dx_pos = dx(defined_on=bs_pos, metadata={"q": qvol})
        dx_neg = dx(defined_on=bs_neg, metadata={"q": qvol})
        dGamma = dNonmatchingInterface(metadata={"q": qvol, "interface": interface_c})

        # Nitsche interface
        n = FacetNormal()
        h = CellDiameter()
        denom = float(mu) + float(mu)
        kappa_pos = Constant(float(mu) / denom)
        kappa_neg = Constant(float(mu) / denom)

        jump_u = Neg(u) - Pos(u)
        jump_v = Neg(v) - Pos(v)

        sigma_pos_u = Constant(2.0 * mu) * eps(Pos(u)) - Pos(p) * Identity(2)
        sigma_neg_u = Constant(2.0 * mu) * eps(Neg(u)) - Neg(p) * Identity(2)
        avg_t_u = kappa_pos * dot(sigma_pos_u, n) + kappa_neg * dot(sigma_neg_u, n)

        sigma_pos_v = Constant(2.0 * mu) * eps(Pos(v)) - Pos(q) * Identity(2)
        sigma_neg_v = Constant(2.0 * mu) * eps(Neg(v)) - Neg(q) * Identity(2)
        avg_t_v = kappa_pos * dot(sigma_pos_v, n) + kappa_neg * dot(sigma_neg_v, n)

        stab = Constant(40.0 * (float(mu) + float(mu))) / h
        a_if = (inner(avg_t_u, jump_v) + inner(avg_t_v, jump_u) + stab * inner(jump_u, jump_v)) * dGamma

        a_vol = (Constant(2.0 * mu) * inner(eps(u), eps(v)) - p * div(v) + q * div(u)) * dx_pos
        a_vol += (Constant(2.0 * mu) * inner(eps(u), eps(v)) - p * div(v) + q * div(u)) * dx_neg
        L = dot(f_vec, v) * dx_pos + dot(f_vec, v) * dx_neg

        # Pressure pin to remove nullspace
        dh.tag_dof_by_locator("pressure_pin", "p", lambda xx, yy: abs(float(xx)) < 1e-12 and abs(float(yy)) < 1e-12)

        bcs = [
            BoundaryCondition("ux", "dirichlet", "boundary", lambda xx, yy: float(ux(xx, yy))),
            BoundaryCondition("uy", "dirichlet", "boundary", lambda xx, yy: float(uy(xx, yy))),
            BoundaryCondition("p", "dirichlet", "pressure_pin", lambda xx, yy: float(p_exact(xx, yy))),
        ]

        K, F = assemble_form(Equation(a_vol + a_if, L), dof_handler=dh, bcs=bcs, backend=backend)
        sol = spla.spsolve(K.tocsc(), F)

        ux_h = Function(name="ux_h", field_name="ux", dof_handler=dh)
        uy_h = Function(name="uy_h", field_name="uy", dof_handler=dh)
        p_h = Function(name="p_h", field_name="p", dof_handler=dh)

        ux_sl = np.asarray(dh.get_field_slice("ux"), dtype=int)
        uy_sl = np.asarray(dh.get_field_slice("uy"), dtype=int)
        p_sl = np.asarray(dh.get_field_slice("p"), dtype=int)
        ux_h.set_nodal_values(ux_sl, sol[ux_sl])
        uy_h.set_nodal_values(uy_sl, sol[uy_sl])
        p_h.set_nodal_values(p_sl, sol[p_sl])

        ux_ex = Analytic(lambda xv, yv: ux(xv, yv), degree=12)
        uy_ex = Analytic(lambda xv, yv: uy(xv, yv), degree=12)
        p_ex = Analytic(lambda xv, yv: p_exact(xv, yv), degree=12)

        err_u2 = (ux_h - ux_ex) * (ux_h - ux_ex) + (uy_h - uy_ex) * (uy_h - uy_ex)
        err_p2 = (p_h - p_ex) * (p_h - p_ex)
        res = assemble_form(
            Equation(err_u2 * dx_pos + err_u2 * dx_neg + err_p2 * dx_pos + err_p2 * dx_neg, None),
            dof_handler=dh,
            assembler_hooks={err_u2: {"name": "eu2"}, err_p2: {"name": "ep2"}},
            backend=backend,
        )
        errs.append((float(np.sqrt(np.asarray(res["eu2"]).ravel()[0])), float(np.sqrt(np.asarray(res["ep2"]).ravel()[0]))))
        hs.append(float(np.sqrt(np.asarray(mesh.areas_list, float)).max()))

    (eu0, ep0), (eu1, ep1) = errs
    h0, h1 = hs
    rate_u = np.log(eu0 / eu1) / np.log(h0 / h1)
    rate_p = np.log(ep0 / ep1) / np.log(h0 / h1)
    _print_table(
        f"[nonmatching] Stokes MMS (Nitsche) element=quad P2/P1 backend={backend}",
        ["ny", "h", "L2(u)", "L2(p)", "rate_u", "rate_p"],
        [
            [f"{4:d}", f"{hs[0]:.3e}", f"{errs[0][0]:.3e}", f"{errs[0][1]:.3e}", "", ""],
            [f"{8:d}", f"{hs[1]:.3e}", f"{errs[1][0]:.3e}", f"{errs[1][1]:.3e}", f"{rate_u:.2f}", f"{rate_p:.2f}"],
        ],
    )
    assert rate_u > 2.5
    assert rate_p > 1.5


@pytest.mark.skipif(not _have_numba(), reason="numba not available")
def test_nonmatching_backend_parity_poisson_nitsche_jit():
    u_exact = lambda x, y: float(x + 2.0 * y - 0.3)
    grad_u = lambda x, y: (1.0, 2.0)
    f = lambda xv, yv: 0.0 * xv

    sol_py, dh_py, *_ = _solve_poisson_nitsche_ufl(
        element="quad",
        degree=1,
        ny_neg=4,
        ny_pos=5,
        k_neg=1.0,
        k_pos=1.0,
        gamma=20.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend="python",
    )
    sol_jit, dh_jit, *_ = _solve_poisson_nitsche_ufl(
        element="quad",
        degree=1,
        ny_neg=4,
        ny_pos=5,
        k_neg=1.0,
        k_pos=1.0,
        gamma=20.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend="jit",
    )

    sl = np.asarray(dh_py.get_field_slice("u"), dtype=int)
    assert float(np.max(np.abs(sol_py[sl] - sol_jit[sl]))) < 1.0e-12


@pytest.mark.skipif(not _have_pybind11(), reason="pybind11 not available")
def test_nonmatching_backend_parity_poisson_nitsche_cpp():
    u_exact = lambda x, y: float(x + 2.0 * y - 0.3)
    grad_u = lambda x, y: (1.0, 2.0)
    f = lambda xv, yv: 0.0 * xv

    sol_py, dh_py, *_ = _solve_poisson_nitsche_ufl(
        element="quad",
        degree=1,
        ny_neg=4,
        ny_pos=5,
        k_neg=1.0,
        k_pos=1.0,
        gamma=20.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend="python",
    )
    sol_cpp, dh_cpp, *_ = _solve_poisson_nitsche_ufl(
        element="quad",
        degree=1,
        ny_neg=4,
        ny_pos=5,
        k_neg=1.0,
        k_pos=1.0,
        gamma=20.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend="cpp",
    )

    sl = np.asarray(dh_py.get_field_slice("u"), dtype=int)
    assert float(np.max(np.abs(sol_py[sl] - sol_cpp[sl]))) < 1.0e-12


@pytest.mark.skipif(not _have_numba(), reason="numba not available")
def test_nonmatching_backend_parity_poisson_mortar_jit():
    u_exact = lambda x, y: float(x + 2.0 * y - 0.3)
    grad_u = lambda x, y: (1.0, 2.0)
    f = lambda xv, yv: 0.0 * xv

    sol_py, dh_py, *_ = _solve_poisson_mortar_ufl(
        element="quad",
        degree=1,
        ny_neg=4,
        ny_pos=5,
        k_neg=1.0,
        k_pos=1.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend="python",
    )
    sol_jit, dh_jit, *_ = _solve_poisson_mortar_ufl(
        element="quad",
        degree=1,
        ny_neg=4,
        ny_pos=5,
        k_neg=1.0,
        k_pos=1.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend="jit",
    )

    sl = np.asarray(dh_py.get_field_slice("u"), dtype=int)
    assert float(np.max(np.abs(sol_py[sl] - sol_jit[sl]))) < 1.0e-12


@pytest.mark.skipif(not _have_pybind11(), reason="pybind11 not available")
def test_nonmatching_backend_parity_poisson_mortar_cpp():
    u_exact = lambda x, y: float(x + 2.0 * y - 0.3)
    grad_u = lambda x, y: (1.0, 2.0)
    f = lambda xv, yv: 0.0 * xv

    sol_py, dh_py, *_ = _solve_poisson_mortar_ufl(
        element="quad",
        degree=1,
        ny_neg=4,
        ny_pos=5,
        k_neg=1.0,
        k_pos=1.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend="python",
    )
    sol_cpp, dh_cpp, *_ = _solve_poisson_mortar_ufl(
        element="quad",
        degree=1,
        ny_neg=4,
        ny_pos=5,
        k_neg=1.0,
        k_pos=1.0,
        u_neg=u_exact,
        grad_u_neg=grad_u,
        u_pos=u_exact,
        grad_u_pos=grad_u,
        f=f,
        backend="cpp",
    )

    sl = np.asarray(dh_py.get_field_slice("u"), dtype=int)
    assert float(np.max(np.abs(sol_py[sl] - sol_cpp[sl]))) < 1.0e-12


@pytest.mark.skipif(not _have_numba(), reason="numba not available")
def test_nonmatching_backend_parity_stokes_nitsche_matrix_jit():
    nodes, elems, edges, corners = structured_quad(0.5, 1.0, nx=1, ny=2, poly_order=2, offset=(0.0, 0.0))
    mesh_neg = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=2)
    nodes, elems, edges, corners = structured_quad(0.5, 1.0, nx=1, ny=3, poly_order=2, offset=(0.5, 0.0))
    mesh_pos = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=2)
    mesh_neg.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})
    mesh_pos.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})

    interface = build_nonmatching_interface(mesh_neg=mesh_neg, mesh_pos=mesh_pos)
    mapping = build_composite_mesh(mesh_pos=mesh_pos, mesh_neg=mesh_neg, order="pos_neg")
    mesh = mapping.mesh
    interface_c = lift_nonmatching_interface_to_composite(interface=interface, mapping=mapping)
    mesh.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})

    fields = {"ux": 2, "uy": 2, "p": 1}
    dh = DofHandler(MixedElement(mesh, fields), method="cg")

    vel_space = FunctionSpace("V", ["ux", "uy"], dim=1)
    u = VectorTrialFunction(vel_space)
    v = VectorTestFunction(vel_space)
    p = TrialFunction("p")
    q = TestFunction("p")

    def eps(w):
        return 0.5 * (grad(w) + grad(w).T)

    mu_val = 1.0
    denom = 2.0 * mu_val
    kappa_pos = Constant(mu_val / denom)
    kappa_neg = Constant(mu_val / denom)
    n = FacetNormal()
    h = CellDiameter()
    dGamma = dNonmatchingInterface(metadata={"q": 6, "interface": interface_c})

    jump_u = Neg(u) - Pos(u)
    jump_v = Neg(v) - Pos(v)
    sigma_pos_u = Constant(2.0 * mu_val) * eps(Pos(u)) - Pos(p) * Identity(2)
    sigma_neg_u = Constant(2.0 * mu_val) * eps(Neg(u)) - Neg(p) * Identity(2)
    avg_t_u = kappa_pos * dot(sigma_pos_u, n) + kappa_neg * dot(sigma_neg_u, n)

    sigma_pos_v = Constant(2.0 * mu_val) * eps(Pos(v)) - Pos(q) * Identity(2)
    sigma_neg_v = Constant(2.0 * mu_val) * eps(Neg(v)) - Neg(q) * Identity(2)
    avg_t_v = kappa_pos * dot(sigma_pos_v, n) + kappa_neg * dot(sigma_neg_v, n)

    stab = Constant(20.0 * (2.0 * mu_val)) / h
    a_if = (inner(avg_t_u, jump_v) + inner(avg_t_v, jump_u) + stab * inner(jump_u, jump_v)) * dGamma

    K_py, _ = assemble_form(Equation(a_if, None), dof_handler=dh, bcs=[], backend="python")
    K_jit, _ = assemble_form(Equation(a_if, None), dof_handler=dh, bcs=[], backend="jit")
    diff = (K_py - K_jit).tocoo()
    assert diff.data.size == 0 or float(np.max(np.abs(diff.data))) < 1.0e-12


@pytest.mark.skipif(not _have_pybind11(), reason="pybind11 not available")
def test_nonmatching_backend_parity_stokes_nitsche_matrix_cpp():
    nodes, elems, edges, corners = structured_quad(0.5, 1.0, nx=1, ny=2, poly_order=2, offset=(0.0, 0.0))
    mesh_neg = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=2)
    nodes, elems, edges, corners = structured_quad(0.5, 1.0, nx=1, ny=3, poly_order=2, offset=(0.5, 0.0))
    mesh_pos = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=2)
    mesh_neg.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})
    mesh_pos.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})

    interface = build_nonmatching_interface(mesh_neg=mesh_neg, mesh_pos=mesh_pos)
    mapping = build_composite_mesh(mesh_pos=mesh_pos, mesh_neg=mesh_neg, order="pos_neg")
    mesh = mapping.mesh
    interface_c = lift_nonmatching_interface_to_composite(interface=interface, mapping=mapping)
    mesh.tag_boundary_edges({"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True})

    fields = {"ux": 2, "uy": 2, "p": 1}
    dh = DofHandler(MixedElement(mesh, fields), method="cg")

    vel_space = FunctionSpace("V", ["ux", "uy"], dim=1)
    u = VectorTrialFunction(vel_space)
    v = VectorTestFunction(vel_space)
    p = TrialFunction("p")
    q = TestFunction("p")

    def eps(w):
        return 0.5 * (grad(w) + grad(w).T)

    mu_val = 1.0
    denom = 2.0 * mu_val
    kappa_pos = Constant(mu_val / denom)
    kappa_neg = Constant(mu_val / denom)
    n = FacetNormal()
    h = CellDiameter()
    dGamma = dNonmatchingInterface(metadata={"q": 6, "interface": interface_c})

    jump_u = Neg(u) - Pos(u)
    jump_v = Neg(v) - Pos(v)
    sigma_pos_u = Constant(2.0 * mu_val) * eps(Pos(u)) - Pos(p) * Identity(2)
    sigma_neg_u = Constant(2.0 * mu_val) * eps(Neg(u)) - Neg(p) * Identity(2)
    avg_t_u = kappa_pos * dot(sigma_pos_u, n) + kappa_neg * dot(sigma_neg_u, n)

    sigma_pos_v = Constant(2.0 * mu_val) * eps(Pos(v)) - Pos(q) * Identity(2)
    sigma_neg_v = Constant(2.0 * mu_val) * eps(Neg(v)) - Neg(q) * Identity(2)
    avg_t_v = kappa_pos * dot(sigma_pos_v, n) + kappa_neg * dot(sigma_neg_v, n)

    stab = Constant(20.0 * (2.0 * mu_val)) / h
    a_if = (inner(avg_t_u, jump_v) + inner(avg_t_v, jump_u) + stab * inner(jump_u, jump_v)) * dGamma

    K_py, _ = assemble_form(Equation(a_if, None), dof_handler=dh, bcs=[], backend="python")
    K_cpp, _ = assemble_form(Equation(a_if, None), dof_handler=dh, bcs=[], backend="cpp")
    diff = (K_py - K_cpp).tocoo()
    assert diff.data.size == 0 or float(np.max(np.abs(diff.data))) < 1.0e-12
