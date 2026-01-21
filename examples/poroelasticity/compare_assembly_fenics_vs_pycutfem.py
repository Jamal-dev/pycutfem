#!/usr/bin/env python3
"""
Assemble the Biot consolidation system in both:
  1) FEniCS (dolfin) monolithic mixed space, and
  2) pycutfem monolithic mixed space,
then compare matrices/vectors by matching DOFs via coordinates.

This is a *diagnostic* script to pinpoint where the two implementations differ
(elasticity block, coupling blocks, diffusion block, traction load, BC handling).

Run (requires dolfin, so use the fenics env):
  conda run --no-capture-output -n fenics \\
    python examples/poroelasticity/compare_assembly_fenics_vs_pycutfem.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la

# Allow running this example without installing pycutfem into the active env.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from consolidation_pycutfem import _build_p2_tri_mesh, _generate_points


@dataclass(frozen=True)
class Params:
    L: float = 20.0
    H: float = 10.0
    pressure_region: float = 5.0
    final_time: float = 36.0
    num_time_steps: int = 120
    theta: float = 0.5
    E: float = 14.4e9
    nu: float = 0.2
    biot: float = 0.78
    biot_modulus: float = 13.5e9
    permeability: float = 2e-10
    p_d: float = 380e6
    p_1: float = 1.54e9

    @property
    def dt(self) -> float:
        return float(self.final_time) / float(self.num_time_steps)

    @property
    def t_1(self) -> float:
        return float(self.final_time) / 10.0

    @property
    def mu(self) -> float:
        return float(self.E) / (2.0 * (1.0 + float(self.nu)))

    @property
    def lam(self) -> float:
        # Match the notebook / reference code (plane strain / 3D Lamé parameter).
        nu = float(self.nu)
        return float(self.E) * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def traction_rate_y(self, t: float) -> float:
        return (-float(self.p_1) / float(self.t_1)) if t < float(self.t_1) else 0.0


def _sorted_dofs(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    return np.lexsort((coords[:, 1], coords[:, 0]))


def _coord_key(coords: np.ndarray, ndigits: int = 12) -> np.ndarray:
    c = np.asarray(coords, dtype=float)
    return np.round(c, decimals=ndigits)


def _compare_block(name: str, A_f: np.ndarray, A_p: np.ndarray, rtol: float = 1e-12, atol: float = 1e-10) -> None:
    diff = A_p - A_f
    nrm = np.linalg.norm(A_f.ravel())
    dnrm = np.linalg.norm(diff.ravel())
    rel = dnrm / nrm if nrm > 0 else float("nan")
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    ok = (dnrm <= atol + rtol * nrm) if np.isfinite(rel) else (max_abs <= atol)
    status = "OK" if ok else "DIFF"
    print(f"[{status}] {name:8s}  ||Δ||={dnrm:.3e}  ||A||={nrm:.3e}  rel={rel:.3e}  max|Δ|={max_abs:.3e}")


def _sorted_free_ord(subspace: dict, *, bc_dofs: set[int] | None, ndigits: int = 12) -> np.ndarray:
    coords = _coord_key(subspace["coords"], ndigits=ndigits)
    order = _sorted_dofs(coords)
    if bc_dofs is None:
        return order
    dofs = np.asarray(subspace["dofs"], dtype=int)
    bc = np.fromiter(bc_dofs, dtype=int) if bc_dofs else np.zeros(0, dtype=int)
    free = ~np.isin(dofs, bc)
    return order[free[order]]


def _map_fenics_parent_to_py_field(
    *,
    fen_sub: dict,
    py_sub: dict,
    fen_parent_values: np.ndarray,
    ndigits: int = 12,
) -> np.ndarray:
    coords_f = _coord_key(fen_sub["coords"], ndigits=ndigits)
    dofs_f = np.asarray(fen_sub["dofs"], dtype=int)
    vals_f = np.asarray(fen_parent_values, dtype=float)[dofs_f]

    coords_p = _coord_key(py_sub["coords"], ndigits=ndigits)
    out = np.zeros(coords_p.shape[0], dtype=float)

    key_to_idx = {tuple(coords_p[i]): int(i) for i in range(coords_p.shape[0])}
    for i in range(coords_f.shape[0]):
        out[key_to_idx[tuple(coords_f[i])]] = vals_f[i]
    return out


def _compare_system(*, label: str, fen: dict, pyc: dict, rtol: float, atol: float) -> None:
    print(f"\n== {label} ==")

    A_f = np.asarray(fen["A"], dtype=float)
    b_f = np.asarray(fen["b"], dtype=float)
    A_p = np.asarray(pyc["A"], dtype=float)
    b_p = np.asarray(pyc["b"], dtype=float)

    fen_bc = set(int(d) for d in np.asarray(fen.get("bc_dofs", []), dtype=int).ravel())
    pyc_bc = set(int(d) for d in np.asarray(pyc.get("bc_dofs", []), dtype=int).ravel())

    for fld in ("ux", "uy", "p"):
        c_f = _coord_key(fen["subspaces"][fld]["coords"])
        c_p = _coord_key(pyc["subspaces"][fld]["coords"])
        if c_f.shape != c_p.shape:
            raise SystemExit(f"{label}: field {fld} dof count mismatch fenics={c_f.shape[0]} pycutfem={c_p.shape[0]}")
        sort_f = _sorted_dofs(c_f)
        sort_p = _sorted_dofs(c_p)
        max_coord = float(np.max(np.abs(c_f[sort_f] - c_p[sort_p])))

        dofs_f = np.asarray(fen["subspaces"][fld]["dofs"], dtype=int)
        dofs_p = np.asarray(pyc["subspaces"][fld]["dofs"], dtype=int)
        n_bc_f = int(np.isin(dofs_f, np.fromiter(fen_bc, dtype=int)).sum()) if fen_bc else 0
        n_bc_p = int(np.isin(dofs_p, np.fromiter(pyc_bc, dtype=int)).sum()) if pyc_bc else 0
        print(f"[coords] {fld}: ndofs={c_f.shape[0]}  max|Δx|={max_coord:.3e}  bc(fenics)={n_bc_f}  bc(pycutfem)={n_bc_p}")

    # Compare free-free blocks so BC elimination strategy differences don't dominate.
    for r in ("ux", "uy", "p"):
        for c in ("ux", "uy", "p"):
            dofs_f_r = np.asarray(fen["subspaces"][r]["dofs"], dtype=int)
            dofs_f_c = np.asarray(fen["subspaces"][c]["dofs"], dtype=int)
            dofs_p_r = np.asarray(pyc["subspaces"][r]["dofs"], dtype=int)
            dofs_p_c = np.asarray(pyc["subspaces"][c]["dofs"], dtype=int)

            ord_f_r = _sorted_free_ord(fen["subspaces"][r], bc_dofs=fen_bc)
            ord_f_c = _sorted_free_ord(fen["subspaces"][c], bc_dofs=fen_bc)
            ord_p_r = _sorted_free_ord(pyc["subspaces"][r], bc_dofs=pyc_bc)
            ord_p_c = _sorted_free_ord(pyc["subspaces"][c], bc_dofs=pyc_bc)

            blk_f = A_f[np.ix_(dofs_f_r, dofs_f_c)][np.ix_(ord_f_r, ord_f_c)]
            blk_p = A_p[np.ix_(dofs_p_r, dofs_p_c)][np.ix_(ord_p_r, ord_p_c)]
            _compare_block(f"{r}-{c}", blk_f, blk_p, rtol=rtol, atol=atol)

    # RHS on free DOFs.
    for fld in ("ux", "uy", "p"):
        dofs_f = np.asarray(fen["subspaces"][fld]["dofs"], dtype=int)
        dofs_p = np.asarray(pyc["subspaces"][fld]["dofs"], dtype=int)
        ord_f = _sorted_free_ord(fen["subspaces"][fld], bc_dofs=fen_bc)
        ord_p = _sorted_free_ord(pyc["subspaces"][fld], bc_dofs=pyc_bc)
        rhs_f = b_f[dofs_f][ord_f]
        rhs_p = b_p[dofs_p][ord_p]
        _compare_block(f"rhs-{fld}", rhs_f, rhs_p, rtol=rtol, atol=atol)

    # Solve and compare (free DOFs only).
    w_f = np.linalg.solve(A_f, b_f)
    w_p = np.linalg.solve(A_p, b_p)
    for fld in ("ux", "uy", "p"):
        dofs_f = np.asarray(fen["subspaces"][fld]["dofs"], dtype=int)
        dofs_p = np.asarray(pyc["subspaces"][fld]["dofs"], dtype=int)
        ord_f = _sorted_free_ord(fen["subspaces"][fld], bc_dofs=fen_bc)
        ord_p = _sorted_free_ord(pyc["subspaces"][fld], bc_dofs=pyc_bc)
        sol_f = w_f[dofs_f][ord_f]
        sol_p = w_p[dofs_p][ord_p]
        _compare_block(f"sol-{fld}", sol_f, sol_p, rtol=1e-10, atol=1e-8)


def _fenics_assemble(
    *,
    points: np.ndarray,
    cells: np.ndarray,
    params: Params,
    quad_degree: int,
    t: float | None = None,
    w0_parent: np.ndarray | None = None,
):
    from dolfin import (
        Constant,
        DirichletBC,
        FiniteElement,
        Function,
        FunctionSpace,
        Identity,
        Measure,
        Mesh,
        MeshEditor,
        MeshFunction,
        MixedElement,
        SubDomain,
        TestFunctions,
        TrialFunctions,
        as_vector,
        assemble_system,
        div,
        dot,
        grad,
        inner,
        near,
        split,
        sym,
        tr,
    )

    vertices = np.asarray(points, dtype=float)
    tri = np.asarray(cells, dtype=int).copy()
    a = vertices[tri[:, 0]]
    b = vertices[tri[:, 1]]
    c = vertices[tri[:, 2]]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    flip = area2 < 0.0
    tri[flip, 1], tri[flip, 2] = tri[flip, 2], tri[flip, 1]

    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, "triangle", 2, 2)
    editor.init_vertices(vertices.shape[0])
    editor.init_cells(tri.shape[0])
    for i, xy in enumerate(vertices):
        editor.add_vertex(int(i), xy)
    for i, cell in enumerate(tri):
        editor.add_cell(int(i), cell)
    editor.close()

    class TopBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], params.H, 0.5) and (x[0] > params.pressure_region) and on_boundary

    class BottomBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0.0) and on_boundary

    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0) and on_boundary

    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], params.L) and on_boundary

    class PressureLoadBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return (x[0] <= params.pressure_region) and near(x[1], params.H, 0.5) and on_boundary

    class BM:
        top = 1
        bottom = 2
        left = 3
        right = 4
        pressure_load = 5

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    LeftBoundary().mark(boundaries, BM.left)
    RightBoundary().mark(boundaries, BM.right)
    TopBoundary().mark(boundaries, BM.top)
    BottomBoundary().mark(boundaries, BM.bottom)
    PressureLoadBoundary().mark(boundaries, BM.pressure_load)

    dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": quad_degree})
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries, metadata={"quadrature_degree": quad_degree})

    cell = mesh.ufl_cell()
    el_u = FiniteElement("CG", cell, 2)
    el_p = FiniteElement("CG", cell, 1)
    W = FunctionSpace(mesh, MixedElement([el_u, el_u, el_p]))

    (ux, uy, p) = TrialFunctions(W)
    (vx, vy, q) = TestFunctions(W)
    u = as_vector((ux, uy))
    v = as_vector((vx, vy))

    mu = Constant(params.mu)
    lam = Constant(params.lam)
    biot = Constant(params.biot)
    invM = Constant(1.0 / params.biot_modulus)
    k_perm = Constant(params.permeability)
    theta = Constant(params.theta)
    dt = Constant(params.dt)

    def eps(w):
        return sym(grad(w))

    def sigma(w):
        return 2.0 * mu * eps(w) + lam * tr(eps(w)) * Identity(2)

    bcs = [
        DirichletBC(W.sub(0), Constant(0.0), boundaries, BM.left),
        DirichletBC(W.sub(0), Constant(0.0), boundaries, BM.right),
        DirichletBC(W.sub(1), Constant(0.0), boundaries, BM.bottom),
        DirichletBC(W.sub(2), Constant(params.p_d), boundaries, BM.top),
    ]

    # Coordinate+DOF maps for matching with pycutfem.
    subspaces = {}
    for name, i in (("ux", 0), ("uy", 1), ("p", 2)):
        V_i, map_i = W.sub(i).collapse(collapsed_dofs=True)
        coords = np.asarray(V_i.tabulate_dof_coordinates().reshape((-1, 2)), dtype=float)
        dofs_parent = np.asarray([map_i[j] for j in range(V_i.dim())], dtype=int)
        subspaces[name] = {"coords": coords, "dofs": dofs_parent}

    # Previous step fields as a mixed Function coefficient.
    w0 = Function(W)
    if w0_parent is not None:
        w0.vector().set_local(np.asarray(w0_parent, dtype=float))
        w0.vector().apply("insert")
    else:
        init = np.zeros(W.dim(), dtype=float)
        init[subspaces["p"]["dofs"]] = float(params.p_d)
        w0.vector().set_local(init)
        w0.vector().apply("insert")
    (ux0, uy0, p0) = split(w0)
    u0 = as_vector((ux0, uy0))

    t_eval = float(params.dt) if t is None else float(t)
    traction_rate = Constant((0.0, float(params.traction_rate_y(t_eval))))

    a = (
        inner(sigma(u), eps(v)) * dx
        - biot * p * div(v) * dx
        + biot * div(u) * q * dx
        + invM * p * q * dx
        + theta * dt * inner(k_perm * grad(p), grad(q)) * dx
    )

    L = (
        inner(sigma(u0), eps(v)) * dx
        - biot * p0 * div(v) * dx
        + dt * dot(traction_rate, v) * ds(BM.pressure_load)
        + biot * div(u0) * q * dx
        + invM * p0 * q * dx
        - (Constant(1.0) - theta) * dt * inner(k_perm * grad(p0), grad(q)) * dx
    )

    A, b = assemble_system(a, L, bcs)
    A_dense = np.asarray(A.array(), dtype=float)
    b_dense = np.asarray(b.get_local(), dtype=float)

    bc_dofs: set[int] = set()
    for bc in bcs:
        bc_dofs.update(int(k) for k in bc.get_boundary_values().keys())

    return {"A": A_dense, "b": b_dense, "subspaces": subspaces, "bc_dofs": np.fromiter(bc_dofs, dtype=int)}


def _pycutfem_assemble(
    *,
    points: np.ndarray,
    simplices: np.ndarray,
    params: Params,
    quad_degree: int,
    backend: str,
    t: float | None = None,
    prev_ux: np.ndarray | None = None,
    prev_uy: np.ndarray | None = None,
    prev_p: np.ndarray | None = None,
):
    from pycutfem.core.dofhandler import DofHandler
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.ufl.compilers import FormCompiler
    from pycutfem.ufl.expressions import (
        Constant,
        Function,
        Identity,
        TestFunction,
        TrialFunction,
        VectorFunction,
        VectorTestFunction,
        VectorTrialFunction,
        div,
        dot,
        grad,
        inner,
        trace,
    )
    from pycutfem.ufl.forms import BoundaryCondition
    from pycutfem.ufl.functionspace import FunctionSpace
    from pycutfem.ufl.measures import dS, dx

    tri = simplices
    mesh = _build_p2_tri_mesh(points, tri)

    top_tol = 0.5
    mesh.tag_boundary_edges(
        {
            "bottom": lambda x, y: np.isclose(y, 0.0),
            "left": lambda x, y: np.isclose(x, 0.0),
            "right": lambda x, y: np.isclose(x, params.L),
            "top_drained": lambda x, y: (abs(y - params.H) <= top_tol) and (x > params.pressure_region),
            "pressure_load": lambda x, y: (abs(y - params.H) <= top_tol) and (x < params.pressure_region),
        }
    )

    mixed_element = MixedElement(mesh, field_specs={"ux": 2, "uy": 2, "p": 1})
    dh = DofHandler(mixed_element, method="cg")

    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)
    pres_space = FunctionSpace("pressure", ["p"], dim=0)

    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)

    u0 = VectorFunction(name="u0", field_names=["ux", "uy"], dof_handler=dh)
    p0 = Function(name="p0", field_name="p", dof_handler=dh)

    n_ux = int(len(dh.get_field_slice("ux")))
    n_uy = int(len(dh.get_field_slice("uy")))
    n_p = int(len(dh.get_field_slice("p")))
    if prev_ux is None:
        u0.nodal_values.fill(0.0)
    else:
        prev_ux = np.asarray(prev_ux, dtype=float).ravel()
        prev_uy = np.asarray(prev_uy, dtype=float).ravel() if prev_uy is not None else np.zeros(n_uy, dtype=float)
        if prev_ux.size != n_ux or prev_uy.size != n_uy:
            raise ValueError(f"prev u sizes mismatch: ux={prev_ux.size} (expected {n_ux}) uy={prev_uy.size} (expected {n_uy})")
        u0.nodal_values = np.concatenate([prev_ux, prev_uy]).astype(float, copy=False)
    if prev_p is None:
        p0.nodal_values.fill(float(params.p_d))
    else:
        prev_p = np.asarray(prev_p, dtype=float).ravel()
        if prev_p.size != n_p:
            raise ValueError(f"prev p size mismatch: {prev_p.size} (expected {n_p})")
        p0.nodal_values = prev_p

    t_eval = float(params.dt) if t is None else float(t)
    traction_rate = Constant(np.asarray([0.0, float(params.traction_rate_y(t_eval))], dtype=float))
    traction_rate._jit_name = "traction_rate"

    theta = Constant(params.theta)
    dt = Constant(params.dt)
    mu = Constant(params.mu)
    lam = Constant(params.lam)
    biot = Constant(params.biot)
    invM = Constant(1.0 / params.biot_modulus)
    k_perm = Constant(params.permeability)
    I2 = Identity(2)

    def eps(w):
        return 0.5 * (grad(w) + grad(w).T)

    def sigma_s(w):
        return Constant(2.0) * mu * eps(w) + lam * trace(eps(w)) * I2

    a = (
        inner(sigma_s(u), eps(v)) * dx(metadata={"q": quad_degree})
        - biot * p * div(v) * dx(metadata={"q": quad_degree})
        + biot * div(u) * q * dx(metadata={"q": quad_degree})
        + invM * p * q * dx(metadata={"q": quad_degree})
        + theta * dt * inner(k_perm * grad(p), grad(q)) * dx(metadata={"q": quad_degree})
    )

    L = (
        inner(sigma_s(u0), eps(v)) * dx(metadata={"q": quad_degree})
        - biot * p0 * div(v) * dx(metadata={"q": quad_degree})
        + dt
        * dot(traction_rate, v)
        * dS(mesh.edge_bitset("pressure_load"), metadata={"q": quad_degree})
        + biot * div(u0) * q * dx(metadata={"q": quad_degree})
        + invM * p0 * q * dx(metadata={"q": quad_degree})
        - (Constant(1.0) - theta)
        * dt
        * inner(k_perm * grad(p0), grad(q))
        * dx(metadata={"q": quad_degree})
    )

    bcs = [
        BoundaryCondition("ux", "dirichlet", "left", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "right", lambda x, y: 0.0),
        BoundaryCondition("uy", "dirichlet", "bottom", lambda x, y: 0.0),
        BoundaryCondition("p", "dirichlet", "top_drained", lambda x, y: float(params.p_d)),
    ]

    compiler = FormCompiler(dh, quadrature_order=None, backend=backend)
    ndofs = dh.total_dofs

    K = sp.lil_matrix((ndofs, ndofs))
    compiler.ctx["rhs"] = False
    compiler._assemble_form(a, K)
    K = K.tocsr()

    F = np.zeros(ndofs, dtype=float)
    compiler.ctx["rhs"] = True
    compiler._assemble_form(L, F)

    dirichlet = dh.get_dirichlet_data(bcs)
    bc_dofs = set(int(k) for k in dirichlet.keys()) if dirichlet else set()

    # Apply BCs to matrix+vector in-place (same machinery as the solver script).
    K_bc = K.copy()
    F_bc = F.copy()
    compiler._apply_bcs(K_bc, F_bc, bcs)

    A_dense = np.asarray(K_bc.toarray(), dtype=float)
    b_dense = np.asarray(F_bc, dtype=float)

    subspaces = {}
    for name in ("ux", "uy", "p"):
        subspaces[name] = {
            "coords": np.asarray(dh.get_dof_coords(name), dtype=float),
            "dofs": np.asarray(dh.get_field_slice(name), dtype=int),
        }

    return {"A": A_dense, "b": b_dense, "subspaces": subspaces, "bc_dofs": np.fromiter(bc_dofs, dtype=int)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=9)
    ap.add_argument("--ny", type=int, default=5)
    ap.add_argument("--quad-degree", type=int, default=5)
    ap.add_argument("--pycutfem-backend", choices=("python", "jit"), default="python")
    ap.add_argument("--rtol", type=float, default=1e-10)
    ap.add_argument("--atol", type=float, default=1e-8)
    args = ap.parse_args()

    params = Params()
    points = _generate_points(L=params.L, H=params.H, nx=args.nx, ny=args.ny)

    from scipy.spatial import Delaunay

    tri = Delaunay(points)
    simplices = np.asarray(tri.simplices, dtype=int)

    dt = float(params.dt)

    # Step 1 (u0=0, p0=p_d).
    fen1 = _fenics_assemble(points=points, cells=simplices, params=params, quad_degree=args.quad_degree, t=dt)
    pyc1 = _pycutfem_assemble(
        points=points,
        simplices=simplices,
        params=params,
        quad_degree=args.quad_degree,
        backend=args.pycutfem_backend,
        t=dt,
    )
    _compare_system(label="step 1", fen=fen1, pyc=pyc1, rtol=args.rtol, atol=args.atol)

    # Step 2: use fenics step-1 solution as the *shared* previous state.
    w1_fen_parent = np.linalg.solve(np.asarray(fen1["A"], dtype=float), np.asarray(fen1["b"], dtype=float))
    prev_ux = _map_fenics_parent_to_py_field(fen_sub=fen1["subspaces"]["ux"], py_sub=pyc1["subspaces"]["ux"], fen_parent_values=w1_fen_parent)
    prev_uy = _map_fenics_parent_to_py_field(fen_sub=fen1["subspaces"]["uy"], py_sub=pyc1["subspaces"]["uy"], fen_parent_values=w1_fen_parent)
    prev_p = _map_fenics_parent_to_py_field(fen_sub=fen1["subspaces"]["p"], py_sub=pyc1["subspaces"]["p"], fen_parent_values=w1_fen_parent)

    fen2 = _fenics_assemble(
        points=points,
        cells=simplices,
        params=params,
        quad_degree=args.quad_degree,
        t=2.0 * dt,
        w0_parent=w1_fen_parent,
    )
    pyc2 = _pycutfem_assemble(
        points=points,
        simplices=simplices,
        params=params,
        quad_degree=args.quad_degree,
        backend=args.pycutfem_backend,
        t=2.0 * dt,
        prev_ux=prev_ux,
        prev_uy=prev_uy,
        prev_p=prev_p,
    )
    _compare_system(label="step 2 (shared prev=fenics step1)", fen=fen2, pyc=pyc2, rtol=args.rtol, atol=args.atol)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
