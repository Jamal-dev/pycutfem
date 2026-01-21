#!/usr/bin/env python3
"""
Compare a *single* assembled linear system (A, b) between:
  - FEniCS (dolfin) monolithic mixed formulation, and
  - pycutfem monolithic formulation,
using sparse matrices and coordinate-based DOF matching.

This is intended to diagnose why the time-marching consolidation benchmark
diverges between the two implementations on larger meshes.

Run (requires dolfin, so use the fenics env):
  conda run --no-capture-output -n fenics \\
    python examples/poroelasticity/compare_system_step_sparse.py \\
      --mesh-file examples/poroelasticity/mesh_files/consolidation_nx31_ny14_delaunay.npz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp

# Allow running without installing pycutfem into the active env.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from consolidation_fenics_reference import _delaunay_cells, _generate_points, _mesh_from_points_cells, _structured_cells
from consolidation_pycutfem import _build_p2_tri_mesh


@dataclass(frozen=True)
class Params:
    L: float = 20.0
    H: float = 10.0
    pressure_region: float = 5.0
    dt: float = 0.3
    t_eval: float = 0.3
    t1: float = 3.6
    E: float = 14.4e9
    nu: float = 0.2
    biot: float = 0.78
    biot_modulus: float = 13.5e9
    permeability: float = 2e-10
    p_d: float = 380e6
    p_1: float = 1.54e9

    @property
    def mu(self) -> float:
        return float(self.E) / (2.0 * (1.0 + float(self.nu)))

    @property
    def lam(self) -> float:
        nu = float(self.nu)
        return float(self.E) * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    @property
    def traction_rate_y(self) -> float:
        return (-float(self.p_1) / float(self.t1)) if float(self.t_eval) < float(self.t1) else 0.0


def _coord_key(coords: np.ndarray, ndigits: int = 12) -> np.ndarray:
    return np.round(np.asarray(coords, dtype=float), decimals=ndigits)


def _sorted_dofs(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    return np.lexsort((coords[:, 1], coords[:, 0]))


def _compare(name: str, A_f: np.ndarray, A_p: np.ndarray) -> None:
    diff = A_p - A_f
    nrm = np.linalg.norm(A_f.ravel())
    dnrm = np.linalg.norm(diff.ravel())
    rel = dnrm / nrm if nrm > 0 else float("nan")
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    print(f"{name:10s}  ||Δ||={dnrm:.3e}  ||A||={nrm:.3e}  rel={rel:.3e}  max|Δ|={max_abs:.3e}")


def _dolfin_to_csr(A) -> sp.csr_matrix:
    from dolfin import as_backend_type

    mat = as_backend_type(A).mat()
    indptr, indices, data = mat.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=(A.size(0), A.size(1)))


def _assemble_fenics(*, points: np.ndarray, cells: np.ndarray, params: Params):
    from dolfin import (
        Constant,
        DirichletBC,
        FiniteElement,
        Function,
        FunctionSpace,
        Identity,
        Measure,
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

    mesh = _mesh_from_points_cells(points, cells)

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

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

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
    theta = Constant(0.5)
    dt = Constant(params.dt)

    def eps(w):
        return sym(grad(w))

    def sigma(w):
        return 2.0 * mu * eps(w) + lam * tr(eps(w)) * Identity(2)

    # Previous state coefficient: u0=0, p0=p_d.
    w0 = Function(W)
    # Fill pressure dofs to p_d.
    Vp, map_p = W.sub(2).collapse(collapsed_dofs=True)
    dofs_p = np.asarray([map_p[i] for i in range(Vp.dim())], dtype=int)
    init = np.zeros(W.dim(), dtype=float)
    init[dofs_p] = float(params.p_d)
    w0.vector().set_local(init)
    w0.vector().apply("insert")
    (ux0, uy0, p0) = split(w0)
    u0 = as_vector((ux0, uy0))

    traction_rate = Constant((0.0, float(params.traction_rate_y)))

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

    bcs = [
        DirichletBC(W.sub(0), Constant(0.0), boundaries, BM.left),
        DirichletBC(W.sub(0), Constant(0.0), boundaries, BM.right),
        DirichletBC(W.sub(1), Constant(0.0), boundaries, BM.bottom),
        DirichletBC(W.sub(2), Constant(params.p_d), boundaries, BM.top),
    ]
    A, b = assemble_system(a, L, bcs)

    # Subspace maps for coordinate matching.
    subspaces = {}
    for name, i in (("ux", 0), ("uy", 1), ("p", 2)):
        V_i, map_i = W.sub(i).collapse(collapsed_dofs=True)
        coords = np.asarray(V_i.tabulate_dof_coordinates().reshape((-1, 2)), dtype=float)
        dofs_parent = np.asarray([map_i[j] for j in range(V_i.dim())], dtype=int)
        subspaces[name] = {"coords": coords, "dofs": dofs_parent}

    bc_dofs: set[int] = set()
    for bc in bcs:
        bc_dofs.update(int(k) for k in bc.get_boundary_values().keys())

    return {"A": _dolfin_to_csr(A), "b": np.asarray(b.get_local(), dtype=float), "subspaces": subspaces, "bc_dofs": bc_dofs}


def _assemble_pycutfem(*, points: np.ndarray, cells: np.ndarray, params: Params, backend: str):
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

    mesh = _build_p2_tri_mesh(points, cells)
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
    u0.nodal_values.fill(0.0)
    p0.nodal_values.fill(float(params.p_d))

    traction_rate = Constant(np.asarray([0.0, float(params.traction_rate_y)], dtype=float))
    theta = Constant(0.5)
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
        inner(sigma_s(u), eps(v)) * dx(metadata={"q": 5})
        - biot * p * div(v) * dx(metadata={"q": 5})
        + biot * div(u) * q * dx(metadata={"q": 5})
        + invM * p * q * dx(metadata={"q": 5})
        + theta * dt * inner(k_perm * grad(p), grad(q)) * dx(metadata={"q": 5})
    )
    L = (
        inner(sigma_s(u0), eps(v)) * dx(metadata={"q": 5})
        - biot * p0 * div(v) * dx(metadata={"q": 5})
        + dt * dot(traction_rate, v) * dS(mesh.edge_bitset("pressure_load"), metadata={"q": 5})
        + biot * div(u0) * q * dx(metadata={"q": 5})
        + invM * p0 * q * dx(metadata={"q": 5})
        - (Constant(1.0) - theta) * dt * inner(k_perm * grad(p0), grad(q)) * dx(metadata={"q": 5})
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
    K_bc = K.copy()
    F_bc = F.copy()
    compiler._apply_bcs(K_bc, F_bc, bcs)

    subspaces = {}
    for name in ("ux", "uy", "p"):
        subspaces[name] = {"coords": np.asarray(dh.get_dof_coords(name), dtype=float), "dofs": np.asarray(dh.get_field_slice(name), dtype=int)}

    return {"A": K_bc.tocsr(), "b": np.asarray(F_bc, dtype=float), "subspaces": subspaces, "bc_dofs": bc_dofs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-file", default=None)
    ap.add_argument("--triangulation", choices=("delaunay", "structured"), default="delaunay")
    ap.add_argument("--nx", type=int, default=31)
    ap.add_argument("--ny", type=int, default=14)
    ap.add_argument("--backend", choices=("python", "jit"), default="python")
    args = ap.parse_args()

    params = Params()

    mesh_path = Path(args.mesh_file).resolve() if args.mesh_file else None
    if mesh_path is not None and mesh_path.exists():
        data = np.load(mesh_path)
        points = np.asarray(data["points"], dtype=float)
        cells = np.asarray(data["cells"], dtype=int)
    else:
        points = _generate_points(L=params.L, H=params.H, nx=args.nx, ny=args.ny)
        if args.triangulation == "structured":
            cells = _structured_cells(nx=args.nx, ny=args.ny)
        else:
            cells = _delaunay_cells(points)
        if mesh_path is not None:
            mesh_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(mesh_path, points=points, cells=cells)
            print(f"[compare] wrote mesh file {mesh_path}")

    fen = _assemble_fenics(points=points, cells=cells, params=params)
    pyc = _assemble_pycutfem(points=points, cells=cells, params=params, backend=args.backend)

    # Compare block-wise on free DOFs.
    fen_bc = fen["bc_dofs"]
    pyc_bc = pyc["bc_dofs"]

    for fld in ("ux", "uy", "p"):
        c_f = _coord_key(fen["subspaces"][fld]["coords"])
        c_p = _coord_key(pyc["subspaces"][fld]["coords"])
        sf = _sorted_dofs(c_f)
        sp_ = _sorted_dofs(c_p)
        maxc = float(np.max(np.abs(c_f[sf] - c_p[sp_])))
        print(f"[coords] {fld}: ndofs={c_f.shape[0]} max|Δ|={maxc:.3e}")

    for r in ("ux", "uy", "p"):
        for c in ("ux", "uy", "p"):
            dofs_f_r = np.asarray(fen["subspaces"][r]["dofs"], dtype=int)
            dofs_f_c = np.asarray(fen["subspaces"][c]["dofs"], dtype=int)
            dofs_p_r = np.asarray(pyc["subspaces"][r]["dofs"], dtype=int)
            dofs_p_c = np.asarray(pyc["subspaces"][c]["dofs"], dtype=int)

            ord_f_r = _sorted_dofs(_coord_key(fen["subspaces"][r]["coords"]))
            ord_f_c = _sorted_dofs(_coord_key(fen["subspaces"][c]["coords"]))
            ord_p_r = _sorted_dofs(_coord_key(pyc["subspaces"][r]["coords"]))
            ord_p_c = _sorted_dofs(_coord_key(pyc["subspaces"][c]["coords"]))

            free_f_r = ~np.isin(dofs_f_r, np.fromiter(fen_bc, dtype=int))
            free_f_c = ~np.isin(dofs_f_c, np.fromiter(fen_bc, dtype=int))
            free_p_r = ~np.isin(dofs_p_r, np.fromiter(pyc_bc, dtype=int))
            free_p_c = ~np.isin(dofs_p_c, np.fromiter(pyc_bc, dtype=int))

            ord_f_r = ord_f_r[free_f_r[ord_f_r]]
            ord_f_c = ord_f_c[free_f_c[ord_f_c]]
            ord_p_r = ord_p_r[free_p_r[ord_p_r]]
            ord_p_c = ord_p_c[free_p_c[ord_p_c]]

            blk_f = fen["A"][np.ix_(dofs_f_r, dofs_f_c)][np.ix_(ord_f_r, ord_f_c)].toarray()
            blk_p = pyc["A"][np.ix_(dofs_p_r, dofs_p_c)][np.ix_(ord_p_r, ord_p_c)].toarray()
            _compare(f"{r}-{c}", blk_f, blk_p)

    # RHS on free dofs.
    for fld in ("ux", "uy", "p"):
        dofs_f = np.asarray(fen["subspaces"][fld]["dofs"], dtype=int)
        dofs_p = np.asarray(pyc["subspaces"][fld]["dofs"], dtype=int)
        ord_f = _sorted_dofs(_coord_key(fen["subspaces"][fld]["coords"]))
        ord_p = _sorted_dofs(_coord_key(pyc["subspaces"][fld]["coords"]))
        free_f = ~np.isin(dofs_f, np.fromiter(fen_bc, dtype=int))
        free_p = ~np.isin(dofs_p, np.fromiter(pyc_bc, dtype=int))
        ord_f = ord_f[free_f[ord_f]]
        ord_p = ord_p[free_p[ord_p]]
        rhs_f = fen["b"][dofs_f][ord_f]
        rhs_p = pyc["b"][dofs_p][ord_p]
        _compare(f"rhs-{fld}", rhs_f, rhs_p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
