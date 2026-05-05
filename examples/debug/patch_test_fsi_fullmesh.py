#!/usr/bin/env python
"""
Patch-style sanity checks for the full-mesh FSI setup.

This is a focused, fast test that validates the two failure modes that made
`comparison_fsi_fullmesh.py` disagree previously:
  1) Gmsh physical cell tags (fluid/solid) being mis-assigned to dolfinx cells.
  2) Outlet boundary facets (tag=13) being incompletely mapped in dolfinx.

It compares a small set of sensitive terms (volume solid stress + outlet
traction) between PyCutFEM and FEniCSx on the full `fsi_conforming.msh` mesh.

Usage:
  conda run --no-capture-output -n fenicsx \\
      python -u examples/debug/patch_test_fsi_fullmesh.py --backend jit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from petsc4py import PETSc
from scipy.sparse import csr_matrix

import dolfinx.fem
import dolfinx.fem.petsc

from pycutfem.ufl.forms import Equation, assemble_form

from examples.debug.comparison_fsi_fullmesh import (
    read_mesh,
    setup_pycutfem,
    setup_fenics,
    initialize_pycutfem,
    initialize_fenics,
    build_pycutfem_terms,
    build_fenics_terms,
    create_true_dof_map,
)


def _max_abs_sparse(A: csr_matrix) -> float:
    return float(np.max(np.abs(A.data))) if A.nnz else 0.0


def _assemble_fenics_vector(form) -> np.ndarray:
    vec = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(form))
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    return vec.array


def _assemble_fenics_matrix(form) -> csr_matrix:
    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(form))
    A.assemble()
    indptr, indices, data = A.getValuesCSR()
    return csr_matrix((data, indices, indptr), shape=A.getSize())


def _check_vec(name: str, pc_form, fx_form, *, P_map: np.ndarray, dh_pc, backend: str, rtol: float, atol: float) -> None:
    _, vec_pc = assemble_form(Equation(None, pc_form), dof_handler=dh_pc, quad_degree=8, backend=backend)
    vec_pc = vec_pc.flatten()
    vec_fx = _assemble_fenics_vector(fx_form)[P_map]

    diff = vec_pc - vec_fx
    maxdiff = float(np.max(np.abs(diff))) if diff.size else 0.0
    scale = max(float(np.max(np.abs(vec_pc))) if vec_pc.size else 0.0, float(np.max(np.abs(vec_fx))) if vec_fx.size else 0.0)
    tol = atol + rtol * scale
    ok = maxdiff <= tol
    status = "ok" if ok else "FAIL"
    print(f"[{status}] {name}: max|diff|={maxdiff:.3e} (tol={tol:.3e})")
    if not ok:
        raise SystemExit(1)


def _check_mat(name: str, pc_form, fx_form, *, P_map: np.ndarray, dh_pc, backend: str, rtol: float, atol: float) -> None:
    mat_pc, _ = assemble_form(Equation(pc_form, None), dof_handler=dh_pc, quad_degree=8, backend=backend)
    mat_pc = mat_pc.tocsr()
    mat_fx = _assemble_fenics_matrix(fx_form)[P_map, :][:, P_map].tocsr()

    diff = (mat_pc - mat_fx).tocsr()
    maxdiff = _max_abs_sparse(diff)
    scale = max(_max_abs_sparse(mat_pc), _max_abs_sparse(mat_fx))
    tol = atol + rtol * scale
    ok = maxdiff <= tol
    status = "ok" if ok else "FAIL"
    print(f"[{status}] {name}: max|diff|={maxdiff:.3e} (tol={tol:.3e})")
    if not ok:
        raise SystemExit(1)


def parse_args():
    ap = argparse.ArgumentParser(description="FSI full-mesh patch checks (PyCutFEM vs FEniCSx).")
    ap.add_argument("--mesh", type=Path, default=Path("examples/meshes/fsi_conforming.msh"))
    ap.add_argument("--backend", choices=("jit", "python"), default="jit", help="PyCutFEM assembly backend.")
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--atol", type=float, default=1e-8)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    marks = {"fluid": 1, "solid": 2, "outlet": 13}
    quad_order = 8

    mesh_data = read_mesh(args.mesh)
    pc = setup_pycutfem(mesh_data, poly_order=2, dt_val=1.0, theta_val=1.0, quad_order=quad_order, marks=marks)
    fx = setup_fenics(mesh_data, poly_order=2, dt_val=1.0, theta_val=1.0, quad_order=quad_order, marks=marks)
    initialize_pycutfem(pc)
    initialize_fenics(fx)

    P_map = create_true_dof_map(pc["dof_handler"], fx["W"])
    res_pc, jac_pc = build_pycutfem_terms(pc)
    res_fx, jac_fx = build_fenics_terms(fx)

    # The two most sensitive residual terms:
    _check_vec(
        "res: solid_stress_theta",
        res_pc["solid_stress_theta"],
        res_fx["solid_stress_theta"],
        P_map=P_map,
        dh_pc=pc["dof_handler"],
        backend=args.backend,
        rtol=args.rtol,
        atol=args.atol,
    )
    _check_vec(
        "res: outlet_theta",
        res_pc["outlet_theta"],
        res_fx["outlet_theta"],
        P_map=P_map,
        dh_pc=pc["dof_handler"],
        backend=args.backend,
        rtol=args.rtol,
        atol=args.atol,
    )

    # And the Jacobian boundary contribution:
    _check_mat(
        "jac: outlet",
        jac_pc["jac_outlet"],
        jac_fx["jac_outlet"],
        P_map=P_map,
        dh_pc=pc["dof_handler"],
        backend=args.backend,
        rtol=args.rtol,
        atol=args.atol,
    )

    print("All patch checks passed.")


if __name__ == "__main__":
    main()

