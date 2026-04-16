from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from examples.NIRB.example2_local_setup import load_example2_local_setup
from examples.NIRB.run_example2_local import (
    CoordinateLookup,
    _assemble_kratos_local_solid_system_full,
    _boundary_point_load_vector,
    _boundary_vector_snapshot,
    _build_solid_problem,
    _load_reference_partitioned_meshes,
    _maybe_build_kratos_local_solid_backend,
    _mesh_node_ids,
    _solid_residual_and_jacobian,
)
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, NewtonSolver


def _coord_key(x: float, y: float, ndigits: int = 12) -> tuple[float, float]:
    return (round(float(x), ndigits), round(float(y), ndigits))


def _unpack_csr(prefix: str, data) -> sp.csr_matrix:
    return sp.csr_matrix(
        (
            np.asarray(data[f"{prefix}_data"], dtype=float),
            np.asarray(data[f"{prefix}_indices"], dtype=np.int32),
            np.asarray(data[f"{prefix}_indptr"], dtype=np.int32),
        ),
        shape=tuple(np.asarray(data[f"{prefix}_shape"], dtype=int).tolist()),
    )


def _diff_stats(local, reference) -> dict[str, float]:
    loc = np.asarray(local, dtype=float).reshape(-1)
    ref = np.asarray(reference, dtype=float).reshape(-1)
    diff = loc - ref
    ref_norm = max(float(np.linalg.norm(ref)), 1.0e-15)
    cosine_denom = max(float(np.linalg.norm(loc) * np.linalg.norm(ref)), 1.0e-15)
    return {
        "local_inf_norm": float(np.max(np.abs(loc))) if loc.size else 0.0,
        "reference_inf_norm": float(np.max(np.abs(ref))) if ref.size else 0.0,
        "diff_inf_norm": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "diff_l2_norm": float(np.linalg.norm(diff)),
        "rel_l2_norm": float(np.linalg.norm(diff) / ref_norm),
        "cosine": float(np.dot(loc, ref) / cosine_denom),
    }


def _sparse_diff_stats(local: sp.csr_matrix, reference: sp.csr_matrix) -> dict[str, float]:
    diff = (local - reference).tocsr()
    loc_norm = float(np.max(np.abs(local.data))) if local.nnz else 0.0
    ref_norm = float(np.max(np.abs(reference.data))) if reference.nnz else 0.0
    diff_inf = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
    diff_l2 = float(np.linalg.norm(diff.data)) if diff.nnz else 0.0
    ref_l2 = max(float(np.linalg.norm(reference.data)) if reference.nnz else 0.0, 1.0e-15)
    return {
        "local_inf_norm": loc_norm,
        "reference_inf_norm": ref_norm,
        "diff_inf_norm": diff_inf,
        "diff_l2_norm": diff_l2,
        "rel_l2_norm": float(diff_l2 / ref_l2),
        "nnz_local": int(local.nnz),
        "nnz_reference": int(reference.nnz),
        "nnz_diff": int(diff.nnz),
    }


def _top_diff_rows(
    *,
    local: sp.csr_matrix,
    reference: sp.csr_matrix,
    rhs_local: np.ndarray,
    rhs_reference: np.ndarray,
    node_ids: np.ndarray,
    var_names: np.ndarray,
    is_fixed: np.ndarray,
    top_k: int = 12,
) -> list[dict[str, object]]:
    diff = (local - reference).tocsr()
    rows = []
    for row in range(int(diff.shape[0])):
        start = int(diff.indptr[row])
        stop = int(diff.indptr[row + 1])
        row_diff_inf = float(np.max(np.abs(diff.data[start:stop]))) if stop > start else 0.0
        rhs_diff = float(abs(float(rhs_local[row]) - float(rhs_reference[row])))
        rows.append((max(row_diff_inf, rhs_diff), row, row_diff_inf, rhs_diff))
    rows.sort(reverse=True)
    out: list[dict[str, object]] = []
    for _, row, row_diff_inf, rhs_diff in rows[: max(int(top_k), 1)]:
        lstart = int(local.indptr[row]); lstop = int(local.indptr[row + 1])
        rstart = int(reference.indptr[row]); rstop = int(reference.indptr[row + 1])
        out.append(
            {
                "row": int(row),
                "node_id": int(node_ids[row]),
                "variable": str(var_names[row]),
                "is_fixed": bool(is_fixed[row]),
                "matrix_row_diff_inf": float(row_diff_inf),
                "rhs_diff_abs": float(rhs_diff),
                "matrix_row_local_inf": float(np.max(np.abs(local.data[lstart:lstop]))) if lstop > lstart else 0.0,
                "matrix_row_reference_inf": float(np.max(np.abs(reference.data[rstart:rstop]))) if rstop > rstart else 0.0,
                "rhs_local": float(rhs_local[row]),
                "rhs_reference": float(rhs_reference[row]),
            }
        )
    return out


def _assign_vector_field(field, dh, *, coords: np.ndarray, values: np.ndarray) -> None:
    lookup = {_coord_key(x, y): np.asarray(values[i], dtype=float) for i, (x, y) in enumerate(np.asarray(coords, dtype=float))}
    dh._ensure_dof_coords()
    for comp_idx, comp in enumerate(field.components):
        ids = np.asarray(dh.get_field_slice(comp.field_name), dtype=int)
        dof_coords = np.asarray(dh._dof_coords[ids], dtype=float)
        vals = np.asarray([lookup[_coord_key(x, y)][comp_idx] for x, y in dof_coords], dtype=float)
        comp.set_nodal_values(ids, vals)


def _sample_point_values(*, coords_src: np.ndarray, values_src: np.ndarray, coords_tgt: np.ndarray) -> np.ndarray:
    lookup = {_coord_key(x, y): np.asarray(values_src[i], dtype=float) for i, (x, y) in enumerate(np.asarray(coords_src, dtype=float))}
    return np.asarray([lookup[_coord_key(x, y)] for x, y in np.asarray(coords_tgt, dtype=float)], dtype=float)


def _pycutfem_full_dof_map(solid, mesh_s) -> dict[tuple[int, str], int]:
    dh = solid["dh"]
    dh._ensure_dof_coords()
    old_node_ids = np.asarray(_mesh_node_ids(mesh_s), dtype=int).reshape(-1)
    node_coords = np.asarray(mesh_s.nodes_x_y_pos[:, :2], dtype=float)
    old_id_by_coord = {_coord_key(x, y): int(old_node_ids[i]) for i, (x, y) in enumerate(node_coords)}
    mapping: dict[tuple[int, str], int] = {}
    var_map = {"dx": "DISPLACEMENT_X", "dy": "DISPLACEMENT_Y"}
    for field_name, var_name in var_map.items():
        ids = np.asarray(dh.get_field_slice(field_name), dtype=int)
        coords = np.asarray(dh._dof_coords[ids], dtype=float)
        for gdof, (x, y) in zip(ids.tolist(), coords.tolist(), strict=False):
            mapping[(old_id_by_coord[_coord_key(x, y)], var_name)] = int(gdof)
    return mapping


def _apply_pycutfem_dirichlet(A_raw: sp.csr_matrix, b_raw: np.ndarray, bc_map: dict[int, float]) -> tuple[sp.csr_matrix, np.ndarray]:
    A = A_raw.tolil(copy=True)
    rhs = np.asarray(b_raw, dtype=float).copy()
    if not bc_map:
        return A.tocsr(), rhs
    rows = np.fromiter((int(g) for g in bc_map.keys()), dtype=int)
    vals = np.fromiter((float(v) for v in bc_map.values()), dtype=float)
    x_bc = np.zeros(int(A_raw.shape[0]), dtype=float)
    x_bc[rows] = vals
    rhs = rhs - A_raw @ x_bc
    A[rows, :] = 0.0
    A[:, rows] = 0.0
    A[rows, rows] = 1.0
    rhs[rows] = vals
    return A.tocsr(), rhs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the pycutfem assembled structural global system against a Kratos dump.")
    parser.add_argument("--kratos-dump", type=Path, required=True)
    parser.add_argument("--backend", choices=("python", "jit", "cpp"), default="cpp")
    parser.add_argument("--linear-backend", choices=("scipy", "petsc"), default="petsc")
    parser.add_argument("--quad-order", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/NIRB/artifacts/compare_kratos_structure_global_system.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(Path(args.kratos_dump).resolve(), allow_pickle=False) as data:
        A_raw_ref = _unpack_csr("A_raw", data)
        A_constrained_ref = _unpack_csr("A_constrained", data)
        b_raw_ref = np.asarray(data["b_raw"], dtype=float)
        b_constrained_ref = np.asarray(data["b_constrained"], dtype=float)
        eq_node_ids = np.asarray(data["equation_node_ids"], dtype=int)
        eq_var_names = np.asarray(data["equation_var_names"], dtype=str)
        eq_is_fixed = np.asarray(data["equation_is_fixed"], dtype=bool)
        node_coords_ref = np.asarray(data["node_coords_ref"], dtype=float)
        displacement = np.asarray(data["displacement"], dtype=float)
        point_load = np.asarray(data["point_load"], dtype=float)
        stage = str(np.asarray(data["stage"]).reshape(()))
        iteration = int(np.asarray(data["iteration"]).reshape(()))

    setup = load_example2_local_setup()
    _, mesh_s = _load_reference_partitioned_meshes(setup=setup)
    solid = _build_solid_problem(mesh_s, poly_order=1)
    kratos_local_solid_backend = _maybe_build_kratos_local_solid_backend(
        benchmark_root=Path(setup.reference.root),
        prob=solid,
    )
    _assign_vector_field(solid["d_k"], solid["dh"], coords=node_coords_ref, values=displacement)
    _assign_vector_field(solid["d_prev"], solid["dh"], coords=node_coords_ref, values=displacement)

    iface_coords, _ = _boundary_vector_snapshot(solid["dh"], solid["d_k"], setup.geometry.interface_tag)
    zero_lookup = CoordinateLookup(
        np.asarray(iface_coords, dtype=float),
        np.zeros((iface_coords.shape[0], 2), dtype=float),
        dim=2,
    )
    solid_res, solid_jac, solid_bcs, solid_bcs_homog = _solid_residual_and_jacobian(
        prob=solid,
        traction_lookup=zero_lookup,
        mu_s=float(setup.material.shear_modulus),
        lambda_s=float(setup.material.lame_lambda),
        interface_tag=setup.geometry.interface_tag,
        clamp_tag=setup.geometry.clamp_tag,
        quad_order=int(args.quad_order),
    )

    solver = NewtonSolver(
        residual_form=solid_res,
        jacobian_form=solid_jac,
        dof_handler=solid["dh"],
        mixed_element=solid["me"],
        bcs=solid_bcs,
        bcs_homog=solid_bcs_homog,
        newton_params=NewtonParameters(
            newton_tol=1.0e-10,
            max_newton_iter=1,
            line_search=False,
            globalization="none",
        ),
        lin_params=LinearSolverParameters(backend=str(args.linear_backend)),
        quad_order=int(args.quad_order),
        backend=str(args.backend),
    )

    iface_load_values = _sample_point_values(coords_src=node_coords_ref, values_src=point_load, coords_tgt=iface_coords)
    point_load_full = _boundary_point_load_vector(
        solid["dh"],
        vector=solid["d_k"],
        tag=setup.geometry.interface_tag,
        values=iface_load_values,
    )

    coeffs = {solid["d_k"].name: solid["d_k"], solid["d_prev"].name: solid["d_prev"]}
    if kratos_local_solid_backend is None:
        A_raw_local, b_internal_local = solver._assemble_system_raw(coeffs, need_matrix=True)
    else:
        del coeffs
        A_raw_local, b_internal_local = _assemble_kratos_local_solid_system_full(
            backend=kratos_local_solid_backend,
            d_k=solid["d_k"],
            need_matrix=True,
        )
    b_raw_local = np.asarray(b_internal_local, dtype=float) - np.asarray(point_load_full, dtype=float)
    bc_map = solid["dh"].get_dirichlet_data(solid_bcs_homog) or {}
    A_constrained_local, b_constrained_local = _apply_pycutfem_dirichlet(A_raw_local, b_raw_local, bc_map)

    py_dof_map = _pycutfem_full_dof_map(solid, mesh_s)
    perm = np.asarray(
        [py_dof_map[(int(node_id), str(var_name))] for node_id, var_name in zip(eq_node_ids.tolist(), eq_var_names.tolist(), strict=False)],
        dtype=int,
    )

    A_raw_perm = A_raw_local.tocsr()[perm][:, perm]
    A_constrained_perm = A_constrained_local.tocsr()[perm][:, perm]
    b_raw_perm = np.asarray(b_raw_local, dtype=float)[perm]
    b_constrained_perm = np.asarray(b_constrained_local, dtype=float)[perm]
    free = np.where(~eq_is_fixed)[0]
    A_free_ref = A_constrained_ref.tocsr()[free][:, free]
    A_free_local = A_constrained_perm.tocsr()[free][:, free]
    b_free_ref = np.asarray(b_constrained_ref, dtype=float)[free]
    b_free_local = np.asarray(b_constrained_perm, dtype=float)[free]

    report = {
        "kratos_dump": str(Path(args.kratos_dump).resolve()),
        "stage": stage,
        "iteration": int(iteration),
        "backend": str(args.backend),
        "linear_backend": str(args.linear_backend),
        "quad_order": int(args.quad_order),
        "rhs_sign_note": "Kratos StructuralMechanics Build() returns external-internal, while the pycutfem residual assembly here is internal-external.",
        "raw_matrix": _sparse_diff_stats(A_raw_perm, A_raw_ref),
        "constrained_matrix": _sparse_diff_stats(A_constrained_perm, A_constrained_ref),
        "raw_rhs": _diff_stats(b_raw_perm, b_raw_ref),
        "raw_rhs_against_negated_reference": _diff_stats(b_raw_perm, -b_raw_ref),
        "constrained_rhs": _diff_stats(b_constrained_perm, b_constrained_ref),
        "constrained_rhs_against_negated_reference": _diff_stats(b_constrained_perm, -b_constrained_ref),
        "free_matrix": _sparse_diff_stats(A_free_local, A_free_ref),
        "free_rhs": _diff_stats(b_free_local, b_free_ref),
        "free_rhs_against_negated_reference": _diff_stats(b_free_local, -b_free_ref),
        "raw_matrix_nnz_match": bool(int(A_raw_perm.nnz) == int(A_raw_ref.nnz)),
        "constrained_matrix_nnz_match": bool(int(A_constrained_perm.nnz) == int(A_constrained_ref.nnz)),
        "raw_top_diff_rows": _top_diff_rows(
            local=A_raw_perm,
            reference=A_raw_ref,
            rhs_local=b_raw_perm,
            rhs_reference=b_raw_ref,
            node_ids=eq_node_ids,
            var_names=eq_var_names,
            is_fixed=eq_is_fixed,
        ),
        "constrained_top_diff_rows": _top_diff_rows(
            local=A_constrained_perm,
            reference=A_constrained_ref,
            rhs_local=b_constrained_perm,
            rhs_reference=b_constrained_ref,
            node_ids=eq_node_ids,
            var_names=eq_var_names,
            is_fixed=eq_is_fixed,
        ),
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
