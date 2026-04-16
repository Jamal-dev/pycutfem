from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from examples.NIRB.dvms import _update_fluid_dvms_predicted_subscale, _update_fluid_dvms_state_from_previous_step
from examples.NIRB.example2_local_setup import load_example2_local_setup
from examples.NIRB.run_example2_local import (
    CoordinateLookup,
    _assemble_fluid_local_velocity_contribution_raw,
    _build_fluid_problem,
    _fluid_boundary_conditions,
    _load_reference_partitioned_meshes,
)


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


def _assign_scalar_field(field, dh, *, coords: np.ndarray, values: np.ndarray) -> None:
    lookup = {_coord_key(x, y): float(values[i]) for i, (x, y) in enumerate(np.asarray(coords, dtype=float))}
    dh._ensure_dof_coords()
    ids = np.asarray(dh.get_field_slice(field.field_name), dtype=int)
    dof_coords = np.asarray(dh._dof_coords[ids], dtype=float)
    vals = np.asarray([lookup[_coord_key(x, y)] for x, y in dof_coords], dtype=float)
    field.set_nodal_values(ids, vals)


def _pycutfem_full_dof_map(fluid, mesh_f) -> dict[tuple[int, str], int]:
    dh = fluid["dh"]
    dh._ensure_dof_coords()
    new_to_old = getattr(mesh_f, "_mdpa_new_to_old_node", None)
    if new_to_old is None:
        raise RuntimeError("Reference fluid mesh does not carry the MDPA node-id mapping.")
    if isinstance(new_to_old, dict):
        old_id_of_new = lambda nid: int(new_to_old[int(nid)])
    else:
        arr = np.asarray(new_to_old, dtype=int).reshape(-1)
        old_id_of_new = lambda nid: int(arr[int(nid)])
    node_coords = np.asarray(mesh_f.nodes_x_y_pos[:, :2], dtype=float)
    old_id_by_coord = {_coord_key(x, y): old_id_of_new(i) for i, (x, y) in enumerate(node_coords)}
    mapping: dict[tuple[int, str], int] = {}
    var_map = {"ux": "VELOCITY_X", "uy": "VELOCITY_Y", "p": "PRESSURE"}
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
    parser = argparse.ArgumentParser(description="Compare the pycutfem assembled global fluid system against a Kratos dump.")
    parser.add_argument("--kratos-dump", type=Path, required=True)
    parser.add_argument("--backend", choices=("python", "jit", "cpp"), default="cpp")
    parser.add_argument(
        "--state-backend",
        choices=("python", "jit", "cpp"),
        default=None,
        help=(
            "Backend for the already-closed DVMS state updates "
            "(_update_fluid_dvms_state_from_previous_step / predicted_subscale). "
            "Defaults to --backend."
        ),
    )
    parser.add_argument("--quad-order", type=int, default=1)
    parser.add_argument("--bossak-alpha", type=float, default=-0.3)
    parser.add_argument("--dynamic-tau", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/NIRB/artifacts/compare_kratos_fluid_global_system.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_backend = str(args.backend if args.state_backend is None else args.state_backend)
    with np.load(Path(args.kratos_dump).resolve(), allow_pickle=False) as data:
        A_raw_ref = _unpack_csr("A_raw", data)
        A_constrained_ref = _unpack_csr("A_constrained", data)
        b_raw_ref = np.asarray(data["b_raw"], dtype=float)
        b_constrained_ref = np.asarray(data["b_constrained"], dtype=float)
        eq_node_ids = np.asarray(data["equation_node_ids"], dtype=int)
        eq_var_names = np.asarray(data["equation_var_names"], dtype=str)
        eq_is_fixed = np.asarray(data["equation_is_fixed"], dtype=bool)
        node_coords_ref = np.asarray(data["node_coords_ref"], dtype=float)
        velocity = np.asarray(data["velocity"], dtype=float)
        velocity_prev = np.asarray(data["velocity_prev"], dtype=float)
        acceleration = np.asarray(data["acceleration"], dtype=float)
        mesh_velocity = np.asarray(data["mesh_velocity"], dtype=float)
        pressure = np.asarray(data["pressure"], dtype=float)
        stage = str(np.asarray(data["stage"]).reshape(()))

    setup = load_example2_local_setup()
    mesh_f, _ = _load_reference_partitioned_meshes(setup=setup)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=int(args.quad_order))
    mu_f = float(setup.material.density * setup.material.kinematic_viscosity)
    dt = float(setup.boundaries.time_step)

    _assign_vector_field(fluid["u_k"], fluid["dh"], coords=node_coords_ref, values=velocity)
    _assign_vector_field(fluid["u_prev"], fluid["dh"], coords=node_coords_ref, values=velocity_prev)
    # Kratos dumps the current-step ACCELERATION. The local DVMS operator expects
    # the previous-step acceleration in a_prev; on the first step that history is
    # identically zero.
    if float(np.max(np.abs(np.asarray(velocity_prev, dtype=float)))) <= 1.0e-14:
        a_prev_values = np.zeros_like(np.asarray(acceleration, dtype=float))
    else:
        a_prev_values = np.asarray(acceleration, dtype=float)
    _assign_vector_field(fluid["a_prev"], fluid["dh"], coords=node_coords_ref, values=a_prev_values)
    _assign_scalar_field(fluid["p_k"], fluid["dh"], coords=node_coords_ref, values=pressure)

    zero_disp = np.zeros_like(node_coords_ref)
    _assign_vector_field(fluid["d_mesh"], fluid["dh"], coords=node_coords_ref, values=zero_disp)
    _assign_vector_field(fluid["d_prev"], fluid["dh"], coords=node_coords_ref, values=-dt * mesh_velocity)

    _update_fluid_dvms_state_from_previous_step(
        state=fluid["dvms_state"],
        dh=fluid["dh"],
        mesh=mesh_f,
        u_prev=fluid["u_prev"],
        d_prev=fluid["d_prev"],
        d_geo=fluid["d_mesh"],
        backend=state_backend,
    )
    if str(stage).strip().lower() == "predicted":
        # Kratos' pre-solve Build() evaluates the first predictor system with
        # the old-step subscale history. Refreshing the current-step predicted
        # subscale here injects a reproducible inlet-strip mismatch on step 1.
        fluid["dvms_state"].predicted_subscale_velocity[:, :] = np.asarray(
            fluid["dvms_state"].old_subscale_velocity,
            dtype=float,
        )
        fluid["dvms_state"].sync_coefficient("predicted_subscale_velocity")
    else:
        _update_fluid_dvms_predicted_subscale(
            state=fluid["dvms_state"],
            dh=fluid["dh"],
            mesh=mesh_f,
            u_k=fluid["u_k"],
            u_prev=fluid["u_prev"],
            a_prev=fluid["a_prev"],
            p_k=fluid["p_k"],
            d_mesh=fluid["d_mesh"],
            d_prev=fluid["d_prev"],
            rho_f=float(setup.material.density),
            mu_f=mu_f,
            dt=dt,
            bossak_alpha=float(args.bossak_alpha),
            dynamic_tau=float(args.dynamic_tau),
            backend=state_backend,
        )

    zero_iface = CoordinateLookup(np.asarray([[0.0, 0.0]], dtype=float), np.zeros((1, 2), dtype=float), dim=2)

    def inlet_profile(x: float, y: float) -> float:
        del x
        return setup.geometry.inlet_velocity(y, dt, reference_velocity=float(setup.material.max_velocity))

    bcs, bcs_homog = _fluid_boundary_conditions(
        iface_velocity=zero_iface,
        inlet_lookup=inlet_profile,
        interface_tag=setup.geometry.interface_tag,
        outlet_tag=setup.geometry.outlet_tag,
        walls_tag=setup.geometry.walls_tag,
        cylinder_tag=setup.geometry.cylinder_tag,
    )
    fluid["_current_bcs"] = bcs

    A_raw_local, b_raw_local = _assemble_fluid_local_velocity_contribution_raw(
        prob=fluid,
        rho_f=float(setup.material.density),
        mu_f=mu_f,
        dt=dt,
        quad_order=int(args.quad_order),
        bossak_alpha=float(args.bossak_alpha),
        need_matrix=True,
        contribution_mode="system",
        apply_dirichlet_lift=False,
        backend=str(args.backend),
    )
    bc_map = fluid["dh"].get_dirichlet_data(bcs_homog) or {}
    A_constrained_local, b_constrained_local = _apply_pycutfem_dirichlet(A_raw_local, b_raw_local, bc_map)

    py_dof_map = _pycutfem_full_dof_map(fluid, mesh_f)
    perm = np.asarray([py_dof_map[(int(node_id), str(var_name))] for node_id, var_name in zip(eq_node_ids.tolist(), eq_var_names.tolist(), strict=False)], dtype=int)

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
        "backend": str(args.backend),
        "state_backend": state_backend,
        "raw_matrix": _sparse_diff_stats(A_raw_perm, A_raw_ref),
        "constrained_matrix": _sparse_diff_stats(A_constrained_perm, A_constrained_ref),
        "raw_rhs": _diff_stats(b_raw_perm, b_raw_ref),
        "constrained_rhs": _diff_stats(b_constrained_perm, b_constrained_ref),
        "free_matrix": _sparse_diff_stats(A_free_local, A_free_ref),
        "free_rhs": _diff_stats(b_free_local, b_free_ref),
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
