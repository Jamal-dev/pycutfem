from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import Constant, TestFunction, TrialFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace

from examples.NIRB.example2_local_setup import load_example2_local_setup
from examples.NIRB.run_example2_local import _build_fluid_problem, _load_reference_partitioned_meshes
from examples.NIRB.dvms.symbolics import build_fluid_dvms_kratos_split_forms
from examples.NIRB.dvms.helpers import (
    _bossak_coefficients,
    _kratos_dvms_current_element_size_array,
    _kratos_dvms_element_size_coefficient,
)


_GROUPED_TO_INTERLEAVED_TRI3 = np.asarray([0, 3, 6, 1, 4, 7, 2, 5, 8], dtype=int)
_INTERLEAVED_TO_GROUPED_TRI3 = np.argsort(_GROUPED_TO_INTERLEAVED_TRI3)


def _local_old_node_ids(mesh, eid: int) -> np.ndarray:
    new_to_old = getattr(mesh, "_mdpa_new_to_old_node", None)
    if new_to_old is None:
        raise RuntimeError("Reference pycutfem mesh does not carry MDPA node-id mapping.")
    conn = np.asarray(mesh.elements_connectivity[int(eid)], dtype=int).reshape(-1)
    if isinstance(new_to_old, dict):
        return np.asarray([int(new_to_old[int(nid)]) for nid in conn], dtype=int)
    arr = np.asarray(new_to_old, dtype=int).reshape(-1)
    return np.asarray([int(arr[int(nid)]) for nid in conn], dtype=int)


def _dump_to_local_node_order(mesh, eid: int, node_ids: np.ndarray) -> np.ndarray:
    dump_ids = np.asarray(node_ids, dtype=int).reshape(-1)
    by_old_id = {int(old_id): i for i, old_id in enumerate(dump_ids)}
    local_old = _local_old_node_ids(mesh, int(eid))
    try:
        return np.asarray([by_old_id[int(old_id)] for old_id in local_old], dtype=int)
    except KeyError as exc:
        raise RuntimeError(
            f"Kratos dump node ids {dump_ids.tolist()} do not match pycutfem element old ids {local_old.tolist()}."
        ) from exc


def _kratos_interleaved_to_local_grouped_perm(mesh, eid: int, node_ids: np.ndarray) -> np.ndarray:
    order = _dump_to_local_node_order(mesh, int(eid), node_ids)
    return np.asarray(
        [3 * int(i) + 0 for i in order]
        + [3 * int(i) + 1 for i in order]
        + [3 * int(i) + 2 for i in order],
        dtype=int,
    )


def _find_pycutfem_element_by_old_node_ids(mesh, node_ids: np.ndarray) -> int:
    target = frozenset(int(i) for i in np.asarray(node_ids, dtype=int).reshape(-1))
    new_to_old = getattr(mesh, "_mdpa_new_to_old_node", None)
    if new_to_old is None:
        raise RuntimeError("Reference pycutfem mesh does not carry MDPA node-id mapping.")
    if isinstance(new_to_old, dict):
        old_id = lambda nid: int(new_to_old[int(nid)])
    else:
        arr = np.asarray(new_to_old, dtype=int).reshape(-1)
        old_id = lambda nid: int(arr[int(nid)])
    for eid, conn in enumerate(np.asarray(mesh.elements_connectivity, dtype=int)):
        old_ids = frozenset(old_id(nid) for nid in conn)
        if old_ids == target:
            return int(eid)
    raise RuntimeError(f"Could not match Kratos node ids {sorted(target)} to a pycutfem fluid element.")


def _local_trial_test_functions(dh):
    v_space = FunctionSpace("FluidVelocityLocal", ["ux", "uy"], dim=1)
    du = VectorTrialFunction(space=v_space, dof_handler=dh)
    v = VectorTestFunction(space=v_space, dof_handler=dh)
    dp = TrialFunction(name="dp_local", field_name="p", dof_handler=dh)
    q = TestFunction(name="q_local", field_name="p", dof_handler=dh)
    return du, dp, v, q


def _compress_single_fluid_block(dh, batch):
    eids = np.asarray(batch.element_ids, dtype=int).reshape(-1)
    if eids.shape[0] != 1:
        raise ValueError("Expected a single-element batch.")
    eid = int(eids[0])
    fluid_gdofs = np.concatenate(
        [
            np.asarray(dh.element_maps["ux"][eid], dtype=int),
            np.asarray(dh.element_maps["uy"][eid], dtype=int),
            np.asarray(dh.element_maps["p"][eid], dtype=int),
        ]
    )
    full = np.asarray(batch.gdofs_map[0], dtype=int)
    pos = np.asarray([int(np.where(full == gdof)[0][0]) for gdof in fluid_gdofs], dtype=int)
    lhs = None
    rhs = None
    if batch.K_elem is not None:
        k_full = np.asarray(batch.K_elem[0], dtype=float)
        lhs = k_full[np.ix_(pos, pos)]
    if batch.F_elem is not None:
        rhs = np.asarray(batch.F_elem[0], dtype=float)[pos]
    return lhs, rhs, fluid_gdofs


def _assemble_single_term(*, compiler, form, eid: int, dh):
    batch = compiler.assemble_volume_local_contributions(form, element_ids=np.asarray([int(eid)], dtype=int))
    return _compress_single_fluid_block(dh, batch)


def _set_local_fields_from_dump(
    *,
    mesh,
    fluid: dict[str, object],
    eid: int,
    dump,
    dt: float,
    zero_previous_history: bool = False,
) -> None:
    ux_g = np.asarray(fluid["dh"].element_maps["ux"][eid], dtype=int)
    uy_g = np.asarray(fluid["dh"].element_maps["uy"][eid], dtype=int)
    p_g = np.asarray(fluid["dh"].element_maps["p"][eid], dtype=int)
    mesh_dh = fluid.get("mesh_dh", fluid["dh"])
    mx_g = np.asarray(mesh_dh.element_maps["mx"][eid], dtype=int)
    my_g = np.asarray(mesh_dh.element_maps["my"][eid], dtype=int)

    local_order = _dump_to_local_node_order(mesh, int(eid), np.asarray(dump["node_ids"], dtype=int))
    velocity = np.asarray(dump["velocity"], dtype=float).reshape(3, 2)[local_order]
    velocity_prev = np.asarray(dump["velocity_prev"], dtype=float).reshape(3, 2)[local_order]
    acceleration = np.asarray(dump["acceleration"], dtype=float).reshape(3, 2)[local_order]
    pressure = np.asarray(dump["pressure"], dtype=float).reshape(3)[local_order]
    coords_ref = np.asarray(dump["coords_ref"], dtype=float).reshape(3, 2)[local_order]
    coords_cur = np.asarray(dump["coords_cur"], dtype=float).reshape(3, 2)[local_order]
    mesh_velocity = np.asarray(dump["mesh_velocity"], dtype=float).reshape(3, 2)[local_order]

    fluid["u_k"].components[0].set_nodal_values(ux_g, velocity[:, 0])
    fluid["u_k"].components[1].set_nodal_values(uy_g, velocity[:, 1])
    fluid["u_prev"].components[0].set_nodal_values(ux_g, velocity_prev[:, 0])
    fluid["u_prev"].components[1].set_nodal_values(uy_g, velocity_prev[:, 1])
    fluid["p_k"].set_nodal_values(p_g, pressure)

    disp = coords_cur - coords_ref
    if bool(zero_previous_history):
        acc_prev = np.zeros_like(acceleration)
        disp_prev = np.zeros_like(disp)
        mesh_vel_prev = np.zeros_like(mesh_velocity)
        mesh_acc_prev = np.zeros_like(mesh_velocity)
        disp_prev2 = np.zeros_like(disp)
    else:
        bossak = _bossak_coefficients(alpha=-0.3, dt=float(dt))
        # Kratos dumps the current-step ACCELERATION. The local operator,
        # however, expects the previous-step history field a_prev. Recover it
        # from the Bossak recurrence instead of reusing the current value.
        acc_prev = (
            np.asarray(acceleration, dtype=float)
            - float(bossak["ma0"]) * (np.asarray(velocity, dtype=float) - np.asarray(velocity_prev, dtype=float))
        ) / float(bossak["ma2"])
        disp_prev = disp - float(dt) * mesh_velocity
        mesh_vel_prev = mesh_velocity
        mesh_acc_prev = np.zeros_like(mesh_velocity)
        disp_prev2 = disp_prev

    fluid["a_prev"].components[0].set_nodal_values(ux_g, acc_prev[:, 0])
    fluid["a_prev"].components[1].set_nodal_values(uy_g, acc_prev[:, 1])
    fluid["d_mesh"].components[0].set_nodal_values(mx_g, disp[:, 0])
    fluid["d_mesh"].components[1].set_nodal_values(my_g, disp[:, 1])
    fluid["d_prev"].components[0].set_nodal_values(mx_g, disp_prev[:, 0])
    fluid["d_prev"].components[1].set_nodal_values(my_g, disp_prev[:, 1])
    fluid["d_prev2"].components[0].set_nodal_values(mx_g, disp_prev2[:, 0])
    fluid["d_prev2"].components[1].set_nodal_values(my_g, disp_prev2[:, 1])
    if "w_mesh_k" in fluid:
        fluid["w_mesh_k"].components[0].set_nodal_values(mx_g, mesh_velocity[:, 0])
        fluid["w_mesh_k"].components[1].set_nodal_values(my_g, mesh_velocity[:, 1])
    fluid["w_mesh_prev"].components[0].set_nodal_values(mx_g, mesh_vel_prev[:, 0])
    fluid["w_mesh_prev"].components[1].set_nodal_values(my_g, mesh_vel_prev[:, 1])
    fluid["a_mesh_prev"].components[0].set_nodal_values(mx_g, mesh_acc_prev[:, 0])
    fluid["a_mesh_prev"].components[1].set_nodal_values(my_g, mesh_acc_prev[:, 1])


def _set_dvms_state_from_dump(*, mesh, fluid: dict[str, object], eid: int, dump) -> None:
    state = fluid["dvms_state"]
    state.predicted_subscale_velocity.fill(0.0)
    state.old_subscale_velocity.fill(0.0)
    state.momentum_projection.fill(0.0)
    state.mass_projection.fill(0.0)
    state.old_mass_residual.fill(0.0)

    start = int(eid) * int(state.n_qp_per_element)
    stop = start + int(state.n_qp_per_element)
    private_predicted = None
    for key in ("predicted_subscale_velocity", "mPredictedSubscaleVelocity"):
        if key in dump:
            private_predicted = np.asarray(dump[key], dtype=float)
            break
    if private_predicted is not None and private_predicted.shape[0] == int(state.n_qp_per_element):
        state.predicted_subscale_velocity[start:stop, :] = np.asarray(private_predicted[:, :2], dtype=float)

    local_order = _dump_to_local_node_order(mesh, int(eid), np.asarray(dump["node_ids"], dtype=int))
    advproj = np.asarray(dump["advproj"], dtype=float).reshape(3, 2)[local_order]
    divproj = np.asarray(dump["divproj"], dtype=float).reshape(3)[local_order]
    geo = fluid["dh"].precompute_geometric_factors(int(state.quadrature_order), reuse=True)
    qp_ref = np.asarray(geo["qp_ref"], dtype=float)
    me = fluid["dh"].mixed_element
    phi_tab = np.asarray(
        [me.basis("ux", float(xi), float(eta))[me.slice("ux")] for xi, eta in qp_ref],
        dtype=float,
    )
    state.momentum_projection[start:stop, :] = phi_tab @ advproj
    state.mass_projection[start:stop] = phi_tab @ divproj
    state.sync_coefficients_from_samples()


def _seed_dvms_state_from_local_probe(*, fluid: dict[str, object], probe_path: Path) -> None:
    state = fluid["dvms_state"]
    with np.load(Path(probe_path).resolve(), allow_pickle=False) as data:
        mapping = {
            "old_subscale_velocity": "dvms_old_subscale_velocity",
            "predicted_subscale_velocity": "dvms_predicted_subscale_velocity",
            "momentum_projection": "dvms_momentum_projection",
            "mass_projection": "dvms_mass_projection",
            "old_mass_residual": "dvms_old_mass_residual",
        }
        for attr, key in mapping.items():
            if key not in data:
                continue
            target = np.asarray(getattr(state, attr), dtype=float)
            values = np.asarray(data[key], dtype=float)
            if values.shape != target.shape:
                raise ValueError(
                    f"Local probe key {key!r} has shape {values.shape}, expected {target.shape}."
                )
            getattr(state, attr)[:, ...] = values
    state.sync_coefficients_from_samples()


def _current_grouped_values(fluid: dict[str, object], eid: int) -> np.ndarray:
    ux_g = np.asarray(fluid["dh"].element_maps["ux"][eid], dtype=int)
    uy_g = np.asarray(fluid["dh"].element_maps["uy"][eid], dtype=int)
    p_g = np.asarray(fluid["dh"].element_maps["p"][eid], dtype=int)
    return np.concatenate(
        [
            np.asarray(fluid["u_k"].components[0].get_nodal_values(ux_g), dtype=float),
            np.asarray(fluid["u_k"].components[1].get_nodal_values(uy_g), dtype=float),
            np.asarray(fluid["p_k"].get_nodal_values(p_g), dtype=float),
        ]
    )


def _diff_stats(local: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    loc = np.asarray(local, dtype=float)
    ref = np.asarray(reference, dtype=float)
    diff = loc - ref
    loc_flat = loc.reshape(-1)
    ref_flat = ref.reshape(-1)
    denom = max(float(np.linalg.norm(ref_flat)), 1.0e-15)
    cosine_denom = float(np.linalg.norm(loc_flat) * np.linalg.norm(ref_flat))
    cosine = 1.0 if cosine_denom <= 1.0e-15 else float(np.dot(loc_flat, ref_flat) / cosine_denom)
    return {
        "local_inf_norm": float(np.max(np.abs(loc))) if loc.size else 0.0,
        "reference_inf_norm": float(np.max(np.abs(ref))) if ref.size else 0.0,
        "diff_inf_norm": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "diff_l2_norm": float(np.linalg.norm(diff.reshape(-1))),
        "rel_l2_norm": float(np.linalg.norm(diff.reshape(-1)) / denom),
        "cosine": cosine,
    }


def _load_extra_matrix_dump(kratos_dump: Path) -> dict[str, np.ndarray]:
    stem = str(kratos_dump.stem)
    candidates = [
        kratos_dump.with_name(f"{stem}_extra_matrices.npz"),
        kratos_dump.with_name(kratos_dump.stem.replace(".npz", "") + "_extra_matrices.npz"),
    ]
    for candidate in candidates:
        if candidate.exists():
            with np.load(candidate, allow_pickle=False) as data:
                return {str(name): np.asarray(data[name], dtype=float) for name in data.files}
    return {}


def _toggle_report(
    *,
    reference: np.ndarray,
    blocks: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    block_names = list(blocks.keys())
    total = np.zeros_like(reference, dtype=float)
    for name in block_names:
        total = total + np.asarray(blocks[name], dtype=float)
    report: dict[str, dict[str, float]] = {
        "full": _diff_stats(total, reference),
    }
    for name in block_names:
        subtotal = total - np.asarray(blocks[name], dtype=float)
        report[f"without_{name}"] = _diff_stats(subtotal, reference)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare termwise local DVMS blocks against the Kratos element dump.")
    parser.add_argument("--kratos-dump", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/NIRB/artifacts/compare_kratos_local_terms.json"),
    )
    parser.add_argument("--quadrature-order", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.008)
    parser.add_argument("--bossak-alpha", type=float, default=-0.3)
    parser.add_argument("--rho", type=float, default=1000.0)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--backend", choices=("python", "jit", "cpp"), default="cpp")
    parser.add_argument(
        "--local-probe-state",
        type=Path,
        default=None,
        help=(
            "Optional pycutfem debug_fluid_stage .npz file whose DVMS quadrature "
            "state should seed the local term assembly. This is useful for "
            "after-solve Kratos dumps because Kratos does not expose the private "
            "mPredictedSubscaleVelocity state."
        ),
    )
    parser.add_argument(
        "--zero-previous-history",
        action="store_true",
        help="Inject zero a_prev / d_prev / d_prev2 / mesh-history fields. Use this for first-step solved-state dumps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dump_path = Path(args.kratos_dump).resolve()
    dump = np.load(dump_path, allow_pickle=False)
    extra_dump = _load_extra_matrix_dump(dump_path)
    node_ids = np.asarray(dump["node_ids"], dtype=int)

    setup = load_example2_local_setup()
    mesh_f, _ = _load_reference_partitioned_meshes(setup=setup)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=int(args.quadrature_order))
    eid = _find_pycutfem_element_by_old_node_ids(mesh_f, node_ids)
    mu_value = (
        float(args.mu)
        if args.mu is not None
        else float(setup.material.density * setup.material.kinematic_viscosity)
    )

    _set_local_fields_from_dump(
        mesh=mesh_f,
        fluid=fluid,
        eid=eid,
        dump=dump,
        dt=float(args.dt),
        zero_previous_history=bool(args.zero_previous_history),
    )
    _set_dvms_state_from_dump(mesh=mesh_f, fluid=fluid, eid=eid, dump=dump)
    if args.local_probe_state is not None:
        _seed_dvms_state_from_local_probe(fluid=fluid, probe_path=Path(args.local_probe_state))

    du, dp, v, q = _local_trial_test_functions(fluid["dh"])
    compiler = FormCompiler(fluid["dh"], quadrature_order=int(args.quadrature_order), backend=str(args.backend))
    bossak = _bossak_coefficients(alpha=float(args.bossak_alpha), dt=float(args.dt))
    split = build_fluid_dvms_kratos_split_forms(
        du=du,
        dp=dp,
        v=v,
        q=q,
        u_k=fluid["u_k"],
        u_prev=fluid["u_prev"],
        a_prev=fluid["a_prev"],
        p_k=fluid["p_k"],
        d_mesh=fluid["d_mesh"],
        d_prev=fluid["d_prev"],
        mesh_v=fluid.get("w_mesh_k"),
        dx_measure=dx(metadata={"q": int(args.quadrature_order)}),
        rho=float(args.rho),
        mu=float(mu_value),
        dt=float(args.dt),
        h=Constant(
            float(
                _kratos_dvms_current_element_size_array(
                    mesh_f,
                    fluid["dh"],
                    fluid["d_mesh"],
                    element_ids=np.asarray([eid], dtype=int),
                )[0]
            )
        ),
        bossak_ma0=float(bossak["ma0"]),
        bossak_ma2=float(bossak["ma2"]),
        bossak_alpha=float(bossak["alpha"]),
        predicted_subscale=fluid["dvms_state"].coefficient("predicted_subscale_velocity"),
        old_subscale=fluid["dvms_state"].coefficient("old_subscale_velocity"),
        momentum_projection=fluid["dvms_state"].coefficient("momentum_projection"),
        mass_projection=fluid["dvms_state"].coefficient("mass_projection"),
        old_mass_residual=fluid["dvms_state"].coefficient("old_mass_residual"),
    )

    current_values = _current_grouped_values(fluid, eid)
    perm = _kratos_interleaved_to_local_grouped_perm(mesh_f, eid, node_ids)
    velocity_lhs_ref = np.asarray(dump["velocity_lhs"], dtype=float)[np.ix_(perm, perm)]
    velocity_rhs_ref = np.asarray(dump["velocity_rhs"], dtype=float)[perm]
    system_lhs_ref = np.asarray(dump["system_lhs"], dtype=float)[np.ix_(perm, perm)]
    system_rhs_ref = np.asarray(dump["system_rhs"], dtype=float)[perm]
    mass_matrix_ref = None
    if "CalculateMassMatrix" in extra_dump:
        mass_matrix_ref = np.asarray(extra_dump["CalculateMassMatrix"], dtype=float)[np.ix_(perm, perm)]
    bossak_mam = float((1.0 - float(args.bossak_alpha)) * float(bossak["ma0"]))
    reconstructed_system_lhs_ref = None
    if mass_matrix_ref is not None:
        reconstructed_system_lhs_ref = velocity_lhs_ref + bossak_mam * mass_matrix_ref

    lhs_report: dict[str, object] = {}
    rhs_report: dict[str, object] = {}
    lhs_total = np.zeros((9, 9), dtype=float)
    rhs_total = np.zeros((9,), dtype=float)
    lhs_blocks: dict[str, np.ndarray] = {}
    rhs_blocks: dict[str, np.ndarray] = {}

    ordered_lhs_terms = [
        "advective",
        "tau_advective",
        "v_grad_p_tau",
        "v_grad_p_dt",
        "pressure_galerkin",
        "q_div_tau",
        "continuity",
        "q_p_stabilization",
        "div_stabilization",
        "viscous",
    ]
    ordered_rhs_terms = [
        "body_old_subscale",
        "tau_velocity_source",
        "tau_q_source",
        "mass_projection",
        "old_mass_residual",
        "minus_advective_current",
        "minus_tau_advective_current",
        "minus_v_grad_p_tau_current",
        "minus_v_grad_p_dt_current",
        "minus_pressure_galerkin_current",
        "minus_q_div_tau_current",
        "minus_continuity_current",
        "minus_q_p_stabilization_current",
        "minus_div_stabilization_current",
        "viscous",
    ]

    for name in ordered_lhs_terms:
        lhs_term, _, _ = _assemble_single_term(compiler=compiler, form=split.lhs_terms[name], eid=eid, dh=fluid["dh"])
        lhs_term = np.asarray(lhs_term, dtype=float)
        lhs_total += lhs_term
        lhs_blocks[name] = lhs_term
        lhs_report[name] = _diff_stats(lhs_term, np.zeros_like(lhs_term))
    for name in ordered_rhs_terms:
        _, rhs_term, _ = _assemble_single_term(compiler=compiler, form=split.rhs_terms[name], eid=eid, dh=fluid["dh"])
        rhs_term = np.asarray(rhs_term, dtype=float)
        rhs_total += rhs_term
        rhs_blocks[name] = rhs_term
        rhs_report[name] = _diff_stats(rhs_term, np.zeros_like(rhs_term))

    mass_lhs_local, _, _ = _assemble_single_term(compiler=compiler, form=split.mass_terms["mass_lhs"], eid=eid, dh=fluid["dh"])
    mass_stab_local, _, _ = _assemble_single_term(compiler=compiler, form=split.mass_terms["mass_stabilization"], eid=eid, dh=fluid["dh"])
    mass_lhs_local = np.asarray(mass_lhs_local, dtype=float)
    mass_stab_local = np.asarray(mass_stab_local, dtype=float)
    system_jac_local, system_rhs_local, _ = _assemble_single_term(
        compiler=compiler,
        form=Equation(split.mass_terms["system_jacobian"], split.mass_terms["system_residual"]),
        eid=eid,
        dh=fluid["dh"],
    )
    system_jac_local = np.asarray(system_jac_local, dtype=float)
    system_rhs_local = np.asarray(system_rhs_local, dtype=float)
    system_condensed_jac_local, system_condensed_rhs_local, _ = _assemble_single_term(
        compiler=compiler,
        form=Equation(split.mass_terms["system_condensed_jacobian"], split.mass_terms["system_condensed_residual"]),
        eid=eid,
        dh=fluid["dh"],
    )
    system_condensed_jac_local = np.asarray(system_condensed_jac_local, dtype=float)
    system_condensed_rhs_local = np.asarray(system_condensed_rhs_local, dtype=float)
    mass_matrix_local = mass_lhs_local + mass_stab_local
    reconstructed_system_lhs_local = lhs_total + bossak_mam * mass_matrix_local
    condensed_correction_local = system_condensed_jac_local - system_jac_local

    report = {
        "kratos_dump": str(dump_path),
        "backend": str(args.backend),
        "quadrature_order": int(args.quadrature_order),
        "local_probe_state": None if args.local_probe_state is None else str(Path(args.local_probe_state).resolve()),
        "element_id_pycutfem": int(eid),
        "element_id_kratos": int(np.asarray(dump["element_id"]).reshape(())),
        "kratos_raw_calculate_local_system_is_zero": bool(
            float(np.max(np.abs(system_lhs_ref))) <= 1.0e-15 and float(np.max(np.abs(system_rhs_ref))) <= 1.0e-15
        ),
        "notes": [
            "On this Kratos build, CalculateLocalSystem() may return a zero matrix/vector for the probed DVMS element.",
            "When that happens, 'system_jacobian_vs_reconstructed_ref' is the meaningful Jacobian comparison.",
            "The reconstructed reference is velocity_lhs + mam * CalculateMassMatrix.",
        ],
        "state_summary": fluid["dvms_state"].summary(),
        "velocity_lhs_total": _diff_stats(lhs_total, velocity_lhs_ref),
        "velocity_rhs_total": _diff_stats(rhs_total, velocity_rhs_ref),
        "system_lhs_total": _diff_stats(system_jac_local, system_lhs_ref),
        "system_rhs_total": _diff_stats(system_rhs_local, system_rhs_ref),
        "system_condensed_lhs_total": _diff_stats(system_condensed_jac_local, system_lhs_ref),
        "system_condensed_rhs_total": _diff_stats(system_condensed_rhs_local, system_rhs_ref),
        "mass_remainder_ref": _diff_stats(system_lhs_ref - velocity_lhs_ref, np.zeros_like(system_lhs_ref)),
        "mass_lhs_local": _diff_stats(mass_lhs_local, np.zeros_like(mass_lhs_local)),
        "mass_stabilization_local": _diff_stats(mass_stab_local, np.zeros_like(mass_stab_local)),
        "velocity_lhs_terms": lhs_report,
        "velocity_rhs_terms": rhs_report,
        "current_values_grouped": current_values.tolist(),
        "projection_inf_norms": {
            "advproj_nodal_inf": float(np.max(np.abs(np.asarray(dump["advproj"], dtype=float)))),
            "divproj_nodal_inf": float(np.max(np.abs(np.asarray(dump["divproj"], dtype=float)))),
        },
        "element_size": {
            "reference_h": float(_kratos_dvms_element_size_coefficient(mesh_f).values[int(eid)]),
            "current_h": float(
                _kratos_dvms_current_element_size_array(
                    mesh_f,
                    fluid["dh"],
                    fluid["d_mesh"],
                    element_ids=np.asarray([eid], dtype=int),
                )[0]
            ),
        },
        "bossak_mam": float(bossak_mam),
    }
    if mass_matrix_ref is not None:
        report["mass_matrix_ref"] = _diff_stats(mass_matrix_local, mass_matrix_ref)
        report["reconstructed_system_lhs_ref"] = _diff_stats(reconstructed_system_lhs_local, reconstructed_system_lhs_ref)
        report["system_jacobian_vs_reconstructed_ref"] = _diff_stats(system_jac_local, reconstructed_system_lhs_ref)
        report["system_condensed_jacobian_vs_reconstructed_ref"] = _diff_stats(
            system_condensed_jac_local,
            reconstructed_system_lhs_ref,
        )
        report["condensed_correction_local"] = _diff_stats(condensed_correction_local, np.zeros_like(condensed_correction_local))
        system_blocks = {
            "velocity": np.asarray(lhs_total, dtype=float),
            "mass_galerkin": float(bossak_mam) * np.asarray(mass_lhs_local, dtype=float),
            "mass_stabilization": float(bossak_mam) * np.asarray(mass_stab_local, dtype=float),
            "condensed_correction": np.asarray(condensed_correction_local, dtype=float),
        }
        report["system_toggle_report_vs_reconstructed_ref"] = _toggle_report(
            reference=np.asarray(reconstructed_system_lhs_ref, dtype=float),
            blocks=system_blocks,
        )
        report["velocity_toggle_report_vs_reference"] = _toggle_report(
            reference=np.asarray(velocity_lhs_ref, dtype=float),
            blocks=lhs_blocks,
        )
        report["velocity_rhs_toggle_report_vs_reference"] = _toggle_report(
            reference=np.asarray(velocity_rhs_ref, dtype=float),
            blocks=rhs_blocks,
        )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
