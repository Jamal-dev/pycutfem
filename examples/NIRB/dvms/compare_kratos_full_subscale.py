from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.NIRB.example2_local_setup import load_example2_local_setup
from examples.NIRB.run_example2_local import _build_fluid_problem, _load_reference_partitioned_meshes
from examples.NIRB.dvms.helpers import _bossak_coefficients
from examples.NIRB.dvms.update import _update_fluid_dvms_predicted_subscale, _update_fluid_dvms_state_from_previous_step


def _coord_key(x: float, y: float, ndigits: int = 12) -> tuple[float, float]:
    return (round(float(x), ndigits), round(float(y), ndigits))


class PointLookup:
    def __init__(self, coords: np.ndarray, values: np.ndarray):
        self.coords = np.asarray(coords, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self._exact = {
            _coord_key(x, y): np.asarray(self.values[i], dtype=float).copy()
            for i, (x, y) in enumerate(self.coords)
        }

    def sample(self, target_coords: np.ndarray) -> np.ndarray:
        target = np.asarray(target_coords, dtype=float)
        out = np.empty((target.shape[0], self.values.shape[1]), dtype=float)
        for i, (x, y) in enumerate(target):
            hit = self._exact.get(_coord_key(x, y))
            if hit is not None:
                out[i, :] = hit
            else:
                dist2 = np.sum((self.coords - target[i][None, :]) ** 2, axis=1)
                out[i, :] = self.values[int(np.argmin(dist2)), :]
        return out


def _assign_vector_field_from_lookup(field, dh, field_names: tuple[str, str], lookup: PointLookup) -> None:
    dh._ensure_dof_coords()
    for comp_idx, field_name in enumerate(field_names):
        ids = np.asarray(dh.get_field_slice(field_name), dtype=int)
        coords = np.asarray(dh._dof_coords[ids], dtype=float)
        values = lookup.sample(coords)
        field.components[comp_idx].set_nodal_values(ids, values[:, comp_idx])


def _assign_scalar_field_from_lookup(field, dh, field_name: str, lookup: PointLookup) -> None:
    dh._ensure_dof_coords()
    ids = np.asarray(dh.get_field_slice(field_name), dtype=int)
    coords = np.asarray(dh._dof_coords[ids], dtype=float)
    values = lookup.sample(coords)
    field.set_nodal_values(ids, values[:, 0])


def _compare_fields(local_coords: np.ndarray, local_values: np.ndarray, ref_coords: np.ndarray, ref_values: np.ndarray) -> dict[str, float]:
    ref = np.asarray(ref_values, dtype=float)
    loc = PointLookup(local_coords, local_values).sample(ref_coords)
    diff = loc - ref
    ref_flat = ref.reshape(-1)
    loc_flat = loc.reshape(-1)
    denom = max(float(np.linalg.norm(ref_flat)), 1.0e-15)
    cosine = float(np.dot(loc_flat, ref_flat) / max(float(np.linalg.norm(loc_flat) * np.linalg.norm(ref_flat)), 1.0e-15))
    return {
        "rel_l2": float(np.linalg.norm(diff.reshape(-1)) / denom),
        "abs_rms": float(np.linalg.norm(diff.reshape(-1)) / np.sqrt(max(diff.size, 1))),
        "cosine": cosine,
        "reference_inf_norm": float(np.max(np.abs(ref))) if ref.size else 0.0,
        "local_inf_norm": float(np.max(np.abs(loc))) if loc.size else 0.0,
    }


def _load_kratos_subscale_samples(data) -> tuple[np.ndarray, np.ndarray]:
    if "q_coords_ref" in data and "subscale_velocity" in data:
        q_coords_ref = np.asarray(data["q_coords_ref"], dtype=float).reshape(-1, 2)
        q_subscale_velocity = np.asarray(data["subscale_velocity"], dtype=float).reshape(-1, 2)
        return q_coords_ref, q_subscale_velocity
    if "q_coords_ref_flat" in data and "subscale_velocity_flat" in data:
        q_coords_ref = np.asarray(data["q_coords_ref_flat"], dtype=float).reshape(-1, 2)
        q_subscale_velocity = np.asarray(data["subscale_velocity_flat"], dtype=float).reshape(-1, 2)
        return q_coords_ref, q_subscale_velocity
    raise KeyError("Kratos state dump does not contain quadrature subscale coordinates/values.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the local predicted DVMS subscale field against a Kratos full-field dump.")
    parser.add_argument("--kratos-state", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/NIRB/artifacts/compare_kratos_full_subscale.json"),
    )
    parser.add_argument("--backend", choices=("python", "jit", "cpp"), default="cpp")
    parser.add_argument("--quad-order", type=int, default=1)
    parser.add_argument("--bossak-alpha", type=float, default=-0.3)
    parser.add_argument("--dynamic-tau", type=float, default=1.0)
    parser.add_argument("--zero-predicted-subscale", action="store_true")
    parser.add_argument(
        "--allow-finalized",
        action="store_true",
        help="Allow comparing against a finalized Kratos dump. By default this is rejected because finalized DVMS dumps have already advanced the old-subscale history.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup = load_example2_local_setup()
    mesh_f, _ = _load_reference_partitioned_meshes(setup=setup)
    fluid = _build_fluid_problem(
        mesh_f,
        poly_order=1,
        pressure_order=1,
        quadrature_order=int(args.quad_order),
    )
    mu_f = float(setup.material.density * setup.material.kinematic_viscosity)
    dt = float(setup.boundaries.time_step)

    stage: str | None = None
    kratos_state_path = Path(args.kratos_state).resolve()
    with np.load(kratos_state_path) as data:
        if "stage" in data.files:
            stage = str(np.asarray(data["stage"]).reshape(()))
        node_coords = np.asarray(data["node_coords_ref"], dtype=float)
        velocity = np.asarray(data["velocity"], dtype=float)
        pressure = np.asarray(data["pressure"], dtype=float).reshape(-1, 1)
        acceleration = np.asarray(data["acceleration"], dtype=float) if "acceleration" in data.files else None
        mesh_displacement = np.asarray(data["mesh_displacement"], dtype=float) if "mesh_displacement" in data.files else None
        mesh_velocity = np.asarray(data["mesh_velocity"], dtype=float) if "mesh_velocity" in data.files else None
        q_coords_ref, q_subscale_velocity = _load_kratos_subscale_samples(data)
    if stage is None:
        sidecar = kratos_state_path.with_suffix(".json")
        if sidecar.exists():
            try:
                stage = str(json.loads(sidecar.read_text(encoding="utf-8")).get("stage") or "").strip() or None
            except Exception:
                stage = None
    if stage is not None and stage.lower() == "finalized" and not bool(args.allow_finalized):
        raise ValueError(
            "Kratos state dump stage='finalized' is not suitable for predicted-subscale comparison. "
            "Use a '--stage solved' or '--stage predicted' dump, or pass --allow-finalized to override."
        )

    vel_lookup = PointLookup(node_coords, velocity)
    p_lookup = PointLookup(node_coords, pressure)
    mesh_lookup = PointLookup(node_coords, mesh_displacement) if mesh_displacement is not None else None
    mesh_velocity_lookup = PointLookup(node_coords, mesh_velocity) if mesh_velocity is not None else None
    acceleration_lookup = PointLookup(node_coords, acceleration) if acceleration is not None else None
    _assign_vector_field_from_lookup(
        fluid["u_k"],
        fluid["dh"],
        (fluid["u_k"].components[0].field_name, fluid["u_k"].components[1].field_name),
        vel_lookup,
    )
    _assign_scalar_field_from_lookup(fluid["p_k"], fluid["dh"], fluid["p_k"].field_name, p_lookup)
    if mesh_lookup is not None:
        _assign_vector_field_from_lookup(
            fluid["d_mesh"],
            fluid["dh"],
            (fluid["d_mesh"].components[0].field_name, fluid["d_mesh"].components[1].field_name),
            mesh_lookup,
        )
    if mesh_lookup is not None and mesh_velocity_lookup is not None:
        prev_mesh_disp = PointLookup(
            node_coords,
            np.asarray(mesh_displacement, dtype=float) - float(dt) * np.asarray(mesh_velocity, dtype=float),
        )
        _assign_vector_field_from_lookup(
            fluid["d_prev"],
            fluid["dh"],
            (fluid["d_prev"].components[0].field_name, fluid["d_prev"].components[1].field_name),
            prev_mesh_disp,
        )
        _assign_vector_field_from_lookup(
            fluid["d_prev2"],
            fluid["dh"],
            (fluid["d_prev2"].components[0].field_name, fluid["d_prev2"].components[1].field_name),
            prev_mesh_disp,
        )
        _assign_vector_field_from_lookup(
            fluid["w_mesh_prev"],
            fluid["dh"],
            (fluid["w_mesh_prev"].components[0].field_name, fluid["w_mesh_prev"].components[1].field_name),
            mesh_velocity_lookup,
        )
    if acceleration_lookup is not None:
        bossak = _bossak_coefficients(alpha=float(args.bossak_alpha), dt=dt)
        a_prev_values = (
            np.asarray(acceleration, dtype=float) - float(bossak["ma0"]) * np.asarray(velocity, dtype=float)
        ) / float(bossak["ma2"])
        a_prev_lookup = PointLookup(node_coords, a_prev_values)
        _assign_vector_field_from_lookup(
            fluid["a_prev"],
            fluid["dh"],
            (fluid["a_prev"].components[0].field_name, fluid["a_prev"].components[1].field_name),
            a_prev_lookup,
        )

    _update_fluid_dvms_state_from_previous_step(
        state=fluid["dvms_state"],
        dh=fluid["dh"],
        mesh=mesh_f,
        u_prev=fluid["u_prev"],
        d_prev=fluid["d_prev"],
        d_geo=fluid["d_mesh"],
        backend=str(args.backend),
    )
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
        backend=str(args.backend),
    )
    if bool(args.zero_predicted_subscale):
        fluid["dvms_state"].predicted_subscale_velocity.fill(0.0)
        fluid["dvms_state"].sync_coefficients_from_samples()

    local_coords = np.asarray(fluid["dvms_state"].sample_coords, dtype=float)
    local_values = np.asarray(fluid["dvms_state"].predicted_subscale_velocity, dtype=float)
    report = {
        "backend": str(args.backend),
        "quadrature_order": int(args.quad_order),
        "kratos_stage": stage,
        "zero_predicted_subscale": bool(args.zero_predicted_subscale),
        "predicted_subscale_vs_kratos": _compare_fields(
            local_coords=local_coords,
            local_values=local_values,
            ref_coords=q_coords_ref,
            ref_values=q_subscale_velocity,
        ),
        "local_state_summary": fluid["dvms_state"].summary(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
