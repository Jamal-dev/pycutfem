from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from examples.NIRB.fluid_fom_operator import FluidFOMOperator
from examples.NIRB.fluid_snapshots import FluidStageRecord, FluidStageSnapshotBatch, FluidStageSnapshotWriter
from examples.NIRB.run_example2_local import (
    CoordinateLookup,
    _resample_lookup_to_coords,
    _restore_fluid_dvms_state,
    _transfer_scalar_field,
    _transfer_vector_field,
)


_PROBE_NAME_RE = re.compile(r"^step(?P<step>\d+)_iter(?P<iteration>\d+)_(?P<label>.+)\.npz$")


@dataclass(frozen=True)
class FluidStageProbePair:
    step: int
    coupling_iter: int
    pre_path: Path
    post_path: Path


@dataclass(frozen=True)
class FluidStageProbe:
    path: Path
    payload: dict[str, np.ndarray]

    @property
    def step(self) -> int:
        return int(np.asarray(self.payload["step"], dtype=int).reshape(-1)[0])

    @property
    def coupling_iter(self) -> int:
        return int(np.asarray(self.payload["coupling_iter"], dtype=int).reshape(-1)[0])

    @property
    def bc_scale(self) -> float:
        return float(np.asarray(self.payload.get("bc_scale", np.asarray(1.0)), dtype=float).reshape(-1)[0])

    def vector_lookup(self, coords_key: str, values_key: str, *, fallback_values_key: str | None = None) -> CoordinateLookup:
        key = str(values_key)
        if key not in self.payload and fallback_values_key is not None:
            key = str(fallback_values_key)
        return CoordinateLookup(
            np.asarray(self.payload[str(coords_key)], dtype=float),
            np.asarray(self.payload[key], dtype=float),
            dim=2,
        )

    def scalar_lookup(self, coords_key: str, values_key: str, *, fallback_values_key: str | None = None) -> CoordinateLookup:
        key = str(values_key)
        if key not in self.payload and fallback_values_key is not None:
            key = str(fallback_values_key)
        return CoordinateLookup(
            np.asarray(self.payload[str(coords_key)], dtype=float),
            np.asarray(self.payload[key], dtype=float),
            dim=1,
        )

    def reaction_lookup(self, *, kind: str = "point") -> CoordinateLookup | None:
        prefix = f"reaction_{str(kind)}"
        coords_key = f"{prefix}_coords"
        values_key = f"{prefix}_values"
        if coords_key not in self.payload or values_key not in self.payload:
            return None
        return CoordinateLookup(
            np.asarray(self.payload[coords_key], dtype=float),
            np.asarray(self.payload[values_key], dtype=float),
            dim=2,
        )


def load_fluid_stage_probe(path: str | Path) -> FluidStageProbe:
    source = Path(path).resolve()
    with np.load(source, allow_pickle=False) as data:
        payload = {str(key): np.asarray(data[key]) for key in data.files}
    return FluidStageProbe(path=source, payload=payload)


def find_fluid_stage_probe_pairs(
    root: str | Path,
    *,
    final_only: bool = True,
    steps: Iterable[int] | None = None,
) -> list[FluidStageProbePair]:
    source = Path(root).resolve()
    probe_dir = source / "debug_fluid_stage" if (source / "debug_fluid_stage").is_dir() else source
    requested_steps = None if steps is None else {int(step) for step in steps}
    stages: dict[tuple[int, int, str], Path] = {}
    for path in sorted(probe_dir.glob("step*_iter*_*.npz")):
        match = _PROBE_NAME_RE.match(path.name)
        if match is None:
            continue
        step = int(match.group("step"))
        if requested_steps is not None and step not in requested_steps:
            continue
        iteration = int(match.group("iteration"))
        label = str(match.group("label"))
        stages[(step, iteration, label)] = path.resolve()

    pairs: list[FluidStageProbePair] = []
    step_iters = sorted({(step, iteration) for step, iteration, _label in stages.keys()})
    for step, iteration in step_iters:
        pre = stages.get((step, iteration, "pre_fluid_solve"))
        post = stages.get((step, iteration, "post_fluid_solve"))
        if pre is None or post is None:
            continue
        pairs.append(FluidStageProbePair(step=step, coupling_iter=iteration, pre_path=pre, post_path=post))

    if not final_only:
        return pairs
    latest: dict[int, FluidStageProbePair] = {}
    for pair in pairs:
        old = latest.get(int(pair.step))
        if old is None or int(pair.coupling_iter) > int(old.coupling_iter):
            latest[int(pair.step)] = pair
    return [latest[step] for step in sorted(latest)]


def _dvms_snapshot_from_probe(probe: FluidStageProbe) -> dict[str, np.ndarray]:
    payload = probe.payload
    mapping = {
        "dvms_old_subscale_velocity": "old_subscale_velocity",
        "dvms_predicted_subscale_velocity": "predicted_subscale_velocity",
        "dvms_momentum_projection": "momentum_projection",
        "dvms_mass_projection": "mass_projection",
        "dvms_old_mass_residual": "old_mass_residual",
        "dvms_nodal_momentum_projection": "nodal_momentum_projection",
        "dvms_nodal_div_projection": "nodal_div_projection",
        "dvms_prev_nodal_div_projection": "prev_nodal_div_projection",
    }
    return {target: np.asarray(payload[source], dtype=float) for source, target in mapping.items() if source in payload}


def configure_fluid_stage_probe_bcs(
    operator: FluidFOMOperator,
    probe: FluidStageProbe,
    *,
    fluid_iface_coords: np.ndarray,
    inlet_lookup: Callable[[float, float], float],
    apply_to_state: bool = True,
) -> None:
    mesh_velocity = _resample_lookup_to_coords(
        probe.vector_lookup("d_coords", "w_mesh_values"),
        np.asarray(fluid_iface_coords, dtype=float),
    )
    scale = float(probe.bc_scale)

    def scaled_inlet(x: float, y: float) -> float:
        return scale * float(inlet_lookup(float(x), float(y)))

    operator.configure_boundary_conditions(
        iface_velocity=mesh_velocity,
        inlet_lookup=scaled_inlet,
        apply_to_state=bool(apply_to_state),
    )


def restore_fluid_stage_probe(
    operator: FluidFOMOperator,
    probe: FluidStageProbe,
    *,
    fluid_iface_coords: np.ndarray,
    inlet_lookup: Callable[[float, float], float],
    apply_bcs_to_state: bool = True,
) -> None:
    dh = operator.dh
    prob = operator.prob
    _transfer_vector_field(target_dh=dh, target_vec=prob["u_k"], source_lookup=probe.vector_lookup("u_coords", "u_values"))
    _transfer_scalar_field(target_dh=dh, target_fun=prob["p_k"], source_lookup=probe.scalar_lookup("p_coords", "p_values"))
    _transfer_vector_field(
        target_dh=dh,
        target_vec=prob["u_prev"],
        source_lookup=probe.vector_lookup("u_coords", "u_prev_values", fallback_values_key="u_values"),
    )
    _transfer_scalar_field(
        target_dh=dh,
        target_fun=prob["p_prev"],
        source_lookup=probe.scalar_lookup("p_coords", "p_prev_values", fallback_values_key="p_values"),
    )
    _transfer_vector_field(
        target_dh=dh,
        target_vec=prob["a_k"],
        source_lookup=probe.vector_lookup("u_coords", "a_k_values", fallback_values_key="a_prev_values"),
    )
    _transfer_vector_field(
        target_dh=dh,
        target_vec=prob["a_prev"],
        source_lookup=probe.vector_lookup("u_coords", "a_prev_values", fallback_values_key="a_k_values"),
    )
    for target_name, value_key in (
        ("d_mesh", "d_values"),
        ("d_prev", "d_prev_values"),
        ("d_prev2", "d_prev2_values"),
        ("w_mesh_k", "w_mesh_values"),
        ("w_mesh_prev", "w_mesh_prev_values"),
        ("a_mesh_k", "a_mesh_values"),
        ("a_mesh_prev", "a_mesh_prev_values"),
    ):
        field = prob.get(target_name)
        if field is not None and value_key in probe.payload:
            _transfer_vector_field(
                target_dh=dh,
                target_vec=field,
                source_lookup=probe.vector_lookup("d_coords", value_key),
            )
    _restore_fluid_dvms_state(prob.get("dvms_state"), _dvms_snapshot_from_probe(probe))
    configure_fluid_stage_probe_bcs(
        operator,
        probe,
        fluid_iface_coords=fluid_iface_coords,
        inlet_lookup=inlet_lookup,
        apply_to_state=bool(apply_bcs_to_state),
    )


def capture_record_from_fluid_stage_probe(
    operator: FluidFOMOperator,
    probe: FluidStageProbe,
    *,
    fluid_iface_coords: np.ndarray,
    inlet_lookup: Callable[[float, float], float],
    metadata: dict[str, Any] | None = None,
) -> FluidStageRecord:
    restore_fluid_stage_probe(
        operator,
        probe,
        fluid_iface_coords=fluid_iface_coords,
        inlet_lookup=inlet_lookup,
        apply_bcs_to_state=True,
    )
    writer = FluidStageSnapshotWriter()
    record_metadata = {
        "probe_path": str(probe.path),
        "step": int(probe.step),
        "coupling_iter": int(probe.coupling_iter),
        "bc_scale": float(probe.bc_scale),
    }
    if metadata:
        record_metadata.update(dict(metadata))
    return writer.append_from_operator(
        operator,
        reaction_loads=probe.reaction_lookup(kind="point"),
        include_reaction=False,
        metadata=record_metadata,
    )


def build_stage_probe_batch(
    operator: FluidFOMOperator,
    probes: Iterable[FluidStageProbe],
    *,
    fluid_iface_coords: np.ndarray,
    inlet_lookup_factory: Callable[[FluidStageProbe], Callable[[float, float], float]],
) -> FluidStageSnapshotBatch:
    writer = FluidStageSnapshotWriter()
    for probe in probes:
        record = capture_record_from_fluid_stage_probe(
            operator,
            probe,
            fluid_iface_coords=fluid_iface_coords,
            inlet_lookup=inlet_lookup_factory(probe),
        )
        writer.append(record)
    return writer.to_batch()


__all__ = [
    "FluidStageProbe",
    "FluidStageProbePair",
    "build_stage_probe_batch",
    "capture_record_from_fluid_stage_probe",
    "configure_fluid_stage_probe_bcs",
    "find_fluid_stage_probe_pairs",
    "load_fluid_stage_probe",
    "restore_fluid_stage_probe",
]
