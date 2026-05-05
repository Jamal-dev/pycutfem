#!/usr/bin/env python3
"""Partitioned free-fluid/porous-flap FPSI example on the NIRB geometry.

This driver is a lightweight convergence gate for replacing the solid flap in
the NIRB double-flap setup by a porous U-Pl block.  It intentionally lives at
the example level: the free fluid is represented by a deterministic reduced
interface response, while the porous flap response uses a small U-Pl interface
block assembled from material parameters.  The fixed-point loop is the same
style used by the NIRB runner: the interface iterate is updated by the shared
coupling accelerators, including IQN-ILS/IQLN with the optional C++ backend.

The purpose of this file is to keep the FPSI coupling signs, unknown ordering,
pressure/flux feedback, and convergence behavior executable while the full
DVMS fluid and mesh-motion path is being wired to the poromechanics block.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.NIRB.example2_problem import DoubleFlapGeometry, load_geometry  # noqa: E402
from examples.utils.poromechanics import UPlMaterial2D  # noqa: E402
from pycutfem.mor.snapshots import SnapshotBatch  # noqa: E402
from pycutfem.solvers import CouplingAccelerator, create_coupling_accelerator  # noqa: E402


def nominal_double_flap_geometry() -> DoubleFlapGeometry:
    """Return the deterministic DoubleFlap geometry used by local NIRB tests."""

    return DoubleFlapGeometry(
        channel_length=2.5,
        channel_height=0.492,
        cylinder_center=(0.2, 0.2),
        cylinder_radius=0.05,
        solid_x0=1.2,
        solid_x1=1.52,
        solid_y0=0.0,
        solid_y1=0.28,
        base_height=0.06,
        arm_width=0.06,
        inlet_ramp_end_time=1.0,
    )


def load_porous_flap_geometry(
    reference_root: str | Path | None = None,
    *,
    use_reference: bool = False,
) -> DoubleFlapGeometry:
    """Load the NIRB DoubleFlap geometry or return the checked-in nominal one.

    Passing ``use_reference=True`` or a ``reference_root`` requires the Kratos
    DoubleFlap reference files to exist and lets ``load_geometry`` raise a clear
    error if they do not.  The default nominal geometry keeps the example and
    tests self-contained.
    """

    if use_reference or reference_root is not None:
        return load_geometry(reference_root)
    return nominal_double_flap_geometry()


def default_porous_flap_material() -> UPlMaterial2D:
    """Return a compressible U-Pl material for the porous flap example."""

    return UPlMaterial2D(
        young_modulus=5.0e5,
        poisson_ratio=0.35,
        porosity=0.32,
        biot_coefficient=0.8,
        permeability_xx=1.0e-8,
        permeability_yy=4.0e-8,
        dynamic_viscosity_liquid=1.0e-3,
        storage_inverse=2.0e-4,
        density_solid=1000.0,
        density_liquid=1000.0,
    )


@dataclass(frozen=True)
class InterfaceStationSet:
    """Midpoint stations on the exposed porous-flap boundary."""

    points: np.ndarray
    normals: np.ndarray
    tangents: np.ndarray
    weights: np.ndarray
    segment_ids: np.ndarray
    station_coordinates: np.ndarray
    segment_names: tuple[str, ...]

    @property
    def size(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True)
class FreeFluidInterfaceResponse:
    """Interface loads returned by the reduced free-fluid side."""

    traction: np.ndarray
    normal_flux: np.ndarray
    pressure: np.ndarray
    inlet_speed: float


@dataclass(frozen=True)
class CoupledIterationRecord:
    """One fixed-point iteration record."""

    step: int
    iteration: int
    time: float
    residual_norm: float
    absolute_residual_norm: float
    update_norm: float
    relaxation: float
    used_history: bool
    method: str


@dataclass(frozen=True)
class CoupledStepRecord:
    """Accepted time-step record for the porous-flap FPSI example."""

    step: int
    time: float
    converged: bool
    iterations: int
    residual_norm: float
    max_displacement: float
    max_pressure: float
    mean_pressure: float
    inlet_speed: float


@dataclass(frozen=True)
class FPSISnapshotMetadata:
    """Metadata for one FPSI snapshot column."""

    step: int
    time_s: float
    coupling_iter: int
    converged: bool
    residual_norm: float


@dataclass(frozen=True)
class PorousFlapFPSIResult:
    """Full time history of the reduced porous-flap FPSI example."""

    geometry: dict[str, float | tuple[float, float] | str | None]
    material: dict[str, float | None]
    fluid_parameters: dict[str, float]
    porous_parameters: dict[str, float]
    station_count: int
    accelerator: str
    accelerator_backend: str
    dt: float
    tolerance: float
    station_points: np.ndarray
    station_normals: np.ndarray
    station_tangents: np.ndarray
    station_weights: np.ndarray
    station_segment_ids: np.ndarray
    station_coordinates: np.ndarray
    station_segment_names: tuple[str, ...]
    steps: tuple[CoupledStepRecord, ...]
    iterations: tuple[CoupledIterationRecord, ...]
    snapshot_metadata: tuple[FPSISnapshotMetadata, ...]
    load_guess_data: np.ndarray
    load_data: np.ndarray
    disp_data: np.ndarray
    pressure_data: np.ndarray
    porous_state_data: np.ndarray
    state_guess_data: np.ndarray
    interface_disp_data: np.ndarray
    interface_pressure_data: np.ndarray
    interface_flux_data: np.ndarray
    interface_traction_data: np.ndarray
    final_state: np.ndarray

    @property
    def converged(self) -> bool:
        return all(step.converged for step in self.steps)

    @property
    def max_iterations(self) -> int:
        return max((step.iterations for step in self.steps), default=0)

    @property
    def n_snapshots(self) -> int:
        return int(self.porous_state_data.shape[1])


@dataclass(frozen=True)
class ReducedFluidParameters:
    """Reduced free-fluid interface parameters."""

    density: float = 1000.0
    dynamic_viscosity: float = 1.0e-3
    reference_velocity: float = 2.5
    shedding_amplitude: float = 0.08
    shedding_frequency: float = 2.0
    pressure_feedback: float = 0.25
    normal_damping: float = 2.0e4
    tangential_damping: float = 1.0e4
    flux_coefficient: float = 3.0e-3
    dilation_flux_coefficient: float = 1.0e-2


@dataclass(frozen=True)
class ReducedPorousParameters:
    """Reduced U-Pl interface block parameters."""

    normal_stiffness_scale: float = 1.0
    tangential_stiffness_scale: float = 0.35
    displacement_smoothing_scale: float = 2.0e-3
    pressure_diffusion_scale: float = 2.0e-2
    pressure_leak: float = 5.0e-4


def _line_stations(
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    normal: tuple[float, float],
    name: str,
    target_spacing: float,
    segment_id: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float], list[int], list[float], str]:
    start_v = np.asarray(start, dtype=float)
    end_v = np.asarray(end, dtype=float)
    normal_v = np.asarray(normal, dtype=float)
    length = float(np.linalg.norm(end_v - start_v))
    if length <= 0.0:
        raise ValueError(f"Interface segment '{name}' has zero length.")
    normal_norm = float(np.linalg.norm(normal_v))
    if normal_norm <= 0.0:
        raise ValueError(f"Interface segment '{name}' has a zero normal.")
    normal_v = normal_v / normal_norm
    tangent_v = (end_v - start_v) / length
    n_stations = max(2, int(math.ceil(length / max(float(target_spacing), 1.0e-12))))

    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    tangents: list[np.ndarray] = []
    weights: list[float] = []
    segment_ids: list[int] = []
    station_coordinates: list[float] = []
    for i in range(n_stations):
        s = (float(i) + 0.5) / float(n_stations)
        points.append(start_v + s * (end_v - start_v))
        normals.append(normal_v.copy())
        tangents.append(tangent_v.copy())
        weights.append(length / float(n_stations))
        segment_ids.append(int(segment_id))
        station_coordinates.append(s * length)
    return points, normals, tangents, weights, segment_ids, station_coordinates, str(name)


def build_porous_flap_interface_stations(
    geometry: DoubleFlapGeometry,
    *,
    target_spacing: float = 0.025,
) -> InterfaceStationSet:
    """Sample the exposed NIRB flap boundary used for FPSI coupling."""

    y0 = float(geometry.solid_y0)
    y_base = float(geometry.solid_y0 + geometry.base_height)
    y1 = float(geometry.solid_y1)
    x0 = float(geometry.solid_x0)
    x1 = float(geometry.solid_x1)
    xl = float(geometry.left_arm_x1)
    xr = float(geometry.right_arm_x0)

    raw_segments = [
        ((x0, y0), (x0, y1), (-1.0, 0.0), "left_outer"),
        ((xl, y_base), (xl, y1), (1.0, 0.0), "left_inner"),
        ((x0, y1), (xl, y1), (0.0, 1.0), "left_top"),
        ((xr, y_base), (xr, y1), (-1.0, 0.0), "right_inner"),
        ((x1, y0), (x1, y1), (1.0, 0.0), "right_outer"),
        ((xr, y1), (x1, y1), (0.0, 1.0), "right_top"),
        ((xl, y_base), (xr, y_base), (0.0, 1.0), "base_gap_floor"),
    ]

    all_points: list[np.ndarray] = []
    all_normals: list[np.ndarray] = []
    all_tangents: list[np.ndarray] = []
    all_weights: list[float] = []
    all_segment_ids: list[int] = []
    all_station_coordinates: list[float] = []
    names: list[str] = []
    for segment_id, (start, end, normal, name) in enumerate(raw_segments):
        points, normals, tangents, weights, segment_ids, station_coordinates, seg_name = _line_stations(
            start=start,
            end=end,
            normal=normal,
            name=name,
            target_spacing=target_spacing,
            segment_id=segment_id,
        )
        all_points.extend(points)
        all_normals.extend(normals)
        all_tangents.extend(tangents)
        all_weights.extend(weights)
        all_segment_ids.extend(segment_ids)
        all_station_coordinates.extend(station_coordinates)
        names.append(seg_name)

    return InterfaceStationSet(
        points=np.asarray(all_points, dtype=float),
        normals=np.asarray(all_normals, dtype=float),
        tangents=np.asarray(all_tangents, dtype=float),
        weights=np.asarray(all_weights, dtype=float),
        segment_ids=np.asarray(all_segment_ids, dtype=int),
        station_coordinates=np.asarray(all_station_coordinates, dtype=float),
        segment_names=tuple(names),
    )


class ReducedFreeFluidBlock:
    """Deterministic free-fluid interface block for the porous-flap gate."""

    def __init__(
        self,
        *,
        geometry: DoubleFlapGeometry,
        stations: InterfaceStationSet,
        parameters: ReducedFluidParameters,
    ) -> None:
        self.geometry = geometry
        self.stations = stations
        self.parameters = parameters

    def evaluate(self, x: np.ndarray, *, time: float, dt: float) -> FreeFluidInterfaceResponse:
        stations = self.stations
        params = self.parameters
        state = np.asarray(x, dtype=float).reshape(stations.size, 3)
        displacement = state[:, :2]
        pressure = state[:, 2]
        normals = stations.normals
        tangents = stations.tangents
        points = stations.points

        center_y = 0.5 * float(self.geometry.channel_height)
        inlet_speed = float(
            self.geometry.inlet_velocity(center_y, float(time), reference_velocity=float(params.reference_velocity))
        )
        dynamic_pressure = 0.5 * float(params.density) * inlet_speed * inlet_speed

        x_fraction = np.clip(points[:, 0] / max(float(self.geometry.channel_length), 1.0e-12), 0.0, 1.0)
        y_fraction = np.clip(points[:, 1] / max(float(self.geometry.channel_height), 1.0e-12), 0.0, 1.0)
        shape = (1.0 - 0.35 * x_fraction) * (0.75 + 0.25 * np.sin(math.pi * y_fraction))
        shedding = 1.0 + float(params.shedding_amplitude) * np.sin(
            2.0 * math.pi * float(params.shedding_frequency) * float(time) + 2.0 * math.pi * x_fraction
        )
        fluid_pressure = dynamic_pressure * shape * shedding

        u_n = np.einsum("ij,ij->i", displacement, normals, optimize=True)
        u_t = np.einsum("ij,ij->i", displacement, tangents, optimize=True)

        normal_load = -fluid_pressure - float(params.normal_damping) * u_n + float(params.pressure_feedback) * pressure
        tangent_load = -float(params.tangential_damping) * u_t
        traction = normal_load[:, None] * normals + tangent_load[:, None] * tangents

        flux = float(params.flux_coefficient) * (fluid_pressure - pressure)
        flux -= float(params.dilation_flux_coefficient) * u_n / max(float(dt), 1.0e-12)

        return FreeFluidInterfaceResponse(
            traction=np.asarray(traction, dtype=float),
            normal_flux=np.asarray(flux, dtype=float),
            pressure=np.asarray(fluid_pressure, dtype=float),
            inlet_speed=inlet_speed,
        )


class ReducedPorousFlapBlock:
    """Small U-Pl interface response for the porous flap."""

    def __init__(
        self,
        *,
        geometry: DoubleFlapGeometry,
        stations: InterfaceStationSet,
        material: UPlMaterial2D,
        dt: float,
        parameters: ReducedPorousParameters,
    ) -> None:
        self.geometry = geometry
        self.stations = stations
        self.material = material
        self.dt = float(dt)
        self.parameters = parameters
        self._matrix = self._build_matrix()

    def _add_pair_laplacian(self, matrix: np.ndarray, a: int, b: int, coeff: float, *, block_size: int) -> None:
        coeff = float(coeff)
        if coeff == 0.0:
            return
        for component in range(block_size):
            ia = block_size * int(a) + component
            ib = block_size * int(b) + component
            matrix[ia, ia] += coeff
            matrix[ib, ib] += coeff
            matrix[ia, ib] -= coeff
            matrix[ib, ia] -= coeff

    def _build_matrix(self) -> np.ndarray:
        stations = self.stations
        n = stations.size
        dim_u = 2 * n
        total = 3 * n
        matrix = np.zeros((total, total), dtype=float)

        E = float(self.material.young_modulus)
        height = max(float(self.geometry.solid_height), 1.0e-12)
        base_k = E / height
        k_n = float(self.parameters.normal_stiffness_scale) * base_k
        k_t = float(self.parameters.tangential_stiffness_scale) * base_k
        storage_rate = float(self.material.biot_modulus_inverse) / max(self.dt, 1.0e-12)
        pressure_leak = float(self.parameters.pressure_leak)
        conductivity = float(np.trace(self.material.darcy_conductivity_matrix) / 2.0)
        pressure_diffusion = float(self.parameters.pressure_diffusion_scale) * max(conductivity, 1.0e-14)

        cup = np.zeros((dim_u, n), dtype=float)
        cdt = np.zeros((n, dim_u), dtype=float)
        alpha = float(self.material.biot_coefficient)

        for i in range(n):
            w = float(stations.weights[i])
            normal = stations.normals[i]
            tangent = stations.tangents[i]
            y = float(stations.points[i, 1])
            clamp_factor = 1.0 + 2.0 * math.exp(-(y - float(self.geometry.solid_y0)) / max(height, 1.0e-12))
            local = w * clamp_factor * (k_n * np.outer(normal, normal) + k_t * np.outer(tangent, tangent))
            row = slice(2 * i, 2 * i + 2)
            matrix[row, row] += local

            cup[row, i] = alpha * w * normal
            cdt[i, row] = alpha * w * normal / max(self.dt, 1.0e-12)
            matrix[dim_u + i, dim_u + i] += w * (storage_rate + pressure_leak)

        for segment_id in np.unique(stations.segment_ids):
            ids = np.flatnonzero(stations.segment_ids == int(segment_id))
            ids = ids[np.argsort(stations.station_coordinates[ids])]
            for a, b in zip(ids[:-1], ids[1:]):
                ds = float(np.linalg.norm(stations.points[int(b)] - stations.points[int(a)]))
                avg_w = 0.5 * (float(stations.weights[int(a)]) + float(stations.weights[int(b)]))
                disp_coeff = float(self.parameters.displacement_smoothing_scale) * E * avg_w / max(ds, 1.0e-12)
                pressure_coeff = pressure_diffusion * avg_w / max(ds, 1.0e-12)
                self._add_pair_laplacian(matrix[:dim_u, :dim_u], int(a), int(b), disp_coeff, block_size=2)
                self._add_pair_laplacian(
                    matrix[dim_u:, dim_u:],
                    int(a),
                    int(b),
                    pressure_coeff,
                    block_size=1,
                )

        matrix[:dim_u, dim_u:] -= cup
        matrix[dim_u:, :dim_u] += cdt

        if not np.all(np.isfinite(matrix)):
            raise FloatingPointError("Reduced porous-flap matrix contains non-finite entries.")
        cond = float(np.linalg.cond(matrix))
        if not np.isfinite(cond) or cond > 1.0e14:
            raise np.linalg.LinAlgError(f"Reduced porous-flap matrix is ill-conditioned: cond={cond:.3e}")
        return matrix

    def solve(self, *, fluid: FreeFluidInterfaceResponse, previous_state: np.ndarray) -> np.ndarray:
        stations = self.stations
        n = stations.size
        dim_u = 2 * n
        previous = np.asarray(previous_state, dtype=float).reshape(n, 3)
        rhs = np.zeros((3 * n,), dtype=float)
        rhs[:dim_u] = (np.asarray(fluid.traction, dtype=float) * stations.weights[:, None]).reshape(-1)

        prev_u = previous[:, :2]
        prev_p = previous[:, 2]
        prev_u_n = np.einsum("ij,ij->i", prev_u, stations.normals, optimize=True)
        storage_rate = float(self.material.biot_modulus_inverse) / max(self.dt, 1.0e-12)
        alpha_rate = float(self.material.biot_coefficient) / max(self.dt, 1.0e-12)
        rhs[dim_u:] = stations.weights * (
            storage_rate * prev_p
            + alpha_rate * prev_u_n
            + np.asarray(fluid.normal_flux, dtype=float)
        )

        solution = np.linalg.solve(self._matrix, rhs)
        if not np.all(np.isfinite(solution)):
            raise FloatingPointError("Reduced porous-flap solution contains non-finite entries.")
        return np.column_stack((solution[:dim_u].reshape(n, 2), solution[dim_u:]))


def _relative_norm(residual: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    abs_norm = float(np.linalg.norm(np.asarray(residual, dtype=float).reshape(-1), ord=np.inf))
    scale = 1.0 + float(np.linalg.norm(np.asarray(reference, dtype=float).reshape(-1), ord=np.inf))
    return abs_norm / scale, abs_norm


def _fpsi_load_vector(fluid: FreeFluidInterfaceResponse) -> np.ndarray:
    """Return the per-station FPSI load vector ``[t_x, t_y, q_n]``."""

    return np.column_stack(
        (
            np.asarray(fluid.traction, dtype=float)[:, 0],
            np.asarray(fluid.traction, dtype=float)[:, 1],
            np.asarray(fluid.normal_flux, dtype=float),
        )
    ).reshape(-1)


def _stack_snapshot_columns(columns: list[np.ndarray], *, rows: int) -> np.ndarray:
    if columns:
        return np.column_stack([np.asarray(column, dtype=float).reshape(-1) for column in columns])
    return np.zeros((int(rows), 0), dtype=float)


def _trace_lines_from_segments(segment_ids: np.ndarray, station_coordinates: np.ndarray) -> list[tuple[int, int]]:
    lines: list[tuple[int, int]] = []
    segment_ids = np.asarray(segment_ids, dtype=int).reshape(-1)
    station_coordinates = np.asarray(station_coordinates, dtype=float).reshape(-1)
    for segment_id in np.unique(segment_ids):
        ids = np.flatnonzero(segment_ids == int(segment_id))
        ids = ids[np.argsort(station_coordinates[ids])]
        for a, b in zip(ids[:-1], ids[1:]):
            lines.append((int(a), int(b)))
    return lines


def _write_legacy_trace_vtk(
    path: str | Path,
    *,
    points: np.ndarray,
    displacement: np.ndarray,
    pressure: np.ndarray,
    traction: np.ndarray,
    normal_flux: np.ndarray,
    normals: np.ndarray,
    segment_ids: np.ndarray,
    lines: list[tuple[int, int]],
    displacement_scale: float,
    title: str,
) -> None:
    """Write one ASCII VTK PolyData trace file for the porous-flap interface."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    displacement = np.asarray(displacement, dtype=float).reshape(points.shape[0], 2)
    pressure = np.asarray(pressure, dtype=float).reshape(points.shape[0])
    traction = np.asarray(traction, dtype=float).reshape(points.shape[0], 2)
    normal_flux = np.asarray(normal_flux, dtype=float).reshape(points.shape[0])
    normals = np.asarray(normals, dtype=float).reshape(points.shape[0], 2)
    segment_ids = np.asarray(segment_ids, dtype=int).reshape(points.shape[0])
    vtk_points = points + float(displacement_scale) * displacement

    with target.open("w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write(f"{title}\n")
        handle.write("ASCII\n")
        handle.write("DATASET POLYDATA\n")
        handle.write(f"POINTS {points.shape[0]} float\n")
        for x, y in vtk_points:
            handle.write(f"{float(x):.17e} {float(y):.17e} 0.00000000000000000e+00\n")

        handle.write(f"LINES {len(lines)} {3 * len(lines)}\n")
        for a, b in lines:
            handle.write(f"2 {int(a)} {int(b)}\n")

        handle.write(f"POINT_DATA {points.shape[0]}\n")
        handle.write("VECTORS displacement float\n")
        for ux, uy in displacement:
            handle.write(f"{float(ux):.17e} {float(uy):.17e} 0.00000000000000000e+00\n")
        handle.write("VECTORS traction float\n")
        for tx, ty in traction:
            handle.write(f"{float(tx):.17e} {float(ty):.17e} 0.00000000000000000e+00\n")
        handle.write("VECTORS normal float\n")
        for nx, ny in normals:
            handle.write(f"{float(nx):.17e} {float(ny):.17e} 0.00000000000000000e+00\n")

        scalar_fields = {
            "pore_pressure": pressure,
            "normal_flux": normal_flux,
            "segment_id": segment_ids,
        }
        for name, values in scalar_fields.items():
            dtype = "int" if np.issubdtype(np.asarray(values).dtype, np.integer) else "float"
            handle.write(f"SCALARS {name} {dtype} 1\n")
            handle.write("LOOKUP_TABLE default\n")
            for value in np.asarray(values).reshape(-1):
                if dtype == "int":
                    handle.write(f"{int(value)}\n")
                else:
                    handle.write(f"{float(value):.17e}\n")


def _geometry_from_result(result: PorousFlapFPSIResult) -> DoubleFlapGeometry:
    data = dict(result.geometry)
    if isinstance(data.get("cylinder_center"), list):
        data["cylinder_center"] = tuple(float(v) for v in data["cylinder_center"])
    return DoubleFlapGeometry(**data)


def _result_fluid_parameters(result: PorousFlapFPSIResult) -> ReducedFluidParameters:
    values = dict(getattr(result, "fluid_parameters", {}) or {})
    return ReducedFluidParameters(**values)


def _build_full_sampling_grid(
    geometry: DoubleFlapGeometry,
    *,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if nx <= 0 or ny <= 0:
        raise ValueError("VTK sampling grid dimensions must be positive.")
    xs = np.linspace(0.0, float(geometry.channel_length), int(nx) + 1)
    ys = np.linspace(0.0, float(geometry.channel_height), int(ny) + 1)
    points = np.asarray([(float(x), float(y)) for y in ys for x in xs], dtype=float)
    point_phase = np.zeros((points.shape[0],), dtype=int)
    cx, cy = geometry.cylinder_center
    radius = float(geometry.cylinder_radius)
    for idx, (x, y) in enumerate(points):
        if (float(x) - cx) ** 2 + (float(y) - cy) ** 2 <= radius * radius:
            point_phase[idx] = 0
        elif geometry.contains_solid_point(float(x), float(y), tol=1.0e-12):
            point_phase[idx] = 2
        else:
            point_phase[idx] = 1

    cells: list[tuple[int, int, int, int]] = []
    cell_phase: list[int] = []
    stride = int(nx) + 1
    for j in range(int(ny)):
        y_mid = 0.5 * (ys[j] + ys[j + 1])
        for i in range(int(nx)):
            x_mid = 0.5 * (xs[i] + xs[i + 1])
            if (float(x_mid) - cx) ** 2 + (float(y_mid) - cy) ** 2 <= radius * radius:
                continue
            domain_id = 2 if geometry.contains_solid_point(float(x_mid), float(y_mid), tol=1.0e-12) else 1
            p0 = j * stride + i
            p1 = p0 + 1
            p2 = p1 + stride
            p3 = p0 + stride
            cells.append((p0, p1, p2, p3))
            cell_phase.append(domain_id)
    return points, np.asarray(cells, dtype=int), np.asarray(cell_phase, dtype=int), point_phase


def _porous_point_weights(points: np.ndarray, station_points: np.ndarray, *, radius: float) -> np.ndarray:
    sample = np.asarray(points, dtype=float).reshape(-1, 2)
    stations = np.asarray(station_points, dtype=float).reshape(-1, 2)
    d2 = np.sum((sample[:, None, :] - stations[None, :, :]) ** 2, axis=2)
    width2 = max(float(radius), 1.0e-12) ** 2
    weights = np.exp(-0.5 * d2 / width2)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-300)
    return weights


def _sample_full_fields(
    *,
    result: PorousFlapFPSIResult,
    geometry: DoubleFlapGeometry,
    points: np.ndarray,
    point_phase: np.ndarray,
    porous_weights: np.ndarray,
    state: np.ndarray,
    load: np.ndarray,
    time: float,
) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    point_phase = np.asarray(point_phase, dtype=int).reshape(-1)
    fluid_params = _result_fluid_parameters(result)
    fluid_mask = point_phase == 1
    porous_mask = point_phase == 2
    x = points[:, 0]
    y = points[:, 1]
    H = max(float(geometry.channel_height), 1.0e-12)
    L = max(float(geometry.channel_length), 1.0e-12)
    y_fraction = np.clip(y / H, 0.0, 1.0)
    x_fraction = np.clip(x / L, 0.0, 1.0)
    inlet = np.asarray(
        [geometry.inlet_velocity(float(yy), float(time), reference_velocity=float(fluid_params.reference_velocity)) for yy in y],
        dtype=float,
    )
    phase = 2.0 * math.pi * float(fluid_params.shedding_frequency) * float(time) + 2.0 * math.pi * x_fraction
    shedding = 1.0 + float(fluid_params.shedding_amplitude) * np.sin(phase)
    dynamic_pressure = 0.5 * float(fluid_params.density) * inlet * inlet
    shape = (1.0 - 0.35 * x_fraction) * (0.75 + 0.25 * np.sin(math.pi * y_fraction))

    fluid_velocity = np.zeros((points.shape[0], 3), dtype=float)
    fluid_pressure = np.zeros((points.shape[0],), dtype=float)
    fluid_velocity[fluid_mask, 0] = inlet[fluid_mask] * shedding[fluid_mask] * (1.0 - 0.08 * x_fraction[fluid_mask])
    fluid_velocity[fluid_mask, 1] = (
        0.08
        * inlet[fluid_mask]
        * float(fluid_params.shedding_amplitude)
        * np.cos(phase[fluid_mask])
        * np.sin(math.pi * y_fraction[fluid_mask])
    )
    fluid_pressure[fluid_mask] = dynamic_pressure[fluid_mask] * shape[fluid_mask] * shedding[fluid_mask]

    state = np.asarray(state, dtype=float).reshape(int(result.station_count), 3)
    load = np.asarray(load, dtype=float).reshape(int(result.station_count), 3)
    interpolated_state = porous_weights @ state
    interpolated_load = porous_weights @ load
    porous_displacement = np.zeros((points.shape[0], 3), dtype=float)
    pore_pressure = np.zeros((points.shape[0],), dtype=float)
    normal_flux = np.zeros((points.shape[0],), dtype=float)
    porous_displacement[porous_mask, :2] = interpolated_state[porous_mask, :2]
    pore_pressure[porous_mask] = interpolated_state[porous_mask, 2]
    normal_flux[porous_mask] = interpolated_load[porous_mask, 2]

    return {
        "fluid_velocity": fluid_velocity,
        "fluid_pressure": fluid_pressure,
        "porous_displacement": porous_displacement,
        "pore_pressure": pore_pressure,
        "normal_flux": normal_flux,
        "phase_id": point_phase.astype(float),
    }


def _xml_values(values: np.ndarray) -> str:
    flat = np.asarray(values).reshape(-1)
    if np.issubdtype(flat.dtype, np.integer):
        return " ".join(str(int(v)) for v in flat)
    return " ".join(f"{float(v):.17e}" for v in flat)


def _write_full_vtu(
    path: str | Path,
    *,
    points: np.ndarray,
    cells: np.ndarray,
    cell_phase: np.ndarray,
    point_data: dict[str, np.ndarray],
    displacement_scale: float,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    cells = np.asarray(cells, dtype=int).reshape(-1, 4)
    cell_phase = np.asarray(cell_phase, dtype=int).reshape(-1)
    deformed = points + float(displacement_scale) * np.asarray(point_data["porous_displacement"], dtype=float)[:, :2]
    vtk_points = np.column_stack((deformed, np.zeros((deformed.shape[0],), dtype=float)))
    connectivity = cells.reshape(-1)
    offsets = 4 * np.arange(1, cells.shape[0] + 1, dtype=int)
    types = np.full((cells.shape[0],), 9, dtype=int)

    with target.open("w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0"?>\n')
        handle.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        handle.write("  <UnstructuredGrid>\n")
        handle.write(f'    <Piece NumberOfPoints="{points.shape[0]}" NumberOfCells="{cells.shape[0]}">\n')
        handle.write('      <PointData Scalars="phase_id" Vectors="fluid_velocity">\n')
        for name, values in point_data.items():
            array = np.asarray(values)
            components = 1 if array.ndim == 1 else int(array.shape[1])
            handle.write(
                f'        <DataArray type="Float64" Name="{name}" NumberOfComponents="{components}" format="ascii">\n'
            )
            handle.write(f"          {_xml_values(array)}\n")
            handle.write("        </DataArray>\n")
        handle.write("      </PointData>\n")
        handle.write('      <CellData Scalars="domain_id">\n')
        handle.write('        <DataArray type="Int32" Name="domain_id" NumberOfComponents="1" format="ascii">\n')
        handle.write(f"          {_xml_values(cell_phase)}\n")
        handle.write("        </DataArray>\n")
        handle.write("      </CellData>\n")
        handle.write("      <Points>\n")
        handle.write('        <DataArray type="Float64" Name="Points" NumberOfComponents="3" format="ascii">\n')
        handle.write(f"          {_xml_values(vtk_points)}\n")
        handle.write("        </DataArray>\n")
        handle.write("      </Points>\n")
        handle.write("      <Cells>\n")
        for name, values, dtype in (
            ("connectivity", connectivity, "Int32"),
            ("offsets", offsets, "Int32"),
            ("types", types, "UInt8"),
        ):
            handle.write(f'        <DataArray type="{dtype}" Name="{name}" format="ascii">\n')
            handle.write(f"          {_xml_values(values)}\n")
            handle.write("        </DataArray>\n")
        handle.write("      </Cells>\n")
        handle.write("    </Piece>\n")
        handle.write("  </UnstructuredGrid>\n")
        handle.write("</VTKFile>\n")


def _write_pvd(path: str | Path, rows: list[dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0"?>\n')
        handle.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        handle.write("  <Collection>\n")
        for row in rows:
            handle.write(
                f'    <DataSet timestep="{float(row["time_s"]):.17e}" group="" part="0" '
                f'file="{Path(str(row["vtu"])).name}"/>\n'
            )
        handle.write("  </Collection>\n")
        handle.write("</VTKFile>\n")


def write_porous_flap_fpsi_vtk(
    result: PorousFlapFPSIResult,
    output_dir: str | Path,
    *,
    mode: str = "converged",
    displacement_scale: float = 1000.0,
    output_kind: str = "full",
    grid_nx: int = 160,
    grid_ny: int = 40,
) -> list[Path]:
    """Write VTK files for FPSI snapshots.

    ``mode="converged"`` writes one file per accepted time step. ``mode="all"``
    writes every stored fixed-point snapshot. ``output_kind="full"`` writes a
    sampled fluid/porous-body 2D slice as `.vtu` plus a `.pvd` time-series file.
    ``output_kind="trace"`` keeps the interface-only legacy `.vtk` trace.
    """

    mode_key = str(mode).strip().lower()
    if mode_key not in {"all", "converged"}:
        raise ValueError("VTK mode must be either 'all' or 'converged'.")
    kind_key = str(output_kind).strip().lower()
    if kind_key not in {"full", "trace", "both"}:
        raise ValueError("VTK output_kind must be one of 'full', 'trace', or 'both'.")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    n = int(result.station_count)
    lines = _trace_lines_from_segments(result.station_segment_ids, result.station_coordinates)
    geometry = _geometry_from_result(result)
    points, cells, cell_phase, point_phase = _build_full_sampling_grid(
        geometry,
        nx=int(grid_nx),
        ny=int(grid_ny),
    )
    porous_radius = max(float(geometry.arm_width), 0.5 * float(geometry.base_height), 1.0e-3)
    porous_weights = _porous_point_weights(points, result.station_points, radius=porous_radius)
    written: list[Path] = []
    manifest_rows: list[dict[str, object]] = []
    for col, meta in enumerate(result.snapshot_metadata):
        if mode_key == "converged" and not bool(meta.converged):
            continue
        state = np.asarray(result.porous_state_data[:, col], dtype=float).reshape(n, 3)
        load = np.asarray(result.load_guess_data[:, col], dtype=float).reshape(n, 3)
        row: dict[str, object] = {
            "step": int(meta.step),
            "time_s": float(meta.time_s),
            "coupling_iter": int(meta.coupling_iter),
            "converged": bool(meta.converged),
            "snapshot_col": int(col),
        }
        stem = (
            f"porous_fpsi_step{int(meta.step):04d}_iter{int(meta.coupling_iter):04d}"
            if mode_key == "converged"
            else f"porous_fpsi_snapshot{int(col) + 1:06d}_step{int(meta.step):04d}_iter{int(meta.coupling_iter):04d}"
        )
        if kind_key in {"full", "both"}:
            vtu_path = target / f"{stem}.vtu"
            point_data = _sample_full_fields(
                result=result,
                geometry=geometry,
                points=points,
                point_phase=point_phase,
                porous_weights=porous_weights,
                state=state,
                load=load,
                time=float(meta.time_s),
            )
            _write_full_vtu(
                vtu_path,
                points=points,
                cells=cells,
                cell_phase=cell_phase,
                point_data=point_data,
                displacement_scale=float(displacement_scale),
            )
            row["vtu"] = str(vtu_path)
            written.append(vtu_path)
        if kind_key in {"trace", "both"}:
            trace_path = target / f"{stem}_trace.vtk"
            _write_legacy_trace_vtk(
                trace_path,
                points=result.station_points,
                displacement=state[:, :2],
                pressure=state[:, 2],
                traction=load[:, :2],
                normal_flux=load[:, 2],
                normals=result.station_normals,
                segment_ids=result.station_segment_ids,
                lines=lines,
                displacement_scale=float(displacement_scale),
                title=f"porous_flap_fpsi step={meta.step} iter={meta.coupling_iter}",
            )
            row["trace_vtk"] = str(trace_path)
            written.append(trace_path)
        manifest_rows.append(row)

    if manifest_rows:
        with (target / "vtk_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            for row in manifest_rows:
                writer.writerow(row)
        if "vtu" in manifest_rows[0]:
            _write_pvd(target / "porous_fpsi.pvd", manifest_rows)
    return written


def _make_accelerator(
    *,
    accelerator: str,
    relaxation: float,
    history: int,
    timestep_horizon: int,
    backend: str,
) -> CouplingAccelerator:
    key = str(accelerator).strip().lower()
    if key == "aitken":
        return create_coupling_accelerator(
            key,
            relaxation=float(relaxation),
            relaxation_min=1.0e-4,
            relaxation_max=1.0,
            history=int(history),
            timestep_horizon=int(timestep_horizon),
            backend=str(backend),
        )
    return create_coupling_accelerator(
        key,
        relaxation=float(relaxation),
        history=int(history),
        timestep_horizon=int(timestep_horizon),
        backend=str(backend),
        regularization=1.0e-12,
    )


def run_porous_flap_fpsi(
    *,
    geometry: DoubleFlapGeometry | None = None,
    material: UPlMaterial2D | None = None,
    n_steps: int = 50,
    dt: float = 0.02,
    tolerance: float = 1.0e-10,
    max_iterations: int = 40,
    target_spacing: float = 0.025,
    accelerator: str = "iqln",
    accelerator_backend: str = "cpp",
    relaxation: float = 0.45,
    history: int = 8,
    timestep_horizon: int = 3,
    fluid_parameters: ReducedFluidParameters | None = None,
    porous_parameters: ReducedPorousParameters | None = None,
    snapshot_mode: str = "all",
) -> PorousFlapFPSIResult:
    """Run the partitioned porous-flap FPSI fixed-point example."""

    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    snapshot_mode_key = str(snapshot_mode).strip().lower()
    if snapshot_mode_key not in {"all", "converged"}:
        raise ValueError("snapshot_mode must be either 'all' or 'converged'.")

    geometry = nominal_double_flap_geometry() if geometry is None else geometry
    material = default_porous_flap_material() if material is None else material
    stations = build_porous_flap_interface_stations(geometry, target_spacing=float(target_spacing))
    fluid_params = ReducedFluidParameters() if fluid_parameters is None else fluid_parameters
    porous_params = ReducedPorousParameters() if porous_parameters is None else porous_parameters
    fluid = ReducedFreeFluidBlock(
        geometry=geometry,
        stations=stations,
        parameters=fluid_params,
    )
    porous = ReducedPorousFlapBlock(
        geometry=geometry,
        stations=stations,
        material=material,
        dt=float(dt),
        parameters=porous_params,
    )
    accel = _make_accelerator(
        accelerator=accelerator,
        relaxation=float(relaxation),
        history=int(history),
        timestep_horizon=int(timestep_horizon),
        backend=str(accelerator_backend),
    )

    state = np.zeros((stations.size, 3), dtype=float)
    step_records: list[CoupledStepRecord] = []
    iteration_records: list[CoupledIterationRecord] = []
    snapshot_metadata: list[FPSISnapshotMetadata] = []
    load_guess_snapshots: list[np.ndarray] = []
    load_return_snapshots: list[np.ndarray] = []
    disp_snapshots: list[np.ndarray] = []
    pressure_snapshots: list[np.ndarray] = []
    porous_state_snapshots: list[np.ndarray] = []
    state_guess_snapshots: list[np.ndarray] = []
    interface_disp_snapshots: list[np.ndarray] = []
    interface_pressure_snapshots: list[np.ndarray] = []
    interface_flux_snapshots: list[np.ndarray] = []
    interface_traction_snapshots: list[np.ndarray] = []

    for step in range(1, int(n_steps) + 1):
        time = float(step) * float(dt)
        accel.initialize_solution_step()
        x_curr = state.reshape(-1).copy()
        accepted = False
        last_fluid_response: FreeFluidInterfaceResponse | None = None
        last_norm = math.inf
        last_abs_norm = math.inf

        for iteration in range(1, int(max_iterations) + 1):
            current_state = x_curr.reshape(stations.size, 3)
            fluid_response = fluid.evaluate(current_state, time=time, dt=float(dt))
            porous_state = porous.solve(fluid=fluid_response, previous_state=state)
            g_curr = porous_state.reshape(-1)
            residual = g_curr - x_curr
            norm, abs_norm = _relative_norm(residual, g_curr)
            last_norm = norm
            last_abs_norm = abs_norm
            last_fluid_response = fluid_response
            converged_iteration = bool(norm <= float(tolerance))
            keep_snapshot = snapshot_mode_key == "all" or converged_iteration
            if keep_snapshot:
                current_state = np.asarray(current_state, dtype=float)
                porous_state = np.asarray(porous_state, dtype=float)
                load_vector = _fpsi_load_vector(fluid_response)
                load_guess_snapshots.append(load_vector)
                load_return_snapshots.append(load_vector.copy())
                state_guess_snapshots.append(current_state.reshape(-1))
                porous_state_snapshots.append(porous_state.reshape(-1))
                disp_snapshots.append(porous_state[:, :2].reshape(-1))
                pressure_snapshots.append(porous_state[:, 2].reshape(-1))
                interface_disp_snapshots.append(porous_state[:, :2].reshape(-1))
                interface_pressure_snapshots.append(porous_state[:, 2].reshape(-1))
                interface_flux_snapshots.append(np.asarray(fluid_response.normal_flux, dtype=float).reshape(-1))
                interface_traction_snapshots.append(np.asarray(fluid_response.traction, dtype=float).reshape(-1))
                snapshot_metadata.append(
                    FPSISnapshotMetadata(
                        step=step,
                        time_s=time,
                        coupling_iter=iteration,
                        converged=converged_iteration,
                        residual_norm=norm,
                    )
                )

            if converged_iteration:
                state = porous_state
                iteration_records.append(
                    CoupledIterationRecord(
                        step=step,
                        iteration=iteration,
                        time=time,
                        residual_norm=norm,
                        absolute_residual_norm=abs_norm,
                        update_norm=0.0,
                        relaxation=1.0,
                        used_history=False,
                        method="accepted",
                    )
                )
                accepted = True
                break

            update = accel.compute_next_iterate(x_curr=x_curr, residual_curr=residual)
            if update.next_iterate.shape != x_curr.shape:
                raise ValueError(
                    "Coupling accelerator returned an iterate with incompatible shape: "
                    f"{update.next_iterate.shape} != {x_curr.shape}"
                )
            update_norm = float(np.linalg.norm(update.delta.reshape(-1), ord=np.inf))
            iteration_records.append(
                CoupledIterationRecord(
                    step=step,
                    iteration=iteration,
                    time=time,
                    residual_norm=norm,
                    absolute_residual_norm=abs_norm,
                    update_norm=update_norm,
                    relaxation=float(update.relaxation),
                    used_history=bool(update.used_history),
                    method=str(update.method),
                )
            )
            x_curr = np.asarray(update.next_iterate, dtype=float).reshape(-1)
            if not np.all(np.isfinite(x_curr)):
                raise FloatingPointError(f"Non-finite coupling iterate at step {step}, iteration {iteration}.")

        if not accepted:
            accel.finalize_solution_step(accepted=False)
            raise RuntimeError(
                f"Porous-flap FPSI fixed point did not converge at step {step} "
                f"after {max_iterations} iterations; residual={last_norm:.3e}, abs={last_abs_norm:.3e}."
            )

        accel.finalize_solution_step(accepted=True)
        accepted_state = state.reshape(stations.size, 3)
        displacement_norm = np.linalg.norm(accepted_state[:, :2], axis=1)
        pressure = accepted_state[:, 2]
        step_records.append(
            CoupledStepRecord(
                step=step,
                time=time,
                converged=True,
                iterations=iteration,
                residual_norm=last_norm,
                max_displacement=float(displacement_norm.max(initial=0.0)),
                max_pressure=float(pressure.max(initial=0.0)),
                mean_pressure=float(pressure.mean() if pressure.size else 0.0),
                inlet_speed=float(0.0 if last_fluid_response is None else last_fluid_response.inlet_speed),
            )
        )

    state_rows = 3 * stations.size
    disp_rows = 2 * stations.size
    scalar_rows = stations.size
    return PorousFlapFPSIResult(
        geometry=asdict(geometry),
        material=asdict(material),
        fluid_parameters=asdict(fluid_params),
        porous_parameters=asdict(porous_params),
        station_count=stations.size,
        accelerator=str(accelerator),
        accelerator_backend=str(accelerator_backend),
        dt=float(dt),
        tolerance=float(tolerance),
        station_points=stations.points.copy(),
        station_normals=stations.normals.copy(),
        station_tangents=stations.tangents.copy(),
        station_weights=stations.weights.copy(),
        station_segment_ids=stations.segment_ids.copy(),
        station_coordinates=stations.station_coordinates.copy(),
        station_segment_names=tuple(stations.segment_names),
        steps=tuple(step_records),
        iterations=tuple(iteration_records),
        snapshot_metadata=tuple(snapshot_metadata),
        load_guess_data=_stack_snapshot_columns(load_guess_snapshots, rows=state_rows),
        load_data=_stack_snapshot_columns(load_return_snapshots, rows=state_rows),
        disp_data=_stack_snapshot_columns(disp_snapshots, rows=disp_rows),
        pressure_data=_stack_snapshot_columns(pressure_snapshots, rows=scalar_rows),
        porous_state_data=_stack_snapshot_columns(porous_state_snapshots, rows=state_rows),
        state_guess_data=_stack_snapshot_columns(state_guess_snapshots, rows=state_rows),
        interface_disp_data=_stack_snapshot_columns(interface_disp_snapshots, rows=disp_rows),
        interface_pressure_data=_stack_snapshot_columns(interface_pressure_snapshots, rows=scalar_rows),
        interface_flux_data=_stack_snapshot_columns(interface_flux_snapshots, rows=scalar_rows),
        interface_traction_data=_stack_snapshot_columns(interface_traction_snapshots, rows=disp_rows),
        final_state=state.copy(),
    )


def write_porous_flap_fpsi_result(
    result: PorousFlapFPSIResult,
    output_dir: str | Path,
    *,
    write_vtk: bool = False,
    vtk_mode: str = "converged",
    vtk_displacement_scale: float = 1000.0,
    vtk_output_kind: str = "full",
    vtk_grid_nx: int = 160,
    vtk_grid_ny: int = 40,
) -> Path:
    """Write summary, histories, and NIRB-style FPSI snapshot matrices."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    co_sim_dir = target / "coSimData"
    co_sim_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "geometry": result.geometry,
        "material": result.material,
        "fluid_parameters": result.fluid_parameters,
        "porous_parameters": result.porous_parameters,
        "station_count": result.station_count,
        "n_snapshots": result.n_snapshots,
        "accelerator": result.accelerator,
        "accelerator_backend": result.accelerator_backend,
        "dt": result.dt,
        "tolerance": result.tolerance,
        "converged": result.converged,
        "max_iterations": result.max_iterations,
        "steps": [asdict(step) for step in result.steps],
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (target / "timeseries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(result.steps[0]).keys()))
        writer.writeheader()
        for step in result.steps:
            writer.writerow(asdict(step))

    with (target / "iterations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(result.iterations[0]).keys()))
        writer.writeheader()
        for iteration in result.iterations:
            writer.writerow(asdict(iteration))

    with (target / "snapshot_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(result.snapshot_metadata[0]).keys()))
        writer.writeheader()
        for row in result.snapshot_metadata:
            writer.writerow(asdict(row))

    np.save(co_sim_dir / "load_guess_data.npy", result.load_guess_data)
    np.save(co_sim_dir / "load_data.npy", result.load_data)
    np.save(co_sim_dir / "disp_data.npy", result.disp_data)
    np.save(co_sim_dir / "pressure_data.npy", result.pressure_data)
    np.save(co_sim_dir / "porous_state_data.npy", result.porous_state_data)
    np.save(co_sim_dir / "state_guess_data.npy", result.state_guess_data)
    np.save(co_sim_dir / "interface_disp_data.npy", result.interface_disp_data)
    np.save(co_sim_dir / "interface_pressure_data.npy", result.interface_pressure_data)
    np.save(co_sim_dir / "interface_flux_data.npy", result.interface_flux_data)
    np.save(co_sim_dir / "interface_traction_data.npy", result.interface_traction_data)
    np.save(co_sim_dir / "iters.npy", np.asarray([step.iterations for step in result.steps], dtype=int))
    np.save(co_sim_dir / "fluid_time.npy", np.zeros((result.n_snapshots,), dtype=float))
    np.save(co_sim_dir / "structure_time.npy", np.zeros((result.n_snapshots,), dtype=float))
    np.save(co_sim_dir / "increment_time.npy", np.zeros((len(result.steps),), dtype=float))

    batch = SnapshotBatch(
        interface_forces=result.load_guess_data,
        full_displacements=result.porous_state_data,
        times=np.asarray([row.time_s for row in result.snapshot_metadata], dtype=float),
        subiterations=np.asarray([row.coupling_iter for row in result.snapshot_metadata], dtype=int),
        converged=np.asarray([row.converged for row in result.snapshot_metadata], dtype=bool),
        solid_times=np.zeros((result.n_snapshots,), dtype=float),
        fluid_times=np.zeros((result.n_snapshots,), dtype=float),
        metadata={
            "source": "porous_flap_fpsi",
            "input_layout": "per_station [traction_x, traction_y, normal_flux]",
            "output_layout": "per_station [displacement_x, displacement_y, pore_pressure]",
            "co_sim_dir": str(co_sim_dir),
            "station_count": int(result.station_count),
        },
    )
    batch.save(target / "fpsi_snapshot_batch.npz")

    if bool(write_vtk):
        write_porous_flap_fpsi_vtk(
            result,
            target / "vtk",
            mode=str(vtk_mode),
            displacement_scale=float(vtk_displacement_scale),
            output_kind=str(vtk_output_kind),
            grid_nx=int(vtk_grid_nx),
            grid_ny=int(vtk_grid_ny),
        )

    np.save(target / "final_state.npy", result.final_state)
    return target


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--target-spacing", type=float, default=0.025)
    parser.add_argument("--accelerator", choices=("constant", "aitken", "iqn_ils", "iqln", "mvqn"), default="iqln")
    parser.add_argument("--accelerator-backend", choices=("python", "cpp"), default="cpp")
    parser.add_argument("--relaxation", type=float, default=0.45)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--timestep-horizon", type=int, default=3)
    parser.add_argument("--snapshot-mode", choices=("all", "converged"), default="all")
    parser.add_argument("--write-vtk", action="store_true")
    parser.add_argument("--vtk-mode", choices=("all", "converged"), default="converged")
    parser.add_argument("--vtk-output-kind", choices=("full", "trace", "both"), default="full")
    parser.add_argument("--vtk-displacement-scale", type=float, default=1000.0)
    parser.add_argument("--vtk-grid-nx", type=int, default=160)
    parser.add_argument("--vtk-grid-ny", type=int, default=40)
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--use-reference-geometry", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("examples/poromechanics/artifacts/porous_flap_fpsi"))
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    geometry = load_porous_flap_geometry(args.reference_root, use_reference=bool(args.use_reference_geometry))
    result = run_porous_flap_fpsi(
        geometry=geometry,
        n_steps=int(args.steps),
        dt=float(args.dt),
        tolerance=float(args.tol),
        max_iterations=int(args.max_iterations),
        target_spacing=float(args.target_spacing),
        accelerator=str(args.accelerator),
        accelerator_backend=str(args.accelerator_backend),
        relaxation=float(args.relaxation),
        history=int(args.history),
        timestep_horizon=int(args.timestep_horizon),
        snapshot_mode=str(args.snapshot_mode),
    )
    output = write_porous_flap_fpsi_result(
        result,
        args.output_dir,
        write_vtk=bool(args.write_vtk),
        vtk_mode=str(args.vtk_mode),
        vtk_displacement_scale=float(args.vtk_displacement_scale),
        vtk_output_kind=str(args.vtk_output_kind),
        vtk_grid_nx=int(args.vtk_grid_nx),
        vtk_grid_ny=int(args.vtk_grid_ny),
    )
    print(
        "porous_flap_fpsi "
        f"converged={result.converged} "
        f"steps={len(result.steps)} "
        f"snapshots={result.n_snapshots} "
        f"max_iterations={result.max_iterations} "
        f"output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
