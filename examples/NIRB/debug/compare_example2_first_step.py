from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv

from examples.NIRB.common import dump_json
from examples.NIRB.double_flap_reference import default_double_flap_root


def _coord_key(x: float, y: float, ndigits: int = 12) -> tuple[float, float]:
    return (round(float(x), ndigits), round(float(y), ndigits))


@dataclass
class PointLookup:
    coords: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        coords = np.asarray(self.coords, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must have shape (n, 2)")
        if values.ndim != 2 or values.shape[0] != coords.shape[0]:
            raise ValueError("values must have shape (n, m)")
        self.coords = coords
        self.values = values
        self._dict = {_coord_key(x, y): values[i].copy() for i, (x, y) in enumerate(coords)}

    def sample(self, target_coords: np.ndarray) -> np.ndarray:
        target = np.asarray(target_coords, dtype=float)
        out = np.empty((target.shape[0], self.values.shape[1]), dtype=float)
        for i, (x, y) in enumerate(target):
            hit = self._dict.get(_coord_key(x, y))
            if hit is not None:
                out[i, :] = hit
                continue
            dist2 = np.sum((self.coords - np.asarray([x, y], dtype=float)) ** 2, axis=1)
            out[i, :] = self.values[int(np.argmin(dist2)), :]
        return out


def _load_reference_field(vtk_path: Path, field_name: str, target_coords: np.ndarray) -> np.ndarray:
    mesh = pv.read(vtk_path)
    coords = np.asarray(mesh.points[:, :2], dtype=float)
    values = np.asarray(mesh.point_data[field_name], dtype=float)[:, :2]
    return PointLookup(coords, values).sample(target_coords)


def _load_local_snapshot(matrix_path: Path, coords_path: Path, snapshot_index: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.load(coords_path)
    matrix = np.load(matrix_path)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.shape[1] == 0:
        raise ValueError(f"snapshot matrix is empty: {matrix_path}")
    resolved_index = int(snapshot_index)
    if resolved_index < 0:
        resolved_index = matrix.shape[1] + resolved_index
    if not (0 <= resolved_index < matrix.shape[1]):
        raise IndexError(f"snapshot index {snapshot_index} is out of range for {matrix_path}")
    values = np.asarray(matrix[:, resolved_index], dtype=float).reshape(-1, 2)
    return coords, values


def _rms(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=float).reshape(-1)
    return float(np.linalg.norm(flat) / np.sqrt(max(flat.size, 1)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the local Example 2 first-step interface u/v against the DoubleFlap reference.")
    parser.add_argument("--local-output-dir", type=Path, required=True)
    parser.add_argument("--snapshot-index", type=int, default=-1)
    parser.add_argument("--reference-root", type=Path, default=default_double_flap_root())
    parser.add_argument(
        "--reference-payload-root",
        type=Path,
        default=Path(".tmp/nirb_benchmarks/DoubleFlap_lfs_step1"),
    )
    parser.add_argument("--reference-case", type=str, default="08.3")
    parser.add_argument("--reference-interface-coords", type=Path, default=None)
    parser.add_argument("--reference-cfd-vtk", type=Path, default=None)
    parser.add_argument("--reference-csm-vtk", type=Path, default=None)
    parser.add_argument("--reference-iters", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reference_root = Path(args.reference_root).resolve()
    payload_root = Path(args.reference_payload_root).resolve()
    local_output_dir = Path(args.local_output_dir).resolve()
    local_co_sim = local_output_dir / "coSimData"

    reference_coords_path = (
        Path(args.reference_interface_coords).resolve()
        if args.reference_interface_coords is not None
        else reference_root / "coSimData" / "downloaded_coords_interf.npy"
    )
    reference_cfd_vtk = (
        Path(args.reference_cfd_vtk).resolve()
        if args.reference_cfd_vtk is not None
        else payload_root / "Double_E10" / str(args.reference_case) / "vtk_data" / "vtk_output_fsi_cfd" / "FluidParts_FluidPart_0_1.vtk"
    )
    reference_csm_vtk = (
        Path(args.reference_csm_vtk).resolve()
        if args.reference_csm_vtk is not None
        else payload_root / "Double_E10" / str(args.reference_case) / "vtk_data" / "vtk_output_fsi_csm" / "Structure_0_1.vtk"
    )
    reference_iters_path = (
        Path(args.reference_iters).resolve()
        if args.reference_iters is not None
        else payload_root / "Double_E10" / str(args.reference_case) / "coSimData" / "iters.npy"
    )
    output_path = (
        Path(args.output_path).resolve()
        if args.output_path is not None
        else local_output_dir / "comparison_step1.json"
    )

    reference_coords = np.load(reference_coords_path)
    reference_disp = _load_reference_field(reference_csm_vtk, "DISPLACEMENT", reference_coords)
    reference_velocity = _load_reference_field(reference_cfd_vtk, "VELOCITY", reference_coords)

    local_coords, local_disp_values = _load_local_snapshot(
        local_co_sim / "interface_disp_data.npy",
        local_co_sim / "coords_interf.npy",
        snapshot_index=int(args.snapshot_index),
    )
    _, local_velocity_values = _load_local_snapshot(
        local_co_sim / "interface_velocity_data.npy",
        local_co_sim / "coords_interf.npy",
        snapshot_index=int(args.snapshot_index),
    )

    local_disp = PointLookup(local_coords, local_disp_values).sample(reference_coords)
    local_velocity = PointLookup(local_coords, local_velocity_values).sample(reference_coords)

    disp_diff = local_disp - reference_disp
    velocity_diff = local_velocity - reference_velocity
    reference_iters = np.load(reference_iters_path)

    summary = {
        "local_output_dir": str(local_output_dir),
        "reference_case": str(args.reference_case),
        "reference_interface_coords_path": str(reference_coords_path),
        "reference_cfd_vtk": str(reference_cfd_vtk),
        "reference_csm_vtk": str(reference_csm_vtk),
        "reference_iters_path": str(reference_iters_path),
        "reference_step1_coupling_iterations": int(reference_iters[0]),
        "reference_interface_points": int(reference_coords.shape[0]),
        "local_interface_points": int(local_coords.shape[0]),
        "snapshot_index": int(args.snapshot_index),
        "disp_abs_rms": _rms(disp_diff),
        "disp_abs_max": float(np.max(np.abs(disp_diff))),
        "disp_rel_l2": float(
            np.linalg.norm(disp_diff.reshape(-1)) / max(np.linalg.norm(reference_disp.reshape(-1)), 1.0e-15)
        ),
        "velocity_abs_rms": _rms(velocity_diff),
        "velocity_abs_max": float(np.max(np.abs(velocity_diff))),
        "velocity_rel_l2": float(
            np.linalg.norm(velocity_diff.reshape(-1)) / max(np.linalg.norm(reference_velocity.reshape(-1)), 1.0e-15)
        ),
        "reference_disp_max_norm": float(np.max(np.linalg.norm(reference_disp, axis=1))),
        "reference_velocity_max_norm": float(np.max(np.linalg.norm(reference_velocity, axis=1))),
        "local_disp_max_norm": float(np.max(np.linalg.norm(local_disp, axis=1))),
        "local_velocity_max_norm": float(np.max(np.linalg.norm(local_velocity, axis=1))),
    }
    dump_json(summary, output_path)
    print(f"comparison: {output_path}")


if __name__ == "__main__":
    main()
