from __future__ import annotations

import json
import math
import re
import subprocess
import urllib.request
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


ARTERIAL_REPO_URL = "https://github.com/FsiROM/ArterialWallROM.git"
DOUBLE_FLAP_REPO_URL = "https://github.com/FsiROM/DoubleFlap.git"
_LFS_CONTENT_TYPE = "application/vnd.git-lfs+json"


@dataclass(frozen=True)
class GitLFSPointer:
    oid: str
    size: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_cache_root() -> Path:
    return repo_root() / ".tmp" / "nirb_benchmarks"


def default_artifacts_root() -> Path:
    return Path(__file__).resolve().parent / "artifacts"


def ensure_git_clone(url: str, destination: Path) -> Path:
    destination = Path(destination)
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(destination)],
        check=True,
        cwd=repo_root(),
    )
    return destination


def is_git_lfs_pointer(path: Path) -> bool:
    try:
        return path.read_text(encoding="utf-8").startswith("version https://git-lfs.github.com/spec/v1")
    except UnicodeDecodeError:
        return False


def parse_git_lfs_pointer(path: str | Path) -> GitLFSPointer:
    pointer_path = Path(path)
    lines = pointer_path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("version https://git-lfs.github.com/spec/v1"):
        raise ValueError(f"{pointer_path} is not a Git LFS pointer file")

    oid = None
    size = None
    for line in lines:
        if line.startswith("oid sha256:"):
            oid = line.split(":", 1)[1]
        elif line.startswith("size "):
            size = int(line.split()[1])
    if oid is None or size is None:
        raise ValueError(f"{pointer_path} does not contain a complete Git LFS pointer")
    return GitLFSPointer(oid=oid, size=size)


def _chunked(values: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def _request_public_lfs_urls(repo_url: str, pointers: Sequence[GitLFSPointer]) -> dict[str, str]:
    api_url = repo_url.rstrip("/") + "/info/lfs/objects/batch"
    urls: dict[str, str] = {}
    unique_pointers = list({(pointer.oid, pointer.size): pointer for pointer in pointers}.values())
    for chunk in _chunked(unique_pointers, chunk_size=32):
        payload = json.dumps(
            {
                "operation": "download",
                "transfers": ["basic"],
                "objects": [asdict(pointer) for pointer in chunk],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            api_url,
            data=payload,
            headers={
                "Accept": _LFS_CONTENT_TYPE,
                "Content-Type": _LFS_CONTENT_TYPE,
            },
        )
        with urllib.request.urlopen(request) as response:
            response_payload = json.load(response)
        for entry in response_payload["objects"]:
            urls[entry["oid"]] = entry["actions"]["download"]["href"]
    return urls


def download_public_lfs_files(
    repo_url: str,
    pointer_files: Sequence[Path],
    source_root: Path,
    destination_root: Path,
) -> list[Path]:
    source_root = Path(source_root)
    destination_root = Path(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    destination_paths = [destination_root / pointer_path.relative_to(source_root) for pointer_path in pointer_files]
    missing_indices = [
        index
        for index, (pointer_path, destination_path) in enumerate(zip(pointer_files, destination_paths))
        if not destination_path.exists()
        or destination_path.stat().st_size != parse_git_lfs_pointer(pointer_path).size
        or is_git_lfs_pointer(destination_path)
    ]
    if not missing_indices:
        return destination_paths

    requested_pointers = [parse_git_lfs_pointer(pointer_files[index]) for index in missing_indices]
    download_urls = _request_public_lfs_urls(repo_url, requested_pointers)
    for index in missing_indices:
        pointer_path = pointer_files[index]
        pointer = parse_git_lfs_pointer(pointer_path)
        destination_path = destination_paths[index]
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(download_urls[pointer.oid], destination_path)
    return destination_paths


def numeric_suffix_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    return (int(match.group(1)) if match else -1, path.name)


def list_numeric_files(directory: Path, pattern: str) -> list[Path]:
    return sorted(directory.glob(pattern), key=numeric_suffix_key)


def read_arterial_vtk_series(directory: Path) -> dict[str, np.ndarray]:
    pressure: list[np.ndarray] = []
    diameter: list[np.ndarray] = []
    velocity: list[np.ndarray] = []
    iterations: list[float] = []

    for vtk_path in list_numeric_files(directory, "out_fluid_*.vtk"):
        mesh = pv.read(vtk_path)
        pressure.append(np.asarray(mesh.point_data["pressure"], dtype=float))
        diameter.append(np.asarray(mesh.point_data["diameter"], dtype=float))
        velocity.append(np.asarray(mesh.point_data["velocity"], dtype=float)[:, 0])
        iterations.append(float(np.asarray(mesh.point_data["iters"], dtype=float)[0]))

    return {
        "pressure": np.column_stack(pressure),
        "diameter": np.column_stack(diameter),
        "velocity": np.column_stack(velocity),
        "iters": np.asarray(iterations, dtype=float),
    }


def read_structural_vtk_series(directory: Path) -> dict[str, np.ndarray]:
    displacements: list[np.ndarray] = []
    point_loads: list[np.ndarray] = []

    for vtk_path in list_numeric_files(directory, "Structure_0_*.vtk"):
        mesh = pv.read(vtk_path)
        displacement = np.asarray(mesh.point_data["DISPLACEMENT"], dtype=float)[:, :2]
        point_load = np.asarray(mesh.point_data["POINT_LOAD"], dtype=float)[:, :2]
        displacements.append(displacement.reshape(-1))
        point_loads.append(point_load.reshape(-1))

    return {
        "displacements": np.column_stack(displacements),
        "point_loads": np.column_stack(point_loads),
    }


def reynolds_from_case(case_name: str) -> float:
    return 0.1 * 2.5 * 1000.0 / (float(case_name) / 10.0)


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if is_dataclass(value):
            return asdict(value)
        raise TypeError(f"unsupported JSON type: {type(value).__name__}")

    path.write_text(json.dumps(data, indent=2, default=default), encoding="utf-8")


def _heatmap_matrix(
    entries: Sequence[Any],
    *,
    x_attr: str,
    y_attr: str,
    value_attr: str,
) -> tuple[np.ndarray, list[int], list[int]]:
    x_values = sorted({int(getattr(entry, x_attr)) for entry in entries})
    y_values = sorted({int(getattr(entry, y_attr)) for entry in entries})
    matrix = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    for entry in entries:
        x_index = x_values.index(int(getattr(entry, x_attr)))
        y_index = y_values.index(int(getattr(entry, y_attr)))
        matrix[y_index, x_index] = float(getattr(entry, value_attr))
    return matrix, x_values, y_values


def plot_heatmap(
    entries: Sequence[Any],
    *,
    x_attr: str,
    y_attr: str,
    value_attr: str,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    matrix, x_values, y_values = _heatmap_matrix(entries, x_attr=x_attr, y_attr=y_attr, value_attr=value_attr)
    figure, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    axis.set_xticks(np.arange(len(x_values)))
    axis.set_xticklabels(x_values)
    axis.set_yticks(np.arange(len(y_values)))
    axis.set_yticklabels(y_values)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_xy(
    x_values: np.ndarray,
    series: Sequence[tuple[str, np.ndarray]],
    *,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for label, values in series:
        axis.plot(x_values, values, label=label)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_cumulative_iterations(
    fom_iterations: np.ndarray,
    rom_iterations: np.ndarray,
    *,
    path: Path,
    title: str,
) -> None:
    fom = np.asarray(fom_iterations, dtype=float).ravel()
    rom = np.asarray(rom_iterations, dtype=float).ravel()
    x_values = np.arange(1, fom.size + 1)
    total = fom.sum()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(x_values, 100.0 * np.cumsum(fom) / total, label="FOM-FOM")
    axis.plot(x_values, 100.0 * np.cumsum(rom) / total, label="ROM-FOM")
    axis.set_title(title)
    axis.set_xlabel("Time increment")
    axis.set_ylabel("Accumulated iterations [%]")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def compute_tube_stress_strain(section: np.ndarray, pressure: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius0 = 1.0 / math.sqrt(math.pi)
    radius = np.sqrt(np.asarray(section, dtype=float) / math.pi)
    strain = (radius - radius0) / radius0
    stress = np.asarray(pressure, dtype=float) * radius
    return strain, stress


def reject_outliers(values: np.ndarray, m: float = 3.0) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        return array
    mask = np.abs(array - np.mean(array)) < m * np.std(array)
    return array[mask]
