from pathlib import Path

import numpy as np
import pyvista as pv

from examples.NIRB.common import numeric_suffix_key, parse_git_lfs_pointer, read_arterial_vtk_series


def _save_polydata_compat(mesh: pv.PolyData, path: Path) -> None:
    # Some PyVista builds leave these metadata registries unset until arrays are
    # marked specially, but save() unconditionally reads them.
    for assoc_name in ("bitarray", "complex"):
        attr = f"_association_{assoc_name}_names"
        if not hasattr(mesh, attr):
            setattr(mesh, attr, {})
    mesh.save(path, binary=False)


def test_parse_git_lfs_pointer_extracts_oid_and_size(tmp_path: Path):
    pointer_path = tmp_path / "array.npy"
    pointer_path.write_text(
        "\n".join(
            [
                "version https://git-lfs.github.com/spec/v1",
                "oid sha256:abc123",
                "size 42",
            ]
        ),
        encoding="utf-8",
    )

    pointer = parse_git_lfs_pointer(pointer_path)
    assert pointer.oid == "abc123"
    assert pointer.size == 42


def test_numeric_suffix_key_orders_by_trailing_number():
    paths = [Path("out_fluid_10.vtk"), Path("out_fluid_2.vtk"), Path("out_fluid_1.vtk")]
    assert [path.name for path in sorted(paths, key=numeric_suffix_key)] == [
        "out_fluid_1.vtk",
        "out_fluid_2.vtk",
        "out_fluid_10.vtk",
    ]


def test_read_arterial_vtk_series_builds_feature_major_arrays(tmp_path: Path):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    for step, pressure0 in enumerate([1.0, 2.0], start=1):
        mesh = pv.PolyData(points)
        mesh.point_data["velocity"] = np.array([[pressure0, 0.0, 0.0], [pressure0 + 1.0, 0.0, 0.0]])
        mesh.point_data["pressure"] = np.array([pressure0, pressure0 + 1.0])
        mesh.point_data["diameter"] = np.array([1.0 + 0.1 * step, 1.0 + 0.2 * step])
        mesh.point_data["iters"] = np.array([3.0 + step, 3.0 + step])
        _save_polydata_compat(mesh, tmp_path / f"out_fluid_{step}.vtk")

    series = read_arterial_vtk_series(tmp_path)
    assert series["pressure"].shape == (2, 2)
    assert series["diameter"].shape == (2, 2)
    assert series["velocity"].shape == (2, 2)
    assert np.allclose(series["iters"], [4.0, 5.0])
