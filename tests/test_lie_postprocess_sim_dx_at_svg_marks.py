from __future__ import annotations

import importlib.util
import py_compile
from pathlib import Path


def test_postprocess_sim_dx_at_svg_marks_compiles_and_parses_steps(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "examples/biofilms/benchmarks/lie/postprocess_sim_dx_at_svg_marks.py"

    py_compile.compile(str(script), doraise=True)

    spec = importlib.util.spec_from_file_location("postprocess_sim_dx_at_svg_marks", str(script))
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    vtk_dir = tmp_path / "vtk"
    vtk_dir.mkdir(parents=True, exist_ok=True)
    (vtk_dir / "step=0000.vtu").write_text("dummy")
    (vtk_dir / "step=0012.vtu").write_text("dummy")
    (vtk_dir / "not_a_step.vtu").write_text("dummy")

    steps = mod._vtk_step_files(vtk_dir)
    assert [s for (s, _p) in steps] == [0, 12]

