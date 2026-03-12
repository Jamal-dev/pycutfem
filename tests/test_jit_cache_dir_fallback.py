from pathlib import Path

from pycutfem.jit import cache as jit_cache


def test_jit_cache_dir_falls_back_to_temp_when_user_cache_is_unwritable(tmp_path, monkeypatch):
    xdg_cache = tmp_path / "xdg_cache"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.chmod(0o555)

    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir(parents=True, exist_ok=True)

    monkeypatch.delenv("PYCUTFEM_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_cache))
    monkeypatch.setattr(jit_cache.tempfile, "tempdir", str(tmpdir))

    resolved = jit_cache._resolve_cache_dir()

    assert resolved == (Path(tmpdir) / "pycutfem_jit").resolve()

