from pathlib import Path

import pycutfem.jit as jit_module
import pycutfem.jit.kernel_args as kernel_args
from pycutfem.jit import cache as jit_cache
from pycutfem.jit.kernel_args import _resolve_ref_table_cache_dir


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


def test_jit_kernel_cache_respects_runtime_env_change(tmp_path, monkeypatch):
    cache_a = tmp_path / "cache_a"
    cache_b = tmp_path / "cache_b"

    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(cache_a))
    jit_cache.KernelCache._CACHE_DIR = None
    jit_module._KERNEL_CACHE_SINGLETON = None
    jit_module._KERNEL_CACHE_DIR_TOKEN = None
    kernel_a = jit_module._get_kernel_cache()

    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(cache_b))
    kernel_b = jit_module._get_kernel_cache()

    assert kernel_a is not kernel_b
    assert kernel_a._cache_dir == cache_a.resolve()
    assert kernel_b._cache_dir == cache_b.resolve()


def test_ref_table_cache_dir_tracks_runtime_env_change(tmp_path, monkeypatch):
    cache_a = tmp_path / "cache_a"
    cache_b = tmp_path / "cache_b"

    kernel_args._REF_TABLE_CACHE.clear()
    kernel_args._REF_TABLE_CACHE_DIR = None
    kernel_args._REF_TABLE_CACHE_DIR_TOKEN = None

    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(cache_a))
    ref_a = _resolve_ref_table_cache_dir()

    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(cache_b))
    ref_b = _resolve_ref_table_cache_dir()

    assert ref_a == (cache_a / "ref_tables").resolve()
    assert ref_b == (cache_b / "ref_tables").resolve()
