# pycutfem/jit/cache.py
import hashlib, pickle, importlib.util, sys, tempfile, textwrap, os
from pathlib import Path
from contextlib import contextmanager
from types import ModuleType
from typing import Tuple, Dict, List, Any

import numpy as np  # numba kernels rely on it

from pycutfem.core import mesh
from pycutfem.jit.ir import LoadAnalytic

class KernelCache:
    """
    Compile-once, reuse-many cache for JIT-generated kernels.

    get_kernel(ir_sequence, codegen) -> (callable kernel, param_order list)
    """

    _CACHE_DIR = Path.home() / ".cache" / "pycutfem_jit"
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def __init__(self) -> None:
        self.in_memory_cache: Dict[str, Tuple[Any, List[str]]] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get_kernel(self, ir_sequence: list, codegen, mesh_sig=None):
        """
        Build (or load) a kernel for the given IR sequence and return
        (kernel_fn, param_order_list).  The codegen object must expose
        `generate_source(ir, "kernel")  -> (source_str, analytic_map, param_order)`.
        """
        ir_hash = self._hash_ir(ir_sequence,mesh_sig)

        # fast path: session cache
        if ir_hash in self.in_memory_cache:
            return self.in_memory_cache[ir_hash]

        module_name = f"_pycutfem_kernel_{ir_hash}"
        source_file = self._CACHE_DIR / f"{module_name}.py"

        # ------------------------------------------------------------------
        # compile if source does not exist
        # ------------------------------------------------------------------
        if not source_file.exists():
            self._compile_and_write(source_file, ir_sequence, codegen)

        # ------------------------------------------------------------------
        # import the module (with stale-kernel detection)
        # ------------------------------------------------------------------
        module = self._import_with_fallback(module_name, source_file,
                                            ir_sequence, codegen)

        param_order: List[str] = getattr(module, "PARAM_ORDER")     # must exist
        kernel_factory = module.get_kernel

        analytic_map = {
            f"analytic_func_{op.func_id}": op.func_ref
            for op in ir_sequence if isinstance(op, LoadAnalytic)
        }
        kernel_fn = kernel_factory(analytic_map)

        self.in_memory_cache[ir_hash] = (kernel_fn, param_order)
        return kernel_fn, param_order

    # ------------------------------------------------------------------
    # --- helpers -------------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _hash_ir(ir_sequence,mesh_sig=None) -> str:
        """Hash IR after stripping non-picklable objects."""
        def _hashable(op):
            if isinstance(op, LoadAnalytic):
                return ("analytic", op.func_id)        # ignore func_ref
            return op
        payload = [_hashable(o) for o in ir_sequence]
        if mesh_sig is not None:
            payload.append(("mesh", mesh_sig))
        return hashlib.sha256(pickle.dumps(payload)).hexdigest()

    # ..................................................................
    # locking helpers (prevent two procs writing same file concurrently)
    # ..................................................................
    @contextmanager
    def _file_lock(self, file: Path):
        lock_path = file.with_suffix(".lock")
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                break          # acquired
            except FileExistsError:
                # another proc has the lock; wait&retry
                import time
                time.sleep(0.05)
        try:
            yield
        finally:
            os.close(fd)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    # ..................................................................
    def _compile_and_write(self, target: Path, ir_seq, codegen):
        """Generate source, sanity-check, and write atomically."""
        print(f"JIT cache miss → compiling {target.name}")
        src, ana_map, param_order = codegen.generate_source(ir_seq, "kernel")
        full_src = self._create_module_source(src, ana_map.keys())

        # syntax sanity-check
        try:
            compile(full_src, f"<{target.name}>", "exec")
        except SyntaxError as e:
            # dump the faulty code for inspection
            fail = Path(tempfile.gettempdir()) / Path(f"{target.stem}_FAILED.py")
            fail.write_text(full_src)
            raise SyntaxError(
                f"Generated kernel has syntax error (saved to {fail})\n"
                f"{e.msg} at line {e.lineno}: {e.text.strip()}"
            ) from e

        # write with lock
        with self._file_lock(target):
            target.write_text(full_src, encoding="utf-8")

    # ..................................................................
    def _import_with_fallback(self, modname: str, path: Path,
                              ir_seq, codegen) -> ModuleType:
        """Import module; if stale (no PARAM_ORDER), regenerate once."""
        module = self._import_module(modname, path)

        if not hasattr(module, "PARAM_ORDER"):
            print(f"Kernel '{modname}' is stale – rebuilding.")
            try:
                path.unlink()
            except OSError:
                pass
            self._compile_and_write(path, ir_seq, codegen)
            module = self._import_module(modname, path)

        return module

    # ..................................................................
    @staticmethod
    def _import_module(modname: str, path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(modname, path)
        if not spec or not spec.loader:
            raise ImportError(f"Could not load spec for {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod

    # ..................................................................
    @staticmethod
    def _create_module_source(kernel_src: str, analytic_names) -> str:
        """
        Wrap the kernel source with imports and a factory that
        injects analytic callables at run-time.
        """
        injection_block = "\n".join(
            f"    globals()['{name}'] = analytic_map['{name}']"
            for name in analytic_names
        ) or "    pass"

        return textwrap.dedent(f"""\
# This file is auto-generated by pycutfem JIT – DO NOT EDIT
import numpy as np
import numba

{kernel_src}

def get_kernel(analytic_map):
{injection_block}
    return {kernel_src.split('def ',1)[1].split('(',1)[0]}
""")
