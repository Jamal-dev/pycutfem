"""Cache and compiler driver for C++ kernels."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import sysconfig
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pycutfem.jit.cache import KernelCache
from .compiler import compile_extension, get_compile_mode_tag

try:  # pragma: no cover - Windows fallback is exercised only off Linux.
    import fcntl
except ImportError:  # pragma: no cover - platform dependent.
    fcntl = None

# Bump when generated C++ changes in a way that requires recompilation of cached kernels.
# Bump when generated C++ changes or when helper semantics change in a way that
# requires recompilation of cached kernels (e.g. stride-aware accessors).
# 2026-03-10:
# - v25 invalidated stale facet-Hessian kernels that still referenced unsided
#   Hx/Hy on side-aware facet geometry.
# - v26 additionally invalidates boundary-facet (`ds`) Hessian kernels after
#   separating unsided Hxi0/Hxi1 requests from sided pos/neg Hxi requests.
# - v27 invalidates kernels that previously over-requested Hxi0/Hxi1 for
#   ordinary Grad(.) operators instead of only grad(div(.)).
# - v28 invalidates kernels after fixing unsided vector trial/test second-derivative
#   table loads in the C++ codegen (the previous code only populated the last
#   component row due to a misplaced for/else).
# - v29 invalidates kernels after fixing physical push-forward of trial/test
#   second derivatives in volume kernels (the previous code used raw reference
#   d20/d11/d02 tables directly).
# - v30 invalidates kernels after wiring the correct Jloc/Hx/Hy selection for
#   those trial/test second-derivative push-forwards.
# - v33 invalidates kernels after adding direct H(div) component physical-table
#   views (hval/hgrad/hhess) to the C++ kernel prologue for higher-order RT
#   whole-domain assembly.
# - v34 invalidates kernels after fixing scalar-times-gradient products with
#   2D stored layouts to keep `StackItem.kind` aligned with emitted storage.
# - v53 invalidates kernels after fixing component-carried basis/test-trial dot
#   lowering for transformed gradient carriers and updating helper overloads.
# - v54 invalidates kernels after preserving scalar-scaled canonical gradient
#   carriers through later inner-product lowering.
# - v55 invalidates kernels after forcing scalar-scaled canonical gradient
#   carriers to retain their basis/value role through codegen stack pushes.
# - v56 invalidates kernels after fixing the dedicated scalar-times-grad product
#   path so 2D gradient carriers are not downgraded to plain matrices.
# - v57 invalidates kernels after extending GradStack +/- matrix helpers to
#   also accept raw vector-of-matrix temporaries emitted by planner-aligned
#   gradient stack sums in the C++ backend.
# - v58 invalidates kernels after fixing scalar-times-mixed-gradient lowering
#   so component-stack temporaries do not fall back to the plain MatrixXd path.
# - v59 invalidates kernels after fixing the generic scalar-times-stack product
#   fallback for mixed and Hessian carriers in the C++ backend.
# - v60 invalidates kernels after preserving mixed-role carriers through
#   normalization even when planner storage kind is reported as plain "mat".
# - v61 invalidates kernels after preventing planner mat-kinds from
#   downgrading emitted grad/hess/mixed stack temporaries inside push_bin().
# - v62 invalidates kernels after routing stack-backed add/sub/scale temporaries
#   through planner-aligned stack helpers instead of MatrixXd shortcuts.
# - v63 invalidates kernels after promoting stack-backed basis tensors to their
#   semantic C++ carrier kind before later dot/add/sub dispatch.
# - v64 invalidates kernels after adding stack-stack contraction overloads and
#   stack-aware dot-case lowering for basis/value tensor contractions.
# - v65 invalidates kernels after fixing sum-path layout lookup to use the
#   lowering result instead of ExpressionMeta.
# - v66 invalidates kernels after normalizing row/column rank-1 values through
#   the shared mixed-dot contraction helpers.
# - v67 invalidates kernels after forcing all vector-tagged C++ temporaries to
#   materialize as true `Eigen::VectorXd` values instead of relying on `auto`
#   expressions that could still be row/column matrices.
# - v68 invalidates kernels after routing 2D basis-rank1 contractions,
#   including scalar-gradient test/trial carriers, through the shared
# - v69 invalidates kernels after adding runtime quadrature-state loads,
#   including matrix-valued quadrature-state support in the C++ backend.
#   mass-style contraction path instead of the stack-only grad-grad helper.
CODEGEN_ABI_CPP = "2026-04-14-cpp-v75-matrix-like-vector-dot-normalization"


@contextmanager
def _module_build_lock(lock_path: Path):
    """
    Serialize cross-process compile/import of a single cached extension module.

    Without this lock two long-running benchmark processes can race on the same
    `<module>.so`: one process starts importing while the other is still linking,
    which shows up as `ImportError: ... file too short`.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _looks_like_incomplete_binary_import(exc: BaseException) -> bool:
    text = str(exc).strip().lower()
    if not text:
        return False
    hints = (
        "file too short",
        "invalid elf header",
        "wrong elf class",
        "cannot dynamically load position-independent executable",
        "failed to map segment from shared object",
    )
    return any(hint in text for hint in hints)


class CppKernelCache:
    """
    Compile-once, reuse-many cache for C++/pybind11 kernels.

    The interface mirrors :class:`pycutfem.jit.cache.KernelCache`.
    """

    # Optional test hook; when None, cache dir is resolved from env per instance.
    _CACHE_DIR: Path | None = None
    # Share the in-memory kernel cache across instances so tests that vary
    # `PYCUTFEM_CACHE_DIR` do not repeatedly compile/import the same kernel
    # under different cache roots (which can exhaust per-process TLS resources).
    _GLOBAL_IN_MEMORY_CACHE: Dict[str, Tuple[Any, List[str], List[str]]] = {}

    def __init__(self) -> None:
        # NOTE: this cache is intentionally shared across instances.
        self.in_memory_cache = self._GLOBAL_IN_MEMORY_CACHE
        self._cache_dir = self._resolve_cache_dir()
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        self._ext_suffix = suffix if suffix else ".so"

    @classmethod
    def _resolve_cache_dir(cls) -> Path:
        """
        Resolve cache dir at runtime so per-test env overrides are honored.

        The default cache location lives under the user's cache directory
        (typically `~/.cache/pycutfem_jit`). Some environments (sandboxed CI,
        read-only home dirs) disallow writes there; in that case we fall back
        to a temp-directory cache so the C++ backend remains usable.
        """
        override = cls._CACHE_DIR
        candidates: list[Path] = []
        if override is not None:
            candidates.append(Path(override))
        else:
            root = os.environ.get("PYCUTFEM_CACHE_DIR", "")
            if root:
                root_dir = Path(root)
            else:
                xdg = os.environ.get("XDG_CACHE_HOME", "")
                root_dir = Path(xdg) / "pycutfem_jit" if xdg else Path.home() / ".cache" / "pycutfem_jit"
            candidates.append(root_dir / "cpp")

        # Always have a safe fallback under the system temp dir.
        candidates.append(Path(tempfile.gettempdir()) / "pycutfem_jit" / "cpp")

        last_err: OSError | None = None
        for candidate in candidates:
            cache_dir = candidate.expanduser().resolve()
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                probe = cache_dir / f".pycutfem_write_probe_{uuid.uuid4().hex}"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink(missing_ok=True)
                return cache_dir
            except OSError as err:
                last_err = err
                continue

        msg = "Could not create a writable C++ JIT cache directory. Tried:\n" + "\n".join(
            f"  - {Path(p).expanduser()}" for p in candidates
        )
        raise OSError(msg) from last_err

    # ------------------------------------------------------------------
    def get_kernel(self, ir_sequence: list, codegen, mesh_sig=None):
        """
        Build (or load) a kernel for the given IR sequence and return
        (kernel_fn, param_order list, active_fields list).
        """
        ir_hash = KernelCache._hash_ir(ir_sequence, mesh_sig)
        compile_tag = get_compile_mode_tag(ir_len=len(ir_sequence))
        cache_key = f"{ir_hash}:{compile_tag}"

        if cache_key in self.in_memory_cache:
            return self.in_memory_cache[cache_key]

        module_name = f"_pycutfem_cpp_kernel_{ir_hash}_{compile_tag}"
        source_file = self._cache_dir / f"{module_name}.cpp"
        built_module = self._cache_dir / f"{module_name}{self._ext_suffix}"
        digest_file = self._cache_dir / f"{module_name}.sha256"
        lock_file = self._cache_dir / f"{module_name}.lock"

        with _module_build_lock(lock_file):
            if cache_key in self.in_memory_cache:
                return self.in_memory_cache[cache_key]

            # Generate source if needed (or if stale ABI is detected).
            regenerate = True
            if source_file.exists():
                try:
                    text = source_file.read_text(encoding="utf-8")
                except OSError:
                    text = ""
                regenerate = f'CODEGEN_ABI") = "{CODEGEN_ABI_CPP}"' not in text

            if regenerate:
                src, _, param_order = codegen.generate_source(
                    ir_sequence, "kernel", module_name=module_name
                )
                source_file.write_text(src, encoding="utf-8")
                built_module.unlink(missing_ok=True)
                digest_file.unlink(missing_ok=True)
            else:
                param_order = None

            try:
                source_digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
            except OSError:
                source_digest = ""

            force_reload = False
            digest_matches = False
            try:
                digest_matches = digest_file.read_text(encoding="utf-8").strip() == source_digest
            except OSError:
                digest_matches = False

            if built_module.exists() and not digest_matches:
                built_module.unlink(missing_ok=True)
                force_reload = True

            # Compile when the shared object is missing.
            if not built_module.exists():
                include_dirs = list(codegen.include_dirs)
                compile_extension(
                    module_name,
                    source_file,
                    self._cache_dir,
                    include_dirs=include_dirs,
                    compile_mode=compile_tag,
                )
                if source_digest:
                    digest_file.write_text(source_digest, encoding="utf-8")
                force_reload = True

            try:
                module = self._import_module(module_name, built_module, force_reload=force_reload)
            except Exception as exc:
                if not _looks_like_incomplete_binary_import(exc):
                    raise
                built_module.unlink(missing_ok=True)
                digest_file.unlink(missing_ok=True)
                compile_extension(
                    module_name,
                    source_file,
                    self._cache_dir,
                    include_dirs=list(codegen.include_dirs),
                    compile_mode=compile_tag,
                )
                if source_digest:
                    digest_file.write_text(source_digest, encoding="utf-8")
                module = self._import_module(module_name, built_module, force_reload=True)

            if getattr(module, "CODEGEN_ABI", None) != CODEGEN_ABI_CPP:
                # Stale ABI – rebuild and reload once.
                source_file.unlink(missing_ok=True)
                built_module.unlink(missing_ok=True)
                digest_file.unlink(missing_ok=True)
                src, _, param_order = codegen.generate_source(
                    ir_sequence, "kernel", module_name=module_name
                )
                source_file.write_text(src, encoding="utf-8")
                source_digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
                compile_extension(
                    module_name,
                    source_file,
                    self._cache_dir,
                    include_dirs=list(codegen.include_dirs),
                    compile_mode=compile_tag,
                )
                digest_file.write_text(source_digest, encoding="utf-8")
                module = self._import_module(module_name, built_module, force_reload=True)

        param_order = (
            list(getattr(module, "PARAM_ORDER"))
            if hasattr(module, "PARAM_ORDER")
            else param_order
            or []
        )
        active_fields = (
            list(getattr(module, "ACTIVE_FIELDS"))
            if hasattr(module, "ACTIVE_FIELDS")
            else list(getattr(codegen, "active_fields", []) or [])
        )

        kernel_fn = getattr(module, codegen.kernel_export_name("kernel"))
        self.in_memory_cache[cache_key] = (kernel_fn, param_order, active_fields)
        return kernel_fn, param_order, active_fields

    # ------------------------------------------------------------------
    @staticmethod
    def _import_module(modname: str, path: Path, *, force_reload: bool = False):
        if force_reload:
            sys.modules.pop(modname, None)
        else:
            cached = sys.modules.get(modname)
            if cached is not None:
                return cached
        spec = importlib.util.spec_from_file_location(modname, path)
        if not spec or not spec.loader:  # pragma: no cover - safety net
            raise ImportError(f"Could not load spec for {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        try:
            spec.loader.exec_module(mod)  # type: ignore[arg-type]
        except Exception:
            # Avoid keeping partially-initialized modules around.
            sys.modules.pop(modname, None)
            raise
        return mod
