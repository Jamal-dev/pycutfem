from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pycutfem.mor import solve_native_online_gauss_newton
from pycutfem.mor import reduced_assembly as _mor_reduced


def _env_flag(name: str) -> bool:
    value = str(os.getenv(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _reduced_projection_backend() -> str:
    if _env_flag("PYCUTFEM_NIRB_REDUCED_CPP") or _env_flag("PYCUTFEM_NIRB_REDUCED_CPP_REQUIRED"):
        return "cpp"
    return "python"


def _env_choice(name: str, default: str, allowed: set[str]) -> str:
    value = str(os.getenv(name, default) or default).strip().lower()
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}.")
    return value


def reduced_fluid_online_backend() -> str:
    """Return the requested NIRB reduced online-loop backend."""

    return _env_choice("PYCUTFEM_NIRB_REDUCED_ONLINE_BACKEND", "python", {"python", "cpp"})


def reduced_fluid_gnat_step_backend() -> str:
    """Return the requested dense GNAT step backend for Python-loop solvers."""

    return _env_choice("PYCUTFEM_NIRB_GNAT_STEP_BACKEND", "python", {"python", "cpp"})


def reduced_fluid_cpp_backend_status() -> dict[str, object]:
    """Return status for the C++ reduced projection backend used by HROM."""

    enabled = _env_flag("PYCUTFEM_NIRB_REDUCED_CPP") or _env_flag("PYCUTFEM_NIRB_REDUCED_CPP_REQUIRED")
    required = _env_flag("PYCUTFEM_NIRB_REDUCED_CPP_REQUIRED")
    try:
        module = _mor_reduced._cpp_projection_module()
    except Exception as exc:  # pragma: no cover - depends on local compiler/toolchain
        if required:
            raise RuntimeError("PYCUTFEM_NIRB_REDUCED_CPP_REQUIRED=1 but the C++ reduced backend failed.") from exc
        return {
            "enabled": bool(enabled),
            "required": bool(required),
            "loaded": False,
            "backend": "python",
            "unavailable_reason": str(exc),
        }
    return {
        "enabled": bool(enabled),
        "required": bool(required),
        "loaded": True,
        "backend": "cpp" if enabled else "available",
        "module": getattr(module, "__name__", type(module).__name__),
        "module_file": getattr(module, "__file__", None),
    }


def reduced_fluid_online_backend_status() -> dict[str, object]:
    """Return status for the native reduced online-loop backend."""

    requested = reduced_fluid_online_backend()
    try:
        from pycutfem.mor.cpp_backend.online_gauss_newton import module

        mod = module()
    except Exception as exc:  # pragma: no cover - depends on local compiler/toolchain
        if requested == "cpp":
            raise RuntimeError("C++ reduced online backend was requested but failed to load.") from exc
        return {
            "requested": requested,
            "loaded": False,
            "backend": "python",
            "unavailable_reason": str(exc),
        }
    return {
        "requested": requested,
        "loaded": True,
        "backend": "cpp" if requested == "cpp" else "available",
        "module": getattr(mod, "__name__", type(mod).__name__),
        "module_file": getattr(mod, "__file__", None),
    }


def _require_reduced_projection_backend() -> str:
    backend = _reduced_projection_backend()
    if backend == "cpp":
        # Force an early import so required-C++ runs fail before timing a Python
        # reduced projection path by accident.
        reduced_fluid_cpp_backend_status()
    return backend


@dataclass(frozen=True)
class ReducedFluidSystem:
    coefficients: np.ndarray
    residual: np.ndarray
    tangent: np.ndarray
    residual_norm: float
    metadata: Mapping[str, object]

    def newton_step(self, *, rcond: float | None = None) -> np.ndarray:
        step, *_ = np.linalg.lstsq(self.tangent, -self.residual, rcond=rcond)
        return np.asarray(step, dtype=float).reshape(-1)


@dataclass(frozen=True)
class ReducedFluidSolveResult:
    coefficients: np.ndarray
    residual_norm: float
    iterations: int
    converged: bool
    trajectory: tuple[dict[str, float], ...]
    reaction_coefficients: np.ndarray | None = None
    metadata: Mapping[str, object] | None = None


ReducedFluidAssembler = Callable[[np.ndarray], ReducedFluidSystem]
ReducedReactionEvaluator = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ReducedFluidNativeOnlineSpec:
    """NIRB-facing specification for the MOR native C++ online GN/GNAT driver."""

    residual_metadata_capsule: Any
    residual_param_order: Sequence[str]
    residual_static_args: Mapping[str, Any]
    tangent_metadata_capsule: Any
    tangent_param_order: Sequence[str]
    tangent_static_args: Mapping[str, Any]
    trial_basis: np.ndarray
    offset: np.ndarray
    row_dofs: np.ndarray
    coefficient_arg_names: Sequence[str] | None = None
    element_weights: np.ndarray | None = None
    row_weights: np.ndarray | None = None
    gnat_lift: np.ndarray | None = None
    residual_state_updates: Sequence[Mapping[str, Any]] | None = None
    tangent_state_updates: Sequence[Mapping[str, Any]] | None = None
    objective: str = "sampled_lspg"

    def solve(
        self,
        initial_coefficients: np.ndarray,
        *,
        max_iterations: int = 8,
        residual_tol: float = 1.0e-10,
        step_tol: float = 1.0e-12,
        damping: float = 0.0,
        adaptive_damping: bool = False,
        line_search: bool = False,
        max_line_search: int = 6,
        sufficient_decrease: float = 1.0e-4,
    ) -> ReducedFluidSolveResult:
        result = solve_native_online_gauss_newton(
            residual_metadata_capsule=self.residual_metadata_capsule,
            residual_param_order=self.residual_param_order,
            residual_static_args=self.residual_static_args,
            tangent_metadata_capsule=self.tangent_metadata_capsule,
            tangent_param_order=self.tangent_param_order,
            tangent_static_args=self.tangent_static_args,
            trial_basis=self.trial_basis,
            offset=self.offset,
            initial_coefficients=initial_coefficients,
            row_dofs=self.row_dofs,
            coefficient_arg_names=self.coefficient_arg_names,
            element_weights=self.element_weights,
            row_weights=self.row_weights,
            gnat_lift=self.gnat_lift,
            residual_state_updates=self.residual_state_updates,
            tangent_state_updates=self.tangent_state_updates,
            max_iterations=max_iterations,
            residual_tol=residual_tol,
            step_tol=step_tol,
            damping=damping,
            adaptive_damping=adaptive_damping,
            line_search=line_search,
            max_line_search=max_line_search,
            sufficient_decrease=sufficient_decrease,
        )
        trajectory = tuple(
            {
                "iteration": float(idx + 1),
                "residual_norm": float(norm),
                "step_norm": float(result.step_norm_history[idx]) if idx < result.step_norm_history.size else 0.0,
                "line_search_alpha": (
                    float(result.line_search_alpha_history[idx])
                    if idx < result.line_search_alpha_history.size
                    else 0.0
                ),
            }
            for idx, norm in enumerate(result.residual_norm_history)
        )
        return ReducedFluidSolveResult(
            coefficients=np.asarray(result.coefficients, dtype=float).reshape(-1),
            residual_norm=float(result.residual_norm),
            iterations=int(result.iterations),
            converged=bool(result.converged),
            trajectory=trajectory,
            metadata={
                "backend": result.backend,
                "objective": str(self.objective),
                "linear_solver": result.linear_solver,
                "damping_history": result.damping_history.tolist(),
                "rejected_step_count": int(result.rejected_step_count),
                "timing_counters": dict(result.timing_counters),
            },
        )


def solve_reduced_fluid_native_online(
    spec: ReducedFluidNativeOnlineSpec,
    initial_coefficients: np.ndarray,
    **kwargs,
) -> ReducedFluidSolveResult:
    """Solve a reduced NIRB fluid problem through the MOR native online driver."""

    return spec.solve(initial_coefficients, **kwargs)


@dataclass(frozen=True)
class SampledFluidStateDecoder:
    """Decode reduced fluid coefficients on a sampled element stencil."""

    basis: np.ndarray
    offset: np.ndarray
    element_ids: np.ndarray
    field_dofs: Mapping[str, np.ndarray]
    field_local_positions: Mapping[str, np.ndarray]
    element_field_dofs: Mapping[str, np.ndarray]
    element_field_local_positions: Mapping[str, np.ndarray]
    field_sizes: Mapping[str, int]

    @classmethod
    def from_sample_elements(
        cls,
        *,
        dh,
        basis: np.ndarray,
        offset: np.ndarray,
        element_ids: np.ndarray,
        fields: tuple[str, ...] = ("ux", "uy", "p"),
    ) -> "SampledFluidStateDecoder":
        trial_basis = np.asarray(basis, dtype=float)
        base = np.asarray(offset, dtype=float).reshape(-1)
        if trial_basis.ndim != 2:
            raise ValueError("sampled fluid basis must be a 2-D array.")
        total_dofs = int(getattr(dh, "total_dofs"))
        if int(trial_basis.shape[0]) != total_dofs:
            raise ValueError("sampled fluid basis shape is incompatible with the dof handler.")
        if int(base.size) != total_dofs:
            raise ValueError("sampled fluid offset size is incompatible with the dof handler.")
        eids = np.asarray(element_ids, dtype=int).reshape(-1)
        if np.any(eids < 0):
            raise ValueError("sampled fluid element_ids contain negative entries.")

        field_dofs: dict[str, np.ndarray] = {}
        field_positions: dict[str, np.ndarray] = {}
        element_field_dofs: dict[str, np.ndarray] = {}
        element_field_positions: dict[str, np.ndarray] = {}
        field_sizes: dict[str, int] = {}
        for field_name in fields:
            name = str(field_name)
            element_map = np.asarray(dh.element_maps[name], dtype=int)
            if eids.size and np.any(eids >= int(element_map.shape[0])):
                raise ValueError(f"sampled fluid element_ids are out of range for field {name!r}.")
            local_map = np.asarray(element_map[eids], dtype=int) if eids.size else np.zeros((0, 0), dtype=int)
            ids = np.unique(local_map.reshape(-1)) if local_map.size else np.zeros(0, dtype=int)
            ids = ids[ids >= 0]
            if ids.size and np.any(ids >= total_dofs):
                raise ValueError(f"sampled fluid field {name!r} contains out-of-range dofs.")
            field_dofs[name] = ids.astype(int, copy=False)
            element_field_dofs[name] = local_map.astype(int, copy=False)

            full_field = np.asarray(dh.get_field_slice(name), dtype=int).reshape(-1)
            field_sizes[name] = int(full_field.size)
            lookup = np.full(total_dofs, -1, dtype=int)
            lookup[full_field] = np.arange(int(full_field.size), dtype=int)
            if ids.size and np.any(lookup[ids] < 0):
                raise RuntimeError(f"Unable to locate sampled {name!r} dofs in the field-local layout.")
            field_positions[name] = lookup[ids]
            element_field_positions[name] = (
                lookup[local_map].astype(int, copy=False) if local_map.size else np.zeros(local_map.shape, dtype=int)
            )

        return cls(
            basis=trial_basis,
            offset=base,
            element_ids=eids,
            field_dofs=field_dofs,
            field_local_positions=field_positions,
            element_field_dofs=element_field_dofs,
            element_field_local_positions=element_field_positions,
            field_sizes=field_sizes,
        )

    @property
    def n_modes(self) -> int:
        return int(self.basis.shape[1])

    def _check_coefficients(self, coefficients: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(coefficients, dtype=float).reshape(-1)
        if int(coeffs.size) != self.n_modes:
            raise ValueError(f"expected {self.n_modes} reduced fluid coefficients, got {coeffs.size}.")
        if not np.all(np.isfinite(coeffs)):
            raise ValueError("reduced fluid coefficients contain non-finite values.")
        return coeffs

    def values_on_dofs(self, dofs: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        coeffs = self._check_coefficients(coefficients)
        ids = np.asarray(dofs, dtype=int).reshape(-1)
        if ids.size == 0:
            return np.zeros(0, dtype=float)
        return _mor_reduced.decode_values_on_dofs(
            offset=self.offset,
            basis=self.basis,
            dofs=ids,
            coefficients=coeffs,
            backend=_require_reduced_projection_backend(),
        )

    def field_values(self, field_name: str, coefficients: np.ndarray) -> np.ndarray:
        return self.values_on_dofs(self.field_dofs[str(field_name)], coefficients)

    def element_field_values(self, field_name: str, coefficients: np.ndarray) -> np.ndarray:
        coeffs = self._check_coefficients(coefficients)
        local_map = np.asarray(self.element_field_dofs[str(field_name)], dtype=int)
        if local_map.size == 0:
            return np.zeros(local_map.shape, dtype=float)
        return _mor_reduced.decode_element_values(
            offset=self.offset,
            basis=self.basis,
            local_map=local_map,
            coefficients=coeffs,
            backend=_require_reduced_projection_backend(),
        )

    def element_local_values(
        self,
        coefficients: np.ndarray,
        *,
        fluid_prev_step_u: np.ndarray | None = None,
        fluid_a_prev_stage: np.ndarray | None = None,
        bossak: Mapping[str, float] | None = None,
    ) -> dict[str, np.ndarray]:
        """Decode sampled element-local fluid values from coefficients."""

        values = {
            "ux": self.element_field_values("ux", coefficients),
            "uy": self.element_field_values("uy", coefficients),
            "p": self.element_field_values("p", coefficients),
        }
        if fluid_prev_step_u is None and fluid_a_prev_stage is None and bossak is None:
            return values
        if fluid_prev_step_u is None or fluid_a_prev_stage is None or bossak is None:
            raise ValueError("Bossak acceleration local decode requires prev velocity, prev acceleration, and bossak.")
        prev_u = np.asarray(fluid_prev_step_u, dtype=float).reshape(-1)
        prev_a = np.asarray(fluid_a_prev_stage, dtype=float).reshape(-1)
        ma0 = float(bossak["ma0"])
        ma2 = float(bossak["ma2"])
        ux_pos = np.asarray(self.element_field_local_positions["ux"], dtype=int)
        uy_pos = np.asarray(self.element_field_local_positions["uy"], dtype=int)
        ux_count = int(self.field_sizes["ux"])
        if int(prev_u.size) < ux_count or int(prev_a.size) < ux_count:
            raise ValueError("Fluid velocity history is too small for sampled ux positions.")
        uy_offset = ux_count
        if uy_pos.size and (np.any(uy_offset + uy_pos >= int(prev_u.size)) or np.any(uy_offset + uy_pos >= int(prev_a.size))):
            raise ValueError("Fluid velocity history is too small for sampled uy positions.")
        values["ux_prev"] = prev_u[ux_pos]
        values["uy_prev"] = prev_u[uy_offset + uy_pos]
        values["ax_prev"] = prev_a[ux_pos]
        values["ay_prev"] = prev_a[uy_offset + uy_pos]
        values["ax_curr"] = ma0 * (values["ux"] - values["ux_prev"]) + ma2 * values["ax_prev"]
        values["ay_curr"] = ma0 * (values["uy"] - values["uy_prev"]) + ma2 * values["ay_prev"]
        return values

    def bossak_acceleration_field_values(
        self,
        field_name: str,
        coefficients: np.ndarray,
        *,
        fluid_prev_step_u: np.ndarray,
        fluid_a_prev_stage: np.ndarray,
        ma0: float,
        ma2: float,
        velocity_field_offsets: Mapping[str, int],
    ) -> np.ndarray:
        values = self.field_values(str(field_name), coefficients)
        positions = np.asarray(self.field_local_positions[str(field_name)], dtype=int).reshape(-1)
        offset = int(velocity_field_offsets[str(field_name)])
        prev_u = np.asarray(fluid_prev_step_u, dtype=float).reshape(-1)
        prev_a = np.asarray(fluid_a_prev_stage, dtype=float).reshape(-1)
        idx = offset + positions
        if idx.size and (np.any(idx < 0) or np.any(idx >= int(prev_u.size)) or np.any(idx >= int(prev_a.size))):
            raise ValueError("fluid velocity history shape does not match sampled field layout.")
        return float(ma0) * (values - prev_u[idx]) + float(ma2) * prev_a[idx]


def validate_element_weights(
    element_count: int,
    element_weights: np.ndarray | None,
    *,
    context: str = "sampled reduced fluid",
) -> np.ndarray | None:
    """Validate optional empirical-cubature weights for element-local blocks."""

    return _mor_reduced.validate_element_weights(
        int(element_count),
        element_weights,
        context=context,
    )


def _validate_local_blocks(
    *,
    K_elem: np.ndarray,
    vector_elem: np.ndarray,
    gdofs_map: np.ndarray,
    trial_basis: np.ndarray,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _mor_reduced.validate_local_blocks(
        K_elem=K_elem,
        vector_elem=vector_elem,
        gdofs_map=gdofs_map,
        trial_basis=trial_basis,
        context=context,
    )


def sampled_lspg_element_contributions_from_local_blocks(
    *,
    K_elem: np.ndarray,
    raw_rhs_elem: np.ndarray,
    gdofs_map: np.ndarray,
    row_dofs: np.ndarray,
    trial_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project element-local fluid blocks into sampled LSPG rows.

    ``raw_rhs_elem`` follows the local system convention used by
    ``FluidDVMSLocalVelocityContributionOperator`` after the driver converts
    ``local.F_elem`` to the raw RHS sign. The returned residual uses the
    Newton residual sign, matching the existing sampled-LSPG verifier.
    """

    return _mor_reduced.sampled_lspg_element_contributions_from_local_blocks(
        K_elem=K_elem,
        raw_rhs_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        row_dofs=row_dofs,
        trial_basis=trial_basis,
        backend=_require_reduced_projection_backend(),
    )


def sampled_lspg_rows_from_local_blocks(
    *,
    K_elem: np.ndarray,
    raw_rhs_elem: np.ndarray,
    gdofs_map: np.ndarray,
    row_dofs: np.ndarray,
    trial_basis: np.ndarray,
    element_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum weighted element-local sampled LSPG rows without full-space scatter."""

    return _mor_reduced.sampled_lspg_rows_from_local_blocks(
        K_elem=K_elem,
        raw_rhs_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        row_dofs=row_dofs,
        trial_basis=trial_basis,
        element_weights=element_weights,
        backend=_require_reduced_projection_backend(),
    )


def sampled_galerkin_element_contributions_from_local_blocks(
    *,
    K_elem: np.ndarray,
    residual_elem: np.ndarray,
    gdofs_map: np.ndarray,
    trial_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project element-local residual/tangent blocks into reduced Galerkin space."""

    return _mor_reduced.sampled_galerkin_element_contributions_from_local_blocks(
        K_elem=K_elem,
        residual_elem=residual_elem,
        gdofs_map=gdofs_map,
        trial_basis=trial_basis,
        backend=_require_reduced_projection_backend(),
    )


def sampled_galerkin_reduced_system_from_local_blocks(
    *,
    K_elem: np.ndarray,
    residual_elem: np.ndarray,
    gdofs_map: np.ndarray,
    trial_basis: np.ndarray,
    element_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum weighted element-local Galerkin residual/tangent contributions."""

    return _mor_reduced.sampled_galerkin_reduced_system_from_local_blocks(
        K_elem=K_elem,
        residual_elem=residual_elem,
        gdofs_map=gdofs_map,
        trial_basis=trial_basis,
        element_weights=element_weights,
        backend=_require_reduced_projection_backend(),
    )


def constrained_reaction_rows_from_local_blocks(
    *,
    raw_rhs_elem: np.ndarray,
    gdofs_map: np.ndarray,
    constrained_row_dofs: np.ndarray,
    element_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate constrained-row interface reactions without full scatter.

    ``raw_rhs_elem`` uses the local-system RHS convention.  This mirrors the
    Example 2 Kratos-style reaction reconstruction, where the constrained-row
    reaction is ``-rhs`` after element RHS accumulation.
    """

    return _mor_reduced.constrained_reaction_rows_from_local_blocks(
        raw_rhs_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        constrained_row_dofs=constrained_row_dofs,
        element_weights=element_weights,
        backend=_require_reduced_projection_backend(),
    )


def reduced_reaction_from_local_blocks(
    *,
    raw_rhs_elem: np.ndarray,
    gdofs_map: np.ndarray,
    constrained_row_dofs: np.ndarray,
    row_to_reduced_load: np.ndarray,
    element_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Map constrained-row local reaction contributions to load coordinates."""

    return _mor_reduced.reduced_reaction_from_local_blocks(
        raw_rhs_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        constrained_row_dofs=constrained_row_dofs,
        row_to_reduced_load=row_to_reduced_load,
        element_weights=element_weights,
        backend=_require_reduced_projection_backend(),
    )


@dataclass
class ReducedFluidDVMSOperator:
    """Coefficient-only reduced fluid solve shell.

    The assembler callback is deliberately coefficient-local: it must return
    a reduced residual and tangent without writing global `u_k/p_k` fields.
    Element-local ALE-DVMS kernels can be plugged in behind this interface.
    """

    n_modes: int
    assembler: ReducedFluidAssembler
    reaction_evaluator: ReducedReactionEvaluator | None = None
    max_iterations: int = 8
    residual_tol: float = 1.0e-10
    step_tol: float = 1.0e-12
    line_search: bool = False
    max_line_search: int = 6
    sufficient_decrease: float = 1.0e-4

    def _validate_coefficients(self, coefficients: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(coefficients, dtype=float).reshape(-1)
        if int(coeffs.size) != int(self.n_modes):
            raise ValueError(f"expected {self.n_modes} reduced fluid coefficients, got {coeffs.size}.")
        if not np.all(np.isfinite(coeffs)):
            raise ValueError("reduced fluid coefficients contain non-finite values.")
        return coeffs

    def assemble(self, coefficients: np.ndarray) -> ReducedFluidSystem:
        coeffs = self._validate_coefficients(coefficients)
        system = self.assembler(coeffs.copy())
        residual = np.asarray(system.residual, dtype=float).reshape(-1)
        tangent = np.asarray(system.tangent, dtype=float)
        if residual.shape != (int(self.n_modes),):
            raise ValueError("reduced residual size must match n_modes.")
        if tangent.shape != (int(self.n_modes), int(self.n_modes)):
            raise ValueError("reduced tangent shape must be (n_modes, n_modes).")
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(tangent)):
            raise RuntimeError("reduced fluid assembler returned non-finite values.")
        return ReducedFluidSystem(
            coefficients=coeffs.copy(),
            residual=residual,
            tangent=tangent,
            residual_norm=float(np.linalg.norm(residual)),
            metadata=dict(system.metadata),
        )

    def solve(self, initial_coefficients: np.ndarray) -> ReducedFluidSolveResult:
        coeffs = self._validate_coefficients(initial_coefficients).copy()
        trajectory: list[dict[str, float]] = []
        converged = False
        last_norm = float("inf")

        for iteration in range(1, max(1, int(self.max_iterations)) + 1):
            system = self.assemble(coeffs)
            last_norm = float(system.residual_norm)
            if last_norm <= float(self.residual_tol):
                converged = True
                trajectory.append(
                    {
                        "iteration": float(iteration),
                        "residual_norm": float(last_norm),
                        "step_norm": 0.0,
                        "line_search_alpha": 0.0,
                    }
                )
                break
            step = system.newton_step()
            if not np.all(np.isfinite(step)):
                raise RuntimeError("reduced fluid solve produced a non-finite Newton step.")
            step_norm = float(np.linalg.norm(step))
            alpha = 1.0

            if bool(self.line_search):
                accepted = coeffs + step
                accepted_norm = float("inf")
                for search_iter in range(max(1, int(self.max_line_search))):
                    trial_alpha = 0.5**search_iter
                    trial = coeffs + float(trial_alpha) * step
                    trial_norm = float(self.assemble(trial).residual_norm)
                    if trial_norm < accepted_norm:
                        accepted = trial
                        accepted_norm = trial_norm
                        alpha = float(trial_alpha)
                    if trial_norm <= (1.0 - float(self.sufficient_decrease) * float(trial_alpha)) * last_norm:
                        break
                coeffs = np.asarray(accepted, dtype=float).reshape(-1)
                last_norm = accepted_norm
            else:
                coeffs = coeffs + step

            trajectory.append(
                {
                    "iteration": float(iteration),
                    "residual_norm": float(last_norm),
                    "step_norm": float(step_norm),
                    "line_search_alpha": float(alpha),
                }
            )
            if step_norm <= float(self.step_tol) * max(1.0, float(np.linalg.norm(coeffs))):
                converged = last_norm <= float(self.residual_tol)
                break

        if self.reaction_evaluator is None:
            reaction = None
        else:
            reaction = np.asarray(self.reaction_evaluator(coeffs.copy()), dtype=float).reshape(-1)
            if not np.all(np.isfinite(reaction)):
                raise RuntimeError("reduced reaction evaluator returned non-finite values.")

        return ReducedFluidSolveResult(
            coefficients=np.asarray(coeffs, dtype=float).reshape(-1),
            residual_norm=float(last_norm),
            iterations=int(len(trajectory)),
            converged=bool(converged),
            trajectory=tuple(trajectory),
            reaction_coefficients=reaction,
        )


__all__ = [
    "ReducedFluidDVMSOperator",
    "ReducedFluidNativeOnlineSpec",
    "ReducedFluidSolveResult",
    "ReducedFluidSystem",
    "SampledFluidStateDecoder",
    "constrained_reaction_rows_from_local_blocks",
    "reduced_fluid_gnat_step_backend",
    "reduced_fluid_online_backend",
    "reduced_fluid_online_backend_status",
    "reduced_reaction_from_local_blocks",
    "reduced_fluid_cpp_backend_status",
    "sampled_galerkin_element_contributions_from_local_blocks",
    "sampled_galerkin_reduced_system_from_local_blocks",
    "sampled_lspg_element_contributions_from_local_blocks",
    "sampled_lspg_rows_from_local_blocks",
    "solve_reduced_fluid_native_online",
    "validate_element_weights",
]
