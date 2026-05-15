from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def bossak_displacement_kinematics(
    *,
    q_curr: np.ndarray,
    q_prev: np.ndarray,
    v_prev: np.ndarray,
    a_prev: np.ndarray,
    dt: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    dt_value = max(float(dt), 1.0e-14)
    gamma = 0.5 - float(alpha)
    beta = 0.25 * (1.0 - float(alpha)) ** 2
    q = np.asarray(q_curr, dtype=float).reshape(-1)
    q_n = np.asarray(q_prev, dtype=float).reshape(-1)
    v_n = np.asarray(v_prev, dtype=float).reshape(-1)
    a_n = np.asarray(a_prev, dtype=float).reshape(-1)
    if not (q.shape == q_n.shape == v_n.shape == a_n.shape):
        raise ValueError("Bossak kinematic vectors must have matching sizes.")
    a_curr = (q - q_n - dt_value * v_n - dt_value * dt_value * (0.5 - beta) * a_n) / (
        beta * dt_value * dt_value
    )
    v_curr = v_n + dt_value * ((1.0 - gamma) * a_n + gamma * a_curr)
    return np.asarray(v_curr, dtype=float), np.asarray(a_curr, dtype=float)


@dataclass(frozen=True)
class ReducedMeshMotionState:
    q_prev: np.ndarray
    q_prev2: np.ndarray
    v_prev: np.ndarray
    a_prev: np.ndarray

    def __post_init__(self) -> None:
        q_prev = np.asarray(self.q_prev, dtype=float).reshape(-1)
        q_prev2 = np.asarray(self.q_prev2, dtype=float).reshape(-1)
        v_prev = np.asarray(self.v_prev, dtype=float).reshape(-1)
        a_prev = np.asarray(self.a_prev, dtype=float).reshape(-1)
        if not (q_prev.shape == q_prev2.shape == v_prev.shape == a_prev.shape):
            raise ValueError("reduced mesh history vectors must have matching sizes.")
        object.__setattr__(self, "q_prev", q_prev)
        object.__setattr__(self, "q_prev2", q_prev2)
        object.__setattr__(self, "v_prev", v_prev)
        object.__setattr__(self, "a_prev", a_prev)

    @classmethod
    def zeros(cls, n_modes: int) -> "ReducedMeshMotionState":
        zeros = np.zeros(int(n_modes), dtype=float)
        return cls(q_prev=zeros.copy(), q_prev2=zeros.copy(), v_prev=zeros.copy(), a_prev=zeros.copy())

    def accept(self, q_curr: np.ndarray, v_curr: np.ndarray, a_curr: np.ndarray) -> "ReducedMeshMotionState":
        return ReducedMeshMotionState(
            q_prev=np.asarray(q_curr, dtype=float).reshape(-1),
            q_prev2=np.asarray(self.q_prev, dtype=float).reshape(-1),
            v_prev=np.asarray(v_curr, dtype=float).reshape(-1),
            a_prev=np.asarray(a_curr, dtype=float).reshape(-1),
        )


@dataclass(frozen=True)
class ReducedMeshMotionResult:
    q: np.ndarray
    v: np.ndarray
    a: np.ndarray
    residual_norm: float


@dataclass(frozen=True)
class ReducedMeshMotionOperator:
    """Intrusive reduced structural-similarity ALE mesh solve.

    The online solve is the small system
    ``K_mm_r q_m = -K_mg_r d_gamma_r``.  All full-order lifting and stiffness
    projection must happen offline; this class stores only reduced matrices.
    """

    stiffness: np.ndarray
    interface_coupling: np.ndarray
    dt: float
    bossak_alpha: float = -0.3

    def __post_init__(self) -> None:
        stiffness = np.asarray(self.stiffness, dtype=float)
        coupling = np.asarray(self.interface_coupling, dtype=float)
        if stiffness.ndim != 2 or stiffness.shape[0] != stiffness.shape[1]:
            raise ValueError("reduced mesh stiffness must be square.")
        if coupling.ndim != 2 or coupling.shape[0] != stiffness.shape[0]:
            raise ValueError("interface coupling rows must match mesh modes.")
        if not np.all(np.isfinite(stiffness)) or not np.all(np.isfinite(coupling)):
            raise ValueError("reduced mesh matrices contain non-finite values.")
        object.__setattr__(self, "stiffness", stiffness)
        object.__setattr__(self, "interface_coupling", coupling)

    @property
    def n_modes(self) -> int:
        return int(self.stiffness.shape[0])

    @property
    def n_interface_modes(self) -> int:
        return int(self.interface_coupling.shape[1])

    def solve(self, interface_displacement: np.ndarray, history: ReducedMeshMotionState) -> ReducedMeshMotionResult:
        d_gamma = np.asarray(interface_displacement, dtype=float).reshape(-1)
        if int(d_gamma.size) != self.n_interface_modes:
            raise ValueError(f"expected {self.n_interface_modes} interface modes, got {d_gamma.size}.")
        if int(history.q_prev.size) != self.n_modes:
            raise ValueError("mesh history size does not match reduced mesh operator.")
        rhs = -(self.interface_coupling @ d_gamma)
        try:
            q = np.linalg.solve(self.stiffness, rhs)
        except np.linalg.LinAlgError:
            q, *_ = np.linalg.lstsq(self.stiffness, rhs, rcond=None)
        residual = self.stiffness @ q - rhs
        v, a = bossak_displacement_kinematics(
            q_curr=q,
            q_prev=history.q_prev,
            v_prev=history.v_prev,
            a_prev=history.a_prev,
            dt=float(self.dt),
            alpha=float(self.bossak_alpha),
        )
        return ReducedMeshMotionResult(
            q=np.asarray(q, dtype=float).reshape(-1),
            v=np.asarray(v, dtype=float).reshape(-1),
            a=np.asarray(a, dtype=float).reshape(-1),
            residual_norm=float(np.linalg.norm(residual)),
        )


@dataclass(frozen=True)
class ReducedMeshSampleEvaluator:
    """Decode reduced mesh fields on preselected elements/quadrature points."""

    value_basis: np.ndarray
    grad_basis: np.ndarray | None = None

    def __post_init__(self) -> None:
        value_basis = np.asarray(self.value_basis, dtype=float)
        if value_basis.ndim < 2:
            raise ValueError("value_basis must have at least two dimensions.")
        if not np.all(np.isfinite(value_basis)):
            raise ValueError("value_basis contains non-finite values.")
        object.__setattr__(self, "value_basis", value_basis)
        if self.grad_basis is not None:
            grad_basis = np.asarray(self.grad_basis, dtype=float)
            if grad_basis.ndim < 2 or grad_basis.shape[-1] != value_basis.shape[-1]:
                raise ValueError("grad_basis must use the same trailing mode count as value_basis.")
            if not np.all(np.isfinite(grad_basis)):
                raise ValueError("grad_basis contains non-finite values.")
            object.__setattr__(self, "grad_basis", grad_basis)

    @property
    def n_modes(self) -> int:
        return int(self.value_basis.shape[-1])

    def values(self, coefficients: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(coefficients, dtype=float).reshape(-1)
        if int(coeffs.size) != self.n_modes:
            raise ValueError(f"expected {self.n_modes} coefficients, got {coeffs.size}.")
        return np.tensordot(self.value_basis, coeffs, axes=([-1], [0]))

    def gradients(self, coefficients: np.ndarray) -> np.ndarray:
        if self.grad_basis is None:
            raise RuntimeError("no gradient sample basis was supplied.")
        coeffs = np.asarray(coefficients, dtype=float).reshape(-1)
        if int(coeffs.size) != self.n_modes:
            raise ValueError(f"expected {self.n_modes} coefficients, got {coeffs.size}.")
        return np.tensordot(self.grad_basis, coeffs, axes=([-1], [0]))


@dataclass(frozen=True)
class ReducedMeshDisplacementMap:
    """Data-reduced ALE displacement map.

    The map is used online as

    ``q_mesh = bias + linear @ q_interface``,

    followed by reconstruction in the mesh POD basis.  It is deliberately
    small and contains no full-order stiffness matrix or mesh solve state.
    """

    interface_basis: np.ndarray
    interface_mean: np.ndarray | None
    mesh_basis: np.ndarray
    mesh_mean: np.ndarray | None
    linear: np.ndarray
    bias: np.ndarray
    fluid_coords_ref: np.ndarray | None = None
    interface_coords_ref: np.ndarray | None = None
    source_path: str = ""
    ridge: float = 0.0
    training_steps: np.ndarray | None = None
    training_relative_errors: np.ndarray | None = None

    schema_version: int = 1

    def __post_init__(self) -> None:
        interface_basis = np.asarray(self.interface_basis, dtype=float)
        mesh_basis = np.asarray(self.mesh_basis, dtype=float)
        linear = np.asarray(self.linear, dtype=float)
        bias = np.asarray(self.bias, dtype=float).reshape(-1)
        if interface_basis.ndim != 2 or mesh_basis.ndim != 2:
            raise ValueError("reduced mesh map bases must be 2-D matrices.")
        if linear.shape != (mesh_basis.shape[1], interface_basis.shape[1]):
            raise ValueError("reduced mesh linear map shape is incompatible with the bases.")
        if bias.size != mesh_basis.shape[1]:
            raise ValueError("reduced mesh bias size must match mesh modes.")
        if not (
            np.all(np.isfinite(interface_basis))
            and np.all(np.isfinite(mesh_basis))
            and np.all(np.isfinite(linear))
            and np.all(np.isfinite(bias))
        ):
            raise ValueError("reduced mesh map contains non-finite values.")
        object.__setattr__(self, "interface_basis", interface_basis)
        object.__setattr__(self, "mesh_basis", mesh_basis)
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "bias", bias)
        if self.interface_mean is not None:
            interface_mean = np.asarray(self.interface_mean, dtype=float).reshape(-1)
            if interface_mean.size != interface_basis.shape[0]:
                raise ValueError("interface mean size must match interface basis rows.")
            if not np.all(np.isfinite(interface_mean)):
                raise ValueError("interface mean contains non-finite values.")
            object.__setattr__(self, "interface_mean", interface_mean)
        if self.mesh_mean is not None:
            mesh_mean = np.asarray(self.mesh_mean, dtype=float).reshape(-1)
            if mesh_mean.size != mesh_basis.shape[0]:
                raise ValueError("mesh mean size must match mesh basis rows.")
            if not np.all(np.isfinite(mesh_mean)):
                raise ValueError("mesh mean contains non-finite values.")
            object.__setattr__(self, "mesh_mean", mesh_mean)
        if self.fluid_coords_ref is not None:
            coords = np.asarray(self.fluid_coords_ref, dtype=float)
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError("fluid reference coordinates must have shape (n, 2).")
            object.__setattr__(self, "fluid_coords_ref", coords)
        if self.interface_coords_ref is not None:
            coords = np.asarray(self.interface_coords_ref, dtype=float)
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError("interface reference coordinates must have shape (n, 2).")
            object.__setattr__(self, "interface_coords_ref", coords)
        if self.training_steps is not None:
            object.__setattr__(self, "training_steps", np.asarray(self.training_steps, dtype=int).reshape(-1))
        if self.training_relative_errors is not None:
            errors = np.asarray(self.training_relative_errors, dtype=float).reshape(-1)
            if not np.all(np.isfinite(errors)):
                raise ValueError("training relative errors contain non-finite values.")
            object.__setattr__(self, "training_relative_errors", errors)

    @property
    def n_interface_modes(self) -> int:
        return int(self.interface_basis.shape[1])

    @property
    def n_mesh_modes(self) -> int:
        return int(self.mesh_basis.shape[1])

    @property
    def n_fluid_nodes(self) -> int:
        if self.mesh_basis.shape[0] % 2 != 0:
            raise ValueError("mesh basis row count must be even for 2-D nodal reconstruction.")
        return int(self.mesh_basis.shape[0] // 2)

    def interface_coefficients(self, interface_values: np.ndarray) -> np.ndarray:
        values = np.asarray(interface_values, dtype=float).reshape(-1)
        if int(values.size) != int(self.interface_basis.shape[0]):
            raise ValueError(
                f"expected {self.interface_basis.shape[0]} interface displacement values, got {values.size}."
            )
        centered = values if self.interface_mean is None else values - np.asarray(self.interface_mean, dtype=float)
        return np.asarray(self.interface_basis.T @ centered, dtype=float).reshape(-1)

    def mesh_coefficients_from_interface_values(self, interface_values: np.ndarray) -> np.ndarray:
        q_gamma = self.interface_coefficients(interface_values)
        q_mesh = np.asarray(self.bias + self.linear @ q_gamma, dtype=float).reshape(-1)
        if not np.all(np.isfinite(q_mesh)):
            raise RuntimeError("reduced mesh map produced non-finite coefficients.")
        return q_mesh

    def reconstruct_mesh_vector(self, mesh_coefficients: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(mesh_coefficients, dtype=float).reshape(-1)
        if int(coeffs.size) != self.n_mesh_modes:
            raise ValueError(f"expected {self.n_mesh_modes} mesh coefficients, got {coeffs.size}.")
        base = 0.0 if self.mesh_mean is None else np.asarray(self.mesh_mean, dtype=float)
        values = np.asarray(base + self.mesh_basis @ coeffs, dtype=float).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("reduced mesh reconstruction produced non-finite nodal values.")
        return values

    def predict_nodal_displacement(self, interface_values: np.ndarray) -> np.ndarray:
        values = self.reconstruct_mesh_vector(self.mesh_coefficients_from_interface_values(interface_values))
        return values.reshape(self.n_fluid_nodes, 2)

    def arrays(self) -> dict[str, np.ndarray]:
        payload: dict[str, np.ndarray] = {
            "schema_version": np.asarray(self.schema_version, dtype=int),
            "interface/basis": np.asarray(self.interface_basis, dtype=float),
            "interface/mean": (
                np.zeros((self.interface_basis.shape[0],), dtype=float)
                if self.interface_mean is None
                else np.asarray(self.interface_mean, dtype=float).reshape(-1)
            ),
            "interface/centered": np.asarray(self.interface_mean is not None, dtype=bool),
            "mesh/basis": np.asarray(self.mesh_basis, dtype=float),
            "mesh/mean": (
                np.zeros((self.mesh_basis.shape[0],), dtype=float)
                if self.mesh_mean is None
                else np.asarray(self.mesh_mean, dtype=float).reshape(-1)
            ),
            "mesh/centered": np.asarray(self.mesh_mean is not None, dtype=bool),
            "map/linear": np.asarray(self.linear, dtype=float),
            "map/bias": np.asarray(self.bias, dtype=float),
            "map/ridge": np.asarray(float(self.ridge), dtype=float),
            "metadata/source_path": np.asarray(str(self.source_path), dtype=np.str_),
        }
        if self.fluid_coords_ref is not None:
            payload["geometry/fluid_coords_ref"] = np.asarray(self.fluid_coords_ref, dtype=float)
        if self.interface_coords_ref is not None:
            payload["geometry/interface_coords_ref"] = np.asarray(self.interface_coords_ref, dtype=float)
        if self.training_steps is not None:
            payload["validation/training_steps"] = np.asarray(self.training_steps, dtype=int)
        if self.training_relative_errors is not None:
            payload["validation/training_relative_errors"] = np.asarray(
                self.training_relative_errors,
                dtype=float,
            )
        return payload

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **self.arrays())
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ReducedMeshDisplacementMap":
        source = Path(path)
        with np.load(source, allow_pickle=False) as raw:
            data = {key: np.asarray(raw[key]) for key in raw.files}
        version = int(np.asarray(data.get("schema_version", np.asarray(-1)), dtype=int).reshape(-1)[0])
        if version != 1:
            raise RuntimeError(f"unsupported reduced mesh displacement map schema_version={version}; expected 1.")
        interface_centered = bool(np.asarray(data.get("interface/centered", np.asarray(False))).reshape(-1)[0])
        mesh_centered = bool(np.asarray(data.get("mesh/centered", np.asarray(False))).reshape(-1)[0])
        return cls(
            interface_basis=np.asarray(data["interface/basis"], dtype=float),
            interface_mean=np.asarray(data["interface/mean"], dtype=float).reshape(-1) if interface_centered else None,
            mesh_basis=np.asarray(data["mesh/basis"], dtype=float),
            mesh_mean=np.asarray(data["mesh/mean"], dtype=float).reshape(-1) if mesh_centered else None,
            linear=np.asarray(data["map/linear"], dtype=float),
            bias=np.asarray(data["map/bias"], dtype=float).reshape(-1),
            fluid_coords_ref=(
                np.asarray(data["geometry/fluid_coords_ref"], dtype=float)
                if "geometry/fluid_coords_ref" in data
                else None
            ),
            interface_coords_ref=(
                np.asarray(data["geometry/interface_coords_ref"], dtype=float)
                if "geometry/interface_coords_ref" in data
                else None
            ),
            source_path=str(np.asarray(data.get("metadata/source_path", np.asarray(""))).reshape(-1)[0]),
            ridge=float(np.asarray(data.get("map/ridge", np.asarray(0.0)), dtype=float).reshape(-1)[0]),
            training_steps=(
                np.asarray(data["validation/training_steps"], dtype=int)
                if "validation/training_steps" in data
                else None
            ),
            training_relative_errors=(
                np.asarray(data["validation/training_relative_errors"], dtype=float)
                if "validation/training_relative_errors" in data
                else None
            ),
        )


def fit_reduced_mesh_displacement_map(
    *,
    interface_basis: np.ndarray,
    interface_mean: np.ndarray | None,
    interface_snapshots: np.ndarray,
    mesh_basis: np.ndarray,
    mesh_mean: np.ndarray | None,
    mesh_snapshots: np.ndarray,
    ridge: float = 0.0,
    fluid_coords_ref: np.ndarray | None = None,
    interface_coords_ref: np.ndarray | None = None,
    source_path: str = "",
    training_steps: np.ndarray | None = None,
) -> ReducedMeshDisplacementMap:
    """Fit a coefficient map from interface displacement to ALE displacement."""

    gamma_basis = np.asarray(interface_basis, dtype=float)
    mesh_basis_arr = np.asarray(mesh_basis, dtype=float)
    gamma = np.asarray(interface_snapshots, dtype=float)
    mesh = np.asarray(mesh_snapshots, dtype=float)
    if gamma.ndim != 2 or mesh.ndim != 2:
        raise ValueError("interface_snapshots and mesh_snapshots must be 2-D snapshot matrices.")
    if gamma.shape[1] != mesh.shape[1]:
        raise ValueError("interface and mesh snapshot counts must match.")
    gamma_mean = None if interface_mean is None else np.asarray(interface_mean, dtype=float).reshape(-1, 1)
    mesh_mean_arr = None if mesh_mean is None else np.asarray(mesh_mean, dtype=float).reshape(-1, 1)
    if gamma_basis.shape[0] != gamma.shape[0]:
        raise ValueError("interface basis row count must match interface snapshots.")
    if mesh_basis_arr.shape[0] != mesh.shape[0]:
        raise ValueError("mesh basis row count must match mesh snapshots.")
    gamma_work = gamma if gamma_mean is None else gamma - gamma_mean
    mesh_work = mesh if mesh_mean_arr is None else mesh - mesh_mean_arr
    gamma_coeffs = np.asarray(gamma_basis.T @ gamma_work, dtype=float)
    mesh_coeffs = np.asarray(mesh_basis_arr.T @ mesh_work, dtype=float)
    ones = np.ones((1, gamma_coeffs.shape[1]), dtype=float)
    design = np.vstack([gamma_coeffs, ones])
    regularization = float(max(float(ridge), 0.0))
    gram = design @ design.T
    if regularization > 0.0:
        penalty = np.eye(int(gram.shape[0]), dtype=float)
        penalty[-1, -1] = 0.0
        gram = gram + regularization * penalty
    rhs = design @ mesh_coeffs.T
    try:
        coeff = np.linalg.solve(gram, rhs).T
    except np.linalg.LinAlgError:
        coeff = np.linalg.lstsq(gram, rhs, rcond=None)[0].T
    linear = coeff[:, :-1]
    bias = coeff[:, -1]
    predicted_coeffs = bias[:, None] + linear @ gamma_coeffs
    mesh_base = 0.0 if mesh_mean_arr is None else mesh_mean_arr
    predicted_mesh = mesh_base + mesh_basis_arr @ predicted_coeffs
    errors = np.linalg.norm(predicted_mesh - mesh, axis=0) / np.maximum(np.linalg.norm(mesh, axis=0), 1.0e-30)
    return ReducedMeshDisplacementMap(
        interface_basis=gamma_basis,
        interface_mean=None if interface_mean is None else np.asarray(interface_mean, dtype=float).reshape(-1),
        mesh_basis=mesh_basis_arr,
        mesh_mean=None if mesh_mean is None else np.asarray(mesh_mean, dtype=float).reshape(-1),
        linear=linear,
        bias=bias,
        fluid_coords_ref=fluid_coords_ref,
        interface_coords_ref=interface_coords_ref,
        source_path=str(source_path),
        ridge=regularization,
        training_steps=training_steps,
        training_relative_errors=errors,
    )


__all__ = [
    "ReducedMeshDisplacementMap",
    "ReducedMeshMotionOperator",
    "ReducedMeshMotionResult",
    "ReducedMeshMotionState",
    "ReducedMeshSampleEvaluator",
    "bossak_displacement_kinematics",
    "fit_reduced_mesh_displacement_map",
]
