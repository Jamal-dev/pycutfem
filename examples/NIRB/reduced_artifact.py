from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


FULLY_REDUCED_ARTIFACT_SCHEMA_VERSION = 2


def _array(data: dict[str, np.ndarray], key: str, *, dtype=float, ndim: int | None = None) -> np.ndarray:
    if key not in data:
        raise KeyError(f"missing reduced artifact array {key!r}")
    arr = np.asarray(data[key], dtype=dtype)
    if ndim is not None and arr.ndim != int(ndim):
        raise ValueError(f"artifact array {key!r} must have ndim={ndim}, got {arr.ndim}")
    if dtype is float and not np.all(np.isfinite(arr)):
        raise ValueError(f"artifact array {key!r} contains non-finite values")
    return arr


def _optional_array(data: dict[str, np.ndarray], key: str, *, dtype=float) -> np.ndarray | None:
    if key not in data:
        return None
    return np.asarray(data[key], dtype=dtype)


@dataclass(frozen=True)
class FullyReducedNIRBArtifact:
    """Schema-v2 artifact for the fully reduced online FSI path.

    This is an intentionally flat `.npz` schema.  Keys use slash-separated
    prefixes to make ownership clear while staying compatible with NumPy's
    archive format.
    """

    interface_load_basis: np.ndarray
    interface_disp_basis: np.ndarray
    interface_mass: np.ndarray
    solid_to_interface_linear: np.ndarray
    solid_to_interface_quadratic: np.ndarray
    solid_to_interface_bias: np.ndarray
    mesh_stiffness: np.ndarray
    mesh_interface_coupling: np.ndarray
    fluid_basis: np.ndarray
    fluid_sample_elements: np.ndarray
    fluid_sample_rows: np.ndarray
    fluid_element_weights: np.ndarray
    fluid_row_weights: np.ndarray
    reaction_matrix: np.ndarray
    mesh_basis: np.ndarray | None = None
    mesh_mean: np.ndarray | None = None
    dvms_sample_layout: np.ndarray | None = None
    validation_training_steps: np.ndarray | None = None
    validation_heldout_steps: np.ndarray | None = None
    validation_error_budget: np.ndarray | None = None

    def __post_init__(self) -> None:
        load_basis = np.asarray(self.interface_load_basis, dtype=float)
        disp_basis = np.asarray(self.interface_disp_basis, dtype=float)
        if load_basis.ndim != 2 or disp_basis.ndim != 2:
            raise ValueError("interface bases must be 2D matrices.")
        if np.asarray(self.interface_mass).ndim not in {1, 2}:
            raise ValueError("interface mass must be a diagonal vector or a matrix.")
        linear = np.asarray(self.solid_to_interface_linear, dtype=float)
        quadratic = np.asarray(self.solid_to_interface_quadratic, dtype=float)
        bias = np.asarray(self.solid_to_interface_bias, dtype=float).reshape(-1)
        if linear.ndim != 2 or quadratic.ndim != 2:
            raise ValueError("solid interface maps must be 2D matrices.")
        if linear.shape[0] != disp_basis.shape[1] or quadratic.shape[0] != disp_basis.shape[1]:
            raise ValueError("solid interface map rows must match interface displacement modes.")
        if bias.size != disp_basis.shape[1]:
            raise ValueError("solid interface bias size must match interface displacement modes.")
        mesh_k = np.asarray(self.mesh_stiffness, dtype=float)
        mesh_g = np.asarray(self.mesh_interface_coupling, dtype=float)
        if mesh_k.ndim != 2 or mesh_k.shape[0] != mesh_k.shape[1]:
            raise ValueError("mesh stiffness must be square.")
        if mesh_g.shape != (mesh_k.shape[0], disp_basis.shape[1]):
            raise ValueError("mesh interface coupling shape is incompatible with mesh/interface modes.")
        if self.mesh_basis is not None:
            mesh_basis = np.asarray(self.mesh_basis, dtype=float)
            if mesh_basis.ndim != 2 or mesh_basis.shape[1] != mesh_k.shape[0]:
                raise ValueError("mesh basis columns must match reduced mesh modes.")
            if self.mesh_mean is not None and np.asarray(self.mesh_mean, dtype=float).reshape(-1).size != mesh_basis.shape[0]:
                raise ValueError("mesh mean size must match mesh basis rows.")
        fluid_basis = np.asarray(self.fluid_basis, dtype=float)
        if fluid_basis.ndim != 2:
            raise ValueError("fluid basis must be a 2D matrix.")
        reaction = np.asarray(self.reaction_matrix, dtype=float)
        if reaction.ndim != 2 or reaction.shape[1] != fluid_basis.shape[1]:
            raise ValueError("reaction matrix columns must match fluid modes.")
        if reaction.shape[0] != load_basis.shape[1]:
            raise ValueError("reaction matrix rows must match interface load modes.")

    def arrays(self) -> dict[str, np.ndarray]:
        payload = {
            "schema_version": np.asarray(FULLY_REDUCED_ARTIFACT_SCHEMA_VERSION, dtype=int),
            "interface/load_basis": np.asarray(self.interface_load_basis, dtype=float),
            "interface/disp_basis": np.asarray(self.interface_disp_basis, dtype=float),
            "interface/mass": np.asarray(self.interface_mass, dtype=float),
            "solid/to_interface_linear": np.asarray(self.solid_to_interface_linear, dtype=float),
            "solid/to_interface_quadratic": np.asarray(self.solid_to_interface_quadratic, dtype=float),
            "solid/to_interface_bias": np.asarray(self.solid_to_interface_bias, dtype=float),
            "mesh/K_mm_r": np.asarray(self.mesh_stiffness, dtype=float),
            "mesh/K_mg_r": np.asarray(self.mesh_interface_coupling, dtype=float),
            "fluid/basis": np.asarray(self.fluid_basis, dtype=float),
            "fluid/sample_elements": np.asarray(self.fluid_sample_elements, dtype=int),
            "fluid/sample_rows": np.asarray(self.fluid_sample_rows, dtype=int),
            "fluid/element_cubature_weights": np.asarray(self.fluid_element_weights, dtype=float),
            "fluid/row_weights": np.asarray(self.fluid_row_weights, dtype=float),
            "reaction/reduced_operator": np.asarray(self.reaction_matrix, dtype=float),
        }
        if self.mesh_basis is not None:
            payload["mesh/basis"] = np.asarray(self.mesh_basis, dtype=float)
        if self.mesh_mean is not None:
            payload["mesh/mean"] = np.asarray(self.mesh_mean, dtype=float)
        if self.dvms_sample_layout is not None:
            payload["dvms/sample_layout"] = np.asarray(self.dvms_sample_layout, dtype=int)
        if self.validation_training_steps is not None:
            payload["validation/training_steps"] = np.asarray(self.validation_training_steps, dtype=int)
        if self.validation_heldout_steps is not None:
            payload["validation/heldout_steps"] = np.asarray(self.validation_heldout_steps, dtype=int)
        if self.validation_error_budget is not None:
            payload["validation/error_budget"] = np.asarray(self.validation_error_budget, dtype=float)
        return payload

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **self.arrays())
        return target

    @classmethod
    def load(cls, path: str | Path) -> "FullyReducedNIRBArtifact":
        source = Path(path)
        with np.load(source, allow_pickle=False) as raw:
            data = {key: np.asarray(raw[key]) for key in raw.files}
        version = int(np.asarray(data.get("schema_version", np.asarray(-1)), dtype=int).reshape(-1)[0])
        if version != FULLY_REDUCED_ARTIFACT_SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported fully reduced artifact schema_version={version}; "
                f"expected {FULLY_REDUCED_ARTIFACT_SCHEMA_VERSION}."
            )
        return cls(
            interface_load_basis=_array(data, "interface/load_basis", ndim=2),
            interface_disp_basis=_array(data, "interface/disp_basis", ndim=2),
            interface_mass=_array(data, "interface/mass"),
            solid_to_interface_linear=_array(data, "solid/to_interface_linear", ndim=2),
            solid_to_interface_quadratic=_array(data, "solid/to_interface_quadratic", ndim=2),
            solid_to_interface_bias=_array(data, "solid/to_interface_bias"),
            mesh_stiffness=_array(data, "mesh/K_mm_r", ndim=2),
            mesh_interface_coupling=_array(data, "mesh/K_mg_r", ndim=2),
            mesh_basis=_optional_array(data, "mesh/basis"),
            mesh_mean=_optional_array(data, "mesh/mean"),
            fluid_basis=_array(data, "fluid/basis", ndim=2),
            fluid_sample_elements=_array(data, "fluid/sample_elements", dtype=int),
            fluid_sample_rows=_array(data, "fluid/sample_rows", dtype=int),
            fluid_element_weights=_array(data, "fluid/element_cubature_weights"),
            fluid_row_weights=_array(data, "fluid/row_weights"),
            reaction_matrix=_array(data, "reaction/reduced_operator", ndim=2),
            dvms_sample_layout=_optional_array(data, "dvms/sample_layout", dtype=int),
            validation_training_steps=_optional_array(data, "validation/training_steps", dtype=int),
            validation_heldout_steps=_optional_array(data, "validation/heldout_steps", dtype=int),
            validation_error_budget=_optional_array(data, "validation/error_budget"),
        )


__all__ = ["FULLY_REDUCED_ARTIFACT_SCHEMA_VERSION", "FullyReducedNIRBArtifact"]
