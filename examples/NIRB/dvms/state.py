from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is optional for this helper
    cKDTree = None

from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.integration.quadrature import volume as quadrature_volume
from pycutfem.state import QuadratureLayout, StateRegistry


@dataclass
class FluidDVMSState:
    sample_coords: np.ndarray
    sample_element_ids: np.ndarray
    sample_ref_coords: np.ndarray
    sample_ref_weights: np.ndarray
    quadrature_order: int
    cell_type: str
    old_subscale_velocity: np.ndarray
    predicted_subscale_velocity: np.ndarray
    momentum_projection: np.ndarray
    mass_projection: np.ndarray
    old_mass_residual: np.ndarray
    _sample_tree: object | None = None

    def __post_init__(self) -> None:
        self.sample_coords = np.asarray(self.sample_coords, dtype=float)
        if self.sample_coords.ndim != 2 or self.sample_coords.shape[1] != 2:
            raise ValueError("sample_coords must have shape (n, 2)")
        npts = int(self.sample_coords.shape[0])
        self.sample_element_ids = np.asarray(self.sample_element_ids, dtype=int).reshape(-1)
        if self.sample_element_ids.shape != (npts,):
            raise ValueError(f"sample_element_ids must have shape ({npts},), got {self.sample_element_ids.shape}")
        self.sample_ref_coords = np.asarray(self.sample_ref_coords, dtype=float)
        if self.sample_ref_coords.shape != (npts, 2):
            raise ValueError(f"sample_ref_coords must have shape ({npts}, 2), got {self.sample_ref_coords.shape}")
        self.sample_ref_weights = np.asarray(self.sample_ref_weights, dtype=float).reshape(-1)
        if self.sample_ref_weights.shape != (npts,):
            raise ValueError(f"sample_ref_weights must have shape ({npts},), got {self.sample_ref_weights.shape}")
        self.quadrature_order = int(self.quadrature_order)
        self.cell_type = str(self.cell_type).strip().lower()
        self.old_subscale_velocity = self._as_vector_field(self.old_subscale_velocity, npts, "old_subscale_velocity")
        self.predicted_subscale_velocity = self._as_vector_field(
            self.predicted_subscale_velocity,
            npts,
            "predicted_subscale_velocity",
        )
        self.momentum_projection = self._as_vector_field(self.momentum_projection, npts, "momentum_projection")
        self.mass_projection = self._as_scalar_field(self.mass_projection, npts, "mass_projection")
        self.old_mass_residual = self._as_scalar_field(self.old_mass_residual, npts, "old_mass_residual")
        self.n_elements = int(np.max(self.sample_element_ids)) + 1 if npts else 0
        self.n_qp_per_element = self._infer_n_qp_per_element(npts)
        if self.n_elements:
            expected_element_ids = np.repeat(np.arange(self.n_elements, dtype=int), self.n_qp_per_element)
            if not np.array_equal(self.sample_element_ids, expected_element_ids):
                raise ValueError(
                    "FluidDVMSState expects quadrature samples grouped contiguously by element "
                    "with a fixed number of quadrature points per element."
                )
            ref_coords = self.sample_ref_coords.reshape(self.n_elements, self.n_qp_per_element, 2)
            ref_weights = self.sample_ref_weights.reshape(self.n_elements, self.n_qp_per_element)
            if not np.allclose(ref_coords, ref_coords[0][None, :, :], atol=1.0e-12, rtol=1.0e-12):
                raise ValueError("sample_ref_coords must repeat the same reference quadrature rule for each element.")
            if not np.allclose(ref_weights, ref_weights[0][None, :], atol=1.0e-12, rtol=1.0e-12):
                raise ValueError("sample_ref_weights must repeat the same reference quadrature rule for each element.")
            reference_points = ref_coords[0]
            reference_weights = ref_weights[0]
        else:
            reference_points = np.zeros((0, 2), dtype=float)
            reference_weights = np.zeros((0,), dtype=float)
        self.quadrature_layout = QuadratureLayout(
            entity_kind="volume_cell",
            cell_type=self.cell_type,
            quadrature_order=self.quadrature_order,
            reference_points=reference_points,
            reference_weights=reference_weights,
        )
        self.registry = StateRegistry()
        self._state_fields = {
            "old_subscale_velocity": self.registry.register_quadrature(
                "example2_local_dvms_old_subscale_velocity",
                layout=self.quadrature_layout,
                values=self._reshape_vector_quadrature(self.old_subscale_velocity),
                tensor_shape=(2,),
                persistence="step",
                copy=False,
            ),
            "predicted_subscale_velocity": self.registry.register_quadrature(
                "example2_local_dvms_predicted_subscale_velocity",
                layout=self.quadrature_layout,
                values=self._reshape_vector_quadrature(self.predicted_subscale_velocity),
                tensor_shape=(2,),
                persistence="iteration",
                copy=False,
            ),
            "momentum_projection": self.registry.register_quadrature(
                "example2_local_dvms_momentum_projection",
                layout=self.quadrature_layout,
                values=self._reshape_vector_quadrature(self.momentum_projection),
                tensor_shape=(2,),
                persistence="iteration",
                copy=False,
            ),
            "mass_projection": self.registry.register_quadrature(
                "example2_local_dvms_mass_projection",
                layout=self.quadrature_layout,
                values=self._reshape_scalar_quadrature(self.mass_projection),
                tensor_shape=(),
                persistence="iteration",
                copy=False,
            ),
            "old_mass_residual": self.registry.register_quadrature(
                "example2_local_dvms_old_mass_residual",
                layout=self.quadrature_layout,
                values=self._reshape_scalar_quadrature(self.old_mass_residual),
                tensor_shape=(),
                persistence="step",
                copy=False,
            ),
        }
        self._coefficients = {
            key: field.coefficient()
            for key, field in self._state_fields.items()
        }
        self._sample_tree = cKDTree(self.sample_coords) if (cKDTree is not None and npts) else None
        self.sync_coefficients_from_samples()

    @staticmethod
    def _as_vector_field(values: np.ndarray, npts: int, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.shape != (npts, 2):
            raise ValueError(f"{name} must have shape ({npts}, 2), got {arr.shape}")
        return arr

    @staticmethod
    def _as_scalar_field(values: np.ndarray, npts: int, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.shape != (npts,):
            raise ValueError(f"{name} must have shape ({npts},), got {arr.shape}")
        return arr

    @property
    def sample_count(self) -> int:
        return int(self.sample_coords.shape[0])

    def _infer_n_qp_per_element(self, npts: int) -> int:
        if self.n_elements <= 0:
            return 0
        counts = np.bincount(self.sample_element_ids, minlength=self.n_elements)
        if counts.shape != (self.n_elements,):
            raise ValueError(
                f"sample_element_ids produced unexpected bincount shape {counts.shape}, expected ({self.n_elements},)."
            )
        n_qp = int(counts[0])
        if n_qp <= 0 or not np.all(counts == n_qp):
            raise ValueError(
                "FluidDVMSState requires the same quadrature-point count on every sampled element."
            )
        if int(npts) != int(self.n_elements * n_qp):
            raise ValueError(
                f"Inconsistent quadrature-state sample count: got {npts}, expected {self.n_elements * n_qp}."
            )
        return n_qp

    def _reshape_vector_quadrature(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(self.sample_count, 2)
        return arr.reshape(self.n_elements, self.n_qp_per_element, 2)

    def _reshape_scalar_quadrature(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(self.sample_count)
        return arr.reshape(self.n_elements, self.n_qp_per_element)

    def sync_coefficient(self, field_name: str) -> None:
        key = str(field_name)
        if key in {"old_subscale_velocity", "predicted_subscale_velocity", "momentum_projection"}:
            self._state_fields[key].assign(self._reshape_vector_quadrature(np.asarray(getattr(self, key), dtype=float)))
            return
        if key in {"mass_projection", "old_mass_residual"}:
            self._state_fields[key].assign(self._reshape_scalar_quadrature(np.asarray(getattr(self, key), dtype=float)))
            return
        raise KeyError(f"Unknown DVMS state coefficient {field_name!r}.")

    def sync_coefficients_from_samples(self) -> None:
        self.sync_coefficient("old_subscale_velocity")
        self.sync_coefficient("predicted_subscale_velocity")
        self.sync_coefficient("momentum_projection")
        self.sync_coefficient("mass_projection")
        self.sync_coefficient("old_mass_residual")

    def _sample(self, values: np.ndarray, x: float | np.ndarray, y: float | np.ndarray, *, dim: int) -> np.ndarray:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        target = np.c_[xa.reshape(-1), ya.reshape(-1)]
        values_arr = np.asarray(values, dtype=float).reshape(self.sample_count, dim)
        if self._sample_tree is not None and target.shape[0]:
            _, nearest = self._sample_tree.query(target, k=1)
            out = values_arr[np.asarray(nearest, dtype=int)].reshape(target.shape[0], dim)
        else:
            diff = self.sample_coords[None, :, :] - target[:, None, :]
            nearest = np.argmin(np.sum(diff * diff, axis=2), axis=1)
            out = values_arr[np.asarray(nearest, dtype=int)].reshape(target.shape[0], dim)
        if dim == 1:
            return out[:, 0].reshape(xa.shape)
        return out.reshape(xa.shape + (dim,))

    def sample_vector_field(self, field_name: str, x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
        return self._sample(np.asarray(getattr(self, field_name), dtype=float), x, y, dim=2)

    def sample_scalar_field(self, field_name: str, x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
        values = np.asarray(getattr(self, field_name), dtype=float).reshape(-1, 1)
        return self._sample(values, x, y, dim=1)

    def coefficient(self, field_name: str):
        try:
            return self._coefficients[str(field_name)]
        except KeyError as exc:
            raise KeyError(f"Unknown DVMS state coefficient {field_name!r}.") from exc

    def summary(self) -> dict[str, object]:
        return {
            "sample_count": int(self.sample_count),
            "n_qp_per_element": int(self.n_qp_per_element),
            "quadrature_order": int(self.quadrature_order),
            "old_subscale_velocity_inf_norm": float(np.max(np.abs(self.old_subscale_velocity))) if self.sample_count else 0.0,
            "predicted_subscale_velocity_inf_norm": float(np.max(np.abs(self.predicted_subscale_velocity)))
            if self.sample_count
            else 0.0,
            "momentum_projection_inf_norm": float(np.max(np.abs(self.momentum_projection))) if self.sample_count else 0.0,
            "mass_projection_inf_norm": float(np.max(np.abs(self.mass_projection))) if self.sample_count else 0.0,
            "old_mass_residual_inf_norm": float(np.max(np.abs(self.old_mass_residual))) if self.sample_count else 0.0,
        }


def _build_fluid_dvms_state(
    mesh: Mesh,
    *,
    quadrature_order: int,
    dof_handler: DofHandler | None = None,
) -> FluidDVMSState:
    quadrature_order = int(quadrature_order)
    if dof_handler is None:
        geom_order = max(int(getattr(mesh, "poly_order", 1)), 1)
        dof_handler = DofHandler(
            MixedElement(mesh, field_specs={"_geom": geom_order}),
            method="cg",
        )
    ref_qp, ref_qw = quadrature_volume(mesh.element_type, quadrature_order)
    ref_qp = np.asarray(ref_qp, dtype=float).reshape(-1, 2)
    ref_qw = np.asarray(ref_qw, dtype=float).reshape(-1)
    geo = dof_handler.precompute_geometric_factors(quadrature_order, reuse=True)
    qp_phys = np.asarray(geo["qp_phys"], dtype=float)
    n_elem = int(qp_phys.shape[0]) if qp_phys.ndim == 3 else 0
    n_qp = int(qp_phys.shape[1]) if qp_phys.ndim == 3 else 0
    sample_coords = qp_phys.reshape(-1, 2).copy() if n_elem and n_qp else np.zeros((0, 2), dtype=float)
    sample_element_ids = (
        np.repeat(np.arange(n_elem, dtype=int), n_qp)
        if n_elem and n_qp
        else np.zeros((0,), dtype=int)
    )
    sample_ref_coords = (
        np.broadcast_to(ref_qp[None, :, :], (n_elem, ref_qp.shape[0], 2)).reshape(-1, 2).copy()
        if n_elem and n_qp
        else np.zeros((0, 2), dtype=float)
    )
    sample_ref_weights = (
        np.broadcast_to(ref_qw[None, :], (n_elem, ref_qw.shape[0])).reshape(-1).copy()
        if n_elem and n_qp
        else np.zeros((0,), dtype=float)
    )
    zeros_vec = np.zeros((sample_coords.shape[0], 2), dtype=float)
    zeros_scal = np.zeros((sample_coords.shape[0],), dtype=float)
    return FluidDVMSState(
        sample_coords=sample_coords,
        sample_element_ids=sample_element_ids,
        sample_ref_coords=sample_ref_coords,
        sample_ref_weights=sample_ref_weights,
        quadrature_order=int(quadrature_order),
        cell_type=str(mesh.element_type),
        old_subscale_velocity=zeros_vec.copy(),
        predicted_subscale_velocity=zeros_vec.copy(),
        momentum_projection=zeros_vec.copy(),
        mass_projection=zeros_scal.copy(),
        old_mass_residual=zeros_scal.copy(),
    )


def _advance_fluid_dvms_history_after_step(
    state: FluidDVMSState,
    *,
    dh: DofHandler | None = None,
    mesh: Mesh | None = None,
    u_curr=None,
    a_curr=None,
    p_curr=None,
    d_curr=None,
    mesh_v_curr=None,
    rho_f: float | None = None,
    mu_f: float | None = None,
    dt: float | None = None,
    dynamic_tau: float | None = None,
    body_force: np.ndarray | None = None,
    backend: str | None = None,
    use_oss: bool = False,
) -> None:
    if int(state.sample_count) == 0:
        return
    if (
        dh is not None
        and mesh is not None
        and u_curr is not None
        and a_curr is not None
        and p_curr is not None
        and d_curr is not None
        and mesh_v_curr is not None
        and rho_f is not None
        and mu_f is not None
        and dt is not None
        and dynamic_tau is not None
    ):
        from .update import _update_fluid_dvms_old_subscale_after_step

        _update_fluid_dvms_old_subscale_after_step(
            state=state,
            dh=dh,
            mesh=mesh,
            u_curr=u_curr,
            a_curr=a_curr,
            p_curr=p_curr,
            d_curr=d_curr,
            mesh_v_curr=mesh_v_curr,
            rho_f=float(rho_f),
            mu_f=float(mu_f),
            dt=float(dt),
            dynamic_tau=float(dynamic_tau),
            body_force=body_force,
            backend=backend,
            use_oss=bool(use_oss),
        )
        nodal_divproj = getattr(state, "_nodal_div_projection", None)
        if nodal_divproj is not None:
            state._prev_nodal_div_projection = np.asarray(nodal_divproj, dtype=float).copy()
        return
    state.old_subscale_velocity[:, :] = np.asarray(state.predicted_subscale_velocity, dtype=float)
    state.sync_coefficients_from_samples()
    nodal_divproj = getattr(state, "_nodal_div_projection", None)
    if nodal_divproj is not None:
        state._prev_nodal_div_projection = np.asarray(nodal_divproj, dtype=float).copy()


def _fluid_dvms_summary(prob: dict[str, object]) -> dict[str, object]:
    state = prob.get("dvms_state")
    if not isinstance(state, FluidDVMSState):
        return {"enabled": False}
    return {"enabled": True, **state.summary()}


__all__ = [
    "FluidDVMSState",
    "_advance_fluid_dvms_history_after_step",
    "_build_fluid_dvms_state",
    "_fluid_dvms_summary",
]
