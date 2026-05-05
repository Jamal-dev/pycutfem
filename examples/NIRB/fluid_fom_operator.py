from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import scipy.sparse as sp

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh

from examples.NIRB.dvms import (
    FluidDVMSState,
    _clear_fluid_dvms_oss_projections,
    _update_fluid_dvms_oss_projections,
    _update_fluid_dvms_predicted_subscale,
)
from examples.NIRB.run_example2_local import (
    CoordinateLookup,
    _apply_dirichlet_bcs_to_state,
    _assemble_fluid_local_velocity_contribution_raw,
    _fluid_boundary_conditions,
    _fluid_interface_reaction_loads,
    _restore_fluid_dvms_state,
    _snapshot_fluid_dvms_state,
)


ResidualConvention = Literal["newton", "kratos_rhs"]


@dataclass(frozen=True)
class FluidBoundaryTags:
    interface_tag: str
    outlet_tag: str
    walls_tag: str
    cylinder_tag: str
    inlet_tag: str = "inlet"


@dataclass(frozen=True)
class FluidFOMParameters:
    rho_f: float
    mu_f: float
    dt: float
    quadrature_order: int
    bossak_alpha: float = -0.3
    dynamic_tau: float = 1.0
    backend: str = "cpp"
    contribution_mode: str = "system"
    use_oss: bool = False


@dataclass(frozen=True)
class FluidFOMAssembly:
    matrix: sp.csr_matrix | None
    residual: np.ndarray
    convention: ResidualConvention
    element_ids: np.ndarray | None


class FluidFOMOperator:
    """Example 2 exact ALE-DVMS fluid operator surface for fluid ROM work.

    This is intentionally example-level plumbing.  It wraps the verified
    Example 2 local ALE-DVMS operator and exposes a stable callable interface
    for projection and hyper-reduction experiments without moving policy into
    the generic :mod:`pycutfem` package.
    """

    def __init__(
        self,
        *,
        prob: dict[str, object],
        mesh: Mesh,
        parameters: FluidFOMParameters,
        boundary_tags: FluidBoundaryTags,
    ) -> None:
        self.prob = prob
        self.mesh = mesh
        self.parameters = parameters
        self.boundary_tags = boundary_tags

    @property
    def dh(self) -> DofHandler:
        return self.prob["dh"]  # type: ignore[return-value]

    @property
    def dvms_state(self) -> FluidDVMSState | None:
        state = self.prob.get("dvms_state")
        return state if isinstance(state, FluidDVMSState) else None

    def configure_boundary_conditions(
        self,
        *,
        iface_velocity: CoordinateLookup,
        inlet_lookup: Callable[[float, float], float],
        apply_to_state: bool = True,
    ):
        tags = self.boundary_tags
        bcs, bcs_homog = _fluid_boundary_conditions(
            iface_velocity=iface_velocity,
            inlet_lookup=inlet_lookup,
            inlet_tag=tags.inlet_tag,
            interface_tag=tags.interface_tag,
            outlet_tag=tags.outlet_tag,
            walls_tag=tags.walls_tag,
            cylinder_tag=tags.cylinder_tag,
        )
        self.prob["_current_bcs"] = bcs
        self.prob["_current_bcs_homog"] = bcs_homog
        if bool(apply_to_state):
            u_k = self.prob["u_k"]
            p_k = self.prob["p_k"]
            _apply_dirichlet_bcs_to_state(
                dh=self.dh,
                field_functions={
                    str(u_k.components[0].field_name): u_k.components[0],
                    str(u_k.components[1].field_name): u_k.components[1],
                    str(p_k.field_name): p_k,
                },
                bcs=bcs,
            )
        return bcs, bcs_homog

    def free_fluid_dofs(self, field_names: tuple[str, ...] = ("ux", "uy", "p")) -> np.ndarray:
        ids: list[int] = []
        for field_name in field_names:
            ids.extend(np.asarray(self.dh.get_field_slice(str(field_name)), dtype=int).ravel().tolist())
        active = np.asarray(sorted(set(ids)), dtype=int)
        bcs = self.prob.get("_current_bcs_homog") or self.prob.get("_current_bcs") or []
        bc_map = self.dh.get_dirichlet_data(bcs) or {}
        if not bc_map:
            return active
        fixed = np.fromiter((int(gdof) for gdof in bc_map.keys()), dtype=int)
        return np.setdiff1d(active, fixed, assume_unique=False).astype(int, copy=False)

    def snapshot_history(self) -> dict[str, np.ndarray] | None:
        return _snapshot_fluid_dvms_state(self.dvms_state)

    def restore_history(self, snapshot: dict[str, np.ndarray] | None) -> None:
        _restore_fluid_dvms_state(self.dvms_state, snapshot)

    def refresh_predicted_subscale(self, *, backend: str | None = None) -> None:
        state = self.dvms_state
        if state is None or int(state.sample_count) == 0:
            return
        p = self.parameters
        if not bool(p.use_oss):
            _clear_fluid_dvms_oss_projections(state)
        _update_fluid_dvms_predicted_subscale(
            state=state,
            dh=self.dh,
            mesh=self.mesh,
            u_k=self.prob["u_k"],
            u_prev=self.prob["u_prev"],
            a_prev=self.prob["a_prev"],
            a_curr=self.prob.get("a_k"),
            p_k=self.prob["p_k"],
            d_mesh=self.prob["d_mesh"],
            d_prev=self.prob["d_prev"],
            d_prev2=self.prob.get("d_prev2"),
            mesh_v=self.prob.get("w_mesh_k"),
            mesh_v_prev=self.prob.get("w_mesh_prev"),
            mesh_a_prev=self.prob.get("a_mesh_prev"),
            rho_f=float(p.rho_f),
            mu_f=float(p.mu_f),
            dt=float(p.dt),
            bossak_alpha=float(p.bossak_alpha),
            dynamic_tau=float(p.dynamic_tau),
            backend=str(p.backend if backend is None else backend),
            use_oss=bool(p.use_oss),
        )

    def update_oss_after_nonlinear_update(self) -> None:
        state = self.dvms_state
        if state is None or int(state.sample_count) == 0:
            return
        p = self.parameters
        if not bool(p.use_oss):
            _clear_fluid_dvms_oss_projections(state)
            return
        _update_fluid_dvms_oss_projections(
            state=state,
            dh=self.dh,
            mesh=self.mesh,
            u_k=self.prob["u_k"],
            p_k=self.prob["p_k"],
            d_mesh=self.prob["d_mesh"],
            d_prev=self.prob["d_prev"],
            d_prev2=self.prob.get("d_prev2"),
            mesh_v=self.prob.get("w_mesh_k"),
            mesh_v_prev=self.prob.get("w_mesh_prev"),
            mesh_a_prev=self.prob.get("a_mesh_prev"),
            rho_f=float(p.rho_f),
            dt=float(p.dt),
            bossak_alpha=float(p.bossak_alpha),
        )

    def assemble(
        self,
        *,
        need_matrix: bool = True,
        element_ids: np.ndarray | None = None,
        convention: ResidualConvention = "newton",
        refresh_predicted: bool | None = None,
        apply_dirichlet_lift: bool = False,
    ) -> FluidFOMAssembly:
        if convention not in {"newton", "kratos_rhs"}:
            raise ValueError(f"Unsupported fluid residual convention {convention!r}.")
        p = self.parameters
        do_refresh = bool(need_matrix) if refresh_predicted is None else bool(refresh_predicted)
        if do_refresh:
            self.refresh_predicted_subscale()
        matrix, raw_rhs = _assemble_fluid_local_velocity_contribution_raw(
            prob=self.prob,
            rho_f=float(p.rho_f),
            mu_f=float(p.mu_f),
            dt=float(p.dt),
            quad_order=int(p.quadrature_order),
            bossak_alpha=float(p.bossak_alpha),
            need_matrix=bool(need_matrix),
            contribution_mode=str(p.contribution_mode),
            apply_dirichlet_lift=bool(apply_dirichlet_lift),
            backend=str(p.backend),
            element_ids=element_ids,
        )
        raw_rhs = np.asarray(raw_rhs, dtype=float).reshape(-1)
        residual = -raw_rhs if convention == "newton" else raw_rhs.copy()
        return FluidFOMAssembly(
            matrix=None if matrix is None else matrix.tocsr(),
            residual=residual,
            convention=convention,
            element_ids=None if element_ids is None else np.asarray(element_ids, dtype=int).reshape(-1).copy(),
        )

    def reaction_loads(self, *, refresh_state: bool = False) -> CoordinateLookup:
        p = self.parameters
        return _fluid_interface_reaction_loads(
            prob=self.prob,
            rho_f=float(p.rho_f),
            mu_f=float(p.mu_f),
            dt=float(p.dt),
            quad_order=int(p.quadrature_order),
            bossak_alpha=float(p.bossak_alpha),
            dynamic_tau=float(p.dynamic_tau),
            interface_tag=self.boundary_tags.interface_tag,
            backend=str(p.backend),
            contribution_mode=str(p.contribution_mode),
            refresh_state=bool(refresh_state),
        )


__all__ = [
    "FluidBoundaryTags",
    "FluidFOMAssembly",
    "FluidFOMOperator",
    "FluidFOMParameters",
    "ResidualConvention",
]
