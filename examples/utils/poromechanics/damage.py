"""Nonlocal damage helpers for 2D poromechanics validation cases.

The classes here mirror Kratos' nonlocal damage update order, but keep the
implementation explicit and testable:

1. compute local equivalent strain at every volume quadrature point,
2. apply a generic nonlocal quadrature averaging map,
3. stage ``kappa = max(kappa_old, nonlocal_equivalent_strain)``,
4. stage scalar damage from the staged history variable,
5. commit staged values only after nonlinear convergence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycutfem.fem.reference import get_reference
from pycutfem.state import NonlocalQuadratureMap, QuadratureLayout, StateRegistry


@dataclass(frozen=True)
class ModifiedMisesNonlocalDamagePlaneStress2D:
    """Kratos-compatible Modified Mises nonlocal damage law in plane stress."""

    young_modulus: float
    poisson_ratio: float
    damage_threshold: float
    strength_ratio: float
    residual_strength: float
    softening_slope: float

    def __post_init__(self) -> None:
        if float(self.young_modulus) <= 0.0:
            raise ValueError("young_modulus must be positive.")
        if not (-1.0 < float(self.poisson_ratio) < 0.5):
            raise ValueError("poisson_ratio must lie in (-1, 0.5).")
        if float(self.damage_threshold) <= 0.0:
            raise ValueError("damage_threshold must be positive.")
        if float(self.strength_ratio) <= 0.0:
            raise ValueError("strength_ratio must be positive.")
        if float(self.residual_strength) < 0.0:
            raise ValueError("residual_strength cannot be negative.")
        if float(self.softening_slope) < 0.0:
            raise ValueError("softening_slope cannot be negative.")

    @property
    def elastic_matrix(self) -> np.ndarray:
        """Plane-stress elastic matrix for ``[exx, eyy, gamma_xy]``."""

        E = float(self.young_modulus)
        nu = float(self.poisson_ratio)
        c = E / (1.0 - nu * nu)
        return np.asarray(
            [
                [c, c * nu, 0.0],
                [c * nu, c, 0.0],
                [0.0, 0.0, 0.5 * c * (1.0 - nu)],
            ],
            dtype=float,
        )

    def equivalent_strain(self, strain_voigt) -> np.ndarray:
        """Modified Mises equivalent strain used by Kratos.

        ``strain_voigt`` uses the engineering shear convention
        ``[eps_xx, eps_yy, gamma_xy]``.
        """

        strain = np.asarray(strain_voigt, dtype=float)
        if strain.shape[-1] != 3:
            raise ValueError(f"strain_voigt must end with length 3, got shape {strain.shape}.")

        exx = strain[..., 0]
        eyy = strain[..., 1]
        exy = 0.5 * strain[..., 2]
        i1 = exx + eyy
        mean = 0.5 * i1
        dxx = exx - mean
        dyy = eyy - mean
        j2 = 0.5 * (dxx * dxx + dyy * dyy + 2.0 * exy * exy)

        k = float(self.strength_ratio)
        nu = float(self.poisson_ratio)
        one_minus_2nu = 1.0 - 2.0 * nu
        sqrt_arg = (
            i1 * i1 * (k - 1.0) * (k - 1.0) / (one_minus_2nu * one_minus_2nu)
            + j2 * 12.0 * k / ((1.0 + nu) * (1.0 + nu))
        )
        if np.any(sqrt_arg < -1.0e-18):
            raise FloatingPointError("Modified Mises square-root argument became negative.")
        sqrt_arg = np.maximum(sqrt_arg, 0.0)
        return (
            i1 * (k - 1.0) / (2.0 * k * one_minus_2nu)
            + np.sqrt(sqrt_arg) / (2.0 * k)
        )

    def damage_from_kappa(self, kappa) -> np.ndarray:
        """Modified exponential damage hardening law."""

        kap = np.asarray(kappa, dtype=float)
        threshold = float(self.damage_threshold)
        safe_kappa = np.maximum(kap, threshold)
        residual = float(self.residual_strength)
        slope = float(self.softening_slope)
        damage = 1.0 - threshold * (1.0 - residual) / safe_kappa - residual * np.exp(
            -slope * (safe_kappa - threshold)
        )
        return np.clip(damage, 0.0, 1.0)

    def stress_from_strain_and_damage(self, strain_voigt, damage) -> np.ndarray:
        strain = np.asarray(strain_voigt, dtype=float)
        d = np.asarray(damage, dtype=float)
        stress = strain @ self.elastic_matrix.T
        return (1.0 - d)[..., None] * stress


@dataclass
class NonlocalDamageState2D:
    """Quadrature-state container for nonlocal scalar damage."""

    registry: StateRegistry
    layout: QuadratureLayout
    local_equivalent_strain: object
    nonlocal_equivalent_strain: object
    kappa: object
    damage: object

    @property
    def damage_coefficient(self):
        from pycutfem.state.coefficient import QuadratureStateCoefficient

        return QuadratureStateCoefficient(
            self.damage,
            jit_name="nonlocal_damage",
            values=self.damage.staged_values,
        )

    def rollback_step(self) -> None:
        self.registry.rollback_step()

    def commit_step(self) -> None:
        self.registry.commit_step()


def create_nonlocal_damage_state_2d(
    *,
    layout: QuadratureLayout,
    n_entities: int,
    material: ModifiedMisesNonlocalDamagePlaneStress2D,
    registry: StateRegistry | None = None,
) -> NonlocalDamageState2D:
    """Register quadrature fields needed by the nonlocal damage update."""

    reg = registry if registry is not None else StateRegistry()
    shape = (int(n_entities), int(layout.n_qp))
    threshold = float(material.damage_threshold)
    local = reg.register_quadrature(
        "local_equivalent_strain",
        layout=layout,
        values=np.zeros(shape, dtype=float),
        tensor_shape=(),
        persistence="step",
        copy=True,
    )
    nonlocal_eq = reg.register_quadrature(
        "nonlocal_equivalent_strain",
        layout=layout,
        values=np.zeros(shape, dtype=float),
        tensor_shape=(),
        persistence="step",
        copy=True,
    )
    kappa = reg.register_quadrature(
        "damage_kappa",
        layout=layout,
        values=np.full(shape, threshold, dtype=float),
        tensor_shape=(),
        persistence="step",
        copy=True,
    )
    damage = reg.register_quadrature(
        "damage",
        layout=layout,
        values=material.damage_from_kappa(np.full(shape, threshold, dtype=float)),
        tensor_shape=(),
        persistence="step",
        copy=True,
    )
    return NonlocalDamageState2D(
        registry=reg,
        layout=layout,
        local_equivalent_strain=local,
        nonlocal_equivalent_strain=nonlocal_eq,
        kappa=kappa,
        damage=damage,
    )


def stage_modified_mises_nonlocal_damage_2d(
    *,
    state: NonlocalDamageState2D,
    material: ModifiedMisesNonlocalDamagePlaneStress2D,
    nonlocal_map: NonlocalQuadratureMap,
    strain_voigt,
) -> None:
    """Stage one nonlocal damage update from quadrature strain values."""

    strain = np.asarray(strain_voigt, dtype=float)
    expected = state.damage.values.shape + (3,)
    if strain.shape != expected:
        raise ValueError(f"strain_voigt expected shape {expected}, got {strain.shape}.")

    local_eq = material.equivalent_strain(strain)
    nonlocal_eq = nonlocal_map.apply(local_eq)
    kappa = np.maximum(state.kappa.values, nonlocal_eq)
    damage = material.damage_from_kappa(kappa)
    state.local_equivalent_strain.stage(local_eq)
    state.nonlocal_equivalent_strain.stage(nonlocal_eq)
    state.kappa.stage(kappa)
    state.damage.stage(damage)


def volume_strain_voigt_2d(
    dof_handler,
    solution,
    *,
    displacement_fields: tuple[str, str] = ("ux", "uy"),
    quadrature_order: int,
) -> np.ndarray:
    """Evaluate small-strain ``[exx, eyy, gamma_xy]`` at volume quadrature points."""

    if dof_handler.mixed_element is None:
        raise RuntimeError("volume_strain_voigt_2d requires a MixedElement-backed DofHandler.")
    ux_field, uy_field = displacement_fields
    mesh = dof_handler.mixed_element.mesh
    field_order_x = int(dof_handler.mixed_element._field_orders[ux_field])
    field_order_y = int(dof_handler.mixed_element._field_orders[uy_field])
    if field_order_x != field_order_y:
        raise ValueError("Displacement components must use the same interpolation order.")

    geo = dof_handler.precompute_geometric_factors(
        int(quadrature_order),
        level_set=lambda *_: 0.0,
        reuse=True,
    )
    qref = np.asarray(geo["qp_ref"], dtype=float)
    j_inv = np.asarray(geo["J_inv"], dtype=float)
    n_entities, n_qp = j_inv.shape[:2]

    ref = get_reference(mesh.element_type, field_order_x)
    n_loc = len(np.asarray(ref.shape(float(qref[0, 0]), float(qref[0, 1]))).reshape(-1))
    grad_ref = np.empty((n_qp, n_loc, 2), dtype=float)
    for q, (xi, eta) in enumerate(qref):
        grad_ref[q] = np.asarray(ref.grad(float(xi), float(eta)), dtype=float)

    ux_maps = np.asarray(dof_handler.element_maps[ux_field], dtype=int)
    uy_maps = np.asarray(dof_handler.element_maps[uy_field], dtype=int)
    values = np.asarray(solution, dtype=float).reshape(-1)
    if ux_maps.shape != (n_entities, n_loc) or uy_maps.shape != (n_entities, n_loc):
        raise ValueError(
            "Displacement element maps are incompatible with the active quadrature layout: "
            f"ux={ux_maps.shape}, uy={uy_maps.shape}, expected {(n_entities, n_loc)}."
        )

    out = np.empty((n_entities, n_qp, 3), dtype=float)
    for e in range(n_entities):
        ux_values = values[ux_maps[e]]
        uy_values = values[uy_maps[e]]
        for q in range(n_qp):
            grad_phys = grad_ref[q] @ j_inv[e, q]
            grad_ux = ux_values @ grad_phys
            grad_uy = uy_values @ grad_phys
            out[e, q, 0] = grad_ux[0]
            out[e, q, 1] = grad_uy[1]
            out[e, q, 2] = grad_ux[1] + grad_uy[0]
    return out
