from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse.linalg as sp_la

from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.state import QuadratureLayout, StateRegistry, build_gaussian_nonlocal_quadrature_map
from pycutfem.ufl.expressions import TestFunction, TrialFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace

from examples.utils.poromechanics.damage import (
    ModifiedMisesNonlocalDamagePlaneStress2D,
    create_nonlocal_damage_state_2d,
    stage_modified_mises_nonlocal_damage_2d,
    volume_strain_voigt_2d,
)
from examples.utils.poromechanics.kratos_parity import build_kratos_consolidation_2d_mesh
from examples.utils.poromechanics.materials import UPlMaterial2D
from examples.utils.poromechanics.upl import plane_stress_internal_work_2d


def _damage_material() -> ModifiedMisesNonlocalDamagePlaneStress2D:
    return ModifiedMisesNonlocalDamagePlaneStress2D(
        young_modulus=2.8e10,
        poisson_ratio=0.1,
        damage_threshold=1.5e-4,
        strength_ratio=10.0,
        residual_strength=0.8,
        softening_slope=9000.0,
    )


def _upl_material() -> UPlMaterial2D:
    return UPlMaterial2D(
        young_modulus=2.8e10,
        poisson_ratio=0.1,
        porosity=0.0,
        biot_coefficient=0.9883333333333333,
        bulk_modulus_solid=1.0e12,
        bulk_modulus_liquid=2.0e9,
        permeability_xx=4.5e-13,
        permeability_yy=4.5e-13,
        dynamic_viscosity_liquid=1.0e-3,
        density_solid=2.5e3,
        density_liquid=0.0,
    )


def _backends() -> tuple[str, ...]:
    return ("python", "jit", "cpp")


def test_modified_mises_equivalent_strain_and_damage_match_kratos_formula() -> None:
    material = _damage_material()
    strain = np.asarray(
        [
            [2.0e-4, -0.5e-4, 1.2e-4],
            [-1.0e-4, 0.3e-4, -0.8e-4],
        ],
        dtype=float,
    )
    eq = material.equivalent_strain(strain)

    exx = strain[:, 0]
    eyy = strain[:, 1]
    exy = 0.5 * strain[:, 2]
    i1 = exx + eyy
    mean = 0.5 * i1
    j2 = 0.5 * ((exx - mean) ** 2 + (eyy - mean) ** 2 + 2.0 * exy**2)
    k = material.strength_ratio
    nu = material.poisson_ratio
    expected = i1 * (k - 1.0) / (2.0 * k * (1.0 - 2.0 * nu)) + np.sqrt(
        i1**2 * (k - 1.0) ** 2 / (1.0 - 2.0 * nu) ** 2
        + j2 * 12.0 * k / (1.0 + nu) ** 2
    ) / (2.0 * k)
    assert np.allclose(eq, expected)

    kappa = np.asarray([material.damage_threshold, 3.0e-4], dtype=float)
    damage = material.damage_from_kappa(kappa)
    expected_damage = 1.0 - material.damage_threshold * (1.0 - material.residual_strength) / kappa
    expected_damage -= material.residual_strength * np.exp(
        -material.softening_slope * (kappa - material.damage_threshold)
    )
    expected_damage = np.clip(expected_damage, 0.0, 1.0)
    assert np.allclose(damage, expected_damage)
    assert damage[0] == pytest.approx(0.0)


def test_nonlocal_damage_state_stages_and_commits_path_history() -> None:
    material = _damage_material()
    layout = QuadratureLayout(
        entity_kind="volume_cell",
        cell_type="tri",
        quadrature_order=1,
        reference_points=np.asarray([[1.0 / 3.0, 1.0 / 3.0]], dtype=float),
        reference_weights=np.asarray([0.5], dtype=float),
    )
    state = create_nonlocal_damage_state_2d(layout=layout, n_entities=2, material=material)
    points = np.asarray([[[0.0, 0.0]], [[10.0, 0.0]]], dtype=float)
    weights = np.ones((2, 1), dtype=float)
    qmap = build_gaussian_nonlocal_quadrature_map(points, weights, characteristic_length=0.5)

    strain = np.asarray([[[3.0e-4, 0.0, 0.0]], [[1.0e-4, 0.0, 0.0]]], dtype=float)
    stage_modified_mises_nonlocal_damage_2d(
        state=state,
        material=material,
        nonlocal_map=qmap,
        strain_voigt=strain,
    )
    assert np.all(state.kappa.staged_values >= material.damage_threshold)
    assert state.damage.staged_values[0, 0] > state.damage.values[0, 0]

    state.commit_step()
    first_damage = state.damage.values.copy()
    smaller_strain = np.zeros_like(strain)
    stage_modified_mises_nonlocal_damage_2d(
        state=state,
        material=material,
        nonlocal_map=qmap,
        strain_voigt=smaller_strain,
    )
    assert np.allclose(state.damage.staged_values, first_damage)


def test_volume_strain_voigt_2d_recovers_affine_displacement_gradient() -> None:
    mesh = build_kratos_consolidation_2d_mesh()
    mixed = MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "p": 1})
    dh = DofHandler(mixed, method="cg")
    solution = np.zeros(dh.total_dofs, dtype=float)

    for field, expression in {
        "ux": lambda x, y: 2.0 * x - 0.25 * y,
        "uy": lambda x, y: 0.5 * x + 3.0 * y,
    }.items():
        sl = np.asarray(dh.get_field_slice(field), dtype=int)
        coords = np.asarray(dh.get_field_dof_coords(field), dtype=float)
        solution[sl] = np.asarray([expression(x, y) for x, y in coords], dtype=float)

    strain = volume_strain_voigt_2d(dh, solution, quadrature_order=2)
    assert np.allclose(strain[..., 0], 2.0)
    assert np.allclose(strain[..., 1], 3.0)
    assert np.allclose(strain[..., 2], 0.25)


@pytest.mark.parametrize("backend", _backends())
def test_damaged_plane_stress_ufl_scales_stiffness_all_backends(backend: str) -> None:
    mesh = build_kratos_consolidation_2d_mesh()
    mixed = MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "p": 1})
    dh = DofHandler(mixed, method="cg")
    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)
    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)
    del p, q

    geo = dh.precompute_geometric_factors(2, level_set=lambda *_: 0.0, reuse=False)
    layout = QuadratureLayout(
        entity_kind="volume_cell",
        cell_type=mesh.element_type,
        quadrature_order=2,
        reference_points=np.asarray(geo["qp_ref"], dtype=float),
        reference_weights=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=float),
    )
    registry = StateRegistry()
    damage = registry.register_quadrature(
        "damage",
        layout=layout,
        values=np.full((mesh.n_elements, layout.n_qp), 0.25, dtype=float),
        tensor_shape=(),
        persistence="step",
    )
    material = _upl_material()
    undamaged = plane_stress_internal_work_2d(u, v, material) * dx(metadata={"q": 2})
    damaged = plane_stress_internal_work_2d(u, v, material, damage=damage.coefficient()) * dx(metadata={"q": 2})

    K0, _ = assemble_form(Equation(undamaged, 0.0), dof_handler=dh, bcs=[], backend=backend)
    Kd, _ = assemble_form(Equation(damaged, 0.0), dof_handler=dh, bcs=[], backend=backend)

    disp = np.asarray(dh.get_field_slice("ux") + dh.get_field_slice("uy"), dtype=int)
    K0d = K0.tocsr()[disp][:, disp]
    Kdd = Kd.tocsr()[disp][:, disp]
    assert sp_la.norm(Kdd - 0.75 * K0d) <= 1.0e-8 * max(1.0, sp_la.norm(K0d))
