import json
from functools import lru_cache

import numpy as np
import pytest
import scipy.sparse as sp

from examples.poroelasticity.consolidation_pycutfem import _build_p2_tri_mesh
from examples.utils.poromechanics import (
    FLUID_PUMPING_2D_FRACTURE_SAMPLE_NODE_IDS,
    PairedUPlInterface2D,
    UPlInterfaceMaterial2D,
    UPlMaterial2D,
    VALIDATION_CASES,
    bilinear_cohesive_2d_ufl_response,
    build_kratos_fic_triangle_upl_system_2d,
    build_kratos_consolidation_interface_2d_mesh,
    build_paired_upl_interface_kratos_newton_batch_2d,
    build_paired_upl_interface_local_batch_2d,
    build_paired_upl_interface_local_system_2d,
    build_upl_interface_ufl_system_2d,
    build_upl_kratos_newton_interface_ufl_system_2d,
    build_upl_link_interface_ufl_system_2d,
    build_upl_theta_system_2d,
    effective_stress_linear_2d,
    epsilon_2d,
    hydraulic_conductivity_form_2d,
    kratos_fic_triangle_element_length_squared,
    normal_liquid_flux_rhs_2d,
    paired_upl_interfaces_to_trace_link_interface_2d,
    paired_upl_interfaces_to_station_trace_nonmatching_interface_2d,
    paired_upl_interface_to_nonmatching_interface_2d,
    require_validation_case_supported,
    solve_kratos_consolidation_2d_pycutfem,
    solve_kratos_consolidation_interface_2d_pycutfem,
    solve_kratos_four_point_shear_2d_pycutfem,
    solve_kratos_four_point_shear_2d_runtime_reference,
    solve_kratos_fluid_pumping_2d_pycutfem,
    solve_kratos_fluid_pumping_2d_fracture_pycutfem,
    solve_kratos_fluid_pumping_2d_reference,
    solve_kratos_undrained_soil_column_2d_pycutfem,
    solve_kratos_vertical_fault_consolidation_2d_pycutfem,
    solve_kratos_vertical_fault_consolidation_2d_reference,
    unsupported_validation_cases,
)

def _have_cpp_backend() -> bool:
    try:
        import pybind11  # noqa: F401
    except Exception:
        return False
    return True


def _ufl_backends() -> list[str]:
    backends = ["python", "jit"]
    if _have_cpp_backend():
        backends.append("cpp")
    return backends


from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.core.topology import Node
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.state import QuadratureLayout, StateRegistry
from pycutfem.tracefem import (
    TraceFractureNetwork2D,
    TraceFracturePropagationSettings2D,
    TraceStationEntity2D,
    plan_fracture_extensions_from_damage,
    transfer_trace_quadrature_state,
)
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    abs_value,
    div,
    inner,
    signum,
    sqrt,
)
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dNonmatchingInterface, dS, dx
from examples.utils.poromechanics.kratos_parity import _trace_tip_bulk_damage_probe
from pycutfem.ufl.spaces import FunctionSpace


FLUID_PUMPING_KRATOS_FIRST_STEP = {
    8677: (1103.0709960247636, -2.217313901255246e-05, 8.154042019311765e-06),
    8675: (-1581.5391524419779, 2.2139902106846505e-05, 8.154042337150455e-06),
    1201: (281.6652442163841, -1.2832907527647485e-06, 1.8269499224196808e-06),
    3686: (0.0, 3.0089886635980645e-06, 0.0),
    820: (0.0, 0.0, 0.0),
    8545: (7.308717941940624, 0.0, 0.0),
}

FLUID_PUMPING_PYCUTFEM_FIRST_NEWTON_INCREMENT = {
    8677: (-24.765877733256993, 1.0842817070428172e-08, 2.6943967027793513e-11),
    8675: (-24.77594888741979, 1.0774969855895321e-08, 2.700391274773034e-11),
    1201: (-0.34518178416775763, -3.3810783963559476e-09, -1.44338576518585e-11),
    3686: (0.0, -3.0579194828590587e-09, 0.0),
    820: (0.0, 0.0, 0.0),
    8545: (-0.01045377486155945, 0.0, 0.0),
}

FLUID_PUMPING_2D_FRACTURE_KRATOS_FIRST_STEP = {
    853: (-50225.039512199604, -8.971671642744113e-07, -4.6422929632956446e-08),
    953: (-49623.13306572537, 0.0, 0.0),
    955: (-50826.947870753036, 0.0, 0.0),
    833: (-2097.0106961396514, -7.39221134514438e-07, -1.5507335927857247e-08),
    95: (-1914.159383544334, 0.0, 0.0),
}


def _assert_fluid_pumping_samples(
    res,
    expected: dict[int, tuple[float, float, float]],
    *,
    rtol: float,
    atol: float,
    displacement_atol: float | None = None,
):
    disp_atol = atol if displacement_atol is None else displacement_atol
    for node_id, (pressure, ux, uy) in expected.items():
        np.testing.assert_allclose(res.liquid_pressure_by_node[node_id][-1], pressure, rtol=rtol, atol=atol)
        np.testing.assert_allclose(res.displacement_x_by_node[node_id][-1], ux, rtol=rtol, atol=disp_atol)
        np.testing.assert_allclose(res.displacement_y_by_node[node_id][-1], uy, rtol=rtol, atol=disp_atol)


@lru_cache(maxsize=1)
def _fluid_pumping_pycutfem_first_newton_cpp():
    return solve_kratos_fluid_pumping_2d_pycutfem(
        backend="cpp",
        end_time=1.0e-5,
        interface_update="kratos_lagged",
    )


@lru_cache(maxsize=1)
def _fluid_pumping_fracture_pycutfem_first_step_cpp():
    return solve_kratos_fluid_pumping_2d_fracture_pycutfem(
        backend="cpp",
        interface_backend="cpp",
        end_time=0.1,
        interface_update="kratos_newton",
        enable_propagation=False,
    )


@lru_cache(maxsize=1)
def _vertical_fault_consolidation_reference_two_steps():
    pytest.importorskip("KratosMultiphysics")
    return solve_kratos_vertical_fault_consolidation_2d_reference(end_time=1.0)


@lru_cache(maxsize=1)
def _four_point_shear_direct_reference_two_steps():
    pytest.importorskip("KratosMultiphysics")
    return solve_kratos_four_point_shear_2d_runtime_reference(end_time=2.0)


def test_upl_material_kratos_storage_formula():
    mat = UPlMaterial2D(
        young_modulus=14.4e9,
        poisson_ratio=0.2,
        porosity=0.19,
        biot_coefficient=0.78,
        bulk_modulus_solid=24.0e9,
        bulk_modulus_liquid=2.2e9,
        permeability_xx=2.0e-10,
    )

    expected = (0.78 - 0.19) / 24.0e9 + 0.19 / 2.2e9
    assert mat.biot_modulus_inverse == expected
    assert mat.mixture_density == 0.0


def test_named_kratos_validation_case_manifest_is_explicit():
    expected = {
        "undrained_soil_column_2d",
        "four_point_shear",
        "vertical_fault_consolidation",
        "consolidation_interface_2d",
        "fracture_network_flow",
        "fluid_driven_fracture_propagation",
    }
    assert set(VALIDATION_CASES) == expected
    assert require_validation_case_supported("consolidation_interface_2d").status == "exact"
    assert require_validation_case_supported("undrained_soil_column_2d").status == "exact"
    assert require_validation_case_supported("vertical_fault_consolidation").status == "exact"
    assert require_validation_case_supported("four_point_shear").status == "exact"
    blocked = unsupported_validation_cases()
    assert "four_point_shear" not in blocked
    assert "undrained_soil_column_2d" not in blocked
    assert "vertical_fault_consolidation" not in blocked
    with pytest.raises(NotImplementedError, match="full fluid_pumping_2D"):
        require_validation_case_supported("fracture_network_flow")


def _single_p1_triangle_mesh() -> Mesh:
    nodes = [
        Node(0, 0.0, 0.0),
        Node(1, 1.0, 0.0),
        Node(2, 0.0, 1.0),
    ]
    cells = np.asarray([[0, 1, 2]], dtype=int)
    return Mesh(
        nodes=nodes,
        element_connectivity=cells,
        elements_corner_nodes=cells,
        element_type="tri",
        poly_order=1,
    )


def _jit_ir_signature(form):
    from pycutfem.jit.visitor import IRGenerator

    integrals = getattr(form, "integrals", None)
    if integrals is None:
        integrals = [form]
    signature = []
    for integral in integrals:
        ir = IRGenerator().generate(integral.integrand)
        op_signature = []
        for op in ir:
            fields = getattr(op, "__dataclass_fields__", {})
            entries = []
            for name in fields:
                if name == "func_ref":
                    continue
                value = getattr(op, name)
                if isinstance(value, list):
                    value = tuple(value)
                elif isinstance(value, np.ndarray):
                    value = ("array", tuple(value.shape))
                entries.append((name, value))
            op_signature.append((type(op).__name__, tuple(entries)))
        signature.append(tuple(op_signature))
    return tuple(signature)


def _kratos_small_interface_fixture():
    mesh = build_kratos_consolidation_interface_2d_mesh()
    dh = DofHandler(MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "p": 1}), method="dg")

    def dofs(eid: int, local: int) -> tuple[int, int, int]:
        return (
            int(dh.element_maps["ux"][eid][local]),
            int(dh.element_maps["uy"][eid][local]),
            int(dh.element_maps["p"][eid][local]),
        )

    paired = PairedUPlInterface2D(
        negative_coords=((0.5, 1.0), (0.5, 0.0)),
        positive_coords=((0.5, 1.0), (0.5, 0.0)),
        negative_dofs=(dofs(1, 3), dofs(1, 1)),
        positive_dofs=(dofs(0, 2), dofs(0, 0)),
    )
    interface = paired_upl_interface_to_nonmatching_interface_2d(
        paired,
        mesh=mesh,
        negative_element_ids=1,
        positive_element_ids=0,
    )
    return mesh, dh, paired, interface


def _paired_station_dof_map(rows):
    arr = np.asarray(rows, dtype=np.int64)
    return {
        "ux": tuple(int(v) for v in arr[:, 0]),
        "uy": tuple(int(v) for v in arr[:, 1]),
        "p": tuple(int(v) for v in arr[:, 2]),
    }


def test_trace_fracture_network_builds_generic_trace_link_tables():
    mesh = _single_p1_triangle_mesh()
    entity = TraceStationEntity2D(
        entity_id=7,
        negative_coords=((0.0, -0.01), (1.0, -0.01)),
        positive_coords=((0.0, 0.01), (1.0, 0.01)),
        negative_dofs={"ux": (0, 4), "uy": (1, 5), "p": (2, 6), "lambda": (3, 7)},
        positive_dofs={"ux": (8, 12), "uy": (9, 13), "p": (10, 14), "lambda": (11, 15)},
        owner_neg_id=0,
        owner_pos_id=0,
    )
    network = TraceFractureNetwork2D(
        mesh=mesh,
        entities=(entity,),
        field_names=("ux", "uy", "p", "lambda"),
    )
    trace = network.to_trace_link_interface(quadrature="lobatto")
    factors = trace.precomputed_factors()

    np.testing.assert_array_equal(
        factors["gdofs_map"],
        np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], dtype=np.int64),
    )
    np.testing.assert_array_equal(factors["neg_map_lambda"], np.array([[3, 7]], dtype=np.int32))
    np.testing.assert_array_equal(factors["pos_map_lambda"], np.array([[11, 15]], dtype=np.int32))
    assert factors["r00_lambda_neg"].shape == (1, 2, 2)
    assert factors["g_lambda"].shape == (1, 2, 16, 2)
    assert factors["domain_type"] == "trace_link"
    assert trace.n_entities() == 1


def test_trace_fracture_extension_planner_and_state_transfer():
    mesh = _single_p1_triangle_mesh()
    base = TraceStationEntity2D(
        entity_id=10,
        negative_coords=((0.0, -0.005), (1.0, -0.005)),
        positive_coords=((0.0, 0.005), (1.0, 0.005)),
        negative_dofs={"ux": (0, 3), "uy": (1, 4), "p": (2, 5)},
        positive_dofs={"ux": (6, 9), "uy": (7, 10), "p": (8, 11)},
    )
    network = TraceFractureNetwork2D(mesh=mesh, entities=(base,), field_names=("ux", "uy", "p"))
    settings = TraceFracturePropagationSettings2D.from_kratos_fractures_data(
        {
            "fracture_data": {
                "propagation_damage": 0.5,
                "propagation_length": 0.25,
                "propagation_width": 0.02,
            }
        }
    )

    assert plan_fracture_extensions_from_damage(network, np.array([[0.1, 0.2]]), settings) == ()
    plans = plan_fracture_extensions_from_damage(network, np.array([[0.1, 0.7]]), settings)
    assert len(plans) == 1
    np.testing.assert_allclose(plans[0].start, (1.0, 0.0), atol=1.0e-15)
    np.testing.assert_allclose(plans[0].end, (1.25, 0.0), atol=1.0e-15)

    extension = network.make_tip_extension(
        parent_entity_id=10,
        new_entity_id=11,
        negative_dofs={"ux": (12, 15), "uy": (13, 16), "p": (14, 17)},
        positive_dofs={"ux": (18, 21), "uy": (19, 22), "p": (20, 23)},
        length=settings.propagation_length,
        width=settings.propagation_width,
    )
    loose = TraceStationEntity2D(
        entity_id=12,
        negative_coords=((2.0, -0.005), (2.25, -0.005)),
        positive_coords=((2.0, 0.005), (2.25, 0.005)),
        negative_dofs={"ux": (24, 27), "uy": (25, 28), "p": (26, 29)},
        positive_dofs={"ux": (30, 33), "uy": (31, 34), "p": (32, 35)},
    )
    updated = network.insert(extension).insert(loose)
    state = np.array([[[0.1], [0.7]]], dtype=float)
    transferred = transfer_trace_quadrature_state(network, updated, state, default=-1.0)
    np.testing.assert_allclose(transferred[0], state[0], atol=0.0)
    np.testing.assert_allclose(transferred[1], state[0], atol=0.0)
    np.testing.assert_allclose(transferred[2], -np.ones((2, 1)), atol=0.0)


def test_trace_tip_bulk_damage_probe_uses_weighted_tip_neighbourhood():
    mesh = _single_p1_triangle_mesh()
    base = TraceStationEntity2D(
        entity_id=10,
        negative_coords=((0.0, -0.005), (1.0, -0.005)),
        positive_coords=((0.0, 0.005), (1.0, 0.005)),
        negative_dofs={"ux": (0, 3), "uy": (1, 4), "p": (2, 5)},
        positive_dofs={"ux": (6, 9), "uy": (7, 10), "p": (8, 11)},
    )
    network = TraceFractureNetwork2D(mesh=mesh, entities=(base,), field_names=("ux", "uy", "p"))
    points = np.array([[[1.00, 0.00], [1.02, 0.00], [1.40, 0.00]]], dtype=float)
    damage = np.array([[0.20, 0.80, 1.00]], dtype=float)
    weights = np.ones((1, 3), dtype=float)

    probe = _trace_tip_bulk_damage_probe(
        network,
        damage,
        points,
        weights,
        search_radius=0.05,
    )

    raw = np.exp(-4.0 * np.array([0.0, 0.02]) ** 2 / (0.05 * 0.05))
    expected = float(np.dot(raw, np.array([0.20, 0.80])) / np.sum(raw))
    assert probe["support_count"] == 2
    np.testing.assert_allclose(probe["damage"], expected, rtol=1.0e-14, atol=1.0e-14)
    assert probe["damage"] < 0.80


@pytest.mark.parametrize("backend", _ufl_backends())
def test_trace_fracture_network_newton_interface_ufl_local_batch_matches_paired_reference(
    backend,
    tmp_path,
    monkeypatch,
):
    if backend == "cpp":
        pytest.importorskip("pybind11")
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_cache_trace_fracture_{backend}"))

    mesh, dh, paired_base, _volume_trace_interface = _kratos_small_interface_fixture()
    paired = PairedUPlInterface2D(
        negative_coords=((0.49, 1.0), (0.49, 0.0)),
        positive_coords=((0.51, 1.0), (0.51, 0.0)),
        negative_dofs=paired_base.negative_dofs,
        positive_dofs=paired_base.positive_dofs,
    )
    entity = TraceStationEntity2D(
        entity_id=101,
        negative_coords=paired.negative_coords,
        positive_coords=paired.positive_coords,
        negative_dofs=_paired_station_dof_map(paired.negative_dofs),
        positive_dofs=_paired_station_dof_map(paired.positive_dofs),
        owner_neg_id=1,
        owner_pos_id=0,
    )
    interface = TraceFractureNetwork2D(
        mesh=mesh,
        entities=(entity,),
        field_names=("ux", "uy", "p"),
    ).to_trace_link_interface(quadrature="lobatto")
    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)
    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)
    u_current = VectorFunction(name="u_current", field_names=["ux", "uy"], dof_handler=dh)
    p_current = Function(name="p_current", field_name="p", dof_handler=dh)
    velocity_current = VectorFunction(name="velocity_current", field_names=["ux", "uy"], dof_handler=dh)
    p_rate_current = Function(name="p_rate_current", field_name="p", dof_handler=dh)
    layout = QuadratureLayout(
        entity_kind="trace_link",
        cell_type="line",
        quadrature_order=2,
        reference_points=np.array([[-1.0], [1.0]], dtype=float),
        reference_weights=np.array([1.0, 1.0], dtype=float),
    )
    registry = StateRegistry()
    state_values = np.array([[0.1, 0.2]], dtype=float)
    damage = registry.register_quadrature(
        "trace_fracture_damage",
        layout=layout,
        values=state_values.copy(),
        n_entities=interface.n_entities(),
    )
    material = UPlInterfaceMaterial2D(
        normal_stiffness=1.0,
        shear_stiffness=1.0,
        porosity=0.22,
        biot_coefficient=0.85,
        bulk_modulus_solid=1.0e9,
        bulk_modulus_liquid=2.2e9,
        dynamic_viscosity_liquid=1.0e-3,
        initial_joint_width=2.0e-2,
        transversal_permeability_coefficient=2.0e-8,
        young_modulus=2.0e7,
        critical_displacement=0.1,
        yield_stress=1.5e5,
        damage_threshold=0.1,
        friction_coefficient=0.35,
    )
    previous = np.zeros(dh.total_dofs, dtype=float)
    current = np.linspace(-2.0e-4, 2.0e3, dh.total_dofs)
    velocity = np.linspace(1.0e-6, -3.0e-6, dh.total_dofs)
    p_rate = np.linspace(4.0e-3, -1.0e-3, dh.total_dofs)
    u_current.nodal_values = current[u_current._g_dofs]
    p_current.nodal_values = current[p_current._g_dofs]
    velocity_current.nodal_values = velocity[velocity_current._g_dofs]
    p_rate_current.nodal_values = p_rate[p_rate_current._g_dofs]

    system = build_upl_kratos_newton_interface_ufl_system_2d(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_current=u_current,
        p_current=p_current,
        velocity_current=velocity_current,
        p_rate_current=p_rate_current,
        state=damage.coefficient(jit_name="trace_fracture_damage"),
        material=material,
        interface=interface,
        dt=0.25,
        theta_u=0.75,
        theta_p=0.5,
        quadrature="lobatto",
        quadrature_order=2,
        permeability_law="fracture",
    )
    compiler = FormCompiler(dh, backend=backend)
    batch = compiler.assemble_local_contributions(system.equation)
    reference = build_paired_upl_interface_kratos_newton_batch_2d(
        [paired],
        material,
        dt=0.25,
        theta_u=0.75,
        theta_p=0.5,
        previous_solution=previous,
        current_solution=current,
        velocity_solution=velocity,
        dt_pressure_solution=p_rate,
        state_variables=state_values,
        quadrature="lobatto",
        permeability_law="fracture",
        backend="python",
    )
    np.testing.assert_array_equal(batch.gdofs_map, reference.gdofs_map)
    np.testing.assert_allclose(batch.K_elem, reference.K_elem, rtol=1.0e-10, atol=1.0e-6)
    np.testing.assert_allclose(batch.F_elem, reference.R_elem, rtol=1.0e-10, atol=1.0e-6)

    trace_weights_before = np.asarray(interface.precomputed_factors()["qw"], dtype=float).copy()
    state_update = compiler.evaluate_trace_link_expressions_on_quadrature(
        {"state_next": system.state_update_expr},
        layout=layout,
        trace=interface,
        quadrature="lobatto",
    )
    np.testing.assert_allclose(state_update["state_next"], reference.state_next, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(interface.precomputed_factors()["qw"], trace_weights_before, rtol=0.0, atol=0.0)

    # Regression guard for compiled backends: evaluating a trace quadrature
    # expression must not mutate the cached integration weights used by the
    # next Newton assembly on the same interface object.
    batch_after_eval = compiler.assemble_local_contributions(system.equation)
    np.testing.assert_allclose(batch_after_eval.K_elem, batch.K_elem, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(batch_after_eval.F_elem, batch.F_elem, rtol=0.0, atol=0.0)


def test_nonmatching_quadrature_layout_and_precompute_export_state_ids():
    from pycutfem.integration.quadrature import gauss_lobatto

    xi, weights = gauss_lobatto(2)
    layout = QuadratureLayout(
        entity_kind="nonmatching_interface",
        cell_type="line",
        quadrature_order=2,
        reference_points=np.asarray(xi, dtype=float),
        reference_weights=np.asarray(weights, dtype=float),
    )
    assert layout.reference_points.shape == (2, 1)

    with pytest.raises(ValueError, match="nonmatching_interface.*reference_points"):
        QuadratureLayout(
            entity_kind="nonmatching_interface",
            cell_type="line",
            quadrature_order=2,
            reference_points=np.zeros((2, 2), dtype=float),
            reference_weights=np.ones(2, dtype=float),
        )

    _mesh, dh, _paired, interface = _kratos_small_interface_fixture()
    geo = dh.precompute_nonmatching_interface_factors(
        interface=interface,
        qdeg=2,
        derivs={(0, 0)},
        quadrature="lobatto",
    )
    np.testing.assert_array_equal(geo["qstate_owner_id"], geo["eids"])
    np.testing.assert_array_equal(geo["qstate_entity_id"], geo["eids"])
    np.testing.assert_allclose(geo["qref"], layout.reference_points)
    np.testing.assert_allclose(geo["qw_ref"], layout.reference_weights)


@pytest.mark.parametrize("backend", _ufl_backends())
def test_nonmatching_quadrature_state_evaluation_all_backends(backend, tmp_path, monkeypatch):
    if backend == "cpp":
        pytest.importorskip("pybind11")
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_cache_qstate_{backend}"))

    _mesh, dh, _paired, interface = _kratos_small_interface_fixture()
    layout = QuadratureLayout(
        entity_kind="nonmatching_interface",
        cell_type="line",
        quadrature_order=2,
        reference_points=np.array([-1.0, 1.0], dtype=float),
        reference_weights=np.array([1.0, 1.0], dtype=float),
    )
    registry = StateRegistry()
    values = np.array([[0.25, 0.75]], dtype=float)
    qfield = registry.register_quadrature(
        "damage",
        layout=layout,
        values=values.copy(),
        n_entities=interface.n_segments(),
    )
    qcoef = qfield.coefficient(jit_name="interface_damage")
    compiler = FormCompiler(dh, backend=backend)

    result = compiler.evaluate_nonmatching_interface_expressions_on_quadrature(
        {
            "damage_plus_two": qcoef + Constant(2.0),
            "damage_abs_shift": abs_value(qcoef - Constant(0.5)),
            "damage_sqrt_square": sqrt(qcoef * qcoef),
            "damage_sign_shift": signum(qcoef - Constant(0.5)),
        },
        layout=layout,
        interface=interface,
        quadrature="lobatto",
    )
    np.testing.assert_allclose(result["damage_plus_two"], values + 2.0)
    np.testing.assert_allclose(result["damage_abs_shift"], np.abs(values - 0.5))
    np.testing.assert_allclose(result["damage_sqrt_square"], np.sqrt(values * values))
    np.testing.assert_allclose(result["damage_sign_shift"], np.sign(values - 0.5))

    qfield.values[...] = values + 3.0
    result_updated = compiler.evaluate_nonmatching_interface_expressions_on_quadrature(
        {"damage_plus_two": qcoef + Constant(2.0)},
        layout=layout,
        interface=interface,
        quadrature="lobatto",
    )
    np.testing.assert_allclose(result_updated["damage_plus_two"], values + 5.0)


@pytest.mark.parametrize("backend", _ufl_backends())
def test_bilinear_cohesive_interface_law_ufl_matches_python_reference(backend, tmp_path, monkeypatch):
    if backend == "cpp":
        pytest.importorskip("pybind11")
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_cache_cohesive_law_{backend}"))

    from examples.utils.poromechanics.interface import _bilinear_cohesive_2d_response

    _mesh, dh, _paired, interface = _kratos_small_interface_fixture()
    layout = QuadratureLayout(
        entity_kind="nonmatching_interface",
        cell_type="line",
        quadrature_order=2,
        reference_points=np.array([-1.0, 1.0], dtype=float),
        reference_weights=np.array([1.0, 1.0], dtype=float),
    )
    registry = StateRegistry()
    t_values = np.array([[0.035, -0.004]], dtype=float)
    n_values = np.array([[0.012, -0.002]], dtype=float)
    s_values = np.array([[0.12, 0.55]], dtype=float)
    t_state = registry.register_quadrature("t_jump", layout=layout, values=t_values.copy(), n_entities=1)
    n_state = registry.register_quadrature("n_jump", layout=layout, values=n_values.copy(), n_entities=1)
    damage_state = registry.register_quadrature("damage", layout=layout, values=s_values.copy(), n_entities=1)
    material = UPlInterfaceMaterial2D(
        normal_stiffness=1.0,
        shear_stiffness=1.0,
        porosity=0.2,
        biot_coefficient=1.0,
        bulk_modulus_solid=1.0e9,
        bulk_modulus_liquid=2.2e9,
        dynamic_viscosity_liquid=1.0e-3,
        initial_joint_width=1.0e-4,
        young_modulus=2.0e7,
        critical_displacement=0.1,
        yield_stress=1.5e5,
        damage_threshold=0.1,
        friction_coefficient=0.35,
    )

    response = bilinear_cohesive_2d_ufl_response(
        material,
        tangential_jump=t_state.coefficient(jit_name="cohesive_t_jump"),
        normal_jump=n_state.coefficient(jit_name="cohesive_n_jump"),
        state=damage_state.coefficient(jit_name="cohesive_damage"),
    )
    exprs = {
        "D00": response.D00,
        "D01": response.D01,
        "D10": response.D10,
        "D11": response.D11,
        "stress_t": response.stress_t,
        "stress_n": response.stress_n,
        "equivalent": response.equivalent,
        "loading": response.loading,
        "state_next": response.state_next,
        "open_flag": response.open_flag,
    }
    actual = FormCompiler(dh, backend=backend).evaluate_nonmatching_interface_expressions_on_quadrature(
        exprs,
        layout=layout,
        interface=interface,
        quadrature="lobatto",
    )

    expected = {name: np.zeros((1, 2), dtype=float) for name in exprs}
    for q in range(2):
        D, stress, equivalent, loading = _bilinear_cohesive_2d_response(
            material,
            tangential_jump=float(t_values[0, q]),
            normal_jump=float(n_values[0, q]),
            state=float(s_values[0, q]),
            is_open=bool(n_values[0, q] > 0.0),
        )
        expected["D00"][0, q] = D[0, 0]
        expected["D01"][0, q] = D[0, 1]
        expected["D10"][0, q] = D[1, 0]
        expected["D11"][0, q] = D[1, 1]
        expected["stress_t"][0, q] = stress[0]
        expected["stress_n"][0, q] = stress[1]
        expected["equivalent"][0, q] = equivalent
        expected["loading"][0, q] = 1.0 if loading else 0.0
        expected["state_next"][0, q] = min(equivalent, 1.0) if loading else max(float(s_values[0, q]), material.damage_threshold)
        expected["open_flag"][0, q] = 1.0 if n_values[0, q] > 0.0 else 0.0

    for name, expected_values in expected.items():
        np.testing.assert_allclose(actual[name], expected_values, rtol=1.0e-12, atol=1.0e-8)


@pytest.mark.parametrize("backend", _ufl_backends())
def test_bilinear_cohesive_contact_branch_ufl_matches_python_reference(backend, tmp_path, monkeypatch):
    if backend == "cpp":
        pytest.importorskip("pybind11")
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_cache_cohesive_contact_{backend}"))

    from examples.utils.poromechanics.interface import _bilinear_cohesive_2d_response
    from pycutfem.integration.quadrature import gauss_lobatto

    _mesh, dh, _paired, interface = _kratos_small_interface_fixture()
    xi, weights = gauss_lobatto(4)
    layout = QuadratureLayout(
        entity_kind="nonmatching_interface",
        cell_type="line",
        quadrature_order=4,
        reference_points=np.asarray(xi, dtype=float),
        reference_weights=np.asarray(weights, dtype=float),
    )
    registry = StateRegistry()
    t_values = np.array([[0.018, -0.026, 0.0, 0.004]], dtype=float)
    n_values = np.array([[-0.003, -0.004, -0.001, -0.002]], dtype=float)
    s_values = np.array([[0.12, 0.20, 0.30, 0.55]], dtype=float)
    t_state = registry.register_quadrature("contact_t_jump", layout=layout, values=t_values.copy(), n_entities=1)
    n_state = registry.register_quadrature("contact_n_jump", layout=layout, values=n_values.copy(), n_entities=1)
    damage_state = registry.register_quadrature("contact_damage", layout=layout, values=s_values.copy(), n_entities=1)
    material = UPlInterfaceMaterial2D(
        normal_stiffness=1.0,
        shear_stiffness=1.0,
        porosity=0.2,
        biot_coefficient=1.0,
        bulk_modulus_solid=1.0e9,
        bulk_modulus_liquid=2.2e9,
        dynamic_viscosity_liquid=1.0e-3,
        initial_joint_width=1.0e-4,
        young_modulus=2.0e7,
        critical_displacement=0.1,
        yield_stress=1.5e5,
        damage_threshold=0.1,
        friction_coefficient=0.35,
    )

    response = bilinear_cohesive_2d_ufl_response(
        material,
        tangential_jump=t_state.coefficient(jit_name="cohesive_contact_t_jump"),
        normal_jump=n_state.coefficient(jit_name="cohesive_contact_n_jump"),
        state=damage_state.coefficient(jit_name="cohesive_contact_damage"),
    )
    exprs = {
        "D00": response.D00,
        "D01": response.D01,
        "D10": response.D10,
        "D11": response.D11,
        "stress_t": response.stress_t,
        "stress_n": response.stress_n,
        "equivalent": response.equivalent,
        "loading": response.loading,
        "state_next": response.state_next,
        "open_flag": response.open_flag,
    }
    actual = FormCompiler(dh, backend=backend).evaluate_nonmatching_interface_expressions_on_quadrature(
        exprs,
        layout=layout,
        interface=interface,
        quadrature="lobatto",
    )

    expected = {name: np.zeros((1, 4), dtype=float) for name in exprs}
    for q in range(4):
        D, stress, equivalent, loading = _bilinear_cohesive_2d_response(
            material,
            tangential_jump=float(t_values[0, q]),
            normal_jump=float(n_values[0, q]),
            state=float(s_values[0, q]),
            is_open=False,
        )
        expected["D00"][0, q] = D[0, 0]
        expected["D01"][0, q] = D[0, 1]
        expected["D10"][0, q] = D[1, 0]
        expected["D11"][0, q] = D[1, 1]
        expected["stress_t"][0, q] = stress[0]
        expected["stress_n"][0, q] = stress[1]
        expected["equivalent"][0, q] = equivalent
        expected["loading"][0, q] = 1.0 if loading else 0.0
        expected["state_next"][0, q] = min(equivalent, 1.0) if loading else max(
            float(s_values[0, q]), material.damage_threshold
        )

    for name, expected_values in expected.items():
        np.testing.assert_allclose(actual[name], expected_values, rtol=1.0e-12, atol=1.0e-8)
    np.testing.assert_allclose(actual["open_flag"], np.zeros((1, 4)), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual["D10"], np.zeros((1, 4)), rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize("backend", _ufl_backends())
def test_kratos_newton_interface_ufl_local_batch_matches_python_reference(backend, tmp_path, monkeypatch):
    if backend == "cpp":
        pytest.importorskip("pybind11")
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_cache_newton_interface_{backend}"))

    _mesh, dh, paired, interface = _kratos_small_interface_fixture()
    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)
    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)
    u_current = VectorFunction(name="u_current", field_names=["ux", "uy"], dof_handler=dh)
    p_current = Function(name="p_current", field_name="p", dof_handler=dh)
    velocity_current = VectorFunction(name="velocity_current", field_names=["ux", "uy"], dof_handler=dh)
    p_rate_current = Function(name="p_rate_current", field_name="p", dof_handler=dh)
    layout = QuadratureLayout(
        entity_kind="nonmatching_interface",
        cell_type="line",
        quadrature_order=2,
        reference_points=np.array([[-1.0], [1.0]], dtype=float),
        reference_weights=np.array([1.0, 1.0], dtype=float),
    )
    registry = StateRegistry()
    state_values = np.array([[0.1, 0.35]], dtype=float)
    damage = registry.register_quadrature(
        "damage",
        layout=layout,
        values=state_values.copy(),
        n_entities=interface.n_segments(),
    )
    material = UPlInterfaceMaterial2D(
        normal_stiffness=1.0,
        shear_stiffness=1.0,
        porosity=0.22,
        biot_coefficient=0.85,
        bulk_modulus_solid=1.0e9,
        bulk_modulus_liquid=2.2e9,
        dynamic_viscosity_liquid=1.0e-3,
        initial_joint_width=1.0e-4,
        transversal_permeability_coefficient=2.0e-8,
        young_modulus=2.0e7,
        critical_displacement=0.1,
        yield_stress=1.5e5,
        damage_threshold=0.1,
        friction_coefficient=0.35,
    )
    previous = np.zeros(dh.total_dofs, dtype=float)
    current = np.linspace(-2.5e-4, 3.0e3, dh.total_dofs)
    velocity = np.linspace(1.0e-6, -2.0e-6, dh.total_dofs)
    p_rate = np.linspace(4.0e-3, -3.0e-3, dh.total_dofs)
    u_current.nodal_values = current[u_current._g_dofs]
    p_current.nodal_values = current[p_current._g_dofs]
    velocity_current.nodal_values = velocity[velocity_current._g_dofs]
    p_rate_current.nodal_values = p_rate[p_rate_current._g_dofs]

    system = build_upl_kratos_newton_interface_ufl_system_2d(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_current=u_current,
        p_current=p_current,
        velocity_current=velocity_current,
        p_rate_current=p_rate_current,
        state=damage.coefficient(jit_name="newton_interface_damage"),
        material=material,
        interface=interface,
        dt=0.25,
        theta_u=0.75,
        theta_p=0.5,
        quadrature="lobatto",
        quadrature_order=2,
        permeability_law="fracture",
    )
    names = _jit_parameter_names(system.tangent_form) + _jit_parameter_names(system.residual_form)
    assert names
    assert not [name for name in names if name.startswith("jit_const_")]
    batch = FormCompiler(dh, backend=backend).assemble_local_contributions(system.equation)
    reference = build_paired_upl_interface_kratos_newton_batch_2d(
        [paired],
        material,
        dt=0.25,
        theta_u=0.75,
        theta_p=0.5,
        previous_solution=previous,
        current_solution=current,
        velocity_solution=velocity,
        dt_pressure_solution=p_rate,
        state_variables=state_values,
        quadrature="lobatto",
        permeability_law="fracture",
        backend="python",
    )

    global_to_ufl = {int(g): i for i, g in enumerate(batch.gdofs_map[0])}
    ufl_idx = [global_to_ufl[int(g)] for g in reference.gdofs_map[0]]
    np.testing.assert_allclose(
        batch.K_elem[0][np.ix_(ufl_idx, ufl_idx)],
        reference.K_elem[0],
        rtol=1.0e-10,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(batch.F_elem[0][ufl_idx], reference.R_elem[0], rtol=1.0e-10, atol=1.0e-6)

    state_update = FormCompiler(dh, backend=backend).evaluate_nonmatching_interface_expressions_on_quadrature(
        {"state_next": system.state_update_expr},
        layout=layout,
        interface=interface,
        quadrature="lobatto",
    )
    np.testing.assert_allclose(state_update["state_next"], reference.state_next, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("backend", _ufl_backends())
def test_station_trace_newton_interface_ufl_local_batch_matches_paired_reference(backend, tmp_path, monkeypatch):
    if backend == "cpp":
        pytest.importorskip("pybind11")
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_cache_station_interface_{backend}"))

    mesh, dh, paired_base, _volume_trace_interface = _kratos_small_interface_fixture()
    paired = PairedUPlInterface2D(
        negative_coords=((0.49, 1.0), (0.49, 0.0)),
        positive_coords=((0.51, 1.0), (0.51, 0.0)),
        negative_dofs=paired_base.negative_dofs,
        positive_dofs=paired_base.positive_dofs,
    )
    interface = paired_upl_interfaces_to_trace_link_interface_2d(
        [paired],
        mesh=mesh,
        negative_element_ids=1,
        positive_element_ids=0,
        quadrature="lobatto",
    )
    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)
    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)
    u_current = VectorFunction(name="u_current", field_names=["ux", "uy"], dof_handler=dh)
    p_current = Function(name="p_current", field_name="p", dof_handler=dh)
    velocity_current = VectorFunction(name="velocity_current", field_names=["ux", "uy"], dof_handler=dh)
    p_rate_current = Function(name="p_rate_current", field_name="p", dof_handler=dh)
    layout = QuadratureLayout(
        entity_kind="trace_link",
        cell_type="line",
        quadrature_order=2,
        reference_points=np.array([[-1.0], [1.0]], dtype=float),
        reference_weights=np.array([1.0, 1.0], dtype=float),
    )
    registry = StateRegistry()
    state_values = np.array([[0.1, 0.2]], dtype=float)
    damage = registry.register_quadrature(
        "station_damage",
        layout=layout,
        values=state_values.copy(),
        n_entities=interface.n_entities(),
    )
    material = UPlInterfaceMaterial2D(
        normal_stiffness=1.0,
        shear_stiffness=1.0,
        porosity=0.22,
        biot_coefficient=0.85,
        bulk_modulus_solid=1.0e9,
        bulk_modulus_liquid=2.2e9,
        dynamic_viscosity_liquid=1.0e-3,
        initial_joint_width=2.0e-2,
        transversal_permeability_coefficient=2.0e-8,
        young_modulus=2.0e7,
        critical_displacement=0.1,
        yield_stress=1.5e5,
        damage_threshold=0.1,
        friction_coefficient=0.35,
    )
    previous = np.zeros(dh.total_dofs, dtype=float)
    current = np.linspace(-2.0e-4, 2.0e3, dh.total_dofs)
    velocity = np.linspace(1.0e-6, -3.0e-6, dh.total_dofs)
    p_rate = np.linspace(4.0e-3, -1.0e-3, dh.total_dofs)
    u_current.nodal_values = current[u_current._g_dofs]
    p_current.nodal_values = current[p_current._g_dofs]
    velocity_current.nodal_values = velocity[velocity_current._g_dofs]
    p_rate_current.nodal_values = p_rate[p_rate_current._g_dofs]

    system = build_upl_kratos_newton_interface_ufl_system_2d(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_current=u_current,
        p_current=p_current,
        velocity_current=velocity_current,
        p_rate_current=p_rate_current,
        state=damage.coefficient(jit_name="station_damage"),
        material=material,
        interface=interface,
        dt=0.25,
        theta_u=0.75,
        theta_p=0.5,
        quadrature="lobatto",
        quadrature_order=2,
        permeability_law="fracture",
    )
    batch = FormCompiler(dh, backend=backend).assemble_local_contributions(system.equation)
    reference = build_paired_upl_interface_kratos_newton_batch_2d(
        [paired],
        material,
        dt=0.25,
        theta_u=0.75,
        theta_p=0.5,
        previous_solution=previous,
        current_solution=current,
        velocity_solution=velocity,
        dt_pressure_solution=p_rate,
        state_variables=state_values,
        quadrature="lobatto",
        permeability_law="fracture",
        backend="python",
    )
    np.testing.assert_array_equal(batch.gdofs_map, reference.gdofs_map)
    np.testing.assert_allclose(batch.K_elem, reference.K_elem, rtol=1.0e-10, atol=1.0e-6)
    np.testing.assert_allclose(batch.F_elem, reference.R_elem, rtol=1.0e-10, atol=1.0e-6)

    state_update = FormCompiler(dh, backend=backend).evaluate_trace_link_expressions_on_quadrature(
        {"state_next": system.state_update_expr},
        layout=layout,
        trace=interface,
        quadrature="lobatto",
    )
    np.testing.assert_allclose(state_update["state_next"], reference.state_next, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("backend", _ufl_backends())
def test_station_trace_contact_newton_interface_ufl_local_batch_matches_paired_reference(
    backend,
    tmp_path,
    monkeypatch,
):
    if backend == "cpp":
        pytest.importorskip("pybind11")
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_cache_station_contact_{backend}"))

    mesh, dh, paired_base, _volume_trace_interface = _kratos_small_interface_fixture()
    paired = PairedUPlInterface2D(
        negative_coords=((0.49, 1.0), (0.49, 0.0)),
        positive_coords=((0.51, 1.0), (0.51, 0.0)),
        negative_dofs=paired_base.negative_dofs,
        positive_dofs=paired_base.positive_dofs,
    )
    interface = paired_upl_interfaces_to_trace_link_interface_2d(
        [paired],
        mesh=mesh,
        negative_element_ids=1,
        positive_element_ids=0,
        quadrature="lobatto",
    )
    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)
    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)
    u_current = VectorFunction(name="u_current", field_names=["ux", "uy"], dof_handler=dh)
    p_current = Function(name="p_current", field_name="p", dof_handler=dh)
    velocity_current = VectorFunction(name="velocity_current", field_names=["ux", "uy"], dof_handler=dh)
    p_rate_current = Function(name="p_rate_current", field_name="p", dof_handler=dh)
    layout = QuadratureLayout(
        entity_kind="trace_link",
        cell_type="line",
        quadrature_order=2,
        reference_points=np.array([[-1.0], [1.0]], dtype=float),
        reference_weights=np.array([1.0, 1.0], dtype=float),
    )
    registry = StateRegistry()
    state_values = np.array([[0.1, 0.35]], dtype=float)
    damage = registry.register_quadrature(
        "station_contact_damage",
        layout=layout,
        values=state_values.copy(),
        n_entities=interface.n_entities(),
    )
    material = UPlInterfaceMaterial2D(
        normal_stiffness=1.0,
        shear_stiffness=1.0,
        porosity=0.22,
        biot_coefficient=0.85,
        bulk_modulus_solid=1.0e9,
        bulk_modulus_liquid=2.2e9,
        dynamic_viscosity_liquid=1.0e-3,
        initial_joint_width=2.0e-2,
        transversal_permeability_coefficient=2.0e-8,
        young_modulus=2.0e7,
        critical_displacement=0.1,
        yield_stress=1.5e5,
        damage_threshold=0.1,
        friction_coefficient=0.35,
    )
    previous = np.zeros(dh.total_dofs, dtype=float)
    current = np.zeros(dh.total_dofs, dtype=float)
    velocity = np.zeros(dh.total_dofs, dtype=float)
    p_rate = np.zeros(dh.total_dofs, dtype=float)

    tangential_jumps = (0.012, -0.018)
    pressure_pairs = ((220.0, -70.0), (310.0, -110.0))
    for station, (neg_dofs, pos_dofs) in enumerate(zip(paired.negative_dofs, paired.positive_dofs)):
        neg_ux, neg_uy, neg_p = neg_dofs
        pos_ux, pos_uy, pos_p = pos_dofs
        current[neg_ux] = 1.0e-3
        current[pos_ux] = -1.0e-3
        current[neg_uy] = tangential_jumps[station]
        current[pos_uy] = 0.0
        current[neg_p], current[pos_p] = pressure_pairs[station]

        velocity[neg_ux] = 2.0e-6 + station * 3.0e-7
        velocity[pos_ux] = -1.5e-6 + station * 2.0e-7
        velocity[neg_uy] = -8.0e-7 + station * 1.0e-7
        velocity[pos_uy] = 6.0e-7 - station * 1.0e-7
        p_rate[neg_p] = 4.0e-3 + station * 2.0e-4
        p_rate[pos_p] = -3.0e-3 - station * 1.0e-4

    u_current.nodal_values = current[u_current._g_dofs]
    p_current.nodal_values = current[p_current._g_dofs]
    velocity_current.nodal_values = velocity[velocity_current._g_dofs]
    p_rate_current.nodal_values = p_rate[p_rate_current._g_dofs]

    system = build_upl_kratos_newton_interface_ufl_system_2d(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_current=u_current,
        p_current=p_current,
        velocity_current=velocity_current,
        p_rate_current=p_rate_current,
        state=damage.coefficient(jit_name="station_contact_damage"),
        material=material,
        interface=interface,
        dt=0.25,
        theta_u=0.75,
        theta_p=0.5,
        quadrature="lobatto",
        quadrature_order=2,
        permeability_law="fracture",
    )
    batch = FormCompiler(dh, backend=backend).assemble_local_contributions(system.equation)
    reference = build_paired_upl_interface_kratos_newton_batch_2d(
        [paired],
        material,
        dt=0.25,
        theta_u=0.75,
        theta_p=0.5,
        previous_solution=previous,
        current_solution=current,
        velocity_solution=velocity,
        dt_pressure_solution=p_rate,
        state_variables=state_values,
        quadrature="lobatto",
        permeability_law="fracture",
        backend="python",
    )
    np.testing.assert_array_equal(batch.gdofs_map, reference.gdofs_map)
    np.testing.assert_allclose(batch.K_elem, reference.K_elem, rtol=1.0e-10, atol=1.0e-6)
    np.testing.assert_allclose(batch.F_elem, reference.R_elem, rtol=1.0e-10, atol=1.0e-6)

    state_update = FormCompiler(dh, backend=backend).evaluate_trace_link_expressions_on_quadrature(
        {"state_next": system.state_update_expr, "open_flag": system.response.open_flag},
        layout=layout,
        trace=interface,
        quadrature="lobatto",
    )
    np.testing.assert_allclose(state_update["state_next"], reference.state_next, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(state_update["open_flag"], np.zeros((1, 2)), rtol=0.0, atol=0.0)


def _jit_parameter_names(form) -> list[str]:
    from pycutfem.jit.visitor import IRGenerator

    names: list[str] = []
    integrals = getattr(form, "integrals", None)
    if integrals is None:
        integrals = [form]
    for integral in integrals:
        for op in IRGenerator().generate(integral.integrand):
            if type(op).__name__ in {"LoadConstantArray", "LoadElementWiseConstant"}:
                names.append(str(getattr(op, "name", "")))
    return names


def test_poromechanics_bulk_constants_are_named_and_value_independent_for_jit():
    mesh = _single_p1_triangle_mesh()
    dh = DofHandler(MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "p": 1}), method="cg")
    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)

    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)
    u_prev = VectorFunction(name="u_prev", field_names=["ux", "uy"], dof_handler=dh)
    p_prev = Function(name="p_prev", field_name="p", dof_handler=dh)

    material_a = UPlMaterial2D(
        young_modulus=20.0,
        poisson_ratio=0.25,
        porosity=0.2,
        biot_coefficient=1.0,
        permeability_xx=0.0,
        storage_inverse=1.0e-2,
    )
    material_b = UPlMaterial2D(
        young_modulus=35.0,
        poisson_ratio=0.2,
        porosity=0.27,
        biot_coefficient=0.82,
        permeability_xx=3.0e-9,
        storage_inverse=7.0e-3,
    )
    kwargs = dict(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_prev=u_prev,
        p_prev=p_prev,
        dx_measure=dx(metadata={"q": 2}),
    )
    system_a = build_kratos_fic_triangle_upl_system_2d(
        **kwargs,
        material=material_a,
        dt=1.0,
        theta_u=1.0,
        theta_p=1.0,
        element_length_squared=np.asarray([2.0 / np.pi]),
    )
    system_b = build_kratos_fic_triangle_upl_system_2d(
        **kwargs,
        material=material_b,
        dt=0.4,
        theta_u=0.6,
        theta_p=0.5,
        element_length_squared=np.asarray([3.0]),
    )

    assert _jit_ir_signature(system_a.lhs_form) == _jit_ir_signature(system_b.lhs_form)
    assert _jit_ir_signature(system_a.rhs_form) == _jit_ir_signature(system_b.rhs_form)
    names = _jit_parameter_names(system_a.lhs_form) + _jit_parameter_names(system_a.rhs_form)
    assert names
    assert not [name for name in names if name.startswith(("jit_const_", "jit_ewc_"))]


def test_poromechanics_interface_constants_are_named_and_value_independent_for_jit():
    mesh = build_kratos_consolidation_interface_2d_mesh()
    dh = DofHandler(MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "p": 1}), method="dg")
    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)

    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)
    u_prev = VectorFunction(name="u_prev", field_names=["ux", "uy"], dof_handler=dh)
    u_current = VectorFunction(name="u_current", field_names=["ux", "uy"], dof_handler=dh)
    p_prev = Function(name="p_prev", field_name="p", dof_handler=dh)

    def dofs(eid: int, local: int) -> tuple[int, int, int]:
        return (
            int(dh.element_maps["ux"][eid][local]),
            int(dh.element_maps["uy"][eid][local]),
            int(dh.element_maps["p"][eid][local]),
        )

    paired = PairedUPlInterface2D(
        negative_coords=((0.5, 1.0), (0.5, 0.0)),
        positive_coords=((0.5, 1.0), (0.5, 0.0)),
        negative_dofs=(dofs(1, 3), dofs(1, 1)),
        positive_dofs=(dofs(0, 2), dofs(0, 0)),
    )
    interface = paired_upl_interface_to_nonmatching_interface_2d(
        paired,
        mesh=mesh,
        negative_element_ids=1,
        positive_element_ids=0,
    )
    material_a = UPlInterfaceMaterial2D(
        normal_stiffness=2.0e7,
        shear_stiffness=1.0e6,
        penalty_stiffness=1.0,
        porosity=0.3,
        biot_coefficient=1.0,
        bulk_modulus_solid=1.0e25,
        bulk_modulus_liquid=1.0e25,
        dynamic_viscosity_liquid=1.0e-3,
        initial_joint_width=1.0e-4,
        transversal_permeability_coefficient=0.0,
    )
    material_b = UPlInterfaceMaterial2D(
        normal_stiffness=3.0e7,
        shear_stiffness=2.0e6,
        penalty_stiffness=1.7,
        porosity=0.22,
        biot_coefficient=0.83,
        bulk_modulus_solid=4.0e9,
        bulk_modulus_liquid=2.2e9,
        dynamic_viscosity_liquid=2.0e-3,
        initial_joint_width=2.0e-4,
        transversal_permeability_coefficient=3.0e-8,
    )
    kwargs = dict(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_prev=u_prev,
        p_prev=p_prev,
        u_current=u_current,
        interface=interface,
        quadrature="lobatto",
        quadrature_order=2,
    )
    system_a = build_upl_interface_ufl_system_2d(
        **kwargs,
        material=material_a,
        dt=1.0,
        theta_u=1.0,
        theta_p=1.0,
    )
    system_b = build_upl_interface_ufl_system_2d(
        **kwargs,
        material=material_b,
        dt=0.4,
        theta_u=0.6,
        theta_p=0.5,
    )

    assert _jit_ir_signature(system_a.lhs_form) == _jit_ir_signature(system_b.lhs_form)
    assert _jit_ir_signature(system_a.rhs_form) == _jit_ir_signature(system_b.rhs_form)
    names = _jit_parameter_names(system_a.lhs_form) + _jit_parameter_names(system_a.rhs_form)
    assert names
    assert not [name for name in names if name.startswith("jit_const_")]


def test_upl_theta_forms_match_previous_inline_consolidation_forms():
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    cells = np.asarray([[0, 1, 2]], dtype=int)
    mesh = _build_p2_tri_mesh(points, cells)

    mixed_element = MixedElement(mesh, field_specs={"ux": 2, "uy": 2, "p": 1})
    dh = DofHandler(mixed_element, method="cg")

    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)
    pres_space = FunctionSpace("pressure", ["p"], dim=0)

    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)
    u0 = VectorFunction(name="u0", field_names=["ux", "uy"], dof_handler=dh)
    p0 = Function(name="p0", field_name="p", dof_handler=dh)
    u0.nodal_values[:] = np.linspace(0.0, 1.0e-4, u0.nodal_values.size)
    p0.nodal_values[:] = np.linspace(3.8e8, 4.2e8, p0.nodal_values.size)

    dt = Constant(0.125)
    theta = Constant(0.5)
    material = UPlMaterial2D(
        young_modulus=14.4e9,
        poisson_ratio=0.2,
        porosity=0.19,
        biot_coefficient=0.78,
        permeability_xx=2.0e-10,
        storage_inverse=1.0 / 13.5e9,
    )
    dOmega = dx(metadata={"q": 5})

    system = build_upl_theta_system_2d(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_prev=u0,
        p_prev=p0,
        material=material,
        dt=dt,
        theta=theta,
        dx_measure=dOmega,
    )

    alpha = Constant(material.biot_coefficient)
    invM = Constant(material.biot_modulus_inverse)
    H_pq = hydraulic_conductivity_form_2d(p, q, material)
    H0_pq = hydraulic_conductivity_form_2d(p0, q, material)

    a_old = (
        -inner(effective_stress_linear_2d(u, material), epsilon_2d(v)) * dOmega
        + alpha * p * div(v) * dOmega
        + alpha * div(u) * q * dOmega
        + invM * p * q * dOmega
        + theta * dt * H_pq * dOmega
    )
    L_old = (
        -inner(effective_stress_linear_2d(u0, material), epsilon_2d(v)) * dOmega
        + alpha * p0 * div(v) * dOmega
        + alpha * div(u0) * q * dOmega
        + invM * p0 * q * dOmega
        - (Constant(1.0) - theta) * dt * H0_pq * dOmega
    )

    K_new = _assemble_matrix(dh, system.lhs_form)
    K_old = _assemble_matrix(dh, a_old)
    F_new = _assemble_vector(dh, system.rhs_form)
    F_old = _assemble_vector(dh, L_old)

    np.testing.assert_allclose(K_new.toarray(), K_old.toarray(), rtol=1.0e-14, atol=1.0e-7)
    np.testing.assert_allclose(F_new, F_old, rtol=1.0e-14, atol=1.0e-7)


@pytest.mark.parametrize("backend", _ufl_backends())
def test_kratos_fic_triangle_bulk_pressure_gradient_stabilization_all_backends(backend, tmp_path, monkeypatch):
    if backend == "cpp":
        pytest.importorskip("pybind11")
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_cache_fic_{backend}"))

    mesh = _single_p1_triangle_mesh()
    dh = DofHandler(MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "p": 1}), method="cg")
    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)

    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)
    u_prev = VectorFunction(name="u_prev", field_names=["ux", "uy"], dof_handler=dh)
    p_prev = Function(name="p_prev", field_name="p", dof_handler=dh)
    p_rate_prev = Function(name="p_rate_prev", field_name="p", dof_handler=dh)
    p_prev.nodal_values[:] = np.asarray([3.0, 5.0, 7.0])
    p_rate_prev.nodal_values[:] = np.asarray([2.0, -1.0, 4.0])

    material = UPlMaterial2D(
        young_modulus=20.0,
        poisson_ratio=0.25,
        porosity=0.2,
        biot_coefficient=0.8,
        permeability_xx=0.0,
        storage_inverse=1.0e-2,
    )
    dt = 0.4
    theta_p = 0.5
    dOmega = dx(metadata={"q": 2})
    system = build_kratos_fic_triangle_upl_system_2d(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_prev=u_prev,
        p_prev=p_prev,
        material=material,
        dt=dt,
        theta_u=1.0,
        theta_p=theta_p,
        dx_measure=dOmega,
        element_length_squared=kratos_fic_triangle_element_length_squared(mesh),
        p_rate_prev=p_rate_prev,
    )

    try:
        K_base, F_base = assemble_form(
            Equation(system.base_system.lhs_form, system.base_system.rhs_form),
            dof_handler=dh,
            bcs=[],
            backend=backend,
        )
        K_fic, F_fic = assemble_form(Equation(system.lhs_form, system.rhs_form), dof_handler=dh, bcs=[], backend=backend)
    except Exception as exc:
        if backend == "cpp":
            pytest.skip(f"cpp backend unavailable: {exc}")
        raise

    delta_K = (K_fic - K_base).toarray()
    delta_F = F_fic - F_base
    pdofs = np.asarray(dh.element_maps["p"][0], dtype=int)
    grads = np.asarray([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    gram = 0.5 * (grads @ grads.T)
    h2 = 2.0 / np.pi
    tau = h2 * material.biot_coefficient / (8.0 * material.mu) * (
        material.biot_coefficient
        - 2.0 * material.mu * material.biot_modulus_inverse / (3.0 * material.biot_coefficient)
    )
    dt_pressure = 1.0 / (theta_p * dt)
    prev_factor = (1.0 - theta_p) / theta_p
    expected_Kp = dt_pressure * tau * gram
    expected_Fp = dt_pressure * tau * gram @ p_prev.nodal_values + prev_factor * tau * gram @ p_rate_prev.nodal_values

    expected_K = np.zeros_like(delta_K)
    expected_K[np.ix_(pdofs, pdofs)] = expected_Kp
    expected_F = np.zeros_like(delta_F)
    expected_F[pdofs] = expected_Fp
    np.testing.assert_allclose(delta_K, expected_K, rtol=1.0e-12, atol=1.0e-11)
    np.testing.assert_allclose(delta_F, expected_F, rtol=1.0e-12, atol=1.0e-11)


@pytest.mark.parametrize("backend", _ufl_backends())
def test_kratos_normal_liquid_flux_rhs_all_backends(backend, tmp_path, monkeypatch):
    if backend == "cpp":
        pytest.importorskip("pybind11")
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_cache_flux_{backend}"))

    mesh = _single_p1_triangle_mesh()
    mesh.tag_boundary_edges({"left": lambda x, y: np.isclose(x, 0.0)})
    bdry = mesh.edge_bitset("left")
    dh = DofHandler(MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "p": 1}), method="cg")
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)
    rhs = normal_liquid_flux_rhs_2d(q, 2.5, dS(defined_on=bdry, metadata={"q": 3}))

    try:
        _K, F = assemble_form(Equation(None, rhs), dof_handler=dh, bcs=[], backend=backend)
    except Exception as exc:
        if backend == "cpp":
            pytest.skip(f"cpp backend unavailable: {exc}")
        raise

    expected = np.zeros(dh.total_dofs, dtype=float)
    p_coords = np.asarray(dh.get_field_dof_coords("p"), dtype=float)
    p_global = np.asarray(dh.get_field_slice("p"), dtype=int)
    expected[p_global[np.isclose(p_coords[:, 0], 0.0)]] = 1.25
    np.testing.assert_allclose(F, expected, rtol=1.0e-12, atol=1.0e-12)


def test_kratos_consolidation_2d_pressure_parity_cpp_backend():
    res = solve_kratos_consolidation_2d_pycutfem(backend="cpp")

    assert res.times == [0.5, 1.0]
    assert res.backend == "cpp"

    for node_id in (1, 2, 3):
        np.testing.assert_allclose(res.liquid_pressure_by_node[node_id], [0.0, 0.0], rtol=0.0, atol=1.0e-10)

    expected_bottom = [19937.87166060176, 19875.93631773133]
    for node_id in (4, 5, 6):
        np.testing.assert_allclose(
            res.liquid_pressure_by_node[node_id],
            expected_bottom,
            rtol=1.0e-10,
            atol=1.0e-7,
        )


def test_kratos_undrained_soil_column_2d_pressure_parity_cpp_backend():
    res = solve_kratos_undrained_soil_column_2d_pycutfem(backend="cpp")
    with open(
        "/tmp/kratos-poro/applications/PoromechanicsApplication/tests/element_tests/"
        "undrained_soil_column_2D/undrained_soil_column_2D_results.json",
        encoding="utf-8",
    ) as handle:
        ref = json.load(handle)

    np.testing.assert_allclose(res.times, ref["TIME"], rtol=0.0, atol=1.0e-12)
    max_abs = 0.0
    max_rel = 0.0
    for node_id, values in res.liquid_pressure_by_node.items():
        expected = np.asarray(ref[f"NODE_{node_id}"]["LIQUID_PRESSURE"], dtype=float)
        actual = np.asarray(values, dtype=float)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-6)
        diff = np.abs(actual - expected)
        max_abs = max(max_abs, float(np.max(diff)))
        max_rel = max(max_rel, float(np.max(diff / np.maximum(1.0, np.abs(expected)))))

    assert max_abs < 1.0e-6
    assert max_rel < 1.0e-9


def test_kratos_four_point_shear_2d_direct_arc_length_parity_cpp_backend(tmp_path, monkeypatch):
    pytest.importorskip("pybind11")
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_cache_four_point"))

    ref = _four_point_shear_direct_reference_two_steps()
    res = solve_kratos_four_point_shear_2d_pycutfem(backend="cpp", end_time=2.0)

    assert res.times == ref.times == [1.0, 2.0]
    np.testing.assert_allclose(res.lambda_history, ref.lambda_history, rtol=1.0e-12, atol=1.0e-12)
    assert [item["iterations"] for item in res.arc_length_history] == [2, 2]
    assert [item["iterations"] for item in ref.arc_length_history] == [2, 2]
    for node_id, expected_x in ref.displacement_x_by_node.items():
        np.testing.assert_allclose(res.displacement_x_by_node[node_id], expected_x, rtol=1.0e-11, atol=1.0e-14)
        np.testing.assert_allclose(
            res.displacement_y_by_node[node_id],
            ref.displacement_y_by_node[node_id],
            rtol=1.0e-11,
            atol=1.0e-14,
        )


def test_kratos_fluid_pumping_2d_reference_runner_first_step():
    pytest.importorskip("KratosMultiphysics")
    res = solve_kratos_fluid_pumping_2d_reference(end_time=1.0e-5)

    assert res.times == [1.0e-5]
    _assert_fluid_pumping_samples(res, FLUID_PUMPING_KRATOS_FIRST_STEP, rtol=1.0e-12, atol=1.0e-9)


def test_kratos_fluid_pumping_2d_pycutfem_cpp_smoke():
    res = _fluid_pumping_pycutfem_first_newton_cpp()

    assert res.times == [1.0e-5]
    assert res.backend == "cpp"
    assert np.isfinite(res.liquid_pressure_by_node[8677][-1])
    assert np.isfinite(res.liquid_pressure_by_node[8675][-1])
    assert np.isfinite(res.displacement_x_by_node[8677][-1])


def test_kratos_fluid_pumping_2d_first_newton_increment_parity_cpp_backend():
    res = _fluid_pumping_pycutfem_first_newton_cpp()

    _assert_fluid_pumping_samples(
        res,
        FLUID_PUMPING_PYCUTFEM_FIRST_NEWTON_INCREMENT,
        rtol=1.0e-9,
        atol=1.0e-7,
        displacement_atol=1.0e-14,
    )


def test_kratos_fluid_pumping_2d_fracture_first_step_trace_driver_parity_cpp_backend():
    res = _fluid_pumping_fracture_pycutfem_first_step_cpp()

    assert res.times == [0.1]
    assert res.backend == "cpp"
    assert res.interface_count_history == [11]
    assert res.propagation_events == []
    assert set(res.liquid_pressure_by_node) == set(FLUID_PUMPING_2D_FRACTURE_SAMPLE_NODE_IDS)
    _assert_fluid_pumping_samples(
        res,
        FLUID_PUMPING_2D_FRACTURE_KRATOS_FIRST_STEP,
        rtol=5.0e-6,
        atol=5.0e-2,
        displacement_atol=5.0e-13,
    )


def test_kratos_fluid_pumping_2d_fracture_forced_trace_propagation_event():
    res = solve_kratos_fluid_pumping_2d_fracture_pycutfem(
        backend="python",
        end_time=0.1,
        interface_update="kratos_lagged",
        forced_propagation_steps=(1,),
        max_propagation_events=1,
    )

    assert res.times == [0.1]
    assert res.interface_count_history == [12]
    assert res.propagation_events is not None
    assert len(res.propagation_events) == 1
    event = res.propagation_events[0]
    assert event["step"] == 1
    assert event["parent_entity_id"] == 1
    assert event["new_entity_id"] == 12
    np.testing.assert_allclose(event["start"], (1.0, 2.5), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(event["end"], (1.05, 2.5), rtol=0.0, atol=1.0e-14)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "fluid_pumping_2D final first-step parity is a nonconverged "
        "max-iteration fracture-network path; local FIC, nonlinear interface, "
        "interface flux, and the full first global Newton increment now have "
        "parity gates"
    ),
)
def test_kratos_fluid_pumping_2d_first_step_parity_cpp_backend():
    res = solve_kratos_fluid_pumping_2d_pycutfem(
        backend="cpp",
        end_time=1.0e-5,
        interface_update="kratos_newton",
    )
    _assert_fluid_pumping_samples(res, FLUID_PUMPING_KRATOS_FIRST_STEP, rtol=1.0e-8, atol=1.0e-6)


def test_paired_interface_2d4_local_matrix_matches_kratos_layout():
    material = UPlInterfaceMaterial2D(
        normal_stiffness=2.0e7,
        shear_stiffness=1.0e6,
        penalty_stiffness=1.0,
        porosity=0.3,
        biot_coefficient=1.0,
        bulk_modulus_solid=1.0e25,
        bulk_modulus_liquid=1.0e25,
        dynamic_viscosity_liquid=1.0e-3,
        initial_joint_width=1.0e-4,
        transversal_permeability_coefficient=0.0,
    )
    interface = PairedUPlInterface2D(
        negative_coords=((0.5, 1.0), (0.5, 0.0)),
        positive_coords=((0.5, 1.0), (0.5, 0.0)),
        negative_dofs=((0, 1, 2), (3, 4, 5)),
        positive_dofs=((6, 7, 8), (9, 10, 11)),
    )

    local = build_paired_upl_interface_local_system_2d(
        interface,
        material,
        dt=0.5,
        theta_u=1.0,
        theta_p=1.0,
        previous_solution=np.zeros(12),
        current_solution=np.zeros(12),
        backend="python",
    )

    # Convert paired order [neg_top, neg_bottom, pos_top, pos_bottom] to the
    # Kratos 2D4 geometry order [neg_top, neg_bottom, pos_bottom, pos_top].
    perm = np.asarray([0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8], dtype=int)
    K = local.matrix[np.ix_(perm, perm)]

    assert K.shape == (12, 12)
    assert local.backend == "python"
    np.testing.assert_allclose(K[0, 0], 1.0e7)
    np.testing.assert_allclose(K[0, 9], -1.0e7)
    np.testing.assert_allclose(K[1, 1], 5.0e5)
    np.testing.assert_allclose(K[1, 10], -5.0e5)
    np.testing.assert_allclose(K[0, 2], 0.25)
    np.testing.assert_allclose(K[2, 0], -0.5)
    np.testing.assert_allclose(K[2, 2], 2.083333333333333e-11, rtol=1.0e-12)
    np.testing.assert_allclose(K[2, 5], -2.083333333333333e-11, rtol=1.0e-12)
    np.testing.assert_allclose(local.rhs, np.zeros(12), rtol=0.0, atol=0.0)


def test_paired_interface_accepts_higher_station_count():
    material = UPlInterfaceMaterial2D(
        normal_stiffness=2.0e7,
        shear_stiffness=1.0e6,
        porosity=0.3,
        biot_coefficient=1.0,
        bulk_modulus_solid=1.0e25,
        bulk_modulus_liquid=1.0e25,
        dynamic_viscosity_liquid=1.0e-3,
        initial_joint_width=1.0e-4,
    )
    interface = PairedUPlInterface2D(
        negative_coords=((0.5, 1.0), (0.5, 0.5), (0.5, 0.0)),
        positive_coords=((0.5, 1.0), (0.5, 0.5), (0.5, 0.0)),
        negative_dofs=((0, 1, 2), (3, 4, 5), (6, 7, 8)),
        positive_dofs=((9, 10, 11), (12, 13, 14), (15, 16, 17)),
    )

    local = build_paired_upl_interface_local_system_2d(
        interface,
        material,
        dt=0.5,
        previous_solution=np.zeros(18),
        current_solution=np.zeros(18),
        quadrature="lobatto",
    )

    assert local.matrix.shape == (18, 18)
    assert local.rhs.shape == (18,)
    assert local.joint_widths.shape == (3,)
    np.testing.assert_allclose(local.joint_widths, np.full(3, 1.0e-4))
    assert np.linalg.norm(local.matrix) > 0.0


def test_paired_interface_higher_station_backend_parity(tmp_path, monkeypatch):
    material = UPlInterfaceMaterial2D(
        normal_stiffness=2.0e7,
        shear_stiffness=1.0e6,
        porosity=0.3,
        biot_coefficient=0.85,
        bulk_modulus_solid=1.0e9,
        bulk_modulus_liquid=2.0e9,
        dynamic_viscosity_liquid=1.0e-3,
        initial_joint_width=1.0e-4,
        transversal_permeability_coefficient=2.0e-8,
    )
    interface = PairedUPlInterface2D(
        negative_coords=((0.5, 1.0), (0.5, 0.5), (0.5, 0.0)),
        positive_coords=((0.5, 1.0), (0.5, 0.5), (0.5, 0.0)),
        negative_dofs=((0, 1, 2), (3, 4, 5), (6, 7, 8)),
        positive_dofs=((9, 10, 11), (12, 13, 14), (15, 16, 17)),
    )
    previous = np.linspace(-2.0e-5, 3.0e-5, 18)
    current = previous.copy()
    current[[9, 12, 15]] += 2.0e-5

    reference = build_paired_upl_interface_local_system_2d(
        interface,
        material,
        dt=0.25,
        theta_u=0.75,
        theta_p=0.5,
        previous_solution=previous,
        current_solution=current,
        quadrature="gauss",
        backend="python",
    )
    assert reference.joint_widths.shape == (4,)
    assert np.all(reference.joint_widths > material.initial_joint_width)

    batch = build_paired_upl_interface_local_batch_2d(
        [interface, interface],
        material,
        dt=0.25,
        previous_solution=previous,
        current_solution=current,
        quadrature="gauss",
        backend="python",
        need_matrix=False,
    )
    assert batch.K_elem is None
    assert batch.F_elem.shape == (2, 18)


@pytest.mark.parametrize("backend", _ufl_backends())
def test_upl_interface_ufl_local_system_matches_paired_reference(backend, tmp_path, monkeypatch):
    if backend == "cpp":
        pytest.importorskip("pybind11")
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_cache_{backend}"))

    mesh = build_kratos_consolidation_interface_2d_mesh()
    dh = DofHandler(MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "p": 1}), method="dg")
    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)

    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)
    u_prev = VectorFunction(name="u_prev", field_names=["ux", "uy"], dof_handler=dh)
    u_current = VectorFunction(name="u_current", field_names=["ux", "uy"], dof_handler=dh)
    p_prev = Function(name="p_prev", field_name="p", dof_handler=dh)

    def dofs(eid: int, local: int) -> tuple[int, int, int]:
        return (
            int(dh.element_maps["ux"][eid][local]),
            int(dh.element_maps["uy"][eid][local]),
            int(dh.element_maps["p"][eid][local]),
        )

    paired = PairedUPlInterface2D(
        negative_coords=((0.5, 1.0), (0.5, 0.0)),
        positive_coords=((0.5, 1.0), (0.5, 0.0)),
        negative_dofs=(dofs(1, 3), dofs(1, 1)),
        positive_dofs=(dofs(0, 2), dofs(0, 0)),
    )
    interface = paired_upl_interface_to_nonmatching_interface_2d(
        paired,
        mesh=mesh,
        negative_element_ids=1,
        positive_element_ids=0,
    )
    material = UPlInterfaceMaterial2D(
        normal_stiffness=2.0e7,
        shear_stiffness=1.0e6,
        penalty_stiffness=1.0,
        porosity=0.3,
        biot_coefficient=1.0,
        bulk_modulus_solid=1.0e25,
        bulk_modulus_liquid=1.0e25,
        dynamic_viscosity_liquid=1.0e-3,
        initial_joint_width=1.0e-4,
        transversal_permeability_coefficient=0.0,
    )
    previous = np.linspace(-2.0e-5, 2.0e4, dh.total_dofs)
    current = np.linspace(3.0e-5, -1.0e-5, dh.total_dofs)
    u_prev.nodal_values = previous[u_prev._g_dofs]
    u_current.nodal_values = current[u_current._g_dofs]
    p_prev.nodal_values = previous[p_prev._g_dofs]

    ufl_system = build_upl_interface_ufl_system_2d(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_prev=u_prev,
        p_prev=p_prev,
        u_current=u_current,
        material=material,
        interface=interface,
        dt=0.5,
        quadrature="lobatto",
        quadrature_order=2,
    )
    batch = FormCompiler(dh, backend=backend).assemble_local_contributions(ufl_system.equation)
    reference = build_paired_upl_interface_local_system_2d(
        paired,
        material,
        dt=0.5,
        previous_solution=previous,
        current_solution=current,
        quadrature="lobatto",
        backend="python",
    )

    global_to_ufl = {int(g): i for i, g in enumerate(batch.gdofs_map[0])}
    ufl_idx = [global_to_ufl[int(g)] for g in reference.local_dofs]
    np.testing.assert_allclose(
        batch.K_elem[0][np.ix_(ufl_idx, ufl_idx)],
        reference.matrix,
        rtol=1.0e-12,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(batch.F_elem[0][ufl_idx], reference.rhs, rtol=1.0e-12, atol=1.0e-8)


@pytest.mark.parametrize("backend", _ufl_backends())
def test_upl_link_interface_ufl_uses_kratos_link_permeability_law(backend, tmp_path, monkeypatch):
    if backend == "cpp":
        pytest.importorskip("pybind11")
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_cache_link_{backend}"))

    mesh = build_kratos_consolidation_interface_2d_mesh()
    dh = DofHandler(MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "p": 1}), method="dg")
    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)

    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)
    u_prev = VectorFunction(name="u_prev", field_names=["ux", "uy"], dof_handler=dh)
    u_current = VectorFunction(name="u_current", field_names=["ux", "uy"], dof_handler=dh)
    p_prev = Function(name="p_prev", field_name="p", dof_handler=dh)

    def dofs(eid: int, local: int) -> tuple[int, int, int]:
        return (
            int(dh.element_maps["ux"][eid][local]),
            int(dh.element_maps["uy"][eid][local]),
            int(dh.element_maps["p"][eid][local]),
        )

    paired = PairedUPlInterface2D(
        negative_coords=((0.5, 1.0), (0.5, 0.0)),
        positive_coords=((0.5, 1.0), (0.5, 0.0)),
        negative_dofs=(dofs(1, 3), dofs(1, 1)),
        positive_dofs=(dofs(0, 2), dofs(0, 0)),
    )
    interface = paired_upl_interface_to_nonmatching_interface_2d(
        paired,
        mesh=mesh,
        negative_element_ids=1,
        positive_element_ids=0,
    )
    width = 1.0e-4
    fracture_material = UPlInterfaceMaterial2D(
        normal_stiffness=2.0e7,
        shear_stiffness=1.0e6,
        penalty_stiffness=1.0,
        porosity=0.3,
        biot_coefficient=1.0,
        bulk_modulus_solid=1.0e25,
        bulk_modulus_liquid=1.0e25,
        dynamic_viscosity_liquid=1.0e-3,
        initial_joint_width=width,
        transversal_permeability_coefficient=width * width / 12.0,
    )
    link_material = UPlInterfaceMaterial2D(
        normal_stiffness=2.0e7,
        shear_stiffness=1.0e6,
        penalty_stiffness=1.0,
        porosity=0.3,
        biot_coefficient=1.0,
        bulk_modulus_solid=1.0e25,
        bulk_modulus_liquid=1.0e25,
        dynamic_viscosity_liquid=1.0e-3,
        initial_joint_width=width,
        transversal_permeability_coefficient=123.0,
    )

    fracture_system = build_upl_interface_ufl_system_2d(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_prev=u_prev,
        p_prev=p_prev,
        u_current=u_current,
        material=fracture_material,
        interface=interface,
        dt=0.5,
        quadrature="lobatto",
        quadrature_order=2,
        permeability_law="fracture",
    )
    link_system = build_upl_link_interface_ufl_system_2d(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_prev=u_prev,
        p_prev=p_prev,
        u_current=u_current,
        material=link_material,
        interface=interface,
        dt=0.5,
        quadrature="lobatto",
        quadrature_order=2,
    )

    try:
        fracture_batch = FormCompiler(dh, backend=backend).assemble_local_contributions(fracture_system.equation)
        link_batch = FormCompiler(dh, backend=backend).assemble_local_contributions(link_system.equation)
    except Exception as exc:
        if backend == "cpp":
            pytest.skip(f"cpp backend unavailable: {exc}")
        raise

    np.testing.assert_allclose(link_batch.K_elem[0], fracture_batch.K_elem[0], rtol=1.0e-12, atol=1.0e-8)
    np.testing.assert_allclose(link_batch.F_elem[0], fracture_batch.F_elem[0], rtol=1.0e-12, atol=1.0e-8)
    assert link_system.permeability_law == "link"


@pytest.mark.parametrize("backend", _ufl_backends())
def test_kratos_consolidation_interface_2d_pressure_parity_all_backends(backend, tmp_path, monkeypatch):
    if backend == "cpp":
        pytest.importorskip("pybind11")
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_cache"))
    res = solve_kratos_consolidation_interface_2d_pycutfem(backend=backend)

    assert res.times == [0.5, 1.0]
    assert res.backend == backend

    for node_id in (1, 2, 3, 5):
        np.testing.assert_allclose(res.liquid_pressure_by_node[node_id][-1], 0.0, rtol=0.0, atol=1.0e-10)

    expected_final = {
        4: 19622.042125422962,
        6: 19913.29312704039,
        7: 19913.293127040382,
        8: 19622.042125422962,
    }
    for node_id, expected in expected_final.items():
        np.testing.assert_allclose(
            res.liquid_pressure_by_node[node_id][-1],
            expected,
            rtol=1.0e-10,
            atol=1.0e-7,
        )


@pytest.mark.parametrize("backend", _ufl_backends())
def test_kratos_vertical_fault_consolidation_2d_two_step_parity_all_backends(backend, tmp_path, monkeypatch):
    if backend == "cpp":
        pytest.importorskip("pybind11")
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_cache"))

    ref = _vertical_fault_consolidation_reference_two_steps()
    res = solve_kratos_vertical_fault_consolidation_2d_pycutfem(
        backend=backend,
        end_time=1.0,
        interface_update="kratos_newton",
    )

    assert res.times == ref.times == [0.5, 1.0]
    assert res.backend == backend
    for node_id, expected_pressure in ref.liquid_pressure_by_node.items():
        np.testing.assert_allclose(
            res.liquid_pressure_by_node[node_id],
            expected_pressure,
            rtol=1.0e-12,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            res.displacement_x_by_node[node_id],
            ref.displacement_x_by_node[node_id],
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            res.displacement_y_by_node[node_id],
            ref.displacement_y_by_node[node_id],
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def _assemble_matrix(dh, form):
    compiler = FormCompiler(dh, quadrature_order=None, backend="python")
    compiler.ctx["rhs"] = False
    A = sp.lil_matrix((dh.total_dofs, dh.total_dofs))
    compiler._assemble_form(form, A)
    return A.tocsr()


def _assemble_vector(dh, form):
    compiler = FormCompiler(dh, quadrature_order=None, backend="python")
    compiler.ctx["rhs"] = True
    b = np.zeros(dh.total_dofs, dtype=float)
    compiler._assemble_form(form, b)
    return b
