"""Small Kratos Poromechanics parity targets implemented with pycutfem forms."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.core.topology import Node
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.integration.quadrature import gauss_lobatto
from pycutfem.nonmatching.interface import NonMatchingInterface
from pycutfem.state import QuadratureLayout, StateRegistry
from pycutfem.state import build_volume_nonlocal_quadrature_map
from pycutfem.tracefem import (
    TraceFractureNetwork2D,
    TraceFracturePropagationSettings2D,
    TraceStationEntity2D,
    plan_fracture_extensions_from_damage,
    transfer_trace_quadrature_state,
)
from pycutfem.solvers.arc_length import (
    RammArcLengthParameters,
    initialize_ramm_arc_length_state,
    ramm_arc_length_step,
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
    dot,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dS, dx
from pycutfem.ufl.spaces import FunctionSpace

from .materials import UPlMaterial2D
from .damage import (
    ModifiedMisesNonlocalDamagePlaneStress2D,
    create_nonlocal_damage_state_2d,
    stage_modified_mises_nonlocal_damage_2d,
    volume_strain_voigt_2d,
)
from .interface import (
    PairedUPlInterface2D,
    UPlInterfaceMaterial2D,
    assemble_paired_upl_interface_2d,
    build_paired_upl_interface_kratos_newton_batch_2d,
    build_paired_upl_interface_local_batch_2d,
    build_upl_elastic_kratos_newton_interface_ufl_system_2d,
    build_upl_interface_ufl_system_2d,
    build_upl_kratos_newton_interface_ufl_system_2d,
    build_upl_link_interface_ufl_system_2d,
    paired_upl_interfaces_to_trace_link_interface_2d,
    paired_upl_interface_to_nonmatching_interface_2d,
)
from .kratos_io import (
    field_values_at_mdpa_nodes,
    load_kratos_json,
    mdpa_volume_mesh_to_pycutfem,
    parse_kratos_mdpa,
)
from .upl import (
    build_kratos_fic_triangle_damage_upl_system_2d,
    build_kratos_quasistatic_damage_upl_system_2d,
    build_kratos_fic_triangle_upl_system_2d,
    build_kratos_quasistatic_upl_system_2d,
    kratos_fic_triangle_element_length_squared,
)


@dataclass(frozen=True)
class KratosConsolidation2DResult:
    times: list[float]
    liquid_pressure_by_node: dict[int, list[float]]
    backend: str
    displacement_x_by_node: dict[int, list[float]] | None = None
    displacement_y_by_node: dict[int, list[float]] | None = None
    newton_history: list[dict[str, float]] | None = None


@dataclass(frozen=True)
class KratosFluidPumping2DResult:
    times: list[float]
    liquid_pressure_by_node: dict[int, list[float]]
    displacement_x_by_node: dict[int, list[float]]
    displacement_y_by_node: dict[int, list[float]]
    backend: str
    n_steps: int
    newton_history: list[dict[str, object]] | None = None
    propagation_events: list[dict[str, object]] | None = None
    interface_count_history: list[int] | None = None
    bulk_damage_history: list[float] | None = None
    tip_damage_history: list[float] | None = None
    bulk_damage_location_history: list[tuple[float, float]] | None = None
    tip_damage_support_history: list[int] | None = None
    tip_damage_centroid_history: list[tuple[float, float]] | None = None


@dataclass(frozen=True)
class KratosFourPointShear2DResult:
    times: list[float]
    displacement_x_by_node: dict[int, list[float]]
    displacement_y_by_node: dict[int, list[float]]
    backend: str
    lambda_history: list[float] | None = None
    arc_length_history: list[dict[str, object]] | None = None


KRATOS_PORO_TESTS_ROOT = Path("/tmp/kratos-poro/applications/PoromechanicsApplication/tests")
KRATOS_EXAMPLES_PORO_ROOT = Path("/tmp/kratos-examples-poro/poromechanics")
FLUID_PUMPING_2D_SAMPLE_NODE_IDS = (8677, 8675, 1201, 3686, 820, 8545)
FLUID_PUMPING_2D_FRACTURE_SAMPLE_NODE_IDS = (853, 953, 955, 833, 95)
VERTICAL_FAULT_CONSOLIDATION_SAMPLE_NODE_IDS = (1, 24, 61, 75, 88, 104, 117, 132)
FOUR_POINT_SHEAR_SAMPLE_NODE_IDS = (1, 985, 1020, 1130, 2972, 3018, 3103)


def build_kratos_consolidation_2d_mesh() -> Mesh:
    """Build Kratos `element_tests/consolidation_2D` as a pycutfem Q1 quad mesh."""

    # Kratos node ids are one-based; pycutfem node ids are zero-based here.
    coords = [
        (0.0, 1.0),
        (0.5, 1.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
    ]
    nodes = [Node(i, float(x), float(y)) for i, (x, y) in enumerate(coords)]

    # pycutfem Q1 lattice order is [bottom-left, bottom-right, top-left, top-right].
    element_connectivity = np.asarray(
        [
            [3, 4, 0, 1],
            [4, 5, 1, 2],
        ],
        dtype=int,
    )
    corner_connectivity = np.asarray(
        [
            [3, 4, 1, 0],
            [4, 5, 2, 1],
        ],
        dtype=int,
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=element_connectivity,
        elements_corner_nodes=corner_connectivity,
        element_type="quad",
        poly_order=1,
    )
    mesh.tag_boundary_edges(
        {
            "top": lambda x, y: np.isclose(y, 1.0),
            "bottom": lambda x, y: np.isclose(y, 0.0),
            "left_right": lambda x, y: np.isclose(x, 0.0) or np.isclose(x, 1.0),
        }
    )
    return mesh


def build_kratos_consolidation_interface_2d_mesh() -> Mesh:
    """Build Kratos `element_tests/consolidation_interface_2D` as two DG Q1 quads."""

    coords = [
        (1.0, 1.0),  # Kratos node 1
        (0.5, 1.0),  # Kratos node 2, left side of interface
        (0.5, 1.0),  # Kratos node 3, right side of interface
        (1.0, 0.0),  # Kratos node 4
        (0.0, 1.0),  # Kratos node 5
        (0.5, 0.0),  # Kratos node 6, right side of interface
        (0.5, 0.0),  # Kratos node 7, left side of interface
        (0.0, 0.0),  # Kratos node 8
    ]
    nodes = [Node(i, float(x), float(y)) for i, (x, y) in enumerate(coords)]

    # Element 0 is the right Kratos body [4,1,3,6], element 1 is the left
    # Kratos body [7,2,5,8]. pycutfem Q1 lattice order is
    # [bottom-left, bottom-right, top-left, top-right].
    element_connectivity = np.asarray(
        [
            [5, 3, 2, 0],
            [7, 6, 4, 1],
        ],
        dtype=int,
    )
    corner_connectivity = np.asarray(
        [
            [5, 3, 0, 2],
            [7, 6, 1, 4],
        ],
        dtype=int,
    )
    return Mesh(
        nodes=nodes,
        element_connectivity=element_connectivity,
        elements_corner_nodes=corner_connectivity,
        element_type="quad",
        poly_order=1,
    )


def solve_kratos_consolidation_2d_pycutfem(*, backend: str = "cpp") -> KratosConsolidation2DResult:
    """Reproduce Kratos Poromechanics `element_tests/consolidation_2D`.

    Reference source:
    `/tmp/kratos-poro/applications/PoromechanicsApplication/tests/element_tests/consolidation_2D`.
    """

    mesh = build_kratos_consolidation_2d_mesh()
    mixed_element = MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "p": 1})
    dh = DofHandler(mixed_element, method="cg")

    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)

    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)

    u_prev = VectorFunction(name="u_prev", field_names=["ux", "uy"], dof_handler=dh)
    p_prev = Function(name="p_prev", field_name="p", dof_handler=dh)
    u_prev.nodal_values.fill(0.0)
    p_prev.nodal_values.fill(0.0)

    material = UPlMaterial2D(
        young_modulus=1.0e6,
        poisson_ratio=0.3,
        porosity=0.2,
        biot_coefficient=1.0,
        bulk_modulus_solid=1.0e25,
        bulk_modulus_liquid=1.0e25,
        permeability_xx=1.15740740740741e-12,
        permeability_yy=1.15740740740741e-12,
        dynamic_viscosity_liquid=1.0e-3,
        density_solid=2.0e3,
        density_liquid=1.0e3,
    )

    dt = 0.5
    theta = 1.0
    system = build_kratos_quasistatic_upl_system_2d(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_prev=u_prev,
        p_prev=p_prev,
        material=material,
        dt=dt,
        theta_u=theta,
        theta_p=theta,
        dx_measure=dx(metadata={"q": 2}),
    )

    face_load = Constant(np.asarray([0.0, -1.0e4], dtype=float))
    face_load._jit_name = "face_load"
    lhs = system.lhs_form
    rhs = system.rhs_form + dot(face_load, v) * dS(mesh.edge_bitset("top"), metadata={"q": 2})

    bcs = [
        BoundaryCondition("ux", "dirichlet", "bottom", lambda x, y: 0.0),
        BoundaryCondition("uy", "dirichlet", "bottom", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "left_right", lambda x, y: 0.0),
        BoundaryCondition("p", "dirichlet", "top", lambda x, y: 0.0),
    ]

    times: list[float] = []
    pressure_by_node: dict[int, list[float]] = {i: [] for i in range(1, 7)}

    for step in range(1, 3):
        K, F = assemble_form(Equation(lhs, rhs), dof_handler=dh, bcs=bcs, backend=backend)
        w = sp_la.spsolve(K.tocsc(), F)

        u_prev.nodal_values = w[u_prev._g_dofs]
        p_prev.nodal_values = w[p_prev._g_dofs]
        times.append(0.5 * float(step))

        p_slice = np.asarray(dh.get_field_slice("p"), dtype=int)
        p_coords = np.asarray(dh.get_field_dof_coords("p"), dtype=float)
        p_values = np.asarray(w[p_slice], dtype=float)
        for node_id, node in enumerate(mesh.nodes_list, start=1):
            xy = np.asarray([node.x, node.y], dtype=float)
            match = np.where(np.linalg.norm(p_coords - xy, axis=1) < 1.0e-12)[0]
            if match.size != 1:
                raise RuntimeError(f"Could not map pressure dof for Kratos node {node_id}.")
            pressure_by_node[node_id].append(float(p_values[int(match[0])]))

    return KratosConsolidation2DResult(times=times, liquid_pressure_by_node=pressure_by_node, backend=backend)


def solve_kratos_undrained_soil_column_2d_pycutfem(
    *,
    backend: str = "cpp",
    root: str | Path | None = None,
    load_time_rule: str = "current",
) -> KratosConsolidation2DResult:
    """Reproduce Kratos `element_tests/undrained_soil_column_2D`.

    This is the first mixed-order validation target: Kratos uses
    `SmallStrainUPlDiffOrderElement2D9N`, i.e. Q2 displacement and Q1 liquid
    pressure on a Q9 geometry. The pycutfem solve imports the mdpa mesh instead
    of hand-building it and compares liquid pressure at the Kratos JSON output
    times.
    """

    case_root = (
        Path(root)
        if root is not None
        else KRATOS_PORO_TESTS_ROOT / "element_tests" / "undrained_soil_column_2D"
    )
    mdpa = parse_kratos_mdpa(case_root / "undrained_soil_column_2D.mdpa")
    params = load_kratos_json(case_root / "ProjectParameters.json")
    materials = load_kratos_json(case_root / "PoroMaterials.json")
    variables = materials["properties"][0]["Material"]["Variables"]

    mesh = mdpa_volume_mesh_to_pycutfem(
        mdpa,
        element_block_names=("SmallStrainUPlDiffOrderElement2D9N",),
        boundary_node_tags={
            "Face_Load-auto-1": "face_load",
        },
        boundary_condition_tags={
            "Face_Load-auto-1": "face_load",
        },
    )
    mixed_element = MixedElement(mesh, field_specs={"ux": 2, "uy": 2, "p": 1})
    dh = DofHandler(mixed_element, method="cg")

    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)
    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)
    u_prev = VectorFunction(name="u_prev", field_names=["ux", "uy"], dof_handler=dh)
    p_prev = Function(name="p_prev", field_name="p", dof_handler=dh)
    velocity_prev = VectorFunction(name="velocity_prev", field_names=["ux", "uy"], dof_handler=dh)
    p_rate_prev = Function(name="p_rate_prev", field_name="p", dof_handler=dh)
    u_prev.nodal_values.fill(0.0)
    p_prev.nodal_values.fill(0.0)
    velocity_prev.nodal_values.fill(0.0)
    p_rate_prev.nodal_values.fill(0.0)

    material = UPlMaterial2D(
        young_modulus=float(variables["YOUNG_MODULUS"]),
        poisson_ratio=float(variables["POISSON_RATIO"]),
        porosity=float(variables["POROSITY"]),
        biot_coefficient=float(variables["BIOT_COEFFICIENT"]),
        bulk_modulus_solid=float(variables["BULK_MODULUS_SOLID"]),
        bulk_modulus_liquid=float(variables["BULK_MODULUS_LIQUID"]),
        permeability_xx=float(variables["PERMEABILITY_XX"]),
        permeability_yy=float(variables["PERMEABILITY_YY"]),
        permeability_xy=float(variables.get("PERMEABILITY_XY", 0.0)),
        dynamic_viscosity_liquid=float(variables["DYNAMIC_VISCOSITY_LIQUID"]),
        density_solid=float(variables.get("DENSITY_SOLID", 0.0)),
        density_liquid=float(variables.get("DENSITY_LIQUID", 0.0)),
        thickness=float(variables.get("THICKNESS", 1.0)),
    )

    solver_settings = params["solver_settings"]
    dt_value = float(solver_settings["time_step"])
    theta_u = float(solver_settings.get("newmark_theta_u", solver_settings.get("newmark_theta", 1.0)))
    theta_p = float(solver_settings.get("newmark_theta_p", solver_settings.get("newmark_theta", 1.0)))
    end_time = float(params["problem_data"]["end_time"])

    system = build_kratos_quasistatic_upl_system_2d(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_prev=u_prev,
        p_prev=p_prev,
        material=material,
        dt=dt_value,
        theta_u=theta_u,
        theta_p=theta_p,
        dx_measure=dx(metadata={"q": 3}),
        velocity_prev=velocity_prev,
        p_rate_prev=p_rate_prev,
    )
    equation = Equation(system.lhs_form, system.rhs_form)

    dirichlet_dofs = _kratos_process_dirichlet_dofs(
        dh,
        mesh,
        mdpa,
        params["processes"]["constraints_process_list"],
    )
    face_load_process = params["processes"]["loads_process_list"][0]["Parameters"]
    face_load_part = str(face_load_process["model_part_name"]).split(".")[-1]
    face_load_component = int(np.flatnonzero(np.asarray(face_load_process["active"], dtype=bool))[0])
    face_load_table = int(face_load_process.get("table", [0, 0, 0])[face_load_component])
    face_load_base = float(face_load_process.get("value", [0.0, 0.0, 0.0])[face_load_component])

    times: list[float] = []
    pressure_by_node: dict[int, list[float]] = {int(node_id): [] for node_id in mdpa.nodes}
    w_prev = np.zeros(dh.total_dofs, dtype=float)
    velocity_prev_values = np.zeros_like(velocity_prev.nodal_values, dtype=float)
    p_rate_prev_values = np.zeros_like(p_rate_prev.nodal_values, dtype=float)
    n_steps = int(round(end_time / dt_value))

    for step in range(1, n_steps + 1):
        time = dt_value * float(step)
        u_prev_values = w_prev[u_prev._g_dofs].copy()
        p_prev_values = w_prev[p_prev._g_dofs].copy()
        u_prev.nodal_values = u_prev_values
        p_prev.nodal_values = p_prev_values
        velocity_prev.nodal_values = velocity_prev_values
        p_rate_prev.nodal_values = p_rate_prev_values

        K, F = assemble_form(equation, dof_handler=dh, bcs=[], backend=backend)
        F = np.asarray(F, dtype=float)
        if load_time_rule == "current":
            load_time = time
        elif load_time_rule == "previous":
            load_time = max(0.0, time - dt_value)
        elif load_time_rule == "theta":
            load_time = max(0.0, time - (1.0 - theta_u) * dt_value)
        else:
            raise ValueError("load_time_rule must be 'current', 'previous', or 'theta'.")
        load_value = mdpa.tables[face_load_table].value(load_time) if face_load_table else face_load_base
        _add_kratos_diff_order_line_load_y(F, dh, mesh, mdpa, face_load_part, load_value)
        K_bc, F_bc = _apply_dirichlet_zero(K.tolil(), F, dirichlet_dofs)
        w_new = np.asarray(sp_la.spsolve(K_bc.tocsc(), F_bc), dtype=float)
        velocity_prev_values = (
            w_new[velocity_prev._g_dofs]
            - u_prev_values
            - (1.0 - theta_u) * dt_value * velocity_prev_values
        ) / (theta_u * dt_value)
        p_rate_prev_values = (
            w_new[p_rate_prev._g_dofs]
            - p_prev_values
            - (1.0 - theta_p) * dt_value * p_rate_prev_values
        ) / (theta_p * dt_value)
        w_prev = w_new

        if step % 25 == 0:
            times.append(time)
            pressure_now = field_values_at_mdpa_nodes(mesh, dh, w_prev, "p")
            for node_id, value in pressure_now.items():
                pressure_by_node[int(node_id)].append(float(value))

    return KratosConsolidation2DResult(times=times, liquid_pressure_by_node=pressure_by_node, backend=backend)


def solve_kratos_consolidation_interface_2d_pycutfem(
    *,
    backend: str = "cpp",
    interface_update: str = "kratos_newton",
) -> KratosConsolidation2DResult:
    """Reproduce Kratos Poromechanics `element_tests/consolidation_interface_2D`.

    The volume and cohesive U-Pl interface terms are assembled by the selected
    pycutfem backend. The interface is written once as a symbolic
    ``dNonmatchingInterface`` form and uses two-point Lobatto quadrature for the
    Kratos small-test parity path.

    ``interface_update="kratos_lagged"`` reproduces the Kratos small-test
    reference exactly. ``"fixed_point"`` is available for experiments that want
    to iterate the displacement-dependent joint width inside each time step.
    """

    mesh = build_kratos_consolidation_interface_2d_mesh()
    mixed_element = MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "p": 1})
    dh = DofHandler(mixed_element, method="dg")

    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)

    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)

    u_prev = VectorFunction(name="u_prev", field_names=["ux", "uy"], dof_handler=dh)
    u_current = VectorFunction(name="u_current", field_names=["ux", "uy"], dof_handler=dh)
    p_prev = Function(name="p_prev", field_name="p", dof_handler=dh)
    u_prev.nodal_values.fill(0.0)
    u_current.nodal_values.fill(0.0)
    p_prev.nodal_values.fill(0.0)

    body_material = UPlMaterial2D(
        young_modulus=1.0e6,
        poisson_ratio=0.3,
        porosity=0.2,
        biot_coefficient=1.0,
        bulk_modulus_solid=1.0e25,
        bulk_modulus_liquid=1.0e25,
        permeability_xx=1.15740740740741e-12,
        permeability_yy=1.15740740740741e-12,
        dynamic_viscosity_liquid=1.0e-3,
        density_solid=2.0e3,
        density_liquid=1.0e3,
    )
    interface_material = UPlInterfaceMaterial2D(
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
        density_solid=2.0e3,
        density_liquid=1.0e3,
    )

    dt_value = 0.5
    theta_value = 1.0
    system = build_kratos_quasistatic_upl_system_2d(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_prev=u_prev,
        p_prev=p_prev,
        material=body_material,
        dt=dt_value,
        theta_u=theta_value,
        theta_p=theta_value,
        dx_measure=dx(metadata={"q": 2}),
    )

    interface = _kratos_consolidation_interface_trace(dh)
    interface_nm = paired_upl_interface_to_nonmatching_interface_2d(
        interface,
        mesh=mesh,
        negative_element_ids=1,
        positive_element_ids=0,
    )
    interface_system = build_upl_interface_ufl_system_2d(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_prev=u_prev,
        p_prev=p_prev,
        u_current=u_current,
        material=interface_material,
        interface=interface_nm,
        dt=dt_value,
        theta_u=theta_value,
        theta_p=theta_value,
        quadrature="lobatto",
        quadrature_order=2,
    )
    coupled_equation = Equation(
        system.lhs_form + interface_system.lhs_form,
        system.rhs_form + interface_system.rhs_form,
    )
    face_load_dofs = _kratos_consolidation_interface_face_load_dofs(dh)
    dirichlet_dofs = _kratos_consolidation_interface_dirichlet_dofs(dh)
    pressure_dofs_by_node = _kratos_consolidation_interface_pressure_dofs_by_node(dh)

    times: list[float] = []
    pressure_by_node: dict[int, list[float]] = {i: [] for i in range(1, 9)}
    w_prev = np.zeros(dh.total_dofs, dtype=float)

    for step in range(1, 3):
        u_prev.nodal_values = w_prev[u_prev._g_dofs]
        p_prev.nodal_values = w_prev[p_prev._g_dofs]
        guess = w_prev.copy()

        max_interface_iterations = 1 if interface_update == "kratos_lagged" else 15
        if interface_update not in {"kratos_lagged", "fixed_point"}:
            raise ValueError("interface_update must be 'kratos_lagged' or 'fixed_point'.")

        for _iteration in range(max_interface_iterations):
            u_current.nodal_values = guess[u_current._g_dofs]
            K, F = assemble_form(coupled_equation, dof_handler=dh, bcs=[], backend=backend)
            K = K.tolil()
            F = np.asarray(F, dtype=float)
            _add_kratos_top_face_load(F, face_load_dofs)
            K_bc, F_bc = _apply_dirichlet_zero(K, F, dirichlet_dofs)
            w = sp_la.spsolve(K_bc.tocsc(), F_bc)
            if interface_update == "kratos_lagged":
                break
            if np.linalg.norm(w - guess, ord=np.inf) <= 1.0e-10 * max(1.0, np.linalg.norm(w, ord=np.inf)):
                break
            guess = np.asarray(w, dtype=float)
        else:
            raise RuntimeError("Kratos interface solve did not converge in 15 fixed-point iterations.")

        w_prev = np.asarray(w, dtype=float)
        times.append(0.5 * float(step))
        for node_id, dof in pressure_dofs_by_node.items():
            pressure_by_node[node_id].append(float(w_prev[int(dof)]))

    return KratosConsolidation2DResult(times=times, liquid_pressure_by_node=pressure_by_node, backend=backend)


def solve_kratos_vertical_fault_consolidation_2d_pycutfem(
    *,
    backend: str = "cpp",
    root: str | Path | None = None,
    end_time: float = 0.5,
    output_node_ids: tuple[int, ...] = VERTICAL_FAULT_CONSOLIDATION_SAMPLE_NODE_IDS,
    interface_update: str = "kratos_newton",
    max_interface_iterations_override: int | None = None,
    collect_newton_history: bool = False,
) -> KratosConsolidation2DResult:
    """Run the full Kratos examples vertical-fault consolidation case locally.

    The imported validation case uses Q1 U-Pl volume quads and 2-node station
    cohesive interface segments. The pycutfem path keeps this generic: the
    volume mesh is created from the MDPA block with ``MixedElement`` and the
    fault is represented as a finite-aperture ``TraceLinkInterface`` assembled
    through the selected UFL backend.
    """

    case_root = (
        Path(root)
        if root is not None
        else KRATOS_EXAMPLES_PORO_ROOT / "validation" / "consolidation_interface_2D" / "source"
    )
    mdpa = parse_kratos_mdpa(case_root / "consolidation_interface_2D.mdpa")
    params = load_kratos_json(case_root / "ProjectParameters.json")
    materials = load_kratos_json(case_root / "PoroMaterials.json")

    mesh = mdpa_volume_mesh_to_pycutfem(
        mdpa,
        element_block_names=("UPwSmallStrainElement2D4N",),
    )
    mixed_element = MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "p": 1})
    dh = DofHandler(mixed_element, method="cg")
    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)
    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)
    u_prev = VectorFunction(name="u_prev", field_names=["ux", "uy"], dof_handler=dh)
    u_current = VectorFunction(name="u_current", field_names=["ux", "uy"], dof_handler=dh)
    p_prev = Function(name="p_prev", field_name="p", dof_handler=dh)
    p_current = Function(name="p_current", field_name="p", dof_handler=dh)
    velocity_prev = VectorFunction(name="velocity_prev", field_names=["ux", "uy"], dof_handler=dh)
    velocity_current = VectorFunction(name="velocity_current", field_names=["ux", "uy"], dof_handler=dh)
    p_rate_prev = Function(name="p_rate_prev", field_name="p", dof_handler=dh)
    p_rate_current = Function(name="p_rate_current", field_name="p", dof_handler=dh)
    for coeff in (u_prev, u_current, p_prev, p_current, velocity_prev, velocity_current, p_rate_prev, p_rate_current):
        coeff.nodal_values.fill(0.0)

    material_by_part = _kratos_material_variables_by_part(materials)
    body_material = _upl_material_from_kratos_variables(material_by_part["Body_Part-auto-1"])
    interface_material = _upl_interface_material_from_kratos_variables(material_by_part["Interface_Part-auto-1"])

    solver_settings = params["solver_settings"]
    dt_value = float(solver_settings["time_step"])
    theta = float(solver_settings.get("newmark_theta", solver_settings.get("newmark_theta_u", 1.0)))
    theta_u = float(solver_settings.get("newmark_theta_u", theta))
    theta_p = float(solver_settings.get("newmark_theta_p", theta))
    n_steps = int(round(float(end_time) / dt_value))
    if n_steps < 1:
        raise ValueError("end_time must cover at least one time step.")

    bulk_system = build_kratos_quasistatic_upl_system_2d(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_prev=u_prev,
        p_prev=p_prev,
        material=body_material,
        dt=dt_value,
        theta_u=theta_u,
        theta_p=theta_p,
        dx_measure=dx(metadata={"q": 2}),
        velocity_prev=velocity_prev,
        p_rate_prev=p_rate_prev,
    )

    paired_interfaces = _paired_interfaces_from_mdpa_owner_dofs(
        mdpa,
        dh,
        mesh,
        body_block_name="UPwSmallStrainElement2D4N",
        element_block_name="UPwSmallStrainInterfaceElement2D4N",
    )
    neg_owner_ids, pos_owner_ids = _mdpa_interface_owner_ids(
        mdpa,
        mesh,
        body_block_name="UPwSmallStrainElement2D4N",
        interface_block_name="UPwSmallStrainInterfaceElement2D4N",
    )
    trace_link = paired_upl_interfaces_to_trace_link_interface_2d(
        paired_interfaces,
        mesh=mesh,
        negative_element_ids=neg_owner_ids,
        positive_element_ids=pos_owner_ids,
        quadrature="lobatto",
    )
    interface_system = build_upl_elastic_kratos_newton_interface_ufl_system_2d(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_current=u_current,
        p_current=p_current,
        velocity_current=velocity_current,
        p_rate_current=p_rate_current,
        material=interface_material,
        interface=trace_link,
        dt=dt_value,
        theta_u=theta_u,
        theta_p=theta_p,
        quadrature="lobatto",
        quadrature_order=2,
    )
    bulk_equation = Equation(bulk_system.lhs_form, bulk_system.rhs_form)
    interface_compiler = FormCompiler(dh, backend=backend)

    dirichlet_dofs = _kratos_process_dirichlet_dofs(
        dh,
        mesh,
        mdpa,
        params["processes"]["constraints_process_list"],
    )
    load_params = params["processes"]["loads_process_list"][0]["Parameters"]
    load_part = str(load_params["model_part_name"]).split(".")[-1]
    load_active = np.asarray(load_params.get("active", [False, True, False]), dtype=bool)
    load_component = int(np.flatnonzero(load_active)[0])
    load_value = float(load_params.get("value", [0.0, 0.0, 0.0])[load_component])
    dirichlet_unique = np.unique(np.asarray(dirichlet_dofs, dtype=int))
    free_dof_count = max(1, int(dh.total_dofs) - int(dirichlet_unique.size))

    if interface_update not in {"kratos_newton", "fixed_point", "kratos_lagged"}:
        raise ValueError("interface_update must be 'kratos_newton', 'fixed_point', or 'kratos_lagged'.")

    times: list[float] = []
    pressure_by_node: dict[int, list[float]] = {int(node_id): [] for node_id in output_node_ids}
    ux_by_node: dict[int, list[float]] = {int(node_id): [] for node_id in output_node_ids}
    uy_by_node: dict[int, list[float]] = {int(node_id): [] for node_id in output_node_ids}
    newton_history: list[dict[str, float]] | None = [] if collect_newton_history else None
    w_prev = np.zeros(dh.total_dofs, dtype=float)
    velocity_prev_values = np.zeros_like(velocity_prev.nodal_values, dtype=float)
    p_rate_prev_values = np.zeros_like(p_rate_prev.nodal_values, dtype=float)

    for step in range(1, n_steps + 1):
        time = dt_value * float(step)
        u_prev_values = w_prev[u_prev._g_dofs].copy()
        p_prev_values = w_prev[p_prev._g_dofs].copy()
        u_prev.nodal_values = u_prev_values
        p_prev.nodal_values = p_prev_values
        velocity_prev.nodal_values = velocity_prev_values
        p_rate_prev.nodal_values = p_rate_prev_values
        guess = w_prev.copy()
        max_iterations = 1 if interface_update == "kratos_lagged" else int(solver_settings.get("max_iteration", 15))
        if max_interface_iterations_override is not None:
            max_iterations = int(max_interface_iterations_override)
        initial_residual_norm: float | None = None

        def assign_current(solution: np.ndarray) -> None:
            u_current.nodal_values = solution[u_current._g_dofs]
            p_current.nodal_values = solution[p_current._g_dofs]
            velocity_current.nodal_values = (
                solution[velocity_prev._g_dofs]
                - u_prev_values
                - (1.0 - theta_u) * dt_value * velocity_prev_values
            ) / (theta_u * dt_value)
            p_rate_current.nodal_values = (
                solution[p_rate_prev._g_dofs]
                - p_prev_values
                - (1.0 - theta_p) * dt_value * p_rate_prev_values
            ) / (theta_p * dt_value)

        def interface_residual_at(solution: np.ndarray, K_bulk_csr, F_bulk_vec: np.ndarray) -> np.ndarray:
            assign_current(solution)
            residual = np.asarray(F_bulk_vec, dtype=float) - K_bulk_csr @ solution
            _add_kratos_line_load_component(residual, dh, mesh, mdpa, load_part, load_component, load_value)
            batch_current = interface_compiler.assemble_local_contributions(interface_system.equation)
            if batch_current.F_elem is None:
                raise RuntimeError("Vertical-fault interface Newton assembly must return a residual.")
            _scatter_local_residual_into_global(residual, batch_current.F_elem, batch_current.gdofs_map)
            residual[dirichlet_unique] = 0.0
            return residual

        for _iteration in range(max_iterations):
            assign_current(guess)

            K_bulk, F_bulk = assemble_form(bulk_equation, dof_handler=dh, bcs=[], backend=backend)
            K_bulk = K_bulk.tocsr()
            K = K_bulk.tolil()
            F = np.asarray(F_bulk, dtype=float) - K_bulk @ guess
            _add_kratos_line_load_component(F, dh, mesh, mdpa, load_part, load_component, load_value)
            batch = interface_compiler.assemble_local_contributions(interface_system.equation)
            if batch.K_elem is None or batch.F_elem is None:
                raise RuntimeError("Vertical-fault interface Newton assembly must return tangent and residual.")
            _scatter_local_batch_into_global(K, F, batch.K_elem, batch.F_elem, batch.gdofs_map)
            K_bc, F_bc = _apply_dirichlet_zero(K, F, dirichlet_dofs)
            residual_norm_before = float(np.linalg.norm(F_bc, ord=2))
            if initial_residual_norm is None:
                initial_residual_norm = max(residual_norm_before, np.finfo(float).tiny)
            delta = np.asarray(sp_la.spsolve(K_bc.tocsc(), F_bc), dtype=float)
            w_new = guess + delta
            post_residual = interface_residual_at(w_new, K_bulk, np.asarray(F_bulk, dtype=float))
            residual_norm = float(np.linalg.norm(post_residual, ord=2))
            residual_ratio = residual_norm / initial_residual_norm
            residual_absolute = residual_norm / float(free_dof_count)
            if newton_history is not None:
                newton_history.append(
                    {
                        "step": float(step),
                        "iteration": float(_iteration + 1),
                        "delta_norm_inf": float(np.linalg.norm(delta, ord=np.inf)),
                        "solution_norm_inf": float(np.linalg.norm(w_new, ord=np.inf)),
                        "residual_norm_before_2": residual_norm_before,
                        "residual_norm_2": residual_norm,
                        "residual_ratio": residual_ratio,
                        "residual_absolute": residual_absolute,
                    }
                )
            if interface_update == "kratos_lagged":
                break
            if residual_ratio <= float(solver_settings.get("residual_relative_tolerance", 1.0e-4)):
                break
            if residual_absolute <= float(solver_settings.get("residual_absolute_tolerance", 1.0e-9)):
                break
            if np.linalg.norm(delta, ord=np.inf) <= 1.0e-10 * max(1.0, np.linalg.norm(w_new, ord=np.inf)):
                break
            guess = w_new
        else:
            last = None if newton_history is None or len(newton_history) == 0 else newton_history[-1]
            if last is None:
                detail = ""
            else:
                detail = (
                    f" Last residual_norm_2={last['residual_norm_2']:.6e}, "
                    f"delta_norm_inf={last['delta_norm_inf']:.6e}."
                )
            raise RuntimeError(f"Vertical-fault interface Newton loop did not converge.{detail}")

        velocity_prev_values = (
            w_new[velocity_prev._g_dofs]
            - u_prev_values
            - (1.0 - theta_u) * dt_value * velocity_prev_values
        ) / (theta_u * dt_value)
        p_rate_prev_values = (
            w_new[p_rate_prev._g_dofs]
            - p_prev_values
            - (1.0 - theta_p) * dt_value * p_rate_prev_values
        ) / (theta_p * dt_value)
        w_prev = w_new
        times.append(time)
        pressure_now = field_values_at_mdpa_nodes(mesh, dh, w_prev, "p")
        ux_now = field_values_at_mdpa_nodes(mesh, dh, w_prev, "ux")
        uy_now = field_values_at_mdpa_nodes(mesh, dh, w_prev, "uy")
        for node_id in output_node_ids:
            pressure_by_node[int(node_id)].append(float(pressure_now[int(node_id)]))
            ux_by_node[int(node_id)].append(float(ux_now[int(node_id)]))
            uy_by_node[int(node_id)].append(float(uy_now[int(node_id)]))

    return KratosConsolidation2DResult(
        times=times,
        liquid_pressure_by_node=pressure_by_node,
        displacement_x_by_node=ux_by_node,
        displacement_y_by_node=uy_by_node,
        backend=backend,
        newton_history=newton_history,
    )


def solve_kratos_vertical_fault_consolidation_2d_reference(
    *,
    root: str | Path | None = None,
    end_time: float = 0.5,
    output_node_ids: tuple[int, ...] = VERTICAL_FAULT_CONSOLIDATION_SAMPLE_NODE_IDS,
) -> KratosConsolidation2DResult:
    """Run the full Kratos vertical-fault consolidation validation case."""

    case_root = (
        Path(root)
        if root is not None
        else KRATOS_EXAMPLES_PORO_ROOT / "validation" / "consolidation_interface_2D" / "source"
    )
    workdir = Path(tempfile.mkdtemp(prefix="kratos_vertical_fault_consolidation_2d_"))
    for name in ("ProjectParameters.json", "PoroMaterials.json", "consolidation_interface_2D.mdpa"):
        shutil.copy2(case_root / name, workdir / name)

    _patch_upl_validation_mdpa_for_installed_kratos(workdir / "consolidation_interface_2D.mdpa")
    _patch_upl_validation_materials_for_installed_kratos(workdir / "PoroMaterials.json")
    _patch_upl_validation_project_parameters_for_installed_kratos(
        workdir / "ProjectParameters.json",
        end_time=float(end_time),
    )

    previous_threads = os.environ.get("OMP_NUM_THREADS")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cwd = Path.cwd()
    try:
        import KratosMultiphysics as KM
        import KratosMultiphysics.PoromechanicsApplication as Poro
        from KratosMultiphysics.PoromechanicsApplication.poromechanics_analysis import PoromechanicsAnalysis

        class _SampleAnalysis(PoromechanicsAnalysis):
            def __init__(self, model, parameters):
                super().__init__(model, parameters)
                self.samples: list[dict[str, object]] = []

            def FinalizeSolutionStep(self):
                super().FinalizeSolutionStep()
                model_part = self.model["PorousModelPart"]
                row: dict[str, object] = {"time": float(model_part.ProcessInfo[KM.TIME])}
                for raw_node_id in output_node_ids:
                    node_id = int(raw_node_id)
                    node = model_part.GetNode(node_id)
                    row[str(node_id)] = (
                        float(node.GetSolutionStepValue(Poro.LIQUID_PRESSURE)),
                        float(node.GetSolutionStepValue(KM.DISPLACEMENT_X)),
                        float(node.GetSolutionStepValue(KM.DISPLACEMENT_Y)),
                    )
                self.samples.append(row)

        os.chdir(workdir)
        model = KM.Model()
        parameters = KM.Parameters((workdir / "ProjectParameters.json").read_text(encoding="utf-8"))
        analysis = _SampleAnalysis(model, parameters)
        analysis.Run()
    finally:
        os.chdir(cwd)
        if previous_threads is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = previous_threads

    times: list[float] = []
    pressure_by_node = {int(node_id): [] for node_id in output_node_ids}
    ux_by_node = {int(node_id): [] for node_id in output_node_ids}
    uy_by_node = {int(node_id): [] for node_id in output_node_ids}
    for sample in analysis.samples:
        times.append(float(sample["time"]))
        for node_id in output_node_ids:
            p_val, ux_val, uy_val = sample[str(int(node_id))]
            pressure_by_node[int(node_id)].append(float(p_val))
            ux_by_node[int(node_id)].append(float(ux_val))
            uy_by_node[int(node_id)].append(float(uy_val))

    return KratosConsolidation2DResult(
        times=times,
        liquid_pressure_by_node=pressure_by_node,
        displacement_x_by_node=ux_by_node,
        displacement_y_by_node=uy_by_node,
        backend="kratos",
    )


def solve_kratos_four_point_shear_2d_reference(
    *,
    output_node_ids: tuple[int, ...] = FOUR_POINT_SHEAR_SAMPLE_NODE_IDS,
    case_root: str | Path | None = None,
) -> KratosFourPointShear2DResult:
    """Read the Kratos four-point shear arc-length strategy reference output."""

    root = (
        Path(case_root)
        if case_root is not None
        else KRATOS_PORO_TESTS_ROOT / "strategy_tests" / "arc_length_test"
    )
    data = json.loads((root / "arc_length_test_results.json").read_text(encoding="utf-8"))
    times = [float(v) for v in data.get("TIME", [])]
    ux_by_node: dict[int, list[float]] = {}
    uy_by_node: dict[int, list[float]] = {}
    for node_id in output_node_ids:
        key = f"NODE_{int(node_id)}"
        node_data = data.get(key)
        if node_data is None:
            raise KeyError(f"Reference output does not contain {key}.")
        ux_by_node[int(node_id)] = [float(v) for v in node_data["DISPLACEMENT_X"]]
        uy_by_node[int(node_id)] = [float(v) for v in node_data["DISPLACEMENT_Y"]]
    return KratosFourPointShear2DResult(
        times=times,
        displacement_x_by_node=ux_by_node,
        displacement_y_by_node=uy_by_node,
        backend="kratos",
    )


def solve_kratos_four_point_shear_2d_runtime_reference(
    *,
    output_node_ids: tuple[int, ...] = FOUR_POINT_SHEAR_SAMPLE_NODE_IDS,
    case_root: str | Path | None = None,
    end_time: float = 2.0,
    linear_solver_type: str = "skyline_lu_factorization",
) -> KratosFourPointShear2DResult:
    """Run Kratos' four-point shear case and sample the live arc-length state.

    The JSON file shipped with the Kratos test was produced with an iterative
    BICGSTAB/ILU0 solve, so it is sensitive to solver tolerances and threading.
    This runtime reference switches Kratos to a direct solver and samples the
    same nodes and arc-length process-info values used by the pycutfem runner.
    That makes the gate a deterministic element/strategy parity check rather
    than a comparison against a solver-noise snapshot.
    """

    root = (
        Path(case_root)
        if case_root is not None
        else KRATOS_PORO_TESTS_ROOT / "strategy_tests" / "arc_length_test"
    )
    work_root = Path(tempfile.mkdtemp(prefix="kratos_four_point_shear_2d_"))
    workdir = work_root / "strategy_tests" / "arc_length_test"
    workdir.mkdir(parents=True, exist_ok=True)
    for name in ("ProjectParameters.json", "PoroMaterials.json", "arc_length_test.mdpa"):
        shutil.copy2(root / name, workdir / name)

    params_path = workdir / "ProjectParameters.json"
    params = load_kratos_json(params_path)
    params["problem_data"]["end_time"] = float(end_time)
    params["problem_data"]["echo_level"] = 0
    params["solver_settings"]["echo_level"] = 0
    params["solver_settings"]["linear_solver_settings"] = {"solver_type": str(linear_solver_type)}
    params.setdefault("processes", {})["auxiliar_process_list"] = []
    params_path.write_text(json.dumps(params, indent=4), encoding="utf-8")

    previous_threads = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "1"
    cwd = Path.cwd()
    try:
        import KratosMultiphysics as KM
        import KratosMultiphysics.PoromechanicsApplication as Poro
        from KratosMultiphysics.PoromechanicsApplication.poromechanics_analysis import PoromechanicsAnalysis

        class _SampleAnalysis(PoromechanicsAnalysis):
            def __init__(self, model, parameters):
                super().__init__(model, parameters)
                self.samples: list[dict[str, object]] = []

            def FinalizeSolutionStep(self):
                super().FinalizeSolutionStep()
                model_part = self.model["PorousModelPart"]
                row: dict[str, object] = {
                    "time": float(model_part.ProcessInfo[KM.TIME]),
                    "lambda": float(model_part.ProcessInfo[Poro.ARC_LENGTH_LAMBDA]),
                    "radius_factor": float(model_part.ProcessInfo[Poro.ARC_LENGTH_RADIUS_FACTOR]),
                    "iterations": int(model_part.ProcessInfo[KM.NL_ITERATION_NUMBER]),
                }
                for raw_node_id in output_node_ids:
                    node_id = int(raw_node_id)
                    node = model_part.GetNode(node_id)
                    row[str(node_id)] = (
                        float(node.GetSolutionStepValue(KM.DISPLACEMENT_X)),
                        float(node.GetSolutionStepValue(KM.DISPLACEMENT_Y)),
                    )
                self.samples.append(row)

        os.chdir(work_root)
        model = KM.Model()
        parameters = KM.Parameters(params_path.read_text(encoding="utf-8"))
        analysis = _SampleAnalysis(model, parameters)
        analysis.Run()
    finally:
        os.chdir(cwd)
        if previous_threads is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = previous_threads

    times: list[float] = []
    ux_by_node = {int(node_id): [] for node_id in output_node_ids}
    uy_by_node = {int(node_id): [] for node_id in output_node_ids}
    lambda_history: list[float] = []
    arc_history: list[dict[str, object]] = []
    for sample in analysis.samples:
        times.append(float(sample["time"]))
        lambda_history.append(float(sample["lambda"]))
        arc_history.append(
            {
                "time": float(sample["time"]),
                "lambda": float(sample["lambda"]),
                "radius_factor": float(sample["radius_factor"]),
                "iterations": int(sample["iterations"]),
            }
        )
        for node_id in output_node_ids:
            ux_val, uy_val = sample[str(int(node_id))]
            ux_by_node[int(node_id)].append(float(ux_val))
            uy_by_node[int(node_id)].append(float(uy_val))

    return KratosFourPointShear2DResult(
        times=times,
        displacement_x_by_node=ux_by_node,
        displacement_y_by_node=uy_by_node,
        backend="kratos-direct",
        lambda_history=lambda_history,
        arc_length_history=arc_history,
    )


def solve_kratos_four_point_shear_2d_pycutfem(
    *,
    backend: str = "cpp",
    end_time: float = 2.0,
    output_node_ids: tuple[int, ...] = FOUR_POINT_SHEAR_SAMPLE_NODE_IDS,
    case_root: str | Path | None = None,
    quadrature_order: int = 2,
) -> KratosFourPointShear2DResult:
    """Run the Kratos four-point shear arc-length case with pycutfem forms.

    This runner keeps the element creation generic through ``MixedElement``.
    The nonlocal damage history is stored at volume quadrature points. The
    stiffness form consumes the staged trial damage during Newton corrections
    and commits the history only after a converged arc-length step.
    """

    root = (
        Path(case_root)
        if case_root is not None
        else KRATOS_PORO_TESTS_ROOT / "strategy_tests" / "arc_length_test"
    )
    params = load_kratos_json(root / "ProjectParameters.json")
    materials = load_kratos_json(root / "PoroMaterials.json")
    mdpa = parse_kratos_mdpa(root / "arc_length_test.mdpa")

    mesh = mdpa_volume_mesh_to_pycutfem(
        mdpa,
        element_block_names=("UPlSmallStrainElement2D3N",),
        domain_tag="body",
    )
    mixed_element = MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "p": 1})
    dh = DofHandler(mixed_element, method="cg")

    material_by_part = _kratos_material_variables_by_part(materials)
    body_vars = material_by_part["Body_Part-auto-1"]
    body_material = _upl_material_from_kratos_variables(body_vars)
    damage_material = _modified_mises_damage_material_from_kratos_variables(body_vars)

    q_order = int(quadrature_order)
    nonlocal_map, layout = build_volume_nonlocal_quadrature_map(
        dh,
        quadrature_order=q_order,
        characteristic_length=float(params["solver_settings"]["characteristic_length"]),
    )
    damage_state = create_nonlocal_damage_state_2d(
        layout=layout,
        n_entities=int(mesh.n_elements),
        material=damage_material,
    )

    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)
    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)
    u_prev = VectorFunction(name="u_prev", field_names=["ux", "uy"], dof_handler=dh)
    p_prev = Function(name="p_prev", field_name="p", dof_handler=dh)
    u_prev.nodal_values.fill(0.0)
    p_prev.nodal_values.fill(0.0)

    solver_settings = params["solver_settings"]
    dt = float(solver_settings["time_step"])
    theta_u = float(solver_settings.get("newmark_theta_u", solver_settings.get("newmark_theta", 1.0)))
    theta_p = float(solver_settings.get("newmark_theta_p", solver_settings.get("newmark_theta", 1.0)))
    system = build_kratos_quasistatic_damage_upl_system_2d(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_prev=u_prev,
        p_prev=p_prev,
        material=body_material,
        damage=damage_state.damage_coefficient,
        dt=dt,
        theta_u=theta_u,
        theta_p=theta_p,
        dx_measure=dx(metadata={"q": q_order}),
    )

    constraint_processes = params.get("processes", {}).get("constraints_process_list", ())
    dirichlet_dofs = _kratos_process_dirichlet_dofs(dh, mesh, mdpa, constraint_processes)
    all_dofs = np.arange(int(dh.total_dofs), dtype=int)
    constrained = np.zeros(int(dh.total_dofs), dtype=bool)
    constrained[np.asarray(dirichlet_dofs, dtype=int)] = True
    free_dofs = all_dofs[~constrained]
    if free_dofs.size == 0:
        raise RuntimeError("Four-point shear runner has no free DOFs after applying constraints.")

    f_ref = np.zeros(int(dh.total_dofs), dtype=float)
    for process in params.get("processes", {}).get("loads_process_list", ()):
        process_params = process["Parameters"]
        part_name = str(process_params["model_part_name"]).split(".")[-1]
        active = np.asarray(process_params.get("active", [False, False, False]), dtype=bool)
        values = np.asarray(process_params.get("value", [0.0, 0.0, 0.0]), dtype=float)
        table_ids = np.asarray(process_params.get("table", [0, 0, 0]), dtype=int)
        for component in np.flatnonzero(active[:2]):
            load_value = float(values[int(component)])
            table_id = int(table_ids[int(component)]) if int(component) < table_ids.size else 0
            if table_id:
                load_value = float(mdpa.tables[table_id].value(0.0))
            _add_kratos_line_load_component(f_ref, dh, mesh, mdpa, part_name, int(component), load_value)
    f_ref[np.asarray(dirichlet_dofs, dtype=int)] = 0.0

    def _stage_damage(x_vec: np.ndarray) -> None:
        strain = volume_strain_voigt_2d(dh, x_vec, quadrature_order=q_order)
        stage_modified_mises_nonlocal_damage_2d(
            state=damage_state,
            material=damage_material,
            nonlocal_map=nonlocal_map,
            strain_voigt=strain,
        )

    def _assemble_tangent(x_vec: np.ndarray):
        _stage_damage(np.asarray(x_vec, dtype=float))
        K, _ = assemble_form(Equation(system.lhs_form, 0.0), dof_handler=dh, bcs=[], backend=backend)
        return K.tocsr()

    def _linear_solver(A, b):
        A_csr = A.tocsr() if sp.issparse(A) else sp.csr_matrix(np.asarray(A, dtype=float))
        rhs = np.asarray(b, dtype=float).reshape(-1)
        out = np.zeros_like(rhs)
        out[free_dofs] = sp_la.spsolve(A_csr[free_dofs][:, free_dofs].tocsc(), rhs[free_dofs])
        out[np.asarray(dirichlet_dofs, dtype=int)] = 0.0
        return out

    def _tangent_callback(x_vec: np.ndarray, lambda_value: float):
        del lambda_value
        return _assemble_tangent(x_vec)

    def _residual_callback(x_vec: np.ndarray, lambda_value: float):
        K = _assemble_tangent(x_vec)
        residual = float(lambda_value) * f_ref - K @ np.asarray(x_vec, dtype=float)
        residual[np.asarray(dirichlet_dofs, dtype=int)] = 0.0
        return residual

    x = np.zeros(int(dh.total_dofs), dtype=float)
    K0 = _assemble_tangent(x)
    arc_state = initialize_ramm_arc_length_state(K0, f_ref, linear_solver=_linear_solver)
    arc_params = RammArcLengthParameters(
        desired_iterations=int(solver_settings.get("desired_iterations", 4)),
        max_iterations=int(solver_settings.get("max_iteration", 25)),
        max_radius_factor=float(solver_settings.get("max_radius_factor", 10.0)),
        min_radius_factor=float(solver_settings.get("min_radius_factor", 1.0)),
        residual_tolerance=float(solver_settings.get("residual_absolute_tolerance", 1.0e-9)),
        residual_relative_tolerance=float(solver_settings.get("residual_relative_tolerance", 1.0e-4)),
        update_tolerance=float(solver_settings.get("displacement_absolute_tolerance", 0.0)),
        update_relative_tolerance=float(solver_settings.get("displacement_relative_tolerance", 0.0)),
    )

    times: list[float] = []
    ux_by_node: dict[int, list[float]] = {int(node_id): [] for node_id in output_node_ids}
    uy_by_node: dict[int, list[float]] = {int(node_id): [] for node_id in output_node_ids}
    lambda_history: list[float] = []
    arc_history: list[dict[str, object]] = []
    n_steps = int(round(float(end_time) / dt))
    for step in range(1, n_steps + 1):
        result = ramm_arc_length_step(
            x,
            arc_state,
            tangent_callback=_tangent_callback,
            residual_callback=_residual_callback,
            reference_load=f_ref,
            params=arc_params,
            linear_solver=_linear_solver,
        )
        if not result.converged:
            damage_state.rollback_step()
            raise RuntimeError(f"four-point shear arc-length step {step} did not converge.")
        x = np.asarray(result.x, dtype=float)
        arc_state = result.state
        damage_state.commit_step()

        times.append(float(step) * dt)
        lambda_history.append(float(arc_state.lambda_value))
        arc_history.append(
            {
                "step": int(step),
                "lambda": float(arc_state.lambda_value),
                "radius": float(arc_state.radius),
                "iterations": int(result.iterations),
                "residual_norms": [float(item.residual_norm) for item in result.history],
            }
        )
        ux_values = field_values_at_mdpa_nodes(mesh, dh, x, "ux")
        uy_values = field_values_at_mdpa_nodes(mesh, dh, x, "uy")
        for node_id in output_node_ids:
            ux_by_node[int(node_id)].append(float(ux_values[int(node_id)]))
            uy_by_node[int(node_id)].append(float(uy_values[int(node_id)]))

    return KratosFourPointShear2DResult(
        times=times,
        displacement_x_by_node=ux_by_node,
        displacement_y_by_node=uy_by_node,
        backend=backend,
        lambda_history=lambda_history,
        arc_length_history=arc_history,
    )


def solve_kratos_fluid_pumping_2d_pycutfem(
    *,
    backend: str = "cpp",
    root: str | Path | None = None,
    end_time: float | None = 1.0e-5,
    output_node_ids: tuple[int, ...] = FLUID_PUMPING_2D_SAMPLE_NODE_IDS,
    interface_update: str = "kratos_lagged",
    interface_state_update: str = "step",
    collect_newton_history: bool = False,
    linear_solver: str = "spsolve",
    dirichlet_mode: str = "zero_rows_columns",
    max_interface_iterations_override: int | None = None,
    interface_assembly: str = "ufl_trace_link",
) -> KratosFluidPumping2DResult:
    """Run the Kratos examples `fluid_pumping_2D` case with pycutfem forms.

    The imported case uses the old examples-repository U-Pw names. This solver
    keeps those names on the pycutfem side and maps only the physics:
    triangular FIC bulk, fracture interfaces, link interfaces, and the
    two-node interface normal-liquid-flux condition.

    ``interface_update="kratos_lagged"`` is the conservative default because it
    is an exact first-global-Newton-increment gate. The full Kratos nonlinear
    path is available as ``"kratos_newton"`` and remains the strict parity
    target once the remaining Poromechanics scheme-state sequencing is ported.

    ``interface_assembly="ufl_trace_link"`` uses the backend-dispatched
    ``dTraceLink`` Newton block for both fracture interfaces and Kratos link
    elements. ``"ufl_fracture_python_link"`` keeps the previous hybrid path
    where link elements use the paired reference kernel.
    ``"python_reference"`` keeps the old paired vectorized assembler for all
    interfaces as a parity oracle.

    ``dirichlet_mode="elimination"`` solves only the free-DOF reduced system.
    ``"kratos_elimination"`` additionally permutes free DOFs in Kratos'
    node-major ``ux, uy, p`` equation order. ``interface_state_update`` exposes
    the cohesive-law state sequencing needed to diagnose Kratos nonlinear
    parity.
    """

    case_root = (
        Path(root)
        if root is not None
        else KRATOS_EXAMPLES_PORO_ROOT / "use_cases" / "fluid_pumping_2D" / "source"
    )
    mdpa = parse_kratos_mdpa(case_root / "fluid_pumping_2D.mdpa")
    params = load_kratos_json(case_root / "ProjectParameters.json")
    materials = load_kratos_json(case_root / "PoroMaterials.json")

    mesh = mdpa_volume_mesh_to_pycutfem(
        mdpa,
        element_block_names=("UPwSmallStrainFICElement2D3N",),
    )
    mixed_element = MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "p": 1})
    dh = DofHandler(mixed_element, method="cg")
    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)

    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)
    u_prev = VectorFunction(name="u_prev", field_names=["ux", "uy"], dof_handler=dh)
    u_current = VectorFunction(name="u_current", field_names=["ux", "uy"], dof_handler=dh)
    p_prev = Function(name="p_prev", field_name="p", dof_handler=dh)
    p_current = Function(name="p_current", field_name="p", dof_handler=dh)
    velocity_prev = VectorFunction(name="velocity_prev", field_names=["ux", "uy"], dof_handler=dh)
    velocity_current = VectorFunction(name="velocity_current", field_names=["ux", "uy"], dof_handler=dh)
    p_rate_prev = Function(name="p_rate_prev", field_name="p", dof_handler=dh)
    p_rate_current = Function(name="p_rate_current", field_name="p", dof_handler=dh)
    for coeff in (u_prev, u_current, p_prev, p_current, velocity_prev, velocity_current, p_rate_prev, p_rate_current):
        coeff.nodal_values.fill(0.0)

    material_by_part = _kratos_material_variables_by_part(materials)
    body_vars = material_by_part["Body_Part-auto-1"]
    body_material = _upl_material_from_kratos_variables(body_vars)
    interface_material = _upl_interface_material_from_kratos_variables(material_by_part["Interface_Part-auto-1"])
    link_material = _upl_interface_material_from_kratos_variables(material_by_part["Interface_Part-auto-2"])

    solver_settings = params["solver_settings"]
    dt_value = float(solver_settings["time_step"])
    theta = float(solver_settings.get("newmark_theta", solver_settings.get("newmark_theta_u", 0.5)))
    theta_u = float(solver_settings.get("newmark_theta_u", theta))
    theta_p = float(solver_settings.get("newmark_theta_p", theta))
    if end_time is None:
        end_time_value = float(params["problem_data"]["end_time"])
    else:
        end_time_value = float(end_time)
    n_steps = int(round(end_time_value / dt_value))
    if n_steps < 1:
        raise ValueError("end_time must cover at least one time step.")

    bulk_system = build_kratos_fic_triangle_upl_system_2d(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_prev=u_prev,
        p_prev=p_prev,
        material=body_material,
        dt=dt_value,
        theta_u=theta_u,
        theta_p=theta_p,
        dx_measure=dx(metadata={"q": 2}),
        element_length_squared=kratos_fic_triangle_element_length_squared(mesh),
        velocity_prev=velocity_prev,
        p_rate_prev=p_rate_prev,
    )

    fracture_interfaces = _fluid_pumping_paired_interfaces_from_mdpa(
        mdpa,
        dh,
        mesh,
        element_block_name="UPwSmallStrainInterfaceElement2D4N",
    )
    link_interfaces = _fluid_pumping_paired_interfaces_from_mdpa(
        mdpa,
        dh,
        mesh,
        element_block_name="UPwSmallStrainLinkInterfaceElement2D4N",
    )
    fracture_owner_interface = _fluid_pumping_interface_from_mdpa(
        mdpa,
        mesh,
        element_block_name="UPwSmallStrainInterfaceElement2D4N",
    )
    fracture_trace_link = paired_upl_interfaces_to_trace_link_interface_2d(
        fracture_interfaces,
        mesh=mesh,
        negative_element_ids=np.asarray(fracture_owner_interface.neg_elem_ids, dtype=np.int32),
        positive_element_ids=np.asarray(fracture_owner_interface.pos_elem_ids, dtype=np.int32),
        quadrature="lobatto",
    )
    link_trace_link = paired_upl_interfaces_to_trace_link_interface_2d(
        link_interfaces,
        mesh=mesh,
        # The imported Kratos link elements connect endpoint/station pairs and
        # do not lie on body-element edges. The U-Pl link form below uses only
        # explicit station trace tables and qstate ids, so mark volume owners
        # as absent instead of fabricating body-edge ownership.
        negative_element_ids=-1,
        positive_element_ids=-1,
        quadrature="lobatto",
    )
    coupled_equation = Equation(
        bulk_system.lhs_form,
        bulk_system.rhs_form,
    )

    dirichlet_dofs = _kratos_process_dirichlet_dofs(
        dh,
        mesh,
        mdpa,
        params["processes"]["constraints_process_list"],
    )
    normal_flux_part, normal_flux_value = _fluid_pumping_normal_flux_process(params)
    structural_dof_pairs = _fluid_pumping_structural_dof_pairs(
        mdpa,
        dh,
        mesh,
        fracture_interfaces=fracture_interfaces,
        link_interfaces=link_interfaces,
    )

    times: list[float] = []
    pressure_by_node: dict[int, list[float]] = {int(node_id): [] for node_id in output_node_ids}
    ux_by_node: dict[int, list[float]] = {int(node_id): [] for node_id in output_node_ids}
    uy_by_node: dict[int, list[float]] = {int(node_id): [] for node_id in output_node_ids}
    newton_history: list[dict[str, object]] | None = [] if collect_newton_history else None
    w_prev = np.zeros(dh.total_dofs, dtype=float)
    velocity_prev_values = np.zeros_like(velocity_prev.nodal_values, dtype=float)
    p_rate_prev_values = np.zeros_like(p_rate_prev.nodal_values, dtype=float)
    fracture_state = _initial_interface_state(fracture_interfaces, interface_material)
    link_state = _initial_interface_state(link_interfaces, link_material)

    if interface_update not in {"kratos_newton", "fixed_point", "kratos_lagged"}:
        raise ValueError("interface_update must be 'kratos_newton', 'fixed_point', or 'kratos_lagged'.")
    if interface_state_update not in {"step", "iteration"}:
        raise ValueError("interface_state_update must be 'step' or 'iteration'.")
    if interface_assembly not in {"ufl_trace_link", "ufl_fracture_python_link", "python_reference"}:
        raise ValueError(
            "interface_assembly must be 'ufl_trace_link', 'ufl_fracture_python_link', or 'python_reference'."
        )
    if dirichlet_mode not in {"zero_rows_columns", "elimination", "kratos_elimination"}:
        raise ValueError(
            "dirichlet_mode must be 'zero_rows_columns', 'elimination', or 'kratos_elimination'."
        )
    kratos_free_dofs = (
        _kratos_node_major_free_dofs(dh, mesh, mdpa, dirichlet_dofs)
        if dirichlet_mode == "kratos_elimination"
        else None
    )

    interface_compiler = (
        FormCompiler(dh, backend=backend)
        if interface_assembly in {"ufl_trace_link", "ufl_fracture_python_link"}
        else None
    )
    fracture_ufl_state = None
    fracture_ufl_system = None
    link_ufl_state = None
    link_ufl_system = None
    if interface_assembly in {"ufl_trace_link", "ufl_fracture_python_link"}:
        layout = _lobatto_trace_link_layout(order=2)
        registry = StateRegistry()
        fracture_ufl_state = registry.register_quadrature(
            "fluid_pumping_fracture_damage",
            layout=layout,
            values=fracture_state.copy(),
            n_entities=fracture_trace_link.n_entities(),
        )
        fracture_ufl_system = build_upl_kratos_newton_interface_ufl_system_2d(
            u_trial=u,
            p_trial=p,
            u_test=v,
            p_test=q,
            u_current=u_current,
            p_current=p_current,
            velocity_current=velocity_current,
            p_rate_current=p_rate_current,
            state=fracture_ufl_state.coefficient(jit_name="fluid_pumping_fracture_damage"),
            material=interface_material,
            interface=fracture_trace_link,
            dt=dt_value,
            theta_u=theta_u,
            theta_p=theta_p,
            quadrature="lobatto",
            quadrature_order=2,
            permeability_law="fracture",
        )
        if interface_assembly == "ufl_trace_link":
            link_ufl_state = registry.register_quadrature(
                "fluid_pumping_link_damage",
                layout=layout,
                values=link_state.copy(),
                n_entities=link_trace_link.n_entities(),
            )
            link_ufl_system = build_upl_kratos_newton_interface_ufl_system_2d(
                u_trial=u,
                p_trial=p,
                u_test=v,
                p_test=q,
                u_current=u_current,
                p_current=p_current,
                velocity_current=velocity_current,
                p_rate_current=p_rate_current,
                state=link_ufl_state.coefficient(jit_name="fluid_pumping_link_damage"),
                material=link_material,
                interface=link_trace_link,
                dt=dt_value,
                theta_u=theta_u,
                theta_p=theta_p,
                quadrature="lobatto",
                quadrature_order=2,
                permeability_law="link",
            )

    for step in range(1, n_steps + 1):
        time = dt_value * float(step)
        u_prev_values = w_prev[u_prev._g_dofs].copy()
        p_prev_values = w_prev[p_prev._g_dofs].copy()
        u_prev.nodal_values = u_prev_values
        p_prev.nodal_values = p_prev_values
        velocity_prev.nodal_values = velocity_prev_values
        p_rate_prev.nodal_values = p_rate_prev_values
        guess = w_prev.copy()
        K_bulk, F_bulk = assemble_form(coupled_equation, dof_handler=dh, bcs=[], backend=backend)
        K_bulk = K_bulk.tocsr()
        F_bulk = np.asarray(F_bulk, dtype=float)

        max_interface_iterations = 1 if interface_update == "kratos_lagged" else int(
            solver_settings.get("max_iteration", 15)
        )
        if max_interface_iterations_override is not None:
            max_interface_iterations = int(max_interface_iterations_override)
        fracture_state_next = fracture_state
        link_state_next = link_state
        fracture_state_iter = fracture_state
        link_state_iter = link_state
        for _iteration in range(max_interface_iterations):
            u_current.nodal_values = guess[u_current._g_dofs]
            p_current.nodal_values = guess[p_current._g_dofs]
            K = K_bulk.tolil()
            F = np.asarray(F_bulk, dtype=float) - K_bulk @ guess
            velocity_guess = (
                guess[velocity_prev._g_dofs]
                - u_prev_values
                - (1.0 - theta_u) * dt_value * velocity_prev_values
            ) / (theta_u * dt_value)
            p_rate_guess = (
                guess[p_rate_prev._g_dofs]
                - p_prev_values
                - (1.0 - theta_p) * dt_value * p_rate_prev_values
            ) / (theta_p * dt_value)
            velocity_current.nodal_values = velocity_guess
            p_rate_current.nodal_values = p_rate_guess
            velocity_global = np.zeros(dh.total_dofs, dtype=float)
            dt_pressure_global = np.zeros(dh.total_dofs, dtype=float)
            velocity_global[velocity_prev._g_dofs] = velocity_guess
            dt_pressure_global[p_rate_prev._g_dofs] = p_rate_guess
            if interface_assembly in {"ufl_trace_link", "ufl_fracture_python_link"}:
                if (
                    interface_compiler is None
                    or fracture_ufl_system is None
                    or fracture_ufl_state is None
                ):
                    raise RuntimeError("UFL interface assembly was not initialized.")
                fracture_state_next = _assemble_ufl_interface_newton_batch_into_global(
                    K,
                    F,
                    compiler=interface_compiler,
                    system=fracture_ufl_system,
                    state_field=fracture_ufl_state,
                    state_variables=fracture_state_iter,
                )
                if interface_assembly == "ufl_trace_link":
                    if link_ufl_system is None or link_ufl_state is None:
                        raise RuntimeError("UFL link interface assembly was not initialized.")
                    link_state_next = _assemble_ufl_interface_newton_batch_into_global(
                        K,
                        F,
                        compiler=interface_compiler,
                        system=link_ufl_system,
                        state_field=link_ufl_state,
                        state_variables=link_state_iter,
                    )
                else:
                    link_state_next = _assemble_paired_interface_newton_batch_into_global(
                        K,
                        F,
                        link_interfaces,
                        link_material,
                        dt=dt_value,
                        theta_u=theta_u,
                        theta_p=theta_p,
                        previous_solution=w_prev,
                        current_solution=guess,
                        velocity_solution=velocity_global,
                        dt_pressure_solution=dt_pressure_global,
                        state_variables=link_state_iter,
                        permeability_law="link",
                    )
            else:
                fracture_state_next = _assemble_paired_interface_newton_batch_into_global(
                    K,
                    F,
                    fracture_interfaces,
                    interface_material,
                    dt=dt_value,
                    theta_u=theta_u,
                    theta_p=theta_p,
                    previous_solution=w_prev,
                    current_solution=guess,
                    velocity_solution=velocity_global,
                    dt_pressure_solution=dt_pressure_global,
                    state_variables=fracture_state_iter,
                    permeability_law="fracture",
                )
                link_state_next = _assemble_paired_interface_newton_batch_into_global(
                    K,
                    F,
                    link_interfaces,
                    link_material,
                    dt=dt_value,
                    theta_u=theta_u,
                    theta_p=theta_p,
                    previous_solution=w_prev,
                    current_solution=guess,
                    velocity_solution=velocity_global,
                    dt_pressure_solution=dt_pressure_global,
                    state_variables=link_state_iter,
                    permeability_law="link",
                )
            _add_kratos_interface_normal_liquid_flux(
                F,
                dh,
                mesh,
                mdpa,
                part_name=normal_flux_part,
                normal_flux=normal_flux_value,
                initial_joint_width=link_material.initial_joint_width,
            )
            delta, residual_for_history = _solve_dirichlet_increment(
                K,
                F,
                dirichlet_dofs,
                linear_solver=linear_solver,
                dirichlet_mode=dirichlet_mode,
                free_dof_order=kratos_free_dofs,
                structural_dof_pairs=structural_dof_pairs
                if dirichlet_mode == "kratos_elimination"
                else None,
            )
            w_new = guess + delta
            if newton_history is not None:
                pressure_iter = field_values_at_mdpa_nodes(mesh, dh, w_new, "p")
                ux_iter = field_values_at_mdpa_nodes(mesh, dh, w_new, "ux")
                uy_iter = field_values_at_mdpa_nodes(mesh, dh, w_new, "uy")
                newton_history.append(
                    {
                        "step": step,
                        "iteration": _iteration + 1,
                        "delta_norm_inf": float(np.linalg.norm(delta, ord=np.inf)),
                        "residual_norm": float(np.linalg.norm(residual_for_history)),
                        "samples": {
                            int(node_id): (
                                float(pressure_iter[int(node_id)]),
                                float(ux_iter[int(node_id)]),
                                float(uy_iter[int(node_id)]),
                            )
                            for node_id in output_node_ids
                        },
                        "fracture_state_min": float(np.min(fracture_state_next))
                        if fracture_state_next.size
                        else 0.0,
                        "fracture_state_max": float(np.max(fracture_state_next))
                        if fracture_state_next.size
                        else 0.0,
                        "link_state_min": float(np.min(link_state_next)) if link_state_next.size else 0.0,
                        "link_state_max": float(np.max(link_state_next)) if link_state_next.size else 0.0,
                    }
                )
            if interface_state_update == "iteration":
                fracture_state_iter = fracture_state_next
                link_state_iter = link_state_next
            if interface_update == "kratos_lagged":
                break
            if interface_update == "fixed_point" and np.linalg.norm(delta, ord=np.inf) <= 1.0e-10 * max(
                1.0, np.linalg.norm(w_new, ord=np.inf)
            ):
                break
            guess = w_new
        else:
            if interface_update == "fixed_point":
                raise RuntimeError("fluid_pumping_2D interface fixed-point loop did not converge.")

        fracture_state = fracture_state_next
        link_state = link_state_next
        velocity_prev_values = (
            w_new[velocity_prev._g_dofs]
            - u_prev_values
            - (1.0 - theta_u) * dt_value * velocity_prev_values
        ) / (theta_u * dt_value)
        p_rate_prev_values = (
            w_new[p_rate_prev._g_dofs]
            - p_prev_values
            - (1.0 - theta_p) * dt_value * p_rate_prev_values
        ) / (theta_p * dt_value)
        w_prev = w_new

        pressure_now = field_values_at_mdpa_nodes(mesh, dh, w_prev, "p")
        ux_now = field_values_at_mdpa_nodes(mesh, dh, w_prev, "ux")
        uy_now = field_values_at_mdpa_nodes(mesh, dh, w_prev, "uy")
        times.append(time)
        for node_id in output_node_ids:
            pressure_by_node[int(node_id)].append(float(pressure_now[int(node_id)]))
            ux_by_node[int(node_id)].append(float(ux_now[int(node_id)]))
            uy_by_node[int(node_id)].append(float(uy_now[int(node_id)]))

    return KratosFluidPumping2DResult(
        times=times,
        liquid_pressure_by_node=pressure_by_node,
        displacement_x_by_node=ux_by_node,
        displacement_y_by_node=uy_by_node,
        backend=backend,
        n_steps=n_steps,
        newton_history=newton_history,
    )


def solve_kratos_fluid_pumping_2d_fracture_pycutfem(
    *,
    backend: str = "cpp",
    root: str | Path | None = None,
    end_time: float | None = 0.1,
    output_node_ids: tuple[int, ...] = FLUID_PUMPING_2D_FRACTURE_SAMPLE_NODE_IDS,
    interface_update: str = "kratos_lagged",
    interface_state_update: str = "step",
    collect_newton_history: bool = False,
    linear_solver: str = "spsolve",
    dirichlet_mode: str = "zero_rows_columns",
    max_interface_iterations_override: int | None = None,
    interface_backend: str | None = None,
    enable_propagation: bool = True,
    max_propagation_events: int | None = 1,
    forced_propagation_steps: tuple[int, ...] = (),
) -> KratosFluidPumping2DResult:
    """Run `fluid_pumping_2D_fracture` with fixed-mesh Trace-FEM fractures.

    The imported Kratos case contains an initial 2D4 fracture network and a
    fracture-utility remeshing workflow.  This driver intentionally keeps the
    bulk mesh fixed: imported rows and inserted rows are represented as
    ``TraceFractureNetwork2D`` entities and assembled with ``dTraceLink``.
    New topology is explicit.  If a propagation event is requested, the driver
    selects side DOFs from the current mesh by the documented nearest-node
    extension rule and records the event in ``propagation_events``.

    ``backend`` controls the bulk U-Pl assembly. ``interface_backend`` controls
    the trace-link Newton block separately; by default the trace assembly uses
    the same backend as the bulk assembly.
    """

    case_root = (
        Path(root)
        if root is not None
        else KRATOS_EXAMPLES_PORO_ROOT / "use_cases" / "fluid_pumping_2D_fracture" / "source"
    )
    mdpa = parse_kratos_mdpa(case_root / "fluid_pumping_2D_fracture.mdpa")
    params = load_kratos_json(case_root / "ProjectParameters.json")
    materials = load_kratos_json(case_root / "PoroMaterials.json")
    fractures_data = load_kratos_json(case_root / "FracturesData.json")
    propagation_settings = TraceFracturePropagationSettings2D.from_kratos_fractures_data(fractures_data)

    mesh = mdpa_volume_mesh_to_pycutfem(
        mdpa,
        element_block_names=("UPwSmallStrainFICElement2D3N",),
    )
    mixed_element = MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "p": 1})
    dh = DofHandler(mixed_element, method="cg")
    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)

    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)
    u_prev = VectorFunction(name="u_prev", field_names=["ux", "uy"], dof_handler=dh)
    u_current = VectorFunction(name="u_current", field_names=["ux", "uy"], dof_handler=dh)
    p_prev = Function(name="p_prev", field_name="p", dof_handler=dh)
    p_current = Function(name="p_current", field_name="p", dof_handler=dh)
    velocity_prev = VectorFunction(name="velocity_prev", field_names=["ux", "uy"], dof_handler=dh)
    velocity_current = VectorFunction(name="velocity_current", field_names=["ux", "uy"], dof_handler=dh)
    p_rate_prev = Function(name="p_rate_prev", field_name="p", dof_handler=dh)
    p_rate_current = Function(name="p_rate_current", field_name="p", dof_handler=dh)
    for coeff in (u_prev, u_current, p_prev, p_current, velocity_prev, velocity_current, p_rate_prev, p_rate_current):
        coeff.nodal_values.fill(0.0)

    material_by_part = _kratos_material_variables_by_part(materials)
    body_vars = _with_default_biot_coefficient(material_by_part["Body_Part-auto-1"])
    interface_vars = _with_default_biot_coefficient(material_by_part["Interface_Part-auto-1"])
    body_material = _upl_material_from_kratos_variables(body_vars)
    body_damage_material = _modified_mises_damage_material_from_kratos_variables(body_vars)
    interface_material = _upl_interface_material_from_kratos_variables(interface_vars)

    solver_settings = params["solver_settings"]
    dt_value = float(solver_settings["time_step"])
    theta = float(solver_settings.get("newmark_theta", solver_settings.get("newmark_theta_u", 0.5)))
    theta_u = float(solver_settings.get("newmark_theta_u", theta))
    theta_p = float(solver_settings.get("newmark_theta_p", theta))
    end_time_value = float(params["problem_data"]["end_time"] if end_time is None else end_time)
    n_steps = int(round(end_time_value / dt_value))
    if n_steps < 1:
        raise ValueError("end_time must cover at least one time step.")

    bulk_q_order = 2
    nonlocal_map, bulk_damage_layout = build_volume_nonlocal_quadrature_map(
        dh,
        quadrature_order=bulk_q_order,
        characteristic_length=float(solver_settings.get("characteristic_length", 0.1)),
    )
    bulk_damage_state = create_nonlocal_damage_state_2d(
        layout=bulk_damage_layout,
        n_entities=int(mesh.n_elements),
        material=body_damage_material,
    )

    def _stage_bulk_damage(x_vec: np.ndarray) -> None:
        strain = volume_strain_voigt_2d(dh, x_vec, quadrature_order=bulk_q_order)
        stage_modified_mises_nonlocal_damage_2d(
            state=bulk_damage_state,
            material=body_damage_material,
            nonlocal_map=nonlocal_map,
            strain_voigt=strain,
        )

    bulk_system = build_kratos_fic_triangle_damage_upl_system_2d(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_prev=u_prev,
        p_prev=p_prev,
        material=body_material,
        damage=bulk_damage_state.damage_coefficient,
        dt=dt_value,
        theta_u=theta_u,
        theta_p=theta_p,
        dx_measure=dx(metadata={"q": bulk_q_order}),
        element_length_squared=kratos_fic_triangle_element_length_squared(mesh),
        velocity_prev=velocity_prev,
        p_rate_prev=p_rate_prev,
    )
    coupled_equation = Equation(bulk_system.lhs_form, bulk_system.rhs_form)

    network = _fluid_pumping_trace_fracture_network_from_mdpa(
        mdpa,
        dh,
        mesh,
        element_block_name="UPwSmallStrainInterfaceElement2D4N",
    )
    fracture_state = np.full(
        (network.n_entities, 2),
        float(interface_material.damage_threshold if interface_material.damage_threshold is not None else 0.0),
        dtype=float,
    )
    layout = _lobatto_trace_link_layout(order=2)
    registry = StateRegistry()
    trace_backend = str(interface_backend if interface_backend is not None else backend)
    compiler = FormCompiler(dh, backend=trace_backend)
    topology_version = 0

    def build_fracture_system(current_network: TraceFractureNetwork2D, state_values: np.ndarray):
        trace_link = current_network.to_trace_link_interface(quadrature="lobatto")
        state_field = registry.register_quadrature(
            f"fluid_pumping_fracture_trace_damage_v{topology_version}",
            layout=layout,
            values=np.asarray(state_values, dtype=float).copy(),
            n_entities=trace_link.n_entities(),
        )
        system = build_upl_kratos_newton_interface_ufl_system_2d(
            u_trial=u,
            p_trial=p,
            u_test=v,
            p_test=q,
            u_current=u_current,
            p_current=p_current,
            velocity_current=velocity_current,
            p_rate_current=p_rate_current,
            state=state_field.coefficient(jit_name=f"fluid_pumping_fracture_trace_damage_v{topology_version}"),
            material=interface_material,
            interface=trace_link,
            dt=dt_value,
            theta_u=theta_u,
            theta_p=theta_p,
            quadrature="lobatto",
            quadrature_order=2,
            permeability_law="fracture",
        )
        return trace_link, state_field, system

    fracture_trace_link, fracture_ufl_state, fracture_ufl_system = build_fracture_system(network, fracture_state)

    dirichlet_dofs = _kratos_process_dirichlet_dofs(
        dh,
        mesh,
        mdpa,
        params["processes"]["constraints_process_list"],
    )
    flux_processes = _fluid_pumping_normal_flux_processes(params)
    structural_dof_pairs = _fluid_pumping_structural_dof_pairs(
        mdpa,
        dh,
        mesh,
        fracture_interfaces=_trace_network_as_paired_interfaces(network),
        link_interfaces=(),
    )
    kratos_free_dofs = (
        _kratos_node_major_free_dofs(dh, mesh, mdpa, dirichlet_dofs)
        if dirichlet_mode == "kratos_elimination"
        else None
    )

    if interface_update not in {"kratos_newton", "fixed_point", "kratos_lagged"}:
        raise ValueError("interface_update must be 'kratos_newton', 'fixed_point', or 'kratos_lagged'.")
    if interface_state_update not in {"step", "iteration"}:
        raise ValueError("interface_state_update must be 'step' or 'iteration'.")
    if dirichlet_mode not in {"zero_rows_columns", "elimination", "kratos_elimination"}:
        raise ValueError(
            "dirichlet_mode must be 'zero_rows_columns', 'elimination', or 'kratos_elimination'."
        )

    times: list[float] = []
    pressure_by_node: dict[int, list[float]] = {int(node_id): [] for node_id in output_node_ids}
    ux_by_node: dict[int, list[float]] = {int(node_id): [] for node_id in output_node_ids}
    uy_by_node: dict[int, list[float]] = {int(node_id): [] for node_id in output_node_ids}
    newton_history: list[dict[str, object]] | None = [] if collect_newton_history else None
    propagation_events: list[dict[str, object]] = []
    interface_count_history: list[int] = []
    bulk_damage_history: list[float] = []
    tip_damage_history: list[float] = []
    bulk_damage_location_history: list[tuple[float, float]] = []
    tip_damage_support_history: list[int] = []
    tip_damage_centroid_history: list[tuple[float, float]] = []
    forced_steps = {int(step) for step in forced_propagation_steps}
    next_entity_id = max(network.entity_ids, default=0) + 1

    w_prev = np.zeros(dh.total_dofs, dtype=float)
    velocity_prev_values = np.zeros_like(velocity_prev.nodal_values, dtype=float)
    p_rate_prev_values = np.zeros_like(p_rate_prev.nodal_values, dtype=float)

    for step in range(1, n_steps + 1):
        time = dt_value * float(step)
        u_prev_values = w_prev[u_prev._g_dofs].copy()
        p_prev_values = w_prev[p_prev._g_dofs].copy()
        u_prev.nodal_values = u_prev_values
        p_prev.nodal_values = p_prev_values
        velocity_prev.nodal_values = velocity_prev_values
        p_rate_prev.nodal_values = p_rate_prev_values
        guess = w_prev.copy()

        max_interface_iterations = 1 if interface_update == "kratos_lagged" else int(
            solver_settings.get("max_iteration", 15)
        )
        if max_interface_iterations_override is not None:
            max_interface_iterations = int(max_interface_iterations_override)
        fracture_state_next = fracture_state
        fracture_state_iter = fracture_state
        for iteration in range(max_interface_iterations):
            u_current.nodal_values = guess[u_current._g_dofs]
            p_current.nodal_values = guess[p_current._g_dofs]
            _stage_bulk_damage(guess)
            K_bulk, F_bulk = assemble_form(coupled_equation, dof_handler=dh, bcs=[], backend=backend)
            K_bulk = K_bulk.tocsr()
            F_bulk = np.asarray(F_bulk, dtype=float)
            K = K_bulk.tolil()
            F = np.asarray(F_bulk, dtype=float) - K_bulk @ guess
            velocity_guess = (
                guess[velocity_prev._g_dofs]
                - u_prev_values
                - (1.0 - theta_u) * dt_value * velocity_prev_values
            ) / (theta_u * dt_value)
            p_rate_guess = (
                guess[p_rate_prev._g_dofs]
                - p_prev_values
                - (1.0 - theta_p) * dt_value * p_rate_prev_values
            ) / (theta_p * dt_value)
            velocity_current.nodal_values = velocity_guess
            p_rate_current.nodal_values = p_rate_guess

            fracture_state_next = _assemble_ufl_interface_newton_batch_into_global(
                K,
                F,
                compiler=compiler,
                system=fracture_ufl_system,
                state_field=fracture_ufl_state,
                state_variables=fracture_state_iter,
            )
            _add_kratos_normal_liquid_flux_processes(
                F,
                dh,
                mesh,
                mdpa,
                flux_processes=flux_processes,
                interface_aperture=interface_material.initial_joint_width,
            )
            delta, residual_for_history = _solve_dirichlet_increment(
                K,
                F,
                dirichlet_dofs,
                linear_solver=linear_solver,
                dirichlet_mode=dirichlet_mode,
                free_dof_order=kratos_free_dofs,
                structural_dof_pairs=structural_dof_pairs
                if dirichlet_mode == "kratos_elimination"
                else None,
            )
            w_new = guess + delta
            if newton_history is not None:
                newton_history.append(
                    {
                        "step": step,
                        "iteration": iteration + 1,
                        "delta_norm_inf": float(np.linalg.norm(delta, ord=np.inf)),
                        "residual_norm": float(np.linalg.norm(residual_for_history)),
                        "fracture_state_min": float(np.min(fracture_state_next)) if fracture_state_next.size else 0.0,
                        "fracture_state_max": float(np.max(fracture_state_next)) if fracture_state_next.size else 0.0,
                        "bulk_damage_min": float(np.min(bulk_damage_state.damage.staged_values)),
                        "bulk_damage_max": float(np.max(bulk_damage_state.damage.staged_values)),
                        "n_trace_entities": int(network.n_entities),
                    }
                )
            if interface_state_update == "iteration":
                fracture_state_iter = fracture_state_next
            if interface_update == "kratos_lagged":
                break
            if interface_update == "fixed_point" and np.linalg.norm(delta, ord=np.inf) <= 1.0e-10 * max(
                1.0, np.linalg.norm(w_new, ord=np.inf)
            ):
                break
            guess = w_new
        else:
            if interface_update == "fixed_point":
                raise RuntimeError("fluid_pumping_2D_fracture interface fixed-point loop did not converge.")

        fracture_state = fracture_state_next
        _stage_bulk_damage(w_new)
        bulk_damage_state.commit_step()
        velocity_prev_values = (
            w_new[velocity_prev._g_dofs]
            - u_prev_values
            - (1.0 - theta_u) * dt_value * velocity_prev_values
        ) / (theta_u * dt_value)
        p_rate_prev_values = (
            w_new[p_rate_prev._g_dofs]
            - p_prev_values
            - (1.0 - theta_p) * dt_value * p_rate_prev_values
        ) / (theta_p * dt_value)
        w_prev = w_new
        tip_damage_probe = _trace_tip_bulk_damage_probe(
            network,
            bulk_damage_state.damage.values,
            nonlocal_map.points,
            nonlocal_map.weights,
            search_radius=float(propagation_settings.propagation_length),
        )
        tip_damage = float(tip_damage_probe["damage"])
        damage_location = _bulk_damage_max_location(
            bulk_damage_state.damage.values,
            nonlocal_map.points,
        )
        bulk_damage_history.append(float(np.max(bulk_damage_state.damage.values)))
        tip_damage_history.append(float(tip_damage))
        bulk_damage_location_history.append(damage_location)
        tip_damage_support_history.append(int(tip_damage_probe["support_count"]))
        tip_damage_centroid_history.append(tuple(float(v) for v in tip_damage_probe["centroid"]))

        pressure_now = field_values_at_mdpa_nodes(mesh, dh, w_prev, "p")
        ux_now = field_values_at_mdpa_nodes(mesh, dh, w_prev, "ux")
        uy_now = field_values_at_mdpa_nodes(mesh, dh, w_prev, "uy")
        times.append(time)
        for node_id in output_node_ids:
            pressure_by_node[int(node_id)].append(float(pressure_now[int(node_id)]))
            ux_by_node[int(node_id)].append(float(ux_now[int(node_id)]))
            uy_by_node[int(node_id)].append(float(uy_now[int(node_id)]))

        if enable_propagation and _should_insert_trace_fracture(
            network,
            fracture_state,
            propagation_settings,
            step=step,
            forced_steps=forced_steps,
            event_count=len(propagation_events),
            max_events=max_propagation_events,
            bulk_tip_damage=tip_damage,
        ):
            old_network = network
            parent_id = _rightmost_trace_entity_id(network)
            parent_state = fracture_state.copy()
            extension = _make_trace_fracture_tip_extension_from_existing_nodes(
                network,
                mdpa,
                dh,
                mesh,
                parent_entity_id=parent_id,
                new_entity_id=next_entity_id,
                propagation_settings=propagation_settings,
            )
            next_entity_id += 1
            network = network.insert(extension)
            fracture_state = transfer_trace_quadrature_state(
                old_network,
                network,
                parent_state,
                default=float(interface_material.damage_threshold or 0.0),
            )
            topology_version += 1
            fracture_trace_link, fracture_ufl_state, fracture_ufl_system = build_fracture_system(network, fracture_state)
            structural_dof_pairs = _fluid_pumping_structural_dof_pairs(
                mdpa,
                dh,
                mesh,
                fracture_interfaces=_trace_network_as_paired_interfaces(network),
                link_interfaces=(),
            )
            propagation_events.append(
                {
                    "step": int(step),
                    "time": float(time),
                    "parent_entity_id": int(parent_id),
                    "new_entity_id": int(extension.entity_id),
                    "state_source_id": int(extension.state_source_id if extension.state_source_id is not None else -1),
                    "n_trace_entities": int(network.n_entities),
                    "bulk_tip_damage": float(tip_damage),
                    "bulk_damage_max": float(np.max(bulk_damage_state.damage.values)),
                    "tip_damage_support_count": int(tip_damage_probe["support_count"]),
                    "tip_damage_centroid": tuple(float(v) for v in tip_damage_probe["centroid"]),
                    "start": tuple(float(v) for v in extension.midpoint_coords()[0]),
                    "end": tuple(float(v) for v in extension.midpoint_coords()[-1]),
                }
            )
        interface_count_history.append(int(network.n_entities))

    return KratosFluidPumping2DResult(
        times=times,
        liquid_pressure_by_node=pressure_by_node,
        displacement_x_by_node=ux_by_node,
        displacement_y_by_node=uy_by_node,
        backend=backend,
        n_steps=n_steps,
        newton_history=newton_history,
        propagation_events=propagation_events,
        interface_count_history=interface_count_history,
        bulk_damage_history=bulk_damage_history,
        tip_damage_history=tip_damage_history,
        bulk_damage_location_history=bulk_damage_location_history,
        tip_damage_support_history=tip_damage_support_history,
        tip_damage_centroid_history=tip_damage_centroid_history,
    )


def solve_kratos_fluid_pumping_2d_reference(
    *,
    root: str | Path | None = None,
    end_time: float = 1.0e-5,
    output_node_ids: tuple[int, ...] = FLUID_PUMPING_2D_SAMPLE_NODE_IDS,
) -> KratosFluidPumping2DResult:
    """Run the Kratos `fluid_pumping_2D` example through the installed U-Pl API."""

    case_root = (
        Path(root)
        if root is not None
        else KRATOS_EXAMPLES_PORO_ROOT / "use_cases" / "fluid_pumping_2D" / "source"
    )
    workdir = Path(tempfile.mkdtemp(prefix="kratos_fluid_pumping_2d_"))
    for name in ("ProjectParameters.json", "PoroMaterials.json", "fluid_pumping_2D.mdpa"):
        shutil.copy2(case_root / name, workdir / name)

    _patch_fluid_pumping_mdpa_for_installed_kratos(workdir / "fluid_pumping_2D.mdpa")
    _patch_fluid_pumping_materials_for_installed_kratos(workdir / "PoroMaterials.json")
    _patch_fluid_pumping_project_parameters_for_installed_kratos(
        workdir / "ProjectParameters.json",
        end_time=float(end_time),
    )

    previous_threads = os.environ.get("OMP_NUM_THREADS")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cwd = Path.cwd()
    try:
        import KratosMultiphysics as KM
        import KratosMultiphysics.PoromechanicsApplication as Poro
        from KratosMultiphysics.PoromechanicsApplication.poromechanics_analysis import PoromechanicsAnalysis

        class _SampleAnalysis(PoromechanicsAnalysis):
            def __init__(self, model, parameters):
                super().__init__(model, parameters)
                self.samples: list[dict[str, object]] = []

            def FinalizeSolutionStep(self):
                super().FinalizeSolutionStep()
                model_part = self.model["PorousModelPart"]
                row: dict[str, object] = {"time": float(model_part.ProcessInfo[KM.TIME])}
                for raw_node_id in output_node_ids:
                    node_id = int(raw_node_id)
                    node = model_part.GetNode(node_id)
                    row[str(node_id)] = (
                        float(node.GetSolutionStepValue(Poro.LIQUID_PRESSURE)),
                        float(node.GetSolutionStepValue(KM.DISPLACEMENT_X)),
                        float(node.GetSolutionStepValue(KM.DISPLACEMENT_Y)),
                    )
                self.samples.append(row)

        os.chdir(workdir)
        model = KM.Model()
        parameters = KM.Parameters((workdir / "ProjectParameters.json").read_text(encoding="utf-8"))
        analysis = _SampleAnalysis(model, parameters)
        analysis.Run()
    finally:
        os.chdir(cwd)
        if previous_threads is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = previous_threads

    times: list[float] = []
    pressure_by_node = {int(node_id): [] for node_id in output_node_ids}
    ux_by_node = {int(node_id): [] for node_id in output_node_ids}
    uy_by_node = {int(node_id): [] for node_id in output_node_ids}
    for sample in analysis.samples:
        times.append(float(sample["time"]))
        for node_id in output_node_ids:
            p_val, ux_val, uy_val = sample[str(int(node_id))]
            pressure_by_node[int(node_id)].append(float(p_val))
            ux_by_node[int(node_id)].append(float(ux_val))
            uy_by_node[int(node_id)].append(float(uy_val))

    dt_value = load_kratos_json(workdir / "ProjectParameters.json")["solver_settings"]["time_step"]
    return KratosFluidPumping2DResult(
        times=times,
        liquid_pressure_by_node=pressure_by_node,
        displacement_x_by_node=ux_by_node,
        displacement_y_by_node=uy_by_node,
        backend="kratos",
        n_steps=int(round(float(end_time) / float(dt_value))),
    )


def _kratos_material_variables_by_part(materials: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for prop in materials.get("properties", ()):
        full_name = str(prop.get("model_part_name", ""))
        part_name = full_name.split(".")[-1]
        variables = dict(prop.get("Material", {}).get("Variables", {}))
        if part_name:
            out[part_name] = variables
    return out


def _mat_var(variables: dict, *names: str, default=None) -> float:
    for name in names:
        if name in variables:
            return float(variables[name])
    if default is not None:
        return float(default)
    raise KeyError(f"None of the material variables {names!r} are present.")


def _upl_material_from_kratos_variables(variables: dict) -> UPlMaterial2D:
    return UPlMaterial2D(
        young_modulus=_mat_var(variables, "YOUNG_MODULUS"),
        poisson_ratio=_mat_var(variables, "POISSON_RATIO"),
        porosity=_mat_var(variables, "POROSITY"),
        biot_coefficient=_mat_var(variables, "BIOT_COEFFICIENT"),
        bulk_modulus_solid=_mat_var(variables, "BULK_MODULUS_SOLID"),
        bulk_modulus_liquid=_mat_var(variables, "BULK_MODULUS_LIQUID", "BULK_MODULUS_FLUID"),
        permeability_xx=_mat_var(variables, "PERMEABILITY_XX"),
        permeability_yy=_mat_var(variables, "PERMEABILITY_YY", default=_mat_var(variables, "PERMEABILITY_XX")),
        permeability_xy=_mat_var(variables, "PERMEABILITY_XY", default=0.0),
        dynamic_viscosity_liquid=_mat_var(
            variables,
            "DYNAMIC_VISCOSITY_LIQUID",
            "DYNAMIC_VISCOSITY",
        ),
        density_solid=_mat_var(variables, "DENSITY_SOLID", default=0.0),
        density_liquid=_mat_var(variables, "DENSITY_LIQUID", "DENSITY_WATER", default=0.0),
        thickness=_mat_var(variables, "THICKNESS", default=1.0),
    )


def _modified_mises_damage_material_from_kratos_variables(
    variables: dict,
) -> ModifiedMisesNonlocalDamagePlaneStress2D:
    return ModifiedMisesNonlocalDamagePlaneStress2D(
        young_modulus=_mat_var(variables, "YOUNG_MODULUS"),
        poisson_ratio=_mat_var(variables, "POISSON_RATIO"),
        damage_threshold=_mat_var(variables, "DAMAGE_THRESHOLD"),
        strength_ratio=_mat_var(variables, "STRENGTH_RATIO"),
        residual_strength=_mat_var(variables, "RESIDUAL_STRENGTH"),
        softening_slope=_mat_var(variables, "SOFTENING_SLOPE"),
    )


def _upl_interface_material_from_kratos_variables(variables: dict) -> UPlInterfaceMaterial2D:
    if "NORMAL_STIFFNESS" in variables:
        normal_stiffness = _mat_var(variables, "NORMAL_STIFFNESS")
    else:
        normal_stiffness = _bilinear_cohesive_initial_stiffness(variables)
    if "SHEAR_STIFFNESS" in variables:
        shear_stiffness = _mat_var(variables, "SHEAR_STIFFNESS")
    else:
        shear_stiffness = _bilinear_cohesive_initial_stiffness(variables)
    return UPlInterfaceMaterial2D(
        normal_stiffness=normal_stiffness,
        shear_stiffness=shear_stiffness,
        porosity=_mat_var(variables, "POROSITY"),
        biot_coefficient=_mat_var(variables, "BIOT_COEFFICIENT"),
        bulk_modulus_solid=_mat_var(variables, "BULK_MODULUS_SOLID"),
        bulk_modulus_liquid=_mat_var(variables, "BULK_MODULUS_LIQUID", "BULK_MODULUS_FLUID"),
        dynamic_viscosity_liquid=_mat_var(
            variables,
            "DYNAMIC_VISCOSITY_LIQUID",
            "DYNAMIC_VISCOSITY",
        ),
        initial_joint_width=_mat_var(variables, "INITIAL_JOINT_WIDTH"),
        transversal_permeability_coefficient=_mat_var(
            variables,
            "TRANSVERSAL_PERMEABILITY_COEFFICIENT",
            default=0.0,
        ),
        penalty_stiffness=_mat_var(variables, "PENALTY_STIFFNESS", default=1.0),
        density_solid=_mat_var(variables, "DENSITY_SOLID", default=0.0),
        density_liquid=_mat_var(variables, "DENSITY_LIQUID", "DENSITY_WATER", default=0.0),
        thickness=_mat_var(variables, "THICKNESS", default=1.0),
        young_modulus=_mat_var(variables, "YOUNG_MODULUS", default=0.0),
        critical_displacement=_mat_var(variables, "CRITICAL_DISPLACEMENT", default=0.0),
        yield_stress=_mat_var(variables, "YIELD_STRESS", default=0.0),
        damage_threshold=_mat_var(variables, "DAMAGE_THRESHOLD", default=0.0),
        friction_coefficient=_mat_var(variables, "FRICTION_COEFFICIENT", default=0.0),
    )


def _bilinear_cohesive_initial_stiffness(variables: dict) -> float:
    yield_stress = _mat_var(variables, "YIELD_STRESS")
    critical_displacement = _mat_var(variables, "CRITICAL_DISPLACEMENT")
    damage_threshold = _mat_var(variables, "DAMAGE_THRESHOLD")
    return yield_stress / (critical_displacement * damage_threshold)


def _softened_bilinear_interface_material(
    material: UPlInterfaceMaterial2D,
    variables: dict,
) -> UPlInterfaceMaterial2D:
    """Use a first nonlinear secant stiffness for the pumping gate.

    The fluid-pumping example activates `BilinearCohesive2DLaw` immediately at
    the inlet. Until the full state-variable constitutive update is available
    in UFL/C++, use the cohesive secant at a representative post-threshold
    state so the imported example is not locked by the initial elastic tangent.
    """

    if not {"YIELD_STRESS", "CRITICAL_DISPLACEMENT", "DAMAGE_THRESHOLD"}.issubset(variables):
        return material
    state = max(_mat_var(variables, "DAMAGE_THRESHOLD"), 0.2)
    state = min(state, 0.999)
    stiffness = _mat_var(variables, "YIELD_STRESS") / (_mat_var(variables, "CRITICAL_DISPLACEMENT") * state)
    stiffness *= (1.0 - state) / (1.0 - _mat_var(variables, "DAMAGE_THRESHOLD"))
    return replace(material, normal_stiffness=stiffness, shear_stiffness=stiffness)


def _with_default_biot_coefficient(variables: dict, *, default: float = 1.0) -> dict:
    out = dict(variables)
    out.setdefault("BIOT_COEFFICIENT", float(default))
    return out


def _dof_map_from_station_rows(rows: tuple[tuple[int, int, int], ...]) -> dict[str, tuple[int, ...]]:
    arr = np.asarray(rows, dtype=np.int64)
    if arr.ndim != 2 or int(arr.shape[1]) != 3:
        raise ValueError(f"Station rows must have shape (n, 3), got {arr.shape}.")
    return {
        "ux": tuple(int(v) for v in arr[:, 0]),
        "uy": tuple(int(v) for v in arr[:, 1]),
        "p": tuple(int(v) for v in arr[:, 2]),
    }


def _paired_interface_to_trace_station_entity(
    entity_id: int,
    paired: PairedUPlInterface2D,
    *,
    owner_neg_id: int,
    owner_pos_id: int,
    state_source_id: int | None = None,
) -> TraceStationEntity2D:
    return TraceStationEntity2D(
        entity_id=int(entity_id),
        negative_coords=paired.negative_coords,
        positive_coords=paired.positive_coords,
        negative_dofs=_dof_map_from_station_rows(paired.negative_dofs),
        positive_dofs=_dof_map_from_station_rows(paired.positive_dofs),
        owner_neg_id=int(owner_neg_id),
        owner_pos_id=int(owner_pos_id),
        state_source_id=state_source_id,
    )


def _fluid_pumping_trace_fracture_network_from_mdpa(
    mdpa,
    dh: DofHandler,
    mesh: Mesh,
    *,
    element_block_name: str,
) -> TraceFractureNetwork2D:
    paired = _fluid_pumping_paired_interfaces_from_mdpa(
        mdpa,
        dh,
        mesh,
        element_block_name=element_block_name,
    )
    neg_owner_ids, pos_owner_ids = _mdpa_interface_owner_ids(
        mdpa,
        mesh,
        body_block_name="UPwSmallStrainFICElement2D3N",
        interface_block_name=element_block_name,
    )
    block = mdpa.element_block(element_block_name)
    entities: list[TraceStationEntity2D] = []
    for idx, (element_id, _conn) in enumerate(sorted(block.elements.items())):
        entities.append(
            _paired_interface_to_trace_station_entity(
                int(element_id),
                paired[int(idx)],
                owner_neg_id=int(neg_owner_ids[int(idx)]),
                owner_pos_id=int(pos_owner_ids[int(idx)]),
            )
        )
    return TraceFractureNetwork2D(
        mesh=mesh,
        entities=tuple(entities),
        field_names=("ux", "uy", "p"),
        name="fluid_pumping_2d_fracture_trace_network",
    )


def _trace_network_as_paired_interfaces(network: TraceFractureNetwork2D) -> tuple[PairedUPlInterface2D, ...]:
    out: list[PairedUPlInterface2D] = []
    for entity in network.entities:
        out.append(
            PairedUPlInterface2D(
                negative_coords=tuple(tuple(float(v) for v in row) for row in entity.negative_coords_array()),
                positive_coords=tuple(tuple(float(v) for v in row) for row in entity.positive_coords_array()),
                negative_dofs=tuple(
                    (
                        int(entity.negative_dofs["ux"][i]),
                        int(entity.negative_dofs["uy"][i]),
                        int(entity.negative_dofs["p"][i]),
                    )
                    for i in range(entity.n_stations)
                ),
                positive_dofs=tuple(
                    (
                        int(entity.positive_dofs["ux"][i]),
                        int(entity.positive_dofs["uy"][i]),
                        int(entity.positive_dofs["p"][i]),
                    )
                    for i in range(entity.n_stations)
                ),
            )
        )
    return tuple(out)


def _fluid_pumping_interface_from_mdpa(
    mdpa,
    mesh: Mesh,
    *,
    element_block_name: str,
) -> NonMatchingInterface:
    body_block = mdpa.element_block("UPwSmallStrainFICElement2D3N")
    edge_to_owner: dict[tuple[int, int], list[int]] = {}
    element_id_to_index = getattr(mesh, "_mdpa_element_id_to_index", {})
    for old_element_id, conn in body_block.elements.items():
        owner = int(element_id_to_index[int(old_element_id)])
        tri = tuple(int(node_id) for node_id in conn)
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_to_owner.setdefault(tuple(sorted((a, b))), []).append(owner)

    block = mdpa.element_block(element_block_name)
    p0: list[np.ndarray] = []
    p1: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    neg_owner_ids: list[int] = []
    pos_owner_ids: list[int] = []
    for _interface_id, conn_raw in sorted(block.elements.items()):
        conn = tuple(int(node_id) for node_id in conn_raw)
        if len(conn) != 4:
            raise ValueError(f"{element_block_name} expects 4-node interface elements.")
        neg0, neg1, pos1, pos0 = conn
        neg_owner_ids.append(_single_edge_owner(edge_to_owner, neg0, neg1))
        pos_owner_ids.append(_single_edge_owner(edge_to_owner, pos0, pos1))

        x_neg0 = np.asarray(mdpa.nodes[neg0], dtype=float)
        x_neg1 = np.asarray(mdpa.nodes[neg1], dtype=float)
        x_pos0 = np.asarray(mdpa.nodes[pos0], dtype=float)
        x_pos1 = np.asarray(mdpa.nodes[pos1], dtype=float)
        mid0 = 0.5 * (x_neg0 + x_pos0)
        mid1 = 0.5 * (x_neg1 + x_pos1)
        gap = 0.5 * ((x_pos0 - x_neg0) + (x_pos1 - x_neg1))
        gap_norm = float(np.linalg.norm(gap))
        if gap_norm <= 1.0e-14:
            tangent = mid1 - mid0
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= 1.0e-14:
                raise ValueError(f"Degenerate interface segment in {element_block_name}.")
            tangent = tangent / tangent_norm
            gap = np.asarray([-tangent[1], tangent[0]], dtype=float)
            gap_norm = float(np.linalg.norm(gap))
        p0.append(mid0)
        p1.append(mid1)
        normals.append(gap / gap_norm)

    n_segments = len(p0)
    edge_ids = np.arange(n_segments, dtype=np.int32)
    return NonMatchingInterface(
        mesh_neg=mesh,
        mesh_pos=mesh,
        neg_edge_ids=edge_ids.copy(),
        pos_edge_ids=edge_ids.copy(),
        neg_elem_ids=np.asarray(neg_owner_ids, dtype=np.int32),
        pos_elem_ids=np.asarray(pos_owner_ids, dtype=np.int32),
        P0=np.asarray(p0, dtype=float),
        P1=np.asarray(p1, dtype=float),
        n=np.asarray(normals, dtype=float),
        h_neg=np.ones(n_segments, dtype=float),
        h_pos=np.ones(n_segments, dtype=float),
    )


def _mdpa_interface_owner_ids(
    mdpa,
    mesh: Mesh,
    *,
    body_block_name: str,
    interface_block_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return body-element owners for Kratos 2D4 interface negative/positive edges."""

    body_block = mdpa.element_block(body_block_name)
    edge_to_owner: dict[tuple[int, int], list[int]] = {}
    element_id_to_index = getattr(mesh, "_mdpa_element_id_to_index", {})
    for old_element_id, conn_raw in body_block.elements.items():
        owner = int(element_id_to_index[int(old_element_id)])
        conn = tuple(int(node_id) for node_id in conn_raw)
        if len(conn) < 3:
            raise ValueError(f"{body_block_name} elements must have at least three nodes.")
        for a, b in zip(conn, conn[1:] + conn[:1]):
            edge_to_owner.setdefault(tuple(sorted((int(a), int(b)))), []).append(owner)

    neg_owner_ids: list[int] = []
    pos_owner_ids: list[int] = []
    block = mdpa.element_block(interface_block_name)
    for _interface_id, conn_raw in sorted(block.elements.items()):
        conn = tuple(int(node_id) for node_id in conn_raw)
        if len(conn) != 4:
            raise ValueError(f"{interface_block_name} expects 4-node interface elements.")
        neg0, neg1, pos1, pos0 = conn
        neg_owner_ids.append(_single_edge_owner(edge_to_owner, neg0, neg1))
        pos_owner_ids.append(_single_edge_owner(edge_to_owner, pos0, pos1))
    return np.asarray(neg_owner_ids, dtype=np.int32), np.asarray(pos_owner_ids, dtype=np.int32)


def _fluid_pumping_paired_interfaces_from_mdpa(
    mdpa,
    dh: DofHandler,
    mesh: Mesh,
    *,
    element_block_name: str,
) -> tuple[PairedUPlInterface2D, ...]:
    old_to_new = getattr(mesh, "_mdpa_old_to_new_node", {})

    def dofs(old_node_id: int) -> tuple[int, int, int]:
        new_node_id = int(old_to_new[int(old_node_id)])
        return (
            int(dh.dof_map["ux"][new_node_id]),
            int(dh.dof_map["uy"][new_node_id]),
            int(dh.dof_map["p"][new_node_id]),
        )

    out: list[PairedUPlInterface2D] = []
    block = mdpa.element_block(element_block_name)
    for _interface_id, conn_raw in sorted(block.elements.items()):
        conn = tuple(int(node_id) for node_id in conn_raw)
        if len(conn) != 4:
            raise ValueError(f"{element_block_name} expects 4-node interface elements.")
        neg0, neg1, pos1, pos0 = conn
        out.append(
            PairedUPlInterface2D(
                negative_coords=(mdpa.nodes[neg0], mdpa.nodes[neg1]),
                positive_coords=(mdpa.nodes[pos0], mdpa.nodes[pos1]),
                negative_dofs=(dofs(neg0), dofs(neg1)),
                positive_dofs=(dofs(pos0), dofs(pos1)),
            )
        )
    return tuple(out)


def _paired_interfaces_from_mdpa_owner_dofs(
    mdpa,
    dh: DofHandler,
    mesh: Mesh,
    *,
    body_block_name: str,
    element_block_name: str,
) -> tuple[PairedUPlInterface2D, ...]:
    neg_owner_ids, pos_owner_ids = _mdpa_interface_owner_ids(
        mdpa,
        mesh,
        body_block_name=body_block_name,
        interface_block_name=element_block_name,
    )

    def dofs(owner: int, old_node_id: int) -> tuple[int, int, int]:
        return (
            _element_local_dof_for_old_node(dh, mesh, int(owner), int(old_node_id), "ux"),
            _element_local_dof_for_old_node(dh, mesh, int(owner), int(old_node_id), "uy"),
            _element_local_dof_for_old_node(dh, mesh, int(owner), int(old_node_id), "p"),
        )

    out: list[PairedUPlInterface2D] = []
    block = mdpa.element_block(element_block_name)
    for idx, (_interface_id, conn_raw) in enumerate(sorted(block.elements.items())):
        conn = tuple(int(node_id) for node_id in conn_raw)
        if len(conn) != 4:
            raise ValueError(f"{element_block_name} expects 4-node interface elements.")
        neg0, neg1, pos1, pos0 = conn
        neg_owner = int(neg_owner_ids[int(idx)])
        pos_owner = int(pos_owner_ids[int(idx)])
        out.append(
            PairedUPlInterface2D(
                negative_coords=(mdpa.nodes[neg0], mdpa.nodes[neg1]),
                positive_coords=(mdpa.nodes[pos0], mdpa.nodes[pos1]),
                negative_dofs=(dofs(neg_owner, neg0), dofs(neg_owner, neg1)),
                positive_dofs=(dofs(pos_owner, pos0), dofs(pos_owner, pos1)),
            )
        )
    return tuple(out)


def _element_local_dof_for_old_node(
    dh: DofHandler,
    mesh: Mesh,
    element_id: int,
    old_node_id: int,
    field: str,
) -> int:
    new_to_old = np.asarray(getattr(mesh, "_mdpa_new_to_old_node", ()), dtype=int)
    conn = np.asarray(mesh.elements_connectivity[int(element_id)], dtype=int)
    old_conn = new_to_old[conn]
    local = np.flatnonzero(old_conn == int(old_node_id))
    if local.size != 1:
        raise RuntimeError(
            f"Expected exactly one local node for old mdpa node {old_node_id} "
            f"in element {element_id}, found {local.size}."
        )
    local_id = int(local[0])
    field_map = np.asarray(dh.element_maps[field][int(element_id)], dtype=int)
    if local_id >= field_map.size:
        raise RuntimeError(
            f"Field {field!r} does not have a local DOF for old mdpa node {old_node_id} "
            f"in element {element_id}."
        )
    return int(field_map[local_id])


def _assemble_paired_interface_batch_into_global(
    matrix,
    rhs: np.ndarray,
    interfaces: tuple[PairedUPlInterface2D, ...],
    material: UPlInterfaceMaterial2D,
    *,
    dt: float,
    theta_u: float,
    theta_p: float,
    previous_solution: np.ndarray,
    current_solution: np.ndarray,
    permeability_law: str,
) -> None:
    if not interfaces:
        return
    batch = build_paired_upl_interface_local_batch_2d(
        interfaces,
        material,
        dt=dt,
        theta_u=theta_u,
        theta_p=theta_p,
        previous_solution=previous_solution,
        current_solution=current_solution,
        quadrature="lobatto",
        backend="python",
        need_matrix=True,
        permeability_law=permeability_law,
    )
    if batch.K_elem is None:
        raise RuntimeError("Paired interface batch did not return element matrices.")
    for K_elem, F_elem, gdofs in zip(batch.K_elem, batch.F_elem, batch.gdofs_map):
        dofs = np.asarray(gdofs, dtype=int)
        rhs[dofs] += np.asarray(F_elem, dtype=float)
        for local_i, global_i in enumerate(dofs):
            for local_j, global_j in enumerate(dofs):
                matrix[int(global_i), int(global_j)] += float(K_elem[int(local_i), int(local_j)])


def _initial_interface_state(
    interfaces: tuple[PairedUPlInterface2D, ...],
    material: UPlInterfaceMaterial2D,
) -> np.ndarray:
    threshold = float(material.damage_threshold if material.damage_threshold is not None else 0.0)
    if not interfaces:
        return np.zeros((0, 0), dtype=float)
    return np.full((len(interfaces), interfaces[0].n_stations), threshold, dtype=float)


def _lobatto_nonmatching_interface_layout(*, order: int) -> QuadratureLayout:
    xi, weights = gauss_lobatto(max(2, int(order)))
    return QuadratureLayout(
        entity_kind="nonmatching_interface",
        cell_type="line",
        quadrature_order=max(2, int(order)),
        reference_points=np.asarray(xi, dtype=float).reshape(-1, 1),
        reference_weights=np.asarray(weights, dtype=float),
    )


def _lobatto_trace_link_layout(*, order: int) -> QuadratureLayout:
    xi, weights = gauss_lobatto(max(2, int(order)))
    return QuadratureLayout(
        entity_kind="trace_link",
        cell_type="line",
        quadrature_order=max(2, int(order)),
        reference_points=np.asarray(xi, dtype=float).reshape(-1, 1),
        reference_weights=np.asarray(weights, dtype=float),
    )


def _assemble_ufl_interface_newton_batch_into_global(
    matrix,
    residual: np.ndarray,
    *,
    compiler: FormCompiler,
    system,
    state_field,
    state_variables: np.ndarray,
) -> np.ndarray:
    state_arr = np.asarray(state_variables, dtype=float)
    if state_arr.shape != tuple(state_field.values.shape):
        raise ValueError(
            f"Interface state shape mismatch for {state_field.name!r}: "
            f"expected {state_field.values.shape}, got {state_arr.shape}."
        )
    state_field.assign(state_arr)

    batch = compiler.assemble_local_contributions(system.equation)
    if batch.K_elem is None or batch.F_elem is None:
        raise RuntimeError("UFL interface Newton assembly must return both local tangent and residual.")
    _scatter_local_batch_into_global(matrix, residual, batch.K_elem, batch.F_elem, batch.gdofs_map)

    md = getattr(getattr(system, "measure", None), "metadata", None) or {}
    quadrature_rule = str(md.get("quadrature", md.get("rule", "gauss_lobatto")))
    if str(state_field.layout.entity_kind) == "trace_link":
        values = compiler.evaluate_trace_link_expressions_on_quadrature(
            {"state_next": system.state_update_expr},
            layout=state_field.layout,
            trace=system.interface,
            quadrature=quadrature_rule,
        )
    else:
        values = compiler.evaluate_nonmatching_interface_expressions_on_quadrature(
            {"state_next": system.state_update_expr},
            layout=state_field.layout,
            interface=system.interface,
            quadrature=quadrature_rule,
        )
    state_next = np.asarray(values["state_next"], dtype=float)
    if state_next.shape != state_arr.shape:
        raise RuntimeError(
            f"UFL interface state update for {state_field.name!r} returned shape "
            f"{state_next.shape}, expected {state_arr.shape}."
        )
    return state_next


def _scatter_local_batch_into_global(
    matrix,
    residual: np.ndarray,
    K_elem: np.ndarray,
    R_elem: np.ndarray,
    gdofs_map: np.ndarray,
) -> None:
    for local_K, local_R, gdofs_raw in zip(K_elem, R_elem, gdofs_map):
        gdofs = np.asarray(gdofs_raw, dtype=int).reshape(-1)
        if gdofs.size == 0:
            continue
        K_active = np.asarray(local_K, dtype=float)
        R_active = np.asarray(local_R, dtype=float).reshape(-1)
        if K_active.shape != (gdofs.size, gdofs.size):
            raise RuntimeError(
                f"Local matrix shape {K_active.shape} does not match DOF row width {gdofs.size}."
            )
        if R_active.shape != (gdofs.size,):
            raise RuntimeError(
                f"Local residual shape {R_active.shape} does not match DOF row width {gdofs.size}."
            )
        np.add.at(residual, gdofs, R_active)
        for local_i, global_i in enumerate(gdofs):
            for local_j, global_j in enumerate(gdofs):
                matrix[int(global_i), int(global_j)] += float(K_active[int(local_i), int(local_j)])


def _scatter_local_residual_into_global(
    residual: np.ndarray,
    R_elem: np.ndarray,
    gdofs_map: np.ndarray,
) -> None:
    for local_R, gdofs_raw in zip(R_elem, gdofs_map):
        gdofs = np.asarray(gdofs_raw, dtype=int).reshape(-1)
        if gdofs.size == 0:
            continue
        R_active = np.asarray(local_R, dtype=float).reshape(-1)
        if R_active.shape != (gdofs.size,):
            raise RuntimeError(
                f"Local residual shape {R_active.shape} does not match DOF row width {gdofs.size}."
            )
        np.add.at(residual, gdofs, R_active)


def _assemble_paired_interface_newton_batch_into_global(
    matrix,
    residual: np.ndarray,
    interfaces: tuple[PairedUPlInterface2D, ...],
    material: UPlInterfaceMaterial2D,
    *,
    dt: float,
    theta_u: float,
    theta_p: float,
    previous_solution: np.ndarray,
    current_solution: np.ndarray,
    velocity_solution: np.ndarray,
    dt_pressure_solution: np.ndarray,
    state_variables: np.ndarray,
    permeability_law: str,
) -> np.ndarray:
    if not interfaces:
        return state_variables
    batch = build_paired_upl_interface_kratos_newton_batch_2d(
        interfaces,
        material,
        dt=dt,
        theta_u=theta_u,
        theta_p=theta_p,
        previous_solution=previous_solution,
        current_solution=current_solution,
        velocity_solution=velocity_solution,
        dt_pressure_solution=dt_pressure_solution,
        state_variables=state_variables,
        quadrature="lobatto",
        backend="python",
        permeability_law=permeability_law,  # type: ignore[arg-type]
    )
    for K_elem, R_elem, gdofs in zip(batch.K_elem, batch.R_elem, batch.gdofs_map):
        dofs = np.asarray(gdofs, dtype=int)
        residual[dofs] += np.asarray(R_elem, dtype=float)
        for local_i, global_i in enumerate(dofs):
            for local_j, global_j in enumerate(dofs):
                matrix[int(global_i), int(global_j)] += float(K_elem[int(local_i), int(local_j)])
    return np.asarray(batch.state_next, dtype=float)


def _single_edge_owner(edge_to_owner: dict[tuple[int, int], list[int]], a: int, b: int) -> int:
    owners = edge_to_owner.get(tuple(sorted((int(a), int(b)))), ())
    if len(owners) != 1:
        raise RuntimeError(f"Expected exactly one body owner for mdpa edge {(a, b)}, found {len(owners)}.")
    return int(owners[0])


def _fluid_pumping_normal_flux_process(params: dict) -> tuple[str, float]:
    for process in params.get("processes", {}).get("loads_process_list", ()):
        process_params = process.get("Parameters", {})
        variable_name = str(process_params.get("variable_name", ""))
        if variable_name in {"NORMAL_FLUID_FLUX", "NORMAL_LIQUID_FLUX"}:
            part_name = str(process_params["model_part_name"]).split(".")[-1]
            return part_name, float(process_params.get("value", 0.0))
    raise RuntimeError("fluid_pumping_2D normal-liquid-flux process was not found.")


def _add_kratos_interface_normal_liquid_flux(
    rhs: np.ndarray,
    dh: DofHandler,
    mesh: Mesh,
    mdpa,
    *,
    part_name: str,
    normal_flux: float,
    initial_joint_width: float,
) -> None:
    part = mdpa.submodelparts[str(part_name)]
    old_to_new = getattr(mesh, "_mdpa_old_to_new_node", {})
    for condition_id in part.condition_ids:
        conn = tuple(int(node_id) for node_id in mdpa.conditions[int(condition_id)])
        if len(conn) != 2:
            raise ValueError("fluid_pumping_2D currently supports 2-node interface flux conditions.")
        # Kratos' 2D interface-flux condition uses the aperture as integration
        # coefficient. The process value convention is already "positive =
        # inlet", so both condition pressure nodes receive the same signed
        # half-aperture contribution.
        weights = np.asarray([0.5, 0.5], dtype=float) * float(initial_joint_width)
        for old_node_id, weight in zip(conn, weights):
            new_node_id = int(old_to_new[int(old_node_id)])
            dof = dh.dof_map["p"].get(new_node_id)
            if dof is not None:
                rhs[int(dof)] += float(normal_flux) * float(weight)


def _fluid_pumping_normal_flux_processes(params: dict) -> tuple[tuple[str, float], ...]:
    out: list[tuple[str, float]] = []
    for process in params.get("processes", {}).get("loads_process_list", ()):
        process_params = process.get("Parameters", {})
        variable_name = str(process_params.get("variable_name", ""))
        if variable_name in {"NORMAL_FLUID_FLUX", "NORMAL_LIQUID_FLUX"}:
            part_name = str(process_params["model_part_name"]).split(".")[-1]
            out.append((part_name, float(process_params.get("value", 0.0))))
    if not out:
        raise RuntimeError("normal-liquid-flux process was not found.")
    return tuple(out)


def _condition_block_name_by_id(mdpa) -> dict[int, str]:
    out: dict[int, str] = {}
    for block in mdpa.condition_blocks:
        for condition_id in block.conditions:
            out[int(condition_id)] = str(block.name)
    return out


def _add_kratos_normal_liquid_flux_processes(
    rhs: np.ndarray,
    dh: DofHandler,
    mesh: Mesh,
    mdpa,
    *,
    flux_processes: tuple[tuple[str, float], ...],
    interface_aperture: float,
) -> None:
    block_by_id = _condition_block_name_by_id(mdpa)
    old_to_new = getattr(mesh, "_mdpa_old_to_new_node", {})
    for part_name, normal_flux in flux_processes:
        part = mdpa.submodelparts[str(part_name)]
        for condition_id in part.condition_ids:
            conn = tuple(int(node_id) for node_id in mdpa.conditions[int(condition_id)])
            if len(conn) != 2:
                raise ValueError("Only 2-node normal-liquid-flux conditions are supported.")
            block_name = block_by_id.get(int(condition_id), "")
            if "Interface" in block_name:
                weights = np.asarray([0.5, 0.5], dtype=float) * float(interface_aperture)
            else:
                x0 = np.asarray(mdpa.nodes[int(conn[0])], dtype=float)
                x1 = np.asarray(mdpa.nodes[int(conn[1])], dtype=float)
                weights = np.asarray([0.5, 0.5], dtype=float) * float(np.linalg.norm(x1 - x0))
            for old_node_id, weight in zip(conn, weights):
                new_node_id = int(old_to_new[int(old_node_id)])
                dof = dh.dof_map["p"].get(new_node_id)
                if dof is not None:
                    rhs[int(dof)] += float(normal_flux) * float(weight)


def _should_insert_trace_fracture(
    network: TraceFractureNetwork2D,
    damage_state: np.ndarray,
    settings: TraceFracturePropagationSettings2D,
    *,
    step: int,
    forced_steps: set[int],
    event_count: int,
    max_events: int | None,
    bulk_tip_damage: float | None = None,
) -> bool:
    if max_events is not None and int(event_count) >= int(max_events):
        return False
    if int(step) in forced_steps:
        return True
    if network.n_entities == 0:
        return False
    if bulk_tip_damage is not None:
        return float(bulk_tip_damage) >= float(settings.damage_threshold)
    parent_id = _rightmost_trace_entity_id(network)
    idx = network.entity_ids.index(int(parent_id))
    damage = np.asarray(damage_state, dtype=float)
    if damage.ndim < 1 or int(damage.shape[0]) != network.n_entities:
        raise ValueError(
            f"damage_state first dimension must match trace entity count {network.n_entities}, got {damage.shape}."
        )
    # Reuse the public planner for the threshold semantics, then restrict the
    # actual insertion to the active crack tip.  This avoids creating a new row
    # for every already-damaged interior interface segment.
    tip_network = TraceFractureNetwork2D(
        mesh=network.mesh,
        entities=(network.entity(parent_id),),
        field_names=network.field_names,
        name=f"{network.name}_tip",
    )
    plans = plan_fracture_extensions_from_damage(
        tip_network,
        damage[idx : idx + 1],
        settings,
        tip="end",
    )
    return bool(plans)


def _trace_tip_bulk_damage_probe(
    network: TraceFractureNetwork2D,
    damage_values: np.ndarray,
    quadrature_points: np.ndarray,
    quadrature_weights: np.ndarray,
    *,
    search_radius: float,
) -> dict[str, object]:
    """Return the Kratos-style weighted bulk-damage probe at the active tip.

    Kratos' 2D fracture utility does not use the global maximum bulk damage and
    it does not use a max over the crack-tip neighbourhood.  It searches volume
    quadrature points within one propagation length from the current tip and
    computes

    ``sum_q w_q exp(-4 r_q^2 / L^2) d_q / sum_q w_q exp(-4 r_q^2 / L^2)``.

    Keeping this exact neighbourhood semantics is important for Trace-FEM:
    damage hot spots away from the active tip must be reported as diagnostics,
    not silently converted into crack growth.
    """

    if network.n_entities == 0:
        return {"damage": 0.0, "support_count": 0, "centroid": (0.0, 0.0), "tip": (0.0, 0.0)}
    damage = np.asarray(damage_values, dtype=float)
    points = np.asarray(quadrature_points, dtype=float)
    weights = np.asarray(quadrature_weights, dtype=float)
    if damage.shape != points.shape[:2]:
        raise ValueError(
            "bulk damage and quadrature-point layouts are incompatible: "
            f"{damage.shape} vs {points.shape[:2]}."
        )
    if weights.shape != damage.shape:
        raise ValueError(
            "bulk damage and quadrature-weight layouts are incompatible: "
            f"{damage.shape} vs {weights.shape}."
        )
    parent = network.entity(_rightmost_trace_entity_id(network))
    tip = np.asarray(parent.midpoint_coords()[-1], dtype=float).reshape(2)
    flat_points = points.reshape(-1, points.shape[-1])
    flat_damage = damage.reshape(-1)
    flat_weights = weights.reshape(-1)
    dist = np.linalg.norm(flat_points[:, :2] - tip[None, :], axis=1)
    radius = float(search_radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("trace-tip bulk-damage search_radius must be positive.")
    mask = dist <= radius
    if not np.any(mask):
        return {"damage": 0.0, "support_count": 0, "centroid": (float(tip[0]), float(tip[1])), "tip": (float(tip[0]), float(tip[1]))}
    local_dist = dist[mask]
    raw = flat_weights[mask] * np.exp(-4.0 * local_dist * local_dist / (radius * radius))
    denom = float(np.sum(raw))
    if not np.isfinite(denom) or denom <= 1.0e-20:
        return {"damage": 0.0, "support_count": int(np.count_nonzero(mask)), "centroid": (float(tip[0]), float(tip[1])), "tip": (float(tip[0]), float(tip[1]))}
    value = float(np.sum(raw * flat_damage[mask]) / denom)
    damage_weight = raw * np.maximum(flat_damage[mask], 0.0)
    damage_denom = float(np.sum(damage_weight))
    if damage_denom > 1.0e-20:
        centroid_arr = np.sum(damage_weight[:, None] * flat_points[mask, :2], axis=0) / damage_denom
    else:
        centroid_arr = tip
    return {
        "damage": value,
        "support_count": int(np.count_nonzero(mask)),
        "centroid": (float(centroid_arr[0]), float(centroid_arr[1])),
        "tip": (float(tip[0]), float(tip[1])),
    }


def _trace_tip_bulk_damage_max(
    network: TraceFractureNetwork2D,
    damage_values: np.ndarray,
    quadrature_points: np.ndarray,
    *,
    search_radius: float,
) -> float:
    """Return the maximum bulk damage around the active trace-fracture tip.

    Retained for old diagnostics. Propagation decisions should use
    :func:`_trace_tip_bulk_damage_probe` so isolated maxima do not drive crack
    growth.
    """

    if network.n_entities == 0:
        return 0.0
    damage = np.asarray(damage_values, dtype=float)
    points = np.asarray(quadrature_points, dtype=float)
    if damage.shape != points.shape[:2]:
        raise ValueError(
            "bulk damage and quadrature-point layouts are incompatible: "
            f"{damage.shape} vs {points.shape[:2]}."
        )
    parent = network.entity(_rightmost_trace_entity_id(network))
    tip = np.asarray(parent.midpoint_coords()[-1], dtype=float).reshape(2)
    flat_points = points.reshape(-1, points.shape[-1])
    flat_damage = damage.reshape(-1)
    dist = np.linalg.norm(flat_points[:, :2] - tip[None, :], axis=1)
    radius = float(search_radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("trace-tip bulk-damage search_radius must be positive.")
    mask = dist <= radius
    if not np.any(mask):
        return 0.0
    return float(np.max(flat_damage[mask]))


def _bulk_damage_max_location(
    damage_values: np.ndarray,
    quadrature_points: np.ndarray,
) -> tuple[float, float]:
    damage = np.asarray(damage_values, dtype=float)
    points = np.asarray(quadrature_points, dtype=float)
    if damage.shape != points.shape[:2]:
        raise ValueError(
            "bulk damage and quadrature-point layouts are incompatible: "
            f"{damage.shape} vs {points.shape[:2]}."
        )
    flat_damage = damage.reshape(-1)
    flat_points = points.reshape(-1, points.shape[-1])
    idx = int(np.argmax(flat_damage))
    return (float(flat_points[idx, 0]), float(flat_points[idx, 1]))


def _rightmost_trace_entity_id(network: TraceFractureNetwork2D) -> int:
    if network.n_entities == 0:
        raise ValueError("Cannot select a crack tip from an empty trace-fracture network.")
    best_id = int(network.entities[0].entity_id)
    best_x = -np.inf
    for entity in network.entities:
        mids = entity.midpoint_coords()
        x_tip = float(mids[-1, 0])
        if x_tip > best_x:
            best_x = x_tip
            best_id = int(entity.entity_id)
    return best_id


def _make_trace_fracture_tip_extension_from_existing_nodes(
    network: TraceFractureNetwork2D,
    mdpa,
    dh: DofHandler,
    mesh: Mesh,
    *,
    parent_entity_id: int,
    new_entity_id: int,
    propagation_settings: TraceFracturePropagationSettings2D,
) -> TraceStationEntity2D:
    parent = network.entity(parent_entity_id)
    parent_neg = parent.negative_coords_array()
    parent_pos = parent.positive_coords_array()
    mids = parent.midpoint_coords()
    tangent = mids[-1] - mids[0]
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1.0e-14:
        raise ValueError(f"Parent trace entity {parent_entity_id} has degenerate tangent.")
    tangent = tangent / tangent_norm
    normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
    side_gap = parent_pos[0] - parent_neg[0]
    if float(np.dot(normal, side_gap)) < 0.0:
        normal = -normal

    start_mid = mids[-1]
    end_mid = start_mid + float(propagation_settings.propagation_length) * tangent
    half_width = 0.5 * float(propagation_settings.propagation_width)
    end_neg = end_mid - half_width * normal
    end_pos = end_mid + half_width * normal

    neg_end_dofs = _nearest_mdpa_node_dofs(mdpa, dh, mesh, end_neg)
    pos_end_dofs = _nearest_mdpa_node_dofs(mdpa, dh, mesh, end_pos)
    negative_dofs = {
        "ux": (int(parent.negative_dofs["ux"][-1]), int(neg_end_dofs[0])),
        "uy": (int(parent.negative_dofs["uy"][-1]), int(neg_end_dofs[1])),
        "p": (int(parent.negative_dofs["p"][-1]), int(neg_end_dofs[2])),
    }
    positive_dofs = {
        "ux": (int(parent.positive_dofs["ux"][-1]), int(pos_end_dofs[0])),
        "uy": (int(parent.positive_dofs["uy"][-1]), int(pos_end_dofs[1])),
        "p": (int(parent.positive_dofs["p"][-1]), int(pos_end_dofs[2])),
    }
    return network.make_tip_extension(
        parent_entity_id=int(parent_entity_id),
        new_entity_id=int(new_entity_id),
        negative_dofs=negative_dofs,
        positive_dofs=positive_dofs,
        length=float(propagation_settings.propagation_length),
        width=float(propagation_settings.propagation_width),
        tip="end",
        state_source_id=int(parent_entity_id),
        owner_neg_id=int(parent.owner_neg_id),
        owner_pos_id=int(parent.owner_pos_id),
    )


def _nearest_mdpa_node_dofs(mdpa, dh: DofHandler, mesh: Mesh, point: np.ndarray) -> tuple[int, int, int]:
    point = np.asarray(point, dtype=float).reshape(2)
    best_node_id = None
    best_dist2 = np.inf
    for old_node_id, coords in mdpa.nodes.items():
        dist2 = float(np.sum((np.asarray(coords, dtype=float) - point) ** 2))
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_node_id = int(old_node_id)
    if best_node_id is None:
        raise RuntimeError("Cannot select nearest node for trace-fracture extension: mdpa has no nodes.")
    old_to_new = getattr(mesh, "_mdpa_old_to_new_node", {})
    new_node_id = int(old_to_new[int(best_node_id)])
    return (
        int(dh.dof_map["ux"][new_node_id]),
        int(dh.dof_map["uy"][new_node_id]),
        int(dh.dof_map["p"][new_node_id]),
    )


def _kratos_node_major_free_dofs(
    dh: DofHandler,
    mesh: Mesh,
    mdpa,
    constrained_dofs: tuple[int, ...],
) -> np.ndarray:
    """Return free pycutfem dofs in Kratos elimination-builder equation order."""

    old_to_new = getattr(mesh, "_mdpa_old_to_new_node", {})
    constrained = set(int(dof) for dof in constrained_dofs)
    ordered: list[int] = []
    for old_node_id in mdpa.nodes:
        new_node_id = old_to_new.get(int(old_node_id))
        if new_node_id is None:
            continue
        for field_name in ("ux", "uy", "p"):
            dof = int(dh.dof_map[field_name][int(new_node_id)])
            if dof not in constrained:
                ordered.append(dof)
    if len(set(ordered)) != len(ordered):
        raise RuntimeError("Kratos node-major free-DOF ordering contains duplicates.")
    return np.asarray(ordered, dtype=int)


def _fluid_pumping_structural_dof_pairs(
    mdpa,
    dh: DofHandler,
    mesh: Mesh,
    *,
    fracture_interfaces: tuple[PairedUPlInterface2D, ...],
    link_interfaces: tuple[PairedUPlInterface2D, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Kratos element graph as global matrix row/column pairs."""

    old_to_new = getattr(mesh, "_mdpa_old_to_new_node", {})

    def node_dofs(old_node_id: int) -> tuple[int, int, int]:
        new_node_id = int(old_to_new[int(old_node_id)])
        return (
            int(dh.dof_map["ux"][new_node_id]),
            int(dh.dof_map["uy"][new_node_id]),
            int(dh.dof_map["p"][new_node_id]),
        )

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []

    def add_pairs(dofs: np.ndarray) -> None:
        dofs = np.asarray(dofs, dtype=np.int64).reshape(-1)
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows.append(rr.reshape(-1))
        cols.append(cc.reshape(-1))

    for conn in mdpa.element_block("UPwSmallStrainFICElement2D3N").elements.values():
        add_pairs(np.asarray([node_dofs(int(node_id)) for node_id in conn], dtype=np.int64).reshape(-1))
    for interface in (*fracture_interfaces, *link_interfaces):
        add_pairs(interface.local_dofs())

    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    key = row.astype(np.int64) * np.int64(dh.total_dofs) + col.astype(np.int64)
    unique_key = np.unique(key)
    return unique_key // np.int64(dh.total_dofs), unique_key % np.int64(dh.total_dofs)


def _patch_fluid_pumping_mdpa_for_installed_kratos(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = {
        "BULK_MODULUS_FLUID": "BULK_MODULUS_LIQUID",
        "DENSITY_WATER": "DENSITY_LIQUID",
        "DYNAMIC_VISCOSITY": "DYNAMIC_VISCOSITY_LIQUID",
        "UPwSmallStrainFICElement2D3N": "UPlSmallStrainFICElement2D3N",
        "UPwSmallStrainInterfaceElement2D4N": "UPlSmallStrainInterfaceElement2D4N",
        "UPwSmallStrainLinkInterfaceElement2D4N": "UPlSmallStrainLinkInterfaceElement2D4N",
        "UPwNormalFluxFICCondition2D2N": "UPlNormalLiquidFluxFICCondition2D2N",
        "UPwNormalFluxInterfaceCondition2D2N": "UPlNormalLiquidFluxInterfaceCondition2D2N",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def _patch_fluid_pumping_materials_for_installed_kratos(path: Path) -> None:
    data = load_kratos_json(path)
    replacements = {
        "BULK_MODULUS_FLUID": "BULK_MODULUS_LIQUID",
        "DENSITY_WATER": "DENSITY_LIQUID",
        "DYNAMIC_VISCOSITY": "DYNAMIC_VISCOSITY_LIQUID",
    }
    for prop in data.get("properties", ()):
        variables = prop.get("Material", {}).get("Variables", {})
        for old, new in replacements.items():
            if old in variables:
                variables[new] = variables.pop(old)
    path.write_text(json.dumps(data), encoding="utf-8")


def _patch_fluid_pumping_project_parameters_for_installed_kratos(path: Path, *, end_time: float) -> None:
    data = load_kratos_json(path)
    data["problem_data"]["end_time"] = float(end_time)
    data["problem_data"]["echo_level"] = 0
    solver_settings = data["solver_settings"]
    solver_settings["solver_type"] = "poromechanics_U_Pl_solver"
    solver_settings["echo_level"] = 0
    if "newmark_theta" in solver_settings:
        theta = solver_settings.pop("newmark_theta")
        solver_settings["newmark_theta_u"] = theta
        solver_settings["newmark_theta_p"] = theta
    data["output_processes"] = {}
    for process in data.get("processes", {}).get("constraints_process_list", ()):
        process_params = process.get("Parameters", {})
        if process_params.get("variable_name") == "WATER_PRESSURE":
            process_params["variable_name"] = "LIQUID_PRESSURE"
    for process in data.get("processes", {}).get("loads_process_list", ()):
        process_params = process.get("Parameters", {})
        if process_params.get("variable_name") == "NORMAL_FLUID_FLUX":
            process_params["variable_name"] = "NORMAL_LIQUID_FLUX"
    path.write_text(json.dumps(data), encoding="utf-8")


def _patch_upl_validation_mdpa_for_installed_kratos(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = {
        "BULK_MODULUS_FLUID": "BULK_MODULUS_LIQUID",
        "DENSITY_WATER": "DENSITY_LIQUID",
        "DYNAMIC_VISCOSITY": "DYNAMIC_VISCOSITY_LIQUID",
        "UPwSmallStrainElement2D4N": "UPlSmallStrainElement2D4N",
        "UPwSmallStrainInterfaceElement2D4N": "UPlSmallStrainInterfaceElement2D4N",
        "UPwFaceLoadCondition2D2N": "UPlFaceLoadCondition2D2N",
        "UPwNormalFluxCondition2D2N": "UPlNormalLiquidFluxCondition2D2N",
        "UPwNormalFluxInterfaceCondition2D2N": "UPlNormalLiquidFluxInterfaceCondition2D2N",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def _patch_upl_validation_materials_for_installed_kratos(path: Path) -> None:
    data = load_kratos_json(path)
    replacements = {
        "BULK_MODULUS_FLUID": "BULK_MODULUS_LIQUID",
        "DENSITY_WATER": "DENSITY_LIQUID",
        "DYNAMIC_VISCOSITY": "DYNAMIC_VISCOSITY_LIQUID",
    }
    for prop in data.get("properties", ()):
        variables = prop.get("Material", {}).get("Variables", {})
        for old, new in replacements.items():
            if old in variables:
                variables[new] = variables.pop(old)
    path.write_text(json.dumps(data), encoding="utf-8")


def _patch_upl_validation_project_parameters_for_installed_kratos(path: Path, *, end_time: float) -> None:
    data = load_kratos_json(path)
    data["problem_data"]["end_time"] = float(end_time)
    data["problem_data"]["echo_level"] = 0
    solver_settings = data["solver_settings"]
    solver_settings["solver_type"] = "poromechanics_U_Pl_solver"
    solver_settings["echo_level"] = 0
    if "newmark_theta" in solver_settings:
        theta = solver_settings.pop("newmark_theta")
        solver_settings["newmark_theta_u"] = theta
        solver_settings["newmark_theta_p"] = theta
    data["output_processes"] = {}
    for process_group in ("constraints_process_list", "loads_process_list"):
        for process in data.get("processes", {}).get(process_group, ()):
            process_params = process.get("Parameters", {})
            if process_params.get("variable_name") == "WATER_PRESSURE":
                process_params["variable_name"] = "LIQUID_PRESSURE"
            if process_params.get("variable_name") == "NORMAL_FLUID_FLUX":
                process_params["variable_name"] = "NORMAL_LIQUID_FLUX"
    path.write_text(json.dumps(data), encoding="utf-8")


def _kratos_consolidation_interface_trace(dh: DofHandler) -> PairedUPlInterface2D:
    def dofs(eid: int, local: int) -> tuple[int, int, int]:
        return (
            int(dh.element_maps["ux"][eid][local]),
            int(dh.element_maps["uy"][eid][local]),
            int(dh.element_maps["p"][eid][local]),
        )

    return PairedUPlInterface2D(
        negative_coords=((0.5, 1.0), (0.5, 0.0)),
        positive_coords=((0.5, 1.0), (0.5, 0.0)),
        # Left body side, ordered top -> bottom: Kratos nodes 2, 7.
        negative_dofs=(dofs(1, 3), dofs(1, 1)),
        # Right body side, ordered top -> bottom: Kratos nodes 3, 6.
        positive_dofs=(dofs(0, 2), dofs(0, 0)),
    )


def _kratos_consolidation_interface_face_load_dofs(dh: DofHandler) -> tuple[int, ...]:
    # Top right edge: Kratos nodes 1,3 -> right element local 3,2.
    # Top left edge: Kratos nodes 2,5 -> left element local 3,2.
    return (
        int(dh.element_maps["uy"][0][3]),
        int(dh.element_maps["uy"][0][2]),
        int(dh.element_maps["uy"][1][3]),
        int(dh.element_maps["uy"][1][2]),
    )


def _add_kratos_top_face_load(rhs: np.ndarray, face_load_dofs: tuple[int, ...]) -> None:
    # FACE_LOAD_Y=-1e4 on two length-0.5 top conditions. Each endpoint receives
    # traction * length / 2 = -2500.
    for dof in face_load_dofs:
        rhs[int(dof)] += -2.5e3


def _kratos_consolidation_interface_dirichlet_dofs(dh: DofHandler) -> tuple[int, ...]:
    bottom = [(0, 1), (0, 0), (1, 1), (1, 0)]  # nodes 4,6,7,8
    outer_x = [(0, 3), (0, 1), (1, 2), (1, 0)]  # nodes 1,4,5,8
    top_p = [(0, 3), (0, 2), (1, 3), (1, 2)]  # nodes 1,3,2,5

    dofs: set[int] = set()
    for eid, local in bottom:
        dofs.add(int(dh.element_maps["ux"][eid][local]))
        dofs.add(int(dh.element_maps["uy"][eid][local]))
    for eid, local in outer_x:
        dofs.add(int(dh.element_maps["ux"][eid][local]))
    for eid, local in top_p:
        dofs.add(int(dh.element_maps["p"][eid][local]))
    return tuple(sorted(dofs))


def _kratos_consolidation_interface_pressure_dofs_by_node(dh: DofHandler) -> dict[int, int]:
    return {
        1: int(dh.element_maps["p"][0][3]),
        2: int(dh.element_maps["p"][1][3]),
        3: int(dh.element_maps["p"][0][2]),
        4: int(dh.element_maps["p"][0][1]),
        5: int(dh.element_maps["p"][1][2]),
        6: int(dh.element_maps["p"][0][0]),
        7: int(dh.element_maps["p"][1][1]),
        8: int(dh.element_maps["p"][1][0]),
    }


def _kratos_process_dirichlet_dofs(
    dh: DofHandler,
    mesh: Mesh,
    mdpa,
    constraint_processes,
) -> tuple[int, ...]:
    old_to_new = getattr(mesh, "_mdpa_old_to_new_node", {})
    dofs: set[int] = set()
    for process in constraint_processes:
        params = process["Parameters"]
        part_name = str(params["model_part_name"]).split(".")[-1]
        part = mdpa.submodelparts.get(part_name)
        if part is None:
            continue
        variable = str(params["variable_name"])
        if variable in {"DISPLACEMENT", "VELOCITY", "ACCELERATION"}:
            active = np.asarray(params.get("active", [True, True, False]), dtype=bool)
            field_names = ["ux", "uy"]
            for old_node_id in part.node_ids:
                for component, field in enumerate(field_names):
                    if component < active.size and bool(active[component]):
                        for dof in _field_dofs_matching_mdpa_nodes(dh, mesh, field, (old_node_id,)):
                            dofs.add(int(dof))
        elif variable in {"LIQUID_PRESSURE", "WATER_PRESSURE"}:
            for dof in _field_dofs_matching_mdpa_nodes(dh, mesh, "p", part.node_ids):
                dofs.add(int(dof))
    return tuple(sorted(dofs))


def _field_dofs_matching_mdpa_nodes(
    dh: DofHandler,
    mesh: Mesh,
    field: str,
    old_node_ids,
) -> tuple[int, ...]:
    old_node_ids = tuple(int(node_id) for node_id in old_node_ids)
    if not old_node_ids:
        return ()
    if int(dh.mixed_element._field_orders[field]) == int(mesh.poly_order):
        exact = _field_dofs_on_mdpa_volume_nodes(dh, mesh, field, old_node_ids)
        if exact:
            return exact
    dof_map = getattr(dh, "dof_map", {}).get(field, {})
    direct_dofs: list[int] = []
    fallback_old_node_ids: list[int] = []
    for node_id in old_node_ids:
        new_node_id = int(mesh._mdpa_old_to_new_node[int(node_id)])
        dof = dof_map.get(new_node_id)
        if dof is None:
            fallback_old_node_ids.append(int(node_id))
        else:
            direct_dofs.append(int(dof))
    if not fallback_old_node_ids:
        return tuple(sorted(set(direct_dofs)))

    targets = np.asarray(
        [mesh.nodes_x_y_pos[mesh._mdpa_old_to_new_node[int(node_id)]] for node_id in fallback_old_node_ids],
        dtype=float,
    )
    dofs: list[int] = []
    coords = np.asarray(dh.get_field_dof_coords(field), dtype=float)
    field_slice = np.asarray(dh.get_field_slice(field), dtype=int)
    for local, xy in enumerate(coords):
        if np.any(np.linalg.norm(targets - xy[None, :], axis=1) <= 1.0e-10):
            dofs.append(int(field_slice[int(local)]))
    return tuple(sorted(set(direct_dofs + dofs)))


def _field_dofs_on_mdpa_volume_nodes(
    dh: DofHandler,
    mesh: Mesh,
    field: str,
    old_node_ids,
) -> tuple[int, ...]:
    targets = {int(node_id) for node_id in old_node_ids}
    if not targets:
        return ()
    new_to_old = np.asarray(getattr(mesh, "_mdpa_new_to_old_node", ()), dtype=int)
    out: list[int] = []
    for eid, conn in enumerate(np.asarray(mesh.elements_connectivity, dtype=int)):
        field_map = np.asarray(dh.element_maps[field][int(eid)], dtype=int)
        for local_id, new_node_id in enumerate(conn):
            if int(new_to_old[int(new_node_id)]) not in targets:
                continue
            if int(local_id) < field_map.size:
                out.append(int(field_map[int(local_id)]))
    return tuple(sorted(set(out)))


def _add_kratos_diff_order_line_load_y(
    rhs: np.ndarray,
    dh: DofHandler,
    mesh: Mesh,
    mdpa,
    part_name: str,
    load_y: float,
) -> None:
    part = mdpa.submodelparts[str(part_name)]
    old_to_new = getattr(mesh, "_mdpa_old_to_new_node", {})
    for condition_id in part.condition_ids:
        conn = tuple(int(node_id) for node_id in mdpa.conditions[int(condition_id)])
        if len(conn) == 2:
            weights = _line_condition_weights(mdpa, conn, kind="linear")
        elif len(conn) == 3:
            weights = _line_condition_weights(mdpa, conn, kind="quadratic")
        else:
            raise ValueError(f"Unsupported line-load condition arity {len(conn)}.")
        for old_node_id, weight in zip(conn, weights):
            new_node_id = int(old_to_new[int(old_node_id)])
            dof = dh.dof_map["uy"].get(new_node_id)
            if dof is not None:
                rhs[int(dof)] += float(load_y) * float(weight)


def _add_kratos_line_load_component(
    rhs: np.ndarray,
    dh: DofHandler,
    mesh: Mesh,
    mdpa,
    part_name: str,
    component: int,
    value: float,
) -> None:
    part = mdpa.submodelparts[str(part_name)]
    field = ("ux", "uy")[int(component)]
    for condition_id in part.condition_ids:
        conn = tuple(int(node_id) for node_id in mdpa.conditions[int(condition_id)])
        if len(conn) == 2:
            weights = _line_condition_weights(mdpa, conn, kind="linear")
        elif len(conn) == 3:
            weights = _line_condition_weights(mdpa, conn, kind="quadratic")
        else:
            raise ValueError(f"Unsupported line-load condition arity {len(conn)}.")
        for old_node_id, weight in zip(conn, weights):
            for dof in _field_dofs_matching_mdpa_nodes(dh, mesh, field, (old_node_id,)):
                rhs[int(dof)] += float(value) * float(weight)


def _line_condition_weights(mdpa, conn: tuple[int, ...], *, kind: str) -> np.ndarray:
    coords = np.asarray([mdpa.nodes[int(node_id)] for node_id in conn], dtype=float)
    if kind == "linear":
        length = float(np.linalg.norm(coords[1] - coords[0]))
        return np.asarray([0.5 * length, 0.5 * length], dtype=float)
    if kind == "quadratic":
        # Kratos L3 diff-order line conditions use node order [end0, end1, mid].
        length = float(np.linalg.norm(coords[1] - coords[0]))
        return np.asarray([length / 6.0, length / 6.0, 2.0 * length / 3.0], dtype=float)
    raise ValueError(kind)


def _apply_dirichlet_zero(matrix, rhs: np.ndarray, dofs: tuple[int, ...]):
    if not sp.isspmatrix_lil(matrix):
        matrix = matrix.tolil()
    for dof in dofs:
        dof_i = int(dof)
        matrix[dof_i, :] = 0.0
        matrix[:, dof_i] = 0.0
        matrix[dof_i, dof_i] = 1.0
        rhs[dof_i] = 0.0
    return matrix.tocsr(), rhs


def _solve_dirichlet_increment(
    matrix,
    rhs: np.ndarray,
    dofs: tuple[int, ...],
    *,
    linear_solver: str,
    dirichlet_mode: str,
    free_dof_order: np.ndarray | None = None,
    structural_dof_pairs: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rhs_array = np.asarray(rhs, dtype=float)
    if dirichlet_mode == "zero_rows_columns":
        K_bc, F_bc = _apply_dirichlet_zero(matrix, rhs_array.copy(), dofs)
        return _solve_sparse_linear_system(K_bc, F_bc, linear_solver=linear_solver), F_bc
    if dirichlet_mode in {"elimination", "kratos_elimination"}:
        K_csr = _matrix_with_structural_zeros(matrix, structural_dof_pairs)
        if free_dof_order is None:
            constrained = np.zeros(K_csr.shape[0], dtype=bool)
            if dofs:
                constrained[np.asarray(dofs, dtype=int)] = True
            free = np.flatnonzero(~constrained)
        else:
            free = np.asarray(free_dof_order, dtype=int)
        out = np.zeros(K_csr.shape[0], dtype=float)
        reduced_rhs = rhs_array[free]
        if free.size:
            reduced_matrix = K_csr[free, :][:, free]
            out[free] = _solve_sparse_linear_system(reduced_matrix, reduced_rhs, linear_solver=linear_solver)
        return out, reduced_rhs
    raise ValueError("dirichlet_mode must be 'zero_rows_columns', 'elimination', or 'kratos_elimination'.")


def _matrix_with_structural_zeros(
    matrix,
    structural_dof_pairs: tuple[np.ndarray, np.ndarray] | None,
):
    if structural_dof_pairs is None:
        return matrix.tocsr()
    pattern_rows, pattern_cols = structural_dof_pairs
    if pattern_rows.size == 0:
        return matrix.tocsr()
    coo = matrix.tocoo()
    rows = np.concatenate((coo.row.astype(np.int64, copy=False), np.asarray(pattern_rows, dtype=np.int64)))
    cols = np.concatenate((coo.col.astype(np.int64, copy=False), np.asarray(pattern_cols, dtype=np.int64)))
    data = np.concatenate((coo.data.astype(float, copy=False), np.zeros(pattern_rows.size, dtype=float)))
    return sp.coo_matrix((data, (rows, cols)), shape=coo.shape).tocsr()


def _solve_sparse_linear_system(matrix, rhs: np.ndarray, *, linear_solver: str) -> np.ndarray:
    """Solve a sparse parity system with an explicit, reproducible LU variant."""

    solver = str(linear_solver)
    if solver == "spsolve":
        return np.asarray(sp_la.spsolve(matrix.tocsc(), rhs), dtype=float)
    if solver.startswith("splu"):
        parts = solver.split(":")
        permc_spec = parts[1] if len(parts) > 1 and parts[1] else "COLAMD"
        diag_pivot_thresh = 1.0
        if len(parts) > 2 and parts[2]:
            diag_pivot_thresh = float(parts[2])
        lu = sp_la.splu(matrix.tocsc(), permc_spec=permc_spec, diag_pivot_thresh=diag_pivot_thresh)
        return np.asarray(lu.solve(np.asarray(rhs, dtype=float)), dtype=float)
    if solver == "kratos_sparse_lu":
        return _solve_with_kratos_sparse_lu(matrix, rhs)
    raise ValueError(
        "linear_solver must be 'spsolve', 'kratos_sparse_lu', or "
        "'splu:<permc_spec>[:diag_pivot_thresh]', for example 'splu:NATURAL' "
        "or 'splu:COLAMD:1.0'."
    )


def _solve_with_kratos_sparse_lu(matrix, rhs: np.ndarray) -> np.ndarray:
    """Solve a reduced sparse system with Kratos LinearSolversApplication SparseLU."""

    from scipy.io import mmwrite

    import KratosMultiphysics as KM
    import KratosMultiphysics.LinearSolversApplication as LSA

    rhs_array = np.asarray(rhs, dtype=float)
    with tempfile.NamedTemporaryFile(prefix="pycutfem_kratos_sparse_lu_", suffix=".mtx") as handle:
        mmwrite(handle.name, matrix.tocsr())
        kratos_matrix = KM.CompressedMatrix()
        if not KM.ReadMatrixMarketMatrix(handle.name, kratos_matrix):
            raise RuntimeError("Kratos failed to read the temporary MatrixMarket matrix.")
    x = KM.Vector(rhs_array.size)
    b = KM.Vector(rhs_array.size)
    for i, value in enumerate(rhs_array):
        x[i] = 0.0
        b[i] = float(value)
    if not LSA.SparseLUSolver().Solve(kratos_matrix, x, b):
        raise RuntimeError("Kratos SparseLU failed to solve the reduced system.")
    return np.asarray([float(x[i]) for i in range(rhs_array.size)], dtype=float)
