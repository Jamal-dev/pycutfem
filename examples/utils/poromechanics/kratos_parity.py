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
from pycutfem.nonmatching.interface import NonMatchingInterface
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
from .interface import (
    PairedUPlInterface2D,
    UPlInterfaceMaterial2D,
    assemble_paired_upl_interface_2d,
    build_paired_upl_interface_kratos_newton_batch_2d,
    build_paired_upl_interface_local_batch_2d,
    build_upl_interface_ufl_system_2d,
    build_upl_link_interface_ufl_system_2d,
    paired_upl_interface_to_nonmatching_interface_2d,
)
from .kratos_io import (
    field_values_at_mdpa_nodes,
    load_kratos_json,
    mdpa_volume_mesh_to_pycutfem,
    parse_kratos_mdpa,
)
from .upl import (
    build_kratos_fic_triangle_upl_system_2d,
    build_kratos_quasistatic_upl_system_2d,
    kratos_fic_triangle_element_length_squared,
)


@dataclass(frozen=True)
class KratosConsolidation2DResult:
    times: list[float]
    liquid_pressure_by_node: dict[int, list[float]]
    backend: str


@dataclass(frozen=True)
class KratosFluidPumping2DResult:
    times: list[float]
    liquid_pressure_by_node: dict[int, list[float]]
    displacement_x_by_node: dict[int, list[float]]
    displacement_y_by_node: dict[int, list[float]]
    backend: str
    n_steps: int


KRATOS_PORO_TESTS_ROOT = Path("/tmp/kratos-poro/applications/PoromechanicsApplication/tests")
KRATOS_EXAMPLES_PORO_ROOT = Path("/tmp/kratos-examples-poro/poromechanics")
FLUID_PUMPING_2D_SAMPLE_NODE_IDS = (8677, 8675, 1201, 3686, 820, 8545)


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

    dt = Constant(0.5)
    theta = Constant(1.0)
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
        dt=Constant(dt_value),
        theta_u=Constant(theta_u),
        theta_p=Constant(theta_p),
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
        dt=Constant(dt_value),
        theta_u=Constant(theta_value),
        theta_p=Constant(theta_value),
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


def solve_kratos_fluid_pumping_2d_pycutfem(
    *,
    backend: str = "cpp",
    root: str | Path | None = None,
    end_time: float | None = 1.0e-5,
    output_node_ids: tuple[int, ...] = FLUID_PUMPING_2D_SAMPLE_NODE_IDS,
    interface_update: str = "kratos_lagged",
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
    velocity_prev = VectorFunction(name="velocity_prev", field_names=["ux", "uy"], dof_handler=dh)
    p_rate_prev = Function(name="p_rate_prev", field_name="p", dof_handler=dh)
    for coeff in (u_prev, u_current, p_prev, velocity_prev, p_rate_prev):
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
        dt=Constant(dt_value),
        theta_u=Constant(theta_u),
        theta_p=Constant(theta_p),
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

    times: list[float] = []
    pressure_by_node: dict[int, list[float]] = {int(node_id): [] for node_id in output_node_ids}
    ux_by_node: dict[int, list[float]] = {int(node_id): [] for node_id in output_node_ids}
    uy_by_node: dict[int, list[float]] = {int(node_id): [] for node_id in output_node_ids}
    w_prev = np.zeros(dh.total_dofs, dtype=float)
    velocity_prev_values = np.zeros_like(velocity_prev.nodal_values, dtype=float)
    p_rate_prev_values = np.zeros_like(p_rate_prev.nodal_values, dtype=float)
    fracture_state = _initial_interface_state(fracture_interfaces, interface_material)
    link_state = _initial_interface_state(link_interfaces, link_material)

    if interface_update not in {"kratos_newton", "fixed_point", "kratos_lagged"}:
        raise ValueError("interface_update must be 'kratos_newton', 'fixed_point', or 'kratos_lagged'.")

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

        max_interface_iterations = 1 if interface_update == "kratos_lagged" else 15
        fracture_state_next = fracture_state
        link_state_next = link_state
        for _iteration in range(max_interface_iterations):
            u_current.nodal_values = guess[u_current._g_dofs]
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
            velocity_global = np.zeros(dh.total_dofs, dtype=float)
            dt_pressure_global = np.zeros(dh.total_dofs, dtype=float)
            velocity_global[velocity_prev._g_dofs] = velocity_guess
            dt_pressure_global[p_rate_prev._g_dofs] = p_rate_guess
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
                state_variables=fracture_state,
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
                state_variables=link_state,
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
            K_bc, F_bc = _apply_dirichlet_zero(K, F, dirichlet_dofs)
            delta = np.asarray(sp_la.spsolve(K_bc.tocsc(), F_bc), dtype=float)
            w_new = guess + delta
            if interface_update == "kratos_lagged":
                break
            if np.linalg.norm(delta, ord=np.inf) <= 1.0e-10 * max(1.0, np.linalg.norm(w_new, ord=np.inf)):
                break
            guess = w_new
        else:
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


def _patch_fluid_pumping_mdpa_for_installed_kratos(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = {
        "BULK_MODULUS_FLUID": "BULK_MODULUS_LIQUID",
        "DENSITY_WATER": "DENSITY_LIQUID",
        "DYNAMIC_VISCOSITY": "DYNAMIC_VISCOSITY_LIQUID",
        "UPwSmallStrainFICElement2D3N": "UPlSmallStrainFICElement2D3N",
        "UPwSmallStrainInterfaceElement2D4N": "UPlSmallStrainInterfaceElement2D4N",
        "UPwSmallStrainLinkInterfaceElement2D4N": "UPlSmallStrainLinkInterfaceElement2D4N",
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
                new_node_id = int(old_to_new[int(old_node_id)])
                for component, field in enumerate(field_names):
                    if component < active.size and bool(active[component]):
                        dof = dh.dof_map[field].get(new_node_id)
                        if dof is not None:
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
