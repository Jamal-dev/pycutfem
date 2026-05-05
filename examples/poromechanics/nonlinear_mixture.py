#!/usr/bin/env python3
"""Validation examples for the nonlinear Eulerian poromechanics mixture model.

Run all validations with:

    conda run --no-capture-output -n fenicsx \
      python examples/poromechanics/nonlinear_mixture.py --case all --backend cpp
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Callable, Iterable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.utils.poromechanics import (  # noqa: E402
    ExponentialFluidEOS,
    NonlinearPoromechanicsMaterial2D,
    build_nonlinear_poromechanics_full_dynamic_theta_system_2d,
    build_nonlinear_poromechanics_reduced_theta_system_2d,
)
from pycutfem.core.dofhandler import DofHandler  # noqa: E402
from pycutfem.core.mesh import Mesh  # noqa: E402
from pycutfem.fem.mixedelement import MixedElement  # noqa: E402
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters  # noqa: E402
from pycutfem.ufl.analytic import Analytic  # noqa: E402
from pycutfem.ufl.expressions import (  # noqa: E402
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
)
from pycutfem.ufl.forms import BoundaryCondition  # noqa: E402
from pycutfem.ufl.measures import dS, dx  # noqa: E402
from pycutfem.ufl.spaces import FunctionSpace  # noqa: E402
from pycutfem.utils.meshgen import structured_quad  # noqa: E402


NONLINEAR_CONSOLIDATION_2D_REFERENCE: dict[str, float | str] = {
    "backend": "cpp",
    "nx": 1.0,
    "ny": 2.0,
    "dt": 0.02,
    "theta": 1.0,
    "n_steps": 1.0,
    "top_load": 500.0,
    "max_pressure": 1023.7267643643008,
    "min_pressure": 0.0,
    "top_vertical_displacement": -0.002950510936437343,
}


@dataclass(frozen=True)
class NonlinearConsolidation2DResult:
    """Time history for the nonlinear column consolidation example."""

    times: list[float]
    max_pressure: list[float]
    min_pressure: list[float]
    top_vertical_displacement: list[float]
    backend: str
    drained_top: bool


@dataclass(frozen=True)
class NonlinearMMSConvergenceResult:
    """MMS convergence rows and observed final rates."""

    rows: list[dict[str, float]]
    displacement_rate: float
    pressure_rate: float


@dataclass(frozen=True)
class NonlinearSteadyValidationResult:
    """Transient-to-steady validation metrics."""

    response: NonlinearConsolidation2DResult
    pressure_decay_ratio: float
    displacement_increment_ratio: float


@dataclass(frozen=True)
class NonlinearBenchmarkComparison:
    """Drained and undrained benchmark comparison metrics."""

    drained: NonlinearConsolidation2DResult
    undrained: NonlinearConsolidation2DResult
    pressure_ratio_undrained_to_drained: float
    settlement_ratio_undrained_to_drained: float


@dataclass(frozen=True)
class FullDynamicConsolidation2DResult:
    """Time history for the full two-velocity nonlinear column example."""

    times: list[float]
    max_pressure: list[float]
    min_pressure: list[float]
    top_vertical_displacement: list[float]
    mean_porosity: list[float]
    min_porosity: list[float]
    max_porosity: list[float]
    max_solid_speed: list[float]
    max_fluid_speed: list[float]
    backend: str
    drained_top: bool


@dataclass(frozen=True)
class FullDynamicMMSConvergenceResult:
    """Full dynamic MMS convergence rows and final observed rates."""

    rows: list[dict[str, float]]
    displacement_rate: float
    solid_velocity_rate: float
    fluid_velocity_rate: float
    pressure_rate: float
    porosity_rate: float


@dataclass(frozen=True)
class FullDynamicSteadyValidationResult:
    """Full dynamic transient-to-steady validation metrics."""

    response: FullDynamicConsolidation2DResult
    pressure_decay_ratio: float
    solid_speed_decay_ratio: float
    fluid_speed_decay_ratio: float


@dataclass(frozen=True)
class FullDynamicBenchmarkComparison:
    """Full dynamic drained and undrained benchmark comparison metrics."""

    drained: FullDynamicConsolidation2DResult
    undrained: FullDynamicConsolidation2DResult
    pressure_ratio_undrained_to_drained: float
    settlement_ratio_undrained_to_drained: float


@dataclass(frozen=True)
class FullDynamicTerzaghiBenchmarkResult:
    """External Terzaghi one-dimensional consolidation benchmark metrics."""

    times: list[float]
    numerical_degree: list[float]
    reference_degree: list[float]
    degree_errors: list[float]
    max_degree_error: float
    rms_degree_error: float
    constrained_modulus: float
    consolidation_coefficient: float
    initial_settlement: float
    final_settlement: float
    reference: str


def _named_constant(value, name: str, *, dim: int | None = None, preserve: bool = True):
    c = Constant(value, dim=dim) if dim is not None else Constant(value)
    c._jit_name = str(name)
    if preserve:
        c._preserve_runtime_structure = True
    return c


def _eoc(h0: float, h1: float, e0: float, e1: float) -> float:
    if not (h0 > 0.0 and h1 > 0.0 and e0 > 0.0 and e1 > 0.0):
        return float("nan")
    return float(math.log(e0 / e1) / math.log(h0 / h1))


def terzaghi_single_drainage_degree(time_factor: float, *, n_terms: int = 400) -> float:
    """Average consolidation degree for Terzaghi single-drainage consolidation."""

    tv = max(float(time_factor), 0.0)
    total = 0.0
    for term in range(int(n_terms)):
        mode = 2 * term + 1
        total += 8.0 / (math.pi**2 * mode**2) * math.exp(-(mode**2 * math.pi**2 / 4.0) * tv)
    return float(1.0 - total)


def build_unit_square_column_mesh(
    *,
    nx: int = 2,
    ny: int = 4,
    poly_order: int = 2,
) -> Mesh:
    """Build a tagged unit-square column mesh for example drivers."""

    nodes, elems, edges, corners = structured_quad(
        1.0,
        1.0,
        nx=int(nx),
        ny=int(ny),
        poly_order=int(poly_order),
        offset=(0.0, 0.0),
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=int(poly_order),
    )
    tol = 1.0e-12
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - 1.0) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - 1.0) <= tol,
            "boundary": lambda x, y: (abs(x - 0.0) <= tol)
            or (abs(x - 1.0) <= tol)
            or (abs(y - 0.0) <= tol)
            or (abs(y - 1.0) <= tol),
        }
    )
    return mesh


def _default_material(
    *,
    phi0: float = 0.3,
    permeability: float = 2.0e-6,
    fluid_compressibility: float = 1.0e-5,
    fluid_density_ref: float = 1.0e3,
    density_solid: float = 2.0e3,
    dynamic_viscosity_fluid: float = 1.0,
    skeleton_mu: float = 4.0e4,
    skeleton_lambda: float = 6.0e4,
) -> NonlinearPoromechanicsMaterial2D:
    return NonlinearPoromechanicsMaterial2D(
        phi0=float(phi0),
        density_solid=float(density_solid),
        fluid_eos=ExponentialFluidEOS(density_ref=float(fluid_density_ref), compressibility=float(fluid_compressibility)),
        dynamic_viscosity_fluid=float(dynamic_viscosity_fluid),
        permeability_ref=((float(permeability), 0.0), (0.0, float(permeability))),
        skeleton_mu=float(skeleton_mu),
        skeleton_lambda=float(skeleton_lambda),
    )


def _build_reduced_space(mesh: Mesh) -> tuple[MixedElement, DofHandler]:
    mixed_element = MixedElement(mesh, field_specs={"ux": 2, "uy": 2, "p": 1})
    return mixed_element, DofHandler(mixed_element, method="cg")


def _build_reduced_functions(dh: DofHandler):
    disp_space = FunctionSpace("solid_displacement", ["ux", "uy"], dim=1)
    du = VectorTrialFunction(disp_space, dof_handler=dh)
    eta = VectorTestFunction(disp_space, dof_handler=dh)
    dp = TrialFunction(name="dp", field_name="p", dof_handler=dh)
    w = TestFunction(name="w", field_name="p", dof_handler=dh)

    u = VectorFunction(name="u", field_names=["ux", "uy"], dof_handler=dh)
    p = Function(name="p", field_name="p", dof_handler=dh)
    u_prev = VectorFunction(name="u_prev", field_names=["ux", "uy"], dof_handler=dh)
    p_prev = Function(name="p_prev", field_name="p", dof_handler=dh)
    u_prevprev = VectorFunction(name="u_prevprev", field_names=["ux", "uy"], dof_handler=dh)
    return du, eta, dp, w, u, p, u_prev, p_prev, u_prevprev


def _build_full_dynamic_space(mesh: Mesh) -> tuple[MixedElement, DofHandler]:
    mixed_element = MixedElement(
        mesh,
        field_specs={"ux": 2, "uy": 2, "vsx": 2, "vsy": 2, "vfx": 2, "vfy": 2, "p": 1, "phi": 1},
    )
    return mixed_element, DofHandler(mixed_element, method="cg")


def _build_full_dynamic_functions(dh: DofHandler):
    disp_space = FunctionSpace("solid_displacement", ["ux", "uy"], dim=1)
    solid_velocity_space = FunctionSpace("solid_velocity", ["vsx", "vsy"], dim=1)
    fluid_velocity_space = FunctionSpace("fluid_velocity", ["vfx", "vfy"], dim=1)

    du = VectorTrialFunction(disp_space, dof_handler=dh)
    dvs = VectorTrialFunction(solid_velocity_space, dof_handler=dh)
    dvf = VectorTrialFunction(fluid_velocity_space, dof_handler=dh)
    dp = TrialFunction(name="dp", field_name="p", dof_handler=dh)
    dphi = TrialFunction(name="dphi", field_name="phi", dof_handler=dh)

    eta_u = VectorTestFunction(disp_space, dof_handler=dh)
    eta_s = VectorTestFunction(solid_velocity_space, dof_handler=dh)
    eta_f = VectorTestFunction(fluid_velocity_space, dof_handler=dh)
    w_p = TestFunction(name="w_p", field_name="p", dof_handler=dh)
    w_phi = TestFunction(name="w_phi", field_name="phi", dof_handler=dh)

    u = VectorFunction(name="u", field_names=["ux", "uy"], dof_handler=dh)
    vs = VectorFunction(name="vs", field_names=["vsx", "vsy"], dof_handler=dh)
    vf = VectorFunction(name="vf", field_names=["vfx", "vfy"], dof_handler=dh)
    p = Function(name="p", field_name="p", dof_handler=dh)
    phi = Function(name="phi", field_name="phi", dof_handler=dh)
    u_prev = VectorFunction(name="u_prev", field_names=["ux", "uy"], dof_handler=dh)
    vs_prev = VectorFunction(name="vs_prev", field_names=["vsx", "vsy"], dof_handler=dh)
    vf_prev = VectorFunction(name="vf_prev", field_names=["vfx", "vfy"], dof_handler=dh)
    p_prev = Function(name="p_prev", field_name="p", dof_handler=dh)
    phi_prev = Function(name="phi_prev", field_name="phi", dof_handler=dh)
    return (
        du,
        dvs,
        dvf,
        dp,
        dphi,
        eta_u,
        eta_s,
        eta_f,
        w_p,
        w_phi,
        u,
        vs,
        vf,
        p,
        phi,
        u_prev,
        vs_prev,
        vf_prev,
        p_prev,
        phi_prev,
    )


def solve_nonlinear_consolidation_2d(
    *,
    backend: str = "cpp",
    nx: int = 2,
    ny: int = 4,
    dt: float = 0.05,
    theta: float = 1.0,
    n_steps: int = 1,
    top_load: float = 1.0e3,
    drained_top: bool = True,
    permeability: float = 2.0e-6,
    fluid_compressibility: float = 1.0e-5,
) -> NonlinearConsolidation2DResult:
    """Run the nonlinear 2D consolidation column example."""

    mesh = build_unit_square_column_mesh(nx=nx, ny=ny, poly_order=2)
    mixed_element, dh = _build_reduced_space(mesh)
    du, eta, dp, w, u, p, u_prev, p_prev, u_prevprev = _build_reduced_functions(dh)
    for fn in (u, p, u_prev, p_prev, u_prevprev):
        fn.nodal_values.fill(0.0)

    material = _default_material(permeability=permeability, fluid_compressibility=fluid_compressibility)
    dt_c = _named_constant(float(dt), "nlp_dt")
    theta_c = _named_constant(float(theta), "nlp_theta")
    top_traction = _named_constant(np.asarray([0.0, -float(top_load)], dtype=float), "nlp_top_traction", dim=1)

    system = build_nonlinear_poromechanics_reduced_theta_system_2d(
        u_trial=du,
        p_trial=dp,
        u_test=eta,
        p_test=w,
        u_current=u,
        p_current=p,
        u_prev=u_prev,
        p_prev=p_prev,
        u_prevprev=u_prevprev,
        material=material,
        dt=dt_c,
        theta=theta_c,
        dx_measure=dx(metadata={"q": 6}),
        traction=top_traction,
        traction_measure=dS(mesh.edge_bitset("top"), metadata={"q": 6}),
    )

    bcs = [
        BoundaryCondition("uy", "dirichlet", "bottom", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "left", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "right", lambda x, y: 0.0),
    ]
    if bool(drained_top):
        bcs.append(BoundaryCondition("p", "dirichlet", "top", lambda x, y: 0.0))
    bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]

    solver = NewtonSolver(
        system.residual_form,
        system.jacobian_form,
        dof_handler=dh,
        mixed_element=mixed_element,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(
            newton_tol=1.0e-8,
            max_newton_iter=25,
            line_search=True,
            print_level=0,
        ),
        quad_order=6,
        backend=backend,
    )

    times: list[float] = []
    max_pressure: list[float] = []
    min_pressure: list[float] = []
    top_vertical_displacement: list[float] = []

    p_slice = np.asarray(dh.get_field_slice("p"), dtype=int)
    uy_slice = np.asarray(dh.get_field_slice("uy"), dtype=int)
    uy_coords = np.asarray(dh.get_field_dof_coords("uy"), dtype=float)
    top_uy = np.where(np.isclose(uy_coords[:, 1], 1.0))[0]

    prevprev_values = u_prevprev.nodal_values.copy()
    for step in range(1, int(n_steps) + 1):
        old_prev_values = u_prev.nodal_values.copy()
        u_prevprev.nodal_values[:] = prevprev_values
        solver.solve_time_interval(
            functions=[u, p],
            prev_functions=[u_prev, p_prev],
            aux_functions={"dt": dt_c, "u_prevprev": u_prevprev},
            time_params=TimeStepperParameters(
                dt=float(dt),
                final_time=float(dt),
                max_steps=1,
                theta=float(theta),
                stop_on_steady=False,
            ),
        )
        p_vals = np.asarray(p.get_nodal_values(p_slice), dtype=float)
        uy_vals = np.asarray(u.get_nodal_values(uy_slice), dtype=float)
        times.append(float(step) * float(dt))
        max_pressure.append(float(np.max(p_vals)))
        min_pressure.append(float(np.min(p_vals)))
        top_vertical_displacement.append(float(np.mean(uy_vals[top_uy])))
        prevprev_values = old_prev_values

    return NonlinearConsolidation2DResult(
        times=times,
        max_pressure=max_pressure,
        min_pressure=min_pressure,
        top_vertical_displacement=top_vertical_displacement,
        backend=str(backend),
        drained_top=bool(drained_top),
    )


def solve_full_dynamic_consolidation_2d(
    *,
    backend: str = "cpp",
    nx: int = 1,
    ny: int = 2,
    dt: float = 0.01,
    theta: float = 1.0,
    n_steps: int = 1,
    top_load: float = 5.0e2,
    drained_top: bool = True,
    permeability: float = 2.0e-6,
    fluid_compressibility: float = 1.0e-5,
    phi0: float = 0.3,
    fluid_density_ref: float = 1.0e3,
    density_solid: float = 2.0e3,
    dynamic_viscosity_fluid: float = 1.0,
    skeleton_mu: float = 4.0e4,
    skeleton_lambda: float = 6.0e4,
) -> FullDynamicConsolidation2DResult:
    """Run the full two-velocity dynamic nonlinear consolidation column."""

    mesh = build_unit_square_column_mesh(nx=nx, ny=ny, poly_order=2)
    mixed_element, dh = _build_full_dynamic_space(mesh)
    (
        du,
        dvs,
        dvf,
        dp,
        dphi,
        eta_u,
        eta_s,
        eta_f,
        w_p,
        w_phi,
        u,
        vs,
        vf,
        p,
        phi,
        u_prev,
        vs_prev,
        vf_prev,
        p_prev,
        phi_prev,
    ) = _build_full_dynamic_functions(dh)

    material = _default_material(
        phi0=phi0,
        permeability=permeability,
        fluid_compressibility=fluid_compressibility,
        fluid_density_ref=fluid_density_ref,
        density_solid=density_solid,
        dynamic_viscosity_fluid=dynamic_viscosity_fluid,
        skeleton_mu=skeleton_mu,
        skeleton_lambda=skeleton_lambda,
    )
    for fn in (u, vs, vf, p, u_prev, vs_prev, vf_prev, p_prev):
        fn.nodal_values.fill(0.0)
    phi.nodal_values.fill(float(material.phi0))
    phi_prev.nodal_values.fill(float(material.phi0))

    dt_c = _named_constant(float(dt), "nlp_dt")
    theta_c = _named_constant(float(theta), "nlp_theta")
    top_traction = _named_constant(np.asarray([0.0, -float(top_load)], dtype=float), "nlp_full_top_traction", dim=1)
    system = build_nonlinear_poromechanics_full_dynamic_theta_system_2d(
        u_trial=du,
        vs_trial=dvs,
        vf_trial=dvf,
        p_trial=dp,
        phi_trial=dphi,
        u_test=eta_u,
        vs_test=eta_s,
        vf_test=eta_f,
        p_test=w_p,
        phi_test=w_phi,
        u_current=u,
        vs_current=vs,
        vf_current=vf,
        p_current=p,
        phi_current=phi,
        u_prev=u_prev,
        vs_prev=vs_prev,
        vf_prev=vf_prev,
        p_prev=p_prev,
        phi_prev=phi_prev,
        material=material,
        dt=dt_c,
        theta=theta_c,
        dx_measure=dx(metadata={"q": 5}),
        solid_traction=top_traction,
        solid_traction_measure=dS(mesh.edge_bitset("top"), metadata={"q": 5}),
    )

    bcs = [
        BoundaryCondition("uy", "dirichlet", "bottom", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "left", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "right", lambda x, y: 0.0),
        BoundaryCondition("vsy", "dirichlet", "bottom", lambda x, y: 0.0),
        BoundaryCondition("vsx", "dirichlet", "left", lambda x, y: 0.0),
        BoundaryCondition("vsx", "dirichlet", "right", lambda x, y: 0.0),
        BoundaryCondition("vfy", "dirichlet", "bottom", lambda x, y: 0.0),
        BoundaryCondition("vfx", "dirichlet", "left", lambda x, y: 0.0),
        BoundaryCondition("vfx", "dirichlet", "right", lambda x, y: 0.0),
    ]
    if bool(drained_top):
        bcs.append(BoundaryCondition("p", "dirichlet", "top", lambda x, y: 0.0))
    else:
        bcs.append(BoundaryCondition("vfy", "dirichlet", "top", lambda x, y: 0.0))
    bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]

    solver = NewtonSolver(
        system.residual_form,
        system.jacobian_form,
        dof_handler=dh,
        mixed_element=mixed_element,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=1.0e-8, max_newton_iter=30, line_search=True, print_level=0),
        quad_order=5,
        backend=backend,
    )

    times: list[float] = []
    max_pressure: list[float] = []
    min_pressure: list[float] = []
    top_vertical_displacement: list[float] = []
    mean_porosity: list[float] = []
    min_porosity: list[float] = []
    max_porosity: list[float] = []
    max_solid_speed: list[float] = []
    max_fluid_speed: list[float] = []

    p_slice = np.asarray(dh.get_field_slice("p"), dtype=int)
    phi_slice = np.asarray(dh.get_field_slice("phi"), dtype=int)
    uy_slice = np.asarray(dh.get_field_slice("uy"), dtype=int)
    vsx_slice = np.asarray(dh.get_field_slice("vsx"), dtype=int)
    vsy_slice = np.asarray(dh.get_field_slice("vsy"), dtype=int)
    vfx_slice = np.asarray(dh.get_field_slice("vfx"), dtype=int)
    vfy_slice = np.asarray(dh.get_field_slice("vfy"), dtype=int)
    uy_coords = np.asarray(dh.get_field_dof_coords("uy"), dtype=float)
    top_uy = np.where(np.isclose(uy_coords[:, 1], 1.0))[0]

    for step in range(1, int(n_steps) + 1):
        solver.solve_time_interval(
            functions=[u, vs, vf, p, phi],
            prev_functions=[u_prev, vs_prev, vf_prev, p_prev, phi_prev],
            aux_functions={"dt": dt_c},
            time_params=TimeStepperParameters(
                dt=float(dt),
                final_time=float(dt),
                max_steps=1,
                theta=float(theta),
                stop_on_steady=False,
            ),
        )
        p_vals = np.asarray(p.get_nodal_values(p_slice), dtype=float)
        phi_vals = np.asarray(phi.get_nodal_values(phi_slice), dtype=float)
        uy_vals = np.asarray(u.get_nodal_values(uy_slice), dtype=float)
        vs_speed = np.hypot(
            np.asarray(vs.get_nodal_values(vsx_slice), dtype=float),
            np.asarray(vs.get_nodal_values(vsy_slice), dtype=float),
        )
        vf_speed = np.hypot(
            np.asarray(vf.get_nodal_values(vfx_slice), dtype=float),
            np.asarray(vf.get_nodal_values(vfy_slice), dtype=float),
        )
        times.append(float(step) * float(dt))
        max_pressure.append(float(np.max(p_vals)))
        min_pressure.append(float(np.min(p_vals)))
        top_vertical_displacement.append(float(np.mean(uy_vals[top_uy])))
        mean_porosity.append(float(np.mean(phi_vals)))
        min_porosity.append(float(np.min(phi_vals)))
        max_porosity.append(float(np.max(phi_vals)))
        max_solid_speed.append(float(np.max(vs_speed)))
        max_fluid_speed.append(float(np.max(vf_speed)))

    return FullDynamicConsolidation2DResult(
        times=times,
        max_pressure=max_pressure,
        min_pressure=min_pressure,
        top_vertical_displacement=top_vertical_displacement,
        mean_porosity=mean_porosity,
        min_porosity=min_porosity,
        max_porosity=max_porosity,
        max_solid_speed=max_solid_speed,
        max_fluid_speed=max_fluid_speed,
        backend=str(backend),
        drained_top=bool(drained_top),
    )


def _mms_exact_u(x: float, y: float, t: float) -> np.ndarray:
    amp = 1.0e-2
    tau = 1.0 + 0.35 * float(t)
    s = math.sin(math.pi * float(x)) * math.sin(math.pi * float(y))
    return amp * tau * np.asarray([s, -0.65 * s], dtype=float)


def _mms_grad_u(x: float, y: float, t: float) -> np.ndarray:
    amp = 1.0e-2
    tau = 1.0 + 0.35 * float(t)
    sx = math.sin(math.pi * float(x))
    sy = math.sin(math.pi * float(y))
    cx = math.cos(math.pi * float(x))
    cy = math.cos(math.pi * float(y))
    base_x = amp * tau * math.pi * cx * sy
    base_y = amp * tau * math.pi * sx * cy
    return np.asarray([[base_x, base_y], [-0.65 * base_x, -0.65 * base_y]], dtype=float)


def _mms_exact_p(x: float, y: float, t: float) -> float:
    amp = 40.0
    tau = 1.0 + 0.25 * float(t)
    return float(amp * tau * math.sin(math.pi * float(x)) * math.sin(math.pi * float(y)))


def _mms_grad_p(x: float, y: float, t: float) -> np.ndarray:
    amp = 40.0
    tau = 1.0 + 0.25 * float(t)
    return amp * tau * math.pi * np.asarray(
        [
            math.cos(math.pi * float(x)) * math.sin(math.pi * float(y)),
            math.sin(math.pi * float(x)) * math.cos(math.pi * float(y)),
        ],
        dtype=float,
    )


def _mms_phi_and_stress(x: float, y: float, t: float, material: NonlinearPoromechanicsMaterial2D):
    grad_u = _mms_grad_u(x, y, t)
    identity = np.eye(2, dtype=float)
    F = np.linalg.inv(identity - grad_u)
    J = float(np.linalg.det(F))
    phi = 1.0 - (1.0 - float(material.phi0)) / J
    B = F @ F.T
    sigma = (float(material.skeleton_mu) / J) * (B - identity)
    sigma += float(material.skeleton_lambda) * (J - 1.0) * identity
    return phi, sigma, F, J


def _mms_q(x: float, y: float, t: float, material: NonlinearPoromechanicsMaterial2D) -> np.ndarray:
    _, _, F, J = _mms_phi_and_stress(x, y, t, material)
    mobility = (F @ material.permeability_ref_array @ F.T) / (J * float(material.dynamic_viscosity_fluid))
    return -mobility @ _mms_grad_p(x, y, t)


def _mms_total_stress_theta(
    x: float,
    y: float,
    *,
    material: NonlinearPoromechanicsMaterial2D,
    dt: float,
    theta: float,
) -> np.ndarray:
    t_k = float(dt)
    t_n = 0.0
    phi_k, sigma_k, _, _ = _mms_phi_and_stress(x, y, t_k, material)
    phi_n, sigma_n, _, _ = _mms_phi_and_stress(x, y, t_n, material)
    phi_theta = float(theta) * phi_k + (1.0 - float(theta)) * phi_n
    sigma_theta = float(theta) * sigma_k + (1.0 - float(theta)) * sigma_n
    p_theta = float(theta) * _mms_exact_p(x, y, t_k) + (1.0 - float(theta)) * _mms_exact_p(x, y, t_n)
    return (1.0 - phi_theta) * sigma_theta - phi_theta * p_theta * np.eye(2, dtype=float)


def _mms_body_force(
    x: float,
    y: float,
    *,
    material: NonlinearPoromechanicsMaterial2D,
    dt: float,
    theta: float,
) -> np.ndarray:
    eps = 1.0e-6

    def stress(xx: float, yy: float) -> np.ndarray:
        return _mms_total_stress_theta(xx, yy, material=material, dt=dt, theta=theta)

    dsdx = (stress(x + eps, y) - stress(x - eps, y)) / (2.0 * eps)
    dsdy = (stress(x, y + eps) - stress(x, y - eps)) / (2.0 * eps)
    div_sigma = np.asarray([dsdx[0, 0] + dsdy[0, 1], dsdx[1, 0] + dsdy[1, 1]], dtype=float)
    return -div_sigma


def _mms_pressure_source(
    x: float,
    y: float,
    *,
    material: NonlinearPoromechanicsMaterial2D,
    dt: float,
    theta: float,
) -> float:
    t_k = float(dt)
    t_n = 0.0
    t_nm1 = -float(dt)
    cf = float(material.fluid_eos.compressibility)
    theta_f = float(theta)

    phi_k, _, _, _ = _mms_phi_and_stress(x, y, t_k, material)
    phi_n, _, _, _ = _mms_phi_and_stress(x, y, t_n, material)
    phi_theta = theta_f * phi_k + (1.0 - theta_f) * phi_n

    p_dot = (_mms_exact_p(x, y, t_k) - _mms_exact_p(x, y, t_n)) / float(dt)
    vs_k = (_mms_exact_u(x, y, t_k) - _mms_exact_u(x, y, t_n)) / float(dt)
    vs_n = (_mms_exact_u(x, y, t_n) - _mms_exact_u(x, y, t_nm1)) / float(dt)
    adv_k = float(vs_k @ _mms_grad_p(x, y, t_k))
    adv_n = float(vs_n @ _mms_grad_p(x, y, t_n))
    material_p_dot = p_dot + theta_f * adv_k + (1.0 - theta_f) * adv_n

    div_vs_k = float(np.trace((_mms_grad_u(x, y, t_k) - _mms_grad_u(x, y, t_n)) / float(dt)))
    div_vs_n = float(np.trace((_mms_grad_u(x, y, t_n) - _mms_grad_u(x, y, t_nm1)) / float(dt)))
    div_vs_theta = theta_f * div_vs_k + (1.0 - theta_f) * div_vs_n

    def q_theta(xx: float, yy: float) -> np.ndarray:
        return theta_f * _mms_q(xx, yy, t_k, material) + (1.0 - theta_f) * _mms_q(xx, yy, t_n, material)

    qth = q_theta(x, y)
    grad_p_theta = theta_f * _mms_grad_p(x, y, t_k) + (1.0 - theta_f) * _mms_grad_p(x, y, t_n)
    eps = 1.0e-6
    dq0dx = (q_theta(x + eps, y)[0] - q_theta(x - eps, y)[0]) / (2.0 * eps)
    dq1dy = (q_theta(x, y + eps)[1] - q_theta(x, y - eps)[1]) / (2.0 * eps)
    div_q = float(dq0dx + dq1dy)
    return float(phi_theta * cf * material_p_dot + cf * float(qth @ grad_p_theta) + div_vs_theta + div_q)


def _trig_basis(x: float, y: float) -> tuple[float, float, float, float, float]:
    sx = math.sin(math.pi * float(x))
    sy = math.sin(math.pi * float(y))
    cx = math.cos(math.pi * float(x))
    cy = math.cos(math.pi * float(y))
    s = sx * sy
    ds_dx = math.pi * cx * sy
    ds_dy = math.pi * sx * cy
    return s, ds_dx, ds_dy, sx, sy


def _full_mms_vector_field(x: float, y: float, t: float, *, amp: float, growth: float, ratio: float) -> np.ndarray:
    s, _, _, _, _ = _trig_basis(x, y)
    tau = 1.0 + float(growth) * float(t)
    return float(amp) * tau * np.asarray([s, float(ratio) * s], dtype=float)


def _full_mms_grad_vector_field(x: float, y: float, t: float, *, amp: float, growth: float, ratio: float) -> np.ndarray:
    _, ds_dx, ds_dy, _, _ = _trig_basis(x, y)
    tau = 1.0 + float(growth) * float(t)
    scale = float(amp) * tau
    return scale * np.asarray([[ds_dx, ds_dy], [float(ratio) * ds_dx, float(ratio) * ds_dy]], dtype=float)


def _full_mms_u(x: float, y: float, t: float) -> np.ndarray:
    return _full_mms_vector_field(x, y, t, amp=6.0e-3, growth=0.30, ratio=-0.45)


def _full_mms_grad_u(x: float, y: float, t: float) -> np.ndarray:
    return _full_mms_grad_vector_field(x, y, t, amp=6.0e-3, growth=0.30, ratio=-0.45)


def _full_mms_vs(x: float, y: float, t: float) -> np.ndarray:
    return _full_mms_vector_field(x, y, t, amp=4.0e-3, growth=0.20, ratio=-0.30)


def _full_mms_grad_vs(x: float, y: float, t: float) -> np.ndarray:
    return _full_mms_grad_vector_field(x, y, t, amp=4.0e-3, growth=0.20, ratio=-0.30)


def _full_mms_vf(x: float, y: float, t: float) -> np.ndarray:
    rel = _full_mms_vector_field(x, y, t, amp=2.5e-3, growth=0.15, ratio=0.55)
    return _full_mms_vs(x, y, t) + rel


def _full_mms_grad_vf(x: float, y: float, t: float) -> np.ndarray:
    rel = _full_mms_grad_vector_field(x, y, t, amp=2.5e-3, growth=0.15, ratio=0.55)
    return _full_mms_grad_vs(x, y, t) + rel


def _full_mms_p(x: float, y: float, t: float) -> float:
    s, _, _, _, _ = _trig_basis(x, y)
    return float(20.0 * (1.0 + 0.25 * float(t)) * s)


def _full_mms_grad_p(x: float, y: float, t: float) -> np.ndarray:
    _, ds_dx, ds_dy, _, _ = _trig_basis(x, y)
    return 20.0 * (1.0 + 0.25 * float(t)) * np.asarray([ds_dx, ds_dy], dtype=float)


def _full_mms_phi(x: float, y: float, t: float) -> float:
    s, _, _, _, _ = _trig_basis(x, y)
    return float(0.32 + 0.02 * (1.0 + 0.10 * float(t)) * s)


def _full_mms_grad_phi(x: float, y: float, t: float) -> np.ndarray:
    _, ds_dx, ds_dy, _, _ = _trig_basis(x, y)
    return 0.02 * (1.0 + 0.10 * float(t)) * np.asarray([ds_dx, ds_dy], dtype=float)


def _full_mms_stress_from_grad(grad_u: np.ndarray, material: NonlinearPoromechanicsMaterial2D) -> np.ndarray:
    identity = np.eye(2, dtype=float)
    F_inv = identity - np.asarray(grad_u, dtype=float)
    F = np.linalg.inv(F_inv)
    J = float(np.linalg.det(F))
    B = F @ F.T
    sigma = (float(material.skeleton_mu) / J) * (B - identity)
    sigma += float(material.skeleton_lambda) * (J - 1.0) * identity
    return sigma


def _full_mms_inverse_permeability_from_grad(grad_u: np.ndarray, material: NonlinearPoromechanicsMaterial2D) -> np.ndarray:
    identity = np.eye(2, dtype=float)
    F_inv = identity - np.asarray(grad_u, dtype=float)
    F = np.linalg.inv(F_inv)
    J = float(np.linalg.det(F))
    return J * (F_inv.T @ np.linalg.inv(material.permeability_ref_array) @ F_inv)


def _full_mms_theta_state(
    x: float,
    y: float,
    *,
    material: NonlinearPoromechanicsMaterial2D,
    dt: float,
    theta: float,
) -> dict[str, np.ndarray | float]:
    t_k = float(dt)
    t_n = 0.0
    th = float(theta)
    one_m_th = 1.0 - th

    p_k = _full_mms_p(x, y, t_k)
    p_n = _full_mms_p(x, y, t_n)
    phi_k = _full_mms_phi(x, y, t_k)
    phi_n = _full_mms_phi(x, y, t_n)
    vf_k = _full_mms_vf(x, y, t_k)
    vf_n = _full_mms_vf(x, y, t_n)
    vs_k = _full_mms_vs(x, y, t_k)
    vs_n = _full_mms_vs(x, y, t_n)
    u_k = _full_mms_u(x, y, t_k)
    u_n = _full_mms_u(x, y, t_n)

    grad_p_theta = th * _full_mms_grad_p(x, y, t_k) + one_m_th * _full_mms_grad_p(x, y, t_n)
    grad_phi_theta = th * _full_mms_grad_phi(x, y, t_k) + one_m_th * _full_mms_grad_phi(x, y, t_n)
    grad_vf_theta = th * _full_mms_grad_vf(x, y, t_k) + one_m_th * _full_mms_grad_vf(x, y, t_n)
    grad_vs_theta = th * _full_mms_grad_vs(x, y, t_k) + one_m_th * _full_mms_grad_vs(x, y, t_n)
    grad_u_k = _full_mms_grad_u(x, y, t_k)

    sigma_theta = th * _full_mms_stress_from_grad(grad_u_k, material)
    sigma_theta += one_m_th * _full_mms_stress_from_grad(_full_mms_grad_u(x, y, t_n), material)
    k_inv_theta = th * _full_mms_inverse_permeability_from_grad(grad_u_k, material)
    k_inv_theta += one_m_th * _full_mms_inverse_permeability_from_grad(_full_mms_grad_u(x, y, t_n), material)

    p_theta = th * p_k + one_m_th * p_n
    phi_theta = th * phi_k + one_m_th * phi_n
    vf_theta = th * vf_k + one_m_th * vf_n
    vs_theta = th * vs_k + one_m_th * vs_n
    q_theta = phi_theta * (vf_theta - vs_theta)
    drag_theta = phi_theta * float(material.dynamic_viscosity_fluid) * (k_inv_theta @ q_theta)

    return {
        "p_theta": p_theta,
        "phi_theta": phi_theta,
        "vf_theta": vf_theta,
        "vs_theta": vs_theta,
        "grad_p_theta": grad_p_theta,
        "grad_phi_theta": grad_phi_theta,
        "grad_vf_theta": grad_vf_theta,
        "grad_vs_theta": grad_vs_theta,
        "div_vf_theta": float(np.trace(grad_vf_theta)),
        "div_vs_theta": float(np.trace(grad_vs_theta)),
        "p_dot": (p_k - p_n) / float(dt),
        "phi_dot": (phi_k - phi_n) / float(dt),
        "vf_dot": (vf_k - vf_n) / float(dt),
        "vs_dot": (vs_k - vs_n) / float(dt),
        "u_dot": (u_k - u_n) / float(dt),
        "grad_u_k": grad_u_k,
        "sigma_theta": sigma_theta,
        "q_theta": q_theta,
        "drag_theta": drag_theta,
        "rho_f_theta": material.fluid_eos.rho_value(float(p_theta)),
    }


def _full_mms_sources(
    x: float,
    y: float,
    *,
    material: NonlinearPoromechanicsMaterial2D,
    dt: float,
    theta: float,
) -> dict[str, np.ndarray | float]:
    state = _full_mms_theta_state(x, y, material=material, dt=dt, theta=theta)
    phi_theta = float(state["phi_theta"])
    rho_f_theta = float(state["rho_f_theta"])
    cf = float(material.fluid_eos.compressibility)
    rho_s = float(material.density_solid)
    grad_p_theta = np.asarray(state["grad_p_theta"], dtype=float)
    grad_phi_theta = np.asarray(state["grad_phi_theta"], dtype=float)
    vf_theta = np.asarray(state["vf_theta"], dtype=float)
    vs_theta = np.asarray(state["vs_theta"], dtype=float)
    grad_vf_theta = np.asarray(state["grad_vf_theta"], dtype=float)
    grad_vs_theta = np.asarray(state["grad_vs_theta"], dtype=float)
    vf_dot = np.asarray(state["vf_dot"], dtype=float)
    vs_dot = np.asarray(state["vs_dot"], dtype=float)
    drag_theta = np.asarray(state["drag_theta"], dtype=float)

    d_f_p = float(state["p_dot"]) + float(grad_p_theta @ vf_theta)
    c_f_phi = float(state["phi_dot"]) + float(grad_phi_theta @ vf_theta) + phi_theta * float(state["div_vf_theta"])
    fluid_mass = phi_theta * cf * d_f_p + c_f_phi
    solid_mass = -float(state["phi_dot"]) - float(grad_phi_theta @ vs_theta) + (1.0 - phi_theta) * float(state["div_vs_theta"])

    d_f_vf = vf_dot + grad_vf_theta @ vf_theta
    fluid_momentum = phi_theta * rho_f_theta * d_f_vf + phi_theta * grad_p_theta + drag_theta

    d_s_vs = vs_dot + grad_vs_theta @ vs_theta

    def stress_part(xx: float, yy: float) -> np.ndarray:
        st = _full_mms_theta_state(xx, yy, material=material, dt=dt, theta=theta)
        return (1.0 - float(st["phi_theta"])) * np.asarray(st["sigma_theta"], dtype=float)

    eps = 1.0e-6
    dA_dx = (stress_part(x + eps, y) - stress_part(x - eps, y)) / (2.0 * eps)
    dA_dy = (stress_part(x, y + eps) - stress_part(x, y - eps)) / (2.0 * eps)
    div_stress = np.asarray([dA_dx[0, 0] + dA_dy[0, 1], dA_dx[1, 0] + dA_dy[1, 1]], dtype=float)
    solid_momentum = (1.0 - phi_theta) * rho_s * d_s_vs - div_stress + float(state["p_theta"]) * grad_phi_theta - drag_theta

    kinematic = np.asarray(state["u_dot"], dtype=float) + np.asarray(state["grad_u_k"], dtype=float) @ vs_theta - vs_theta
    return {
        "fluid_mass": float(fluid_mass),
        "solid_mass": float(solid_mass),
        "fluid_momentum": fluid_momentum,
        "solid_momentum": solid_momentum,
        "kinematic": kinematic,
    }


def _analytic_scalar(fn: Callable[[float, float], float], *, degree: int = 8) -> Analytic:
    def wrapped(x, y):
        return np.vectorize(lambda xx, yy: float(fn(float(xx), float(yy))), otypes=[float])(x, y)

    return Analytic(wrapped, degree=int(degree))


def _analytic_vector(fn: Callable[[float, float], np.ndarray], *, degree: int = 8) -> Analytic:
    def wrapped(x, y):
        x_arr, y_arr = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        c0 = np.vectorize(lambda xx, yy: float(fn(float(xx), float(yy))[0]), otypes=[float])(x_arr, y_arr)
        c1 = np.vectorize(lambda xx, yy: float(fn(float(xx), float(yy))[1]), otypes=[float])(x_arr, y_arr)
        return np.stack([c0, c1], axis=-1)

    return Analytic(wrapped, degree=int(degree))


def run_nonlinear_mms_convergence_2d(
    *,
    backend: str = "cpp",
    nx_values: Iterable[int] = (2, 4),
    dt: float = 0.05,
    theta: float = 1.0,
    qdeg: int = 8,
    qerr: int = 8,
) -> NonlinearMMSConvergenceResult:
    """Solve a forced nonlinear MMS and report observed L2 convergence."""

    material = _default_material(permeability=8.0e-7, fluid_compressibility=8.0e-6)
    rows: list[dict[str, float]] = []

    for nx in [int(v) for v in nx_values]:
        mesh = build_unit_square_column_mesh(nx=nx, ny=nx, poly_order=2)
        mixed_element, dh = _build_reduced_space(mesh)
        du, eta, dp, w, u, p, u_prev, p_prev, u_prevprev = _build_reduced_functions(dh)

        u_prev.set_values_from_function(lambda x, y: _mms_exact_u(x, y, 0.0))
        p_prev.set_values_from_function(lambda x, y: _mms_exact_p(x, y, 0.0))
        u_prevprev.set_values_from_function(lambda x, y: _mms_exact_u(x, y, -float(dt)))
        u.set_values_from_function(lambda x, y: _mms_exact_u(x, y, 0.0))
        p.set_values_from_function(lambda x, y: _mms_exact_p(x, y, 0.0))

        dt_c = _named_constant(float(dt), "nlp_dt")
        theta_c = _named_constant(float(theta), "nlp_theta")
        body_force = _analytic_vector(
            lambda x, y: _mms_body_force(x, y, material=material, dt=float(dt), theta=float(theta)),
            degree=int(qdeg),
        )
        pressure_source = _analytic_scalar(
            lambda x, y: _mms_pressure_source(x, y, material=material, dt=float(dt), theta=float(theta)),
            degree=int(qdeg),
        )

        system = build_nonlinear_poromechanics_reduced_theta_system_2d(
            u_trial=du,
            p_trial=dp,
            u_test=eta,
            p_test=w,
            u_current=u,
            p_current=p,
            u_prev=u_prev,
            p_prev=p_prev,
            u_prevprev=u_prevprev,
            material=material,
            dt=dt_c,
            theta=theta_c,
            dx_measure=dx(metadata={"q": int(qdeg)}),
            pressure_source=pressure_source,
            body_force=body_force,
        )

        bcs = []
        for tag in ("left", "right", "bottom", "top"):
            bcs.extend(
                [
                    BoundaryCondition("ux", "dirichlet", tag, lambda x, y, t: float(_mms_exact_u(x, y, t)[0])),
                    BoundaryCondition("uy", "dirichlet", tag, lambda x, y, t: float(_mms_exact_u(x, y, t)[1])),
                    BoundaryCondition("p", "dirichlet", tag, lambda x, y, t: float(_mms_exact_p(x, y, t))),
                ]
            )
        bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]

        solver = NewtonSolver(
            system.residual_form,
            system.jacobian_form,
            dof_handler=dh,
            mixed_element=mixed_element,
            bcs=bcs,
            bcs_homog=bcs_homog,
            newton_params=NewtonParameters(newton_tol=1.0e-8, max_newton_iter=25, line_search=True, print_level=0),
            quad_order=int(qdeg),
            backend=backend,
        )
        solver.solve_time_interval(
            functions=[u, p],
            prev_functions=[u_prev, p_prev],
            aux_functions={"dt": dt_c, "u_prevprev": u_prevprev},
            time_params=TimeStepperParameters(dt=float(dt), final_time=float(dt), max_steps=1, theta=float(theta)),
        )

        err_u = dh.l2_error(
            u,
            exact={
                "ux": lambda x, y: float(_mms_exact_u(x, y, float(dt))[0]),
                "uy": lambda x, y: float(_mms_exact_u(x, y, float(dt))[1]),
            },
            fields=["ux", "uy"],
            quad_order=int(qerr),
            relative=False,
        )
        err_p = dh.l2_error(
            p,
            exact={"p": lambda x, y: float(_mms_exact_p(x, y, float(dt)))},
            fields=["p"],
            quad_order=int(qerr),
            relative=False,
        )
        h = 1.0 / float(nx)
        rows.append({"nx": float(nx), "h": h, "err_u": float(err_u), "err_p": float(err_p)})

    if len(rows) >= 2:
        displacement_rate = _eoc(rows[-2]["h"], rows[-1]["h"], rows[-2]["err_u"], rows[-1]["err_u"])
        pressure_rate = _eoc(rows[-2]["h"], rows[-1]["h"], rows[-2]["err_p"], rows[-1]["err_p"])
    else:
        displacement_rate = float("nan")
        pressure_rate = float("nan")

    return NonlinearMMSConvergenceResult(rows=rows, displacement_rate=displacement_rate, pressure_rate=pressure_rate)


def run_full_dynamic_mms_convergence_2d(
    *,
    backend: str = "cpp",
    nx_values: Iterable[int] = (2, 4),
    dt: float = 0.04,
    theta: float = 1.0,
    qdeg: int = 6,
    qerr: int = 6,
) -> FullDynamicMMSConvergenceResult:
    """Solve a forced MMS for the full two-velocity nonlinear model."""

    material = _default_material(permeability=1.0e-6, fluid_compressibility=7.0e-6)
    rows: list[dict[str, float]] = []

    for nx in [int(v) for v in nx_values]:
        mesh = build_unit_square_column_mesh(nx=nx, ny=nx, poly_order=2)
        mixed_element, dh = _build_full_dynamic_space(mesh)
        (
            du,
            dvs,
            dvf,
            dp,
            dphi,
            eta_u,
            eta_s,
            eta_f,
            w_p,
            w_phi,
            u,
            vs,
            vf,
            p,
            phi,
            u_prev,
            vs_prev,
            vf_prev,
            p_prev,
            phi_prev,
        ) = _build_full_dynamic_functions(dh)

        u_prev.set_values_from_function(lambda x, y: _full_mms_u(x, y, 0.0))
        vs_prev.set_values_from_function(lambda x, y: _full_mms_vs(x, y, 0.0))
        vf_prev.set_values_from_function(lambda x, y: _full_mms_vf(x, y, 0.0))
        p_prev.set_values_from_function(lambda x, y: _full_mms_p(x, y, 0.0))
        phi_prev.set_values_from_function(lambda x, y: _full_mms_phi(x, y, 0.0))
        u.set_values_from_function(lambda x, y: _full_mms_u(x, y, 0.0))
        vs.set_values_from_function(lambda x, y: _full_mms_vs(x, y, 0.0))
        vf.set_values_from_function(lambda x, y: _full_mms_vf(x, y, 0.0))
        p.set_values_from_function(lambda x, y: _full_mms_p(x, y, 0.0))
        phi.set_values_from_function(lambda x, y: _full_mms_phi(x, y, 0.0))

        def source_component(name: str, component: int | None = None):
            if component is None:
                return _analytic_scalar(
                    lambda x, y, key=name: float(
                        _full_mms_sources(x, y, material=material, dt=float(dt), theta=float(theta))[key]
                    ),
                    degree=int(qdeg),
                )
            return _analytic_vector(
                lambda x, y, key=name, comp=component: np.asarray(
                    _full_mms_sources(x, y, material=material, dt=float(dt), theta=float(theta))[key],
                    dtype=float,
                ),
                degree=int(qdeg),
            )

        dt_c = _named_constant(float(dt), "nlp_dt")
        theta_c = _named_constant(float(theta), "nlp_theta")
        system = build_nonlinear_poromechanics_full_dynamic_theta_system_2d(
            u_trial=du,
            vs_trial=dvs,
            vf_trial=dvf,
            p_trial=dp,
            phi_trial=dphi,
            u_test=eta_u,
            vs_test=eta_s,
            vf_test=eta_f,
            p_test=w_p,
            phi_test=w_phi,
            u_current=u,
            vs_current=vs,
            vf_current=vf,
            p_current=p,
            phi_current=phi,
            u_prev=u_prev,
            vs_prev=vs_prev,
            vf_prev=vf_prev,
            p_prev=p_prev,
            phi_prev=phi_prev,
            material=material,
            dt=dt_c,
            theta=theta_c,
            dx_measure=dx(metadata={"q": int(qdeg)}),
            fluid_mass_source=source_component("fluid_mass"),
            solid_mass_source=source_component("solid_mass"),
            fluid_momentum_source=source_component("fluid_momentum", 0),
            solid_momentum_source=source_component("solid_momentum", 0),
            kinematic_source=source_component("kinematic", 0),
        )

        bcs = []
        for tag in ("left", "right", "bottom", "top"):
            bcs.extend(
                [
                    BoundaryCondition("ux", "dirichlet", tag, lambda x, y, t: float(_full_mms_u(x, y, t)[0])),
                    BoundaryCondition("uy", "dirichlet", tag, lambda x, y, t: float(_full_mms_u(x, y, t)[1])),
                    BoundaryCondition("vsx", "dirichlet", tag, lambda x, y, t: float(_full_mms_vs(x, y, t)[0])),
                    BoundaryCondition("vsy", "dirichlet", tag, lambda x, y, t: float(_full_mms_vs(x, y, t)[1])),
                    BoundaryCondition("vfx", "dirichlet", tag, lambda x, y, t: float(_full_mms_vf(x, y, t)[0])),
                    BoundaryCondition("vfy", "dirichlet", tag, lambda x, y, t: float(_full_mms_vf(x, y, t)[1])),
                    BoundaryCondition("p", "dirichlet", tag, lambda x, y, t: float(_full_mms_p(x, y, t))),
                    BoundaryCondition("phi", "dirichlet", tag, lambda x, y, t: float(_full_mms_phi(x, y, t))),
                ]
            )
        bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]

        solver = NewtonSolver(
            system.residual_form,
            system.jacobian_form,
            dof_handler=dh,
            mixed_element=mixed_element,
            bcs=bcs,
            bcs_homog=bcs_homog,
            newton_params=NewtonParameters(newton_tol=1.0e-8, max_newton_iter=25, line_search=True, print_level=0),
            quad_order=int(qdeg),
            backend=backend,
        )
        solver.solve_time_interval(
            functions=[u, vs, vf, p, phi],
            prev_functions=[u_prev, vs_prev, vf_prev, p_prev, phi_prev],
            aux_functions={"dt": dt_c},
            time_params=TimeStepperParameters(dt=float(dt), final_time=float(dt), max_steps=1, theta=float(theta)),
        )

        t_err = float(dt)
        err_u = dh.l2_error(
            u,
            exact={
                "ux": lambda x, y: float(_full_mms_u(x, y, t_err)[0]),
                "uy": lambda x, y: float(_full_mms_u(x, y, t_err)[1]),
            },
            fields=["ux", "uy"],
            quad_order=int(qerr),
            relative=False,
        )
        err_vs = dh.l2_error(
            vs,
            exact={
                "vsx": lambda x, y: float(_full_mms_vs(x, y, t_err)[0]),
                "vsy": lambda x, y: float(_full_mms_vs(x, y, t_err)[1]),
            },
            fields=["vsx", "vsy"],
            quad_order=int(qerr),
            relative=False,
        )
        err_vf = dh.l2_error(
            vf,
            exact={
                "vfx": lambda x, y: float(_full_mms_vf(x, y, t_err)[0]),
                "vfy": lambda x, y: float(_full_mms_vf(x, y, t_err)[1]),
            },
            fields=["vfx", "vfy"],
            quad_order=int(qerr),
            relative=False,
        )
        err_p = dh.l2_error(
            p,
            exact={"p": lambda x, y: float(_full_mms_p(x, y, t_err))},
            fields=["p"],
            quad_order=int(qerr),
            relative=False,
        )
        err_phi = dh.l2_error(
            phi,
            exact={"phi": lambda x, y: float(_full_mms_phi(x, y, t_err))},
            fields=["phi"],
            quad_order=int(qerr),
            relative=False,
        )
        rows.append(
            {
                "nx": float(nx),
                "h": 1.0 / float(nx),
                "err_u": float(err_u),
                "err_vs": float(err_vs),
                "err_vf": float(err_vf),
                "err_p": float(err_p),
                "err_phi": float(err_phi),
            }
        )

    if len(rows) >= 2:
        r0, r1 = rows[-2], rows[-1]
        displacement_rate = _eoc(r0["h"], r1["h"], r0["err_u"], r1["err_u"])
        solid_velocity_rate = _eoc(r0["h"], r1["h"], r0["err_vs"], r1["err_vs"])
        fluid_velocity_rate = _eoc(r0["h"], r1["h"], r0["err_vf"], r1["err_vf"])
        pressure_rate = _eoc(r0["h"], r1["h"], r0["err_p"], r1["err_p"])
        porosity_rate = _eoc(r0["h"], r1["h"], r0["err_phi"], r1["err_phi"])
    else:
        displacement_rate = solid_velocity_rate = fluid_velocity_rate = pressure_rate = porosity_rate = float("nan")

    return FullDynamicMMSConvergenceResult(
        rows=rows,
        displacement_rate=displacement_rate,
        solid_velocity_rate=solid_velocity_rate,
        fluid_velocity_rate=fluid_velocity_rate,
        pressure_rate=pressure_rate,
        porosity_rate=porosity_rate,
    )


def validate_consolidation_reference_2d(*, backend: str = "cpp", rtol: float = 5.0e-7) -> dict[str, float]:
    """Compare the canonical first consolidation step to stored reference values."""

    ref = NONLINEAR_CONSOLIDATION_2D_REFERENCE
    result = solve_nonlinear_consolidation_2d(
        backend=backend,
        nx=int(ref["nx"]),
        ny=int(ref["ny"]),
        dt=float(ref["dt"]),
        theta=float(ref["theta"]),
        n_steps=int(ref["n_steps"]),
        top_load=float(ref["top_load"]),
    )
    got = {
        "max_pressure": result.max_pressure[-1],
        "min_pressure": result.min_pressure[-1],
        "top_vertical_displacement": result.top_vertical_displacement[-1],
    }
    errors = {}
    for key, value in got.items():
        ref_value = float(ref[key])
        if not math.isfinite(ref_value):
            errors[key] = float("nan")
            continue
        denom = max(1.0, abs(ref_value))
        errors[key] = abs(float(value) - ref_value) / denom
        if errors[key] > float(rtol):
            raise AssertionError(f"{key} relative error {errors[key]:.3e} exceeds {rtol:.3e}")
    return errors


def run_transient_to_steady_validation_2d(
    *,
    backend: str = "cpp",
    nx: int = 1,
    ny: int = 2,
    dt: float = 0.08,
    n_steps: int = 5,
) -> NonlinearSteadyValidationResult:
    """Check that transient consolidation decays toward a steady drained response."""

    response = solve_nonlinear_consolidation_2d(
        backend=backend,
        nx=nx,
        ny=ny,
        dt=dt,
        n_steps=n_steps,
        top_load=500.0,
        drained_top=True,
        permeability=5.0e-5,
    )
    p0 = max(abs(response.max_pressure[0]), 1.0e-14)
    pressure_decay_ratio = abs(response.max_pressure[-1]) / p0
    if len(response.top_vertical_displacement) >= 3:
        first_inc = abs(response.top_vertical_displacement[1] - response.top_vertical_displacement[0])
        last_inc = abs(response.top_vertical_displacement[-1] - response.top_vertical_displacement[-2])
    else:
        first_inc = abs(response.top_vertical_displacement[0])
        last_inc = first_inc
    displacement_increment_ratio = last_inc / max(first_inc, 1.0e-14)
    return NonlinearSteadyValidationResult(
        response=response,
        pressure_decay_ratio=float(pressure_decay_ratio),
        displacement_increment_ratio=float(displacement_increment_ratio),
    )


def run_drained_undrained_benchmark_2d(
    *,
    backend: str = "cpp",
    nx: int = 1,
    ny: int = 2,
    dt: float = 0.06,
) -> NonlinearBenchmarkComparison:
    """Compare drained and undrained column responses under the same load."""

    drained = solve_nonlinear_consolidation_2d(
        backend=backend,
        nx=nx,
        ny=ny,
        dt=dt,
        n_steps=5,
        top_load=500.0,
        drained_top=True,
        permeability=5.0e-5,
    )
    undrained = solve_nonlinear_consolidation_2d(
        backend=backend,
        nx=nx,
        ny=ny,
        dt=dt,
        n_steps=1,
        top_load=500.0,
        drained_top=False,
        permeability=1.0e-10,
    )
    drained_pressure = max(abs(drained.max_pressure[-1]), 1.0e-14)
    pressure_ratio = abs(undrained.max_pressure[-1]) / drained_pressure
    drained_settlement = max(abs(drained.top_vertical_displacement[-1]), 1.0e-14)
    settlement_ratio = abs(undrained.top_vertical_displacement[-1]) / drained_settlement
    return NonlinearBenchmarkComparison(
        drained=drained,
        undrained=undrained,
        pressure_ratio_undrained_to_drained=float(pressure_ratio),
        settlement_ratio_undrained_to_drained=float(settlement_ratio),
    )


def run_full_dynamic_transient_to_steady_validation_2d(
    *,
    backend: str = "cpp",
    nx: int = 1,
    ny: int = 2,
    dt: float = 0.05,
    n_steps: int = 15,
) -> FullDynamicSteadyValidationResult:
    """Check that full dynamic drainage dissipates after the pressure peak."""

    response = solve_full_dynamic_consolidation_2d(
        backend=backend,
        nx=nx,
        ny=ny,
        dt=dt,
        n_steps=n_steps,
        top_load=500.0,
        drained_top=True,
        permeability=5.0e-5,
    )
    p_peak = max(max(abs(v) for v in response.max_pressure), 1.0e-14)
    solid_peak = max(max(abs(v) for v in response.max_solid_speed), 1.0e-14)
    fluid_peak = max(max(abs(v) for v in response.max_fluid_speed), 1.0e-14)
    return FullDynamicSteadyValidationResult(
        response=response,
        pressure_decay_ratio=float(abs(response.max_pressure[-1]) / p_peak),
        solid_speed_decay_ratio=float(abs(response.max_solid_speed[-1]) / solid_peak),
        fluid_speed_decay_ratio=float(abs(response.max_fluid_speed[-1]) / fluid_peak),
    )


def run_full_dynamic_drained_undrained_benchmark_2d(
    *,
    backend: str = "cpp",
    nx: int = 1,
    ny: int = 2,
    dt: float = 0.01,
) -> FullDynamicBenchmarkComparison:
    """Compare full dynamic drained and undrained column responses."""

    drained = solve_full_dynamic_consolidation_2d(
        backend=backend,
        nx=nx,
        ny=ny,
        dt=dt,
        n_steps=1,
        top_load=500.0,
        drained_top=True,
        permeability=5.0e-5,
    )
    undrained = solve_full_dynamic_consolidation_2d(
        backend=backend,
        nx=nx,
        ny=ny,
        dt=dt,
        n_steps=1,
        top_load=500.0,
        drained_top=False,
        permeability=5.0e-5,
    )
    drained_pressure = max(abs(drained.max_pressure[-1]), 1.0e-14)
    drained_settlement = max(abs(drained.top_vertical_displacement[-1]), 1.0e-14)
    return FullDynamicBenchmarkComparison(
        drained=drained,
        undrained=undrained,
        pressure_ratio_undrained_to_drained=float(abs(undrained.max_pressure[-1]) / drained_pressure),
        settlement_ratio_undrained_to_drained=float(abs(undrained.top_vertical_displacement[-1]) / drained_settlement),
    )


def run_full_dynamic_terzaghi_benchmark_2d(
    *,
    backend: str = "cpp",
    nx: int = 1,
    ny: int = 12,
    dt: float = 0.02,
    n_steps: int = 10,
    top_load: float = 10.0,
    permeability: float = 1.0e-5,
    fluid_compressibility: float = 1.0e-4,
    phi0: float = 0.3,
    skeleton_mu: float = 4.0e4,
    skeleton_lambda: float = 6.0e4,
) -> FullDynamicTerzaghiBenchmarkResult:
    """Compare the full model with Terzaghi's single-drainage series solution.

    The comparison is made in the small-strain, low-inertia oedometric limit.
    Terzaghi's degree of consolidation is applied to the consolidation
    settlement after subtracting the immediate undrained settlement.
    """

    constrained_modulus = (1.0 - float(phi0)) * (float(skeleton_lambda) + 2.0 * float(skeleton_mu))
    storage = float(phi0) * (float(fluid_compressibility) + 1.0 / constrained_modulus)
    consolidation_coefficient = float(permeability) / storage
    final_settlement = float(top_load) / constrained_modulus
    initial_settlement = float(top_load) / (constrained_modulus + 1.0 / float(fluid_compressibility))
    consolidation_settlement = final_settlement - initial_settlement
    if consolidation_settlement <= 0.0:
        raise ValueError("Terzaghi benchmark requires positive delayed consolidation settlement.")

    response = solve_full_dynamic_consolidation_2d(
        backend=backend,
        nx=nx,
        ny=ny,
        dt=dt,
        theta=1.0,
        n_steps=n_steps,
        top_load=top_load,
        drained_top=True,
        permeability=permeability,
        fluid_compressibility=fluid_compressibility,
        phi0=phi0,
        fluid_density_ref=1.0e-6,
        density_solid=1.0e-6,
        dynamic_viscosity_fluid=1.0,
        skeleton_mu=skeleton_mu,
        skeleton_lambda=skeleton_lambda,
    )

    numerical_degree: list[float] = []
    reference_degree: list[float] = []
    degree_errors: list[float] = []
    for time, displacement in zip(response.times, response.top_vertical_displacement):
        numerical = (abs(float(displacement)) - initial_settlement) / consolidation_settlement
        reference = terzaghi_single_drainage_degree(consolidation_coefficient * float(time))
        numerical = float(np.clip(numerical, 0.0, 1.0))
        numerical_degree.append(numerical)
        reference_degree.append(reference)
        degree_errors.append(abs(numerical - reference))

    errors = np.asarray(degree_errors, dtype=float)
    return FullDynamicTerzaghiBenchmarkResult(
        times=list(response.times),
        numerical_degree=numerical_degree,
        reference_degree=reference_degree,
        degree_errors=degree_errors,
        max_degree_error=float(np.max(errors)),
        rms_degree_error=float(np.sqrt(np.mean(errors**2))),
        constrained_modulus=float(constrained_modulus),
        consolidation_coefficient=float(consolidation_coefficient),
        initial_settlement=float(initial_settlement),
        final_settlement=float(final_settlement),
        reference=(
            "Terzaghi single-drainage consolidation series; see Terzaghi (1943) "
            "and Mei & Chen (2013), DOI:10.1007/s11771-013-1730-5."
        ),
    )


def _print_mms(result: NonlinearMMSConvergenceResult) -> None:
    for row in result.rows:
        print(
            f"MMS nx={int(row['nx'])} h={row['h']:.3e} "
            f"err_u={row['err_u']:.3e} err_p={row['err_p']:.3e}",
            flush=True,
        )
    print(f"MMS rates: u={result.displacement_rate:.2f}, p={result.pressure_rate:.2f}", flush=True)


def _print_full_mms(result: FullDynamicMMSConvergenceResult) -> None:
    for row in result.rows:
        print(
            f"Full MMS nx={int(row['nx'])} h={row['h']:.3e} "
            f"err_u={row['err_u']:.3e} err_vs={row['err_vs']:.3e} "
            f"err_vf={row['err_vf']:.3e} err_p={row['err_p']:.3e} err_phi={row['err_phi']:.3e}",
            flush=True,
        )
    print(
        "Full MMS rates: "
        f"u={result.displacement_rate:.2f}, vs={result.solid_velocity_rate:.2f}, "
        f"vf={result.fluid_velocity_rate:.2f}, p={result.pressure_rate:.2f}, "
        f"phi={result.porosity_rate:.2f}",
        flush=True,
    )


def _print_full_terzaghi(result: FullDynamicTerzaghiBenchmarkResult) -> None:
    print(
        "Full Terzaghi benchmark: "
        f"max_degree_error={result.max_degree_error:.3e}, "
        f"rms_degree_error={result.rms_degree_error:.3e}, "
        f"cv={result.consolidation_coefficient:.3e}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "all",
            "mms",
            "consolidation",
            "steady",
            "drained-undrained",
            "full-mms",
            "full-consolidation",
            "full-steady",
            "full-drained-undrained",
            "full-terzaghi",
        ),
        default="all",
    )
    parser.add_argument("--backend", choices=("cpp", "jit", "python"), default="cpp")
    args = parser.parse_args()

    if args.case in {"all", "mms"}:
        _print_mms(run_nonlinear_mms_convergence_2d(backend=args.backend))
    if args.case in {"all", "full-mms"}:
        _print_full_mms(run_full_dynamic_mms_convergence_2d(backend=args.backend))
    if args.case in {"all", "consolidation"}:
        result = solve_nonlinear_consolidation_2d(backend=args.backend, nx=1, ny=2, dt=0.02, n_steps=1, top_load=500.0)
        print(
            "Consolidation first step: "
            f"pmax={result.max_pressure[-1]:.6e}, pmin={result.min_pressure[-1]:.6e}, "
            f"uy_top={result.top_vertical_displacement[-1]:.6e}",
            flush=True,
        )
    if args.case in {"all", "full-consolidation"}:
        result = solve_full_dynamic_consolidation_2d(backend=args.backend, nx=1, ny=2, dt=0.01, n_steps=1, top_load=500.0)
        print(
            "Full consolidation first step: "
            f"pmax={result.max_pressure[-1]:.6e}, pmin={result.min_pressure[-1]:.6e}, "
            f"uy_top={result.top_vertical_displacement[-1]:.6e}, "
            f"phi_mean={result.mean_porosity[-1]:.6e}",
            flush=True,
        )
    if args.case in {"all", "steady"}:
        steady = run_transient_to_steady_validation_2d(backend=args.backend)
        print(
            "Transient-to-steady: "
            f"pressure_decay={steady.pressure_decay_ratio:.3e}, "
            f"last_increment_ratio={steady.displacement_increment_ratio:.3e}",
            flush=True,
        )
    if args.case in {"all", "full-steady"}:
        steady = run_full_dynamic_transient_to_steady_validation_2d(backend=args.backend)
        print(
            "Full transient-to-steady: "
            f"pressure_decay={steady.pressure_decay_ratio:.3e}, "
            f"solid_speed_decay={steady.solid_speed_decay_ratio:.3e}, "
            f"fluid_speed_decay={steady.fluid_speed_decay_ratio:.3e}",
            flush=True,
        )
    if args.case in {"all", "drained-undrained"}:
        bench = run_drained_undrained_benchmark_2d(backend=args.backend)
        print(
            "Drained/undrained: "
            f"pressure_ratio={bench.pressure_ratio_undrained_to_drained:.3e}, "
            f"settlement_ratio={bench.settlement_ratio_undrained_to_drained:.3e}",
            flush=True,
        )
    if args.case in {"all", "full-drained-undrained"}:
        bench = run_full_dynamic_drained_undrained_benchmark_2d(backend=args.backend)
        print(
            "Full drained/undrained: "
            f"pressure_ratio={bench.pressure_ratio_undrained_to_drained:.3e}, "
            f"settlement_ratio={bench.settlement_ratio_undrained_to_drained:.3e}",
            flush=True,
        )
    if args.case in {"all", "full-terzaghi"}:
        _print_full_terzaghi(run_full_dynamic_terzaghi_benchmark_2d(backend=args.backend))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
