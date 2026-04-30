import math

import numpy as np
import pytest

from examples.utils.poromechanics import (
    ExponentialFluidEOS,
    NonlinearPoromechanicsMaterial2D,
    build_nonlinear_poromechanics_full_dynamic_theta_system_2d,
    build_nonlinear_poromechanics_reduced_theta_system_2d,
    porosity_from_jacobian_value,
    spatial_inverse_permeability_from_reference_value,
    spatial_permeability_from_reference_value,
)
from examples.poromechanics import (
    run_full_dynamic_drained_undrained_benchmark_2d,
    run_full_dynamic_mms_convergence_2d,
    run_full_dynamic_terzaghi_benchmark_2d,
    run_full_dynamic_transient_to_steady_validation_2d,
    run_drained_undrained_benchmark_2d,
    run_nonlinear_mms_convergence_2d,
    run_transient_to_steady_validation_2d,
    solve_full_dynamic_consolidation_2d,
    solve_nonlinear_consolidation_2d,
    validate_consolidation_reference_2d,
)
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _have_cpp_backend() -> bool:
    try:
        import pybind11  # noqa: F401
    except Exception:
        return False
    return True


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


def _mesh_and_fields():
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "p": 1})
    dh = DofHandler(me, method="cg")
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
    return mesh, me, dh, du, eta, dp, w, u, p, u_prev, p_prev, u_prevprev


def _material(scale: float = 1.0) -> NonlinearPoromechanicsMaterial2D:
    return NonlinearPoromechanicsMaterial2D(
        phi0=0.25 + 0.03 * scale,
        density_solid=1800.0 + 50.0 * scale,
        fluid_eos=ExponentialFluidEOS(density_ref=950.0 + 10.0 * scale, compressibility=1.0e-5 * scale),
        dynamic_viscosity_fluid=1.0 + 0.1 * scale,
        permeability_ref=((1.0e-6 * scale, 0.2e-6), (0.2e-6, 1.5e-6 * scale)),
        skeleton_mu=2.0e4 * scale,
        skeleton_lambda=3.0e4 * scale,
    )


def _build_system(material: NonlinearPoromechanicsMaterial2D, *, dt: float = 0.1, theta: float = 1.0):
    mesh, me, dh, du, eta, dp, w, u, p, u_prev, p_prev, u_prevprev = _mesh_and_fields()
    return (
        build_nonlinear_poromechanics_reduced_theta_system_2d(
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
            dt=dt,
            theta=theta,
            dx_measure=dx(metadata={"q": 6}),
        ),
        (mesh, me, dh, u, p, u_prev, p_prev, u_prevprev),
    )


def test_nonlinear_poromechanics_eos_porosity_and_permeability_identities():
    eos = ExponentialFluidEOS(density_ref=1000.0, compressibility=2.0e-5, pressure_ref=10.0)
    rho = eos.rho_value(15.0)
    assert rho > 0.0
    np.testing.assert_allclose(eos.drho_dp_value(15.0), rho * 2.0e-5)

    phi = porosity_from_jacobian_value(1.2, 0.3)
    np.testing.assert_allclose((1.0 - phi) * 1.2, 1.0 - 0.3)
    assert 0.0 < phi < 1.0

    angle = 0.37
    R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    k0 = np.diag([2.0e-6, 5.0e-7])
    k = spatial_permeability_from_reference_value(R, k0)
    kinv = spatial_inverse_permeability_from_reference_value(R, k0)

    np.testing.assert_allclose(k, R @ k0 @ R.T, rtol=1.0e-13, atol=1.0e-18)
    np.testing.assert_allclose(kinv, np.linalg.inv(k), rtol=1.0e-12, atol=1.0e-8)


def _build_full_dynamic_system(material: NonlinearPoromechanicsMaterial2D, *, dt: float = 0.05, theta: float = 0.6):
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(
        mesh,
        field_specs={"ux": 1, "uy": 1, "vsx": 1, "vsy": 1, "vfx": 1, "vfy": 1, "p": 1, "phi": 1},
    )
    dh = DofHandler(me, method="cg")
    Vu = FunctionSpace("solid_displacement", ["ux", "uy"], dim=1)
    Vsf = FunctionSpace("solid_velocity", ["vsx", "vsy"], dim=1)
    Vff = FunctionSpace("fluid_velocity", ["vfx", "vfy"], dim=1)

    du = VectorTrialFunction(Vu, dof_handler=dh)
    dvs = VectorTrialFunction(Vsf, dof_handler=dh)
    dvf = VectorTrialFunction(Vff, dof_handler=dh)
    dp = TrialFunction(name="dp", field_name="p", dof_handler=dh)
    dphi = TrialFunction(name="dphi", field_name="phi", dof_handler=dh)

    eta_u = VectorTestFunction(Vu, dof_handler=dh)
    eta_s = VectorTestFunction(Vsf, dof_handler=dh)
    eta_f = VectorTestFunction(Vff, dof_handler=dh)
    w_p = TestFunction(name="w_p", field_name="p", dof_handler=dh)
    w_phi = TestFunction(name="w_phi", field_name="phi", dof_handler=dh)

    u = VectorFunction("u", ["ux", "uy"], dof_handler=dh)
    vs = VectorFunction("vs", ["vsx", "vsy"], dof_handler=dh)
    vf = VectorFunction("vf", ["vfx", "vfy"], dof_handler=dh)
    p = Function("p", "p", dof_handler=dh)
    phi = Function("phi", "phi", dof_handler=dh)
    u_prev = VectorFunction("u_prev", ["ux", "uy"], dof_handler=dh)
    vs_prev = VectorFunction("vs_prev", ["vsx", "vsy"], dof_handler=dh)
    vf_prev = VectorFunction("vf_prev", ["vfx", "vfy"], dof_handler=dh)
    p_prev = Function("p_prev", "p", dof_handler=dh)
    phi_prev = Function("phi_prev", "phi", dof_handler=dh)

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
        dt=dt,
        theta=theta,
        dx_measure=dx(metadata={"q": 4}),
    )
    fields = {
        "ux": u,
        "uy": u,
        "vsx": vs,
        "vsy": vs,
        "vfx": vf,
        "vfy": vf,
        "p": p,
        "phi": phi,
        "u": u,
        "vs": vs,
        "vf": vf,
        "p_current": p,
        "phi_current": phi,
    }
    prev = (u_prev, vs_prev, vf_prev, p_prev, phi_prev)
    current = (u, vs, vf, p, phi)
    return system, (mesh, me, dh, fields, current, prev)


def test_nonlinear_poromechanics_constants_are_named_and_value_independent():
    system_a, _ = _build_system(_material(1.0), dt=0.1, theta=1.0)
    system_b, _ = _build_system(_material(2.0), dt=0.05, theta=0.5)

    assert _jit_ir_signature(system_a.residual_form) == _jit_ir_signature(system_b.residual_form)
    assert _jit_ir_signature(system_a.jacobian_form) == _jit_ir_signature(system_b.jacobian_form)

    names = _jit_parameter_names(system_a.residual_form) + _jit_parameter_names(system_a.jacobian_form)
    assert names
    assert not [name for name in names if name.startswith(("jit_const_", "jit_ewc_"))]


def test_full_dynamic_nonlinear_poromechanics_constants_are_named():
    system, _ = _build_full_dynamic_system(_material(1.0))
    names = _jit_parameter_names(system.residual_form) + _jit_parameter_names(system.jacobian_form)
    assert names
    assert not [name for name in names if name.startswith(("jit_const_", "jit_ewc_"))]


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_nonlinear_poromechanics_reduced_jacobian_matches_fd_cpp_backend():
    system, (_, _, dh, u, p, u_prev, p_prev, u_prevprev) = _build_system(_material(1.0), dt=0.08, theta=0.7)

    u.set_values_from_function(lambda x, y: np.array([0.015 * x + 0.004 * y, -0.006 * y + 0.003 * x]))
    p.set_values_from_function(lambda x, y: 0.1 * x - 0.03 * y)
    u_prev.set_values_from_function(lambda x, y: np.array([0.006 * x, -0.002 * y]))
    p_prev.set_values_from_function(lambda x, y: 0.02 * x + 0.01 * y)
    u_prevprev.nodal_values.fill(0.0)

    compiler = FormCompiler(dh, quadrature_order=6, backend="cpp")
    K, _ = compiler.assemble(Equation(system.jacobian_form, None), bcs=[])
    _, R0 = compiler.assemble(Equation(None, system.residual_form), bcs=[])

    direction = np.zeros(dh.total_dofs, dtype=float)
    for field in ("ux", "uy", "p"):
        dofs = np.asarray(dh.element_dofs(field, 0), dtype=int)
        direction[int(dofs[0])] = 1.0

    fields = {"ux": u, "uy": u, "p": p}
    base_values = {}
    touched = {}
    for gdof in np.flatnonzero(direction):
        field, _ = dh._dof_to_node_map[int(gdof)]
        touched.setdefault(field, []).append(int(gdof))
    for field, dofs in touched.items():
        arr = np.asarray(dofs, dtype=int)
        base_values[field] = fields[field].get_nodal_values(arr)

    eps = 1.0e-7
    for sign in (+1.0, -1.0):
        for field, dofs in touched.items():
            arr = np.asarray(dofs, dtype=int)
            fields[field].set_nodal_values(arr, base_values[field] + sign * eps * direction[arr])
        _, residual = compiler.assemble(Equation(None, system.residual_form), bcs=[])
        if sign > 0.0:
            R_plus = residual
        else:
            R_minus = residual
    for field, dofs in touched.items():
        arr = np.asarray(dofs, dtype=int)
        fields[field].set_nodal_values(arr, base_values[field])

    fd = (R_plus - R_minus) / (2.0 * eps)
    jac = K.dot(direction)
    err = float(np.linalg.norm(fd - jac, ord=np.inf))
    mag = float(np.linalg.norm(jac, ord=np.inf))
    rel = err / (mag + 1.0e-14)

    assert np.linalg.norm(R0, ord=np.inf) > 0.0
    assert math.isfinite(err) and math.isfinite(rel)
    assert err < 1.0e-5
    assert rel < 1.0e-4


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_full_dynamic_nonlinear_poromechanics_jacobian_matches_fd_cpp_backend():
    system, (_, _, dh, fields, current, prev) = _build_full_dynamic_system(_material(1.0), dt=0.05, theta=0.6)
    u, vs, vf, p, phi = current
    u_prev, vs_prev, vf_prev, p_prev, phi_prev = prev

    u.set_values_from_function(lambda x, y: np.array([0.005 * x, -0.003 * y]))
    vs.set_values_from_function(lambda x, y: np.array([0.02 * x + 0.01 * y, -0.01 * y]))
    vf.set_values_from_function(lambda x, y: np.array([0.03 * x, -0.02 * y + 0.004 * x]))
    p.set_values_from_function(lambda x, y: 0.04 * x - 0.01 * y)
    phi.set_values_from_function(lambda x, y: 0.31 + 0.01 * x - 0.005 * y)
    u_prev.set_values_from_function(lambda x, y: np.array([0.002 * x, -0.001 * y]))
    vs_prev.set_values_from_function(lambda x, y: np.array([0.01 * x, -0.004 * y]))
    vf_prev.set_values_from_function(lambda x, y: np.array([0.015 * x, -0.006 * y]))
    p_prev.set_values_from_function(lambda x, y: 0.02 * x)
    phi_prev.set_values_from_function(lambda x, y: 0.30 + 0.002 * x)

    compiler = FormCompiler(dh, quadrature_order=4, backend="cpp")
    K, _ = compiler.assemble(Equation(system.jacobian_form, None), bcs=[])
    _, R0 = compiler.assemble(Equation(None, system.residual_form), bcs=[])

    direction = np.zeros(dh.total_dofs, dtype=float)
    for field in ("ux", "vsx", "vfx", "p", "phi"):
        dofs = np.asarray(dh.element_dofs(field, 0), dtype=int)
        direction[int(dofs[0])] = 1.0

    touched = {}
    for gdof in np.flatnonzero(direction):
        field, _ = dh._dof_to_node_map[int(gdof)]
        touched.setdefault(field, []).append(int(gdof))
    base_values = {}
    for field, dofs in touched.items():
        arr = np.asarray(dofs, dtype=int)
        base_values[field] = fields[field].get_nodal_values(arr)

    eps = 2.0e-7
    for sign in (+1.0, -1.0):
        for field, dofs in touched.items():
            arr = np.asarray(dofs, dtype=int)
            fields[field].set_nodal_values(arr, base_values[field] + sign * eps * direction[arr])
        _, residual = compiler.assemble(Equation(None, system.residual_form), bcs=[])
        if sign > 0.0:
            R_plus = residual
        else:
            R_minus = residual
    for field, dofs in touched.items():
        arr = np.asarray(dofs, dtype=int)
        fields[field].set_nodal_values(arr, base_values[field])

    fd = (R_plus - R_minus) / (2.0 * eps)
    jac = K.dot(direction)
    err = float(np.linalg.norm(fd - jac, ord=np.inf))
    mag = float(np.linalg.norm(jac, ord=np.inf))
    rel = err / (mag + 1.0e-14)

    assert np.linalg.norm(R0, ord=np.inf) > 0.0
    assert math.isfinite(err) and math.isfinite(rel)
    assert err < 5.0e-5
    assert rel < 5.0e-4


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_nonlinear_poromechanics_consolidation_first_step_cpp_backend():
    result = solve_nonlinear_consolidation_2d(backend="cpp", nx=1, ny=2, dt=0.02, n_steps=1, top_load=500.0)

    assert result.backend == "cpp"
    assert result.times == [0.02]
    assert np.isfinite(result.max_pressure[-1])
    assert np.isfinite(result.min_pressure[-1])
    assert np.isfinite(result.top_vertical_displacement[-1])
    assert result.max_pressure[-1] > 0.0
    assert result.top_vertical_displacement[-1] < 0.0


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_nonlinear_poromechanics_consolidation_reference_cpp_backend():
    errors = validate_consolidation_reference_2d(backend="cpp", rtol=1.0e-10)
    assert errors["max_pressure"] < 1.0e-10
    assert errors["top_vertical_displacement"] < 1.0e-10


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_nonlinear_poromechanics_mms_converges_cpp_backend():
    result = run_nonlinear_mms_convergence_2d(backend="cpp", nx_values=(2, 4), dt=0.05, theta=1.0)

    assert result.rows[-1]["err_u"] < result.rows[0]["err_u"]
    assert result.rows[-1]["err_p"] < result.rows[0]["err_p"]
    assert result.displacement_rate > 2.0
    assert result.pressure_rate > 1.3


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_nonlinear_poromechanics_transient_reaches_steady_trend_cpp_backend():
    result = run_transient_to_steady_validation_2d(backend="cpp", nx=1, ny=2, dt=0.08, n_steps=5)

    assert result.pressure_decay_ratio < 0.05
    assert result.displacement_increment_ratio < 0.1
    assert result.response.max_pressure[-1] < result.response.max_pressure[0]


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_nonlinear_poromechanics_drained_undrained_benchmark_cpp_backend():
    result = run_drained_undrained_benchmark_2d(backend="cpp", nx=1, ny=2, dt=0.06)

    assert result.pressure_ratio_undrained_to_drained > 20.0
    assert result.settlement_ratio_undrained_to_drained < 0.8
    assert result.undrained.max_pressure[-1] > result.drained.max_pressure[-1]


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_full_dynamic_nonlinear_poromechanics_mms_converges_cpp_backend():
    result = run_full_dynamic_mms_convergence_2d(backend="cpp", nx_values=(2, 4), dt=0.04, theta=1.0)

    assert result.rows[-1]["err_u"] < result.rows[0]["err_u"]
    assert result.rows[-1]["err_vs"] < result.rows[0]["err_vs"]
    assert result.rows[-1]["err_vf"] < result.rows[0]["err_vf"]
    assert result.rows[-1]["err_p"] < result.rows[0]["err_p"]
    assert result.rows[-1]["err_phi"] < result.rows[0]["err_phi"]
    assert result.displacement_rate > 2.0
    assert result.solid_velocity_rate > 2.0
    assert result.fluid_velocity_rate > 2.0
    assert result.pressure_rate > 1.3
    assert result.porosity_rate > 1.3


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_full_dynamic_nonlinear_poromechanics_consolidation_first_step_cpp_backend():
    result = solve_full_dynamic_consolidation_2d(backend="cpp", nx=1, ny=2, dt=0.01, n_steps=1, top_load=500.0)

    assert result.backend == "cpp"
    assert result.times == [0.01]
    assert np.isfinite(result.max_pressure[-1])
    assert np.isfinite(result.top_vertical_displacement[-1])
    assert result.max_pressure[-1] > 0.0
    assert result.top_vertical_displacement[-1] < 0.0
    assert result.mean_porosity[-1] < 0.3
    assert result.max_solid_speed[-1] > 0.0
    assert result.max_fluid_speed[-1] > 0.0


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_full_dynamic_nonlinear_poromechanics_transient_dissipates_cpp_backend():
    result = run_full_dynamic_transient_to_steady_validation_2d(backend="cpp", nx=1, ny=2, dt=0.05, n_steps=15)

    assert result.pressure_decay_ratio < 0.05
    assert result.solid_speed_decay_ratio < 0.05
    assert result.fluid_speed_decay_ratio < 0.2
    assert result.response.top_vertical_displacement[-1] < 0.0


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_full_dynamic_nonlinear_poromechanics_drained_undrained_benchmark_cpp_backend():
    result = run_full_dynamic_drained_undrained_benchmark_2d(backend="cpp", nx=1, ny=2, dt=0.01)

    assert result.pressure_ratio_undrained_to_drained > 5.0
    assert 0.8 < result.settlement_ratio_undrained_to_drained < 1.2
    assert result.undrained.max_pressure[-1] > result.drained.max_pressure[-1]


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_full_dynamic_nonlinear_poromechanics_terzaghi_external_benchmark_cpp_backend():
    result = run_full_dynamic_terzaghi_benchmark_2d(backend="cpp", nx=1, ny=12, dt=0.02, n_steps=10)

    assert result.max_degree_error < 2.0e-2
    assert result.rms_degree_error < 1.0e-2
    assert result.numerical_degree[-1] > result.numerical_degree[0]
    assert result.reference_degree[-1] > result.reference_degree[0]
