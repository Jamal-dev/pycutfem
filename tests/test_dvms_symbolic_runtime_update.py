from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from examples.NIRB.dvms import (
    FluidDVMSSolverOperator,
    _bossak_coefficients,
    _build_fluid_dvms_state,
    _kratos_dvms_current_element_size_array,
)
from examples.NIRB.dvms.local_operator import (
    _compress_batch_to_fluid_block,
    assemble_dvms_calculate_local_system,
    assemble_fluid_dvms_local_contribution_batch,
)
from examples.NIRB.dvms.symbolics import (
    build_fluid_dvms_old_mass_residual,
    build_fluid_dvms_predictor_symbolics,
)
from examples.NIRB.dvms.update import (
    _update_fluid_dvms_predicted_subscale,
    _update_fluid_dvms_state_from_previous_step,
)
from pycutfem.operators import LocalAssemblyResult
from pycutfem.solvers.nonlinear_solver import NewtonSolver
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem import transform
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import Function, VectorFunction
from pycutfem.utils.meshgen import structured_triangles


def _make_dvms_problem(
    *,
    poly_order: int = 2,
    pressure_order: int = 1,
    quadrature_order: int = 3,
    nx_quads: int = 1,
    ny_quads: int = 1,
):
    nodes, elems, edges, corners = structured_triangles(
        1.0,
        1.0,
        nx_quads=int(nx_quads),
        ny_quads=int(ny_quads),
        poly_order=poly_order,
    )
    mesh = Mesh(
        nodes,
        elems,
        edges,
        corners,
        element_type="tri",
        poly_order=poly_order,
    )
    me = MixedElement(
        mesh,
        field_specs={
            "ux": poly_order,
            "uy": poly_order,
            "p": pressure_order,
            "mx": poly_order,
            "my": poly_order,
        },
    )
    dh = DofHandler(me, method="cg")

    u_k = VectorFunction("u_k", ["ux", "uy"], dh)
    u_prev = VectorFunction("u_prev", ["ux", "uy"], dh)
    a_prev = VectorFunction("a_prev", ["ux", "uy"], dh)
    d_mesh = VectorFunction("d_mesh", ["mx", "my"], dh)
    d_prev = VectorFunction("d_prev", ["mx", "my"], dh)
    p_k = Function("p_k", "p", dh)

    u_k.set_values_from_function(lambda x, y: np.array([x + y + 0.1 * x * y, x * x + 0.3 * y]))
    u_prev.set_values_from_function(lambda x, y: np.array([0.7 * x + 0.2 * y, 0.4 * x + 0.6 * y]))
    a_prev.set_values_from_function(lambda x, y: np.array([0.2 + 0.1 * x, -0.15 + 0.05 * y]))
    d_mesh.set_values_from_function(lambda x, y: np.array([0.03 * x * x, 0.02 * y * y]))
    d_prev.set_values_from_function(lambda x, y: np.array([0.015 * x * x, 0.01 * y * y]))
    p_k.set_values_from_function(lambda x, y: x - 0.5 * y + 0.1 * x * y)

    state = _build_fluid_dvms_state(mesh, quadrature_order=quadrature_order)
    xy = np.asarray(state.sample_coords, dtype=float)
    state.old_subscale_velocity[:, 0] = 0.01 + 0.02 * xy[:, 0]
    state.old_subscale_velocity[:, 1] = -0.015 + 0.01 * xy[:, 1]
    state.predicted_subscale_velocity[:, 0] = 0.02 - 0.005 * xy[:, 0]
    state.predicted_subscale_velocity[:, 1] = -0.01 + 0.004 * xy[:, 1]
    state.momentum_projection[:, 0] = 0.03 * xy[:, 0]
    state.momentum_projection[:, 1] = -0.025 * xy[:, 1]
    state.mass_projection[:] = 0.1 * (xy[:, 0] - xy[:, 1])
    state.old_mass_residual[:] = -0.05 + 0.02 * xy[:, 0]
    state.sync_coefficients_from_samples()
    return mesh, dh, state, u_k, u_prev, a_prev, p_k, d_mesh, d_prev


def _assemble_global_fluid_block_for_mode(
    *,
    mesh,
    dh,
    state,
    u_k,
    u_prev,
    a_prev,
    p_k,
    d_mesh,
    d_prev,
    rho_f: float,
    mu_f: float,
    dt: float,
    bossak_alpha: float,
    quadrature_order: int,
    contribution_mode: str,
    backend: str,
):
    batch = assemble_fluid_dvms_local_contribution_batch(
        mesh=mesh,
        dh=dh,
        u_k=u_k,
        u_prev=u_prev,
        a_prev=a_prev,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        state=state,
        rho_f=float(rho_f),
        mu_f=float(mu_f),
        dt=float(dt),
        bossak_alpha=float(bossak_alpha),
        quadrature_order=int(quadrature_order),
        element_ids=np.arange(int(mesh.n_elements), dtype=int),
        contribution_mode=str(contribution_mode),
        backend=str(backend),
    )
    K_elem, F_elem, gdofs_map = _compress_batch_to_fluid_block(
        dh,
        LocalAssemblyResult(
            K_elem=None if batch.K_elem is None else np.asarray(batch.K_elem, dtype=float),
            F_elem=None if batch.F_elem is None else np.asarray(batch.F_elem, dtype=float),
            element_ids=np.asarray(batch.element_ids, dtype=int),
            gdofs_map=np.asarray(batch.gdofs_map, dtype=int),
        ),
    )

    ndof = int(dh.total_dofs)
    R_full = np.zeros((ndof,), dtype=float)
    scatter_owner = NewtonSolver.__new__(NewtonSolver)
    A_sparse, R_full = NewtonSolver.scatter_element_contribs_full(
        scatter_owner,
        K_elem=K_elem,
        F_elem=F_elem,
        element_ids=np.asarray(batch.element_ids, dtype=int),
        gdofs_map=np.asarray(gdofs_map, dtype=int),
        A_full=None if K_elem is None else sp.lil_matrix((ndof, ndof), dtype=float),
        R_full=R_full,
    )
    return (None if A_sparse is None else A_sparse.toarray()), np.asarray(R_full, dtype=float)


def _eval_scalar_with_grad_on_sample(*, state, dh, mesh, scalar, sample_idx: int):
    eid = int(state.sample_element_ids[sample_idx])
    xi_eta = np.asarray(state.sample_ref_coords[sample_idx], dtype=float)
    xi = float(xi_eta[0])
    eta = float(xi_eta[1])
    me = dh.mixed_element
    local_phi = me.basis(scalar.field_name, xi, eta)[me.slice(scalar.field_name)]
    local_grad_ref = me.grad_basis(scalar.field_name, xi, eta)[me.slice(scalar.field_name)]
    local_grad = transform.map_grad_scalar(mesh, eid, local_grad_ref, (xi, eta))
    gdofs = dh.element_maps[scalar.field_name][eid]
    vals = scalar.get_nodal_values(gdofs)
    return float(local_phi @ vals), np.asarray(vals, dtype=float) @ np.asarray(local_grad, dtype=float)


def _eval_vector_with_grad_on_sample(*, state, dh, mesh, vector, sample_idx: int):
    values = []
    grads = []
    for component in vector.components:
        value, grad_value = _eval_scalar_with_grad_on_sample(
            state=state,
            dh=dh,
            mesh=mesh,
            scalar=component,
            sample_idx=sample_idx,
        )
        values.append(value)
        grads.append(grad_value)
    return np.asarray(values, dtype=float), np.vstack(grads)


def _manual_old_mass_residual(*, state, dh, mesh, u_prev, d_geo) -> np.ndarray:
    values = np.zeros((state.sample_count,), dtype=float)
    for idx in range(int(state.sample_count)):
        _, grad_u_prev = _eval_vector_with_grad_on_sample(state=state, dh=dh, mesh=mesh, vector=u_prev, sample_idx=idx)
        _, grad_d_geo = _eval_vector_with_grad_on_sample(state=state, dh=dh, mesh=mesh, vector=d_geo, sample_idx=idx)
        F_geo = np.eye(2, dtype=float) + np.asarray(grad_d_geo, dtype=float)
        J_geo = float(np.linalg.det(F_geo))
        cof_F_geo = J_geo * np.linalg.inv(F_geo).T
        values[idx] = -float(np.sum(cof_F_geo * np.asarray(grad_u_prev, dtype=float)) / J_geo)
    return values.reshape(state.n_elements, state.n_qp_per_element)


def _manual_predicted_subscale(
    *,
    state,
    dh,
    mesh,
    u_k,
    u_prev,
    a_prev,
    p_k,
    d_mesh,
    d_prev,
    rho_f: float,
    mu_f: float,
    dt: float,
    bossak_alpha: float,
    max_iterations: int,
    rel_tol: float,
    abs_tol: float,
    use_oss: bool = False,
) -> np.ndarray:
    c1 = 8.0
    c2 = 2.0
    dt_value = max(float(dt), 1.0e-14)
    bossak = _bossak_coefficients(alpha=float(bossak_alpha), dt=dt_value)
    expected = state._reshape_vector_quadrature(state.predicted_subscale_velocity).copy()
    for idx in range(int(state.sample_count)):
        u_val, grad_u = _eval_vector_with_grad_on_sample(state=state, dh=dh, mesh=mesh, vector=u_k, sample_idx=idx)
        u_prev_val, _ = _eval_vector_with_grad_on_sample(state=state, dh=dh, mesh=mesh, vector=u_prev, sample_idx=idx)
        a_prev_val, _ = _eval_vector_with_grad_on_sample(state=state, dh=dh, mesh=mesh, vector=a_prev, sample_idx=idx)
        _, grad_p = _eval_scalar_with_grad_on_sample(state=state, dh=dh, mesh=mesh, scalar=p_k, sample_idx=idx)
        d_val, grad_d = _eval_vector_with_grad_on_sample(state=state, dh=dh, mesh=mesh, vector=d_mesh, sample_idx=idx)
        d_prev_val, _ = _eval_vector_with_grad_on_sample(state=state, dh=dh, mesh=mesh, vector=d_prev, sample_idx=idx)

        F = np.eye(2, dtype=float) + np.asarray(grad_d, dtype=float)
        Finv = np.linalg.inv(F)
        grad_u_phys = np.asarray(grad_u, dtype=float) @ Finv
        grad_p_phys = Finv.T @ np.asarray(grad_p, dtype=float)
        w_mesh = (np.asarray(d_val, dtype=float) - np.asarray(d_prev_val, dtype=float)) / dt_value
        resolved_conv_velocity = np.asarray(u_val, dtype=float) - w_mesh
        a_curr = float(bossak["ma0"]) * (np.asarray(u_val, dtype=float) - np.asarray(u_prev_val, dtype=float)) + float(
            bossak["ma2"]
        ) * np.asarray(a_prev_val, dtype=float)
        a_relaxed = a_curr
        old_subscale = np.asarray(state.old_subscale_velocity[idx], dtype=float)
        momentum_proj = (
            np.asarray(state.momentum_projection[idx], dtype=float)
            if bool(use_oss)
            else np.zeros((2,), dtype=float)
        )
        static_residual = -(
            float(rho_f) * a_relaxed
            + float(rho_f) * (grad_u_phys @ resolved_conv_velocity)
            + grad_p_phys
            + momentum_proj
        ) + (float(rho_f) / dt_value) * old_subscale

        eid = int(state.sample_element_ids[idx])
        q = idx - eid * int(state.n_qp_per_element)
        h_values = _kratos_dvms_current_element_size_array(mesh, dh, d_mesh)
        h = max(float(h_values[eid]), 1.0e-14)
        pred0 = float(expected[eid, q, 0])
        pred1 = float(expected[eid, q, 1])
        converged = False
        for _ in range(max(int(max_iterations), 1)):
            fc0 = resolved_conv_velocity[0] + pred0
            fc1 = resolved_conv_velocity[1] + pred1
            conv_norm = math.sqrt(fc0 * fc0 + fc1 * fc1)
            inv_tau = c1 * float(mu_f) / (h * h) + float(rho_f) * (1.0 / dt_value + c2 * conv_norm / h)
            a11 = float(rho_f) * grad_u_phys[0, 0] + inv_tau
            a12 = float(rho_f) * grad_u_phys[0, 1]
            a21 = float(rho_f) * grad_u_phys[1, 0]
            a22 = float(rho_f) * grad_u_phys[1, 1] + inv_tau
            rhs0 = static_residual[0] - (a11 * pred0 + a12 * pred1)
            rhs1 = static_residual[1] - (a21 * pred0 + a22 * pred1)
            residual_norm_sq = rhs0 * rhs0 + rhs1 * rhs1
            det_a = a11 * a22 - a12 * a21
            if abs(det_a) <= 1.0e-20:
                pred0 = 0.0
                pred1 = 0.0
                break
            delta0 = (a22 * rhs0 - a12 * rhs1) / det_a
            delta1 = (-a21 * rhs0 + a11 * rhs1) / det_a
            pred0 += delta0
            pred1 += delta1
            velocity_error = delta0 * delta0 + delta1 * delta1
            norm_u_sq = pred0 * pred0 + pred1 * pred1
            if norm_u_sq > rel_tol:
                velocity_error /= norm_u_sq
            if velocity_error <= rel_tol or residual_norm_sq <= abs_tol:
                converged = True
                break
        if converged:
            expected[eid, q, 0] = pred0
            expected[eid, q, 1] = pred1
        else:
            expected[eid, q, :] = 0.0
    return expected


def test_form_compiler_evaluates_dvms_symbolics_on_p2_triangles() -> None:
    mesh, dh, state, u_k, u_prev, a_prev, p_k, d_mesh, d_prev = _make_dvms_problem()
    compiler = FormCompiler(dh, quadrature_order=state.quadrature_order, backend="python")
    dt = 0.2
    rho = 1.5
    bossak = _bossak_coefficients(alpha=-0.2, dt=dt)

    predictor_symbolics = build_fluid_dvms_predictor_symbolics(
        u_k=u_k,
        u_prev=u_prev,
        a_prev=a_prev,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        dt=dt,
        bossak_ma0=float(bossak["ma0"]),
        bossak_ma2=float(bossak["ma2"]),
        bossak_alpha=float(bossak["alpha"]),
        rho=rho,
        old_subscale=state.coefficient("old_subscale_velocity"),
        momentum_projection=state.coefficient("momentum_projection"),
    )
    results = compiler.evaluate_volume_expressions_on_quadrature(
        {
            "old_mass_residual": build_fluid_dvms_old_mass_residual(u_prev=u_prev, d_prev=d_prev, d_geo=d_mesh),
            "grad_u_phys": predictor_symbolics.kinematics.grad_u_phys,
            "resolved_conv_velocity": predictor_symbolics.kinematics.resolved_conv_velocity,
            "static_residual": predictor_symbolics.static_residual,
        },
        layout=state.quadrature_layout,
    )

    expected_old_mass = _manual_old_mass_residual(state=state, dh=dh, mesh=mesh, u_prev=u_prev, d_geo=d_mesh)
    expected_grad_u = np.zeros_like(results["grad_u_phys"])
    expected_resolved_conv = np.zeros_like(results["resolved_conv_velocity"])
    expected_static_residual = np.zeros_like(results["static_residual"])

    for idx in range(int(state.sample_count)):
        eid = int(state.sample_element_ids[idx])
        q = idx - eid * int(state.n_qp_per_element)
        u_val, grad_u = _eval_vector_with_grad_on_sample(state=state, dh=dh, mesh=mesh, vector=u_k, sample_idx=idx)
        u_prev_val, _ = _eval_vector_with_grad_on_sample(state=state, dh=dh, mesh=mesh, vector=u_prev, sample_idx=idx)
        a_prev_val, _ = _eval_vector_with_grad_on_sample(state=state, dh=dh, mesh=mesh, vector=a_prev, sample_idx=idx)
        _, grad_p = _eval_scalar_with_grad_on_sample(state=state, dh=dh, mesh=mesh, scalar=p_k, sample_idx=idx)
        d_val, grad_d = _eval_vector_with_grad_on_sample(state=state, dh=dh, mesh=mesh, vector=d_mesh, sample_idx=idx)
        d_prev_val, _ = _eval_vector_with_grad_on_sample(state=state, dh=dh, mesh=mesh, vector=d_prev, sample_idx=idx)

        F = np.eye(2, dtype=float) + np.asarray(grad_d, dtype=float)
        Finv = np.linalg.inv(F)
        grad_u_phys = np.asarray(grad_u, dtype=float) @ Finv
        grad_p_phys = Finv.T @ np.asarray(grad_p, dtype=float)
        resolved_conv = np.asarray(u_val, dtype=float) - (np.asarray(d_val, dtype=float) - np.asarray(d_prev_val, dtype=float)) / dt
        a_curr = float(bossak["ma0"]) * (np.asarray(u_val, dtype=float) - np.asarray(u_prev_val, dtype=float)) + float(
            bossak["ma2"]
        ) * np.asarray(a_prev_val, dtype=float)
        a_relaxed = a_curr
        static_residual = -(
            rho * a_relaxed
            + rho * (grad_u_phys @ resolved_conv)
            + grad_p_phys
            + np.asarray(state.momentum_projection[idx], dtype=float)
        ) + (rho / dt) * np.asarray(state.old_subscale_velocity[idx], dtype=float)

        expected_grad_u[eid, q, :, :] = grad_u_phys
        expected_resolved_conv[eid, q, :] = resolved_conv
        expected_static_residual[eid, q, :] = static_residual

    np.testing.assert_allclose(results["old_mass_residual"], expected_old_mass, rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(results["grad_u_phys"], expected_grad_u, rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(results["resolved_conv_velocity"], expected_resolved_conv, rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(results["static_residual"], expected_static_residual, rtol=1.0e-11, atol=1.0e-11)


@pytest.mark.parametrize("backend", ["jit", "cpp"])
def test_form_compiler_evaluates_dvms_symbolics_on_p2_triangles_compiled_backends_match_python(
    backend: str,
    tmp_path,
    monkeypatch,
) -> None:
    if backend == "cpp":
        pytest.importorskip("pybind11")
        monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "cpp_quadrature_eval_cache"))

    mesh, dh, state, u_k, u_prev, a_prev, p_k, d_mesh, d_prev = _make_dvms_problem()
    dt = 0.2
    rho = 1.5
    bossak = _bossak_coefficients(alpha=-0.2, dt=dt)

    predictor_symbolics = build_fluid_dvms_predictor_symbolics(
        u_k=u_k,
        u_prev=u_prev,
        a_prev=a_prev,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        dt=dt,
        bossak_ma0=float(bossak["ma0"]),
        bossak_ma2=float(bossak["ma2"]),
        bossak_alpha=float(bossak["alpha"]),
        rho=rho,
        old_subscale=state.coefficient("old_subscale_velocity"),
        momentum_projection=state.coefficient("momentum_projection"),
    )
    expressions = {
        "old_mass_residual": build_fluid_dvms_old_mass_residual(u_prev=u_prev, d_prev=d_prev, d_geo=d_mesh),
        "grad_u_phys": predictor_symbolics.kinematics.grad_u_phys,
        "resolved_conv_velocity": predictor_symbolics.kinematics.resolved_conv_velocity,
        "static_residual": predictor_symbolics.static_residual,
    }

    results_ref = FormCompiler(dh, quadrature_order=state.quadrature_order, backend="python").evaluate_volume_expressions_on_quadrature(
        expressions,
        layout=state.quadrature_layout,
    )
    results_backend = FormCompiler(dh, quadrature_order=state.quadrature_order, backend=backend).evaluate_volume_expressions_on_quadrature(
        expressions,
        layout=state.quadrature_layout,
    )

    for key in expressions:
        np.testing.assert_allclose(results_backend[key], results_ref[key], rtol=1.0e-11, atol=1.0e-11)


def test_dvms_runtime_update_matches_manual_p2_pointwise_path() -> None:
    mesh, dh, state, u_k, u_prev, a_prev, p_k, d_mesh, d_prev = _make_dvms_problem()
    expected_old_mass = _manual_old_mass_residual(state=state, dh=dh, mesh=mesh, u_prev=u_prev, d_geo=d_mesh)

    _update_fluid_dvms_state_from_previous_step(
        state=state,
        dh=dh,
        mesh=mesh,
        u_prev=u_prev,
        d_prev=d_prev,
        d_geo=d_mesh,
    )
    np.testing.assert_allclose(
        state._reshape_scalar_quadrature(state.old_mass_residual),
        expected_old_mass,
        rtol=1.0e-11,
        atol=1.0e-11,
    )

    expected_predicted = _manual_predicted_subscale(
        state=state,
        dh=dh,
        mesh=mesh,
        u_k=u_k,
        u_prev=u_prev,
        a_prev=a_prev,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        rho_f=1.5,
        mu_f=0.05,
        dt=0.2,
        bossak_alpha=-0.2,
        max_iterations=6,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    )

    _update_fluid_dvms_predicted_subscale(
        state=state,
        dh=dh,
        mesh=mesh,
        u_k=u_k,
        u_prev=u_prev,
        a_prev=a_prev,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        rho_f=1.5,
        mu_f=0.05,
        dt=0.2,
        bossak_alpha=-0.2,
        dynamic_tau=1.0,
        max_iterations=6,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    )

    np.testing.assert_allclose(
        state._reshape_vector_quadrature(state.predicted_subscale_velocity),
        expected_predicted,
        rtol=1.0e-11,
        atol=1.0e-11,
    )


def test_dvms_predictor_zeroes_capped_nonconverged_points() -> None:
    mesh, dh, state, u_k, u_prev, a_prev, p_k, d_mesh, d_prev = _make_dvms_problem()

    _update_fluid_dvms_predicted_subscale(
        state=state,
        dh=dh,
        mesh=mesh,
        u_k=u_k,
        u_prev=u_prev,
        a_prev=a_prev,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        rho_f=1.5,
        mu_f=0.05,
        dt=0.2,
        bossak_alpha=-0.2,
        dynamic_tau=1.0,
        max_iterations=1,
        rel_tol=0.0,
        abs_tol=0.0,
    )

    np.testing.assert_allclose(state.predicted_subscale_velocity, 0.0, atol=0.0)


@pytest.mark.parametrize("backend", ["jit", "cpp"])
def test_dvms_runtime_update_compiled_backends_match_python_p2(
    backend: str,
    tmp_path,
    monkeypatch,
) -> None:
    if backend == "cpp":
        pytest.importorskip("pybind11")
        monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "cpp_runtime_update_cache"))
    monkeypatch.delenv("PYCUTFEM_DVMS_CPP_FAST_PATH", raising=False)

    mesh_ref, dh_ref, state_ref, u_k_ref, u_prev_ref, a_prev_ref, p_k_ref, d_mesh_ref, d_prev_ref = _make_dvms_problem()
    mesh_backend, dh_backend, state_backend, u_k_backend, u_prev_backend, a_prev_backend, p_k_backend, d_mesh_backend, d_prev_backend = _make_dvms_problem()

    _update_fluid_dvms_state_from_previous_step(
        state=state_ref,
        dh=dh_ref,
        mesh=mesh_ref,
        u_prev=u_prev_ref,
        d_prev=d_prev_ref,
        d_geo=d_mesh_ref,
        backend="python",
    )
    _update_fluid_dvms_state_from_previous_step(
        state=state_backend,
        dh=dh_backend,
        mesh=mesh_backend,
        u_prev=u_prev_backend,
        d_prev=d_prev_backend,
        d_geo=d_mesh_backend,
        backend=backend,
    )
    np.testing.assert_allclose(
        state_backend._reshape_scalar_quadrature(state_backend.old_mass_residual),
        state_ref._reshape_scalar_quadrature(state_ref.old_mass_residual),
        rtol=1.0e-11,
        atol=1.0e-11,
    )

    _update_fluid_dvms_predicted_subscale(
        state=state_ref,
        dh=dh_ref,
        mesh=mesh_ref,
        u_k=u_k_ref,
        u_prev=u_prev_ref,
        a_prev=a_prev_ref,
        p_k=p_k_ref,
        d_mesh=d_mesh_ref,
        d_prev=d_prev_ref,
        rho_f=1.5,
        mu_f=0.05,
        dt=0.2,
        bossak_alpha=-0.2,
        dynamic_tau=1.0,
        max_iterations=6,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
        backend="python",
    )
    _update_fluid_dvms_predicted_subscale(
        state=state_backend,
        dh=dh_backend,
        mesh=mesh_backend,
        u_k=u_k_backend,
        u_prev=u_prev_backend,
        a_prev=a_prev_backend,
        p_k=p_k_backend,
        d_mesh=d_mesh_backend,
        d_prev=d_prev_backend,
        rho_f=1.5,
        mu_f=0.05,
        dt=0.2,
        bossak_alpha=-0.2,
        dynamic_tau=1.0,
        max_iterations=6,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
        backend=backend,
    )
    np.testing.assert_allclose(
        state_backend._reshape_vector_quadrature(state_backend.predicted_subscale_velocity),
        state_ref._reshape_vector_quadrature(state_ref.predicted_subscale_velocity),
        rtol=1.0e-11,
        atol=1.0e-11,
    )


@pytest.mark.parametrize("backend", ["python", "jit", "cpp"])
def test_dvms_runtime_operator_matches_update_helper(
    backend: str,
    tmp_path,
    monkeypatch,
) -> None:
    if backend == "cpp":
        pytest.importorskip("pybind11")
        monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "cpp_runtime_operator_cache"))
    monkeypatch.delenv("PYCUTFEM_DVMS_CPP_FAST_PATH", raising=False)

    mesh_op, dh_op, state_op, u_k_op, u_prev_op, a_prev_op, p_k_op, d_mesh_op, d_prev_op = _make_dvms_problem()
    mesh_ref, dh_ref, state_ref, u_k_ref, u_prev_ref, a_prev_ref, p_k_ref, d_mesh_ref, d_prev_ref = _make_dvms_problem()

    op = FluidDVMSSolverOperator(
        state=state_op,
        dh=dh_op,
        mesh=mesh_op,
        u_k=u_k_op,
        u_prev=u_prev_op,
        a_prev=a_prev_op,
        p_k=p_k_op,
        d_mesh=d_mesh_op,
        d_prev=d_prev_op,
        rho_f=1.5,
        mu_f=0.05,
        dt=0.2,
        bossak_alpha=-0.2,
        dynamic_tau=1.0,
        max_iterations=6,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    )
    solver = type("SolverStub", (), {"backend": backend})()
    initial_predicted = state_op._reshape_vector_quadrature(state_op.predicted_subscale_velocity).copy()
    op.before_assembly(solver=solver, coeffs={"unused": True}, need_matrix=False)
    np.testing.assert_allclose(
        state_op._reshape_vector_quadrature(state_op.predicted_subscale_velocity),
        initial_predicted,
        rtol=0.0,
        atol=0.0,
    )
    op.before_assembly(solver=solver, coeffs={"unused": True}, need_matrix=True)

    _update_fluid_dvms_predicted_subscale(
        state=state_ref,
        dh=dh_ref,
        mesh=mesh_ref,
        u_k=u_k_ref,
        u_prev=u_prev_ref,
        a_prev=a_prev_ref,
        p_k=p_k_ref,
        d_mesh=d_mesh_ref,
        d_prev=d_prev_ref,
        rho_f=1.5,
        mu_f=0.05,
        dt=0.2,
        bossak_alpha=-0.2,
        dynamic_tau=1.0,
        max_iterations=6,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
        backend=backend,
    )

    np.testing.assert_allclose(
        state_op._reshape_vector_quadrature(state_op.predicted_subscale_velocity),
        state_ref._reshape_vector_quadrature(state_ref.predicted_subscale_velocity),
        rtol=1.0e-11,
        atol=1.0e-11,
    )


@pytest.mark.parametrize("contribution_mode", ["velocity", "system"])
@pytest.mark.parametrize("backend", ["jit", "cpp"])
def test_dvms_symbolic_local_batches_compiled_backends_match_python(
    contribution_mode: str,
    backend: str,
    tmp_path,
    monkeypatch,
) -> None:
    if backend == "cpp":
        pytest.importorskip("pybind11")
        monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"cpp_dvms_local_batch_{contribution_mode}"))

    mesh_ref, dh_ref, state_ref, u_k_ref, u_prev_ref, a_prev_ref, p_k_ref, d_mesh_ref, d_prev_ref = _make_dvms_problem(
        poly_order=1,
        pressure_order=1,
        quadrature_order=6,
    )
    kwargs = dict(
        mesh=mesh_ref,
        dh=dh_ref,
        u_k=u_k_ref,
        u_prev=u_prev_ref,
        a_prev=a_prev_ref,
        p_k=p_k_ref,
        d_mesh=d_mesh_ref,
        d_prev=d_prev_ref,
        state=state_ref,
        rho_f=1.5,
        mu_f=0.05,
        dt=0.2,
        bossak_alpha=-0.2,
        quadrature_order=6,
        contribution_mode=contribution_mode,
    )

    batch_ref = assemble_fluid_dvms_local_contribution_batch(**kwargs, backend="python")
    batch_backend = assemble_fluid_dvms_local_contribution_batch(**kwargs, backend=backend)

    np.testing.assert_array_equal(batch_backend.element_ids, batch_ref.element_ids)
    np.testing.assert_array_equal(batch_backend.gdofs_map, batch_ref.gdofs_map)
    np.testing.assert_allclose(batch_backend.K_elem, batch_ref.K_elem, rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(batch_backend.F_elem, batch_ref.F_elem, rtol=1.0e-11, atol=1.0e-11)


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_dvms_local_system_batch_matches_single_element_on_structured_2x2(
    backend: str,
    tmp_path,
    monkeypatch,
) -> None:
    if backend == "cpp":
        pytest.importorskip("pybind11")
        monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "cpp_dvms_local_system_structured_2x2"))

    mesh_ref, dh_ref, state_ref, u_k_ref, u_prev_ref, a_prev_ref, p_k_ref, d_mesh_ref, d_prev_ref = _make_dvms_problem(
        poly_order=1,
        pressure_order=1,
        quadrature_order=1,
        nx_quads=2,
        ny_quads=2,
    )
    kwargs = dict(
        mesh=mesh_ref,
        dh=dh_ref,
        u_k=u_k_ref,
        u_prev=u_prev_ref,
        a_prev=a_prev_ref,
        p_k=p_k_ref,
        d_mesh=d_mesh_ref,
        d_prev=d_prev_ref,
        state=state_ref,
        rho_f=1.5,
        mu_f=0.05,
        dt=0.2,
        bossak_alpha=-0.2,
        quadrature_order=1,
        contribution_mode="system",
    )

    element_ids = np.arange(int(mesh_ref.n_elements), dtype=int)
    batch = assemble_fluid_dvms_local_contribution_batch(**kwargs, element_ids=element_ids, backend=backend)
    K_batch, F_batch, gdofs_batch = _compress_batch_to_fluid_block(dh_ref, batch)

    for idx, eid in enumerate(element_ids.tolist()):
        K_elem, F_elem, gdofs_elem = assemble_dvms_calculate_local_system(
            mesh=mesh_ref,
            dh=dh_ref,
            eid=int(eid),
            u_k=u_k_ref,
            u_prev=u_prev_ref,
            a_prev=a_prev_ref,
            p_k=p_k_ref,
            d_mesh=d_mesh_ref,
            d_prev=d_prev_ref,
            state=state_ref,
            rho_f=1.5,
            mu_f=0.05,
            dt=0.2,
            bossak_alpha=-0.2,
            quadrature_order=1,
            backend=backend,
        )
        np.testing.assert_array_equal(gdofs_batch[idx], gdofs_elem)
        np.testing.assert_allclose(K_batch[idx], K_elem, rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(F_batch[idx], F_elem, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_dvms_system_rhs_uses_bossak_scheme_acceleration(
    backend: str,
    tmp_path,
    monkeypatch,
) -> None:
    if backend == "cpp":
        pytest.importorskip("pybind11")
        monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "cpp_dvms_system_rhs_bossak"))

    mesh, dh, state, u_k, u_prev, a_prev, p_k, d_mesh, d_prev = _make_dvms_problem(
        poly_order=1,
        pressure_order=1,
        quadrature_order=1,
        nx_quads=2,
        ny_quads=2,
    )
    rho_f = 1.5
    mu_f = 0.05
    dt = 0.2
    alpha = -0.2
    bossak = _bossak_coefficients(alpha=alpha, dt=dt)

    A_vel, b_vel = _assemble_global_fluid_block_for_mode(
        mesh=mesh,
        dh=dh,
        state=state,
        u_k=u_k,
        u_prev=u_prev,
        a_prev=a_prev,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        rho_f=rho_f,
        mu_f=mu_f,
        dt=dt,
        bossak_alpha=alpha,
        quadrature_order=1,
        contribution_mode="velocity",
        backend=backend,
    )
    A_sys, b_sys = _assemble_global_fluid_block_for_mode(
        mesh=mesh,
        dh=dh,
        state=state,
        u_k=u_k,
        u_prev=u_prev,
        a_prev=a_prev,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        rho_f=rho_f,
        mu_f=mu_f,
        dt=dt,
        bossak_alpha=alpha,
        quadrature_order=1,
        contribution_mode="system",
        backend=backend,
    )
    A_mass_lhs, _ = _assemble_global_fluid_block_for_mode(
        mesh=mesh,
        dh=dh,
        state=state,
        u_k=u_k,
        u_prev=u_prev,
        a_prev=a_prev,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        rho_f=rho_f,
        mu_f=mu_f,
        dt=dt,
        bossak_alpha=alpha,
        quadrature_order=1,
        contribution_mode="mass_lhs",
        backend=backend,
    )
    A_mass_stab, _ = _assemble_global_fluid_block_for_mode(
        mesh=mesh,
        dh=dh,
        state=state,
        u_k=u_k,
        u_prev=u_prev,
        a_prev=a_prev,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        rho_f=rho_f,
        mu_f=mu_f,
        dt=dt,
        bossak_alpha=alpha,
        quadrature_order=1,
        contribution_mode="mass_stabilization",
        backend=backend,
    )

    a_scheme = np.zeros((int(dh.total_dofs),), dtype=float)
    ux_ids = np.asarray(dh.get_field_slice("ux"), dtype=int)
    uy_ids = np.asarray(dh.get_field_slice("uy"), dtype=int)
    ux_curr = np.asarray(u_k.components[0].get_nodal_values(ux_ids), dtype=float)
    uy_curr = np.asarray(u_k.components[1].get_nodal_values(uy_ids), dtype=float)
    ux_prev = np.asarray(u_prev.components[0].get_nodal_values(ux_ids), dtype=float)
    uy_prev = np.asarray(u_prev.components[1].get_nodal_values(uy_ids), dtype=float)
    ax_prev = np.asarray(a_prev.components[0].get_nodal_values(ux_ids), dtype=float)
    ay_prev = np.asarray(a_prev.components[1].get_nodal_values(uy_ids), dtype=float)
    ax_curr = float(bossak["ma0"]) * (ux_curr - ux_prev) + float(bossak["ma2"]) * ax_prev
    ay_curr = float(bossak["ma0"]) * (uy_curr - uy_prev) + float(bossak["ma2"]) * ay_prev
    a_scheme[ux_ids] = (1.0 - alpha) * ax_curr + alpha * ax_prev
    a_scheme[uy_ids] = (1.0 - alpha) * ay_curr + alpha * ay_prev

    mass_matrix = np.asarray(A_mass_lhs, dtype=float) + np.asarray(A_mass_stab, dtype=float)
    expected_delta = -mass_matrix @ a_scheme
    np.testing.assert_allclose(
        np.asarray(b_sys, dtype=float) - np.asarray(b_vel, dtype=float),
        expected_delta,
        rtol=1.0e-11,
        atol=1.0e-11,
    )

    current_acc = np.zeros_like(a_scheme)
    current_acc[ux_ids] = ax_curr
    current_acc[uy_ids] = ay_curr
    wrong_delta = -mass_matrix @ current_acc
    assert float(np.max(np.abs(expected_delta - wrong_delta))) > 1.0e-6

    bossak_mam = float(bossak["mam"])
    np.testing.assert_allclose(
        np.asarray(A_sys, dtype=float),
        np.asarray(A_vel, dtype=float) + bossak_mam * mass_matrix,
        rtol=1.0e-11,
        atol=1.0e-11,
    )
