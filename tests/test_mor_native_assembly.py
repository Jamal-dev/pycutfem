from __future__ import annotations

import sys

import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.mor.native_assembly import (
    call_native_kernel,
    gnat_system_from_native_kernel,
    native_kernel_metadata_from_runner,
    sampled_galerkin_reduced_system_from_native_kernel,
    sampled_lspg_rows_from_native_kernel,
)
from pycutfem.mor.reduced_assembly import (
    apply_gnat_lift,
    sampled_galerkin_reduced_system_from_local_blocks,
    sampled_lspg_rows_from_local_blocks,
)
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import Constant, Grad, Inner, TestFunction, TrialFunction
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_triangles


def _have_cpp_backend() -> bool:
    try:
        import pybind11  # noqa: F401

        return True
    except Exception:
        return False


def _make_problem():
    nodes, elems, edges, corners = structured_triangles(
        1.0,
        1.0,
        nx_quads=1,
        ny_quads=1,
        poly_order=1,
    )
    mesh = Mesh(nodes, elems, edges, corners, element_type="tri", poly_order=1)
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")
    u = TrialFunction("u", dh)
    v = TestFunction("u", dh)
    form = (Inner(Grad(u), Grad(v)) + Constant(3.0) * u * v) * dx(metadata={"q": 4})
    rhs = Constant(2.0) * v * dx(metadata={"q": 4})
    return dh, Equation(form, rhs)


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_native_reduced_assembler_matches_pybind_and_python_projections(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "mor_native_assembly"))
    dh, equation = _make_problem()
    integral = equation.a.integrals[0]
    element_ids = np.arange(int(dh.mixed_element.mesh.n_elements), dtype=np.int32)
    compiler = FormCompiler(dh, quadrature_order=4, backend="cpp")
    runner, current_funcs, static_args, gdofs_map = compiler._prepare_volume_jit_kernel(
        integral,
        element_ids=element_ids,
        full_local_layout=True,
    )
    K_py, F_py, J_py = runner(current_funcs, static_args)
    metadata = native_kernel_metadata_from_runner(runner)

    module = sys.modules[getattr(runner.kernel, "__module__")]
    assert getattr(module, "NATIVE_HAS_RAW_ENTRYPOINT") is True

    K_native, F_native, J_native = call_native_kernel(
        metadata_capsule=metadata,
        param_order=runner.param_order,
        static_args=static_args,
    )
    np.testing.assert_allclose(K_native, np.asarray(K_py, dtype=float), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(F_native, np.asarray(F_py, dtype=float), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(J_native, np.asarray(J_py, dtype=float), rtol=1.0e-12, atol=1.0e-12)

    basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, -0.25],
            [-0.25, 0.5],
        ],
        dtype=float,
    )
    rows = np.array([0, 2, 3], dtype=int)
    weights = np.array([0.75, 1.5], dtype=float)

    residual_ref, trial_ref = sampled_lspg_rows_from_local_blocks(
        K_elem=np.asarray(K_py, dtype=float),
        raw_rhs_elem=-np.asarray(F_py, dtype=float),
        gdofs_map=gdofs_map,
        row_dofs=rows,
        trial_basis=basis,
        element_weights=weights,
        backend="python",
    )
    residual_native, trial_native = sampled_lspg_rows_from_native_kernel(
        metadata_capsule=metadata,
        param_order=runner.param_order,
        static_args=static_args,
        row_dofs=rows,
        trial_basis=basis,
        element_weights=weights,
    )
    np.testing.assert_allclose(residual_native, residual_ref, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(trial_native, trial_ref, rtol=1.0e-12, atol=1.0e-12)

    galerkin_res_ref, galerkin_tan_ref = sampled_galerkin_reduced_system_from_local_blocks(
        K_elem=np.asarray(K_py, dtype=float),
        residual_elem=np.asarray(F_py, dtype=float),
        gdofs_map=gdofs_map,
        trial_basis=basis,
        element_weights=weights,
        backend="python",
    )
    galerkin_res_native, galerkin_tan_native = sampled_galerkin_reduced_system_from_native_kernel(
        metadata_capsule=metadata,
        param_order=runner.param_order,
        static_args=static_args,
        trial_basis=basis,
        element_weights=weights,
    )
    np.testing.assert_allclose(galerkin_res_native, galerkin_res_ref, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(galerkin_tan_native, galerkin_tan_ref, rtol=1.0e-12, atol=1.0e-12)

    lift = np.array([[1.0, -0.25, 0.5], [0.0, 1.5, -1.0]], dtype=float)
    gnat_res_ref, gnat_trial_ref = apply_gnat_lift(
        sample_to_residual_coefficients=lift,
        sampled_residual=residual_ref,
        sampled_trial_jacobian=trial_ref,
        backend="python",
    )
    gnat_res_native, gnat_trial_native = gnat_system_from_native_kernel(
        metadata_capsule=metadata,
        param_order=runner.param_order,
        static_args=static_args,
        row_dofs=rows,
        trial_basis=basis,
        sample_to_residual_coefficients=lift,
        element_weights=weights,
    )
    np.testing.assert_allclose(gnat_res_native, gnat_res_ref, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(gnat_trial_native, gnat_trial_ref, rtol=1.0e-12, atol=1.0e-12)
