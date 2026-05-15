from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.mor.reduced_assembly import (
    AffineReducedState,
    ReducedLocalAssembler,
    apply_gnat_lift,
    constrained_reaction_rows_from_local_blocks,
    decode_element_values,
    decode_values_on_dofs,
    reduced_reaction_from_local_blocks,
    sampled_galerkin_element_contributions_from_local_blocks,
    sampled_galerkin_reduced_system_from_local_blocks,
    sampled_lspg_element_contributions_from_local_blocks,
    sampled_lspg_rows_from_local_blocks,
)
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import Constant, TestFunction, TrialFunction, grad, inner
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_triangles

try:
    import numba as _nb
except Exception:  # pragma: no cover - optional dependency
    _nb = None


def _have_cpp_backend() -> bool:
    try:
        import pybind11  # noqa: F401

        return True
    except Exception:
        return False


def _scatter_local_blocks(
    K_elem: np.ndarray,
    vector_elem: np.ndarray,
    gdofs_map: np.ndarray,
    total_dofs: int,
) -> tuple[np.ndarray, np.ndarray]:
    K_full = np.zeros((total_dofs, total_dofs), dtype=float)
    vector_full = np.zeros(total_dofs, dtype=float)
    for elem, gdofs in enumerate(gdofs_map):
        vector_full[gdofs] += vector_elem[elem]
        for local_i, global_i in enumerate(gdofs):
            K_full[global_i, gdofs] += K_elem[elem, local_i, :]
    return K_full, vector_full


def _make_scalar_problem():
    nodes, elems, edges, corners = structured_triangles(
        1.0,
        1.0,
        nx_quads=1,
        ny_quads=1,
        poly_order=1,
    )
    mesh = Mesh(
        nodes,
        elems,
        edges,
        corners,
        element_type="tri",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")
    u = TrialFunction("u", dh)
    v = TestFunction("u", dh)
    qmeta = {"q": 4}
    a = (inner(grad(u), grad(v)) + Constant(3.0) * u * v) * dx(metadata=qmeta)
    L = Constant(2.0) * v * dx(metadata=qmeta)
    return dh, Equation(a, L)


def test_affine_reduced_state_decodes_global_and_element_values() -> None:
    offset = np.array([1.0, -2.0, 0.5, 3.0])
    basis = np.array(
        [
            [1.0, 0.0],
            [0.5, -1.0],
            [2.0, 0.25],
            [-0.5, 1.5],
        ]
    )
    coeffs = np.array([0.25, -2.0])
    state = AffineReducedState(basis=basis, offset=offset)

    np.testing.assert_allclose(state.values_on_dofs(np.array([3, 1]), coeffs), (offset + basis @ coeffs)[[3, 1]])
    np.testing.assert_allclose(
        state.element_values(np.array([[0, 2], [1, 3]], dtype=int), coeffs),
        (offset + basis @ coeffs)[np.array([[0, 2], [1, 3]], dtype=int)],
    )


def test_reduced_local_block_projections_match_full_scatter() -> None:
    K_elem = np.array(
        [
            [[2.0, -1.0, 0.5], [0.0, 3.0, -0.25], [1.5, 0.0, 1.0]],
            [[-1.0, 0.2, 0.0], [0.75, 1.25, -0.5], [0.0, -2.0, 4.0]],
        ]
    )
    raw_rhs_elem = np.array([[1.0, -2.0, 0.5], [3.0, -1.0, 2.0]])
    gdofs_map = np.array([[0, 2, 4], [2, 3, 5]], dtype=int)
    basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.5, -1.0],
            [1.5, 0.25],
            [-0.5, 2.0],
            [0.25, -0.75],
        ]
    )
    rows = np.array([2, 4, 5], dtype=int)
    weights = np.array([0.25, 2.0])

    residual_rows, trial_rows = sampled_lspg_rows_from_local_blocks(
        K_elem=K_elem,
        raw_rhs_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        row_dofs=rows,
        trial_basis=basis,
        element_weights=weights,
    )
    reduced_residual, reduced_tangent = sampled_galerkin_reduced_system_from_local_blocks(
        K_elem=K_elem,
        residual_elem=-raw_rhs_elem,
        gdofs_map=gdofs_map,
        trial_basis=basis,
        element_weights=weights,
    )

    K_full, raw_rhs_full = _scatter_local_blocks(
        K_elem * weights[:, None, None],
        raw_rhs_elem * weights[:, None],
        gdofs_map,
        total_dofs=basis.shape[0],
    )
    np.testing.assert_allclose(residual_rows, -raw_rhs_full[rows])
    np.testing.assert_allclose(trial_rows, K_full[rows, :] @ basis)
    np.testing.assert_allclose(reduced_residual, basis.T @ (-raw_rhs_full))
    np.testing.assert_allclose(reduced_tangent, basis.T @ K_full @ basis)


def test_reduced_reaction_projection_matches_full_scatter() -> None:
    raw_rhs_elem = np.array([[1.0, -2.0], [0.5, 4.0]])
    gdofs_map = np.array([[0, 3], [3, 5]], dtype=int)
    constrained_rows = np.array([3, 5], dtype=int)
    row_to_load = np.array([[2.0, -1.0], [0.5, 0.25]])

    rows, reaction = constrained_reaction_rows_from_local_blocks(
        raw_rhs_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        constrained_row_dofs=constrained_rows,
    )
    reduced = reduced_reaction_from_local_blocks(
        raw_rhs_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        constrained_row_dofs=constrained_rows,
        row_to_reduced_load=row_to_load,
    )

    _K_full, rhs_full = _scatter_local_blocks(
        np.zeros((2, 2, 2), dtype=float),
        raw_rhs_elem,
        gdofs_map,
        total_dofs=6,
    )
    np.testing.assert_array_equal(rows, constrained_rows)
    np.testing.assert_allclose(reaction, -rhs_full[constrained_rows])
    np.testing.assert_allclose(reduced, row_to_load @ reaction)


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_cpp_reduced_projection_helpers_match_python(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "mor_cpp_projection_helpers"))
    K_elem = np.array(
        [
            [[2.0, -1.0, 0.5], [0.0, 3.0, -0.25], [1.5, 0.0, 1.0]],
            [[-1.0, 0.2, 0.0], [0.75, 1.25, -0.5], [0.0, -2.0, 4.0]],
            [[0.5, 1.0, -1.5], [2.0, -0.75, 0.25], [1.25, 0.5, 3.0]],
        ],
        dtype=float,
    )
    raw_rhs_elem = np.array([[1.0, -2.0, 0.5], [3.0, -1.0, 2.0], [-0.5, 1.5, -3.0]])
    residual_elem = -raw_rhs_elem
    gdofs_map = np.array([[0, 2, 4], [2, 3, 5], [1, 4, 5]], dtype=int)
    basis = np.array(
        [
            [1.0, 0.0],
            [0.25, 1.5],
            [0.5, -1.0],
            [1.5, 0.25],
            [-0.5, 2.0],
            [0.25, -0.75],
        ],
        dtype=float,
    )
    offset = np.linspace(-1.0, 1.0, basis.shape[0])
    coefficients = np.array([0.5, -1.25], dtype=float)
    rows = np.array([2, 4, 5], dtype=int)
    weights = np.array([0.25, 2.0, 0.75], dtype=float)

    np.testing.assert_allclose(
        decode_values_on_dofs(offset=offset, basis=basis, dofs=rows, coefficients=coefficients, backend="cpp"),
        decode_values_on_dofs(offset=offset, basis=basis, dofs=rows, coefficients=coefficients, backend="python"),
    )
    np.testing.assert_allclose(
        decode_element_values(offset=offset, basis=basis, local_map=gdofs_map, coefficients=coefficients, backend="cpp"),
        decode_element_values(offset=offset, basis=basis, local_map=gdofs_map, coefficients=coefficients, backend="python"),
    )

    residual_py, trial_py = sampled_lspg_element_contributions_from_local_blocks(
        K_elem=K_elem,
        raw_rhs_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        row_dofs=rows,
        trial_basis=basis,
        backend="python",
    )
    residual_cpp, trial_cpp = sampled_lspg_element_contributions_from_local_blocks(
        K_elem=K_elem,
        raw_rhs_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        row_dofs=rows,
        trial_basis=basis,
        backend="cpp",
    )
    np.testing.assert_allclose(residual_cpp, residual_py)
    np.testing.assert_allclose(trial_cpp, trial_py)

    lspg_residual_py, lspg_trial_py = sampled_lspg_rows_from_local_blocks(
        K_elem=K_elem,
        raw_rhs_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        row_dofs=rows,
        trial_basis=basis,
        element_weights=weights,
        backend="python",
    )
    lspg_residual_cpp, lspg_trial_cpp = sampled_lspg_rows_from_local_blocks(
        K_elem=K_elem,
        raw_rhs_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        row_dofs=rows,
        trial_basis=basis,
        element_weights=weights,
        backend="cpp",
    )
    np.testing.assert_allclose(lspg_residual_cpp, lspg_residual_py)
    np.testing.assert_allclose(lspg_trial_cpp, lspg_trial_py)

    residual_py, tangent_py = sampled_galerkin_element_contributions_from_local_blocks(
        K_elem=K_elem,
        residual_elem=residual_elem,
        gdofs_map=gdofs_map,
        trial_basis=basis,
        backend="python",
    )
    residual_cpp, tangent_cpp = sampled_galerkin_element_contributions_from_local_blocks(
        K_elem=K_elem,
        residual_elem=residual_elem,
        gdofs_map=gdofs_map,
        trial_basis=basis,
        backend="cpp",
    )
    np.testing.assert_allclose(residual_cpp, residual_py)
    np.testing.assert_allclose(tangent_cpp, tangent_py)

    reduced_res_py, reduced_tan_py = sampled_galerkin_reduced_system_from_local_blocks(
        K_elem=K_elem,
        residual_elem=residual_elem,
        gdofs_map=gdofs_map,
        trial_basis=basis,
        element_weights=weights,
        backend="python",
    )
    reduced_res_cpp, reduced_tan_cpp = sampled_galerkin_reduced_system_from_local_blocks(
        K_elem=K_elem,
        residual_elem=residual_elem,
        gdofs_map=gdofs_map,
        trial_basis=basis,
        element_weights=weights,
        backend="cpp",
    )
    np.testing.assert_allclose(reduced_res_cpp, reduced_res_py)
    np.testing.assert_allclose(reduced_tan_cpp, reduced_tan_py)

    constrained_rows = np.array([3, 5], dtype=int)
    row_to_load = np.array([[2.0, -1.0], [0.5, 0.25]], dtype=float)
    rows_py, reaction_py = constrained_reaction_rows_from_local_blocks(
        raw_rhs_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        constrained_row_dofs=constrained_rows,
        element_weights=weights,
        backend="python",
    )
    rows_cpp, reaction_cpp = constrained_reaction_rows_from_local_blocks(
        raw_rhs_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        constrained_row_dofs=constrained_rows,
        element_weights=weights,
        backend="cpp",
    )
    np.testing.assert_array_equal(rows_cpp, rows_py)
    np.testing.assert_allclose(reaction_cpp, reaction_py)
    np.testing.assert_allclose(
        reduced_reaction_from_local_blocks(
            raw_rhs_elem=raw_rhs_elem,
            gdofs_map=gdofs_map,
            constrained_row_dofs=constrained_rows,
            row_to_reduced_load=row_to_load,
            element_weights=weights,
            backend="cpp",
        ),
        reduced_reaction_from_local_blocks(
            raw_rhs_elem=raw_rhs_elem,
            gdofs_map=gdofs_map,
            constrained_row_dofs=constrained_rows,
            row_to_reduced_load=row_to_load,
            element_weights=weights,
            backend="python",
        ),
    )

    lift = np.array([[1.0, -0.5, 0.25], [0.0, 2.0, -1.0]], dtype=float)
    gnat_res_py, gnat_trial_py = apply_gnat_lift(
        sample_to_residual_coefficients=lift,
        sampled_residual=lspg_residual_py,
        sampled_trial_jacobian=lspg_trial_py,
        backend="python",
    )
    gnat_res_cpp, gnat_trial_cpp = apply_gnat_lift(
        sample_to_residual_coefficients=lift,
        sampled_residual=lspg_residual_py,
        sampled_trial_jacobian=lspg_trial_py,
        backend="cpp",
    )
    np.testing.assert_allclose(gnat_res_cpp, gnat_res_py)
    np.testing.assert_allclose(gnat_trial_cpp, gnat_trial_py)


@pytest.mark.parametrize(
    "backend",
    ["python"] + (["jit"] if _nb is not None else []) + (["cpp"] if _have_cpp_backend() else []),
)
def test_reduced_local_assembler_projects_ufl_local_blocks(
    backend: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"mor_reduced_assembler_{backend}"))
    dh, eq = _make_scalar_problem()
    basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, -0.25],
            [-0.25, 0.5],
        ],
        dtype=float,
    )

    A_ref, R_ref = FormCompiler(dh, quadrature_order=4, backend="python").assemble(eq, bcs=[])
    A_dense = A_ref.toarray()
    rows = np.array([0, 2, 3], dtype=int)
    assembler = ReducedLocalAssembler(
        dof_handler=dh,
        form_or_equation=eq,
        trial_basis=basis,
        quadrature_order=4,
        backend=backend,
    )

    reduced_residual, reduced_tangent = assembler.galerkin_system()
    sampled_residual, sampled_trial = assembler.sampled_lspg_rows(row_dofs=rows)

    np.testing.assert_allclose(reduced_residual, basis.T @ R_ref, rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(reduced_tangent, basis.T @ A_dense @ basis, rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(sampled_residual, R_ref[rows], rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(sampled_trial, A_dense[rows, :] @ basis, rtol=1.0e-11, atol=1.0e-11)
