from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.jit.cpp_backend.compiler import compile_extension
from pycutfem.nonmatching import (
    build_composite_mesh,
    build_nonmatching_interface,
    lift_nonmatching_interface_to_composite,
)
from pycutfem.operators import CallbackLocalAssemblyOperator, SymbolicLocalAssemblyOperator
from pycutfem.state import QuadratureLayout
from pycutfem.solvers.nonlinear_solver import NewtonSolver
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import Constant, Jump, TestFunction, TrialFunction, grad, inner
from pycutfem.ufl.forms import CondensedQuadratureLocalSystem, Equation
from pycutfem.ufl.measures import dInterface, dNonmatchingInterface, dS, dx
from pycutfem.utils.meshgen import structured_quad, structured_triangles

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


def _make_reduced_solver(backend: str) -> NewtonSolver:
    solver = NewtonSolver.__new__(NewtonSolver)
    solver.backend = str(backend)
    solver.full_to_red = np.asarray([0, 1, 2], dtype=int)
    return solver


def _make_identity_solver(*, backend: str, ndofs: int) -> NewtonSolver:
    solver = NewtonSolver.__new__(NewtonSolver)
    solver.backend = str(backend)
    solver.full_to_red = np.arange(int(ndofs), dtype=int)
    return solver


def _make_symbolic_scalar_problem():
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
    eq = Equation(a, L)
    return dh, eq


def _make_boundary_scalar_problem():
    nodes, elems, edges, corners = structured_quad(
        1.0,
        1.0,
        nx=1,
        ny=1,
        poly_order=1,
    )
    mesh = Mesh(
        nodes,
        elems,
        edges,
        corners,
        element_type="quad",
        poly_order=1,
    )
    mesh.tag_boundary_edges({"right_wall": lambda x, y: np.isclose(x, 1.0)})
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")
    u = TrialFunction("u", dh)
    v = TestFunction("u", dh)
    dSb = dS(defined_on=mesh.edge_bitset("right_wall"), metadata={"q": 4})
    eq = Equation(Constant(3.0) * u * v * dSb, Constant(2.0) * v * dSb)
    return dh, eq


def _make_cut_interface_scalar_problem():
    nodes, elems, edges, corners = structured_quad(
        1.0,
        1.0,
        nx=1,
        ny=1,
        poly_order=1,
    )
    mesh = Mesh(
        nodes,
        elems,
        edges,
        corners,
        element_type="quad",
        poly_order=1,
    )
    level_set = AffineLevelSet(a=1.0, b=1.0, c=-0.8)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")
    u = TrialFunction("u", dh)
    v = TestFunction("u", dh)
    dGamma = dInterface(defined_on=mesh.element_bitset("cut"), level_set=level_set, metadata={"q": 4})
    eq = Equation(u * v * dGamma, Constant(1.5) * v * dGamma)
    return dh, eq


def _make_aligned_interface_scalar_problem():
    nodes, elems, edges, corners = structured_quad(
        1.0,
        1.0,
        nx=2,
        ny=1,
        poly_order=1,
    )
    mesh = Mesh(
        nodes,
        elems,
        edges,
        corners,
        element_type="quad",
        poly_order=1,
    )
    level_set = AffineLevelSet(a=1.0, b=0.0, c=-0.5)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="dg")
    u = TrialFunction("u", dh)
    v = TestFunction("u", dh)
    dGamma = dInterface(level_set=level_set, metadata={"q": 4})
    eq = Equation(Jump(u) * Jump(v) * dGamma, None)
    return dh, eq


def _make_nonmatching_submesh(*, nx: int, ny: int, offset_x: float) -> Mesh:
    nodes, elems, edges, corners = structured_quad(
        0.5,
        1.0,
        nx=nx,
        ny=ny,
        poly_order=1,
        offset=(offset_x, 0.0),
    )
    mesh = Mesh(
        nodes,
        elems,
        edges,
        corners,
        element_type="quad",
        poly_order=1,
    )
    mesh.tag_boundary_edges(
        {
            "interface": lambda x, y: np.isclose(x, 0.5),
            "boundary": lambda x, y: True,
        }
    )
    return mesh


def _make_nonmatching_interface_scalar_problem():
    mesh_neg = _make_nonmatching_submesh(nx=2, ny=4, offset_x=0.0)
    mesh_pos = _make_nonmatching_submesh(nx=2, ny=5, offset_x=0.5)
    interface = build_nonmatching_interface(mesh_neg=mesh_neg, mesh_pos=mesh_pos)
    mapping = build_composite_mesh(mesh_pos=mesh_pos, mesh_neg=mesh_neg, order="pos_neg")
    interface_c = lift_nonmatching_interface_to_composite(interface=interface, mapping=mapping)

    dh = DofHandler(MixedElement(mapping.mesh, field_specs={"u": 1}), method="cg")
    u = TrialFunction("u", dh)
    v = TestFunction("u", dh)
    dGamma = dNonmatchingInterface(metadata={"q": 4, "interface": interface_c})
    eq = Equation(Jump(u) * Jump(v) * dGamma, Constant(1.5) * Jump(v) * dGamma)
    return dh, eq


def _workset_builder(*, solver, coeffs, need_matrix: bool):
    del solver
    workset = {
        "need_matrix": bool(need_matrix),
        "element_ids": np.asarray([0, 1], dtype=int),
        "gdofs_map": np.asarray([[0, 1], [1, 2]], dtype=int),
        "payload": {
            "weights": np.asarray(coeffs["weights"], dtype=float),
        },
    }
    if "backend_override" in coeffs:
        workset["backend"] = coeffs["backend_override"]
    return workset


def _python_local_kernel(workset):
    weights = np.asarray(workset.payload["weights"], dtype=float).reshape(-1)
    n_elem = int(weights.shape[0])
    n_loc = int(workset.gdofs_map.shape[1])
    K_elem = np.zeros((n_elem, n_loc, n_loc), dtype=float) if workset.need_matrix else None
    F_elem = np.zeros((n_elem, n_loc), dtype=float)
    for eid in range(n_elem):
        for i in range(n_loc):
            if K_elem is not None:
                K_elem[eid, i, i] = weights[eid]
            F_elem[eid, i] = weights[eid] * float(i + 1)
    return K_elem, F_elem


if _nb is not None:

    @_nb.njit(cache=True)
    def _jit_diag_kernel(weights: np.ndarray, n_loc: int, need_matrix: bool):
        n_elem = int(weights.shape[0])
        K_elem = np.zeros((n_elem, n_loc, n_loc), dtype=np.float64)
        F_elem = np.zeros((n_elem, n_loc), dtype=np.float64)
        for eid in range(n_elem):
            for i in range(n_loc):
                if need_matrix:
                    K_elem[eid, i, i] = weights[eid]
                F_elem[eid, i] = weights[eid] * float(i + 1)
        return K_elem, F_elem


def _jit_local_kernel(workset):
    if _nb is None:  # pragma: no cover - guarded in parametrized test
        raise RuntimeError("numba is not available.")
    K_elem, F_elem = _jit_diag_kernel(
        np.asarray(workset.payload["weights"], dtype=np.float64).reshape(-1),
        int(workset.gdofs_map.shape[1]),
        bool(workset.need_matrix),
    )
    if not workset.need_matrix:
        K_elem = None
    return K_elem, F_elem


def _compile_cpp_local_kernel(tmp_path: Path):
    module_name = "_pycutfem_cpp_local_operator_test"
    build_dir = tmp_path / "cpp_local_operator"
    build_dir.mkdir(parents=True, exist_ok=True)
    source_path = build_dir / f"{module_name}.cpp"
    source_path.write_text(
        f"""
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>

namespace py = pybind11;

py::tuple batch_diag_kernel(
    py::array_t<double, py::array::c_style | py::array::forcecast> weights,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> gdofs_map,
    bool need_matrix
) {{
    auto W = weights.unchecked<1>();
    auto G = gdofs_map.unchecked<2>();
    const py::ssize_t n_elem = G.shape(0);
    const py::ssize_t n_loc = G.shape(1);

    py::object K_obj = py::none();
    if (need_matrix) {{
        py::array_t<double> K_arr({{n_elem, n_loc, n_loc}});
        auto K = K_arr.mutable_unchecked<3>();
        for (py::ssize_t e = 0; e < n_elem; ++e) {{
            for (py::ssize_t i = 0; i < n_loc; ++i) {{
                for (py::ssize_t j = 0; j < n_loc; ++j) {{
                    K(e, i, j) = (i == j) ? W(e) : 0.0;
                }}
            }}
        }}
        K_obj = std::move(K_arr);
    }}

    py::array_t<double> F_arr({{n_elem, n_loc}});
    auto F = F_arr.mutable_unchecked<2>();
    for (py::ssize_t e = 0; e < n_elem; ++e) {{
        for (py::ssize_t i = 0; i < n_loc; ++i) {{
            F(e, i) = W(e) * double(i + 1);
        }}
    }}
    return py::make_tuple(K_obj, F_arr);
}}

PYBIND11_MODULE({module_name}, m) {{
    m.attr("CODEGEN_ABI") = "local-operator-test-v1";
    m.def("batch_diag_kernel", &batch_diag_kernel);
}}
        """,
        encoding="utf-8",
    )
    ext_path = compile_extension(module_name, source_path, build_dir)
    spec = importlib.util.spec_from_file_location(module_name, ext_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load compiled local-operator module {ext_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cpp_local_kernel_factory(tmp_path: Path):
    module = _compile_cpp_local_kernel(tmp_path)

    def _kernel(workset):
        K_elem, F_elem = module.batch_diag_kernel(
            np.asarray(workset.payload["weights"], dtype=float).reshape(-1),
            np.asarray(workset.gdofs_map, dtype=np.int64),
            bool(workset.need_matrix),
        )
        return (None if K_elem is None else np.asarray(K_elem, dtype=float), np.asarray(F_elem, dtype=float))

    return _kernel


def _expected_outputs() -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=float,
    )
    R = np.asarray([2.0, 7.0, 6.0], dtype=float)
    return A, R


def test_local_assembly_operator_python_reference_scatter() -> None:
    solver = _make_reduced_solver("python")
    op = CallbackLocalAssemblyOperator(
        workset_builder=_workset_builder,
        python_kernel=_python_local_kernel,
    )
    A_red = sp.lil_matrix((3, 3), dtype=float)
    R_red = np.zeros(3, dtype=float)

    A_red, R_red = op.after_assembly(
        solver=solver,
        coeffs={"weights": np.asarray([2.0, 3.0], dtype=float)},
        A_red=A_red,
        R_red=R_red,
        need_matrix=True,
    )

    A_exp, R_exp = _expected_outputs()
    assert np.allclose(A_red.toarray(), A_exp)
    assert np.allclose(R_red, R_exp)


@pytest.mark.parametrize(
    "backend",
    ["python"] + (["jit"] if _nb is not None else []) + (["cpp"] if _have_cpp_backend() else []),
)
def test_local_assembly_operator_backend_parity(backend: str, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"local_operator_cache_{backend}"))
    solver = _make_reduced_solver(backend)
    cpp_kernel = _cpp_local_kernel_factory(tmp_path) if backend == "cpp" else None
    op = CallbackLocalAssemblyOperator(
        workset_builder=_workset_builder,
        python_kernel=_python_local_kernel,
        jit_kernel=_jit_local_kernel if _nb is not None else None,
        cpp_kernel=cpp_kernel,
    )
    A_red = sp.lil_matrix((3, 3), dtype=float)
    R_red = np.zeros(3, dtype=float)

    A_red, R_red = op.after_assembly(
        solver=solver,
        coeffs={"weights": np.asarray([2.0, 3.0], dtype=float)},
        A_red=A_red,
        R_red=R_red,
        need_matrix=True,
    )

    A_exp, R_exp = _expected_outputs()
    assert np.allclose(A_red.toarray(), A_exp)
    assert np.allclose(R_red, R_exp)


def test_newton_solver_full_scatter_helper_adds_matrix_and_vector() -> None:
    solver = _make_reduced_solver("python")
    A_full = sp.lil_matrix((3, 3), dtype=float)
    R_full = np.zeros(3, dtype=float)

    A_full, R_full = NewtonSolver.scatter_element_contribs_full(
        solver,
        K_elem=np.asarray(
            [
                [[2.0, 0.0], [0.0, 2.0]],
                [[3.0, 0.0], [0.0, 3.0]],
            ],
            dtype=float,
        ),
        F_elem=np.asarray(
            [
                [2.0, 4.0],
                [3.0, 6.0],
            ],
            dtype=float,
        ),
        element_ids=np.asarray([0, 1], dtype=int),
        gdofs_map=np.asarray([[0, 1], [1, 2]], dtype=int),
        A_full=A_full,
        R_full=R_full,
    )

    A_exp, R_exp = _expected_outputs()
    assert np.allclose(A_full.toarray(), A_exp)
    assert np.allclose(R_full, R_exp)


@pytest.mark.parametrize(
    "backend",
    ["python"] + (["jit"] if _nb is not None else []) + (["cpp"] if _have_cpp_backend() else []),
)
def test_form_compiler_local_volume_contributions_match_global_assembly(
    backend: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"symbolic_local_batch_{backend}"))
    dh, eq = _make_symbolic_scalar_problem()

    A_ref, R_ref = FormCompiler(dh, quadrature_order=4, backend="python").assemble(eq, bcs=[])
    batch = FormCompiler(dh, quadrature_order=4, backend=backend).assemble_volume_local_contributions(eq)

    solver = _make_identity_solver(backend=backend, ndofs=dh.total_dofs)
    A_full = sp.lil_matrix((dh.total_dofs, dh.total_dofs), dtype=float)
    R_full = np.zeros(dh.total_dofs, dtype=float)
    A_full, R_full = NewtonSolver.scatter_element_contribs_full(
        solver,
        K_elem=batch.K_elem,
        F_elem=batch.F_elem,
        element_ids=batch.element_ids,
        gdofs_map=batch.gdofs_map,
        A_full=A_full,
        R_full=R_full,
    )

    assert np.allclose(A_full.toarray(), A_ref.toarray())
    assert np.allclose(R_full, R_ref)


@pytest.mark.parametrize(
    "backend",
    ["python"] + (["jit"] if _nb is not None else []) + (["cpp"] if _have_cpp_backend() else []),
)
def test_form_compiler_local_boundary_contributions_match_global_assembly(
    backend: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"symbolic_local_boundary_{backend}"))
    dh, eq = _make_boundary_scalar_problem()

    A_ref, R_ref = FormCompiler(dh, quadrature_order=4, backend="python").assemble(eq, bcs=[])
    batch = FormCompiler(dh, quadrature_order=4, backend=backend).assemble_local_contributions(eq)

    solver = _make_identity_solver(backend=backend, ndofs=dh.total_dofs)
    A_full = sp.lil_matrix((dh.total_dofs, dh.total_dofs), dtype=float)
    R_full = np.zeros(dh.total_dofs, dtype=float)
    A_full, R_full = NewtonSolver.scatter_element_contribs_full(
        solver,
        K_elem=batch.K_elem,
        F_elem=batch.F_elem,
        element_ids=batch.element_ids,
        gdofs_map=batch.gdofs_map,
        A_full=A_full,
        R_full=R_full,
    )

    assert np.allclose(A_full.toarray(), A_ref.toarray())
    assert np.allclose(R_full, R_ref)


@pytest.mark.parametrize(
    "backend",
    ["python"] + (["jit"] if _nb is not None else []) + (["cpp"] if _have_cpp_backend() else []),
)
def test_form_compiler_local_cut_interface_contributions_match_global_assembly(
    backend: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"symbolic_local_cut_interface_{backend}"))
    dh, eq = _make_cut_interface_scalar_problem()

    A_ref, R_ref = FormCompiler(dh, quadrature_order=4, backend="python").assemble(eq, bcs=[])
    batch = FormCompiler(dh, quadrature_order=4, backend=backend).assemble_local_contributions(eq)

    solver = _make_identity_solver(backend=backend, ndofs=dh.total_dofs)
    A_full = sp.lil_matrix((dh.total_dofs, dh.total_dofs), dtype=float)
    R_full = np.zeros(dh.total_dofs, dtype=float)
    A_full, R_full = NewtonSolver.scatter_element_contribs_full(
        solver,
        K_elem=batch.K_elem,
        F_elem=batch.F_elem,
        element_ids=batch.element_ids,
        gdofs_map=batch.gdofs_map,
        A_full=A_full,
        R_full=R_full,
    )

    assert np.allclose(A_full.toarray(), A_ref.toarray())
    assert np.allclose(R_full, R_ref)


@pytest.mark.parametrize(
    "backend",
    ["python"] + (["jit"] if _nb is not None else []) + (["cpp"] if _have_cpp_backend() else []),
)
def test_form_compiler_local_aligned_interface_contributions_match_global_assembly(
    backend: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"symbolic_local_aligned_interface_{backend}"))
    dh, eq = _make_aligned_interface_scalar_problem()

    A_ref, _ = FormCompiler(dh, quadrature_order=4, backend="python").assemble(eq, bcs=[])
    batch = FormCompiler(dh, quadrature_order=4, backend=backend).assemble_local_contributions(eq)

    solver = _make_identity_solver(backend=backend, ndofs=dh.total_dofs)
    A_full = sp.lil_matrix((dh.total_dofs, dh.total_dofs), dtype=float)
    R_full = np.zeros(dh.total_dofs, dtype=float)
    A_full, R_full = NewtonSolver.scatter_element_contribs_full(
        solver,
        K_elem=batch.K_elem,
        F_elem=batch.F_elem,
        element_ids=batch.element_ids,
        gdofs_map=batch.gdofs_map,
        A_full=A_full,
        R_full=R_full,
    )

    assert np.allclose(A_full.toarray(), A_ref.toarray())
    assert np.allclose(R_full, 0.0)


@pytest.mark.parametrize(
    "backend",
    ["python"] + (["jit"] if _nb is not None else []) + (["cpp"] if _have_cpp_backend() else []),
)
def test_form_compiler_local_nonmatching_interface_contributions_match_global_assembly(
    backend: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"symbolic_local_nonmatching_interface_{backend}"))
    dh, eq = _make_nonmatching_interface_scalar_problem()

    A_ref, R_ref = FormCompiler(dh, quadrature_order=4, backend="python").assemble(eq, bcs=[])
    batch = FormCompiler(dh, quadrature_order=4, backend=backend).assemble_local_contributions(eq)

    solver = _make_identity_solver(backend=backend, ndofs=dh.total_dofs)
    A_full = sp.lil_matrix((dh.total_dofs, dh.total_dofs), dtype=float)
    R_full = np.zeros(dh.total_dofs, dtype=float)
    A_full, R_full = NewtonSolver.scatter_element_contribs_full(
        solver,
        K_elem=batch.K_elem,
        F_elem=batch.F_elem,
        element_ids=batch.element_ids,
        gdofs_map=batch.gdofs_map,
        A_full=A_full,
        R_full=R_full,
    )

    assert np.allclose(A_full.toarray(), A_ref.toarray())
    assert np.allclose(R_full, R_ref)


@pytest.mark.parametrize(
    "backend",
    ["python"] + (["jit"] if _nb is not None else []) + (["cpp"] if _have_cpp_backend() else []),
)
def test_symbolic_local_assembly_operator_matches_global_assembly(
    backend: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"symbolic_local_operator_{backend}"))
    dh, eq = _make_symbolic_scalar_problem()
    solver = _make_identity_solver(backend=backend, ndofs=dh.total_dofs)

    op = SymbolicLocalAssemblyOperator(
        dof_handler=dh,
        form_or_equation=eq,
        quadrature_order=4,
    )
    A_red = sp.lil_matrix((dh.total_dofs, dh.total_dofs), dtype=float)
    R_red = np.zeros(dh.total_dofs, dtype=float)
    A_red, R_red = op.after_assembly(
        solver=solver,
        coeffs={},
        A_red=A_red,
        R_red=R_red,
        need_matrix=True,
    )

    A_ref, R_ref = FormCompiler(dh, quadrature_order=4, backend="python").assemble(eq, bcs=[])
    assert np.allclose(A_red.toarray(), A_ref.toarray())
    assert np.allclose(R_red, R_ref)


@pytest.mark.parametrize(
    "backend",
    ["python"] + (["jit"] if _nb is not None else []) + (["cpp"] if _have_cpp_backend() else []),
)
def test_symbolic_local_assembly_operator_matches_boundary_global_assembly(
    backend: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"symbolic_local_operator_boundary_{backend}"))
    dh, eq = _make_boundary_scalar_problem()
    solver = _make_identity_solver(backend=backend, ndofs=dh.total_dofs)

    op = SymbolicLocalAssemblyOperator(
        dof_handler=dh,
        form_or_equation=eq,
        quadrature_order=4,
    )
    A_red = sp.lil_matrix((dh.total_dofs, dh.total_dofs), dtype=float)
    R_red = np.zeros(dh.total_dofs, dtype=float)
    A_red, R_red = op.after_assembly(
        solver=solver,
        coeffs={},
        A_red=A_red,
        R_red=R_red,
        need_matrix=True,
    )

    A_ref, R_ref = FormCompiler(dh, quadrature_order=4, backend="python").assemble(eq, bcs=[])
    assert np.allclose(A_red.toarray(), A_ref.toarray())
    assert np.allclose(R_red, R_ref)


@pytest.mark.parametrize(
    "backend",
    ["python"] + (["jit"] if _nb is not None else []) + (["cpp"] if _have_cpp_backend() else []),
)
def test_form_compiler_local_quadrature_expressions_match_python(
    backend: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"local_quad_expr_{backend}"))
    dh, _ = _make_symbolic_scalar_problem()
    u = TrialFunction("u", dh)
    v = TestFunction("u", dh)
    layout = QuadratureLayout(
        entity_kind="volume_cell",
        cell_type="tri",
        quadrature_order=1,
        reference_points=np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float),
        reference_weights=np.array([0.5], dtype=float),
    )
    fc_ref = FormCompiler(dh, quadrature_order=1, backend="python")
    fc = FormCompiler(dh, quadrature_order=1, backend=backend)
    ref = fc_ref.evaluate_volume_local_expressions_on_quadrature(
        {"v": v},
        layout=layout,
    )
    got = fc.evaluate_volume_local_expressions_on_quadrature(
        {"v": v},
        layout=layout,
    )
    assert ref["v"].mode == got["v"].mode == "vector"
    assert np.array_equal(ref["v"].gdofs_map, got["v"].gdofs_map)
    np.testing.assert_allclose(got["v"].values, ref["v"].values, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize(
    "backend",
    ["python"] + (["jit"] if _nb is not None else []) + (["cpp"] if _have_cpp_backend() else []),
)
def test_form_compiler_condensed_local_system_matches_manual_reference(
    backend: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"condensed_local_system_{backend}"))
    dh, eq = _make_symbolic_scalar_problem()
    u = TrialFunction("u", dh)
    v = TestFunction("u", dh)
    layout = QuadratureLayout(
        entity_kind="volume_cell",
        cell_type="tri",
        quadrature_order=1,
        reference_points=np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float),
        reference_weights=np.array([0.5], dtype=float),
    )
    condensed = CondensedQuadratureLocalSystem(
        base_form_or_equation=eq,
        coupling_left=(v,),
        coupling_right=(v,),
        hidden_jacobian=((Constant(2.0),),),
        hidden_residual=(Constant(4.0),),
        quadrature_layout=layout,
        sign=-1.0,
    )
    fc_ref = FormCompiler(dh, quadrature_order=1, backend="python")
    base_batch = fc_ref.assemble_local_contributions(eq)
    v_batch = fc_ref.evaluate_volume_local_expressions_on_quadrature({"v": v}, layout=layout)["v"]
    weights = fc_ref._volume_physical_quadrature_weights(layout=layout, element_ids=base_batch.element_ids)
    vvals = np.asarray(v_batch.values, dtype=float)
    K_expected = np.asarray(base_batch.K_elem, dtype=float) - 0.5 * np.einsum(
        "eq,eqi,eqj->eij",
        weights,
        vvals,
        vvals,
        optimize=True,
    )
    F_expected = np.asarray(base_batch.F_elem, dtype=float) - 2.0 * np.einsum(
        "eq,eqi->ei",
        weights,
        vvals,
        optimize=True,
    )
    batch = FormCompiler(dh, quadrature_order=1, backend=backend).assemble_local_contributions(condensed)
    np.testing.assert_allclose(np.asarray(batch.K_elem, dtype=float), K_expected, rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(np.asarray(batch.F_elem, dtype=float), F_expected, rtol=1.0e-11, atol=1.0e-11)


@pytest.mark.parametrize(
    "backend",
    ["python"] + (["jit"] if _nb is not None else []) + (["cpp"] if _have_cpp_backend() else []),
)
def test_form_compiler_condensed_direct_scatter_matches_batch_scatter(
    backend: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"condensed_direct_scatter_{backend}"))
    dh, eq = _make_symbolic_scalar_problem()
    v = TestFunction("u", dh)
    layout = QuadratureLayout(
        entity_kind="volume_cell",
        cell_type="tri",
        quadrature_order=1,
        reference_points=np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float),
        reference_weights=np.array([0.5], dtype=float),
    )
    condensed = CondensedQuadratureLocalSystem(
        base_form_or_equation=eq,
        coupling_left=(v,),
        coupling_right=(v,),
        hidden_jacobian=((Constant(2.0),),),
        hidden_residual=(Constant(4.0),),
        quadrature_layout=layout,
        sign=-1.0,
    )
    fc = FormCompiler(dh, quadrature_order=1, backend=backend)
    batch = fc.assemble_local_contributions(condensed)
    solver = _make_identity_solver(backend=backend, ndofs=dh.total_dofs)
    A_ref = np.zeros((dh.total_dofs, dh.total_dofs), dtype=float)
    R_ref = np.zeros((dh.total_dofs,), dtype=float)
    A_ref, R_ref = solver.scatter_element_contribs_reduced(
        K_elem=batch.K_elem,
        F_elem=batch.F_elem,
        element_ids=batch.element_ids,
        gdofs_map=batch.gdofs_map,
        A_red=A_ref,
        R_red=R_ref,
    )
    A_got = np.zeros_like(A_ref)
    R_got = np.zeros_like(R_ref)
    A_got, R_got = fc.assemble_and_scatter_local_contributions_reduced(
        condensed,
        solver=solver,
        A_red=A_got,
        R_red=R_got,
        need_matrix=True,
    )
    np.testing.assert_allclose(A_got, A_ref, rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(R_got, R_ref, rtol=1.0e-11, atol=1.0e-11)
