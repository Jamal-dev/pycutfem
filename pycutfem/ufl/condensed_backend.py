from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np

try:  # pragma: no cover - optional dependency
    import numba as _nb
except Exception:  # pragma: no cover - optional dependency
    _nb = None


def _condense_local_system_numpy(
    left: np.ndarray,
    right: np.ndarray,
    hidden_jacobian: np.ndarray,
    hidden_residual: np.ndarray,
    weights: np.ndarray,
    *,
    sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    jac_arr = np.asarray(hidden_jacobian, dtype=float)
    res_arr = np.asarray(hidden_residual, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    if int(left_arr.shape[0]) == 0:
        n_loc = int(left_arr.shape[-1]) if left_arr.ndim >= 4 else 0
        return np.zeros((0, n_loc, n_loc), dtype=float), np.zeros((0, n_loc), dtype=float)
    rhs_mat = np.asarray(right_arr, dtype=float)
    sol_mat = np.zeros_like(rhs_mat, dtype=float)
    sol_vec = np.zeros_like(res_arr, dtype=float)
    for e in range(int(jac_arr.shape[0])):
        for q in range(int(jac_arr.shape[1])):
            A = np.asarray(jac_arr[e, q], dtype=float)
            B = np.asarray(rhs_mat[e, q], dtype=float)
            r = np.asarray(res_arr[e, q], dtype=float)
            try:
                sol_mat[e, q] = np.asarray(np.linalg.solve(A, B), dtype=float)
                sol_vec[e, q] = np.asarray(np.linalg.solve(A, r), dtype=float)
            except np.linalg.LinAlgError:
                sol_mat[e, q] = np.asarray(np.linalg.pinv(A) @ B, dtype=float)
                sol_vec[e, q] = np.asarray(np.linalg.pinv(A) @ r, dtype=float)
    K_corr = float(sign) * np.einsum("eq,eqai,eqaj->eij", w_arr, left_arr, sol_mat, optimize=True)
    F_corr = float(sign) * np.einsum("eq,eqai,eqa->ei", w_arr, left_arr, sol_vec, optimize=True)
    return np.asarray(K_corr, dtype=float), np.asarray(F_corr, dtype=float)


if _nb is not None:

    @_nb.njit(cache=True)
    def _gauss_solve_small_matrix(A, B):
        m = A.shape[0]
        nrhs = B.shape[1]
        M = np.empty((m, m), dtype=np.float64)
        RHS = np.empty((m, nrhs), dtype=np.float64)
        for i in range(m):
            for j in range(m):
                M[i, j] = A[i, j]
            for j in range(nrhs):
                RHS[i, j] = B[i, j]

        for k in range(m):
            pivot = k
            pivot_abs = abs(M[k, k])
            for i in range(k + 1, m):
                val = abs(M[i, k])
                if val > pivot_abs:
                    pivot = i
                    pivot_abs = val
            if pivot_abs <= 1.0e-20:
                for i in range(m):
                    for j in range(nrhs):
                        RHS[i, j] = 0.0
                return RHS, False
            if pivot != k:
                for j in range(m):
                    tmp = M[k, j]
                    M[k, j] = M[pivot, j]
                    M[pivot, j] = tmp
                for j in range(nrhs):
                    tmp = RHS[k, j]
                    RHS[k, j] = RHS[pivot, j]
                    RHS[pivot, j] = tmp
            diag = M[k, k]
            for i in range(k + 1, m):
                factor = M[i, k] / diag
                M[i, k] = 0.0
                for j in range(k + 1, m):
                    M[i, j] -= factor * M[k, j]
                for j in range(nrhs):
                    RHS[i, j] -= factor * RHS[k, j]

        X = np.empty((m, nrhs), dtype=np.float64)
        for col in range(nrhs):
            for i in range(m - 1, -1, -1):
                acc = RHS[i, col]
                for j in range(i + 1, m):
                    acc -= M[i, j] * X[j, col]
                X[i, col] = acc / M[i, i]
        return X, True


    @_nb.njit(cache=True)
    def _condense_local_system_numba_impl(left, right, hidden_jacobian, hidden_residual, weights, sign):
        n_elem = left.shape[0]
        n_q = left.shape[1]
        n_hidden = left.shape[2]
        n_loc = left.shape[3]
        K_corr = np.zeros((n_elem, n_loc, n_loc), dtype=np.float64)
        F_corr = np.zeros((n_elem, n_loc), dtype=np.float64)
        rhs_mat = np.empty((n_hidden, n_loc), dtype=np.float64)
        rhs_vec = np.empty((n_hidden, 1), dtype=np.float64)
        for e in range(n_elem):
            for q in range(n_q):
                wq = weights[e, q]
                if wq == 0.0:
                    continue
                for a in range(n_hidden):
                    for j in range(n_loc):
                        rhs_mat[a, j] = right[e, q, a, j]
                    rhs_vec[a, 0] = hidden_residual[e, q, a]
                sol_mat, ok_mat = _gauss_solve_small_matrix(hidden_jacobian[e, q], rhs_mat)
                sol_vec, ok_vec = _gauss_solve_small_matrix(hidden_jacobian[e, q], rhs_vec)
                if not ok_mat or not ok_vec:
                    continue
                for i in range(n_loc):
                    corr_f = 0.0
                    for a in range(n_hidden):
                        corr_f += left[e, q, a, i] * sol_vec[a, 0]
                    F_corr[e, i] += sign * wq * corr_f
                    for j in range(n_loc):
                        corr_k = 0.0
                        for a in range(n_hidden):
                            corr_k += left[e, q, a, i] * sol_mat[a, j]
                        K_corr[e, i, j] += sign * wq * corr_k
        return K_corr, F_corr


def _condense_local_system_numba(
    left: np.ndarray,
    right: np.ndarray,
    hidden_jacobian: np.ndarray,
    hidden_residual: np.ndarray,
    weights: np.ndarray,
    *,
    sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    if _nb is None:  # pragma: no cover - import guard
        return _condense_local_system_numpy(left, right, hidden_jacobian, hidden_residual, weights, sign=sign)
    return _condense_local_system_numba_impl(
        np.ascontiguousarray(left, dtype=np.float64),
        np.ascontiguousarray(right, dtype=np.float64),
        np.ascontiguousarray(hidden_jacobian, dtype=np.float64),
        np.ascontiguousarray(hidden_residual, dtype=np.float64),
        np.ascontiguousarray(weights, dtype=np.float64),
        float(sign),
    )


@lru_cache(maxsize=1)
def _load_cpp_condense_module():
    from pycutfem.jit.cpp_backend.compiler import compile_extension

    module_name = "_pycutfem_condensed_local_ops"
    cache_dir = Path.home() / ".cache" / "pycutfem_jit" / "cpp" / "condensed_local_ops"
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_path = cache_dir / f"{module_name}.cpp"
    if not source_path.exists():
        source_path.write_text(
            r"""
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <cstdint>

namespace py = pybind11;

py::tuple condense_local_system(
    py::array_t<double, py::array::c_style | py::array::forcecast> left_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> right_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> jac_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> res_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> w_arr,
    double sign
) {
    auto left = left_arr.unchecked<4>();   // (e,q,a,i)
    auto right = right_arr.unchecked<4>(); // (e,q,a,j)
    auto jac = jac_arr.unchecked<4>();     // (e,q,a,b)
    auto res = res_arr.unchecked<3>();     // (e,q,a)
    auto w = w_arr.unchecked<2>();         // (e,q)

    const py::ssize_t n_elem = left.shape(0);
    const py::ssize_t n_q = left.shape(1);
    const py::ssize_t n_hidden = left.shape(2);
    const py::ssize_t n_loc = left.shape(3);

    py::array_t<double> K_arr({n_elem, n_loc, n_loc});
    py::array_t<double> F_arr({n_elem, n_loc});
    auto K = K_arr.mutable_unchecked<3>();
    auto F = F_arr.mutable_unchecked<2>();
    for (py::ssize_t e = 0; e < n_elem; ++e) {
        for (py::ssize_t i = 0; i < n_loc; ++i) {
            F(e, i) = 0.0;
            for (py::ssize_t j = 0; j < n_loc; ++j) {
                K(e, i, j) = 0.0;
            }
        }
    }

    for (py::ssize_t e = 0; e < n_elem; ++e) {
        for (py::ssize_t q = 0; q < n_q; ++q) {
            const double wq = w(e, q);
            if (wq == 0.0) {
                continue;
            }
            Eigen::MatrixXd A(n_hidden, n_hidden);
            Eigen::MatrixXd B(n_hidden, n_loc);
            Eigen::VectorXd r(n_hidden);
            for (py::ssize_t a = 0; a < n_hidden; ++a) {
                r(a) = res(e, q, a);
                for (py::ssize_t b = 0; b < n_hidden; ++b) {
                    A(a, b) = jac(e, q, a, b);
                }
                for (py::ssize_t j = 0; j < n_loc; ++j) {
                    B(a, j) = right(e, q, a, j);
                }
            }
            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver(A);
            if (solver.rank() < n_hidden) {
                continue;
            }
            Eigen::MatrixXd X = solver.solve(B);
            Eigen::VectorXd y = solver.solve(r);
            for (py::ssize_t i = 0; i < n_loc; ++i) {
                double corr_f = 0.0;
                for (py::ssize_t a = 0; a < n_hidden; ++a) {
                    corr_f += left(e, q, a, i) * y(a);
                }
                F(e, i) += sign * wq * corr_f;
                for (py::ssize_t j = 0; j < n_loc; ++j) {
                    double corr_k = 0.0;
                    for (py::ssize_t a = 0; a < n_hidden; ++a) {
                        corr_k += left(e, q, a, i) * X(a, j);
                    }
                    K(e, i, j) += sign * wq * corr_k;
                }
            }
        }
    }
    return py::make_tuple(K_arr, F_arr);
}

PYBIND11_MODULE(_pycutfem_condensed_local_ops, m) {
    m.attr("CODEGEN_ABI") = "condensed-local-ops-v1";
    m.def("condense_local_system", &condense_local_system);
}
            """,
            encoding="utf-8",
        )
    ext_path = compile_extension(module_name, source_path, cache_dir)
    spec = importlib.util.spec_from_file_location(module_name, ext_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load condensed C++ module {ext_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _condense_local_system_cpp(
    left: np.ndarray,
    right: np.ndarray,
    hidden_jacobian: np.ndarray,
    hidden_residual: np.ndarray,
    weights: np.ndarray,
    *,
    sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    module = _load_cpp_condense_module()
    K_corr, F_corr = module.condense_local_system(
        np.ascontiguousarray(left, dtype=float),
        np.ascontiguousarray(right, dtype=float),
        np.ascontiguousarray(hidden_jacobian, dtype=float),
        np.ascontiguousarray(hidden_residual, dtype=float),
        np.ascontiguousarray(weights, dtype=float),
        float(sign),
    )
    return np.asarray(K_corr, dtype=float), np.asarray(F_corr, dtype=float)


def condense_local_system(
    left: np.ndarray,
    right: np.ndarray,
    hidden_jacobian: np.ndarray,
    hidden_residual: np.ndarray,
    weights: np.ndarray,
    *,
    backend: str,
    sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    name = str(backend or "python").strip().lower()
    if name == "python":
        return _condense_local_system_numpy(left, right, hidden_jacobian, hidden_residual, weights, sign=sign)
    if name == "jit":
        return _condense_local_system_numba(left, right, hidden_jacobian, hidden_residual, weights, sign=sign)
    if name in {"cpp", "c++"}:
        return _condense_local_system_cpp(left, right, hidden_jacobian, hidden_residual, weights, sign=sign)
    raise ValueError(f"Unsupported condensed local-system backend {backend!r}.")


__all__ = ["condense_local_system"]
