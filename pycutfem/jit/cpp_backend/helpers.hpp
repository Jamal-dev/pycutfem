// Native C++ implementations of selected helper routines used by the
// generated kernels. We will grow this file incrementally to cover the
// full helper surface.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <type_traits>
bool is_debug = false;
namespace py = pybind11;
namespace pycutfem::cpp_backend {

// ---------------------------------------------------------------------------
// compute_physical_laplacian (scalar component)
// Mirrors pycutfem.jit.numba_helpers.compute_physical_laplacian
// ---------------------------------------------------------------------------
inline py::array_t<double> compute_physical_laplacian(const py::array_t<double>& d20,
                                                      const py::array_t<double>& d11,
                                                      const py::array_t<double>& d02,
                                                      const py::array_t<double>& d10,
                                                      const py::array_t<double>& d01,
                                                      const py::array_t<double>& j_inv,
                                                      const py::array_t<double>& hx,
                                                      const py::array_t<double>& hy,
                                                      py::object /*dtype*/ = py::none()) {
    auto d20v = d20.unchecked<1>();
    auto d11v = d11.unchecked<1>();
    auto d02v = d02.unchecked<1>();
    auto d10v = d10.unchecked<1>();
    auto d01v = d01.unchecked<1>();
    auto J    = j_inv.unchecked<2>(); // (2,2)
    auto Hxv  = hx.unchecked<2>();    // (2,2)
    auto Hyv  = hy.unchecked<2>();    // (2,2)

    ssize_t nloc = d20v.shape(0);
    py::array_t<double> out({nloc});
    auto ov = out.mutable_unchecked<1>();

    for (ssize_t j = 0; j < nloc; ++j) {
        // href = [[d20, d11],[d11, d02]]
        double href00 = d20v(j);
        double href01 = d11v(j);
        double href11 = d02v(j);

        // tmp = href @ J
        double tmp00 = href00 * J(0, 0) + href01 * J(1, 0);
        double tmp01 = href00 * J(0, 1) + href01 * J(1, 1);
        double tmp10 = href01 * J(0, 0) + href11 * J(1, 0);
        double tmp11 = href01 * J(0, 1) + href11 * J(1, 1);

        // core = J^T @ tmp
        double core00 = J(0, 0) * tmp00 + J(1, 0) * tmp10;
        double core11 = J(0, 1) * tmp01 + J(1, 1) * tmp11;

        // hphys = core + d10 * hx + d01 * hy
        double h00 = core00 + d10v(j) * Hxv(0, 0) + d01v(j) * Hyv(0, 0);
        double h11 = core11 + d10v(j) * Hxv(1, 1) + d01v(j) * Hyv(1, 1);

        ov(j) = h00 + h11;
    }

    return out;
}

// ---------------------------------------------------------------------------
// compute_physical_hessian (single component)
// Mirrors pycutfem.jit.numba_helpers.compute_physical_hessian
// ---------------------------------------------------------------------------
inline py::array_t<double> compute_physical_hessian(const py::array_t<double>& d20,
                                                    const py::array_t<double>& d11,
                                                    const py::array_t<double>& d02,
                                                    const py::array_t<double>& d10,
                                                    const py::array_t<double>& d01,
                                                    const py::array_t<double>& j_inv,
                                                    const py::array_t<double>& hx,
                                                    const py::array_t<double>& hy,
                                                    py::object /*dtype*/ = py::none()) {
    auto d20v = d20.unchecked<1>();
    auto d11v = d11.unchecked<1>();
    auto d02v = d02.unchecked<1>();
    auto d10v = d10.unchecked<1>();
    auto d01v = d01.unchecked<1>();
    auto J    = j_inv.unchecked<2>(); // (2,2)
    auto Hxv  = hx.unchecked<2>();    // (2,2)
    auto Hyv  = hy.unchecked<2>();    // (2,2)

    ssize_t nloc = d20v.shape(0);
    std::vector<ssize_t> shape = {nloc, 2, 2};
    py::array_t<double> out(shape);
    auto ov = out.mutable_unchecked<3>();

    for (ssize_t j = 0; j < nloc; ++j) {
        double href00 = d20v(j);
        double href01 = d11v(j);
        double href11 = d02v(j);

        double tmp00 = href00 * J(0, 0) + href01 * J(1, 0);
        double tmp01 = href00 * J(0, 1) + href01 * J(1, 1);
        double tmp10 = href01 * J(0, 0) + href11 * J(1, 0);
        double tmp11 = href01 * J(0, 1) + href11 * J(1, 1);

        double core00 = J(0, 0) * tmp00 + J(1, 0) * tmp10;
        double core01 = J(0, 1) * tmp00 + J(1, 1) * tmp10;
        double core10 = J(0, 0) * tmp01 + J(1, 0) * tmp11;
        double core11 = J(0, 1) * tmp01 + J(1, 1) * tmp11;

        ov(j, 0, 0) = core00 + d10v(j) * Hxv(0, 0) + d01v(j) * Hyv(0, 0);
        ov(j, 0, 1) = core01 + d10v(j) * Hxv(0, 1) + d01v(j) * Hyv(0, 1);
        ov(j, 1, 0) = core10 + d10v(j) * Hxv(1, 0) + d01v(j) * Hyv(1, 0);
        ov(j, 1, 1) = core11 + d10v(j) * Hxv(1, 1) + d01v(j) * Hyv(1, 1);
    }

    return out;
}

// ---------------------------------------------------------------------------
// pushforward_d3 (exact third-order pullback)
// Mirrors pycutfem.jit.numba_helpers.pushforward_d3
// ---------------------------------------------------------------------------
inline py::array_t<double> pushforward_d3(const py::array_t<double>& d10,
                                          const py::array_t<double>& d01,
                                          const py::array_t<double>& d20,
                                          const py::array_t<double>& d11,
                                          const py::array_t<double>& d02,
                                          const py::array_t<double>& d30,
                                          const py::array_t<double>& d21,
                                          const py::array_t<double>& d12,
                                          const py::array_t<double>& d03,
                                          const py::array_t<double>& A,
                                          const py::array_t<double>& Hx,
                                          const py::array_t<double>& Hy,
                                          const py::array_t<double>& Tx0,
                                          const py::array_t<double>& Tx1,
                                          const py::array_t<int>& axes,
                                          py::object /*dtype*/ = py::none()) {
    auto ax = axes.unchecked<1>();
    int ax0 = static_cast<int>(ax(0));
    int ax1 = static_cast<int>(ax(1));
    int ax2 = static_cast<int>(ax(2));

    auto d10v = d10.unchecked<1>();
    auto d01v = d01.unchecked<1>();
    auto d20v = d20.unchecked<1>();
    auto d11v = d11.unchecked<1>();
    auto d02v = d02.unchecked<1>();
    auto d30v = d30.unchecked<1>();
    auto d21v = d21.unchecked<1>();
    auto d12v = d12.unchecked<1>();
    auto d03v = d03.unchecked<1>();

    auto Av   = A.unchecked<2>();   // (2,2)
    auto Hxv  = Hx.unchecked<2>();  // (2,2)
    auto Hyv  = Hy.unchecked<2>();  // (2,2)
    auto Tx0v = Tx0.unchecked<3>(); // (2,2,2)
    auto Tx1v = Tx1.unchecked<3>(); // (2,2,2)

    ssize_t nloc = d20v.shape(0);
    py::array_t<double> out({nloc});
    auto ov = out.mutable_unchecked<1>();

    for (ssize_t j = 0; j < nloc; ++j) {
        double s = 0.0;

        // cubic terms
        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
                for (int c = 0; c < 2; ++c) {
                    int ones = (a == 1) + (b == 1) + (c == 1);
                    double g3 = (ones == 0) ? d30v(j)
                                : (ones == 1) ? d21v(j)
                                : (ones == 2) ? d12v(j)
                                              : d03v(j);
                    s += g3 * Av(a, ax0) * Av(b, ax1) * Av(c, ax2);
                }
            }
        }

        // quadratic terms with Hessian
        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
                double g2 = (a == 0 && b == 0) ? d20v(j)
                            : (a != b) ? d11v(j)
                                       : d02v(j);
                const auto& Hb = (b == 0) ? Hxv : Hyv;
                s += g2 * (Av(a, ax0) * Hb(ax1, ax2)
                          +Av(a, ax1) * Hb(ax0, ax2)
                          +Av(a, ax2) * Hb(ax0, ax1));
            }
        }

        // linear terms with third-order tensors
        s += d10v(j) * Tx0v(ax0, ax1, ax2) + d01v(j) * Tx1v(ax0, ax1, ax2);

        ov(j) = s;
    }

    return out;
}


// ---------------------------------------------------------------------------
// Variable Evaluation
// ---------------------------------------------------------------------------

// Mirrors load_variable_qp
// u_e: (ndof) or (ndof, ncomp)
// phi_q: (ndof)
inline py::array_t<double> load_variable_qp(const py::array_t<double>& u_e,
                                            const py::array_t<double>& phi_q) {
    auto u_req = u_e.request();
    auto p_req = phi_q.request();

    // Map as Eigen vectors/matrices
    // phi is always (ndof,)
    Eigen::Map<const Eigen::VectorXd> phi((double*)p_req.ptr, p_req.size);

    if (u_req.ndim == 1) {
        // Scalar field: dot(u, phi)
        Eigen::Map<const Eigen::VectorXd> u((double*)u_req.ptr, u_req.size);
        return py::cast(u.dot(phi));
    } else {
        // Vector field: (ndof, ncomp).T @ phi -> (ncomp)
        ssize_t ndof = u_req.shape[0];
        ssize_t ncomp = u_req.shape[1];
        // Eigen defaults to ColumnMajor, but numpy is RowMajor usually.
        // Map as RowMajor Matrix: (ndof, ncomp)
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            U((double*)u_req.ptr, ndof, ncomp);
        
        // Result = U.T * phi
        Eigen::VectorXd res = U.transpose() * phi;
        return py::cast(res);
    }
}

// Mirrors gradient_qp
// u_e: (ndof) or (ndof, ncomp)
// grad_phi_q: (ndof, dim)
inline py::array_t<double> gradient_qp(const py::array_t<double>& u_e,
                                       const py::array_t<double>& grad_phi_q) {
    auto u_req = u_e.request();
    auto g_req = grad_phi_q.request();
    
    ssize_t ndof = g_req.shape[0];
    ssize_t dim  = g_req.shape[1];
    
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        G((double*)g_req.ptr, ndof, dim);

    if (u_req.ndim == 1) {
        // Scalar field: G.T @ u -> (dim,)
        Eigen::Map<const Eigen::VectorXd> u((double*)u_req.ptr, u_req.size);
        Eigen::VectorXd res = G.transpose() * u;
        return py::cast(res);
    } else {
        // Vector field: G.T @ U -> (dim, ncomp)
        ssize_t ncomp = u_req.shape[1];
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            U((double*)u_req.ptr, ndof, ncomp);
        
        Eigen::MatrixXd res = G.transpose() * U;
        return py::cast(res);
    }
}

// Laplacian evaluation at a qp
inline py::object laplacian_qp(const py::array_t<double>& u_e,
                               const py::array_t<double>& lap_phi_q) {
    auto u_req = u_e.request();
    auto l_req = lap_phi_q.request();
    ssize_t ndof = l_req.shape[0];
    Eigen::Map<const Eigen::VectorXd> lap((double*)l_req.ptr, ndof);

    if (u_req.ndim == 1) {
        Eigen::Map<const Eigen::VectorXd> u((double*)u_req.ptr, ndof);
        double val = lap.dot(u);
        return py::cast(val);
    }
    ssize_t ncomp = u_req.shape[1];
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        U((double*)u_req.ptr, ndof, ncomp);
    Eigen::VectorXd res = U.transpose() * lap;
    return py::cast(res);
}

// Hessian evaluation at a qp
inline py::object hessian_qp(const py::array_t<double>& u_e,
                             const py::array_t<double>& hess_phi_q) {
    auto u_req = u_e.request();
    auto h_req = hess_phi_q.request();

    ssize_t ndof = h_req.shape[0];
    ssize_t dim  = h_req.shape[1];
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Hflat((double*)h_req.ptr, ndof, dim * dim);

    if (u_req.ndim == 1) {
        Eigen::Map<const Eigen::VectorXd> u((double*)u_req.ptr, ndof);
        Eigen::VectorXd res = Hflat.transpose() * u; // (dim*dim,)
        Eigen::MatrixXd out = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            res.data(), dim, dim);
        return py::cast(out);
    }

    ssize_t ncomp = u_req.shape[1];
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        U((double*)u_req.ptr, ndof, ncomp);
    Eigen::MatrixXd prod = U.transpose() * Hflat; // (ncomp, dim*dim)
    std::vector<Eigen::MatrixXd> out;
    out.reserve(static_cast<size_t>(ncomp));
    for (ssize_t i = 0; i < ncomp; ++i) {
        Eigen::MatrixXd m = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            prod.row(i).data(), dim, dim);
        out.push_back(m);
    }
    return py::cast(out);
}

// ---------------------------------------------------------------------------
// CutFEM Basis Handling
// ---------------------------------------------------------------------------

// Mirrors pad_basis_to_union
// Scatter local basis values to the union-sized vector
inline py::array_t<double> pad_basis_to_union(const py::array_t<double>& local,
                                              const py::array_t<int>& mapping,
                                              int n_union,
                                              int s0, int s1,
                                              py::object /*dtype*/ = py::none()) {
    py::array_t<double> out(n_union);
    auto out_v = out.mutable_unchecked<1>();
    // Zero initialize
    std::fill(out_v.mutable_data(0), out_v.mutable_data(0) + n_union, 0.0);
    
    auto loc_v = local.unchecked<1>();
    auto map_v = mapping.unchecked<1>();
    
    ssize_t m = map_v.shape(0);
    bool slice_input = (local.size() != m);

    for (ssize_t j = 0; j < m; ++j) {
        int idx = map_v(j);
        if (idx >= 0 && idx < n_union) {
            double val = slice_input ? loc_v(s0 + j) : loc_v(j);
            out_v(idx) = val;
        }
    }
    return out;
}

// Mirrors pushforward_grad_to_union
// 1. Transform ref grads (d10, d01) by J_inv
// 2. Scatter to union vector
inline py::array_t<double> pushforward_grad_to_union(const py::array_t<double>& d10,
                                                     const py::array_t<double>& d01,
                                                     const py::array_t<double>& j_inv,
                                                     const py::array_t<int>& mapping,
                                                     int n_union,
                                                     int s0, int s1,
                                                     py::object /*dtype*/ = py::none()) {
    // Output: (n_union, 2)
    py::array_t<double> out({static_cast<ssize_t>(n_union), ssize_t(2)});
    std::fill(out.mutable_data(), out.mutable_data() + out.size(), 0.0);
    auto out_a = out.mutable_unchecked<2>();

    auto d10_v = d10.unchecked<1>();
    auto d01_v = d01.unchecked<1>();
    auto map_v = mapping.unchecked<1>();
    auto J     = j_inv.unchecked<2>(); // (2,2) assumed

    ssize_t m = map_v.shape(0);
    bool slice_input = (d10.size() != m);

    for (ssize_t j = 0; j < m; ++j) {
        int idx = map_v(j);
        if (idx >= 0 && idx < n_union) {
            // Apply slice if needed (s0)
            ssize_t src_idx = slice_input ? (s0 + j) : j;
            
            double g0 = d10_v(src_idx);
            double g1 = d01_v(src_idx);

            // pushforward: g_phys = g_ref @ J_inv
            out_a(idx, 0) = g0 * J(0,0) + g1 * J(1,0);
            out_a(idx, 1) = g0 * J(0,1) + g1 * J(1,1);
        }
    }
    return out;
}

/**
 * Robustly performs the contraction: c_n = sum_k (A_kn * b_k).
 * * Logic:
 * - A is (K, N). b is vector of size K.
 * - Contracts the first index (rows) of A with b.
 * - This is mathematically equivalent to b.transpose() * A.
 * - Result is ALWAYS returned as a Row Vector (1, N).
 */
inline Eigen::MatrixXd contract_mat_vector_first_index(const Eigen::MatrixXd& A,
                                                       const Eigen::VectorXd& b) {
    if (A.rows() != b.size()) {
        throw std::runtime_error("Dimension mismatch in contract_mat_vector_first_index: "
                                 "Matrix rows must match Vector size.");
    }
    return b.transpose() * A;  // (1,N)
}

/**
 * Robustly performs Matrix-Vector contraction: c_i = sum_j (A_ij * b_j).
 * * Logic:
 * - Treats 'b' as a vector of size N (auto-detects row vs column).
 * - Contracts columns of A (M x N) with b (N).
 * - Result is ALWAYS returned as a Row Vector (1, M).
 */
inline Eigen::MatrixXd contract_matrix_vector(const Eigen::MatrixXd& A,
                                              const Eigen::VectorXd& b) {
    if (A.cols() != b.size()) {
        throw std::runtime_error("Dimension mismatch in contract_matrix_vector: "
                                 "Matrix columns must match Vector size.");
    }
    return (A * b).transpose();  // (1,M)
}

/**
 * Robustly performs the contraction c_j = sum_i (a_i * b_ij).
 * * Logic:
 * - Always treats 'a' as a row vector (1, M) to contract with the rows of 'b'.
 * - Result is ALWAYS a MatrixXd of shape (1, N).
 */
/**
 * Robust vector-matrix contraction for generic Eigen expressions.
 * Accepts row/column vectors and 1xK/Kx1 expressions; returns shape (1,N).
 */
template <typename DerivedA>
inline Eigen::MatrixXd contract_vector_matrix(const Eigen::MatrixBase<DerivedA>& a_expr,
                                              const Eigen::MatrixXd& b) {
    const Eigen::MatrixXd a = a_expr;
    if (a.rows() == 1 && a.cols() == b.rows()) {
        return a * b;
    }
    if (a.cols() == 1 && a.rows() == b.rows()) {
        return a.transpose() * b;
    }
    if (a.size() == b.rows()) {
        Eigen::Map<const Eigen::VectorXd> avec(a.data(), a.size());
        return avec.transpose() * b;
    }
    std::stringstream ss;
    ss << "Dimension mismatch in contract_vector_matrix: "
       << "A=(" << a.rows() << "," << a.cols() << "), "
       << "B=(" << b.rows() << "," << b.cols() << ")";
    throw std::runtime_error(ss.str());
}

// ---------------------------------------------------------------------------
// Tensor Algebra (Dot Products / Contractions)
// ---------------------------------------------------------------------------

inline std::vector<Eigen::MatrixXd> contract_last_first(
    const std::vector<Eigen::MatrixXd>& A, 
    const Eigen::MatrixXd& B) 
{

    if (is_debug) {std::cout << "C++ backend: contract_last_first called" << std::endl;}
    if (A.empty()) return {};

    // 1. Get dimensions
    long n_matrices = A.size();
    long rows_per_matrix = A[0].rows();
    long contraction_dim = A[0].cols(); // This is 'K' (last dim of A)

    // Check consistency
    if (B.rows() != contraction_dim) {
        throw std::runtime_error("Dimension mismatch: A.cols() must match B.rows()");
    }

    long result_cols = B.cols(); // Resulting width

    // 2. Optimization: If the vector has only 1 matrix, just multiply directly
    if (n_matrices == 1) {
        return { A[0] * B };
    }

    // 3. Stack A into one large Matrix (N*R, K)
    //    We map the memory to avoid deep copies if possible, but std::vector
    //    storage of Eigen objects is not contiguous across matrices. 
    //    So we must copy into a large buffer.
    Eigen::MatrixXd A_stacked(n_matrices * rows_per_matrix, contraction_dim);
    
    for (long i = 0; i < n_matrices; ++i) {
        // Validation check for safety
        if (A[i].rows() != rows_per_matrix || A[i].cols() != contraction_dim) {
             throw std::runtime_error("All matrices in vector must have same dimensions");
        }
        // Copy block
        A_stacked.block(i * rows_per_matrix, 0, rows_per_matrix, contraction_dim) = A[i];
    }

    // 4. Perform the massive Matrix Multiplication
    //    (N*R, K) * (K, C) -> (N*R, C)
    Eigen::MatrixXd Res_stacked = A_stacked * B;

    // 5. Unpack result back into std::vector<Matrix>
    std::vector<Eigen::MatrixXd> out;
    out.reserve(n_matrices);

    for (long i = 0; i < n_matrices; ++i) {
        out.push_back(
            Res_stacked.block(i * rows_per_matrix, 0, rows_per_matrix, result_cols)
        );
    }

    return out;
}

// Semantic last-first contraction for two stack-backed tensors:
// A: (i, n_a, c), B: (c, n_b, j) -> out: (i, n_a, n_b, j)
inline std::vector<Eigen::MatrixXd> contract_last_first(
    const std::vector<Eigen::MatrixXd>& A,
    const std::vector<Eigen::MatrixXd>& B)
{
    if (is_debug) {std::cout << "C++ backend: contract_last_first(stack,stack) called" << std::endl;}
    if (A.empty() || B.empty()) return {};

    const int i_dim = static_cast<int>(A.size());
    const int n_a = static_cast<int>(A[0].rows());
    const int c_dim = static_cast<int>(A[0].cols());
    if (static_cast<int>(B.size()) != c_dim) {
        throw std::runtime_error("contract_last_first(stack,stack): contracted dimension mismatch.");
    }

    const int n_b = static_cast<int>(B[0].rows());
    const int j_dim = static_cast<int>(B[0].cols());
    std::vector<Eigen::MatrixXd> out(static_cast<size_t>(i_dim * j_dim), Eigen::MatrixXd::Zero(n_a, n_b));

    for (int i = 0; i < i_dim; ++i) {
        if (A[static_cast<size_t>(i)].rows() != n_a || A[static_cast<size_t>(i)].cols() != c_dim) {
            throw std::runtime_error("contract_last_first(stack,stack): inconsistent lhs component shape.");
        }
        for (int c = 0; c < c_dim; ++c) {
            if (B[static_cast<size_t>(c)].rows() != n_b || B[static_cast<size_t>(c)].cols() != j_dim) {
                throw std::runtime_error("contract_last_first(stack,stack): inconsistent rhs component shape.");
            }
            for (int j = 0; j < j_dim; ++j) {
                out[static_cast<size_t>(i * j_dim + j)].noalias() +=
                    A[static_cast<size_t>(i)].col(c) * B[static_cast<size_t>(c)].col(j).transpose();
            }
        }
    }

    return out;
}

/**
 * Overload for Vector-Matrix contraction.
 * Handles A (Vector) contracted with B (Matrix).
 * * Logic:
 * If A is (K) vector or (K,1) matrix, and B is (K, N) matrix:
 * Result = A.transpose() * B  -> Shape (1, N) or (N) depending on preference.
 */
inline Eigen::MatrixXd contract_last_first(const Eigen::VectorXd& A, 
                                           const Eigen::MatrixXd& B) {
    // 1. Check dimensions
    // We expect A (length K) to contract with B (rows K)
    if (A.size() != B.rows()) {
        throw std::runtime_error("contract_last_first Dimension mismatch: Vector size must match Matrix rows");
    }

    // 2. Perform Multiplication
    // In Eigen, VectorXd is a column vector.
    // A.transpose() turns it into a RowVector (1, K).
    // (1, K) * (K, N) = (1, N)
    return (A.transpose() * B).transpose(); // Return as (N, 1) column vector
}

// ALSO handle the case where A is passed as a thin MatrixXd (K, 1)
inline Eigen::MatrixXd contract_last_first(const Eigen::MatrixXd& A, 
                                           const Eigen::MatrixXd& B) {
    
    // Case 1: Standard Matrix Multiplication (M, K) * (K, N)
    if (A.cols() == B.rows()) {
        return A * B;
    }
    // Case 2: Column Vector Contraction (K, 1) * (K, N)
    // We treat the rows of A as the contraction dimension 'k'.
    else if (A.cols() == 1 && A.rows() == B.rows()) {
        // We implicitly transpose A to make it conformable
        return A.transpose() * B;
    }
    else {
        // Detailed error for debugging
        std::stringstream ss;
        ss << "Dimension mismatch in contract_last_first. "
           << "A: (" << A.rows() << "," << A.cols() << "), "
           << "B: (" << B.rows() << "," << B.cols() << "). "
           << "Expected A.cols() == B.rows() OR (A is column vector and A.rows() == B.rows())";
        throw std::runtime_error(ss.str());
    }
}


inline Eigen::MatrixXd contract_first_first(const std::vector<Eigen::MatrixXd>& A,
                                            const Eigen::VectorXd& B) {
    // 1. Validation
    if (A.empty()) {
        throw std::runtime_error("contract_first_first: Input A is empty");
    }
    
    // In C++, A.size() corresponds to the 'k' dimension
    if (static_cast<long>(A.size()) != B.size()) {
        throw std::runtime_error("contract_first_first: Dimension mismatch (A size vs B size)");
    }

    // 2. Get matrix dimensions (n, m) from the first element
    long rows = A[0].rows();
    long cols = A[0].cols();

    // 3. Initialize result as Zero matrix
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(rows, cols);

    // 4. Compute Linear Combination: result += A[k] * scalar_weight
    // This is more memory efficient than reshaping because we avoid 
    // copying the std::vector contents into a new buffer.
    for (size_t k = 0; k < A.size(); ++k) {
        // Check ensuring all matrices in vector are same size (optional safety)
        // if(A[k].rows() != rows || A[k].cols() != cols) throw ...

        // Accumulate: Matrix + (Matrix * scalar)
        result += A[k] * B(k);
    }

    return result;
}

inline Eigen::MatrixXd contract_first_first(const std::vector<Eigen::MatrixXd>& A,
                                            const Eigen::MatrixXd& B) {
    if (B.rows() == 1) {
        Eigen::VectorXd vec = B.row(0).transpose().eval();
        return contract_first_first(A, vec);
    }
    if (B.cols() == 1) {
        Eigen::VectorXd vec = B.col(0).eval();
        return contract_first_first(A, vec);
    }
    throw std::runtime_error("contract_first_first: second operand must be a vector or row/column matrix");
}

// Matrix (m,k) contracted with the component axis of a gradient stack (k,n,d)
// -> gradient stack (m,n,d).
inline std::vector<Eigen::MatrixXd> contract_component_matrix_grad(
    const Eigen::MatrixXd& A,
    const std::vector<Eigen::MatrixXd>& grad_basis)
{
    if (grad_basis.empty()) return {};
    if (A.cols() != static_cast<Eigen::Index>(grad_basis.size())) {
        throw std::runtime_error("contract_component_matrix_grad: matrix columns must match gradient component count.");
    }

    const Eigen::Index n = grad_basis[0].rows();
    const Eigen::Index d = grad_basis[0].cols();
    std::vector<Eigen::MatrixXd> out(static_cast<size_t>(A.rows()), Eigen::MatrixXd::Zero(n, d));

    for (Eigen::Index i = 0; i < A.rows(); ++i) {
        for (Eigen::Index k = 0; k < A.cols(); ++k) {
            if (grad_basis[static_cast<size_t>(k)].rows() != n || grad_basis[static_cast<size_t>(k)].cols() != d) {
                throw std::runtime_error("contract_component_matrix_grad: inconsistent gradient component shapes.");
            }
            out[static_cast<size_t>(i)].noalias() += A(i, k) * grad_basis[static_cast<size_t>(k)];
        }
    }
    return out;
}
// Mirrors dot_grad_basis_vector
// grad_basis: (k, n, d)
// vec: (d,)
// Output: (k, n) -> const vector scaling of gradient components
inline py::array_t<double> dot_grad_basis_vector(const py::array_t<double>& grad_basis,
                                                 const py::array_t<double>& vec) {
    auto g_req = grad_basis.request();
    ssize_t k = g_req.shape[0];
    ssize_t n = g_req.shape[1];
    ssize_t d = g_req.shape[2];

    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        G((double*)g_req.ptr, k * n, d);
    Eigen::Map<const Eigen::VectorXd> v((double*)vec.request().ptr, d);

    Eigen::VectorXd res = G * v; // (k*n, 1)
    
    return py::array_t<double>({k, n}, res.data());
}

// Mirrors dot_vec_vec
inline double dot_vec_vec(const py::array_t<double>& a, const py::array_t<double>& b) {
    auto a_r = a.unchecked<1>();
    auto b_r = b.unchecked<1>();
    double sum = 0.0;
    for (ssize_t i=0; i<a_r.size(); ++i) sum += a_r(i) * b_r(i);
    return sum;
}

// Mirrors transpose_grad_tensor
// Input: (k, n, d). Output: (k, n, d) but transposed in (k, d)? 
// NO, "transpose gradient" usually means (grad u)^T.
// numba_helper: "Transpose vector gradient tensor (k,n,d) assuming square k==d"
// It swaps the component axis (0) with the spatial axis (2).
// wait, (k, n, d) -> res[:, n, :] = tensor[:, n, :].T is impossible if result is (k,n,d)
// Actually numba_helpers implementation:
//   res[:, n, :] = tensor[:, n, :].T  <-- This implies k=d and it swaps dim 0 and 2 for each n.
inline py::array_t<double> transpose_grad_tensor(const py::array_t<double>& tensor) {
    auto t_req = tensor.request();
    ssize_t k = t_req.shape[0];
    ssize_t n = t_req.shape[1];
    ssize_t d = t_req.shape[2];

    py::array_t<double> out({k, n, d});
    auto in_v = tensor.unchecked<3>();
    auto out_v = out.mutable_unchecked<3>();

    // Mirror numba_helpers.transpose_grad_tensor:
    // res[i, n, l] = tensor[l, n, i] (assumes square k==d)
    for (ssize_t j = 0; j < n; ++j) {
        for (ssize_t i = 0; i < k; ++i) {
            for (ssize_t l = 0; l < d; ++l) {
                out_v(i, j, l) = in_v(l, j, i);
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Binary Operations (Broadcasting)
// ---------------------------------------------------------------------------

// Mirrors binary_sub_generic (and sub_3_4, sub_4_3)
// Handles (k, n, d) - (k, n, m, d) etc. via explicit loops for common FEM ranks
inline py::array_t<double> binary_sub_generic(const py::array_t<double>& a,
                                              const py::array_t<double>& b) {
    auto a_req = a.request();
    auto b_req = b.request();

    // 1. Same shape -> Flat subtraction
    if (a_req.ndim == b_req.ndim && std::equal(a_req.shape.begin(), a_req.shape.end(), b_req.shape.begin())) {
        py::array_t<double> out(a_req.shape);
        Eigen::Map<const Eigen::VectorXd> va((double*)a_req.ptr, a_req.size);
        Eigen::Map<const Eigen::VectorXd> vb((double*)b_req.ptr, b_req.size);
        Eigen::Map<Eigen::VectorXd> vout((double*)out.request().ptr, out.size());
        vout = va - vb;
        return out;
    }

    // 2. Broadcast Case: (K, N, D) - (K, N, M, D) -> (K, N, M, D)
    // Common in subtract_grad_grad (basis vs basis_mixed)
    if (a_req.ndim == 3 && b_req.ndim == 4) {
        // Expand A
        ssize_t K = a_req.shape[0];
        ssize_t N = a_req.shape[1];
        ssize_t M = b_req.shape[2];
        ssize_t D = a_req.shape[2];
        
        py::array_t<double> out(b_req.shape);
        auto A = a.unchecked<3>();
        auto B = b.unchecked<4>();
        auto O = out.mutable_unchecked<4>();

        #pragma omp parallel for collapse(3)
        for (ssize_t k=0; k<K; ++k) {
            for (ssize_t n=0; n<N; ++n) {
                for (ssize_t m=0; m<M; ++m) {
                    for (ssize_t d=0; d<D; ++d) {
                        O(k,n,m,d) = A(k,n,d) - B(k,n,m,d);
                    }
                }
            }
        }
        return out;
    }

    // 3. Broadcast Case: (K, N, M, D) - (K, N, D)
    if (a_req.ndim == 4 && b_req.ndim == 3) {
        ssize_t K = b_req.shape[0];
        ssize_t N = b_req.shape[1];
        ssize_t M = a_req.shape[2];
        ssize_t D = b_req.shape[2];

        py::array_t<double> out(a_req.shape);
        auto A = a.unchecked<4>();
        auto B = b.unchecked<3>();
        auto O = out.mutable_unchecked<4>();

        #pragma omp parallel for collapse(3)
        for (ssize_t k=0; k<K; ++k) {
            for (ssize_t n=0; n<N; ++n) {
                for (ssize_t m=0; m<M; ++m) {
                    for (ssize_t d=0; d<D; ++d) {
                        O(k,n,m,d) = A(k,n,m,d) - B(k,n,d);
                    }
                }
            }
        }
        return out;
    }

    throw std::runtime_error("binary_sub_generic: Unsupported broadcasting shapes");
}

inline py::array_t<double> binary_add_generic(const py::array_t<double>& a,
                                              const py::array_t<double>& b) {
    // Basic flat implementation for same-shape
    // Full broadcasting implementation would mirror binary_sub above
    auto a_req = a.request();
    auto b_req = b.request();
    if (a_req.size == b_req.size) {
        py::array_t<double> out(a_req.shape);
        Eigen::Map<const Eigen::VectorXd> va((double*)a_req.ptr, a_req.size);
        Eigen::Map<const Eigen::VectorXd> vb((double*)b_req.ptr, b_req.size);
        Eigen::Map<Eigen::VectorXd> vout((double*)out.request().ptr, out.size());
        vout = va + vb;
        return out;
    }
     throw std::runtime_error("binary_add_generic: Unsupported broadcasting shapes");
}
// ---------------------------------------------------------------------------
// IR-driven C++ codegen helpers
// ---------------------------------------------------------------------------
// Dot(Test, Trial) mass matrix
inline Eigen::MatrixXd dot_mass_test_trial(const Eigen::MatrixXd& test_vec,
                                           const Eigen::MatrixXd& trial_vec) {
    if (is_debug) {std::cout<< "-----------------dot_mass_test_trial---------------------"<<std::endl;}
    return test_vec.transpose() * trial_vec;
}

inline std::vector<Eigen::MatrixXd> dot_mass_test_trial(
    const std::vector<Eigen::MatrixXd>& test_vec,
    const Eigen::MatrixXd& trial_vec) {
    if (is_debug) {std::cout<< "-----------------dot_mass_test_trial(component-carried)---------------------"<<std::endl;}
    return contract_last_first(test_vec, trial_vec);
}

inline Eigen::MatrixXd dot_grad_grad_value(const Eigen::MatrixXd& grad_a,
                                           const Eigen::MatrixXd& grad_b) {
    
    if (is_debug) {std::cout<< "-----------------dot_grad_grad_value---------------------"<<std::endl;}
                                            // Optional: Safety check for dimensions
    if (grad_a.cols() != grad_b.rows()) {
        throw std::runtime_error("Dimension mismatch in dot_grad_grad_value: " 
                                 "cols(A) must match rows(B)");
    }

    // In Eigen, '*' is Matrix Multiplication
    return grad_a * grad_b;
}

inline Eigen::MatrixXd dot_mass_trial_test(const Eigen::MatrixXd& trial_vec,
                                           const Eigen::MatrixXd& test_vec) {
    if (is_debug) {std::cout<< "-----------------dot_mass_trial_test---------------------"<<std::endl;}
    return trial_vec.transpose() * test_vec;
}

inline std::vector<Eigen::MatrixXd> dot_mass_trial_test(
    const std::vector<Eigen::MatrixXd>& trial_vec,
    const Eigen::MatrixXd& test_vec) {
    if (is_debug) {std::cout<< "-----------------dot_mass_trial_test(component-carried)---------------------"<<std::endl;}
    return contract_last_first(trial_vec, test_vec);
}

inline std::vector<Eigen::MatrixXd> vector_trial_times_scalar_test(
    const Eigen::MatrixXd& trial_vec,
    const Eigen::MatrixXd& test_scalar) {
    if (is_debug) {std::cout<< "-----------------vector_trial_times_scalar_test---------------------"<<std::endl;}
    if (test_scalar.rows() != 1) {
        throw std::runtime_error("vector_trial_times_scalar_test expects scalar test basis as (1,n_test)");
    }
    std::vector<Eigen::MatrixXd> out;
    out.reserve(static_cast<size_t>(trial_vec.rows()));
    const Eigen::VectorXd phi_test = test_scalar.row(0).transpose();
    for (Eigen::Index comp = 0; comp < trial_vec.rows(); ++comp) {
        out.push_back(phi_test * trial_vec.row(comp));
    }
    return out;
}

inline std::vector<Eigen::MatrixXd> vector_test_times_scalar_trial(
    const Eigen::MatrixXd& test_vec,
    const Eigen::MatrixXd& trial_scalar) {
    if (is_debug) {std::cout<< "-----------------vector_test_times_scalar_trial---------------------"<<std::endl;}
    if (trial_scalar.rows() != 1) {
        throw std::runtime_error("vector_test_times_scalar_trial expects scalar trial basis as (1,n_trial)");
    }
    std::vector<Eigen::MatrixXd> out;
    out.reserve(static_cast<size_t>(test_vec.rows()));
    const Eigen::RowVectorXd phi_trial = trial_scalar.row(0);
    for (Eigen::Index comp = 0; comp < test_vec.rows(); ++comp) {
        out.push_back(test_vec.row(comp).transpose() * phi_trial);
    }
    return out;
}

// grad(Function) @ Trial (grad_func: k x d, trial: k x n) -> d x n
inline Eigen::MatrixXd dot_grad_func_trial_vec(const Eigen::MatrixXd& grad_func,
                                               const Eigen::MatrixXd& trial_vec) {
    if (is_debug) {std::cout<< "-----------------dot_grad_func_trial_vec---------------------"<<std::endl;}
    return grad_func * trial_vec;
}

// grad(basis) (vector<k x n, dim>) dotted with vector (dim,) -> k x n
inline Eigen::MatrixXd dot_grad_basis_vector(const std::vector<Eigen::MatrixXd>& grad_basis,
                                             const Eigen::VectorXd& vec) {
    if (is_debug) {std::cout<< "-----------------dot_grad_basis_vector---------------------"<<std::endl;}
    int k = static_cast<int>(grad_basis.size());
    if (k == 0) return Eigen::MatrixXd();
    int n = static_cast<int>(grad_basis[0].rows());
    int d = static_cast<int>(grad_basis[0].cols());
    if (vec.size() != d) {
        throw std::runtime_error("dot_grad_basis_vector: vector length must match gradient columns");
    }
    Eigen::MatrixXd out(k, n);
    for (int c = 0; c < k; ++c) {
        if (grad_basis[c].rows() != n || grad_basis[c].cols() != d) {
            throw std::runtime_error("dot_grad_basis_vector: inconsistent grad_basis matrix shapes");
        }
        out.row(c) = (grad_basis[c] * vec).transpose();
    }
    return out;
}

// Canonical scalar gradient basis: (d, n) dotted with spatial vector (d,) -> (1, n)
inline Eigen::MatrixXd dot_grad_basis_vector(const Eigen::MatrixXd& grad_basis,
                                             const Eigen::VectorXd& vec) {
    if (is_debug) {std::cout<< "-----------------dot_grad_basis_vector (matrix)---------------------"<<std::endl;}
    if (grad_basis.rows() != vec.size()) {
        throw std::runtime_error("dot_grad_basis_vector(matrix): vector length must match gradient rows");
    }
    Eigen::MatrixXd out(1, grad_basis.cols());
    out.row(0) = vec.transpose() * grad_basis;
    return out;
}

// Vector dotted with grad(basis) -> d x n
inline Eigen::MatrixXd dot_vec_grad(const Eigen::VectorXd& vec,
                                    const std::vector<Eigen::MatrixXd>& grad_basis) {
    if (is_debug) { std::cout << "-----------------dot_vec_grad---------------------" << std::endl; }
    
    // 1. Safety Checks
    if (grad_basis.empty()) return Eigen::MatrixXd();
    
    long k = static_cast<long>(grad_basis.size());
    
    // Validate Vector Size
    if (vec.size() != k) {
        throw std::runtime_error("Dimension mismatch: Vector size must match grad_basis size (k).");
    }

    // 2. Get Dimensions
    // grad_basis[0] is (n, d). We want output (d, n).
    long n = grad_basis[0].rows();
    long d = grad_basis[0].cols();

    // 3. Allocate Output
    Eigen::MatrixXd out(d, n);
    out.setZero();

    // 4. Compute Weighted Sum
    // vec(c) works correctly for both RowVector and ColumnVector in Eigen
    for (long c = 0; c < k; ++c) {
        // Validation check for inner matrices (optional but safe)
        if (grad_basis[c].rows() != n || grad_basis[c].cols() != d) {
             throw std::runtime_error("Irregular matrix size in grad_basis vector.");
        }
        
        // Accumulate: scalar * (n, d).transpose() -> (d, n)
        out += grad_basis[c].transpose() * vec(c);
    }

    return out;
}

inline Eigen::MatrixXd dot_vec_grad(const Eigen::RowVectorXd& vec,
                                    const std::vector<Eigen::MatrixXd>& grad_basis) {
    if (is_debug) { std::cout << "-----------------dot_vec_grad---------------------" << std::endl; }

    if (grad_basis.empty()) return Eigen::MatrixXd();
    long k = static_cast<long>(grad_basis.size());
    if (vec.size() != k) {
        throw std::runtime_error("Dimension mismatch: Vector size must match grad_basis size (k).");
    }

    long n = grad_basis[0].rows();
    long d = grad_basis[0].cols();
    Eigen::MatrixXd out(d, n);
    out.setZero();
    for (long c = 0; c < k; ++c) {
        if (grad_basis[c].rows() != n || grad_basis[c].cols() != d) {
            throw std::runtime_error("Irregular matrix size in grad_basis vector.");
        }
        out += grad_basis[c].transpose() * vec(c);
    }
    return out;
}

// Vector dotted with grad(basis) with optional spatial-vs-component disambiguation.
// If the vector length matches spatial dim and k==1, contract spatially to preserve basis rows (1 x n).
// Otherwise, if the vector length matches component count (k), contract components -> (d x n).
inline Eigen::MatrixXd vector_dot_grad_basis(const Eigen::VectorXd& vec,
                                             const std::vector<Eigen::MatrixXd>& grad_basis) {
    if (is_debug) {std::cout<< "-----------------vector_dot_grad_basis---------------------"<<std::endl;}
    int k = static_cast<int>(grad_basis.size());
    if (k == 0) return Eigen::MatrixXd();
    int d = static_cast<int>(grad_basis[0].cols());
    int vlen = static_cast<int>(vec.size());
    if (k == 1 && vlen == d) {
        Eigen::RowVectorXd row = grad_basis[0] * vec;
        return Eigen::MatrixXd(row); // (1, n)
    }
    if (vlen == k) {
        return dot_vec_grad(vec, grad_basis); // (d, n)
    }
    throw std::runtime_error("vector_dot_grad_basis: incompatible shapes.");
}

// Canonical scalar gradient basis: vector (d,) dotted with (d, n) -> (1, n)
inline Eigen::MatrixXd vector_dot_grad_basis(const Eigen::VectorXd& vec,
                                             const Eigen::MatrixXd& grad_basis) {
    if (is_debug) {std::cout<< "-----------------vector_dot_grad_basis (matrix)---------------------"<<std::endl;}
    if (grad_basis.rows() != vec.size()) {
        throw std::runtime_error("vector_dot_grad_basis(matrix): vector length must match gradient rows");
    }
    Eigen::MatrixXd out(1, grad_basis.cols());
    out.row(0) = vec.transpose() * grad_basis;
    return out;
}

// Vector basis (k,n) dotted with grad(value) (k,d) -> (d,n)
inline Eigen::MatrixXd vector_dot_grad_value(const Eigen::MatrixXd& vec_basis,
                                             const Eigen::MatrixXd& grad_value) {
    if (is_debug) {std::cout<< "-----------------vector_dot_grad_value---------------------"<<std::endl;}
    if (grad_value.rows() == 1 && vec_basis.rows() == grad_value.cols()) {
        Eigen::MatrixXd out(1, vec_basis.cols());
        const Eigen::VectorXd g = grad_value.row(0).transpose();
        for (int j = 0; j < vec_basis.cols(); ++j) {
            out(0, j) = vec_basis.col(j).dot(g);
        }
        return out;
    }
    if (vec_basis.rows() != grad_value.rows()) {
        throw std::runtime_error("vector_dot_grad_value: incompatible shapes.");
    }
    const int n = static_cast<int>(vec_basis.cols());
    const int d = static_cast<int>(grad_value.cols());
    Eigen::MatrixXd out(d, n);
    for (int j = 0; j < n; ++j) {
        out.col(j).noalias() = grad_value.transpose() * vec_basis.col(j);
    }
    return out;
}

// Contract vector basis (components) with grad basis: sum_c grad[c][:,r] * vec_basis[c,:]
// Produces one matrix per spatial component (d) of shape (rows x cols).
inline std::vector<Eigen::MatrixXd> dot_vec_grad_components(const Eigen::MatrixXd& vec_basis,
                                                            const std::vector<Eigen::MatrixXd>& grad_basis,
                                                            bool swap_roles=false) {
    if (is_debug) {std::cout<< "-----------------dot_vec_grad_components---------------------"<<std::endl;}
    int k = static_cast<int>(grad_basis.size());
    if (k == 0) return {};
    int d = static_cast<int>(grad_basis[0].cols());
    int n_rows = swap_roles ? static_cast<int>(vec_basis.cols()) : static_cast<int>(grad_basis[0].rows());
    int n_cols = swap_roles ? static_cast<int>(grad_basis[0].rows()) : static_cast<int>(vec_basis.cols());
    std::vector<Eigen::MatrixXd> out(static_cast<size_t>(d), Eigen::MatrixXd::Zero(n_rows, n_cols));
    for (int r = 0; r < d; ++r) {
        for (int c = 0; c < k; ++c) {
            if (!swap_roles) {
                // rows from grad_basis (test), cols from vec_basis (trial)
                out[r].noalias() += grad_basis[c].col(r) * vec_basis.row(c);
            } else {
                // rows from vec_basis (test), cols from grad_basis (trial)
                out[r].noalias() += vec_basis.row(c).transpose() * grad_basis[c].col(r).transpose();
            }
        }
    }
    return out;
}

inline std::vector<Eigen::MatrixXd> dot_vec_grad_components(const std::vector<Eigen::MatrixXd>& vec_basis,
                                                            const std::vector<Eigen::MatrixXd>& grad_basis,
                                                            bool swap_roles=false) {
    if (is_debug) {std::cout<< "-----------------dot_vec_grad_components(stack,stack)---------------------"<<std::endl;}
    const int k = static_cast<int>(grad_basis.size());
    if (k == 0 || vec_basis.empty()) return {};
    if (static_cast<int>(vec_basis.size()) != k) {
        throw std::runtime_error("dot_vec_grad_components(stack,stack): component mismatch.");
    }
    const int n_vec = static_cast<int>(vec_basis[0].rows());
    const int j_dim = static_cast<int>(vec_basis[0].cols());
    const int n_grad = static_cast<int>(grad_basis[0].rows());
    const int d_dim = static_cast<int>(grad_basis[0].cols());
    const int n_rows = swap_roles ? n_vec : n_grad;
    const int n_cols = swap_roles ? n_grad : n_vec;
    std::vector<Eigen::MatrixXd> out(static_cast<size_t>(j_dim * d_dim), Eigen::MatrixXd::Zero(n_rows, n_cols));
    for (int c = 0; c < k; ++c) {
        if (vec_basis[static_cast<size_t>(c)].rows() != n_vec || vec_basis[static_cast<size_t>(c)].cols() != j_dim) {
            throw std::runtime_error("dot_vec_grad_components(stack,stack): inconsistent lhs component shape.");
        }
        if (grad_basis[static_cast<size_t>(c)].rows() != n_grad || grad_basis[static_cast<size_t>(c)].cols() != d_dim) {
            throw std::runtime_error("dot_vec_grad_components(stack,stack): inconsistent rhs component shape.");
        }
        for (int j = 0; j < j_dim; ++j) {
            for (int r = 0; r < d_dim; ++r) {
                auto& block = out[static_cast<size_t>(j * d_dim + r)];
                if (!swap_roles) {
                    block.noalias() += grad_basis[static_cast<size_t>(c)].col(r) *
                                       vec_basis[static_cast<size_t>(c)].col(j).transpose();
                } else {
                    block.noalias() += vec_basis[static_cast<size_t>(c)].col(j) *
                                       grad_basis[static_cast<size_t>(c)].col(r).transpose();
                }
            }
        }
    }
    return out;
}

// Component-wise contraction for Hessian·vec outputs (k,n,2) with a spatial vector (2,) -> (k,n)
inline Eigen::MatrixXd hessvec_dot_vector_basis(const std::vector<Eigen::MatrixXd>& hessvec_basis,
                                                const Eigen::VectorXd& vec) {
    if (is_debug) {std::cout<< "-----------------hessvec_dot_vector_basis---------------------"<<std::endl;}
    int k = static_cast<int>(hessvec_basis.size());
    if (k == 0) return Eigen::MatrixXd();
    int n = static_cast<int>(hessvec_basis[0].rows());
    Eigen::MatrixXd out(k, n);
    for (int c = 0; c < k; ++c) {
        out.row(c) = hessvec_basis[c] * vec;
    }
    return out;
}


/**
 * Scalar Trial (n_trial) times grad(Test) (k, n_test, d) -> (k, n_test, n_trial, d).
 * Representation: std::vector of length k*d, each entry is an (n_test x n_trial)
 * block corresponding to a single spatial column.
 */
inline std::vector<Eigen::MatrixXd> scalar_trial_times_grad_test(
    const std::vector<Eigen::MatrixXd>& grad_test, // Size k, Mats (n_test, d)
    const Eigen::VectorXd& trial_vals)             // Size n_trial
{
    if (is_debug) {std::cout << "-----------------scalar_trial_times_grad_test---------------------" << std::endl;}

    int k = static_cast<int>(grad_test.size());
    if (k == 0) return {};

    long d = grad_test[0].cols();

    std::vector<Eigen::MatrixXd> out(static_cast<size_t>(k * d));
    for (int c = 0; c < k; ++c) {
        for (int j = 0; j < d; ++j) {
            // Outer product: (n_test,) * (n_trial,)^T -> (n_test x n_trial)
            out[static_cast<size_t>(c * d + j)] =
                grad_test[c].col(j) * trial_vals.transpose();
        }
    }

    return out;
}

/**
 * Grad(Trial) (k,n_trial,d) times scalar Test (n_test) -> mixed gradient (k,n_test,n_trial,d).
 * Representation matches scalar_trial_times_grad_test (k*d entries of n_test x n_trial).
 */
inline std::vector<Eigen::MatrixXd> grad_trial_times_scalar_test(
    const std::vector<Eigen::MatrixXd>& grad_trial,
    const Eigen::VectorXd& test_vals) {
    int k = static_cast<int>(grad_trial.size());
    if (k == 0) return {};
    long d = grad_trial[0].cols();

    std::vector<Eigen::MatrixXd> out(static_cast<size_t>(k * d));
    for (int c = 0; c < k; ++c) {
        for (int j = 0; j < d; ++j) {
            // Outer product: (n_test,) * (n_trial,)^T -> (n_test x n_trial)
            out[static_cast<size_t>(c * d + j)] =
                test_vals * grad_trial[c].col(j).transpose();
        }
    }
    return out;
}

// Scalar basis row (1,n) scaled by a value vector (k,) -> matrix (k,n).
inline Eigen::MatrixXd scalar_basis_times_vector(
    const Eigen::MatrixXd& scalar_basis,
    const Eigen::VectorXd& vector_vals) {
    if (is_debug) {std::cout<< "-----------------scalar_basis_times_vector---------------------"<<std::endl;}
    if (scalar_basis.rows() != 1) {
        throw std::runtime_error("scalar_basis_times_vector expects scalar basis as (1,n)");
    }
    return vector_vals * scalar_basis;
}

inline Eigen::MatrixXd scalar_basis_times_vector(
    const Eigen::VectorXd& scalar_basis,
    const Eigen::VectorXd& vector_vals) {
    if (is_debug) {std::cout<< "-----------------scalar_basis_times_vector(vec)---------------------"<<std::endl;}
    return vector_vals * scalar_basis.transpose();
}

// Scalar basis row (1,n) scaled by a value/const matrix (k,d) -> tensor basis (k,n,d).
// Representation matches the existing grad-stack convention: one (n x d) matrix per component row.
inline std::vector<Eigen::MatrixXd> scalar_basis_times_matrix_tensor(
    const Eigen::MatrixXd& scalar_basis,
    const Eigen::MatrixXd& value_mat) {
    if (is_debug) {std::cout<< "-----------------scalar_basis_times_matrix_tensor---------------------"<<std::endl;}
    if (scalar_basis.rows() != 1) {
        throw std::runtime_error("scalar_basis_times_matrix_tensor expects scalar basis as (1,n)");
    }
    std::vector<Eigen::MatrixXd> out;
    out.reserve(static_cast<size_t>(value_mat.rows()));
    const Eigen::VectorXd phi = scalar_basis.transpose();
    for (Eigen::Index comp = 0; comp < value_mat.rows(); ++comp) {
        out.push_back(phi * value_mat.row(comp));
    }
    return out;
}

// Inner product of grad stacks -> n x n
inline Eigen::MatrixXd inner_grad_grad(const std::vector<Eigen::MatrixXd>& test,
                                       const std::vector<Eigen::MatrixXd>& trial) {
    if (is_debug) {std::cout<< "-----------------inner_grad_grad---------------------"<<std::endl;}
    int k = static_cast<int>(test.size());
    if (k == 0) return Eigen::MatrixXd();
    
    int n_test = static_cast<int>(test[0].rows());
    int n_trial = static_cast<int>(trial[0].rows()); // Read trial rows separately
    
    // Initialize with correct rectangular dimensions
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n_test, n_trial); 

    for (int c = 0; c < k; ++c) {
        // Optional safety: Check if inner dimensions match (columns of test == columns of trial)
        // if(test[c].cols() != trial[c].cols()) throw ...
        
        out += test[c] * trial[c].transpose();
    }
    return out;
}

// Inner product of canonical scalar-grad basis/value matrices (d,n) -> (n,n)
inline Eigen::MatrixXd inner_grad_grad(const Eigen::MatrixXd& test,
                                       const Eigen::MatrixXd& trial) {
    if (is_debug) {std::cout<< "-----------------inner_grad_grad(rank1_basis)---------------------"<<std::endl;}
    if (test.rows() != trial.rows()) {
        throw std::runtime_error(
            "inner_grad_grad(rank1_basis): spatial dimensions must match");
    }
    return test.transpose() * trial;
}

// Inner of grad(test) (k,n,d) with grad(value) (k,d) -> (n,)
inline Eigen::VectorXd inner_grad_const(const std::vector<Eigen::MatrixXd>& grad_test,
                                        const Eigen::MatrixXd& grad_val) {
    if (is_debug) {std::cout<< "-----------------inner_grad_const---------------------"<<std::endl;}
    int k = static_cast<int>(grad_test.size());
    if (k == 0) return Eigen::VectorXd();
    int n = static_cast<int>(grad_test[0].rows());
    Eigen::VectorXd out = Eigen::VectorXd::Zero(n);
    for (int c = 0; c < k; ++c) {
        out += grad_test[c] * grad_val.row(c).transpose();
    }
    return out;
}

// Inner of canonical scalar grad(test) basis (d,n) with grad(value) (1,d) -> (n,)
inline Eigen::VectorXd inner_grad_const(const Eigen::MatrixXd& grad_test,
                                        const Eigen::MatrixXd& grad_val) {
    if (is_debug) {std::cout<< "-----------------inner_grad_const(rank1_basis)---------------------"<<std::endl;}
    if (grad_val.rows() != 1) {
        throw std::runtime_error("inner_grad_const(rank1_basis): expected scalar grad(value) with shape (1,d)");
    }
    if (grad_test.rows() == grad_val.cols()) {
        return grad_test.transpose() * grad_val.row(0).transpose();
    }
    if (grad_test.cols() == grad_val.cols()) {
        return grad_test * grad_val.row(0).transpose();
    }
    throw std::runtime_error("inner_grad_const(rank1_basis): incompatible scalar grad basis/value shapes");
}

// Inner product of Hessian stacks stored as (n,4) flattened blocks
inline Eigen::MatrixXd inner_hessian_hessian(const std::vector<Eigen::MatrixXd>& test,
                                             const std::vector<Eigen::MatrixXd>& trial) {
    if (is_debug) {std::cout<< "-----------------inner_hessian_hessian---------------------"<<std::endl;}
    int k = static_cast<int>(test.size());
    if (k == 0) return Eigen::MatrixXd();
    int n_test = static_cast<int>(test[0].rows());
    int n_trial = static_cast<int>(trial[0].rows());
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n_test, n_trial);
    for (int c = 0; c < k; ++c) {
        out += test[c] * trial[c].transpose();
    }
    return out;
}

// Inner of Hessian(value) (k,4) with Hessian(test) (k,n,4) -> (n,)
inline Eigen::VectorXd inner_hessian_const(const std::vector<Eigen::MatrixXd>& hess_test,
                                           const Eigen::MatrixXd& hess_value) {
    if (is_debug) {std::cout<< "-----------------inner_hessian_const---------------------"<<std::endl;}
    int k = static_cast<int>(hess_test.size());
    if (k == 0) return Eigen::VectorXd();
    int n = static_cast<int>(hess_test[0].rows());
    if (hess_value.rows() != k || hess_value.cols() != 4) {
        throw std::runtime_error("inner_hessian_const: value Hessian must be (k,4)");
    }
    Eigen::VectorXd out = Eigen::VectorXd::Zero(n);
    for (int c = 0; c < k; ++c) {
        if (hess_test[c].cols() != 4 || hess_test[c].rows() != n) {
            throw std::runtime_error("inner_hessian_const: inconsistent test Hessian shape");
        }
        out += hess_test[c] * hess_value.row(c).transpose();
    }
    return out;
}

// Hessian (basis/value flattened as (n,4) or (k,4)) dotted with spatial vector -> grad-like
inline std::vector<Eigen::MatrixXd> hessian_dot_vector_basis(const std::vector<Eigen::MatrixXd>& hess,
                                                             const Eigen::VectorXd& vec) {
    if (is_debug) {std::cout<< "-----------------hessian_dot_vector_basis---------------------"<<std::endl;}
    int k = static_cast<int>(hess.size());
    if (k == 0) return {};
    int n = static_cast<int>(hess[0].rows());
    std::vector<Eigen::MatrixXd> out(static_cast<size_t>(k), Eigen::MatrixXd(n, 2));
    double v0 = vec(0), v1 = vec(1);
    for (int c = 0; c < k; ++c) {
        const auto& H = hess[c];
        for (int j = 0; j < n; ++j) {
            double h00 = H(j, 0), h01 = H(j, 1), h10 = H(j, 2), h11 = H(j, 3);
            out[c](j, 0) = h00 * v0 + h01 * v1;
            out[c](j, 1) = h10 * v0 + h11 * v1;
        }
    }
    return out;
}

inline Eigen::MatrixXd hessian_dot_vector_value(const Eigen::MatrixXd& hess,
                                                const Eigen::VectorXd& vec) {
    if (is_debug) {std::cout<< "-----------------hessian_dot_vector_value---------------------"<<std::endl;}
    int k = static_cast<int>(hess.rows());
    Eigen::MatrixXd out(k, 2);
    double v0 = vec(0), v1 = vec(1);
    for (int c = 0; c < k; ++c) {
        double h00 = hess(c, 0), h01 = hess(c, 1), h10 = hess(c, 2), h11 = hess(c, 3);
        out(c, 0) = h00 * v0 + h01 * v1;
        out(c, 1) = h10 * v0 + h11 * v1;
    }
    return out;
}

// vector · Hessian(basis/value)
inline std::vector<Eigen::MatrixXd> vector_dot_hessian_basis(const Eigen::VectorXd& vec,
                                                             const std::vector<Eigen::MatrixXd>& hess) {
    if (is_debug) {std::cout << "-----------------vector_dot_hessian_basis---------------------" << std::endl;}

    int k = static_cast<int>(hess.size());
    if (k == 0) return {};
    int n = static_cast<int>(hess[0].rows());
    int vec_len = static_cast<int>(vec.size());

    // Assume 2D physics for now (hessian is flattened Nx4)
    // 4 columns correspond to 2x2 matrix flattened: h00, h01, h10, h11
    
    // ---------------------------------------------------------
    // CASE A: Component Contraction (Sum over k)
    // Python: if v.shape[0] == k and k > 1
    // ---------------------------------------------------------
    if (vec_len == k && k > 1) {
        // Output is split into 2 matrices (for d1=0 and d1=1)
        std::vector<Eigen::MatrixXd> out(2, Eigen::MatrixXd(n, 2));
        
        for (int j = 0; j < n; ++j) {
            double acc00 = 0.0, acc01 = 0.0, acc10 = 0.0, acc11 = 0.0;
            for (int c = 0; c < k; ++c) {
                // Here vec(c) is the weight for the c-th component
                double v = vec(c);
                const auto& H = hess[c];
                acc00 += v * H(j, 0); // Component contribution to h00
                acc01 += v * H(j, 1);
                acc10 += v * H(j, 2);
                acc11 += v * H(j, 3);
            }
            // Reconstruct the structure:
            // out[0] gets the first row of the 'virtual' tensor
            out[0](j, 0) = acc00;
            out[0](j, 1) = acc01;
            // out[1] gets the second row
            out[1](j, 0) = acc10;
            out[1](j, 1) = acc11;
        }
        return out;
    }

    // ---------------------------------------------------------
    // CASE B: Spatial Contraction (Vector-Matrix product)
    // Python: elif v.shape[0] == d1 and k == 1
    // ---------------------------------------------------------
    // In this context, d1 is 2. So we check if vec size is 2.
    else if (k == 1 && vec_len == 2) {
        // Output is a single matrix (1, n, d2) -> std::vector size 1
        std::vector<Eigen::MatrixXd> out(1, Eigen::MatrixXd(n, 2));
        
        double v0 = vec(0);
        double v1 = vec(1);
        const auto& H = hess[0]; // There is only 1 component

        for (int j = 0; j < n; ++j) {
            // H is flattened 2x2: [h00, h01, h10, h11]
            double h00 = H(j, 0);
            double h01 = H(j, 1);
            double h10 = H(j, 2);
            double h11 = H(j, 3);

            // Operation: [v0, v1] @ [[h00, h01], [h10, h11]]
            // Row 0 col 0: v0*h00 + v1*h10
            // Row 0 col 1: v0*h01 + v1*h11
            
            out[0](j, 0) = v0 * h00 + v1 * h10;
            out[0](j, 1) = v0 * h01 + v1 * h11;
        }
        return out;
    }

    else {
        throw std::runtime_error("vector_dot_hessian_basis: Incompatible shapes.");
    }
}

inline Eigen::MatrixXd vector_dot_hessian_value(const Eigen::VectorXd& vec,
                                                const Eigen::MatrixXd& hess) {
    if (is_debug) {std::cout<< "-----------------vector_dot_hessian_value---------------------"<<std::endl;}
    int k = static_cast<int>(hess.rows());
    int vec_len = static_cast<int>(vec.size());
    if (vec_len == k && k > 1) {
        double acc00 = 0.0, acc01 = 0.0, acc10 = 0.0, acc11 = 0.0;
        for (int c = 0; c < k; ++c) {
            double v = vec(c);
            acc00 += v * hess(c, 0);
            acc01 += v * hess(c, 1);
            acc10 += v * hess(c, 2);
            acc11 += v * hess(c, 3);
        }
        Eigen::MatrixXd out(2, 2);
        out(0, 0) = acc00; out(0, 1) = acc01;
        out(1, 0) = acc10; out(1, 1) = acc11;
        return out;
    }
    if (k == 1 && vec_len == 2) {
        Eigen::MatrixXd out(1, 2);
        const double v0 = vec(0);
        const double v1 = vec(1);
        const double h00 = hess(0, 0);
        const double h01 = hess(0, 1);
        const double h10 = hess(0, 2);
        const double h11 = hess(0, 3);
        out(0, 0) = v0 * h00 + v1 * h10;
        out(0, 1) = v0 * h01 + v1 * h11;
        return out;
    }
    throw std::runtime_error("vector_dot_hessian_value: Incompatible shapes.");
}

// Inner of grad(value) (k,d) with grad(test) (k,n,d) -> (n,)
inline Eigen::VectorXd inner_const_grad(const Eigen::MatrixXd& grad_val,
                                        const std::vector<Eigen::MatrixXd>& grad_test) {
    return inner_grad_const(grad_test, grad_val);
}

inline Eigen::VectorXd inner_const_grad(const Eigen::MatrixXd& grad_val,
                                        const Eigen::MatrixXd& grad_test) {
    return inner_grad_const(grad_test, grad_val);
}

inline Eigen::VectorXd inner_rank2_value_rank2_basis(const Eigen::MatrixXd& value_rank2,
                                                     const std::vector<Eigen::MatrixXd>& basis_rank2) {
    if (is_debug) {std::cout<< "-----------------inner_rank2_value_rank2_basis---------------------"<<std::endl;}
    if (basis_rank2.empty()) return Eigen::VectorXd();
    const int r0 = static_cast<int>(basis_rank2.size());
    const int n = static_cast<int>(basis_rank2[0].rows());
    const int r1 = static_cast<int>(basis_rank2[0].cols());
    if (value_rank2.rows() != r0 || value_rank2.cols() != r1) {
        throw std::runtime_error("inner_rank2_value_rank2_basis: incompatible value/basis shapes");
    }
    Eigen::VectorXd out = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < r0; ++i) {
        if (basis_rank2[i].rows() != n || basis_rank2[i].cols() != r1) {
            throw std::runtime_error("inner_rank2_value_rank2_basis: inconsistent basis matrix shapes");
        }
        out.noalias() += basis_rank2[i] * value_rank2.row(i).transpose();
    }
    return out;
}

inline Eigen::VectorXd inner_rank2_basis_rank2_value(const std::vector<Eigen::MatrixXd>& basis_rank2,
                                                     const Eigen::MatrixXd& value_rank2) {
    return inner_rank2_value_rank2_basis(value_rank2, basis_rank2);
}

// elementwise add/sub for common scalar/vector/matrix types
inline double binary_add(double a, double b) { return a + b; }
inline double binary_sub(double a, double b) { return a - b; }
inline Eigen::VectorXd binary_add(const Eigen::VectorXd& a, const Eigen::VectorXd& b) { return a + b; }
inline Eigen::VectorXd binary_sub(const Eigen::VectorXd& a, const Eigen::VectorXd& b) { return a - b; }
inline Eigen::MatrixXd binary_add(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) { return a + b; }
inline Eigen::MatrixXd binary_sub(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) { return a - b; }

// Load variable component value: coeffs (nE,nU), basis (nE,nQ,nU)
inline double load_variable_component(const py::detail::unchecked_reference<double, 2>& coeffs,
                                      const py::detail::unchecked_reference<double, 3>& basis,
                                      ssize_t e, ssize_t q, int s0, int s1) {
    const int n = s1 - s0;
    if (n <= 0) return 0.0;
    const double* cptr = coeffs.data(e, s0);
    const double* bptr = basis.data(e, q, s0);

    // Numpy tables may be non-contiguous (e.g. views/transposes used to avoid
    // large copies during setup). Respect the actual axis strides by deriving
    // a runtime inner stride from adjacent pointers.
    ptrdiff_t cstride = 1;
    ptrdiff_t bstride = 1;
    if (n > 1) {
        const double* cptr1 = coeffs.data(e, s0 + 1);
        const double* bptr1 = basis.data(e, q, s0 + 1);
        cstride = cptr1 - cptr;
        bstride = bptr1 - bptr;
        if (cstride == 0) cstride = 1;
        if (bstride == 0) bstride = 1;
    }

    Eigen::Map<const Eigen::VectorXd, 0, Eigen::InnerStride<>> cvec(
        cptr, n, Eigen::InnerStride<>(cstride));
    Eigen::Map<const Eigen::VectorXd, 0, Eigen::InnerStride<>> bvec(
        bptr, n, Eigen::InnerStride<>(bstride));
    return cvec.dot(bvec);
}

// gradient of a component: accumulates coefficients * phys grad
inline void gradient_component(Eigen::MatrixXd& out, int row,
                               const py::detail::unchecked_reference<double, 2>& coeffs,
                               const py::detail::unchecked_reference<double, 4>& gref,
                               const Eigen::Matrix<double, 2, 2>& Jloc,
                               ssize_t e, ssize_t q, int s0, int s1) {
    const int n = s1 - s0;
    if (n <= 0) {
        out(row, 0) = 0.0;
        out(row, 1) = 0.0;
        return;
    }

    const double* cptr = coeffs.data(e, s0);
    const double* gptr0 = gref.data(e, q, s0, 0);
    const double* gptr1 = gref.data(e, q, s0, 1);

    // Derive runtime strides to support arbitrary numpy layouts.
    ptrdiff_t cstride = 1;
    ptrdiff_t gstride = 2;  // default for contiguous (...,n,2) layout
    if (n > 1) {
        const double* cptr1 = coeffs.data(e, s0 + 1);
        const double* gptr0_1 = gref.data(e, q, s0 + 1, 0);
        cstride = cptr1 - cptr;
        gstride = gptr0_1 - gptr0;
        if (cstride == 0) cstride = 1;
        if (gstride == 0) gstride = 2;
    }

    Eigen::Map<const Eigen::VectorXd, 0, Eigen::InnerStride<>> cvec(
        cptr, n, Eigen::InnerStride<>(cstride));
    Eigen::Map<const Eigen::VectorXd, 0, Eigen::InnerStride<>> g0(
        gptr0, n, Eigen::InnerStride<>(gstride));
    Eigen::Map<const Eigen::VectorXd, 0, Eigen::InnerStride<>> g1(
        gptr1, n, Eigen::InnerStride<>(gstride));

    const double s_g0 = cvec.dot(g0);
    const double s_g1 = cvec.dot(g1);
    const double gx = Jloc(0, 0) * s_g0 + Jloc(1, 0) * s_g1;
    const double gy = Jloc(0, 1) * s_g0 + Jloc(1, 1) * s_g1;
    // Store as (ncomp, dim): rows are components, cols are spatial dimensions
    out(row, 0) = gx;
    out(row, 1) = gy;
}

// Grad stack dotted with trial basis
inline Eigen::MatrixXd dot_grad_trial(const std::vector<Eigen::MatrixXd>& grad_stack,
                                      const Eigen::MatrixXd& trial) {

    if (is_debug) {std::cout<< "-----------------dot_grad_trial---------------------"<<std::endl;}
    int k = static_cast<int>(grad_stack.size());
    int n = static_cast<int>(trial.cols());
    Eigen::MatrixXd out(grad_stack[0].cols(), n);
    out.setZero();
    for (int c = 0; c < k; ++c) {
        out += grad_stack[c].transpose() * trial.row(c);
    }
    return out;
}


inline std::vector<Eigen::MatrixXd> dot_grad_basis_with_grad_value(
    const std::vector<Eigen::MatrixXd>& grad_basis, // Shape: (k, n, d)
    const Eigen::MatrixXd& grad_value) {            // Shape: (d, j)

    if (is_debug) {std::cout << "-----------------dot_grad_basis_with_grad_value---------------------" << std::endl;}

    // 1. Get Dimensions from the Basis
    int k = static_cast<int>(grad_basis.size()); // The 'i' dimension (List size)
    if (k == 0) return {};

    int n = static_cast<int>(grad_basis[0].rows()); // The 'n' dimension
    int d = static_cast<int>(grad_basis[0].cols()); // The 'd' dimension (Inner dimension)
    if (k != d) {
        throw std::runtime_error("dot_grad_basis_with_grad_value: currently expects k == d (square vector gradients)");
    }

    // 2. Validation
    // grad_value is the gradient of a vector value: shape (k, d).
    if (grad_value.rows() != k || grad_value.cols() != d) {
        throw std::runtime_error("dot_grad_basis_with_grad_value: Dimension mismatch. grad_value must be (k, d).");
    }
    
    // Let 'j' be the number of columns in grad_value
    // The resulting matrices will have shape (n, j)
    
    // 3. Initialize Output
    // We create a vector of k matrices.
    std::vector<Eigen::MatrixXd> out(static_cast<size_t>(k));

    // 4. Perform the Contraction
    // Loop over 'i' (The free list index)
    for (int i = 0; i < k; ++i) {
        
        // Safety check for consistent basis shapes
        if (grad_basis[i].rows() != n || grad_basis[i].cols() != d) {
            throw std::runtime_error("dot_grad_basis_with_grad_value: Inconsistent basis component shape");
        }

        // MATRIX MULTIPLICATION LOGIC:
        // Input:  (n, d)
        // Weight: (d, j)
        // Result: (n, j)
        // The '*' operator automatically sums (contracts) over the 'd' axis.
        out[i] = grad_basis[i] * grad_value; 
    }

    return out;
}

inline std::vector<Eigen::MatrixXd> dot_grad_value_with_grad_basis(
    const Eigen::MatrixXd& grad_value,
    const std::vector<Eigen::MatrixXd>& grad_basis) {
    if (is_debug) {std::cout << "-----------------dot_grad_value_with_grad_basis---------------------" << std::endl;}
    // 1. Checks
    int k = static_cast<int>(grad_basis.size());
    if (k == 0) return {};
    int n = static_cast<int>(grad_basis[0].rows());
    int d = static_cast<int>(grad_basis[0].cols());

    // 2. Flatten Basis: (k, n, d) -> (k, n*d)
    // We want each ROW of 'flat' to contain the full data of one matrix from the vector.
    // Note: Eigen defaults to Column-Major. We must map carefully.
    Eigen::MatrixXd flat(k, n * d);
    
    for (int c = 0; c < k; ++c) {
        // Map the (n,d) matrix c as a vector and copy it to the row c of flat
        // We use Map with the input matrix's storage order
        Eigen::Map<const Eigen::VectorXd> vec_view(grad_basis[c].data(), grad_basis[c].size());
        flat.row(c) = vec_view;
    }

    // 3. Multiply: (k, k) * (k, n*d) -> (k, n*d)
    Eigen::MatrixXd prod = grad_value * flat;

    // 4. Unflatten: (k, n*d) -> vector of (n, d)
    std::vector<Eigen::MatrixXd> out; 
    out.reserve(k);

    for (int c = 0; c < k; ++c) {
        Eigen::MatrixXd mat(n, d);
        // Copy the row back into the matrix structure
        // We must match the order we flattened with (Column-Major copy)
        Eigen::VectorXd row_vec = prod.row(c);
        // Map the target matrix memory and copy the vector into it
        Eigen::Map<Eigen::VectorXd>(mat.data(), mat.size()) = row_vec;
        out.push_back(mat);
    }

    return out;
}

// value (k,) dotted with grad (k,d) -> (d,)
inline Eigen::VectorXd dot_value_with_grad(const Eigen::VectorXd& value_vec,
                                           const Eigen::MatrixXd& grad_mat) {
    if (is_debug) {std::cout<< "-----------------dot_value_with_grad---------------------"<<std::endl;}
    if (grad_mat.rows() != value_vec.size()) {
        throw std::runtime_error("dot_value_with_grad: incompatible shapes");
    }
    return grad_mat.transpose() * value_vec;
}

// grad (k,d) dotted with value (k,) -> (d,)  (contracts components)
inline Eigen::VectorXd dot_grad_with_value(const Eigen::MatrixXd& grad_mat,
                                           const Eigen::VectorXd& value_vec) {
    if (is_debug) {std::cout<< "-----------------dot_grad_with_value---------------------"<<std::endl;}
    if (grad_mat.rows() != value_vec.size()) {
        throw std::runtime_error("dot_grad_with_value: incompatible shapes");
    }
    return grad_mat * value_vec;
}

inline Eigen::VectorXd const_vector_dot_basis_1d(const Eigen::VectorXd& const_vec,
                                                 const Eigen::MatrixXd& basis) {
    //
    // Constant vector dotted with a rank-1 basis carrier -> basis vector.
    // Supports both component-first storage (k,n) and free-last storage (n,k).
    //
    if (is_debug) {std::cout<< "-----------------const_vector_dot_basis_1d---------------------"<<std::endl;}
    if (basis.rows() == const_vec.size()) {
        return basis.transpose() * const_vec;
    }
    if (basis.cols() == const_vec.size()) {
        return basis * const_vec;
    }
    throw std::runtime_error("const_vector_dot_basis_1d: incompatible shapes "
                             + std::to_string(basis.rows()) + "x" + std::to_string(basis.cols())
                             + " and vec(" + std::to_string(const_vec.size()) + ")");
}


template <typename Derived,
          typename std::enable_if_t<
              !std::is_same_v<std::decay_t<Derived>, Eigen::VectorXd>
              && !std::is_same_v<std::decay_t<Derived>, Eigen::MatrixXd>,
              int> = 0>
inline Eigen::VectorXd const_vector_dot_basis_1d(const Eigen::MatrixBase<Derived>& const_vec_expr,
                                                 const Eigen::MatrixXd& basis) {
    Eigen::MatrixXd mat_like = const_vec_expr;
    Eigen::VectorXd const_vec;
    if (mat_like.rows() == 1) {
        const_vec = mat_like.row(0).transpose();
    } else if (mat_like.cols() == 1) {
        const_vec = mat_like.col(0);
    } else {
        throw std::runtime_error("const_vector_dot_basis_1d: expected a row/column vector expression");
    }
    return const_vector_dot_basis_1d(const_vec, basis);
}

inline Eigen::VectorXd const_vector_dot_basis_1d(const Eigen::VectorXd& const_vec,
                                                 const std::vector<Eigen::MatrixXd>& basis) {
    if (is_debug) {std::cout<< "-----------------const_vector_dot_basis_1d(vector)---------------------"<<std::endl;}
    if (basis.empty()) {
        return Eigen::VectorXd();
    }
    if (basis.size() != 1) {
        throw std::runtime_error("const_vector_dot_basis_1d(vector): expected a scalar carried basis (size 1)");
    }
    return const_vector_dot_basis_1d(const_vec, basis[0]);
}

inline Eigen::MatrixXd basis_dot_const_vector(const Eigen::MatrixXd& basis,
                                              const Eigen::VectorXd& const_vec) {
    if (is_debug) {std::cout<< "-----------------basis_dot_const_vector---------------------"<<std::endl;}
    if (basis.rows() == const_vec.size()) {
        Eigen::MatrixXd out(1, basis.cols());
        out.row(0) = basis.transpose() * const_vec;
        return out;
    }
    if (basis.cols() == const_vec.size()) {
        Eigen::MatrixXd out(1, basis.rows());
        out.row(0) = (basis * const_vec).transpose();
        return out;
    }
    throw std::runtime_error("basis_dot_const_vector: incompatible shapes "
                             + std::to_string(basis.rows()) + "x" + std::to_string(basis.cols())
                             + " and vec(" + std::to_string(const_vec.size()) + ")");
}

inline Eigen::MatrixXd basis_dot_const_vector(const std::vector<Eigen::MatrixXd>& basis,
                                              const Eigen::VectorXd& const_vec) {
    if (is_debug) {std::cout<< "-----------------basis_dot_const_vector(vector)---------------------"<<std::endl;}
    if (basis.empty()) {
        return Eigen::MatrixXd(0, 0);
    }
    const int carried = static_cast<int>(basis.size());
    int basis_len = -1;
    Eigen::MatrixXd out;
    for (int c = 0; c < carried; ++c) {
        const auto& B = basis[static_cast<size_t>(c)];
        Eigen::VectorXd contracted;
        if (B.rows() == const_vec.size()) {
            contracted = B.transpose() * const_vec;
        } else if (B.cols() == const_vec.size()) {
            contracted = B * const_vec;
        } else {
            throw std::runtime_error("basis_dot_const_vector(vector): incompatible component shape "
                                     + std::to_string(B.rows()) + "x" + std::to_string(B.cols())
                                     + " and vec(" + std::to_string(const_vec.size()) + ")");
        }
        if (basis_len < 0) {
            basis_len = static_cast<int>(contracted.size());
            out.resize(carried, basis_len);
        } else if (basis_len != contracted.size()) {
            throw std::runtime_error("basis_dot_const_vector(vector): inconsistent basis lengths across components");
        }
        out.row(c) = contracted.transpose();
    }
    return out;
}

inline Eigen::VectorXd matrix_like_to_vector(const Eigen::MatrixXd& mat_like) {
    if (mat_like.rows() == 1) {
        return mat_like.row(0).transpose();
    }
    if (mat_like.cols() == 1) {
        return mat_like.col(0);
    }
    throw std::runtime_error("matrix_like_to_vector: expected a row/column matrix");
}

inline Eigen::VectorXd ensure_vector(const Eigen::VectorXd& vec_like) {
    return vec_like;
}

inline Eigen::VectorXd ensure_vector(const Eigen::MatrixXd& mat_like) {
    return matrix_like_to_vector(mat_like);
}

inline double dot_matrix_like_vector(const Eigen::MatrixXd& mat_like,
                                     const Eigen::VectorXd& vec) {
    if (is_debug) {std::cout<< "-----------------dot_matrix_like_vector---------------------"<<std::endl;}
    if (mat_like.rows() == 1 && mat_like.cols() == vec.size()) {
        return mat_like.row(0).dot(vec);
    }
    if (mat_like.cols() == 1 && mat_like.rows() == vec.size()) {
        return mat_like.col(0).dot(vec);
    }
    throw std::runtime_error("dot_matrix_like_vector: expected a row/column matrix compatible with the vector");
}

inline double dot_vector_matrix_like(const Eigen::VectorXd& vec,
                                     const Eigen::MatrixXd& mat_like) {
    return dot_matrix_like_vector(mat_like, vec);
}
inline Eigen::MatrixXd const_vector_dot_basis_1d(const Eigen::MatrixXd& const_mat,
                                                 const Eigen::MatrixXd& basis) {

    // Constant matrix  (k,k) dotted with basis (k,n) -> (k,n).
    
    if (is_debug) {std::cout<< "-----------------const_vector_dot_basis_1d---------------------"<<std::endl;}
    if (basis.rows() != const_mat.rows()) {
        throw std::runtime_error("const_vector_dot_basis_1d: incompatible shapes "
                                 + std::to_string(basis.cols()) + " vs " + std::to_string(const_mat.rows())
                                + ". The shapes were given as basis (" + std::to_string(basis.rows()) + ", " 
                                + std::to_string(basis.cols()) + ") and const_mat (" + std::to_string(const_mat.rows()) 
                                + ", " + std::to_string(const_mat.cols()) + ")." );
    }
    return basis.transpose() * const_mat;
}

// Trial vector basis (k,n) dotted with grad(Function).T (d,k) -> (d,n)
inline Eigen::MatrixXd dot_trial_vec_grad_func(const Eigen::MatrixXd& trial_vec,
                                               const Eigen::MatrixXd& grad_func) {
    if (is_debug) {std::cout<< "-----------------dot_trial_vec_grad_func---------------------"<<std::endl;}
    if (grad_func.rows() == 1 && grad_func.cols() == trial_vec.rows()) {
        return grad_func * trial_vec;
    }
    if (grad_func.cols() != trial_vec.rows()) {
        throw std::runtime_error("dot_trial_vec_grad_func: incompatible shapes");
    }
    return grad_func.transpose() * trial_vec;
}

// Mixed gradient tensor helpers (mirror numba_helpers mixed ops)
inline std::vector<Eigen::MatrixXd> dot_grad_grad_mixed(const std::vector<Eigen::MatrixXd>& a,
                                                        const std::vector<Eigen::MatrixXd>& b,
                                                        int flag) {
    if (is_debug) {std::cout<< "-----------------dot_grad_grad_mixed---------------------"<<std::endl;}
    int k = static_cast<int>(a.size());
    if (k == 0 || b.empty()) return {};
    int d = static_cast<int>(a[0].cols());   // spatial dimension (assumed equal to k)
    int n_a = static_cast<int>(a[0].rows()); // test/trial basis size for 'a'
    int n_b = static_cast<int>(b[0].rows()); // test/trial basis size for 'b'
    if (static_cast<int>(b.size()) != d) {
        throw std::runtime_error("dot_grad_grad_mixed expects square gradient tensor (k == d).");
    }

    // Flatten a: (k, n_a, d) -> (k*n_a, d)
    Eigen::MatrixXd A(k * n_a, d);
    for (int i = 0; i < k; ++i) {
        A.block(i * n_a, 0, n_a, d) = a[i];
    }

    // Flatten b treating the first axis as the contracted dimension (d == k):
    // b shape (k, n_b, d) -> B shape (d, n_b * d) with column ordering (basis, spatial)
    Eigen::MatrixXd B(d, n_b * d);
    for (int r = 0; r < d; ++r) {          // contracted dimension (component/spatial)
        for (int s = 0; s < n_b; ++s) {    // basis index
            for (int j = 0; j < d; ++j) {  // spatial axis (last of b)
                B(r, s * d + j) = b[r](s, j);
            }
        }
    }

    Eigen::MatrixXd prod = A * B; // (k*n_a) x (n_b*d)

    std::vector<Eigen::MatrixXd> out(static_cast<size_t>(k * d), Eigen::MatrixXd(n_a, n_b));
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < d; ++j) {
            Eigen::MatrixXd block(n_a, n_b);
            for (int p = 0; p < n_a; ++p) {
                for (int s = 0; s < n_b; ++s) {
                    block(p, s) = prod(i * n_a + p, s * d + j);
                }
            }
            if (flag == 1) {
                out[static_cast<size_t>(i * d + j)] = block.transpose(); // (n_b x n_a)
            } else {
                out[static_cast<size_t>(i * d + j)] = block;             // (n_a x n_b)
            }
        }
    }
    return out;
}

inline std::vector<Eigen::MatrixXd> transpose_mixed_grad_tensor(const std::vector<Eigen::MatrixXd>& tensor,
                                                                int k, int d) {
    if (is_debug) {std::cout<< "-----------------transpose_mixed_grad_tensor---------------------"<<std::endl;}
    std::vector<Eigen::MatrixXd> out(static_cast<size_t>(k * d));
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < d; ++j) {
            int dst = i * d + j;
            int src = j * d + i;
            if (src < static_cast<int>(tensor.size())) {
                out[static_cast<size_t>(dst)] = tensor[static_cast<size_t>(src)];
            }
        }
    }
    return out;
}

inline std::vector<Eigen::MatrixXd> trace_mixed_tensor(const std::vector<Eigen::MatrixXd>& tensor,
                                                       int k, int d) {
    if (is_debug) {std::cout<< "-----------------trace_mixed_tensor---------------------"<<std::endl;}
    if (tensor.empty()) return {};
    int n_rows = static_cast<int>(tensor[0].rows());
    int n_cols = static_cast<int>(tensor[0].cols());
    Eigen::MatrixXd acc = Eigen::MatrixXd::Zero(n_rows, n_cols);
    int diag = std::min(k, d);
    for (int i = 0; i < diag; ++i) {
        int idx = i * d + i;
        if (idx < static_cast<int>(tensor.size())) acc += tensor[static_cast<size_t>(idx)];
    }
    return {acc};
}

inline std::vector<Eigen::MatrixXd> scale_mixed_basis_with_coeffs(const std::vector<Eigen::MatrixXd>& mixed_basis,
                                                                  const Eigen::MatrixXd& coeffs) {
    if (is_debug) {std::cout<< "-----------------scale_mixed_basis_with_coeffs---------------------"<<std::endl;}
    if (mixed_basis.empty()) return {};
    Eigen::MatrixXd base = Eigen::MatrixXd::Zero(mixed_basis[0].rows(), mixed_basis[0].cols());
    for (const auto& m : mixed_basis) base += m;
    int k_out = static_cast<int>(coeffs.rows());
    int d_cols = static_cast<int>(coeffs.cols());
    std::vector<Eigen::MatrixXd> out(static_cast<size_t>(k_out * d_cols));
    for (int i = 0; i < k_out; ++i) {
        for (int j = 0; j < d_cols; ++j) {
            out[static_cast<size_t>(i * d_cols + j)] = base * coeffs(i, j);
        }
    }
    return out;
}

inline Eigen::MatrixXd inner_mixed_grad_const(const std::vector<Eigen::MatrixXd>& mixed_grad,
                                              const Eigen::MatrixXd& grad_const,
                                              int k, int d,
                                              int n_test, int n_trial) {
    if (is_debug) {std::cout<< "-----------------inner_mixed_grad_const---------------------"<<std::endl;}
    if (mixed_grad.empty()) return Eigen::MatrixXd();
    const int total_entries = static_cast<int>(mixed_grad.size());
    Eigen::Index mat_rows = mixed_grad[0].rows();
    Eigen::Index mat_cols = mixed_grad[0].cols();

    // Determine tensor layout
    bool flattened = (total_entries == k);       // shape (k, n_test*n_trial, d)
    bool split_components = (total_entries == k * d); // shape (k*d, n_test, n_trial)

    if (!flattened && !split_components) {
        throw std::runtime_error("inner_mixed_grad_const: unsupported mixed_grad layout");
    }

    // Infer n_test/n_trial when not provided
    if (flattened) {
        if (n_test <= 0 || n_trial <= 0) {
            int guess = static_cast<int>(std::round(std::sqrt(static_cast<double>(mat_rows))));
            if (guess <= 0) guess = static_cast<int>(mat_rows);
            n_test = guess;
            n_trial = (guess != 0) ? static_cast<int>(mat_rows) / guess : static_cast<int>(mat_rows);
            if (n_test * n_trial != mat_rows) {
                n_test = static_cast<int>(mat_rows);
                n_trial = 1;
            }
        }
    } else { // split_components
        if (n_test <= 0) n_test = static_cast<int>(mat_rows);
        if (n_trial <= 0) n_trial = static_cast<int>(mat_cols);
    }

    // Build flat matrix A (n_test*n_trial, k*d)
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n_test * n_trial), k * d);
    if (split_components) {
        for (int comp = 0; comp < k; ++comp) {
            for (int j = 0; j < d; ++j) {
                int idx = comp * d + j;
                if (idx >= total_entries) continue;
                const auto& M = mixed_grad[static_cast<size_t>(idx)]; // (n_test, n_trial)
                for (int t = 0; t < n_test; ++t) {
                    for (int s = 0; s < n_trial; ++s) {
                        int row = t * n_trial + s;
                        if (t < M.rows() && s < M.cols()) {
                            A(row, comp * d + j) = M(t, s);
                        }
                    }
                }
            }
        }
    } else { // flattened
        for (int comp = 0; comp < k; ++comp) {
            if (comp >= total_entries) break;
            const auto& M = mixed_grad[static_cast<size_t>(comp)]; // (n_test*n_trial, d)
            for (int row = 0; row < std::min<int>(M.rows(), n_test * n_trial); ++row) {
                for (int j = 0; j < d && j < M.cols(); ++j) {
                    A(row, comp * d + j) = M(row, j);
                }
            }
        }
    }

    Eigen::VectorXd b = Eigen::VectorXd::Zero(k * d);
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < d; ++j) {
            if (i < grad_const.rows() && j < grad_const.cols()) {
                b(i * d + j) = grad_const(i, j);
            }
        }
    }
    Eigen::VectorXd flat = A * b; // length n_test * n_trial
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n_test, n_trial);
    for (int t = 0; t < n_test; ++t) {
        for (int s = 0; s < n_trial; ++s) {
            int idx = t * n_trial + s;
            if (idx < flat.size()) out(t, s) = flat(idx);
        }
    }
    return out;
}

inline Eigen::MatrixXd inner_grad_const_mixed(const Eigen::MatrixXd& grad_const,
                                              const std::vector<Eigen::MatrixXd>& mixed_grad,
                                              int k, int d,
                                              int n_test, int n_trial) {
    return inner_mixed_grad_const(mixed_grad, grad_const, k, d, n_test, n_trial);
}

// Backward compatibility wrapper (infers n_test/n_trial from row count if not provided)
inline Eigen::MatrixXd inner_mixed_grad_const(const std::vector<Eigen::MatrixXd>& mixed_grad,
                                              const Eigen::MatrixXd& grad_const,
                                              int k, int d) {
    if (mixed_grad.empty()) return Eigen::MatrixXd();
    const int total_entries = static_cast<int>(mixed_grad.size());
    if (total_entries == k * d) {
        // Split-component storage: each entry already carries one (n_test x n_trial)
        // block, so rows/cols are the real mixed basis extents.
        return inner_mixed_grad_const(
            mixed_grad,
            grad_const,
            k,
            d,
            static_cast<int>(mixed_grad[0].rows()),
            static_cast<int>(mixed_grad[0].cols()));
    }
    int n_rows = static_cast<int>(mixed_grad[0].rows());
    int n_trial = static_cast<int>(std::round(std::sqrt(static_cast<double>(n_rows))));
    int n_test = (n_trial > 0) ? n_rows / n_trial : n_rows;
    if (n_test * n_trial != n_rows) {
        n_test = n_rows;
        n_trial = 1;
    }
    return inner_mixed_grad_const(mixed_grad, grad_const, k, d, n_test, n_trial);
}

inline Eigen::MatrixXd inner_grad_const_mixed(const Eigen::MatrixXd& grad_const,
                                              const std::vector<Eigen::MatrixXd>& mixed_grad,
                                              int k, int d) {
    return inner_mixed_grad_const(mixed_grad, grad_const, k, d);
}

inline std::vector<Eigen::MatrixXd> add_mixed(const std::vector<Eigen::MatrixXd>& a,
                                              const std::vector<Eigen::MatrixXd>& b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    if (a.size() != b.size()) {
        size_t big = std::max(a.size(), b.size());
        size_t small = std::min(a.size(), b.size());
        if (small == 0 || big % small != 0) {
            throw std::runtime_error("add_mixed: size mismatch a=" + std::to_string(a.size()) +
                                     " b=" + std::to_string(b.size()));
        }
        std::vector<Eigen::MatrixXd> out(big);
        for (size_t i = 0; i < big; ++i) {
            const auto& lhs = (a.size() == big) ? a[i] : a[i % a.size()];
            const auto& rhs = (b.size() == big) ? b[i] : b[i % b.size()];
            if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
                throw std::runtime_error("add_mixed: shape mismatch at " + std::to_string(i) + " (" +
                                         std::to_string(lhs.rows()) + "x" + std::to_string(lhs.cols()) +
                                         ") vs (" + std::to_string(rhs.rows()) + "x" + std::to_string(rhs.cols()) + ")");
            }
            out[i] = lhs + rhs;
        }
        return out;
    }
    std::vector<Eigen::MatrixXd> out = a;
    for (size_t i = 0; i < out.size(); ++i) out[i] += b[i];
    return out;
}

inline std::vector<Eigen::MatrixXd> sub_mixed(const std::vector<Eigen::MatrixXd>& a,
                                              const std::vector<Eigen::MatrixXd>& b) {
    if (a.empty()) {
        std::vector<Eigen::MatrixXd> out = b;
        for (auto& m : out) m = -m;
        return out;
    }
    if (b.empty()) return a;
    if (a.size() != b.size()) {
        size_t big = std::max(a.size(), b.size());
        size_t small = std::min(a.size(), b.size());
        if (small == 0 || big % small != 0) {
            throw std::runtime_error("sub_mixed: size mismatch a=" + std::to_string(a.size()) +
                                     " b=" + std::to_string(b.size()));
        }
        std::vector<Eigen::MatrixXd> out(big);
        for (size_t i = 0; i < big; ++i) {
            const auto& lhs = (a.size() == big) ? a[i] : a[i % a.size()];
            const auto& rhs = (b.size() == big) ? b[i] : b[i % b.size()];
            if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
                throw std::runtime_error("sub_mixed: shape mismatch (" +
                                         std::to_string(lhs.rows()) + "x" + std::to_string(lhs.cols()) +
                                         ") vs (" + std::to_string(rhs.rows()) + "x" + std::to_string(rhs.cols()) + ")");
            }
            out[i] = lhs - rhs;
        }
        return out;
    }
    std::vector<Eigen::MatrixXd> out = a;
    for (size_t i = 0; i < out.size(); ++i) out[i] -= b[i];
    return out;
}

inline std::vector<Eigen::MatrixXd> scale_mixed(const std::vector<Eigen::MatrixXd>& a, double s) {
    std::vector<Eigen::MatrixXd> out = a;
    for (auto& m : out) m *= s;
    return out;
}

// Generic component-stack helpers. These intentionally work on the raw
// std::vector<MatrixXd> runtime carrier used for gradients and plain
// basis-carrying rank-2+ tensors alike.
inline std::vector<Eigen::MatrixXd> add_stack(const std::vector<Eigen::MatrixXd>& a,
                                              const std::vector<Eigen::MatrixXd>& b) {
    return add_mixed(a, b);
}

inline std::vector<Eigen::MatrixXd> sub_stack(const std::vector<Eigen::MatrixXd>& a,
                                              const std::vector<Eigen::MatrixXd>& b) {
    return sub_mixed(a, b);
}

inline std::vector<Eigen::MatrixXd> scale_stack(const std::vector<Eigen::MatrixXd>& a, double s) {
    return scale_mixed(a, s);
}

inline std::vector<Eigen::MatrixXd> add_stack_mat(const std::vector<Eigen::MatrixXd>& a,
                                                  const Eigen::MatrixXd& b) {
    std::vector<Eigen::MatrixXd> out = a;
    for (auto& m : out) {
        if (m.rows() != b.rows() || m.cols() != b.cols()) {
            throw std::runtime_error("add_stack_mat: shape mismatch");
        }
        m += b;
    }
    return out;
}

inline std::vector<Eigen::MatrixXd> add_mat_stack(const Eigen::MatrixXd& a,
                                                  const std::vector<Eigen::MatrixXd>& b) {
    return add_stack_mat(b, a);
}

inline std::vector<Eigen::MatrixXd> sub_stack_mat(const std::vector<Eigen::MatrixXd>& a,
                                                  const Eigen::MatrixXd& b) {
    std::vector<Eigen::MatrixXd> out = a;
    for (auto& m : out) {
        if (m.rows() != b.rows() || m.cols() != b.cols()) {
            throw std::runtime_error("sub_stack_mat: shape mismatch");
        }
        m -= b;
    }
    return out;
}

inline std::vector<Eigen::MatrixXd> sub_mat_stack(const Eigen::MatrixXd& a,
                                                  const std::vector<Eigen::MatrixXd>& b) {
    std::vector<Eigen::MatrixXd> out = b;
    for (auto& m : out) {
        if (m.rows() != a.rows() || m.cols() != a.cols()) {
            throw std::runtime_error("sub_mat_stack: shape mismatch");
        }
        m = a - m;
    }
    return out;
}

inline std::vector<Eigen::MatrixXd> add_stack_scalar(const std::vector<Eigen::MatrixXd>& a, double s) {
    std::vector<Eigen::MatrixXd> out = a;
    for (auto& m : out) {
        m = (m.array() + s).matrix();
    }
    return out;
}

inline std::vector<Eigen::MatrixXd> sub_stack_scalar(const std::vector<Eigen::MatrixXd>& a, double s) {
    std::vector<Eigen::MatrixXd> out = a;
    for (auto& m : out) {
        m = (m.array() - s).matrix();
    }
    return out;
}

inline std::vector<Eigen::MatrixXd> sub_scalar_stack(double s,
                                                     const std::vector<Eigen::MatrixXd>& a) {
    std::vector<Eigen::MatrixXd> out = a;
    for (auto& m : out) {
        m = (-m.array() + s).matrix();
    }
    return out;
}

inline std::vector<Eigen::MatrixXd> dot_mixed_mat(const std::vector<Eigen::MatrixXd>& mixed,
                                                  const Eigen::MatrixXd& mat,
                                                  int k, int d) {
    if (is_debug) {std::cout<< "-----------------dot_mixed_mat---------------------"<<std::endl;}
    if (mixed.empty()) return {};
    int d_out = static_cast<int>(mat.cols());
    Eigen::Index n_rows = mixed[0].rows();
    Eigen::Index n_cols = mixed[0].cols();
    std::vector<Eigen::MatrixXd> out(static_cast<size_t>(k * d_out), Eigen::MatrixXd::Zero(n_rows, n_cols));
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < d_out; ++j) {
            Eigen::MatrixXd acc = Eigen::MatrixXd::Zero(n_rows, n_cols);
            for (int r = 0; r < d; ++r) {
                int idx = i * d + r;
                if (idx < static_cast<int>(mixed.size()) && r < mat.rows()) {
                    acc += mixed[static_cast<size_t>(idx)] * mat(r, j);
                }
            }
            out[static_cast<size_t>(i * d_out + j)] = acc;
        }
    }
    return out;
}

inline std::vector<Eigen::MatrixXd> dot_mat_mixed(const Eigen::MatrixXd& mat,
                                                  const std::vector<Eigen::MatrixXd>& mixed,
                                                  int k, int d) {
    if (is_debug) {std::cout<< "-----------------dot_mat_mixed---------------------"<<std::endl;}
    if (mixed.empty()) return {};
    int k_out = static_cast<int>(mat.rows());
    Eigen::Index n_rows = mixed[0].rows();
    Eigen::Index n_cols = mixed[0].cols();
    std::vector<Eigen::MatrixXd> out(static_cast<size_t>(k_out * d), Eigen::MatrixXd::Zero(n_rows, n_cols));
    for (int i = 0; i < k_out; ++i) {
        for (int j = 0; j < d; ++j) {
            Eigen::MatrixXd acc = Eigen::MatrixXd::Zero(n_rows, n_cols);
            for (int r = 0; r < k; ++r) {
                int idx = r * d + j;
                if (idx < static_cast<int>(mixed.size()) && r < mat.cols()) {
                    acc += mat(i, r) * mixed[static_cast<size_t>(idx)];
                }
            }
            out[static_cast<size_t>(i * d + j)] = acc;
        }
    }
    return out;
}

// Contract mixed tensor (k, rows, cols, d) with spatial vector (d,)
inline Eigen::MatrixXd dot_mixed_with_vec(const std::vector<Eigen::MatrixXd>& mixed,
                                          const Eigen::VectorXd& vec,
                                          int k, int d) {
    if (is_debug) {std::cout<< "-----------------dot_mixed_with_vec---------------------"<<std::endl;}
    if (mixed.empty()) return Eigen::MatrixXd();
    if (vec.size() != d) {
        throw std::runtime_error("dot_mixed_with_vec: vector length must match spatial dimension");
    }
    Eigen::Index n_rows = mixed[0].rows();
    Eigen::Index n_cols = mixed[0].cols();
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n_rows, n_cols);
    for (int i = 0; i < k; ++i) {
        for (int r = 0; r < d; ++r) {
            int idx = i * d + r;
            if (idx < static_cast<int>(mixed.size())) {
                out.noalias() += mixed[static_cast<size_t>(idx)] * vec(r);
            }
        }
    }
    return out;
}

// Transpose gradient stack: swap component and spatial axes (k,n,d) -> (d,n,k)
inline std::vector<Eigen::MatrixXd> transpose_grad_stack(const std::vector<Eigen::MatrixXd>& tensor) {
    int k = static_cast<int>(tensor.size());
    if (k == 0) return {};
    int n = static_cast<int>(tensor[0].rows());
    int d = static_cast<int>(tensor[0].cols());
    if (d != k) {
        throw std::runtime_error("transpose_grad_stack expects square gradient (k == d).");
    }
    // Keep the outer vector sized by component axis (k) to mirror numba_helpers.
    std::vector<Eigen::MatrixXd> out(static_cast<size_t>(k), Eigen::MatrixXd(n, d));
    for (int c = 0; c < k; ++c) {
        if (tensor[c].rows() != n || tensor[c].cols() != d) {
            throw std::runtime_error("transpose_grad_stack: inconsistent component shape");
        }
    }
    // out[i](j, r) = tensor[r](j, i)
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < k; ++i) {
            for (int r = 0; r < d; ++r) {
                out[i](j, r) = tensor[r](j, i);
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Lightweight OpInfo-style helpers for higher-rank operations
// ---------------------------------------------------------------------------

struct GradStack {
    std::vector<Eigen::MatrixXd> comps; // each (n_union, d)

    GradStack() = default;
    explicit GradStack(std::vector<Eigen::MatrixXd> c) : comps(std::move(c)) {}

    ssize_t n_comps() const { return static_cast<ssize_t>(comps.size()); }
    bool empty() const { return comps.empty(); }
    ssize_t rows() const { return comps.empty() ? 0 : comps[0].rows(); }
    ssize_t cols() const { return comps.empty() ? 0 : comps[0].cols(); }
};

inline GradStack add_grad(const GradStack& a, const GradStack& b) {
    GradStack out;
    if (a.n_comps() != b.n_comps()) throw std::runtime_error("add_grad: component mismatch");
    out.comps.resize(a.n_comps());
    for (size_t i = 0; i < a.comps.size(); ++i) {
        if (a.comps[i].rows() != b.comps[i].rows() || a.comps[i].cols() != b.comps[i].cols())
            throw std::runtime_error("add_grad: shape mismatch");
        out.comps[i] = a.comps[i] + b.comps[i];
    }
    return out;
}

inline GradStack add_grad_mat(const GradStack& g, const Eigen::MatrixXd& m) {
    GradStack out;
    out.comps.resize(g.n_comps());
    for (size_t i = 0; i < g.comps.size(); ++i) {
        const auto& Gi = g.comps[i];
        // Exact shape match
        if (Gi.rows() == m.rows() && Gi.cols() == m.cols()) {
            out.comps[i] = Gi + m;
            continue;
        }
        // Broadcast a (k x d) constant across all rows, matching gradient components.
        if (m.rows() == static_cast<int>(g.n_comps()) && m.cols() == Gi.cols()) {
            // Take the i-th row and add to every row of component i.
            out.comps[i] = Gi.rowwise() + m.row(static_cast<Eigen::Index>(i));
            continue;
        }
        // Component-first gradient storage (d x n) appears in a few lowered
        // paths for rank-1 basis gradients. Accept it by transposing into the
        // row-wise GradStack convention (n x d) at this bridge point.
        if (m.rows() == Gi.cols() && m.cols() == Gi.rows()) {
            out.comps[i] = Gi + m.transpose();
            continue;
        }
        // Simple broadcast cases: column vector or row vector
        if (Gi.rows() == m.rows() && m.cols() == 1) {
            out.comps[i] = Gi.colwise() + m.col(0);
            continue;
        }
        if (Gi.cols() == m.cols() && m.rows() == 1) {
            // Add row vector to each row
            out.comps[i] = Gi.rowwise() + m.row(0);
            continue;
        }
        if (m.rows() == 1 && m.cols() == Gi.rows()) {
            // Row vector matching gradient rows; transpose then broadcast across cols
            Eigen::VectorXd col = m.transpose();
            out.comps[i] = Gi.colwise() + col;
            continue;
        }
        throw std::runtime_error("add_grad_mat: shape mismatch");
    }
    return out;
}

inline GradStack add_grad_mat(const GradStack& g, const std::vector<Eigen::MatrixXd>& mats) {
    if (mats.size() == 1) {
        return add_grad_mat(g, mats[0]);
    }
    if (static_cast<ssize_t>(mats.size()) == g.n_comps()) {
        return add_grad(g, GradStack{mats});
    }
    throw std::runtime_error("add_grad_mat: component mismatch");
}

inline GradStack sub_grad_mat(const GradStack& g, const Eigen::MatrixXd& m) {
    GradStack out;
    out.comps.resize(g.n_comps());
    for (size_t i = 0; i < g.comps.size(); ++i) {
        const auto& Gi = g.comps[i];
        // Exact shape match
        if (Gi.rows() == m.rows() && Gi.cols() == m.cols()) {
            out.comps[i] = Gi - m;
            continue;
        }
        // Broadcast (k x d) constant across rows, matching gradient components.
        if (m.rows() == static_cast<int>(g.n_comps()) && m.cols() == Gi.cols()) {
            out.comps[i] = Gi.rowwise() - m.row(static_cast<Eigen::Index>(i));
            continue;
        }
        if (m.rows() == Gi.cols() && m.cols() == Gi.rows()) {
            out.comps[i] = Gi - m.transpose();
            continue;
        }
        if (Gi.rows() == m.rows() && m.cols() == 1) {
            out.comps[i] = Gi.colwise() - m.col(0);
            continue;
        }
        if (Gi.cols() == m.cols() && m.rows() == 1) {
            out.comps[i] = Gi.rowwise() - m.row(0);
            continue;
        }
        if (m.rows() == 1 && m.cols() == Gi.rows()) {
            Eigen::VectorXd col = m.transpose();
            out.comps[i] = Gi.colwise() - col;
            continue;
        }
        throw std::runtime_error("sub_grad_mat: shape mismatch");
    }
    return out;
}

inline GradStack sub_grad_mat(const GradStack& g, const std::vector<Eigen::MatrixXd>& mats) {
    if (mats.size() == 1) {
        return sub_grad_mat(g, mats[0]);
    }
    if (static_cast<ssize_t>(mats.size()) == g.n_comps()) {
        GradStack out;
        out.comps.resize(g.n_comps());
        for (size_t i = 0; i < g.comps.size(); ++i) {
            if (g.comps[i].rows() != mats[i].rows() || g.comps[i].cols() != mats[i].cols()) {
                throw std::runtime_error("sub_grad_mat: shape mismatch");
            }
            out.comps[i] = g.comps[i] - mats[i];
        }
        return out;
    }
    throw std::runtime_error("sub_grad_mat: component mismatch");
}

inline GradStack mat_sub_grad(const Eigen::MatrixXd& m, const GradStack& g) {
    GradStack out;
    out.comps.resize(g.n_comps());
    for (size_t i = 0; i < g.comps.size(); ++i) {
        const auto& Gi = g.comps[i];
        if (Gi.rows() == m.rows() && Gi.cols() == m.cols()) {
            out.comps[i] = m - Gi;
            continue;
        }
        if (m.rows() == Gi.cols() && m.cols() == Gi.rows()) {
            out.comps[i] = m.transpose() - Gi;
            continue;
        }
        throw std::runtime_error("mat_sub_grad: shape mismatch");
    }
    return out;
}

inline GradStack mat_sub_grad(const std::vector<Eigen::MatrixXd>& mats, const GradStack& g) {
    if (mats.size() == 1) {
        return mat_sub_grad(mats[0], g);
    }
    if (static_cast<ssize_t>(mats.size()) == g.n_comps()) {
        GradStack out;
        out.comps.resize(g.n_comps());
        for (size_t i = 0; i < g.comps.size(); ++i) {
            if (mats[i].rows() != g.comps[i].rows() || mats[i].cols() != g.comps[i].cols()) {
                throw std::runtime_error("mat_sub_grad: shape mismatch");
            }
            out.comps[i] = mats[i] - g.comps[i];
        }
        return out;
    }
    throw std::runtime_error("mat_sub_grad: component mismatch");
}

inline GradStack scale_grad(const GradStack& g, double s) {
    GradStack out;
    out.comps.resize(g.n_comps());
    for (size_t i = 0; i < g.comps.size(); ++i) out.comps[i] = g.comps[i] * s;
    return out;
}

inline GradStack div_grad(const GradStack& g, double s) {
    GradStack out;
    out.comps.resize(g.n_comps());
    for (size_t i = 0; i < g.comps.size(); ++i) out.comps[i] = g.comps[i] / s;
    return out;
}

inline Eigen::MatrixXd trace_grad(const GradStack& g) {
    if (g.empty()) return Eigen::MatrixXd();
    Eigen::MatrixXd out(1, g.rows());
    out.setZero();
    for (size_t c = 0; c < g.comps.size(); ++c) {
        // add diagonal component if available
        if (g.comps[c].cols() > static_cast<Eigen::Index>(c)) {
            out.row(0) += g.comps[c].col(static_cast<Eigen::Index>(c)).transpose();
        }
    }
    return out;
}

inline GradStack cwise_grad_mat(const GradStack& g, const Eigen::MatrixXd& m) {
    GradStack out;
    out.comps.resize(g.n_comps());
    for (size_t i = 0; i < g.comps.size(); ++i) {
        const auto& Gi = g.comps[i];
        // Exact match
        if (Gi.rows() == m.rows() && Gi.cols() == m.cols()) {
            out.comps[i] = Gi.cwiseProduct(m);
            continue;
        }
        // Column vector broadcast across columns
        if (Gi.rows() == m.rows() && m.cols() == 1) {
            out.comps[i] = Gi.array().colwise() * m.col(0).array();
            continue;
        }
        // Row vector whose length matches gradient rows -> treat as column vector
        if (m.rows() == 1 && m.cols() == Gi.rows()) {
            Eigen::VectorXd col = m.transpose();
            out.comps[i] = Gi.array().colwise() * col.array();
            continue;
        }
        // Row vector broadcast across rows
        if (Gi.cols() == m.cols() && m.rows() == 1) {
            out.comps[i] = Gi.array().rowwise() * m.row(0).array();
            continue;
        }
        // Component-wise scaling with (k x d) matrix: take row i and broadcast over rows.
        if (m.rows() == static_cast<int>(g.n_comps()) && m.cols() == Gi.cols()) {
            out.comps[i] = Gi.array().rowwise() * m.row(static_cast<Eigen::Index>(i)).array();
            continue;
        }
        // Component-wise scalar scaling with (k x 1)
        if (m.rows() == static_cast<int>(g.n_comps()) && m.cols() == 1) {
            out.comps[i] = Gi * m(static_cast<Eigen::Index>(i), 0);
            continue;
        }
        throw std::runtime_error("cwise_grad_mat: shape mismatch");
    }
    return out;
}

inline GradStack cwise_mat_grad(const Eigen::MatrixXd& m, const GradStack& g) {
    return cwise_grad_mat(g, m);
}

inline GradStack cwise_grad_grad(const GradStack& a, const GradStack& b) {
    GradStack out;
    if (a.n_comps() != b.n_comps()) throw std::runtime_error("cwise_grad_grad: component mismatch");
    out.comps.resize(a.n_comps());
    for (size_t i = 0; i < a.comps.size(); ++i) {
        if (a.comps[i].rows() != b.comps[i].rows() || a.comps[i].cols() != b.comps[i].cols())
            throw std::runtime_error("cwise_grad_grad: shape mismatch");
        out.comps[i] = a.comps[i].cwiseProduct(b.comps[i]);
    }
    return out;
}

/**
 * Helper to flatten any Eigen matrix/vector to a 1D vector.
 * Replicates Python: _flatten_to_1d
 */
inline Eigen::VectorXd flatten_to_1d(const Eigen::MatrixXd& input) {
    // reshaped() creates a view, constructing VectorXd copies it to contiguous memory
    return Eigen::VectorXd(input.reshaped());
}

/**
 * Trace data (flattened) times identity matrix (k,d) -> (k, n_rows, d).
 * * * Logic:
 * For each component 'i' in k:
 * OutputMatrix_i = OuterProduct(trace_vals, identity_row_i)
 * * * Dimensions:
 * trace_vals: (n)  (after flattening)
 * identity:   (k, d)
 * Output:     std::vector of size k, containing (n, d) matrices.
 */
inline std::vector<Eigen::MatrixXd> trace_times_identity(const Eigen::MatrixXd& trace_vals_in,
                                                         const Eigen::MatrixXd& identity) {
    if (is_debug) {std::cout << "-----------------trace_times_identity---------------------" << std::endl;}
    // 1. Flatten trace_vals (corresponds to _flatten_to_1d)
    //    We take MatrixXd as input to support both vectors and matrices genericly.
    Eigen::VectorXd flat = flatten_to_1d(trace_vals_in);

    // 2. Get dimensions from identity matrix
    long k_comps = identity.rows();

    // 3. Initialize Output Vector (representing the 'k' axis)
    std::vector<Eigen::MatrixXd> res(k_comps);

    // 4. Compute Outer Products
    // Python Loop: res[i, r, j] = flat[r] * identity[i, j]
    // Vectorized:  res[i]       = flat    * identity.row(i)
    // Dimensions:  (n, d)       = (n, 1)  * (1, d)
    for (int i = 0; i < k_comps; ++i) {
        res[i] = flat * identity.row(i);
    }

    return res;
}

inline std::vector<Eigen::MatrixXd> identity_times_trace_matrix(const Eigen::MatrixXd& identity,
                                                                const Eigen::MatrixXd& trace_matrix) {
    if (is_debug) {std::cout << "-----------------identity_times_trace_matrix---------------------" << std::endl;}
    const Eigen::Index k_comps = identity.rows();
    const Eigen::Index d_dim = identity.cols();
    std::vector<Eigen::MatrixXd> res(static_cast<size_t>(k_comps * d_dim));
    for (Eigen::Index i = 0; i < k_comps; ++i) {
        for (Eigen::Index j = 0; j < d_dim; ++j) {
            res[static_cast<size_t>(i * d_dim + j)] = trace_matrix * identity(i, j);
        }
    }
    return res;
}

inline std::vector<Eigen::MatrixXd> identity_times_trace_matrix(const Eigen::MatrixXd& identity,
                                                                const std::vector<Eigen::MatrixXd>& trace_stack) {
    if (is_debug) {std::cout << "-----------------identity_times_trace_matrix(stack)---------------------" << std::endl;}
    if (trace_stack.empty()) return {};
    Eigen::MatrixXd acc = Eigen::MatrixXd::Zero(trace_stack[0].rows(), trace_stack[0].cols());
    for (const auto& block : trace_stack) {
        if (block.rows() != acc.rows() || block.cols() != acc.cols()) {
            throw std::runtime_error("identity_times_trace_matrix(stack): inconsistent trace block shape.");
        }
        acc += block;
    }
    return identity_times_trace_matrix(identity, acc);
}

// Generic tensor contraction helper (einsum-style) using Eigen::Tensor
template<int RankA, int RankB, int NumPairs>
inline auto tensor_contract(const Eigen::Tensor<double, RankA>& A,
                            const Eigen::Tensor<double, RankB>& B,
                            const Eigen::array<Eigen::IndexPair<int>, NumPairs>& pairs) {
    return A.contract(B, pairs);
}

}  // namespace pycutfem::cpp_backend
