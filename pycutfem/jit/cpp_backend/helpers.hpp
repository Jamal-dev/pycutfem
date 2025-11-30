// Native C++ implementations of selected helper routines used by the
// generated kernels. We will grow this file incrementally to cover the
// full helper surface.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>

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

// ---------------------------------------------------------------------------
// Tensor Algebra (Dot Products / Contractions)
// ---------------------------------------------------------------------------

// Mirrors contract_last_first
// Generic contraction: A[..., i] * B[i, ...]
inline py::array_t<double> contract_last_first(const py::array_t<double>& A,
                                               const py::array_t<double>& B) {
    auto A_req = A.request();
    auto B_req = B.request();
    
    ssize_t K = A_req.shape[A_req.ndim - 1]; // Contraction dim
    if (B_req.shape[0] != K) throw std::runtime_error("Dimension mismatch in contract_last_first");

    // Flatten A to (Rows, K) and B to (K, Cols)
    ssize_t rows = A_req.size / K;
    ssize_t cols = B_req.size / K;

    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        MatA((double*)A_req.ptr, rows, K);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        MatB((double*)B_req.ptr, K, cols);

    Eigen::MatrixXd Res = MatA * MatB; // (Rows, Cols)

    // Reshape output to A.shape[:-1] + B.shape[1:]
    std::vector<ssize_t> out_shape;
    for(int i=0; i<A_req.ndim-1; ++i) out_shape.push_back(A_req.shape[i]);
    for(int i=1; i<B_req.ndim; ++i)   out_shape.push_back(B_req.shape[i]);

    // If B was 1D (vector), the trailing dimension is gone (numpy dot behavior)
    if (B_req.ndim == 1 && out_shape.empty()) {
        // Scalar result
        return py::cast(Res(0,0));
    }
    
    return py::array_t<double>(out_shape, Res.data());
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

    for (ssize_t i=0; i<k; ++i) {
        for (ssize_t j=0; j<n; ++j) {
            for (ssize_t l=0; l<d; ++l) {
                // Swap dim 0 (component) and dim 2 (spatial)
                // In generic transpose(grad u), (grad u)_ij = d u_i / d x_j
                // (grad u)^T _ij = d u_j / d x_i
                // Here tensor is G_ijk = d psi_j / d x_k (for component i) -- wait.
                // Standard layout (k, n, d) -> basis 'n' for component 'k', deriv 'd'.
                // If we transpose the gradient of the vector field:
                // We map input (k, n, d) -> output (d, n, k).
                // But numba_helper returns (k, n, d) and assumes k=d.
                out_v(l, j, i) = in_v(i, j, l);
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
    std::cout<< "-----------------dot_mass_test_trial---------------------"<<std::endl;
    return test_vec.transpose() * trial_vec;
}

inline Eigen::MatrixXd dot_mass_trial_test(const Eigen::MatrixXd& trial_vec,
                                           const Eigen::MatrixXd& test_vec) {
    std::cout<< "-----------------dot_mass_trial_test---------------------"<<std::endl;
    return trial_vec.transpose() * test_vec;
}

// grad(Function) @ Trial (grad_func: k x d, trial: k x n) -> d x n
inline Eigen::MatrixXd dot_grad_func_trial_vec(const Eigen::MatrixXd& grad_func,
                                               const Eigen::MatrixXd& trial_vec) {
    std::cout<< "-----------------dot_grad_func_trial_vec---------------------"<<std::endl;
    return grad_func * trial_vec;
}

// grad(basis) (vector<k x n, dim>) dotted with vector (dim,) -> k x n
inline Eigen::MatrixXd dot_grad_basis_vector(const std::vector<Eigen::MatrixXd>& grad_basis,
                                             const Eigen::VectorXd& vec) {
    std::cout<< "-----------------dot_grad_basis_vector---------------------"<<std::endl;
    int k = static_cast<int>(grad_basis.size());
    if (k == 0) return Eigen::MatrixXd();
    int n = static_cast<int>(grad_basis[0].rows());
    Eigen::MatrixXd out(k, n);
    for (int c = 0; c < k; ++c) {
        out.row(c) = grad_basis[c] * vec;
    }
    return out;
}

// Vector dotted with grad(basis) -> d x n
inline Eigen::MatrixXd dot_vec_grad(const Eigen::VectorXd& vec,
                                    const std::vector<Eigen::MatrixXd>& grad_basis) {
    std::cout<< "-----------------dot_vec_grad---------------------"<<std::endl;
    int k = static_cast<int>(grad_basis.size());
    if (k == 0) return Eigen::MatrixXd();
    int n = static_cast<int>(grad_basis[0].rows());
    Eigen::MatrixXd out(grad_basis[0].cols(), n);
    out.setZero();
    for (int c = 0; c < k; ++c) {
        out += grad_basis[c].transpose() * vec(c);
    }
    return out;
}

inline Eigen::MatrixXd vector_dot_grad_basis(const Eigen::VectorXd& vec,
                                             const std::vector<Eigen::MatrixXd>& grad_basis) {
    std::cout<< "-----------------vector_dot_grad_basis---------------------"<<std::endl;
    int k = static_cast<int>(grad_basis.size());
    if (k == 0) return Eigen::MatrixXd();
    int d = static_cast<int>(grad_basis[0].cols());
    if (k == 1 && vec.size() == d) {
        Eigen::RowVectorXd row = grad_basis[0] * vec;
        return Eigen::MatrixXd(row);
    }
    return dot_vec_grad(vec, grad_basis);
}

// Inner product of grad stacks -> n x n
inline Eigen::MatrixXd inner_grad_grad(const std::vector<Eigen::MatrixXd>& test,
                                       const std::vector<Eigen::MatrixXd>& trial) {
    std::cout<< "-----------------inner_grad_grad---------------------"<<std::endl;
    int k = static_cast<int>(test.size());
    if (k == 0) return Eigen::MatrixXd();
    int n = static_cast<int>(test[0].rows());
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n, n);
    for (int c = 0; c < k; ++c) {
        out += test[c] * trial[c].transpose();
    }
    return out;
}

// Inner of grad(test) (k,n,d) with grad(value) (k,d) -> (n,)
inline Eigen::VectorXd inner_grad_const(const std::vector<Eigen::MatrixXd>& grad_test,
                                        const Eigen::MatrixXd& grad_val) {
    std::cout<< "-----------------inner_grad_const---------------------"<<std::endl;
    int k = static_cast<int>(grad_test.size());
    if (k == 0) return Eigen::VectorXd();
    int n = static_cast<int>(grad_test[0].rows());
    Eigen::VectorXd out = Eigen::VectorXd::Zero(n);
    for (int c = 0; c < k; ++c) {
        out += grad_test[c] * grad_val.row(c).transpose();
    }
    return out;
}

// Inner of grad(value) (k,d) with grad(test) (k,n,d) -> (n,)
inline Eigen::VectorXd inner_const_grad(const Eigen::MatrixXd& grad_val,
                                        const std::vector<Eigen::MatrixXd>& grad_test) {
    return inner_grad_const(grad_test, grad_val);
}

// elementwise add/sub for Eigen matrices/vectors
template <typename T>
inline T binary_add(const T& a, const T& b) {
    return a + b;
}
template <typename T>
inline T binary_sub(const T& a, const T& b) {
    return a - b;
}

// Load variable component value: coeffs (nE,nU), basis (nE,nQ,nU)
inline double load_variable_component(const py::detail::unchecked_reference<double, 2>& coeffs,
                                      const py::detail::unchecked_reference<double, 3>& basis,
                                      ssize_t e, ssize_t q, int s0, int s1) {
    double val = 0.0;
    for (int j = s0; j < s1; ++j) {
        val += coeffs(e, j) * basis(e, q, j);
    }
    return val;
}

// gradient of a component: accumulates coefficients * phys grad
inline void gradient_component(Eigen::MatrixXd& out, int row,
                               const py::detail::unchecked_reference<double, 2>& coeffs,
                               const py::detail::unchecked_reference<double, 4>& gref,
                               const Eigen::Matrix<double, 2, 2>& Jloc,
                               ssize_t e, ssize_t q, int s0, int s1) {
    double gx = 0.0, gy = 0.0;
    for (int j = s0; j < s1; ++j) {
        double gr0 = gref(e, q, j, 0);
        double gr1 = gref(e, q, j, 1);
        double px = gr0 * Jloc(0, 0) + gr1 * Jloc(1, 0);
        double py = gr0 * Jloc(0, 1) + gr1 * Jloc(1, 1);
        double c = coeffs(e, j);
        gx += c * px;
        gy += c * py;
    }
    // Store as (ncomp, dim): rows are components, cols are spatial dimensions
    out(row, 0) = gx;
    out(row, 1) = gy;
}

// Grad stack dotted with trial basis
inline Eigen::MatrixXd dot_grad_trial(const std::vector<Eigen::MatrixXd>& grad_stack,
                                      const Eigen::MatrixXd& trial) {

    std::cout<< "-----------------dot_grad_trial---------------------"<<std::endl;
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

    std::cout << "-----------------dot_grad_basis (Contracting D-Axis)---------------------" << std::endl;

    // 1. Get Dimensions from the Basis
    int k = static_cast<int>(grad_basis.size()); // The 'i' dimension (List size)
    if (k == 0) return {};

    int n = static_cast<int>(grad_basis[0].rows()); // The 'n' dimension
    int d = static_cast<int>(grad_basis[0].cols()); // The 'd' dimension (Inner dimension)

    // 2. Validation
    // For Matrix Multiplication (n,d) * (rows, cols), 
    // the rows of grad_value must match 'd'.
    if (grad_value.rows() != d) {
        throw std::runtime_error("dot_grad_basis: Dimension mismatch. Basis cols (d) must match grad_value rows.");
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
            throw std::runtime_error("dot_grad_basis: Inconsistent basis component shape");
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

// grad(value) (k,k) dotted with grad(basis) (k,n,k) -> grad stack (k,n,k)
inline std::vector<Eigen::MatrixXd> dot_grad_value_with_grad_basis(
    const Eigen::MatrixXd& grad_value,
    const std::vector<Eigen::MatrixXd>& grad_basis) {
    std::cout<< "-----------------dot_grad_value_with_grad_basis---------------------"<<std::endl;
    int k = static_cast<int>(grad_basis.size());
    if (grad_value.rows() != k || grad_value.cols() != grad_value.rows()) {
        throw std::runtime_error("dot_grad_value_with_grad_basis: expect square grad_value with k==d");
    }
    if (k == 0) return {};
    int n = static_cast<int>(grad_basis[0].rows());
    int d = static_cast<int>(grad_basis[0].cols());
    if (d != k) {
        throw std::runtime_error("dot_grad_value_with_grad_basis: grad_basis spatial dim mismatch");
    }

    Eigen::MatrixXd flat(k, n * d);
    for (int c = 0; c < k; ++c) {
        if (grad_basis[c].rows() != n || grad_basis[c].cols() != d) {
            throw std::runtime_error("dot_grad_value_with_grad_basis: inconsistent basis component shape");
        }
        for (int j = 0; j < n; ++j) {
            for (int r = 0; r < d; ++r) {
                flat(c, j * d + r) = grad_basis[c](j, r);
            }
        }
    }

    Eigen::MatrixXd prod = grad_value * flat; // (k, n*d)
    std::vector<Eigen::MatrixXd> out(static_cast<size_t>(k), Eigen::MatrixXd(n, d));
    for (int c = 0; c < k; ++c) {
        for (int j = 0; j < n; ++j) {
            for (int r = 0; r < d; ++r) {
                out[c](j, r) = prod(c, j * d + r);
            }
        }
    }
    return out;
}

// value (k,) dotted with grad (k,d) -> (d,)
inline Eigen::VectorXd dot_value_with_grad(const Eigen::VectorXd& value_vec,
                                           const Eigen::MatrixXd& grad_mat) {
    if (grad_mat.rows() != value_vec.size()) {
        throw std::runtime_error("dot_value_with_grad: incompatible shapes");
    }
    return grad_mat.transpose() * value_vec;
}

// grad (k,d) dotted with value (k,) -> (d,)  (contracts components)
inline Eigen::VectorXd dot_grad_with_value(const Eigen::MatrixXd& grad_mat,
                                           const Eigen::VectorXd& value_vec) {
    if (grad_mat.rows() != value_vec.size()) {
        throw std::runtime_error("dot_grad_with_value: incompatible shapes");
    }
    return grad_mat.transpose() * value_vec;
}

// Trial vector basis (k,n) dotted with grad(Function).T (d,k) -> (d,n)
inline Eigen::MatrixXd dot_trial_vec_grad_func(const Eigen::MatrixXd& trial_vec,
                                               const Eigen::MatrixXd& grad_func) {
    if (grad_func.cols() != trial_vec.rows()) {
        throw std::runtime_error("dot_trial_vec_grad_func: incompatible shapes");
    }
    return grad_func.transpose() * trial_vec;
}

// Transpose gradient stack: swap component and spatial axes (k,n,d) -> (d,n,k)
inline std::vector<Eigen::MatrixXd> transpose_grad_stack(const std::vector<Eigen::MatrixXd>& tensor) {
    int k = static_cast<int>(tensor.size());
    if (k == 0) return {};
    int n = static_cast<int>(tensor[0].rows());
    int d = static_cast<int>(tensor[0].cols());
    std::vector<Eigen::MatrixXd> out(static_cast<size_t>(d), Eigen::MatrixXd(n, k));
    for (int c = 0; c < k; ++c) {
        if (tensor[c].rows() != n || tensor[c].cols() != d) {
            throw std::runtime_error("transpose_grad_stack: inconsistent component shape");
        }
        for (int j = 0; j < n; ++j) {
            for (int r = 0; r < d; ++r) {
                out[r](j, c) = tensor[c](j, r);
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
        if (g.comps[i].rows() != m.rows() || g.comps[i].cols() != m.cols())
            throw std::runtime_error("add_grad_mat: shape mismatch");
        out.comps[i] = g.comps[i] + m;
    }
    return out;
}

inline GradStack sub_grad_mat(const GradStack& g, const Eigen::MatrixXd& m) {
    GradStack out;
    out.comps.resize(g.n_comps());
    for (size_t i = 0; i < g.comps.size(); ++i) {
        if (g.comps[i].rows() != m.rows() || g.comps[i].cols() != m.cols())
            throw std::runtime_error("sub_grad_mat: shape mismatch");
        out.comps[i] = g.comps[i] - m;
    }
    return out;
}

inline GradStack mat_sub_grad(const Eigen::MatrixXd& m, const GradStack& g) {
    GradStack out;
    out.comps.resize(g.n_comps());
    for (size_t i = 0; i < g.comps.size(); ++i) {
        if (g.comps[i].rows() != m.rows() || g.comps[i].cols() != m.cols())
            throw std::runtime_error("mat_sub_grad: shape mismatch");
        out.comps[i] = m - g.comps[i];
    }
    return out;
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
        if (g.comps[i].rows() != m.rows() || g.comps[i].cols() != m.cols())
            throw std::runtime_error("cwise_grad_mat: shape mismatch");
        out.comps[i] = g.comps[i].cwiseProduct(m);
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

// Generic tensor contraction helper (einsum-style) using Eigen::Tensor
template<int RankA, int RankB, int NumPairs>
inline auto tensor_contract(const Eigen::Tensor<double, RankA>& A,
                            const Eigen::Tensor<double, RankB>& B,
                            const Eigen::array<Eigen::IndexPair<int>, NumPairs>& pairs) {
    return A.contract(B, pairs);
}

}  // namespace pycutfem::cpp_backend
