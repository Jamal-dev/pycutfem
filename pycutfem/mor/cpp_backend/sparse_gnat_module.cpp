#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

using scalar_type = double;
using index_type = std::int64_t;
using matrix_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>;
using vector_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>;

struct CsrMatrix {
    index_type rows = 0;
    index_type cols = 0;
    std::vector<index_type> indptr;
    std::vector<index_type> indices;
    std::vector<double> data;
};

py::array_t<double> vector_to_array(const vector_type& values)
{
    py::array_t<double> out(values.size());
    auto view = out.mutable_unchecked<1>();
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        view(i) = values(i);
    }
    return out;
}

py::array_t<double> matrix_to_array(const matrix_type& values)
{
    py::array_t<double> out({values.rows(), values.cols()});
    auto view = out.mutable_unchecked<2>();
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        for (ssize_t j = 0; j < view.shape(1); ++j) {
            view(i, j) = values(i, j);
        }
    }
    return out;
}

vector_type as_vector(py::handle obj, const char* label)
{
    auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr || arr.ndim() != 1) {
        throw py::value_error(std::string(label) + " must be a rank-1 float64 array.");
    }
    vector_type out(arr.shape(0));
    auto view = arr.unchecked<1>();
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        const double value = view(i);
        if (!std::isfinite(value)) {
            throw py::value_error(std::string(label) + " must contain only finite values.");
        }
        out(i) = value;
    }
    return out;
}

matrix_type as_matrix(py::handle obj, const char* label)
{
    auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr || arr.ndim() != 2) {
        throw py::value_error(std::string(label) + " must be a rank-2 float64 array.");
    }
    matrix_type out(arr.shape(0), arr.shape(1));
    auto view = arr.unchecked<2>();
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        for (ssize_t j = 0; j < view.shape(1); ++j) {
            const double value = view(i, j);
            if (!std::isfinite(value)) {
                throw py::value_error(std::string(label) + " must contain only finite values.");
            }
            out(i, j) = value;
        }
    }
    return out;
}

std::vector<index_type> as_index_vector(py::handle obj, const char* label)
{
    auto arr = py::array_t<index_type, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr || arr.ndim() != 1) {
        throw py::value_error(std::string(label) + " must be a rank-1 int64 array.");
    }
    std::vector<index_type> out(static_cast<std::size_t>(arr.shape(0)));
    auto view = arr.unchecked<1>();
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = static_cast<index_type>(view(i));
    }
    return out;
}

std::vector<double> as_double_vector(py::handle obj, const char* label)
{
    auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr || arr.ndim() != 1) {
        throw py::value_error(std::string(label) + " must be a rank-1 float64 array.");
    }
    std::vector<double> out(static_cast<std::size_t>(arr.shape(0)));
    auto view = arr.unchecked<1>();
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        const double value = view(i);
        if (!std::isfinite(value)) {
            throw py::value_error(std::string(label) + " must contain only finite values.");
        }
        out[static_cast<std::size_t>(i)] = value;
    }
    return out;
}

CsrMatrix parse_csr(py::dict payload)
{
    const std::string layout = py::str(payload[py::str("layout")]);
    if (layout != "csr") {
        throw py::value_error("native sparse GNAT lift must use CSR layout.");
    }
    auto shape = as_index_vector(payload[py::str("shape")], "sparse shape");
    if (shape.size() != 2 || shape[0] < 0 || shape[1] < 0) {
        throw py::value_error("sparse shape must have two nonnegative entries.");
    }
    CsrMatrix csr;
    csr.rows = shape[0];
    csr.cols = shape[1];
    csr.indptr = as_index_vector(payload[py::str("indptr")], "CSR indptr");
    csr.indices = as_index_vector(payload[py::str("indices")], "CSR indices");
    csr.data = as_double_vector(payload[py::str("data")], "CSR data");
    if (static_cast<index_type>(csr.indptr.size()) != csr.rows + 1) {
        throw py::value_error("CSR indptr length must be n_rows + 1.");
    }
    if (csr.indices.size() != csr.data.size()) {
        throw py::value_error("CSR indices/data size mismatch.");
    }
    if (!csr.indptr.empty() && csr.indptr.front() != 0) {
        throw py::value_error("CSR indptr must start at zero.");
    }
    if (!csr.indptr.empty() && csr.indptr.back() != static_cast<index_type>(csr.indices.size())) {
        throw py::value_error("CSR indptr[-1] must equal nnz.");
    }
    for (index_type row = 0; row < csr.rows; ++row) {
        const index_type start = csr.indptr[static_cast<std::size_t>(row)];
        const index_type stop = csr.indptr[static_cast<std::size_t>(row + 1)];
        if (start > stop || start < 0 || stop > static_cast<index_type>(csr.indices.size())) {
            throw py::value_error("CSR indptr is invalid.");
        }
        index_type previous = -1;
        for (index_type p = start; p < stop; ++p) {
            const index_type col = csr.indices[static_cast<std::size_t>(p)];
            if (col < 0 || col >= csr.cols) {
                throw py::value_error("CSR column index is out of range.");
            }
            if (col <= previous) {
                throw py::value_error("CSR row indices must be strictly increasing.");
            }
            previous = col;
        }
    }
    return csr;
}

vector_type csr_matvec(const CsrMatrix& csr, const vector_type& vector)
{
    if (vector.size() != csr.cols) {
        throw py::value_error("CSR matvec dimension mismatch.");
    }
    vector_type out = vector_type::Zero(csr.rows);
    for (index_type row = 0; row < csr.rows; ++row) {
        double sum = 0.0;
        const index_type start = csr.indptr[static_cast<std::size_t>(row)];
        const index_type stop = csr.indptr[static_cast<std::size_t>(row + 1)];
        for (index_type p = start; p < stop; ++p) {
            sum += csr.data[static_cast<std::size_t>(p)] * vector(csr.indices[static_cast<std::size_t>(p)]);
        }
        out(row) = sum;
    }
    return out;
}

matrix_type csr_matmat(const CsrMatrix& csr, const matrix_type& matrix)
{
    if (matrix.rows() != csr.cols) {
        throw py::value_error("CSR matmat dimension mismatch.");
    }
    matrix_type out = matrix_type::Zero(csr.rows, matrix.cols());
    for (index_type row = 0; row < csr.rows; ++row) {
        const index_type start = csr.indptr[static_cast<std::size_t>(row)];
        const index_type stop = csr.indptr[static_cast<std::size_t>(row + 1)];
        for (index_type p = start; p < stop; ++p) {
            const double value = csr.data[static_cast<std::size_t>(p)];
            const index_type col = csr.indices[static_cast<std::size_t>(p)];
            for (Eigen::Index mode = 0; mode < matrix.cols(); ++mode) {
                out(row, mode) += value * matrix(col, mode);
            }
        }
    }
    return out;
}

py::tuple apply_sparse_gnat_lift(py::dict sparse_lift, py::handle sampled_residual_obj, py::handle sampled_trial_jacobian_obj)
{
    const CsrMatrix csr = parse_csr(sparse_lift);
    const vector_type sampled_residual = as_vector(sampled_residual_obj, "sampled_residual");
    const matrix_type sampled_trial = as_matrix(sampled_trial_jacobian_obj, "sampled_trial_jacobian");
    if (csr.cols != sampled_residual.size() || csr.cols != sampled_trial.rows()) {
        throw py::value_error("Sparse GNAT lift columns must match sampled residual/Jacobian rows.");
    }
    return py::make_tuple(
        vector_to_array(csr_matvec(csr, sampled_residual)),
        matrix_to_array(csr_matmat(csr, sampled_trial))
    );
}

py::dict sparse_gnat_normal_equations(py::dict sparse_lift, py::handle sampled_residual_obj, py::handle sampled_trial_jacobian_obj)
{
    const CsrMatrix csr = parse_csr(sparse_lift);
    const vector_type sampled_residual = as_vector(sampled_residual_obj, "sampled_residual");
    const matrix_type sampled_trial = as_matrix(sampled_trial_jacobian_obj, "sampled_trial_jacobian");
    if (csr.cols != sampled_residual.size() || csr.cols != sampled_trial.rows()) {
        throw py::value_error("Sparse GNAT lift columns must match sampled residual/Jacobian rows.");
    }
    const Eigen::Index n_modes = sampled_trial.cols();
    matrix_type normal = matrix_type::Zero(n_modes, n_modes);
    vector_type rhs = vector_type::Zero(n_modes);
    double norm_sq = 0.0;
    for (index_type row = 0; row < csr.rows; ++row) {
        double lifted_residual = 0.0;
        vector_type lifted_trial = vector_type::Zero(n_modes);
        const index_type start = csr.indptr[static_cast<std::size_t>(row)];
        const index_type stop = csr.indptr[static_cast<std::size_t>(row + 1)];
        for (index_type p = start; p < stop; ++p) {
            const double value = csr.data[static_cast<std::size_t>(p)];
            const index_type col = csr.indices[static_cast<std::size_t>(p)];
            lifted_residual += value * sampled_residual(col);
            for (Eigen::Index mode = 0; mode < n_modes; ++mode) {
                lifted_trial(mode) += value * sampled_trial(col, mode);
            }
        }
        norm_sq += lifted_residual * lifted_residual;
        rhs.noalias() -= lifted_trial * lifted_residual;
        normal.noalias() += lifted_trial * lifted_trial.transpose();
    }
    py::dict out;
    out["normal_matrix"] = matrix_to_array(normal);
    out["normal_rhs"] = vector_to_array(rhs);
    out["lifted_residual_norm"] = std::sqrt(norm_sq);
    out["nnz"] = static_cast<index_type>(csr.data.size());
    out["path"] = "csr_direct_normal";
    return out;
}

}  // namespace

PYBIND11_MODULE(_pycutfem_mor_sparse_gnat_2026_05_15_mor_sparse_gnat_v1, m)
{
    m.doc() = "pycutfem MOR sparse GNAT backend";
    m.def("apply_sparse_gnat_lift", &apply_sparse_gnat_lift, py::arg("sparse_lift"), py::arg("sampled_residual"), py::arg("sampled_trial_jacobian"));
    m.def("sparse_gnat_normal_equations", &sparse_gnat_normal_equations, py::arg("sparse_lift"), py::arg("sampled_residual"), py::arg("sampled_trial_jacobian"));
}
