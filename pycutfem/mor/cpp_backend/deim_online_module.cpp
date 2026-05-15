#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

using scalar_type = double;
using matrix_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>;
using vector_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>;

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

std::vector<matrix_type> as_matrix_stack(py::handle obj, const char* label)
{
    auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr || arr.ndim() != 3) {
        throw py::value_error(std::string(label) + " must be a rank-3 float64 array.");
    }
    std::vector<matrix_type> out;
    out.reserve(static_cast<std::size_t>(arr.shape(0)));
    auto view = arr.unchecked<3>();
    for (ssize_t k = 0; k < view.shape(0); ++k) {
        matrix_type block(view.shape(1), view.shape(2));
        for (ssize_t i = 0; i < view.shape(1); ++i) {
            for (ssize_t j = 0; j < view.shape(2); ++j) {
                const double value = view(k, i, j);
                if (!std::isfinite(value)) {
                    throw py::value_error(std::string(label) + " must contain only finite values.");
                }
                block(i, j) = value;
            }
        }
        out.emplace_back(std::move(block));
    }
    return out;
}

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

struct InterpolationSolve {
    matrix_type coefficients;
    int rank = 0;
    std::string method;
};

InterpolationSolve solve_coefficients(const matrix_type& selected_basis, const matrix_type& rhs, double rcond)
{
    if (selected_basis.rows() != rhs.rows()) {
        throw py::value_error("selected_basis row count must match selected_values row count.");
    }
    if (selected_basis.cols() == 0) {
        throw py::value_error("selected_basis must have at least one column.");
    }

    InterpolationSolve out;
    if (selected_basis.rows() == selected_basis.cols()) {
        Eigen::ColPivHouseholderQR<matrix_type> qr(selected_basis);
        if (rcond > 0.0) {
            qr.setThreshold(rcond);
        }
        out.coefficients = qr.solve(rhs);
        out.rank = static_cast<int>(qr.rank());
        out.method = "qr";
        if (out.rank == selected_basis.cols() && out.coefficients.allFinite()) {
            return out;
        }
    }

    Eigen::JacobiSVD<matrix_type> svd(selected_basis, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (rcond > 0.0) {
        svd.setThreshold(rcond);
    }
    out.coefficients = svd.solve(rhs);
    out.rank = static_cast<int>(svd.rank());
    out.method = "svd";
    if (!out.coefficients.allFinite()) {
        throw py::value_error("interpolation solve produced non-finite coefficients.");
    }
    return out;
}

py::dict solve_interpolation(py::handle selected_basis_obj, py::handle selected_values_obj, double rcond)
{
    const matrix_type selected_basis = as_matrix(selected_basis_obj, "selected_basis");
    py::dict out;
    auto values_1d = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(selected_values_obj);
    if (!values_1d) {
        throw py::value_error("selected_values must be a float64 vector or matrix.");
    }

    if (values_1d.ndim() == 1) {
        const vector_type rhs_vec = as_vector(selected_values_obj, "selected_values");
        matrix_type rhs(rhs_vec.size(), 1);
        rhs.col(0) = rhs_vec;
        const InterpolationSolve solved = solve_coefficients(selected_basis, rhs, rcond);
        out["coefficients"] = vector_to_array(solved.coefficients.col(0));
        out["rank"] = solved.rank;
        out["method"] = solved.method;
        return out;
    }
    if (values_1d.ndim() == 2) {
        const matrix_type rhs = as_matrix(selected_values_obj, "selected_values");
        const InterpolationSolve solved = solve_coefficients(selected_basis, rhs, rcond);
        out["coefficients"] = matrix_to_array(solved.coefficients);
        out["rank"] = solved.rank;
        out["method"] = solved.method;
        return out;
    }
    throw py::value_error("selected_values must be a float64 vector or matrix.");
}

py::dict compose_reduced_system(py::handle coefficients_obj, py::handle residual_terms_obj, py::handle jacobian_terms_obj)
{
    const vector_type coeffs = as_vector(coefficients_obj, "coefficients");
    const matrix_type residual_terms = as_matrix(residual_terms_obj, "residual_terms");
    if (residual_terms.rows() != coeffs.size()) {
        throw py::value_error("residual_terms first dimension must match coefficient count.");
    }
    vector_type residual = vector_type::Zero(residual_terms.cols());
    for (Eigen::Index term = 0; term < coeffs.size(); ++term) {
        residual.noalias() += coeffs(term) * residual_terms.row(term).transpose();
    }

    py::dict out;
    out["residual"] = vector_to_array(residual);
    if (!jacobian_terms_obj.is_none()) {
        const auto jacobian_terms = as_matrix_stack(jacobian_terms_obj, "jacobian_terms");
        if (static_cast<Eigen::Index>(jacobian_terms.size()) != coeffs.size()) {
            throw py::value_error("jacobian_terms first dimension must match coefficient count.");
        }
        if (!jacobian_terms.empty()) {
            matrix_type jacobian = matrix_type::Zero(jacobian_terms[0].rows(), jacobian_terms[0].cols());
            for (Eigen::Index term = 0; term < coeffs.size(); ++term) {
                if (jacobian_terms[static_cast<std::size_t>(term)].rows() != jacobian.rows()
                    || jacobian_terms[static_cast<std::size_t>(term)].cols() != jacobian.cols()) {
                    throw py::value_error("all jacobian term blocks must have the same shape.");
                }
                jacobian.noalias() += coeffs(term) * jacobian_terms[static_cast<std::size_t>(term)];
            }
            out["jacobian"] = matrix_to_array(jacobian);
        }
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_pycutfem_mor_deim_online_2026_05_15_mor_deim_online_v1, m)
{
    m.doc() = "pycutfem MOR native DEIM/QDEIM online evaluator";
    m.def("solve_interpolation", &solve_interpolation, py::arg("selected_basis"), py::arg("selected_values"), py::arg("rcond") = -1.0);
    m.def("compose_reduced_system", &compose_reduced_system, py::arg("coefficients"), py::arg("residual_terms"), py::arg("jacobian_terms") = py::none());
}
