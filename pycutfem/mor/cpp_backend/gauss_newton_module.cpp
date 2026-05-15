#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

using scalar_type = double;
using matrix_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>;
using vector_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>;

py::array_t<scalar_type, py::array::c_style | py::array::forcecast>
as_double_array(py::handle obj, const char* label)
{
    auto arr = py::array_t<scalar_type, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr) {
        throw py::value_error(std::string(label) + " must be convertible to a contiguous float64 array.");
    }
    return arr;
}

matrix_type as_matrix(py::handle obj, const char* label)
{
    auto arr = as_double_array(obj, label);
    if (arr.ndim() != 2) {
        throw py::value_error(std::string(label) + " must be a rank-2 array.");
    }
    matrix_type out(arr.shape(0), arr.shape(1));
    auto view = arr.unchecked<2>();
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        for (ssize_t j = 0; j < view.shape(1); ++j) {
            const scalar_type value = view(i, j);
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
    auto arr = as_double_array(obj, label);
    if (arr.ndim() != 1) {
        throw py::value_error(std::string(label) + " must be a rank-1 array.");
    }
    vector_type out(arr.shape(0));
    auto view = arr.unchecked<1>();
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        const scalar_type value = view(i);
        if (!std::isfinite(value)) {
            throw py::value_error(std::string(label) + " must contain only finite values.");
        }
        out(i) = value;
    }
    return out;
}

vector_type optional_weights(py::handle obj, Eigen::Index n_rows)
{
    vector_type weights(n_rows);
    weights.setOnes();
    if (obj.is_none()) {
        return weights;
    }
    vector_type raw = as_vector(obj, "weights");
    if (raw.size() != n_rows) {
        throw py::value_error("weights size must match the residual/Jacobian row count.");
    }
    for (Eigen::Index i = 0; i < raw.size(); ++i) {
        if (raw(i) < 0.0) {
            throw py::value_error("weights must be nonnegative.");
        }
        weights(i) = raw(i);
    }
    return weights;
}

vector_type optional_damping_diagonal(py::handle obj, Eigen::Index n_cols)
{
    vector_type diag(n_cols);
    diag.setOnes();
    if (obj.is_none()) {
        return diag;
    }
    vector_type raw = as_vector(obj, "damping_diagonal");
    if (raw.size() != n_cols) {
        throw py::value_error("damping_diagonal size must match the number of columns.");
    }
    for (Eigen::Index i = 0; i < raw.size(); ++i) {
        if (raw(i) < 0.0) {
            throw py::value_error("damping_diagonal must be nonnegative.");
        }
        diag(i) = raw(i);
    }
    return diag;
}

py::array_t<scalar_type> vector_to_array(const vector_type& values)
{
    py::array_t<scalar_type> out(values.size());
    auto view = out.mutable_unchecked<1>();
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        view(i) = values(i);
    }
    return out;
}

py::array_t<scalar_type> matrix_to_array(const matrix_type& values)
{
    py::array_t<scalar_type> out({values.rows(), values.cols()});
    auto view = out.mutable_unchecked<2>();
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        for (ssize_t j = 0; j < view.shape(1); ++j) {
            view(i, j) = values(i, j);
        }
    }
    return out;
}

struct WeightedSystem {
    matrix_type A;
    vector_type b;
    matrix_type normal_matrix;
    vector_type normal_rhs;
    vector_type gradient;
    double weighted_residual_norm = 0.0;
};

WeightedSystem build_weighted_system(
    const matrix_type& jacobian,
    const vector_type& residual,
    const vector_type& weights,
    double damping,
    const vector_type& damping_diagonal
)
{
    if (jacobian.rows() != residual.size()) {
        throw py::value_error("jacobian row count must match residual size.");
    }
    if (weights.size() != residual.size()) {
        throw py::value_error("weights size must match residual size.");
    }
    if (damping_diagonal.size() != jacobian.cols()) {
        throw py::value_error("damping_diagonal size must match jacobian column count.");
    }
    if (!std::isfinite(damping) || damping < 0.0) {
        throw py::value_error("damping must be finite and nonnegative.");
    }

    const Eigen::Index rows = jacobian.rows();
    const Eigen::Index cols = jacobian.cols();
    const bool has_damping = damping > 0.0 && cols > 0;
    const Eigen::Index aug_rows = rows + (has_damping ? cols : 0);
    WeightedSystem system;
    system.A.resize(aug_rows, cols);
    system.b.resize(aug_rows);
    system.A.setZero();
    system.b.setZero();

    double norm_sq = 0.0;
    for (Eigen::Index i = 0; i < rows; ++i) {
        const double w = weights(i);
        const double scale = std::sqrt(std::max(w, 0.0));
        system.A.row(i) = scale * jacobian.row(i);
        system.b(i) = -scale * residual(i);
        norm_sq += w * residual(i) * residual(i);
    }
    system.weighted_residual_norm = std::sqrt(std::max(norm_sq, 0.0));

    if (has_damping) {
        const double damp_scale = std::sqrt(damping);
        for (Eigen::Index j = 0; j < cols; ++j) {
            system.A(rows + j, j) = damp_scale * damping_diagonal(j);
        }
    }

    system.normal_matrix = system.A.transpose() * system.A;
    system.normal_rhs = system.A.transpose() * system.b;
    system.gradient = -system.normal_rhs;
    return system;
}

struct StepSolve {
    vector_type step;
    int rank = 0;
    std::string method;
    double condition_estimate = std::numeric_limits<double>::quiet_NaN();
};

double condition_from_singular_values(const Eigen::VectorXd& singular_values)
{
    if (singular_values.size() == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double smax = singular_values(0);
    double smin = std::numeric_limits<double>::infinity();
    for (Eigen::Index i = 0; i < singular_values.size(); ++i) {
        const double s = singular_values(i);
        if (s > 0.0 && s < smin) {
            smin = s;
        }
    }
    if (!std::isfinite(smax) || smax <= 0.0 || !std::isfinite(smin) || smin <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return smax / smin;
}

StepSolve solve_svd(const matrix_type& A, const vector_type& b, double rcond)
{
    Eigen::JacobiSVD<matrix_type> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (rcond > 0.0) {
        svd.setThreshold(rcond);
    }
    StepSolve out;
    out.step = svd.solve(b);
    out.rank = static_cast<int>(svd.rank());
    out.method = "svd";
    out.condition_estimate = condition_from_singular_values(svd.singularValues());
    return out;
}

StepSolve solve_qr(const matrix_type& A, const vector_type& b, double rcond)
{
    Eigen::ColPivHouseholderQR<matrix_type> qr(A);
    if (rcond > 0.0) {
        qr.setThreshold(rcond);
    }
    StepSolve out;
    out.step = qr.solve(b);
    out.rank = static_cast<int>(qr.rank());
    out.method = "qr";
    return out;
}

bool ldlt_has_full_rank(const Eigen::LDLT<matrix_type>& ldlt, Eigen::Index cols, double rcond)
{
    const auto diag = ldlt.vectorD();
    if (diag.size() != cols) {
        return false;
    }
    double scale = 0.0;
    for (Eigen::Index i = 0; i < diag.size(); ++i) {
        scale = std::max(scale, std::abs(diag(i)));
    }
    if (!std::isfinite(scale) || scale <= 0.0) {
        return false;
    }
    const double relative = rcond > 0.0
        ? rcond
        : std::numeric_limits<double>::epsilon() * static_cast<double>(std::max<Eigen::Index>(cols, 1));
    const double threshold = relative * scale;
    for (Eigen::Index i = 0; i < diag.size(); ++i) {
        if (std::abs(diag(i)) <= threshold) {
            return false;
        }
    }
    return true;
}

StepSolve solve_normal(const matrix_type& H, const vector_type& rhs, const matrix_type& A, const vector_type& b, double rcond)
{
    StepSolve out;
    Eigen::LDLT<matrix_type> ldlt(H);
    if (ldlt.info() == Eigen::Success && ldlt_has_full_rank(ldlt, H.cols(), rcond)) {
        out.step = ldlt.solve(rhs);
        if (ldlt.info() == Eigen::Success && out.step.allFinite()) {
            out.rank = static_cast<int>(H.cols());
            out.method = "normal_ldlt";
            return out;
        }
    }
    out = solve_svd(A, b, rcond);
    out.method = "normal_svd_fallback";
    return out;
}

StepSolve solve_step(
    const WeightedSystem& system,
    const std::string& method,
    double rcond
)
{
    if (system.A.cols() == 0) {
        StepSolve out;
        out.step.resize(0);
        out.rank = 0;
        out.method = "empty";
        return out;
    }

    const std::string requested = method.empty() ? "auto" : method;
    if (requested == "svd") {
        return solve_svd(system.A, system.b, rcond);
    }
    if (requested == "qr") {
        return solve_qr(system.A, system.b, rcond);
    }
    if (requested == "normal") {
        return solve_normal(system.normal_matrix, system.normal_rhs, system.A, system.b, rcond);
    }
    if (requested != "auto") {
        throw py::value_error("method must be one of 'auto', 'qr', 'svd', or 'normal'.");
    }

    StepSolve qr = solve_qr(system.A, system.b, rcond);
    if (qr.rank >= system.A.cols() && qr.step.allFinite()) {
        qr.method = "auto_qr";
        return qr;
    }
    StepSolve svd = solve_svd(system.A, system.b, rcond);
    svd.method = "auto_svd";
    return svd;
}

py::dict form_normal_equations(
    py::handle jacobian_obj,
    py::handle residual_obj,
    py::handle weights_obj,
    double damping,
    py::handle damping_diagonal_obj
)
{
    const matrix_type jacobian = as_matrix(jacobian_obj, "jacobian");
    const vector_type residual = as_vector(residual_obj, "residual");
    const vector_type weights = optional_weights(weights_obj, residual.size());
    const vector_type damping_diagonal = optional_damping_diagonal(damping_diagonal_obj, jacobian.cols());
    const auto system = build_weighted_system(jacobian, residual, weights, damping, damping_diagonal);

    py::dict out;
    out["normal_matrix"] = matrix_to_array(system.normal_matrix);
    out["normal_rhs"] = vector_to_array(system.normal_rhs);
    out["gradient"] = vector_to_array(system.gradient);
    out["weighted_residual_norm"] = system.weighted_residual_norm;
    return out;
}

py::dict gauss_newton_step(
    py::handle jacobian_obj,
    py::handle residual_obj,
    py::handle weights_obj,
    double damping,
    py::handle damping_diagonal_obj,
    const std::string& method,
    double rcond
)
{
    const matrix_type jacobian = as_matrix(jacobian_obj, "jacobian");
    const vector_type residual = as_vector(residual_obj, "residual");
    const vector_type weights = optional_weights(weights_obj, residual.size());
    const vector_type damping_diagonal = optional_damping_diagonal(damping_diagonal_obj, jacobian.cols());
    const auto system = build_weighted_system(jacobian, residual, weights, damping, damping_diagonal);
    const auto solved = solve_step(system, method, rcond);

    py::dict out;
    out["step"] = vector_to_array(solved.step);
    out["rank"] = solved.rank;
    out["method"] = solved.method;
    out["condition_estimate"] = solved.condition_estimate;
    out["weighted_residual_norm"] = system.weighted_residual_norm;
    out["linearized_residual_norm"] = solved.step.size() == 0
        ? system.b.norm()
        : (system.A * solved.step - system.b).norm();
    out["gradient_norm"] = system.gradient.norm();
    out["damping"] = damping;
    out["converged"] = static_cast<bool>(solved.step.allFinite());
    return out;
}

}  // namespace

PYBIND11_MODULE(_pycutfem_mor_gauss_newton_2026_05_15_mor_gauss_newton_v1, m)
{
    m.doc() = "pycutfem MOR dense Gauss-Newton step backend";
    m.def(
        "form_normal_equations",
        &form_normal_equations,
        py::arg("jacobian"),
        py::arg("residual"),
        py::arg("weights") = py::none(),
        py::arg("damping") = 0.0,
        py::arg("damping_diagonal") = py::none()
    );
    m.def(
        "gauss_newton_step",
        &gauss_newton_step,
        py::arg("jacobian"),
        py::arg("residual"),
        py::arg("weights") = py::none(),
        py::arg("damping") = 0.0,
        py::arg("damping_diagonal") = py::none(),
        py::arg("method") = "auto",
        py::arg("rcond") = -1.0
    );
}
