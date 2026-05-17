#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>
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
        throw py::value_error("weights size must match residual size.");
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
    WeightedSystem system;
    system.A.resize(rows + (has_damping ? cols : 0), cols);
    system.b.resize(rows + (has_damping ? cols : 0));
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

double svd_threshold(const Eigen::VectorXd& singular_values, Eigen::Index rows, Eigen::Index cols, double rcond)
{
    if (singular_values.size() == 0) {
        return 0.0;
    }
    double scale = 0.0;
    for (Eigen::Index i = 0; i < singular_values.size(); ++i) {
        scale = std::max(scale, std::abs(singular_values(i)));
    }
    if (!std::isfinite(scale) || scale <= 0.0) {
        return 0.0;
    }
    const double relative = rcond > 0.0
        ? rcond
        : std::numeric_limits<double>::epsilon() * static_cast<double>(std::max<Eigen::Index>(std::max(rows, cols), 1));
    return relative * scale;
}

int rank_from_svd(const Eigen::VectorXd& singular_values, Eigen::Index rows, Eigen::Index cols, double rcond)
{
    const double threshold = svd_threshold(singular_values, rows, cols, rcond);
    int rank = 0;
    for (Eigen::Index i = 0; i < singular_values.size(); ++i) {
        if (singular_values(i) > threshold) {
            ++rank;
        }
    }
    return rank;
}

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

struct LeastSquaresSolve {
    vector_type x;
    int rank = 0;
    double condition = std::numeric_limits<double>::quiet_NaN();
};

LeastSquaresSolve solve_lstsq_svd(const matrix_type& A, const vector_type& b, double rcond)
{
    LeastSquaresSolve out;
    out.x.resize(A.cols());
    out.x.setZero();
    if (A.cols() == 0) {
        out.rank = 0;
        return out;
    }
    if (A.rows() == 0) {
        out.rank = 0;
        return out;
    }
    Eigen::JacobiSVD<matrix_type> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (rcond > 0.0) {
        svd.setThreshold(rcond);
    }
    out.x = svd.solve(b);
    out.rank = static_cast<int>(svd.rank());
    out.condition = condition_from_singular_values(svd.singularValues());
    return out;
}

struct ConstraintBasis {
    vector_type particular;
    matrix_type nullspace;
    int rank = 0;
    double violation = 0.0;
};

ConstraintBasis constraint_particular_and_nullspace(const matrix_type& C, const vector_type& h, double rcond)
{
    ConstraintBasis out;
    const Eigen::Index n = C.cols();
    if (C.rows() == 0) {
        out.particular.resize(n);
        out.particular.setZero();
        out.nullspace.setIdentity(n, n);
        out.rank = 0;
        out.violation = 0.0;
        return out;
    }

    Eigen::JacobiSVD<matrix_type> svd(C, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto& singular_values = svd.singularValues();
    const int rank = rank_from_svd(singular_values, C.rows(), C.cols(), rcond);
    out.rank = rank;
    out.particular.resize(n);
    out.particular.setZero();
    for (int i = 0; i < rank; ++i) {
        out.particular += svd.matrixV().col(i) * (svd.matrixU().col(i).dot(h) / singular_values(i));
    }

    const Eigen::Index nullity = n - rank;
    out.nullspace.resize(n, nullity);
    for (Eigen::Index j = 0; j < nullity; ++j) {
        out.nullspace.col(j) = svd.matrixV().col(rank + j);
    }
    out.violation = (C * out.particular - h).norm();
    return out;
}

vector_type multipliers_from_stationarity(const matrix_type& C, const vector_type& stationarity_rhs, double rcond)
{
    if (C.rows() == 0) {
        vector_type out(0);
        return out;
    }
    return solve_lstsq_svd(C.transpose(), stationarity_rhs, rcond).x;
}

struct ConstrainedSolve {
    vector_type step;
    vector_type multipliers;
    int rank = 0;
    int constraint_rank = 0;
    std::string method;
    double condition_estimate = std::numeric_limits<double>::quiet_NaN();
    double constraint_violation = 0.0;
};

ConstrainedSolve solve_nullspace(
    const WeightedSystem& system,
    const matrix_type& C,
    const vector_type& h,
    double rcond,
    const std::string& method
)
{
    ConstrainedSolve out;
    const ConstraintBasis basis = constraint_particular_and_nullspace(C, h, rcond);
    const double tolerance = 100.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::max(C.norm(), h.norm()));
    if (basis.violation > std::max(tolerance, 1.0e-12)) {
        throw py::value_error("equality constraints are inconsistent.");
    }

    if (basis.nullspace.cols() == 0) {
        out.step = basis.particular;
        out.rank = 0;
        out.condition_estimate = std::numeric_limits<double>::quiet_NaN();
    } else {
        const matrix_type A_red = system.A * basis.nullspace;
        const vector_type b_red = system.b - system.A * basis.particular;
        const LeastSquaresSolve reduced = solve_lstsq_svd(A_red, b_red, rcond);
        out.step = basis.particular + basis.nullspace * reduced.x;
        out.rank = reduced.rank;
        out.condition_estimate = reduced.condition;
    }
    out.constraint_rank = basis.rank;
    out.constraint_violation = (C * out.step - h).norm();
    out.multipliers = multipliers_from_stationarity(
        C,
        system.normal_rhs - system.normal_matrix * out.step,
        rcond
    );
    out.method = method;
    return out;
}

ConstrainedSolve solve_kkt_or_fallback(
    const WeightedSystem& system,
    const matrix_type& C,
    const vector_type& h,
    double rcond
)
{
    const Eigen::Index n = system.normal_matrix.rows();
    const Eigen::Index m = C.rows();
    matrix_type K(n + m, n + m);
    K.setZero();
    K.topLeftCorner(n, n) = system.normal_matrix;
    K.topRightCorner(n, m) = C.transpose();
    K.bottomLeftCorner(m, n) = C;
    vector_type rhs(n + m);
    rhs.head(n) = system.normal_rhs;
    rhs.tail(m) = h;

    Eigen::LDLT<matrix_type> ldlt(K);
    if (ldlt.info() == Eigen::Success) {
        vector_type sol = ldlt.solve(rhs);
        if (ldlt.info() == Eigen::Success && sol.allFinite()) {
            ConstrainedSolve out;
            out.step = sol.head(n);
            out.multipliers = sol.tail(m);
            out.constraint_violation = (C * out.step - h).norm();
            if (out.constraint_violation <= 1.0e-10) {
                out.rank = solve_lstsq_svd(system.A, system.b, rcond).rank;
                out.constraint_rank = constraint_particular_and_nullspace(C, h, rcond).rank;
                out.method = "kkt_ldlt";
                Eigen::JacobiSVD<matrix_type> svd(K, Eigen::ComputeThinU | Eigen::ComputeThinV);
                out.condition_estimate = condition_from_singular_values(svd.singularValues());
                return out;
            }
        }
    }

    ConstrainedSolve fallback = solve_nullspace(system, C, h, rcond, "kkt_nullspace_fallback");
    return fallback;
}

py::dict equality_constrained_gauss_newton_step(
    py::handle jacobian_obj,
    py::handle residual_obj,
    py::handle constraint_matrix_obj,
    py::handle constraint_rhs_obj,
    py::handle weights_obj,
    double damping,
    py::handle damping_diagonal_obj,
    const std::string& method,
    double rcond
)
{
    const matrix_type jacobian = as_matrix(jacobian_obj, "jacobian");
    const vector_type residual = as_vector(residual_obj, "residual");
    const matrix_type C = as_matrix(constraint_matrix_obj, "constraint_matrix");
    const vector_type h = as_vector(constraint_rhs_obj, "constraint_rhs");
    if (C.cols() != jacobian.cols()) {
        throw py::value_error("constraint_matrix column count must match jacobian column count.");
    }
    if (C.rows() != h.size()) {
        throw py::value_error("constraint_rhs size must match constraint_matrix row count.");
    }
    const vector_type weights = optional_weights(weights_obj, residual.size());
    const vector_type damping_diagonal = optional_damping_diagonal(damping_diagonal_obj, jacobian.cols());
    const WeightedSystem system = build_weighted_system(jacobian, residual, weights, damping, damping_diagonal);

    const std::string requested = method.empty() ? "auto" : method;
    ConstrainedSolve solved;
    if (requested == "kkt") {
        solved = solve_kkt_or_fallback(system, C, h, rcond);
    } else if (requested == "auto") {
        solved = solve_nullspace(system, C, h, rcond, "auto_nullspace_svd");
    } else if (requested == "nullspace" || requested == "svd") {
        solved = solve_nullspace(system, C, h, rcond, "nullspace_svd");
    } else {
        throw py::value_error("method must be one of 'auto', 'nullspace', 'svd', or 'kkt'.");
    }

    const double linearized_residual_norm = (system.A * solved.step - system.b).norm();
    py::dict out;
    out["step"] = vector_to_array(solved.step);
    out["multipliers"] = vector_to_array(solved.multipliers);
    out["rank"] = solved.rank;
    out["constraint_rank"] = solved.constraint_rank;
    out["method"] = solved.method;
    out["weighted_residual_norm"] = system.weighted_residual_norm;
    out["linearized_residual_norm"] = linearized_residual_norm;
    out["constraint_violation_norm"] = solved.constraint_violation;
    out["gradient_norm"] = system.gradient.norm();
    out["damping"] = damping;
    out["converged"] = static_cast<bool>(solved.step.allFinite() && solved.constraint_violation <= 1.0e-10);
    out["condition_estimate"] = solved.condition_estimate;
    return out;
}

}  // namespace

PYBIND11_MODULE(_pycutfem_mor_constrained_gauss_newton_2026_05_15_mor_constrained_gauss_newton_v1, m)
{
    m.doc() = "pycutfem MOR equality-constrained Gauss-Newton backend";
    m.def(
        "equality_constrained_gauss_newton_step",
        &equality_constrained_gauss_newton_step,
        py::arg("jacobian"),
        py::arg("residual"),
        py::arg("constraint_matrix"),
        py::arg("constraint_rhs"),
        py::arg("weights") = py::none(),
        py::arg("damping") = 0.0,
        py::arg("damping_diagonal") = py::none(),
        py::arg("method") = "auto",
        py::arg("rcond") = -1.0
    );
}
