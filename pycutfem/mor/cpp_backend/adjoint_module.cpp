#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>
#include <Eigen/SVD>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

using matrix_type = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using vector_type = Eigen::Matrix<double, Eigen::Dynamic, 1>;

py::array_t<double, py::array::c_style | py::array::forcecast>
as_double_array(py::handle obj, const char* label)
{
    auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obj);
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
    auto arr = as_double_array(obj, label);
    if (arr.ndim() != 1) {
        throw py::value_error(std::string(label) + " must be a rank-1 array.");
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

py::array_t<double> vector_to_array(const vector_type& values)
{
    py::array_t<double> out(values.size());
    auto view = out.mutable_unchecked<1>();
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        view(i) = values(i);
    }
    return out;
}

py::array_t<int> int_vector_to_array(const std::vector<int>& values)
{
    py::array_t<int> out(values.size());
    auto view = out.mutable_unchecked<1>();
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        view(i) = values[static_cast<std::size_t>(i)];
    }
    return out;
}

py::array_t<double> double_vector_to_array(const std::vector<double>& values)
{
    py::array_t<double> out(values.size());
    auto view = out.mutable_unchecked<1>();
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        view(i) = values[static_cast<std::size_t>(i)];
    }
    return out;
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

struct TransposeSolve {
    vector_type solution;
    int rank = 0;
    double residual_norm = 0.0;
    double condition = std::numeric_limits<double>::quiet_NaN();
};

TransposeSolve solve_transpose(const matrix_type& jacobian, const vector_type& rhs, double rcond)
{
    if (jacobian.cols() != rhs.size()) {
        throw py::value_error("rhs size must match jacobian column count for J.T z = rhs.");
    }
    const matrix_type A = jacobian.transpose();
    Eigen::JacobiSVD<matrix_type> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (rcond > 0.0) {
        svd.setThreshold(rcond);
    }
    TransposeSolve out;
    out.solution = svd.solve(rhs);
    out.rank = static_cast<int>(svd.rank());
    out.residual_norm = (A * out.solution - rhs).norm();
    out.condition = condition_from_singular_values(svd.singularValues());
    return out;
}

py::dict solve_transpose_system(py::handle jacobian_obj, py::handle rhs_obj, double rcond)
{
    const matrix_type J = as_matrix(jacobian_obj, "jacobian");
    const vector_type rhs = as_vector(rhs_obj, "rhs");
    TransposeSolve solved = solve_transpose(J, rhs, rcond);
    py::dict out;
    out["solution"] = vector_to_array(solved.solution);
    out["rank"] = solved.rank;
    out["residual_norm"] = solved.residual_norm;
    out["condition_estimate"] = solved.condition;
    return out;
}

std::vector<matrix_type> matrix_sequence(py::sequence seq, const char* label)
{
    std::vector<matrix_type> out;
    out.reserve(static_cast<std::size_t>(py::len(seq)));
    for (py::handle item : seq) {
        out.push_back(as_matrix(item, label));
    }
    return out;
}

std::vector<vector_type> vector_sequence(py::sequence seq, const char* label)
{
    std::vector<vector_type> out;
    out.reserve(static_cast<std::size_t>(py::len(seq)));
    for (py::handle item : seq) {
        out.push_back(as_vector(item, label));
    }
    return out;
}

py::dict solve_discrete_adjoint(
    py::sequence jacobians_obj,
    py::sequence qoi_gradients_obj,
    py::handle previous_state_jacobians_obj,
    double rcond
)
{
    const std::vector<matrix_type> jacobians = matrix_sequence(jacobians_obj, "jacobian");
    const std::vector<vector_type> gradients = vector_sequence(qoi_gradients_obj, "qoi_gradient");
    const std::size_t n_steps = jacobians.size();
    if (gradients.size() != n_steps) {
        throw py::value_error("qoi_gradients must contain one vector per jacobian.");
    }
    std::vector<matrix_type> previous_jacobians;
    if (!previous_state_jacobians_obj.is_none()) {
        previous_jacobians = matrix_sequence(previous_state_jacobians_obj.cast<py::sequence>(), "previous_state_jacobian");
        if (previous_jacobians.size() != n_steps) {
            throw py::value_error("previous_state_jacobians must be empty/None or contain one matrix per step.");
        }
    }

    std::vector<vector_type> adjoints(n_steps);
    std::vector<int> ranks(n_steps, 0);
    std::vector<double> residual_norms(n_steps, 0.0);
    std::vector<double> conditions(n_steps, std::numeric_limits<double>::quiet_NaN());

    for (std::size_t rev = 0; rev < n_steps; ++rev) {
        const std::size_t i = n_steps - 1 - rev;
        vector_type rhs = gradients[i];
        if (i + 1 < n_steps && !previous_jacobians.empty()) {
            const matrix_type& coupling = previous_jacobians[i + 1];
            if (coupling.rows() != adjoints[i + 1].size() || coupling.cols() != rhs.size()) {
                throw py::value_error("previous-state jacobian has incompatible shape for adjoint coupling.");
            }
            rhs -= coupling.transpose() * adjoints[i + 1];
        }
        TransposeSolve solved = solve_transpose(jacobians[i], rhs, rcond);
        adjoints[i] = solved.solution;
        ranks[i] = solved.rank;
        residual_norms[i] = solved.residual_norm;
        conditions[i] = solved.condition;
    }

    py::list z_list;
    for (const auto& z : adjoints) {
        z_list.append(vector_to_array(z));
    }
    py::dict out;
    out["adjoints"] = z_list;
    out["rank_history"] = int_vector_to_array(ranks);
    out["residual_norm_history"] = double_vector_to_array(residual_norms);
    out["condition_estimate_history"] = double_vector_to_array(conditions);
    out["backend"] = "cpp_native_adjoint";
    return out;
}

}  // namespace

PYBIND11_MODULE(_pycutfem_mor_adjoint_2026_05_16_mor_adjoint_v1, m)
{
    m.doc() = "pycutfem MOR native adjoint and DWR algebra";
    m.def(
        "solve_transpose_system",
        &solve_transpose_system,
        py::arg("jacobian"),
        py::arg("rhs"),
        py::arg("rcond") = -1.0
    );
    m.def(
        "solve_discrete_adjoint",
        &solve_discrete_adjoint,
        py::arg("jacobians"),
        py::arg("qoi_gradients"),
        py::arg("previous_state_jacobians") = py::none(),
        py::arg("rcond") = -1.0
    );
}
