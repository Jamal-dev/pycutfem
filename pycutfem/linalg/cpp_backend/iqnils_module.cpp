#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

using scalar_type = double;
using vector_type = Eigen::VectorXd;
using matrix_type = Eigen::MatrixXd;

struct FlatArray {
    std::vector<ssize_t> shape;
    vector_type values;
};

FlatArray flatten_array(py::handle obj, const char* label)
{
    auto arr = py::array_t<scalar_type, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr) {
        throw std::runtime_error(std::string(label) + " must be convertible to a contiguous float64 array.");
    }

    auto info = arr.request();
    if (info.ndim < 1 || info.ndim > 2) {
        throw std::runtime_error(std::string(label) + " must have 1 or 2 dimensions.");
    }

    ssize_t size = 1;
    std::vector<ssize_t> shape;
    shape.reserve(static_cast<std::size_t>(info.ndim));
    for (ssize_t i = 0; i < info.ndim; ++i) {
        shape.push_back(info.shape[static_cast<std::size_t>(i)]);
        size *= info.shape[static_cast<std::size_t>(i)];
    }

    FlatArray out;
    out.shape = std::move(shape);
    out.values.resize(size);
    auto* data = static_cast<const scalar_type*>(info.ptr);
    for (ssize_t i = 0; i < size; ++i) {
        out.values[i] = data[i];
    }
    return out;
}


py::array_t<scalar_type> reshape_like(const vector_type& values, const std::vector<ssize_t>& shape)
{
    if (shape.empty()) {
        py::array_t<scalar_type> out(1);
        auto view = out.mutable_unchecked<1>();
        view(0) = values[0];
        return out;
    }
    if (shape.size() == 1) {
        py::array_t<scalar_type> out(shape[0]);
        auto view = out.mutable_unchecked<1>();
        for (ssize_t i = 0; i < shape[0]; ++i) {
            view(i) = values[i];
        }
        return out;
    }
    if (shape.size() == 2) {
        py::array_t<scalar_type> out({shape[0], shape[1]});
        auto view = out.mutable_unchecked<2>();
        ssize_t idx = 0;
        for (ssize_t i = 0; i < shape[0]; ++i) {
            for (ssize_t j = 0; j < shape[1]; ++j) {
                view(i, j) = values[idx++];
            }
        }
        return out;
    }
    throw std::runtime_error("Unsupported output rank for IQN-ILS reshape.");
}


std::vector<vector_type> recent_history_vectors(
    py::handle history_obj,
    int keep_count,
    Eigen::Index expected_size,
    const char* label
)
{
    py::list history = py::reinterpret_borrow<py::list>(history_obj);
    const auto count = static_cast<int>(history.size());
    const int keep = std::min(std::max(keep_count, 0), count);
    std::vector<vector_type> seq;
    seq.reserve(static_cast<std::size_t>(keep));

    for (int i = count - keep; i < count; ++i) {
        FlatArray entry = flatten_array(history[static_cast<ssize_t>(i)], label);
        if (entry.values.size() != expected_size) {
            throw std::runtime_error(std::string(label) + " entry size does not match x_curr.");
        }
        seq.push_back(std::move(entry.values));
    }
    return seq;
}


matrix_type hstack_blocks(py::handle blocks_obj, Eigen::Index expected_rows, const char* label)
{
    py::list blocks = py::reinterpret_borrow<py::list>(blocks_obj);
    Eigen::Index total_cols = 0;
    std::vector<matrix_type> mats;
    mats.reserve(static_cast<std::size_t>(blocks.size()));

    for (py::handle item : blocks) {
        auto arr = py::array_t<scalar_type, py::array::c_style | py::array::forcecast>::ensure(item);
        if (!arr) {
            throw std::runtime_error(std::string(label) + " blocks must be float64 arrays.");
        }
        auto info = arr.request();
        if (info.ndim != 2) {
            throw std::runtime_error(std::string(label) + " blocks must be rank-2 arrays.");
        }
        if (static_cast<Eigen::Index>(info.shape[0]) != expected_rows) {
            throw std::runtime_error(std::string(label) + " block row count does not match x_curr.");
        }
        const Eigen::Index rows = static_cast<Eigen::Index>(info.shape[0]);
        const Eigen::Index cols = static_cast<Eigen::Index>(info.shape[1]);
        if (cols <= 0) {
            continue;
        }
        matrix_type mat(rows, cols);
        auto* data = static_cast<const scalar_type*>(info.ptr);
        Eigen::Index idx = 0;
        for (Eigen::Index r = 0; r < rows; ++r) {
            for (Eigen::Index c = 0; c < cols; ++c) {
                mat(r, c) = data[idx++];
            }
        }
        total_cols += cols;
        mats.push_back(std::move(mat));
    }

    if (total_cols <= 0) {
        return matrix_type(expected_rows, 0);
    }

    matrix_type out(expected_rows, total_cols);
    Eigen::Index offset = 0;
    for (const auto& mat : mats) {
        out.middleCols(offset, mat.cols()) = mat;
        offset += mat.cols();
    }
    return out;
}


matrix_type build_difference_matrix(const std::vector<vector_type>& newest_first)
{
    const auto cols = static_cast<Eigen::Index>(newest_first.size()) - 1;
    const auto rows = newest_first.empty() ? Eigen::Index{0} : newest_first.front().size();
    if (cols <= 0 || rows <= 0) {
        return matrix_type(rows, 0);
    }
    matrix_type out(rows, cols);
    for (Eigen::Index i = 0; i < cols; ++i) {
        out.col(i) = newest_first[static_cast<std::size_t>(i)] - newest_first[static_cast<std::size_t>(i + 1)];
    }
    return out;
}


vector_type solve_least_squares(const matrix_type& matrix, const vector_type& rhs, double regularization)
{
    if (matrix.cols() <= 0) {
        return vector_type();
    }

    if (regularization > 0.0) {
        const Eigen::Index ncols = matrix.cols();
        matrix_type augmented(matrix.rows() + ncols, ncols);
        augmented.topRows(matrix.rows()) = matrix;
        augmented.bottomRows(ncols).setZero();
        augmented.bottomRows(ncols).diagonal().array() = std::sqrt(regularization);

        vector_type rhs_aug(rhs.size() + ncols);
        rhs_aug.head(rhs.size()) = rhs;
        rhs_aug.tail(ncols).setZero();

        Eigen::JacobiSVD<matrix_type> svd(augmented, Eigen::ComputeThinU | Eigen::ComputeThinV);
        return svd.solve(rhs_aug);
    }

    Eigen::JacobiSVD<matrix_type> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    return svd.solve(rhs);
}


py::array_t<scalar_type> next_iterate(
    py::handle x_curr_obj,
    py::handle g_curr_obj,
    py::handle x_history_obj,
    py::handle g_history_obj,
    py::handle dr_old_mats_obj,
    py::handle dg_old_mats_obj,
    double alpha,
    int horizon,
    double regularization
)
{
    const FlatArray x_curr = flatten_array(x_curr_obj, "x_curr");
    const FlatArray g_curr = flatten_array(g_curr_obj, "g_curr");
    if (x_curr.values.size() != g_curr.values.size()) {
        throw std::runtime_error("x_curr and g_curr must have the same flattened size.");
    }

    const double alpha_value = std::clamp(alpha, 0.0, 1.0);
    const vector_type r_curr = g_curr.values - x_curr.values;
    const vector_type picard = x_curr.values + alpha_value * r_curr;

    py::list x_history = py::reinterpret_borrow<py::list>(x_history_obj);
    py::list g_history = py::reinterpret_borrow<py::list>(g_history_obj);
    const int keep_count = std::min(std::max(horizon, 1), std::min(static_cast<int>(x_history.size()), static_cast<int>(g_history.size())));
    if (keep_count <= 0) {
        return reshape_like(picard, x_curr.shape);
    }

    auto x_seq = recent_history_vectors(x_history, keep_count, x_curr.values.size(), "x_history");
    auto g_seq = recent_history_vectors(g_history, keep_count, x_curr.values.size(), "g_history");
    std::vector<vector_type> r_recent;
    std::vector<vector_type> g_recent;
    r_recent.reserve(static_cast<std::size_t>(keep_count));
    g_recent.reserve(static_cast<std::size_t>(keep_count));
    for (int i = keep_count - 1; i >= 0; --i) {
        r_recent.push_back(g_seq[static_cast<std::size_t>(i)] - x_seq[static_cast<std::size_t>(i)]);
        g_recent.push_back(g_seq[static_cast<std::size_t>(i)]);
    }

    const Eigen::Index k = static_cast<Eigen::Index>(r_recent.size()) - 1;
    const matrix_type v_old = dr_old_mats_obj.is_none()
        ? matrix_type(x_curr.values.size(), 0)
        : hstack_blocks(dr_old_mats_obj, x_curr.values.size(), "dr_old_mats");
    const matrix_type w_old = dg_old_mats_obj.is_none()
        ? matrix_type(x_curr.values.size(), 0)
        : hstack_blocks(dg_old_mats_obj, x_curr.values.size(), "dg_old_mats");
    const bool has_old = (v_old.cols() > 0) && (w_old.cols() > 0);

    if (!has_old && k == 0) {
        return reshape_like(picard, x_curr.shape);
    }

    const matrix_type v_new = build_difference_matrix(r_recent);
    const matrix_type w_new = build_difference_matrix(g_recent);

    matrix_type v;
    matrix_type w;
    if (has_old) {
        if (k > 0) {
            v.resize(x_curr.values.size(), v_new.cols() + v_old.cols());
            w.resize(x_curr.values.size(), w_new.cols() + w_old.cols());
            v << v_new, v_old;
            w << w_new, w_old;
        } else {
            v = v_old;
            w = w_old;
        }
    } else {
        v = v_new;
        w = w_new;
    }

    if (v.cols() <= 0 || w.cols() <= 0) {
        return reshape_like(picard, x_curr.shape);
    }

    const vector_type delta_r = -r_recent.front();
    vector_type gamma;
    try {
        gamma = solve_least_squares(v, delta_r, std::max(regularization, 0.0));
    } catch (const std::exception&) {
        return reshape_like(picard, x_curr.shape);
    }
    if (gamma.size() == 0) {
        return reshape_like(picard, x_curr.shape);
    }

    const vector_type delta_x = w * gamma - delta_r;
    if (!delta_x.allFinite()) {
        return reshape_like(picard, x_curr.shape);
    }
    return reshape_like(x_curr.values + delta_x, x_curr.shape);
}


py::tuple iteration_matrices(
    py::handle x_history_obj,
    py::handle g_history_obj,
    int iteration_horizon
)
{
    py::list x_history = py::reinterpret_borrow<py::list>(x_history_obj);
    py::list g_history = py::reinterpret_borrow<py::list>(g_history_obj);
    const int keep_count = std::min(std::max(iteration_horizon, 1), std::min(static_cast<int>(x_history.size()), static_cast<int>(g_history.size())));
    if (keep_count <= 1) {
        return py::make_tuple(py::none(), py::none());
    }

    const FlatArray x_last = flatten_array(x_history[static_cast<ssize_t>(x_history.size() - 1)], "x_history");
    auto x_seq = recent_history_vectors(x_history, keep_count, x_last.values.size(), "x_history");
    auto g_seq = recent_history_vectors(g_history, keep_count, x_last.values.size(), "g_history");

    std::vector<vector_type> r_recent;
    std::vector<vector_type> g_recent;
    r_recent.reserve(static_cast<std::size_t>(keep_count));
    g_recent.reserve(static_cast<std::size_t>(keep_count));
    for (int i = keep_count - 1; i >= 0; --i) {
        r_recent.push_back(g_seq[static_cast<std::size_t>(i)] - x_seq[static_cast<std::size_t>(i)]);
        g_recent.push_back(g_seq[static_cast<std::size_t>(i)]);
    }

    const matrix_type v_new = build_difference_matrix(r_recent);
    const matrix_type w_new = build_difference_matrix(g_recent);
    py::array_t<scalar_type> v_out({v_new.rows(), v_new.cols()});
    py::array_t<scalar_type> w_out({w_new.rows(), w_new.cols()});
    auto v_view = v_out.mutable_unchecked<2>();
    auto w_view = w_out.mutable_unchecked<2>();
    for (Eigen::Index i = 0; i < v_new.rows(); ++i) {
        for (Eigen::Index j = 0; j < v_new.cols(); ++j) {
            v_view(i, j) = v_new(i, j);
            w_view(i, j) = w_new(i, j);
        }
    }
    return py::make_tuple(v_out, w_out);
}

}  // namespace

PYBIND11_MODULE(_pycutfem_cpp_iqnils_2026_04_22_iqnils_v1, m)
{
    m.doc() = "pycutfem Kratos-style IQN-ILS coupling accelerator kernel";
    m.def(
        "next_iterate",
        &next_iterate,
        py::arg("x_curr"),
        py::arg("g_curr"),
        py::arg("x_history"),
        py::arg("g_history"),
        py::arg("dr_old_mats") = py::none(),
        py::arg("dg_old_mats") = py::none(),
        py::arg("alpha") = 0.5,
        py::arg("horizon") = 1,
        py::arg("regularization") = 0.0
    );
    m.def(
        "iteration_matrices",
        &iteration_matrices,
        py::arg("x_history"),
        py::arg("g_history"),
        py::arg("iteration_horizon")
    );
}
