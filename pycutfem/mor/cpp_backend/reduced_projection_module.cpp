#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

using scalar_type = double;
using index_type = std::int64_t;
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

py::array_t<index_type, py::array::c_style | py::array::forcecast>
as_index_array(py::handle obj, const char* label)
{
    auto arr = py::array_t<index_type, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr) {
        throw py::value_error(std::string(label) + " must be convertible to a contiguous int64 array.");
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

std::vector<index_type> as_index_vector(py::handle obj, const char* label)
{
    auto arr = as_index_array(obj, label);
    if (arr.ndim() != 1) {
        throw py::value_error(std::string(label) + " must be a rank-1 array.");
    }
    std::vector<index_type> out(static_cast<std::size_t>(arr.shape(0)));
    auto view = arr.unchecked<1>();
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = view(i);
    }
    return out;
}

vector_type optional_element_weights(py::handle obj, Eigen::Index n_elements, const char* context)
{
    vector_type weights(n_elements);
    weights.setOnes();
    if (obj.is_none()) {
        return weights;
    }
    vector_type raw = as_vector(obj, "element_weights");
    if (raw.size() != n_elements) {
        throw py::value_error(std::string(context) + " element_weights size must match the sampled element count.");
    }
    for (Eigen::Index i = 0; i < raw.size(); ++i) {
        if (raw(i) < 0.0) {
            throw py::value_error(std::string(context) + " element_weights must be nonnegative.");
        }
        weights(i) = raw(i);
    }
    return weights;
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

int thread_count()
{
#ifdef _OPENMP
    return std::max(1, omp_get_max_threads());
#else
    return 1;
#endif
}

int thread_id()
{
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

void validate_unique_nonnegative(const std::vector<index_type>& values, const char* label)
{
    std::unordered_map<index_type, bool> seen;
    seen.reserve(values.size());
    for (const index_type value : values) {
        if (value < 0) {
            throw py::value_error(std::string(label) + " contains negative entries.");
        }
        if (seen.find(value) != seen.end()) {
            throw py::value_error(std::string(label) + " must be unique.");
        }
        seen[value] = true;
    }
}

py::array_t<scalar_type> decode_values_on_dofs(
    py::handle offset_obj,
    py::handle basis_obj,
    py::handle dofs_obj,
    py::handle coefficients_obj
)
{
    const vector_type offset = as_vector(offset_obj, "offset");
    const matrix_type basis = as_matrix(basis_obj, "basis");
    const vector_type coefficients = as_vector(coefficients_obj, "coefficients");
    const std::vector<index_type> dofs = as_index_vector(dofs_obj, "dofs");
    if (basis.rows() != offset.size()) {
        throw py::value_error("basis and offset have incompatible row counts.");
    }
    if (basis.cols() != coefficients.size()) {
        throw py::value_error("basis column count must match coefficients size.");
    }
    for (const index_type gdof : dofs) {
        if (gdof < 0 || gdof >= basis.rows()) {
            throw py::value_error("sampled reduced dofs contain out-of-range entries.");
        }
    }

    py::array_t<scalar_type> out(static_cast<ssize_t>(dofs.size()));
    auto values = out.mutable_unchecked<1>();
#pragma omp parallel for if(dofs.size() > 256)
    for (ssize_t k = 0; k < static_cast<ssize_t>(dofs.size()); ++k) {
        const index_type gdof = dofs[static_cast<std::size_t>(k)];
        double value = offset(gdof);
        for (Eigen::Index m = 0; m < basis.cols(); ++m) {
            value += basis(gdof, m) * coefficients(m);
        }
        values(k) = value;
    }
    return out;
}

py::array_t<scalar_type> decode_element_values(
    py::handle offset_obj,
    py::handle basis_obj,
    py::handle local_map_obj,
    py::handle coefficients_obj
)
{
    const vector_type offset = as_vector(offset_obj, "offset");
    const matrix_type basis = as_matrix(basis_obj, "basis");
    const vector_type coefficients = as_vector(coefficients_obj, "coefficients");
    auto local_map = as_index_array(local_map_obj, "local_map");
    if (local_map.ndim() != 2) {
        throw py::value_error("local_map must have shape (n_elements, n_local_dofs).");
    }
    if (basis.rows() != offset.size()) {
        throw py::value_error("basis and offset have incompatible row counts.");
    }
    if (basis.cols() != coefficients.size()) {
        throw py::value_error("basis column count must match coefficients size.");
    }
    auto map = local_map.unchecked<2>();
    for (ssize_t e = 0; e < map.shape(0); ++e) {
        for (ssize_t i = 0; i < map.shape(1); ++i) {
            const index_type gdof = map(e, i);
            if (gdof < 0 || gdof >= basis.rows()) {
                throw py::value_error("local_map contains out-of-range entries.");
            }
        }
    }

    py::array_t<scalar_type> out({local_map.shape(0), local_map.shape(1)});
    auto values = out.mutable_unchecked<2>();
#pragma omp parallel for collapse(2) if(local_map.shape(0) * local_map.shape(1) > 256)
    for (ssize_t e = 0; e < map.shape(0); ++e) {
        for (ssize_t i = 0; i < map.shape(1); ++i) {
            const index_type gdof = map(e, i);
            double value = offset(gdof);
            for (Eigen::Index m = 0; m < basis.cols(); ++m) {
                value += basis(gdof, m) * coefficients(m);
            }
            values(e, i) = value;
        }
    }
    return out;
}

void validate_local_blocks(
    const py::array_t<scalar_type, py::array::c_style | py::array::forcecast>& K_elem,
    const py::array_t<scalar_type, py::array::c_style | py::array::forcecast>& vector_elem,
    const py::array_t<index_type, py::array::c_style | py::array::forcecast>& gdofs_map,
    const matrix_type& basis,
    const char* context
)
{
    if (K_elem.ndim() != 3 || vector_elem.ndim() != 2 || gdofs_map.ndim() != 2) {
        throw py::value_error(std::string(context) + " local blocks must have shapes K=(e,l,l), vector=(e,l), gdofs=(e,l).");
    }
    if (
        K_elem.shape(0) != vector_elem.shape(0) ||
        K_elem.shape(0) != gdofs_map.shape(0) ||
        K_elem.shape(1) != K_elem.shape(2) ||
        K_elem.shape(1) != vector_elem.shape(1) ||
        K_elem.shape(1) != gdofs_map.shape(1)
    ) {
        throw py::value_error(std::string(context) + " local block dimensions are inconsistent.");
    }
    auto K = K_elem.unchecked<3>();
    auto vec = vector_elem.unchecked<2>();
    auto gdofs = gdofs_map.unchecked<2>();
    for (ssize_t e = 0; e < K.shape(0); ++e) {
        for (ssize_t i = 0; i < K.shape(1); ++i) {
            if (!std::isfinite(vec(e, i))) {
                throw py::value_error(std::string(context) + " local vector blocks must be finite.");
            }
            const index_type gi = gdofs(e, i);
            if (gi < 0 || gi >= basis.rows()) {
                throw py::value_error(std::string(context) + " gdofs_map contains dofs outside trial_basis.");
            }
            for (ssize_t j = 0; j < K.shape(2); ++j) {
                if (!std::isfinite(K(e, i, j))) {
                    throw py::value_error(std::string(context) + " K_elem must be finite.");
                }
            }
        }
    }
}

py::tuple sampled_lspg_element_contributions_from_local_blocks(
    py::handle K_elem_obj,
    py::handle raw_rhs_elem_obj,
    py::handle gdofs_map_obj,
    py::handle row_dofs_obj,
    py::handle trial_basis_obj
)
{
    auto K_arr = as_double_array(K_elem_obj, "K_elem");
    auto rhs_arr = as_double_array(raw_rhs_elem_obj, "raw_rhs_elem");
    auto gdofs_arr = as_index_array(gdofs_map_obj, "gdofs_map");
    const matrix_type basis = as_matrix(trial_basis_obj, "trial_basis");
    const std::vector<index_type> rows = as_index_vector(row_dofs_obj, "row_dofs");
    validate_unique_nonnegative(rows, "sampled LSPG row_dofs");
    validate_local_blocks(K_arr, rhs_arr, gdofs_arr, basis, "sampled LSPG");
    for (const index_type row : rows) {
        if (row >= basis.rows()) {
            throw py::value_error("sampled LSPG row_dofs contain out-of-range entries.");
        }
    }

    std::vector<index_type> row_positions(static_cast<std::size_t>(basis.rows()), -1);
    for (std::size_t s = 0; s < rows.size(); ++s) {
        row_positions[static_cast<std::size_t>(rows[s])] = static_cast<index_type>(s);
    }

    auto K = K_arr.unchecked<3>();
    auto raw_rhs = rhs_arr.unchecked<2>();
    auto gdofs = gdofs_arr.unchecked<2>();
    const ssize_t n_elements = K.shape(0);
    const ssize_t local_size = K.shape(1);
    const ssize_t n_rows = static_cast<ssize_t>(rows.size());
    const ssize_t n_modes = basis.cols();
    py::array_t<scalar_type> residual_out({n_elements, n_rows});
    py::array_t<scalar_type> trial_out({n_elements, n_rows, n_modes});
    auto residual = residual_out.mutable_unchecked<2>();
    auto trial = trial_out.mutable_unchecked<3>();
    for (ssize_t e = 0; e < n_elements; ++e) {
        for (ssize_t s = 0; s < n_rows; ++s) {
            residual(e, s) = 0.0;
            for (ssize_t m = 0; m < n_modes; ++m) {
                trial(e, s, m) = 0.0;
            }
        }
    }

#pragma omp parallel for if(n_elements * local_size * std::max<ssize_t>(n_modes, 1) > 1024)
    for (ssize_t e = 0; e < n_elements; ++e) {
        for (ssize_t i = 0; i < local_size; ++i) {
            const index_type sample = row_positions[static_cast<std::size_t>(gdofs(e, i))];
            if (sample < 0) {
                continue;
            }
            residual(e, sample) -= raw_rhs(e, i);
            for (ssize_t j = 0; j < local_size; ++j) {
                const index_type gj = gdofs(e, j);
                const double kij = K(e, i, j);
                for (ssize_t m = 0; m < n_modes; ++m) {
                    trial(e, sample, m) += kij * basis(gj, m);
                }
            }
        }
    }

    return py::make_tuple(residual_out, trial_out);
}

py::tuple sampled_lspg_rows_from_local_blocks(
    py::handle K_elem_obj,
    py::handle raw_rhs_elem_obj,
    py::handle gdofs_map_obj,
    py::handle row_dofs_obj,
    py::handle trial_basis_obj,
    py::handle element_weights_obj
)
{
    auto K_arr = as_double_array(K_elem_obj, "K_elem");
    auto rhs_arr = as_double_array(raw_rhs_elem_obj, "raw_rhs_elem");
    auto gdofs_arr = as_index_array(gdofs_map_obj, "gdofs_map");
    const matrix_type basis = as_matrix(trial_basis_obj, "trial_basis");
    const std::vector<index_type> rows = as_index_vector(row_dofs_obj, "row_dofs");
    validate_unique_nonnegative(rows, "sampled LSPG row_dofs");
    validate_local_blocks(K_arr, rhs_arr, gdofs_arr, basis, "sampled LSPG");
    for (const index_type row : rows) {
        if (row >= basis.rows()) {
            throw py::value_error("sampled LSPG row_dofs contain out-of-range entries.");
        }
    }
    const vector_type weights = optional_element_weights(element_weights_obj, K_arr.shape(0), "sampled LSPG");

    std::vector<index_type> row_positions(static_cast<std::size_t>(basis.rows()), -1);
    for (std::size_t s = 0; s < rows.size(); ++s) {
        row_positions[static_cast<std::size_t>(rows[s])] = static_cast<index_type>(s);
    }

    auto K = K_arr.unchecked<3>();
    auto raw_rhs = rhs_arr.unchecked<2>();
    auto gdofs = gdofs_arr.unchecked<2>();
    const Eigen::Index n_elements = K.shape(0);
    const Eigen::Index local_size = K.shape(1);
    const Eigen::Index n_rows = static_cast<Eigen::Index>(rows.size());
    const Eigen::Index n_modes = basis.cols();
    const int n_threads = thread_count();
    std::vector<vector_type> residual_tls(static_cast<std::size_t>(n_threads), vector_type::Zero(n_rows));
    std::vector<matrix_type> trial_tls(static_cast<std::size_t>(n_threads), matrix_type::Zero(n_rows, n_modes));

#pragma omp parallel for if(n_elements * local_size * std::max<Eigen::Index>(n_modes, 1) > 1024)
    for (Eigen::Index e = 0; e < n_elements; ++e) {
        const int tid = thread_id();
        vector_type& residual = residual_tls[static_cast<std::size_t>(tid)];
        matrix_type& trial = trial_tls[static_cast<std::size_t>(tid)];
        const double weight = weights(e);
        if (weight == 0.0) {
            continue;
        }
        for (Eigen::Index i = 0; i < local_size; ++i) {
            const index_type sample = row_positions[static_cast<std::size_t>(gdofs(e, i))];
            if (sample < 0) {
                continue;
            }
            residual(sample) -= weight * raw_rhs(e, i);
            for (Eigen::Index j = 0; j < local_size; ++j) {
                const index_type gj = gdofs(e, j);
                const double scaled_kij = weight * K(e, i, j);
                for (Eigen::Index m = 0; m < n_modes; ++m) {
                    trial(sample, m) += scaled_kij * basis(gj, m);
                }
            }
        }
    }

    vector_type residual = vector_type::Zero(n_rows);
    matrix_type trial = matrix_type::Zero(n_rows, n_modes);
    for (int tid = 0; tid < n_threads; ++tid) {
        residual += residual_tls[static_cast<std::size_t>(tid)];
        trial += trial_tls[static_cast<std::size_t>(tid)];
    }
    return py::make_tuple(vector_to_array(residual), matrix_to_array(trial));
}

py::tuple sampled_galerkin_element_contributions_from_local_blocks(
    py::handle K_elem_obj,
    py::handle residual_elem_obj,
    py::handle gdofs_map_obj,
    py::handle trial_basis_obj
)
{
    auto K_arr = as_double_array(K_elem_obj, "K_elem");
    auto residual_arr = as_double_array(residual_elem_obj, "residual_elem");
    auto gdofs_arr = as_index_array(gdofs_map_obj, "gdofs_map");
    const matrix_type basis = as_matrix(trial_basis_obj, "trial_basis");
    validate_local_blocks(K_arr, residual_arr, gdofs_arr, basis, "sampled Galerkin");

    auto K = K_arr.unchecked<3>();
    auto residual_elem = residual_arr.unchecked<2>();
    auto gdofs = gdofs_arr.unchecked<2>();
    const ssize_t n_elements = K.shape(0);
    const ssize_t local_size = K.shape(1);
    const ssize_t n_modes = basis.cols();
    py::array_t<scalar_type> residual_out({n_elements, n_modes});
    py::array_t<scalar_type> tangent_out({n_elements, n_modes, n_modes});
    auto residual = residual_out.mutable_unchecked<2>();
    auto tangent = tangent_out.mutable_unchecked<3>();
    for (ssize_t e = 0; e < n_elements; ++e) {
        for (ssize_t m = 0; m < n_modes; ++m) {
            residual(e, m) = 0.0;
            for (ssize_t n = 0; n < n_modes; ++n) {
                tangent(e, m, n) = 0.0;
            }
        }
    }

#pragma omp parallel for if(n_elements * local_size * std::max<ssize_t>(n_modes, 1) > 1024)
    for (ssize_t e = 0; e < n_elements; ++e) {
        for (ssize_t i = 0; i < local_size; ++i) {
            const index_type gi = gdofs(e, i);
            const double ri = residual_elem(e, i);
            for (ssize_t m = 0; m < n_modes; ++m) {
                residual(e, m) += basis(gi, m) * ri;
            }
            for (ssize_t j = 0; j < local_size; ++j) {
                const index_type gj = gdofs(e, j);
                const double kij = K(e, i, j);
                for (ssize_t m = 0; m < n_modes; ++m) {
                    const double left = basis(gi, m) * kij;
                    for (ssize_t n = 0; n < n_modes; ++n) {
                        tangent(e, m, n) += left * basis(gj, n);
                    }
                }
            }
        }
    }

    return py::make_tuple(residual_out, tangent_out);
}

py::tuple sampled_galerkin_reduced_system_from_local_blocks(
    py::handle K_elem_obj,
    py::handle residual_elem_obj,
    py::handle gdofs_map_obj,
    py::handle trial_basis_obj,
    py::handle element_weights_obj
)
{
    auto K_arr = as_double_array(K_elem_obj, "K_elem");
    auto residual_arr = as_double_array(residual_elem_obj, "residual_elem");
    auto gdofs_arr = as_index_array(gdofs_map_obj, "gdofs_map");
    const matrix_type basis = as_matrix(trial_basis_obj, "trial_basis");
    validate_local_blocks(K_arr, residual_arr, gdofs_arr, basis, "sampled Galerkin");
    const vector_type weights = optional_element_weights(element_weights_obj, K_arr.shape(0), "sampled Galerkin");

    auto K = K_arr.unchecked<3>();
    auto residual_elem = residual_arr.unchecked<2>();
    auto gdofs = gdofs_arr.unchecked<2>();
    const Eigen::Index n_elements = K.shape(0);
    const Eigen::Index local_size = K.shape(1);
    const Eigen::Index n_modes = basis.cols();
    const int n_threads = thread_count();
    std::vector<vector_type> residual_tls(static_cast<std::size_t>(n_threads), vector_type::Zero(n_modes));
    std::vector<matrix_type> tangent_tls(static_cast<std::size_t>(n_threads), matrix_type::Zero(n_modes, n_modes));

#pragma omp parallel for if(n_elements * local_size * std::max<Eigen::Index>(n_modes, 1) > 1024)
    for (Eigen::Index e = 0; e < n_elements; ++e) {
        const int tid = thread_id();
        vector_type& residual = residual_tls[static_cast<std::size_t>(tid)];
        matrix_type& tangent = tangent_tls[static_cast<std::size_t>(tid)];
        const double weight = weights(e);
        if (weight == 0.0) {
            continue;
        }
        for (Eigen::Index i = 0; i < local_size; ++i) {
            const index_type gi = gdofs(e, i);
            const double ri = weight * residual_elem(e, i);
            for (Eigen::Index m = 0; m < n_modes; ++m) {
                residual(m) += basis(gi, m) * ri;
            }
            for (Eigen::Index j = 0; j < local_size; ++j) {
                const index_type gj = gdofs(e, j);
                const double scaled_kij = weight * K(e, i, j);
                for (Eigen::Index m = 0; m < n_modes; ++m) {
                    const double left = basis(gi, m) * scaled_kij;
                    for (Eigen::Index n = 0; n < n_modes; ++n) {
                        tangent(m, n) += left * basis(gj, n);
                    }
                }
            }
        }
    }

    vector_type residual = vector_type::Zero(n_modes);
    matrix_type tangent = matrix_type::Zero(n_modes, n_modes);
    for (int tid = 0; tid < n_threads; ++tid) {
        residual += residual_tls[static_cast<std::size_t>(tid)];
        tangent += tangent_tls[static_cast<std::size_t>(tid)];
    }
    return py::make_tuple(vector_to_array(residual), matrix_to_array(tangent));
}

py::tuple constrained_reaction_rows_from_local_blocks(
    py::handle raw_rhs_elem_obj,
    py::handle gdofs_map_obj,
    py::handle constrained_row_dofs_obj,
    py::handle element_weights_obj
)
{
    auto rhs_arr = as_double_array(raw_rhs_elem_obj, "raw_rhs_elem");
    auto gdofs_arr = as_index_array(gdofs_map_obj, "gdofs_map");
    if (rhs_arr.ndim() != 2 || gdofs_arr.ndim() != 2 || rhs_arr.shape(0) != gdofs_arr.shape(0) || rhs_arr.shape(1) != gdofs_arr.shape(1)) {
        throw py::value_error("reaction local blocks must have shapes raw_rhs=(e,l), gdofs=(e,l).");
    }
    const std::vector<index_type> rows = as_index_vector(constrained_row_dofs_obj, "constrained_row_dofs");
    validate_unique_nonnegative(rows, "constrained_row_dofs");
    const vector_type weights = optional_element_weights(element_weights_obj, rhs_arr.shape(0), "constrained reaction");
    std::unordered_map<index_type, index_type> row_positions;
    row_positions.reserve(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        row_positions[rows[i]] = static_cast<index_type>(i);
    }

    auto raw_rhs = rhs_arr.unchecked<2>();
    auto gdofs = gdofs_arr.unchecked<2>();
    const Eigen::Index n_elements = rhs_arr.shape(0);
    const Eigen::Index local_size = rhs_arr.shape(1);
    const Eigen::Index n_rows = static_cast<Eigen::Index>(rows.size());
    for (Eigen::Index e = 0; e < n_elements; ++e) {
        for (Eigen::Index i = 0; i < local_size; ++i) {
            if (!std::isfinite(raw_rhs(e, i))) {
                throw py::value_error("reaction raw_rhs_elem must be finite.");
            }
            if (gdofs(e, i) < 0) {
                throw py::value_error("reaction gdofs_map contains negative dofs.");
            }
        }
    }
    const int n_threads = thread_count();
    std::vector<vector_type> values_tls(static_cast<std::size_t>(n_threads), vector_type::Zero(n_rows));

#pragma omp parallel for if(n_elements * local_size > 1024)
    for (Eigen::Index e = 0; e < n_elements; ++e) {
        const int tid = thread_id();
        vector_type& values = values_tls[static_cast<std::size_t>(tid)];
        const double weight = weights(e);
        for (Eigen::Index i = 0; i < local_size; ++i) {
            const index_type gdof = gdofs(e, i);
            const auto it = row_positions.find(gdof);
            if (it != row_positions.end()) {
                values(it->second) -= weight * raw_rhs(e, i);
            }
        }
    }

    vector_type values = vector_type::Zero(n_rows);
    for (int tid = 0; tid < n_threads; ++tid) {
        values += values_tls[static_cast<std::size_t>(tid)];
    }

    py::array_t<index_type> rows_out(static_cast<ssize_t>(rows.size()));
    auto row_view = rows_out.mutable_unchecked<1>();
    for (ssize_t i = 0; i < row_view.shape(0); ++i) {
        row_view(i) = rows[static_cast<std::size_t>(i)];
    }
    return py::make_tuple(rows_out, vector_to_array(values));
}

py::array_t<scalar_type> reduced_reaction_from_local_blocks(
    py::handle raw_rhs_elem_obj,
    py::handle gdofs_map_obj,
    py::handle constrained_row_dofs_obj,
    py::handle row_to_reduced_load_obj,
    py::handle element_weights_obj
)
{
    const py::tuple rows_and_values = constrained_reaction_rows_from_local_blocks(
        raw_rhs_elem_obj,
        gdofs_map_obj,
        constrained_row_dofs_obj,
        element_weights_obj
    );
    const vector_type values = as_vector(rows_and_values[1], "reaction_rows");
    const matrix_type transfer = as_matrix(row_to_reduced_load_obj, "row_to_reduced_load");
    if (transfer.cols() != values.size()) {
        throw py::value_error("row_to_reduced_load columns must match constrained reaction rows.");
    }
    return vector_to_array(transfer * values);
}

py::tuple apply_gnat_lift(
    py::handle sample_to_residual_coefficients_obj,
    py::handle sampled_residual_obj,
    py::handle sampled_trial_jacobian_obj
)
{
    const matrix_type lift = as_matrix(sample_to_residual_coefficients_obj, "sample_to_residual_coefficients");
    const vector_type sampled_residual = as_vector(sampled_residual_obj, "sampled_residual");
    const matrix_type sampled_trial = as_matrix(sampled_trial_jacobian_obj, "sampled_trial_jacobian");
    if (lift.cols() != sampled_residual.size() || lift.cols() != sampled_trial.rows()) {
        throw py::value_error("GNAT lift columns must match sampled residual/Jacobian rows.");
    }
    return py::make_tuple(
        vector_to_array(lift * sampled_residual),
        matrix_to_array(lift * sampled_trial)
    );
}

}  // namespace

PYBIND11_MODULE(_pycutfem_mor_reduced_projection_2026_05_15_mor_reduced_projection_v1, m)
{
    m.doc() = "pycutfem MOR reduced projection backend";
    m.def(
        "decode_values_on_dofs",
        &decode_values_on_dofs,
        py::arg("offset"),
        py::arg("basis"),
        py::arg("dofs"),
        py::arg("coefficients")
    );
    m.def(
        "decode_element_values",
        &decode_element_values,
        py::arg("offset"),
        py::arg("basis"),
        py::arg("local_map"),
        py::arg("coefficients")
    );
    m.def(
        "sampled_lspg_element_contributions_from_local_blocks",
        &sampled_lspg_element_contributions_from_local_blocks,
        py::arg("K_elem"),
        py::arg("raw_rhs_elem"),
        py::arg("gdofs_map"),
        py::arg("row_dofs"),
        py::arg("trial_basis")
    );
    m.def(
        "sampled_lspg_rows_from_local_blocks",
        &sampled_lspg_rows_from_local_blocks,
        py::arg("K_elem"),
        py::arg("raw_rhs_elem"),
        py::arg("gdofs_map"),
        py::arg("row_dofs"),
        py::arg("trial_basis"),
        py::arg("element_weights") = py::none()
    );
    m.def(
        "sampled_galerkin_element_contributions_from_local_blocks",
        &sampled_galerkin_element_contributions_from_local_blocks,
        py::arg("K_elem"),
        py::arg("residual_elem"),
        py::arg("gdofs_map"),
        py::arg("trial_basis")
    );
    m.def(
        "sampled_galerkin_reduced_system_from_local_blocks",
        &sampled_galerkin_reduced_system_from_local_blocks,
        py::arg("K_elem"),
        py::arg("residual_elem"),
        py::arg("gdofs_map"),
        py::arg("trial_basis"),
        py::arg("element_weights") = py::none()
    );
    m.def(
        "constrained_reaction_rows_from_local_blocks",
        &constrained_reaction_rows_from_local_blocks,
        py::arg("raw_rhs_elem"),
        py::arg("gdofs_map"),
        py::arg("constrained_row_dofs"),
        py::arg("element_weights") = py::none()
    );
    m.def(
        "reduced_reaction_from_local_blocks",
        &reduced_reaction_from_local_blocks,
        py::arg("raw_rhs_elem"),
        py::arg("gdofs_map"),
        py::arg("constrained_row_dofs"),
        py::arg("row_to_reduced_load"),
        py::arg("element_weights") = py::none()
    );
    m.def(
        "apply_gnat_lift",
        &apply_gnat_lift,
        py::arg("sample_to_residual_coefficients"),
        py::arg("sampled_residual"),
        py::arg("sampled_trial_jacobian")
    );
}
