#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <Python.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "native_kernel.hpp"

namespace py = pybind11;
using namespace pycutfem::cpp_backend;

namespace {

using scalar_type = double;
using matrix_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>;
using vector_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>;
using clock_type = std::chrono::steady_clock;

double elapsed_seconds(clock_type::time_point start, clock_type::time_point end)
{
    return std::chrono::duration<double>(end - start).count();
}

NativeKernelMetadata* metadata_from_capsule(py::handle capsule)
{
    auto* raw = PyCapsule_GetPointer(capsule.ptr(), NATIVE_KERNEL_METADATA_CAPSULE_NAME);
    if (raw == nullptr) {
        throw py::value_error("Invalid native kernel metadata capsule.");
    }
    auto* metadata = static_cast<NativeKernelMetadata*>(raw);
    if (metadata->entrypoint == nullptr) {
        throw py::value_error("Native kernel metadata does not expose a raw entrypoint.");
    }
    return metadata;
}

bool is_int32_arg(const std::string& name)
{
    return name == "gdofs_map"
        || name == "owner_id"
        || name == "owner_pos_id"
        || name == "owner_neg_id"
        || name == "pos_eids"
        || name == "neg_eids"
        || name == "qstate_owner_id"
        || name.rfind("pos_map", 0) == 0
        || name.rfind("neg_map", 0) == 0;
}

bool is_uint8_arg(const std::string& name)
{
    return name.rfind("domain_flag_", 0) == 0;
}

template <typename T>
py::array_t<T, py::array::c_style | py::array::forcecast> as_array_for_arg(py::handle obj, const std::string& name)
{
    auto arr = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr) {
        throw py::value_error("Could not convert native kernel argument '" + name + "' to the required array type.");
    }
    if (arr.ndim() > 6) {
        throw py::value_error("Native kernel argument '" + name + "' has rank > 6.");
    }
    return arr;
}

template <typename T>
void push_arg(
    std::vector<py::array>& keepalive,
    std::vector<NativeArrayView>& views,
    std::vector<std::string>& name_storage,
    const std::string& name,
    py::array_t<T, py::array::c_style | py::array::forcecast> arr
)
{
    NativeArrayView view;
    view.data = arr.mutable_data();
    view.dtype = NativeDType<T>::value;
    view.ndim = static_cast<std::int32_t>(arr.ndim());
    for (ssize_t axis = 0; axis < arr.ndim(); ++axis) {
        view.shape[axis] = static_cast<std::int64_t>(arr.shape(axis));
        view.strides[axis] = static_cast<std::int64_t>(arr.strides(axis));
    }
    keepalive.emplace_back(std::move(arr));
    views.push_back(view);
    name_storage.push_back(name);
}

struct NativeArgBundle {
    std::vector<py::array> keepalive;
    std::vector<NativeArrayView> views;
    std::vector<std::string> name_storage;
    std::vector<const char*> names;
    KernelStaticArgs static_args;

    NativeArrayView* find(const std::string& name)
    {
        for (std::size_t i = 0; i < name_storage.size(); ++i) {
            if (name_storage[i] == name) {
                return &views[i];
            }
        }
        return nullptr;
    }

    const NativeArrayView* find(const std::string& name) const
    {
        for (std::size_t i = 0; i < name_storage.size(); ++i) {
            if (name_storage[i] == name) {
                return &views[i];
            }
        }
        return nullptr;
    }
};

NativeArgBundle build_native_args(py::sequence param_order, py::dict static_args)
{
    NativeArgBundle bundle;
    const ssize_t n = py::len(param_order);
    bundle.keepalive.reserve(static_cast<std::size_t>(n));
    bundle.views.reserve(static_cast<std::size_t>(n));
    bundle.name_storage.reserve(static_cast<std::size_t>(n));
    for (ssize_t i = 0; i < n; ++i) {
        const std::string name = py::str(param_order[i]);
        py::object key = py::str(name);
        if (!static_args.contains(key)) {
            throw py::key_error("Missing native kernel static argument '" + name + "'.");
        }
        py::handle value = static_args[key];
        if (is_int32_arg(name)) {
            push_arg<std::int32_t>(bundle.keepalive, bundle.views, bundle.name_storage, name, as_array_for_arg<std::int32_t>(value, name));
        } else if (is_uint8_arg(name)) {
            push_arg<std::uint8_t>(bundle.keepalive, bundle.views, bundle.name_storage, name, as_array_for_arg<std::uint8_t>(value, name));
        } else {
            push_arg<double>(bundle.keepalive, bundle.views, bundle.name_storage, name, as_array_for_arg<double>(value, name));
        }
    }
    bundle.names.reserve(bundle.name_storage.size());
    for (const auto& name : bundle.name_storage) {
        bundle.names.push_back(name.c_str());
    }
    bundle.static_args.arrays = bundle.views.data();
    bundle.static_args.names = bundle.names.data();
    bundle.static_args.count = bundle.views.size();
    return bundle;
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

std::vector<std::int64_t> as_index_vector(py::handle obj, const char* label)
{
    auto arr = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr || arr.ndim() != 1) {
        throw py::value_error(std::string(label) + " must be a rank-1 integer array.");
    }
    std::vector<std::int64_t> out(static_cast<std::size_t>(arr.shape(0)));
    auto view = arr.unchecked<1>();
    std::unordered_set<std::int64_t> seen;
    seen.reserve(static_cast<std::size_t>(arr.shape(0)));
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        const auto value = static_cast<std::int64_t>(view(i));
        if (value < 0) {
            throw py::value_error(std::string(label) + " contains negative entries.");
        }
        if (seen.find(value) != seen.end()) {
            throw py::value_error(std::string(label) + " must be unique.");
        }
        seen.insert(value);
        out[static_cast<std::size_t>(i)] = value;
    }
    return out;
}

std::vector<std::string> string_list(py::sequence seq)
{
    std::vector<std::string> out;
    const ssize_t n = py::len(seq);
    out.reserve(static_cast<std::size_t>(n));
    for (ssize_t i = 0; i < n; ++i) {
        out.push_back(py::str(seq[i]));
    }
    return out;
}

vector_type optional_weights(py::handle obj, Eigen::Index n, const char* label)
{
    vector_type weights(n);
    weights.setOnes();
    if (obj.is_none()) {
        return weights;
    }
    auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr || arr.ndim() != 1 || arr.shape(0) != n) {
        throw py::value_error(std::string(label) + " must be a rank-1 array with the expected size.");
    }
    auto view = arr.unchecked<1>();
    for (Eigen::Index i = 0; i < n; ++i) {
        if (!std::isfinite(view(i)) || view(i) < 0.0) {
            throw py::value_error(std::string(label) + " must be finite and nonnegative.");
        }
        weights(i) = view(i);
    }
    return weights;
}

double* double_ptr(NativeArrayView& view)
{
    if (view.dtype != NativeArrayDType::Float64) {
        throw std::runtime_error("Expected float64 native array.");
    }
    return static_cast<double*>(view.data);
}

const std::int32_t* int32_ptr(const NativeArrayView& view)
{
    if (view.dtype != NativeArrayDType::Int32) {
        throw std::runtime_error("Expected int32 native array.");
    }
    return static_cast<const std::int32_t*>(view.data);
}

std::int64_t stride_elems(const NativeArrayView& view, int axis, std::int64_t item_size)
{
    return view.strides[axis] / item_size;
}

std::int64_t flat_size(const NativeArrayView& view)
{
    std::int64_t n = 1;
    for (int axis = 0; axis < view.ndim; ++axis) {
        n *= view.shape[axis];
    }
    return n;
}

struct AffineStateUpdate {
    std::string name;
    matrix_type basis;
    vector_type offset;
};

struct SymbolicStateUpdate {
    NativeKernelMetadata* metadata = nullptr;
    NativeArgBundle args;
    std::string target_name;
    double scale = 1.0;
    double offset = 0.0;
};

struct CsrMatrix {
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::vector<std::int64_t> indptr;
    std::vector<std::int64_t> indices;
    std::vector<double> data;
};

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

std::vector<std::int64_t> as_raw_int64_vector(py::handle obj, const char* label)
{
    auto arr = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr || arr.ndim() != 1) {
        throw py::value_error(std::string(label) + " must be a rank-1 int64 array.");
    }
    std::vector<std::int64_t> out(static_cast<std::size_t>(arr.shape(0)));
    auto view = arr.unchecked<1>();
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = static_cast<std::int64_t>(view(i));
    }
    return out;
}

CsrMatrix parse_csr_matrix(py::dict payload, const char* label)
{
    const std::string layout = py::str(payload[py::str("layout")]);
    if (layout != "csr") {
        throw py::value_error(std::string(label) + " must use CSR layout.");
    }
    const auto shape = as_raw_int64_vector(payload[py::str("shape")], "CSR shape");
    if (shape.size() != 2 || shape[0] < 0 || shape[1] < 0) {
        throw py::value_error(std::string(label) + " shape must have two nonnegative entries.");
    }
    CsrMatrix csr;
    csr.rows = shape[0];
    csr.cols = shape[1];
    csr.indptr = as_raw_int64_vector(payload[py::str("indptr")], "CSR indptr");
    csr.indices = as_raw_int64_vector(payload[py::str("indices")], "CSR indices");
    csr.data = as_double_vector(payload[py::str("data")], "CSR data");
    if (static_cast<std::int64_t>(csr.indptr.size()) != csr.rows + 1) {
        throw py::value_error(std::string(label) + " indptr length must be n_rows + 1.");
    }
    if (csr.indices.size() != csr.data.size()) {
        throw py::value_error(std::string(label) + " indices/data size mismatch.");
    }
    if (!csr.indptr.empty() && csr.indptr.front() != 0) {
        throw py::value_error(std::string(label) + " indptr must start at zero.");
    }
    if (!csr.indptr.empty() && csr.indptr.back() != static_cast<std::int64_t>(csr.indices.size())) {
        throw py::value_error(std::string(label) + " indptr[-1] must equal nnz.");
    }
    for (std::int64_t row = 0; row < csr.rows; ++row) {
        const std::int64_t start = csr.indptr[static_cast<std::size_t>(row)];
        const std::int64_t stop = csr.indptr[static_cast<std::size_t>(row + 1)];
        if (start > stop || start < 0 || stop > static_cast<std::int64_t>(csr.indices.size())) {
            throw py::value_error(std::string(label) + " indptr is invalid.");
        }
        std::int64_t previous = -1;
        for (std::int64_t p = start; p < stop; ++p) {
            const std::int64_t col = csr.indices[static_cast<std::size_t>(p)];
            if (col < 0 || col >= csr.cols) {
                throw py::value_error(std::string(label) + " column index is out of range.");
            }
            if (col <= previous) {
                throw py::value_error(std::string(label) + " row indices must be strictly increasing.");
            }
            previous = col;
        }
    }
    return csr;
}

vector_type csr_matvec(const CsrMatrix& csr, const vector_type& vector)
{
    if (vector.size() != csr.cols) {
        throw std::runtime_error("CSR matvec dimension mismatch.");
    }
    vector_type out = vector_type::Zero(csr.rows);
    for (std::int64_t row = 0; row < csr.rows; ++row) {
        double sum = 0.0;
        const std::int64_t start = csr.indptr[static_cast<std::size_t>(row)];
        const std::int64_t stop = csr.indptr[static_cast<std::size_t>(row + 1)];
        for (std::int64_t p = start; p < stop; ++p) {
            sum += csr.data[static_cast<std::size_t>(p)] * vector(csr.indices[static_cast<std::size_t>(p)]);
        }
        out(row) = sum;
    }
    return out;
}

matrix_type csr_matmat(const CsrMatrix& csr, const matrix_type& matrix)
{
    if (matrix.rows() != csr.cols) {
        throw std::runtime_error("CSR matmat dimension mismatch.");
    }
    matrix_type out = matrix_type::Zero(csr.rows, matrix.cols());
    for (std::int64_t row = 0; row < csr.rows; ++row) {
        const std::int64_t start = csr.indptr[static_cast<std::size_t>(row)];
        const std::int64_t stop = csr.indptr[static_cast<std::size_t>(row + 1)];
        for (std::int64_t p = start; p < stop; ++p) {
            const double value = csr.data[static_cast<std::size_t>(p)];
            const std::int64_t col = csr.indices[static_cast<std::size_t>(p)];
            for (Eigen::Index mode = 0; mode < matrix.cols(); ++mode) {
                out(row, mode) += value * matrix(col, mode);
            }
        }
    }
    return out;
}

std::vector<AffineStateUpdate> parse_affine_state_updates(py::sequence raw, Eigen::Index n_modes)
{
    std::vector<AffineStateUpdate> out;
    const ssize_t n = py::len(raw);
    out.reserve(static_cast<std::size_t>(n));
    for (ssize_t i = 0; i < n; ++i) {
        py::dict item = raw[i].cast<py::dict>();
        AffineStateUpdate spec;
        spec.name = py::str(item[py::str("name")]);
        spec.basis = as_matrix(item[py::str("basis")], "affine state basis");
        spec.offset = as_vector(item[py::str("offset")], "affine state offset");
        if (spec.basis.cols() != n_modes) {
            throw py::value_error("affine state basis column count must match reduced coefficient size.");
        }
        if (spec.basis.rows() != spec.offset.size()) {
            throw py::value_error("affine state basis and offset sizes are incompatible.");
        }
        out.emplace_back(std::move(spec));
    }
    return out;
}

std::vector<SymbolicStateUpdate> parse_symbolic_state_updates(py::sequence raw)
{
    std::vector<SymbolicStateUpdate> out;
    const ssize_t n = py::len(raw);
    out.reserve(static_cast<std::size_t>(n));
    for (ssize_t i = 0; i < n; ++i) {
        py::dict item = raw[i].cast<py::dict>();
        SymbolicStateUpdate spec;
        spec.metadata = metadata_from_capsule(item[py::str("metadata_capsule")]);
        spec.args = build_native_args(
            item[py::str("param_order")].cast<py::sequence>(),
            item[py::str("static_args")].cast<py::dict>()
        );
        spec.target_name = py::str(item[py::str("target_name")]);
        if (spec.target_name.empty()) {
            throw py::value_error("symbolic state update target_name must not be empty.");
        }
        if (item.contains(py::str("scale"))) {
            spec.scale = py::float_(item[py::str("scale")]);
        }
        if (item.contains(py::str("offset"))) {
            spec.offset = py::float_(item[py::str("offset")]);
        }
        if (!std::isfinite(spec.scale) || !std::isfinite(spec.offset)) {
            throw py::value_error("symbolic state update scale/offset must be finite.");
        }
        out.emplace_back(std::move(spec));
    }
    return out;
}

void update_coefficient_arg(
    NativeArgBundle& bundle,
    const std::string& name,
    const matrix_type& basis,
    const vector_type& offset,
    const vector_type& q
)
{
    NativeArrayView* target = bundle.find(name);
    const NativeArrayView* gdofs = bundle.find("gdofs_map");
    if (target == nullptr || gdofs == nullptr) {
        return;
    }
    if (target->dtype != NativeArrayDType::Float64 || target->ndim != 2 || gdofs->dtype != NativeArrayDType::Int32 || gdofs->ndim != 2) {
        throw std::runtime_error("coefficient local arrays and gdofs_map must be rank-2 float64/int32 arrays.");
    }
    if (target->shape[0] != gdofs->shape[0] || target->shape[1] != gdofs->shape[1]) {
        throw std::runtime_error("coefficient local array shape must match gdofs_map shape.");
    }
    double* values = double_ptr(*target);
    const std::int32_t* map = int32_ptr(*gdofs);
    const auto ve = stride_elems(*target, 0, sizeof(double));
    const auto vi = stride_elems(*target, 1, sizeof(double));
    const auto me = stride_elems(*gdofs, 0, sizeof(std::int32_t));
    const auto mi = stride_elems(*gdofs, 1, sizeof(std::int32_t));
    for (std::int64_t e = 0; e < gdofs->shape[0]; ++e) {
        for (std::int64_t i = 0; i < gdofs->shape[1]; ++i) {
            const std::int32_t gdof = map[e * me + i * mi];
            double value = 0.0;
            if (gdof >= 0) {
                if (gdof >= basis.rows()) {
                    throw std::runtime_error("gdofs_map contains dofs outside reduced basis.");
                }
                value = offset(gdof);
                for (Eigen::Index m = 0; m < basis.cols(); ++m) {
                    value += basis(gdof, m) * q(m);
                }
            }
            values[e * ve + i * vi] = value;
        }
    }
}

void update_affine_state_arg(
    NativeArgBundle& bundle,
    const AffineStateUpdate& spec,
    const vector_type& q
)
{
    NativeArrayView* target = bundle.find(spec.name);
    if (target == nullptr) {
        return;
    }
    if (target->dtype != NativeArrayDType::Float64) {
        throw std::runtime_error("affine runtime state target must be float64.");
    }
    const std::int64_t n = flat_size(*target);
    if (n != spec.offset.size()) {
        throw std::runtime_error("affine runtime state target size does not match update offset size.");
    }
    double* values = double_ptr(*target);
    for (std::int64_t i = 0; i < n; ++i) {
        double value = spec.offset(i);
        for (Eigen::Index m = 0; m < spec.basis.cols(); ++m) {
            value += spec.basis(i, m) * q(m);
        }
        values[i] = value;
    }
}

struct LocalBlocks {
    std::vector<double> K;
    std::vector<double> F;
    std::vector<double> J;
    std::int64_t n_entities = 0;
    std::int64_t n_local_dofs = 0;
    std::int64_t functional_dim = 0;
};

LocalBlocks call_kernel(NativeKernelMetadata* metadata, NativeArgBundle& bundle)
{
    KernelMutableArgs mutable_args;
    KernelOutputs outputs;
    metadata->entrypoint(bundle.static_args, mutable_args, outputs);
    if (outputs.n_entities < 0 || outputs.n_local_dofs < 0 || outputs.functional_dim < 0) {
        throw std::runtime_error("native kernel returned invalid output dimensions.");
    }
    LocalBlocks local;
    local.n_entities = outputs.n_entities;
    local.n_local_dofs = outputs.n_local_dofs;
    local.functional_dim = outputs.functional_dim;
    local.K.assign(static_cast<std::size_t>(local.n_entities * local.n_local_dofs * local.n_local_dofs), 0.0);
    local.F.assign(static_cast<std::size_t>(local.n_entities * local.n_local_dofs), 0.0);
    local.J.assign(static_cast<std::size_t>(local.n_entities * local.functional_dim), 0.0);
    outputs.K = local.K.data();
    outputs.F = local.F.data();
    outputs.J = local.J.data();
    metadata->entrypoint(bundle.static_args, mutable_args, outputs);
    return local;
}

double K_at(const LocalBlocks& local, std::int64_t e, std::int64_t i, std::int64_t j)
{
    return local.K[static_cast<std::size_t>((e * local.n_local_dofs + i) * local.n_local_dofs + j)];
}

double F_at(const LocalBlocks& local, std::int64_t e, std::int64_t i)
{
    return local.F[static_cast<std::size_t>(e * local.n_local_dofs + i)];
}

double J_at(const LocalBlocks& local, std::int64_t e, std::int64_t j)
{
    return local.J[static_cast<std::size_t>(e * local.functional_dim + j)];
}

std::int64_t flat_offset(const NativeArrayView& view, std::int64_t flat_index)
{
    std::int64_t remaining = flat_index;
    std::int64_t offset = 0;
    for (int axis = view.ndim - 1; axis >= 0; --axis) {
        const std::int64_t extent = view.shape[axis];
        if (extent <= 0) {
            throw std::runtime_error("Cannot write into empty state array.");
        }
        const std::int64_t index = remaining % extent;
        remaining /= extent;
        offset += index * stride_elems(view, axis, sizeof(double));
    }
    if (remaining != 0) {
        throw std::runtime_error("State array flat index is out of range.");
    }
    return offset;
}

void copy_state_update_output_to_bundle(NativeArgBundle& bundle, const SymbolicStateUpdate& spec, const LocalBlocks& output)
{
    NativeArrayView* target = bundle.find(spec.target_name);
    if (target == nullptr) {
        return;
    }
    if (target->dtype != NativeArrayDType::Float64) {
        throw std::runtime_error("symbolic state update target must be float64.");
    }
    const std::int64_t target_size = flat_size(*target);
    const std::int64_t source_size = output.n_entities * output.functional_dim;
    if (target_size != source_size) {
        throw std::runtime_error("symbolic state update target size does not match kernel functional output size.");
    }
    double* values = double_ptr(*target);
    for (std::int64_t e = 0; e < output.n_entities; ++e) {
        for (std::int64_t j = 0; j < output.functional_dim; ++j) {
            const std::int64_t flat = e * output.functional_dim + j;
            values[flat_offset(*target, flat)] = spec.offset + spec.scale * J_at(output, e, j);
        }
    }
}

void execute_symbolic_state_update(
    NativeArgBundle& residual_args,
    NativeArgBundle& tangent_args,
    SymbolicStateUpdate& spec
)
{
    LocalBlocks output = call_kernel(spec.metadata, spec.args);
    copy_state_update_output_to_bundle(residual_args, spec, output);
    copy_state_update_output_to_bundle(tangent_args, spec, output);
    copy_state_update_output_to_bundle(spec.args, spec, output);
}

struct TargetSystem {
    vector_type residual;
    matrix_type jacobian;
};

struct TimingCounters {
    double state_update_seconds = 0.0;
    double kernel_seconds = 0.0;
    double projection_seconds = 0.0;
    double sparse_lift_seconds = 0.0;
    double deim_interpolation_seconds = 0.0;
    double deim_composition_seconds = 0.0;
    double step_solve_seconds = 0.0;
    double line_search_seconds = 0.0;
    int assemblies = 0;
    int kernel_calls = 0;
    int dense_lift_applications = 0;
    int sparse_lift_applications = 0;
    int deim_interpolation_applications = 0;
    int deim_composition_applications = 0;
    int symbolic_state_update_calls = 0;
    std::int64_t sparse_lift_nonzeros = 0;
    std::int64_t sparse_lift_rows = 0;
    std::int64_t sparse_lift_cols = 0;
};

struct OnlineProblem {
    NativeKernelMetadata* residual_metadata = nullptr;
    NativeKernelMetadata* tangent_metadata = nullptr;
    NativeArgBundle residual_args;
    NativeArgBundle tangent_args;
    matrix_type basis;
    vector_type offset;
    std::vector<std::int64_t> row_dofs;
    vector_type element_weights;
    vector_type row_weights;
    bool has_row_weights = false;
    matrix_type lift;
    bool has_dense_lift = false;
    CsrMatrix sparse_lift;
    bool has_sparse_lift = false;
    std::vector<std::string> coefficient_arg_names;
    std::vector<AffineStateUpdate> residual_state_updates;
    std::vector<AffineStateUpdate> tangent_state_updates;
    std::vector<SymbolicStateUpdate> residual_symbolic_state_updates;
    std::vector<SymbolicStateUpdate> tangent_symbolic_state_updates;
    TimingCounters timings;
};

struct DeimTarget {
    matrix_type interpolation_operator;
    matrix_type residual_terms;
};

void update_all_state(OnlineProblem& problem, const vector_type& q)
{
    for (const auto& name : problem.coefficient_arg_names) {
        update_coefficient_arg(problem.residual_args, name, problem.basis, problem.offset, q);
        update_coefficient_arg(problem.tangent_args, name, problem.basis, problem.offset, q);
        for (auto& spec : problem.residual_symbolic_state_updates) {
            update_coefficient_arg(spec.args, name, problem.basis, problem.offset, q);
        }
        for (auto& spec : problem.tangent_symbolic_state_updates) {
            update_coefficient_arg(spec.args, name, problem.basis, problem.offset, q);
        }
    }
    for (const auto& spec : problem.residual_state_updates) {
        update_affine_state_arg(problem.residual_args, spec, q);
    }
    for (const auto& spec : problem.tangent_state_updates) {
        update_affine_state_arg(problem.tangent_args, spec, q);
    }
    for (auto& spec : problem.residual_symbolic_state_updates) {
        execute_symbolic_state_update(problem.residual_args, problem.tangent_args, spec);
        problem.timings.symbolic_state_update_calls += 1;
    }
    for (auto& spec : problem.tangent_symbolic_state_updates) {
        execute_symbolic_state_update(problem.residual_args, problem.tangent_args, spec);
        problem.timings.symbolic_state_update_calls += 1;
    }
}

TargetSystem apply_configured_lift(OnlineProblem& problem, TargetSystem sampled)
{
    TargetSystem target;
    if (problem.has_sparse_lift) {
        if (problem.sparse_lift.cols != sampled.residual.size()) {
            throw std::runtime_error("Sparse GNAT lift column count must match sampled row count.");
        }
        const auto t_sparse0 = clock_type::now();
        target.residual = csr_matvec(problem.sparse_lift, sampled.residual);
        target.jacobian = csr_matmat(problem.sparse_lift, sampled.jacobian);
        const auto t_sparse1 = clock_type::now();
        problem.timings.sparse_lift_seconds += elapsed_seconds(t_sparse0, t_sparse1);
        problem.timings.sparse_lift_applications += 1;
    } else if (problem.has_dense_lift) {
        if (problem.lift.cols() != sampled.residual.size()) {
            throw std::runtime_error("GNAT lift column count must match sampled row count.");
        }
        target.residual = problem.lift * sampled.residual;
        target.jacobian = problem.lift * sampled.jacobian;
        problem.timings.dense_lift_applications += 1;
    } else {
        target.residual = std::move(sampled.residual);
        target.jacobian = std::move(sampled.jacobian);
    }
    return target;
}

TargetSystem assemble_sampled_rows(OnlineProblem& problem, const vector_type& q)
{
    const auto t_state0 = clock_type::now();
    update_all_state(problem, q);
    const auto t_state1 = clock_type::now();
    LocalBlocks residual_blocks = call_kernel(problem.residual_metadata, problem.residual_args);
    LocalBlocks tangent_blocks = call_kernel(problem.tangent_metadata, problem.tangent_args);
    const auto t_kernel1 = clock_type::now();
    problem.timings.state_update_seconds += elapsed_seconds(t_state0, t_state1);
    problem.timings.kernel_seconds += elapsed_seconds(t_state1, t_kernel1);
    problem.timings.assemblies += 1;
    problem.timings.kernel_calls += 2;
    if (residual_blocks.n_entities != tangent_blocks.n_entities || residual_blocks.n_local_dofs != tangent_blocks.n_local_dofs) {
        throw std::runtime_error("residual and tangent kernels returned incompatible local block dimensions.");
    }

    const auto t_projection0 = clock_type::now();
    const NativeArrayView* gdofs = problem.tangent_args.find("gdofs_map");
    if (gdofs == nullptr || gdofs->dtype != NativeArrayDType::Int32 || gdofs->ndim != 2) {
        throw std::runtime_error("tangent static args must contain rank-2 int32 gdofs_map.");
    }
    if (gdofs->shape[0] != tangent_blocks.n_entities || gdofs->shape[1] != tangent_blocks.n_local_dofs) {
        throw std::runtime_error("gdofs_map shape is incompatible with native local blocks.");
    }
    const std::int32_t* map = int32_ptr(*gdofs);
    const auto me = stride_elems(*gdofs, 0, sizeof(std::int32_t));
    const auto mi = stride_elems(*gdofs, 1, sizeof(std::int32_t));

    std::vector<std::int64_t> row_positions(static_cast<std::size_t>(problem.basis.rows()), -1);
    for (std::size_t s = 0; s < problem.row_dofs.size(); ++s) {
        const auto row = problem.row_dofs[s];
        if (row < 0 || row >= problem.basis.rows()) {
            throw std::runtime_error("row_dofs contains out-of-range entries.");
        }
        row_positions[static_cast<std::size_t>(row)] = static_cast<std::int64_t>(s);
    }

    vector_type sampled_residual = vector_type::Zero(static_cast<Eigen::Index>(problem.row_dofs.size()));
    matrix_type sampled_jacobian = matrix_type::Zero(static_cast<Eigen::Index>(problem.row_dofs.size()), problem.basis.cols());
    for (std::int64_t e = 0; e < tangent_blocks.n_entities; ++e) {
        const double weight = problem.element_weights(e);
        for (std::int64_t i = 0; i < tangent_blocks.n_local_dofs; ++i) {
            const std::int32_t gi = map[e * me + i * mi];
            if (gi < 0) {
                continue;
            }
            const auto sample = row_positions[static_cast<std::size_t>(gi)];
            if (sample < 0) {
                continue;
            }
            sampled_residual(sample) += weight * F_at(residual_blocks, e, i);
            for (std::int64_t j = 0; j < tangent_blocks.n_local_dofs; ++j) {
                const std::int32_t gj = map[e * me + j * mi];
                if (gj < 0) {
                    continue;
                }
                const double kij = weight * K_at(tangent_blocks, e, i, j);
                for (Eigen::Index m = 0; m < problem.basis.cols(); ++m) {
                    sampled_jacobian(sample, m) += kij * problem.basis(gj, m);
                }
            }
        }
    }

    if (problem.has_row_weights) {
        for (Eigen::Index i = 0; i < sampled_residual.size(); ++i) {
            const double scale = std::sqrt(problem.row_weights(i));
            sampled_residual(i) *= scale;
            sampled_jacobian.row(i) *= scale;
        }
    }

    const auto t_projection1 = clock_type::now();
    problem.timings.projection_seconds += elapsed_seconds(t_projection0, t_projection1);

    TargetSystem sampled;
    sampled.residual = std::move(sampled_residual);
    sampled.jacobian = std::move(sampled_jacobian);
    return sampled;
}

TargetSystem assemble_target(OnlineProblem& problem, const vector_type& q)
{
    return apply_configured_lift(problem, assemble_sampled_rows(problem, q));
}

matrix_type interpolation_operator_from_selected_basis(const matrix_type& selected_basis, double rcond)
{
    if (selected_basis.rows() == 0 || selected_basis.cols() == 0) {
        throw py::value_error("selected_basis must be nonempty.");
    }
    matrix_type identity = matrix_type::Identity(selected_basis.rows(), selected_basis.rows());
    if (selected_basis.rows() == selected_basis.cols()) {
        Eigen::ColPivHouseholderQR<matrix_type> qr(selected_basis);
        if (rcond > 0.0) {
            qr.setThreshold(rcond);
        }
        matrix_type op = qr.solve(identity);
        if (qr.rank() == selected_basis.cols() && op.allFinite()) {
            return op;
        }
    }
    Eigen::JacobiSVD<matrix_type> svd(selected_basis, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (rcond > 0.0) {
        svd.setThreshold(rcond);
    }
    matrix_type op = svd.solve(identity);
    if (!op.allFinite()) {
        throw py::value_error("DEIM/QDEIM interpolation operator contains non-finite values.");
    }
    return op;
}

TargetSystem assemble_deim_target(OnlineProblem& problem, const DeimTarget& deim, const vector_type& q)
{
    TargetSystem sampled = assemble_sampled_rows(problem, q);
    if (deim.interpolation_operator.cols() != sampled.residual.size()) {
        throw std::runtime_error("DEIM interpolation operator columns must match sampled row count.");
    }
    if (deim.residual_terms.rows() != deim.interpolation_operator.rows()) {
        throw std::runtime_error("DEIM residual term count must match interpolation mode count.");
    }

    const auto t_interp0 = clock_type::now();
    vector_type coefficients = deim.interpolation_operator * sampled.residual;
    matrix_type coefficient_jacobian = deim.interpolation_operator * sampled.jacobian;
    const auto t_interp1 = clock_type::now();
    problem.timings.deim_interpolation_seconds += elapsed_seconds(t_interp0, t_interp1);
    problem.timings.deim_interpolation_applications += 1;

    const auto t_comp0 = clock_type::now();
    TargetSystem target;
    target.residual = deim.residual_terms.transpose() * coefficients;
    target.jacobian = deim.residual_terms.transpose() * coefficient_jacobian;
    const auto t_comp1 = clock_type::now();
    problem.timings.deim_composition_seconds += elapsed_seconds(t_comp0, t_comp1);
    problem.timings.deim_composition_applications += 1;
    return apply_configured_lift(problem, std::move(target));
}

struct StepSolve {
    vector_type step;
    int rank = 0;
    std::string method;
};

StepSolve solve_step(const matrix_type& J, const vector_type& residual, double damping, double rcond)
{
    if (J.cols() == 0) {
        StepSolve empty;
        empty.step.resize(0);
        empty.method = "empty";
        return empty;
    }
    const bool has_damping = damping > 0.0;
    matrix_type A(J.rows() + (has_damping ? J.cols() : 0), J.cols());
    vector_type b(residual.size() + (has_damping ? J.cols() : 0));
    A.topRows(J.rows()) = J;
    b.head(residual.size()) = -residual;
    if (has_damping) {
        A.bottomRows(J.cols()).setZero();
        for (Eigen::Index j = 0; j < J.cols(); ++j) {
            A(J.rows() + j, j) = std::sqrt(damping);
        }
        b.tail(J.cols()).setZero();
    }

    Eigen::ColPivHouseholderQR<matrix_type> qr(A);
    if (rcond > 0.0) {
        qr.setThreshold(rcond);
    }
    StepSolve out;
    out.step = qr.solve(b);
    out.rank = static_cast<int>(qr.rank());
    out.method = "qr";
    if (out.rank < A.cols() || !out.step.allFinite()) {
        Eigen::JacobiSVD<matrix_type> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        if (rcond > 0.0) {
            svd.setThreshold(rcond);
        }
        out.step = svd.solve(b);
        out.rank = static_cast<int>(svd.rank());
        out.method = "svd";
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

py::array_t<double> std_vector_to_array(const std::vector<double>& values)
{
    py::array_t<double> out(values.size());
    auto view = out.mutable_unchecked<1>();
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        view(i) = values[static_cast<std::size_t>(i)];
    }
    return out;
}

py::dict timing_to_dict(const TimingCounters& timing)
{
    py::dict out;
    out["state_update_seconds"] = timing.state_update_seconds;
    out["kernel_seconds"] = timing.kernel_seconds;
    out["projection_seconds"] = timing.projection_seconds;
    out["sparse_lift_seconds"] = timing.sparse_lift_seconds;
    out["deim_interpolation_seconds"] = timing.deim_interpolation_seconds;
    out["deim_composition_seconds"] = timing.deim_composition_seconds;
    out["step_solve_seconds"] = timing.step_solve_seconds;
    out["line_search_seconds"] = timing.line_search_seconds;
    out["assemblies"] = timing.assemblies;
    out["kernel_calls"] = timing.kernel_calls;
    out["dense_lift_applications"] = timing.dense_lift_applications;
    out["sparse_lift_applications"] = timing.sparse_lift_applications;
    out["deim_interpolation_applications"] = timing.deim_interpolation_applications;
    out["deim_composition_applications"] = timing.deim_composition_applications;
    out["symbolic_state_update_calls"] = timing.symbolic_state_update_calls;
    out["sparse_lift_nonzeros"] = timing.sparse_lift_nonzeros;
    out["sparse_lift_rows"] = timing.sparse_lift_rows;
    out["sparse_lift_cols"] = timing.sparse_lift_cols;
    return out;
}

py::dict solve_online_gauss_newton(
    py::handle residual_metadata_capsule,
    py::sequence residual_param_order,
    py::dict residual_static_args,
    py::handle tangent_metadata_capsule,
    py::sequence tangent_param_order,
    py::dict tangent_static_args,
    py::handle trial_basis_obj,
    py::handle offset_obj,
    py::handle initial_coefficients_obj,
    py::handle row_dofs_obj,
    py::handle coefficient_arg_names_obj,
    py::handle element_weights_obj,
    py::handle row_weights_obj,
    py::handle gnat_lift_obj,
    py::sequence residual_state_updates_obj,
    py::sequence tangent_state_updates_obj,
    py::sequence residual_symbolic_state_updates_obj,
    py::sequence tangent_symbolic_state_updates_obj,
    int max_iterations,
    double residual_tol,
    double step_tol,
    double damping,
    bool adaptive_damping,
    int max_damping_retries,
    double damping_increase,
    double damping_decrease,
    bool line_search,
    int max_line_search,
    double sufficient_decrease,
    double rcond
)
{
    OnlineProblem problem;
    problem.residual_metadata = metadata_from_capsule(residual_metadata_capsule);
    problem.tangent_metadata = metadata_from_capsule(tangent_metadata_capsule);
    problem.residual_args = build_native_args(residual_param_order, residual_static_args);
    problem.tangent_args = build_native_args(tangent_param_order, tangent_static_args);
    problem.basis = as_matrix(trial_basis_obj, "trial_basis");
    problem.offset = as_vector(offset_obj, "offset");
    vector_type q = as_vector(initial_coefficients_obj, "initial_coefficients");
    if (problem.basis.rows() != problem.offset.size() || problem.basis.cols() != q.size()) {
        throw py::value_error("trial_basis, offset, and initial_coefficients have incompatible dimensions.");
    }
    problem.row_dofs = as_index_vector(row_dofs_obj, "row_dofs");
    if (problem.row_dofs.empty()) {
        throw py::value_error("row_dofs must not be empty.");
    }
    problem.coefficient_arg_names = string_list(coefficient_arg_names_obj.cast<py::sequence>());
    if (!gnat_lift_obj.is_none() && py::isinstance<py::dict>(gnat_lift_obj)) {
        problem.sparse_lift = parse_csr_matrix(gnat_lift_obj.cast<py::dict>(), "gnat_lift");
        if (problem.sparse_lift.cols != static_cast<std::int64_t>(problem.row_dofs.size())) {
            throw py::value_error("sparse gnat_lift column count must match row_dofs size.");
        }
        problem.has_sparse_lift = true;
        problem.timings.sparse_lift_nonzeros = static_cast<std::int64_t>(problem.sparse_lift.data.size());
        problem.timings.sparse_lift_rows = problem.sparse_lift.rows;
        problem.timings.sparse_lift_cols = problem.sparse_lift.cols;
    } else if (!gnat_lift_obj.is_none()) {
        problem.lift = as_matrix(gnat_lift_obj, "gnat_lift");
        if (problem.lift.cols() != static_cast<Eigen::Index>(problem.row_dofs.size())) {
            throw py::value_error("gnat_lift column count must match row_dofs size.");
        }
        problem.has_dense_lift = true;
    }

    // Probe once after initial decode to size element weights.
    update_all_state(problem, q);
    LocalBlocks probe = call_kernel(problem.tangent_metadata, problem.tangent_args);
    problem.element_weights = optional_weights(element_weights_obj, probe.n_entities, "element_weights");
    problem.has_row_weights = !row_weights_obj.is_none();
    problem.row_weights = optional_weights(row_weights_obj, static_cast<Eigen::Index>(problem.row_dofs.size()), "row_weights");
    problem.residual_state_updates = parse_affine_state_updates(residual_state_updates_obj, q.size());
    problem.tangent_state_updates = parse_affine_state_updates(tangent_state_updates_obj, q.size());
    problem.residual_symbolic_state_updates = parse_symbolic_state_updates(residual_symbolic_state_updates_obj);
    problem.tangent_symbolic_state_updates = parse_symbolic_state_updates(tangent_symbolic_state_updates_obj);

    if (!std::isfinite(residual_tol) || residual_tol < 0.0 || !std::isfinite(step_tol) || step_tol < 0.0) {
        throw py::value_error("residual_tol and step_tol must be finite and nonnegative.");
    }
    if (!std::isfinite(damping) || damping < 0.0) {
        throw py::value_error("damping must be finite and nonnegative.");
    }
    if (!std::isfinite(damping_increase) || damping_increase <= 1.0) {
        throw py::value_error("damping_increase must be finite and greater than one.");
    }
    if (!std::isfinite(damping_decrease) || damping_decrease <= 0.0 || damping_decrease > 1.0) {
        throw py::value_error("damping_decrease must be in the interval (0, 1].");
    }

    std::vector<double> residual_history;
    std::vector<double> step_history;
    std::vector<double> alpha_history;
    std::vector<double> damping_history;
    residual_history.reserve(static_cast<std::size_t>(std::max(1, max_iterations)));
    step_history.reserve(static_cast<std::size_t>(std::max(1, max_iterations)));
    alpha_history.reserve(static_cast<std::size_t>(std::max(1, max_iterations)));
    damping_history.reserve(static_cast<std::size_t>(std::max(1, max_iterations)));
    bool converged = false;
    int iterations = 0;
    int rejected_step_count = 0;
    std::string final_method = "";
    double final_norm = std::numeric_limits<double>::infinity();
    double current_damping = damping;

    {
        py::gil_scoped_release release;
        for (int iteration = 1; iteration <= std::max(1, max_iterations); ++iteration) {
            iterations = iteration;
            TargetSystem target = assemble_target(problem, q);
            final_norm = target.residual.norm();
            residual_history.push_back(final_norm);
            if (final_norm <= residual_tol) {
                step_history.push_back(0.0);
                alpha_history.push_back(0.0);
                damping_history.push_back(current_damping);
                converged = true;
                break;
            }

            StepSolve solved;
            double step_norm = std::numeric_limits<double>::infinity();
            double accepted_alpha = 1.0;
            double accepted_norm = std::numeric_limits<double>::infinity();
            double used_damping = current_damping;
            vector_type accepted_q = q;
            bool step_accepted = false;
            const int n_attempts = adaptive_damping ? std::max(1, max_damping_retries) : 1;
            for (int attempt = 0; attempt < n_attempts; ++attempt) {
                used_damping = current_damping;
                const auto t_solve0 = clock_type::now();
                solved = solve_step(target.jacobian, target.residual, current_damping, rcond);
                const auto t_solve1 = clock_type::now();
                problem.timings.step_solve_seconds += elapsed_seconds(t_solve0, t_solve1);
                final_method = solved.method;
                step_norm = solved.step.norm();
                if (!solved.step.allFinite()) {
                    break;
                }

                accepted_alpha = 1.0;
                accepted_q = q + solved.step;
                accepted_norm = std::numeric_limits<double>::infinity();
                bool armijo_accepted = !line_search;

                if (line_search) {
                    const auto t_ls0 = clock_type::now();
                    double best_norm = std::numeric_limits<double>::infinity();
                    vector_type best_q = accepted_q;
                    const int n_search = std::max(1, max_line_search);
                    for (int search_iter = 0; search_iter < n_search; ++search_iter) {
                        const double alpha = std::pow(0.5, static_cast<double>(search_iter));
                        vector_type trial_q = q + alpha * solved.step;
                        TargetSystem trial_target = assemble_target(problem, trial_q);
                        const double trial_norm = trial_target.residual.norm();
                        if (trial_norm < best_norm) {
                            best_norm = trial_norm;
                            best_q = trial_q;
                        }
                        if (trial_norm <= (1.0 - sufficient_decrease * alpha) * final_norm) {
                            accepted_q = trial_q;
                            accepted_alpha = alpha;
                            accepted_norm = trial_norm;
                            armijo_accepted = true;
                            break;
                        }
                    }
                    const auto t_ls1 = clock_type::now();
                    problem.timings.line_search_seconds += elapsed_seconds(t_ls0, t_ls1);
                    if (!armijo_accepted) {
                        accepted_q = best_q;
                        accepted_alpha = 0.0;
                        accepted_norm = best_norm;
                    }
                } else if (adaptive_damping) {
                    TargetSystem trial_target = assemble_target(problem, accepted_q);
                    accepted_norm = trial_target.residual.norm();
                    armijo_accepted = accepted_norm < final_norm;
                }

                if (armijo_accepted || !adaptive_damping) {
                    step_accepted = true;
                    break;
                }

                ++rejected_step_count;
                current_damping = current_damping > 0.0 ? current_damping * damping_increase : 1.0e-12;
            }

            step_history.push_back(step_norm);
            alpha_history.push_back(step_accepted ? accepted_alpha : 0.0);
            damping_history.push_back(used_damping);
            if (!step_accepted) {
                break;
            }
            q = accepted_q;
            if (adaptive_damping && accepted_norm < final_norm && current_damping > 0.0) {
                current_damping *= damping_decrease;
            }
            if (step_norm <= step_tol * std::max(1.0, q.norm())) {
                TargetSystem final_target = assemble_target(problem, q);
                final_norm = final_target.residual.norm();
                residual_history.push_back(final_norm);
                converged = final_norm <= residual_tol;
                break;
            }
        }
        update_all_state(problem, q);
    }

    py::dict out;
    out["coefficients"] = vector_to_array(q);
    out["converged"] = converged;
    out["iterations"] = iterations;
    out["residual_norm"] = final_norm;
    out["linear_solver"] = final_method;
    out["residual_norm_history"] = std_vector_to_array(residual_history);
    out["step_norm_history"] = std_vector_to_array(step_history);
    out["line_search_alpha_history"] = std_vector_to_array(alpha_history);
    out["damping_history"] = std_vector_to_array(damping_history);
    out["rejected_step_count"] = rejected_step_count;
    out["timing_counters"] = timing_to_dict(problem.timings);
    out["backend"] = "cpp_native_online";
    return out;
}

py::dict solve_deim_online_gauss_newton(
    py::handle residual_metadata_capsule,
    py::sequence residual_param_order,
    py::dict residual_static_args,
    py::handle tangent_metadata_capsule,
    py::sequence tangent_param_order,
    py::dict tangent_static_args,
    py::handle trial_basis_obj,
    py::handle offset_obj,
    py::handle initial_coefficients_obj,
    py::handle row_dofs_obj,
    py::handle coefficient_arg_names_obj,
    py::handle selected_basis_obj,
    py::handle residual_terms_obj,
    py::handle element_weights_obj,
    py::handle row_weights_obj,
    py::handle gnat_lift_obj,
    py::sequence residual_state_updates_obj,
    py::sequence tangent_state_updates_obj,
    py::sequence residual_symbolic_state_updates_obj,
    py::sequence tangent_symbolic_state_updates_obj,
    int max_iterations,
    double residual_tol,
    double step_tol,
    double damping,
    bool adaptive_damping,
    int max_damping_retries,
    double damping_increase,
    double damping_decrease,
    bool line_search,
    int max_line_search,
    double sufficient_decrease,
    double rcond
)
{
    OnlineProblem problem;
    problem.residual_metadata = metadata_from_capsule(residual_metadata_capsule);
    problem.tangent_metadata = metadata_from_capsule(tangent_metadata_capsule);
    problem.residual_args = build_native_args(residual_param_order, residual_static_args);
    problem.tangent_args = build_native_args(tangent_param_order, tangent_static_args);
    problem.basis = as_matrix(trial_basis_obj, "trial_basis");
    problem.offset = as_vector(offset_obj, "offset");
    vector_type q = as_vector(initial_coefficients_obj, "initial_coefficients");
    if (problem.basis.rows() != problem.offset.size() || problem.basis.cols() != q.size()) {
        throw py::value_error("trial_basis, offset, and initial_coefficients have incompatible dimensions.");
    }
    problem.row_dofs = as_index_vector(row_dofs_obj, "row_dofs");
    if (problem.row_dofs.empty()) {
        throw py::value_error("row_dofs must not be empty.");
    }
    problem.coefficient_arg_names = string_list(coefficient_arg_names_obj.cast<py::sequence>());

    DeimTarget deim;
    const matrix_type selected_basis = as_matrix(selected_basis_obj, "selected_basis");
    if (selected_basis.rows() != static_cast<Eigen::Index>(problem.row_dofs.size())) {
        throw py::value_error("selected_basis row count must match row_dofs size.");
    }
    deim.interpolation_operator = interpolation_operator_from_selected_basis(selected_basis, rcond);
    deim.residual_terms = as_matrix(residual_terms_obj, "residual_terms");
    if (deim.residual_terms.rows() != selected_basis.cols()) {
        throw py::value_error("residual_terms row count must match selected_basis column count.");
    }
    if (deim.residual_terms.cols() == 0) {
        throw py::value_error("residual_terms must contain at least one target residual row.");
    }
    const Eigen::Index target_rows = deim.residual_terms.cols();

    if (!gnat_lift_obj.is_none() && py::isinstance<py::dict>(gnat_lift_obj)) {
        problem.sparse_lift = parse_csr_matrix(gnat_lift_obj.cast<py::dict>(), "gnat_lift");
        if (problem.sparse_lift.cols != static_cast<std::int64_t>(target_rows)) {
            throw py::value_error("sparse gnat_lift column count must match DEIM target residual size.");
        }
        problem.has_sparse_lift = true;
        problem.timings.sparse_lift_nonzeros = static_cast<std::int64_t>(problem.sparse_lift.data.size());
        problem.timings.sparse_lift_rows = problem.sparse_lift.rows;
        problem.timings.sparse_lift_cols = problem.sparse_lift.cols;
    } else if (!gnat_lift_obj.is_none()) {
        problem.lift = as_matrix(gnat_lift_obj, "gnat_lift");
        if (problem.lift.cols() != target_rows) {
            throw py::value_error("gnat_lift column count must match DEIM target residual size.");
        }
        problem.has_dense_lift = true;
    }

    update_all_state(problem, q);
    LocalBlocks probe = call_kernel(problem.tangent_metadata, problem.tangent_args);
    problem.element_weights = optional_weights(element_weights_obj, probe.n_entities, "element_weights");
    problem.has_row_weights = !row_weights_obj.is_none();
    problem.row_weights = optional_weights(row_weights_obj, static_cast<Eigen::Index>(problem.row_dofs.size()), "row_weights");
    problem.residual_state_updates = parse_affine_state_updates(residual_state_updates_obj, q.size());
    problem.tangent_state_updates = parse_affine_state_updates(tangent_state_updates_obj, q.size());
    problem.residual_symbolic_state_updates = parse_symbolic_state_updates(residual_symbolic_state_updates_obj);
    problem.tangent_symbolic_state_updates = parse_symbolic_state_updates(tangent_symbolic_state_updates_obj);

    if (!std::isfinite(residual_tol) || residual_tol < 0.0 || !std::isfinite(step_tol) || step_tol < 0.0) {
        throw py::value_error("residual_tol and step_tol must be finite and nonnegative.");
    }
    if (!std::isfinite(damping) || damping < 0.0) {
        throw py::value_error("damping must be finite and nonnegative.");
    }
    if (!std::isfinite(damping_increase) || damping_increase <= 1.0) {
        throw py::value_error("damping_increase must be finite and greater than one.");
    }
    if (!std::isfinite(damping_decrease) || damping_decrease <= 0.0 || damping_decrease > 1.0) {
        throw py::value_error("damping_decrease must be in the interval (0, 1].");
    }

    std::vector<double> residual_history;
    std::vector<double> step_history;
    std::vector<double> alpha_history;
    std::vector<double> damping_history;
    residual_history.reserve(static_cast<std::size_t>(std::max(1, max_iterations)));
    step_history.reserve(static_cast<std::size_t>(std::max(1, max_iterations)));
    alpha_history.reserve(static_cast<std::size_t>(std::max(1, max_iterations)));
    damping_history.reserve(static_cast<std::size_t>(std::max(1, max_iterations)));
    bool converged = false;
    int iterations = 0;
    int rejected_step_count = 0;
    std::string final_method = "";
    double final_norm = std::numeric_limits<double>::infinity();
    double current_damping = damping;

    {
        py::gil_scoped_release release;
        for (int iteration = 1; iteration <= std::max(1, max_iterations); ++iteration) {
            iterations = iteration;
            TargetSystem target = assemble_deim_target(problem, deim, q);
            final_norm = target.residual.norm();
            residual_history.push_back(final_norm);
            if (final_norm <= residual_tol) {
                step_history.push_back(0.0);
                alpha_history.push_back(0.0);
                damping_history.push_back(current_damping);
                converged = true;
                break;
            }

            StepSolve solved;
            double step_norm = std::numeric_limits<double>::infinity();
            double accepted_alpha = 1.0;
            double accepted_norm = std::numeric_limits<double>::infinity();
            double used_damping = current_damping;
            vector_type accepted_q = q;
            bool step_accepted = false;
            const int n_attempts = adaptive_damping ? std::max(1, max_damping_retries) : 1;
            for (int attempt = 0; attempt < n_attempts; ++attempt) {
                used_damping = current_damping;
                const auto t_solve0 = clock_type::now();
                solved = solve_step(target.jacobian, target.residual, current_damping, rcond);
                const auto t_solve1 = clock_type::now();
                problem.timings.step_solve_seconds += elapsed_seconds(t_solve0, t_solve1);
                final_method = solved.method;
                step_norm = solved.step.norm();
                if (!solved.step.allFinite()) {
                    break;
                }

                accepted_alpha = 1.0;
                accepted_q = q + solved.step;
                accepted_norm = std::numeric_limits<double>::infinity();
                bool armijo_accepted = !line_search;

                if (line_search) {
                    const auto t_ls0 = clock_type::now();
                    double best_norm = std::numeric_limits<double>::infinity();
                    vector_type best_q = accepted_q;
                    const int n_search = std::max(1, max_line_search);
                    for (int search_iter = 0; search_iter < n_search; ++search_iter) {
                        const double alpha = std::pow(0.5, static_cast<double>(search_iter));
                        vector_type trial_q = q + alpha * solved.step;
                        TargetSystem trial_target = assemble_deim_target(problem, deim, trial_q);
                        const double trial_norm = trial_target.residual.norm();
                        if (trial_norm < best_norm) {
                            best_norm = trial_norm;
                            best_q = trial_q;
                        }
                        if (trial_norm <= (1.0 - sufficient_decrease * alpha) * final_norm) {
                            accepted_q = trial_q;
                            accepted_alpha = alpha;
                            accepted_norm = trial_norm;
                            armijo_accepted = true;
                            break;
                        }
                    }
                    const auto t_ls1 = clock_type::now();
                    problem.timings.line_search_seconds += elapsed_seconds(t_ls0, t_ls1);
                    if (!armijo_accepted) {
                        accepted_q = best_q;
                        accepted_alpha = 0.0;
                        accepted_norm = best_norm;
                    }
                } else if (adaptive_damping) {
                    TargetSystem trial_target = assemble_deim_target(problem, deim, accepted_q);
                    accepted_norm = trial_target.residual.norm();
                    armijo_accepted = accepted_norm < final_norm;
                }

                if (armijo_accepted || !adaptive_damping) {
                    step_accepted = true;
                    break;
                }

                ++rejected_step_count;
                current_damping = current_damping > 0.0 ? current_damping * damping_increase : 1.0e-12;
            }

            step_history.push_back(step_norm);
            alpha_history.push_back(step_accepted ? accepted_alpha : 0.0);
            damping_history.push_back(used_damping);
            if (!step_accepted) {
                break;
            }
            q = accepted_q;
            if (adaptive_damping && accepted_norm < final_norm && current_damping > 0.0) {
                current_damping *= damping_decrease;
            }
            if (step_norm <= step_tol * std::max(1.0, q.norm())) {
                TargetSystem final_target = assemble_deim_target(problem, deim, q);
                final_norm = final_target.residual.norm();
                residual_history.push_back(final_norm);
                converged = final_norm <= residual_tol;
                break;
            }
        }
        update_all_state(problem, q);
    }

    py::dict out;
    out["coefficients"] = vector_to_array(q);
    out["converged"] = converged;
    out["iterations"] = iterations;
    out["residual_norm"] = final_norm;
    out["linear_solver"] = final_method;
    out["residual_norm_history"] = std_vector_to_array(residual_history);
    out["step_norm_history"] = std_vector_to_array(step_history);
    out["line_search_alpha_history"] = std_vector_to_array(alpha_history);
    out["damping_history"] = std_vector_to_array(damping_history);
    out["rejected_step_count"] = rejected_step_count;
    out["timing_counters"] = timing_to_dict(problem.timings);
    out["backend"] = "cpp_native_deim_online";
    return out;
}

}  // namespace

PYBIND11_MODULE(_pycutfem_mor_online_gauss_newton_2026_05_15_mor_online_gauss_newton_v5, m)
{
    m.doc() = "pycutfem MOR native online Gauss-Newton driver";
    m.def(
        "solve_online_gauss_newton",
        &solve_online_gauss_newton,
        py::arg("residual_metadata_capsule"),
        py::arg("residual_param_order"),
        py::arg("residual_static_args"),
        py::arg("tangent_metadata_capsule"),
        py::arg("tangent_param_order"),
        py::arg("tangent_static_args"),
        py::arg("trial_basis"),
        py::arg("offset"),
        py::arg("initial_coefficients"),
        py::arg("row_dofs"),
        py::arg("coefficient_arg_names"),
        py::arg("element_weights") = py::none(),
        py::arg("row_weights") = py::none(),
        py::arg("gnat_lift") = py::none(),
        py::arg("residual_state_updates") = py::tuple(),
        py::arg("tangent_state_updates") = py::tuple(),
        py::arg("residual_symbolic_state_updates") = py::tuple(),
        py::arg("tangent_symbolic_state_updates") = py::tuple(),
        py::arg("max_iterations") = 8,
        py::arg("residual_tol") = 1.0e-10,
        py::arg("step_tol") = 1.0e-12,
        py::arg("damping") = 0.0,
        py::arg("adaptive_damping") = false,
        py::arg("max_damping_retries") = 4,
        py::arg("damping_increase") = 10.0,
        py::arg("damping_decrease") = 0.25,
        py::arg("line_search") = false,
        py::arg("max_line_search") = 6,
        py::arg("sufficient_decrease") = 1.0e-4,
        py::arg("rcond") = -1.0
    );
    m.def(
        "solve_deim_online_gauss_newton",
        &solve_deim_online_gauss_newton,
        py::arg("residual_metadata_capsule"),
        py::arg("residual_param_order"),
        py::arg("residual_static_args"),
        py::arg("tangent_metadata_capsule"),
        py::arg("tangent_param_order"),
        py::arg("tangent_static_args"),
        py::arg("trial_basis"),
        py::arg("offset"),
        py::arg("initial_coefficients"),
        py::arg("row_dofs"),
        py::arg("coefficient_arg_names"),
        py::arg("selected_basis"),
        py::arg("residual_terms"),
        py::arg("element_weights") = py::none(),
        py::arg("row_weights") = py::none(),
        py::arg("gnat_lift") = py::none(),
        py::arg("residual_state_updates") = py::tuple(),
        py::arg("tangent_state_updates") = py::tuple(),
        py::arg("residual_symbolic_state_updates") = py::tuple(),
        py::arg("tangent_symbolic_state_updates") = py::tuple(),
        py::arg("max_iterations") = 8,
        py::arg("residual_tol") = 1.0e-10,
        py::arg("step_tol") = 1.0e-12,
        py::arg("damping") = 0.0,
        py::arg("adaptive_damping") = false,
        py::arg("max_damping_retries") = 4,
        py::arg("damping_increase") = 10.0,
        py::arg("damping_decrease") = 0.25,
        py::arg("line_search") = false,
        py::arg("max_line_search") = 6,
        py::arg("sufficient_decrease") = 1.0e-4,
        py::arg("rcond") = -1.0
    );
}
