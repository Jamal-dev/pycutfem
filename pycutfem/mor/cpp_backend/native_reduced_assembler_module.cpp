#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <Python.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "native_kernel.hpp"

namespace py = pybind11;
using namespace pycutfem::cpp_backend;

namespace {

using scalar_type = double;
using matrix_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>;
using vector_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>;

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

struct NativeArgBundle {
    std::vector<py::array> keepalive;
    std::vector<NativeArrayView> views;
    std::vector<std::string> name_storage;
    std::vector<const char*> names;
    KernelStaticArgs static_args;
};

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
void push_arg(NativeArgBundle& bundle, const std::string& name, py::array_t<T, py::array::c_style | py::array::forcecast> arr)
{
    NativeArrayView view;
    view.data = arr.mutable_data();
    view.dtype = NativeDType<T>::value;
    view.ndim = static_cast<std::int32_t>(arr.ndim());
    for (ssize_t axis = 0; axis < arr.ndim(); ++axis) {
        view.shape[axis] = static_cast<std::int64_t>(arr.shape(axis));
        view.strides[axis] = static_cast<std::int64_t>(arr.strides(axis));
    }
    bundle.keepalive.emplace_back(std::move(arr));
    bundle.views.push_back(view);
    bundle.name_storage.push_back(name);
}

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
            push_arg<std::int32_t>(bundle, name, as_array_for_arg<std::int32_t>(value, name));
        } else if (is_uint8_arg(name)) {
            push_arg<std::uint8_t>(bundle, name, as_array_for_arg<std::uint8_t>(value, name));
        } else {
            push_arg<double>(bundle, name, as_array_for_arg<double>(value, name));
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

py::tuple call_native_kernel(py::handle metadata_capsule, py::sequence param_order, py::dict static_args)
{
    NativeKernelMetadata* metadata = metadata_from_capsule(metadata_capsule);
    NativeArgBundle bundle = build_native_args(param_order, static_args);
    KernelMutableArgs mutable_args;
    KernelOutputs outputs;
    {
        py::gil_scoped_release release;
        metadata->entrypoint(bundle.static_args, mutable_args, outputs);
    }
    if (outputs.n_entities < 0 || outputs.n_local_dofs < 0 || outputs.functional_dim < 0) {
        throw std::runtime_error("Native kernel returned invalid output dimensions.");
    }

    py::array_t<double> K({outputs.n_entities, outputs.n_local_dofs, outputs.n_local_dofs});
    py::array_t<double> F({outputs.n_entities, outputs.n_local_dofs});
    py::array_t<double> J({outputs.n_entities, outputs.functional_dim});
    outputs.K = static_cast<double*>(K.mutable_data());
    outputs.F = static_cast<double*>(F.mutable_data());
    outputs.J = static_cast<double*>(J.mutable_data());
    {
        py::gil_scoped_release release;
        metadata->entrypoint(bundle.static_args, mutable_args, outputs);
    }
    return py::make_tuple(K, F, J);
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
            out(i, j) = view(i, j);
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
        out(i) = view(i);
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
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        if (view(i) < 0) {
            throw py::value_error(std::string(label) + " contains negative entries.");
        }
        out[static_cast<std::size_t>(i)] = view(i);
    }
    return out;
}

vector_type optional_weights(py::handle obj, Eigen::Index n)
{
    vector_type weights(n);
    weights.setOnes();
    if (obj.is_none()) {
        return weights;
    }
    auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr || arr.ndim() != 1 || arr.shape(0) != n) {
        throw py::value_error("element_weights must be a rank-1 array matching the element count.");
    }
    auto view = arr.unchecked<1>();
    for (Eigen::Index i = 0; i < n; ++i) {
        if (view(i) < 0.0) {
            throw py::value_error("element_weights must be nonnegative.");
        }
        weights(i) = view(i);
    }
    return weights;
}

py::tuple sampled_lspg_rows_from_native_kernel(
    py::handle metadata_capsule,
    py::sequence param_order,
    py::dict static_args,
    py::handle row_dofs_obj,
    py::handle trial_basis_obj,
    py::handle element_weights_obj
)
{
    py::tuple local = call_native_kernel(metadata_capsule, param_order, static_args);
    auto K_arr = local[0].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    auto F_arr = local[1].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    auto gdofs_arr = as_array_for_arg<std::int32_t>(static_args[py::str("gdofs_map")], "gdofs_map");
    const matrix_type basis = as_matrix(trial_basis_obj, "trial_basis");
    const std::vector<std::int64_t> rows = as_index_vector(row_dofs_obj, "row_dofs");
    const vector_type weights = optional_weights(element_weights_obj, K_arr.shape(0));

    std::vector<std::int64_t> row_positions(static_cast<std::size_t>(basis.rows()), -1);
    for (std::size_t s = 0; s < rows.size(); ++s) {
        if (rows[s] >= basis.rows()) {
            throw py::value_error("row_dofs contains out-of-range entries.");
        }
        row_positions[static_cast<std::size_t>(rows[s])] = static_cast<std::int64_t>(s);
    }

    auto K = K_arr.unchecked<3>();
    auto F = F_arr.unchecked<2>();
    auto gdofs = gdofs_arr.unchecked<2>();
    vector_type residual = vector_type::Zero(static_cast<Eigen::Index>(rows.size()));
    matrix_type trial = matrix_type::Zero(static_cast<Eigen::Index>(rows.size()), basis.cols());
    for (ssize_t e = 0; e < K.shape(0); ++e) {
        const double weight = weights(e);
        for (ssize_t i = 0; i < K.shape(1); ++i) {
            const auto sample = row_positions[static_cast<std::size_t>(gdofs(e, i))];
            if (sample < 0) {
                continue;
            }
            residual(sample) += weight * F(e, i);
            for (ssize_t j = 0; j < K.shape(2); ++j) {
                const auto gj = gdofs(e, j);
                const double kij = weight * K(e, i, j);
                for (Eigen::Index m = 0; m < basis.cols(); ++m) {
                    trial(sample, m) += kij * basis(gj, m);
                }
            }
        }
    }
    return py::make_tuple(vector_to_array(residual), matrix_to_array(trial));
}

py::tuple sampled_galerkin_reduced_system_from_native_kernel(
    py::handle metadata_capsule,
    py::sequence param_order,
    py::dict static_args,
    py::handle trial_basis_obj,
    py::handle element_weights_obj
)
{
    py::tuple local = call_native_kernel(metadata_capsule, param_order, static_args);
    auto K_arr = local[0].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    auto F_arr = local[1].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    auto gdofs_arr = as_array_for_arg<std::int32_t>(static_args[py::str("gdofs_map")], "gdofs_map");
    const matrix_type basis = as_matrix(trial_basis_obj, "trial_basis");
    const vector_type weights = optional_weights(element_weights_obj, K_arr.shape(0));

    auto K = K_arr.unchecked<3>();
    auto F = F_arr.unchecked<2>();
    auto gdofs = gdofs_arr.unchecked<2>();
    vector_type residual = vector_type::Zero(basis.cols());
    matrix_type tangent = matrix_type::Zero(basis.cols(), basis.cols());
    for (ssize_t e = 0; e < K.shape(0); ++e) {
        const double weight = weights(e);
        for (ssize_t i = 0; i < K.shape(1); ++i) {
            const auto gi = gdofs(e, i);
            const double fi = weight * F(e, i);
            for (Eigen::Index m = 0; m < basis.cols(); ++m) {
                residual(m) += basis(gi, m) * fi;
            }
            for (ssize_t j = 0; j < K.shape(2); ++j) {
                const auto gj = gdofs(e, j);
                const double kij = weight * K(e, i, j);
                for (Eigen::Index m = 0; m < basis.cols(); ++m) {
                    const double left = basis(gi, m) * kij;
                    for (Eigen::Index n = 0; n < basis.cols(); ++n) {
                        tangent(m, n) += left * basis(gj, n);
                    }
                }
            }
        }
    }
    return py::make_tuple(vector_to_array(residual), matrix_to_array(tangent));
}

py::tuple gnat_system_from_native_kernel(
    py::handle metadata_capsule,
    py::sequence param_order,
    py::dict static_args,
    py::handle row_dofs_obj,
    py::handle trial_basis_obj,
    py::handle sample_to_residual_coefficients_obj,
    py::handle element_weights_obj
)
{
    py::tuple sampled = sampled_lspg_rows_from_native_kernel(
        metadata_capsule,
        param_order,
        static_args,
        row_dofs_obj,
        trial_basis_obj,
        element_weights_obj
    );
    const vector_type sampled_residual = as_vector(sampled[0], "sampled_residual");
    const matrix_type sampled_trial = as_matrix(sampled[1], "sampled_trial_jacobian");
    const matrix_type lift = as_matrix(sample_to_residual_coefficients_obj, "sample_to_residual_coefficients");
    if (lift.cols() != sampled_residual.size() || lift.cols() != sampled_trial.rows()) {
        throw py::value_error("GNAT lift columns must match sampled residual/Jacobian rows.");
    }
    return py::make_tuple(
        vector_to_array(lift * sampled_residual),
        matrix_to_array(lift * sampled_trial)
    );
}

}  // namespace

PYBIND11_MODULE(_pycutfem_mor_native_reduced_assembler_2026_05_15_mor_native_reduced_assembler_v1, m)
{
    m.doc() = "pycutfem MOR native reduced assembler backend";
    m.def("call_native_kernel", &call_native_kernel, py::arg("metadata_capsule"), py::arg("param_order"), py::arg("static_args"));
    m.def(
        "sampled_lspg_rows_from_native_kernel",
        &sampled_lspg_rows_from_native_kernel,
        py::arg("metadata_capsule"),
        py::arg("param_order"),
        py::arg("static_args"),
        py::arg("row_dofs"),
        py::arg("trial_basis"),
        py::arg("element_weights") = py::none()
    );
    m.def(
        "sampled_galerkin_reduced_system_from_native_kernel",
        &sampled_galerkin_reduced_system_from_native_kernel,
        py::arg("metadata_capsule"),
        py::arg("param_order"),
        py::arg("static_args"),
        py::arg("trial_basis"),
        py::arg("element_weights") = py::none()
    );
    m.def(
        "gnat_system_from_native_kernel",
        &gnat_system_from_native_kernel,
        py::arg("metadata_capsule"),
        py::arg("param_order"),
        py::arg("static_args"),
        py::arg("row_dofs"),
        py::arg("trial_basis"),
        py::arg("sample_to_residual_coefficients"),
        py::arg("element_weights") = py::none()
    );
}
