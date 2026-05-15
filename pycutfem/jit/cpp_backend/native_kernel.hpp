#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace pycutfem::cpp_backend {

inline constexpr const char* NATIVE_KERNEL_ABI = "2026-05-15-native-kernel-v1";
inline constexpr const char* NATIVE_KERNEL_METADATA_CAPSULE_NAME =
    "pycutfem.native_kernel_metadata.v1";

enum class NativeArrayDType : std::int32_t {
    Float64 = 1,
    Int32 = 2,
    UInt8 = 3,
};

struct NativeArrayView {
    void* data = nullptr;
    NativeArrayDType dtype = NativeArrayDType::Float64;
    std::int32_t ndim = 0;
    std::int64_t shape[6] = {0, 0, 0, 0, 0, 0};
    std::int64_t strides[6] = {0, 0, 0, 0, 0, 0};
};

struct KernelStaticArgs {
    const NativeArrayView* arrays = nullptr;
    const char* const* names = nullptr;
    std::size_t count = 0;
};

struct KernelMutableArgs {
    NativeArrayView* arrays = nullptr;
    const char* const* names = nullptr;
    std::size_t count = 0;
};

struct KernelOutputs {
    double* K = nullptr;
    double* F = nullptr;
    double* J = nullptr;
    std::int64_t n_entities = 0;
    std::int64_t n_local_dofs = 0;
    std::int64_t functional_dim = 0;
    std::int32_t form_rank = 0;
};

using NativeKernelFn = void (*)(
    const KernelStaticArgs& static_args,
    KernelMutableArgs& mutable_args,
    KernelOutputs& outputs
);

struct NativeKernelMetadata {
    const char* abi = NATIVE_KERNEL_ABI;
    const char* name = nullptr;
    std::int32_t form_rank = 0;
    bool on_facet = false;
    const char* const* parameter_names = nullptr;
    std::size_t parameter_count = 0;
    const char* const* active_fields = nullptr;
    std::size_t active_field_count = 0;
    std::int64_t functional_dim = 1;
    NativeKernelFn entrypoint = nullptr;
};

template <typename T>
struct NativeDType;

template <>
struct NativeDType<double> {
    static constexpr NativeArrayDType value = NativeArrayDType::Float64;
};

template <>
struct NativeDType<std::int32_t> {
    static constexpr NativeArrayDType value = NativeArrayDType::Int32;
};

template <>
struct NativeDType<std::uint8_t> {
    static constexpr NativeArrayDType value = NativeArrayDType::UInt8;
};

template <typename T, int Rank>
struct NativeTensorView {
    T* ptr = nullptr;
    std::int64_t dims[Rank] = {};
    std::int64_t step[Rank] = {};

    NativeTensorView() = default;

    explicit NativeTensorView(T* data, const std::int64_t (&shape_values)[Rank])
        : ptr(data)
    {
        std::int64_t stride = 1;
        for (int axis = Rank - 1; axis >= 0; --axis) {
            dims[axis] = shape_values[axis];
            step[axis] = stride;
            stride *= shape_values[axis];
        }
    }

    NativeTensorView(T* data, const std::int64_t (&shape_values)[Rank], const std::int64_t (&stride_values)[Rank])
        : ptr(data)
    {
        for (int axis = 0; axis < Rank; ++axis) {
            dims[axis] = shape_values[axis];
            step[axis] = stride_values[axis];
        }
    }

    std::int64_t shape(int axis) const
    {
        return dims[axis];
    }

    template <typename... Indices>
    T& operator()(Indices... indices) const
    {
        static_assert(sizeof...(Indices) == Rank, "NativeTensorView rank mismatch");
        const std::int64_t idx[Rank] = {static_cast<std::int64_t>(indices)...};
        std::int64_t offset = 0;
        for (int axis = 0; axis < Rank; ++axis) {
            offset += idx[axis] * step[axis];
        }
        return ptr[offset];
    }
};

inline const NativeArrayView& native_find_arg(
    const KernelStaticArgs& args,
    const char* name
)
{
    for (std::size_t i = 0; i < args.count; ++i) {
        if (args.names != nullptr && args.names[i] != nullptr && std::strcmp(args.names[i], name) == 0) {
            return args.arrays[i];
        }
    }
    throw std::invalid_argument(std::string("Missing native kernel argument: ") + name);
}

template <typename T, int Rank>
NativeTensorView<const T, Rank> native_static_arg(
    const KernelStaticArgs& args,
    const char* name
)
{
    const NativeArrayView& raw = native_find_arg(args, name);
    if (raw.dtype != NativeDType<T>::value) {
        throw std::invalid_argument(std::string("Native kernel argument has wrong dtype: ") + name);
    }
    if (raw.ndim != Rank) {
        throw std::invalid_argument(std::string("Native kernel argument has wrong rank: ") + name);
    }
    std::int64_t shape_values[Rank] = {};
    std::int64_t stride_values[Rank] = {};
    for (int axis = 0; axis < Rank; ++axis) {
        shape_values[axis] = raw.shape[axis];
        stride_values[axis] = raw.strides[axis] / static_cast<std::int64_t>(sizeof(T));
    }
    return NativeTensorView<const T, Rank>(
        static_cast<const T*>(raw.data),
        shape_values,
        stride_values
    );
}

}  // namespace pycutfem::cpp_backend
