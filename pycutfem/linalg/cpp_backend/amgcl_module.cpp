#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/range/iterator_range.hpp>

#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/adapter/zero_copy.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/value_type/static_matrix.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace py = pybind11;

namespace {

using index_type = std::int64_t;
using scalar_type = double;

struct SolverSettings {
    std::string preconditioner_type = "amg";
    std::string smoother_type = "ilu0";
    std::string krylov_type = "gmres";
    std::string coarsening_type = "aggregation";
    double tolerance = 1.0e-6;
    int max_iteration = 100;
    int gmres_krylov_space_dimension = 100;
    int verbosity = 1;
    bool scaling = false;
    int block_size = 1;
    bool use_block_matrices_if_possible = true;
    int coarse_enough = 1000;
    int max_levels = -1;
    int pre_sweeps = 1;
    int post_sweeps = 1;
};


struct CSRStorage {
    std::size_t rows = 0;
    std::vector<index_type> indptr;
    std::vector<index_type> indices;
    std::vector<scalar_type> values;
};


boost::property_tree::ptree BuildParameters(const SolverSettings& settings)
{
    if (settings.scaling) {
        throw std::runtime_error(
            "The pycutfem AMGCL wrapper does not implement the Kratos scaling wrapper."
        );
    }

    boost::property_tree::ptree params;
    params.put("precond.class", settings.preconditioner_type);
    params.put("solver.tol", settings.tolerance);
    params.put("solver.maxiter", settings.max_iteration);
    params.put("solver.type", settings.krylov_type);

    if (settings.krylov_type == "gmres" || settings.krylov_type == "fgmres" || settings.krylov_type == "lgmres") {
        params.put("solver.M", settings.gmres_krylov_space_dimension);
    }

    if (settings.preconditioner_type == "relaxation") {
        params.put("precond.relax.type", settings.smoother_type);
    }

    if (settings.preconditioner_type == "amg") {
        params.put("precond.relax.type", settings.smoother_type);
        params.put("precond.coarsening.type", settings.coarsening_type);
        if (settings.max_levels >= 0) {
            params.put("precond.max_levels", settings.max_levels);
        }
        params.put("precond.npre", settings.pre_sweeps);
        params.put("precond.npost", settings.post_sweeps);
        if (settings.coarsening_type != "ruge_stuben") {
            params.put("precond.coarsening.aggr.eps_strong", 0.0);
            params.put("precond.coarsening.aggr.block_size", 1);
        }
        params.put("precond.coarse_enough", std::max(1, settings.coarse_enough / std::max(1, settings.block_size)));
    }

    return params;
}


SolverSettings MakeSettings(
    std::string preconditioner_type,
    std::string smoother_type,
    std::string krylov_type,
    std::string coarsening_type,
    double tolerance,
    int max_iteration,
    int gmres_krylov_space_dimension,
    int verbosity,
    bool scaling,
    int block_size,
    bool use_block_matrices_if_possible,
    int coarse_enough,
    int max_levels,
    int pre_sweeps,
    int post_sweeps
)
{
    SolverSettings settings;
    settings.preconditioner_type = std::move(preconditioner_type);
    settings.smoother_type = std::move(smoother_type);
    settings.krylov_type = std::move(krylov_type);
    settings.coarsening_type = std::move(coarsening_type);
    settings.tolerance = tolerance;
    settings.max_iteration = max_iteration;
    settings.gmres_krylov_space_dimension = gmres_krylov_space_dimension;
    settings.verbosity = verbosity;
    settings.scaling = scaling;
    settings.block_size = std::max(1, block_size);
    settings.use_block_matrices_if_possible = use_block_matrices_if_possible;
    settings.coarse_enough = coarse_enough;
    settings.max_levels = max_levels;
    settings.pre_sweeps = pre_sweeps;
    settings.post_sweeps = post_sweeps;
    return settings;
}


CSRStorage MakeCSRStorage(
    py::array_t<index_type, py::array::c_style | py::array::forcecast> indptr_arr,
    py::array_t<index_type, py::array::c_style | py::array::forcecast> indices_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> data_arr
)
{
    auto indptr = indptr_arr.unchecked<1>();
    auto indices = indices_arr.unchecked<1>();
    auto data = data_arr.unchecked<1>();

    if (indptr.shape(0) < 2) {
        throw std::runtime_error("AMGCL CSR indptr must contain at least two entries.");
    }
    if (indices.shape(0) != data.shape(0)) {
        throw std::runtime_error("AMGCL CSR indices/data arrays must have the same length.");
    }

    CSRStorage csr;
    csr.rows = static_cast<std::size_t>(indptr.shape(0) - 1);
    csr.indptr.resize(static_cast<std::size_t>(indptr.shape(0)));
    for (ssize_t i = 0; i < indptr.shape(0); ++i) {
        csr.indptr[static_cast<std::size_t>(i)] = indptr(i);
    }
    csr.indices.resize(static_cast<std::size_t>(indices.shape(0)));
    for (ssize_t i = 0; i < indices.shape(0); ++i) {
        csr.indices[static_cast<std::size_t>(i)] = indices(i);
    }
    csr.values.resize(static_cast<std::size_t>(data.shape(0)));
    for (ssize_t i = 0; i < data.shape(0); ++i) {
        csr.values[static_cast<std::size_t>(i)] = data(i);
    }
    return csr;
}


std::vector<scalar_type> CopyDenseVector(
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> array
)
{
    auto view = array.unchecked<1>();
    std::vector<scalar_type> out(static_cast<std::size_t>(view.shape(0)));
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = view(i);
    }
    return out;
}


std::vector<scalar_type> CopyInitialGuess(py::object initial_guess_obj, std::size_t expected_size)
{
    std::vector<scalar_type> solution(expected_size, 0.0);
    if (!initial_guess_obj.is_none()) {
        auto guess_arr = py::array_t<scalar_type, py::array::c_style | py::array::forcecast>::ensure(initial_guess_obj);
        if (!guess_arr) {
            throw std::runtime_error("AMGCL initial guess must be convertible to a contiguous float64 array.");
        }
        auto guess = guess_arr.unchecked<1>();
        if (static_cast<std::size_t>(guess.shape(0)) != expected_size) {
            throw std::runtime_error("AMGCL initial guess length does not match the RHS length.");
        }
        for (ssize_t i = 0; i < guess.shape(0); ++i) {
            solution[static_cast<std::size_t>(i)] = guess(i);
        }
    }
    return solution;
}


py::array_t<scalar_type> MakeSolutionArray(const std::vector<scalar_type>& solution)
{
    py::array_t<scalar_type> solution_arr(solution.size());
    auto out = solution_arr.mutable_unchecked<1>();
    for (ssize_t i = 0; i < out.shape(0); ++i) {
        out(i) = solution[static_cast<std::size_t>(i)];
    }
    return solution_arr;
}


py::dict BuildResultDict(
    const std::vector<scalar_type>& solution,
    const std::tuple<std::size_t, scalar_type>& solve_result,
    const SolverSettings& settings
)
{
    py::dict result;
    result["solution"] = MakeSolutionArray(solution);
    result["iterations"] = static_cast<int>(std::get<0>(solve_result));
    result["residual_norm"] = static_cast<double>(std::get<1>(solve_result));
    result["converged"] = static_cast<bool>(std::get<1>(solve_result) < settings.tolerance);
    return result;
}


template <int BlockSize>
struct BackendTraits {
    using MatrixValue = std::conditional_t<
        BlockSize == 1,
        scalar_type,
        amgcl::static_matrix<scalar_type, BlockSize, BlockSize>
    >;
    using VectorValue = std::conditional_t<
        BlockSize == 1,
        scalar_type,
        amgcl::static_matrix<scalar_type, BlockSize, 1>
    >;
    using Backend = amgcl::backend::builtin<MatrixValue>;
    using Solver = amgcl::make_solver<
        amgcl::runtime::preconditioner<Backend>,
        amgcl::runtime::solver::wrapper<Backend>
    >;
};


template <int BlockSize>
class MatrixAdaptor {
public:
    explicit MatrixAdaptor(const CSRStorage& storage)
        : m_storage(storage)
    {
    }

    auto make()
    {
        if constexpr (BlockSize == 1) {
            return amgcl::adapter::zero_copy(
                m_storage.rows,
                m_storage.indptr.data(),
                m_storage.indices.data(),
                m_storage.values.data()
            );
        } else {
            using BlockType = typename BackendTraits<BlockSize>::MatrixValue;
            m_block_storage.emplace(
                m_storage.rows,
                m_storage.indptr,
                m_storage.indices,
                m_storage.values
            );
            return amgcl::adapter::block_matrix<BlockType>(m_block_storage.value());
        }
    }

private:
    const CSRStorage& m_storage;
    std::optional<std::tuple<
        std::size_t,
        const std::vector<index_type>&,
        const std::vector<index_type>&,
        const std::vector<scalar_type>&
    >> m_block_storage;
};


template <int BlockSize>
py::dict SolveWithBlockSize(
    const CSRStorage& csr,
    const std::vector<scalar_type>& rhs,
    std::vector<scalar_type> solution,
    const SolverSettings& settings
)
{
    using Traits = BackendTraits<BlockSize>;
    using SolverType = typename Traits::Solver;
    using VectorValue = typename Traits::VectorValue;

    const std::size_t rows = rhs.size();
    if (rows % BlockSize != 0) {
        throw std::runtime_error("AMGCL block size is incompatible with the RHS length.");
    }
    const std::size_t block_system_size = rows / static_cast<std::size_t>(BlockSize);
    const auto params = BuildParameters(settings);

    std::tuple<std::size_t, scalar_type> solve_result = {0, 0.0};
    MatrixAdaptor<BlockSize> matrix_adaptor(csr);
    auto matrix = matrix_adaptor.make();
    SolverType solver(matrix, params);
    auto* x_ptr = reinterpret_cast<VectorValue*>(solution.data());
    const auto* b_ptr = reinterpret_cast<const VectorValue*>(rhs.data());
    solve_result = solver(
        boost::make_iterator_range(b_ptr, b_ptr + block_system_size),
        boost::make_iterator_range(x_ptr, x_ptr + block_system_size)
    );

    py::array_t<scalar_type> solution_arr(solution.size());
    auto out = solution_arr.mutable_unchecked<1>();
    for (ssize_t i = 0; i < out.shape(0); ++i) {
        out(i) = solution[static_cast<std::size_t>(i)];
    }

    py::dict result;
    result["solution"] = std::move(solution_arr);
    result["iterations"] = static_cast<int>(std::get<0>(solve_result));
    result["residual_norm"] = static_cast<double>(std::get<1>(solve_result));
    result["converged"] = static_cast<bool>(std::get<1>(solve_result) < settings.tolerance);
    return result;
}


template <int BlockSize>
using SolverPointer = std::unique_ptr<typename BackendTraits<BlockSize>::Solver>;


using SolverVariant = std::variant<
    std::monostate,
    SolverPointer<1>,
    SolverPointer<2>,
    SolverPointer<3>,
    SolverPointer<4>,
    SolverPointer<5>,
    SolverPointer<6>
>;


class SolverHandle {
public:
    SolverHandle(CSRStorage csr, SolverSettings settings)
        : m_csr(std::move(csr))
        , m_settings(std::move(settings))
        , m_solver_block_size(m_settings.use_block_matrices_if_possible ? m_settings.block_size : 1)
    {
        if (m_csr.rows == 0) {
            throw std::runtime_error("AMGCL solver requires a non-empty matrix.");
        }
        if (m_csr.rows % static_cast<std::size_t>(m_solver_block_size) != 0) {
            throw std::runtime_error("AMGCL block size is incompatible with the matrix shape.");
        }
        Build();
    }

    py::dict solve(
        py::array_t<scalar_type, py::array::c_style | py::array::forcecast> rhs_arr,
        py::object initial_guess_obj = py::none()
    )
    {
        auto rhs = CopyDenseVector(rhs_arr);
        if (rhs.size() != m_csr.rows) {
            throw std::runtime_error("AMGCL CSR rhs length does not match the matrix shape.");
        }
        auto solution = CopyInitialGuess(initial_guess_obj, rhs.size());
        return SolveVectors(rhs, std::move(solution));
    }

private:
    template <int BlockSize>
    void BuildWithBlockSize()
    {
        using SolverType = typename BackendTraits<BlockSize>::Solver;
        const auto params = BuildParameters(m_settings);
        MatrixAdaptor<BlockSize> matrix_adaptor(m_csr);
        auto matrix = matrix_adaptor.make();
        SolverPointer<BlockSize> solver;
        {
            py::gil_scoped_release release;
            solver = std::make_unique<SolverType>(matrix, params);
        }
        m_solver = std::move(solver);
    }

    void Build()
    {
        switch (m_solver_block_size) {
        case 1:
            BuildWithBlockSize<1>();
            return;
        case 2:
            BuildWithBlockSize<2>();
            return;
        case 3:
            BuildWithBlockSize<3>();
            return;
        case 4:
            BuildWithBlockSize<4>();
            return;
        case 5:
            BuildWithBlockSize<5>();
            return;
        case 6:
            BuildWithBlockSize<6>();
            return;
        default:
            throw std::runtime_error("Unsupported AMGCL block size. Expected an integer in [1, 6].");
        }
    }

    template <int BlockSize>
    py::dict SolveWithCachedSolver(std::vector<scalar_type> rhs, std::vector<scalar_type> solution)
    {
        using VectorValue = typename BackendTraits<BlockSize>::VectorValue;

        const std::size_t rows = rhs.size();
        if (rows % BlockSize != 0) {
            throw std::runtime_error("AMGCL block size is incompatible with the RHS length.");
        }
        const std::size_t block_system_size = rows / static_cast<std::size_t>(BlockSize);

        auto* solver_ptr = std::get_if<SolverPointer<BlockSize>>(&m_solver);
        if (solver_ptr == nullptr || !(*solver_ptr)) {
            throw std::runtime_error("AMGCL cached solver is uninitialized.");
        }

        auto* x_ptr = reinterpret_cast<VectorValue*>(solution.data());
        const auto* b_ptr = reinterpret_cast<const VectorValue*>(rhs.data());
        std::tuple<std::size_t, scalar_type> solve_result = {0, 0.0};
        {
            py::gil_scoped_release release;
            solve_result = (**solver_ptr)(
                boost::make_iterator_range(b_ptr, b_ptr + block_system_size),
                boost::make_iterator_range(x_ptr, x_ptr + block_system_size)
            );
        }

        return BuildResultDict(solution, solve_result, m_settings);
    }

    py::dict SolveVectors(std::vector<scalar_type> rhs, std::vector<scalar_type> solution)
    {
        switch (m_solver_block_size) {
        case 1:
            return SolveWithCachedSolver<1>(std::move(rhs), std::move(solution));
        case 2:
            return SolveWithCachedSolver<2>(std::move(rhs), std::move(solution));
        case 3:
            return SolveWithCachedSolver<3>(std::move(rhs), std::move(solution));
        case 4:
            return SolveWithCachedSolver<4>(std::move(rhs), std::move(solution));
        case 5:
            return SolveWithCachedSolver<5>(std::move(rhs), std::move(solution));
        case 6:
            return SolveWithCachedSolver<6>(std::move(rhs), std::move(solution));
        default:
            throw std::runtime_error("Unsupported AMGCL block size. Expected an integer in [1, 6].");
        }
    }

    CSRStorage m_csr;
    SolverSettings m_settings;
    int m_solver_block_size = 1;
    SolverVariant m_solver;
};


py::dict SolveCSR(
    py::array_t<index_type, py::array::c_style | py::array::forcecast> indptr_arr,
    py::array_t<index_type, py::array::c_style | py::array::forcecast> indices_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> data_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> rhs_arr,
    py::object initial_guess_obj,
    std::string preconditioner_type,
    std::string smoother_type,
    std::string krylov_type,
    std::string coarsening_type,
    double tolerance,
    int max_iteration,
    int gmres_krylov_space_dimension,
    int verbosity,
    bool scaling,
    int block_size,
    bool use_block_matrices_if_possible,
    int coarse_enough,
    int max_levels,
    int pre_sweeps,
    int post_sweeps
)
{
    SolverHandle solver(
        MakeCSRStorage(indptr_arr, indices_arr, data_arr),
        MakeSettings(
            std::move(preconditioner_type),
            std::move(smoother_type),
            std::move(krylov_type),
            std::move(coarsening_type),
            tolerance,
            max_iteration,
            gmres_krylov_space_dimension,
            verbosity,
            scaling,
            block_size,
            use_block_matrices_if_possible,
            coarse_enough,
            max_levels,
            pre_sweeps,
            post_sweeps
        )
    );
    return solver.solve(rhs_arr, initial_guess_obj);
}

}  // namespace


PYBIND11_MODULE(_pycutfem_cpp_amgcl_2026_04_22_amgcl_v5, m)
{
    m.doc() = "pycutfem AMGCL linear solver wrapper";
    py::class_<SolverHandle>(m, "AMGCLSolverHandle")
        .def(
            py::init([](
                py::array_t<index_type, py::array::c_style | py::array::forcecast> indptr_arr,
                py::array_t<index_type, py::array::c_style | py::array::forcecast> indices_arr,
                py::array_t<scalar_type, py::array::c_style | py::array::forcecast> data_arr,
                std::string preconditioner_type,
                std::string smoother_type,
                std::string krylov_type,
                std::string coarsening_type,
                double tolerance,
                int max_iteration,
                int gmres_krylov_space_dimension,
                int verbosity,
                bool scaling,
                int block_size,
                bool use_block_matrices_if_possible,
                int coarse_enough,
                int max_levels,
                int pre_sweeps,
                int post_sweeps
            ) {
                return std::make_unique<SolverHandle>(
                    MakeCSRStorage(indptr_arr, indices_arr, data_arr),
                    MakeSettings(
                        std::move(preconditioner_type),
                        std::move(smoother_type),
                        std::move(krylov_type),
                        std::move(coarsening_type),
                        tolerance,
                        max_iteration,
                        gmres_krylov_space_dimension,
                        verbosity,
                        scaling,
                        block_size,
                        use_block_matrices_if_possible,
                        coarse_enough,
                        max_levels,
                        pre_sweeps,
                        post_sweeps
                    )
                );
            }),
            py::arg("indptr"),
            py::arg("indices"),
            py::arg("data"),
            py::kw_only(),
            py::arg("preconditioner_type") = "amg",
            py::arg("smoother_type") = "ilu0",
            py::arg("krylov_type") = "gmres",
            py::arg("coarsening_type") = "aggregation",
            py::arg("tolerance") = 1.0e-6,
            py::arg("max_iteration") = 100,
            py::arg("gmres_krylov_space_dimension") = 100,
            py::arg("verbosity") = 1,
            py::arg("scaling") = false,
            py::arg("block_size") = 1,
            py::arg("use_block_matrices_if_possible") = true,
            py::arg("coarse_enough") = 1000,
            py::arg("max_levels") = -1,
            py::arg("pre_sweeps") = 1,
            py::arg("post_sweeps") = 1
        )
        .def(
            "solve",
            &SolverHandle::solve,
            py::arg("rhs"),
            py::arg("initial_guess") = py::none()
        );
    m.def(
        "solve_csr",
        &SolveCSR,
        py::arg("indptr"),
        py::arg("indices"),
        py::arg("data"),
        py::arg("rhs"),
        py::arg("initial_guess") = py::none(),
        py::kw_only(),
        py::arg("preconditioner_type") = "amg",
        py::arg("smoother_type") = "ilu0",
        py::arg("krylov_type") = "gmres",
        py::arg("coarsening_type") = "aggregation",
        py::arg("tolerance") = 1.0e-6,
        py::arg("max_iteration") = 100,
        py::arg("gmres_krylov_space_dimension") = 100,
        py::arg("verbosity") = 1,
        py::arg("scaling") = false,
        py::arg("block_size") = 1,
        py::arg("use_block_matrices_if_possible") = true,
        py::arg("coarse_enough") = 1000,
        py::arg("max_levels") = -1,
        py::arg("pre_sweeps") = 1,
        py::arg("post_sweeps") = 1
    );
}
