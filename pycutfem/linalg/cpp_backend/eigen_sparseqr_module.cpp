#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

namespace {

using index_type = std::int64_t;
using scalar_type = double;
using storage_index_type = int;
using matrix_type = Eigen::SparseMatrix<scalar_type, Eigen::RowMajor, storage_index_type>;
using vector_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>;
using solver_type = Eigen::SparseQR<matrix_type, Eigen::COLAMDOrdering<storage_index_type>>;

matrix_type make_matrix(
    py::array_t<index_type, py::array::c_style | py::array::forcecast> indptr_arr,
    py::array_t<index_type, py::array::c_style | py::array::forcecast> indices_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> data_arr
) {
    auto indptr = indptr_arr.unchecked<1>();
    auto indices = indices_arr.unchecked<1>();
    auto data = data_arr.unchecked<1>();

    if (indptr.shape(0) < 2) {
        throw std::runtime_error("Eigen SparseQR CSR indptr must contain at least two entries.");
    }
    if (indices.shape(0) != data.shape(0)) {
        throw std::runtime_error("Eigen SparseQR CSR indices/data arrays must have the same length.");
    }

    const auto rows = static_cast<storage_index_type>(indptr.shape(0) - 1);
    matrix_type matrix(rows, rows);
    matrix.resizeNonZeros(static_cast<storage_index_type>(data.shape(0)));

    auto* outer = matrix.outerIndexPtr();
    auto* inner = matrix.innerIndexPtr();
    auto* values = matrix.valuePtr();

    for (ssize_t i = 0; i < indptr.shape(0); ++i) {
        const auto value = indptr(i);
        if (value < 0 || value > static_cast<index_type>(data.shape(0))) {
            throw std::runtime_error("Eigen SparseQR CSR indptr contains an out-of-range entry.");
        }
        outer[i] = static_cast<storage_index_type>(value);
    }
    for (ssize_t i = 0; i < indices.shape(0); ++i) {
        const auto value = indices(i);
        if (value < 0 || value >= static_cast<index_type>(rows)) {
            throw std::runtime_error("Eigen SparseQR CSR indices contain an out-of-range entry.");
        }
        inner[i] = static_cast<storage_index_type>(value);
        values[i] = static_cast<scalar_type>(data(i));
    }

    matrix.finalize();
    matrix.makeCompressed();
    return matrix;
}

vector_type make_rhs(py::array_t<scalar_type, py::array::c_style | py::array::forcecast> rhs_arr) {
    auto rhs = rhs_arr.unchecked<1>();
    vector_type out(rhs.shape(0));
    for (ssize_t i = 0; i < rhs.shape(0); ++i) {
        out[i] = static_cast<scalar_type>(rhs(i));
    }
    return out;
}

py::array_t<scalar_type> make_solution_array(const vector_type& solution) {
    py::array_t<scalar_type> out(solution.size());
    auto view = out.mutable_unchecked<1>();
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        view(i) = solution[i];
    }
    return out;
}

py::dict build_result(const vector_type& solution, const solver_type& solver) {
    py::dict result;
    result["solution"] = make_solution_array(solution);
    result["rank"] = static_cast<int>(solver.rank());
    result["converged"] = static_cast<bool>(solver.info() == Eigen::Success);
    return result;
}

class EigenSparseQRHandle {
public:
    EigenSparseQRHandle(
        py::array_t<index_type, py::array::c_style | py::array::forcecast> indptr_arr,
        py::array_t<index_type, py::array::c_style | py::array::forcecast> indices_arr,
        py::array_t<scalar_type, py::array::c_style | py::array::forcecast> data_arr
    )
        : m_matrix(make_matrix(indptr_arr, indices_arr, data_arr))
    {
        m_solver.compute(m_matrix);
        if (m_solver.info() != Eigen::Success) {
            throw std::runtime_error("Eigen SparseQR factorization failed.");
        }
    }

    py::dict solve(py::array_t<scalar_type, py::array::c_style | py::array::forcecast> rhs_arr) const {
        auto rhs = make_rhs(rhs_arr);
        if (rhs.size() != m_matrix.rows()) {
            throw std::runtime_error("Eigen SparseQR RHS size does not match the matrix size.");
        }
        auto solution = m_solver.solve(rhs);
        if (m_solver.info() != Eigen::Success) {
            throw std::runtime_error("Eigen SparseQR solve failed.");
        }
        return build_result(solution, m_solver);
    }

private:
    matrix_type m_matrix;
    solver_type m_solver;
};

py::dict solve_csr(
    py::array_t<index_type, py::array::c_style | py::array::forcecast> indptr_arr,
    py::array_t<index_type, py::array::c_style | py::array::forcecast> indices_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> data_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> rhs_arr
) {
    auto matrix = make_matrix(indptr_arr, indices_arr, data_arr);
    auto rhs = make_rhs(rhs_arr);
    if (rhs.size() != matrix.rows()) {
        throw std::runtime_error("Eigen SparseQR RHS size does not match the matrix size.");
    }
    solver_type solver;
    solver.compute(matrix);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen SparseQR factorization failed.");
    }
    auto solution = solver.solve(rhs);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen SparseQR solve failed.");
    }
    return build_result(solution, solver);
}

}  // namespace

PYBIND11_MODULE(_pycutfem_cpp_eigen_sparseqr_2026_04_22_eigen_sparseqr_v1, m) {
    m.doc() = "pycutfem Eigen SparseQR direct solver";
    m.def("solve_csr", &solve_csr, py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("rhs"));
    py::class_<EigenSparseQRHandle>(m, "EigenSparseQRHandle")
        .def(
            py::init<
                py::array_t<index_type, py::array::c_style | py::array::forcecast>,
                py::array_t<index_type, py::array::c_style | py::array::forcecast>,
                py::array_t<scalar_type, py::array::c_style | py::array::forcecast>
            >(),
            py::arg("indptr"),
            py::arg("indices"),
            py::arg("data")
        )
        .def("solve", &EigenSparseQRHandle::solve, py::arg("rhs"));
}
