#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace {

using index_type = std::int64_t;
using scalar_type = double;
using storage_index_type = int;
using matrix_type = Eigen::SparseMatrix<scalar_type, Eigen::RowMajor, storage_index_type>;
using vector_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>;
using solver_type = Eigen::SparseQR<matrix_type, Eigen::COLAMDOrdering<storage_index_type>>;
using triplet_type = Eigen::Triplet<scalar_type, storage_index_type>;

matrix_type build_structural_similarity_matrix_from_buffers(
    const scalar_type* node_coords,
    const storage_index_type nnodes,
    const index_type* connectivity,
    const storage_index_type nelem,
    const double poisson,
    const double factor,
    const double xi
) {
    const storage_index_type ndof = 2 * nnodes;
    std::vector<triplet_type> triplets;
    triplets.reserve(static_cast<std::size_t>(nelem) * 36u);

    constexpr scalar_type dn_de[3][2] = {
        {-1.0, -1.0},
        { 1.0,  0.0},
        { 0.0,  1.0},
    };
    constexpr scalar_type ref_weight = 0.5;
    constexpr scalar_type eps = 1.0e-30;

    for (storage_index_type e = 0; e < nelem; ++e) {
        const storage_index_type n0 = static_cast<storage_index_type>(connectivity[3 * e + 0]);
        const storage_index_type n1 = static_cast<storage_index_type>(connectivity[3 * e + 1]);
        const storage_index_type n2 = static_cast<storage_index_type>(connectivity[3 * e + 2]);
        if (n0 < 0 || n0 >= nnodes || n1 < 0 || n1 >= nnodes || n2 < 0 || n2 >= nnodes) {
            throw std::runtime_error("connectivity contains an out-of-range node index.");
        }

        const scalar_type x0 = node_coords[2 * n0 + 0];
        const scalar_type y0 = node_coords[2 * n0 + 1];
        const scalar_type x1 = node_coords[2 * n1 + 0];
        const scalar_type y1 = node_coords[2 * n1 + 1];
        const scalar_type x2 = node_coords[2 * n2 + 0];
        const scalar_type y2 = node_coords[2 * n2 + 1];

        const scalar_type j00 = x1 - x0;
        const scalar_type j01 = x2 - x0;
        const scalar_type j10 = y1 - y0;
        const scalar_type j11 = y2 - y0;
        const scalar_type det_j = j00 * j11 - j01 * j10;
        if (!(det_j > eps)) {
            throw std::runtime_error("Degenerate Triangle2D3 mesh-moving element encountered.");
        }

        const scalar_type inv_det = 1.0 / det_j;
        const scalar_type inv_j00 =  inv_det * j11;
        const scalar_type inv_j01 = -inv_det * j01;
        const scalar_type inv_j10 = -inv_det * j10;
        const scalar_type inv_j11 =  inv_det * j00;

        scalar_type dn_dx[3][2];
        for (int i = 0; i < 3; ++i) {
            dn_dx[i][0] = dn_de[i][0] * inv_j00 + dn_de[i][1] * inv_j10;
            dn_dx[i][1] = dn_de[i][0] * inv_j01 + dn_de[i][1] * inv_j11;
        }

        scalar_type B[3][6] = {};
        for (int i = 0; i < 3; ++i) {
            B[0][2 * i + 0] = dn_dx[i][0];
            B[1][2 * i + 1] = dn_dx[i][1];
            B[2][2 * i + 0] = dn_dx[i][1];
            B[2][2 * i + 1] = dn_dx[i][0];
        }

        const scalar_type weighting = det_j * std::pow(factor / det_j, xi);
        const scalar_type lambda = weighting * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
        const scalar_type mu = weighting / (2.0 * (1.0 + poisson));
        const scalar_type C[3][3] = {
            {lambda + 2.0 * mu, lambda, 0.0},
            {lambda, lambda + 2.0 * mu, 0.0},
            {0.0, 0.0, mu},
        };

        scalar_type CB[3][6] = {};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 6; ++j) {
                for (int k = 0; k < 3; ++k) {
                    CB[i][j] += C[i][k] * B[k][j];
                }
            }
        }

        scalar_type K[6][6] = {};
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                for (int k = 0; k < 3; ++k) {
                    K[i][j] += B[k][i] * CB[k][j];
                }
                K[i][j] *= ref_weight;
            }
        }

        const storage_index_type gdofs[6] = {
            static_cast<storage_index_type>(2 * n0 + 0),
            static_cast<storage_index_type>(2 * n0 + 1),
            static_cast<storage_index_type>(2 * n1 + 0),
            static_cast<storage_index_type>(2 * n1 + 1),
            static_cast<storage_index_type>(2 * n2 + 0),
            static_cast<storage_index_type>(2 * n2 + 1),
        };
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                triplets.emplace_back(gdofs[i], gdofs[j], K[i][j]);
            }
        }
    }

    matrix_type A(ndof, ndof);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();
    return A;
}

matrix_type build_structural_similarity_matrix(
    const py::array_t<scalar_type, py::array::c_style | py::array::forcecast>& node_coords_arr,
    const py::array_t<index_type, py::array::c_style | py::array::forcecast>& connectivity_arr,
    const double poisson,
    const double factor,
    const double xi
) {
    auto node_coords = node_coords_arr.unchecked<2>();
    auto connectivity = connectivity_arr.unchecked<2>();
    if (node_coords.shape(1) != 2) {
        throw std::runtime_error("node_coords must have shape (n_nodes, 2).");
    }
    if (connectivity.shape(1) != 3) {
        throw std::runtime_error("connectivity must have shape (n_elements, 3).");
    }
    return build_structural_similarity_matrix_from_buffers(
        static_cast<const scalar_type*>(node_coords_arr.request().ptr),
        static_cast<storage_index_type>(node_coords.shape(0)),
        static_cast<const index_type*>(connectivity_arr.request().ptr),
        static_cast<storage_index_type>(connectivity.shape(0)),
        poisson,
        factor,
        xi
    );
}

std::vector<storage_index_type> make_node_ids(
    const py::array_t<index_type, py::array::c_style | py::array::forcecast>& node_ids_arr,
    const storage_index_type nnodes,
    const char* label
) {
    auto node_ids = node_ids_arr.unchecked<1>();
    std::vector<storage_index_type> out;
    out.reserve(static_cast<std::size_t>(node_ids.shape(0)));
    std::unordered_set<storage_index_type> seen;
    for (ssize_t i = 0; i < node_ids.shape(0); ++i) {
        const auto node_id = static_cast<storage_index_type>(node_ids(i));
        if (node_id < 0 || node_id >= nnodes) {
            throw std::runtime_error(std::string(label) + " contains an out-of-range node index.");
        }
        if (seen.insert(node_id).second) {
            out.push_back(node_id);
        }
    }
    return out;
}

vector_type make_state_vector(py::handle obj, const storage_index_type nnodes, const char* label) {
    auto arr = py::array_t<scalar_type, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr) {
        throw std::runtime_error(std::string(label) + " must be convertible to a contiguous float64 array.");
    }
    auto info = arr.request();
    vector_type out(2 * nnodes);
    out.setZero();
    if (info.ndim == 1) {
        if (info.shape[0] != 2 * nnodes) {
            throw std::runtime_error(std::string(label) + " flat size does not match the expected mesh dof count.");
        }
        auto* data = static_cast<const scalar_type*>(info.ptr);
        for (storage_index_type i = 0; i < 2 * nnodes; ++i) {
            out[i] = data[i];
        }
        return out;
    }
    if (info.ndim == 2) {
        if (info.shape[0] != nnodes || info.shape[1] != 2) {
            throw std::runtime_error(std::string(label) + " must have shape (n_nodes, 2).");
        }
        auto* data = static_cast<const scalar_type*>(info.ptr);
        for (storage_index_type node = 0; node < nnodes; ++node) {
            out[2 * node + 0] = data[2 * node + 0];
            out[2 * node + 1] = data[2 * node + 1];
        }
        return out;
    }
    throw std::runtime_error(std::string(label) + " must have rank 1 or 2.");
}

py::array_t<scalar_type> make_nodal_array(const vector_type& state, const storage_index_type nnodes) {
    py::array_t<scalar_type> out({static_cast<ssize_t>(nnodes), static_cast<ssize_t>(2)});
    auto view = out.mutable_unchecked<2>();
    for (storage_index_type node = 0; node < nnodes; ++node) {
        view(node, 0) = state[2 * node + 0];
        view(node, 1) = state[2 * node + 1];
    }
    return out;
}

class StructuralMeshMotionStrategyHandle {
public:
    StructuralMeshMotionStrategyHandle(
        py::array_t<scalar_type, py::array::c_style | py::array::forcecast> node_coords_arr,
        py::array_t<index_type, py::array::c_style | py::array::forcecast> connectivity_arr,
        py::array_t<index_type, py::array::c_style | py::array::forcecast> fixed_node_ids_arr,
        py::array_t<index_type, py::array::c_style | py::array::forcecast> interface_node_ids_arr,
        double poisson = 0.3,
        double factor = 100.0,
        double xi = 1.5
    )
        : m_nnodes(static_cast<storage_index_type>(node_coords_arr.shape(0))),
          m_nelem(static_cast<storage_index_type>(connectivity_arr.shape(0))),
          m_fixed_nodes(make_node_ids(fixed_node_ids_arr, m_nnodes, "fixed_node_ids")),
          m_interface_nodes(make_node_ids(interface_node_ids_arr, m_nnodes, "interface_node_ids")),
          m_state(2 * m_nnodes),
          m_poisson(poisson),
          m_factor(factor),
          m_xi(xi)
    {
        m_state.setZero();
        auto node_info = node_coords_arr.request();
        auto conn_info = connectivity_arr.request();
        const auto* node_ptr = static_cast<const scalar_type*>(node_info.ptr);
        const auto* conn_ptr = static_cast<const index_type*>(conn_info.ptr);
        m_node_coords_ref.assign(node_ptr, node_ptr + static_cast<std::size_t>(2 * m_nnodes));
        m_connectivity.assign(conn_ptr, conn_ptr + static_cast<std::size_t>(3 * m_nelem));
        build_constrained_rows();
    }

    void reset_state() {
        m_state.setZero();
    }

    void set_state(py::handle state_obj) {
        m_state = make_state_vector(state_obj, m_nnodes, "state");
    }

    py::array_t<scalar_type> get_state() const {
        return make_nodal_array(m_state, m_nnodes);
    }

    py::dict solve(
        py::array_t<scalar_type, py::array::c_style | py::array::forcecast> interface_values_arr,
        py::object current_state_obj = py::none(),
        bool preserve_free_state = true
    ) {
        auto interface_values = interface_values_arr.unchecked<2>();
        if (interface_values.shape(0) != static_cast<ssize_t>(m_interface_nodes.size()) || interface_values.shape(1) != 2) {
            throw std::runtime_error("interface_values must have shape (n_interface_nodes, 2).");
        }

        if (!current_state_obj.is_none()) {
            m_state = make_state_vector(current_state_obj, m_nnodes, "current_state");
        } else if (!preserve_free_state) {
            m_state.setZero();
        }

        vector_type x_trial = m_state;
        for (const auto node_id : m_fixed_nodes) {
            x_trial[2 * node_id + 0] = 0.0;
            x_trial[2 * node_id + 1] = 0.0;
        }
        for (storage_index_type i = 0; i < static_cast<storage_index_type>(m_interface_nodes.size()); ++i) {
            const auto node_id = m_interface_nodes[static_cast<std::size_t>(i)];
            x_trial[2 * node_id + 0] = interface_values(i, 0);
            x_trial[2 * node_id + 1] = interface_values(i, 1);
        }

        // Kratos' StructuralMeshMovingStrategy calls
        // UpdateCurrentToInitialConfiguration() before BuildAndSolve().  The
        // carried MESH_DISPLACEMENT enters only through the residual
        // -K(reference) * x_trial, not through the element geometry used for K.
        matrix_type A_raw = build_structural_similarity_matrix_from_buffers(
            m_node_coords_ref.data(),
            m_nnodes,
            m_connectivity.data(),
            m_nelem,
            m_poisson,
            m_factor,
            m_xi
        );
        vector_type rhs = -(A_raw * x_trial);
        for (const auto row : m_constrained_rows) {
            rhs[row] = 0.0;
        }

        matrix_type A_constrained = apply_constrained_rows(A_raw);
        solver_type solver;
        solver.compute(A_constrained);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Structural mesh motion strategy factorization failed.");
        }

        vector_type delta = solver.solve(rhs);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Structural mesh motion strategy solve failed.");
        }

        m_state = x_trial + delta;
        for (const auto node_id : m_fixed_nodes) {
            m_state[2 * node_id + 0] = 0.0;
            m_state[2 * node_id + 1] = 0.0;
        }
        for (storage_index_type i = 0; i < static_cast<storage_index_type>(m_interface_nodes.size()); ++i) {
            const auto node_id = m_interface_nodes[static_cast<std::size_t>(i)];
            m_state[2 * node_id + 0] = interface_values(i, 0);
            m_state[2 * node_id + 1] = interface_values(i, 1);
        }

        py::dict result;
        result["solution"] = make_nodal_array(m_state, m_nnodes);
        result["rank"] = static_cast<int>(solver.rank());
        result["converged"] = static_cast<bool>(solver.info() == Eigen::Success);
        return result;
    }

private:
    void build_constrained_rows() {
        const storage_index_type ndof = 2 * m_nnodes;
        std::vector<char> constrained_mask(static_cast<std::size_t>(ndof), 0);

        auto mark_node = [&](const storage_index_type node_id) {
            const storage_index_type row_x = 2 * node_id + 0;
            const storage_index_type row_y = 2 * node_id + 1;
            if (!constrained_mask[static_cast<std::size_t>(row_x)]) {
                constrained_mask[static_cast<std::size_t>(row_x)] = 1;
                m_constrained_rows.push_back(row_x);
            }
            if (!constrained_mask[static_cast<std::size_t>(row_y)]) {
                constrained_mask[static_cast<std::size_t>(row_y)] = 1;
                m_constrained_rows.push_back(row_y);
            }
        };
        for (const auto node_id : m_fixed_nodes) {
            mark_node(node_id);
        }
        for (const auto node_id : m_interface_nodes) {
            mark_node(node_id);
        }
    }

    matrix_type apply_constrained_rows(matrix_type A_constrained) const {
        const storage_index_type ndof = static_cast<storage_index_type>(A_constrained.rows());
        std::vector<char> constrained_mask(static_cast<std::size_t>(ndof), 0);
        for (const auto row : m_constrained_rows) {
            constrained_mask[static_cast<std::size_t>(row)] = 1;
        }
        scalar_type scale_factor = 1.0;
        const auto diag = A_constrained.diagonal();
        if (diag.size() > 0) {
            scale_factor = std::max<scalar_type>(1.0, diag.cwiseAbs().maxCoeff());
        }

        for (storage_index_type row = 0; row < ndof; ++row) {
            bool has_nonzero = false;
            for (typename matrix_type::InnerIterator it(A_constrained, row); it; ++it) {
                if (std::abs(it.value()) > std::numeric_limits<scalar_type>::epsilon()) {
                    has_nonzero = true;
                    break;
                }
            }
            if (!has_nonzero) {
                A_constrained.coeffRef(row, row) = scale_factor;
            }
        }
        A_constrained.makeCompressed();

        for (storage_index_type row = 0; row < ndof; ++row) {
            if (constrained_mask[static_cast<std::size_t>(row)]) {
                for (typename matrix_type::InnerIterator it(A_constrained, row); it; ++it) {
                    if (it.col() != row) {
                        it.valueRef() = 0.0;
                    }
                }
            } else {
                for (typename matrix_type::InnerIterator it(A_constrained, row); it; ++it) {
                    if (constrained_mask[static_cast<std::size_t>(it.col())]) {
                        it.valueRef() = 0.0;
                    }
                }
            }
        }
        A_constrained.prune(0.0);
        A_constrained.makeCompressed();
        return A_constrained;
    }

    storage_index_type m_nnodes;
    storage_index_type m_nelem;
    std::vector<storage_index_type> m_fixed_nodes;
    std::vector<storage_index_type> m_interface_nodes;
    std::vector<storage_index_type> m_constrained_rows;
    std::vector<scalar_type> m_node_coords_ref;
    std::vector<index_type> m_connectivity;
    vector_type m_state;
    double m_poisson;
    double m_factor;
    double m_xi;
};

}  // namespace

PYBIND11_MODULE(_pycutfem_cpp_structural_mesh_motion_strategy_2026_04_26_v3, m) {
    m.doc() = "pycutfem structural mesh-motion strategy backend";
    py::class_<StructuralMeshMotionStrategyHandle>(m, "StructuralMeshMotionStrategyHandle")
        .def(
            py::init<
                py::array_t<scalar_type, py::array::c_style | py::array::forcecast>,
                py::array_t<index_type, py::array::c_style | py::array::forcecast>,
                py::array_t<index_type, py::array::c_style | py::array::forcecast>,
                py::array_t<index_type, py::array::c_style | py::array::forcecast>,
                double,
                double,
                double
            >(),
            py::arg("node_coords"),
            py::arg("connectivity"),
            py::arg("fixed_node_ids"),
            py::arg("interface_node_ids"),
            py::arg("poisson") = 0.3,
            py::arg("factor") = 100.0,
            py::arg("xi") = 1.5
        )
        .def("reset_state", &StructuralMeshMotionStrategyHandle::reset_state)
        .def("set_state", &StructuralMeshMotionStrategyHandle::set_state, py::arg("state"))
        .def("get_state", &StructuralMeshMotionStrategyHandle::get_state)
        .def(
            "solve",
            &StructuralMeshMotionStrategyHandle::solve,
            py::arg("interface_values"),
            py::arg("current_state") = py::none(),
            py::arg("preserve_free_state") = true
        );
}
