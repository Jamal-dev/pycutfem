#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

using scalar_type = double;

static void require_shape(const py::array &arr, const std::vector<py::ssize_t> &shape, const char *name)
{
    if (arr.ndim() != static_cast<py::ssize_t>(shape.size())) {
        throw std::runtime_error(std::string(name) + " has wrong rank.");
    }
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(shape.size()); ++i) {
        if (shape[static_cast<std::size_t>(i)] >= 0 && arr.shape(i) != shape[static_cast<std::size_t>(i)]) {
            throw std::runtime_error(std::string(name) + " has wrong shape.");
        }
    }
}

py::tuple assemble_kratos_system_core(
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> basis_u_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> basis_p_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> detJ_geo_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> ref_weights_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> detF_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> cof_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> grad_phi_u_ref_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> grad_phi_u_cur_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> grad_phi_p_cur_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> grad_u_phys_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> resolved_conv_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> predicted_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> old_subscale_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> momentum_projection_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> mass_projection_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> old_mass_residual_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> h_e_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> x_local_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> a_local_arr,
    py::array_t<scalar_type, py::array::c_style | py::array::forcecast> body_arr,
    scalar_type rho,
    scalar_type mu,
    scalar_type inv_dt,
    scalar_type inc_scale,
    scalar_type mam)
{
    const py::ssize_t n_elem = x_local_arr.shape(0);
    const py::ssize_t n_q = ref_weights_arr.shape(0);
    require_shape(basis_u_arr, {n_q, 3}, "basis_u");
    require_shape(basis_p_arr, {n_q, 3}, "basis_p");
    require_shape(detJ_geo_arr, {n_elem}, "detJ_geo");
    require_shape(detF_arr, {n_elem}, "detF");
    require_shape(cof_arr, {n_elem, 2, 2}, "cof");
    require_shape(grad_phi_u_ref_arr, {n_elem, 3, 2}, "grad_phi_u_ref");
    require_shape(grad_phi_u_cur_arr, {n_elem, 3, 2}, "grad_phi_u_cur");
    require_shape(grad_phi_p_cur_arr, {n_elem, 3, 2}, "grad_phi_p_cur");
    require_shape(grad_u_phys_arr, {n_elem, 2, 2}, "grad_u_phys");
    require_shape(resolved_conv_arr, {n_elem, n_q, 2}, "resolved_conv");
    require_shape(predicted_arr, {n_elem, n_q, 2}, "predicted");
    require_shape(old_subscale_arr, {n_elem, n_q, 2}, "old_subscale");
    require_shape(momentum_projection_arr, {n_elem, n_q, 2}, "momentum_projection");
    require_shape(mass_projection_arr, {n_elem, n_q}, "mass_projection");
    require_shape(old_mass_residual_arr, {n_elem, n_q}, "old_mass_residual");
    require_shape(h_e_arr, {n_elem}, "h_e");
    require_shape(x_local_arr, {n_elem, 9}, "x_local");
    require_shape(a_local_arr, {n_elem, 9}, "a_local");
    require_shape(body_arr, {2}, "body");

    auto basis_u = basis_u_arr.unchecked<2>();
    auto basis_p = basis_p_arr.unchecked<2>();
    auto detJ_geo = detJ_geo_arr.unchecked<1>();
    auto ref_weights = ref_weights_arr.unchecked<1>();
    auto detF = detF_arr.unchecked<1>();
    auto cof = cof_arr.unchecked<3>();
    auto grad_phi_u_ref = grad_phi_u_ref_arr.unchecked<3>();
    auto grad_phi_u_cur = grad_phi_u_cur_arr.unchecked<3>();
    auto grad_phi_p_cur = grad_phi_p_cur_arr.unchecked<3>();
    auto grad_u_phys = grad_u_phys_arr.unchecked<3>();
    auto resolved_conv = resolved_conv_arr.unchecked<3>();
    auto predicted = predicted_arr.unchecked<3>();
    auto old_subscale = old_subscale_arr.unchecked<3>();
    auto momentum_projection = momentum_projection_arr.unchecked<3>();
    auto mass_projection = mass_projection_arr.unchecked<2>();
    auto old_mass_residual = old_mass_residual_arr.unchecked<2>();
    auto h_e = h_e_arr.unchecked<1>();
    auto x_local = x_local_arr.unchecked<2>();
    auto a_local = a_local_arr.unchecked<2>();
    auto body = body_arr.unchecked<1>();

    const std::size_t block_size = static_cast<std::size_t>(n_elem) * 9u * 9u;
    const std::size_t rhs_size = static_cast<std::size_t>(n_elem) * 9u;
    std::vector<scalar_type> K_nonvisc(block_size, 0.0);
    std::vector<scalar_type> K_visc(block_size, 0.0);
    std::vector<scalar_type> M_elem(block_size, 0.0);
    std::vector<scalar_type> raw_rhs(rhs_size, 0.0);

    auto kidx = [](py::ssize_t e, py::ssize_t i, py::ssize_t j) -> std::size_t {
        return static_cast<std::size_t>((e * 9 + i) * 9 + j);
    };
    auto ridx = [](py::ssize_t e, py::ssize_t i) -> std::size_t {
        return static_cast<std::size_t>(e * 9 + i);
    };

    for (py::ssize_t e = 0; e < n_elem; ++e) {
        const scalar_type div_u = grad_u_phys(e, 0, 0) + grad_u_phys(e, 1, 1);
        scalar_type visc_sigma[2][2];
        for (int r = 0; r < 2; ++r) {
            for (int s = 0; s < 2; ++s) {
                const scalar_type delta = (r == s) ? 1.0 : 0.0;
                visc_sigma[r][s] =
                    mu * (grad_u_phys(e, r, s) + grad_u_phys(e, s, r) - (2.0 / 3.0) * div_u * delta);
            }
        }
        scalar_type visc_action[2][2];
        for (int r = 0; r < 2; ++r) {
            for (int t = 0; t < 2; ++t) {
                visc_action[r][t] = visc_sigma[r][0] * cof(e, 0, t) + visc_sigma[r][1] * cof(e, 1, t);
            }
        }

        for (py::ssize_t q = 0; q < n_q; ++q) {
            const scalar_type wgeo = detJ_geo(e) * ref_weights(q);
            const scalar_type weight = wgeo * detF(e);
            const scalar_type conv0 = resolved_conv(e, q, 0) + predicted(e, q, 0);
            const scalar_type conv1 = resolved_conv(e, q, 1) + predicted(e, q, 1);
            const scalar_type conv_speed = std::sqrt(conv0 * conv0 + conv1 * conv1);
            const scalar_type h_value = h_e(e);
            const scalar_type h_safe = h_value > 1.0e-30 ? h_value : 1.0e-30;
            const scalar_type tau_one =
                1.0 / (8.0 * mu / (h_safe * h_safe) + rho * (inv_dt + 2.0 * conv_speed / h_safe));
            const scalar_type tau_two = mu + rho * conv_speed * h_value / 4.0;
            const scalar_type tau_p = rho * h_value * h_value * inv_dt / 8.0;
            const scalar_type old_uss0 = rho * inv_dt * old_subscale(e, q, 0);
            const scalar_type old_uss1 = rho * inv_dt * old_subscale(e, q, 1);
            const scalar_type source0 = body(0) - momentum_projection(e, q, 0) + old_uss0;
            const scalar_type source1 = body(1) - momentum_projection(e, q, 1) + old_uss1;
            const scalar_type body_old0 = body(0) + old_uss0;
            const scalar_type body_old1 = body(1) + old_uss1;

            scalar_type gdot[3];
            for (int i = 0; i < 3; ++i) {
                gdot[i] = grad_phi_u_cur(e, i, 0) * conv0 + grad_phi_u_cur(e, i, 1) * conv1;
            }

            for (int c = 0; c < 2; ++c) {
                const scalar_type body_old_c = c == 0 ? body_old0 : body_old1;
                const scalar_type source_c = c == 0 ? source0 : source1;
                for (int i = 0; i < 3; ++i) {
                    const int row = c * 3 + i;
                    const scalar_type phi_i = basis_u(q, i);
                    const scalar_type div_v_cur = grad_phi_u_cur(e, i, c);
                    const scalar_type div_v_cof =
                        cof(e, c, 0) * grad_phi_u_ref(e, i, 0) + cof(e, c, 1) * grad_phi_u_ref(e, i, 1);
                    const scalar_type tau_source_i = rho * (gdot[i] - inv_dt * phi_i);
                    const scalar_type tau_mass_i = rho * gdot[i] - inv_dt * phi_i;

                    raw_rhs[ridx(e, row)] += weight * phi_i * body_old_c;
                    raw_rhs[ridx(e, row)] += weight * tau_one * tau_source_i * source_c;
                    raw_rhs[ridx(e, row)] -=
                        weight * inc_scale * (tau_two + tau_p) * mass_projection(e, q) * div_v_cur;
                    raw_rhs[ridx(e, row)] -= weight * inc_scale * tau_p * old_mass_residual(e, q) * div_v_cur;
                    raw_rhs[ridx(e, row)] -=
                        wgeo * (visc_action[c][0] * grad_phi_u_ref(e, i, 0) +
                                visc_action[c][1] * grad_phi_u_ref(e, i, 1));

                    for (int a = 0; a < 2; ++a) {
                        for (int j = 0; j < 3; ++j) {
                            const int col = a * 3 + j;
                            if (a == c) {
                                K_nonvisc[kidx(e, row, col)] += weight * rho * gdot[j] * phi_i;
                                K_nonvisc[kidx(e, row, col)] += weight * tau_one * tau_source_i * rho * gdot[j];
                                M_elem[kidx(e, row, col)] += weight * rho * phi_i * basis_u(q, j);
                                M_elem[kidx(e, row, col)] +=
                                    weight * tau_one * tau_mass_i * rho * basis_u(q, j);
                            }
                            K_nonvisc[kidx(e, row, col)] +=
                                weight * inc_scale * (tau_two + tau_p) * div_v_cur * grad_phi_u_cur(e, j, a);

                            scalar_type grad_du[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
                            grad_du[a][0] = grad_phi_u_cur(e, j, 0);
                            grad_du[a][1] = grad_phi_u_cur(e, j, 1);
                            const scalar_type div_du = grad_phi_u_cur(e, j, a);
                            scalar_type sigma_du[2][2];
                            for (int r = 0; r < 2; ++r) {
                                for (int s = 0; s < 2; ++s) {
                                    const scalar_type delta = (r == s) ? 1.0 : 0.0;
                                    sigma_du[r][s] =
                                        mu * (grad_du[r][s] + grad_du[s][r] - (2.0 / 3.0) * div_du * delta);
                                }
                            }
                            const scalar_type action0 = sigma_du[c][0] * cof(e, 0, 0) + sigma_du[c][1] * cof(e, 1, 0);
                            const scalar_type action1 = sigma_du[c][0] * cof(e, 0, 1) + sigma_du[c][1] * cof(e, 1, 1);
                            K_visc[kidx(e, row, col)] +=
                                wgeo * (action0 * grad_phi_u_ref(e, i, 0) + action1 * grad_phi_u_ref(e, i, 1));
                        }
                    }

                    for (int j = 0; j < 3; ++j) {
                        const int colp = 6 + j;
                        K_nonvisc[kidx(e, row, colp)] +=
                            weight * tau_one * tau_source_i * grad_phi_p_cur(e, j, c);
                        K_nonvisc[kidx(e, row, colp)] -= wgeo * div_v_cof * basis_p(q, j);
                    }
                }
            }

            for (int i = 0; i < 3; ++i) {
                const int row = 6 + i;
                const scalar_type grad_q0 = grad_phi_p_cur(e, i, 0);
                const scalar_type grad_q1 = grad_phi_p_cur(e, i, 1);
                raw_rhs[ridx(e, row)] += weight * inc_scale * tau_one * (grad_q0 * source0 + grad_q1 * source1);
                for (int a = 0; a < 2; ++a) {
                    const scalar_type grad_qa = a == 0 ? grad_q0 : grad_q1;
                    for (int j = 0; j < 3; ++j) {
                        const int col = a * 3 + j;
                        K_nonvisc[kidx(e, row, col)] += weight * inc_scale * tau_one * grad_qa * rho * gdot[j];
                        K_nonvisc[kidx(e, row, col)] +=
                            wgeo * basis_p(q, i) *
                            (cof(e, a, 0) * grad_phi_u_ref(e, j, 0) +
                             cof(e, a, 1) * grad_phi_u_ref(e, j, 1));
                        M_elem[kidx(e, row, col)] += weight * tau_one * grad_qa * rho * basis_u(q, j);
                    }
                }
                for (int j = 0; j < 3; ++j) {
                    const int colp = 6 + j;
                    K_nonvisc[kidx(e, row, colp)] +=
                        weight * inc_scale * tau_one *
                        (grad_q0 * grad_phi_p_cur(e, j, 0) + grad_q1 * grad_phi_p_cur(e, j, 1));
                }
            }
        }
    }

    py::array_t<scalar_type> K_elem_arr({n_elem, static_cast<py::ssize_t>(9), static_cast<py::ssize_t>(9)});
    py::array_t<scalar_type> raw_rhs_arr({n_elem, static_cast<py::ssize_t>(9)});
    auto K_elem = K_elem_arr.mutable_unchecked<3>();
    auto out_rhs = raw_rhs_arr.mutable_unchecked<2>();
    for (py::ssize_t e = 0; e < n_elem; ++e) {
        for (int i = 0; i < 9; ++i) {
            scalar_type kx = 0.0;
            scalar_type ma = 0.0;
            for (int j = 0; j < 9; ++j) {
                kx += K_nonvisc[kidx(e, i, j)] * x_local(e, j);
                ma += M_elem[kidx(e, i, j)] * a_local(e, j);
                K_elem(e, i, j) = K_nonvisc[kidx(e, i, j)] + K_visc[kidx(e, i, j)] + mam * M_elem[kidx(e, i, j)];
            }
            out_rhs(e, i) = -(raw_rhs[ridx(e, i)] - kx - ma);
        }
    }

    return py::make_tuple(K_elem_arr, raw_rhs_arr);
}

PYBIND11_MODULE(_pycutfem_nirb_reduced_dvms_2026_05_15_nirb_reduced_dvms_v1, m)
{
    m.def("assemble_kratos_system_core", &assemble_kratos_system_core, "Assemble NIRB ALE-DVMS local core in C++");
}

