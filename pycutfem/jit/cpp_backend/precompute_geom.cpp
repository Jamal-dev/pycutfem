#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <vector>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

inline void jacobian_q1(
    double s,
    double t,
    const double* x,
    const double* y,
    double& j00,
    double& j01,
    double& j10,
    double& j11)
{
    // Node ordering matches pycutfem.integration.pre_tabulates._tabulate_q1:
    // 0: (-1,-1), 1: ( 1,-1), 2: (-1, 1), 3: ( 1, 1)
    const double dNds[4] = {
        -0.25 * (1.0 - t),
         0.25 * (1.0 - t),
        -0.25 * (1.0 + t),
         0.25 * (1.0 + t),
    };
    const double dNdt[4] = {
        -0.25 * (1.0 - s),
        -0.25 * (1.0 + s),
         0.25 * (1.0 - s),
         0.25 * (1.0 + s),
    };

    j00 = 0.0;
    j01 = 0.0;
    j10 = 0.0;
    j11 = 0.0;
    for (int i = 0; i < 4; ++i) {
        j00 += dNds[i] * x[i];
        j01 += dNdt[i] * x[i];
        j10 += dNds[i] * y[i];
        j11 += dNdt[i] * y[i];
    }
}

inline void jacobian_q2(
    double s,
    double t,
    const double* x,
    const double* y,
    double& j00,
    double& j01,
    double& j10,
    double& j11)
{
    // Node ordering matches pycutfem.integration.pre_tabulates._tabulate_q2:
    // Row-major tensor product on (xi,eta) nodes {-1,0,1}×{-1,0,1}.
    const double L[3] = {
        0.5 * s * (s - 1.0),
        1.0 - s * s,
        0.5 * s * (s + 1.0),
    };
    const double dL[3] = {
        s - 0.5,
        -2.0 * s,
        s + 0.5,
    };
    const double M[3] = {
        0.5 * t * (t - 1.0),
        1.0 - t * t,
        0.5 * t * (t + 1.0),
    };
    const double dM[3] = {
        t - 0.5,
        -2.0 * t,
        t + 0.5,
    };

    j00 = 0.0;
    j01 = 0.0;
    j10 = 0.0;
    j11 = 0.0;

    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            const int k = j * 3 + i;
            const double dNds = dL[i] * M[j];
            const double dNdt = L[i] * dM[j];
            j00 += dNds * x[k];
            j01 += dNdt * x[k];
            j10 += dNds * y[k];
            j11 += dNdt * y[k];
        }
    }
}

}  // namespace

py::tuple quad_jacobian_det_inv(
    py::array_t<double, py::array::c_style | py::array::forcecast> coords,
    py::array_t<double, py::array::c_style | py::array::forcecast> xi,
    py::array_t<double, py::array::c_style | py::array::forcecast> eta,
    int poly_order)
{
    if (coords.ndim() != 3 || coords.shape(2) != 2) {
        throw std::runtime_error("coords must have shape (nE, nLoc, 2)");
    }
    if (xi.ndim() != 2 || eta.ndim() != 2) {
        throw std::runtime_error("xi/eta must have shape (nE, nQ)");
    }
    const ssize_t nE = coords.shape(0);
    const ssize_t nLoc = coords.shape(1);
    const ssize_t nQ = xi.shape(1);
    if (xi.shape(0) != nE || eta.shape(0) != nE || eta.shape(1) != nQ) {
        throw std::runtime_error("xi/eta shapes must match coords.shape(0)");
    }
    if (poly_order == 1 && nLoc != 4) {
        throw std::runtime_error("poly_order=1 expects coords.shape(1)==4 (Q1)");
    }
    if (poly_order == 2 && nLoc != 9) {
        throw std::runtime_error("poly_order=2 expects coords.shape(1)==9 (Q2)");
    }
    if (poly_order != 1 && poly_order != 2) {
        throw std::runtime_error("poly_order must be 1 or 2");
    }

    py::array_t<double> J(std::vector<py::ssize_t>{nE, nQ, 2, 2});
    py::array_t<double> det(std::vector<py::ssize_t>{nE, nQ});
    py::array_t<double> inv(std::vector<py::ssize_t>{nE, nQ, 2, 2});

    const double* cptr = static_cast<const double*>(coords.request().ptr);
    const double* xiptr = static_cast<const double*>(xi.request().ptr);
    const double* etaptr = static_cast<const double*>(eta.request().ptr);
    double* Jptr = static_cast<double*>(J.request().ptr);
    double* detptr = static_cast<double*>(det.request().ptr);
    double* invptr = static_cast<double*>(inv.request().ptr);

    const ssize_t coords_stride_elem = nLoc * 2;
    const ssize_t xi_stride_elem = nQ;

#pragma omp parallel for schedule(static)
    for (ssize_t e = 0; e < nE; ++e) {
        const double* ce = cptr + e * coords_stride_elem;

        // Pack x/y for this element into small stack arrays for cache friendliness.
        double xbuf[9];
        double ybuf[9];
        for (ssize_t i = 0; i < nLoc; ++i) {
            xbuf[i] = ce[2 * i + 0];
            ybuf[i] = ce[2 * i + 1];
        }

        const double* xi_e = xiptr + e * xi_stride_elem;
        const double* eta_e = etaptr + e * xi_stride_elem;

        for (ssize_t q = 0; q < nQ; ++q) {
            const double s = xi_e[q];
            const double t = eta_e[q];

            double j00, j01, j10, j11;
            if (poly_order == 1) {
                jacobian_q1(s, t, xbuf, ybuf, j00, j01, j10, j11);
            } else {
                jacobian_q2(s, t, xbuf, ybuf, j00, j01, j10, j11);
            }

            const double d = j00 * j11 - j01 * j10;
            const ssize_t idxJ = ((e * nQ + q) * 4);
            Jptr[idxJ + 0] = j00;
            Jptr[idxJ + 1] = j01;
            Jptr[idxJ + 2] = j10;
            Jptr[idxJ + 3] = j11;

            detptr[e * nQ + q] = d;

            const double invd = 1.0 / d;
            invptr[idxJ + 0] =  j11 * invd;
            invptr[idxJ + 1] = -j01 * invd;
            invptr[idxJ + 2] = -j10 * invd;
            invptr[idxJ + 3] =  j00 * invd;
        }
    }

    return py::make_tuple(J, det, inv);
}


PYBIND11_MODULE(_pycutfem_cpp_precompute_geom, m)
{
    m.doc() = "C++ geometry helpers for pycutfem precompute (MPI-safe, OpenMP-enabled).";
    m.def(
        "quad_jacobian_det_inv",
        &quad_jacobian_det_inv,
        py::arg("coords"),
        py::arg("xi"),
        py::arg("eta"),
        py::arg("poly_order"),
        "Compute (J, detJ, J_inv) for batched Q1/Q2 quads at given reference points."
    );
}
