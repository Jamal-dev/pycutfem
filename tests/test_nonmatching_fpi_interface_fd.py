import numpy as np

from examples.FPI.fpi_mms_example41_nonmatching import build_two_mesh_problem
from pycutfem.nonmatching.fpi_cutfem_nitsche import assemble_fpi_interface_nitsche


def test_nonmatching_interface_fd_consistency():
    """
    Finite-difference check for the explicit nonmatching Γ^FP assembler.

    This is intentionally tiny (nx=2) so it runs fast in CI and catches sign / index bugs.
    """
    prob = build_two_mesh_problem(nx_f=2, nx_p=1, poly_order=1, qdeg=4, x0=-0.45)
    n_f = int(prob.dh_f.total_dofs)
    n_p = int(prob.dh_p.total_dofs)

    rng = np.random.default_rng(0)
    Uf = 0.1 * rng.standard_normal(n_f)
    Up = 0.1 * rng.standard_normal(n_p)
    Up_n = Up.copy()

    K, R = assemble_fpi_interface_nitsche(
        interface=prob.iface_fp,
        dh_f=prob.dh_f,
        dh_p=prob.dh_p,
        Uf=Uf,
        Up=Up,
        Up_n=Up_n,
        dt=0.05,
        rho_f=1.0,
        mu_f=1.0,
        porosity=0.5,
        beta_BJ=1.0,
        kappa=0.3,
        gamma_n=1.0 / 45.0,
        gamma_t=1.0 / 45.0,
        zeta=-1.0,
        vF_inf=0.1,
        c_v_gamma=1.0 / 6.0,
        c_t_gamma=1.0 / 12.0,
        quad_order=4,
    )

    # Probe a few DOFs from both meshes
    probes = []
    for fld in ("v_pos_x", "v_pos_y", "p_pos_"):
        sl = prob.dh_f.get_field_slice(fld)
        if sl:
            probes.append(int(sl[len(sl) // 2]))
    for fld in ("v_neg_x", "u_neg_x"):
        sl = prob.dh_p.get_field_slice(fld)
        if sl:
            probes.append(n_f + int(sl[len(sl) // 2]))

    eps = 1.0e-7
    for j in probes:
        Uf_p = Uf.copy()
        Up_p = Up.copy()
        if j < n_f:
            Uf_p[j] += eps
        else:
            Up_p[j - n_f] += eps

        _, R_p = assemble_fpi_interface_nitsche(
            interface=prob.iface_fp,
            dh_f=prob.dh_f,
            dh_p=prob.dh_p,
            Uf=Uf_p,
            Up=Up_p,
            Up_n=Up_n,
            dt=0.05,
            rho_f=1.0,
            mu_f=1.0,
            porosity=0.5,
            beta_BJ=1.0,
            kappa=0.3,
            gamma_n=1.0 / 45.0,
            gamma_t=1.0 / 45.0,
            zeta=-1.0,
            vF_inf=0.1,
            c_v_gamma=1.0 / 6.0,
            c_t_gamma=1.0 / 12.0,
            quad_order=4,
        )

        fd = (R_p - R) / eps
        col = np.asarray(K.getcol(j).toarray()).reshape(-1)
        denom = max(1.0, float(np.linalg.norm(fd, ord=np.inf)), float(np.linalg.norm(col, ord=np.inf)))
        rel = float(np.linalg.norm(fd - col, ord=np.inf)) / denom
        assert rel < 1.0e-6, f"FD mismatch at dof {j}: rel={rel:.2e}"

