import numpy as np
import pytest

from examples.variational_inequalities.collision_elastic_disk_rigid_wall import (
    run_elastic_disk_collision_with_wall,
)

from pycutfem.solvers.nonlinear_solver import HAS_PETSC


def test_elastic_disk_collision_wall_contact_contour_and_release_internal_pdas():
    res = run_elastic_disk_collision_with_wall(
        radius=0.5,
        gap=0.05,
        h=0.15,
        n_boundary=48,
        E=1.0e4,
        nu=0.3,
        rho=1.0,
        v0=-5.0,
        dt=0.01,
        n_steps=6,
        wall_y=0.0,
        method="internal-pdas",
        backend="jit",
        output_dir=None,
    )

    assert int(np.max(res.n_active)) > 0

    # Identify the step with the largest contact patch.
    k = int(np.argmax(res.n_active))
    u_k = res.u_hist[k + 1]
    lam_k = res.lam_hist[k]
    active_k = res.active_hist[k]

    dh = res.dof_handler
    mesh = res.mesh
    boundary_uy = res.boundary_uy
    wall_y = float(res.wall_y)

    # Marker 1: zero penetration on constrained boundary nodes.
    y_ref = dh._dof_coords[boundary_uy, 1]
    y_def = y_ref + u_k[boundary_uy]
    assert float(np.min(y_def)) >= wall_y - 1.0e-10

    # Active nodes sit on the wall.
    act_b = active_k[boundary_uy]
    assert int(np.count_nonzero(act_b)) == int(res.n_active[k])
    if np.any(act_b):
        assert float(np.max(np.abs(y_def[act_b] - wall_y))) <= 1.0e-10

    # Dual feasibility + complementarity structure from the restricted solve.
    assert float(np.min(lam_k[boundary_uy])) >= -1.0e-10
    assert float(np.max(np.abs(lam_k[boundary_uy[~act_b]]))) <= 1.0e-14

    # Contact contour: active boundary nodes form one contiguous arc.
    if int(np.count_nonzero(act_b)) >= 4:
        node_xy = np.asarray(mesh.nodes_x_y_pos, dtype=float)
        cx, cy = res.center

        nids = np.array([int(dh._dof_to_node_map[int(gd)][1]) for gd in boundary_uy], dtype=int)
        ang = np.arctan2(node_xy[nids, 1] - float(cy), node_xy[nids, 0] - float(cx))

        # Shift angles so the active cluster is centered (avoid wrap-around artifacts).
        phi0 = np.angle(np.mean(np.exp(1j * ang[act_b])))
        ang_s = (ang - phi0 + np.pi) % (2.0 * np.pi) - np.pi
        order = np.argsort(ang_s)
        m = act_b[order]
        transitions = int(np.count_nonzero(m[1:] != m[:-1]))
        assert transitions <= 2

        # Symmetry: contact arc should be roughly centered around x=cx.
        x_act = node_xy[nids[act_b], 0] + u_k[[int(dh.dof_map["ux"][int(nid)]) for nid in nids[act_b]]]
        if x_act.size >= 4:
            xmin = float(np.min(x_act - float(cx)))
            xmax = float(np.max(x_act - float(cx)))
            width = max(1.0e-14, xmax - xmin)
            assert abs(xmax + xmin) <= 0.25 * width

    # Release: after impact, the active set should clear (no sticky contact).
    assert int(np.max(res.n_active[-2:])) == 0


@pytest.mark.skipif(not HAS_PETSC, reason="petsc4py not available")
def test_internal_pdas_matches_petsc_snesvi_on_disk_collision():
    common = dict(
        radius=0.5,
        gap=0.05,
        h=0.15,
        n_boundary=48,
        E=1.0e4,
        nu=0.3,
        rho=1.0,
        v0=-5.0,
        dt=0.01,
        n_steps=2,
        wall_y=0.0,
        backend="jit",
        output_dir=None,
    )
    res_p = run_elastic_disk_collision_with_wall(method="internal-pdas", **common)
    res_s = run_elastic_disk_collision_with_wall(method="snesvi", **common)

    assert np.array_equal(res_p.n_active, res_s.n_active)
    assert np.allclose(res_p.contact_half_width, res_s.contact_half_width, rtol=0.0, atol=1.0e-10)

    k = int(np.argmax(res_p.n_active))
    du = res_p.u_hist[k + 1] - res_s.u_hist[k + 1]
    assert float(np.linalg.norm(du, ord=np.inf)) <= 1.0e-10
