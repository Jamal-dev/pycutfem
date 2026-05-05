from __future__ import annotations

import json
import math

import numpy as np

from examples.biofilms.benchmarks.three_constituent.seboldt_physical import (
    _pore_momentum_outflow_key,
    _seed_reactivated_inactive_dofs,
)
from examples.biofilms.benchmarks.three_constituent.paper1_three_constituent_benchmark_suite import (
    contents,
    darcy_column_velocity,
    drag_relaxation_exact,
    finite_insert_alpha,
    moving_tanh_derivatives,
    run_all_benchmarks,
    stokes_darcy_bed_reference,
    write_benchmark_outputs,
)
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Function
from pycutfem.utils.meshgen import structured_quad


def test_three_constituent_analytic_benchmark_gates_pass_and_write_outputs(tmp_path):
    results = run_all_benchmarks()

    assert [result.case_id for result in results] == [
        "pure_free_fluid_poiseuille",
        "fixed_porous_darcy_column",
        "pore_solid_drag_relaxation",
        "moving_tanh_porous_body",
        "free_flow_over_fixed_porous_bed",
        "stoter_fixed_bed_canonical",
    ]
    assert all(result.passed for result in results)

    summary_path = write_benchmark_outputs(results, tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(summary) == 6
    assert all(item["passed"] for item in summary)
    assert (tmp_path / "pure_free_fluid_poiseuille" / "poiseuille_profile.csv").exists()
    assert (tmp_path / "stoter_fixed_bed_canonical" / "stoter_fixed_bed_centerline.csv").exists()


def test_three_constituent_physical_benchmark_2_to_5_entrypoints_are_exported(tmp_path):
    from examples.biofilms.benchmarks import three_constituent as tc
    from examples.biofilms.benchmarks.three_constituent.paper1_physical_benchmarks_2_to_5 import (
        PhysicalBenchmarkResult,
    )

    assert callable(tc.run_physical_benchmark2_darcy_column)
    assert callable(tc.run_physical_benchmark3_drag_relaxation)
    assert callable(tc.run_physical_benchmark4_moving_tanh_body)
    assert callable(tc.run_physical_benchmark5_fixed_porous_bed)
    assert callable(tc.run_physical_benchmarks_2_to_5)

    result = PhysicalBenchmarkResult(
        case_id="contract_check",
        passed=True,
        outdir=tmp_path,
        summary={"passed": True},
    )
    assert result.case_id == "contract_check"
    assert result.passed is True
    assert result.outdir == tmp_path
    assert result.summary["passed"] is True


def test_three_constituent_seboldt_pore_outflow_key_contract():
    assert _pore_momentum_outflow_key("conservative") == "conservative"
    assert _pore_momentum_outflow_key("weak-conservative") == "conservative"
    assert _pore_momentum_outflow_key("upwind") == "outflow_only"
    assert _pore_momentum_outflow_key("off") == "none"


def test_three_constituent_benchmark_core_identities():
    F, P, B = contents(np.asarray([0.0, 0.4, 1.0]), np.asarray([0.2, 0.5, 0.7]))
    np.testing.assert_allclose(F + P + B, np.ones(3), rtol=0.0, atol=1.0e-15)
    assert P[0] == 0.0
    assert B[0] == 0.0

    vp = darcy_column_velocity(pressure_drop=2.0, length=4.0, phi=0.5, R_ps=8.0)
    assert math.isclose(0.5 * vp, 2.0 / (8.0 * 4.0), rel_tol=0.0, abs_tol=1.0e-15)

    t = np.linspace(0.0, 0.7, 20)
    v_p, v_s, rate = drag_relaxation_exact(t, phi=0.4, rho_p=1.0, rho_s=1.3, R_ps=2.0, v_p0=1.2, v_s0=-0.2)
    relative = v_p - v_s
    np.testing.assert_allclose(relative, relative[0] * np.exp(-rate * t), rtol=1.0e-14, atol=1.0e-14)

    alpha_t, alpha_x = moving_tanh_derivatives(np.linspace(0.0, 1.0, 50), 0.2, x0=0.4, speed=0.3, eps=0.06)
    np.testing.assert_allclose(alpha_t + 0.3 * alpha_x, np.zeros_like(alpha_t), rtol=0.0, atol=1.0e-14)

    y = np.asarray([0.35, 1.0])
    u, u_darcy, _ = stokes_darcy_bed_reference(y, height=1.0, bed_height=0.35, pressure_gradient=1.0, mu=1.0, permeability=0.02)
    assert math.isclose(float(u[0]), float(u_darcy), rel_tol=0.0, abs_tol=1.0e-14)
    assert math.isclose(float(u[1]), 0.0, rel_tol=0.0, abs_tol=1.0e-14)

    xg, yg = np.meshgrid(np.linspace(0.0, 1.0, 11), np.linspace(0.0, 1.0, 11), indexing="xy")
    alpha = finite_insert_alpha(xg, yg, eps=0.02)
    assert float(alpha[0, 0]) == 0.0
    assert float(np.max(alpha)) <= 1.0


def test_three_constituent_reactivated_inactive_dofs_are_seeded_by_l2_projection():
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"phi": 1})
    dh = DofHandler(me, method="cg")
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)

    gds = np.asarray(dh.get_field_slice("phi"), dtype=int)
    coords = np.asarray(dh.get_dof_coords("phi"), dtype=float)
    accepted_values = 0.2 + 0.2 * coords[:, 0] + 0.1 * coords[:, 1]
    phi_k.set_nodal_values(gds, accepted_values)
    phi_n.set_nodal_values(gds, accepted_values)

    target = int(gds[np.argmin(np.sum((coords - np.asarray([0.5, 0.0])) ** 2, axis=1))])
    phi_k.set_nodal_values(np.asarray([target], dtype=int), np.asarray([-99.0]))
    phi_n.set_nodal_values(np.asarray([target], dtype=int), np.asarray([-88.0]))

    expected = float(accepted_values[np.where(gds == target)[0][0]])

    stats = _seed_reactivated_inactive_dofs(
        dh,
        {target},
        functions=[phi_k],
        prev_functions=[phi_n],
        inactive_dofs=set(),
        fallback_by_field={"phi": 0.18},
        projection="l2_patch",
        quad_order=4,
    )

    np.testing.assert_allclose(phi_k.get_nodal_values(np.asarray([target], dtype=int)), [expected])
    np.testing.assert_allclose(phi_n.get_nodal_values(np.asarray([target], dtype=int)), [expected])
    assert stats["phi"]["reactivated"] == 1
    assert stats["phi"]["l2"] == 1
