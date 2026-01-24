import numpy as np

from pycutfem.core.geometry import hansbo_cut_ratio
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, ElementWiseConstant, TestFunction as UFLTestFunction, TrialFunction
from pycutfem.ufl.forms import assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.utils.fsi_fully_eulerian import make_domain_sets
from pycutfem.utils.meshgen import structured_quad


def _assemble_sliver_mass_diag(*, theta_target: float, theta_floor: float) -> tuple[float, float, float]:
    """
    Assemble the DG0 sliver-mass diagonal on a single cut cell:

      a(u,v) = (1/θ_used) ∫_{Ω_cut,+} u v dx

    Returns (theta_raw, diag_raw, diag_clipped).
    """
    # Single element [0,1]×[0,1].
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1, offset=(0.0, 0.0))
    mesh = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=1)

    # DG0 scalar unknown (one DOF per element).
    me = MixedElement(mesh, field_specs={"u_pos_": 0})
    dh = DofHandler(me, method="dg")

    # Level set: φ = δ - x - y -> "+" is a small triangle near (0,0) for small δ.
    delta = float(np.sqrt(max(2.0 * float(theta_target), 0.0)))
    level_set = AffineLevelSet(a=-1.0, b=-1.0, c=delta)

    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set, tol=1.0e-14)
    mesh.build_interface_segments(level_set)
    domains = make_domain_sets(mesh, use_aligned_interface=False)

    theta_raw = hansbo_cut_ratio(mesh, level_set, side="+")
    assert theta_raw.size == 1
    theta_raw_val = float(theta_raw[0])

    dx_cut = dx(
        defined_on=domains["cut_domain"],
        level_set=level_set,
        metadata={"q": 6, "side": "+"},
    )

    u = TrialFunction(name="u_trial", field_name="u_pos_", dof_handler=dh, side="+")
    v = UFLTestFunction(name="u_test", field_name="u_pos_", dof_handler=dh, side="+")

    eps = Constant(1.0e-30)
    inv_theta_raw = Constant(1.0) / (ElementWiseConstant(theta_raw) + eps)
    theta_clip = np.clip(theta_raw, float(theta_floor), 1.0)
    inv_theta_clip = Constant(1.0) / (ElementWiseConstant(theta_clip) + eps)

    bcs: list = []

    zero = Constant(0.0)
    K_raw, _ = assemble_form((inv_theta_raw * u * v) * dx_cut == zero * v * dx_cut, dh, bcs=bcs, quad_order=6, backend="python")
    K_clip, _ = assemble_form((inv_theta_clip * u * v) * dx_cut == zero * v * dx_cut, dh, bcs=bcs, quad_order=6, backend="python")

    diag_raw = float(K_raw.diagonal()[0])
    diag_clip = float(K_clip.diagonal()[0])
    return theta_raw_val, diag_raw, diag_clip


def test_sliver_mass_auto_scaling_removes_theta_floor_capping() -> None:
    """
    On a DG0 cut cell, ∫ (1/θ) u v dx stays O(1) as θ→0, while using a floored θ
    makes the contribution scale like θ/theta_floor and effectively vanish.
    """
    theta_floor = 1.0e-6
    theta_hi = 1.0e-4   # above floor -> clipped == raw
    theta_lo = 1.0e-10  # below floor -> clipped differs

    th_hi, a_hi_raw, a_hi_clip = _assemble_sliver_mass_diag(theta_target=theta_hi, theta_floor=theta_floor)
    th_lo, a_lo_raw, a_lo_clip = _assemble_sliver_mass_diag(theta_target=theta_lo, theta_floor=theta_floor)

    assert th_hi > theta_floor
    assert th_lo < theta_floor

    # Above the floor, clipping should not change the term.
    assert abs(a_hi_raw - a_hi_clip) <= 1.0e-10

    # Below the floor, clipped θ makes the term much smaller than the raw-θ version.
    # Because raw: (1/th)*th ~ 1. Clipped: (1/th_floor)*th ~ 0.
    assert a_lo_raw > 1.0e3 * a_lo_clip