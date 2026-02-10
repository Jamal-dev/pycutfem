import math
import os
import numpy as np
import scipy.sparse.linalg as spla

from pycutfem.core.geometry import hansbo_cut_ratio
from pycutfem.core.levelset import BeamLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from examples.utils.fsi.fully_eulerian import make_domain_sets, refresh_sliver_weights
from pycutfem.utils.meshgen import structured_quad
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    ElementWiseConstant,
    FacetNormal,
    Neg,
    Pos,
    TestFunction,
    TrialFunction,
    VectorTestFunction,
    VectorTrialFunction,
    div,
    dot,
    grad,
    inner,
    jump,
)
from pycutfem.ufl.measures import dx, dGhost
from pycutfem.ufl.forms import BoundaryCondition, assemble_form


def _refresh_geometry(mesh, level_set):
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set, tol=1.0e-8)
    mesh.build_interface_segments(level_set)


def _tag_inactive(dh, mesh):
    dh.dof_tags["inactive"] = set()
    inactive = mesh.element_bitset("inside")
    for field in ("u_pos_x", "u_pos_y", "p_pos_"):
        dh.tag_dofs_from_element_bitset("inactive", field, inactive, strict=True)


def _kappa_sum_deviation(mesh, theta_pos_vals, theta_neg_vals):
    interface_edges = mesh.edge_bitset("interface").to_indices()
    max_dev = 0.0
    checked = 0
    for eid in interface_edges:
        edge = mesh.edges_list[int(eid)]
        if edge.left is None or edge.right is None:
            continue
        left_id = int(edge.left)
        right_id = int(edge.right)
        left_tag = getattr(mesh.elements_list[left_id], "tag", "")
        right_tag = getattr(mesh.elements_list[right_id], "tag", "")
        if left_tag == "outside" and right_tag == "inside":
            pos_id, neg_id = left_id, right_id
        elif right_tag == "outside" and left_tag == "inside":
            pos_id, neg_id = right_id, left_id
        else:
            continue
        thp = float(theta_pos_vals[pos_id])
        thn = float(theta_neg_vals[neg_id])
        denom = thp + thn
        if denom <= 0.0:
            continue
        kappa_sum = thp / denom + thn / denom
        max_dev = max(max_dev, abs(kappa_sum - 1.0))
        checked += 1
    return max_dev, checked, int(interface_edges.size)


def _build_stokes_system(mesh, dh, u, v, p, q, bcs, *, level_set, domains, w_pos_cell, backend):
    dx_fluid = dx(
        defined_on=domains["fluid_interface"],
        level_set=level_set,
        metadata={"q": 4, "side": "+"},
    )
    dG = dGhost(
        defined_on=domains["fluid_ghost"],
        level_set=level_set,
        metadata={"q": 4, "derivs": {(0, 1), (1, 0)}},
    )

    cell_h = CellDiameter()
    n = FacetNormal()

    def grad_inner_jump(phi_1, phi_2):
        a = dot(jump(grad(phi_1)), n)
        b = dot(jump(grad(phi_2)), n)
        return inner(a, b)

    w_gp = Constant(0.5) * (Pos(w_pos_cell) + Neg(w_pos_cell))
    gamma_v = Constant(0.5) * w_gp
    gamma_p = Constant(0.5) * w_gp

    a = (inner(grad(u), grad(v)) - p * div(v) + q * div(u)) * dx_fluid
    a_stab = (gamma_v * cell_h * grad_inner_jump(u, v) + gamma_p * (cell_h**3.0) * grad_inner_jump(p, q)) * dG
    f = Constant([0.0, 0.0], dim=1)
    L = dot(f, v) * dx_fluid

    equation = a + a_stab == L
    K, F = assemble_form(equation, dh, bcs=bcs, quad_order=4, backend=backend)
    return K, F


def test_stokes_levelset_refresh_solve():
    backend = os.getenv("STOKES_REFRESH_BACKEND", "python").strip().lower()
    nodes, elems, _, corners = structured_quad(
        2.0,
        1.0,
        nx=8,
        ny=4,
        poly_order=2,
        offset=(-1.0, -0.5),
    )
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=2)
    me = MixedElement(mesh, field_specs={"u_pos_x": 2, "u_pos_y": 2, "p_pos_": 1})
    dh = DofHandler(me, method="cg")

    vel_space = FunctionSpace("velocity", ["u_pos_x", "u_pos_y"], dim=1, side="+")
    u = VectorTrialFunction(space=vel_space, dof_handler=dh, side="+")
    v = VectorTestFunction(space=vel_space, dof_handler=dh, side="+")
    p = TrialFunction(name="p_trial", field_name="p_pos_", dof_handler=dh, side="+")
    q = TestFunction(name="p_test", field_name="p_pos_", dof_handler=dh, side="+")

    walls = {
        "left": lambda x, y: np.isclose(x, -1.0),
        "right": lambda x, y: np.isclose(x, 1.0),
        "bottom": lambda x, y: np.isclose(y, -0.5),
        "top": lambda x, y: np.isclose(y, 0.5),
    }
    mesh.tag_boundary_edges(walls)
    dh.tag_dof_by_locator("pressure_pin", "p_pos_", lambda x, y: np.isclose(x, 0.5) and np.isclose(y, 0.0))

    bcs = [
        *[
            BoundaryCondition(comp, "dirichlet", side, lambda x, y: 0.0)
            for comp in ("u_pos_x", "u_pos_y")
            for side in ("left", "right", "bottom", "top")
        ],
        *[
            BoundaryCondition(comp, "dirichlet", "inactive", lambda x, y: 0.0)
            for comp in ("u_pos_x", "u_pos_y", "p_pos_")
        ],
        BoundaryCondition("p_pos_", "dirichlet", "pressure_pin", lambda x, y: 0.0),
    ]

    rect_length = 1.1
    rect_height = 0.5
    centers_y = (0.0, 1.0e-2, 0.2, -0.2)

    theta_min = 1.0e-8
    theta_pos_vals = np.ones(len(mesh.elements_list), dtype=float)
    theta_neg_vals = np.ones(len(mesh.elements_list), dtype=float)
    w_pos_vals = np.ones_like(theta_pos_vals)
    w_neg_vals = np.ones_like(theta_neg_vals)

    cut_counts = []
    interface_counts = []
    saw_both = False

    for step, center_y in enumerate(centers_y):
        level_set = BeamLevelSet(center=(0.0, center_y), Lb=rect_length, Hb=rect_height)
        _refresh_geometry(mesh, level_set)
        domains = make_domain_sets(mesh, use_aligned_interface=True)
        cut_ids = mesh.element_bitset("cut").to_indices()
        interface_edges = mesh.edge_bitset("interface").to_indices()
        cut_counts.append(int(cut_ids.size))
        interface_counts.append(int(interface_edges.size))
        if cut_ids.size > 0 and interface_edges.size > 0:
            saw_both = True
        print(
            f"[stokes-refresh {step}] backend={backend} cy={center_y:.3e} "
            f"cut={cut_ids.size} interface_edges={interface_edges.size}"
        )
        assert cut_ids.size > 0
        if step == 0:
            assert interface_edges.size > 0

        theta_pos_vals[:] = np.clip(hansbo_cut_ratio(mesh, level_set, side="+"), theta_min, 1.0)
        theta_neg_vals[:] = np.clip(hansbo_cut_ratio(mesh, level_set, side="-"), theta_min, 1.0)
        refresh_sliver_weights(
            mesh,
            theta_pos_vals,
            theta_neg_vals,
            w_pos_vals,
            w_neg_vals,
            theta0=0.05,
            p=1.0,
            wmax=1000.0,
            thetamin=1.0e-6,
            smooth=0.5,
        )
        if cut_ids.size:
            thp = theta_pos_vals[cut_ids]
            thn = theta_neg_vals[cut_ids]
            wp = w_pos_vals[cut_ids]
            wn = w_neg_vals[cut_ids]
            print(
                "[sliver] min θ+={:.3e} min θ-={:.3e} max w+={:.3e} max w-={:.3e}".format(
                    float(thp.min()),
                    float(thn.min()),
                    float(wp.max()),
                    float(wn.max()),
                )
            )
            assert float(wp.max()) <= 1000.0 + 1.0e-12
            assert float(wn.max()) <= 1000.0 + 1.0e-12

        max_dev, checked, ifc_count = _kappa_sum_deviation(mesh, theta_pos_vals, theta_neg_vals)
        if ifc_count > 0:
            print(f"[kappa] edges_checked={checked}/{ifc_count} max|sum-1|={max_dev:.3e}")
            assert checked > 0
            assert max_dev < 1.0e-12
        else:
            print("[kappa] no interface edges to check on this step")

        _tag_inactive(dh, mesh)
        w_pos_cell = ElementWiseConstant(w_pos_vals)
        K, F = _build_stokes_system(
            mesh,
            dh,
            u,
            v,
            p,
            q,
            bcs,
            level_set=level_set,
            domains=domains,
            w_pos_cell=w_pos_cell,
            backend=backend,
        )
        sol = spla.spsolve(K, F)
        assert np.all(np.isfinite(sol))

    print(f"[stokes-refresh] cut_counts={cut_counts} interface_counts={interface_counts}")
    assert saw_both
    assert len(set(cut_counts)) > 1
    assert len(set(interface_counts)) > 1
