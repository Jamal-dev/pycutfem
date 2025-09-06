import numpy as np
import pytest

# mesh + core
from pycutfem.utils.meshgen import structured_quad, structured_triangles
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler

# UFL-like
from pycutfem.ufl.expressions import (TrialFunction, 
                                      TestFunction, 
                                      grad, 
                                      dot, 
                                      Jump, 
                                      CellDiameter, 
                                      Function,
                                      FacetNormal)
from pycutfem.ufl.measures import dGhost
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.core.levelset import AffineLevelSet
import matplotlib.pyplot as plt


# smooth target (analytic), shifted so interface is at x = x0
x0 = 0.5
def u_exact_xy(x, y):
    return np.exp(0.7*(x - x0)) * np.sin(np.pi*y)   # any smooth, non-polynomial works


def _make_mesh(elem_type: str, p: int, nx: int, ny: int) -> Mesh:
    Lx = Ly = 1.0
    if elem_type == "quad":
        nodes, elems, edges, corners = structured_quad(Lx, Ly, nx=nx, ny=ny, poly_order=p)
        return Mesh(nodes, element_connectivity=elems, edges_connectivity=edges,
                    elements_corner_nodes=corners, element_type="quad", poly_order=p)
    elif elem_type == "tri":
        nodes, elems, edges, corners = structured_triangles(Lx, Ly, nx_quads=nx, ny_quads=ny, poly_order=p)
        return Mesh(nodes, elems, edges, corners, element_type="tri", poly_order=p)
    else:
        raise ValueError(elem_type)


def ghost_energy(elem_type: str, p: int, nx=14, ny=14) -> float:
    mesh = _make_mesh(elem_type, p, nx, ny)

    # Level set strictly inside → defines ghost patches around Γ
    ls = AffineLevelSet(1.0, 0.0, -x0).normalised()  # φ(x,y)=x-x0
    mesh.classify_elements(ls)
    mesh.classify_edges(ls)
    mesh.build_interface_segments(ls)

    # Ghost patches on both sides (include interface set too)
    ghost_pos = mesh.edge_bitset("ghost_pos") | mesh.edge_bitset("ghost_both") | mesh.edge_bitset("interface")
    ghost_neg = mesh.edge_bitset("ghost_neg") | mesh.edge_bitset("ghost_both") | mesh.edge_bitset("interface")

    # FE space + handler
    me = MixedElement(mesh, field_specs={'u': p})
    dh = DofHandler(me, method='cg')

    # Build nodal interpolant of u_exact: u(gd) = u_exact(x_gd, y_gd)
    dh._ensure_dof_coords()
    ids = np.asarray(dh.get_field_slice('u'), dtype=int)
    XY  = dh._dof_coords[ids]
    u_vals = np.array([u_exact_xy(float(x), float(y)) for (x, y) in XY], dtype=float)

    uh = Function("u_h", field_name='u', dof_handler=dh)   # just for convenience if needed elsewhere
    uh.set_nodal_values(ids, u_vals)

    # Ghost-only bilinear form: J = u^T K_gp u  with K_gp assembled from ghost terms only
    u = TrialFunction('u', dh)
    v = TestFunction ('u', dh)
    nQ = max(2*p + 4, 8)
    h  = CellDiameter()

    n = FacetNormal()

    # normal-derivative jump on ghost patches
    jn_tr = dot(Jump(grad(u)), n)   # Jump(...) then dot with facet normal of ls
    jn_te = dot(Jump(grad(v)), n)

    dG_pos = dGhost(defined_on=ghost_pos, level_set=ls, metadata={'q': nQ, 'derivs': {(1,0),(0,1)}})
    dG_neg = dGhost(defined_on=ghost_neg, level_set=ls, metadata={'q': nQ, 'derivs': {(1,0),(0,1)}})

    gamma_g = 1.0   # scale is irrelevant for monotonicity; keep =1
    a = gamma_g * h * jn_tr * jn_te * dG_pos \
      + gamma_g * h * jn_tr * jn_te * dG_neg

    eq = Equation(a, None)
    K, _ = assemble_form(eq, dof_handler=dh, bcs=None, quad_order=nQ, backend='python')

    # Quadratic form u^T K u
    energy = float(u_vals @ (K[ids][:, ids] @ u_vals))   # restrict to field slice 'u'
    assert np.isfinite(energy), "Non-finite ghost energy (check h and normals)"
    return energy

def check_monotonicity(errors):
    prev = None
    for e in errors:
        if prev is not None:
            assert e <= prev * 1.05 + 1e-14, "Error did not decrease with p"
        prev = e

def _run_suite(elem_type: str, Ps=(1,2,3,4,5), nx=14, ny=14):
    print(f"\n== Ghost-only seminorm p-refinement ({elem_type}) ==")
    print(f"{'p':>2}  {'J_gp(u_I)':>14}")
    prev = None
    J_values = []
    for p in Ps:
        J = ghost_energy(elem_type, p, nx, ny)
        print(f"{p:2d}  {J:14.6e}")
        J_values.append(J)
        # if prev is not None:
        #     # monotone decay with a little slack (conditioning)
        #     assert J <= prev * 1.10 + 1e-14
        # prev = J

    # Return all J values for further analysis if needed
    return J_values


def test_ghost_only_quad():
    J_values = _run_suite("quad", Ps=(1,2,3,4,5), nx=12, ny=12)
    check_monotonicity(J_values)
    plt.plot(J_values, label="Quad")
    plt.xlabel("Polynomial Degree p")
    plt.ylabel("Ghost Energy J")
    plt.title("Ghost Energy vs. Polynomial Degree (Quad)")
    plt.legend()
    plt.show()

def test_ghost_only_tri():
    J_values = _run_suite("tri",  Ps=(1,2,3,4,5), nx=16, ny=16)
    check_monotonicity(J_values)
    plt.plot(J_values, label="Tri")
    plt.xlabel("Polynomial Degree p")
    plt.ylabel("Ghost Energy J")
    plt.title("Ghost Energy vs. Polynomial Degree (Tri)")
    plt.legend()
    plt.show()
