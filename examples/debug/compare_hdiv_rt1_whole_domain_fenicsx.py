from __future__ import annotations

import numpy as np
from mpi4py import MPI
from scipy.sparse import csr_matrix

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import ufl

from examples.debug.comparison_with_fenics import _build_rt1_local_transform, one_to_one_map_coords
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import HdivTestFunction, HdivTrialFunction, TestFunction, TrialFunction, div, grad, inner
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad

def _build_pycutfem_problem():
    nodes, elem_conn, edges, corner = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes,
        elem_conn,
        edges_connectivity=edges,
        elements_corner_nodes=corner,
        element_type="quad",
        poly_order=1,
    )
    mesh.tag_boundary_edges({"all": lambda x, y: True})

    me = MixedElement(mesh, {"u": ("RT", 1), "p": 1})
    dh = DofHandler(me, method="cg")

    du = HdivTrialFunction("u")
    v = HdivTestFunction("u")
    dp = TrialFunction("p", dof_handler=dh)
    q = TestFunction("p", dof_handler=dh)
    qmeta = {"q": 6}
    a = (
        inner(du, v)
        - dp * div(v)
        + q * div(du)
    ) * dx(metadata=qmeta)

    A_pc, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend="python")
    A_pc = A_pc.tocsr()

    x_pc = np.zeros(dh.total_dofs, dtype=float)
    u_slice = np.asarray(dh.get_field_slice("u"), dtype=int)
    p_slice = np.asarray(dh.get_field_slice("p"), dtype=int)
    x_pc[u_slice] = np.linspace(-0.3, 0.4, u_slice.size)
    p_xy = np.asarray(dh.get_dof_coords("p"), dtype=float)
    x_pc[p_slice] = 0.2 + 0.1 * p_xy[:, 0] - 0.05 * p_xy[:, 1]
    r_pc = np.asarray(A_pc @ x_pc, dtype=float)
    return me, dh, A_pc, r_pc, x_pc


def _build_fenicsx_problem(T_pc_to_fx: np.ndarray, x_pc: np.ndarray):
    mesh_fx = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 1, 1, dolfinx.mesh.CellType.quadrilateral)
    RT_fx = basix.ufl.element("RT", "quadrilateral", 2)
    P1_fx = basix.ufl.element("Lagrange", "quadrilateral", 1)
    W_el = basix.ufl.mixed_element([RT_fx, P1_fx])
    W_fx = dolfinx.fem.functionspace(mesh_fx, W_el)

    dw_fx = ufl.TrialFunction(W_fx)
    w_fx = ufl.TestFunction(W_fx)
    du_fx, dp_fx = ufl.split(dw_fx)
    v_fx, q_fx = ufl.split(w_fx)
    qdeg = 11

    a_fx = (
        ufl.inner(du_fx, v_fx)
        - dp_fx * ufl.div(v_fx)
        + q_fx * ufl.div(du_fx)
    ) * ufl.dx(metadata={"quadrature_degree": qdeg})

    A_fx = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a_fx))
    A_fx.assemble()
    indptr, indices, data = A_fx.getValuesCSR()
    A_fx = csr_matrix((data, indices, indptr), shape=A_fx.getSize())
    r_fx = np.asarray(A_fx @ np.asarray(T_pc_to_fx @ x_pc, dtype=float), dtype=float)
    return A_fx, r_fx


def _build_global_transform(me_pc: MixedElement, dh_pc: DofHandler, x_pc_len: int):
    mesh_fx = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 1, 1, dolfinx.mesh.CellType.quadrilateral)
    RT_fx = basix.ufl.element("RT", "quadrilateral", 2)
    P1_fx = basix.ufl.element("Lagrange", "quadrilateral", 1)
    W_el = basix.ufl.mixed_element([RT_fx, P1_fx])
    W_fx = dolfinx.fem.functionspace(mesh_fx, W_el)

    V_fx, V_to_W = W_fx.sub(0).collapse()
    Q_fx, Q_to_W = W_fx.sub(1).collapse()
    u_slice = np.asarray(dh_pc.get_field_slice("u"), dtype=int)
    p_slice = np.asarray(dh_pc.get_field_slice("p"), dtype=int)

    T = np.zeros((W_fx.dofmap.index_map.size_global, x_pc_len), dtype=float)
    B_rt = _build_rt1_local_transform(me_pc, V_fx, field="u")
    T[np.ix_(np.asarray(V_to_W, dtype=int), u_slice)] = np.asarray(B_rt, dtype=float)

    p_pc_coords = np.asarray(dh_pc.get_dof_coords("p"), dtype=float)
    p_fx_coords = Q_fx.tabulate_dof_coordinates()[:, :2]
    p_to_fx = one_to_one_map_coords(p_pc_coords, p_fx_coords)
    T[np.asarray(Q_to_W, dtype=int)[p_to_fx], p_slice] = 1.0
    return W_fx, T


def main():
    me_pc, dh_pc, A_pc, r_pc, x_pc = _build_pycutfem_problem()
    W_fx, T_pc_to_fx = _build_global_transform(me_pc, dh_pc, x_pc.size)
    A_fx, r_fx = _build_fenicsx_problem(T_pc_to_fx, x_pc)

    A_fx_in_pc = np.asarray(T_pc_to_fx.T @ A_fx @ T_pc_to_fx, dtype=float)
    r_fx_in_pc = np.asarray(T_pc_to_fx.T @ r_fx, dtype=float)

    np.testing.assert_allclose(A_pc.toarray(), A_fx_in_pc, rtol=0.0, atol=5.0e-12)
    np.testing.assert_allclose(r_pc, r_fx_in_pc, rtol=0.0, atol=5.0e-12)
    print("RT1 whole-domain H(div) volume mixed operator python vs FEniCSx: residual and Jacobian match.")


if __name__ == "__main__":
    main()
