from __future__ import annotations

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from scipy.sparse import csr_matrix

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import ufl

from examples.utils.biofilm.one_domain import _tangential_component_2d
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.fem.reference.rt import _edge_normal, _edge_param, _legendre_all, _monomials_quad, gauss_legendre, _quad_rule
from pycutfem.ufl.expressions import FacetNormal, HdivFunction, HdivTestFunction, HdivTrialFunction, dot, grad, inner
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dS
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
    me = MixedElement(mesh, {"u": ("RT", 1)})
    dh = DofHandler(me, method="cg")

    u = HdivTrialFunction("u")
    v = HdivTestFunction("u")
    u_k = HdivFunction(name="u_k", field_name="u", dof_handler=dh)
    rng = np.random.default_rng(29)
    u_k.nodal_values[:] = rng.standard_normal(u_k.nodal_values.size)

    n = FacetNormal()

    def tcomp(w):
        return _tangential_component_2d(w, n)

    qmeta = {"q": 4}
    a = (inner(grad(u), grad(v)) - tcomp(dot(2.0 * grad(u), n)) * tcomp(v)) * dS(metadata=qmeta)
    L = (inner(grad(u_k), grad(v)) - tcomp(dot(2.0 * grad(u_k), n)) * tcomp(v)) * dS(metadata=qmeta)
    K_pc, F_pc = assemble_form(Equation(a, L), dof_handler=dh, bcs=[], backend="python")
    return me, dh, u_k, K_pc.tocsr(), np.asarray(F_pc, dtype=float)


def _evaluate_fenicsx_basis_values(V_fx, pts_phys: np.ndarray) -> np.ndarray:
    pts_phys = np.asarray(pts_phys, dtype=float)
    ndofs = int(V_fx.dofmap.index_map.size_global)
    cells = np.array([0], dtype=np.int32)
    values = np.empty((pts_phys.shape[0], ndofs, 2), dtype=float)
    # dolfinx.fem.Expression on quadrilateral cells expects the local point
    # coordinates in the cell-reference ordering used internally by FFCx,
    # which is the reverse of the (x, y) ordering we use elsewhere here.
    expr_points = np.ascontiguousarray(pts_phys[:, ::-1], dtype=float)
    basis_fn = dolfinx.fem.Function(V_fx)
    for j in range(ndofs):
        basis_fn.x.array[:] = 0.0
        basis_fn.x.array[int(j)] = 1.0
        basis_fn.x.scatter_forward()
        expr = dolfinx.fem.Expression(basis_fn, expr_points)
        raw = np.asarray(expr.eval(V_fx.mesh, cells), dtype=float).reshape(1, pts_phys.shape[0], 2)
        values[:, j, :] = raw[0]
    return values


def _build_rt1_local_transform(me_pc: MixedElement, V_fx) -> np.ndarray:
    ref_pc = me_pc._ref["u"]
    nloc = int(ref_pc.n_dofs)
    if ref_pc.element_type != "quad" or int(ref_pc.k) != 1 or nloc != 12:
        raise NotImplementedError("This helper currently targets RT1 on a single quadrilateral only.")

    C_fx_to_pc = np.zeros((nloc, nloc), dtype=float)
    row = 0

    # Edge moments on the pycutfem reference [-1,1]^2, evaluated through the
    # physical unit-square FEniCSx basis. The 0.5 factor is the inverse Piola
    # pullback scale det(J) J^{-1} for the affine map [-1,1]^2 -> [0,1]^2.
    s, w = gauss_legendre(4)
    P = _legendre_all(1, s)
    for edge in range(4):
        xi_ref, eta_ref, w_scale = _edge_param("quad", edge, s)
        pts_phys = np.column_stack((0.5 * (xi_ref + 1.0), 0.5 * (eta_ref + 1.0)))
        vals_fx = _evaluate_fenicsx_basis_values(V_fx, pts_phys)
        nvec = _edge_normal("quad", edge)
        flux = vals_fx[:, :, 0] * float(nvec[0]) + vals_fx[:, :, 1] * float(nvec[1])
        ww = (0.5 * w * w_scale).reshape(-1, 1)
        for mode in range(2):
            C_fx_to_pc[row, :] = np.sum(ww * P[mode][:, None] * flux, axis=0)
            row += 1

    # Cell moments: component-wise against Q_{0,1} and Q_{1,0} on the
    # pycutfem reference cell, again using the inverse Piola pullback scale.
    qp_ref, qw_ref = _quad_rule(4)
    qp_ref = np.asarray(qp_ref, dtype=float)
    qw_ref = np.asarray(qw_ref, dtype=float)
    pts_phys = 0.5 * (qp_ref + 1.0)
    vals_fx = _evaluate_fenicsx_basis_values(V_fx, pts_phys)
    cell_monos = ((_monomials_quad(0, 1), 0), (_monomials_quad(1, 0), 1))
    for monos, comp in cell_monos:
        for i, j in monos:
            weight = (0.5 * qw_ref * (qp_ref[:, 0] ** int(i)) * (qp_ref[:, 1] ** int(j))).reshape(-1, 1)
            C_fx_to_pc[row, :] = np.sum(weight * vals_fx[:, :, int(comp)], axis=0)
            row += 1

    if np.linalg.matrix_rank(C_fx_to_pc) != nloc:
        raise RuntimeError("RT1 local FEniCSx->pycutfem transform is singular.")
    B_pc_to_fx = np.linalg.inv(C_fx_to_pc)
    # DOLFINx/Basix quadrilateral RT1 uses an additional local sign on the odd
    # edge modes of facets 0 and 2 in the assembled cell basis. Account for it
    # here so the transformed coefficients live in the same local basis that the
    # FEniCSx matrix/vector assembly uses.
    fx_sign = np.ones((nloc,), dtype=float)
    fx_sign[1] = -1.0
    fx_sign[5] = -1.0
    return fx_sign[:, None] * B_pc_to_fx


def _build_fenicsx_problem(B_pc_to_fx: np.ndarray, u_k_pc: np.ndarray):
    mesh_fx = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 1, 1, dolfinx.mesh.CellType.quadrilateral)
    RT_fx = basix.ufl.element("RT", "quadrilateral", 2)
    V_fx = dolfinx.fem.functionspace(mesh_fx, RT_fx)

    du_fx = ufl.TrialFunction(V_fx)
    v_fx = ufl.TestFunction(V_fx)
    u_k_fx = dolfinx.fem.Function(V_fx)
    u_k_fx.x.array[:] = np.asarray(B_pc_to_fx @ u_k_pc, dtype=float)
    u_k_fx.x.scatter_forward()

    n_fx = ufl.FacetNormal(mesh_fx)
    t_fx = ufl.as_vector((n_fx[1], -n_fx[0]))
    qdeg = 4
    a_fx = (
        ufl.inner(ufl.grad(du_fx), ufl.grad(v_fx))
        - ufl.dot(2.0 * ufl.grad(du_fx) * n_fx, t_fx) * ufl.dot(v_fx, t_fx)
    ) * ufl.ds(metadata={"quadrature_degree": qdeg})
    L_fx = (
        ufl.inner(ufl.grad(u_k_fx), ufl.grad(v_fx))
        - ufl.dot(2.0 * ufl.grad(u_k_fx) * n_fx, t_fx) * ufl.dot(v_fx, t_fx)
    ) * ufl.ds(metadata={"quadrature_degree": qdeg})

    A_fx = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a_fx))
    A_fx.assemble()
    indptr, indices, data = A_fx.getValuesCSR()
    K_fx = csr_matrix((data, indices, indptr), shape=A_fx.getSize()).toarray()

    F_vec = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(L_fx))
    F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    F_fx = np.asarray(F_vec.array, dtype=float)
    return K_fx, F_fx


def main():
    me_pc, _dh_pc, u_k_pc, K_pc, F_pc = _build_pycutfem_problem()

    mesh_fx = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 1, 1, dolfinx.mesh.CellType.quadrilateral)
    RT_fx = basix.ufl.element("RT", "quadrilateral", 2)
    V_fx = dolfinx.fem.functionspace(mesh_fx, RT_fx)
    B_pc_to_fx = _build_rt1_local_transform(me_pc, V_fx)

    K_fx, F_fx = _build_fenicsx_problem(B_pc_to_fx, np.asarray(u_k_pc.nodal_values, dtype=float))
    K_fx_in_pc = np.asarray(B_pc_to_fx.T @ K_fx @ B_pc_to_fx, dtype=float)
    F_fx_in_pc = np.asarray(B_pc_to_fx.T @ F_fx, dtype=float)

    np.testing.assert_allclose(K_pc.toarray(), K_fx_in_pc, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(F_pc, F_fx_in_pc, rtol=0.0, atol=2.0e-12)
    print("RT1 boundary H(div) python vs FEniCSx: residual and Jacobian match.")


if __name__ == "__main__":
    main()
