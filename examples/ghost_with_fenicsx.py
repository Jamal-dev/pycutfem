
# test_ghost_with_fenicsx.py
# Compare a CutFEM ghost penalty assembly vs. an equivalent FEniCSx dS integral
#
# Form tested (scalar pressure space):
#     a(p, q) = gamma * jump(p) * jump(q) over GHOST interior facets
#
# We deliberately omit h-scaling to avoid owner-side ambiguities and focus on
# validating sided DOF padding and jump logic.
#
# How the comparison works:
# - We generate the same structured quadrilateral mesh in both frameworks.
# - We classify 'ghost' facets from the same affine level set phi(x,y)=x-0.5:
#   a facet is GHOST if exactly one adjacent cell is 'cut' and the other
#   is strictly 'inside' or strictly 'outside', and the facet does not lie
#   on the interface itself.
# - We assemble the bilinear form in both frameworks and compare matrices after
#   reordering FEniCSx DOFs to the CutFEM ordering by matching coordinates.
#
# Run:  python test_ghost_with_fenicsx.py
# Requires: pycutfem, dolfinx, basix, petsc4py, mpi4py

import numpy as np
from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import basix
import basix.ufl
from petsc4py import PETSc

# ---- CutFEM imports (mirror your comparison harness style) ------------------
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import TrialFunction, TestFunction, Constant, Jump
from pycutfem.ufl.measures import dGhost
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.core.levelset import AffineLevelSet, CircleLevelSet



# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def classify_cell(phi_at_vertices, phi_at_centroid, tol=1e-12):
    # Replicate CutFEM classification: 'inside' (all < -tol), 'outside' (all > tol), else 'cut'.
    mn = np.minimum(np.min(phi_at_vertices), phi_at_centroid)
    mx = np.maximum(np.max(phi_at_vertices), phi_at_centroid)
    if mx < -tol:
        return "inside"
    if mn > tol:
        return "outside"
    return "cut"


def locate_ghost_facets_fenicsx(mesh, levelset, tol=1e-12):
    '''
    Return an array of interior-facet indices (local to the mesh) that satisfy:
      - not interface: phi changes sign at facet endpoints -> exclude
      - exactly one adjacent cell is 'cut' and the other is not
    '''
    tdim = mesh.topology.dim
    fdim = tdim - 1

    # Build needed connectivities
    mesh.topology.create_connectivity(fdim, tdim)  # facets -> cells
    mesh.topology.create_connectivity(tdim, 0)     # cells  -> vertices
    mesh.topology.create_connectivity(fdim, 0)     # facets -> vertices

    f2c = mesh.topology.connectivity(fdim, tdim)
    c2v = mesh.topology.connectivity(tdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)

    ghost_facets = []

    x = mesh.geometry.x

    for f in range(f2c.num_nodes):  # iterate facets
        cells = f2c.links(f)
        if len(cells) != 2:
            continue  # boundary facet

        # Check if the facet itself crosses the interface
        vs = f2v.links(f)
        p0, p1 = x[vs[0]], x[vs[1]]
        phi0, phi1 = levelset(p0[:2]), levelset(p1[:2])
        if phi0 * phi1 < 0:  # interface edge -> skip for ghost
            continue

        tags = []
        for c in cells:
            vs_c = c2v.links(c)
            verts = x[vs_c][:, :2]
            phi_verts = np.array([levelset(v) for v in verts])
            centroid = verts.mean(axis=0)
            phi_cent = levelset(centroid)
            tags.append(classify_cell(phi_verts, phi_cent, tol=tol))

        tags = set(tags)
        if "cut" in tags and len(tags) == 2:  # {"cut","inside"} or {"cut","outside"}
            ghost_facets.append(f)

    return np.array(ghost_facets, dtype=np.int32)


def one_to_one_map_coords(coords_pc, coords_fx):
    # Hungarian assignment to match DOF coordinates across frameworks.
    from scipy.optimize import linear_sum_assignment
    C = np.linalg.norm(coords_fx[:, None, :] - coords_pc[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(C)
    return rows[np.argsort(cols)]  # fx indices ordered to match pc order


# -----------------------------------------------------------------------------
# Build both problems
# -----------------------------------------------------------------------------
def setup_cut_p1(nx=4, ny=2):
    # Mesh + discrete space (scalar Q1 only)
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=1)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, edges_connectivity=edges,
                elements_corner_nodes=corners, element_type="quad", poly_order=1)
    me = MixedElement(mesh, field_specs={"p": 1})
    dh = DofHandler(me, method="cg")

    # Space and symbols
    Q = FunctionSpace("pressure", ["p"], dim=0)
    dp = TrialFunction(Q, dh)
    q  = TestFunction(Q, dh)

    return mesh, dh, {"dp": dp, "q": q}


def setup_fenicsx_p1(nx=4, ny=2):
    from dolfinx import mesh as dmesh

    domain = dmesh.create_rectangle(
        MPI.COMM_WORLD,
        points=((0.0, 0.0), (1.0, 1.0)),
        n=(nx, ny),
        cell_type=dmesh.CellType.quadrilateral,
    )
    P1_el = basix.ufl.element("Lagrange", "quadrilateral", 1)
    Q = dolfinx.fem.functionspace(domain, P1_el)
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)
    return domain, Q, p, q


# -----------------------------------------------------------------------------
# Main test
# -----------------------------------------------------------------------------
def main():
    # Mesh resolution (must match between frameworks)
    nx, ny = 4, 2
    ls = AffineLevelSet(a=1.0, b=0.0, c=-0.5)  # phi(x,y)=x-0.5
    backends = {"python": "python", "jit": "jit"}

    # ----- CutFEM side -----
    mesh_pc, dh, pc = setup_cut_p1(nx, ny)

    # classify for ghost measure
    mesh_pc.classify_elements(ls)
    mesh_pc.classify_edges(ls)
    ghost_set = mesh_pc.edge_bitset("ghost")

    gamma = Constant(10.0)
    deg = 4
    int_ghost = dGhost(defined_on=ghost_set, level_set=ls, metadata={"q": deg})
    a_pc = gamma * Jump(pc["dp"]) * Jump(pc["q"]) * int_ghost

    J_pc_dic = {}
    for name, backend in backends.items():
        J_pc, _ = assemble_form(Equation(a_pc, None), dof_handler=dh, bcs=[], backend=backend)
        J_pc_dic[name] = J_pc

    # ----- FEniCSx side -----
    mesh_fx, Q_fx, p_fx, q_fx = setup_fenicsx_p1(nx, ny)

    # tag ghost facets identically
    ghost_facets = locate_ghost_facets_fenicsx(mesh_fx, ls, tol=1e-12)
    if ghost_facets.size == 0:
        raise RuntimeError("No ghost facets found; increase mesh resolution or check level set.")

    values = np.full(ghost_facets.shape, 1, dtype=np.int32)  # tag id = 1
    ghost_mt = dolfinx.mesh.meshtags(mesh_fx, mesh_fx.topology.dim - 1, ghost_facets, values)

    # put quadrature degree on the measure
    dS_ghost = ufl.Measure(
        "dS", domain=mesh_fx, subdomain_data=ghost_mt,
        metadata={"quadrature_degree": deg}
    )
    gamma_fx = dolfinx.fem.Constant(mesh_fx, 10.0)
    a_fx = dolfinx.fem.form(gamma_fx * ufl.jump(p_fx) * ufl.jump(q_fx) * dS_ghost(1))

    A = dolfinx.fem.petsc.assemble_matrix(a_fx)
    A.assemble()

    # Convert PETSc Mat to dense NumPy (small problems only)
    A_fx = A[:, :]

    # ----- Reorder FEniCSx DOFs to CutFEM order by matching coordinates -----
    coords_pc = dh.get_dof_coords("p")  # (ndofs, 2)
    coords_fx = Q_fx.tabulate_dof_coordinates()[:, :2]

    # Map from pc-order to fx-order
    fx_indices = one_to_one_map_coords(coords_pc, coords_fx)

    # Permute A_fx into pc ordering
    A_fx_perm = A_fx[np.ix_(fx_indices, fx_indices)]

    # ----- Compare -----
    for name, J_pc in J_pc_dic.items():
        J = J_pc  # already a dense or CSR array-like? ensure dense for comparison
        try:
            J_pc_dense = J.toarray() if hasattr(J, "toarray") else np.array(J, dtype=float)
        except Exception:
            J_pc_dense = np.array(J, dtype=float)

        diff = np.linalg.norm(J_pc_dense - A_fx_perm, ord=np.inf)
        rel = diff / (np.linalg.norm(J_pc_dense, ord=np.inf) + 1e-14)
        print("\n=== Ghost penalty comparison ( CutFEM vs FEniCSx ) ===")
        print(f"  backend = {name}")
        print(f"  shape pc = {J_pc_dense.shape}, fx = {A_fx.shape}")
        print(f"  ||A_pc - A_fx||_inf = {diff:.3e}")
        print(f"  rel error (inf-norm) = {rel:.3e}")
        if diff < 1e-10:
            print("  MATCH (within 1e-10)")
        else:
            print("  MISMATCH — inspect facet tagging or quadrature settings")

if __name__ == '__main__':
    main()
