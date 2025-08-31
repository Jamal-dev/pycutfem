
# test_mixed_ghost_grad_with_fenicsx.py
# Compare CutFEM ghost-penalty blocks for *normal-derivative jumps* on mixed fields
# (u_pos,u_neg,p_pos,p_neg) against FEniCSx assemblies on ghost_pos/ghost_neg.
#
# Blocks:
#   POS: d_n u_x, d_n u_y, d_n p  on ghost_pos
#   NEG: d_n u_x, d_n u_y, d_n p  on ghost_neg
#
# Default: no h-scaling. Add --use-h to multiply by avg(h) (Fenicsx) and h (CutFEM).
#
import argparse
import numpy as np
from mpi4py import MPI
import dolfinx
import ufl
import basix
import basix.ufl
from petsc4py import PETSc
import dolfinx.fem.petsc

# ---- CutFEM imports ---------------------------------------------------------
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    TrialFunction, TestFunction, Constant, Jump, CellDiameter, Grad, Dot, FacetNormal
)
from pycutfem.ufl.measures import dGhost
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.forms import Equation, assemble_form

# ------------------------------- Level set -----------------------------------
from pycutfem.core.levelset import AffineLevelSet

def classify_cell(phi_at_vertices, phi_at_centroid, tol=1e-12):
    mn = np.minimum(np.min(phi_at_vertices), phi_at_centroid)
    mx = np.maximum(np.max(phi_at_vertices), phi_at_centroid)
    if mx < -tol:
        return "inside"
    if mn > tol:
        return "outside"
    return "cut"

def locate_ghost_pos_neg(mesh, levelset, tol=1e-12):
    """
    Interior facets:
      - if one neighbor is 'cut' and the other is 'outside' -> ghost_pos
      - if one neighbor is 'cut' and the other is 'inside'  -> ghost_neg
      - exclude interface facets (sign change on endpoints)
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    mesh.topology.create_connectivity(tdim, 0)
    mesh.topology.create_connectivity(fdim, 0)
    f2c = mesh.topology.connectivity(fdim, tdim)
    c2v = mesh.topology.connectivity(tdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)
    x = mesh.geometry.x
    pos, neg = [], []
    for f in range(f2c.num_nodes):
        cells = f2c.links(f)
        if len(cells) != 2:
            continue
        vs = f2v.links(f)
        p0, p1 = x[vs[0]], x[vs[1]]
        phi0, phi1 = levelset(p0[:2]), levelset(p1[:2])
        if phi0 * phi1 < 0:
            continue
        tags = []
        for c in cells:
            vs_c = c2v.links(c)
            verts = x[vs_c][:, :2]
            phiv = np.array([levelset(v) for v in verts])
            centroid = verts.mean(axis=0)
            phic = levelset(centroid)
            tags.append(classify_cell(phiv, phic, tol))
        if tags.count("cut") == 1 and tags.count("outside") == 1:
            pos.append(f)
        elif tags.count("cut") == 1 and tags.count("inside") == 1:
            neg.append(f)
    return np.array(pos, dtype=np.int32), np.array(neg, dtype=np.int32)

def one_to_one_map(coords_pc, coords_fx):
    coords_pc = np.asarray(coords_pc, float)
    coords_fx = np.asarray(coords_fx, float)
    n = coords_pc.shape[0]
    used = np.zeros(coords_fx.shape[0], dtype=bool)
    idx = np.empty(n, dtype=int)
    for i in range(n):
        d2 = ((coords_fx - coords_pc[i])**2).sum(1)
        d2[used] = np.inf
        j = int(np.argmin(d2)); idx[i] = j; used[j] = True
    return idx

# --------------------------- Build both sides ---------------------------------
def setup_cut_mixed(nx=6, ny=4, order=1):
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=order)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, edges_connectivity=edges,
                elements_corner_nodes=corners, element_type="quad", poly_order=order)
    me = MixedElement(mesh, field_specs={
        "u_pos_x": order, "u_pos_y": order,
        "u_neg_x": order, "u_neg_y": order,
        "p_pos_":  order, "p_neg_":  order
    })
    dh = DofHandler(me, method="cg")
    return mesh, dh

def setup_fenicsx_scalar(nx=6, ny=4, order=1):
    from dolfinx import mesh as dmesh
    domain = dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [1.0, 1.0]],
        [nx, ny],
        dmesh.CellType.quadrilateral,
    )

    P_el = basix.ufl.element("Lagrange", "quadrilateral", order)
    Q = dolfinx.fem.functionspace(domain, P_el)
    p = ufl.TrialFunction(Q); q = ufl.TestFunction(Q)
    return domain, Q, p, q

def assemble_fx_grad_on_facets(Q, p, q, mesh_fx, facets, deg, use_h=False):
    if facets.size == 0:
        return None
    mt = dolfinx.mesh.meshtags(mesh_fx, mesh_fx.topology.dim-1,
                               facets, np.ones_like(facets, dtype=np.int32))
    dS = ufl.Measure("dS", domain=mesh_fx, subdomain_data=mt,
                     metadata={"quadrature_degree": deg})
    n = ufl.FacetNormal(mesh_fx)
    form = ufl.jump(ufl.grad(p), n) * ufl.jump(ufl.grad(q), n) * dS(1)
    if use_h:
        form = ufl.avg(ufl.CellDiameter(mesh_fx)) * form
    a_fx = dolfinx.fem.form(form)
    A = dolfinx.fem.petsc.assemble_matrix(a_fx); A.assemble()
    try:
        A_np = A[:, :]
    except Exception:
        A.convert(PETSc.Mat.Type.DENSE); A.assemble(); A_np = A.getDenseArray()
    return A_np

# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=6)
    ap.add_argument("--ny", type=int, default=4)
    ap.add_argument("--deg", type=int, default=6)
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--use-h", action="store_true")
    args = ap.parse_args()

    nx, ny, deg, order = args.nx, args.ny, args.deg, args.order
    use_h = args.use_h
    ls = AffineLevelSet(1.0, 0.0, -0.5)

    # CutFEM
    mesh_pc, dh = setup_cut_mixed(nx, ny, order)
    mesh_pc.classify_elements(ls); mesh_pc.classify_edges(ls)

    ghost_pos = mesh_pc.edge_bitset("ghost_pos")
    ghost_neg = mesh_pc.edge_bitset("ghost_neg")

    # scalar Trial/Test per component
    upx = TrialFunction("u_pos_x", dh); vpx = TestFunction("u_pos_x", dh)
    upy = TrialFunction("u_pos_y", dh); vpy = TestFunction("u_pos_y", dh)
    unx = TrialFunction("u_neg_x", dh); vnx = TestFunction("u_neg_x", dh)
    uny = TrialFunction("u_neg_y", dh); vny = TestFunction("u_neg_y", dh)
    ppos= TrialFunction("p_pos_",  dh); qpos= TestFunction("p_pos_", dh)
    pneg= TrialFunction("p_neg_",  dh); qneg= TestFunction("p_neg_", dh)

    gamma = Constant(1.0)
    h = CellDiameter()
    n = FacetNormal()

    dG_pos = dGhost(defined_on=ghost_pos, level_set=ls, metadata={"q": deg, "derivs": {(0,1), (1,0)}})
    dG_neg = dGhost(defined_on=ghost_neg, level_set=ls, metadata={"q": deg, "derivs": {(0,1), (1,0)}})

    scale = (gamma * h) if use_h else gamma

    # Build normal-derivative jump terms
    def dn(expr): return Dot(Grad(expr), n)

    A_pos_x = scale * Jump(dn(upx)) * Jump(dn(vpx)) * dG_pos
    A_pos_y = scale * Jump(dn(upy)) * Jump(dn(vpy)) * dG_pos
    A_pos_p = scale * Jump(dn(ppos)) * Jump(dn(qpos)) * dG_pos

    A_neg_x = scale * Jump(dn(unx)) * Jump(dn(vnx)) * dG_neg
    A_neg_y = scale * Jump(dn(uny)) * Jump(dn(vny)) * dG_neg
    A_neg_p = scale * Jump(dn(pneg)) * Jump(dn(qneg)) * dG_neg

    def assemble_pc(expr, backend):
        K, _ = assemble_form(Equation(expr, None), dof_handler=dh, bcs=[],
                             backend=backend, quad_order=deg)
        try:
            return K.toarray()
        except Exception:
            return np.array(K, float)

    blocks = [
        ("POS u_x", A_pos_x, "u_pos_x"),
        ("POS u_y", A_pos_y, "u_pos_y"),
        ("POS p",   A_pos_p, "p_pos_"),
        ("NEG u_x", A_neg_x, "u_neg_x"),
        ("NEG u_y", A_neg_y, "u_neg_y"),
        ("NEG p",   A_neg_p, "p_neg_"),
    ]

    # FEniCSx side: scalar space reused with facet normals
    mesh_fx, Q_fx, p_fx, q_fx = setup_fenicsx_scalar(nx, ny, order)
    pos_facets, neg_facets = locate_ghost_pos_neg(mesh_fx, ls)

    A_fx_pos = assemble_fx_grad_on_facets(Q_fx, p_fx, q_fx, mesh_fx, pos_facets, deg, use_h=use_h)
    A_fx_neg = assemble_fx_grad_on_facets(Q_fx, p_fx, q_fx, mesh_fx, neg_facets, deg, use_h=use_h)

    coords_fx = Q_fx.tabulate_dof_coordinates()[:, :2]

    def compare_block(name, K_pc_full, field_name, which):
        sl = dh.get_field_slice(field_name)
        Kb = K_pc_full[sl, :][:, sl]
        coords_pc = dh.get_dof_coords(field_name)
        fx_idx = one_to_one_map(coords_pc, coords_fx)
        A_ref = A_fx_pos if which == "POS" else A_fx_neg
        A_perm = A_ref[np.ix_(fx_idx, fx_idx)]
        diff = np.linalg.norm(Kb - A_perm, ord=np.inf)
        rel = diff / (np.linalg.norm(A_perm, ord=np.inf) + 1e-14)
        print(f"  {name:<8s}  ||Δ||_inf={diff:8.2e}   rel={rel:7.2e}")
        return diff, rel

    print("\\n=== Mixed ghost grad(n·∇) blocks: CutFEM vs FEniCSx ===")
    for backend in ["python", "jit"]:
        print(f"\\nBackend = {backend}")
        diffs = []
        for (label, expr, fname) in blocks:
            K_pc = assemble_pc(expr, backend=backend)
            which = "POS" if label.startswith("POS") else "NEG"
            d, r = compare_block(label, K_pc, fname, which)
            diffs.append(d)
        ok = max(diffs) < 1e-10
        print("  Result:", "MATCH ✅" if ok else "MISMATCH ⚠️")

if __name__ == "__main__":
    main()
