"""
Isolated finite-difference check for the fluid ghost-velocity stabilization.

This strips the full FSI model down to:
  • a structured quad mesh cut by a vertical level set (x = 1),
  • one velocity field on the + side,
  • the ghost penalty term 2*mu*gamma*h * ⟦∇u⟧·n ⟦∇v⟧·n on ghost_pos|ghost_both,
and prints the per-DoF FD vs assembled Jacobian error.

Usage:
    RUN_FD_CHECK=1 FD_BACKEND=python POLY_ORDER=1 MESH_SIZE=0.05 \
        python examples/debug/ghost_fluid_fd.py
"""
from __future__ import annotations

import os
import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.utils.meshgen import structured_quad
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler

from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.expressions import (
    Function,
    VectorFunction,
    VectorTrialFunction,
    VectorTestFunction,
    restrict,
    CellDiameter,
    FacetNormal,
    Constant,
    jump,
    grad,
    dot,
    inner,
)
from pycutfem.ufl.measures import dGhost
from pycutfem.ufl.forms import Equation, BoundaryCondition
from pycutfem.ufl.compilers import FormCompiler

# --------------------------------------------------------------------------- #
# Problem setup (tiny stand-alone cut mesh)
# --------------------------------------------------------------------------- #
POLY_ORDER = int(os.getenv("POLY_ORDER", "1"))
MESH_SIZE = float(os.getenv("MESH_SIZE", "0.05"))

# Use a small structured quad mesh (2 x 1 rectangle) with a vertical cut at x=1.
L, H = 2.0, 1.0
nx = max(8, int(L / MESH_SIZE))
ny = max(4, int(H / MESH_SIZE))
nodes, elements, edges, corners = structured_quad(L, H, nx=nx, ny=ny, poly_order=POLY_ORDER)
mesh = Mesh(
    nodes=nodes,
    element_connectivity=elements,
    edges_connectivity=edges,
    elements_corner_nodes=corners,
    element_type="quad",
    poly_order=POLY_ORDER,
)

# Classify with a simple level set: φ(x,y) = x - 1 (positive on the fluid/right).
level_set = AffineLevelSet(a=1.0, b=0.0, c=-1.0)
mesh.classify_elements(level_set)
mesh.build_interface_segments(level_set)
mesh.classify_edges(level_set)

# --------------------------------------------------------------------------- #
# FE spaces / functions (fluid velocity only, + side)
# --------------------------------------------------------------------------- #
me = MixedElement(
    mesh,
    field_specs={
        "u_pos_x": POLY_ORDER,
        "u_pos_y": POLY_ORDER,
        "p_pos_": POLY_ORDER - 1,
        "vs_neg_x": POLY_ORDER - 1,
        "vs_neg_y": POLY_ORDER - 1,
        "d_neg_x": POLY_ORDER - 1,
        "d_neg_y": POLY_ORDER - 1,
    },
)
dh = DofHandler(me, method="cg")
# Pad the union size if the static estimate is too small for the current mesh/fields.
try:
    ghost_ids = mesh.edge_bitset("ghost").to_indices()
except Exception:
    ghost_ids = []
max_union = me.n_union_cg
for gid in ghost_ids:
    e = mesh.edge(int(gid))
    if e.right is None:
        continue
    pos_dofs = dh.get_elemental_dofs(e.left)
    neg_dofs = dh.get_elemental_dofs(e.right)
    max_union = max(max_union, len(np.unique(np.concatenate([pos_dofs, neg_dofs]))))
if max_union > me.n_union_cg:
    me.n_union_cg = max_union

vel_space = FunctionSpace(name="vel", field_names=["u_pos_x", "u_pos_y"], dim=1, side="+")
du = VectorTrialFunction(space=vel_space, dof_handler=dh)
v = VectorTestFunction(space=vel_space, dof_handler=dh)
u_k = VectorFunction(name="u_k", field_names=["u_pos_x", "u_pos_y"], dof_handler=dh, side="+")
u_k.nodal_values.fill(0.0)

# No Dirichlet BCs; keep API compatible with finite_difference_check.
bcs: list[BoundaryCondition] = []
bcs_homog: list[BoundaryCondition] = []

# --------------------------------------------------------------------------- #
# Ghost penalty term (velocity-only part of the fluid ghost stabilization)
# --------------------------------------------------------------------------- #
n = FacetNormal()
cell_h = CellDiameter()
mu_f = Constant(1.0e-3)
gamma = Constant(0.1 * POLY_ORDER**2)


def grad_inner_jump(u, v):
    return inner(dot(jump(grad(u)), n), dot(jump(grad(v)), n))


def g_v_f(gamma, phi_1, phi_2):
    return gamma * (cell_h * grad_inner_jump(phi_1, phi_2))


# Ghost measure on + side (include ghost_both to mimic production setup).
ghost_pos = mesh.edge_bitset("ghost_pos") | mesh.edge_bitset("ghost_both")
dG = dGhost(defined_on=ghost_pos, level_set=level_set, metadata={"q": POLY_ORDER + 2, "derivs": {(1, 0), (0, 1)}})

# Restrict to the (+) domain (outside ∪ cut) to mimic the FSI setup.
has_pos = mesh.element_bitset("outside") | mesh.element_bitset("cut")
du_R = restrict(du, has_pos)
v_R = restrict(v, has_pos)
u_k_R = restrict(u_k, has_pos)

jac_form = (Constant(2.0) * mu_f * g_v_f(gamma, du_R, v_R)) * dG
res_form = (Constant(2.0) * mu_f * g_v_f(gamma, u_k_R, v_R)) * dG

# --------------------------------------------------------------------------- #
# Minimal FD Jacobian check (adapted from the full FSI driver)
# --------------------------------------------------------------------------- #


def select_fd_dofs(dh: DofHandler, fields_to_probe: dict[str, int], elem_tag: str = "cut") -> np.ndarray:
    selected: list[int] = []
    elems = dh.element_bitset(elem_tag).to_indices()
    probe_eid = int(elems[0]) if len(elems) else 0
    for field, count in fields_to_probe.items():
        try:
            local = dh.element_dofs(field, probe_eid)
        except Exception:
            local = []
        selected.extend(list(local[:count]))
    return np.array(sorted(set(selected)), dtype=int)


def finite_difference_check(
    jac_form,
    res_form,
    dh: DofHandler,
    bcs: list[BoundaryCondition],
    functions: dict[str, Function | VectorFunction],
    probe_dofs: np.ndarray,
    eps: float = 1.0e-7,
) -> None:
    backend = os.getenv("FD_BACKEND", "jit")
    compiler = FormCompiler(dh, backend=backend)
    eq = Equation(jac_form, res_form)
    base_K, base_R = compiler.assemble(eq, bcs=bcs)
    if base_K is None or base_R is None:
        print("Skipping FD check: Jacobian or residual form missing.")
        return

    rows = []
    bc_dofs = set(dh.get_dirichlet_data(bcs).keys())
    inactive = set(dh.dof_tags.get("inactive", set()))
    for gdof in probe_dofs:
        field, _ = dh._dof_to_node_map[int(gdof)]
        if field not in functions:
            continue
        if int(gdof) in bc_dofs or int(gdof) in inactive:
            continue
        func = functions[field]
        old_val = func.get_nodal_values(np.array([gdof], dtype=int))[0]
        func.set_nodal_values(np.array([gdof], dtype=int), np.array([old_val + eps], dtype=float))
        K_plus, R_plus = compiler.assemble(eq, bcs=bcs)
        func.set_nodal_values(np.array([gdof], dtype=int), np.array([old_val - eps], dtype=float))
        K_minus, R_minus = compiler.assemble(eq, bcs=bcs)
        func.set_nodal_values(np.array([gdof], dtype=int), np.array([old_val], dtype=float))
        fd_col = (R_plus - R_minus) / (2 * eps)
        jac_col = base_K[:, int(gdof)].toarray().ravel()
        err_vec = fd_col - jac_col
        err = np.linalg.norm(err_vec, ord=np.inf)
        mag = np.linalg.norm(jac_col, ord=np.inf)
        rel = err / (mag + 1.0e-14)
        rows.append((gdof, field, err, mag, rel))
    print("Finite-difference Jacobian check (gdof, field, err, |J|, rel):")
    for gd, fld, err, mag, rel in rows:
        print(f"  {gd:5d}  {fld:10s}  err={err:9.3e}  |J|={mag:9.3e}  rel={rel:9.3e}")


if __name__ == "__main__":
    probe = select_fd_dofs(dh, {"u_pos_x": 2, "u_pos_y": 2}, elem_tag="cut")
    functions = {"u_pos_x": u_k, "u_pos_y": u_k}
    finite_difference_check(jac_form, res_form, dh, bcs_homog, functions, probe, eps=1.0e-7)
