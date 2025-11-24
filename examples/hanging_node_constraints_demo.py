"""
Minimal demonstration of the hanging-node constraint machinery.

We create two Q2 quadrilateral elements that share the vertical edge x=1.
The left element deliberately *reuses* its bottom-right corner node for the
mid-edge position, so its edge nodes are {bottom, top}. The right element
uses the proper mid-edge node, so its edge-node set is {bottom, mid, top}.
The detection routine marks the mid-edge DOF on the right as a slave and
expresses it as a linear combination of the coarse-edge masters.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pycutfem.core.topology import Node
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler


def _q2_rect_nodes(x0: float, x1: float, y0: float, y1: float) -> list[tuple[float, float]]:
    xs = [x0, 0.5 * (x0 + x1), x1]
    ys = [y0, 0.5 * (y0 + y1), y1]
    return [(x, y) for y in ys for x in xs]


def build_hanging_mesh() -> Mesh:
    # Left element: [0,1] x [0,1]
    left_coords = _q2_rect_nodes(0.0, 1.0, 0.0, 1.0)
    nodes = [Node(i, x, y) for i, (x, y) in enumerate(left_coords)]

    # Right element shares the interface x=1 and extends to x=2
    right_extra = [
        (1.5, 0.0),
        (2.0, 0.0),
        (1.5, 0.5),
        (2.0, 0.5),
        (1.5, 1.0),
        (2.0, 1.0),
    ]
    base_id = len(nodes)
    for j, (x, y) in enumerate(right_extra):
        nodes.append(Node(base_id + j, x, y))

    # Element connectivity (Q2, row-major in the 3x3 lattice)
    # Left element drops the mid-edge node on x=1 by reusing the corner id (node 2)
    left_conn = [0, 1, 2, 3, 4, 2, 6, 7, 8]
    right_conn = [
        2,              # (1,0)
        base_id + 0,    # (1.5,0)
        base_id + 1,    # (2,0)
        5,              # (1,0.5)  ← extra node not seen by the left element
        base_id + 2,    # (1.5,0.5)
        base_id + 3,    # (2,0.5)
        8,              # (1,1)
        base_id + 4,    # (1.5,1)
        base_id + 5,    # (2,1)
    ]

    corner_conn = np.array([[0, 2, 8, 6], [2, base_id + 1, base_id + 5, 8]], dtype=int)
    elem_conn = np.array([left_conn, right_conn], dtype=int)

    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elem_conn,
        elements_corner_nodes=corner_conn,
        element_type="quad",
        poly_order=2,
    )
    return mesh


def main():
    mesh = build_hanging_mesh()
    # Use Q2 field so that edge midpoints carry DOFs that become slaves.
    me = MixedElement(mesh, {"u": 2})
    dh = DofHandler(me, method="cg")

    constraints = dh.build_hanging_node_constraints()
    if constraints is None:
        print("No hanging nodes detected.")
        return

    print(f"Master DOFs: {constraints.master_ids.tolist()}")
    print(f"Slave -> master weights: {constraints.slave_to_master}")

    ndof = dh.total_dofs
    K_full = sp.eye(ndof, format="csr")

    # Choose a reference master solution and assemble the consistent RHS
    u_master_true = np.linspace(1.0, float(constraints.n_master), constraints.n_master)
    u_full_true = constraints.prolong(u_master_true)
    F_full = K_full @ u_full_true

    # Condense: A_red = Eᵀ A E, F_red = Eᵀ F
    A_red = constraints.E_T @ (K_full @ constraints.E)
    F_red = constraints.E_T @ F_full

    u_red = spla.spsolve(A_red, F_red)
    u_recovered = constraints.prolong(u_red)

    print("\nRecovered full solution (including slaves):")
    for i, val in enumerate(u_recovered):
        print(f"  dof {i:02d}: {val:8.4f}")

    # Check the slave value matches the weighted combination of its masters
    for sdof, combo in constraints.slave_to_master.items():
        predicted = sum(w * u_recovered[mdof] for mdof, w in combo)
        print(
            f"Slave dof {sdof}: value {u_recovered[sdof]:.4f} "
            f"(predicted {predicted:.4f} from masters)"
        )


if __name__ == "__main__":
    main()
