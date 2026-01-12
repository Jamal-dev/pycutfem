import numpy as np

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.xfem import AgFEMMapper


def test_agfem_q2_polynomial_reproduction_on_sliver():
    """
    High-order AgFEM sanity check (Q2 on Q1 geometry):

    Build a near-zero cut fraction ('+' sliver) and enforce aggregation
    constraints. The constraint operator should reproduce a quadratic
    polynomial at the slave DOF coordinates (polynomial extension).
    """
    # Q1 geometry mesh; Q2 scalar field
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=4, ny=1, poly_order=1)
    mesh = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=1)
    me = MixedElement(mesh, {"u": 2})
    dh = DofHandler(me, method="cg")

    # Vertical interface extremely close to x=0.25 so that the first column is a tiny '+' sliver
    ls = AffineLevelSet(1.0, 0.0, -0.249999)  # x - 0.249999
    dh.classify_from_levelset(ls)

    mapper = AgFEMMapper(dh)
    ag = mapper.build_aggregation_map(ls, side="+", theta_min=0.05)
    cons = mapper.build_constraints(ag, fields=["u"])

    # Quadratic polynomial (exactly representable by Q2 on an affine map)
    coords = dh.get_all_dof_coords()
    master_xy = coords[cons.master_ids]
    u_master = (master_xy[:, 0] ** 2 + 0.5 * master_xy[:, 1] + 0.25).astype(float)

    u_full = np.asarray(cons.prolong(u_master), dtype=float).ravel()

    # Check reproduction on the constrained (slave) DOFs only.
    for sd in cons.slaves.tolist():
        x, y = coords[int(sd)]
        u_ex = float(x * x + 0.5 * y + 0.25)
        assert abs(float(u_full[int(sd)]) - u_ex) < 1.0e-10

