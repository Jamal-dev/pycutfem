import numpy as np
import scipy.sparse.linalg as spla

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.xfem import XFEMDofHandler, l2_project_moving_interface

from pycutfem.ufl.measures import dx
from pycutfem.ufl.expressions import Constant, TrialFunction, TestFunction, Function
from pycutfem.ufl.forms import Equation, assemble_form


def test_moving_interface_l2_projection_conserves_mass():
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=7, ny=7, poly_order=1)
    mesh = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=1)
    me = MixedElement(mesh, {"u": 1})
    base = DofHandler(me, method="cg")

    ls_old = AffineLevelSet(1.0, 0.0, -0.50)  # x - 0.50
    base.classify_from_levelset(ls_old)
    dh_old = XFEMDofHandler(base)
    dh_old.rebuild_enrichment(ls_old, enrich={"u": "heaviside"})

    # Build an exact-ish jump solution (same as the XFEM patch test).
    u = TrialFunction("u", dof_handler=dh_old)
    v = TestFunction("u", dof_handler=dh_old)
    dxp_old = dx(level_set=ls_old, metadata={"side": "+", "q": 4})
    dxm_old = dx(level_set=ls_old, metadata={"side": "-", "q": 4})
    a = (u * v) * dxp_old + (u * v) * dxm_old
    L = Constant(1.0) * v * dxp_old
    M_old, b_old = assemble_form(Equation(a, L), dof_handler=dh_old, backend="python")
    sol_old = spla.spsolve(M_old, b_old)

    uh_old = Function(name="uh_old", field_name="u", dof_handler=dh_old)
    uh_old.set_nodal_values(dh_old.get_field_slice("u"), sol_old[dh_old.get_field_slice("u")])

    res_old = assemble_form(
        Equation(uh_old * dxp_old + uh_old * dxm_old, None),
        dof_handler=dh_old,
        assembler_hooks={uh_old: {"name": "mass"}},
        backend="python",
    )
    mass_old = float(np.asarray(res_old["mass"]).ravel()[0])

    # Move the interface and rebuild enrichment.
    ls_new = AffineLevelSet(1.0, 0.0, -0.55)  # x - 0.55
    base.classify_from_levelset(ls_new)
    dh_new = XFEMDofHandler(base)
    dh_new.rebuild_enrichment(ls_new, enrich={"u": "heaviside"})

    sol_new = l2_project_moving_interface(
        dh_old=dh_old,
        u_old=np.asarray(sol_old, dtype=float),
        level_set_old=ls_old,
        dh_new=dh_new,
        level_set_new=ls_new,
        field="u",
        q=4,
        backend="python",
    )

    uh_new = Function(name="uh_new", field_name="u", dof_handler=dh_new)
    uh_new.set_nodal_values(dh_new.get_field_slice("u"), sol_new[dh_new.get_field_slice("u")])

    dxp_new = dx(level_set=ls_new, metadata={"side": "+", "q": 4})
    dxm_new = dx(level_set=ls_new, metadata={"side": "-", "q": 4})
    res_new = assemble_form(
        Equation(uh_new * dxp_new + uh_new * dxm_new, None),
        dof_handler=dh_new,
        assembler_hooks={uh_new: {"name": "mass"}},
        backend="python",
    )
    mass_new = float(np.asarray(res_new["mass"]).ravel()[0])

    assert abs(mass_new - mass_old) < 1.0e-10

