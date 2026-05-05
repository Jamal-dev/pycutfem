import numpy as np
import pytest
import scipy.sparse.linalg as spla

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.xfem import XFEMDofHandler

from pycutfem.ufl.measures import dx
from pycutfem.ufl.expressions import Constant, TrialFunction, TestFunction, Function
from pycutfem.ufl.forms import Equation, assemble_form


@pytest.mark.parametrize("backend", ["python", "jit"])
def test_discontinuous_patch_test_xfem(backend):
    """
    XFEM discontinuous patch test:
      u = 1 in Ω⁺, u = 0 in Ω⁻
    L2 projection should be (near) exact for P1 with a straight interface.
    """
    # Use an odd element count so the interface x=0.5 cuts cells (not aligned on nodes/edges),
    # ensuring enriched DOFs are activated.
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=7, ny=7, poly_order=1)
    mesh = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=1)

    me = MixedElement(mesh, {"u": 1})
    dh0 = DofHandler(me, method="cg")

    ls = AffineLevelSet(1.0, 0.0, -0.5)  # x - 0.5
    dh0.classify_from_levelset(ls)

    # Baseline: continuous space cannot represent the jump exactly.
    u0 = TrialFunction("u", dof_handler=dh0)
    v0 = TestFunction("u", dof_handler=dh0)
    dx_pos = dx(level_set=ls, metadata={"side": "+", "q": 4})
    dx_neg = dx(level_set=ls, metadata={"side": "-", "q": 4})
    a0 = (u0 * v0) * dx_pos + (u0 * v0) * dx_neg
    L0 = Constant(1.0) * v0 * dx_pos
    K0, F0 = assemble_form(Equation(a0, L0), dof_handler=dh0, backend=backend)
    sol0 = spla.spsolve(K0, F0)

    uh0 = Function(name="uh0", field_name="u", dof_handler=dh0)
    gd0 = dh0.get_field_slice("u")
    uh0.set_nodal_values(gd0, sol0[gd0])

    err_pos0 = (uh0 - Constant(1.0)) * (uh0 - Constant(1.0))
    err_neg0 = uh0 * uh0
    err_form0 = err_pos0 * dx_pos + err_neg0 * dx_neg
    res0 = assemble_form(
        Equation(err_form0, None),
        dof_handler=dh0,
        assembler_hooks={err_pos0: {"name": "err2"}, err_neg0: {"name": "err2"}},
        backend=backend,
    )
    err2_0 = float(np.asarray(res0["err2"]).ravel()[0])
    assert err2_0 > 1.0e-4

    dh = XFEMDofHandler(dh0)
    dh.rebuild_enrichment(ls, enrich={"u": "heaviside"})

    u = TrialFunction("u", dof_handler=dh)
    v = TestFunction("u", dof_handler=dh)

    a = (u * v) * dx_pos + (u * v) * dx_neg
    L = Constant(1.0) * v * dx_pos

    K, F = assemble_form(Equation(a, L), dof_handler=dh, backend=backend)
    sol = spla.spsolve(K, F)

    uh = Function(name="uh", field_name="u", dof_handler=dh)
    gd = dh.get_field_slice("u")
    uh.set_nodal_values(gd, sol[gd])

    err_pos = (uh - Constant(1.0)) * (uh - Constant(1.0))
    err_neg = uh * uh
    err_form = err_pos * dx_pos + err_neg * dx_neg

    res = assemble_form(
        Equation(err_form, None),
        dof_handler=dh,
        assembler_hooks={err_pos: {"name": "err2"}, err_neg: {"name": "err2"}},
        backend=backend,
    )
    err2 = float(np.asarray(res["err2"]).ravel()[0])
    assert err2 < 1.0e-10
