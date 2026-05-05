import time
import numpy as np
from numpy.testing import assert_allclose

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.expressions import TrialFunction, TestFunction, Grad, Inner
from pycutfem.ufl.measures import dx
from pycutfem.ufl.forms import Equation, assemble_form


def _assemble_poisson(n_fields: int):
    """Assemble a tiny Poisson stiffness matrix with a configurable number of fields."""
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )

    # active field 'u' plus optional inactive extras
    specs = {"u": 1}
    for i in range(1, n_fields):
        specs[f"junk{i}"] = 1

    me = MixedElement(mesh, field_specs=specs)
    dh = DofHandler(me, method="cg")

    u = TrialFunction("u", "u", dh)
    v = TestFunction("u", "u", dh)
    a = Inner(Grad(u), Grad(v)) * dx(metadata={"q": 2})

    t0 = time.perf_counter()
    K, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend="jit")
    elapsed = time.perf_counter() - t0

    return K, dh, me, elapsed


def test_jit_warmup_skips_inactive_fields():
    """
    Warm-up time with many inactive fields should stay close to the single-field case.
    The assembled block for the active field must match the single-field stiffness,
    and inactive rows/cols should remain near zero.
    """
    K1, dh1, me1, t1 = _assemble_poisson(n_fields=1)
    K5, dh5, me5, t5 = _assemble_poisson(n_fields=5)

    # Confirm total DOFs grow with extra fields
    assert dh5.total_dofs > dh1.total_dofs

    # Active block (field 'u') matches single-field assembly
    u_gdofs_1 = np.asarray(dh1.get_field_slice("u"), dtype=int)
    u_gdofs_5 = np.asarray(dh5.get_field_slice("u"), dtype=int)
    assert_allclose(K5.toarray()[np.ix_(u_gdofs_5, u_gdofs_5)], K1.toarray()[np.ix_(u_gdofs_1, u_gdofs_1)], rtol=1e-10, atol=1e-12)

    # Inactive field rows/cols remain zero (within numerical noise)
    inactive = np.setdiff1d(np.arange(K5.shape[0]), u_gdofs_5)
    if inactive.size:
        sub = K5.toarray()[np.ix_(inactive, inactive)]
        assert np.allclose(sub, 0.0, atol=1e-12)

    # Warm-up time should not explode when unused fields are present.
    # Allow generous factor for jitter and cache misses.
    assert t5 <= 2.5 * t1 + 0.05, f"Inactive fields should not inflate warm-up (t1={t1:.3f}s, t5={t5:.3f}s)"
