import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Jump, Hessian, inner, restrict, VectorTestFunction, VectorTrialFunction
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dFacetPatch
from pycutfem.utils.meshgen import structured_quad


def test_facet_patch_hessian_trial_restriction_python_cpp_backend_parity(monkeypatch, tmp_path) -> None:
    """
    Regression test: `Hessian(restrict(trial/test, domain))` must honor restriction masks
    on facet-patch (dFacetPatch) assembly.

    This used to be broken in the Python backend: restriction masks were only applied
    to collapsed Function Hessians, but not to trial/test Hessian tables.
    """
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))

    nodes, elems, edges, corners = structured_quad(2.0, 1.0, nx=10, ny=4, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )

    me = MixedElement(mesh, {"ux": 2, "uy": 2})
    dh = DofHandler(me, method="cg")

    # A non-aligned cut so we have cut + inside elements and a non-empty ghost set.
    ls = AffineLevelSet(a=1.0, b=0.2, c=-0.93)
    dh.classify_from_levelset(ls)

    ghost = mesh.edge_bitset("ghost")
    assert ghost.cardinality() > 0

    physical = mesh.element_bitset("outside") | mesh.element_bitset("cut")
    inside = mesh.element_bitset("inside")
    assert physical.cardinality() > 0
    assert inside.cardinality() > 0

    V = FunctionSpace(name="V", field_names=["ux", "uy"], dim=1, side="+")
    u_pos = VectorTrialFunction(space=V, dof_handler=dh, side="+")
    u_neg = VectorTrialFunction(space=V, dof_handler=dh, side="-")
    v_pos = VectorTestFunction(space=V, dof_handler=dh, side="+")
    v_neg = VectorTestFunction(space=V, dof_handler=dh, side="-")

    u_jump_phys = Jump(restrict(u_pos, physical), restrict(u_neg, physical))
    v_jump_phys = Jump(restrict(v_pos, physical), restrict(v_neg, physical))

    dW = dFacetPatch(
        defined_on=ghost,
        level_set=ls,
        metadata={"q": 6, "derivs": {(2, 0), (1, 1), (0, 2)}},
    )
    a = inner(Hessian(u_jump_phys), Hessian(v_jump_phys)) * dW

    K_py, _ = assemble_form(Equation(a, None), dof_handler=dh, backend="python")
    K_cpp, _ = assemble_form(Equation(a, None), dof_handler=dh, backend="cpp")

    K_py = K_py.tocsr()
    K_cpp = K_cpp.tocsr()

    assert K_py.nnz > 0
    assert K_cpp.nnz > 0

    # Restriction must have a visible structural effect (otherwise the test isn't meaningful).
    a_un = inner(Hessian(Jump(u_pos, u_neg)), Hessian(Jump(v_pos, v_neg))) * dW
    K_cpp_un, _ = assemble_form(Equation(a_un, None), dof_handler=dh, backend="cpp")
    K_cpp_un = K_cpp_un.tocsr()
    assert K_cpp.nnz < K_cpp_un.nnz

    diff = (K_py - K_cpp).tocoo()
    max_diff = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
    assert max_diff < 1.0e-9

