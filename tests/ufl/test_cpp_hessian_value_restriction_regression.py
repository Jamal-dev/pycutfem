import os

import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Function, FacetNormal, Hessian, dot, inner, restrict, jump
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dGhost
from pycutfem.utils.domain_manager import get_domain_bitset
from pycutfem.utils.meshgen import structured_quad


def test_cpp_hessian_value_restriction_jump_dotn_regression(monkeypatch, tmp_path) -> None:
    """
    Regression test for the C++ backend:

    - `Hessian(Function)` is represented as an Eigen::MatrixXd (k,4) in the kernel.
    - `Restriction(Hessian(Function), domain)` must therefore zero the whole matrix
      (not iterate with `operator[]` as if it were a std::vector<MatrixXd>).
    - `Jump` on that restricted Hessian must subtract the matrices (not loop over
      `size()` and index with `operator[]`).

    This is exactly the pattern used by the Turek-cylinder Hessian ghost penalty:
      nᵀ jump(H(u_phys)) n  with u_phys = restrict(u, physical_domain).
    """
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")

    nodes, elems, edges, corners = structured_quad(2.0, 1.0, nx=8, ny=4, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )

    # A non-mesh-aligned interface so ghost edges are non-empty.
    ls = AffineLevelSet(a=1.0, b=0.2, c=-0.95)
    mesh.classify_elements(ls)
    mesh.classify_edges(ls)
    ghost = mesh.edge_bitset("ghost")
    assert ghost.cardinality() > 0

    me = MixedElement(mesh, field_specs={"u": 2})
    dh = DofHandler(me, method="cg")

    physical = get_domain_bitset(mesh, "element", "outside")
    uh = Function(name="uh", field_name="u", dof_handler=dh)
    uh.set_values_from_function(lambda x, y: np.sin(2.3 * x) + 0.1 * np.cos(3.1 * y))
    uh_phys = restrict(uh, physical)

    n = FacetNormal()
    dG = dGhost(
        defined_on=ghost,
        level_set=ls,
        metadata={"q": 6, "derivs": {(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)}},
    )

    d2n = dot(n, dot(jump(Hessian(uh_phys)), n))
    Eform = inner(d2n, d2n) * dG
    hooks = {Eform.integrand: {"name": "E"}}

    res_py = assemble_form(Equation(None, Eform), dof_handler=dh, bcs=[], assembler_hooks=hooks, backend="python")
    res_cpp = assemble_form(Equation(None, Eform), dof_handler=dh, bcs=[], assembler_hooks=hooks, backend="cpp")

    E_py = float(np.asarray(res_py["E"], dtype=float).reshape(-1)[0])
    E_cpp = float(np.asarray(res_cpp["E"], dtype=float).reshape(-1)[0])
    assert np.isfinite(E_py)
    assert np.isfinite(E_cpp)
    assert abs(E_py - E_cpp) < 1e-10
