import math

import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet, MinLevelSet, RotatedBoxLevelSet, ScaledLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Function
from pycutfem.utils.meshgen import structured_quad


def _have_cpp_backend() -> bool:
    try:
        from pycutfem.jit.cpp_backend import compile_backend_cpp  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.parametrize("backend", ["python", "jit"] + (["cpp"] if _have_cpp_backend() else []))
def test_error_norms_restrict_to_multilevelset_domain(backend):
    """
    Regression: error norms must not integrate over inactive regions.

    We build a composite (multi-level-set) fluid domain as:
        Ω^F = { poro_ls > 0 } ∩ { x >= x0 }
    via φ_F = min(poro_ls, cut_pos).

    We then construct a Q1 field u_h that matches a linear exact solution on Ω^F,
    but is intentionally wrong in the inactive region (inside elements only).
    The CutFEM-restricted error must be ~0 while the full-mesh error is > 0.
    """
    # Background mesh Ω = [-0.5, 0.5]^2
    L = 1.0
    nodes, elems, _, corners = structured_quad(L, L, nx=8, ny=8, poly_order=1, offset=[-0.5, -0.5])
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=1)

    me = MixedElement(mesh, field_specs={"u_pos": 1})
    dh = DofHandler(me, method="cg")

    poro_ls = RotatedBoxLevelSet(center=(0.0, 0.0), hx=0.25, hy=0.25, angle=math.pi / 6.0)
    x0 = -0.1
    cut_ls = AffineLevelSet(-1.0, 0.0, float(x0))  # negative on {x>=x0}
    cut_pos = ScaledLevelSet(-1.0, cut_ls)  # positive on {x>=x0}
    fluid_ls = MinLevelSet(poro_ls, cut_pos)  # positive in Ω^F

    mesh.classify_elements(fluid_ls)

    # Tag inside-only DOFs as inactive (these belong exclusively to the removed/poro region).
    dh.dof_tags["inactive"] = set()
    dh.tag_dofs_from_element_bitset("inactive", "u_pos", "inside", strict=True)
    inactive = np.asarray(sorted(dh.dof_tags["inactive"]), dtype=int)
    assert inactive.size > 0

    # Exact Q1 field: u = x + 2y (representable exactly on Q1).
    def u_exact(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return x + 2.0 * y

    sl = np.asarray(dh.get_field_slice("u_pos"), dtype=int)
    coords = np.asarray(dh.get_dof_coords("u_pos"), dtype=float)

    u_vec = np.zeros(int(dh.total_dofs), dtype=float)
    u_vec[sl] = coords[:, 0] + 2.0 * coords[:, 1]

    # Corrupt only inactive DOFs: does not affect Ω^F elements by construction.
    u_vec[inactive] = 0.0

    # Full-mesh L2 error (includes inactive region) should be non-zero.
    err_full = float(dh.l2_error(u_vec, exact={"u_pos": u_exact}, quad_order=6, relative=False))
    assert err_full > 1.0e-6

    # CutFEM-restricted error should be ~0 on Ω^F.
    err_fluid = float(
        dh.l2_error(
            u_vec,
            exact={"u_pos": u_exact},
            quad_order=6,
            relative=False,
            level_set=fluid_ls,
            side="+",
            backend=backend,
        )
    )
    assert err_fluid < 1.0e-10

    # Same for H1-seminorm error (grad error) on Ω^F.
    uh = Function(name="uh", field_name="u_pos", dof_handler=dh)
    uh.set_nodal_values(sl, u_vec[sl])

    def grad_exact(x, y):
        _ = (x, y)
        return np.array([1.0, 2.0], dtype=float)

    if backend in {"jit", "cpp"}:
        err_h1 = dh.h1_error_scalar_on_side_compiled(
            uh,
            grad_exact,
            fluid_ls,
            side="+",
            field="u_pos",
            relative=False,
            quad_order=6,
            backend=backend,
        )
    else:
        err_h1 = dh.h1_error_scalar_on_side(
            uh,
            grad_exact,
            fluid_ls,
            side="+",
            field="u_pos",
            relative=False,
            quad_increase=0,
        )
    assert float(err_h1) < 1.0e-10
