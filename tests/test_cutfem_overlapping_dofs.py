import numpy as np
import pytest
import scipy.sparse.linalg as spla
import sympy as sp

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import LevelSetFunction, CircleLevelSet

from pycutfem.ufl.expressions import TrialFunction, TestFunction, grad, inner, dot, FacetNormal, CellDiameter, Constant, Jump
from pycutfem.ufl.measures import dx, dInterface, dGhost
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.analytic import x as x_ana
from pycutfem.ufl.analytic import y as y_ana


class BarrierLevelSet(LevelSetFunction):
    """
    Wall band around x=0.5:
      φ>0 in the wall (inactive), φ<0 in the fluid (active for side='-').
    """

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        dist = np.abs(x[..., 0] - 0.5)
        return 0.05 - dist

    def gradient(self, x):
        x = np.asarray(x, dtype=float)
        g = np.zeros_like(x, dtype=float)
        g[..., 0] = np.where(x[..., 0] > 0.5, -1.0, 1.0)
        g[..., 0] = np.where(np.isclose(x[..., 0], 0.5), 0.0, g[..., 0])
        return g


def _have_cpp_backend() -> bool:
    try:
        from pycutfem.jit.cpp_backend import compile_backend_cpp  # noqa: F401
        return True
    except Exception:
        return False


def test_cutfem_split_node_barrier_q1():
    # 2x2 Q1 -> 9 background DOFs
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=1)
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=1)

    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")
    ndofs_bg = dh.total_dofs

    ls = BarrierLevelSet()
    dh.distribute_dofs_cutfem(ls, domain_side="-")

    assert dh.total_dofs > ndofs_bg, "Expected DOF count to increase after splitting."

    # Elements 0 (bottom-left) and 1 (bottom-right) are separated by a wall band on x=0.5.
    el0 = set(int(i) for i in dh.element_maps["u"][0] if int(i) >= 0)
    el1 = set(int(i) for i in dh.element_maps["u"][1] if int(i) >= 0)
    shared = el0.intersection(el1)
    assert len(shared) == 0, f"Elements across the barrier share DOFs: {shared}"


@pytest.mark.parametrize("backend", ["python", "jit"] + (["cpp"] if _have_cpp_backend() else []))
def test_cutfem_poisson_circle_mms_convergence_with_split(backend, monkeypatch):
    # Keep JIT stable/deterministic: allow user to override JIT choice via env,
    # but default to cpp when available.
    if backend == "jit":
        monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")

    # MMS: u = sin(pi x) sin(pi y), -Δu = 2*pi^2*sin(pi x) sin(pi y)
    u_sym = sp.sin(sp.pi * x_ana) * sp.sin(sp.pi * y_ana)
    f_sym = 2 * (sp.pi**2) * sp.sin(sp.pi * x_ana) * sp.sin(sp.pi * y_ana)
    u_exact = Analytic(u_sym)
    f_exact = Analytic(f_sym)
    u_exact_xy = sp.lambdify((x_ana, y_ana), u_sym, "numpy")

    ls = CircleLevelSet(center=(0.5, 0.5), radius=0.30)

    hs = []
    errs = []

    for n in (12, 24, 48):
        nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=n, ny=n, poly_order=1)
        mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=1)

        mesh.classify_elements(ls)
        mesh.classify_edges(ls)
        mesh.build_interface_segments(level_set=ls)

        phys = mesh.element_bitset("inside") | mesh.element_bitset("cut")
        cut = mesh.element_bitset("cut")
        ghost = mesh.edge_bitset("ghost_neg") | mesh.edge_bitset("ghost_both")

        me = MixedElement(mesh, field_specs={"u": 1})
        dh = DofHandler(me, method="cg")
        dh.distribute_dofs_cutfem(ls, domain_side="-")

        # Ensure interface/ghost sets exist after re-tagging from distribute_dofs_cutfem
        mesh.classify_edges(ls)
        mesh.build_interface_segments(level_set=ls)

        u = TrialFunction("u", dh)
        v = TestFunction("u", dh)

        qvol = 8
        qedge = 10
        dx_phys = dx(defined_on=phys, level_set=ls, metadata={"side": "-", "q": qvol})
        dG = dInterface(defined_on=cut, level_set=ls, metadata={"q": qedge})
        dGhost_stab = dGhost(
            defined_on=ghost,
            level_set=ls,
            metadata={"q": qedge, "derivs": {(1, 0), (0, 1)}},
        )

        nrm = FacetNormal()
        h = CellDiameter()
        gamma_N = Constant(40.0) / h
        gamma_G = Constant(0.05)

        a = (
            inner(grad(u), grad(v)) * dx_phys
            - dot(grad(u), nrm) * v * dG
            - dot(grad(v), nrm) * u * dG
            + gamma_N * u * v * dG
        )
        a += gamma_G * h * inner(dot(nrm, Jump(grad(u))), dot(Jump(grad(v)), nrm)) * dGhost_stab

        L = (
            f_exact * v * dx_phys
            - u_exact * dot(grad(v), nrm) * dG
            + u_exact * gamma_N * v * dG
        )

        K, F = assemble_form(Equation(a, L), dof_handler=dh, bcs=[], backend=backend)
        uh = spla.spsolve(K.tocsr(), F)

        from pycutfem.ufl.expressions import Function

        u_h = Function("u_h", "u", dof_handler=dh)
        u_h.set_nodal_values(np.arange(dh.total_dofs, dtype=int), uh)

        err = dh.l2_error_on_side(function=u_h, exact={"u": u_exact_xy}, level_set=ls, side="-", quad_order=10)
        errs.append(float(err))
        hs.append(1.0 / float(n))

    rates = np.diff(np.log(np.asarray(errs))) / np.diff(np.log(np.asarray(hs)))
    assert np.all(np.isfinite(rates))
    # Q1 should be close to 2 in L2 on smooth solutions; allow slack for cut geometry.
    assert np.min(rates) > 1.4, f"Suboptimal L2 rates: {rates}"
