import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl import HdivFunction, HdivTestFunction, HdivTrialFunction
from pycutfem.ufl import Function, TestFunction as UFLTestFunction, TrialFunction as UFLTrialFunction
from pycutfem.ufl import div, dx, inner
from pycutfem.ufl.analytic import Analytic
from pycutfem.utils.meshgen import structured_quad
from pycutfem.utils.refinement import TensorRefiner


def _make_single_interface_hanging_mesh() -> Mesh:
    nodes, elems, _edges, corners = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=1)
    mesh0 = Mesh(
        nodes,
        element_connectivity=elems,
        edges_connectivity=None,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    refiner = TensorRefiner(max_ref=3)
    rx = np.zeros((len(mesh0.elements_list),), dtype=int)
    ry = np.zeros_like(rx)
    ry[0] = 1
    return refiner.refine(mesh0, rx, ry)


@pytest.mark.parametrize("backend", ("python", "jit", "cpp"))
def test_newton_mixed_darcy_hdiv_runs_on_hanging_mesh(backend, monkeypatch, tmp_path):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_{backend}"))
    if backend == "jit":
        monkeypatch.delenv("PYCUTFEM_JIT_BACKEND", raising=False)
    elif backend == "cpp":
        monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")
    else:
        monkeypatch.delenv("PYCUTFEM_JIT_BACKEND", raising=False)

    mesh = _make_single_interface_hanging_mesh()
    me = MixedElement(mesh, {"u": ("RT", 0), "p": ("DG", 0)})
    dh = DofHandler(me, method="cg")

    u_k = HdivFunction(name="u_k", field_name="u", dof_handler=dh)
    p_k = Function(name="p_k", field_name="p", dof_handler=dh)
    u_n = HdivFunction(name="u_n", field_name="u", dof_handler=dh)
    p_n = Function(name="p_n", field_name="p", dof_handler=dh)

    du = HdivTrialFunction("u")
    dp = UFLTrialFunction("p", dof_handler=dh)
    v = HdivTestFunction("u")
    q = UFLTestFunction("p", dof_handler=dh)

    pi = float(np.pi)
    g = Analytic(lambda x, y: 2.0 * (pi**2) * np.sin(pi * x) * np.sin(pi * y), degree=6)

    qdeg = 6
    residual_form = (inner(u_k, v) - p_k * div(v) + div(u_k) * q - g * q) * dx(metadata={"q": qdeg})
    jacobian_form = (inner(du, v) - dp * div(v) + div(du) * q) * dx(metadata={"q": qdeg})

    solver = NewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(newton_tol=1.0e-10, max_newton_iter=10, line_search=False),
        quad_order=qdeg,
        backend=backend,
    )
    assert solver.constraints is not None, "expected hanging-node constraints to be detected"
    u_gids = set(map(int, dh.get_field_slice("u")))
    assert any(int(sd) in u_gids for sd in solver.constraints.slaves), "expected RT slave DOFs"

    u_k.nodal_values[:] = 0.0
    p_k.nodal_values[:] = 0.0
    u_n.nodal_values[:] = 0.0
    p_n.nodal_values[:] = 0.0

    solver.solve_time_interval(
        functions=[u_k, p_k],
        prev_functions=[u_n, p_n],
        time_params=TimeStepperParameters(dt=1.0, max_steps=1, stop_on_steady=True),
    )

    _, R = solver._assemble_system({"u_k": u_k, "p_k": p_k, "u_n": u_n, "p_n": p_n}, need_matrix=False)
    # With hanging-node constraints we solve in the *master* space. The full-space
    # residual can have nonzero entries on slave DOFs; the meaningful check is
    # the condensed/master residual Eᵀ R_full.
    if solver.constraints is not None:
        R_master = solver.constraints.restrict_full(np.asarray(R, dtype=float))
        assert float(np.linalg.norm(np.asarray(R_master, dtype=float), ord=np.inf)) < 1.0e-8
    else:
        assert float(np.linalg.norm(np.asarray(R, dtype=float), ord=np.inf)) < 1.0e-8

    # Postcondition: stored full-space vectors satisfy constraint relations.
    U_full = np.zeros(dh.total_dofs, dtype=float)
    U_full[dh.get_field_slice("u")] = u_k.nodal_values
    U_full[dh.get_field_slice("p")] = p_k.nodal_values
    for sdof, combo in solver.constraints.slave_to_master.items():
        approx = 0.0
        for mdof, w in combo:
            approx += float(w) * float(U_full[int(mdof)])
        assert abs(float(U_full[int(sdof)]) - approx) < 2.0e-9
