import numpy as np
import pytest
import scipy.sparse.linalg as spla
import os

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem import transform
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.integration.quadrature import volume
from pycutfem.ufl import (
    Equation,
    HdivTestFunction,
    HdivTrialFunction,
    TestFunction as UFLTestFunction,
    TrialFunction as UFLTrialFunction,
    assemble_form,
    div,
    dx,
    inner,
)
from pycutfem.ufl.analytic import Analytic
from pycutfem.utils.meshgen import structured_quad


def _solve_mixed_darcy_rt1_dg1(nx: int, *, backend: str):
    nodes, elem_conn, edges, corner = structured_quad(1.0, 1.0, nx=nx, ny=nx, poly_order=1)
    mesh = Mesh(
        nodes,
        elem_conn,
        edges_connectivity=edges,
        elements_corner_nodes=corner,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, {"u": ("RT", 1), "p": ("DG", 1)})
    dh = DofHandler(me, method="cg")

    u = HdivTrialFunction("u")
    v = HdivTestFunction("u")
    p = UFLTrialFunction("p")
    q = UFLTestFunction("p")

    pi = float(np.pi)

    def p_exact(x: float, y: float) -> float:
        return float(np.sin(pi * x) * np.sin(pi * y))

    def u_exact(x: float, y: float) -> np.ndarray:
        return np.array(
            [
                -pi * np.cos(pi * x) * np.sin(pi * y),
                -pi * np.sin(pi * x) * np.cos(pi * y),
            ],
            dtype=float,
        )

    g = Analytic(lambda x, y: 2.0 * (pi**2) * np.sin(pi * x) * np.sin(pi * y), degree=6)

    a = (inner(u, v) - p * div(v) + div(u) * q) * dx()
    L = (g * q) * dx()
    K, F = assemble_form(Equation(a, L), dof_handler=dh, bcs=[], backend=backend)
    sol = np.asarray(spla.spsolve(K.tocsr(), np.asarray(F, dtype=float)), dtype=float)

    qp, qw = volume("quad", order=4)
    qp = np.asarray(qp, dtype=float)
    qw = np.asarray(qw, dtype=float)

    err_u = 0.0
    err_p = 0.0
    for eid in range(mesh.num_elements()):
        u_g = np.asarray(dh.element_maps["u"][int(eid)], dtype=int)
        sgn_u = np.asarray(dh.element_signs["u"][int(eid)], dtype=float)
        p_g = np.asarray(dh.element_maps["p"][int(eid)], dtype=int)
        u_loc = np.asarray(sol[u_g], dtype=float) * sgn_u
        p_loc = np.asarray(sol[p_g], dtype=float)

        for (xi, eta), w in zip(qp, qw):
            xi = float(xi)
            eta = float(eta)
            J = np.asarray(transform.jacobian(mesh, int(eid), (xi, eta)), dtype=float)
            detJ = float(np.linalg.det(J))
            w_phys = float(w) * abs(detJ)

            X = np.asarray(transform.x_mapping(mesh, int(eid), (xi, eta)), dtype=float)
            x = float(X[0])
            y = float(X[1])

            Vu = np.asarray(me.tabulate_value("u", xi, eta, element_id=int(eid)), dtype=float)
            uh = np.asarray(u_loc @ Vu, dtype=float)
            ue = u_exact(x, y)
            err_u += w_phys * float(np.dot(uh - ue, uh - ue))

            phip = np.asarray(me._ref["p"].shape(xi, eta), dtype=float).ravel()
            ph = float(p_loc @ phip)
            pe = p_exact(x, y)
            err_p += w_phys * float((ph - pe) ** 2)

    h = 1.0 / float(nx)
    return h, float(np.sqrt(err_u)), float(np.sqrt(err_p))


@pytest.mark.parametrize("backend", ("python", "jit", "cpp"))
def test_mixed_darcy_mms_converges_rt1_dg1(backend, monkeypatch, tmp_path):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_{backend}"))
    if backend == "jit":
        monkeypatch.delenv("PYCUTFEM_JIT_BACKEND", raising=False)

    nx_spec = str(os.environ.get("PYCUTFEM_HDIV_DARCY_NX_LIST", "")).strip()
    if nx_spec:
        mesh_ns = [int(x.strip()) for x in nx_spec.split(",") if x.strip()]
    else:
        mesh_ns = [2, 4]
    if len(mesh_ns) < 2:
        raise ValueError("PYCUTFEM_HDIV_DARCY_NX_LIST must contain at least two mesh sizes (e.g. '2,4').")
    hs: list[float] = []
    eu: list[float] = []
    ep: list[float] = []
    for nx in mesh_ns:
        h, err_u, err_p = _solve_mixed_darcy_rt1_dg1(int(nx), backend=backend)
        hs.append(h)
        eu.append(err_u)
        ep.append(err_p)

    hs = np.asarray(hs, dtype=float)
    eu = np.asarray(eu, dtype=float)
    ep = np.asarray(ep, dtype=float)

    rate_u = np.diff(np.log(eu)) / np.diff(np.log(hs))
    rate_p = np.diff(np.log(ep)) / np.diff(np.log(hs))

    assert np.all(np.isfinite(rate_u))
    assert np.all(np.isfinite(rate_p))
    assert np.all(rate_u > 1.7)
    assert np.all(rate_u < 2.3)
    assert np.all(rate_p > 1.7)
    assert np.all(rate_p < 2.3)
