#!/usr/bin/env python
"""
Compare sided interface traces/jumps between PyCutFEM and NGSolve/XFEM.

Why:
  The existing comparison harness mostly uses symmetric constant fields, so
  many interface jump terms evaluate to ~0. This script uses *different* (+)/(-)
  constants so jumps are non-zero and sign/side mistakes show up immediately.

What is checked (on Γ: φ(x,y)=x = 0 in [-1,1]×[-0.5,0.5]):
  - mean_Γ(u^+), mean_Γ(u^-), mean_Γ(jump(u)) with jump = u^+ - u^-
  - mean_Γ(n) from the level-set normal (orientation check)
  - mean_Γ((-p^+) n_x) to validate the pressure traction sign

We compare *means* on Γ:
  mean_Γ(f) := (∫_Γ f ds) / (∫_Γ 1 ds)
This removes mesh-dependent geometry differences (each library has its own mesh).

Run (requires ngsolve-dev env):
  conda run --no-capture-output -n ngsolve-dev \
    python examples/debug/compare_sided_interface_integrals_to_ngsolve.py

Optional:
  BACKEND=python|jit|cpp   (PyCutFEM assembler backend; default: python)
"""

from __future__ import annotations

import argparse
import os

import numpy as np

# ------------------------- PyCutFEM -------------------------
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh as PCMesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, FacetNormal, Function, Neg, Pos, TestFunction, VectorFunction, jump
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dInterface
from pycutfem.utils.meshgen import structured_quad

# ------------------------- NGSolve/XFEM -------------------------
from netgen.geom2d import SplineGeometry
from ngsolve import *  # noqa: F403
from xfem import *  # noqa: F403


def integrate_pc_scalar(dh: DofHandler, integrand, measure, *, backend: str) -> float:
    """Integrate a scalar UFL expression by assembling into the NumberSpace 'lm'."""
    w = TestFunction(field_name="lm", dof_handler=dh)
    _, F = assemble_form(Equation(None, integrand * w * measure), dof_handler=dh, bcs=[], backend=backend)
    lm_inds = dh.get_field_slice("lm")
    return float(F[lm_inds][0])


def integrate_ng_scalar(mesh: Mesh, integrand, measure) -> float:
    """Integrate a scalar CoefficientFunction by assembling a NumberSpace linear form."""
    ns = NumberSpace(mesh)
    w = ns.TestFunction()
    lf = LinearForm(ns)
    lf += integrand * w * measure
    with TaskManager():
        lf.Assemble()
    return float(lf.vec.FV()[0])


def _mean(label: str, num: float, den: float) -> tuple[str, float]:
    if abs(den) < 1.0e-30:
        raise ZeroDivisionError(f"Zero interface measure in '{label}' (den={den}).")
    return label, float(num / den)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=8, help="PyCutFEM structured mesh cells in x")
    parser.add_argument("--ny", type=int, default=4, help="PyCutFEM structured mesh cells in y")
    parser.add_argument("--maxh", type=float, default=0.25, help="NGSolve maxh (independent mesh)")
    parser.add_argument("--order", type=int, default=2, help="Polynomial order (velocity)")
    parser.add_argument("--q", type=int, default=8, help="Interface quadrature order")
    parser.add_argument(
        "--probe",
        type=str,
        default="both",
        choices=("const", "poly", "both"),
        help="Which probe to run (const: piecewise constants; poly: y^2 profile; both: run both).",
    )
    args = parser.parse_args()

    backend = os.getenv("BACKEND", "python").lower()
    if backend not in {"python", "jit", "cpp", "c++"}:
        raise ValueError(f"Unsupported BACKEND='{backend}'. Use python/jit/cpp.")

    # ------------------------------------------------------------
    # Geometry / interface: φ(x,y)=x; Γ is the vertical line x=0.
    # ------------------------------------------------------------
    x0, x1 = -1.0, 1.0
    y0, y1 = -0.5, 0.5
    Lx, Ly = x1 - x0, y1 - y0

    # ---------------- PyCutFEM setup ----------------
    nodes, elems, edges, corners = structured_quad(
        Lx,
        Ly,
        nx=int(args.nx),
        ny=int(args.ny),
        poly_order=int(args.order),
        offset=(x0, y0),
    )
    pc_mesh = PCMesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=int(args.order),
    )
    pc_level_set = AffineLevelSet(a=1.0, b=0.0, c=0.0)  # x
    pc_mesh.classify_elements(pc_level_set)
    pc_mesh.classify_edges(pc_level_set)
    pc_mesh.build_interface_segments(pc_level_set)
    cut_elems = pc_mesh.element_bitset("cut")
    dGamma_pc = dInterface(defined_on=cut_elems, level_set=pc_level_set, metadata={"q": int(args.q)})

    me = MixedElement(
        pc_mesh,
        field_specs={
            "u_pos_x": int(args.order),
            "u_pos_y": int(args.order),
            "u_neg_x": int(args.order),
            "u_neg_y": int(args.order),
            "p_pos_": max(1, int(args.order) - 1),
            "p_neg_": max(1, int(args.order) - 1),
            "lm": ":number:",
        },
    )
    dh = DofHandler(me, method="cg")

    u_pos = VectorFunction("u_pos", ["u_pos_x", "u_pos_y"], dof_handler=dh, side="+")
    u_neg = VectorFunction("u_neg", ["u_neg_x", "u_neg_y"], dof_handler=dh, side="-")
    p_pos = Function("p_pos", "p_pos_", dof_handler=dh, side="+")
    p_neg = Function("p_neg", "p_neg_", dof_handler=dh, side="-")

    def _set_field_values(field: str, values: np.ndarray, target) -> None:
        gdofs = np.asarray(dh.get_field_slice(field), dtype=int)
        if values.shape != gdofs.shape:
            raise ValueError(f"Value shape mismatch for field '{field}': {values.shape} vs {gdofs.shape}")
        target.set_nodal_values(gdofs, np.asarray(values, dtype=float))

    def _set_const(field: str, value: float, target) -> None:
        gdofs = np.asarray(dh.get_field_slice(field), dtype=int)
        _set_field_values(field, np.full(gdofs.shape, float(value), dtype=float), target)

    def _set_y2(field: str, a0: float, a2: float, target) -> None:
        coords = np.asarray(dh.get_dof_coords(field), dtype=float)
        vals = float(a0) + float(a2) * coords[:, 1] ** 2
        _set_field_values(field, vals, target)

    # ---------------- NGSolve setup ----------------
    square = SplineGeometry()
    square.AddRectangle((x0, y0), (x1, y1), bcs=[1, 2, 3, 4])
    ng_mesh = Mesh(square.GenerateMesh(maxh=float(args.maxh), quad_dominated=True))

    ng_levelset = x  # φ(x,y)=x
    lsetp1 = GridFunction(H1(ng_mesh, order=1))
    InterpolateToP1(ng_levelset, lsetp1)
    ci = CutInfo(ng_mesh, lsetp1)

    Vhbase = VectorH1(ng_mesh, order=int(args.order))  # noqa: F405
    Qhbase = H1(ng_mesh, order=max(1, int(args.order) - 1))  # noqa: F405
    Vhneg = Compress(Vhbase, GetDofsOfElements(Vhbase, ci.GetElementsOfType(HASNEG)))  # noqa: F405
    Vhpos = Compress(Vhbase, GetDofsOfElements(Vhbase, ci.GetElementsOfType(HASPOS)))  # noqa: F405
    Qhneg = Compress(Qhbase, GetDofsOfElements(Qhbase, ci.GetElementsOfType(HASNEG)))  # noqa: F405
    Qhpos = Compress(Qhbase, GetDofsOfElements(Qhbase, ci.GetElementsOfType(HASPOS)))  # noqa: F405

    Wh = FESpace([Vhneg * Vhpos, Qhneg * Qhpos, NumberSpace(ng_mesh)], dgjumps=True)  # noqa: F405
    gfu = GridFunction(Wh)  # noqa: F405

    gfu.vec[:] = 0.0
    with TaskManager():
        gfu.components[2].Set(CoefficientFunction(1.0))  # noqa: F405

    dGamma_ng = dCut(lsetp1, IF, definedonelements=ci.GetElementsOfType(IF), order=int(args.q))  # noqa: F405

    # level-set normal (points to + side)
    grad_lset = Grad(lsetp1)  # noqa: F405
    n_ng = grad_lset / sqrt(InnerProduct(grad_lset, grad_lset))  # noqa: F405

    def _run_probe(title: str, *, ng_set_cb, expected: dict[str, float]) -> None:
        ONE_pc = Constant(1.0)
        ONE_ng = CoefficientFunction(1.0)  # noqa: F405

        L_pc = integrate_pc_scalar(dh, ONE_pc, dGamma_pc, backend=backend)
        L_ng = integrate_ng_scalar(ng_mesh, ONE_ng, dGamma_ng)

        # Sided traces (PyCutFEM)
        I_pc_u_pos_x = integrate_pc_scalar(dh, Pos(u_pos[0]), dGamma_pc, backend=backend)
        I_pc_u_neg_x = integrate_pc_scalar(dh, Neg(u_neg[0]), dGamma_pc, backend=backend)
        I_pc_u_pos_y = integrate_pc_scalar(dh, Pos(u_pos[1]), dGamma_pc, backend=backend)
        I_pc_u_neg_y = integrate_pc_scalar(dh, Neg(u_neg[1]), dGamma_pc, backend=backend)
        I_pc_p_pos = integrate_pc_scalar(dh, Pos(p_pos), dGamma_pc, backend=backend)
        I_pc_p_neg = integrate_pc_scalar(dh, Neg(p_neg), dGamma_pc, backend=backend)

        I_pc_jump_u_x = integrate_pc_scalar(dh, jump(u_pos[0], u_neg[0]), dGamma_pc, backend=backend)
        I_pc_jump_u_y = integrate_pc_scalar(dh, jump(u_pos[1], u_neg[1]), dGamma_pc, backend=backend)
        I_pc_jump_p = integrate_pc_scalar(dh, jump(p_pos, p_neg), dGamma_pc, backend=backend)

        n_pc = FacetNormal()
        I_pc_nx = integrate_pc_scalar(dh, n_pc[0], dGamma_pc, backend=backend)
        I_pc_ny = integrate_pc_scalar(dh, n_pc[1], dGamma_pc, backend=backend)
        I_pc_tpx = integrate_pc_scalar(dh, -(Pos(p_pos) * n_pc[0]), dGamma_pc, backend=backend)

        # NGSolve counterparts
        u_neg_ng = gfu.components[0].components[0]
        u_pos_ng = gfu.components[0].components[1]
        p_neg_ng = gfu.components[1].components[0]
        p_pos_ng = gfu.components[1].components[1]

        I_ng_u_pos_x = integrate_ng_scalar(ng_mesh, u_pos_ng[0], dGamma_ng)
        I_ng_u_neg_x = integrate_ng_scalar(ng_mesh, u_neg_ng[0], dGamma_ng)
        I_ng_u_pos_y = integrate_ng_scalar(ng_mesh, u_pos_ng[1], dGamma_ng)
        I_ng_u_neg_y = integrate_ng_scalar(ng_mesh, u_neg_ng[1], dGamma_ng)
        I_ng_p_pos = integrate_ng_scalar(ng_mesh, p_pos_ng, dGamma_ng)
        I_ng_p_neg = integrate_ng_scalar(ng_mesh, p_neg_ng, dGamma_ng)

        I_ng_jump_u_x = integrate_ng_scalar(ng_mesh, (u_pos_ng[0] - u_neg_ng[0]), dGamma_ng)
        I_ng_jump_u_y = integrate_ng_scalar(ng_mesh, (u_pos_ng[1] - u_neg_ng[1]), dGamma_ng)
        I_ng_jump_p = integrate_ng_scalar(ng_mesh, (p_pos_ng - p_neg_ng), dGamma_ng)

        I_ng_nx = integrate_ng_scalar(ng_mesh, n_ng[0], dGamma_ng)
        I_ng_ny = integrate_ng_scalar(ng_mesh, n_ng[1], dGamma_ng)
        I_ng_tpx = integrate_ng_scalar(ng_mesh, (-p_pos_ng * n_ng[0]), dGamma_ng)

        means_pc = dict(
            [
                _mean("u_pos_x", I_pc_u_pos_x, L_pc),
                _mean("u_neg_x", I_pc_u_neg_x, L_pc),
                _mean("u_pos_y", I_pc_u_pos_y, L_pc),
                _mean("u_neg_y", I_pc_u_neg_y, L_pc),
                _mean("p_pos", I_pc_p_pos, L_pc),
                _mean("p_neg", I_pc_p_neg, L_pc),
                _mean("jump(u_x)", I_pc_jump_u_x, L_pc),
                _mean("jump(u_y)", I_pc_jump_u_y, L_pc),
                _mean("jump(p)", I_pc_jump_p, L_pc),
                _mean("n_x", I_pc_nx, L_pc),
                _mean("n_y", I_pc_ny, L_pc),
                _mean("(-p_pos*n_x)", I_pc_tpx, L_pc),
            ]
        )
        means_ng = dict(
            [
                _mean("u_pos_x", I_ng_u_pos_x, L_ng),
                _mean("u_neg_x", I_ng_u_neg_x, L_ng),
                _mean("u_pos_y", I_ng_u_pos_y, L_ng),
                _mean("u_neg_y", I_ng_u_neg_y, L_ng),
                _mean("p_pos", I_ng_p_pos, L_ng),
                _mean("p_neg", I_ng_p_neg, L_ng),
                _mean("jump(u_x)", I_ng_jump_u_x, L_ng),
                _mean("jump(u_y)", I_ng_jump_u_y, L_ng),
                _mean("jump(p)", I_ng_jump_p, L_ng),
                _mean("n_x", I_ng_nx, L_ng),
                _mean("n_y", I_ng_ny, L_ng),
                _mean("(-p_pos*n_x)", I_ng_tpx, L_ng),
            ]
        )

        print(f"\n=== {title} ===")
        print(f"[pc] backend={backend}  nx={args.nx} ny={args.ny} q={args.q}  LΓ={L_pc:.8e}")
        print(f"[ng] maxh={args.maxh} order={args.order} q={args.q}           LΓ={L_ng:.8e}")
        print("")

        keys = list(expected.keys())
        max_err_pc_ng = 0.0
        max_err_pc_exp = 0.0
        for k in keys:
            pc_val = means_pc[k]
            ng_val = means_ng[k]
            exp_val = expected[k]
            err_pc_ng = abs(pc_val - ng_val)
            err_pc_exp = abs(pc_val - exp_val)
            max_err_pc_ng = max(max_err_pc_ng, err_pc_ng)
            max_err_pc_exp = max(max_err_pc_exp, err_pc_exp)
            print(
                f"{k:12s}  pc={pc_val:+.12e}  ng={ng_val:+.12e}  exp={exp_val:+.6e}  "
                f"|pc-ng|={err_pc_ng:.3e}  |pc-exp|={err_pc_exp:.3e}"
            )

        print("")
        print(f"max |pc-ng|  = {max_err_pc_ng:.3e}")
        print(f"max |pc-exp| = {max_err_pc_exp:.3e}")

    # --- Probe 1: constants (non-symmetric so jump != 0) ---
    if args.probe in {"const", "both"}:
        u_pos_const = np.array([1.25, -0.75], dtype=float)
        u_neg_const = np.array([-2.0, 0.5], dtype=float)
        p_pos_const = 3.0
        p_neg_const = -1.0

        _set_const("u_pos_x", u_pos_const[0], u_pos)
        _set_const("u_pos_y", u_pos_const[1], u_pos)
        _set_const("u_neg_x", u_neg_const[0], u_neg)
        _set_const("u_neg_y", u_neg_const[1], u_neg)
        _set_const("p_pos_", p_pos_const, p_pos)
        _set_const("p_neg_", p_neg_const, p_neg)

        with TaskManager():
            gfu.components[0].components[1].Set(CoefficientFunction(tuple(u_pos_const)))  # noqa: F405
            gfu.components[0].components[0].Set(CoefficientFunction(tuple(u_neg_const)))  # noqa: F405
            gfu.components[1].components[1].Set(CoefficientFunction(float(p_pos_const)))  # noqa: F405
            gfu.components[1].components[0].Set(CoefficientFunction(float(p_neg_const)))  # noqa: F405

        expected = {
            "u_pos_x": float(u_pos_const[0]),
            "u_neg_x": float(u_neg_const[0]),
            "u_pos_y": float(u_pos_const[1]),
            "u_neg_y": float(u_neg_const[1]),
            "p_pos": float(p_pos_const),
            "p_neg": float(p_neg_const),
            "jump(u_x)": float(u_pos_const[0] - u_neg_const[0]),
            "jump(u_y)": float(u_pos_const[1] - u_neg_const[1]),
            "jump(p)": float(p_pos_const - p_neg_const),
            "n_x": 1.0,
            "n_y": 0.0,
            "(-p_pos*n_x)": float(-p_pos_const),
        }
        _run_probe("Probe: piecewise constants", ng_set_cb=None, expected=expected)

    # --- Probe 2: quadratic profile on Γ (y^2) ---
    if args.probe in {"poly", "both"}:
        if int(args.order) < 2:
            raise ValueError("--probe poly requires --order >= 2 (to represent y^2 exactly).")

        # Define fields on Γ via global extension: a0 + a2 y^2
        u_pos_x = (0.7, 1.0)   # 0.7 + 1.0 y^2
        u_neg_x = (-0.2, 2.0)  # -0.2 + 2.0 y^2
        u_pos_y = (-1.1, 0.5)  # -1.1 + 0.5 y^2
        u_neg_y = (0.3, -0.25) # 0.3 - 0.25 y^2
        p_pos_const = 2.2
        p_neg_const = -0.4

        _set_y2("u_pos_x", u_pos_x[0], u_pos_x[1], u_pos)
        _set_y2("u_neg_x", u_neg_x[0], u_neg_x[1], u_neg)
        _set_y2("u_pos_y", u_pos_y[0], u_pos_y[1], u_pos)
        _set_y2("u_neg_y", u_neg_y[0], u_neg_y[1], u_neg)
        _set_const("p_pos_", p_pos_const, p_pos)
        _set_const("p_neg_", p_neg_const, p_neg)

        with TaskManager():
            gfu.components[0].components[1].Set(CoefficientFunction((u_pos_x[0] + u_pos_x[1] * y * y, u_pos_y[0] + u_pos_y[1] * y * y)))  # noqa: F405,E501
            gfu.components[0].components[0].Set(CoefficientFunction((u_neg_x[0] + u_neg_x[1] * y * y, u_neg_y[0] + u_neg_y[1] * y * y)))  # noqa: F405,E501
            gfu.components[1].components[1].Set(CoefficientFunction(float(p_pos_const)))  # noqa: F405,E501
            gfu.components[1].components[0].Set(CoefficientFunction(float(p_neg_const)))  # noqa: F405,E501

        # mean(y^2) on y∈[y0,y1]
        mean_y2 = (y1**3 - y0**3) / (3.0 * (y1 - y0))

        def mean_a0_a2(a0: float, a2: float) -> float:
            return float(a0 + a2 * mean_y2)

        expected = {
            "u_pos_x": mean_a0_a2(*u_pos_x),
            "u_neg_x": mean_a0_a2(*u_neg_x),
            "u_pos_y": mean_a0_a2(*u_pos_y),
            "u_neg_y": mean_a0_a2(*u_neg_y),
            "p_pos": float(p_pos_const),
            "p_neg": float(p_neg_const),
            "jump(u_x)": mean_a0_a2(u_pos_x[0] - u_neg_x[0], u_pos_x[1] - u_neg_x[1]),
            "jump(u_y)": mean_a0_a2(u_pos_y[0] - u_neg_y[0], u_pos_y[1] - u_neg_y[1]),
            "jump(p)": float(p_pos_const - p_neg_const),
            "n_x": 1.0,
            "n_y": 0.0,
            "(-p_pos*n_x)": float(-p_pos_const),
        }
        _run_probe("Probe: quadratic y^2 profiles", ng_set_cb=None, expected=expected)


if __name__ == "__main__":
    main()
