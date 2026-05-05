"""
Duddu et al. (2007) growth model (XFEM + level set): 1D verification (Fig. 4 / Table I).

This script solves the *quasi-steady* substrate diffusion + growth potential equations for a
flat slab biofilm of height `h_b` in a 2D strip domain (uniform-in-x).

Paper
-----
R. Duddu, S. Bordas, D. L. Chopp, B. Moran (2007)
"A combined extended finite element and level set method for biofilm growth"
Int. J. Numer. Meth. Engng.

What we reproduce
-----------------
- Fig. 4: substrate concentration S(y) and velocity potential Φ(y) profiles.
- Table I: interface normal speed F at slab height h_b=0.2 mm.

Notes
-----
- We follow Duddu et al. (2007) and use the discontinuous-derivative (shifted-|phi|) XFEM
  enrichment for both S and Φ:
      D_J(x) = N_J(x) (|phi(x)| - |phi_J|).
- Dirichlet Φ=0 on Γ_int is imposed with a penalty term (as in the paper).
- All outputs are written under examples/biofilms/benchmarks/dadu/results/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

import logging

from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.xfem import XFEMDofHandler
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    Function,
    Neg,
    TestFunction,
    TrialFunction,
    grad,
    inner,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dInterface, dx
from pycutfem.utils.meshgen import structured_triangles

from examples.biofilms.benchmarks.dadu.duddu2007_speed import (
    average_interface_speed,
    compute_interface_segment_speeds_duddu2007,
)


try:
    from petsc4py import PETSc  # type: ignore

    _HAS_PETSC = True
except Exception:
    PETSc = None
    _HAS_PETSC = False


@dataclass(frozen=True)
class Duddu2007Kinetics:
    # Active biomass volume fraction (assumed constant in Duddu 2007)
    f_active: float = 0.5
    # Table A.1 (Duddu 2009 appendix, Chopp 2002 parameters)
    rho_x: float = 1.0250  # mg VS / mm^3
    rho_w: float = 1.0125  # mg VS / mm^3
    Y_xO: float = 0.583  # mg VS / mg O2
    # NOTE: Duddu (2007) Table I (FD3000 speed at h_b=0.2mm) is only reproducible
    # with a smaller EPS yield than the value later reported in Duddu (2009) Appendix.
    # Keep this default aligned with the 2007 benchmark; override with --Y-wO if needed.
    Y_wO: float = 0.215  # mg VS / mg O2
    qhat0: float = 8.0  # mg O2 / (mg VS day)
    K0: float = 5.0e-7  # mg O2 / mm^3
    b: float = 0.3  # 1/day
    f_D: float = 0.8  # -
    g: float = 1.42  # mg O2 / mg VS

    def monod(self, S):
        return S / (Constant(float(self.K0)) + S)

    def consumption(self, S):
        """
        Positive substrate sink R(S) [mg O2 / (mm^3 day)] inside the biofilm.
        Duddu 2007 Eq.(1) has a notation inconsistency; we use the dimensionally
        consistent form (cf. Duddu 2009 Appendix Table A.1):
            R = f * rho_x * (qhat0 + g f_D b) * S/(K0+S).
        """
        f = Constant(float(self.f_active))
        rho_x = Constant(float(self.rho_x))
        qhat0 = Constant(float(self.qhat0))
        g = Constant(float(self.g))
        f_D = Constant(float(self.f_D))
        b = Constant(float(self.b))
        return f * rho_x * (qhat0 + g * f_D * b) * self.monod(S)

    def d_consumption(self, S):
        """dR/dS for Newton."""
        f = Constant(float(self.f_active))
        rho_x = Constant(float(self.rho_x))
        qhat0 = Constant(float(self.qhat0))
        g = Constant(float(self.g))
        f_D = Constant(float(self.f_D))
        b = Constant(float(self.b))
        K0 = Constant(float(self.K0))
        return f * rho_x * (qhat0 + g * f_D * b) * (K0 / (K0 + S) / (K0 + S))

    def divU(self, S):
        """
        Divergence source for Φ: ΔΦ = div(U) with U=∇Φ.
        Duddu 2007: div(U) = f (ρ_x + ρ_w) with rates:
            ρ_x = (Y_xO qhat0 - b) * S/(K0+S)
            ρ_w = (rho_x/rho_w) * ((1-f_D) b + Y_wO qhat0) * S/(K0+S)
        Units: 1/day.
        """
        f = Constant(float(self.f_active))
        Y_xO = Constant(float(self.Y_xO))
        Y_wO = Constant(float(self.Y_wO))
        qhat0 = Constant(float(self.qhat0))
        b = Constant(float(self.b))
        f_D = Constant(float(self.f_D))
        rho_x = Constant(float(self.rho_x))
        rho_w = Constant(float(self.rho_w))
        mon = self.monod(S)
        rho_x_rate = (Y_xO * qhat0 - b) * mon
        rho_w_rate = (rho_x / rho_w) * ((Constant(1.0) - f_D) * b + Y_wO * qhat0) * mon
        return f * (rho_x_rate + rho_w_rate)

    def divU_numpy(self, S: np.ndarray) -> np.ndarray:
        """Vectorized div(U) for postprocessing (S in mgO2/mm^3)."""
        S = np.asarray(S, dtype=float)
        K0 = float(self.K0)
        mon = S / (K0 + S)
        rho_x_rate = (float(self.Y_xO) * float(self.qhat0) - float(self.b)) * mon
        rho_w_rate = (float(self.rho_x) / float(self.rho_w)) * (
            (1.0 - float(self.f_D)) * float(self.b) + float(self.Y_wO) * float(self.qhat0)
        ) * mon
        return float(self.f_active) * (rho_x_rate + rho_w_rate)


def _solve_linear_system(A, rhs, *, linear_solver: str) -> np.ndarray:
    linear_solver = str(linear_solver).strip().lower()
    if linear_solver in {"scipy", "spsolve", "direct"}:
        return spla.spsolve(A.tocsc(), rhs)
    if linear_solver in {"petsc", "ksp"}:
        if not _HAS_PETSC:
            raise RuntimeError("petsc4py is not available but --linear-solver petsc was requested.")
        A_csr = A.tocsr()
        Ap = PETSc.Mat().createAIJ(size=A_csr.shape, csr=(A_csr.indptr, A_csr.indices, A_csr.data))
        bp = PETSc.Vec().createWithArray(np.asarray(rhs, dtype=float))
        xp = bp.duplicate()
        ksp = PETSc.KSP().create()
        ksp.setOperators(Ap)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        ksp.setFromOptions()
        ksp.solve(bp, xp)
        return np.asarray(xp.getArray(), dtype=float)
    raise ValueError(f"Unknown linear_solver={linear_solver!r}. Use 'scipy' or 'petsc'.")


def _grid_view_from_field(dh: DofHandler, values: np.ndarray, *, nx: int, ny: int) -> np.ndarray:
    """
    Reshape nodal CG(Q1) field values into (ny+1, nx+1) assuming structured_quad ordering.
    """
    arr = np.asarray(values, dtype=float).ravel()
    expected = int((nx + 1) * (ny + 1))
    if arr.size != expected:
        raise ValueError(f"Expected {(ny+1)}x{(nx+1)} nodal values ({expected}), got {arr.size}.")
    return arr.reshape((ny + 1, nx + 1))


def _fd_grad_xy(u: np.ndarray, *, dxh: float, dyh: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Second-order central differences in the interior, one-sided at boundaries.
    u is (ny+1, nx+1).
    """
    u = np.asarray(u, dtype=float)
    du_dx = np.zeros_like(u)
    du_dy = np.zeros_like(u)

    # x-derivative
    du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * float(dxh))
    du_dx[:, 0] = (u[:, 1] - u[:, 0]) / float(dxh)
    du_dx[:, -1] = (u[:, -1] - u[:, -2]) / float(dxh)

    # y-derivative
    du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2.0 * float(dyh))
    du_dy[0, :] = (u[1, :] - u[0, :]) / float(dyh)
    du_dy[-1, :] = (u[-1, :] - u[-2, :]) / float(dyh)

    return du_dx, du_dy


def _assemble_scalar(dof_handler, functional, *, backend: str) -> float:
    """Assemble a pure functional (no test/trial) and return it as a float."""
    hooks = {functional.integrand: {"name": "val"}}
    res = assemble_form(
        Equation(None, functional),
        dof_handler=dof_handler,
        bcs=[],
        backend=str(backend),
        assembler_hooks=hooks,
    )
    return float(np.asarray(res["val"], dtype=float).reshape(()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="examples/biofilms/benchmarks/dadu/results/duddu2007_fig4_1d")
    p.add_argument("--backend", choices=("python", "jit", "cpp"), default="cpp")
    p.add_argument("--linear-solver", choices=("scipy", "petsc"), default="petsc")
    p.add_argument("--q", type=int, default=6)
    p.add_argument("--nx", type=int, default=6)
    p.add_argument("--ny", type=int, default=200)
    p.add_argument(
        "--enrichment",
        choices=("abs", "none"),
        default="abs",
        help="XFEM enrichment for S and Phi (paper uses shifted-|phi|='abs').",
    )
    p.add_argument("--Y-xO", type=float, default=Duddu2007Kinetics.Y_xO, help="Yield of active biomass due to substrate consumption.")
    p.add_argument(
        "--Y-wO",
        type=float,
        default=Duddu2007Kinetics.Y_wO,
        help="Yield of EPS due to substrate consumption. Duddu (2009) Appendix lists 0.477, while Duddu (2007) Table I aligns with ~0.215.",
    )
    p.add_argument("--width", type=float, default=0.01, help="Strip width (mm).")
    p.add_argument("--H", type=float, default=0.3, help="Domain height (mm).")
    p.add_argument("--h-b", type=float, default=0.2, help="Biofilm slab height (mm).")
    p.add_argument("--Sbar", type=float, default=8.3e-6, help="Bulk substrate concentration (mgO2/mm^3).")
    p.add_argument("--Db", type=float, default=146.88, help="Biofilm diffusion coefficient (mm^2/day).")
    p.add_argument("--Df", type=float, default=183.6, help="Fluid diffusion coefficient (mm^2/day).")
    p.add_argument("--newton-tol", type=float, default=1.0e-10)
    p.add_argument("--max-it", type=int, default=50)
    p.add_argument("--newton-verbose", action="store_true", help="Print residual norms per Newton iteration.")
    p.add_argument("--S-min", type=float, default=1.0e-16, help="Lower bound for Newton updates (avoid negative S).")
    p.add_argument(
        "--phi-penalty",
        type=float,
        default=1.0e6,
        help="Penalty parameter (multiplied by 1/h) for imposing Φ=0 on Γ_int.",
    )
    p.add_argument("--compiler-verbose", action="store_true", help="Enable compiler/assembly INFO logs.")
    args = p.parse_args()

    if not bool(getattr(args, "compiler_verbose", False)):
        logging.getLogger("pycutfem").setLevel(logging.WARNING)
        logging.getLogger("pycutfem.ufl.compilers").setLevel(logging.WARNING)

    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    kin = Duddu2007Kinetics(Y_xO=float(args.Y_xO), Y_wO=float(args.Y_wO))

    # --- mesh / handler -------------------------------------------------
    nodes, elems, edges, corners = structured_triangles(
        Lx=float(args.width),
        Ly=float(args.H),
        nx_quads=int(args.nx),
        ny_quads=int(args.ny),
        poly_order=1,
        offset=(0.0, 0.0),
    )
    mesh = Mesh(nodes, elems, edges, corners, element_type="tri", poly_order=1)
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= 1.0e-12,
            "right": lambda x, y: abs(x - float(args.width)) <= 1.0e-12,
            "bottom": lambda x, y: abs(y - 0.0) <= 1.0e-12,
            "top": lambda x, y: abs(y - float(args.H)) <= 1.0e-12,
        }
    )

    me = MixedElement(mesh, {"S": 1, "Phi": 1})
    dh0 = DofHandler(me, method="cg")

    # slab interface: φ(x,y)=y-h_b (negative in biofilm)
    level_set = AffineLevelSet(a=0.0, b=1.0, c=-float(args.h_b)).normalised()
    dh0.classify_from_levelset(level_set)

    if str(args.enrichment).lower().strip() == "none":
        dh = dh0
        base_dh = dh0
    else:
        dh = XFEMDofHandler(dh0)
        dh.rebuild_enrichment(level_set, enrich={"S": "abs", "Phi": "abs"})
        base_dh = dh.base

    # --- unknowns -------------------------------------------------------
    S_k = Function(name="S", field_name="S", dof_handler=dh)
    Phi_k = Function(name="Phi", field_name="Phi", dof_handler=dh)

    bc_top_S_val = BoundaryCondition("S", "dirichlet", "top", lambda x, y: float(args.Sbar))
    # Newton increment ΔS must satisfy homogeneous Dirichlet BCs.
    bc_top_S_homog = BoundaryCondition("S", "dirichlet", "top", lambda x, y: 0.0)

    # --- measures -------------------------------------------------------
    q = int(args.q)
    dx_pos = dx(level_set=level_set, metadata={"side": "+", "q": q})
    dx_neg = dx(level_set=level_set, metadata={"side": "-", "q": q})
    dGamma = dInterface(defined_on=mesh.element_bitset("cut"), level_set=level_set, metadata={"q": q})

    # ------------------------------------------------------------------
    # Substrate: Newton solve on Ω = Ω_b ∪ Ω_f
    # ------------------------------------------------------------------
    # initial guess
    S_k.nodal_values[:] = float(args.Sbar)
    # enforce the Dirichlet values on S (works even on unfitted meshes)
    S_dir = dh.get_dirichlet_data([bc_top_S_val])
    for gd, vv in S_dir.items():
        li = S_k._g2l.get(int(gd))
        if li is not None:
            S_k.nodal_values[int(li)] = float(vv)

    vS = TestFunction("S", dof_handler=dh)
    dS = TrialFunction("S", dof_handler=dh)

    Db = Constant(float(args.Db))
    Df = Constant(float(args.Df))

    R = kin.consumption(S_k)
    dR = kin.d_consumption(S_k)

    r_S = Db * inner(grad(S_k), grad(vS)) * dx_neg + Df * inner(grad(S_k), grad(vS)) * dx_pos + R * vS * dx_neg
    a_S = Db * inner(grad(dS), grad(vS)) * dx_neg + Df * inner(grad(dS), grad(vS)) * dx_pos + dR * dS * vS * dx_neg

    # Φ equation (linear), assembled alongside S to avoid singular inactive blocks.
    dPhi = TrialFunction("Phi", dof_handler=dh)
    vPhi = TestFunction("Phi", dof_handler=dh)
    pen = Constant(float(args.phi_penalty))
    h = CellDiameter()
    f_src = kin.divU(S_k)
    r_Phi = inner(grad(Phi_k), grad(vPhi)) * dx_neg + inner(grad(Phi_k), grad(vPhi)) * dx_pos
    # Duddu 2007 Eq.(7): ΔΦ = f(ρ_x+ρ_w) in Ω_b, Φ=0 on Γ_int ⇒ weak form uses +f_src*v.
    r_Phi += f_src * vPhi * dx_neg + (pen / h) * Phi_k * vPhi * dGamma
    a_Phi = inner(grad(dPhi), grad(vPhi)) * dx_neg + inner(grad(dPhi), grad(vPhi)) * dx_pos
    a_Phi += (pen / h) * dPhi * vPhi * dGamma

    # Newton iterations
    S_min = float(args.S_min)
    for it in range(int(args.max_it)):
        # keep Dirichlet values fixed in the current iterate
        for gd, vv in S_dir.items():
            li = S_k._g2l.get(int(gd))
            if li is not None:
                S_k.nodal_values[int(li)] = float(vv)

        A, r_vec = assemble_form(
            Equation(a_S + a_Phi, r_S + r_Phi),
            dof_handler=dh,
            bcs=[bc_top_S_homog],
            backend=str(args.backend),
        )

        slS = np.asarray(dh.get_field_slice("S"), dtype=int)
        slPhi = np.asarray(dh.get_field_slice("Phi"), dtype=int)
        rS_norm = float(np.linalg.norm(np.asarray(r_vec[slS], dtype=float)))
        rP_norm = float(np.linalg.norm(np.asarray(r_vec[slPhi], dtype=float)))
        if bool(getattr(args, "newton_verbose", False)):
            print(f"[newton] it={it:02d}  ||r_S||={rS_norm:.3e}  ||r_Phi||={rP_norm:.3e}")
        if (rS_norm <= float(args.newton_tol)) and (rP_norm <= float(args.newton_tol)):
            break

        delta = _solve_linear_system(A, -np.asarray(r_vec, dtype=float), linear_solver=str(args.linear_solver))

        # Update S (clip only base nodal DOFs; enriched coefficients are not nodal values).
        dS_vec = np.asarray(delta[slS], dtype=float)
        S_new = np.asarray(S_k.nodal_values, dtype=float) + dS_vec
        if S_min > 0.0:
            n_base_S = int(np.asarray(base_dh.get_field_slice("S"), dtype=int).size)
            S_new[:n_base_S] = np.maximum(S_new[:n_base_S], S_min)
        S_k.nodal_values[:] = S_new

        # Update Φ
        Phi_k.nodal_values[:] = np.asarray(Phi_k.nodal_values, dtype=float) + np.asarray(delta[slPhi], dtype=float)
    else:
        raise RuntimeError("Substrate Newton did not converge.")

    # final Dirichlet enforcement
    for gd, vv in S_dir.items():
        li = S_k._g2l.get(int(gd))
        if li is not None:
            S_k.nodal_values[int(li)] = float(vv)

    # ------------------------------------------------------------------
    # Postprocess: profiles + speed
    # ------------------------------------------------------------------
    # Duddu (2007) normal speed computation (Section 5.2 / Fig. 3).
    seg_speeds = compute_interface_segment_speeds_duddu2007(dof_handler=dh, level_set=level_set, Phi=Phi_k, field="Phi")
    F_est_duddu = average_interface_speed(seg_speeds)

    # Preferred (uses the full XFEM field, incl. enrichment): for a 1D slab,
    #   F = ∫_0^{h_b} div(U) dy = (1/width) ∫_{Ω_b} div(U) dΩ.
    I_divU = _assemble_scalar(dh, kin.divU(S_k) * dx_neg, backend=str(args.backend))
    F_est_int = float(I_divU) / float(args.width)

    # Extract 1D profiles at x ~ mid using base DOF coordinates (CG P1).
    def _profile(field_name: str, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coords = np.asarray(base_dh.get_dof_coords(field_name), dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise RuntimeError(f"Invalid DOF coordinates for field {field_name!r}.")
        x_levels = np.unique(np.round(coords[:, 0], decimals=14))
        x_mid = 0.5 * float(args.width)
        x_sel = float(x_levels[int(np.argmin(np.abs(x_levels - x_mid)))])
        tolx = 1.0e-10 * max(1.0, float(args.width))
        mask = np.abs(coords[:, 0] - x_sel) <= tolx
        y = coords[mask, 1]
        u = np.asarray(values, dtype=float).ravel()[mask]
        order = np.argsort(y)
        return np.asarray(y[order], dtype=float), np.asarray(u[order], dtype=float)

    n_base_S = int(np.asarray(base_dh.get_field_slice("S"), dtype=int).size)
    n_base_Phi = int(np.asarray(base_dh.get_field_slice("Phi"), dtype=int).size)
    y_prof, S_prof = _profile("S", np.asarray(S_k.nodal_values[:n_base_S], dtype=float))
    y_phi, P_prof = _profile("Phi", np.asarray(Phi_k.nodal_values[:n_base_Phi], dtype=float))
    if y_phi.size == y_prof.size and np.max(np.abs(y_phi - y_prof)) <= 1.0e-12:
        y_out = y_prof
    else:
        y_out = y_prof

    np.savetxt(outdir / "profile_S.txt", np.column_stack([y_prof, S_prof]), header="y_mm  S_mgO2_per_mm3")
    np.savetxt(outdir / "profile_Phi.txt", np.column_stack([y_phi, P_prof]), header="y_mm  Phi")

    print("\nDuddu2007 Fig4/TabI | slab h_b=%.3f mm" % float(args.h_b))
    print(f"- Estimated interface speed F_duddu ≈ {F_est_duddu:.6g} mm/day (Duddu 5.2 scheme)")
    print(f"- Estimated interface speed F_int  ≈ {F_est_int:.6g} mm/day (from ∫ div(U) dy)")
    print("  Table I (paper): XFEM200 ~ 0.0097 mm/day")
    print(f"- Wrote {outdir/'profile_S.txt'}")
    print(f"- Wrote {outdir/'profile_Phi.txt'}")

    # Machine-readable summary for reproducible comparisons.
    summary = {
        "nx_quads": int(args.nx),
        "ny_quads": int(args.ny),
        "width_mm": float(args.width),
        "H_mm": float(args.H),
        "h_b_mm": float(args.h_b),
        "Y_xO": float(args.Y_xO),
        "Y_wO": float(args.Y_wO),
        "Db_mm2_per_day": float(args.Db),
        "Df_mm2_per_day": float(args.Df),
        "Sbar_mgO2_per_mm3": float(args.Sbar),
        "enrichment": str(args.enrichment),
        "phi_penalty": float(args.phi_penalty),
        "F_duddu_mm_per_day": float(F_est_duddu),
        "F_int_mm_per_day": float(F_est_int),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"- Wrote {outdir/'summary.json'}")

    # Optional plots (matplotlib is not a hard dep)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        ax[0].plot(y_prof, S_prof, "k-")
        ax[0].set_xlabel("y (mm)")
        ax[0].set_ylabel("S (mgO2/mm^3)")
        ax[0].set_title("Substrate S(y)")
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(y_phi, P_prof, "k-")
        ax[1].set_xlabel("y (mm)")
        ax[1].set_ylabel("Phi")
        ax[1].set_title("Velocity potential Φ(y)")
        ax[1].grid(True, alpha=0.3)

        fig.suptitle("Duddu et al. (2007) | 1D slab verification", fontsize=12)
        outpng = outdir / "fig4_profiles.png"
        fig.savefig(outpng, dpi=200)
        plt.close(fig)
        print(f"- Wrote {outpng}")
    except Exception as e:
        print(f"[warn] plotting skipped: {e}")


if __name__ == "__main__":
    main()
