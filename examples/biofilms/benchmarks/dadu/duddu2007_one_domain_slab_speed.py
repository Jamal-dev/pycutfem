"""
One-domain calibration against Duddu et al. (2007) Table I (1D slab speed).

Goal
----
Use the *one-domain* diffuse-interface model (`examples/utils/biofilm/one_domain.py`)
in a Duddu(2007)-equivalent limit (growth-only, no detachment) and compute the
initial interface speed for a slab biofilm of height h_b = 0.2mm in a 0.3mm
domain (Fig. 4 / Table I setup).

We keep the biofilm indicator `alpha` *fixed* (diffuse step profile) and solve a
quasi-steady substrate field S and a Darcy-like skeleton velocity vS driven by
the Duddu(2007) growth source s_v(S). The speed is then estimated by:
  (1) source-integral:   F_int = ∫_{Ω} alpha * divU(S) dΩ / width
  (2) interface average: F_vS  = ∫ vS·n δ(alpha) dΩ / ∫ δ(alpha) dΩ

All outputs are written under examples/biofilms/benchmarks/dadu/results/.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import (
    HAS_PETSC,
    LinearSolverParameters,
    NewtonParameters,
    NewtonSolver,
    PdasNewtonSolver,
    TimeStepperParameters,
    VIParameters,
)
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    dot,
    grad,
)
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad

from examples.utils.biofilm.adhesion import assemble_scalar
from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms


def _tag_rectangle_boundaries(mesh: Mesh, *, L: float, H: float, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - float(L)) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - float(H)) <= tol,
        }
    )


def _set_inactive_fields(dh: DofHandler, field_names: list[str]) -> None:
    inactive: set[int] = set()
    for fname in field_names:
        try:
            sl = np.asarray(dh.get_field_slice(fname), dtype=int).ravel()
        except Exception:
            continue
        inactive.update(int(i) for i in sl)
    dh.dof_tags = {"inactive": inactive}


def _smooth_step(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(np.asarray(z, dtype=float)))


def _update_phi_from_alpha(
    *,
    phi: Function,
    alpha: Function,
    phi_b: float,
    alpha0: float = 0.1,
    alpha_width: float = 0.05,
) -> None:
    """
    Set porosity field from the diffuse indicator.

    For the Duddu(2007) growth-only limit we want:
      - phi ≈ phi_b in the biofilm, and
      - phi ≈ 1 in the fluid,
    while keeping substrate uptake and growth sources localized like the sharp-interface
    model (avoid (1-phi)*alpha ~ alpha^2 when phi is blended linearly with alpha).

    We therefore use a sharp-but-smooth mapping in alpha-space:
      phi = 1 - (1-phi_b) * H_alpha(alpha),
    where H_alpha transitions from 0→1 around alpha≈alpha0 with width alpha_width.
    """
    a = np.asarray(alpha.nodal_values, dtype=float)
    w = _smooth_step((a - float(alpha0)) / (2.0 * max(float(alpha_width), 1.0e-12)))
    phi.nodal_values[:] = 1.0 - (1.0 - float(phi_b)) * w


def _alpha_slab(x, y, *, h_b: float, eps: float) -> np.ndarray:
    """
    Diffuse Heaviside: alpha≈1 for y<h_b, alpha≈0 for y>h_b.

    Uses a tanh profile with half-thickness ~eps (consistent with other benchmarks).
    """
    eps = max(float(eps), 1.0e-12)
    xx, yy = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return np.clip(_smooth_step((float(h_b) - yy) / (2.0 * eps)), 0.0, 1.0)


def _as_float_time(fn):
    return lambda x, y, t: float(np.asarray(fn(np.asarray(x), np.asarray(y), float(t))).reshape(()))


def _write_vertical_profiles(
    *,
    outdir: Path,
    dh: DofHandler,
    L: float,
    S: Function,
    p: Function,
    vS: VectorFunction,
    h_b: float,
) -> None:
    """
    Write 1D profiles S(y) and Φ(y) for the Duddu(2007) Fig.4 comparison.

    For the slab setup the solution is uniform in x; we therefore extract the
    profile on the x-column closest to x=L/2.
    """

    def _profile(field_name: str, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coords = np.asarray(dh.get_dof_coords(field_name), dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise RuntimeError(f"Invalid DOF coordinates for field {field_name!r}.")
        x_levels = np.unique(np.round(coords[:, 0], decimals=14))
        x_mid = 0.5 * float(L)
        x_sel = float(x_levels[int(np.argmin(np.abs(x_levels - x_mid)))])
        tolx = 1.0e-10 * max(1.0, float(L))
        mask = np.abs(coords[:, 0] - x_sel) <= tolx
        y = coords[mask, 1]
        u = np.asarray(values, dtype=float).ravel()[mask]
        order = np.argsort(y)
        return np.asarray(y[order], dtype=float), np.asarray(u[order], dtype=float)

    yS, S_prof = _profile("S", np.asarray(S.nodal_values, dtype=float))

    # Duddu(2007) defines the growth velocity as U = ∇Φ in the biofilm and enforces Φ=0
    # in the fluid. To obtain a directly comparable potential profile in the one-domain
    # setting, we reconstruct Φ from the solved skeleton velocity by integrating
    # vS_y = dΦ/dy and enforcing Φ(h_b)=0 (and Φ=0 for y>h_b).
    yV, vSy = _profile("vS_y", np.asarray(vS.nodal_values_component(1), dtype=float))
    Phi = np.zeros_like(yV)
    # Closest y-index at/just below the interface location.
    cand = np.nonzero(yV <= float(h_b) + 1.0e-12)[0]
    j_int = int(cand[-1]) if cand.size else int(np.argmin(np.abs(yV - float(h_b))))
    Phi[j_int:] = 0.0
    for j in range(j_int - 1, -1, -1):
        dy = float(yV[j + 1] - yV[j])
        Phi[j] = float(Phi[j + 1] - 0.5 * (float(vSy[j + 1]) + float(vSy[j])) * dy)

    np.savetxt(outdir / "profile_S.txt", np.column_stack([yS, S_prof]), header="y_mm  S_mgO2_per_mm3")
    np.savetxt(outdir / "profile_Phi.txt", np.column_stack([yV, Phi]), header="y_mm  Phi")
    # Keep the raw pressure/potential surrogate for debugging (not used in Fig.4 plot).
    yp, p_prof = _profile("p", np.asarray(p.nodal_values, dtype=float))
    np.savetxt(outdir / "profile_p.txt", np.column_stack([yp, p_prof]), header="y_mm  p")
    print(f"- Wrote {outdir/'profile_S.txt'}")
    print(f"- Wrote {outdir/'profile_Phi.txt'}")
    print(f"- Wrote {outdir/'profile_p.txt'}")


@dataclass(frozen=True)
class Duddu2007Params:
    # active biomass fraction
    f_active: float = 0.5
    # densities (mgVS/mm^3)
    rho_x: float = 1.0250
    rho_w: float = 1.0125
    # yields (mgVS/mgO2)
    Y_xO: float = 0.583
    Y_wO: float = 0.215
    # kinetics
    qhat0: float = 8.0  # mgO2/(mgVS day)
    K0: float = 5.0e-7  # mgO2/mm^3
    b: float = 0.3  # 1/day
    f_D: float = 0.8
    g: float = 1.42  # mgO2/mgVS
    # substrate Dirichlet
    Sbar: float = 8.3e-6  # mgO2/mm^3

    def monod(self, S):
        K0 = Constant(float(self.K0))
        return S / (S + K0)

    def divU(self, S):
        """
        Duddu (2007) Eq.(4)-(7) reduced: divU(S) [1/day] used in ΔΦ = divU in Ω_b.
        """
        f = Constant(float(self.f_active))
        rho_x = Constant(float(self.rho_x))
        rho_w = Constant(float(self.rho_w))
        Y_xO = Constant(float(self.Y_xO))
        Y_wO = Constant(float(self.Y_wO))
        qhat0 = Constant(float(self.qhat0))
        b = Constant(float(self.b))
        f_D = Constant(float(self.f_D))
        mon = self.monod(S)
        rho_x_rate = (Y_xO * qhat0 - b) * mon
        rho_w_rate = (rho_x / rho_w) * ((Constant(1.0) - f_D) * b + Y_wO * qhat0) * mon
        return f * (rho_x_rate + rho_w_rate)

    def d_divU(self, S, dS):
        """
        Gateaux derivative of divU w.r.t S in the trial direction dS.
        """
        f = Constant(float(self.f_active))
        rho_x = Constant(float(self.rho_x))
        rho_w = Constant(float(self.rho_w))
        Y_xO = Constant(float(self.Y_xO))
        Y_wO = Constant(float(self.Y_wO))
        qhat0 = Constant(float(self.qhat0))
        b = Constant(float(self.b))
        f_D = Constant(float(self.f_D))
        K0 = Constant(float(self.K0))

        coeff = (Y_xO * qhat0 - b) + (rho_x / rho_w) * ((Constant(1.0) - f_D) * b + Y_wO * qhat0)
        denom = S + K0
        dmon = (K0 / (denom * denom)) * dS
        return f * coeff * dmon


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="examples/biofilms/benchmarks/dadu/results/duddu2007_one_domain_slab_speed")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--linear-solver", type=str, default="petsc", choices=("petsc", "scipy"))

    # Geometry (mm)
    ap.add_argument("--L", type=float, default=0.05, help="Strip width (mm).")
    ap.add_argument("--H", type=float, default=0.3, help="Domain height (mm).")
    ap.add_argument("--h-b", type=float, default=0.2, help="Initial slab biofilm height (mm).")
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--ny", type=int, default=240)
    ap.add_argument("--q", type=int, default=4)

    # Diffuse interface / porosity
    ap.add_argument("--eps-alpha", type=float, default=0.01, help="Diffuse interface half-thickness (mm).")
    ap.add_argument("--phi-b", type=float, default=0.3, help="Frozen biofilm porosity phi (dimensionless).")

    # Substrate diffusion (mm^2/day). Duddu (2007): Df=183.6, Db=146.88.
    ap.add_argument("--D-S", type=float, default=183.6, help="Effective substrate diffusion coefficient.")

    # Darcy/extension regularization for vS outside biofilm
    ap.add_argument("--kappa-inv", type=float, default=1.0e6, help="Inverse permeability κ^{-1} (scalar).")
    ap.add_argument("--mu-f", type=float, default=1.0, help="Fluid viscosity in model units (only affects drag scaling).")
    ap.add_argument("--gamma-vS", type=float, default=1.0e2, help="vS extension penalty weight in fluid region.")
    ap.add_argument("--vS-ext-mode", type=str, default="l2", choices=("l2", "grad"))
    ap.add_argument("--gamma-vS-pin", type=float, default=1.0e-6, help="Small pinning to remove rigid-translation nullspace.")
    ap.add_argument(
        "--gamma-p-out",
        type=float,
        default=1.0e6,
        help="Penalty to enforce p≈0 in the fluid (alpha≈0), mimicking Duddu's Φ≡0 in Ω_f.",
    )
    ap.add_argument(
        "--gamma-p-out-power",
        type=int,
        default=8,
        help="Exponent m in the weight (1-alpha)^m used for the p-out penalty (larger m localizes the penalty deeper in the fluid).",
    )

    # Newton
    ap.add_argument("--newton-tol", type=float, default=1.0e-10)
    ap.add_argument("--max-it", type=int, default=40)
    ap.add_argument("--dt-steady", type=float, default=1.0e6, help="Large dt used to approximate quasi-steady S solve (days).")

    args = ap.parse_args()

    if str(args.linear_solver).lower() == "petsc" and not HAS_PETSC:
        raise RuntimeError("PETSc requested but not available in this environment.")

    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    L = float(args.L)
    H = float(args.H)
    nx = int(args.nx)
    ny = int(args.ny)
    qdeg = int(args.q)

    # Mesh
    nodes, elems, _, corners = structured_quad(L, H, nx=nx, ny=ny, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    _tag_rectangle_boundaries(mesh, L=L, H=H)

    field_specs: dict[str, object] = {
        "v_x": 2,
        "v_y": 2,
        "p": 1,
        "vS_x": 2,
        "vS_y": 2,
        "u_x": 2,
        "u_y": 2,
        "phi": 1,
        "alpha": 1,
        "S": 1,
    }
    me = MixedElement(mesh, field_specs=field_specs)
    dh = DofHandler(me, method="cg")

    # We solve only (p, vS, S). All other fields are frozen coefficients.
    _set_inactive_fields(dh, ["v_x", "v_y", "u_x", "u_y", "phi", "alpha"])

    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    VS = FunctionSpace("VS", ["vS_x", "vS_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    dv = VectorTrialFunction(space=V, dof_handler=dh)
    dvS = VectorTrialFunction(space=VS, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    dphi = TrialFunction("phi", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    dS = TrialFunction("S", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    vS_test = VectorTestFunction(space=VS, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    S_test = TestFunction("S", dof_handler=dh)

    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    vS_k = VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    S_k = Function("S_k", "S", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    vS_n = VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    S_n = Function("S_n", "S", dof_handler=dh)

    # Frozen fields
    for vf in (v_k, vS_k, u_k, v_n, vS_n, u_n):
        vf.nodal_values[:] = 0.0
    for sf in (p_k, p_n):
        sf.nodal_values[:] = 0.0

    phi_b = float(args.phi_b)
    phi_n.set_values_from_function(lambda x, y: float(phi_b))
    phi_k.nodal_values[:] = phi_n.nodal_values[:]

    h_b = float(args.h_b)
    eps_alpha = float(args.eps_alpha)
    alpha_n.set_values_from_function(lambda x, y: _alpha_slab(x, y, h_b=h_b, eps=eps_alpha))
    alpha_k.nodal_values[:] = alpha_n.nodal_values[:]

    # Porosity: set to phi_b in biofilm (alpha≈1) and 1 in fluid (alpha≈0).
    _update_phi_from_alpha(phi=phi_n, alpha=alpha_n, phi_b=phi_b)
    phi_k.nodal_values[:] = phi_n.nodal_values[:]

    kin = Duddu2007Params()
    S_n.set_values_from_function(lambda x, y: float(kin.Sbar))
    S_k.nodal_values[:] = S_n.nodal_values[:]

    dt_c = Constant(float(args.dt_steady))
    dx_q = dx(metadata={"q": int(qdeg)})

    # ------------------------------------------------------------------
    # Source in mass/volume constraint: match Duddu's divU(S) in the biofilm.
    # Constraint is div(C v + B vS) = alpha * s_v.
    #
    # With v frozen to 0 and B=alpha*(1-phi), we set s_v=(1-phi)*divU(S) so that
    # alpha*s_v = B*divU. This avoids spurious forcing in the diffuse interface
    # where phi transitions toward 1 (fluid).
    # ------------------------------------------------------------------
    one_m_phi_b = 1.0 - float(phi_b)
    if one_m_phi_b <= 0.0:
        raise ValueError("--phi-b must be < 1.")

    divU_k = kin.divU(S_k)
    s_v = ((-phi_k) + Constant(1.0)) * divU_k
    # In this benchmark we solve the substrate equation first and then solve (p,vS)
    # with S treated as frozen data. Therefore we do not need the consistent
    # derivative ds_v/dS in the Newton step for the (p,vS) block.
    ds_v = Constant(0.0)

    # ------------------------------------------------------------------
    # Match Duddu's substrate consumption μ_S by choosing mu_max so that the
    # one-domain sink RS = (rho_s_star/Y) (monod-k_d) (1-phi) alpha equals:
    #   μ_S = f*rho_x*(qhat0 + g f_D b) * S/(K0+S)  in Ω_b.
    # With k_d=0, K_S=K0, rho_s_star=rho_x, Y=Y_xO this yields:
    #   mu_max = f*(qhat0 + g f_D b) * Y_xO / (1-phi_b).
    # ------------------------------------------------------------------
    desired_uptake = kin.f_active * (kin.qhat0 + kin.g * kin.f_D * kin.b)  # [mgO2/(mgVS day)]
    mu_max = float(desired_uptake) * float(kin.Y_xO) / float(one_m_phi_b)  # [1/day]

    forms = build_biofilm_one_domain_forms(
        v_k=v_k,
        p_k=p_k,
        vS_k=vS_k,
        u_k=u_k,
        phi_k=phi_k,
        alpha_k=alpha_k,
        S_k=S_k,
        v_n=v_n,
        p_n=p_n,
        vS_n=vS_n,
        u_n=u_n,
        phi_n=phi_n,
        alpha_n=alpha_n,
        S_n=S_n,
        dv=dv,
        dp=dp,
        dvS=dvS,
        du=du,
        dphi=dphi,
        dalpha=dalpha,
        dS=dS,
        v_test=v_test,
        q_test=q_test,
        vS_test=vS_test,
        u_test=u_test,
        phi_test=phi_test,
        alpha_test=alpha_test,
        S_test=S_test,
        dx=dx_q,
        dt=dt_c,
        theta=1.0,
        # Flow: disable inertia (Stokes/Brinkman)
        rho_f=Constant(0.0),
        mu_f=Constant(float(args.mu_f)),
        kappa_inv=Constant(float(args.kappa_inv)),
        mu_b_model="phi_mu",
        # Skeleton: essentially rigid/frozen (u is inactive)
        solid_model="linear",
        mu_s=Constant(1.0e-8),
        lambda_s=Constant(1.0e-8),
        solid_visco_eta=0.0,
        # Transport: freeze phi, alpha (inactive), solve only S
        D_phi=0.0,
        gamma_phi=0.0,
        D_alpha=0.0,
        alpha_advection_form="advective",
        alpha_cahn_M=0.0,
        alpha_cahn_gamma=0.0,
        alpha_cahn_eps=1.0,
        alpha_cahn_conservative=False,
        alpha_cahn_conservative_mode="eliminate",
        # Substrate
        D_S=float(args.D_S),
        substrate_reaction_scheme="implicit",
        substrate_diffusion_scheme="implicit",
        mu_max=float(mu_max),
        K_S=float(kin.K0),
        k_g=0.0,
        k_d=0.0,
        Y=float(kin.Y_xO),
        rho_s_star=float(kin.rho_x),
        k_det=0.0,
        s_v=s_v,
        ds_v=ds_v,
        D_det_prev=Constant(0.0),
        # vS extension (critical since vS is otherwise unconstrained in fluid)
        gamma_vS=float(args.gamma_vS),
        vS_extension_mode=str(args.vS_ext_mode),
        gamma_vS_pin=float(args.gamma_vS_pin),
    )

    # Boundary conditions
    # -------------------
    #
    # We follow Duddu (2007) Fig. 4 / Table I: substrate Dirichlet at the top,
    # no-penetration at the substratum, and no lateral flux (1D slab kinematics).
    bc_S_top = BoundaryCondition("S", "dirichlet", "top", _as_float_time(lambda x, y, t: float(kin.Sbar)))
    bc_S_top_homog = BoundaryCondition("S", "dirichlet", "top", (lambda x, y: 0.0))

    bc_vSy_bottom = BoundaryCondition("vS_y", "dirichlet", "bottom", _as_float_time(lambda x, y, t: 0.0))
    bc_vSy_bottom_homog = BoundaryCondition("vS_y", "dirichlet", "bottom", (lambda x, y: 0.0))
    bc_vSx_left = BoundaryCondition("vS_x", "dirichlet", "left", _as_float_time(lambda x, y, t: 0.0))
    bc_vSx_left_homog = BoundaryCondition("vS_x", "dirichlet", "left", (lambda x, y: 0.0))
    bc_vSx_right = BoundaryCondition("vS_x", "dirichlet", "right", _as_float_time(lambda x, y, t: 0.0))
    bc_vSx_right_homog = BoundaryCondition("vS_x", "dirichlet", "right", (lambda x, y: 0.0))

    # Pressure gauge (removes the nullspace; does not affect vS which depends on ∇p).
    bc_p_top = BoundaryCondition("p", "dirichlet", "top", _as_float_time(lambda x, y, t: 0.0))
    bc_p_top_homog = BoundaryCondition("p", "dirichlet", "top", (lambda x, y: 0.0))

    bcs_S = [bc_S_top]
    bcs_S_homog = [bc_S_top_homog]
    bcs_pvS = [bc_vSy_bottom, bc_vSx_left, bc_vSx_right, bc_p_top]
    bcs_pvS_homog = [bc_vSy_bottom_homog, bc_vSx_left_homog, bc_vSx_right_homog, bc_p_top_homog]

    newton_params = NewtonParameters(
        newton_tol=float(args.newton_tol),
        newton_rtol=0.0,
        max_newton_iter=int(args.max_it),
        ls_mode="dealii",
    )
    newton_params.line_search = True

    # ------------------------------------------------------------------
    # Duddu-style operator splitting:
    #   (1) solve substrate S (nonlinear diffusion-reaction),
    #   (2) solve (p, vS) given S (linear).
    # This is more robust than a monolithic (p,vS,S) solve in PDAS mode.
    # ------------------------------------------------------------------
    inactive_S = ["v_x", "v_y", "p", "vS_x", "vS_y", "u_x", "u_y", "phi", "alpha"]
    _set_inactive_fields(dh, inactive_S)
    newton_params_S = NewtonParameters(
        newton_tol=float(args.newton_tol),
        newton_rtol=0.0,
        max_newton_iter=int(args.max_it),
        # For the S-only VI solve, the semismooth line search can be overly strict
        # on the first active-set change; full-step PDAS is robust because S is
        # kept feasible by the box constraint S >= 0.
        line_search=False,
        ls_mode="dealii",
    )
    solver_S = PdasNewtonSolver(
        forms.r_substrate,
        forms.a_substrate,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs_S,
        bcs_homog=bcs_S_homog,
        vi_params=VIParameters(c=0.0, active_tol=0.0, project_initial_guess=True),
        newton_params=newton_params_S,
        lin_params=LinearSolverParameters(backend=str(args.linear_solver)),
        quad_order=int(qdeg),
        backend=str(args.backend),
    )
    solver_S.set_box_bounds(by_field={"S": (0.0, None)})

    inactive_pvS = ["v_x", "v_y", "u_x", "u_y", "phi", "alpha", "S"]
    _set_inactive_fields(dh, inactive_pvS)
    # Weakly pin p in the fluid region to mimic Duddu's Φ≡0 in Ω_f while keeping
    # the biofilm-side potential essentially unaffected.
    m_pow = int(max(1, int(args.gamma_p_out_power)))
    one_m_alpha = (-alpha_k) + Constant(1.0)
    w_p_out = one_m_alpha
    for _ in range(m_pow - 1):
        w_p_out = w_p_out * one_m_alpha
    gamma_p_out = Constant(float(args.gamma_p_out))
    r_p_out = gamma_p_out * w_p_out * p_k * q_test * dx_q
    a_p_out = gamma_p_out * w_p_out * dp * q_test * dx_q
    solver_pvS = NewtonSolver(
        forms.r_mass + forms.r_skeleton + r_p_out,
        forms.a_mass + forms.a_skeleton + a_p_out,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs_pvS,
        bcs_homog=bcs_pvS_homog,
        newton_params=NewtonParameters(
            newton_tol=float(args.newton_tol),
            max_newton_iter=int(args.max_it),
            # This block is linear once S is fixed; avoid wasting time in line-search.
            line_search=False,
            ls_mode="dealii",
        ),
        lin_params=LinearSolverParameters(backend=str(args.linear_solver)),
        quad_order=int(qdeg),
        backend=str(args.backend),
    )

    functions = [v_k, p_k, vS_k, u_k, phi_k, alpha_k, S_k]
    prev_functions = [v_n, p_n, vS_n, u_n, phi_n, alpha_n, S_n]

    # One big pseudo-step to a quasi-steady substrate.
    solver_S.solve_time_interval(
        functions=functions,
        prev_functions=prev_functions,
        aux_functions={"dt": dt_c},
        time_params=TimeStepperParameters(
            dt=float(args.dt_steady),
            final_time=float(args.dt_steady),
            max_steps=1,
            theta=1.0,
            t0=0.0,
            stop_on_steady=False,
            on_dt_change=(lambda new_dt: setattr(dt_c, "value", float(new_dt))),
        ),
    )
    S_min = float(np.min(S_k.nodal_values))
    S_max = float(np.max(S_k.nodal_values))
    print(f"[substrate] S_min={S_min:.3e}  S_max={S_max:.3e}")

    # Solve p,vS with S frozen at the converged substrate field.
    solver_pvS.solve_time_interval(
        functions=functions,
        prev_functions=prev_functions,
        aux_functions={"dt": dt_c},
        time_params=TimeStepperParameters(
            dt=float(args.dt_steady),
            final_time=float(args.dt_steady),
            max_steps=1,
            theta=1.0,
            t0=0.0,
            stop_on_steady=False,
            on_dt_change=(lambda new_dt: setattr(dt_c, "value", float(new_dt))),
        ),
    )

    # --- profiles (Fig.4) -------------------------------------------------
    _write_vertical_profiles(outdir=outdir, dh=dh, L=L, S=S_k, p=p_k, vS=vS_k, h_b=h_b)

    # --- speed diagnostics ----------------------------------------------------
    alpha_expr = alpha_k
    delta_expr = Constant(4.0) * alpha_expr * (Constant(1.0) - alpha_expr)

    # Integral estimate: F = ∫_Ω alpha * divU(S) / width.
    I_divU = assemble_scalar(dh, alpha_expr * divU_k * dx_q, backend=str(args.backend), quad_order=int(qdeg))
    F_int = float(I_divU) / max(float(L), 1.0e-16)

    # Interface-weighted average of vS_y (delta weight).
    vS_y = vS_k  # VectorFunction
    I0 = assemble_scalar(dh, delta_expr * dx_q, backend=str(args.backend), quad_order=int(qdeg))
    I1 = assemble_scalar(dh, (vS_y[1] * delta_expr) * dx_q, backend=str(args.backend), quad_order=int(qdeg))
    F_vS = float(I1) / float(I0) if float(I0) > 0.0 else float("nan")

    # Bias the interface average to the *biofilm side* (alpha≈1) to avoid
    # cancellations when vS is extended into the fluid with opposite sign.
    I0_b = assemble_scalar(dh, (alpha_expr * delta_expr) * dx_q, backend=str(args.backend), quad_order=int(qdeg))
    I1_b = assemble_scalar(dh, (alpha_expr * vS_y[1] * delta_expr) * dx_q, backend=str(args.backend), quad_order=int(qdeg))
    F_vS_biofilm = float(I1_b) / float(I0_b) if float(I0_b) > 0.0 else float("nan")

    summary = {
        "L_mm": float(L),
        "H_mm": float(H),
        "h_b_mm": float(h_b),
        "eps_alpha_mm": float(eps_alpha),
        "phi_b": float(phi_b),
        "D_S_mm2_per_day": float(args.D_S),
        "kappa_inv": float(args.kappa_inv),
        "mu_f": float(args.mu_f),
        "mu_max_1_per_day": float(mu_max),
        "K_S": float(kin.K0),
        "Y": float(kin.Y_xO),
        "rho_s_star": float(kin.rho_x),
        "F_int_mm_per_day": float(F_int),
        "F_vS_mm_per_day": float(F_vS),
        "F_vS_biofilm_mm_per_day": float(F_vS_biofilm),
        "S_min": float(S_min),
        "S_max": float(S_max),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"- Wrote {outdir/'summary.json'}")
    print(f"[speed] F_int={F_int:.6g} mm/day  F_vS={F_vS:.6g} mm/day")


if __name__ == "__main__":
    main()
