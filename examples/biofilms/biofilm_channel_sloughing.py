import argparse
import os

import logging
import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
)
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dS, ds, dx
from pycutfem.utils.biofilm_adhesion import (
    assemble_scalar,
    update_adhesion_integrity,
    update_adhesion_integrity_field_on_boundary,
    wall_shear_rms_on_boundary,
)
from pycutfem.utils.biofilm_one_domain import build_biofilm_one_domain_forms
from pycutfem.utils.meshgen import structured_quad


def _tag_rectangle_boundaries(mesh: Mesh, *, L: float, H: float, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - float(L)) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - float(H)) <= tol,
        }
    )


def _smooth_step(z):
    # 0.5*(1+tanh(z)) is a robust sigmoid.
    return 0.5 * (1.0 + np.tanh(z))


def main() -> None:
    ap = argparse.ArgumentParser(description="Channel flow with an immersed diffuse-interface biofilm block + wall adhesion degradation.")
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ny", type=int, default=16)
    ap.add_argument("--L", type=float, default=4.0)
    ap.add_argument("--H", type=float, default=1.0)
    ap.add_argument("--q", type=int, default=6, help="Quadrature order (dx/dS metadata + NewtonSolver quad_order).")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--t-final", type=float, default=1.0)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--newton-tol", type=float, default=1.0e-5)
    ap.add_argument("--max-it", type=int, default=40)
    ap.add_argument(
        "--ls-mode",
        type=str,
        default="dealii",
        choices=("armijo", "dealii"),
        help="Newton line-search mode. 'dealii' is often more robust for the sloughing run.",
    )
    ap.add_argument("--outdir", type=str, default="examples/biofilms/results/channel_sloughing")
    ap.add_argument("--vtk-every", type=int, default=1, help="Write VTK every N accepted steps (0 disables).")
    # Initial biofilm block geometry (smooth)
    ap.add_argument("--x1", type=float, default=1.2)
    ap.add_argument("--x2", type=float, default=2.8)
    ap.add_argument("--h-biofilm", type=float, default=0.35)
    ap.add_argument("--eps", type=float, default=0.05, help="Diffuse-interface thickness for alpha initialization.")
    ap.add_argument("--phi-b", type=float, default=0.3)
    # Flow driving
    ap.add_argument("--Umax", type=float, default=0.3)
    ap.add_argument("--Tramp", type=float, default=0.1, help="Inflow ramp time (seconds).")
    ap.add_argument(
        "--process",
        type=str,
        default="both",
        choices=("erosion", "sloughing", "both"),
        help="Enable erosion (alpha sink + X source), sloughing (wall adhesion degradation), or both.",
    )
    # Adhesion model
    ap.add_argument("--a0", type=float, default=1.0)
    ap.add_argument(
        "--adhesion-integrity",
        type=str,
        default="scalar",
        choices=("scalar", "spatial"),
        help="Use a scalar a(t) (legacy) or a spatial wall field a(s) (mass-lumped update).",
    )
    ap.add_argument(
        "--a-perturb",
        type=float,
        default=0.0,
        help="Optional sinusoidal perturbation amplitude for initial wall a(s) in spatial mode (seeds localized sloughing).",
    )
    ap.add_argument("--a-perturb-k", type=int, default=1, help="Number of sine waves across [x1,x2] for --a-perturb.")
    ap.add_argument("--k-n", type=float, default=50.0, help="Normal spring stiffness [Pa/m].")
    ap.add_argument("--k-t", type=float, default=10.0, help="Tangential spring stiffness [Pa/m].")
    ap.add_argument("--gamma-n", type=float, default=5.0, help="Normal dashpot [Pa*s/m].")
    ap.add_argument("--gamma-t", type=float, default=1.0, help="Tangential dashpot [Pa*s/m].")
    ap.add_argument("--k-break", type=float, default=2.0, help="Adhesion degradation rate [1/s].")
    ap.add_argument("--tau-c", type=float, default=0.2, help="Critical wall shear stress [Pa].")
    ap.add_argument("--m-break", type=float, default=1.0, help="Shear exponent m>=1.")
    # Material / model parameters (kept simple)
    ap.add_argument("--rho-f", type=float, default=1.0)
    ap.add_argument("--mu-f", type=float, default=0.1)
    ap.add_argument("--kappa-inv", type=float, default=10.0)
    ap.add_argument(
        "--kappa-inv-model",
        type=str,
        default="spatial",
        choices=("spatial", "kozeny", "kozeny_carman", "kc"),
        help="Inverse permeability model. 'spatial' uses a constant kappa_inv; 'kozeny_carman' scales it with phi.",
    )
    ap.add_argument(
        "--kappa-phi-ref",
        type=float,
        default=None,
        help="Reference porosity for Kozeny–Carman normalization (k^{-1}(phi_ref)=kappa_inv). Defaults to --phi-b.",
    )
    ap.add_argument("--mu-s", type=float, default=0.5)
    ap.add_argument("--lambda-s", type=float, default=0.5)
    # Solid inertia: for sloughing/both runs we typically want inertia enabled to allow chunk motion.
    # Use a tri-state default (None) so we can enable it automatically for sloughing unless the user overrides.
    ap.add_argument(
        "--solid-inertia",
        dest="solid_inertia",
        action="store_true",
        default=None,
        help="Enable Eulerian skeleton inertia term in the u-equation (conservative form).",
    )
    ap.add_argument(
        "--no-solid-inertia",
        dest="solid_inertia",
        action="store_false",
        default=None,
        help="Disable skeleton inertia term (overrides the sloughing default).",
    )
    ap.add_argument(
        "--rho-s0",
        type=float,
        default=None,
        help="Skeleton inertia coefficient rho_s0_tilde (used if inertia is enabled).",
    )
    ap.add_argument(
        "--u-predictor",
        type=str,
        default="auto",
        choices=("auto", "copy", "extrapolate"),
        help="Initial guess strategy for u at each time step (auto: extrapolate when --solid-inertia, else copy).",
    )
    ap.add_argument(
        "--gamma-u",
        type=float,
        default=5.0,
        help=(
            "Extension penalty factor for u in the fluid. "
            "Use with --u-extension l2 (gamma_u/h^2 * (1-alpha) u) or --u-extension grad "
            "(gamma_u * (1-alpha) |grad u|^2)."
        ),
    )
    ap.add_argument(
        "--u-extension",
        type=str,
        default="l2",
        choices=("l2", "grad"),
        help="Type of u-extension stabilization to use outside the biofilm.",
    )
    ap.add_argument(
        "--gamma-u-pin",
        type=float,
        default=1.0e-4,
        help="Tiny L2 pinning coefficient used only with --u-extension grad (removes global-translation nullspace). Set 0 to disable.",
    )
    # Transport regularization (tune for interface sharpness vs robustness)
    ap.add_argument("--D-phi", type=float, default=0.0, help="Porosity diffusion (recommend ~0 for channel sloughing).")
    ap.add_argument("--gamma-phi", type=float, default=5.0, help="Penalty enforcing phi->1 in free fluid.")
    ap.add_argument(
        "--phi-supg",
        type=float,
        default=0.0,
        help="SUPG stabilization factor for the phi advection equation (0 disables). Recommended when --D-phi 0.",
    )
    ap.add_argument(
        "--phi-cip",
        type=float,
        default=0.0,
        help="CIP stabilization factor for phi (jump of normal gradient across interior facets; 0 disables). Recommended when --D-phi 0.",
    )
    ap.add_argument("--D-alpha", type=float, default=0.001, help="Indicator diffusion (interface regularization).")
    # Phase-field / crack options for alpha
    ap.add_argument(
        "--alpha-cahn-M",
        type=float,
        default=0.0,
        help="Allen–Cahn mobility M_alpha (0 disables phase-field regularization).",
    )
    ap.add_argument(
        "--alpha-cahn-gamma",
        type=float,
        default=0.0,
        help="Allen–Cahn surface-energy coefficient gamma_alpha (0 disables phase-field regularization).",
    )
    ap.add_argument(
        "--alpha-cahn-eps",
        type=float,
        default=None,
        help="Phase-field interface thickness epsilon for Allen–Cahn/crack terms (defaults to --eps).",
    )
    ap.add_argument(
        "--k-crack",
        type=float,
        default=0.0,
        help="Crack propagation coefficient k_c (0 disables crack term).",
    )
    ap.add_argument(
        "--D-crack",
        type=float,
        default=0.0,
        help="Crack threshold D_c (mechanical driver must exceed this).",
    )
    ap.add_argument("--m-crack", type=float, default=1.0, help="Crack exponent m>=1.")
    ap.add_argument(
        "--gamma-kappa",
        type=float,
        default=0.0,
        help="Curvature resistance coefficient gamma_kappa in the crack driver D_mech - gamma_kappa*kappa - D_c.",
    )
    ap.add_argument("--eta-kappa", type=float, default=1.0e-12, help="Curvature regularization eta_kappa.")
    ap.add_argument("--eta-pos", type=float, default=1.0e-12, help="Positive-part regularization eta for <x>_+.")
    ap.add_argument("--eta-mech", type=float, default=1.0e-12, help="Mechanical driver regularization inside sqrt.")
    ap.add_argument(
        "--crack-driver",
        type=str,
        default="shear",
        choices=("shear", "solid_strain", "drag"),
        help=(
            "Mechanical driver used for crack speed. "
            "'shear' uses a fluid shear-stress proxy 2*mu*||eps(v)||, "
            "'solid_strain' uses ||eps(u)||, and 'drag' is an alias for 'shear' "
            "(current backend limitation prevents a direct |beta(v-vS)| norm)."
        ),
    )

    # Initial crack geometry (optional)
    ap.add_argument("--crack-depth", type=float, default=0.0, help="Initial crack depth from the bottom wall (0 disables).")
    ap.add_argument("--crack-width", type=float, default=0.05, help="Initial crack width in x.")
    ap.add_argument(
        "--crack-x0",
        type=float,
        default=None,
        help="Initial crack center x-position (defaults to midpoint (x1+x2)/2).",
    )
    ap.add_argument(
        "--fix-base",
        action="store_true",
        help="Clamp the skeleton displacement u=0 on the bottom wall (static base) to study crack-driven detachment.",
    )
    ap.add_argument(
        "--alpha-supg",
        type=float,
        default=0.0,
        help="SUPG stabilization factor for the alpha advection equation (0 disables). Recommended when --D-alpha 0.",
    )
    ap.add_argument(
        "--alpha-cip",
        type=float,
        default=0.0,
        help="CIP stabilization factor for alpha (jump of normal gradient across interior facets; 0 disables). Recommended when --D-alpha 0.",
    )
    ap.add_argument(
        "--u-cip",
        type=float,
        default=0.0,
        help="CIP stabilization factor for skeleton displacement u (jump of normal gradient across interior facets; 0 disables). Recommended with --solid-inertia.",
    )
    ap.add_argument("--D-X", type=float, default=0.001, help="Detached biomass diffusion.")
    # Erosion / detachment (produces X and removes alpha at the diffuse interface)
    ap.add_argument("--k-det", type=float, default=1.0, help="Detachment strength for D_det=k_det*||eps(v^n)|| (lagged).")
    ap.add_argument("--rho-s-star", type=float, default=1.0, help="Intrinsic solid density scale used in the X source.")
    ap.add_argument("--mass-every", type=int, default=10, help="Print scalar detachment diagnostics (int_X) every N steps (0 disables).")
    ap.add_argument(
        "--no-clip",
        action="store_true",
        help="Disable post-step clipping of (alpha,phi,S) to physical bounds. "
        "Clipping keeps alpha in [0,1], phi in [0,1] and S>=0 to avoid non-physical coefficients.",
    )
    # Simple sloughing onset indicators (diagnostics only)
    ap.add_argument(
        "--slough-a-thresh",
        type=float,
        default=0.4,
        help="Print a sloughing indicator once when a_min drops below this threshold (spatial adhesion mode).",
    )
    ap.add_argument(
        "--slough-contact-rel-thresh",
        type=float,
        default=0.97,
        help="Print a sloughing indicator once when alpha wall-contact measure (alpha_area) drops below this fraction of its initial value.",
    )
    ap.add_argument(
        "--slough-liftoff-dy",
        type=float,
        default=0.02,
        help="Print a sloughing indicator once when the high-alpha center-of-mass y increases by this amount from its initial value.",
    )
    ap.add_argument(
        "--slough-vSx-thresh",
        type=float,
        default=1.0e-2,
        help="Print a sloughing indicator once when |mean(vSx)| over alpha>0.5 exceeds this threshold.",
    )
    args = ap.parse_args()

    # Silence verbose assembly logs from scalar post-processing (assemble_scalar).
    logging.getLogger("pycutfem.ufl.compilers").setLevel(logging.WARNING)

    L = float(args.L)
    H = float(args.H)
    qdeg = int(args.q)
    dt_val = float(args.dt)
    theta = float(args.theta)
    backend = str(args.backend)
    process = str(getattr(args, "process", "both")).strip().lower()
    use_erosion = process in {"erosion", "both"}
    use_sloughing = process in {"sloughing", "both"}
    adhesion_integrity = str(getattr(args, "adhesion_integrity", "scalar")).strip().lower()
    use_spatial_adhesion = bool(use_sloughing and adhesion_integrity == "spatial")

    # Kozeny–Carman permeability reference porosity
    kappa_inv_model = str(getattr(args, "kappa_inv_model", "spatial")).strip().lower()
    if getattr(args, "kappa_phi_ref", None) is None and kappa_inv_model in {"kozeny", "kozeny_carman", "kc"}:
        args.kappa_phi_ref = float(args.phi_b)

    # Alpha phase-field thickness defaults to the same smoothing length used to build alpha0.
    if getattr(args, "alpha_cahn_eps", None) is None:
        args.alpha_cahn_eps = float(args.eps)

    if getattr(args, "crack_x0", None) is None:
        args.crack_x0 = 0.5 * (float(args.x1) + float(args.x2))

    # Default inertia for sloughing/both runs unless the user explicitly disables it.
    if getattr(args, "solid_inertia", None) is None:
        args.solid_inertia = bool(use_sloughing)
        if bool(args.solid_inertia):
            print("[info] sloughing mode detected; enabling --solid-inertia by default (use --no-solid-inertia to disable).")

    if getattr(args, "rho_s0", None) is None:
        args.rho_s0 = 1.0 if bool(getattr(args, "solid_inertia", False)) else 0.0

    os.makedirs(str(args.outdir), exist_ok=True)

    alpha_supg = float(getattr(args, "alpha_supg", 0.0) or 0.0)
    alpha_cip = float(getattr(args, "alpha_cip", 0.0) or 0.0)
    if float(args.D_alpha) == 0.0 and alpha_supg == 0.0 and alpha_cip == 0.0:
        # When users set D-alpha=0, the alpha equation becomes pure CG advection (by vS),
        # which is prone to spurious oscillations/overshoots that can destabilize Newton.
        # Prefer consistent stabilization (SUPG/CIP) over adding physical diffusion.
        alpha_supg = 1.0
        alpha_cip = 10.0
        print(
            "[info] --D-alpha 0 detected with no alpha stabilization specified; enabling "
            "default consistent stabilization: --alpha-supg 1 --alpha-cip 10."
        )

    phi_supg = float(getattr(args, "phi_supg", 0.0) or 0.0)
    phi_cip = float(getattr(args, "phi_cip", 0.0) or 0.0)
    if float(args.D_phi) == 0.0 and phi_supg == 0.0 and phi_cip == 0.0:
        print(
            "[warn] --D-phi 0 detected with no phi stabilization specified. "
            "If you see oscillations in phi (and therefore in beta), consider adding "
            "--phi-supg 1 and/or --phi-cip 10."
        )

    solid_inertia = bool(getattr(args, "solid_inertia", False))
    u_predictor = str(getattr(args, "u_predictor", "auto")).strip().lower()
    if u_predictor == "auto":
        u_predictor = "extrapolate" if solid_inertia else "copy"

    u_cip = float(getattr(args, "u_cip", 0.0) or 0.0)
    if u_cip == 0.0 and solid_inertia and float(args.gamma_u) <= 1.0:
        u_cip = 1.0
        print("[info] --solid-inertia with small --gamma-u detected; enabling default u facet stabilization: --u-cip 1.")

    if float(args.D_alpha) == 0.0 and str(getattr(args, "u_extension", "l2")).strip().lower() in {"l2"} and float(args.gamma_u) < 1.0 and solid_inertia:
        print(
            "[warn] You are using --D-alpha 0 with --solid-inertia and a small --gamma-u. "
            "This combination often makes the u-block near-singular outside the biofilm and can "
            "cause Newton stagnation. Try --gamma-u 1 (or keep the default 5), add --u-cip 1, or add mild "
            "interface regularization like --D-alpha 1e-3."
        )

    # ------------------------------------------------------------------
    # Mesh + boundary tags
    # ------------------------------------------------------------------
    nodes, elems, _, corners = structured_quad(L, H, nx=int(args.nx), ny=int(args.ny), poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    _tag_rectangle_boundaries(mesh, L=L, H=H)

    # ------------------------------------------------------------------
    # Mixed space (v,p,u,phi,alpha,S)
    # ------------------------------------------------------------------
    field_specs = {
        "v_x": 2,
        "v_y": 2,
        "p": 1,
        "u_x": 2,
        "u_y": 2,
        "phi": 1,
        "alpha": 1,
        "S": 1,
        "X": 1,
    }
    if use_spatial_adhesion:
        field_specs["a"] = 1

    me = MixedElement(mesh, field_specs=field_specs)
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    dv = VectorTrialFunction(space=V, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    dphi = TrialFunction("phi", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    dS_trial = TrialFunction("S", dof_handler=dh)
    dX_trial = TrialFunction("X", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    S_test = TestFunction("S", dof_handler=dh)
    X_test = TestFunction("X", dof_handler=dh)

    # Unknowns (k) and previous state (n)
    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    S_k = Function("S_k", "S", dof_handler=dh)
    X_k = Function("X_k", "X", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    # Keep a copy of u at the previous accepted step for diagnostics (the solver
    # promotes prev←current before calling the time-loop callback).
    u_prev = VectorFunction("u_prev", ["u_x", "u_y"], dof_handler=dh)
    u_nm1 = None
    if bool(getattr(args, "solid_inertia", False)):
        u_nm1 = VectorFunction("u_nm1", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    S_n = Function("S_n", "S", dof_handler=dh)
    X_n = Function("X_n", "X", dof_handler=dh)
    if use_spatial_adhesion:
        a_prev = Function("a_prev", "a", dof_handler=dh)
        # Do not solve for a: mark its DOFs inactive in Newton.
        tags = getattr(dh, "dof_tags", None) or {}
        inactive = set(tags.get("inactive", set()))
        inactive.update(int(i) for i in np.asarray(dh.get_field_slice("a"), dtype=int).ravel())
        tags["inactive"] = inactive
        dh.dof_tags = tags
    else:
        a_prev = None

    # ------------------------------------------------------------------
    # Initial biofilm block (smooth indicator)
    # ------------------------------------------------------------------
    x1 = float(args.x1)
    x2 = float(args.x2)
    h_b = float(args.h_biofilm)
    eps = float(args.eps)
    phi_b = float(args.phi_b)
    crack_depth = float(getattr(args, "crack_depth", 0.0) or 0.0)
    crack_width = float(getattr(args, "crack_width", 0.0) or 0.0)
    crack_x0 = float(getattr(args, "crack_x0", 0.5 * (x1 + x2)))

    eps_x = max(eps, 1.0e-12)
    eps_y = max(eps, 1.0e-12)

    def alpha0(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        wx = _smooth_step((x - x1) / eps_x) * _smooth_step((x2 - x) / eps_x)
        wy = _smooth_step((h_b - y) / eps_y)
        a0 = wx * wy
        # Optional initial crack notch (fluid gap) from the bottom wall into the biofilm.
        # This seeds an internal diffuse interface so the crack-speed term can propagate it.
        if crack_depth > 0.0 and crack_width > 0.0:
            xL = crack_x0 - 0.5 * crack_width
            xR = crack_x0 + 0.5 * crack_width
            wcx = _smooth_step((x - xL) / eps_x) * _smooth_step((xR - x) / eps_x)
            wcy = _smooth_step((crack_depth - y) / eps_y)
            a0 = a0 * (1.0 - wcx * wcy)
        return np.clip(a0, 0.0, 1.0)

    alpha_n.set_values_from_function(lambda x, y: float(alpha0(x, y)))
    phi_n.set_values_from_function(lambda x, y: float(1.0 - (1.0 - phi_b) * alpha0(x, y)))
    S_n.set_values_from_function(lambda x, y: 0.0)
    X_n.set_values_from_function(lambda x, y: 0.0)
    v_n.set_values_from_function(lambda x, y: np.array([0.0, 0.0], dtype=float))
    u_n.set_values_from_function(lambda x, y: np.array([0.0, 0.0], dtype=float))
    if u_nm1 is not None:
        u_nm1.set_values_from_function(lambda x, y: np.array([0.0, 0.0], dtype=float))
    p_n.set_values_from_function(lambda x, y: 0.0)
    if a_prev is not None:
        a0 = float(args.a0)
        a_pert = float(getattr(args, "a_perturb", 0.0) or 0.0)
        a_pert_k = int(getattr(args, "a_perturb_k", 1) or 1)
        Lb = max(1.0e-12, float(x2 - x1))

        def a_init(x, y):
            # Base value everywhere; optional perturbation localized to the biofilm x-range.
            x = float(x)
            mask = float(alpha0(x, 0.0))  # smooth mask for [x1,x2]
            if a_pert != 0.0 and mask > 1.0e-12:
                phase = 2.0 * np.pi * float(a_pert_k) * (x - float(x1)) / Lb
                return float(np.clip(a0 * (1.0 - a_pert * mask * np.sin(phase)), 0.0, 1.0))
            return float(np.clip(a0, 0.0, 1.0))

        a_prev.set_values_from_function(a_init)

    # ------------------------------------------------------------------
    # Forms with adhesion traction on the bottom wall
    # ------------------------------------------------------------------
    dt_c = Constant(dt_val)
    a_c = Constant(float(args.a0))

    ds_bottom = dS(defined_on=mesh.edge_bitset("bottom"), metadata={"q": int(qdeg)})
    ds_int = ds(metadata={"q": int(qdeg)})
    dx_q = dx(metadata={"q": int(qdeg)})

    rho_f_c = Constant(float(args.rho_f))
    mu_f_c = Constant(float(args.mu_f))
    kappa_inv_c = Constant(float(args.kappa_inv))
    mu_s_c = Constant(float(args.mu_s))
    lambda_s_c = Constant(float(args.lambda_s))
    rho_s0_c = Constant(float(getattr(args, "rho_s0", 0.0)))

    forms = build_biofilm_one_domain_forms(
        v_k=v_k,
        p_k=p_k,
        u_k=u_k,
        phi_k=phi_k,
        alpha_k=alpha_k,
        S_k=S_k,
        v_n=v_n,
        p_n=p_n,
        u_n=u_n,
        u_nm1=u_nm1,
        phi_n=phi_n,
        alpha_n=alpha_n,
        S_n=S_n,
        dv=dv,
        dp=dp,
        du=du,
        dphi=dphi,
        dalpha=dalpha,
        dS=dS_trial,
        v_test=v_test,
        q_test=q_test,
        u_test=u_test,
        phi_test=phi_test,
        alpha_test=alpha_test,
        S_test=S_test,
        X_test=X_test,
        dx=dx(metadata={"q": int(qdeg)}),
        dt=dt_c,
        theta=theta,
        rho_f=rho_f_c,
        mu_f=mu_f_c,
        kappa_inv=kappa_inv_c,
        kappa_inv_model=str(getattr(args, "kappa_inv_model", "spatial")),
        kappa_inv_phi_ref=float(getattr(args, "kappa_phi_ref", args.phi_b)),
        mu_s=mu_s_c,
        lambda_s=lambda_s_c,
        rho_s0_tilde=rho_s0_c,
        include_skeleton_acceleration=bool(getattr(args, "solid_inertia", False)),
        gamma_u=float(args.gamma_u),
        u_extension_mode=str(getattr(args, "u_extension", "l2")),
        gamma_u_pin=float(getattr(args, "gamma_u_pin", 0.0)),
        # Mild diffusion/stabilization to keep transport variables well-posed in the free-fluid region.
        D_phi=float(args.D_phi),
        gamma_phi=float(args.gamma_phi),
        phi_supg=float(phi_supg),
        phi_cip=float(phi_cip),
        D_alpha=float(args.D_alpha),
        alpha_cahn_M=float(getattr(args, "alpha_cahn_M", 0.0)),
        alpha_cahn_gamma=float(getattr(args, "alpha_cahn_gamma", 0.0)),
        alpha_cahn_eps=float(getattr(args, "alpha_cahn_eps", float(args.eps))),
        alpha_crack_k=float(getattr(args, "k_crack", 0.0)),
        alpha_crack_Dc=float(getattr(args, "D_crack", 0.0)),
        alpha_crack_m=float(getattr(args, "m_crack", 1.0)),
        alpha_crack_gamma_kappa=float(getattr(args, "gamma_kappa", 0.0)),
        alpha_crack_eta_kappa=float(getattr(args, "eta_kappa", 1.0e-12)),
        alpha_crack_eta_pos=float(getattr(args, "eta_pos", 1.0e-12)),
        alpha_crack_eta_mech=float(getattr(args, "eta_mech", 1.0e-12)),
        alpha_crack_driver=str(getattr(args, "crack_driver", "drag")),
        alpha_supg=float(alpha_supg),
        alpha_cip=float(alpha_cip),
        u_cip=float(u_cip),
        ds_cip=ds_int,
        D_S=0.01,
        D_X=float(args.D_X),
        rho_s_star=float(args.rho_s_star),
        # Disable growth here; detachment/erosion is controlled separately via k_det.
        mu_max=0.0,
        k_g=0.0,
        k_d=0.0,
        k_det=float(args.k_det) if use_erosion else 0.0,
        X_k=X_k,
        X_n=X_n,
        dX=dX_trial,
        ds_adh=ds_bottom if use_sloughing else None,
        adhesion_k_n=float(args.k_n) if use_sloughing else 0.0,
        adhesion_k_t=float(args.k_t) if use_sloughing else 0.0,
        adhesion_gamma_n=float(args.gamma_n) if use_sloughing else 0.0,
        adhesion_gamma_t=float(args.gamma_t) if use_sloughing else 0.0,
        adhesion_a_prev=a_prev if a_prev is not None else a_c,
    )

    # ------------------------------------------------------------------
    # Dirichlet BCs for a simple channel profile (imposed at left/right)
    # ------------------------------------------------------------------
    Umax = float(args.Umax)
    Tramp = max(1.0e-12, float(args.Tramp))

    def ramp(t):
        return 1.0 - float(np.exp(-float(t) / Tramp))

    def inflow_vx(x, y, t):
        yy = float(y) / H
        return float(Umax * ramp(t) * 4.0 * yy * (1.0 - yy))

    bcs = []
    # Inflow: prescribe velocity profile on the left boundary.
    bcs.append(BoundaryCondition("v_x", "dirichlet", "left", inflow_vx))
    bcs.append(BoundaryCondition("v_y", "dirichlet", "left", lambda x, y, t: 0.0))
    for tag in ("bottom", "top"):
        bcs.append(BoundaryCondition("v_x", "dirichlet", tag, lambda x, y, t: 0.0))
        bcs.append(BoundaryCondition("v_y", "dirichlet", tag, lambda x, y, t: 0.0))

    # Optional static base for crack-propagation studies: clamp skeleton displacement on the bottom wall.
    if bool(getattr(args, "fix_base", False)):
        bcs.append(BoundaryCondition("u_x", "dirichlet", "bottom", lambda x, y, t: 0.0))
        bcs.append(BoundaryCondition("u_y", "dirichlet", "bottom", lambda x, y, t: 0.0))
    # Outlet: pin the pressure to remove the nullspace (velocity is left free -> natural traction).
    bcs.append(BoundaryCondition("p", "dirichlet", "right", lambda x, y, t: 0.0))

    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, lambda x, y: 0.0) for b in bcs]

    # ------------------------------------------------------------------
    # Post-step callback: compute wall shear, update adhesion a, write output
    # ------------------------------------------------------------------
    step_counter = {"k": 0}
    a_state = {"val": float(args.a0)}
    alpha_area0 = {"val": None}
    slough_flags = {"a_drop": False, "contact_loss": False, "liftoff": False, "motion": False}
    slough_t = {"a_drop": None, "contact_loss": None, "liftoff": None, "motion": None}
    alpha_cm_hi0 = {"y": None}
    num_nodes = len(mesh.nodes_list)
    alpha_dof_xy = np.asarray(dh.get_dof_coords("alpha"), dtype=float)

    def _scalar_to_nodes(f: Function) -> np.ndarray:
        out = np.zeros(num_nodes, dtype=float)
        for gdof, lidx in f._g2l.items():
            _field, node_id = dh._dof_to_node_map[gdof]
            if node_id is None:
                continue
            out[int(node_id)] = float(f.nodal_values[lidx])
        return out

    def _vector_to_nodes(vf: VectorFunction) -> np.ndarray:
        out = np.zeros((num_nodes, 2), dtype=float)
        field_names = list(vf.field_names)
        for gdof, lidx in vf._g2l.items():
            field, node_id = dh._dof_to_node_map[gdof]
            if node_id is None or field not in field_names:
                continue
            out[int(node_id), field_names.index(field)] = float(vf.nodal_values[lidx])
        return out

    def post_step(functions):
        step_counter["k"] += 1
        step_no = int(step_counter["k"])
        t_now = step_no * dt_val

        shear = wall_shear_rms_on_boundary(
            dof_handler=dh,
            v=v_k,
            alpha=alpha_k,
            phi=phi_k,
            ds_wall=ds_bottom,
            mu_f=mu_f_c,
            backend=backend,
            quad_order=qdeg,
        )
        a_msg = ""
        a_min_val = float(args.a0)
        a_max_val = float(args.a0)
        if use_sloughing:
            if a_prev is None:
                # Legacy scalar a(t) update using RMS shear.
                a_new = update_adhesion_integrity(
                    a_n=a_state["val"],
                    dt=dt_val,
                    tau_rms=shear.tau_rms,
                    k_break=float(args.k_break),
                    tau_c=float(args.tau_c),
                    m=float(args.m_break),
                )
                a_state["val"] = float(a_new)
                a_c.value = float(a_new)
                a_msg = f"a={a_state['val']:.6f}"
                a_min_val = float(a_state["val"])
                a_max_val = float(a_state["val"])
            else:
                upd = update_adhesion_integrity_field_on_boundary(
                    dof_handler=dh,
                    a_field=a_prev,
                    dt=dt_val,
                    v=v_k,
                    alpha=alpha_k,
                    phi=phi_k,
                    ds_wall=ds_bottom,
                    mu_f=mu_f_c,
                    k_break=float(args.k_break),
                    tau_c=float(args.tau_c),
                    m=float(args.m_break),
                    backend=backend,
                    quad_order=qdeg,
                )
                a_msg = f"a[min,max]=[{upd.a_min:.3f},{upd.a_max:.3f}]"
                a_min_val = float(upd.a_min)
                a_max_val = float(upd.a_max)

        if alpha_area0["val"] is None:
            alpha_area0["val"] = float(shear.alpha_area)

        print(
            f"[step {step_no:04d}] t={t_now:.3f}  tau_rms={shear.tau_rms:.3e}  "
            f"{a_msg}  (alpha_area={shear.alpha_area:.3e}, rel={float(shear.alpha_area)/(alpha_area0['val']+1e-16):.3f})"
        )
        # Detached biomass tracking
        if use_erosion and float(args.k_det) != 0.0:
            try:
                X_max = float(np.max(X_k.nodal_values))
                mass_every = int(getattr(args, "mass_every", 0) or 0)
                if mass_every > 0 and (step_no % mass_every == 0):
                    X_mass = assemble_scalar(dh, X_k * dx_q, backend=backend, quad_order=qdeg)
                    print(f"           X[max]={X_max:.3e}  int_X={X_mass:.3e}")
                else:
                    print(f"           X[max]={X_max:.3e}")
            except Exception:
                pass
        try:
            a_min = float(np.min(alpha_k.nodal_values))
            a_max = float(np.max(alpha_k.nodal_values))
            p_min = float(np.min(phi_k.nodal_values))
            p_max = float(np.max(phi_k.nodal_values))
            print(f"           alpha[min,max]=[{a_min:.3e},{a_max:.3e}]  phi[min,max]=[{p_min:.3e},{p_max:.3e}]")

            # Lightweight nodal diagnostics (helps interpret "no Darcy effect / wrong motion")
            alpha_nodes = _scalar_to_nodes(alpha_k)
            phi_nodes = _scalar_to_nodes(phi_k)
            beta_nodes = alpha_nodes * float(args.mu_f) * (phi_nodes * phi_nodes) * float(args.kappa_inv)
            bmin = float(beta_nodes.min())
            bmax = float(beta_nodes.max())

            mask = alpha_nodes > 0.5
            if np.any(mask):
                alpha_vals = np.asarray(alpha_k.nodal_values, dtype=float)
                alpha_vals_pos = np.maximum(alpha_vals, 0.0)
                y_cm_hi_curr = None
                wsum_all = float(np.sum(alpha_vals_pos))
                if wsum_all > 0.0:
                    x_cm_all = float(np.sum(alpha_vals_pos * alpha_dof_xy[:, 0]) / wsum_all)
                    y_cm_all = float(np.sum(alpha_vals_pos * alpha_dof_xy[:, 1]) / wsum_all)
                    cm_msg = f"  alpha_cm=({x_cm_all:.5f},{y_cm_all:.5f})"

                    mask_dofs = alpha_vals_pos > 0.5
                    if np.any(mask_dofs):
                        w_hi = alpha_vals_pos[mask_dofs]
                        wsum_hi = float(np.sum(w_hi))
                        if wsum_hi > 0.0:
                            xy_hi = alpha_dof_xy[mask_dofs, :]
                            x_cm_hi = float(np.sum(w_hi * xy_hi[:, 0]) / wsum_hi)
                            y_cm_hi = float(np.sum(w_hi * xy_hi[:, 1]) / wsum_hi)
                            y_cm_hi_curr = float(y_cm_hi)
                            cm_msg += f"  alpha_cm_hi=({x_cm_hi:.5f},{y_cm_hi:.5f})"
                            if alpha_cm_hi0["y"] is None:
                                alpha_cm_hi0["y"] = float(y_cm_hi)
                else:
                    cm_msg = ""
                v_nodes = _vector_to_nodes(v_k)
                u_nodes = _vector_to_nodes(u_k)
                u_prev_nodes = _vector_to_nodes(u_prev)
                vS_nodes = (u_nodes - u_prev_nodes) / float(dt_val)
                vx_mean = float(np.mean(v_nodes[mask, 0]))
                vSx_mean = float(np.mean(vS_nodes[mask, 0]))
                mask_int = (alpha_nodes > 0.4) & (alpha_nodes < 0.6)
                vSx_int = float(np.mean(vS_nodes[mask_int, 0])) if np.any(mask_int) else float("nan")
                print(
                    f"           beta[min,max]=[{bmin:.3e},{bmax:.3e}]  "
                    f"mean(vx|alpha>0.5)={vx_mean:.3e}  mean(vSx|alpha>0.5)={vSx_mean:.3e}  "
                    f"mean(vSx|0.4<alpha<0.6)={vSx_int:.3e}{cm_msg}"
                )

                # ----------------------------------------------------------
                # Sloughing onset indicators (best-effort diagnostics)
                # ----------------------------------------------------------
                if use_sloughing:
                    rel_contact = float(shear.alpha_area) / (float(alpha_area0["val"]) + 1.0e-16)
                    if (not slough_flags["a_drop"]) and (a_min_val < float(args.slough_a_thresh)):
                        slough_flags["a_drop"] = True
                        slough_t["a_drop"] = float(t_now)
                        print(
                            f"           [sloughing] adhesion weakened: a_min={a_min_val:.3f} < {float(args.slough_a_thresh):.3f} (t={t_now:.3f})"
                        )
                    if (not slough_flags["contact_loss"]) and (rel_contact < float(args.slough_contact_rel_thresh)):
                        slough_flags["contact_loss"] = True
                        slough_t["contact_loss"] = float(t_now)
                        print(
                            f"           [sloughing] wall contact loss: rel_contact={rel_contact:.3f} < {float(args.slough_contact_rel_thresh):.3f} (t={t_now:.3f})"
                        )
                    if (not slough_flags["motion"]) and (abs(vSx_mean) > float(args.slough_vSx_thresh)):
                        slough_flags["motion"] = True
                        slough_t["motion"] = float(t_now)
                        print(
                            f"           [sloughing] chunk motion: |mean(vSx)|={abs(vSx_mean):.3e} > {float(args.slough_vSx_thresh):.3e} (t={t_now:.3f})"
                        )
                    if (not slough_flags["liftoff"]) and (alpha_cm_hi0["y"] is not None) and (y_cm_hi_curr is not None):
                        y0 = float(alpha_cm_hi0["y"])
                        dy = float(y_cm_hi_curr) - y0
                        if dy > float(args.slough_liftoff_dy):
                            slough_flags["liftoff"] = True
                            slough_t["liftoff"] = float(t_now)
                            print(
                                f"           [sloughing] liftoff: Δy_cm_hi={dy:.3e} > {float(args.slough_liftoff_dy):.3e} (t={t_now:.3f})"
                            )
            else:
                print(f"           beta[min,max]=[{bmin:.3e},{bmax:.3e}]")
        except Exception:
            pass

        vtk_every = int(args.vtk_every)
        if vtk_every > 0 and (step_no % vtk_every == 0):
            # Export derived fields for debugging in ParaView.
            try:
                alpha_nodes = _scalar_to_nodes(alpha_k)
                phi_nodes = _scalar_to_nodes(phi_k)
                beta_nodes = alpha_nodes * float(args.mu_f) * (phi_nodes * phi_nodes) * float(args.kappa_inv)
                u_nodes = _vector_to_nodes(u_k)
                u_prev_nodes = _vector_to_nodes(u_prev)
                vS_nodes = (u_nodes - u_prev_nodes) / float(dt_val)
            except Exception:
                beta_nodes = None
                vS_nodes = None

            export_vtk(
                filename=os.path.join(str(args.outdir), f"solution_{step_no:04d}.vtu"),
                mesh=mesh,
                dof_handler=dh,
                functions={
                    "v": v_k,
                    "p": p_k,
                    "u": u_k,
                    "vS": vS_nodes if vS_nodes is not None else (lambda x, y: (0.0, 0.0)),
                    "phi": phi_k,
                    "alpha": alpha_k,
                    "a": a_prev if a_prev is not None else (lambda x, y: float(a_state["val"])),
                    "S": S_k,
                    "X": X_k,
                    "beta": beta_nodes if beta_nodes is not None else (lambda x, y: 0.0),
                },
            )

    # ------------------------------------------------------------------
    # Solve in time
    # ------------------------------------------------------------------
    solver = NewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(
            newton_tol=float(args.newton_tol),
            max_newton_iter=int(args.max_it),
            ls_mode=str(getattr(args, "ls_mode", "dealii")),
        ),
        quad_order=qdeg,
        backend=backend,
        postproc_timeloop_cb=post_step,
    )

    # Optional u predictor (applied once per step, before the first Newton assembly).
    _pred_state = {"step_no": None}

    def _preproc_predictor(_funcs):
        step_no = getattr(solver, "_current_step_no", None)
        if step_no is None or _pred_state["step_no"] == int(step_no):
            return
        _pred_state["step_no"] = int(step_no)
        if u_predictor == "extrapolate" and u_nm1 is not None:
            # Constant-velocity predictor: u^{n+1} ≈ u^n + (u^n - u^{n-1}).
            u_k.nodal_values[:] = u_n.nodal_values + (u_n.nodal_values - u_nm1.nodal_values)

    solver.pre_cb = _preproc_predictor

    def post_step_refiner(step, bcs_now, functions, prev_functions):
        # NOTE: Called after an accepted Newton step and **before** the solver promotes current -> previous.
        # Keep history needed for diagnostics and (optionally) skeleton inertia.
        u_prev.nodal_values[:] = u_n.nodal_values[:]
        if u_nm1 is not None:
            u_nm1.nodal_values[:] = u_n.nodal_values[:]

        if not bool(args.no_clip):
            # Keep alpha/phi bounded so blended coefficients (rho, mu, beta, delta_eps(alpha)) stay physical.
            alpha_k.nodal_values[:] = np.clip(alpha_k.nodal_values, 0.0, 1.0)
            phi_k.nodal_values[:] = np.clip(phi_k.nodal_values, 0.0, 1.0)
            S_k.nodal_values[:] = np.maximum(S_k.nodal_values, 0.0)
            X_k.nodal_values[:] = np.maximum(X_k.nodal_values, 0.0)
            if a_prev is not None:
                a_prev.nodal_values[:] = np.clip(a_prev.nodal_values, 0.0, 1.0)

    solver.solve_time_interval(
        functions=[v_k, p_k, u_k, phi_k, alpha_k, S_k, X_k],
        prev_functions=[v_n, p_n, u_n, phi_n, alpha_n, S_n, X_n],
        aux_functions={
            "dt": dt_c,
            **({"a_prev": a_prev} if a_prev is not None else {}),
            **({"u_nm1": u_nm1} if u_nm1 is not None else {}),
        },
        time_params=TimeStepperParameters(dt=dt_val, final_time=float(args.t_final), max_steps=10_000, theta=theta),
        post_step_refiner=post_step_refiner,
    )


if __name__ == "__main__":
    main()
