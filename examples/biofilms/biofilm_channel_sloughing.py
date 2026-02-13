import argparse
import os
import sys

import logging
import numpy as np
from pathlib import Path

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
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.measures import dS, ds, dx
from examples.utils.biofilm.adhesion import (
    assemble_scalar,
    solid_von_mises_mass_lumped_in_domain,
    update_adhesion_integrity,
    update_adhesion_integrity_field_on_boundary,
    update_adhesion_integrity_field_on_boundary_von_mises,
    wall_shear_rms_on_boundary,
)
from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms
from pycutfem.utils.meshgen import structured_quad
from examples.utils.shared.volume_correction import logit_shift_to_match_integral


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if raw is None:
        return bool(default)
    raw = str(raw).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "y", "on"}


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    with np.load(str(path)) as data:
        return {k: data[k] for k in data.files}


def _npz_scalar(npz: dict[str, np.ndarray], key: str, default: float) -> float:
    if key not in npz:
        return float(default)
    try:
        return float(np.asarray(npz[key]).reshape(()))
    except Exception:
        return float(default)


def _collect_function_arrays(funcs: list[object]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for f in funcs:
        if f is None:
            continue
        name = getattr(f, "name", None)
        if not name:
            continue
        try:
            vals = getattr(f, "nodal_values", None)
            if vals is None:
                continue
            out[str(name)] = np.asarray(vals).copy()
        except Exception:
            continue
    return out


def _restore_function_from_state(f: object, state: dict[str, np.ndarray]) -> bool:
    if f is None:
        return False
    name = getattr(f, "name", None)
    if not name:
        return False
    key = str(name)
    src = state.get(key, None)
    if src is None and key.endswith("_k"):
        src = state.get(key[:-2] + "_n", None)
    if src is None and key.endswith("_n"):
        src = state.get(key[:-2] + "_k", None)
    if src is None:
        return False
    try:
        arr = np.asarray(src, dtype=float)
        dst = getattr(f, "nodal_values", None)
        if dst is None:
            return False
        if np.asarray(dst).shape != arr.shape:
            raise ValueError(f"shape mismatch for {key}: dump has {arr.shape}, function has {np.asarray(dst).shape}")
        getattr(f, "nodal_values")[:] = arr
        return True
    except Exception as exc:
        raise RuntimeError(f"Failed restoring state for {key}: {exc}") from exc


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


def _read_polygon_csv(path: str, *, scale: float = 1.0, translate: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    arr = np.genfromtxt(str(path), delimiter=",", skip_header=1, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Polygon CSV must have at least 2 columns (x,y); got shape {arr.shape} from {path}")
    poly = np.asarray(arr[:, :2], dtype=float) * float(scale)
    poly[:, 0] += float(translate[0])
    poly[:, 1] += float(translate[1])
    if poly.shape[0] < 3:
        raise ValueError(f"Polygon must have at least 3 points; got {poly.shape[0]} from {path}")
    # Drop repeated last vertex if it matches the first (common for closed polygons).
    if np.allclose(poly[0], poly[-1], rtol=0.0, atol=1.0e-14):
        poly = poly[:-1]
    return poly


def _signed_distance_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Signed distance to a simple polygon.

    Returns φ where:
      - φ < 0 inside the polygon
      - φ > 0 outside the polygon
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError(f"points must be (N,2), got {P.shape}")
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise ValueError(f"polygon must be (M,2) with M>=3, got {poly.shape}")

    # Ensure "open" polygon representation (no repeated last vertex).
    if np.allclose(poly[0], poly[-1], rtol=0.0, atol=1.0e-14):
        poly = poly[:-1]

    N = P.shape[0]
    M = poly.shape[0]

    # --- min distance to edges (squared) ---
    min_d2 = np.full(N, np.inf, dtype=float)
    x = P[:, 0]
    y = P[:, 1]
    x1 = poly[:, 0]
    y1 = poly[:, 1]
    x2 = np.roll(x1, -1)
    y2 = np.roll(y1, -1)
    for i in range(M):
        ax = float(x1[i])
        ay = float(y1[i])
        bx = float(x2[i])
        by = float(y2[i])
        abx = bx - ax
        aby = by - ay
        denom = abx * abx + aby * aby
        if denom <= 0.0:
            continue
        apx = x - ax
        apy = y - ay
        t = (apx * abx + apy * aby) / denom
        t = np.clip(t, 0.0, 1.0)
        dx = apx - t * abx
        dy = apy - t * aby
        d2 = dx * dx + dy * dy
        min_d2 = np.minimum(min_d2, d2)
    dist = np.sqrt(min_d2)

    # --- inside/outside via ray casting (x+ direction) ---
    inside = np.zeros(N, dtype=bool)
    for i in range(M):
        y1i = float(y1[i])
        y2i = float(y2[i])
        dy = y2i - y1i
        if abs(dy) <= 1.0e-30:
            continue
        cond = (y1i > y) != (y2i > y)
        if not np.any(cond):
            continue
        x1i = float(x1[i])
        x2i = float(x2[i])
        x_int = (x2i - x1i) * (y - y1i) / dy + x1i
        inside ^= cond & (x < x_int)

    dist[inside] *= -1.0
    return dist


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
    ap.add_argument(
        "--case",
        type=str,
        default="generic",
        choices=("generic", "dian_paper", "dian_paper_sloughing", "dian_paper_sloughing_gap"),
        help="Preconfigured setups. 'dian_paper' matches the microchannel biofilm deformation case described in "
        "`examples/biofilms/dian_paper/latex/main.tex` (SI units, polygon alpha0, deformation-only run).",
    )
    ap.add_argument("--vtk-every", type=int, default=1, help="Write VTK every N accepted steps (0 disables).")

    # Debugging: per-step state dumps + restart (checkpoints)
    dump_state_default = _env_bool("PYCUTFEM_DUMP_STATE", False)
    ap.add_argument(
        "--dump-state",
        dest="dump_state",
        action="store_true",
        default=dump_state_default,
        help="Write full state dumps (npz) for restart/debug. Env: PYCUTFEM_DUMP_STATE.",
    )
    ap.add_argument(
        "--no-dump-state",
        dest="dump_state",
        action="store_false",
        help="Disable state dumps. Env: PYCUTFEM_DUMP_STATE.",
    )
    ap.add_argument(
        "--dump-state-every",
        type=int,
        default=int(os.getenv("PYCUTFEM_DUMP_STATE_EVERY", "1") or "1"),
        help="Dump every N accepted steps (1 = every step). Env: PYCUTFEM_DUMP_STATE_EVERY.",
    )
    ap.add_argument(
        "--dump-state-dir",
        type=str,
        default=os.getenv("PYCUTFEM_STATE_DUMP_DIR", "").strip(),
        help="Directory for state dumps (default: <outdir>/state_dumps). Env: PYCUTFEM_STATE_DUMP_DIR.",
    )

    restart_dir_default = os.getenv("RESTART_DIR", "").strip()
    ap.add_argument(
        "--restart-dir",
        type=str,
        default=restart_dir_default if restart_dir_default else None,
        help="Base directory containing state_dumps/ for restart. Env: RESTART_DIR.",
    )
    ap.add_argument(
        "--restart-step",
        type=int,
        default=int(os.getenv("RESTART_STEP", "-1") or "-1"),
        help="Dump step number to restart from (matches state_step_####.npz). Env: RESTART_STEP.",
    )
    restart_tag_default = os.getenv("RESTART_TAG", "step").strip().lower() or "step"
    ap.add_argument(
        "--restart-tag",
        choices=("step", "fail"),
        default=restart_tag_default,
        help="Which dump tag to load: step (accepted) or fail (Newton failure). Env: RESTART_TAG.",
    )
    ap.add_argument(
        "--restart-reset-counters",
        action="store_true",
        default=_env_bool("RESTART_RESET_COUNTERS", False),
        help=(
            "Reset step counters for logs/dumps after restart (physical time t is still loaded from checkpoint). "
            "Env: RESTART_RESET_COUNTERS."
        ),
    )
    # Initial biofilm block geometry (smooth)
    ap.add_argument(
        "--alpha0-kind",
        type=str,
        default="block",
        choices=("block", "polygon"),
        help="Initial alpha geometry. 'block' uses (x1,x2,h-biofilm); 'polygon' reads a closed polygon from CSV.",
    )
    ap.add_argument(
        "--alpha0-file",
        type=str,
        default=None,
        help="CSV file for --alpha0-kind polygon (columns x,y; header allowed).",
    )
    ap.add_argument(
        "--alpha0-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to polygon coordinates (e.g. 1e-6 to convert micrometers->meters).",
    )
    ap.add_argument("--alpha0-tx", type=float, default=0.0, help="Translate polygon x by this amount after scaling.")
    ap.add_argument("--alpha0-ty", type=float, default=0.0, help="Translate polygon y by this amount after scaling.")
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
    ap.add_argument(
        "--a-snap",
        type=float,
        default=0.0,
        help="If >0, snap adhesion to full failure by setting a(s)=0 wherever a(s)<a_snap after each update "
        "(irreversible). Useful to prevent tiny residual adhesion from holding a nearly-detached chunk.",
    )
    ap.add_argument("--k-n", type=float, default=50.0, help="Normal spring stiffness [Pa/m].")
    ap.add_argument("--k-t", type=float, default=10.0, help="Tangential spring stiffness [Pa/m].")
    ap.add_argument("--gamma-n", type=float, default=5.0, help="Normal dashpot [Pa*s/m].")
    ap.add_argument("--gamma-t", type=float, default=1.0, help="Tangential dashpot [Pa*s/m].")
    ap.add_argument(
        "--k-break",
        type=float,
        default=2.0,
        help="Adhesion degradation rate [1/s]. Used for shear-based degradation (tau-c) when --sigma-cr=0, "
        "and as the von Mises-driven degradation rate when --sigma-cr>0 (spatial mode).",
    )
    ap.add_argument("--tau-c", type=float, default=0.2, help="Critical wall shear stress [Pa].")
    ap.add_argument("--m-break", type=float, default=1.0, help="Shear exponent m>=1.")
    ap.add_argument(
        "--sigma-cr",
        type=float,
        default=0.0,
        help="Optional solid von Mises stress threshold [Pa] for irreversible wall-adhesion failure in spatial mode. "
        "If >0, then at each step we set a(s)=0 where σ_vm(u)>sigma_cr on Γ_b (paper-style criterion). "
        "Set to 0 to disable and use shear-based degradation (k-break/tau-c).",
    )
    # Bulk damage / cohesion loss (distinct from alpha, which is the phase/occupancy indicator)
    ap.add_argument("--damage-k", type=float, default=0.0, help="Bulk damage rate coefficient k_d [1/s] (0 disables).")
    ap.add_argument(
        "--damage-sigma-cr",
        type=float,
        default=0.0,
        help="Critical von Mises stress [Pa] for bulk damage (0 means: use --sigma-cr).",
    )
    ap.add_argument("--damage-m", type=float, default=2.0, help="Bulk damage exponent m>=1.")
    ap.add_argument("--damage-D", type=float, default=0.0, help="Bulk damage diffusion/regularization coefficient D_d [m^2/s].")
    ap.add_argument(
        "--damage-gamma-out",
        type=float,
        default=0.0,
        help="Penalty coefficient enforcing d->0 in free fluid (stabilizes d DOFs where alpha≈0). Units [1/s].",
    )
    ap.add_argument("--damage-eta-pos", type=float, default=1.0e-12, help="Positive-part smoothing eta for bulk damage activation.")
    ap.add_argument(
        "--damage-kappa-stiff",
        type=float,
        default=1.0e-8,
        help="Stiffness degradation floor κ in g(d)=(1-κ)(1-d)^2+κ (so g(0)=1 and g(1)=κ).",
    )
    ap.add_argument(
        "--damage-kappa-perm",
        type=float,
        default=1.0e-8,
        help="Permeability degradation floor κ in g_perm(d)=(1-κ)(1-d)^2+κ (so g_perm(0)=1 and g_perm(1)=κ).",
    )
    ap.add_argument(
        "--damage-model",
        type=str,
        default="kinetic",
        choices=("kinetic", "phase_field", "phase-field", "at2"),
        help="Bulk damage model: legacy kinetic law or AT2-like phase-field damage.",
    )
    ap.add_argument("--damage-eta", type=float, default=0.0, help="Damage viscosity η_d for phase-field model.")
    ap.add_argument("--damage-Gc", type=float, default=0.0, help="Fracture toughness G_c [J/m^2] for phase-field damage.")
    ap.add_argument("--damage-l", type=float, default=0.0, help="Damage length scale l [m] for phase-field damage.")
    ap.add_argument("--damage-psi0", type=float, default=0.0, help="Optional stress-drive energy scale for phase-field damage (defaults to G_c/l).")
    ap.add_argument(
        "--damage-pf-driver",
        type=str,
        default="von_mises",
        choices=("von_mises", "miehe_energy", "miehe"),
        help="Driving field H_prev for the phase-field damage (AT2) model. "
        "'von_mises' uses a lagged von-Mises proxy (legacy). "
        "'miehe_energy' uses the Miehe tensile elastic energy density ψ⁺(u) (linear elasticity, 2D).",
    )
    ap.add_argument(
        "--damage-stiff-split",
        type=str,
        default="full",
        choices=("full", "miehe"),
        help="How damage d degrades skeleton stiffness. "
        "'full' multiplies the full elastic stress by g(d) (legacy). "
        "'miehe' multiplies only the tensile stress part σ⁺ (Miehe split; linear elasticity, 2D).",
    )
    # Material / model parameters (kept simple)
    ap.add_argument("--rho-f", type=float, default=1.0)
    ap.add_argument("--mu-f", type=float, default=0.1)
    ap.add_argument("--kappa-inv", type=float, default=10.0)
    ap.add_argument(
        "--kappa-inv-model",
        type=str,
        default="kozeny_carman",
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
    ap.add_argument(
        "--solid-model",
        type=str,
        default="linear",
        choices=("linear", "neo_hookean", "neo-hookean", "nh"),
        help="Skeleton constitutive model. Use 'neo_hookean' for large deformations (Eulerian reference-map formulation).",
    )
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
        choices=("shear", "solid_strain", "solid_von_mises", "drag"),
        help=(
            "Mechanical driver used for crack speed. "
            "'shear' uses a fluid shear-stress proxy 2*mu*||eps(v)||, "
            "'solid_strain' uses ||eps(u)||, 'solid_von_mises' uses the skeleton von Mises stress, "
            "and 'drag' is an alias for 'shear' "
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
    # Validation / debug options
    ap.add_argument(
        "--freeze-alpha",
        dest="freeze_alpha",
        action="store_true",
        default=None,
        help="Treat alpha as prescribed (do not solve/update its DOFs). Useful for deformation-only validation.",
    )
    ap.add_argument(
        "--no-freeze-alpha",
        dest="freeze_alpha",
        action="store_false",
        default=None,
        help="Solve/update alpha DOFs (overrides presets that freeze alpha).",
    )
    ap.add_argument(
        "--freeze-phi",
        dest="freeze_phi",
        action="store_true",
        default=None,
        help="Treat phi as prescribed (do not solve/update its DOFs). Useful for deformation-only validation.",
    )
    ap.add_argument(
        "--no-freeze-phi",
        dest="freeze_phi",
        action="store_false",
        default=None,
        help="Solve/update phi DOFs (overrides presets that freeze phi).",
    )
    ap.add_argument(
        "--freeze-S",
        dest="freeze_S",
        action="store_true",
        default=None,
        help="Treat substrate S as prescribed (do not solve/update its DOFs).",
    )
    ap.add_argument(
        "--no-freeze-S",
        dest="freeze_S",
        action="store_false",
        default=None,
        help="Solve/update S DOFs (overrides presets that freeze S).",
    )
    ap.add_argument(
        "--freeze-X",
        dest="freeze_X",
        action="store_true",
        default=None,
        help="Treat detached biomass X as prescribed (do not solve/update its DOFs).",
    )
    ap.add_argument(
        "--no-freeze-X",
        dest="freeze_X",
        action="store_false",
        default=None,
        help="Solve/update X DOFs (overrides presets that freeze X).",
    )
    ap.add_argument(
        "--freeze-damage",
        dest="freeze_damage",
        action="store_true",
        default=None,
        help="Treat bulk damage d as prescribed (do not solve/update its DOFs).",
    )
    ap.add_argument(
        "--no-freeze-damage",
        dest="freeze_damage",
        action="store_false",
        default=None,
        help="Solve/update bulk damage d DOFs (overrides presets that freeze damage).",
    )
    ap.add_argument(
        "--alpha-from-refmap",
        dest="alpha_from_refmap",
        action="store_true",
        default=None,
        help="Recompute alpha after each accepted step from the Eulerian reference-map field u: "
        "alpha(x,t) := alpha0(x - u(x,t)). This keeps alpha sharp and avoids non-conservative Allen–Cahn drift.",
    )
    ap.add_argument(
        "--no-alpha-from-refmap",
        dest="alpha_from_refmap",
        action="store_false",
        default=None,
        help="Disable alpha-from-refmap update (solve alpha if it is not frozen).",
    )
    ap.add_argument(
        "--alpha-refmap-clamp",
        dest="alpha_refmap_clamp",
        action="store_true",
        default=None,
        help="Clamp alpha-from-refmap values into [0,1] (recommended).",
    )
    ap.add_argument(
        "--no-alpha-refmap-clamp",
        dest="alpha_refmap_clamp",
        action="store_false",
        default=None,
        help="Disable clamping of alpha-from-refmap values into [0,1].",
    )
    ap.add_argument(
        "--conserve-alpha",
        dest="conserve_alpha",
        action="store_true",
        default=None,
        help="After each accepted step, apply a logit-shift volume correction so that int_Omega alpha dx matches its t=0 value. "
        "Use only when alpha should be conserved (e.g. k_det=0, no Allen–Cahn/crack alpha dynamics).",
    )
    ap.add_argument(
        "--no-conserve-alpha",
        dest="conserve_alpha",
        action="store_false",
        default=None,
        help="Disable alpha volume correction (overrides presets).",
    )
    ap.add_argument(
        "--deformation-only",
        action="store_true",
        help="Convenience mode for validating fluid–poroelastic deformation: disables detachment/crack/Allen–Cahn, "
        "freezes (alpha,phi,S,X), and clamps the base (--fix-base).",
    )
    ap.add_argument(
        "--deformation-only-phi",
        action="store_true",
        help="Deformation-only validation with evolving phi: disables detachment/crack/Allen–Cahn, freezes (S,X), "
        "clamps the base (--fix-base), and sets alpha from the reference map each step (alpha0(x-u)).",
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

    # ------------------------------------------------------------------
    # Case presets (applied unless overridden on CLI)
    # ------------------------------------------------------------------
    case = str(getattr(args, "case", "generic")).strip().lower()
    argv = list(sys.argv[1:])

    def _has_flag(flag: str) -> bool:
        return any(a == flag or a.startswith(flag + "=") for a in argv)

    if case in {"dian_paper", "dian_paper_sloughing", "dian_paper_sloughing_gap"}:
        # Microchannel biofilm deformation case (Blauert et al. 2015) used in the
        # paper reprint under `examples/biofilms/dian_paper/latex/main.tex`.
        if not _has_flag("--L"):
            args.L = 3500.0e-6
        if not _has_flag("--H"):
            args.H = 1000.0e-6
        if not _has_flag("--Umax"):
            args.Umax = 6.84e-2
        if not _has_flag("--Tramp"):
            # Keep the inflow ramp short on an ms-scale run.
            args.Tramp = 5.0e-4
        if not _has_flag("--phi-b"):
            args.phi_b = 0.47
        if not _has_flag("--eps"):
            # Interface thickness ~ O(10–20 µm) in the paper’s finest scenario.
            args.eps = 20.0e-6

        if not _has_flag("--alpha0-kind"):
            args.alpha0_kind = "polygon"
        if not _has_flag("--alpha0-file"):
            args.alpha0_file = "examples/biofilms/dian_paper/biofilm_initial_closed_polygon_from_spline_um.csv"
        if not _has_flag("--alpha0-scale"):
            args.alpha0_scale = 1.0e-6  # µm -> m

        # Fluid properties: pick mu so that Re≈91 with rho=1000, U=0.0684, H=1e-3.
        if not _has_flag("--rho-f"):
            args.rho_f = 1000.0
        if not _has_flag("--mu-f"):
            args.mu_f = 7.5e-4

        # Drag from hydraulic conductivity K=1e-5 m/s via k = K*mu/(rho*g), so k^{-1} = rho*g/(K*mu).
        if not _has_flag("--kappa-inv"):
            K = 1.0e-5
            g = 9.81
            args.kappa_inv = (float(args.rho_f) * g) / (K * float(args.mu_f))

        # Solid material: linear elastic, E=200 Pa, nu=0.4.
        E = 200.0
        nu = 0.4
        mu_s = E / (2.0 * (1.0 + nu))
        lam_s = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        if not _has_flag("--mu-s"):
            args.mu_s = mu_s
        if not _has_flag("--lambda-s"):
            args.lambda_s = lam_s

        # Time: paper snapshots at 0.8/1.6/3.2 ms and steady at 8 ms.
        if not _has_flag("--dt"):
            args.dt = 1.0e-4
        if not _has_flag("--t-final"):
            args.t_final = 8.0e-3

        # Default to deformation-only mode unless user explicitly requested otherwise.
        # NOTE: For the sloughing preset we want the sloughing physics active by default,
        # so only enable deformation-only automatically for the pure deformation case.
        if case == "dian_paper" and not (
            _has_flag("--process") or _has_flag("--deformation-only") or _has_flag("--deformation-only-phi")
        ):
            args.deformation_only_phi = True

        if not _has_flag("--outdir"):
            args.outdir = "examples/biofilms/results/dian_paper_deformation"

        if case in {"dian_paper_sloughing", "dian_paper_sloughing_gap"}:
            # Stress-driven sloughing (paper-style): use a critical von Mises stress σ_cr on the wall
            # to irreversibly weaken adhesion, while transporting the biofilm indicator α with the
            # Eulerian reference map (α(x,t)=α0(x-u)).
            if not _has_flag("--process"):
                args.process = "sloughing"
            if not _has_flag("--adhesion-integrity"):
                args.adhesion_integrity = "spatial"
            if not _has_flag("--k-det"):
                args.k_det = 0.0
            if not _has_flag("--sigma-cr"):
                args.sigma_cr = 40.0

            # Bulk damage (cohesion loss) driven by σ_vm(u)>σ_cr.
            # Default to an AT2-like phase-field model for thermodynamic consistency.
            if not _has_flag("--damage-model"):
                args.damage_model = "phase_field"
            if not _has_flag("--damage-sigma-cr"):
                args.damage_sigma_cr = float(args.sigma_cr)
            if not _has_flag("--damage-m"):
                args.damage_m = 2.0
            dmg_model_key = str(getattr(args, "damage_model", "kinetic")).strip().lower()
            if dmg_model_key in {"kinetic"}:
                if not _has_flag("--damage-k"):
                    # Legacy kinetic defaults kept for backward compatibility.
                    args.damage_k = 5.0e6
            else:
                # Calibrate Gc from target σ_cr using AT2 scaling:
                #   sigma_c = (3/16) * sqrt(3 E' Gc / l)
                # => Gc = 256 l sigma_c^2 / (27 E')
                # with E' = E/(1-nu^2) for plane-strain-like scaling.
                if not _has_flag("--damage-k"):
                    args.damage_k = 0.0
                if not _has_flag("--damage-l"):
                    args.damage_l = max(float(args.eps), 5.0e-5)
                if not _has_flag("--damage-Gc"):
                    E_eff = E / (1.0 - nu * nu)
                    ell = float(args.damage_l)
                    sigma_c = max(1.0e-12, float(args.damage_sigma_cr))
                    args.damage_Gc = (256.0 * ell * sigma_c * sigma_c) / (27.0 * E_eff)
                if not _has_flag("--damage-eta"):
                    args.damage_eta = 1.0
            if not _has_flag("--damage-D"):
                args.damage_D = 0.0
            if not _has_flag("--damage-gamma-out"):
                # Stabilize damage DOFs where α≈0 (free fluid): enforce d≈0 there.
                args.damage_gamma_out = 1.0e4
            if not _has_flag("--damage-kappa-perm"):
                # Make failed regions hydraulically open (β≈0) even with κ^{-1} ~ O(1e12).
                args.damage_kappa_perm = 1.0e-12
            if not _has_flag("--damage-kappa-stiff"):
                # Keep a tiny stiffness floor for conditioning without impeding crack opening.
                args.damage_kappa_stiff = 1.0e-12
            if not _has_flag("--damage-pf-driver"):
                args.damage_pf_driver = "miehe_energy"
            if not _has_flag("--damage-stiff-split"):
                args.damage_stiff_split = "miehe"

            # Paper sloughing is a failure model (no growth/erosion). Keep α/S/X prescribed by default:
            #   - α is transported by the reference map to follow the deforming/moving biofilm.
            #   - S and X are held at zero.
            if getattr(args, "freeze_alpha", None) is None:
                args.freeze_alpha = True
            if getattr(args, "alpha_from_refmap", None) is None:
                args.alpha_from_refmap = True
            if getattr(args, "alpha_refmap_clamp", None) is None:
                args.alpha_refmap_clamp = True
            if getattr(args, "freeze_S", None) is None:
                args.freeze_S = True
            # Only freeze X when erosion is not active; allow `--process both` to solve X.
            if getattr(args, "freeze_X", None) is None:
                proc_key = str(getattr(args, "process", "both")).strip().lower()
                args.freeze_X = proc_key not in {"erosion", "both"}

            # Keep adhesion traction; use σ_cr-driven *binary* failure by default.
            # (Set --k-break>0 to use the smoother rate law in `update_adhesion_integrity_field_on_boundary_von_mises`.)
            if not _has_flag("--k-break"):
                args.k_break = 0.0
            if not _has_flag("--tau-c"):
                args.tau_c = 1.0e9

            # Scale adhesion stiffness to SI units: to generate O(10–100) Pa tractions
            # for O(10–100) µm displacements, k should be O(1e5–1e6) Pa/m.
            if not _has_flag("--k-n"):
                args.k_n = 5.0e5
            if not _has_flag("--k-t"):
                args.k_t = 2.0e5
            if not _has_flag("--gamma-n"):
                args.gamma_n = 5.0e3
            if not _has_flag("--gamma-t"):
                args.gamma_t = 2.0e3

            # Disable Allen–Cahn / crack α dynamics in this preset (paper uses σ_cr failure instead).
            if not _has_flag("--alpha-cahn-M"):
                args.alpha_cahn_M = 0.0
            if not _has_flag("--alpha-cahn-gamma"):
                args.alpha_cahn_gamma = 0.0
            if not _has_flag("--k-crack"):
                args.k_crack = 0.0

            # φ is a pure CG transport equation when D_phi=0; enable consistent stabilization by default.
            if not _has_flag("--phi-supg"):
                args.phi_supg = 1.0
            if not _has_flag("--phi-cip"):
                args.phi_cip = 10.0

            # Favor robust Newton parameters.
            if not _has_flag("--ls-mode"):
                args.ls_mode = "dealii"
            if not _has_flag("--newton-tol"):
                args.newton_tol = 1.0e-4
            if not _has_flag("--max-it"):
                args.max_it = 60
            if not _has_flag("--u-extension"):
                args.u_extension = "grad"
            if not _has_flag("--gamma-u"):
                args.gamma_u = 2.0
            if not _has_flag("--gamma-u-pin"):
                # NOTE: the pin enters as (gamma_u_pin / h^2) in the form. On micro-scale meshes
                # even 1e-8 can become O(1). Keep this truly tiny and rely on u-CIP for conditioning.
                args.gamma_u_pin = 1.0e-12
            if not _has_flag("--u-cip"):
                args.u_cip = 1.0

            # Default to the paper’s detachment-pattern time (6.4 ms) and output at 0.8/1.6/3.2/6.4 ms.
            if not _has_flag("--dt"):
                args.dt = 2.0e-4
            if not _has_flag("--t-final"):
                args.t_final = 6.4e-3
            if not _has_flag("--vtk-every"):
                args.vtk_every = 4
            if not _has_flag("--outdir"):
                args.outdir = "examples/biofilms/results/dian_paper_sloughing"

            # Sloughing onset indicators tuned to the micro-scale Dian run.
            if not _has_flag("--slough-liftoff-dy"):
                args.slough_liftoff_dy = 20.0e-6
            if not _has_flag("--slough-contact-rel-thresh"):
                args.slough_contact_rel_thresh = 0.98

            if case == "dian_paper_sloughing_gap":
                # Gap/peeling variant: seed a small notch (initial fluid gap) at the wall.
                # Keep α transported by the Eulerian reference map (α(x,t)=α0(x-u)) for robustness
                # and to avoid non-conservative Allen–Cahn drift. The crack then opens mechanically
                # as wall adhesion fails, allowing lift-off to be detected.
                if not _has_flag("--crack-depth"):
                    # Keep the seeded gap *small* and mesh-resolvable. The default in earlier
                    # experiments (100 µm) can fully detach the high-α region on coarse meshes.
                    args.crack_depth = 80.0e-6
                if not _has_flag("--crack-width"):
                    # Narrow notch so the biofilm remains attached outside the gap region.
                    args.crack_width = 400.0e-6
                if not _has_flag("--k-break"):
                    # For the gap case, use smooth stress-driven degradation by default.
                    # This avoids binary "stick/slip" behavior and improves peel-off robustness.
                    args.k_break = 20.0
                if not _has_flag("--a-snap"):
                    # Prevent tiny residual adhesion values from holding the chunk once it is effectively detached.
                    args.a_snap = 0.05
                if getattr(args, "conserve_alpha", None) is None:
                    # In the gap preset we disable detachment (k_det=0) and transport alpha via refmap.
                    # Enforce a global volume constraint to avoid long-run alpha depletion from mesh-level drift.
                    args.conserve_alpha = True
                if not _has_flag("--outdir"):
                    args.outdir = "examples/biofilms/results/dian_paper_sloughing_gap"

    L = float(args.L)
    H = float(args.H)
    qdeg = int(args.q)
    dt_val = float(args.dt)
    theta = float(args.theta)
    backend = str(args.backend)

    # Basic mesh/interface resolution sanity: a too-thin diffuse interface (eps << h)
    # behaves like an under-resolved step function in CG spaces and can lead to
    # noisy coefficients in the one-domain blend.
    try:
        h_min = min(float(L) / max(1, int(args.nx)), float(H) / max(1, int(args.ny)))
        eps_init = float(getattr(args, "eps", 0.0) or 0.0)
        if eps_init > 0.0 and eps_init < 0.75 * h_min:
            print(
                f"[warn] Diffuse-interface thickness --eps={eps_init:.3e} is smaller than the element size "
                f"h≈{h_min:.3e}. This can be under-resolved and may cause non-physical transients; "
                "consider eps≈(1–2)h for validation runs."
            )
    except Exception:
        pass
    if bool(getattr(args, "deformation_only", False)):
        args.process = "none"
        args.k_det = 0.0
        args.crack_depth = 0.0
        args.alpha_cahn_M = 0.0
        args.alpha_cahn_gamma = 0.0
        args.alpha_crack_k = 0.0
        args.damage_k = 0.0
        args.damage_D = 0.0
        args.damage_gamma_out = 0.0
        args.freeze_damage = True
        args.freeze_alpha = True
        args.freeze_phi = True
        args.freeze_S = True
        args.freeze_X = True
        args.fix_base = True
    if bool(getattr(args, "deformation_only_phi", False)):
        args.process = "none"
        args.k_det = 0.0
        args.crack_depth = 0.0
        args.alpha_cahn_M = 0.0
        args.alpha_cahn_gamma = 0.0
        args.alpha_crack_k = 0.0
        args.damage_k = 0.0
        args.damage_D = 0.0
        args.damage_gamma_out = 0.0
        args.freeze_damage = True
        args.freeze_alpha = True  # alpha is prescribed via refmap update
        args.freeze_phi = False
        args.freeze_S = True
        args.freeze_X = True
        args.fix_base = True
        args.alpha_from_refmap = True
        args.alpha_refmap_clamp = True

    process = str(getattr(args, "process", "both")).strip().lower()
    use_erosion = process in {"erosion", "both"}
    use_sloughing = process in {"sloughing", "both"}
    adhesion_integrity = str(getattr(args, "adhesion_integrity", "scalar")).strip().lower()
    use_spatial_adhesion = bool(use_sloughing and adhesion_integrity == "spatial")

    # Bulk damage is enabled if any of its coefficients are non-zero.
    damage_model_key = str(getattr(args, "damage_model", "kinetic")).strip().lower()
    use_damage = bool(
        float(getattr(args, "damage_k", 0.0) or 0.0) != 0.0
        or float(getattr(args, "damage_D", 0.0) or 0.0) != 0.0
        or float(getattr(args, "damage_gamma_out", 0.0) or 0.0) != 0.0
        or (damage_model_key in {"phase_field", "phase-field", "at2"} and (
            float(getattr(args, "damage_Gc", 0.0) or 0.0) != 0.0
            or float(getattr(args, "damage_l", 0.0) or 0.0) != 0.0
            or float(getattr(args, "damage_eta", 0.0) or 0.0) != 0.0
        ))
    )
    if use_damage and float(getattr(args, "damage_sigma_cr", 0.0) or 0.0) <= 0.0:
        # Default to the same stress threshold used for wall adhesion if provided.
        args.damage_sigma_cr = float(getattr(args, "sigma_cr", 0.0) or 0.0)

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

    # ------------------------------------------------------------------
    # Debugging: state dumps + restart
    # ------------------------------------------------------------------
    dump_state = bool(getattr(args, "dump_state", False))
    try:
        dump_state_every = int(getattr(args, "dump_state_every", 1))
    except Exception:
        dump_state_every = 1
    dump_state_every = max(0, int(dump_state_every))
    state_dump_dir = str(getattr(args, "dump_state_dir", "") or "").strip() or os.path.join(str(args.outdir), "state_dumps")

    restart_payload = None
    restart_t0 = 0.0
    restart_step0 = 0
    restart_dir = getattr(args, "restart_dir", None)
    try:
        restart_step = int(getattr(args, "restart_step", -1))
    except Exception:
        restart_step = -1
    restart_tag = str(getattr(args, "restart_tag", "step") or "step").strip().lower() or "step"
    restart_reset = bool(getattr(args, "restart_reset_counters", False))
    if restart_dir is not None and restart_step >= 0:
        restart_base = Path(restart_dir)
        st_dir_raw = os.getenv("RESTART_STATE_DIR", "").strip()
        state_dir = Path(st_dir_raw) if st_dir_raw else restart_base / "state_dumps"
        state_path = state_dir / f"state_{restart_tag}_{int(restart_step):04d}.npz"
        restart_state = _load_npz_dict(state_path)
        restart_t0 = float(_npz_scalar(restart_state, "t", 0.0))
        restart_dt0 = float(_npz_scalar(restart_state, "dt", dt_val))
        try:
            restart_step_loaded = int(np.asarray(restart_state.get("step", restart_step)).reshape(()))
        except Exception:
            restart_step_loaded = int(restart_step)
        if restart_reset:
            restart_step0 = 0
        else:
            restart_step0 = (
                max(0, int(restart_step_loaded) - 1) if str(restart_tag) == "fail" else int(restart_step_loaded)
            )
        restart_payload = {
            "state_path": str(state_path),
            "state": restart_state,
            "t": float(restart_t0),
            "dt": float(restart_dt0),
            "step": int(restart_step_loaded),
        }
        msg = f"[restart] loaded {str(state_path)} (step={restart_step_loaded} t={restart_t0:.6e} dt={restart_dt0:.3e})"
        if restart_reset:
            msg += " [reset counters]"
        print(msg)
        if abs(float(restart_dt0) - float(dt_val)) > 0.0:
            print(
                f"[restart] note: dump dt={restart_dt0:.3e} differs from --dt={dt_val:.3e}; continuing with --dt."
            )

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

    solid_model = str(getattr(args, "solid_model", "linear")).strip().lower()
    if solid_model in {"neo-hookean", "nh"}:
        solid_model = "neo_hookean"
    if bool(use_sloughing) and solid_model == "linear":
        print("[warn] --process sloughing uses large biofilm motion; consider --solid-model neo_hookean for better large-strain behavior.")

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
        # Bulk damage / cohesion loss (optional).
        **({"d": 1} if use_damage else {}),
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
    dd = TrialFunction("d", dof_handler=dh) if use_damage else None
    dS_trial = TrialFunction("S", dof_handler=dh)
    dX_trial = TrialFunction("X", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    d_test = TestFunction("d", dof_handler=dh) if use_damage else None
    S_test = TestFunction("S", dof_handler=dh)
    X_test = TestFunction("X", dof_handler=dh)

    # Unknowns (k) and previous state (n)
    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    d_k = Function("d_k", "d", dof_handler=dh) if use_damage else None
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
    d_n = Function("d_n", "d", dof_handler=dh) if use_damage else None
    S_n = Function("S_n", "S", dof_handler=dh)
    X_n = Function("X_n", "X", dof_handler=dh)
    if use_spatial_adhesion:
        a_prev = Function("a_prev", "a", dof_handler=dh)
    else:
        a_prev = None

    def _mark_inactive_fields(*field_names: str) -> None:
        tags = getattr(dh, "dof_tags", None) or {}
        inactive = set(tags.get("inactive", set()))
        for fname in field_names:
            try:
                sl = np.asarray(dh.get_field_slice(fname), dtype=int).ravel()
            except Exception:
                continue
            inactive.update(int(i) for i in sl)
        tags["inactive"] = inactive
        dh.dof_tags = tags

    if use_spatial_adhesion:
        _mark_inactive_fields("a")
    if bool(getattr(args, "freeze_alpha", False)):
        _mark_inactive_fields("alpha")
    if bool(getattr(args, "freeze_phi", False)):
        _mark_inactive_fields("phi")
    if bool(getattr(args, "freeze_S", False)):
        _mark_inactive_fields("S")
    if bool(getattr(args, "freeze_X", False)):
        _mark_inactive_fields("X")
    if use_damage and bool(getattr(args, "freeze_damage", False)):
        _mark_inactive_fields("d")

    # ------------------------------------------------------------------
    # Initial biofilm indicator (smooth diffuse interface)
    # ------------------------------------------------------------------
    alpha0_kind = str(getattr(args, "alpha0_kind", "block")).strip().lower()
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

    poly_alpha0 = None
    if alpha0_kind == "polygon":
        if not getattr(args, "alpha0_file", None):
            raise ValueError("--alpha0-kind polygon requires --alpha0-file")
        poly_alpha0 = _read_polygon_csv(
            str(args.alpha0_file),
            scale=float(getattr(args, "alpha0_scale", 1.0)),
            translate=(float(getattr(args, "alpha0_tx", 0.0)), float(getattr(args, "alpha0_ty", 0.0))),
        )
        xmin, ymin = np.min(poly_alpha0, axis=0)
        xmax, ymax = np.max(poly_alpha0, axis=0)
        # Use polygon bbox for convenience diagnostics and (optional) wall-adhesion perturbation defaults.
        x1 = float(xmin)
        x2 = float(xmax)
        h_b = float(ymax)
        # If the user did not explicitly specify --crack-x0, default to the polygon midpoint.
        # (The global default computed earlier uses the block parameters and is not meaningful for polygon alpha0.)
        if not _has_flag("--crack-x0"):
            crack_x0 = 0.5 * (x1 + x2)
            args.crack_x0 = float(crack_x0)
        print(
            f"[info] alpha0 polygon: {str(args.alpha0_file)} "
            f"(bbox x=[{xmin:.3e},{xmax:.3e}], y=[{ymin:.3e},{ymax:.3e}], eps={eps:.3e})"
        )
        if xmin < -1.0e-12 or xmax > float(L) + 1.0e-12 or ymin < -1.0e-12 or ymax > float(H) + 1.0e-12:
            print(
                f"[warn] alpha0 polygon bbox does not fit strictly inside the domain [0,L]x[0,H]: "
                f"L={float(L):.3e}, H={float(H):.3e}."
            )

        def alpha0_eval(x, y):
            x = np.asarray(x, float)
            y = np.asarray(y, float)
            xx, yy = np.broadcast_arrays(x, y)
            pts = np.column_stack((xx.ravel(), yy.ravel()))
            phi = _signed_distance_polygon(pts, poly_alpha0)
            a0 = _smooth_step(-phi / eps_x).reshape(xx.shape)
            # Optional initial crack notch (fluid gap) from the bottom wall into the biofilm.
            # This seeds an internal diffuse interface so the crack-speed term can propagate it.
            if crack_depth > 0.0 and crack_width > 0.0:
                xL = crack_x0 - 0.5 * crack_width
                xR = crack_x0 + 0.5 * crack_width
                wcx = _smooth_step((xx - xL) / eps_x) * _smooth_step((xR - xx) / eps_x)
                wcy = _smooth_step((crack_depth - yy) / eps_y)
                a0 = a0 * (1.0 - wcx * wcy)
            return np.clip(a0, 0.0, 1.0)

    else:

        def alpha0_eval(x, y):
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

    def alpha0(x, y):
        # Scalar wrapper (kept for compatibility with existing scalar callbacks).
        return alpha0_eval(x, y)

    # Vectorized initialization on CG nodes (avoids per-node Python loops for polygon alpha0).
    alpha_gdofs = np.asarray(dh.get_field_dofs_on_nodes("alpha"), dtype=int).ravel()
    alpha_xy = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    alpha_vals = np.asarray(alpha0_eval(alpha_xy[:, 0], alpha_xy[:, 1]), dtype=float).ravel()
    alpha_n.set_nodal_values(alpha_gdofs, alpha_vals)

    phi_gdofs = np.asarray(dh.get_field_dofs_on_nodes("phi"), dtype=int).ravel()
    phi_xy = np.asarray(dh.get_dof_coords("phi"), dtype=float)
    if phi_xy.shape == alpha_xy.shape and np.allclose(phi_xy, alpha_xy, rtol=0.0, atol=1.0e-14):
        phi_vals = 1.0 - (1.0 - float(phi_b)) * alpha_vals
    else:
        phi_vals = 1.0 - (1.0 - float(phi_b)) * np.asarray(alpha0_eval(phi_xy[:, 0], phi_xy[:, 1]), dtype=float).ravel()
    phi_n.set_nodal_values(phi_gdofs, np.asarray(phi_vals, dtype=float).ravel())
    S_n.set_values_from_function(lambda x, y: 0.0)
    X_n.set_values_from_function(lambda x, y: 0.0)
    v_n.set_values_from_function(lambda x, y: np.array([0.0, 0.0], dtype=float))
    u_n.set_values_from_function(lambda x, y: np.array([0.0, 0.0], dtype=float))
    if u_nm1 is not None:
        u_nm1.set_values_from_function(lambda x, y: np.array([0.0, 0.0], dtype=float))
    p_n.set_values_from_function(lambda x, y: 0.0)
    if use_damage and d_n is not None:
        d_n.set_values_from_function(lambda x, y: 0.0)
    if a_prev is not None:
        a0 = float(args.a0)
        a_pert = float(getattr(args, "a_perturb", 0.0) or 0.0)
        a_pert_k = int(getattr(args, "a_perturb_k", 1) or 1)
        Lb = max(1.0e-12, float(x2 - x1))
        a_gdofs = np.asarray(dh.get_field_dofs_on_nodes("a"), dtype=int).ravel()
        a_xy = np.asarray(dh.get_dof_coords("a"), dtype=float)
        ax = a_xy[:, 0]
        mask = np.asarray(alpha0_eval(ax, np.zeros_like(ax)), dtype=float).ravel()
        a_vals = np.full_like(ax, fill_value=float(a0), dtype=float)
        if a_pert != 0.0:
            phase = 2.0 * np.pi * float(a_pert_k) * (ax - float(x1)) / Lb
            a_vals = np.clip(a0 * (1.0 - a_pert * mask * np.sin(phase)), 0.0, 1.0)
        a_prev.set_nodal_values(a_gdofs, np.asarray(a_vals, dtype=float).ravel())

    # ------------------------------------------------------------------
    # Optional restart: load solution snapshots from a prior run
    # ------------------------------------------------------------------
    if restart_payload is not None:
        st = dict(restart_payload.get("state", {}) or {})
        restored: list[str] = []
        for f in [
            # Current/previous unknowns
            v_k, p_k, u_k, phi_k, alpha_k, d_k, S_k, X_k,
            v_n, p_n, u_n, phi_n, alpha_n, d_n, S_n, X_n,
            # Extra history / coefficients
            u_prev, u_nm1, a_prev,
        ]:
            if f is None:
                continue
            if _restore_function_from_state(f, st):
                restored.append(str(getattr(f, "name", "?")))
        if a_prev is None and ("a_scalar" in st):
            try:
                restart_payload["a_scalar"] = float(_npz_scalar(st, "a_scalar", float(args.a0)))
            except Exception:
                restart_payload["a_scalar"] = float(args.a0)
        if restored:
            print(f"[restart] restored {len(restored)} fields: {', '.join(restored)}")
        else:
            print("[restart] warning: no matching fields restored from dump.")

    # ------------------------------------------------------------------
    # Forms with adhesion traction on the bottom wall
    # ------------------------------------------------------------------
    dt_c = Constant(dt_val)
    a_c = Constant(float(restart_payload.get("a_scalar", args.a0)) if restart_payload is not None else float(args.a0))

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
        d_k=d_k,
        S_k=S_k,
        v_n=v_n,
        p_n=p_n,
        u_n=u_n,
        u_nm1=u_nm1,
        phi_n=phi_n,
        alpha_n=alpha_n,
        d_n=d_n,
        S_n=S_n,
        dv=dv,
        dp=dp,
        du=du,
        dphi=dphi,
        dalpha=dalpha,
        dS=dS_trial,
        dd=dd,
        v_test=v_test,
        q_test=q_test,
        u_test=u_test,
        phi_test=phi_test,
        alpha_test=alpha_test,
        S_test=S_test,
        d_test=d_test,
        X_test=X_test,
        dx=dx(metadata={"q": int(qdeg)}),
        dt=dt_c,
        theta=theta,
        rho_f=rho_f_c,
        mu_f=mu_f_c,
        kappa_inv=kappa_inv_c,
        kappa_inv_model=str(getattr(args, "kappa_inv_model", "spatial")),
        # Only used by the Kozeny–Carman permeability model; default to phi_b otherwise.
        kappa_inv_phi_ref=float(getattr(args, "kappa_phi_ref", None) or args.phi_b),
        solid_model=solid_model,
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
        damage_k=float(getattr(args, "damage_k", 0.0) or 0.0),
        damage_sigma_cr=float(getattr(args, "damage_sigma_cr", 0.0) or 0.0),
        damage_m=float(getattr(args, "damage_m", 1.0) or 1.0),
        damage_D=float(getattr(args, "damage_D", 0.0) or 0.0),
        damage_gamma_out=float(getattr(args, "damage_gamma_out", 0.0) or 0.0),
        damage_eta_pos=float(getattr(args, "damage_eta_pos", 1.0e-12) or 1.0e-12),
        damage_kappa_stiff=float(getattr(args, "damage_kappa_stiff", 1.0e-8) or 1.0e-8),
        damage_kappa_perm=float(getattr(args, "damage_kappa_perm", 1.0e-8) or 1.0e-8),
        damage_model=str(getattr(args, "damage_model", "kinetic")),
        damage_eta=float(getattr(args, "damage_eta", 0.0) or 0.0),
        damage_Gc=float(getattr(args, "damage_Gc", 0.0) or 0.0),
        damage_l=float(getattr(args, "damage_l", 0.0) or 0.0),
        damage_psi0=float(getattr(args, "damage_psi0", 0.0) or 0.0),
        damage_pf_driver=str(getattr(args, "damage_pf_driver", "von_mises")),
        damage_stiff_split=str(getattr(args, "damage_stiff_split", "full")),
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
    step_counter = {"k": int(restart_step0)}
    a_state = {
        "val": float(restart_payload.get("a_scalar", args.a0)) if restart_payload is not None else float(args.a0)
    }
    alpha_area0 = {"val": None}
    slough_flags = {"a_drop": False, "contact_loss": False, "liftoff": False, "motion": False, "wall_clear": False}
    slough_t = {"a_drop": None, "contact_loss": None, "liftoff": None, "motion": None, "wall_clear": None}
    alpha_cm_hi0 = {"y": None}

    if dump_state:
        os.makedirs(state_dump_dir, exist_ok=True)
        print(f"[dump] state dumps enabled: dir={state_dump_dir} every={dump_state_every}")

    def _dump_state_snapshot(step_no: int, t_curr: float, dt_curr: float, *, tag: str, funcs: list[object]) -> None:
        if not dump_state:
            return
        if str(tag) == "step":
            if int(dump_state_every) <= 0:
                return
            if (int(step_no) % int(dump_state_every)) != 0:
                return
        try:
            os.makedirs(state_dump_dir, exist_ok=True)
            arrays = _collect_function_arrays(list(funcs or []))
            payload: dict[str, object] = {
                "step": int(step_no),
                "t": float(t_curr),
                "dt": float(dt_curr),
                "mesh_n_nodes": int(getattr(mesh.nodes_x_y_pos, "shape", (0,))[0]),
                "mesh_n_elements": int(getattr(mesh, "n_elements", len(getattr(mesh, "elements_list", [])))),
                "mesh_element_type": str(getattr(mesh, "element_type", "")),
                "mesh_poly_order": int(getattr(mesh, "poly_order", 1)),
                "dof_total": int(getattr(dh, "total_dofs", -1)),
            }
            if a_prev is None:
                payload["a_scalar"] = float(a_state["val"])
            payload.update(arrays)
            fname = os.path.join(state_dump_dir, f"state_{str(tag)}_{int(step_no):04d}.npz")
            np.savez_compressed(fname, **payload)
        except Exception as exc:
            print(f"[warn] state dump failed (tag={tag} step={step_no}): {exc}")

    def _on_step_failure(**ctx):
        step_fail = int(ctx.get("step_no", ctx.get("step", 0)))
        t_fail = float(ctx.get("t", 0.0))
        dt_fail = float(ctx.get("dt", dt_val))
        funcs_fail = list(ctx.get("functions", []) or [])
        prev_fail = list(ctx.get("prev_functions", []) or [])
        aux_vals = list((ctx.get("aux_functions", {}) or {}).values())
        _dump_state_snapshot(
            step_fail,
            t_fail,
            dt_fail,
            tag="fail",
            funcs=funcs_fail + prev_fail + aux_vals + [u_prev],
        )
        return False
    num_nodes = len(mesh.nodes_list)
    node_xy = np.asarray(getattr(mesh, "nodes_x_y_pos", np.zeros((num_nodes, 2), dtype=float)), dtype=float)
    alpha_dof_xy = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    alpha_dof_gids = np.asarray(dh.get_field_slice("alpha"), dtype=int).ravel()
    alpha_node_ids = []
    for gd in alpha_dof_gids:
        _fld, nid = dh._dof_to_node_map.get(int(gd), (None, None))
        if nid is None:
            raise RuntimeError("alpha-from-refmap requires CG alpha DOFs to be node-attached.")
        alpha_node_ids.append(int(nid))
    alpha_node_ids = np.asarray(alpha_node_ids, dtype=int)

    alpha_conserve_enabled = bool(getattr(args, "conserve_alpha", False))
    alpha_weights = None
    alpha_mass_target = None
    if alpha_conserve_enabled:
        try:
            elem_corners = np.asarray(getattr(mesh, "corner_connectivity", None), dtype=int)
            if elem_corners.ndim != 2 or elem_corners.shape[1] != 4:
                raise ValueError(f"expected quad corner-node connectivity, got {getattr(elem_corners, 'shape', None)}")
            node_w = np.zeros(num_nodes, dtype=float)
            for corners in elem_corners:
                pts = node_xy[np.asarray(corners, dtype=int), :]
                x = pts[:, 0]
                y = pts[:, 1]
                # Polygon area formula for a (planar) quad.
                area = 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
                node_w[np.asarray(corners, dtype=int)] += area / 4.0
            alpha_weights = node_w[alpha_node_ids]
            alpha_mass_target = float(np.dot(alpha_weights, np.asarray(alpha_n.nodal_values, dtype=float)))
            print(f"[info] --conserve-alpha enabled: target int_alpha={alpha_mass_target:.3e}")
        except Exception as exc:
            print(f"[warn] --conserve-alpha requested but weight build failed ({exc}); disabling alpha conservation.")
            alpha_conserve_enabled = False
            alpha_weights = None
            alpha_mass_target = None
    # Reuse the initial geometry definition for refmap transport and diagnostics.
    _alpha0_eval = alpha0_eval

    def _update_alpha_from_refmap():
        if not bool(getattr(args, "alpha_from_refmap", False)):
            return
        # Reference map: χ = x - u (u is the stored "reference map displacement").
        # Evaluate u at the alpha DOF nodes (alpha is Q1 on the Q2 geometry mesh).
        u_nodes = _vector_to_nodes(u_k)
        u_at_alpha = u_nodes[alpha_node_ids, :]
        chi = alpha_dof_xy - u_at_alpha
        a_dofs = _alpha0_eval(chi[:, 0], chi[:, 1])
        if bool(getattr(args, "alpha_refmap_clamp", False)):
            a_dofs = np.clip(a_dofs, 0.0, 1.0)
        alpha_k.nodal_values[:] = np.asarray(a_dofs, dtype=float)

    def _apply_alpha_volume_constraint():
        if not alpha_conserve_enabled or alpha_weights is None or alpha_mass_target is None:
            return
        # Only meaningful when alpha has no intended sinks/sources.
        if bool(use_erosion) and float(getattr(args, "k_det", 0.0) or 0.0) != 0.0:
            return
        if float(getattr(args, "k_crack", 0.0) or 0.0) != 0.0:
            return
        if float(getattr(args, "alpha_cahn_M", 0.0) or 0.0) != 0.0 or float(getattr(args, "alpha_cahn_gamma", 0.0) or 0.0) != 0.0:
            return
        res = logit_shift_to_match_integral(
            np.asarray(alpha_k.nodal_values, dtype=float),
            weights=np.asarray(alpha_weights, dtype=float),
            target_mass=float(alpha_mass_target),
        )
        alpha_k.nodal_values[:] = np.asarray(res.values, dtype=float)

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
        dt_step = float(getattr(solver, "_current_dt", dt_val))
        t_now = step_no * dt_val
        try:
            t_start = getattr(solver, "_current_t", None)
            if t_start is not None:
                t_now = float(t_start) + float(dt_step)
        except Exception:
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
                sigma_cr = float(getattr(args, "sigma_cr", 0.0) or 0.0)
                if sigma_cr > 0.0:
                    upd = update_adhesion_integrity_field_on_boundary_von_mises(
                        dof_handler=dh,
                        a_field=a_prev,
                        dt=dt_val,
                        u=u_k,
                        alpha=alpha_k,
                        ds_wall=ds_bottom,
                        mu_s=mu_s_c,
                        lambda_s=lambda_s_c,
                        k_break=float(args.k_break),
                        sigma_cr=sigma_cr,
                        m=float(args.m_break),
                        a_snap=float(getattr(args, "a_snap", 0.0) or 0.0),
                        backend=backend,
                        quad_order=qdeg,
                    )
                    a_msg = f"a[min,max]=[{upd.a_min:.3f},{upd.a_max:.3f}]  sigma_vm[max]={upd.tau_max:.3e}"
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
                        a_snap=float(getattr(args, "a_snap", 0.0) or 0.0),
                        backend=backend,
                        quad_order=qdeg,
                    )
                    a_msg = f"a[min,max]=[{upd.a_min:.3f},{upd.a_max:.3f}]"
                a_min_val = float(upd.a_min)
                a_max_val = float(upd.a_max)

        # NOTE: alpha-from-refmap is applied in post_step_refiner (before promotion),
        # so alpha_n matches the updated alpha_k at the accepted state.

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
        # Mass/volume diagnostics (independent of erosion/X)
        try:
            mass_every = int(getattr(args, "mass_every", 0) or 0)
            if mass_every > 0 and (step_no % mass_every == 0):
                int_alpha = assemble_scalar(dh, alpha_k * dx_q, backend=backend, quad_order=qdeg)
                int_B = assemble_scalar(
                    dh,
                    alpha_k * (Constant(1.0) - phi_k) * dx_q,
                    backend=backend,
                    quad_order=qdeg,
                )
                print(f"           int_alpha={int_alpha:.3e}  int_B={int_B:.3e}")
        except Exception:
            pass
        try:
            a_min = float(np.min(alpha_k.nodal_values))
            a_max = float(np.max(alpha_k.nodal_values))
            p_min = float(np.min(phi_k.nodal_values))
            p_max = float(np.max(phi_k.nodal_values))
            dmg_msg = ""
            if d_k is not None:
                d_min = float(np.min(d_k.nodal_values))
                d_max = float(np.max(d_k.nodal_values))
                dmg_msg = f"  d[min,max]=[{d_min:.3e},{d_max:.3e}]"
            print(f"           alpha[min,max]=[{a_min:.3e},{a_max:.3e}]  phi[min,max]=[{p_min:.3e},{p_max:.3e}]{dmg_msg}")

            # Lightweight nodal diagnostics (helps interpret "no Darcy effect / wrong motion")
            alpha_nodes = _scalar_to_nodes(alpha_k)
            phi_nodes = _scalar_to_nodes(phi_k)
            # Diagnostic drag coefficient β = α μ_f φ^2 κ^{-1}(φ) g_perm(d).
            kappa_inv_eff = float(args.kappa_inv)
            if str(getattr(args, "kappa_inv_model", "spatial")).strip().lower() in {"kozeny", "kozeny_carman", "kc"}:
                phi_ref = float(getattr(args, "kappa_phi_ref", None) or args.phi_b)
                eps_kc = 1.0e-12
                g = ((1.0 - phi_nodes) ** 2) / (phi_nodes**3 + eps_kc)
                g0 = ((1.0 - phi_ref) ** 2) / (phi_ref**3 + eps_kc)
                kappa_inv_eff = float(args.kappa_inv) * (g / max(1.0e-30, g0))
            g_perm_nodes = 1.0
            if d_k is not None:
                d_nodes = _scalar_to_nodes(d_k)
                kappa_perm = float(getattr(args, "damage_kappa_perm", 1.0e-8) or 1.0e-8)
                g_perm_nodes = (1.0 - kappa_perm) * ((1.0 - d_nodes) ** 2) + kappa_perm
            beta_nodes = alpha_nodes * float(args.mu_f) * (phi_nodes * phi_nodes) * kappa_inv_eff * g_perm_nodes
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

                # Damage transport sanity: d is a material/Eulerian field advected by vS,
                # so it should remain co-located with the moving biofilm (up to diffusion).
                # Print a simple alpha-weighted damage center-of-mass to spot "damage trails".
                if d_k is not None:
                    try:
                        w_d = np.maximum(d_nodes, 0.0) * np.maximum(alpha_nodes, 0.0)
                        wsum_d = float(np.sum(w_d))
                        if wsum_d > 0.0:
                            x_cm_d = float(np.sum(w_d * node_xy[:, 0]) / wsum_d)
                            y_cm_d = float(np.sum(w_d * node_xy[:, 1]) / wsum_d)
                            cm_msg += f"  d_cm=({x_cm_d:.5f},{y_cm_d:.5f})"
                    except Exception:
                        pass

                v_nodes = _vector_to_nodes(v_k)
                u_nodes = _vector_to_nodes(u_k)
                u_prev_nodes = _vector_to_nodes(u_prev)
                vS_nodes = (u_nodes - u_prev_nodes) / float(dt_val)
                p_nodes = _scalar_to_nodes(p_k)
                vx_mean = float(np.mean(v_nodes[mask, 0]))
                vSx_mean = float(np.mean(vS_nodes[mask, 0]))
                phi_min_bio = float(np.min(phi_nodes[mask]))
                phi_max_bio = float(np.max(phi_nodes[mask]))
                uy_mean = float(np.mean(u_nodes[mask, 1]))
                uy_min = float(np.min(u_nodes[mask, 1]))
                uy_max = float(np.max(u_nodes[mask, 1]))
                p_mean_bio = float(np.mean(p_nodes[mask]))
                p_min_bio = float(np.min(p_nodes[mask]))
                p_max_bio = float(np.max(p_nodes[mask]))
                mask_int = (alpha_nodes > 0.4) & (alpha_nodes < 0.6)
                vSx_int = float(np.mean(vS_nodes[mask_int, 0])) if np.any(mask_int) else float("nan")
                u_mag = np.linalg.norm(u_nodes[mask, :], axis=1)
                u_max = float(np.max(u_mag)) if u_mag.size else 0.0

                # Bounding box (in the fixed Eulerian grid) of the high-alpha region.
                y_hi_max = float("nan")
                y_hi_min = float("nan")
                if np.any(mask_dofs):
                    y_hi_min = float(np.min(alpha_dof_xy[mask_dofs, 1]))
                    y_hi_max = float(np.max(alpha_dof_xy[mask_dofs, 1]))

                # Velocity profile inside the porous/biofilm block: top should be faster than bottom.
                y_bio = node_xy[mask, 1]
                vx_bio = v_nodes[mask, 0]
                vx_top = float("nan")
                vx_bot = float("nan")
                if y_bio.size:
                    y0 = float(np.min(y_bio))
                    y1 = float(np.max(y_bio))
                    if y1 > y0 + 1.0e-12:
                        y_lo = y0 + 0.2 * (y1 - y0)
                        y_hi = y0 + 0.8 * (y1 - y0)
                        bot = y_bio <= y_lo
                        top = y_bio >= y_hi
                        if np.any(bot):
                            vx_bot = float(np.mean(vx_bio[bot]))
                        if np.any(top):
                            vx_top = float(np.mean(vx_bio[top]))

                print(
                    f"           beta[min,max]=[{bmin:.3e},{bmax:.3e}]  "
                    f"mean(vx|alpha>0.5)={vx_mean:.3e}  mean(vSx|alpha>0.5)={vSx_mean:.3e}  "
                    f"mean(vSx|0.4<alpha<0.6)={vSx_int:.3e}{cm_msg}"
                )
                print(
                    f"           u[max|alpha>0.5]={u_max:.3e}  "
                    f"uy[mean,min,max|alpha>0.5]=[{uy_mean:.3e},{uy_min:.3e},{uy_max:.3e}]  "
                    f"p_bio[mean|min,max]=[{p_mean_bio:.3e}|{p_min_bio:.3e},{p_max_bio:.3e}]  "
                    f"alpha_hi[ymin,ymax]=[{y_hi_min:.3e},{y_hi_max:.3e}]  "
                    f"vx_biofilm[bot,top]=[{vx_bot:.3e},{vx_top:.3e}]  "
                    f"phi_bio[min,max]=[{phi_min_bio:.3e},{phi_max_bio:.3e}]"
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
                    if (not slough_flags["wall_clear"]) and np.isfinite(y_hi_min):
                        # A more direct detachment indicator: the high-alpha region no longer touches the wall.
                        # Use a small fraction of eps (diffuse thickness) to avoid triggering from round-off.
                        y_thresh = 0.5 * float(getattr(args, "eps", 0.0) or 0.0)
                        if y_thresh > 0.0 and y_hi_min > y_thresh:
                            slough_flags["wall_clear"] = True
                            slough_t["wall_clear"] = float(t_now)
                            print(
                                f"           [sloughing] wall clearance: alpha_hi[ymin]={y_hi_min:.3e} > {y_thresh:.3e} (t={t_now:.3f})"
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
                kappa_inv_eff = float(args.kappa_inv)
                if str(getattr(args, "kappa_inv_model", "spatial")).strip().lower() in {"kozeny", "kozeny_carman", "kc"}:
                    phi_ref = float(getattr(args, "kappa_phi_ref", None) or args.phi_b)
                    eps_kc = 1.0e-12
                    g = ((1.0 - phi_nodes) ** 2) / (phi_nodes**3 + eps_kc)
                    g0 = ((1.0 - phi_ref) ** 2) / (phi_ref**3 + eps_kc)
                    kappa_inv_eff = float(args.kappa_inv) * (g / max(1.0e-30, g0))
                g_perm_nodes = 1.0
                if d_k is not None:
                    d_nodes = _scalar_to_nodes(d_k)
                    kappa_perm = float(getattr(args, "damage_kappa_perm", 1.0e-8) or 1.0e-8)
                    g_perm_nodes = (1.0 - kappa_perm) * ((1.0 - d_nodes) ** 2) + kappa_perm
                beta_nodes = alpha_nodes * float(args.mu_f) * (phi_nodes * phi_nodes) * kappa_inv_eff * g_perm_nodes
                u_nodes = _vector_to_nodes(u_k)
                u_prev_nodes = _vector_to_nodes(u_prev)
                vS_nodes = (u_nodes - u_prev_nodes) / float(dt_val)
                sigma_u, _w_sigma = solid_von_mises_mass_lumped_in_domain(
                    dof_handler=dh,
                    field="u_x",
                    u=u_k,
                    alpha=alpha_k,
                    dx_domain=dx_q,
                    mu_s=mu_s_c,
                    lambda_s=lambda_s_c,
                    solid_model=solid_model,
                    backend=backend,
                    quad_order=qdeg,
                )
                sigma_vm_nodes = np.zeros(num_nodes, dtype=float)
                sl_u = np.asarray(dh.get_field_slice("u_x"), dtype=int).ravel()
                for i, gdof in enumerate(sl_u):
                    _fld, node_id = dh._dof_to_node_map[int(gdof)]
                    if node_id is None:
                        continue
                    sigma_vm_nodes[int(node_id)] = float(sigma_u[i])
            except Exception:
                beta_nodes = None
                vS_nodes = None
                sigma_vm_nodes = None

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
                    "d": d_k if d_k is not None else (lambda x, y: 0.0),
                    "a": a_prev if a_prev is not None else (lambda x, y: float(a_state["val"])),
                    "S": S_k,
                    "X": X_k,
                    "beta": beta_nodes if beta_nodes is not None else (lambda x, y: 0.0),
                    "sigma_vm": sigma_vm_nodes if sigma_vm_nodes is not None else (lambda x, y: 0.0),
                },
            )

        # Full state dump for restart/offline analysis (written after updating a_prev / a_state).
        _dump_state_snapshot(
            step_no,
            float(t_now),
            float(dt_step),
            tag="step",
            funcs=[
                v_n,
                p_n,
                u_n,
                phi_n,
                alpha_n,
                d_n,
                S_n,
                X_n,
                u_prev,
                u_nm1,
                a_prev,
            ],
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

        # If requested, recompute alpha from the Eulerian reference map (χ = x - u).
        # Do this BEFORE promotion so alpha_n becomes the refmap-updated state.
        if bool(getattr(args, "alpha_from_refmap", False)):
            _update_alpha_from_refmap()
        _apply_alpha_volume_constraint()

        if not bool(args.no_clip):
            # Keep alpha/phi bounded so blended coefficients (rho, mu, beta, delta_eps(alpha)) stay physical.
            alpha_k.nodal_values[:] = np.clip(alpha_k.nodal_values, 0.0, 1.0)
            phi_k.nodal_values[:] = np.clip(phi_k.nodal_values, 0.0, 1.0)
            if d_k is not None:
                # Damage is bounded in [0,1]. NOTE: do NOT enforce Eulerian
                # pointwise irreversibility d^{k}(x) >= d^{n}(x) here: in an
                # Eulerian formulation the material can advect, so a fixed
                # spatial point may legitimately see d decrease as damaged
                # material moves away. Irreversibility (no healing) must be
                # enforced along material trajectories (or via a history
                # variable), not by a pointwise max in x.
                d_k.nodal_values[:] = np.clip(d_k.nodal_values, 0.0, 1.0)
            S_k.nodal_values[:] = np.maximum(S_k.nodal_values, 0.0)
            X_k.nodal_values[:] = np.maximum(X_k.nodal_values, 0.0)
            if a_prev is not None:
                a_prev.nodal_values[:] = np.clip(a_prev.nodal_values, 0.0, 1.0)

    solver.solve_time_interval(
        functions=[v_k, p_k, u_k, phi_k, alpha_k, *([d_k] if d_k is not None else []), S_k, X_k],
        prev_functions=[v_n, p_n, u_n, phi_n, alpha_n, *([d_n] if d_n is not None else []), S_n, X_n],
        aux_functions={
            "dt": dt_c,
            **({"a_prev": a_prev} if a_prev is not None else {}),
            **({"u_nm1": u_nm1} if u_nm1 is not None else {}),
        },
        time_params=TimeStepperParameters(
            dt=dt_val,
            final_time=float(args.t_final),
            max_steps=10_000,
            theta=theta,
            t0=float(restart_t0),
            step0=int(restart_step0),
            on_step_failure=_on_step_failure,
        ),
        post_step_refiner=post_step_refiner,
    )


if __name__ == "__main__":
    main()
