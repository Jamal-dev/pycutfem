"""FPI MMS Example 4.1 (paper) with two meshes and UFL-based nonmatching coupling.

This script mirrors the paper geometry (Fig. 3):
  - fluid mesh: rotated 45° square, CutFEM-truncated by x=x0 and by Ω^P,
  - poro mesh: rotated 30° body-fitted square (Ω^P),
and couples Γ^{FP} via `dNonmatchingInterface` so it runs on python/jit/cpp backends
without any hand-written interface kernels.

Notes
-----
* We integrate Γ^{FP} using CutFEM interface segments from the fluid mesh and
  evaluate porous fields by inverse-mapping into the owning porous element.
* The inlet traction at x=x0 is also integrated via `dNonmatchingInterface` over
  the CutFEM inlet segments.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet, MinLevelSet, RotatedBoxLevelSet, ScaledLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.nonmatching import build_composite_mesh
from pycutfem.nonmatching.cutfem_segments import build_interface_from_cutfem_segments
from pycutfem.nonmatching.interface import NonMatchingInterface
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import Constant, ElementWiseConstant, FacetNormal, Neg, Pos, dot, inner
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dNonmatchingInterface, dx
from pycutfem.utils.bitset import BitSet
from examples.utils.fpi.fully_eulerian import build_fpi_eulerian_forms
from examples.utils.fpi.mms_example41 import build_example41_mms
from examples.utils.fsi.fully_eulerian import build_measures, make_domain_sets
from pycutfem.utils.meshgen import structured_quad


def _approx_vinf(mms, bbox, n: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(int(seed))
    x0, x1, y0, y1 = map(float, bbox)
    X = np.empty((n, 2), dtype=float)
    X[:, 0] = rng.uniform(x0, x1, size=n)
    X[:, 1] = rng.uniform(y0, y1, size=n)
    V = np.asarray(mms.vF_k(X[:, 0], X[:, 1]), dtype=float)
    return float(np.max(np.linalg.norm(V, axis=1)))


@dataclass(frozen=True)
class TwoMeshUFLProblem:
    mesh: Mesh
    mesh_f: Mesh
    mesh_p: Mesh
    dh: DofHandler
    me: MixedElement
    fluid_ls: object
    poro_ls: object
    dx_f: object
    dx_p: object
    dGamma_fp: object
    dGamma_inlet: object
    dG_f: object
    dG_p: object
    h_gamma: object
    mapping: object


def _bitset_from_ids(mesh: Mesh, ids: np.ndarray) -> BitSet:
    nE = int(getattr(mesh, "n_elements", len(mesh.elements_list)))
    mask = np.zeros(nE, dtype=bool)
    mask[np.asarray(ids, dtype=int)] = True
    return BitSet(mask)


def _eoc(h0: float, h1: float, e0: float, e1: float) -> float:
    if not (h0 > 0.0 and h1 > 0.0 and e0 > 0.0 and e1 > 0.0):
        return float("nan")
    return float(np.log(e0 / e1) / np.log(h0 / h1))


def _interface_errors(
    *,
    dh: DofHandler,
    dGamma,
    vF_k,
    vP_k,
    uP_k,
    uP_n,
    dt,
    porosity,
    beta_BJ,
    vF_A,
    vP_A,
    uP_Ak,
    uP_An,
    quad_order: int,
    backend: str,
) -> dict[str, float]:
    """Kinematic interface errors En/Et (paper eq. (45)) on Γ^{FP}."""
    from pycutfem.ufl.expressions import FacetNormal, dot, inner
    from pycutfem.ufl.forms import Equation, assemble_form

    n = FacetNormal()

    u_dot = (uP_k - uP_n) / dt
    u_dot_A = (uP_Ak - uP_An) / dt

    kin_n_num = vF_k - u_dot - porosity * (vP_k - u_dot)
    kin_t_num = vF_k - u_dot - porosity * beta_BJ * (vP_k - u_dot)

    kin_n_A = vF_A - u_dot_A - porosity * (vP_A - u_dot_A)
    kin_t_A = vF_A - u_dot_A - porosity * beta_BJ * (vP_A - u_dot_A)

    diff_n = kin_n_num - kin_n_A
    diff_t = kin_t_num - kin_t_A

    proj_n = dot(diff_n, n) * n
    proj_t = diff_t - dot(diff_t, n) * n

    En2 = inner(proj_n, proj_n)
    Et2 = inner(proj_t, proj_t)

    hooks = {En2: {"name": "En2"}, Et2: {"name": "Et2"}}
    scalars = assemble_form(
        Equation(None, En2 * dGamma + Et2 * dGamma),
        dof_handler=dh,
        bcs=[],
        quad_order=int(quad_order),
        assembler_hooks=hooks,
        backend=str(backend),
    )
    en2 = float(np.asarray(scalars["En2"]).reshape(()))
    et2 = float(np.asarray(scalars["Et2"]).reshape(()))
    return {"En": float(math.sqrt(max(en2, 0.0))), "Et": float(math.sqrt(max(et2, 0.0)))}


def _l2_error_on_measure(
    *,
    dh: DofHandler,
    measure,
    backend: str,
    quad_order: int,
    hooks: dict,
):
    """Assemble scalar error integrals with `assembler_hooks` and return a dict of floats."""
    from pycutfem.ufl.forms import Equation, assemble_form

    # Build one integral per hooked expression (required by the hook matcher).
    L = None
    for e in hooks.keys():
        term = e * measure
        L = term if L is None else (L + term)

    scalars = assemble_form(
        Equation(None, L),
        dof_handler=dh,
        bcs=[],
        quad_order=int(quad_order),
        assembler_hooks={e: {"name": name} for e, name in hooks.items()},
        backend=str(backend),
    )
    out: dict[str, float] = {}
    for name in hooks.values():
        out[str(name)] = float(np.asarray(scalars[str(name)]).reshape(()))
    return out


def _interface_traction_pressure_errors(
    *,
    dh: DofHandler,
    dGamma,
    vF,
    pF,
    uP,
    pP,
    mu_f,
    c_nh,
    beta_nh,
    tractionF_A,
    tractionP_A,
    g_sigma,
    g_sigma_n,
    pA,
    quad_order: int,
    backend: str,
) -> dict[str, float]:
    """Traction and pressure errors on Γ^{FP} (debugging aid)."""
    from pycutfem.ufl.expressions import Constant, FacetNormal, Identity, dot, grad, inner
    from examples.utils.fpi.poro import poro_sigma_neo_hookean

    n = FacetNormal()  # (-)->(+) (poro -> fluid)
    nF = Constant(-1.0) * n  # fluid outward

    def _epsilon(v):
        return Constant(0.5) * (grad(v) + grad(v).T)

    I2 = Identity(2)
    sigmaF_h = -pF * I2 + Constant(2.0) * mu_f * _epsilon(vF)
    tractionF_h = dot(sigmaF_h, nF)

    sigmaP_h = poro_sigma_neo_hookean(uP, c_nh, beta_nh)
    tractionP_h = dot(sigmaP_h, nF)

    # traction mismatch (vector) and manufactured traction jump
    e_tF2 = inner(tractionF_h - tractionF_A, tractionF_h - tractionF_A)
    e_tP2 = inner(tractionP_h - tractionP_A, tractionP_h - tractionP_A)
    e_tjump2 = inner(tractionF_h - tractionP_h - g_sigma, tractionF_h - tractionP_h - g_sigma)

    # scalar manufactured normal-viscous traction (paper uses it via g_sigma_n)
    visc_n_h = dot(nF, tractionF_h) + pF
    e_gsig_n2 = (visc_n_h - g_sigma_n) * (visc_n_h - g_sigma_n)

    # pressures at the interface (both sides)
    e_pF2 = (pF - pA) * (pF - pA)
    e_pP2 = (pP - pA) * (pP - pA)

    hooks = {
        e_tF2: "e_tF2",
        e_tP2: "e_tP2",
        e_tjump2: "e_tjump2",
        e_gsig_n2: "e_gsig_n2",
        e_pF2: "e_pF2",
        e_pP2: "e_pP2",
    }
    ints = _l2_error_on_measure(
        dh=dh,
        measure=dGamma,
        backend=str(backend),
        quad_order=int(quad_order),
        hooks=hooks,
    )
    return {
        "err_tF": float(math.sqrt(max(float(ints["e_tF2"]), 0.0))),
        "err_tP": float(math.sqrt(max(float(ints["e_tP2"]), 0.0))),
        "err_tjump": float(math.sqrt(max(float(ints["e_tjump2"]), 0.0))),
        "err_gsigma_n": float(math.sqrt(max(float(ints["e_gsig_n2"]), 0.0))),
        "err_pF_G": float(math.sqrt(max(float(ints["e_pF2"]), 0.0))),
        "err_pP_G": float(math.sqrt(max(float(ints["e_pP2"]), 0.0))),
    }


def _mesh_tag_counts(mesh: Mesh) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in ("inside", "outside", "cut"):
        out[f"elem_{t}"] = int(mesh.element_bitset(t).cardinality())
    for t in ("ghost_pos", "ghost_neg", "ghost_both", "ghost", "interface"):
        out[f"edge_{t}"] = int(mesh.edge_bitset(t).cardinality())
    out["n_elements"] = int(getattr(mesh, "n_elements", len(mesh.elements_list)))
    out["n_edges"] = int(len(getattr(mesh, "edges_list", [])))
    return out


def _save_mesh_plots(*, prob: TwoMeshUFLProblem, nx: int, outdir: Path, prefix: str) -> None:
    """Save labeled mesh PNGs for debugging CutFEM tagging (per refinement)."""
    outdir.mkdir(parents=True, exist_ok=True)
    expected = [
        outdir / f"{prefix}_fluid_nx{nx}.png",
        outdir / f"{prefix}_fluid_nx{nx}_ghost.png",
        outdir / f"{prefix}_composite_nx{nx}.png",
        outdir / f"{prefix}_poro_nx{nx}.png",
    ]
    if all(p.exists() for p in expected):
        return
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from pycutfem.io.visualization import plot_mesh_2
    except Exception as exc:
        print(f"[warn] matplotlib unavailable; skipping mesh plots ({exc})")
        return

    stats_f = _mesh_tag_counts(prob.mesh_f)
    fig, ax = plt.subplots(figsize=(10.0, 9.0))
    plot_mesh_2(
        prob.mesh_f,
        level_set=prob.fluid_ls,
        show=False,
        ax=ax,
        plot_nodes=False,
        plot_interface=True,
        elem_tags=True,
        edge_colors=True,
    )
    ax.set_title(
        f"Fluid mesh nx={nx} | cut={stats_f['elem_cut']} | ghost={stats_f['edge_ghost']} | interface={stats_f['edge_interface']}"
    )
    fig.savefig(outdir / f"{prefix}_fluid_nx{nx}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.0, 9.0))
    plot_mesh_2(
        prob.mesh_f,
        level_set=prob.fluid_ls,
        show=False,
        ax=ax,
        plot_nodes=False,
        plot_interface=True,
        elem_tags=True,
        edge_colors=True,
        edge_filter=["ghost_pos", "ghost_neg", "ghost_both", "interface"],
    )
    ax.set_title(f"Fluid mesh nx={nx} | ghost/interface edges only")
    fig.savefig(outdir / f"{prefix}_fluid_nx{nx}_ghost.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    stats_c = _mesh_tag_counts(prob.mesh)
    fig, ax = plt.subplots(figsize=(11.0, 10.0))
    plot_mesh_2(
        prob.mesh,
        level_set=prob.fluid_ls,
        show=False,
        ax=ax,
        plot_nodes=False,
        plot_interface=True,
        elem_tags=True,
        edge_colors=True,
    )
    ax.set_title(
        f"Composite mesh nx={nx} | cut={stats_c['elem_cut']} | ghost={stats_c['edge_ghost']} | interface={stats_c['edge_interface']}"
    )
    fig.savefig(outdir / f"{prefix}_composite_nx{nx}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 9.0))
    plot_mesh_2(prob.mesh_p, level_set=prob.poro_ls, show=False, ax=ax, plot_nodes=False, plot_interface=False)
    ax.set_title(f"Poro mesh nx={max(1, int(nx) // 2)} (body-fitted)")
    fig.savefig(outdir / f"{prefix}_poro_nx{nx}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _run_one(
    *,
    nx: int,
    poly_order: int,
    qdeg: int,
    qerr: int,
    dt_val: float,
    backend: str,
    interface: str,
    x0: float,
    newton_it: int,
    newton_tol: float,
    outdir: Path | None = None,
    save_mesh: bool = False,
) -> dict[str, float]:
    # Build geometry + dofhandler
    prob = build_problem(nx=int(nx), poly_order=int(poly_order), qdeg=int(qdeg), x0=float(x0))

    if outdir is not None and bool(save_mesh):
        _save_mesh_plots(prob=prob, nx=int(nx), outdir=outdir, prefix=f"mesh_backend-{backend}")

    # Paper h: edge length of square quads
    h = 1.0 / float(nx)

    poro_ls = prob.poro_ls
    interface_name = "rotated_box"
    interface_params = (float(poro_ls.center[0]), float(poro_ls.center[1]), float(poro_ls.hx), float(poro_ls.hy), float(poro_ls.angle))

    interface = str(interface).strip().lower()
    if interface == "bj":
        beta_bj_val = 1.0
    elif interface == "bjs":
        beta_bj_val = 0.0
    else:
        raise ValueError(f"Unknown interface variant {interface!r}; expected 'bj' or 'bjs'.")

    mms = build_example41_mms(
        dt_val=float(dt_val),
        kinv_case="iso",
        t_prev=0.0,
        beta_BJ=float(beta_bj_val),
        interface=interface_name,
        interface_params=interface_params,
    )
    ana_deg = max(10, int(qdeg) + 4)

    # Physical parameters (match MMS builder)
    from pycutfem.ufl.expressions import Function, VectorFunction
    from pycutfem.ufl.spaces import FunctionSpace
    from pycutfem.ufl.expressions import TestFunction, TrialFunction, VectorTestFunction, VectorTrialFunction

    rho_f = Constant(1.0)
    mu_f = Constant(1.0)
    rho_s0_tilde = Constant(1.0)
    porosity = Constant(0.5)
    K = 0.10
    K_inv = Constant(((1.0 / K) * np.eye(2)).tolist(), dim=2)
    E = 1000.0
    nu = 0.30
    mu_s = E / (2.0 * (1.0 + nu))
    c_nh = Constant(mu_s / 2.0)
    beta_nh = Constant(nu / (1.0 - 2.0 * nu))
    beta_BJ = Constant(beta_bj_val)
    kappa = Constant(math.sqrt(K) / (1.0 * float(mu_f.value) * math.sqrt(float(porosity.value))))

    # Paper uses inverse penalties; forms use gamma itself.
    gamma_inv = 45.0
    gamma = 1.0 / float(gamma_inv)

    # Approximate ||v||_inf for paper penalty scaling.
    xy = np.asarray(prob.mesh.nodes_x_y_pos, dtype=float)
    bbox = (float(xy[:, 0].min()), float(xy[:, 0].max()), float(xy[:, 1].min()), float(xy[:, 1].max()))
    v_inf = _approx_vinf(mms, bbox=bbox)

    dh = prob.dh
    me = prob.me

    Vf = FunctionSpace(name="Vf", field_names=["v_pos_x", "v_pos_y"], dim=1)
    Vp = FunctionSpace(name="Vp", field_names=["v_neg_x", "v_neg_y"], dim=1)
    Up = FunctionSpace(name="Up", field_names=["u_neg_x", "u_neg_y"], dim=1)

    dvF = VectorTrialFunction(space=Vf, dof_handler=dh)
    dpF = TrialFunction(name="dpF", field_name="p_pos_", dof_handler=dh)
    dvP = VectorTrialFunction(space=Vp, dof_handler=dh)
    duP = VectorTrialFunction(space=Up, dof_handler=dh)
    dpP = TrialFunction(name="dpP", field_name="p_neg_", dof_handler=dh)

    vF_test = VectorTestFunction(space=Vf, dof_handler=dh)
    qF_test = TestFunction(name="qF", field_name="p_pos_", dof_handler=dh)
    vP_test = VectorTestFunction(space=Vp, dof_handler=dh)
    uP_test = VectorTestFunction(space=Up, dof_handler=dh)
    qP_test = TestFunction(name="qP", field_name="p_neg_", dof_handler=dh)

    vF_k = VectorFunction(name="vF_k", field_names=["v_pos_x", "v_pos_y"], dof_handler=dh)
    pF_k = Function(name="pF_k", field_name="p_pos_", dof_handler=dh)
    vP_k = VectorFunction(name="vP_k", field_names=["v_neg_x", "v_neg_y"], dof_handler=dh)
    uP_k = VectorFunction(name="uP_k", field_names=["u_neg_x", "u_neg_y"], dof_handler=dh)
    pP_k = Function(name="pP_k", field_name="p_neg_", dof_handler=dh)

    vF_n = VectorFunction(name="vF_n", field_names=["v_pos_x", "v_pos_y"], dof_handler=dh)
    pF_n = Function(name="pF_n", field_name="p_pos_", dof_handler=dh)
    vP_n = VectorFunction(name="vP_n", field_names=["v_neg_x", "v_neg_y"], dof_handler=dh)
    uP_n = VectorFunction(name="uP_n", field_names=["u_neg_x", "u_neg_y"], dof_handler=dh)
    uP_nm1 = VectorFunction(name="uP_nm1", field_names=["u_neg_x", "u_neg_y"], dof_handler=dh)
    pP_n = Function(name="pP_n", field_name="p_neg_", dof_handler=dh)

    # Initial conditions at t=0 (exact)
    vF_n.set_values_from_function(lambda x, y: mms.vF_n(x, y))
    vP_n.set_values_from_function(lambda x, y: mms.vP_n(x, y))
    uP_n.set_values_from_function(lambda x, y: mms.uP_n(x, y))
    pF_n.set_values_from_function(lambda x, y: mms.pF_n(x, y))
    pP_n.set_values_from_function(lambda x, y: mms.pP_n(x, y))
    uP_nm1.set_values_from_function(lambda x, y: mms.uP_n(x, y))

    # Initial guess at t=dt (exact)
    vF_k.set_values_from_function(lambda x, y: mms.vF_k(x, y))
    vP_k.set_values_from_function(lambda x, y: mms.vP_k(x, y))
    uP_k.set_values_from_function(lambda x, y: mms.uP_k(x, y))
    pF_k.set_values_from_function(lambda x, y: mms.pF_k(x, y))
    pP_k.set_values_from_function(lambda x, y: mms.pP_k(x, y))

    dtc = Constant(float(dt_val))

    # Manufactured interface data using discrete normals (Remark 8).
    n = FacetNormal()  # (-)->(+) (poro -> fluid) on Γ^{FP}; outside->fluid on inlet
    nF = Constant(-1.0) * n
    vF_A = Analytic(lambda x, y: mms.vF_k(x, y), degree=ana_deg)
    vP_A = Analytic(lambda x, y: mms.vP_k(x, y), degree=ana_deg)
    uP_Ak = Analytic(lambda x, y: mms.uP_k(x, y), degree=ana_deg)
    uP_An = Analytic(lambda x, y: mms.uP_n(x, y), degree=ana_deg)
    pP_A = Analytic(lambda x, y: mms.pP_k(x, y), degree=ana_deg)
    sigmaF_A = Analytic(lambda x, y: mms.sigmaF_k(x, y), degree=ana_deg)
    sigmaP_A = Analytic(lambda x, y: mms.sigmaP_k(x, y), degree=ana_deg)

    u_dot_A = (uP_Ak - uP_An) / dtc
    tractionF_A = dot(sigmaF_A, nF)
    g_sigma = dot(sigmaF_A - sigmaP_A, nF)
    g_sigma_n = dot(nF, tractionF_A) + pP_A
    g_n = vF_A - u_dot_A - porosity * (vP_A - u_dot_A)
    g_t = vF_A - u_dot_A - beta_BJ * porosity * (vP_A - u_dot_A) + kappa * tractionF_A

    forms = build_fpi_eulerian_forms(
        vF_k=vF_k,
        pF_k=pF_k,
        vP_k=vP_k,
        uP_k=uP_k,
        pP_k=pP_k,
        vF_n=vF_n,
        pF_n=pF_n,
        vP_n=vP_n,
        uP_n=uP_n,
        pP_n=pP_n,
        uP_nm1=uP_nm1,
        dvF=dvF,
        dpF=dpF,
        dvP=dvP,
        duP=duP,
        dpP=dpP,
        vF_test=vF_test,
        qF_test=qF_test,
        vP_test=vP_test,
        uP_test=uP_test,
        qP_test=qP_test,
        dx_f=prob.dx_f,
        dx_p=prob.dx_p,
        dGamma=prob.dGamma_fp,
        dG_f=prob.dG_f,
        dG_p=prob.dG_p,
        level_set=prob.fluid_ls,
        dt=dtc,
        theta=1.0,
        rho_f=rho_f,
        mu_f=mu_f,
        rho_s0_tilde=rho_s0_tilde,
        porosity=porosity,
        K_inv=K_inv,
        c_nh=c_nh,
        beta_nh=beta_nh,
        beta_BJ=beta_BJ,
        kappa=kappa,
        gamma_n=Constant(gamma),
        gamma_t=Constant(gamma),
        zeta=1.0,
        gamma_F_p=0.0,
        gamma_F_gp=0.0,
        gamma_P_p=0.0,
        gamma_P_gp=0.0,
        g_sigma=g_sigma,
        g_sigma_n=g_sigma_n,
        g_n=g_n,
        g_t=g_t,
        use_paper_phi_gamma=True,
        use_paper_stabilization=True,
        poly_order=int(poly_order),
        vF_inf=v_inf,
        c_v_gamma=1.0 / 6.0,
        c_t_gamma=1.0 / 12.0,
        h_gamma=prob.h_gamma,
        use_interface_terms=True,
        use_stabilization=True,
    )

    # Forcing terms (volume)
    fF = Analytic(lambda x, y: mms.fF(x, y), degree=ana_deg)
    fD = Analytic(lambda x, y: mms.fD(x, y), degree=ana_deg)
    fS = Analytic(lambda x, y: mms.fS(x, y), degree=ana_deg)
    f_mass = Analytic(lambda x, y: mms.f_mass(x, y), degree=ana_deg)

    residual_form = (
        forms.residual_form
        - dot(fF, vF_test) * prob.dx_f
        - dot(fD, vP_test) * prob.dx_p
        - dot(fS, uP_test) * prob.dx_p
        - f_mass * qP_test * prob.dx_p
    )

    # Inlet traction: add -∫ (σ^F_A n^F) · w_F ds on the fluid side.
    traction_in = dot(sigmaF_A, nF)
    residual_form = residual_form - inner(traction_in, Pos(vF_test)) * prob.dGamma_inlet

    # Robust Dirichlet DOF tagging on the rotated outer boundary:
    r = math.sqrt(2.0) / 2.0

    def _outer_dof_locator(x: float, y: float) -> bool:
        x = float(x)
        y = float(y)
        if x < float(x0) - 1.0e-12:
            return False
        return abs(abs(x) + abs(y) - r) <= 5.0e-10

    dirichlet_dof_tag = "outer_dirichlet_dofs"
    dh.tag_dofs_by_locator_map({dirichlet_dof_tag: _outer_dof_locator}, fields=["v_pos_x", "v_pos_y"])

    # Pressure pinning (one DOF per pressure field) to remove nullspaces.
    dh.dof_tags.setdefault("p_pos_pin", set())
    dh.dof_tags.setdefault("p_neg_pin", set())
    for fld, tag_name in (("p_pos_", "p_pos_pin"), ("p_neg_", "p_neg_pin")):
        sl = np.asarray(dh.get_field_slice(fld), dtype=int)
        if sl.size == 0:
            continue
        inactive = set(getattr(dh, "dof_tags", {}).get("inactive", set()))
        for gd in sl.tolist():
            if int(gd) not in inactive:
                dh.dof_tags[tag_name].add(int(gd))
                break

    bcs = [
        BoundaryCondition("v_pos_x", "dirichlet", dirichlet_dof_tag, mms.vF_x),
        BoundaryCondition("v_pos_y", "dirichlet", dirichlet_dof_tag, mms.vF_y),
        BoundaryCondition("p_pos_", "dirichlet", "p_pos_pin", mms.pF_s),
        BoundaryCondition("p_neg_", "dirichlet", "p_neg_pin", mms.pP_s),
    ]
    bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]

    solver = NewtonSolver(
        residual_form,
        forms.jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=float(newton_tol), max_newton_iter=int(newton_it)),
        quad_order=int(qdeg),
        backend=str(backend),
    )

    solver.solve_time_interval(
        functions=[vF_k, pF_k, vP_k, uP_k, pP_k],
        prev_functions=[vF_n, pF_n, vP_n, uP_n, pP_n],
        aux_functions={"dt": dtc, "uP_nm1": uP_nm1},
        time_params=TimeStepperParameters(dt=float(dt_val), final_time=float(dt_val), max_steps=1),
    )

    # --- Error metrics (Fig. 5) ---
    # Fluid L2/H1 on Ω^F: use compiled side-norms (fast for convergence studies).
    err_vF = dh.l2_error_on_side_compiled(
        functions=vF_k,
        exact={"v_pos_x": mms.vF_x, "v_pos_y": mms.vF_y},
        level_set=prob.fluid_ls,
        side="+",
        quad_order=int(qerr),
        relative=False,
        backend=str(backend),
    )
    err_pF = dh.l2_error_on_side_compiled(
        functions=pF_k,
        exact={"p_pos_": mms.pF_s},
        level_set=prob.fluid_ls,
        side="+",
        quad_order=int(qerr),
        relative=False,
        backend=str(backend),
    )
    err_gvF = dh.h1_error_vector_on_side_compiled(
        vF_k,
        mms.grad_vF_k,
        prob.fluid_ls,
        side="+",
        quad_order=int(qerr),
        relative=False,
        backend=str(backend),
    )

    # Poro L2/H1 on Ω^P: integrate on the neg-component element set (body-fitted).
    vP_exact = Analytic(lambda x, y: mms.vP_k(x, y), degree=ana_deg)
    uP_exact = Analytic(lambda x, y: mms.uP_k(x, y), degree=ana_deg)
    pP_exact = Analytic(lambda x, y: mms.pP_k(x, y), degree=ana_deg)

    # L2 error integrals
    evP2 = inner(vP_k - vP_exact, vP_k - vP_exact)
    euP2 = inner(uP_k - uP_exact, uP_k - uP_exact)
    epP2 = (pP_k - pP_exact) * (pP_k - pP_exact)

    # H1-seminorm error for uP (componentwise).
    from pycutfem.ufl.expressions import grad

    def _G_u(x, y):
        return np.asarray(mms.grad_uP_k(x, y), dtype=float)

    g00 = Analytic(lambda x, y: _G_u(x, y)[..., 0, 0], degree=ana_deg)
    g01 = Analytic(lambda x, y: _G_u(x, y)[..., 0, 1], degree=ana_deg)
    g10 = Analytic(lambda x, y: _G_u(x, y)[..., 1, 0], degree=ana_deg)
    g11 = Analytic(lambda x, y: _G_u(x, y)[..., 1, 1], degree=ana_deg)
    du0x = grad(uP_k[0])[0] - g00
    du0y = grad(uP_k[0])[1] - g01
    du1x = grad(uP_k[1])[0] - g10
    du1y = grad(uP_k[1])[1] - g11
    guP2 = du0x * du0x + du0y * du0y + du1x * du1x + du1y * du1y

    # Assemble scalars on Ω^P
    hooks = {evP2: "evP2", euP2: "euP2", epP2: "epP2", guP2: "guP2"}
    ints = _l2_error_on_measure(dh=dh, measure=prob.dx_p, backend=str(backend), quad_order=int(qerr), hooks=hooks)
    err_vP = float(math.sqrt(max(ints["evP2"], 0.0)))
    err_uP = float(math.sqrt(max(ints["euP2"], 0.0)))
    err_pP = float(math.sqrt(max(ints["epP2"], 0.0)))
    err_guP = float(math.sqrt(max(ints["guP2"], 0.0)))

    # Interface kinematic errors En/Et on Γ^{FP}
    ifc = _interface_errors(
        dh=dh,
        dGamma=prob.dGamma_fp,
        vF_k=Pos(vF_k),
        vP_k=Neg(vP_k),
        uP_k=Neg(uP_k),
        uP_n=Neg(uP_n),
        dt=dtc,
        porosity=porosity,
        beta_BJ=beta_BJ,
        vF_A=vF_A,
        vP_A=vP_A,
        uP_Ak=uP_Ak,
        uP_An=uP_An,
        quad_order=int(qerr),
        backend=str(backend),
    )

    # Interface traction/pressure diagnostics on Γ^{FP} (helps debugging Nitsche terms).
    tractionP_A = dot(sigmaP_A, nF)
    ifc_tr = _interface_traction_pressure_errors(
        dh=dh,
        dGamma=prob.dGamma_fp,
        vF=Pos(vF_k),
        pF=Pos(pF_k),
        uP=Neg(uP_k),
        pP=Neg(pP_k),
        mu_f=mu_f,
        c_nh=c_nh,
        beta_nh=beta_nh,
        tractionF_A=tractionF_A,
        tractionP_A=tractionP_A,
        g_sigma=g_sigma,
        g_sigma_n=g_sigma_n,
        pA=pP_A,
        quad_order=int(qerr),
        backend=str(backend),
    )

    row = dict(
        nx=float(nx),
        h=float(h),
        err_vF=float(err_vF),
        err_pF=float(err_pF),
        err_vP=float(err_vP),
        err_uP=float(err_uP),
        err_pP=float(err_pP),
        err_gvF=float(err_gvF),
        err_guP=float(err_guP),
        En=float(ifc["En"]),
        Et=float(ifc["Et"]),
        err_tF=float(ifc_tr["err_tF"]),
        err_tP=float(ifc_tr["err_tP"]),
        err_tjump=float(ifc_tr["err_tjump"]),
        err_gsigma_n=float(ifc_tr["err_gsigma_n"]),
        err_pF_G=float(ifc_tr["err_pF_G"]),
        err_pP_G=float(ifc_tr["err_pP_G"]),
    )

    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        np.save(outdir / f"row_nx{nx}_{interface}_{backend}.npy", row)

    return row


def build_problem(*, nx: int, poly_order: int, qdeg: int, x0: float) -> TwoMeshUFLProblem:
    # --- geometry level sets (paper sec. 4.1) ---
    poro_ls = RotatedBoxLevelSet(center=(0.0, 0.0), hx=0.25, hy=0.25, angle=math.pi / 6.0)  # Ω^P (30°)

    # Truncation: keep {x >= x0}. `cut_ls` is negative on Ω^F.
    cut_ls = AffineLevelSet(-1.0, 0.0, float(x0))  # φ = x0 - x
    cut_pos = ScaledLevelSet(-1.0, cut_ls)  # positive on {x >= x0}
    fluid_ls = MinLevelSet(poro_ls, cut_pos)  # positive in Ω^F

    # --- fluid mesh: rotated 45° square mesh of size 1 ---
    nodes_f, elems_f, edges_f, corners_f = structured_quad(
        1.0,
        1.0,
        nx=int(nx),
        ny=int(nx),
        poly_order=int(poly_order),
        offset=(-0.5, -0.5),
        rotation=math.pi / 4.0,
        rotation_center=(0.0, 0.0),
    )
    mesh_f = Mesh(nodes_f, elems_f, edges_f, corners_f, element_type="quad", poly_order=int(poly_order))
    mesh_f.classify_elements(fluid_ls)
    mesh_f.classify_edges(fluid_ls)
    mesh_f.build_interface_segments(fluid_ls)

    # --- poro mesh: rotated 30° square mesh of size 0.5 ---
    nx_p = max(1, int(nx) // 2)
    nodes_p, elems_p, edges_p, corners_p = structured_quad(
        0.5,
        0.5,
        nx=nx_p,
        ny=nx_p,
        poly_order=int(poly_order),
        offset=(-0.25, -0.25),
        rotation=math.pi / 6.0,
        rotation_center=(0.0, 0.0),
    )
    mesh_p = Mesh(nodes_p, elems_p, edges_p, corners_p, element_type="quad", poly_order=int(poly_order))
    if hasattr(mesh_p, "build_grid_search"):
        mesh_p.build_grid_search()

    # Build CutFEM segments for Γ^{FP} and Γ^{F,N} on the *fluid* mesh.
    iface_fp, inlet = build_interface_from_cutfem_segments(
        mesh_f=mesh_f,
        fluid_ls=fluid_ls,
        poro_ls=poro_ls,
        mesh_p=mesh_p,
        x0=float(x0),
    )

    # Composite mesh: pos=fluid, neg=poro.
    mapping = build_composite_mesh(mesh_pos=mesh_f, mesh_neg=mesh_p, order="pos_neg")
    mesh = mapping.mesh

    # Re-classify composite mesh for CutFEM fluid volume integrals and inactive tagging.
    mesh.classify_elements(fluid_ls)
    mesh.classify_edges(fluid_ls)
    mesh.build_interface_segments(fluid_ls)
    domains_fluid = make_domain_sets(mesh, use_aligned_interface=False)
    dx_f, _dx_inactive, _dGamma_unused, dG_f, dG_p = build_measures(mesh, fluid_ls, domains_fluid, qvol=int(qdeg))

    # Poro volume measure on the neg component (body-fitted).
    bs_poro = _bitset_from_ids(mesh, mapping.neg_elem_ids)
    dx_p = dx(defined_on=bs_poro, metadata={"q": int(qdeg)})

    # Build nonmatching interfaces on the composite numbering.
    n_fp = int(iface_fp.n_segments())
    iface_fp_c = NonMatchingInterface(
        mesh_neg=mesh,
        mesh_pos=mesh,
        neg_edge_ids=np.zeros(n_fp, dtype=int),
        pos_edge_ids=np.zeros(n_fp, dtype=int),
        neg_elem_ids=np.asarray(iface_fp.neg_elem_ids, dtype=int) + int(mapping.neg_elem_offset),
        pos_elem_ids=np.asarray(iface_fp.pos_elem_ids, dtype=int) + int(mapping.pos_elem_offset),
        P0=np.asarray(iface_fp.P0, dtype=float),
        P1=np.asarray(iface_fp.P1, dtype=float),
        n=np.asarray(iface_fp.n, dtype=float),
        h_neg=np.asarray(iface_fp.h_neg, dtype=float),
        h_pos=np.asarray(iface_fp.h_pos, dtype=float),
    )

    # Inlet as a one-sided nonmatching interface: neg and pos point to the same
    # fluid element id, but `n` is oriented outside->fluid so nF=-n is outward.
    inlet_P0 = np.asarray(inlet.get("inlet_P0", np.zeros((0, 2), float)), dtype=float)
    inlet_P1 = np.asarray(inlet.get("inlet_P1", np.zeros((0, 2), float)), dtype=float)
    inlet_pos = np.asarray(inlet.get("inlet_pos_elem_ids", np.zeros((0,), int)), dtype=int) + int(mapping.pos_elem_offset)
    n_in = int(inlet_pos.size)
    inlet_iface = NonMatchingInterface(
        mesh_neg=mesh,
        mesh_pos=mesh,
        neg_edge_ids=np.zeros(n_in, dtype=int),
        pos_edge_ids=np.zeros(n_in, dtype=int),
        neg_elem_ids=np.asarray(inlet_pos, dtype=int),
        pos_elem_ids=np.asarray(inlet_pos, dtype=int),
        P0=inlet_P0,
        P1=inlet_P1,
        n=np.tile(np.array([[1.0, 0.0]], dtype=float), (n_in, 1)),
        h_neg=np.ones(n_in, dtype=float),
        h_pos=np.ones(n_in, dtype=float),
    )

    # Measures on Γ^{FP} and Γ^{F,N}
    dGamma_fp = dNonmatchingInterface(metadata={"q": int(qdeg) + 2, "interface": iface_fp_c})
    dGamma_inlet = dNonmatchingInterface(metadata={"q": int(qdeg) + 2, "interface": inlet_iface})

    # MixedElement and handler on composite mesh
    me = MixedElement(
        mesh,
        field_specs={
            "v_pos_x": int(poly_order),
            "v_pos_y": int(poly_order),
            "p_pos_": int(poly_order),
            "v_neg_x": int(poly_order),
            "v_neg_y": int(poly_order),
            "u_neg_x": int(poly_order),
            "u_neg_y": int(poly_order),
            "p_neg_": int(poly_order),
        },
    )
    dh = DofHandler(me, method="cg")

    # Inactive DOFs: fluid fields are inactive on "inside" elements of fluid_ls.
    dh.dof_tags["inactive"] = set()
    for fld in ("v_pos_x", "v_pos_y", "p_pos_"):
        dh.tag_dofs_from_element_bitset("inactive", fld, "inside", strict=True)

    # Inactive DOFs: poro fields are inactive on the fluid component elements.
    pos_mask = np.zeros(int(mesh.n_elements), dtype=bool)
    pos_mask[np.asarray(mapping.pos_elem_ids, dtype=int)] = True
    for fld in ("v_neg_x", "v_neg_y", "u_neg_x", "u_neg_y", "p_neg_"):
        dh.tag_dofs_from_element_bitset("inactive", fld, pos_mask, strict=True)

    # Interface mesh size parameter h_Γ: area/segment-length on cut cells of Γ^{FP}.
    areas = np.asarray(getattr(mesh, "areas_list", None), dtype=float)
    if areas.size != len(mesh.elements_list):
        areas = np.asarray(mesh.areas(), dtype=float)
    h_gamma_vals = np.sqrt(np.maximum(areas, 0.0))
    by_elem: dict[int, float] = {}
    for eid, p0, p1 in zip(np.asarray(iface_fp_c.pos_elem_ids, int), np.asarray(iface_fp_c.P0, float), np.asarray(iface_fp_c.P1, float)):
        by_elem[int(eid)] = float(by_elem.get(int(eid), 0.0) + np.linalg.norm(np.asarray(p1, float) - np.asarray(p0, float)))
    for eid, seg_len in by_elem.items():
        if seg_len > 1.0e-14:
            h_gamma_vals[int(eid)] = float(areas[int(eid)]) / float(seg_len)
    h_gamma = ElementWiseConstant(np.asarray(h_gamma_vals, dtype=float))

    return TwoMeshUFLProblem(
        mesh=mesh,
        mesh_f=mesh_f,
        mesh_p=mesh_p,
        dh=dh,
        me=me,
        fluid_ls=fluid_ls,
        poro_ls=poro_ls,
        dx_f=dx_f,
        dx_p=dx_p,
        dGamma_fp=dGamma_fp,
        dGamma_inlet=dGamma_inlet,
        dG_f=dG_f,
        dG_p=dG_p,
        h_gamma=h_gamma,
        mapping=mapping,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--q", type=int, default=6)
    parser.add_argument("--q-error", type=int, default=0, help="Quadrature order for error norms (0 -> max(q, 10)).")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--x0", type=float, default=-0.45)
    parser.add_argument("--backend", type=str, default="jit", choices=["python", "jit", "cpp"])
    parser.add_argument("--newton-it", type=int, default=20)
    parser.add_argument("--newton-tol", type=float, default=1e-10)
    parser.add_argument("--interface", type=str, default="bj", choices=["bj", "bjs", "both"])
    parser.add_argument("--convergence", action="store_true", help="Run an h-refinement study (prints a table + saves plots).")
    parser.add_argument("--levels", type=int, default=5, help="Number of h-refinement levels (convergence mode).")
    parser.add_argument("--nx-list", type=str, default="", help="In convergence mode: comma-separated nx list (overrides --nx/--levels).")
    parser.add_argument(
        "--paper-h-range",
        action="store_true",
        help="In convergence mode: use the paper's h-range [0.25, 0.00390625] with 12 points (nx=[4,6,8,12,16,24,32,48,64,96,128,256]).",
    )
    parser.add_argument("--outdir", type=str, default="examples/FPI/_two_mesh_convergence")
    parser.add_argument("--save-mesh", action="store_true", help="Save labeled mesh plots per refinement (PNG) into --outdir.")
    args = parser.parse_args()

    nx = int(args.nx)
    p = int(args.p)
    qdeg = int(args.q)
    qerr = int(args.q_error) if int(args.q_error) > 0 else max(int(qdeg), 10)
    dt0 = float(args.dt)

    if args.convergence:
        outdir = Path(str(args.outdir))
        if str(args.nx_list).strip():
            nx_list = [int(v.strip()) for v in str(args.nx_list).split(",") if v.strip()]
        elif bool(args.paper_h_range):
            nx_list = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256]
        else:
            nx0 = int(nx)
            levels = int(args.levels)
            nx_list = [int(nx0 * (2**k)) for k in range(max(1, levels))]

        iface_list = [str(args.interface)] if str(args.interface) in {"bj", "bjs"} else ["bj", "bjs"]

        rows_by_iface: dict[str, list[dict[str, float]]] = {}
        for iface in iface_list:
            rows = []
            for nx_i in nx_list:
                rows.append(
                    _run_one(
                        nx=int(nx_i),
                        poly_order=p,
                        qdeg=qdeg,
                        qerr=qerr,
                        dt_val=dt0,
                        backend=str(args.backend),
                        interface=str(iface),
                        x0=float(args.x0),
                        newton_it=int(args.newton_it),
                        newton_tol=float(args.newton_tol),
                        outdir=outdir,
                        save_mesh=bool(args.save_mesh),
                    )
                )
            rows_by_iface[iface] = rows

            # Print table
            print(f"\nFPI Example 4.1 (two-mesh, BE) | backend={args.backend} | p={p} | interface={iface.upper()}")
            print(
                f"{'h':>10} |e(vF)|    eoc |e(pF)|    eoc |e(vP)|    eoc |e(uP)|    eoc |e(pP)|    eoc |En    eoc |Et    eoc"
            )
            prev = None
            for r in rows:
                if prev is None:
                    print(
                        f"{r['h']:10.3e}  {r['err_vF']:9.3e}   -   {r['err_pF']:9.3e}   -   {r['err_vP']:9.3e}   -   "
                        f"{r['err_uP']:9.3e}   -   {r['err_pP']:9.3e}   -   {r['En']:6.3e}  -   {r['Et']:6.3e}  -"
                    )
                else:
                    def fmt(x):
                        return f"{x:4.2f}" if np.isfinite(x) else "  - "

                    e_vf = _eoc(prev["h"], r["h"], prev["err_vF"], r["err_vF"])
                    e_pf = _eoc(prev["h"], r["h"], prev["err_pF"], r["err_pF"])
                    e_vp = _eoc(prev["h"], r["h"], prev["err_vP"], r["err_vP"])
                    e_up = _eoc(prev["h"], r["h"], prev["err_uP"], r["err_uP"])
                    e_pp = _eoc(prev["h"], r["h"], prev["err_pP"], r["err_pP"])
                    e_en = _eoc(prev["h"], r["h"], prev["En"], r["En"])
                    e_et = _eoc(prev["h"], r["h"], prev["Et"], r["Et"])
                    print(
                        f"{r['h']:10.3e}  {r['err_vF']:9.3e} {fmt(e_vf)} {r['err_pF']:9.3e} {fmt(e_pf)} {r['err_vP']:9.3e} {fmt(e_vp)} "
                        f"{r['err_uP']:9.3e} {fmt(e_up)} {r['err_pP']:9.3e} {fmt(e_pp)} {r['En']:6.3e} {fmt(e_en)} {r['Et']:6.3e} {fmt(e_et)}"
                    )
                prev = r

        # Save log-log plots for Fig. 5-style diagnostics.
        try:
            import matplotlib.pyplot as plt

            plot_specs = [
                ("err_vF", "e(vF)", 2.0),
                ("err_pF", "e(pF)", 2.0),
                ("err_vP", "e(vP)", 2.0),
                ("err_pP", "e(pP)", 2.0),
                ("err_uP", "e(uP)", 2.0),
                ("err_gvF", "e(grad vF)", 1.0),
                ("err_guP", "e(grad uP)", 1.0),
                ("En", "En", 2.0),
                ("Et", "Et", 2.0),
                ("err_tjump", "e(traction jump)", 1.0),
                ("err_pF_G", "e(pF|Γ)", 2.0),
                ("err_pP_G", "e(pP|Γ)", 2.0),
            ]

            outdir.mkdir(parents=True, exist_ok=True)
            colors = {"bj": "C0", "bjs": "C1"}
            for key, ylabel, slope_ref in plot_specs:
                plt.figure(figsize=(6.0, 4.0))
                for iface, rows in rows_by_iface.items():
                    hs = np.array([r["h"] for r in rows], dtype=float)
                    ys = np.array([r[key] for r in rows], dtype=float)
                    plt.loglog(hs, ys, "o-", label=iface.upper(), color=colors.get(iface, None))

                # Reference slope line anchored at the first BJ point.
                rows0 = rows_by_iface.get("bj", next(iter(rows_by_iface.values())))
                h0 = float(rows0[0]["h"])
                y0 = float(rows0[0][key])
                href = np.array([float(np.min([r["h"] for r in rows0])), float(np.max([r["h"] for r in rows0]))], dtype=float)
                href.sort()
                if h0 > 0.0 and y0 > 0.0:
                    yref = y0 * (href / h0) ** float(slope_ref)
                    plt.loglog(href, yref, "--", color="0.35", alpha=0.7, label=f"h^{slope_ref:g} ref")

                # Special: tangential error Et differs (paper): BJ ~ h^{3/2}, BJS ~ h^2.
                if key == "Et" and "bj" in rows_by_iface and "bjs" in rows_by_iface:
                    h0_bj = float(rows_by_iface["bj"][0]["h"])
                    y0_bj = float(rows_by_iface["bj"][0]["Et"])
                    yref_bj = y0_bj * (href / h0_bj) ** 1.5
                    plt.loglog(href, yref_bj, "-.", color="0.35", alpha=0.7, label="BJ h^{3/2} ref")
                    h0_bjs = float(rows_by_iface["bjs"][0]["h"])
                    y0_bjs = float(rows_by_iface["bjs"][0]["Et"])
                    yref_bjs = y0_bjs * (href / h0_bjs) ** 2.0
                    plt.loglog(href, yref_bjs, ":", color="0.35", alpha=0.7, label="BJS h^2 ref")

                plt.gca().invert_xaxis()
                plt.xlabel("h")
                plt.ylabel(ylabel)
                plt.title(f"Two-mesh Example 4.1 {ylabel} | backend={args.backend} | p={p}")
                plt.grid(True, which="both", ls=":")
                plt.legend()
                outpath = outdir / f"two_mesh_example41_{key}_backend-{args.backend}_p{p}_iface-{args.interface}.png"
                plt.tight_layout()
                plt.savefig(outpath, dpi=200)
                plt.close()
                print(f"Saved plot: {outpath}")
        except Exception as e:
            print(f"Plotting skipped: {e}")

        return

    prob = build_problem(nx=nx, poly_order=p, qdeg=qdeg, x0=float(args.x0))

    # MMS (use continuous interface only for analytic fields; jumps use *discrete* normals).
    poro_ls = prob.poro_ls
    interface_name = "rotated_box"
    interface_params = (float(poro_ls.center[0]), float(poro_ls.center[1]), float(poro_ls.hx), float(poro_ls.hy), float(poro_ls.angle))
    iface = str(args.interface).strip().lower()
    if iface not in {"bj", "bjs"}:
        iface = "bj"
    beta_bj_val = 1.0 if iface == "bj" else 0.0
    mms = build_example41_mms(
        dt_val=dt0,
        kinv_case="iso",
        t_prev=0.0,
        beta_BJ=float(beta_bj_val),
        interface=interface_name,
        interface_params=interface_params,
    )
    ana_deg = max(10, qdeg + 4)

    # Physical parameters (match MMS builder)
    rho_f = Constant(1.0)
    mu_f = Constant(1.0)
    rho_s0_tilde = Constant(1.0)
    porosity = Constant(0.5)
    K = 0.10
    K_inv = Constant(((1.0 / K) * np.eye(2)).tolist(), dim=2)
    E = 1000.0
    nu = 0.30
    mu_s = E / (2.0 * (1.0 + nu))
    c_nh = Constant(mu_s / 2.0)
    beta_nh = Constant(nu / (1.0 - 2.0 * nu))
    beta_BJ = Constant(float(beta_bj_val))
    kappa = Constant(math.sqrt(K) / (1.0 * float(mu_f.value) * math.sqrt(float(porosity.value))))

    # Paper uses inverse penalties; forms use gamma itself.
    gamma_inv = 45.0
    gamma = 1.0 / float(gamma_inv)

    # Approximate ||v||_inf for paper penalty scaling.
    xy = np.asarray(prob.mesh.nodes_x_y_pos, dtype=float)
    bbox = (float(xy[:, 0].min()), float(xy[:, 0].max()), float(xy[:, 1].min()), float(xy[:, 1].max()))
    v_inf = _approx_vinf(mms, bbox=bbox)

    # --- Unknowns (current and previous) on the composite mesh ---
    from pycutfem.ufl.expressions import Function, VectorFunction
    from pycutfem.ufl.spaces import FunctionSpace
    from pycutfem.ufl.expressions import TestFunction, TrialFunction, VectorTestFunction, VectorTrialFunction

    dh = prob.dh
    me = prob.me

    Vf = FunctionSpace(name="Vf", field_names=["v_pos_x", "v_pos_y"], dim=1)
    Vp = FunctionSpace(name="Vp", field_names=["v_neg_x", "v_neg_y"], dim=1)
    Up = FunctionSpace(name="Up", field_names=["u_neg_x", "u_neg_y"], dim=1)

    dvF = VectorTrialFunction(space=Vf, dof_handler=dh)
    dpF = TrialFunction(name="dpF", field_name="p_pos_", dof_handler=dh)
    dvP = VectorTrialFunction(space=Vp, dof_handler=dh)
    duP = VectorTrialFunction(space=Up, dof_handler=dh)
    dpP = TrialFunction(name="dpP", field_name="p_neg_", dof_handler=dh)

    vF_test = VectorTestFunction(space=Vf, dof_handler=dh)
    qF_test = TestFunction(name="qF", field_name="p_pos_", dof_handler=dh)
    vP_test = VectorTestFunction(space=Vp, dof_handler=dh)
    uP_test = VectorTestFunction(space=Up, dof_handler=dh)
    qP_test = TestFunction(name="qP", field_name="p_neg_", dof_handler=dh)

    vF_k = VectorFunction(name="vF_k", field_names=["v_pos_x", "v_pos_y"], dof_handler=dh)
    pF_k = Function(name="pF_k", field_name="p_pos_", dof_handler=dh)
    vP_k = VectorFunction(name="vP_k", field_names=["v_neg_x", "v_neg_y"], dof_handler=dh)
    uP_k = VectorFunction(name="uP_k", field_names=["u_neg_x", "u_neg_y"], dof_handler=dh)
    pP_k = Function(name="pP_k", field_name="p_neg_", dof_handler=dh)

    vF_n = VectorFunction(name="vF_n", field_names=["v_pos_x", "v_pos_y"], dof_handler=dh)
    pF_n = Function(name="pF_n", field_name="p_pos_", dof_handler=dh)
    vP_n = VectorFunction(name="vP_n", field_names=["v_neg_x", "v_neg_y"], dof_handler=dh)
    uP_n = VectorFunction(name="uP_n", field_names=["u_neg_x", "u_neg_y"], dof_handler=dh)
    uP_nm1 = VectorFunction(name="uP_nm1", field_names=["u_neg_x", "u_neg_y"], dof_handler=dh)
    pP_n = Function(name="pP_n", field_name="p_neg_", dof_handler=dh)

    # Initial conditions at t=0 (exact)
    vF_n.set_values_from_function(lambda x, y: mms.vF_n(x, y))
    vP_n.set_values_from_function(lambda x, y: mms.vP_n(x, y))
    uP_n.set_values_from_function(lambda x, y: mms.uP_n(x, y))
    pF_n.set_values_from_function(lambda x, y: mms.pF_n(x, y))
    pP_n.set_values_from_function(lambda x, y: mms.pP_n(x, y))
    uP_nm1.set_values_from_function(lambda x, y: mms.uP_n(x, y))

    # Initial guess at t=dt (exact)
    vF_k.set_values_from_function(lambda x, y: mms.vF_k(x, y))
    vP_k.set_values_from_function(lambda x, y: mms.vP_k(x, y))
    uP_k.set_values_from_function(lambda x, y: mms.uP_k(x, y))
    pF_k.set_values_from_function(lambda x, y: mms.pF_k(x, y))
    pP_k.set_values_from_function(lambda x, y: mms.pP_k(x, y))

    dtc = Constant(dt0)

    # Manufactured interface data using discrete normals (Remark 8).
    n = FacetNormal()  # (-)->(+) (poro -> fluid) on Γ^{FP}; outside->fluid on inlet
    nF = Constant(-1.0) * n
    vF_A = Analytic(lambda x, y: mms.vF_k(x, y), degree=ana_deg)
    vP_A = Analytic(lambda x, y: mms.vP_k(x, y), degree=ana_deg)
    uP_Ak = Analytic(lambda x, y: mms.uP_k(x, y), degree=ana_deg)
    uP_An = Analytic(lambda x, y: mms.uP_n(x, y), degree=ana_deg)
    pP_A = Analytic(lambda x, y: mms.pP_k(x, y), degree=ana_deg)
    sigmaF_A = Analytic(lambda x, y: mms.sigmaF_k(x, y), degree=ana_deg)
    sigmaP_A = Analytic(lambda x, y: mms.sigmaP_k(x, y), degree=ana_deg)

    u_dot_A = (uP_Ak - uP_An) / dtc
    tractionF_A = dot(sigmaF_A, nF)
    g_sigma = dot(sigmaF_A - sigmaP_A, nF)
    g_sigma_n = dot(nF, tractionF_A) + pP_A
    g_n = vF_A - u_dot_A - porosity * (vP_A - u_dot_A)
    g_t = vF_A - u_dot_A - beta_BJ * porosity * (vP_A - u_dot_A) + kappa * tractionF_A

    forms = build_fpi_eulerian_forms(
        vF_k=vF_k,
        pF_k=pF_k,
        vP_k=vP_k,
        uP_k=uP_k,
        pP_k=pP_k,
        vF_n=vF_n,
        pF_n=pF_n,
        vP_n=vP_n,
        uP_n=uP_n,
        pP_n=pP_n,
        uP_nm1=uP_nm1,
        dvF=dvF,
        dpF=dpF,
        dvP=dvP,
        duP=duP,
        dpP=dpP,
        vF_test=vF_test,
        qF_test=qF_test,
        vP_test=vP_test,
        uP_test=uP_test,
        qP_test=qP_test,
        dx_f=prob.dx_f,
        dx_p=prob.dx_p,
        dGamma=prob.dGamma_fp,
        dG_f=prob.dG_f,
        dG_p=prob.dG_p,
        level_set=prob.fluid_ls,
        dt=dtc,
        theta=1.0,
        rho_f=rho_f,
        mu_f=mu_f,
        rho_s0_tilde=rho_s0_tilde,
        porosity=porosity,
        K_inv=K_inv,
        c_nh=c_nh,
        beta_nh=beta_nh,
        beta_BJ=beta_BJ,
        kappa=kappa,
        gamma_n=Constant(gamma),
        gamma_t=Constant(gamma),
        zeta=1.0,
        gamma_F_p=0.0,
        gamma_F_gp=0.0,
        gamma_P_p=0.0,
        gamma_P_gp=0.0,
        g_sigma=g_sigma,
        g_sigma_n=g_sigma_n,
        g_n=g_n,
        g_t=g_t,
        use_paper_phi_gamma=True,
        use_paper_stabilization=True,
        poly_order=p,
        vF_inf=v_inf,
        c_v_gamma=1.0 / 6.0,
        c_t_gamma=1.0 / 12.0,
        h_gamma=prob.h_gamma,
        use_interface_terms=True,
        use_stabilization=True,
    )

    # Forcing terms (volume)
    fF = Analytic(lambda x, y: mms.fF(x, y), degree=ana_deg)
    fD = Analytic(lambda x, y: mms.fD(x, y), degree=ana_deg)
    fS = Analytic(lambda x, y: mms.fS(x, y), degree=ana_deg)
    f_mass = Analytic(lambda x, y: mms.f_mass(x, y), degree=ana_deg)

    residual_form = (
        forms.residual_form
        - dot(fF, vF_test) * prob.dx_f
        - dot(fD, vP_test) * prob.dx_p
        - dot(fS, uP_test) * prob.dx_p
        - f_mass * qP_test * prob.dx_p
    )

    # Inlet traction: add -∫ (σ^F_A n^F) · w_F ds on the fluid side.
    traction_in = dot(sigmaF_A, nF)
    residual_form = residual_form - inner(traction_in, Pos(vF_test)) * prob.dGamma_inlet

    # Robust Dirichlet DOF tagging on the *rotated* outer boundary (paper Fig. 3):
    # |x|+|y| = sqrt(2)/2, excluding the removed corner x < x0.
    r = math.sqrt(2.0) / 2.0

    def _outer_dof_locator(x: float, y: float) -> bool:
        x = float(x)
        y = float(y)
        if x < float(args.x0) - 1.0e-12:
            return False
        return abs(abs(x) + abs(y) - r) <= 5.0e-10

    dirichlet_dof_tag = "outer_dirichlet_dofs"
    dh.tag_dofs_by_locator_map({dirichlet_dof_tag: _outer_dof_locator}, fields=["v_pos_x", "v_pos_y"])

    # Pressure pinning (one DOF per pressure field) to remove nullspaces.
    dh.dof_tags.setdefault("p_pos_pin", set())
    dh.dof_tags.setdefault("p_neg_pin", set())
    for fld, tag_name in (("p_pos_", "p_pos_pin"), ("p_neg_", "p_neg_pin")):
        sl = np.asarray(dh.get_field_slice(fld), dtype=int)
        if sl.size == 0:
            continue
        inactive = set(getattr(dh, "dof_tags", {}).get("inactive", set()))
        for gd in sl.tolist():
            if int(gd) not in inactive:
                dh.dof_tags[tag_name].add(int(gd))
                break

    bcs = [
        BoundaryCondition("v_pos_x", "dirichlet", dirichlet_dof_tag, mms.vF_x),
        BoundaryCondition("v_pos_y", "dirichlet", dirichlet_dof_tag, mms.vF_y),
        BoundaryCondition("p_pos_", "dirichlet", "p_pos_pin", mms.pF_s),
        BoundaryCondition("p_neg_", "dirichlet", "p_neg_pin", mms.pP_s),
    ]
    bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]

    solver = NewtonSolver(
        residual_form,
        forms.jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=float(args.newton_tol), max_newton_iter=int(args.newton_it)),
        quad_order=int(qdeg),
        backend=str(args.backend),
    )

    solver.solve_time_interval(
        functions=[vF_k, pF_k, vP_k, uP_k, pP_k],
        prev_functions=[vF_n, pF_n, vP_n, uP_n, pP_n],
        aux_functions={"dt": dtc, "uP_nm1": uP_nm1},
        time_params=TimeStepperParameters(dt=dt0, final_time=dt0, max_steps=1),
    )

    print("Solved FPI Example 4.1 on two meshes (UFL nonmatching).")


if __name__ == "__main__":
    main()
