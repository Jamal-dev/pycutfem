import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

for _k in (
    "NUMBA_DEBUG",
    "NUMBA_DUMP_BYTECODE",
    "NUMBA_DUMP_IR",
    "NUMBA_DUMP_SSA",
    "NUMBA_DEBUG_ARRAY_OPT",
):
    os.environ[_k] = "0"

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, NewtonSolver
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import Constant, FacetNormal, Function, MeshSize, TestFunction, TrialFunction, avg, dot, grad, inner, jump
from pycutfem.ufl.measures import ds, dx
from pycutfem.utils.functionals import NamedFunctionalEvaluator
from pycutfem.utils.meshgen import structured_quad

from examples.utils.biofilm.deformation_only import _W_prime, _W_second, _c
from examples.utils.biofilm.interface_transport_cases import InterfaceTransportCase, build_interface_transport_case


@dataclass(frozen=True)
class GridSampler:
    resolution: int
    x: np.ndarray
    y: np.ndarray
    cell_area: float


def _sqrt(expr):
    return expr ** _c(0.5)


def _grad_inner_jump(u, v, n):
    ju = jump(grad(u), n)
    jv = jump(grad(v), n)
    return inner(ju, jv)


def _tag_unit_square_boundaries(mesh: Mesh, *, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - 1.0) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - 1.0) <= tol,
        }
    )


def _create_problem(nx: int) -> dict[str, object]:
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=int(nx), ny=int(nx), poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    _tag_unit_square_boundaries(mesh)
    me = MixedElement(mesh, field_specs={"alpha": 1, "mu_alpha": 1})
    dh = DofHandler(me, method="cg")
    return {
        "mesh": mesh,
        "me": me,
        "dh": dh,
        "dalpha": TrialFunction("alpha", dof_handler=dh),
        "dmu": TrialFunction("mu_alpha", dof_handler=dh),
        "alpha_test": TestFunction("alpha", dof_handler=dh),
        "mu_test": TestFunction("mu_alpha", dof_handler=dh),
        "alpha_k": Function("alpha_k", "alpha", dof_handler=dh),
        "mu_k": Function("mu_k", "mu_alpha", dof_handler=dh),
        "alpha_n": Function("alpha_n", "alpha", dof_handler=dh),
        "mu_n": Function("mu_n", "mu_alpha", dof_handler=dh),
    }


def _build_forms(
    problem: dict[str, object],
    *,
    dt: float,
    theta: float,
    M_alpha: float,
    gamma_alpha: float,
    eps_alpha: float,
    v_adv,
    div_v_adv,
    qdeg: int,
    alpha_supg: float,
    alpha_cip: float,
) -> dict[str, object]:
    alpha_k = problem["alpha_k"]
    mu_k = problem["mu_k"]
    alpha_n = problem["alpha_n"]
    dalpha = problem["dalpha"]
    dmu = problem["dmu"]
    alpha_test = problem["alpha_test"]
    mu_test = problem["mu_test"]

    dxm = dx(metadata={"q": int(qdeg)})
    dsm = ds(metadata={"q": int(qdeg)})
    th = _c(float(theta))
    one_m_th = _c(1.0) - th
    inv_dt = _c(1.0) / Constant(float(dt))
    M_c = _c(float(M_alpha))
    gamma_c = _c(float(gamma_alpha))
    eps_c = _c(float(eps_alpha))

    adv_k = grad(alpha_k)[0] * v_adv[0] + grad(alpha_k)[1] * v_adv[1]
    adv_n = grad(alpha_n)[0] * v_adv[0] + grad(alpha_n)[1] * v_adv[1]

    r_alpha = alpha_test * ((alpha_k - alpha_n) * inv_dt) * dxm
    r_alpha += th * alpha_test * (adv_k + alpha_k * div_v_adv) * dxm
    r_alpha += one_m_th * alpha_test * (adv_n + alpha_n * div_v_adv) * dxm
    r_alpha += M_c * inner(grad(mu_k), grad(alpha_test)) * dxm

    a_alpha = alpha_test * (dalpha * inv_dt) * dxm
    a_alpha += th * alpha_test * ((grad(dalpha)[0] * v_adv[0]) + (grad(dalpha)[1] * v_adv[1]) + dalpha * div_v_adv) * dxm
    a_alpha += M_c * inner(grad(dmu), grad(alpha_test)) * dxm

    if float(alpha_supg) != 0.0:
        h_a = MeshSize()
        vmag2 = v_adv[0] * v_adv[0] + v_adv[1] * v_adv[1]
        vmag = _sqrt(vmag2 + _c(1.0e-12))
        denom = (_c(2.0) * inv_dt) * (_c(2.0) * inv_dt) + (_c(2.0) * vmag / (h_a + _c(1.0e-12))) * (
            _c(2.0) * vmag / (h_a + _c(1.0e-12))
        )
        tau_supg = _c(float(alpha_supg)) / _sqrt(denom + _c(1.0e-16))
        g_test = grad(alpha_test)
        w_supg = g_test[0] * v_adv[0] + g_test[1] * v_adv[1]
        f_alpha_k = (alpha_k - alpha_n) * inv_dt
        f_alpha_k += th * (adv_k + alpha_k * div_v_adv)
        f_alpha_k += one_m_th * (adv_n + alpha_n * div_v_adv)
        r_alpha += tau_supg * w_supg * f_alpha_k * dxm

        df_alpha_k = dalpha * inv_dt
        df_alpha_k += th * ((grad(dalpha)[0] * v_adv[0]) + (grad(dalpha)[1] * v_adv[1]) + dalpha * div_v_adv)
        a_alpha += tau_supg * w_supg * df_alpha_k * dxm

    if float(alpha_cip) != 0.0:
        n_int = FacetNormal()
        h_F = avg(MeshSize())
        tau_cip = _c(float(alpha_cip)) * (h_F * h_F * h_F) * inv_dt
        r_alpha += tau_cip * _grad_inner_jump(alpha_k, alpha_test, n_int) * dsm
        a_alpha += tau_cip * _grad_inner_jump(dalpha, alpha_test, n_int) * dsm

    Wp_k = _W_prime(alpha_k)
    Wpp_k = _W_second(alpha_k)
    r_mu = mu_test * mu_k * dxm
    r_mu += -(gamma_c * eps_c) * inner(grad(alpha_k), grad(mu_test)) * dxm
    r_mu += -mu_test * ((gamma_c / eps_c) * Wp_k) * dxm

    a_mu = mu_test * dmu * dxm
    a_mu += -(gamma_c * eps_c) * inner(grad(dalpha), grad(mu_test)) * dxm
    a_mu += -mu_test * ((gamma_c / eps_c) * Wpp_k * dalpha) * dxm

    return {
        "residual": r_alpha + r_mu,
        "jacobian": a_alpha + a_mu,
    }


def _build_functionals(problem: dict[str, object], *, quad_order: int, backend: str) -> NamedFunctionalEvaluator:
    alpha_k = problem["alpha_k"]
    x_coord = Analytic(lambda x, y: x, degree=1)
    y_coord = Analytic(lambda x, y: y, degree=1)
    one = Constant(1.0)
    forms = {
        "mass": alpha_k * dx(metadata={"q": int(quad_order)}),
        "mx": x_coord * alpha_k * dx(metadata={"q": int(quad_order)}),
        "my": y_coord * alpha_k * dx(metadata={"q": int(quad_order)}),
        "band_moment": (_c(4.0) * alpha_k * (one - alpha_k)) * dx(metadata={"q": int(quad_order)}),
    }
    return NamedFunctionalEvaluator(
        forms=forms,
        dof_handler=problem["dh"],
        mixed_element=problem["me"],
        backend=str(backend),
        quad_order=int(quad_order),
    )


def _make_grid_sampler(resolution: int) -> GridSampler:
    n = int(max(16, resolution))
    coords = (np.arange(n, dtype=float) + 0.5) / float(n)
    x, y = np.meshgrid(coords, coords)
    return GridSampler(resolution=n, x=x, y=y, cell_area=1.0 / float(n * n))


def _sample_alpha_on_grid(problem: dict[str, object], alpha_f: Function, sampler: GridSampler) -> np.ndarray:
    from matplotlib.tri import LinearTriInterpolator

    tri = problem["alpha_tri"]
    interp = LinearTriInterpolator(tri, np.asarray(alpha_f.nodal_values, dtype=float).ravel())
    vals = interp(sampler.x, sampler.y)
    if np.ma.isMaskedArray(vals):
        vals = vals.filled(np.nan)
    vals = np.asarray(vals, dtype=float)
    if np.any(~np.isfinite(vals)):
        raise RuntimeError("Non-finite alpha samples encountered on the geometry grid.")
    return vals


def _geometry_metrics_from_samples(alpha_num: np.ndarray, alpha_exact: np.ndarray, sampler: GridSampler, *, threshold: float = 0.5) -> dict[str, float]:
    cell_area = float(sampler.cell_area)
    mask_num = np.asarray(alpha_num >= float(threshold), dtype=bool)
    mask_exact = np.asarray(alpha_exact >= float(threshold), dtype=bool)
    area_num = cell_area * float(np.sum(mask_num))
    area_exact = cell_area * float(np.sum(mask_exact))

    def _centroid(mask: np.ndarray, area_val: float) -> tuple[float, float]:
        if area_val <= 1.0e-30:
            return float("nan"), float("nan")
        cx = cell_area * float(np.sum(sampler.x[mask])) / area_val
        cy = cell_area * float(np.sum(sampler.y[mask])) / area_val
        return cx, cy

    cx_num, cy_num = _centroid(mask_num, area_num)
    cx_exact, cy_exact = _centroid(mask_exact, area_exact)
    if np.isfinite(cx_num) and np.isfinite(cx_exact):
        geom_centroid_err = float(math.hypot(cx_num - cx_exact, cy_num - cy_exact))
    else:
        geom_centroid_err = float("nan")

    band_num = cell_area * float(np.sum((alpha_num >= 0.1) & (alpha_num <= 0.9)))
    band_exact = cell_area * float(np.sum((alpha_exact >= 0.1) & (alpha_exact <= 0.9)))
    shape_mismatch = cell_area * float(np.sum(np.logical_xor(mask_num, mask_exact)))

    return {
        "area_05": area_num,
        "area_05_exact": area_exact,
        "rel_area_05_drift": (area_num - area_exact) / max(abs(area_exact), 1.0e-30),
        "cx_geom": cx_num,
        "cy_geom": cy_num,
        "cx_geom_exact": cx_exact,
        "cy_geom_exact": cy_exact,
        "geom_centroid_err": geom_centroid_err,
        "band_01_09": band_num,
        "band_01_09_exact": band_exact,
        "rel_band_01_09_drift": (band_num - band_exact) / max(abs(band_exact), 1.0e-30),
        "shape_mismatch_rel": shape_mismatch / max(abs(area_exact), 1.0e-30),
    }


def _sample_geometry_metrics(problem: dict[str, object], alpha_f: Function, case: InterfaceTransportCase, *, t_now: float, sampler: GridSampler) -> dict[str, float]:
    alpha_num = _sample_alpha_on_grid(problem, alpha_f, sampler)
    alpha_exact = np.asarray(case.alpha(sampler.x, sampler.y, float(t_now)), dtype=float)
    return _geometry_metrics_from_samples(alpha_num, alpha_exact, sampler)


def _save_snapshot_png(
    problem: dict[str, object],
    alpha_f: Function,
    case: InterfaceTransportCase,
    *,
    t_now: float,
    path: Path,
    dpi: int,
    grid_resolution: int,
    metrics: dict[str, float] | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pycutfem.plotting.triangulate import triangulate_field

    tri = problem.get("alpha_tri")
    if tri is None:
        tri = triangulate_field(problem["mesh"], problem["dh"], "alpha")
    z = np.asarray(alpha_f.nodal_values, dtype=float).ravel()

    fig, ax = plt.subplots(figsize=(5.4, 4.2), constrained_layout=True)
    cf = ax.tricontourf(tri, z, levels=np.linspace(0.0, 1.0, 21), cmap="viridis")
    grid = np.linspace(0.0, 1.0, int(max(256, grid_resolution)))
    gx, gy = np.meshgrid(grid, grid)
    exact = np.asarray(case.alpha(gx, gy, float(t_now)), dtype=float)
    ax.contour(gx, gy, exact, levels=[0.5], colors="white", linewidths=1.5)
    ax.tricontour(tri, z, levels=[0.5], colors="black", linewidths=1.0)
    ax.set_aspect("equal", "box")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{case.title}, t={float(t_now):.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(cf, ax=ax, label=r"$\alpha$")

    if metrics:
        text = "\n".join(
            [
                f"|dm|={abs(float(metrics['rel_mass_drift'])):.2e}",
                f"|dA|={abs(float(metrics['rel_area_05_drift'])):.2e}",
                f"ec={float(metrics['geom_centroid_err']):.2e}",
                f"eshape={float(metrics['shape_mismatch_rel']):.2e}",
            ]
        )
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def _save_snapshot_vtk(problem: dict[str, object], alpha_f: Function, mu_f: Function, case: InterfaceTransportCase, *, t_now: float, path: Path) -> None:
    export_vtk(
        str(path),
        problem["mesh"],
        problem["dh"],
        {
            "alpha": alpha_f,
            "mu_alpha": mu_f,
            "alpha_exact": lambda x, y: case.alpha(x, y, float(t_now)),
            "v_adv": lambda x, y: case.velocity(x, y, float(t_now)),
        },
    )


def _write_snapshot_panel(image_paths: list[tuple[float, Path]], *, title: str, out_path: Path) -> None:
    if not image_paths:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(image_paths), figsize=(4.8 * len(image_paths), 4.3), constrained_layout=True)
    if len(image_paths) == 1:
        axes = [axes]
    for ax, (t_now, img_path) in zip(axes, image_paths):
        ax.imshow(mpimg.imread(str(img_path)))
        ax.set_axis_off()
        ax.set_title(f"t={float(t_now):.3f}")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _solve_case(
    *,
    case_key: str,
    nx: int,
    cfl: float,
    theta: float,
    alpha_supg: float | None,
    alpha_cip: float | None,
    backend: str,
    qdeg: int,
    q_metrics: int,
    newton_tol: float,
    max_it: int,
    outdir: Path,
    vtk_snapshots: bool,
    png_dpi: int,
    geom_grid: int,
    final_grid: int,
    snapshot_grid: int,
    geom_every: int,
) -> dict[str, float]:
    case = build_interface_transport_case(case_key)
    problem = _create_problem(nx)
    dh = problem["dh"]
    from pycutfem.plotting.triangulate import triangulate_field

    problem["alpha_tri"] = triangulate_field(problem["mesh"], dh, "alpha")

    h = 1.0 / float(nx)
    speed_scale = max(float(case.speed_scale), 1.0e-12)
    dt_guess = float(cfl) * h / speed_scale
    nsteps = max(1, int(math.ceil(float(case.t_final) / max(dt_guess, 1.0e-12))))
    dt = float(case.t_final) / float(nsteps)
    supg_val = float(alpha_supg) if alpha_supg is not None else (1.0 if float(case.M_alpha) == 0.0 else 0.0)
    cip_val = float(alpha_cip) if alpha_cip is not None else (10.0 if float(case.M_alpha) == 0.0 else 0.0)

    current_time = [0.0]
    v_adv = Analytic(lambda x, y: case.velocity(x, y, current_time[0]), degree=4)
    div_v_adv = Analytic(lambda x, y: case.div_velocity(x, y, current_time[0]), degree=2)

    forms = _build_forms(
        problem,
        dt=dt,
        theta=theta,
        M_alpha=float(case.M_alpha),
        gamma_alpha=float(case.gamma_alpha),
        eps_alpha=float(case.eps_alpha),
        v_adv=v_adv,
        div_v_adv=div_v_adv,
        qdeg=qdeg,
        alpha_supg=supg_val,
        alpha_cip=cip_val,
    )
    solver = NewtonSolver(
        forms["residual"],
        forms["jacobian"],
        dof_handler=dh,
        mixed_element=problem["me"],
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(
            newton_tol=float(newton_tol),
            max_newton_iter=int(max_it),
            print_level=0,
        ),
        lin_params=LinearSolverParameters(backend="scipy", tol=1.0e-12, maxit=10000),
        quad_order=int(qdeg),
        backend=str(backend),
    )
    functional_eval = _build_functionals(problem, quad_order=q_metrics, backend=backend)

    problem["alpha_n"].set_values_from_function(lambda x, y: float(case.alpha(x, y, 0.0)))
    problem["mu_n"].set_values_from_function(lambda x, y: 0.0)
    problem["alpha_k"].set_nodal_values(problem["alpha_k"]._g_dofs, problem["alpha_n"].nodal_values.copy())
    problem["mu_k"].set_nodal_values(problem["mu_k"]._g_dofs, problem["mu_n"].nodal_values.copy())

    case_outdir = Path(outdir)
    case_outdir.mkdir(parents=True, exist_ok=True)
    vtk_dir = case_outdir / "vtk"
    png_dir = case_outdir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    if bool(vtk_snapshots):
        vtk_dir.mkdir(parents=True, exist_ok=True)

    geom_sampler = _make_grid_sampler(geom_grid)
    final_sampler = _make_grid_sampler(final_grid)
    snapshot_sampler = _make_grid_sampler(snapshot_grid)

    coeffs = {problem["alpha_k"].name: problem["alpha_k"]}
    vals0 = functional_eval.evaluate(coeffs)
    mass0 = float(vals0["mass"])
    band_moment0 = float(vals0["band_moment"])

    timeseries_path = case_outdir / "timeseries.csv"
    snapshot_records: list[tuple[float, Path]] = []
    pending_snapshots = list(case.snapshot_times)

    fieldnames = [
        "step",
        "t",
        "mass",
        "rel_mass_drift",
        "cx_mass",
        "cy_mass",
        "cx_mass_exact",
        "cy_mass_exact",
        "mass_centroid_err",
        "band_moment",
        "rel_band_moment_drift",
        "area_05",
        "area_05_exact",
        "rel_area_05_drift",
        "cx_geom",
        "cy_geom",
        "cx_geom_exact",
        "cy_geom_exact",
        "geom_centroid_err",
        "band_01_09",
        "band_01_09_exact",
        "rel_band_01_09_drift",
        "shape_mismatch_rel",
        "alpha_min_nodal",
        "alpha_max_nodal",
        "newton_iters",
        "solve_seconds",
    ]

    def _collect_row(step_no: int, t_now: float, n_newton: int, solve_seconds: float, *, sampler: GridSampler | None) -> dict[str, float]:
        vals = functional_eval.evaluate(coeffs)
        mass = float(vals["mass"])
        mx = float(vals["mx"])
        my = float(vals["my"])
        band_moment = float(vals["band_moment"])
        cx_mass = mx / max(mass, 1.0e-30)
        cy_mass = my / max(mass, 1.0e-30)
        c_exact = np.asarray(case.centroid_exact(float(t_now)), dtype=float)
        if sampler is None:
            geom = {
                "area_05": float("nan"),
                "area_05_exact": float("nan"),
                "rel_area_05_drift": float("nan"),
                "cx_geom": float("nan"),
                "cy_geom": float("nan"),
                "cx_geom_exact": float("nan"),
                "cy_geom_exact": float("nan"),
                "geom_centroid_err": float("nan"),
                "band_01_09": float("nan"),
                "band_01_09_exact": float("nan"),
                "rel_band_01_09_drift": float("nan"),
                "shape_mismatch_rel": float("nan"),
            }
        else:
            geom = _sample_geometry_metrics(problem, problem["alpha_k"], case, t_now=t_now, sampler=sampler)
        alpha_vals = np.asarray(problem["alpha_k"].nodal_values, dtype=float).ravel()
        return {
            "step": int(step_no),
            "t": float(t_now),
            "mass": mass,
            "rel_mass_drift": (mass - mass0) / max(abs(mass0), 1.0e-30),
            "cx_mass": cx_mass,
            "cy_mass": cy_mass,
            "cx_mass_exact": float(c_exact[0]),
            "cy_mass_exact": float(c_exact[1]),
            "mass_centroid_err": float(np.linalg.norm(np.array([cx_mass, cy_mass]) - c_exact)),
            "band_moment": band_moment,
            "rel_band_moment_drift": (band_moment - band_moment0) / max(abs(band_moment0), 1.0e-30),
            "area_05": float(geom["area_05"]),
            "area_05_exact": float(geom["area_05_exact"]),
            "rel_area_05_drift": float(geom["rel_area_05_drift"]),
            "cx_geom": float(geom["cx_geom"]),
            "cy_geom": float(geom["cy_geom"]),
            "cx_geom_exact": float(geom["cx_geom_exact"]),
            "cy_geom_exact": float(geom["cy_geom_exact"]),
            "geom_centroid_err": float(geom["geom_centroid_err"]),
            "band_01_09": float(geom["band_01_09"]),
            "band_01_09_exact": float(geom["band_01_09_exact"]),
            "rel_band_01_09_drift": float(geom["rel_band_01_09_drift"]),
            "shape_mismatch_rel": float(geom["shape_mismatch_rel"]),
            "alpha_min_nodal": float(np.min(alpha_vals)),
            "alpha_max_nodal": float(np.max(alpha_vals)),
            "newton_iters": int(n_newton),
            "solve_seconds": float(solve_seconds),
        }

    def _write_snapshot(t_now: float, step_no: int, row: dict[str, float]) -> None:
        img_path = png_dir / f"snapshot_{int(step_no):04d}.png"
        _save_snapshot_png(
            problem,
            problem["alpha_k"],
            case,
            t_now=float(t_now),
            path=img_path,
            dpi=int(png_dpi),
            grid_resolution=int(snapshot_sampler.resolution),
            metrics=row,
        )
        snapshot_records.append((float(t_now), img_path))
        if bool(vtk_snapshots):
            _save_snapshot_vtk(
                problem,
                problem["alpha_k"],
                problem["mu_k"],
                case,
                t_now=float(t_now),
                path=vtk_dir / f"step_{int(step_no):04d}.vtu",
            )

    with timeseries_path.open("w", encoding="utf-8", newline="") as f_ts:
        writer = csv.DictWriter(f_ts, fieldnames=fieldnames)
        writer.writeheader()

        row0 = _collect_row(0, 0.0, 0, 0.0, sampler=geom_sampler)
        writer.writerow(row0)
        _write_snapshot(0.0, 0, row0)
        if pending_snapshots and abs(float(pending_snapshots[0])) <= 1.0e-14:
            pending_snapshots.pop(0)

        for step in range(1, nsteps + 1):
            t_n = float((step - 1) * dt)
            t_k = float(step * dt)
            current_time[0] = t_n
            solver._current_step_no = int(step)
            solver._current_t = t_n
            solver._current_dt = dt
            aux_functions = {"dt": Constant(float(dt))}
            bcs_now = solver._freeze_bcs(solver.bcs, t_k)
            if bcs_now:
                dh.apply_bcs(bcs_now, problem["alpha_k"], problem["mu_k"])

            t_start = time.perf_counter()
            _, converged, n_iters = solver._newton_loop(
                [problem["alpha_k"], problem["mu_k"]],
                [problem["alpha_n"], problem["mu_n"]],
                aux_functions,
                bcs_now,
            )
            solve_seconds = time.perf_counter() - t_start
            if not converged:
                raise RuntimeError(f"Newton did not converge for case={case.case_id}, nx={nx}, step={step}.")

            should_snapshot = False
            while pending_snapshots and t_k + 1.0e-12 >= float(pending_snapshots[0]):
                should_snapshot = True
                pending_snapshots.pop(0)
            sample_geom = (int(geom_every) <= 1) or (step % int(geom_every) == 0) or should_snapshot or (step == nsteps)

            row = _collect_row(step, t_k, int(n_iters), float(solve_seconds), sampler=geom_sampler if sample_geom else None)
            writer.writerow(row)
            if should_snapshot:
                snap_row = _collect_row(step, t_k, int(n_iters), float(solve_seconds), sampler=snapshot_sampler)
                _write_snapshot(t_k, step, snap_row)

            problem["alpha_n"].set_nodal_values(problem["alpha_n"]._g_dofs, problem["alpha_k"].nodal_values.copy())
            problem["mu_n"].set_nodal_values(problem["mu_n"]._g_dofs, problem["mu_k"].nodal_values.copy())

    series = pd.read_csv(timeseries_path)
    final_t = float(series["t"].iloc[-1])
    final_geom = _sample_geometry_metrics(problem, problem["alpha_k"], case, t_now=final_t, sampler=final_sampler)

    alpha_l2 = float(
        dh.l2_error(
            problem["alpha_k"],
            exact={"alpha": lambda x, y: case.alpha(x, y, final_t)},
            fields=["alpha"],
            quad_order=max(int(q_metrics), 8),
            relative=False,
        )
    )
    coords = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    nodal_exact = np.asarray(case.alpha(coords[:, 0], coords[:, 1], final_t), dtype=float).ravel()
    nodal_vals = np.asarray(problem["alpha_k"].nodal_values, dtype=float).ravel()
    nodal_max_err = float(np.max(np.abs(nodal_vals - nodal_exact)))
    avg_newton = float(np.mean(series["newton_iters"].to_numpy(dtype=float)[1:])) if len(series.index) > 1 else 0.0

    def _nanmax_abs(series_key: str) -> float:
        vals = np.asarray(series[series_key].to_numpy(dtype=float), dtype=float)
        if np.all(np.isnan(vals)):
            return float("nan")
        return float(np.nanmax(np.abs(vals)))

    def _nanmax(series_key: str) -> float:
        vals = np.asarray(series[series_key].to_numpy(dtype=float), dtype=float)
        if np.all(np.isnan(vals)):
            return float("nan")
        return float(np.nanmax(vals))

    summary = {
        "case": case.case_id,
        "title": case.title,
        "geometry": case.geometry,
        "nx": int(nx),
        "h": h,
        "dt": float(dt),
        "nsteps": int(nsteps),
        "eps_alpha": float(case.eps_alpha),
        "M_alpha": float(case.M_alpha),
        "gamma_alpha": float(case.gamma_alpha),
        "theta": float(theta),
        "cfl": float(cfl),
        "alpha_supg": float(supg_val),
        "alpha_cip": float(cip_val),
        "final_mass_drift": abs(float(series["rel_mass_drift"].iloc[-1])),
        "max_mass_drift": float(np.max(np.abs(series["rel_mass_drift"].to_numpy(dtype=float)))),
        "final_mass_centroid_err": float(series["mass_centroid_err"].iloc[-1]),
        "max_mass_centroid_err": float(np.max(series["mass_centroid_err"].to_numpy(dtype=float))),
        "final_geom_centroid_err": float(final_geom["geom_centroid_err"]),
        "max_geom_centroid_err": _nanmax("geom_centroid_err"),
        "final_thickness_drift": abs(float(series["rel_band_moment_drift"].iloc[-1])),
        "max_thickness_drift": float(np.max(np.abs(series["rel_band_moment_drift"].to_numpy(dtype=float)))),
        "final_area_error": abs(float(final_geom["rel_area_05_drift"])),
        "max_area_error": _nanmax_abs("rel_area_05_drift"),
        "final_shape_mismatch": float(final_geom["shape_mismatch_rel"]),
        "max_shape_mismatch": _nanmax("shape_mismatch_rel"),
        "alpha_l2_final": alpha_l2,
        "alpha_linf_nodal_final": nodal_max_err,
        "avg_newton_iters": avg_newton,
        "max_newton_iters": int(np.max(series["newton_iters"].to_numpy(dtype=int))),
        "max_alpha_overshoot": float(np.max(np.maximum(series["alpha_max_nodal"].to_numpy(dtype=float) - 1.0, 0.0))),
        "max_alpha_undershoot": float(np.max(np.maximum(-series["alpha_min_nodal"].to_numpy(dtype=float), 0.0))),
        "geom_grid": int(geom_sampler.resolution),
        "final_grid": int(final_sampler.resolution),
        "snapshot_grid": int(snapshot_sampler.resolution),
    }
    (case_outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_snapshot_panel(snapshot_records, title=case.title, out_path=case_outdir / f"{case.case_id}_snapshots.png")
    return summary


def _eoc(prev_h: float, h: float, prev_err: float, err: float) -> float:
    if not (prev_h > 0.0 and h > 0.0 and prev_err > 0.0 and err > 0.0):
        return float("nan")
    return float(math.log(prev_err / err) / math.log(prev_h / h))


def _add_eocs(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    metrics = [
        "alpha_l2_final",
        "alpha_linf_nodal_final",
        "final_area_error",
        "final_geom_centroid_err",
        "final_thickness_drift",
        "final_shape_mismatch",
        "max_mass_drift",
        "max_thickness_drift",
        "max_area_error",
        "max_shape_mismatch",
    ]
    out: list[dict[str, float]] = []
    prev = None
    for row in rows:
        cur = dict(row)
        for key in metrics:
            cur[f"eoc_{key}"] = float("nan") if prev is None else _eoc(prev["h"], cur["h"], prev[key], cur[key])
        out.append(cur)
        prev = cur
    return out


def _write_outputs(case: str, rows: list[dict[str, float]], *, outdir: Path, save_plot: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    stem = f"deformation_only_interface_transport_{case}"

    (outdir / f"{stem}.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (outdir / f"{stem}.tex").write_text(
        df.to_latex(index=False, float_format=lambda x: f"{x:.3e}", na_rep="-"),
        encoding="utf-8",
    )
    (outdir / f"{stem}.json").write_text(
        json.dumps({"case": case, "rows": rows, "best_row": rows[-1]}, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        md_text = df.to_markdown(index=False)
    except Exception:
        md_text = df.to_string(index=False)
    (outdir / f"{stem}.md").write_text(md_text + "\n", encoding="utf-8")

    if not save_plot:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable; skipping transport convergence plot: {exc}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    h = df["h"].to_numpy(dtype=float)
    panels = [
        ("alpha_l2_final", r"Final $\|e_\alpha\|_{L^2}$"),
        ("max_mass_drift", r"Max $|\delta m|$"),
        ("max_thickness_drift", r"Max thickness drift"),
        ("max_shape_mismatch", r"Max shape mismatch"),
    ]
    for ax, (key, title) in zip(axes.ravel(), panels):
        vals = np.maximum(np.asarray(df[key], dtype=float), 1.0e-18)
        ax.loglog(h, vals, marker="o", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel(r"$h$")
        ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    fig.savefig(outdir / f"{stem}_convergence.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Reduced interface transport / geometry-preservation benchmark.")
    ap.add_argument("--case", type=str, default="translation", choices=("translation", "rotation", "shear_return"))
    ap.add_argument("--nx-list", type=str, default="32,64")
    ap.add_argument("--q", type=int, default=6)
    ap.add_argument("--q-metrics", type=int, default=8)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--cfl", type=float, default=0.35)
    ap.add_argument(
        "--alpha-supg",
        type=float,
        default=None,
        help="SUPG factor for alpha transport. Default: auto-enable 1.0 when M_alpha=0; explicit 0 disables.",
    )
    ap.add_argument(
        "--alpha-cip",
        type=float,
        default=None,
        help="CIP factor for alpha transport. Default: auto-enable 10.0 when M_alpha=0; explicit 0 disables.",
    )
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--newton-tol", type=float, default=1.0e-10)
    ap.add_argument("--max-it", type=int, default=20)
    ap.add_argument("--vtk-snapshots", action="store_true")
    ap.add_argument("--png-dpi", type=int, default=360)
    ap.add_argument("--geom-grid", type=int, default=160)
    ap.add_argument("--final-grid", type=int, default=640)
    ap.add_argument("--snapshot-grid", type=int, default=700)
    ap.add_argument("--geom-every", type=int, default=1)
    ap.add_argument("--convergence", action="store_true")
    ap.add_argument("--outdir", type=str, default="examples/biofilms/results/deformation_only_interface_transport")
    args = ap.parse_args()

    nx_list = [int(s.strip()) for s in str(args.nx_list).split(",") if s.strip()]
    if not nx_list:
        raise ValueError("Provide at least one mesh size in --nx-list.")

    case_root = Path(str(args.outdir)) / str(args.case)
    rows: list[dict[str, float]] = []
    finest_nx = max(nx_list)
    for nx in nx_list:
        row = _solve_case(
            case_key=str(args.case),
            nx=int(nx),
            cfl=float(args.cfl),
            theta=float(args.theta),
            alpha_supg=args.alpha_supg,
            alpha_cip=args.alpha_cip,
            backend=str(args.backend),
            qdeg=int(args.q),
            q_metrics=int(args.q_metrics),
            newton_tol=float(args.newton_tol),
            max_it=int(args.max_it),
            outdir=case_root / f"nx{int(nx):03d}",
            vtk_snapshots=bool(args.vtk_snapshots) and int(nx) == int(finest_nx),
            png_dpi=int(args.png_dpi),
            geom_grid=max(int(args.geom_grid), min(256, 2 * int(nx))),
            final_grid=max(int(args.final_grid), min(1024, 6 * int(nx))),
            snapshot_grid=max(int(args.snapshot_grid), min(1200, 8 * int(nx))),
            geom_every=max(1, int(args.geom_every)),
        )
        rows.append(row)
    if len(rows) >= 2:
        rows = _add_eocs(rows)

    df = pd.DataFrame(rows)
    print(df)
    print()
    print(df.to_latex(index=False, float_format=lambda x: f"{x:.3e}", na_rep="-"))
    _write_outputs(str(args.case), rows, outdir=case_root, save_plot=bool(args.convergence))


if __name__ == "__main__":
    main()
