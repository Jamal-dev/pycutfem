#!/usr/bin/env python3
"""Paper 1 Benchmark 4: Terzaghi consolidation for the poroelastic block.

This benchmark tailors the existing pycutfem Biot consolidation machinery to a
canonical single-drainage Terzaghi setting:

  - rectangular column in plane strain,
  - drained top boundary, impermeable base,
  - lateral pinning that enforces one-dimensional vertical consolidation,
  - instantaneous step load represented through a consistent undrained initial
    excess-pressure state together with the maintained compressive top traction
    during subsequent Biot diffusion.

The benchmark is intended to verify the poroelastic pressure-diffusion and
settlement response of the reduced Paper 1 mechanics block. It reports
dimensionless pressure-profile and settlement errors against the analytic
Terzaghi series solution and generates manuscript-ready plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional in light-weight test envs
    matplotlib = None
    plt = None

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as sp_la
except Exception:  # pragma: no cover - solver path is validated in fenicsx env
    sp = None
    sp_la = None


_TRAPEZOID = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass(frozen=True)
class TerzaghiParameters:
    L: float = 0.25
    H: float = 1.0
    sigma0: float = 1.0
    poisson_ratio: float = 0.2
    E: float = 1.0
    biot_coef: float = 1.0
    biot_modulus: float = 10.0
    permeability: float = 1.0
    theta_step: float = 1.0

    @property
    def lambda_(self) -> float:
        nu = float(self.poisson_ratio)
        return float(self.E) * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    @property
    def mu(self) -> float:
        nu = float(self.poisson_ratio)
        return float(self.E) / (2.0 * (1.0 + nu))

    @property
    def constrained_modulus(self) -> float:
        return float(self.lambda_) + 2.0 * float(self.mu)

    @property
    def storage_coefficient(self) -> float:
        return 1.0 / float(self.biot_modulus) + (float(self.biot_coef) ** 2) / float(self.constrained_modulus)

    @property
    def consolidation_coefficient(self) -> float:
        return float(self.permeability) / float(self.storage_coefficient)

    @property
    def initial_pressure(self) -> float:
        return float(self.sigma0) / float(self.biot_coef)

    @property
    def final_settlement(self) -> float:
        return -float(self.sigma0) * float(self.H) / float(self.constrained_modulus)


@dataclass(frozen=True)
class CaseResult:
    row: dict[str, float]
    history: dict[str, np.ndarray]
    profile_samples: dict[str, dict[str, np.ndarray]]
    params: TerzaghiParameters


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for item in str(raw).split(","):
        text = item.strip()
        if not text:
            continue
        value = int(text)
        if value < 8:
            raise ValueError("Each n_y value must be at least 8.")
        out.append(value)
    if not out:
        raise ValueError("Expected at least one n_y value.")
    return out


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for item in str(raw).split(","):
        text = item.strip()
        if not text:
            continue
        out.append(float(text))
    if not out:
        raise ValueError("Expected at least one numeric value.")
    return out


def _generate_uniform_points(*, L: float, H: float, nx_cells: int, ny_cells: int) -> np.ndarray:
    x = np.linspace(0.0, float(L), int(nx_cells) + 1, dtype=float)
    y = np.linspace(0.0, float(H), int(ny_cells) + 1, dtype=float)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel()])


def _pressure_series_terms(y: np.ndarray, Tv: float, *, H: float, n_terms: int) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    modes = np.arange(int(n_terms), dtype=float).reshape(1, -1)
    odd = 2.0 * modes + 1.0
    coeff = (4.0 / math.pi) / odd
    exponent = np.exp(-0.25 * (math.pi ** 2) * odd * odd * float(Tv))
    cosine = np.cos(0.5 * math.pi * odd * (y / float(H)))
    return np.sum(coeff * cosine * exponent, axis=1)


def terzaghi_pressure_exact(y: np.ndarray, t: float, *, params: TerzaghiParameters, n_terms: int = 400) -> np.ndarray:
    Tv = float(params.consolidation_coefficient) * max(float(t), 0.0) / (float(params.H) ** 2)
    return float(params.initial_pressure) * _pressure_series_terms(y, Tv, H=float(params.H), n_terms=n_terms)


def terzaghi_pressure_bar_exact(y: np.ndarray, t: float, *, params: TerzaghiParameters, n_terms: int = 400) -> np.ndarray:
    return terzaghi_pressure_exact(y, t, params=params, n_terms=n_terms) / float(params.initial_pressure)


def terzaghi_settlement_bar_exact(t: float, *, params: TerzaghiParameters, n_terms: int = 400) -> float:
    Tv = float(params.consolidation_coefficient) * max(float(t), 0.0) / (float(params.H) ** 2)
    odd = 2.0 * np.arange(int(n_terms), dtype=float) + 1.0
    series = np.sum((8.0 / (math.pi ** 2)) * np.exp(-0.25 * (math.pi ** 2) * odd * odd * float(Tv)) / (odd * odd))
    return float(1.0 - series)


def terzaghi_settlement_exact(t: float, *, params: TerzaghiParameters, n_terms: int = 400) -> float:
    return float(params.final_settlement) * terzaghi_settlement_bar_exact(t, params=params, n_terms=n_terms)


def _group_profile_levels(coords_y: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    y_key = np.round(np.asarray(coords_y, dtype=float), 12)
    unique, inverse = np.unique(y_key, return_inverse=True)
    order = np.argsort(unique)
    levels = unique[order]
    groups = [np.where(inverse == int(k))[0] for k in order]
    return levels, groups


def _collapse_profile(values: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    return np.asarray([float(np.mean(arr[idx])) for idx in groups], dtype=float)


def _locate_element(mesh, point: np.ndarray):
    from pycutfem.fem import transform

    xy = np.asarray(point, dtype=float)
    for elem in mesh.elements_list:
        node_ids = elem.nodes
        coords = mesh.nodes_x_y_pos[list(node_ids)]
        if not (
            coords[:, 0].min() - 1.0e-12 <= xy[0] <= coords[:, 0].max() + 1.0e-12
            and coords[:, 1].min() - 1.0e-12 <= xy[1] <= coords[:, 1].max() + 1.0e-12
        ):
            continue
        try:
            xi, eta = transform.inverse_mapping(mesh, elem.id, xy)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if -1.0001 <= xi <= 1.0001 and -1.0001 <= eta <= 1.0001:
            return elem.id, xi, eta
    return None, None, None


def _build_scalar_profile_evaluator(dof_handler: DofHandler, mesh, *, field_name: str, points: np.ndarray):
    me = dof_handler.mixed_element
    cache: list[tuple[np.ndarray, np.ndarray]] = []
    for point in np.asarray(points, dtype=float):
        eid, xi, eta = _locate_element(mesh, point)
        if eid is None:
            raise RuntimeError(f"Failed to locate evaluation point {point.tolist()} in the Terzaghi mesh.")
        phi = np.asarray(me.basis(field_name, xi, eta)[me.slice(field_name)], dtype=float).ravel()
        gdofs = np.asarray(dof_handler.element_maps[field_name][eid], dtype=int).ravel()
        cache.append((phi, gdofs))

    def _evaluate(func: Function) -> np.ndarray:
        out = np.zeros((len(cache),), dtype=float)
        for idx, (phi, gdofs) in enumerate(cache):
            out[idx] = float(phi @ np.asarray(func.get_nodal_values(gdofs), dtype=float).ravel())
        return out

    return _evaluate


def _series_label(Tv: float) -> str:
    text = f"{float(Tv):.2f}".rstrip("0").rstrip(".")
    return text or "0"


def _solve_case(
    *,
    ny_cells: int,
    nx_cells: int,
    steps_per_ny: int,
    Tv_final: float,
    params: TerzaghiParameters,
    backend: str,
    n_terms: int,
    sample_Tv: list[float],
    print_progress: bool,
) -> CaseResult:
    if sp is None or sp_la is None:
        raise RuntimeError("scipy is required to run Benchmark 4 solves.")
    from examples.poroelasticity.consolidation_pycutfem import _build_p2_tri_mesh, _structured_cells
    from pycutfem.core.dofhandler import DofHandler
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.ufl.compilers import FormCompiler
    from pycutfem.ufl.expressions import (
        Constant,
        Function,
        Identity,
        TestFunction,
        TrialFunction,
        VectorFunction,
        VectorTestFunction,
        VectorTrialFunction,
        div,
        grad,
        inner,
        trace,
    )
    from pycutfem.ufl.forms import BoundaryCondition
    from pycutfem.ufl.measures import dS, dx
    from pycutfem.ufl.spaces import FunctionSpace
    if int(nx_cells) < 2:
        raise ValueError("nx_cells must be at least 2.")
    if int(steps_per_ny) < 1:
        raise ValueError("steps_per_ny must be at least 1.")

    nx_cells = int(nx_cells)
    ny_cells = int(ny_cells)
    sample_Tv = [float(v) for v in sample_Tv]
    sample_Tv = sorted(sample_Tv)
    # The step-load Terzaghi solution has a startup singular layer near T_v=0.
    # Convergence-style profile metrics should therefore use later sample times;
    # the full history plots still show the entire transient.

    final_time = float(Tv_final) * (float(params.H) ** 2) / float(params.consolidation_coefficient)
    num_time_steps = int(max(8, steps_per_ny * ny_cells))
    dt = float(final_time) / float(num_time_steps)

    points = _generate_uniform_points(L=float(params.L), H=float(params.H), nx_cells=nx_cells, ny_cells=ny_cells)
    cells = _structured_cells(nx=nx_cells + 1, ny=ny_cells + 1)
    mesh = _build_p2_tri_mesh(points, cells)
    mesh.tag_boundary_edges(
        {
            "bottom": lambda x, y: np.isclose(y, 0.0),
            "left": lambda x, y: np.isclose(x, 0.0),
            "right": lambda x, y: np.isclose(x, float(params.L)),
            "top": lambda x, y: np.isclose(y, float(params.H)),
        }
    )

    mixed_element = MixedElement(mesh, field_specs={"ux": 2, "uy": 2, "p": 1})
    dh = DofHandler(mixed_element, method="cg")

    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)
    pres_space = FunctionSpace("pressure", ["p"], dim=0)

    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)

    u_prev = VectorFunction(name="u_prev", field_names=["ux", "uy"], dof_handler=dh)
    p_prev = Function(name="p_prev", field_name="p", dof_handler=dh)
    u_prev.nodal_values.fill(0.0)
    p_prev.nodal_values.fill(float(params.initial_pressure))

    theta = Constant(float(params.theta_step))
    theta._jit_name = "theta_step"
    dt_c = Constant(float(dt))
    dt_c._jit_name = "dt"
    mu = Constant(float(params.mu))
    mu._jit_name = "mu_s"
    lam = Constant(float(params.lambda_))
    lam._jit_name = "lambda_s"
    biot = Constant(float(params.biot_coef))
    biot._jit_name = "biot_coef"
    invM = Constant(1.0 / float(params.biot_modulus))
    invM._jit_name = "inv_biot_modulus"
    k_perm = Constant(float(params.permeability))
    k_perm._jit_name = "permeability"
    I2 = Identity(2)
    top_load = Constant(-float(params.sigma0))
    top_load._jit_name = "sigma_top"
    ds_top = dS(defined_on=mesh.edge_bitset("top"), metadata={"q": 5})

    def eps(w):
        return 0.5 * (grad(w) + grad(w).T)

    def sigma_s(w):
        return Constant(2.0) * mu * eps(w) + lam * trace(eps(w)) * I2

    H_pq = inner(k_perm * grad(p), grad(q))
    H0_pq = inner(k_perm * grad(p_prev), grad(q))

    a = (
        inner(sigma_s(u), eps(v)) * dx(metadata={"q": 5})
        - biot * p * div(v) * dx(metadata={"q": 5})
        + biot * div(u) * q * dx(metadata={"q": 5})
        + invM * p * q * dx(metadata={"q": 5})
        + theta * dt_c * H_pq * dx(metadata={"q": 5})
    )

    # Terzaghi consolidation is quasi-static in the mechanics block:
    # only the fluid mass balance carries previous-time terms. The
    # displacement/pressure coupling in the equilibrium equation must be
    # enforced at the current step, not incrementally against (u_prev, p_prev).
    L = (
        biot * div(u_prev) * q * dx(metadata={"q": 5})
        + invM * p_prev * q * dx(metadata={"q": 5})
        - (Constant(1.0) - theta) * dt_c * H0_pq * dx(metadata={"q": 5})
        + top_load * v[1] * ds_top
    )

    bcs = [
        BoundaryCondition("ux", "dirichlet", "left", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "right", lambda x, y: 0.0),
        BoundaryCondition("uy", "dirichlet", "bottom", lambda x, y: 0.0),
        BoundaryCondition("p", "dirichlet", "top", lambda x, y: 0.0),
    ]

    compiler = FormCompiler(dh, quadrature_order=None, backend=backend)
    ndofs = dh.total_dofs
    K_lil = sp.lil_matrix((ndofs, ndofs))
    compiler._basis_cache.clear()
    compiler._coeff_cache.clear()
    compiler._collapsed_cache.clear()
    compiler.ctx["rhs"] = False
    compiler._assemble_form(a, K_lil)
    K_raw = K_lil.tocsr()

    dirichlet = dh.get_dirichlet_data(bcs)
    if dirichlet:
        bc_rows = np.fromiter(dirichlet.keys(), dtype=int)
        bc_vals = np.fromiter(dirichlet.values(), dtype=float)
    else:
        bc_rows = np.zeros(0, dtype=int)
        bc_vals = np.zeros(0, dtype=float)
    u_bc = np.zeros(ndofs, dtype=float)
    if bc_rows.size:
        u_bc[bc_rows] = bc_vals
    bc_shift = K_raw @ u_bc

    K_bc = K_lil.copy()
    compiler._apply_bcs(K_bc, np.zeros(ndofs, dtype=float), bcs)
    lu = sp_la.splu(K_bc.tocsc())

    uy_slice = np.asarray(dh.get_field_slice("uy"), dtype=int)
    uy_coords = dh.get_dof_coords("uy")
    top_mask = np.isclose(uy_coords[:, 1], float(params.H))
    if not np.any(top_mask):
        raise RuntimeError("Failed to identify top displacement DOFs for the Terzaghi benchmark.")
    profile_y = np.linspace(0.0, float(params.H), max(257, 4 * ny_cells + 1), dtype=float)
    profile_x = np.full(profile_y.shape, 0.5 * float(params.L), dtype=float)
    profile_eval = _build_scalar_profile_evaluator(
        dh,
        mesh,
        field_name="p",
        points=np.column_stack([profile_x, profile_y]),
    )

    times: list[float] = []
    time_factor: list[float] = []
    mid_pressure_bar: list[float] = []
    mid_pressure_bar_exact: list[float] = []
    settlement_bar: list[float] = []
    settlement_bar_exact: list[float] = []
    pressure_l2_bar_error: list[float] = []
    pressure_linf_bar_error: list[float] = []

    exact_mid_y = 0.5 * float(params.H)
    sample_targets = {
        float(Tv): {
            "distance": float("inf"),
            "entry": None,
        }
        for Tv in sample_Tv
    }

    w_prev = np.zeros(ndofs, dtype=float)
    w_prev[u_prev._g_dofs] = u_prev.nodal_values
    w_prev[p_prev._g_dofs] = p_prev.nodal_values

    for step in range(1, num_time_steps + 1):
        t = float(step) * dt

        compiler._basis_cache.clear()
        compiler._coeff_cache.clear()
        compiler._collapsed_cache.clear()
        compiler.ctx["rhs"] = True
        F_raw = np.zeros(ndofs, dtype=float)
        compiler._assemble_form(L, F_raw)

        F = F_raw - bc_shift
        if bc_rows.size:
            F[bc_rows] = bc_vals

        w = lu.solve(F)

        u_prev.nodal_values = w[u_prev._g_dofs]
        p_prev.nodal_values = w[p_prev._g_dofs]
        w_prev[:] = w

        p_profile = np.asarray(profile_eval(p_prev), dtype=float)
        p_profile_bar = p_profile / float(params.initial_pressure)

        settlement = float(np.mean(np.asarray(w[uy_slice], dtype=float)[top_mask]))
        settlement_bar_num = settlement / float(params.final_settlement)

        Tv = float(params.consolidation_coefficient) * t / (float(params.H) ** 2)
        p_exact_bar = terzaghi_pressure_bar_exact(profile_y, t, params=params, n_terms=n_terms)
        settlement_bar_ex = terzaghi_settlement_bar_exact(t, params=params, n_terms=n_terms)

        err_profile = p_profile_bar - p_exact_bar
        l2_bar = math.sqrt(float(_TRAPEZOID(err_profile * err_profile, profile_y / float(params.H))))
        linf_bar = float(np.max(np.abs(err_profile)))

        p_mid_bar = float(np.interp(exact_mid_y, profile_y, p_profile_bar))
        p_mid_bar_ex = float(terzaghi_pressure_bar_exact(np.asarray([exact_mid_y]), t, params=params, n_terms=n_terms)[0])

        times.append(t)
        time_factor.append(Tv)
        mid_pressure_bar.append(p_mid_bar)
        mid_pressure_bar_exact.append(p_mid_bar_ex)
        settlement_bar.append(settlement_bar_num)
        settlement_bar_exact.append(settlement_bar_ex)
        pressure_l2_bar_error.append(l2_bar)
        pressure_linf_bar_error.append(linf_bar)

        for Tv_target, data in sample_targets.items():
            distance = abs(Tv - float(Tv_target))
            if distance < float(data["distance"]):
                field_l2_bar = float(
                    dh.l2_error(
                        p_prev,
                        exact={"p": (lambda x, y, _t=t: terzaghi_pressure_exact(y, _t, params=params, n_terms=n_terms))},
                        fields=["p"],
                        quad_order=8,
                        relative=False,
                    )
                ) / float(params.initial_pressure)
                data["distance"] = distance
                data["entry"] = {
                    "Tv": Tv,
                    "time": t,
                    "y": np.asarray(profile_y, dtype=float).copy(),
                    "p_bar_num": np.asarray(p_profile_bar, dtype=float).copy(),
                    "p_bar_exact": np.asarray(p_exact_bar, dtype=float).copy(),
                    "p_field_l2_bar": np.asarray([float(field_l2_bar)], dtype=float),
                }

        if print_progress and step % max(1, num_time_steps // 10) == 0:
            print(
                f"[benchmark4-terzaghi] ny={ny_cells:4d} step={step:5d}/{num_time_steps} "
                f"T_v={Tv:.3e}  p_mid={p_mid_bar:.3e}  s_top={settlement_bar_num:.3e}"
            )

    history = {
        "time": np.asarray(times, dtype=float),
        "time_factor": np.asarray(time_factor, dtype=float),
        "mid_pressure_bar": np.asarray(mid_pressure_bar, dtype=float),
        "mid_pressure_bar_exact": np.asarray(mid_pressure_bar_exact, dtype=float),
        "settlement_bar": np.asarray(settlement_bar, dtype=float),
        "settlement_bar_exact": np.asarray(settlement_bar_exact, dtype=float),
        "pressure_l2_bar_error": np.asarray(pressure_l2_bar_error, dtype=float),
        "pressure_linf_bar_error": np.asarray(pressure_linf_bar_error, dtype=float),
    }

    profile_samples: dict[str, dict[str, np.ndarray]] = {}
    for Tv_target, data in sample_targets.items():
        entry = data["entry"]
        if entry is None:
            continue
        profile_samples[_series_label(Tv_target)] = {
            "Tv": np.asarray([float(entry["Tv"])], dtype=float),
            "time": np.asarray([float(entry["time"])], dtype=float),
            "y": np.asarray(entry["y"], dtype=float),
            "p_bar_num": np.asarray(entry["p_bar_num"], dtype=float),
            "p_bar_exact": np.asarray(entry["p_bar_exact"], dtype=float),
            "p_field_l2_bar": np.asarray(entry["p_field_l2_bar"], dtype=float),
        }

    sampled_l2: list[float] = []
    sampled_linf: list[float] = []
    sampled_mid: list[float] = []
    sampled_field_l2: list[float] = []
    for data in profile_samples.values():
        y_hat = np.asarray(data["y"], dtype=float) / float(params.H)
        err = np.asarray(data["p_bar_num"], dtype=float) - np.asarray(data["p_bar_exact"], dtype=float)
        sampled_l2.append(math.sqrt(float(_TRAPEZOID(err * err, y_hat))))
        sampled_linf.append(float(np.max(np.abs(err))))
        sampled_mid.append(float(abs(np.interp(exact_mid_y, np.asarray(data["y"], dtype=float), err))))
        sampled_field_l2.append(float(np.asarray(data["p_field_l2_bar"], dtype=float).ravel()[0]))

    row = {
        "ny": float(ny_cells),
        "nx": float(nx_cells),
        "num_time_steps": float(num_time_steps),
        "dt": float(dt),
        "Tv_final": float(Tv_final),
        "theta_step": float(params.theta_step),
        "max_pbar_l2": float(max(sampled_l2) if sampled_l2 else np.max(history["pressure_l2_bar_error"])),
        "max_pbar_linf": float(max(sampled_linf) if sampled_linf else np.max(history["pressure_linf_bar_error"])),
        "max_pbar_field_l2": float(max(sampled_field_l2) if sampled_field_l2 else float("nan")),
        "max_mid_pressure_bar_error": float(
            max(sampled_mid) if sampled_mid else np.max(np.abs(history["mid_pressure_bar"] - history["mid_pressure_bar_exact"]))
        ),
        "max_settlement_bar_error": float(
            np.max(np.abs(history["settlement_bar"] - history["settlement_bar_exact"]))
        ),
        "final_settlement_bar_error": float(
            abs(history["settlement_bar"][-1] - history["settlement_bar_exact"][-1])
        ),
    }
    return CaseResult(row=row, history=history, profile_samples=profile_samples, params=params)


def _write_summary_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = [
        "ny",
        "nx",
        "num_time_steps",
        "dt",
        "Tv_final",
        "theta_step",
        "max_pbar_l2",
        "max_pbar_linf",
        "max_pbar_field_l2",
        "max_mid_pressure_bar_error",
        "max_settlement_bar_error",
        "final_settlement_bar_error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_history_csv(path: Path, history: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history.keys())
    n = len(np.asarray(history["time"], dtype=float))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i in range(n):
            writer.writerow([f"{float(np.asarray(history[key])[i]):.16e}" for key in fieldnames])


def _write_profile_samples_csv(path: Path, profile_samples: dict[str, dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample", "Tv_actual", "time", "y", "p_bar_num", "p_bar_exact"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for sample, data in profile_samples.items():
            Tv_val = float(np.asarray(data["Tv"], dtype=float)[0])
            t_val = float(np.asarray(data["time"], dtype=float)[0])
            y = np.asarray(data["y"], dtype=float)
            p_num = np.asarray(data["p_bar_num"], dtype=float)
            p_ex = np.asarray(data["p_bar_exact"], dtype=float)
            for yy, pn, pe in zip(y, p_num, p_ex):
                writer.writerow(
                    [
                        sample,
                        f"{Tv_val:.16e}",
                        f"{t_val:.16e}",
                        f"{float(yy):.16e}",
                        f"{float(pn):.16e}",
                        f"{float(pe):.16e}",
                    ]
                )


def _plot_histories(path: Path, *, results: list[CaseResult], dpi: int) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to generate Benchmark 4 plots.")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    colors = ["#145A7B", "#2E8B57", "#BA3F1D", "#6A4C93"]

    for idx, res in enumerate(results):
        color = colors[idx % len(colors)]
        hist = res.history
        label = rf"$n_y={int(res.row['ny'])}$"
        axes[0].plot(hist["time_factor"], hist["mid_pressure_bar"], color=color, lw=1.8, label=label)
        axes[1].plot(hist["time_factor"], hist["settlement_bar"], color=color, lw=1.8, label=label)

    ref = results[-1].history
    axes[0].plot(
        ref["time_factor"],
        ref["mid_pressure_bar_exact"],
        color="black",
        lw=1.6,
        linestyle="--",
        label="analytic",
    )
    axes[1].plot(
        ref["time_factor"],
        ref["settlement_bar_exact"],
        color="black",
        lw=1.6,
        linestyle="--",
        label="analytic",
    )

    axes[0].set_xlabel(r"$T_v = c_v t / H^2$")
    axes[0].set_ylabel(r"$\bar p(H/2,t)$")
    axes[0].set_title("Mid-depth pressure history")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].set_xlabel(r"$T_v = c_v t / H^2$")
    axes[1].set_ylabel(r"$\bar s(t)$")
    axes[1].set_title("Top-settlement history")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _plot_profiles(path: Path, *, finest: CaseResult, dpi: int) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to generate Benchmark 4 plots.")
    samples = sorted(
        finest.profile_samples.items(),
        key=lambda item: float(np.asarray(item[1]["Tv"], dtype=float)[0]),
    )
    ncols = len(samples)
    fig, axes = plt.subplots(1, ncols, figsize=(4.4 * ncols, 4.6), constrained_layout=True, squeeze=False)
    for ax, (label, data) in zip(axes[0], samples):
        y_hat = np.asarray(data["y"], dtype=float) / float(finest.params.H)
        p_num = np.asarray(data["p_bar_num"], dtype=float)
        p_ex = np.asarray(data["p_bar_exact"], dtype=float)
        Tv_actual = float(np.asarray(data["Tv"], dtype=float)[0])
        ax.plot(p_num, y_hat, lw=2.2, color="#145A7B", label="pycutfem")
        ax.plot(p_ex, y_hat, lw=1.8, color="#BA3F1D", linestyle="--", label="analytic")
        ax.set_title(rf"$T_v \approx {Tv_actual:.2f}$")
        ax.set_xlabel(r"$\bar p(y,t)$")
        ax.set_ylabel(r"$y/H$")
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False, loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _plot_error_trends(path: Path, *, rows: list[dict[str, float]], dpi: int) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to generate Benchmark 4 plots.")
    ny = np.asarray([float(row["ny"]) for row in rows], dtype=float)
    p_err = np.asarray([float(row["max_pbar_l2"]) for row in rows], dtype=float)
    s_err = np.asarray([float(row["max_settlement_bar_error"]) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    ax.loglog(ny, p_err, marker="o", color="#145A7B", lw=2.0, label=r"$\max_t \|\bar p_h-\bar p_{\rm ex}\|_{L^2}$")
    ax.loglog(ny, s_err, marker="s", color="#BA3F1D", lw=2.0, label=r"$\max_t |\bar s_h-\bar s_{\rm ex}|$")
    ax.set_xlabel(r"$n_y$")
    ax.set_ylabel("normalized error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def run_benchmark(
    *,
    outdir: str | Path,
    ny_list: list[int],
    nx_cells: int,
    steps_per_ny: int,
    Tv_final: float,
    backend: str,
    n_terms: int,
    sample_Tv: list[float],
    png_dpi: int,
    print_progress: bool,
) -> dict[str, object]:
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    params = TerzaghiParameters()

    results: list[CaseResult] = []
    for ny_cells in ny_list:
        results.append(
            _solve_case(
                ny_cells=int(ny_cells),
                nx_cells=int(nx_cells),
                steps_per_ny=int(steps_per_ny),
                Tv_final=float(Tv_final),
                params=params,
                backend=backend,
                n_terms=int(n_terms),
                sample_Tv=list(sample_Tv),
                print_progress=print_progress,
            )
        )

    rows = [res.row for res in results]
    summary_csv = outdir / "benchmark4_terzaghi_summary.csv"
    history_csv = outdir / f"benchmark4_terzaghi_history_ny{int(rows[-1]['ny']):03d}.csv"
    profiles_csv = outdir / f"benchmark4_terzaghi_profiles_ny{int(rows[-1]['ny']):03d}.csv"
    history_png = outdir / "benchmark4_terzaghi_history.png"
    profiles_png = outdir / "benchmark4_terzaghi_profiles.png"
    errors_png = outdir / "benchmark4_terzaghi_error_trends.png"

    _write_summary_csv(summary_csv, rows)
    _write_history_csv(history_csv, results[-1].history)
    _write_profile_samples_csv(profiles_csv, results[-1].profile_samples)
    _plot_histories(history_png, results=results, dpi=png_dpi)
    _plot_profiles(profiles_png, finest=results[-1], dpi=png_dpi)
    _plot_error_trends(errors_png, rows=rows, dpi=png_dpi)

    payload = {
        "paper1_scope": "alpha-independent poroelastic response benchmark",
        "params": asdict(params),
        "summary_csv": str(summary_csv),
        "history_csv": str(history_csv),
        "profiles_csv": str(profiles_csv),
        "history_png": str(history_png),
        "profiles_png": str(profiles_png),
        "errors_png": str(errors_png),
        "rows": rows,
    }
    (outdir / "benchmark4_terzaghi_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper 1 Benchmark 4: Terzaghi consolidation (alpha-independent poroelastic layer).")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--ny-list", type=str, default="32,64,128")
    ap.add_argument("--nx-cells", type=int, default=4)
    ap.add_argument("--steps-per-ny", type=int, default=8)
    ap.add_argument("--Tv-final", type=float, default=1.0)
    ap.add_argument("--backend", choices=("jit", "python", "cpp"), default="jit")
    ap.add_argument("--n-terms", type=int, default=400)
    ap.add_argument("--sample-tv", type=str, default="0.20,0.50,1.00")
    ap.add_argument("--png-dpi", type=int, default=220)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    payload = run_benchmark(
        outdir=args.outdir,
        ny_list=_parse_int_list(args.ny_list),
        nx_cells=int(args.nx_cells),
        steps_per_ny=int(args.steps_per_ny),
        Tv_final=float(args.Tv_final),
        backend=str(args.backend),
        n_terms=int(args.n_terms),
        sample_Tv=_parse_float_list(args.sample_tv),
        png_dpi=int(args.png_dpi),
        print_progress=not bool(args.quiet),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
