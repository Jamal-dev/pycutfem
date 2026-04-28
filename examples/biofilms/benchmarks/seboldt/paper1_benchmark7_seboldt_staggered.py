#!/usr/bin/env python3
"""Paper 1 Benchmark 7: Seboldt Example 2 solved by a staggered split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional in lightweight envs
    matplotlib = None
    plt = None

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import paper1_benchmark7_seboldt as mono

from pycutfem.io.vtk import export_vtk
from pycutfem.solvers.coupling_acceleration import create_coupling_accelerator
from pycutfem.solvers.nonlinear_solver import (
    InteriorPointNewtonSolver,
    LinearSolverParameters,
    NewtonParameters,
    NewtonSolver,
    PdasNewtonSolver,
    VIParameters,
)
from pycutfem.ufl.expressions import Constant, Function, HdivFunction, VectorFunction
from pycutfem.ufl.forms import BoundaryCondition


def _arg_provided(argv: list[str], *option_names: str) -> bool:
    prefixes = tuple(f"{name}=" for name in option_names)
    for token in argv:
        if token in option_names:
            return True
        if any(token.startswith(prefix) for prefix in prefixes):
            return True
    return False


def _base_parse_args(argv: list[str]) -> argparse.Namespace:
    prev_argv = sys.argv[:]
    try:
        sys.argv = [prev_argv[0]] + list(argv)
        return mono._parse_args()
    finally:
        sys.argv = prev_argv


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    split_parser = argparse.ArgumentParser(
        add_help=True,
        description=(
            "Benchmark 7 staggered/operator-splitting driver. "
            "All mesh/material/closure flags from paper1_benchmark7_seboldt.py "
            "remain available; only the split-specific options are listed here."
        ),
    )
    split_parser.add_argument(
        "--outer-it",
        type=int,
        default=8,
        help=(
            "Maximum number of staggered fixed-point sweeps per time step. "
            "The default mirrors the NIRB-style outer coupling loop and gives the split solve "
            "enough room to settle smoothly."
        ),
    )
    split_parser.add_argument(
        "--outer-tol",
        type=float,
        default=1.0e-2,
        help="Scaled coupling-field fixed-point residual threshold.",
    )
    split_parser.add_argument(
        "--outer-abs-tol",
        type=float,
        default=0.0,
        help=(
            "Optional absolute coupling-field fixed-point residual threshold. "
            "Set <= 0 to disable the absolute check."
        ),
    )
    split_parser.add_argument(
        "--outer-update",
        type=str,
        default="aitken",
        choices=("constant", "aitken", "iqn_ils", "iqln", "mvqn"),
        help="Outer fixed-point update used after each staggered sweep.",
    )
    split_parser.add_argument(
        "--outer-history",
        type=int,
        default=6,
        help="History length used by the IQN/IQLN-style outer coupling update.",
    )
    split_parser.add_argument(
        "--outer-timestep-history",
        type=int,
        default=1,
        help="Accepted time-step horizon reused by the outer IQN-ILS history; MVQN reuses its Jacobian implicitly.",
    )
    split_parser.add_argument(
        "--outer-regularization",
        type=float,
        default=1.0e-10,
        help="Tikhonov regularization used in the outer IQN/IQLN least-squares system.",
    )
    split_parser.add_argument(
        "--outer-relaxation",
        type=float,
        default=0.5,
        help="Base fixed-point relaxation used by the outer coupling update.",
    )
    split_parser.add_argument(
        "--outer-relaxation-min",
        type=float,
        default=1.0e-3,
        help="Minimum relaxation allowed by the outer fixed-point update.",
    )
    split_parser.add_argument(
        "--outer-relaxation-max",
        type=float,
        default=1.0,
        help="Maximum relaxation allowed by the outer fixed-point update.",
    )
    split_parser.add_argument(
        "--stage-newton-tol",
        type=float,
        default=None,
        help=(
            "Absolute residual tolerance used by the staggered stage solvers. "
            "Defaults to --newton-tol when unset."
        ),
    )
    split_parser.add_argument(
        "--stage-newton-rtol",
        type=float,
        default=0.0,
        help=(
            "Relative residual tolerance used by the staggered stage solvers. "
            "Defaults to 0 so the split stages stop on an absolute residual threshold."
        ),
    )
    split_parser.add_argument(
        "--stage-accept-nonconverged-atol-factor",
        type=float,
        default=1.0,
        help=(
            "Relaxed near-tolerance acceptance factor for staggered stage solvers. "
            "Defaults to 1 so best-effort globalization is only accepted when the "
            "updated residual is already within the absolute Newton tolerance."
        ),
    )
    split_parser.add_argument(
        "--stage-order",
        type=str,
        default="flow,solid,transport",
        help=(
            "Comma-separated Gauss-Seidel stage order using each of flow, solid, transport exactly once. "
            "Default: flow,solid,transport. With post-accept transport enabled, the leading flow/solid pair "
            "defines the fixed-point coupling order and transport is updated after the coupled sweep accepts."
        ),
    )
    split_parser.add_argument(
        "--flow-solid-substeps",
        type=int,
        default=1,
        help=(
            "If the stage order is flow,solid,transport or solid,flow,transport, "
            "repeat the leading solid-hydro pair this many times before each transport solve."
        ),
    )
    split_parser.add_argument(
        "--interface-robin-l",
        type=float,
        default=2000.0,
        help=(
            "Generalized Robin combination parameter L used in the split free-fluid / poro-solid coupling. "
            "This is distinct from the physical fluid-entry resistance delta passed by --interface-entry-delta."
        ),
    )
    split_parser.add_argument(
        "--solid-solver",
        type=str,
        default="newton",
        choices=("newton", "pdas", "ipm"),
        help="Nonlinear solver used on the poro-solid/reference-map stage.",
    )
    split_parser.add_argument(
        "--transport-solver",
        type=str,
        default="pdas",
        choices=("newton", "pdas", "ipm"),
        help="Nonlinear solver used on the support-transport stage.",
    )
    split_parser.add_argument(
        "--flow-solver",
        type=str,
        default="newton",
        choices=("newton", "pdas", "ipm"),
        help="Nonlinear solver used on the free-fluid stage.",
    )
    split_parser.add_argument(
        "--solid-max-it",
        type=int,
        default=None,
        help="Optional Newton/VI iteration cap for the solid stage. Defaults to --max-it.",
    )
    split_parser.add_argument(
        "--transport-max-it",
        type=int,
        default=None,
        help="Optional Newton/VI iteration cap for the transport stage. Defaults to --max-it.",
    )
    split_parser.add_argument(
        "--flow-max-it",
        type=int,
        default=None,
        help="Optional Newton/VI iteration cap for the flow stage. Defaults to --max-it.",
    )
    split_parser.add_argument(
        "--transport-post-accept",
        dest="transport_post_accept",
        action="store_true",
        help=(
            "Freeze transport during the fluid/poro-solid fixed-point and update it once after the coupled "
            "sweep accepts. This matches the intended two-solver interface map."
        ),
    )
    split_parser.add_argument(
        "--no-transport-post-accept",
        dest="transport_post_accept",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    split_parser.set_defaults(transport_post_accept=True)
    split_parser.add_argument(
        "--startup-preload-sweeps",
        type=int,
        default=0,
        help=(
            "Before the first strict split step, run this many relaxed staggered preload sweeps "
            "to build a smoother startup guess. Set 0 to disable."
        ),
    )
    split_parser.add_argument(
        "--startup-preload-accept-factor",
        type=float,
        default=5.0e3,
        help=(
            "Temporary Newton near-tolerance accept factor used only during the first-step preload "
            "sweeps so the split startup can keep materially improved stage states."
        ),
    )
    split_parser.add_argument(
        "--startup-flow-scales",
        type=str,
        default="0.025,0.1,0.25,0.5,1.0",
        help=(
            "Comma-separated continuation scales used for every first-step flow stage solve, "
            "including preload sweeps. The final value is forced to 1.0."
        ),
    )
    split_parser.add_argument(
        "--startup-preload-prev-blend",
        type=float,
        default=1.0,
        help=(
            "After the first-step preload, keep the transport fields but blend the remaining "
            "fields back toward the previous accepted state. 1.0 keeps the staged state."
        ),
    )
    split_parser.add_argument(
        "--freeze-B-total-mixture-continuity-diagnostic",
        action="store_true",
        help=(
            "Diagnostic switch: keep B frozen and replace the split porous p_pore row "
            "by the total-mixture continuity div(F v + q + alpha vS)=0 in the poro-solid stage."
        ),
    )
    split_args, remaining = split_parser.parse_known_args(raw_argv)
    args = _base_parse_args(remaining)

    staged_defaults: list[tuple[tuple[str, ...], str, object]] = [
        (("--outdir",), "outdir", "out/benchmark7_seboldt_staggered"),
        (("--full-ratio-free-state", "--no-full-ratio-free-state"), "full_ratio_free_state", True),
        (("--enable-phi-evolution", "--no-enable-phi-evolution"), "enable_phi_evolution", True),
        (("--split-primary-darcy-flux", "--no-split-primary-darcy-flux"), "split_primary_darcy_flux", True),
        (("--fluid-space",), "fluid_space", "cg"),
        (("--support-physics",), "support_physics", "stored_support"),
        (("--geometry-indicator-beta",), "geometry_indicator_beta", 12.0),
        (("--solid-bc-mode",), "solid_bc_mode", "lateral_clamped"),
        (("--drag-formulation",), "drag_formulation", "mixed_lm"),
        (("--alpha-regularization",), "alpha_regularization", "none"),
        (("--alpha-advect-with",), "alpha_advect_with", "vS"),
        (("--alpha-advection-form",), "alpha_advection_form", "advective"),
        (("--storativity-c0",), "storativity_c0", 1.0e-3),
        (("--alpha-box-constraints", "--no-alpha-box-constraints"), "alpha_box_constraints", True),
        (("--pressure-mean-constraint", "--no-pressure-mean-constraint"), "pressure_mean_constraint", True),
        (("--pressure-mean-gauge", "--no-pressure-mean-gauge"), "pressure_mean_gauge", False),
        (("--pressure-interface-closure", "--no-pressure-interface-closure"), "pressure_interface_closure", False),
        (("--p-pore-fluid-gauge", "--no-p-pore-fluid-gauge"), "p_pore_fluid_gauge", True),
        (("--interface-entry-closure", "--no-interface-entry-closure"), "interface_entry_closure", True),
        (("--interface-bjs-closure", "--no-interface-bjs-closure"), "interface_bjs_closure", False),
        (
            ("--interface-velocity-continuity-closure", "--no-interface-velocity-continuity-closure"),
            "interface_velocity_continuity_closure",
            True,
        ),
        (("--interface-velocity-tangential-strength",), "interface_velocity_tangential_strength", 0.0),
        (
            ("--interface-traction-continuity-closure", "--no-interface-traction-continuity-closure"),
            "interface_traction_continuity_closure",
            False,
        ),
        (("--interface-traction-normal-strength",), "interface_traction_normal_strength", 1.0),
        (("--interface-traction-tangential-strength",), "interface_traction_tangential_strength", 0.0),
    ]
    for option_names, attr, value in staged_defaults:
        if not _arg_provided(raw_argv, *option_names):
            setattr(args, attr, value)

    for key, value in vars(split_args).items():
        setattr(args, key, value)

    args._staggered_entry_delta_user_provided = bool(_arg_provided(raw_argv, "--interface-entry-delta"))
    args._allow_seboldt_interface_split_combo = True
    args._allow_one_domain_interface_traction_penalty = True
    args = mono._normalize_benchmark7_solver_choice(args)
    return args


def _validate_supported_configuration(args: argparse.Namespace) -> None:
    if bool(getattr(args, "latent_bounded_transport", False)):
        raise NotImplementedError(
            "The staggered Benchmark 7 driver does not support --latent-bounded-transport. "
            "Use the physical alpha/B fields directly."
        )
    if bool(getattr(args, "alpha_from_refmap", False)):
        raise NotImplementedError(
            "The staggered Benchmark 7 driver does not support --alpha-from-refmap. "
            "Use the stored-support alpha/B transport branch."
        )
    if str(getattr(args, "fluid_space", "cg")).strip().lower() != "cg":
        raise NotImplementedError(
            "The staggered Benchmark 7 driver currently supports only --fluid-space=cg because "
            "the explicit interface traction-transfer form is currently wired on the CG velocity branch."
        )
    if not bool(getattr(args, "full_ratio_free_state", False)):
        raise ValueError(
            "The staggered Benchmark 7 driver is wired for the full ratio-free stored-support q-primary branch "
            "with split pressures p/p_pore."
        )
    if not bool(mono._split_primary_darcy_flux_enabled(args)):
        raise ValueError(
            "The staggered Benchmark 7 driver requires --split-primary-darcy-flux on the full ratio-free "
            "stored-support branch so the free-fluid stage solves only (v,p) while the poro-solid stage owns "
            "(p_pore,q,vS,u)."
        )


def _filter_bcs_by_fields(
    bcs_in: list[BoundaryCondition],
    field_names: list[str] | tuple[str, ...],
) -> list[BoundaryCondition]:
    allowed = {str(name) for name in list(field_names or [])}
    return [bc for bc in list(bcs_in or []) if str(getattr(bc, "field", "")) in allowed]


def _scaled_dirichlet_bcs(
    bcs_in: list[BoundaryCondition],
    *,
    scale: float,
    scaled_fields: tuple[str, ...] = ("v", "v_x", "v_y"),
) -> list[BoundaryCondition]:
    theta = float(scale)
    targets = {str(name) for name in tuple(scaled_fields)}
    out: list[BoundaryCondition] = []
    for bc in list(bcs_in or []):
        if str(getattr(bc, "method", "")).lower() != "dirichlet" or str(getattr(bc, "field", "")) not in targets:
            out.append(bc)
            continue
        value_fn = bc.value
        if callable(value_fn):
            def _scaled_value(x, y, _f=value_fn, _theta=theta):
                return float(_theta) * float(_f(x, y))
        else:
            raw = float(value_fn)
            def _scaled_value(x, y, _raw=raw, _theta=theta):
                del x, y
                return float(_theta) * float(_raw)
        out.append(BoundaryCondition(bc.field, bc.method, bc.domain_tag, _scaled_value))
    return out


def _sum_forms(*parts):
    kept = [part for part in parts if part is not None]
    if not kept:
        raise ValueError("Cannot build a stage form from an empty part list.")
    total = kept[0]
    for part in kept[1:]:
        total = total + part
    return total


def _find_named_function(funcs_in, template):
    for func in list(funcs_in or []):
        if str(getattr(func, "name", "")) == str(getattr(template, "name", "")):
            return func
    return template


def _snapshot_function_values(functions) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(func.nodal_values, dtype=float).copy() for func in list(functions or []))


def _clone_function_like(template, *, name: str):
    if isinstance(template, VectorFunction):
        return VectorFunction(name, list(template.field_names), template._dof_handler)
    if isinstance(template, HdivFunction):
        return HdivFunction(name, template.field_name, template._dof_handler)
    if isinstance(template, Function):
        return Function(name, template.field_name, template._dof_handler)
    raise TypeError(f"Unsupported function template type: {type(template)!r}")


def _copy_function_values(source, target) -> None:
    src = np.asarray(source.nodal_values, dtype=float).ravel()
    dst = np.asarray(target.nodal_values, dtype=float).ravel()
    if src.size != dst.size:
        raise ValueError(
            f"Cannot copy function values: size mismatch {src.size} -> {dst.size} for "
            f"{getattr(source, 'name', '?')} -> {getattr(target, 'name', '?')}."
        )
    target.nodal_values[:] = src


def _copy_function_list_values(sources, targets) -> None:
    src_list = list(sources or [])
    dst_list = list(targets or [])
    if len(src_list) != len(dst_list):
        raise ValueError("Cannot copy function list values: source/target count mismatch.")
    for source, target in zip(src_list, dst_list):
        _copy_function_values(source, target)


def _build_flow_coupling_state(problem: dict[str, object]) -> None:
    flow_coupling_v = _clone_function_like(problem["v_n"], name="v_flow_coupling")
    flow_coupling_p = _clone_function_like(problem["p_n"], name="p_flow_coupling")
    _copy_function_values(problem["v_n"], flow_coupling_v)
    _copy_function_values(problem["p_n"], flow_coupling_p)
    problem["_staggered_flow_coupling_v"] = flow_coupling_v
    problem["_staggered_flow_coupling_p"] = flow_coupling_p
    problem["_staggered_flow_coupling_scale"] = mono._named_constant("b7_staggered_flow_coupling_scale", 1.0)


def _build_interface_transfer_state(problem: dict[str, object]) -> None:
    template = problem["p_n"]
    problem["_staggered_if_entry_drive"] = _clone_function_like(template, name="if_entry_drive")
    problem["_staggered_if_entry_drive_candidate"] = _clone_function_like(template, name="if_entry_drive_candidate")
    problem["_staggered_if_porous_flux_n"] = _clone_function_like(template, name="if_porous_flux_n")
    problem["_staggered_if_porous_flux_n_candidate"] = _clone_function_like(template, name="if_porous_flux_n_candidate")
    problem["_staggered_if_porous_traction_n"] = _clone_function_like(template, name="if_porous_traction_n")
    problem["_staggered_if_porous_traction_n_candidate"] = _clone_function_like(template, name="if_porous_traction_n_candidate")
    problem["_staggered_if_free_traction_n"] = _clone_function_like(template, name="if_free_traction_n")
    problem["_staggered_if_free_traction_n_candidate"] = _clone_function_like(template, name="if_free_traction_n_candidate")
    problem["_staggered_if_free_flux_n"] = _clone_function_like(template, name="if_free_flux_n")
    problem["_staggered_if_free_flux_n_candidate"] = _clone_function_like(template, name="if_free_flux_n_candidate")
    for key in (
        "_staggered_if_entry_drive",
        "_staggered_if_entry_drive_candidate",
        "_staggered_if_porous_flux_n",
        "_staggered_if_porous_flux_n_candidate",
        "_staggered_if_porous_traction_n",
        "_staggered_if_porous_traction_n_candidate",
        "_staggered_if_free_traction_n",
        "_staggered_if_free_traction_n_candidate",
        "_staggered_if_free_flux_n",
        "_staggered_if_free_flux_n_candidate",
    ):
        problem[key].nodal_values[:] = 0.0


def _commit_flow_interface_response(problem: dict[str, object]) -> None:
    for state_key, candidate_key in (
        ("_staggered_if_free_traction_n", "_staggered_if_free_traction_n_candidate"),
        ("_staggered_if_free_flux_n", "_staggered_if_free_flux_n_candidate"),
    ):
        state = problem.get(state_key)
        candidate = problem.get(candidate_key)
        if state is None or candidate is None:
            continue
        _copy_function_values(candidate, state)


def _commit_porous_interface_response(problem: dict[str, object]) -> None:
    for state_key, candidate_key in (
        ("_staggered_if_porous_traction_n", "_staggered_if_porous_traction_n_candidate"),
    ):
        state = problem.get(state_key)
        candidate = problem.get(candidate_key)
        if state is None or candidate is None:
            continue
        _copy_function_values(candidate, state)


def _interface_state_functions(problem: dict[str, object]) -> list[object]:
    funcs: list[object] = []
    if bool(problem.get("_staggered_state_use_porous_flux_n", True)) and problem.get("_staggered_if_porous_flux_n") is not None:
        funcs.append(problem["_staggered_if_porous_flux_n"])
    if bool(problem.get("_staggered_state_use_porous_traction_n", False)) and problem.get("_staggered_if_porous_traction_n") is not None:
        funcs.append(problem["_staggered_if_porous_traction_n"])
    if bool(problem.get("_staggered_state_use_free_traction_n", False)) and problem.get("_staggered_if_free_traction_n") is not None:
        funcs.append(problem["_staggered_if_free_traction_n"])
    if bool(problem.get("_staggered_state_use_free_flux_n", False)) and problem.get("_staggered_if_free_flux_n") is not None:
        funcs.append(problem["_staggered_if_free_flux_n"])
    return funcs


def _interface_candidate_functions(problem: dict[str, object]) -> list[object]:
    funcs: list[object] = []
    if bool(problem.get("_staggered_state_use_porous_flux_n", True)) and problem.get("_staggered_if_porous_flux_n_candidate") is not None:
        funcs.append(problem["_staggered_if_porous_flux_n_candidate"])
    if bool(problem.get("_staggered_state_use_porous_traction_n", False)) and problem.get("_staggered_if_porous_traction_n_candidate") is not None:
        funcs.append(problem["_staggered_if_porous_traction_n_candidate"])
    if bool(problem.get("_staggered_state_use_free_traction_n", False)) and problem.get("_staggered_if_free_traction_n_candidate") is not None:
        funcs.append(problem["_staggered_if_free_traction_n_candidate"])
    if bool(problem.get("_staggered_state_use_free_flux_n", False)) and problem.get("_staggered_if_free_flux_n_candidate") is not None:
        funcs.append(problem["_staggered_if_free_flux_n_candidate"])
    return funcs


def _compute_interface_diag(
    *,
    args: argparse.Namespace,
    problem: dict[str, object],
    eps_alpha: float,
    interface_entry_delta: float,
) -> dict[str, object]:
    return mono._compute_interface_probe_diagnostics(
        problem=problem,
        Lx=float(args.Lx),
        y_interface=float(args.y_interface),
        y_profile=float(args.y_profile),
        eps_alpha=float(eps_alpha),
        mu_f=float(args.mu_f),
        mu_s=float(args.mu_s),
        lambda_s=float(args.lambda_s),
        solid_visco_eta=float(args.solid_visco_eta),
        solid_model=str(args.solid_model),
        skeleton_pressure_mode=str(args.skeleton_pressure_mode),
        alpha_biot=(None if args.alpha_biot is None else float(args.alpha_biot)),
        interface_entry_delta=float(interface_entry_delta),
        interface_bjs_gamma=float(getattr(args, "interface_bjs_gamma", 1.0e3)),
    )


def _profile_rows_to_xy(rows: list[dict[str, float]], value_key: str) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    buckets: dict[float, list[float]] = {}
    for row in list(rows):
        try:
            x_val = float(row.get("x", float("nan")))
            y_val = float(row.get(value_key, float("nan")))
        except Exception:
            continue
        if not (np.isfinite(x_val) and np.isfinite(y_val)):
            continue
        buckets.setdefault(float(x_val), []).append(float(y_val))
    if not buckets:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    x_sorted = np.asarray(sorted(buckets), dtype=float)
    y_sorted = np.asarray([float(np.mean(buckets[float(xv)])) for xv in x_sorted], dtype=float)
    return x_sorted, y_sorted


def _interface_transfer_rows(interface_diag: dict[str, object]) -> list[dict[str, float]]:
    rows = list(interface_diag.get("interface_profile_rows") or [])
    if rows:
        return rows
    return list(interface_diag.get("interface_line_rows") or [])


def _set_scalar_function_from_profile(target, *, x_samples: np.ndarray, values: np.ndarray) -> None:
    coords = np.asarray(target._dof_handler.get_dof_coords(target.field_name), dtype=float)
    out = np.zeros((coords.shape[0],), dtype=float)
    x_samples = np.asarray(x_samples, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    if x_samples.size > 0 and values.size == x_samples.size:
        out[:] = np.interp(coords[:, 0], x_samples, values, left=float(values[0]), right=float(values[-1]))
    target.nodal_values[:] = out


def _update_interface_transfer_function_from_rows(problem: dict[str, object], *, target_key: str, rows: list[dict[str, float]], value_key: str) -> None:
    target = problem.get(target_key)
    if target is None:
        return
    x_samples, y_samples = _profile_rows_to_xy(list(rows or []), value_key)
    _set_scalar_function_from_profile(target, x_samples=x_samples, values=y_samples)


def _refresh_flow_to_solid_transfer(
    *,
    args: argparse.Namespace,
    problem: dict[str, object],
    eps_alpha: float,
    interface_entry_delta: float,
) -> dict[str, object]:
    interface_diag = _compute_interface_diag(
        args=args,
        problem=problem,
        eps_alpha=float(eps_alpha),
        interface_entry_delta=float(interface_entry_delta),
    )
    _update_interface_transfer_function_from_rows(
        problem,
        target_key="_staggered_if_free_traction_n_candidate",
        rows=_interface_transfer_rows(interface_diag),
        value_key="traction_free_n",
    )
    _update_interface_transfer_function_from_rows(
        problem,
        target_key="_staggered_if_free_flux_n_candidate",
        rows=_interface_transfer_rows(interface_diag),
        value_key="free_flux_n",
    )
    return interface_diag


def _refresh_solid_interface_candidate(
    *,
    args: argparse.Namespace,
    problem: dict[str, object],
    eps_alpha: float,
    interface_entry_delta: float,
) -> dict[str, object]:
    interface_diag = _compute_interface_diag(
        args=args,
        problem=problem,
        eps_alpha=float(eps_alpha),
        interface_entry_delta=float(interface_entry_delta),
    )
    _update_interface_transfer_function_from_rows(
        problem,
        target_key="_staggered_if_entry_drive_candidate",
        rows=_interface_transfer_rows(interface_diag),
        value_key="entry_drive",
    )
    _update_interface_transfer_function_from_rows(
        problem,
        target_key="_staggered_if_porous_flux_n_candidate",
        rows=_interface_transfer_rows(interface_diag),
        value_key="porous_flux_n",
    )
    _update_interface_transfer_function_from_rows(
        problem,
        target_key="_staggered_if_porous_traction_n_candidate",
        rows=_interface_transfer_rows(interface_diag),
        value_key="traction_porous_support_n",
    )
    return interface_diag


def _interface_transfer_field_scales(
    problem: dict[str, object],
    *,
    interface_diag: dict[str, object],
) -> dict[str, float]:
    summary = dict(interface_diag.get("summary", {}) or {})
    v_scale = max(
        1.0e-12,
        float(np.max(np.abs(problem["v_k"].nodal_values))),
        float(np.max(np.abs(problem["vS_k"].nodal_values))),
        float(np.max(np.abs(problem["q_flux_k"].nodal_values))) if problem.get("q_flux_k") is not None else 0.0,
        abs(float(summary.get("centerline_at_y_interface_free_flux_n", 0.0))),
        abs(float(summary.get("centerline_at_y_interface_q_n", 0.0))),
    )
    p_scale = max(
        1.0e-12,
        float(np.max(np.abs(problem["p_k"].nodal_values))),
        float(np.max(np.abs(problem["p_pore_k"].nodal_values))) if problem.get("p_pore_k") is not None else 0.0,
        abs(float(summary.get("centerline_at_y_interface_traction_free_n", 0.0))),
        abs(float(summary.get("centerline_at_y_interface_p_pore_support", 0.0))),
    )
    return {
        "if_entry_drive": float(p_scale),
        "if_porous_flux_n": float(v_scale),
        "if_porous_traction_n": float(p_scale),
        "if_free_traction_n": float(p_scale),
        "if_free_flux_n": float(v_scale),
    }


def _interface_state_delta(
    problem: dict[str, object],
    *,
    interface_diag: dict[str, object],
) -> tuple[float, float, dict[str, float], np.ndarray, np.ndarray]:
    state_funcs = _interface_state_functions(problem)
    candidate_funcs = _interface_candidate_functions(problem)
    if len(state_funcs) != len(candidate_funcs):
        raise ValueError("Interface state/candidate count mismatch.")
    scales = _interface_transfer_field_scales(problem, interface_diag=interface_diag)
    by_field: dict[str, float] = {}
    max_scaled = 0.0
    max_abs = 0.0
    for state_func, candidate_func in zip(state_funcs, candidate_funcs):
        name = str(getattr(state_func, "name", ""))
        state_vals = np.asarray(state_func.nodal_values, dtype=float).ravel()
        cand_vals = np.asarray(candidate_func.nodal_values, dtype=float).ravel()
        if state_vals.size != cand_vals.size:
            raise ValueError(f"Interface state size mismatch for {name!r}.")
        abs_delta = float(np.max(np.abs(cand_vals - state_vals))) if state_vals.size else 0.0
        scale = max(float(scales.get(name, 1.0) or 1.0), 1.0e-30)
        scaled = float(abs_delta / scale)
        by_field[name] = scaled
        max_scaled = max(max_scaled, scaled)
        max_abs = max(max_abs, abs_delta)
    x_curr = _functions_to_vector(state_funcs)
    residual_vec = _functions_to_vector(candidate_funcs) - x_curr
    return float(max_scaled), float(max_abs), by_field, residual_vec, x_curr


def _restore_function_values(functions, snapshot: tuple[np.ndarray, ...]) -> None:
    if len(list(functions or [])) != len(tuple(snapshot or tuple())):
        raise ValueError("Cannot restore function values: snapshot/function count mismatch.")
    for func, values in zip(list(functions or []), list(snapshot or [])):
        func.nodal_values[:] = np.asarray(values, dtype=float).ravel()


def _copy_current_into_previous(current_functions, previous_functions) -> None:
    for cur, prev in zip(list(current_functions or []), list(previous_functions or [])):
        prev.nodal_values[:] = np.asarray(cur.nodal_values, dtype=float).ravel()


def _copy_previous_into_current(current_functions, previous_functions) -> None:
    for cur, prev in zip(list(current_functions or []), list(previous_functions or [])):
        cur.nodal_values[:] = np.asarray(prev.nodal_values, dtype=float).ravel()


def _max_snapshot_delta(functions, snapshot: tuple[np.ndarray, ...]) -> float:
    max_delta = 0.0
    for func, values in zip(list(functions or []), list(snapshot or [])):
        delta = float(np.max(np.abs(np.asarray(func.nodal_values, dtype=float).ravel() - np.asarray(values, dtype=float).ravel())))
        max_delta = max(max_delta, delta)
    return max_delta


def _flow_solution_functions(problem: dict[str, object]) -> list[object]:
    return [problem["v_k"], problem["p_k"]]


def _flow_coupling_functions(problem: dict[str, object]) -> list[object]:
    funcs: list[object] = []
    if problem.get("_staggered_flow_coupling_v") is not None:
        funcs.append(problem["_staggered_flow_coupling_v"])
    if problem.get("_staggered_flow_coupling_p") is not None:
        funcs.append(problem["_staggered_flow_coupling_p"])
    return funcs


def _reset_flow_coupling_from_previous(problem: dict[str, object]) -> None:
    coupling_funcs = _flow_coupling_functions(problem)
    if not coupling_funcs:
        return
    _copy_function_list_values([problem["v_n"], problem["p_n"]], coupling_funcs)


def _commit_flow_coupling_from_current(problem: dict[str, object]) -> None:
    coupling_funcs = _flow_coupling_functions(problem)
    if not coupling_funcs:
        return
    _copy_function_list_values(_flow_solution_functions(problem), coupling_funcs)


def _copy_flow_coupling_into_current(problem: dict[str, object]) -> None:
    coupling_funcs = _flow_coupling_functions(problem)
    if not coupling_funcs:
        return
    _copy_function_list_values(coupling_funcs, _flow_solution_functions(problem))


def _set_flow_coupling_scale(problem: dict[str, object], *, scale: float) -> None:
    target = problem.get("_staggered_flow_coupling_scale")
    if target is not None:
        target.value = float(scale)


def _relax_flow_coupling_toward_current(problem: dict[str, object], *, omega: float) -> None:
    theta = float(np.clip(float(omega), 0.0, 1.0))
    if theta <= 0.0:
        return
    current_funcs = _flow_solution_functions(problem)
    coupling_funcs = _flow_coupling_functions(problem)
    if len(current_funcs) != len(coupling_funcs):
        raise ValueError("Cannot relax flow coupling state: source/target count mismatch.")
    for current, coupling in zip(current_funcs, coupling_funcs):
        old = np.asarray(coupling.nodal_values, dtype=float).ravel()
        cur = np.asarray(current.nodal_values, dtype=float).ravel()
        if old.size != cur.size:
            raise ValueError(
                f"Cannot relax flow coupling state: size mismatch {old.size} vs {cur.size} for "
                f"{getattr(current, 'name', '?')}."
            )
        coupling.nodal_values[:] = old + theta * (cur - old)


def _flow_coupling_delta(
    problem: dict[str, object],
    *,
    field_scales: dict[str, float] | None = None,
) -> tuple[float, float, dict[str, float], np.ndarray]:
    current_funcs = _flow_solution_functions(problem)
    coupling_snapshot = _snapshot_function_values(_flow_coupling_functions(problem))
    delta_inf, delta_abs_inf, delta_by_field = _scaled_snapshot_delta(
        current_funcs,
        coupling_snapshot,
        field_scales=field_scales,
    )
    residual_vec = _functions_to_vector(current_funcs) - _snapshot_to_vector(current_funcs, coupling_snapshot)
    return delta_inf, delta_abs_inf, delta_by_field, residual_vec


def _set_function_vector(functions, values: np.ndarray) -> None:
    flat = np.asarray(values, dtype=float).ravel()
    offset = 0
    for func in list(functions or []):
        cur = np.asarray(func.nodal_values, dtype=float).ravel()
        size = int(cur.size)
        if offset + size > flat.size:
            raise ValueError("Cannot scatter vector into functions: input vector too short.")
        func.nodal_values[:] = flat[offset : offset + size]
        offset += size
    if offset != flat.size:
        raise ValueError("Cannot scatter vector into functions: input vector has trailing entries.")


def _set_selected_function_vector(
    functions,
    values: np.ndarray,
    *,
    field_names: set[str] | list[str] | tuple[str, ...] | None = None,
) -> None:
    selected = None if field_names is None else {str(name) for name in set(field_names)}
    flat = np.asarray(values, dtype=float).ravel()
    offset = 0
    for func in list(functions or []):
        func_name = str(getattr(func, "name", ""))
        if selected is not None and func_name not in selected:
            continue
        cur = np.asarray(func.nodal_values, dtype=float).ravel()
        size = int(cur.size)
        if offset + size > flat.size:
            raise ValueError("Cannot scatter vector into selected functions: input vector too short.")
        func.nodal_values[:] = flat[offset : offset + size]
        offset += size
    if offset != flat.size:
        raise ValueError("Cannot scatter vector into selected functions: input vector has trailing entries.")


def _outer_interface_physics(
    *,
    args: argparse.Namespace,
    problem: dict[str, object],
    eps_alpha: float,
    interface_entry_delta: float,
) -> tuple[float, dict[str, float]]:
    interface_diag = mono._compute_interface_probe_diagnostics(
        problem=problem,
        Lx=float(args.Lx),
        y_interface=float(args.y_interface),
        y_profile=float(args.y_profile),
        eps_alpha=float(eps_alpha),
        mu_f=float(args.mu_f),
        mu_s=float(args.mu_s),
        lambda_s=float(args.lambda_s),
        solid_visco_eta=float(args.solid_visco_eta),
        solid_model=str(args.solid_model),
        skeleton_pressure_mode=str(args.skeleton_pressure_mode),
        alpha_biot=(None if args.alpha_biot is None else float(args.alpha_biot)),
        interface_entry_delta=float(interface_entry_delta),
        interface_bjs_gamma=float(getattr(args, "interface_bjs_gamma", 1.0e3)),
    )
    summary = dict(interface_diag.get("summary", {}) or {})
    rows = _interface_transfer_rows(interface_diag)
    v_scale = max(
        1.0e-12,
        float(np.max(np.abs(problem["v_k"].nodal_values))),
        float(np.max(np.abs(problem["vS_k"].nodal_values))),
        float(np.max(np.abs(problem["q_flux_k"].nodal_values))) if problem.get("q_flux_k") is not None else 0.0,
        abs(float(summary.get("centerline_at_y_interface_free_flux_n", 0.0))),
        abs(float(summary.get("centerline_at_y_interface_q_n", 0.0))),
    )
    p_scale = max(
        1.0e-12,
        float(np.max(np.abs(problem["p_k"].nodal_values))),
        float(np.max(np.abs(problem["p_pore_k"].nodal_values))) if problem.get("p_pore_k") is not None else 0.0,
        abs(float(summary.get("centerline_at_y_interface_traction_free_n", 0.0))),
        abs(float(summary.get("centerline_at_y_interface_traction_porous_support_n", 0.0))),
        abs(float(summary.get("centerline_at_y_interface_p", 0.0))),
        abs(float(summary.get("centerline_at_y_interface_p_pore_support", 0.0))),
    )
    metrics: dict[str, float] = {}
    for key in ("mass_flux_jump_n", "traction_jump_n", "traction_jump_t", "entry_residual"):
        values = np.asarray([float(row.get(key, float("nan"))) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        metrics[f"{key}_maxabs"] = float(np.max(np.abs(values))) if values.size else 0.0
    scaled = {
        "physics_mass": metrics["mass_flux_jump_n_maxabs"] / v_scale,
        "physics_traction_n": metrics["traction_jump_n_maxabs"] / p_scale,
        "physics_traction_t": metrics["traction_jump_t_maxabs"] / p_scale,
        "physics_entry": metrics["entry_residual_maxabs"] / p_scale,
    }
    include = [scaled["physics_mass"], scaled["physics_traction_n"]]
    if bool(getattr(args, "interface_entry_closure", False)):
        include.append(scaled["physics_entry"])
    if bool(getattr(args, "interface_bjs_closure", False)):
        include.append(scaled["physics_traction_t"])
    return float(max(include) if include else 0.0), {**metrics, **scaled}


def _outer_interface_residual_vector(
    *,
    args: argparse.Namespace,
    problem: dict[str, object],
    eps_alpha: float,
    interface_entry_delta: float,
) -> np.ndarray:
    interface_diag = mono._compute_interface_probe_diagnostics(
        problem=problem,
        Lx=float(args.Lx),
        y_interface=float(args.y_interface),
        y_profile=float(args.y_profile),
        eps_alpha=float(eps_alpha),
        mu_f=float(args.mu_f),
        mu_s=float(args.mu_s),
        lambda_s=float(args.lambda_s),
        solid_visco_eta=float(args.solid_visco_eta),
        solid_model=str(args.solid_model),
        skeleton_pressure_mode=str(args.skeleton_pressure_mode),
        alpha_biot=(None if args.alpha_biot is None else float(args.alpha_biot)),
        interface_entry_delta=float(interface_entry_delta),
        interface_bjs_gamma=float(getattr(args, "interface_bjs_gamma", 1.0e3)),
    )
    rows = _interface_transfer_rows(interface_diag)
    if not rows:
        return np.zeros((0,), dtype=float)
    summary = dict(interface_diag.get("summary", {}) or {})
    v_scale = max(
        1.0e-12,
        float(np.max(np.abs(problem["v_k"].nodal_values))),
        float(np.max(np.abs(problem["vS_k"].nodal_values))),
        float(np.max(np.abs(problem["q_flux_k"].nodal_values))) if problem.get("q_flux_k") is not None else 0.0,
        abs(float(summary.get("centerline_at_y_interface_free_flux_n", 0.0))),
        abs(float(summary.get("centerline_at_y_interface_q_n", 0.0))),
    )
    p_scale = max(
        1.0e-12,
        float(np.max(np.abs(problem["p_k"].nodal_values))),
        float(np.max(np.abs(problem["p_pore_k"].nodal_values))) if problem.get("p_pore_k") is not None else 0.0,
        abs(float(summary.get("centerline_at_y_interface_traction_free_n", 0.0))),
        abs(float(summary.get("centerline_at_y_interface_traction_porous_support_n", 0.0))),
        abs(float(summary.get("centerline_at_y_interface_p", 0.0))),
        abs(float(summary.get("centerline_at_y_interface_p_pore_support", 0.0))),
    )
    pieces: list[np.ndarray] = []
    for key, scale in (
        ("mass_flux_jump_n", v_scale),
        ("traction_jump_n", p_scale),
        ("entry_residual", p_scale),
        ("bjs_residual", p_scale),
    ):
        if key == "bjs_residual" and not bool(getattr(args, "interface_bjs_closure", False)):
            continue
        values = np.asarray([float(row.get(key, 0.0)) for row in rows], dtype=float)
        if values.size == 0:
            continue
        pieces.append(values / float(scale))
    if not pieces:
        return np.zeros((0,), dtype=float)
    return np.hstack(pieces)


def _candidate_outer_omegas(
    *,
    omega_seed: float,
    omega_min: float,
    omega_max: float,
) -> list[float]:
    lo = float(max(1.0e-6, omega_min))
    hi = float(np.clip(float(omega_max), lo, 1.0))
    seed = float(np.clip(float(omega_seed), lo, hi))
    candidates: list[float] = [hi, seed]
    trial = seed
    for _ in range(6):
        trial *= 0.5
        if trial < lo:
            break
        candidates.append(float(trial))
    candidates.append(lo)
    unique: list[float] = []
    seen: set[int] = set()
    for value in candidates:
        clipped = float(np.clip(float(value), lo, hi))
        key = int(round(clipped * 1.0e12))
        if key in seen:
            continue
        seen.add(key)
        unique.append(clipped)
    return unique


def _choose_outer_relaxation_by_physics(
    *,
    args: argparse.Namespace,
    problem: dict[str, object],
    current_functions,
    sweep_snapshot: tuple[np.ndarray, ...],
    raw_iterate_snapshot: tuple[np.ndarray, ...],
    field_names: set[str] | list[str] | tuple[str, ...],
    omega_seed: float,
    eps_alpha: float,
    interface_entry_delta: float,
) -> tuple[float, float, dict[str, float]]:
    best_omega = float(np.clip(float(omega_seed), 0.0, 1.0))
    best_physics = float("inf")
    best_metrics: dict[str, float] = {}
    for omega in _candidate_outer_omegas(
        omega_seed=float(omega_seed),
        omega_min=float(getattr(args, "outer_relaxation_min", 1.0e-3)),
        omega_max=float(getattr(args, "outer_relaxation_max", 1.0)),
    ):
        _restore_function_values(current_functions, raw_iterate_snapshot)
        _blend_toward_snapshot(
            current_functions,
            sweep_snapshot,
            omega=float(omega),
            field_names=field_names,
        )
        physics_inf, physics_metrics = _outer_interface_physics(
            args=args,
            problem=problem,
            eps_alpha=float(eps_alpha),
            interface_entry_delta=float(interface_entry_delta),
        )
        if not np.isfinite(physics_inf):
            continue
        if physics_inf < best_physics:
            best_omega = float(omega)
            best_physics = float(physics_inf)
            best_metrics = dict(physics_metrics)
    _restore_function_values(current_functions, raw_iterate_snapshot)
    return float(best_omega), float(best_physics), dict(best_metrics)


def _stage_order(args: argparse.Namespace) -> tuple[str, str, str]:
    raw = str(getattr(args, "stage_order", "flow,solid,transport") or "flow,solid,transport")
    raw = raw.replace("->", ",")
    names = tuple(str(token).strip().lower() for token in raw.split(",") if str(token).strip())
    if len(names) != 3 or set(names) != {"flow", "solid", "transport"}:
        raise ValueError(
            "--stage-order must contain each of flow, solid, transport exactly once "
            f"(received {raw!r})."
        )
    return names


def _parse_scale_schedule(raw: str | None, *, default: tuple[float, ...]) -> tuple[float, ...]:
    text = str(raw or "").strip()
    if not text:
        return tuple(float(v) for v in tuple(default))
    values: list[float] = []
    for token in text.replace(";", ",").split(","):
        piece = str(token).strip()
        if not piece:
            continue
        value = float(piece)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid continuation scale {piece!r}; expected a positive finite number.")
        values.append(float(np.clip(value, 1.0e-8, 1.0)))
    if not values:
        return (1.0,)
    ordered: list[float] = []
    for value in values:
        if ordered and abs(float(value) - float(ordered[-1])) <= 1.0e-12:
            continue
        ordered.append(float(value))
    if ordered[-1] < 1.0 - 1.0e-14:
        ordered.append(1.0)
    return tuple(float(v) for v in ordered)


def _startup_flow_scale_schedule(args: argparse.Namespace, *, step_no: int) -> tuple[float, ...]:
    if int(step_no) != 0:
        return (1.0,)
    return _parse_scale_schedule(
        getattr(args, "startup_flow_scales", None),
        default=(0.025, 0.1, 0.25, 0.5, 1.0),
    )


def _scaled_bc_schedule(
    bcs_now: list[BoundaryCondition],
    *,
    scales: tuple[float, ...],
) -> tuple[tuple[float, list[BoundaryCondition]], ...]:
    out: list[tuple[float, list[BoundaryCondition]]] = []
    for scale in tuple(scales or (1.0,)):
        if float(scale) < 1.0 - 1.0e-14:
            out.append((float(scale), _scaled_dirichlet_bcs(bcs_now, scale=float(scale))))
        else:
            out.append((1.0, bcs_now))
    return tuple(out)


def _scaled_snapshot_delta(
    functions,
    snapshot: tuple[np.ndarray, ...],
    *,
    field_scales: dict[str, float] | None = None,
    field_names: set[str] | None = None,
) -> tuple[float, float, dict[str, float]]:
    scales = dict(field_scales or {})
    selected = None if field_names is None else {str(name) for name in set(field_names)}
    max_scaled = 0.0
    max_abs = 0.0
    by_field: dict[str, float] = {}
    for func, values in zip(list(functions or []), list(snapshot or [])):
        func_name = str(getattr(func, "name", ""))
        if selected is not None and func_name not in selected:
            continue
        cur = np.asarray(func.nodal_values, dtype=float).ravel()
        old = np.asarray(values, dtype=float).ravel()
        abs_delta = float(np.max(np.abs(cur - old))) if cur.size else 0.0
        ref = float(scales.get(func_name, 0.0) or 0.0)
        if not np.isfinite(ref) or ref <= 0.0:
            ref = max(
                1.0,
                float(np.max(np.abs(old))) if old.size else 0.0,
                float(np.max(np.abs(cur))) if cur.size else 0.0,
            )
        scaled = float(abs_delta / max(ref, 1.0e-30))
        by_field[func_name] = scaled
        max_scaled = max(max_scaled, scaled)
        max_abs = max(max_abs, abs_delta)
    return max_scaled, max_abs, by_field


def _functions_to_vector(
    functions,
    *,
    field_names: set[str] | list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    selected = None if field_names is None else {str(name) for name in set(field_names)}
    parts: list[np.ndarray] = []
    for func in list(functions or []):
        func_name = str(getattr(func, "name", ""))
        if selected is not None and func_name not in selected:
            continue
        parts.append(np.asarray(func.nodal_values, dtype=float).ravel())
    if not parts:
        return np.zeros((0,), dtype=float)
    return np.hstack(parts)


def _snapshot_to_vector(
    functions,
    snapshot: tuple[np.ndarray, ...],
    *,
    field_names: set[str] | list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    selected = None if field_names is None else {str(name) for name in set(field_names)}
    parts: list[np.ndarray] = []
    for func, values in zip(list(functions or []), list(snapshot or [])):
        func_name = str(getattr(func, "name", ""))
        if selected is not None and func_name not in selected:
            continue
        parts.append(np.asarray(values, dtype=float).ravel())
    if not parts:
        return np.zeros((0,), dtype=float)
    return np.hstack(parts)


def _outer_coupling_function_names(problem: dict[str, object]) -> set[str]:
    names = {
        str(problem["vS_k"].name),
        str(problem["u_k"].name),
        str(problem["alpha_k"].name),
        str(problem["mu_k"].name),
    }
    if problem.get("p_pore_k") is not None:
        names.add(str(problem["p_pore_k"].name))
    if problem.get("B_k") is not None:
        names.add(str(problem["B_k"].name))
    if problem.get("phi_k") is not None:
        names.add(str(problem["phi_k"].name))
    if problem.get("S_k") is not None:
        names.add(str(problem["S_k"].name))
    return names


def _poro_solid_coupling_function_names(problem: dict[str, object]) -> set[str]:
    names = {
        str(problem["vS_k"].name),
        str(problem["u_k"].name),
    }
    if problem.get("p_pore_k") is not None:
        names.add(str(problem["p_pore_k"].name))
    if problem.get("pi_s_k") is not None:
        names.add(str(problem["pi_s_k"].name))
    if problem.get("lambda_drag_k") is not None:
        names.add(str(problem["lambda_drag_k"].name))
    return names


def _blend_toward_snapshot(
    functions,
    snapshot: tuple[np.ndarray, ...],
    *,
    omega: float,
    field_names: set[str] | list[str] | tuple[str, ...] | None = None,
) -> None:
    theta = min(max(float(omega), 0.0), 1.0)
    if theta >= 1.0 - 1.0e-14:
        return
    selected = None if field_names is None else {str(name) for name in set(field_names)}
    for func, values in zip(list(functions or []), list(snapshot or [])):
        func_name = str(getattr(func, "name", ""))
        if selected is not None and func_name not in selected:
            continue
        old = np.asarray(values, dtype=float).ravel()
        cur = np.asarray(func.nodal_values, dtype=float).ravel()
        func.nodal_values[:] = old + theta * (cur - old)


def _relax_preload_toward_previous(
    current_functions,
    previous_functions,
    *,
    keep_fields: list[str] | tuple[str, ...],
    blend: float,
) -> bool:
    theta = float(np.clip(float(blend), 0.0, 1.0))
    if theta >= 1.0 - 1.0e-14:
        return False
    keep = {str(name) for name in list(keep_fields or [])}
    changed = False
    for cur, prev in zip(list(current_functions or []), list(previous_functions or [])):
        field_name = str(getattr(cur, "field_name", "") or getattr(cur, "name", "") or "")
        if field_name in keep:
            continue
        cur_vals = np.asarray(cur.nodal_values, dtype=float).ravel()
        prev_vals = np.asarray(prev.nodal_values, dtype=float).ravel()
        if cur_vals.size != prev_vals.size:
            continue
        changed = changed or bool(np.any(np.abs(cur_vals - prev_vals) > 1.0e-14))
        cur.nodal_values[:] = prev_vals + theta * (cur_vals - prev_vals)
    return bool(changed)


def _flow_stage_fields(problem: dict[str, object]) -> list[str]:
    fields = ["v_x", "v_y", "p"]
    if problem.get("p_mean_k") is not None:
        fields.append("p_mean")
    return fields


def _transport_stage_fields(problem: dict[str, object]) -> list[str]:
    fields = ["alpha", "mu_alpha"]
    if problem.get("B_k") is not None and not bool(problem.get("_freeze_B_total_mixture_continuity_diagnostic", False)):
        fields.append("B")
    if problem.get("alpha_mass_lm_k") is not None:
        fields.append("alpha_mass_lm")
    if problem.get("S_k") is not None:
        fields.append("S")
    if problem.get("phi_k") is not None:
        fields.append("phi")
    return fields


def _solid_stage_fields(problem: dict[str, object]) -> list[str]:
    fields = ["vS_x", "vS_y", "u_x", "u_y"]
    if problem.get("p_pore_k") is not None:
        fields.append("p_pore")
    if problem.get("pi_s_k") is not None:
        fields.append("pi_s")
    if problem.get("lambda_drag_k") is not None:
        fields.extend(["lambda_drag_x", "lambda_drag_y"])
    return fields


def _build_staggered_interface_forms(
    problem: dict[str, object],
    *,
    qdeg: int,
    eps_alpha: float,
    mu_f: float,
    mu_s: float,
    lambda_s: float,
    solid_visco_eta: float,
    solid_model: str,
    skeleton_pressure_mode: str,
    alpha_biot: float | None,
    interface_entry_closure: bool = True,
    interface_entry_closure_strength: float = 1.0,
    interface_entry_delta: float = 10.0,
    interface_velocity_continuity_closure: bool = True,
    interface_traction_continuity_closure: bool = False,
    interface_velocity_normal_strength: float = 1.0,
    interface_traction_normal_strength: float = 1.0,
    interface_robin_l: float | None = None,
) -> None:
    if interface_robin_l is not None:
        interface_velocity_normal_strength = float(interface_robin_l)
    problem["_staggered_flow_transfer_residual_form"] = None
    problem["_staggered_flow_transfer_jacobian_form"] = None
    problem["_staggered_solid_transfer_residual_form"] = None
    problem["_staggered_solid_transfer_jacobian_form"] = None
    problem["_staggered_state_use_porous_flux_n"] = bool(interface_velocity_continuity_closure)
    # Outer relaxation acts only on porous-side data. The free-fluid traces are
    # direct responses of the current flow solve and are committed
    # Gauss-Seidel-style before the porous stage runs.
    problem["_staggered_state_use_free_traction_n"] = False
    problem["_staggered_state_use_free_flux_n"] = False
    problem["_staggered_state_use_porous_traction_n"] = bool(interface_traction_continuity_closure)
    if problem.get("p_pore_k") is None:
        return
    has_q_primary_flux = (
        problem.get("q_flux_k") is not None
        and problem.get("dq_flux") is not None
        and problem.get("q_flux_test") is not None
    )
    geometry_indicator_beta = float(problem.get("geometry_indicator_beta", 0.0) or 0.0)
    interface_eta = max(1.0e-12, 1.0e-3 * float(eps_alpha))
    interface_weight_scale = 1.0 / max(float(eps_alpha), 1.0e-12)
    interface_corner_taper = problem.get("interface_corner_taper")
    if interface_corner_taper is None:
        interface_corner_taper = Constant(1.0)
    alpha_if = mono._geometry_indicator_expr(problem["alpha_k"], beta=geometry_indicator_beta)
    interface_weight = (
        mono._named_constant("b7_staggered_interface_weight_scale", interface_weight_scale)
        * interface_corner_taper
        * Constant(4.0)
        * alpha_if
        * (Constant(1.0) - alpha_if)
    )
    n_if = mono._interface_unit_normal_2d(alpha_if, eta=interface_eta)
    v_n_k = mono._dot_basis_2d(problem["v_k"], n_if)
    dv_n = mono._dot_basis_2d(problem["dv"], n_if)
    v_n_test = mono._dot_basis_2d(problem["v_test"], n_if)
    vS_n_k = mono._dot_basis_2d(problem["vS_k"], n_if)
    dvS_n = mono._dot_basis_2d(problem["dvS"], n_if)
    if has_q_primary_flux:
        q_n_k = mono._dot_basis_2d(problem["q_flux_k"], n_if)
        d_q_n = mono._dot_basis_2d(problem["dq_flux"], n_if)
        q_n_test = mono._dot_basis_2d(problem["q_flux_test"], n_if)
        q_pore_test = problem.get("q_pore_test", q_n_test)
    else:
        q_n_k = Constant(0.0)
        d_q_n = Constant(0.0)
        q_pore_test = problem.get("q_pore_test")
        if q_pore_test is None:
            raise ValueError(
                "The staggered interface-transfer split requires either q-primary Darcy flux fields "
                "or the split-pressure q_pore test field."
            )
        q_n_test = q_pore_test
    solid_n_test = mono._dot_basis_2d(problem["vS_test"], n_if)
    mu_f_c = mono._named_constant("b7_staggered_mu_f", float(mu_f))
    visc_n_k = mono._normal_viscous_traction_scalar_2d(problem["v_k"], mu_f_c, n_if)
    mu_s_c = mono._named_constant("b7_staggered_mu_s", float(mu_s))
    lambda_s_c = mono._named_constant("b7_staggered_lambda_s", float(lambda_s))
    solid_model_key = mono._benchmark7_solid_model_key(solid_model)
    if solid_model_key in {"linear", "small_strain", "linear_elastic"}:
        eps_u_k = Constant(0.5) * (mono.grad(problem["u_k"]) + mono.grad(problem["u_k"]).T)
        eps_du = Constant(0.5) * (mono.grad(problem["du"]) + mono.grad(problem["du"]).T)
        solid_sig_k = Constant(2.0) * mu_s_c * eps_u_k + lambda_s_c * mono.div(problem["u_k"]) * mono.Identity(2)
        dsolid_sig = Constant(2.0) * mu_s_c * eps_du + lambda_s_c * mono.div(problem["du"]) * mono.Identity(2)
    elif solid_model_key in {"neo_hookean_seboldt", "seboldt_neo_hookean"}:
        solid_sig_k = mono.sigma_neo_hookean_seboldt(problem["u_k"], mu_s_c, lambda_s_c, dim=2)
        dsolid_sig = mono.dsigma_neo_hookean_seboldt(problem["u_k"], problem["du"], mu_s_c, lambda_s_c, dim=2)
    elif solid_model_key in {"stvk", "svk", "saint_venant_kirchhoff", "saint-venant-kirchhoff"}:
        solid_sig_k = mono.sigma_svk(problem["u_k"], mu_s_c, lambda_s_c, dim=2)
        dsolid_sig = mono.dsigma_svk(problem["u_k"], problem["du"], mu_s_c, lambda_s_c, dim=2)
    else:
        raise NotImplementedError(
            f"Staggered Seboldt solid Robin form does not yet support solid_model={solid_model!r}."
        )
    eta_s_c = mono._named_constant("b7_staggered_solid_visco_eta", float(solid_visco_eta))
    solid_visc_n_k = mono._normal_viscous_traction_scalar_2d(problem["vS_k"], eta_s_c, n_if)
    dsolid_visc_n = mono._normal_viscous_traction_scalar_2d(problem["dvS"], eta_s_c, n_if)
    solid_trac_n_k = mono._normal_tensor_traction_scalar_2d(solid_sig_k, n_if)
    dsolid_trac_n = mono._normal_tensor_traction_scalar_2d(dsolid_sig, n_if)
    pore_interface_coeff_c = mono._named_constant(
        "b7_staggered_pore_interface_coeff",
        float(1.0 if alpha_biot is None else alpha_biot),
    )
    entry_delta_c = mono._named_constant("b7_staggered_interface_entry_delta", float(interface_entry_delta))
    pore_interface_pressure_k = pore_interface_coeff_c * problem["p_pore_k"]
    d_pore_interface_pressure = pore_interface_coeff_c * problem["dp_pore"]
    pressure_q_drive_k = pore_interface_pressure_k + entry_delta_c * q_n_k
    d_pressure_q_drive = d_pore_interface_pressure + entry_delta_c * d_q_n
    entry_robin_drive_k = entry_delta_c * q_n_k
    d_entry_robin_drive = entry_delta_c * d_q_n
    porous_support_trac_n_current = solid_trac_n_k + solid_visc_n_k
    free_trac_n_current = -problem["p_k"] + visc_n_k
    dfree_trac_n_current = -problem["dp"] + mono._normal_viscous_traction_scalar_2d(problem["dv"], mu_f_c, n_if)
    free_trac_n_transfer = (
        problem["_staggered_if_free_traction_n"]
        if problem.get("_staggered_if_free_traction_n") is not None
        else free_trac_n_current
    )
    porous_support_trac_n_transfer = (
        problem["_staggered_if_porous_traction_n"]
        if problem.get("_staggered_if_porous_traction_n") is not None
        else porous_support_trac_n_current
    )

    flow_residual = None
    flow_jacobian = None
    solid_residual = None
    solid_jacobian = None
    dx_q = mono.dx(metadata={"q": int(qdeg)})

    if (
        bool(interface_velocity_continuity_closure)
        and problem.get("_staggered_if_porous_flux_n") is not None
        and float(interface_velocity_normal_strength) != 0.0
    ):
        vel_n_strength_c = mono._named_constant(
            "b7_staggered_interface_velocity_normal_strength_flow",
            float(interface_velocity_normal_strength),
        )
        # Fluid stage: lagged Dirichlet-type transfer of the porous normal
        # interface velocity (vS + q).
        flow_vel_residual = (
            vel_n_strength_c
            * interface_weight
            * (v_n_k - problem["_staggered_if_porous_flux_n"])
            * v_n_test
            * dx_q
        )
        flow_vel_jacobian = vel_n_strength_c * interface_weight * dv_n * v_n_test * dx_q
        flow_residual = flow_vel_residual if flow_residual is None else flow_residual + flow_vel_residual
        flow_jacobian = flow_vel_jacobian if flow_jacobian is None else flow_jacobian + flow_vel_jacobian

    if bool(interface_velocity_continuity_closure) and problem.get("_staggered_if_free_flux_n") is not None:
        vel_n_strength_c = mono._named_constant(
            "b7_staggered_interface_velocity_normal_strength_solid",
            float(interface_velocity_normal_strength),
        )
        solid_vel_residual = vel_n_strength_c * interface_weight * (
            problem["_staggered_if_free_flux_n"] - vS_n_k - q_n_k
        ) * q_pore_test * dx_q
        solid_vel_jacobian = vel_n_strength_c * interface_weight * (-(dvS_n + d_q_n)) * q_pore_test * dx_q
        solid_residual = solid_vel_residual if solid_residual is None else solid_residual + solid_vel_residual
        solid_jacobian = solid_vel_jacobian if solid_jacobian is None else solid_jacobian + solid_vel_jacobian

    if (
        bool(interface_entry_closure)
        and float(interface_entry_closure_strength) != 0.0
    ):
        entry_strength_c = mono._named_constant(
            "b7_staggered_interface_entry_closure_strength_solid",
            float(interface_entry_closure_strength),
        )
        # Keep the porous-side entry law only on the Darcy/flux row. In the
        # staggered split the solid stage already carries:
        #   1. Biot pore-pressure loading in the bulk skeleton operator, and
        #   2. explicit free-fluid Neumann traction transfer below.
        # Re-inserting -p_P * (vS_test.n) here double-counts the normal pore
        # pressure on the porous momentum stage and was the main source of the
        # outer-sweep solid blow-up.
        if problem.get("_staggered_if_free_traction_n") is not None:
            # On the staggered split the Darcy trace must carry the full scalar
            # entry law against the lagged free-fluid traction:
            #   t_F,n + p_P + delta q_n = 0.
            # The free traction is explicit interface data, while p_P and q_n
            # remain current porous unknowns, so only the porous part enters
            # the stage Jacobian.
            q_row_drive_k = free_trac_n_transfer + pressure_q_drive_k
            d_q_row_drive = d_pressure_q_drive
        else:
            q_row_drive_k = pressure_q_drive_k
            d_q_row_drive = d_pressure_q_drive
        solid_entry_q_residual = entry_strength_c * interface_weight * q_row_drive_k * q_n_test * dx_q
        solid_entry_q_jacobian = entry_strength_c * interface_weight * d_q_row_drive * q_n_test * dx_q
        solid_residual = solid_entry_q_residual if solid_residual is None else solid_residual + solid_entry_q_residual
        solid_jacobian = solid_entry_q_jacobian if solid_jacobian is None else solid_jacobian + solid_entry_q_jacobian

    if (
        bool(interface_traction_continuity_closure)
        and float(interface_traction_normal_strength) != 0.0
    ):
        tr_n_strength_c = mono._named_constant(
            "b7_staggered_interface_traction_normal_strength_flow",
            float(interface_traction_normal_strength),
        )
        flow_trac_residual = (
            tr_n_strength_c
            * interface_weight
            * (porous_support_trac_n_transfer - free_trac_n_current - pore_interface_pressure_k)
            * v_n_test
            * dx_q
        )
        flow_trac_jacobian = (
            tr_n_strength_c
            * interface_weight
            * (-(dfree_trac_n_current + d_pore_interface_pressure))
            * v_n_test
            * dx_q
        )
        flow_residual = flow_trac_residual if flow_residual is None else flow_residual + flow_trac_residual
        flow_jacobian = flow_trac_jacobian if flow_jacobian is None else flow_jacobian + flow_trac_jacobian

    if (
        float(interface_traction_normal_strength) != 0.0
        and not bool(interface_traction_continuity_closure)
    ):
        tr_n_strength_c = mono._named_constant(
            "b7_staggered_interface_neumann_normal_strength",
            float(interface_traction_normal_strength),
        )
        # Match the monolithic traction-jump split: the fluid contribution to
        # the solid test enters with the same sign as the free-traction part of
        # (traction_porous - traction_free) * (v_test.n - vS_test.n).
        solid_neumann_residual = (
            tr_n_strength_c
            * interface_weight
            * free_trac_n_transfer
            * solid_n_test
            * dx_q
        )
        solid_residual = solid_neumann_residual if solid_residual is None else solid_residual + solid_neumann_residual

    if (
        bool(interface_traction_continuity_closure)
        and float(interface_traction_normal_strength) != 0.0
    ):
        tr_n_strength_c = mono._named_constant(
            "b7_staggered_interface_traction_normal_strength",
            float(interface_traction_normal_strength),
        )
        solid_trac_residual = tr_n_strength_c * interface_weight * (
            (free_trac_n_transfer + pore_interface_pressure_k - porous_support_trac_n_current) * solid_n_test
        ) * dx_q
        solid_trac_jacobian = tr_n_strength_c * interface_weight * (
            (d_pore_interface_pressure - (dsolid_trac_n + dsolid_visc_n)) * solid_n_test
        ) * dx_q
        solid_residual = solid_trac_residual if solid_residual is None else solid_residual + solid_trac_residual
        solid_jacobian = solid_trac_jacobian if solid_jacobian is None else solid_jacobian + solid_trac_jacobian

    problem["_staggered_flow_transfer_residual_form"] = flow_residual
    problem["_staggered_flow_transfer_jacobian_form"] = flow_jacobian
    problem["_staggered_solid_transfer_residual_form"] = solid_residual
    problem["_staggered_solid_transfer_jacobian_form"] = solid_jacobian


def _build_stage_forms(problem: dict[str, object], forms) -> SimpleNamespace:
    flow_fields = _flow_stage_fields(problem)
    transport_fields = _transport_stage_fields(problem)
    solid_fields = _solid_stage_fields(problem)
    freeze_B_total_mass = bool(problem.get("_freeze_B_total_mixture_continuity_diagnostic", False))

    full_residual = forms.residual_form
    full_jacobian = forms.jacobian_form
    flow_residual = _sum_forms(
        forms.r_momentum,
        forms.r_mass,
        problem.get("_pressure_mean_residual_form"),
        problem.get("_staggered_flow_transfer_residual_form"),
    )
    flow_jacobian = _sum_forms(
        forms.a_momentum,
        forms.a_mass,
        problem.get("_pressure_mean_jacobian_form"),
        problem.get("_staggered_flow_transfer_jacobian_form"),
    )
    transport_residual = _sum_forms(
        getattr(forms, "r_alpha", None),
        (None if freeze_B_total_mass else getattr(forms, "r_B", None)),
        getattr(forms, "r_mu_alpha", None),
        problem.get("_alpha_mass_constraint_residual_form"),
        getattr(forms, "r_phi", None),
        getattr(forms, "r_substrate", None),
    )
    transport_jacobian = _sum_forms(
        getattr(forms, "a_alpha", None),
        (None if freeze_B_total_mass else getattr(forms, "a_B", None)),
        getattr(forms, "a_mu_alpha", None),
        problem.get("_alpha_mass_constraint_jacobian_form"),
        getattr(forms, "a_phi", None),
        getattr(forms, "a_substrate", None),
    )
    solid_mass_residual = getattr(forms, "r_total_mass", None) if freeze_B_total_mass else getattr(forms, "r_pore", None)
    solid_mass_jacobian = getattr(forms, "a_total_mass", None) if freeze_B_total_mass else getattr(forms, "a_pore", None)
    solid_residual = _sum_forms(
        forms.r_skeleton,
        forms.r_kinematics,
        solid_mass_residual,
        problem.get("_p_pore_fluid_gauge_residual_form"),
        problem.get("_staggered_solid_transfer_residual_form"),
    )
    solid_jacobian = _sum_forms(
        forms.a_skeleton,
        forms.a_kinematics,
        solid_mass_jacobian,
        problem.get("_p_pore_fluid_gauge_jacobian_form"),
        problem.get("_staggered_solid_transfer_jacobian_form"),
    )
    return SimpleNamespace(
        flow=SimpleNamespace(
            name="flow",
            active_fields=flow_fields,
            residual_form=flow_residual,
            jacobian_form=flow_jacobian,
        ),
        transport=SimpleNamespace(
            name="transport",
            active_fields=transport_fields,
            residual_form=transport_residual,
            jacobian_form=transport_jacobian,
        ),
        solid=SimpleNamespace(
            name="solid",
            active_fields=solid_fields,
            residual_form=solid_residual,
            jacobian_form=solid_jacobian,
        ),
    )


def _configure_solver_scaling(target_solver, problem: dict[str, object]) -> None:
    row_scales = dict(problem.get("_condition_balanced_row_field_scales", {}) or {})
    col_scales = dict(problem.get("_condition_balanced_col_field_scales", {}) or {})
    if row_scales or col_scales:
        target_solver.set_manual_reduced_system_scaling(
            equation_row_field_scales=row_scales,
            variable_column_field_scales=col_scales,
        )
        target_solver.set_reduced_system_scaling(
            equation_row_scaling=True,
            variable_column_scaling=True,
            variable_column_scaling_fields=tuple(sorted(col_scales)),
            mode="field",
            ruiz_iters=6,
        )
        if getattr(target_solver, "vi_params", None) is not None:
            target_solver.vi_params.equation_row_scaling = True
            target_solver.vi_params.variable_column_scaling = True


def _transport_box_bounds_pre_cb(target_solver, *, problem: dict[str, object]):
    def _wrapped(funcs) -> None:
        total_dofs = int(problem["dh"].total_dofs)
        lower = np.full(total_dofs, -np.inf, dtype=float)
        upper = np.full(total_dofs, np.inf, dtype=float)
        mono._apply_field_box_bounds(
            lower,
            upper,
            dof_handler=problem["dh"],
            field_name="alpha",
            lo=0.0,
            hi=1.0,
        )
        if problem.get("B_k") is not None:
            alpha_ref = _find_named_function(funcs, problem["alpha_k"])
            mono._apply_field_box_bounds(
                lower,
                upper,
                dof_handler=problem["dh"],
                field_name="B",
                lo=0.0,
                hi=None,
            )
            mono._apply_field_dependent_upper_bound(
                upper,
                problem=problem,
                source_field_name="alpha",
                target_field_name="B",
                source_values=np.clip(
                    np.asarray(alpha_ref.nodal_values, dtype=float),
                    0.0,
                    1.0,
                ),
            )
        target_solver.set_box_bounds(lower=lower, upper=upper)

    return _wrapped


def _make_stage_solver(
    *,
    args: argparse.Namespace,
    problem: dict[str, object],
    stage,
    bcs: list[BoundaryCondition],
    bcs_homog: list[BoundaryCondition],
    qdeg: int,
    aux_functions: dict[str, object],
):
    solver_kind = str(getattr(args, f"{stage.name}_solver")).strip().lower()
    max_it = getattr(args, f"{stage.name}_max_it")
    if max_it is None:
        max_it = int(args.max_it)
    stage_newton_tol_raw = getattr(args, "stage_newton_tol", None)
    if stage_newton_tol_raw is None:
        stage_newton_tol = max(float(getattr(args, "newton_tol", 1.0e-8) or 1.0e-8), 1.0e-16)
    else:
        stage_newton_tol = max(float(stage_newton_tol_raw), 1.0e-16)
    stage_newton_rtol = max(float(getattr(args, "stage_newton_rtol", 0.0) or 0.0), 0.0)
    stage_accept_factor = max(
        float(getattr(args, "stage_accept_nonconverged_atol_factor", 0.0) or 0.0),
        0.0,
    )
    relaxed_stage_ginf = max(1.0e3 * stage_newton_tol, 1.0e-6)
    if stage.name == "transport":
        relaxed_stage_ginf = max(relaxed_stage_ginf, 5.0e-2)
    newton_params = NewtonParameters(
        newton_tol=float(stage_newton_tol),
        newton_rtol=float(stage_newton_rtol),
        max_newton_iter=int(max_it),
        ls_mode=str(args.ls_mode),
        globalization=str(getattr(args, "newton_globalization", "line_search")),
        accept_nonconverged_atol_factor=float(stage_accept_factor),
        tr_max_iter=int(getattr(args, "trust_max_it", 8)),
        tr_radius_init=float(getattr(args, "trust_radius_init", 1.0)),
        tr_radius_max=float(getattr(args, "trust_radius_max", 1.0e3)),
        tr_eta_accept=float(getattr(args, "trust_eta_accept", 1.0e-4)),
        tr_eta_contract=float(getattr(args, "trust_eta_contract", 2.5e-1)),
        tr_eta_expand=float(getattr(args, "trust_eta_expand", 7.5e-1)),
        tr_shrink=float(getattr(args, "trust_shrink", 2.5e-1)),
        tr_expand=float(getattr(args, "trust_expand", 2.0)),
        tr_min_radius=float(getattr(args, "trust_min_radius", 1.0e-10)),
        tr_min_abs_decrease_inf=float(getattr(args, "trust_min_abs_residual_drop", 0.0)),
        tr_min_rel_decrease_inf=float(getattr(args, "trust_min_rel_residual_drop", 0.0)),
        stall_window=int(getattr(args, "newton_stall_window", 0)),
        stall_min_abs_decrease_inf=float(getattr(args, "newton_stall_min_abs_residual_drop", 0.0)),
        stall_min_rel_decrease_inf=float(getattr(args, "newton_stall_min_rel_residual_drop", 0.0)),
        ptc_recovery=bool(getattr(args, "vi_ptc_recovery", False)),
        ptc_fields=mono._parse_csv_fields(getattr(args, "vi_ptc_fields", "")),
        ptc_sigma0=float(getattr(args, "vi_ptc_sigma0", 1.0e-2)),
        ptc_sigma_max=float(getattr(args, "vi_ptc_sigma_max", 1.0e6)),
        ptc_growth=float(getattr(args, "vi_ptc_growth", 10.0)),
        ptc_decay=float(getattr(args, "vi_ptc_decay", 0.2)),
        ptc_freeze_complement=False,
        ptc_operator_mode=str(getattr(args, "newton_ptc_operator_mode", "row_normalized")),
        ptc_late_fields=mono._parse_csv_fields(getattr(args, "newton_ptc_late_fields", "")),
        ptc_late_switch_residual=float(getattr(args, "newton_ptc_late_switch_residual", 0.0)),
        ptc_late_operator_mode=str(getattr(args, "newton_ptc_late_operator_mode", "")),
        print_level=2,
    )
    newton_params.line_search = bool(args.line_search)
    common_kwargs = dict(
        dof_handler=problem["dh"],
        mixed_element=problem["me"],
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=newton_params,
        lin_params=LinearSolverParameters(
            backend=str(args.linear_backend),
            tol=float(args.lin_tol),
            maxit=int(args.lin_maxit),
            distributed=bool(
                mono.MPI_CTX.enabled
                and bool(getattr(args, "petsc_distributed", True))
                and str(args.linear_backend).strip().lower() == "petsc"
            ),
        ),
        quad_order=int(qdeg),
        backend=str(args.backend),
    )
    if solver_kind == "newton":
        target_solver = NewtonSolver(stage.residual_form, stage.jacobian_form, **common_kwargs)
    else:
        solver_cls = PdasNewtonSolver if solver_kind == "pdas" else InteriorPointNewtonSolver
        target_solver = solver_cls(
            stage.residual_form,
            stage.jacobian_form,
            vi_params=VIParameters(
                c=float(args.vi_c),
                enter_tol=float(args.vi_enter_tol),
                leave_tol=float(args.vi_leave_tol),
                active_set_persistence=int(args.vi_persistence),
                project_initial_guess=True,
                project_each_iteration=False,
                inactive_reg_lambda0=float(args.vi_lambda0),
                inactive_reg_lambda_max=float(args.vi_lambda_max),
                inactive_reg_growth=float(args.vi_lambda_growth),
                inactive_reg_decay=float(args.vi_lambda_decay),
                active_step_delta_active_trigger=int(getattr(args, "vi_active_soft_threshold", 0)),
                active_step_soft_alpha=float(getattr(args, "vi_active_soft_alpha", 0.0)),
                active_step_strong_factor=float(getattr(args, "vi_active_strong_factor", 1.0)),
                filter_max_delta_active=int(getattr(args, "vi_filter_max_delta_active", 0)),
                filter_max_residual_growth=float(getattr(args, "vi_filter_max_residual_growth", 0.0)),
                filter_max_gap_growth=float(getattr(args, "vi_filter_max_gap_growth", 0.0)),
                relaxed_filter_accept_ginf=float(relaxed_stage_ginf),
                relaxed_filter_merit_growth=1.05,
                bound_step_limit=True,
                bound_step_tau=1.0,
                bound_blocking_activate=True,
                bound_blocking_trigger_alpha=0.95,
                bound_blocking_max_iter=16,
                equation_row_scaling=True,
                variable_column_scaling=True,
                variable_column_scaling_fields=(
                    mono._parse_csv_fields(getattr(args, "vi_variable_column_scaling_fields", ""))
                    or tuple(sorted(problem.get("_condition_balanced_col_field_scales", {}) or {}))
                ),
                unconstrained_lm=bool(args.vi_unconstrained_lm),
                unconstrained_lm_lambda0=float(args.vi_lm_lambda0),
                unconstrained_lm_lambda_max=float(args.vi_lm_lambda_max),
                unconstrained_lm_growth=float(args.vi_lm_growth),
                unconstrained_lm_decay=float(args.vi_lm_decay),
                unconstrained_lm_accept_ratio=float(args.vi_lm_accept_ratio),
                unconstrained_lm_good_ratio=float(args.vi_lm_good_ratio),
                unconstrained_lm_max_tries=int(args.vi_lm_max_tries),
                line_search_nonmonotone_window=0,
                line_search_nonmonotone_active_stable_iters=0,
                line_search_nonmonotone_ginf_trigger=0.0,
                line_search_nonmonotone_gap_ratio=1.0,
                line_search_nonmonotone_eq_abs=1.0e-10,
                line_search_nonmonotone_disable_filter=False,
                affine_identified_acceleration=False,
                ptc_recovery=bool(getattr(args, "vi_ptc_recovery", False)),
                ptc_fields=mono._parse_csv_fields(getattr(args, "vi_ptc_fields", "")),
                ptc_sigma0=float(getattr(args, "vi_ptc_sigma0", 1.0e-2)),
                ptc_sigma_max=float(getattr(args, "vi_ptc_sigma_max", 1.0e6)),
                ptc_growth=float(getattr(args, "vi_ptc_growth", 10.0)),
                ptc_decay=float(getattr(args, "vi_ptc_decay", 0.2)),
                ptc_ginf_trigger=float(getattr(args, "vi_ptc_ginf_trigger", 0.0)),
                ptc_ginf_max=float(getattr(args, "vi_ptc_ginf_max", 0.0)),
                anderson_acceleration=bool(getattr(args, "vi_anderson_acceleration", False)),
                anderson_history=int(getattr(args, "vi_anderson_history", 4)),
                anderson_regularization=float(getattr(args, "vi_anderson_regularization", 1.0e-8)),
                anderson_damping=float(getattr(args, "vi_anderson_damping", 1.0)),
                interior_point_mu0=float(getattr(args, "vi_ipm_mu0", 1.0e-2)),
                interior_point_mu_min=float(getattr(args, "vi_ipm_mu_min", 1.0e-10)),
                interior_point_mu_decay=float(getattr(args, "vi_ipm_mu_decay", 0.2)),
                interior_point_max_barrier_steps=int(getattr(args, "vi_ipm_max_barrier_steps", 12)),
                interior_point_fraction_to_boundary=float(getattr(args, "vi_ipm_fraction_to_boundary", 0.995)),
                interior_point_armijo_c1=float(getattr(args, "vi_ipm_armijo_c1", 1.0e-4)),
                interior_point_step_reduction=float(getattr(args, "vi_ipm_step_reduction", 0.5)),
                interior_point_step_min=float(getattr(args, "vi_ipm_step_min", 1.0e-10)),
                interior_point_initial_push=float(getattr(args, "vi_ipm_initial_push", 1.0e-8)),
                interior_point_stage_tol_factor=float(getattr(args, "vi_ipm_stage_tol_factor", 0.25)),
            ),
            **common_kwargs,
        )
    mono._set_solver_active_fields_with_tracking(target_solver, stage.active_fields)
    mono._bind_solver_inactive_solid_interface_retagging(problem=problem, target_solver=target_solver)
    _configure_solver_scaling(target_solver, problem)
    if stage.name == "transport" and solver_kind in {"pdas", "ipm"}:
        target_solver.pre_cb = _transport_box_bounds_pre_cb(target_solver, problem=problem)
    return target_solver


def _stage_function_lists(problem: dict[str, object]) -> tuple[list[object], list[object], dict[str, object]]:
    current_functions = [
        problem["v_k"],
        problem["p_k"],
        *([problem["p_pore_k"]] if problem.get("p_pore_k") is not None else []),
        problem["vS_k"],
        problem["u_k"],
        problem["alpha_k"],
        *([problem["B_k"]] if problem.get("B_k") is not None else []),
        problem["mu_k"],
    ]
    previous_functions = [
        problem["v_n"],
        problem["p_n"],
        *([problem["p_pore_n"]] if problem.get("p_pore_n") is not None else []),
        problem["vS_n"],
        problem["u_n"],
        problem["alpha_n"],
        *([problem["B_n"]] if problem.get("B_n") is not None else []),
        problem["mu_n"],
    ]
    if problem.get("p_mean_k") is not None:
        current_functions.insert(2, problem["p_mean_k"])
        previous_functions.insert(2, problem["p_mean_n"])
    if problem.get("alpha_mass_lm_k") is not None:
        current_functions.insert(3, problem["alpha_mass_lm_k"])
        previous_functions.insert(3, problem["alpha_mass_lm_n"])
    if problem.get("pi_s_k") is not None:
        current_functions.insert(4, problem["pi_s_k"])
        previous_functions.insert(4, problem["pi_s_n"])
    if problem.get("lambda_drag_k") is not None:
        drag_insert = current_functions.index(problem["alpha_k"])
        prev_drag_insert = previous_functions.index(problem["alpha_n"])
        current_functions.insert(drag_insert, problem["lambda_drag_k"])
        previous_functions.insert(prev_drag_insert, problem["lambda_drag_n"])
    if problem.get("phi_k") is not None:
        current_functions.append(problem["phi_k"])
        previous_functions.append(problem["phi_n"])
    if problem.get("S_k") is not None:
        current_functions.append(problem["S_k"])
        previous_functions.append(problem["S_n"])
    dt_c = problem["_dt_constant"]
    aux_functions: dict[str, object] = {"dt": dt_c}
    if problem.get("reg_weight") is not None:
        aux_functions["reg_weight"] = problem["reg_weight"]
    if problem.get("interface_corner_taper") is not None:
        aux_functions["interface_corner_taper"] = problem["interface_corner_taper"]
    if problem.get("_staggered_flow_coupling_v") is not None:
        aux_functions["v_flow_coupling"] = problem["_staggered_flow_coupling_v"]
    if problem.get("_staggered_flow_coupling_p") is not None:
        aux_functions["p_flow_coupling"] = problem["_staggered_flow_coupling_p"]
    if problem.get("_staggered_flow_coupling_scale") is not None:
        aux_functions["flow_coupling_scale"] = problem["_staggered_flow_coupling_scale"]
    for key, aux_name in (
        ("_staggered_if_entry_drive", "if_entry_drive"),
        ("_staggered_if_entry_drive_candidate", "if_entry_drive_candidate"),
        ("_staggered_if_porous_flux_n", "if_porous_flux_n"),
        ("_staggered_if_porous_flux_n_candidate", "if_porous_flux_n_candidate"),
        ("_staggered_if_porous_traction_n", "if_porous_traction_n"),
        ("_staggered_if_porous_traction_n_candidate", "if_porous_traction_n_candidate"),
        ("_staggered_if_free_traction_n", "if_free_traction_n"),
        ("_staggered_if_free_traction_n_candidate", "if_free_traction_n_candidate"),
        ("_staggered_if_free_flux_n", "if_free_flux_n"),
        ("_staggered_if_free_flux_n_candidate", "if_free_flux_n_candidate"),
    ):
        if problem.get(key) is not None:
            aux_functions[aux_name] = problem[key]
    return current_functions, previous_functions, aux_functions


def _interface_summary_to_metrics(interface_summary: dict[str, object]) -> dict[str, float]:
    row: dict[str, float] = {}
    for src_key, dst_key in (
        ("interface_band_point_count", "interface_band_point_count"),
        ("interface_line_point_count", "interface_line_point_count"),
        ("interface_line_y", "interface_line_y"),
        ("interface_line_mass_flux_jump_n_maxabs", "interface_mass_flux_jump_n_maxabs"),
        ("interface_line_traction_jump_n_maxabs", "interface_traction_jump_n_maxabs"),
        ("interface_line_traction_jump_t_maxabs", "interface_traction_jump_t_maxabs"),
        ("interface_line_traction_jump_mag_maxabs", "interface_traction_jump_mag_maxabs"),
        ("interface_line_entry_residual_maxabs", "interface_entry_residual_maxabs"),
        ("interface_line_pressure_jump_eff_maxabs", "interface_pressure_jump_eff_maxabs"),
        ("interface_band_drag_power_density_meanabs", "interface_drag_power_density_meanabs"),
        ("interface_band_fluid_visc_diss_density_meanabs", "interface_fluid_visc_diss_density_meanabs"),
        ("interface_band_support_solid_visc_diss_density_meanabs", "interface_support_solid_visc_diss_density_meanabs"),
        ("centerline_at_y_interface_p", "centerline_at_y_interface_p"),
        ("centerline_at_y_interface_p_pore", "centerline_at_y_interface_p_pore"),
        ("centerline_at_y_interface_p_pore_support", "centerline_at_y_interface_p_pore_support"),
        ("centerline_at_y_interface_mass_flux_jump_n", "centerline_at_y_interface_mass_flux_jump_n"),
        ("centerline_at_y_interface_traction_jump_n", "centerline_at_y_interface_traction_jump_n"),
        ("centerline_at_y_interface_traction_jump_t", "centerline_at_y_interface_traction_jump_t"),
        ("centerline_at_y_interface_entry_residual", "centerline_at_y_interface_entry_residual"),
        ("centerline_at_y_profile_p", "centerline_at_y_profile_p"),
        ("centerline_at_y_profile_p_pore", "centerline_at_y_profile_p_pore"),
        ("centerline_at_y_profile_p_pore_support", "centerline_at_y_profile_p_pore_support"),
        ("centerline_at_y_profile_mass_flux_jump_n", "centerline_at_y_profile_mass_flux_jump_n"),
        ("centerline_at_y_profile_traction_jump_n", "centerline_at_y_profile_traction_jump_n"),
        ("centerline_at_y_profile_traction_jump_t", "centerline_at_y_profile_traction_jump_t"),
        ("centerline_at_y_profile_entry_residual", "centerline_at_y_profile_entry_residual"),
    ):
        if src_key not in interface_summary:
            continue
        try:
            value = float(interface_summary.get(src_key, float("nan")))
        except Exception:
            continue
        if np.isfinite(value):
            row[dst_key] = value
    return row


def _write_interface_diagnostics(outdir: Path, interface_diag: dict[str, object], *, latest: bool) -> None:
    suffix = "_latest" if latest else ""
    if interface_diag.get("band_rows"):
        mono._write_timeseries_csv(outdir / f"interface_band_diagnostics{suffix}.csv", interface_diag["band_rows"])
    if interface_diag.get("interface_profile_rows"):
        mono._write_timeseries_csv(outdir / f"interface_profile_diagnostics{suffix}.csv", interface_diag["interface_profile_rows"])
    if interface_diag.get("interface_line_rows"):
        mono._write_timeseries_csv(outdir / f"interface_line_diagnostics{suffix}.csv", interface_diag["interface_line_rows"])
    if interface_diag.get("centerline_rows"):
        mono._write_timeseries_csv(outdir / f"centerline_diagnostics{suffix}.csv", interface_diag["centerline_rows"])
    (outdir / f"interface_diagnostics_summary{suffix}.json").write_text(
        json.dumps(dict(interface_diag.get("summary", {}) or {}), indent=2),
        encoding="utf-8",
    )


def _record_step(
    *,
    args: argparse.Namespace,
    problem: dict[str, object],
    outdir: Path,
    vtk_dir: Path,
    timeseries_rows: list[dict[str, float]],
    alpha_diagnostics,
    alpha_area0: float,
    alpha_band0: float,
    moving_ref,
    fixed_ref,
    nonlinear_ref,
    eps_alpha_eff: float,
    t_now: float,
    dt_now: float,
    step_no: int,
    outer_sweeps: int,
    outer_delta_inf: float,
    outer_delta_abs_inf: float,
    outer_converged: bool,
    outer_last_omega: float,
    stage_iters: dict[str, int],
    interface_entry_delta: float,
) -> dict[str, object]:
    alpha_diag = alpha_diagnostics.evaluate({problem["alpha_k"].name: problem["alpha_k"]})
    alpha_area = float(alpha_diag.get("alpha_area", float("nan")))
    alpha_band = float(alpha_diag.get("alpha_band", float("nan")))
    interface_diag = mono._compute_interface_probe_diagnostics(
        problem=problem,
        Lx=float(args.Lx),
        y_interface=float(args.y_interface),
        y_profile=float(args.y_profile),
        eps_alpha=float(eps_alpha_eff),
        mu_f=float(args.mu_f),
        mu_s=float(args.mu_s),
        lambda_s=float(args.lambda_s),
        solid_visco_eta=float(args.solid_visco_eta),
        solid_model=str(args.solid_model),
        skeleton_pressure_mode=str(args.skeleton_pressure_mode),
        alpha_biot=(None if args.alpha_biot is None else float(args.alpha_biot)),
        interface_entry_delta=float(interface_entry_delta),
        interface_bjs_gamma=float(getattr(args, "interface_bjs_gamma", 1.0e3)),
    )
    interface_summary = dict(interface_diag.get("summary", {}) or {})
    row = {
        "step": float(step_no),
        "t": float(t_now),
        "dt": float(dt_now),
        "outer_sweeps": float(outer_sweeps),
        "outer_delta_inf": float(outer_delta_inf),
        "outer_delta_abs_inf": float(outer_delta_abs_inf),
        "outer_converged": float(1.0 if bool(outer_converged) else 0.0),
        "outer_last_omega": float(outer_last_omega),
        "solid_iters": float(stage_iters.get("solid", 0)),
        "transport_iters": float(stage_iters.get("transport", 0)),
        "flow_iters": float(stage_iters.get("flow", 0)),
        "v_max": float(np.max(np.abs(problem["v_k"].nodal_values))),
        "p_max": float(np.max(np.abs(problem["p_k"].nodal_values))),
        "p_pore_max": float(np.max(np.abs(problem["p_pore_k"].nodal_values))),
        "vS_max": float(np.max(np.abs(problem["vS_k"].nodal_values))),
        "u_y_max": float(np.max(mono._vector_component_values(problem["u_k"], 1))),
        "u_y_min": float(np.min(mono._vector_component_values(problem["u_k"], 1))),
        "alpha_min": float(np.min(problem["alpha_k"].nodal_values)),
        "alpha_max": float(np.max(problem["alpha_k"].nodal_values)),
        "alpha_area": alpha_area,
        "alpha_area_rel_drift": float((alpha_area - alpha_area0) / max(abs(alpha_area0), 1.0e-30)),
        "alpha_band": alpha_band,
        "alpha_band_rel_drift": float((alpha_band - alpha_band0) / max(abs(alpha_band0), 1.0e-30)),
    }
    row.update(_interface_summary_to_metrics(interface_summary))
    if problem.get("B_k") is not None:
        row["B_min"] = float(np.min(problem["B_k"].nodal_values))
        row["B_max"] = float(np.max(problem["B_k"].nodal_values))
    if problem.get("S_k") is not None:
        row["S_min"] = float(np.min(problem["S_k"].nodal_values))
        row["S_max"] = float(np.max(problem["S_k"].nodal_values))
    profile_x, profile_uy = mono._sample_profile(
        problem=problem,
        Lx=float(args.Lx),
        y_profile=float(args.y_profile),
        n_samples=int(args.profile_samples),
    )
    if moving_ref is not None:
        moving_metrics = mono._compute_profile_metrics(
            x_num=profile_x,
            y_num=profile_uy,
            x_ref=moving_ref[0],
            y_ref=moving_ref[1],
        )
        row["rmse_over_amp_moving_linear"] = float(moving_metrics.rmse_over_amplitude)
        row["linf_over_amp_moving_linear"] = float(moving_metrics.linf_over_amplitude)
    if fixed_ref is not None:
        fixed_metrics = mono._compute_profile_metrics(
            x_num=profile_x,
            y_num=profile_uy,
            x_ref=fixed_ref[0],
            y_ref=fixed_ref[1],
        )
        row["rmse_over_amp_fixed_linear"] = float(fixed_metrics.rmse_over_amplitude)
    if nonlinear_ref is not None:
        nonlinear_metrics = mono._compute_profile_metrics(
            x_num=profile_x,
            y_num=profile_uy,
            x_ref=nonlinear_ref[0],
            y_ref=nonlinear_ref[1],
        )
        row["rmse_over_amp_moving_nonlinear"] = float(nonlinear_metrics.rmse_over_amplitude)
    timeseries_rows.append(row)
    if mono._mpi_io_root():
        mono._write_timeseries_csv(outdir / "timeseries.csv", timeseries_rows)
        mono._write_profile_csv(outdir / "profile_latest.csv", x=profile_x, u_y=profile_uy)
        _write_interface_diagnostics(outdir, interface_diag, latest=True)
        if int(args.vtk_every) > 0 and (step_no % int(args.vtk_every) == 0):
            vtk_dir.mkdir(parents=True, exist_ok=True)
            vtk_functions = {
                "v": problem["v_k"],
                "p": problem["p_k"],
                "p_pore": problem["p_pore_k"],
                "vS": problem["vS_k"],
                "u": problem["u_k"],
                "alpha": problem["alpha_k"],
                **({"B": problem["B_k"]} if problem.get("B_k") is not None else {}),
                "mu_alpha": problem["mu_k"],
            }
            if problem.get("S_k") is not None:
                vtk_functions["S"] = problem["S_k"]
            if problem.get("reg_weight") is not None:
                vtk_functions["reg_weight"] = problem["reg_weight"]
            if problem.get("interface_corner_taper") is not None:
                vtk_functions["interface_corner_taper"] = problem["interface_corner_taper"]
            export_vtk(
                str(vtk_dir / f"step={step_no:05d}.vtu"),
                mesh=problem["mesh"],
                dof_handler=problem["dh"],
                functions=vtk_functions,
            )
    if "rmse_over_amp_moving_linear" in row:
        print(
            "    [profile] "
            f"step={step_no} t={t_now:.6e} "
            f"u_y_max={row['u_y_max']:.6e} "
            f"rmse_amp_moving_linear={row['rmse_over_amp_moving_linear']:.6e} "
            f"alpha_area_rel_drift={row['alpha_area_rel_drift']:.3e}",
            flush=True,
        )
    if (
        "interface_mass_flux_jump_n_maxabs" in row
        or "interface_traction_jump_mag_maxabs" in row
        or "interface_entry_residual_maxabs" in row
    ):
        print(
            "    [interface] "
            f"step={step_no} t={t_now:.6e} "
            f"mass_flux_jump_n_maxabs={row.get('interface_mass_flux_jump_n_maxabs', float('nan')):.6e} "
            f"traction_jump_mag_maxabs={row.get('interface_traction_jump_mag_maxabs', float('nan')):.6e} "
            f"entry_residual_maxabs={row.get('interface_entry_residual_maxabs', float('nan')):.6e}",
            flush=True,
        )
    return interface_diag


def _temporarily_relax_stage_solvers(
    stage_solvers: dict[str, object],
    *,
    accept_factor: float,
) -> list[tuple[object, float, int, float, float, float, float]]:
    saved: list[tuple[object, float, int, float, float, float, float]] = []
    for solver in list(stage_solvers.values()):
        np_params = getattr(solver, "np", None)
        if np_params is None:
            continue
        saved.append(
            (
                solver,
                float(getattr(np_params, "accept_nonconverged_atol_factor", 0.0) or 0.0),
                int(getattr(np_params, "stall_window", 0) or 0),
                float(getattr(np_params, "stall_min_abs_decrease_inf", 0.0) or 0.0),
                float(getattr(np_params, "stall_min_rel_decrease_inf", 0.0) or 0.0),
                float(getattr(np_params, "tr_min_abs_decrease_inf", 0.0) or 0.0),
                float(getattr(np_params, "tr_min_rel_decrease_inf", 0.0) or 0.0),
            )
        )
        np_params.accept_nonconverged_atol_factor = max(
            float(getattr(np_params, "accept_nonconverged_atol_factor", 0.0) or 0.0),
            float(accept_factor),
        )
        np_params.stall_window = 0
        np_params.stall_min_abs_decrease_inf = 0.0
        np_params.stall_min_rel_decrease_inf = 0.0
        np_params.tr_min_abs_decrease_inf = 0.0
        np_params.tr_min_rel_decrease_inf = 0.0
    return saved


def _restore_stage_solver_relaxation(saved: list[tuple[object, float, int, float, float, float, float]]) -> None:
    for solver, accept_factor, stall_window, stall_abs, stall_rel, tr_abs, tr_rel in list(saved or []):
        np_params = getattr(solver, "np", None)
        if np_params is None:
            continue
        np_params.accept_nonconverged_atol_factor = float(accept_factor)
        np_params.stall_window = int(stall_window)
        np_params.stall_min_abs_decrease_inf = float(stall_abs)
        np_params.stall_min_rel_decrease_inf = float(stall_rel)
        np_params.tr_min_abs_decrease_inf = float(tr_abs)
        np_params.tr_min_rel_decrease_inf = float(tr_rel)


def _run_stage_with_bc_scales(
    *,
    args: argparse.Namespace,
    problem: dict[str, object],
    stage_name: str,
    solver,
    stage_bcs_base: list[BoundaryCondition],
    current_functions,
    previous_functions,
    aux_functions: dict[str, object],
    t_now: float,
    dt_try: float,
    step_no: int,
    outer_it: int | None = None,
    preload_it: int | None = None,
    apply_startup_flow_scales: bool = True,
) -> int:
    bc_scales = (1.0,)
    if bool(apply_startup_flow_scales) and str(stage_name) == "flow":
        bc_scales = _startup_flow_scale_schedule(args, step_no=int(step_no))
    stage_nit_total = 0
    for bc_scale in tuple(bc_scales):
        stage_bcs_now = (
            _scaled_dirichlet_bcs(stage_bcs_base, scale=float(bc_scale))
            if float(bc_scale) < 1.0 - 1.0e-14
            else stage_bcs_base
        )
        problem["dh"].apply_bcs(stage_bcs_now, *current_functions)
        solver._current_t = float(t_now)
        solver._current_dt = float(dt_try)
        solver._current_step_no = int(step_no + 1)
        if str(stage_name) == "flow" and len(tuple(bc_scales)) > 1:
            context_bits = [f"step={int(step_no) + 1}"]
            if preload_it is not None:
                context_bits.append(f"preload={int(preload_it)}")
            if outer_it is not None:
                context_bits.append(f"outer={int(outer_it)}")
            context_bits.append(f"scale={float(bc_scale):.3f}")
            print("    [flow-continuation] " + " ".join(context_bits), flush=True)
        _, converged, nit = solver._newton_loop(
            current_functions,
            previous_functions,
            aux_functions,
            stage_bcs_now,
        )
        stage_nit_total += int(nit)
        if not bool(converged):
            where = f"outer sweep {int(outer_it)}" if outer_it is not None else f"preload sweep {int(preload_it)}"
            raise RuntimeError(f"{stage_name} stage did not converge on {where}")
        if str(stage_name) == "flow":
            _commit_flow_coupling_from_current(problem)
    return int(stage_nit_total)


def _run_case(args: argparse.Namespace, *, kappa: float, outdir: Path) -> mono.CaseResult:
    _validate_supported_configuration(args)
    io_root = mono._mpi_io_root()
    stage_order = _stage_order(args)
    transport_post_accept = bool(getattr(args, "transport_post_accept", True))
    coupling_stage_order = tuple(name for name in stage_order if str(name) != "transport")
    flow_solid_substeps = max(1, int(getattr(args, "flow_solid_substeps", 1)))
    use_flow_solid_inner = (
        len(stage_order) == 3
        and stage_order[-1] == "transport"
        and set(stage_order[:2]) == {"flow", "solid"}
    )
    poly_order, pressure_order, scalar_order = mono._resolved_orders(args)
    qdeg = int(args.quad_order) if args.quad_order is not None else max(6, 2 * int(poly_order) + 2)
    eps_alpha_eff = mono._effective_eps_alpha(args)
    alpha_reg_eps_normal = float(args.alpha_reg_eps_normal) if args.alpha_reg_eps_normal is not None else float(eps_alpha_eff)
    alpha_reg_eps_tangent = float(args.alpha_reg_eps_tangent) if args.alpha_reg_eps_tangent is not None else float(0.25 * eps_alpha_eff)
    h_char = mono._characteristic_h(Lx=float(args.Lx), Ly=float(args.Ly), nx=int(args.nx), ny=int(args.ny))
    entry_delta_eff = float(getattr(args, "interface_entry_delta", 10.0))
    interface_robin_l = float(getattr(args, "interface_robin_l", 2000.0))

    mechanics_nondim_key = str(args.mechanics_nondim_mode).strip().lower()
    volume_setup = mono._condition_balanced_volume_setup(
        mechanics_nondim_mode=mechanics_nondim_key,
        mu_f=float(args.mu_f),
        kappa_inv=float(1.0 / float(kappa)),
        gamma_div=float(args.gamma_div),
        auto_gamma_div=bool(getattr(args, "condition_balanced_auto_gamma_div", True)),
    )
    effective_gamma_div = float(volume_setup["gamma_div"])
    if bool(mono._full_ratio_free_state_enabled(args)):
        effective_gamma_div = 0.0
    solid_dof_y_cut = mono._condition_balanced_solid_cutoff_y(
        mechanics_nondim_mode=mechanics_nondim_key,
        y_interface=float(args.y_interface),
        solid_dof_y_cut=args.solid_dof_y_cut,
        condition_balanced_solid_cut_fix=bool(getattr(args, "condition_balanced_solid_cut_fix", True)),
    )

    problem = mono._create_problem(
        Lx=float(args.Lx),
        Ly=float(args.Ly),
        nx=int(args.nx),
        ny=int(args.ny),
        poly_order=int(poly_order),
        pressure_order=int(pressure_order),
        scalar_order=int(scalar_order),
        fluid_space=str(args.fluid_space),
        fluid_hdiv_order=int(args.fluid_hdiv_order),
        enable_phi_evolution=bool(args.enable_phi_evolution),
        reduced_support_state=str(getattr(args, "reduced_support_state", "alpha_B")),
        latent_bounded_transport=False,
        latent_bounded_fields=tuple(),
        latent_bounded_map=str(args.latent_bounded_map),
        latent_bounded_formulation=str(args.latent_bounded_formulation),
        alpha_mass_constraint=bool(args.alpha_mass_constraint),
        pressure_mean_constraint=bool(args.pressure_mean_constraint),
        solid_volumetric_split=bool(args.solid_volumetric_split),
        drag_formulation=str(args.drag_formulation),
        full_ratio_free_state=bool(getattr(args, "full_ratio_free_state", False)),
    )
    if problem.get("q_flux_k") is None and problem.get("lambda_drag_k") is not None:
        problem["q_flux_k"] = problem.get("lambda_drag_k")
        problem["q_flux_n"] = problem.get("lambda_drag_n")
        problem["dq_flux"] = problem.get("dlambda_drag")
        problem["q_flux_test"] = problem.get("lambda_drag_test")
    problem["_effective_gamma_div"] = float(effective_gamma_div)
    problem["_interface_entry_delta"] = float(entry_delta_eff)
    problem["_interface_bjs_gamma"] = float(getattr(args, "interface_bjs_gamma", 1.0e3))
    problem["_interface_bjs_closure_enabled"] = bool(getattr(args, "interface_bjs_closure", False))
    dt_c = mono._named_constant("b7_dt", float(args.dt))
    problem["_dt_constant"] = dt_c
    _build_flow_coupling_state(problem)
    _build_interface_transfer_state(problem)

    reg_mask_meta = mono._configure_regularization_mask(
        problem=problem,
        enabled=bool(args.reg_rect),
        Lx=float(args.Lx),
        Ly=float(args.Ly),
        y_interface=float(args.y_interface),
        center_x=args.reg_rect_center_x,
        center_y=args.reg_rect_center_y,
        half_width=args.reg_rect_half_width,
        half_height=args.reg_rect_half_height,
    )
    interface_corner_taper_meta = mono._configure_interface_corner_taper(
        problem=problem,
        Lx=float(args.Lx),
        nx=int(args.nx),
        eps_alpha=float(eps_alpha_eff),
    )
    if solid_dof_y_cut is not None:
        solid_inactive_counts = mono._tag_inactive_solid_dofs_above_y(problem, y_cut=solid_dof_y_cut)
    else:
        problem["_inactive_solid_reference_y"] = None
        problem["_inactive_solid_alpha_band_halfwidth"] = float("nan")
        problem["_inactive_solid_alpha_phase"] = "none"
        solid_inactive_counts = {}

    ds_alpha_transport, ds_B_transport = mono._build_transport_measures(
        problem=problem,
        qdeg=int(qdeg),
        enable_phi_evolution=bool(args.enable_phi_evolution),
        top_drainage_transport=bool(args.top_drainage_transport),
        support_physics=str(args.support_physics),
        internal_conversion_open_top_b_transport=bool(getattr(args, "internal_conversion_open_top_b_transport", False)),
    )

    alpha_init = lambda x, y: mono._alpha_equilibrium(
        y,
        y_interface=float(args.y_interface),
        eps_alpha=float(eps_alpha_eff),
    )
    problem["alpha_n"].set_values_from_function(lambda x, y: float(alpha_init(x, y)))
    problem["alpha_k"].nodal_values[:] = problem["alpha_n"].nodal_values[:]
    if problem.get("B_n") is not None:
        B_init = (1.0 - float(args.phi_b)) * np.asarray(problem["alpha_n"].nodal_values, dtype=float)
        problem["B_n"].nodal_values[:] = B_init
        problem["B_k"].nodal_values[:] = B_init
    problem["_freeze_B_total_mixture_continuity_diagnostic"] = bool(
        getattr(args, "freeze_B_total_mixture_continuity_diagnostic", False)
    )
    if problem.get("mu_n") is not None:
        problem["mu_n"].nodal_values[:] = 0.0
        problem["mu_k"].nodal_values[:] = 0.0
    if problem.get("S_n") is not None:
        problem["S_n"].nodal_values[:] = 0.0
        problem["S_k"].nodal_values[:] = 0.0
    if problem.get("alpha_mass_lm_n") is not None:
        problem["alpha_mass_lm_n"].nodal_values[:] = 0.0
        problem["alpha_mass_lm_k"].nodal_values[:] = 0.0
    for key in ("p_k", "p_n", "p_pore_k", "p_pore_n", "p_mean_k", "p_mean_n", "pi_s_k", "pi_s_n"):
        if problem.get(key) is not None:
            problem[key].nodal_values[:] = 0.0

    forms = mono._build_forms(
        problem,
        qdeg=qdeg,
        dt_c=dt_c,
        theta=float(args.theta),
        rho_f=float(args.rho_f),
        mu_f=float(args.mu_f),
        mu_b=float(args.mu_b),
        mu_b_model=str(args.mu_b_model),
        kappa_inv=1.0 / float(kappa),
        mu_s=float(args.mu_s),
        lambda_s=float(args.lambda_s),
        phi_b=float(args.phi_b),
        M_alpha=float(args.M_alpha),
        gamma_alpha=float(args.gamma_alpha),
        eps_alpha=float(eps_alpha_eff),
        solid_visco_eta=float(args.solid_visco_eta),
        gamma_div=float(effective_gamma_div),
        mechanics_nondim_mode=str(args.mechanics_nondim_mode),
        solid_volumetric_split=bool(args.solid_volumetric_split),
        solid_volumetric_penalty=float(args.solid_volumetric_penalty),
        gamma_u=float(args.gamma_u),
        u_extension_mode=str(args.u_extension),
        gamma_u_pin=float(args.gamma_u_pin),
        u_cip=float(args.u_cip),
        u_cip_weight=str(args.u_cip_weight),
        vS_cip=float(args.vS_cip),
        gamma_vS=args.gamma_vS,
        vS_extension_mode=args.vS_ext_mode,
        gamma_vS_pin=args.gamma_vS_pin,
        D_phi=float(args.D_phi),
        phi_diffusion_weight=str(args.phi_diffusion_weight),
        gamma_phi=float(args.gamma_phi),
        phi_supg=float(args.phi_supg),
        phi_cip=float(args.phi_cip),
        alpha_supg=float(args.alpha_supg),
        alpha_cip=float(args.alpha_cip),
        v_supg=float(getattr(args, "v_supg", 0.0)),
        v_supg_mode=str(getattr(args, "v_supg_mode", "streamline")),
        v_supg_c_nu=float(getattr(args, "v_supg_c_nu", 4.0)),
        u_supg=float(getattr(args, "u_supg", 0.0)),
        v_cip=float(getattr(args, "v_cip", 0.0)),
        alpha_regularization=str(args.alpha_regularization),
        alpha_reg_gamma=float(args.alpha_reg_gamma),
        alpha_reg_eps_normal=float(alpha_reg_eps_normal),
        alpha_reg_eps_tangent=float(alpha_reg_eps_tangent),
        alpha_reg_eta=float(args.alpha_reg_eta),
        alpha_advect_with=str(args.alpha_advect_with),
        alpha_advection_form=str(args.alpha_advection_form),
        support_physics=str(args.support_physics),
        solid_model=str(args.solid_model),
        kappa_inv_model=str(args.kappa_inv_model),
        reduced_support_state=str(getattr(args, "reduced_support_state", "alpha_B")),
        drag_formulation=str(args.drag_formulation),
        full_ratio_free_state=bool(getattr(args, "full_ratio_free_state", False)),
        pressure_interface_closure=bool(getattr(args, "pressure_interface_closure", True)),
        pressure_interface_closure_strength=float(getattr(args, "pressure_interface_closure_strength", 1.0)),
        p_pore_fluid_gauge=bool(getattr(args, "p_pore_fluid_gauge", True)),
        p_pore_fluid_gauge_strength=float(getattr(args, "p_pore_fluid_gauge_strength", 1.0)),
        interface_entry_closure=bool(getattr(args, "interface_entry_closure", True)),
        interface_entry_closure_strength=float(getattr(args, "interface_entry_closure_strength", 1.0)),
        interface_entry_delta=float(entry_delta_eff),
        interface_bjs_closure=bool(getattr(args, "interface_bjs_closure", False)),
        interface_bjs_closure_strength=float(getattr(args, "interface_bjs_closure_strength", 1.0)),
        interface_bjs_gamma=float(getattr(args, "interface_bjs_gamma", 1.0e3)),
        fluid_convection=str(args.fluid_convection),
        skeleton_pressure_mode=str(args.skeleton_pressure_mode),
        alpha_biot=(None if args.alpha_biot is None else float(args.alpha_biot)),
        enable_phi_evolution=bool(args.enable_phi_evolution),
        include_skeleton_acceleration=bool(args.include_skeleton_acceleration),
        rho_s0_tilde=float(args.rho_s0_tilde),
        storativity_c0=float(args.storativity_c0),
        skeleton_inertia_convection=str(args.skeleton_inertia_convection),
        ds_hdiv_tangential=None,
        ds_alpha_transport=ds_alpha_transport,
        ds_B_transport=ds_B_transport,
        hdiv_tangential_gamma=float(args.hdiv_tangential_gamma),
        hdiv_tangential_method=str(args.hdiv_tangential_method),
        interface_traction_continuity_closure=bool(getattr(args, "interface_traction_continuity_closure", False)),
        interface_traction_normal_strength=float(getattr(args, "interface_traction_normal_strength", 1.0)),
        interface_traction_tangential_strength=float(getattr(args, "interface_traction_tangential_strength", 0.0)),
    )
    _build_staggered_interface_forms(
        problem,
        qdeg=int(qdeg),
        eps_alpha=float(eps_alpha_eff),
        mu_f=float(args.mu_f),
        mu_s=float(args.mu_s),
        lambda_s=float(args.lambda_s),
        solid_visco_eta=float(args.solid_visco_eta),
        solid_model=str(args.solid_model),
        skeleton_pressure_mode=str(args.skeleton_pressure_mode),
        alpha_biot=(None if args.alpha_biot is None else float(args.alpha_biot)),
        interface_entry_closure=bool(getattr(args, "interface_entry_closure", True)),
        interface_entry_closure_strength=float(getattr(args, "interface_entry_closure_strength", 1.0)),
        interface_entry_delta=float(entry_delta_eff),
        interface_velocity_continuity_closure=bool(getattr(args, "interface_velocity_continuity_closure", True)),
        interface_traction_continuity_closure=bool(getattr(args, "interface_traction_continuity_closure", False)),
        interface_velocity_normal_strength=float(getattr(args, "interface_velocity_normal_strength", 1.0)),
        interface_traction_normal_strength=float(getattr(args, "interface_traction_normal_strength", 1.0)),
    )

    bcs = mono._build_bcs(
        fluid_space=str(args.fluid_space),
        enable_phi_evolution=bool(args.enable_phi_evolution),
        full_ratio_free_state=bool(getattr(args, "full_ratio_free_state", False)),
        y_interface=float(args.y_interface),
        eps_alpha=float(eps_alpha_eff),
        phi_b=float(args.phi_b),
        v_in=float(args.v_in),
        t_ramp=float(args.t_ramp),
        alpha_bc_mode=str(args.alpha_bc_mode),
        phi_bc_mode=str(getattr(args, "phi_bc_mode", "natural")),
        alpha_solid_dirichlet_sides=bool(args.alpha_solid_dirichlet_sides),
        alpha_solid_dirichlet_bottom=bool(args.alpha_solid_dirichlet_bottom),
        solid_bc_mode=str(args.solid_bc_mode),
        latent_bounded_fields=tuple(),
        latent_bounded_eps=float(getattr(args, "latent_bounded_eps", 1.0e-8)),
        latent_bounded_map=str(getattr(args, "latent_bounded_map", "sigmoid")),
        pressure_mean_constraint=bool(args.pressure_mean_constraint),
    )
    bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, (lambda x, y: 0.0)) for bc in bcs]
    stage_forms = _build_stage_forms(problem, forms)

    current_functions, previous_functions, aux_functions = _stage_function_lists(problem)
    outer_coupling_fields = _outer_coupling_function_names(problem)
    alpha_diagnostics = mono._build_alpha_diagnostics(problem, quad_order=int(qdeg), backend=str(args.backend))
    alpha_diag0 = alpha_diagnostics.evaluate({problem["alpha_k"].name: problem["alpha_k"]})
    alpha_area0 = float(alpha_diag0.get("alpha_area", float("nan")))
    alpha_band0 = float(alpha_diag0.get("alpha_band", float("nan")))

    if io_root:
        outdir.mkdir(parents=True, exist_ok=True)
    mono.barrier(mono.MPI_CTX)
    vtk_dir = outdir / "vtk"

    stage_solvers = {}
    for stage_name in stage_order:
        stage = getattr(stage_forms, stage_name)
        stage_bcs = _filter_bcs_by_fields(bcs, stage.active_fields)
        stage_bcs_homog = _filter_bcs_by_fields(bcs_homog, stage.active_fields)
        stage_solvers[stage.name] = _make_stage_solver(
            args=args,
            problem=problem,
            stage=stage,
            bcs=stage_bcs,
            bcs_homog=stage_bcs_homog,
            qdeg=qdeg,
            aux_functions=aux_functions,
        )

    reference_csv = Path(args.reference_csv).resolve()
    moving_ref = mono._load_reference_curve(
        reference_csv=reference_csv,
        kappa=float(kappa),
        curve_label="partitioned_moving_linear",
    )
    fixed_ref = mono._load_reference_curve(
        reference_csv=reference_csv,
        kappa=float(kappa),
        curve_label="partitioned_fixed_linear",
    )
    nonlinear_ref = mono._load_reference_curve(
        reference_csv=reference_csv,
        kappa=float(kappa),
        curve_label="partitioned_moving_nonlinear",
    )

    solve_error = ""
    timeseries_rows: list[dict[str, float]] = []
    last_interface_diag: dict[str, object] | None = None
    t_now = 0.0
    dt_step = float(args.dt)
    dt_min = float(getattr(args, "dt_min", 0.0) or 0.0)
    step_no = 0
    base_outer_omega = float(
        np.clip(
            float(getattr(args, "outer_relaxation", 1.0)),
            float(getattr(args, "outer_relaxation_min", 1.0e-3)),
            float(getattr(args, "outer_relaxation_max", 1.0)),
        )
    )
    outer_update = str(getattr(args, "outer_update", "constant")).strip().lower()
    outer_accelerator = create_coupling_accelerator(
        outer_update,
        relaxation=float(base_outer_omega),
        relaxation_min=float(getattr(args, "outer_relaxation_min", 1.0e-3)),
        relaxation_max=float(getattr(args, "outer_relaxation_max", 1.0)),
        history=int(getattr(args, "outer_history", 6)),
        regularization=float(getattr(args, "outer_regularization", 1.0e-10)),
        timestep_horizon=int(getattr(args, "outer_timestep_history", 1)),
    )
    t_start = time.perf_counter()
    try:
        while t_now < float(args.t_final) - 1.0e-14:
            dt_try = min(float(dt_step), float(args.t_final) - float(t_now))
            if dt_try <= 0.0:
                break
            while True:
                dt_c.value = float(dt_try)
                _copy_previous_into_current(current_functions, previous_functions)
                _reset_flow_coupling_from_previous(problem)
                _set_flow_coupling_scale(problem, scale=1.0)
                t_bc = float(t_now + dt_try)
                bcs_now = NewtonSolver._freeze_bcs(bcs, t_bc)
                problem["dh"].apply_bcs(bcs_now, *current_functions)
                if int(step_no) == 0 and int(getattr(args, "startup_preload_sweeps", 0) or 0) > 0:
                    preload_snapshot = _snapshot_function_values(current_functions)
                    preload_saved = _temporarily_relax_stage_solvers(
                        stage_solvers,
                        accept_factor=float(getattr(args, "startup_preload_accept_factor", 5.0e3)),
                    )
                    preload_sweeps_done = 0
                    preload_stage_sequence = [
                        name
                        for name in (
                            coupling_stage_order
                            if bool(transport_post_accept)
                            else stage_order
                        )
                        if str(name) in stage_solvers
                    ]
                    preload_bc_schedule = _scaled_bc_schedule(
                        bcs_now,
                        scales=_startup_flow_scale_schedule(args, step_no=int(step_no)),
                    )
                    try:
                        for preload_it in range(1, int(getattr(args, "startup_preload_sweeps", 0)) + 1):
                            for startup_scale, preload_bcs_now in preload_bc_schedule:
                                _set_flow_coupling_scale(problem, scale=float(startup_scale))
                                if len(preload_bc_schedule) > 1:
                                    print(
                                        "    [startup-continuation] "
                                        f"step=1 preload={int(preload_it)} scale={float(startup_scale):.3f}",
                                        flush=True,
                                    )
                                for stage_name in preload_stage_sequence:
                                    stage = getattr(stage_forms, stage_name)
                                    solver = stage_solvers[stage_name]
                                    stage_bcs_now = _filter_bcs_by_fields(preload_bcs_now, stage.active_fields)
                                    try:
                                        _run_stage_with_bc_scales(
                                            args=args,
                                            problem=problem,
                                            stage_name=str(stage_name),
                                            solver=solver,
                                            stage_bcs_base=stage_bcs_now,
                                            current_functions=current_functions,
                                            previous_functions=previous_functions,
                                            aux_functions=aux_functions,
                                            t_now=float(t_now),
                                            dt_try=float(dt_try),
                                            step_no=int(step_no),
                                            preload_it=int(preload_it),
                                            apply_startup_flow_scales=False,
                                        )
                                    except Exception as preload_exc:
                                        print(
                                            "    [startup-preload] "
                                            f"stage={stage_name} sweep={int(preload_it)} "
                                            f"kept best available iterate after: {preload_exc}",
                                            flush=True,
                                        )
                                        break
                                    if str(stage_name) == "flow":
                                        _refresh_flow_to_solid_transfer(
                                            args=args,
                                            problem=problem,
                                            eps_alpha=float(eps_alpha_eff),
                                            interface_entry_delta=float(entry_delta_eff),
                                        )
                                    problem["dh"].apply_bcs(preload_bcs_now, *current_functions)
                            _commit_flow_coupling_from_current(problem)
                            _set_flow_coupling_scale(problem, scale=1.0)
                            preload_sweeps_done = int(preload_it)
                    finally:
                        _restore_stage_solver_relaxation(preload_saved)
                    preload_relaxed = _relax_preload_toward_previous(
                        current_functions,
                        previous_functions,
                        keep_fields=_transport_stage_fields(problem),
                        blend=float(getattr(args, "startup_preload_prev_blend", 1.0)),
                    )
                    preload_delta_inf, preload_delta_abs_inf, _ = _scaled_snapshot_delta(
                        current_functions,
                        preload_snapshot,
                        field_scales=problem.get("_condition_balanced_field_scales", {}),
                        field_names=outer_coupling_fields,
                    )
                    if preload_sweeps_done > 0:
                        print(
                            "    [startup-preload] "
                            f"sweeps={int(preload_sweeps_done)} "
                            f"rel={float(preload_delta_inf):.3e} "
                            f"abs={float(preload_delta_abs_inf):.3e}"
                            + (
                                f" blend_prev={float(getattr(args, 'startup_preload_prev_blend', 1.0)):.3f}"
                                if bool(preload_relaxed)
                                else ""
                            ),
                            flush=True,
                        )
                    problem["dh"].apply_bcs(bcs_now, *current_functions)
                _refresh_flow_to_solid_transfer(
                    args=args,
                    problem=problem,
                    eps_alpha=float(eps_alpha_eff),
                    interface_entry_delta=float(entry_delta_eff),
                )
                _refresh_solid_interface_candidate(
                    args=args,
                    problem=problem,
                    eps_alpha=float(eps_alpha_eff),
                    interface_entry_delta=float(entry_delta_eff),
                )
                _commit_flow_interface_response(problem)
                _copy_function_list_values(
                    _interface_candidate_functions(problem),
                    _interface_state_functions(problem),
                )
                stage_iters = {"solid": 0, "transport": 0, "flow": 0}
                outer_delta_inf = float("inf")
                outer_delta_abs_inf = float("inf")
                outer_delta_by_field: dict[str, float] = {}
                outer_physics_inf = float("inf")
                outer_physics_by_field: dict[str, float] = {}
                outer_sweeps = 0
                outer_converged = False
                outer_accelerator.initialize_solution_step()
                outer_last_omega = float(base_outer_omega)
                outer_saved = []
                try:
                    try:
                        for outer_it in range(1, int(args.outer_it) + 1):
                            sweep_snapshot = _snapshot_function_values(_interface_state_functions(problem))
                            stage_sequence: list[str] = []
                            outer_bc_schedule = ((1.0, bcs_now),)
                            if int(step_no) == 0:
                                if int(outer_it) == 1:
                                    print(
                                        "    [startup-order] "
                                        f"step=1 using {' -> '.join(coupling_stage_order)}"
                                        + (" with post-accept transport." if bool(transport_post_accept) else "."),
                                        flush=True,
                                    )
                                    outer_bc_schedule = _scaled_bc_schedule(
                                        bcs_now,
                                        scales=_startup_flow_scale_schedule(args, step_no=int(step_no)),
                                    )
                            if bool(transport_post_accept):
                                if use_flow_solid_inner:
                                    for _ in range(flow_solid_substeps):
                                        stage_sequence.extend(coupling_stage_order)
                                else:
                                    stage_sequence.extend(coupling_stage_order)
                            elif use_flow_solid_inner:
                                for _ in range(flow_solid_substeps):
                                    stage_sequence.extend(stage_order[:2])
                                stage_sequence.append("transport")
                            else:
                                stage_sequence.extend(stage_order)
                            for startup_scale, scaled_bcs_now in outer_bc_schedule:
                                _set_flow_coupling_scale(problem, scale=float(startup_scale))
                                if int(step_no) == 0 and len(tuple(outer_bc_schedule)) > 1:
                                    print(
                                        "    [startup-continuation] "
                                        f"step=1 outer={int(outer_it)} scale={float(startup_scale):.3f}",
                                        flush=True,
                                    )
                                for stage_name in stage_sequence:
                                    stage = getattr(stage_forms, stage_name)
                                    solver = stage_solvers[stage_name]
                                    stage_bcs_base = _filter_bcs_by_fields(scaled_bcs_now, stage.active_fields)
                                    stage_nit_total = _run_stage_with_bc_scales(
                                        args=args,
                                        problem=problem,
                                        stage_name=str(stage_name),
                                        solver=solver,
                                        stage_bcs_base=stage_bcs_base,
                                        current_functions=current_functions,
                                        previous_functions=previous_functions,
                                        aux_functions=aux_functions,
                                        t_now=float(t_now),
                                        dt_try=float(dt_try),
                                        step_no=int(step_no),
                                        outer_it=int(outer_it),
                                        apply_startup_flow_scales=False,
                                    )
                                    stage_iters[stage_name] += int(stage_nit_total)
                                    if str(stage_name) == "flow":
                                        _refresh_flow_to_solid_transfer(
                                            args=args,
                                            problem=problem,
                                            eps_alpha=float(eps_alpha_eff),
                                            interface_entry_delta=float(entry_delta_eff),
                                        )
                                        _commit_flow_interface_response(problem)
                                    problem["dh"].apply_bcs(scaled_bcs_now, *current_functions)
                            _set_flow_coupling_scale(problem, scale=1.0)
                            problem["dh"].apply_bcs(bcs_now, *current_functions)
                            interface_diag_current = _refresh_solid_interface_candidate(
                                args=args,
                                problem=problem,
                                eps_alpha=float(eps_alpha_eff),
                                interface_entry_delta=float(entry_delta_eff),
                            )
                            (
                                outer_delta_inf,
                                outer_delta_abs_inf,
                                outer_delta_by_field,
                                outer_residual_vec,
                                x_input,
                            ) = _interface_state_delta(
                                problem,
                                interface_diag=interface_diag_current,
                            )
                            outer_physics_inf, outer_physics_by_field = _outer_interface_physics(
                                args=args,
                                problem=problem,
                                eps_alpha=float(eps_alpha_eff),
                                interface_entry_delta=float(entry_delta_eff),
                            )
                            outer_sweeps = int(outer_it)
                            rel_ok = bool(np.isfinite(outer_delta_inf) and outer_delta_inf <= float(args.outer_tol))
                            abs_tol = float(getattr(args, "outer_abs_tol", 0.0) or 0.0)
                            abs_ok = bool(abs_tol > 0.0 and np.isfinite(outer_delta_abs_inf) and outer_delta_abs_inf <= abs_tol)
                            physics_ok = bool(np.isfinite(outer_physics_inf) and outer_physics_inf <= float(args.outer_tol))
                            outer_converged = bool((rel_ok or abs_ok) and physics_ok)
                            ranked_fields = dict(outer_delta_by_field)
                            ranked_fields.update(outer_physics_by_field)
                            top_fields = ", ".join(
                                f"{name}={value:.3e}"
                                for name, value in sorted(ranked_fields.items(), key=lambda item: float(item[1]), reverse=True)[:6]
                            )
                            print(
                                "    [fixed-point] "
                                f"step={int(step_no) + 1} iter={int(outer_it)}/{int(args.outer_it)} "
                                f"rel={outer_delta_inf:.3e} abs={outer_delta_abs_inf:.3e} "
                                f"physics={outer_physics_inf:.3e} "
                                f"omega={float(outer_last_omega):.3e} "
                                f"converged={int(bool(outer_converged))}"
                                + (f" [top: {top_fields}]" if top_fields else ""),
                                flush=True,
                            )
                            if outer_converged:
                                break
                            update = outer_accelerator.compute_next_iterate(
                                x_curr=np.asarray(x_input, dtype=float),
                                residual_curr=np.asarray(outer_residual_vec, dtype=float),
                            )
                            _set_function_vector(_interface_state_functions(problem), update.next_iterate)
                            outer_last_omega = float(update.relaxation)
                        if bool(transport_post_accept) and bool(outer_converged) and "transport" in stage_solvers:
                            stage = getattr(stage_forms, "transport")
                            solver = stage_solvers["transport"]
                            stage_bcs_base = _filter_bcs_by_fields(bcs_now, stage.active_fields)
                            stage_nit_total = _run_stage_with_bc_scales(
                                args=args,
                                problem=problem,
                                stage_name="transport",
                                solver=solver,
                                stage_bcs_base=stage_bcs_base,
                                current_functions=current_functions,
                                previous_functions=previous_functions,
                                aux_functions=aux_functions,
                                t_now=float(t_now),
                                dt_try=float(dt_try),
                                step_no=int(step_no),
                                apply_startup_flow_scales=False,
                            )
                            stage_iters["transport"] += int(stage_nit_total)
                            problem["dh"].apply_bcs(bcs_now, *current_functions)
                        if int(args.outer_it) > 1 and not bool(outer_converged):
                            top_fields = ", ".join(
                                f"{name}={value:.3e}"
                                for name, value in sorted({**outer_delta_by_field, **outer_physics_by_field}.items(), key=lambda item: float(item[1]), reverse=True)[:6]
                            )
                            raise RuntimeError(
                                f"staggered outer loop stalled at |delta|_inf={outer_delta_inf:.3e} "
                                f"(abs={outer_delta_abs_inf:.3e}, physics={outer_physics_inf:.3e}) "
                                f"[top: {top_fields}] "
                                f"after {int(args.outer_it)} sweeps"
                            )
                        outer_accelerator.finalize_solution_step(accepted=True)
                    finally:
                        _restore_stage_solver_relaxation(outer_saved)
                except Exception as exc:
                    outer_accelerator.finalize_solution_step(accepted=False)
                    if bool(getattr(args, "no_dt_reduction", False)):
                        raise RuntimeError(str(exc)) from exc
                    dt_new = float(dt_try) * float(getattr(args, "dt_reduction_factor", 0.5))
                    if dt_new <= max(dt_min, 1.0e-14) * (1.0 + 1.0e-12):
                        raise RuntimeError(str(exc)) from exc
                    print(
                        f"    [retry] step {int(step_no) + 1} failed with dt={float(dt_try):.3e}: {exc}. "
                        f"Reducing dt to {float(dt_new):.3e}.",
                        flush=True,
                    )
                    dt_try = float(dt_new)
                    continue
                step_no += 1
                t_now = float(t_now + dt_try)
                dt_step = float(dt_try)
                _copy_current_into_previous(current_functions, previous_functions)
                _commit_flow_coupling_from_current(problem)
                last_interface_diag = _record_step(
                    args=args,
                    problem=problem,
                    outdir=outdir,
                    vtk_dir=vtk_dir,
                    timeseries_rows=timeseries_rows,
                    alpha_diagnostics=alpha_diagnostics,
                    alpha_area0=alpha_area0,
                    alpha_band0=alpha_band0,
                    moving_ref=moving_ref,
                    fixed_ref=fixed_ref,
                    nonlinear_ref=nonlinear_ref,
                    eps_alpha_eff=float(eps_alpha_eff),
                    t_now=float(t_now),
                    dt_now=float(dt_try),
                    step_no=int(step_no),
                    outer_sweeps=int(outer_sweeps),
                    outer_delta_inf=float(outer_delta_inf),
                    outer_delta_abs_inf=float(outer_delta_abs_inf),
                    outer_converged=bool(outer_converged),
                    outer_last_omega=float(outer_last_omega),
                    stage_iters=stage_iters,
                    interface_entry_delta=float(entry_delta_eff),
                )
                break
    except Exception as exc:
        solve_error = str(exc)
        print(f"[warn] staggered solve terminated early: {solve_error}", flush=True)
        _copy_previous_into_current(current_functions, previous_functions)
        _reset_flow_coupling_from_previous(problem)
    solve_seconds = time.perf_counter() - t_start

    alpha_diag_final = alpha_diagnostics.evaluate({problem["alpha_k"].name: problem["alpha_k"]})
    alpha_area_final = float(alpha_diag_final.get("alpha_area", float("nan")))
    alpha_band_final = float(alpha_diag_final.get("alpha_band", float("nan")))
    interface_diag_final = (
        last_interface_diag
        if isinstance(last_interface_diag, dict)
        else mono._compute_interface_probe_diagnostics(
            problem=problem,
            Lx=float(args.Lx),
            y_interface=float(args.y_interface),
            y_profile=float(args.y_profile),
            eps_alpha=float(eps_alpha_eff),
            mu_f=float(args.mu_f),
            mu_s=float(args.mu_s),
            lambda_s=float(args.lambda_s),
            solid_visco_eta=float(args.solid_visco_eta),
            solid_model=str(args.solid_model),
            skeleton_pressure_mode=str(args.skeleton_pressure_mode),
            alpha_biot=(None if args.alpha_biot is None else float(args.alpha_biot)),
            interface_entry_delta=float(entry_delta_eff),
            interface_bjs_gamma=float(getattr(args, "interface_bjs_gamma", 1.0e3)),
        )
    )
    interface_summary_final = dict(interface_diag_final.get("summary", {}) or {})
    profile_x, profile_uy = mono._sample_profile(
        problem=problem,
        Lx=float(args.Lx),
        y_profile=float(args.y_profile),
        n_samples=int(args.profile_samples),
    )
    if io_root:
        mono._write_profile_csv(outdir / "profile_final.csv", x=profile_x, u_y=profile_uy)
        mono._write_timeseries_csv(outdir / "timeseries.csv", timeseries_rows)
        _write_interface_diagnostics(outdir, interface_diag_final, latest=False)
        vtk_final = {
            "v": problem["v_k"],
            "p": problem["p_k"],
            "p_pore": problem["p_pore_k"],
            "vS": problem["vS_k"],
            "u": problem["u_k"],
            "alpha": problem["alpha_k"],
            **({"B": problem["B_k"]} if problem.get("B_k") is not None else {}),
            "mu_alpha": problem["mu_k"],
        }
        if problem.get("S_k") is not None:
            vtk_final["S"] = problem["S_k"]
        if problem.get("reg_weight") is not None:
            vtk_final["reg_weight"] = problem["reg_weight"]
        if problem.get("interface_corner_taper") is not None:
            vtk_final["interface_corner_taper"] = problem["interface_corner_taper"]
        export_vtk(str(outdir / "final_state.vtu"), mesh=problem["mesh"], dof_handler=problem["dh"], functions=vtk_final)

    moving_metrics = (
        mono._compute_profile_metrics(x_num=profile_x, y_num=profile_uy, x_ref=moving_ref[0], y_ref=moving_ref[1])
        if moving_ref is not None
        else None
    )
    fixed_metrics = (
        mono._compute_profile_metrics(x_num=profile_x, y_num=profile_uy, x_ref=fixed_ref[0], y_ref=fixed_ref[1])
        if fixed_ref is not None
        else None
    )

    summary_row: dict[str, object] = {
        "kappa": float(kappa),
        "kappa_inv": float(1.0 / float(kappa)),
        "nx": float(args.nx),
        "ny": float(args.ny),
        "dt": float(args.dt),
        "t_reached": float(t_now),
        "t_final": float(args.t_final),
        "steps_recorded": float(len(timeseries_rows)),
        "solve_seconds": float(solve_seconds),
        "solve_completed": float(0.0 if solve_error else 1.0),
        "solve_error": str(solve_error),
        "backend": str(args.backend),
        "linear_backend": str(args.linear_backend),
        "split_scheme": "_".join(stage_order),
        "outer_it": float(args.outer_it),
        "outer_tol": float(args.outer_tol),
        "outer_abs_tol": float(getattr(args, "outer_abs_tol", 0.0)),
        "outer_update": str(getattr(args, "outer_update", "constant")),
        "outer_relaxation": float(args.outer_relaxation),
        "outer_relaxation_min": float(getattr(args, "outer_relaxation_min", 1.0e-3)),
        "outer_relaxation_max": float(getattr(args, "outer_relaxation_max", 1.0)),
        "outer_timestep_history": int(getattr(args, "outer_timestep_history", 1)),
        "transport_post_accept": float(1.0 if bool(getattr(args, "transport_post_accept", True)) else 0.0),
        "flow_solid_substeps": float(flow_solid_substeps),
        "solid_solver": str(args.solid_solver),
        "transport_solver": str(args.transport_solver),
        "flow_solver": str(args.flow_solver),
        "solid_max_it": float(args.max_it if args.solid_max_it is None else args.solid_max_it),
        "transport_max_it": float(args.max_it if args.transport_max_it is None else args.transport_max_it),
        "flow_max_it": float(args.max_it if args.flow_max_it is None else args.flow_max_it),
        "poly_order": float(poly_order),
        "pressure_order": float(pressure_order),
        "scalar_order": float(scalar_order),
        "fluid_space": str(args.fluid_space),
        "solid_model": str(args.solid_model),
        "solid_model_effective": mono._benchmark7_solid_model_key(getattr(args, "solid_model", "linear")),
        "mu_b_model": str(args.mu_b_model),
        "drag_formulation": str(args.drag_formulation),
        "full_ratio_free_state": float(1.0 if bool(args.full_ratio_free_state) else 0.0),
        "support_physics": str(args.support_physics),
        "solid_bc_mode": str(args.solid_bc_mode),
        "skeleton_pressure_mode": str(args.skeleton_pressure_mode),
        "storativity_c0": float(args.storativity_c0),
        "pressure_interface_closure": float(1.0 if bool(args.pressure_interface_closure) else 0.0),
        "pressure_interface_closure_strength": float(args.pressure_interface_closure_strength),
        "p_pore_fluid_gauge": float(1.0 if bool(getattr(args, "p_pore_fluid_gauge", True)) else 0.0),
        "p_pore_fluid_gauge_strength": float(getattr(args, "p_pore_fluid_gauge_strength", 1.0)),
        "interface_entry_closure": float(1.0 if bool(args.interface_entry_closure) else 0.0),
        "interface_entry_closure_strength": float(args.interface_entry_closure_strength),
        "interface_entry_delta": float(args.interface_entry_delta),
        "interface_entry_delta_effective": float(entry_delta_eff),
        "interface_robin_l": float(interface_robin_l),
        "interface_bjs_closure": float(1.0 if bool(args.interface_bjs_closure) else 0.0),
        "interface_bjs_closure_strength": float(args.interface_bjs_closure_strength),
        "interface_bjs_gamma": float(args.interface_bjs_gamma),
        "gamma_div_input": float(args.gamma_div),
        "gamma_div_effective": float(effective_gamma_div),
        "condition_balanced_auto_gamma_div": float(
            1.0 if bool(getattr(args, "condition_balanced_auto_gamma_div", True)) else 0.0
        ),
        "condition_balanced_solid_cut_fix": float(
            1.0 if bool(getattr(args, "condition_balanced_solid_cut_fix", True)) else 0.0
        ),
        "solid_dof_y_cut": ("none" if solid_dof_y_cut is None else float(solid_dof_y_cut)),
        "inactive_solid_dofs_above_cut": float(sum(int(v) for v in solid_inactive_counts.values())),
        "inactive_solid_dof_counts": json.dumps(solid_inactive_counts, sort_keys=True),
        "alpha_regularization": str(args.alpha_regularization),
        "alpha_advect_with": str(args.alpha_advect_with),
        "alpha_advection_form": str(args.alpha_advection_form),
        "alpha_supg": float(args.alpha_supg),
        "alpha_cip": float(args.alpha_cip),
        "h_char": float(h_char),
        "eps_alpha": float(eps_alpha_eff),
        "eps_alpha_over_h": float(eps_alpha_eff / max(h_char, 1.0e-14)),
        "reg_rect": float(reg_mask_meta["enabled"]),
        "reg_rect_center_x": float(reg_mask_meta["center_x"]),
        "reg_rect_center_y": float(reg_mask_meta["center_y"]),
        "reg_rect_half_width": float(reg_mask_meta["half_width"]),
        "reg_rect_half_height": float(reg_mask_meta["half_height"]),
        "reg_rect_fraction": float(reg_mask_meta["fraction"]),
        "interface_corner_taper_width": float(interface_corner_taper_meta.get("width", float("nan"))),
        "interface_corner_taper_positive_fraction": float(interface_corner_taper_meta.get("positive_fraction", float("nan"))),
        "interface_corner_taper_active_fraction": float(interface_corner_taper_meta.get("active_fraction", float("nan"))),
        "alpha_area0": float(alpha_area0),
        "alpha_area_final": float(alpha_area_final),
        "alpha_area_rel_drift": float((alpha_area_final - alpha_area0) / max(abs(alpha_area0), 1.0e-30)),
        "alpha_band0": float(alpha_band0),
        "alpha_band_final": float(alpha_band_final),
        "alpha_band_rel_drift": float((alpha_band_final - alpha_band0) / max(abs(alpha_band0), 1.0e-30)),
        "alpha_min": float(np.min(problem["alpha_k"].nodal_values)),
        "alpha_max": float(np.max(problem["alpha_k"].nodal_values)),
        "B_min": float(np.min(problem["B_k"].nodal_values)) if problem.get("B_k") is not None else float("nan"),
        "B_max": float(np.max(problem["B_k"].nodal_values)) if problem.get("B_k") is not None else float("nan"),
        "u_y_max": float(np.max(profile_uy)),
        "u_y_min": float(np.min(profile_uy)),
        "u_y_peak_x": float(profile_x[int(np.argmax(profile_uy))]),
    }
    summary_row.update(_interface_summary_to_metrics(interface_summary_final))
    if moving_metrics is not None:
        summary_row.update(
            {
                "rmse_to_moving_linear": float(moving_metrics.rmse),
                "linf_to_moving_linear": float(moving_metrics.linf),
                "rmse_over_amp_moving_linear": float(moving_metrics.rmse_over_amplitude),
                "linf_over_amp_moving_linear": float(moving_metrics.linf_over_amplitude),
                "peak_amp_relerr_moving_linear": float(moving_metrics.peak_amplitude_relative_error),
                "peak_x_error_moving_linear": float(moving_metrics.peak_x_error),
            }
        )
    if fixed_metrics is not None:
        summary_row.update(
            {
                "rmse_to_fixed_linear": float(fixed_metrics.rmse),
                "rmse_over_amp_fixed_linear": float(fixed_metrics.rmse_over_amplitude),
                "closer_to_moving_than_fixed": float(
                    1.0 if (moving_metrics is not None and moving_metrics.rmse <= fixed_metrics.rmse) else 0.0
                ),
            }
        )

    summary_payload = {
        "case": summary_row,
        "moving_linear_metrics": None if moving_metrics is None else moving_metrics.__dict__,
        "fixed_linear_metrics": None if fixed_metrics is None else fixed_metrics.__dict__,
        "reference_csv": str(reference_csv) if reference_csv.exists() else "",
        "profile_csv": str(outdir / "profile_final.csv"),
        "timeseries_csv": str(outdir / "timeseries.csv"),
        "interface_summary_json": str(outdir / "interface_diagnostics_summary.json"),
        "vtk_final": str(outdir / "final_state.vtu"),
    }
    if io_root:
        (outdir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        mono._write_case_plot(
            outdir / "profile_compare.png",
            kappa=float(kappa),
            x_num=profile_x,
            y_num=profile_uy,
            moving_ref=moving_ref,
            fixed_ref=fixed_ref,
            nonlinear_ref=nonlinear_ref,
        )
    mono.barrier(mono.MPI_CTX)
    return mono.CaseResult(
        kappa=float(kappa),
        outdir=outdir,
        summary_row=summary_row,
        profile_x=profile_x,
        profile_uy=profile_uy,
        moving_metrics=moving_metrics,
        fixed_metrics=fixed_metrics,
    )


def _print_run_settings(args: argparse.Namespace, *, kappa: float, case_outdir: Path) -> None:
    stage_order = " -> ".join(_stage_order(args))
    print(f"[run] kappa={float(kappa):.6e} -> {case_outdir}", flush=True)
    print(
        "[run-config] "
        f"formulation={mono._benchmark7_formulation_label(args)}; "
        f"alpha_equation={mono._benchmark7_alpha_equation_label(args)}; "
        f"scheme={stage_order}; "
        f"stage_solvers=(solid={args.solid_solver}, transport={args.transport_solver}, flow={args.flow_solver}); "
        f"transport_post_accept={int(bool(getattr(args, 'transport_post_accept', True)))}; "
        f"flow_solid_substeps={max(1, int(getattr(args, 'flow_solid_substeps', 1)))}; "
        f"outer_it={int(args.outer_it)}; "
        f"outer_tol={float(args.outer_tol):.1e}; "
        f"outer_abs_tol={float(getattr(args, 'outer_abs_tol', 0.0)):.1e}; "
        f"outer_update={str(getattr(args, 'outer_update', 'constant'))}; "
        f"outer_relaxation={float(getattr(args, 'outer_relaxation', 1.0)):.3e}; "
        f"storativity_c0={float(args.storativity_c0):.3e}; "
        f"entry_closure={int(bool(args.interface_entry_closure))}; "
        f"bjs_closure={int(bool(args.interface_bjs_closure))}.",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    prev_cpp_fuse = mono._configure_benchmark7_cpp_fuse_integrals(
        backend=str(args.backend),
        enabled=getattr(args, "cpp_fuse_integrals", None),
    )
    if (
        str(getattr(args, "backend", "")).strip().lower() in {"cpp", "c++"}
        and mono.MPI_CTX.is_root
    ):
        cpp_fuse_value = os.environ.get("PYCUTFEM_CPP_FUSE_INTEGRALS", prev_cpp_fuse)
        if cpp_fuse_value is not None:
            print(
                f"[setup] Benchmark 7 staggered cpp integral fusion: "
                f"PYCUTFEM_CPP_FUSE_INTEGRALS={cpp_fuse_value}",
                flush=True,
            )
    outdir = Path(args.outdir).resolve()
    kappas = mono._parse_float_list(args.kappa_list)
    results: list[mono.CaseResult] = []
    for kappa in kappas:
        case_id = f"kappa_{kappa:.0e}".replace("+0", "").replace("-0", "-")
        case_outdir = outdir / case_id
        _print_run_settings(args, kappa=float(kappa), case_outdir=case_outdir)
        result = _run_case(args, kappa=float(kappa), outdir=case_outdir)
        results.append(result)

    summary_csv = outdir / "benchmark7_summary.csv"
    if mono._mpi_io_root():
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        if results:
            with summary_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0].summary_row.keys()))
                writer.writeheader()
                for result in results:
                    writer.writerow(result.summary_row)
        combined = {
            "cases": [result.summary_row for result in results],
            "reference_csv": str(Path(args.reference_csv).resolve()),
        }
        (outdir / "benchmark7_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
        mono._write_combined_profiles_plot(
            outdir / "benchmark7_seboldt_profiles.png",
            results,
            reference_csv=Path(args.reference_csv).resolve(),
        )
        print(f"[done] wrote {summary_csv}", flush=True)
        print(f"[done] wrote {outdir / 'benchmark7_summary.json'}", flush=True)
        if (outdir / "benchmark7_seboldt_profiles.png").exists():
            print(f"[done] wrote {outdir / 'benchmark7_seboldt_profiles.png'}", flush=True)
    mono.barrier(mono.MPI_CTX)


if __name__ == "__main__":
    main()
