from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
from xml.sax.saxutils import escape as _xml_escape

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.core.topology import Node
from pycutfem.fem import transform
from pycutfem.fem.reference import get_reference
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.linalg import (
    StructuralMeshMotionStrategy,
    StructuralMeshMotionStrategySettings,
    kratos_iqnils_iteration_matrices_cpp,
    kratos_iqnils_next_iterate_cpp,
)
from pycutfem.mor.interface import build_restriction_matrix
from pycutfem.nirb.coupling import NIRBSolidPredictor
from pycutfem.operators import LocalAssemblyResult, RuntimeOperator
from pycutfem.solvers.nonlinear_solver import (
    LinearSolverParameters,
    NewtonParameters,
    NewtonSolver,
    PetscSnesNewtonSolver,
    TimeStepperParameters,
)
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import (
    Constant,
    ElementWiseConstant,
    FacetNormal,
    Function,
    Identity,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    det,
    div,
    dot,
    grad,
    inner,
    inv,
    log,
    trace,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dS, dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.bitset import BitSet
from pycutfem.utils.gmsh_loader import mesh_from_gmsh

from examples.NIRB.common import dump_json
from examples.NIRB.dvms import (
    FluidDVMSState,
    FluidDVMSCondensedLocalSystemOperator,
    FluidDVMSLocalVelocityContributionOperator,
    FluidDVMSSolverOperator,
    assemble_fluid_dvms_local_contribution_batch,
    _advance_fluid_dvms_history_after_step,
    _bossak_coefficients,
    _build_fluid_dvms_state,
    _clear_fluid_dvms_oss_projections,
    _eval_scalar_with_grad,
    _eval_vector_with_grad,
    _fluid_dvms_summary,
    _kratos_dvms_current_element_size_coefficient,
    _update_fluid_dvms_oss_projections,
    _update_fluid_dvms_predicted_subscale,
    _update_fluid_dvms_state_from_previous_step,
)
from examples.NIRB.dvms.local_operator import _compress_batch_to_fluid_block
from examples.NIRB.dvms.symbolics import (
    build_fluid_cauchy_stress,
    build_fluid_dvms_kinematics,
    build_fluid_dvms_predictor_symbolics,
)
from examples.NIRB.double_flap_reference import MDPAMesh, _load_json
from examples.NIRB.example2_local_setup import load_example2_local_setup
from examples.NIRB.example2_problem import (
    _EX2_HALF,
    _EX2_ONE,
    _EX2_TWO,
    _EX2_TWO_THIRDS,
    _named_constant,
    build_conforming_mesh,
)
from examples.utils.poromechanics import (
    UPlMaterial2D,
    build_kratos_quasistatic_upl_system_2d,
)


_EX2L_HALF = _named_constant("example2_local_half", 0.5)
_EX2L_ONE = _named_constant("example2_local_one", 1.0)
_EX2L_TWO = _named_constant("example2_local_two", 2.0)
_EX2L_TWO_THIRDS = _named_constant("example2_local_two_thirds", 2.0 / 3.0)
_EX2L_FOUR = _named_constant("example2_local_four", 4.0)
_EX2L_ZERO = _named_constant("example2_local_zero", 0.0)
_EX2L_CONV_EPS = _named_constant("example2_local_conv_eps", 1.0e-12)
_EX2L_ZERO_VEC = _named_constant("example2_local_zero_vec", np.zeros(2, dtype=float), dim=1)
_CHECKPOINT_SCHEMA_VERSION = 3
# Kratos' Triangle2D3 DVMS/QSVMS elements use the 3-point simplex rule.
# In pycutfem this corresponds to quadrature_order=1 on triangles.
_EX2L_KRATOS_MATCHED_QUAD_ORDER = 1
# Keep the production structural solve on the standard converged Newton path by
# default. A one-step structural accept remains available only as an explicit
# experiment switch while auditing monitored stage semantics.
_EX2L_KRATOS_STRUCT_ONE_STEP_ACCEPT_FACTOR = 1.0e12


def _coord_key(x: float, y: float, ndigits: int = 12) -> tuple[float, float]:
    return (round(float(x), ndigits), round(float(y), ndigits))


def _call_with_supported_keywords(func: Callable[..., object], /, **kwargs):
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return func(**kwargs)
    accepted = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return func(**accepted)


def _kratos_hyperelastic_plane_strain_pk1(F, mu_s: Constant, lambda_s: Constant):
    J = det(F)
    Finv = inv(F)
    logJ = log(J)
    return mu_s * F + (lambda_s * logJ - mu_s) * Finv.T


def _kratos_hyperelastic_plane_strain_delta_pk1(F, grad_dd, mu_s: Constant, lambda_s: Constant):
    J = det(F)
    Finv = inv(F)
    FinvT = Finv.T
    logJ = log(J)
    tr_term = trace(dot(Finv, grad_dd))
    return (
        mu_s * grad_dd
        + lambda_s * tr_term * FinvT
        + (mu_s - lambda_s * logJ) * dot(dot(FinvT, grad_dd.T), FinvT)
    )


def _kratos_structural_mesh_area_coefficient(mesh: Mesh) -> ElementWiseConstant:
    areas = np.maximum(np.abs(np.asarray(mesh.areas_list, dtype=float).reshape(-1)), 1.0e-30)
    return ElementWiseConstant(areas)


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def _section(enabled: bool, title: str) -> None:
    if enabled:
        print("-" * 50, flush=True)
        print(title, flush=True)
        print("-" * 50, flush=True)


def _should_monitor_stress_interface_loads(*, load_transfer: str, monitor_interface_loads: bool) -> bool:
    if str(load_transfer).strip().lower() == "stress":
        return True
    if not bool(monitor_interface_loads):
        return False
    return _env_bool("PYCUTFEM_EX2_MONITOR_STRESS_LOADS", False)


def _configure_kratos_thread_env(*, enable: bool) -> dict[str, object]:
    info: dict[str, object] = {
        "enabled": bool(enable),
        "already_imported": "KratosMultiphysics" in sys.modules,
        "changes": {},
        "effective": {},
        "runtime_openmp_library": "",
        "runtime_openmp_max_threads": None,
    }
    if not bool(enable):
        return info

    omp_threads = _env_str("PYCUTFEM_EX2_KRATOS_OMP_NUM_THREADS", "1").strip() or "1"
    blas_threads = _env_str("PYCUTFEM_EX2_KRATOS_BLAS_NUM_THREADS", "1").strip() or "1"
    target_env = {
        "OMP_NUM_THREADS": omp_threads,
        "OPENBLAS_NUM_THREADS": blas_threads,
        "MKL_NUM_THREADS": blas_threads,
    }

    changes: dict[str, str] = {}
    effective: dict[str, str] = {}
    for name, value in target_env.items():
        current = str(os.getenv(name, "") or "").strip()
        if not current:
            os.environ[name] = str(value)
            changes[name] = str(value)
            current = str(value)
        effective[name] = current

    try:
        import ctypes
        import ctypes.util

        omp_threads_int = max(1, int(float(omp_threads)))
        openmp_lib_names: list[str] = []
        for candidate in ("gomp", "omp", "iomp5"):
            found = ctypes.util.find_library(candidate)
            if found:
                openmp_lib_names.append(str(found))
        openmp_lib_names.extend(("libgomp.so.1", "libomp.so", "libomp.so.5", "libiomp5.so"))

        seen_libs: set[str] = set()
        for lib_name in openmp_lib_names:
            if not lib_name or lib_name in seen_libs:
                continue
            seen_libs.add(lib_name)
            try:
                openmp_lib = ctypes.CDLL(lib_name)
            except OSError:
                continue
            if not hasattr(openmp_lib, "omp_set_num_threads"):
                continue
            try:
                if hasattr(openmp_lib, "omp_set_dynamic"):
                    openmp_lib.omp_set_dynamic.argtypes = [ctypes.c_int]
                    openmp_lib.omp_set_dynamic(0)
                openmp_lib.omp_set_num_threads.argtypes = [ctypes.c_int]
                openmp_lib.omp_set_num_threads(int(omp_threads_int))
                info["runtime_openmp_library"] = str(lib_name)
                if hasattr(openmp_lib, "omp_get_max_threads"):
                    openmp_lib.omp_get_max_threads.restype = ctypes.c_int
                    info["runtime_openmp_max_threads"] = int(openmp_lib.omp_get_max_threads())
                break
            except Exception:
                continue
    except Exception:
        pass

    info["changes"] = changes
    info["effective"] = effective
    return info


def _scalar_field_matrix(dh: DofHandler, function: Function) -> tuple[np.ndarray, np.ndarray]:
    dh._ensure_dof_coords()
    field_ids = np.asarray(dh.get_field_slice(function.field_name), dtype=int)
    coords = np.asarray(dh._dof_coords[field_ids], dtype=float)
    values = np.asarray(function.get_nodal_values(field_ids), dtype=float)
    return coords, values


def _scalar_lookup_from_field(dh: DofHandler, function: Function) -> CoordinateLookup:
    coords, values = _scalar_field_matrix(dh, function)
    return CoordinateLookup(coords, values, dim=1)


def _maybe_dump_exact_fluid_probe(
    *,
    output_dir: Path,
    step: int,
    coupling_iter: int,
    stage_label: str,
    bc_scale: float,
    dt: float,
    bossak_alpha: float,
    fluid: dict[str, object],
    reaction_point_load_lookup: CoordinateLookup | None,
    reaction_solid_load_lookup: CoordinateLookup | None,
) -> None:
    flag = str(os.getenv("PYCUTFEM_EX2_DUMP_FLUID_STAGE", "0") or "0").strip().lower()
    if flag not in {"1", "true", "yes"}:
        return
    target_step_raw = str(os.getenv("PYCUTFEM_EX2_DUMP_FLUID_STEP", "1") or "1").strip().lower()
    target_iter_raw = str(os.getenv("PYCUTFEM_EX2_DUMP_FLUID_ITER", "1") or "1").strip().lower()
    dump_all_steps = target_step_raw in {"all", "*", "any"}
    dump_all_iters = target_iter_raw in {"all", "*", "any"}
    target_step = 1 if dump_all_steps else int(target_step_raw)
    target_iter = 1 if dump_all_iters else int(target_iter_raw)
    if (not dump_all_steps and int(step) != target_step) or (
        not dump_all_iters and int(coupling_iter) != target_iter
    ):
        return

    probe_dir = Path(output_dir) / "debug_fluid_stage"
    probe_dir.mkdir(parents=True, exist_ok=True)
    u_coords, u_values = _vector_field_matrix(fluid["dh"], fluid["u_k"])
    p_coords, p_values = _scalar_field_matrix(fluid["dh"], fluid["p_k"])
    u_prev_source = fluid.get("u_prev", fluid["u_k"])
    p_prev_source = fluid.get("p_prev", fluid["p_k"])
    _, u_prev_values = _vector_field_matrix(fluid["dh"], u_prev_source)
    _, p_prev_values = _scalar_field_matrix(fluid["dh"], p_prev_source)
    d_coords, d_values = _vector_field_matrix(fluid["dh"], fluid["d_mesh"])
    _, d_prev_values = _vector_field_matrix(fluid["dh"], fluid["d_prev"])
    _, d_prev2_values = _vector_field_matrix(fluid["dh"], fluid["d_prev2"])
    _, w_mesh_prev_values = _vector_field_matrix(fluid["dh"], fluid["w_mesh_prev"])
    _, a_mesh_prev_values = _vector_field_matrix(fluid["dh"], fluid["a_mesh_prev"])
    _, a_prev_values = _vector_field_matrix(fluid["dh"], fluid["a_prev"])
    if "a_k" in fluid and fluid["a_k"] is not None:
        _, a_k_values = _vector_field_matrix(fluid["dh"], fluid["a_k"])
    else:
        a_k_values = np.asarray(a_prev_values, dtype=float)
    if "w_mesh_k" in fluid and "a_mesh_k" in fluid:
        _, w_mesh_values = _vector_field_matrix(fluid["dh"], fluid["w_mesh_k"])
        _, a_mesh_values = _vector_field_matrix(fluid["dh"], fluid["a_mesh_k"])
    else:
        w_mesh_values, a_mesh_values = _bossak_displacement_kinematics_values(
            d_curr=d_values,
            d_prev=d_prev_values,
            v_prev=w_mesh_prev_values,
            a_prev=a_mesh_prev_values,
            dt=dt,
            alpha=bossak_alpha,
        )
    payload = {
        "step": np.asarray(int(step), dtype=int),
        "coupling_iter": np.asarray(int(coupling_iter), dtype=int),
        "bc_scale": np.asarray(float(bc_scale), dtype=float),
        "u_coords": np.asarray(u_coords, dtype=float),
        "u_values": np.asarray(u_values, dtype=float),
        "p_coords": np.asarray(p_coords, dtype=float),
        "p_values": np.asarray(p_values, dtype=float),
        "u_prev_values": np.asarray(u_prev_values, dtype=float),
        "p_prev_values": np.asarray(p_prev_values, dtype=float),
        "d_coords": np.asarray(d_coords, dtype=float),
        "d_values": np.asarray(d_values, dtype=float),
        "d_prev_values": np.asarray(d_prev_values, dtype=float),
        "d_prev2_values": np.asarray(d_prev2_values, dtype=float),
        "w_mesh_prev_values": np.asarray(w_mesh_prev_values, dtype=float),
        "a_mesh_prev_values": np.asarray(a_mesh_prev_values, dtype=float),
        "a_prev_values": np.asarray(a_prev_values, dtype=float),
        "a_k_values": np.asarray(a_k_values, dtype=float),
        "w_mesh_values": np.asarray(w_mesh_values, dtype=float),
        "a_mesh_values": np.asarray(a_mesh_values, dtype=float),
    }
    dvms_state = fluid.get("dvms_state")
    if isinstance(dvms_state, FluidDVMSState):
        payload["dvms_sample_coords"] = np.asarray(dvms_state.sample_coords, dtype=float)
        payload["dvms_sample_element_ids"] = np.asarray(dvms_state.sample_element_ids, dtype=int)
        payload["dvms_old_subscale_velocity"] = np.asarray(dvms_state.old_subscale_velocity, dtype=float)
        payload["dvms_predicted_subscale_velocity"] = np.asarray(dvms_state.predicted_subscale_velocity, dtype=float)
        payload["dvms_momentum_projection"] = np.asarray(dvms_state.momentum_projection, dtype=float)
        payload["dvms_mass_projection"] = np.asarray(dvms_state.mass_projection, dtype=float)
        payload["dvms_old_mass_residual"] = np.asarray(dvms_state.old_mass_residual, dtype=float)
        nodal_momentum_projection = getattr(dvms_state, "_nodal_momentum_projection", None)
        nodal_div_projection = getattr(dvms_state, "_nodal_div_projection", None)
        prev_nodal_div_projection = getattr(dvms_state, "_prev_nodal_div_projection", None)
        if nodal_momentum_projection is not None or nodal_div_projection is not None or prev_nodal_div_projection is not None:
            dh = fluid["dh"]
            try:
                dh._ensure_dof_coords()
                ux_ids = np.asarray(dh.get_field_slice("ux"), dtype=int).reshape(-1)
                payload["dvms_projection_coords"] = np.asarray(dh._dof_coords[ux_ids], dtype=float)
            except Exception:
                pass
        if nodal_momentum_projection is not None:
            payload["dvms_nodal_momentum_projection"] = np.asarray(nodal_momentum_projection, dtype=float)
        if nodal_div_projection is not None:
            payload["dvms_nodal_div_projection"] = np.asarray(nodal_div_projection, dtype=float)
        if prev_nodal_div_projection is not None:
            payload["dvms_prev_nodal_div_projection"] = np.asarray(prev_nodal_div_projection, dtype=float)
    if reaction_point_load_lookup is not None:
        payload["reaction_point_coords"] = np.asarray(reaction_point_load_lookup.coords, dtype=float)
        payload["reaction_point_values"] = np.asarray(reaction_point_load_lookup.values, dtype=float)
    if reaction_solid_load_lookup is not None:
        payload["reaction_solid_coords"] = np.asarray(reaction_solid_load_lookup.coords, dtype=float)
        payload["reaction_solid_values"] = np.asarray(reaction_solid_load_lookup.values, dtype=float)
    np.savez(probe_dir / f"step{int(step):04d}_iter{int(coupling_iter):04d}_{str(stage_label)}.npz", **payload)
    abort_stage = str(os.getenv("PYCUTFEM_EX2_DUMP_FLUID_ABORT_AFTER_STAGE", "") or "").strip()
    if abort_stage and abort_stage == str(stage_label):
        raise RuntimeError(f"Aborting after requested fluid debug probe stage {stage_label!r}.")


def _maybe_dump_structure_stage_probe(
    *,
    output_dir: Path,
    step: int,
    coupling_iter: int,
    stage_label: str,
    dt: float,
    solid: dict[str, object],
    interface_tag: str,
    structure_load_lookup: CoordinateLookup | None,
    point_load_full: np.ndarray | None,
) -> None:
    flag = str(os.getenv("PYCUTFEM_EX2_DUMP_STRUCTURE_STAGE", "0") or "0").strip().lower()
    if flag not in {"1", "true", "yes"}:
        return
    target_step = int(os.getenv("PYCUTFEM_EX2_DUMP_STRUCTURE_STEP", "1") or "1")
    target_iter = int(os.getenv("PYCUTFEM_EX2_DUMP_STRUCTURE_ITER", "1") or "1")
    if int(step) != target_step or int(coupling_iter) != target_iter:
        return

    probe_dir = Path(output_dir) / "debug_structure_stage"
    probe_dir.mkdir(parents=True, exist_ok=True)
    coords_ref = np.asarray(solid["mesh"].nodes_x_y_pos, dtype=float)
    disp_values = _vector_point_data_from_function(solid["dh"], solid["d_k"])
    disp_prev_values = _vector_point_data_from_function(solid["dh"], solid["d_prev"])
    iface_coords_ref, iface_disp_values = _boundary_vector_snapshot(
        solid["dh"],
        solid["d_k"],
        str(interface_tag),
    )
    dt_value = max(float(dt), 1.0e-14)
    vel_values = (disp_values - disp_prev_values) / dt_value
    payload: dict[str, np.ndarray] = {
        "step": np.asarray(int(step), dtype=int),
        "coupling_iter": np.asarray(int(coupling_iter), dtype=int),
        "node_ids": _mesh_node_ids(solid["mesh"]),
        "coords_ref": coords_ref,
        "coords_cur": coords_ref + disp_values,
        "displacement_nodal_values": np.asarray(disp_values, dtype=float),
        "displacement_prev_nodal_values": np.asarray(disp_prev_values, dtype=float),
        "displacement_raw_values": np.asarray(solid["d_k"].nodal_values, dtype=float).copy(),
        "displacement_prev_raw_values": np.asarray(solid["d_prev"].nodal_values, dtype=float).copy(),
        "velocity_nodal_values": np.asarray(vel_values, dtype=float),
        "interface_disp_coords_ref": np.asarray(iface_coords_ref, dtype=float),
        "interface_disp_values": np.asarray(iface_disp_values, dtype=float),
    }
    if point_load_full is not None:
        payload["point_load_nodal_values"] = _nodal_vector_point_data_from_global_values(
            solid["dh"],
            vector=solid["d_k"],
            global_values=np.asarray(point_load_full, dtype=float),
        )
    if structure_load_lookup is not None:
        payload["structure_load_coords_ref"] = np.asarray(structure_load_lookup.coords, dtype=float)
        payload["structure_load_values"] = np.asarray(structure_load_lookup.values, dtype=float)
    np.savez(probe_dir / f"step{int(step):04d}_iter{int(coupling_iter):04d}_{str(stage_label)}.npz", **payload)


def _maybe_dump_coupling_update_probe(
    *,
    output_dir: Path,
    step: int,
    coupling_iter: int,
    stage_label: str,
    current_load_lookup: CoordinateLookup,
    returned_accel_load_lookup: CoordinateLookup,
    next_load_lookup: CoordinateLookup | None,
    load_guess_history: list[np.ndarray],
    load_return_history: list[np.ndarray],
    iqn_old_dr_mats: list[np.ndarray],
    iqn_old_dg_mats: list[np.ndarray],
    omega_force: float,
    active_force_update: str,
    force_iteration_horizon: int,
    force_regularization: float,
    accel_backend: str,
    v_new: np.ndarray | None,
    w_new: np.ndarray | None,
) -> None:
    flag = str(os.getenv("PYCUTFEM_EX2_DUMP_COUPLING_STATE", "0") or "0").strip().lower()
    if flag not in {"1", "true", "yes"}:
        return
    target_step = int(os.getenv("PYCUTFEM_EX2_DUMP_COUPLING_STEP", "1") or "1")
    target_iter = int(os.getenv("PYCUTFEM_EX2_DUMP_COUPLING_ITER", "1") or "1")
    if int(step) != target_step or int(coupling_iter) != target_iter:
        return

    probe_dir = Path(output_dir) / "debug_coupling_update"
    probe_dir.mkdir(parents=True, exist_ok=True)
    current_values = np.asarray(current_load_lookup.values, dtype=float)
    returned_values = np.asarray(returned_accel_load_lookup.values, dtype=float)
    payload: dict[str, np.ndarray] = {
        "step": np.asarray(int(step), dtype=int),
        "coupling_iter": np.asarray(int(coupling_iter), dtype=int),
        "omega_force": np.asarray(float(omega_force), dtype=float),
        "force_iteration_horizon": np.asarray(int(force_iteration_horizon), dtype=int),
        "force_regularization": np.asarray(float(force_regularization), dtype=float),
        "current_load_coords_ref": np.asarray(current_load_lookup.coords, dtype=float),
        "current_load_values": np.asarray(current_values, dtype=float),
        "returned_load_coords_ref": np.asarray(returned_accel_load_lookup.coords, dtype=float),
        "returned_load_values": np.asarray(returned_values, dtype=float),
        "returned_load_residual": np.asarray(returned_values - current_values, dtype=float),
        "load_guess_history_count": np.asarray(int(len(load_guess_history)), dtype=int),
        "load_return_history_count": np.asarray(int(len(load_return_history)), dtype=int),
        "iqn_old_dr_count": np.asarray(int(len(iqn_old_dr_mats)), dtype=int),
        "iqn_old_dg_count": np.asarray(int(len(iqn_old_dg_mats)), dtype=int),
        "active_force_update_ascii": np.asarray([ord(ch) for ch in str(active_force_update)], dtype=np.int64),
        "accel_backend_ascii": np.asarray([ord(ch) for ch in str(accel_backend)], dtype=np.int64),
    }
    if next_load_lookup is not None:
        payload["next_load_coords_ref"] = np.asarray(next_load_lookup.coords, dtype=float)
        payload["next_load_values"] = np.asarray(next_load_lookup.values, dtype=float)
    if v_new is not None:
        payload["v_new"] = np.asarray(v_new, dtype=float)
    if w_new is not None:
        payload["w_new"] = np.asarray(w_new, dtype=float)
    for idx, values in enumerate(list(load_guess_history)):
        payload[f"load_guess_history_{idx}"] = np.asarray(values, dtype=float)
    for idx, values in enumerate(list(load_return_history)):
        payload[f"load_return_history_{idx}"] = np.asarray(values, dtype=float)
    for idx, mat in enumerate(list(iqn_old_dr_mats)):
        payload[f"iqn_old_dr_{idx}"] = np.asarray(mat, dtype=float)
    for idx, mat in enumerate(list(iqn_old_dg_mats)):
        payload[f"iqn_old_dg_{idx}"] = np.asarray(mat, dtype=float)
    np.savez(probe_dir / f"step{int(step):04d}_iter{int(coupling_iter):04d}_{str(stage_label)}.npz", **payload)


@dataclass
class CoordinateLookup:
    coords: np.ndarray
    values: np.ndarray
    dim: int

    def __post_init__(self) -> None:
        coords = np.asarray(self.coords, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must have shape (n, 2)")
        if self.dim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[0] != coords.shape[0] or values.shape[1] != int(self.dim):
            raise ValueError("values shape does not match coords / dim")
        self.coords = coords
        self.values = values
        self._dict = {_coord_key(x, y): values[i].copy() for i, (x, y) in enumerate(coords)}

    def __call__(self, x: float | np.ndarray, y: float | np.ndarray):
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        flat = np.c_[xa.reshape(-1), ya.reshape(-1)]
        out = np.empty((flat.shape[0], self.dim), dtype=float)
        for i, (xx, yy) in enumerate(flat):
            hit = self._dict.get(_coord_key(xx, yy))
            if hit is not None:
                out[i, :] = hit
                continue
            dist2 = np.sum((self.coords - np.asarray([xx, yy], dtype=float)) ** 2, axis=1)
            out[i, :] = self.values[int(np.argmin(dist2)), :]
        if self.dim == 1:
            return out[:, 0].reshape(xa.shape)
        return out.reshape(xa.shape + (self.dim,))

    def component(self, idx: int) -> Callable[[float, float], float]:
        if not (0 <= int(idx) < self.dim):
            raise IndexError(idx)

        def _wrapped(x, y):
            val = np.asarray(self(x, y), dtype=float)
            if val.ndim == 0:
                return float(val.reshape(()))
            return float(val.reshape(-1, self.dim)[0, int(idx)])

        return _wrapped


class _FluidDVMSSolverOperator(FluidDVMSSolverOperator):
    """Example-side wrapper for Kratos-matched predicted-subscale staging."""

    def __init__(
        self,
        *,
        refresh_on_initial_assembly: bool = True,
        reset_predicted_to_old_on_step_begin: bool = False,
        **kwargs,
    ) -> None:
        self.refresh_on_initial_assembly = bool(refresh_on_initial_assembly)
        self.reset_predicted_to_old_on_step_begin = bool(reset_predicted_to_old_on_step_begin)
        self._skip_refresh_once = not self.refresh_on_initial_assembly
        self._reset_predicted_on_step_begin_once = bool(self.reset_predicted_to_old_on_step_begin)
        self._first_assembly_probe: dict[str, object] | None = None
        self._first_assembly_probe_dumped = False
        self._probe_dumped_stage_labels: set[str] = set()
        super().__init__(**kwargs)

    def arm_initial_old_subscale_build(self) -> None:
        """Use old-step subscale data on the next solver step-begin only."""
        self._skip_refresh_once = True
        self._reset_predicted_on_step_begin_once = True

    def preserve_initial_predicted_subscale(self) -> None:
        """Keep the current predicted subscale for the next first assembly only."""
        self._skip_refresh_once = True
        self._reset_predicted_on_step_begin_once = False

    def configure_first_assembly_probe(
        self,
        *,
        output_dir: Path,
        step: int,
        coupling_iter: int,
        bc_scale: float,
        dt: float,
        bossak_alpha: float,
    ) -> None:
        self._first_assembly_probe = {
            "output_dir": Path(output_dir),
            "step": int(step),
            "coupling_iter": int(coupling_iter),
            "bc_scale": float(bc_scale),
            "dt": float(dt),
            "bossak_alpha": float(bossak_alpha),
        }
        self._first_assembly_probe_dumped = False
        self._probe_dumped_stage_labels = set()

    def _dump_first_assembly_probe(self, *, stage_label: str) -> None:
        if self._first_assembly_probe is None:
            return
        stage_key = str(stage_label)
        if stage_key in self._probe_dumped_stage_labels:
            return
        probe = self._first_assembly_probe
        _maybe_dump_exact_fluid_probe(
            output_dir=Path(probe["output_dir"]),
            step=int(probe["step"]),
            coupling_iter=int(probe["coupling_iter"]),
            stage_label=str(stage_label),
            bc_scale=float(probe["bc_scale"]),
            dt=float(probe["dt"]),
            bossak_alpha=float(probe["bossak_alpha"]),
            fluid={
                "dh": self.dh,
                "u_k": self.u_k,
                "u_prev": self.u_prev,
                "p_k": self.p_k,
                "d_mesh": self.d_mesh,
                "d_prev": self.d_prev,
                "d_prev2": self.d_prev2,
                "w_mesh_prev": self.mesh_v_prev,
                "a_mesh_prev": self.mesh_a_prev,
                "a_prev": self.a_prev,
                "a_k": self.a_curr,
                "w_mesh_k": self.mesh_v,
                "dvms_state": self.state,
            },
            reaction_point_load_lookup=None,
            reaction_solid_load_lookup=None,
        )
        self._probe_dumped_stage_labels.add(stage_key)
        if stage_key in {"first_assembly_skip_refresh", "first_assembly_after_refresh"}:
            self._first_assembly_probe_dumped = True

    def on_step_begin(
        self,
        *,
        solver,
        functions,
        prev_functions,
        aux_functions,
        step: int,
        step_no: int,
        t: float,
        dt: float,
        bcs,
    ) -> None:
        del solver, functions, prev_functions, aux_functions, step, step_no, t, dt, bcs
        if self._reset_predicted_on_step_begin_once and int(self.state.sample_count) > 0:
            self.state.predicted_subscale_velocity[:, :] = np.asarray(
                self.state.old_subscale_velocity,
                dtype=float,
            )
            self.state.sync_coefficient("predicted_subscale_velocity")
        self._reset_predicted_on_step_begin_once = False

    def before_assembly(self, *, solver, coeffs, need_matrix: bool) -> None:
        if self._skip_refresh_once:
            self._dump_first_assembly_probe(stage_label="before_predictor_refresh")
            self._dump_first_assembly_probe(stage_label="first_assembly_skip_refresh")
            self._skip_refresh_once = False
            return
        if bool(need_matrix):
            self._dump_first_assembly_probe(stage_label="before_predictor_refresh")
        super().before_assembly(solver=solver, coeffs=coeffs, need_matrix=need_matrix)
        if bool(need_matrix):
            self._dump_first_assembly_probe(stage_label="after_predictor_refresh")
        self._dump_first_assembly_probe(stage_label="first_assembly_after_refresh")

    def after_nonlinear_update(self, *, solver, functions) -> None:
        super().after_nonlinear_update(solver=solver, functions=functions)
        self._dump_first_assembly_probe(stage_label="after_oss_projection_update")


class _FluidBossakAccelerationOperator(RuntimeOperator):
    """Seed Kratos-style prefluid acceleration, then update with accepted-step Bossak history."""

    def __init__(
        self,
        *,
        u_k: VectorFunction,
        a_k: VectorFunction,
        dt: float,
        bossak_alpha: float,
    ) -> None:
        self.u_k = u_k
        self.a_k = a_k
        coeffs = _bossak_coefficients(alpha=float(bossak_alpha), dt=max(float(dt), 1.0e-14))
        self.ma0 = float(coeffs["ma0"])
        self.ma2 = float(coeffs["ma2"])
        self._accepted_u_prev_snapshot: np.ndarray | None = None
        self._accepted_a_prev_snapshot: np.ndarray | None = None
        self._preserve_seed_on_first_assembly = True
        self._first_assembly_pending = True

    def prime_stage_state(
        self,
        *,
        u_prev_snapshot: np.ndarray,
        a_prev_snapshot: np.ndarray,
        preserve_seed_on_first_assembly: bool = True,
    ) -> None:
        self._accepted_u_prev_snapshot = np.asarray(u_prev_snapshot, dtype=float).copy()
        self._accepted_a_prev_snapshot = np.asarray(a_prev_snapshot, dtype=float).copy()
        self._preserve_seed_on_first_assembly = bool(preserve_seed_on_first_assembly)
        self._first_assembly_pending = True

    def update_current_acceleration(self) -> None:
        if self._accepted_u_prev_snapshot is None or self._accepted_a_prev_snapshot is None:
            self._accepted_u_prev_snapshot = np.zeros_like(np.asarray(self.u_k.nodal_values, dtype=float))
            self._accepted_a_prev_snapshot = np.zeros_like(np.asarray(self.a_k.nodal_values, dtype=float))
        self.a_k.nodal_values[:] = (
            float(self.ma0)
            * (np.asarray(self.u_k.nodal_values, dtype=float) - np.asarray(self._accepted_u_prev_snapshot, dtype=float))
            + float(self.ma2) * np.asarray(self._accepted_a_prev_snapshot, dtype=float)
        )

    def on_step_begin(
        self,
        *,
        solver,
        functions,
        prev_functions,
        aux_functions,
        step: int,
        step_no: int,
        t: float,
        dt: float,
        bcs,
    ) -> None:
        del solver, functions, prev_functions, aux_functions, step, step_no, t, dt, bcs
        if self._accepted_u_prev_snapshot is None:
            self._accepted_u_prev_snapshot = np.zeros_like(np.asarray(self.u_k.nodal_values, dtype=float))
        if self._accepted_a_prev_snapshot is None:
            self._accepted_a_prev_snapshot = np.zeros_like(np.asarray(self.a_k.nodal_values, dtype=float))

    def before_assembly(self, *, solver, coeffs, need_matrix: bool) -> None:
        del solver, coeffs, need_matrix
        if self._first_assembly_pending and bool(self._preserve_seed_on_first_assembly):
            self._first_assembly_pending = False
            return
        self._first_assembly_pending = False
        self.update_current_acceleration()

    def after_nonlinear_update(self, *, solver, functions) -> None:
        del solver, functions
        self.update_current_acceleration()


class _ReducedResidualShiftOperator(RuntimeOperator):
    """Runtime operator that injects a fixed reduced-space residual shift."""

    def __init__(self, shift) -> None:
        self.shift = np.asarray(shift, dtype=float).reshape(-1)

    def after_assembly(self, *, solver, coeffs, A_red, R_red, need_matrix: bool):
        del solver, coeffs, need_matrix
        residual = np.asarray(R_red, dtype=float).reshape(-1)
        if residual.shape != self.shift.shape:
            raise ValueError(
                f"Reduced residual shift shape mismatch: expected {self.shift.shape}, got {residual.shape}."
            )
        return A_red, residual - self.shift


def _fluid_cauchy_stress_ufl(*, p, grad_u_phys, div_u_phys, mu_const):
    identity = Identity(2)
    strain_rate = _EX2L_HALF * (grad_u_phys + grad_u_phys.T)
    viscous = _EX2_TWO * mu_const * (strain_rate - (_EX2L_HALF * _EX2_TWO_THIRDS) * div_u_phys * identity)
    return -p * identity + viscous


def _fluid_cauchy_stress_numpy(*, p_val: float, grad_u_phys: np.ndarray, div_u_phys: float, mu_f: float) -> np.ndarray:
    identity = np.eye(2, dtype=float)
    grad_u_arr = np.asarray(grad_u_phys, dtype=float).reshape(2, 2)
    div_val = float(div_u_phys)
    strain_rate = 0.5 * (grad_u_arr + grad_u_arr.T)
    viscous = 2.0 * float(mu_f) * (strain_rate - (1.0 / 3.0) * div_val * identity)
    return -float(p_val) * identity + viscous


def _resample_lookup_to_coords(source_lookup: CoordinateLookup, target_coords: np.ndarray) -> CoordinateLookup:
    coords = np.asarray(target_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("target_coords must have shape (n, 2)")
    if coords.shape[0] == 0:
        return CoordinateLookup(coords, np.zeros((0, int(source_lookup.dim)), dtype=float), dim=int(source_lookup.dim))
    values = np.asarray(source_lookup(coords[:, 0], coords[:, 1]), dtype=float).reshape(coords.shape[0], int(source_lookup.dim))
    return CoordinateLookup(coords, values, dim=int(source_lookup.dim))


def _negate_lookup(source_lookup: CoordinateLookup) -> CoordinateLookup:
    return CoordinateLookup(
        np.asarray(source_lookup.coords, dtype=float),
        -np.asarray(source_lookup.values, dtype=float),
        dim=int(source_lookup.dim),
    )


def _scaled_lookup(source_lookup: CoordinateLookup, scale: float) -> CoordinateLookup:
    return CoordinateLookup(
        np.asarray(source_lookup.coords, dtype=float),
        float(scale) * np.asarray(source_lookup.values, dtype=float),
        dim=int(source_lookup.dim),
    )


def _solve_linear(
    *,
    eq: Equation,
    dh: DofHandler,
    bcs: list[BoundaryCondition],
    quad_order: int,
    backend: str,
    linear_backend: str,
    functions: list[Function | VectorFunction],
) -> None:
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    K, F = assemble_form(eq, dof_handler=dh, bcs=bcs, quad_order=quad_order, backend=backend)
    lin_backend = str(linear_backend).lower()
    if lin_backend == "petsc":
        try:
            from petsc4py import PETSc  # type: ignore
        except Exception as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("petsc4py is required for linear_backend='petsc'.") from exc

        A = K.tocsr() if hasattr(K, "tocsr") else sp.csr_matrix(np.asarray(K, dtype=float))
        rhs = np.asarray(F, dtype=float).ravel()
        mat = PETSc.Mat().createAIJ(
            size=A.shape,
            csr=(
                A.indptr.astype(PETSc.IntType, copy=False),
                A.indices.astype(PETSc.IntType, copy=False),
                np.asarray(A.data, dtype=float),
            ),
            comm=PETSc.COMM_SELF,
        )
        mat.assemblyBegin()
        mat.assemblyEnd()
        b = PETSc.Vec().createSeq(A.shape[0], comm=PETSc.COMM_SELF)
        x = PETSc.Vec().createSeq(A.shape[0], comm=PETSc.COMM_SELF)
        idx = np.arange(A.shape[0], dtype=PETSc.IntType)
        b.setValues(idx, rhs, addv=PETSc.InsertMode.INSERT_VALUES)
        b.assemblyBegin()
        b.assemblyEnd()
        ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp.setOperators(mat)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        ksp.solve(b, x)
        sol = x.getArray(readonly=True).copy()
        if not np.all(np.isfinite(sol)):
            sol = spsolve(A, rhs)
    elif hasattr(K, "tocsr"):
        sol = spsolve(K.tocsr(), F)
    else:
        sol = np.linalg.solve(np.asarray(K, dtype=float), np.asarray(F, dtype=float))
    for function in functions:
        function.nodal_values.fill(0.0)
    dh.add_to_functions(np.asarray(sol, dtype=float), functions)
    dh.apply_bcs(bcs, *functions)


def _solve_sparse_linear_system(
    *,
    A,
    rhs: np.ndarray,
    linear_backend: str,
    solve_mode: str = "direct",
    initial_guess: np.ndarray | None = None,
) -> np.ndarray:
    import scipy.sparse as sp
    from scipy.sparse.linalg import bicgstab, spsolve

    rhs_arr = np.asarray(rhs, dtype=float).reshape(-1)
    if rhs_arr.size == 0:
        return rhs_arr.copy()

    A_csr = A.tocsr() if hasattr(A, "tocsr") else sp.csr_matrix(np.asarray(A, dtype=float))
    mode = str(solve_mode or "direct").strip().lower()
    backend_name = str(linear_backend or "scipy").strip().lower()
    guess_arr = None if initial_guess is None else np.asarray(initial_guess, dtype=float).reshape(-1).copy()
    if guess_arr is not None and guess_arr.size != rhs_arr.size:
        raise ValueError(
            "initial_guess size does not match the linear-system RHS: "
            f"expected {rhs_arr.size}, got {guess_arr.size}."
        )

    if mode in {"amgcl", "pycutfem_amgcl", "cpp_amgcl", "local_amgcl"}:
        from pycutfem.linalg import AMGCLSettings, solve_sparse_amgcl

        rel_tol = float(_env_float("PYCUTFEM_EX2_MESH_LINEAR_RTOL", 1.0e-7))
        max_it = int(_env_float("PYCUTFEM_EX2_MESH_LINEAR_MAX_IT", 200))
        gmres_dim = int(_env_float("PYCUTFEM_EX2_MESH_LINEAR_GMRES_DIM", 100))
        block_size_raw = _env_str("PYCUTFEM_EX2_MESH_LINEAR_BLOCK_SIZE", "1").strip().lower()
        if block_size_raw == "auto":
            block_size_value = 1
        else:
            block_size_value = max(1, int(float(block_size_raw)))
        coarse_enough = int(_env_float("PYCUTFEM_EX2_MESH_LINEAR_COARSE_ENOUGH", 5000))

        sol, report = solve_sparse_amgcl(
            A_csr,
            rhs_arr,
            x0=guess_arr,
            params=AMGCLSettings(
                preconditioner_type="amg",
                smoother_type="ilu0",
                krylov_type="gmres",
                coarsening_type="aggregation",
                tolerance=float(rel_tol),
                max_iteration=int(max_it),
                gmres_krylov_space_dimension=int(gmres_dim),
                verbosity=0,
                scaling=False,
                block_size=int(block_size_value),
                use_block_matrices_if_possible=True,
                coarse_enough=int(coarse_enough),
            ),
        )
        sol = np.asarray(sol, dtype=float).reshape(-1)
        if not bool(report.converged) or not np.all(np.isfinite(sol)):
            raise RuntimeError(
                "mesh-extension AMGCL solve did not converge "
                f"(iterations={int(report.iterations)}, residual={float(report.residual_norm):.3e}, "
                f"backend={backend_name}, mode={mode})."
            )
        return sol

    if mode in {"kratos_amgcl", "kratos_like"}:
        import KratosMultiphysics as KM
        from KratosMultiphysics.python_linear_solver_factory import ConstructSolver

        rel_tol = float(_env_float("PYCUTFEM_EX2_MESH_LINEAR_RTOL", 1.0e-7))
        max_it = int(_env_float("PYCUTFEM_EX2_MESH_LINEAR_MAX_IT", 200))
        gmres_dim = int(_env_float("PYCUTFEM_EX2_MESH_LINEAR_GMRES_DIM", 100))
        block_size_raw = _env_str("PYCUTFEM_EX2_MESH_LINEAR_BLOCK_SIZE", "1").strip().lower()
        coarse_enough = int(_env_float("PYCUTFEM_EX2_MESH_LINEAR_COARSE_ENOUGH", 5000))
        if block_size_raw == "auto":
            block_size_value: int | str = "auto"
        else:
            block_size_value = max(1, int(float(block_size_raw)))

        A_kratos = KM.CompressedMatrix(int(A_csr.shape[0]), int(A_csr.shape[1]))
        A_coo = A_csr.tocoo()
        for row, col, value in zip(
            A_coo.row.tolist(),
            A_coo.col.tolist(),
            A_coo.data.tolist(),
            strict=False,
        ):
            A_kratos[int(row), int(col)] = float(value)

        b_kratos = KM.Vector(int(rhs_arr.size))
        x_kratos = KM.Vector(int(rhs_arr.size))
        for idx, value in enumerate(rhs_arr.tolist()):
            b_kratos[int(idx)] = float(value)
            x_kratos[int(idx)] = 0.0 if guess_arr is None else float(guess_arr[int(idx)])

        settings = {
            "solver_type": "amgcl",
            "krylov_type": "gmres",
            "smoother_type": "ilu0",
            "coarsening_type": "aggregation",
            "tolerance": float(rel_tol),
            "max_iteration": int(max_it),
            "gmres_krylov_space_dimension": int(gmres_dim),
            "verbosity": 0,
            "provide_coordinates": False,
            "scaling": False,
            "use_block_matrices_if_possible": True,
            "coarse_enough": int(coarse_enough),
        }
        if block_size_value != "auto":
            settings["block_size"] = int(block_size_value)
        solver = ConstructSolver(KM.Parameters(json.dumps(settings)))
        solver.Solve(A_kratos, x_kratos, b_kratos)
        return np.asarray([float(x_kratos[i]) for i in range(int(rhs_arr.size))], dtype=float)

    if mode in {"petsc_gmres_ilu", "petsc_gmres"}:
        try:
            from petsc4py import PETSc  # type: ignore
        except Exception as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "petsc4py is required for mesh solve mode "
                f"{mode!r}."
            ) from exc

        rel_tol = float(_env_float("PYCUTFEM_EX2_MESH_LINEAR_RTOL", 1.0e-7))
        abs_tol = float(_env_float("PYCUTFEM_EX2_MESH_LINEAR_ATOL", 0.0))
        max_it = int(_env_float("PYCUTFEM_EX2_MESH_LINEAR_MAX_IT", 200))
        gmres_dim = int(_env_float("PYCUTFEM_EX2_MESH_LINEAR_GMRES_DIM", 100))

        mat = PETSc.Mat().createAIJ(
            size=A_csr.shape,
            csr=(
                A_csr.indptr.astype(PETSc.IntType, copy=False),
                A_csr.indices.astype(PETSc.IntType, copy=False),
                np.asarray(A_csr.data, dtype=float),
            ),
            comm=PETSc.COMM_SELF,
        )
        mat.assemblyBegin()
        mat.assemblyEnd()

        b = PETSc.Vec().createSeq(A_csr.shape[0], comm=PETSc.COMM_SELF)
        x = PETSc.Vec().createSeq(A_csr.shape[0], comm=PETSc.COMM_SELF)
        idx = np.arange(A_csr.shape[0], dtype=PETSc.IntType)
        b.setValues(idx, rhs_arr, addv=PETSc.InsertMode.INSERT_VALUES)
        b.assemblyBegin()
        b.assemblyEnd()
        if guess_arr is not None:
            x.setValues(idx, guess_arr, addv=PETSc.InsertMode.INSERT_VALUES)
            x.assemblyBegin()
            x.assemblyEnd()

        ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp.setOperators(mat)
        ksp.setType("gmres")
        ksp.setTolerances(rtol=rel_tol, atol=abs_tol, max_it=max_it)
        ksp.setGMRESRestart(gmres_dim)
        pc = ksp.getPC()
        pc.setType("ilu")
        try:
            pc.setFactorLevels(0)
        except Exception:
            pass
        if guess_arr is not None:
            ksp.setInitialGuessNonzero(True)
        ksp.setFromOptions()
        ksp.solve(b, x)
        reason = int(ksp.getConvergedReason())
        sol = x.getArray(readonly=True).copy()
        if reason < 0 or not np.all(np.isfinite(sol)):
            raise RuntimeError(
                "mesh-extension PETSc GMRES+ILU solve did not converge "
                f"(reason={reason}, backend={backend_name}, mode={mode})."
            )
        return np.asarray(sol, dtype=float).reshape(-1)

    if mode in {"iterative", "bicgstab_ilu"}:
        from pycutfem.linalg.preconditioners import build_subsolver

        rel_tol = float(_env_float("PYCUTFEM_EX2_MESH_LINEAR_RTOL", 1.0e-7))
        max_it = int(_env_float("PYCUTFEM_EX2_MESH_LINEAR_MAX_IT", 200))
        ilu_drop_tol = float(_env_float("PYCUTFEM_EX2_MESH_LINEAR_ILU_DROP_TOL", 0.0))
        ilu_fill_factor = float(_env_float("PYCUTFEM_EX2_MESH_LINEAR_ILU_FILL_FACTOR", 1.0))

        pre = build_subsolver(
            A_csr,
            {
                "kind": "ilu",
                "drop_tol": ilu_drop_tol,
                "fill_factor": ilu_fill_factor,
            },
        )
        precond = sp.linalg.LinearOperator(A_csr.shape, matvec=pre.solve, dtype=float)
        sol, info = bicgstab(
            A_csr,
            rhs_arr,
            x0=guess_arr,
            rtol=rel_tol,
            atol=0.0,
            maxiter=max_it,
            M=precond,
        )
        sol = np.asarray(sol, dtype=float).reshape(-1)
        if int(info) != 0 or not np.all(np.isfinite(sol)):
            raise RuntimeError(
                "mesh-extension iterative solve did not converge "
                f"(info={int(info)}, backend={backend_name}, mode={mode})."
            )
        return sol

    if backend_name == "petsc":
        try:
            from petsc4py import PETSc  # type: ignore
        except Exception as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("petsc4py is required for linear_backend='petsc'.") from exc

        mat = PETSc.Mat().createAIJ(
            size=A_csr.shape,
            csr=(
                A_csr.indptr.astype(PETSc.IntType, copy=False),
                A_csr.indices.astype(PETSc.IntType, copy=False),
                np.asarray(A_csr.data, dtype=float),
            ),
            comm=PETSc.COMM_SELF,
        )
        mat.assemblyBegin()
        mat.assemblyEnd()
        b = PETSc.Vec().createSeq(A_csr.shape[0], comm=PETSc.COMM_SELF)
        x = PETSc.Vec().createSeq(A_csr.shape[0], comm=PETSc.COMM_SELF)
        idx = np.arange(A_csr.shape[0], dtype=PETSc.IntType)
        b.setValues(idx, rhs_arr, addv=PETSc.InsertMode.INSERT_VALUES)
        b.assemblyBegin()
        b.assemblyEnd()
        ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp.setOperators(mat)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        ksp.solve(b, x)
        sol = x.getArray(readonly=True).copy()
        if np.all(np.isfinite(sol)):
            return np.asarray(sol, dtype=float).reshape(-1)

    sol = spsolve(A_csr, rhs_arr)
    sol = np.asarray(sol, dtype=float).reshape(-1)
    if not np.all(np.isfinite(sol)):
        raise RuntimeError(f"sparse solve returned non-finite values for backend={backend_name}, mode={mode}.")
    return sol


def _mesh_linear_amgcl_settings_from_env():
    from pycutfem.linalg import AMGCLSettings

    rel_tol = float(_env_float("PYCUTFEM_EX2_MESH_LINEAR_RTOL", 1.0e-7))
    max_it = int(_env_float("PYCUTFEM_EX2_MESH_LINEAR_MAX_IT", 200))
    gmres_dim = int(_env_float("PYCUTFEM_EX2_MESH_LINEAR_GMRES_DIM", 100))
    block_size_raw = _env_str("PYCUTFEM_EX2_MESH_LINEAR_BLOCK_SIZE", "auto").strip().lower()
    if block_size_raw in {"", "auto"}:
        # The mesh-moving solver in ProjectParametersCFD.json pins AMGCL to
        # block_size = 1. Keep the local default aligned with that production
        # Kratos path; larger block sizes remain available as explicit opt-ins.
        block_size_value = 1
    else:
        block_size_value = max(1, int(float(block_size_raw)))
    coarse_enough = int(_env_float("PYCUTFEM_EX2_MESH_LINEAR_COARSE_ENOUGH", 5000))
    return AMGCLSettings(
        preconditioner_type="amg",
        smoother_type="ilu0",
        krylov_type="gmres",
        coarsening_type="aggregation",
        tolerance=float(rel_tol),
        max_iteration=int(max_it),
        gmres_krylov_space_dimension=int(gmres_dim),
        verbosity=0,
        scaling=False,
        block_size=int(block_size_value),
        use_block_matrices_if_possible=True,
        coarse_enough=int(coarse_enough),
    )


def _mesh_extension_node_block_permutation(
    dh: DofHandler,
    *,
    mesh: Mesh | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    total_dofs = int(dh.total_dofs)
    node_map: dict[int, dict[str, int]] = {}
    for gdof in range(total_dofs):
        field_name, node_id = dh._dof_to_node_map[int(gdof)]
        if node_id is None:
            continue
        node_map.setdefault(int(node_id), {})[str(field_name)] = int(gdof)

    node_sort_ids: dict[int, int] = {}
    if mesh is not None:
        try:
            old_node_ids = np.asarray(_mesh_node_ids(mesh), dtype=int).reshape(-1)
            if old_node_ids.size == len(getattr(mesh, "nodes_list", ())):
                node_sort_ids = {
                    int(local_node_id): int(old_node_ids[int(local_node_id)])
                    for local_node_id in node_map
                    if 0 <= int(local_node_id) < old_node_ids.size
                }
        except Exception:
            node_sort_ids = {}

    perm: list[int] = []
    for node_id in sorted(node_map, key=lambda nid: (node_sort_ids.get(int(nid), int(nid)), int(nid))):
        entry = node_map[int(node_id)]
        if "mx" not in entry or "my" not in entry:
            return None
        perm.extend([int(entry["mx"]), int(entry["my"])])

    perm_arr = np.asarray(perm, dtype=int)
    if perm_arr.size != total_dofs:
        return None
    inv_perm = np.empty_like(perm_arr)
    inv_perm[perm_arr] = np.arange(total_dofs, dtype=int)
    return perm_arr, inv_perm


def _reference_geometry_mesh_extension_field(
    prob: dict[str, object],
) -> VectorFunction:
    cached = prob.get("_mesh_extension_reference_geom")
    if isinstance(cached, VectorFunction):
        return cached

    ref = VectorFunction("m_ref_geom", ["mx", "my"], dof_handler=prob["dh"])
    ref.nodal_values.fill(0.0)
    prob["_mesh_extension_reference_geom"] = ref
    return ref


def _cached_absolute_mesh_extension_system(
    *,
    prob: dict[str, object],
    bcs: list[BoundaryCondition],
) -> dict[str, object]:
    import scipy.sparse as sp

    dh: DofHandler = prob["dh"]
    mesh: Mesh = dh.mixed_element.mesh
    bc_map = dh.get_dirichlet_data(bcs) or {}
    bc_rows = np.fromiter((int(g) for g in bc_map.keys()), dtype=int, count=len(bc_map))
    bc_signature = tuple(sorted(int(row) for row in bc_rows.tolist()))

    cache = prob.get("_mesh_extension_absolute_cache")
    if (
        isinstance(cache, dict)
        and cache.get("bc_signature") == bc_signature
        and cache.get("shape") == (int(dh.total_dofs), int(dh.total_dofs))
    ):
        return cache

    A_raw = _assemble_structural_similarity_mesh_matrix(
        dh=dh,
        mesh=mesh,
        # StructuralMeshMovingStrategy resets coordinates to the initial
        # configuration before building the mesh system. Keep the cached local
        # operator on that same reference geometry.
        m_prev_geom=_reference_geometry_mesh_extension_field(prob),
    )
    A_raw_csr = A_raw.tocsr(copy=True) if hasattr(A_raw, "tocsr") else sp.csr_matrix(np.asarray(A_raw, dtype=float))
    A_constrained = A_raw_csr.tolil(copy=True)
    if bc_rows.size:
        A_constrained[bc_rows, :] = 0.0
        A_constrained[:, bc_rows] = 0.0
        A_constrained[bc_rows, bc_rows] = 1.0
    node_block_perm = _mesh_extension_node_block_permutation(dh, mesh=mesh)

    cache = {
        "shape": (int(dh.total_dofs), int(dh.total_dofs)),
        "bc_signature": bc_signature,
        "bc_rows": bc_rows,
        "A_raw": A_raw_csr,
        "A_constrained": A_constrained.tocsr(),
        "node_block_perm": None if node_block_perm is None else node_block_perm[0],
        "node_block_inv_perm": None if node_block_perm is None else node_block_perm[1],
    }
    prob["_mesh_extension_absolute_cache"] = cache
    return cache


def _cached_reference_mesh_extension_matrix(*, prob: dict[str, object]):
    cache = prob.get("_mesh_extension_reference_matrix_cache")
    dh: DofHandler = prob["dh"]
    mesh: Mesh = dh.mixed_element.mesh
    shape = (int(dh.total_dofs), int(dh.total_dofs))
    if (
        isinstance(cache, dict)
        and cache.get("shape") == shape
        and cache.get("mesh_elements") == int(mesh.n_elements)
    ):
        return cache["A_raw"]
    A_raw = _assemble_structural_similarity_mesh_matrix(
        dh=dh,
        mesh=mesh,
        # Kratos' StructuralMeshMovingStrategy resets coordinates to the
        # initial configuration before building the mesh-motion operator.
        m_prev_geom=_reference_geometry_mesh_extension_field(prob),
    )
    A_csr = A_raw.tocsr(copy=True) if hasattr(A_raw, "tocsr") else A_raw
    cache = {
        "shape": shape,
        "mesh_elements": int(mesh.n_elements),
        "A_raw": A_csr,
    }
    prob["_mesh_extension_reference_matrix_cache"] = cache
    return A_csr


def _solve_cached_absolute_mesh_extension_system(
    *,
    prob: dict[str, object],
    cache: dict[str, object],
    bcs: list[BoundaryCondition],
    solve_mode_name: str,
    linear_backend: str,
) -> np.ndarray:
    import scipy.sparse.linalg as spla

    dh: DofHandler = prob["dh"]
    total_dofs = int(dh.total_dofs)
    bc_map = dh.get_dirichlet_data(bcs) or {}
    bc_rows = np.asarray(cache["bc_rows"], dtype=int).reshape(-1)
    A_raw = cache["A_raw"]
    A_constrained = cache["A_constrained"]

    rhs = np.zeros(total_dofs, dtype=float)
    abs_guess = _global_dof_vector_from_function(dh, prob["m_k"])
    if bc_rows.size:
        x_bc = np.zeros(total_dofs, dtype=float)
        bc_values = np.asarray([float(bc_map[int(row)]) for row in bc_rows.tolist()], dtype=float)
        x_bc[bc_rows] = bc_values
        rhs = -np.asarray(A_raw @ x_bc, dtype=float).reshape(-1)
        rhs[bc_rows] = bc_values
        abs_guess[bc_rows] = bc_values

    mode = str(solve_mode_name).strip().lower()
    iterative_modes = {
        "amgcl",
        "pycutfem_amgcl",
        "cpp_amgcl",
        "local_amgcl",
        "kratos_amgcl",
        "kratos_like",
        "petsc_gmres_ilu",
        "petsc_gmres",
        "iterative",
        "bicgstab_ilu",
    }
    use_kratos_order_permutation = _env_bool(
        "PYCUTFEM_EX2_MESH_LINEAR_PERMUTE_TO_KRATOS_ORDER",
        False,
    )
    perm = np.asarray(cache.get("node_block_perm"), dtype=int).reshape(-1) if cache.get("node_block_perm") is not None else None
    use_permuted_iterative_system = bool(use_kratos_order_permutation) and perm is not None and mode in iterative_modes
    A_iter = A_constrained
    rhs_iter = rhs
    guess_iter = abs_guess
    if use_permuted_iterative_system:
        A_iter = cache.get("A_constrained_node_order")
        if A_iter is None:
            A_iter = A_constrained[perm, :][:, perm].tocsr()
            cache["A_constrained_node_order"] = A_iter
        rhs_iter = rhs[perm]
        guess_iter = abs_guess[perm]

    if mode in {"amgcl", "pycutfem_amgcl", "cpp_amgcl", "local_amgcl"}:
        from pycutfem.linalg import AMGCLSubsolver

        amgcl_settings = _mesh_linear_amgcl_settings_from_env()
        use_permuted_block_system = (
            use_permuted_iterative_system
            and int(amgcl_settings.block_size) == 2
            and bool(amgcl_settings.use_block_matrices_if_possible)
        )
        A_solver = A_iter
        rhs_solver = rhs_iter
        guess_solver = guess_iter
        solver_cache_key = "amgcl_subsolver"
        if use_permuted_block_system:
            A_solver = cache.get("A_constrained_block2")
            if A_solver is None:
                A_solver = A_iter
                cache["A_constrained_block2"] = A_solver
            solver_cache_key = "amgcl_subsolver_block2"
        elif int(amgcl_settings.block_size) != 1:
            amgcl_settings = type(amgcl_settings)(
                preconditioner_type=amgcl_settings.preconditioner_type,
                smoother_type=amgcl_settings.smoother_type,
                krylov_type=amgcl_settings.krylov_type,
                coarsening_type=amgcl_settings.coarsening_type,
                tolerance=amgcl_settings.tolerance,
                max_iteration=amgcl_settings.max_iteration,
                gmres_krylov_space_dimension=amgcl_settings.gmres_krylov_space_dimension,
                verbosity=amgcl_settings.verbosity,
                scaling=amgcl_settings.scaling,
                block_size=1,
                use_block_matrices_if_possible=False,
                coarse_enough=amgcl_settings.coarse_enough,
                max_levels=amgcl_settings.max_levels,
                pre_sweeps=amgcl_settings.pre_sweeps,
                post_sweeps=amgcl_settings.post_sweeps,
            )
        amgcl_signature = (
            float(amgcl_settings.tolerance),
            int(amgcl_settings.max_iteration),
            int(amgcl_settings.gmres_krylov_space_dimension),
            int(amgcl_settings.block_size),
            bool(amgcl_settings.use_block_matrices_if_possible),
            int(amgcl_settings.coarse_enough),
            int(amgcl_settings.max_levels),
            int(amgcl_settings.pre_sweeps),
            int(amgcl_settings.post_sweeps),
        )
        solver = cache.get(solver_cache_key)
        if solver is None or cache.get("amgcl_signature") != amgcl_signature:
            solver = AMGCLSubsolver(A_solver, params=amgcl_settings)
            cache[solver_cache_key] = solver
            cache["amgcl_signature"] = amgcl_signature
        solution = np.asarray(solver.solve(rhs_solver, x0=guess_solver), dtype=float).reshape(-1)
        if use_permuted_iterative_system:
            unpermuted = np.empty_like(solution)
            unpermuted[perm] = solution
            return unpermuted
        return solution

    if mode == "direct":
        factor = cache.get("direct_factor")
        if factor is None:
            factor = spla.factorized(A_constrained.tocsc())
            cache["direct_factor"] = factor
        return np.asarray(factor(rhs), dtype=float).reshape(-1)

    solution = _solve_sparse_linear_system(
        A=A_iter,
        rhs=rhs_iter,
        linear_backend=str(linear_backend),
        solve_mode=mode,
        initial_guess=guess_iter,
    )
    if use_permuted_iterative_system:
        unpermuted = np.empty_like(np.asarray(solution, dtype=float).reshape(-1))
        unpermuted[perm] = np.asarray(solution, dtype=float).reshape(-1)
        return unpermuted
    return solution


def _mesh_extension_dirichlet_bcs(
    *,
    interface_disp: CoordinateLookup,
    interface_tag: str,
    fixed_tags: tuple[str, ...],
) -> list[BoundaryCondition]:
    zero = lambda x, y, t=0.0: 0.0
    return [
        *[BoundaryCondition("mx", "dirichlet", tag, zero) for tag in fixed_tags],
        *[BoundaryCondition("my", "dirichlet", tag, zero) for tag in fixed_tags],
        BoundaryCondition("mx", "dirichlet", interface_tag, interface_disp.component(0)),
        BoundaryCondition("my", "dirichlet", interface_tag, interface_disp.component(1)),
    ]


def _apply_kratos_block_builder_dirichlet(
    *,
    A_raw,
    rhs: np.ndarray,
    fixed_rows: np.ndarray,
) -> tuple["sp.csr_matrix", np.ndarray]:
    import scipy.sparse as sp

    A_csr = A_raw.tocsr(copy=True) if hasattr(A_raw, "tocsr") else sp.csr_matrix(np.asarray(A_raw, dtype=float))
    rhs_out = np.asarray(rhs, dtype=float).reshape(-1).copy()
    if rhs_out.size != int(A_csr.shape[0]):
        raise ValueError(
            "Kratos-like Dirichlet application expected rhs size "
            f"{int(A_csr.shape[0])}, got {rhs_out.size}."
        )

    data = A_csr.data
    indices = A_csr.indices
    indptr = A_csr.indptr
    nrows = int(A_csr.shape[0])
    fixed_mask = np.zeros(nrows, dtype=bool)
    fixed_mask[np.asarray(fixed_rows, dtype=int).reshape(-1)] = True

    diag = np.asarray(A_csr.diagonal(), dtype=float).reshape(-1)
    scale_factor = float(np.max(np.abs(diag))) if diag.size else 1.0
    if not np.isfinite(scale_factor) or scale_factor <= 0.0:
        scale_factor = 1.0
    zero_tolerance = float(np.finfo(float).eps)

    for i in range(nrows):
        start = int(indptr[i])
        end = int(indptr[i + 1])
        row_cols = indices[start:end]
        row_vals = data[start:end]
        if row_vals.size == 0:
            continue
        if not np.any(np.abs(row_vals) > zero_tolerance):
            diag_pos = np.flatnonzero(row_cols == i)
            if diag_pos.size:
                row_vals[int(diag_pos[0])] = scale_factor
            rhs_out[i] = 0.0

    for i in range(nrows):
        start = int(indptr[i])
        end = int(indptr[i + 1])
        row_cols = indices[start:end]
        row_vals = data[start:end]
        if fixed_mask[i]:
            keep_diag = row_cols == i
            row_vals[~keep_diag] = 0.0
            rhs_out[i] = 0.0
        else:
            row_vals[fixed_mask[row_cols]] = 0.0

    return A_csr, rhs_out


def _assemble_mesh_extension_increment_system(
    *,
    A_raw,
    dh: DofHandler,
    bcs: list[BoundaryCondition],
    current_state: np.ndarray | None = None,
    kratos_builder_dirichlet: bool = False,
) -> tuple["sp.csr_matrix", np.ndarray, np.ndarray]:
    bc_map = dh.get_dirichlet_data(bcs) or {}
    if current_state is None:
        x_curr = np.zeros(int(dh.total_dofs), dtype=float)
    else:
        x_curr = np.asarray(current_state, dtype=float).reshape(-1).copy()
        if x_curr.size != int(dh.total_dofs):
            raise ValueError(
                "current_state size does not match mesh-extension dof count: "
                f"expected {int(dh.total_dofs)}, got {x_curr.size}."
            )
    for gdof, value in bc_map.items():
        x_curr[int(gdof)] = float(value)

    rows = np.fromiter((int(g) for g in bc_map.keys()), dtype=int) if bc_map else np.empty((0,), dtype=int)
    rhs = -(A_raw @ x_curr)
    if kratos_builder_dirichlet:
        A_constrained, rhs_constrained = _apply_kratos_block_builder_dirichlet(
            A_raw=A_raw,
            rhs=rhs,
            fixed_rows=rows,
        )
        return A_constrained, np.asarray(rhs_constrained, dtype=float).reshape(-1), x_curr

    A = A_raw.tolil(copy=True)
    if rows.size:
        A[rows, :] = 0.0
        A[:, rows] = 0.0
        A[rows, rows] = 1.0
        rhs[rows] = 0.0
    return A.tocsr(), np.asarray(rhs, dtype=float).reshape(-1), x_curr


def _assemble_structural_similarity_mesh_matrix(
    *,
    dh: DofHandler,
    mesh: Mesh,
    m_prev_geom: VectorFunction,
) -> "sp.csr_matrix":
    import scipy.sparse as sp

    elemental_dofs = np.asarray(dh.get_elemental_dofs_map(dtype=np.int64, copy=False), dtype=int)
    if elemental_dofs.ndim != 2 or elemental_dofs.shape[1] != 6:
        raise NotImplementedError("Kratos-matched structural-similarity assembly currently supports Triangle2D3 meshes only.")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    dn_de = np.asarray(
        [
            [-1.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    ref_weight = 0.5
    poisson = 0.3
    factor = 100.0
    xi = 1.5
    eps = 1.0e-30
    block_perm = np.asarray([0, 2, 4, 1, 3, 5], dtype=int)

    for eid in range(int(mesh.n_elements)):
        gdofs_x = np.asarray(dh.element_dofs("mx", eid), dtype=int).reshape(-1)
        gdofs_y = np.asarray(dh.element_dofs("my", eid), dtype=int).reshape(-1)
        if gdofs_x.size != 3 or gdofs_y.size != 3:
            raise NotImplementedError(
                "Kratos-matched structural-similarity assembly currently supports Triangle2D3 meshes only."
            )
        coords_ref = np.asarray(dh.element_dof_coords("mx", eid), dtype=float)
        disp_x = np.asarray(m_prev_geom.components[0].get_nodal_values(gdofs_x), dtype=float).reshape(-1)
        disp_y = np.asarray(m_prev_geom.components[1].get_nodal_values(gdofs_y), dtype=float).reshape(-1)
        coords = coords_ref + np.column_stack([disp_x, disp_y])
        j = np.asarray(
            [
                [coords[1, 0] - coords[0, 0], coords[2, 0] - coords[0, 0]],
                [coords[1, 1] - coords[0, 1], coords[2, 1] - coords[0, 1]],
            ],
            dtype=float,
        )
        det_j = float(np.linalg.det(j))
        if det_j <= eps:
            raise RuntimeError(f"Degenerate mesh-moving element encountered with detJ={det_j:.3e}.")
        inv_j = np.linalg.inv(j)
        dn_dx = dn_de @ inv_j

        b = np.zeros((3, 6), dtype=float)
        for i in range(3):
            b[0, 2 * i] = dn_dx[i, 0]
            b[1, 2 * i + 1] = dn_dx[i, 1]
            b[2, 2 * i] = dn_dx[i, 1]
            b[2, 2 * i + 1] = dn_dx[i, 0]

        weighting = det_j * (factor / det_j) ** xi
        lam = weighting * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        mu = weighting / (2.0 * (1.0 + poisson))
        c = np.asarray(
            [
                [lam + 2.0 * mu, lam, 0.0],
                [lam, lam + 2.0 * mu, 0.0],
                [0.0, 0.0, mu],
            ],
            dtype=float,
        )
        # The mixed-element DofHandler stacks field-local DOFs as
        # [mx(node0..2), my(node0..2)], while the local B-matrix above is built
        # in the conventional interleaved order [ux0, uy0, ux1, uy1, ux2, uy2].
        # Reorder the local stiffness before assembly so the global matrix lands
        # in the exact element-local ordering used everywhere else in pycutfem.
        k_elem = ref_weight * (b.T @ c @ b)
        k_elem = np.asarray(k_elem[np.ix_(block_perm, block_perm)], dtype=float)
        gdofs = np.asarray(elemental_dofs[eid, :], dtype=int)
        rr, cc = np.meshgrid(gdofs, gdofs, indexing="ij")
        rows.extend(rr.reshape(-1).tolist())
        cols.extend(cc.reshape(-1).tolist())
        data.extend(np.asarray(k_elem, dtype=float).reshape(-1).tolist())

    return sp.csr_matrix((data, (rows, cols)), shape=(int(dh.total_dofs), int(dh.total_dofs)))


def _build_kratos_local_mesh_backend(
    *,
    fluid_mdpa_path: Path,
    dt: float,
    bossak_alpha: float,
    prob: dict[str, object],
) -> dict[str, object]:
    backend = _build_kratos_mesh_motion_backend(
        fluid_mdpa_path=Path(fluid_mdpa_path),
        dt=float(dt),
        bossak_alpha=float(bossak_alpha),
    )

    mesh_model_part = backend["model"].GetModelPart("FluidModelPart_MeshPart")
    mesh: Mesh = prob["dh"].mixed_element.mesh
    dh: DofHandler = prob["dh"]
    dh._ensure_dof_coords()
    old_node_ids = np.asarray(_mesh_node_ids(mesh), dtype=int).reshape(-1)
    old_id_by_coord = {
        _coord_key(float(x), float(y)): int(old_node_ids[i])
        for i, (x, y) in enumerate(np.asarray(mesh.nodes_x_y_pos[:, :2], dtype=float))
    }

    mx_ids = np.asarray(dh.get_field_slice("mx"), dtype=int)
    my_ids = np.asarray(dh.get_field_slice("my"), dtype=int)
    mx_coords = np.asarray(dh._dof_coords[mx_ids], dtype=float)
    my_coords = np.asarray(dh._dof_coords[my_ids], dtype=float)
    if mx_coords.shape != my_coords.shape or (mx_coords.size and not np.allclose(mx_coords, my_coords)):
        raise RuntimeError("Kratos local mesh backend requires matching CG nodal coordinates for mx and my.")
    field_index_by_old_id = {
        old_id_by_coord[_coord_key(float(x), float(y))]: i
        for i, (x, y) in enumerate(mx_coords.tolist())
    }

    kratos_elem_by_nodes: dict[frozenset[int], object] = {}
    for elem in mesh_model_part.Elements:
        geom_ids = frozenset(int(node.Id) for node in elem.GetGeometry())
        kratos_elem_by_nodes[geom_ids] = elem

    n_elem = int(mesh.n_elements)
    gdofs_map = np.empty((n_elem, int(dh.mixed_element.n_dofs_per_elem)), dtype=int)
    local_perm = np.empty((n_elem, 6), dtype=int)
    kratos_elements: list[object] = [None] * n_elem
    for eid in range(n_elem):
        gdofs_map[eid, :] = np.asarray(dh.get_elemental_dofs(eid), dtype=int)
        py_mx_coords = np.asarray(dh.element_dof_coords("mx", eid), dtype=float)
        py_old_ids = [old_id_by_coord[_coord_key(float(x), float(y))] for x, y in py_mx_coords.tolist()]
        elem = kratos_elem_by_nodes.get(frozenset(int(node_id) for node_id in py_old_ids))
        if elem is None:
            raise RuntimeError(f"Could not match pycutfem mesh element {eid} to a Kratos mesh-moving element.")
        kratos_elements[eid] = elem
        py_pos_by_old_id = {int(node_id): j for j, node_id in enumerate(py_old_ids)}
        perm = np.empty(6, dtype=int)
        for k, node in enumerate(elem.GetGeometry()):
            py_pos = py_pos_by_old_id[int(node.Id)]
            perm[2 * k] = py_pos
            perm[2 * k + 1] = 3 + py_pos
        local_perm[eid, :] = perm

    return {
        "KM": backend["KM"],
        "analysis": backend["analysis"],
        "model": backend["model"],
        "solver": backend["solver"],
        "mesh_solver": backend["mesh_solver"],
        "main_model_part": backend["main_model_part"],
        "mesh_model_part": mesh_model_part,
        "interface_nodes": list(backend["interface_nodes"]),
        "zero_parts": tuple(backend["zero_parts"]),
        "all_nodes": list(mesh_model_part.Nodes),
        "node_coords_ref": np.asarray(backend["node_coords_ref"], dtype=float),
        "time": float(backend["time"]),
        "field_index_by_old_id": field_index_by_old_id,
        "kratos_row_by_old_id": {int(node.Id): i for i, node in enumerate(backend["all_nodes"])},
        "mx_ids": mx_ids,
        "my_ids": my_ids,
        "kratos_elements": tuple(kratos_elements),
        "gdofs_map": np.asarray(gdofs_map, dtype=int),
        "local_perm": np.asarray(local_perm, dtype=int),
    }


def _sync_kratos_local_mesh_backend_state(
    *,
    backend: dict[str, object],
    mesh_disp_nodal_values: np.ndarray | None = None,
) -> None:
    KM = backend["KM"]
    disp_values = None
    disp_order = "kratos_nodes"
    if mesh_disp_nodal_values is not None:
        disp_values = np.asarray(mesh_disp_nodal_values, dtype=float)
        if disp_values.ndim == 1:
            mx_ids = np.asarray(backend["mx_ids"], dtype=int)
            my_ids = np.asarray(backend["my_ids"], dtype=int)
            if disp_values.size <= int(max(np.max(mx_ids), np.max(my_ids), 0)):
                raise ValueError(
                    "Flat mesh_disp_nodal_values for Kratos local mesh backend is smaller than the required mesh DOF ids."
                )
            disp_values = np.column_stack([disp_values[mx_ids], disp_values[my_ids]])
            disp_order = "pycutfem_field"
        if disp_values.ndim != 2 or disp_values.shape[1] < 2:
            raise ValueError(
                "mesh_disp_nodal_values for Kratos local mesh backend must have shape (n_nodes, 2) or a flat mesh DOF vector."
            )
        if disp_values.shape[0] != len(backend["all_nodes"]):
            raise ValueError(
                "mesh_disp_nodal_values row count does not match the Kratos local mesh backend node count: "
                f"expected {len(backend['all_nodes'])}, got {disp_values.shape[0]}."
            )
    for node in backend["all_nodes"]:
        disp_x = 0.0
        disp_y = 0.0
        if disp_values is not None:
            if disp_order == "pycutfem_field":
                row_idx = backend["field_index_by_old_id"].get(int(node.Id))
                if row_idx is None:
                    raise RuntimeError(f"Missing field index for Kratos local mesh node {int(node.Id)}.")
            else:
                row_idx = backend["kratos_row_by_old_id"].get(int(node.Id))
                if row_idx is None:
                    raise RuntimeError(f"Missing Kratos row index for local mesh node {int(node.Id)}.")
            disp_x = float(disp_values[int(row_idx), 0])
            disp_y = float(disp_values[int(row_idx), 1])
        node.X = float(node.X0) + disp_x
        node.Y = float(node.Y0) + disp_y
        node.Z = float(node.Z0)
        disp_vec = KM.Array3()
        disp_vec[0] = disp_x
        disp_vec[1] = disp_y
        disp_vec[2] = 0.0
        node.SetSolutionStepValue(KM.MESH_DISPLACEMENT, 0, disp_vec)
        node.SetSolutionStepValue(KM.MESH_DISPLACEMENT, 1, disp_vec)


def _assemble_kratos_local_mesh_matrix_batch(
    *,
    backend: dict[str, object],
    need_matrix: bool,
    mesh_disp_nodal_values: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    _sync_kratos_local_mesh_backend_state(
        backend=backend,
        mesh_disp_nodal_values=mesh_disp_nodal_values,
    )
    KM = backend["KM"]
    perm_map = np.asarray(backend["local_perm"], dtype=int)
    gdofs_map = np.asarray(backend["gdofs_map"], dtype=int)
    elements = tuple(backend["kratos_elements"])
    process_info = backend["mesh_model_part"].ProcessInfo

    n_elem = int(len(elements))
    K_elem = np.zeros((n_elem, 6, 6), dtype=float) if bool(need_matrix) else None
    F_elem = np.zeros((n_elem, 6), dtype=float)
    for e, elem in enumerate(elements):
        lhs = KM.Matrix()
        rhs = KM.Vector()
        elem.CalculateLocalSystem(lhs, rhs, process_info)
        perm = np.asarray(perm_map[e], dtype=int)
        rhs_kr = np.asarray(rhs, dtype=float).reshape(-1)
        rhs_py = np.zeros((6,), dtype=float)
        rhs_py[perm] = rhs_kr
        F_elem[e, :] = rhs_py
        if K_elem is not None:
            lhs_kr = np.asarray(lhs, dtype=float)
            lhs_py = np.zeros((6, 6), dtype=float)
            lhs_py[np.ix_(perm, perm)] = lhs_kr
            K_elem[e, :, :] = lhs_py
    element_ids = np.arange(n_elem, dtype=int)
    return K_elem, F_elem, element_ids, gdofs_map


def _ensure_kratos_local_mesh_equation_map(
    *,
    backend: dict[str, object],
) -> np.ndarray:
    gdof_to_eq_cached = np.asarray(backend.get("gdof_to_eq", np.empty((0,), dtype=int)), dtype=int).reshape(-1)
    if gdof_to_eq_cached.size:
        return gdof_to_eq_cached

    KM = backend["KM"]
    field_index_by_old_id = dict(backend["field_index_by_old_id"])
    mx_ids = np.asarray(backend["mx_ids"], dtype=int)
    my_ids = np.asarray(backend["my_ids"], dtype=int)
    eq_entries: list[tuple[int, int]] = []
    for node in backend["all_nodes"]:
        field_idx = int(field_index_by_old_id[int(node.Id)])
        if node.HasDofFor(KM.MESH_DISPLACEMENT_X):
            eq_entries.append((int(node.GetDof(KM.MESH_DISPLACEMENT_X).EquationId), int(mx_ids[field_idx])))
        if node.HasDofFor(KM.MESH_DISPLACEMENT_Y):
            eq_entries.append((int(node.GetDof(KM.MESH_DISPLACEMENT_Y).EquationId), int(my_ids[field_idx])))
    ndof = max((int(eq_id) for eq_id, _ in eq_entries), default=-1) + 1
    eq_to_gdof = np.full((ndof,), -1, dtype=int)
    for eq_id, gdof in eq_entries:
        eq_to_gdof[int(eq_id)] = int(gdof)
    if ndof > 0 and np.any(eq_to_gdof < 0):
        raise RuntimeError("Incomplete Kratos local mesh equation-to-gdof mapping.")
    total_gdofs = 1 + max((int(gdof) for _eq_id, gdof in eq_entries), default=-1)
    gdof_to_eq = np.full((total_gdofs,), -1, dtype=int)
    if ndof > 0 and total_gdofs > 0:
        gdof_to_eq[eq_to_gdof] = np.arange(ndof, dtype=int)
    backend["eq_to_gdof"] = np.asarray(eq_to_gdof, dtype=int)
    backend["gdof_to_eq"] = np.asarray(gdof_to_eq, dtype=int)
    return np.asarray(gdof_to_eq, dtype=int)


def _solve_kratos_local_mesh_displacement(
    *,
    backend: dict[str, object],
    interface_disp: CoordinateLookup,
) -> CoordinateLookup:
    disp_lookup, _, _ = _solve_kratos_mesh_motion_backend(
        backend=backend,
        interface_disp=interface_disp,
    )
    return disp_lookup


def _solve_structural_similarity_mesh_extension(
    *,
    prob: dict[str, object],
    interface_disp: CoordinateLookup,
    interface_tag: str,
    fixed_tags: tuple[str, ...],
    kratos_local_backend: dict[str, object] | None = None,
    quad_order: int = 6,
    backend: str = "cpp",
    linear_backend: str = "petsc",
) -> None:
    dh: DofHandler = prob["dh"]
    solve_mode = _env_str(
        "PYCUTFEM_EX2_MESH_EXTENSION_SOLVER",
        "direct",
    ).strip().lower()
    if solve_mode in {"cpp_kratos_strategy", "cpp_mesh_strategy", "cpp_strategy"}:
        cpp_backend = _build_cpp_mesh_strategy_backend(
            prob=prob,
            interface_tag=interface_tag,
            fixed_tags=fixed_tags,
        )
        interface_coords = np.asarray(cpp_backend["interface_coords"], dtype=float)
        interface_values = np.asarray(
            interface_disp(interface_coords[:, 0], interface_coords[:, 1]),
            dtype=float,
        ).reshape(interface_coords.shape[0], 2)
        solution = np.asarray(
            cpp_backend["strategy"].solve(
                interface_values=interface_values,
                current_state=_node_matrix_from_vector_function(dh, prob["m_k"]),
                preserve_free_state=True,
            ),
            dtype=float,
        )
        _transfer_vector_field(
            target_dh=dh,
            target_vec=prob["m_k"],
            source_lookup=CoordinateLookup(
                np.asarray(cpp_backend["node_coords"], dtype=float),
                solution,
                dim=2,
            ),
        )
        return
    if solve_mode in {"kratos_local", "kratos"}:
        if kratos_local_backend is None:
            raise RuntimeError("kratos_local mesh-extension solver requested, but no Kratos local mesh backend is available.")
        mesh_lookup = _solve_kratos_local_mesh_displacement(
            backend=kratos_local_backend,
            interface_disp=interface_disp,
        )
        _transfer_vector_field(
            target_dh=dh,
            target_vec=prob["m_k"],
            source_lookup=mesh_lookup,
        )
        return

    if solve_mode in {"reference_form", "ufl", "equation"}:
        mesh_eq, mesh_bcs = _mesh_extension_equation(
            prob=prob,
            interface_disp=interface_disp,
            interface_tag=interface_tag,
            fixed_tags=fixed_tags,
            quad_order=int(quad_order),
        )
        _solve_linear(
            eq=mesh_eq,
            dh=dh,
            bcs=mesh_bcs,
            quad_order=int(quad_order),
            backend=str(backend),
            linear_backend=str(linear_backend),
            functions=[prob["m_k"]],
        )
        return

    if solve_mode not in {"direct", "matrix"}:
        raise ValueError(
            "Unsupported mesh-extension solver mode "
            f"{solve_mode!r}. Use one of: reference_form, direct."
        )

    bcs = _mesh_extension_dirichlet_bcs(
        interface_disp=interface_disp,
        interface_tag=interface_tag,
        fixed_tags=fixed_tags,
    )

    solve_mode_name_raw = os.getenv("PYCUTFEM_EX2_MESH_LINEAR_SOLVE_MODE", "").strip().lower()
    # Kratos' structural mesh-moving strategy uses AMGCL/GMRES on the
    # reference-configuration mesh operator. Mirror that by default so the
    # local path follows the same runtime solve path.
    solve_mode_name = solve_mode_name_raw or "amgcl"
    system_form_raw = os.getenv("PYCUTFEM_EX2_MESH_EXTENSION_FORMULATION", "").strip().lower()
    if system_form_raw:
        system_form = system_form_raw
    else:
        # Kratos' structural mesh-moving strategy assembles the residual on the
        # current mesh-displacement state and solves for an increment through
        # ResidualBasedLinearStrategy + ResidualBasedBlockBuilderAndSolver.
        # Mirror that incremental path for iterative mesh linear solvers.
        if solve_mode_name in {
            "amgcl",
            "pycutfem_amgcl",
            "cpp_amgcl",
            "local_amgcl",
            "kratos_amgcl",
            "kratos_like",
            "petsc_gmres_ilu",
            "petsc_gmres",
            "iterative",
            "bicgstab_ilu",
        }:
            system_form = "incremental"
        else:
            system_form = "absolute"
    if system_form not in {"incremental", "absolute"}:
        raise ValueError(
            "Unsupported mesh-extension formulation "
            f"{system_form!r}. Use one of: incremental, absolute."
        )

    if system_form == "incremental":
        mesh: Mesh = dh.mixed_element.mesh
        A_raw = _cached_reference_mesh_extension_matrix(prob=prob)
        current_state = _global_dof_vector_from_function(dh, prob["m_k"])
        use_kratos_builder = solve_mode_name in {
            "amgcl",
            "pycutfem_amgcl",
            "cpp_amgcl",
            "local_amgcl",
            "kratos_amgcl",
            "kratos_like",
        }
        A_constrained, rhs_constrained, x_curr = _assemble_mesh_extension_increment_system(
            A_raw=A_raw,
            dh=dh,
            bcs=bcs,
            current_state=current_state,
            kratos_builder_dirichlet=use_kratos_builder,
        )
        iterative_modes = {
            "amgcl",
            "pycutfem_amgcl",
            "cpp_amgcl",
            "local_amgcl",
            "kratos_amgcl",
            "kratos_like",
            "petsc_gmres_ilu",
            "petsc_gmres",
            "iterative",
            "bicgstab_ilu",
        }
        use_kratos_order_permutation = _env_bool(
            "PYCUTFEM_EX2_MESH_LINEAR_PERMUTE_TO_KRATOS_ORDER",
            True,
        )
        perm_info = _mesh_extension_node_block_permutation(dh, mesh=mesh)
        perm = None if perm_info is None else np.asarray(perm_info[0], dtype=int).reshape(-1)
        use_permuted_iterative_system = (
            bool(use_kratos_order_permutation)
            and perm is not None
            and solve_mode_name in iterative_modes
        )
        A_solver = A_constrained
        rhs_solver = rhs_constrained
        delta_guess = None if use_kratos_builder else (current_state - np.asarray(x_curr, dtype=float).reshape(-1))
        if use_permuted_iterative_system:
            A_solver = A_constrained[perm, :][:, perm].tocsr()
            rhs_solver = np.asarray(rhs_constrained, dtype=float).reshape(-1)[perm]
            if delta_guess is not None:
                delta_guess = np.asarray(delta_guess, dtype=float).reshape(-1)[perm]
        local_amgcl_modes = {"amgcl", "pycutfem_amgcl", "cpp_amgcl", "local_amgcl"}
        if solve_mode_name in local_amgcl_modes:
            from pycutfem.linalg import AMGCLSettings, AMGCLSubsolver

            amgcl_settings = _mesh_linear_amgcl_settings_from_env()
            if int(amgcl_settings.block_size) != 1 and not use_permuted_iterative_system:
                amgcl_settings = AMGCLSettings(
                    preconditioner_type=amgcl_settings.preconditioner_type,
                    smoother_type=amgcl_settings.smoother_type,
                    krylov_type=amgcl_settings.krylov_type,
                    coarsening_type=amgcl_settings.coarsening_type,
                    tolerance=amgcl_settings.tolerance,
                    max_iteration=amgcl_settings.max_iteration,
                    gmres_krylov_space_dimension=amgcl_settings.gmres_krylov_space_dimension,
                    verbosity=amgcl_settings.verbosity,
                    scaling=amgcl_settings.scaling,
                    block_size=1,
                    use_block_matrices_if_possible=False,
                    coarse_enough=amgcl_settings.coarse_enough,
                    max_levels=amgcl_settings.max_levels,
                    pre_sweeps=amgcl_settings.pre_sweeps,
                    post_sweeps=amgcl_settings.post_sweeps,
                )
            use_cached_incremental_solver = True
            delta_solver = None
            delta_solver_signature = (
                tuple(map(int, A_solver.shape)),
                int(getattr(A_solver, "nnz", 0)),
                float(amgcl_settings.tolerance),
                int(amgcl_settings.max_iteration),
                int(amgcl_settings.gmres_krylov_space_dimension),
                int(amgcl_settings.block_size),
                bool(amgcl_settings.use_block_matrices_if_possible),
                int(amgcl_settings.coarse_enough),
                int(amgcl_settings.max_levels),
                int(amgcl_settings.pre_sweeps),
                int(amgcl_settings.post_sweeps),
                bool(use_permuted_iterative_system),
            )
            if use_cached_incremental_solver:
                delta_solver = prob.get("_mesh_extension_incremental_amgcl_solver")
                if delta_solver is None or prob.get("_mesh_extension_incremental_amgcl_signature") != delta_solver_signature:
                    delta_solver = AMGCLSubsolver(A_solver, params=amgcl_settings)
                    prob["_mesh_extension_incremental_amgcl_solver"] = delta_solver
                    prob["_mesh_extension_incremental_amgcl_signature"] = delta_solver_signature
            else:
                delta_solver = AMGCLSubsolver(A_solver, params=amgcl_settings)
            delta = np.asarray(delta_solver.solve(rhs_solver, x0=delta_guess), dtype=float).reshape(-1)
        else:
            delta = _solve_sparse_linear_system(
                A=A_solver,
                rhs=rhs_solver,
                linear_backend=str(linear_backend),
                solve_mode=solve_mode_name,
                initial_guess=delta_guess,
            )
        if use_permuted_iterative_system:
            delta_unpermuted = np.empty_like(np.asarray(delta, dtype=float).reshape(-1))
            delta_unpermuted[perm] = np.asarray(delta, dtype=float).reshape(-1)
            delta = delta_unpermuted
        sol = np.asarray(x_curr, dtype=float).reshape(-1) + np.asarray(delta, dtype=float).reshape(-1)
    else:
        cache = _cached_absolute_mesh_extension_system(prob=prob, bcs=bcs)
        if cache is None:
            mesh: Mesh = dh.mixed_element.mesh
            A_raw = _assemble_structural_similarity_mesh_matrix(
                dh=dh,
                mesh=mesh,
                m_prev_geom=_reference_geometry_mesh_extension_field(prob),
            )
            rhs_raw = np.zeros(int(dh.total_dofs), dtype=float)
            bc_map = dh.get_dirichlet_data(bcs) or {}
            A = A_raw.tolil(copy=True)
            rhs = np.asarray(rhs_raw, dtype=float).copy()
            abs_guess = _global_dof_vector_from_function(dh, prob["m_k"])
            if bc_map:
                rows = np.fromiter((int(g) for g in bc_map.keys()), dtype=int)
                vals = np.fromiter((float(v) for v in bc_map.values()), dtype=float)
                x_bc = np.zeros(int(dh.total_dofs), dtype=float)
                x_bc[rows] = vals
                rhs = rhs - A_raw @ x_bc
                A[rows, :] = 0.0
                A[:, rows] = 0.0
                A[rows, rows] = 1.0
                rhs[rows] = vals
                abs_guess[rows] = vals
            sol = _solve_sparse_linear_system(
                A=A.tocsr(),
                rhs=rhs,
                linear_backend=str(linear_backend),
                solve_mode=solve_mode_name,
                initial_guess=abs_guess,
            )
        else:
            sol = _solve_cached_absolute_mesh_extension_system(
                prob=prob,
                cache=cache,
                bcs=bcs,
                solve_mode_name=solve_mode_name,
                linear_backend=str(linear_backend),
            )
    prob["m_k"].nodal_values.fill(0.0)
    dh.add_to_functions(np.asarray(sol, dtype=float), [prob["m_k"]])
    dh.apply_bcs(bcs, prob["m_k"])


def _kratos_node_step_vector_values(nodes, variable, *, step: int = 0) -> np.ndarray:
    values = np.zeros((len(nodes), 2), dtype=float)
    for i, node in enumerate(nodes):
        try:
            raw = node.GetSolutionStepValue(variable, int(step))
        except Exception:
            continue
        arr = np.asarray(raw, dtype=float).reshape(-1)
        if arr.size >= 2:
            values[i, :] = arr[:2]
        elif arr.size == 1:
            values[i, 0] = arr[0]
    return values


def _kratos_node_step_scalar_values(nodes, variable, *, step: int = 0) -> np.ndarray:
    values = np.zeros((len(nodes), 1), dtype=float)
    for i, node in enumerate(nodes):
        try:
            values[i, 0] = float(node.GetSolutionStepValue(variable, int(step)))
        except Exception:
            continue
    return values


def _build_kratos_mesh_motion_backend(
    *,
    fluid_mdpa_path: Path,
    dt: float,
    bossak_alpha: float,
) -> dict[str, object]:
    import KratosMultiphysics as KM
    from KratosMultiphysics.FluidDynamicsApplication.fluid_dynamics_analysis import FluidDynamicsAnalysis

    mdpa_path = Path(fluid_mdpa_path).resolve()
    if mdpa_path.suffix.lower() != ".mdpa":
        raise ValueError(f"Expected an MDPA path for Kratos mesh backend, got: {mdpa_path}")

    benchmark_root = mdpa_path.parent.parent.resolve()
    settings_data = _load_json(benchmark_root / "ProjectParametersCFD.json")
    settings_data["problem_data"]["echo_level"] = 0
    settings_data["problem_data"]["end_time"] = float(dt)
    settings_data["solver_settings"]["fluid_solver_settings"]["echo_level"] = 0
    settings_data["solver_settings"]["fluid_solver_settings"]["time_stepping"]["time_step"] = float(dt)
    settings_data["solver_settings"]["mesh_motion_solver_settings"]["echo_level"] = 0
    settings_data["solver_settings"]["mesh_motion_solver_settings"]["time_stepping"] = {
        "time_step": float(dt),
    }
    settings_data["solver_settings"]["mesh_motion_solver_settings"]["mesh_velocity_calculation"] = {
        "time_scheme": "bossak",
        "alpha_m": float(bossak_alpha),
    }
    settings_data["output_processes"] = {}

    model = KM.Model()
    analysis = FluidDynamicsAnalysis(model, KM.Parameters(json.dumps(settings_data)))
    cwd = Path.cwd()
    try:
        os.chdir(benchmark_root)
        analysis.Initialize()
    finally:
        os.chdir(cwd)

    solver = analysis._GetSolver()
    mesh_solver = solver.GetMeshMotionSolver()
    main_model_part = model.GetModelPart("FluidModelPart")
    mesh_model_part = mesh_solver.GetComputingModelPart()

    subparts = {
        "interface": main_model_part.GetSubModelPart("NoSlip2D_Interface"),
        "top": main_model_part.GetSubModelPart("NoSlip2D_Top"),
        "bottom": main_model_part.GetSubModelPart("NoSlip2D_Bottom"),
        "cylinder": main_model_part.GetSubModelPart("NoSlip2D_Cylinder"),
        "inlet": main_model_part.GetSubModelPart("AutomaticInlet2D_Inlet"),
        "outlet": main_model_part.GetSubModelPart("Outlet2D_Outlet"),
    }

    var_utils = KM.VariableUtils()
    for name in ("interface", "top", "bottom", "cylinder", "inlet", "outlet"):
        nodes = subparts[name].Nodes
        var_utils.ApplyFixity(KM.MESH_DISPLACEMENT_X, True, nodes)
        var_utils.ApplyFixity(KM.MESH_DISPLACEMENT_Y, True, nodes)

    zero_parts = tuple(subparts[name] for name in ("top", "bottom", "cylinder", "inlet", "outlet"))
    interface_nodes = list(subparts["interface"].Nodes)
    all_nodes = list(main_model_part.Nodes)
    node_coords_ref = np.asarray([[float(node.X0), float(node.Y0)] for node in all_nodes], dtype=float)

    def _zero_node_history(nodes) -> None:
        zero_vec = KM.Array3()
        zero_vec[0] = 0.0
        zero_vec[1] = 0.0
        zero_vec[2] = 0.0
        for node in nodes:
            node.SetSolutionStepValue(KM.MESH_DISPLACEMENT, 0, zero_vec)
            node.SetSolutionStepValue(KM.MESH_DISPLACEMENT, 1, zero_vec)
            node.SetSolutionStepValue(KM.MESH_VELOCITY, 0, zero_vec)
            node.SetSolutionStepValue(KM.MESH_VELOCITY, 1, zero_vec)
            if node.SolutionStepsDataHas(KM.MESH_ACCELERATION):
                node.SetSolutionStepValue(KM.MESH_ACCELERATION, 0, zero_vec)
                node.SetSolutionStepValue(KM.MESH_ACCELERATION, 1, zero_vec)

    _zero_node_history(all_nodes)
    return {
        "KM": KM,
        "analysis": analysis,
        "model": model,
        "solver": solver,
        "mesh_solver": mesh_solver,
        "main_model_part": main_model_part,
        "mesh_model_part": mesh_model_part,
        "interface_nodes": interface_nodes,
        "zero_parts": zero_parts,
        "all_nodes": all_nodes,
        "node_coords_ref": node_coords_ref,
        "time": 0.0,
    }


def _build_kratos_exact_fluid_backend(
    *,
    fluid_mdpa_path: Path,
    dt: float,
    bossak_alpha: float,
) -> dict[str, object]:
    import KratosMultiphysics as KM
    from KratosMultiphysics.FluidDynamicsApplication.fluid_dynamics_analysis import FluidDynamicsAnalysis

    mdpa_path = Path(fluid_mdpa_path).resolve()
    if mdpa_path.suffix.lower() != ".mdpa":
        raise ValueError(f"Expected an MDPA path for Kratos fluid backend, got: {mdpa_path}")

    benchmark_root = mdpa_path.parent.parent.resolve()
    settings_data = _load_json(benchmark_root / "ProjectParametersCFD.json")
    settings_data["problem_data"]["echo_level"] = 0
    settings_data["problem_data"]["end_time"] = float(dt)
    settings_data["solver_settings"]["fluid_solver_settings"]["echo_level"] = 0
    settings_data["solver_settings"]["fluid_solver_settings"]["time_stepping"]["time_step"] = float(dt)
    settings_data["solver_settings"]["mesh_motion_solver_settings"]["echo_level"] = 0
    settings_data["solver_settings"]["mesh_motion_solver_settings"]["mesh_velocity_calculation"] = {
        "time_scheme": "bossak",
        "alpha_m": float(bossak_alpha),
    }
    settings_data["output_processes"] = {}

    model = KM.Model()
    analysis = FluidDynamicsAnalysis(model, KM.Parameters(json.dumps(settings_data)))
    cwd = Path.cwd()
    try:
        os.chdir(benchmark_root)
        analysis.Initialize()
    finally:
        os.chdir(cwd)

    solver = analysis._GetSolver()
    main_model_part = model.GetModelPart("FluidModelPart")
    subparts = {
        "interface": main_model_part.GetSubModelPart("NoSlip2D_Interface"),
        "top": main_model_part.GetSubModelPart("NoSlip2D_Top"),
        "bottom": main_model_part.GetSubModelPart("NoSlip2D_Bottom"),
        "cylinder": main_model_part.GetSubModelPart("NoSlip2D_Cylinder"),
        "inlet": main_model_part.GetSubModelPart("AutomaticInlet2D_Inlet"),
        "outlet": main_model_part.GetSubModelPart("Outlet2D_Outlet"),
    }

    var_utils = KM.VariableUtils()
    for name in ("interface", "top", "bottom", "cylinder", "inlet", "outlet"):
        nodes = subparts[name].Nodes
        var_utils.ApplyFixity(KM.MESH_DISPLACEMENT_X, True, nodes)
        var_utils.ApplyFixity(KM.MESH_DISPLACEMENT_Y, True, nodes)

    zero_parts = tuple(subparts[name] for name in ("top", "bottom", "cylinder", "inlet", "outlet"))
    interface_nodes = list(subparts["interface"].Nodes)
    all_nodes = list(main_model_part.Nodes)
    node_coords_ref = np.asarray([[float(node.X0), float(node.Y0)] for node in all_nodes], dtype=float)
    interface_coords_ref = np.asarray([[float(node.X0), float(node.Y0)] for node in interface_nodes], dtype=float)

    zero_vec = KM.Array3()
    zero_vec[0] = 0.0
    zero_vec[1] = 0.0
    zero_vec[2] = 0.0
    for node in all_nodes:
        node.SetSolutionStepValue(KM.MESH_DISPLACEMENT, 0, zero_vec)
        node.SetSolutionStepValue(KM.MESH_DISPLACEMENT, 1, zero_vec)
        node.SetSolutionStepValue(KM.MESH_VELOCITY, 0, zero_vec)
        node.SetSolutionStepValue(KM.MESH_VELOCITY, 1, zero_vec)
        if node.SolutionStepsDataHas(KM.MESH_ACCELERATION):
            node.SetSolutionStepValue(KM.MESH_ACCELERATION, 0, zero_vec)
            node.SetSolutionStepValue(KM.MESH_ACCELERATION, 1, zero_vec)

    return {
        "KM": KM,
        "analysis": analysis,
        "model": model,
        "solver": solver,
        "main_model_part": main_model_part,
        "all_nodes": all_nodes,
        "interface_nodes": interface_nodes,
        "node_coords_ref": node_coords_ref,
        "interface_coords_ref": interface_coords_ref,
        "zero_parts": zero_parts,
        "time": 0.0,
    }


def _advance_kratos_exact_fluid_backend_step(
    *,
    backend: dict[str, object],
) -> None:
    analysis = backend["analysis"]
    solver = backend["solver"]
    current_time = float(backend["time"])
    new_time = float(solver.AdvanceInTime(current_time))
    analysis.time = new_time
    solver.Predict()
    analysis.InitializeSolutionStep()
    backend["time"] = new_time


def _finalize_kratos_exact_fluid_backend_step(
    *,
    backend: dict[str, object],
) -> None:
    backend["analysis"].FinalizeSolutionStep()


def _solve_kratos_exact_fluid_backend(
    *,
    backend: dict[str, object],
    interface_disp: CoordinateLookup,
) -> dict[str, CoordinateLookup]:
    KM = backend["KM"]

    zero_vec = KM.Array3()
    zero_vec[0] = 0.0
    zero_vec[1] = 0.0
    zero_vec[2] = 0.0

    for part in backend["zero_parts"]:
        for node in part.Nodes:
            node.SetSolutionStepValue(KM.MESH_DISPLACEMENT, 0, zero_vec)

    for node in backend["interface_nodes"]:
        values = np.asarray(interface_disp(float(node.X0), float(node.Y0)), dtype=float).reshape(2)
        disp_vec = KM.Array3()
        disp_vec[0] = float(values[0])
        disp_vec[1] = float(values[1])
        disp_vec[2] = 0.0
        node.SetSolutionStepValue(KM.MESH_DISPLACEMENT, 0, disp_vec)

    backend["solver"].SolveSolutionStep()

    all_nodes = backend["all_nodes"]
    coords = np.asarray(backend["node_coords_ref"], dtype=float)
    interface_nodes = backend["interface_nodes"]
    interface_coords = np.asarray(backend["interface_coords_ref"], dtype=float)

    mesh_disp = _kratos_node_step_vector_values(all_nodes, KM.MESH_DISPLACEMENT)
    mesh_vel = _kratos_node_step_vector_values(all_nodes, KM.MESH_VELOCITY)
    mesh_acc = _kratos_node_step_vector_values(all_nodes, KM.MESH_ACCELERATION)
    velocity = _kratos_node_step_vector_values(all_nodes, KM.VELOCITY)
    acceleration = _kratos_node_step_vector_values(all_nodes, KM.ACCELERATION)
    pressure = _kratos_node_step_scalar_values(all_nodes, KM.PRESSURE)
    reaction = _kratos_node_step_vector_values(interface_nodes, KM.REACTION)

    return {
        "mesh_displacement": CoordinateLookup(coords, mesh_disp, dim=2),
        "mesh_velocity": CoordinateLookup(coords, mesh_vel, dim=2),
        "mesh_acceleration": CoordinateLookup(coords, mesh_acc, dim=2),
        "velocity": CoordinateLookup(coords, velocity, dim=2),
        "acceleration": CoordinateLookup(coords, acceleration, dim=2),
        "pressure": CoordinateLookup(coords, pressure, dim=1),
        "reaction": CoordinateLookup(interface_coords, reaction, dim=2),
    }


def _advance_kratos_mesh_motion_backend_step(
    *,
    backend: dict[str, object],
) -> None:
    solver = backend["mesh_solver"]
    current_time = float(backend["time"])
    new_time = float(solver.AdvanceInTime(current_time))
    analysis = backend.get("analysis")
    if analysis is not None:
        for process in analysis._GetListOfProcesses():
            process.ExecuteInitializeSolutionStep()
    solver.Predict()
    solver.InitializeSolutionStep()
    backend["time"] = new_time


def _finalize_kratos_mesh_motion_backend_step(
    *,
    backend: dict[str, object],
) -> None:
    solver = backend["mesh_solver"]
    solver.FinalizeSolutionStep()
    analysis = backend.get("analysis")
    if analysis is not None:
        for process in analysis._GetListOfProcesses():
            process.ExecuteFinalizeSolutionStep()


def _solve_kratos_mesh_motion_backend(
    *,
    backend: dict[str, object],
    interface_disp: CoordinateLookup,
) -> tuple[CoordinateLookup, CoordinateLookup, CoordinateLookup]:
    import KratosMultiphysics as KM

    zero_vec = KM.Array3()
    zero_vec[0] = 0.0
    zero_vec[1] = 0.0
    zero_vec[2] = 0.0

    for part in backend["zero_parts"]:
        for node in part.Nodes:
            node.SetSolutionStepValue(KM.MESH_DISPLACEMENT, 0, zero_vec)

    for node in backend["interface_nodes"]:
        values = np.asarray(interface_disp(float(node.X0), float(node.Y0)), dtype=float).reshape(2)
        disp_vec = KM.Array3()
        disp_vec[0] = float(values[0])
        disp_vec[1] = float(values[1])
        disp_vec[2] = 0.0
        node.SetSolutionStepValue(KM.MESH_DISPLACEMENT, 0, disp_vec)

    backend["mesh_solver"].SolveSolutionStep()

    all_nodes = backend["all_nodes"]
    coords = np.asarray(backend["node_coords_ref"], dtype=float)
    disp = np.asarray(
        [[float(node.GetSolutionStepValue(KM.MESH_DISPLACEMENT)[0]), float(node.GetSolutionStepValue(KM.MESH_DISPLACEMENT)[1])] for node in all_nodes],
        dtype=float,
    )
    vel = np.asarray(
        [[float(node.GetSolutionStepValue(KM.MESH_VELOCITY)[0]), float(node.GetSolutionStepValue(KM.MESH_VELOCITY)[1])] for node in all_nodes],
        dtype=float,
    )
    acc = np.asarray(
        [[float(node.GetSolutionStepValue(KM.MESH_ACCELERATION)[0]), float(node.GetSolutionStepValue(KM.MESH_ACCELERATION)[1])] for node in all_nodes],
        dtype=float,
    )
    return (
        CoordinateLookup(coords, disp, dim=2),
        CoordinateLookup(coords, vel, dim=2),
        CoordinateLookup(coords, acc, dim=2),
    )


def _sync_kratos_mesh_motion_backend_current_state(
    *,
    backend: dict[str, object],
    mesh_disp: CoordinateLookup,
    mesh_vel: CoordinateLookup | None = None,
    mesh_acc: CoordinateLookup | None = None,
) -> None:
    import KratosMultiphysics as KM

    all_nodes = backend["all_nodes"]
    for node in all_nodes:
        x0 = float(node.X0)
        y0 = float(node.Y0)
        disp_vals = np.asarray(mesh_disp(x0, y0), dtype=float).reshape(2)
        vel_vals = (
            np.zeros(2, dtype=float)
            if mesh_vel is None
            else np.asarray(mesh_vel(x0, y0), dtype=float).reshape(2)
        )
        acc_vals = (
            np.zeros(2, dtype=float)
            if mesh_acc is None
            else np.asarray(mesh_acc(x0, y0), dtype=float).reshape(2)
        )

        disp_vec = KM.Array3()
        disp_vec[0] = float(disp_vals[0])
        disp_vec[1] = float(disp_vals[1])
        disp_vec[2] = 0.0
        vel_vec = KM.Array3()
        vel_vec[0] = float(vel_vals[0])
        vel_vec[1] = float(vel_vals[1])
        vel_vec[2] = 0.0
        acc_vec = KM.Array3()
        acc_vec[0] = float(acc_vals[0])
        acc_vec[1] = float(acc_vals[1])
        acc_vec[2] = 0.0

        node.X = x0 + float(disp_vals[0])
        node.Y = y0 + float(disp_vals[1])
        node.Z = 0.0
        node.SetSolutionStepValue(KM.MESH_DISPLACEMENT, 0, disp_vec)
        node.SetSolutionStepValue(KM.MESH_VELOCITY, 0, vel_vec)
        if node.SolutionStepsDataHas(KM.MESH_ACCELERATION):
            node.SetSolutionStepValue(KM.MESH_ACCELERATION, 0, acc_vec)


def _mdpa_mesh_to_pycutfem(
    *,
    mdpa: MDPAMesh,
    domain_tag: str,
    boundary_condition_tags: dict[str, str],
    boundary_node_tags: dict[str, str] | None = None,
) -> Mesh:
    old_node_ids = sorted(int(node_id) for node_id in mdpa.nodes)
    old_to_new = {int(node_id): idx for idx, node_id in enumerate(old_node_ids)}
    new_to_old = np.asarray(old_node_ids, dtype=int)
    nodes = [
        Node(idx, float(mdpa.nodes[int(node_id)][0]), float(mdpa.nodes[int(node_id)][1]))
        for idx, node_id in enumerate(old_node_ids)
    ]

    element_ids = sorted(int(elem_id) for elem_id in mdpa.elements)
    raw_connectivity = np.asarray(
        [[old_to_new[int(node_id)] for node_id in mdpa.elements[int(elem_id)]] for elem_id in element_ids],
        dtype=int,
    )
    coords_all = np.asarray([[node.x, node.y] for node in nodes], dtype=float)
    corner_connectivity = raw_connectivity.copy()
    if corner_connectivity.ndim != 2 or corner_connectivity.shape[0] == 0:
        raise ValueError(f"No elements found in {mdpa.path}")

    nloc = int(corner_connectivity.shape[1])
    if nloc == 3:
        element_type = "tri"
        pts = np.asarray(coords_all[corner_connectivity], dtype=float)
        signed_area2 = (
            (pts[:, 1, 0] - pts[:, 0, 0]) * (pts[:, 2, 1] - pts[:, 0, 1])
            - (pts[:, 1, 1] - pts[:, 0, 1]) * (pts[:, 2, 0] - pts[:, 0, 0])
        )
        flip = np.asarray(signed_area2 < 0.0, dtype=bool)
        if np.any(flip):
            corner_connectivity[flip] = corner_connectivity[flip][:, [0, 2, 1]]
        element_connectivity = corner_connectivity.copy()
    elif nloc == 4:
        element_type = "quad"
        pts = np.asarray(coords_all[corner_connectivity], dtype=float)
        center = np.mean(pts, axis=1, keepdims=True)
        angles = np.arctan2(pts[:, :, 1] - center[:, :, 1], pts[:, :, 0] - center[:, :, 0])
        ccw_order = np.argsort(angles, axis=1)
        conn_ccw = np.take_along_axis(corner_connectivity, ccw_order, axis=1)
        pts_ccw = np.take_along_axis(pts, ccw_order[:, :, None], axis=1)
        x_coords = pts_ccw[:, :, 0]
        y_coords = pts_ccw[:, :, 1]
        # pycutfem Q1 quads use a stable corner convention:
        #   corner_connectivity = [bl, br, tr, tl]
        #   elements_connectivity = [bl, br, tl, tr]
        # Choosing the start corner by minimum y alone is not robust for
        # distorted quads with a slightly tilted bottom edge, where the true
        # bottom-left corner can have a marginally larger y than bottom-right.
        # Instead, pick the CCW corner with minimum x+y, tie-breaking by x.
        start = np.lexsort((x_coords, x_coords + y_coords))[:, 0]
        rolled_order = (start[:, None] + np.arange(4, dtype=int)[None, :]) % 4
        corner_connectivity = np.take_along_axis(conn_ccw, rolled_order, axis=1)
        element_connectivity = corner_connectivity[:, [0, 1, 3, 2]].copy()
    else:
        raise ValueError(f"Unsupported MDPA element arity {nloc} in {mdpa.path}")

    mesh = Mesh(
        nodes=nodes,
        element_connectivity=element_connectivity,
        elements_corner_nodes=corner_connectivity,
        element_type=element_type,
        poly_order=1,
    )
    mesh._mdpa_old_to_new_node = dict(old_to_new)
    mesh._mdpa_new_to_old_node = new_to_old.copy()
    for elem in mesh.elements_list:
        elem.tag = str(domain_tag)
    mesh._elem_bitsets = {str(domain_tag): BitSet(np.ones(len(mesh.elements_list), dtype=bool))}

    condition_edge_tags: dict[frozenset[int], str] = {}
    for part_name, tag in boundary_condition_tags.items():
        part = mdpa.submodelparts.get(str(part_name))
        if part is None:
            continue
        for condition_id in part.condition_ids:
            condition_nodes = tuple(int(node_id) for node_id in mdpa.conditions.get(int(condition_id), ()))
            if len(condition_nodes) != 2:
                continue
            condition_edge_tags[frozenset(condition_nodes)] = str(tag)

    boundary_node_sets: list[tuple[str, set[int]]] = []
    for part_name, tag in (boundary_node_tags or {}).items():
        part = mdpa.submodelparts.get(str(part_name))
        if part is None:
            continue
        boundary_node_sets.append((str(tag), {int(node_id) for node_id in part.node_ids}))

    for edge in mesh.edges_list:
        if edge.right is not None:
            continue
        edge_endpoint_ids = frozenset(int(new_to_old[int(node_id)]) for node_id in edge.nodes)
        tag = condition_edge_tags.get(edge_endpoint_ids)
        if tag is None:
            edge_all_nodes = {int(new_to_old[int(node_id)]) for node_id in edge.nodes}
            for node_tag, node_set in boundary_node_sets:
                if edge_all_nodes.issubset(node_set):
                    tag = node_tag
                    break
        if tag is not None:
            edge.tag = str(tag)

    mesh.rebuild_edge_bitsets()
    return mesh


def _load_reference_partitioned_meshes(*, setup) -> tuple[Mesh, Mesh]:
    geometry = setup.geometry
    fluid_mesh = _mdpa_mesh_to_pycutfem(
        mdpa=setup.reference.fluid,
        domain_tag="fluid",
        boundary_condition_tags={
            "AutomaticInlet2D_Inlet": geometry.inlet_tag,
            "Outlet2D_Outlet": geometry.outlet_tag,
            "NoSlip2D_Top": geometry.walls_tag,
            "NoSlip2D_Bottom": geometry.walls_tag,
            "NoSlip2D_Cylinder": geometry.cylinder_tag,
            "NoSlip2D_Interface": geometry.interface_tag,
        },
    )
    solid_mesh = _mdpa_mesh_to_pycutfem(
        mdpa=setup.reference.solid,
        domain_tag="solid",
        boundary_condition_tags={
            "StructureInterface2D_Struc_Fsi": geometry.interface_tag,
        },
        boundary_node_tags={
            "DISPLACEMENT_BCDisp": geometry.clamp_tag,
        },
    )
    return fluid_mesh, solid_mesh


def _build_scalar_problem(mesh: Mesh, field: str, order: int) -> tuple[MixedElement, DofHandler, TrialFunction, TestFunction, Function]:
    me = MixedElement(mesh, field_specs={field: int(order)})
    dh = DofHandler(me, method="cg")
    trial = TrialFunction(name=f"d{field}", field_name=field, dof_handler=dh)
    test = TestFunction(name=f"{field}_test", field_name=field, dof_handler=dh)
    fn = Function(name=f"{field}_k", field_name=field, dof_handler=dh)
    fn.nodal_values.fill(0.0)
    return me, dh, trial, test, fn


def _build_vector_problem(
    mesh: Mesh,
    *,
    prefix: str,
    order: int,
) -> tuple[MixedElement, DofHandler, VectorTrialFunction, VectorTestFunction, VectorFunction]:
    field_names = [f"{prefix}x", f"{prefix}y"]
    me = MixedElement(mesh, field_specs={field_names[0]: int(order), field_names[1]: int(order)})
    dh = DofHandler(me, method="cg")
    space = FunctionSpace(name=f"{prefix.upper()}Space", field_names=field_names, dim=1)
    trial = VectorTrialFunction(space=space, dof_handler=dh)
    test = VectorTestFunction(space=space, dof_handler=dh)
    fn = VectorFunction(name=f"{prefix}_k", field_names=field_names, dof_handler=dh)
    fn.nodal_values.fill(0.0)
    return me, dh, trial, test, fn


def _build_fluid_problem(
    mesh: Mesh,
    *,
    poly_order: int,
    pressure_order: int,
    mesh_order: int | None = None,
    quadrature_order: int | None = None,
) -> dict[str, object]:
    mesh_order_value = int(mesh_order if mesh_order is not None else poly_order)
    me = MixedElement(
        mesh,
        field_specs={
            "ux": int(poly_order),
            "uy": int(poly_order),
            "p": int(pressure_order),
            "mx": int(mesh_order_value),
            "my": int(mesh_order_value),
        },
    )
    dh = DofHandler(me, method="cg")
    v_space = FunctionSpace("FluidVelocity", ["ux", "uy"], dim=1)
    m_space = FunctionSpace("FluidMesh", ["mx", "my"], dim=1)
    du = VectorTrialFunction(space=v_space, dof_handler=dh)
    v = VectorTestFunction(space=v_space, dof_handler=dh)
    dp = TrialFunction(name="dp", field_name="p", dof_handler=dh)
    q = TestFunction(name="q", field_name="p", dof_handler=dh)
    dm = VectorTrialFunction(space=m_space, dof_handler=dh)
    z = VectorTestFunction(space=m_space, dof_handler=dh)
    u_k = VectorFunction("u_k", ["ux", "uy"], dof_handler=dh)
    u_prev = VectorFunction("u_prev", ["ux", "uy"], dof_handler=dh)
    a_prev = VectorFunction("a_prev", ["ux", "uy"], dof_handler=dh)
    a_k = VectorFunction("a_k", ["ux", "uy"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    p_prev = Function("p_prev", "p", dof_handler=dh)
    d_mesh = VectorFunction("d_mesh", ["mx", "my"], dof_handler=dh)
    d_prev = VectorFunction("d_prev", ["mx", "my"], dof_handler=dh)
    d_prev2 = VectorFunction("d_prev2", ["mx", "my"], dof_handler=dh)
    w_mesh_prev = VectorFunction("w_mesh_prev", ["mx", "my"], dof_handler=dh)
    a_mesh_prev = VectorFunction("a_mesh_prev", ["mx", "my"], dof_handler=dh)
    w_mesh_k = VectorFunction("w_mesh_k", ["mx", "my"], dof_handler=dh)
    a_mesh_k = VectorFunction("a_mesh_k", ["mx", "my"], dof_handler=dh)
    for function in (u_k, u_prev, a_prev, a_k, d_mesh, d_prev, d_prev2, w_mesh_prev, a_mesh_prev, w_mesh_k, a_mesh_k):
        function.nodal_values.fill(0.0)
    for function in (p_k, p_prev):
        function.nodal_values.fill(0.0)
    quad_order_value = (
        int(quadrature_order)
        if quadrature_order is not None
        else int(_EX2L_KRATOS_MATCHED_QUAD_ORDER)
    )
    dvms_state = _build_fluid_dvms_state(mesh, quadrature_order=quad_order_value, dof_handler=dh)
    return {
        "me": me,
        "dh": dh,
        "du": du,
        "v": v,
        "dp": dp,
        "q": q,
        "dm": dm,
        "z": z,
        "u_k": u_k,
        "u_prev": u_prev,
        "a_prev": a_prev,
        "a_k": a_k,
        "p_k": p_k,
        "p_prev": p_prev,
        "d_mesh": d_mesh,
        "d_prev": d_prev,
        "d_prev2": d_prev2,
        "w_mesh_prev": w_mesh_prev,
        "a_mesh_prev": a_mesh_prev,
        "w_mesh_k": w_mesh_k,
        "a_mesh_k": a_mesh_k,
        "dvms_state": dvms_state,
        "velocity_order": int(poly_order),
        "pressure_order": int(pressure_order),
        "mesh_order": int(mesh_order_value),
    }


def _build_mesh_extension_problem(mesh: Mesh, *, poly_order: int) -> dict[str, object]:
    me, dh, dm, z, m_k = _build_vector_problem(mesh, prefix="m", order=poly_order)
    m_prev_geom = VectorFunction("m_prev_geom", ["mx", "my"], dof_handler=dh)
    m_prev_geom.nodal_values.fill(0.0)
    return {
        "me": me,
        "dh": dh,
        "dm": dm,
        "z": z,
        "m_k": m_k,
        "m_prev_geom": m_prev_geom,
        "_cpp_mesh_strategy_backend": None,
    }


def _boundary_field_node_ids(dh: DofHandler, field: str, tag: str) -> np.ndarray:
    _, gdofs = _boundary_field_data(dh, field, tag)
    node_ids: list[int] = []
    seen: set[int] = set()
    for gdof in np.asarray(gdofs, dtype=int).reshape(-1):
        _field_name, node_id = dh._dof_to_node_map[int(gdof)]
        if node_id is None:
            continue
        node_id_int = int(node_id)
        if node_id_int in seen:
            continue
        seen.add(node_id_int)
        node_ids.append(node_id_int)
    return np.asarray(node_ids, dtype=int)


def _node_matrix_from_vector_function(dh: DofHandler, vector: VectorFunction) -> np.ndarray:
    num_nodes = int(len(dh.mixed_element.mesh.nodes_list))
    out = np.zeros((num_nodes, 2), dtype=float)
    for component_idx, component in enumerate(vector.components[:2]):
        gdofs = np.asarray(dh.get_field_slice(component.field_name), dtype=int).reshape(-1)
        values = np.asarray(component.get_nodal_values(gdofs), dtype=float).reshape(-1)
        for local_idx, gdof in enumerate(gdofs):
            _field_name, node_id = dh._dof_to_node_map[int(gdof)]
            if node_id is None:
                continue
            out[int(node_id), int(component_idx)] = float(values[int(local_idx)])
    return out


def _global_dof_vector_from_function(
    dh: DofHandler,
    function: Function | VectorFunction,
) -> np.ndarray:
    """Gather values in true global DOF order.

    Function/VectorFunction storage follows the object's `_g_dofs` order, which
    is not guaranteed to coincide with the raw global numbering.
    """
    g_dofs = np.asarray(getattr(function, "_g_dofs", np.empty((0,), dtype=int)), dtype=int).reshape(-1)
    values = np.asarray(getattr(function, "nodal_values", np.empty((0,), dtype=float)), dtype=float).reshape(-1)
    if g_dofs.size != values.size:
        raise ValueError(
            f"Function '{getattr(function, 'name', '<unnamed>')}' has {values.size} stored values but {g_dofs.size} mapped DOFs."
        )
    out = np.zeros(int(dh.total_dofs), dtype=float)
    if g_dofs.size:
        out[g_dofs] = values
    return out


def _build_cpp_mesh_strategy_backend(
    *,
    prob: dict[str, object],
    interface_tag: str,
    fixed_tags: tuple[str, ...],
) -> dict[str, object]:
    cached = prob.get("_cpp_mesh_strategy_backend")
    fixed_tags_norm = tuple(str(tag) for tag in fixed_tags)
    if isinstance(cached, dict):
        cached_key = (
            str(cached.get("interface_tag", "")),
            tuple(str(tag) for tag in cached.get("fixed_tags", ())),
        )
        if cached_key == (str(interface_tag), fixed_tags_norm):
            return cached

    dh: DofHandler = prob["dh"]
    mesh: Mesh = dh.mixed_element.mesh
    connectivity_data = getattr(mesh, "corner_connectivity", None)
    if connectivity_data is None:
        connectivity_data = getattr(mesh, "elements_connectivity", None)
    if connectivity_data is None:
        raise AttributeError("Mesh does not expose corner_connectivity/elements_connectivity for the C++ mesh strategy backend.")

    interface_node_ids = _boundary_field_node_ids(dh, "mx", str(interface_tag))
    fixed_chunks = [
        _boundary_field_node_ids(dh, "mx", str(tag))
        for tag in fixed_tags_norm
    ]
    fixed_node_ids = (
        np.unique(np.concatenate([chunk for chunk in fixed_chunks if chunk.size]).astype(int, copy=False))
        if any(chunk.size for chunk in fixed_chunks)
        else np.empty((0,), dtype=int)
    )

    strategy = StructuralMeshMotionStrategy(
        node_coords=np.asarray(mesh.nodes_x_y_pos[:, :2], dtype=float),
        connectivity=np.asarray(connectivity_data, dtype=np.int64),
        fixed_node_ids=np.asarray(fixed_node_ids, dtype=np.int64),
        interface_node_ids=np.asarray(interface_node_ids, dtype=np.int64),
        settings=StructuralMeshMotionStrategySettings(poisson=0.3, factor=100.0, xi=1.5),
    )
    backend = {
        "strategy": strategy,
        "interface_tag": str(interface_tag),
        "fixed_tags": fixed_tags_norm,
        "interface_node_ids": np.asarray(interface_node_ids, dtype=int),
        "interface_coords": np.asarray(mesh.nodes_x_y_pos[np.asarray(interface_node_ids, dtype=int), :2], dtype=float),
        "node_coords": np.asarray(mesh.nodes_x_y_pos[:, :2], dtype=float),
    }
    prob["_cpp_mesh_strategy_backend"] = backend
    return backend


def _build_solid_problem(mesh: Mesh, *, poly_order: int) -> dict[str, object]:
    me, dh, dd, w, d_k = _build_vector_problem(mesh, prefix="d", order=poly_order)
    d_prev = VectorFunction("d_prev", ["dx", "dy"], dof_handler=dh)
    d_prev.nodal_values.fill(0.0)
    return {
        "me": me,
        "dh": dh,
        "dd": dd,
        "w": w,
        "d_k": d_k,
        "d_prev": d_prev,
    }


def _build_porous_solid_problem(
    mesh: Mesh,
    *,
    displacement_order: int,
    pressure_order: int,
) -> dict[str, object]:
    me = MixedElement(
        mesh,
        field_specs={
            "dx": int(displacement_order),
            "dy": int(displacement_order),
            "pl": int(pressure_order),
        },
    )
    dh = DofHandler(me, method="cg")
    d_space = FunctionSpace("PorousDisplacement", ["dx", "dy"], dim=1)
    dd = VectorTrialFunction(space=d_space, dof_handler=dh)
    w = VectorTestFunction(space=d_space, dof_handler=dh)
    dp = TrialFunction(name="dpl", field_name="pl", dof_handler=dh)
    q = TestFunction(name="qpl", field_name="pl", dof_handler=dh)
    d_k = VectorFunction("d_k", ["dx", "dy"], dof_handler=dh)
    d_prev = VectorFunction("d_prev", ["dx", "dy"], dof_handler=dh)
    p_k = Function("pl_k", "pl", dof_handler=dh)
    p_prev = Function("pl_prev", "pl", dof_handler=dh)
    for function in (d_k, d_prev):
        function.nodal_values.fill(0.0)
    for function in (p_k, p_prev):
        function.nodal_values.fill(0.0)
    return {
        "me": me,
        "dh": dh,
        "dd": dd,
        "w": w,
        "dp": dp,
        "q": q,
        "d_k": d_k,
        "d_prev": d_prev,
        "p_k": p_k,
        "p_prev": p_prev,
    }


def _solid_current_functions(solid: dict[str, object]) -> list[Function | VectorFunction]:
    functions: list[Function | VectorFunction] = [solid["d_k"]]
    if "p_k" in solid:
        functions.append(solid["p_k"])
    return functions


def _solid_prev_functions(solid: dict[str, object]) -> list[Function | VectorFunction]:
    functions: list[Function | VectorFunction] = [solid["d_prev"]]
    if "p_prev" in solid:
        functions.append(solid["p_prev"])
    return functions


def _solid_current_and_prev_functions(solid: dict[str, object]) -> list[Function | VectorFunction]:
    return _solid_current_functions(solid) + _solid_prev_functions(solid)


def _default_upl_material_from_lame(
    *,
    mu_s: float,
    lambda_s: float,
    porosity: float,
    biot_coefficient: float,
    permeability: float,
    storage_inverse: float,
    dynamic_viscosity_liquid: float,
    density_solid: float,
    density_liquid: float,
) -> UPlMaterial2D:
    mu = float(mu_s)
    lam = float(lambda_s)
    if mu <= 0.0:
        raise ValueError("porous solid shear modulus must be positive.")
    denom = max(2.0 * (lam + mu), 1.0e-14)
    poisson = lam / denom
    young = mu * (3.0 * lam + 2.0 * mu) / max(lam + mu, 1.0e-14)
    return UPlMaterial2D(
        young_modulus=float(young),
        poisson_ratio=float(poisson),
        porosity=float(porosity),
        biot_coefficient=float(biot_coefficient),
        permeability_xx=float(permeability),
        permeability_yy=float(permeability),
        dynamic_viscosity_liquid=float(dynamic_viscosity_liquid),
        storage_inverse=float(storage_inverse),
        density_solid=float(density_solid),
        density_liquid=float(density_liquid),
    )


def _boundary_field_data(dh: DofHandler, field: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    cache = getattr(dh, "_nirb_boundary_field_data_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(dh, "_nirb_boundary_field_data_cache", cache)
    cache_key = (str(field), str(tag))
    cached = cache.get(cache_key)
    if cached is not None:
        coords_cached, gdofs_cached = cached
        return np.asarray(coords_cached, dtype=float), np.asarray(gdofs_cached, dtype=int)

    dh._ensure_dof_coords()
    mesh = dh.mixed_element.mesh
    boundary_node_ids: list[int] = []
    seen_nodes: set[int] = set()
    boundary_points: list[np.ndarray] = []
    boundary_segments: list[tuple[np.ndarray, np.ndarray]] = []
    for eid in mesh.edge_bitset(tag).to_indices():
        edge = mesh.edge(int(eid))
        if edge.right is not None:
            continue
        node_ids = list(getattr(edge, "all_nodes", None) or edge.nodes)
        for node_id in node_ids:
            node_id_int = int(node_id)
            if node_id_int not in seen_nodes:
                seen_nodes.add(node_id_int)
                boundary_node_ids.append(node_id_int)
        pts = np.asarray(mesh.nodes_x_y_pos[node_ids], dtype=float)
        for point in pts:
            boundary_points.append(np.asarray(point, dtype=float))
        if pts.shape[0] >= 2:
            boundary_segments.append((pts[0], pts[-1]))
    if not boundary_points:
        out = (np.empty((0, 2), dtype=float), np.empty((0,), dtype=int))
        cache[cache_key] = out
        return out

    field_ids = np.asarray(dh.get_field_slice(field), dtype=int)
    field_coords = np.asarray(dh._dof_coords[field_ids], dtype=float)
    if boundary_node_ids:
        boundary_node_set = set(int(v) for v in boundary_node_ids)
        keep_node = []
        for idx, gdof in enumerate(field_ids):
            try:
                _field_name, node_id = dh._dof_to_node_map[int(gdof)]
            except Exception:
                node_id = None
            if node_id is not None and int(node_id) in boundary_node_set:
                keep_node.append(idx)
        if keep_node:
            keep_arr = np.asarray(keep_node, dtype=int)
            out = (field_coords[keep_arr], field_ids[keep_arr])
            cache[cache_key] = out
            return out

    points_arr = np.asarray(boundary_points, dtype=float)
    span = np.ptp(np.asarray(mesh.nodes_x_y_pos, dtype=float), axis=0)
    span_max = float(np.max(span)) if span.size else 1.0
    tol_sq = (1.0e-8 * max(span_max, 1.0)) ** 2
    keep: list[int] = []
    for idx, xy in enumerate(field_coords):
        dist2_pts = np.sum((points_arr - xy[None, :]) ** 2, axis=1)
        on_boundary = bool(np.min(dist2_pts) <= tol_sq)
        if not on_boundary:
            for a, b in boundary_segments:
                ab = b - a
                denom = float(np.dot(ab, ab))
                if denom <= 1.0e-20:
                    continue
                t = float(np.clip(np.dot(xy - a, ab) / denom, 0.0, 1.0))
                proj = a + t * ab
                if float(np.dot(xy - proj, xy - proj)) <= tol_sq:
                    on_boundary = True
                    break
        if on_boundary:
            keep.append(idx)
    if not keep:
        out = (np.empty((0, 2), dtype=float), np.empty((0,), dtype=int))
        cache[cache_key] = out
        return out
    out = (field_coords[np.asarray(keep, dtype=int)], field_ids[np.asarray(keep, dtype=int)])
    cache[cache_key] = out
    return out


def _vector_field_matrix(dh: DofHandler, vector: VectorFunction) -> tuple[np.ndarray, np.ndarray]:
    dh._ensure_dof_coords()
    x_ids = np.asarray(dh.get_field_slice(vector.components[0].field_name), dtype=int)
    y_ids = np.asarray(dh.get_field_slice(vector.components[1].field_name), dtype=int)
    coords = np.asarray(dh._dof_coords[x_ids], dtype=float)
    values = np.column_stack(
        [
            np.asarray(vector.components[0].get_nodal_values(x_ids), dtype=float),
            np.asarray(vector.components[1].get_nodal_values(y_ids), dtype=float),
        ]
    )
    return coords, values


def _vector_lookup_from_field(dh: DofHandler, vector: VectorFunction) -> CoordinateLookup:
    coords, values = _vector_field_matrix(dh, vector)
    return CoordinateLookup(coords, values, dim=2)


def _vector_point_data_from_function(dh: DofHandler, vector: VectorFunction) -> np.ndarray:
    num_nodes = int(len(dh.mixed_element.mesh.nodes_list))
    point_data = np.zeros((num_nodes, 2), dtype=float)
    for component_idx, component in enumerate(vector.components[:2]):
        gdofs = np.asarray(dh.get_field_slice(component.field_name), dtype=int).reshape(-1)
        if gdofs.size == 0:
            continue
        values = np.asarray(component.get_nodal_values(gdofs), dtype=float).reshape(-1)
        for local_idx, gdof in enumerate(gdofs):
            _field, node_id = dh._dof_to_node_map[int(gdof)]
            if node_id is None:
                continue
            point_data[int(node_id), int(component_idx)] = float(values[int(local_idx)])
    return point_data


def _nodal_vector_point_data_from_global_values(
    dh: DofHandler,
    *,
    vector: VectorFunction,
    global_values: np.ndarray,
) -> np.ndarray:
    dof_values = np.asarray(global_values, dtype=float).reshape(-1)
    if dof_values.shape[0] != int(dh.total_dofs):
        raise ValueError(f"Expected {dh.total_dofs} dof values, got {dof_values.shape[0]}.")
    num_nodes = int(len(dh.mixed_element.mesh.nodes_list))
    point_data = np.zeros((num_nodes, 2), dtype=float)
    for component_idx, component in enumerate(vector.components):
        for gdof in dh.get_field_slice(component.field_name):
            _field, node_id = dh._dof_to_node_map[int(gdof)]
            if node_id is None:
                continue
            point_data[int(node_id), int(component_idx)] = float(dof_values[int(gdof)])
    return point_data


def _boundary_vector_snapshot(dh: DofHandler, vector: VectorFunction, tag: str) -> tuple[np.ndarray, np.ndarray]:
    coords_x, gdofs_x = _boundary_field_data(dh, vector.components[0].field_name, tag)
    coords_y, gdofs_y = _boundary_field_data(dh, vector.components[1].field_name, tag)
    if coords_x.shape != coords_y.shape:
        raise RuntimeError(f"Mismatched boundary DOF counts for tag {tag!r}: {coords_x.shape} vs {coords_y.shape}")
    if coords_x.size and not np.allclose(coords_x, coords_y):
        raise RuntimeError(f"Mismatched vector boundary coordinate ordering for tag {tag!r}")
    values = np.column_stack(
        [
            np.asarray(vector.components[0].get_nodal_values(gdofs_x), dtype=float),
            np.asarray(vector.components[1].get_nodal_values(gdofs_y), dtype=float),
        ]
    )
    return coords_x, values


def _flatten_vector_snapshot(dh: DofHandler, vector: VectorFunction) -> np.ndarray:
    _, values = _vector_field_matrix(dh, vector)
    return np.asarray(values, dtype=float).reshape(-1)


def _scalar_point_data_from_function(dh: DofHandler, function: Function) -> np.ndarray:
    num_nodes = int(len(dh.mixed_element.mesh.nodes_list))
    point_data = np.zeros((num_nodes, 1), dtype=float)
    gdofs = np.asarray(dh.get_field_slice(function.field_name), dtype=int).reshape(-1)
    if gdofs.size == 0:
        return point_data
    values = np.asarray(function.get_nodal_values(gdofs), dtype=float).reshape(-1)
    for local_idx, gdof in enumerate(gdofs):
        _field, node_id = dh._dof_to_node_map[int(gdof)]
        if node_id is None:
            continue
        point_data[int(node_id), 0] = float(values[int(local_idx)])
    return point_data


def _mesh_node_ids(mesh: Mesh) -> np.ndarray:
    mapped = getattr(mesh, "_mdpa_new_to_old_node", None)
    if mapped is not None:
        arr = np.asarray(mapped, dtype=int).reshape(-1)
        if arr.size == len(mesh.nodes_list):
            return arr.copy()
    return np.arange(1, int(len(mesh.nodes_list)) + 1, dtype=int)


def _structure_kratos_active_dof_permutation(
    *,
    dh: DofHandler,
    mesh: Mesh,
    active_dofs: np.ndarray,
) -> np.ndarray | None:
    active = np.asarray(active_dofs, dtype=int).reshape(-1)
    if active.size == 0:
        return np.empty((0,), dtype=int)

    old_node_ids = np.asarray(_mesh_node_ids(mesh), dtype=int).reshape(-1)
    items: list[tuple[int, int, int]] = []
    for red_idx, gdof in enumerate(active.tolist()):
        field_name, node_id = dh._dof_to_node_map[int(gdof)]
        if node_id is None:
            return None
        if str(field_name) == "dx":
            var_ord = 0
        elif str(field_name) == "dy":
            var_ord = 1
        else:
            return None
        node_int = int(node_id)
        if node_int < 0 or node_int >= old_node_ids.size:
            return None
        items.append((int(old_node_ids[node_int]), int(var_ord), int(red_idx)))

    perm = np.asarray([int(red_idx) for _, _, red_idx in sorted(items)], dtype=int)
    if perm.size != active.size:
        return None
    return perm


def _map_kratos_structure_linear_solver_backend(solver_type: str) -> str:
    solver_name = str(solver_type or "").strip().lower()
    if "sparse_qr" in solver_name:
        # Kratos' LinearSolversApplication.sparse_qr is Eigen::SparseQR with
        # COLAMD ordering, not SuiteSparse SPQR.
        return "eigen_sparseqr"
    if solver_name.endswith("amgcl") or ".amgcl" in solver_name or "amgcl" in solver_name:
        return "amgcl"
    if "petsc" in solver_name:
        return "petsc"
    if "scipy" in solver_name or "klu" in solver_name:
        return "scipy"
    return "petsc"


def _load_kratos_structure_solver_profile(*, benchmark_root: Path) -> dict[str, object]:
    benchmark_root = Path(benchmark_root).resolve()
    settings_data = _load_json(benchmark_root / "ProjectParametersCSM.json")
    solver_settings = dict(settings_data.get("solver_settings", {}))
    linear_solver_settings = dict(solver_settings.get("linear_solver_settings", {}))
    return {
        "convergence_criterion": str(solver_settings.get("convergence_criterion", "residual_criterion")),
        "line_search": bool(solver_settings.get("line_search", False)),
        "residual_absolute_tolerance": float(solver_settings.get("residual_absolute_tolerance", 1.0e-6)),
        "residual_relative_tolerance": float(solver_settings.get("residual_relative_tolerance", 1.0e-6)),
        "displacement_absolute_tolerance": float(solver_settings.get("displacement_absolute_tolerance", 1.0e-6)),
        "displacement_relative_tolerance": float(solver_settings.get("displacement_relative_tolerance", 1.0e-6)),
        "max_iteration": int(solver_settings.get("max_iteration", 25)),
        "linear_solver_type": str(linear_solver_settings.get("solver_type", "")),
        "linear_backend": _map_kratos_structure_linear_solver_backend(
            str(linear_solver_settings.get("solver_type", ""))
        ),
    }


def _solid_system_backend_mode() -> str:
    return str(os.getenv("PYCUTFEM_EX2_SOLID_SYSTEM_BACKEND", "symbolic") or "symbolic").strip().lower()


def _diagnostic_solid_system_backend_mode() -> str:
    return str(os.getenv("PYCUTFEM_EX2_DIAGNOSTIC_SOLID_SYSTEM_BACKEND", "") or "").strip().lower()


def _build_kratos_local_solid_backend(
    *,
    benchmark_root: Path,
    prob: dict[str, object],
) -> dict[str, object]:
    import KratosMultiphysics as KM
    import KratosMultiphysics.StructuralMechanicsApplication as KSM
    from KratosMultiphysics.StructuralMechanicsApplication import python_solvers_wrapper_structural as struct_wrapper

    benchmark_root = Path(benchmark_root).resolve()
    settings_data = _load_json(benchmark_root / "ProjectParametersCSM.json")
    solver_settings = KM.Parameters(json.dumps(settings_data["solver_settings"]))
    solver_settings["echo_level"].SetInt(0)

    model = KM.Model()
    solver = struct_wrapper.CreateSolverByParameters(model, solver_settings, "OpenMP")
    cwd = Path.cwd()
    try:
        os.chdir(benchmark_root)
        solver.AddVariables()
        solver.ImportModelPart()
        solver.PrepareModelPart()
        solver.AddDofs()
        solver.Initialize()
    finally:
        os.chdir(cwd)

    mesh: Mesh = prob["dh"].mixed_element.mesh
    dh: DofHandler = prob["dh"]
    dh._ensure_dof_coords()
    old_node_ids = np.asarray(_mesh_node_ids(mesh), dtype=int).reshape(-1)
    old_id_by_coord = {
        _coord_key(float(x), float(y)): int(old_node_ids[i])
        for i, (x, y) in enumerate(np.asarray(mesh.nodes_x_y_pos[:, :2], dtype=float))
    }

    dx_ids = np.asarray(dh.get_field_slice("dx"), dtype=int)
    dy_ids = np.asarray(dh.get_field_slice("dy"), dtype=int)
    dx_coords = np.asarray(dh._dof_coords[dx_ids], dtype=float)
    dy_coords = np.asarray(dh._dof_coords[dy_ids], dtype=float)
    if dx_coords.shape != dy_coords.shape or (dx_coords.size and not np.allclose(dx_coords, dy_coords)):
        raise RuntimeError("Kratos local solid backend requires matching CG nodal coordinates for dx and dy.")
    field_index_by_old_id = {
        old_id_by_coord[_coord_key(float(x), float(y))]: i
        for i, (x, y) in enumerate(dx_coords.tolist())
    }

    main_model_part = solver.main_model_part
    process_info = main_model_part.ProcessInfo
    node_by_id = {int(node.Id): node for node in main_model_part.Nodes}
    all_nodes = [node_by_id[int(node_id)] for node_id in sorted(node_by_id)]
    point_conditions = [cond for cond in main_model_part.Conditions if cond.GetGeometry().PointsNumber() == 1]

    zero_vec = KM.Array3()
    zero_vec[0] = 0.0
    zero_vec[1] = 0.0
    zero_vec[2] = 0.0
    for node in all_nodes:
        try:
            node.SetSolutionStepValue(KSM.POINT_LOAD, 0, zero_vec)
            node.SetSolutionStepValue(KSM.POINT_LOAD, 1, zero_vec)
        except Exception:
            pass
    for cond in point_conditions:
        try:
            cond.SetValue(KSM.POINT_LOAD, zero_vec)
        except Exception:
            pass

    kratos_elem_by_nodes: dict[frozenset[int], object] = {}
    for elem in main_model_part.Elements:
        geom_ids = frozenset(int(node.Id) for node in elem.GetGeometry())
        kratos_elem_by_nodes[geom_ids] = elem

    n_elem = int(mesh.n_elements)
    gdofs_map = np.empty((n_elem, int(dh.mixed_element.n_dofs_per_elem)), dtype=int)
    local_perm = np.empty((n_elem, 8), dtype=int)
    kratos_elements: list[object] = [None] * n_elem
    for eid in range(n_elem):
        gdofs_map[eid, :] = np.asarray(dh.get_elemental_dofs(eid), dtype=int)
        py_dx_coords = np.asarray(dh.element_dof_coords("dx", eid), dtype=float)
        py_old_ids = [old_id_by_coord[_coord_key(float(x), float(y))] for x, y in py_dx_coords.tolist()]
        elem = kratos_elem_by_nodes.get(frozenset(int(node_id) for node_id in py_old_ids))
        if elem is None:
            raise RuntimeError(f"Could not match pycutfem solid element {eid} to a Kratos element.")
        kratos_elements[eid] = elem
        py_pos_by_old_id = {int(node_id): j for j, node_id in enumerate(py_old_ids)}
        perm = np.empty(8, dtype=int)
        for k, node in enumerate(elem.GetGeometry()):
            py_pos = py_pos_by_old_id[int(node.Id)]
            perm[2 * k] = py_pos
            perm[2 * k + 1] = 4 + py_pos
        local_perm[eid, :] = perm

    eq_to_gdof = np.empty((0,), dtype=int)
    gdof_to_eq = np.empty((0,), dtype=int)

    return {
        "KM": KM,
        "KSM": KSM,
        "solver": solver,
        "main_model_part": main_model_part,
        "process_info": process_info,
        "all_nodes": all_nodes,
        "point_conditions": tuple(point_conditions),
        "field_index_by_old_id": field_index_by_old_id,
        "dx_ids": dx_ids,
        "dy_ids": dy_ids,
        "kratos_elements": tuple(kratos_elements),
        "gdofs_map": np.asarray(gdofs_map, dtype=int),
        "local_perm": np.asarray(local_perm, dtype=int),
        "eq_to_gdof": np.asarray(eq_to_gdof, dtype=int),
        "gdof_to_eq": np.asarray(gdof_to_eq, dtype=int),
    }


def _build_kratos_exact_structure_backend(
    *,
    benchmark_root: Path,
    dt: float,
) -> dict[str, object]:
    import KratosMultiphysics as KM
    import KratosMultiphysics.StructuralMechanicsApplication as KSM
    from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import (
        StructuralMechanicsAnalysis,
    )

    benchmark_root = Path(benchmark_root).resolve()
    settings_data = _load_json(benchmark_root / "ProjectParametersCSM.json")
    settings_data["problem_data"]["echo_level"] = 0
    settings_data["problem_data"]["end_time"] = float(dt)
    settings_data["solver_settings"]["echo_level"] = 0
    if "time_stepping" in settings_data["solver_settings"]:
        settings_data["solver_settings"]["time_stepping"]["time_step"] = float(dt)
    settings_data["output_processes"] = {}

    model = KM.Model()
    analysis = StructuralMechanicsAnalysis(model, KM.Parameters(json.dumps(settings_data)))
    cwd = Path.cwd()
    try:
        os.chdir(benchmark_root)
        analysis.Initialize()
    finally:
        os.chdir(cwd)

    solver = analysis._GetSolver()
    main_model_part = solver.main_model_part
    interface_part = main_model_part.GetSubModelPart("StructureInterface2D_Struc_Fsi")
    all_nodes = list(main_model_part.Nodes)
    interface_nodes = list(interface_part.Nodes)
    node_coords_ref = np.asarray([[float(node.X0), float(node.Y0)] for node in all_nodes], dtype=float)
    interface_coords_ref = np.asarray([[float(node.X0), float(node.Y0)] for node in interface_nodes], dtype=float)

    zero_vec = KM.Array3()
    zero_vec[0] = 0.0
    zero_vec[1] = 0.0
    zero_vec[2] = 0.0
    for node in interface_nodes:
        node.SetSolutionStepValue(KSM.POINT_LOAD, 0, zero_vec)

    return {
        "KM": KM,
        "KSM": KSM,
        "analysis": analysis,
        "solver": solver,
        "main_model_part": main_model_part,
        "all_nodes": all_nodes,
        "interface_nodes": interface_nodes,
        "node_coords_ref": node_coords_ref,
        "interface_coords_ref": interface_coords_ref,
        "time": 0.0,
    }


def _advance_kratos_exact_structure_backend_step(
    *,
    backend: dict[str, object],
) -> None:
    analysis = backend["analysis"]
    solver = backend["solver"]
    current_time = float(backend["time"])
    new_time = float(solver.AdvanceInTime(current_time))
    analysis.time = new_time
    solver.Predict()
    analysis.InitializeSolutionStep()
    backend["time"] = new_time


def _finalize_kratos_exact_structure_backend_step(
    *,
    backend: dict[str, object],
) -> None:
    backend["analysis"].FinalizeSolutionStep()


def _solve_kratos_exact_structure_backend(
    *,
    backend: dict[str, object],
    structure_load: CoordinateLookup,
) -> dict[str, CoordinateLookup]:
    KM = backend["KM"]
    KSM = backend["KSM"]

    for node in backend["interface_nodes"]:
        values = np.asarray(structure_load(float(node.X0), float(node.Y0)), dtype=float).reshape(2)
        load_vec = KM.Array3()
        load_vec[0] = float(values[0])
        load_vec[1] = float(values[1])
        load_vec[2] = 0.0
        node.SetSolutionStepValue(KSM.POINT_LOAD, 0, load_vec)

    backend["solver"].SolveSolutionStep()

    all_nodes = backend["all_nodes"]
    coords = np.asarray(backend["node_coords_ref"], dtype=float)
    interface_nodes = backend["interface_nodes"]
    interface_coords = np.asarray(backend["interface_coords_ref"], dtype=float)
    disp = _kratos_node_step_vector_values(all_nodes, KM.DISPLACEMENT)
    iface_disp = _kratos_node_step_vector_values(interface_nodes, KM.DISPLACEMENT)
    return {
        "displacement": CoordinateLookup(coords, disp, dim=2),
        "interface_displacement": CoordinateLookup(interface_coords, iface_disp, dim=2),
    }


def _maybe_build_kratos_local_solid_backend(
    *,
    benchmark_root: Path,
    prob: dict[str, object],
) -> dict[str, object] | None:
    mode = _solid_system_backend_mode()
    if mode in {"", "symbolic", "none"}:
        return None
    if mode != "kratos_local":
        raise ValueError(f"Unsupported solid-system backend mode {mode!r}.")
    return _build_kratos_local_solid_backend(benchmark_root=benchmark_root, prob=prob)


def _sync_kratos_local_solid_backend_state(
    *,
    backend: dict[str, object],
    d_k: VectorFunction,
) -> None:
    KM = backend["KM"]
    dx_ids = np.asarray(backend["dx_ids"], dtype=int)
    dy_ids = np.asarray(backend["dy_ids"], dtype=int)
    field_index_by_old_id = dict(backend["field_index_by_old_id"])
    ux_vals = np.asarray(d_k.components[0].get_nodal_values(dx_ids), dtype=float)
    uy_vals = np.asarray(d_k.components[1].get_nodal_values(dy_ids), dtype=float)
    for node in backend["all_nodes"]:
        idx = int(field_index_by_old_id[int(node.Id)])
        ux = float(ux_vals[idx])
        uy = float(uy_vals[idx])
        vec = KM.Array3()
        vec[0] = ux
        vec[1] = uy
        vec[2] = 0.0
        node.X = float(node.X0 + ux)
        node.Y = float(node.Y0 + uy)
        node.Z = 0.0
        node.SetSolutionStepValue(KM.DISPLACEMENT, 0, vec)
        node.SetSolutionStepValue(KM.DISPLACEMENT, 1, vec)


def _set_kratos_local_solid_point_loads(
    *,
    backend: dict[str, object],
    structure_load: CoordinateLookup,
) -> None:
    KM = backend["KM"]
    KSM = backend["KSM"]

    zero_vec = KM.Array3()
    zero_vec[0] = 0.0
    zero_vec[1] = 0.0
    zero_vec[2] = 0.0
    for node in backend["all_nodes"]:
        try:
            node.SetSolutionStepValue(KSM.POINT_LOAD, 0, zero_vec)
            node.SetSolutionStepValue(KSM.POINT_LOAD, 1, zero_vec)
        except Exception:
            pass
    for cond in backend["point_conditions"]:
        try:
            cond.SetValue(KSM.POINT_LOAD, zero_vec)
        except Exception:
            pass

    for cond in backend["point_conditions"]:
        node = cond.GetGeometry()[0]
        values = np.asarray(structure_load(float(node.X0), float(node.Y0)), dtype=float).reshape(2)
        load_vec = KM.Array3()
        load_vec[0] = float(values[0])
        load_vec[1] = float(values[1])
        load_vec[2] = 0.0
        try:
            node.SetSolutionStepValue(KSM.POINT_LOAD, 0, load_vec)
            node.SetSolutionStepValue(KSM.POINT_LOAD, 1, load_vec)
        except Exception:
            pass


def _ensure_kratos_local_solid_equation_map(
    *,
    backend: dict[str, object],
) -> np.ndarray:
    gdof_to_eq_cached = np.asarray(backend.get("gdof_to_eq", np.empty((0,), dtype=int)), dtype=int).reshape(-1)
    if gdof_to_eq_cached.size:
        return gdof_to_eq_cached

    KM = backend["KM"]
    field_index_by_old_id = dict(backend["field_index_by_old_id"])
    dx_ids = np.asarray(backend["dx_ids"], dtype=int)
    dy_ids = np.asarray(backend["dy_ids"], dtype=int)
    eq_entries: list[tuple[int, int]] = []
    for node in backend["all_nodes"]:
        field_idx = int(field_index_by_old_id[int(node.Id)])
        if node.HasDofFor(KM.DISPLACEMENT_X):
            eq_entries.append((int(node.GetDof(KM.DISPLACEMENT_X).EquationId), int(dx_ids[field_idx])))
        if node.HasDofFor(KM.DISPLACEMENT_Y):
            eq_entries.append((int(node.GetDof(KM.DISPLACEMENT_Y).EquationId), int(dy_ids[field_idx])))
    ndof = max((int(eq_id) for eq_id, _ in eq_entries), default=-1) + 1
    eq_to_gdof = np.full((ndof,), -1, dtype=int)
    for eq_id, gdof in eq_entries:
        eq_to_gdof[int(eq_id)] = int(gdof)
    if ndof > 0 and np.any(eq_to_gdof < 0):
        raise RuntimeError("Incomplete Kratos local solid equation-to-gdof mapping.")
    total_gdofs = 1 + max((int(gdof) for _eq_id, gdof in eq_entries), default=-1)
    gdof_to_eq = np.full((total_gdofs,), -1, dtype=int)
    if ndof > 0 and total_gdofs > 0:
        gdof_to_eq[eq_to_gdof] = np.arange(ndof, dtype=int)
    backend["eq_to_gdof"] = np.asarray(eq_to_gdof, dtype=int)
    backend["gdof_to_eq"] = np.asarray(gdof_to_eq, dtype=int)
    return np.asarray(gdof_to_eq, dtype=int)


def _assemble_kratos_local_solid_system_batch(
    *,
    backend: dict[str, object],
    d_k: VectorFunction,
    structure_load: CoordinateLookup | None = None,
    need_matrix: bool,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    _sync_kratos_local_solid_backend_state(backend=backend, d_k=d_k)
    if structure_load is not None:
        _set_kratos_local_solid_point_loads(
            backend=backend,
            structure_load=structure_load,
        )
    KM = backend["KM"]
    perm_map = np.asarray(backend["local_perm"], dtype=int)
    gdofs_map = np.asarray(backend["gdofs_map"], dtype=int)
    elements = tuple(backend["kratos_elements"])
    process_info = backend["process_info"]

    n_elem = int(len(elements))
    K_elem = np.zeros((n_elem, 8, 8), dtype=float) if bool(need_matrix) else None
    F_elem = np.zeros((n_elem, 8), dtype=float)
    for e, elem in enumerate(elements):
        lhs = KM.Matrix()
        rhs = KM.Vector()
        elem.CalculateLocalSystem(lhs, rhs, process_info)
        perm = np.asarray(perm_map[e], dtype=int)
        rhs_kr = np.asarray(rhs, dtype=float).reshape(-1)
        rhs_py = np.zeros((8,), dtype=float)
        rhs_py[perm] = rhs_kr
        # Kratos local RHS uses external-internal; pycutfem residual assembly
        # expects internal-external before the point-load shift operator runs.
        F_elem[e, :] = -rhs_py
        if K_elem is not None:
            lhs_kr = np.asarray(lhs, dtype=float)
            lhs_py = np.zeros((8, 8), dtype=float)
            lhs_py[np.ix_(perm, perm)] = lhs_kr
            K_elem[e, :, :] = lhs_py
    element_ids = np.arange(n_elem, dtype=int)
    return K_elem, F_elem, element_ids, gdofs_map


def _assemble_kratos_local_solid_point_condition_batch(
    *,
    backend: dict[str, object],
    need_matrix: bool,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    KM = backend["KM"]
    process_info = backend["process_info"]
    point_conditions = tuple(backend["point_conditions"])
    field_index_by_old_id = dict(backend["field_index_by_old_id"])
    dx_ids = np.asarray(backend["dx_ids"], dtype=int)
    dy_ids = np.asarray(backend["dy_ids"], dtype=int)

    n_cond = int(len(point_conditions))
    K_cond = np.zeros((n_cond, 2, 2), dtype=float) if bool(need_matrix) else None
    F_cond = np.zeros((n_cond, 2), dtype=float)
    gdofs_cond = np.empty((n_cond, 2), dtype=int)
    for c, cond in enumerate(point_conditions):
        lhs = KM.Matrix()
        rhs = KM.Vector()
        cond.CalculateLocalSystem(lhs, rhs, process_info)
        node = cond.GetGeometry()[0]
        field_idx = int(field_index_by_old_id[int(node.Id)])
        gdofs_cond[c, :] = np.asarray([int(dx_ids[field_idx]), int(dy_ids[field_idx])], dtype=int)
        F_cond[c, :] = -np.asarray(rhs, dtype=float).reshape(-1)[:2]
        if K_cond is not None:
            lhs_arr = np.asarray(lhs, dtype=float)
            if lhs_arr.size:
                K_cond[c, :, :] = lhs_arr[:2, :2]
    cond_ids = np.arange(n_cond, dtype=int)
    return K_cond, F_cond, cond_ids, gdofs_cond


def _assemble_kratos_local_solid_system_full(
    *,
    backend: dict[str, object],
    d_k: VectorFunction,
    need_matrix: bool,
):
    import scipy.sparse as sp

    K_elem, F_elem, element_ids, gdofs_map = _assemble_kratos_local_solid_system_batch(
        backend=backend,
        d_k=d_k,
        need_matrix=bool(need_matrix),
    )
    ndof = int(np.asarray(gdofs_map, dtype=int).max()) + 1 if gdofs_map.size else 0
    A_full = sp.lil_matrix((ndof, ndof), dtype=float) if bool(need_matrix) else None
    R_full = np.zeros((ndof,), dtype=float)
    scatter_owner = NewtonSolver.__new__(NewtonSolver)
    A_full, R_full = NewtonSolver.scatter_element_contribs_full(
        scatter_owner,
        K_elem=K_elem,
        F_elem=F_elem,
        element_ids=element_ids,
        gdofs_map=gdofs_map,
        A_full=A_full,
        R_full=R_full,
    )
    return (A_full.tocsr() if A_full is not None else None), R_full


def _assemble_kratos_local_solid_global_system_full(
    *,
    backend: dict[str, object],
    d_k: VectorFunction,
    structure_load: CoordinateLookup,
    need_matrix: bool,
):
    import scipy.sparse as sp
    from KratosMultiphysics import scipy_conversion_tools

    _sync_kratos_local_solid_backend_state(backend=backend, d_k=d_k)
    _set_kratos_local_solid_point_loads(
        backend=backend,
        structure_load=structure_load,
    )

    KM = backend["KM"]
    solver = backend["solver"]
    strategy = solver._GetSolutionStrategy()
    builder = solver._GetBuilderAndSolver()
    scheme = solver._GetScheme()
    computing_model_part = solver.GetComputingModelPart()
    space = KM.UblasSparseSpace()

    A = strategy.GetSystemMatrix()
    b = strategy.GetSystemVector()
    space.SetToZeroVector(b)
    if bool(need_matrix):
        space.SetToZeroMatrix(A)
    builder.Build(scheme, computing_model_part, A, b)

    gdof_to_eq = _ensure_kratos_local_solid_equation_map(backend=backend)
    rhs_py = -np.asarray(b, dtype=float).reshape(-1)[gdof_to_eq]
    A_py = None
    if bool(need_matrix):
        A_raw = scipy_conversion_tools.to_csr(A).copy()
        A_py = A_raw[gdof_to_eq, :][:, gdof_to_eq].tocsr()
    elif gdof_to_eq.size:
        A_py = sp.csr_matrix((gdof_to_eq.size, gdof_to_eq.size), dtype=float)
    else:
        A_py = sp.csr_matrix((0, 0), dtype=float)
    return A_py, rhs_py


class _KratosLocalSolidSystemOperator(RuntimeOperator):
    """Replace the symbolic solid assembly with the exact Kratos local system."""

    def __init__(
        self,
        *,
        backend: dict[str, object],
        d_k: VectorFunction,
        structure_load: CoordinateLookup | None = None,
    ) -> None:
        self.backend = backend
        self.d_k = d_k
        self.structure_load = structure_load

    def after_assembly(self, *, solver, coeffs, A_red, R_red, need_matrix: bool):
        import scipy.sparse as sp

        del coeffs
        K_elem, F_elem, element_ids, gdofs_map = _assemble_kratos_local_solid_system_batch(
            backend=self.backend,
            d_k=self.d_k,
            structure_load=self.structure_load,
            need_matrix=bool(need_matrix),
        )
        A_exact = None
        if bool(need_matrix) and A_red is not None:
            # Preserve the solver-owned reduced sparsity pattern. The runtime
            # scatter helper caches row/column positions against that pattern.
            A_exact = A_red if sp.isspmatrix_csr(A_red) else A_red.tocsr(copy=True)
            A_exact = A_exact.copy()
            A_exact.data.fill(0.0)
        R_exact = np.zeros_like(np.asarray(R_red, dtype=float))
        A_exact, R_exact = solver.scatter_element_contribs_reduced(
            K_elem=K_elem,
            F_elem=F_elem,
            element_ids=element_ids,
            gdofs_map=gdofs_map,
            A_red=A_exact,
            R_red=R_exact,
        )
        if self.structure_load is not None and self.backend["point_conditions"]:
            K_cond, F_cond, cond_ids, gdofs_cond = _assemble_kratos_local_solid_point_condition_batch(
                backend=self.backend,
                need_matrix=bool(need_matrix),
            )
            A_exact, R_exact = solver.scatter_element_contribs_reduced(
                K_elem=K_cond,
                F_elem=F_cond,
                element_ids=cond_ids,
                gdofs_map=gdofs_cond,
                A_red=A_exact,
                R_red=R_exact,
            )
        return A_exact, R_exact


class _KratosLocalSolidGlobalSystemOperator(RuntimeOperator):
    """Replace the reduced solid system with the exact live Kratos global system."""

    def __init__(
        self,
        *,
        backend: dict[str, object],
        d_k: VectorFunction,
        structure_load: CoordinateLookup,
    ) -> None:
        self.backend = backend
        self.d_k = d_k
        self.structure_load = structure_load

    def after_assembly(self, *, solver, coeffs, A_red, R_red, need_matrix: bool):
        del coeffs, A_red, R_red
        A_full, R_full = _assemble_kratos_local_solid_global_system_full(
            backend=self.backend,
            d_k=self.d_k,
            structure_load=self.structure_load,
            need_matrix=bool(need_matrix),
        )
        active = np.asarray(solver.active_dofs, dtype=int).reshape(-1)
        A_exact = A_full[np.ix_(active, active)].tocsr() if bool(need_matrix) else None
        R_exact = np.asarray(R_full, dtype=float).reshape(-1)[active]
        return A_exact, R_exact


def _write_local_step_history(
    *,
    step_history_dir: Path,
    step: int,
    time_s: float,
    mesh_f: Mesh,
    mesh_s: Mesh,
    fluid: dict[str, object],
    solid: dict[str, object],
    interface_load_lookup: CoordinateLookup,
    interface_disp_lookup: CoordinateLookup,
    interface_velocity_lookup: CoordinateLookup,
) -> Path:
    step_history_dir.mkdir(parents=True, exist_ok=True)
    fluid_coords_ref = np.asarray(mesh_f.nodes_x_y_pos, dtype=float)
    fluid_mesh_displacement = _vector_point_data_from_function(fluid["dh"], fluid["d_mesh"])
    fluid_coords_cur = fluid_coords_ref + fluid_mesh_displacement
    fluid_velocity = _vector_point_data_from_function(fluid["dh"], fluid["u_k"])
    fluid_pressure = _scalar_point_data_from_function(fluid["dh"], fluid["p_k"])
    fluid_mesh_velocity = _vector_point_data_from_function(fluid["dh"], fluid["w_mesh_prev"])

    solid_coords_ref = np.asarray(mesh_s.nodes_x_y_pos, dtype=float)
    solid_displacement = _vector_point_data_from_function(solid["dh"], solid["d_k"])
    solid_coords_cur = solid_coords_ref + solid_displacement
    solid_pressure = (
        _scalar_point_data_from_function(solid["dh"], solid["p_k"])
        if "p_k" in solid
        else np.zeros((int(len(mesh_s.nodes_list)), 1), dtype=float)
    )

    step_path = step_history_dir / f"step{int(step):04d}.npz"
    np.savez_compressed(
        step_path,
        step=np.asarray(int(step), dtype=int),
        time_s=np.asarray(float(time_s), dtype=float),
        fluid_node_ids=_mesh_node_ids(mesh_f),
        fluid_coords_ref=fluid_coords_ref,
        fluid_coords_cur=fluid_coords_cur,
        fluid_velocity_nodal_values=fluid_velocity,
        fluid_pressure_nodal_values=fluid_pressure,
        fluid_mesh_displacement_nodal_values=fluid_mesh_displacement,
        fluid_mesh_velocity_nodal_values=fluid_mesh_velocity,
        structure_node_ids=_mesh_node_ids(mesh_s),
        structure_coords_ref=solid_coords_ref,
        structure_coords_cur=solid_coords_cur,
        structure_displacement_nodal_values=solid_displacement,
        structure_liquid_pressure_nodal_values=solid_pressure,
        interface_load_coords_ref=np.asarray(interface_load_lookup.coords, dtype=float),
        interface_load_values=np.asarray(interface_load_lookup.values, dtype=float),
        interface_disp_coords_ref=np.asarray(interface_disp_lookup.coords, dtype=float),
        interface_disp_values=np.asarray(interface_disp_lookup.values, dtype=float),
        interface_velocity_coords_ref=np.asarray(interface_velocity_lookup.coords, dtype=float),
        interface_velocity_values=np.asarray(interface_velocity_lookup.values, dtype=float),
    )
    return step_path


def _build_interface_restriction_matrix(dh: DofHandler, vector: VectorFunction, tag: str) -> np.ndarray:
    full_x_ids = np.asarray(dh.get_field_slice(vector.components[0].field_name), dtype=int)
    _, boundary_x_ids = _boundary_field_data(dh, vector.components[0].field_name, tag)
    positions = {int(gdof): idx for idx, gdof in enumerate(full_x_ids.tolist())}
    flat_indices: list[int] = []
    for gid in boundary_x_ids:
        pos = positions[int(gid)]
        flat_indices.extend([2 * pos, 2 * pos + 1])
    return build_restriction_matrix(flat_indices, 2 * full_x_ids.size)


def _build_interface_mass_matrix(mesh: Mesh, coords: np.ndarray, tag: str) -> np.ndarray:
    iface_coords = np.asarray(coords, dtype=float)
    if iface_coords.ndim != 2 or iface_coords.shape[1] != 2:
        raise ValueError("coords must have shape (n, 2)")
    if iface_coords.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)

    coord_to_idx = {_coord_key(x, y): i for i, (x, y) in enumerate(iface_coords)}
    mass = np.zeros((iface_coords.shape[0], iface_coords.shape[0]), dtype=float)
    for eid in mesh.edge_bitset(tag).to_indices():
        edge = mesh.edge(int(eid))
        if edge.right is not None:
            continue
        node_ids = list(getattr(edge, "all_nodes", None) or edge.nodes)
        if len(node_ids) != 2:
            raise NotImplementedError("Interface load conversion currently supports linear boundary edges only.")
        pts = np.asarray(mesh.nodes_x_y_pos[node_ids], dtype=float)
        ia = coord_to_idx.get(_coord_key(pts[0, 0], pts[0, 1]))
        ib = coord_to_idx.get(_coord_key(pts[1, 0], pts[1, 1]))
        if ia is None or ib is None:
            continue
        length = float(np.linalg.norm(pts[1] - pts[0]))
        local_mass = (length / 6.0) * np.asarray([[2.0, 1.0], [1.0, 2.0]], dtype=float)
        ids = np.asarray([ia, ib], dtype=int)
        mass[np.ix_(ids, ids)] += local_mass
    return mass


def _interface_load_from_traction(mass_matrix: np.ndarray, traction_values: np.ndarray) -> np.ndarray:
    traction = np.asarray(traction_values, dtype=float)
    if traction.ndim != 2 or traction.shape[1] != 2:
        raise ValueError("traction_values must have shape (n, 2)")
    if traction.shape[0] != mass_matrix.shape[0]:
        raise ValueError("mass_matrix / traction_values size mismatch")
    if traction.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(mass_matrix, dtype=float) @ traction


def _interface_traction_from_load(
    mass_matrix: np.ndarray,
    load_values: np.ndarray,
    *,
    regularization: float = 1.0e-12,
) -> np.ndarray:
    load = np.asarray(load_values, dtype=float)
    if load.ndim != 2 or load.shape[1] != 2:
        raise ValueError("load_values must have shape (n, 2)")
    if load.shape[0] != mass_matrix.shape[0]:
        raise ValueError("mass_matrix / load_values size mismatch")
    if load.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)
    reg = float(max(regularization, 0.0))
    operator = np.asarray(mass_matrix, dtype=float)
    if reg > 0.0:
        operator = operator + reg * np.eye(operator.shape[0], dtype=float)
    return np.linalg.solve(operator, load)


def _edge_linear_shape_functions(xi: float) -> np.ndarray:
    xi_value = float(xi)
    return np.asarray([0.5 * (1.0 - xi_value), 0.5 * (1.0 + xi_value)], dtype=float)


def _reference_interface_point_loads_from_lookup(
    *,
    mesh: Mesh,
    iface_coords: np.ndarray,
    tag: str,
    traction_callback: Callable[[np.ndarray, np.ndarray], np.ndarray],
    quad_order: int,
) -> np.ndarray:
    coords = np.asarray(iface_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("iface_coords must have shape (n, 2)")
    if coords.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)

    coord_to_idx = {_coord_key(x, y): i for i, (x, y) in enumerate(coords)}
    loads = np.zeros((coords.shape[0], 2), dtype=float)
    n_edge_q = max(2, int(math.ceil((max(int(quad_order), 1) + 1) / 2)))
    quad_pts, quad_w = np.polynomial.legendre.leggauss(n_edge_q)

    for eid in mesh.edge_bitset(tag).to_indices():
        edge = mesh.edge(int(eid))
        if edge.right is not None:
            continue
        node_ids = list(getattr(edge, "all_nodes", None) or edge.nodes)
        if len(node_ids) != 2:
            raise NotImplementedError("Interface point-load conversion currently supports linear boundary edges only.")
        pts = np.asarray(mesh.nodes_x_y_pos[node_ids], dtype=float)
        ia = coord_to_idx.get(_coord_key(pts[0, 0], pts[0, 1]))
        ib = coord_to_idx.get(_coord_key(pts[1, 0], pts[1, 1]))
        if ia is None or ib is None:
            continue
        length = float(np.linalg.norm(pts[1] - pts[0]))
        if length <= 1.0e-20:
            continue
        jac_line = 0.5 * length
        normal = np.asarray(edge.normal, dtype=float)
        n_norm = float(np.linalg.norm(normal))
        if n_norm <= 1.0e-20:
            continue
        normal /= n_norm
        ids = np.asarray([ia, ib], dtype=int)
        for qp, wq in zip(np.asarray(quad_pts, dtype=float), np.asarray(quad_w, dtype=float)):
            phi = _edge_linear_shape_functions(float(qp))
            X = phi[0] * pts[0] + phi[1] * pts[1]
            traction = np.asarray(traction_callback(X, normal), dtype=float).reshape(2)
            weight = jac_line * float(wq)
            loads[ids, :] += weight * phi[:, None] * traction[None, :]
    return loads


def _boundary_point_load_vector(
    dh: DofHandler,
    *,
    vector: VectorFunction,
    tag: str,
    values: np.ndarray,
) -> np.ndarray:
    coords_x, gdofs_x = _boundary_field_data(dh, vector.components[0].field_name, tag)
    coords_y, gdofs_y = _boundary_field_data(dh, vector.components[1].field_name, tag)
    if coords_x.shape != coords_y.shape:
        raise RuntimeError(f"Mismatched boundary DOF counts for tag {tag!r}: {coords_x.shape} vs {coords_y.shape}")
    if coords_x.size and not np.allclose(coords_x, coords_y):
        raise RuntimeError(f"Mismatched vector boundary coordinate ordering for tag {tag!r}")
    load_vals = np.asarray(values, dtype=float)
    if load_vals.shape != (coords_x.shape[0], 2):
        raise ValueError(f"Expected point-load values with shape {(coords_x.shape[0], 2)}, got {load_vals.shape}")
    rhs = np.zeros(dh.total_dofs, dtype=float)
    rhs[np.asarray(gdofs_x, dtype=int)] = load_vals[:, 0]
    rhs[np.asarray(gdofs_y, dtype=int)] = load_vals[:, 1]
    return rhs


def _boundary_vector_from_global_values(
    dh: DofHandler,
    *,
    vector: VectorFunction,
    tag: str,
    global_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coords_x, gdofs_x = _boundary_field_data(dh, vector.components[0].field_name, tag)
    coords_y, gdofs_y = _boundary_field_data(dh, vector.components[1].field_name, tag)
    if coords_x.shape != coords_y.shape:
        raise RuntimeError(f"Mismatched boundary DOF counts for tag {tag!r}: {coords_x.shape} vs {coords_y.shape}")
    if coords_x.size and not np.allclose(coords_x, coords_y):
        raise RuntimeError(f"Mismatched vector boundary coordinate ordering for tag {tag!r}")
    values = np.column_stack(
        [
            np.asarray(global_values, dtype=float)[np.asarray(gdofs_x, dtype=int)],
            np.asarray(global_values, dtype=float)[np.asarray(gdofs_y, dtype=int)],
        ]
    )
    return coords_x, values


def _sample_boundary_normals(mesh: Mesh, tag: str, coords: np.ndarray) -> np.ndarray:
    seg_mid = []
    seg_n = []
    for eid in mesh.edge_bitset(tag).to_indices():
        edge = mesh.edge(int(eid))
        if edge.right is not None:
            continue
        pts = mesh.nodes_x_y_pos[list(edge.nodes)]
        seg_mid.append(np.mean(pts, axis=0))
        seg_n.append(np.asarray(edge.normal, dtype=float))
    if not seg_mid:
        raise ValueError(f"Boundary tag {tag!r} has no boundary edges.")
    mids = np.asarray(seg_mid, dtype=float)
    normals = np.asarray(seg_n, dtype=float)
    out = np.empty((len(coords), 2), dtype=float)
    for i, xy in enumerate(np.asarray(coords, dtype=float)):
        dist2 = np.sum((mids - xy[None, :]) ** 2, axis=1)
        normal = normals[int(np.argmin(dist2))]
        norm = np.linalg.norm(normal)
        out[i, :] = normal / max(norm, 1.0e-14)
    return out


def _transfer_vector_field(*, target_dh: DofHandler, target_vec: VectorFunction, source_lookup: CoordinateLookup) -> None:
    target_dh._ensure_dof_coords()
    for idx, component in enumerate(target_vec.components):
        ids = np.asarray(target_dh.get_field_slice(component.field_name), dtype=int)
        xy = np.asarray(target_dh._dof_coords[ids], dtype=float)
        vals = np.asarray(source_lookup(xy[:, 0], xy[:, 1]), dtype=float)
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        component.set_nodal_values(ids, vals[:, idx])


def _transfer_scalar_field(*, target_dh: DofHandler, target_fun: Function, source_lookup: CoordinateLookup) -> None:
    if int(source_lookup.dim) != 1:
        raise ValueError(f"Expected scalar lookup, got dim={source_lookup.dim}")
    target_dh._ensure_dof_coords()
    ids = np.asarray(target_dh.get_field_slice(target_fun.field_name), dtype=int)
    xy = np.asarray(target_dh._dof_coords[ids], dtype=float)
    vals = np.asarray(source_lookup(xy[:, 0], xy[:, 1]), dtype=float).reshape(-1)
    target_fun.set_nodal_values(ids, vals)


def _apply_dirichlet_bcs_to_state(
    *,
    dh: DofHandler,
    field_functions: dict[str, Function],
    bcs: list[BoundaryCondition],
) -> None:
    if not bcs:
        return
    for bc in bcs:
        if str(getattr(bc, "method", "")).strip().lower() != "dirichlet":
            continue
        field_name = str(getattr(bc, "field", "")).strip()
        target = field_functions.get(field_name)
        if target is None:
            continue
        dirichlet_data = dh.get_dirichlet_data([bc]) or {}
        if not dirichlet_data:
            continue
        ids = np.fromiter((int(gd) for gd in dirichlet_data.keys()), dtype=int)
        vals = np.fromiter((float(val) for val in dirichlet_data.values()), dtype=float)
        target.set_nodal_values(ids, vals)


def _fluid_boundary_conditions(
    *,
    iface_velocity: CoordinateLookup,
    inlet_lookup: Callable[[float, float], float],
    inlet_tag: str = "inlet",
    interface_tag: str,
    outlet_tag: str,
    walls_tag: str,
    cylinder_tag: str,
) -> tuple[list[BoundaryCondition], list[BoundaryCondition]]:
    zero = lambda x, y, t=0.0: 0.0
    bcs = [
        BoundaryCondition("ux", "dirichlet", inlet_tag, inlet_lookup),
        BoundaryCondition("uy", "dirichlet", inlet_tag, zero),
        BoundaryCondition("ux", "dirichlet", walls_tag, zero),
        BoundaryCondition("uy", "dirichlet", walls_tag, zero),
        BoundaryCondition("ux", "dirichlet", cylinder_tag, zero),
        BoundaryCondition("uy", "dirichlet", cylinder_tag, zero),
        BoundaryCondition("ux", "dirichlet", interface_tag, iface_velocity.component(0)),
        BoundaryCondition("uy", "dirichlet", interface_tag, iface_velocity.component(1)),
        BoundaryCondition("p", "dirichlet", outlet_tag, zero),
    ]
    bcs_homog = [
        BoundaryCondition("ux", "dirichlet", inlet_tag, zero),
        BoundaryCondition("uy", "dirichlet", inlet_tag, zero),
        BoundaryCondition("ux", "dirichlet", walls_tag, zero),
        BoundaryCondition("uy", "dirichlet", walls_tag, zero),
        BoundaryCondition("ux", "dirichlet", cylinder_tag, zero),
        BoundaryCondition("uy", "dirichlet", cylinder_tag, zero),
        BoundaryCondition("ux", "dirichlet", interface_tag, zero),
        BoundaryCondition("uy", "dirichlet", interface_tag, zero),
        BoundaryCondition("p", "dirichlet", outlet_tag, zero),
    ]
    return bcs, bcs_homog


def _fluid_zero_local_operator_forms(
    *,
    prob: dict[str, object],
    iface_velocity: CoordinateLookup,
    inlet_lookup: Callable[[float, float], float],
    inlet_tag: str = "inlet",
    interface_tag: str,
    outlet_tag: str,
    walls_tag: str,
    cylinder_tag: str,
    quad_order: int,
):
    u_k: VectorFunction = prob["u_k"]
    p_k: Function = prob["p_k"]
    du = prob["du"]
    v = prob["v"]
    dp = prob["dp"]
    q = prob["q"]
    cache = prob.get("_fluid_zero_local_operator_form_cache")
    if not isinstance(cache, dict):
        cache = {}
        prob["_fluid_zero_local_operator_form_cache"] = cache
    key = int(quad_order)
    cached = cache.get(key)
    if cached is None:
        dx_f = dx(metadata={"q": int(quad_order)})
        # Keep the placeholder forms structurally coupled to the fluid unknowns
        # so the compiler emits the full local DOF layout. A pure Constant(0)
        # form can be folded into an empty local vector/matrix, which breaks
        # reduced scatter when the exact discrete DVMS operator supplies the
        # real element blocks.
        zero_u = u_k - u_k
        zero_p = p_k - p_k
        residual = (dot(v, zero_u) + q * zero_p) * dx_f
        jacobian = (dot(du, v) * zero_p + dp * q * zero_p) * dx_f
        cached = (residual, jacobian)
        cache[key] = cached
    residual, jacobian = cached
    bcs, bcs_homog = _fluid_boundary_conditions(
        iface_velocity=iface_velocity,
        inlet_lookup=inlet_lookup,
        inlet_tag=inlet_tag,
        interface_tag=interface_tag,
        outlet_tag=outlet_tag,
        walls_tag=walls_tag,
        cylinder_tag=cylinder_tag,
    )
    return residual, jacobian, bcs, bcs_homog


def _get_or_create_cached_stage_solver(
    *,
    cache_owner: dict[str, object],
    cache_name: str,
    cache_key: tuple[object, ...],
    residual_form,
    jacobian_form,
    dof_handler: DofHandler,
    mixed_element: MixedElement,
    bcs,
    bcs_homog,
    newton_params: NewtonParameters,
    lin_params: LinearSolverParameters,
    quad_order: int,
    backend: str,
    operators: list[RuntimeOperator],
    active_fields: list[str] | tuple[str, ...],
) -> NewtonSolver:
    cache = cache_owner.get(cache_name)
    if not isinstance(cache, dict):
        cache = {}
        cache_owner[cache_name] = cache
    solver = cache.get(cache_key)
    if isinstance(solver, NewtonSolver):
        solver.bcs = list(bcs)
        solver.bcs_homog = list(bcs_homog)
        solver.np = newton_params
        solver.lp = lin_params
        solver.set_runtime_operators(list(operators))
        solver._ls_alpha_prev = 1.0
        return solver
    solver = NewtonSolver(
        residual_form=residual_form,
        jacobian_form=jacobian_form,
        dof_handler=dof_handler,
        mixed_element=mixed_element,
        bcs=list(bcs),
        bcs_homog=list(bcs_homog),
        newton_params=newton_params,
        lin_params=lin_params,
        quad_order=int(quad_order),
        backend=str(backend),
        operators=list(operators),
    )
    solver.set_active_fields(list(active_fields))
    cache[cache_key] = solver
    return solver


def _attach_runtime_operator_post_update_hook(
    *,
    solver: NewtonSolver,
    operators: Sequence[RuntimeOperator],
) -> None:
    callbacks = []
    for operator in tuple(operators or ()):
        callback = getattr(operator, "after_nonlinear_update", None)
        if callable(callback):
            callbacks.append(callback)
    if not callbacks:
        solver.post_cb = None
        return

    def _callback(functions) -> None:
        for callback in callbacks:
            callback(solver=solver, functions=functions)

    solver.post_cb = _callback


def _is_newton_maxiter_nonconvergence(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        "Newton did not converge" in msg
        or "Newton max_iter reached" in msg
        or "max iterations" in msg
    )


def _warm_fluid_exact_operator_kernels(
    *,
    prob: dict[str, object],
    mesh: Mesh,
    rho_f: float,
    mu_f: float,
    dt: float,
    bossak_alpha: float,
    dynamic_tau: float,
    quad_order: int,
    backend: str,
    contribution_mode: str = "system",
) -> None:
    backend_name = str(backend).strip().lower()
    if backend_name == "python":
        return
    cache_key = (
        backend_name,
        int(quad_order),
        float(rho_f),
        float(mu_f),
        float(dt),
        float(bossak_alpha),
        float(dynamic_tau),
        str(contribution_mode).strip().lower(),
    )
    warmed = getattr(prob, "_exact_local_operator_warm_cache", None)
    if not isinstance(warmed, set):
        warmed = set()
        prob["_exact_local_operator_warm_cache"] = warmed
    if cache_key in warmed:
        return

    first_eid = np.asarray([0], dtype=int)
    mesh_v_curr = prob.get("w_mesh_k")
    dvms_state = prob.get("dvms_state")
    predicted_snapshot = None
    if isinstance(dvms_state, FluidDVMSState) and int(dvms_state.sample_count) > 0:
        # Warm the predictor kernels without perturbing the live hidden state
        # that seeds the next real Newton assembly.
        predicted_snapshot = np.asarray(dvms_state.predicted_subscale_velocity, dtype=float).copy()
    assemble_fluid_dvms_local_contribution_batch(
        mesh=mesh,
        dh=prob["dh"],
        u_k=prob["u_k"],
        u_prev=prob["u_prev"],
        a_prev=prob["a_prev"],
        p_k=prob["p_k"],
        d_mesh=prob["d_mesh"],
        d_prev=prob["d_prev"],
        mesh_v=mesh_v_curr,
        mesh_v_prev=prob["w_mesh_prev"],
        mesh_a_prev=prob["a_mesh_prev"],
        state=prob.get("dvms_state"),
        rho_f=float(rho_f),
        mu_f=float(mu_f),
        dt=float(dt),
        bossak_alpha=float(bossak_alpha),
        element_ids=first_eid,
        quadrature_order=int(quad_order),
        contribution_mode=str(contribution_mode),
        backend=backend_name,
    )
    _update_fluid_dvms_predicted_subscale(
        state=prob["dvms_state"],
        dh=prob["dh"],
        mesh=mesh,
        u_k=prob["u_k"],
        u_prev=prob["u_prev"],
        a_prev=prob["a_prev"],
        p_k=prob["p_k"],
        d_mesh=prob["d_mesh"],
        d_prev=prob["d_prev"],
        mesh_v=mesh_v_curr,
        mesh_v_prev=prob["w_mesh_prev"],
        mesh_a_prev=prob["a_mesh_prev"],
        rho_f=float(rho_f),
        mu_f=float(mu_f),
        dt=float(dt),
        bossak_alpha=float(bossak_alpha),
        dynamic_tau=float(dynamic_tau),
        backend=backend_name,
    )
    if predicted_snapshot is not None:
        dvms_state.predicted_subscale_velocity[:, :] = predicted_snapshot
        dvms_state.sync_coefficient("predicted_subscale_velocity")
    warmed.add(cache_key)


def _fluid_residual_and_jacobian(
    *,
    prob: dict[str, object],
    rho_f: float,
    mu_f: float,
    dt: float,
    bossak_alpha: float,
    dynamic_tau: float,
    pressure_gauge: float,
    iface_velocity: CoordinateLookup,
    inlet_lookup: Callable[[float, float], float],
    inlet_tag: str = "inlet",
    interface_tag: str,
    outlet_tag: str,
    walls_tag: str,
    cylinder_tag: str,
    quad_order: int,
):
    u_k: VectorFunction = prob["u_k"]
    u_prev: VectorFunction = prob["u_prev"]
    a_prev: VectorFunction = prob["a_prev"]
    p_k: Function = prob["p_k"]
    d_mesh: VectorFunction = prob["d_mesh"]
    d_prev: VectorFunction = prob["d_prev"]
    d_prev2: VectorFunction = prob["d_prev2"]
    w_mesh_k: VectorFunction | None = prob.get("w_mesh_k")
    w_mesh_prev: VectorFunction = prob["w_mesh_prev"]
    a_mesh_prev: VectorFunction = prob["a_mesh_prev"]
    dvms_state: FluidDVMSState | None = prob.get("dvms_state")
    du = prob["du"]
    v = prob["v"]
    dp = prob["dp"]
    q = prob["q"]

    dx_f = dx(metadata={"q": int(quad_order)})
    h = _kratos_dvms_current_element_size_coefficient(prob["dh"].mixed_element.mesh, prob["dh"], d_mesh)
    dt_const = _named_constant("example2_local_dt", float(dt))
    bossak = _bossak_coefficients(alpha=float(bossak_alpha), dt=float(dt))
    bossak_ma0 = _named_constant("example2_local_bossak_ma0", float(bossak["ma0"]))
    bossak_ma2 = _named_constant("example2_local_bossak_ma2", float(bossak["ma2"]))
    bossak_mass_coeff = _named_constant("example2_local_bossak_mass_coeff", float(bossak["mam"]))
    bossak_alpha_const = _named_constant("example2_local_bossak_alpha", float(bossak["alpha"]))
    if isinstance(dvms_state, FluidDVMSState) and int(dvms_state.sample_count) > 0:
        predicted_subscale = dvms_state.coefficient("predicted_subscale_velocity")
        old_subscale = dvms_state.coefficient("old_subscale_velocity")
        momentum_projection = dvms_state.coefficient("momentum_projection")
        mass_projection = dvms_state.coefficient("mass_projection")
        old_mass_residual = dvms_state.coefficient("old_mass_residual")
    else:
        predicted_subscale = _EX2L_ZERO_VEC
        old_subscale = _EX2L_ZERO_VEC
        momentum_projection = _EX2L_ZERO_VEC
        mass_projection = _EX2L_ZERO
        old_mass_residual = _EX2L_ZERO
    rho = _named_constant("example2_local_rho_f", float(rho_f))
    predictor_symbolics = build_fluid_dvms_predictor_symbolics(
        u_k=u_k,
        u_prev=u_prev,
        a_prev=a_prev,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        d_prev2=d_prev2,
        mesh_v=w_mesh_k,
        mesh_v_prev=w_mesh_prev,
        mesh_a_prev=a_mesh_prev,
        dt=dt_const,
        bossak_ma0=bossak_ma0,
        bossak_ma2=bossak_ma2,
        bossak_alpha=bossak_alpha_const,
        rho=rho,
        old_subscale=old_subscale,
        momentum_projection=momentum_projection,
    )
    kin = predictor_symbolics.kinematics
    F = kin.F
    Finv = kin.Finv
    J = kin.J
    cof_F = kin.cof_F
    grad_u_phys = kin.grad_u_phys
    div_u_phys = kin.div_u_phys
    grad_du_phys = dot(grad(du), Finv)
    div_du_phys = inner(cof_F, grad(du)) / J
    div_v_phys = inner(cof_F, grad(v)) / J
    a_scheme = predictor_symbolics.a_scheme
    conv_velocity = kin.resolved_conv_velocity + predicted_subscale
    conv_speed = (dot(conv_velocity, conv_velocity) + _EX2L_CONV_EPS) ** _EX2_HALF
    mu_const = _named_constant("example2_local_mu_f", float(mu_f))
    sigma = _fluid_cauchy_stress_ufl(
        p=p_k,
        grad_u_phys=grad_u_phys,
        div_u_phys=div_u_phys,
        mu_const=mu_const,
    )
    gauge = _named_constant("example2_local_pressure_gauge", float(pressure_gauge))
    inv_dt = _named_constant("example2_local_inv_dt", 1.0 / max(float(dt), 1.0e-14))
    tau_c1 = _named_constant("example2_local_tau_c1", 8.0)
    tau_c2 = _named_constant("example2_local_tau_c2", 2.0)
    dynamic_tau_const = _named_constant("example2_local_dynamic_tau", float(dynamic_tau))
    tau_one = dynamic_tau_const * _EX2_ONE / (
        tau_c1 * mu_const / (h * h)
        + rho * (inv_dt + tau_c2 * conv_speed / h)
    )
    tau_two = mu_const + rho * conv_speed * h / _EX2L_FOUR
    tau_p = rho * h * h / (tau_c1 * dt_const)
    grad_p_phys = predictor_symbolics.grad_p_phys
    grad_q_phys = dot(Finv.T, grad(q))
    grad_dp_phys = dot(Finv.T, grad(dp))
    old_uss_term = rho * inv_dt * old_subscale
    tau_test_conv = rho * dot(dot(grad(v), Finv), conv_velocity)
    tau_test_pres = grad_q_phys
    tau_test_velocity = tau_test_conv - rho * inv_dt * v + tau_test_pres
    tau_test_mass = tau_test_conv - inv_dt * v + tau_test_pres
    tau_res_dynamic = rho * a_scheme
    tau_res_static_conv = rho * dot(grad_u_phys, conv_velocity)
    tau_res_static_pres = grad_p_phys
    tau_dtest_conv = rho * dot(dot(grad(v), Finv), du)
    tau_dtest_velocity = tau_dtest_conv
    tau_dtest_mass = tau_dtest_conv
    tau_dres_dynamic = rho * bossak_mass_coeff * du
    tau_dres_static_conv_1 = rho * dot(grad_du_phys, conv_velocity)
    tau_dres_static_conv_2 = rho * dot(grad_u_phys, du)
    tau_dres_static_pres = grad_dp_phys

    residual = rho * J * dot(a_scheme, v) * dx_f
    residual += J * rho * dot(dot(grad_u_phys, conv_velocity), v) * dx_f
    residual += inner(J * dot(sigma, Finv.T), grad(v)) * dx_f
    residual += inner(cof_F, grad(u_k)) * q * dx_f
    residual += gauge * p_k * q * dx_f
    residual += J * tau_one * dot(tau_test_mass, tau_res_dynamic) * dx_f
    residual += J * tau_one * dot(tau_test_velocity, tau_res_static_conv + tau_res_static_pres) * dx_f
    residual += J * dot(old_uss_term, v) * dx_f
    residual += J * tau_one * dot(tau_test_velocity, old_uss_term - momentum_projection) * dx_f
    residual -= J * (((tau_two + tau_p) * mass_projection + tau_p * old_mass_residual) * div_v_phys) * dx_f
    residual += J * ((tau_two + tau_p) * div_u_phys * div_v_phys) * dx_f

    jacobian = rho * J * dot(bossak_mass_coeff * du, v) * dx_f
    jacobian += J * rho * dot(dot(grad_du_phys, conv_velocity), v) * dx_f
    jacobian += J * rho * dot(dot(grad_u_phys, du), v) * dx_f
    sigma_du = _fluid_cauchy_stress_ufl(
        p=dp,
        grad_u_phys=grad_du_phys,
        div_u_phys=div_du_phys,
        mu_const=mu_const,
    )
    jacobian += inner(J * dot(sigma_du, Finv.T), grad(v)) * dx_f
    jacobian += inner(cof_F, grad(du)) * q * dx_f
    jacobian += gauge * dp * q * dx_f
    jacobian += J * tau_one * dot(tau_dtest_mass, tau_res_dynamic) * dx_f
    jacobian += J * tau_one * dot(tau_test_mass, tau_dres_dynamic) * dx_f
    jacobian += J * tau_one * dot(tau_dtest_velocity, tau_res_static_conv + tau_res_static_pres) * dx_f
    jacobian += J * tau_one * dot(
        tau_test_velocity,
        tau_dres_static_conv_1 + tau_dres_static_conv_2 + tau_dres_static_pres,
    ) * dx_f
    jacobian += J * ((tau_two + tau_p) * div_du_phys * div_v_phys) * dx_f

    bcs, bcs_homog = _fluid_boundary_conditions(
        iface_velocity=iface_velocity,
        inlet_lookup=inlet_lookup,
        inlet_tag=inlet_tag,
        interface_tag=interface_tag,
        outlet_tag=outlet_tag,
        walls_tag=walls_tag,
        cylinder_tag=cylinder_tag,
    )
    return residual, jacobian, bcs, bcs_homog


def _solid_residual_and_jacobian(
    *,
    prob: dict[str, object],
    traction_lookup: CoordinateLookup,
    mu_s: float,
    lambda_s: float,
    interface_tag: str,
    clamp_tag: str,
    quad_order: int,
):
    d_k: VectorFunction = prob["d_k"]
    dd = prob["dd"]
    w = prob["w"]

    traction = Analytic(lambda x, y: traction_lookup(x, y), dim=1)
    dx_s = dx(metadata={"q": int(quad_order)})
    ds_iface = dS(defined_on=prob["dh"].mixed_element.mesh.edge_bitset(interface_tag), metadata={"q": int(quad_order)})

    F = Identity(2) + grad(d_k)
    mu_s_const = _named_constant("example2_local_mu_s", float(mu_s))
    lambda_s_const = _named_constant("example2_local_lambda_s", float(lambda_s))
    P = _kratos_hyperelastic_plane_strain_pk1(F, mu_s_const, lambda_s_const)
    deltaP = _kratos_hyperelastic_plane_strain_delta_pk1(F, grad(dd), mu_s_const, lambda_s_const)
    residual = inner(P, grad(w)) * dx_s - dot(traction, w) * ds_iface
    jacobian = inner(deltaP, grad(w)) * dx_s

    zero = lambda x, y, t=0.0: 0.0
    bcs = [
        BoundaryCondition("dx", "dirichlet", clamp_tag, zero),
        BoundaryCondition("dy", "dirichlet", clamp_tag, zero),
    ]
    bcs_homog = [
        BoundaryCondition("dx", "dirichlet", clamp_tag, zero),
        BoundaryCondition("dy", "dirichlet", clamp_tag, zero),
    ]
    return residual, jacobian, bcs, bcs_homog


def _porous_solid_residual_and_jacobian(
    *,
    prob: dict[str, object],
    material: UPlMaterial2D,
    dt: float,
    clamp_tag: str,
    quad_order: int,
):
    d_k: VectorFunction = prob["d_k"]
    p_k: Function = prob["p_k"]
    d_prev: VectorFunction = prob["d_prev"]
    p_prev: Function = prob["p_prev"]
    dd = prob["dd"]
    dp = prob["dp"]
    w = prob["w"]
    q = prob["q"]

    dx_s = dx(metadata={"q": int(quad_order)})
    theta = _named_constant("example2_porous_backward_euler_theta", 1.0)
    current_system = build_kratos_quasistatic_upl_system_2d(
        u_trial=d_k,
        p_trial=p_k,
        u_test=w,
        p_test=q,
        u_prev=d_prev,
        p_prev=p_prev,
        material=material,
        dt=_named_constant("example2_porous_dt", float(dt)),
        theta_u=theta,
        theta_p=theta,
        dx_measure=dx_s,
    )
    tangent_system = build_kratos_quasistatic_upl_system_2d(
        u_trial=dd,
        p_trial=dp,
        u_test=w,
        p_test=q,
        u_prev=d_prev,
        p_prev=p_prev,
        material=material,
        dt=_named_constant("example2_porous_dt", float(dt)),
        theta_u=theta,
        theta_p=theta,
        dx_measure=dx_s,
    )
    residual = current_system.lhs_form - current_system.rhs_form
    jacobian = tangent_system.lhs_form

    zero = lambda x, y, t=0.0: 0.0
    bcs = [
        BoundaryCondition("dx", "dirichlet", clamp_tag, zero),
        BoundaryCondition("dy", "dirichlet", clamp_tag, zero),
    ]
    bcs_homog = [
        BoundaryCondition("dx", "dirichlet", clamp_tag, zero),
        BoundaryCondition("dy", "dirichlet", clamp_tag, zero),
    ]
    return residual, jacobian, bcs, bcs_homog


def _mesh_extension_equation(
    *,
    prob: dict[str, object],
    interface_disp: CoordinateLookup,
    interface_tag: str,
    fixed_tags: tuple[str, ...],
    quad_order: int,
) -> tuple[Equation, list[BoundaryCondition]]:
    dm = prob["dm"]
    z = prob["z"]
    zero_vec = _EX2L_ZERO_VEC
    # The mesh-moving strategy resets the mesh part to the initial coordinates
    # before each solve, so the structural-similarity operator is assembled on
    # the reference geometry every time.
    poisson_ratio = 0.3
    young_modulus = 1.0
    mu_m = young_modulus / (2.0 * (1.0 + poisson_ratio))
    lambda_m = young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    mesh_mu = _named_constant("example2_local_mesh_mu", float(mu_m))
    mesh_lambda = _named_constant("example2_local_mesh_lambda", float(lambda_m))
    mesh_stiffening_factor = _named_constant("example2_local_mesh_stiffening_factor", 100.0)
    mesh_stiffening_exponent = _named_constant("example2_local_mesh_stiffening_exponent", 1.5)
    mesh_jac_eps = _named_constant("example2_local_mesh_jac_eps", 1.0e-14)
    area_ref = _kratos_structural_mesh_area_coefficient(prob["dh"].mixed_element.mesh)
    stiffening = (mesh_stiffening_factor / (area_ref + mesh_jac_eps)) ** mesh_stiffening_exponent
    grad_dm_phys = grad(dm)
    div_dm_phys = div(dm)
    eps_dm_phys = _EX2L_HALF * (grad_dm_phys + grad_dm_phys.T)
    sigma_dm_phys = stiffening * (_EX2L_TWO * mesh_mu * eps_dm_phys + mesh_lambda * div_dm_phys * Identity(2))
    equation = Equation(
        inner(sigma_dm_phys, grad(z)) * dx(metadata={"q": int(quad_order)}),
        _EX2L_ZERO * dot(z, zero_vec) * dx(metadata={"q": int(quad_order)}),
    )
    zero = lambda x, y, t=0.0: 0.0
    bcs = [
        BoundaryCondition("mx", "dirichlet", tag, zero) for tag in fixed_tags
    ] + [
        BoundaryCondition("my", "dirichlet", tag, zero) for tag in fixed_tags
    ] + [
        BoundaryCondition("mx", "dirichlet", interface_tag, interface_disp.component(0)),
        BoundaryCondition("my", "dirichlet", interface_tag, interface_disp.component(1)),
    ]
    return equation, bcs


def _fluid_interface_samples(
    *,
    dh: DofHandler,
    mesh: Mesh,
    u: VectorFunction,
    p: Function,
    d_mesh: VectorFunction,
    iface_coords: np.ndarray,
    interface_tag: str,
    mu_f: float,
) -> CoordinateLookup:
    normals = _sample_boundary_normals(mesh, interface_tag, iface_coords)
    traction_vals = np.empty((iface_coords.shape[0], 2), dtype=float)
    for i, xy in enumerate(np.asarray(iface_coords, dtype=float)):
        p_val, _ = _eval_scalar_with_grad(dh, mesh, p, tuple(xy))
        _, grad_u = _eval_vector_with_grad(dh, mesh, u, tuple(xy))
        _, grad_m = _eval_vector_with_grad(dh, mesh, d_mesh, tuple(xy))
        F = np.eye(2, dtype=float) + grad_m
        Finv = np.linalg.inv(F)
        J = float(np.linalg.det(F))
        grad_u_phys = np.asarray(grad_u, dtype=float) @ Finv
        div_u_phys = float(np.trace(grad_u_phys))
        sigma = _fluid_cauchy_stress_numpy(
            p_val=float(p_val),
            grad_u_phys=grad_u_phys,
            div_u_phys=div_u_phys,
            mu_f=float(mu_f),
        )
        n_solid = -normals[i]
        traction_vals[i, :] = J * ((sigma @ Finv.T) @ n_solid)
    return CoordinateLookup(iface_coords, traction_vals, dim=2)


def _fluid_reference_traction_at_point(
    *,
    dh: DofHandler,
    mesh: Mesh,
    u: VectorFunction,
    p: Function,
    d_mesh: VectorFunction,
    point: np.ndarray,
    solid_ref_normal: np.ndarray,
    mu_f: float,
) -> np.ndarray:
    xy = np.asarray(point, dtype=float)
    N_s = np.asarray(solid_ref_normal, dtype=float).reshape(2)
    p_val, _ = _eval_scalar_with_grad(dh, mesh, p, tuple(xy))
    _, grad_u = _eval_vector_with_grad(dh, mesh, u, tuple(xy))
    _, grad_m = _eval_vector_with_grad(dh, mesh, d_mesh, tuple(xy))
    F = np.eye(2, dtype=float) + grad_m
    Finv = np.linalg.inv(F)
    J = float(np.linalg.det(F))
    grad_u_phys = np.asarray(grad_u, dtype=float) @ Finv
    div_u_phys = float(np.trace(grad_u_phys))
    sigma = _fluid_cauchy_stress_numpy(
        p_val=float(p_val),
        grad_u_phys=grad_u_phys,
        div_u_phys=div_u_phys,
        mu_f=float(mu_f),
    )
    # Pull back the fluid Cauchy traction to the reference solid interface:
    # G_N,s = J * sigma * F^{-T} * N_s
    return J * ((sigma @ Finv.T) @ N_s)


def _assemble_fluid_local_velocity_contribution_raw(
    *,
    prob: dict[str, object],
    rho_f: float,
    mu_f: float,
    dt: float,
    quad_order: int,
    bossak_alpha: float = -0.3,
    need_matrix: bool = False,
    contribution_mode: str = "system",
    apply_dirichlet_lift: bool = False,
    backend: str = "python",
    element_ids: np.ndarray | None = None,
):
    import scipy.sparse as sp

    dh: DofHandler = prob["dh"]
    mesh: Mesh = dh.mixed_element.mesh
    ndof = int(dh.total_dofs)
    A_full = sp.lil_matrix((ndof, ndof), dtype=float) if need_matrix else None
    R_full = np.zeros(ndof, dtype=float)
    if element_ids is None:
        element_ids_arr = np.arange(int(mesh.n_elements), dtype=int)
    else:
        element_ids_arr = np.asarray(element_ids, dtype=int).reshape(-1)
    raw_residualization = str(os.getenv("PYCUTFEM_EX2_DVMS_RESIDUALIZATION", "kratos") or "kratos").strip().lower()
    if str(contribution_mode).strip().lower() == "system" and raw_residualization == "kratos":
        op = FluidDVMSLocalVelocityContributionOperator(
            mesh=mesh,
            dh=dh,
            u_k=prob["u_k"],
            u_prev=prob.get("u_prev"),
            a_prev=prob.get("a_prev"),
            a_curr=prob.get("a_k"),
            p_k=prob["p_k"],
            d_mesh=prob["d_mesh"],
            d_prev=prob["d_prev"],
            d_prev2=prob.get("d_prev2"),
            mesh_v=prob.get("w_mesh_k"),
            mesh_v_prev=prob.get("w_mesh_prev"),
            mesh_a_prev=prob.get("a_mesh_prev"),
            state=prob.get("dvms_state"),
            rho_f=float(rho_f),
            mu_f=float(mu_f),
            dt=float(dt),
            bossak_alpha=float(bossak_alpha),
            element_ids=element_ids_arr,
            quadrature_order=int(quad_order),
            contribution_mode=str(contribution_mode),
            apply_dirichlet_lift=bool(apply_dirichlet_lift),
            residualization="kratos",
        )
        solver_stub = NewtonSolver.__new__(NewtonSolver)
        solver_stub.backend = str(backend)
        workset = op.build_local_workset(solver=solver_stub, coeffs=None, need_matrix=bool(need_matrix))
        local = op.assemble_local(workset)
        K_elem = None if local.K_elem is None else np.asarray(local.K_elem, dtype=float)
        F_elem = None if local.F_elem is None else -np.asarray(local.F_elem, dtype=float)
        gdofs_map = np.asarray(local.gdofs_map, dtype=int)
        batch_element_ids = np.asarray(local.element_ids, dtype=int)
    else:
        batch = assemble_fluid_dvms_local_contribution_batch(
            mesh=mesh,
            dh=dh,
            u_k=prob["u_k"],
            u_prev=prob.get("u_prev"),
            a_prev=prob.get("a_prev"),
            a_curr=prob.get("a_k"),
            p_k=prob["p_k"],
            d_mesh=prob["d_mesh"],
            d_prev=prob["d_prev"],
            mesh_v=prob.get("w_mesh_k"),
            mesh_v_prev=prob.get("w_mesh_prev"),
            mesh_a_prev=prob.get("a_mesh_prev"),
            state=prob.get("dvms_state"),
            rho_f=float(rho_f),
            mu_f=float(mu_f),
            dt=float(dt),
            bossak_alpha=float(bossak_alpha),
            element_ids=element_ids_arr,
            quadrature_order=int(quad_order),
            contribution_mode=str(contribution_mode),
            backend=str(backend),
        )
        K_elem, F_elem, gdofs_map = _compress_batch_to_fluid_block(
            dh,
            LocalAssemblyResult(
                K_elem=None if batch.K_elem is None else np.asarray(batch.K_elem, dtype=float),
                F_elem=None if batch.F_elem is None else np.asarray(batch.F_elem, dtype=float),
                element_ids=np.asarray(batch.element_ids, dtype=int),
                gdofs_map=np.asarray(batch.gdofs_map, dtype=int),
            ),
        )
        batch_element_ids = np.asarray(batch.element_ids, dtype=int)
        if bool(apply_dirichlet_lift) and K_elem is not None:
            bcs_apply = prob.get("_current_bcs")
            if not bcs_apply:
                bcs_apply = prob.get("bcs")
            bc_map = dh.get_dirichlet_data(bcs_apply) or {}
            if bc_map:
                bc_values_full = np.zeros(ndof, dtype=float)
                bc_ids = np.fromiter((int(gdof) for gdof in bc_map.keys()), dtype=int)
                bc_vals = np.fromiter((float(val) for val in bc_map.values()), dtype=float)
                bc_values_full[bc_ids] = bc_vals
                local_bc = np.asarray(bc_values_full[np.asarray(gdofs_map, dtype=int)], dtype=float)
                lifted = np.einsum("eij,ej->ei", np.asarray(K_elem, dtype=float), local_bc, optimize=True)
                if F_elem is None:
                    F_elem = -lifted
                else:
                    F_elem = F_elem - lifted
    scatter_owner = NewtonSolver.__new__(NewtonSolver)
    A_full, R_full = NewtonSolver.scatter_element_contribs_full(
        scatter_owner,
        K_elem=None if not need_matrix else K_elem,
        F_elem=F_elem,
        element_ids=batch_element_ids,
        gdofs_map=np.asarray(gdofs_map, dtype=int),
        A_full=A_full,
        R_full=R_full,
    )

    return (A_full.tocsr() if A_full is not None else None), R_full


def _assemble_fluid_sampled_lspg_element_contributions_raw(
    *,
    prob: dict[str, object],
    rho_f: float,
    mu_f: float,
    dt: float,
    quad_order: int,
    bossak_alpha: float,
    contribution_mode: str,
    backend: str,
    element_ids: np.ndarray,
    row_dofs: np.ndarray,
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble sampled residual/Jacobian contributions before cubature summation.

    The returned residual has the same sign convention as the global Newton
    residual used by the LSPG verifier. The element axis is intentionally kept
    explicit so empirical cubature can scale or prune complete element blocks.
    """

    dh: DofHandler = prob["dh"]
    mesh: Mesh = dh.mixed_element.mesh
    rows = np.asarray(row_dofs, dtype=int).reshape(-1)
    trial_basis = np.asarray(basis, dtype=float)
    if trial_basis.ndim != 2 or int(trial_basis.shape[0]) != int(dh.total_dofs):
        raise ValueError("Sampled LSPG basis shape is incompatible with the fluid dof handler.")
    if np.any(rows < 0) or np.any(rows >= int(dh.total_dofs)):
        raise ValueError("Sampled LSPG row_dofs contain out-of-range entries.")
    element_ids_arr = np.asarray(element_ids, dtype=int).reshape(-1)
    op = FluidDVMSLocalVelocityContributionOperator(
        mesh=mesh,
        dh=dh,
        u_k=prob["u_k"],
        u_prev=prob.get("u_prev"),
        a_prev=prob.get("a_prev"),
        a_curr=prob.get("a_k"),
        p_k=prob["p_k"],
        d_mesh=prob["d_mesh"],
        d_prev=prob["d_prev"],
        d_prev2=prob.get("d_prev2"),
        mesh_v=prob.get("w_mesh_k"),
        mesh_v_prev=prob.get("w_mesh_prev"),
        mesh_a_prev=prob.get("a_mesh_prev"),
        state=prob.get("dvms_state"),
        rho_f=float(rho_f),
        mu_f=float(mu_f),
        dt=float(dt),
        bossak_alpha=float(bossak_alpha),
        element_ids=element_ids_arr,
        quadrature_order=int(quad_order),
        contribution_mode=str(contribution_mode),
        apply_dirichlet_lift=False,
        residualization="kratos",
    )
    solver_stub = NewtonSolver.__new__(NewtonSolver)
    solver_stub.backend = str(backend)
    workset = op.build_local_workset(solver=solver_stub, coeffs=None, need_matrix=True)
    local = op.assemble_local(workset)
    if local.K_elem is None or local.F_elem is None:
        raise RuntimeError("Sampled LSPG local assembly did not return both K and RHS.")
    K_elem = np.asarray(local.K_elem, dtype=float)
    F_elem = -np.asarray(local.F_elem, dtype=float)
    gdofs_map = np.asarray(local.gdofs_map, dtype=int)
    if K_elem.ndim != 3 or F_elem.ndim != 2 or gdofs_map.ndim != 2:
        raise RuntimeError("Sampled LSPG local assembly returned invalid local block shapes.")
    if int(K_elem.shape[0]) != int(element_ids_arr.size):
        raise RuntimeError("Sampled LSPG local assembly element count does not match element_ids.")

    row_positions = np.full(int(dh.total_dofs), -1, dtype=int)
    row_positions[rows] = np.arange(int(rows.size), dtype=int)
    local_row_positions = row_positions[gdofs_map]
    selected = local_row_positions >= 0
    elem_idx, local_i = np.nonzero(selected)
    residual_by_element = np.zeros((int(element_ids_arr.size), int(rows.size)), dtype=float)
    trial_by_element = np.zeros(
        (int(element_ids_arr.size), int(rows.size), int(trial_basis.shape[1])),
        dtype=float,
    )
    if elem_idx.size:
        sample_idx = local_row_positions[elem_idx, local_i]
        np.add.at(residual_by_element, (elem_idx, sample_idx), F_elem[elem_idx, local_i])
        local_basis = trial_basis[gdofs_map[elem_idx], :]
        local_k_rows = K_elem[elem_idx, local_i, :]
        trial_contrib = np.einsum("sl,slm->sm", local_k_rows, local_basis, optimize=True)
        for source_idx, row_idx, contrib in zip(elem_idx, sample_idx, trial_contrib, strict=False):
            trial_by_element[int(source_idx), int(row_idx), :] += contrib
    return element_ids_arr.copy(), -residual_by_element, trial_by_element


def _assemble_fluid_sampled_lspg_rows_raw(
    *,
    prob: dict[str, object],
    rho_f: float,
    mu_f: float,
    dt: float,
    quad_order: int,
    bossak_alpha: float,
    contribution_mode: str,
    backend: str,
    element_ids: np.ndarray,
    row_dofs: np.ndarray,
    basis: np.ndarray,
    element_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    dh: DofHandler = prob["dh"]
    rows = np.asarray(row_dofs, dtype=int).reshape(-1)
    trial_basis = np.asarray(basis, dtype=float)
    if trial_basis.ndim != 2 or int(trial_basis.shape[0]) != int(dh.total_dofs):
        raise ValueError("Sampled LSPG basis shape is incompatible with the fluid dof handler.")
    if np.any(rows < 0) or np.any(rows >= int(dh.total_dofs)):
        raise ValueError("Sampled LSPG row_dofs contain out-of-range entries.")
    element_ids_arr = np.asarray(element_ids, dtype=int).reshape(-1)
    element_weights_arr: np.ndarray | None = None
    if element_weights is not None:
        element_weights_arr = np.asarray(element_weights, dtype=float).reshape(-1)
        if int(element_weights_arr.size) != int(element_ids_arr.size):
            raise ValueError("Sampled LSPG element_weights size must match element_ids.")
        if np.any(element_weights_arr < 0.0) or not np.all(np.isfinite(element_weights_arr)):
            raise ValueError("Sampled LSPG element_weights must be finite and nonnegative.")
    raw_residualization = str(os.getenv("PYCUTFEM_EX2_DVMS_RESIDUALIZATION", "kratos") or "kratos").strip().lower()
    if str(contribution_mode).strip().lower() != "system" or raw_residualization != "kratos":
        if element_weights_arr is not None and not np.allclose(element_weights_arr, 1.0):
            raise RuntimeError(
                "Element-level cubature requires the local kratos/system sampled-LSPG assembly path."
            )
        matrix, raw_rhs = _assemble_fluid_local_velocity_contribution_raw(
            prob=prob,
            rho_f=float(rho_f),
            mu_f=float(mu_f),
            dt=float(dt),
            quad_order=int(quad_order),
            bossak_alpha=float(bossak_alpha),
            need_matrix=True,
            contribution_mode=str(contribution_mode),
            apply_dirichlet_lift=False,
            backend=str(backend),
            element_ids=element_ids_arr,
        )
        if matrix is None:
            raise RuntimeError("Sampled LSPG HROM requires a sampled Jacobian matrix.")
        sampled_residual = -np.asarray(raw_rhs, dtype=float).reshape(-1)[rows]
        sampled_trial = np.asarray(matrix.tocsr()[rows, :] @ trial_basis, dtype=float)
        return sampled_residual, sampled_trial

    mesh: Mesh = dh.mixed_element.mesh
    op = FluidDVMSLocalVelocityContributionOperator(
        mesh=mesh,
        dh=dh,
        u_k=prob["u_k"],
        u_prev=prob.get("u_prev"),
        a_prev=prob.get("a_prev"),
        a_curr=prob.get("a_k"),
        p_k=prob["p_k"],
        d_mesh=prob["d_mesh"],
        d_prev=prob["d_prev"],
        d_prev2=prob.get("d_prev2"),
        mesh_v=prob.get("w_mesh_k"),
        mesh_v_prev=prob.get("w_mesh_prev"),
        mesh_a_prev=prob.get("a_mesh_prev"),
        state=prob.get("dvms_state"),
        rho_f=float(rho_f),
        mu_f=float(mu_f),
        dt=float(dt),
        bossak_alpha=float(bossak_alpha),
        element_ids=element_ids_arr,
        quadrature_order=int(quad_order),
        contribution_mode=str(contribution_mode),
        apply_dirichlet_lift=False,
        residualization="kratos",
    )
    solver_stub = NewtonSolver.__new__(NewtonSolver)
    solver_stub.backend = str(backend)
    workset = op.build_local_workset(solver=solver_stub, coeffs=None, need_matrix=True)
    local = op.assemble_local(workset)
    if local.K_elem is None or local.F_elem is None:
        raise RuntimeError("Sampled LSPG local assembly did not return both K and RHS.")
    K_elem = np.asarray(local.K_elem, dtype=float)
    F_elem = -np.asarray(local.F_elem, dtype=float)
    gdofs_map = np.asarray(local.gdofs_map, dtype=int)
    if K_elem.ndim != 3 or F_elem.ndim != 2 or gdofs_map.ndim != 2:
        raise RuntimeError("Sampled LSPG local assembly returned invalid local block shapes.")
    if element_weights_arr is not None:
        scale = element_weights_arr.reshape(-1)
        K_elem = K_elem * scale[:, None, None]
        F_elem = F_elem * scale[:, None]
    row_positions = np.full(int(dh.total_dofs), -1, dtype=int)
    row_positions[rows] = np.arange(int(rows.size), dtype=int)
    local_row_positions = row_positions[gdofs_map]
    selected = local_row_positions >= 0
    elem_idx, local_i = np.nonzero(selected)
    sampled_raw_rhs = np.zeros(int(rows.size), dtype=float)
    sampled_trial = np.zeros((int(rows.size), int(trial_basis.shape[1])), dtype=float)
    if elem_idx.size:
        sample_idx = local_row_positions[elem_idx, local_i]
        np.add.at(sampled_raw_rhs, sample_idx, F_elem[elem_idx, local_i])
        local_basis = trial_basis[gdofs_map[elem_idx], :]
        local_k_rows = K_elem[elem_idx, local_i, :]
        trial_contrib = np.einsum("sl,slm->sm", local_k_rows, local_basis, optimize=True)
        np.add.at(sampled_trial, sample_idx, trial_contrib)
    return -sampled_raw_rhs, sampled_trial


def _refresh_fluid_reaction_reconstruction_state(
    *,
    prob: dict[str, object],
    rho_f: float,
    mu_f: float,
    dt: float,
    bossak_alpha: float,
    dynamic_tau: float,
    backend: str,
) -> None:
    state = prob.get("dvms_state")
    if state is None:
        return
    required = ("dh", "u_k", "u_prev", "a_prev", "p_k", "d_mesh", "d_prev")
    if any(key not in prob for key in required):
        return
    dh: DofHandler = prob["dh"]
    mesh: Mesh = dh.mixed_element.mesh
    if hasattr(state, "sample_count"):
        _clear_fluid_dvms_oss_projections(state)
    _call_with_supported_keywords(
        _update_fluid_dvms_predicted_subscale,
        state=state,
        dh=dh,
        mesh=mesh,
        u_k=prob["u_k"],
        u_prev=prob["u_prev"],
        a_prev=prob["a_prev"],
        a_curr=prob.get("a_k"),
        p_k=prob["p_k"],
        d_mesh=prob["d_mesh"],
        d_prev=prob["d_prev"],
        d_prev2=prob.get("d_prev2"),
        mesh_v=prob.get("w_mesh_k"),
        mesh_v_prev=prob.get("w_mesh_prev"),
        mesh_a_prev=prob.get("a_mesh_prev"),
        rho_f=float(rho_f),
        mu_f=float(mu_f),
        dt=float(dt),
        bossak_alpha=float(bossak_alpha),
        dynamic_tau=float(dynamic_tau),
        backend=str(backend),
        use_oss=False,
    )


def _fluid_interface_velocity_dofs(
    prob: dict[str, object],
    *,
    interface_tag: str,
) -> np.ndarray:
    dh: DofHandler = prob["dh"]
    u_k: VectorFunction = prob["u_k"]
    ux_ids = np.asarray(
        _boundary_field_data(dh, u_k.components[0].field_name, interface_tag)[1],
        dtype=int,
    ).reshape(-1)
    uy_ids = np.asarray(
        _boundary_field_data(dh, u_k.components[1].field_name, interface_tag)[1],
        dtype=int,
    ).reshape(-1)
    return np.unique(np.concatenate([ux_ids, uy_ids]).astype(int, copy=False))


def _fluid_interface_reaction_element_ids(
    prob: dict[str, object],
    *,
    interface_tag: str,
) -> np.ndarray:
    """Elements whose local velocity rows can contribute to interface reactions."""
    cache = prob.get("_interface_reaction_element_ids_cache")
    if not isinstance(cache, dict):
        cache = {}
        prob["_interface_reaction_element_ids_cache"] = cache
    key = str(interface_tag)
    cached = cache.get(key)
    if cached is not None:
        return np.asarray(cached, dtype=int)

    dh: DofHandler = prob["dh"]
    u_k: VectorFunction = prob["u_k"]
    interface_rows = _fluid_interface_velocity_dofs(prob, interface_tag=interface_tag)
    if interface_rows.size == 0:
        out = np.zeros((0,), dtype=int)
        cache[key] = out
        return out

    ux_map = np.asarray(dh.element_maps[u_k.components[0].field_name], dtype=int)
    uy_map = np.asarray(dh.element_maps[u_k.components[1].field_name], dtype=int)
    mask = np.isin(ux_map, interface_rows).any(axis=1) | np.isin(uy_map, interface_rows).any(axis=1)
    out = np.flatnonzero(mask).astype(int, copy=False)
    cache[key] = out
    return out


def _cached_fluid_reaction_operator(
    *,
    prob: dict[str, object],
    rho_f: float,
    mu_f: float,
    dt: float,
    quad_order: int,
    bossak_alpha: float,
    contribution_mode: str,
    element_ids: np.ndarray,
) -> FluidDVMSLocalVelocityContributionOperator:
    cache = prob.get("_interface_reaction_operator_cache")
    if not isinstance(cache, dict):
        cache = {}
        prob["_interface_reaction_operator_cache"] = cache
    element_ids_arr = np.asarray(element_ids, dtype=int).reshape(-1)
    key = (
        float(rho_f),
        float(mu_f),
        float(dt),
        int(quad_order),
        float(bossak_alpha),
        str(contribution_mode).strip().lower(),
        tuple(int(v) for v in element_ids_arr),
    )
    cached = cache.get(key)
    if cached is not None:
        return cached

    dh: DofHandler = prob["dh"]
    mesh: Mesh = dh.mixed_element.mesh
    op = FluidDVMSLocalVelocityContributionOperator(
        mesh=mesh,
        dh=dh,
        u_k=prob["u_k"],
        u_prev=prob.get("u_prev"),
        a_prev=prob.get("a_prev"),
        a_curr=prob.get("a_k"),
        p_k=prob["p_k"],
        d_mesh=prob["d_mesh"],
        d_prev=prob["d_prev"],
        d_prev2=prob.get("d_prev2"),
        mesh_v=prob.get("w_mesh_k"),
        mesh_v_prev=prob.get("w_mesh_prev"),
        mesh_a_prev=prob.get("a_mesh_prev"),
        state=prob.get("dvms_state"),
        rho_f=float(rho_f),
        mu_f=float(mu_f),
        dt=float(dt),
        bossak_alpha=float(bossak_alpha),
        element_ids=element_ids_arr,
        quadrature_order=int(quad_order),
        contribution_mode=str(contribution_mode),
    )
    cache[key] = op
    return op


def _assemble_fluid_reaction_residual_cached(
    *,
    prob: dict[str, object],
    rho_f: float,
    mu_f: float,
    dt: float,
    quad_order: int,
    bossak_alpha: float,
    interface_tag: str,
    backend: str,
    contribution_mode: str,
) -> np.ndarray:
    dh: DofHandler = prob["dh"]
    t0 = time.perf_counter()
    element_ids = _fluid_interface_reaction_element_ids(prob, interface_tag=interface_tag)
    t_ids = time.perf_counter()
    op = _cached_fluid_reaction_operator(
        prob=prob,
        rho_f=float(rho_f),
        mu_f=float(mu_f),
        dt=float(dt),
        quad_order=int(quad_order),
        bossak_alpha=float(bossak_alpha),
        contribution_mode=str(contribution_mode),
        element_ids=element_ids,
    )
    t_op = time.perf_counter()
    solver_stub = NewtonSolver.__new__(NewtonSolver)
    solver_stub.backend = str(backend)
    workset = op.build_local_workset(solver=solver_stub, coeffs=None, need_matrix=False)
    t_workset = time.perf_counter()
    local = op.assemble_local(workset)
    t_assemble = time.perf_counter()
    # Runtime operators expose the Newton residual sign. Reaction reconstruction
    # needs the raw Kratos local-system RHS convention used by the legacy helper.
    F_elem = None if local.F_elem is None else -np.asarray(local.F_elem, dtype=float)
    rhs = np.zeros(int(dh.total_dofs), dtype=float)
    scatter_owner = NewtonSolver.__new__(NewtonSolver)
    _, rhs = NewtonSolver.scatter_element_contribs_full(
        scatter_owner,
        K_elem=None,
        F_elem=F_elem,
        element_ids=np.asarray(local.element_ids, dtype=int),
        gdofs_map=np.asarray(local.gdofs_map, dtype=int),
        A_full=None,
        R_full=rhs,
    )
    if _env_bool("PYCUTFEM_EX2_STAGE_TIMING", False):
        _log(
            True,
            "[timing] reaction_residual "
            f"elements={int(np.asarray(element_ids).size)} "
            f"ids={t_ids - t0:.6f}s "
            f"operator={t_op - t_ids:.6f}s "
            f"workset={t_workset - t_op:.6f}s "
            f"assemble={t_assemble - t_workset:.6f}s "
            f"scatter={time.perf_counter() - t_assemble:.6f}s",
        )
    return rhs


def _can_use_cached_fluid_reaction_operator(prob: dict[str, object]) -> bool:
    dh = prob.get("dh")
    u_k = prob.get("u_k")
    if dh is None or u_k is None:
        return False
    if not hasattr(dh, "element_maps") or not hasattr(dh, "mixed_element") or not hasattr(dh, "total_dofs"):
        return False
    if not hasattr(u_k, "components"):
        return False
    return all(key in prob for key in ("p_k", "d_mesh", "d_prev"))


def _fluid_interface_constrained_reaction_vector(
    *,
    prob: dict[str, object],
    system_rhs: np.ndarray,
    interface_tag: str,
) -> np.ndarray:
    rhs = np.asarray(system_rhs, dtype=float).reshape(-1)
    reaction = np.zeros_like(rhs)
    interface_rows = _fluid_interface_velocity_dofs(prob, interface_tag=interface_tag)
    if interface_rows.size == 0:
        return reaction

    dh: DofHandler = prob["dh"]
    bcs_apply = prob.get("_current_bcs_homog")
    if not bcs_apply:
        bcs_apply = prob.get("_current_bcs")
    if not bcs_apply:
        bcs_apply = prob.get("bcs_homog")
    if not bcs_apply:
        bcs_apply = prob.get("bcs")
    if _env_bool("PYCUTFEM_EX2_REACTION_ASSUME_INTERFACE_CONSTRAINED", True) and not bcs_apply:
        reaction[interface_rows] = -rhs[interface_rows]
        return reaction
    bc_map = dh.get_dirichlet_data(bcs_apply) or {}
    if bc_map:
        bc_rows = np.fromiter((int(gdof) for gdof in bc_map.keys()), dtype=int)
        constrained_rows = np.intersect1d(interface_rows, bc_rows, assume_unique=False)
    else:
        constrained_rows = interface_rows
    if constrained_rows.size == 0:
        return reaction

    # Kratos' monolithic builder writes REACTION from the constrained system
    # using CalculateReactions(...), which on the fixed velocity rows gives
    # reaction = -b for the final linearized system RHS.
    reaction[constrained_rows] = -rhs[constrained_rows]
    return reaction


def _fluid_interface_reaction_loads(
    *,
    prob: dict[str, object],
    rho_f: float,
    mu_f: float,
    dt: float,
    quad_order: int,
    bossak_alpha: float = -0.3,
    dynamic_tau: float = 1.0,
    interface_tag: str,
    backend: str = "python",
    contribution_mode: str = "system",
    apply_dirichlet_lift: bool = False,
    refresh_state: bool = True,
) -> CoordinateLookup:
    state_snapshot = _snapshot_fluid_dvms_state(prob.get("dvms_state")) if bool(refresh_state) else None
    try:
        if bool(refresh_state):
            _refresh_fluid_reaction_reconstruction_state(
                prob=prob,
                rho_f=float(rho_f),
                mu_f=float(mu_f),
                dt=float(dt),
                bossak_alpha=float(bossak_alpha),
                dynamic_tau=float(dynamic_tau),
                backend=str(backend),
            )
        use_cached_reaction_operator = (
            (not bool(apply_dirichlet_lift)) and _can_use_cached_fluid_reaction_operator(prob)
        )
        if use_cached_reaction_operator:
            raw_residual = _assemble_fluid_reaction_residual_cached(
                prob=prob,
                rho_f=float(rho_f),
                mu_f=float(mu_f),
                dt=float(dt),
                quad_order=int(quad_order),
                bossak_alpha=float(bossak_alpha),
                interface_tag=interface_tag,
                backend=str(backend),
                contribution_mode=str(contribution_mode),
            )
        else:
            element_ids = (
                _fluid_interface_reaction_element_ids(prob, interface_tag=interface_tag)
                if _can_use_cached_fluid_reaction_operator(prob)
                else None
            )
            _, raw_residual = _call_with_supported_keywords(
                _assemble_fluid_local_velocity_contribution_raw,
                prob=prob,
                rho_f=float(rho_f),
                mu_f=float(mu_f),
                dt=float(dt),
                quad_order=int(quad_order),
                bossak_alpha=float(bossak_alpha),
                need_matrix=False,
                contribution_mode=str(contribution_mode),
                apply_dirichlet_lift=bool(apply_dirichlet_lift),
                backend=str(backend),
                element_ids=element_ids,
            )
    finally:
        if state_snapshot is not None:
            _restore_fluid_dvms_state(prob.get("dvms_state"), state_snapshot)
    reaction_vector = _fluid_interface_constrained_reaction_vector(
        prob=prob,
        system_rhs=np.asarray(raw_residual, dtype=float),
        interface_tag=interface_tag,
    )
    coords, values = _boundary_vector_from_global_values(
        prob["dh"],
        vector=prob["u_k"],
        tag=interface_tag,
        global_values=reaction_vector,
    )
    return CoordinateLookup(coords, values, dim=2)


def _fluid_interface_point_loads(
    *,
    dh: DofHandler,
    mesh: Mesh,
    u: VectorFunction,
    p: Function,
    d_mesh: VectorFunction,
    interface_tag: str,
    mu_f: float,
    quad_order: int,
    backend: str = "python",
) -> CoordinateLookup:
    v_space = FunctionSpace("example2_local_interface_velocity_load", ["ux", "uy"], dim=1)
    v_test = VectorTestFunction(space=v_space, dof_handler=dh)
    dS_iface = dS(defined_on=mesh.edge_bitset(interface_tag), metadata={"q": int(quad_order)})
    kin = build_fluid_dvms_kinematics(
        u=u,
        d=d_mesh,
        d_prev=d_mesh,
        d_prev2=d_mesh,
        dt=Constant(1.0),
    )
    sigma = build_fluid_cauchy_stress(
        p=p,
        grad_u_phys=kin.grad_u_phys,
        div_u_phys=kin.div_u_phys,
        mu=Constant(float(mu_f)),
    )
    n_fluid = FacetNormal()
    traction_ref_solid = kin.J * dot(dot(sigma, kin.Finv.T), Constant(-1.0) * n_fluid)
    form = dot(traction_ref_solid, v_test) * dS_iface

    batch = FormCompiler(
        dh,
        quadrature_order=int(quad_order),
        backend=str(backend),
    ).assemble_local_contributions(form)

    rhs = np.zeros(dh.total_dofs, dtype=float)
    solver = NewtonSolver.__new__(NewtonSolver)
    solver.backend = str(backend)
    solver.full_to_red = np.arange(int(dh.total_dofs), dtype=int)
    _, rhs = NewtonSolver.scatter_element_contribs_full(
        solver,
        K_elem=None,
        F_elem=batch.F_elem,
        element_ids=batch.element_ids,
        gdofs_map=batch.gdofs_map,
        A_full=None,
        R_full=rhs,
    )

    iface_coords, iface_values = _boundary_vector_from_global_values(
        dh,
        vector=u,
        tag=interface_tag,
        global_values=rhs,
    )
    return CoordinateLookup(iface_coords, iface_values, dim=2)


def _fluid_interface_point_loads_on_solid(
    *,
    fluid_dh: DofHandler,
    fluid_mesh: Mesh,
    solid_mesh: Mesh,
    u: VectorFunction,
    p: Function,
    d_mesh: VectorFunction,
    solid_iface_coords: np.ndarray,
    interface_tag: str,
    mu_f: float,
    quad_order: int,
) -> CoordinateLookup:
    loads = _reference_interface_point_loads_from_lookup(
        mesh=solid_mesh,
        iface_coords=solid_iface_coords,
        tag=interface_tag,
        quad_order=quad_order,
        traction_callback=lambda xy, N_s: _fluid_reference_traction_at_point(
            dh=fluid_dh,
            mesh=fluid_mesh,
            u=u,
            p=p,
            d_mesh=d_mesh,
            point=xy,
            solid_ref_normal=N_s,
            mu_f=mu_f,
        ),
    )
    return CoordinateLookup(np.asarray(solid_iface_coords, dtype=float), loads, dim=2)


def _solid_interface_disp_velocity(
    *,
    dh: DofHandler,
    mesh: Mesh,
    d_curr: VectorFunction,
    d_prev: VectorFunction,
    iface_coords: np.ndarray,
    dt: float,
    v_prev_lookup: CoordinateLookup | None = None,
    a_prev_lookup: CoordinateLookup | None = None,
    bossak_alpha: float | None = None,
) -> tuple[CoordinateLookup, CoordinateLookup]:
    disp_vals = np.empty((iface_coords.shape[0], 2), dtype=float)
    disp_prev_vals = np.empty((iface_coords.shape[0], 2), dtype=float)
    for i, xy in enumerate(np.asarray(iface_coords, dtype=float)):
        disp_curr, _ = _eval_vector_with_grad(dh, mesh, d_curr, tuple(xy))
        disp_prev_val, _ = _eval_vector_with_grad(dh, mesh, d_prev, tuple(xy))
        disp_vals[i, :] = disp_curr
        disp_prev_vals[i, :] = disp_prev_val
    return _interface_disp_velocity_from_values(
        iface_coords=iface_coords,
        disp_vals=disp_vals,
        disp_prev_vals=disp_prev_vals,
        dt=dt,
        v_prev_lookup=v_prev_lookup,
        a_prev_lookup=a_prev_lookup,
        bossak_alpha=bossak_alpha,
    )


def _interface_disp_velocity_from_values(
    *,
    iface_coords: np.ndarray,
    disp_vals: np.ndarray,
    disp_prev_vals: np.ndarray,
    dt: float,
    v_prev_lookup: CoordinateLookup | None = None,
    a_prev_lookup: CoordinateLookup | None = None,
    bossak_alpha: float | None = None,
) -> tuple[CoordinateLookup, CoordinateLookup]:
    iface_coords_arr = np.asarray(iface_coords, dtype=float)
    disp_curr_arr = np.asarray(disp_vals, dtype=float).reshape(-1, 2)
    disp_prev_arr = np.asarray(disp_prev_vals, dtype=float).reshape(-1, 2)
    if disp_curr_arr.shape != disp_prev_arr.shape or disp_curr_arr.shape[0] != iface_coords_arr.shape[0]:
        raise ValueError("interface displacement arrays must match interface coordinates")
    vel_vals = np.empty_like(disp_curr_arr)
    if v_prev_lookup is not None and a_prev_lookup is not None and bossak_alpha is not None:
        prev_vel_vals = np.asarray(
            v_prev_lookup(iface_coords_arr[:, 0], iface_coords_arr[:, 1]),
            dtype=float,
        ).reshape(-1, 2)
        prev_acc_vals = np.asarray(
            a_prev_lookup(iface_coords_arr[:, 0], iface_coords_arr[:, 1]),
            dtype=float,
        ).reshape(-1, 2)
        vel_vals[:, :], _ = _bossak_displacement_kinematics_values(
            d_curr=disp_curr_arr,
            d_prev=disp_prev_arr,
            v_prev=prev_vel_vals,
            a_prev=prev_acc_vals,
            dt=dt,
            alpha=float(bossak_alpha),
        )
    else:
        vel_vals[:, :] = (disp_curr_arr - disp_prev_arr) / max(float(dt), 1.0e-14)
    return CoordinateLookup(iface_coords_arr, disp_curr_arr, dim=2), CoordinateLookup(iface_coords_arr, vel_vals, dim=2)


def _assign_nirb_full_displacement(
    *,
    predictor: NIRBSolidPredictor,
    prediction,
    solid_displacement: VectorFunction,
) -> np.ndarray:
    if prediction.full_displacement is not None:
        full_displacement = np.asarray(prediction.full_displacement, dtype=float).reshape(-1)
    else:
        if prediction.reduced_displacement is None:
            raise RuntimeError("NIRB prediction does not contain reduced coordinates for full reconstruction.")
        full_displacement = np.asarray(
            predictor.reconstruct_full(prediction.reduced_displacement),
            dtype=float,
        ).reshape(-1)
    if not np.all(np.isfinite(full_displacement)):
        raise RuntimeError("NIRB full displacement reconstruction returned non-finite values.")
    target_shape = np.asarray(solid_displacement.nodal_values).shape
    if full_displacement.size != int(np.prod(target_shape)):
        raise RuntimeError(
            "NIRB full displacement size does not match the solid field: "
            f"{full_displacement.size} != {int(np.prod(target_shape))}"
        )
    solid_displacement.nodal_values[:] = full_displacement.reshape(target_shape)
    return full_displacement


def _bossak_displacement_kinematics_values(
    *,
    d_curr: np.ndarray,
    d_prev: np.ndarray,
    v_prev: np.ndarray,
    a_prev: np.ndarray,
    dt: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    dt_value = max(float(dt), 1.0e-14)
    coeffs = _bossak_coefficients(alpha=float(alpha), dt=dt_value)
    beta = float(coeffs["beta"])
    gamma = float(coeffs["gamma"])
    d_curr_arr = np.asarray(d_curr, dtype=float)
    d_prev_arr = np.asarray(d_prev, dtype=float)
    v_prev_arr = np.asarray(v_prev, dtype=float)
    a_prev_arr = np.asarray(a_prev, dtype=float)
    a_curr = (
        d_curr_arr
        - d_prev_arr
        - dt_value * v_prev_arr
        - (dt_value * dt_value) * (0.5 - beta) * a_prev_arr
    ) / (beta * dt_value * dt_value)
    v_curr = v_prev_arr + dt_value * ((1.0 - gamma) * a_prev_arr + gamma * a_curr)
    return v_curr, a_curr


def _vector_history_full_lookup(
    *,
    dh: DofHandler,
    d_curr: VectorFunction,
    d_prev: VectorFunction,
    v_prev: VectorFunction,
    a_prev: VectorFunction,
    dt: float,
    alpha: float,
) -> tuple[CoordinateLookup, CoordinateLookup]:
    curr_coords, curr_vals = _vector_field_matrix(dh, d_curr)
    _, prev_vals = _vector_field_matrix(dh, d_prev)
    _, vel_prev_vals = _vector_field_matrix(dh, v_prev)
    _, acc_prev_vals = _vector_field_matrix(dh, a_prev)
    vel_curr_vals, acc_curr_vals = _bossak_displacement_kinematics_values(
        d_curr=curr_vals,
        d_prev=prev_vals,
        v_prev=vel_prev_vals,
        a_prev=acc_prev_vals,
        dt=dt,
        alpha=alpha,
    )
    return CoordinateLookup(curr_coords, vel_curr_vals, dim=2), CoordinateLookup(curr_coords, acc_curr_vals, dim=2)


def _solid_velocity_point_data(
    *,
    dh: DofHandler,
    d_curr: VectorFunction,
    d_prev_values: np.ndarray,
    dt: float,
) -> np.ndarray:
    current = _vector_point_data_from_function(dh, d_curr)
    previous_fun_values = np.asarray(d_prev_values, dtype=float)
    original = np.asarray(d_curr.nodal_values, dtype=float).copy()
    try:
        d_curr.nodal_values[:] = previous_fun_values.reshape(np.asarray(d_curr.nodal_values).shape)
        previous = _vector_point_data_from_function(dh, d_curr)
    finally:
        d_curr.nodal_values[:] = original
    return (current - previous) / max(float(dt), 1.0e-14)


def _darcy_flux_point_data_from_pressure(
    *,
    dh: DofHandler,
    mesh: Mesh,
    pressure: Function,
    material: UPlMaterial2D,
) -> np.ndarray:
    """Return nodal Darcy discharge ``q = -K/mu grad(p)`` for VTK output."""

    conductivity = np.asarray(material.darcy_conductivity_matrix, dtype=float).reshape(2, 2)
    out = np.zeros((int(len(mesh.nodes_list)), 2), dtype=float)
    counts = np.zeros((int(len(mesh.nodes_list)),), dtype=float)
    me = dh.mixed_element
    field = str(pressure.field_name)
    field_slice = me.slice(field)
    conn_all = np.asarray(getattr(mesh, "elements_connectivity", mesh.corner_connectivity), dtype=int)
    for elem in mesh.elements_list:
        eid = int(elem.id)
        if eid < 0 or eid >= int(conn_all.shape[0]):
            continue
        gdofs = dh.element_maps[field][eid]
        vals = pressure.get_nodal_values(gdofs)
        for node_id in np.unique(conn_all[eid]):
            xy = np.asarray(mesh.nodes_x_y_pos[int(node_id)], dtype=float)
            try:
                xi, eta = transform.inverse_mapping(mesh, eid, xy)
                grad_ref = me.grad_basis(field, float(xi), float(eta))[field_slice]
                grad_phys = transform.map_grad_scalar(mesh, eid, grad_ref, (float(xi), float(eta)))
            except Exception:
                continue
            grad_p = np.asarray(vals, dtype=float) @ np.asarray(grad_phys, dtype=float)
            out[int(node_id), :] += -conductivity @ np.asarray(grad_p, dtype=float).reshape(2)
            counts[int(node_id)] += 1.0
    mask = counts > 0.0
    out[mask, :] /= counts[mask, None]
    return out


def _write_vtk_outputs(
    *,
    output_dir: Path,
    step: int,
    time_value: float,
    fluid: dict[str, object],
    solid: dict[str, object],
    geometry,
    returned_load_lookup: CoordinateLookup | None,
    dt: float,
    porous_material: UPlMaterial2D | None = None,
    solid_prev_displacement_values: np.ndarray | None = None,
) -> tuple[Path, Path]:
    vtk_root = Path(output_dir) / "vtk_data"
    fluid_dir = vtk_root / "vtk_output_fsi_cfd"
    solid_dir = vtk_root / "vtk_output_fsi_csm"
    fluid_dir.mkdir(parents=True, exist_ok=True)
    solid_dir.mkdir(parents=True, exist_ok=True)

    fluid_path = fluid_dir / f"FluidParts_FluidPart_0_{int(step):04d}.vtu"
    solid_path = solid_dir / f"Structure_0_{int(step):04d}.vtu"

    fluid_mesh_displacement = _vector_point_data_from_function(fluid["dh"], fluid["d_mesh"])
    export_vtk(
        filename=str(fluid_path),
        mesh=fluid["mesh"],
        dof_handler=fluid["dh"],
        functions={
            "VELOCITY": fluid["u_k"],
            "PRESSURE": fluid["p_k"],
            "MESH_DISPLACEMENT": fluid["d_mesh"],
            "MESH_VELOCITY": fluid["w_mesh_prev"],
            "ACCELERATION": fluid["a_prev"],
            "TIME": np.full(int(len(fluid["mesh"].nodes_list)), float(time_value), dtype=float),
        },
        point_displacement=fluid_mesh_displacement,
    )

    solid_displacement = _vector_point_data_from_function(solid["dh"], solid["d_k"])
    solid_fields: dict[str, object] = {
        "DISPLACEMENT": solid["d_k"],
        "TIME": np.full(int(len(solid["mesh"].nodes_list)), float(time_value), dtype=float),
    }
    solid_velocity = None
    if solid_prev_displacement_values is not None:
        solid_velocity = _solid_velocity_point_data(
            dh=solid["dh"],
            d_curr=solid["d_k"],
            d_prev_values=np.asarray(solid_prev_displacement_values, dtype=float),
            dt=float(dt),
        )
        solid_fields["VS"] = solid_velocity
        solid_fields["SOLID_VELOCITY"] = solid_velocity
    if "p_k" in solid:
        solid_fields["LIQUID_PRESSURE"] = solid["p_k"]
        if porous_material is not None:
            darcy_flux = _darcy_flux_point_data_from_pressure(
                dh=solid["dh"],
                mesh=solid["mesh"],
                pressure=solid["p_k"],
                material=porous_material,
            )
            solid_fields["DARCY_FLUX"] = darcy_flux
            if solid_velocity is not None:
                porosity = max(float(porous_material.porosity), 1.0e-14)
                solid_fields["PORE_VELOCITY"] = solid_velocity + darcy_flux / porosity
    if returned_load_lookup is not None:
        solid_point_load_full = _boundary_point_load_vector(
            dh=solid["dh"],
            vector=solid["d_k"],
            tag=geometry.interface_tag,
            values=np.asarray(returned_load_lookup.values, dtype=float),
        )
        solid_fields["POINT_LOAD"] = _nodal_vector_point_data_from_global_values(
            solid["dh"],
            vector=solid["d_k"],
            global_values=solid_point_load_full,
        )
    export_vtk(
        filename=str(solid_path),
        mesh=solid["mesh"],
        dof_handler=solid["dh"],
        functions=solid_fields,
        point_displacement=solid_displacement,
    )
    return fluid_path, solid_path


def _snapshot_function_values(functions: list[Function | VectorFunction]) -> list[np.ndarray]:
    return [np.asarray(function.nodal_values, dtype=float).copy() for function in functions]


def _restore_function_values(functions: list[Function | VectorFunction], snapshots: list[np.ndarray]) -> None:
    if len(functions) != len(snapshots):
        raise ValueError("functions / snapshots length mismatch")
    for function, values in zip(functions, snapshots):
        function.nodal_values[:] = np.asarray(values, dtype=float)


def _predict_fluid_bossak_step_state(
    *,
    fluid: dict[str, object],
    dt: float,
    bossak_alpha: float,
    predictor_bcs: list[BoundaryCondition] | None = None,
) -> None:
    """Match the Kratos fluid Bossak Predict() call made once per time step.

    Kratos applies the process-level Dirichlet conditions before the underlying
    fluid solver Predict() call. The ALE interface velocity copy happens later,
    immediately before solving the fluid, and does not refresh ACCELERATION.
    """
    bossak = _bossak_coefficients(alpha=float(bossak_alpha), dt=max(float(dt), 1.0e-14))
    fluid["u_k"].nodal_values[:] = fluid["u_prev"].nodal_values[:]
    fluid["p_k"].nodal_values[:] = fluid["p_prev"].nodal_values[:]
    if predictor_bcs:
        _apply_dirichlet_bcs_to_state(
            dh=fluid["dh"],
            field_functions={
                str(fluid["u_k"].components[0].field_name): fluid["u_k"].components[0],
                str(fluid["u_k"].components[1].field_name): fluid["u_k"].components[1],
                str(fluid["p_k"].field_name): fluid["p_k"],
            },
            bcs=predictor_bcs,
        )
    fluid["a_k"].nodal_values[:] = (
        float(bossak["ma0"])
        * (fluid["u_k"].nodal_values[:] - fluid["u_prev"].nodal_values[:])
        + float(bossak["ma2"]) * fluid["a_prev"].nodal_values[:]
    )


def _snapshot_fluid_dvms_state(state: FluidDVMSState | None) -> dict[str, np.ndarray] | None:
    if not isinstance(state, FluidDVMSState):
        return None
    snapshot = {
        "old_subscale_velocity": np.asarray(state.old_subscale_velocity, dtype=float).copy(),
        "predicted_subscale_velocity": np.asarray(state.predicted_subscale_velocity, dtype=float).copy(),
        "momentum_projection": np.asarray(state.momentum_projection, dtype=float).copy(),
        "mass_projection": np.asarray(state.mass_projection, dtype=float).copy(),
        "old_mass_residual": np.asarray(state.old_mass_residual, dtype=float).copy(),
    }
    nodal_momentum_projection = getattr(state, "_nodal_momentum_projection", None)
    nodal_div_projection = getattr(state, "_nodal_div_projection", None)
    prev_nodal_div_projection = getattr(state, "_prev_nodal_div_projection", None)
    if nodal_momentum_projection is not None:
        snapshot["nodal_momentum_projection"] = np.asarray(nodal_momentum_projection, dtype=float).copy()
    if nodal_div_projection is not None:
        snapshot["nodal_div_projection"] = np.asarray(nodal_div_projection, dtype=float).copy()
    if prev_nodal_div_projection is not None:
        snapshot["prev_nodal_div_projection"] = np.asarray(prev_nodal_div_projection, dtype=float).copy()
    return snapshot


def _restore_fluid_dvms_state(state: FluidDVMSState | None, snapshot: dict[str, np.ndarray] | None) -> None:
    if not isinstance(state, FluidDVMSState) or snapshot is None:
        return
    nodal_attr_map = {
        "nodal_momentum_projection": "_nodal_momentum_projection",
        "nodal_div_projection": "_nodal_div_projection",
        "prev_nodal_div_projection": "_prev_nodal_div_projection",
    }
    for key, values in snapshot.items():
        if key in nodal_attr_map:
            setattr(state, nodal_attr_map[key], np.asarray(values, dtype=float).copy())
            continue
        getattr(state, key)[:, ...] = np.asarray(values, dtype=float)
    state.sync_coefficients_from_samples()


def _restore_fluid_dvms_state_except(
    state: FluidDVMSState | None,
    snapshot: dict[str, np.ndarray] | None,
    *,
    skip_keys: set[str] | None = None,
) -> None:
    if not isinstance(state, FluidDVMSState) or snapshot is None:
        return
    skip = set() if skip_keys is None else {str(key) for key in skip_keys}
    nodal_attr_map = {
        "nodal_momentum_projection": "_nodal_momentum_projection",
        "nodal_div_projection": "_nodal_div_projection",
        "prev_nodal_div_projection": "_prev_nodal_div_projection",
    }
    for key, values in snapshot.items():
        if key in skip:
            continue
        if key in nodal_attr_map:
            setattr(state, nodal_attr_map[key], np.asarray(values, dtype=float).copy())
            continue
        getattr(state, key)[:, ...] = np.asarray(values, dtype=float)
    state.sync_coefficients_from_samples()


def _fluid_dvms_restart_snapshot_from_payload(
    state: FluidDVMSState | None,
    restart_payload: dict[str, np.ndarray],
) -> dict[str, np.ndarray] | None:
    if not isinstance(state, FluidDVMSState):
        return None
    snapshot: dict[str, np.ndarray] = {
        "old_subscale_velocity": np.asarray(
            restart_payload.get("dvms_old_subscale_velocity", np.zeros_like(state.old_subscale_velocity)),
            dtype=float,
        ),
        "predicted_subscale_velocity": np.asarray(
            restart_payload.get("dvms_predicted_subscale_velocity", np.zeros_like(state.predicted_subscale_velocity)),
            dtype=float,
        ),
        "momentum_projection": np.asarray(
            restart_payload.get("dvms_momentum_projection", np.zeros_like(state.momentum_projection)),
            dtype=float,
        ),
        "mass_projection": np.asarray(
            restart_payload.get("dvms_mass_projection", np.zeros_like(state.mass_projection)),
            dtype=float,
        ),
        "old_mass_residual": np.asarray(
            restart_payload.get("dvms_old_mass_residual", np.zeros_like(state.old_mass_residual)),
            dtype=float,
        ),
    }
    optional_nodal_keys = (
        "nodal_momentum_projection",
        "nodal_div_projection",
        "prev_nodal_div_projection",
    )
    for key in optional_nodal_keys:
        payload_key = f"dvms_{key}"
        if payload_key in restart_payload:
            snapshot[key] = np.asarray(restart_payload[payload_key], dtype=float)
    return snapshot


def _snapshot_fluid_stage_state(prob: dict[str, object]) -> dict[str, object]:
    return {
        "u_k": np.asarray(prob["u_k"].nodal_values, dtype=float).copy(),
        "p_k": np.asarray(prob["p_k"].nodal_values, dtype=float).copy(),
        "u_prev": np.asarray(prob["u_prev"].nodal_values, dtype=float).copy(),
        "p_prev": np.asarray(prob["p_prev"].nodal_values, dtype=float).copy(),
        "a_prev": np.asarray(prob["a_prev"].nodal_values, dtype=float).copy(),
        "a_k": np.asarray(prob["a_k"].nodal_values, dtype=float).copy(),
        "d_mesh": np.asarray(prob["d_mesh"].nodal_values, dtype=float).copy(),
        "d_prev": np.asarray(prob["d_prev"].nodal_values, dtype=float).copy(),
        "d_prev2": np.asarray(prob["d_prev2"].nodal_values, dtype=float).copy(),
        "w_mesh_prev": np.asarray(prob["w_mesh_prev"].nodal_values, dtype=float).copy(),
        "a_mesh_prev": np.asarray(prob["a_mesh_prev"].nodal_values, dtype=float).copy(),
        "w_mesh_k": np.asarray(prob["w_mesh_k"].nodal_values, dtype=float).copy(),
        "a_mesh_k": np.asarray(prob["a_mesh_k"].nodal_values, dtype=float).copy(),
        "dvms_state": _snapshot_fluid_dvms_state(prob.get("dvms_state")),
    }


def _restore_fluid_stage_state(prob: dict[str, object], snapshot: dict[str, object] | None) -> None:
    if not isinstance(snapshot, dict):
        return
    prob["u_k"].nodal_values[:] = np.asarray(snapshot["u_k"], dtype=float)
    prob["p_k"].nodal_values[:] = np.asarray(snapshot["p_k"], dtype=float)
    prob["u_prev"].nodal_values[:] = np.asarray(snapshot["u_prev"], dtype=float)
    prob["p_prev"].nodal_values[:] = np.asarray(snapshot["p_prev"], dtype=float)
    prob["a_prev"].nodal_values[:] = np.asarray(snapshot["a_prev"], dtype=float)
    prob["a_k"].nodal_values[:] = np.asarray(snapshot["a_k"], dtype=float)
    prob["d_mesh"].nodal_values[:] = np.asarray(snapshot["d_mesh"], dtype=float)
    prob["d_prev"].nodal_values[:] = np.asarray(snapshot["d_prev"], dtype=float)
    prob["d_prev2"].nodal_values[:] = np.asarray(snapshot["d_prev2"], dtype=float)
    prob["w_mesh_prev"].nodal_values[:] = np.asarray(snapshot["w_mesh_prev"], dtype=float)
    prob["a_mesh_prev"].nodal_values[:] = np.asarray(snapshot["a_mesh_prev"], dtype=float)
    prob["w_mesh_k"].nodal_values[:] = np.asarray(snapshot["w_mesh_k"], dtype=float)
    prob["a_mesh_k"].nodal_values[:] = np.asarray(snapshot["a_mesh_k"], dtype=float)
    _restore_fluid_dvms_state(prob.get("dvms_state"), snapshot.get("dvms_state"))


def _column_list_from_matrix(matrix: np.ndarray) -> list[np.ndarray]:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0 or arr.shape[1] == 0:
        return []
    return [np.asarray(arr[:, j], dtype=float).copy() for j in range(int(arr.shape[1]))]


def _load_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv_rows(path: Path, *, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_vtk_pvd_collection(path: Path, *, rows: list[dict[str, object]], vtk_key: str) -> None:
    """Write a ParaView time-series collection for the accepted-step VTK files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    base_dir = path.parent
    for row in rows:
        vtk_raw = str(row.get(vtk_key, "") or "").strip()
        if not vtk_raw:
            continue
        try:
            time_value = float(row.get("time_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            time_value = 0.0
        vtk_path = Path(vtk_raw)
        if vtk_path.is_absolute():
            file_name = os.path.relpath(vtk_path, base_dir)
        else:
            file_name = os.path.relpath((Path.cwd() / vtk_path).resolve(), base_dir.resolve())
        lines.append(
            f'    <DataSet timestep="{time_value:.16g}" group="" part="0" file="{_xml_escape(file_name)}"/>'
        )
    lines.extend(["  </Collection>", "</VTKFile>", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _iqn_history_to_payload(prefix: str, history: deque[np.ndarray]) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        f"{prefix}_count": np.asarray(int(len(history)), dtype=int),
    }
    for idx, mat in enumerate(list(history)):
        payload[f"{prefix}_{idx}"] = np.asarray(mat, dtype=float)
    return payload


def _iqn_history_from_payload(
    *,
    data: dict[str, np.ndarray],
    prefix: str,
    maxlen: int,
) -> deque[np.ndarray]:
    history: deque[np.ndarray] = deque(maxlen=max(int(maxlen), 0))
    count = int(np.asarray(data.get(f"{prefix}_count", np.asarray(0)), dtype=int).reshape(-1)[0])
    for idx in range(max(count, 0)):
        key = f"{prefix}_{idx}"
        if key in data:
            history.append(np.asarray(data[key], dtype=float).copy())
    return history


def _checkpoint_payload(
    *,
    step: int,
    time_s: float,
    solid: dict[str, object],
    fluid: dict[str, object],
    current_load_lookup: CoordinateLookup,
    iqn_old_dr_mats: deque[np.ndarray],
    iqn_old_dg_mats: deque[np.ndarray],
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "checkpoint_schema_version": np.asarray(int(_CHECKPOINT_SCHEMA_VERSION), dtype=int),
        "step": np.asarray(int(step), dtype=int),
        "time_s": np.asarray(float(time_s), dtype=float),
        "solid_d_k": np.asarray(solid["d_k"].nodal_values, dtype=float).copy(),
        "solid_d_prev": np.asarray(solid["d_prev"].nodal_values, dtype=float).copy(),
        "fluid_u_k": np.asarray(fluid["u_k"].nodal_values, dtype=float).copy(),
        "fluid_p_k": np.asarray(fluid["p_k"].nodal_values, dtype=float).copy(),
        "fluid_u_prev": np.asarray(fluid["u_prev"].nodal_values, dtype=float).copy(),
        "fluid_p_prev": np.asarray(fluid["p_prev"].nodal_values, dtype=float).copy(),
        "fluid_a_prev": np.asarray(fluid["a_prev"].nodal_values, dtype=float).copy(),
        "fluid_a_k": np.asarray(fluid["a_k"].nodal_values, dtype=float).copy(),
        "fluid_d_mesh": np.asarray(fluid["d_mesh"].nodal_values, dtype=float).copy(),
        "fluid_d_prev": np.asarray(fluid["d_prev"].nodal_values, dtype=float).copy(),
        "fluid_d_prev2": np.asarray(fluid["d_prev2"].nodal_values, dtype=float).copy(),
        "fluid_w_mesh_prev": np.asarray(fluid["w_mesh_prev"].nodal_values, dtype=float).copy(),
        "fluid_a_mesh_prev": np.asarray(fluid["a_mesh_prev"].nodal_values, dtype=float).copy(),
        "current_load_values": np.asarray(current_load_lookup.values, dtype=float).copy(),
    }
    if "p_k" in solid:
        payload["solid_p_k"] = np.asarray(solid["p_k"].nodal_values, dtype=float).copy()
    if "p_prev" in solid:
        payload["solid_p_prev"] = np.asarray(solid["p_prev"].nodal_values, dtype=float).copy()
    dvms_snapshot = _snapshot_fluid_dvms_state(fluid.get("dvms_state"))
    if isinstance(dvms_snapshot, dict):
        for key, values in dvms_snapshot.items():
            payload[f"dvms_{key}"] = np.asarray(values, dtype=float)
    payload.update(_iqn_history_to_payload("iqn_old_dr", iqn_old_dr_mats))
    payload.update(_iqn_history_to_payload("iqn_old_dg", iqn_old_dg_mats))
    return payload


def _write_checkpoint(
    *,
    checkpoint_dir: Path,
    step: int,
    time_s: float,
    solid: dict[str, object],
    fluid: dict[str, object],
    current_load_lookup: CoordinateLookup,
    iqn_old_dr_mats: deque[np.ndarray],
    iqn_old_dg_mats: deque[np.ndarray],
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{int(step):04d}.npz"
    np.savez_compressed(
        checkpoint_path,
        **_checkpoint_payload(
            step=int(step),
            time_s=float(time_s),
            solid=solid,
            fluid=fluid,
            current_load_lookup=current_load_lookup,
            iqn_old_dr_mats=iqn_old_dr_mats,
            iqn_old_dg_mats=iqn_old_dg_mats,
        ),
    )
    (checkpoint_dir / "latest_checkpoint.txt").write_text(checkpoint_path.name, encoding="utf-8")
    return checkpoint_path


def _load_checkpoint_payload(checkpoint_path: Path) -> dict[str, np.ndarray]:
    with np.load(checkpoint_path, allow_pickle=False) as data:
        return {str(k): np.asarray(data[k]) for k in data.files}


def _flush_progress_artifacts(
    *,
    co_sim_dir: Path,
    output_dir: Path,
    disp_snapshots: list[np.ndarray],
    load_snapshots: list[np.ndarray],
    load_guess_snapshots: list[np.ndarray],
    load_return_snapshots: list[np.ndarray],
    fluid_load_guess_snapshots: list[np.ndarray],
    fluid_load_return_snapshots: list[np.ndarray],
    interface_disp_snapshots: list[np.ndarray],
    interface_velocity_snapshots: list[np.ndarray],
    interface_traction_snapshots: list[np.ndarray],
    reaction_load_snapshots: list[np.ndarray],
    stress_load_snapshots: list[np.ndarray],
    snapshot_rows: list[dict[str, object]],
    step_rows: list[dict[str, object]],
    vtk_rows: list[dict[str, object]],
    coupling_iters_per_step: list[int],
    fluid_times: list[float],
    structure_times: list[float],
    increment_times: list[float],
    monitor_interface_loads: bool,
) -> None:
    disp_matrix = np.column_stack(disp_snapshots) if disp_snapshots else np.zeros((0, 0), dtype=float)
    load_matrix = np.column_stack(load_snapshots) if load_snapshots else np.zeros((0, 0), dtype=float)
    load_guess_matrix = np.column_stack(load_guess_snapshots) if load_guess_snapshots else np.zeros((0, 0), dtype=float)
    load_return_matrix = np.column_stack(load_return_snapshots) if load_return_snapshots else np.zeros((0, 0), dtype=float)
    fluid_load_guess_matrix = (
        np.column_stack(fluid_load_guess_snapshots) if fluid_load_guess_snapshots else np.zeros((0, 0), dtype=float)
    )
    fluid_load_return_matrix = (
        np.column_stack(fluid_load_return_snapshots) if fluid_load_return_snapshots else np.zeros((0, 0), dtype=float)
    )
    interface_disp_matrix = np.column_stack(interface_disp_snapshots) if interface_disp_snapshots else np.zeros((0, 0), dtype=float)
    interface_velocity_matrix = np.column_stack(interface_velocity_snapshots) if interface_velocity_snapshots else np.zeros((0, 0), dtype=float)
    interface_traction_matrix = np.column_stack(interface_traction_snapshots) if interface_traction_snapshots else np.zeros((0, 0), dtype=float)
    reaction_load_matrix = np.column_stack(reaction_load_snapshots) if reaction_load_snapshots else np.zeros((0, 0), dtype=float)
    stress_load_matrix = np.column_stack(stress_load_snapshots) if stress_load_snapshots else np.zeros((0, 0), dtype=float)
    np.save(co_sim_dir / "disp_data.npy", disp_matrix)
    np.save(co_sim_dir / "load_data.npy", load_matrix)
    np.save(co_sim_dir / "load_guess_data.npy", load_guess_matrix)
    np.save(co_sim_dir / "load_return_data.npy", load_return_matrix)
    np.save(co_sim_dir / "fluid_load_guess_data.npy", fluid_load_guess_matrix)
    np.save(co_sim_dir / "fluid_load_return_data.npy", fluid_load_return_matrix)
    np.save(co_sim_dir / "interface_disp_data.npy", interface_disp_matrix)
    np.save(co_sim_dir / "interface_velocity_data.npy", interface_velocity_matrix)
    np.save(co_sim_dir / "interface_traction_data.npy", interface_traction_matrix)
    if bool(monitor_interface_loads):
        np.save(co_sim_dir / "load_reaction_data.npy", reaction_load_matrix)
        np.save(co_sim_dir / "load_stress_data.npy", stress_load_matrix)
    np.save(co_sim_dir / "iters.npy", np.asarray(coupling_iters_per_step, dtype=int))
    np.save(co_sim_dir / "fluid_time.npy", np.asarray(fluid_times, dtype=float))
    np.save(co_sim_dir / "structure_time.npy", np.asarray(structure_times, dtype=float))
    np.save(co_sim_dir / "increment_time.npy", np.asarray(increment_times, dtype=float))

    _write_csv_rows(
        output_dir / "snapshot_metadata.csv",
        fieldnames=["step", "time_s", "coupling_iter", "converged"],
        rows=snapshot_rows,
    )
    _write_csv_rows(
        output_dir / "vtk_manifest.csv",
        fieldnames=["step", "time_s", "fluid_vtk", "solid_vtk"],
        rows=vtk_rows,
    )
    _write_vtk_pvd_collection(output_dir / "vtk_data" / "fluid_timeseries.pvd", rows=vtk_rows, vtk_key="fluid_vtk")
    _write_vtk_pvd_collection(output_dir / "vtk_data" / "solid_timeseries.pvd", rows=vtk_rows, vtk_key="solid_vtk")
    _write_csv_rows(
        output_dir / "timeseries.csv",
        fieldnames=[
            "step",
            "time_s",
            "attempt",
            "coupling_iter",
            "disp_abs",
            "disp_rel",
            "load_abs",
            "load_rel",
            "solid_time_s",
            "fluid_time_s",
            "disp_max",
            "load_guess_max",
            "load_return_max",
            "force_update_active",
            "force_omega",
            "fluid_hrom_used",
            "fluid_hrom_trial_used",
            "fluid_hrom_trial_load_rel_error",
            "fluid_hrom_disabled_by_trial_monitor",
            "fluid_hrom_exact_accept_forced",
            "fluid_hrom_interface_trust_alpha",
            "fluid_hrom_interface_load_rel",
            "fluid_hrom_interface_step_ratio",
            "fluid_hrom_interface_corrected",
            "fluid_hrom_interface_rejected",
            "fluid_hrom_interface_reason",
            "strict_converged",
            "kratos_disp_converged_5e-3",
            "kratos_load_converged_5e-3",
            "kratos_step_converged_5e-3",
        ],
        rows=step_rows,
    )


def _guess_callback_from_snapshots(snapshots: list[np.ndarray]):
    def _callback(*, functions, **kwargs) -> None:
        del kwargs
        _restore_function_values(list(functions), snapshots)

    return _callback


def _apply_dirichlet_guess(
    *,
    dh: DofHandler,
    functions: list[Function | VectorFunction],
    bcs: list[BoundaryCondition] | None,
) -> None:
    if not bcs:
        return
    bc_map = dh.get_dirichlet_data(bcs) or {}
    if not bc_map:
        return
    bc_values_full = np.zeros(int(dh.total_dofs), dtype=float)
    bc_ids = np.fromiter((int(gdof) for gdof in bc_map.keys()), dtype=int)
    bc_vals = np.fromiter((float(val) for val in bc_map.values()), dtype=float)
    bc_values_full[bc_ids] = bc_vals

    scalar_functions: list[Function] = []
    for function in list(functions):
        if isinstance(function, VectorFunction):
            scalar_functions.extend(list(function.components))
        else:
            scalar_functions.append(function)

    for function in scalar_functions:
        field_name = str(function.field_name)
        field_slice = np.asarray(dh.get_field_slice(field_name), dtype=int)
        if field_slice.size == 0:
            continue
        values = np.asarray(function.nodal_values, dtype=float).copy()
        mask = np.isin(field_slice, bc_ids, assume_unique=False)
        if np.any(mask):
            values[mask] = np.asarray(bc_values_full[field_slice[mask]], dtype=float)
            function.nodal_values = values


def _guess_callback_from_snapshots_with_dirichlet(
    *,
    snapshots: list[np.ndarray],
    dh: DofHandler,
    bcs: list[BoundaryCondition] | None,
):
    def _callback(*, functions, **kwargs) -> None:
        del kwargs
        function_list = list(functions)
        _restore_function_values(function_list, snapshots)
        _apply_dirichlet_guess(dh=dh, functions=function_list, bcs=bcs)

    return _callback


def _field_abs_max(function: Function | VectorFunction) -> float:
    if isinstance(function, VectorFunction):
        values = np.asarray(function.nodal_values, dtype=float).reshape(-1)
    else:
        values = np.asarray(function.nodal_values, dtype=float).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def _relative_change(new: np.ndarray, old: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(new, dtype=float) - np.asarray(old, dtype=float)
    size = max(int(diff.size), 1)
    abs_norm = float(np.linalg.norm(diff.ravel(), ord=2) / math.sqrt(float(size)))
    rel_num = float(np.linalg.norm(diff.ravel(), ord=2))
    base = max(float(np.linalg.norm(np.asarray(new, dtype=float).ravel(), ord=2)), 1.0e-14)
    return abs_norm, rel_num / base


@dataclass(frozen=True)
class _FluidHROMInterfaceTrustResult:
    accepted: bool
    corrected: bool
    alpha: float
    update_abs: float
    update_rel: float
    step_ratio: float
    reason: str
    values: np.ndarray


def _fluid_hrom_interface_trust_region(
    *,
    current_values: np.ndarray,
    proposed_values: np.ndarray,
    previous_load_abs: float,
    mode: str,
    max_step_ratio: float,
    max_load_rel: float,
    min_correction_alpha: float,
) -> _FluidHROMInterfaceTrustResult:
    """Gate or clip a committed HROM interface load in the coupled norm."""

    current = np.asarray(current_values, dtype=float)
    proposed = np.asarray(proposed_values, dtype=float)
    if current.shape != proposed.shape:
        raise ValueError(f"Interface trust-region shape mismatch: {current.shape} != {proposed.shape}.")
    update_abs, update_rel = _relative_change(proposed, current)
    if not np.all(np.isfinite(proposed)) or not np.isfinite(update_abs) or not np.isfinite(update_rel):
        return _FluidHROMInterfaceTrustResult(
            accepted=False,
            corrected=False,
            alpha=0.0,
            update_abs=float(update_abs),
            update_rel=float(update_rel),
            step_ratio=float("inf"),
            reason="nonfinite_interface_load",
            values=proposed.copy(),
        )

    mode_value = str(mode).strip().lower()
    if mode_value == "none":
        return _FluidHROMInterfaceTrustResult(
            accepted=True,
            corrected=False,
            alpha=1.0,
            update_abs=float(update_abs),
            update_rel=float(update_rel),
            step_ratio=0.0,
            reason="disabled",
            values=proposed.copy(),
        )

    alpha = 1.0
    step_ratio = 0.0
    if np.isfinite(float(max_step_ratio)) and float(max_step_ratio) >= 0.0:
        if np.isfinite(float(previous_load_abs)) and float(previous_load_abs) > 1.0e-15:
            step_ratio = float(update_abs) / max(float(previous_load_abs), 1.0e-15)
            allowed_abs = float(max_step_ratio) * max(float(previous_load_abs), 1.0e-15)
            if float(update_abs) > allowed_abs and float(update_abs) > 0.0:
                alpha = min(float(alpha), float(allowed_abs) / float(update_abs))
    if np.isfinite(float(max_load_rel)) and float(max_load_rel) >= 0.0:
        if float(update_rel) > float(max_load_rel) and float(update_rel) > 0.0:
            alpha = min(float(alpha), float(max_load_rel) / float(update_rel))

    alpha = float(np.clip(float(alpha), 0.0, 1.0))
    if alpha >= 1.0 - 1.0e-14:
        return _FluidHROMInterfaceTrustResult(
            accepted=True,
            corrected=False,
            alpha=1.0,
            update_abs=float(update_abs),
            update_rel=float(update_rel),
            step_ratio=float(step_ratio),
            reason="accepted",
            values=proposed.copy(),
        )

    if mode_value == "fallback":
        return _FluidHROMInterfaceTrustResult(
            accepted=False,
            corrected=False,
            alpha=float(alpha),
            update_abs=float(update_abs),
            update_rel=float(update_rel),
            step_ratio=float(step_ratio),
            reason="outside_interface_trust_region",
            values=proposed.copy(),
        )
    if mode_value != "clip":
        raise ValueError(f"Unsupported fluid HROM interface trust mode {mode!r}.")
    if alpha < float(min_correction_alpha):
        return _FluidHROMInterfaceTrustResult(
            accepted=False,
            corrected=False,
            alpha=float(alpha),
            update_abs=float(update_abs),
            update_rel=float(update_rel),
            step_ratio=float(step_ratio),
            reason="interface_correction_alpha_too_small",
            values=proposed.copy(),
        )

    corrected = current + float(alpha) * (proposed - current)
    return _FluidHROMInterfaceTrustResult(
        accepted=True,
        corrected=True,
        alpha=float(alpha),
        update_abs=float(update_abs),
        update_rel=float(update_rel),
        step_ratio=float(step_ratio),
        reason="clipped",
        values=np.asarray(corrected, dtype=float),
    )


def _relaxed_lookup(
    coords: np.ndarray,
    old_values: np.ndarray,
    new_values: np.ndarray,
    *,
    omega: float,
) -> CoordinateLookup:
    omega_value = float(np.clip(float(omega), 0.0, 1.0))
    values = (1.0 - omega_value) * np.asarray(old_values, dtype=float) + omega_value * np.asarray(new_values, dtype=float)
    return CoordinateLookup(np.asarray(coords, dtype=float), values, dim=2)


def _aitken_relaxation_factor(
    *,
    omega_prev: float,
    residual_prev: np.ndarray | None,
    residual_curr: np.ndarray,
    omega_min: float,
    omega_max: float,
) -> float:
    omega = float(np.clip(float(omega_prev), float(omega_min), float(omega_max)))
    if residual_prev is None:
        return omega
    r_prev = np.asarray(residual_prev, dtype=float).ravel()
    r_curr = np.asarray(residual_curr, dtype=float).ravel()
    delta = r_curr - r_prev
    denom = float(np.dot(delta, delta))
    if denom <= 1.0e-30 or not np.isfinite(denom):
        return omega
    omega_new = -omega * float(np.dot(r_prev, delta)) / denom
    if not np.isfinite(omega_new):
        return omega
    return float(np.clip(omega_new, float(omega_min), float(omega_max)))


def _iqnils_next_iterate(
    *,
    x_curr: np.ndarray,
    g_curr: np.ndarray,
    x_history: list[np.ndarray],
    g_history: list[np.ndarray],
    dr_old_mats: list[np.ndarray] | None = None,
    dg_old_mats: list[np.ndarray] | None = None,
    omega: float,
    horizon: int,
    regularization: float,
) -> np.ndarray:
    backend_mode = _env_str("PYCUTFEM_EX2_IQN_BACKEND", "python").strip().lower()
    if backend_mode == "cpp":
        return kratos_iqnils_next_iterate_cpp(
            x_curr=np.asarray(x_curr, dtype=float),
            g_curr=np.asarray(g_curr, dtype=float),
            x_history=[np.asarray(values, dtype=float) for values in list(x_history)],
            g_history=[np.asarray(values, dtype=float) for values in list(g_history)],
            dr_old_mats=[np.asarray(block, dtype=float) for block in list(dr_old_mats or [])],
            dg_old_mats=[np.asarray(block, dtype=float) for block in list(dg_old_mats or [])],
            alpha=float(omega),
            horizon=int(horizon),
            regularization=float(regularization),
        )

    x_curr_arr = np.asarray(x_curr, dtype=float)
    g_curr_arr = np.asarray(g_curr, dtype=float)
    x_curr_vec = x_curr_arr.reshape(-1)
    g_curr_vec = g_curr_arr.reshape(-1)
    r_curr = g_curr_vec - x_curr_vec
    alpha_value = float(np.clip(float(omega), 0.0, 1.0))
    picard = (x_curr_vec + alpha_value * r_curr).reshape(x_curr_arr.shape)

    # Mirror Kratos IQN-ILS exactly:
    # - residual buffer R stores newest-first residuals
    # - prediction buffer X stores newest-first x + r = g
    # - V/W are residual/prediction differences in newest-first order
    # Kratos stores at most ``iteration_horizon`` residual/prediction snapshots
    # in the current time-step buffers. Since V/W are built from pairwise
    # differences of those buffers, the resulting column count is ``len(R)-1``.
    # Mirror that exact truncation here instead of keeping ``horizon+1`` states.
    keep_count = min(max(int(horizon), 1), len(x_history), len(g_history))
    if keep_count <= 0:
        return picard
    x_seq = [np.asarray(values, dtype=float).reshape(-1) for values in x_history[-keep_count:]]
    g_seq = [np.asarray(values, dtype=float).reshape(-1) for values in g_history[-keep_count:]]
    r_seq = [g_vec - x_vec for x_vec, g_vec in zip(x_seq, g_seq)]
    r_recent = list(reversed(r_seq))
    g_recent = list(reversed(g_seq))
    k = len(r_recent) - 1

    v_old_blocks = [np.asarray(block, dtype=float) for block in (dr_old_mats or []) if np.asarray(block).size]
    w_old_blocks = [np.asarray(block, dtype=float) for block in (dg_old_mats or []) if np.asarray(block).size]
    has_old = bool(v_old_blocks and w_old_blocks)

    if (not has_old) and k == 0:
        return picard

    if k > 0:
        v_new = np.column_stack(r_recent[:-1]) - np.column_stack(r_recent[1:])
        w_new = np.column_stack(g_recent[:-1]) - np.column_stack(g_recent[1:])
    else:
        v_new = np.empty((r_curr.size, 0), dtype=float)
        w_new = np.empty((g_curr_vec.size, 0), dtype=float)

    if has_old:
        v_old = np.hstack(v_old_blocks)
        w_old = np.hstack(w_old_blocks)
        v = np.hstack((v_new, v_old)) if k > 0 else v_old
        w = np.hstack((w_new, w_old)) if k > 0 else w_old
    else:
        v = v_new
        w = w_new

    if v.size == 0 or w.size == 0:
        return picard

    delta_r = -r_recent[0]
    reg = max(float(regularization), 0.0)
    try:
        if reg > 0.0:
            n_cols = v.shape[1]
            v_aug = np.vstack([v, np.sqrt(reg) * np.eye(n_cols, dtype=float)])
            rhs_aug = np.concatenate([delta_r, np.zeros(n_cols, dtype=float)])
            gamma = np.linalg.lstsq(v_aug, rhs_aug, rcond=None)[0]
        else:
            gamma = np.linalg.lstsq(v, delta_r, rcond=None)[0]
    except np.linalg.LinAlgError:
        return picard

    delta_x = w @ gamma - delta_r
    if not np.all(np.isfinite(delta_x)):
        return picard

    return (x_curr_vec + delta_x).reshape(x_curr_arr.shape)


def _iqnils_iteration_matrices(
    *,
    x_history: list[np.ndarray],
    g_history: list[np.ndarray],
    iteration_horizon: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    backend_mode = _env_str("PYCUTFEM_EX2_IQN_BACKEND", "python").strip().lower()
    if backend_mode == "cpp":
        return kratos_iqnils_iteration_matrices_cpp(
            x_history=[np.asarray(values, dtype=float) for values in list(x_history)],
            g_history=[np.asarray(values, dtype=float) for values in list(g_history)],
            iteration_horizon=int(iteration_horizon),
        )

    keep_count = min(max(int(iteration_horizon), 1), len(x_history), len(g_history))
    if keep_count <= 1:
        return None, None

    x_seq = [np.asarray(values, dtype=float).reshape(-1) for values in x_history[-keep_count:]]
    g_seq = [np.asarray(values, dtype=float).reshape(-1) for values in g_history[-keep_count:]]
    r_seq = [g_vec - x_vec for x_vec, g_vec in zip(x_seq, g_seq)]
    r_recent = list(reversed(r_seq))
    g_recent = list(reversed(g_seq))
    k = len(r_recent) - 1
    if k <= 0:
        return None, None

    v_new = np.column_stack(r_recent[:-1]) - np.column_stack(r_recent[1:])
    w_new = np.column_stack(g_recent[:-1]) - np.column_stack(g_recent[1:])
    return v_new, w_new


def _advance_coupling_load_guess(
    *,
    step_converged: bool,
    active_force_update: str,
    iface_coords: np.ndarray | None = None,
    solid_iface_coords: np.ndarray | None = None,
    load_guess_vals: np.ndarray,
    returned_load_vals: np.ndarray,
    load_guess_history: list[np.ndarray],
    load_return_history: list[np.ndarray],
    iqn_old_dr_mats: list[np.ndarray] | None,
    iqn_old_dg_mats: list[np.ndarray] | None,
    omega_force: float,
    force_iteration_horizon: int,
    force_regularization: float,
    include_debug: bool = False,
) -> tuple[CoordinateLookup, bool] | tuple[CoordinateLookup, bool, dict[str, object]]:
    if iface_coords is None:
        iface_coords = solid_iface_coords
    if iface_coords is None:
        raise TypeError("_advance_coupling_load_guess() requires 'iface_coords'.")

    def _result(
        lookup: CoordinateLookup,
        update_applied: bool,
        debug: dict[str, object],
    ) -> tuple[CoordinateLookup, bool] | tuple[CoordinateLookup, bool, dict[str, object]]:
        if bool(include_debug):
            return lookup, bool(update_applied), debug
        return lookup, bool(update_applied)

    iface_coords_arr = np.asarray(iface_coords, dtype=float)
    load_guess_arr = np.asarray(load_guess_vals, dtype=float).copy()
    returned_arr = np.asarray(returned_load_vals, dtype=float).copy()
    returned_lookup = CoordinateLookup(iface_coords_arr, returned_arr, dim=2)
    if bool(step_converged):
        return _result(returned_lookup, False, {
            "backend": str(_env_str("PYCUTFEM_EX2_IQN_BACKEND", "python")).strip().lower(),
            "used_history": False,
            "v_new": None,
            "w_new": None,
        })

    load_guess_history.append(load_guess_arr.copy())
    load_return_history.append(returned_arr.copy())
    active_backend = str(_env_str("PYCUTFEM_EX2_IQN_BACKEND", "python")).strip().lower()
    if str(active_force_update).lower() == "iqnils":
        v_new, w_new = _iqnils_iteration_matrices(
            x_history=load_guess_history,
            g_history=load_return_history,
            iteration_horizon=int(force_iteration_horizon),
        )
        next_load_values = _iqnils_next_iterate(
            x_curr=load_guess_arr,
            g_curr=returned_arr,
            x_history=load_guess_history,
            g_history=load_return_history,
            dr_old_mats=list(iqn_old_dr_mats or []),
            dg_old_mats=list(iqn_old_dg_mats or []),
            omega=float(omega_force),
            horizon=int(force_iteration_horizon),
            regularization=float(force_regularization),
        )
        return _result(CoordinateLookup(iface_coords_arr, next_load_values, dim=2), True, {
            "backend": active_backend,
            "used_history": bool(
                (v_new is not None and int(v_new.size) > 0)
                or any(np.asarray(block).size for block in (iqn_old_dr_mats or []))
            ),
            "v_new": None if v_new is None else np.asarray(v_new, dtype=float),
            "w_new": None if w_new is None else np.asarray(w_new, dtype=float),
        })

    return _result(
        _relaxed_lookup(
            iface_coords_arr,
            load_guess_arr,
            returned_arr,
            omega=float(omega_force),
        ),
        True,
        {
            "backend": "python",
            "used_history": False,
            "v_new": None,
            "w_new": None,
        },
    )


@dataclass(frozen=True)
class _SampledLSPGHybridModel:
    basis: np.ndarray
    free_dofs: np.ndarray
    sample_row_dofs: np.ndarray
    sample_element_ids: np.ndarray
    sample_weights: np.ndarray
    sample_element_weights: np.ndarray
    objective: str
    max_iterations: int
    residual_tol: float
    line_search: bool
    lspg_block_scale: bool
    lspg_block_scale_relative_floor: float
    recommended_switch_iter: int
    source_path: Path

    @property
    def n_modes(self) -> int:
        return int(self.basis.shape[1])


def _npz_scalar(data: object, key: str, default: object) -> object:
    try:
        value = data[key]  # type: ignore[index]
    except KeyError:
        return default
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return value


def _load_sampled_lspg_hybrid_model(path: Path, *, total_dofs: int, n_elements: int) -> _SampledLSPGHybridModel:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Fluid HROM model not found: {source}")
    with np.load(source, allow_pickle=False) as data:
        schema_version = int(_npz_scalar(data, "schema_version", 0))
        if schema_version != 1:
            raise RuntimeError(f"Unsupported fluid HROM schema_version={schema_version}; expected 1.")
        basis = np.asarray(data["basis"], dtype=float)
        free_dofs = np.asarray(data["free_dofs"], dtype=int).reshape(-1)
        sample_row_dofs = np.asarray(data["sample_row_dofs"], dtype=int).reshape(-1)
        sample_element_ids = np.asarray(data["sample_element_ids"], dtype=int).reshape(-1)
        sample_weights = np.asarray(data["sample_weights"], dtype=float).reshape(-1)
        if "sample_element_weights" in data.files:
            sample_element_weights = np.asarray(data["sample_element_weights"], dtype=float).reshape(-1)
        else:
            sample_element_weights = np.ones(int(sample_element_ids.size), dtype=float)
        if basis.ndim != 2 or int(basis.shape[0]) != int(total_dofs):
            raise RuntimeError(
                f"Fluid HROM basis shape {tuple(basis.shape)} is incompatible with total_dofs={int(total_dofs)}."
            )
        if np.any(free_dofs < 0) or np.any(free_dofs >= int(total_dofs)):
            raise RuntimeError("Fluid HROM free_dofs contain out-of-range entries.")
        if np.any(sample_row_dofs < 0) or np.any(sample_row_dofs >= int(total_dofs)):
            raise RuntimeError("Fluid HROM sample rows contain out-of-range entries.")
        if np.any(sample_element_ids < 0) or np.any(sample_element_ids >= int(n_elements)):
            raise RuntimeError("Fluid HROM sample elements contain out-of-range entries.")
        if int(sample_weights.size) != int(sample_row_dofs.size):
            raise RuntimeError("Fluid HROM sample_weights size does not match sample_row_dofs.")
        if int(sample_element_weights.size) != int(sample_element_ids.size):
            raise RuntimeError("Fluid HROM sample_element_weights size does not match sample_element_ids.")
        if np.any(sample_element_weights < 0.0) or not np.all(np.isfinite(sample_element_weights)):
            raise RuntimeError("Fluid HROM sample_element_weights must be finite and nonnegative.")
        objective = str(_npz_scalar(data, "objective", "sampled_lspg"))
        if objective.strip().lower().replace("-", "_") != "sampled_lspg":
            raise RuntimeError(
                "The production hybrid branch currently accepts only sampled_lspg HROM models; "
                f"got objective={objective!r}."
            )
        return _SampledLSPGHybridModel(
            basis=basis,
            free_dofs=free_dofs,
            sample_row_dofs=sample_row_dofs,
            sample_element_ids=sample_element_ids,
            sample_weights=sample_weights,
            sample_element_weights=sample_element_weights,
            objective=objective,
            max_iterations=int(_npz_scalar(data, "max_iterations", 8)),
            residual_tol=float(_npz_scalar(data, "residual_tol", 1.0e-8)),
            line_search=bool(_npz_scalar(data, "line_search", False)),
            lspg_block_scale=bool(_npz_scalar(data, "lspg_block_scale", False)),
            lspg_block_scale_relative_floor=float(_npz_scalar(data, "lspg_block_scale_relative_floor", 0.0)),
            recommended_switch_iter=max(1, int(_npz_scalar(data, "recommended_switch_iter", 4))),
            source_path=source,
        )


def _pack_fluid_state_for_hrom(*, dh: DofHandler, u_k: VectorFunction, p_k: Function) -> np.ndarray:
    values = np.zeros(int(dh.total_dofs), dtype=float)
    for component in u_k.components:
        ids = np.asarray(dh.get_field_slice(component.field_name), dtype=int)
        values[ids] = np.asarray(component.get_nodal_values(ids), dtype=float)
    p_ids = np.asarray(dh.get_field_slice(p_k.field_name), dtype=int)
    values[p_ids] = np.asarray(p_k.get_nodal_values(p_ids), dtype=float)
    return values


def _write_fluid_state_for_hrom(*, dh: DofHandler, u_k: VectorFunction, p_k: Function, state: np.ndarray) -> None:
    values = np.asarray(state, dtype=float).reshape(-1)
    if int(values.size) != int(dh.total_dofs):
        raise ValueError(f"Fluid HROM state has size {values.size}, expected {int(dh.total_dofs)}.")
    for component in u_k.components:
        ids = np.asarray(dh.get_field_slice(component.field_name), dtype=int)
        component.set_nodal_values(ids, values[ids])
    p_ids = np.asarray(dh.get_field_slice(p_k.field_name), dtype=int)
    p_k.set_nodal_values(p_ids, values[p_ids])


def _sampled_lspg_block_weights(
    *,
    dh: DofHandler,
    row_dofs: np.ndarray,
    residual: np.ndarray,
    floor: float = 1.0e-12,
    relative_floor: float = 0.0,
) -> np.ndarray:
    rows = np.asarray(row_dofs, dtype=int).reshape(-1)
    res = np.asarray(residual, dtype=float).reshape(-1)
    if int(rows.size) != int(res.size):
        raise ValueError("Block scaling residual size must match row_dofs size.")
    weights = np.ones(int(rows.size), dtype=float)
    velocity_rows = np.concatenate(
        [
            np.asarray(dh.get_field_slice("ux"), dtype=int).reshape(-1),
            np.asarray(dh.get_field_slice("uy"), dtype=int).reshape(-1),
        ]
    )
    pressure_rows = np.asarray(dh.get_field_slice("p"), dtype=int).reshape(-1)
    block_data: list[tuple[np.ndarray, float]] = []
    for block in (velocity_rows, pressure_rows):
        mask = np.isin(rows, block)
        if not np.any(mask):
            continue
        rms = float(np.linalg.norm(res[mask]) / np.sqrt(max(int(np.count_nonzero(mask)), 1)))
        block_data.append((mask, rms))
    reference_rms = max((rms for _mask, rms in block_data), default=0.0)
    effective_floor = max(float(floor), max(float(relative_floor), 0.0) * float(reference_rms))
    for mask, rms in block_data:
        weights[mask] = 1.0 / max(float(rms), float(effective_floor)) ** 2
    return weights


def _solve_sampled_lspg_hybrid_fluid_stage(
    *,
    model: _SampledLSPGHybridModel,
    prob: dict[str, object],
    mesh: Mesh,
    rho_f: float,
    mu_f: float,
    dt: float,
    bossak_alpha: float,
    dynamic_tau: float,
    quad_order: int,
    backend: str,
    fluid_prev_step: Sequence[np.ndarray],
    fluid_a_prev_stage: np.ndarray,
    max_iterations: int,
    residual_tol: float,
    line_search: bool,
) -> dict[str, object]:
    dh = prob["dh"]
    u_k = prob["u_k"]
    p_k = prob["p_k"]
    basis = np.asarray(model.basis, dtype=float)
    rows = np.asarray(model.sample_row_dofs, dtype=int).reshape(-1)
    element_ids = np.asarray(model.sample_element_ids, dtype=int).reshape(-1)
    sample_weights = np.maximum(np.asarray(model.sample_weights, dtype=float).reshape(-1), 0.0)
    element_weights = np.maximum(np.asarray(model.sample_element_weights, dtype=float).reshape(-1), 0.0)
    offset = _pack_fluid_state_for_hrom(dh=dh, u_k=u_k, p_k=p_k)
    coeffs = np.zeros(int(basis.shape[1]), dtype=float)
    bossak = _bossak_coefficients(alpha=float(bossak_alpha), dt=max(float(dt), 1.0e-14))
    prev_u = np.asarray(fluid_prev_step[0], dtype=float)
    prev_a = np.asarray(fluid_a_prev_stage, dtype=float)
    seed_a = np.asarray(prob["a_k"].nodal_values, dtype=float).copy()
    row_weights: np.ndarray | None = None
    last_norm = float("inf")
    first_assembly = True
    trajectory: list[dict[str, float]] = []

    def write_coefficients(current: np.ndarray, *, preserve_seed: bool) -> None:
        nonlocal first_assembly
        state = offset + basis @ np.asarray(current, dtype=float).reshape(-1)
        _write_fluid_state_for_hrom(dh=dh, u_k=u_k, p_k=p_k, state=state)
        if bool(first_assembly) and bool(preserve_seed) and float(np.linalg.norm(current)) <= 1.0e-14:
            prob["a_k"].nodal_values[:] = seed_a
        else:
            prob["a_k"].nodal_values[:] = float(bossak["ma0"]) * (
                np.asarray(u_k.nodal_values, dtype=float) - prev_u
            ) + float(bossak["ma2"]) * prev_a
        first_assembly = False

    def assemble(current: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        nonlocal row_weights
        write_coefficients(current, preserve_seed=True)
        _update_fluid_dvms_predicted_subscale(
            state=prob["dvms_state"],
            dh=dh,
            mesh=mesh,
            u_k=u_k,
            u_prev=prob["u_prev"],
            a_prev=prob["a_prev"],
            a_curr=prob.get("a_k"),
            p_k=p_k,
            d_mesh=prob["d_mesh"],
            d_prev=prob["d_prev"],
            d_prev2=prob.get("d_prev2"),
            mesh_v=prob.get("w_mesh_k"),
            mesh_v_prev=prob.get("w_mesh_prev"),
            mesh_a_prev=prob.get("a_mesh_prev"),
            rho_f=float(rho_f),
            mu_f=float(mu_f),
            dt=float(dt),
            bossak_alpha=float(bossak_alpha),
            dynamic_tau=float(dynamic_tau),
            backend=str(backend),
            use_oss=False,
            element_ids=element_ids,
        )
        sampled_residual, sampled_trial = _assemble_fluid_sampled_lspg_rows_raw(
            prob=prob,
            rho_f=float(rho_f),
            mu_f=float(mu_f),
            dt=float(dt),
            quad_order=int(quad_order),
            bossak_alpha=float(bossak_alpha),
            contribution_mode="system",
            backend=str(backend),
            element_ids=element_ids,
            row_dofs=rows,
            basis=basis,
            element_weights=element_weights,
        )
        if row_weights is None:
            if bool(model.lspg_block_scale):
                row_weights = _sampled_lspg_block_weights(
                    dh=dh,
                    row_dofs=rows,
                    residual=sampled_residual,
                    relative_floor=float(model.lspg_block_scale_relative_floor),
                )
            else:
                row_weights = np.ones(int(rows.size), dtype=float)
        combined_weights = sample_weights * np.maximum(np.asarray(row_weights, dtype=float).reshape(-1), 0.0)
        scale = np.sqrt(combined_weights)
        weighted_residual = sampled_residual * scale
        weighted_trial = sampled_trial * scale[:, None]
        return weighted_residual, weighted_trial, float(np.linalg.norm(weighted_residual))

    for iteration in range(1, max(1, int(max_iterations)) + 1):
        residual, trial, norm = assemble(coeffs)
        last_norm = float(norm)
        if not np.isfinite(last_norm):
            raise RuntimeError("Sampled LSPG HROM produced a non-finite residual norm.")
        if last_norm <= float(residual_tol):
            trajectory.append({"iteration": float(iteration), "residual_norm": float(last_norm), "step_norm": 0.0})
            break
        step, *_ = np.linalg.lstsq(trial, -residual, rcond=None)
        step = np.asarray(step, dtype=float).reshape(-1)
        if not np.all(np.isfinite(step)):
            raise RuntimeError("Sampled LSPG HROM produced a non-finite reduced step.")
        step_norm = float(np.linalg.norm(step))
        if bool(line_search):
            best_coeffs = coeffs + step
            best_norm = float("inf")
            for search_iter in range(6):
                alpha = 0.5**search_iter
                trial_coeffs = coeffs + float(alpha) * step
                first_before = bool(first_assembly)
                residual_trial, _trial_matrix, trial_norm = assemble(trial_coeffs)
                del residual_trial, _trial_matrix
                if trial_norm < best_norm:
                    best_norm = float(trial_norm)
                    best_coeffs = trial_coeffs
                if trial_norm <= (1.0 - 1.0e-4 * float(alpha)) * last_norm:
                    break
                first_assembly = first_before
            coeffs = np.asarray(best_coeffs, dtype=float).reshape(-1)
            last_norm = float(best_norm)
        else:
            coeffs = coeffs + step
        _clear_fluid_dvms_oss_projections(prob.get("dvms_state"))
        trajectory.append(
            {"iteration": float(iteration), "residual_norm": float(last_norm), "step_norm": float(step_norm)}
        )
        if step_norm <= 1.0e-12 * max(1.0, float(np.linalg.norm(coeffs))):
            break

    residual, _trial, last_norm = assemble(coeffs)
    del residual, _trial
    write_coefficients(coeffs, preserve_seed=False)
    _update_fluid_dvms_predicted_subscale(
        state=prob["dvms_state"],
        dh=dh,
        mesh=mesh,
        u_k=u_k,
        u_prev=prob["u_prev"],
        a_prev=prob["a_prev"],
        a_curr=prob.get("a_k"),
        p_k=p_k,
        d_mesh=prob["d_mesh"],
        d_prev=prob["d_prev"],
        d_prev2=prob.get("d_prev2"),
        mesh_v=prob.get("w_mesh_k"),
        mesh_v_prev=prob.get("w_mesh_prev"),
        mesh_a_prev=prob.get("a_mesh_prev"),
        rho_f=float(rho_f),
        mu_f=float(mu_f),
        dt=float(dt),
        bossak_alpha=float(bossak_alpha),
        dynamic_tau=float(dynamic_tau),
        backend=str(backend),
        use_oss=False,
    )
    if not np.all(np.isfinite(np.asarray(u_k.nodal_values, dtype=float))) or not np.all(
        np.isfinite(np.asarray(p_k.nodal_values, dtype=float))
    ):
        raise RuntimeError("Sampled LSPG HROM produced a non-finite fluid state.")
    return {
        "coefficients": np.asarray(coeffs, dtype=float).reshape(-1),
        "iterations": int(len(trajectory)),
        "estimated_residual_norm": float(last_norm),
        "trajectory": trajectory,
        "sample_rows": int(rows.size),
        "sample_elements": int(element_ids.size),
        "sample_element_weight_sum": float(np.sum(element_weights)),
    }


@dataclass(frozen=True)
class _CouplingRetryPolicy:
    force_update: str
    force_relaxation: float
    max_coupling_iters: int
    reset_interface_history: bool = False
    fluid_max_newton_iter: int | None = None
    fluid_globalization: str | None = None
    fluid_line_search: bool | None = None
    fluid_ls_fail_hard: bool | None = None
    fluid_continuation_scales: tuple[float, ...] = ()


def _build_coupling_retry_policies(
    *,
    force_update: str,
    force_relaxation: float,
    force_relaxation_min: float,
    force_relaxation_max: float,
    base_max_coupling_iters: int,
    base_max_newton_iter: int,
    max_retries: int,
) -> list[_CouplingRetryPolicy]:
    policies: list[_CouplingRetryPolicy] = []

    def _append(
        mode: str,
        omega: float,
        *,
        max_iters: int,
        reset: bool,
        fluid_max_newton_iter: int | None = None,
        fluid_globalization: str | None = None,
        fluid_line_search: bool | None = None,
        fluid_ls_fail_hard: bool | None = None,
        fluid_continuation_scales: tuple[float, ...] = (),
    ) -> None:
        omega_clamped = float(np.clip(float(omega), float(force_relaxation_min), float(force_relaxation_max)))
        policy = _CouplingRetryPolicy(
            str(mode),
            omega_clamped,
            max(int(max_iters), 1),
            bool(reset),
            None if fluid_max_newton_iter is None else max(int(fluid_max_newton_iter), 1),
            None if fluid_globalization is None else str(fluid_globalization),
            None if fluid_line_search is None else bool(fluid_line_search),
            None if fluid_ls_fail_hard is None else bool(fluid_ls_fail_hard),
            tuple(float(v) for v in fluid_continuation_scales),
        )
        if policy not in policies:
            policies.append(policy)

    base_iters = max(int(base_max_coupling_iters), 1)
    base_newton = max(int(base_max_newton_iter), 1)
    _append(str(force_update).lower(), float(force_relaxation), max_iters=base_iters, reset=False)
    fallback_specs = [
        {
            "omega": 0.05,
            "max_iters": max(base_iters, 8 * base_iters),
            "fluid_max_newton_iter": max(base_newton, 32),
            "fluid_globalization": "line_search_then_trust",
            "fluid_line_search": True,
            "fluid_ls_fail_hard": False,
        },
        {
            "omega": 0.02,
            "max_iters": max(base_iters, 12 * base_iters),
            "fluid_max_newton_iter": max(base_newton, 50),
            "fluid_globalization": "line_search_then_trust",
            "fluid_line_search": True,
            "fluid_ls_fail_hard": False,
        },
        {
            "omega": 0.01,
            "max_iters": max(base_iters, 16 * base_iters),
            "fluid_max_newton_iter": max(base_newton, 64),
            "fluid_globalization": "line_search_then_trust",
            "fluid_line_search": True,
            "fluid_ls_fail_hard": False,
        },
        {
            "omega": 0.005,
            "max_iters": max(base_iters, 20 * base_iters),
            "fluid_max_newton_iter": max(base_newton, 80),
            "fluid_globalization": "line_search_then_trust",
            "fluid_line_search": True,
            "fluid_ls_fail_hard": False,
        },
    ]
    for spec in fallback_specs:
        _append("constant", reset=True, **spec)
        if len(policies) >= int(max_retries) + 1:
            break
    return policies[: max(int(max_retries), 0) + 1]


def _copy_lookup(lookup: CoordinateLookup) -> CoordinateLookup:
    return CoordinateLookup(
        np.asarray(lookup.coords, dtype=float).copy(),
        np.asarray(lookup.values, dtype=float).copy(),
        dim=int(lookup.dim),
    )


def _step_progress_marker(
    *,
    disp_snapshots: list[np.ndarray],
    load_snapshots: list[np.ndarray],
    load_guess_snapshots: list[np.ndarray],
    load_return_snapshots: list[np.ndarray],
    fluid_load_guess_snapshots: list[np.ndarray],
    fluid_load_return_snapshots: list[np.ndarray],
    interface_disp_snapshots: list[np.ndarray],
    interface_velocity_snapshots: list[np.ndarray],
    interface_traction_snapshots: list[np.ndarray],
    reaction_load_snapshots: list[np.ndarray],
    stress_load_snapshots: list[np.ndarray],
    snapshot_rows: list[dict[str, object]],
    step_rows: list[dict[str, object]],
    fluid_times: list[float],
    structure_times: list[float],
    increment_times: list[float],
) -> dict[str, int]:
    return {
        "disp_snapshots": len(disp_snapshots),
        "load_snapshots": len(load_snapshots),
        "load_guess_snapshots": len(load_guess_snapshots),
        "load_return_snapshots": len(load_return_snapshots),
        "fluid_load_guess_snapshots": len(fluid_load_guess_snapshots),
        "fluid_load_return_snapshots": len(fluid_load_return_snapshots),
        "interface_disp_snapshots": len(interface_disp_snapshots),
        "interface_velocity_snapshots": len(interface_velocity_snapshots),
        "interface_traction_snapshots": len(interface_traction_snapshots),
        "reaction_load_snapshots": len(reaction_load_snapshots),
        "stress_load_snapshots": len(stress_load_snapshots),
        "snapshot_rows": len(snapshot_rows),
        "step_rows": len(step_rows),
        "fluid_times": len(fluid_times),
        "structure_times": len(structure_times),
        "increment_times": len(increment_times),
    }


def _truncate_step_progress(
    *,
    marker: dict[str, int],
    disp_snapshots: list[np.ndarray],
    load_snapshots: list[np.ndarray],
    load_guess_snapshots: list[np.ndarray],
    load_return_snapshots: list[np.ndarray],
    fluid_load_guess_snapshots: list[np.ndarray],
    fluid_load_return_snapshots: list[np.ndarray],
    interface_disp_snapshots: list[np.ndarray],
    interface_velocity_snapshots: list[np.ndarray],
    interface_traction_snapshots: list[np.ndarray],
    reaction_load_snapshots: list[np.ndarray],
    stress_load_snapshots: list[np.ndarray],
    snapshot_rows: list[dict[str, object]],
    step_rows: list[dict[str, object]],
    fluid_times: list[float],
    structure_times: list[float],
    increment_times: list[float],
) -> None:
    del disp_snapshots[int(marker["disp_snapshots"]):]
    del load_snapshots[int(marker["load_snapshots"]):]
    del load_guess_snapshots[int(marker["load_guess_snapshots"]):]
    del load_return_snapshots[int(marker["load_return_snapshots"]):]
    del fluid_load_guess_snapshots[int(marker["fluid_load_guess_snapshots"]):]
    del fluid_load_return_snapshots[int(marker["fluid_load_return_snapshots"]):]
    del interface_disp_snapshots[int(marker["interface_disp_snapshots"]):]
    del interface_velocity_snapshots[int(marker["interface_velocity_snapshots"]):]
    del interface_traction_snapshots[int(marker["interface_traction_snapshots"]):]
    del reaction_load_snapshots[int(marker["reaction_load_snapshots"]):]
    del stress_load_snapshots[int(marker["stress_load_snapshots"]):]
    del snapshot_rows[int(marker["snapshot_rows"]):]
    del step_rows[int(marker["step_rows"]):]
    del fluid_times[int(marker["fluid_times"]):]
    del structure_times[int(marker["structure_times"]):]
    del increment_times[int(marker["increment_times"]):]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return str(default)
    value = str(raw).strip()
    return value if value else str(default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local DoubleFlap Example 2 with a strong staggered fixed-point FSI loop.")
    parser.add_argument("--reference-root", type=Path, default=None, help="Downloaded DoubleFlap reference directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("examples/NIRB/artifacts/example2_local_fom"))
    parser.add_argument("--mesh-source", choices=("reference", "conforming"), default="reference")
    parser.add_argument("--mesh-size", type=float, default=0.20)
    parser.add_argument("--mesh-order", type=int, default=1)
    parser.add_argument("--poly-order", type=int, default=1)
    parser.add_argument("--pressure-order", type=int, default=None)
    parser.add_argument("--mesh-backend", choices=("python", "kratos"), default="python")
    parser.add_argument(
        "--quad-order",
        type=int,
        default=None,
        help=(
            "Volume quadrature order for the fluid DVMS state and assembly. "
            "Default: 1 (Kratos-matched 3-point triangle rule)."
        ),
    )
    parser.add_argument(
        "--solid-quad-order",
        type=int,
        default=None,
        help=(
            "Quadrature order for the hyperelastic solid solve. "
            "Default: 2, matching Kratos' Q1 2x2 Gauss structural rule."
        ),
    )
    parser.add_argument("--reynolds", type=float, default=301.0)
    parser.add_argument("--reference-velocity", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-coupling-iters", type=int, default=50)
    parser.add_argument("--coupling-rel-tol", type=float, default=5.0e-3)
    parser.add_argument("--coupling-abs-tol", type=float, default=5.0e-3)
    parser.add_argument("--load-transfer", choices=("stress", "reaction"), default="reaction")
    parser.add_argument("--force-update", choices=("constant", "aitken", "iqnils"), default="iqnils")
    parser.add_argument("--force-relaxation", type=float, default=0.5)
    parser.add_argument("--force-relaxation-min", type=float, default=1.0e-3)
    parser.add_argument("--force-relaxation-max", type=float, default=1.0)
    parser.add_argument("--force-iteration-horizon", type=int, default=50)
    parser.add_argument("--force-history", type=int, default=3)
    parser.add_argument("--force-regularization", type=float, default=0.0)
    parser.add_argument(
        "--step-retries",
        type=int,
        default=0,
        help="Optional rollback/retry ladder for failed outer FSI steps. Default: disabled on the production path.",
    )
    parser.add_argument("--newton-tol", type=float, default=1.0e-6)
    parser.add_argument("--max-newton-iter", type=int, default=20)
    parser.add_argument("--bossak-alpha", type=float, default=-0.3)
    parser.add_argument("--dynamic-tau", type=float, default=1.0)
    parser.add_argument("--pressure-gauge", type=float, default=1.0e-5)
    parser.add_argument(
        "--fluid-operator",
        choices=("exact", "continuous", "sampled_lspg_hybrid"),
        default="exact",
    )
    parser.add_argument(
        "--fluid-hrom-model-path",
        type=Path,
        default=None,
        help="Deployable .npz sampled-LSPG fluid HROM model when --fluid-operator=sampled_lspg_hybrid.",
    )
    parser.add_argument(
        "--fluid-hrom-switch-iter",
        type=int,
        default=None,
        help="First coupling iteration that may use the sampled-LSPG fluid HROM. Default: model recommendation.",
    )
    parser.add_argument(
        "--fluid-hrom-max-iterations",
        type=int,
        default=None,
        help="Override the sampled-LSPG reduced Gauss-Newton iteration limit stored in the HROM model.",
    )
    parser.add_argument(
        "--fluid-hrom-residual-tol",
        type=float,
        default=None,
        help="Override the sampled-LSPG residual tolerance stored in the HROM model.",
    )
    parser.add_argument(
        "--fluid-hrom-fallback-exact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fall back to the exact local fluid solve when the HROM gate or monitor fails.",
    )
    parser.add_argument(
        "--fluid-hrom-max-previous-load-rel",
        type=float,
        default=1.0e-2,
        help=(
            "Only allow sampled-LSPG HROM when the previous coupling load relative residual is at or below this value. "
            "This keeps the first HROM call in the asymptotic fixed-point regime."
        ),
    )
    parser.add_argument(
        "--fluid-hrom-min-previous-load-rel",
        type=float,
        default=0.0,
        help=(
            "Only allow sampled-LSPG HROM when the previous coupling load relative residual is at or above this value. "
            "Use this to keep the final tolerance-closing correction on the exact fluid operator."
        ),
    )
    parser.add_argument(
        "--fluid-hrom-max-previous-disp-rel",
        type=float,
        default=float("inf"),
        help="Optional matching gate on the previous coupling displacement relative residual.",
    )
    parser.add_argument(
        "--fluid-hrom-max-consecutive-stages",
        type=int,
        default=0,
        help=(
            "Maximum number of consecutive sampled-LSPG fluid stages inside one FSI step. "
            "Use 0 for unlimited."
        ),
    )
    parser.add_argument(
        "--fluid-hrom-require-exact-accept",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If a sampled-LSPG stage satisfies the coupling tolerance, take one exact-fluid correction before accepting "
            "the FSI step."
        ),
    )
    parser.add_argument(
        "--fluid-hrom-trial-exact-correct",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use sampled-LSPG only as a nonlinear predictor and immediately correct the stage with the exact local "
            "ALE-DVMS fluid operator. This prevents unvalidated HROM states from entering the FSI load/IQN history."
        ),
    )
    parser.add_argument(
        "--fluid-hrom-max-trial-load-rel-error",
        type=float,
        default=float("inf"),
        help=(
            "When --fluid-hrom-trial-exact-correct is enabled, disable future HROM trials if the HROM trial "
            "interface load differs from the exact corrected load by more than this relative value."
        ),
    )
    parser.add_argument(
        "--fluid-hrom-disable-steps-after-trial-failure",
        type=int,
        default=0,
        help=(
            "Number of global steps, including the current step, for which HROM trials are disabled after the "
            "trial load monitor fails. Use 0 to disable only the remaining coupling iterations of the current step."
        ),
    )
    parser.add_argument(
        "--fluid-hrom-interface-trust",
        choices=("none", "fallback", "clip"),
        default="none",
        help=(
            "Coupled-interface trust operator for committed sampled-LSPG stages. "
            "'fallback' rejects HROM loads outside the interface trust region; "
            "'clip' commits the HROM state but clips the returned interface-load increment."
        ),
    )
    parser.add_argument(
        "--fluid-hrom-interface-max-step-ratio",
        type=float,
        default=float("inf"),
        help=(
            "Maximum allowed ratio between the committed HROM interface-load increment and the previous "
            "fixed-point load increment. Used by --fluid-hrom-interface-trust."
        ),
    )
    parser.add_argument(
        "--fluid-hrom-interface-max-load-rel",
        type=float,
        default=float("inf"),
        help="Maximum relative interface-load increment for a committed HROM stage.",
    )
    parser.add_argument(
        "--fluid-hrom-interface-min-correction-alpha",
        type=float,
        default=0.0,
        help=(
            "When --fluid-hrom-interface-trust=clip, fall back to exact fluid if the required load clipping "
            "factor is below this value."
        ),
    )
    parser.add_argument(
        "--solid-operator",
        choices=("exact", "nirb", "porous"),
        default="exact",
        help="Use the full structural solve, a trained NIRB solid surrogate, or the U-Pl porous flap.",
    )
    parser.add_argument("--porous-pressure-order", type=int, default=None)
    parser.add_argument("--porous-porosity", type=float, default=0.32)
    parser.add_argument("--porous-biot-coefficient", type=float, default=0.8)
    parser.add_argument("--porous-permeability", type=float, default=1.0e-8)
    parser.add_argument("--porous-storage-inverse", type=float, default=2.0e-4)
    parser.add_argument(
        "--nirb-model-path",
        type=Path,
        default=None,
        help="Path to a trained pycutfem NIRB model when --solid-operator=nirb.",
    )
    parser.add_argument(
        "--nirb-start-step",
        type=int,
        default=1,
        help="First accepted global step that uses the NIRB solid surrogate.",
    )
    parser.add_argument("--backend", choices=("python", "jit", "cpp"), default="cpp")
    parser.add_argument("--linear-backend", choices=("scipy", "sparseqr", "petsc", "amgcl"), default="scipy")
    parser.add_argument("--snapshot-mode", choices=("all", "converged"), default="all")
    parser.add_argument("--save-vtk", action="store_true", help="Write fluid/solid VTK outputs under output_dir/vtk_data.")
    parser.add_argument("--vtk-every", type=int, default=1, help="Write VTK every N accepted steps when --save-vtk is enabled.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Write a restart checkpoint every N accepted global steps (0 disables checkpoints).")
    parser.add_argument("--step-history-every", type=int, default=1, help="Write accepted-step full-state npz files every N steps (0 disables).")
    parser.add_argument("--restart-from", type=Path, default=None, help="Resume from a checkpoint .npz file or a checkpoints/ directory.")
    parser.add_argument(
        "--monitor-interface-loads",
        action="store_true",
        help="Export per-iteration interface load diagnostics, including guessed load and both stress/reaction returns.",
    )
    parser.add_argument("--reuse-mesh", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_local_example2(
    *,
    output_dir: Path,
    reference_root: Path | None = None,
    mesh_source: str = "reference",
    mesh_size: float = 0.20,
    mesh_order: int = 1,
    poly_order: int = 1,
    pressure_order: int | None = None,
    mesh_backend: str = "python",
    quadrature_order: int | None = None,
    solid_quadrature_order: int | None = None,
    reynolds: float = 250.0,
    reference_velocity: float | None = None,
    dt: float | None = None,
    end_time: float | None = None,
    max_steps: int | None = None,
    max_coupling_iters: int = 50,
    coupling_rel_tol: float = 5.0e-3,
    coupling_abs_tol: float = 5.0e-3,
    load_transfer: str = "reaction",
    force_update: str = "iqnils",
    force_relaxation: float = 0.5,
    force_relaxation_min: float = 1.0e-3,
    force_relaxation_max: float = 1.0,
    force_iteration_horizon: int = 50,
    force_history: int = 3,
    force_regularization: float = 0.0,
    step_retries: int = 0,
    newton_tol: float = 1.0e-6,
    max_newton_iter: int = 12,
    bossak_alpha: float = -0.3,
    dynamic_tau: float = 1.0,
    pressure_gauge: float = 1.0e-5,
    fluid_operator: str = "exact",
    fluid_hrom_model_path: Path | None = None,
    fluid_hrom_switch_iter: int | None = None,
    fluid_hrom_max_iterations: int | None = None,
    fluid_hrom_residual_tol: float | None = None,
    fluid_hrom_fallback_exact: bool = True,
    fluid_hrom_max_previous_load_rel: float = 1.0e-2,
    fluid_hrom_min_previous_load_rel: float = 0.0,
    fluid_hrom_max_previous_disp_rel: float = float("inf"),
    fluid_hrom_max_consecutive_stages: int = 0,
    fluid_hrom_require_exact_accept: bool = False,
    fluid_hrom_trial_exact_correct: bool = False,
    fluid_hrom_max_trial_load_rel_error: float = float("inf"),
    fluid_hrom_disable_steps_after_trial_failure: int = 0,
    fluid_hrom_interface_trust: str = "none",
    fluid_hrom_interface_max_step_ratio: float = float("inf"),
    fluid_hrom_interface_max_load_rel: float = float("inf"),
    fluid_hrom_interface_min_correction_alpha: float = 0.0,
    solid_operator: str = "exact",
    porous_pressure_order: int | None = None,
    porous_porosity: float = 0.32,
    porous_biot_coefficient: float = 0.8,
    porous_permeability: float = 1.0e-8,
    porous_storage_inverse: float = 2.0e-4,
    nirb_model_path: Path | None = None,
    nirb_start_step: int = 1,
    backend: str = "cpp",
    linear_backend: str = "scipy",
    snapshot_mode: str = "all",
    save_vtk: bool = False,
    vtk_every: int = 1,
    monitor_interface_loads: bool = False,
    checkpoint_every: int = 1,
    step_history_every: int = 1,
    restart_from: Path | None = None,
    reuse_mesh: bool = False,
    verbose: bool = False,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    co_sim_dir = output_dir / "coSimData"
    co_sim_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    step_history_dir = output_dir / "step_history"
    restart_payload: dict[str, np.ndarray] | None = None
    restart_path = Path(restart_from) if restart_from is not None else None
    if restart_path is not None:
        if restart_path.is_dir():
            latest = restart_path / "latest_checkpoint.txt"
            if latest.exists():
                restart_path = restart_path / latest.read_text(encoding="utf-8").strip()
        if not restart_path.exists():
            raise FileNotFoundError(f"Restart checkpoint not found: {restart_path}")
        restart_payload = _load_checkpoint_payload(restart_path)
        schema_version = int(np.asarray(restart_payload.get("checkpoint_schema_version", np.asarray(-1)), dtype=int).reshape(-1)[0])
        if schema_version != int(_CHECKPOINT_SCHEMA_VERSION):
            raise RuntimeError(
                "Checkpoint schema mismatch. "
                f"Expected schema_version={_CHECKPOINT_SCHEMA_VERSION}, got {schema_version}. "
                "This checkpoint was written by an older solver path and should not be reused with the current code. "
                "Generate a fresh checkpoint with the current driver or convert from Kratos/local step_history."
            )

    setup = load_example2_local_setup(
        reference_root=reference_root,
        mesh_size_default=mesh_size,
        mesh_order_default=mesh_order,
    )
    geometry = setup.geometry
    reference_velocity_value = (
        float(reference_velocity)
        if reference_velocity is not None
        else float(setup.material.max_velocity)
    )
    dt_value = float(setup.boundaries.time_step if dt is None else dt)
    end_time_value = float(setup.boundaries.end_time if end_time is None else end_time)
    max_steps_value = int(
        max_steps if max_steps is not None else max(1, math.ceil(end_time_value / max(dt_value, 1.0e-14)))
    )
    step_count = min(max_steps_value, int(max(1, math.ceil(end_time_value / max(dt_value, 1.0e-14)))))
    pressure_order_value = int(pressure_order if pressure_order is not None else max(1, poly_order - 1))
    quad_order = (
        int(quadrature_order)
        if quadrature_order is not None
        else int(_EX2L_KRATOS_MATCHED_QUAD_ORDER)
    )
    # The fluid uses the Kratos-matched Triangle2D3 rule (q=1 in pycutfem).
    # The structural reference mesh is Q1 quadrilaterals; Kratos' monitored
    # TotalLagrangianElement2D4N response matches with the standard 2x2 Gauss
    # rule (q=2 here), while higher tensor-product rules drift slightly.
    # The weighted structural-similarity mesh solve is a separate branch from
    # the fluid DVMS operator; the audited Kratos-matched local mesh operator
    # uses a higher quadrature than the fluid q=1 rule.
    solid_quad_order = (
        int(solid_quadrature_order)
        if solid_quadrature_order is not None
        else 2
    )
    mesh_path = output_dir / "double_flap_conforming.msh"
    mesh_source_value = str(mesh_source).lower()
    if mesh_source_value == "reference":
        _log(verbose, "[mesh] loading reference MDPA fluid/solid meshes")
        mesh_f, mesh_s = _load_reference_partitioned_meshes(setup=setup)
        mesh_descriptor = {
            "mesh_source": "reference",
            "fluid_mesh_path": str(setup.reference.fluid.path),
            "solid_mesh_path": str(setup.reference.solid.path),
        }
    elif mesh_source_value == "conforming":
        if (not mesh_path.exists()) or (not reuse_mesh):
            _log(verbose, f"[mesh] building {mesh_path} (h={mesh_size:.3f}, order={mesh_order})")
            build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=float(mesh_size), order=int(mesh_order))
        _log(verbose, "[mesh] loading fluid and solid submeshes")
        mesh_f = mesh_from_gmsh(mesh_path, surface_physical_names=["fluid"], apply_boundary_tags=True)
        mesh_s = mesh_from_gmsh(mesh_path, surface_physical_names=["solid"], apply_boundary_tags=True)
        mesh_descriptor = {
            "mesh_source": "conforming",
            "fluid_mesh_path": str(mesh_path),
            "solid_mesh_path": str(mesh_path),
        }
    else:
        raise ValueError(f"Unsupported mesh_source={mesh_source!r}")

    mesh_backend_value = str(mesh_backend).lower()
    if mesh_backend_value not in {"python", "kratos"}:
        raise ValueError(f"Unsupported mesh_backend={mesh_backend!r}")
    if mesh_backend_value == "kratos" and mesh_source_value != "reference":
        raise ValueError("mesh_backend='kratos' currently requires mesh_source='reference'.")
    if mesh_backend_value == "kratos" and restart_payload is not None:
        raise RuntimeError("mesh_backend='kratos' is not yet supported together with restart_from.")
    fluid_operator_mode = str(fluid_operator).strip().lower()
    fluid_operator_mode = fluid_operator_mode.replace("-", "_")
    if fluid_operator_mode in {"condensed", "local", "exact_local"}:
        fluid_operator_mode = "exact"
    if fluid_operator_mode in {"hrom", "sampled_lspg", "sampled_lspg_hrom"}:
        fluid_operator_mode = "sampled_lspg_hybrid"
    if fluid_operator_mode not in {"exact", "continuous", "sampled_lspg_hybrid"}:
        raise ValueError(f"Unsupported fluid_operator={fluid_operator!r}")
    if fluid_operator_mode != "exact":
        legacy_exact_flag = str(os.getenv("PYCUTFEM_EX2_ATTEMPT_LOCAL_CORRECTION", "0") or "0").strip().lower()
        if legacy_exact_flag in {"1", "true", "yes"}:
            fluid_operator_mode = "exact"
    use_sampled_lspg_hybrid_fluid = fluid_operator_mode == "sampled_lspg_hybrid"
    if bool(use_sampled_lspg_hybrid_fluid) and fluid_hrom_model_path is None:
        raise ValueError("--fluid-hrom-model-path is required when --fluid-operator=sampled_lspg_hybrid")
    use_exact_fluid_operator = fluid_operator_mode in {"exact", "sampled_lspg_hybrid"}
    solid_operator_mode = str(solid_operator).strip().lower()
    if solid_operator_mode not in {"exact", "nirb", "porous"}:
        raise ValueError(f"Unsupported solid_operator={solid_operator!r}")
    if solid_operator_mode == "nirb" and nirb_model_path is None:
        raise ValueError("--nirb-model-path is required when --solid-operator=nirb")
    porous_pressure_order_value = int(
        porous_pressure_order if porous_pressure_order is not None else max(1, int(poly_order) - 1)
    )
    nirb_start_step_value = max(1, int(nirb_start_step))
    exact_backend_notes: list[str] = []
    requested_exact_fluid_backend_mode = _env_str(
        "PYCUTFEM_EX2_EXACT_FLUID_BACKEND",
        "auto",
    ).strip().lower()
    if requested_exact_fluid_backend_mode in {"", "auto"}:
        exact_fluid_backend_mode = (
            "kratos_live"
            if bool(use_exact_fluid_operator) and mesh_source_value == "reference" and restart_payload is None
            else "local"
        )
    elif requested_exact_fluid_backend_mode in {"local", "pycutfem"}:
        exact_fluid_backend_mode = "local"
    elif requested_exact_fluid_backend_mode in {"kratos", "kratos_live", "live"}:
        exact_fluid_backend_mode = "kratos_live"
    else:
        raise ValueError(
            "Unsupported exact-fluid backend mode "
            f"{requested_exact_fluid_backend_mode!r}. "
            "Use one of: auto, local, kratos_live."
        )
    if bool(use_sampled_lspg_hybrid_fluid):
        if exact_fluid_backend_mode != "local":
            exact_backend_notes.append(
                "sampled_lspg_hybrid fluid forces exact-fluid fallback backend to local "
                "because the HROM uses the example-level local ALE-DVMS operator"
            )
        exact_fluid_backend_mode = "local"
    if requested_exact_fluid_backend_mode in {"", "auto"} and exact_fluid_backend_mode == "local":
        fluid_auto_reasons: list[str] = []
        if not bool(use_exact_fluid_operator):
            fluid_auto_reasons.append("fluid_operator != exact")
        if mesh_source_value != "reference":
            fluid_auto_reasons.append("mesh_source != reference")
        if restart_payload is not None:
            fluid_auto_reasons.append("restart_from is set")
        if fluid_auto_reasons:
            exact_backend_notes.append(
                "exact fluid backend auto -> local because " + ", ".join(fluid_auto_reasons)
            )
    if exact_fluid_backend_mode == "kratos_live" and mesh_source_value != "reference":
        raise RuntimeError("The persistent Kratos exact-fluid backend requires mesh_source='reference'.")
    if exact_fluid_backend_mode == "kratos_live" and restart_payload is not None:
        raise RuntimeError("The persistent Kratos exact-fluid backend is not yet supported with restart_from.")
    requested_exact_structure_backend_mode = _env_str(
        "PYCUTFEM_EX2_EXACT_STRUCTURE_BACKEND",
        "auto",
    ).strip().lower()
    if requested_exact_structure_backend_mode in {"", "auto"}:
        exact_structure_backend_mode = (
            "kratos_live"
            if (
                solid_operator_mode not in {"nirb", "porous"}
                and bool(use_exact_fluid_operator)
                and mesh_source_value == "reference"
                and restart_payload is None
            )
            else "local"
        )
    elif requested_exact_structure_backend_mode in {"local", "pycutfem"}:
        exact_structure_backend_mode = "local"
    elif requested_exact_structure_backend_mode in {"kratos", "kratos_live", "live"}:
        exact_structure_backend_mode = "kratos_live"
    else:
        raise ValueError(
            "Unsupported exact-structure backend mode "
            f"{requested_exact_structure_backend_mode!r}. "
            "Use one of: auto, local, kratos_live."
        )
    if requested_exact_structure_backend_mode in {"", "auto"} and exact_structure_backend_mode == "local":
        structure_auto_reasons: list[str] = []
        if solid_operator_mode == "nirb":
            structure_auto_reasons.append("solid_operator=nirb")
        if solid_operator_mode == "porous":
            structure_auto_reasons.append("solid_operator=porous")
        if not bool(use_exact_fluid_operator):
            structure_auto_reasons.append("fluid_operator != exact")
        if mesh_source_value != "reference":
            structure_auto_reasons.append("mesh_source != reference")
        if restart_payload is not None:
            structure_auto_reasons.append("restart_from is set")
        if structure_auto_reasons:
            exact_backend_notes.append(
                "exact structure backend auto -> local because " + ", ".join(structure_auto_reasons)
            )
    if exact_structure_backend_mode == "kratos_live" and mesh_source_value != "reference":
        raise RuntimeError("The persistent Kratos exact-structure backend requires mesh_source='reference'.")
    if exact_structure_backend_mode == "kratos_live" and restart_payload is not None:
        raise RuntimeError("The persistent Kratos exact-structure backend is not yet supported with restart_from.")
    if solid_operator_mode == "nirb" and exact_structure_backend_mode == "kratos_live":
        raise RuntimeError("solid_operator='nirb' cannot be combined with the kratos_live structure backend.")
    if solid_operator_mode == "porous" and exact_structure_backend_mode == "kratos_live":
        raise RuntimeError("solid_operator='porous' cannot be combined with the kratos_live structure backend.")
    requested_exact_fluid_linear_backend = str(
        os.getenv("PYCUTFEM_EX2_EXACT_FLUID_LINEAR_BACKEND", "") or ""
    ).strip().lower()
    allowed_linear_backends = {"scipy", "sparseqr", "petsc", "amgcl"}

    def _select_exact_fluid_linear_backend() -> str:
        selected = (
            requested_exact_fluid_linear_backend
            if requested_exact_fluid_linear_backend
            else (
                "amgcl"
                if bool(use_exact_fluid_operator) and exact_fluid_backend_mode == "local"
                else str(linear_backend)
            )
        )
        selected = str(selected).strip().lower()
        if selected not in allowed_linear_backends:
            raise ValueError(
                "Unsupported exact-fluid linear backend "
                f"{selected!r}. Use one of: {', '.join(sorted(allowed_linear_backends))}."
            )
        return selected

    exact_fluid_linear_backend = _select_exact_fluid_linear_backend()
    structure_solver_profile = _load_kratos_structure_solver_profile(
        benchmark_root=Path(setup.reference.root),
    )

    fluid = _build_fluid_problem(
        mesh_f,
        poly_order=int(poly_order),
        pressure_order=pressure_order_value,
        mesh_order=int(mesh_order),
        quadrature_order=quad_order,
    )
    mesh_ext = _build_mesh_extension_problem(mesh_f, poly_order=int(mesh_order))
    if solid_operator_mode == "porous":
        solid = _build_porous_solid_problem(
            mesh_s,
            displacement_order=int(poly_order),
            pressure_order=int(porous_pressure_order_value),
        )
    else:
        solid = _build_solid_problem(mesh_s, poly_order=int(poly_order))
    diagnostic_solid_system_backend = _diagnostic_solid_system_backend_mode()
    if diagnostic_solid_system_backend in {"", "none", "symbolic"}:
        kratos_local_solid_backend = None
    elif diagnostic_solid_system_backend == "kratos_local":
        kratos_local_solid_backend = _build_kratos_local_solid_backend(
            benchmark_root=Path(setup.reference.root),
            prob=solid,
        )
    else:
        raise ValueError(
            "Unsupported diagnostic solid-system backend mode "
            f"{diagnostic_solid_system_backend!r}."
        )
    requested_mesh_extension_solver = _env_str("PYCUTFEM_EX2_MESH_EXTENSION_SOLVER", "").strip().lower()
    requested_mesh_linear_solve_mode = _env_str("PYCUTFEM_EX2_MESH_LINEAR_SOLVE_MODE", "").strip().lower()
    need_kratos_local_mesh_backend = (
        mesh_source_value == "reference"
        and mesh_backend_value == "python"
        and requested_mesh_extension_solver in {"kratos_local", "kratos"}
    )
    need_local_amgcl_openmp_cap = requested_mesh_linear_solve_mode in {
        "amgcl",
        "pycutfem_amgcl",
        "cpp_amgcl",
        "local_amgcl",
    }
    need_kratos_runtime = (
        diagnostic_solid_system_backend == "kratos_local"
        or bool(need_kratos_local_mesh_backend)
        or exact_structure_backend_mode == "kratos_live"
        or exact_fluid_backend_mode == "kratos_live"
        or mesh_backend_value == "kratos"
    )
    kratos_thread_env = _configure_kratos_thread_env(enable=bool(need_kratos_runtime or need_local_amgcl_openmp_cap))
    if bool(kratos_thread_env.get("already_imported")) and bool(kratos_thread_env.get("changes")):
        exact_backend_notes.append(
            "Kratos thread defaults were applied after KratosMultiphysics was already imported; "
            "restart the process if you need the new thread cap to take effect"
        )
    kratos_local_mesh_backend = (
        _build_kratos_local_mesh_backend(
            fluid_mdpa_path=Path(mesh_descriptor["fluid_mesh_path"]),
            dt=float(dt_value),
            bossak_alpha=float(bossak_alpha),
            prob=mesh_ext,
        )
        if need_kratos_local_mesh_backend
        else None
    )
    kratos_exact_structure_backend = (
        _build_kratos_exact_structure_backend(
            benchmark_root=Path(setup.reference.root),
            dt=float(dt_value),
        )
        if exact_structure_backend_mode == "kratos_live"
        else None
    )
    kratos_exact_fluid_backend = None
    if exact_fluid_backend_mode == "kratos_live":
        try:
            kratos_exact_fluid_backend = _build_kratos_exact_fluid_backend(
                fluid_mdpa_path=Path(mesh_descriptor["fluid_mesh_path"]),
                dt=float(dt_value),
                bossak_alpha=float(bossak_alpha),
            )
        except ModuleNotFoundError:
            if requested_exact_fluid_backend_mode not in {"", "auto"}:
                raise
            exact_backend_notes.append(
                "exact fluid backend auto -> local because kratos_live imports are unavailable in the current environment"
            )
            exact_fluid_backend_mode = "local"
            exact_fluid_linear_backend = _select_exact_fluid_linear_backend()
    kratos_mesh_backend = (
        _build_kratos_mesh_motion_backend(
            fluid_mdpa_path=Path(mesh_descriptor["fluid_mesh_path"]),
            dt=float(dt_value),
            bossak_alpha=float(bossak_alpha),
        )
        if mesh_backend_value == "kratos" and kratos_exact_fluid_backend is None
        else None
    )
    _section(verbose, "[config] Example 2 backend resolution")
    _log(
        verbose,
        "[config] "
        f"mesh_source={mesh_source_value} mesh_backend={mesh_backend_value} "
        f"fluid_operator={fluid_operator_mode} restart_from={'none' if restart_path is None else restart_path}",
    )
    _log(
        verbose,
        "[config] "
        f"exact_fluid_backend requested={requested_exact_fluid_backend_mode or 'auto'} "
        f"resolved={exact_fluid_backend_mode} linear_backend={exact_fluid_linear_backend}",
    )
    _log(
        verbose,
        "[config] "
        f"exact_structure_backend requested={requested_exact_structure_backend_mode or 'auto'} "
        f"resolved={exact_structure_backend_mode}",
    )
    if bool(kratos_thread_env.get("enabled")):
        effective_env = dict(kratos_thread_env.get("effective", {}))
        _log(
            verbose,
            "[config] "
            "kratos_threads "
            + " ".join(f"{name}={value}" for name, value in effective_env.items()),
        )
        runtime_omp_max_threads = kratos_thread_env.get("runtime_openmp_max_threads")
        runtime_omp_library = str(kratos_thread_env.get("runtime_openmp_library", "") or "").strip()
        if runtime_omp_max_threads is not None or runtime_omp_library:
            _log(
                verbose,
                "[config] "
                f"kratos_openmp_runtime library={runtime_omp_library or 'unknown'} "
                f"max_threads={runtime_omp_max_threads if runtime_omp_max_threads is not None else 'unknown'}",
            )
    _log(
        verbose,
        "[config] "
        f"monitor_interface_loads={int(bool(monitor_interface_loads))} "
        f"monitor_stress_interface_loads={int(bool(_should_monitor_stress_interface_loads(load_transfer=load_transfer, monitor_interface_loads=monitor_interface_loads)))}",
    )
    for note in exact_backend_notes:
        _log(verbose, f"[config-warning] {note}")
    fluid["mesh"] = mesh_f
    solid["mesh"] = mesh_s
    nirb_solid_predictor: NIRBSolidPredictor | None = None
    if mesh_source_value == "reference":
        clamp_part = setup.reference.solid.submodelparts.get("DISPLACEMENT_BCDisp")
        old_to_new = getattr(mesh_s, "_mdpa_old_to_new_node", {})
        if clamp_part is not None and old_to_new:
            clamp_dofs: set[int] = set()
            for old_node_id in clamp_part.node_ids:
                local_node_id = old_to_new.get(int(old_node_id))
                if local_node_id is None:
                    continue
                for field_name in ("dx", "dy"):
                    gdof = solid["dh"].dof_map.get(field_name, {}).get(int(local_node_id))
                    if gdof is not None:
                        clamp_dofs.add(int(gdof))
            if clamp_dofs:
                solid["dh"].dof_tags.setdefault(geometry.clamp_tag, set()).update(clamp_dofs)

    fluid_iface_coords, _ = _boundary_field_data(fluid["dh"], "ux", geometry.interface_tag)
    solid_iface_coords, _ = _boundary_field_data(solid["dh"], "dx", geometry.interface_tag)
    if fluid_iface_coords.size == 0 or solid_iface_coords.size == 0:
        raise RuntimeError("Failed to extract interface DOF coordinates from the local fluid/solid subproblems.")
    map_used = _build_interface_restriction_matrix(solid["dh"], solid["d_k"], geometry.interface_tag)
    np.save(co_sim_dir / "map_used.npy", map_used)
    np.save(co_sim_dir / "coords_interf.npy", solid_iface_coords)
    np.save(co_sim_dir / "coords_interf_fluid.npy", fluid_iface_coords)
    if solid_operator_mode == "nirb":
        nirb_solid_predictor = NIRBSolidPredictor.from_path(
            Path(nirb_model_path),
            full_shape=tuple(np.asarray(solid["d_k"].nodal_values).shape),
            interface_matrix=map_used,
            interface_shape=(int(solid_iface_coords.shape[0]), 2),
        )
        _log(
            verbose,
            "[nirb] "
            f"loaded solid model from {Path(nirb_model_path)} "
            f"with interface restriction {tuple(map_used.shape)}",
        )
    solid_interface_mass = _build_interface_mass_matrix(mesh_s, solid_iface_coords, geometry.interface_tag)

    zero_load_lookup = CoordinateLookup(
        solid_iface_coords,
        np.zeros((solid_iface_coords.shape[0], 2), dtype=float),
        dim=2,
    )
    load_transfer_value = str(load_transfer).lower()
    accelerate_on_fluid_load = load_transfer_value == "reaction"
    monitor_stress_interface_loads = _should_monitor_stress_interface_loads(
        load_transfer=load_transfer_value,
        monitor_interface_loads=bool(monitor_interface_loads),
    )
    zero_fluid_load_lookup = CoordinateLookup(
        fluid_iface_coords,
        np.zeros((fluid_iface_coords.shape[0], 2), dtype=float),
        dim=2,
    )
    current_load_lookup = zero_fluid_load_lookup if accelerate_on_fluid_load else zero_load_lookup
    prev_disp_iter_vals = np.zeros((solid_iface_coords.shape[0], 2), dtype=float)

    mu_f = float(setup.material.density * setup.material.kinematic_viscosity)
    mu_s = float(setup.material.shear_modulus)
    lambda_s = float(setup.material.lame_lambda)
    sampled_lspg_hybrid_model: _SampledLSPGHybridModel | None = None
    sampled_lspg_hybrid_switch_iter = 0
    sampled_lspg_hybrid_max_iterations = 0
    sampled_lspg_hybrid_residual_tol = 0.0
    sampled_lspg_hybrid_stage_count = 0
    sampled_lspg_hybrid_exact_stage_count = 0
    sampled_lspg_hybrid_fallback_count = 0
    sampled_lspg_hybrid_load_gate_skips = 0
    sampled_lspg_hybrid_disp_gate_skips = 0
    sampled_lspg_hybrid_consecutive_gate_skips = 0
    sampled_lspg_hybrid_disabled_gate_skips = 0
    sampled_lspg_hybrid_exact_accept_forced_count = 0
    sampled_lspg_hybrid_trial_stage_count = 0
    sampled_lspg_hybrid_exact_correction_count = 0
    sampled_lspg_hybrid_trial_monitor_failures = 0
    sampled_lspg_hybrid_interface_reject_count = 0
    sampled_lspg_hybrid_interface_correction_count = 0
    fluid_hrom_disabled_until_step = -1
    fluid_hrom_min_previous_load_rel_value = max(0.0, float(fluid_hrom_min_previous_load_rel))
    fluid_hrom_max_previous_load_rel_value = float(fluid_hrom_max_previous_load_rel)
    fluid_hrom_max_trial_load_rel_error_value = float(fluid_hrom_max_trial_load_rel_error)
    if fluid_hrom_max_trial_load_rel_error_value < 0.0:
        raise ValueError("fluid_hrom_max_trial_load_rel_error must be non-negative.")
    fluid_hrom_disable_steps_after_trial_failure_value = max(0, int(fluid_hrom_disable_steps_after_trial_failure))
    fluid_hrom_interface_trust_value = str(fluid_hrom_interface_trust).strip().lower()
    if fluid_hrom_interface_trust_value not in {"none", "fallback", "clip"}:
        raise ValueError(f"Unsupported fluid_hrom_interface_trust={fluid_hrom_interface_trust!r}.")
    fluid_hrom_interface_max_step_ratio_value = float(fluid_hrom_interface_max_step_ratio)
    fluid_hrom_interface_max_load_rel_value = float(fluid_hrom_interface_max_load_rel)
    fluid_hrom_interface_min_correction_alpha_value = float(
        np.clip(float(fluid_hrom_interface_min_correction_alpha), 0.0, 1.0)
    )
    if fluid_hrom_interface_max_step_ratio_value < 0.0:
        raise ValueError("fluid_hrom_interface_max_step_ratio must be non-negative.")
    if fluid_hrom_interface_max_load_rel_value < 0.0:
        raise ValueError("fluid_hrom_interface_max_load_rel must be non-negative.")
    if fluid_hrom_min_previous_load_rel_value > fluid_hrom_max_previous_load_rel_value:
        raise ValueError(
            "fluid_hrom_min_previous_load_rel must be <= fluid_hrom_max_previous_load_rel "
            f"({fluid_hrom_min_previous_load_rel_value} > {fluid_hrom_max_previous_load_rel_value})."
        )
    fluid_hrom_max_consecutive_stages_value = max(0, int(fluid_hrom_max_consecutive_stages))
    if bool(use_sampled_lspg_hybrid_fluid):
        sampled_lspg_hybrid_model = _load_sampled_lspg_hybrid_model(
            Path(fluid_hrom_model_path),
            total_dofs=int(fluid["dh"].total_dofs),
            n_elements=int(mesh_f.n_elements),
        )
        sampled_lspg_hybrid_switch_iter = (
            max(1, int(fluid_hrom_switch_iter))
            if fluid_hrom_switch_iter is not None
            else int(sampled_lspg_hybrid_model.recommended_switch_iter)
        )
        sampled_lspg_hybrid_max_iterations = (
            max(1, int(fluid_hrom_max_iterations))
            if fluid_hrom_max_iterations is not None
            else int(sampled_lspg_hybrid_model.max_iterations)
        )
        sampled_lspg_hybrid_residual_tol = (
            float(fluid_hrom_residual_tol)
            if fluid_hrom_residual_tol is not None
            else float(sampled_lspg_hybrid_model.residual_tol)
        )
        _log(
            verbose,
            "[fluid-hrom] "
            f"loaded sampled-LSPG model from {sampled_lspg_hybrid_model.source_path} "
            f"modes={sampled_lspg_hybrid_model.n_modes} "
            f"sample_rows={sampled_lspg_hybrid_model.sample_row_dofs.size} "
            f"sample_elements={sampled_lspg_hybrid_model.sample_element_ids.size} "
            f"active_element_weights={int(np.count_nonzero(sampled_lspg_hybrid_model.sample_element_weights > 0.0))} "
            f"element_weight_sum={float(np.sum(sampled_lspg_hybrid_model.sample_element_weights)):.6e} "
            f"switch_iter={sampled_lspg_hybrid_switch_iter} "
            f"max_iterations={sampled_lspg_hybrid_max_iterations} "
            f"previous_load_rel_gate=[{float(fluid_hrom_min_previous_load_rel_value):.3e}, "
            f"{float(fluid_hrom_max_previous_load_rel_value):.3e}] "
            f"previous_disp_rel_gate={float(fluid_hrom_max_previous_disp_rel):.3e} "
            f"max_consecutive={int(fluid_hrom_max_consecutive_stages_value)} "
            f"require_exact_accept={int(bool(fluid_hrom_require_exact_accept))} "
            f"trial_exact_correct={int(bool(fluid_hrom_trial_exact_correct))} "
            f"trial_load_error_gate={float(fluid_hrom_max_trial_load_rel_error_value):.3e} "
            f"disable_steps_after_trial_failure={int(fluid_hrom_disable_steps_after_trial_failure_value)} "
            f"interface_trust={fluid_hrom_interface_trust_value} "
            f"interface_max_step_ratio={float(fluid_hrom_interface_max_step_ratio_value):.3e} "
            f"interface_max_load_rel={float(fluid_hrom_interface_max_load_rel_value):.3e} "
            f"interface_min_alpha={float(fluid_hrom_interface_min_correction_alpha_value):.3e}",
        )
    porous_material = _default_upl_material_from_lame(
        mu_s=mu_s,
        lambda_s=lambda_s,
        porosity=float(porous_porosity),
        biot_coefficient=float(porous_biot_coefficient),
        permeability=float(porous_permeability),
        storage_inverse=float(porous_storage_inverse),
        dynamic_viscosity_liquid=mu_f,
        density_solid=float(setup.material.density),
        density_liquid=float(setup.material.density),
    )

    fixed_mesh_tags = (
        geometry.inlet_tag,
        geometry.outlet_tag,
        geometry.walls_tag,
        geometry.cylinder_tag,
    )

    disp_snapshots: list[np.ndarray] = (
        _column_list_from_matrix(np.load(co_sim_dir / "disp_data.npy"))
        if restart_payload is not None and (co_sim_dir / "disp_data.npy").exists()
        else []
    )
    load_snapshots: list[np.ndarray] = (
        _column_list_from_matrix(np.load(co_sim_dir / "load_data.npy"))
        if restart_payload is not None and (co_sim_dir / "load_data.npy").exists()
        else []
    )
    load_guess_snapshots: list[np.ndarray] = (
        _column_list_from_matrix(np.load(co_sim_dir / "load_guess_data.npy"))
        if restart_payload is not None and (co_sim_dir / "load_guess_data.npy").exists()
        else []
    )
    load_return_snapshots: list[np.ndarray] = (
        _column_list_from_matrix(np.load(co_sim_dir / "load_return_data.npy"))
        if restart_payload is not None and (co_sim_dir / "load_return_data.npy").exists()
        else []
    )
    fluid_load_guess_snapshots: list[np.ndarray] = (
        _column_list_from_matrix(np.load(co_sim_dir / "fluid_load_guess_data.npy"))
        if restart_payload is not None and (co_sim_dir / "fluid_load_guess_data.npy").exists()
        else []
    )
    fluid_load_return_snapshots: list[np.ndarray] = (
        _column_list_from_matrix(np.load(co_sim_dir / "fluid_load_return_data.npy"))
        if restart_payload is not None and (co_sim_dir / "fluid_load_return_data.npy").exists()
        else []
    )
    interface_disp_snapshots: list[np.ndarray] = (
        _column_list_from_matrix(np.load(co_sim_dir / "interface_disp_data.npy"))
        if restart_payload is not None and (co_sim_dir / "interface_disp_data.npy").exists()
        else []
    )
    interface_velocity_snapshots: list[np.ndarray] = (
        _column_list_from_matrix(np.load(co_sim_dir / "interface_velocity_data.npy"))
        if restart_payload is not None and (co_sim_dir / "interface_velocity_data.npy").exists()
        else []
    )
    interface_traction_snapshots: list[np.ndarray] = (
        _column_list_from_matrix(np.load(co_sim_dir / "interface_traction_data.npy"))
        if restart_payload is not None and (co_sim_dir / "interface_traction_data.npy").exists()
        else []
    )
    reaction_load_snapshots: list[np.ndarray] = (
        _column_list_from_matrix(np.load(co_sim_dir / "load_reaction_data.npy"))
        if restart_payload is not None and (co_sim_dir / "load_reaction_data.npy").exists()
        else []
    )
    stress_load_snapshots: list[np.ndarray] = (
        _column_list_from_matrix(np.load(co_sim_dir / "load_stress_data.npy"))
        if restart_payload is not None and (co_sim_dir / "load_stress_data.npy").exists()
        else []
    )
    snapshot_rows: list[dict[str, object]] = (
        _load_csv_rows(output_dir / "snapshot_metadata.csv") if restart_payload is not None else []
    )
    step_rows: list[dict[str, object]] = _load_csv_rows(output_dir / "timeseries.csv") if restart_payload is not None else []
    fluid_times: list[float] = (
        np.load(co_sim_dir / "fluid_time.npy").astype(float).tolist()
        if restart_payload is not None and (co_sim_dir / "fluid_time.npy").exists()
        else []
    )
    structure_times: list[float] = (
        np.load(co_sim_dir / "structure_time.npy").astype(float).tolist()
        if restart_payload is not None and (co_sim_dir / "structure_time.npy").exists()
        else []
    )
    increment_times: list[float] = (
        np.load(co_sim_dir / "increment_time.npy").astype(float).tolist()
        if restart_payload is not None and (co_sim_dir / "increment_time.npy").exists()
        else []
    )
    vtk_rows: list[dict[str, object]] = _load_csv_rows(output_dir / "vtk_manifest.csv") if restart_payload is not None else []
    coupling_iters_per_step: list[int] = (
        np.load(co_sim_dir / "iters.npy").astype(int).tolist()
        if restart_payload is not None and (co_sim_dir / "iters.npy").exists()
        else []
    )
    converged_steps = 0
    t_total_start = time.perf_counter()
    iqn_old_dr_mats: deque[np.ndarray] = deque(maxlen=max(int(force_history) - 1, 0))
    iqn_old_dg_mats: deque[np.ndarray] = deque(maxlen=max(int(force_history) - 1, 0))
    restart_step = 0
    restart_time = 0.0
    exact_fluid_newton_tol = min(
        float(newton_tol),
        float(
            _env_float(
                "PYCUTFEM_EX2_EXACT_FLUID_NEWTON_TOL",
                1.0e-6,
            )
        ),
    )
    fluid_mixed_solution_criteria = (
        ("ux,uy", float(exact_fluid_newton_tol), float(exact_fluid_newton_tol)),
        ("p", float(exact_fluid_newton_tol), float(exact_fluid_newton_tol)),
    )
    if _env_bool("PYCUTFEM_EX2_DISABLE_EXACT_FLUID_MIXED_CRITERIA", False):
        fluid_mixed_solution_criteria = ()
    fluid_mixed_solution_residual_factor = _env_float(
        "PYCUTFEM_EX2_EXACT_FLUID_MIXED_RES_FACTOR",
        0.0,
    )

    if restart_payload is not None:
        solid["d_k"].nodal_values[:] = np.asarray(restart_payload["solid_d_k"], dtype=float)
        solid["d_prev"].nodal_values[:] = np.asarray(restart_payload["solid_d_prev"], dtype=float)
        if "p_k" in solid and "solid_p_k" in restart_payload:
            solid["p_k"].nodal_values[:] = np.asarray(restart_payload["solid_p_k"], dtype=float)
        if "p_prev" in solid and "solid_p_prev" in restart_payload:
            solid["p_prev"].nodal_values[:] = np.asarray(restart_payload["solid_p_prev"], dtype=float)
        fluid["u_k"].nodal_values[:] = np.asarray(restart_payload["fluid_u_k"], dtype=float)
        fluid["p_k"].nodal_values[:] = np.asarray(restart_payload["fluid_p_k"], dtype=float)
        fluid["u_prev"].nodal_values[:] = np.asarray(restart_payload["fluid_u_prev"], dtype=float)
        fluid["p_prev"].nodal_values[:] = np.asarray(restart_payload["fluid_p_prev"], dtype=float)
        fluid["a_prev"].nodal_values[:] = np.asarray(restart_payload["fluid_a_prev"], dtype=float)
        fluid["a_k"].nodal_values[:] = np.asarray(
            restart_payload.get("fluid_a_k", restart_payload["fluid_a_prev"]),
            dtype=float,
        )
        fluid["d_mesh"].nodal_values[:] = np.asarray(restart_payload["fluid_d_mesh"], dtype=float)
        fluid["d_prev"].nodal_values[:] = np.asarray(restart_payload["fluid_d_prev"], dtype=float)
        fluid["d_prev2"].nodal_values[:] = np.asarray(restart_payload["fluid_d_prev2"], dtype=float)
        fluid["w_mesh_prev"].nodal_values[:] = np.asarray(restart_payload["fluid_w_mesh_prev"], dtype=float)
        fluid["a_mesh_prev"].nodal_values[:] = np.asarray(restart_payload["fluid_a_mesh_prev"], dtype=float)
        fluid["w_mesh_k"].nodal_values[:] = fluid["w_mesh_prev"].nodal_values[:]
        fluid["a_mesh_k"].nodal_values[:] = fluid["a_mesh_prev"].nodal_values[:]
        _restore_fluid_dvms_state(
            fluid.get("dvms_state"),
            _fluid_dvms_restart_snapshot_from_payload(fluid.get("dvms_state"), restart_payload),
        )
        current_load_lookup = CoordinateLookup(
            fluid_iface_coords if accelerate_on_fluid_load else solid_iface_coords,
            np.asarray(restart_payload["current_load_values"], dtype=float),
            dim=2,
        )
        iqn_old_dr_mats = _iqn_history_from_payload(
            data=restart_payload,
            prefix="iqn_old_dr",
            maxlen=max(int(force_history) - 1, 0),
        )
        iqn_old_dg_mats = _iqn_history_from_payload(
            data=restart_payload,
            prefix="iqn_old_dg",
            maxlen=max(int(force_history) - 1, 0),
        )
        restart_step = int(np.asarray(restart_payload["step"], dtype=int).reshape(-1)[0])
        restart_time = float(np.asarray(restart_payload["time_s"], dtype=float).reshape(-1)[0])
        converged_steps = max(int(restart_step), int(len(coupling_iters_per_step)))
        mesh_restart_lookup = _vector_lookup_from_field(fluid["dh"], fluid["d_mesh"])
        _transfer_vector_field(
            target_dh=mesh_ext["dh"],
            target_vec=mesh_ext["m_k"],
            source_lookup=mesh_restart_lookup,
        )
        _transfer_vector_field(
            target_dh=mesh_ext["dh"],
            target_vec=mesh_ext["m_prev_geom"],
            source_lookup=CoordinateLookup(
                np.asarray(mesh_restart_lookup.coords, dtype=float),
                np.zeros_like(np.asarray(mesh_restart_lookup.values, dtype=float)),
                dim=2,
            ),
        )
        _section(
            verbose,
            f"[restart] resumed from {restart_path} at accepted step={restart_step} time={restart_time:.6f}s",
        )

    for step in range(int(restart_step) + 1, step_count + 1):
        t_now = min(end_time_value, step * dt_value)
        _section(verbose, f"[time] start step={step}/{step_count} t={t_now:.6f}s dt={dt_value:.6e}")
        if kratos_exact_structure_backend is not None:
            _advance_kratos_exact_structure_backend_step(backend=kratos_exact_structure_backend)
        if kratos_exact_fluid_backend is not None:
            _advance_kratos_exact_fluid_backend_step(backend=kratos_exact_fluid_backend)
        elif kratos_mesh_backend is not None:
            _advance_kratos_mesh_motion_backend_step(backend=kratos_mesh_backend)
        elif kratos_local_mesh_backend is not None:
            _advance_kratos_mesh_motion_backend_step(backend=kratos_local_mesh_backend)
        increment_start = time.perf_counter()
        solid_prev_step = _snapshot_function_values(_solid_prev_functions(solid))
        fluid_prev_step = _snapshot_function_values([fluid["u_prev"], fluid["p_prev"]])
        fluid_mesh_prev_step = _snapshot_function_values(
            [fluid["d_prev"], fluid["d_prev2"], fluid["w_mesh_prev"], fluid["a_mesh_prev"]]
        )
        fluid_dvms_prev_step = _snapshot_fluid_dvms_state(fluid.get("dvms_state"))
        if step == 1:
            prev_disp_iter_vals.fill(0.0)
        else:
            _, prev_disp_iter_vals = _boundary_vector_snapshot(solid["dh"], solid["d_prev"], geometry.interface_tag)

        def inlet_profile(x: float, y: float) -> float:
            del x
            return geometry.inlet_velocity(y, t_now, reference_velocity=reference_velocity_value)

        if kratos_exact_fluid_backend is None:
            zero_predictor_iface_velocity = CoordinateLookup(
                fluid_iface_coords,
                np.zeros((fluid_iface_coords.shape[0], 2), dtype=float),
                dim=2,
            )
            fluid_predictor_bcs, _ = _fluid_boundary_conditions(
                iface_velocity=zero_predictor_iface_velocity,
                inlet_lookup=inlet_profile,
                interface_tag=geometry.interface_tag,
                outlet_tag=geometry.outlet_tag,
                walls_tag=geometry.walls_tag,
                cylinder_tag=geometry.cylinder_tag,
            )
            _predict_fluid_bossak_step_state(
                fluid=fluid,
                dt=float(dt_value),
                bossak_alpha=float(bossak_alpha),
                predictor_bcs=fluid_predictor_bcs,
            )

        step_prev_disp_iter_vals = np.asarray(prev_disp_iter_vals, dtype=float).copy()
        solid_step_state = _snapshot_function_values(_solid_current_and_prev_functions(solid))
        fluid_step_state = _snapshot_function_values(
            [
                fluid["u_k"],
                fluid["p_k"],
                fluid["u_prev"],
                fluid["p_prev"],
                fluid["a_prev"],
                fluid["a_k"],
                fluid["d_mesh"],
                fluid["d_prev"],
                fluid["d_prev2"],
                fluid["w_mesh_prev"],
                fluid["a_mesh_prev"],
                fluid["w_mesh_k"],
                fluid["a_mesh_k"],
            ]
        )
        mesh_ext_step_state = _snapshot_function_values([mesh_ext["m_k"], mesh_ext["m_prev_geom"]])
        current_load_lookup_step_start = _copy_lookup(current_load_lookup)
        iqn_old_dr_step_start = [np.asarray(mat, dtype=float).copy() for mat in iqn_old_dr_mats]
        iqn_old_dg_step_start = [np.asarray(mat, dtype=float).copy() for mat in iqn_old_dg_mats]
        allow_step_retries = (
            kratos_exact_structure_backend is None
            and kratos_exact_fluid_backend is None
            and kratos_mesh_backend is None
        )
        if allow_step_retries:
            retry_policies = _build_coupling_retry_policies(
                force_update=str(force_update).lower(),
                force_relaxation=float(force_relaxation),
                force_relaxation_min=float(force_relaxation_min),
                force_relaxation_max=float(force_relaxation_max),
                base_max_coupling_iters=int(max_coupling_iters),
                base_max_newton_iter=int(max_newton_iter),
                max_retries=int(step_retries),
            )
        else:
            retry_policies = [
                _CouplingRetryPolicy(
                    force_update=str(force_update).lower(),
                    force_relaxation=float(force_relaxation),
                    max_coupling_iters=int(max_coupling_iters),
                    reset_interface_history=False,
                )
            ]

        step_converged = False
        last_disp_abs = last_disp_rel = last_load_abs = last_load_rel = float("nan")
        last_force_omega = float(force_relaxation)
        last_returned_load_lookup: CoordinateLookup | None = None
        last_mesh_vel_fluid_lookup: CoordinateLookup | None = None
        last_mesh_accel_fluid_lookup: CoordinateLookup | None = None
        last_fluid_accel_lookup: CoordinateLookup | None = None
        last_nirb_prediction = None
        coupling_iter = 0
        active_force_update = str(force_update).lower()
        active_force_relaxation = float(force_relaxation)
        attempt_index = 1
        increment_elapsed = float("nan")

        for attempt_index, retry_policy in enumerate(retry_policies, start=1):
            step_marker = _step_progress_marker(
                disp_snapshots=disp_snapshots,
                load_snapshots=load_snapshots,
                load_guess_snapshots=load_guess_snapshots,
                load_return_snapshots=load_return_snapshots,
                fluid_load_guess_snapshots=fluid_load_guess_snapshots,
                fluid_load_return_snapshots=fluid_load_return_snapshots,
                interface_disp_snapshots=interface_disp_snapshots,
                interface_velocity_snapshots=interface_velocity_snapshots,
                interface_traction_snapshots=interface_traction_snapshots,
                reaction_load_snapshots=reaction_load_snapshots,
                stress_load_snapshots=stress_load_snapshots,
                snapshot_rows=snapshot_rows,
                step_rows=step_rows,
                fluid_times=fluid_times,
                structure_times=structure_times,
                increment_times=increment_times,
            )
            _restore_function_values(_solid_current_and_prev_functions(solid), solid_step_state)
            _restore_function_values(
                [
                    fluid["u_k"],
                    fluid["p_k"],
                    fluid["u_prev"],
                    fluid["p_prev"],
                    fluid["a_prev"],
                    fluid["a_k"],
                    fluid["d_mesh"],
                    fluid["d_prev"],
                    fluid["d_prev2"],
                    fluid["w_mesh_prev"],
                    fluid["a_mesh_prev"],
                    fluid["w_mesh_k"],
                    fluid["a_mesh_k"],
                ],
                fluid_step_state,
            )
            _restore_function_values([mesh_ext["m_k"], mesh_ext["m_prev_geom"]], mesh_ext_step_state)
            _restore_fluid_dvms_state(
                fluid.get("dvms_state"),
                fluid_dvms_prev_step,
            )
            current_load_lookup = _copy_lookup(current_load_lookup_step_start)
            if bool(retry_policy.reset_interface_history):
                iqn_old_dr_mats = deque(maxlen=max(int(force_history) - 1, 0))
                iqn_old_dg_mats = deque(maxlen=max(int(force_history) - 1, 0))
            else:
                iqn_old_dr_mats = deque(
                    [np.asarray(mat, dtype=float).copy() for mat in iqn_old_dr_step_start],
                    maxlen=max(int(force_history) - 1, 0),
                )
                iqn_old_dg_mats = deque(
                    [np.asarray(mat, dtype=float).copy() for mat in iqn_old_dg_step_start],
                    maxlen=max(int(force_history) - 1, 0),
                )

            step_converged = False
            coupling_iter = 0
            active_force_update = str(retry_policy.force_update).lower()
            active_force_relaxation = float(retry_policy.force_relaxation)
            step_max_coupling_iters = int(retry_policy.max_coupling_iters)
            prev_force_residual: np.ndarray | None = None
            load_guess_history = []
            load_return_history = []
            prev_disp_iter_vals = np.asarray(step_prev_disp_iter_vals, dtype=float).copy()
            last_disp_abs = last_disp_rel = last_load_abs = last_load_rel = float("nan")
            last_force_omega = float(active_force_relaxation)
            last_returned_load_lookup = None
            last_mesh_vel_fluid_lookup = None
            last_mesh_accel_fluid_lookup = None
            last_fluid_accel_lookup = None
            last_nirb_prediction = None
            consecutive_hrom_stages = 0
            increment_start = time.perf_counter()

            try:
                for coupling_iter in range(1, int(step_max_coupling_iters) + 1):
                    _section(verbose, f"[fsi] step={step} fixed-point iter={coupling_iter}/{step_max_coupling_iters}")
                    if accelerate_on_fluid_load:
                        current_structure_load_lookup = _resample_lookup_to_coords(
                            _negate_lookup(current_load_lookup),
                            solid_iface_coords,
                        )
                    else:
                        current_structure_load_lookup = current_load_lookup
                    load_guess_vals = np.asarray(current_structure_load_lookup.values, dtype=float).copy()

                    t_solid0 = time.perf_counter()
                    nirb_prediction = None
                    if kratos_exact_structure_backend is not None:
                        kratos_structure_state = _solve_kratos_exact_structure_backend(
                            backend=kratos_exact_structure_backend,
                            structure_load=current_structure_load_lookup,
                        )
                        _transfer_vector_field(
                            target_dh=solid["dh"],
                            target_vec=solid["d_k"],
                            source_lookup=kratos_structure_state["displacement"],
                        )
                    elif nirb_solid_predictor is not None and int(step) >= int(nirb_start_step_value):
                        nirb_prediction = nirb_solid_predictor.predict_interface(load_guess_vals)
                        predicted_interface = np.asarray(nirb_prediction.interface_displacement, dtype=float)
                        if not np.all(np.isfinite(predicted_interface)):
                            raise RuntimeError("NIRB solid predictor returned non-finite interface displacement values.")
                        last_nirb_prediction = nirb_prediction
                    elif solid_operator_mode == "porous":
                        solid_res, solid_jac, solid_bcs, solid_bcs_homog = _porous_solid_residual_and_jacobian(
                            prob=solid,
                            material=porous_material,
                            dt=float(dt_value),
                            clamp_tag=geometry.clamp_tag,
                            quad_order=solid_quad_order,
                        )
                        exact_structure_newton_tol = max(
                            0.0,
                            float(_env_float("PYCUTFEM_EX2_POROUS_STRUCTURE_NEWTON_TOL", float(newton_tol))),
                        )
                        exact_structure_max_newton_iter = int(
                            max(
                                1,
                                _env_float(
                                    "PYCUTFEM_EX2_POROUS_STRUCTURE_MAX_NEWTON_ITER",
                                    float(max_newton_iter),
                                ),
                            )
                        )
                        exact_structure_linear_backend = _env_str(
                            "PYCUTFEM_EX2_POROUS_STRUCTURE_LINEAR_BACKEND",
                            str(linear_backend),
                        ).strip().lower()
                        solid_solver = NewtonSolver(
                            residual_form=solid_res,
                            jacobian_form=solid_jac,
                            dof_handler=solid["dh"],
                            mixed_element=solid["me"],
                            bcs=solid_bcs,
                            bcs_homog=solid_bcs_homog,
                            newton_params=NewtonParameters(
                                newton_tol=float(exact_structure_newton_tol),
                                newton_rtol=0.0,
                                residual_norm="linf",
                                max_newton_iter=int(exact_structure_max_newton_iter),
                                print_level=3,
                                line_search=False,
                                globalization="none",
                            ),
                            lin_params=LinearSolverParameters(backend=str(exact_structure_linear_backend)),
                            quad_order=solid_quad_order,
                            backend=str(backend),
                        )
                        solid_point_load_full = _boundary_point_load_vector(
                            solid["dh"],
                            vector=solid["d_k"],
                            tag=geometry.interface_tag,
                            values=load_guess_vals,
                        )
                        solid_point_load_red = np.asarray(
                            solid_point_load_full[np.asarray(solid_solver.active_dofs, dtype=int)],
                            dtype=float,
                        )
                        solid_solver.set_runtime_operators([_ReducedResidualShiftOperator(solid_point_load_red)])
                        solid_guess = _snapshot_function_values(_solid_current_functions(solid))
                        _restore_function_values(_solid_prev_functions(solid), solid_prev_step)
                        solid_solver.solve_time_interval(
                            functions=_solid_current_functions(solid),
                            prev_functions=_solid_prev_functions(solid),
                            time_params=TimeStepperParameters(
                                dt=1.0,
                                max_steps=1,
                                final_time=1.0,
                                stop_on_steady=False,
                                step_initial_guess_callback=_guess_callback_from_snapshots_with_dirichlet(
                                    snapshots=solid_guess,
                                    dh=solid["dh"],
                                    bcs=solid_bcs,
                                ),
                            ),
                        )
                        _restore_function_values(_solid_prev_functions(solid), solid_prev_step)
                    else:
                        solid_res, solid_jac, solid_bcs, solid_bcs_homog = _solid_residual_and_jacobian(
                            prob=solid,
                            traction_lookup=zero_load_lookup,
                            mu_s=mu_s,
                            lambda_s=lambda_s,
                            interface_tag=geometry.interface_tag,
                            clamp_tag=geometry.clamp_tag,
                            quad_order=solid_quad_order,
                        )
                        use_struct_one_step = _env_bool(
                            "PYCUTFEM_EX2_STRUCT_ONE_STEP",
                            False,
                        )
                        default_use_kratos_structure_profile = bool(
                            mesh_source_value == "reference" and fluid_operator_mode == "exact"
                        )
                        use_kratos_structure_profile = _env_bool(
                            "PYCUTFEM_EX2_LOCAL_STRUCTURE_USE_KRATOS_PROFILE",
                            default_use_kratos_structure_profile,
                        )
                        default_structure_newton_tol = (
                            float(structure_solver_profile["residual_absolute_tolerance"])
                            if bool(use_kratos_structure_profile)
                            else float(newton_tol)
                        )
                        default_structure_newton_rtol = (
                            float(structure_solver_profile["residual_relative_tolerance"])
                            if bool(use_kratos_structure_profile)
                            else 0.0
                        )
                        default_structure_max_newton_iter = (
                            int(structure_solver_profile["max_iteration"])
                            if bool(use_kratos_structure_profile)
                            else int(max_newton_iter)
                        )
                        default_structure_linear_backend = (
                            str(structure_solver_profile["linear_backend"])
                            if bool(use_kratos_structure_profile)
                            else str(linear_backend)
                        )
                        default_structure_residual_norm = (
                            "kratos_l2_over_ndof"
                            if bool(use_kratos_structure_profile)
                            else "linf"
                        )
                        exact_structure_newton_tol = max(
                            0.0,
                            float(
                                _env_float(
                                    "PYCUTFEM_EX2_LOCAL_STRUCTURE_NEWTON_TOL",
                                    float(default_structure_newton_tol),
                                )
                            ),
                        )
                        exact_structure_newton_rtol = max(
                            0.0,
                            float(
                                _env_float(
                                    "PYCUTFEM_EX2_LOCAL_STRUCTURE_NEWTON_RTOL",
                                    float(default_structure_newton_rtol),
                                )
                            ),
                        )
                        exact_structure_max_newton_iter = int(
                            max(
                                1,
                                _env_float(
                                    "PYCUTFEM_EX2_LOCAL_STRUCTURE_MAX_NEWTON_ITER",
                                    float(default_structure_max_newton_iter),
                                ),
                            )
                        )
                        exact_structure_linear_backend = _env_str(
                            "PYCUTFEM_EX2_LOCAL_STRUCTURE_LINEAR_BACKEND",
                            str(default_structure_linear_backend),
                        ).strip().lower()
                        exact_structure_residual_norm = _env_str(
                            "PYCUTFEM_EX2_LOCAL_STRUCTURE_RESIDUAL_NORM",
                            str(default_structure_residual_norm),
                        ).strip().lower()
                        solid_solver = NewtonSolver(
                            residual_form=solid_res,
                            jacobian_form=solid_jac,
                            dof_handler=solid["dh"],
                            mixed_element=solid["me"],
                            bcs=solid_bcs,
                            bcs_homog=solid_bcs_homog,
                            newton_params=NewtonParameters(
                                newton_tol=float(exact_structure_newton_tol),
                                newton_rtol=float(exact_structure_newton_rtol),
                                residual_norm=str(exact_structure_residual_norm),
                                max_newton_iter=1 if bool(use_struct_one_step) else int(exact_structure_max_newton_iter),
                                print_level=3,
                                accept_nonconverged_atol_factor=(
                                    float(_EX2L_KRATOS_STRUCT_ONE_STEP_ACCEPT_FACTOR)
                                    if bool(use_struct_one_step)
                                    else 0.0
                                ),
                                line_search=False,
                                globalization="none",
                            ),
                            lin_params=LinearSolverParameters(backend=str(exact_structure_linear_backend)),
                            quad_order=solid_quad_order,
                            backend=str(backend),
                        )
                        use_kratos_structure_linear_permutation = _env_bool(
                            "PYCUTFEM_EX2_LOCAL_STRUCTURE_PERMUTE_TO_KRATOS_ORDER",
                            bool(use_kratos_structure_profile),
                        )
                        if bool(use_kratos_structure_linear_permutation):
                            structure_linear_perm = _structure_kratos_active_dof_permutation(
                                dh=solid["dh"],
                                mesh=mesh_s,
                                active_dofs=np.asarray(solid_solver.active_dofs, dtype=int),
                            )
                            if structure_linear_perm is not None and structure_linear_perm.size:
                                solid_solver._linear_solve_perm = np.asarray(structure_linear_perm, dtype=int)
                        solid_point_load_full = _boundary_point_load_vector(
                            solid["dh"],
                            vector=solid["d_k"],
                            tag=geometry.interface_tag,
                            values=load_guess_vals,
                        )
                        solid_point_load_red = np.asarray(
                            solid_point_load_full[np.asarray(solid_solver.active_dofs, dtype=int)],
                            dtype=float,
                        )
                        solid_runtime_ops: list[RuntimeOperator] = []
                        use_kratos_local_global_solid_system = (
                            kratos_local_solid_backend is not None
                            and _env_bool(
                                "PYCUTFEM_EX2_LOCAL_STRUCTURE_USE_KRATOS_GLOBAL_SYSTEM",
                                False,
                            )
                        )
                        use_kratos_local_structure_conditions = (
                            kratos_local_solid_backend is not None
                            and _env_bool(
                                "PYCUTFEM_EX2_LOCAL_STRUCTURE_USE_KRATOS_POINT_CONDITIONS",
                                False,
                            )
                        )
                        if bool(use_kratos_local_global_solid_system):
                            solid_runtime_ops.append(
                                _KratosLocalSolidGlobalSystemOperator(
                                    backend=kratos_local_solid_backend,
                                    d_k=solid["d_k"],
                                    structure_load=current_structure_load_lookup,
                                )
                            )
                        else:
                            if kratos_local_solid_backend is not None:
                                solid_runtime_ops.append(
                                    _KratosLocalSolidSystemOperator(
                                        backend=kratos_local_solid_backend,
                                        d_k=solid["d_k"],
                                        structure_load=(
                                            current_structure_load_lookup
                                            if bool(use_kratos_local_structure_conditions)
                                            else None
                                        ),
                                    )
                                )
                            if not bool(use_kratos_local_structure_conditions):
                                solid_runtime_ops.append(_ReducedResidualShiftOperator(solid_point_load_red))
                        solid_solver.set_runtime_operators(solid_runtime_ops)
                        solid_guess = _snapshot_function_values([solid["d_k"]])
                        _restore_function_values([solid["d_prev"]], solid_prev_step)
                        _maybe_dump_structure_stage_probe(
                            output_dir=output_dir,
                            step=int(step),
                            coupling_iter=int(coupling_iter),
                            stage_label="pre_structure_solve",
                            dt=float(dt_value),
                            solid=solid,
                            interface_tag=geometry.interface_tag,
                            structure_load_lookup=current_structure_load_lookup,
                            point_load_full=solid_point_load_full,
                        )
                        solid_solver.solve_time_interval(
                            functions=[solid["d_k"]],
                            prev_functions=[solid["d_prev"]],
                            time_params=TimeStepperParameters(
                                dt=1.0,
                                max_steps=1,
                                final_time=1.0,
                                stop_on_steady=False,
                                step_initial_guess_callback=_guess_callback_from_snapshots_with_dirichlet(
                                    snapshots=solid_guess,
                                    dh=solid["dh"],
                                    bcs=solid_bcs,
                                ),
                            ),
                        )
                        _maybe_dump_structure_stage_probe(
                            output_dir=output_dir,
                            step=int(step),
                            coupling_iter=int(coupling_iter),
                            stage_label="post_structure_solve",
                            dt=float(dt_value),
                            solid=solid,
                            interface_tag=geometry.interface_tag,
                            structure_load_lookup=current_structure_load_lookup,
                            point_load_full=solid_point_load_full,
                        )
                        _restore_function_values([solid["d_prev"]], solid_prev_step)
                    solid_elapsed = time.perf_counter() - t_solid0
                    structure_times.append(float(solid_elapsed))

                    if kratos_exact_structure_backend is not None:
                        solid_disp_solid_lookup = _resample_lookup_to_coords(
                            kratos_structure_state["interface_displacement"],
                            solid_iface_coords,
                        )
                    elif nirb_prediction is not None:
                        prev_iface_mesh_vel_lookup = _resample_lookup_to_coords(
                            _vector_lookup_from_field(fluid["dh"], fluid["w_mesh_prev"]),
                            solid_iface_coords,
                        )
                        prev_iface_mesh_acc_lookup = _resample_lookup_to_coords(
                            _vector_lookup_from_field(fluid["dh"], fluid["a_mesh_prev"]),
                            solid_iface_coords,
                        )
                        _, disp_prev_vals = _boundary_vector_snapshot(
                            solid["dh"],
                            solid["d_prev"],
                            geometry.interface_tag,
                        )
                        solid_disp_solid_lookup, _ = _interface_disp_velocity_from_values(
                            iface_coords=solid_iface_coords,
                            disp_vals=np.asarray(nirb_prediction.interface_displacement, dtype=float).reshape(-1, 2),
                            disp_prev_vals=disp_prev_vals,
                            dt=dt_value,
                            v_prev_lookup=prev_iface_mesh_vel_lookup,
                            a_prev_lookup=prev_iface_mesh_acc_lookup,
                            bossak_alpha=float(bossak_alpha),
                        )
                    else:
                        prev_iface_mesh_vel_lookup = _resample_lookup_to_coords(
                            _vector_lookup_from_field(fluid["dh"], fluid["w_mesh_prev"]),
                            solid_iface_coords,
                        )
                        prev_iface_mesh_acc_lookup = _resample_lookup_to_coords(
                            _vector_lookup_from_field(fluid["dh"], fluid["a_mesh_prev"]),
                            solid_iface_coords,
                        )
                        solid_disp_solid_lookup, _ = _solid_interface_disp_velocity(
                            dh=solid["dh"],
                            mesh=mesh_s,
                            d_curr=solid["d_k"],
                            d_prev=solid["d_prev"],
                            iface_coords=solid_iface_coords,
                            dt=dt_value,
                            v_prev_lookup=prev_iface_mesh_vel_lookup,
                            a_prev_lookup=prev_iface_mesh_acc_lookup,
                            bossak_alpha=float(bossak_alpha),
                        )
                    # Kratos uses a nearest-neighbor mapper for structure
                    # DISPLACEMENT -> fluid MESH_DISPLACEMENT in DoubleFlap. Replicate
                    # that transfer instead of evaluating the solid FE field directly
                    # at the fluid interface coordinates.
                    solid_disp_fluid_lookup = _resample_lookup_to_coords(
                        solid_disp_solid_lookup,
                        fluid_iface_coords,
                    )
                    disp_snapshot = None if nirb_prediction is not None else _flatten_vector_snapshot(solid["dh"], solid["d_k"])
        
                    reaction_point_load_lookup = None
                    reaction_solid_load_lookup = None
                    stress_point_load_lookup = None
                    if kratos_exact_fluid_backend is not None:
                        t_fluid0 = time.perf_counter()
                        kratos_fluid_state = _solve_kratos_exact_fluid_backend(
                            backend=kratos_exact_fluid_backend,
                            interface_disp=solid_disp_fluid_lookup,
                        )
                        mesh_lookup = kratos_fluid_state["mesh_displacement"]
                        mesh_vel_fluid_lookup = kratos_fluid_state["mesh_velocity"]
                        mesh_accel_fluid_lookup = kratos_fluid_state["mesh_acceleration"]
                        last_fluid_accel_lookup = kratos_fluid_state["acceleration"]
                        _transfer_vector_field(target_dh=fluid["dh"], target_vec=fluid["d_mesh"], source_lookup=mesh_lookup)
                        _transfer_vector_field(
                            target_dh=fluid["dh"],
                            target_vec=fluid["w_mesh_k"],
                            source_lookup=mesh_vel_fluid_lookup,
                        )
                        _transfer_vector_field(
                            target_dh=fluid["dh"],
                            target_vec=fluid["a_mesh_k"],
                            source_lookup=mesh_accel_fluid_lookup,
                        )
                        _transfer_vector_field(
                            target_dh=fluid["dh"],
                            target_vec=fluid["u_k"],
                            source_lookup=kratos_fluid_state["velocity"],
                        )
                        _transfer_vector_field(
                            target_dh=fluid["dh"],
                            target_vec=fluid["a_k"],
                            source_lookup=kratos_fluid_state["acceleration"],
                        )
                        _transfer_scalar_field(
                            target_dh=fluid["dh"],
                            target_fun=fluid["p_k"],
                            source_lookup=kratos_fluid_state["pressure"],
                        )
                        _transfer_vector_field(target_dh=mesh_ext["dh"], target_vec=mesh_ext["m_k"], source_lookup=mesh_lookup)
                        _transfer_vector_field(
                            target_dh=mesh_ext["dh"],
                            target_vec=mesh_ext["m_prev_geom"],
                            source_lookup=mesh_lookup,
                        )
                        last_mesh_vel_fluid_lookup = mesh_vel_fluid_lookup
                        last_mesh_accel_fluid_lookup = mesh_accel_fluid_lookup
                        fluid_elapsed = time.perf_counter() - t_fluid0
                        fluid_times.append(float(fluid_elapsed))
                        _log(
                            verbose,
                            "[fluid-stage] "
                            f"mode={fluid_operator_mode}/{exact_fluid_backend_mode} "
                            f"u_max={_field_abs_max(fluid['u_k']):.3e} "
                            f"p_max={_field_abs_max(fluid['p_k']):.3e}",
                        )
                        if load_transfer_value == "reaction" or bool(monitor_interface_loads):
                            reaction_point_load_lookup = _resample_lookup_to_coords(
                                kratos_fluid_state["reaction"],
                                fluid_iface_coords,
                            )
                            reaction_solid_load_lookup = _resample_lookup_to_coords(
                                _negate_lookup(reaction_point_load_lookup),
                                solid_iface_coords,
                            )
                        if bool(monitor_stress_interface_loads):
                            stress_point_load_lookup = _fluid_interface_point_loads_on_solid(
                                fluid_dh=fluid["dh"],
                                fluid_mesh=mesh_f,
                                solid_mesh=mesh_s,
                                u=fluid["u_k"],
                                p=fluid["p_k"],
                                d_mesh=fluid["d_mesh"],
                                solid_iface_coords=solid_iface_coords,
                                interface_tag=geometry.interface_tag,
                                mu_f=mu_f,
                                quad_order=quad_order,
                            )
                        if use_exact_fluid_operator:
                            _maybe_dump_exact_fluid_probe(
                                output_dir=output_dir,
                                step=int(step),
                                coupling_iter=int(coupling_iter),
                                stage_label="post_fluid_solve",
                                bc_scale=1.0,
                                dt=float(dt_value),
                                bossak_alpha=float(bossak_alpha),
                                fluid=fluid,
                                reaction_point_load_lookup=reaction_point_load_lookup,
                                reaction_solid_load_lookup=reaction_solid_load_lookup,
                            )
                        continuation_scales = (1.0,)
                    else:
                        t_mesh_stage0 = time.perf_counter()
                        if kratos_mesh_backend is not None:
                            if (
                                int(coupling_iter) > 1
                                and _env_bool("PYCUTFEM_EX2_SYNC_KRATOS_MESH_CURRENT_STATE", False)
                                and last_mesh_vel_fluid_lookup is not None
                                and last_mesh_accel_fluid_lookup is not None
                            ):
                                _sync_kratos_mesh_motion_backend_current_state(
                                    backend=kratos_mesh_backend,
                                    mesh_disp=_vector_lookup_from_field(fluid["dh"], fluid["d_mesh"]),
                                    mesh_vel=last_mesh_vel_fluid_lookup,
                                    mesh_acc=last_mesh_accel_fluid_lookup,
                                )
                            mesh_lookup, mesh_vel_fluid_lookup, mesh_accel_fluid_lookup = _solve_kratos_mesh_motion_backend(
                                backend=kratos_mesh_backend,
                                interface_disp=solid_disp_fluid_lookup,
                            )
                            _transfer_vector_field(target_dh=fluid["dh"], target_vec=fluid["d_mesh"], source_lookup=mesh_lookup)
                            _transfer_vector_field(
                                target_dh=fluid["dh"],
                                target_vec=fluid["w_mesh_k"],
                                source_lookup=mesh_vel_fluid_lookup,
                            )
                            _transfer_vector_field(
                                target_dh=fluid["dh"],
                                target_vec=fluid["a_mesh_k"],
                                source_lookup=mesh_accel_fluid_lookup,
                            )
                            _transfer_vector_field(target_dh=mesh_ext["dh"], target_vec=mesh_ext["m_k"], source_lookup=mesh_lookup)
                            _transfer_vector_field(
                                target_dh=mesh_ext["dh"],
                                target_vec=mesh_ext["m_prev_geom"],
                                source_lookup=CoordinateLookup(
                                    np.asarray(mesh_lookup.coords, dtype=float),
                                    np.zeros_like(np.asarray(mesh_lookup.values, dtype=float)),
                                    dim=2,
                                ),
                            )
                        elif kratos_local_mesh_backend is not None:
                            if (
                                int(coupling_iter) > 1
                                and _env_bool("PYCUTFEM_EX2_SYNC_KRATOS_MESH_CURRENT_STATE", False)
                                and last_mesh_vel_fluid_lookup is not None
                                and last_mesh_accel_fluid_lookup is not None
                            ):
                                _sync_kratos_mesh_motion_backend_current_state(
                                    backend=kratos_local_mesh_backend,
                                    mesh_disp=_vector_lookup_from_field(fluid["dh"], fluid["d_mesh"]),
                                    mesh_vel=last_mesh_vel_fluid_lookup,
                                    mesh_acc=last_mesh_accel_fluid_lookup,
                                )
                            mesh_lookup, mesh_vel_fluid_lookup, mesh_accel_fluid_lookup = _solve_kratos_mesh_motion_backend(
                                backend=kratos_local_mesh_backend,
                                interface_disp=solid_disp_fluid_lookup,
                            )
                            _transfer_vector_field(target_dh=fluid["dh"], target_vec=fluid["d_mesh"], source_lookup=mesh_lookup)
                            _transfer_vector_field(
                                target_dh=fluid["dh"],
                                target_vec=fluid["w_mesh_k"],
                                source_lookup=mesh_vel_fluid_lookup,
                            )
                            _transfer_vector_field(
                                target_dh=fluid["dh"],
                                target_vec=fluid["a_mesh_k"],
                                source_lookup=mesh_accel_fluid_lookup,
                            )
                            _transfer_vector_field(target_dh=mesh_ext["dh"], target_vec=mesh_ext["m_k"], source_lookup=mesh_lookup)
                            _transfer_vector_field(
                                target_dh=mesh_ext["dh"],
                                target_vec=mesh_ext["m_prev_geom"],
                                source_lookup=CoordinateLookup(
                                    np.asarray(mesh_lookup.coords, dtype=float),
                                    np.zeros_like(np.asarray(mesh_lookup.values, dtype=float)),
                                    dim=2,
                                ),
                            )
                        else:
                            use_local_mesh_current_geometry = _env_bool(
                                "PYCUTFEM_EX2_LOCAL_MESH_USE_CURRENT_GEOMETRY",
                                True,
                            )
                            if use_local_mesh_current_geometry:
                                _transfer_vector_field(
                                    target_dh=mesh_ext["dh"],
                                    target_vec=mesh_ext["m_prev_geom"],
                                    source_lookup=_vector_lookup_from_field(fluid["dh"], fluid["d_mesh"]),
                                )
                            else:
                                mesh_ext["m_prev_geom"].nodal_values.fill(0.0)
                            if np.max(np.abs(np.asarray(solid_disp_fluid_lookup.values, dtype=float))) <= 1.0e-18:
                                mesh_ext["m_k"].nodal_values.fill(0.0)
                            else:
                                _solve_structural_similarity_mesh_extension(
                                    prob=mesh_ext,
                                    interface_disp=solid_disp_fluid_lookup,
                                    interface_tag=geometry.interface_tag,
                                    fixed_tags=fixed_mesh_tags,
                                    kratos_local_backend=kratos_local_mesh_backend,
                                    quad_order=max(6, int(quad_order)),
                                    backend=str(backend),
                                    linear_backend=str(linear_backend),
                                )
                            mesh_lookup = _vector_lookup_from_field(mesh_ext["dh"], mesh_ext["m_k"])
                            _transfer_vector_field(target_dh=fluid["dh"], target_vec=fluid["d_mesh"], source_lookup=mesh_lookup)
                            mesh_vel_full_lookup, mesh_accel_full_lookup = _vector_history_full_lookup(
                                dh=fluid["dh"],
                                d_curr=fluid["d_mesh"],
                                d_prev=fluid["d_prev"],
                                v_prev=fluid["w_mesh_prev"],
                                a_prev=fluid["a_mesh_prev"],
                                dt=dt_value,
                                alpha=float(bossak_alpha),
                            )
                            _transfer_vector_field(
                                target_dh=fluid["dh"],
                                target_vec=fluid["w_mesh_k"],
                                source_lookup=mesh_vel_full_lookup,
                            )
                            _transfer_vector_field(
                                target_dh=fluid["dh"],
                                target_vec=fluid["a_mesh_k"],
                                source_lookup=mesh_accel_full_lookup,
                            )
                            mesh_vel_fluid_lookup = _resample_lookup_to_coords(mesh_vel_full_lookup, fluid_iface_coords)
                            mesh_accel_fluid_lookup = _resample_lookup_to_coords(mesh_accel_full_lookup, fluid_iface_coords)
                        if _env_bool("PYCUTFEM_EX2_STAGE_TIMING", False):
                            _log(
                                True,
                                "[timing] "
                                f"step={step} iter={coupling_iter} "
                                f"mesh_stage={time.perf_counter() - t_mesh_stage0:.6f}s",
                            )
                        last_mesh_vel_fluid_lookup = mesh_vel_fluid_lookup
                        last_mesh_accel_fluid_lookup = mesh_accel_fluid_lookup
                        # Kratos keeps the predicted DVMS subscale live across
                        # outer coupling iterations within a time step. The
                        # nonlinear-iteration state (predicted subscale and OSS
                        # projections) must therefore remain live here; only the
                        # accepted-step history fields are re-synced from the
                        # previous accepted step.
                        _restore_fluid_dvms_state_except(
                            fluid.get("dvms_state"),
                            fluid_dvms_prev_step,
                            skip_keys={
                                "predicted_subscale_velocity",
                                "momentum_projection",
                                "mass_projection",
                            },
                        )
                        _update_fluid_dvms_state_from_previous_step(
                            state=fluid["dvms_state"],
                            dh=fluid["dh"],
                            mesh=mesh_f,
                            u_prev=fluid["u_prev"],
                            d_prev=fluid["d_prev"],
                            d_geo=fluid["d_mesh"],
                            backend=str(backend),
                        )
                        _warm_fluid_exact_operator_kernels(
                            prob=fluid,
                            mesh=mesh_f,
                            rho_f=float(setup.material.density),
                            mu_f=mu_f,
                            dt=dt_value,
                            bossak_alpha=float(bossak_alpha),
                            dynamic_tau=float(dynamic_tau),
                            quad_order=quad_order,
                            backend=str(backend),
                            contribution_mode="system",
                        )
        
                        # Kratos does not use an artificial step-1 continuation path for
                        # the fluid stage. Keep the production default on the direct
                        # Kratos-matched solve, with an opt-in escape hatch only for
                        # robustness experiments.
                        use_step1_fluid_continuation = _env_bool(
                            "PYCUTFEM_EX2_STEP1_FLUID_CONTINUATION",
                            False,
                        )
                        continuation_scales = tuple(float(v) for v in retry_policy.fluid_continuation_scales)
                        if not continuation_scales:
                            continuation_scales = (
                                (0.25, 0.5, 1.0)
                                if (bool(use_step1_fluid_continuation) and int(step) == 1)
                                else (1.0,)
                            )
                        t_fluid0 = time.perf_counter()
                        fluid_guess = _snapshot_function_values([fluid["u_k"], fluid["p_k"]])
                        if _env_bool("PYCUTFEM_EX2_ZERO_PRESSURE_GUESS", False):
                            fluid_guess[1].fill(0.0)
                        if _env_bool("PYCUTFEM_EX2_ZERO_VELOCITY_GUESS", False):
                            fluid_guess[0].fill(0.0)
                        fluid_exact_operator = FluidDVMSCondensedLocalSystemOperator(
                            mesh=mesh_f,
                            dh=fluid["dh"],
                            u_k=fluid["u_k"],
                            u_prev=fluid["u_prev"],
                            a_prev=fluid["a_prev"],
                            a_curr=fluid["a_k"],
                            p_k=fluid["p_k"],
                            d_mesh=fluid["d_mesh"],
                            d_prev=fluid["d_prev"],
                            d_prev2=fluid["d_prev2"],
                            mesh_v=fluid["w_mesh_k"],
                            mesh_v_prev=fluid["w_mesh_prev"],
                            mesh_a_prev=fluid["a_mesh_prev"],
                            state=fluid["dvms_state"],
                            rho_f=float(setup.material.density),
                            mu_f=mu_f,
                            dt=dt_value,
                            bossak_alpha=float(bossak_alpha),
                            quadrature_order=quad_order,
                            dynamic_tau=float(dynamic_tau),
                            refresh_predicted_subscale=False,
                            apply_dirichlet_lift=False,
                        )
                        fluid_predictor_operator = _FluidDVMSSolverOperator(
                            state=fluid["dvms_state"],
                            dh=fluid["dh"],
                            mesh=mesh_f,
                            u_k=fluid["u_k"],
                            u_prev=fluid["u_prev"],
                            a_prev=fluid["a_prev"],
                            a_curr=fluid["a_k"],
                            p_k=fluid["p_k"],
                            d_mesh=fluid["d_mesh"],
                            d_prev=fluid["d_prev"],
                            d_prev2=fluid["d_prev2"],
                            mesh_v=fluid["w_mesh_k"],
                            mesh_v_prev=fluid["w_mesh_prev"],
                            mesh_a_prev=fluid["a_mesh_prev"],
                            rho_f=float(setup.material.density),
                            mu_f=mu_f,
                            dt=dt_value,
                            bossak_alpha=float(bossak_alpha),
                            dynamic_tau=float(dynamic_tau),
                            reset_predicted_to_old_on_step_begin=_env_bool(
                                "PYCUTFEM_EX2_RESET_PREDICTED_SUBSCALE_EACH_FLUID_STAGE",
                                False,
                            ),
                        )
                        fluid_acceleration_operator = _FluidBossakAccelerationOperator(
                            u_k=fluid["u_k"],
                            a_k=fluid["a_k"],
                            dt=dt_value,
                            bossak_alpha=float(bossak_alpha),
                        )
                        first_build_old_subscale_mode = _env_str(
                            "PYCUTFEM_EX2_FIRST_BUILD_OLD_SUBSCALE_MODE",
                            "",
                        ).strip().lower()
                        use_first_build_old_subscale = _env_bool(
                            "PYCUTFEM_EX2_FIRST_BUILD_OLD_SUBSCALE",
                            False,
                        )
                        preserve_predicted_subscale = _env_bool(
                            "PYCUTFEM_EX2_PRESERVE_PREDICTED_SUBSCALE_EACH_FLUID_STAGE",
                            False,
                        )
                        # Keep the last converged predicted subscale live by
                        # default. Rebuilding from old-step subscale on the
                        # first coupling iteration is an opt-in experiment,
                        # not the production Kratos-matched path.
                        apply_first_build_old_subscale = False
                        if first_build_old_subscale_mode in {"off", "false", "never", "none"}:
                            apply_first_build_old_subscale = False
                        elif first_build_old_subscale_mode in {"all", "each", "every", "stage"}:
                            apply_first_build_old_subscale = True
                        elif first_build_old_subscale_mode in {"first", "step_first", "iter1"}:
                            apply_first_build_old_subscale = int(coupling_iter) == 1
                        elif bool(use_first_build_old_subscale):
                            apply_first_build_old_subscale = int(coupling_iter) == 1
                        apply_preserve_predicted_subscale = bool(
                            use_exact_fluid_operator
                            and bool(preserve_predicted_subscale)
                            and not bool(apply_first_build_old_subscale)
                        )
                        # Keep the first-assembly skip as an opt-in probe only.
                        # Kratos still refreshes the predictor before each
                        # nonlinear iteration; the carried predicted subscale is
                        # the Newton initial guess inside that refresh, not a
                        # replacement for the refresh itself.
                        if bool(apply_first_build_old_subscale):
                            fluid_predictor_operator.arm_initial_old_subscale_build()
                        elif bool(apply_preserve_predicted_subscale):
                            fluid_predictor_operator.preserve_initial_predicted_subscale()
                        for bc_scale in continuation_scales:
                            fluid_predictor_operator.configure_first_assembly_probe(
                                output_dir=output_dir,
                                step=int(step),
                                coupling_iter=int(coupling_iter),
                                bc_scale=float(bc_scale),
                                dt=float(dt_value),
                                bossak_alpha=float(bossak_alpha),
                            )
                            scaled_iface_velocity = _scaled_lookup(mesh_vel_fluid_lookup, float(bc_scale))
                            scale_value = float(bc_scale)
        
                            def _scaled_inlet_profile(x: float, y: float) -> float:
                                return scale_value * float(inlet_profile(x, y))
        
                            if use_exact_fluid_operator:
                                fluid_res, fluid_jac, fluid_bcs, fluid_bcs_homog = _fluid_zero_local_operator_forms(
                                    prob=fluid,
                                    iface_velocity=scaled_iface_velocity,
                                    inlet_lookup=_scaled_inlet_profile,
                                    interface_tag=geometry.interface_tag,
                                    outlet_tag=geometry.outlet_tag,
                                    walls_tag=geometry.walls_tag,
                                    cylinder_tag=geometry.cylinder_tag,
                                    quad_order=quad_order,
                                )
                                stage_ops = [
                                    fluid_acceleration_operator,
                                    fluid_predictor_operator,
                                    fluid_exact_operator,
                                ]
                                stage_max_newton_iter = int(
                                    retry_policy.fluid_max_newton_iter
                                    if retry_policy.fluid_max_newton_iter is not None
                                    else max(
                                        int(max_newton_iter),
                                        int(
                                            _env_float(
                                                "PYCUTFEM_EX2_EXACT_FLUID_MAX_NEWTON_ITER",
                                                20,
                                            )
                                        ),
                                    )
                                )
                                stage_globalization = (
                                    str(retry_policy.fluid_globalization)
                                    if retry_policy.fluid_globalization is not None
                                    else _env_str(
                                        "PYCUTFEM_EX2_EXACT_FLUID_GLOBALIZATION",
                                        "line_search",
                                    )
                                )
                                stage_line_search = (
                                    bool(retry_policy.fluid_line_search)
                                    if retry_policy.fluid_line_search is not None
                                    else _env_bool(
                                        "PYCUTFEM_EX2_EXACT_FLUID_LINE_SEARCH",
                                        False,
                                    )
                                )
                                stage_ls_fail_hard = (
                                    bool(retry_policy.fluid_ls_fail_hard)
                                    if retry_policy.fluid_ls_fail_hard is not None
                                    else _env_bool(
                                        "PYCUTFEM_EX2_EXACT_FLUID_LS_FAIL_HARD",
                                        True,
                                    )
                                )
                            else:
                                fluid_res, fluid_jac, fluid_bcs, fluid_bcs_homog = _fluid_residual_and_jacobian(
                                    prob=fluid,
                                    rho_f=float(setup.material.density),
                                    mu_f=mu_f,
                                    dt=dt_value,
                                    bossak_alpha=float(bossak_alpha),
                                    dynamic_tau=float(dynamic_tau),
                                    pressure_gauge=float(pressure_gauge),
                                    iface_velocity=scaled_iface_velocity,
                                    inlet_lookup=_scaled_inlet_profile,
                                    interface_tag=geometry.interface_tag,
                                    outlet_tag=geometry.outlet_tag,
                                    walls_tag=geometry.walls_tag,
                                    cylinder_tag=geometry.cylinder_tag,
                                    quad_order=quad_order,
                                )
                                stage_ops = [fluid_predictor_operator]
                                stage_max_newton_iter = max(2, min(int(max_newton_iter), 4))
                                stage_globalization = "line_search_then_trust"
                                stage_line_search = True
                                stage_ls_fail_hard = True
                            fluid["_current_bcs"] = fluid_bcs
                            fluid["_current_bcs_homog"] = fluid_bcs_homog
                            _apply_dirichlet_bcs_to_state(
                                dh=fluid["dh"],
                                field_functions={
                                    str(fluid["u_k"].components[0].field_name): fluid["u_k"].components[0],
                                    str(fluid["u_k"].components[1].field_name): fluid["u_k"].components[1],
                                    str(fluid["p_k"].field_name): fluid["p_k"],
                                },
                                bcs=fluid_bcs,
                            )
                            stage_linear_backend = (
                                str(exact_fluid_linear_backend)
                                if use_exact_fluid_operator
                                else str(linear_backend)
                            )
                            fluid_guess = _snapshot_function_values([fluid["u_k"], fluid["p_k"]])
                            _section(
                                verbose,
                                "[fluid-solver] "
                                f"step={step} coupling_iter={coupling_iter} scale={float(bc_scale):.3f} "
                                f"mode={fluid_operator_mode} globalization={stage_globalization} "
                                f"linear_backend={stage_linear_backend} "
                                f"line_search={stage_line_search} ls_fail_hard={stage_ls_fail_hard} "
                                f"max_newton_iter={stage_max_newton_iter}",
                            )
                            stage_newton_tol = float(newton_tol)
                            if use_exact_fluid_operator:
                                # Kratos' transient monolithic fluid solver uses
                                # MixedGenericCriteria(VELOCITY, PRESSURE), not a
                                # separate raw residual-norm stop. Keep the local
                                # residual gate disabled by default so Newton
                                # acceptance follows the Kratos criterion.
                                stage_newton_tol = float(
                                    _env_float(
                                        "PYCUTFEM_EX2_EXACT_FLUID_RESIDUAL_GATE_TOL",
                                        0.0 if fluid_mixed_solution_criteria else float(exact_fluid_newton_tol),
                                    )
                                )
                            stage_newton_params = NewtonParameters(
                                newton_tol=float(stage_newton_tol),
                                max_newton_iter=int(stage_max_newton_iter),
                                line_search=bool(stage_line_search),
                                ls_fail_hard=bool(stage_ls_fail_hard),
                                globalization=str(stage_globalization),
                                mixed_solution_criteria=fluid_mixed_solution_criteria,
                                mixed_solution_max_residual_factor=float(fluid_mixed_solution_residual_factor),
                                mixed_solution_residual_tol=float(exact_fluid_newton_tol),
                            )
                            stage_lin_params = LinearSolverParameters(backend=str(stage_linear_backend))
                            if use_exact_fluid_operator:
                                stage_solver = _get_or_create_cached_stage_solver(
                                    cache_owner=fluid,
                                    cache_name="_stage_solver_cache",
                                    cache_key=(
                                        "exact",
                                        int(id(fluid_res)),
                                        int(id(fluid_jac)),
                                        int(quad_order),
                                        str(backend),
                                        str(stage_linear_backend),
                                        int(stage_max_newton_iter),
                                        bool(stage_line_search),
                                        bool(stage_ls_fail_hard),
                                        str(stage_globalization),
                                    ),
                                    residual_form=fluid_res,
                                    jacobian_form=fluid_jac,
                                    dof_handler=fluid["dh"],
                                    mixed_element=fluid["me"],
                                    bcs=fluid_bcs,
                                    bcs_homog=fluid_bcs_homog,
                                    newton_params=stage_newton_params,
                                    lin_params=stage_lin_params,
                                    quad_order=quad_order,
                                    backend=str(backend),
                                    operators=stage_ops,
                                    active_fields=("ux", "uy", "p"),
                                )
                            else:
                                stage_solver = NewtonSolver(
                                    residual_form=fluid_res,
                                    jacobian_form=fluid_jac,
                                    dof_handler=fluid["dh"],
                                    mixed_element=fluid["me"],
                                    bcs=fluid_bcs,
                                    bcs_homog=fluid_bcs_homog,
                                    newton_params=stage_newton_params,
                                    lin_params=stage_lin_params,
                                    quad_order=quad_order,
                                    backend=str(backend),
                                    operators=stage_ops,
                                )
                                stage_solver.set_active_fields(["ux", "uy", "p"])
                            _attach_runtime_operator_post_update_hook(
                                solver=stage_solver,
                                operators=stage_ops,
                            )
                            _restore_function_values([fluid["u_prev"], fluid["p_prev"]], fluid_prev_step)
                            _restore_function_values(
                                [fluid["d_prev"], fluid["d_prev2"], fluid["w_mesh_prev"], fluid["a_mesh_prev"]],
                                fluid_mesh_prev_step,
                            )
                            fluid_a_prev_stage = np.asarray(fluid["a_prev"].nodal_values, dtype=float).copy()
                            fluid_acceleration_operator.prime_stage_state(
                                u_prev_snapshot=np.asarray(fluid_prev_step[0], dtype=float),
                                a_prev_snapshot=np.asarray(fluid_a_prev_stage, dtype=float),
                                # Kratos does not refresh ACCELERATION when the
                                # ALE/interface boundary state is synchronized
                                # before the fluid solve. The first nonlinear
                                # assembly sees the carried value from the
                                # previous fluid solve; subsequent assemblies
                                # are refreshed after Newton velocity updates.
                                preserve_seed_on_first_assembly=_env_bool(
                                    "PYCUTFEM_EX2_PRESERVE_FLUID_ACCELERATION_SEED",
                                    True,
                                ),
                            )
                            if use_exact_fluid_operator:
                                _maybe_dump_exact_fluid_probe(
                                    output_dir=output_dir,
                                    step=int(step),
                                    coupling_iter=int(coupling_iter),
                                    stage_label="pre_fluid_solve",
                                    bc_scale=float(bc_scale),
                                    dt=float(dt_value),
                                    bossak_alpha=float(bossak_alpha),
                                    fluid=fluid,
                                    reaction_point_load_lookup=None,
                                    reaction_solid_load_lookup=None,
                                )
                            fluid_newton_accepted_after_maxiter = False
                            fluid_hrom_used = False
                            fluid_hrom_trial_used = False
                            fluid_hrom_trial_load_rel_error = float("nan")
                            fluid_hrom_trial_disabled_by_monitor = False
                            hrom_trial_return_load_values: np.ndarray | None = None
                            fluid_hrom_fallback_reason = ""
                            fluid_hrom_info: dict[str, object] | None = None
                            fluid_hrom_interface_trust_alpha = 1.0
                            fluid_hrom_interface_load_rel = float("nan")
                            fluid_hrom_interface_step_ratio = float("nan")
                            fluid_hrom_interface_corrected = False
                            fluid_hrom_interface_rejected = False
                            fluid_hrom_interface_reason = ""
                            hrom_reaction_point_load_lookup: CoordinateLookup | None = None
                            hrom_reaction_solid_load_lookup: CoordinateLookup | None = None
                            hrom_stress_point_load_lookup: CoordinateLookup | None = None
                            hrom_iteration_gate = bool(
                                use_sampled_lspg_hybrid_fluid
                                and sampled_lspg_hybrid_model is not None
                                and int(coupling_iter) >= int(sampled_lspg_hybrid_switch_iter)
                                and float(bc_scale) == float(continuation_scales[-1])
                            )
                            hrom_previous_load_high_ok = bool(
                                int(coupling_iter) > 1
                                and np.isfinite(float(last_load_rel))
                                and float(last_load_rel) <= float(fluid_hrom_max_previous_load_rel_value)
                            )
                            hrom_previous_load_low_ok = bool(
                                int(coupling_iter) > 1
                                and np.isfinite(float(last_load_rel))
                                and float(last_load_rel) >= float(fluid_hrom_min_previous_load_rel_value)
                            )
                            hrom_previous_disp_ok = bool(
                                int(coupling_iter) > 1
                                and np.isfinite(float(last_disp_rel))
                                and float(last_disp_rel) <= float(fluid_hrom_max_previous_disp_rel)
                            )
                            hrom_consecutive_ok = bool(
                                int(fluid_hrom_max_consecutive_stages_value) <= 0
                                or int(consecutive_hrom_stages) < int(fluid_hrom_max_consecutive_stages_value)
                            )
                            hrom_disabled_ok = int(step) > int(fluid_hrom_disabled_until_step)
                            use_hrom_stage = bool(
                                hrom_iteration_gate
                                and hrom_previous_load_high_ok
                                and hrom_previous_load_low_ok
                                and hrom_previous_disp_ok
                                and hrom_consecutive_ok
                                and hrom_disabled_ok
                            )
                            if bool(hrom_iteration_gate) and not bool(use_hrom_stage):
                                if not (bool(hrom_previous_load_high_ok) and bool(hrom_previous_load_low_ok)):
                                    sampled_lspg_hybrid_load_gate_skips += 1
                                if not bool(hrom_previous_disp_ok):
                                    sampled_lspg_hybrid_disp_gate_skips += 1
                                if not bool(hrom_consecutive_ok):
                                    sampled_lspg_hybrid_consecutive_gate_skips += 1
                                if not bool(hrom_disabled_ok):
                                    sampled_lspg_hybrid_disabled_gate_skips += 1
                                _log(
                                    verbose,
                                    "[fluid-hrom] "
                                    f"step={step} coupling_iter={coupling_iter} gate_skip=1 "
                                    f"previous_load_rel={float(last_load_rel):.3e} "
                                    f"load_gate=[{float(fluid_hrom_min_previous_load_rel_value):.3e}, "
                                    f"{float(fluid_hrom_max_previous_load_rel_value):.3e}] "
                                    f"previous_disp_rel={float(last_disp_rel):.3e} "
                                    f"disp_gate={float(fluid_hrom_max_previous_disp_rel):.3e} "
                                    f"consecutive={int(consecutive_hrom_stages)} "
                                    f"max_consecutive={int(fluid_hrom_max_consecutive_stages_value)} "
                                    f"disabled_until_step={int(fluid_hrom_disabled_until_step)}",
                                )
                            if bool(use_hrom_stage):
                                hrom_history = _snapshot_fluid_dvms_state(fluid.get("dvms_state"))
                                hrom_state_guess = [np.asarray(item, dtype=float).copy() for item in fluid_guess]
                                hrom_a_guess = np.asarray(fluid["a_k"].nodal_values, dtype=float).copy()
                                try:
                                    fluid_hrom_info = _solve_sampled_lspg_hybrid_fluid_stage(
                                        model=sampled_lspg_hybrid_model,
                                        prob=fluid,
                                        mesh=mesh_f,
                                        rho_f=float(setup.material.density),
                                        mu_f=mu_f,
                                        dt=dt_value,
                                        bossak_alpha=float(bossak_alpha),
                                        dynamic_tau=float(dynamic_tau),
                                        quad_order=quad_order,
                                        backend=str(backend),
                                        fluid_prev_step=fluid_prev_step,
                                        fluid_a_prev_stage=fluid_a_prev_stage,
                                        max_iterations=int(sampled_lspg_hybrid_max_iterations),
                                        residual_tol=float(sampled_lspg_hybrid_residual_tol),
                                        line_search=bool(sampled_lspg_hybrid_model.line_search),
                                    )
                                    max_hrom_residual = _env_float(
                                        "PYCUTFEM_EX2_HROM_MAX_ESTIMATED_RESIDUAL_NORM",
                                        float("inf"),
                                    )
                                    if float(fluid_hrom_info["estimated_residual_norm"]) > float(max_hrom_residual):
                                        raise RuntimeError(
                                            "sampled-LSPG residual monitor failed: "
                                            f"{float(fluid_hrom_info['estimated_residual_norm']):.6e} "
                                            f"> {float(max_hrom_residual):.6e}"
                                        )
                                    fluid_hrom_used = True
                                    if bool(fluid_hrom_trial_exact_correct):
                                        if load_transfer_value == "reaction":
                                            trial_reaction_point_lookup = _fluid_interface_reaction_loads(
                                                prob=fluid,
                                                rho_f=float(setup.material.density),
                                                mu_f=mu_f,
                                                dt=dt_value,
                                                quad_order=quad_order,
                                                bossak_alpha=float(bossak_alpha),
                                                dynamic_tau=float(dynamic_tau),
                                                interface_tag=geometry.interface_tag,
                                                backend=str(backend),
                                                contribution_mode="system",
                                                refresh_state=False,
                                            )
                                            trial_reaction_solid_load_lookup = _resample_lookup_to_coords(
                                                _negate_lookup(trial_reaction_point_lookup),
                                                solid_iface_coords,
                                            )
                                            hrom_trial_return_load_values = np.asarray(
                                                (
                                                    trial_reaction_point_lookup
                                                    if bool(accelerate_on_fluid_load)
                                                    else trial_reaction_solid_load_lookup
                                                ).values,
                                                dtype=float,
                                            ).copy()
                                        elif load_transfer_value == "stress":
                                            trial_stress_point_load_lookup = _fluid_interface_point_loads_on_solid(
                                                fluid_dh=fluid["dh"],
                                                fluid_mesh=mesh_f,
                                                solid_mesh=mesh_s,
                                                u=fluid["u_k"],
                                                p=fluid["p_k"],
                                                d_mesh=fluid["d_mesh"],
                                                solid_iface_coords=solid_iface_coords,
                                                interface_tag=geometry.interface_tag,
                                                mu_f=mu_f,
                                                quad_order=quad_order,
                                            )
                                            hrom_trial_return_load_values = np.asarray(
                                                trial_stress_point_load_lookup.values,
                                                dtype=float,
                                            ).copy()
                                        hrom_corrector_guess = _snapshot_function_values([fluid["u_k"], fluid["p_k"]])
                                        hrom_corrector_a = np.asarray(fluid["a_k"].nodal_values, dtype=float).copy()
                                        _restore_fluid_dvms_state(fluid.get("dvms_state"), hrom_history)
                                        _restore_function_values([fluid["u_k"], fluid["p_k"]], hrom_corrector_guess)
                                        fluid["a_k"].nodal_values[:] = hrom_corrector_a
                                        fluid_guess = hrom_corrector_guess
                                        fluid_hrom_trial_used = True
                                        fluid_hrom_used = False
                                    if bool(fluid_hrom_used) and fluid_hrom_interface_trust_value != "none":
                                        if load_transfer_value == "reaction":
                                            hrom_reaction_point_load_lookup = _fluid_interface_reaction_loads(
                                                prob=fluid,
                                                rho_f=float(setup.material.density),
                                                mu_f=mu_f,
                                                dt=dt_value,
                                                quad_order=quad_order,
                                                bossak_alpha=float(bossak_alpha),
                                                dynamic_tau=float(dynamic_tau),
                                                interface_tag=geometry.interface_tag,
                                                backend=str(backend),
                                                contribution_mode="system",
                                                refresh_state=False,
                                            )
                                            hrom_reaction_solid_load_lookup = _resample_lookup_to_coords(
                                                _negate_lookup(hrom_reaction_point_load_lookup),
                                                solid_iface_coords,
                                            )
                                            hrom_accel_lookup = (
                                                hrom_reaction_point_load_lookup
                                                if bool(accelerate_on_fluid_load)
                                                else hrom_reaction_solid_load_lookup
                                            )
                                        elif load_transfer_value == "stress":
                                            hrom_stress_point_load_lookup = _fluid_interface_point_loads_on_solid(
                                                fluid_dh=fluid["dh"],
                                                fluid_mesh=mesh_f,
                                                solid_mesh=mesh_s,
                                                u=fluid["u_k"],
                                                p=fluid["p_k"],
                                                d_mesh=fluid["d_mesh"],
                                                solid_iface_coords=solid_iface_coords,
                                                interface_tag=geometry.interface_tag,
                                                mu_f=mu_f,
                                                quad_order=quad_order,
                                            )
                                            hrom_accel_lookup = hrom_stress_point_load_lookup
                                        else:
                                            raise ValueError(f"Unsupported load_transfer={load_transfer!r}")
                                        current_accel_values = _resample_lookup_to_coords(
                                            current_load_lookup,
                                            np.asarray(hrom_accel_lookup.coords, dtype=float),
                                        ).values
                                        trust_result = _fluid_hrom_interface_trust_region(
                                            current_values=current_accel_values,
                                            proposed_values=np.asarray(hrom_accel_lookup.values, dtype=float),
                                            previous_load_abs=float(last_load_abs),
                                            mode=fluid_hrom_interface_trust_value,
                                            max_step_ratio=float(fluid_hrom_interface_max_step_ratio_value),
                                            max_load_rel=float(fluid_hrom_interface_max_load_rel_value),
                                            min_correction_alpha=float(fluid_hrom_interface_min_correction_alpha_value),
                                        )
                                        fluid_hrom_interface_trust_alpha = float(trust_result.alpha)
                                        fluid_hrom_interface_load_rel = float(trust_result.update_rel)
                                        fluid_hrom_interface_step_ratio = float(trust_result.step_ratio)
                                        fluid_hrom_interface_reason = str(trust_result.reason)
                                        if not bool(trust_result.accepted):
                                            fluid_hrom_interface_rejected = True
                                            sampled_lspg_hybrid_interface_reject_count += 1
                                            raise RuntimeError(
                                                "sampled-LSPG interface trust monitor failed: "
                                                f"reason={trust_result.reason} "
                                                f"load_rel={float(trust_result.update_rel):.6e} "
                                                f"step_ratio={float(trust_result.step_ratio):.6e} "
                                                f"alpha={float(trust_result.alpha):.6e}"
                                            )
                                        if bool(trust_result.corrected):
                                            fluid_hrom_interface_corrected = True
                                            sampled_lspg_hybrid_interface_correction_count += 1
                                            corrected_lookup = CoordinateLookup(
                                                np.asarray(hrom_accel_lookup.coords, dtype=float),
                                                np.asarray(trust_result.values, dtype=float),
                                                dim=2,
                                            )
                                            if load_transfer_value == "reaction" and bool(accelerate_on_fluid_load):
                                                hrom_reaction_point_load_lookup = corrected_lookup
                                                hrom_reaction_solid_load_lookup = _resample_lookup_to_coords(
                                                    _negate_lookup(hrom_reaction_point_load_lookup),
                                                    solid_iface_coords,
                                                )
                                            elif load_transfer_value == "reaction":
                                                hrom_reaction_solid_load_lookup = corrected_lookup
                                            else:
                                                hrom_stress_point_load_lookup = corrected_lookup
                                            _log(
                                                verbose,
                                                "[fluid-hrom] "
                                                f"step={step} coupling_iter={coupling_iter} "
                                                "interface_load_clipped=1 "
                                                f"alpha={float(trust_result.alpha):.3e} "
                                                f"load_rel={float(trust_result.update_rel):.3e} "
                                                f"step_ratio={float(trust_result.step_ratio):.3e}",
                                            )
                                except Exception as exc:
                                    fluid_hrom_fallback_reason = str(exc)
                                    fluid_hrom_used = False
                                    if not bool(fluid_hrom_fallback_exact):
                                        raise
                                    _restore_fluid_dvms_state(fluid.get("dvms_state"), hrom_history)
                                    _restore_function_values([fluid["u_k"], fluid["p_k"]], hrom_state_guess)
                                    fluid["a_k"].nodal_values[:] = hrom_a_guess
                                    _log(
                                        verbose,
                                        "[fluid-hrom] "
                                        f"step={step} coupling_iter={coupling_iter} fallback_to_exact=1 "
                                        f"reason={fluid_hrom_fallback_reason}",
                                    )
                            if not bool(fluid_hrom_used):
                                try:
                                    stage_solver.solve_time_interval(
                                        functions=[fluid["u_k"], fluid["p_k"]],
                                        prev_functions=[fluid["u_prev"], fluid["p_prev"]],
                                        aux_functions={
                                            "a_prev": fluid["a_prev"],
                                            "a_k": fluid["a_k"],
                                            "d_mesh": fluid["d_mesh"],
                                            "d_prev": fluid["d_prev"],
                                            "d_prev2": fluid["d_prev2"],
                                        },
                                        time_params=TimeStepperParameters(
                                            dt=dt_value,
                                            max_steps=1,
                                            final_time=dt_value,
                                            stop_on_steady=False,
                                            step_initial_guess_callback=_guess_callback_from_snapshots_with_dirichlet(
                                                snapshots=fluid_guess,
                                                dh=fluid["dh"],
                                                bcs=fluid_bcs,
                                            ),
                                        ),
                                    )
                                except Exception as exc:
                                    if use_exact_fluid_operator:
                                        _maybe_dump_exact_fluid_probe(
                                            output_dir=output_dir,
                                            step=int(step),
                                            coupling_iter=int(coupling_iter),
                                            stage_label="failed_fluid_solve",
                                            bc_scale=float(bc_scale),
                                            dt=float(dt_value),
                                            bossak_alpha=float(bossak_alpha),
                                            fluid=fluid,
                                            reaction_point_load_lookup=None,
                                            reaction_solid_load_lookup=None,
                                        )
                                    accept_maxiter = _env_bool(
                                        "PYCUTFEM_EX2_EXACT_FLUID_ACCEPT_MAXITER",
                                        True,
                                    )
                                    if (
                                        bool(use_exact_fluid_operator)
                                        and bool(accept_maxiter)
                                        and _is_newton_maxiter_nonconvergence(exc)
                                    ):
                                        fluid_newton_accepted_after_maxiter = True
                                        _log(
                                            verbose,
                                            "[fluid-stage] "
                                            f"step={step} coupling_iter={coupling_iter} "
                                            "Newton reached the iteration cap; continuing with the last iterate "
                                            "to match Kratos NavierStokesMonolithicSolver warning semantics.",
                                        )
                                    else:
                                        raise
                            if bool(use_sampled_lspg_hybrid_fluid):
                                if bool(fluid_hrom_trial_used):
                                    sampled_lspg_hybrid_trial_stage_count += 1
                                    sampled_lspg_hybrid_exact_correction_count += 1
                                if bool(fluid_hrom_used):
                                    sampled_lspg_hybrid_stage_count += 1
                                else:
                                    sampled_lspg_hybrid_exact_stage_count += 1
                                if fluid_hrom_fallback_reason:
                                    sampled_lspg_hybrid_fallback_count += 1
                            # Kratos stores the solved-step ACCELERATION field as the
                            # Bossak recurrence against the previous accepted-step
                            # history, then carries that hidden state into the next
                            # outer coupling iteration.
                            fluid_bossak = _bossak_coefficients(
                                alpha=float(bossak_alpha),
                                dt=max(float(dt_value), 1.0e-14),
                            )
                            fluid["a_k"].nodal_values[:] = (
                                float(fluid_bossak["ma0"])
                                * (fluid["u_k"].nodal_values[:] - np.asarray(fluid_prev_step[0], dtype=float))
                                + float(fluid_bossak["ma2"]) * fluid_a_prev_stage
                            )
                            fluid_guess = _snapshot_function_values([fluid["u_k"], fluid["p_k"]])
                            _log(
                                verbose,
                                "[fluid-stage] "
                                f"scale={float(bc_scale):.3f} "
                                f"mode={fluid_operator_mode} "
                                f"hrom_used={int(fluid_hrom_used)} "
                                f"hrom_trial={int(fluid_hrom_trial_used)} "
                                f"u_max={_field_abs_max(fluid['u_k']):.3e} "
                                f"p_max={_field_abs_max(fluid['p_k']):.3e} "
                                f"accepted_after_maxiter={int(fluid_newton_accepted_after_maxiter)}"
                                + (
                                    " "
                                    f"hrom_iterations={int(fluid_hrom_info['iterations'])} "
                                    f"hrom_residual={float(fluid_hrom_info['estimated_residual_norm']):.3e} "
                                    f"interface_alpha={float(fluid_hrom_interface_trust_alpha):.3e} "
                                    f"interface_load_rel={float(fluid_hrom_interface_load_rel):.3e}"
                                    if (bool(fluid_hrom_used) or bool(fluid_hrom_trial_used)) and fluid_hrom_info is not None
                                    else (
                                        f" hrom_fallback_reason={fluid_hrom_fallback_reason}"
                                        if fluid_hrom_fallback_reason
                                        else ""
                                    )
                                ),
                            )
                        _restore_function_values([fluid["u_prev"], fluid["p_prev"]], fluid_prev_step)
                        _restore_function_values(
                            [fluid["d_prev"], fluid["d_prev2"], fluid["w_mesh_prev"], fluid["a_mesh_prev"]],
                            fluid_mesh_prev_step,
                        )
                        fluid_elapsed = time.perf_counter() - t_fluid0
                        fluid_times.append(float(fluid_elapsed))
                        if bool(fluid_hrom_used):
                            consecutive_hrom_stages += 1
                        else:
                            consecutive_hrom_stages = 0
        
                        if load_transfer_value == "reaction" or bool(monitor_interface_loads):
                            t_reaction0 = time.perf_counter()
                            if bool(fluid_hrom_used) and hrom_reaction_point_load_lookup is not None:
                                reaction_point_load_lookup = hrom_reaction_point_load_lookup
                                reaction_solid_load_lookup = (
                                    hrom_reaction_solid_load_lookup
                                    if hrom_reaction_solid_load_lookup is not None
                                    else _resample_lookup_to_coords(
                                        _negate_lookup(reaction_point_load_lookup),
                                        solid_iface_coords,
                                    )
                                )
                            else:
                                reaction_point_load_lookup = _fluid_interface_reaction_loads(
                                    prob=fluid,
                                    rho_f=float(setup.material.density),
                                    mu_f=mu_f,
                                    dt=dt_value,
                                    quad_order=quad_order,
                                    bossak_alpha=float(bossak_alpha),
                                    dynamic_tau=float(dynamic_tau),
                                    interface_tag=geometry.interface_tag,
                                    backend=str(backend),
                                    contribution_mode="system",
                                    refresh_state=False,
                                )
                                reaction_solid_load_lookup = _resample_lookup_to_coords(
                                    _negate_lookup(reaction_point_load_lookup),
                                    solid_iface_coords,
                                )
                            if _env_bool("PYCUTFEM_EX2_STAGE_TIMING", False):
                                _log(
                                    True,
                                    "[timing] "
                                    f"step={step} iter={coupling_iter} "
                                    f"reaction_stage={time.perf_counter() - t_reaction0:.6f}s",
                                )
                        if bool(monitor_stress_interface_loads):
                            if bool(fluid_hrom_used) and hrom_stress_point_load_lookup is not None:
                                stress_point_load_lookup = hrom_stress_point_load_lookup
                            else:
                                stress_point_load_lookup = _fluid_interface_point_loads_on_solid(
                                    fluid_dh=fluid["dh"],
                                    fluid_mesh=mesh_f,
                                    solid_mesh=mesh_s,
                                    u=fluid["u_k"],
                                    p=fluid["p_k"],
                                    d_mesh=fluid["d_mesh"],
                                    solid_iface_coords=solid_iface_coords,
                                    interface_tag=geometry.interface_tag,
                                    mu_f=mu_f,
                                    quad_order=quad_order,
                                )
                        if use_exact_fluid_operator:
                            _maybe_dump_exact_fluid_probe(
                                output_dir=output_dir,
                                step=int(step),
                                coupling_iter=int(coupling_iter),
                                stage_label="post_fluid_solve",
                                bc_scale=float(continuation_scales[-1]),
                                dt=float(dt_value),
                                bossak_alpha=float(bossak_alpha),
                                fluid=fluid,
                                reaction_point_load_lookup=reaction_point_load_lookup,
                                reaction_solid_load_lookup=reaction_solid_load_lookup,
                            )
                    if load_transfer_value == "reaction":
                        if reaction_solid_load_lookup is None:
                            raise RuntimeError("Reaction load lookup was not computed.")
                        fluid_point_load_lookup = reaction_solid_load_lookup
                    elif load_transfer_value == "stress":
                        if stress_point_load_lookup is None:
                            raise RuntimeError("Stress load lookup was not computed.")
                        fluid_point_load_lookup = stress_point_load_lookup
                    else:
                        raise ValueError(f"Unsupported load_transfer={load_transfer!r}")
                    if accelerate_on_fluid_load:
                        returned_accel_load_lookup = reaction_point_load_lookup
                    else:
                        returned_accel_load_lookup = fluid_point_load_lookup
                    if bool(fluid_hrom_trial_used) and hrom_trial_return_load_values is not None:
                        if returned_accel_load_lookup is None:
                            raise RuntimeError("HROM trial monitor requested an exact corrected load, but none was computed.")
                        _hrom_trial_abs, fluid_hrom_trial_load_rel_error = _relative_change(
                            np.asarray(returned_accel_load_lookup.values, dtype=float),
                            np.asarray(hrom_trial_return_load_values, dtype=float),
                        )
                        if float(fluid_hrom_trial_load_rel_error) > float(fluid_hrom_max_trial_load_rel_error_value):
                            sampled_lspg_hybrid_trial_monitor_failures += 1
                            fluid_hrom_trial_disabled_by_monitor = True
                            fluid_hrom_disabled_until_step = max(
                                int(fluid_hrom_disabled_until_step),
                                int(step) + int(fluid_hrom_disable_steps_after_trial_failure_value),
                            )
                            _log(
                                verbose,
                                "[fluid-hrom] "
                                f"step={step} coupling_iter={coupling_iter} trial_monitor_failed=1 "
                                f"trial_load_rel_error={float(fluid_hrom_trial_load_rel_error):.3e} "
                                f"gate={float(fluid_hrom_max_trial_load_rel_error_value):.3e} "
                                f"disabled_until_step={int(fluid_hrom_disabled_until_step)}",
                            )
                    returned_load_lookup = _resample_lookup_to_coords(fluid_point_load_lookup, solid_iface_coords)
                    last_returned_load_lookup = returned_load_lookup
                    mesh_vel_solid_lookup = _resample_lookup_to_coords(mesh_vel_fluid_lookup, solid_iface_coords)
                    interface_velocity_snapshot = np.asarray(mesh_vel_solid_lookup.values, dtype=float).reshape(-1)
                    interface_disp_snapshot = np.asarray(solid_disp_solid_lookup.values, dtype=float).reshape(-1)
                    load_snapshot = np.asarray(returned_load_lookup.values, dtype=float).reshape(-1)
                    traction_snapshot = _interface_traction_from_load(
                        solid_interface_mass,
                        np.asarray(returned_load_lookup.values, dtype=float),
                    ).reshape(-1)
                    accel_guess_vals = np.asarray(current_load_lookup.values, dtype=float)
                    accel_return_vals = np.asarray(returned_accel_load_lookup.values, dtype=float)
                    load_residual = accel_return_vals - accel_guess_vals
                    disp_abs, disp_rel = _relative_change(solid_disp_solid_lookup.values, prev_disp_iter_vals)
                    load_abs, load_rel = _relative_change(accel_return_vals, accel_guess_vals)
                    last_disp_abs = disp_abs
                    last_disp_rel = disp_rel
                    last_load_abs = load_abs
                    last_load_rel = load_rel
                    omega_force = float(active_force_relaxation)
                    if str(active_force_update).lower() == "aitken":
                        omega_force = _aitken_relaxation_factor(
                            omega_prev=float(last_force_omega),
                            residual_prev=prev_force_residual,
                            residual_curr=load_residual,
                            omega_min=float(force_relaxation_min),
                            omega_max=float(force_relaxation_max),
                        )
                    else:
                        omega_force = float(
                            np.clip(float(active_force_relaxation), float(force_relaxation_min), float(force_relaxation_max))
                        )

                    disp_max = float(np.max(np.linalg.norm(np.asarray(solid_disp_solid_lookup.values, dtype=float), axis=1)))
                    load_guess_max = float(np.max(np.linalg.norm(accel_guess_vals, axis=1)))
                    load_return_max = float(np.max(np.linalg.norm(accel_return_vals, axis=1)))
                    row = {
                        "step": int(step),
                        "time_s": float(t_now),
                        "attempt": int(attempt_index),
                        "coupling_iter": int(coupling_iter),
                        "disp_abs": float(disp_abs),
                        "disp_rel": float(disp_rel),
                        "load_abs": float(load_abs),
                        "load_rel": float(load_rel),
                        "solid_time_s": float(solid_elapsed),
                        "fluid_time_s": float(fluid_elapsed),
                        "disp_max": disp_max,
                        "load_guess_max": load_guess_max,
                        "load_return_max": load_return_max,
                        "force_update_active": str(active_force_update),
                        "force_omega": float(omega_force),
                    }
                    keep_snapshot = snapshot_mode == "all"
                    disp_converged = bool((disp_abs <= coupling_abs_tol) or (disp_rel <= coupling_rel_tol))
                    load_converged = bool((load_abs <= coupling_abs_tol) or (load_rel <= coupling_rel_tol))
                    step_converged = bool(disp_converged and load_converged)
                    kratos_coupling_tol = 5.0e-3
                    kratos_disp_converged = bool((disp_abs <= kratos_coupling_tol) or (disp_rel <= kratos_coupling_tol))
                    kratos_load_converged = bool((load_abs <= kratos_coupling_tol) or (load_rel <= kratos_coupling_tol))
                    kratos_step_converged = bool(kratos_disp_converged and kratos_load_converged)
                    hrom_exact_accept_forced = bool(
                        step_converged and bool(fluid_hrom_used) and bool(fluid_hrom_require_exact_accept)
                    )
                    if bool(hrom_exact_accept_forced):
                        sampled_lspg_hybrid_exact_accept_forced_count += 1
                        step_converged = False
                        _log(
                            verbose,
                            "[fluid-hrom] "
                            f"step={step} coupling_iter={coupling_iter} exact_accept_required=1 "
                            f"load_rel={float(load_rel):.3e} disp_rel={float(disp_rel):.3e}",
                        )
                    row["fluid_hrom_used"] = bool(fluid_hrom_used)
                    row["fluid_hrom_trial_used"] = bool(fluid_hrom_trial_used)
                    row["fluid_hrom_trial_load_rel_error"] = float(fluid_hrom_trial_load_rel_error)
                    row["fluid_hrom_disabled_by_trial_monitor"] = bool(fluid_hrom_trial_disabled_by_monitor)
                    row["fluid_hrom_exact_accept_forced"] = bool(hrom_exact_accept_forced)
                    row["fluid_hrom_interface_trust_alpha"] = float(fluid_hrom_interface_trust_alpha)
                    row["fluid_hrom_interface_load_rel"] = float(fluid_hrom_interface_load_rel)
                    row["fluid_hrom_interface_step_ratio"] = float(fluid_hrom_interface_step_ratio)
                    row["fluid_hrom_interface_corrected"] = bool(fluid_hrom_interface_corrected)
                    row["fluid_hrom_interface_rejected"] = bool(fluid_hrom_interface_rejected)
                    row["fluid_hrom_interface_reason"] = str(fluid_hrom_interface_reason)
                    row["strict_converged"] = bool(step_converged)
                    row["kratos_disp_converged_5e-3"] = bool(kratos_disp_converged)
                    row["kratos_load_converged_5e-3"] = bool(kratos_load_converged)
                    row["kratos_step_converged_5e-3"] = bool(kratos_step_converged)
                    step_rows.append(row)
                    if snapshot_mode == "converged" and step_converged:
                        keep_snapshot = True
                    if keep_snapshot:
                        if disp_snapshot is None:
                            if nirb_prediction is None or nirb_solid_predictor is None:
                                disp_snapshot = _flatten_vector_snapshot(solid["dh"], solid["d_k"])
                            else:
                                disp_snapshot = _assign_nirb_full_displacement(
                                    predictor=nirb_solid_predictor,
                                    prediction=nirb_prediction,
                                    solid_displacement=solid["d_k"],
                                )
                        load_guess_snapshots.append(np.asarray(load_guess_vals, dtype=float).reshape(-1))
                        load_return_snapshots.append(load_snapshot)
                        fluid_load_guess_snapshots.append(np.asarray(accel_guess_vals, dtype=float).reshape(-1))
                        fluid_load_return_snapshots.append(np.asarray(accel_return_vals, dtype=float).reshape(-1))
                        disp_snapshots.append(disp_snapshot)
                        load_snapshots.append(load_snapshot)
                        interface_disp_snapshots.append(interface_disp_snapshot)
                        interface_velocity_snapshots.append(interface_velocity_snapshot)
                        interface_traction_snapshots.append(traction_snapshot)
                        if bool(monitor_interface_loads):
                            reaction_values = (
                                np.asarray(reaction_solid_load_lookup.values, dtype=float).reshape(-1)
                                if reaction_solid_load_lookup is not None
                                else np.zeros_like(load_snapshot)
                            )
                            stress_values = (
                                np.asarray(stress_point_load_lookup.values, dtype=float).reshape(-1)
                                if stress_point_load_lookup is not None
                                else np.zeros_like(load_snapshot)
                            )
                            reaction_load_snapshots.append(reaction_values)
                            stress_load_snapshots.append(stress_values)
                        snapshot_rows.append(
                            {
                                "step": int(step),
                                "time_s": float(t_now),
                                "coupling_iter": int(coupling_iter),
                                "converged": bool(step_converged),
                            }
                        )

                    _log(
                        verbose,
                        "[fixed-point] "
                        f"step={step} iter={coupling_iter} "
                        f"disp_abs={disp_abs:.3e} disp_rel={disp_rel:.3e} "
                        f"load_abs={load_abs:.3e} load_rel={load_rel:.3e} "
                        f"disp_max={disp_max:.3e} load_guess_max={load_guess_max:.3e} "
                        f"load_return_max={load_return_max:.3e} omega={omega_force:.3e} "
                        f"strict_ok={int(step_converged)} kratos_5e-3_ok={int(kratos_step_converged)}",
                    )
                    _maybe_dump_coupling_update_probe(
                        output_dir=output_dir,
                        step=int(step),
                        coupling_iter=int(coupling_iter),
                        stage_label="pre_update",
                        current_load_lookup=current_load_lookup,
                        returned_accel_load_lookup=returned_accel_load_lookup,
                        next_load_lookup=None,
                        load_guess_history=load_guess_history,
                        load_return_history=load_return_history,
                        iqn_old_dr_mats=list(iqn_old_dr_mats),
                        iqn_old_dg_mats=list(iqn_old_dg_mats),
                        omega_force=float(omega_force),
                        active_force_update=str(active_force_update),
                        force_iteration_horizon=int(force_iteration_horizon),
                        force_regularization=float(force_regularization),
                        accel_backend=str(_env_str("PYCUTFEM_EX2_IQN_BACKEND", "python")).strip().lower(),
                        v_new=None,
                        w_new=None,
                    )
                    current_load_lookup, _, load_update_debug = _advance_coupling_load_guess(
                        step_converged=bool(step_converged),
                        active_force_update=str(active_force_update),
                        iface_coords=fluid_iface_coords if accelerate_on_fluid_load else solid_iface_coords,
                        load_guess_vals=accel_guess_vals,
                        returned_load_vals=accel_return_vals,
                        load_guess_history=load_guess_history,
                        load_return_history=load_return_history,
                        iqn_old_dr_mats=list(iqn_old_dr_mats),
                        iqn_old_dg_mats=list(iqn_old_dg_mats),
                        omega_force=float(omega_force),
                        force_iteration_horizon=int(force_iteration_horizon),
                        force_regularization=float(force_regularization),
                        include_debug=True,
                    )
                    _maybe_dump_coupling_update_probe(
                        output_dir=output_dir,
                        step=int(step),
                        coupling_iter=int(coupling_iter),
                        stage_label="post_update",
                        current_load_lookup=CoordinateLookup(
                            np.asarray(
                                fluid_iface_coords if accelerate_on_fluid_load else solid_iface_coords,
                                dtype=float,
                            ),
                            np.asarray(accel_guess_vals, dtype=float),
                            dim=2,
                        ),
                        returned_accel_load_lookup=returned_accel_load_lookup,
                        next_load_lookup=current_load_lookup,
                        load_guess_history=load_guess_history,
                        load_return_history=load_return_history,
                        iqn_old_dr_mats=list(iqn_old_dr_mats),
                        iqn_old_dg_mats=list(iqn_old_dg_mats),
                        omega_force=float(omega_force),
                        active_force_update=str(active_force_update),
                        force_iteration_horizon=int(force_iteration_horizon),
                        force_regularization=float(force_regularization),
                        accel_backend=str(load_update_debug.get("backend", "python")),
                        v_new=load_update_debug.get("v_new"),
                        w_new=load_update_debug.get("w_new"),
                    )
                    if step_converged:
                        break

                    prev_disp_iter_vals = np.asarray(solid_disp_solid_lookup.values, dtype=float).copy()
                    prev_force_residual = np.asarray(load_residual, dtype=float).copy()
                    last_force_omega = float(omega_force)

                if not step_converged:
                    raise RuntimeError(
                        "FSI fixed-point did not converge for step "
                        f"{int(step)} after {int(coupling_iter)} iterations "
                        f"(disp_rel={float(last_disp_rel):.6e}, load_rel={float(last_load_rel):.6e})."
                    )
            except Exception as exc:
                _truncate_step_progress(
                    marker=step_marker,
                    disp_snapshots=disp_snapshots,
                    load_snapshots=load_snapshots,
                    load_guess_snapshots=load_guess_snapshots,
                    load_return_snapshots=load_return_snapshots,
                    fluid_load_guess_snapshots=fluid_load_guess_snapshots,
                    fluid_load_return_snapshots=fluid_load_return_snapshots,
                    interface_disp_snapshots=interface_disp_snapshots,
                    interface_velocity_snapshots=interface_velocity_snapshots,
                    interface_traction_snapshots=interface_traction_snapshots,
                    reaction_load_snapshots=reaction_load_snapshots,
                    stress_load_snapshots=stress_load_snapshots,
                    snapshot_rows=snapshot_rows,
                    step_rows=step_rows,
                    fluid_times=fluid_times,
                    structure_times=structure_times,
                    increment_times=increment_times,
                )
                if attempt_index < len(retry_policies):
                    next_policy = retry_policies[int(attempt_index)]
                    _log(
                        verbose,
                        "[retry] "
                        f"step={step} attempt={attempt_index} failed: {exc}. "
                        f"Retrying with force_update={next_policy.force_update} "
                        f"omega={next_policy.force_relaxation:.3e} "
                        f"max_coupling_iters={next_policy.max_coupling_iters} "
                        f"reset_history={int(next_policy.reset_interface_history)} "
                        f"fluid_newton={int(next_policy.fluid_max_newton_iter or max_newton_iter)} "
                        f"fluid_globalization={next_policy.fluid_globalization or 'inherit'} "
                        f"fluid_continuation={next_policy.fluid_continuation_scales or (1.0,)}.",
                    )
                    continue
                raise

            increment_elapsed = time.perf_counter() - increment_start
            increment_times.append(float(increment_elapsed))
            break

        if not step_converged:
            _log(
                verbose,
                "[time] "
                f"step={step} failed after {attempt_index} attempt(s); "
                f"disp_rel={last_disp_rel:.3e} load_rel={last_load_rel:.3e}.",
            )
            raise RuntimeError(
                "FSI fixed-point did not converge for step "
                f"{int(step)} after {int(coupling_iter)} iterations "
                f"(disp_rel={float(last_disp_rel):.6e}, load_rel={float(last_load_rel):.6e})."
            )

        converged_steps += 1
        coupling_iters_per_step.append(int(coupling_iter))
        if last_nirb_prediction is not None and nirb_solid_predictor is not None:
            _assign_nirb_full_displacement(
                predictor=nirb_solid_predictor,
                prediction=last_nirb_prediction,
                solid_displacement=solid["d_k"],
            )

        bossak = _bossak_coefficients(alpha=float(bossak_alpha), dt=float(dt_value))
        u_prev_old = np.asarray(fluid["u_prev"].nodal_values, dtype=float).copy()
        a_prev_old = np.asarray(fluid["a_prev"].nodal_values, dtype=float).copy()
        d_prev_old = np.asarray(fluid["d_prev"].nodal_values, dtype=float).copy()
        solid_prev_displacement_for_vtk = np.asarray(solid["d_prev"].nodal_values, dtype=float).copy()
        solid["d_prev"].nodal_values[:] = solid["d_k"].nodal_values[:]
        if "p_k" in solid and "p_prev" in solid:
            solid["p_prev"].nodal_values[:] = solid["p_k"].nodal_values[:]
        if kratos_exact_structure_backend is not None:
            _finalize_kratos_exact_structure_backend_step(backend=kratos_exact_structure_backend)
        fluid["u_prev"].nodal_values[:] = fluid["u_k"].nodal_values[:]
        fluid["p_prev"].nodal_values[:] = fluid["p_k"].nodal_values[:]
        fluid["d_prev2"].nodal_values[:] = fluid["d_prev"].nodal_values[:]
        fluid["d_prev"].nodal_values[:] = fluid["d_mesh"].nodal_values[:]
        if kratos_exact_fluid_backend is not None:
            if (
                last_mesh_vel_fluid_lookup is None
                or last_mesh_accel_fluid_lookup is None
                or last_fluid_accel_lookup is None
            ):
                raise RuntimeError("Kratos exact-fluid backend did not produce the accepted-step kinematics/state.")
            _transfer_vector_field(
                target_dh=fluid["dh"],
                target_vec=fluid["w_mesh_prev"],
                source_lookup=last_mesh_vel_fluid_lookup,
            )
            _transfer_vector_field(
                target_dh=fluid["dh"],
                target_vec=fluid["a_mesh_prev"],
                source_lookup=last_mesh_accel_fluid_lookup,
            )
            fluid["w_mesh_k"].nodal_values[:] = fluid["w_mesh_prev"].nodal_values[:]
            fluid["a_mesh_k"].nodal_values[:] = fluid["a_mesh_prev"].nodal_values[:]
            _transfer_vector_field(
                target_dh=fluid["dh"],
                target_vec=fluid["a_prev"],
                source_lookup=last_fluid_accel_lookup,
            )
            fluid["a_k"].nodal_values[:] = fluid["a_prev"].nodal_values[:]
            _finalize_kratos_exact_fluid_backend_step(backend=kratos_exact_fluid_backend)
        elif kratos_mesh_backend is not None:
            if last_mesh_vel_fluid_lookup is None or last_mesh_accel_fluid_lookup is None:
                raise RuntimeError("Kratos mesh backend did not produce mesh kinematics for the accepted step.")
            _transfer_vector_field(
                target_dh=fluid["dh"],
                target_vec=fluid["w_mesh_prev"],
                source_lookup=last_mesh_vel_fluid_lookup,
            )
            _transfer_vector_field(
                target_dh=fluid["dh"],
                target_vec=fluid["a_mesh_prev"],
                source_lookup=last_mesh_accel_fluid_lookup,
            )
            fluid["w_mesh_k"].nodal_values[:] = fluid["w_mesh_prev"].nodal_values[:]
            fluid["a_mesh_k"].nodal_values[:] = fluid["a_mesh_prev"].nodal_values[:]
            _finalize_kratos_mesh_motion_backend_step(backend=kratos_mesh_backend)
        elif kratos_local_mesh_backend is not None:
            if last_mesh_vel_fluid_lookup is None or last_mesh_accel_fluid_lookup is None:
                raise RuntimeError("Kratos local mesh backend did not produce mesh kinematics for the accepted step.")
            _transfer_vector_field(
                target_dh=fluid["dh"],
                target_vec=fluid["w_mesh_prev"],
                source_lookup=last_mesh_vel_fluid_lookup,
            )
            _transfer_vector_field(
                target_dh=fluid["dh"],
                target_vec=fluid["a_mesh_prev"],
                source_lookup=last_mesh_accel_fluid_lookup,
            )
            fluid["w_mesh_k"].nodal_values[:] = fluid["w_mesh_prev"].nodal_values[:]
            fluid["a_mesh_k"].nodal_values[:] = fluid["a_mesh_prev"].nodal_values[:]
            _finalize_kratos_mesh_motion_backend_step(backend=kratos_local_mesh_backend)
        else:
            fluid["w_mesh_prev"].nodal_values[:] = fluid["w_mesh_k"].nodal_values[:]
            fluid["a_mesh_prev"].nodal_values[:] = fluid["a_mesh_k"].nodal_values[:]
        mesh_ext["m_prev_geom"].nodal_values.fill(0.0)
        if kratos_exact_fluid_backend is None:
            fluid["a_prev"].nodal_values[:] = (
                float(bossak["ma0"]) * (fluid["u_k"].nodal_values[:] - u_prev_old)
                + float(bossak["ma2"]) * a_prev_old
            )
            fluid["a_k"].nodal_values[:] = fluid["a_prev"].nodal_values[:]
        # Kratos persists the last converged predicted subscale across steps and
        # only promotes it to the old-step history at FinalizeSolutionStep.
        # Rebuilding the predictor here collapses those two distinct states and
        # perturbs the next step's first nonlinear iteration.
        _advance_fluid_dvms_history_after_step(
            fluid["dvms_state"],
            dh=fluid["dh"],
            mesh=mesh_f,
            u_curr=fluid["u_prev"],
            a_curr=fluid["a_prev"],
            p_curr=fluid["p_prev"],
            d_curr=fluid["d_prev"],
            mesh_v_curr=fluid["w_mesh_prev"],
            rho_f=float(setup.material.density),
            mu_f=mu_f,
            dt=float(dt_value),
            dynamic_tau=float(dynamic_tau),
            backend=str(backend),
        )
        if use_exact_fluid_operator:
            _maybe_dump_exact_fluid_probe(
                output_dir=output_dir,
                step=int(step),
                coupling_iter=int(coupling_iter),
                stage_label="after_accepted_old_subscale_promotion",
                bc_scale=1.0,
                dt=float(dt_value),
                bossak_alpha=float(bossak_alpha),
                fluid=fluid,
                reaction_point_load_lookup=None,
                reaction_solid_load_lookup=None,
            )
        # The dynamic pressure-subscale old-mass residual for step n+1 uses the
        # previous-step velocity and the previous-step DIVPROJ history. Promote
        # the accepted-step DIVPROJ buffer first, then rebuild old_mass_residual
        # on the carried current geometry so it sees the same (VELOCITY,1,
        # DIVPROJ,1) state that Kratos exposes after the step buffer shift.
        _update_fluid_dvms_state_from_previous_step(
            state=fluid["dvms_state"],
            dh=fluid["dh"],
            mesh=mesh_f,
            u_prev=fluid["u_prev"],
            d_prev=fluid["d_prev"],
            d_geo=fluid["d_mesh"],
            backend=str(backend),
        )
        if use_exact_fluid_operator:
            _maybe_dump_exact_fluid_probe(
                output_dir=output_dir,
                step=int(step),
                coupling_iter=int(coupling_iter),
                stage_label="accepted_step_after_dvms_history",
                bc_scale=1.0,
                dt=float(dt_value),
                bossak_alpha=float(bossak_alpha),
                fluid=fluid,
                reaction_point_load_lookup=None,
                reaction_solid_load_lookup=None,
            )
        if str(active_force_update).lower() == "iqnils" and len(load_guess_history) >= 2 and len(load_return_history) >= 2:
            v_new, w_new = _iqnils_iteration_matrices(
                x_history=load_guess_history,
                g_history=load_return_history,
                iteration_horizon=int(force_iteration_horizon),
            )
            if v_new is not None and w_new is not None:
                iqn_old_dr_mats.appendleft(v_new)
                iqn_old_dg_mats.appendleft(w_new)

        if bool(save_vtk) and int(vtk_every) > 0 and (int(step) % int(vtk_every) == 0):
            fluid_vtk_path, solid_vtk_path = _write_vtk_outputs(
                output_dir=output_dir,
                step=int(step),
                time_value=float(t_now),
                fluid=fluid,
                solid=solid,
                geometry=geometry,
                returned_load_lookup=last_returned_load_lookup,
                dt=float(dt_value),
                porous_material=porous_material if solid_operator_mode == "porous" else None,
                solid_prev_displacement_values=solid_prev_displacement_for_vtk,
            )
            vtk_rows.append(
                {
                    "step": int(step),
                    "time_s": float(t_now),
                    "fluid_vtk": str(fluid_vtk_path),
                    "solid_vtk": str(solid_vtk_path),
                }
            )
            _log(verbose, f"[vtk] step={step} fluid={fluid_vtk_path} solid={solid_vtk_path}")

        _flush_progress_artifacts(
            co_sim_dir=co_sim_dir,
            output_dir=output_dir,
            disp_snapshots=disp_snapshots,
            load_snapshots=load_snapshots,
            load_guess_snapshots=load_guess_snapshots,
            load_return_snapshots=load_return_snapshots,
            fluid_load_guess_snapshots=fluid_load_guess_snapshots,
            fluid_load_return_snapshots=fluid_load_return_snapshots,
            interface_disp_snapshots=interface_disp_snapshots,
            interface_velocity_snapshots=interface_velocity_snapshots,
            interface_traction_snapshots=interface_traction_snapshots,
            reaction_load_snapshots=reaction_load_snapshots,
            stress_load_snapshots=stress_load_snapshots,
            snapshot_rows=snapshot_rows,
            step_rows=step_rows,
            vtk_rows=vtk_rows,
            coupling_iters_per_step=coupling_iters_per_step,
            fluid_times=fluid_times,
            structure_times=structure_times,
            increment_times=increment_times,
            monitor_interface_loads=bool(monitor_interface_loads),
        )
        if int(checkpoint_every) > 0 and (int(step) % int(checkpoint_every) == 0):
            checkpoint_path = _write_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=int(step),
                time_s=float(t_now),
                solid=solid,
                fluid=fluid,
                current_load_lookup=current_load_lookup,
                iqn_old_dr_mats=iqn_old_dr_mats,
                iqn_old_dg_mats=iqn_old_dg_mats,
            )
            _log(verbose, f"[checkpoint] wrote {checkpoint_path}")
        if int(step_history_every) > 0 and (int(step) % int(step_history_every) == 0):
            step_history_path = _write_local_step_history(
                step_history_dir=step_history_dir,
                step=int(step),
                time_s=float(t_now),
                mesh_f=mesh_f,
                mesh_s=mesh_s,
                fluid=fluid,
                solid=solid,
                interface_load_lookup=last_returned_load_lookup,
                interface_disp_lookup=solid_disp_solid_lookup,
                interface_velocity_lookup=mesh_vel_solid_lookup,
            )
            _log(verbose, f"[step-history] wrote {step_history_path}")

        _log(
            verbose,
            "[time] "
            f"step={step} done "
            f"iters={coupling_iter} converged={step_converged} "
            f"disp_abs={last_disp_abs:.3e} load_abs={last_load_abs:.3e} "
            f"disp_rel={last_disp_rel:.3e} load_rel={last_load_rel:.3e} "
            f"wall={increment_elapsed:.3f}s",
        )

    total_elapsed = time.perf_counter() - t_total_start
    _flush_progress_artifacts(
        co_sim_dir=co_sim_dir,
        output_dir=output_dir,
        disp_snapshots=disp_snapshots,
        load_snapshots=load_snapshots,
        load_guess_snapshots=load_guess_snapshots,
        load_return_snapshots=load_return_snapshots,
        fluid_load_guess_snapshots=fluid_load_guess_snapshots,
        fluid_load_return_snapshots=fluid_load_return_snapshots,
        interface_disp_snapshots=interface_disp_snapshots,
        interface_velocity_snapshots=interface_velocity_snapshots,
        interface_traction_snapshots=interface_traction_snapshots,
        reaction_load_snapshots=reaction_load_snapshots,
        stress_load_snapshots=stress_load_snapshots,
        snapshot_rows=snapshot_rows,
        step_rows=step_rows,
        vtk_rows=vtk_rows,
        coupling_iters_per_step=coupling_iters_per_step,
        fluid_times=fluid_times,
        structure_times=structure_times,
        increment_times=increment_times,
        monitor_interface_loads=bool(monitor_interface_loads),
    )
    np.save(co_sim_dir / "total_solving_time.npy", np.asarray(float(total_elapsed), dtype=float))
    disp_matrix = np.load(co_sim_dir / "disp_data.npy")
    load_matrix = np.load(co_sim_dir / "load_data.npy")
    interface_disp_matrix = np.load(co_sim_dir / "interface_disp_data.npy")
    metadata_path = output_dir / "snapshot_metadata.csv"
    vtk_manifest_path = output_dir / "vtk_manifest.csv"
    fluid_pvd_path = output_dir / "vtk_data" / "fluid_timeseries.pvd"
    solid_pvd_path = output_dir / "vtk_data" / "solid_timeseries.pvd"
    timeseries_path = output_dir / "timeseries.csv"

    summary = {
        "output_dir": str(output_dir),
        "mesh_source": str(mesh_descriptor["mesh_source"]),
        "mesh_path": str(mesh_descriptor["fluid_mesh_path"]),
        "fluid_mesh_path": str(mesh_descriptor["fluid_mesh_path"]),
        "solid_mesh_path": str(mesh_descriptor["solid_mesh_path"]),
        "velocity_order": int(poly_order),
        "pressure_order": int(pressure_order_value),
        "fluid_mesh_order": int(mesh_order),
        "solid_order": int(poly_order),
        "reynolds": float(reynolds),
        "reference_velocity": float(reference_velocity_value),
        "dt": float(dt_value),
        "end_time": float(end_time_value),
        "newton_tol": float(newton_tol),
        "exact_fluid_newton_tol": float(exact_fluid_newton_tol),
        "pressure_gauge": float(pressure_gauge),
        "bossak_alpha": float(bossak_alpha),
        "dynamic_tau": float(dynamic_tau),
        "solid_quad_order": int(solid_quad_order),
        "fluid_operator": str(fluid_operator_mode),
        "fluid_hrom_model_path": (
            None if sampled_lspg_hybrid_model is None else str(sampled_lspg_hybrid_model.source_path)
        ),
        "fluid_hrom_switch_iter": (
            None if sampled_lspg_hybrid_model is None else int(sampled_lspg_hybrid_switch_iter)
        ),
        "fluid_hrom_modes": None if sampled_lspg_hybrid_model is None else int(sampled_lspg_hybrid_model.n_modes),
        "fluid_hrom_sample_rows": (
            None if sampled_lspg_hybrid_model is None else int(sampled_lspg_hybrid_model.sample_row_dofs.size)
        ),
        "fluid_hrom_sample_elements": (
            None if sampled_lspg_hybrid_model is None else int(sampled_lspg_hybrid_model.sample_element_ids.size)
        ),
        "fluid_hrom_active_element_weights": (
            None
            if sampled_lspg_hybrid_model is None
            else int(np.count_nonzero(sampled_lspg_hybrid_model.sample_element_weights > 0.0))
        ),
        "fluid_hrom_element_weight_sum": (
            None
            if sampled_lspg_hybrid_model is None
            else float(np.sum(sampled_lspg_hybrid_model.sample_element_weights))
        ),
        "fluid_hrom_stage_count": int(sampled_lspg_hybrid_stage_count),
        "fluid_hrom_exact_stage_count": int(sampled_lspg_hybrid_exact_stage_count),
        "fluid_hrom_fallback_count": int(sampled_lspg_hybrid_fallback_count),
        "fluid_hrom_load_gate_skips": int(sampled_lspg_hybrid_load_gate_skips),
        "fluid_hrom_disp_gate_skips": int(sampled_lspg_hybrid_disp_gate_skips),
        "fluid_hrom_consecutive_gate_skips": int(sampled_lspg_hybrid_consecutive_gate_skips),
        "fluid_hrom_disabled_gate_skips": int(sampled_lspg_hybrid_disabled_gate_skips),
        "fluid_hrom_exact_accept_forced_count": int(sampled_lspg_hybrid_exact_accept_forced_count),
        "fluid_hrom_trial_stage_count": int(sampled_lspg_hybrid_trial_stage_count),
        "fluid_hrom_exact_correction_count": int(sampled_lspg_hybrid_exact_correction_count),
        "fluid_hrom_trial_monitor_failures": int(sampled_lspg_hybrid_trial_monitor_failures),
        "fluid_hrom_interface_reject_count": int(sampled_lspg_hybrid_interface_reject_count),
        "fluid_hrom_interface_correction_count": int(sampled_lspg_hybrid_interface_correction_count),
        "fluid_hrom_disabled_until_step": (
            None if sampled_lspg_hybrid_model is None else int(fluid_hrom_disabled_until_step)
        ),
        "fluid_hrom_max_previous_load_rel": (
            None if sampled_lspg_hybrid_model is None else float(fluid_hrom_max_previous_load_rel_value)
        ),
        "fluid_hrom_min_previous_load_rel": (
            None if sampled_lspg_hybrid_model is None else float(fluid_hrom_min_previous_load_rel_value)
        ),
        "fluid_hrom_max_previous_disp_rel": (
            None if sampled_lspg_hybrid_model is None else float(fluid_hrom_max_previous_disp_rel)
        ),
        "fluid_hrom_max_consecutive_stages": (
            None if sampled_lspg_hybrid_model is None else int(fluid_hrom_max_consecutive_stages_value)
        ),
        "fluid_hrom_require_exact_accept": (
            None if sampled_lspg_hybrid_model is None else bool(fluid_hrom_require_exact_accept)
        ),
        "fluid_hrom_trial_exact_correct": (
            None if sampled_lspg_hybrid_model is None else bool(fluid_hrom_trial_exact_correct)
        ),
        "fluid_hrom_max_trial_load_rel_error": (
            None if sampled_lspg_hybrid_model is None else float(fluid_hrom_max_trial_load_rel_error_value)
        ),
        "fluid_hrom_disable_steps_after_trial_failure": (
            None if sampled_lspg_hybrid_model is None else int(fluid_hrom_disable_steps_after_trial_failure_value)
        ),
        "fluid_hrom_interface_trust": (
            None if sampled_lspg_hybrid_model is None else str(fluid_hrom_interface_trust_value)
        ),
        "fluid_hrom_interface_max_step_ratio": (
            None if sampled_lspg_hybrid_model is None else float(fluid_hrom_interface_max_step_ratio_value)
        ),
        "fluid_hrom_interface_max_load_rel": (
            None if sampled_lspg_hybrid_model is None else float(fluid_hrom_interface_max_load_rel_value)
        ),
        "fluid_hrom_interface_min_correction_alpha": (
            None if sampled_lspg_hybrid_model is None else float(fluid_hrom_interface_min_correction_alpha_value)
        ),
        "solid_operator": str(solid_operator_mode),
        "porous_pressure_order": int(porous_pressure_order_value) if solid_operator_mode == "porous" else None,
        "porous_material": (
            {
                "young_modulus": float(porous_material.young_modulus),
                "poisson_ratio": float(porous_material.poisson_ratio),
                "porosity": float(porous_material.porosity),
                "biot_coefficient": float(porous_material.biot_coefficient),
                "permeability_xx": float(porous_material.permeability_xx),
                "permeability_yy": (
                    None if porous_material.permeability_yy is None else float(porous_material.permeability_yy)
                ),
                "storage_inverse": float(porous_material.biot_modulus_inverse),
                "dynamic_viscosity_liquid": float(porous_material.dynamic_viscosity_liquid),
            }
            if solid_operator_mode == "porous"
            else None
        ),
        "nirb_model_path": None if nirb_model_path is None else str(Path(nirb_model_path)),
        "nirb_start_step": int(nirb_start_step_value),
        "exact_structure_backend": str(exact_structure_backend_mode),
        "exact_fluid_backend": str(exact_fluid_backend_mode),
        "exact_fluid_linear_backend": str(exact_fluid_linear_backend),
        "solid_system_backend": str(_solid_system_backend_mode()),
        "diagnostic_solid_system_backend": str(_diagnostic_solid_system_backend_mode()),
        "kratos_structure_solver_profile": dict(structure_solver_profile),
        "mesh_extension_solver": str(_env_str("PYCUTFEM_EX2_MESH_EXTENSION_SOLVER", "direct").strip().lower()),
        "mesh_extension_formulation": str(
            _env_str(
                "PYCUTFEM_EX2_MESH_EXTENSION_FORMULATION",
                "auto",
            ).strip().lower()
            or "auto"
        ),
        "mesh_linear_solve_mode": str(
            _env_str(
                "PYCUTFEM_EX2_MESH_LINEAR_SOLVE_MODE",
                "",
            ).strip().lower()
            or "direct"
        ),
        "mesh_linear_block_size": str(
            _env_str(
                "PYCUTFEM_EX2_MESH_LINEAR_BLOCK_SIZE",
                "auto",
            ).strip().lower()
            or "auto"
        ),
        "steps_requested": int(step_count),
        "steps_converged": int(converged_steps),
        "max_coupling_iters": int(max_coupling_iters),
        "load_transfer": str(load_transfer),
        "force_update": str(force_update),
        "force_relaxation": float(force_relaxation),
        "force_relaxation_min": float(force_relaxation_min),
        "force_relaxation_max": float(force_relaxation_max),
        "force_iteration_horizon": int(force_iteration_horizon),
        "force_history": int(force_history),
        "force_regularization": float(force_regularization),
        "step_retries": int(step_retries),
        "snapshot_mode": str(snapshot_mode),
        "snapshot_count": int(disp_matrix.shape[1]),
        "interface_dofs": int(load_matrix.shape[0]),
        "interface_snapshot_dofs": int(interface_disp_matrix.shape[0]),
        "solid_state_dofs": int(disp_matrix.shape[0]),
        "coupling_iters_per_step": [int(v) for v in coupling_iters_per_step],
        "mean_coupling_iters": float(np.mean(coupling_iters_per_step)) if coupling_iters_per_step else 0.0,
        "total_wall_time_s": float(total_elapsed),
        "mean_increment_time_s": float(np.mean(increment_times)) if increment_times else 0.0,
        "mean_solid_solve_time_s": float(np.mean(structure_times)) if structure_times else 0.0,
        "mean_fluid_solve_time_s": float(np.mean(fluid_times)) if fluid_times else 0.0,
        "co_sim_dir": str(co_sim_dir),
        "timeseries_path": str(timeseries_path),
        "snapshot_metadata_path": str(metadata_path),
        "save_vtk": bool(save_vtk),
        "vtk_every": int(vtk_every),
        "vtk_manifest_path": str(vtk_manifest_path),
        "fluid_pvd_path": str(fluid_pvd_path),
        "solid_pvd_path": str(solid_pvd_path),
        "vtk_count": int(len(vtk_rows)),
        "monitor_interface_loads": bool(monitor_interface_loads),
        "checkpoint_every": int(checkpoint_every),
        "checkpoint_dir": str(checkpoint_dir),
        "step_history_every": int(step_history_every),
        "step_history_dir": str(step_history_dir),
        "restart_from": None if restart_path is None else str(restart_path),
        "fluid_dvms_state": _fluid_dvms_summary(fluid),
    }
    dump_json(summary, output_dir / "summary.json")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_local_example2(
        output_dir=args.output_dir,
        reference_root=args.reference_root,
        mesh_source=str(args.mesh_source),
        mesh_size=float(args.mesh_size),
        mesh_order=int(args.mesh_order),
        poly_order=int(args.poly_order),
        pressure_order=args.pressure_order,
        mesh_backend=str(args.mesh_backend),
        quadrature_order=args.quad_order,
        solid_quadrature_order=args.solid_quad_order,
        reynolds=float(args.reynolds),
        reference_velocity=args.reference_velocity,
        dt=args.dt,
        end_time=args.end_time,
        max_steps=args.max_steps,
        max_coupling_iters=int(args.max_coupling_iters),
        coupling_rel_tol=float(args.coupling_rel_tol),
        coupling_abs_tol=float(args.coupling_abs_tol),
        load_transfer=str(args.load_transfer),
        force_update=str(args.force_update),
        force_relaxation=float(args.force_relaxation),
        force_relaxation_min=float(args.force_relaxation_min),
        force_relaxation_max=float(args.force_relaxation_max),
        force_iteration_horizon=int(args.force_iteration_horizon),
        force_history=int(args.force_history),
        force_regularization=float(args.force_regularization),
        step_retries=int(args.step_retries),
        newton_tol=float(args.newton_tol),
        max_newton_iter=int(args.max_newton_iter),
        bossak_alpha=float(args.bossak_alpha),
        dynamic_tau=float(args.dynamic_tau),
        pressure_gauge=float(args.pressure_gauge),
        fluid_operator=str(args.fluid_operator),
        fluid_hrom_model_path=args.fluid_hrom_model_path,
        fluid_hrom_switch_iter=args.fluid_hrom_switch_iter,
        fluid_hrom_max_iterations=args.fluid_hrom_max_iterations,
        fluid_hrom_residual_tol=args.fluid_hrom_residual_tol,
        fluid_hrom_fallback_exact=bool(args.fluid_hrom_fallback_exact),
        fluid_hrom_max_previous_load_rel=float(args.fluid_hrom_max_previous_load_rel),
        fluid_hrom_min_previous_load_rel=float(args.fluid_hrom_min_previous_load_rel),
        fluid_hrom_max_previous_disp_rel=float(args.fluid_hrom_max_previous_disp_rel),
        fluid_hrom_max_consecutive_stages=int(args.fluid_hrom_max_consecutive_stages),
        fluid_hrom_require_exact_accept=bool(args.fluid_hrom_require_exact_accept),
        fluid_hrom_trial_exact_correct=bool(args.fluid_hrom_trial_exact_correct),
        fluid_hrom_max_trial_load_rel_error=float(args.fluid_hrom_max_trial_load_rel_error),
        fluid_hrom_disable_steps_after_trial_failure=int(args.fluid_hrom_disable_steps_after_trial_failure),
        fluid_hrom_interface_trust=str(args.fluid_hrom_interface_trust),
        fluid_hrom_interface_max_step_ratio=float(args.fluid_hrom_interface_max_step_ratio),
        fluid_hrom_interface_max_load_rel=float(args.fluid_hrom_interface_max_load_rel),
        fluid_hrom_interface_min_correction_alpha=float(args.fluid_hrom_interface_min_correction_alpha),
        solid_operator=str(args.solid_operator),
        porous_pressure_order=args.porous_pressure_order,
        porous_porosity=float(args.porous_porosity),
        porous_biot_coefficient=float(args.porous_biot_coefficient),
        porous_permeability=float(args.porous_permeability),
        porous_storage_inverse=float(args.porous_storage_inverse),
        nirb_model_path=args.nirb_model_path,
        nirb_start_step=int(args.nirb_start_step),
        backend=str(args.backend),
        linear_backend=str(args.linear_backend),
        snapshot_mode=str(args.snapshot_mode),
        save_vtk=bool(args.save_vtk),
        vtk_every=int(args.vtk_every),
        monitor_interface_loads=bool(args.monitor_interface_loads),
        checkpoint_every=int(args.checkpoint_every),
        step_history_every=int(args.step_history_every),
        restart_from=args.restart_from,
        reuse_mesh=bool(args.reuse_mesh),
        verbose=bool(args.verbose),
    )
    print(f"mesh: {summary['mesh_path']}")
    print(f"steps_requested: {summary['steps_requested']}")
    print(f"steps_converged: {summary['steps_converged']}")
    print(f"snapshots: {summary['snapshot_count']}")
    print(f"mean_coupling_iters: {summary['mean_coupling_iters']:.3f}")
    print(f"mean_solid_solve_time_s: {summary['mean_solid_solve_time_s']:.6f}")
    print(f"mean_fluid_solve_time_s: {summary['mean_fluid_solve_time_s']:.6f}")
    print(f"vtk_count: {summary['vtk_count']}")
    print(f"summary: {Path(args.output_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
