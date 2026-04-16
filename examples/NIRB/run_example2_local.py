from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.core.topology import Node
from pycutfem.fem import transform
from pycutfem.fem.reference import get_reference
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.mor.interface import build_restriction_matrix
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
    _eval_scalar_with_grad,
    _eval_vector_with_grad,
    _fluid_dvms_summary,
    _kratos_dvms_current_element_size_coefficient,
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
from examples.NIRB.example2_problem import _named_constant, build_conforming_mesh


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
    target_step = int(os.getenv("PYCUTFEM_EX2_DUMP_FLUID_STEP", "1") or "1")
    target_iter = int(os.getenv("PYCUTFEM_EX2_DUMP_FLUID_ITER", "1") or "1")
    if int(step) != target_step or int(coupling_iter) != target_iter:
        return

    probe_dir = Path(output_dir) / "debug_fluid_stage"
    probe_dir.mkdir(parents=True, exist_ok=True)
    u_coords, u_values = _vector_field_matrix(fluid["dh"], fluid["u_k"])
    p_coords, p_values = _scalar_field_matrix(fluid["dh"], fluid["p_k"])
    d_coords, d_values = _vector_field_matrix(fluid["dh"], fluid["d_mesh"])
    _, d_prev_values = _vector_field_matrix(fluid["dh"], fluid["d_prev"])
    _, d_prev2_values = _vector_field_matrix(fluid["dh"], fluid["d_prev2"])
    _, w_mesh_prev_values = _vector_field_matrix(fluid["dh"], fluid["w_mesh_prev"])
    _, a_mesh_prev_values = _vector_field_matrix(fluid["dh"], fluid["a_mesh_prev"])
    _, a_prev_values = _vector_field_matrix(fluid["dh"], fluid["a_prev"])
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
        "d_coords": np.asarray(d_coords, dtype=float),
        "d_values": np.asarray(d_values, dtype=float),
        "d_prev_values": np.asarray(d_prev_values, dtype=float),
        "d_prev2_values": np.asarray(d_prev2_values, dtype=float),
        "w_mesh_prev_values": np.asarray(w_mesh_prev_values, dtype=float),
        "a_mesh_prev_values": np.asarray(a_mesh_prev_values, dtype=float),
        "a_prev_values": np.asarray(a_prev_values, dtype=float),
        "w_mesh_values": np.asarray(w_mesh_values, dtype=float),
        "a_mesh_values": np.asarray(a_mesh_values, dtype=float),
    }
    dvms_state = fluid.get("dvms_state")
    if isinstance(dvms_state, FluidDVMSState):
        payload["dvms_old_subscale_velocity"] = np.asarray(dvms_state.old_subscale_velocity, dtype=float)
        payload["dvms_predicted_subscale_velocity"] = np.asarray(dvms_state.predicted_subscale_velocity, dtype=float)
        payload["dvms_momentum_projection"] = np.asarray(dvms_state.momentum_projection, dtype=float)
        payload["dvms_mass_projection"] = np.asarray(dvms_state.mass_projection, dtype=float)
        payload["dvms_old_mass_residual"] = np.asarray(dvms_state.old_mass_residual, dtype=float)
    if reaction_point_load_lookup is not None:
        payload["reaction_point_coords"] = np.asarray(reaction_point_load_lookup.coords, dtype=float)
        payload["reaction_point_values"] = np.asarray(reaction_point_load_lookup.values, dtype=float)
    if reaction_solid_load_lookup is not None:
        payload["reaction_solid_coords"] = np.asarray(reaction_solid_load_lookup.coords, dtype=float)
        payload["reaction_solid_values"] = np.asarray(reaction_solid_load_lookup.values, dtype=float)
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
        super().__init__(**kwargs)

    def arm_initial_old_subscale_build(self) -> None:
        """Use old-step subscale data on the next solver step-begin only."""
        self._skip_refresh_once = True
        self._reset_predicted_on_step_begin_once = True

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

    def _dump_first_assembly_probe(self, *, stage_label: str) -> None:
        if self._first_assembly_probe is None or self._first_assembly_probe_dumped:
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
                "p_k": self.p_k,
                "d_mesh": self.d_mesh,
                "d_prev": self.d_prev,
                "d_prev2": self.d_prev2,
                "w_mesh_prev": self.mesh_v_prev,
                "a_mesh_prev": self.mesh_a_prev,
                "a_prev": self.a_prev,
                "w_mesh_k": self.mesh_v,
                "dvms_state": self.state,
            },
            reaction_point_load_lookup=None,
            reaction_solid_load_lookup=None,
        )
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
            self._dump_first_assembly_probe(stage_label="first_assembly_skip_refresh")
            self._skip_refresh_once = False
            return
        super().before_assembly(solver=solver, coeffs=coeffs, need_matrix=need_matrix)
        self._dump_first_assembly_probe(stage_label="first_assembly_after_refresh")


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
    viscous = mu_const * (grad_u_phys + grad_u_phys.T - _EX2L_TWO_THIRDS * div_u_phys * identity)
    return -p * identity + viscous


def _fluid_cauchy_stress_numpy(*, p_val: float, grad_u_phys: np.ndarray, div_u_phys: float, mu_f: float) -> np.ndarray:
    identity = np.eye(2, dtype=float)
    grad_u_arr = np.asarray(grad_u_phys, dtype=float).reshape(2, 2)
    div_val = float(div_u_phys)
    viscous = float(mu_f) * (grad_u_arr + grad_u_arr.T - (2.0 / 3.0) * div_val * identity)
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


def _assemble_structural_similarity_mesh_matrix(
    *,
    dh: DofHandler,
    mesh: Mesh,
    m_prev_geom: VectorFunction,
) -> "sp.csr_matrix":
    import scipy.sparse as sp

    connectivity_data = getattr(mesh, "corner_connectivity", None)
    if connectivity_data is None:
        connectivity_data = getattr(mesh, "elements_connectivity", None)
    if connectivity_data is None:
        raise AttributeError("Mesh does not expose corner_connectivity/elements_connectivity for mesh-motion assembly.")
    connectivity = np.asarray(connectivity_data, dtype=int)
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise NotImplementedError("Kratos-matched structural-similarity assembly currently supports Triangle2D3 meshes only.")

    node_coords_ref = np.asarray(mesh.nodes_x_y_pos, dtype=float)
    node_coords_cur = node_coords_ref.copy()
    if hasattr(m_prev_geom, "components") and len(getattr(m_prev_geom, "components")) >= 2:
        disp_prev = np.column_stack(
            [
                np.asarray(m_prev_geom.components[0].nodal_values, dtype=float),
                np.asarray(m_prev_geom.components[1].nodal_values, dtype=float),
            ]
        )
        if disp_prev.shape == node_coords_cur.shape:
            node_coords_cur = node_coords_cur + disp_prev
        elif disp_prev.size != 0:
            raise ValueError(
                "m_prev_geom component shape mismatch for structural-similarity assembly: "
                f"expected {node_coords_cur.shape}, got {disp_prev.shape}."
            )

    node_to_gdof_x = np.full(node_coords_ref.shape[0], -1, dtype=int)
    node_to_gdof_y = np.full(node_coords_ref.shape[0], -1, dtype=int)
    for gdof in range(int(dh.total_dofs)):
        field_name, node_id = dh._dof_to_node_map[int(gdof)]
        if node_id is None:
            continue
        if field_name == "mx":
            node_to_gdof_x[int(node_id)] = int(gdof)
        elif field_name == "my":
            node_to_gdof_y[int(node_id)] = int(gdof)

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

    for nodes in connectivity:
        elem_nodes = np.asarray(nodes, dtype=int)
        coords = np.asarray(node_coords_cur[elem_nodes], dtype=float)
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
        k_elem = ref_weight * (b.T @ c @ b)
        gdofs = np.asarray(
            [
                node_to_gdof_x[int(elem_nodes[0])],
                node_to_gdof_y[int(elem_nodes[0])],
                node_to_gdof_x[int(elem_nodes[1])],
                node_to_gdof_y[int(elem_nodes[1])],
                node_to_gdof_x[int(elem_nodes[2])],
                node_to_gdof_y[int(elem_nodes[2])],
            ],
            dtype=int,
        )
        if np.any(gdofs < 0):
            raise RuntimeError("Could not resolve mesh-moving DOFs for a triangle element.")
        rr, cc = np.meshgrid(gdofs, gdofs, indexing="ij")
        rows.extend(rr.reshape(-1).tolist())
        cols.extend(cc.reshape(-1).tolist())
        data.extend(np.asarray(k_elem, dtype=float).reshape(-1).tolist())

    return sp.csr_matrix((data, (rows, cols)), shape=(int(dh.total_dofs), int(dh.total_dofs)))


def _solve_structural_similarity_mesh_extension(
    *,
    prob: dict[str, object],
    interface_disp: CoordinateLookup,
    interface_tag: str,
    fixed_tags: tuple[str, ...],
    quad_order: int = 6,
    backend: str = "cpp",
    linear_backend: str = "petsc",
) -> None:
    dh: DofHandler = prob["dh"]
    solve_mode = _env_str(
        "PYCUTFEM_EX2_MESH_EXTENSION_SOLVER",
        "reference_form",
    ).strip().lower()
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

    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    zero = lambda x, y, t=0.0: 0.0
    bcs = [
        BoundaryCondition("mx", "dirichlet", tag, zero) for tag in fixed_tags
    ] + [
        BoundaryCondition("my", "dirichlet", tag, zero) for tag in fixed_tags
    ] + [
        BoundaryCondition("mx", "dirichlet", interface_tag, interface_disp.component(0)),
        BoundaryCondition("my", "dirichlet", interface_tag, interface_disp.component(1)),
    ]

    mesh: Mesh = dh.mixed_element.mesh
    A_raw = _assemble_structural_similarity_mesh_matrix(
        dh=dh,
        mesh=mesh,
        m_prev_geom=prob["m_prev_geom"],
    )
    rhs_raw = np.zeros(int(dh.total_dofs), dtype=float)
    bc_map = dh.get_dirichlet_data(bcs) or {}
    A = A_raw.tolil(copy=True)
    rhs = np.asarray(rhs_raw, dtype=float).copy()
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
    sol = spsolve(A.tocsr(), rhs)
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
    quadrature_order: int | None = None,
) -> dict[str, object]:
    me = MixedElement(
        mesh,
        field_specs={
            "ux": int(poly_order),
            "uy": int(poly_order),
            "p": int(pressure_order),
            "mx": int(poly_order),
            "my": int(poly_order),
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
    p_k = Function("p_k", "p", dof_handler=dh)
    p_prev = Function("p_prev", "p", dof_handler=dh)
    d_mesh = VectorFunction("d_mesh", ["mx", "my"], dof_handler=dh)
    d_prev = VectorFunction("d_prev", ["mx", "my"], dof_handler=dh)
    d_prev2 = VectorFunction("d_prev2", ["mx", "my"], dof_handler=dh)
    w_mesh_prev = VectorFunction("w_mesh_prev", ["mx", "my"], dof_handler=dh)
    a_mesh_prev = VectorFunction("a_mesh_prev", ["mx", "my"], dof_handler=dh)
    w_mesh_k = VectorFunction("w_mesh_k", ["mx", "my"], dof_handler=dh)
    a_mesh_k = VectorFunction("a_mesh_k", ["mx", "my"], dof_handler=dh)
    for function in (u_k, u_prev, a_prev, d_mesh, d_prev, d_prev2, w_mesh_prev, a_mesh_prev, w_mesh_k, a_mesh_k):
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
    }


def _build_mesh_extension_problem(mesh: Mesh, *, poly_order: int) -> dict[str, object]:
    me, dh, dm, z, m_k = _build_vector_problem(mesh, prefix="m", order=poly_order)
    m_prev_geom = VectorFunction("m_prev_geom", ["mx", "my"], dof_handler=dh)
    m_prev_geom.nodal_values.fill(0.0)
    return {"me": me, "dh": dh, "dm": dm, "z": z, "m_k": m_k, "m_prev_geom": m_prev_geom}


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


def _boundary_field_data(dh: DofHandler, field: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    dh._ensure_dof_coords()
    mesh = dh.mixed_element.mesh
    boundary_points: list[np.ndarray] = []
    boundary_segments: list[tuple[np.ndarray, np.ndarray]] = []
    for eid in mesh.edge_bitset(tag).to_indices():
        edge = mesh.edge(int(eid))
        if edge.right is not None:
            continue
        node_ids = list(getattr(edge, "all_nodes", None) or edge.nodes)
        pts = np.asarray(mesh.nodes_x_y_pos[node_ids], dtype=float)
        for point in pts:
            boundary_points.append(np.asarray(point, dtype=float))
        if pts.shape[0] >= 2:
            boundary_segments.append((pts[0], pts[-1]))
    if not boundary_points:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=int)

    points_arr = np.asarray(boundary_points, dtype=float)
    field_ids = np.asarray(dh.get_field_slice(field), dtype=int)
    field_coords = np.asarray(dh._dof_coords[field_ids], dtype=float)
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
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=int)
    return field_coords[np.asarray(keep, dtype=int)], field_ids[np.asarray(keep, dtype=int)]


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
    for gdof, lidx in vector._g2l.items():
        field, node_id = dh._dof_to_node_map[int(gdof)]
        if node_id is None:
            continue
        if field not in vector.field_names:
            continue
        component_idx = int(vector.field_names.index(field))
        if component_idx >= 2:
            continue
        point_data[int(node_id), int(component_idx)] = float(vector.nodal_values[int(lidx)])
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
    for gdof, lidx in function._g2l.items():
        _field, node_id = dh._dof_to_node_map[int(gdof)]
        if node_id is None:
            continue
        point_data[int(node_id), 0] = float(function.nodal_values[int(lidx)])
    return point_data


def _mesh_node_ids(mesh: Mesh) -> np.ndarray:
    mapped = getattr(mesh, "_mdpa_new_to_old_node", None)
    if mapped is not None:
        arr = np.asarray(mapped, dtype=int).reshape(-1)
        if arr.size == len(mesh.nodes_list):
            return arr.copy()
    return np.arange(1, int(len(mesh.nodes_list)) + 1, dtype=int)


def _solid_system_backend_mode() -> str:
    return str(os.getenv("PYCUTFEM_EX2_SOLID_SYSTEM_BACKEND", "kratos_local") or "kratos_local").strip().lower()


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

    return {
        "KM": KM,
        "solver": solver,
        "main_model_part": main_model_part,
        "process_info": process_info,
        "all_nodes": all_nodes,
        "field_index_by_old_id": field_index_by_old_id,
        "dx_ids": dx_ids,
        "dy_ids": dy_ids,
        "kratos_elements": tuple(kratos_elements),
        "gdofs_map": np.asarray(gdofs_map, dtype=int),
        "local_perm": np.asarray(local_perm, dtype=int),
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


def _assemble_kratos_local_solid_system_batch(
    *,
    backend: dict[str, object],
    d_k: VectorFunction,
    need_matrix: bool,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    _sync_kratos_local_solid_backend_state(backend=backend, d_k=d_k)
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


class _KratosLocalSolidSystemOperator(RuntimeOperator):
    """Replace the symbolic solid assembly with the exact Kratos local system."""

    def __init__(self, *, backend: dict[str, object], d_k: VectorFunction) -> None:
        self.backend = backend
        self.d_k = d_k

    def after_assembly(self, *, solver, coeffs, A_red, R_red, need_matrix: bool):
        import scipy.sparse as sp

        del coeffs
        K_elem, F_elem, element_ids, gdofs_map = _assemble_kratos_local_solid_system_batch(
            backend=self.backend,
            d_k=self.d_k,
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
        return solver.scatter_element_contribs_reduced(
            K_elem=K_elem,
            F_elem=F_elem,
            element_ids=element_ids,
            gdofs_map=gdofs_map,
            A_red=A_exact,
            R_red=R_exact,
        )


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


def _fluid_boundary_conditions(
    *,
    iface_velocity: CoordinateLookup,
    inlet_lookup: Callable[[float, float], float],
    interface_tag: str,
    outlet_tag: str,
    walls_tag: str,
    cylinder_tag: str,
) -> tuple[list[BoundaryCondition], list[BoundaryCondition]]:
    zero = lambda x, y, t=0.0: 0.0
    bcs = [
        BoundaryCondition("ux", "dirichlet", "inlet", inlet_lookup),
        BoundaryCondition("uy", "dirichlet", "inlet", zero),
        BoundaryCondition("ux", "dirichlet", walls_tag, zero),
        BoundaryCondition("uy", "dirichlet", walls_tag, zero),
        BoundaryCondition("ux", "dirichlet", cylinder_tag, zero),
        BoundaryCondition("uy", "dirichlet", cylinder_tag, zero),
        BoundaryCondition("ux", "dirichlet", interface_tag, iface_velocity.component(0)),
        BoundaryCondition("uy", "dirichlet", interface_tag, iface_velocity.component(1)),
        BoundaryCondition("p", "dirichlet", outlet_tag, zero),
    ]
    bcs_homog = [
        BoundaryCondition("ux", "dirichlet", "inlet", zero),
        BoundaryCondition("uy", "dirichlet", "inlet", zero),
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
    assemble_fluid_dvms_local_contribution_batch(
        mesh=mesh,
        dh=prob["dh"],
        u_k=prob["u_k"],
        u_prev=prob["u_prev"],
        a_prev=prob["a_prev"],
        p_k=prob["p_k"],
        d_mesh=prob["d_mesh"],
        d_prev=prob["d_prev"],
        mesh_v=None,
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
        mesh_v=None,
        mesh_v_prev=prob["w_mesh_prev"],
        mesh_a_prev=prob["a_mesh_prev"],
        rho_f=float(rho_f),
        mu_f=float(mu_f),
        dt=float(dt),
        bossak_alpha=float(bossak_alpha),
        dynamic_tau=float(dynamic_tau),
        backend=backend_name,
    )
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
        mesh_v=None,
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
    conv_speed = (dot(conv_velocity, conv_velocity) + _EX2L_CONV_EPS) ** _EX2L_HALF
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
    tau_one = dynamic_tau_const * _EX2L_ONE / (
        tau_c1 * mu_const / (h * h)
        + rho * (inv_dt + tau_c2 * conv_speed / h)
    )
    tau_two = mu_const + rho * conv_speed * h / _EX2L_FOUR
    tau_p = rho * h * h * inv_dt / tau_c1
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
    # Kratos StructuralMeshMovingElement is a weighted linear-elastic mesh solve.
    # The constitutive law is scaled pointwise with
    #   weighting_factor = detJ * (100 / detJ)^1.5
    # and the FE integration is carried out on the current mesh geometry. In
    # the transformed reference-domain form below, the physical current-cell
    # Jacobian is area_ref * J_prev for the reference partitioned triangles
    # used here, so the constitutive part only needs the remaining stiffening
    # factor (100 / (area_ref * J_prev))^1.5.
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
    m_prev_geom: VectorFunction = prob["m_prev_geom"]
    F_prev = Identity(2) + grad(m_prev_geom)
    Finv_prev = inv(F_prev)
    J_prev = det(F_prev)
    stiffening = (mesh_stiffening_factor / (area_ref * J_prev + mesh_jac_eps)) ** mesh_stiffening_exponent
    grad_dm_phys = dot(grad(dm), Finv_prev)
    div_dm_phys = inner(Finv_prev.T, grad(dm))
    eps_dm_phys = _EX2L_HALF * (grad_dm_phys + grad_dm_phys.T)
    sigma_dm_phys = stiffening * (_EX2L_TWO * mesh_mu * eps_dm_phys + mesh_lambda * div_dm_phys * Identity(2))
    equation = Equation(
        inner(J_prev * dot(sigma_dm_phys, Finv_prev.T), grad(z)) * dx(metadata={"q": int(quad_order)}),
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
):
    import scipy.sparse as sp

    dh: DofHandler = prob["dh"]
    mesh: Mesh = dh.mixed_element.mesh
    ndof = int(dh.total_dofs)
    A_full = sp.lil_matrix((ndof, ndof), dtype=float) if need_matrix else None
    R_full = np.zeros(ndof, dtype=float)
    batch = assemble_fluid_dvms_local_contribution_batch(
        mesh=mesh,
        dh=dh,
        u_k=prob["u_k"],
        u_prev=prob.get("u_prev"),
        a_prev=prob.get("a_prev"),
        p_k=prob["p_k"],
        d_mesh=prob["d_mesh"],
        d_prev=prob["d_prev"],
        mesh_v=None,
        mesh_v_prev=prob.get("w_mesh_prev"),
        mesh_a_prev=prob.get("a_mesh_prev"),
        state=prob.get("dvms_state"),
        rho_f=float(rho_f),
        mu_f=float(mu_f),
        dt=float(dt),
        bossak_alpha=float(bossak_alpha),
        element_ids=np.arange(int(mesh.n_elements), dtype=int),
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
        element_ids=np.asarray(batch.element_ids, dtype=int),
        gdofs_map=np.asarray(gdofs_map, dtype=int),
        A_full=A_full,
        R_full=R_full,
    )

    return (A_full.tocsr() if A_full is not None else None), R_full


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
    _update_fluid_dvms_predicted_subscale(
        state=state,
        dh=dh,
        mesh=mesh,
        u_k=prob["u_k"],
        u_prev=prob["u_prev"],
        a_prev=prob["a_prev"],
        p_k=prob["p_k"],
        d_mesh=prob["d_mesh"],
        d_prev=prob["d_prev"],
        d_prev2=prob.get("d_prev2"),
        mesh_v=None,
        mesh_v_prev=prob.get("w_mesh_prev"),
        mesh_a_prev=prob.get("a_mesh_prev"),
        rho_f=float(rho_f),
        mu_f=float(mu_f),
        dt=float(dt),
        bossak_alpha=float(bossak_alpha),
        dynamic_tau=float(dynamic_tau),
        backend=str(backend),
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
) -> CoordinateLookup:
    _refresh_fluid_reaction_reconstruction_state(
        prob=prob,
        rho_f=float(rho_f),
        mu_f=float(mu_f),
        dt=float(dt),
        bossak_alpha=float(bossak_alpha),
        dynamic_tau=float(dynamic_tau),
        backend=str(backend),
    )
    _, raw_residual = _assemble_fluid_local_velocity_contribution_raw(
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
    )
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
) -> tuple[CoordinateLookup, CoordinateLookup]:
    disp_vals = np.empty((iface_coords.shape[0], 2), dtype=float)
    vel_vals = np.empty((iface_coords.shape[0], 2), dtype=float)
    for i, xy in enumerate(np.asarray(iface_coords, dtype=float)):
        disp_curr, _ = _eval_vector_with_grad(dh, mesh, d_curr, tuple(xy))
        disp_prev_val, _ = _eval_vector_with_grad(dh, mesh, d_prev, tuple(xy))
        disp_vals[i, :] = disp_curr
        vel_vals[i, :] = (disp_curr - disp_prev_val) / max(float(dt), 1.0e-14)
    return CoordinateLookup(iface_coords, disp_vals, dim=2), CoordinateLookup(iface_coords, vel_vals, dim=2)


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


def _write_vtk_outputs(
    *,
    output_dir: Path,
    step: int,
    time_value: float,
    fluid: dict[str, object],
    solid: dict[str, object],
    geometry,
    returned_load_lookup: CoordinateLookup | None,
) -> tuple[Path, Path]:
    vtk_root = Path(output_dir) / "vtk_data"
    fluid_dir = vtk_root / "vtk_output_fsi_cfd"
    solid_dir = vtk_root / "vtk_output_fsi_csm"
    fluid_dir.mkdir(parents=True, exist_ok=True)
    solid_dir.mkdir(parents=True, exist_ok=True)

    fluid_path = fluid_dir / f"FluidParts_FluidPart_0_{int(step):04d}.vtu"
    solid_path = solid_dir / f"Structure_0_{int(step):04d}.vtu"

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
    )

    solid_fields: dict[str, object] = {
        "DISPLACEMENT": solid["d_k"],
        "TIME": np.full(int(len(solid["mesh"].nodes_list)), float(time_value), dtype=float),
    }
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
    )
    return fluid_path, solid_path


def _snapshot_function_values(functions: list[Function | VectorFunction]) -> list[np.ndarray]:
    return [np.asarray(function.nodal_values, dtype=float).copy() for function in functions]


def _restore_function_values(functions: list[Function | VectorFunction], snapshots: list[np.ndarray]) -> None:
    if len(functions) != len(snapshots):
        raise ValueError("functions / snapshots length mismatch")
    for function, values in zip(functions, snapshots):
        function.nodal_values[:] = np.asarray(values, dtype=float)


def _snapshot_fluid_dvms_state(state: FluidDVMSState | None) -> dict[str, np.ndarray] | None:
    if not isinstance(state, FluidDVMSState):
        return None
    return {
        "old_subscale_velocity": np.asarray(state.old_subscale_velocity, dtype=float).copy(),
        "predicted_subscale_velocity": np.asarray(state.predicted_subscale_velocity, dtype=float).copy(),
        "momentum_projection": np.asarray(state.momentum_projection, dtype=float).copy(),
        "mass_projection": np.asarray(state.mass_projection, dtype=float).copy(),
        "old_mass_residual": np.asarray(state.old_mass_residual, dtype=float).copy(),
    }


def _restore_fluid_dvms_state(state: FluidDVMSState | None, snapshot: dict[str, np.ndarray] | None) -> None:
    if not isinstance(state, FluidDVMSState) or snapshot is None:
        return
    for key, values in snapshot.items():
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
    for key, values in snapshot.items():
        if key in skip:
            continue
        getattr(state, key)[:, ...] = np.asarray(values, dtype=float)
    state.sync_coefficients_from_samples()


def _snapshot_fluid_stage_state(prob: dict[str, object]) -> dict[str, object]:
    return {
        "u_k": np.asarray(prob["u_k"].nodal_values, dtype=float).copy(),
        "p_k": np.asarray(prob["p_k"].nodal_values, dtype=float).copy(),
        "u_prev": np.asarray(prob["u_prev"].nodal_values, dtype=float).copy(),
        "p_prev": np.asarray(prob["p_prev"].nodal_values, dtype=float).copy(),
        "a_prev": np.asarray(prob["a_prev"].nodal_values, dtype=float).copy(),
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
        "fluid_d_mesh": np.asarray(fluid["d_mesh"].nodal_values, dtype=float).copy(),
        "fluid_d_prev": np.asarray(fluid["d_prev"].nodal_values, dtype=float).copy(),
        "fluid_d_prev2": np.asarray(fluid["d_prev2"].nodal_values, dtype=float).copy(),
        "fluid_w_mesh_prev": np.asarray(fluid["w_mesh_prev"].nodal_values, dtype=float).copy(),
        "fluid_a_mesh_prev": np.asarray(fluid["a_mesh_prev"].nodal_values, dtype=float).copy(),
        "current_load_values": np.asarray(current_load_lookup.values, dtype=float).copy(),
    }
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
    iface_coords: np.ndarray,
    load_guess_vals: np.ndarray,
    returned_load_vals: np.ndarray,
    load_guess_history: list[np.ndarray],
    load_return_history: list[np.ndarray],
    iqn_old_dr_mats: list[np.ndarray] | None,
    iqn_old_dg_mats: list[np.ndarray] | None,
    omega_force: float,
    force_iteration_horizon: int,
    force_regularization: float,
) -> tuple[CoordinateLookup, bool]:
    load_guess_arr = np.asarray(load_guess_vals, dtype=float).copy()
    returned_arr = np.asarray(returned_load_vals, dtype=float).copy()
    returned_lookup = CoordinateLookup(iface_coords, returned_arr, dim=2)
    if bool(step_converged):
        return returned_lookup, False

    load_guess_history.append(load_guess_arr.copy())
    load_return_history.append(returned_arr.copy())
    if str(active_force_update).lower() == "iqnils":
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
        return CoordinateLookup(iface_coords, next_load_values, dim=2), True

    return (
        _relaxed_lookup(
            iface_coords,
            load_guess_arr,
            returned_arr,
            omega=float(omega_force),
        ),
        True,
    )


@dataclass(frozen=True)
class _CouplingRetryPolicy:
    force_update: str
    force_relaxation: float
    reset_interface_history: bool = False


def _build_coupling_retry_policies(
    *,
    force_update: str,
    force_relaxation: float,
    force_relaxation_min: float,
    force_relaxation_max: float,
    max_retries: int,
) -> list[_CouplingRetryPolicy]:
    policies: list[_CouplingRetryPolicy] = []

    def _append(mode: str, omega: float, *, reset: bool) -> None:
        omega_clamped = float(np.clip(float(omega), float(force_relaxation_min), float(force_relaxation_max)))
        policy = _CouplingRetryPolicy(str(mode), omega_clamped, bool(reset))
        if policy not in policies:
            policies.append(policy)

    _append(str(force_update).lower(), float(force_relaxation), reset=False)
    fallback_omegas = [min(float(force_relaxation), 0.10), 0.05, 0.02]
    for omega in fallback_omegas:
        _append("constant", omega, reset=True)
        if len(policies) >= int(max_retries) + 1:
            break
    return policies[: max(int(max_retries), 0) + 1]


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
    parser.add_argument("--max-coupling-iters", type=int, default=6)
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
    parser.add_argument("--step-retries", type=int, default=2, help="Retry a failed outer FSI step with safer interface updates.")
    parser.add_argument("--newton-tol", type=float, default=1.0e-6)
    parser.add_argument("--max-newton-iter", type=int, default=20)
    parser.add_argument("--bossak-alpha", type=float, default=-0.3)
    parser.add_argument("--dynamic-tau", type=float, default=1.0)
    parser.add_argument("--pressure-gauge", type=float, default=1.0e-5)
    parser.add_argument("--fluid-operator", choices=("exact", "continuous"), default="exact")
    parser.add_argument("--backend", choices=("python", "jit", "cpp"), default="python")
    parser.add_argument("--linear-backend", choices=("scipy", "petsc"), default="scipy")
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
    max_coupling_iters: int = 6,
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
    step_retries: int = 2,
    newton_tol: float = 1.0e-6,
    max_newton_iter: int = 12,
    bossak_alpha: float = -0.3,
    dynamic_tau: float = 1.0,
    pressure_gauge: float = 1.0e-5,
    fluid_operator: str = "exact",
    backend: str = "python",
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
    if fluid_operator_mode in {"condensed", "local", "exact_local"}:
        fluid_operator_mode = "exact"
    if fluid_operator_mode not in {"exact", "continuous"}:
        raise ValueError(f"Unsupported fluid_operator={fluid_operator!r}")
    if fluid_operator_mode != "exact":
        legacy_exact_flag = str(os.getenv("PYCUTFEM_EX2_ATTEMPT_LOCAL_CORRECTION", "0") or "0").strip().lower()
        if legacy_exact_flag in {"1", "true", "yes"}:
            fluid_operator_mode = "exact"
    use_exact_fluid_operator = fluid_operator_mode == "exact"
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
            if bool(use_exact_fluid_operator) and mesh_source_value == "reference" and restart_payload is None
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
    if exact_structure_backend_mode == "kratos_live" and mesh_source_value != "reference":
        raise RuntimeError("The persistent Kratos exact-structure backend requires mesh_source='reference'.")
    if exact_structure_backend_mode == "kratos_live" and restart_payload is not None:
        raise RuntimeError("The persistent Kratos exact-structure backend is not yet supported with restart_from.")

    fluid = _build_fluid_problem(
        mesh_f,
        poly_order=int(poly_order),
        pressure_order=pressure_order_value,
        quadrature_order=quad_order,
    )
    mesh_ext = _build_mesh_extension_problem(mesh_f, poly_order=int(poly_order))
    solid = _build_solid_problem(mesh_s, poly_order=int(poly_order))
    kratos_local_solid_backend = _maybe_build_kratos_local_solid_backend(
        benchmark_root=Path(setup.reference.root),
        prob=solid,
    ) if exact_structure_backend_mode == "local" else None
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
            exact_fluid_backend_mode = "local"
    kratos_mesh_backend = (
        _build_kratos_mesh_motion_backend(
            fluid_mdpa_path=Path(mesh_descriptor["fluid_mesh_path"]),
            dt=float(dt_value),
            bossak_alpha=float(bossak_alpha),
        )
        if mesh_backend_value == "kratos" and kratos_exact_fluid_backend is None
        else None
    )
    fluid["mesh"] = mesh_f
    solid["mesh"] = mesh_s
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
    solid_interface_mass = _build_interface_mass_matrix(mesh_s, solid_iface_coords, geometry.interface_tag)

    zero_load_lookup = CoordinateLookup(
        solid_iface_coords,
        np.zeros((solid_iface_coords.shape[0], 2), dtype=float),
        dim=2,
    )
    zero_fluid_load_lookup = CoordinateLookup(
        fluid_iface_coords,
        np.zeros((fluid_iface_coords.shape[0], 2), dtype=float),
        dim=2,
    )
    load_transfer_value = str(load_transfer).lower()
    accelerate_on_fluid_load = load_transfer_value == "reaction"
    current_load_lookup = zero_fluid_load_lookup if accelerate_on_fluid_load else zero_load_lookup
    prev_disp_iter_vals = np.zeros((solid_iface_coords.shape[0], 2), dtype=float)

    mu_f = float(setup.material.density * setup.material.kinematic_viscosity)
    mu_s = float(setup.material.shear_modulus)
    lambda_s = float(setup.material.lame_lambda)

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
                1.0e-8,
            )
        ),
    )
    fluid_mixed_solution_criteria = (
        ("ux,uy", float(exact_fluid_newton_tol), float(exact_fluid_newton_tol)),
        ("p", float(exact_fluid_newton_tol), float(exact_fluid_newton_tol)),
    )
    if _env_bool("PYCUTFEM_EX2_DISABLE_EXACT_FLUID_MIXED_CRITERIA", False):
        fluid_mixed_solution_criteria = ()
    # Kratos' monolithic fluid solve is driven by residual tolerances. Keep the
    # local mixed-solution shortcut, but guard it with the same residual scale
    # so the exact-operator path cannot accept an iterate purely on small state
    # updates while the nonlinear residual is still above the Kratos tolerance.
    fluid_mixed_solution_residual_factor = _env_float(
        "PYCUTFEM_EX2_EXACT_FLUID_MIXED_RES_FACTOR",
        1.0,
    )

    if restart_payload is not None:
        solid["d_k"].nodal_values[:] = np.asarray(restart_payload["solid_d_k"], dtype=float)
        solid["d_prev"].nodal_values[:] = np.asarray(restart_payload["solid_d_prev"], dtype=float)
        fluid["u_k"].nodal_values[:] = np.asarray(restart_payload["fluid_u_k"], dtype=float)
        fluid["p_k"].nodal_values[:] = np.asarray(restart_payload["fluid_p_k"], dtype=float)
        fluid["u_prev"].nodal_values[:] = np.asarray(restart_payload["fluid_u_prev"], dtype=float)
        fluid["p_prev"].nodal_values[:] = np.asarray(restart_payload["fluid_p_prev"], dtype=float)
        fluid["a_prev"].nodal_values[:] = np.asarray(restart_payload["fluid_a_prev"], dtype=float)
        fluid["d_mesh"].nodal_values[:] = np.asarray(restart_payload["fluid_d_mesh"], dtype=float)
        fluid["d_prev"].nodal_values[:] = np.asarray(restart_payload["fluid_d_prev"], dtype=float)
        fluid["d_prev2"].nodal_values[:] = np.asarray(restart_payload["fluid_d_prev2"], dtype=float)
        fluid["w_mesh_prev"].nodal_values[:] = np.asarray(restart_payload["fluid_w_mesh_prev"], dtype=float)
        fluid["a_mesh_prev"].nodal_values[:] = np.asarray(restart_payload["fluid_a_mesh_prev"], dtype=float)
        fluid["w_mesh_k"].nodal_values[:] = fluid["w_mesh_prev"].nodal_values[:]
        fluid["a_mesh_k"].nodal_values[:] = fluid["a_mesh_prev"].nodal_values[:]
        _restore_fluid_dvms_state(
            fluid.get("dvms_state"),
            {
                "old_subscale_velocity": np.asarray(restart_payload.get("dvms_old_subscale_velocity", np.zeros_like(fluid["dvms_state"].old_subscale_velocity)), dtype=float),
                "predicted_subscale_velocity": np.asarray(restart_payload.get("dvms_predicted_subscale_velocity", np.zeros_like(fluid["dvms_state"].predicted_subscale_velocity)), dtype=float),
                "momentum_projection": np.asarray(restart_payload.get("dvms_momentum_projection", np.zeros_like(fluid["dvms_state"].momentum_projection)), dtype=float),
                "mass_projection": np.asarray(restart_payload.get("dvms_mass_projection", np.zeros_like(fluid["dvms_state"].mass_projection)), dtype=float),
                "old_mass_residual": np.asarray(restart_payload.get("dvms_old_mass_residual", np.zeros_like(fluid["dvms_state"].old_mass_residual)), dtype=float),
            },
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
            source_lookup=mesh_restart_lookup,
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
        increment_start = time.perf_counter()
        solid_prev_step = _snapshot_function_values([solid["d_prev"]])
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

        step_converged = False
        last_disp_abs = last_disp_rel = last_load_abs = last_load_rel = float("nan")
        last_force_omega = float(force_relaxation)
        last_returned_load_lookup: CoordinateLookup | None = None
        last_mesh_vel_fluid_lookup: CoordinateLookup | None = None
        last_mesh_accel_fluid_lookup: CoordinateLookup | None = None
        last_fluid_accel_lookup: CoordinateLookup | None = None
        coupling_iter = 0
        active_force_update = str(force_update).lower()
        active_force_relaxation = float(force_relaxation)
        prev_force_residual: np.ndarray | None = None
        load_guess_history: list[np.ndarray] = []
        load_return_history: list[np.ndarray] = []

        for coupling_iter in range(1, int(max_coupling_iters) + 1):
            _section(verbose, f"[fsi] step={step} fixed-point iter={coupling_iter}/{max_coupling_iters}")
            if accelerate_on_fluid_load:
                current_structure_load_lookup = _resample_lookup_to_coords(
                    _negate_lookup(current_load_lookup),
                    solid_iface_coords,
                )
            else:
                current_structure_load_lookup = current_load_lookup
            load_guess_vals = np.asarray(current_structure_load_lookup.values, dtype=float).copy()

            t_solid0 = time.perf_counter()
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
                solid_solver = NewtonSolver(
                    residual_form=solid_res,
                    jacobian_form=solid_jac,
                    dof_handler=solid["dh"],
                    mixed_element=solid["me"],
                    bcs=solid_bcs,
                    bcs_homog=solid_bcs_homog,
                    newton_params=NewtonParameters(
                        newton_tol=float(newton_tol),
                        max_newton_iter=1 if bool(use_struct_one_step) else int(max_newton_iter),
                        print_level=3,
                        accept_nonconverged_atol_factor=(
                            float(_EX2L_KRATOS_STRUCT_ONE_STEP_ACCEPT_FACTOR)
                            if bool(use_struct_one_step)
                            else 0.0
                        ),
                        line_search=False,
                        globalization="none",
                    ),
                    lin_params=LinearSolverParameters(backend=str(linear_backend)),
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
                solid_runtime_ops: list[RuntimeOperator] = []
                if kratos_local_solid_backend is not None:
                    solid_runtime_ops.append(
                        _KratosLocalSolidSystemOperator(
                            backend=kratos_local_solid_backend,
                            d_k=solid["d_k"],
                        )
                    )
                solid_runtime_ops.append(_ReducedResidualShiftOperator(solid_point_load_red))
                solid_solver.set_runtime_operators(solid_runtime_ops)
                solid_guess = _snapshot_function_values([solid["d_k"]])
                _restore_function_values([solid["d_prev"]], solid_prev_step)
                solid_solver.solve_time_interval(
                    functions=[solid["d_k"]],
                    prev_functions=[solid["d_prev"]],
                    time_params=TimeStepperParameters(
                        dt=1.0,
                        max_steps=1,
                        final_time=1.0,
                        stop_on_steady=False,
                        step_initial_guess_callback=_guess_callback_from_snapshots(solid_guess),
                    ),
                )
                _restore_function_values([solid["d_prev"]], solid_prev_step)
            solid_elapsed = time.perf_counter() - t_solid0
            structure_times.append(float(solid_elapsed))

            solid_disp_solid_lookup, _ = _solid_interface_disp_velocity(
                dh=solid["dh"],
                mesh=mesh_s,
                d_curr=solid["d_k"],
                d_prev=solid["d_prev"],
                iface_coords=solid_iface_coords,
                dt=dt_value,
            )
            # Kratos uses a nearest-neighbor mapper for structure
            # DISPLACEMENT -> fluid MESH_DISPLACEMENT in DoubleFlap. Replicate
            # that transfer instead of evaluating the solid FE field directly
            # at the fluid interface coordinates.
            solid_disp_fluid_lookup = _resample_lookup_to_coords(
                solid_disp_solid_lookup,
                fluid_iface_coords,
            )
            disp_snapshot = _flatten_vector_snapshot(solid["dh"], solid["d_k"])

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
                if load_transfer_value == "stress" or bool(monitor_interface_loads):
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
                        source_lookup=mesh_lookup,
                    )
                else:
                    mesh_ext["m_prev_geom"].nodal_values[:] = mesh_ext["m_k"].nodal_values[:]
                    if np.max(np.abs(np.asarray(solid_disp_fluid_lookup.values, dtype=float))) <= 1.0e-18:
                        mesh_ext["m_k"].nodal_values.fill(0.0)
                    else:
                        _solve_structural_similarity_mesh_extension(
                            prob=mesh_ext,
                            interface_disp=solid_disp_fluid_lookup,
                            interface_tag=geometry.interface_tag,
                            fixed_tags=fixed_mesh_tags,
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
                last_mesh_vel_fluid_lookup = mesh_vel_fluid_lookup
                last_mesh_accel_fluid_lookup = mesh_accel_fluid_lookup
                _restore_fluid_dvms_state(
                    fluid.get("dvms_state"),
                    fluid_dvms_prev_step,
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
                continuation_scales = (
                    (0.25, 0.5, 1.0)
                    if (bool(use_step1_fluid_continuation) and int(step) == 1)
                    else (1.0,)
                )
                t_fluid0 = time.perf_counter()
                fluid_guess = _snapshot_function_values([fluid["u_k"], fluid["p_k"]])
                fluid_exact_operator = FluidDVMSCondensedLocalSystemOperator(
                    mesh=mesh_f,
                    dh=fluid["dh"],
                    u_k=fluid["u_k"],
                    u_prev=fluid["u_prev"],
                    a_prev=fluid["a_prev"],
                    p_k=fluid["p_k"],
                    d_mesh=fluid["d_mesh"],
                    d_prev=fluid["d_prev"],
                    d_prev2=fluid["d_prev2"],
                    mesh_v=None,
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
                    p_k=fluid["p_k"],
                    d_mesh=fluid["d_mesh"],
                    d_prev=fluid["d_prev"],
                    d_prev2=fluid["d_prev2"],
                    mesh_v=None,
                    mesh_v_prev=fluid["w_mesh_prev"],
                    mesh_a_prev=fluid["a_mesh_prev"],
                    rho_f=float(setup.material.density),
                    mu_f=mu_f,
                    dt=dt_value,
                    bossak_alpha=float(bossak_alpha),
                    dynamic_tau=float(dynamic_tau),
                )
                use_first_build_old_subscale = _env_bool(
                    "PYCUTFEM_EX2_FIRST_BUILD_OLD_SUBSCALE",
                    True,
                )
                if int(coupling_iter) == 1 and bool(use_first_build_old_subscale):
                    fluid_predictor_operator.arm_initial_old_subscale_build()
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
                        stage_ops = [fluid_predictor_operator, fluid_exact_operator]
                        stage_max_newton_iter = int(max_newton_iter)
                        stage_globalization = _env_str(
                            "PYCUTFEM_EX2_EXACT_FLUID_GLOBALIZATION",
                            "line_search",
                        )
                        stage_line_search = _env_bool(
                            "PYCUTFEM_EX2_EXACT_FLUID_LINE_SEARCH",
                            False,
                        )
                        stage_ls_fail_hard = _env_bool(
                            "PYCUTFEM_EX2_EXACT_FLUID_LS_FAIL_HARD",
                            True,
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
                    _section(
                        verbose,
                        "[fluid-solver] "
                        f"step={step} coupling_iter={coupling_iter} scale={float(bc_scale):.3f} "
                        f"mode={fluid_operator_mode} globalization={stage_globalization} "
                        f"line_search={stage_line_search} ls_fail_hard={stage_ls_fail_hard} "
                        f"max_newton_iter={stage_max_newton_iter}",
                    )
                    stage_newton_params = NewtonParameters(
                        newton_tol=float(exact_fluid_newton_tol if use_exact_fluid_operator else newton_tol),
                        max_newton_iter=int(stage_max_newton_iter),
                        line_search=bool(stage_line_search),
                        ls_fail_hard=bool(stage_ls_fail_hard),
                        globalization=str(stage_globalization),
                        mixed_solution_criteria=fluid_mixed_solution_criteria,
                        mixed_solution_max_residual_factor=float(fluid_mixed_solution_residual_factor),
                    )
                    stage_lin_params = LinearSolverParameters(backend=str(linear_backend))
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
                                str(linear_backend),
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
                    _restore_function_values([fluid["u_prev"], fluid["p_prev"]], fluid_prev_step)
                    _restore_function_values(
                        [fluid["d_prev"], fluid["d_prev2"], fluid["w_mesh_prev"], fluid["a_mesh_prev"]],
                        fluid_mesh_prev_step,
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
                    try:
                        stage_solver.solve_time_interval(
                            functions=[fluid["u_k"], fluid["p_k"]],
                            prev_functions=[fluid["u_prev"], fluid["p_prev"]],
                            aux_functions={
                                "a_prev": fluid["a_prev"],
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
                    except Exception:
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
                        raise
                    fluid_guess = _snapshot_function_values([fluid["u_k"], fluid["p_k"]])
                    _log(
                        verbose,
                        "[fluid-stage] "
                        f"scale={float(bc_scale):.3f} "
                        f"mode={fluid_operator_mode} "
                        f"u_max={_field_abs_max(fluid['u_k']):.3e} "
                        f"p_max={_field_abs_max(fluid['p_k']):.3e}",
                    )
                _restore_function_values([fluid["u_prev"], fluid["p_prev"]], fluid_prev_step)
                _restore_function_values(
                    [fluid["d_prev"], fluid["d_prev2"], fluid["w_mesh_prev"], fluid["a_mesh_prev"]],
                    fluid_mesh_prev_step,
                )
                fluid_elapsed = time.perf_counter() - t_fluid0
                fluid_times.append(float(fluid_elapsed))

                if load_transfer_value == "reaction" or bool(monitor_interface_loads):
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
                    )
                    reaction_solid_load_lookup = _resample_lookup_to_coords(
                        _negate_lookup(reaction_point_load_lookup),
                        solid_iface_coords,
                    )
                if load_transfer_value == "stress" or bool(monitor_interface_loads):
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
                "attempt": 1,
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
            row["strict_converged"] = bool(step_converged)
            row["kratos_disp_converged_5e-3"] = bool(kratos_disp_converged)
            row["kratos_load_converged_5e-3"] = bool(kratos_load_converged)
            row["kratos_step_converged_5e-3"] = bool(kratos_step_converged)
            step_rows.append(row)
            if snapshot_mode == "converged" and step_converged:
                keep_snapshot = True
            if keep_snapshot:
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
            current_load_lookup, _ = _advance_coupling_load_guess(
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
            )
            if step_converged:
                break

            prev_disp_iter_vals = np.asarray(solid_disp_solid_lookup.values, dtype=float).copy()
            prev_force_residual = np.asarray(load_residual, dtype=float).copy()
            last_force_omega = float(omega_force)

        if not step_converged:
            _log(
                verbose,
                "[time] "
                f"step={step} failed to converge after {coupling_iter} fixed-point iterations; "
                f"disp_rel={last_disp_rel:.3e} load_rel={last_load_rel:.3e}. "
                "Refusing to commit the step state.",
            )
            raise RuntimeError(
                "FSI fixed-point did not converge for step "
                f"{int(step)} after {int(coupling_iter)} iterations "
                f"(disp_rel={float(last_disp_rel):.6e}, load_rel={float(last_load_rel):.6e})."
            )

        converged_steps += 1
        coupling_iters_per_step.append(int(coupling_iter))
        increment_elapsed = time.perf_counter() - increment_start
        increment_times.append(float(increment_elapsed))

        bossak = _bossak_coefficients(alpha=float(bossak_alpha), dt=float(dt_value))
        u_prev_old = np.asarray(fluid["u_prev"].nodal_values, dtype=float).copy()
        a_prev_old = np.asarray(fluid["a_prev"].nodal_values, dtype=float).copy()
        d_prev_old = np.asarray(fluid["d_prev"].nodal_values, dtype=float).copy()
        solid["d_prev"].nodal_values[:] = solid["d_k"].nodal_values[:]
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
        else:
            w_mesh_prev_old = np.asarray(fluid["w_mesh_prev"].nodal_values, dtype=float).copy()
            a_mesh_prev_old = np.asarray(fluid["a_mesh_prev"].nodal_values, dtype=float).copy()
            w_mesh_curr, a_mesh_curr = _bossak_displacement_kinematics_values(
                d_curr=np.asarray(fluid["d_mesh"].nodal_values, dtype=float),
                d_prev=d_prev_old,
                v_prev=w_mesh_prev_old,
                a_prev=a_mesh_prev_old,
                dt=dt_value,
                alpha=float(bossak_alpha),
            )
            fluid["w_mesh_prev"].nodal_values[:] = w_mesh_curr
            fluid["a_mesh_prev"].nodal_values[:] = a_mesh_curr
            fluid["w_mesh_k"].nodal_values[:] = w_mesh_curr
            fluid["a_mesh_k"].nodal_values[:] = a_mesh_curr
        if kratos_exact_fluid_backend is None:
            fluid["a_prev"].nodal_values[:] = (
                float(bossak["ma0"]) * (fluid["u_k"].nodal_values[:] - u_prev_old)
                + float(bossak["ma2"]) * a_prev_old
            )
        _advance_fluid_dvms_history_after_step(fluid["dvms_state"])
        _update_fluid_dvms_state_from_previous_step(
            state=fluid["dvms_state"],
            dh=fluid["dh"],
            mesh=mesh_f,
            u_prev=fluid["u_prev"],
            d_prev=fluid["d_prev"],
            d_geo=fluid["d_mesh"],
            backend=str(backend),
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
    timeseries_path = output_dir / "timeseries.csv"

    summary = {
        "output_dir": str(output_dir),
        "mesh_source": str(mesh_descriptor["mesh_source"]),
        "mesh_path": str(mesh_descriptor["fluid_mesh_path"]),
        "fluid_mesh_path": str(mesh_descriptor["fluid_mesh_path"]),
        "solid_mesh_path": str(mesh_descriptor["solid_mesh_path"]),
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
        "exact_structure_backend": str(exact_structure_backend_mode),
        "exact_fluid_backend": str(exact_fluid_backend_mode),
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
