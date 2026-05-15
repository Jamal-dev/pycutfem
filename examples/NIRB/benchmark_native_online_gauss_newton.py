from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.mor import (
    gauss_newton_step,
    native_kernel_metadata_from_runner,
    sampled_lspg_rows_from_native_kernel,
    solve_native_online_gauss_newton,
)
from pycutfem.mor.reduced_assembly import sampled_lspg_rows_from_local_blocks
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_triangles


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float))) if values else 0.0


def _time_repeated(repeats: int, fn):
    timings: list[float] = []
    last = None
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        last = fn()
        timings.append(time.perf_counter() - t0)
    return _median(timings), last


def _build_problem(nx: int):
    nodes, elems, edges, corners = structured_triangles(
        1.0,
        1.0,
        nx_quads=int(nx),
        ny_quads=int(nx),
        poly_order=1,
    )
    mesh = Mesh(nodes, elems, edges, corners, element_type="tri", poly_order=1)
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")
    uh = Function("uh", "u", dh)
    uh.nodal_values[:] = 1.0
    v = TestFunction("u", dh)
    du = TrialFunction("u", dh)
    residual = ((uh * uh) - Constant(4.0)) * v * dx(metadata={"q": 4})
    tangent = Constant(2.0) * uh * du * v * dx(metadata={"q": 4})
    compiler = FormCompiler(dh, quadrature_order=4, backend="cpp")
    element_ids = np.arange(int(mesh.n_elements), dtype=np.int32)
    residual_runner, residual_funcs, residual_args, gdofs = compiler._prepare_volume_jit_kernel(
        residual,
        element_ids=element_ids,
        full_local_layout=True,
    )
    tangent_runner, tangent_funcs, tangent_args, _ = compiler._prepare_volume_jit_kernel(
        tangent,
        element_ids=element_ids,
        full_local_layout=True,
    )
    residual_runner(residual_funcs, residual_args)
    tangent_runner(tangent_funcs, tangent_args)
    return {
        "mesh": mesh,
        "dh": dh,
        "uh": uh,
        "residual_runner": residual_runner,
        "residual_funcs": residual_funcs,
        "residual_args": residual_args,
        "tangent_runner": tangent_runner,
        "tangent_funcs": tangent_funcs,
        "tangent_args": tangent_args,
        "gdofs": gdofs,
    }


def _python_online_loop(problem, *, max_iterations: int, residual_tol: float, line_search: bool):
    dh = problem["dh"]
    uh = problem["uh"]
    basis = np.ones((int(dh.total_dofs), 1), dtype=float)
    rows = np.arange(int(dh.total_dofs), dtype=np.int64)
    q = np.array([1.0], dtype=float)
    residual_runner = problem["residual_runner"]
    tangent_runner = problem["tangent_runner"]
    residual_args = problem["residual_args"]
    tangent_args = problem["tangent_args"]
    gdofs = problem["gdofs"]
    for _iteration in range(max(1, int(max_iterations))):
        uh.nodal_values[:] = q[0]
        _K_res, F_res, _J_res = residual_runner(problem["residual_funcs"], residual_args)
        K_tan, _F_tan, _J_tan = tangent_runner(problem["tangent_funcs"], tangent_args)
        residual, jacobian = sampled_lspg_rows_from_local_blocks(
            K_elem=np.asarray(K_tan, dtype=float),
            raw_rhs_elem=-np.asarray(F_res, dtype=float),
            gdofs_map=gdofs,
            row_dofs=rows,
            trial_basis=basis,
            backend="cpp",
        )
        if float(np.linalg.norm(residual)) <= float(residual_tol):
            break
        step = gauss_newton_step(jacobian, residual, backend="cpp").step
        if line_search:
            best = q + step
            best_norm = float("inf")
            for search_iter in range(6):
                trial = q + (0.5**search_iter) * step
                uh.nodal_values[:] = trial[0]
                _K_trial, F_trial, _J_trial = residual_runner(problem["residual_funcs"], residual_args)
                trial_residual, _trial_jacobian = sampled_lspg_rows_from_local_blocks(
                    K_elem=np.asarray(K_tan, dtype=float),
                    raw_rhs_elem=-np.asarray(F_trial, dtype=float),
                    gdofs_map=gdofs,
                    row_dofs=rows,
                    trial_basis=basis,
                    backend="cpp",
                )
                trial_norm = float(np.linalg.norm(trial_residual))
                if trial_norm < best_norm:
                    best = trial
                    best_norm = trial_norm
            q = np.asarray(best, dtype=float).reshape(-1)
        else:
            q = q + step
    return q


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MOR native online Gauss-Newton paths.")
    parser.add_argument("--nx", type=int, default=2, help="Structured triangle mesh subdivisions per direction.")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if args.cache_dir is not None:
        os.environ["PYCUTFEM_CACHE_DIR"] = str(Path(args.cache_dir).resolve())

    problem = _build_problem(int(args.nx))
    dh = problem["dh"]
    basis = np.ones((int(dh.total_dofs), 1), dtype=float)
    rows = np.arange(int(dh.total_dofs), dtype=np.int64)
    residual_metadata = native_kernel_metadata_from_runner(problem["residual_runner"])
    tangent_metadata = native_kernel_metadata_from_runner(problem["tangent_runner"])

    J = np.array([[2.0], [0.5], [3.0], [-1.5]], dtype=float)
    r = np.array([1.0, -0.25, 2.0, 0.5], dtype=float)
    gauss_newton_step(J, r, backend="cpp")
    step_median, _ = _time_repeated(args.repeats, lambda: gauss_newton_step(J, r, backend="cpp").step)

    sampled_lspg_rows_from_native_kernel(
        metadata_capsule=tangent_metadata,
        param_order=problem["tangent_runner"].param_order,
        static_args=problem["tangent_args"],
        row_dofs=rows,
        trial_basis=basis,
    )
    assembler_median, _ = _time_repeated(
        args.repeats,
        lambda: sampled_lspg_rows_from_native_kernel(
            metadata_capsule=tangent_metadata,
            param_order=problem["tangent_runner"].param_order,
            static_args=problem["tangent_args"],
            row_dofs=rows,
            trial_basis=basis,
        ),
    )

    _python_online_loop(
        problem,
        max_iterations=int(args.max_iterations),
        residual_tol=1.0e-11,
        line_search=True,
    )
    python_median, python_q = _time_repeated(
        args.repeats,
        lambda: _python_online_loop(
            problem,
            max_iterations=int(args.max_iterations),
            residual_tol=1.0e-11,
            line_search=True,
        ),
    )

    native_result = None

    def _native_solve():
        nonlocal native_result
        native_result = solve_native_online_gauss_newton(
            residual_metadata_capsule=residual_metadata,
            residual_param_order=problem["residual_runner"].param_order,
            residual_static_args=problem["residual_args"],
            tangent_metadata_capsule=tangent_metadata,
            tangent_param_order=problem["tangent_runner"].param_order,
            tangent_static_args=problem["tangent_args"],
            trial_basis=basis,
            offset=np.zeros(int(dh.total_dofs), dtype=float),
            initial_coefficients=np.array([1.0], dtype=float),
            row_dofs=rows,
            coefficient_arg_names=("u_uh_loc",),
            max_iterations=int(args.max_iterations),
            residual_tol=1.0e-11,
            line_search=True,
            adaptive_damping=True,
        )
        return native_result

    _native_solve()
    native_median, native_result = _time_repeated(args.repeats, _native_solve)
    payload = {
        "mesh_elements": int(problem["mesh"].n_elements),
        "dofs": int(dh.total_dofs),
        "repeats": int(args.repeats),
        "median_seconds": {
            "python_current_loop": float(python_median),
            "cpp_step_only": float(step_median),
            "cpp_native_assembler": float(assembler_median),
            "cpp_full_online_loop": float(native_median),
        },
        "solutions": {
            "python_current_loop": np.asarray(python_q, dtype=float).reshape(-1).tolist(),
            "cpp_full_online_loop": native_result.coefficients.tolist(),
        },
        "native_timing_counters": dict(native_result.timing_counters),
        "native_iterations": int(native_result.iterations),
        "native_converged": bool(native_result.converged),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        Path(args.json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
