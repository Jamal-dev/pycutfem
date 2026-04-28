#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path


def _sanitize_runtime_environment() -> dict[str, str]:
    updates: dict[str, str] = {}
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    lib_entries: list[str] = []
    if conda_prefix:
        conda_lib = str((Path(conda_prefix) / "lib").resolve())
        lib_entries.append(conda_lib)
        conda_root = str(Path(conda_prefix).resolve())
        for raw_entry in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
            entry = raw_entry.strip()
            if not entry:
                continue
            resolved = str(Path(entry).resolve())
            if resolved.startswith(conda_root) and resolved not in lib_entries:
                lib_entries.append(resolved)
    if lib_entries:
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_entries)
        updates["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    else:
        if "LD_LIBRARY_PATH" in os.environ:
            updates["LD_LIBRARY_PATH"] = ""
        os.environ.pop("LD_LIBRARY_PATH", None)
    return updates


_RUNTIME_ENV_UPDATES = _sanitize_runtime_environment()

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI

from pycutfem.coupling import PreCICEPointParticipant

from examples.biofilms.benchmarks.seboldt import paper1_benchmark7_seboldt_partitioned_moving_linear as base


def _clip_probe_points(points: np.ndarray, *, x0: float, x1: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float).copy()
    pts[:, 0] = np.clip(pts[:, 0], float(x0) + 1.0e-10, float(x1) - 1.0e-10)
    return pts


def solve_participant(args) -> dict[str, float | int | str]:
    params = base.Example2Params()
    nx = int(args.nx)
    ny_fluid = int(round((params.y_fluid1 - params.y_fluid0) / ((params.x1 - params.x0) / nx)))
    dt = float(args.dt)
    t_final = float(args.t_final)
    num_steps = int(round(t_final / dt))
    if abs(num_steps * dt - t_final) > 1.0e-12:
        raise ValueError("t_final must be an integer multiple of dt.")
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    compiler_updates = base._ensure_working_jit_compiler()
    if MPI.COMM_WORLD.rank == 0:
        print(
            "[seboldt-fluid-precice] start "
            f"kappa={float(args.kappa):.6e} nx={nx} dt={dt:.6e} t_final={t_final:.6e} "
            f"factor_solver={args.factor_solver!r} outdir={outdir}",
            flush=True,
        )
        if compiler_updates:
            print(
                "[seboldt-fluid-precice] compiler-fallback "
                + " ".join(f"{k}={v}" for k, v in compiler_updates.items()),
                flush=True,
            )
        if _RUNTIME_ENV_UPDATES:
            print(
                "[seboldt-fluid-precice] runtime-env "
                + " ".join(f"{k}={v}" for k, v in _RUNTIME_ENV_UPDATES.items()),
                flush=True,
            )

    msh_f_ref = base._build_rect_mesh(params.x0, params.y_fluid0, params.x1, params.y_fluid1, nx, ny_fluid)
    msh_f_cur = base._build_rect_mesh(params.x0, params.y_fluid0, params.x1, params.y_fluid1, nx, ny_fluid)
    tags_f_ref = base._build_boundary_markers(msh_f_ref, "fluid", params)
    tags_f_cur = base._build_boundary_markers(msh_f_cur, "fluid", params)
    ds_f_ref = ufl.Measure("ds", domain=msh_f_ref, subdomain_data=tags_f_ref.meshtags)
    ds_f_cur = ufl.Measure("ds", domain=msh_f_cur, subdomain_data=tags_f_cur.meshtags)
    dx_f_ref = ufl.dx(domain=msh_f_ref)
    dx_f_cur = ufl.dx(domain=msh_f_cur)

    V_f_ref_geom = fem.functionspace(msh_f_ref, ("Lagrange", 2, (2,)))
    V_f_ref_vel = fem.functionspace(msh_f_ref, ("Lagrange", 2, (2,)))
    T_f_ref = fem.functionspace(msh_f_ref, ("DG", 1, (2, 2)))

    cell_f = msh_f_cur.ufl_cell().cellname()
    vel_el = basix.ufl.element("Lagrange", cell_f, 2, shape=(2,))
    pres_el = basix.ufl.element("Lagrange", cell_f, 1)
    W_f_cur = fem.functionspace(msh_f_cur, basix.ufl.mixed_element([vel_el, pres_el]))
    V_f_cur_vel, map_f_vel = W_f_cur.sub(0).collapse()
    Q_f_cur_pres, map_f_pres = W_f_cur.sub(1).collapse()
    T_f_cur = fem.functionspace(msh_f_cur, ("DG", 1, (2, 2)))
    V_f_cur_scalar = fem.functionspace(msh_f_cur, ("Lagrange", 2))

    eta_f_ref_old = fem.Function(V_f_ref_geom, name="eta_f_ref_old")
    w_f_ref = fem.Function(V_f_ref_geom, name="w_f_ref")
    v_f_ref_old = fem.Function(V_f_ref_vel, name="v_f_ref_old")
    v_f_cur_old = fem.Function(V_f_cur_vel, name="v_f_cur_old")
    p_f_cur_old = fem.Function(Q_f_cur_pres, name="p_f_cur_old")
    v_f_cur_old_on_new = fem.Function(V_f_cur_vel, name="v_f_cur_old_on_new")
    w_f_cur = fem.Function(V_f_cur_vel, name="w_f_cur")
    grad_eta_ref = fem.Function(T_f_ref, name="grad_eta_ref")
    grad_v_cur = fem.Function(T_f_cur, name="grad_v_cur")

    eta_top_ref = fem.Function(V_f_ref_geom)
    xi_n_f_cur = fem.Function(V_f_cur_scalar)
    xi_t_f_cur = fem.Function(V_f_cur_scalar)
    q_n_f_cur = fem.Function(V_f_cur_scalar)
    traction_f_cur = fem.Function(V_f_cur_scalar)

    eta_f_ref_new = fem.Function(V_f_ref_geom, name="eta_f_ref_new")
    w_f_cur_mixed = fem.Function(W_f_cur, name="w_f_cur")
    v_f_cur_new = fem.Function(V_f_cur_vel, name="v_f_cur_new")
    p_f_cur_new = fem.Function(Q_f_cur_pres, name="p_f_cur_new")
    v_f_ref_new = fem.Function(V_f_ref_vel, name="v_f_ref_new")

    fluid_ref_coords, fluid_vertex_order = base._build_vertex_map(msh_f_ref, V_f_ref_geom)
    fluid_ref_geom_to_cur_vel = base._build_pointwise_transfer_order(V_f_ref_geom, V_f_cur_vel)
    fluid_ref_vel_to_cur_vel = base._build_pointwise_transfer_order(V_f_ref_vel, V_f_cur_vel)
    fluid_cur_vel_to_ref_vel = base._build_pointwise_transfer_order(V_f_cur_vel, V_f_ref_vel)

    fdim_f_ref = msh_f_ref.topology.dim - 1
    fdim_f_cur = msh_f_cur.topology.dim - 1
    dofs_f_ref_interface, coords_f_ref_interface = base._boundary_dof_info(
        V_f_ref_geom, fdim_f_ref, tags_f_ref.facets["interface"]
    )
    dofs_f_cur_interface, _ = base._boundary_dof_info(V_f_cur_scalar, fdim_f_cur, tags_f_cur.facets["interface"])
    x_ref_trace = np.asarray(coords_f_ref_interface[:, 0], dtype=float)
    pts_ref_trace = np.column_stack([x_ref_trace, np.full_like(x_ref_trace, float(params.y_fluid1))])
    locate_pad = 1.0e-10
    pts_f_ref_loc = np.column_stack([x_ref_trace, np.full_like(x_ref_trace, float(params.y_fluid1) - locate_pad)])
    cells_f_ref_interface = base._locate_cells_for_points(msh_f_ref, pts_f_ref_loc)
    cells_f_cur_interface = base._locate_cells_for_points(msh_f_cur, pts_f_ref_loc)

    dofs_f_ref_zero = np.concatenate(
        [
            fem.locate_dofs_topological(V_f_ref_geom, fdim_f_ref, tags_f_ref.facets["left"]),
            fem.locate_dofs_topological(V_f_ref_geom, fdim_f_ref, tags_f_ref.facets["right"]),
            fem.locate_dofs_topological(V_f_ref_geom, fdim_f_ref, tags_f_ref.facets["bottom"]),
        ]
    )
    dofs_f_ref_top = np.asarray(
        fem.locate_dofs_topological(V_f_ref_geom, fdim_f_ref, tags_f_ref.facets["interface"]),
        dtype=np.int32,
    )
    bc_f_ref_zero = fem.dirichletbc(fem.Function(V_f_ref_geom), dofs_f_ref_zero)

    V_f_sub0, _ = W_f_cur.sub(0).collapse()
    dofs_f_cur_left = fem.locate_dofs_topological((W_f_cur.sub(0), V_f_sub0), fdim_f_cur, tags_f_cur.facets["left"])
    dofs_f_cur_right = fem.locate_dofs_topological((W_f_cur.sub(0), V_f_sub0), fdim_f_cur, tags_f_cur.facets["right"])
    dofs_f_cur_bottom = fem.locate_dofs_topological((W_f_cur.sub(0), V_f_sub0), fdim_f_cur, tags_f_cur.facets["bottom"])
    Q_f_sub1, _ = W_f_cur.sub(1).collapse()
    dofs_f_cur_p_pin = fem.locate_dofs_geometrical(
        (W_f_cur.sub(1), Q_f_sub1),
        lambda x: np.logical_and(np.isclose(x[0], params.x0), np.isclose(x[1], params.y_fluid0)),
    )

    wall_bc_fun = fem.Function(V_f_cur_vel)
    wall_bc_fun.x.array[:] = 0.0
    wall_bc_fun.x.scatter_forward()

    inflow_fun = fem.Function(V_f_cur_vel)

    def _inflow_cb(x):
        out = np.zeros((2, x.shape[1]), dtype=float)
        out[1, :] = 4.0 * float(params.v_in) * x[0] * (1.0 - x[0])
        return out

    inflow_fun.interpolate(_inflow_cb)
    inflow_fun.x.scatter_forward()
    p_pin = fem.Function(Q_f_sub1)
    p_pin.x.array[:] = 0.0
    p_pin.x.scatter_forward()
    bc_f_cur_left = fem.dirichletbc(wall_bc_fun, dofs_f_cur_left, W_f_cur.sub(0))
    bc_f_cur_right = fem.dirichletbc(wall_bc_fun, dofs_f_cur_right, W_f_cur.sub(0))
    bc_f_cur_bottom = fem.dirichletbc(inflow_fun, dofs_f_cur_bottom, W_f_cur.sub(0))
    bc_f_cur_p_pin = fem.dirichletbc(p_pin, dofs_f_cur_p_pin, W_f_cur.sub(1))
    bcs_f_cur = [bc_f_cur_left, bc_f_cur_right, bc_f_cur_bottom, bc_f_cur_p_pin]

    x_ref_samples = np.linspace(params.x0, params.x1, int(args.interface_samples), dtype=float)
    interface_eval_pad = 1.0e-9
    petsc_options = base._petsc_options(args.factor_solver)
    t0 = time.time()
    current_L = float(args.L0)
    window_history: list[dict[str, float]] = []
    mms_sample_every = max(int(args.mms_sample_every), 0)
    fluid_sample_csv = outdir / "mms_fluid_samples.csv"
    fluid_sample_fieldnames = ["step", "time", "x_ref", "y_ref", "x_phys", "y_phys", "vx", "vy", "p"]
    fluid_sample_pts_ref = None
    fluid_sample_ref_cells = None
    if mms_sample_every > 0:
        fluid_sample_pts_ref = base._make_reference_tensor_grid(
            x0=params.x0,
            x1=params.x1,
            y0=params.y_fluid0,
            y1=params.y_fluid1,
            nx=int(args.mms_fluid_nx),
            ny=int(args.mms_fluid_ny),
        )
        fluid_sample_ref_cells = base._locate_cells_for_points(msh_f_ref, fluid_sample_pts_ref)
        if MPI.COMM_WORLD.rank == 0 and fluid_sample_csv.exists():
            fluid_sample_csv.unlink()

    participant = PreCICEPointParticipant(
        participant_name="FluidSolver",
        config_file=args.precice_config,
        mesh_name="Fluid-Interface-Mesh",
        coordinates=pts_ref_trace,
        read_fields=("Displacement", "SkeletonVelocity", "DarcyFlux", "PorePressure"),
        write_fields=("FluidVelocity", "FluidTraction", "RobinL"),
    )
    zero_vec = np.zeros((len(x_ref_trace), 2), dtype=float)
    zero_scalar = np.zeros((len(x_ref_trace),), dtype=float)
    dt_precice = min(
        float(dt),
        float(
            participant.initialize(
                initial_write_data={
                    "FluidVelocity": zero_vec,
                    "FluidTraction": zero_scalar,
                    "RobinL": np.full((len(x_ref_trace),), float(current_L), dtype=float),
                }
            )
        ),
    )

    accepted_windows = 0
    time_value = 0.0
    iterations_total = 0
    iterations_in_window = 0
    last_fluid_velocity = zero_vec.copy()
    last_fluid_traction = zero_scalar.copy()
    last_disp = zero_vec.copy()
    last_xi = zero_vec.copy()
    last_q = zero_vec.copy()
    last_p_pore = zero_scalar.copy()
    last_emc_rel = float("nan")
    last_epc_rel = float("nan")
    last_robin_l_candidate = float(current_L)

    try:
        while participant.is_coupling_ongoing():
            if participant.requires_writing_checkpoint():
                participant.store_checkpoint({"L": float(current_L)}, time=time_value, time_window=accepted_windows)

            disp_vals = np.asarray(participant.read("Displacement", dt_precice), dtype=float).reshape((-1, 2))
            xi_vals = np.asarray(participant.read("SkeletonVelocity", dt_precice), dtype=float).reshape((-1, 2))
            q_vals = np.asarray(participant.read("DarcyFlux", dt_precice), dtype=float).reshape((-1, 2))
            p_pore_vals = np.asarray(participant.read("PorePressure", dt_precice), dtype=float).reshape((-1,))

            base._set_boundary_vector_values(eta_top_ref, dofs_f_ref_interface, disp_vals)
            bc_f_ref_top = fem.dirichletbc(eta_top_ref, dofs_f_ref_top)

            eta_f_trial = ufl.TrialFunction(V_f_ref_geom)
            chi = ufl.TestFunction(V_f_ref_geom)
            zero_geom_rhs = fem.Constant(msh_f_ref, np.array((0.0, 0.0), dtype=np.float64))
            problem_geom = base._linear_problem(
                fem.form(ufl.inner(ufl.grad(eta_f_trial), ufl.grad(chi)) * dx_f_ref),
                fem.form(ufl.inner(zero_geom_rhs, chi) * dx_f_ref),
                bcs=[bc_f_ref_zero, bc_f_ref_top],
                u=eta_f_ref_new,
                petsc_options=petsc_options,
                petsc_options_prefix=f"b7_precice_fluid_geom_step{accepted_windows:04d}_iter{iterations_in_window:02d}_",
            )
            problem_geom.solve()
            eta_f_ref_new.x.scatter_forward()
            base._ensure_finite("eta_f_ref_new", eta_f_ref_new.x.array)
            w_f_ref.x.array[:] = (eta_f_ref_new.x.array - eta_f_ref_old.x.array) / dt
            w_f_ref.x.scatter_forward()
            base._ensure_finite("w_f_ref", w_f_ref.x.array)

            base._update_current_mesh(msh_f_cur, fluid_ref_coords, eta_f_ref_old, V_f_ref_geom, fluid_vertex_order)
            grad_eta_ref.interpolate(fem.Expression(ufl.grad(eta_f_ref_old), base._interpolation_points(T_f_ref.element)))
            grad_eta_ref.x.scatter_forward()
            geom_old_trace = base._sample_interface_geometry(
                eta_f_ref_old,
                grad_eta_ref,
                x_ref_trace,
                params.y_fluid1,
                cells=cells_f_ref_interface,
            )
            grad_v_cur.interpolate(fem.Expression(ufl.grad(v_f_cur_old), base._interpolation_points(T_f_cur.element)))
            grad_v_cur.x.scatter_forward()
            grad_v_old_vals = base._eval_function_at_points(v_f_cur_old, geom_old_trace["pts_phys"], cells=cells_f_cur_interface)
            grad_v_old_tensor = base._eval_function_at_points(grad_v_cur, geom_old_trace["pts_phys"], cells=cells_f_cur_interface).reshape((-1, 2, 2))
            p_f_old_vals = base._eval_function_at_points(p_f_cur_old, geom_old_trace["pts_phys"], cells=cells_f_cur_interface).reshape((-1,))
            Dv_old = 0.5 * (grad_v_old_tensor + np.transpose(grad_v_old_tensor, axes=(0, 2, 1)))
            traction_guess_vals = -p_f_old_vals + 2.0 * float(params.mu_f) * np.einsum(
                "ni,nij,nj->n",
                geom_old_trace["n_f"],
                Dv_old,
                geom_old_trace["n_f"],
            )

            base._update_current_mesh(msh_f_cur, fluid_ref_coords, eta_f_ref_new, V_f_ref_geom, fluid_vertex_order)
            base._copy_coefficients_by_order(w_f_cur, w_f_ref, fluid_ref_geom_to_cur_vel)
            base._copy_coefficients_by_order(v_f_cur_old_on_new, v_f_ref_old, fluid_ref_vel_to_cur_vel)

            grad_eta_ref.interpolate(fem.Expression(ufl.grad(eta_f_ref_new), base._interpolation_points(T_f_ref.element)))
            grad_eta_ref.x.scatter_forward()
            geom_new_trace = base._sample_interface_geometry(
                eta_f_ref_new,
                grad_eta_ref,
                x_ref_trace,
                params.y_fluid1,
                cells=cells_f_ref_interface,
            )
            xi_n_vals = np.einsum("ni,ni->n", xi_vals, geom_new_trace["n_f"])
            xi_t_vals = np.einsum("ni,ni->n", xi_vals, geom_new_trace["tau"])
            q_n_vals = np.einsum("ni,ni->n", q_vals, geom_new_trace["n_f"])
            base._set_boundary_scalar_values(xi_n_f_cur, dofs_f_cur_interface, xi_n_vals)
            base._set_boundary_scalar_values(xi_t_f_cur, dofs_f_cur_interface, xi_t_vals)
            base._set_boundary_scalar_values(q_n_f_cur, dofs_f_cur_interface, q_n_vals)
            base._set_boundary_scalar_values(traction_f_cur, dofs_f_cur_interface, traction_guess_vals)

            (v_trial, pF_trial) = ufl.TrialFunctions(W_f_cur)
            (phi_v, psi_p) = ufl.TestFunctions(W_f_cur)
            n_f_cur = ufl.FacetNormal(msh_f_cur)
            tau_f_cur = ufl.as_vector((n_f_cur[1], -n_f_cur[0]))
            advector = v_f_cur_old_on_new - w_f_cur

            a_fluid = (
                (float(params.rho_f) / dt) * ufl.inner(v_trial, phi_v)
                + float(params.rho_f) * ufl.inner(ufl.grad(v_trial) * advector, phi_v)
                + 2.0 * float(params.mu_f) * ufl.inner(base._eps(v_trial), base._eps(phi_v))
                - pF_trial * ufl.div(phi_v)
                + psi_p * ufl.div(v_trial)
            ) * dx_f_cur
            a_fluid += float(current_L) * ufl.dot(v_trial, n_f_cur) * ufl.dot(phi_v, n_f_cur) * ds_f_cur(tags_f_cur.ids["interface"])
            a_fluid += float(params.gamma) * ufl.dot(v_trial, tau_f_cur) * ufl.dot(phi_v, tau_f_cur) * ds_f_cur(
                tags_f_cur.ids["interface"]
            )

            L_fluid = (float(params.rho_f) / dt) * ufl.inner(v_f_cur_old_on_new, phi_v) * dx_f_cur
            L_fluid += (traction_f_cur + float(current_L) * (xi_n_f_cur + q_n_f_cur)) * ufl.dot(phi_v, n_f_cur) * ds_f_cur(
                tags_f_cur.ids["interface"]
            )
            L_fluid += float(params.gamma) * xi_t_f_cur * ufl.dot(phi_v, tau_f_cur) * ds_f_cur(tags_f_cur.ids["interface"])

            problem_f = base._linear_problem(
                fem.form(a_fluid),
                fem.form(L_fluid),
                bcs=bcs_f_cur,
                u=w_f_cur_mixed,
                petsc_options=petsc_options,
                petsc_options_prefix=f"b7_precice_fluid_step{accepted_windows:04d}_iter{iterations_in_window:02d}_",
            )
            problem_f.solve()
            w_f_cur_mixed.x.scatter_forward()
            base._ensure_finite("fluid_mixed_solution", w_f_cur_mixed.x.array)
            base._extract_subfunction(w_f_cur_mixed, V_f_cur_vel, map_f_vel, target=v_f_cur_new)
            base._extract_subfunction(w_f_cur_mixed, Q_f_cur_pres, map_f_pres, target=p_f_cur_new)
            base._copy_coefficients_by_order(v_f_ref_new, v_f_cur_new, fluid_cur_vel_to_ref_vel)

            grad_v_cur.interpolate(fem.Expression(ufl.grad(v_f_cur_new), base._interpolation_points(T_f_cur.element)))
            grad_v_cur.x.scatter_forward()
            pts_fluid_eval = _clip_probe_points(
                geom_new_trace["pts_phys"] - interface_eval_pad * geom_new_trace["n_f"],
                x0=params.x0,
                x1=params.x1,
            )
            fluid_velocity_vals = base._eval_function_at_points(v_f_cur_new, pts_fluid_eval, cells=cells_f_cur_interface)
            grad_v_new_tensor = base._eval_function_at_points(grad_v_cur, pts_fluid_eval, cells=cells_f_cur_interface).reshape((-1, 2, 2))
            p_f_new_vals = base._eval_function_at_points(p_f_cur_new, pts_fluid_eval, cells=cells_f_cur_interface).reshape((-1,))
            Dv_new = 0.5 * (grad_v_new_tensor + np.transpose(grad_v_new_tensor, axes=(0, 2, 1)))
            fluid_traction_vals = -p_f_new_vals + 2.0 * float(params.mu_f) * np.einsum(
                "ni,nij,nj->n",
                geom_new_trace["n_f"],
                Dv_new,
                geom_new_trace["n_f"],
            )
            robin_l_candidate, iter_emc_rel, iter_epc_rel = base._dynamic_L3_update(
                current_L,
                xi_vals=xi_vals,
                q_vals=q_vals,
                fluid_velocity_vals=fluid_velocity_vals,
                fluid_traction_vals=fluid_traction_vals,
                p_pore_vals=p_pore_vals,
                n_f=geom_new_trace["n_f"],
                Jg=geom_new_trace["Jg"],
                x_ref_trace=geom_new_trace["x_ref"],
                x_ref_samples=x_ref_samples,
            )

            participant.write("FluidVelocity", fluid_velocity_vals)
            participant.write("FluidTraction", fluid_traction_vals)
            participant.write("RobinL", np.full((len(x_ref_trace),), float(robin_l_candidate), dtype=float))
            last_disp = np.asarray(disp_vals, dtype=float).copy()
            last_xi = np.asarray(xi_vals, dtype=float).copy()
            last_q = np.asarray(q_vals, dtype=float).copy()
            last_p_pore = np.asarray(p_pore_vals, dtype=float).copy()
            last_fluid_velocity = np.asarray(fluid_velocity_vals, dtype=float).copy()
            last_fluid_traction = np.asarray(fluid_traction_vals, dtype=float).copy()
            last_robin_l_candidate = float(robin_l_candidate)
            last_emc_rel = float(iter_emc_rel)
            last_epc_rel = float(iter_epc_rel)

            dt_precice = min(float(dt), float(participant.advance(dt_precice)))
            iterations_total += 1
            iterations_in_window += 1

            if MPI.COMM_WORLD.rank == 0:
                print(
                    "[seboldt-fluid-precice] "
                    f"window={accepted_windows + 1}/{num_steps} "
                    f"iter={iterations_in_window} "
                    f"L={current_L:.6e} "
                    f"disp_y=[{float(np.min(disp_vals[:, 1])):.3e}, {float(np.max(disp_vals[:, 1])):.3e}] "
                    f"traction=[{float(np.min(fluid_traction_vals)):.3e}, {float(np.max(fluid_traction_vals)):.3e}]",
                    flush=True,
                )

            if participant.requires_reading_checkpoint():
                checkpoint = participant.retrieve_checkpoint()
                current_L = float(checkpoint.payload["L"])
                time_value = float(checkpoint.time)
                accepted_windows = int(checkpoint.time_window)
                iterations_in_window = 0
                last_robin_l_candidate = float(current_L)
                continue

            if participant.is_time_window_complete():
                current_L = float(last_robin_l_candidate)

                base._copy_coefficients(eta_f_ref_old, eta_f_ref_new)
                base._copy_coefficients(v_f_ref_old, v_f_ref_new)
                base._copy_coefficients(v_f_cur_old, v_f_cur_new)
                base._copy_coefficients(p_f_cur_old, p_f_cur_new)

                accepted_windows += 1
                time_value += float(dt)
                if mms_sample_every > 0 and fluid_sample_pts_ref is not None and fluid_sample_ref_cells is not None:
                    if accepted_windows % mms_sample_every == 0 or accepted_windows == num_steps:
                        fluid_mesh_disp_vals = base._eval_function_at_points(
                            eta_f_ref_old,
                            fluid_sample_pts_ref,
                            cells=fluid_sample_ref_cells,
                        )
                        fluid_phys_pts = fluid_sample_pts_ref + fluid_mesh_disp_vals[:, :2]
                        fluid_phys_cells = base._locate_cells_for_points(msh_f_cur, fluid_phys_pts)
                        fluid_vel_vals = base._eval_function_at_points(
                            v_f_cur_old,
                            fluid_phys_pts,
                            cells=fluid_phys_cells,
                        )
                        fluid_pressure_vals = base._eval_function_at_points(
                            p_f_cur_old,
                            fluid_phys_pts,
                            cells=fluid_phys_cells,
                        ).reshape((-1,))
                        base._ensure_finite("mms_fluid_mesh_disp_vals", fluid_mesh_disp_vals)
                        base._ensure_finite("mms_fluid_phys_pts", fluid_phys_pts)
                        base._ensure_finite("mms_fluid_vel_vals", fluid_vel_vals)
                        base._ensure_finite("mms_fluid_pressure_vals", fluid_pressure_vals)
                        fluid_rows = base._space_time_sample_rows(
                            step=accepted_windows,
                            time_value=time_value,
                            pts_ref=fluid_sample_pts_ref,
                            pts_phys=fluid_phys_pts,
                            field_values={
                                "vx": fluid_vel_vals[:, 0],
                                "vy": fluid_vel_vals[:, 1],
                                "p": fluid_pressure_vals,
                            },
                        )
                        if MPI.COMM_WORLD.rank == 0:
                            base._append_dict_rows_csv(
                                fluid_sample_csv,
                                fieldnames=fluid_sample_fieldnames,
                                rows=fluid_rows,
                            )
                window_history.append(
                    {
                        "step": float(accepted_windows),
                        "time": float(time_value),
                        "L": float(current_L),
                        "e_mc_rel": float(last_emc_rel),
                        "e_pc_rel": float(last_epc_rel),
                        "precice_iters": float(iterations_in_window),
                        "traction_max": float(np.max(fluid_traction_vals)),
                        "velocity_y_max": float(np.max(fluid_velocity_vals[:, 1])),
                    }
                )
                if MPI.COMM_WORLD.rank == 0 and (
                    accepted_windows % max(int(args.report_every), 1) == 0 or accepted_windows == num_steps
                ):
                    print(
                        f"[seboldt-fluid-precice] step={accepted_windows}/{num_steps} "
                        f"t={time_value:.4f} L_next={current_L:.6e} precice_iters={iterations_in_window} "
                        f"e_mc={last_emc_rel:.3e} e_pc={last_epc_rel:.3e} "
                        f"traction_max={float(np.max(fluid_traction_vals)):.6e}",
                        flush=True,
                    )
                iterations_in_window = 0
    finally:
        participant.finalize()

    with (outdir / "window_history.json").open("w", encoding="utf-8") as handle:
        json.dump(window_history, handle, indent=2)

    summary = {
        "model": "seboldt_partitioned_moving_linear_precice_fluid",
        "paper": "Seboldt et al. 2021 Example 2",
        "participant": "FluidSolver",
        "kappa": float(args.kappa),
        "nx": int(nx),
        "ny_fluid": int(ny_fluid),
        "dt": float(dt),
        "t_final": float(t_final),
        "num_steps": int(num_steps),
        "accepted_windows": int(accepted_windows),
        "mms_sample_every": int(mms_sample_every),
        "mms_fluid_nx": int(args.mms_fluid_nx),
        "mms_fluid_ny": int(args.mms_fluid_ny),
        "mms_fluid_samples_file": str(fluid_sample_csv) if mms_sample_every > 0 else "",
        "iterations_total": int(iterations_total),
        "final_L_write": float(current_L),
        "runtime_seconds": float(time.time() - t0),
        "last_e_mc_rel": float(last_emc_rel),
        "last_e_pc_rel": float(last_epc_rel),
        "last_fluid_traction_min": float(np.min(last_fluid_traction)) if last_fluid_traction.size else 0.0,
        "last_fluid_traction_max": float(np.max(last_fluid_traction)) if last_fluid_traction.size else 0.0,
        "last_fluid_velocity_y_max": float(np.max(last_fluid_velocity[:, 1])) if last_fluid_velocity.size else 0.0,
        "last_disp_y_max": float(np.max(last_disp[:, 1])) if last_disp.size else 0.0,
        "last_xi_y_max": float(np.max(last_xi[:, 1])) if last_xi.size else 0.0,
        "last_q_y_max": float(np.max(last_q[:, 1])) if last_q.size else 0.0,
        "last_p_pore_max": float(np.max(last_p_pore)) if last_p_pore.size else 0.0,
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--precice-config", required=True)
    ap.add_argument("--kappa", type=float, default=1.0e-3)
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--dt", type=float, default=1.0e-3)
    ap.add_argument("--t-final", type=float, default=3.0)
    ap.add_argument("--L0", type=float, default=2000.0)
    ap.add_argument("--interface-samples", type=int, default=201)
    ap.add_argument("--profile-samples", type=int, default=401)
    ap.add_argument("--factor-solver", default="mumps")
    ap.add_argument("--report-every", type=int, default=100)
    ap.add_argument("--mms-sample-every", type=int, default=0)
    ap.add_argument("--mms-fluid-nx", type=int, default=21)
    ap.add_argument("--mms-fluid-ny", type=int, default=21)
    ap.add_argument("--mms-solid-nx", type=int, default=21)
    ap.add_argument("--mms-solid-ny", type=int, default=11)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    summary = solve_participant(args)
    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
