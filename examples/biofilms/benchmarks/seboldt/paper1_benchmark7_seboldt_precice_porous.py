#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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


def _copy_ref_scalar_to_cur(dst, src) -> None:
    dst.x.array[:] = src.x.array
    dst.x.scatter_forward()


def _write_top_center_history(outdir: Path, rows: list[dict[str, float]]) -> None:
    with (outdir / "top_edge_center_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "time",
                "x_ref",
                "y_ref",
                "uy_top_center",
                "y_top_center",
                "L",
                "precice_iters",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _clip_probe_points(points: np.ndarray, *, x0: float, x1: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float).copy()
    pts[:, 0] = np.clip(pts[:, 0], float(x0) + 1.0e-10, float(x1) - 1.0e-10)
    return pts


def solve_participant(args) -> dict[str, float | int | str]:
    params = base.Example2Params()
    nx = int(args.nx)
    ny_solid = int(round((params.y_solid1 - params.y_solid0) / ((params.x1 - params.x0) / nx)))
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
            "[seboldt-porous-precice] start "
            f"kappa={float(args.kappa):.6e} nx={nx} dt={dt:.6e} t_final={t_final:.6e} "
            f"factor_solver={args.factor_solver!r} outdir={outdir}",
            flush=True,
        )
        if compiler_updates:
            print(
                "[seboldt-porous-precice] compiler-fallback "
                + " ".join(f"{k}={v}" for k, v in compiler_updates.items()),
                flush=True,
            )
        if _RUNTIME_ENV_UPDATES:
            print(
                "[seboldt-porous-precice] runtime-env "
                + " ".join(f"{k}={v}" for k, v in _RUNTIME_ENV_UPDATES.items()),
                flush=True,
            )

    msh_s_ref = base._build_rect_mesh(params.x0, params.y_solid0, params.x1, params.y_solid1, nx, ny_solid)
    msh_s_cur = base._build_rect_mesh(params.x0, params.y_solid0, params.x1, params.y_solid1, nx, ny_solid)
    tags_s_ref = base._build_boundary_markers(msh_s_ref, "solid", params)
    tags_s_cur = base._build_boundary_markers(msh_s_cur, "solid", params)
    ds_s_ref = ufl.Measure("ds", domain=msh_s_ref, subdomain_data=tags_s_ref.meshtags)
    ds_s_cur = ufl.Measure("ds", domain=msh_s_cur, subdomain_data=tags_s_cur.meshtags)
    dx_s_ref = ufl.dx(domain=msh_s_ref)
    dx_s_cur = ufl.dx(domain=msh_s_cur)

    V_s_ref = fem.functionspace(msh_s_ref, ("Lagrange", 2, (2,)))
    V_s_ref_scalar = fem.functionspace(msh_s_ref, ("Lagrange", 2))
    Q_s_ref = fem.functionspace(msh_s_ref, ("DG", 1))
    T_s_ref = fem.functionspace(msh_s_ref, ("DG", 1, (2, 2)))

    V_s_cur_vis = fem.functionspace(msh_s_cur, ("Lagrange", 2, (2,)))
    V_s_cur_scalar = fem.functionspace(msh_s_cur, ("Lagrange", 2))
    cell_s = msh_s_cur.ufl_cell().cellname()
    rt_el = basix.ufl.element("RT", cell_s, 1)
    dg1_el = basix.ufl.element("DG", cell_s, 1)
    W_s_cur = fem.functionspace(msh_s_cur, basix.ufl.mixed_element([rt_el, dg1_el]))
    Q_s_cur_flux, map_s_flux = W_s_cur.sub(0).collapse()
    Q_s_cur_pres, map_s_pres = W_s_cur.sub(1).collapse()

    eta_s_ref_old = fem.Function(V_s_ref, name="eta_s_ref_old")
    xi_s_ref_old = fem.Function(V_s_ref, name="xi_s_ref_old")
    p_s_ref_old = fem.Function(Q_s_ref, name="p_s_ref_old")
    eta_s_cur = fem.Function(V_s_cur_vis, name="eta_s_cur")
    xi_s_cur = fem.Function(V_s_cur_vis, name="xi_s_cur")
    q_s_cur_old = fem.Function(Q_s_cur_flux, name="q_s_cur_old")
    p_s_cur_old = fem.Function(Q_s_cur_pres, name="p_s_cur_old")
    p_s_cur_old_on_new = fem.Function(Q_s_cur_pres, name="p_s_cur_old_on_new")

    traction_ref_old = fem.Function(V_s_ref_scalar)
    vn_ref_old = fem.Function(V_s_ref_scalar)
    vt_ref_old = fem.Function(V_s_ref_scalar)
    qn_ref_old = fem.Function(V_s_ref_scalar)
    traction_s_cur = fem.Function(V_s_cur_scalar)
    vn_s_cur = fem.Function(V_s_cur_scalar)

    grad_eta_ref = fem.Function(T_s_ref, name="grad_eta_ref")
    eta_s_ref_new = fem.Function(V_s_ref, name="eta_s_ref_new")
    xi_s_ref_new = fem.Function(V_s_ref, name="xi_s_ref_new")
    w_s_cur = fem.Function(W_s_cur, name="w_s_cur")
    q_s_cur_new = fem.Function(Q_s_cur_flux, name="q_s_cur_new")
    p_s_cur_new = fem.Function(Q_s_cur_pres, name="p_s_cur_new")
    p_s_ref_new = fem.Function(Q_s_ref, name="p_s_ref_new")

    solid_ref_coords, solid_vertex_order = base._build_vertex_map(msh_s_ref, V_s_ref)

    fdim_s_ref = msh_s_ref.topology.dim - 1
    fdim_s_cur = msh_s_cur.topology.dim - 1
    dofs_s_ref_interface, coords_s_ref_interface = base._boundary_dof_info(
        V_s_ref_scalar, fdim_s_ref, tags_s_ref.facets["interface"]
    )
    dofs_s_cur_interface, _ = base._boundary_dof_info(V_s_cur_scalar, fdim_s_cur, tags_s_cur.facets["interface"])
    x_ref_trace = np.asarray(coords_s_ref_interface[:, 0], dtype=float)
    pts_ref_trace = np.column_stack([x_ref_trace, np.full_like(x_ref_trace, float(params.y_solid0))])
    locate_pad = 1.0e-10
    pts_s_ref_loc = np.column_stack([x_ref_trace, np.full_like(x_ref_trace, float(params.y_solid0) + locate_pad)])
    cells_s_ref_interface = base._locate_cells_for_points(msh_s_ref, pts_s_ref_loc)
    cells_s_cur_interface = base._locate_cells_for_points(msh_s_cur, pts_s_ref_loc)
    top_center_ref = np.asarray([[0.5, float(params.y_solid1)]], dtype=float)
    top_center_loc = np.asarray([[0.5, float(params.y_solid1) - locate_pad]], dtype=float)
    top_center_cell = base._locate_cells_for_points(msh_s_ref, top_center_loc)

    dofs_s_ref_lr = np.concatenate(
        [
            fem.locate_dofs_topological(V_s_ref, fdim_s_ref, tags_s_ref.facets["left"]),
            fem.locate_dofs_topological(V_s_ref, fdim_s_ref, tags_s_ref.facets["right"]),
        ]
    )
    zero_vec_ref = fem.Function(V_s_ref)
    bc_s_ref_lr = fem.dirichletbc(zero_vec_ref, dofs_s_ref_lr)

    V_s_sub0, _ = W_s_cur.sub(0).collapse()
    zero_flux = fem.Function(Q_s_cur_flux)
    zero_flux.x.array[:] = 0.0
    zero_flux.x.scatter_forward()
    dofs_s_cur_left = fem.locate_dofs_topological((W_s_cur.sub(0), V_s_sub0), fdim_s_cur, tags_s_cur.facets["left"])
    dofs_s_cur_right = fem.locate_dofs_topological((W_s_cur.sub(0), V_s_sub0), fdim_s_cur, tags_s_cur.facets["right"])
    bc_s_cur_flux_left = fem.dirichletbc(zero_flux, dofs_s_cur_left, W_s_cur.sub(0))
    bc_s_cur_flux_right = fem.dirichletbc(zero_flux, dofs_s_cur_right, W_s_cur.sub(0))
    bcs_s_cur = [bc_s_cur_flux_left, bc_s_cur_flux_right]

    ref_x, ref_eta_y = base._load_reference_curve(float(args.kappa))
    profile_x = np.linspace(params.x0, params.x1, int(args.profile_samples), dtype=float)
    profile_pts = np.column_stack([profile_x, np.full_like(profile_x, 1.25)])
    profile_pts_phys = np.column_stack([profile_x, np.full_like(profile_x, 1.25)])
    x_ref_samples = np.linspace(params.x0, params.x1, int(args.interface_samples), dtype=float)
    interface_eval_pad = 1.0e-9
    petsc_options = base._petsc_options(args.factor_solver)
    l_history: list[dict[str, float]] = []
    progress_history: list[dict[str, float]] = []
    top_center_history: list[dict[str, float]] = []
    mms_sample_every = max(int(args.mms_sample_every), 0)
    solid_sample_csv = outdir / "mms_solid_samples.csv"
    solid_sample_fieldnames = ["step", "time", "x_ref", "y_ref", "x_phys", "y_phys", "ux", "uy", "p_p"]
    solid_sample_pts_ref = None
    solid_sample_ref_cells = None
    if mms_sample_every > 0:
        solid_sample_pts_ref = base._make_reference_tensor_grid(
            x0=params.x0,
            x1=params.x1,
            y0=params.y_solid0,
            y1=params.y_solid1,
            nx=int(args.mms_solid_nx),
            ny=int(args.mms_solid_ny),
        )
        solid_sample_ref_cells = base._locate_cells_for_points(msh_s_ref, solid_sample_pts_ref)
        if MPI.COMM_WORLD.rank == 0 and solid_sample_csv.exists():
            solid_sample_csv.unlink()
    t0 = time.time()

    participant = PreCICEPointParticipant(
        participant_name="PorousSolver",
        config_file=args.precice_config,
        mesh_name="Porous-Interface-Mesh",
        coordinates=pts_ref_trace,
        read_fields=("FluidTraction", "FluidVelocity", "RobinL"),
        write_fields=("Displacement", "SkeletonVelocity", "DarcyFlux", "PorePressure"),
    )
    zero_vec = np.zeros((len(x_ref_trace), 2), dtype=float)
    zero_scalar = np.zeros((len(x_ref_trace),), dtype=float)
    current_L = float(args.L0)
    dt_precice = min(
        float(dt),
        float(
            participant.initialize(
                initial_write_data={
                    "Displacement": zero_vec,
                    "SkeletonVelocity": zero_vec,
                    "DarcyFlux": zero_vec,
                    "PorePressure": zero_scalar,
                }
            )
        ),
    )

    accepted_windows = 0
    time_value = 0.0
    iterations_total = 0
    iterations_in_window = 0
    last_displacement = zero_vec.copy()
    last_skeleton_velocity = zero_vec.copy()
    last_darcy_flux = zero_vec.copy()
    last_pore_pressure = zero_scalar.copy()
    last_fluid_velocity = zero_vec.copy()
    last_fluid_traction = zero_scalar.copy()
    last_emc_rel = float("nan")
    last_epc_rel = float("nan")
    last_robin_l_vals = np.full((len(x_ref_trace),), float(current_L), dtype=float)

    try:
        while participant.is_coupling_ongoing():
            if participant.requires_writing_checkpoint():
                participant.store_checkpoint({"L": float(current_L)}, time=time_value, time_window=accepted_windows)

            fluid_traction_vals = np.asarray(participant.read("FluidTraction", dt_precice), dtype=float).reshape((-1,))
            fluid_velocity_vals = np.asarray(participant.read("FluidVelocity", dt_precice), dtype=float).reshape((-1, 2))
            robin_l_vals = np.asarray(participant.read("RobinL", dt_precice), dtype=float).reshape((-1,))
            last_robin_l_vals = np.asarray(robin_l_vals, dtype=float).copy()

            grad_eta_ref.interpolate(fem.Expression(ufl.grad(eta_s_ref_old), base._interpolation_points(T_s_ref.element)))
            grad_eta_ref.x.scatter_forward()
            geom_old_trace = base._sample_interface_geometry(
                eta_s_ref_old,
                grad_eta_ref,
                x_ref_trace,
                params.y_solid0,
                cells=cells_s_ref_interface,
            )
            pts_old_porous = _clip_probe_points(
                geom_old_trace["pts_phys"] + interface_eval_pad * geom_old_trace["n_f"],
                x0=params.x0,
                x1=params.x1,
            )
            q_old_vals = base._eval_function_at_points(q_s_cur_old, pts_old_porous, cells=cells_s_cur_interface)
            qn_old_vals = np.einsum("ni,ni->n", q_old_vals, geom_old_trace["n_p"])
            vn_old_vals = np.einsum("ni,ni->n", fluid_velocity_vals, geom_old_trace["n_p"])
            vt_old_vals = np.einsum("ni,ni->n", fluid_velocity_vals, geom_old_trace["tau"])

            base._set_boundary_scalar_values(traction_ref_old, dofs_s_ref_interface, fluid_traction_vals)
            base._set_boundary_scalar_values(vn_ref_old, dofs_s_ref_interface, vn_old_vals)
            base._set_boundary_scalar_values(vt_ref_old, dofs_s_ref_interface, vt_old_vals)
            base._set_boundary_scalar_values(qn_ref_old, dofs_s_ref_interface, qn_old_vals)

            eta_trial = ufl.TrialFunction(V_s_ref)
            zeta = ufl.TestFunction(V_s_ref)
            I2 = ufl.Identity(2)
            F_old = I2 + ufl.grad(eta_s_ref_old)
            J_old = ufl.det(F_old)
            FinvT_old = ufl.inv(F_old).T
            tangent_old = ufl.as_vector((1.0 + eta_s_ref_old[0].dx(0), eta_s_ref_old[1].dx(0)))
            Jg_old = ufl.sqrt(ufl.dot(tangent_old, tangent_old))
            tau_old = tangent_old / Jg_old
            n_p_old = ufl.as_vector((tau_old[1], -tau_old[0]))
            xi_trial_unknown = eta_trial / dt
            xi_old_from_eta = eta_s_ref_old / dt

            a_s_ref = (float(params.rho_s) / (dt * dt)) * ufl.inner(eta_trial, zeta) * dx_s_ref
            a_s_ref += (
                2.0 * float(params.mu_p) * ufl.inner(base._eps(eta_trial), base._eps(zeta))
                + float(params.lambda_p) * ufl.div(eta_trial) * ufl.div(zeta)
            ) * dx_s_ref
            a_s_ref += float(current_L) * Jg_old * ufl.dot(xi_trial_unknown, n_p_old) * ufl.dot(zeta, n_p_old) * ds_s_ref(
                tags_s_ref.ids["interface"]
            )
            a_s_ref += float(params.gamma) * Jg_old * ufl.dot(xi_trial_unknown, tau_old) * ufl.dot(zeta, tau_old) * ds_s_ref(
                tags_s_ref.ids["interface"]
            )

            L_s_ref = (float(params.rho_s) / (dt * dt)) * ufl.inner(eta_s_ref_old + dt * xi_s_ref_old, zeta) * dx_s_ref
            if str(args.structure_pressure_coupling) == "linear_divergence":
                L_s_ref += float(params.alpha) * p_s_ref_old * ufl.div(zeta) * dx_s_ref
            else:
                L_s_ref += float(params.alpha) * J_old * p_s_ref_old * ufl.inner(FinvT_old, ufl.grad(zeta)) * dx_s_ref
            L_s_ref += float(current_L) * Jg_old * ufl.dot(xi_old_from_eta, n_p_old) * ufl.dot(zeta, n_p_old) * ds_s_ref(
                tags_s_ref.ids["interface"]
            )
            L_s_ref += float(params.gamma) * Jg_old * ufl.dot(xi_old_from_eta, tau_old) * ufl.dot(zeta, tau_old) * ds_s_ref(
                tags_s_ref.ids["interface"]
            )
            L_s_ref += Jg_old * (traction_ref_old - float(current_L) * (qn_ref_old - vn_ref_old)) * ufl.dot(zeta, n_p_old) * ds_s_ref(
                tags_s_ref.ids["interface"]
            )
            L_s_ref += float(params.gamma) * Jg_old * vt_ref_old * ufl.dot(zeta, tau_old) * ds_s_ref(tags_s_ref.ids["interface"])

            problem_s = base._linear_problem(
                fem.form(a_s_ref),
                fem.form(L_s_ref),
                bcs=[bc_s_ref_lr],
                u=eta_s_ref_new,
                petsc_options=petsc_options,
                petsc_options_prefix=f"b7_precice_porous_struct_step{accepted_windows:04d}_iter{iterations_in_window:02d}_",
            )
            problem_s.solve()
            eta_s_ref_new.x.scatter_forward()
            base._ensure_finite("eta_s_ref_new", eta_s_ref_new.x.array)
            xi_s_ref_new.x.array[:] = (eta_s_ref_new.x.array - eta_s_ref_old.x.array) / dt
            xi_s_ref_new.x.scatter_forward()
            base._ensure_finite("xi_s_ref_new", xi_s_ref_new.x.array)

            base._update_current_mesh(msh_s_cur, solid_ref_coords, eta_s_ref_new, V_s_ref, solid_vertex_order)
            base._copy_coefficients(eta_s_cur, eta_s_ref_new)
            base._copy_coefficients(xi_s_cur, xi_s_ref_new)
            _copy_ref_scalar_to_cur(p_s_cur_old_on_new, p_s_ref_old)

            grad_eta_ref.interpolate(fem.Expression(ufl.grad(eta_s_ref_new), base._interpolation_points(T_s_ref.element)))
            grad_eta_ref.x.scatter_forward()
            geom_new_trace = base._sample_interface_geometry(
                eta_s_ref_new,
                grad_eta_ref,
                x_ref_trace,
                params.y_solid0,
                cells=cells_s_ref_interface,
            )
            base._ensure_finite("geom_new_trace_pts_phys", geom_new_trace["pts_phys"])

            base._set_boundary_scalar_values(traction_s_cur, dofs_s_cur_interface, fluid_traction_vals)
            vn_new_vals = np.einsum("ni,ni->n", fluid_velocity_vals, geom_new_trace["n_p"])
            base._set_boundary_scalar_values(vn_s_cur, dofs_s_cur_interface, vn_new_vals)

            (q_trial, p_trial) = ufl.TrialFunctions(W_s_cur)
            (r_test, phi_test) = ufl.TestFunctions(W_s_cur)
            n_p_cur = ufl.FacetNormal(msh_s_cur)
            xi_n_cur = ufl.dot(xi_s_cur, n_p_cur)

            a_darcy = (
                float(args.kappa) ** (-1.0) * ufl.inner(q_trial, r_test)
                - p_trial * ufl.div(r_test)
                + phi_test * ufl.div(q_trial)
                + (float(params.c0) / dt) * p_trial * phi_test
            ) * dx_s_cur
            a_darcy += (float(params.delta) + float(current_L)) * ufl.dot(q_trial, n_p_cur) * ufl.dot(r_test, n_p_cur) * ds_s_cur(
                tags_s_cur.ids["interface"]
            )

            L_darcy = (float(params.c0) / dt) * p_s_cur_old_on_new * phi_test * dx_s_cur
            L_darcy += -(float(params.alpha) * ufl.div(xi_s_cur) * phi_test) * dx_s_cur
            L_darcy += (traction_s_cur - float(current_L) * (xi_n_cur - vn_s_cur)) * ufl.dot(r_test, n_p_cur) * ds_s_cur(
                tags_s_cur.ids["interface"]
            )

            problem_d = base._linear_problem(
                fem.form(a_darcy),
                fem.form(L_darcy),
                bcs=bcs_s_cur,
                u=w_s_cur,
                petsc_options=petsc_options,
                petsc_options_prefix=f"b7_precice_porous_darcy_step{accepted_windows:04d}_iter{iterations_in_window:02d}_",
            )
            problem_d.solve()
            w_s_cur.x.scatter_forward()
            base._ensure_finite("darcy_mixed_solution", w_s_cur.x.array)
            base._extract_subfunction(w_s_cur, Q_s_cur_flux, map_s_flux, target=q_s_cur_new)
            base._extract_subfunction(w_s_cur, Q_s_cur_pres, map_s_pres, target=p_s_cur_new)
            _copy_ref_scalar_to_cur(p_s_ref_new, p_s_cur_new)

            displacement_vals = base._eval_function_at_points(eta_s_ref_new, pts_ref_trace, cells=cells_s_ref_interface)
            pts_new_porous = _clip_probe_points(
                geom_new_trace["pts_phys"] + interface_eval_pad * geom_new_trace["n_f"],
                x0=params.x0,
                x1=params.x1,
            )
            skeleton_velocity_vals = base._eval_function_at_points(xi_s_cur, pts_new_porous, cells=cells_s_cur_interface)
            darcy_flux_vals = base._eval_function_at_points(q_s_cur_new, pts_new_porous, cells=cells_s_cur_interface)
            pore_pressure_vals = base._eval_function_at_points(p_s_cur_new, pts_new_porous, cells=cells_s_cur_interface).reshape((-1,))

            base._ensure_finite("displacement_vals", displacement_vals)
            base._ensure_finite("skeleton_velocity_vals", skeleton_velocity_vals)
            base._ensure_finite("darcy_flux_vals", darcy_flux_vals)
            base._ensure_finite("pore_pressure_vals", pore_pressure_vals)

            participant.write("Displacement", displacement_vals)
            participant.write("SkeletonVelocity", skeleton_velocity_vals)
            participant.write("DarcyFlux", darcy_flux_vals)
            participant.write("PorePressure", pore_pressure_vals)
            last_displacement = np.asarray(displacement_vals, dtype=float).copy()
            last_skeleton_velocity = np.asarray(skeleton_velocity_vals, dtype=float).copy()
            last_darcy_flux = np.asarray(darcy_flux_vals, dtype=float).copy()
            last_pore_pressure = np.asarray(pore_pressure_vals, dtype=float).copy()
            last_fluid_velocity = np.asarray(fluid_velocity_vals, dtype=float).copy()
            last_fluid_traction = np.asarray(fluid_traction_vals, dtype=float).copy()

            dt_precice = min(float(dt), float(participant.advance(dt_precice)))
            iterations_total += 1
            iterations_in_window += 1

            if MPI.COMM_WORLD.rank == 0:
                print(
                    "[seboldt-porous-precice] "
                    f"window={accepted_windows + 1}/{num_steps} "
                    f"iter={iterations_in_window} "
                    f"L={current_L:.6e} "
                    f"traction=[{float(np.min(fluid_traction_vals)):.3e}, {float(np.max(fluid_traction_vals)):.3e}] "
                    f"uy=[{float(np.min(displacement_vals[:, 1])):.3e}, {float(np.max(displacement_vals[:, 1])):.3e}]",
                    flush=True,
                )

            if participant.requires_reading_checkpoint():
                checkpoint = participant.retrieve_checkpoint()
                current_L = float(checkpoint.payload["L"])
                time_value = float(checkpoint.time)
                accepted_windows = int(checkpoint.time_window)
                iterations_in_window = 0
                last_robin_l_vals[:] = float(current_L)
                continue

            if participant.is_time_window_complete():
                _local_L_unused, last_emc_rel, last_epc_rel = base._dynamic_L3_update(
                    current_L,
                    xi_vals=skeleton_velocity_vals,
                    q_vals=darcy_flux_vals,
                    fluid_velocity_vals=fluid_velocity_vals,
                    fluid_traction_vals=fluid_traction_vals,
                    p_pore_vals=pore_pressure_vals,
                    n_f=geom_new_trace["n_f"],
                    Jg=geom_new_trace["Jg"],
                    x_ref_trace=geom_new_trace["x_ref"],
                    x_ref_samples=x_ref_samples,
                )
                final_robin_l_vals = last_robin_l_vals
                try:
                    final_robin_l_vals = np.asarray(participant.read("RobinL", 0.0), dtype=float).reshape((-1,))
                except Exception:
                    final_robin_l_vals = last_robin_l_vals
                current_L = base._reduce_robin_l(final_robin_l_vals, fallback=current_L)
                last_robin_l_vals = np.asarray(final_robin_l_vals, dtype=float).copy()
                base._copy_coefficients(eta_s_ref_old, eta_s_ref_new)
                base._copy_coefficients(xi_s_ref_old, xi_s_ref_new)
                base._copy_coefficients(q_s_cur_old, q_s_cur_new)
                base._copy_coefficients(p_s_cur_old, p_s_cur_new)
                _copy_ref_scalar_to_cur(p_s_ref_old, p_s_ref_new)

                accepted_windows += 1
                time_value += float(dt)
                if mms_sample_every > 0 and solid_sample_pts_ref is not None and solid_sample_ref_cells is not None:
                    if accepted_windows % mms_sample_every == 0 or accepted_windows == num_steps:
                        solid_disp_vals = base._eval_function_at_points(
                            eta_s_ref_old,
                            solid_sample_pts_ref,
                            cells=solid_sample_ref_cells,
                        )
                        solid_phys_pts = solid_sample_pts_ref + solid_disp_vals[:, :2]
                        solid_phys_cells = base._locate_cells_for_points(msh_s_cur, solid_phys_pts)
                        solid_pore_pressure_vals = base._eval_function_at_points(
                            p_s_cur_old,
                            solid_phys_pts,
                            cells=solid_phys_cells,
                        ).reshape((-1,))
                        base._ensure_finite("mms_solid_disp_vals", solid_disp_vals)
                        base._ensure_finite("mms_solid_phys_pts", solid_phys_pts)
                        base._ensure_finite("mms_solid_pore_pressure_vals", solid_pore_pressure_vals)
                        solid_rows = base._space_time_sample_rows(
                            step=accepted_windows,
                            time_value=time_value,
                            pts_ref=solid_sample_pts_ref,
                            pts_phys=solid_phys_pts,
                            field_values={
                                "ux": solid_disp_vals[:, 0],
                                "uy": solid_disp_vals[:, 1],
                                "p_p": solid_pore_pressure_vals,
                            },
                        )
                        if MPI.COMM_WORLD.rank == 0:
                            base._append_dict_rows_csv(
                                solid_sample_csv,
                                fieldnames=solid_sample_fieldnames,
                                rows=solid_rows,
                            )
                top_center_vals = base._eval_function_at_points(eta_s_ref_old, top_center_ref, cells=top_center_cell)
                uy_top_center = float(np.asarray(top_center_vals, dtype=float).reshape((-1, 2))[0, 1])
                top_center_history.append(
                    {
                        "step": float(accepted_windows),
                        "time": float(time_value),
                        "x_ref": float(top_center_ref[0, 0]),
                        "y_ref": float(top_center_ref[0, 1]),
                        "uy_top_center": float(uy_top_center),
                        "y_top_center": float(top_center_ref[0, 1] + uy_top_center),
                        "L": float(current_L),
                        "precice_iters": float(iterations_in_window),
                    }
                )
                l_history.append(
                    {
                        "step": float(accepted_windows),
                        "time": float(time_value),
                        "L": float(current_L),
                        "e_mc_rel": float(last_emc_rel),
                        "e_pc_rel": float(last_epc_rel),
                    }
                )
                if MPI.COMM_WORLD.rank == 0:
                    print(
                        f"[seboldt-porous-precice] accepted step={accepted_windows}/{num_steps} "
                        f"t={time_value:.4f} uy_top_center={uy_top_center:.6e} "
                        f"y_top_center={top_center_ref[0, 1] + uy_top_center:.6e} "
                        f"L_next={current_L:.6e} e_mc={last_emc_rel:.3e} e_pc={last_epc_rel:.3e} "
                        f"precice_iters={iterations_in_window}",
                        flush=True,
                    )
                    _write_top_center_history(outdir, top_center_history)

                if MPI.COMM_WORLD.rank == 0 and (
                    accepted_windows % max(int(args.report_every), 1) == 0 or accepted_windows == num_steps
                ):
                    report_profiles = base._profile_metrics_for_frames(
                        profile_x=profile_x,
                        profile_pts_ref=profile_pts,
                        profile_pts_phys=profile_pts_phys,
                        eta_ref_fn=eta_s_ref_old,
                        eta_phys_fn=eta_s_cur,
                        ref_x=ref_x,
                        ref_eta_y=ref_eta_y,
                    )
                    report_metrics = report_profiles["metrics_reference"]
                    report_metrics_phys = report_profiles["metrics_physical"]
                    progress_history.append(
                        {
                            "step": float(accepted_windows),
                            "time": float(time_value),
                            "L": float(current_L),
                            "e_mc_rel": float(last_emc_rel),
                            "e_pc_rel": float(last_epc_rel),
                            "outer_sweeps": float(iterations_in_window),
                            "outer_rel_inf": float("nan"),
                            "outer_abs_inf": float("nan"),
                            "outer_omega": float("nan"),
                            "uy_top_center": float(uy_top_center),
                            "y_top_center": float(top_center_ref[0, 1] + uy_top_center),
                            "uy_max": float(report_metrics["num_amplitude"]),
                            "uy_max_physical": float(report_metrics_phys["num_amplitude"]),
                            "rmse": float(report_metrics["rmse"]),
                            "rmse_physical": float(report_metrics_phys["rmse"]),
                            "linf": float(report_metrics["linf"]),
                            "peak_amplitude_relative_error": float(report_metrics["peak_amplitude_relative_error"]),
                            "peak_amplitude_relative_error_physical": float(report_metrics_phys["peak_amplitude_relative_error"]),
                        }
                    )
                    print(
                        f"[seboldt-porous-precice] step={accepted_windows}/{num_steps} "
                        f"t={time_value:.4f} L={current_L:.6e} precice_iters={iterations_in_window} "
                        f"e_mc={last_emc_rel:.3e} e_pc={last_epc_rel:.3e} "
                        f"uy_top_center={uy_top_center:.6e} "
                        f"uy_max={report_metrics['num_amplitude']:.6e} uy_max_phys={report_metrics_phys['num_amplitude']:.6e} "
                        f"rmse={report_metrics['rmse']:.3e} rmse_phys={report_metrics_phys['rmse']:.3e}",
                        flush=True,
                    )
                    base._write_history_csvs(outdir, l_history, progress_history)
                iterations_in_window = 0
    finally:
        participant.finalize()

    final_profiles = base._profile_metrics_for_frames(
        profile_x=profile_x,
        profile_pts_ref=profile_pts,
        profile_pts_phys=profile_pts_phys,
        eta_ref_fn=eta_s_ref_old,
        eta_phys_fn=eta_s_cur,
        ref_x=ref_x,
        ref_eta_y=ref_eta_y,
    )
    eta_y = final_profiles["eta_y_reference"]
    eta_y_phys = final_profiles["eta_y_physical"]
    metrics = final_profiles["metrics_reference"]
    metrics_phys = final_profiles["metrics_physical"]
    top_center_final_vals = base._eval_function_at_points(eta_s_ref_old, top_center_ref, cells=top_center_cell)
    uy_top_center_final = float(np.asarray(top_center_final_vals, dtype=float).reshape((-1, 2))[0, 1])
    base._write_profile_csv(outdir / "uy_profile.csv", profile_x, eta_y, ref_x, ref_eta_y)
    base._write_profile_csv(outdir / "uy_profile_physical.csv", profile_x, eta_y_phys, ref_x, ref_eta_y)
    base._write_history_csvs(outdir, l_history, progress_history)
    if MPI.COMM_WORLD.rank == 0:
        _write_top_center_history(outdir, top_center_history)

    summary = {
        "model": "seboldt_partitioned_moving_linear_precice_porous",
        "paper": "Seboldt et al. 2021 Example 2",
        "participant": "PorousSolver",
        "kappa": float(args.kappa),
        "nx": int(nx),
        "ny_solid": int(ny_solid),
        "dt": float(dt),
        "t_final": float(t_final),
        "num_steps": int(num_steps),
        "accepted_windows": int(accepted_windows),
        "mms_sample_every": int(mms_sample_every),
        "mms_solid_nx": int(args.mms_solid_nx),
        "mms_solid_ny": int(args.mms_solid_ny),
        "mms_solid_samples_file": str(solid_sample_csv) if mms_sample_every > 0 else "",
        "iterations_total": int(iterations_total),
        "final_L_read": float(current_L),
        "runtime_seconds": float(time.time() - t0),
        "last_e_mc_rel": float(last_emc_rel),
        "last_e_pc_rel": float(last_epc_rel),
        "top_center_x_ref": float(top_center_ref[0, 0]),
        "top_center_y_ref": float(top_center_ref[0, 1]),
        "uy_top_center": float(uy_top_center_final),
        "y_top_center": float(top_center_ref[0, 1] + uy_top_center_final),
        **metrics,
        "num_amplitude_physical": float(metrics_phys["num_amplitude"]),
        "peak_amplitude_relative_error_physical": float(metrics_phys["peak_amplitude_relative_error"]),
        "rmse_physical": float(metrics_phys["rmse"]),
        "linf_physical": float(metrics_phys["linf"]),
        "peak_x_error_physical": float(metrics_phys["peak_x_error"]),
        "last_fluid_traction_min": float(np.min(last_fluid_traction)) if last_fluid_traction.size else 0.0,
        "last_fluid_traction_max": float(np.max(last_fluid_traction)) if last_fluid_traction.size else 0.0,
        "last_displacement_y_max": float(np.max(last_displacement[:, 1])) if last_displacement.size else 0.0,
        "last_skeleton_velocity_y_max": float(np.max(last_skeleton_velocity[:, 1])) if last_skeleton_velocity.size else 0.0,
        "last_darcy_flux_y_max": float(np.max(last_darcy_flux[:, 1])) if last_darcy_flux.size else 0.0,
        "last_pore_pressure_max": float(np.max(last_pore_pressure)) if last_pore_pressure.size else 0.0,
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
    ap.add_argument(
        "--structure-pressure-coupling",
        choices=("nonlinear_reference", "linear_divergence"),
        default="nonlinear_reference",
    )
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    summary = solve_participant(args)
    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
