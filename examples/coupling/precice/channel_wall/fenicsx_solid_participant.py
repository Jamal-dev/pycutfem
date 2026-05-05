#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem import petsc as fem_petsc
from fenicsxprecice import Adapter, CouplingMesh
from fenicsxprecice.adapter_core import CouplingBoundaryInterpolation
from mpi4py import MPI


def _build_mesh(*, x0: float, y0: float, x1: float, y1: float, nx: int, ny: int):
    return dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.asarray([float(x0), float(y0)], dtype=np.float64), np.asarray([float(x1), float(y1)], dtype=np.float64)],
        [int(nx), int(ny)],
        cell_type=dmesh.CellType.triangle,
    )


def _boundary_facets(msh, *, x0: float, x1: float, y0: float, y1: float) -> dict[str, np.ndarray]:
    fdim = msh.topology.dim - 1
    tol = 1.0e-12
    return {
        "left": dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], float(x0), atol=tol)),
        "right": dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], float(x1), atol=tol)),
        "top": dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], float(y1), atol=tol)),
        "interface": dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], float(y0), atol=tol)),
    }


def _build_facet_tags(msh, named_facets: dict[str, np.ndarray]):
    fdim = msh.topology.dim - 1
    values = []
    facets = []
    ids = {name: i + 1 for i, name in enumerate(named_facets)}
    for name, entities in named_facets.items():
        arr = np.asarray(entities, dtype=np.int32)
        facets.append(arr)
        values.append(np.full_like(arr, ids[name], dtype=np.int32))
    facet_ids = np.concatenate(facets) if facets else np.zeros((0,), dtype=np.int32)
    facet_values = np.concatenate(values) if values else np.zeros((0,), dtype=np.int32)
    order = np.argsort(facet_ids)
    return dmesh.meshtags(msh, fdim, facet_ids[order], facet_values[order]), ids


def _update_displacement_trace(target, displacement, scalar_space) -> None:
    interpolation_points = getattr(scalar_space.element, "interpolation_points")
    if callable(interpolation_points):
        interpolation_points = interpolation_points()
    expr = fem.Expression(displacement[1], interpolation_points)
    target.interpolate(expr)
    target.x.scatter_forward()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter-config", default="solid-precice-adapter-config.json")
    ap.add_argument("--mesh-name", default="Solid-Interface-Mesh")
    ap.add_argument("--x0", type=float, default=0.0)
    ap.add_argument("--x1", type=float, default=1.0)
    ap.add_argument("--y0", type=float, default=1.0)
    ap.add_argument("--y1", type=float, default=1.25)
    ap.add_argument("--nx", type=int, default=24)
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--young", type=float, default=5.0e3)
    ap.add_argument("--poisson", type=float, default=0.35)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--outdir", default="out/solid")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    msh = _build_mesh(
        x0=float(args.x0),
        y0=float(args.y0),
        x1=float(args.x1),
        y1=float(args.y1),
        nx=int(args.nx),
        ny=int(args.ny),
    )
    named_facets = _boundary_facets(
        msh,
        x0=float(args.x0),
        x1=float(args.x1),
        y0=float(args.y0),
        y1=float(args.y1),
    )
    facet_tags, facet_ids = _build_facet_tags(msh, named_facets)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    dx = ufl.dx(domain=msh)

    V = fem.functionspace(msh, ("Lagrange", 2, (2,)))
    Q = fem.functionspace(msh, ("Lagrange", 2))
    displacement = fem.Function(V, name="displacement")
    pressure_bc = fem.Function(Q, name="pressure_interface")
    disp_y = fem.Function(Q, name="displacement_y")
    displacement.x.array[:] = 0.0
    displacement.x.scatter_forward()
    pressure_bc.x.array[:] = 0.0
    pressure_bc.x.scatter_forward()
    disp_y.x.array[:] = 0.0
    disp_y.x.scatter_forward()

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    lame_lambda = float(args.young) * float(args.poisson) / (
        (1.0 + float(args.poisson)) * (1.0 - 2.0 * float(args.poisson))
    )
    lame_mu = float(args.young) / (2.0 * (1.0 + float(args.poisson)))

    def epsilon(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0 * float(lame_mu) * epsilon(w) + float(lame_lambda) * ufl.div(w) * ufl.Identity(2)

    n = ufl.FacetNormal(msh)
    traction = -pressure_bc * n
    a = ufl.inner(sigma(u), epsilon(v)) * dx
    L = ufl.dot(traction, v) * ds(facet_ids["interface"])

    fdim = msh.topology.dim - 1
    Vx, _ = V.sub(0).collapse()
    Vy, _ = V.sub(1).collapse()
    ux_zero = fem.Function(Vx)
    uy_zero = fem.Function(Vy)
    ux_zero.x.array[:] = 0.0
    uy_zero.x.array[:] = 0.0
    ux_zero.x.scatter_forward()
    uy_zero.x.scatter_forward()
    dofs_left_x = fem.locate_dofs_topological((V.sub(0), Vx), fdim, named_facets["left"])
    dofs_right_x = fem.locate_dofs_topological((V.sub(0), Vx), fdim, named_facets["right"])
    dofs_top_y = fem.locate_dofs_topological((V.sub(1), Vy), fdim, named_facets["top"])
    bcs = [
        fem.dirichletbc(ux_zero, dofs_left_x, V.sub(0)),
        fem.dirichletbc(ux_zero, dofs_right_x, V.sub(0)),
        fem.dirichletbc(uy_zero, dofs_top_y, V.sub(1)),
    ]
    petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}

    adapter = Adapter(
        MPI.COMM_WORLD,
        adapter_config_filename=str(Path(args.adapter_config).resolve()),
        boundary_processing_mode=CouplingBoundaryInterpolation.ADAPTER,
    )
    coupling_mesh = CouplingMesh(
        str(args.mesh_name),
        lambda x, y0=float(args.y0): np.isclose(x[1], y0, atol=1.0e-12),
        read_fields={"Pressure": Q},
        write_fields={"DisplacementY": disp_y},
    )

    time = 0.0
    time_window = 0
    iterations = 0
    accepted_windows = 0
    adapter.initialize([coupling_mesh])
    dt = min(float(args.dt), float(adapter.get_max_time_step_size()))
    try:
        while adapter.is_coupling_ongoing():
            if adapter.requires_writing_checkpoint():
                adapter.store_checkpoint(displacement, time, time_window)
            adapter.read_data(str(args.mesh_name), "Pressure", dt, pressure_bc)
            problem = fem_petsc.LinearProblem(
                fem.form(a),
                fem.form(L),
                bcs=bcs,
                u=displacement,
                petsc_options_prefix="solid_",
                petsc_options=petsc_options,
            )
            problem.solve()
            displacement.x.scatter_forward()
            _update_displacement_trace(disp_y, displacement, Q)
            adapter.write_data(str(args.mesh_name), "DisplacementY", disp_y)
            adapter.advance(dt)
            dt = min(float(args.dt), float(adapter.get_max_time_step_size()))
            iterations += 1
            root = int(MPI.COMM_WORLD.rank) == 0
            if root:
                print(
                    "[solid] "
                    f"iter={iterations} "
                    f"pressure=[{float(np.min(pressure_bc.x.array)):.3e}, {float(np.max(pressure_bc.x.array)):.3e}] "
                    f"disp_y=[{float(np.min(disp_y.x.array)):.3e}, {float(np.max(disp_y.x.array)):.3e}]",
                    flush=True,
                )
            if adapter.requires_reading_checkpoint():
                checkpoint, time, time_window = adapter.retrieve_checkpoint()
                displacement.x.array[:] = checkpoint.x.array
                displacement.x.scatter_forward()
                _update_displacement_trace(disp_y, displacement, Q)
            else:
                if adapter.is_time_window_complete():
                    time += float(args.dt)
                    time_window += 1
                    accepted_windows += 1
    finally:
        adapter.finalize()

    if int(MPI.COMM_WORLD.rank) == 0:
        summary = {
            "participant": "SolidSolver",
            "iterations": int(iterations),
            "accepted_windows": int(accepted_windows),
            "pressure_min": float(np.min(pressure_bc.x.array)) if pressure_bc.x.array.size else 0.0,
            "pressure_max": float(np.max(pressure_bc.x.array)) if pressure_bc.x.array.size else 0.0,
            "disp_y_min": float(np.min(disp_y.x.array)) if disp_y.x.array.size else 0.0,
            "disp_y_max": float(np.max(disp_y.x.array)) if disp_y.x.array.size else 0.0,
        }
        with (outdir / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
