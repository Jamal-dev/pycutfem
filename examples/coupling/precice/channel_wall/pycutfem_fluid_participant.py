#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.coupling import PreCICEPointParticipant
from pycutfem.fem import transform
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    VectorTestFunction,
    VectorTrialFunction,
    div,
    grad,
    inner,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _build_mesh(*, length: float, height: float, nx: int, ny: int) -> tuple[Mesh, DofHandler]:
    nodes, elems, edges, corners = structured_quad(length, height, nx=nx, ny=ny, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    mesh.tag_boundary_edges(
        {
            "inlet": lambda x, y: np.isclose(x, 0.0),
            "outlet": lambda x, y, x1=length: np.isclose(x, x1),
            "bottom": lambda x, y: np.isclose(y, 0.0),
            "interface": lambda x, y, y1=height: np.isclose(y, y1),
        }
    )
    mixed_element = MixedElement(mesh, field_specs={"ux": 2, "uy": 2, "p": 1})
    dh = DofHandler(mixed_element, method="cg")
    dh.tag_dof_by_locator(
        "pressure_pin",
        "p",
        lambda x, y, x1=length: np.isclose(x, x1) and np.isclose(y, 0.0),
        find_first=True,
    )
    return mesh, dh


def _evaluate_scalar_field(
    *,
    dof_handler: DofHandler,
    mesh: Mesh,
    solution: np.ndarray,
    field_name: str,
    point: tuple[float, float],
) -> float:
    x, y = float(point[0]), float(point[1])
    xy = np.asarray([x, y], dtype=float)
    element_id = None
    for elem in mesh.elements_list:
        node_ids = elem.nodes
        coords = mesh.nodes_x_y_pos[list(node_ids)]
        if not (coords[:, 0].min() - 1.0e-12 <= x <= coords[:, 0].max() + 1.0e-12):
            continue
        if not (coords[:, 1].min() - 1.0e-12 <= y <= coords[:, 1].max() + 1.0e-12):
            continue
        try:
            xi, eta = transform.inverse_mapping(mesh, elem.id, xy)
        except Exception:
            continue
        if -1.00001 <= float(xi) <= 1.00001 and -1.00001 <= float(eta) <= 1.00001:
            element_id = int(elem.id)
            break
    if element_id is None:
        raise ValueError(f"Point {point!r} was not located in any element.")
    basis = dof_handler.mixed_element.basis(field_name, xi, eta)[dof_handler.mixed_element.slice(field_name)]
    global_dofs = np.asarray(dof_handler.element_maps[field_name][element_id], dtype=int)
    return float(np.asarray(solution, dtype=float)[global_dofs] @ np.asarray(basis, dtype=float))


def _solve_fluid_pressure(
    *,
    mesh: Mesh,
    dof_handler: DofHandler,
    interface_x: np.ndarray,
    displacement_y: np.ndarray,
    inlet_umax: float,
    velocity_gain: float,
    viscosity: float,
    quad_order: int,
    backend: str,
) -> tuple[np.ndarray, dict[str, float]]:
    displacement_y = np.asarray(displacement_y, dtype=float).reshape(-1)
    x_samples = np.asarray(interface_x, dtype=float).reshape(-1)
    if displacement_y.size != x_samples.size:
        raise ValueError("The incoming interface displacement profile does not match the fluid coupling mesh.")

    vel_space = FunctionSpace("velocity", ["ux", "uy"], dim=1)
    pressure_space = FunctionSpace("pressure", ["p"], dim=0)
    du = VectorTrialFunction(vel_space, dof_handler=dof_handler)
    dp = TrialFunction("p", dof_handler=dof_handler)
    v = VectorTestFunction(vel_space, dof_handler=dof_handler)
    q = TestFunction("p", dof_handler=dof_handler)

    def eps(w):
        return Constant(0.5) * (grad(w) + grad(w).T)

    zero_vec = Constant(np.asarray([0.0, 0.0], dtype=float), dim=1)
    a = (Constant(2.0 * float(viscosity)) * inner(eps(du), eps(v)) - dp * div(v) + q * div(du)) * dx()
    L = inner(zero_vec, v) * dx()

    def inlet_profile(x, y):
        y_arr = np.asarray(y, dtype=float)
        return float(inlet_umax) * 4.0 * y_arr * (1.0 - y_arr)

    def interface_uy(x, y):
        return float(velocity_gain) * np.interp(
            float(x),
            x_samples,
            displacement_y,
            left=float(displacement_y[0]),
            right=float(displacement_y[-1]),
        )

    bcs = [
        BoundaryCondition("ux", "dirichlet", "inlet", inlet_profile),
        BoundaryCondition("uy", "dirichlet", "inlet", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "bottom", lambda x, y: 0.0),
        BoundaryCondition("uy", "dirichlet", "bottom", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "interface", lambda x, y: 0.0),
        BoundaryCondition("uy", "dirichlet", "interface", interface_uy),
        BoundaryCondition("p", "dirichlet", "pressure_pin", lambda x, y: 0.0),
    ]

    matrix, rhs = assemble_form(
        Equation(a, L),
        dof_handler=dof_handler,
        bcs=bcs,
        quad_order=int(quad_order),
        backend=str(backend),
    )
    solution = np.asarray(spla.spsolve(matrix.tocsc(), rhs), dtype=float).reshape(-1)
    probe_y = float(np.max(mesh.nodes_x_y_pos[:, 1])) - 1.0e-10
    pressure_profile = np.asarray(
        [
            _evaluate_scalar_field(
                dof_handler=dof_handler,
                mesh=mesh,
                solution=solution,
                field_name="p",
                point=(float(xv), probe_y),
            )
            for xv in x_samples
        ],
        dtype=float,
    )
    diag = {
        "pressure_min": float(np.min(pressure_profile)),
        "pressure_max": float(np.max(pressure_profile)),
        "pressure_l2": float(np.linalg.norm(pressure_profile)),
        "displacement_min": float(np.min(displacement_y)),
        "displacement_max": float(np.max(displacement_y)),
    }
    return pressure_profile, diag


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--precice-config", default="precice-config.xml")
    ap.add_argument("--mesh-name", default="Fluid-Interface-Mesh")
    ap.add_argument("--length", type=float, default=1.0)
    ap.add_argument("--height", type=float, default=1.0)
    ap.add_argument("--nx", type=int, default=16)
    ap.add_argument("--ny", type=int, default=12)
    ap.add_argument("--interface-points", type=int, default=21)
    ap.add_argument("--inlet-umax", type=float, default=2.0)
    ap.add_argument("--velocity-gain", type=float, default=20.0)
    ap.add_argument("--viscosity", type=float, default=0.05)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--quad-order", type=int, default=4)
    ap.add_argument("--backend", default="jit")
    ap.add_argument("--outdir", default="out/fluid")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    mesh, dof_handler = _build_mesh(
        length=float(args.length),
        height=float(args.height),
        nx=int(args.nx),
        ny=int(args.ny),
    )
    interface_x = np.linspace(0.0, float(args.length), int(args.interface_points), dtype=float)
    interface_coords = np.column_stack([interface_x, np.full_like(interface_x, float(args.height))])

    participant = PreCICEPointParticipant(
        participant_name="FluidSolver",
        config_file=args.precice_config,
        mesh_name=args.mesh_name,
        coordinates=interface_coords,
        read_fields=("DisplacementY",),
        write_fields=("Pressure",),
    )

    time = 0.0
    time_window = 0
    iterations = 0
    accepted_windows = 0
    current_pressure = np.zeros((interface_coords.shape[0],), dtype=float)
    current_displacement = np.zeros((interface_coords.shape[0],), dtype=float)

    dt = min(float(args.dt), participant.initialize())
    try:
        while participant.is_coupling_ongoing():
            if participant.requires_writing_checkpoint():
                participant.store_checkpoint(
                    {"pressure": current_pressure, "displacement": current_displacement},
                    time=time,
                    time_window=time_window,
                )
            current_displacement = participant.read("DisplacementY", dt)
            current_pressure, diag = _solve_fluid_pressure(
                mesh=mesh,
                dof_handler=dof_handler,
                interface_x=interface_x,
                displacement_y=current_displacement,
                inlet_umax=float(args.inlet_umax),
                velocity_gain=float(args.velocity_gain),
                viscosity=float(args.viscosity),
                quad_order=int(args.quad_order),
                backend=str(args.backend),
            )
            participant.write("Pressure", current_pressure)
            dt = min(float(args.dt), participant.advance(dt))
            iterations += 1
            print(
                "[fluid] "
                f"iter={iterations} "
                f"disp=[{diag['displacement_min']:.3e}, {diag['displacement_max']:.3e}] "
                f"pressure=[{diag['pressure_min']:.3e}, {diag['pressure_max']:.3e}]",
                flush=True,
            )
            if participant.requires_reading_checkpoint():
                checkpoint = participant.retrieve_checkpoint()
                current_pressure = np.asarray(checkpoint.payload["pressure"], dtype=float).copy()
                current_displacement = np.asarray(checkpoint.payload["displacement"], dtype=float).copy()
                time = float(checkpoint.time)
                time_window = int(checkpoint.time_window)
            else:
                if participant.is_time_window_complete():
                    time += float(args.dt)
                    time_window += 1
                    accepted_windows += 1
    finally:
        participant.finalize()

    summary = {
        "participant": "FluidSolver",
        "iterations": int(iterations),
        "accepted_windows": int(accepted_windows),
        "pressure_min": float(np.min(current_pressure)) if current_pressure.size else 0.0,
        "pressure_max": float(np.max(current_pressure)) if current_pressure.size else 0.0,
        "pressure_l2": float(np.linalg.norm(current_pressure)),
        "displacement_min": float(np.min(current_displacement)) if current_displacement.size else 0.0,
        "displacement_max": float(np.max(current_displacement)) if current_displacement.size else 0.0,
    }
    with (outdir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
