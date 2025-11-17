#!/usr/bin/env python
"""
Volume-only variant of the Turek–Schafer 2D-2 benchmark that uses a Gmsh mesh
with an explicit cylinder hole instead of CutFEM.

The script generates (or reuses) a triangular mesh of the channel with a
rectangular domain (2.2 x 0.41) and a cylindrical obstacle.  The mesh is
imported with :func:`pycutfem.utils.gmsh_loader.mesh_from_gmsh`, after which a
standard Taylor–Hood discretization with a one-step theta time scheme is
assembled.  Drag and lift coefficients are computed by integrating the fluid
traction over the cylinder boundary using only volume forms (no ghost terms).
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import gmsh
import numba
import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem import transform
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.solvers.nonlinear_solver import (
    NewtonParameters,
    NewtonSolver,
    TimeStepperParameters,
)
from pycutfem.ufl.expressions import (
    Constant,
    FacetNormal,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    div,
    dot,
    grad,
    inner,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dS, dx
from pycutfem.utils.gmsh_loader import mesh_from_gmsh
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
#                        Problem / geometry parameters
# --------------------------------------------------------------------------- #
H = 0.41   # channel height
L = 2.2    # channel length
D = 0.1    # cylinder diameter
CENTER = (0.2, 0.2)
RADIUS = D / 2.0
RHO = 1.0
MU = 1.0e-3
U_MEAN = 1.5
FE_ORDER = 2  # Taylor–Hood (Q2/Q1) equivalent on triangles


def _configure_numba():
    """Set numba to use all available threads."""
    try:
        numba.set_num_threads(os.cpu_count())
        print(f"Numba threads: {numba.get_num_threads()}")
    except (ImportError, AttributeError):
        print("Numba not configured; running in pure Python mode.")


# --------------------------------------------------------------------------- #
#                              Mesh generation
# --------------------------------------------------------------------------- #
def build_turek_channel_mesh(path: Path, mesh_size: float, cell_type: str = "tri", view_mesh: bool = False) -> None:
    """
    Build the Turek benchmark mesh with a cylinder hole using Gmsh.
    The mesh is written to ``path``.
    """
    if cell_type not in {"tri", "quad"}:
        raise ValueError("cell_type must be 'tri' or 'quad'")

    gmsh.initialize()
    try:
        gmsh.model.add("turek_channel_volume")
        rect = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, H)
        cyl = gmsh.model.occ.addDisk(*CENTER, 0.0, RADIUS, RADIUS)
        volumes, _ = gmsh.model.occ.cut(
            [(2, rect)],
            [(2, cyl)],
            removeObject=True,
            removeTool=True,
        )
        if not volumes:
            raise RuntimeError("Boolean cut failed while creating the hole.")
        gmsh.model.occ.synchronize()
        surface_tags = [tag for dim, tag in volumes if dim == 2]
        gmsh.model.addPhysicalGroup(2, surface_tags, tag=1)
        gmsh.model.setPhysicalName(2, 1, "fluid")

        if cell_type == "quad":
            for tag in surface_tags:
                gmsh.model.mesh.setRecombine(2, tag)
            gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal quad
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)   # quad-dominant
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2) # Blossom
        gmsh.model.mesh.setOrder(1)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)

        inlet_edges: list[int] = []
        outlet_edges: list[int] = []
        wall_edges: list[int] = []
        cylinder_edges: list[int] = []
        boundary_entities = gmsh.model.getBoundary(volumes, oriented=False, recursive=False)
        tol = 1e-7
        near = lambda value, target: abs(value - target) <= tol
        for dim, tag in boundary_entities:
            if dim != 1:
                continue
            cx, cy, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
            if near(cx, 0.0):
                inlet_edges.append(tag)
            elif near(cx, L):
                outlet_edges.append(tag)
            elif near(cy, 0.0) or near(cy, H):
                wall_edges.append(tag)
            else:
                cylinder_edges.append(tag)

        def _add_group(name: str, edges: list[int], tag_hint: int) -> None:
            if not edges:
                return
            tag = gmsh.model.addPhysicalGroup(1, edges, tag=tag_hint)
            gmsh.model.setPhysicalName(1, tag, name)

        _add_group("inlet", inlet_edges, 11)
        _add_group("outlet", outlet_edges, 12)
        _add_group("walls", wall_edges, 13)
        _add_group("cylinder", cylinder_edges, 14)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(1)  # enforce linear elements even after meshing
        if cell_type == "quad":
            tri_types = []
            for etype in gmsh.model.mesh.getElementTypes(2):
                name, *_ = gmsh.model.mesh.getElementProperties(etype)
                if "triangle" in name.lower():
                    tri_types.append(etype)
            if tri_types:
                gmsh.model.mesh.removeElements(elementTypes=tri_types)
            gmsh.model.mesh.recombine()

        path.parent.mkdir(parents=True, exist_ok=True)
        if view_mesh:
            try:
                gmsh.fltk.initialize()
                gmsh.fltk.run()
            except Exception:
                print("Gmsh GUI not available; skipping mesh preview.")
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def prepare_mesh(mesh_file: Path | None, mesh_size: float, rebuild: bool, cell_type: str, view_mesh: bool) -> tuple:
    """
    Generate (if needed) and load the Gmsh mesh.
    Returns the in-memory :class:`Mesh` and the path to the mesh file (if kept).
    """
    if mesh_file is not None:
        mesh_file = mesh_file.expanduser().resolve()
        if rebuild or not mesh_file.exists():
            print(f"Generating Gmsh mesh at {mesh_file} (h={mesh_size}, cell_type={cell_type})")
            build_turek_channel_mesh(mesh_file, mesh_size, cell_type, view_mesh=view_mesh)
        else:
            print(f"Reusing existing mesh at {mesh_file}")
        if view_mesh and mesh_file.exists():
            try:
                gmsh.initialize()
                gmsh.open(str(mesh_file))
                gmsh.fltk.run()
            except Exception:
                print("Gmsh GUI not available; skipping mesh preview.")
            finally:
                gmsh.finalize()
        return mesh_from_gmsh(mesh_file), mesh_file

    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "turek_channel.msh"
        print(f"Generating temporary Gmsh mesh (h={mesh_size}, cell_type={cell_type})")
        build_turek_channel_mesh(tmp_path, mesh_size, cell_type, view_mesh=view_mesh)
        mesh = mesh_from_gmsh(tmp_path)
    # tmpdir is cleaned up here; mesh lives on in memory
    return mesh, None


# --------------------------------------------------------------------------- #
#                        Helper evaluation utilities
# --------------------------------------------------------------------------- #
def locate_element(mesh, point: np.ndarray):
    """Locate the element containing ``point`` (returns (eid, xi, eta))."""
    xy = np.asarray(point, dtype=float)
    for elem in mesh.elements_list:
        node_ids = elem.nodes
        coords = mesh.nodes_x_y_pos[list(node_ids)]
        if not (
            coords[:, 0].min() - 1e-12 <= xy[0] <= coords[:, 0].max() + 1e-12
            and coords[:, 1].min() - 1e-12 <= xy[1] <= coords[:, 1].max() + 1e-12
        ):
            continue
        try:
            xi, eta = transform.inverse_mapping(mesh, elem.id, xy)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if -1.0001 <= xi <= 1.0001 and -1.0001 <= eta <= 1.0001:
            return elem.id, xi, eta
    return None, None, None


def evaluate_scalar_field(dof_handler, mesh, field_name: str, func: Function, point):
    """Evaluate a scalar Function at a physical point."""
    eid, xi, eta = locate_element(mesh, point)
    if eid is None:
        return math.nan
    me = dof_handler.mixed_element
    phi = me.basis(field_name, xi, eta)[me.slice(field_name)]
    gdofs = np.asarray(dof_handler.element_maps[field_name][eid], dtype=int)
    vals = func.get_nodal_values(gdofs)
    return float(phi @ vals)


def evaluate_vector_field(dof_handler, mesh, v_func: VectorFunction, point):
    """Evaluate the vector function at a point by evaluating each component."""
    return np.array(
        [evaluate_scalar_field(dof_handler, mesh, field.field_name, field, point) for field in v_func],
        dtype=float,
    )


def nearest_pressure_dof_value(dof_handler: DofHandler, p_func: Function, point: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Evaluate the pressure function at the DOF located closest to ``point``.
    Works robustly even if the requested location is outside the mesh (e.g. inside the cylinder).
    """
    coords = dof_handler.get_dof_coords("p")
    diffs = coords - point
    idx = int(np.argmin(np.einsum("ij,ij->i", diffs, diffs)))
    return float(p_func.nodal_values[idx]), coords[idx]


# --------------------------------------------------------------------------- #
#                            Weak form components
# --------------------------------------------------------------------------- #
def epsilon(u):
    """Symmetric gradient."""
    return 0.5 * (grad(u) + grad(u).T)


def build_volume_forms(
    u_trial,
    v_test,
    p_trial,
    q_test,
    u_k,
    u_n,
    p_k,
    p_n,
    rho_const,
    mu_const,
    dt_const,
    theta_const,
    dx_measure,
):
    """Return (jacobian_form, residual_form) for the standard theta-scheme."""
    a_vol = (
        rho_const * dot(u_trial, v_test) / dt_const
        + theta_const * rho_const * dot(dot(grad(u_k), u_trial), v_test)
        + theta_const * rho_const * dot(dot(grad(u_trial), u_k), v_test)
        + 2.0 * theta_const * mu_const * inner(epsilon(u_trial), epsilon(v_test))
        - p_trial * div(v_test)
        + q_test * div(u_trial)
    ) * dx_measure

    r_vol = (
        rho_const * dot(u_k - u_n, v_test) / dt_const
        + theta_const * rho_const * dot(dot(grad(u_k), u_k), v_test)
        + (1.0 - theta_const) * rho_const * dot(dot(grad(u_n), u_n), v_test)
        + 2.0 * theta_const * mu_const * inner(epsilon(u_k), epsilon(v_test))
        + 2.0 * (1.0 - theta_const) * mu_const * inner(epsilon(u_n), epsilon(v_test))
        - p_k * div(v_test)
        + q_test * div(u_k)
    ) * dx_measure

    return a_vol, r_vol


def traction_dot_direction(u_vec, p_scal, direction, mu_const):
    """Return ((σ(u,p)·n)·direction) on boundary facets."""
    n = FacetNormal()
    grad_u = grad(u_vec)
    sigma_n = mu_const * (dot(grad_u, n) + dot(grad_u.T, n)) - p_scal * n
    return dot(sigma_n, direction)


# --------------------------------------------------------------------------- #
#                            Main driver routine
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Volume-only Turek benchmark using a Gmsh mesh."
    )
    parser.add_argument("--backend", choices=("python", "jit"), default="jit", help="Assembly backend to use.")
    parser.add_argument("--mesh-size", type=float, default=0.02, help="Target edge size for gmsh.")
    parser.add_argument("--mesh-type", choices=("tri", "quad"), default="quad",
                        help="Element type generated by gmsh (default: quad).")
    parser.add_argument("--mesh-file", type=Path, help="Optional path to reuse/store the .msh file.")
    parser.add_argument("--rebuild-mesh", action="store_true", help="Force rebuilding the gmsh mesh.")
    parser.add_argument("--view-gmsh", action="store_true", help="Preview the generated mesh in Gmsh.")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step size.")
    parser.add_argument("--theta", type=float, default=0.5, help="Theta parameter for the time-stepping scheme.")
    parser.add_argument("--max-steps", type=int, default=36, help="Maximum number of time steps.")
    parser.add_argument(
        "--save-vtk",
        dest="save_vtk",
        action="store_true",
        help="Write VTU files for each step (default: on).",
    )
    parser.add_argument(
        "--no-save-vtk",
        dest="save_vtk",
        action="store_false",
        help="Disable VTU output.",
    )
    parser.set_defaults(save_vtk=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("turek_volume_results"),
        help="Directory for VTU files / diagnostic plots.",
    )
    parser.add_argument("--stop-on-steady", action="store_true", help="Stop when reaching steady state.")
    args = parser.parse_args()

    _configure_numba()

    mesh, persistent_mesh_path = prepare_mesh(
        args.mesh_file, args.mesh_size, args.rebuild_mesh, args.mesh_type, args.view_gmsh
    )
    print(mesh)

    cylinder_edges = mesh.edge_bitset("cylinder")
    if cylinder_edges.cardinality() == 0:
        raise RuntimeError("Cylinder boundary tag not found in the imported mesh.")
    for name in ("inlet", "outlet", "walls", "cylinder"):
        print(f"Edges tagged '{name}': {mesh.edge_bitset(name).cardinality()}")

    # Mixed Taylor–Hood space (vector velocity + scalar pressure)
    mixed_element = MixedElement(
        mesh,
        field_specs={"ux": FE_ORDER, "uy": FE_ORDER, "p": FE_ORDER - 1},
    )
    dof_handler = DofHandler(mixed_element, method="cg")

    # Boundary conditions (no-slip on walls + cylinder, parabolic inflow)
    def parabolic_inflow(x, y):
        return 4.0 * U_MEAN * y * (H - y) / (H**2)

    bcs: list[BoundaryCondition] = [
        BoundaryCondition("ux", "dirichlet", "inlet", parabolic_inflow),
        BoundaryCondition("uy", "dirichlet", "inlet", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "walls", lambda x, y: 0.0),
        BoundaryCondition("uy", "dirichlet", "walls", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "cylinder", lambda x, y: 0.0),
        BoundaryCondition("uy", "dirichlet", "cylinder", lambda x, y: 0.0),
    ]

    bcs_homog = [
        BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs
    ]

    # --- Function spaces / functions ---
    velocity_space = FunctionSpace(name="velocity", field_names=["ux", "uy"], dim=1, side="+")
    pressure_space = FunctionSpace(name="pressure", field_names=["p"], dim=0, side="+")

    du = VectorTrialFunction(space=velocity_space, dof_handler=dof_handler, side="+")
    v = VectorTestFunction(space=velocity_space, dof_handler=dof_handler, side="+")
    dp = TrialFunction(name="trial_pressure", field_name="p", dof_handler=dof_handler, side="+")
    q = TestFunction(name="test_pressure", field_name="p", dof_handler=dof_handler, side="+")

    u_k = VectorFunction(name="u_k", field_names=["ux", "uy"], dof_handler=dof_handler, side="+")
    u_n = VectorFunction(name="u_n", field_names=["ux", "uy"], dof_handler=dof_handler, side="+")
    p_k = Function(name="p_k", field_name="p", dof_handler=dof_handler, side="+")
    p_n = Function(name="p_n", field_name="p", dof_handler=dof_handler, side="+")

    # Initialize
    for func in (u_k, u_n):
        func.nodal_values.fill(0.0)
    for func in (p_k, p_n):
        func.nodal_values.fill(0.0)
    dof_handler.apply_bcs(bcs, u_n, p_n)
    dof_handler.apply_bcs(bcs, u_k, p_k)

    rho_const = Constant(RHO)
    mu_const = Constant(MU)
    dt_const = Constant(args.dt)
    theta_const = Constant(args.theta)

    volume_quadrature = 2 * FE_ORDER + 2
    dx_vol = dx(metadata={"q": volume_quadrature})

    jacobian_form, residual_form = build_volume_forms(
        du,
        v,
        dp,
        q,
        u_k,
        u_n,
        p_k,
        p_n,
        rho_const,
        mu_const,
        dt_const,
        theta_const,
        dx_vol,
    )

    # --- Force diagnostics on the cylinder boundary ---
    boundary_quadrature = max(8, FE_ORDER * 3)
    d_gamma_cyl = dS(defined_on=cylinder_edges, metadata={"q": boundary_quadrature})

    e_x = Constant(np.array([1.0, 0.0]), dim=1)
    e_y = Constant(np.array([0.0, 1.0]), dim=1)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    histories: dict[str, list[float]] = {"time": [], "cd": [], "cl": [], "dp": []}
    monitor_point = np.array([0.15, 0.2])

    probe_A = np.array([CENTER[0] - RADIUS - 0.01, CENTER[1]])
    probe_B = np.array([CENTER[0] + RADIUS + 0.01, CENTER[1]])

    def save_solution(funcs):
        """Callback executed after every converged time step."""
        velocity = funcs[0]  # VectorFunction
        pressure = funcs[1]  # Function

        # Optionally write VTK outputs
        step_id = len(histories["time"])
        if args.save_vtk:
            filename = output_dir / f"solution_{step_id:04d}.vtu"
            export_vtk(
                filename=str(filename),
                mesh=mesh,
                dof_handler=dof_handler,
                functions={"velocity": velocity, "pressure": pressure},
            )

        traction_drag = traction_dot_direction(velocity, pressure, e_x, mu_const) * d_gamma_cyl
        traction_lift = traction_dot_direction(velocity, pressure, e_y, mu_const) * d_gamma_cyl

        drag = assemble_form(
            Equation(None, traction_drag),
            dof_handler=dof_handler,
            assembler_hooks={traction_drag.integrand: {"name": "drag"}},
            backend="python",
        )["drag"]
        lift = assemble_form(
            Equation(None, traction_lift),
            dof_handler=dof_handler,
            assembler_hooks={traction_lift.integrand: {"name": "lift"}},
            backend="python",
        )["lift"]

        coeff = 2.0 / (RHO * U_MEAN**2 * D)
        c_d = coeff * drag
        c_l = coeff * lift

        # Pressure drop measurement (points upstream/downstream of cylinder)
        pA, _ = nearest_pressure_dof_value(dof_handler, pressure, probe_A)
        pB, _ = nearest_pressure_dof_value(dof_handler, pressure, probe_B)
        dp = pA - pB

        u_monitor = evaluate_vector_field(dof_handler, mesh, velocity, monitor_point)

        histories["time"].append(step_id * args.dt)
        histories["cd"].append(c_d)
        histories["cl"].append(c_l)
        histories["dp"].append(dp)

        print(
            f"[step {step_id:04d}] "
            f"Cd={c_d:.4f}  Cl={c_l:.4f}  Δp={dp:.4f}  "
            f"u(0.15,0.20)=({u_monitor[0]:.4f}, {u_monitor[1]:.4f})"
        )

    # Solver setup
    time_params = TimeStepperParameters(
        dt=args.dt,
        max_steps=args.max_steps,
        stop_on_steady=args.stop_on_steady,
        steady_tol=1e-6,
        theta=args.theta,
    )

    solver = NewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dof_handler,
        mixed_element=mixed_element,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=1e-6, line_search=True),
        postproc_timeloop_cb=save_solution,
        backend=args.backend,
    )

    functions = [u_k, p_k]
    prev_functions = [u_n, p_n]

    t0 = time.time()
    try:
        solver.solve_time_interval(
            functions=functions,
            prev_functions=prev_functions,
            time_params=time_params,
        )
    except Exception as exc:  # pragma: no cover - diagnostic output
        print(f"Solver failed: {exc}")
        raise
    finally:
        print(f"Total runtime: {time.time() - t0:.1f} s")
        if persistent_mesh_path:
            print(f"Mesh stored at: {persistent_mesh_path}")
        if histories["time"]:
            print(
                f"Final Cd={histories['cd'][-1]:.4f}, "
                f"Cl={histories['cl'][-1]:.4f}, "
                f"Δp={histories['dp'][-1]:.4f}"
            )
            try:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
                axes[0].plot(histories["time"], histories["cd"], color="tab:blue")
                axes[0].set_ylabel("Cd")
                axes[0].set_title("Drag coefficient")
                axes[0].grid(True, linestyle=":", linewidth=0.5)

                axes[1].plot(histories["time"], histories["cl"], color="tab:green")
                axes[1].set_ylabel("Cl")
                axes[1].set_title("Lift coefficient")
                axes[1].grid(True, linestyle=":", linewidth=0.5)

                axes[2].plot(histories["time"], histories["dp"], color="tab:red")
                axes[2].set_ylabel("Δp")
                axes[2].set_xlabel("Time")
                axes[2].set_title("Pressure drop")
                axes[2].grid(True, linestyle=":", linewidth=0.5)

                fig.tight_layout()
                fig.savefig(output_dir / "diagnostics.png", dpi=200)
                plt.close(fig)
            except Exception as exc:  # pragma: no cover
                print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()
