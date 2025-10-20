#!/usr/bin/env python3
"""
CutFEM Turek benchmark (DFG 2D-2) implemented with NGSolve/ngsxfem.

The cylinder is described implicitly with a level-set function on a
structured background mesh.  No-slip on the immersed boundary is
enforced by a symmetric Nitsche formulation while the inflow and wall
conditions are imposed strongly on the degrees of freedom exposed on
the outer boundary.  The temporal discretisation solves the fully
implicit backward-Euler Navier–Stokes
step with Newton's method at every time level.

The script records drag, lift and the pressure drop between the
benchmark probe points in CSV files inside ``--output-dir``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path

from netgen.geom2d import SplineGeometry
from ngsolve import (
    BilinearForm,
    CoefficientFunction,
    GridFunction,
    Grad,
    H1,
    Id,
    InnerProduct,
    Integrate,
    Mesh,
    Norm,
    NumberSpace,
    SetHeapSize,
    TaskManager,
    VectorH1,
    VTKOutput,
    div,
    specialcf,
    x,
    y,
    IfPos,
    FESpace,
)
from xfem import (
    HASNEG,
    CutInfo,
    POS,
    NEG,
    IF,
    HASPOS,
    dCut,
    GetFacetsWithNeighborTypes,
    SymbolicFacetPatchBFI,
)
from xfem.lsetcurv import LevelSetMeshAdaptation
from ngsolve.solvers import Newton

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


SetHeapSize(200 * 10**6)


@dataclass
class SimulationParameters:
    rho: float
    mu: float
    dt: float
    gamma_n: float
    gamma_gp: float
    u_mean: float
    diameter: float
    p_reg: float = 1.0e-8


def parabolic_inflow(height: float, mean_velocity: float) -> CoefficientFunction:
    """Return the DFG parabolic inflow profile."""
    return CoefficientFunction((4.0 * mean_velocity * y * (height - y) / (height ** 2), 0.0))


def build_background_mesh(length: float, height: float, maxh: float) -> Mesh:
    """Generate a tensor-product background mesh without the cylinder cut-out."""
    geo = SplineGeometry()
    p0 = geo.AddPoint(0.0, 0.0)
    p1 = geo.AddPoint(length, 0.0)
    p2 = geo.AddPoint(length, height)
    p3 = geo.AddPoint(0.0, height)
    geo.Append(["line", p0, p1], leftdomain=1, rightdomain=0, bc="bottom")
    geo.Append(["line", p1, p2], leftdomain=1, rightdomain=0, bc="outlet")
    geo.Append(["line", p2, p3], leftdomain=1, rightdomain=0, bc="top")
    geo.Append(["line", p3, p0], leftdomain=1, rightdomain=0, bc="inlet")
    mesh = Mesh(geo.GenerateMesh(maxh=maxh, quad_dominated=False))
    mesh.Curve(3)
    return mesh


def levelset_for_cylinder(cx: float, cy: float, radius: float) -> CoefficientFunction:
    """Signed distance to the circular obstacle (positive outside)."""
    return Norm(CoefficientFunction((x - cx, y - cy))) - radius


def symgrad(v):
    """Return the symmetric gradient tensor."""
    grad_v = Grad(v)
    return 0.5 * (grad_v + grad_v.trans)


def divergence(v):
    grad_v = Grad(v)
    dim = grad_v.dims[0]
    return sum(grad_v[i, i] for i in range(dim))


def setup_cut_geometry(mesh: Mesh, levelset_cf: CoefficientFunction, order: int):
    """Create the isoparametric deformation and CutInfo for the level-set."""
    lset_adapter = LevelSetMeshAdaptation(
        mesh,
        order=max(2, order),
        threshold=10.5,
        discontinuous_qn=True,
    )
    deformation = lset_adapter.CalcDeformation(levelset_cf)
    mesh.deformation = deformation
    lset_p1 = lset_adapter.lset_p1
    cut_info = CutInfo(mesh, lset_p1)
    return lset_p1, deformation, cut_info


def build_cut_spaces(mesh: Mesh, _cut_info: CutInfo, order: int):
    """Taylor-Hood spaces on the background mesh with jump couplings enabled."""
    V = VectorH1(mesh, order=order, dirichlet="inlet|top|bottom", dgjumps=True)
    Q = H1(mesh, order=max(1, order - 1), dgjumps=True)
    lagrange = NumberSpace(mesh)
    return FESpace([V, Q, lagrange], dgjumps=True)


def build_measures(mesh: Mesh, cut_info: CutInfo, lset_p1, deformation):
    """Return volume, interface and ghost facet sets."""
    els_outer = cut_info.GetElementsOfType(HASPOS)
    els_inner = cut_info.GetElementsOfType(HASNEG)
    dx_fluid = dCut(levelset=lset_p1, domain_type=POS, 
                    deformation=deformation,
                    definedonelements=els_outer)
    dx_inside = dCut(levelset=lset_p1, domain_type=NEG, deformation=deformation, definedonelements=els_inner)
    ds_interface = dCut(levelset=lset_p1, domain_type=IF, deformation=deformation)

    positive_elements = cut_info.GetElementsOfType(HASPOS)
    interface_elements = cut_info.GetElementsOfType(IF)
    ghost_facets = GetFacetsWithNeighborTypes(mesh, a=positive_elements, b=interface_elements)
    return dx_fluid, ds_interface, ghost_facets, dx_inside


def build_nonlinear_form(
    space,
    params: SimulationParameters,
    measures,
    lset_p1,
    previous_velocity: GridFunction,
):
    """Create the semilinear form defining the Navier–Stokes residual."""
    dx_fluid, ds_interface, ghost_facets, dx_inside = measures
    (u, p, lam), (v, q, eta) = space.TnT()

    vel_prev_cf = CoefficientFunction(previous_velocity)
    normal_vec = -Grad(lset_p1)
    normal_vec = normal_vec / Norm(normal_vec)
    h = specialcf.mesh_size

    form = BilinearForm(space, check_unused=False)
    form += params.rho / params.dt * InnerProduct(u - vel_prev_cf, v) * dx_fluid
    form += params.rho * InnerProduct(Grad(u) * u, v) * dx_fluid
    form += 2.0 * params.mu * InnerProduct(symgrad(u), symgrad(v)) * dx_fluid
    form +=  -  p * div(v) * dx_fluid
    form +=   q * divergence(u) * dx_fluid
    # form += params.p_reg * p * q * dx_fluid
    form += lam * q * dx_fluid
    form += eta * p * dx_fluid

    form += (
        - 2.0 * params.mu * InnerProduct(symgrad(u) * normal_vec, v)
        - 2.0 * params.mu * InnerProduct(symgrad(v) * normal_vec, u)
        + params.gamma_n * params.mu / h * InnerProduct(u, v)
        + p * InnerProduct(normal_vec, v)
        - q * InnerProduct(normal_vec, u)
    ) * ds_interface

    if params.gamma_gp > 0.0:
        ghost_integrand = params.gamma_gp * params.mu / h ** 2 * InnerProduct(
            u - u.Other(), v - v.Other()
        )
        ghost_integrand += params.gamma_gp / params.mu * (p - p.Other()) * (
            q - q.Other()
        )
        form += SymbolicFacetPatchBFI(
            form=ghost_integrand,
            skeleton=False,
            definedonelements=ghost_facets,
        )

    return form




def compute_forces(
    mesh: Mesh,
    velocity: GridFunction,
    pressure: GridFunction,
    params: SimulationParameters,
    ds_interface,
    normal_cf,
):
    """Return drag, lift and their dimensionless coefficients."""
    vel_cf = CoefficientFunction(velocity)
    press_cf = CoefficientFunction(pressure)

    traction = (2.0 * params.mu * symgrad(vel_cf) - press_cf * Id(mesh.dim)) * normal_cf
    ex = CoefficientFunction((1.0, 0.0))
    ey = CoefficientFunction((0.0, 1.0))
    drag = Integrate(InnerProduct(traction, ex) * ds_interface, mesh=mesh)
    lift = -Integrate(InnerProduct(traction, ey) * ds_interface, mesh=mesh)
    coeff = 2.0 / (params.rho * (params.u_mean ** 2) * params.diameter)
    return float(drag), float(lift), float(coeff * drag), float(coeff * lift)


def compute_pressure_drop(mesh: Mesh, pressure: GridFunction, point_a, point_b) -> float:
    """Evaluate Δp between two probe points."""
    pa = pressure(mesh(*point_a, 0.0))
    pb = pressure(mesh(*point_b, 0.0))
    return float(pa - pb)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NGSolve/ngsxfem CutFEM Turek benchmark (DFG 2D-2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dt", type=float, default=0.01, help="time step size")
    parser.add_argument("--tend", type=float, default=0.5, help="final simulation time")
    parser.add_argument("--num-steps", type=int, help="override the number of time steps")
    parser.add_argument("--order", type=int, default=2, help="velocity polynomial order")
    parser.add_argument("--maxh", type=float, default=0.05, help="background mesh size")
    parser.add_argument("--rho", type=float, default=1.0, help="density")
    parser.add_argument("--mu", type=float, default=1.0e-3, help="dynamic viscosity")
    parser.add_argument("--gamma-n", type=float, default=40.0, help="Nitsche penalty")
    parser.add_argument("--gamma-gp", type=float, default=1e-3, help="ghost penalty coefficient")
    parser.add_argument("--u-mean", type=float, default=1.0, help="mean inflow velocity")
    parser.add_argument("--newton-tol", type=float, default=1.0e-8, help="Newton residual tolerance")
    parser.add_argument("--newton-max-it", type=int, default=15, help="maximum Newton iterations")
    parser.add_argument("--newton-damp", type=float, default=1.0, help="Newton damping factor")
    parser.add_argument("--quiet-newton", action="store_true", help="suppress Newton iteration output")
    parser.add_argument("--output-dir", type=Path, default=Path("turek_ngsolve_results"))
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    params = SimulationParameters(
        rho=args.rho,
        mu=args.mu,
        dt=args.dt,
        gamma_n=args.gamma_n,
        gamma_gp=args.gamma_gp,
        u_mean=args.u_mean,
        diameter=0.1,
    )

    num_steps = args.num_steps or max(1, ceil(args.tend / args.dt))

    length, height = 2.2, 0.41
    mesh = build_background_mesh(length, height, args.maxh)

    levelset_cf = levelset_for_cylinder(0.2, 0.2, params.diameter * 0.5)
    lset_p1, deformation, cut_info = setup_cut_geometry(mesh, levelset_cf, args.order)
    space = build_cut_spaces(mesh, cut_info, args.order)
    measures = build_measures(mesh, cut_info, lset_p1, deformation)

    gfu = GridFunction(space, name="state")
    velocity, pressure, lagrange = gfu.components
    initial_profile = parabolic_inflow(height, params.u_mean)
    # initial_field = IfPos(lset_p1, initial_profile, CoefficientFunction((0.0, 0.0)))
    velocity.Set(initial_profile, definedon=mesh.Boundaries("inlet"))
    pressure.Set(0.0)
    lagrange.Set(0.0)

    inflow_profile = parabolic_inflow(height, params.u_mean)

    previous_velocity = GridFunction(space.components[0])
    previous_velocity.Set(velocity, definedon=mesh.Boundaries("inlet"))
    nonlinear_form = build_nonlinear_form(space, params, measures, lset_p1, previous_velocity)
    freedofs = space.FreeDofs()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    forces_path = output_dir / "forces.csv"
    pressure_path = output_dir / "pressure_drop.csv"

    velocity_cf = CoefficientFunction(velocity)
    velocity_mag = Norm(velocity_cf)

    vtk_output = VTKOutput(
        mesh,
        coefs=[velocity, pressure, velocity_mag],
        names=["velocity", "pressure", "velocity_magnitude"],
        filename=str((output_dir / "turek_ngsolve").resolve()),
        subdivision=2,
    )

    normal_cf = Grad(lset_p1) / Norm(Grad(lset_p1))

    times: list[float] = []
    drag_hist: list[float] = []
    lift_hist: list[float] = []
    cd_hist: list[float] = []
    cl_hist: list[float] = []
    dp_hist: list[float] = []

    with forces_path.open("w", newline="") as f_forces, pressure_path.open("w", newline="") as f_press:
        forces_writer = csv.writer(f_forces)
        pressure_writer = csv.writer(f_press)
        forces_writer.writerow(["step", "time", "drag", "lift", "cd", "cl"])
        pressure_writer.writerow(["step", "time", "delta_p"])

        current_time = 0.0
        print("--- Starting CutFEM Turek benchmark (NGSolve/ngsxfem) ---")
        for step in range(1, num_steps + 1):
            current_time += params.dt
            print(f"\nStep {step:03d}/{num_steps:03d} | t = {current_time:.3f}")

            velocity.Set(inflow_profile, definedon=mesh.Boundaries("inlet"))
            velocity.Set((0.0, 0.0), definedon=mesh.Boundaries("top|bottom"))
            # lagrange.Set(0.0)

            status, iters = Newton(
                nonlinear_form,
                gfu,
                freedofs=freedofs,
                maxit=args.newton_max_it,
                maxerr=args.newton_tol,
                dampfactor=args.newton_damp,
                printing=not args.quiet_newton,
            )
            if status != 0:
                print("  Warning: Newton did not converge within the allotted iterations.")
            elif not args.quiet_newton:
                print(f"  Newton converged in {iters} iterations.")

            drag, lift, cd, cl = compute_forces(
                mesh,
                velocity,
                pressure,
                params,
                measures[1],
                normal_cf,
            )
            delta_p = compute_pressure_drop(mesh, pressure, (0.15, 0.2), (0.25, 0.2))

            print(f"  Drag={drag:.4e}, Lift={lift:.4e}, Cd={cd:.4f}, Cl={cl:.4f}, Δp={delta_p:.4e}")
            forces_writer.writerow([step, current_time, drag, lift, cd, cl])
            pressure_writer.writerow([step, current_time, delta_p])
            f_forces.flush()
            f_press.flush()

            times.append(current_time)
            drag_hist.append(drag)
            lift_hist.append(lift)
            cd_hist.append(cd)
            cl_hist.append(cl)
            dp_hist.append(delta_p)

            vtk_output.Do(time=current_time)

            previous_velocity.Set(velocity)

    print("\nSimulation finished.")

    if plt is not None and times:
         # Create a 3x2 grid of subplots.
        # `sharex=True` links the x-axes of all plots.
        # `figsize` is adjusted for a wider layout.
        fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
        
        # --- Row 0: Coefficients ---
        # Subplot (0, 0) for Drag Coefficient
        axes[0, 0].plot(times, cd_hist, label="Cd", color="blue")
        axes[0, 0].set_ylabel("Drag Coefficient (Cd)")
        axes[0, 0].grid(True, linestyle=":", linewidth=0.5)
        axes[0, 0].set_title("Drag Coefficient over Time")

        # Subplot (0, 1) for Lift Coefficient
        axes[0, 1].plot(times, cl_hist, label="Cl", color="green")
        axes[0, 1].set_ylabel("Lift Coefficient (Cl)")
        axes[0, 1].grid(True, linestyle=":", linewidth=0.5)
        axes[0, 1].set_title("Lift Coefficient over Time")

        # --- Row 1: Forces ---
        # Subplot (1, 0) for Drag Force
        axes[1, 0].plot(times, drag_hist, label="Drag", color="red")
        axes[1, 0].set_ylabel("Drag Force")
        axes[1, 0].grid(True, linestyle=":", linewidth=0.5)
        axes[1, 0].set_title("Drag Force over Time")

        # Subplot (1, 1) for Lift Force
        axes[1, 1].plot(times, lift_hist, label="Lift", color="purple")
        axes[1, 1].set_ylabel("Lift Force")
        axes[1, 1].grid(True, linestyle=":", linewidth=0.5)
        axes[1, 1].set_title("Lift Force over Time")

        # --- Row 2: Pressure Drop ---
        # Subplot (2, 0) for Pressure Drop
        axes[2, 0].plot(times, dp_hist, label="Δp", color="orange")
        axes[2, 0].set_xlabel("Time")
        axes[2, 0].set_ylabel("Pressure Drop (Δp)")
        axes[2, 0].grid(True, linestyle=":", linewidth=0.5)
        axes[2, 0].set_title("Pressure Drop over Time")

        # Subplot (2, 1) is unused, so we turn it off
        fig.delaxes(axes[2, 1])

        # Add a main title to the entire figure
        fig.suptitle("Flow Diagnostics for Turek Benchmark", fontsize=16)

        # Adjust layout to prevent titles and labels from overlapping
        fig.tight_layout(rect=[0, 0, 1, 0.96]) # rect leaves space for suptitle
        plt.show()
        # plt.close(fig)
    elif plt is None:
        print("Matplotlib not available; skipped diagnostic plot.")


if __name__ == "__main__":
    main()
