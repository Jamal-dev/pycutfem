#!/usr/bin/env python
# coding: utf-8

"""
Python implementation of the deal.II step-85 tutorial using the pycutfem library.

This script solves the Poisson equation on a circular domain embedded in a
larger square mesh using an immersed boundary method (CutFEM) with a Newton solver.

Problem setup:
- PDE: -Δu = 4  in Ω = {(x,y) | x² + y² < 1}
- BC:   u = 1  on ∂Ω
- Analytical Solution: u(x,y) = 2 - (x² + y²)
"""

import numpy as np
import os
import argparse
import time

# --- Core pycutfem imports ---
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.levelset import CircleLevelSet, LevelSetMeshAdaptation

# --- UFL-like imports ---
from pycutfem.ufl.expressions import (
    TrialFunction, TestFunction,
    Function, Constant, grad, inner, dot, jump, FacetNormal, CellDiameter
)
from pycutfem.ufl.measures import dx, dGhost, dInterface
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.io.vtk import export_vtk
# --- NEW: Import Newton Solver ---
from pycutfem.solvers.nonlinear_solver import NewtonSolver, NewtonParameters, TimeStepperParameters

def run_step85_newton(*, backend: str = "jit", with_deformation: bool = False, cycles: int = 4):
    """
    Main function to run the step-85 simulation with Newton's method.
    """
    total_start = time.perf_counter()
    jit_warmup_time = 0.0
    total_solver_time = 0.0
    total_assembly_time = 0.0
    total_linear_time = 0.0
    total_line_search_time = 0.0
    init_times: list[float] = []
    # ========================================================================
    #    1. PROBLEM SETUP
    # ========================================================================
    print("--- Setting up the Poisson problem from deal.II step-85 ---")

    # --- Geometry and FE Degree ---
    domain_radius = 1.0
    domain_center = (0.0, 0.0)
    mesh_bounds = (-1.21, 1.21)
    fe_degree = 1

    # --- Physical and Numerical Parameters ---
    rhs_val = 4.0
    bc_val = 1.0
    nitsche_parameter = 5.0 * (fe_degree + 1) * fe_degree
    ghost_parameter = 0.5

    # --- Analytical solution for error computation ---
    def analytical_solution(x, y):
        return 2.0 - (x**2 + y**2)

    # ========================================================================
    #    2. CONVERGENCE LOOP
    # ========================================================================
    n_refinements = cycles
    convergence_data = []

    for cycle in range(n_refinements):
        print(f"\n--- Refinement cycle {cycle} ---")

        # --- Mesh and Level Set ---
        nx = ny = 10 * (2**cycle)
        h_max = (mesh_bounds[1] - mesh_bounds[0]) / nx
        print(f"Creating mesh with h = {h_max:.4f}...")

        geom_order = 2 if with_deformation else fe_degree
        nodes, elems, _, corners = structured_quad(
            Lx=mesh_bounds[1] - mesh_bounds[0],
            Ly=mesh_bounds[1] - mesh_bounds[0],
            nx=nx, ny=ny,
            poly_order=geom_order,
            offset=(mesh_bounds[0], mesh_bounds[0])
        )
        mesh = Mesh(nodes=nodes, element_connectivity=elems,
                    elements_corner_nodes=corners, element_type="quad", poly_order=geom_order)
        level_set = CircleLevelSet(center=domain_center, radius=domain_radius)

        # --- Mesh Classification and Domain Definition ---
        print("Classifying mesh entities...")
        mesh.classify_elements(level_set)
        mesh.classify_edges(level_set)
        mesh.build_interface_segments(level_set=level_set)

        physical_domain = mesh.element_bitset("inside") | mesh.element_bitset("cut")
        cut_domain = mesh.element_bitset("cut")
        ghost_domain = mesh.edge_bitset("ghost_neg") | mesh.edge_bitset("ghost_both")
        
        print('-'*60)
        print(f"Physical domain: {physical_domain.cardinality()} elements, "
              f"Cut domain: {cut_domain.cardinality()} elements, "
              f"Ghost domain: {ghost_domain.cardinality()} elements")
        print('-'*60)

        # --- Finite Element Space and DoF Handler ---
        if with_deformation:
            adapter = LevelSetMeshAdaptation(
                mesh,
                order=max(2, geom_order),
                threshold=1.0,
                max_steps=6,
            )
            deformation = adapter.calc_deformation(level_set, q_vol=2 * fe_degree + 4)
            level_set = adapter.lset_p1
        else:
            deformation = None

        element = MixedElement(mesh, field_specs={'u': fe_degree})
        dof_handler = DofHandler(element, method='cg')

        # --- Deactivate DoFs on exterior cells ---
        print("Deactivating degrees of freedom on exterior cells...")
        outside_elements = mesh.element_bitset("outside")
        inactive_dofs = dof_handler.tag_dof_bitset(
            tag="inactive",
            field="u",
            elem_mask=outside_elements,
            strict=True,
        )
        print(f"Found {len(inactive_dofs)} inactive dofs to constrain.")

        total_dofs = dof_handler.total_dofs
        active_dofs_count = total_dofs - len(inactive_dofs)
        print(f"Number of active degrees of freedom: {active_dofs_count}")
        print(f"Total degrees of freedom (before constraints): {total_dofs}")

        # --- NEW: Define Boundary Conditions for Newton Solver ---
        bcs = [BoundaryCondition('u', 'dirichlet', 'inactive', lambda x, y: 0.0)]
        bcs_homog = [BoundaryCondition('u', 'dirichlet', 'inactive', lambda x, y: 0.0)]


        # ====================================================================
        #    3. UFL WEAK FORM FOR NEWTON SOLVER
        # ====================================================================
        print("Defining the weak form for Newton's method...")
        
        # --- NEW: Define solution Function and Trial/Test Functions ---
        u_k = Function(name="u_k", field_name="u", dof_handler=dof_handler) # Current Newton iterate
        du  = TrialFunction(name="du", field_name='u', dof_handler=dof_handler)                  # Newton update direction
        v   = TestFunction(name="v", field_name='u', dof_handler=dof_handler)                   # Test function

        # --- Symbolic constants and helpers ---
        f = Constant(rhs_val)
        g = Constant(bc_val)
        n = FacetNormal()
        h = CellDiameter()
        
        def normal_grad(expr):
            """Compact helper for n·∇expr on interface/ghost facets."""
            return dot(grad(expr), n)

        gamma_N = Constant(nitsche_parameter)
        gamma_G = Constant(ghost_parameter)

        # --- Define integration measures ---
        q_base = fe_degree + 2 + (2 if with_deformation else 0)
        dx_phys = dx(defined_on=physical_domain, level_set=level_set,
                     metadata={'side': '-',"q": q_base}, deformation=deformation)
        dGamma = dInterface(defined_on=cut_domain, level_set=level_set,
                            metadata={"q": q_base + 1}, deformation=deformation)
        dGhost_stab = dGhost(defined_on=ghost_domain, level_set=level_set,
                             metadata={"q": q_base + 1}, deformation=deformation)

        # --- NEW: Define Residual Form R(u_k, v) ---
        # This is equivalent to a(u_k, v) - L(v) = 0
        # interface_terms_residual = (- dot(grad(u_k), n) * v * dGamma
        #     - dot(grad(v), n) * (u_k - g) * dGamma
        #     + (gamma_N / h) * (u_k - g) * v * dGamma)
        interface_terms_residual = (- dot(grad(u_k), n) * v 
                - dot(grad(v), n) * (u_k - g)
                + (gamma_N / h) * (u_k - g) * v )* dGamma
        residual = (
            # Volume terms
            (inner(grad(u_k), grad(v)) 
            -f * v) * dx_phys
            # Nitsche terms for BC
            + interface_terms_residual
            # Ghost penalty stabilization
            + (0.5 * gamma_G * h * jump(normal_grad(u_k)) * jump(normal_grad(v))) * dGhost_stab
        )

        # --- NEW: Define Jacobian Form J(du, v) ---
        # This is the Gateaux derivative of the residual w.r.t u_k in direction du
        # interface_terms_jacobian = (- dot(grad(du), n) * v * dGamma
        #     - dot(grad(v), n) * du * dGamma
        #     + (gamma_N / h) * du * v * dGamma)
        interface_terms_jacobian = (- dot(grad(du), n) * v 
                - dot(grad(v), n) * du
                + (gamma_N / h) * du * v )* dGamma
        jacobian = (
            # Volume terms
            inner(grad(du), grad(v)) * dx_phys
            # Nitsche terms
            + interface_terms_jacobian
            # Ghost penalty
            + (0.5 * gamma_G * h * jump(normal_grad(du)) * jump(normal_grad(v))) * dGhost_stab
        )

        # ====================================================================
        #    4. ASSEMBLE AND SOLVE WITH NEWTON'S METHOD
        # ====================================================================
        print("Setting up and running the Newton solver...")

        # --- NEW: Initialize and run the solver ---
        init_t0 = time.perf_counter()
        solver = NewtonSolver(
            residual_form=residual,
            jacobian_form=jacobian,
            dof_handler=dof_handler,
            mixed_element=element,
            bcs=bcs,
            bcs_homog=bcs_homog,
            newton_params=NewtonParameters(newton_tol=1e-8, max_newton_iter=10),
            backend=backend,
            deformation=deformation,
        )
        init_elapsed = time.perf_counter() - init_t0
        init_times.append(init_elapsed)
        if cycle == 0:
            jit_warmup_time = init_elapsed

        # Since this is a linear problem, the Newton solver will converge in one step.
        # We need to provide the function to be solved for.
        # The solver updates u_k in place.
        dt = Constant(1.0)  # Time step, not used in steady state
        time_params = TimeStepperParameters(dt=dt.value ,
                                            stop_on_steady=False, 
                                            max_steps = 1)
        u_k_prev = Function(name="u_k_prev", field_name="u", dof_handler=dof_handler)
        u_k_prev.nodal_values[:] = 0.0
        solve_t0 = time.perf_counter()
        solver.solve_time_interval(
            functions=[u_k],
            prev_functions=[u_k_prev], # For steady state, prev_functions is not used but required
            time_params=time_params
        )
        total_solver_time += time.perf_counter() - solve_t0

        iter_totals = getattr(solver, "_last_iteration_totals", {})
        total_assembly_time += iter_totals.get("assembly", 0.0)
        total_linear_time += iter_totals.get("linear_solve", 0.0)
        total_line_search_time += iter_totals.get("line_search", 0.0)

        # ====================================================================
        #    5. POST-PROCESSING AND ERROR COMPUTATION
        # ====================================================================
        print("Post-processing and computing errors...")

        # --- Output results to VTK file for one cycle ---
        if cycle == 1:
            output_dir = "step85_newton_results"
            os.makedirs(output_dir, exist_ok=True)
            vtk_path = os.path.join(output_dir, f"step85_solution_cycle{cycle}.vtu")
            print(f"Writing solution to {vtk_path}")
            export_vtk(
                filename=vtk_path,
                mesh=mesh,
                dof_handler=dof_handler,
                functions={"solution": u_k}
            )

        grad_exact = lambda x, y: np.array([-2.0 * x, -2.0 * y])

        l2_error = dof_handler.l2_error_on_side(
            functions=u_k,
            exact={'u': analytical_solution},
            level_set=level_set,
            side='-',
            relative=False,
            deformation=deformation,
        )
        h1_error = dof_handler.h1_error_scalar_on_side(
            uh=u_k,
            exact_grad=lambda x, y: grad_exact(x, y),
            level_set=level_set,
            side='-',
            relative=False,
            deformation=deformation,
        )

        convergence_data.append({
            'cycle': cycle,
            'h': h_max,
            'L2-error': l2_error,
            'H1-error': h1_error,
            'ndofs': active_dofs_count
        })
        print(f"Cycle {cycle} finished. L2 Error: {l2_error:.5e}, H1 Error: {h1_error:.5e}")

    # ========================================================================
    #    6. PRINT CONVERGENCE TABLE
    # ========================================================================
    print("\n" + "="*60)
    print("        Convergence Table for L2 and H1 Errors")
    print("="*60)
    print(f"{'Cycle':>5} | {'h':>12} | {'NDOFs':>9} | {'L2-Error':>12} | {'Rate(L2)':>8} | {'H1-Error':>12} | {'Rate(H1)':>8}")
    print("-"*60)
    for i, data in enumerate(convergence_data):
        rate_l2 = "----"
        rate_h1 = "----"
        if i > 0:
            prev = convergence_data[i-1]
            rate_l2 = f"{np.log2(prev['L2-error'] / data['L2-error']):.2f}" if data['L2-error'] > 0 else "----"
            rate_h1 = f"{np.log2(prev['H1-error'] / data['H1-error']):.2f}" if data['H1-error'] > 0 else "----"
        print(
            f"{data['cycle']:>5d} | {data['h']:>12.4e} | {data['ndofs']:>9d} | "
            f"{data['L2-error']:>12.5e} | {rate_l2:>8} | {data['H1-error']:>12.5e} | {rate_h1:>8}"
        )
    print("="*60)

    total_elapsed = time.perf_counter() - total_start
    total_init_time = sum(init_times)
    residual_time = total_elapsed - total_init_time - total_solver_time

    print("\nTiming summary")
    print("="*60)
    print(f"Total wall time            : {total_elapsed:.3f} s")
    if init_times:
        print(f"  Solver setup (first cycle): {jit_warmup_time:.3f} s")
        if total_init_time - jit_warmup_time > 0:
            print(f"  Solver setup (other cycles): {total_init_time - jit_warmup_time:.3f} s")
    print(f"  Newton solve stage        : {total_solver_time:.3f} s")
    print(f"    Assembly subtotal       : {total_assembly_time:.3f} s")
    print(f"    Linear solve subtotal   : {total_linear_time:.3f} s")
    if total_line_search_time > 0.0:
        print(f"    Line-search subtotal    : {total_line_search_time:.3f} s")
    print(f"  Other overhead            : {max(residual_time, 0.0):.3f} s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CutFEM Poisson problem (step-85) with Newton solver")
    parser.add_argument(
        "--backend",
        choices=("python", "jit"),
        default="jit",
        help="form assembly backend (jit is faster after first compile)",
    )
    parser.add_argument(
        "--with-deformation",
        action="store_true",
        help="enable isoparametric deformation of the cut geometry",
    )
    parser.add_argument(
        "--no-deformation",
        action="store_true",
        help="force the run without deformation even if other flags set it",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=4,
        help="number of refinement cycles to execute (default: 4)",
    )
    args, _ = parser.parse_known_args()
    if args.with_deformation and args.no_deformation:
        raise SystemExit("Choose at most one of --with-deformation or --no-deformation.")
    use_deformation = args.with_deformation and not args.no_deformation
    print(f"Backend: {args.backend}")
    print(f"With deformation: {use_deformation}")
    run_step85_newton(backend=args.backend, with_deformation=use_deformation, cycles=args.cycles)
