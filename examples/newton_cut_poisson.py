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
import scipy.sparse
import scipy.sparse.linalg as sp_la
import os

# --- Core pycutfem imports ---
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.levelset import CircleLevelSet

# --- UFL-like imports ---
from pycutfem.ufl.expressions import (
    TrialFunction, TestFunction,
    Function, Constant, grad, inner, dot, jump, FacetNormal, CellDiameter, Derivative
)
from pycutfem.ufl.measures import dx, dGhost, dInterface
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.io.vtk import export_vtk
from pycutfem.fem import transform
from pycutfem.integration.quadrature import volume
from pycutfem.ufl.helpers_geom import (
    clip_triangle_to_side, fan_triangulate, map_ref_tri_to_phys, corner_tris
)
# --- NEW: Import Newton Solver ---
from pycutfem.solvers.nonlinear_solver import NewtonSolver, NewtonParameters, TimeStepperParameters

def run_step85_newton():
    """
    Main function to run the step-85 simulation with Newton's method.
    """
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
    n_refinements = 4
    convergence_data = []

    for cycle in range(n_refinements):
        print(f"\n--- Refinement cycle {cycle} ---")

        # --- Mesh and Level Set ---
        nx = ny = 10 * (2**cycle)
        h_max = (mesh_bounds[1] - mesh_bounds[0]) / nx
        print(f"Creating mesh with h = {h_max:.4f}...")

        nodes, elems, _, corners = structured_quad(
            Lx=mesh_bounds[1] - mesh_bounds[0],
            Ly=mesh_bounds[1] - mesh_bounds[0],
            nx=nx, ny=ny,
            poly_order=fe_degree,
            offset=(mesh_bounds[0], mesh_bounds[0])
        )
        mesh = Mesh(nodes=nodes, element_connectivity=elems,
                    elements_corner_nodes=corners, element_type="quad", poly_order=fe_degree)
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
        dx_phys = dx(defined_on=physical_domain, level_set=level_set, metadata={'side': '-',"q": fe_degree + 2})
        dGamma = dInterface(defined_on=cut_domain, level_set=level_set, metadata={"q": fe_degree + 3})
        dGhost_stab = dGhost(defined_on=ghost_domain, level_set=level_set, metadata={"q": fe_degree + 3})

        # --- NEW: Define Residual Form R(u_k, v) ---
        # This is equivalent to a(u_k, v) - L(v) = 0
        residual = (
            # Volume terms
            inner(grad(u_k), grad(v)) * dx_phys
            -f * v * dx_phys
            # Nitsche terms for BC
            - dot(grad(u_k), n) * v * dGamma
            - dot(grad(v), n) * (u_k - g) * dGamma
            + (gamma_N / h) * (u_k - g) * v * dGamma
            # Ghost penalty stabilization
            + (0.5 * gamma_G * h * jump(normal_grad(u_k)) * jump(normal_grad(v))) * dGhost_stab
        )

        # --- NEW: Define Jacobian Form J(du, v) ---
        # This is the Gateaux derivative of the residual w.r.t u_k in direction du
        jacobian = (
            # Volume terms
            inner(grad(du), grad(v)) * dx_phys
            # Nitsche terms
            - dot(grad(du), n) * v * dGamma
            - dot(grad(v), n) * du * dGamma
            + (gamma_N / h) * du * v * dGamma
            # Ghost penalty
            + (0.5 * gamma_G * h * jump(normal_grad(du)) * jump(normal_grad(v))) * dGhost_stab
        )

        # ====================================================================
        #    4. ASSEMBLE AND SOLVE WITH NEWTON'S METHOD
        # ====================================================================
        print("Setting up and running the Newton solver...")

        # --- NEW: Initialize and run the solver ---
        solver = NewtonSolver(
            residual_form=residual,
            jacobian_form=jacobian,
            dof_handler=dof_handler,
            mixed_element=element,
            bcs=bcs,
            bcs_homog=bcs_homog,
            newton_params=NewtonParameters(newton_tol=1e-8, max_newton_iter=10)
        )
        
        # Since this is a linear problem, the Newton solver will converge in one step.
        # We need to provide the function to be solved for.
        # The solver updates u_k in place.
        dt = Constant(1.0)  # Time step, not used in steady state
        time_params = TimeStepperParameters(dt=dt.value ,
                                            stop_on_steady=False, 
                                            max_steps = 1)
        u_k_prev = Function(name="u_k_prev", field_name="u", dof_handler=dof_handler)
        u_k_prev.nodal_values[:] = 0.0
        solver.solve_time_interval(
            functions=[u_k],
            prev_functions=[u_k_prev], # For steady state, prev_functions is not used but required
            time_params=time_params
        )

        # ====================================================================
        #    5. POST-PROCESSING AND ERROR COMPUTATION
        # ====================================================================
        print("Post-processing and computing L2 error...")

        # --- NEW: Get solution vector from the Function object ---
        solution_vec = u_k.nodal_values

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

        # --- Manual L2 Error Calculation (unchanged) ---
        l2_error_sq = 0.0
        q_order_err = 2 * fe_degree + 2
        qp_ref, qw_ref = volume(mesh.element_type, q_order_err)
        for eid in mesh.element_bitset("inside").to_indices():
            gdofs = dof_handler.get_elemental_dofs(eid)
            u_loc = u_k.get_nodal_values(gdofs)
            for (xi, eta), w in zip(qp_ref, qw_ref):
                J = transform.jacobian(mesh, eid, (xi, eta))
                detJ = abs(np.linalg.det(J))
                x, y = transform.x_mapping(mesh, eid, (xi, eta))
                phi = element.basis('u', xi, eta)
                u_h_val = np.dot(phi, u_loc)
                u_ex_val = analytical_solution(x, y)
                l2_error_sq += (u_h_val - u_ex_val)**2 * w * detJ

        qp_ref_tri, qw_ref_tri = volume("tri", q_order_err)
        for eid in mesh.element_bitset("cut").to_indices():
            elem = mesh.elements_list[eid]
            tri_local, corner_ids = corner_tris(mesh, elem)
            gdofs = dof_handler.get_elemental_dofs(eid)
            u_loc = u_k.get_nodal_values(gdofs)

            for loc_tri in tri_local:
                v_ids = [corner_ids[i] for i in loc_tri]
                v_coords = mesh.nodes_x_y_pos[v_ids]
                v_phi = np.array([level_set(np.asarray(xy)) for xy in v_coords])
                polygons = clip_triangle_to_side(v_coords, v_phi, side='-')
                for poly in polygons:
                    for A, B, C in fan_triangulate(poly):
                        qp_phys, qw_phys = map_ref_tri_to_phys(A, B, C, qp_ref_tri, qw_ref_tri)
                        for x_phys, w_phys in zip(qp_phys, qw_phys):
                            xi, eta = transform.inverse_mapping(mesh, eid, x_phys)
                            phi = element.basis('u', xi, eta)
                            u_h_val = np.dot(phi, u_loc)
                            u_ex_val = analytical_solution(x_phys[0], x_phys[1])
                            l2_error_sq += (u_h_val - u_ex_val)**2 * w_phys

        l2_error = np.sqrt(l2_error_sq)
        convergence_data.append({'cycle': cycle, 'h': h_max, 'L2-error': l2_error, 'ndofs': active_dofs_count})
        print(f"Cycle {cycle} finished. L2 Error: {l2_error:.5e}")

    # ========================================================================
    #    6. PRINT CONVERGENCE TABLE
    # ========================================================================
    print("\n" + "="*60)
    print("                 Convergence Table for L2-Error")
    print("="*60)
    print(f"{'Cycle':>5} | {'h':>12} | {'NDOFs':>9} | {'L2-Error':>15} | {'Rate':>6}")
    print("-"*60)
    for i, data in enumerate(convergence_data):
        rate_str = "----"
        if i > 0:
            rate = np.log2(convergence_data[i-1]['L2-error'] / data['L2-error'])
            rate_str = f"{rate:.2f}"
        print(f"{data['cycle']:>5d} | {data['h']:>12.4e} | {data['ndofs']:>9d} | {data['L2-error']:>15.5e} | {rate_str:>6}")
    print("="*60)


if __name__ == '__main__':
    run_step85_newton()
