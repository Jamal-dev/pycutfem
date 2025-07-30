#!/usr/bin/env python
# coding: utf-8

"""
Method of Manufactured Solutions (MMS) Test for the Stokes Equations
on an internal circular domain using CutFEM.

This script verifies the implementation of a mixed-element CutFEM formulation
for the steady-state Stokes equations.

Problem setup:
- Computational Domain: A circle Ω = {(x,y) | x² + y² < 1}
- Background Mesh: A square mesh on [-1.2, 1.2]² that encapsulates Ω.
- Stokes Equations:
    -μΔu + ∇p = f   (in Ω)
     div(u)   = g   (in Ω)
- Boundary Conditions:
    u = u_exact on ∂Ω (the circle boundary), applied weakly via Nitsche's method.

- Method:
  1. An analytical solution for u and p is defined over the entire plane.
  2. Forcing functions f and g are derived symbolically.
  3. The problem is solved on the background mesh, but the PDE is only active
     on the physical domain Ω. DOFs in the exterior are constrained to zero.
  4. The numerical solution's L2 error is computed only on the physical
     domain Ω, correctly handling cut cells.
"""

import numpy as np
import scipy.sparse.linalg as sp_la
import sympy as sp
import os

# --- Core pycutfem imports ---
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.levelset import CircleLevelSet

# --- UFL-like imports ---
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    TrialFunction, TestFunction, VectorTrialFunction, VectorTestFunction,
    Function, VectorFunction, Constant, grad, inner, dot, div,
    jump, FacetNormal, CellDiameter, Derivative
)
from pycutfem.ufl.measures import dx, dInterface, dGhost
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.solvers.nonlinear_solver import NewtonSolver, NewtonParameters, TimeStepperParameters
from pycutfem.fem import transform
from pycutfem.integration.quadrature import volume
from pycutfem.ufl.helpers_geom import clip_triangle_to_side, fan_triangulate, map_ref_tri_to_phys, corner_tris
from tests.ufl.test_face_integrals import bcs, dof_handler, mesh
import matplotlib.pyplot as plt

# --- SymPy symbols for MMS ---
x, y = sp.symbols('x y')

def run_stokes_mms_cut_internal():
    """
    Main function to set up and run the internal CutFEM Stokes MMS test.
    """
    # ========================================================================
    #    1. MANUFACTURED SOLUTION SETUP
    # ========================================================================
    print("--- Setting up the CutFEM Stokes MMS problem (Internal Domain) ---")

    # --- Physical & Geometric Parameters ---
    mu_val = 0.1
    domain_radius = 1.0
    domain_center = (0.0, 0.0)

    # --- Define a smooth, non-trivial analytical solution for u and p ---
    u_exact_sym_x = sp.sin(sp.pi * x) * sp.cos(sp.pi * y)
    u_exact_sym_y = -sp.cos(sp.pi * x) * sp.sin(sp.pi * y)
    p_exact_sym = sp.sin(sp.pi * x) * sp.cos(sp.pi * y) - sp.pi**2 # Adjust to have non-zero pressure

    # --- Derive forcing terms f and g ---
    f_sym_x = -mu_val * (sp.diff(u_exact_sym_x, x, 2) + sp.diff(u_exact_sym_x, y, 2)) + sp.diff(p_exact_sym, x)
    f_sym_y = -mu_val * (sp.diff(u_exact_sym_y, x, 2) + sp.diff(u_exact_sym_y, y, 2)) + sp.diff(p_exact_sym, y)
    g_sym = sp.diff(u_exact_sym_x, x) + sp.diff(u_exact_sym_y, y)
    
    # --- Lambdify expressions to create callable Python functions ---
    u_exact_func_x = sp.lambdify((x, y), u_exact_sym_x, 'numpy')
    u_exact_func_y = sp.lambdify((x, y), u_exact_sym_y, 'numpy')
    p_exact_func = sp.lambdify((x, y), p_exact_sym, 'numpy')
    f_func = sp.lambdify((x, y), [f_sym_x, f_sym_y], 'numpy')
    g_func = sp.lambdify((x, y), g_sym, 'numpy')
    
    u_exact_vector_func = lambda x, y: np.array([u_exact_func_x(x, y), u_exact_func_y(x, y)])

    # ========================================================================
    #    2. FINITE ELEMENT AND MESH SETUP
    # ========================================================================
    nx = ny = 20
    poly_order_vel = 2
    poly_order_p = 1
    mesh_bounds = (-1.2, 1.2)

    n_refinements = 2  # No mesh refinement for simplicity
    convergence_data = []
    
    for cycle in range(n_refinements):
        nx = ny = 10 * (2**cycle)
        nodes, elems, _, corners = structured_quad(
            Lx=mesh_bounds[1] - mesh_bounds[0],
            Ly=mesh_bounds[1] - mesh_bounds[0],
            nx=nx, ny=ny,
            poly_order=poly_order_vel,
            offset=(mesh_bounds[0], mesh_bounds[0])
        )
        mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=poly_order_vel)
        
        # --- Level Set and Mesh Classification ---
        # phi > 0 is INSIDE the circle (physical domain)
        level_set = CircleLevelSet(center=domain_center, radius=domain_radius)
        mesh.classify_elements(level_set)
        mesh.classify_edges(level_set)
        mesh.build_interface_segments(level_set=level_set)

        # --- Define Domains with BitSets ---
        physical_domain = mesh.element_bitset("inside") | mesh.element_bitset("cut")
        cut_domain = mesh.element_bitset("cut")
        # Ghost penalty is applied on edges between cut cells and exterior ('outside') cells
        ghost_domain = mesh.edge_bitset("ghost_neg") 
        for domain_name, bitset in mesh._edge_bitsets.items():
            print(f"Domain name: {domain_name}, size: {bitset.cardinality()}")
        
        # --- Taylor-Hood Elements (Q2 for velocity, Q1 for pressure) ---
        mixed_element = MixedElement(mesh, field_specs={'ux': poly_order_vel, 'uy': poly_order_vel, 'p': poly_order_p})
        dof_handler = DofHandler(mixed_element, method='cg')
        dof_handler.info()
        
        # ========================================================================
        #    3. BOUNDARY CONDITIONS
        # ========================================================================
        
        # --- Tag and constrain inactive DOFs OUTSIDE the circle ---
        dof_handler.tag_dofs_from_element_bitset("inactive", "ux", "outside", strict=True)
        dof_handler.tag_dofs_from_element_bitset("inactive", "uy", "outside", strict=True)
        dof_handler.tag_dofs_from_element_bitset("inactive", "p", "outside", strict=True)

        # --- Define BC list ---
        bcs = [
            # Strong BCs for inactive DOFs in the exterior
            BoundaryCondition('ux', 'dirichlet', 'inactive', lambda x, y: u_exact_func_x(x, y)),
            BoundaryCondition('uy', 'dirichlet', 'inactive', lambda x, y: u_exact_func_y(x, y)),
            BoundaryCondition('p', 'dirichlet', 'inactive', lambda x, y: p_exact_func(x, y)),
        ]
        dof_handler.tag_dof_by_locator(
            'p_pin', 'p',
            locator=lambda X, Y: (X**2 + Y**2) < (0.8 * domain_radius)**2,  # any interior fluid point
            find_first=True
        )
        bcs.append(BoundaryCondition('p', 'dirichlet', 'p_pin', lambda x, y: p_exact_func(x, y)))
        bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]
        pdof = next(iter(dof_handler.dof_tags['p_pin']))
        xy   = dof_handler.get_all_dof_coords()[pdof]
        print("Pinned p-DOF:", pdof, "coords:", xy, "phi(xy)=", level_set(xy))
        from pycutfem.io.visualization import plot_mesh_2
        # fig, ax = plt.subplots(figsize=(15, 30))
        # plot_mesh_2(mesh, ax=ax, level_set=level_set, show=True,
        #              plot_nodes=False, elem_tags=False, edge_colors=True, plot_interface=False, resolution=300)
        # bcs_homog.append(BoundaryCondition('p', 'dirichlet', 'p_pin', lambda x, y: 0.0))

        # ========================================================================
        #    4. UFL WEAK FORM
        # ========================================================================
        print("\n--- Defining the weak form for the Cut-Stokes problem ---")

        # --- Functions and Constants ---
        velocity_space = FunctionSpace("velocity", ['ux', 'uy'],dim=1)
        u_k = VectorFunction(name="u_k", field_names=['ux', 'uy'], dof_handler=dof_handler)
        p_k = Function(name="p_k", field_name='p', dof_handler=dof_handler)
        du = VectorTrialFunction(velocity_space, dof_handler)
        dp = TrialFunction(name='dp', field_name='p', dof_handler=dof_handler)
        v = VectorTestFunction(velocity_space, dof_handler)
        q = TestFunction(name='q', field_name='p', dof_handler=dof_handler)

        f = VectorFunction("f", ['ux', 'uy'], dof_handler); f.set_values_from_function(lambda x,y: f_func(x,y))
        g = Function(name="g", field_name='p', dof_handler=dof_handler); g.set_values_from_function(g_func)
        u_exact = VectorFunction("u_exact", ['ux', 'uy'], dof_handler); u_exact.set_values_from_function(u_exact_vector_func)
        
        # --- UFL Helpers and Parameters ---
        n = FacetNormal()
        h = CellDiameter()
        mu = Constant(mu_val)
        beta_N = Constant(20.0 * poly_order_vel**2) # Nitsche penalty
        penalty_val = 0.0
        penalty_grad = 0.05
        gamma_v = Constant(penalty_val * poly_order_vel**2)
        gamma_G= Constant(penalty_grad * poly_order_vel**2)
        gamma_p  = Constant(penalty_val * poly_order_p**1)
        gamma_pG = Constant(penalty_grad * poly_order_p**1)

        # --- Integration Measures ---
        # Integrate where phi > 0
        dx_phys = dx(defined_on=physical_domain, level_set=level_set, metadata={'side': '-',"q":6})
        dGamma = dInterface(defined_on=cut_domain, level_set=level_set, metadata={"q":6})
        dGhost_stab = dGhost(defined_on=ghost_domain, level_set=level_set, metadata={"q":6,'derivs': {(0,1),(1,0)}})

        # --- Reusable UFL expressions ---
        def epsilon(u_vec): return 0.5 * (grad(u_vec) + grad(u_vec).T)

        def _dn(expr):
            """Normal derivative  n·∇expr  on an (interior) edge."""
            Dx = Derivative(expr, 1, 0)
            Dy = Derivative(expr, 0, 1)
            _ = Dx + Dy
            return n[0]*Dx + n[1]*Dy

        def grad_inner(u, v):
            """⟨∂ₙu, ∂ₙv⟩  (scalar or 2‑D vector)."""
            if getattr(u, "num_components", 1) == 1:      # scalar
                return _dn(u) * _dn(v)

            if u.num_components == v.num_components == 2: # vector
                return _dn(u[0]) * _dn(v[0]) + _dn(u[1]) * _dn(v[1])
        # --- Volume Terms ---
        residual_vol = (
            + 2 * mu * inner(epsilon(u_k), epsilon(v)) 
            + (q * div(u_k) - p_k * div(v)) 
            - dot(f, v) 
            - g * q 
        ) * dx_phys
        jacobian_vol = (
            + 2 * mu * inner(epsilon(du), epsilon(v)) 
            + (q * div(du) - dp * div(v)) 
        ) * dx_phys


        def sigma_dot_n_v(u_vec, p_scal,v_test,n):
            """
            Expanded form of (σ(u, p) · n) without using the '@' operator.

                σ(u, p)·n = μ (∇u + ∇uᵀ)·n  −  p n
            """
            # first term: μ (∇u)·n
            a = dot(grad(u_vec), n)
            # second term: μ (∇uᵀ)·n
            b = dot(grad(u_vec).T, n)
            # combine and subtract pressure part
            return  mu * dot((a + b),v_test) - p_scal * dot(v_test,n)         # vector of size 2

        # --- Nitsche Terms for Immersed Boundary ---
        residual_nitsche = (
            - sigma_dot_n_v(u_k, p_k, v, n)
            - sigma_dot_n_v(v, q, u_k-u_exact, n)
            + (beta_N / h) * dot(u_k - u_exact, v) 
        ) * dGamma
        jacobian_nitsche = (
            - sigma_dot_n_v(du,dp,v,n)
            - sigma_dot_n_v(v,q,du,n) 
            + (beta_N / h) * dot(du, v) 
        ) * dGamma
        
        # --- Ghost Penalty Stabilization Terms ---
        residual_ghost = (
            gamma_v /h * dot(jump(u_k), jump(v)) 
            + gamma_G * h * grad_inner(jump(u_k), jump(v))
            + gamma_p /h * jump(p_k) * jump(q) 
            + gamma_pG * h * grad_inner(jump(p_k), jump(q)) 
        ) * dGhost_stab
        jacobian_ghost = (
            gamma_v /h * dot(jump(du), jump(v))
            + gamma_G * h * grad_inner(jump(du), jump(v))
            + gamma_p /h * jump(dp) * jump(q)
            + gamma_pG * h * grad_inner(jump(dp), jump(q)) 
        ) * dGhost_stab

        # --- Final Weak Form ---
        residual = residual_vol + residual_nitsche + residual_ghost
        jacobian = jacobian_vol + jacobian_nitsche + jacobian_ghost

        # ========================================================================
        #    5. SOLVE AND VERIFY
        # ========================================================================
        print("\n--- Setting up and running the Newton solver ---")

        solver = NewtonSolver(
            residual_form=residual,
            jacobian_form=jacobian,
            dof_handler=dof_handler,
            mixed_element=mixed_element,
            bcs=bcs,
            bcs_homog=bcs_homog,
            newton_params=NewtonParameters(newton_tol=1e-8, max_newton_iter=10)
        )
        
        u_n = VectorFunction("u_n", ['ux', 'uy'], dof_handler)
        p_n = Function(name="p_n", field_name='p', dof_handler=dof_handler)
        u_n.nodal_values[:] = 0.0
        p_n.nodal_values[:] = 0.0
        solver.solve_time_interval(
            functions=[u_k, p_k],
            prev_functions=[u_n, p_n],
            time_params=TimeStepperParameters(max_steps=1),
            aux_functions={"f": f, "g": g, "u_exact": u_exact},
        )

        print("\n--- Post-processing and computing L2 error on internal cut domain---")
        exact = {
            'ux': lambda x,y: u_exact_vector_func(x,y)[0],
            'uy': lambda x,y: u_exact_vector_func(x,y)[1],
            'p' : lambda x,y: p_exact_func(x,y),
        }

        # velocity (φ<0 internal domain)
        l2_u = dof_handler.l2_error_on_side(
            functions=u_k,                # VectorFunction
            exact={'ux': exact['ux'], 'uy': exact['uy']},
            level_set=level_set,
            side='-',                     # inside
            quad_order=2*poly_order_vel+2
        )

        # pressure
        l2_p = dof_handler.l2_error_on_side(
            functions=p_k,                # scalar Function
            exact={'p': exact['p']},
            level_set=level_set,
            side='-',
            quad_order=2*poly_order_vel+2
        )

        # --- Optional: Export for visualization ---
        output_dir = "stokes_mms_cut_internal_results"
        os.makedirs(output_dir, exist_ok=True)
        vtk_path = os.path.join(output_dir, "stokes_mms_cut_solution.vtu")
        print(f"Writing solution to {vtk_path}")
        from pycutfem.io.vtk import export_vtk
        export_vtk(
            filename=vtk_path,
            mesh=mesh,
            dof_handler=dof_handler,
            functions={"velocity": u_k, "pressure": p_k}
        )
        
        h_max = (mesh_bounds[1] - mesh_bounds[0]) / nx
        convergence_data.append({'cycle': cycle, 'l2_u': l2_u, 'l2_p': l2_p, 'h': h_max, 'num_active_elems': physical_domain.cardinality()})
        print("\n" + "="*50)
        print(f"L2(u) on φ<0: {l2_u:.3e},  L2(p) on φ<0: {l2_p:.3e}")
        print("="*50)
        # assert l2_u < 1e-3, f"L2 error in velocity ({l2_u}) is too high!"
        # assert l2_p < 1e-3, f"L2 error in pressure ({l2_p}) is too high!"

    # ========================================================================
    #    6. PRINT CONVERGENCE TABLE
    # ========================================================================
    print("\n" + "="*60)
    print("               Convergence Table for L2-Error")
    print("="*60)
    print(f"{'Cycle':>5} | {'h':>12} | {'Active elements':>9} | {'L2-u-Error':>15} | {'L2-p-Error':>15} | {'Rate':>6}")
    print("-"*60)
    for i, data in enumerate(convergence_data):
        rate_str = "----"
        if i > 0:
            rate_u = np.log(convergence_data[i-1]['l2_u'] / data['l2_u']) / np.log(convergence_data[i-1]['h'] / data['h'])
            rate_p = np.log(convergence_data[i-1]['l2_p'] / data['l2_p']) / np.log(convergence_data[i-1]['h'] / data['h'])
            rate_str = f"{rate_u:.2f}/{rate_p:.2f}"
        print(f"{data['cycle']:>5d} | {data['h']:>12.4e} | {data['num_active_elems']:>9d} | {data['l2_u']:>15.5e} | {data['l2_p']:>15.5e} | {rate_str:>6}")
    print("="*60)
if __name__ == '__main__':
    run_stokes_mms_cut_internal()
