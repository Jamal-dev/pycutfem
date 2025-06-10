import numpy as np
import scipy.sparse as sp
import ufl.expressions as _expr
from ufl.expressions import (Constant, TrialFunction, TestFunction, Grad,
                             Inner, Restriction, Jump, Avg, FacetNormal,
                             Sum, Sub, Prod, Div, Derivative)
from ufl.analytic import Analytic
from pycutfem.fem.reference import get_reference
from pycutfem.integration import volume, edge
from pycutfem.fem import transform
from pycutfem.core.dofhandler import DofHandler

def _inverse_map(mesh, elem_id, x_phys, tol=1e-10, maxiter=10):
    ref = get_reference(mesh.element_type, mesh.poly_order)
    xi_ref = np.array([0.0, 0.0])
    for _ in range(maxiter):
        x_curr = transform.x_mapping(mesh, elem_id, xi_ref)
        J = transform.jacobian(mesh, elem_id, xi_ref)
        if abs(np.linalg.det(J)) < 1e-12: break
        delta = np.linalg.solve(J, x_phys - x_curr)
        xi_ref += delta
        if np.linalg.norm(delta) < tol: break
    return xi_ref

def _find_functions(expression):
    if isinstance(expression, TrialFunction): return (expression, None)
    if isinstance(expression, TestFunction): return (None, expression)
    trial_func, test_func = None, None
    if hasattr(expression, 'a') and hasattr(expression, 'b'):
        tf1, tsf1 = _find_functions(expression.a)
        tf2, tsf2 = _find_functions(expression.b)
        trial_func, test_func = tf1 or tf2, tsf1 or tsf2
    elif hasattr(expression, 'operand'):
        trial_func, test_func = _find_functions(expression.operand)
    elif hasattr(expression, 'f'):
        trial_func, test_func = _find_functions(expression.f)
    elif hasattr(expression, 'v'):
        trial_func, test_func = _find_functions(expression.v)
    return trial_func, test_func

class FormCompiler:
    def __init__(self, dof_handler: DofHandler, quad_order=None):
        self.dof_handler = dof_handler
        self.quad_order = quad_order
        self.context = {}

    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise TypeError(f"No visit_{type(node).__name__} method for {type(node)}")

    def visit_Constant(self, node): return node.value
    def visit_Sum(self, node): return self.visit(node.a) + self.visit(node.b)
    def visit_Sub(self, node): return self.visit(node.a) - self.visit(node.b)
    
    def visit_Prod(self, node):
        val_a = self.visit(node.a)
        val_b = self.visit(node.b)
        if not self.context['is_rhs']:
            _, test_fn_a = _find_functions(node.a)
            trial_fn_b, _ = _find_functions(node.b)
            if test_fn_a and trial_fn_b: return np.outer(val_a, val_b)
            trial_fn_a, _ = _find_functions(node.a)
            _, test_fn_b = _find_functions(node.b)
            if trial_fn_a and test_fn_b: return np.outer(val_b, val_a)
        return val_a * val_b

    def visit_Div(self, node): return self.visit(node.a) / self.visit(node.b)
    def visit_FacetNormal(self, node): return self.context['n']
    def visit_Analytic(self, node): return node.eval(self.context['x_phys'])

    def visit_TrialFunction(self, node):
        if self.context['is_rhs']: raise TypeError("Cannot have TrialFunction on RHS.")
        return self.context['basis_trial']
        
    def visit_TestFunction(self, node): return self.context['basis_test']

    def visit_Grad(self, node):
        if isinstance(node.operand, TrialFunction): return self.context['grad_basis_trial']
        if isinstance(node.operand, TestFunction): return self.context['grad_basis_test']
        raise TypeError(f"grad() can only be applied to Trial/TestFunction, not {type(node.operand)}")
    
    def visit_Derivative(self, node):
        grad_basis = self.visit(Grad(node.f))
        return grad_basis[:, node.component_index]

    def visit_Inner(self, node):
        val_a = self.visit(node.a)
        val_b = self.visit(node.b)
        if self.context['is_rhs']: return np.einsum('i...,i...->...', val_a, val_b)
        return np.einsum('ik,jk->ij', val_a, val_b)
    
    def assemble(self, system, bcs):
        n_dofs = self.dof_handler.total_dofs
        K = sp.lil_matrix((n_dofs, n_dofs))
        F = np.zeros(n_dofs)

        if not isinstance(system, (list, tuple)):
            system = [system]

        for equation in system:
            self.context['is_rhs'] = False
            self._assemble_form(equation.a, K)
            self.context['is_rhs'] = True
            self._assemble_form(equation.L, F)

        K, F = self._apply_bcs(K, F, bcs)
        return K.tocsr(), F
    
    def _apply_bcs(self, K, F, bcs):
        """
        Applies strong Dirichlet boundary conditions by delegating to the DofHandler.
        """
        if not bcs: return K, F
        
        # The DofHandler is now solely responsible for interpreting the BCs
        dirichlet_data = self.dof_handler.get_dirichlet_data(bcs)
        
        if not dirichlet_data: return K, F
        
        dofs = list(dirichlet_data.keys())
        values = np.array(list(dirichlet_data.values()))
        
        # Apply by lifting
        u_d = np.zeros_like(F)
        u_d[dofs] = values
        F -= K @ u_d
        
        K_lil = K.tolil()
        for dof in dofs:
            K_lil[dof, :] = 0
            K_lil[:, dof] = 0
            K_lil[dof, dof] = 1.0
            
        F[dofs] = values
        return K_lil, F

    def _assemble_form(self, form, matrix_or_vector):
        for integral in form.integrals:
            if integral.measure.domain_type == "volume":
                self._assemble_volume_integral(integral, matrix_or_vector)

    def _assemble_volume_integral(self, integral, matrix_or_vector):
        is_rhs = self.context['is_rhs']
        
        trial_fn, test_fn = _find_functions(integral.integrand)
        if not test_fn: raise ValueError(f"Integral is missing a TestFunction: {integral.integrand}")
            
        test_field = test_fn.field_name
        trial_field = trial_fn.field_name if trial_fn else None

        test_mesh = self.dof_handler.fe_map[test_field]
        test_ref = get_reference(test_mesh.element_type, test_mesh.poly_order)
        test_map = self.dof_handler.element_maps[test_field]
        n_loc_test = len(test_ref.shape(0,0))
        
        if trial_field:
            trial_mesh = self.dof_handler.fe_map[trial_field]
            trial_ref = get_reference(trial_mesh.element_type, trial_mesh.poly_order)
            trial_map = self.dof_handler.element_maps[trial_field]
            n_loc_trial = len(trial_ref.shape(0,0))
        else:
             n_loc_trial = 0

        quad_order = self.quad_order or test_mesh.poly_order + 2

        for elem_id in range(len(test_mesh.elements_list)):
            local_contrib = np.zeros((n_loc_test, n_loc_trial)) if not is_rhs else np.zeros(n_loc_test)
            pts, wts = volume(test_mesh.element_type, quad_order)
            
            for (xi, eta), w in zip(pts, wts):
                self.context['x_phys'] = transform.x_mapping(test_mesh, elem_id, (xi, eta))
                J_test = transform.jacobian(test_mesh, elem_id, (xi, eta)); invJ_T_test = np.linalg.inv(J_test).T
                self.context['basis_test'] = test_ref.shape(xi, eta)
                self.context['grad_basis_test'] = test_ref.grad(xi, eta) @ invJ_T_test

                if not is_rhs and trial_map:
                    # Assumes trial and test elements are geometrically identical
                    J_trial = J_test 
                    invJ_T_trial = invJ_T_test
                    self.context['basis_trial'] = trial_ref.shape(xi, eta)
                    self.context['grad_basis_trial'] = trial_ref.grad(xi, eta) @ invJ_T_trial
                
                val = self.visit(integral.integrand)
                local_contrib += w * abs(np.linalg.det(J_test)) * val
            
            dofs_row = test_map[elem_id]
            if not is_rhs and trial_map:
                dofs_col = trial_map[elem_id]
                matrix_or_vector[np.ix_(dofs_row, dofs_col)] += local_contrib
            else:
                matrix_or_vector[dofs_row] += local_contrib
