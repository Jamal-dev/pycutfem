import numpy as np
import scipy.sparse as sp
from ufl.expressions import (Constant, TrialFunction, TestFunction, Grad,
                             Inner, Sum, Sub, Prod, Div, Derivative)
from ufl.analytic import Analytic

from pycutfem.fem.reference import get_reference
from pycutfem.integration import volume, edge
from pycutfem.fem import transform
from pycutfem.core.dofhandler import DofHandler

def _find_functions(expression):
    """Recursively finds the first TrialFunction and TestFunction in an expression tree."""
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

def _flatten_integrand(expression):
    """
    Decomposes a complex expression into a list of (sign, term) tuples,
    distributing products over sums to handle mixed-element forms.
    """
    if isinstance(expression, Sum):
        return _flatten_integrand(expression.a) + _flatten_integrand(expression.b)
    if isinstance(expression, Sub):
        return _flatten_integrand(expression.a) + [(-s, t) for s, t in _flatten_integrand(expression.b)]
    
    # FIX: Distribute products over sums to correctly handle mixed terms.
    # This is crucial for expressions like q * (div(u)) which expands to q*dux/dx + q*duy/dy.
    if isinstance(expression, Prod):
        # Case (A+B)*C -> A*C + B*C
        if isinstance(expression.a, Sum):
            return _flatten_integrand(Prod(expression.a.a, expression.b)) + _flatten_integrand(Prod(expression.a.b, expression.b))
        # Case A*(B+C) -> A*B + A*C
        if isinstance(expression.b, Sum):
            return _flatten_integrand(Prod(expression.a, expression.b.a)) + _flatten_integrand(Prod(expression.a, expression.b.b))
        # Handle subtraction distribution as well
        # Case (A-B)*C -> A*C - B*C
        if isinstance(expression.a, Sub):
            return _flatten_integrand(Prod(expression.a.a, expression.b)) + [(-s, t) for s, t in _flatten_integrand(Prod(expression.a.b, expression.b))]
        # Case A*(B-C) -> A*B - A*C
        if isinstance(expression.b, Sub):
            return _flatten_integrand(Prod(expression.a, expression.b.a)) + [(-s, t) for s, t in _flatten_integrand(Prod(expression.a, expression.b.b))]

    return [(1, expression)]


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
            if hasattr(val_a, 'ndim') and hasattr(val_b, 'ndim') and val_a.ndim == 1 and val_b.ndim == 1:
                _, test_fn_a = _find_functions(node.a)
                trial_fn_b, _ = _find_functions(node.b)
                if test_fn_a and trial_fn_b:
                    return np.outer(val_a, val_b)

                trial_fn_a, _ = _find_functions(node.a)
                _, test_fn_b = _find_functions(node.b)
                if trial_fn_a and test_fn_b:
                    return np.outer(val_b, val_a)

        return val_a * val_b

    def visit_Analytic(self, node): return node.eval(self.context['x_phys'])

    def visit_TrialFunction(self, node):
        if self.context['is_rhs']: raise TypeError("Cannot have TrialFunction on RHS.")
        return self.context['basis_values'][node.field_name]['val']
        
    def visit_TestFunction(self, node):
        return self.context['basis_values'][node.field_name]['val']

    def visit_Grad(self, node):
        field = node.operand.field_name
        if isinstance(node.operand, TrialFunction):
            return self.context['basis_values'][field]['grad']
        if isinstance(node.operand, TestFunction):
            return self.context['basis_values'][field]['grad']
        raise TypeError(f"grad() can only be applied to Trial/TestFunction.")
    
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
        if not bcs: return K, F
        dirichlet_data = self.dof_handler.get_dirichlet_data(bcs)
        if not dirichlet_data: return K, F
        dofs = list(dirichlet_data.keys())
        values = np.array(list(dirichlet_data.values()))
        u_d = np.zeros_like(F)
        u_d[dofs] = values
        F -= K @ u_d
        K_lil = K.tolil()
        for dof in dofs:
            K_lil[dof, :] = 0; K_lil[:, dof] = 0; K_lil[dof, dof] = 1.0
        F[dofs] = values
        return K_lil, F

    def _assemble_form(self, form, matrix_or_vector):
        for integral in form.integrals:
            if integral.measure.domain_type == "volume":
                self._assemble_volume_integral(integral, matrix_or_vector)

    def _assemble_volume_integral(self, integral, matrix_or_vector):
        is_rhs = self.context['is_rhs']
        
        terms = _flatten_integrand(integral.integrand)
        
        all_fields = set()
        for _, term in terms:
            trial_fn, test_fn = _find_functions(term)
            if trial_fn: all_fields.add(trial_fn.field_name)
            if test_fn: all_fields.add(test_fn.field_name)

        primary_test_fn = _find_functions(integral.integrand)[1]
        if not primary_test_fn: raise ValueError("Integral is missing a TestFunction.")
        mesh = self.dof_handler.fe_map[primary_test_fn.field_name]
        
        quad_order = self.quad_order or mesh.poly_order + 2

        for elem_id in range(len(mesh.elements_list)):
            basis_values_at_quad_points = []
            pts, wts = volume(mesh.element_type, quad_order)
            for (xi, eta), w in zip(pts, wts):
                basis_vals_for_point = {}
                for field in all_fields:
                    field_mesh = self.dof_handler.fe_map[field]
                    field_ref = get_reference(field_mesh.element_type, field_mesh.poly_order)
                    J = transform.jacobian(field_mesh, elem_id, (xi, eta))
                    invJ_T = np.linalg.inv(J).T
                    basis_vals_for_point[field] = {
                        'val': field_ref.shape(xi, eta),
                        'grad': field_ref.grad(xi, eta) @ invJ_T
                    }
                basis_values_at_quad_points.append(basis_vals_for_point)

            for sign, term in terms:
                trial_fn, test_fn = _find_functions(term)
                if not test_fn: 
                    if is_rhs:
                        test_fn = primary_test_fn
                    else:
                        continue

                test_field = test_fn.field_name
                trial_field = trial_fn.field_name if trial_fn else None

                test_mesh_term = self.dof_handler.fe_map[test_field]
                test_ref_term = get_reference(test_mesh_term.element_type, test_mesh_term.poly_order)
                n_loc_test = len(test_ref_term.shape(0,0))
                
                if trial_field:
                    trial_mesh_term = self.dof_handler.fe_map[trial_field]
                    trial_ref_term = get_reference(trial_mesh_term.element_type, trial_mesh_term.poly_order)
                    n_loc_trial = len(trial_ref_term.shape(0,0))
                else:
                    n_loc_trial = 0
                
                local_contrib = np.zeros((n_loc_test, n_loc_trial)) if not is_rhs else np.zeros(n_loc_test)

                for i, ((xi, eta), w) in enumerate(zip(pts, wts)):
                    self.context['x_phys'] = transform.x_mapping(mesh, elem_id, (xi, eta))
                    self.context['basis_values'] = basis_values_at_quad_points[i]
                    
                    val = self.visit(term)
                    J = transform.jacobian(mesh, elem_id, (xi, eta))
                    local_contrib += sign * w * abs(np.linalg.det(J)) * val

                dofs_row = self.dof_handler.element_maps[test_field][elem_id]
                if not is_rhs:
                    if trial_field:
                        dofs_col = self.dof_handler.element_maps[trial_field][elem_id]
                        matrix_or_vector[np.ix_(dofs_row, dofs_col)] += local_contrib
                else:
                    matrix_or_vector[dofs_row] += local_contrib
