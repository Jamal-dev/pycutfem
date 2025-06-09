import numpy as np
import scipy.sparse as sp
import ufl.expressions as _expr                # use absolute imports
from ufl.expressions import (Constant, TrialFunction, TestFunction, Grad,
                         Inner, Restriction, Jump, Avg, FacetNormal,
                         Sum, Sub, Prod, Div)

from pycutfem.fem.reference import get_reference
from pycutfem.integration import volume, edge
from pycutfem.fem import transform
from utils.boundary import get_boundary_dofs
from pycutfem.assembly.boundary_conditions import apply_dirichlet

# This helper must be defined outside the class if used in a module
def _inverse_map(mesh, elem_id, x_phys, tol=1e-10, maxiter=10):
    """Newton solver to find reference coordinates for a physical point."""
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

class FormCompiler:
    def __init__(self, mesh,function_space, quad_order=None):
        self.mesh = mesh
        self.fs = function_space
        self.poly_order = self.fs.p
        self.quad_order = quad_order or self.poly_order + 2
        self.context = {}

    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise TypeError(f"No visit_{type(node).__name__} method for {type(node)}")

    # --- Expression Visitors ---
    def visit_Constant(self, node): return node.value
    def visit_Sum(self, node): return self.visit(node.a) + self.visit(node.b)
    def visit_Sub(self, node): return self.visit(node.a) - self.visit(node.b)
    def visit_Prod(self, node): return self.visit(node.a) * self.visit(node.b)
    def visit_Div(self, node): return self.visit(node.a) / self.visit(node.b)
    def visit_FacetNormal(self, node): return self.context['n']
    
    def visit_TrialFunction(self, node):
        if self.context['is_rhs']: raise TypeError("Cannot have TrialFunction on RHS.")
        side = self.context.get('side', 'L')
        return self.context[f'basis_{side}_trial']
        
    def visit_TestFunction(self, node):
        side = self.context.get('side', 'L')
        return self.context[f'basis_{side}_test']

    def visit_Grad(self, node):
        from ufl.expressions import Restriction        # local import, no cycle
        if isinstance(node.operand, Restriction):
            tag = node.operand.domain_tag
            if tag in ('+', '-'):                  # facet sides
                side_orig = self.context.get('side', 'L')
                self.context['side'] = tag
                val = self.visit(_expr.Grad(node.operand.f))
                self.context['side'] = side_orig
                return val
            # volume restriction: just recurse
            return self.visit(_expr.Grad(node.operand.f))

        # --- plain Trial/TestFunction ------------------------------------
        side = self.context.get('side', 'L')
        if isinstance(node.operand, TrialFunction):
            return self.context[f'grad_basis_{side}_trial']
        if isinstance(node.operand, TestFunction):
            return self.context[f'grad_basis_{side}_test']
        raise TypeError(
            f"grad() can only be applied to Trial/TestFunction, not {type(node.operand)}"
        )

    def visit_Inner(self, node):
        val_a = self.visit(node.a)
        val_b = self.visit(node.b)
        # Use einsum for robust inner product of scalars, vectors, and tensors.
        if self.context['is_rhs']: return np.einsum('i...,i...->...', val_a, val_b)
        return np.einsum('i...,j...->ij...', val_a, val_b)
    
    def visit_Restriction(self, node):
        current_tag = self.context.get('elem_tag')
        # print(f"Visiting Restriction: {node.domain_tag}, current tag: {current_tag}")
        if current_tag == node.domain_tag:
             return self.visit(node.f)
        # Return a zero of the correct shape by visiting the operand
        # and then zeroing it out. This is robust.
        ref_val = self.visit(node.f)
        return np.zeros_like(ref_val)

    def visit_Side(self, node):
        side_orig = self.context.get('side', 'L')
        self.context['side'] = node.side
        val = self.visit(node.f)
        self.context['side'] = side_orig # Reset
        return val

    def visit_Jump(self, node):
        # Jump is defined as v(+) * n(+) + v(-) * n(-)
        # Since n_R = -n_L, this becomes (v_L - v_R) for scalar fields
        val_L = self.visit(node.v.restrict('+'))
        val_R = self.visit(node.v.restrict('-'))
        return np.concatenate((val_L, -val_R))

    def visit_Avg(self, node):
        val_L = self.visit(node.v.restrict('+'))
        val_R = self.visit(node.v.restrict('-'))
        return 0.5 * np.concatenate((val_L, val_R))

    def assemble(self, equation,bcs):
        n_dofs = self.fs.num_global_dofs()
        K = sp.lil_matrix((n_dofs, n_dofs))
        F = np.zeros(n_dofs)

        self.context['is_rhs'] = False
        self._assemble_form(equation.a, K)
        
        self.context['is_rhs'] = True
        self._assemble_form(equation.L, F)

        K,F =self._apply_bcs(K, F, bcs)
        return K.tocsr(), F
    
    def _apply_bcs(self, K, F, bcs):
        """Applies strong Dirichlet boundary conditions."""
        if not bcs: return
        
        boundary_dofs = get_boundary_dofs(bcs[0].V, bcs)
        K,F =apply_dirichlet(K, F, boundary_dofs)
        return K, F

    def _assemble_form(self, form, matrix_or_vector):
        for integral in form.integrals:
            if integral.measure.domain_type == "volume":
                self._assemble_volume_integral(integral, matrix_or_vector)
            elif integral.measure.domain_type == "interior_facet":
                self._assemble_face_integral(integral, matrix_or_vector)

    def _assemble_volume_integral(self, integral, matrix_or_vector):
        is_rhs = self.context['is_rhs']
        ref = get_reference(self.mesh.element_type, self.poly_order)
        n_loc = len(ref.shape(0,0))
        element_ids = [e.id for e in self.mesh.elements_list if e.tag == integral.measure.subdomain_tag]
        
        for elem_id in element_ids:
            self.context['elem_id'] = elem_id
            self.context['elem_tag'] = integral.measure.subdomain_tag
            
            local_contrib = np.zeros((n_loc, n_loc)) if not is_rhs else np.zeros(n_loc)
            pts, wts = volume(self.mesh.element_type, self.quad_order)
            
            for (xi, eta), w in zip(pts, wts):
                J = transform.jacobian(self.mesh, elem_id, (xi, eta)); invJ_T = np.linalg.inv(J).T
                self.context['basis_L_trial'] = self.context['basis_L_test'] = ref.shape(xi, eta)
                self.context['grad_basis_L_trial'] = self.context['grad_basis_L_test'] = ref.grad(xi, eta) @ invJ_T
                val = self.visit(integral.integrand)
                local_contrib += w * abs(np.linalg.det(J)) * val
                
            dofs = np.arange(n_loc) + elem_id * n_loc
            if not is_rhs: matrix_or_vector[np.ix_(dofs, dofs)] += local_contrib
            else: matrix_or_vector[dofs] += local_contrib

    def _assemble_face_integral(self, integral, matrix_or_vector):
        is_rhs = self.context['is_rhs']
        ref = get_reference(self.mesh.element_type, self.poly_order)
        n_loc = len(ref.shape(0,0))
        
        edge_ids = [e.gid for e in self.mesh.edges_list if e.right is not None and e.tag == integral.measure.subdomain_tag]
                    
        for edge_id in edge_ids:
            edge_obj = self.mesh.edge(edge_id)
            eL, eR = edge_obj.left, edge_obj.right
            local_edge_idx = self.mesh.elements_list[eL].edges.index(edge_id)
            pts, wts = edge(self.mesh.element_type, local_edge_idx, self.quad_order)

            n_loc_face = n_loc * 2
            local_contrib = np.zeros((n_loc_face, n_loc_face)) if not is_rhs else np.zeros(n_loc_face)
            
            for (xi, eta), w in zip(pts, wts):
                self.context['n'] = edge_obj.normal
                jac1d = transform.jacobian_1d(self.mesh, eL, (xi, eta), local_edge_idx)
                
                # Setup context for Left side ('+')
                JL = transform.jacobian(self.mesh, eL, (xi, eta)); invJL_T = np.linalg.inv(JL).T
                self.context['basis_+_trial'] = self.context['basis_+_test'] = ref.shape(xi, eta)
                self.context['grad_basis_+_trial'] = self.context['grad_basis_+_test'] = ref.grad(xi, eta) @ invJL_T

                # Setup context for Right side ('-')
                x_phys = transform.x_mapping(self.mesh, eL, (xi, eta))
                xi_R, eta_R = _inverse_map(self.mesh, eR, x_phys)
                JR = transform.jacobian(self.mesh, eR, (xi_R, eta_R)); invJR_T = np.linalg.inv(JR).T
                self.context['basis_-_trial'] = self.context['basis_-_test'] = ref.shape(xi_R, eta_R)
                self.context['grad_basis_-_trial'] = self.context['grad_basis_-_test'] = ref.grad(xi_R, eta_R) @ invJR_T
                
                val = self.visit(integral.integrand)
                local_contrib += w * jac1d * val

            dofsL = np.arange(n_loc) + eL * n_loc
            dofsR = np.arange(n_loc) + eR * n_loc
            dofs = np.concatenate([dofsL, dofsR])
            if not is_rhs: matrix_or_vector[np.ix_(dofs, dofs)] += local_contrib
            else: matrix_or_vector[dofs] += local_contrib
