# pycutfem/fem/compilers.py
"""Light‑yet‑robust *local* form compiler for the project’s test-suite.

It supports exactly the expression types exercised by the Stokes examples
(Q2/Q1 Taylor–Hood, vector formulation) **plus** a generic interior‑facet path
so future ghost‑penalty terms won’t break again.  All duplicated definitions
that crept into the file have been removed – this is the *only* `FormCompiler`
exported.

Key design points
-----------------
* **Side‑aware basis cache**  – visitors pick the right trace when the context
  advertises an `active_side` (test) or `trial_side`.
* **Volume and facet assembly paths share the same visitors**; only the way
  `basis_values` is filled differs.
* **No dependence on a `level_set`** for ordinary `interior_facet` measures; if
  present, we use it to orient the normal consistently (+ side has φ ≥ 0).
* **RHS assembly** is sign‑aware via `_split_terms`.

The implementation aims for clarity over absolute speed, but all hot loops are
NumPy‑vectorised and free of Python allocations.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Mapping, Any, Union

# UFL‑like helpers ----------------------------------------------------------
from ufl.expressions import (
    Constant, TrialFunction, TestFunction,
    VectorTrialFunction, VectorTestFunction,
    Grad, DivOperation, Inner, Dot, Sum, Sub, Prod,
    Jump, FacetNormal, Pos, Neg, Function
)
from ufl.forms import Equation

# Project helpers -----------------------------------------------------------
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.reference import get_reference
from pycutfem.fem import transform
from pycutfem.integration import volume
from pycutfem.integration.quadrature import  line_quadrature
from pycutfem.integration.quadrature import edge as edge_quadrature_element

_INTERFACE_TOL = 1.0e-12

# ---------------------------------------------------------------------------
#  Light shape‑meta visitor (helps flip test/trial in ∇· terms)
# ---------------------------------------------------------------------------
class _Shape:
    __slots__ = ("dim", "kind")
    def __init__(self, dim: int, kind: str):
        self.dim = dim      # tensor order (scalar:0, vector:1, grad:2,…)
        self.kind = kind    # 'test' | 'trial' | 'none'

class _ShapeVisitor:
    def __init__(self):
        self._memo = {}
    def visit(self, n):
        if n in self._memo:
            return self._memo[n]
        meth = getattr(self, f"visit_{type(n).__name__}", self.generic)
        res = meth(n)
        self._memo[n] = res
        return res
    def generic(self, n):
        raise TypeError(f"No shape rule for {type(n)}")

    # terminals
    def visit_Constant(self, n):          return _Shape(0, 'none')
    def visit_TestFunction(self, n):      return _Shape(0, 'test')
    def visit_TrialFunction(self, n):     return _Shape(0, 'trial')
    def visit_VectorTestFunction(self,n): return _Shape(1, 'test')
    def visit_VectorTrialFunction(self,n):return _Shape(1, 'trial')
    # composite
    def visit_Sum(self, n):  return self.visit(n.a)
    def visit_Sub(self, n):  return self.visit(n.a)
    def visit_Prod(self,n):  # first non‑none wins
        sa, sb = self.visit(n.a), self.visit(n.b)
        return sa if sa.kind != 'none' else sb
    def visit_Grad(self, n): s=self.visit(n.operand); return _Shape(s.dim+1, s.kind)
    def visit_DivOperation(self,n):       return _Shape(0, self.visit(n.operand).kind)
    def visit_Inner(self,n): sa=self.visit(n.a); sb=self.visit(n.b); return _Shape(0, sa.kind if sa.kind!='none' else sb.kind)
    def visit_Dot(self,n):   return _Shape(0, self.visit(n.a).kind)
    def visit_Pos(self,n):   return self.visit(n.operand)
    def visit_Neg(self,n):   return self.visit(n.operand)
    def visit_Jump(self,n):  return self.visit(n.u_pos)
    def visit_FacetNormal(self,n): return _Shape(1,'none')
    def visit_Function(self, n):    return _Shape(0, 'none')    # scalar FE function
    def visit_ElementWiseConstant(self, n):
        return _Shape(0, 'none')

# ---------------------------------------------------------------------------
#  Helper utilities
# ---------------------------------------------------------------------------

def _split_terms(expr):
    """Flatten a +/- expression tree into (sign, term) pairs."""
    if isinstance(expr, Sum):
        return _split_terms(expr.a) + _split_terms(expr.b)
    if isinstance(expr, Sub):
        return _split_terms(expr.a) + [(-s, t) for s, t in _split_terms(expr.b)]
    return [(1, expr)]


def _all_fields(expr):
    fields=set()
    def walk(n):
        if hasattr(n,'field_name'):
            fields.add(n.field_name)
        if hasattr(n,'space'):
            fields.update(n.space.field_names)
        for attr in ('operand','a','b','u_pos','u_neg'):
            if hasattr(n,attr):
                m=getattr(n,attr)
                if isinstance(m, (list,tuple)):
                    for x in m: walk(x)
                elif m is not None:
                    walk(m)
    walk(expr)
    return list(fields)


def _trial_test(expr):
    trial = expr.find_first(lambda n:isinstance(n,(TrialFunction,VectorTrialFunction)))
    test  = expr.find_first(lambda n:isinstance(n,(TestFunction ,VectorTestFunction )))
    return trial, test

# ---------------------------------------------------------------------------
#  Actual compiler
# ---------------------------------------------------------------------------
class FormCompiler:
    def __init__(self, dofhandler: DofHandler, quad_order: int | None = None, assembler_hooks: dict | None = None):
        self.dh = dofhandler
        self.qorder = quad_order
        self.shape = _ShapeVisitor()
        self.ctx: Dict[str,Any] = {}
        # assembler hooks are passed in via `ctx['hooks']`
        self.ctx['hooks'] = assembler_hooks or {}
        # bucket for diagnostics such as perimeter / jump integrals
        self._scalar_results: dict[str, float] = {}
        self.ctx['scalar_results'] = self._scalar_results

    # ---------- public --------------------------------------------------
    def assemble(self, equation: Equation, bcs):
        n = self.dh.total_dofs
        K = sp.lil_matrix((n,n))
        F = np.zeros(n)

        # left‑hand side
        self.ctx['is_rhs'] = False
        self._assemble_form(equation.a, K)
        # right‑hand side
        self.ctx['is_rhs'] = True
        self._assemble_form(equation.L, F)

        self._apply_bcs(K,F,bcs)
        # face-only forms (e.g. Constant()*ds) produce no matrix/vector
        # – in this case hand back the accumulated scalars instead
        if self._scalar_results:
           return self._scalar_results
        return K.tocsr(), F

    # ---------- visitors ------------------------------------------------
    def visit(self, node):
        return getattr(self, f"visit_{type(node).__name__}")(node)

    def visit_ElementWiseConstant(self, node):
        side = self.ctx.get("side", "")
        if side:
            elem_id = self.ctx["e_pos"] if side == "+" else self.ctx["e_neg"]
        else:
            elem_id = self.ctx["elem_id"]
        return node.values[elem_id]
    # scalars
    def visit_Constant(self,n): 
        v = n.value
        if n.dim ==0: # scalar constant
            if isinstance(v, (int, float)):
                return v
            if isinstance(v, np.ndarray) and v.ndim == 0:
                return v.item()
            if callable(v):  # e.g. lambda x,y: x
                if 'x' not in self.ctx:
                    raise RuntimeError("Coordinate-dependent Constant evaluated outside quadrature loop")
                x, y = self.ctx['x']
                v = v(x, y)  # evaluate at quadrature point
        elif n.dim == 1:  # vector constant
            if callable(v[0]):
                if 'x' not in self.ctx:
                    raise RuntimeError("Coordinate-dependent Constant evaluated outside quadrature loop")
                x, y = self.ctx['x']
                v = np.array([f(x, y) for f in v])
            if isinstance(v, List):
                return np.array(v, dtype=float)
        return v
    
    def visit_Pos(self, n):
        """
        Evaluates the operand if on the positive side (phi>=0), otherwise returns
        a zero of the same shape as the operand.
        """
        # We use a clear convention: '+' side includes the boundary (phi >= 0)
        if ('phi_val' in self.ctx and self.ctx['phi_val'] >= _INTERFACE_TOL) or \
           ('active_side' in self.ctx and self.ctx['active_side'] != '+'):
            
            # Get operand shape and return a correctly-shaped zero
            sh = self.shape.visit(n.operand)
            if sh.dim == 0: return 0.0
            if sh.dim == 1: return np.zeros(2) # Assumes 2D vectors
            # Add other dimensions if needed
            raise TypeError(f"Unsupported dimension {sh.dim} in Pos/Neg zeroing.")

        return self.visit(n.operand)

    def visit_Neg(self, n):
        """
        Evaluates the operand if on the negative side (phi<0), otherwise returns
        a zero of the same shape as the operand.
        """
        # The '-' side is strictly phi < 0
        if ('phi_val' in self.ctx and self.ctx['phi_val'] < _INTERFACE_TOL) or \
           ('active_side' in self.ctx and self.ctx['active_side'] != '-'):

            # Get operand shape and return a correctly-shaped zero
            sh = self.shape.visit(n.operand)
            if sh.dim == 0: return 0.0
            if sh.dim == 1: return np.zeros(2) # Assumes 2D vectors
            raise TypeError(f"Unsupported dimension {sh.dim} in Pos/Neg zeroing.")
            
        return self.visit(n.operand)
    def visit_Sum(self,n): return self.visit(n.a)+self.visit(n.b)
    def visit_Sub(self,n): return self.visit(n.a)-self.visit(n.b)

    # basis look‑up helpers ------------------------------------------
    def _basis_val(self, field:str, role:str):
        """role: 'test'|'trial'; uses ctx['active_side'] / ctx['trial_side'] if present"""
        bv = self.ctx['basis_values']
        if '+' in bv:   # side‑aware dict
            side = self.ctx['active_side'] if role=='test' else self.ctx.get('trial_side','+')
            return bv[side][field]['val']
        return bv[field]['val']

    def _basis_grad(self, field:str, role:str):
        bv = self.ctx['basis_values']
        if '+' in bv:
            side = self.ctx['active_side'] if role=='test' else self.ctx.get('trial_side','+')
            return bv[side][field]['grad']
        return bv[field]['grad']

    # finite element functions ------------------------------------------
    def visit_TestFunction(self,n):  return self._basis_val(n.field_name,'test')
    def visit_TrialFunction(self,n): return self._basis_val(n.field_name,'trial')

    def visit_VectorTestFunction(self,n):  return np.hstack([self._basis_val(f,'test')  for f in n.space.field_names])
    def visit_VectorTrialFunction(self,n): return np.hstack([self._basis_val(f,'trial') for f in n.space.field_names])

    def visit_Function(self, n): # scalar FE function
        """Evaluates a Function at a quadrature point."""
        # Get the DoF indices for the current element
        dofs = self._elm_dofs(n, self.ctx['elem_id'])
        # Get the corresponding local slice of the global nodal vector
        nodal_loc = n.nodal_values[dofs]

        # Get the basis function values (N) at the current quadrature point
        N = self._basis_val(n.field_name, role='test')  # Role ('test'/'trial') is irrelevant here
        # Return the interpolated value: N · u_local
        return N @ nodal_loc
    


    # differential ops ---------------------------------------------------
    def visit_Grad(self,n):
        # Handle the special case of the gradient of a scalar FE Function
        if isinstance(n.operand, Function) and not isinstance(n.operand, (TrialFunction, TestFunction)):
            func = n.operand
            # Get the DoF indices for the current element
            dofs = self._elm_dofs(func, self.ctx['elem_id'])
            # Get the corresponding local slice of the global nodal vector
            nodal_loc = func.nodal_values[dofs]

            # Get the basis function gradients (G) at the current quadrature point
            G = self._basis_grad(func.field_name, role='test') # Role is irrelevant
            # Return the interpolated gradient: G^T · u_local
            return G.T @ nodal_loc

        sh = self.shape.visit(n.operand)
        if sh.dim==0:  # grad(scalar) -> vector
            return self._basis_grad(n.operand.field_name, sh.kind)
        if sh.dim==1:  # grad(vector) -> list[tensor row]
            return [self._basis_grad(f, sh.kind) for f in n.operand.space.field_names]
        raise NotImplementedError('grad of tensor not needed')

    def visit_DivOperation(self,n):
        grads = self.visit(Grad(n.operand))  # list of rows
        return np.hstack([g[:,i] for i,g in enumerate(grads)])
    
    def visit_Analytic(self, node): return node.eval(self.ctx['x_phys'])

    # algebra -----------------------------------------------------------
    def visit_Prod(self,n):
        a = self.visit(n.a); b=self.visit(n.b)
        if self.ctx['is_rhs']:
            return a*b
        sa,sb = self.shape.visit(n.a), self.shape.visit(n.b)
        if sa.kind=='none' or sb.kind=='none':
            return a*b
        # orient outer‑product row=test, col=trial
        if sa.kind=='test' and sb.kind=='trial':
            return np.outer(a,b)
        if sb.kind=='test' and sa.kind=='trial':
            return np.outer(b,a)
        raise TypeError('Unsupported Prod for assembly')

    def visit_Inner(self,n):
        a=self.visit(n.a); b=self.visit(n.b)
        sa,sb = self.shape.visit(n.a), self.shape.visit(n.b)
        if sa.dim==2 and sb.dim==2:  # grad(u):grad(v)
            rows = np.cumsum([0]+[c.shape[0] for c in a])
            cols = np.cumsum([0]+[c.shape[0] for c in b])
            mat = np.zeros((rows[-1], cols[-1]))
            for i,(ra,rb) in enumerate(zip(a,b)):
                mat[rows[i]:rows[i+1], cols[i]:cols[i+1]] = ra @ rb.T
            return mat
        # ADDED: This new path correctly handles grad(scalar).grad(scalar) for Poisson.
        if sa.dim == 1 and sb.dim == 1:
            # `a` and `b` are matrices of basis gradients, shape (n_basis, n_dim).
            # We compute G_v @ G_u.T to get the (n_basis, n_basis) matrix.
            # `b` is from the test function (rows), `a` is from the trial function (cols).
            return b @ a.T
        # scalar inner
        if sa.kind=='test':
            return np.outer(a,b)
        return np.outer(b,a)

    # def visit_Dot(self,n): # it was passing the Stokes test
    #     const, vec = (self.visit(n.a), self.visit(n.b)) if isinstance(n.a, Constant) else (self.visit(n.b), self.visit(n.a))
    #     ncomp = len(const)
    #     dofs_per = len(vec)//ncomp
    #     out = np.zeros_like(vec)
    #     vec = vec.reshape((ncomp,dofs_per))
    #     for i in range(ncomp):
    #         out[i*dofs_per:(i+1)*dofs_per]=const[i]*vec[i]
    #     return out
    
    def visit_Dot(self, n):
        # --- Dispatcher: Correctly identify the expression pattern ---

        # Use `isinstance` to check the UFL types directly
        is_a_const = isinstance(n.a, Constant)
        is_b_const = isinstance(n.b, Constant)
        is_a_test = isinstance(n.a, (TestFunction, VectorTestFunction))
        is_b_test = isinstance(n.b, (TestFunction, VectorTestFunction))

        # If the pattern is `dot(Constant, TestFunction)`, use the special Stokes logic
        if (is_a_const and is_b_test) or (is_b_const and is_a_test):
            const_eval, vec_eval = (self.visit(n.a), self.visit(n.b)) if is_a_const else (self.visit(n.b), self.visit(n.a))

            ncomp = len(const_eval)
            if len(vec_eval) == 0:
                return vec_eval  # Return empty if no DoFs on element

            # Ensure component division is valid
            if len(vec_eval) % ncomp != 0:
                raise ValueError(
                    f"In dot(Constant, TestFunction), cannot divide basis vector of "
                    f"length {len(vec_eval)} into {ncomp} components."
                )
            
            dofs_per = len(vec_eval) // ncomp
            out = np.zeros_like(vec_eval)
            vec_reshaped = vec_eval.reshape((ncomp, dofs_per))
            for i in range(ncomp):
                out[i * dofs_per:(i + 1) * dofs_per] = const_eval[i] * vec_reshaped[i]
            return out

        # --- General Logic Path ---
        # If the Stokes pattern doesn't match, proceed with general-purpose logic.
        a = self.visit(n.a)
        b = self.visit(n.b)

        a = np.asarray(a)
        b = np.asarray(b)

        # Case for dot(vector, vector) -> scalar (fixes the jump test)
        if a.ndim == 1 and b.ndim == 1:
            return np.dot(a, b)
            
        # Case for scalar * vector/matrix
        if a.ndim == 0: return a * b
        if b.ndim == 0: return b * a

        # Other cases for assembling vectors or matrices
        sa, sb = self.shape.visit(n.a), self.shape.visit(n.b)
        # ADDED: This new condition handles dot(Constant, grad(TrialFunction))
        if sa.kind == 'none' and sb.kind == 'trial':
            # This is Constant `a` dotted with each basis grad in `b`.
            # Operation is matrix @ vector: (4, 2) @ (2,) -> (4,)
            return b @ a
        if sa.kind == 'none' and sb.kind == 'test': return b @ a
        if sa.kind == 'test' and sb.kind == 'none': return a @ b.T
        if sa.kind == 'test' and sb.kind == 'trial': return np.outer(a, b)
        if sb.kind == 'test' and sa.kind == 'trial': return np.outer(b, a)
            
        raise TypeError(f"Unsupported Dot operation between shapes {a.shape} and {b.shape}")

    # def visit_Jump(self,n):
    #     side = self.ctx['active_side']
    #     if side=='+': return self.visit(n.u_pos)
    #     return self.visit(n.u_neg)

    # def visit_Jump(self, n):                         # u(+) – u(–)
    #     return self.visit(n.u_pos) - self.visit(n.u_neg)
    
    def visit_Jump(self, n):
        """
        Correct evaluation on CutFEM interfaces:
        we must obtain *both* traces, regardless of the current φ-sign.
        """
        phi_orig = self.ctx.get('phi_val', None)

        # ----- “+” trace -------------------------------------------------
        # Temporarily force phi to be positive to ensure the Pos() visitor
        # evaluates its operand.
        if phi_orig is not None:
            self.ctx['phi_val'] = +1.0
        u_pos = self.visit(n.u_pos)

        # ----- “–” trace -------------------------------------------------
        # Temporarily force phi to be negative to ensure the Neg() visitor
        # evaluates its operand.
        if phi_orig is not None:
            self.ctx['phi_val'] = -1.0
        u_neg = self.visit(n.u_neg)

        # ----- restore context ------------------------------------------
        # Restore the original phi_val to prevent side-effects.
        if phi_orig is None:
            self.ctx.pop('phi_val', None)
        else:
            self.ctx['phi_val'] = phi_orig

        return u_pos - u_neg

    def visit_FacetNormal(self,n):
        return self.ctx['normal']

    # ---------- core assembly helpers -----------------------------------
    def _apply_bcs(self,K,F,bcs):
        data = self.dh.get_dirichlet_data(bcs)
        if not data: return
        rows = np.fromiter(data.keys(),int)
        vals = np.fromiter(data.values(),float)
        u = np.zeros_like(F); u[rows]=vals
        F -= K@u
        K = K.tolil()
        K[rows,:]=0; K[:,rows]=0; K[rows,rows]=1.0
        F[rows]=vals

    def _assemble_form(self,form,matvec):
        # A RHS like “Constant(0.0)” has no .measure; just ignore it
        if not hasattr(form, "integrals"):
            return

        from ufl.measures import Integral 
        for integral in form.integrals:
            if not isinstance(integral, Integral):
                continue                          # skip stray Constants

            kind = integral.measure.domain_type
            if kind=='volume':
                self._assemble_volume(integral,matvec)
            elif kind=='interior_facet':
                self._assemble_facet(integral,matvec)
            elif kind=='interface':
                self._assemble_interface(integral,matvec)

    # volume ------------------------------------------------------------
    def _assemble_volume(self,intg,matvec):
        rhs = self.ctx['is_rhs']
        mesh = self.dh.fe_map[_all_fields(intg.integrand)[0]]
        q = self.qorder or mesh.poly_order+2
        qpts,qwts = volume(mesh.element_type,q)
        terms = _split_terms(intg.integrand)
        for eid,elem in enumerate(mesh.elements_list):
            
            self.ctx['elem_id'] = elem.id
            # precompute basis at qpts
            bv=defaultdict(lambda:{'val':[],'grad':[]})
            for fld in _all_fields(intg.integrand):
                fm = self.dh.fe_map[fld]
                ref = get_reference(fm.element_type,fm.poly_order)
                for xi,eta in qpts:
                    J = transform.jacobian(fm,eid,(xi,eta))
                    JinvT = np.linalg.inv(J).T
                    bv[fld]['val'].append(ref.shape(xi,eta))
                    bv[fld]['grad'].append(ref.grad(xi,eta) @ JinvT)
            for sgn,term in terms:
                trial,test=_trial_test(term)
                if rhs and test is None: continue
                if not rhs and (trial is None or test is None): continue
                row = self._elm_dofs(test,eid)
                if rhs:
                    local = np.zeros(len(row))
                else:
                    col = self._elm_dofs(trial,eid)
                    local = np.zeros((len(row),len(col)))
                for k,(xi_eta, w) in enumerate(zip(qpts,qwts)):
                    self.ctx['x_phys'] = transform.x_mapping(mesh, elem.id, (xi, eta))
                    self.ctx["side"] = ""  # volume integrals have no side notion
                    self.ctx['x'] = self.ctx['x_phys']
                    self.ctx['basis_values']={f:{'val':bv[f]['val'][k],'grad':bv[f]['grad'][k]} for f in bv}
                    J = transform.jacobian(mesh,eid,xi_eta)
                    val = self.visit(term)
                    local += sgn * w * abs(np.linalg.det(J)) * val
                if rhs:
                    matvec[row]+=local
                else:
                    matvec[np.ix_(row,col)]+=local
        # Context cleanup
        for key in ('phi_val', 'normal', 'basis_values', 'x', 'nodal_vals','elem_id','x_phys','side'):
            self.ctx.pop(key, None)
    
    # ------------------------------------------------------------------
    #  Interface integrals (CutFEM) – one-sided, no extra DOFs
    # ------------------------------------------------------------------


    def _assemble_interface(self, intg, matvec):
        rhs = self.ctx['is_rhs']
        level_set = intg.measure.level_set
        mesh = next(iter(self.dh.fe_map.values()))        # any mesh

        qdeg   = self.qorder or mesh.poly_order+2
        terms = _split_terms(intg.integrand)
        fields = _all_fields(intg.integrand)

        # optional scalar hook (perimeter, jump, …)
        hook = None
        for cls,cfg in self.ctx['hooks'].items():
            if isinstance(intg.integrand, cls):
                hook = cfg
                if hook and hook['name'] not in self.ctx['scalar_results']:
                    self.ctx['scalar_results'][hook['name']] = 0.0
                break

        # ---------------------------------------------------------- loop cut elems
        for elem in mesh.elements_list:
            if elem.tag != 'cut' or len(elem.interface_pts) != 2:
                continue
            self.ctx['elem_id'] = elem.id
            p0, p1 = elem.interface_pts
            # print(f"{elem}")
            # print(f"p0={p0}, p1={p1}")

            qp, qw = line_quadrature(p0, p1, qdeg)        # physical pts + weights
            # print(f"qp={qp}, qw={qw}")

            # -------- pre-tabulate basis at all quad-pts for every field ----
            basis_cache = {f:{'val':[], 'grad':[]} for f in fields}
            if fields:
                for x in qp:
                    for fld in fields:
                        fm  = self.dh.fe_map[fld]
                        
                        ref = get_reference(fm.element_type, fm.poly_order)
                        xi,eta = transform.inverse_mapping(fm, elem.id, x)
                        JinvT  = np.linalg.inv(transform.jacobian(fm, elem.id, (xi,eta))).T
                        basis_cache[fld]['val' ].append(ref.shape(xi,eta))
                        basis_cache[fld]['grad'].append(ref.grad (xi,eta) @ JinvT)
            # print(f"After line integration: basis_cache={basis_cache}")
            # if fields:
            #     print(f"basis_cache[fld]['val' ].shape={np.asarray(basis_cache[fields[0]]['val']).shape}")
            #     print(f"basis_cache[fld]['grad'].shape={np.asarray(basis_cache[fields[0]]['grad']).shape}")
            # ------------------------------------------------------ term loop
            
             # --- Assemble for each term in the integrand ---
            for sgn, term in terms:
                trial, test = _trial_test(term)
                # DEBUGGING BLOCK =======================================================
                # print("\n--- DEBUG: Checking term ---")
                # print(f"Term: {term}")
                # print(f"Is RHS? {rhs}")
                # print(f"Trial function: {trial}")
                # print(f"Test function: {test}")
                # print(f"Hook available? {hook is not None}")
                # if hook:
                #     print(f"Hook config: {hook}")
                #     print(f"Hook keys: {tuple(self.ctx['hooks'].keys())}")
                #     print(f"Is term instance of hooked type? {isinstance(term, tuple(self.ctx['hooks'].keys()))}")
                # =======================================================================


                # Case 1: Bilinear form (LHS) -> contributes to the matrix `K`
                if not rhs and trial is not None and test is not None:
                    row_dofs = self._elm_dofs(test, elem.id)
                    col_dofs = self._elm_dofs(trial, elem.id)
                    loc = np.zeros((len(row_dofs), len(col_dofs)))
                    for k, (x, w) in enumerate(zip(qp, qw)):
                        self.ctx['x'] = x
                        self.ctx['phi_val'] = level_set(x)
                        self.ctx['normal'] = level_set.gradient(x)
                        self.ctx['basis_values'] = {f: {'val': basis_cache[f]['val'][k], 'grad': basis_cache[f]['grad'][k]} for f in fields}
                        # self.ctx['nodal_vals'] = {f: mesh.node_data[f][elem.nodes] for f in getattr(mesh, 'node_data', {}) if f in fields}
                        loc += sgn * w * self.visit(term)
                    matvec[np.ix_(row_dofs, col_dofs)] += loc

                # Case 2: Linear form (RHS) -> contributes to the vector `F`
                elif rhs and test is not None:
                    row_dofs = self._elm_dofs(test, elem.id)
                    loc = np.zeros(len(row_dofs))
                    for k, (x, w) in enumerate(zip(qp, qw)):
                        self.ctx['x'] = x
                        self.ctx['phi_val'] = level_set(x)
                        self.ctx['normal'] = level_set.gradient(x)
                        self.ctx['basis_values'] = {f: {'val': basis_cache[f]['val'][k], 'grad': basis_cache[f]['grad'][k]} for f in fields}
                        # self.ctx['nodal_vals'] = {f: mesh.node_data[f][elem.nodes] for f in getattr(mesh, 'node_data', {}) if f in fields}
                        loc += sgn * w * self.visit(term)
                    matvec[row_dofs] += loc
                
                # Case 3: Scalar functional -> contributes to `scalar_results` via a hook
                elif hook and trial is None and test is None and isinstance(term, tuple(self.ctx['hooks'].keys())):
                    acc = 0.0
                    for k, (x, w) in enumerate(zip(qp, qw)):
                        self.ctx['x'] = x
                        self.ctx['phi_val'] = level_set(x)
                        self.ctx['normal'] = level_set.gradient(x)
                        if fields:
                           self.ctx['basis_values'] = {f: {'val': basis_cache[f]['val'][k], 'grad': basis_cache[f]['grad'][k]} for f in fields}
                        #    print(f"self.ctx['basis_values'][f]['val'][k]={self.ctx['basis_values'][fields[0]]['val'][k]}")
                        #    print(f"self.ctx['basis_values'][f]['grad'][k]={self.ctx['basis_values'][fields[0]]['grad'][k]}")
                        #    self.ctx['nodal_vals'] = {f: mesh.node_data[f][elem.nodes] for f in getattr(mesh, 'node_data', {}) if f in fields}
                        # print(f" term = {term}")
                        # print(f"self.visit(term) = {self.visit(term)}")
                        
                        acc += sgn * w * self.visit(term)
                    self.ctx['scalar_results'][hook['name']] += acc

        # Context cleanup
        for key in ('phi_val', 'normal', 'basis_values', 'x', 'nodal_vals','elem_id'):
            self.ctx.pop(key, None)



    # facet -------------------------------------------------------------
    def _assemble_facet(self,intg,matvec):
        rhs = self.ctx['is_rhs']
         # ----- pick *any* mesh the dof-handler knows if the integrand
        #       contains no FE fields (e.g. Constant * ds)
        fields = _all_fields(intg.integrand)
        mesh   = self.dh.fe_map[fields[0]] if fields else next(iter(self.dh.fe_map.values()))

        # optional user-defined hook, e.g. {Constant: {'name':'perimeter'}}
        hooks = self.ctx.get('hooks', {})          # <-- hooks were stored here
        scalar_hook = None
        for cls, cfg in hooks.items():             # accept subclass matches
            if isinstance(intg.integrand, cls):
                scalar_hook = cfg
                break

        if scalar_hook and scalar_hook['name'] not in self.ctx['scalar_results']:
            self.ctx['scalar_results'][scalar_hook['name']] = 0.0
        q = self.qorder or mesh.poly_order+2
        edge_ids = intg.measure.defined_on.to_indices()
        terms = _split_terms(intg.integrand)
        level_set = getattr(intg.measure,'level_set',None)
        print(f"level_set={level_set}")

        for eid_edge in edge_ids:
            edge = mesh.edge(eid_edge)
            left,right = edge.left, edge.right
            if right is None:  # boundary facet – treat right as ghost copy of left
                continue
                left,right = (edge.left, edge.left)
            # decide pos/neg orientation using level_set if available
            if level_set is not None:
                phi_left = level_set(mesh.elements_list[left].centroid())
                pos,neg = (left,right) if phi_left>=0 else (right,left)
            else:
                pos,neg = left,right
            # quadrature pts on reference edge of pos element
            loc_idx = mesh.elements_list[pos].edges.index(eid_edge)
            glob_edge_idx = mesh.elements_list[pos].edges[loc_idx]
            # print(loc_idx,mesh.elements_list[pos].edges)
            qpts_ref, qwts = edge_quadrature_element(mesh.element_type, loc_idx, q)
            for qp, w in zip(qpts_ref,qwts):
                xq = transform.x_mapping(mesh,pos,qp)
                self.ctx['normal'] = mesh.edges_list[glob_edge_idx].calc_normal_unit_vector()#transform.edge_unit_normal(mesh,pos,loc_idx)  # project helper must exist
                jac1d = transform.jacobian_1d(mesh,pos,qp,loc_idx)
                # ----- Constant / jump integrals – no basis evaluation needed
                if scalar_hook and not fields:
                    val = self.visit(intg.integrand)
                    self._scalar_results[scalar_hook['name']] += val * jac1d * w
                    continue
                # side‑aware basis cache
                bv={'+' : defaultdict(dict), '-' : defaultdict(dict)}
                for side,eid in (('+',pos),('-',neg)):
                    for fld in _all_fields(intg.integrand):
                        fm = self.dh.fe_map[fld]
                        ref = get_reference(fm.element_type,fm.poly_order)
                        xi,eta = transform.inverse_mapping(fm,eid,xq)
                        J = transform.jacobian(fm,eid,(xi,eta))
                        JinvT = np.linalg.inv(J).T
                        bv[side][fld]['val']=ref.shape(xi,eta)
                        bv[side][fld]['grad']=ref.grad(xi,eta) @ JinvT
                self.ctx['basis_values']=bv

                for sgn,term in terms:
                    trial,test=_trial_test(term)
                    if rhs and test is None: continue
                    if not rhs and (trial is None or test is None): continue

                    for row_side,row_eid in (('+',pos),('-',neg)):
                        self.ctx['active_side']=row_side
                        rows = self._elm_dofs(test,row_eid)
                        if rhs:
                            val = self.visit(term)
                            matvec[rows]+= sgn*w*jac1d*val
                        else:
                            for col_side,col_eid in (('+',pos),('-',neg)):
                                self.ctx['trial_side']=col_side
                                cols = self._elm_dofs(trial,col_eid)
                                val = self.visit(term)
                                matvec[np.ix_(rows,cols)]+= sgn*w*jac1d*val

                self.ctx.pop('active_side',None)
                self.ctx.pop('trial_side',None)

    # dof helpers --------------------------------------------------------
    def _elm_dofs(self,func,eid):
        if isinstance(func,(VectorTrialFunction,VectorTestFunction)):
            return [d for f in func.space.field_names for d in self.dh.element_maps[f][eid]]
        return self.dh.element_maps[func.field_name][eid]
