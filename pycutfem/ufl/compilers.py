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
import logging,functools

# UFL‑like helpers ----------------------------------------------------------
from pycutfem.fem.operators import grad
from pycutfem.ufl.expressions import (
    Constant, TrialFunction, TestFunction,
    VectorTrialFunction, VectorTestFunction,
    Grad, DivOperation, Inner, Dot, Sum, Sub, Prod,
    Jump, FacetNormal, Pos, Neg, Function, VectorFunction, Div
)
from pycutfem.ufl.forms import Equation

# Project helpers -----------------------------------------------------------
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.reference import get_reference
from pycutfem.fem import transform
from pycutfem.integration import volume
from pycutfem.integration.quadrature import  line_quadrature
from pycutfem.integration.quadrature import edge as edge_quadrature_element
import sympy
from pycutfem.fem.mixedelement import MixedElement

logger = logging.getLogger(__name__)


_INTERFACE_TOL = 1.0e-12



# ---------------------------------------------------------------------------
#  Light shape‑meta visitor (helps flip test/trial in ∇· terms)
# ---------------------------------------------------------------------------
class _Shape:
    __slots__ = ("dim", "kind")
    def __init__(self, dim: int, kind: str):
        self.dim = dim      # tensor order (scalar:0, vector:1, grad:2,…)
        self.kind = kind    # 'test' | 'trial' | 'none'



# In ufl/compilers.py

class _ShapeVisitor:
    def __init__(self):
        self._memo = {}
    def visit(self, n):
        if n in self._memo: return self._memo[n]
        
        # This dispatch needs to be specific
        meth_name = f"visit_{type(n).__name__}"
        meth = getattr(self, meth_name, self.generic_visit)
        
        res = meth(n)
        self._memo[n] = res
        return res

    def generic_visit(self, n):
        # Fallback for base classes if a specific visitor doesn't exist
        if isinstance(n, TestFunction): return self.visit_TestFunction(n)
        if isinstance(n, TrialFunction): return self.visit_TrialFunction(n)
        if isinstance(n, Function): return self.visit_Function(n)
        raise TypeError(f"No shape rule for {type(n)}")

    # --- Corrected Visitors for all types ---
    def visit_Constant(self, n):      return _Shape(n.dim, 'none')
    def visit_Analytic(self, n):      return _Shape(0, 'none')
    
    # --- Specific handlers for each type ---
    def visit_TestFunction(self, n):      return _Shape(0, 'test')
    def visit_TrialFunction(self, n):     return _Shape(0, 'trial')
    def visit_Function(self, n):          return _Shape(0, 'none')
    
    def visit_VectorTestFunction(self, n):  return _Shape(1, 'test')
    def visit_VectorTrialFunction(self, n): return _Shape(1, 'trial')
    def visit_VectorFunction(self, n):      return _Shape(1, 'none')

    # --- Visitors for Operators ---
    def visit_Sum(self, n):  return self.visit(n.a)
    def visit_Sub(self, n):  return self.visit(n.a)
    def visit_Prod(self,n):
        sa, sb = self.visit(n.a), self.visit(n.b)
        dim = sa.dim + sb.dim
        kind = sa.kind if sa.kind != 'none' else sb.kind
        return _Shape(dim, kind)

    def visit_Grad(self, n): 
        s = self.visit(n.operand)
        return _Shape(s.dim + 1, s.kind)

    def visit_DivOperation(self,n):
        return _Shape(0, self.visit(n.operand).kind)

    def visit_Inner(self,n): 
        sa = self.visit(n.a); sb = self.visit(n.b)
        kind = sa.kind if sa.kind != 'none' else sb.kind
        return _Shape(max(0, sa.dim - 1), kind)
    
    def visit_Dot(self,n):
        sa = self.visit(n.a); sb = self.visit(n.b)
        kind = sa.kind if sa.kind != 'none' else sb.kind
        # Dot product reduces total tensor rank by 2, or dimension by 1 for vectors
        dim = max(0, sa.dim + sb.dim - 2) if (sa.dim > 0 and sb.dim > 0) else sa.dim + sb.dim
        return _Shape(dim, kind)

    def visit_Div(self, n): return self.visit(n.a)
    def visit_Pos(self,n):   return self.visit(n.operand)
    def visit_Neg(self,n):   return self.visit(n.operand)
    def visit_Jump(self,n):  return self.visit(n.u_pos)
    def visit_FacetNormal(self,n): return _Shape(1,'none')
    def visit_ElementWiseConstant(self, n):
        return _Shape(0, 'none')

# ---------------------------------------------------------------------------
#  Helper utilities
# ---------------------------------------------------------------------------

def _split_terms(expr, coef=1.0):
    """
    Flatten a ± tree **and** pull purely-scalar factors in front.
    Returns a list  [(scalar, term), …]
    where each *term* still contains the symbolic FE factors.
    """
    from pycutfem.ufl.expressions import Sum, Sub, Prod, Constant
    # 1. walk through +/–
    if isinstance(expr, Sum):
        return _split_terms(expr.a, coef) + _split_terms(expr.b, coef)
    if isinstance(expr, Sub):
        return _split_terms(expr.a, coef) + _split_terms(expr.b, -coef)

    # 2. pull scalar constants out of a product
    if isinstance(expr, Prod):
        # “Constant ⋅ something”  or  “something ⋅ Constant”
        if isinstance(expr.a, Constant) and expr.a.dim == 0:
            return _split_terms(expr.b, coef * expr.a.value)
        if isinstance(expr.b, Constant) and expr.b.dim == 0:
            return _split_terms(expr.a, coef * expr.b.value)

    # 3. everything else stays as-is
    return [(coef, expr)]




def _all_fields(expr):
    fields=set()
    def walk(n):
        if hasattr(n,'field_name'):
            fields.add(n.field_name)
        if hasattr(n,'space'):
            fields.update(n.space.field_names)
        if isinstance(n, VectorFunction):
            fields.update(n.field_names)

        # Recurse through the expression tree
        for attr in ('operand', 'a', 'b', 'u_pos', 'u_neg', 'components','f'):
            if hasattr(n, attr):
                m = getattr(n, attr)
                if isinstance(m, (list, tuple)):
                    for x in m: walk(x)
                elif m is not None:
                    walk(m)
    walk(expr)
    return list(fields)


def _trial_test(expr):
    trial = expr.find_first(lambda n:isinstance(n,(TrialFunction,VectorTrialFunction)))
    test  = expr.find_first(lambda n:isinstance(n,(TestFunction ,VectorTestFunction )))
    return trial, test

def _is_component(obj) -> bool:
    """True if *obj* is a scalar component extracted from a vector."""
    return getattr(obj, "_parent_vector", None) is not None

def _n_components(obj) -> int:
    """How many components in the vector that *obj* belongs to?"""
    if _is_component(obj):
        return len(obj._parent_vector.components)
    if hasattr(obj, "components"):
        return len(obj.components)
    return 1                # scalar stand-alone


def _mask(n_dofs: int, slc: slice) -> np.ndarray:
    """Vector of length *n_dofs* with ones on *slc* and zeros elsewhere."""
    m = np.zeros(n_dofs)
    m[slc] = 1.0
    return m

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
        if self.dh.mixed_element is None:
            raise RuntimeError("This compiler requires a MixedElement‑backed DofHandler.")
        self.me: MixedElement = self.dh.mixed_element

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
        
    # ---------------- Derivative single ------------------------------------------
    def visit_Derivative(self, n): # needs to be corrected
        """
        Evaluates a single component of a gradient.
        e.g., Derivative(f, 0) corresponds to ∂f/∂x.

        This method is robust enough to handle both 2D arrays of basis
        gradients (from Trial/TestFunctions) and 1D arrays of interpolated
        gradient values (from data-carrying Functions).
        """
        # First, evaluate the full gradient of the function.

        base=self.visit(Grad(n.f)); ax=n.component_index
        return base[ax] if base.ndim==1 else base[:,ax]

    # scalars
    def visit_Constant(self, n):
        """Return an array whose length matches the target expression.

        * scalar → one full-length (n_dofs_local) vector of that value
        * iterable of length == n_components → *concatenate* one padded
        vector **per component**, mirroring VectorTest/Trial layout
        """
        if np.isscalar(n.value):
            return np.full(self.me.n_dofs_local, float(n.value))

        parts = []
        for comp_val, fld in zip(n.value, self.me.field_names):
            vec = np.zeros(self.me.n_dofs_local)
            vec[self.me.slice(fld)] = comp_val
            parts.append(vec)
        return np.concatenate(parts)           # length = n_components · n_dofs_local
    
    def visit_Pos(self, n):
        """
        Evaluates the operand if on the positive side (phi>=0), otherwise returns
        a zero of the same shape as the operand.
        """
        # print("visit Pos zeroing: ctx=", self.ctx)
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
        # print("visit Neg zeroing: ctx=", self.ctx)
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

    # ------------------------------------------------------------------
    #  Core basis cache (single side – volume) -------------------------
    # ------------------------------------------------------------------
    def _make_basis_cache(self, xi: float, eta: float, JinvT: np.ndarray):
        """Full‑length (n_dofs_local) basis & grad for every field."""
        φ_full = self.me.basis(xi, eta)            # (ndofs,)
        g_full = self.me.grad(xi, eta) @ JinvT      # (ndofs,2)

        cache: Dict[str, Dict[str, np.ndarray]] = {}
        for fld in self.me.field_names:
            sl = self.me.slice(fld)
            val  = np.zeros_like(φ_full);  val[sl]  = φ_full[sl]
            grad = np.zeros_like(g_full); grad[sl] = g_full[sl]
            cache[fld] = { 'val': val, 'grad': grad }
        return cache, g_full

    # ------------------------------------------------------------------
    #  Role‑aware helpers ----------------------------------------------
    # ------------------------------------------------------------------
    def _basis_val(self, field: str, role: str):
        bv = self.ctx['basis_values']
        if '+' in bv:  # interface dict  { '+': …, '-': … }
            side = self.ctx['active_side'] if role == 'test' else self.ctx.get('trial_side', '+')
            return bv[side][field]['val']
        return bv[field]['val']

    def _basis_grad(self, field: str, role: str):
        bg = self.ctx['basis_values']
        if '+' in bg:
            side = self.ctx['active_side'] if role == 'test' else self.ctx.get('trial_side', '+')
            return bg[side][field]['grad']
        return bg[field]['grad']

    # finite element functions ------------------------------------------

    def visit_TestFunction(self, n):
        return self._basis_val(n.field_name, 'test')

    def visit_TrialFunction(self, n):
        return self._basis_val(n.field_name, 'trial')

    def visit_VectorTestFunction(self, n):
        return np.concatenate([self._basis_val(f, 'test') for f in n.field_names])

    def visit_VectorTrialFunction(self, n):
        return np.concatenate([self._basis_val(f, 'trial') for f in n.field_names])


    # ------------------------------------------------------------------
    #  Data visitors ----------------------------------------------------
    # ------------------------------------------------------------------
    def _local_dofs(self): return self.dh.get_elemental_dofs(self.ctx['element_id'])
    
    def visit_Function(self, n):
        u_loc = n.get_nodal_values(self._local_dofs())
        ϕ = self._basis_val(n.field_name,'test')  # role irrelevant for value
        return ϕ @ u_loc
    
    def visit_VectorFunction(self, n):
        u_loc = n.get_nodal_values(self._local_dofs())
        out=[]
        for f in n.field_names:
            ϕ=self._basis_val(f,'test'); out.append(ϕ @ u_loc[self.me.slice(f)])
        return np.array(out)

    # differential ops ---------------------------------------------------
    def visit_Grad(self, n):
        op = n.operand
        if isinstance(op, (TestFunction, TrialFunction)):
            role = 'test' if isinstance(op, TestFunction) else 'trial'
            return self._basis_grad(op.field_name, role)
        if isinstance(op, (VectorTestFunction, VectorTrialFunction)):
            return self.ctx['grad_full']  # 22×2 once, no duplication
        if isinstance(op, Function):
            gradϕ = self._basis_grad(op.field_name, 'data')
            return gradϕ.T @ op.get_nodal_values(self._local_dofs())
        if isinstance(op, VectorFunction):
            G = np.zeros((len(op.field_names), 2))
            u_loc = op.get_nodal_values(self._local_dofs())
            for i, f in enumerate(op.field_names):
                G[i, :] = self._basis_grad(f, 'test').T @ u_loc[self.me.slice(f)]
            return G
        raise NotImplementedError(type(op))



    def visit_DivOperation(self,n):
        grads = self.visit(Grad(n.operand))
        return grads[:,0]+grads[:,1] if grads.ndim==2 else grads[0]+grads[1]
    
    def visit_Analytic(self, node): return node.eval(self.ctx['x_phys'])

    # algebra -----------------------------------------------------------
    
    
    def visit_Div(self, n):
        """Handles scalar, vector, or tensor division by a scalar."""
        numerator = self.visit(n.a)
        denominator = self.visit(n.b)

        # NumPy correctly handles element-wise division of an array by a scalar.
        return numerator / denominator

    def _maybe_outer(self,a,b):
        if not self.ctx['is_rhs'] and a.ndim==b.ndim==1 and a.size==self.me.n_dofs_local:
            return np.outer(a,b)
        return None
    
    def visit_Inner(self, n):
        A=self.visit(n.a); B=self.visit(n.b)
        if not self.ctx['is_rhs'] and A.ndim==B.ndim==2 and A.shape[0]==self.me.n_dofs_local:
            return np.outer(A[:,0],B[:,0])+np.outer(A[:,1],B[:,1])
        return np.sum(A*B)


    # dot product ---------------------------------------------------
    #----------------------------------------------------------------------------------------------------------
    def visit_Prod(self, n):
        a=self.visit(n.a); b=self.visit(n.b); out=self._maybe_outer(a,b); return out if out is not None else a*b
    

   
   

    def visit_Dot(self, n):
        a=self.visit(n.a); b=self.visit(n.b); out=self._maybe_outer(a,b)
        return out if out is not None else a @ b
           

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

    def _assemble_form(self, form, matvec):
        """
        Dispatch each integral in *form* to the proper low-level routine.
        Works whether *form* is an Integral or a Form holding many integrals.
        """
        from pycutfem.ufl.measures import Integral

        # 1. Normalise *form* → iterable of Integrals
        if isinstance(form, Integral):
            integrals = [form]
        elif hasattr(form, "integrals"):
            integrals = form.integrals
        else:                           # e.g. Constant(0.0) on the dummy LHS
            return

        # 2. Route every integral to its dedicated assembler
        for intg in integrals:
            if not isinstance(intg, Integral):        # skip Stray constants, etc.
                continue

            kind = intg.measure.domain_type
            if   kind == "volume":          self._assemble_volume(intg, matvec)
            elif kind == "interior_facet":  self._assemble_facet(intg,  matvec)
            elif kind == "interface":       self._assemble_interface(intg, matvec)

      


    # volume ------------------------------------------------------------
    # In ufl/compilers.py

    def _assemble_volume(self, integral, matvec):
        mesh = self.me.mesh; q = self.qorder or mesh.poly_order + 2
        q_pts, q_wts = volume(mesh.element_type, q)
        for el in mesh.elements_list:
            dofs = self.dh.get_elemental_dofs(el.id)
            local = (np.zeros(self.me.n_dofs_local) if self.ctx['is_rhs']
                     else np.zeros((self.me.n_dofs_local, self.me.n_dofs_local)))
            for (xi, eta), w in zip(q_pts, q_wts):
                J = transform.jacobian(mesh, el.id, (xi, eta))
                detJ = abs(np.linalg.det(J)); JinvT = np.linalg.inv(J).T
                cache, g_full = self._make_basis_cache(xi, eta, JinvT)
                self.ctx.update({
                    'element_id':   el.id,
                    'basis_values': cache,
                    'grad_full':    g_full,
                    'x_phys': transform.x_mapping(mesh, el.id, (xi,eta)),
                    'x':      transform.x_mapping(mesh, el.id, (xi,eta)),
                    'side': ""
                })
                local += w * detJ * self.visit(integral.integrand)
            if self.ctx['is_rhs']:
                np.add.at(matvec, dofs, local)
            else:
                r, c = np.meshgrid(dofs, dofs, indexing='ij'); matvec[r, c] += local            
        # Context cleanup (good practice)
        for key in ('phi_val', 'normal', 'basis_values', 'x', 'nodal_vals', 'elem_id', 'x_phys', 'side'):
            self.ctx.pop(key, None)
    
    
    # ------------------------------------------------------------------
    #  Interface integrals (CutFEM) – one-sided, no extra DOFs
    # ------------------------------------------------------------------
    # In class FormCompiler:

    def _assemble_interface(self, intg, matvec):
        """
        Assembles integrals over non-conforming interfaces defined by a level set.

        This robust implementation evaluates the entire integrand at once, correctly
        handling scalar and vector-valued expressions for bilinear forms, linear
        forms, and hooked functionals without relying on term-splitting.
        """
        # Use a logger for clear, hierarchical debugging output.
        # (Assumes you have `import logging` at the top of the file)
        log = logging.getLogger(__name__)
        log.debug(f"Assembling interface integral: {intg}")

        # --- 1. Initial Setup ---
        rhs = self.ctx['is_rhs']
        level_set = intg.measure.level_set
        if level_set is None:
            raise ValueError("dInterface measure requires a level_set.")
            
        mesh = next(iter(self.dh.fe_map.values())) # Any mesh known to the DoF handler
        qdeg = self.qorder or mesh.poly_order + 2
        fields = _all_fields(intg.integrand)
        
        # Check for a user-defined hook for functionals.
        hook = self.ctx['hooks'].get(type(intg.integrand))
        if hook and hook.get('name') not in self.ctx.get('scalar_results', {}):
            # Initialize the result bucket; it might hold a scalar or a vector.
            self.ctx.setdefault('scalar_results', {})[hook['name']] = 0.0

        try:
            # --- 2. Loop Over All Cut Elements in the Mesh ---
            for elem in mesh.elements_list:
                if elem.tag != 'cut' or len(elem.interface_pts) != 2:
                    continue
                
                self.ctx['elem_id'] = elem.id
                log.debug(f"  Processing cut element: {elem.id}")

                # --- 3. Quadrature Rule on the Physical Interface Segment ---
                p0, p1 = elem.interface_pts
                qp, qw = line_quadrature(p0, p1, qdeg)

                # --- 4. Pre-compute Basis Functions for this Element ---
                # This is a key optimization. We evaluate all basis functions for all
                # quadrature points on this element's interface segment at once.
                basis_cache = {}
                if fields:
                    for fld in fields:
                        fm = self.dh.fe_map[fld]
                        ref = get_reference(fm.element_type, fm.poly_order)
                        vals, grads = [], []
                        for x_q in qp:
                            xi, eta = transform.inverse_mapping(fm, elem.id, x_q)
                            J = transform.jacobian(fm, elem.id, (xi, eta))
                            # The mathematically correct transformation for row-vector gradients
                            Jinv = np.linalg.inv(J)
                            vals.append(ref.shape(xi, eta))
                            grads.append(ref.grad(xi, eta) @ Jinv)
                        # Store as efficient NumPy arrays
                        basis_cache[fld] = {'val': np.asarray(vals), 'grad': np.asarray(grads)}
                
                # --- 5. Determine Assembly Path (Bilinear, Linear, or Functional) ---
                trial, test = _trial_test(intg.integrand)

                # --- PATH A: Bilinear Form (e.g., a(u,v), contributes to the matrix) ---
                if not rhs and trial and test:
                    row_dofs = self._elm_dofs(test, elem.id)
                    col_dofs = self._elm_dofs(trial, elem.id)
                    loc = np.zeros((len(row_dofs), len(col_dofs)))
                    
                    for k, (x, w) in enumerate(zip(qp, qw)):
                        # Set context for the current quadrature point
                        self.ctx['x'], self.ctx['phi_val'] = x, level_set(x)
                        self.ctx['normal'] = level_set.gradient(x)
                        if fields:
                            self.ctx['basis_values'] = {f: {'val': basis_cache[f]['val'][k], 'grad': basis_cache[f]['grad'][k]} for f in fields}
                        
                        # Evaluate the ENTIRE integrand at once
                        integrand_val = self.visit(intg.integrand)
                        loc += w * integrand_val
                    
                    matvec[np.ix_(row_dofs, col_dofs)] += loc
                    log.debug(f"    Assembled {loc.shape} local matrix for element {elem.id}")

                # --- PATH B: Linear Form (e.g., L(v)) or Functional (e.g., dot(jump,n)) ---
                else:
                    # Initialize accumulator correctly based on the result shape
                    acc = None
                    
                    # Main Quadrature Loop
                    for k, (x, w) in enumerate(zip(qp, qw)):
                        self.ctx['x'], self.ctx['phi_val'] = x, level_set(x)
                        self.ctx['normal'] = level_set.gradient(x)
                        if fields:
                            self.ctx['basis_values'] = {f: {'val': basis_cache[f]['val'][k], 'grad': basis_cache[f]['grad'][k]} for f in fields}
                        
                        integrand_val = self.visit(intg.integrand)
                        
                        # Initialize accumulator on the first pass
                        if acc is None:
                            acc = w * np.asarray(integrand_val)
                        else:
                            acc += w * integrand_val

                    # Add the accumulated result to the correct global object
                    if acc is not None:
                        if rhs: # It's a linear form (RHS)
                            row_dofs = self._elm_dofs(test, elem.id)
                            matvec[row_dofs] += acc
                            log.debug(f"    Assembled {acc.shape} local vector for element {elem.id}")
                        elif hook: # It's a hooked functional
                            self.ctx['scalar_results'][hook['name']] += acc
                            log.debug(f"    Accumulated functional '{hook['name']}' for element {elem.id}")

        finally:
            # --- 6. Final Context Cleanup ---
            # Use a `finally` block to guarantee cleanup happens even if an error occurs.
            for key in ('phi_val', 'normal', 'basis_values', 'x', 'elem_id'):
                self.ctx.pop(key, None)
            log.debug("Interface assembly finished. Context cleaned.")




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
    def _elm_dofs(self, func, eid):
        """Final, robust method to get all global element DOFs for any function type."""
        if func is None:
            return []
        
        # Check for any class with a `.space` attribute (VectorTrial/TestFunction)
        if hasattr(func, 'space'):
            return np.concatenate([self.dh.element_maps[f][eid] for f in func.space.field_names])

        # Check for any class with a `.field_names` attribute (VectorFunction)
        if hasattr(func, 'field_names'):
            return np.concatenate([self.dh.element_maps[f][eid] for f in func.field_names])

        # Fallback for scalar types (Function, TrialFunction, TestFunction)
        if hasattr(func, 'field_name'):
            return self.dh.element_maps[func.field_name][eid]
            
        raise TypeError(f"Cannot get element DOFs for object of type {type(func)}")
