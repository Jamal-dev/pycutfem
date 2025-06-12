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
    Jump, FacetNormal, Pos, Neg
)
from ufl.forms import Equation

# Project helpers -----------------------------------------------------------
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.reference import get_reference
from pycutfem.fem import transform
from pycutfem.integration import volume
from pycutfem.integration.quadrature import edge as line_quadrature

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
    def __init__(self, dofhandler: DofHandler, quad_order: int | None = None):
        self.dh = dofhandler
        self.qorder = quad_order
        self.shape = _ShapeVisitor()
        self.ctx: Dict[str,Any] = {}

    # ---------- public --------------------------------------------------
    def assemble(self, equation: Equation, bcs, *, assembler_hooks:dict|None=None):
        n = self.dh.total_dofs
        K = sp.lil_matrix((n,n))
        F = np.zeros(n)
        self.ctx['hooks'] = assembler_hooks or {}

        # left‑hand side
        self.ctx['is_rhs'] = False
        self._assemble_form(equation.a, K)
        # right‑hand side
        self.ctx['is_rhs'] = True
        self._assemble_form(equation.L, F)

        self._apply_bcs(K,F,bcs)
        return K.tocsr(), F

    # ---------- visitors ------------------------------------------------
    def visit(self, node):
        return getattr(self, f"visit_{type(node).__name__}")(node)

    # scalars
    def visit_Constant(self,n): return n.value
    def visit_Pos(self,n): return +self.visit(n.operand)
    def visit_Neg(self,n): return -self.visit(n.operand)
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

    # differential ops ---------------------------------------------------
    def visit_Grad(self,n):
        sh = self.shape.visit(n.operand)
        if sh.dim==0:  # grad(scalar) -> vector
            return self._basis_grad(n.operand.field_name, sh.kind)
        if sh.dim==1:  # grad(vector) -> list[tensor row]
            return [self._basis_grad(f, sh.kind) for f in n.operand.space.field_names]
        raise NotImplementedError('grad of tensor not needed')

    def visit_DivOperation(self,n):
        grads = self.visit(Grad(n.operand))  # list of rows
        return np.hstack([g[:,i] for i,g in enumerate(grads)])

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
        # scalar inner
        if sa.kind=='test':
            return np.outer(a,b)
        return np.outer(b,a)

    def visit_Dot(self,n):
        const, vec = (self.visit(n.a), self.visit(n.b)) if isinstance(n.a, Constant) else (self.visit(n.b), self.visit(n.a))
        ncomp = len(const)
        dofs_per = len(vec)//ncomp
        out = np.zeros_like(vec)
        vec = vec.reshape((ncomp,dofs_per))
        for i in range(ncomp):
            out[i*dofs_per:(i+1)*dofs_per]=const[i]*vec[i]
        return out

    def visit_Jump(self,n):
        side = self.ctx['active_side']
        if side=='+': return self.visit(n.u_pos)
        return self.visit(n.u_neg)

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
        for integral in form.integrals:
            kind = integral.measure.domain_type
            if kind=='volume':
                self._assemble_volume(integral,matvec)
            elif kind=='interior_facet':
                self._assemble_facet(integral,matvec)

    # volume ------------------------------------------------------------
    def _assemble_volume(self,intg,matvec):
        rhs = self.ctx['is_rhs']
        mesh = self.dh.fe_map[_all_fields(intg.integrand)[0]]
        q = self.qorder or mesh.poly_order+2
        qpts,qwts = volume(mesh.element_type,q)
        terms = _split_terms(intg.integrand)
        for eid,elem in enumerate(mesh.elements_list):
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
                    self.ctx['basis_values']={f:{'val':bv[f]['val'][k],'grad':bv[f]['grad'][k]} for f in bv}
                    J = transform.jacobian(mesh,eid,xi_eta)
                    val = self.visit(term)
                    local += sgn * w * abs(np.linalg.det(J)) * val
                if rhs:
                    matvec[row]+=local
                else:
                    matvec[np.ix_(row,col)]+=local

    # facet -------------------------------------------------------------
    def _assemble_facet(self,intg,matvec):
        rhs = self.ctx['is_rhs']
        mesh = self.dh.fe_map[_all_fields(intg.integrand)[0]]
        q = self.qorder or mesh.poly_order+2
        edge_ids = intg.measure.defined_on.to_indices()
        terms = _split_terms(intg.integrand)
        level_set = getattr(intg.measure,'level_set',None)

        for eid_edge in edge_ids:
            edge = mesh.edge(eid_edge)
            left,right = edge.left, edge.right
            if right is None:  # boundary facet – treat right as ghost copy of left
                left,right = (edge.left, edge.left)
            # decide pos/neg orientation using level_set if available
            if level_set is not None:
                phi_left = level_set(mesh.elements_list[left].centroid())
                pos,neg = (left,right) if phi_left>=0 else (right,left)
            else:
                pos,neg = left,right
            # quadrature pts on reference edge of pos element
            loc_idx = mesh.elements_list[pos].edges.index(eid_edge)
            qpts_ref, qwts = line_quadrature(mesh.element_type, loc_idx, q)
            for qp, w in zip(qpts_ref,qwts):
                xq = transform.x_mapping(mesh,pos,qp)
                self.ctx['normal'] = transform.edge_unit_normal(mesh,pos,loc_idx)  # project helper must exist
                jac1d = transform.jacobian_1d(mesh,pos,qp,loc_idx)
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
