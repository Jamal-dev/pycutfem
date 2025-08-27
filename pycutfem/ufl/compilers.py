# pycutfem/ufl/compilers.py
"""
A minimal‑yet‑robust volume‑form compiler geared towards the project’s
current needs (Taylor–Hood Stokes, Poisson, linear elasticity, etc.).

This version integrates the ``GradOpInfo`` and ``VecOpInfo`` bookkeeping and a
shape- and role-aware algebraic layer. This distinguishes between symbolic
basis functions (for unknowns) and evaluated numerical data (for knowns),
enabling robust assembly of LHS matrices and RHS vectors.
"""
from __future__ import annotations
import re



from matplotlib.pylab import f
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Any, Tuple, Iterable, Mapping, Union
from collections import defaultdict
import logging
from dataclasses import dataclass
import math
from math import comb
import os

# -------------------------------------------------------------------------
#  Project imports – only the really needed bits
# -------------------------------------------------------------------------
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.fem import transform
from pycutfem.integration import volume
from pycutfem.integration.quadrature import line_quadrature

# Symbolic building blocks
from pycutfem.ufl.expressions import (
    Constant,    TestFunction,   TrialFunction,
    VectorTestFunction, VectorTrialFunction,
    Function,    VectorFunction,
    Grad, DivOperation, Inner, Dot,
    Sum, Sub, Prod, Pos, Neg,Div, Jump, FacetNormal,
    ElementWiseConstant, Derivative, Transpose,
    CellDiameter, NormalComponent,
    Restriction, Power, Trace, Hessian, Laplacian
)
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.measures import Integral
from pycutfem.ufl.quadrature import PolynomialDegreeEstimator
from pycutfem.ufl.helpers import (VecOpInfo, GradOpInfo, HessOpInfo, 
                                  required_multi_indices,
                                  _all_fields,_find_all,
                                  _trial_test, 
                                  phys_scalar_third_row, 
                                  phys_scalar_fourth_row,
                                  _as_indices,
                                  HelpersFieldAware as _hfa)
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.helpers_jit import  _build_jit_kernel_args, _scatter_element_contribs, _stack_ragged
from pycutfem.ufl.helpers_geom import (
    phi_eval, clip_triangle_to_side, fan_triangulate, map_ref_tri_to_phys, corner_tris
)



logger = logging.getLogger(__name__)
_INTERFACE_TOL = 1.0e-12 # New




# ========================================================================
#  The Compiler
# ========================================================================
class FormCompiler:
    """A single‑file *volume* compiler for mixed continuous Galerkin forms."""

    def __init__(self, dh: DofHandler, quadrature_order: int | None = None, 
                 assembler_hooks: Dict[str, Any] = None,
                 backend: str = "python"):
        if dh.mixed_element is None:
            raise RuntimeError("A MixedElement‑backed DofHandler is required.")
        self.dh, self.me = dh, dh.mixed_element
        self.qorder = quadrature_order
        self.ctx: Dict[str, Any] = {"hooks": assembler_hooks or {}}
        self._coeff_cache: Dict[tuple, np.ndarray] = {}
        self._collapsed_cache: Dict[tuple, np.ndarray] = {}
        self._basis_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self.degree_estimator = PolynomialDegreeEstimator(dh)
        self.backend = backend
        if self.backend == "jit":
            from pycutfem.jit import compile_backend
            self._compile_backend = compile_backend
        self._dispatch = {
            Constant: self._visit_Constant, 
            TestFunction: self._visit_TestFunction,
            TrialFunction: self._visit_TrialFunction, 
            VectorTestFunction: self._visit_VectorTestFunction,
            VectorTrialFunction: self._visit_VectorTrialFunction, 
            Function: self._visit_Function,
            VectorFunction: self._visit_VectorFunction, 
            Grad: self._visit_Grad,
            DivOperation: self._visit_DivOperation, Sum: self._visit_Sum,
            Sub: self._visit_Sub, Prod: self._visit_Prod, Dot: self._visit_Dot,
            Inner: self._visit_Inner, Pos: self._visit_Pos, Neg: self._visit_Neg,
            Div: self._visit_Div,
            Analytic: self._visit_Analytic,
            FacetNormal: self._visit_FacetNormal,
            ElementWiseConstant: self._visit_EWC,
            Jump: self._visit_Jump,
            Derivative        : self._visit_Derivative,
            Transpose: self._visit_Transpose,
            CellDiameter: self._visit_CellDiameter,
            NormalComponent: self._visit_NormalComponent,
            Restriction: self._visit_Restriction,
            Power: self._visit_Power,
            Trace: self._visit_Trace,
            Hessian: self._visit_Hessian,
            Laplacian: self._visit_Laplacian
        }

    # ============================ PUBLIC API ===============================
    def assemble(self, eq: Equation, bcs: Union[Mapping, Iterable, None] = None):
        ndofs = self.dh.total_dofs
        K = sp.lil_matrix((ndofs, ndofs))
        F = np.zeros(ndofs)

        # Assemble LHS if it is provided.
        if eq.a is not None:
            logger.info("Assembling LHS...")
            self.ctx["rhs"] = False
            self._assemble_form(eq.a, K)

        # Assemble RHS if it is provided.
        if eq.L is not None:
            logger.info("Assembling RHS vector F...")
            self.ctx["rhs"] = True
            self._assemble_form(eq.L, F)
        
        # --- THE FIX ---
        # Apply boundary conditions AFTER both LHS and RHS have been assembled.
        # This block is now outside of the `if eq.L` check.
        logger.info("Applying Dirichlet boundary conditions...")
        self._apply_bcs(K, F, bcs)

        logger.info("Assembly complete.")
        return K.tocsr(), F

    # ===================== VISITORS: LEAF NODES =======================
    # --------------------------------------------------------------
    # Unified, side-aware, cached basis/derivative retrieval
    # --------------------------------------------------------------
    def _basis_row(self, field: str, alpha: tuple[int, int]) -> np.ndarray:
        """
        Row for D^{alpha} φ_field at the current QP.

        • Ghost/interface path:
            - Per-QP, per-side cache in ctx["basis_values"][side][field][alpha]
            - Evaluates on (pos_eid|neg_eid) using physical mapping
            - Optionally applies per-field side masks (pos/neg)
            - Collapses element-union → field-local when padding via field map
            - Pads to ctx["global_dofs"] iff present
        • Standard element-local path:
            - Uses self._basis_cache[field] with entries:
            "val", "grad", "hess", and "derivs" (generic high-order)
        """
        ox, oy = map(int, alpha)

        # ---------- Ghost/interface: side-aware path ----------
        bv = self.ctx.get("basis_values")
        if isinstance(bv, dict):
            # Determine side
            side = self.ctx.get("side")
            if side not in ('+', '-'):
                side = '+' if float(self.ctx.get("phi_val", 0.0)) >= 0.0 else '-'

            # Per-side / per-field cache
            per_field = bv.setdefault(side, {}).setdefault(field, {})
            if alpha in per_field:
                return per_field[alpha]

            # Coordinates (physical) and owner element for this side
            x_phys = self.ctx.get("x_phys")
            if x_phys is None:
                raise KeyError("x_phys missing in context for derivative evaluation.")
            eid = int(self.ctx["pos_eid"] if side == '+' else self.ctx["neg_eid"])

            # Compute derivative row on the element (length = n_dofs_local; zero-padded across fields)
            xi, eta = transform.inverse_mapping(self.me.mesh, eid, np.asarray(x_phys))
            row = self._phys_scalar_deriv_row(field, float(xi), float(eta), ox, oy, eid)

            # Target union layout (if assembling mat/vec) and mapping
            g = self.ctx.get("global_dofs")  # None for pure functionals
            amap = _hfa.get_field_map(self.ctx, side, field)
            if amap is None:
                amap = self.ctx.get("pos_map") if side == '+' else self.ctx.get("neg_map")

            # Collapse element-union → field-local when we’re about to use a field map
            field_slice = None
            try:
                field_slice = self.me.component_dof_slices[field]
            except Exception:
                field_slice = None
            if g is not None and amap is not None:
                if len(row) not in (len(g), len(amap)):
                    if (
                        field_slice is not None
                        and len(row) == self.me.n_dofs_local
                        and (field_slice.stop - field_slice.start) == len(amap)
                    ):
                        row = row[field_slice]  # now field-local

            # Optional per-field side mask
            mask_dict = self.ctx.get("pos_mask_by_field") if side == '+' else self.ctx.get("neg_mask_by_field")
            if isinstance(mask_dict, dict):
                m_local = mask_dict.get(field)
                if m_local is not None:
                    if g is not None and amap is not None and len(row) == len(g):
                        # row already union-padded → expand mask to union and apply
                        m_full = np.zeros(len(g), dtype=row.dtype)
                        m_full[np.asarray(amap, dtype=int)] = m_local
                        row = row * m_full
                    elif g is not None and amap is not None and len(row) == len(amap):
                        # field-local before padding
                        row = row * m_local
                    elif field_slice is not None and len(row) == self.me.n_dofs_local:
                        # element-union → apply mask on the field slice
                        if (field_slice.stop - field_slice.start) == len(m_local):
                            tmp = row.copy()
                            tmp[field_slice] *= m_local
                            row = tmp
                    elif len(row) == len(m_local):
                        # field-local without padding
                        row = row * m_local
                    # otherwise: silently skip (safer than crashing)

            # Pad to union layout only if assembling a matrix/vector
            if g is not None:
                if amap is None:
                    # Assume already in target union layout
                    if len(row) != len(g):
                        raise ValueError(
                            f"Cannot pad basis row for '{field}': no map and "
                            f"len(row)={len(row)} != len(global_dofs)={len(g)}"
                        )
                else:
                    if len(row) == len(g):
                        pass  # already union-sized
                    elif len(row) == len(amap):
                        full = np.zeros(len(g), dtype=row.dtype)
                        full[np.asarray(amap, dtype=int)] = row
                        row = full
                    else:
                        raise ValueError(
                            f"Shape mismatch padding basis row for '{field}': "
                            f"len(row)={len(row)}, len(amap)={len(amap)}, len(global_dofs)={len(g)}"
                        )

            # Cache final row for this QP/side/field/alpha and return
            per_field[alpha] = row
            return row

        # ---------- Standard element-local path (volume/boundary/subcells) ----------
        cache = self._basis_cache.setdefault(field, {})

        # Fast paths if value/grad/Hessian already present
        if alpha == (0, 0):
            if "val" in cache:
                return cache["val"]
        if alpha in ((1, 0), (0, 1)):
            if "grad" in cache:
                gx, gy = cache["grad"][:, 0], cache["grad"][:, 1]
                return gx if alpha == (1, 0) else gy
        if alpha in ((2, 0), (1, 1), (0, 2)) and "hess" in cache:
            H = cache["hess"]
            return H[:, 0, 0] if alpha == (2, 0) else (H[:, 0, 1] if alpha == (1, 1) else H[:, 1, 1])

        # Generic high-order cache
        dcache = cache.setdefault("derivs", {})
        if alpha in dcache:
            return dcache[alpha]

        # Compute using exact physical mapping (needs eid & x_phys)
        eid    = self.ctx.get("eid")
        x_phys = self.ctx.get("x_phys")
        if eid is None or x_phys is None:
            raise KeyError("Derivative requested outside element context (missing 'eid'/'x_phys').")
        xi, eta = transform.inverse_mapping(self.me.mesh, int(eid), np.asarray(x_phys))
        row = self._phys_scalar_deriv_row(field, float(xi), float(eta), ox, oy, int(eid))

        # If we just computed one Hessian entry and none cached yet, compute all and store
        if alpha in ((2, 0), (1, 1), (0, 2)) and "hess" not in cache:
            d20 = row if alpha == (2, 0) else self._phys_scalar_deriv_row(field, float(xi), float(eta), 2, 0, int(eid))
            d11 = row if alpha == (1, 1) else self._phys_scalar_deriv_row(field, float(xi), float(eta), 1, 1, int(eid))
            d02 = row if alpha == (0, 2) else self._phys_scalar_deriv_row(field, float(xi), float(eta), 0, 2, int(eid))
            H = np.empty((self.me.n_dofs_local, 2, 2), dtype=row.dtype)
            H[:, 0, 0] = d20; H[:, 0, 1] = d11
            H[:, 1, 0] = d11; H[:, 1, 1] = d02
            cache["hess"] = H
            # also stash each entry into derivs for direct hits next time
            dcache[(2, 0)] = d20; dcache[(1, 1)] = d11; dcache[(0, 2)] = d02
            return row

        # Cache and return
        dcache[alpha] = row
        return row




    
    #----------------------------------------------------------------------
    def _b(self, fld): 
        return self._basis_row(fld, (0,0))
    def _v(self, fld):
        # 1) Prefer per-QP, side-aware cache
        bv = self.ctx.get("basis_values")
        if bv is not None:
            side = self.ctx.get("side")
            if side is None:
                side = '+' if self.ctx.get("phi_val", 0.0) >= 0 else '-'
            val = bv[side][fld].get((0, 0))
            if val is not None:
                return val
        # 2) Fallback to global cache
        if fld in self._basis_cache:
            return self._basis_cache[fld]["val"]
        # 3) Last resort: compute fresh (element-local)
        return self._basis_from_element_context(fld, kind="val")

    def _g(self, fld):
        gx = self._basis_row(fld, (1,0))
        gy = self._basis_row(fld, (0,1))
        return np.stack([gx, gy], axis=1)  # (nloc, 2)

    def _hess(self, fld):
        d20 = self._basis_row(fld, (2,0))
        d11 = self._basis_row(fld, (1,1))
        d02 = self._basis_row(fld, (0,2))
        H = np.empty((len(d20), 2, 2), dtype=d20.dtype)
        H[:,0,0] = d20; H[:,0,1] = d11
        H[:,1,0] = d11; H[:,1,1] = d02
        return H
    def _lookup_basis(self, field: str, alpha: tuple[int, int]) -> np.ndarray:
        """Unified entry point for any ∂^{alpha}φ_field (0 ≤ |alpha| ≤ 4)."""
        return self._basis_row(field, alpha)
    
    def _local_dofs(self): 
        if 'eid' in self.ctx:           # matrix/vector path
            return self.dh.get_elemental_dofs(self.ctx["eid"])
        return self.ctx['global_dofs']   # functional path
    
    def _visit_Restriction(self, n: Restriction):
        """
        Evaluates the operand only if the current element's tag matches the
        restriction tag. Otherwise, returns a zero-like value.
        """
        # 1. Get the current element's tag.
        #    The element ID is reliably in the context during volume/interface assembly.
        eid = self.ctx.get("eid")
        if eid is None:
            # No element context (e.g., early scalar eval) → do nothing.
            in_domain = True
        else:
            in_domain = (n.domain[eid] if hasattr(n.domain, "__getitem__")
                         else (eid in n.domain))

        # 2. Check if the element is in the active domain.
        if in_domain:
            # If it matches, proceed as if the Restriction node wasn't there.
            return self._visit(n.operand)
        else:
            # If it does not match, return a "zero" value that has the same
            # structure (shape, role) as the real result would have.
            # This is crucial for subsequent operations like dot, inner, etc.
            
            # We can achieve this by visiting the operand and then nullifying it.
            result = self._visit(n.operand)
            
            if isinstance(result, (VecOpInfo, GradOpInfo)):
                # Use the overloaded multiplication to create a zeroed-out copy.
                return result * 0.0
            elif isinstance(result, np.ndarray):
                # For raw numpy arrays, just return a zero-filled array of the same shape.
                return np.zeros_like(result)
            else: # for scalars
                return 0.0
    
    def _visit_NormalComponent(self, n:NormalComponent):
        return self._visit_FacetNormal(FacetNormal())[n.idx]
    
    def _visit_Trace(self, node: Trace):
        """trace(A): sum of diagonal on the *first* and *last* axes for GradOpInfo.
        - For function gradients: returns a 1-component VecOpInfo with a scalar value.
        - For test/trial gradients: returns a 1-component VecOpInfo of length n.
        - For plain numpy arrays: returns np.trace(A)."""
        A = self._visit(node.A)

        # Numeric tensor: use NumPy
        if isinstance(A, np.ndarray):
            return np.trace(A)

        # Gradient-like object
        if isinstance(A, GradOpInfo):
            # Function: evaluate to 2D (k,d) or (d,k) and take trace → scalar
            if A.role == "function":
                if A.data.ndim == 3 and A.coeffs is not None:
                    # (k,n,d) ⨯ (k,n) → (k,d)
                    M = np.einsum("knd,kn->kd", A.data, A.coeffs, optimize=True)
                else:
                    # already 2D from a previous Transpose
                    M = A.data
                tr_val = float(np.trace(M))
                # Wrap as a 1-component VecOpInfo (shape (1,)) so RHS assembly paths work
                return VecOpInfo(np.array([tr_val]), role="function")

            # Test/Trial: sum diagonal across first/last axes → (n,)
            if A.data.ndim != 3:
                raise ValueError(f"Trace expects a rank-3 GradOpInfo for test/trial, got {A.data.shape}.")
            k_dim, n_loc, d_dim = A.data.shape
            m = min(k_dim, d_dim)
            # sum_i A[i, :, i]  — works for either (k,n,d) or (d,n,k) because we always use axes (0,2)
            tr_vec = np.zeros((n_loc,), dtype=A.data.dtype)
            for i in range(m):
                tr_vec += A.data[i, :, i]
            # Return as a 1-component vector basis block: (1, n_loc)
            return VecOpInfo(np.stack([tr_vec]), role=A.role)
        
        # Hessian-like object
        if isinstance(A, HessOpInfo):
            # Function: already collapsed to (k,2,2) or needs coeffs contraction
            if A.role == "function":
                if A.coeffs is not None and A.data.ndim == 4:
                    Hval = np.einsum("knij,kn->kij", A.data, A.coeffs, optimize=True)
                else:
                    Hval = A.data  # (k,2,2)
                # trace over spatial axes → (k,)
                tr = Hval[..., 0, 0] + Hval[..., 1, 1]
                return VecOpInfo(tr, role="function")
            else:
                # test/trial: return (k,n) table of Laplacians
                tr = A.data[..., 0, 0] + A.data[..., 1, 1]
                return VecOpInfo(tr, role=A.role)

        # VecOpInfo or other types are not meaningful for trace
        raise TypeError(f"Trace not implemented for {type(A)}"
                        f" for role '{A.role}' with shape {A.data.shape}.")
    
    def _visit_Power(self, n: Power):
        """
        Handles the power operation, which is not a standard UFL operation.
        It can be used to raise a field to a constant power.
        """
        # Visit the base and exponent separately.
        base = self._visit(n.a)
        exponent = self._visit(n.b)

        # Handle custom VecOpInfo and GradOpInfo types explicitly
        if isinstance(base, (VecOpInfo, GradOpInfo)):
            new_data = base.data ** exponent
            # Return a new instance of the same class with the new data
            # This uses getattr for robustness, as VecOpInfo lacks 'coeffs'
            return type(base)(data=new_data, role=base.role, coeffs=getattr(base, 'coeffs', None))

        # Ensure the exponent is a scalar (constant).
        if not np.isscalar(exponent):
            raise ValueError("Exponent must be a scalar constant.")

        # Raise the base to the power of the exponent.
        return base ** exponent


    def _visit_Derivative(self, op: Derivative):
        # --------------------------------------------------------------
        # 1) Operand is a Jump  →  Jump of derivatives
        # --------------------------------------------------------------
        if isinstance(op.f, Jump):
            # Build two new Derivative nodes with the same multi-index
            dpos = Derivative(op.f.u_pos, *op.order)
            dneg = Derivative(op.f.u_neg, *op.order)

            phi_old = self.ctx.get('phi_val', 0.0)
            eid_old = self.ctx.get('eid')
            old_side  = self.ctx.get("side", None)

            # + side
            self.ctx['phi_val'] = 1.0
            self.ctx["side"]    = '+' 
            if 'pos_eid' in self.ctx:
                self.ctx['eid'] = self.ctx['pos_eid']
            v_pos = self._visit(dpos)

            # – side
            self.ctx['phi_val'] = -1.0
            self.ctx["side"]    = '-'
            if 'neg_eid' in self.ctx:
                self.ctx['eid'] = self.ctx['neg_eid']
            v_neg = self._visit(dneg)

            # restore
            self.ctx['phi_val'] = phi_old
            if eid_old is None:
                self.ctx.pop('eid', None)
            else:
                self.ctx['eid'] = eid_old
            if old_side is None:
                self.ctx.pop('side', None)
            else:
                self.ctx['side'] = old_side

            return v_pos - v_neg

        # --------------------------------------------------------------
        # 2) Operand is *not* a Jump (previous logic, unchanged)
        # --------------------------------------------------------------
        fld, alpha = op.f.field_name, op.order
        row = self._lookup_basis(fld, alpha)[np.newaxis, :]      # (1,n)

        if op.f.is_function:
            coeffs = op.f.get_nodal_values(self._local_dofs())[np.newaxis, :]
            data, role = coeffs * row, "function"
        elif op.f.is_trial:
            data, role = row, "trial"
        else:
            data, role = row, "test"

        return VecOpInfo(data, role=role)
   
    # --- Knowns (evaluated to numerical values) ---
    def _visit_Constant(self, n: Constant):
        if n.dim ==0:
            return float(n.value)
        return np.asarray(n.value)
    
    def _visit_FacetNormal(self, n: FacetNormal): # New
        """Returns the normal vector from the context.""" # New
        if 'normal' not in self.ctx: # New
            raise RuntimeError("FacetNormal accessed outside of a facet or interface integral context.") # New
        return self.ctx['normal'] # New
    
    # -- Element-wise constants ----------------------------------------
    def _visit_EWC(self, n:ElementWiseConstant):
        eid = self.ctx.get("eid")
        if eid is None:
            # Robust fallback for interface/ghost paths where only side-specific ids are set.
            # On the interface, pos_eid == neg_eid == owner element.
            eid = self.ctx.get("pos_eid", self.ctx.get("neg_eid"))
        if eid is None:
            raise RuntimeError("ElementWiseConstant evaluated outside an element loop.")
        return n.value_on_element(int(eid))
    
    
    def _visit_Pos(self, n: Pos):
        """Restrict operand to the '+' side. On Γ, set ctx['side'] and eid accordingly."""
        if self.ctx.get('on_interface', False):
            phi_old  = self.ctx.get('phi_val', None)
            side_old = self.ctx.get('side', None)
            eid_old  = self.ctx.get('eid', None)
            try:
                self.ctx['phi_val'] =  1.0
                self.ctx['side']    = '+'
                if 'pos_eid' in self.ctx:
                    self.ctx['eid'] = self.ctx['pos_eid']
                return self._visit(n.operand)
            finally:
                if phi_old is None: self.ctx.pop('phi_val', None)
                else:               self.ctx['phi_val'] = phi_old
                if side_old is None: self.ctx.pop('side', None)
                else:                self.ctx['side'] = side_old
                if eid_old is None:  self.ctx.pop('eid', None)
                else:                self.ctx['eid'] = eid_old
        # The '+' side is where phi >= 0 (a closed set)
        if 'phi_val' in self.ctx and self.ctx['phi_val'] < 0.0: 
            # We are on the strictly negative side, so return zero.
            op_val = self._visit(n.operand)
            return op_val * 0.0
        return self._visit(n.operand)

    def _visit_Neg(self, n: Neg):
        """Restrict operand to the '−' side. On Γ, set ctx['side'] and eid accordingly."""
        if self.ctx.get('on_interface', False):
            phi_old  = self.ctx.get('phi_val', None)
            side_old = self.ctx.get('side', None)
            eid_old  = self.ctx.get('eid', None)
            try:
                self.ctx['phi_val'] = -1.0
                self.ctx['side']    = '-'
                if 'neg_eid' in self.ctx:
                    self.ctx['eid'] = self.ctx['neg_eid']
                return self._visit(n.operand)
            finally:
                if phi_old is None: self.ctx.pop('phi_val', None)
                else:               self.ctx['phi_val'] = phi_old
                if side_old is None: self.ctx.pop('side', None)
                else:                self.ctx['side'] = side_old
                if eid_old is None:  self.ctx.pop('eid', None)
                else:                self.ctx['eid'] = eid_old
        # The '-' side is where phi < 0 (an open set)
        if 'phi_val' in self.ctx and self.ctx['phi_val'] >= 0.0: 
            # We are on the positive or zero side, so return zero.
            op_val = self._visit(n.operand)
            return op_val * 0.0
        return self._visit(n.operand) 

    def _visit_Jump(self, n: Jump):
        phi_old  = self.ctx.get('phi_val', 0.0)
        eid_old  = self.ctx.get('eid')
        side_old = self.ctx.get('side')

        # u(+)  on the + side / pos_eid
        self.ctx['phi_val'] =  1.0
        self.ctx['side']    = '+'
        if 'pos_eid' in self.ctx:
            self.ctx['eid'] = self.ctx['pos_eid']
        u_pos = self._visit(n.u_pos)

        # u(-)  on the – side / neg_eid
        self.ctx['phi_val'] = -1.0
        self.ctx['side']    = '-'
        if 'neg_eid' in self.ctx:
            self.ctx['eid'] = self.ctx['neg_eid']
        u_neg = self._visit(n.u_neg)

        # restore
        self.ctx['phi_val'] = phi_old
        if eid_old is None: self.ctx.pop('eid', None)
        else:               self.ctx['eid'] = eid_old
        if side_old is None: self.ctx.pop('side', None)
        else:                self.ctx['side'] = side_old

        return u_pos - u_neg

###########################################################################################3
########### --- Functions and VectorFunctions (evaluated to numerical values) ---
    def _visit_Function(self, n: Function):
        logger.debug(f"Visiting Function: {n.field_name}")
        u_loc = n.get_nodal_values(self._local_dofs())      # element-local coeffs
        phi   = self._b(n.field_name)                       # trace on current side

        # ghost-edge: enlarge coeff vector to |global_dofs|
        if phi.shape[0] != u_loc.shape[0] and "global_dofs" in self.ctx:
            padded = np.zeros_like(phi)
            side = '+' if self.ctx.get("phi_val", 0.0) >= 0 else '-'
            amap = _hfa.get_field_map(self.ctx, side, n.field_name)
            if amap is None:
                # Fallback for safety if per-field map is missing
                amap = self.ctx["pos_map"] if side == '+' else self.ctx["neg_map"]
            padded[amap] = u_loc
            u_loc = padded

        data = [u_loc * phi]
        return VecOpInfo(np.stack(data), role="function")

    def _visit_VectorFunction(self, n: VectorFunction):
        """
        Return VecOpInfo(k, n_loc) with k = len(field_names).
        Handles both volume and ghost-edge assembly: coefficients are padded
        exactly as in _visit_Function when we are on a ghost edge.
        """
        logger.debug(f"Visiting VectorFunction: {n.field_names}")
        data = []
        local_dofs = self._local_dofs()                          # element or union
        for i, fld in enumerate(n.field_names):
            coeffs = n.components[i].get_nodal_values(local_dofs)  # (n_loc,)
            phi    = self._b(fld)                                   # (n_loc,) – padded on ghost edge

            # --- pad coefficients on ghost edges --------------------
            if phi.shape[0] != coeffs.shape[0] and 'global_dofs' in self.ctx:
                padded = np.zeros_like(phi)
                side = '+' if self.ctx.get('phi_val', 0.0) >= 0 else '-'
                amap = _hfa.get_field_map(self.ctx, side, fld)
                if amap is None:
                    amap = self.ctx['pos_map'] if side == '+' else self.ctx['neg_map']
                padded[amap] = coeffs
                coeffs = padded

            data.append(coeffs * phi)                              # component-wise
        return VecOpInfo(np.stack(data), role="function")


    # --- Unknowns (represented by basis functions) ---
    def _visit_TestFunction(self, n):
        logger.debug(f"Visiting TestFunction: {n.field_name}")
        row = self._lookup_basis(n.field_name, (0, 0))[np.newaxis, :]
        return VecOpInfo(row, role="test")
    def _visit_TrialFunction(self, n):
        logger.debug(f"Visiting TrialFunction: {n.field_name}")
        row = self._lookup_basis(n.field_name, (0, 0))[np.newaxis, :]
        return VecOpInfo(row, role="trial")

    def _visit_VectorTestFunction(self, n):
        logger.debug(f"Visiting VectorTestFunction: {n.field_names}")
        names = _hfa._filter_fields_for_side(n, list(n.field_names))
        return VecOpInfo(np.stack([self._lookup_basis(f, (0, 0)) for f in names]), role="test")
    def _visit_VectorTrialFunction(self, n):
        logger.debug(f"Visiting VectorTrialFunction: {n.field_names}")
        names = _hfa._filter_fields_for_side(n, list(n.field_names))
        return VecOpInfo(np.stack([self._lookup_basis(f, (0, 0)) for f in names]), role="trial")
    # ================== VISITORS: OPERATORS ========================
    def _visit_Grad(self, n: Grad):
        """Return GradOpInfo with shape (k, n_dofs_local, d) for test/trial, or (k,d) for function (collapsed)."""
        op = n.operand
        logger.debug(f"Entering _visit_Grad for operand type {type(op)}")
        # Distribute grad over sided restrictions so Pos/Neg work with grad()
        #   grad(Pos(u)) = Pos(grad(u))
        #   grad(Neg(u)) = Neg(grad(u))
        if isinstance(op, Pos):
            return self._visit(Pos(Grad(op.operand)))
        if isinstance(op, Neg):
            return self._visit(Neg(Grad(op.operand)))
        if isinstance(op, Jump):
            return self._visit(Jump(Grad(op.u_pos), Grad(op.u_neg)))

        # 1) role
        if isinstance(op, (TestFunction, VectorTestFunction)):
            role = "test"
        elif isinstance(op, (TrialFunction, VectorTrialFunction)):
            role = "trial"
        elif isinstance(op, (Function, VectorFunction)):
            role = "function"
        else:
            raise NotImplementedError(f"grad() not implemented for {type(op)}")

        # 2) fields
        fields = op.field_names if hasattr(op, "field_names") else [op.field_name]
        fields = _hfa._filter_fields_for_side(op, list(fields))

        k_blocks = []
        if role == "function":
            # collapse once per component and cache
            local_dofs = self._local_dofs()
            for i, fld in enumerate(fields):
                key_coll = (id(op), "grad", i, self.ctx.get("eid"), self.ctx.get("side"), self.ctx.get("phi_val"))
                gval = self._collapsed_cache.get(key_coll)
                if gval is None:
                    g = self._g(fld)                                    # (n,2)
                    # coefficients padded once per component
                    key_c = (id(op), i, tuple(local_dofs))
                    coeffs = self._coeff_cache.get(key_c)
                    if coeffs is None:
                        if hasattr(op, "components"):
                            coeffs = op.components[i].padded_values(local_dofs)
                        else:
                            coeffs = op.padded_values(local_dofs)
                        self._coeff_cache[key_c] = coeffs
                    gval = np.einsum("nd,n->d", g, coeffs, optimize=True)  # (2,)
                    self._collapsed_cache[key_coll] = gval
                k_blocks.append(gval)
            return GradOpInfo(np.stack(k_blocks), role=role)
        else:
            for fld in fields:
                g = self._g(fld)
                k_blocks.append(g)
            return GradOpInfo(np.stack(k_blocks), role=role)
    
    def _visit_Hessian(self, n: Hessian):
        """
        Build per-component Hessian tables.
          • Test/Trial → data shape (k, n, 2, 2).
          • Function   → collapsed per component, data shape (k, 2, 2).
        """
        op = n.operand

        # Jump(u_pos,u_neg): evaluate sided Hessians via _visit_Jump
        if isinstance(op, Jump):
            return self._visit(Jump(Hessian(op.u_pos), Hessian(op.u_neg)))

        # Pos/Neg: gate the Hessian result on the active side
        if isinstance(op, Pos):
            return self._visit(Pos(Hessian(op.operand)))
        if isinstance(op, Neg):
            return self._visit(Neg(Hessian(op.operand)))

        # Determine role
        if isinstance(op, (TestFunction, VectorTestFunction)):
            role = "test"
        elif isinstance(op, (TrialFunction, VectorTrialFunction)):
            role = "trial"
        elif isinstance(op, (Function, VectorFunction)):
            role = "function"
        else:
            raise NotImplementedError(f"Hessian not implemented for {type(op)}")

        # Determine component list
        if isinstance(op, (VectorFunction, VectorTestFunction, VectorTrialFunction)):
            fields = list(op.field_names)  # e.g. ["ux","uy"]
        else:
            fields = [op.field_name]

        local_dofs = self._local_dofs()

        # helper to get padded coefficients once per component
        def _get_coeff(i):
            key = (id(op), i, tuple(local_dofs))
            c = self._coeff_cache.get(key)
            if c is None:
                if hasattr(op, "components"):
                    c = op.components[i].padded_values(local_dofs)
                else:
                    c = op.padded_values(local_dofs)
                self._coeff_cache[key] = c
            return c

        k_blocks = []
        if role == "function":
            # collapse once per component and cache result per QP
            for i, fld in enumerate(fields):
                key_coll = (id(op), "hess", i, self.ctx.get("eid"), self.ctx.get("side"), self.ctx.get("phi_val"))
                Hval = self._collapsed_cache.get(key_coll)
                if Hval is None:
                    Htbl = self._hess(fld)                    # (n,2,2)
                    coeffs = _get_coeff(i)                    # (n,)
                    Hval = np.einsum("nij,n->ij", Htbl, coeffs, optimize=True)  # (2,2)
                    self._collapsed_cache[key_coll] = Hval
                k_blocks.append(Hval)
            data = np.stack(k_blocks)                          # (k,2,2)
            return HessOpInfo(data, role=role)                 # coeffs=None → already collapsed
        else:
            # build basis tables per component
            for fld in fields:
                Htbl = self._hess(fld)                         # (n,2,2)
                k_blocks.append(Htbl)
            data = np.stack(k_blocks)                          # (k,n,2,2)
            return HessOpInfo(data, role=role)

    def _visit_Laplacian(self, n: Laplacian):
        """
        Laplacian(u) := trace(Hessian(u)).

        For Function/VectorFunction operands, returns a VecOpInfo with role="function"
        containing the collapsed per-component Laplacians (k,).
        For Test/Trial operands, returns a VecOpInfo (k, n) of basis Laplacians.
        """
        H = self._visit_Hessian(Hessian(n.operand))
        tr = H.trace()      # VecOpInfo with role preserved
        return tr

    def _visit_DivOperation(self, n: DivOperation):
        grad_op = self._visit(Grad(n.operand))           # (k, n_loc, d)
        logger.debug(f"Visiting DivOperation for operand of type {type(n.operand)}, grad_op shape: {grad_op.data.shape}")

        if grad_op.role == "function":
            # scaled the gradient
            if grad_op.coeffs is not None:
                grad_val = np.einsum("knd,kn->kd", grad_op.data, grad_op.coeffs, optimize=True) # (k,d) (2,2)
            else:
                grad_val = grad_op.data
            div_vec = np.sum([grad_val[i, i] for i in range(grad_val.shape[0])], axis=0)  # sum over k
        else:
            # ∇·v  =  Σ_i ∂v_i/∂x_i   → length n_loc (22) vector
            div_vec = np.sum([grad_op.data[i, :, i]          # pick diagonal components
                            for i in range(grad_op.data.shape[0])],
                            axis=0)

        # Decide which side of the bilinear form this lives on
        op = n.operand
        if isinstance(op, (TestFunction, VectorTestFunction)):
            role = "test"
        elif isinstance(op, (TrialFunction, VectorTrialFunction)):
            role = "trial"
        elif isinstance(op, (Function, VectorFunction)):
            role = "function"
        else:
            role = "none"

        # Wrap as VecOpInfo with a single component so that _visit_Prod
        # can use the role information to orient the outer product.
        return VecOpInfo(np.stack([div_vec]), role=role)
    

    def _visit_CellDiameter(self, node):
        eid = self.ctx.get("eid")
        if eid is None:
            eid = self.ctx.get("pos_eid", self.ctx.get("neg_eid"))
        if eid is None:
            raise KeyError("CellDiameter() requires 'eid' in context; set it in the element loop.")
        return self.me.mesh.element_char_length(int(eid))


    
    # ================= VISITORS: ALGEBRAIC ==========================
    def _visit_Sum(self, n): return self._visit(n.a) + self._visit(n.b)
    def _visit_Sub(self, n):
        a = self._visit(n.a)
        b = self._visit(n.b)
        return a - b
    
    def _visit_Transpose(self, node: Transpose):
        A = self._visit(node.A)

        # Plain numpy: use .T
        if isinstance(A, np.ndarray):
            return A.T

        # Grad basis/operators
        if isinstance(A, GradOpInfo):
            return A.transpose()
        # Hessian basis/operators
        if isinstance(A, HessOpInfo):
            return A.transpose()

        # Vector basis/operators
        if isinstance(A, VecOpInfo):
            # VecOpInfo stores (k, n) -> transpose to (n, k)
            return VecOpInfo(A.data.T, role=A.role)

        raise TypeError(f"Transpose not implemented for {type(A)}")

    def _visit_Div(self, n):
        """Handles scalar, vector, or tensor division by a scalar."""
        numerator = self._visit(n.a)
        denominator = self._visit(n.b)

        # NumPy correctly handles element-wise division of an array by a scalar.
        return numerator / denominator

    def _visit_Prod(self, n: Prod):
        a = self._visit(n.a)
        b = self._visit(n.b)
        a_data = a.data if isinstance(a, (VecOpInfo, GradOpInfo)) else a
        b_data = b.data if isinstance(b, (VecOpInfo, GradOpInfo)) else b
        shape_a = getattr(a_data,"shape", None)
        shape_b = getattr(b_data,"shape", None)
        # a_vec = np.squeeze(a_data) 
        # b_vec = np.squeeze(b_data)
        # role_a = getattr(a, 'role', None)
        # role_b = getattr(b, 'role', None)
        logger.debug(f"Entering _visit_Prod for  ('{n.a!r}' * '{n.b!r}') on {'RHS' if self.ctx['rhs'] else 'LHS'}") #, a.info={getattr(a, 'info', None)}, b.info={getattr(b, 'info', None)}
        # print(f" Product: a type={type(a)}, shape={shape_a}, b type={type(b)}, shape={shape_b}, side: {'RHS' if self.ctx['rhs'] else 'LHS'}"
        #       f" roles: a={getattr(a, 'role', None)}, b={getattr(b, 'role', None)}")

        # scalar * scalar multiplication
        if np.isscalar(a) and np.isscalar(b):
            return a * b
        # RHS: (Function) * (Function)  → scalar
        if (isinstance(a, VecOpInfo) and isinstance(b, VecOpInfo)
            and a.role == b.role == "function"):
            return a * b          # delegates to VecOpInfo.__mul_
        # First, handle context-specific RHS assembly.
        if self.ctx["rhs"]:
            # Case 1: (known Function) * (Test Function) -> results in a vector
            if isinstance(a, VecOpInfo) and a.role == "function" and isinstance(b, VecOpInfo) and b.role == "test":
                if a.data.shape[0] == 1: # Scalar field * test function
                    return np.sum(a.data) * b.data[0]
                return np.einsum("kn,kn->n", a.data, b.data, optimize=True) # Vector field

            # Symmetric orientation
            if isinstance(b, VecOpInfo) and b.role == "function" and isinstance(a, VecOpInfo) and a.role == "test":
                if b.data.shape[0] == 1: # Scalar field * test function
                    return np.sum(b.data) * a.data[0]
                return np.einsum("kn,kn->n", b.data, a.data, optimize=True) # Vector field
            

            # Case 2: (scalar Constant) * (Test Function) -> results in a vector
            # This now takes precedence over the general scalar multiplication.
            if np.isscalar(a) and isinstance(b, VecOpInfo) and b.role == "test":
                if b.data.shape[0] == 1: # special scalar case
                    # This correctly returns a numeric array, not a VecOpInfo object.
                    return a * b.data[0]
                return VecOpInfo(a * b.data, role="test")
            if np.isscalar(b) and isinstance(a, VecOpInfo) and a.role == "test":
                if a.data.shape[0] == 1:
                    return b * a.data[0]
                return VecOpInfo(b * a.data, role="test")
            elif isinstance(b, np.ndarray) and isinstance(a, VecOpInfo):
                return a * b # p * normal
            elif isinstance(a, np.ndarray) and isinstance(b, VecOpInfo):
                return b * a # normal * p
        
        # --- General Purpose Logic (used mostly for LHS) ---

        # General scalar multiplication (if not a special RHS case)
        if np.isscalar(a) and isinstance(b, (VecOpInfo, GradOpInfo, np.ndarray)):
            return a * b
        if np.isscalar(b) and isinstance(a, (VecOpInfo, GradOpInfo, np.ndarray)):
            return b * a
            
        # LHS Matrix Assembly: (Test * Trial)
        if not self.ctx["rhs"]:
            if isinstance(a, VecOpInfo) and isinstance(b, VecOpInfo):
                return a * b
            elif isinstance(b, np.ndarray) and isinstance(a, VecOpInfo):
                return a * b
            elif isinstance(a, np.ndarray) and isinstance(b, VecOpInfo):
                return b * a
        
        raise TypeError(f"Unsupported product 'type(a)={type(a)}, type(b)={type(b)}'{n.a} * {n.b}' for {'RHS' if self.ctx['rhs'] else 'LHS'}"
                        f" for roles a={getattr(a, 'role', None)}, b={getattr(b, 'role', None)}"
                        f" and data shapes a={shape_a}, b={shape_b}"    )

        

    def _visit_Dot(self, n: Dot):
        a = self._visit(n.a)
        b = self._visit(n.b)
        a_data = a.data if isinstance(a, (VecOpInfo, GradOpInfo)) else a
        b_data = b.data if isinstance(b, (VecOpInfo, GradOpInfo)) else b
        role_a = getattr(a, 'role', None)
        role_b = getattr(b, 'role', None)
        # print(f"visit dot: role_a={role_a}, role_b={role_b}, a={a}, b={b}, side: {'RHS' if self.ctx['rhs'] else 'LHS'}"
        #       f" type_a={type(a)}, type_b={type(b)}")
        logger.debug(f"Entering _visit_Dot for types {type(a)} . {type(b)}")

        def rhs():
            # ------------------------------------------------------------------
            # RHS  •  dot(  Function ,  Test  )   or   dot( Test , Function )
            # ------------------------------------------------------------------
            # Case 1: Function · Test  (VecOpInfo, VecOpInfo)
            if  isinstance(a, VecOpInfo) and isinstance(b, VecOpInfo):
                if a.role == "function" and b.role == "test":

                    return a.dot_vec(b)  # function . test
                if b.role == "function" and a.role == "test":
                    
                    return b.dot_vec(a)  # test . function


            # ------------------------------------------------------------------
            # Function · Function   (needed for  dot(grad(u_k), u_k)  on RHS)
            # ------------------------------------------------------------------
            # Case 2: Function · Function  (GradOpInfo, VecOpInfo)
            if  isinstance(a, GradOpInfo) and isinstance(b, VecOpInfo) \
            and a.role == b.role == "function":
                return a.dot_vec(b)  # grad(u_k) . u_k
            if  isinstance(b, GradOpInfo) and isinstance(a, VecOpInfo) \
            and b.role == a.role == "function":
                return a.dot_grad(b)
            
            
            # Constant (numpy 1-D) · test VecOpInfo  → length-n vector
            # Case 3: Constant(np.array([u1,u2])) · Test
            if isinstance(a, np.ndarray) and a.ndim == 1 and \
            isinstance(b, VecOpInfo) and b.role == "test":
                return b.dot_const(a)

            if isinstance(b, np.ndarray) and b.ndim == 1 and \
            isinstance(a, VecOpInfo) and a.role == "test":
                return a.dot_const(b)
            
            # Case 4: Constant(np.array([u1,u2])) · Trial or Function, no test so output is VecOpInfo
            if isinstance(a, np.ndarray) and a.ndim == 1 and \
             isinstance(b, (VecOpInfo)) and b.role in {"trial", "function"}:
                # return b.dot_const(a) if b.ndim==2 else b.dot_vec(a)
                return b.dot_const_vec(a) 
            elif isinstance(b, np.ndarray) and b.ndim == 1 and \
             isinstance(a, (VecOpInfo)) and a.role in {"trial", "function"}:
                # return a.dot_const(b) if a.ndim==2 else a.dot_vec(b)
                return a.dot_const_vec(b) 
            elif isinstance(a, np.ndarray) and isinstance(b, GradOpInfo):
                return b.left_dot(a)  # np.ndarray · ∇u
            elif isinstance(b, np.ndarray) and isinstance(a, GradOpInfo):
                return a.dot_vec(b)
            # ------------------------------------------------------------------
            # Case:  Grad(Trial) · Grad(Function)       ∇u_trial · ∇u_k 
            # ------------------------------------------------------------------
            if isinstance(a, GradOpInfo) and  (a.role == "function") \
            and isinstance(b, GradOpInfo) and (b.role == "function"):
                return a.dot(b)
            if isinstance(b, np.ndarray) and  (role_b == None) \
            and isinstance(a, np.ndarray) and (role_a == None):
                return np.dot(a, b)  # plain numpy dot product
            # --- Hessian · vector (right) and vector · Hessian (left) -------------
            # Accept geometric constant vectors (facet normals) or plain numpy 1D vectors.
            if isinstance(a, HessOpInfo) and (
                isinstance(b, np.ndarray) or (isinstance(b, VecOpInfo) and role_b in {"function",None})
            ):
                return a.dot_right(b)

            if isinstance(b, HessOpInfo) and (
                isinstance(a, np.ndarray) or (isinstance(a, VecOpInfo) and role_a in {"function",None} )
            ):
                return b.dot_left(a)

        if self.ctx["rhs"]:
            result = rhs()
            if result is not None:
                return result
            else:
                raise TypeError(f"Unsupported dot product for RHS: '{n.a} . {n.b}'")
        
        if role_a == None and role_b == "test":
            # Special case like dot( Constat(np.array([u1,u2])), TestFunction('v') )
            # This is a special case where we have a numerical vector on the LHS
            # and a test function basis on the RHS.
            if isinstance(a, np.ndarray) and isinstance(b, VecOpInfo) and b.role == "test":
                return b.dot_const(a)
            if isinstance(b, np.ndarray) and isinstance(a, VecOpInfo) and a.role == "test":
                return a.dot_const(b)
        # Dot product between a basis field and a numerical vector
        if isinstance(a, (VecOpInfo)) and isinstance(b, np.ndarray): 
            return a.dot_const_vec(b)
        elif isinstance(b, (VecOpInfo)) and isinstance(a, np.ndarray): 
            return b.dot_const_vec(a)
        elif isinstance(a, (GradOpInfo)) and isinstance(b, np.ndarray):
            result = a.dot_vec(b)
            # print(f"visit dot: GradOpInfo . result: {result}, result shape: {result.shape}, role: {getattr(result, 'role', None)}")
            return result
        elif isinstance(b, (GradOpInfo)) and isinstance(a, np.ndarray):
            return b.left_dot(a)
        

        # mass matrix case: VecOpInfo . VecOpInfo
        if isinstance(a, VecOpInfo) and isinstance(b, VecOpInfo):
            logger.debug(f"visit dot: Both operands are VecOpInfo: {a.role} . {b.role}")
            if a.role == "test" and b.role in {"trial", "function"}:
                # return np.dot(a_data.T, b_data)  # test . trial
                return b.dot_vec(a)  # test . trial
            elif b.role == "test" and a.role in {"trial", "function"}:
                # return np.dot(b_data.T, a_data)  # tiral . test
                return a.dot_vec(b)  # trial . test
        
        # ------------------------------------------------------------------
        # Case:  VectorFunction · Grad(⋅)      u_k · ∇w_test
        # ------------------------------------------------------------------
        if isinstance(a, VecOpInfo) and isinstance(b, GradOpInfo) \
        and a.role == "function" and b.role == "test":
            return b.left_dot(a)  # u_k · ∇w
        
        # ------------------------------------------------------------------
        # case grad(u_trial) . u_k
        if isinstance(a, GradOpInfo) and ((isinstance(b, VecOpInfo) \
            and (b.role == "function" )) and a.role == "trial" 
            ):

            return a.dot_vec(b)  # ∇u_trial · u_k
        # ------------------------------------------------------------------
        # case u_trial . grad(u_k)
        if isinstance(b, GradOpInfo) and ((isinstance(a, VecOpInfo) \
            and (a.role == "trial" )) and b.role == "function"
            ):
            return b.left_dot(a)  # u_trial · ∇u_k
        
 
        
        
        # ------------------------------------------------------------------
        # Case:  Grad(Function) · Vec(Trial)      (∇u_k) · u
        # ------------------------------------------------------------------
        if isinstance(a, GradOpInfo) and a.role == "function" \
        and isinstance(b, VecOpInfo)  and b.role == "trial":

            return a.dot_vec(b)  # ∇u_k · u_trial
        
        # ------------------------------------------------------------------
        # Case:  Vec(Function) · Grad(Trial)      u_k · ∇u_trial
        # ------------------------------------------------------------------
        if isinstance(a, VecOpInfo)  and a.role == "function" \
        and isinstance(b, GradOpInfo) and b.role == "trial":
            return b.left_dot(a)  # u_k · ∇u_trial
            
        # ------------------------------------------------------------------
        # Case:  Grad(Trial) · Grad(Function)       ∇u_trial · ∇u_k 
        # ------------------------------------------------------------------
        if isinstance(a, GradOpInfo) and (a.role == "trial" or a.role == "function") \
        and isinstance(b, GradOpInfo) and (b.role == "trial" or b.role == "function"):
            return a.dot(b)
        
        # --- Hessian · vector (right) and vector · Hessian (left) -------------
        # Accept geometric constant vectors (facet normals) or plain numpy 1D vectors.
        if isinstance(a, HessOpInfo) and (
            isinstance(b, np.ndarray) or (isinstance(b, VecOpInfo) and role_b in {"function",None})
        ):
            return a.dot_right(b)

        if isinstance(b, HessOpInfo) and (
            isinstance(a, np.ndarray) or (isinstance(a, VecOpInfo) and role_a in {"function",None} )
        ):
            return b.dot_left(a)
        
        # Both are numerical vectors (RHS)
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray): return np.dot(a,b)

        raise TypeError(f"Unsupported dot product '{n.a} . {n.b}'"
                        f" for roles a={getattr(a, 'role', None)}, b={getattr(b, 'role', None)}"
                        f" and data shapes a={getattr(a_data, 'shape', None)}, b={getattr(b_data, 'shape', None)}")

    # ================ VISITORS: INNER PRODUCTS =========================
    def _visit_Inner(self, n: Inner):
        a = self._visit(n.a)
        b = self._visit(n.b)
        role_a = getattr(a, 'role', None)
        role_b = getattr(b, 'role', None)
        logger.debug(f"Entering _visit_Inner for types {type(a)} : {type(b)}")
        # print(f"Inner: {a} . {b}, side: {'RHS' if self.ctx['rhs'] else 'LHS'}"
        #       f" role_a={role_a}, role_b={role_b}, a.shape={getattr(a, 'shape', None)}, b.shape={getattr(b, 'shape', None)}")

        rhs = bool(self.ctx.get("rhs"))

        # ============================= RHS =============================
        if rhs:
            # ---- Hessian(Function) · Hessian(Test/Trial)  → (n,) ----
            if isinstance(a, HessOpInfo) and isinstance(b, HessOpInfo):
                if a.role == 'function' and b.role in ('test', 'trial'):
                    # a.data is either (k,2,2) collapsed or (k,n,2,2) + a.coeffs
                    if a.coeffs is not None and a.data.ndim == 4:
                        kdij = np.einsum('knij,kn->kij', a.data, a.coeffs, optimize=True)  # (k,2,2)
                    else:
                        kdij = a.data  # (k,2,2) already collapsed
                    f = np.einsum('kij,knij->n', kdij, b.data, optimize=True)  # (n,)
                    return f
                if b.role == 'function' and a.role in ('test', 'trial'):
                    if b.coeffs is not None and b.data.ndim == 4:
                        kdij = np.einsum('knij,kn->kij', b.data, b.coeffs, optimize=True)
                    else:
                        kdij = b.data
                    f = np.einsum('kij,knij->n', kdij, a.data, optimize=True)
                    return f
                if a.role == 'function' and b.role == 'function':
                    def _collapse(g: HessOpInfo):
                        if g.data.ndim == 4:               # (k,n,2) with coeffs
                            if g.coeffs is None:
                                raise ValueError("HessOpInfo(function) with 4D data requires coeffs to collapse.")
                            return np.einsum("knij,kn->kij", g.data, g.coeffs, optimize=True)  # (k,2)
                        return g.data                      # already (k,2)
                    A = _collapse(a)
                    B = _collapse(b)
                    if A.ndim == B.ndim == 3:
                        return float(np.einsum("kij,kij->", A, B, optimize=True))
                    raise ValueError(f"RHS inner(Hess,Hess) expects 3D data; got {A.shape}, {B.shape}")

                raise NotImplementedError(f"Unsupported RHS Hessian inner-product configuration: "
                                        f"a.role={getattr(a,'role',None)}, b.role={getattr(b,'role',None)}")

            # ---- Grad(Function) · Grad(Test/Trial)  → (n,) ----
            if isinstance(a, GradOpInfo) and isinstance(b, GradOpInfo):
                if a.role == "function" and b.role in ("test", "trial"):
                    # a is (k,d) collapsed OR (k,n,d)+coeffs
                    if a.coeffs is not None and a.data.ndim == 3:
                        grad_val = np.einsum("knd,kn->kd", a.data, a.coeffs, optimize=True)  # (k,d)
                    else:
                        grad_val = a.data  # (k,d)
                    return np.einsum("kd,knd->n", grad_val, b.data, optimize=True)
                if b.role == "function" and a.role in ("test", "trial"):
                    if b.coeffs is not None and b.data.ndim == 3:
                        grad_val = np.einsum("knd,kn->kd", b.data, b.coeffs, optimize=True)
                    else:
                        grad_val = b.data
                    return np.einsum("kd,knd->n", grad_val, a.data, optimize=True)
                if a.role == "function" and b.role == "function":
                    # collapse to (k,2) if needed
                    def _collapse(g: GradOpInfo):
                        if g.data.ndim == 3:               # (k,n,2) with coeffs
                            if g.coeffs is None:
                                raise ValueError("GradOpInfo(function) with 3D data requires coeffs to collapse.")
                            return np.einsum("knd,kn->kd", g.data, g.coeffs, optimize=True)  # (k,2)
                        return g.data                      # already (k,2)
                    A = _collapse(a)
                    B = _collapse(b)
                    if A.ndim == B.ndim == 2:
                        return float(np.einsum("kd,kd->", A, B, optimize=True))
                    raise ValueError(f"RHS inner(Grad,Grad) expects 2D data; got {A.shape}, {B.shape}")
                # (test,trial) or (trial,test) would be LHS; fall through to LHS block below.
            
            # ---- Vec(Function) · Vec(Test/Trial)  (e.g., Laplacian) → (n,) ----
            if isinstance(a, VecOpInfo) and isinstance(b, VecOpInfo):
                # We rely on visitors to collapse Function operands to (k,)
                if a.role == "function" and b.role in ("test", "trial"):
                    return a.inner(b)  # handles (k,) ⨯ (k,n) → (n,)
                if b.role == "function" and a.role in ("test", "trial"):
                    return b.inner(a)  # handles (k,n) ⨯ (k,) → (n,)
                if a.role == "function" and b.role == "function":
                    # both collapsed to (k,) → scalar
                    A, B = a.data, b.data
                    if A.ndim == B.ndim == 1:
                        return float(np.einsum("k,k->", A, B, optimize=True))
                    raise ValueError(f"RHS inner(Function,Function) expects 1D data; got {A.shape}, {B.shape}")

            # ---- Numeric tensor with Grad basis on RHS ----
            if isinstance(a, np.ndarray) and isinstance(b, VecOpInfo) and a.ndim == 1:
                if b.role == "function":
                    u_vals = np.sum(b.data, axis=1)
                    return np.dot(a, u_vals)
                return np.einsum("k,kn->n", a, b.data, optimize=True)
            if isinstance(b, np.ndarray) and isinstance(a, VecOpInfo) and b.ndim == 1:
                if a.role == "function":
                    u_vals = np.sum(a.data, axis=1)
                    return np.dot(b, u_vals)
                return np.einsum("k,kn->n", b, a.data, optimize=True)
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                if a.ndim == 1 and b.ndim == 1:
                    return float(np.einsum("k,k->", a, b, optimize=True))
                return np.tensordot(a, b, axes=([0], [0]))

            raise TypeError(f"Unsupported RHS inner '{n.a} : {n.b}' "
                            f"a={type(a)}, b={type(b)}, roles=({getattr(a,'role',None)},{getattr(b,'role',None)})"
                            f" and data shapes a={getattr(getattr(a, 'data', a), 'shape', None)}, b={getattr(getattr(b, 'data', b), 'shape', None)}")

        # ============================= LHS =============================
        # Orientation: rows = test space, cols = trial space

        # ---- Hessian LHS ----
        if isinstance(a, HessOpInfo) and isinstance(b, HessOpInfo):
            if a.role == "test" and b.role == "trial":
                return a.inner(b)  # (n_test, n_trial)
            if a.role == "trial" and b.role == "test":
                return b.inner(a)  # (n_test, n_trial)
            raise ValueError(f"Hessian LHS expects test vs trial; got {a.role} vs {b.role}.")

        # ---- Grad LHS ----
        if isinstance(a, GradOpInfo) and isinstance(b, GradOpInfo):
            if a.role == "test" and b.role == "trial":
                return a.inner(b)
            if a.role == "trial" and b.role == "test":
                return b.inner(a)
            raise ValueError(f"Grad LHS expects test vs trial; got {a.role} vs {b.role}.")

        # ---- Vec LHS (e.g., Laplacian(LHS) yields VecOpInfo) ----
        if isinstance(a, VecOpInfo) and isinstance(b, VecOpInfo):
            if a.role == "test" and b.role == "trial":
                return a.inner(b)
            if a.role == "trial" and b.role == "test":
                return b.inner(a)
            raise ValueError(f"Vec LHS expects test vs trial; got {a.role} vs {b.role}.")

        # ---- Numerical fallbacks ----
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            # generic numeric inner (use tensordot if needed)
            return float(np.einsum("..., ...->", a, b, optimize=True)) if a.ndim == b.ndim == 1 else np.tensordot(a, b, axes=([0], [0]))

        # Grad with numeric (complex LHS) supported above if needed; otherwise:
        if isinstance(a, GradOpInfo) and isinstance(b, np.ndarray):
            return a.contracted_with_tensor(b)
        if isinstance(b, GradOpInfo) and isinstance(a, np.ndarray):
            return b.contracted_with_tensor(a)

        raise TypeError(f"Unsupported inner product '{n.a} : {n.b}' "
                        f"for roles a={getattr(a, 'role', None)}, b={getattr(b, 'role', None)} "
                        f"and data shapes a={getattr(a, 'data', None)}, b={getattr(b, 'data', None)}")

    
        
    # Visitor dispatch
    def _visit(self, node): return self._dispatch[type(node)](node)

    def _visit_Analytic(self, node): return node.eval(self.ctx['x_phys'])

    # ======================= ASSEMBLY CORE ============================
    def _assemble_form(self, form, target):
        """Accept a Form object and iterate through its integrals."""
        from pycutfem.ufl.forms import Form
        # from pycutfem.ufl.measures import Integral

        # This guard clause correctly handles one-sided equations (e.g., L=None or a=None)
        if form is None or not isinstance(form, Form):
            return

        for integral in form.integrals:
            if not isinstance(integral, Integral):
                continue
            
            # --- The rest of the dispatching logic remains the same ---
            if integral.measure.domain_type == "interface":
                # Handle interface integrals with a level set
                logger.info(f"Assembling interface integral: {integral}")
                self._assemble_interface(integral, target)
                continue
            if integral.measure.domain_type == "volume":
                # Handle volume integrals
                logger.info(f"Assembling volume integral: {integral}")
                self._assemble_volume(integral, target)
                continue
            if integral.measure.domain_type == "ghost_edge":
                logger.info(f"Assembling ghost edge integral: {integral}")
                self._assemble_ghost_edge(integral, target)
                continue
            if integral.measure.domain_type == "exterior_facet":
                logger.info(f"Assembling exterior-facet integral: {integral}")
                self._assemble_boundary_edge(integral, target)
                continue

            logger.warning(f"Skipping unsupported integral type: {integral.measure.domain_type}")


    def _find_q_order(self, integral: Integral) -> int:
        md = integral.measure.metadata or {}

        # 1) explicit override
        for key in ('quad_degree', 'quad_order', 'q'):
            if key in md:
                return int(md[key])

        # 2) global fallback
        if self.qorder is not None:
            return int(self.qorder)

        # 3) auto + derivative-aware bump
        try:
            p = self.degree_estimator.estimate_degree(integral.integrand)
        except Exception as e:
            logger.warning(f'Could not estimate polynomial degree: {e}.')
            p = max(1, self.me.mesh.poly_order)   # conservative but not huge

        # highest total derivative order requested on this measure (if any)
        r = 0
        if 'derivs' in md and md['derivs']:
            r = max(int(ox) + int(oy) for (ox, oy) in md['derivs'])

        geom = max(0, self.me.mesh.poly_order - 1)

        # Gauss–Legendre exactness guideline in 1D:
        q_est   = max(1, math.ceil((p + 1) / 2))
        # Small geometry/derivative floor that matches what your JIT needed (e.g. r=2, geom=1 → 6)
        q_floor = max(1, 2*geom + 2*r)

        return max(q_est, q_floor)

    
    def _hook_for(self, expr):
        hooks = self.ctx.get("hooks", {})
        # print(f"expr={expr}, hooks={hooks}")
        if expr is None:
            return hooks.get(Function)           # default
        if expr in hooks:                        # ← object hash & equality!
            return hooks[expr]
        return hooks.get(type(expr))             # fallback by type
    
    # ====================== BC handling ===============================
    def _apply_bcs(self, K, F, bcs):
        # This method remains unchanged from your provided file
        if not bcs: return
        data = self.dh.get_dirichlet_data(bcs)
        if not data: return
        rows = np.fromiter(data.keys(), dtype=int)
        vals = np.fromiter(data.values(), dtype=float)
        
        # Apply to RHS vector F
        F -= K @ np.bincount(rows, weights=vals, minlength=F.size)
        
        # Zero out rows and columns in the matrix
        K_lil = K.tolil()
        K_lil[rows, :] = 0
        K_lil[:, rows] = 0
        K_lil[rows, rows] = 1.0
        # copy the edited data back
        K[:] = K_lil  # Convert back to CSR format
        
        # Set values in the RHS vector
        F[rows] = vals
    
    def _assemble_volume(self, integral: Integral, matvec):
        # if a level-set was attached to dx → do cut-cell assembly in Python path
        if getattr(integral.measure, "level_set", None) is not None:
            # CUT-VOLUME integral
            if self.backend == "jit":
                self._assemble_volume_cut_jit(integral, matvec)
            else:
                # existing pure-Python path (already working from earlier work)
                self._assemble_volume_cut_python(integral, matvec)
            return
        # … otherwise the existing back-end selection …
        if self.backend == "python":
            self._assemble_volume_python(integral, matvec)
        elif self.backend == "jit":
            self._assemble_volume_jit(integral, matvec)
        else:
            raise ValueError("Unsupported backend.")
    
    # ----------------------------------------------------------------------
    def _expr_requires_hess(self, node):
        """
        Return True iff the expression tree contains a Hessian or Laplacian node.

        Robust DFS that:
        - visits each node once (cycle-safe),
        - follows common child attributes ('operand', 'a', 'b', 'A', 'args', 'children'),
        - only iterates over true containers (list/tuple/set),
        - never tries to iterate arbitrary objects like Pos/Neg.
        """

        stack = [node]
        seen  = set()

        while stack:
            x = stack.pop()
            if x is None:
                continue
            xid = id(x)
            if xid in seen:
                continue
            seen.add(xid)

            if isinstance(x, (Hessian, Laplacian)):
                return True

            # Follow standard single-child attributes (covers Pos/Neg, unary ops, etc.)
            for attr in ("operand", "a", "b", "A", "args", "children"):
                child = getattr(x, attr, None)
                if child is None:
                    continue
                # If the attribute is a container, push each element; else push the single child
                if isinstance(child, (list, tuple, set)):
                    for y in child:
                        if y is not None:
                            stack.append(y)
                else:
                    stack.append(child)

            # If the node itself is a container, visit its items
            if isinstance(x, (list, tuple, set)):
                for y in x:
                    if y is not None:
                        stack.append(y)

        return False

    def _assemble_volume_jit(self, integral: Integral, matvec):
        """
        Assembles a volume integral using the JIT backend, correctly handling
        element subsets and ensuring argument lists match the specific integral.
        """
        dbg = os.getenv("PYCUTFEM_JIT_DEBUG", "").lower() in {"1", "true", "yes"}
        mesh = self.me.mesh

        # 1. Determine the exact subset of elements for this integral.
        defined_on = integral.measure.defined_on
        if defined_on is not None:
            element_ids = np.asarray(defined_on.to_indices(), dtype=np.int32) if hasattr(defined_on, "to_indices") else np.flatnonzero(np.asarray(defined_on)).astype(np.int32)
        else:
            element_ids = np.arange(mesh.n_elements, dtype=np.int32)

        if element_ids.size == 0:
            return # Nothing to assemble.

        # 2. Collect the up-to-date coefficient functions from the expression FIRST.
        # This aligns with the calling sequence of other robust JIT assemblers.
        current_funcs = self._get_data_functions_objs(integral)

        # 3. Compile the backend for the CURRENT integral to get the specific runner and IR.
        # This ensures we have the correct param_order for this exact expression.
        runner, ir = self._compile_backend(
            integral.integrand,
            self.dh, self.me
        )
        derivs, need_hess, need_o3, need_o4 = self._find_req_derivs(integral, runner)


        # 4. Build the static arguments required by this specific kernel.
        # We build these fresh every time to avoid caching collisions.
        q_order = self._find_q_order(integral)

        # (A) Get full-mesh geometric factors and then SLICE them for the subset.
        geo_args_all = self.dh.precompute_geometric_factors(q_order, 
                                                            need_hess=need_hess, need_o3=need_o3, need_o4=need_o4)
        pre_built = {k: (v[element_ids] if isinstance(v, np.ndarray) else v) for k, v in geo_args_all.items()}
        pre_built["entity_kind"] = "element"
        pre_built["is_interface"] = False
        pre_built["owner_id"]    = geo_args_all.get("owner_id", geo_args_all.get("eids"))[element_ids]


        # (B) Build the DOF map for the subset.
        gdofs_map = np.vstack([
            self.dh.get_elemental_dofs(e) for e in element_ids
        ]).astype(np.int32)

        # (C) Build all other required tables (basis, etc.) based on the correct param_order.
        basis_args = _build_jit_kernel_args(
            ir, integral.integrand, self.me, q_order,
            dof_handler=self.dh,
            gdofs_map=gdofs_map,
            param_order=runner.param_order, # Use the param_order from the CURRENT runner
            pre_built=pre_built
        )

        # (D) Finalize the dictionary of static arguments for the kernel runner.
        static_args = {
            "gdofs_map":  gdofs_map,
            "node_coords": self.dh.get_all_dof_coords(),
            **pre_built,
            **basis_args,
        }

        # 5. Execute the kernel via the runner.
        K_loc, F_loc, J_loc = runner(current_funcs, static_args)
        if dbg:
            print(f"[Assembler] kernel returned K_loc {K_loc.shape}, F_loc {F_loc.shape}")

        # 6. Handle scalar functional result or scatter the element contributions.
        hook = self._hook_for(integral.integrand)
        if self._functional_calculate(integral, J_loc, hook):
            return

        _scatter_element_contribs(
            K_loc, F_loc, J_loc,
            element_ids=element_ids,
            gdofs_map=gdofs_map,
            matvec=matvec,
            ctx=self.ctx,
            integrand=integral.integrand,
            hook=hook
        )
        



    def _assemble_volume_python(self, integral: Integral, matvec):
        mesh = self.me.mesh
        # Use a higher quadrature order for safety, esp. with mixed orders
        q_order = self._find_q_order(integral)
        qp, qw = volume(mesh.element_type, q_order)
        fields = _all_fields(integral.integrand)
        rhs = self.ctx["rhs"]

        for eid, element in enumerate(mesh.elements_list):
            if rhs:
                loc = np.zeros(self.me.n_dofs_local)
            else:
                loc = np.zeros((self.me.n_dofs_local, self.me.n_dofs_local))
            
            for (xi, eta), w in zip(qp, qw):
                J = transform.jacobian(mesh, eid, (xi, eta))
                detJ = abs(np.linalg.det(J))
                Ji = np.linalg.inv(J)
                self.ctx['x_phys'] = transform.x_mapping(mesh, eid, (xi, eta))
                
                # Cache basis values and gradients for this quadrature point
                self._basis_cache.clear();self._coeff_cache.clear(); self._collapsed_cache.clear();
                for f in fields:
                    val = self.me.basis(f, xi, eta)
                    g_ref = self.me.grad_basis(f, xi, eta)
                    self._basis_cache[f] = {"val": val, "grad": g_ref @ Ji}
                
                self.ctx["eid"] = eid
                integrand_val = self._visit(integral.integrand)
                loc += w * detJ * integrand_val
            
            gdofs = self.dh.get_elemental_dofs(eid)
            if rhs:
                np.add.at(matvec, gdofs, loc)
            else:
                # Efficiently add local matrix to sparse global matrix
                r, c = np.meshgrid(gdofs, gdofs, indexing="ij")
                matvec[r, c] += loc
        
        self.ctx.pop("eid", None); self.ctx.pop("x_phys", None)  # Clean up context

    
    
    def _assemble_interface_python(self, intg: Integral, matvec):
        """
        Assemble interface integrals (pure Python) with component/side-aware padding.

        Strategy
        --------
        • For each cut element with a valid segment, treat it as the union owner:
            pos_eid = neg_eid = eid
        • Build per-field maps from local field DOFs → union (global_dofs for the element).
        • (Optional) Build per-field side masks from φ(DOF coords): '+' if φ>=-tol else '−'.
        • For each interface quadrature point:
            - prime side-aware context keys (basis_values cache, maps, masks, normal, etc.)
            - evaluate integrand; `_basis_row` will pad/mask appropriately
            - accumulate into local vector/matrix or functional
        • Scatter local contributions to the global matvec.
        """

        log = logging.getLogger(__name__)
        log.info(f"Assembling interface integral: {intg}")

        rhs = bool(self.ctx.get('rhs', False))
        level_set = intg.measure.level_set
        if level_set is None:
            raise ValueError("dInterface measure requires a level_set.")

        # Mark that we are on Γ so Pos/Neg visitors set ctx['side'] & select (pos|neg)_eid
        self.ctx['on_interface'] = True

        mesh = self.me.mesh
        qdeg = self._find_q_order(intg)
        fields = _all_fields(intg.integrand)

        trial, test = _trial_test(intg.integrand)
        hook = self._hook_for(intg.integrand)
        is_functional = (hook is not None) and not (trial or test)
        if is_functional:
            self.ctx.setdefault('scalar_results', {}).setdefault(hook['name'], 0.0)

        try:
            for elem in mesh.elements_list:
                # Only elements that are actually cut and have a valid interface segment
                if getattr(elem, 'tag', None) != 'cut':
                    continue
                seg = getattr(elem, 'interface_pts', None)
                if not (isinstance(seg, (list, tuple)) and len(seg) == 2):
                    continue

                eid = int(elem.id)
                self.ctx['eid'] = eid
                self.ctx['pos_eid'] = eid
                self.ctx['neg_eid'] = eid

                # Union dofs for this element (order provided by DofHandler)
                global_dofs = np.asarray(self.dh.get_elemental_dofs(eid), dtype=int)

                # Per-field padding maps (field-local → union index), robust to unsorted union
                pos_map_by_field, neg_map_by_field = _hfa.build_field_union_maps(
                    self.dh, fields, eid, eid, global_dofs
                )

                # Optional: per-field side masks (+/−) based on φ at DOF coordinates
                try:
                    pos_mask_by_field, neg_mask_by_field = _hfa.build_side_masks_by_field(
                        self.dh, fields, eid, level_set, tol=0.0
                    )
                except Exception:
                    pos_mask_by_field, neg_mask_by_field = {}, {}

                # Quadrature on the physical interface segment
                p0, p1 = elem.interface_pts
                qpts, qwts = line_quadrature(p0, p1, qdeg)

                # Local accumulator in union layout (or scalar for functional)
                if is_functional:
                    acc = 0.0
                else:
                    n = len(global_dofs)
                    acc = np.zeros(n, dtype=float) if rhs else np.zeros((n, n), dtype=float)

                # --- Quadrature loop on Γ_e
                for xq, w in zip(qpts, qwts):
                    # Side-aware, per-QP cache + context priming
                    self.ctx['x_phys'] = xq
                    # On Γ; let Pos/Neg set ctx['side'], but keep phi_val ~ 0 for any gating
                    self.ctx['phi_val'] = 0.0
                    g = level_set.gradient(xq)
                    self.ctx['normal'] = g / (np.linalg.norm(g) + 1e-30)
                    self.ctx['basis_values'] = {}  # per-QP, per-side cache for _basis_row

                    if not is_functional:
                        self.ctx['global_dofs'] = global_dofs
                        # identity maps (generic fallback); per-field maps are primary
                        self.ctx['pos_map'] = np.arange(len(global_dofs), dtype=int)
                        self.ctx['neg_map'] = self.ctx['pos_map']
                        # per-field maps (preferred)
                        self.ctx['pos_map_by_field'] = pos_map_by_field
                        self.ctx['neg_map_by_field'] = neg_map_by_field
                        # masks applied inside _basis_row when present
                        self.ctx['pos_mask_by_field'] = pos_mask_by_field
                        self.ctx['neg_mask_by_field'] = neg_mask_by_field

                    # Evaluate integrand; visitors will respect Pos/Neg and padding
                    val = self._visit(intg.integrand)

                    # Accumulate
                    if is_functional:
                        arr = np.asarray(val)
                        acc += w * float(arr if arr.ndim == 0 else arr.sum())
                    else:
                        acc += w * np.asarray(val)

                # --- Scatter to global structures
                if is_functional:
                    self.ctx['scalar_results'][hook['name']] += float(acc)
                else:
                    if rhs:
                        # Vector scatter
                        np.add.at(matvec, global_dofs, acc)
                    else:
                        # Matrix scatter
                        rr, cc = np.meshgrid(global_dofs, global_dofs, indexing='ij')
                        matvec[rr, cc] += acc

        finally:
            # Context cleanup
            for k in (
                "basis_values", "global_dofs",
                "pos_map", "neg_map",
                "pos_map_by_field", "neg_map_by_field",
                "pos_mask_by_field", "neg_mask_by_field",
                "eid", "pos_eid", "neg_eid",
                "on_interface", "x_phys", "phi_val", "normal"
            ):
                self.ctx.pop(k, None)

    
    def _phys_scalar_deriv_row(self, fld: str, xi: float, eta: float,
                           ox: int, oy: int, eid: int) -> np.ndarray:
        """
        Return a length-(nloc_elem) row for D^{(ox,oy)} of scalar field 'fld' in physical coords.
        Exact for orders 0, 1, 2 (includes inverse-map curvature). Uses cached exact formulas for
        orders 3 and 4 (f_{ijk}, f_{ijkl}). Falls back to binomial (no curvature) for higher orders.
        """
        me = self.me

        # --- ensure we have per-object caches we can reuse across calls
        if not hasattr(self, "_ref_cache"):
            self._ref_cache = transform.RefDerivCache(me)
        ref_cache = self._ref_cache


        # order 0: just values in reference
        if ox == 0 and oy == 0:
            return ref_cache.get(fld, xi, eta, 0, 0)

        # pick how far we need geometry jets
        total = ox + oy
        upto = 2 if total <= 2 else (3 if total == 3 else (4 if total == 4 else 2))
        rec = transform.JET_CACHE.get(me.mesh, eid, xi, eta, upto=upto)
        A = rec["A"]  # (2,2)

        # order 1: ∇_x = ∇_X A
        if total == 1:
            dN_dxi  = ref_cache.get(fld, xi, eta, 1, 0)
            dN_deta = ref_cache.get(fld, xi, eta, 0, 1)
            if (ox, oy) == (1, 0):  # ∂/∂x
                return dN_dxi * A[0, 0] + dN_deta * A[1, 0]
            else:                   # (0,1) ∂/∂y
                return dN_dxi * A[0, 1] + dN_deta * A[1, 1]

        # order 2: exact with curvature  (uses A2 = ∂^2 X / ∂x^2)
        if total == 2:
            A2 = rec["A2"]  # shape (2,2,2) with A2[I,i,j] = A^I_{ij}
            # reference derivatives
            dN_xi     = ref_cache.get(fld, xi, eta, 1, 0)
            dN_eta    = ref_cache.get(fld, xi, eta, 0, 1)
            dN_xixi   = ref_cache.get(fld, xi, eta, 2, 0)
            dN_xieta  = ref_cache.get(fld, xi, eta, 1, 1)
            dN_etaeta = ref_cache.get(fld, xi, eta, 0, 2)

            def second(i: int, j: int):
                # (H_X(w) pulled back) + curvature term (grad_X w ⋅ A_{,ij})
                Axi_i, Aeta_i = A[0, i], A[1, i]
                Axi_j, Aeta_j = A[0, j], A[1, j]
                hess_pull = (dN_xixi   * (Axi_i * Axi_j)
                        + dN_xieta  * (Axi_i * Aeta_j + Aeta_i * Axi_j)
                        + dN_etaeta * (Aeta_i * Aeta_j))
                curv = dN_xi * A2[0, i, j] + dN_eta * A2[1, i, j]
                return hess_pull + curv

            if   (ox, oy) == (2, 0):  # ∂²/∂x²
                return second(0, 0)
            elif (ox, oy) == (1, 1):  # ∂²/∂x∂y
                return second(0, 1)
            else:                     # (0,2): ∂²/∂y²
                return second(1, 1)

        # order 3: exact (cached) f_{ijk}
        if total == 3:
            i, j, k = _as_indices(ox, oy)
            return phys_scalar_third_row(me, fld, xi, eta, i, j, k, me.mesh, eid, ref_cache)

        # order 4: exact (cached) f_{ijkl}
        if total == 4:
            i, j, k, l = _as_indices(ox, oy)
            return phys_scalar_fourth_row(me, fld, xi, eta, i, j, k, l, me.mesh, eid, ref_cache)

        # fallback (>=5): pure binomial chain rule (no A-derivative curvature terms)
        out = np.zeros(me.n_dofs_local, dtype=float)
        A11, A12, A21, A22 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
        ktot = total
        for i in range(ox + 1):
            cx = comb(ox, i) * (A11 ** i) * (A21 ** (ox - i))
            for j in range(oy + 1):
                cy = comb(oy, j) * (A12 ** j) * (A22 ** (oy - j))
                orx = i + j
                ory = ktot - orx
                try:
                    out += (cx * cy) * ref_cache.get(fld, xi, eta, orx, ory)
                except Exception:
                    pass
        return out
    
    def _assemble_ghost_edge_python(self, intg: "Integral", matvec):
        """
        Assemble integrals over ghost facets (pure Python path).

        Key points:
        • Pads (+) and (–) traces into the union DOF set of the two owner elements.
        • Guarantees value + first-derivative entries in the basis cache.
        • Orients the facet normal from (–) to (+).
        • Supports scalar-valued functionals via assembler hooks.
        """

        rhs   = self.ctx.get("rhs", False)
        mesh  = self.me.mesh

        # ---- derivative orders we need (robust) -------------------------
        md      = intg.measure.metadata or {}
        derivs  = set(md.get("derivs", set())) | set(required_multi_indices(intg.integrand))
        derivs |= {(0, 0), (1, 0), (0, 1)}  # always have value and ∂x, ∂y

        # Close up to max total order (covers Hessian cross-terms etc.)
        max_total = max((ox + oy for (ox, oy) in derivs), default=0)
        for p in range(max_total + 1):
            for ox in range(p + 1):
                derivs.add((ox, p - ox))

        # ---- which ghost edges ------------------------------------------
        defined = intg.measure.defined_on
        if defined is None:
            edge_set = mesh.edge_bitset("ghost")
            edge_ids = edge_set.to_indices() if hasattr(edge_set, "to_indices") else list(edge_set)
        else:
            edge_ids = defined.to_indices()

        # level set required to choose (+)/(–)
        level_set = getattr(intg.measure, "level_set", None)
        if level_set is None:
            raise ValueError("dGhost measure requires a 'level_set' callable.")

        # Quadrature degree: safe upper bound
        qdeg = max(self._find_q_order(intg), 2 * max_total + 4)

        # Fields needed (fallbacks if tree-walk misses some due to Jump/Pos/Neg)
        fields = set(_all_fields(intg.integrand))
        trial, test = _trial_test(intg.integrand)
        # Add from trial/test if missed
        for tt in (trial, test):
            if tt is None:
                continue
            if hasattr(tt, "field_names"): fields.update(tt.field_names)
            elif hasattr(tt, "field_name"): fields.add(tt.field_name)
        if not fields:
            # last ditch: include all handler fields
            fields = set(getattr(self.dh, "field_names", []))
        fields = sorted(fields)

        # Functional hook support (scalar accumulation)
        hook = self._hook_for(intg.integrand)
        is_functional = (hook is not None and trial is None and test is None)
        if is_functional:
            self.ctx.setdefault("scalar_results", {}).setdefault(hook["name"], 0.0)

        # ------------------------ main loop over ghost edges ------------------------
        try:
            for eid_edge in map(int, edge_ids):
                e = mesh.edge(eid_edge)
                # Ghost facets must be interior (two neighbors)
                if e.right is None:
                    continue

                # Narrow band or explicitly tagged as ghost
                lt = mesh.elements_list[e.left].tag
                rt = mesh.elements_list[e.right].tag
                etag = str(getattr(e, "tag", ""))
                if not (("cut" in (lt, rt)) or etag.startswith("ghost")):
                    continue

                # Choose (+)/(–) by phi at centroids: pos = higher φ, neg = lower φ
                phiL = float(level_set(np.asarray(mesh.elements_list[e.left ].centroid())))
                phiR = float(level_set(np.asarray(mesh.elements_list[e.right].centroid())))
                pos_eid, neg_eid = (e.left, e.right) if phiL >= phiR else (e.right, e.left)

                # Edge quadrature (physical) and oriented normal (– → +)
                p0, p1 = mesh.nodes_x_y_pos[list(e.nodes)]
                qpts_phys, qwts = line_quadrature(p0, p1, qdeg)

                normal_vec = e.normal.copy()
                cpos = np.asarray(mesh.elements_list[pos_eid].centroid())
                cneg = np.asarray(mesh.elements_list[neg_eid].centroid())
                if np.dot(normal_vec, cpos - cneg) < 0.0:
                    normal_vec = -normal_vec

                # Local union DOFs / accumulator
                if is_functional:
                    loc_acc = 0.0
                    global_dofs = pos_map = neg_map = None
                else:
                    pos_dofs = self.dh.get_elemental_dofs(pos_eid)
                    neg_dofs = self.dh.get_elemental_dofs(neg_eid)
                    global_dofs = np.unique(np.concatenate([pos_dofs, neg_dofs]))
                    pos_map = np.searchsorted(global_dofs, pos_dofs)
                    neg_map = np.searchsorted(global_dofs, neg_dofs)
                    # NEW: field-specific maps (for gradient/value padding)
                    pos_map_by_field, neg_map_by_field = _hfa.build_field_union_maps(
                        self.dh, fields, pos_eid, neg_eid, global_dofs
                    )
                    loc_acc = (np.zeros(len(global_dofs))
                               if rhs else np.zeros((len(global_dofs), len(global_dofs))))

                # ---------------- QP loop ----------------
                for xq, w in zip(qpts_phys, qwts):
                    # Side-aware store that _basis_row() will lazily populate:
                    bv = {"+": defaultdict(dict), "-": defaultdict(dict)}

                    # Context for visitors (enough info for _basis_row to compute & pad)
                    self.ctx.update({
                        "basis_values": bv,          # per-QP, side-aware derivative cache
                        "normal": normal_vec,
                        "phi_val": level_set(xq),    # used if 'side' isn't set explicitly
                        "x_phys": xq,
                        "global_dofs": (global_dofs if not is_functional else None),
                        "pos_map": (pos_map if not is_functional else None),
                        "neg_map": (neg_map if not is_functional else None),
                        "pos_map_by_field": (pos_map_by_field if not is_functional else None),
                        "neg_map_by_field": (neg_map_by_field if not is_functional else None),
                        "pos_eid": pos_eid,
                        "neg_eid": neg_eid
                    })
                    if not is_functional:
                        self.ctx.update({
                            "global_dofs": global_dofs,
                            "pos_map": pos_map, "neg_map": neg_map,
                            "pos_map_by_field": pos_map_by_field,
                            "neg_map_by_field": neg_map_by_field,
                        })

                    # Evaluate integrand and accumulate
                    val = self._visit(intg.integrand)
                    if is_functional:
                        # Reduce to scalar robustly
                        if isinstance(val, VecOpInfo):
                            v = float(np.sum(val.data))
                        else:
                            arr = np.asarray(val)
                            v = float(arr if arr.ndim == 0 else arr.sum())
                        loc_acc += w * v
                    else:
                        loc_acc += w * np.asarray(val)

                # Scatter
                if is_functional:
                    self.ctx["scalar_results"][hook["name"]] += loc_acc
                else:
                    if rhs:
                        np.add.at(matvec, global_dofs, loc_acc)
                    else:
                        r, c = np.meshgrid(global_dofs, global_dofs, indexing="ij")
                        matvec[r, c] += loc_acc

        finally:
            # Clean context keys we may have set
            for k in ("basis_values", "normal", "phi_val", "x_phys",
                    "global_dofs", "pos_map", "neg_map", "pos_eid", "neg_eid",
                    "pos_map_by_field", "neg_map_by_field", "on_interface"):
                self.ctx.pop(k, None)


    

    def _assemble_interface_jit(self, intg, matvec):
        """
        Assemble interface terms (∫_Γ …) with the JIT back‑end **using
        cut‑only tables**.

        All geometry/basis tables coming from
        `DofHandler.precompute_interface_factors` are sized
        ``n_cut_elems × …`` and contain *only* the rows that correspond to
        the global element ids listed in ``geo["eids"]``.  We therefore:

        1. build a DOF‑map of the same length,
        2. feed those aligned arrays to the kernel, and
        3. scatter the kernel output with the matching global‑id list.
        """
        dh, me  = self.dh, self.me
        mesh    = me.mesh

        # ------------------------------------------------------------------
        # 1. Which elements carry an interface?  (BitSet → array of ids)
        # ------------------------------------------------------------------
        cut_bs = (intg.measure.defined_on
                if intg.measure.defined_on is not None
                else mesh.element_bitset("cut"))

        if cut_bs.cardinality() == 0:          # nothing to do
            return

        # ------------------------------------------------------------------
        # 2. Pre‑compute geometry for the interface elements only
        # ------------------------------------------------------------------
        qdeg      = self._find_q_order(intg)
        level_set = intg.measure.level_set
        if level_set is None:
            raise ValueError("dInterface measure requires a level_set.")

        
        runner, ir = self._compile_backend(intg.integrand, dh, me)
        derivs, need_hess, need_o3, need_o4 = self._find_req_derivs(intg, runner)
        geo = dh.precompute_interface_factors(cut_bs, qdeg, level_set, 
                                              need_hess=need_hess, need_o3=need_o3, need_o4=need_o4)
        geo["is_interface"] = True
        cut_eids = geo["eids"].astype(np.int32)      # 1‑D array, len = n_cut

        # 2a) ALIAS: for interface, both sides are the same element.
        #    Provide r**_*_{pos|neg} by aliasing to existing d**_* tables.
        fields = dh.mixed_element.field_names
        for fld in fields:
            gkey = f"g_{fld}"
            if gkey not in geo:
                continue
            g = geo[gkey]                         # (nE, nQ, n_union, 2) = [d/dξ, d/dη]
            r10 = np.ascontiguousarray(g[..., 0]) # (nE, nQ, n_union)
            r01 = np.ascontiguousarray(g[..., 1]) # (nE, nQ, n_union)
            # Only set if absent
            geo.setdefault(f"r10_{fld}_pos", r10)
            geo.setdefault(f"r01_{fld}_pos", r01)
            geo.setdefault(f"r10_{fld}_neg", r10)
            geo.setdefault(f"r01_{fld}_neg", r01)

        # ------------------------------------------------------------------
        # 3. Element‑to‑DOF map  (shape = n_cut × n_loc)
        # ------------------------------------------------------------------
        gdofs_map = np.vstack(
            [dh.get_elemental_dofs(eid) for eid in cut_eids]
        ).astype(np.int32)
        geo["gdofs_map"] = gdofs_map
        geo["entity_kind"] = "element"

        # ------------------------------------------------------------------
        # 4. Gather coefficient Functions once
        # ------------------------------------------------------------------
        current_funcs = self._get_data_functions_objs(intg)

        # ------------------------------------------------------------------
        # 5. Compile kernel & build static argument dict
        # ------------------------------------------------------------------

        

        basis_args = _build_jit_kernel_args(
            ir, intg.integrand, me, qdeg,
            dof_handler = dh,
            gdofs_map   = gdofs_map,
            param_order = runner.param_order,
            pre_built   = geo
        )

        static_args = {k: v for k, v in geo.items() if k != "eids"}
        static_args.update(basis_args)

        # ------------------------------------------------------------------
        # 6. Execute kernel  → element buffers K/F/J
        # ------------------------------------------------------------------
        K_cut, F_cut, J_cut = runner(current_funcs, static_args)

        # ------------------------------------------------------------------
        # 7. Scatter the contributions from the cut elements
        # ------------------------------------------------------------------
        hook = self._hook_for(intg.integrand)
        if self._functional_calculate(intg, J_cut, hook): return

        _scatter_element_contribs(
            K_cut, F_cut, J_cut,
            cut_eids, gdofs_map,
            matvec, self.ctx, intg.integrand,
            hook = hook,          # scalar‑functional hook, if any
        )


    def _assemble_interface(self, intg: Integral, matvec):
        """Assemble integrals over non-conforming interfaces defined by a level set."""
        if self.backend == "python":
            self._assemble_interface_python(intg, matvec)
        elif self.backend == "jit":
            self._assemble_interface_jit(intg, matvec)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}. Use 'python' or 'jit'.")

    def _find_req_derivs(self, intg: "Integral", runner):
        """Find all required derivative multi-indices for the given integral."""
        md = intg.measure.metadata or {}
        derivs = set(md.get("derivs", set())) | set(required_multi_indices(intg.integrand))
        derivs |= {(0, 0)}
        max_total = max((ox + oy for (ox, oy) in derivs), default=0)
        for p in range(max_total + 1):
            for ox in range(p + 1):
                derivs.add((ox, p - ox))
        req_list = list(getattr(runner, "param_order", []))
        req = set(req_list)

        # Which jets are actually required?
        need_hess = any(k in req for k in
                        ("Hxi0","Hxi1","pos_Hxi0","pos_Hxi1","neg_Hxi0","neg_Hxi1")) \
                    or self._expr_requires_hess(intg.integrand)
        need_o3   = any(k in req for k in
                        ("Txi0","Txi1","pos_Txi0","pos_Txi1","neg_Txi0","neg_Txi1"))
        need_o4   = any(k in req for k in
                        ("Qxi0","Qxi1","pos_Qxi0","pos_Qxi1","neg_Qxi0","neg_Qxi1"))

        # Ensure deriv tables include everything the chain rule can touch
        if need_hess:
            derivs |= {(1,0),(0,1),(2,0),(1,1),(0,2)}
        if need_o3:
            derivs |= {(3,0),(2,1),(1,2),(0,3)}
        if need_o4:
            derivs |= {(4,0),(3,1),(2,2),(1,3),(0,4)}

        return derivs, need_hess, need_o3, need_o4

    def _assemble_ghost_edge_jit(self, intg: "Integral", matvec):
        """Assembles ghost-edge integrals using the JIT backend."""


        mesh = self.me.mesh
        dh, me = self.dh, self.me


        edge_ids = (intg.measure.defined_on
                    if intg.measure.defined_on is not None
                    else mesh.edge_bitset('ghost'))
        level_set = getattr(intg.measure, 'level_set', None)
        if level_set is None:
            raise ValueError("dGhost measure requires a 'level_set' callable.")

        if edge_ids.cardinality() == 0:
            raise ValueError("No ghost edges found for the integral.")

        # 2) Compile kernel FIRST (to know exact static params it will require)
        runner, ir = self._compile_backend(intg.integrand, dh, me)
        derivs, need_hess, need_o3, need_o4 = self._find_req_derivs(intg, runner)

        max_total = max((ox + oy for (ox, oy) in derivs), default=0)
        qdeg = max(self._find_q_order(intg), 2*max_total + 4)

        # 3) Precompute sided ghost factors with the exact flags
        geo = self.dh.precompute_ghost_factors(
            ghost_edge_ids=edge_ids,
            qdeg=qdeg,
            level_set=level_set,
            derivs=derivs,
            need_hess=need_hess,
            need_o3=need_o3,
            need_o4=need_o4,
            reuse=True,
        )
        geo["is_interface"] = False

        valid_eids = geo.get("eids")
        if valid_eids is None or len(valid_eids) == 0:
            return

        # 4) Build static kernel args from what we precomputed
        kernel_args = _build_jit_kernel_args(
            ir=ir,
            expression=intg.integrand,
            mixed_element=me,
            q_order=qdeg,
            dof_handler=dh,
            gdofs_map=geo["gdofs_map"],
            param_order=runner.param_order,
            pre_built=geo,
        )

        # ---- baseline statics some kernels always expect ----
        if ("node_coords" in runner.param_order) and ("node_coords" not in kernel_args):
            kernel_args["node_coords"] = np.asarray(mesh.nodes_x_y_pos, dtype=float)
        if ("element_nodes" in runner.param_order) and ("element_nodes" not in kernel_args):
            kernel_args["element_nodes"] = np.asarray(mesh.elements_connectivity, dtype=np.int64)

        # 4b) Patch unsided dXY_<fld> if requested but missing: average of sided rXY
        missing_now = [p for p in runner.param_order if p not in kernel_args]
        if missing_now:
            for name in list(missing_now):
                m = re.match(r"^d(\d)(\d)_([A-Za-z0-9_]+)$", name)
                if not m:
                    continue
                dx, dy, fld = int(m.group(1)), int(m.group(2)), m.group(3)
                kpos = f"r{dx}{dy}_{fld}_pos"
                kneg = f"r{dx}{dy}_{fld}_neg"
                if (kpos in kernel_args) and (kneg in kernel_args):
                    kernel_args[name] = 0.5*(kernel_args[kpos] + kernel_args[kneg])
                elif kpos in kernel_args:
                    kernel_args[name] = kernel_args[kpos]
                elif kneg in kernel_args:
                    kernel_args[name] = kernel_args[kneg]

        # 4c) Alias sided jets from unsided if necessary (belt-and-suspenders)
        req_list = list(getattr(runner, "param_order", []))
        req = set(req_list)
        for side in ("pos","neg"):
            for t in ("Hxi0","Hxi1","Txi0","Txi1","Qxi0","Qxi1"):
                need_key = f"{side}_{t}"
                if (need_key in req) and (need_key not in kernel_args) and (t in geo):
                    kernel_args[need_key] = geo[t]

        # Final sanity: everything the kernel listed must be present now
        still_missing = [p for p in runner.param_order if p not in kernel_args]
        if still_missing:
            raise KeyError(
                "KernelRunner: the following static arrays are still missing after automatic completion: "
                f"{still_missing}. Check that precompute_ghost_factors(...) produced them; "
                "if not, ensure need_hess/need_o3/need_o4 and derivs were computed from runner.param_order."
            )

        # 5) Gather current coefficient Functions and run the kernel
        current_funcs = self._get_data_functions_objs(intg)
        K_edge, F_edge, J_edge = runner(current_funcs, kernel_args)

        hook = self._hook_for(intg.integrand)
        if self._functional_calculate(intg, J_edge, hook):
            return

        # 6) Scatter contributions (per-edge union DOFs)
        _scatter_element_contribs(
            K_edge, F_edge, J_edge,
            valid_eids,
            geo["gdofs_map"],
            matvec, self.ctx, intg.integrand,
            hook=hook,
        )



    
    def _assemble_ghost_edge(self, intg: "Integral", matvec):
        """Assemble ghost edge integrals."""
        if self.backend == "python":
            self._assemble_ghost_edge_python(intg, matvec)
        elif self.backend == "jit":
            self._assemble_ghost_edge_jit(intg, matvec)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}. Use 'python' or 'jit'.")


    # inside FormCompiler -------------------------------------------------
    def _assemble_boundary_edge_python(self, intg: Integral, matvec):
        """
        Assemble ∫_Γ  f  dS   over a *set* of boundary edges  Γ  (Neumann BC,
        surface traction in elasticity, etc.).  Works for bilinear, linear
        and scalar functional forms – scalar forms can be captured with the
        same “assembler_hooks” mechanism used for interfaces / ghost edges.
        """
        if self.backend == "jit":
            raise NotImplementedError("dS + JIT not wired yet")

        rhs   = self.ctx.get("rhs", False)
        mesh  = self.me.mesh
        qdeg  = self._find_q_order(intg)
        hook  = self._hook_for(intg.integrand)
        trial, test = _trial_test(intg.integrand)
        is_functional = hook and trial is None and test is None
        if is_functional:
            self.ctx.setdefault("scalar_results", {})[hook["name"]] = 0.0

        # decide which edges we visit -------------------------------------
        defined = intg.measure.defined_on
        edge_ids = (defined.to_indices() if defined is not None
                    else np.fromiter((e.right is None for e in mesh.edges_list), bool).nonzero()[0])

        fields = _all_fields(intg.integrand)

        if edge_ids.size == 0: raise ValueError(
            f"Integral {intg} has no defined edges. "
            "Check the measure or the mesh.")
        for gid in edge_ids:
            edge   = mesh.edge(gid)
            eid    = edge.left                      # the unique owner element
            p0, p1 = mesh.nodes_x_y_pos[list(edge.nodes)]
            qp, qw = line_quadrature(p0, p1, qdeg)  # *weights already in phys. space*

            # outward normal of the *element*:
            n = edge.normal.copy()
            # if np.dot(n, np.asarray(mesh.elements_list[eid].centroid()) - qp[0]) < 0:
            #     n *= -1.0
            self.ctx["normal"] = n                  # constant along that edge

            # precompute element dofs once
            gdofs = self.dh.get_elemental_dofs(eid)

            # ------------------------------------------------------------------
            loc = None
            for x_phys, w in zip(qp, qw):
                xi, eta = transform.inverse_mapping(mesh, eid, x_phys)
                Ji    = np.linalg.inv(transform.jacobian(mesh, eid, (xi, eta)))

                # basis cache ---------------------------------------------------
                self._basis_cache.clear();self._coeff_cache.clear(); self._collapsed_cache.clear();
                for f in fields:
                    self._basis_cache[f] = {
                        "val" : self.me.basis      (f, xi, eta),
                        "grad": self.me.grad_basis (f, xi, eta) @ Ji
                    }

                # context for visitors
                self.ctx.update({"eid": eid, "x_phys": x_phys})

                val = self._visit(intg.integrand)    # scalar, vec, or matrix
                loc = (w * np.asarray(val)           # **NO detJ re-multiplication!**
                    if loc is None else loc + w * np.asarray(val))

            # ---------------- scatter -----------------------------------------
            if rhs and test:                   # linear form  (vector assemble)
                np.add.at(matvec, gdofs, loc)
            elif not rhs and trial and test:   # bilinear form (matrix assemble)
                r, c = np.meshgrid(gdofs, gdofs, indexing="ij")
                matvec[r, c] += loc
            elif is_functional:               # hooked scalar functional
                # print('-'*50)
                # print(f"loc.shape: {loc.shape}, type: {type(loc)}")
                if isinstance(loc, VecOpInfo):    # collapse (k,n) → scalar
                    loc = np.sum(loc.data, axis=1)
                self.ctx["global_dofs"] = gdofs
                self.ctx["scalar_results"][hook["name"]] += loc

        # tidy
        for k in ("eid", "x_phys", "normal", "global_dofs"):
            self.ctx.pop(k, None)
    def _get_data_functions_objs(self, intg: Integral):
        """
        Extracts all Function and VectorFunction objects from the integrand.
        This is used to gather the current state of coefficient functions
        before executing the JIT kernel.
        """
        coeffs = (Function, VectorFunction)
        current = {f.name: f
                   for f in _find_all(intg.integrand, coeffs)
                   if not getattr(f, "is_test", False)
                   and not getattr(f, "is_trial", False)}
        
        # Add parent vectors if they look like real coefficient vectors
        for f in list(current.values()):
            pv = getattr(f, "_parent_vector", None)
            if pv is not None and hasattr(pv, "name"):
                current.setdefault(pv.name, pv)

        return current
    
    def _functional_calculate(self, intg: Integral, J_loc, hook):
        trial, test = _trial_test(intg.integrand)
        is_functional = (trial is None and test is None)
        flag = False
        if is_functional:
            # J_loc is (n_edges,) or scalar; collapse to plain float
            if J_loc is not None:
                if J_loc.ndim > 1:
                    total = np.sum(J_loc, axis=0)  # (n_edges,) → scalar
                    # print(f"J_loc.shape={J_loc.shape}, total={total}")
                if J_loc.ndim == 1:
                    total = J_loc.sum()
                    # print(f"J_loc.shape={J_loc.shape}, total={total}")
            name  = hook["name"]               # retrieved earlier via _hook_for
            scal  = self.ctx.setdefault("scalar_results", {})
            scal[name] =  total
            flag = True
        return flag                            # 🔑  skip the scatter stage

    def _assemble_boundary_edge_jit(self, intg: Integral, matvec):
        mesh, dh, me = self.me.mesh, self.dh, self.me

        # 1. pick edges -----------------------------------------------
        edge_set = (intg.measure.defined_on
                    if intg.measure.defined_on is not None
                    else mesh.get_domain_bitset(intg.measure.tag, entity="edge"))
        if edge_set.cardinality() == 0:
            raise ValueError(f"[Assembler: boundary edge JIT] No edges defined for {intg.measure.tag}.")
        else:
            print(f"Found boundary edges: {edge_set.cardinality()}")

            

        # 2. geometry tables ------------------------------------------
        qdeg = self._find_q_order(intg)
        print(f"[Assembler: boundary edge JIT] Using quadrature degree: {qdeg}")
        # 3. kernel compilation ---------------------------------------
        runner, ir = self._compile_backend(intg.integrand, dh, me)
        derivs, need_hess, need_o3, need_o4 = self._find_req_derivs(intg, runner)
        geo  = dh.precompute_boundary_factors(edge_set, qdeg, derivs, 
                                              need_hess=need_hess, need_o3=need_o3, need_o4=need_o4)

        geo["is_interface"] = False
        valid = geo["eids"]
        if valid.size == 0:
            return


        args = _build_jit_kernel_args(
            ir, intg.integrand, me, qdeg,
            dof_handler = dh,
            gdofs_map   = geo["gdofs_map"],
            param_order = runner.param_order,
            pre_built   = geo
        )

        # if anything is not aligned to edges
        n_edges = geo["qp_phys"].shape[0]
        owner   = geo.get("owner_id")                # per-edge owner element id
        n_elems = self.me.mesh.n_elements

        for name in runner.param_order:
            arr = args.get(name)
            if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] != n_edges:
                if arr.shape[0] == n_elems and owner is not None:
                    # Gather per-edge rows from per-element array
                    args[name] = arr[owner]
                else:
                    raise ValueError(
                        f"Static array '{name}' has shape[0]={arr.shape[0]} "
                        f"but boundary kernel expects {n_edges} edges."
                    )

        # 4. up-to-date coefficient Functions -------------------------
        #    • ignore test/trial symbols
        #    • add parent vectors only if they look like real coefficient vectors
        current = self._get_data_functions_objs(intg)

        # 5. execute and scatter --------------------------------------
        K_loc, F_loc, J_loc = runner(current, args)


        hook = self._hook_for(intg.integrand)
        if self._functional_calculate(intg, J_loc, hook): return
            
        _scatter_element_contribs(
            K_loc, F_loc, J_loc,
            element_ids=geo["eids"],                 # rows ≡ edge ids
            gdofs_map=geo["gdofs_map"],
            matvec=matvec, 
            ctx=self.ctx, 
            integrand=intg.integrand,
            hook=hook,
        )
    def _assemble_boundary_edge(self, intg: Integral, matvec):
        """Assemble integrals over boundary edges."""
        if self.backend == "python":
            self._assemble_boundary_edge_python(intg, matvec)
        elif self.backend == "jit":
            self._assemble_boundary_edge_jit(intg, matvec)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}. Use 'python' or 'jit'.")
    # --- CUT-CELL helpers -------------------------------------------------

    def _physical_tri_quadrature(self, tri_coords, q_order):
        """
        Map the reference-triangle quadrature to the physical triangle with vertices tri_coords.
        Returns (qpoints_phys, qweights_phys). Weights are *physical* areas.
        """
        v0, v1, v2 = map(np.asarray, tri_coords)
        qp_ref, qw_ref = volume('tri', q_order)     # reference rule (r,s) on Δ(0,0)-(1,0)-(0,1)
        J = np.column_stack((v1 - v0, v2 - v0))     # 2x2 affine map
        detJ = abs(np.linalg.det(J))
        qp_phys = (qp_ref @ J.T) + v0               # (nq,2)
        qw_phys = qw_ref * detJ                     # area scaling → physical weights
        return qp_phys, qw_phys
    
    def _integrate_on_cut_element(self, eid: int, integral, level_set, q_order: int, side: str):
        """
        Integrate over the physical part of element `eid` by clipping each geometric
        corner-triangle against phi=0, then sub-triangulating and integrating each piece.
        Returns the local element vector/matrix (shape matches self.ctx['rhs']).
        """
        mesh   = self.me.mesh
        fields = _all_fields(integral.integrand)
        rhs    = self.ctx.get('rhs', False)

        # local accumulator
        loc = (np.zeros(self.me.n_dofs_local) if rhs
            else np.zeros((self.me.n_dofs_local, self.me.n_dofs_local)))

        # 1) split geometry into corner triangles (tri stays as-is; quad → two tris)
        elem = mesh.elements_list[eid]
        tri_local, corner_ids = corner_tris(mesh, elem)

        # 2) reference rule on the unit triangle
        qp_ref, qw_ref = volume("tri", q_order)

        # 3) process each geometric triangle
        for loc_tri in tri_local:
            v_ids   = [corner_ids[i] for i in loc_tri]
            v_coords = mesh.nodes_x_y_pos[v_ids]  # (3,2)
            v_phi    = np.array([phi_eval(level_set, xy) for xy in v_coords], dtype=float)

            # clip to requested side and fan-triangulate each polygon
            polygons = clip_triangle_to_side(v_coords, v_phi, side=side)
            for poly in polygons:
                for A, B, C in fan_triangulate(poly):
                    qp_phys, qw_phys = map_ref_tri_to_phys(A, B, C, qp_ref, qw_ref)

                    # 4) quadrature loop in *physical* space (weights are physical)
                    for x_phys, w in zip(qp_phys, qw_phys):
                        # reference point of the *parent element* that owns x_phys
                        xi, eta = transform.inverse_mapping(mesh, eid, x_phys)
                        J       = transform.jacobian(mesh, eid, (xi, eta))
                        Ji     = np.linalg.inv(J)

                        # cache bases at this point (zero-padded across mixed fields)
                        self._basis_cache.clear();self._coeff_cache.clear(); self._collapsed_cache.clear();
                        for f in fields:
                            self._basis_cache[f] = {
                                "val" : self.me.basis(f, xi, eta),
                                "grad": self.me.grad_basis(f, xi, eta) @ Ji
                            }

                        # context for UFL visitors
                        self.ctx["eid"]     = eid
                        self.ctx["x_phys"]  = x_phys
                        self.ctx["phi_val"] = phi_eval(level_set, x_phys)

                        integrand_val = self._visit(integral.integrand)
                        # IMPORTANT: 'w' already includes the area Jacobian of the sub-triangle.
                        loc += w * integrand_val

        return loc

    def _assemble_volume_cut_python(self, integral, matvec):
        """
        Assemble a volume integral restricted by the 'side' of a level set:
            dx(level_set=phi, metadata={'side': '+/-'})
        - full elements on the selected side: standard reference rule
        - cut elements: clipped sub-triangle physical quadrature
        """
        mesh      = self.me.mesh
        level_set = integral.measure.level_set
        if level_set is None:
            raise ValueError("Cut-cell volume assembly requires measure.level_set")

        side    = integral.measure.metadata.get('side', '+')  # '+' → phi>0, '-' → phi<0
        q_order = self._find_q_order(integral)
        rhs     = self.ctx.get("rhs", False)
        fields  = _all_fields(integral.integrand)

        # --- 1) classify (fills tags & caches)
        inside_ids, outside_ids, cut_ids = mesh.classify_elements(level_set)

        # Reference rule & scatter util
        qp_ref, qw_ref = volume(mesh.element_type, q_order)

        def _scatter(eid, loc):
            gdofs = self.dh.get_elemental_dofs(eid)
            if rhs:
                np.add.at(matvec, gdofs, loc)
            else:
                r, c = np.meshgrid(gdofs, gdofs, indexing="ij")
                matvec[r, c] += loc

        # --- 2) full elements on the selected side → standard reference rule
        full_eids = outside_ids if side == '+' else inside_ids
        for eid in full_eids:
            loc = (np.zeros(self.me.n_dofs_local) if rhs
                else np.zeros((self.me.n_dofs_local, self.me.n_dofs_local)))

            for (xi, eta), w in zip(qp_ref, qw_ref):
                J    = transform.jacobian(mesh, eid, (xi, eta))
                detJ = abs(np.linalg.det(J))
                Ji  = np.linalg.inv(J)

                self._basis_cache.clear();self._coeff_cache.clear(); self._collapsed_cache.clear();
                for f in fields:
                    self._basis_cache[f] = {
                        "val" : self.me.basis(f, xi, eta),
                        "grad": self.me.grad_basis(f, xi, eta) @ Ji
                    }
                self.ctx["eid"]    = eid
                self.ctx["x_phys"] = transform.x_mapping(mesh, eid, (xi, eta))
                integrand_val      = self._visit(integral.integrand)
                loc += w * detJ * integrand_val

            _scatter(eid, loc)

        # --- 3) cut elements → clipped sub-triangles
        for eid in cut_ids:
            loc = self._integrate_on_cut_element(eid, integral, level_set, q_order, side=side)
            _scatter(eid, loc)

        # cleanup
        for key in ("eid", "x_phys", "phi_val"):
            self.ctx.pop(key, None)

    
    def _assemble_volume_cut_jit(self, intg, matvec):
        """
        Assemble ∫_{Ω∩{φ▹0}} (...) dx with the JIT back-end.

        Strategy:
        1) classify elements w.r.t. φ
        2) assemble full elements on the selected side with the *standard* JIT volume
        3) assemble cut elements with element-specific (clipped) quadrature
        """
        mesh, dh, me = self.me.mesh, self.dh, self.me
        qdeg   = self._find_q_order(intg)
        level_set = intg.measure.level_set
        side   = intg.measure.metadata.get('side', '+')   # '+' → φ>0, '-' → φ<0

        # --- 1) classification (must be called; sets element tags & caches)
        inside_ids, outside_ids, cut_ids = mesh.classify_elements(level_set)

        # --- compile kernel once
        runner, ir = self._compile_backend(intg.integrand, dh, me)
        derivs, need_hess, need_o3, need_o4 = self._find_req_derivs(intg, runner)

        # helper to run kernel on a subset of elements with prebuilt geo
        def _run_subset(eids: np.ndarray, prebuilt: dict):
            if eids.size == 0:
                return
            gdofs_map = np.vstack([dh.get_elemental_dofs(e) for e in eids]).astype(np.int32)
            # kernel wants node_coords in the signature, provide it once
            prebuilt = dict(prebuilt)
            prebuilt["gdofs_map"]  = gdofs_map
            prebuilt["node_coords"] = dh.get_all_dof_coords()  # required in param_order

            static_args = _build_jit_kernel_args(
                ir, intg.integrand, me, qdeg,
                dof_handler=dh,
                gdofs_map=gdofs_map,
                param_order=runner.param_order,
                pre_built=prebuilt
            )
            current_funcs = self._get_data_functions_objs(intg)
            K_loc, F_loc, J_loc = runner(current_funcs, static_args)

            hook = self._hook_for(intg.integrand)
            _scatter_element_contribs(
                K_loc, F_loc, J_loc, eids, gdofs_map,
                matvec, self.ctx, intg.integrand, hook=hook
            )

        # --- 2) full elements on the selected side → standard volume tables sliced
        full_ids = np.asarray(outside_ids if side == '+' else inside_ids, dtype=np.int32)
        if full_ids.size:
            # include level_set so 'phis' is populated (still fine if None)
            geo_all = dh.precompute_geometric_factors(qdeg, level_set, 
                                                      need_hess=need_hess, need_o3=need_o3, need_o4=need_o4)

            # slice what the kernel signature expects
            prebuilt_full = {
                "qp_phys": geo_all["qp_phys"][full_ids],
                "qw":      geo_all["qw"][full_ids],
                "detJ":    geo_all["detJ"][full_ids],
                "J_inv":   geo_all["J_inv"][full_ids],
                "normals": geo_all["normals"][full_ids],
                # 'phis' may be None if level_set was None; keep it as None in that case
                "phis":    None if geo_all["phis"] is None else geo_all["phis"][full_ids],
                "owner_id": geo_all.get("owner_id", geo_all["eids"])[full_ids].astype(np.int32),
                "entity_kind": "element",
                "is_interface": False
            }
            _run_subset(full_ids, prebuilt_full)

        # --- 3) cut elements → clipped triangles (physical weights); detJ := 1
        if len(cut_ids):
            from pycutfem.ufl.helpers import required_multi_indices
            derivs = required_multi_indices(intg.integrand) | {(0, 0)}

            # precomputed, element-specific qp / qw / J_inv  (+ basis tables b_* / dxy_*)
            geo_cut = dh.precompute_cut_volume_factors(
                mesh.element_bitset("cut"), qdeg, derivs, level_set, side=side, need_hess=need_hess
            )
            geo_cut["is_interface"] = False
            cut_eids = np.asarray(geo_cut.get("eids", []), dtype=np.int32)
            if cut_eids.size:
                # ensure detJ is present and neutral (if your helper has not added it)
                if "detJ" not in geo_cut:
                    geo_cut["detJ"] = np.ones_like(geo_cut["qw"])
                _run_subset(cut_eids, geo_cut)

        # cleanup context (if you stored any debug keys)
        for k in ("eid", "x_phys", "phi_val"):
            self.ctx.pop(k, None)





