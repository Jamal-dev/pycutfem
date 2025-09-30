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
from pycutfem.integration.cut_integration import CutIntegration

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
                                  HelpersFieldAware as _hfa,
                                  HelpersAlignCoefficents as _hac,
                                  normalize_edge_ids,
                                  normalize_elem_ids)
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.helpers_jit import  _build_jit_kernel_args, _scatter_element_contribs, _stack_ragged
from pycutfem.ufl.helpers_geom import (
    phi_eval, clip_triangle_to_side, fan_triangulate, map_ref_tri_to_phys, corner_tris
)
from contextlib import contextmanager
from pycutfem.core.sideconvention import SIDE





logger = logging.getLogger(__name__)
_INTERFACE_TOL = SIDE.tol

def interface_normal_for_edge(mesh, e, level_set):
    # mid-point on the physical edge
    p0, p1 = mesh.nodes_x_y_pos[list(e.nodes)]
    mid = 0.5*(p0 + p1)

    # normal from ∇φ, fallback to edge-normal
    g = np.asarray(level_set.gradient(mid), float)
    if np.linalg.norm(g) > 1e-15:
        n = g / np.linalg.norm(g)
    else:
        t = p1 - p0
        t /= max(np.linalg.norm(t), 1e-16)
        n = np.array([t[1], -t[0]], float)

    # Decide which neighbor is (+) and which is (−) by the global convention
    cl = np.asarray(mesh.elements_list[e.left ].centroid())
    cr = np.asarray(mesh.elements_list[e.right].centroid())
    phiL = float(level_set(*cl))
    phiR = float(level_set(*cr))

    # Primary: classify by SIDE; Fallback: compare values
    if SIDE.is_pos(phiL) and not SIDE.is_pos(phiR):
        cpos, cneg = cl, cr
    elif SIDE.is_pos(phiR) and not SIDE.is_pos(phiL):
        cpos, cneg = cr, cl
    else:
        # ambiguous (both on same side): orient by larger φ as a stable fallback
        cpos, cneg = (cl, cr) if phiL >= phiR else (cr, cl)

    # Orient normal from (−) → (+)
    if np.dot(n, cpos - cneg) < 0.0:
        n = -n
    return n




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
    
    @contextmanager
    def _push_ctx(self, key, value):
        old = self.ctx.get(key, None)
        self.ctx[key] = value
        try:
            yield
        finally:
            if old is None:
                self.ctx.pop(key, None)
            else:
                self.ctx[key] = old
    # --------------------- OpInfo meta helpers ---------------------
    def _op_meta_from_ctx(self, node, field_names):
        """Build consistent metadata for VecOpInfo/GradOpInfo from ctx + node."""
        # Side: prefer context (Pos/Neg/Jump set this), then node; else ""
        side = self.ctx.get("side")
        if side not in ("+","-"):
            try:
                side = _hfa._side_from_node(self, node)
            except Exception:
                side = None
        side = side if side in ("+","-") else ""

        # Parent/container name
        parent_name = getattr(node, "parent_name", "")

        # DO NOT infer per-field sides from names — leave empty unless node declares them
        fs = getattr(node, "field_sides", None)
        field_sides = fs if (isinstance(fs, list) and len(fs) == len(field_names)) else []

        is_rhs = bool(self.ctx.get("rhs", True))
        return {
            "field_names": list(field_names),
            "parent_name": parent_name,
            "side": side,
            "field_sides": list(field_sides),
            "is_rhs": is_rhs,
        }


    def _vecinfo(self, data: np.ndarray, role: str, node, field_names):
        meta = self._op_meta_from_ctx(node, field_names)
        return VecOpInfo(np.asarray(data), role=role, **meta)

    def _gradinfo(self, data: np.ndarray, role: str, node, field_names, coeffs=None):
        meta = self._op_meta_from_ctx(node, field_names)
        return GradOpInfo(np.asarray(data), role=role, coeffs=coeffs, **meta)
    def _hessinfo(self, data: np.ndarray, role: str, node, field_names, coeffs=None):
        meta = self._op_meta_from_ctx(node, field_names)
        return HessOpInfo(np.asarray(data), role=role, coeffs=coeffs, **meta)



    # ============================ PUBLIC API ===============================
    def assemble(self, eq: Equation, bcs: Union[Mapping, Iterable, None] = None):
        ndofs = self.dh.total_dofs
        K = sp.lil_matrix((ndofs, ndofs))
        F = np.zeros(ndofs)

        # Assemble LHS if it is provided.
        if eq.a is not None and not (isinstance(eq.a, (int, float)) or eq.a != 0.0):
            logger.info("Assembling LHS...")
            self.ctx["rhs"] = False
            self._assemble_form(eq.a, K)

        # Assemble RHS if it is provided.
        if eq.L is not None and not (isinstance(eq.L, (int, float)) or eq.L != 0.0):
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
    
    def _active_side(self, default: str | None = None) -> str:
        """
        Resolve the current side with a strict precedence:
          1) explicit ctx['side'] set by Pos/Neg/Jump or the assembler,
          2) ctx['measure_side'] injected by dx/dInterface/dGhost assemblers,
          3) owner matching (eid == pos_eid/neg_eid) on ghost/interface,
          4) sign of ctx['phi_val'] (last resort only),
        and finally fall back to '+' if nothing is known.
        """
        s = self.ctx.get("side", None)
        if s in ("+", "-"):
            return s
        ms = self.ctx.get("measure_side", None)
        if ms in ("+", "-"):
            return ms
        eid = self.ctx.get("eid", None)
        if eid is not None:
            if eid == self.ctx.get("pos_eid", None):
                return "+"
            if eid == self.ctx.get("neg_eid", None):
                return "-"
        pv = self.ctx.get("phi_val", None)
        if pv is not None:
            try:
                return "+" if SIDE.is_pos(float(pv)) else "-"
            except Exception:
                pass

        return default if default in ("+", "-") else "+"


    # ===================== VISITORS: LEAF NODES =======================
    # --------------------------------------------------------------
    # Unified, side-aware, cached basis/derivative retrieval
    # --------------------------------------------------------------
    def _basis_row(self, field: str, alpha: tuple[int, int]) -> np.ndarray:
        """
        Row for D^{alpha} φ_field at the current QP.

        Ghost/interface path:
        • Per-QP, per-side cache in ctx["basis_values"][side][field][alpha]
        • Evaluates on (pos_eid|neg_eid) using physical mapping
        • ALWAYS slices element row → field-local row before any padding
        • Applies optional per-field side masks (pos/neg)
        • Pads to ctx["global_dofs"] (union) iff present

        Standard path:
        • Uses self._basis_cache[field] for val/grad/hess/derivs
        """

        ox, oy = map(int, alpha)

        # ---------- Ghost / Interface: side-aware path ----------
        bv = self.ctx.get("basis_values")
        if isinstance(bv, dict) and self._on_sided_path():
            # Determine active side
            side = self._active_side()

            # Side/field cache
            per_field = bv.setdefault(side, {}).setdefault(field, {})
            if alpha in per_field:
                return per_field[alpha]

            # Coordinates and owner element on this side
            x_phys = self.ctx.get("x_phys")
            if x_phys is None:
                raise KeyError("x_phys missing in context for derivative evaluation.")
            eid = int(self.ctx["pos_eid"] if side == '+' else self.ctx["neg_eid"])

            # Map to reference and evaluate derivative row on the element
            xi, eta = transform.inverse_mapping(self.me.mesh, eid, np.asarray(x_phys))
            row = self._phys_scalar_deriv_row(field, float(xi), float(eta), ox, oy, eid)
            # row is length n_dofs_local and (in our library) zero-padded across fields.

            # Target "union" layout (present when assembling a matrix/vector)
            g = self.ctx.get("global_dofs")  # None for functionals
            # Prefer per-field map; then fall back to side-wide map-by-field; finally generic side map.
            amap = None
            try:
                amap = _hfa.get_field_map(self.ctx, side, field)
            except Exception:
                amap = None
            if amap is None:
                maps_by_field = self.ctx.get("pos_map_by_field" if side == '+' else "neg_map_by_field")
                if isinstance(maps_by_field, dict):
                    amap = maps_by_field.get(field)
            if amap is None:
                side_map = self.ctx.get("pos_map") if side == '+' else self.ctx.get("neg_map")
                # 'side_map' is usually the identity over the union; only used if row is already union-sized.

            # ALWAYS collapse element-union → field-local when we can
            field_slice = None
            try:
                field_slice = self.me.component_dof_slices[field]
            except Exception:
                field_slice = None
            if field_slice is not None and len(row) == self.me.n_dofs_local:
                row = row[field_slice]  # make it field-local unconditionally

            # Optional per-field side mask (field-local)
            apply_mask = bool(self.ctx.get("mask_basis", False))
            mask_dict = None
            if apply_mask:
                mask_dict = self.ctx.get("pos_mask_by_field") if side == '+' else self.ctx.get("neg_mask_by_field")
                if isinstance(mask_dict, dict):
                    m_local = mask_dict.get(field)
                    if m_local is not None and len(row) == len(m_local):
                        row = row * m_local  # apply while still field-local

            # Pad to union layout only if assembling a mat/vec
            if g is not None:
                if amap is not None:
                    if len(row) == len(amap):
                        full = np.zeros(len(g), dtype=row.dtype)
                        full[np.asarray(amap, dtype=int)] = row
                        row = full
                    elif len(row) != len(g):
                        raise ValueError(
                            f"Shape mismatch padding basis row for '{field}': "
                            f"len(row)={len(row)}, len(amap)={len(amap)}, len(global_dofs)={len(g)}"
                        )
                    # else len(row) == len(g): already union-sized (rare here)
                else:
                    # No per-field map; row must already be union-sized
                    if len(row) != len(g):
                        # As a last resort, allow a side-wide identity map
                        side_map = self.ctx.get("pos_map") if side == '+' else self.ctx.get("neg_map")
                        if side_map is not None and len(side_map) == len(g) and len(row) == len(side_map):
                            # already union-sized w.r.t. the side map; nothing to do
                            pass
                        else:
                            raise ValueError(
                                f"Cannot pad basis row for '{field}': no field map and "
                                f"len(row)={len(row)} != len(global_dofs)={len(g)}"
                            )

                # If we padded to union after masking (field-local), we may need to lift the mask too.
                if isinstance(mask_dict, dict):
                    m_local = mask_dict.get(field)
                    if m_local is not None and amap is not None and len(row) == len(g) and len(m_local) != len(row):
                        m_full = np.zeros(len(g), dtype=row.dtype)
                        m_full[np.asarray(amap, dtype=int)] = m_local
                        row = row * m_full

            # Cache final form (per-side/field/alpha) and return
            per_field[alpha] = row
            return row

        # ---------- Standard element-local path (volume/boundary) ----------
        cache = self._basis_cache.setdefault(field, {})


        # Fast paths
        if alpha == (0, 0) and "val" in cache:
            return cache["val"]
        if alpha in ((1, 0), (0, 1)) and "grad" in cache:
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
            # also stash each entry into derivs
            dcache[(2, 0)] = d20; dcache[(1, 1)] = d11; dcache[(0, 2)] = d02
            return row

        dcache[alpha] = row
        return row








    
    #----------------------------------------------------------------------
    def _b(self, fld): 
        return self._basis_row(fld, (0,0))
    def _v(self, fld):
        # 1) Prefer per-QP, side-aware cache *only* on sided paths:
        if self._on_sided_path():
            bv = self.ctx.get("basis_values")
            if isinstance(bv, dict):
                side = self._active_side()
                if side in ('+','-'):
                    val = bv[side][fld].get((0, 0))
                    if val is not None:
                        return val
        # 2) Fallback to element-local caches (prefilled in volume paths)
        if fld in self._basis_cache:
            return self._basis_cache[fld]["val"]
        # 3) Last resort: compute from eid/x_phys
        return self._basis_from_element_context(fld, kind="val")

    def _g(self, fld):
        gx = self._basis_row(fld, (1,0))
        gy = self._basis_row(fld, (0,1))
        result = np.stack([gx, gy], axis=1)  # (nloc, 2)
        return result

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
                return self._vecinfo(np.array([tr_val]), role="function", node=node, field_names=A.field_names)

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
            return self._vecinfo(np.stack([tr_vec]), role=A.role, node=node, field_names=A.field_names)

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
                return self._vecinfo(tr, role="function", node=node, field_names=A.field_names)
            else:
                # test/trial: return (k,n) table of Laplacians
                tr = A.data[..., 0, 0] + A.data[..., 1, 1]
                return self._vecinfo(tr, role=A.role, node=node, field_names=A.field_names)

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

        return self._vecinfo(data, role=role, node=op, field_names=[fld]) 
   
    # --- Knowns (evaluated to numerical values) ---
    def _visit_Constant(self, n: Constant):
        if n.dim ==0:
            return float(n.value)
        return np.asarray(n.value)
    
    def _visit_FacetNormal(self, n: FacetNormal): 
        """Returns the normal vector from the context.""" 
        if 'normal' not in self.ctx: 
            raise RuntimeError("FacetNormal accessed outside of a facet or interface integral context.") 
        return self.ctx['normal'] 
    
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
    
    
    def _on_sided_path(self) -> bool:
        """
        Return True only for interface/ghost evaluations.
        Volume integrals must use the element-local path, even if some
        sided keys linger in ctx from a previous integral.
        """
        return bool(self.ctx.get('is_interface', False) or self.ctx.get('is_ghost', False))

    def _with_side(self, side: str, eid_key: str, operand):
        phi_old  = self.ctx.get('phi_val', None)
        side_old = self.ctx.get('side', None)
        eid_old  = self.ctx.get('eid', None)
        try:
            # self.ctx['phi_val'] = +1.0 if side == '+' else -1.0
            self.ctx['side']    = side
            if eid_key in self.ctx:
                self.ctx['eid'] = int(self.ctx[eid_key])
            return self._visit(operand)
        finally:
            if phi_old  is None: self.ctx.pop('phi_val', None)
            else:                self.ctx['phi_val'] = phi_old
            if side_old is None: self.ctx.pop('side',    None)
            else:                self.ctx['side']    = side_old
            if eid_old  is None: self.ctx.pop('eid',     None)
            else:                self.ctx['eid']     = eid_old
    def _visit_Pos(self, n):
        if self._on_sided_path():
            return self._with_side('+', 'pos_eid', n.operand)
        # VOLUME path: φ-gate
        val = self._with_side('+', 'pos_eid', n.operand)
        pv = float(self.ctx.get('phi_val', 0.0))
        return val if SIDE.is_pos(pv) else (val * 0.0)

    def _visit_Neg(self, n):
        if self._on_sided_path():
            return self._with_side('-', 'neg_eid', n.operand)
        # VOLUME path: φ-gate
        val = self._with_side('-', 'neg_eid', n.operand)
        # return val
        pv = float(self.ctx.get('phi_val', 0.0))
        return (val * 0.0) if SIDE.is_pos(pv) else val

    def _visit_Jump(self, n: Jump):
        phi_old  = self.ctx.get('phi_val', None)
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
        if phi_old is None: self.ctx.pop('phi_val', None)
        else:               self.ctx['phi_val'] = phi_old
        if eid_old is None: self.ctx.pop('eid', None)
        else:               self.ctx['eid'] = eid_old
        if side_old is None: self.ctx.pop('side', None)
        else:                self.ctx['side'] = side_old
        # print(f'Jump: type(u_pos):{type(u_pos)}, type(u_neg):{type(u_neg)}'
        #       f';role_a: {getattr(u_pos, "role", None)}, role_b: {getattr(u_neg, "role", None)}')

        return u_pos - u_neg

###########################################################################################3
########### --- Functions and VectorFunctions (evaluated to numerical values) ---
###########################################################################################
    def _get_side(self):
        return self._active_side()
    def _zero_width(self):
        g = self.ctx.get("global_dofs")
        return len(g) if g is not None else len(self._local_dofs())


    def _visit_Function(self, n: Function):
        logger.debug(f"Visiting Function: {n.field_name}")
        # gatting with the mask
        self.ctx["mask_basis"] = True

        # element-union coeffs for the *current side* and the side-trace basis row
        local_dofs = self._local_dofs()
        u_loc = n.get_nodal_values(local_dofs)   # (n_loc,)
        basis_00   = self._b(n.field_name)            # (n_loc,) or already padded

        # On ghost/interface paths, align BOTH basis and coeffs to the same global layout
        gd = self.ctx.get("global_dofs", None)
        if gd is not None:
            side = self._get_side()  # '+' or '-'
            basis_00, u_loc = _hac._align_phi_and_coeffs_to_global(self.ctx, side, n.field_name, basis_00, u_loc)

        # Elementwise product: returns a single block (keep legacy shape: list of 1 vector)
        data = [u_loc * basis_00]
        return self._vecinfo(data, role="function", node=n, field_names=[n.field_name])

    def _visit_VectorFunction(self, n: VectorFunction):
        logger.debug(f"Visiting VectorFunction: {n.field_names}")
        # gatting with the mask
        self.ctx["mask_basis"] = True
        # Keep only components matching an explicit side (if any filter is in effect)
        names = _hfa._filter_fields_for_side(self, n, list(n.field_names))
        comp_by_name = {fn: comp for fn, comp in zip(n.field_names, n.components)}

        local_dofs = self._local_dofs()
        blocks = []
        for fld in names:
            coeffs = comp_by_name[fld].get_nodal_values(local_dofs)  # (n_loc,) or (ncomp, n_loc)
            basis_00    = self._b(fld)                                    # (n_loc,) or (ncomp, n_loc)

            gd = self.ctx.get("global_dofs", None)
            if gd is not None:
                side = self._get_side()
                basis_00, coeffs = _hac._align_phi_and_coeffs_to_global(self.ctx, side, fld, basis_00, coeffs)

            # Elementwise product per component
            blocks.append(coeffs * basis_00)

        if blocks:
            # stack along component axis; resulting shape: (ncomp_total, ndofs_global_or_local)
            data = np.stack(blocks)
        else:
            # produce an empty (0, width) block with the correct width for downstream code
            width = len(self.ctx["global_dofs"]) if self.ctx.get("global_dofs") is not None else len(local_dofs)
            data = np.zeros((0, width))

        return self._vecinfo(data, role="function", node=n, field_names=names)




    # --- Unknowns (represented by basis functions) ---
    def _visit_TestFunction(self, n):
        logger.debug(f"Visiting TestFunction: {n.field_name}")
        # gatting with the mask
        self.ctx["mask_basis"] = False
        row = self._lookup_basis(n.field_name, (0, 0))[np.newaxis, :]
        return self._vecinfo(row, role="test", node=n, field_names=[n.field_name])

    def _visit_TrialFunction(self, n):
        logger.debug(f"Visiting TrialFunction: {n.field_name}")
        # gatting with the mask
        self.ctx["mask_basis"] = False
        row = self._lookup_basis(n.field_name, (0, 0))[np.newaxis, :]
        return self._vecinfo(row, role="trial", node=n, field_names=[n.field_name])


    def _visit_VectorTestFunction(self, n):
        logger.debug(f"Visiting VectorTestFunction: {n.field_names}")
        # gatting with the mask
        self.ctx["mask_basis"] = False
        names = _hfa._filter_fields_for_side(self, n, list(n.field_names))
        rows = [self._lookup_basis(f, (0, 0)) for f in names]
        data  = np.stack(rows) if rows else np.zeros((len(n.field_names), self._zero_width()))
        return self._vecinfo(data, role="test", node=n, field_names=names)

    def _visit_VectorTrialFunction(self, n):
        logger.debug(f"Visiting VectorTrialFunction: {n.field_names}")
        # gatting with the mask
        self.ctx["mask_basis"] = False
        names = _hfa._filter_fields_for_side(self, n, list(n.field_names))
        rows = [self._lookup_basis(f, (0, 0)) for f in names]
        data  = np.stack(rows) if rows else np.zeros((len(n.field_names), self._zero_width()))
        return self._vecinfo(data, role="trial", node=n, field_names=names)

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
        fields = _hfa._filter_fields_for_side(self, op, list(fields))

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
            return self._gradinfo(np.stack(k_blocks), role=role, node=op, field_names=fields)

        else:
            for fld in fields:
                g = self._g(fld)
                k_blocks.append(g)
            return self._gradinfo(np.stack(k_blocks), role=role, node=op, field_names=fields)
    
    def _visit_Hessian(self, n: Hessian):
        """
        Build per-component Hessian tables.
        • Test/Trial → (k, n, 2, 2) per-component per-DOF Hessian tables.
        • Function   → (k, 2, 2) per-component collapsed Hessians at the QP.
        """
        op = n.operand

        # Distribute over sided/jump operators so side selection happens upstream.
        if isinstance(op, Jump):
            return self._visit(Jump(Hessian(op.u_pos), Hessian(op.u_neg)))
        if isinstance(op, Pos):
            return self._visit(Pos(Hessian(op.operand)))
        if isinstance(op, Neg):
            return self._visit(Neg(Hessian(op.operand)))

        # Role + component names
        if isinstance(op, (TestFunction, VectorTestFunction)):
            role = "test"
        elif isinstance(op, (TrialFunction, VectorTrialFunction)):
            role = "trial"
        elif isinstance(op, (Function, VectorFunction)):
            role = "function"
        else:
            role = "none"

        fields = op.field_names if hasattr(op, "field_names") else [op.field_name]

        # -- Function: collapse per component: H_ij(xq) = Σ_n H_ij(n) * u_n
        if role == "function":
            local_dofs = self._local_dofs()

            def _get_coeff(comp_idx: int) -> np.ndarray:
                key = (id(op), "coef", comp_idx, tuple(local_dofs))
                c = self._coeff_cache.get(key)
                if c is None:
                    if hasattr(op, "components"):
                        c = op.components[comp_idx].padded_values(local_dofs)
                    else:
                        c = op.padded_values(local_dofs)
                    self._coeff_cache[key] = c
                return c

            k_blocks = []
            for i, fld in enumerate(fields):
                key_coll = (id(op), "Hcoll", i, tuple(local_dofs))
                Hval = self._collapsed_cache.get(key_coll)
                if Hval is None:
                    Htbl   = self._hess(fld)             # (n, 2, 2); padded to union if on ghost/interface
                    coeffs = _get_coeff(i)               # (n,)
                    Hval   = np.einsum("nij,n->ij", Htbl, coeffs, optimize=True)  # (2, 2)
                    self._collapsed_cache[key_coll] = Hval
                k_blocks.append(Hval)

            data = np.stack(k_blocks)                    # (k, 2, 2)
            return self._hessinfo(data, role=role, node=op, field_names=fields)

        # -- Test/Trial: return the per-DOF Hessian tables
        k_blocks = []
        for fld in fields:
            Htbl = self._hess(fld)                       # (n, 2, 2); padded to union if on ghost/interface
            k_blocks.append(Htbl)

        data = np.stack(k_blocks)                        # (k, n, 2, 2)
        return self._hessinfo(data, role=role, node=op, field_names=fields)


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
        d = self.me.mesh.spatial_dim

        num_comps = grad_op.data.shape[0]  # k
        if num_comps < 2:
            raise ValueError(f"DivOperation expects a vector/tensor operand with at least 2 components, got {num_comps}.")
        if grad_op.role == "function":
            # scaled the gradient
            if grad_op.coeffs is not None and grad_op.data.ndim == 3:
                grad_val = np.einsum("knd,kn->kd", grad_op.data, grad_op.coeffs, optimize=True) # (k,d) (2,2)
            else:
                grad_val = grad_op.data
            div_vec = np.sum([grad_val[i, i] for i in range(grad_val.shape[0])], axis=0)  # sum over k
        else:
            # ∇·v  =  Σ_i ∂v_i/∂x_i   → length n_loc (22) vector
            if d == num_comps:
                div_vec = np.sum([grad_op.data[i, :, i]          # pick diagonal components
                                for i in range(grad_op.data.shape[0])],
                                axis=0)
            else:
                I = np.eye(d, dtype=grad_op.data.dtype)
                div_k = np.einsum("knd,kd->kn", grad_op.data, I, optimize=True)  # (k, n)
                div_vec = div_k.sum(axis=0)                                       # (n,)

        # Decide which side of the bilinear form this lives on
        op = n.operand
        fields = op.field_names if hasattr(op, "field_names") else [op.field_name]
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
        return self._vecinfo(np.stack([div_vec]), role=role, node=op, field_names=fields)

    def _visit_CellDiameter(self, node):
        eid = self.ctx.get("eid")
        if eid is None:
            eid = self.ctx.get("pos_eid", self.ctx.get("neg_eid"))
        # if eid is None:
        #     raise KeyError("CellDiameter() requires 'eid' in context; set it in the element loop.")
        # h = self.me.mesh.element_char_length(eid)
        if self.ctx.get('is_interface'):
            # use min(h_left, h_right) on the face (robust even on aligned Γ)
            h = self.me.mesh.face_char_length(self.ctx.get('pos_eid'), self.ctx.get('neg_eid'))
        else:
            h = self.me.mesh.element_char_length(eid)
        # print(f"CellDiameter on eid={eid}: CellDiameter={h}")
        return float(h)


    
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
        # if isinstance(A, VecOpInfo):
        #     # VecOpInfo stores (k, n) -> transpose to (n, k)
        #     return VecOpInfo(A.data.T, role=A.role)

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
        a_data = a.data if isinstance(a, (VecOpInfo, GradOpInfo, HessOpInfo)) else a
        b_data = b.data if isinstance(b, (VecOpInfo, GradOpInfo, HessOpInfo)) else b
        shape_a = getattr(a_data,"shape", None)
        shape_b = getattr(b_data,"shape", None)
        # a_vec = np.squeeze(a_data) 
        # b_vec = np.squeeze(b_data)
        # role_a = getattr(a, 'role', None)
        # role_b = getattr(b, 'role', None)
        logger.debug(f"Entering _visit_Prod for  ('{n.a!r}' * '{n.b!r}') on {'RHS' if self.ctx['rhs'] else 'LHS'}") #, a.info={getattr(a, 'info', None)}, b.info={getattr(b, 'info', None)}
        # print(f" Product: a type={type(a)}, shape={shape_a}, b type={type(b)}, shape={shape_b}, side: {'RHS' if self.ctx['rhs'] else 'LHS'}"
        #       f" roles: a={getattr(a, 'role', None)}, b={getattr(b, 'role', None)}")

        result = a * b
        # print(f" Product result: {getattr(result, 'shape', None)}"
        #       f" roles: result={getattr(result, 'role', None)}"
        #       f" types: result={type(result)}")
        return result
        

        

    def _visit_Dot(self, n: Dot):
        a = self._visit(n.a)
        b = self._visit(n.b)
        a_data = a.data if isinstance(a, (VecOpInfo, GradOpInfo,HessOpInfo)) else a
        b_data = b.data if isinstance(b, (VecOpInfo, GradOpInfo,HessOpInfo)) else b
        role_a = getattr(a, 'role', None)
        role_b = getattr(b, 'role', None)
        shape_a = getattr(a_data,"shape", None)
        shape_b = getattr(b_data,"shape", None)
        # print(f"visit dot: role_a={role_a}, role_b={role_b}, side: {'RHS' if self.ctx['rhs'] else 'LHS'}"
        #       f" type_a={type(a)}, type_b={type(b)}"
        #       f" shape_a={shape_a}, shape_b={shape_b}")
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
            if ((isinstance(a, np.ndarray) and a.ndim == 1) or \
               (isinstance(a, VecOpInfo) and role_a == "vector")) and \
               isinstance(b, VecOpInfo) and b.role == "test":
                return b.dot_const(a)

            if ((isinstance(b, np.ndarray) and b.ndim == 1) or \
               (isinstance(b, VecOpInfo) and role_b == "vector")) and \
               isinstance(a, VecOpInfo) and a.role == "test":
                return a.dot_const(b)
            
            # Case 4: Constant(np.array([u1,u2])) · Trial or Function, no test so output is VecOpInfo
            if ((isinstance(a, np.ndarray) and a.ndim == 1) or \
                (isinstance(a, VecOpInfo) and role_a == "vector")) and \
             isinstance(b, (VecOpInfo)) and b.role in {"trial", "function"}:
                # return b.dot_const(a) if b.ndim==2 else b.dot_vec(a)
                return b.dot_const_vec(a) 
            elif (isinstance(b, np.ndarray) and b.ndim == 1  or \
               (isinstance(b, VecOpInfo) and role_b == "vector")) and \
             isinstance(a, (VecOpInfo)) and a.role in {"trial", "function"}:
                # return a.dot_const(b) if a.ndim==2 else a.dot_vec(b)
                return a.dot_const_vec(b)
            if isinstance(a, VecOpInfo) and role_a == "vector" and role_b == None:
                return a.dot_const(b)
            elif isinstance(b, VecOpInfo) and role_b == "vector" and role_a == None:
                return b.dot_const(a)
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
                raise TypeError(f"Unsupported dot product for RHS: '{n.a} . {n.b}'"
                                f" (roles: {role_a}, {role_b})"
                                f" (shapes: {getattr(a_data, 'shape', None)}, {getattr(b_data, 'shape', None)})")

        if role_a == None and role_b == "test":
            # Special case like dot( Constat(np.array([u1,u2])), TestFunction('v') )
            # This is a special case where we have a numerical vector on the LHS
            # and a test function basis on the RHS.
            if (isinstance(a, np.ndarray) and a.ndim == 1) and isinstance(b, VecOpInfo) and b.role == "test":
                return b.dot_const(a)
            if (isinstance(b, np.ndarray) and b.ndim == 1) and isinstance(a, VecOpInfo) and a.role == "test":
                return a.dot_const(b)
        # Dot product between a basis field and a numerical vector
        if isinstance(a, (VecOpInfo)) and (isinstance(b, np.ndarray) and b.ndim == 1): 
            return a.dot_const(b)
        elif isinstance(b, (VecOpInfo)) and (isinstance(a, np.ndarray) and a.ndim == 1): 
            return b.dot_const(a)
        elif isinstance(a, (GradOpInfo)) and (isinstance(b, np.ndarray) and b.ndim == 1):
            result = a.dot_vec(b)
            # print(f"visit dot: GradOpInfo . result: {result}, result shape: {result.shape}, role: {getattr(result, 'role', None)}")
            return result
        elif isinstance(b, (GradOpInfo)) and (isinstance(a, np.ndarray) and a.ndim == 1):
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
        if isinstance(a, GradOpInfo) and (a.role in {"trial", "function", "test"}) \
        and isinstance(b, GradOpInfo) and (b.role in {"trial", "function", "test"}):
            return a.dot(b)
        
        # --- Hessian · vector (right) and vector · Hessian (left) -------------
        # Accept geometric constant vectors (facet normals) or plain numpy 1D vectors.
        if isinstance(a, HessOpInfo) and (
            (isinstance(b, np.ndarray) and b.ndim == 1) or (isinstance(b, VecOpInfo) and role_b in {"function",None,"vector"})
        ):
            return a.dot_right(b)

        if isinstance(b, HessOpInfo) and (
            (isinstance(a, np.ndarray) and a.ndim == 1) or (isinstance(a, VecOpInfo) and role_a in {"function",None,"vector"})
        ):
            return b.dot_left(a)
        
        # Both are numerical vectors (RHS)
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray): return np.dot(a,b)

        raise TypeError(f"Unsupported dot product LHS '{n.a} . {n.b}'"
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
            if isinstance(a, (HessOpInfo, VecOpInfo, GradOpInfo)) and isinstance(b, (HessOpInfo, VecOpInfo, GradOpInfo)):
                return a.inner(b)  # returns (n,) vector
            # ---- Numeric tensor with Grad basis on RHS ----
            if isinstance(a, ( VecOpInfo)) and isinstance(b, (np.ndarray, VecOpInfo)):
                return a.inner(b)  # returns (n,) vector
            if isinstance(b, ( VecOpInfo)) and isinstance(a, (np.ndarray, VecOpInfo)):
                return b.inner(a)
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
    
    def _union_element_dofs(self, eid: int, fields: list[str]) -> np.ndarray:
        """
        Concatenate per-field element DOFs in the same order `fields` are visited.
        This matches the mixed-element local block layout used by the integrand.
        """
        return np.concatenate(
            [np.asarray(self.dh.element_dofs(f, eid), dtype=int) for f in fields],
            axis=0
        )

    def _scatter_local(self, eid: int, loc, fields, matvec, rhs: bool) -> None:
        """
        Scatter a local mixed-element block to the global vector/matrix.

        If `fields` is None, uses ALL per-element DOFs as returned by
        `DofHandler.get_elemental_dofs(eid)` so the index grid matches the
        full local block returned by the visitor (common for mixed integrands).

        If `fields` is a list of field names, concatenates their DOFs in that
        order.

        Parameters
        ----------
        eid : int
            Element id.
        loc : ndarray
            Local vector (n) or matrix (n,n) to add.
        fields : list[str] | None
            Field subset to scatter (None → all fields on the element).
        matvec : scipy.sparse matrix or ndarray
            Global accumulation target.
        rhs : bool
            True for vector assembly; False for matrix.
        """
        import numpy as np

        if fields is None:
            gdofs = np.asarray(self.dh.get_elemental_dofs(eid), dtype=int)
        else:
            gdofs = np.concatenate(
                [np.asarray(self.dh.element_dofs(f, eid), dtype=int) for f in fields],
                axis=0
            )
        if rhs:
            np.add.at(matvec, gdofs, loc)
        else:
            r, c = np.meshgrid(gdofs, gdofs, indexing="ij")
            matvec[r, c] += loc

    
    def _fields_for(self, integrand) -> list[str]:
        """
        Robust field discovery for an integrand:
        _all_fields ∪ fields from (trial,test), or all handler fields as fallback.
        Keeps a stable, sorted order.
        """
        fields = set(_all_fields(integrand))
        tr, te = _trial_test(integrand)
        for tt in (tr, te):
            if tt is None:
                continue
            if hasattr(tt, "field_names"):
                fields.update(tt.field_names)
            elif hasattr(tt, "field_name"):
                fields.add(tt.field_name)
        if not fields:
            fields = set(getattr(self.dh, "field_names", []))
        return sorted(fields)
    
    def _clear_sided_ctx(self):
        """Remove any interface/ghost residue before a volume loop."""
        for k in ("basis_values", "pos_map_by_field", "neg_map_by_field",
                "pos_eid", "neg_eid", "side", "normal"):
            self.ctx.pop(k, None)
        # We keep 'phi_val' and 'measure_side' managed by the volume routine itself.
    def _assemble_volume(self, integral: Integral, matvec):
        # if a level-set was attached to dx → do cut-cell assembly in Python path
        if getattr(integral.measure, "level_set", None) is not None:
            # CUT-VOLUME integral
            if self.backend == "jit":
                self._assemble_volume_cut_jit(integral, matvec)
            else:
                self._clear_sided_ctx()
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
        on_facet = integral.measure.on_facet
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
            self.dh, self.me, on_facet=on_facet
        )
        derivs, need_hess, need_o3, need_o4 = self._find_req_derivs(integral, runner)


        # 4. Build the static arguments required by this specific kernel.
        # We build these fresh every time to avoid caching collisions.
        q_order = self._find_q_order(integral)
        deformation = getattr(integral.measure, "deformation", None)

        # (A) Get full-mesh geometric factors and then SLICE them for the subset.
        geo_args_all = self.dh.precompute_geometric_factors(q_order, 
                                                            need_hess=need_hess, need_o3=need_o3, need_o4=need_o4,
                                                            deformation=deformation)
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
        """
        Assemble a standard volume integral (no level-set clipping).

        Respects:
        - measure.defined_on  : optional element selection (BitSet / ids / mask / tag / int)
        - mixed-element layout: local blocks are full-size (me.n_dofs_local)
        - curved geometry     : quadrature order inflated by mesh.poly_order
        """
        import numpy as np
        from pycutfem.fem import transform
        from pycutfem.ufl.helpers import normalize_elem_ids

        mesh   = self.me.mesh
        rhs    = self.ctx.get("rhs", False)
        fields = self._fields_for(integral.integrand)
        self.ctx["is_interface"] = False
        self.ctx["is_ghost"] = False
        self.ctx["measure_side"] = None          # no LS side here

        logger.info(f"Assembling volume integral: {integral}")

        # --- element restriction via 'defined_on' (BitSet / mask / ids / tag / int)
        allowed = normalize_elem_ids(mesh, getattr(integral.measure, "defined_on", None))
        if allowed is None:
            elem_ids = range(len(mesh.elements_list))
        else:
            elem_ids = [int(e) for e in allowed]

        # --- geometry-aware quadrature for curved mapping (Qp_geo, p_geo>=1)
        q_base  = int(self._find_q_order(integral))
        p_geo   = int(getattr(mesh, "poly_order", 1))
        q_infl  = 2 * max(0, p_geo - 1)
        q_order = q_base + q_infl

        qp, qw = volume(mesh.element_type, q_order)

        # local block shape == full mixed-element local size (matches visitor output)
        loc_shape = (self.me.n_dofs_local,) if rhs else (self.me.n_dofs_local, self.me.n_dofs_local)

        for eid in elem_ids:
            loc = np.zeros(loc_shape, dtype=float)

            for (xi, eta), w in zip(qp, qw):
                J     = transform.jacobian(mesh, eid, (xi, eta))
                detJ  = abs(np.linalg.det(J))
                Ji    = np.linalg.inv(J)

                # cache basis/grad for exactly the fields used by the integrand
                self._basis_cache.clear(); self._coeff_cache.clear(); self._collapsed_cache.clear()
                for f in fields:
                    self._basis_cache[f] = {
                        "val" : self.me.basis(f, xi, eta),
                        "grad": self.me.grad_basis(f, xi, eta) @ Ji
                    }

                self.ctx["eid"]    = eid
                self.ctx["x_phys"] = transform.x_mapping(mesh, eid, (xi, eta))

                integrand_val = self._visit(integral.integrand)
                loc += (w * detJ) * integrand_val

            # full-size local block → scatter against ALL element DOFs
            self._scatter_local(eid, loc, fields=None, matvec=matvec, rhs=rhs)

        # cleanup
        for k in ("eid", "x_phys", "is_interface"):
            self.ctx.pop(k, None)

    
    
    def _assemble_interface_python(self, intg: Integral, matvec):
        from pycutfem.integration.quadrature import  curved_line_quadrature_batch
        try:
            from pycutfem.integration.quadrature import isoparam_interface_line_quadrature_batch as _iso_ifc
        except Exception:
            _iso_ifc = None
        from pycutfem.fem import transform
        import numpy as np, logging, time as _time
        log = logging.getLogger(__name__)

        rhs = bool(self.ctx.get('rhs', False))
        log.info(f"Assembling interface integral: {intg}, is_rhs={rhs}")
        level_set = intg.measure.level_set
        if level_set is None:
            raise ValueError("dInterface measure requires a level_set.")

        deformation = getattr(intg.measure, "deformation", None)
        mesh, me, dh = self.me.mesh, self.me, self.dh

        # context: we are on Γ
        self.ctx['is_interface'] = True
        self.ctx['is_ghost']     = False
        md = intg.measure.metadata or {}
        side_md = md.get('side', None)
        if side_md in ('+', '-'):
            self.ctx['measure_side'] = side_md

        qdeg   = self._find_q_order(intg)
        p_geo  = int(getattr(mesh, "poly_order", 1))
        qdeg  += 2 * max(0, p_geo - 1)
        fields = self._fields_for(intg.integrand)
        nseg   = int(md.get("nseg", max(3, p_geo + qdeg//2)))
        ref_geom = transform.get_reference(mesh.element_type, p_geo)

        # scalar‑functional?
        trial, test = _trial_test(intg.integrand)
        hook = self._hook_for(intg.integrand)
        is_functional = (hook is not None) and not (trial or test)
        if is_functional:
            self.ctx.setdefault('scalar_results', {}).setdefault(hook['name'], 0.0)

        def _is_valid(elem):
            return getattr(elem, 'tag', None) == 'cut' and len(getattr(elem, 'interface_pts', ())) == 2

        # gather interface elements
        eids, P0, P1 = [], [], []
        for elem in mesh.elements_list:
            if _is_valid(elem):
                eids.append(int(elem.id))
                p0, p1 = elem.interface_pts
                P0.append(np.asarray(p0, float)); P1.append(np.asarray(p1, float))
        if not eids:
            return

        eids  = np.asarray(eids, dtype=int)
        P0    = np.asarray(P0, float)
        P1    = np.asarray(P1, float)

        # per‑element global dofs & maps
        gdofs_per_e = [np.asarray(dh.get_elemental_dofs(e), dtype=int) for e in eids]
        maps_by_field = [_hfa.build_field_union_maps(dh, fields, int(e), int(e), gdofs_per_e[i])
                        for i, e in enumerate(eids)]
        masks_by_field = []
        for i, eid in enumerate(eids):
            try:
                masks_by_field.append(_hfa.build_side_masks_by_field(dh, fields, int(eid), level_set, tol=SIDE.tol))
            except Exception:
                masks_by_field.append(({}, {}))

        # one iso call for all Γ segments: return reference coords + tangents
        t0 = _time.perf_counter()
        if _iso_ifc is not None:
            qb, wb, tb, rb = _iso_ifc(level_set, P0, P1,
                                    p=p_geo, order=qdeg, project_steps=3, tol=SIDE.tol,
                                    mesh=mesh, eids=eids,
                                    return_tangent=True, return_qref=True)
            qpts_all, qwts_all, that_all, qref_all = qb, wb, tb, rb
        else:
            # fallback: polyline rule (no qref/tangent); will use inverse map — slower
            qpts, qwts = curved_line_quadrature_batch(level_set, P0, P1, order=qdeg, nseg=nseg, project_steps=3, tol=SIDE.tol)
            qpts_all, qwts_all, that_all, qref_all = qpts, qwts, None, None
        t_iso = _time.perf_counter() - t0

        # micro helpers (avoid np.linalg.inv per QP)
        def _solve2x2(A, b):
            # solves A v = b (both 2d); returns v
            a, b0 = float(A[0,0]), float(A[0,1])
            c, d  = float(A[1,0]), float(A[1,1])
            det   = a*d - b0*c
            invd  = 1.0/(det + 1e-300)
            return np.array([( d*b[0] - b0*b[1]) * invd,
                            (-c*b[0] +  a*b[1]) * invd], float)

        # main per‑element loop
        t_qploop = 0.0
        try:
            t1 = _time.perf_counter()
            for ei, eid in enumerate(eids):
                self.ctx['eid']     = int(eid)
                self.ctx['pos_eid'] = int(eid)
                self.ctx['neg_eid'] = int(eid)

                global_dofs = gdofs_per_e[ei]
                pos_map_by_field, neg_map_by_field = maps_by_field[ei]
                try:
                    pos_mask_by_field, neg_mask_by_field = masks_by_field[ei]
                except Exception:
                    pos_mask_by_field, neg_mask_by_field = {}, {}

                # deformation locals (avoid repeat lookups)
                if deformation is not None:
                    conn = np.asarray(mesh.elements_connectivity[int(eid)], dtype=int)
                    Uloc = np.asarray(deformation.node_displacements[conn], float)
                ref_shape = ref_geom.shape
                ref_grad  = ref_geom.grad
                x_map     = transform.x_mapping
                jac_fun   = transform.jacobian

                # local accumulator
                if is_functional:
                    acc = 0.0
                else:
                    n = len(global_dofs)
                    acc = np.zeros(n, float) if rhs else np.zeros((n, n), float)

                nQ = qpts_all.shape[1]
                for iq in range(nQ):
                    xq = qpts_all[ei, iq, :]
                    w  = float(qwts_all[ei, iq])

                    if isinstance(qref_all, np.ndarray):
                        xi = float(qref_all[ei, iq, 0]); eta = float(qref_all[ei, iq, 1])
                    else:
                        xi, eta = transform.inverse_mapping(mesh, int(eid), np.asarray(xq, float))  # slow fallback

                    # geometric Jacobian
                    Jg = jac_fun(mesh, int(eid), (xi, eta))

                    # weight stretch & total Jacobian for basis mapping
                    if deformation is not None:
                        dN   = np.asarray(ref_grad(xi, eta), float)           # (nloc,2)
                        Jd   = Uloc.T @ dN                                    # (2,2)
                        Jt   = Jg + Jd

                        # stretch factor = || (I + G_ref Jg^{-1}) tau_hat ||
                        tau  = that_all[ei, iq, :] if isinstance(that_all, np.ndarray) else np.array([1.0, 0.0], float)
                        v    = _solve2x2(Jg, tau)                             # v = Jg^{-1} tau
                        Ftau = tau + (Uloc.T @ dN) @ v                        # Uloc.T@dN == G_ref
                        w_eff = w * float(np.linalg.norm(Ftau))

                        # inverse for gradient mapping (Jt^{-1})
                        a, b0 = float(Jt[0,0]), float(Jt[0,1])
                        c, d  = float(Jt[1,0]), float(Jt[1,1])
                        det_t = a*d - b0*c
                        invd  = 1.0/(det_t + 1e-300)
                        Ji    = np.array([[ d, -b0], [-c, a]], float) * invd
                        # deformed point
                        y = x_map(mesh, int(eid), (xi, eta)) + (np.asarray(ref_shape(xi, eta), float) @ Uloc)
                    else:
                        Ji    = np.array(np.linalg.inv(Jg), float)            # tiny cost w/o deformation
                        w_eff = w
                        y     = np.asarray(xq, float)

                    # normal at Γ (from φ)
                    if hasattr(level_set, 'gradient_on_element'):
                        g_here = np.asarray(level_set.gradient_on_element(int(eid), (xi, eta)), float)
                    else:
                        g_here = np.asarray(level_set.gradient(np.asarray(y, float)), float)
                    nrm = float(np.linalg.norm(g_here)) + 1e-30
                    self.ctx['normal'] = g_here / nrm

                    # basis cache at (xi,eta)
                    self.ctx['basis_values'] = {"+": {}, "-": {}}
                    if fields:
                        for f in fields:
                            vrow = me.basis(f, xi, eta)
                            Gtab = me.grad_basis(f, xi, eta) @ Ji
                            slotp = self.ctx['basis_values']['+'].setdefault(f, {})
                            slotn = self.ctx['basis_values']['-'].setdefault(f, {})
                            slotp[(0,0)] = vrow; slotn[(0,0)] = vrow
                            slotp[(1,0)] = Gtab[:, 0]; slotn[(1,0)] = Gtab[:, 0]
                            slotp[(0,1)] = Gtab[:, 1]; slotn[(0,1)] = Gtab[:, 1]

                    # context maps/masks
                    self.ctx['x_phys']           = np.asarray(y, float)
                    self.ctx['phi_val']          = 0.0
                    self.ctx['global_dofs']      = global_dofs
                    self.ctx['pos_map']          = np.arange(len(global_dofs), dtype=int)
                    self.ctx['neg_map']          = self.ctx['pos_map']
                    self.ctx['pos_map_by_field'] = pos_map_by_field
                    self.ctx['neg_map_by_field'] = neg_map_by_field
                    self.ctx['pos_mask_by_field']= pos_mask_by_field
                    self.ctx['neg_mask_by_field']= neg_mask_by_field

                    # evaluate & accumulate
                    val = self._visit(intg.integrand)
                    if is_functional:
                        arr = np.asarray(val)
                        acc += w_eff * float(arr if arr.ndim == 0 else arr.sum())
                    else:
                        arr = np.asarray(getattr(val, 'data', val))
                        if rhs and arr.ndim == 2 and 1 in arr.shape:
                            arr = arr.reshape(-1)
                        acc += w_eff * arr

                # scatter
                if is_functional:
                    self.ctx['scalar_results'][hook['name']] += float(acc)
                else:
                    if rhs:
                        np.add.at(matvec, global_dofs, acc)
                    else:
                        rr, cc = np.meshgrid(global_dofs, global_dofs, indexing='ij')
                        matvec[rr, cc] += acc
            t_qploop = _time.perf_counter() - t1
        finally:
            for k in ("basis_values", "global_dofs", "pos_map", "neg_map", "pos_map_by_field",
                    "neg_map_by_field", "pos_mask_by_field", "neg_mask_by_field",
                    "eid", "pos_eid", "neg_eid", "is_interface", "x_phys", "phi_val", "normal"):
                self.ctx.pop(k, None)

        if bool((intg.measure.metadata or {}).get('profile', False)) or os.getenv('PYCUTFEM_PROFILE_INTERFACE', '').lower() in {'1','true','yes'}:
            log.info(f"[PC] interface profile: total={(t_iso+t_qploop):.4f}s, iso_setup={t_iso:.4f}s, qp_loop={t_qploop:.4f}s, elems={len(eids)}, q={qdeg}")

  

    
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
        logger.info(f"Assembling ghost edge integral: {intg}")

        # ---- derivative orders we need (robust) -------------------------
        md      = intg.measure.metadata or {}
        derivs  = set(md.get("derivs", set())) | set(required_multi_indices(intg.integrand))
        derivs |= {(0, 0), (1, 0), (0, 1)}  # always have value and ∂x, ∂y
        self.ctx['is_interface'] = False
        self.ctx['is_ghost'] = True

        # Close up to max total order (covers Hessian cross-terms etc.)
        max_total = max((ox + oy for (ox, oy) in derivs), default=0)
        for p in range(max_total + 1):
            for ox in range(p + 1):
                derivs.add((ox, p - ox))

        # ---- which ghost edges ------------------------------------------
        defined = intg.measure.defined_on
        edge_ids = normalize_edge_ids(mesh, defined)
        if edge_ids is None:
            # default to the 'ghost' bitset when user didn't pass one
            bs = mesh.edge_bitset("ghost")
            edge_ids = bs.to_indices() if hasattr(bs, "to_indices") else list(bs)
        edge_ids = [int(e) for e in edge_ids]

        # level set required to choose (+)/(–)
        level_set = getattr(intg.measure, "level_set", None)
        if level_set is None:
            raise ValueError("dGhost measure requires a 'level_set' callable.")

        # Quadrature degree: safe upper bound (+ light inflation for curved geometry)
        qdeg_base = max(self._find_q_order(intg), 2 * max_total + 4)
        p_geo     = int(getattr(mesh, "poly_order", 1))
        qdeg      = qdeg_base + max(0, p_geo - 1)

        # Fields needed (fallbacks if tree-walk misses some due to Jump/Pos/Neg)
        fields = set(self._fields_for(intg.integrand))
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

                def choose_pos_neg_ids():
                    cL = np.asarray(mesh.elements_list[e.left ].centroid())
                    cR = np.asarray(mesh.elements_list[e.right].centroid())
                    phiL = float(level_set(cL))
                    phiR = float(level_set(cR))
                    if SIDE.is_pos(phiL) and not SIDE.is_pos(phiR):
                        return e.left, e.right
                    if SIDE.is_pos(phiR) and not SIDE.is_pos(phiL):
                        return e.right, e.left
                    # fallback: larger φ is '+'
                    return (e.left, e.right) if phiL >= phiR else (e.right, e.left)


                pos_eid, neg_eid = choose_pos_neg_ids()
                def get_normal_vec(e):
                    p0, p1 = mesh.nodes_x_y_pos[list(e.nodes)]
                    tangent = p1 - p0
                    magnitude = max([np.linalg.norm(tangent) , 1e-16])
                    unit_tangent = tangent / magnitude
                    unit_normal = np.array([unit_tangent[1], - unit_tangent[0]], float)
                    cpos = np.asarray(mesh.elements_list[pos_eid].centroid())
                    cneg = np.asarray(mesh.elements_list[neg_eid].centroid())
                    if np.dot(unit_normal, cpos - cneg) < 0.0:
                        unit_normal = -unit_normal
                    return unit_normal

                # Edge quadrature (physical) and oriented normal (– → +)
                p0, p1 = mesh.nodes_x_y_pos[list(e.nodes)]
                qpts_phys, qwts = line_quadrature(p0, p1, qdeg)

                normal_vec = get_normal_vec(e)
                

                # Local union DOFs / accumulator
                 # Always compute union layout & maps; accumulator depends on functional vs mat/vec
                pos_dofs = self.dh.get_elemental_dofs(pos_eid)
                neg_dofs = self.dh.get_elemental_dofs(neg_eid)
                global_dofs = np.unique(np.concatenate([pos_dofs, neg_dofs]))
                pos_map = np.searchsorted(global_dofs, pos_dofs)
                neg_map = np.searchsorted(global_dofs, neg_dofs)
                pos_map_by_field, neg_map_by_field = _hfa.build_field_union_maps(
                    self.dh, fields, pos_eid, neg_eid, global_dofs
                )
                loc_acc = 0.0 if is_functional else (
                    np.zeros(len(global_dofs)) if rhs
                    else np.zeros((len(global_dofs), len(global_dofs)))
                )

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
                    "pos_map_by_field", "neg_map_by_field", "is_interface"):
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
        on_facet = intg.measure.on_facet

        if cut_bs.cardinality() == 0:          # nothing to do
            return

        # ------------------------------------------------------------------
        # 2. Pre‑compute geometry for the interface elements only
        # ------------------------------------------------------------------
        qdeg      = self._find_q_order(intg)
        level_set = intg.measure.level_set
        if level_set is None:
            raise ValueError("dInterface measure requires a level_set.")


        runner, ir = self._compile_backend(intg.integrand, dh, me, on_facet=on_facet)
        derivs, need_hess, need_o3, need_o4 = self._find_req_derivs(intg, runner)
        geo = dh.precompute_interface_factors(cut_bs, qdeg, level_set, 
                                              need_hess=need_hess, need_o3=need_o3, need_o4=need_o4,
                                              deformation=getattr(intg.measure, "deformation", None))
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
            b = geo.get(f"b_{fld}")  # (nE, nQ, n_union) value table on interface
            if b is not None:
                geo.setdefault(f"r00_{fld}_pos", b)
                geo.setdefault(f"r00_{fld}_neg", b)

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
        on_facet = intg.measure.on_facet
        level_set = getattr(intg.measure, 'level_set', None)
        if level_set is None:
            raise ValueError("dGhost measure requires a 'level_set' callable.")

        if edge_ids.cardinality() == 0:
            raise ValueError("No ghost edges found for the integral.")

        # 2) Compile kernel FIRST (to know exact static params it will require)
        runner, ir = self._compile_backend(intg.integrand, dh, me, on_facet=on_facet)
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
        self.ctx['is_ghost'] = False
        self.ctx['is_interface'] = False
        if is_functional:
            self.ctx.setdefault("scalar_results", {})[hook["name"]] = 0.0

        # decide which edges we visit -------------------------------------
        defined = intg.measure.defined_on
        edge_ids = (defined.to_indices() if defined is not None
                    else np.fromiter((e.right is None for e in mesh.edges_list), bool).nonzero()[0])

        fields = self._fields_for(intg.integrand)

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
        on_facet = intg.measure.on_facet
        if edge_set.cardinality() == 0:
            raise ValueError(f"[Assembler: boundary edge JIT] No edges defined for {intg.measure.tag}.")
        else:
            print(f"Found boundary edges: {edge_set.cardinality()}")

            

        # 2. geometry tables ------------------------------------------
        qdeg = self._find_q_order(intg)
        print(f"[Assembler: boundary edge JIT] Using quadrature degree: {qdeg}")
        # 3. kernel compilation ---------------------------------------
        runner, ir = self._compile_backend(intg.integrand, dh, me, on_facet=on_facet)
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

    
    


    def _assemble_volume_cut_python(self, integral, matvec):
        """
        Assemble a volume integral restricted by a level set:
            dx(level_set=phi, metadata={'side': '+/-', 'defined_on': <elem sel>})

        • Full cells on the requested side: reference quadrature × |detJ|
        • Cut cells: physical-space sub-triangle quadrature (weights already include area)
        • Pure functionals (no test/trial) are accumulated as scalars into ctx['scalar_results'][hook['name']]
        """
        self._clear_sided_ctx()

        from pycutfem.fem import transform
        from pycutfem.ufl.helpers import normalize_elem_ids, _trial_test
        from pycutfem.integration.quadrature import volume as _vol_rule
        from pycutfem.ufl.helpers_geom import (
            phi_eval, corner_tris, clip_triangle_to_side,
            fan_triangulate, map_ref_tri_to_phys,
            clip_triangle_to_side_pn,
            curved_subcell_quadrature_for_cut_triangle
        )

        mesh      = self.me.mesh
        level_set = integral.measure.level_set
        deformation = getattr(integral.measure, "deformation", None)
        if level_set is None:
            raise ValueError("Cut-cell volume assembly requires measure.level_set")

        side   = integral.measure.metadata.get('side', '')  # '+' → φ≥0, '-' → φ≤0
        rhs    = self.ctx.get("rhs", False)
        fields = self._fields_for(integral.integrand)
        # fields     = _filter_fields_by_measure_side(fields_all, side)
        # print(f"="*50)
        # print(f"fields for integral: {fields}, side: {side}, rhs: {rhs}")
        # print(f"-"*50)
        # print(f"Integral: {integral}")
        # print(f"="*50)
        # raise NotImplementedError("Cut-cell volume assembly not implemented in Python backend.")

        # --- functional detection (structure-based) ---
        trial, test = _trial_test(integral.integrand)
        hook = self._hook_for(integral.integrand)
        is_pure_functional = (trial is None) and (test is None)
        accumulate_scalar  = is_pure_functional and (hook is not None)
        if accumulate_scalar:
            self.ctx.setdefault('scalar_results', {}).setdefault(hook['name'], 0.0)

        # indices (element-local union ordering) for the chosen fields, in order
        if fields:
            fld_local_idx = np.concatenate([
                np.arange(self.me.component_dof_slices[f].start,
                        self.me.component_dof_slices[f].stop, dtype=int)
                for f in fields
            ])
            # self.ctx["global_dofs"] = fld_local_idx                    # ← align component rows
        else:
            fld_local_idx = np.array([], dtype=int)

        # context flags for visitors
        self.ctx["is_interface"] = False
        self.ctx["is_ghost"]     = False
        # self.ctx["measure_side"] = side
        # default φ sign (used on full cells)
        self.ctx["phi_val"]      = 1.0 if side == '+' else -1.0

        # --- element restriction via 'defined_on' (BitSet / mask / ids / tag / int)
        allowed = normalize_elem_ids(mesh, getattr(integral.measure, "defined_on", None))
        allowed = set(allowed) if allowed is not None else None

        # --- classify against φ, then restrict by 'allowed'
        inside_ids, outside_ids, cut_ids = mesh.classify_elements(level_set)
        if allowed is not None:
            inside_ids  = [e for e in inside_ids  if e in allowed]
            outside_ids = [e for e in outside_ids if e in allowed]
            cut_ids     = [e for e in cut_ids     if e in allowed]

        # print(f"Volume integral on {len(inside_ids)} inside,"
        #       f" {len(outside_ids)} outside,"
        #       f" {len(cut_ids)} cut elements (after 'defined_on' filter).")
        # --- quadrature orders (account for geometry order)
        q_base  = int(self._find_q_order(integral))
        p_geo   = int(getattr(mesh, "poly_order", 1))
        q_infl  = 2 * max(0, p_geo - 1)
        q_order = q_base + q_infl

        # reference rule on parent element
        qp_ref, qw_ref = _vol_rule(mesh.element_type, q_order)
        # reference rule on triangles (for fan-triangulated polygons)
        qp_tri, qw_tri = _vol_rule("tri", q_order)
        # high-order reference for geometry/def
        ref_geom = transform.get_reference(mesh.element_type, p_geo)

        # local block shape for FULL cells (union-sized; visitor returns union size)
        loc_shape_full = (self.me.n_dofs_local,) if rhs else (self.me.n_dofs_local, self.me.n_dofs_local)
        def phi_list_3(level_set, V, v_ids):
            if hasattr(level_set, "dh") and hasattr(level_set, "field"):
                node_map = level_set.dh.dof_map.get(level_set.field, {})
                g2l      = getattr(level_set._f, "_g2l", {})
                nv       = level_set._f.nodal_values
                v_phi    = np.empty(3, float)
                for j, nid in enumerate(v_ids):  # mesh node ids for the tri’s vertices
                    gd = node_map.get(int(nid))
                    if gd is not None and gd in g2l:
                        v_phi[j] = float(nv[g2l[gd]])
                    else:
                        v_phi[j] = float(phi_eval(level_set, V[j]))    # rare fallback
            else:
                v_phi = [phi_eval(level_set, V[i]) for i in range(3)]
            return v_phi
        

        # ----------------------------
        # Matrix / Vector path
        # ----------------------------

        # --- 1) FULL elements on requested side (union-sized loc, scatter fields=None)
        full_eids = outside_ids if side == '+' else inside_ids
        if is_pure_functional:
            acc = 0.0
        for eid in full_eids:
            loc = np.zeros(loc_shape_full, dtype=float)
            for (xi, eta), w in zip(qp_ref, qw_ref):
                Jg = transform.jacobian(mesh, int(eid), (float(xi), float(eta)))
                xg = transform.x_mapping(mesh, int(eid), (float(xi), float(eta)))
                if deformation is None:
                    Jt = Jg; det_t = abs(float(np.linalg.det(Jg)))
                    Ji = np.linalg.inv(Jg)
                    y  = xg
                else:
                    conn = np.asarray(mesh.elements_connectivity[int(eid)], dtype=int)
                    dN   = np.asarray(ref_geom.grad(float(xi), float(eta)), float)
                    Uloc = np.asarray(deformation.node_displacements[conn], float)
                    Jd   = Uloc.T @ dN
                    Jt   = Jg + Jd
                    det_t = abs(float(np.linalg.det(Jt)))
                    Ji = np.linalg.inv(Jt)
                    y  = xg + deformation.displacement_ref(int(eid), (float(xi), float(eta)))

                # basis/grad caches for exactly the fields used by the integrand
                self._basis_cache.clear(); self._coeff_cache.clear(); self._collapsed_cache.clear()
                for f in fields:
                    self._basis_cache[f] = {
                        "val":  self.me.basis(f, float(xi), float(eta)),
                        "grad": self.me.grad_basis(f, float(xi), float(eta)) @ Ji,
                    }

                self.ctx["eid"]    = int(eid)
                self.ctx["x_phys"] = np.asarray(y, float)

                val = self._visit(integral.integrand)
                if is_pure_functional:
                    arr = np.asarray(val.data) if hasattr(val, "data") else np.asarray(val)
                    v   = float(arr if arr.ndim == 0 else arr.sum())
                    acc += float(w) * float(det_t) * v
                else:
                    loc += float(w) * float(det_t) * val

            if not is_pure_functional:
                self._scatter_local(int(eid), loc, fields=None, matvec=matvec, rhs=rhs)

        # --- 2) CUT elements: subset-sized loc, scatter fields=fields
        if fld_local_idx.size == 0 or len(fields) == 0:
            # Nothing to assemble for cut cells if no participating fields
            for key in ("eid", "x_phys", "phi_val", "measure_side", "side"):
                self.ctx.pop(key, None)
            return

        for eid in cut_ids:
            loc = np.zeros(loc_shape_full, dtype=float)

            if mesh.element_type == 'quad':
                order_y = max(2, q_order // 2)
                order_x = max(2, q_order // 2)
                qpref, qwref = CutIntegration.straight_cut_rule_quad_ref(mesh, int(eid), level_set,
                                                                          side=side, order_y=order_y, order_x=order_x,
                                                                          tol=_INTERFACE_TOL)
                if qpref.size:
                    for (xi, eta), w in zip(qpref, qwref):
                        Jg = transform.jacobian(mesh, int(eid), (float(xi), float(eta)))
                        xg = transform.x_mapping(mesh, int(eid), (float(xi), float(eta)))
                        if deformation is None:
                            Jt = Jg; det_t = abs(float(np.linalg.det(Jg)))
                            Ji = np.linalg.inv(Jg)
                            y  = xg
                        else:
                            conn = np.asarray(mesh.elements_connectivity[int(eid)], dtype=int)
                            dN   = np.asarray(ref_geom.grad(float(xi), float(eta)), float)
                            Uloc = np.asarray(deformation.node_displacements[conn], float)
                            Jd   = Uloc.T @ dN
                            Jt   = Jg + Jd
                            det_t = abs(float(np.linalg.det(Jt)))
                            Ji = np.linalg.inv(Jt)
                            y  = xg + deformation.displacement_ref(int(eid), (float(xi), float(eta)))

                        self._basis_cache.clear(); self._coeff_cache.clear(); self._collapsed_cache.clear()
                        for f in fields:
                            self._basis_cache[f] = {
                                "val":  self.me.basis(f, float(xi), float(eta)),
                                "grad": self.me.grad_basis(f, float(xi), float(eta)) @ Ji,
                            }
                        self.ctx["eid"]    = int(eid)
                        self.ctx["x_phys"] = np.asarray(y, float)

                        val = self._visit(integral.integrand)
                        if is_pure_functional:
                            arr = np.asarray(val.data) if hasattr(val, "data") else np.asarray(val)
                            v   = float(arr if arr.ndim == 0 else arr.sum())
                            acc += float(w) * float(det_t) * v
                        elif rhs:
                            vec = np.asarray(val.data) if hasattr(val, "data") else np.asarray(val)
                            if 1 in vec.shape: vec = vec.flatten()
                            loc += float(w) * float(det_t) * vec
                        else:
                            mat = np.asarray(val.data) if hasattr(val, "data") else np.asarray(val)
                            loc += float(w) * float(det_t) * mat

                if not is_pure_functional:
                    self._scatter_local(int(eid), loc, fields=None, matvec=matvec, rhs=rhs)
                continue

            elem = mesh.elements_list[eid]
            tri_local, corner_ids = corner_tris(mesh, elem)
            for loc_tri in tri_local:
                qx, qw = curved_subcell_quadrature_for_cut_triangle(
                    mesh, eid, loc_tri, corner_ids, level_set,
                    side=side, qvol=q_order, nseg_hint=None, tol=_INTERFACE_TOL
                )
                for x_phys, w in zip(qx, qw):
                    xi, eta = transform.inverse_mapping(mesh, eid, x_phys)
                    Jg  = transform.jacobian(mesh, eid, (xi, eta))
                    if deformation is None:
                        Ji = np.linalg.inv(Jg)
                        y  = np.asarray(x_phys, float)
                        w_eff = float(w)
                    else:
                        conn = np.asarray(mesh.elements_connectivity[int(eid)], dtype=int)
                        dN   = np.asarray(ref_geom.grad(float(xi), float(eta)), float)
                        Uloc = np.asarray(deformation.node_displacements[conn], float)
                        Jd   = Uloc.T @ dN
                        Jt   = Jg + Jd
                        det_t = abs(float(np.linalg.det(Jt)))
                        det_g = abs(float(np.linalg.det(Jg))) + 1e-300
                        Ji = np.linalg.inv(Jt)
                        y  = transform.x_mapping(mesh, int(eid), (float(xi), float(eta))) \
                             + deformation.displacement_ref(int(eid), (float(xi), float(eta)))
                        w_eff = float(w) * float(det_t / det_g)

                    self._basis_cache.clear(); self._coeff_cache.clear(); self._collapsed_cache.clear()
                    for f in fields:
                        self._basis_cache[f] = {
                            "val":  self.me.basis(f, float(xi), float(eta)),
                            "grad": self.me.grad_basis(f, float(xi), float(eta)) @ Ji,
                        }
                    self.ctx["eid"]          = eid
                    self.ctx["x_phys"]       = np.asarray(y, float)
                    self.ctx["phi_val"]      = 1.0 if side == '+' else -1.0
                    self.ctx["measure_side"] = side

                    val = self._visit(integral.integrand)
                    if is_pure_functional:
                        arr = np.asarray(val.data) if hasattr(val, "data") else np.asarray(val)
                        v   = float(arr if arr.ndim == 0 else arr.sum())
                        acc += w_eff * v
                    elif rhs:
                        vec = np.asarray(val.data) if hasattr(val, "data") else np.asarray(val)
                        if 1 in vec.shape:
                            vec = vec.flatten()
                        loc += w_eff * vec
                    else:
                        mat = np.asarray(val.data) if hasattr(val, "data") else np.asarray(val)
                        loc += w_eff * mat

            if not is_pure_functional:
                self._scatter_local(eid, loc, fields=None, matvec=matvec, rhs=rhs)

        if accumulate_scalar:
            self.ctx['scalar_results'][hook['name']] += acc
        # cleanup
        for key in ("eid", "x_phys", "phi_val", "measure_side", "side"):
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
        on_facet = intg.measure.on_facet
        deformation = getattr(intg.measure, "deformation", None)
        bs            = intg.measure.defined_on

        if level_set is None:
            # This routine is for cut-volume only
            raise ValueError("Cut-cell volume assembly requires measure.level_set")

        # --- 1) classification (must be called; sets element tags & caches)
        inside_ids, outside_ids, cut_ids = mesh.classify_elements(level_set)

        # --- compile kernel once
        runner, ir = self._compile_backend(intg.integrand, dh, me, on_facet=on_facet)
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
            prebuilt["eids"]      = eids 
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
        if bs is not None:
            # Intersect with defined_on (BitSet or boolean/indices)
            try:
                allowed = np.asarray(bs.to_indices(), dtype=np.int32)
            except AttributeError:
                arr = np.asarray(bs)
                allowed = (np.nonzero(arr)[0].astype(np.int32)
                        if arr.dtype == bool else arr.astype(np.int32))
            full_ids = np.intersect1d(full_ids, allowed, assume_unique=False)

        if full_ids.size:
            # include level_set so 'phis' is populated (still fine if None)
            geo_all = dh.precompute_geometric_factors(qdeg, level_set, 
                                                      need_hess=need_hess, need_o3=need_o3, need_o4=need_o4,
                                                      deformation=deformation)

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
                "is_interface": False,
                "is_ghost": False,
                "eids": full_ids,
                "J_inv_pos": geo_all["J_inv"][full_ids],
                "J_inv_neg": geo_all["J_inv"][full_ids],
            }
            _run_subset(full_ids, prebuilt_full)

        # --- 3) cut elements → clipped triangles (physical weights); detJ := 1
        if len(cut_ids):
            # The kernel may need value and first derivative tables at least
            req_derivs = required_multi_indices(intg.integrand) | {(0, 0)}
            cut_mask   = mesh.element_bitset("cut")
            cut_bs     = (bs & cut_mask) if bs is not None else cut_mask

            geo_cut = dh.precompute_cut_volume_factors(
                cut_bs, qdeg, req_derivs, level_set,
                side=side,
                need_hess=need_hess, need_o3=need_o3, need_o4=need_o4,
                deformation=deformation
            )
            cut_eids = np.asarray(geo_cut.get("eids", []), dtype=np.int32)
            if cut_eids.size:
                # Physical subcell weights already include area → neutral detJ
                if "detJ" not in geo_cut:
                    geo_cut["detJ"] = np.ones_like(geo_cut["qw"])
                geo_cut["is_interface"] = False
                _run_subset(cut_eids, geo_cut)

        # cleanup context (if you stored any debug keys)
        for k in ("eid", "x_phys", "phi_val"):
            self.ctx.pop(k, None)
