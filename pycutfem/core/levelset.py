"""pycutfem.core.levelset
Enhanced to support multiple level-set functions via *CompositeLevelSet*.
"""
from __future__ import annotations
from typing import Sequence, Tuple, Callable
import numpy as np
from pycutfem.ufl.expressions import Function 
from pycutfem.fem import transform 

class LevelSetFunction:
    """Abstract base class"""
    def __call__(self, x: np.ndarray) -> float:
        raise NotImplementedError
    def gradient(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def evaluate_on_nodes(self, mesh) -> np.ndarray:
        return np.apply_along_axis(self, 1, mesh.nodes_x_y_pos)

class CircleLevelSet(LevelSetFunction):
    def __init__(self, center: Tuple[float,float]=(0.,0.), radius: float=1.0):
        self.center=np.asarray(center,dtype=float)
        self.radius=float(radius)
    def __call__(self, x):
        """Signed distance; works for shape (..., 2) or plain (2,)."""
        x = np.asarray(x, dtype=float)
        rel = x - self.center
        # norm along the last axis keeps the leading shape intact
        return np.linalg.norm(rel, axis=-1) - self.radius
    def gradient(self, x):
        d=np.asarray(x-self.center)
        nrm=np.linalg.norm(d)
        return d/nrm if nrm else np.zeros_like(d)

class AffineLevelSet(LevelSetFunction):
    """
    φ(x, y) = a * x + b * y + c
    Any straight line: choose (a, b, c) so that φ=0 is the line.
    """
    def __init__(self, a: float, b: float, c: float):
        self.a, self.b, self.c = float(a), float(b), float(c)

    # ---- value ------------------------------------------------------
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Works with shape (2,) or (..., 2)
        x = np.asarray(x, dtype=float)
        return self.a * x[..., 0] + self.b * x[..., 1] + self.c

    # ---- gradient ---------------------------------------------------
    def gradient(self, x: np.ndarray) -> np.ndarray:
        g = np.array([self.a, self.b])
        g = g/np.linalg.norm(g) if np.linalg.norm(g) else np.zeros_like(g)
        return g if x.ndim == 1 else np.tile(g, (x.shape[0], 1))

    # ---- optional: signed-distance normalisation --------------------
    def normalised(self):
        """Return a copy scaled so that ‖∇φ‖ = 1 (signed-distance)."""
        norm = np.hypot(self.a, self.b)
        return AffineLevelSet(self.a / norm, self.b / norm, self.c / norm)


class CompositeLevelSet(LevelSetFunction):
    """Hold several independent level‑set functions.

    Calling the composite returns **an array** of shape (n_levelsets,).
    Gradients stack to (n_levelsets, 2).
    """
    def __init__(self, levelsets: Sequence[LevelSetFunction]):
        self.levelsets=list(levelsets)
    def __call__(self, x):
        return np.array([ls(x) for ls in self.levelsets])
    def gradient(self, x):
        return np.stack([ls.gradient(x) for ls in self.levelsets])
    def evaluate_on_nodes(self, mesh):
        return np.vstack([ls.evaluate_on_nodes(mesh) for ls in self.levelsets])

# --- Numba helpers for common level sets -----------------------------------
try:
    import numba as _nb  # type: ignore
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

if _HAVE_NUMBA:
    @_nb.njit(cache=True, fastmath=True)
    def _circle_value(x, cx, cy, r):
        dx = x[0] - cx; dy = x[1] - cy
        return (dx*dx + dy*dy) ** 0.5 - r

    @_nb.njit(cache=True, fastmath=True)
    def _circle_grad(x, cx, cy):
        dx = x[0] - cx; dy = x[1] - cy
        n = (dx*dx + dy*dy) ** 0.5
        g = np.zeros(2)
        if n > 0.0:
            g[0] = dx / n; g[1] = dy / n
        return g

    @_nb.njit(cache=True, fastmath=True)
    def _affine_value(x, a, b, c):
        return a * x[0] + b * x[1] + c

    @_nb.njit(cache=True, fastmath=True)
    def _affine_unit_grad(a, b):
        n = (a*a + b*b) ** 0.5
        g = np.zeros(2)
        if n > 0.0:
            g[0] = a / n; g[1] = b / n
        return g

class LevelSetGridFunction:
    """
    A level-set represented as a FE GridFunction (P1/P2) on the mesh handled by a DofHandler.
    - Works with your existing MixedElement CG spaces (no new mesh nodes).
    - Mutable: nodal values can be updated each time step; we bump a cache token.
    - Implements the same protocol as analytic level sets: __call__, gradient, evaluate_on_nodes.
    """

    def __init__(self, dh, field: str = "phi"):
        self.dh     = dh
        self.field  = field
        self._mesh  = dh.mixed_element.mesh
        # Standalone scalar Function on the same handler/field
        self._f     = Function(name="phi", field_name=field, dof_handler=dh)
        # version → used by DofHandler._ls_fingerprint to invalidate caches
        self._version = 0
        self.cache_token = ("lsgrid", id(self._mesh), field, self._version)  # <- recognized by _ls_fingerprint
        # Precompute: which global DOFs of 'phi' coincide with mesh nodes
        self._node_dofs = np.asarray(self.dh.get_field_dofs_on_nodes(field), int)  # CG only
        # Local accelerator: global->local index mapping inside _f
        self._g2l = getattr(self._f, "_g2l", {})

    # --------------------- population / update ---------------------
    def interpolate(self, fun_xy: Callable[[float, float], float]) -> None:
        """Fill nodal values by evaluating fun(x,y) at the 'phi' DOF coordinates."""
        gdofs = np.asarray(self.dh.get_field_slice(self.field), int)
        XY    = self.dh.get_dof_coords(self.field)                # (nphi, 2)
        vals  = np.array([float(fun_xy(float(x), float(y))) for (x, y) in XY], float)
        self._f.set_nodal_values(gdofs, vals)
        self._bump()

    def set_from_array(self, values: np.ndarray) -> None:
        """Set nodal values in the 'phi' field order (same length as get_field_slice)."""
        gdofs = np.asarray(self.dh.get_field_slice(self.field), int)
        if len(values) != len(gdofs):
            raise ValueError("Size mismatch: values vs 'phi' field DOFs.")
        self._f.set_nodal_values(gdofs, np.asarray(values, float))
        self._bump()

    def nodal_values(self) -> np.ndarray:
        """Return the (compact) data array for 'phi'."""
        return self._f.nodal_values

    def commit(self, tol: float = 1e-12) -> None:
        """
        Bump version, reclassify cut/inside/outside elements & edges,
        rebuild interface segments.  Call this once per time step (after updating φ).
        """
        self._bump()
        self.dh.classify_from_levelset(self, tol=tol)  # uses evaluate_on_nodes(...) under the hood
        # any cached assembler geometry keyed by the ls fingerprint will be naturally invalidated

    def _bump(self):
        self._version += 1
        self.cache_token = ("lsgrid", id(self._mesh), self.field, self._version)

    # -------------------- level-set protocol -----------------------
    def evaluate_on_nodes(self, mesh) -> np.ndarray:
        out = np.full(len(mesh.nodes_list), np.nan, float)

        # fast path: nodes that are DOFs
        node_map = self.dh.dof_map.get(self.field, {})  # {mesh_node_id -> global_dof}
        for nid, gd in node_map.items():
            li = self._g2l.get(int(gd))
            if li is not None:
                out[int(nid)] = float(self._f.nodal_values[int(li)])

        # fallback: evaluate at remaining nodes via FE interpolation
        if np.any(np.isnan(out)):
            X = np.asarray(mesh.nodes_x_y_pos, float)
            for nid in np.where(np.isnan(out))[0]:
                out[nid] = float(self(X[nid]))
        return out


    def __call__(self, x: np.ndarray) -> float | np.ndarray:
        """
        Evaluate φ(x) by FE interpolation. This is mainly used at quadrature points.
        We do a simple 'find an owner element' via inverse mapping (robust in debug mode).
        The assembler uses a faster path (value_on_element) so perf is OK.
        """
        x = np.asarray(x, float)
        if x.ndim == 1:
            eid = self._find_element_containing(x)
            xi, eta = transform.inverse_mapping(self._mesh, int(eid), x)
            return self.value_on_element(int(eid), (float(xi), float(eta)))
        # vectorized
        out = np.empty(x.shape[0], float)
        for i, pt in enumerate(x):
            eid = self._find_element_containing(pt)
            xi, eta = transform.inverse_mapping(self._mesh, int(eid), pt)
            out[i] = self.value_on_element(int(eid), (float(xi), float(eta)))
        return out

    def value_on_element(self, eid: int, xi_eta: Tuple[float, float]) -> float:
        """Fast path when element id & (xi,eta) are known."""
        xi, eta = xi_eta
        N_hat   = self.dh.mixed_element._eval_scalar_basis(self.field, xi, eta)  # shape fn (union-sized → 'phi' slice)
        gd      = np.asarray(self.dh.element_maps[self.field][int(eid)], int)     # element's 'phi' DOFs
        vals    = self._f.get_nodal_values(gd)                                    # compact → padded to element layout (zeros elsewhere)
        return float(N_hat @ vals)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        if x.ndim != 1:
            raise NotImplementedError("gradient(x) expects a single point.")
        eid = self._find_element_containing(x)
        xi, eta = transform.inverse_mapping(self._mesh, int(eid), x)

        # reference grads of scalar basis for this field
        G_ref_full = self.dh.mixed_element._eval_scalar_grad(self.field, float(xi), float(eta))  # (n_field_loc, 2)
        G_ref = G_ref_full  # already field-local

        # element Jacobian and inverse
        J  = transform.jacobian(self._mesh, int(eid), (float(xi), float(eta)))
        A  = np.linalg.inv(J)  # = d(ξ,η)/d(x,y)

        # local φ DOFs and their coefficients
        gd   = np.asarray(self.dh.element_maps[self.field][int(eid)], int)
        vals = self._f.get_nodal_values(gd)  # (n_field_loc,)

        # row-form push-forward: ∇̂φ (row) → ∇φ (row)
        g_ref = vals @ G_ref        # (2,)
        return (g_ref @ A)          # (2,)


    # ---------- very simple owner search (debug-friendly) ----------
    def _find_element_containing(self, x: np.ndarray, tol = 1e-12) -> int:
        # O(nelem) robust scan; OK for Python path / debugging.
        for e in self._mesh.elements_list:
            try:
                xi, eta = transform.inverse_mapping(self._mesh, e.id, x)
                if self._mesh.element_type == "quad":
                    if -1.0-tol <= xi <= 1.0+tol and -1.0-tol <= eta <= 1.0+tol:
                        return int(e.id)
                else:
                    # tri: barycentric in reference
                    if xi >= -tol and eta >= -tol and xi+eta <= 1.0+tol:
                        return int(e.id)
            except Exception:
                pass
        # Fallback: nearest centroid (should not happen if mappings are OK)
        d = [np.linalg.norm(np.asarray(e.centroid()) - x) for e in self._mesh.elements_list]
        return int(np.argmin(d))
    def gradient_on_element(self, eid: int, xi_eta: tuple[float, float]) -> np.ndarray:
        xi, eta = xi_eta
        dN_ref  = self.dh.mixed_element._eval_scalar_grad(self.field, xi, eta)    # (nloc,2)
        gd      = self.dh.element_maps[self.field][int(eid)]
        vals    = self._f.get_nodal_values(np.asarray(gd, int))                   # (nloc,)
        J       = transform.jacobian(self._mesh, int(eid), (xi, eta))
        A       = np.linalg.inv(J)                                                # = ∂(ξ,η)/∂(x,y)
        g_ref   = vals @ dN_ref                                                   # (2,) row
        return g_ref @ A 
    def values_on_element_many(self, eid: int, xi_arr, eta_arr) -> np.ndarray:
        """
        φ at many (ξ,η) on a known element. Returns (n_qp,).
        """
        gd   = np.asarray(self.dh.element_maps[self.field][int(eid)], int)
        vals = self._f.get_nodal_values(gd)  # (n_loc,)
        N    = self.dh.mixed_element._eval_scalar_basis_many(self.field, xi_arr, eta_arr)  # (n_qp, n_loc)
        return N @ vals  # (n_qp,)

    def gradients_on_element_many(self, eid: int, xi_arr, eta_arr) -> np.ndarray:
        """
        ∇φ at many (ξ,η) on an element. Returns (n_qp, 2) in physical (row) form.
        """
        xi_arr  = np.asarray(xi_arr, float).ravel()
        eta_arr = np.asarray(eta_arr, float).ravel()
        gd      = np.asarray(self.dh.element_maps[self.field][int(eid)], int)
        vals    = self._f.get_nodal_values(gd)  # (n_loc,)
        dN_ref  = self.dh.mixed_element._eval_scalar_grad_many(self.field, xi_arr, eta_arr)  # (n_qp, n_loc, 2)

        out = np.empty((xi_arr.size, 2), dtype=float)
        # push-forward per point: f_i = g_I A^I_i  with A = J^{-1}
        for k, (xi, eta) in enumerate(zip(xi_arr, eta_arr)):
            J  = transform.jacobian(self._mesh, int(eid), (float(xi), float(eta)))
            A  = np.linalg.inv(J)
            g_ref = vals @ dN_ref[k, :, :]   # (2,)
            out[k, :] = g_ref @ A            # (2,)
        return out



def phi_eval(level_set, x_phys, *, eid=None, xi_eta=None, mesh=None):
    # Fast path: FE level set with element context
    if hasattr(level_set, "value_on_element") and (eid is not None):
        if xi_eta is None:
            if mesh is None:
                raise ValueError("phi_eval needs xi_eta or mesh to inverse-map.")
            xi_eta = transform.inverse_mapping(mesh, int(eid), np.asarray(x_phys, float))
        return level_set.value_on_element(int(eid), (float(xi_eta[0]), float(xi_eta[1])))
    # Generic fallback (analytic or FE): may do owner search
    return level_set(np.asarray(x_phys, float))
