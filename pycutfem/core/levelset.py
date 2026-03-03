"""pycutfem.core.levelset
Enhanced to support multiple level-set functions via *CompositeLevelSet*.
"""
from __future__ import annotations
from typing import Sequence, Tuple, Callable, Optional
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

class AnnulusLevelSet(LevelSetFunction):
    """
    Annulus (ring) level set centered at `center`:

        φ(x) < 0  for r_inner < ‖x-center‖ < r_outer
        φ(x) = 0  on both circles r = r_inner and r = r_outer
        φ(x) > 0  otherwise

    The value is a *piecewise* signed distance to the nearest of the two circles:
      - outside (near r_outer):  φ = r - r_outer
      - inside (near r_inner):  φ = r_inner - r

    Sign convention
    ---------------
    We use the CutFEM convention Ω⁻ = {φ<0}, Ω⁺ = {φ>0}. With this choice,
    the outward normal on Γ is n = ∇φ/||∇φ|| pointing from Ω⁻ to Ω⁺.
    """

    def __init__(
        self,
        center: Tuple[float, float] = (0.0, 0.0),
        r_inner: float = 0.25,
        r_outer: float = 0.75,
    ):
        self.center = np.asarray(center, dtype=float)
        self.r_inner = float(r_inner)
        self.r_outer = float(r_outer)
        if not (self.r_inner > 0.0 and self.r_outer > self.r_inner):
            raise ValueError("Require 0 < r_inner < r_outer.")

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        rel = x - self.center
        r = np.linalg.norm(rel, axis=-1)
        rc = 0.5 * (self.r_inner + self.r_outer)
        return np.where(r >= rc, r - self.r_outer, self.r_inner - r)

    def gradient(self, x):
        x = np.asarray(x, dtype=float)
        rel = x - self.center
        r = np.linalg.norm(rel, axis=-1)
        rc = 0.5 * (self.r_inner + self.r_outer)

        # unit radial direction (safe at r=0)
        r_safe = np.where(r == 0.0, 1.0, r)
        unit = rel / r_safe[..., None]

        # φ = r - r_outer  -> grad =  +unit
        # φ = r_inner - r  -> grad =  -unit
        sign = np.where(r >= rc, 1.0, -1.0)
        g = unit * sign[..., None]
        if x.ndim == 1:
            return g.reshape(2,)
        return g

class SuperellipseLevelSet(LevelSetFunction):
    """
    L^4 "ball" (superellipse / squircle-like):

        φ(x,y) = (|x-cx|^4 + |y-cy|^4)^(1/4) - radius

    Negative inside, positive outside.
    """

    def __init__(
        self,
        center: Tuple[float, float] = (0.0, 0.0),
        radius: float = 1.0,
    ):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        if not (self.radius > 0.0):
            raise ValueError("radius must be positive.")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        rel = x - self.center
        ax = np.abs(rel[..., 0])
        ay = np.abs(rel[..., 1])
        r44 = ax**4 + ay**4
        r41 = np.power(r44, 0.25)
        return r41 - self.radius

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        rel = x - self.center
        # Use the analytic derivative of (ax^4 + ay^4)^(1/4)
        ax = np.abs(rel[..., 0])
        ay = np.abs(rel[..., 1])
        sx = np.sign(rel[..., 0])
        sy = np.sign(rel[..., 1])
        r44 = ax**4 + ay**4
        denom = np.power(r44, 0.75)

        denom_safe = np.where(denom == 0.0, 1.0, denom)
        gx = (ax**3 * sx) / denom_safe
        gy = (ay**3 * sy) / denom_safe
        gx = np.where(denom == 0.0, 0.0, gx)
        gy = np.where(denom == 0.0, 0.0, gy)

        g = np.stack((gx, gy), axis=-1)
        if x.ndim == 1:
            return g.reshape(2,)
        return g

class AffineLevelSet(LevelSetFunction):
    """
    φ(x, y) = a * x + b * y + c
    Any straight line: choose (a, b, c) so that φ=0 is the line.
    """
    def __init__(self, a: float, b: float, c: float):
        self.a, self.b, self.c = float(a), float(b), float(c)

    @property
    def cache_token(self):
        # NOTE: `Mesh.classify_elements` caches per-level-set results using
        # either an explicit `cache_token` or the object id. For affine level
        # sets we often *mutate* the coefficients (e.g. moving internal
        # boundaries), so the token must depend on (a,b,c) to avoid stale
        # cached cut/inside/outside classifications.
        return ("affine_ls", float(self.a), float(self.b), float(self.c))

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


class BeamLevelSet(LevelSetFunction):
    """
    Axis-aligned rectangular beam:
    center = (cx, cy)
    length in x = Lb, thickness in y = Hb.
    φ < 0 inside the beam, > 0 outside.
    """

    def __init__(self, center, Lb, Hb):
        self.cx, self.cy = center
        self.hx = 0.5 * Lb
        self.hy = 0.5 * Hb
        # Used to invalidate caches when the LS changes
        self.cache_token = ("beam_rect_ref", float(self.cx), float(self.cy),
                            float(self.hx), float(self.hy))

    def __call__(self, x):
        """
        Signed distance in the L∞-sense.
        Supports both shape (2,) and (..., 2) inputs.
        """
        x = np.asarray(x, float)
        dx = (x[..., 0] - self.cx) / self.hx
        dy = (x[..., 1] - self.cy) / self.hy
        ax = np.abs(dx)
        ay = np.abs(dy)
        m = np.maximum(ax, ay)
        return m - 1.0        # <0 inside

    def gradient(self, x):
        """
        Piecewise-constant gradient pointing normal to the closest face.
        Norm is ~1/h on that face; we normalise to a unit normal.
        """
        x = np.asarray(x, float)
        dx = (x[..., 0] - self.cx) / self.hx
        dy = (x[..., 1] - self.cy) / self.hy
        ax = np.abs(dx)
        ay = np.abs(dy)

        # Broadcast-friendly normal selection
        prefer_x = ax > ay
        gx = np.where(prefer_x, np.sign(dx) / self.hx, 0.0)
        gy = np.where(prefer_x, 0.0, np.sign(dy) / self.hy)

        g = np.stack((gx, gy), axis=-1)
        nrm = np.linalg.norm(g, axis=-1, keepdims=True)
        nrm_safe = np.where(nrm == 0.0, 1.0, nrm)
        return g / nrm_safe


class RotatedBoxLevelSet(LevelSetFunction):
    """
    Rotated axis-aligned box in a local coordinate frame.

    The local box is centered at `center` with half-lengths (hx, hy) in the
    rotated frame. The signed distance is in the L∞-sense:

        φ(x) = max(|x'|/hx, |y'|/hy) - 1

    where (x', y') are coordinates rotated by `angle` about the center.
    Negative inside, positive outside.
    """

    def __init__(
        self,
        *,
        center: Tuple[float, float] = (0.0, 0.0),
        hx: float,
        hy: float,
        angle: float = 0.0,
    ):
        self.center = np.asarray(center, dtype=float)
        self.hx = float(hx)
        self.hy = float(hy)
        if not (self.hx > 0.0 and self.hy > 0.0):
            raise ValueError("hx and hy must be positive.")
        self.angle = float(angle)
        self._c = float(np.cos(self.angle))
        self._s = float(np.sin(self.angle))
        # Used to invalidate caches when the LS changes.
        self.cache_token = (
            "rotated_box_ref",
            float(self.center[0]),
            float(self.center[1]),
            float(self.hx),
            float(self.hy),
            float(self.angle),
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        rel = x - self.center
        # Rotate into the box frame (rotation by -angle).
        xp = self._c * rel[..., 0] + self._s * rel[..., 1]
        yp = -self._s * rel[..., 0] + self._c * rel[..., 1]
        ax = np.abs(xp) / self.hx
        ay = np.abs(yp) / self.hy
        return np.maximum(ax, ay) - 1.0

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Piecewise-constant unit normal pointing to the closest face.

        For points near corners the normal is ambiguous; we pick a consistent
        branch via the L∞-distance face selection.
        """
        x = np.asarray(x, dtype=float)
        rel = x - self.center
        # Rotate into the box frame (rotation by -angle).
        xp = self._c * rel[..., 0] + self._s * rel[..., 1]
        yp = -self._s * rel[..., 0] + self._c * rel[..., 1]
        ax = np.abs(xp) / self.hx
        ay = np.abs(yp) / self.hy
        prefer_x = ax > ay
        # Local frame gradient before normalisation.
        gx_p = np.where(prefer_x, np.sign(xp) / self.hx, 0.0)
        gy_p = np.where(prefer_x, 0.0, np.sign(yp) / self.hy)
        # Rotate back to global frame (rotation by +angle).
        gx = self._c * gx_p - self._s * gy_p
        gy = self._s * gx_p + self._c * gy_p
        g = np.stack((gx, gy), axis=-1)
        nrm = np.linalg.norm(g, axis=-1, keepdims=True)
        nrm_safe = np.where(nrm == 0.0, 1.0, nrm)
        out = g / nrm_safe
        if x.ndim == 1:
            return out.reshape(2,)
        return out

class ScaledLevelSet(LevelSetFunction):
    """Scale a level set by a nonzero constant.

    Notes
    -----
    Many level sets in this code base implement `gradient()` as a *unit* normal.
    Scaling a signed-distance-like function by a positive constant keeps the
    unit normal unchanged; scaling by a negative constant flips its direction.
    """

    def __init__(self, scale: float, level_set: LevelSetFunction):
        s = float(scale)
        if abs(s) <= 0.0:
            raise ValueError("scale must be nonzero.")
        self.scale = s
        self.level_set = level_set
        tok = getattr(level_set, "cache_token", None)
        self.cache_token = ("scaled_ls", float(self.scale), tok if tok is not None else int(id(level_set)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.scale * np.asarray(self.level_set(x), dtype=float)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        g = np.asarray(self.level_set.gradient(x), dtype=float)
        sgn = 1.0 if self.scale > 0.0 else -1.0
        return sgn * g


class MinLevelSet(LevelSetFunction):
    """Pointwise minimum of several level sets."""

    def __init__(self, *levelsets: LevelSetFunction):
        if len(levelsets) < 2:
            raise ValueError("MinLevelSet requires at least two level sets.")
        self.levelsets = tuple(levelsets)
        toks = []
        for ls in self.levelsets:
            tok = getattr(ls, "cache_token", None)
            toks.append(tok if tok is not None else int(id(ls)))
        self.cache_token = ("min_ls", tuple(toks))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        vals = [np.asarray(ls(x), dtype=float) for ls in self.levelsets]
        out = vals[0]
        for v in vals[1:]:
            out = np.minimum(out, v)
        return out

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            vals = [float(ls(x)) for ls in self.levelsets]
            k = int(np.argmin(np.asarray(vals, dtype=float)))
            return np.asarray(self.levelsets[k].gradient(x), dtype=float)

        vals = np.stack([np.asarray(ls(x), dtype=float) for ls in self.levelsets], axis=0)
        idx = np.argmin(vals, axis=0)  # (...,)
        grads = np.stack([np.asarray(ls.gradient(x), dtype=float) for ls in self.levelsets], axis=0)  # (k,...,2)
        sel = np.take_along_axis(grads, idx[None, ..., None], axis=0)
        return sel[0]


class MaxLevelSet(LevelSetFunction):
    """Pointwise maximum of several level sets."""

    def __init__(self, *levelsets: LevelSetFunction):
        if len(levelsets) < 2:
            raise ValueError("MaxLevelSet requires at least two level sets.")
        self.levelsets = tuple(levelsets)
        toks = []
        for ls in self.levelsets:
            tok = getattr(ls, "cache_token", None)
            toks.append(tok if tok is not None else int(id(ls)))
        self.cache_token = ("max_ls", tuple(toks))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        vals = [np.asarray(ls(x), dtype=float) for ls in self.levelsets]
        out = vals[0]
        for v in vals[1:]:
            out = np.maximum(out, v)
        return out

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            vals = [float(ls(x)) for ls in self.levelsets]
            k = int(np.argmax(np.asarray(vals, dtype=float)))
            return np.asarray(self.levelsets[k].gradient(x), dtype=float)

        vals = np.stack([np.asarray(ls(x), dtype=float) for ls in self.levelsets], axis=0)
        idx = np.argmax(vals, axis=0)  # (...,)
        grads = np.stack([np.asarray(ls.gradient(x), dtype=float) for ls in self.levelsets], axis=0)  # (k,...,2)
        sel = np.take_along_axis(grads, idx[None, ..., None], axis=0)
        return sel[0]

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
    - Works with existing MixedElement CG spaces (no new mesh nodes).
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


    # ---------- element-owner search: use shared helper ----------
    def _find_element_containing(self, x: np.ndarray, tol: float = 1e-12) -> int:
        return _find_owner_element(self._mesh, np.asarray(x, float), tol=tol)
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


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def _find_owner_element(mesh, x: np.ndarray, tol: float = 1e-12) -> int:
    """Return an element id whose reference image contains point ``x``.

    - Tries an O(nelem) robust scan via ``inverse_mapping`` with simple
      reference-domain checks.
    - Falls back to the nearest centroid if no element claims the point.

    Args:
        mesh: Geometry mesh providing mapping and element list.
        x: Physical point as array-like of length 2.
        tol: Numerical tolerance for reference-domain checks.

    Returns:
        Integer element id.
    """
    x = np.asarray(x, float)
    
    # 1. Try Fast Grid Search
    if hasattr(mesh, 'find_owner_element_fast'):
        candidates = mesh.find_owner_element_fast(x, tol)
        # Verify candidates
        for eid in candidates:
            try:
                xi, eta = transform.inverse_mapping(mesh, int(eid), x)
                if mesh.element_type == "quad":
                    if -1.0 - tol <= xi <= 1.0 + tol and -1.0 - tol <= eta <= 1.0 + tol:
                        return int(eid)
                else: 
                    if xi >= -tol and eta >= -tol and xi + eta <= 1.0 + tol:
                        return int(eid)
            except Exception:
                pass
        # If fast search failed (robustness), fall back to global could be added here
        # but typically grid search covers it.

    # 2. Standard Global Search (Fallback / Original)
    for e in mesh.elements_list:
        try:
            xi, eta = transform.inverse_mapping(mesh, int(e.id), x)
            if mesh.element_type == "quad":
                if -1.0 - tol <= xi <= 1.0 + tol and -1.0 - tol <= eta <= 1.0 + tol:
                    return int(e.id)
            else:
                if xi >= -tol and eta >= -tol and xi + eta <= 1.0 + tol:
                    return int(e.id)
        except Exception:
            pass
            
    # Nearest centroid fallback
    d = [np.linalg.norm(np.asarray(e.centroid()) - x) for e in mesh.elements_list]
    return int(np.argmin(d))


# -----------------------------------------------------------------------------
# Piecewise linear (P1/Q1) surrogate level set on the geometry mesh
# -----------------------------------------------------------------------------
class PiecewiseLinearLevelSet(LevelSetFunction):
    """Per-element P1 (tri) or Q1 (quad) approximation of a high-order level set.

    This class builds a piecewise linear/bilinear surrogate by matching φ on
    element corner nodes.

    Args:
        mesh: Geometry mesh whose corner topology defines elements.
        coeffs: For each element, the coefficient vector:
            - tri: [a, b, c] s.t. φ(x,y) = a x + b y + c (affine on the element)
            - quad: [v00, v10, v11, v01] corner nodal values on the reference
              square [-1,1]^2 in the mesh corner ordering
              (-1,-1),(1,-1),(1,1),(-1,1).
        node_values: φ evaluated at all mesh nodes (for quick nodal access).

    Notes:
        Construction is best done using ``from_level_set(mesh, level_set)``.
    """

    def __init__(self, mesh, coeffs: np.ndarray, node_values: np.ndarray):
        self.mesh = mesh
        self.coeffs = np.asarray(coeffs, float)
        self.node_values = np.asarray(node_values, float)
        self.element_type = mesh.element_type

    @classmethod
    def from_level_set(cls, mesh, level_set: LevelSetFunction) -> "PiecewiseLinearLevelSet":
        """Build a P1/Q1 surrogate by sampling ``level_set`` on mesh nodes.

        For each element, solve the small linear system that matches φ at its
        corner nodes to obtain the local linear/bilinear coefficients.

        Args:
            mesh: Geometry mesh (tri or quad).
            level_set: High-order or analytic level-set to approximate.

        Returns:
            Instance of ``PiecewiseLinearLevelSet`` bound to ``mesh``.
        """
        XY = np.asarray(mesh.nodes_x_y_pos, float)
        node_values = np.array([float(level_set(pt)) for pt in XY], float)
        coeffs: list[np.ndarray] = []

        if mesh.element_type == "tri":
            # Three corner points determine a plane φ(x,y) = a x + b y + c
            for corner_ids in mesh.corner_connectivity:
                # global node ids for the three corners
                cids = np.asarray(corner_ids, int)
                verts = XY[cids]
                phi = node_values[cids]
                # [x y 1] [a b c]^T = φ
                M = np.column_stack((verts, np.ones(3)))
                coeffs.append(np.linalg.solve(M, phi))
        elif mesh.element_type == "quad":
            # Store the corner nodal values (Q1 field is bilinear in reference coords,
            # not in physical x/y on general quads).
            for corner_ids in mesh.corner_connectivity:
                cids = np.asarray(corner_ids, int)
                coeffs.append(node_values[cids])
        else:
            raise KeyError(mesh.element_type)

        return cls(mesh, np.asarray(coeffs, float), node_values)

    # ---- small utilities ----------------------------------------------------
    def _coeff(self, eid: int) -> np.ndarray:
        return self.coeffs[int(eid)]

    def node_value(self, nid: int) -> float:
        return float(self.node_values[int(nid)])

    # ---- level-set protocol -------------------------------------------------
    def __call__(self, x: np.ndarray, eid: Optional[int] = None) -> float:
        """Evaluate φ at a point.

        Args:
            x: Physical point (2,).
            eid: Optional known owner element to avoid a global search.

        Returns:
            Scalar φ(x).
        """
        x = np.asarray(x, float)
        if eid is None:
            eid = _find_owner_element(self.mesh, x)
        c = self._coeff(int(eid))
        if self.element_type == "tri":
            a, b, d = c
            return float(a * x[0] + b * x[1] + d)
        # quad: Q1 (bilinear in reference coords)
        xi, eta = transform.inverse_mapping(self.mesh, int(eid), x)
        xi = float(xi)
        eta = float(eta)
        v00, v10, v11, v01 = c
        N00 = 0.25 * (1.0 - xi) * (1.0 - eta)
        N10 = 0.25 * (1.0 + xi) * (1.0 - eta)
        N11 = 0.25 * (1.0 + xi) * (1.0 + eta)
        N01 = 0.25 * (1.0 - xi) * (1.0 + eta)
        return float(v00 * N00 + v10 * N10 + v11 * N11 + v01 * N01)

    def gradient(self, x: np.ndarray, eid: Optional[int] = None) -> np.ndarray:
        """Gradient ∇φ(x) with optional owner element hint."""
        x = np.asarray(x, float)
        if eid is None:
            eid = _find_owner_element(self.mesh, x)
        c = self._coeff(int(eid))
        if self.element_type == "tri":
            a, b, _ = c
            return np.array([a, b], float)
        # quad: ∇φ = (∇̂φ)·J^{-1}, where φ is Q1 on the reference square.
        xi, eta = transform.inverse_mapping(self.mesh, int(eid), x)
        return self.gradient_on_element(int(eid), (float(xi), float(eta)))

    def value_on_element(self, eid: int, xi_eta: Tuple[float, float]) -> float:
        """Fast φ when element id and (ξ,η) are known."""
        if self.element_type == "tri":
            x = transform.x_mapping(self.mesh, int(eid), (float(xi_eta[0]), float(xi_eta[1])))
            return self.__call__(np.asarray(x, float), eid=int(eid))
        return self.value_on_element_ref(int(eid), (float(xi_eta[0]), float(xi_eta[1])))

    def gradient_on_element(self, eid: int, xi_eta: Tuple[float, float]) -> np.ndarray:
        """Fast ∇φ when element id and (ξ,η) are known (physical gradient)."""
        if self.element_type == "tri":
            x = transform.x_mapping(self.mesh, int(eid), (float(xi_eta[0]), float(xi_eta[1])))
            return self.gradient(np.asarray(x, float), eid=int(eid))

        xi, eta = float(xi_eta[0]), float(xi_eta[1])
        v00, v10, v11, v01 = self._coeff(int(eid))

        # reference derivatives of Q1 basis
        dN00_dxi = -0.25 * (1.0 - eta)
        dN10_dxi = 0.25 * (1.0 - eta)
        dN11_dxi = 0.25 * (1.0 + eta)
        dN01_dxi = -0.25 * (1.0 + eta)

        dN00_deta = -0.25 * (1.0 - xi)
        dN10_deta = -0.25 * (1.0 + xi)
        dN11_deta = 0.25 * (1.0 + xi)
        dN01_deta = 0.25 * (1.0 - xi)

        dphi_dxi = v00 * dN00_dxi + v10 * dN10_dxi + v11 * dN11_dxi + v01 * dN01_dxi
        dphi_deta = v00 * dN00_deta + v10 * dN10_deta + v11 * dN11_deta + v01 * dN01_deta
        grad_ref = np.array([dphi_dxi, dphi_deta], float)

        J = transform.jacobian(self.mesh, int(eid), (xi, eta))
        try:
            invJ = np.linalg.inv(J)
        except np.linalg.LinAlgError:
            invJ = np.linalg.pinv(J)
        return grad_ref @ invJ

    def evaluate_on_nodes(self, mesh) -> np.ndarray:
        if mesh is not self.mesh:
            raise ValueError("PiecewiseLinearLevelSet only defined on its mesh")
        return self.node_values.copy()
    def value_on_element_ref(self, eid: int, xi_eta: tuple[float,float]) -> float:
        xi, eta = float(xi_eta[0]), float(xi_eta[1])
        v = np.asarray(self._coeff(int(eid)), float)  # φ at the 3 (tri) or 4 (quad) corners
        if self.mesh.element_type == "tri":
            # Tri surrogate is stored as affine coefficients in physical space:
            #   φ(x,y) = a x + b y + c  with v = [a,b,c].
            x_phys = transform.x_mapping(self.mesh, int(eid), (xi, eta))
            return float(v[0] * float(x_phys[0]) + v[1] * float(x_phys[1]) + v[2])
        else:
            # reference square [-1,1]^2
            N00 = 0.25*(1.0 - xi)*(1.0 - eta)
            N10 = 0.25*(1.0 + xi)*(1.0 - eta)
            N11 = 0.25*(1.0 + xi)*(1.0 + eta)
            N01 = 0.25*(1.0 - xi)*(1.0 + eta)
            return float(v[0]*N00 + v[1]*N10 + v[2]*N11 + v[3]*N01)


# -----------------------------------------------------------------------------
# Level-set driven deformation on the geometry mesh
# -----------------------------------------------------------------------------
class LevelSetDeformation:
    """Deformation field defined by nodal displacements on the geometry mesh.

    Args:
        mesh: Geometry mesh. Displacements are defined at its geometry nodes.
        node_displacements: Array of shape (n_nodes, 2) with per-node (dx, dy).
    """

    def __init__(self, mesh, node_displacements: np.ndarray):
        self.mesh = mesh
        self.node_displacements = np.asarray(node_displacements, float)
        k = int(getattr(mesh, 'poly_order', 1))
        self._ref = transform.get_reference(mesh.element_type, k)

    def displacement_ref(self, eid: int, xi_eta: Tuple[float, float]) -> np.ndarray:
        """Interpolate displacement at reference coordinates on an element.

        Args:
            eid: Element id.
            xi_eta: Reference coordinates (ξ, η).

        Returns:
            Displacement vector (2,) in physical space at the mapped point.
        """
        xi, eta = float(xi_eta[0]), float(xi_eta[1])
        N = np.asarray(self._ref.shape(xi, eta)).ravel()
        # geometry node ids for this element
        gidx = np.asarray(self.mesh.nodes[self.mesh.elements_connectivity[int(eid)]], int)
        disp_nodes = self.node_displacements[gidx]
        return N @ disp_nodes

    def mapped_point(self, eid: int, xi_eta: Tuple[float, float]) -> np.ndarray:
        """Map a reference point and add its displacement in physical space."""
        x_phys = transform.x_mapping(self.mesh, int(eid), (float(xi_eta[0]), float(xi_eta[1])))
        return x_phys + self.displacement_ref(int(eid), xi_eta)

    def map_physical_point(self, eid: int, x_phys: np.ndarray) -> np.ndarray:
        """Displace a given physical point by the interpolated field on its element."""
        xi, eta = transform.inverse_mapping(self.mesh, int(eid), np.asarray(x_phys, float))
        return self.mapped_point(int(eid), (float(xi), float(eta)))


class LevelSetMeshAdaptation:
    """Level-set based mesh adaptation / isoparametric deformation.

    Given a high-order level set φ and a P1/Q1 surrogate φ_P1, this class computes
    a displacement field u_h (defined at the geometry nodes) so that the deformed
    mapping X̃ = X + u_h yields a better geometric approximation of the physical
    interface Γ = {φ=0}. In practice, this is used to improve cut integration by
    integrating over the deformed subcells K̃∩Ω̃ instead of K∩Ω.

    The construction follows standard ideas from unfitted FEM:
      - build a continuous search direction by Oswald-averaging local data,
      - compute element-local shifts (Newton-like) that reduce φ_P1−φ mismatch,
      - Oswald-average the high-order shifts to obtain a globally continuous u_h,
      - restrict the deformation to a narrow band around Γ to avoid moving the
        far-field mesh unnecessarily.

    Args:
        mesh: Geometry mesh.
        order: Reference order used to evaluate φ within elements (default 2).
        threshold: Maximum relative step (in element h) per Newton update.
        discontinuous_qn: Currently unused flag kept for parity.
        max_steps: Max Newton-like iterations per quadrature point.
        tol: Convergence tolerance for the residual |φ_P1 - φ|.
    """

    def __init__(self, mesh, *, order: int = 2, threshold: float = -1.0,
                 discontinuous_qn: bool = False, max_steps: int = 20, tol: float = 1e-12,
                 eps_perturbation: float = 1e-14):
        self.mesh = mesh
        self.order = int(order)
        self.threshold = float(threshold)
        self.discontinuous_qn = bool(discontinuous_qn)
        self.max_steps = int(max_steps)
        self.tol = float(tol)
        self.eps_perturbation = float(eps_perturbation)
        self.lset_p1: Optional[PiecewiseLinearLevelSet] = None
        self.deformation: Optional[LevelSetDeformation] = None

    def _project_level_set_to_h1(self, level_set: LevelSetFunction, *, q_vol: Optional[int] = None):
        """L2-project ``level_set`` into an H1/Qp space of order ``self.order``.

        Returns (dh, field, coeff) or None if SciPy is unavailable.
        """
        try:
            import scipy.sparse as _sp
            import scipy.sparse.linalg as _spla
        except Exception:
            return None

        mesh = self.mesh
        p = int(self.order)
        if p < 1:
            return None

        from pycutfem.fem.mixedelement import MixedElement
        from pycutfem.core.dofhandler import DofHandler
        from pycutfem.integration.quadrature import volume as vol_rule
        from pycutfem.fem.reference import get_reference

        # Our quadrature provider uses "number of Gauss points" per direction (quads)
        # or per element (tris). If the caller supplies an integration *degree*,
        # convert it to a safe Gauss rule size (Gauss-Legendre in 1D is exact up to
        # degree 2q-1, so q ≈ ceil((deg+1)/2)).
        deg_proj = int(2 * p) if q_vol is None else int(q_vol)
        if mesh.element_type == "quad":
            q_proj = max(3, int(np.ceil((deg_proj + 1) / 2)))
        else:
            q_proj = max(2, deg_proj)

        me = MixedElement(mesh, {"__lset__": p})
        dh = DofHandler(me, method="cg")
        field = "__lset__"

        qp_ref, qw_ref = vol_rule(mesh.element_type, q_proj)
        ref = get_reference(mesh.element_type, p)

        n_dofs = len(dh.get_field_slice(field))
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        rhs = np.zeros((n_dofs,), dtype=float)

        # Assemble global consistent mass matrix and RHS for L2 projection.
        for eid in range(len(mesh.elements_list)):
            gdofs = np.asarray(dh.element_maps[field][int(eid)], dtype=int)
            n_loc = int(len(gdofs))
            Mloc = np.zeros((n_loc, n_loc), dtype=float)
            bloc = np.zeros((n_loc,), dtype=float)

            for (xi, eta), w in zip(qp_ref, qw_ref):
                xi = float(xi)
                eta = float(eta)
                N = np.asarray(ref.shape(xi, eta), dtype=float).ravel()
                if N.shape[0] != n_loc:
                    raise ValueError("basis/local-dof size mismatch in level-set projection")

                x_phys = transform.x_mapping(mesh, int(eid), (xi, eta))
                J = transform.jacobian(mesh, int(eid), (xi, eta))
                detJ = abs(float(np.linalg.det(J)))

                phi_val = float(level_set(np.asarray(x_phys, float)))
                ww = float(w) * detJ
                Mloc += ww * np.outer(N, N)
                bloc += ww * phi_val * N

            for a in range(n_loc):
                ia = int(gdofs[a])
                rhs[ia] += float(bloc[a])
                for b in range(n_loc):
                    ib = int(gdofs[b])
                    rows.append(ia)
                    cols.append(ib)
                    data.append(float(Mloc[a, b]))

        M = _sp.coo_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()
        coeff = np.asarray(_spla.spsolve(M, rhs), dtype=float).reshape(-1)
        return dh, field, coeff

    def _interpolate_to_p1_from_h1(self, dh, field: str, coeff: np.ndarray) -> PiecewiseLinearLevelSet:
        """
        Build a P1/Q1 surrogate level set from an H1/Qp coefficient vector.

        We take the (continuous) vertex values from the H1 field and construct a
        piecewise linear/bilinear φ_P1. Very small vertex values are perturbed by
        `eps_perturbation` so that sign-based classification (inside/outside/cut)
        is robust: exactly-zero vertex values can lead to ambiguous "touching"
        configurations and brittle element/edge tagging.
        """
        mesh = self.mesh
        XY = np.asarray(mesh.nodes_x_y_pos, float)
        n_nodes = int(XY.shape[0])

        dof_map = getattr(dh, "dof_map", {}).get(field, {})
        vertex_ids = np.unique(np.asarray(mesh.corner_connectivity, dtype=int).ravel())

        vertex_vals = np.zeros((n_nodes,), dtype=float)
        eps = float(self.eps_perturbation)
        for nid in vertex_ids:
            gd = dof_map.get(int(nid))
            if gd is None:
                # Fallback: evaluate the analytic level set at the vertex.
                vertex_vals[int(nid)] = 0.0
                continue
            val = float(coeff[int(gd)])
            if abs(val) < eps:
                val = eps
            vertex_vals[int(nid)] = val

        coeffs: list[np.ndarray] = []
        if mesh.element_type == "tri":
            for corner_ids in mesh.corner_connectivity:
                cids = np.asarray(corner_ids, int)
                verts = XY[cids]
                phi = vertex_vals[cids]
                A = np.column_stack((verts, np.ones(3)))
                coeffs.append(np.linalg.solve(A, phi))
        elif mesh.element_type == "quad":
            for corner_ids in mesh.corner_connectivity:
                cids = np.asarray(corner_ids, int)
                coeffs.append(vertex_vals[cids])
        else:
            raise KeyError(mesh.element_type)

        # Map each mesh node to a containing element id once.
        node_to_eid: dict[int, int] = {}
        conn = np.asarray(mesh.elements_connectivity, dtype=int)
        for eid in range(conn.shape[0]):
            for nid in conn[eid]:
                node_to_eid.setdefault(int(nid), int(eid))

        tmp = PiecewiseLinearLevelSet(mesh, np.asarray(coeffs, float), np.zeros((n_nodes,), dtype=float))
        node_vals = np.zeros((n_nodes,), dtype=float)
        for nid in range(n_nodes):
            eid = node_to_eid.get(int(nid))
            if eid is None:
                node_vals[int(nid)] = float(vertex_vals[int(nid)])
                continue
            node_vals[int(nid)] = float(tmp(XY[int(nid)], eid=int(eid)))

        # Enforce perturbed vertex values exactly.
        node_vals[vertex_ids] = vertex_vals[vertex_ids]
        return PiecewiseLinearLevelSet(mesh, np.asarray(coeffs, float), node_vals)

    def calc_deformation(self, level_set: LevelSetFunction, q_vol: Optional[int] = None) -> LevelSetDeformation:
        """Compute nodal displacements that align φ_P1 with φ (Oswald projection).

        Goal (geometric viewpoint)
        --------------------------
        Construct an isoparametric deformation map X̃ = X + u_h that improves the
        representation of Γ in cut integration. The deformation is computed so that,
        in a narrow band around Γ, the surrogate φ_P1 aligns with the high-order φ.

        Outline
        -------
        1) Build a continuous search direction at corner nodes by Oswald-averaging
           element-local data (typically derived from ∇φ).
        2) On each "relevant" element, compute element-local shifts in reference
           coordinates that reduce the mismatch φ_P1−φ (Newton-like search), then
           fit these shifts in the deformation FE space via a local mass matrix.
        3) Oswald-average the high-order nodal shifts to obtain a globally continuous
           deformation u_h and set u_h=0 outside the relevant band.
        """
        mesh = self.mesh
        node_coords = mesh.nodes_x_y_pos
        n_nodes = node_coords.shape[0]

        proj = self._project_level_set_to_h1(level_set, q_vol=q_vol)
        if proj is None:
            # Robust fallback: nodal sampling + legacy first-order displacement.
            self.lset_p1 = PiecewiseLinearLevelSet.from_level_set(mesh, level_set)
            phi_p1_nodes = self.lset_p1.node_values

            mesh.classify_elements(self.lset_p1, tol=self.tol)
            cut_ids = mesh.element_bitset("cut").to_indices()
            search_dirs_at_nodes = np.zeros((n_nodes, 2), float)
            dir_counts_at_nodes = np.zeros(n_nodes, float)

            for eid in cut_ids:
                elem = mesh.elements_list[int(eid)]
                for nid in elem.corner_nodes:
                    xcorner = np.asarray(mesh.nodes_x_y_pos[int(nid)], float)
                    g = np.asarray(level_set.gradient(xcorner), float)
                    nrm = float(np.linalg.norm(g))
                    if nrm < 1e-14:
                        continue
                    search_dirs_at_nodes[int(nid)] += g
                    dir_counts_at_nodes[int(nid)] += 1.0

            mask = dir_counts_at_nodes > 0
            if np.any(mask):
                search_dirs_at_nodes[mask] /= dir_counts_at_nodes[mask, None]
                norms = np.linalg.norm(search_dirs_at_nodes, axis=1, keepdims=True) + 1e-30
                search_dirs_at_nodes[mask] /= norms[mask]
        else:
            dh_ho, field_ho, coeff_ho = proj
            self.lset_p1 = self._interpolate_to_p1_from_h1(dh_ho, field_ho, coeff_ho)
            phi_p1_nodes = self.lset_p1.node_values

            # Restrict the deformation solve to a narrow band around Γ: elements whose
            # P1 surrogate changes sign at their vertices (or touches zero). This keeps
            # the deformation local and avoids moving far-field geometry unnecessarily.
            corner_ids = np.asarray(mesh.corner_connectivity, dtype=int)
            corner_vals = phi_p1_nodes[corner_ids]
            has_pos = (corner_vals > 0.0).any(axis=1) | (corner_vals == 0.0).any(axis=1)
            has_neg = (corner_vals < 0.0).any(axis=1) | (corner_vals == 0.0).any(axis=1)
            cut_ids = np.where(has_pos & has_neg)[0].astype(int)

        # helper: clamp reference coordinates back to the parent element
        def _clamp_ref(z: np.ndarray) -> np.ndarray:
            eps = 1e-14
            if mesh.element_type == "tri":
                xi, eta = float(z[0]), float(z[1])
                xi = max(eps, min(1.0 - eps, xi))
                eta = max(eps, min(1.0 - xi - eps, eta))
                return np.array([xi, eta], float)
            else:
                xi, eta = float(z[0]), float(z[1])
                xi = max(-1.0 + eps, min(1.0 - eps, xi))
                eta = max(-1.0 + eps, min(1.0 - eps, eta))
                return np.array([xi, eta], float)

        # --- STEP 2: element-local solve + Oswald averaging ---
        #
        # Oswald averaging is a projection onto a globally continuous H1 space: shared
        # DOF coefficients (vertices/edges) are averaged across adjacent elements,
        # while element-interior DOFs remain local. This is well-defined in terms of
        # the FE *coefficients* (global DOFs), not in terms of point samples.
        #
        # For triangles, the Qp geometry lattice is a nodal Lagrange basis, so nodal
        # values coincide with FE DOFs and "nodal Oswald" is equivalent.
        #
        # For quads with p>=2 we store deformation values on a tensor-product lattice,
        # but the natural H1-high-order coefficient decomposition is hierarchical
        # (vertex/edge/interior modes). Averaging lattice point values directly is
        # *not* equivalent to Oswald averaging of the H1 coefficients and can yield an
        # overly local deformation (no propagation through shared edge modes).
        #
        # To respect the mathematical meaning of Oswald averaging on quads, we:
        #   (i)  transform nodal values -> hierarchical H1 coefficients,
        #   (ii) average shared vertex/edge coefficients (Oswald),
        #   (iii) evaluate the resulting hierarchical field back at lattice nodes.
        displacements = np.zeros((n_nodes, 2), float)
        counts = np.zeros(n_nodes, float)

        from pycutfem.integration.quadrature import volume as volume_rule

        # Deformation space is represented on the geometry lattice; currently require matching orders.
        p_def = int(mesh.poly_order)
        if proj is not None and int(self.order) != p_def:
            raise ValueError("LevelSetMeshAdaptation currently requires mesh.poly_order == order for ProjectShift parity.")

        ref_def = transform.get_reference(mesh.element_type, p_def)
        if proj is not None:
            from pycutfem.fem.reference import get_reference
            ref_ho = get_reference(mesh.element_type, int(self.order))

        use_hier_oswald = (
            proj is not None
            and mesh.element_type == "quad"
            and p_def >= 2
            and getattr(mesh, "edges_connectivity", None) is not None
        )

        hier_A = None
        hier_T = None
        hier_edge_key_to_gid: dict[tuple[int, int], int] | None = None
        hier_edge_sum = None
        hier_edge_cnt = None
        hier_vtx_sum = None
        hier_vtx_cnt = None
        hier_int_coeff = None
        hier_cut_mask = None
        if use_hier_oswald:
            # Cache per polynomial order; matrices are small ((p+1)^2 x (p+1)^2).
            cache = getattr(LevelSetMeshAdaptation, "_hier_quad_cache", {})
            key = ("quad_h1hier", int(p_def))
            if key in cache:
                hier_A, hier_T = cache[key]
            else:
                nlat = int(p_def) + 1
                xis = np.linspace(-1.0, 1.0, nlat)
                etas = np.linspace(-1.0, 1.0, nlat)

                def _legendre_all(n: int, x: float) -> np.ndarray:
                    P = np.zeros((n + 1,), dtype=float)
                    P[0] = 1.0
                    if n >= 1:
                        P[1] = float(x)
                    for k in range(2, n + 1):
                        P[k] = ((2 * k - 1) * float(x) * P[k - 1] - (k - 1) * P[k - 2]) / k
                    return P

                def _integrated_legendre(n: int, x: float, P: np.ndarray) -> float:
                    # Integrated Legendre basis: L_n(x) = ∫_{-1}^x P_{n-1}(t) dt
                    # For n>=2, L_n(±1)=0 and we have the closed form:
                    #   L_n(x) = (P_n(x) - P_{n-2}(x)) / (2n-1).
                    return float(P[n] - P[n - 2]) / float(2 * n - 1)

                def _h1_hier_quad_basis(p: int, xi: float, eta: float) -> np.ndarray:
                    vx0 = 0.5 * (1.0 - float(xi))
                    vx1 = 0.5 * (1.0 + float(xi))
                    vy0 = 0.5 * (1.0 - float(eta))
                    vy1 = 0.5 * (1.0 + float(eta))

                    Pxi = _legendre_all(p, float(xi))
                    Peta = _legendre_all(p, float(eta))

                    Lxi = [_integrated_legendre(k, float(xi), Pxi) for k in range(2, p + 1)]
                    Leta = [_integrated_legendre(k, float(eta), Peta) for k in range(2, p + 1)]

                    vals: list[float] = []
                    # vertices: bl, br, tr, tl (matches Mesh.corner_connectivity for quads)
                    vals.extend([vx0 * vy0, vx1 * vy0, vx1 * vy1, vx0 * vy1])
                    # edges: bottom, right, top, left; modes k=2..p
                    vals.extend([vy0 * Lk for Lk in Lxi])  # bottom (eta=-1)
                    vals.extend([vx1 * Lk for Lk in Leta])  # right  (xi=+1)
                    vals.extend([vy1 * Lk for Lk in Lxi])  # top    (eta=+1)
                    vals.extend([vx0 * Lk for Lk in Leta])  # left   (xi=-1)
                    # interior: tensor-product (i,j) with i,j=2..p
                    vals.extend([Lx * Ly for Ly in Leta for Lx in Lxi])
                    return np.asarray(vals, dtype=float)

                nloc = nlat * nlat
                A = np.zeros((nloc, nloc), dtype=float)
                row = 0
                for eta in etas:
                    for xi in xis:
                        A[row, :] = _h1_hier_quad_basis(int(p_def), float(xi), float(eta))
                        row += 1
                # Solve A * c = u(node) for coefficients c in the hierarchical basis
                T = np.linalg.solve(A, np.eye(nloc, dtype=float))
                hier_A, hier_T = A, T
                cache[key] = (hier_A, hier_T)
                LevelSetMeshAdaptation._hier_quad_cache = cache

            # Build edge lookup for geometric edges (corner-to-corner) by endpoint ids.
            hier_edge_key_to_gid = {}
            for e in mesh.edges_list:
                v0, v1 = int(e.nodes[0]), int(e.nodes[1])
                hier_edge_key_to_gid[(min(v0, v1), max(v0, v1))] = int(e.gid)

            n_edges = len(mesh.edges_list)
            n_edge_modes = int(p_def) - 1  # modes k=2..p
            hier_edge_sum = np.zeros((n_edges * n_edge_modes, 2), dtype=float)
            hier_edge_cnt = np.zeros((n_edges * n_edge_modes,), dtype=float)
            hier_vtx_sum = np.zeros((n_nodes, 2), dtype=float)
            hier_vtx_cnt = np.zeros((n_nodes,), dtype=float)
            n_int = (int(p_def) - 1) ** 2
            hier_int_coeff = np.zeros((len(mesh.elements_connectivity), n_int, 2), dtype=float)
            hier_cut_mask = np.zeros((len(mesh.elements_connectivity),), dtype=bool)

        if mesh.element_type == "quad":
            deg_shift = 2 * int(p_def)
            q_order = max(3, int(np.ceil((deg_shift + 1) / 2)))
        else:
            q_order = max(2 * int(p_def), 2)
        qpts_ref, qw_ref = volume_rule(mesh.element_type, q_order)

        def _eval_ho_local(u_loc: np.ndarray, xi_eta: np.ndarray) -> float:
            xi = float(xi_eta[0]); eta = float(xi_eta[1])
            N = np.asarray(ref_ho.shape(xi, eta), float).ravel()
            return float(N @ u_loc)

        def _grad_ho_ref_local(u_loc: np.ndarray, xi_eta: np.ndarray) -> np.ndarray:
            xi = float(xi_eta[0]); eta = float(xi_eta[1])
            dN = np.asarray(ref_ho.grad(xi, eta), float)  # (nloc,2)
            return np.asarray(dN.T @ u_loc, float).reshape(2,)

        def _search_corresponding_point(eid: int, u_loc: np.ndarray, init_ref: np.ndarray, goal: float, search_dir: np.ndarray) -> np.ndarray:
            curr = np.asarray(init_ref, float).reshape(2,)
            for _ in range(int(self.max_steps) if int(self.max_steps) > 0 else 20):
                val = _eval_ho_local(u_loc, curr)
                defect = float(goal - val)
                if abs(defect) < 1e-14:
                    return curr
                grad_ref = _grad_ho_ref_local(u_loc, curr)
                dphidn = float(np.dot(grad_ref, search_dir))
                if abs(dphidn) < 1e-30:
                    break
                curr = curr + (defect / dphidn) * search_dir
            return np.asarray(init_ref, float).reshape(2,)

        for eid in cut_ids:
            eid = int(eid)
            geom_node_ids = np.asarray(mesh.elements_connectivity[eid], int)
            nloc = int(geom_node_ids.size)

            mass = np.zeros((nloc, nloc), float)
            rhs = np.zeros((nloc, 2), float)

            if proj is not None:
                gdofs_ho = np.asarray(dh_ho.element_maps[field_ho][eid], dtype=int)
                u_loc_ho = np.asarray(coeff_ho[gdofs_ho], float).reshape(-1)

            for (xi, eta), w in zip(qpts_ref, qw_ref):
                xi = float(xi); eta = float(eta)
                J = transform.jacobian(mesh, eid, (xi, eta))
                detJ = abs(float(np.linalg.det(J)))
                if detJ <= 0.0:
                    continue
                ww = float(w) * detJ

                N_def = np.asarray(ref_def.shape(xi, eta), float).ravel()
                if N_def.shape[0] != nloc:
                    raise ValueError("deformation basis/local-node size mismatch")
                mass += ww * np.outer(N_def, N_def)

                if proj is None:
                    # Legacy: simple first-order update using analytic φ and ∇φ.
                    x_phys = transform.x_mapping(mesh, eid, (xi, eta))
                    phi_lin = float(self.lset_p1.value_on_element(eid, (xi, eta)))
                    phi_val = float(level_set(x_phys))
                    residual = phi_lin - phi_val
                    if abs(residual) < self.tol:
                        continue
                    g = np.asarray(level_set.gradient(x_phys), float).reshape(2,)
                    g2 = float(np.dot(g, g))
                    if g2 <= 1e-30:
                        continue
                    delta_x = (residual / g2) * g
                    if float(self.threshold) >= 0.0:
                        h_elem = mesh.element_char_length(eid) if hasattr(mesh, "element_char_length") else 1.0
                        max_disp = float(self.threshold) * float(h_elem)
                        n_dx = float(np.linalg.norm(delta_x))
                        if n_dx > max_disp and n_dx > 0.0:
                            delta_x = delta_x * (max_disp / n_dx)
                    rhs += ww * np.outer(N_def, delta_x)
                else:
                    # ProjectShift parity: Newton search in reference space for lset_ho == lset_p1.
                    orig_ref = np.array([xi, eta], float)
                    goal_val = float(self.lset_p1.value_on_element_ref(eid, (xi, eta)))

                    grad_ref_orig = _grad_ho_ref_local(u_loc_ho, orig_ref)
                    try:
                        Jinv = np.linalg.inv(J)
                    except np.linalg.LinAlgError:
                        Jinv = np.linalg.pinv(J)
                    qn_phys = Jinv.T @ grad_ref_orig
                    search_dir = Jinv @ qn_phys
                    if float(np.dot(search_dir, search_dir)) <= 1e-30:
                        continue

                    final_ref = _search_corresponding_point(eid, u_loc_ho, orig_ref, goal_val, search_dir)
                    ref_dist = final_ref - orig_ref
                    ref_dist_norm = float(np.linalg.norm(ref_dist))
                    if float(self.threshold) >= 0.0 and ref_dist_norm > float(self.threshold) and ref_dist_norm > 0.0:
                        ref_dist = ref_dist * (float(self.threshold) / ref_dist_norm)

                    deform = np.asarray(J @ ref_dist, float).reshape(2,)
                    rhs += ww * np.outer(N_def, deform)

            if not np.any(rhs):
                continue
            try:
                shift = np.linalg.solve(mass, rhs)
            except np.linalg.LinAlgError:
                shift, *_ = np.linalg.lstsq(mass, rhs, rcond=None)

            # Remove an element-wise affine component by anchoring the displacement at
            # D+1 non-collinear vertices. In 2D, fixing 3 vertex values is enough to
            # eliminate any affine vector field (A x + b) component on the element,
            # which helps keep the deformation local and prevents mesh-scale drift.
            # The remaining vertex value is later made globally continuous by Oswald
            # averaging across elements.
            corner_ids = np.asarray(mesh.corner_connectivity[eid], dtype=int)
            freeze = set(map(int, corner_ids[: 2 + 1]))  # D+1 with D=2
            for a, node_id in enumerate(geom_node_ids):
                if int(node_id) in freeze:
                    shift[a, :] = 0.0

            if use_hier_oswald:
                # Convert nodal values (Lagrange) -> hierarchical H1 coefficients (integrated Legendre).
                # hier_T maps: coeff_hier = hier_T @ values_at_nodes.
                hier_cut_mask[eid] = True
                coeff_hier = np.asarray(hier_T @ shift, dtype=float)  # (nloc,2)

                # (1) vertex coefficients (shared by node id)
                for lv in range(4):
                    nid = int(corner_ids[lv])
                    hier_vtx_sum[nid, :] += coeff_hier[lv, :]
                    hier_vtx_cnt[nid] += 1.0

                # (2) edge coefficients (shared by geometric edge + mode; orientation for odd modes)
                # Local reference edge orientations (s=-1 -> s=+1):
                #   bottom: bl->br, right: br->tr, top: tl->tr, left: bl->tl
                bl, br, tr, tl = map(int, corner_ids)
                edge_endpoints = {
                    0: (bl, br),  # bottom
                    1: (br, tr),  # right
                    2: (tl, tr),  # top
                    3: (bl, tl),  # left
                }
                n_edge_modes = int(p_def) - 1
                edge_off = 4
                for e_local in range(4):
                    v_minus, v_plus = edge_endpoints[e_local]
                    key_edge = (min(v_minus, v_plus), max(v_minus, v_plus))
                    gid = hier_edge_key_to_gid.get(key_edge)
                    if gid is None:
                        continue
                    match = (v_minus == key_edge[0] and v_plus == key_edge[1])
                    for mi in range(n_edge_modes):
                        k = 2 + mi
                        sign = 1.0 if match else (-1.0 if (k % 2 == 1) else 1.0)
                        idx_loc = edge_off + e_local * n_edge_modes + mi
                        idx_glob = int(gid) * n_edge_modes + mi
                        hier_edge_sum[idx_glob, :] += sign * coeff_hier[idx_loc, :]
                        hier_edge_cnt[idx_glob] += 1.0

                # (3) interior coefficients (local per element; no averaging)
                int_off = 4 + 4 * n_edge_modes
                n_int = (int(p_def) - 1) ** 2
                if coeff_hier.shape[0] < int_off + n_int:
                    raise ValueError("hierarchical basis size mismatch for quad element")
                hier_int_coeff[eid, :, :] = coeff_hier[int_off:int_off + n_int, :]
            else:
                for a, node_id in enumerate(geom_node_ids):
                    nid = int(node_id)
                    dv = shift[a, :]
                    if not np.any(dv):
                        continue
                    displacements[nid, :] += dv
                    counts[nid] += 1.0

        if use_hier_oswald:
            # Average the shared hierarchical coefficients (Oswald).
            vmask = hier_vtx_cnt > 0
            if np.any(vmask):
                hier_vtx_sum[vmask, :] /= hier_vtx_cnt[vmask, None]

            emask = hier_edge_cnt > 0
            if np.any(emask):
                hier_edge_sum[emask, :] /= hier_edge_cnt[emask, None]

            # Reconstruct nodal values on every element by evaluating the hierarchical field at
            # the Qp lattice nodes, then Oswald-average the nodal values on shared nodes.
            n_edge_modes = int(p_def) - 1
            n_int = (int(p_def) - 1) ** 2
            displacements = np.zeros((n_nodes, 2), dtype=float)
            counts = np.zeros((n_nodes,), dtype=float)
            for eid in range(len(mesh.elements_connectivity)):
                conn = np.asarray(mesh.elements_connectivity[int(eid)], dtype=int)
                if conn.size != hier_A.shape[0]:
                    raise ValueError("quad deformation lattice size mismatch")

                corners = np.asarray(mesh.corner_connectivity[int(eid)], dtype=int)
                bl, br, tr, tl = map(int, corners)
                # Local coeffs in our hierarchical ordering.
                coeff_loc = np.zeros((hier_A.shape[0], 2), dtype=float)
                # vertices
                coeff_loc[0, :] = hier_vtx_sum[bl, :]
                coeff_loc[1, :] = hier_vtx_sum[br, :]
                coeff_loc[2, :] = hier_vtx_sum[tr, :]
                coeff_loc[3, :] = hier_vtx_sum[tl, :]

                # edges with orientation back-transform (global -> local)
                edge_endpoints = {
                    0: (bl, br),  # bottom
                    1: (br, tr),  # right
                    2: (tl, tr),  # top
                    3: (bl, tl),  # left
                }
                edge_off = 4
                for e_local in range(4):
                    v_minus, v_plus = edge_endpoints[e_local]
                    key_edge = (min(v_minus, v_plus), max(v_minus, v_plus))
                    gid = hier_edge_key_to_gid.get(key_edge)
                    if gid is None:
                        continue
                    match = (v_minus == key_edge[0] and v_plus == key_edge[1])
                    for mi in range(n_edge_modes):
                        k = 2 + mi
                        sign = 1.0 if match else (-1.0 if (k % 2 == 1) else 1.0)
                        idx_loc = edge_off + e_local * n_edge_modes + mi
                        idx_glob = int(gid) * n_edge_modes + mi
                        coeff_loc[idx_loc, :] = sign * hier_edge_sum[idx_glob, :]

                # interior modes are element-local; keep them only in the cut band to
                # localize the deformation near Γ and avoid introducing high-order
                # oscillations in the far field.
                int_off = 4 + 4 * n_edge_modes
                if bool(hier_cut_mask[int(eid)]):
                    coeff_loc[int_off:int_off + n_int, :] = hier_int_coeff[int(eid), :, :]

                vals_nodes = np.asarray(hier_A @ coeff_loc, dtype=float)  # (nloc,2)
                displacements[conn, :] += vals_nodes
                counts[conn] += 1.0

            mask2 = counts > 0
            if np.any(mask2):
                displacements[mask2, :] /= counts[mask2, None]
            displacements[~mask2, :] = 0.0
        else:
            mask2 = counts > 0
            if np.any(mask2):
                displacements[mask2, :] /= counts[mask2, None]
            displacements[~mask2, :] = 0.0

        self.deformation = LevelSetDeformation(mesh, displacements)
        return self.deformation
