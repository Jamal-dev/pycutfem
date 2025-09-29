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
    for e in mesh.elements_list:
        try:
            xi, eta = transform.inverse_mapping(mesh, int(e.id), x)
            if mesh.element_type == "quad":
                if -1.0 - tol <= xi <= 1.0 + tol and -1.0 - tol <= eta <= 1.0 + tol:
                    return int(e.id)
            else:  # tri
                if xi >= -tol and eta >= -tol and xi + eta <= 1.0 + tol:
                    return int(e.id)
        except Exception:
            pass
    d = [np.linalg.norm(np.asarray(e.centroid()) - x) for e in mesh.elements_list]
    return int(np.argmin(d))


# -----------------------------------------------------------------------------
# Piecewise linear (P1/Q1) surrogate level set on the geometry mesh
# -----------------------------------------------------------------------------
class PiecewiseLinearLevelSet(LevelSetFunction):
    """Per-element P1 (tri) or Q1 (quad) approximation of a high-order level set.

    This class builds a piecewise linear/bilinear surrogate by matching φ on
    element corner nodes, and uses closed-form evaluation and gradients.

    Args:
        mesh: Geometry mesh whose corner topology defines elements.
        coeffs: For each element, the coefficient vector:
            - tri: [a, b, c] s.t. φ = a x + b y + c
            - quad: [c0, c1, c2, c3] s.t. φ = c0 + c1 x + c2 y + c3 x y
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
            # Four corners determine a bilinear form φ = c0 + c1 x + c2 y + c3 x y
            for corner_ids in mesh.corner_connectivity:
                cids = np.asarray(corner_ids, int)
                verts = XY[cids]
                phi = node_values[cids]
                X, Y = verts[:, 0], verts[:, 1]
                M = np.column_stack((np.ones(4), X, Y, X * Y))
                coeffs.append(np.linalg.solve(M, phi))
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
        # quad bilinear
        c0, c1, c2, c3 = c
        px, py = float(x[0]), float(x[1])
        return float(c0 + c1 * px + c2 * py + c3 * px * py)

    def gradient(self, x: np.ndarray, eid: Optional[int] = None) -> np.ndarray:
        """Gradient ∇φ(x) with optional owner element hint."""
        x = np.asarray(x, float)
        if eid is None:
            eid = _find_owner_element(self.mesh, x)
        c = self._coeff(int(eid))
        if self.element_type == "tri":
            a, b, _ = c
            return np.array([a, b], float)
        c0, c1, c2, c3 = c
        px, py = float(x[0]), float(x[1])
        return np.array([c1 + c3 * py, c2 + c3 * px], float)

    def value_on_element(self, eid: int, xi_eta: Tuple[float, float]) -> float:
        """Fast φ when element id and (ξ,η) are known."""
        x = transform.x_mapping(self.mesh, int(eid), (float(xi_eta[0]), float(xi_eta[1])))
        return self.__call__(np.asarray(x, float), eid=int(eid))

    def gradient_on_element(self, eid: int, xi_eta: Tuple[float, float]) -> np.ndarray:
        """Fast ∇φ when element id and (ξ,η) are known (physical gradient)."""
        x = transform.x_mapping(self.mesh, int(eid), (float(xi_eta[0]), float(xi_eta[1])))
        return self.gradient(np.asarray(x, float), eid=int(eid))

    def evaluate_on_nodes(self, mesh) -> np.ndarray:
        if mesh is not self.mesh:
            raise ValueError("PiecewiseLinearLevelSet only defined on its mesh")
        return self.node_values.copy()


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
    """Lightweight analogue of NGSolve's LevelSetMeshAdaptation.

    Builds an Oswald-averaged search direction and computes a displacement field
    that moves geometry nodes such that a piecewise-linear surrogate φ_P1 aligns
    with the high-order φ within a bounded threshold per element.

    Args:
        mesh: Geometry mesh.
        order: Reference order used to evaluate φ within elements (default 2).
        threshold: Maximum relative step (in element h) per Newton update.
        discontinuous_qn: Currently unused flag kept for parity.
        max_steps: Max Newton-like iterations per quadrature point.
        tol: Convergence tolerance for the residual |φ_P1 - φ|.
    """

    def __init__(self, mesh, *, order: int = 2, threshold: float = 1.0,
                 discontinuous_qn: bool = True, max_steps: int = 6, tol: float = 1e-12):
        self.mesh = mesh
        self.order = int(order)
        self.threshold = float(threshold)
        self.discontinuous_qn = bool(discontinuous_qn)
        self.max_steps = int(max_steps)
        self.tol = float(tol)
        self.lset_p1: Optional[PiecewiseLinearLevelSet] = None
        self.deformation: Optional[LevelSetDeformation] = None

    def calc_deformation(self, level_set: LevelSetFunction, q_vol: Optional[int] = None) -> LevelSetDeformation:
        """Compute nodal displacements that align φ_P1 with φ.

        The method performs two stages:
        1) Build an Oswald-averaged search direction at corner nodes using the
           true ∇φ evaluated at cut-element corners, normalised by magnitude.
        2) For each cut element, assemble a small mass system and a right-hand
           side derived from a Newton-like update along the search direction to
           compute target displacements, averaged back to nodes.

        Args:
            level_set: High-order or analytic level-set used as the reference.
            q_vol: Optional quadrature order override for the volume integration.

        Returns:
            LevelSetDeformation with nodal displacement field.
        """
        mesh = self.mesh
        node_coords = np.asarray(mesh.nodes_x_y_pos, float)
        n_nodes = node_coords.shape[0]

        # Piecewise linear surrogate and nodal φ values
        self.lset_p1 = PiecewiseLinearLevelSet.from_level_set(mesh, level_set)
        phi_p1_nodes = self.lset_p1.node_values

        # Classify and pick cut elements
        mesh.classify_elements(self.lset_p1, tol=self.tol)
        cut_ids = mesh.element_bitset("cut").to_indices()

        # --- STEP 1: Oswald-averaged search direction at corner nodes ---
        search_dirs_at_nodes = np.zeros((n_nodes, 2), float)
        dir_counts_at_nodes = np.zeros(n_nodes, float)

        for eid in cut_ids:
            elem = mesh.elements_list[int(eid)]
            for nid in elem.corner_nodes:
                xcorner = np.asarray(mesh.nodes_x_y_pos[int(nid)], float)
                g = np.asarray(level_set.gradient(xcorner), float)
                n = np.linalg.norm(g)
                if n < 1e-14:
                    continue
                search_dirs_at_nodes[int(nid)] += g
                dir_counts_at_nodes[int(nid)] += 1.0

        mask = dir_counts_at_nodes > 0
        if np.any(mask):
            search_dirs_at_nodes[mask] /= dir_counts_at_nodes[mask, None]
            norms = np.linalg.norm(search_dirs_at_nodes, axis=1, keepdims=True) + 1e-30
            search_dirs_at_nodes[mask] /= norms[mask]

        # --- STEP 2: Main loop to calculate nodal displacements ---
        # Local nodal h (min per incident element), fallback to 1.0
        node_h = np.full(n_nodes, np.inf, float)
        for elem in mesh.elements_list:
            h_e = mesh.element_char_length(int(elem.id)) if hasattr(mesh, "element_char_length") else 1.0
            for nid in elem.corner_nodes:
                idx = int(nid)
                node_h[idx] = min(node_h[idx], h_e)
        node_h[node_h == np.inf] = 1.0

        displacements = np.zeros((n_nodes, 2), float)
        counts = np.zeros(n_nodes, float)

        # Geometry/reference helpers
        ref = transform.get_reference(mesh.element_type, mesh.poly_order)
        ref_geom = transform.get_reference(mesh.element_type, 1)  # corner-node shapes

        # Volume quadrature rule
        if q_vol is None:
            # use slightly higher order to reduce projection error
            q_order = max(2 * int(getattr(mesh, "poly_order", 1)) + 4, 6)
        else:
            q_order = int(q_vol)
        from pycutfem.integration.quadrature import volume as volume_rule
        qpts_ref, qw_ref = volume_rule(mesh.element_type, q_order)

        for eid in cut_ids:
            elem = mesh.elements_list[int(eid)]
            geom_local = np.asarray(mesh.elements_connectivity[int(eid)], int)
            geom_ids = np.asarray(mesh.nodes[geom_local], int)

            corner_nids = np.asarray(elem.corner_nodes, int)
            corner_node_search_dirs = search_dirs_at_nodes[corner_nids]

            nloc = geom_ids.size
            mass = np.zeros((nloc, nloc), float)
            rhs = np.zeros((nloc, 2), float)

            for (xi, eta), w in zip(qpts_ref, qw_ref):
                J = transform.jacobian(mesh, int(eid), (float(xi), float(eta)))
                detJ = abs(np.linalg.det(J))
                if detJ <= 0.0:
                    continue

                N = np.asarray(ref.shape(float(xi), float(eta)), float).ravel()
                mass += (w * detJ) * np.outer(N, N)

                # Prefer local high-order gradient as search direction
                x_phys = transform.x_mapping(mesh, int(eid), (float(xi), float(eta)))
                if hasattr(level_set, "gradient_on_element"):
                    g0 = np.asarray(level_set.gradient_on_element(int(eid), (float(xi), float(eta))), float)
                else:
                    g0 = np.asarray(level_set.gradient(x_phys), float)

                if np.linalg.norm(g0) >= 1e-14:
                    search_dir_phys = g0 / np.linalg.norm(g0)
                else:
                    # Fallback to Oswald-averaged corner field
                    N_geom = np.asarray(ref_geom.shape(float(xi), float(eta))).ravel()
                    tmp = N_geom @ corner_node_search_dirs
                    if np.linalg.norm(tmp) < 1e-14:
                        continue
                    search_dir_phys = tmp / np.linalg.norm(tmp)

                # Newton-like step towards matching φ_P1 and φ
                phi_lin = self.lset_p1.value_on_element(int(eid), (float(xi), float(eta)))
                x_target = x_phys
                for _ in range(self.max_steps):
                    phi_curr = level_set(x_target)
                    residual = phi_lin - phi_curr
                    if abs(residual) < self.tol:
                        break
                    grad_curr = level_set.gradient(x_target)
                    denom = float(np.dot(grad_curr, search_dir_phys))
                    if abs(denom) < 1e-14:
                        break
                    step = residual / denom
                    x_target = x_target + step * search_dir_phys

                delta_x = x_target - x_phys
                h_elem = mesh.element_char_length(int(eid)) if hasattr(mesh, "element_char_length") else 1.0
                max_disp = abs(self.threshold) * h_elem
                norm_dx = float(np.linalg.norm(delta_x))
                if norm_dx > max_disp and norm_dx > 0.0:
                    delta_x = delta_x * (max_disp / norm_dx)

                rhs += (w * detJ) * np.outer(N, delta_x)

            if not np.any(rhs):
                continue

            try:
                shift = np.linalg.solve(mass, rhs)
            except np.linalg.LinAlgError:
                shift, *_ = np.linalg.lstsq(mass, rhs, rcond=None)

            # Accumulate to global nodes (no artificial band cutoff to preserve accuracy)
            for local_idx, node_id in enumerate(geom_ids):
                nid = int(node_id)
                disp_vec = shift[local_idx, :]
                if not np.any(disp_vec):
                    continue
                displacements[nid, :] += disp_vec
                counts[nid] += 1.0

        mask2 = counts > 0
        if np.any(mask2):
            displacements[mask2, :] /= counts[mask2, None]
        displacements[~mask2, :] = 0.0

        self.deformation = LevelSetDeformation(mesh, displacements)
        return self.deformation
