import math
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import AffineLevelSet, LevelSetGridFunction
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.measures import dInterface, dx
from pycutfem.ufl.expressions import Constant, FacetNormal
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.helpers_geom import edge_root_pn


def _assemble_scalar(form, dof_handler, *, backend: str):
    hook = {type(form.integrand): {"name": "scalar"}}
    res = assemble_form(
        Equation(None, form),
        dof_handler=dof_handler,
        bcs=[],
        backend=backend,
        assembler_hooks=hook,
    )
    val = res["scalar"]
    if isinstance(val, np.ndarray):
        return float(val.reshape(-1)[0])
    return float(val)


def _refresh_geometry(mesh: Mesh, level_set, *, tol: float = 1e-12) -> None:
    mesh.classify_elements(level_set, tol=tol)
    mesh.classify_edges(level_set, tol=tol)
    mesh.build_interface_segments(level_set, tol=tol)


class _SignedDistanceBoxLevelSet:
    """Signed distance to an axis-aligned rectangle (box) in 2D."""

    def __init__(self, *, center: tuple[float, float], L: float, H: float, token: str = "box_sdf"):
        self.center = np.asarray(center, dtype=float)
        self.hx = 0.5 * float(L)
        self.hy = 0.5 * float(H)
        self.cache_token = (token, float(self.center[0]), float(self.center[1]), float(self.hx), float(self.hy))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        qx = np.abs(x[..., 0] - self.center[0]) - self.hx
        qy = np.abs(x[..., 1] - self.center[1]) - self.hy
        outside = np.stack((np.maximum(qx, 0.0), np.maximum(qy, 0.0)), axis=-1)
        outside_dist = np.linalg.norm(outside, axis=-1)
        inside_dist = np.minimum(np.maximum(qx, qy), 0.0)
        return outside_dist + inside_dist

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            qx = float(x[0] - self.center[0])
            qy = float(x[1] - self.center[1])
            ax = abs(qx)
            ay = abs(qy)
            dx = ax - self.hx
            dy = ay - self.hy
            if dx > 0.0 or dy > 0.0:
                clx = float(np.clip(qx, -self.hx, self.hx))
                cly = float(np.clip(qy, -self.hy, self.hy))
                vx = qx - clx
                vy = qy - cly
                nrm = math.hypot(vx, vy)
                if nrm <= 1e-30:
                    return np.array([1.0, 0.0])
                return np.array([vx / nrm, vy / nrm])
            # inside: normal of nearest face (ties arbitrary but stable)
            if dx >= dy:
                sx = 1.0 if qx >= 0.0 else -1.0
                return np.array([sx, 0.0])
            sy = 1.0 if qy >= 0.0 else -1.0
            return np.array([0.0, sy])

        qx = x[..., 0] - self.center[0]
        qy = x[..., 1] - self.center[1]
        ax = np.abs(qx)
        ay = np.abs(qy)
        dx = ax - self.hx
        dy = ay - self.hy
        outside = (dx > 0.0) | (dy > 0.0)

        # outside: direction from closest point on box to x
        clx = np.clip(qx, -self.hx, self.hx)
        cly = np.clip(qy, -self.hy, self.hy)
        vx = qx - clx
        vy = qy - cly
        nrm = np.hypot(vx, vy)
        inv = np.divide(1.0, nrm, out=np.zeros_like(nrm), where=nrm > 1e-30)
        gx_out = vx * inv
        gy_out = vy * inv

        # inside: normal of nearest face
        choose_x = dx >= dy
        sx = np.where(qx >= 0.0, 1.0, -1.0)
        sy = np.where(qy >= 0.0, 1.0, -1.0)
        gx_in = np.where(choose_x, sx, 0.0)
        gy_in = np.where(choose_x, 0.0, sy)

        gx = np.where(outside, gx_out, gx_in)
        gy = np.where(outside, gy_out, gy_in)
        return np.stack((gx, gy), axis=-1)

    def evaluate_on_nodes(self, mesh) -> np.ndarray:
        return self(mesh.nodes_x_y_pos)


class _LinearTransformLevelSet:
    """Affine image of a reference level set: x = c + F (X - c) + t."""

    def __init__(
        self,
        base,
        *,
        F: np.ndarray,
        center: tuple[float, float],
        translation: tuple[float, float] = (0.0, 0.0),
        token: str = "xfm",
    ):
        self.base = base
        self.center = np.asarray(center, dtype=float)
        self.translation = np.asarray(translation, dtype=float)
        self.F = np.asarray(F, dtype=float).reshape(2, 2)
        self.F_inv = np.linalg.inv(self.F)
        tok_base = getattr(base, "cache_token", ("base", id(base)))
        self.cache_token = (
            token,
            tok_base,
            float(self.center[0]),
            float(self.center[1]),
            float(self.translation[0]),
            float(self.translation[1]),
            tuple(float(v) for v in self.F.ravel()),
        )

    def _to_ref(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        v = x - self.translation - self.center
        return self.center + v @ self.F_inv.T

    def __call__(self, x: np.ndarray) -> np.ndarray:
        X = self._to_ref(x)
        return self.base(X)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        X = self._to_ref(x)
        g_ref = self.base.gradient(X)
        return np.asarray(g_ref, dtype=float) @ self.F_inv

    def evaluate_on_nodes(self, mesh) -> np.ndarray:
        return self(mesh.nodes_x_y_pos)


class _EdgeEndpointZeroLevelSet:
    """
    φ(x,y) = (x-x0) + α (y-y0)(y-y1)

    On the vertical line x=x0, φ=0 at y=y0 and y=y1, but φ≠0 at the midpoint.
    This reproduces the "endpoints on Γ, mid-edge node off Γ" degeneracy.
    """

    def __init__(self, *, x0: float, y0: float, y1: float, alpha: float):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.y1 = float(y1)
        self.alpha = float(alpha)
        self.cache_token = ("edge_endpoint_zero", self.x0, self.y0, self.y1, self.alpha)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        yy = x[..., 1]
        return (x[..., 0] - self.x0) + self.alpha * (yy - self.y0) * (yy - self.y1)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        yy = x[..., 1]
        gx = np.ones_like(yy)
        gy = self.alpha * (2.0 * yy - (self.y0 + self.y1))
        return np.stack((gx, gy), axis=-1) if x.ndim > 1 else np.array([float(gx), float(gy)])

    def evaluate_on_nodes(self, mesh) -> np.ndarray:
        return self(mesh.nodes_x_y_pos)


class _BoxWithWindowedTopBumpLevelSet:
    """
    Polygonal "box" level set with one top-edge window replaced by a small triangular bump.

    This creates a mixed configuration where:
      - many edges remain fully aligned (tagged as 'interface' edges), and
      - a local region introduces true cut cells (with interface segments).
    """

    def __init__(
        self,
        *,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        xw0: float,
        xw1: float,
        bump: float,
        token: str = "box_window_bump",
    ):
        self.x0 = float(x0)
        self.x1 = float(x1)
        self.y0 = float(y0)
        self.y1 = float(y1)
        self.xw0 = float(xw0)
        self.xw1 = float(xw1)
        self.bump = float(bump)
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError("Invalid box extents.")
        if not (self.x0 <= self.xw0 <= self.xw1 <= self.x1):
            raise ValueError("Window must lie on the top edge interval [x0,x1].")
        self.xm = 0.5 * (self.xw0 + self.xw1)
        self.cache_token = (
            token,
            self.x0,
            self.x1,
            self.y0,
            self.y1,
            self.xw0,
            self.xw1,
            self.bump,
        )

    def _tri_bump(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if abs(self.bump) < 1e-30 or self.xw1 <= self.xw0:
            return np.zeros_like(x)
        left = (x - self.xw0) / max(self.xm - self.xw0, 1e-30)
        right = (self.xw1 - x) / max(self.xw1 - self.xm, 1e-30)
        tri = np.where(x <= self.xm, left, right)
        tri = np.clip(tri, 0.0, 1.0)
        return self.bump * tri

    def _tri_bump_prime(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if abs(self.bump) < 1e-30 or self.xw1 <= self.xw0:
            return np.zeros_like(x)
        left_slope = self.bump / max(self.xm - self.xw0, 1e-30)
        right_slope = -self.bump / max(self.xw1 - self.xm, 1e-30)
        out = np.zeros_like(x)
        out = np.where((x >= self.xw0) & (x < self.xm), left_slope, out)
        out = np.where((x > self.xm) & (x <= self.xw1), right_slope, out)
        return out

    def _y_top(self, x: np.ndarray) -> np.ndarray:
        return self.y1 + self._tri_bump(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        xx = x[..., 0]
        yy = x[..., 1]
        g_left = self.x0 - xx
        g_right = xx - self.x1
        g_bottom = self.y0 - yy
        g_top = yy - self._y_top(xx)
        return np.maximum.reduce([g_left, g_right, g_bottom, g_top])

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        xx = x[..., 0]
        yy = x[..., 1]
        g_left = self.x0 - xx
        g_right = xx - self.x1
        g_bottom = self.y0 - yy
        top_y = self._y_top(xx)
        g_top = yy - top_y
        vals = np.stack((g_left, g_right, g_bottom, g_top), axis=-1)
        k = np.argmax(vals, axis=-1)
        # Gradients of the half-space constraints.
        gx = np.zeros_like(xx)
        gy = np.zeros_like(yy)
        gx = np.where(k == 0, -1.0, gx)
        gx = np.where(k == 1, 1.0, gx)
        gy = np.where(k == 2, -1.0, gy)
        if np.any(k == 3):
            dtop = self._tri_bump_prime(xx)
            gx = np.where(k == 3, -dtop, gx)
            gy = np.where(k == 3, 1.0, gy)
        return np.stack((gx, gy), axis=-1) if x.ndim > 1 else np.array([float(gx), float(gy)])

    def evaluate_on_nodes(self, mesh) -> np.ndarray:
        return self(mesh.nodes_x_y_pos)


def _box_window_bump_area_perimeter(
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    xw0: float,
    xw1: float,
    bump: float,
) -> tuple[float, float]:
    w = float(x1 - x0)
    h = float(y1 - y0)
    ww = float(xw1 - xw0)
    bump = float(bump)
    area = w * h + 0.5 * ww * bump
    base_perim = 2.0 * (w + h)
    if ww <= 0.0 or abs(bump) < 1e-30:
        return area, base_perim
    seg = math.hypot(0.5 * ww, bump)
    perim = base_perim - ww + 2.0 * seg
    return float(area), float(perim)


def _rotation(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=float)


def _parallelogram_area_perimeter(*, L: float, H: float, F: np.ndarray) -> tuple[float, float]:
    F = np.asarray(F, dtype=float).reshape(2, 2)
    e1 = F @ np.array([float(L), 0.0], dtype=float)
    e2 = F @ np.array([0.0, float(H)], dtype=float)
    area = abs(float(np.linalg.det(F))) * float(L) * float(H)
    perim = 2.0 * (float(np.linalg.norm(e1)) + float(np.linalg.norm(e2)))
    return area, perim


def _parabola_area_length(*, x0: float, y0: float, y1: float, alpha: float) -> tuple[float, float]:
    """
    Domain is [0,1]x[0,1] and interface is x(y) = x0 - α (y-y0)(y-y1).
    Assumes x(y)∈[0,1] for all y∈[0,1] (no clipping).
    """
    x0 = float(x0)
    y0 = float(y0)
    y1 = float(y1)
    a = float(alpha)
    # ∫ (y-y0)(y-y1) dy from 0..1
    I2 = (1.0 / 3.0) - 0.5 * (y0 + y1) + (y0 * y1)
    area_inside = x0 - a * I2

    if abs(a) < 1e-30:
        return area_inside, 1.0

    s = y0 + y1
    t0 = -s
    t1 = 2.0 - s

    def H(t: float) -> float:
        return 0.5 * t * math.sqrt(1.0 + (a * t) ** 2) + math.asinh(a * t) / (2.0 * a)

    length = 0.5 * (H(t1) - H(t0))
    return area_inside, float(length)


class _VertexHitLevelSet:
    """
    Construct a 'grazing' case on the vertical edge x=0.5 of a Q2 mesh:
      φ(x,y) = (x-0.5) + α*y*(1-y)
    so that φ=0 at the edge endpoints (y=0,1) but φ>0 at the mid-edge node (y=0.5).
    This must NOT be treated as a fully aligned interface edge.
    """

    def __init__(self, *, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.cache_token = ("vertex_hit", self.alpha)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        xx = x[..., 0]
        yy = x[..., 1]
        return (xx - 0.5) + self.alpha * yy * (1.0 - yy)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        yy = x[..., 1]
        gx = np.ones_like(yy)
        gy = self.alpha * (1.0 - 2.0 * yy)
        return np.stack((gx, gy), axis=-1) if x.ndim > 1 else np.array([float(gx), float(gy)])

    def evaluate_on_nodes(self, mesh) -> np.ndarray:
        coords = mesh.nodes_x_y_pos
        return (coords[:, 0] - 0.5) + self.alpha * coords[:, 1] * (1.0 - coords[:, 1])


def test_interface_alignment_requires_all_edge_nodes_q2():
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )

    ls = _VertexHitLevelSet(alpha=1.0)
    mesh.classify_elements(ls, tol=1e-12)
    mesh.classify_edges(ls, tol=1e-12)

    assert mesh.edge_bitset("interface").cardinality() == 0
    assert mesh.element_bitset("cut").cardinality() > 0
    assert mesh.edge_bitset("ghost_pos").cardinality() > 0


@pytest.mark.parametrize("backend", ["python", "jit"])
def test_interface_length_continuous_through_alignment_switch(backend):
    Lx, Ly = 2.0, 1.0
    nx, ny = 10, 4
    nodes, elems, edges, corners = structured_quad(Lx, Ly, nx=nx, ny=ny, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    h = Lx / nx
    eps_list = (-h / 1000.0, 0.0, +h / 1000.0)
    length_vals = []
    nx_vals = []
    n = FacetNormal()

    for eps in eps_list:
        ls = AffineLevelSet(1.0, 0.0, -(1.0 + float(eps)))
        mesh.classify_elements(ls, tol=1e-12)
        mesh.classify_edges(ls, tol=1e-12)
        mesh.build_interface_segments(ls, tol=1e-12)

        if abs(eps) < 1e-30:
            assert mesh.edge_bitset("interface").cardinality() > 0
            assert mesh.element_bitset("cut").cardinality() == 0
        else:
            assert mesh.edge_bitset("interface").cardinality() == 0
            assert mesh.element_bitset("cut").cardinality() > 0

        I_len = Constant(1.0) * dInterface(level_set=ls, metadata={"q": 4})
        I_nx = n[0] * dInterface(level_set=ls, metadata={"q": 4})
        length_vals.append(_assemble_scalar(I_len, dh, backend=backend))
        nx_vals.append(_assemble_scalar(I_nx, dh, backend=backend))

    assert_allclose(length_vals, [Ly, Ly, Ly], rtol=2e-6, atol=2e-6)
    assert_allclose(nx_vals, [Ly, Ly, Ly], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("backend", ["python", "jit"])
@pytest.mark.parametrize("bump", [0.0, 0.04])
def test_mixed_aligned_and_cut_interface_integrals_windowed_bump(backend: str, bump: float):
    """
    Mixed aligned+cut configuration:
      - start with an axis-aligned box (all interface facets are aligned edges, no cut cells),
      - introduce a small local window bump on the top edge, creating both:
          (i) aligned interface edges away from the window, and
          (ii) true cut cells within the window region.

    Validate cut volume partitioning and total interface length in this hybrid state.
    """
    Lx = Ly = 1.0
    nx = ny = 8
    nodes, elems, edges, corners = structured_quad(Lx, Ly, nx=nx, ny=ny, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    x0, x1 = 0.25, 0.75
    y0, y1 = 0.25, 0.75
    xw0, xw1 = 0.50, 0.75  # aligns with mesh edges for nx=8, Q2 grid

    ls = _BoxWithWindowedTopBumpLevelSet(x0=x0, x1=x1, y0=y0, y1=y1, xw0=xw0, xw1=xw1, bump=float(bump))
    _refresh_geometry(mesh, ls, tol=1e-12)

    iface_edges = int(mesh.edge_bitset("interface").cardinality())
    cut_elems = int(mesh.element_bitset("cut").cardinality())

    if abs(float(bump)) < 1e-30:
        assert iface_edges > 0
        assert cut_elems == 0
    else:
        assert iface_edges > 0
        assert cut_elems > 0

    inside = mesh.element_bitset("inside")
    outside = mesh.element_bitset("outside")
    cut = mesh.element_bitset("cut")
    q = 10
    dx_neg = dx(defined_on=inside | cut, level_set=ls, metadata={"side": "-", "q": q})
    dx_pos = dx(defined_on=outside | cut, level_set=ls, metadata={"side": "+", "q": q})

    A_neg = _assemble_scalar(Constant(1.0) * dx_neg, dh, backend=backend)
    A_pos = _assemble_scalar(Constant(1.0) * dx_pos, dh, backend=backend)
    assert_allclose(A_neg + A_pos, Lx * Ly, rtol=2e-6, atol=2e-6)

    A_ref, L_ref = _box_window_bump_area_perimeter(x0=x0, x1=x1, y0=y0, y1=y1, xw0=xw0, xw1=xw1, bump=bump)
    if abs(float(bump)) < 1e-30:
        assert_allclose(A_neg, A_ref, rtol=0.0, atol=1e-12)
    else:
        assert_allclose(A_neg, A_ref, rtol=5e-4, atol=5e-4)

    L_ifc = _assemble_scalar(Constant(1.0) * dInterface(level_set=ls, metadata={"q": q + 2}), dh, backend=backend)
    assert_allclose(L_ifc, L_ref, rtol=8e-4, atol=8e-4)


@pytest.mark.parametrize("backend", ["python", "jit"])
@pytest.mark.parametrize("alpha", [+1.0, -1.0])
def test_edge_endpoints_zero_midnode_nonzero_integrates_correctly(backend: str, alpha: float):
    """
    Reproduce the problematic configuration on a *single interior mesh edge*:
      φ(node0)=φ(node1)=0 but φ(mid-edge node)≠0.

    Validate that:
      - the edge is NOT classified as an aligned 'interface' facet,
      - area and interface-length integrals remain correct,
      - the scenario works for both signs (edge interior on fluid/solid side).
    """
    Lx = Ly = 1.0
    nx = ny = 16
    nodes, elems, edges, corners = structured_quad(Lx, Ly, nx=nx, ny=ny, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    x0 = 0.5
    j = ny // 2
    y0 = float(j) / float(ny)
    y1 = float(j + 1) / float(ny)
    ls = _EdgeEndpointZeroLevelSet(x0=x0, y0=y0, y1=y1, alpha=float(alpha))
    _refresh_geometry(mesh, ls, tol=1e-12)

    # Find the specific interior edge segment on x=x0 between y0 and y1.
    target_edge = None
    for e in mesh.edges_list:
        if e.right is None:
            continue
        xy = mesh.nodes_x_y_pos[list(e.nodes)]
        if not np.allclose(xy[:, 0], x0, atol=1e-14):
            continue
        ys = sorted([float(v) for v in xy[:, 1]])
        if np.allclose(ys, [y0, y1], atol=1e-14):
            target_edge = e
            break
    assert target_edge is not None

    edge_nodes = tuple(int(n) for n in (getattr(target_edge, "all_nodes", None) or target_edge.nodes))
    phi_edge = ls(mesh.nodes_x_y_pos[list(edge_nodes)])
    assert abs(float(phi_edge[0])) <= 1e-12
    assert abs(float(phi_edge[-1])) <= 1e-12
    assert abs(float(phi_edge[len(phi_edge) // 2])) > 1.0e-8
    assert getattr(target_edge, "tag", "") != "interface"

    # Area and interface length against analytic references (domain = unit square).
    expected_area, expected_len = _parabola_area_length(x0=x0, y0=y0, y1=y1, alpha=float(alpha))

    inside = mesh.element_bitset("inside")
    outside = mesh.element_bitset("outside")
    cut = mesh.element_bitset("cut")

    qvol = 8
    qedge = 10
    dx_neg = dx(defined_on=inside | cut, level_set=ls, metadata={"side": "-", "q": qvol})
    dx_pos = dx(defined_on=outside | cut, level_set=ls, metadata={"side": "+", "q": qvol})

    A_neg = _assemble_scalar(Constant(1.0) * dx_neg, dh, backend=backend)
    A_pos = _assemble_scalar(Constant(1.0) * dx_pos, dh, backend=backend)
    L_ifc = _assemble_scalar(Constant(1.0) * dInterface(level_set=ls, metadata={"q": qedge}), dh, backend=backend)

    assert_allclose(A_neg + A_pos, Lx * Ly, rtol=5e-6, atol=5e-6)
    assert_allclose(A_neg, expected_area, rtol=8e-3, atol=2e-3)
    assert_allclose(L_ifc, expected_len, rtol=8e-3, atol=2e-3)


@pytest.mark.parametrize("backend", ["python", "jit"])
def test_box_rigid_rotation_preserves_area_and_perimeter(backend: str):
    Lx = Ly = 1.0
    nx = ny = 18
    nodes, elems, edges, corners = structured_quad(Lx, Ly, nx=nx, ny=ny, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    box_L = 0.5
    box_H = 0.25
    center = (0.5, 0.5)
    base = _SignedDistanceBoxLevelSet(center=center, L=box_L, H=box_H)
    expected_area, expected_perim = _parallelogram_area_perimeter(L=box_L, H=box_H, F=np.eye(2))

    angles = [0.0, 0.08, 0.22, 0.41]
    areas = []
    perims = []
    for theta in angles:
        ls = _LinearTransformLevelSet(base, F=_rotation(theta), center=center)
        _refresh_geometry(mesh, ls, tol=1e-12)

        inside = mesh.element_bitset("inside")
        cut = mesh.element_bitset("cut")
        dx_neg = dx(defined_on=inside | cut, level_set=ls, metadata={"side": "-", "q": 8})
        A = _assemble_scalar(Constant(1.0) * dx_neg, dh, backend=backend)
        P = _assemble_scalar(Constant(1.0) * dInterface(level_set=ls, metadata={"q": 10}), dh, backend=backend)
        areas.append(A)
        perims.append(P)

    assert_allclose(areas, [expected_area] * len(areas), rtol=6e-3, atol=2e-3)
    assert_allclose(perims, [expected_perim] * len(perims), rtol=6e-3, atol=2e-3)


@pytest.mark.parametrize("backend", ["python", "jit"])
def test_box_full_alignment_switch_no_jump_in_integrals(backend: str):
    """
    Shift an initially grid-aligned box by a tiny (ε,ε) to force the assembly
    path to switch from fully-aligned facets to cut-cell interface quadrature.
    """
    Lx = Ly = 1.0
    nx = ny = 16
    nodes, elems, edges, corners = structured_quad(Lx, Ly, nx=nx, ny=ny, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    box_L = 0.5
    box_H = 0.25
    center = (0.5, 0.5)
    base = _SignedDistanceBoxLevelSet(center=center, L=box_L, H=box_H)
    expected_area, expected_perim = _parallelogram_area_perimeter(L=box_L, H=box_H, F=np.eye(2))

    h = Lx / nx
    eps_list = (-h / 1000.0, 0.0, +h / 1000.0)
    areas = []
    perims = []
    iface_counts = []
    cut_counts = []
    for eps in eps_list:
        ls = _LinearTransformLevelSet(base, F=np.eye(2), center=center, translation=(float(eps), float(eps)))
        _refresh_geometry(mesh, ls, tol=1e-12)

        cut = mesh.element_bitset("cut")
        iface = mesh.edge_bitset("interface")
        cut_counts.append(int(cut.cardinality()))
        iface_counts.append(int(iface.cardinality()))

        inside = mesh.element_bitset("inside")
        dx_neg = dx(defined_on=inside | cut, level_set=ls, metadata={"side": "-", "q": 8})
        A = _assemble_scalar(Constant(1.0) * dx_neg, dh, backend=backend)
        P = _assemble_scalar(Constant(1.0) * dInterface(level_set=ls, metadata={"q": 10}), dh, backend=backend)
        areas.append(A)
        perims.append(P)

    assert iface_counts[1] > 0
    assert cut_counts[1] == 0
    assert iface_counts[0] == 0 and iface_counts[2] == 0
    assert cut_counts[0] > 0 and cut_counts[2] > 0

    assert_allclose(areas, [expected_area] * len(areas), rtol=6e-3, atol=2e-3)
    assert_allclose(perims, [expected_perim] * len(perims), rtol=6e-3, atol=2e-3)


@pytest.mark.parametrize("backend", ["python", "jit"])
@pytest.mark.parametrize("gamma", [0.5, -0.75])
def test_box_pure_shear_matches_area_and_perimeter(backend: str, gamma: float):
    Lx = Ly = 1.0
    nx = ny = 18
    nodes, elems, edges, corners = structured_quad(Lx, Ly, nx=nx, ny=ny, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    box_L = 0.5
    box_H = 0.25
    center = (0.5, 0.5)
    base = _SignedDistanceBoxLevelSet(center=center, L=box_L, H=box_H)
    F = np.array([[1.0, float(gamma)], [0.0, 1.0]], dtype=float)
    expected_area, expected_perim = _parallelogram_area_perimeter(L=box_L, H=box_H, F=F)

    ls = _LinearTransformLevelSet(base, F=F, center=center)
    _refresh_geometry(mesh, ls, tol=1e-12)

    inside = mesh.element_bitset("inside")
    cut = mesh.element_bitset("cut")
    dx_neg = dx(defined_on=inside | cut, level_set=ls, metadata={"side": "-", "q": 8})
    A = _assemble_scalar(Constant(1.0) * dx_neg, dh, backend=backend)
    P = _assemble_scalar(Constant(1.0) * dInterface(level_set=ls, metadata={"q": 10}), dh, backend=backend)

    assert_allclose(A, expected_area, rtol=6e-3, atol=2e-3)
    assert_allclose(P, expected_perim, rtol=6e-3, atol=2e-3)


@pytest.mark.parametrize("backend", ["python", "jit"])
@pytest.mark.parametrize("scale_y", [0.8, 1.2])
def test_box_breathing_scale_matches_area_and_perimeter(backend: str, scale_y: float):
    Lx = Ly = 1.0
    nx = ny = 18
    nodes, elems, edges, corners = structured_quad(Lx, Ly, nx=nx, ny=ny, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    box_L = 0.5
    box_H = 0.25
    center = (0.5, 0.5)
    base = _SignedDistanceBoxLevelSet(center=center, L=box_L, H=box_H)
    F = np.array([[1.0, 0.0], [0.0, float(scale_y)]], dtype=float)
    expected_area, expected_perim = _parallelogram_area_perimeter(L=box_L, H=box_H, F=F)

    # Baseline (scale_y = 1): used to validate the *scaling trend* of the volume
    # integral, which is more robust than an absolute check for non-polynomial φ.
    ls0 = _LinearTransformLevelSet(base, F=np.eye(2), center=center, token="scale_base")
    _refresh_geometry(mesh, ls0, tol=1e-12)
    inside0 = mesh.element_bitset("inside")
    cut0 = mesh.element_bitset("cut")
    dx0 = dx(defined_on=inside0 | cut0, level_set=ls0, metadata={"side": "-", "q": 8})
    A0 = _assemble_scalar(Constant(1.0) * dx0, dh, backend=backend)

    ls = _LinearTransformLevelSet(base, F=F, center=center)
    _refresh_geometry(mesh, ls, tol=1e-12)

    inside = mesh.element_bitset("inside")
    cut = mesh.element_bitset("cut")
    dx_neg = dx(defined_on=inside | cut, level_set=ls, metadata={"side": "-", "q": 8})
    A = _assemble_scalar(Constant(1.0) * dx_neg, dh, backend=backend)
    P = _assemble_scalar(Constant(1.0) * dInterface(level_set=ls, metadata={"q": 10}), dh, backend=backend)

    assert A0 > 0.0
    assert_allclose(A / A0, float(scale_y), rtol=3e-2, atol=2e-2)
    assert_allclose(P, expected_perim, rtol=6e-3, atol=2e-3)


def test_q2_edge_two_internal_roots_even_if_endpoints_same_sign():
    """
    Parabolic-dip edge restriction (p=2): φ endpoints share the same sign, but
    φ has two interior roots along the edge. The root finder must detect both.
    """
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    me = MixedElement(mesh, field_specs={"phi": 2})
    dh = DofHandler(me, method="cg")
    ls = LevelSetGridFunction(dh, field="phi")

    delta = 0.04  # roots at x=0.5±0.2 on the bottom edge
    ls.interpolate(lambda x, y: (x - 0.5) ** 2 - float(delta))

    pts = edge_root_pn(ls, mesh, eid=0, local_edge=0, tol=1e-12)
    assert len(pts) == 2
    xs = sorted(float(p[0]) for p in pts)
    ys = [float(p[1]) for p in pts]
    assert_allclose(xs, [0.5 - math.sqrt(delta), 0.5 + math.sqrt(delta)], atol=2e-8, rtol=0.0)
    assert_allclose(ys, [0.0, 0.0], atol=2e-12, rtol=0.0)


@pytest.mark.parametrize("backend", ["python", "jit"])
def test_q2_whole_edge_on_interface_integrates_exact_edge_length(backend: str):
    """
    Whole-edge-on-Γ degenerate case: φ=y-0.5 makes the *interior* horizontal edge Γ.
    Ensure aligned-interface extraction/integration returns the exact length.
    """
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=1, ny=2, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    me = MixedElement(mesh, field_specs={"u": 1, "phi": 2})
    dh = DofHandler(me, method="cg")
    ls = LevelSetGridFunction(dh, field="phi")
    ls.interpolate(lambda x, y: float(y) - 0.5)

    _refresh_geometry(mesh, ls, tol=1e-12)
    assert mesh.edge_bitset("interface").cardinality() > 0
    assert mesh.element_bitset("cut").cardinality() == 0

    L = _assemble_scalar(Constant(1.0) * dInterface(level_set=ls, metadata={"q": 6}), dh, backend=backend)
    assert_allclose(L, 1.0, rtol=1e-12, atol=1e-12)
