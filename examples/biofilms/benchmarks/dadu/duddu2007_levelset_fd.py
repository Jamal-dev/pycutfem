"""
Finite-difference level set helpers for Duddu et al. (2007) XFEM+LS benchmarks.

The paper uses a level set function φ on a finite-difference grid and updates it via:

  φ_t + F_ext ||∇φ|| = 0,

with an upwind Godunov scheme (Eq. 33) and velocity extensions constructed with a
fast marching method (Eq. 32). For reproducible benchmarks without extra deps,
we approximate velocity extension by nearest-interface projection using
`scipy.ndimage.distance_transform_edt` (which is equivalent to a closest-point
extension for a signed-distance φ).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FDGrid:
    x: np.ndarray  # (nx+1,)
    y: np.ndarray  # (ny+1,)

    @property
    def dx(self) -> float:
        if self.x.size < 2:
            return 0.0
        return float(self.x[1] - self.x[0])

    @property
    def dy(self) -> float:
        if self.y.size < 2:
            return 0.0
        return float(self.y[1] - self.y[0])

    @property
    def shape(self) -> tuple[int, int]:
        # (ny+1, nx+1) for array indexing [j,i] = [y,x]
        return int(self.y.size), int(self.x.size)


def make_uniform_grid(*, Lx: float, Ly: float, nx: int, ny: int) -> FDGrid:
    x = np.linspace(0.0, float(Lx), int(nx) + 1)
    y = np.linspace(0.0, float(Ly), int(ny) + 1)
    return FDGrid(x=np.asarray(x, float), y=np.asarray(y, float))


def phi_union_disks_on_wall(
    grid: FDGrid,
    *,
    centers_x: list[float],
    radii: list[float],
    wall_y: float = 0.0,
) -> np.ndarray:
    """
    Signed distance to a union of circles centered at (cx, wall_y).
    Negative inside the union. On the FD grid we only use y>=0, so this
    corresponds to a union of *semi*-circles attached to the wall.
    """
    if len(centers_x) != len(radii):
        raise ValueError("centers_x and radii must have the same length.")
    X, Y = np.meshgrid(grid.x, grid.y, indexing="xy")
    phi = np.full(X.shape, np.inf, dtype=float)
    for cx, r in zip(centers_x, radii):
        d = np.sqrt((X - float(cx)) ** 2 + (Y - float(wall_y)) ** 2) - float(r)
        phi = np.minimum(phi, d)
    return np.asarray(phi, float)


def upwind_godunov_norm_grad(phi: np.ndarray, *, dx: float, dy: float, F: np.ndarray) -> np.ndarray:
    """
    Compute the Godunov upwind approximation of ||∇φ|| as in Duddu (2007) Eq.(33),
    including the sign(F) switching.
    """
    phi = np.asarray(phi, dtype=float)
    F = np.asarray(F, dtype=float)
    if phi.shape != F.shape:
        raise ValueError("phi and F must have the same shape.")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx, dy must be positive.")

    # Backward/forward differences.
    Dxm = np.zeros_like(phi)
    Dxp = np.zeros_like(phi)
    Dym = np.zeros_like(phi)
    Dyp = np.zeros_like(phi)

    Dxm[:, 1:] = (phi[:, 1:] - phi[:, :-1]) / float(dx)
    Dxm[:, 0] = Dxm[:, 1]
    Dxp[:, :-1] = (phi[:, 1:] - phi[:, :-1]) / float(dx)
    Dxp[:, -1] = Dxp[:, -2]

    Dym[1:, :] = (phi[1:, :] - phi[:-1, :]) / float(dy)
    Dym[0, :] = Dym[1, :]
    Dyp[:-1, :] = (phi[1:, :] - phi[:-1, :]) / float(dy)
    Dyp[-1, :] = Dyp[-2, :]

    s = np.sign(F)
    ax = np.maximum.reduce([s * Dxm, -s * Dxp, np.zeros_like(phi)])
    ay = np.maximum.reduce([s * Dym, -s * Dyp, np.zeros_like(phi)])
    return np.sqrt(ax * ax + ay * ay)


def level_set_update(phi: np.ndarray, *, F_ext: np.ndarray, dx: float, dy: float, dt: float) -> np.ndarray:
    """
    One explicit time step for φ_t + F_ext ||∇φ|| = 0.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    g = upwind_godunov_norm_grad(phi, dx=dx, dy=dy, F=F_ext)
    return np.asarray(phi, float) - float(dt) * np.asarray(F_ext, float) * g


def extend_speed_nearest_interface(
    grid: FDGrid,
    *,
    interface_points: np.ndarray,
    interface_speeds: np.ndarray,
) -> np.ndarray:
    """
    Build a speed field on the FD grid by:
      1) assigning speed samples to the nearest grid node,
      2) extending to all nodes via nearest-interface projection using EDT.
    """
    pts = np.asarray(interface_points, dtype=float).reshape(-1, 2)
    spd = np.asarray(interface_speeds, dtype=float).reshape(-1)
    if pts.shape[0] != spd.shape[0]:
        raise ValueError("interface_points and interface_speeds must have compatible lengths.")

    ny, nx = grid.shape
    dx = float(grid.dx)
    dy = float(grid.dy)
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("Grid spacing must be positive.")

    # Accumulate samples on nodes.
    F0 = np.zeros((ny, nx), dtype=float)
    C0 = np.zeros((ny, nx), dtype=float)

    i = np.rint(pts[:, 0] / dx).astype(int)
    j = np.rint(pts[:, 1] / dy).astype(int)
    i = np.clip(i, 0, nx - 1)
    j = np.clip(j, 0, ny - 1)
    for jj, ii, ff in zip(j.tolist(), i.tolist(), spd.tolist()):
        if not np.isfinite(ff):
            continue
        F0[int(jj), int(ii)] += float(ff)
        C0[int(jj), int(ii)] += 1.0

    mask = C0 > 0.0
    if not bool(np.any(mask)):
        raise RuntimeError("No interface speed samples landed on the FD grid.")

    F_iface = np.zeros_like(F0)
    F_iface[mask] = F0[mask] / C0[mask]

    # Nearest-interface extension using Euclidean distance transform.
    from scipy.ndimage import distance_transform_edt

    # distance_transform_edt computes distance to zeros; use ~mask so that interface nodes are zeros.
    _, inds = distance_transform_edt(~mask, return_indices=True)
    jj = np.asarray(inds[0], dtype=int)
    ii = np.asarray(inds[1], dtype=int)
    return np.asarray(F_iface[jj, ii], dtype=float)


def reinitialize_signed_distance(phi: np.ndarray, *, dx: float, dy: float) -> np.ndarray:
    """
    Reinitialize φ to a signed distance function (grid-based EDT).
    """
    phi = np.asarray(phi, dtype=float)
    inside = phi < 0.0
    from scipy.ndimage import distance_transform_edt

    # distances in grid units; scale by spacing (assume dx≈dy in this benchmark).
    # `distance_transform_edt` computes the distance to the nearest *zero*.
    # For `inside=True` this means:
    #   - distance_transform_edt( inside) : distance to outside (False/0) -> nonzero inside, 0 outside
    #   - distance_transform_edt(~inside) : distance to inside  (False/0) -> nonzero outside, 0 inside
    d_to_outside = distance_transform_edt(inside) * float(dx)
    d_to_inside = distance_transform_edt(~inside) * float(dx)
    # Preserve sign convention: φ<0 inside, φ>0 outside
    sd = d_to_inside - d_to_outside
    return np.asarray(sd, dtype=float)
