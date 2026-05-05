"""Nonlocal quadrature-state averaging utilities.

The utilities in this module are intentionally independent of a constitutive
law. They build sparse quadrature-point averaging operators from physical
quadrature coordinates and weights, so nonlinear material models can keep
history variables in :mod:`pycutfem.state` while sharing the same nonlocal
neighbour semantics across examples and applications.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from pycutfem.state.registry import QuadratureLayout


@dataclass(frozen=True)
class NonlocalQuadratureMap:
    """Sparse row-normalized nonlocal averaging map for quadrature fields.

    ``operator`` maps flattened quadrature data ordered as
    ``entity-major, quadrature-minor``. If the input values have shape
    ``(n_entities, n_qp, ...)``, :meth:`apply` returns the same shape.
    """

    operator: sp.csr_matrix
    points: np.ndarray
    weights: np.ndarray
    characteristic_length: float
    n_entities: int
    n_qp: int
    spatial_dim: int
    row_denominator: np.ndarray

    def __post_init__(self) -> None:
        if not sp.isspmatrix_csr(self.operator):
            raise TypeError("operator must be a scipy.sparse.csr_matrix.")
        if self.operator.shape != (self.n_entities * self.n_qp, self.n_entities * self.n_qp):
            raise ValueError(
                "operator shape does not match n_entities*n_qp: "
                f"{self.operator.shape} vs {(self.n_entities * self.n_qp, self.n_entities * self.n_qp)}."
            )
        if self.points.shape != (self.n_entities, self.n_qp, self.spatial_dim):
            raise ValueError(
                f"points expected shape {(self.n_entities, self.n_qp, self.spatial_dim)}, got {self.points.shape}."
            )
        if self.weights.shape != (self.n_entities, self.n_qp):
            raise ValueError(f"weights expected shape {(self.n_entities, self.n_qp)}, got {self.weights.shape}.")

    @property
    def n_points(self) -> int:
        return int(self.operator.shape[0])

    def apply(self, values) -> np.ndarray:
        """Apply the nonlocal average to scalar or tensor quadrature values."""

        arr = np.asarray(values, dtype=float)
        expected_prefix = (int(self.n_entities), int(self.n_qp))
        if arr.shape[:2] != expected_prefix:
            raise ValueError(f"values must start with shape {expected_prefix}, got {arr.shape}.")
        flat = arr.reshape(self.n_points, *arr.shape[2:])
        averaged = self.operator @ flat.reshape(self.n_points, -1)
        return averaged.reshape(arr.shape)


def build_gaussian_nonlocal_quadrature_map(
    points,
    weights,
    *,
    characteristic_length: float,
) -> NonlocalQuadratureMap:
    """Build Kratos-compatible Gaussian nonlocal quadrature averaging.

    The scalar kernel is

    ``w_ij = volume_weight_j * exp(-4 * distance(i, j)^2 / characteristic_length^2)``

    for all source quadrature points within ``characteristic_length`` of the
    receiver point. The row is normalized by ``sum_j w_ij``. Self contribution is
    included naturally with distance zero, matching Kratos' nonlocal damage
    utility.
    """

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 3:
        raise ValueError(f"points must have shape (n_entities, n_qp, dim), got {pts.shape}.")
    if pts.shape[2] < 1:
        raise ValueError("points must have at least one spatial dimension.")

    w = np.asarray(weights, dtype=float)
    if w.shape != pts.shape[:2]:
        raise ValueError(f"weights must have shape {pts.shape[:2]}, got {w.shape}.")
    if np.any(~np.isfinite(pts)):
        raise ValueError("points contain non-finite coordinates.")
    if np.any(~np.isfinite(w)):
        raise ValueError("weights contain non-finite values.")
    if np.any(w <= 0.0):
        raise ValueError("nonlocal quadrature weights must be strictly positive.")

    length = float(characteristic_length)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("characteristic_length must be a positive finite value.")

    n_entities = int(pts.shape[0])
    n_qp = int(pts.shape[1])
    spatial_dim = int(pts.shape[2])
    flat_points = np.ascontiguousarray(pts.reshape(n_entities * n_qp, spatial_dim), dtype=float)
    flat_weights = np.ascontiguousarray(w.reshape(n_entities * n_qp), dtype=float)

    tree = cKDTree(flat_points)
    neighbours = tree.query_ball_point(flat_points, r=length)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    denominators = np.empty((flat_points.shape[0],), dtype=float)
    length2 = length * length
    for receiver, source_ids in enumerate(neighbours):
        if not source_ids:
            raise RuntimeError(f"No nonlocal neighbours found for quadrature point {receiver}.")
        src = np.asarray(source_ids, dtype=np.int64)
        delta = flat_points[src] - flat_points[receiver]
        dist2 = np.einsum("ij,ij->i", delta, delta, optimize=True)
        raw = flat_weights[src] * np.exp(-4.0 * dist2 / length2)
        denom = float(np.sum(raw))
        if not np.isfinite(denom) or denom <= 0.0:
            raise RuntimeError(f"Invalid nonlocal row denominator at quadrature point {receiver}: {denom}.")
        denominators[receiver] = denom
        rows.extend([receiver] * int(src.size))
        cols.extend(int(v) for v in src)
        data.extend(float(v) for v in raw / denom)

    op = sp.csr_matrix(
        (np.asarray(data, dtype=float), (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(flat_points.shape[0], flat_points.shape[0]),
    )
    return NonlocalQuadratureMap(
        operator=op,
        points=np.ascontiguousarray(pts, dtype=float),
        weights=np.ascontiguousarray(w, dtype=float),
        characteristic_length=length,
        n_entities=n_entities,
        n_qp=n_qp,
        spatial_dim=spatial_dim,
        row_denominator=denominators,
    )


def build_volume_nonlocal_quadrature_map(
    dof_handler,
    *,
    quadrature_order: int,
    characteristic_length: float,
    reuse_geometry: bool = True,
) -> tuple[NonlocalQuadratureMap, QuadratureLayout]:
    """Build a nonlocal averaging map from a volume quadrature layout.

    The returned :class:`QuadratureLayout` can be used to register matching
    :class:`~pycutfem.state.registry.QuadratureStateField` instances.
    """

    geo = dof_handler.precompute_geometric_factors(
        int(quadrature_order),
        level_set=lambda *_: 0.0,
        reuse=bool(reuse_geometry),
    )
    points = np.asarray(geo.get("qp_phys"), dtype=float)
    weights = np.asarray(geo.get("qw"), dtype=float)
    qref = np.asarray(geo.get("qp_ref"), dtype=float)
    if points.ndim != 3 or points.shape[2] != 2:
        raise ValueError(f"Volume quadrature points must have shape (n_entities, n_qp, 2), got {points.shape}.")
    if weights.shape != points.shape[:2]:
        raise ValueError(f"Volume quadrature weights must have shape {points.shape[:2]}, got {weights.shape}.")
    if qref.ndim != 2 or qref.shape[0] != points.shape[1]:
        raise ValueError(
            "Volume quadrature reference points are missing or incompatible with physical quadrature data."
        )

    mesh = dof_handler.mixed_element.mesh
    layout = QuadratureLayout(
        entity_kind="volume_cell",
        cell_type=str(mesh.element_type),
        quadrature_order=int(quadrature_order),
        reference_points=qref,
        reference_weights=np.asarray(geo.get("qw_ref", weights[0] / np.maximum(geo.get("detJ", weights)[0], 1.0)), dtype=float)
        if "qw_ref" in geo
        else _reference_weights_from_layout(mesh.element_type, int(quadrature_order)),
    )
    return (
        build_gaussian_nonlocal_quadrature_map(
            points,
            weights,
            characteristic_length=float(characteristic_length),
        ),
        layout,
    )


def _reference_weights_from_layout(cell_type: str, quadrature_order: int) -> np.ndarray:
    from pycutfem.integration.quadrature import volume

    _, weights = volume(str(cell_type), int(quadrature_order))
    return np.asarray(weights, dtype=float)
