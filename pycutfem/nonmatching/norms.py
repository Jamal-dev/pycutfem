from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem import transform
from pycutfem.integration import volume


def scalar_L2_error(
    *,
    dh: DofHandler,
    uh: np.ndarray,
    u_exact: Callable[[float, float], float],
    field: str = "u",
    quad_order: int | None = None,
) -> float:
    """Compute ||uh - u_exact||_{L2(Omega)} by explicit quadrature."""
    mesh = dh.mixed_element.mesh
    me = dh.mixed_element
    uh = np.asarray(uh, dtype=float)

    if quad_order is None:
        p = int(getattr(me, "_field_orders", {}).get(field, 1))
        quad_order = int(2 * p + 2)

    qp, qw = volume(mesh.element_type, int(quad_order))

    acc = 0.0
    for eid in range(len(mesh.elements_list)):
        gd = np.asarray(dh.get_elemental_dofs(int(eid)), dtype=int)
        uel = uh[gd]
        for (xi, eta), w in zip(qp, qw):
            J = transform.jacobian(mesh, int(eid), (float(xi), float(eta)))
            detJ = abs(float(np.linalg.det(J)))
            xq = transform.x_mapping(mesh, int(eid), (float(xi), float(eta)))
            phi = np.asarray(me.basis(field, float(xi), float(eta)), dtype=float).ravel()
            uq = float(phi @ uel)
            ue = float(u_exact(float(xq[0]), float(xq[1])))
            diff = uq - ue
            acc += float(w) * detJ * diff * diff
    return float(np.sqrt(acc))


def scalar_H1_semi_error(
    *,
    dh: DofHandler,
    uh: np.ndarray,
    grad_u_exact: Callable[[float, float], tuple[float, float] | np.ndarray],
    field: str = "u",
    quad_order: int | None = None,
) -> float:
    """Compute |uh - u_exact|_{H1(Omega)} by explicit quadrature.

    Notes
    -----
    `grad_u_exact(x,y)` must return a length-2 iterable (du/dx, du/dy).
    """
    mesh = dh.mixed_element.mesh
    me = dh.mixed_element
    uh = np.asarray(uh, dtype=float)

    if quad_order is None:
        p = int(getattr(me, "_field_orders", {}).get(field, 1))
        quad_order = int(2 * p + 2)

    qp, qw = volume(mesh.element_type, int(quad_order))

    acc = 0.0
    for eid in range(len(mesh.elements_list)):
        gd = np.asarray(dh.get_elemental_dofs(int(eid)), dtype=int)
        uel = uh[gd]
        for (xi, eta), w in zip(qp, qw):
            J = transform.jacobian(mesh, int(eid), (float(xi), float(eta)))
            detJ = abs(float(np.linalg.det(J)))
            Jinv = np.linalg.inv(J)
            xq = transform.x_mapping(mesh, int(eid), (float(xi), float(eta)))
            gphi = np.asarray(me.grad_basis(field, float(xi), float(eta)), dtype=float) @ Jinv
            guh = np.asarray(uel, dtype=float) @ gphi
            gue = np.asarray(grad_u_exact(float(xq[0]), float(xq[1])), dtype=float).ravel()
            diff = guh - gue
            acc += float(w) * detJ * float(diff @ diff)
    return float(np.sqrt(acc))

