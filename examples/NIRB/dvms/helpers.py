from __future__ import annotations

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem import transform
from pycutfem.ufl.expressions import ElementWiseConstant, Function, VectorFunction


def _bossak_coefficients(alpha: float, dt: float) -> dict[str, float]:
    alpha_value = float(alpha)
    dt_value = max(float(dt), 1.0e-14)
    gamma = 0.5 - alpha_value
    if gamma <= 0.0:
        raise ValueError(f"Bossak alpha={alpha_value} yields non-positive gamma={gamma}.")
    beta = 0.25 * (1.0 - alpha_value) ** 2
    ma0 = 1.0 / (gamma * dt_value)
    ma2 = (-1.0 + gamma) / gamma
    mam = (1.0 - alpha_value) * ma0
    return {
        "alpha": alpha_value,
        "dt": dt_value,
        "gamma": float(gamma),
        "beta": float(beta),
        "ma0": float(ma0),
        "ma2": float(ma2),
        "mam": float(mam),
    }


def _field_values_on_global_dofs(dh: DofHandler, field_name: str, values: np.ndarray) -> np.ndarray:
    field_dofs = np.asarray(dh.get_field_slice(field_name), dtype=int)
    field_values = np.asarray(values, dtype=float).reshape(-1)
    if field_values.shape[0] != field_dofs.shape[0]:
        raise ValueError(
            f"Field '{field_name}' has {field_values.shape[0]} local values but {field_dofs.shape[0]} global dofs."
        )
    if field_dofs.size == 0:
        return np.zeros((0,), dtype=float)
    scattered = np.zeros((int(np.max(field_dofs)) + 1,), dtype=float)
    scattered[field_dofs] = field_values
    return scattered


def _kratos_dvms_element_size(mesh: Mesh, eid: int) -> float:
    conn = np.asarray(mesh.elements_connectivity[int(eid)], dtype=int).reshape(-1)
    coords = np.asarray(mesh.nodes_x_y_pos[conn], dtype=float)
    if coords.shape != (3, 2):
        area = float(np.asarray(mesh.areas_list, dtype=float).reshape(-1)[int(eid)])
        return float(np.sqrt(max(abs(area), 1.0e-30)))
    area = float(abs(np.asarray(mesh.areas_list, dtype=float).reshape(-1)[int(eid)]))
    edges = np.asarray(
        [
            np.linalg.norm(coords[0] - coords[1]),
            np.linalg.norm(coords[1] - coords[2]),
            np.linalg.norm(coords[2] - coords[0]),
        ],
        dtype=float,
    )
    max_edge = float(np.max(edges)) if edges.size else 0.0
    if max_edge <= 1.0e-30:
        return float(np.sqrt(max(area, 1.0e-30)))
    # Kratos DVMS uses the minimum triangle altitude as the element size in 2D.
    return float((2.0 * area) / max_edge)


def _kratos_dvms_element_size_array(mesh: Mesh) -> np.ndarray:
    conn = np.asarray(mesh.elements_connectivity, dtype=int)
    coords = np.asarray(mesh.nodes_x_y_pos[conn], dtype=float)
    if coords.ndim != 3 or coords.shape[1:] != (3, 2):
        areas = np.asarray(mesh.areas_list, dtype=float).reshape(-1)
        return np.sqrt(np.maximum(np.abs(areas), 1.0e-30))
    areas = np.maximum(np.abs(np.asarray(mesh.areas_list, dtype=float).reshape(-1)), 1.0e-30)
    edge01 = np.linalg.norm(coords[:, 0, :] - coords[:, 1, :], axis=1)
    edge12 = np.linalg.norm(coords[:, 1, :] - coords[:, 2, :], axis=1)
    edge20 = np.linalg.norm(coords[:, 2, :] - coords[:, 0, :], axis=1)
    max_edge = np.maximum.reduce([edge01, edge12, edge20])
    h = np.sqrt(areas)
    mask = max_edge > 1.0e-30
    h[mask] = (2.0 * areas[mask]) / max_edge[mask]
    return h


def _kratos_dvms_element_size_coefficient(mesh: Mesh) -> ElementWiseConstant:
    return ElementWiseConstant(_kratos_dvms_element_size_array(mesh))


def _find_element_containing_point(mesh: Mesh, point: np.ndarray) -> int:
    xy = np.asarray(point, dtype=float)
    for elem in mesh.elements_list:
        verts = np.asarray(mesh.nodes_x_y_pos[list(elem.nodes)], dtype=float)
        if not (
            verts[:, 0].min() - 1.0e-12 <= xy[0] <= verts[:, 0].max() + 1.0e-12
            and verts[:, 1].min() - 1.0e-12 <= xy[1] <= verts[:, 1].max() + 1.0e-12
        ):
            continue
        try:
            xi, eta = transform.inverse_mapping(mesh, elem.id, xy)
        except Exception:
            continue
        if -1.0001 <= xi <= 1.0001 and -1.0001 <= eta <= 1.0001:
            return int(elem.id)
    raise ValueError(f"Point {tuple(point)} not found in mesh.")


def _eval_scalar_with_grad(
    dh: DofHandler,
    mesh: Mesh,
    scalar: Function,
    point: tuple[float, float],
) -> tuple[float, np.ndarray]:
    xy = np.asarray(point, dtype=float)
    eid = _find_element_containing_point(mesh, xy)
    xi, eta = transform.inverse_mapping(mesh, eid, xy)
    me = dh.mixed_element
    local_phi = me.basis(scalar.field_name, float(xi), float(eta))[me.slice(scalar.field_name)]
    local_grad_ref = me.grad_basis(scalar.field_name, float(xi), float(eta))[me.slice(scalar.field_name)]
    local_grad = transform.map_grad_scalar(mesh, eid, local_grad_ref, (float(xi), float(eta)))
    gdofs = dh.element_maps[scalar.field_name][eid]
    vals = scalar.get_nodal_values(gdofs)
    return float(local_phi @ vals), np.asarray(vals, dtype=float) @ np.asarray(local_grad, dtype=float)


def _eval_vector_with_grad(
    dh: DofHandler,
    mesh: Mesh,
    vector: VectorFunction,
    point: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    grads = []
    for component in vector.components:
        value, grad_value = _eval_scalar_with_grad(dh, mesh, component, point)
        values.append(value)
        grads.append(grad_value)
    return np.asarray(values, dtype=float), np.vstack(grads)


__all__ = [
    "_bossak_coefficients",
    "_eval_scalar_with_grad",
    "_eval_vector_with_grad",
    "_field_values_on_global_dofs",
    "_find_element_containing_point",
    "_kratos_dvms_element_size",
    "_kratos_dvms_element_size_array",
    "_kratos_dvms_element_size_coefficient",
]
