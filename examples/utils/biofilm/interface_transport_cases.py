"""Exact kinematics for reduced interface-transport benchmarks.

These cases are used for Benchmark 2 in Paper 1. They isolate the conservative
interface-transport claim by prescribing divergence-free motions and by setting
the Cahn--Hilliard mobility to zero in the benchmark drivers. In that regime,
the exact diffuse indicator is a pure pull-back of the initial profile, so any
observed geometry error is a transport/discretization artifact rather than
curvature-driven phase relaxation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _as_array(x):
    return np.asarray(x, dtype=float)


def _disk_alpha(x_ref, y_ref, *, xc0: float, yc0: float, radius: float, eps_alpha: float):
    x_ref = _as_array(x_ref)
    y_ref = _as_array(y_ref)
    r = np.sqrt((x_ref - float(xc0)) ** 2 + (y_ref - float(yc0)) ** 2)
    return 0.5 * (1.0 - np.tanh((r - float(radius)) / float(eps_alpha)))


@dataclass(frozen=True)
class InterfaceTransportCase:
    case_id: str
    title: str
    geometry: str
    t_final: float
    snapshot_times: tuple[float, ...]
    eps_alpha: float
    gamma_alpha: float
    M_alpha: float
    centroid0: tuple[float, float]
    radius: float
    speed_scale: float
    alpha: callable
    velocity: callable
    div_velocity: callable
    centroid_exact: callable


def build_translation_case() -> InterfaceTransportCase:
    xc0 = 0.28
    yc0 = 0.56
    radius = 0.12
    eps_alpha = 0.04
    speed = 0.35
    t_final = 0.60

    def alpha(x, y, t):
        return _disk_alpha(_as_array(x) - speed * float(t), y, xc0=xc0, yc0=yc0, radius=radius, eps_alpha=eps_alpha)

    def velocity(x, y, t):
        x = _as_array(x)
        y = _as_array(y)
        shape = np.broadcast(x, y).shape
        return np.stack([np.full(shape, speed, dtype=float), np.zeros(shape, dtype=float)], axis=-1)

    def div_velocity(x, y, t):
        return np.zeros(np.broadcast(_as_array(x), _as_array(y)).shape, dtype=float)

    def centroid_exact(t):
        return np.array([xc0 + speed * float(t), yc0], dtype=float)

    return InterfaceTransportCase(
        case_id="translation",
        title="Rigid translation of a diffuse disk",
        geometry="Unit square with a diffuse circular patch transported horizontally.",
        t_final=t_final,
        snapshot_times=(0.0, 0.20, 0.40, t_final),
        eps_alpha=eps_alpha,
        gamma_alpha=2.0e-3,
        M_alpha=0.0,
        centroid0=(xc0, yc0),
        radius=radius,
        speed_scale=abs(speed),
        alpha=alpha,
        velocity=velocity,
        div_velocity=div_velocity,
        centroid_exact=centroid_exact,
    )


def build_rotation_case() -> InterfaceTransportCase:
    xc0 = 0.68
    yc0 = 0.50
    radius = 0.10
    eps_alpha = 0.04
    center = np.array([0.50, 0.50], dtype=float)
    t_final = 1.0
    omega = 2.0 * math.pi / t_final

    def _rotate_to_reference(x, y, t):
        theta = -omega * float(t)
        ct = math.cos(theta)
        st = math.sin(theta)
        xr = _as_array(x) - center[0]
        yr = _as_array(y) - center[1]
        x_ref = center[0] + ct * xr - st * yr
        y_ref = center[1] + st * xr + ct * yr
        return x_ref, y_ref

    def alpha(x, y, t):
        x_ref, y_ref = _rotate_to_reference(x, y, t)
        return _disk_alpha(x_ref, y_ref, xc0=xc0, yc0=yc0, radius=radius, eps_alpha=eps_alpha)

    def velocity(x, y, t):
        x = _as_array(x)
        y = _as_array(y)
        return np.stack(
            [
                -omega * (y - center[1]),
                omega * (x - center[0]),
            ],
            axis=-1,
        )

    def div_velocity(x, y, t):
        return np.zeros(np.broadcast(_as_array(x), _as_array(y)).shape, dtype=float)

    def centroid_exact(t):
        theta = omega * float(t)
        ct = math.cos(theta)
        st = math.sin(theta)
        rel = np.array([xc0, yc0], dtype=float) - center
        return center + np.array([ct * rel[0] - st * rel[1], st * rel[0] + ct * rel[1]], dtype=float)

    return InterfaceTransportCase(
        case_id="rotation",
        title="Large rigid-body rotation of a diffuse disk",
        geometry="Unit square with an off-center diffuse disk making one full revolution about the box center.",
        t_final=t_final,
        snapshot_times=(0.0, 0.25, 0.50, t_final),
        eps_alpha=eps_alpha,
        gamma_alpha=2.0e-3,
        M_alpha=0.0,
        centroid0=(xc0, yc0),
        radius=radius,
        speed_scale=omega * (np.linalg.norm(np.array([xc0, yc0], dtype=float) - center) + radius),
        alpha=alpha,
        velocity=velocity,
        div_velocity=div_velocity,
        centroid_exact=centroid_exact,
    )


def build_shear_return_case() -> InterfaceTransportCase:
    xc0 = 0.38
    yc0 = 0.68
    radius = 0.10
    eps_alpha = 0.04
    y_ref = 0.50
    t_final = 1.0
    k_max = 1.75

    def _kappa(t):
        return k_max * math.sin(math.pi * float(t) / t_final)

    def _kappa_dot(t):
        return k_max * (math.pi / t_final) * math.cos(math.pi * float(t) / t_final)

    def alpha(x, y, t):
        x = _as_array(x)
        y = _as_array(y)
        x_ref = x - _kappa(t) * (y - y_ref)
        return _disk_alpha(x_ref, y, xc0=xc0, yc0=yc0, radius=radius, eps_alpha=eps_alpha)

    def velocity(x, y, t):
        x = _as_array(x)
        y = _as_array(y)
        shape = np.broadcast(x, y).shape
        return np.stack([_kappa_dot(t) * (y - y_ref), np.zeros(shape, dtype=float)], axis=-1)

    def div_velocity(x, y, t):
        return np.zeros(np.broadcast(_as_array(x), _as_array(y)).shape, dtype=float)

    def centroid_exact(t):
        return np.array([xc0 + _kappa(t) * (yc0 - y_ref), yc0], dtype=float)

    return InterfaceTransportCase(
        case_id="shear_return",
        title="Affine shear-and-return transport",
        geometry="Unit square with a diffuse disk subjected to a large affine shear that returns to the reference shape at final time.",
        t_final=t_final,
        snapshot_times=(0.0, 0.50, 0.75, t_final),
        eps_alpha=eps_alpha,
        gamma_alpha=2.0e-3,
        M_alpha=0.0,
        centroid0=(xc0, yc0),
        radius=radius,
        speed_scale=abs(k_max * math.pi * (yc0 - y_ref) / t_final),
        alpha=alpha,
        velocity=velocity,
        div_velocity=div_velocity,
        centroid_exact=centroid_exact,
    )


def build_interface_transport_case(case: str) -> InterfaceTransportCase:
    key = str(case).strip().lower()
    if key == "translation":
        return build_translation_case()
    if key == "rotation":
        return build_rotation_case()
    if key in {"shear", "shear_return", "shear-return"}:
        return build_shear_return_case()
    raise ValueError(f"Unknown interface-transport case {case!r}.")
