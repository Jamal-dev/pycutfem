from __future__ import annotations

import numpy as np

from .cpp_backend.iqnils import module as _iqnils_cpp_module


def kratos_iqnils_next_iterate_cpp(
    *,
    x_curr: np.ndarray,
    g_curr: np.ndarray,
    x_history: list[np.ndarray],
    g_history: list[np.ndarray],
    dr_old_mats: list[np.ndarray] | None = None,
    dg_old_mats: list[np.ndarray] | None = None,
    alpha: float,
    horizon: int,
    regularization: float = 0.0,
) -> np.ndarray:
    return np.asarray(
        _iqnils_cpp_module().next_iterate(
            np.asarray(x_curr, dtype=float),
            np.asarray(g_curr, dtype=float),
            [np.asarray(values, dtype=float) for values in list(x_history)],
            [np.asarray(values, dtype=float) for values in list(g_history)],
            [np.asarray(block, dtype=float) for block in list(dr_old_mats or [])],
            [np.asarray(block, dtype=float) for block in list(dg_old_mats or [])],
            float(alpha),
            int(horizon),
            float(regularization),
        ),
        dtype=float,
    )


def kratos_iqnils_iteration_matrices_cpp(
    *,
    x_history: list[np.ndarray],
    g_history: list[np.ndarray],
    iteration_horizon: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    raw_v, raw_w = _iqnils_cpp_module().iteration_matrices(
        [np.asarray(values, dtype=float) for values in list(x_history)],
        [np.asarray(values, dtype=float) for values in list(g_history)],
        int(iteration_horizon),
    )
    v_new = None if raw_v is None else np.asarray(raw_v, dtype=float)
    w_new = None if raw_w is None else np.asarray(raw_w, dtype=float)
    return v_new, w_new


__all__ = [
    "kratos_iqnils_iteration_matrices_cpp",
    "kratos_iqnils_next_iterate_cpp",
]
