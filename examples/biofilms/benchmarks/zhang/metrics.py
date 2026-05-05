"""
Utilities for quantitative comparisons in the Zhang (2008) cavity benchmark.

The Zhang paper reports mostly *contour snapshots*. For reproducible comparisons
we therefore focus on geometry metrics extracted from a diffuse-interface proxy
and a few flow/pressure signature checks at reported times.

All helpers here are NumPy-only so they can run both:
  - in-solver (using `dh.get_dof_coords(...)` + `.nodal_values`), and
  - as pure postprocessing on `.npz` snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InterfaceProfile:
    x: np.ndarray
    h: np.ndarray


def _sorted_unique(values: np.ndarray) -> np.ndarray:
    u = np.unique(np.asarray(values))
    return np.asarray(np.sort(u), dtype=float)


def interface_height_profile(
    *,
    coords: np.ndarray,
    values: np.ndarray,
    level: float = 0.5,
    x_round: int = 10,
) -> InterfaceProfile:
    """
    Build an interface height profile h(x) from nodal data by scanning vertical columns.

    The interface is defined as the y-location where `values(x,y)=level`, found by
    1D linear interpolation between the first pair (from bottom to top) that brackets
    `level`.
    """
    xy = np.asarray(coords, dtype=float)
    v = np.asarray(values, dtype=float).reshape(-1)
    if xy.ndim != 2 or xy.shape[1] < 2 or xy.shape[0] != v.size:
        return InterfaceProfile(x=np.asarray([], dtype=float), h=np.asarray([], dtype=float))

    x = xy[:, 0]
    y = xy[:, 1]
    key = np.round(x, int(x_round))
    x_levels = _sorted_unique(key)
    h_out = np.full(x_levels.shape, np.nan, dtype=float)

    lvl = float(level)
    for i, k in enumerate(x_levels):
        m = key == k
        if not np.any(m):
            continue
        yy = y[m]
        vv = v[m]
        order = np.argsort(yy)
        yy = yy[order]
        vv = vv[order]

        # Find first crossing from bottom: vv[j] >= lvl and vv[j+1] < lvl.
        # If all vv >= lvl, the column is "fully biofilm" → use max y.
        if np.all(vv >= lvl):
            h_out[i] = float(np.max(yy))
            continue
        if np.all(vv < lvl):
            h_out[i] = 0.0
            continue

        idx = None
        for j in range(int(vv.size) - 1):
            if vv[j] >= lvl and vv[j + 1] < lvl:
                idx = j
                break
        if idx is None:
            # Non-monotone column; fall back to the highest crossing.
            for j in range(int(vv.size) - 1):
                if (vv[j] - lvl) * (vv[j + 1] - lvl) <= 0:
                    idx = j
            if idx is None:
                continue

        y0, y1 = float(yy[idx]), float(yy[idx + 1])
        v0, v1 = float(vv[idx]), float(vv[idx + 1])
        if abs(v1 - v0) <= 1.0e-16:
            h_out[i] = y0
        else:
            s = (lvl - v0) / (v1 - v0)
            h_out[i] = y0 + s * (y1 - y0)

    return InterfaceProfile(x=np.asarray(x_levels, dtype=float), h=h_out)


def profile_base_thickness(*, x: np.ndarray, h: np.ndarray, Lx: float, far_frac: float = 0.25) -> float:
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    if x.size == 0 or h.size != x.size:
        return float("nan")
    Lx = float(Lx)
    f = float(far_frac)
    if not (0.0 < f < 0.5):
        f = 0.25
    m = (x <= f * Lx) | (x >= (1.0 - f) * Lx)
    if not np.any(m):
        m = np.isfinite(h)
    with np.errstate(all="ignore"):
        return float(np.nanmedian(h[m]))


def profile_peak(*, x: np.ndarray, h: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    if x.size == 0 or h.size != x.size:
        return float("nan"), float("nan")
    if not np.any(np.isfinite(h)):
        return float("nan"), float("nan")
    i = int(np.nanargmax(h))
    return float(x[i]), float(h[i])


def profile_width_at_height(
    *,
    x: np.ndarray,
    h: np.ndarray,
    y_level: float,
    x_focus: float | None = None,
) -> float:
    """
    Width (span in x) of the connected region where h(x) >= y_level.

    If `x_focus` is given, returns the width of the connected component that contains
    x_focus (or, if none contains it, the largest component).
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    if x.size == 0 or h.size != x.size:
        return 0.0

    m = np.isfinite(h) & (h >= float(y_level))
    idx = np.where(m)[0]
    if idx.size == 0:
        return 0.0

    # Split into contiguous index runs.
    splits = np.where(np.diff(idx) > 1)[0]
    runs = np.split(idx, splits + 1)
    if not runs:
        return 0.0

    def _run_width(run):
        return float(x[run[-1]] - x[run[0]])

    chosen = None
    if x_focus is not None and np.isfinite(float(x_focus)):
        xf = float(x_focus)
        for run in runs:
            if float(x[run[0]]) <= xf <= float(x[run[-1]]):
                chosen = run
                break
    if chosen is None:
        # Largest run by width.
        chosen = max(runs, key=_run_width)

    return _run_width(chosen)


@dataclass(frozen=True)
class ProfileMetrics:
    H_base: float
    x_peak: float
    H_max: float
    H_hump: float
    W_half: float
    W_02: float
    W_08: float
    mushroomness: float


def compute_profile_metrics(
    *,
    profile: InterfaceProfile,
    Lx: float,
    far_frac: float = 0.25,
) -> ProfileMetrics:
    x = np.asarray(profile.x, dtype=float)
    h = np.asarray(profile.h, dtype=float)
    if x.size == 0 or h.size != x.size or not np.any(np.isfinite(h)):
        nan = float("nan")
        return ProfileMetrics(H_base=nan, x_peak=nan, H_max=nan, H_hump=nan, W_half=nan, W_02=nan, W_08=nan, mushroomness=nan)

    H_base = profile_base_thickness(x=x, h=h, Lx=float(Lx), far_frac=float(far_frac))
    x_peak, H_max = profile_peak(x=x, h=h)
    H_hump = float(H_max - H_base) if np.isfinite(H_max) and np.isfinite(H_base) else float("nan")
    if not np.isfinite(H_hump) or H_hump <= 0.0:
        return ProfileMetrics(H_base=float(H_base), x_peak=float(x_peak), H_max=float(H_max), H_hump=float(H_hump), W_half=0.0, W_02=0.0, W_08=0.0, mushroomness=float("nan"))

    y_half = float(H_base + 0.5 * H_hump)
    y_02 = float(H_base + 0.2 * H_hump)
    y_08 = float(H_base + 0.8 * H_hump)
    W_half = profile_width_at_height(x=x, h=h, y_level=y_half, x_focus=x_peak)
    W_02 = profile_width_at_height(x=x, h=h, y_level=y_02, x_focus=x_peak)
    W_08 = profile_width_at_height(x=x, h=h, y_level=y_08, x_focus=x_peak)
    mushroomness = float(W_08 / W_02) if W_02 > 0.0 else float("nan")
    return ProfileMetrics(
        H_base=float(H_base),
        x_peak=float(x_peak),
        H_max=float(H_max),
        H_hump=float(H_hump),
        W_half=float(W_half),
        W_02=float(W_02),
        W_08=float(W_08),
        mushroomness=float(mushroomness),
    )


def coord_match_indices(
    *,
    src_coords: np.ndarray,
    tgt_coords: np.ndarray,
    decimals: int = 12,
) -> np.ndarray:
    """
    Return an index map `src -> tgt` by coordinate matching (rounded keys).

    Entries that cannot be matched are set to -1.
    """
    src = np.asarray(src_coords, dtype=float)
    tgt = np.asarray(tgt_coords, dtype=float)
    if src.ndim != 2 or tgt.ndim != 2 or src.shape[1] < 2 or tgt.shape[1] < 2:
        return np.full((src.shape[0],), -1, dtype=int)

    dec = int(decimals)
    tgt_key = {
        (float(np.round(x, dec)), float(np.round(y, dec))): int(i)
        for i, (x, y) in enumerate(tgt[:, :2])
    }
    out = np.full((src.shape[0],), -1, dtype=int)
    for i, (x, y) in enumerate(src[:, :2]):
        out[i] = tgt_key.get((float(np.round(x, dec)), float(np.round(y, dec))), -1)
    return out
