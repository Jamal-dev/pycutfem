#!/usr/bin/env python3
"""
Minimal, reproducible DIC utilities for the Lie benchmark (Video S1).

This implements a classic subset-based *translation* DIC tracker:
  - rigid stabilization using a similarity transform from the SVG-traced base line,
  - template matching in a local search window (ZNCC / ZNSSD via OpenCV),
  - optional multi-scale (Gaussian pyramid),
  - optional 1D quadratic subpixel peak refinement.

This is intentionally small and dependency-light (OpenCV + NumPy) so it can be
used in scripts under `examples/biofilms/benchmarks/lie/`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import cv2
import numpy as np


def similarity_affine_from_2pts(
    *,
    src_left: np.ndarray,
    src_right: np.ndarray,
    dst_left: np.ndarray,
    dst_right: np.ndarray,
) -> np.ndarray:
    """
    Return a 2x3 similarity (scale+rotation+translation) mapping src -> dst.

    The transform is defined to map:
      src_left  -> dst_left
      src_right -> dst_right
    """
    a0 = np.asarray(src_left, dtype=float).reshape(2)
    a1 = np.asarray(src_right, dtype=float).reshape(2)
    b0 = np.asarray(dst_left, dtype=float).reshape(2)
    b1 = np.asarray(dst_right, dtype=float).reshape(2)

    va = a1 - a0
    vb = b1 - b0
    la = float(np.hypot(float(va[0]), float(va[1])))
    lb = float(np.hypot(float(vb[0]), float(vb[1])))
    if not np.isfinite(la) or la <= 1.0e-12:
        raise ValueError("Degenerate src segment for similarity transform.")
    if not np.isfinite(lb) or lb <= 1.0e-12:
        raise ValueError("Degenerate dst segment for similarity transform.")

    sa = float(math.atan2(float(va[1]), float(va[0])))
    sb = float(math.atan2(float(vb[1]), float(vb[0])))
    ang = float(sb - sa)
    scale = float(lb / la)

    ca = float(math.cos(ang))
    sa2 = float(math.sin(ang))
    A = scale * np.array([[ca, -sa2], [sa2, ca]], dtype=float)
    t = b0 - A @ a0
    M = np.array([[A[0, 0], A[0, 1], t[0]], [A[1, 0], A[1, 1], t[1]]], dtype=float)
    return np.asarray(M, dtype=float)


def warp_affine_src_to_dst(img: np.ndarray, *, M: np.ndarray, dsize: tuple[int, int], border_value: int = 0) -> np.ndarray:
    """
    Warp with a src->dst affine matrix M (2x3).

    OpenCV's warpAffine expects a matrix that maps source->destination in the
    common usage patterns (e.g. positive translation shifts content positively),
    so we keep this explicit for readability.
    """
    M = np.asarray(M, dtype=float)
    if M.shape != (2, 3):
        raise ValueError("M must have shape (2,3)")
    w, h = int(dsize[0]), int(dsize[1])
    if w <= 0 or h <= 0:
        raise ValueError("dsize must be positive")
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=int(border_value),
    )


def polygon_mask_u8(*, shape_hw: tuple[int, int], poly_xy: np.ndarray) -> np.ndarray:
    h, w = (int(shape_hw[0]), int(shape_hw[1]))
    if h <= 0 or w <= 0:
        raise ValueError("shape_hw must be positive")
    pts = np.asarray(poly_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise ValueError("poly_xy must have shape (N,2) with N>=3")
    pts_i = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_i], 255)
    return mask


def erode_mask_u8(mask_u8: np.ndarray, *, radius_px: int) -> np.ndarray:
    m = np.asarray(mask_u8, dtype=np.uint8)
    r = int(max(0, int(radius_px)))
    if r <= 0:
        return m.copy()
    k = 2 * r + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(m, ker, iterations=1)


def build_pyramid_u8(gray_u8: np.ndarray, *, levels: int) -> list[np.ndarray]:
    levels = int(max(1, int(levels)))
    g = np.asarray(gray_u8, dtype=np.uint8)
    pyr = [g]
    for _ in range(1, levels):
        g = cv2.pyrDown(g)
        pyr.append(g)
    return pyr


def _ensure_odd(n: int) -> int:
    n = int(n)
    if n <= 0:
        return 1
    return n if (n % 2) == 1 else (n + 1)


def _subpixel_delta_1d(fm: float, f0: float, fp: float) -> float:
    """
    Quadratic peak refinement for samples at x=-1,0,+1.

    Returns delta in [-0.5, 0.5] (typically), with 0 meaning the integer peak.
    """
    denom = float(fm - 2.0 * f0 + fp)
    if not np.isfinite(denom) or abs(denom) <= 1.0e-30:
        return 0.0
    d = 0.5 * float(fm - fp) / denom
    if not np.isfinite(d):
        return 0.0
    return float(np.clip(d, -0.75, 0.75))


@dataclass(frozen=True)
class DicSettings:
    subset_px: int = 41
    search_radius_px: int = 20
    pyramid_levels: int = 3
    method: Literal["zncc", "znssd"] = "zncc"
    subpixel: bool = True
    min_score: float = 0.2  # only used for zncc ([-1,1])


def track_points_translation_dic(
    *,
    ref_gray_u8: np.ndarray,
    cur_gray_u8: np.ndarray,
    pts_ref_xy: np.ndarray,
    disp0_xy: np.ndarray | None,
    settings: DicSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Track points from ref -> cur using subset translation DIC.

    Returns:
      disp_xy:  (N,2) displacement in pixels
      score:    (N,)  per-point quality score (higher is better)
    """
    ref0 = np.asarray(ref_gray_u8, dtype=np.uint8)
    cur0 = np.asarray(cur_gray_u8, dtype=np.uint8)
    if ref0.shape != cur0.shape:
        raise ValueError("ref_gray_u8 and cur_gray_u8 must have the same shape")

    pts = np.asarray(pts_ref_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts_ref_xy must have shape (N,2)")

    n = int(pts.shape[0])
    if disp0_xy is None:
        disp_prev = np.zeros((n, 2), dtype=float)
    else:
        disp_prev = np.asarray(disp0_xy, dtype=float)
        if disp_prev.shape != (n, 2):
            raise ValueError("disp0_xy must have shape (N,2)")

    subset = _ensure_odd(int(settings.subset_px))
    rad = int(max(0, int(settings.search_radius_px)))
    levels = int(max(1, int(settings.pyramid_levels)))
    method = str(settings.method).strip().lower()
    if method not in {"zncc", "znssd"}:
        raise ValueError("settings.method must be 'zncc' or 'znssd'")
    cv_method = cv2.TM_CCOEFF_NORMED if method == "zncc" else cv2.TM_SQDIFF_NORMED
    maximize = bool(method == "zncc")

    ref_pyr = build_pyramid_u8(ref0, levels=levels)
    cur_pyr = build_pyramid_u8(cur0, levels=levels)

    disp_out = np.full((n, 2), float("nan"), dtype=float)
    score_out = np.full((n,), float("nan"), dtype=float)

    # Start from coarsest.
    scale_coarse = float(2 ** (levels - 1))
    disp_lvl = disp_prev / scale_coarse

    for i in range(n):
        if not np.all(np.isfinite(pts[i, :])):
            continue
        if not np.all(np.isfinite(disp_lvl[i, :])):
            continue
        off = disp_lvl[i, :].copy()

        ok = True
        peak_score = float("nan")
        for lvl in range(levels - 1, -1, -1):
            ref_img = ref_pyr[lvl]
            cur_img = cur_pyr[lvl]
            h, w = ref_img.shape[:2]
            s = float(2**lvl)
            pt = pts[i, :] / s

            subset_l = _ensure_odd(int(round(float(subset) / s)))
            subset_l = int(max(9, subset_l))
            half = subset_l // 2
            rad_l = int(max(2, int(round(float(rad) / s))))

            cx = int(round(float(pt[0])))
            cy = int(round(float(pt[1])))
            if cx - half < 0 or cx + half >= w or cy - half < 0 or cy + half >= h:
                ok = False
                break
            templ = ref_img[cy - half : cy + half + 1, cx - half : cx + half + 1]

            px = float(pt[0] + off[0])
            py = float(pt[1] + off[1])
            sx = int(round(px))
            sy = int(round(py))
            x0 = sx - half - rad_l
            y0 = sy - half - rad_l
            x1 = x0 + subset_l + 2 * rad_l
            y1 = y0 + subset_l + 2 * rad_l
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                ok = False
                break
            search = cur_img[y0:y1, x0:x1]

            res = cv2.matchTemplate(search, templ, cv_method)
            if res.size == 0:
                ok = False
                break

            if maximize:
                _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
                ix, iy = int(max_loc[0]), int(max_loc[1])
                peak = float(max_val)
                score_surf = res
            else:
                min_val, _max_val, min_loc, _max_loc = cv2.minMaxLoc(res)
                ix, iy = int(min_loc[0]), int(min_loc[1])
                peak = float(-min_val)  # higher is better
                score_surf = -res

            if maximize and np.isfinite(float(settings.min_score)) and peak < float(settings.min_score):
                ok = False
                break

            dx = float(ix - rad_l)
            dy = float(iy - rad_l)
            if bool(settings.subpixel) and 0 < ix < (res.shape[1] - 1) and 0 < iy < (res.shape[0] - 1):
                fxm = float(score_surf[iy, ix - 1])
                fx0 = float(score_surf[iy, ix])
                fxp = float(score_surf[iy, ix + 1])
                fym = float(score_surf[iy - 1, ix])
                fy0 = float(score_surf[iy, ix])
                fyp = float(score_surf[iy + 1, ix])
                dx += _subpixel_delta_1d(fxm, fx0, fxp)
                dy += _subpixel_delta_1d(fym, fy0, fyp)

            off = off + np.array([dx, dy], dtype=float)
            peak_score = float(peak)
            if lvl > 0:
                off *= 2.0

        if not ok:
            continue
        disp_out[i, :] = off
        score_out[i] = float(peak_score)

    return disp_out, score_out

