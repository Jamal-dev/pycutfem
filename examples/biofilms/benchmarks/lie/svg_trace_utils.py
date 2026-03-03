#!/usr/bin/env python3
"""
Utilities for working with manually traced Inkscape SVG contours (Lie benchmark).

This module extracts:
  - the rigid-support top "base" line (used for anchor + pixel->mm scaling),
  - the biofilm outline curve(s),
from per-frame SVGs stored in `examples/biofilms/benchmarks/lie/svg_fles/`.

The SVGs are created by importing a video frame image into Inkscape and drawing
the base line + outline using the pen/bezier tool. We avoid external SVG
dependencies by implementing a minimal SVG path parser for the command set
observed in the traces (M/m, L/l, H/h, V/v, C/c, S/s, Z/z).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import itertools
import json
import math
import re
import xml.etree.ElementTree as ET

import numpy as np


_CMD_RE = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]")
_NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _tokenize_path_d(d: str) -> list[str]:
    # Split into commands and numbers, ignoring separators.
    d = str(d).strip()
    if not d:
        return []
    # Regex alternation preserves order.
    tok_re = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
    return tok_re.findall(d)


def _sample_cubic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, *, n: int) -> np.ndarray:
    n = int(max(1, int(n)))
    t = np.linspace(0.0, 1.0, num=n + 1, endpoint=True, dtype=float)
    omt = 1.0 - t
    # (1-t)^3 p0 + 3 (1-t)^2 t p1 + 3 (1-t) t^2 p2 + t^3 p3
    out = (
        (omt**3)[:, None] * p0[None, :]
        + (3.0 * (omt**2) * t)[:, None] * p1[None, :]
        + (3.0 * omt * (t**2))[:, None] * p2[None, :]
        + (t**3)[:, None] * p3[None, :]
    )
    return np.asarray(out, dtype=float)


def _dedup_consecutive(points: np.ndarray, *, tol: float = 0.0) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] <= 1:
        return pts
    tol = float(max(0.0, float(tol)))
    if tol <= 0.0:
        keep = np.ones((pts.shape[0],), dtype=bool)
        keep[1:] = np.any(pts[1:] != pts[:-1], axis=1)
        return pts[keep]
    d = np.sqrt(np.sum((pts[1:] - pts[:-1]) ** 2, axis=1))
    keep = np.ones((pts.shape[0],), dtype=bool)
    keep[1:] = d > tol
    return pts[keep]


def sample_svg_path_d(d: str, *, cubic_samples: int = 20) -> list[np.ndarray]:
    """
    Sample an SVG path `d` string into polylines.

    Returns a list of subpaths, each an array of shape (N,2) in SVG pixel coords.
    """
    toks = _tokenize_path_d(d)
    if not toks:
        return []

    out: list[list[list[float]]] = []
    cur: list[list[float]] = []

    cx = cy = 0.0
    sx = sy = 0.0
    prev_cmd: str | None = None
    prev_c2: tuple[float, float] | None = None  # last cubic control point (absolute)

    i = 0
    cmd: str | None = None

    def _start_new_subpath(x: float, y: float) -> None:
        nonlocal cx, cy, sx, sy, cur
        if cur:
            out.append(cur)
        cur = [[float(x), float(y)]]
        cx, cy = float(x), float(y)
        sx, sy = float(x), float(y)

    def _line_to(x: float, y: float) -> None:
        nonlocal cx, cy, cur
        cur.append([float(x), float(y)])
        cx, cy = float(x), float(y)

    while i < len(toks):
        t = toks[i]
        if _CMD_RE.fullmatch(t):
            cmd = t
            i += 1
            if cmd in {"Z", "z"}:
                if cur:
                    # Close to the subpath start.
                    cur.append([float(sx), float(sy)])
                    cx, cy = float(sx), float(sy)
                prev_cmd = cmd
                prev_c2 = None
                cmd = None
            continue

        if cmd is None:
            raise ValueError("Invalid SVG path: number without active command.")

        rel = cmd.islower()
        up = cmd.upper()

        if up == "M":
            if i + 1 >= len(toks):
                break
            x = float(toks[i])
            y = float(toks[i + 1])
            i += 2
            if rel:
                x += cx
                y += cy
            _start_new_subpath(x, y)
            # Subsequent pairs are treated as implicit lineto.
            cmd = "l" if rel else "L"
            prev_cmd = "M"
            prev_c2 = None
            continue

        if up == "L":
            if i + 1 >= len(toks):
                break
            x = float(toks[i])
            y = float(toks[i + 1])
            i += 2
            if rel:
                x += cx
                y += cy
            _line_to(x, y)
            prev_cmd = cmd
            prev_c2 = None
            continue

        if up == "H":
            x = float(toks[i])
            i += 1
            if rel:
                x += cx
            _line_to(x, cy)
            prev_cmd = cmd
            prev_c2 = None
            continue

        if up == "V":
            y = float(toks[i])
            i += 1
            if rel:
                y += cy
            _line_to(cx, y)
            prev_cmd = cmd
            prev_c2 = None
            continue

        if up == "C":
            if i + 5 >= len(toks):
                break
            x1 = float(toks[i])
            y1 = float(toks[i + 1])
            x2 = float(toks[i + 2])
            y2 = float(toks[i + 3])
            x3 = float(toks[i + 4])
            y3 = float(toks[i + 5])
            i += 6
            if rel:
                x1 += cx
                y1 += cy
                x2 += cx
                y2 += cy
                x3 += cx
                y3 += cy
            p0 = np.array([cx, cy], dtype=float)
            p1 = np.array([x1, y1], dtype=float)
            p2 = np.array([x2, y2], dtype=float)
            p3 = np.array([x3, y3], dtype=float)
            seg = _sample_cubic_bezier(p0, p1, p2, p3, n=int(cubic_samples))
            # Append without repeating the start point.
            for q in seg[1:]:
                cur.append([float(q[0]), float(q[1])])
            cx, cy = float(x3), float(y3)
            prev_cmd = cmd
            prev_c2 = (float(x2), float(y2))
            continue

        if up == "S":
            if i + 3 >= len(toks):
                break
            x2 = float(toks[i])
            y2 = float(toks[i + 1])
            x3 = float(toks[i + 2])
            y3 = float(toks[i + 3])
            i += 4
            if rel:
                x2 += cx
                y2 += cy
                x3 += cx
                y3 += cy
            if prev_cmd is not None and prev_cmd.upper() in {"C", "S"} and prev_c2 is not None:
                x1 = 2.0 * cx - float(prev_c2[0])
                y1 = 2.0 * cy - float(prev_c2[1])
            else:
                x1 = cx
                y1 = cy
            p0 = np.array([cx, cy], dtype=float)
            p1 = np.array([x1, y1], dtype=float)
            p2 = np.array([x2, y2], dtype=float)
            p3 = np.array([x3, y3], dtype=float)
            seg = _sample_cubic_bezier(p0, p1, p2, p3, n=int(cubic_samples))
            for q in seg[1:]:
                cur.append([float(q[0]), float(q[1])])
            cx, cy = float(x3), float(y3)
            prev_cmd = cmd
            prev_c2 = (float(x2), float(y2))
            continue

        # Unsupported command: consume numbers until the next command token.
        # We keep this conservative (skip), so the caller can decide to fail if needed.
        prev_cmd = cmd
        prev_c2 = None
        i += 1

    if cur:
        out.append(cur)
    return [_dedup_consecutive(np.asarray(p, dtype=float), tol=0.0) for p in out if len(p) >= 2]


@dataclass(frozen=True)
class SvgFrameGeometry:
    base_left_px: np.ndarray  # shape (2,)
    base_right_px: np.ndarray  # shape (2,)
    angle_rad: float  # rotation angle of base (right-left vector) in SVG coords
    mm_per_px: float  # pixel -> mm
    boundary_px: np.ndarray  # shape (N,2), starts at base_left, ends at base_right
    polygon_mm: np.ndarray  # shape (M,2), closed, anchored at base_left, y up

    def px_to_mm(self, pts_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_px, dtype=float)
        a = float(self.angle_rad)
        ca, sa = math.cos(-a), math.sin(-a)
        dx = pts[:, 0] - float(self.base_left_px[0])
        dy = pts[:, 1] - float(self.base_left_px[1])
        xr = ca * dx - sa * dy
        yr = sa * dx + ca * dy
        x_mm = xr * float(self.mm_per_px)
        y_mm = (-yr) * float(self.mm_per_px)
        return np.column_stack([x_mm, y_mm])

    def mm_to_px(self, pts_mm: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_mm, dtype=float)
        a = float(self.angle_rad)
        ca, sa = math.cos(a), math.sin(a)
        xr = pts[:, 0] / float(self.mm_per_px)
        yr = -pts[:, 1] / float(self.mm_per_px)
        x = ca * xr - sa * yr + float(self.base_left_px[0])
        y = sa * xr + ca * yr + float(self.base_left_px[1])
        return np.column_stack([x, y])


def _path_polylines_from_svg(svg_path: Path, *, cubic_samples: int) -> list[np.ndarray]:
    root = ET.parse(str(svg_path)).getroot()
    polylines: list[np.ndarray] = []
    for el in root.iter():
        if el.tag.split("}")[-1] != "path":
            continue
        d = el.get("d")
        if not d:
            continue
        sub = sample_svg_path_d(d, cubic_samples=int(cubic_samples))
        for s in sub:
            if s.shape[0] >= 2:
                polylines.append(np.asarray(s, dtype=float))
    return polylines


def _polyline_bbox(poly: np.ndarray) -> tuple[float, float, float, float]:
    p = np.asarray(poly, dtype=float)
    return float(np.min(p[:, 0])), float(np.max(p[:, 0])), float(np.min(p[:, 1])), float(np.max(p[:, 1]))


def _choose_base_polyline(polylines: list[np.ndarray]) -> int:
    if not polylines:
        raise ValueError("No polylines to choose from.")
    best = None
    for i, p in enumerate(polylines):
        xmin, xmax, ymin, ymax = _polyline_bbox(p)
        dx = float(xmax - xmin)
        dy = float(ymax - ymin)
        if dx <= 0.0:
            continue
        score = dx / max(1.0e-6, dy)
        mean_y = float(np.mean(p[:, 1]))
        key = (score, dx, mean_y)
        if best is None or key > best[0]:
            best = (key, i)
    if best is None:
        raise RuntimeError("Failed selecting base polyline.")
    return int(best[1])


def _endpoints(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(poly, dtype=float)
    if p.shape[0] < 2:
        raise ValueError("poly must have >=2 points")
    return np.asarray(p[0], dtype=float), np.asarray(p[-1], dtype=float)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def _chain_boundary_segments(
    segments: list[np.ndarray],
    *,
    base_left: np.ndarray,
    base_right: np.ndarray,
    join_tol_px: float,
) -> np.ndarray:
    if not segments:
        raise ValueError("No boundary segments.")
    if len(segments) == 1:
        b = np.asarray(segments[0], dtype=float)
    else:
        # Brute force all permutations/orientations. The SVGs may contain extra small
        # paths (e.g., reference marks) that should be ignored, so we allow dropping
        # disconnected segments by searching over subsets, preferring larger subsets.
        best_poly = None
        best_key: tuple[float, float] | None = None
        seg_ids = list(range(len(segments)))

        def _poly_len(p: np.ndarray) -> float:
            p = np.asarray(p, dtype=float)
            if p.shape[0] < 2:
                return 0.0
            d = np.diff(p, axis=0)
            return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))

        for k in range(len(segments), 0, -1):
            found_any = False
            for subset in itertools.combinations(seg_ids, k):
                for perm in itertools.permutations(subset):
                    for flips in itertools.product([False, True], repeat=k):
                        polys = []
                        for idx, flip in zip(perm, flips):
                            p = np.asarray(segments[int(idx)], dtype=float)
                            polys.append(p[::-1].copy() if bool(flip) else p)

                        # Check join distances.
                        join_d = 0.0
                        ok = True
                        for a, b2 in zip(polys[:-1], polys[1:]):
                            d = _dist(a[-1], b2[0])
                            join_d += d
                            if d > float(join_tol_px):
                                ok = False
                                break
                        if not ok:
                            continue

                        # Prefer matching base endpoints too.
                        start = polys[0][0]
                        end = polys[-1][-1]
                        d_lr = _dist(start, base_left) + _dist(end, base_right)
                        d_rl = _dist(start, base_right) + _dist(end, base_left)
                        base_d = min(d_lr, d_rl)

                        # Tie-break using total chained length (prefer longer).
                        total_len = float(sum(_poly_len(p) for p in polys))
                        score = float(join_d + 5.0 * float(base_d))
                        key = (score, -total_len)
                        if best_key is None or key < best_key:
                            # Concatenate, removing duplicate join points.
                            cat = [polys[0]]
                            for nxt in polys[1:]:
                                cat.append(nxt[1:] if _dist(cat[-1][-1], nxt[0]) <= float(join_tol_px) else nxt)
                            best_poly = np.vstack(cat)
                            best_key = key
                        found_any = True

            if found_any:
                break

        if best_poly is None:
            raise RuntimeError("Failed chaining boundary segments (no consistent endpoint matching).")
        b = np.asarray(best_poly, dtype=float)

    # Ensure orientation base_left -> base_right (if possible).
    start = np.asarray(b[0], dtype=float)
    end = np.asarray(b[-1], dtype=float)
    d_lr = _dist(start, base_left) + _dist(end, base_right)
    d_rl = _dist(start, base_right) + _dist(end, base_left)
    if d_rl < d_lr:
        b = b[::-1].copy()
        start = np.asarray(b[0], dtype=float)
        end = np.asarray(b[-1], dtype=float)

    # Snap endpoints if close enough (avoid cumulative drift from sampling).
    if _dist(start, base_left) <= float(join_tol_px):
        b[0, :] = np.asarray(base_left, dtype=float)
    else:
        b = np.vstack([np.asarray(base_left, dtype=float).reshape(1, 2), b])
    if _dist(end, base_right) <= float(join_tol_px):
        b[-1, :] = np.asarray(base_right, dtype=float)
    else:
        b = np.vstack([b, np.asarray(base_right, dtype=float).reshape(1, 2)])

    return _dedup_consecutive(np.asarray(b, dtype=float), tol=0.0)


def extract_svg_frame_geometry(
    svg_path: str | Path,
    *,
    block_w_mm: float = 1.0,
    m_per_px: float | None = None,
    cubic_samples: int = 20,
    join_tol_px: float = 5.0,
) -> SvgFrameGeometry:
    """
    Extract base+boundary from a single SVG file and return a frame geometry object.
    """
    svg_path = Path(str(svg_path))
    polylines = _path_polylines_from_svg(svg_path, cubic_samples=int(cubic_samples))
    if len(polylines) < 1:
        raise RuntimeError(f"No sampled paths in {svg_path}")

    all_pts = np.vstack(polylines)
    y_max_all = float(np.max(all_pts[:, 1]))

    def _line_like(p: np.ndarray, *, dx_min: float = 20.0, dy_max: float = 6.0) -> bool:
        xmin, xmax, ymin, ymax = _polyline_bbox(p)
        dx = float(xmax - xmin)
        dy = float(ymax - ymin)
        return bool(dx >= float(dx_min) and dy <= float(dy_max))

    # Base endpoints: collect all near-bottom line-like segments (base can be split across paths).
    band = float(max(1.0, 0.5 * float(join_tol_px)))
    base_idxs: list[int] = []
    for i, p in enumerate(polylines):
        if not _line_like(p):
            continue
        mean_y = float(np.mean(p[:, 1]))
        if mean_y >= (y_max_all - band):
            base_idxs.append(int(i))

    base_is_separate = bool(base_idxs and len(polylines) > len(base_idxs))
    if base_idxs:
        base_pts = np.vstack([polylines[i] for i in base_idxs])
        j_left = int(np.argmin(base_pts[:, 0]))
        j_right = int(np.argmax(base_pts[:, 0]))
        base_left = np.asarray(base_pts[j_left], dtype=float)
        base_right = np.asarray(base_pts[j_right], dtype=float)
    else:
        # Fall back to inferring from the lowest y band over all sampled points.
        sel = all_pts[:, 1] >= (y_max_all - band)
        if int(np.sum(sel)) < 2:
            sel = all_pts[:, 1] >= (y_max_all - float(max(2.0, float(join_tol_px))))
        pts = all_pts[sel] if int(np.sum(sel)) >= 2 else all_pts
        j_left = int(np.argmin(pts[:, 0]))
        j_right = int(np.argmax(pts[:, 0]))
        base_left = np.asarray(pts[j_left], dtype=float)
        base_right = np.asarray(pts[j_right], dtype=float)

    if float(base_left[0]) > float(base_right[0]):
        base_left, base_right = base_right, base_left

    dx = float(base_right[0] - base_left[0])
    dy = float(base_right[1] - base_left[1])
    base_len = float(math.hypot(dx, dy))
    if not np.isfinite(base_len) or base_len <= 1.0e-6:
        raise RuntimeError(f"Degenerate base length in {svg_path} (len={base_len})")
    angle = float(math.atan2(dy, dx))
    if m_per_px is not None and np.isfinite(float(m_per_px)) and float(m_per_px) > 0.0:
        mm_per_px = float(m_per_px) * 1.0e3
    else:
        mm_per_px = float(block_w_mm) / float(base_len)

    # Boundary segments: everything except the dedicated base segment(s) (if present).
    segs = [p for k, p in enumerate(polylines) if k not in set(base_idxs)] if base_is_separate else list(polylines)

    if not segs:
        raise RuntimeError(f"No boundary segments found in {svg_path}")

    if len(segs) == 1 and (not base_is_separate) and segs[0].shape[0] >= 4:
        # Some frames were saved with a *single* path that includes the base segment.
        # Convert it to an open boundary polyline by cutting between the two base endpoints
        # and selecting the *longer* arc (the actual outline).
        p = _dedup_consecutive(np.asarray(segs[0], dtype=float), tol=0.0)
        if _dist(p[0], p[-1]) <= float(join_tol_px):
            p = p[:-1].copy()
        i_left = int(np.argmin(np.sum((p - base_left[None, :]) ** 2, axis=1)))
        i_right = int(np.argmin(np.sum((p - base_right[None, :]) ** 2, axis=1)))
        if i_left == i_right:
            # Fallback: treat as-is.
            boundary_px = _chain_boundary_segments([p], base_left=base_left, base_right=base_right, join_tol_px=float(join_tol_px))
        else:
            if i_left < i_right:
                arc1 = p[i_left : i_right + 1]
                arc2 = np.vstack([p[i_right:], p[: i_left + 1]])
            else:
                arc1 = np.vstack([p[i_left:], p[: i_right + 1]])
                arc2 = p[i_right : i_left + 1]

            def _arc_len(a: np.ndarray) -> float:
                d = np.diff(a, axis=0)
                return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))

            b0 = arc1 if _arc_len(arc1) >= _arc_len(arc2) else arc2
            boundary_px = _chain_boundary_segments([b0], base_left=base_left, base_right=base_right, join_tol_px=float(join_tol_px))
    else:
        boundary_px = _chain_boundary_segments(segs, base_left=base_left, base_right=base_right, join_tol_px=float(join_tol_px))

    # Convert boundary to mm and close with the base segment.
    geom_tmp = SvgFrameGeometry(
        base_left_px=base_left,
        base_right_px=base_right,
        angle_rad=angle,
        mm_per_px=mm_per_px,
        boundary_px=boundary_px,
        polygon_mm=np.zeros((0, 2), dtype=float),
    )
    boundary_mm = geom_tmp.px_to_mm(boundary_px)
    if boundary_mm.shape[0] < 3:
        raise RuntimeError(f"Boundary has too few points in {svg_path}")
    poly_mm = boundary_mm
    # Ensure closed.
    if not np.allclose(poly_mm[0], poly_mm[-1], rtol=0.0, atol=1.0e-9):
        poly_mm = np.vstack([poly_mm, poly_mm[0]])

    return SvgFrameGeometry(
        base_left_px=base_left,
        base_right_px=base_right,
        angle_rad=angle,
        mm_per_px=mm_per_px,
        boundary_px=boundary_px,
        polygon_mm=poly_mm,
    )


def extract_svg_mark_points_px(
    svg_path: str | Path,
    *,
    cubic_samples: int = 20,
    join_tol_px: float = 5.0,
) -> np.ndarray:
    """
    Extract candidate *mark* points from an SVG frame (pixel coordinates).

    Marks are interpreted as any sampled path segments that are neither:
      - the rigid-support base line, nor
      - the main biofilm boundary outline.

    This is intended for frame-0 annotations (e.g. 3 user-defined points) that can be
    used to seed DIC tracking in the video.
    """
    svg_path = Path(str(svg_path))
    polylines = _path_polylines_from_svg(svg_path, cubic_samples=int(cubic_samples))
    if not polylines:
        return np.zeros((0, 2), dtype=float)

    all_pts = np.vstack(polylines)
    y_max_all = float(np.max(all_pts[:, 1]))

    def _line_like(p: np.ndarray, *, dx_min: float = 20.0, dy_max: float = 6.0) -> bool:
        xmin, xmax, ymin, ymax = _polyline_bbox(p)
        dx = float(xmax - xmin)
        dy = float(ymax - ymin)
        return bool(dx >= float(dx_min) and dy <= float(dy_max))

    # Identify base-like segments near the bottom (same heuristic as extract_svg_frame_geometry).
    band = float(max(1.0, 0.5 * float(join_tol_px)))
    base_idxs: list[int] = []
    for i, p in enumerate(polylines):
        if not _line_like(p):
            continue
        mean_y = float(np.mean(p[:, 1]))
        if mean_y >= (y_max_all - band):
            base_idxs.append(int(i))

    base_is_separate = bool(base_idxs and len(polylines) > len(base_idxs))
    if base_idxs:
        base_pts = np.vstack([polylines[i] for i in base_idxs])
        j_left = int(np.argmin(base_pts[:, 0]))
        j_right = int(np.argmax(base_pts[:, 0]))
        base_left = np.asarray(base_pts[j_left], dtype=float)
        base_right = np.asarray(base_pts[j_right], dtype=float)
    else:
        sel = all_pts[:, 1] >= (y_max_all - band)
        if int(np.sum(sel)) < 2:
            sel = all_pts[:, 1] >= (y_max_all - float(max(2.0, float(join_tol_px))))
        pts = all_pts[sel] if int(np.sum(sel)) >= 2 else all_pts
        j_left = int(np.argmin(pts[:, 0]))
        j_right = int(np.argmax(pts[:, 0]))
        base_left = np.asarray(pts[j_left], dtype=float)
        base_right = np.asarray(pts[j_right], dtype=float)

    if float(base_left[0]) > float(base_right[0]):
        base_left, base_right = base_right, base_left

    segs = [p for k, p in enumerate(polylines) if k not in set(base_idxs)] if base_is_separate else list(polylines)
    if not segs:
        return np.zeros((0, 2), dtype=float)

    def _poly_len(p: np.ndarray) -> float:
        p = np.asarray(p, dtype=float)
        if p.shape[0] < 2:
            return 0.0
        d = np.diff(p, axis=0)
        return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))

    lens = np.asarray([_poly_len(p) for p in segs], dtype=float)
    if not np.any(np.isfinite(lens)) or float(np.max(lens)) <= 0.0:
        return np.zeros((0, 2), dtype=float)
    j_boundary = int(np.nanargmax(lens))
    mark_segs = [p for j, p in enumerate(segs) if j != j_boundary]
    if not mark_segs:
        return np.zeros((0, 2), dtype=float)

    pts = np.vstack([np.asarray(p, dtype=float) for p in mark_segs])
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)

    # Deduplicate points and remove the base endpoints (anchor points).
    pts_r = np.round(pts, decimals=3)
    _, idx = np.unique(pts_r, axis=0, return_index=True)
    pts_u = pts[np.sort(idx)]

    def _far_from_base(pt: np.ndarray) -> bool:
        return (float(_dist(pt, base_left)) > float(join_tol_px)) and (float(_dist(pt, base_right)) > float(join_tol_px))

    keep = np.array([_far_from_base(p) for p in pts_u], dtype=bool)
    pts_u = pts_u[keep] if np.any(keep) else np.zeros((0, 2), dtype=float)
    return np.asarray(pts_u, dtype=float)


def write_svg_frame_transform_json(geom: SvgFrameGeometry, *, out_json: str | Path) -> None:
    out_json = Path(str(out_json))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_left_px": {"x": float(geom.base_left_px[0]), "y": float(geom.base_left_px[1])},
        "base_right_px": {"x": float(geom.base_right_px[0]), "y": float(geom.base_right_px[1])},
        "angle_rad": float(geom.angle_rad),
        "mm_per_px": float(geom.mm_per_px),
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
