from __future__ import annotations

import numpy as np

from examples.biofilms.benchmarks.lie.dic_utils import (
    DicSettings,
    similarity_affine_from_2pts,
    track_points_translation_dic,
    warp_affine_src_to_dst,
)


def _shift_integer(img: np.ndarray, *, dx: int, dy: int) -> np.ndarray:
    """
    Integer shift without wrap-around.

    dx > 0 moves content to the right.
    dy > 0 moves content down.
    """
    a = np.asarray(img)
    h, w = a.shape[:2]
    out = np.zeros_like(a)

    x_src0 = max(0, -int(dx))
    x_src1 = min(w, w - int(dx))
    x_dst0 = max(0, int(dx))
    x_dst1 = min(w, w + int(dx))

    y_src0 = max(0, -int(dy))
    y_src1 = min(h, h - int(dy))
    y_dst0 = max(0, int(dy))
    y_dst1 = min(h, h + int(dy))

    if x_src1 > x_src0 and y_src1 > y_src0:
        out[y_dst0:y_dst1, x_dst0:x_dst1] = a[y_src0:y_src1, x_src0:x_src1]
    return out


def test_warp_affine_src_to_dst_translation_direction() -> None:
    img = np.zeros((64, 64), dtype=np.uint8)
    img[20:24, 30:34] = 255

    dx, dy = 7.0, 5.0
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=float)
    out = warp_affine_src_to_dst(img, M=M, dsize=(64, 64), border_value=0)

    # Expect the bright block to move by (+dx,+dy).
    assert int(np.argmax(out) % 64) in range(30 + int(dx), 34 + int(dx))
    assert int(np.argmax(out) // 64) in range(20 + int(dy), 24 + int(dy))


def test_similarity_affine_from_2pts_maps_endpoints() -> None:
    src_left = np.array([10.0, 20.0])
    src_right = np.array([50.0, 20.0])
    dst_left = np.array([12.0, 18.0])
    dst_right = np.array([92.0, 18.0])
    M = similarity_affine_from_2pts(src_left=src_left, src_right=src_right, dst_left=dst_left, dst_right=dst_right)
    a0 = np.array([src_left[0], src_left[1], 1.0])
    a1 = np.array([src_right[0], src_right[1], 1.0])
    b0 = M @ a0
    b1 = M @ a1
    assert np.allclose(b0, dst_left, atol=1.0e-9)
    assert np.allclose(b1, dst_right, atol=1.0e-9)


def test_translation_dic_recovers_integer_shift() -> None:
    rng = np.random.default_rng(0)
    ref = (rng.random((128, 128)) * 255.0).astype(np.uint8)
    # Add a little smoothing-like structure (DIC likes texture).
    ref = ((ref.astype(np.float32) * 0.6) + (_shift_integer(ref, dx=1, dy=0).astype(np.float32) * 0.4)).astype(np.uint8)

    dx, dy = 9, -6
    cur = _shift_integer(ref, dx=dx, dy=dy)

    pts = np.array([[64.0, 64.0], [80.0, 70.0], [50.0, 90.0]], dtype=float)
    settings = DicSettings(subset_px=31, search_radius_px=20, pyramid_levels=2, method="zncc", subpixel=True, min_score=-1.0)
    disp, score = track_points_translation_dic(ref_gray_u8=ref, cur_gray_u8=cur, pts_ref_xy=pts, disp0_xy=None, settings=settings)
    assert np.all(np.isfinite(score))
    assert np.allclose(disp[:, 0], float(dx), atol=0.25)
    assert np.allclose(disp[:, 1], float(dy), atol=0.25)

