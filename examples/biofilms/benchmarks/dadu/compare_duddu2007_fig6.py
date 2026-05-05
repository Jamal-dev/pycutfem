"""
Create side-by-side visual comparisons for Duddu et al. (2007) Fig. 6 (Example 2).

This script is intentionally lightweight and keeps all outputs inside the Duddu
benchmark folder tree, as requested.

By default it produces a 2-column comparison (paper | our). If additional result
directories are provided via --extra-dir, it produces a multi-column montage
(paper | our | extra1 | ...).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _resize_to_height(img: Image.Image, h: int) -> Image.Image:
    w0, h0 = img.size
    if h0 == h:
        return img
    w = max(1, int(round(w0 * (h / float(h0)))))
    return img.resize((w, h), resample=Image.Resampling.LANCZOS)


def _with_label(img: Image.Image, label: str, *, height: int = 28, bg=(255, 255, 255), fg=(0, 0, 0)) -> Image.Image:
    w0, h0 = img.size
    out = Image.new("RGB", (w0, h0 + int(height)), color=bg)
    out.paste(img, (0, int(height)))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), str(label), font=font)
        tw = int(bbox[2] - bbox[0])
        th = int(bbox[3] - bbox[1])
    except Exception:
        tw, th = draw.textsize(str(label), font=font)
    x = max(0, int((w0 - tw) // 2))
    y = max(0, int((int(height) - th) // 2))
    draw.text((x, y), str(label), fill=fg, font=font)
    return out


def _hcat(a: Image.Image, b: Image.Image, *, pad: int = 8, bg=(255, 255, 255)) -> Image.Image:
    ha = max(a.size[1], b.size[1])
    wa = a.size[0]
    wb = b.size[0]
    out = Image.new("RGB", (wa + pad + wb, ha), color=bg)
    out.paste(a, (0, 0))
    out.paste(b, (wa + pad, 0))
    return out


def _hcat_many(cols: list[Image.Image], *, pad: int = 8, bg=(255, 255, 255)) -> Image.Image:
    if not cols:
        raise ValueError("Need at least one column image.")
    h = max(im.size[1] for im in cols)
    w = sum(im.size[0] for im in cols) + pad * (len(cols) - 1)
    out = Image.new("RGB", (w, h), color=bg)
    x = 0
    for i, im in enumerate(cols):
        out.paste(im, (x, 0))
        x += im.size[0] + (pad if i + 1 < len(cols) else 0)
    return out


def _vcat(rows: list[Image.Image], *, pad: int = 10, bg=(255, 255, 255)) -> Image.Image:
    w = max(r.size[0] for r in rows)
    h = sum(r.size[1] for r in rows) + pad * (len(rows) - 1)
    out = Image.new("RGB", (w, h), color=bg)
    y = 0
    for i, r in enumerate(rows):
        out.paste(r, (0, y))
        y += r.size[1] + (pad if i + 1 < len(rows) else 0)
    return out


def _bbox_from_roi(gray: np.ndarray, *, roi: tuple[int, int, int, int], thr: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(v) for v in roi)
    sub = np.asarray(gray[y0:y1, x0:x1], dtype=np.uint8)
    ys, xs = np.where(sub < int(thr))
    if ys.size == 0:
        raise RuntimeError(f"No foreground pixels found in ROI={roi} with thr={thr}.")
    return (int(x0 + xs.min()), int(y0 + ys.min()), int(x0 + xs.max() + 1), int(y0 + ys.max() + 1))


def _expand_bbox(b: tuple[int, int, int, int], *, pad_x: int, pad_y: int, W: int, H: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(v) for v in b)
    x0 = max(0, x0 - int(pad_x))
    y0 = max(0, y0 - int(pad_y))
    x1 = min(int(W), x1 + int(pad_x))
    y1 = min(int(H), y1 + int(pad_y))
    return (int(x0), int(y0), int(x1), int(y1))


def _auto_crop_paper_fig6_panels(page_png: Path) -> dict[str, Image.Image]:
    """
    Heuristic crop for the three Fig. 6 panels from a rendered paper page.

    We deliberately keep this dependency-light (PIL+NumPy only) so it can run
    outside the FEniCS environment.
    """
    page = Image.open(page_png).convert("RGB")
    gray = np.asarray(page.convert("L"), dtype=np.uint8)
    H, W = gray.shape

    # ROIs as fractions of (W,H). These values are tuned for the bundled
    # page renders under results/paper2007_pages/ (1182x1536) but scale with size.
    def _roi(x0f, x1f, y0f, y1f):
        return (int(round(x0f * W)), int(round(y0f * H)), int(round(x1f * W)), int(round(y1f * H)))

    thr = 140
    bbox_iface = _bbox_from_roi(gray, roi=_roi(0.04, 0.49, 0.10, 0.37), thr=thr)
    bbox_S = _bbox_from_roi(gray, roi=_roi(0.47, 0.98, 0.10, 0.37), thr=thr)
    bbox_Phi = _bbox_from_roi(gray, roi=_roi(0.17, 0.83, 0.38, 0.66), thr=thr)

    pad_x = int(round(0.04 * W))
    pad_y = int(round(0.05 * H))
    bbox_iface = _expand_bbox(bbox_iface, pad_x=pad_x, pad_y=pad_y, W=W, H=H)
    bbox_S = _expand_bbox(bbox_S, pad_x=pad_x, pad_y=pad_y, W=W, H=H)
    bbox_Phi = _expand_bbox(bbox_Phi, pad_x=pad_x, pad_y=pad_y, W=W, H=H)

    return {
        "interface": page.crop(bbox_iface),
        "S": page.crop(bbox_S),
        "Phi": page.crop(bbox_Phi),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--our-dir",
        type=str,
        required=True,
        help="Directory containing fig6a_interface.png, fig6b_S.png, fig6c_Phi.png.",
    )
    ap.add_argument(
        "--extra-dir",
        type=str,
        action="append",
        default=[],
        help="Additional result directory to add as extra columns (repeatable).",
    )
    ap.add_argument("--paper-label", type=str, default="Paper")
    ap.add_argument("--our-label", type=str, default="Our")
    ap.add_argument("--extra-label", type=str, action="append", default=[], help="Labels for --extra-dir (repeatable).")
    ap.add_argument("--no-labels", action="store_true", help="Disable column labels above each panel.")
    ap.add_argument("--label-height", type=int, default=28, help="Label band height in pixels (if labels enabled).")
    ap.add_argument(
        "--paper-dir",
        type=str,
        default="examples/biofilms/benchmarks/dadu/results/paper2007_pages",
        help="Directory containing rendered paper pages (e.g. page-18.png) or pre-cropped panels.",
    )
    ap.add_argument(
        "--paper-page",
        type=str,
        default="page-17.png",
        help="Paper page PNG filename inside --paper-dir that contains Fig. 6 (default: page-17.png).",
    )
    ap.add_argument("--out", type=str, default="", help="Output path for the combined comparison PNG (default: <our-dir>/compare_fig6.png).")
    args = ap.parse_args()

    our_dir = Path(str(args.our_dir))
    paper_dir = Path(str(args.paper_dir))
    out_path = Path(str(args.out)) if str(args.out).strip() else (our_dir / "compare_fig6.png")

    result_dirs = [our_dir] + [Path(str(d)) for d in (args.extra_dir or [])]
    if args.extra_label and (len(args.extra_label) != len(result_dirs) - 1):
        raise ValueError("--extra-label must be provided once per --extra-dir (or omitted).")
    labels = [str(args.our_label)] + (list(args.extra_label) if args.extra_label else [d.name for d in result_dirs[1:]])

    # Load or auto-crop paper panels.
    paper_panels: dict[str, Image.Image] = {}
    # Preferred: pre-cropped panels in the directory (if user has them).
    page_stem = Path(str(args.paper_page)).stem
    cand_iface = paper_dir / f"{page_stem}_interface_fullcrop.png"
    cand_S = paper_dir / f"{page_stem}_S_crop.png"
    cand_Phi = paper_dir / f"{page_stem}_Phi_crop.png"
    if cand_iface.exists() and cand_S.exists() and cand_Phi.exists():
        paper_panels = {"interface": _load_rgb(cand_iface), "S": _load_rgb(cand_S), "Phi": _load_rgb(cand_Phi)}
    else:
        page_png = paper_dir / str(args.paper_page)
        if not page_png.exists():
            raise FileNotFoundError(f"Paper page not found: {page_png}")
        paper_panels = _auto_crop_paper_fig6_panels(page_png)

    def _cols(fig: str, *, height: int) -> list[Image.Image]:
        cols: list[Image.Image] = []

        paper_img = _resize_to_height(paper_panels[fig], int(height))
        if not bool(args.no_labels):
            paper_img = _with_label(paper_img, str(args.paper_label), height=int(args.label_height))
        cols.append(paper_img)

        for d, lab in zip(result_dirs, labels):
            p = d / {"interface": "fig6a_interface.png", "S": "fig6b_S.png", "Phi": "fig6c_Phi.png"}[fig]
            if not p.exists():
                raise FileNotFoundError(f"Missing required file: {p}")
            im = _resize_to_height(_load_rgb(p), int(height))
            if not bool(args.no_labels):
                im = _with_label(im, str(lab), height=int(args.label_height))
            cols.append(im)
        return cols

    # Row 1: interface motion
    row1 = _hcat_many(_cols("interface", height=420))

    # Row 2: substrate
    row2 = _hcat_many(_cols("S", height=360))

    # Row 3: Phi
    row3 = _hcat_many(_cols("Phi", height=360))

    out = _vcat([row1, row2, row3])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    print(f"- Wrote {out_path}")


if __name__ == "__main__":
    main()