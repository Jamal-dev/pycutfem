#!/usr/bin/env python3
"""
Strip embedded raster images from Inkscape SVGs.

Why
---
When tracing outlines on top of video frames in Inkscape, saving as SVG often embeds
the full PNG frame as a base64 `data:image/...` payload inside an <image> element.
This makes each SVG several MB and is not suitable for git.

This script removes only those <image> elements whose href is a data URL, keeping
the traced vector geometry (paths + marks) intact for downstream extraction.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


_IMG_DATA_URL_RE = re.compile(
    r"""(?is)            # ignore-case + dot matches newline
    <image\b             # opening tag
    [^>]*?               # attributes
    (?:xlink:href|href)  # href attribute name
    \s*=\s*              # equals
    "data:image/[^"]*"   # data URL payload (base64 etc)
    [^>]*?               # remaining attributes
    (?:/>|</image>)      # end tag (self-closing or explicit)
    """,
    re.VERBOSE,
)


def _strip_embedded_images(svg_text: str) -> tuple[str, int]:
    out, n = _IMG_DATA_URL_RE.subn("", svg_text)
    if n:
        # Avoid leaving large blank gaps where the image tag used to be.
        out = re.sub(r"\n{3,}", "\n\n", out)
    return out, int(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        type=str,
        default="examples/biofilms/benchmarks/lie/svg_fles",
        help="Directory containing frame_XXXX.svg files.",
    )
    ap.add_argument("--glob", type=str, default="*.svg", help="Glob for SVG filenames inside --dir.")
    ap.add_argument("--dry-run", action="store_true", help="Report potential changes but do not write files.")
    args = ap.parse_args()

    svg_dir = Path(str(args.dir))
    paths = sorted(svg_dir.glob(str(args.glob)))
    if not paths:
        raise SystemExit(f"No SVG files found in {svg_dir} matching {args.glob!r}.")

    total_removed = 0
    total_before = 0
    total_after = 0
    changed = 0

    for p in paths:
        raw = p.read_text(encoding="utf-8", errors="replace")
        total_before += len(raw.encode("utf-8", errors="replace"))
        stripped, n = _strip_embedded_images(raw)
        total_removed += n
        if stripped != raw:
            changed += 1
            if not bool(args.dry_run):
                p.write_text(stripped, encoding="utf-8")
        total_after += len(stripped.encode("utf-8", errors="replace"))

    saved = total_before - total_after
    print(
        f"[strip_svg_embedded_images] files={len(paths)} changed={changed} removed_image_tags={total_removed} "
        f"saved={saved/1024/1024:.1f} MiB",
        flush=True,
    )


if __name__ == "__main__":
    main()

