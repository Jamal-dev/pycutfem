"""
Extract rendered paper pages (PNG) from Duddu et al. (2007) PDF for comparisons.

This uses `pdftoppm` (Poppler) because it is typically available on Linux and
does not add Python dependencies. Outputs are written under
`examples/biofilms/benchmarks/dadu/results/`, which is gitignored.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


def _parse_pages(spec: str) -> list[int]:
    """
    Parse a page spec like: "18", "17,18,20", "17-20", "1,3-5,8".
    """
    out: set[int] = set()
    spec = str(spec).strip()
    if not spec:
        return []
    for part in re.split(r"[,\s]+", spec):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo = int(a)
            hi = int(b)
            if hi < lo:
                lo, hi = hi, lo
            for p in range(lo, hi + 1):
                out.add(int(p))
        else:
            out.add(int(part))
    return sorted(out)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=str, default="examples/biofilms/benchmarks/dadu/dadu_2007.pdf")
    ap.add_argument(
        "--outdir",
        type=str,
        default="examples/biofilms/benchmarks/dadu/results/paper2007_pages",
        help="Output directory for rendered pages.",
    )
    ap.add_argument("--pages", type=str, default="18", help="Pages to extract, e.g. '18' or '17,18'.")
    ap.add_argument("--dpi", type=int, default=150, help="Render DPI (pdftoppm -r).")
    ap.add_argument("--force", action="store_true", help="Overwrite existing page PNGs.")
    args = ap.parse_args()

    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise RuntimeError("pdftoppm not found on PATH. Install poppler-utils (Ubuntu/Debian) or poppler (conda).")

    pdf = Path(str(args.pdf))
    if not pdf.exists():
        raise FileNotFoundError(pdf)

    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    pages = _parse_pages(str(args.pages))
    if not pages:
        raise ValueError("No pages requested. Use --pages, e.g. --pages 18.")

    prefix = outdir / "page"
    for p in pages:
        dst = outdir / f"page-{p:02d}.png"
        if dst.exists() and not bool(args.force):
            print(f"- Skipping existing {dst}")
            continue

        # pdftoppm writes <prefix>-<p>.png (no zero padding).
        _run([pdftoppm, "-png", "-r", str(int(args.dpi)), "-f", str(int(p)), "-l", str(int(p)), str(pdf), str(prefix)])
        src = outdir / f"page-{int(p)}.png"
        if not src.exists():
            # Fallback: sometimes poppler appends leading zeros in uncommon builds; try glob.
            cands = sorted(outdir.glob(f"page-{int(p)}*.png"))
            if not cands:
                raise FileNotFoundError(f"Expected output page PNG not found for page {p}.")
            src = cands[0]

        if src.resolve() != dst.resolve():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
        print(f"- Wrote {dst}")


if __name__ == "__main__":
    main()

