"""
Compare interface height time series (y_top) between two Duddu (2007) Fig. 6 runs.

Inputs can be either:
  - a directory containing y_top_timeseries.csv, or
  - a direct path to a CSV file with columns: t_days, y_top_mm.

Outputs:
  - y_top_compare.csv (target-time samples + error metrics)
  - y_top_compare.png (overlay plot)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


_DEFAULT_TARGETS = [0.0, 1.1, 2.7, 4.6, 6.6, 8.7, 10.7, 12.7, 14.7, 16.7, 18.6, 20.6, 22.5, 24.5, 26.5, 28.6]


def _resolve_csv(path_or_dir: Path) -> Path:
    p = Path(str(path_or_dir))
    if p.is_dir():
        p = p / "y_top_timeseries.csv"
    if not p.exists():
        raise FileNotFoundError(f"y_top_timeseries.csv not found: {p}")
    return p


def _read_timeseries(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    t: list[float] = []
    y: list[float] = []
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        if (r.fieldnames is None) or ("t_days" not in r.fieldnames) or ("y_top_mm" not in r.fieldnames):
            raise ValueError(f"CSV must contain columns t_days and y_top_mm: {csv_path}")
        for row in r:
            if not row:
                continue
            try:
                t.append(float(row["t_days"]))
                y.append(float(row["y_top_mm"]))
            except Exception:
                continue
    if not t:
        raise ValueError(f"No data rows read from {csv_path}")
    tt = np.asarray(t, dtype=float)
    yy = np.asarray(y, dtype=float)
    order = np.argsort(tt)
    tt = tt[order]
    yy = yy[order]
    # Drop any duplicate times (keep last)
    if tt.size >= 2:
        keep = np.ones(tt.size, dtype=bool)
        keep[:-1] = tt[1:] != tt[:-1]
        tt = tt[keep]
        yy = yy[keep]
    return tt, yy


def _sample(tt: np.ndarray, yy: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if targets.size == 0:
        raise ValueError("No targets provided.")
    if np.any(targets < tt[0] - 1.0e-12) or np.any(targets > tt[-1] + 1.0e-12):
        raise ValueError(f"Targets must lie within [{tt[0]}, {tt[-1]}].")
    return np.interp(targets, tt, yy)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=str, required=True, help="Dir or CSV path for model A (e.g. XFEM).")
    ap.add_argument("--b", type=str, required=True, help="Dir or CSV path for model B (e.g. one-domain).")
    ap.add_argument("--label-a", type=str, default="A")
    ap.add_argument("--label-b", type=str, default="B")
    ap.add_argument(
        "--targets",
        type=str,
        default="",
        help="Comma-separated target times in days (default: Duddu 2007 Fig.6 times).",
    )
    ap.add_argument("--outdir", type=str, default="", help="Output directory (default: directory of --a).")
    args = ap.parse_args()

    a_csv = _resolve_csv(Path(str(args.a)))
    b_csv = _resolve_csv(Path(str(args.b)))

    outdir = Path(str(args.outdir)).expanduser() if str(args.outdir).strip() else a_csv.parent
    outdir.mkdir(parents=True, exist_ok=True)

    ta, ya = _read_timeseries(a_csv)
    tb, yb = _read_timeseries(b_csv)

    if str(args.targets).strip():
        targets = np.asarray([float(s) for s in str(args.targets).split(",") if s.strip()], dtype=float)
    else:
        targets = np.asarray(_DEFAULT_TARGETS, dtype=float)

    # If the requested targets exceed the overlap of both time spans, auto-clip.
    t_min = float(max(float(ta[0]), float(tb[0])))
    t_max = float(min(float(ta[-1]), float(tb[-1])))
    mask = (targets >= t_min - 1.0e-12) & (targets <= t_max + 1.0e-12)
    if not bool(np.all(mask)):
        dropped = targets[~mask]
        targets = targets[mask]
        if targets.size == 0:
            raise ValueError(f"No target times fall in the overlap [{t_min:.3g}, {t_max:.3g}] days.")
        print(f"[y_top] Clipped targets to overlap [{t_min:.3g}, {t_max:.3g}] d; dropped {dropped.size} target(s).")

    ya_t = _sample(ta, ya, targets)
    yb_t = _sample(tb, yb, targets)
    abs_err = np.abs(ya_t - yb_t)
    rel_err = abs_err / np.maximum(np.maximum(np.abs(ya_t), np.abs(yb_t)), 1.0e-14)

    rows: list[dict[str, object]] = []
    for i in range(int(targets.size)):
        rows.append(
            {
                "t_days": float(targets[i]),
                f"y_top_{args.label_a}_mm": float(ya_t[i]),
                f"y_top_{args.label_b}_mm": float(yb_t[i]),
                "abs_err_mm": float(abs_err[i]),
                "rel_err": float(rel_err[i]),
            }
        )

    out_csv = outdir / "y_top_compare.csv"
    _write_csv(out_csv, rows)

    # Summary stats
    mae = float(np.mean(abs_err))
    maxe = float(np.max(abs_err))
    final_a = float(ya_t[-1])
    final_b = float(yb_t[-1])
    final_abs = float(abs_err[-1])
    final_rel = float(rel_err[-1])
    print(f"[y_top] targets={targets.size}  MAE={mae:.4e} mm  max={maxe:.4e} mm")
    print(f"[y_top] final t={targets[-1]:.3f} d  {args.label_a}={final_a:.4e} mm  {args.label_b}={final_b:.4e} mm  |Δ|={final_abs:.4e} mm  rel={final_rel:.3%}")
    print(f"- Wrote {out_csv}")

    # Plot (optional: matplotlib is not always installed outside the FEniCS env)
    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
        ax.plot(ta, ya, "-", lw=1.5, label=str(args.label_a))
        ax.plot(tb, yb, "-", lw=1.5, label=str(args.label_b))
        ax.plot(targets, ya_t, "o", ms=3.0, alpha=0.75)
        ax.plot(targets, yb_t, "o", ms=3.0, alpha=0.75)
        ax.set_xlabel("t (days)")
        ax.set_ylabel("y_top (mm)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", frameon=False)
        out_png = outdir / "y_top_compare.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"- Wrote {out_png}")
    except Exception as exc:
        print(f"[y_top] Skipped plot (matplotlib unavailable): {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
