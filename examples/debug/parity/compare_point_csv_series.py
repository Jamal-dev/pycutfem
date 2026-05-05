from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CsvField:
    coords: np.ndarray  # (N,2)
    values: np.ndarray  # (N,M)
    headers: list[str]


def _read_csv(path: Path) -> CsvField:
    raw = np.genfromtxt(path, delimiter=",", names=True)
    headers = list(raw.dtype.names or [])
    if len(headers) < 3 or headers[0] != "x" or headers[1] != "y":
        raise ValueError(f"{path} must have columns x,y,<field...>; got {headers}")
    coords = np.column_stack([raw["x"], raw["y"]]).astype(float)
    vals = np.column_stack([raw[name] for name in headers[2:]]).astype(float)
    return CsvField(coords=coords, values=vals, headers=headers[2:])


def _match_by_coords(a: CsvField, b: CsvField, *, tol: float) -> np.ndarray:
    """
    Return index array `ib` so that a.coords[i] matches b.coords[ib[i]].
    Uses nearest-neighbor matching with a strict tolerance check.
    """
    if a.coords.shape[0] != b.coords.shape[0]:
        raise ValueError(f"Different point counts: {a.coords.shape[0]} vs {b.coords.shape[0]}")
    if tol <= 0:
        raise ValueError("--tol must be positive")

    diff = a.coords[:, None, :] - b.coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)

    ib = np.argmin(dist2, axis=1).astype(int)
    min_dist = np.sqrt(np.min(dist2, axis=1))
    max_dist = float(np.max(min_dist))
    if max_dist > tol:
        bad = int(np.argmax(min_dist))
        raise ValueError(
            f"Coordinate mismatch: max_dist={max_dist:.3e} > tol={tol:.3e}; "
            f"a={a.coords[bad]} b={b.coords[ib[bad]]}"
        )

    uniq = np.unique(ib)
    if uniq.size != a.coords.shape[0]:
        raise ValueError(
            "Nearest-neighbor match is not one-to-one (duplicate matches). "
            "Try a smaller mesh or increase --tol."
        )
    return ib


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare a time series of pointwise CSV outputs by coordinate matching (deal.II vs pycutfem)."
    )
    ap.add_argument("--dealii-dir", type=str, required=True, help="Directory with deal.II CSV files.")
    ap.add_argument("--pycutfem-dir", type=str, required=True, help="Directory with pycutfem CSV files.")
    ap.add_argument(
        "--dealii-pattern",
        type=str,
        default="solution_Miehe_test_{step:06d}.csv",
        help="Filename pattern inside --dealii-dir.",
    )
    ap.add_argument(
        "--pycutfem-pattern",
        type=str,
        default="points_{step:06d}.csv",
        help="Filename pattern inside --pycutfem-dir.",
    )
    ap.add_argument("--step-start", type=int, default=1, help="First step index to compare (inclusive).")
    ap.add_argument("--step-end", type=int, required=True, help="Last step index to compare (inclusive).")
    ap.add_argument("--tol", type=float, default=1.0e-12, help="Coordinate matching tolerance.")
    args = ap.parse_args()

    dealii_dir = Path(args.dealii_dir)
    pycutfem_dir = Path(args.pycutfem_dir)
    step_start = int(args.step_start)
    step_end = int(args.step_end)
    if step_end < step_start:
        raise ValueError("--step-end must be >= --step-start")

    # Build fixed coordinate map from the first compared step.
    a0 = _read_csv(dealii_dir / args.dealii_pattern.format(step=step_start))
    b0 = _read_csv(pycutfem_dir / args.pycutfem_pattern.format(step=step_start))
    if a0.headers != b0.headers:
        raise ValueError(f"Field headers differ: {a0.headers} vs {b0.headers}")
    ib = _match_by_coords(a0, b0, tol=float(args.tol))

    n_fields = len(a0.headers)
    max_abs_global = np.zeros(n_fields)
    max_abs_step = np.full(n_fields, -1, dtype=int)
    max_abs_point = np.full(n_fields, -1, dtype=int)
    rms_global = np.zeros(n_fields)
    rms_step = np.full(n_fields, -1, dtype=int)

    final_max_abs = None
    final_rms = None

    for step in range(step_start, step_end + 1):
        a = _read_csv(dealii_dir / args.dealii_pattern.format(step=step))
        b = _read_csv(pycutfem_dir / args.pycutfem_pattern.format(step=step))
        if a.headers != a0.headers or b.headers != a0.headers:
            raise ValueError("CSV headers changed across steps.")

        diffs = a.values - b.values[ib]
        absdiff = np.abs(diffs)
        max_abs = np.max(absdiff, axis=0)
        rms = np.sqrt(np.mean(diffs**2, axis=0))

        for k in range(n_fields):
            if max_abs[k] > max_abs_global[k]:
                max_abs_global[k] = float(max_abs[k])
                max_abs_step[k] = step
                max_abs_point[k] = int(np.argmax(absdiff[:, k]))
            if rms[k] > rms_global[k]:
                rms_global[k] = float(rms[k])
                rms_step[k] = step

        if step == step_end:
            final_max_abs = max_abs
            final_rms = rms

    print(f"[series_compare] steps compared: {step_start}..{step_end}")
    for k, name in enumerate(a0.headers):
        pt = a0.coords[max_abs_point[k]]
        print(
            f"[series_compare] field={name:>4s} global_max_abs={max_abs_global[k]:.3e} "
            f"(step={max_abs_step[k]} at x={pt[0]:.5f},y={pt[1]:.5f}) "
            f"global_rms_max={rms_global[k]:.3e} (step={rms_step[k]})"
        )

    print(f"\n[series_compare] final step ({step_end}) stats:")
    assert final_max_abs is not None
    assert final_rms is not None
    for k, name in enumerate(a0.headers):
        print(f"[series_compare] field={name:>4s} max_abs={float(final_max_abs[k]):.3e} rms={float(final_rms[k]):.3e}")


if __name__ == "__main__":
    main()

