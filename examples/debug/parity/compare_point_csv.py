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


def _match_by_coords(a: CsvField, b: CsvField, *, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return index arrays (ia, ib) so that a.coords[ia] matches b.coords[ib].
    Uses nearest-neighbor matching with a strict tolerance check.
    """
    if a.coords.shape[0] != b.coords.shape[0]:
        raise ValueError(f"Different point counts: {a.coords.shape[0]} vs {b.coords.shape[0]}")
    if tol <= 0:
        raise ValueError("--tol must be positive")

    # Brute-force nearest-neighbor (N~1e3 for our parity runs; OK).
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

    # Ensure bijection (each point in b matched exactly once).
    uniq = np.unique(ib)
    if uniq.size != a.coords.shape[0]:
        raise ValueError(
            "Nearest-neighbor match is not one-to-one (duplicate matches). "
            "Try a smaller mesh or increase --tol."
        )

    ia = np.arange(a.coords.shape[0], dtype=int)
    return ia, ib


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare pointwise CSV outputs by coordinate matching.")
    ap.add_argument("a", type=str, help="CSV A with columns x,y,field...")
    ap.add_argument("b", type=str, help="CSV B with columns x,y,field...")
    ap.add_argument("--tol", type=float, default=1.0e-12, help="Coordinate matching tolerance.")
    args = ap.parse_args()

    a = _read_csv(Path(args.a))
    b = _read_csv(Path(args.b))
    if a.headers != b.headers:
        raise ValueError(f"Field headers differ: {a.headers} vs {b.headers}")

    ia, ib = _match_by_coords(a, b, tol=float(args.tol))
    diffs = a.values[ia] - b.values[ib]
    max_abs = np.max(np.abs(diffs), axis=0)
    rms = np.sqrt(np.mean(diffs**2, axis=0))

    for k, name in enumerate(a.headers):
        print(f"[compare_csv] field={name} max_abs={max_abs[k]:.3e} rms={rms[k]:.3e}")


if __name__ == "__main__":
    main()
