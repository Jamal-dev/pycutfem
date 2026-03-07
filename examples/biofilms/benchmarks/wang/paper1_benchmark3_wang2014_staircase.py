#!/usr/bin/env python3
"""Paper 1 Benchmark 3: Wang2014-style stepped-interface reduced comparison.

This benchmark complements the layered Wang2014 Example 6.1 analogue with a
non-flat stepped geometry inspired by Wang2014 Example 6.2. The goal is not to
rebuild the full mixed Stokes--Darcy discretization from the original paper
inside this repository. Instead, for Paper 1 we compare:

  - a one-domain diffuse transition-layer solve for the first velocity
    component on the full rectangle,
  - against a two-region sharp-interface reduction of the same tangential
    operator on Wang's three-step geometry.

The benchmark is therefore a geometry-sensitive representation test for the
reduced Paper 1 mechanics narrative:

  - layered benchmark: clean asymptotic rates on a flat interface,
  - staircase benchmark: non-flat sharp-interface agreement on a stepped
    geometry with interface-resolving meshes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception:  # pragma: no cover - handled at runtime for fenicsx-only deps
    sp = None
    spla = None


@dataclass(frozen=True)
class WangExample62Parameters:
    phi_p: float = 0.4
    mu: float = 1.0
    transition_sharpness: float = 100.0
    x_max: float = 2.0
    y_max: float = 2.0
    profile_x: float = 0.25


STEP_SEGMENTS = (
    (0.0, 1.5, 0.5, 1.5),
    (0.5, 1.5, 0.5, 1.0),
    (0.5, 1.0, 1.0, 1.0),
    (1.0, 1.0, 1.0, 0.5),
    (1.0, 0.5, 1.5, 0.5),
    (1.5, 0.5, 1.5, 0.0),
)


def _require_scipy() -> None:
    if sp is None or spla is None:
        raise RuntimeError(
            "Benchmark 3 staircase requires scipy. "
            "Run it inside the fenicsx environment."
        )


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for item in str(raw).split(","):
        text = item.strip()
        if not text:
            continue
        out.append(float(text))
    if not out:
        raise ValueError("Expected at least one numeric value.")
    return out


def _parse_grid_pairs(raw: str) -> list[tuple[float, int]]:
    pairs: list[tuple[float, int]] = []
    for item in str(raw).split(","):
        text = item.strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(
                "Expected permeability/grid pairs in the form '1e-2:128,1e-3:256'."
            )
        k_text, n_text = text.split(":", 1)
        K = float(k_text.strip())
        nxy = int(n_text.strip())
        if nxy < 16:
            raise ValueError("Each staircase grid resolution must be at least 16.")
        pairs.append((K, nxy))
    if not pairs:
        raise ValueError("Expected at least one permeability/grid pair.")
    return pairs


def _format_k_tag(K: float) -> str:
    return f"{float(K):.0e}".replace("+", "")


def _interface_height(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    out[(x >= 0.0) & (x < 0.5)] = 1.5
    out[(x >= 0.5) & (x < 1.0)] = 1.0
    out[(x >= 1.0) & (x < 1.5)] = 0.5
    out[(x >= 1.5)] = 0.0
    return out


def _point_segment_distance(
    px: np.ndarray,
    py: np.ndarray,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> np.ndarray:
    vx = float(x1 - x0)
    vy = float(y1 - y0)
    wx = np.asarray(px, dtype=float) - float(x0)
    wy = np.asarray(py, dtype=float) - float(y0)
    vv = vx * vx + vy * vy
    t = np.clip((wx * vx + wy * vy) / vv, 0.0, 1.0)
    projx = float(x0) + t * vx
    projy = float(y0) + t * vy
    return np.sqrt((np.asarray(px, dtype=float) - projx) ** 2 + (np.asarray(py, dtype=float) - projy) ** 2)


def _signed_distance_to_staircase(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    dist = np.full_like(np.asarray(X, dtype=float), float("inf"), dtype=float)
    for x0, y0, x1, y1 in STEP_SEGMENTS:
        dist = np.minimum(
            dist,
            _point_segment_distance(X, Y, x0=x0, y0=y0, x1=x1, y1=y1),
        )
    porous = np.asarray(Y, dtype=float) <= _interface_height(np.asarray(X, dtype=float))
    return np.where(porous, -dist, dist)


def _dirichlet_data(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return y * y * (y - 2.0)


def _diffuse_coefficients(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    K: float,
    params: WangExample62Parameters,
) -> tuple[np.ndarray, np.ndarray]:
    eps = math.sqrt(float(K))
    dist = _signed_distance_to_staircase(X, Y)
    arg = float(params.transition_sharpness) * dist / eps
    phi1 = 0.5 * (1.0 - 1.0 / float(params.phi_p)) * np.tanh(arg) + 0.5 * (1.0 + 1.0 / float(params.phi_p))
    phi2 = 0.5 - 0.5 * np.tanh(arg)
    return phi1, (float(params.mu) / float(K)) * phi2


def _sharp_coefficients(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    K: float,
    params: WangExample62Parameters,
) -> tuple[np.ndarray, np.ndarray]:
    porous = np.asarray(Y, dtype=float) <= _interface_height(np.asarray(X, dtype=float))
    a = np.where(porous, float(params.mu) / float(params.phi_p), float(params.mu))
    b = np.where(porous, float(params.mu) / float(K), 0.0)
    return a, b


def _solve_scalar_operator(
    *,
    nxy: int,
    K: float,
    params: WangExample62Parameters,
    diffuse: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _require_scipy()
    nxy = int(nxy)
    x = np.linspace(0.0, float(params.x_max), nxy + 1, dtype=float)
    y = np.linspace(0.0, float(params.y_max), nxy + 1, dtype=float)
    hx = float(x[1] - x[0])
    hy = float(y[1] - y[0])
    X, Y = np.meshgrid(x, y, indexing="ij")
    if diffuse:
        a, b = _diffuse_coefficients(X, Y, K=float(K), params=params)
    else:
        a, b = _sharp_coefficients(X, Y, K=float(K), params=params)

    ndofs = (nxy + 1) * (nxy + 1)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros((ndofs,), dtype=float)

    def idx(i: int, j: int) -> int:
        return int(i) * (nxy + 1) + int(j)

    for i in range(nxy + 1):
        for j in range(nxy + 1):
            row = idx(i, j)

            # Dirichlet on top and right boundaries.
            if i == nxy or j == nxy:
                rows.append(row)
                cols.append(row)
                data.append(1.0)
                rhs[row] = float(_dirichlet_data(np.array([y[j]], dtype=float))[0]) if i == nxy and j < nxy else 0.0
                continue

            # Zero Neumann on left and bottom boundaries via mirrored one-sided equations.
            if i == 0 and j == 0:
                rows.extend([row, row, row])
                cols.extend([idx(0, 0), idx(1, 0), idx(0, 1)])
                data.extend([2.0, -1.0, -1.0])
                rhs[row] = 0.0
                continue
            if i == 0:
                rows.extend([row, row])
                cols.extend([idx(0, j), idx(1, j)])
                data.extend([1.0, -1.0])
                rhs[row] = 0.0
                continue
            if j == 0:
                rows.extend([row, row])
                cols.extend([idx(i, 0), idx(i, 1)])
                data.extend([1.0, -1.0])
                rhs[row] = 0.0
                continue

            ae = 0.5 * (float(a[i, j]) + float(a[i + 1, j]))
            aw = 0.5 * (float(a[i, j]) + float(a[i - 1, j]))
            an = 0.5 * (float(a[i, j]) + float(a[i, j + 1]))
            a_s = 0.5 * (float(a[i, j]) + float(a[i, j - 1]))

            rows.extend([row, row, row, row, row])
            cols.extend(
                [
                    idx(i, j),
                    idx(i + 1, j),
                    idx(i - 1, j),
                    idx(i, j + 1),
                    idx(i, j - 1),
                ]
            )
            data.extend(
                [
                    (ae + aw) / (hx * hx) + (an + a_s) / (hy * hy) + float(b[i, j]),
                    -ae / (hx * hx),
                    -aw / (hx * hx),
                    -an / (hy * hy),
                    -a_s / (hy * hy),
                ]
            )

    mat = sp.csr_matrix((np.asarray(data, dtype=float), (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))), shape=(ndofs, ndofs))
    sol = np.asarray(spla.spsolve(mat, rhs), dtype=float)
    return x, y, sol.reshape((nxy + 1, nxy + 1)), X, Y


def _rho_wang(err: float, K: float) -> float:
    if err <= 0.0 or K <= 0.0:
        return float("nan")
    return math.log(err) / math.log(K)


def _regional_error_metrics(
    *,
    x: np.ndarray,
    y: np.ndarray,
    u_one: np.ndarray,
    u_two: np.ndarray,
    K: float,
    nxy: int,
    params: WangExample62Parameters,
) -> dict[str, float]:
    X, Y = np.meshgrid(np.asarray(x, dtype=float), np.asarray(y, dtype=float), indexing="ij")
    porous = np.asarray(Y, dtype=float) <= _interface_height(np.asarray(X, dtype=float))
    fluid = ~porous
    diff = np.asarray(u_one, dtype=float) - np.asarray(u_two, dtype=float)
    grad_x, grad_y = np.gradient(diff, x, y, edge_order=2)
    weight = float((x[-1] - x[0]) / (len(x) - 1) * (y[-1] - y[0]) / (len(y) - 1))

    l2_fluid = math.sqrt(float(np.sum(diff[fluid] ** 2) * weight))
    l2_porous = math.sqrt(float(np.sum(diff[porous] ** 2) * weight))
    h1_fluid = math.sqrt(float(np.sum((grad_x[fluid] ** 2 + grad_y[fluid] ** 2)) * weight))
    h1_porous = math.sqrt(float(np.sum((grad_x[porous] ** 2 + grad_y[porous] ** 2)) * weight))

    ix = int(np.argmin(np.abs(np.asarray(x, dtype=float) - float(params.profile_x))))
    profile_diff = np.asarray(diff[ix, :], dtype=float)
    profile_l2 = math.sqrt(float(np.trapezoid(profile_diff ** 2, np.asarray(y, dtype=float))))
    profile_linf = float(np.max(np.abs(profile_diff)))
    profile_y_int = float(_interface_height(np.array([float(x[ix])], dtype=float))[0])
    band = np.abs(np.asarray(y, dtype=float) - profile_y_int) <= max(float(x[1] - x[0]), 0.05)
    interface_band_linf = float(np.max(np.abs(profile_diff[band]))) if np.any(band) else profile_linf

    return {
        "K": float(K),
        "epsilon": math.sqrt(float(K)),
        "nxy": int(nxy),
        "hx": float(x[1] - x[0]),
        "profile_x_sample": float(x[ix]),
        "profile_y_interface": profile_y_int,
        "l2_fluid": l2_fluid,
        "rho_l2_fluid": _rho_wang(l2_fluid, float(K)),
        "l2_porous": l2_porous,
        "rho_l2_porous": _rho_wang(l2_porous, float(K)),
        "h1_fluid": h1_fluid,
        "rho_h1_fluid": _rho_wang(h1_fluid, float(K)),
        "h1_porous": h1_porous,
        "rho_h1_porous": _rho_wang(h1_porous, float(K)),
        "profile_l2": profile_l2,
        "profile_linf": profile_linf,
        "interface_band_linf": interface_band_linf,
        "max_field_abs_error": float(np.max(np.abs(diff))),
    }


def _write_summary_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = [
        "K",
        "epsilon",
        "nxy",
        "hx",
        "profile_x_sample",
        "profile_y_interface",
        "l2_fluid",
        "rho_l2_fluid",
        "l2_porous",
        "rho_l2_porous",
        "h1_fluid",
        "rho_h1_fluid",
        "h1_porous",
        "rho_h1_porous",
        "profile_l2",
        "profile_linf",
        "interface_band_linf",
        "max_field_abs_error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_profile_csv(
    path: Path,
    *,
    y: np.ndarray,
    one_domain: np.ndarray,
    two_domain: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y", "one_domain", "two_domain", "abs_error"])
        for yy, uo, ut in zip(np.asarray(y, dtype=float), np.asarray(one_domain, dtype=float), np.asarray(two_domain, dtype=float)):
            writer.writerow(
                [
                    f"{float(yy):.16e}",
                    f"{float(uo):.16e}",
                    f"{float(ut):.16e}",
                    f"{float(abs(uo - ut)):.16e}",
                ]
            )


def _polyline_xy() -> tuple[np.ndarray, np.ndarray]:
    xs = [0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5]
    ys = [1.5, 1.5, 1.0, 1.0, 0.5, 0.5, 0.0]
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _plot_geometry(path: Path, *, params: WangExample62Parameters, dpi: int) -> None:
    xs, ys = _polyline_xy()
    fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    ax.fill([0.0, 0.0, *xs, 1.5, 0.0], [0.0, 1.5, *ys, 0.0, 0.0], color="#D8D8C2", alpha=0.85, label="porous region")
    ax.plot(xs, ys, color="#1F4E5F", lw=2.4, label="sharp interface")
    ax.axvline(float(params.profile_x), color="#BA3F1D", lw=1.5, linestyle="--", label=fr"profile line $x={float(params.profile_x):.2f}$")
    ax.set_xlim(0.0, float(params.x_max))
    ax.set_ylim(0.0, float(params.y_max))
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Wang2014-style stepped geometry")
    ax.grid(True, alpha=0.20)
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def _plot_profiles(
    path: Path,
    *,
    plot_ks: list[float],
    solutions: dict[float, dict[str, object]],
    dpi: int,
) -> None:
    nrows = len(plot_ks)
    fig, axes = plt.subplots(nrows, 2, figsize=(10.5, 4.2 * nrows), constrained_layout=True, squeeze=False)
    for row, K in enumerate(plot_ks):
        payload = solutions[float(K)]
        y = np.asarray(payload["y"], dtype=float)
        one_prof = np.asarray(payload["one_profile"], dtype=float)
        two_prof = np.asarray(payload["two_profile"], dtype=float)
        y_int = float(payload["profile_y_interface"])

        ax_full = axes[row, 0]
        ax_zoom = axes[row, 1]
        ax_full.plot(one_prof, y, color="#145A7B", lw=2.2, label="one-domain diffuse")
        ax_full.plot(two_prof, y, color="#BA3F1D", lw=2.0, linestyle="--", label="sharp split model")
        ax_full.axhline(y_int, color="0.35", lw=1.0, linestyle=":")
        ax_full.set_title(f"$K={float(K):.0e}$ profile at $x={float(payload['profile_x_sample']):.2f}$")
        ax_full.set_xlabel(r"$u_x$")
        ax_full.set_ylabel(r"$y$")
        ax_full.grid(True, alpha=0.25)
        if row == 0:
            ax_full.legend(loc="best")

        ax_zoom.plot(one_prof, y, color="#145A7B", lw=2.2)
        ax_zoom.plot(two_prof, y, color="#BA3F1D", lw=2.0, linestyle="--")
        ax_zoom.axhline(y_int, color="0.35", lw=1.0, linestyle=":")
        ax_zoom.set_xlim(
            min(float(np.min(one_prof)), float(np.min(two_prof))) - 0.01,
            max(float(np.max(one_prof)), float(np.max(two_prof))) + 0.01,
        )
        ax_zoom.set_ylim(y_int - 0.14, y_int + 0.14)
        ax_zoom.set_title("Zoom near the interface")
        ax_zoom.set_xlabel(r"$u_x$")
        ax_zoom.set_ylabel(r"$y$")
        ax_zoom.grid(True, alpha=0.25)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def _plot_field_panels(
    path: Path,
    *,
    plot_ks: list[float],
    solutions: dict[float, dict[str, object]],
    dpi: int,
) -> None:
    nrows = len(plot_ks)
    fig, axes = plt.subplots(nrows, 3, figsize=(12.8, 4.0 * nrows), constrained_layout=True, squeeze=False)
    xs, ys = _polyline_xy()
    for row, K in enumerate(plot_ks):
        payload = solutions[float(K)]
        x = np.asarray(payload["x"], dtype=float)
        y = np.asarray(payload["y"], dtype=float)
        one_field = np.asarray(payload["one_field"], dtype=float).T
        two_field = np.asarray(payload["two_field"], dtype=float).T
        err_field = np.abs(one_field - two_field)
        for ax, field, title, cmap in (
            (axes[row, 0], one_field, "one-domain diffuse", "viridis"),
            (axes[row, 1], two_field, "sharp split model", "viridis"),
            (axes[row, 2], err_field, "absolute error", "magma"),
        ):
            im = ax.imshow(
                field,
                origin="lower",
                extent=(float(x[0]), float(x[-1]), float(y[0]), float(y[-1])),
                aspect="equal",
                cmap=cmap,
            )
            ax.plot(xs, ys, color="w" if title == "absolute error" else "k", lw=1.1, linestyle="--")
            ax.set_title(f"$K={float(K):.0e}$, {title}")
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def _plot_error_trends(path: Path, *, rows: list[dict[str, float]], dpi: int) -> None:
    K = np.asarray([row["K"] for row in rows], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), constrained_layout=True)

    axes[0].loglog(K, [row["l2_fluid"] for row in rows], "o-", lw=2.0, color="#145A7B", label=r"$L^2(\Omega_f)$")
    axes[0].loglog(K, [row["l2_porous"] for row in rows], "s-", lw=2.0, color="#2F8F5B", label=r"$L^2(\Omega_p)$")
    axes[0].set_xlabel(r"$K$")
    axes[0].set_ylabel("regional L2 error")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].loglog(K, [row["h1_fluid"] for row in rows], "o-", lw=2.0, color="#BA3F1D", label=r"$H^1(\Omega_f)$")
    axes[1].loglog(K, [row["h1_porous"] for row in rows], "s-", lw=2.0, color="#6A4C93", label=r"$H^1(\Omega_p)$")
    axes[1].set_xlabel(r"$K$")
    axes[1].set_ylabel("regional semi-H1 error")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(loc="best")

    axes[2].loglog(K, [row["profile_linf"] for row in rows], "o-", lw=2.0, color="#7A5C28", label=r"profile $L^\infty$")
    axes[2].loglog(K, [row["interface_band_linf"] for row in rows], "s-", lw=2.0, color="#B44F6B", label=r"interface-band $L^\infty$")
    axes[2].set_xlabel(r"$K$")
    axes[2].set_ylabel("profile error")
    axes[2].grid(True, which="both", alpha=0.25)
    axes[2].legend(loc="best")

    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def run_benchmark(
    *,
    outdir: Path,
    grid_pairs: list[tuple[float, int]],
    plot_ks: list[float],
    dpi: int,
    params: WangExample62Parameters | None = None,
) -> dict[str, object]:
    params = params or WangExample62Parameters()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_ks = [float(K) for K in plot_ks]
    rows: list[dict[str, float]] = []
    solutions: dict[float, dict[str, object]] = {}

    for K, nxy in grid_pairs:
        K = float(K)
        nxy = int(nxy)
        x, y, u_one, _X, _Y = _solve_scalar_operator(nxy=nxy, K=K, params=params, diffuse=True)
        _x2, _y2, u_two, _X2, _Y2 = _solve_scalar_operator(nxy=nxy, K=K, params=params, diffuse=False)
        metric = _regional_error_metrics(
            x=x,
            y=y,
            u_one=u_one,
            u_two=u_two,
            K=K,
            nxy=nxy,
            params=params,
        )
        rows.append(metric)

        ix = int(np.argmin(np.abs(np.asarray(x, dtype=float) - float(params.profile_x))))
        one_profile = np.asarray(u_one[ix, :], dtype=float)
        two_profile = np.asarray(u_two[ix, :], dtype=float)

        if K in plot_ks:
            tag = _format_k_tag(K)
            _write_profile_csv(
                outdir / f"benchmark3_wang2014_staircase_profile_{tag}.csv",
                y=y,
                one_domain=one_profile,
                two_domain=two_profile,
            )
            solutions[K] = {
                "x": np.asarray(x, dtype=float),
                "y": np.asarray(y, dtype=float),
                "one_field": np.asarray(u_one, dtype=float),
                "two_field": np.asarray(u_two, dtype=float),
                "one_profile": one_profile,
                "two_profile": two_profile,
                "profile_x_sample": float(metric["profile_x_sample"]),
                "profile_y_interface": float(metric["profile_y_interface"]),
            }

    rows.sort(key=lambda row: float(row["K"]), reverse=True)
    summary_csv = outdir / "benchmark3_wang2014_staircase_summary.csv"
    _write_summary_csv(summary_csv, rows)
    _plot_geometry(outdir / "benchmark3_wang2014_staircase_geometry.png", params=params, dpi=int(dpi))
    if solutions:
        _plot_profiles(
            outdir / "benchmark3_wang2014_staircase_profiles.png",
            plot_ks=plot_ks,
            solutions=solutions,
            dpi=int(dpi),
        )
        _plot_field_panels(
            outdir / "benchmark3_wang2014_staircase_fields.png",
            plot_ks=plot_ks,
            solutions=solutions,
            dpi=int(dpi),
        )
    _plot_error_trends(
        outdir / "benchmark3_wang2014_staircase_error_trends.png",
        rows=rows,
        dpi=int(dpi),
    )

    summary = {
        "benchmark": "wang2014_example62_staircase_reduced",
        "params": asdict(params),
        "grid_pairs": [{"K": float(K), "nxy": int(nxy)} for K, nxy in grid_pairs],
        "plot_ks": plot_ks,
        "rows": rows,
        "summary_csv": str(summary_csv),
        "geometry_png": str(outdir / "benchmark3_wang2014_staircase_geometry.png"),
        "profiles_png": str(outdir / "benchmark3_wang2014_staircase_profiles.png"),
        "fields_png": str(outdir / "benchmark3_wang2014_staircase_fields.png"),
        "error_trends_png": str(outdir / "benchmark3_wang2014_staircase_error_trends.png"),
    }
    (outdir / "benchmark3_wang2014_staircase_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper 1 Benchmark 3: Wang2014-style stepped-interface reduced comparison.")
    ap.add_argument(
        "--outdir",
        type=str,
        default="examples/biofilms/benchmarks/wang/results/paper1_benchmark3_wang2014_staircase",
    )
    ap.add_argument("--grid-pairs", type=str, default="1e-2:128,1e-3:256,1e-4:512")
    ap.add_argument("--plot-k", type=str, default="1e-2,1e-4")
    ap.add_argument("--png-dpi", type=int, default=220)
    ap.add_argument("--transition-sharpness", type=float, default=100.0)
    args = ap.parse_args()

    summary = run_benchmark(
        outdir=Path(args.outdir),
        grid_pairs=_parse_grid_pairs(args.grid_pairs),
        plot_ks=_parse_float_list(args.plot_k),
        dpi=int(args.png_dpi),
        params=WangExample62Parameters(transition_sharpness=float(args.transition_sharpness)),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
