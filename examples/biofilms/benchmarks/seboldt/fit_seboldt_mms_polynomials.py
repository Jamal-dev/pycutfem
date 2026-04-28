#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def _load_columns(path: Path, *, required: list[str]) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing sample file: {path}")
    columns = {name: [] for name in required}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError(f"Sample file {path} is missing required columns: {missing}")
        for row in reader:
            for name in required:
                columns[name].append(float(row[name]))
    if not columns[required[0]]:
        raise RuntimeError(f"Sample file {path} contains no rows.")
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def _build_scale(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape((-1,))
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    center = 0.5 * (vmin + vmax)
    halfspan = max(0.5 * (vmax - vmin), 1.0e-14)
    return {
        "min": vmin,
        "max": vmax,
        "center": center,
        "halfspan": halfspan,
    }


def _normalize(values: np.ndarray, *, center: float, halfspan: float) -> np.ndarray:
    return (np.asarray(values, dtype=float) - float(center)) / float(halfspan)


def _tensor_basis(
    x_values: np.ndarray,
    y_values: np.ndarray,
    t_values: np.ndarray,
    *,
    deg_x: int,
    deg_y: int,
    deg_t: int,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    x_arr = np.asarray(x_values, dtype=float).reshape((-1,))
    y_arr = np.asarray(y_values, dtype=float).reshape((-1,))
    t_arr = np.asarray(t_values, dtype=float).reshape((-1,))
    if not (x_arr.shape == y_arr.shape == t_arr.shape):
        raise ValueError("x, y, and t arrays must share the same shape.")
    vx = np.column_stack([x_arr**i for i in range(int(deg_x) + 1)])
    vy = np.column_stack([y_arr**j for j in range(int(deg_y) + 1)])
    vt = np.column_stack([t_arr**k for k in range(int(deg_t) + 1)])
    terms: list[tuple[int, int, int]] = []
    blocks: list[np.ndarray] = []
    for ix in range(int(deg_x) + 1):
        for iy in range(int(deg_y) + 1):
            for it in range(int(deg_t) + 1):
                terms.append((ix, iy, it))
                blocks.append((vx[:, ix] * vy[:, iy] * vt[:, it]).reshape((-1, 1)))
    return np.hstack(blocks), terms


def _format_term(coef: float, *, ix: int, iy: int, it: int) -> str:
    pieces = [f"{coef:.16e}"]
    if ix > 0:
        pieces.append("x_hat" if ix == 1 else f"x_hat**{ix}")
    if iy > 0:
        pieces.append("y_hat" if iy == 1 else f"y_hat**{iy}")
    if it > 0:
        pieces.append("t_hat" if it == 1 else f"t_hat**{it}")
    return " * ".join(pieces)


def _build_expression(coeff_tensor: np.ndarray, *, tolerance: float) -> str:
    coeffs = np.asarray(coeff_tensor, dtype=float)
    lines: list[str] = []
    for ix in range(coeffs.shape[0]):
        for iy in range(coeffs.shape[1]):
            for it in range(coeffs.shape[2]):
                coef = float(coeffs[ix, iy, it])
                if abs(coef) <= float(tolerance):
                    continue
                lines.append(_format_term(coef, ix=ix, iy=iy, it=it))
    if not lines:
        return "0.0"
    return "\n    + ".join([lines[0], *lines[1:]])


def _fit_one_field(
    *,
    name: str,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    time_values: np.ndarray,
    samples: np.ndarray,
    deg_x: int,
    deg_y: int,
    deg_t: int,
    expression_tolerance: float,
) -> dict[str, object]:
    x_scale = _build_scale(x_ref)
    y_scale = _build_scale(y_ref)
    t_scale = _build_scale(time_values)
    x_hat = _normalize(x_ref, center=x_scale["center"], halfspan=x_scale["halfspan"])
    y_hat = _normalize(y_ref, center=y_scale["center"], halfspan=y_scale["halfspan"])
    t_hat = _normalize(time_values, center=t_scale["center"], halfspan=t_scale["halfspan"])
    design, terms = _tensor_basis(x_hat, y_hat, t_hat, deg_x=deg_x, deg_y=deg_y, deg_t=deg_t)
    coeff_vec, residuals, rank, singular_values = np.linalg.lstsq(design, np.asarray(samples, dtype=float), rcond=None)
    prediction = design @ coeff_vec
    diff = prediction - np.asarray(samples, dtype=float)
    coeff_tensor = coeff_vec.reshape((int(deg_x) + 1, int(deg_y) + 1, int(deg_t) + 1))
    sample_norm = float(np.linalg.norm(samples))
    diff_norm = float(np.linalg.norm(diff))
    r2_num = float(np.sum(diff * diff))
    centered = np.asarray(samples, dtype=float) - float(np.mean(samples))
    r2_den = float(np.sum(centered * centered))
    return {
        "name": name,
        "degrees": {"x": int(deg_x), "y": int(deg_y), "t": int(deg_t)},
        "num_samples": int(len(samples)),
        "rank": int(rank),
        "singular_values": np.asarray(singular_values, dtype=float).tolist(),
        "residual_sum_squares": float(residuals[0]) if residuals.size else float(np.sum(diff * diff)),
        "rmse": float(math.sqrt(float(np.mean(diff * diff)))),
        "linf": float(np.max(np.abs(diff))),
        "relative_l2": float(diff_norm / max(sample_norm, 1.0e-16)),
        "r2": float(1.0 - r2_num / max(r2_den, 1.0e-16)),
        "scales": {
            "x": x_scale,
            "y": y_scale,
            "t": t_scale,
        },
        "terms": [list(term) for term in terms],
        "coefficients": coeff_tensor.tolist(),
        "expression": _build_expression(coeff_tensor, tolerance=expression_tolerance),
    }


def _fit_domain(
    *,
    name: str,
    sample_path: Path,
    field_names: list[str],
    deg_x: int,
    deg_y: int,
    deg_t: int,
    expression_tolerance: float,
) -> dict[str, object]:
    required = ["step", "time", "x_ref", "y_ref", "x_phys", "y_phys", *field_names]
    data = _load_columns(sample_path, required=required)
    fields = {
        field_name: _fit_one_field(
            name=field_name,
            x_ref=data["x_ref"],
            y_ref=data["y_ref"],
            time_values=data["time"],
            samples=data[field_name],
            deg_x=deg_x,
            deg_y=deg_y,
            deg_t=deg_t,
            expression_tolerance=expression_tolerance,
        )
        for field_name in field_names
    }
    return {
        "name": name,
        "sample_path": str(sample_path),
        "num_rows": int(len(data["time"])),
        "steps": {
            "min": int(round(float(np.min(data["step"])))),
            "max": int(round(float(np.max(data["step"])))),
            "count_unique": int(np.unique(np.asarray(data["step"], dtype=int)).size),
        },
        "time": {
            "min": float(np.min(data["time"])),
            "max": float(np.max(data["time"])),
        },
        "reference_bounds": {
            "x": [float(np.min(data["x_ref"])), float(np.max(data["x_ref"]))],
            "y": [float(np.min(data["y_ref"])), float(np.max(data["y_ref"]))],
        },
        "physical_bounds": {
            "x": [float(np.min(data["x_phys"])), float(np.max(data["x_phys"]))],
            "y": [float(np.min(data["y_phys"])), float(np.max(data["y_phys"]))],
        },
        "fields": fields,
    }


def _render_expression_text(report: dict[str, object]) -> str:
    lines: list[str] = []
    for domain_name, domain_report in report["domains"].items():
        lines.append(f"[{domain_name}]")
        lines.append(f"sample_path = {domain_report['sample_path']}")
        lines.append(
            f"time_range = [{domain_report['time']['min']:.16e}, {domain_report['time']['max']:.16e}]"
        )
        for field_name, field_report in domain_report["fields"].items():
            scales = field_report["scales"]
            lines.append("")
            lines.append(f"{domain_name}.{field_name}:")
            lines.append(
                f"  x_hat = (x - {scales['x']['center']:.16e}) / {scales['x']['halfspan']:.16e}"
            )
            lines.append(
                f"  y_hat = (y - {scales['y']['center']:.16e}) / {scales['y']['halfspan']:.16e}"
            )
            lines.append(
                f"  t_hat = (t - {scales['t']['center']:.16e}) / {scales['t']['halfspan']:.16e}"
            )
            lines.append("  value =")
            lines.append(f"    {field_report['expression']}")
            lines.append(
                "  fit_metrics = "
                f"rmse={field_report['rmse']:.6e}, "
                f"linf={field_report['linf']:.6e}, "
                f"relative_l2={field_report['relative_l2']:.6e}, "
                f"r2={field_report['r2']:.6e}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_python_module(report: dict[str, object]) -> str:
    payload = json.dumps(report, indent=2, sort_keys=True)
    return f"""#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import numpy as np


REPORT = json.loads({payload!r})


def _normalize(values, *, center: float, halfspan: float):
    arr = np.asarray(values, dtype=float)
    return (arr - float(center)) / float(halfspan)


def _derivative_factor(power: int, order: int) -> float:
    if order > power:
        return 0.0
    value = 1.0
    for shift in range(order):
        value *= float(power - shift)
    return value


def _evaluate_tensor(coefficients, x, y, t, *, scales, dx: int = 0, dy: int = 0, dt: int = 0):
    x_hat = _normalize(x, center=scales["x"]["center"], halfspan=scales["x"]["halfspan"])
    y_hat = _normalize(y, center=scales["y"]["center"], halfspan=scales["y"]["halfspan"])
    t_hat = _normalize(t, center=scales["t"]["center"], halfspan=scales["t"]["halfspan"])
    out = np.zeros(np.broadcast(x_hat, y_hat, t_hat).shape, dtype=float)
    coeffs = np.asarray(coefficients, dtype=float)
    x_scale = float(scales["x"]["halfspan"]) ** int(dx)
    y_scale = float(scales["y"]["halfspan"]) ** int(dy)
    t_scale = float(scales["t"]["halfspan"]) ** int(dt)
    normalization = x_scale * y_scale * t_scale
    for ix in range(coeffs.shape[0]):
        fx = _derivative_factor(ix, int(dx))
        if fx == 0.0:
            continue
        for iy in range(coeffs.shape[1]):
            fy = _derivative_factor(iy, int(dy))
            if fy == 0.0:
                continue
            for it in range(coeffs.shape[2]):
                ft = _derivative_factor(it, int(dt))
                if ft == 0.0:
                    continue
                out = out + (
                    coeffs[ix, iy, it]
                    * fx
                    * fy
                    * ft
                    * np.power(x_hat, ix - int(dx))
                    * np.power(y_hat, iy - int(dy))
                    * np.power(t_hat, it - int(dt))
                )
    return out / normalization


def evaluate(domain: str, field: str, x, y, t):
    field_report = REPORT["domains"][str(domain)]["fields"][str(field)]
    return _evaluate_tensor(field_report["coefficients"], x, y, t, scales=field_report["scales"])


def evaluate_derivative(domain: str, field: str, x, y, t, *, dx: int = 0, dy: int = 0, dt: int = 0):
    field_report = REPORT["domains"][str(domain)]["fields"][str(field)]
    return _evaluate_tensor(
        field_report["coefficients"],
        x,
        y,
        t,
        scales=field_report["scales"],
        dx=int(dx),
        dy=int(dy),
        dt=int(dt),
    )


def solid_ux(x, y, t):
    return evaluate("solid", "ux", x, y, t)


def solid_uy(x, y, t):
    return evaluate("solid", "uy", x, y, t)


def solid_p_p(x, y, t):
    return evaluate("solid", "p_p", x, y, t)


def fluid_vx(x, y, t):
    return evaluate("fluid", "vx", x, y, t)


def fluid_vy(x, y, t):
    return evaluate("fluid", "vy", x, y, t)


def fluid_p(x, y, t):
    return evaluate("fluid", "p", x, y, t)
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="Coupled preCICE run directory containing fluid/ and porous/ outputs.")
    ap.add_argument("--solid-samples", help="Optional override for porous/mms_solid_samples.csv.")
    ap.add_argument("--fluid-samples", help="Optional override for fluid/mms_fluid_samples.csv.")
    ap.add_argument("--outdir", help="Output directory for polynomial fit artifacts. Defaults to <run-dir>/mms_fit.")
    ap.add_argument("--deg-x", type=int, default=4)
    ap.add_argument("--deg-y", type=int, default=4)
    ap.add_argument("--deg-t", type=int, default=4)
    ap.add_argument("--expression-tolerance", type=float, default=1.0e-12)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    solid_samples = Path(args.solid_samples).resolve() if args.solid_samples else run_dir / "porous" / "mms_solid_samples.csv"
    fluid_samples = Path(args.fluid_samples).resolve() if args.fluid_samples else run_dir / "fluid" / "mms_fluid_samples.csv"
    outdir = Path(args.outdir).resolve() if args.outdir else run_dir / "mms_fit"
    outdir.mkdir(parents=True, exist_ok=True)

    report = {
        "run_dir": str(run_dir),
        "degrees": {"x": int(args.deg_x), "y": int(args.deg_y), "t": int(args.deg_t)},
        "domains": {
            "solid": _fit_domain(
                name="solid",
                sample_path=solid_samples,
                field_names=["ux", "uy", "p_p"],
                deg_x=int(args.deg_x),
                deg_y=int(args.deg_y),
                deg_t=int(args.deg_t),
                expression_tolerance=float(args.expression_tolerance),
            ),
            "fluid": _fit_domain(
                name="fluid",
                sample_path=fluid_samples,
                field_names=["vx", "vy", "p"],
                deg_x=int(args.deg_x),
                deg_y=int(args.deg_y),
                deg_t=int(args.deg_t),
                expression_tolerance=float(args.expression_tolerance),
            ),
        },
    }

    report_path = outdir / "mms_polynomial_fit.json"
    expressions_path = outdir / "polynomial_expressions.txt"
    module_path = outdir / "mms_polynomials.py"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    expressions_path.write_text(_render_expression_text(report), encoding="utf-8")
    module_path.write_text(_render_python_module(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "expressions": str(expressions_path),
                "module": str(module_path),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
