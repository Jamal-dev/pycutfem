from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class SolveReport:
    method: str
    converged: bool
    iterations: int
    residual_inf: float
    state: list[float]
    sigma_final: float | None = None
    accepted_steps: int | None = None
    note: str = ""


def plateau_residual(z: np.ndarray) -> np.ndarray:
    x, y = np.asarray(z, dtype=float).ravel()
    return np.array(
        [
            np.arctan(x + 2.0 * y),
            np.arctan(2.0 * x - y),
        ],
        dtype=float,
    )


def plateau_jacobian(z: np.ndarray) -> np.ndarray:
    x, y = np.asarray(z, dtype=float).ravel()
    u = x + 2.0 * y
    v = 2.0 * x - y
    with np.errstate(over="ignore", invalid="ignore"):
        du = 1.0 / (1.0 + u * u)
        dv = 1.0 / (1.0 + v * v)
    return np.array(
        [
            [du, 2.0 * du],
            [2.0 * dv, -dv],
        ],
        dtype=float,
    )


def plateau_root_linearization() -> np.ndarray:
    return np.array([[1.0, 2.0], [2.0, -1.0]], dtype=float)


def residual_inf(z: np.ndarray) -> float:
    return float(np.linalg.norm(plateau_residual(z), ord=np.inf))


def newton_solve(
    z0: np.ndarray,
    *,
    tol: float = 1.0e-8,
    max_iter: int = 12,
) -> SolveReport:
    z = np.asarray(z0, dtype=float).copy()
    for it in range(max_iter):
        r = plateau_residual(z)
        if float(np.linalg.norm(r, ord=np.inf)) <= tol:
            return SolveReport(
                method="newton",
                converged=True,
                iterations=it,
                residual_inf=residual_inf(z),
                state=z.tolist(),
                accepted_steps=it,
                note="Plain Newton reached the root.",
            )
        try:
            dz = np.linalg.solve(plateau_jacobian(z), -r)
        except np.linalg.LinAlgError as exc:
            return SolveReport(
                method="newton",
                converged=False,
                iterations=it,
                residual_inf=residual_inf(z),
                state=z.tolist(),
                accepted_steps=it,
                note=f"Linear solve failed: {exc}",
            )
        z = z + dz
        if not np.all(np.isfinite(z)):
            return SolveReport(
                method="newton",
                converged=False,
                iterations=it + 1,
                residual_inf=float("inf"),
                state=[float("nan"), float("nan")],
                accepted_steps=it + 1,
                note="Newton produced a non-finite state.",
            )
    return SolveReport(
        method="newton",
        converged=False,
        iterations=max_iter,
        residual_inf=residual_inf(z),
        state=z.tolist(),
        accepted_steps=max_iter,
        note="Newton diverged on the plateau system.",
    )


def _ptc_mass_operator(kind: str) -> np.ndarray:
    if kind == "identity":
        return np.eye(2, dtype=float)
    if kind == "root":
        return plateau_root_linearization()
    raise ValueError(f"Unsupported PTC mass operator: {kind}")


def ptc_solve(
    z0: np.ndarray,
    *,
    mass: str = "identity",
    sign: float = 1.0,
    sigma0: float = 1.0e-1,
    sigma_max: float = 1.0e8,
    growth: float = 3.0,
    decay: float = 0.5,
    tol: float = 1.0e-8,
    max_iter: int = 50,
) -> SolveReport:
    z = np.asarray(z0, dtype=float).copy()
    sigma = float(sigma0)
    accepted = 0
    M = _ptc_mass_operator(str(mass))
    for it in range(max_iter):
        r = plateau_residual(z)
        r_inf = float(np.linalg.norm(r, ord=np.inf))
        if r_inf <= tol:
            return SolveReport(
                method=f"ptc[{mass}]",
                converged=True,
                iterations=it,
                residual_inf=r_inf,
                state=z.tolist(),
                sigma_final=float(sigma),
                accepted_steps=accepted,
                note="PTC reached the root.",
            )
        try:
            dz = np.linalg.solve(plateau_jacobian(z) + float(sign) * sigma * M, -r)
        except np.linalg.LinAlgError as exc:
            return SolveReport(
                method=f"ptc[{mass}]",
                converged=False,
                iterations=it,
                residual_inf=r_inf,
                state=z.tolist(),
                sigma_final=float(sigma),
                accepted_steps=accepted,
                note=f"Linear solve failed: {exc}",
            )
        z_trial = z + dz
        trial_inf = residual_inf(z_trial)
        if np.isfinite(trial_inf) and trial_inf < r_inf:
            z = z_trial
            sigma = max(float(decay) * sigma, 1.0e-12)
            accepted += 1
        else:
            sigma = min(float(growth) * sigma, float(sigma_max))
    return SolveReport(
        method=f"ptc[{mass}]",
        converged=False,
        iterations=max_iter,
        residual_inf=residual_inf(z),
        state=z.tolist(),
        sigma_final=float(sigma),
        accepted_steps=accepted,
        note="PTC did not contract to the requested tolerance.",
    )


def plateau_fixed_point_map(z: np.ndarray, *, omega: float = 1.0) -> np.ndarray:
    Ainv = np.linalg.inv(plateau_root_linearization())
    return np.asarray(z, dtype=float).ravel() - float(omega) * (Ainv @ plateau_residual(z))


def fixed_point_solve(
    z0: np.ndarray,
    *,
    omega: float = 1.0,
    tol: float = 1.0e-8,
    max_iter: int = 50,
) -> SolveReport:
    z = np.asarray(z0, dtype=float).copy()
    for it in range(max_iter):
        r_inf = residual_inf(z)
        if r_inf <= tol:
            return SolveReport(
                method=f"fixed-point[w={omega:g}]",
                converged=True,
                iterations=it,
                residual_inf=r_inf,
                state=z.tolist(),
                accepted_steps=it,
                note="Fixed-point iteration reached the root.",
            )
        z = plateau_fixed_point_map(z, omega=float(omega))
    return SolveReport(
        method=f"fixed-point[w={omega:g}]",
        converged=False,
        iterations=max_iter,
        residual_inf=residual_inf(z),
        state=z.tolist(),
        accepted_steps=max_iter,
        note="Fixed-point iteration stalled before the requested tolerance.",
    )


def _solve_anderson_weights(F: np.ndarray, reg: float) -> np.ndarray | None:
    m = int(F.shape[1])
    if m <= 1:
        return None
    kkt = np.zeros((m + 1, m + 1), dtype=float)
    kkt[:m, :m] = np.asarray(F.T @ F, dtype=float)
    if reg > 0.0:
        kkt[:m, :m] += float(reg) * np.eye(m, dtype=float)
    kkt[:m, m] = 1.0
    kkt[m, :m] = 1.0
    rhs = np.zeros((m + 1,), dtype=float)
    rhs[m] = 1.0
    try:
        sol = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError:
        return None
    alpha = np.asarray(sol[:m], dtype=float)
    if not np.all(np.isfinite(alpha)):
        return None
    return alpha


def anderson_solve(
    z0: np.ndarray,
    *,
    omega: float = 0.5,
    history: int = 4,
    damping: float = 1.0,
    regularization: float = 1.0e-10,
    mode: str = "state",
    tol: float = 1.0e-8,
    max_iter: int = 50,
) -> SolveReport:
    z = np.asarray(z0, dtype=float).copy()
    pairs: list[dict[str, np.ndarray]] = []
    accepted = 0
    for it in range(max_iter):
        r_inf = residual_inf(z)
        if r_inf <= tol:
            return SolveReport(
                method=f"anderson[{mode},w={omega:g}]",
                converged=True,
                iterations=it,
                residual_inf=r_inf,
                state=z.tolist(),
                accepted_steps=accepted,
                note="Anderson-accelerated fixed-point iteration reached the root.",
            )
        g = plateau_fixed_point_map(z, omega=float(omega))
        z_next = g.copy()
        trial_pairs = pairs[-max(1, int(history) - 1) :] + [{"z": z.copy(), "g": g.copy()}]
        if len(trial_pairs) >= 2:
            if str(mode) == "state":
                G = np.column_stack([item["g"] for item in trial_pairs])
                F = np.column_stack([item["g"] - item["z"] for item in trial_pairs])
                alpha = _solve_anderson_weights(F, float(regularization))
                if alpha is not None:
                    z_mix = np.asarray(G @ alpha, dtype=float).ravel()
                    z_mix = z + float(damping) * (z_mix - z)
                    if np.isfinite(residual_inf(z_mix)) and residual_inf(z_mix) < r_inf:
                        z_next = z_mix
            elif str(mode) == "increment":
                D = np.column_stack([item["g"] - item["z"] for item in trial_pairs])
                alpha = _solve_anderson_weights(D, float(regularization))
                if alpha is not None:
                    d_mix = np.asarray(D @ alpha, dtype=float).ravel()
                    z_mix = z + float(damping) * d_mix
                    if np.isfinite(residual_inf(z_mix)) and residual_inf(z_mix) < r_inf:
                        z_next = z_mix
            else:
                raise ValueError(f"Unsupported Anderson mode: {mode}")
        pairs.append({"z": z.copy(), "g": g.copy()})
        pairs = pairs[-max(1, int(history)) :]
        z = z_next
        accepted += 1
    return SolveReport(
        method=f"anderson[{mode},w={omega:g}]",
        converged=False,
        iterations=max_iter,
        residual_inf=residual_inf(z),
        state=z.tolist(),
        accepted_steps=accepted,
        note="Anderson iteration stalled before the requested tolerance.",
    )


def run_suite() -> list[SolveReport]:
    z0 = np.array([3.0, 3.0], dtype=float)
    return [
        newton_solve(z0),
        ptc_solve(z0, mass="identity", sign=1.0),
        ptc_solve(z0, mass="root", sign=1.0),
        ptc_solve(z0, mass="root", sign=-1.0),
        fixed_point_solve(z0, omega=0.5),
        anderson_solve(z0, omega=0.5, mode="state"),
        anderson_solve(z0, omega=0.5, mode="increment"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Toy benchmark for Newton, PTC, and Anderson on an arctangent plateau system.")
    ap.add_argument("--json", action="store_true", help="Print the benchmark summary as JSON.")
    args = ap.parse_args()

    reports = run_suite()
    if args.json:
        print(json.dumps([asdict(r) for r in reports], indent=2))
        return

    for report in reports:
        status = "ok" if report.converged else "fail"
        sigma_msg = "" if report.sigma_final is None else f" sigma={report.sigma_final:.2e}"
        acc_msg = "" if report.accepted_steps is None else f" accepted={report.accepted_steps}"
        print(
            f"{report.method:24s} {status:4s} it={report.iterations:2d} "
            f"|R|_inf={report.residual_inf:.3e} z=({report.state[0]:.6e}, {report.state[1]:.6e})"
            f"{sigma_msg}{acc_msg}"
        )
        if report.note:
            print(f"  note: {report.note}")


if __name__ == "__main__":
    main()
