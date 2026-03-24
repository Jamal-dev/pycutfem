from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class SolveResult:
    converged: bool
    state: np.ndarray
    residual_history: list[float]


def exact_residual(x: np.ndarray, *, coupling: float) -> np.ndarray:
    u, v, w = np.asarray(x, dtype=float)
    return np.array(
        [
            np.arctan(u + 2.0 * v + coupling * w),
            np.arctan(2.0 * u - v + coupling * w),
            np.arctan(coupling * u + v + w),
        ],
        dtype=float,
    )


def exact_jacobian(x: np.ndarray, *, coupling: float) -> np.ndarray:
    u, v, w = np.asarray(x, dtype=float)
    a = 1.0 / (1.0 + (u + 2.0 * v + coupling * w) ** 2)
    b = 1.0 / (1.0 + (2.0 * u - v + coupling * w) ** 2)
    d = 1.0 / (1.0 + (coupling * u + v + w) ** 2)
    return np.array(
        [
            [a, 2.0 * a, coupling * a],
            [2.0 * b, -b, coupling * b],
            [coupling * d, d, d],
        ],
        dtype=float,
    )


def _stage_newton(
    residual,
    jacobian,
    x0: np.ndarray,
    *,
    max_it: int,
    accept_on_improve: bool,
) -> SolveResult:
    x = np.asarray(x0, dtype=float).copy()
    hist: list[float] = []
    best_x = x.copy()
    best_n = float(np.linalg.norm(residual(x), ord=np.inf))
    for _ in range(max(1, int(max_it))):
        r = np.asarray(residual(x), dtype=float)
        nrm = float(np.linalg.norm(r, ord=np.inf))
        hist.append(nrm)
        if nrm < best_n:
            best_n = nrm
            best_x = x.copy()
        if nrm <= 1.0e-10:
            return SolveResult(True, x, hist)
        try:
            dx = np.linalg.solve(np.asarray(jacobian(x), dtype=float), -r)
        except Exception:
            return SolveResult(bool(accept_on_improve and best_n < hist[0]), best_x, hist)
        alpha = 1.0
        while alpha > 1.0e-10:
            trial = x + alpha * dx
            if float(np.linalg.norm(residual(trial), ord=np.inf)) < nrm:
                x = trial
                break
            alpha *= 0.5
        else:
            return SolveResult(bool(accept_on_improve and best_n < hist[0]), best_x, hist)
    if accept_on_improve and best_n < hist[0]:
        return SolveResult(True, best_x, hist)
    return SolveResult(False, x, hist)


def p1_predictor(
    x0: np.ndarray,
    *,
    coupling: float,
    frozen_w: float,
    max_it: int,
) -> SolveResult:
    def residual(y: np.ndarray) -> np.ndarray:
        u, v = np.asarray(y, dtype=float)
        return np.array(
            [
                np.arctan(u + 2.0 * v + coupling * frozen_w),
                np.arctan(2.0 * u - v + coupling * frozen_w),
            ],
            dtype=float,
        )

    def jacobian(y: np.ndarray) -> np.ndarray:
        u, v = np.asarray(y, dtype=float)
        a = 1.0 / (1.0 + (u + 2.0 * v + coupling * frozen_w) ** 2)
        b = 1.0 / (1.0 + (2.0 * u - v + coupling * frozen_w) ** 2)
        return np.array([[a, 2.0 * a], [2.0 * b, -b]], dtype=float)

    res = _stage_newton(residual, jacobian, np.asarray(x0[:2], dtype=float), max_it=max_it, accept_on_improve=True)
    state = np.array([res.state[0], res.state[1], float(frozen_w)], dtype=float)
    return SolveResult(res.converged, state, res.residual_history)


def p2_continuation(
    x0: np.ndarray,
    *,
    coupling: float,
    frozen_w: float,
    lambda_steps: int,
    max_it_per_lambda: int,
) -> SolveResult:
    x = np.asarray(x0, dtype=float).copy()
    full_hist: list[float] = []
    for i in range(1, max(1, int(lambda_steps)) + 1):
        lam = float(i) / float(max(1, int(lambda_steps)))

        def residual(z: np.ndarray, lam_local: float = lam) -> np.ndarray:
            u, v, w = np.asarray(z, dtype=float)
            w_eff = (1.0 - lam_local) * frozen_w + lam_local * w
            return np.array(
                [
                    np.arctan(u + 2.0 * v + coupling * w_eff),
                    np.arctan(2.0 * u - v + coupling * w_eff),
                    (1.0 - lam_local) * (w - frozen_w) + lam_local * np.arctan(coupling * u + v + w),
                ],
                dtype=float,
            )

        def jacobian(z: np.ndarray, lam_local: float = lam) -> np.ndarray:
            u, v, w = np.asarray(z, dtype=float)
            w_eff = (1.0 - lam_local) * frozen_w + lam_local * w
            a = 1.0 / (1.0 + (u + 2.0 * v + coupling * w_eff) ** 2)
            b = 1.0 / (1.0 + (2.0 * u - v + coupling * w_eff) ** 2)
            d = 1.0 / (1.0 + (coupling * u + v + w) ** 2)
            return np.array(
                [
                    [a, 2.0 * a, coupling * lam_local * a],
                    [2.0 * b, -b, coupling * lam_local * b],
                    [lam_local * coupling * d, lam_local * d, (1.0 - lam_local) + lam_local * d],
                ],
                dtype=float,
            )

        stage = _stage_newton(
            residual,
            jacobian,
            x,
            max_it=max_it_per_lambda,
            accept_on_improve=True,
        )
        full_hist.extend(stage.residual_history)
        if not stage.converged:
            return SolveResult(False, stage.state, full_hist)
        x = stage.state.copy()
    return SolveResult(True, x, full_hist)


def run_current_design(
    *,
    coupling: float = 6.0,
    x0: tuple[float, float, float] = (3.0, 3.0, 3.0),
    p1_max_it: int = 6,
    exact_max_it: int = 6,
) -> dict[str, object]:
    p1 = p1_predictor(np.asarray(x0, dtype=float), coupling=coupling, frozen_w=float(x0[2]), max_it=p1_max_it)
    exact = _stage_newton(
        lambda z: exact_residual(z, coupling=coupling),
        lambda z: exact_jacobian(z, coupling=coupling),
        p1.state,
        max_it=exact_max_it,
        accept_on_improve=False,
    )
    return {
        "p1": p1,
        "exact": exact,
        "final_residual_inf": float(np.linalg.norm(exact_residual(exact.state, coupling=coupling), ord=np.inf)),
    }


def run_predictor_corrector_p2(
    *,
    coupling: float = 6.0,
    x0: tuple[float, float, float] = (3.0, 3.0, 3.0),
    p1_max_it: int = 6,
    p2_lambda_steps: int = 6,
    p2_max_it_per_lambda: int = 2,
    exact_max_it: int = 6,
) -> dict[str, object]:
    p1 = p1_predictor(np.asarray(x0, dtype=float), coupling=coupling, frozen_w=float(x0[2]), max_it=p1_max_it)
    p2 = p2_continuation(
        p1.state,
        coupling=coupling,
        frozen_w=float(x0[2]),
        lambda_steps=p2_lambda_steps,
        max_it_per_lambda=p2_max_it_per_lambda,
    )
    exact = _stage_newton(
        lambda z: exact_residual(z, coupling=coupling),
        lambda z: exact_jacobian(z, coupling=coupling),
        p2.state,
        max_it=exact_max_it,
        accept_on_improve=False,
    )
    return {
        "p1": p1,
        "p2": p2,
        "exact": exact,
        "final_residual_inf": float(np.linalg.norm(exact_residual(exact.state, coupling=coupling), ord=np.inf)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Toy predictor-corrector P2 continuation demo.")
    ap.add_argument("--coupling", type=float, default=6.0)
    args = ap.parse_args()
    current = run_current_design(coupling=float(args.coupling))
    staged = run_predictor_corrector_p2(coupling=float(args.coupling))
    print(
        f"current_design: converged={int(bool(current['exact'].converged))}, "
        f"final_residual_inf={float(current['final_residual_inf']):.3e}, "
        f"history={list(current['exact'].residual_history)}"
    )
    print(
        f"predictor_corrector_p2: converged={int(bool(staged['exact'].converged))}, "
        f"final_residual_inf={float(staged['final_residual_inf']):.3e}, "
        f"p2_hist_len={len(staged['p2'].residual_history)}, "
        f"exact_history={list(staged['exact'].residual_history)}"
    )


if __name__ == "__main__":
    main()
