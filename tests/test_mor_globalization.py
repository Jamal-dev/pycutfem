from __future__ import annotations

import numpy as np
import pytest

from dataclasses import dataclass

from pycutfem.mor import (
    BranchGlobalizationSpec,
    clip_step_to_trust_region,
    solve_with_branch_backtracking,
    step_alpha_to_branch_radius,
)


@dataclass
class _FakeSolveResult:
    converged: bool
    residual_norm: float
    timing_counters: dict[str, float]


def test_branch_globalization_uses_decoded_state_distance_and_merit() -> None:
    basis = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.5]], dtype=float)
    spec = BranchGlobalizationSpec(
        reference_coefficients=np.array([1.0, -1.0], dtype=float),
        trial_basis=basis,
        max_reference_distance=0.25,
        state_merit_weight=2.0,
        require_residual_convergence=True,
    )

    q = np.array([1.1, -0.95], dtype=float)
    distance = np.linalg.norm(basis @ (q - spec.reference_coefficients))
    assert spec.decoded_distance(q) == pytest.approx(distance)
    assert spec.within_radius(q)
    assert spec.merit(3.0, q) > 3.0
    native = spec.to_native_options()
    np.testing.assert_allclose(native["reference_coefficients"], spec.reference_coefficients)
    assert native["require_residual_convergence"] is True


def test_clip_step_to_trust_region() -> None:
    clipped, changed = clip_step_to_trust_region(np.array([3.0, 4.0], dtype=float), 2.5)
    assert changed is True
    assert np.linalg.norm(clipped) == pytest.approx(2.5)

    same, changed = clip_step_to_trust_region(np.array([0.1, 0.2], dtype=float), 2.5)
    assert changed is False
    np.testing.assert_allclose(same, np.array([0.1, 0.2]))


def test_step_alpha_to_branch_radius() -> None:
    spec = BranchGlobalizationSpec(
        reference_coefficients=np.array([0.0], dtype=float),
        trial_basis=np.array([[2.0]], dtype=float),
        max_reference_distance=1.0,
    )
    alpha = step_alpha_to_branch_radius(np.array([0.0]), np.array([2.0]), spec)
    assert alpha == pytest.approx(0.25)


def test_solve_with_branch_backtracking_retries_until_acceptance() -> None:
    calls: list[dict[str, float | None]] = []

    def solve_once(**options):
        calls.append(dict(options))
        radius = options.get("max_reference_distance")
        ok = radius == pytest.approx(0.25)
        return _FakeSolveResult(
            converged=bool(ok),
            residual_norm=0.0 if ok else 10.0,
            timing_counters={"final_branch_distance": float("nan") if radius is None else float(radius)},
        )

    result = solve_with_branch_backtracking(
        solve_once,
        branch_radii=(1.0, 0.25, 0.1),
        trust_radii=(None,),
    )

    assert result.accepted is True
    assert len(result.attempts) == 2
    assert result.attempts[-1].accepted is True
    assert calls[1]["max_reference_distance"] == 0.25
