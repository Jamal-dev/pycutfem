from examples.debug.predictor_corrector_p2_toy import (
    run_current_design,
    run_predictor_corrector_p2,
)


def test_predictor_corrector_p2_toy_beats_direct_corrector():
    current = run_current_design(coupling=6.0, p1_max_it=6, exact_max_it=6)
    staged = run_predictor_corrector_p2(
        coupling=6.0,
        p1_max_it=6,
        p2_lambda_steps=6,
        p2_max_it_per_lambda=2,
        exact_max_it=6,
    )

    assert not bool(current["exact"].converged)
    assert bool(staged["p2"].converged)
    assert bool(staged["exact"].converged)
    assert float(staged["final_residual_inf"]) < 1.0e-8
    assert float(staged["final_residual_inf"]) < float(current["final_residual_inf"])
