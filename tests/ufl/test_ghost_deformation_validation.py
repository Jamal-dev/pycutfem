import numpy as np

from scripts.validate_ghost_deformation import run_validation


def test_deformation_validation_matches_reference():
    results = run_validation(log=False)

    for key in ("scalar_bilinear", "scalar_rhs", "vector_bilinear", "vector_rhs"):
        stats = results[key]
        assert stats["diff_K"] < 1e-12, f"{key} matrix mismatch too large"
        assert stats["diff_F"] < 1e-12, f"{key} rhs mismatch too large"

    analytic = results["analytic"]
    assert analytic["err_def"] < 1e-12
    assert analytic["err_def"] < analytic["err_no"]
