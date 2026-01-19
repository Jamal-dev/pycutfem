import json
from pathlib import Path

import examples.turek_cylinder.optimize_params as opt


def test_tuned_params_updates_only_on_strict_improvement(tmp_path: Path):
    tuned_path = tmp_path / "tuned_params.json"
    meta = {
        "benchmark": "2d-2",
        "level": 1,
        "dt": 0.1,
        "theta": 0.5,
        "ghost_measure": "patch",
        "with_deformation": False,
        "fe_order": 2,
        "p_order": 1,
    }

    original = {
        "version": 1,
        "entries": [
            {
                "benchmark": meta["benchmark"],
                "level": meta["level"],
                "dt": meta["dt"],
                "theta": meta["theta"],
                "ghost_measure": meta["ghost_measure"],
                "with_deformation": meta["with_deformation"],
                "fe_order": meta["fe_order"],
                "p_order": meta["p_order"],
                "beta0": 30.0,
                "gamma_gp": 0.01,
                "gamma_gp_hess": 0.0,
                "score": 1.0,
                "n_done": 10,
                "ts": 0,
            }
        ],
    }
    tuned_path.write_text(json.dumps(original, indent=2, sort_keys=True) + "\n")
    before_text = tuned_path.read_text()

    # Worse score -> no update (file content stays identical).
    worse = opt.RunResult(
        candidate=opt.Candidate(beta0=20.0, gamma_gp=0.03, gamma_gp_p=None, gamma_gp_hess=0.0),
        budget_steps=10,
        exit_code=0,
        n_done=10,
        score=2.0,
        score_fail=0.0,
        score_ref=2.0,
        functionals_csv=tmp_path / "functionals.csv",
        run_dir=tmp_path,
    )
    opt._update_tuned_params(worse, tuned_path=tuned_path, meta=meta)
    assert tuned_path.read_text() == before_text

    # Strictly better score -> update.
    better = opt.RunResult(
        candidate=opt.Candidate(beta0=20.0, gamma_gp=0.03, gamma_gp_p=None, gamma_gp_hess=0.0),
        budget_steps=10,
        exit_code=0,
        n_done=10,
        score=0.5,
        score_fail=0.0,
        score_ref=0.5,
        functionals_csv=tmp_path / "functionals.csv",
        run_dir=tmp_path,
    )
    opt._update_tuned_params(better, tuned_path=tuned_path, meta=meta)
    doc = json.loads(tuned_path.read_text())
    assert isinstance(doc.get("entries"), list) and len(doc["entries"]) == 1
    entry = doc["entries"][0]
    assert entry["score"] == 0.5
    assert entry["beta0"] == 20.0
    assert entry["gamma_gp"] == 0.03
