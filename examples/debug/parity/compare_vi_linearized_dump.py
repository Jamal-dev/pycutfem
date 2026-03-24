from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import scipy.sparse as sp  # type: ignore
import scipy.sparse.linalg as spla  # type: ignore


def _linf(v: np.ndarray) -> float:
    arr = np.asarray(v, dtype=float).ravel()
    return float(np.linalg.norm(arr, ord=np.inf)) if arr.size else 0.0


def _load_dump(dump_dir: Path, stem: str) -> dict[str, object]:
    A = sp.load_npz(dump_dir / f"{stem}_A_red.npz").tocsr()
    return {
        "A": A,
        "x0": np.asarray(np.load(dump_dir / f"{stem}_x_red.npy"), dtype=float).ravel(),
        "lo": np.asarray(np.load(dump_dir / f"{stem}_lo_red.npy"), dtype=float).ravel(),
        "hi": np.asarray(np.load(dump_dir / f"{stem}_hi_red.npy"), dtype=float).ravel(),
        "c": np.asarray(np.load(dump_dir / f"{stem}_c_red.npy"), dtype=float).ravel(),
        "stat0": np.asarray(np.load(dump_dir / f"{stem}_stat_red.npy"), dtype=float).ravel(),
        "b_eff": np.asarray(np.load(dump_dir / f"{stem}_eq_b_eff.npy"), dtype=float).ravel(),
        "lambda0": np.asarray(np.load(dump_dir / f"{stem}_eq_lambda.npy"), dtype=float).ravel(),
        "B": sp.csr_matrix(np.asarray(np.load(dump_dir / f"{stem}_eq_B_red.npy"), dtype=float)),
    }


def _state_metrics(
    *,
    A: sp.csr_matrix,
    B: sp.csr_matrix,
    rhs_base: np.ndarray,
    b_eff: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    y: np.ndarray,
    lam: np.ndarray,
    state: np.ndarray,
) -> dict[str, float]:
    stationarity = np.asarray(A @ y, dtype=float).ravel() - rhs_base + np.asarray(B.T @ lam, dtype=float).ravel()
    eq_res = np.asarray(B @ y, dtype=float).ravel() - b_eff
    G = stationarity.copy()
    act_lo = state == 1
    act_hi = state == -1
    G[act_lo] = y[act_lo] - lo[act_lo]
    G[act_hi] = y[act_hi] - hi[act_hi]
    inactive = ~(act_lo | act_hi)
    gap = np.zeros_like(y)
    gap[act_lo] = y[act_lo] - lo[act_lo]
    gap[act_hi] = y[act_hi] - hi[act_hi]
    return {
        "g_inf": _linf(G),
        "inactive_res_inf": _linf(stationarity[inactive]),
        "active_gap_inf": _linf(gap[act_lo | act_hi]),
        "equality_inf": _linf(eq_res),
    }


def _predict_state(
    *,
    y: np.ndarray,
    lam: np.ndarray,
    A: sp.csr_matrix,
    B: sp.csr_matrix,
    rhs_base: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    stationarity = np.asarray(A @ y, dtype=float).ravel() - rhs_base + np.asarray(B.T @ lam, dtype=float).ravel()
    state = np.zeros(y.shape, dtype=np.int8)
    lo_f = np.isfinite(lo)
    hi_f = np.isfinite(hi)
    ind_lo = stationarity - c * (y - lo)
    ind_hi = stationarity + c * (hi - y)
    state[lo_f & (ind_lo > 0.0)] = 1
    state[hi_f & (ind_hi < 0.0)] = -1
    return state


def _build_augmented_matrix(
    *,
    A: sp.csr_matrix,
    B: sp.csr_matrix,
    rhs_base: np.ndarray,
    state: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    b_eff: np.ndarray,
    shift_factor: float,
) -> tuple[sp.csr_matrix, np.ndarray]:
    n = A.shape[0]
    m = B.shape[0]
    A_work = A.tolil(copy=True)
    rhs = rhs_base.copy()
    active = np.asarray(state != 0, dtype=bool)
    for i in np.flatnonzero(active).tolist():
        A_work.rows[i] = [int(i)]
        A_work.data[i] = [1.0]
        rhs[i] = float(lo[i] if state[i] == 1 else hi[i])
    A_work = A_work.tocsr()
    if shift_factor > 0.0:
        diag = np.abs(np.asarray(A.diagonal(), dtype=float).ravel())
        scale = np.where(np.isfinite(diag) & (diag > 1.0e-12), diag, 1.0e-12)
        add = np.zeros((n,), dtype=float)
        add[~active] = shift_factor * scale[~active]
        A_work = A_work + sp.diags(add, format="csr")
    B_top = B.T.tolil(copy=True)
    for i in np.flatnonzero(active).tolist():
        B_top.rows[i] = []
        B_top.data[i] = []
    B_top = B_top.tocsr()
    top = sp.hstack([A_work, B_top], format="csr")
    bot = sp.hstack([B, sp.csr_matrix((m, m), dtype=float)], format="csr")
    M = sp.vstack([top, bot], format="csr")
    rhs_aug = np.concatenate([rhs, b_eff], dtype=float)
    return M, rhs_aug


def _python_replay(data: dict[str, object]) -> dict[str, object]:
    A = data["A"]
    B = data["B"]
    x0 = np.asarray(data["x0"], dtype=float)
    lo = np.asarray(data["lo"], dtype=float)
    hi = np.asarray(data["hi"], dtype=float)
    c = np.asarray(data["c"], dtype=float)
    stat0 = np.asarray(data["stat0"], dtype=float)
    b_eff = np.asarray(data["b_eff"], dtype=float)
    lambda0 = np.asarray(data["lambda0"], dtype=float)
    rhs_base = np.asarray(A @ x0, dtype=float).ravel() - stat0

    y = x0.copy()
    lam = lambda0.copy()
    prev_state = np.full(y.shape, 7, dtype=np.int8)
    history: list[dict[str, object]] = []
    shift_schedule = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)
    converged = False

    for it in range(1, 51):
        state = _predict_state(y=y, lam=lam, A=A, B=B, rhs_base=rhs_base, lo=lo, hi=hi, c=c)
        delta_active = int(np.count_nonzero(state != prev_state))
        used_shift = None
        lin_res_inf = float("inf")
        sol = None
        for shift in shift_schedule:
            M, rhs_aug = _build_augmented_matrix(
                A=A,
                B=B,
                rhs_base=rhs_base,
                state=state,
                lo=lo,
                hi=hi,
                b_eff=b_eff,
                shift_factor=float(shift),
            )
            try:
                trial = spla.spsolve(M, rhs_aug)
            except Exception:
                continue
            if not np.all(np.isfinite(trial)):
                continue
            res = np.asarray(M @ trial, dtype=float).ravel() - rhs_aug
            used_shift = float(shift)
            lin_res_inf = _linf(res)
            sol = np.asarray(trial, dtype=float).ravel()
            break
        if sol is None:
            raise RuntimeError(f"Python replay failed to solve iteration {it}.")
        y = sol[: A.shape[0]].copy()
        lam = sol[A.shape[0] :].copy()
        metrics = _state_metrics(A=A, B=B, rhs_base=rhs_base, b_eff=b_eff, lo=lo, hi=hi, y=y, lam=lam, state=state)
        history.append(
            {
                "iter": it,
                "n_active_lo": int(np.count_nonzero(state == 1)),
                "n_active_hi": int(np.count_nonzero(state == -1)),
                "delta_active": delta_active,
                "shift_factor": used_shift,
                "linear_res_inf": lin_res_inf,
                **metrics,
            }
        )
        if np.array_equal(state, prev_state):
            converged = True
            break
        prev_state = state.copy()

    return {
        "converged": converged,
        "iterations": len(history),
        "y_red": y.tolist(),
        "lambda_red": lam.tolist(),
        "history": history,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare the dumped reduced Benchmark 7 VI state between deal.II and a Python replay.")
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--stem", type=str, default="step0001_it01")
    ap.add_argument("--dealii-bin", type=Path, default=Path("/tmp/vi_linearized_dump_dealii_build/vi_linearized_dump_dealii"))
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    stem = str(args.stem)
    out_json = Path(args.out_json)
    out_dir = out_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    dealii_dir = out_dir / "dealii_input"
    subprocess.run(
        [
            "python",
            "examples/debug/parity/prepare_vi_dump_for_dealii.py",
            "--dump-dir",
            str(dump_dir),
            "--stem",
            stem,
            "--out-dir",
            str(dealii_dir),
        ],
        check=True,
    )
    dealii_summary = dealii_dir / "summary.json"
    dealii_env = dict(os.environ)
    dealii_env.setdefault("PYCUTFEM_VI_DEALII_TRACE", "1")
    proc = subprocess.run(
        [str(args.dealii_bin), str(dealii_dir), str(dealii_summary)],
        check=True,
        capture_output=True,
        text=True,
        env=dealii_env,
    )
    (dealii_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (dealii_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")

    data = _load_dump(dump_dir, stem)
    py = _python_replay(data)
    dealii_info = json.loads(dealii_summary.read_text(encoding="utf-8"))
    dealii_history = json.loads((dealii_dir / "history.json").read_text(encoding="utf-8"))
    y_dealii = np.loadtxt(dealii_dir / "y_red.txt", skiprows=1)
    lam_dealii = np.loadtxt(dealii_dir / "lambda_red.txt", skiprows=1)

    y_py = np.asarray(py["y_red"], dtype=float)
    lam_py = np.asarray(py["lambda_red"], dtype=float)
    comparison = {
        "python": {
            "converged": bool(py["converged"]),
            "iterations": int(py["iterations"]),
            "history": py["history"],
        },
        "dealii": {
            **dealii_info,
            "history": dealii_history,
        },
        "diff": {
            "y_linf": _linf(y_py - y_dealii),
            "lambda_linf": _linf(lam_py - lam_dealii),
            "history_len_python": int(len(py["history"])),
            "history_len_dealii": int(len(dealii_history)),
        },
    }
    out_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"[done] wrote {out_json}")


if __name__ == "__main__":
    main()
