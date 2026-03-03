"""
Duddu et al. (2007) growth-only model: 1D finite-difference check (Table I).

This script solves the quasi-steady 1D substrate diffusion problem on y∈[0,H]
with a slab biofilm on y∈[0,h_b] and a fluid boundary layer on y∈[h_b,H]:

  Biofilm (0<h<h_b):   D_b S'' - f * m_S(S) = 0
  Fluid   (h_b<h<H):   D_f S''              = 0

with boundary/interface conditions:
  S'(0)=0,  S(H)=Sbar,  S continuous at h_b,  D_b S'(h_b-) = D_f S'(h_b+).

Given S, it computes the interface normal speed (slab case) as:
  F = ∫_0^{h_b} div(U)(S) dy,  where div(U)=f(m_x+m_w).

The paper reports FD results for N={500,2000,2500,3000} line elements (Table I).
In Duddu (2007), the Dirichlet substrate boundary is 0.1mm above the interface,
so we interpret N as the number of uniform elements *in the biofilm thickness*
(0≤y≤h_b) and use the same spacing in the fluid layer (h_b≤y≤H). This keeps the
interface aligned with a grid node for all N in Table I.

We implement a Newton solve for S on that uniform grid with a second-order
one-sided flux continuity condition at the interface node.

Outputs are written under examples/biofilms/benchmarks/dadu/results/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class Duddu2007Params:
    f_active: float = 0.5

    # Kinetics (see Duddu 2009 Appendix Table A.1; Duddu 2007 refers to Chopp 2002/2003)
    rho_x: float = 1.0250  # mg VS / mm^3
    rho_w: float = 1.0125  # mg VS / mm^3
    Y_xO: float = 0.583
    # Duddu (2007) Table I aligns with a smaller EPS yield than the value later
    # reported in Duddu (2009) Appendix Table A.1. Override with --Y-wO if needed.
    Y_wO: float = 0.215
    qhat0: float = 8.0  # mg O2 / (mg VS day)
    K0: float = 5.0e-7  # mg O2 / mm^3
    b: float = 0.3  # 1/day
    f_D: float = 0.8
    g: float = 1.42  # mg O2 / mg VS

    # Diffusion (mm^2/day)
    Db: float = 146.88
    Df: float = 183.6

    # Geometry (mm)
    H: float = 0.3
    h_b: float = 0.2

    # Bulk substrate (mgO2/mm^3)
    Sbar: float = 8.3e-6


def _monod(S: np.ndarray, K0: float) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    return S / (K0 + S)


def _mS(S: np.ndarray, p: Duddu2007Params) -> np.ndarray:
    # Positive substrate utilization rate per unit volume (paper uses different symbols/signs).
    return p.rho_x * (p.qhat0 + p.g * p.f_D * p.b) * _monod(S, p.K0)


def _dmS_dS(S: np.ndarray, p: Duddu2007Params) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    K0 = float(p.K0)
    return p.rho_x * (p.qhat0 + p.g * p.f_D * p.b) * (K0 / (K0 + S) ** 2)


def _divU(S: np.ndarray, p: Duddu2007Params) -> np.ndarray:
    mon = _monod(S, p.K0)
    mx = (p.Y_xO * p.qhat0 - p.b) * mon
    mw = (p.rho_x / p.rho_w) * ((1.0 - p.f_D) * p.b + p.Y_wO * p.qhat0) * mon
    return p.f_active * (mx + mw)


def solve_substrate_fd(
    *,
    N_biofilm: int,
    p: Duddu2007Params,
    tol: float,
    max_it: int,
    S_min: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Solve for S on a uniform grid with N_biofilm elements in [0,h_b].
    The same spacing is used in the fluid layer [h_b,H].
    Returns (y, S, iters).
    """
    N_b = int(N_biofilm)
    if N_b < 10:
        raise ValueError("N must be >= 10")

    H = float(p.H)
    hb = float(p.h_b)
    dy = hb / float(N_b)
    N_f = int(round((H - hb) / dy))
    if N_f < 1:
        raise ValueError("Need at least one fluid element.")
    N = int(N_b + N_f)
    # uniform spacing, interface at index N_b
    y = np.linspace(0.0, H, N + 1)
    i_int = int(N_b)

    # initial guess: linear in the fluid, tiny in the biofilm
    S = np.full(N + 1, float(p.Sbar), dtype=float)
    S[: i_int + 1] = float(S_min)

    Db = float(p.Db)
    Df = float(p.Df)

    for it in range(int(max_it)):
        r = np.zeros_like(S)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        # Bottom Neumann: S1 - S0 = 0
        r[0] = S[1] - S[0]
        rows += [0, 0]
        cols += [0, 1]
        data += [-1.0, 1.0]

        # Biofilm interior: Db*S'' - f*mS(S) = 0
        for i in range(1, i_int):
            Si = S[i]
            r[i] = Db * (S[i + 1] - 2.0 * Si + S[i - 1]) / (dy * dy) - float(p.f_active) * float(_mS(Si, p))
            rows += [i, i, i]
            cols += [i - 1, i, i + 1]
            data += [Db / (dy * dy), -2.0 * Db / (dy * dy) - float(p.f_active) * float(_dmS_dS(Si, p)), Db / (dy * dy)]

        # Interface flux continuity (second-order one-sided on both sides)
        i = i_int
        r[i] = (
            Db * (3.0 * S[i] - 4.0 * S[i - 1] + S[i - 2]) / (2.0 * dy)
            - Df * (-3.0 * S[i] + 4.0 * S[i + 1] - S[i + 2]) / (2.0 * dy)
        )
        rows += [i, i, i, i, i]
        cols += [i - 2, i - 1, i, i + 1, i + 2]
        data += [
            Db / (2.0 * dy),
            -2.0 * Db / dy,
            (3.0 * Db + 3.0 * Df) / (2.0 * dy),
            -2.0 * Df / dy,
            Df / (2.0 * dy),
        ]

        # Fluid interior: Df*S'' = 0
        for i in range(i_int + 1, N):
            r[i] = Df * (S[i + 1] - 2.0 * S[i] + S[i - 1]) / (dy * dy)
            rows += [i, i, i]
            cols += [i - 1, i, i + 1]
            data += [Df / (dy * dy), -2.0 * Df / (dy * dy), Df / (dy * dy)]

        # Top Dirichlet: S_N - Sbar = 0
        r[N] = S[N] - float(p.Sbar)
        rows.append(N)
        cols.append(N)
        data.append(1.0)

        res_norm = float(np.linalg.norm(r, ord=2))
        if res_norm <= float(tol):
            return y, S, it

        J = sp.csr_matrix((np.asarray(data, dtype=float), (np.asarray(rows), np.asarray(cols))), shape=(N + 1, N + 1))
        dS = spla.spsolve(J.tocsc(), -r)
        S = S + np.asarray(dS, dtype=float)
        if S_min > 0.0:
            S = np.maximum(S, float(S_min))

    raise RuntimeError(f"Newton did not converge in {max_it} iterations (last ||r||={res_norm:.3e}).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--N",
        type=int,
        default=3000,
        help="Number of uniform 1D elements in the biofilm layer (0<=y<=h_b). The fluid layer uses the same spacing.",
    )
    # With the clipping safeguard S>=S_min, Newton can stagnate around ~1e-8.
    ap.add_argument("--tol", type=float, default=1.0e-7)
    ap.add_argument("--max-it", type=int, default=50)
    ap.add_argument("--S-min", type=float, default=1.0e-16)
    ap.add_argument("--Y-xO", type=float, default=Duddu2007Params.Y_xO)
    ap.add_argument("--Y-wO", type=float, default=Duddu2007Params.Y_wO)
    ap.add_argument("--outdir", type=str, default="examples/biofilms/benchmarks/dadu/results/duddu2007_fd_1d_table1")
    args = ap.parse_args()

    p = Duddu2007Params(Y_xO=float(args.Y_xO), Y_wO=float(args.Y_wO))
    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    y, S, iters = solve_substrate_fd(N_biofilm=int(args.N), p=p, tol=float(args.tol), max_it=int(args.max_it), S_min=float(args.S_min))
    dy = float(p.h_b) / float(args.N)
    i_int = int(args.N)

    # Speed: integrate divU over the biofilm thickness (trapezoid rule)
    F = float(np.trapezoid(_divU(S[: i_int + 1], p), y[: i_int + 1]))

    np.savetxt(outdir / "profile_S.txt", np.column_stack([y, S]), header="y_mm  S_mgO2_per_mm3")
    # Phi profile (1D): Phi'' = divU in [0,h_b], Phi'(0)=0, Phi(h_b)=0, Phi=0 in (h_b,H].
    f = _divU(S[: i_int + 1], p)
    Phi_prime = np.zeros_like(f)
    Phi_bio = np.zeros_like(f)
    # cumulative trapezoid for Phi' and Phi
    Phi_prime[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * dy)
    Phi_bio[1:] = np.cumsum(0.5 * (Phi_prime[1:] + Phi_prime[:-1]) * dy)
    Phi_bio = Phi_bio - Phi_bio[-1]  # enforce Phi(h_b)=0
    Phi = np.zeros_like(S)
    Phi[: i_int + 1] = Phi_bio
    np.savetxt(outdir / "profile_Phi.txt", np.column_stack([y, Phi]), header="y_mm  Phi")

    summary = {
        "N_biofilm": int(args.N),
        "H_mm": float(p.H),
        "h_b_mm": float(p.h_b),
        "Y_xO": float(args.Y_xO),
        "Y_wO": float(args.Y_wO),
        "Db_mm2_per_day": float(p.Db),
        "Df_mm2_per_day": float(p.Df),
        "Sbar_mgO2_per_mm3": float(p.Sbar),
        "F_mm_per_day": float(F),
        "newton_iterations": int(iters),
        "tol": float(args.tol),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Duddu2007 FD 1D (N_biofilm={int(args.N)})")
    print(f"- Newton iterations: {iters}")
    print(f"- Estimated interface speed F ≈ {F:.6g} mm/day")
    print("  Table I (paper): FD3000 ~ 0.0103 mm/day")
    print(f"- Wrote {outdir/'profile_S.txt'}")
    print(f"- Wrote {outdir/'profile_Phi.txt'}")
    print(f"- Wrote {outdir/'summary.json'}")


if __name__ == "__main__":
    main()
