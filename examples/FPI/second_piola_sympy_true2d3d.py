#!/usr/bin/env python
"""
Symbolic checks for the compressible Neo-Hookean second Piola stress (true 2D vs 3D).

This script supports the derivation used in `examples/FPI/fpi_eq30_residual_jacobian.tex`,
in particular Eq. (Sel):

  S_el(F) = 2c ( I - J^{-2β} C^{-1} ),   C = F^T F,  J = det(F)

with
  c = E / (4(1+ν)),
  β = ν / (1 - (d-1)ν),  d∈{2,3}.
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, Tuple

import sympy as sp


def _beta(nu: sp.Symbol, *, dim: int) -> sp.Expr:
    return nu / (1 - (dim - 1) * nu)


def _nh_second_piola(F: sp.Matrix, *, E: sp.Symbol, nu: sp.Symbol, dim: int) -> sp.Matrix:
    c = E / (4 * (1 + nu))
    beta = _beta(nu, dim=dim)
    C = F.T * F
    J = F.det()
    I = sp.eye(dim)
    return sp.simplify(2 * c * (I - J ** (-2 * beta) * C.inv()))


def _nh_pk1_from_energy(F: sp.Matrix, *, E: sp.Symbol, nu: sp.Symbol, dim: int) -> Tuple[sp.Matrix, sp.Expr]:
    c = E / (4 * (1 + nu))
    beta = _beta(nu, dim=dim)
    C = F.T * F
    J = F.det()
    psi = c * (sp.trace(C) - dim) + (c / beta) * (J ** (-2 * beta) - 1)
    P = sp.Matrix([[sp.diff(psi, F[i, j]) for j in range(dim)] for i in range(dim)])
    return sp.simplify(P), sp.simplify(psi)


def _random_invertible_F(dim: int, *, seed: int) -> Dict[sp.Symbol, float]:
    rng = random.Random(seed)
    vals: Dict[sp.Symbol, float] = {}
    while True:
        data = [[rng.uniform(0.5, 1.5) if i == j else rng.uniform(-0.4, 0.4) for j in range(dim)] for i in range(dim)]
        # Ensure it's not too close to singular
        det = float(sp.Matrix(data).det())
        if abs(det) > 0.2:
            break
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, choices=(2, 3), default=2)
    ap.add_argument("--check", action="store_true", default=True)
    ap.add_argument("--n-check", type=int, default=3)
    args = ap.parse_args()

    dim = int(args.dim)
    E, nu = sp.symbols("E nu", positive=True, finite=True)

    Fsyms = sp.Matrix(dim, dim, lambda i, j: sp.Symbol(f"F{i+1}{j+1}"))
    P, psi = _nh_pk1_from_energy(Fsyms, E=E, nu=nu, dim=dim)
    S_expected = _nh_second_piola(Fsyms, E=E, nu=nu, dim=dim)

    print(f"dim={dim}")
    print("beta =", _beta(nu, dim=dim))
    print("psi(F) =", psi)
    print("S_expected(F) =", S_expected)

    if not bool(args.check):
        return

    # Numerical spot-check: verify that P == F * S_expected for random invertible F.
    for k in range(int(args.n_check)):
        subs: Dict[sp.Symbol, float] = {E: 10.0, nu: 0.25}
        rng = random.Random(1234 + k)
        while True:
            data = [
                [rng.uniform(0.6, 1.4) if i == j else rng.uniform(-0.3, 0.3) for j in range(dim)] for i in range(dim)
            ]
            det = float(sp.Matrix(data).det())
            if abs(det) > 0.2:
                break
        for i in range(dim):
            for j in range(dim):
                subs[Fsyms[i, j]] = float(data[i][j])

        P_num = np.asarray(P.subs(subs), dtype=float)
        F_num = np.asarray(Fsyms.subs(subs), dtype=float)
        S_num = np.asarray(S_expected.subs(subs), dtype=float)
        lhs = P_num
        rhs = F_num @ S_num
        err = float(np.max(np.abs(lhs - rhs)))
        print(f"[check {k}] max|P - F*S| = {err:.3e}")
        if not (err < 1.0e-8):
            raise SystemExit(2)


if __name__ == "__main__":
    import numpy as np

    main()

