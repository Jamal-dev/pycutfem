# Mode-I phase-field crack (Miehe split)

This folder contains a small **reference** example to validate the 2D spectral (tension/compression) split used in phase‑field fracture models (Miehe et al., 2010) and to produce VTK output that can be compared to deal.II’s `examples/dynamic_fracture` (Miehe tension / slit setups).

## What it solves

- Small‑strain linear elasticity with **tensile/compressive split**:
  \[
  \sigma = g(d)\,\sigma^+ + \sigma^-,
  \qquad g(d)=(1-\kappa)(1-d)^2+\kappa,
  \]
  where \(\sigma^+\) is built from the strictly positive principal strains.
- AT2‑like phase‑field damage evolution for \(d\in[0,1]\) driven by a **history** field \(H=\max_{\tau\le t}\psi^+(\boldsymbol u(\tau))\) (updated explicitly between staggered sub‑solves).
- A simple **staggered** (alternate minimization) time step:
  1) solve mechanics for \((\boldsymbol u,\boldsymbol v)\) with fixed \(d\),
  2) update \(H\) from \(\psi^+(\boldsymbol u)\),
  3) solve the damage equation for \(d\) with fixed \(H\),
  4) enforce irreversibility by clipping \(d^{n+1}\leftarrow\max(d^{n+1},d^n)\).

## Running

From repo root:

```bash
python examples/fracture/mode_I_crack/mode_I_crack.py --nx 80 --ny 80 --dt 1e-4 --t-final 2e-2 --vtk-every 1 --outdir examples/debug/out/mode_I
```

Outputs are written as `solution_0000.vtu`, `solution_0001.vtu`, ... under `--outdir`.

## Notes for deal.II comparison

- deal.II’s `dynamic_fracture` example uses an intactness field \(\varphi\in[0,1]\) with
  \(g(\varphi)=(1-\kappa)\varphi^2+\kappa\). This script uses a damage field \(d=1-\varphi\), so the formulas match.
- The loading/parameters in `mode_I_crack.py` can be set to match the deal.II `MIEHE_TENSION` case (unit square / slit).
