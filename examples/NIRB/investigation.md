# NIRB implementation note

This note makes the paper-to-code mapping explicit before the benchmark-specific drivers are built.

## Matrix convention

The reusable MOR code introduced for this benchmark uses the paper convention for high-dimensional snapshots:

- `F in R^(N x m)` stores interface-force snapshots as `features x snapshots`
- `U in R^(N_u x m)` stores displacement snapshots as `features x snapshots`
- reduced coordinates are also stored as `modes x snapshots`
- regressors operate on sample-major latent matrices, so the offline/online pipeline transposes reduced coordinates when calling the regression backend

This keeps the core linear-algebra formulas aligned with Eqs. (10)-(21) while still interoperating cleanly with `scikit-learn`.

## Equation map

| Paper | Meaning | Repo module |
| --- | --- | --- |
| Eq. (10)-(13) | POD/PCA encoding of interface forces | `pycutfem/mor/pod.py` |
| Eq. (14)-(17) | Quadratic-manifold decoding of displacement snapshots | `pycutfem/mor/quadratic_manifold.py` |
| Eq. (19)-(20) | Interface restriction / direct interface reconstruction | `pycutfem/mor/interface.py` |
| Eq. (21) | Reduced-space regression operator | `pycutfem/mor/regressors.py` |
| Eq. (26), (30), (31), (34) | Validation and online error metrics | `pycutfem/mor/metrics.py` |
| Alg. 1 | Offline ROM assembly | `pycutfem/nirb/offline.py` |
| Alg. 2 | Online ROM evaluation | `pycutfem/nirb/online.py` |

## Example-specific targets that must still be wired in

- Example 1:
  - thin-plate-spline RBF regression
  - 20% held-out mode sweep
  - linear decoder fallback when quadratic fitting is under-resolved
- Example 2:
  - 9 displacement modes
  - degree-2 polynomial regression
  - Lasso/LARS/BIC model selection
  - interface-only online exchange

## Current implementation slice

The first implementation slice adds reusable MOR and NIRB building blocks plus synthetic tests. The benchmark drivers, FOM data collection, and reproduction reports still need to be connected on top of this core.
