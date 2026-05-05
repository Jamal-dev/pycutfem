# Duddu (2009) benchmark: detachment models for `one_domain.py`

## What I aligned from `main.tex`

From `examples/biofilms/benchmarks/dadu/main.tex` (section **Detachment Model Suggested in Rittman (1982)**), the paper:

- Uses a continuous detachment *interface speed* (recession speed) `F_det = a |tau|^b`.
- Derives, for a **1D biofilm** in a half-channel of height `H`, a closed form for the interfacial shear magnitude
  \[
  |\tau| = \frac{2\mu u_{\max}^0 H}{(H-l)^2},
  \]
  where `l` is biofilm thickness.
- For `b = 1/2` shows
  \[
  F_{det}(l) = k(1-l/H)^{-1} = k\left(1 + \frac{l}{H} + \left(\frac{l}{H}\right)^2 + \cdots\right),
  \qquad
  k = a\sqrt{\frac{2\mu u_{\max}^0}{H}}.
  \]
- Gives the concrete parameter example (used in that derivation):
  `a = 0.1 mm/day/(Pa^1/2)`, `u_max^0 = 12.5 mm/s`, `mu = 0.001 Pa*s`, `H = 0.5 mm`, leading to `k ≈ 0.00707 mm/day`.

This is a good match to the one-domain model because `one_domain.py` already supports a surface-localized detachment sink in the `alpha` equation.

## How I mapped the literature detachment speed into the one-domain model

In `examples/utils/biofilm/one_domain.py`, the `alpha` equation uses the surface-localized term:

- Detachment sink: `D_det_prev * δ(alpha)`, with `δ(alpha) = 4 alpha (1-alpha)`.

From `examples/biofilms/model/model.tex` (Eq. “rate_from_speed”), the implemented coefficient `D_det_prev` is a **rate**:

- `D_det_prev = V_det / (4 eps_alpha)`

where:

- `V_det` is the physical interface recession speed (same units as “length/time”), and
- `eps_alpha` is the diffuse-interface thickness parameter (same units as “length”).

So to use the paper’s `F_det` directly as the recession speed, the required one-domain coefficient is:

- `D_det_prev = F_det / (4 eps_alpha)`

This mapping is exactly what the driver script below uses.

Implementation detail: the driver initializes `alpha` with a logistic/tanh profile
`alpha(y) ≈ 0.5*(1+tanh((l0-y)/(2*eps_alpha)))` so that the key identity
`|∇alpha| ≈ δ(alpha)/(4 eps_alpha)` holds (matching the derivation in `model.tex`).

Allen–Cahn note: with the specific `W'(alpha)=2 alpha(1-alpha)(1-2alpha)` used in
`one_domain.py`, the exact 1D equilibrium profile is
`alpha(s)=0.5*(1+tanh(s/(sqrt(2)*eps_alpha)))`, which implies
`|∇alpha| = δ(alpha)/(2*sqrt(2)*eps_alpha) = sqrt(2) * δ(alpha)/(4 eps_alpha)`.
So when Allen–Cahn regularization is enabled (`--ac-M` and `--ac-gamma` nonzero), the
driver applies an extra `sqrt(2)` factor in the `F_det -> D_det_prev` conversion to
match the intended physical recession speed.

## What I changed to remove “poroelastic material” differences

Duddu (2009) is not using the same poroelastic/mixture mechanics as our full one-domain model. To make a clean, publishable “apples-to-apples” verification of the **detachment mapping**:

- I **freeze** all fields except `alpha` and treat them as coefficients (no poroelastic solve).
- I **prescribe** a 1D-like growth kinematics via skeleton velocity:
  - `vS_y(y) = g*y`, `vS_x = 0`
  - This yields an interface growth speed `vS_y(l) = g*l`.
- I solve **only** the `alpha` block with PDAS bounds `0 ≤ alpha ≤ 1`.

This isolates the detachment law and its conversion into the `alpha` sink term.

## Code added (all inside the Duddu benchmark folder)

- `examples/biofilms/benchmarks/dadu/duddu2009_detachment_1d.py`
  - Runs a 2D strip (uniform-in-x) “1D” benchmark.
  - Updates `D_det_prev` each step from the measured thickness `l(t)`.
  - Writes a CSV time series including:
    - `L_pde_mm`: thickness extracted from the diffuse `alpha` field,
    - `L_ode_mm`: RK4 integrated thickness ODE reference,
    - `F_det_mm_per_day`, and `D_det_prev_per_day`.

## How to run (recommended)

Use the requested speed configuration:

```bash
conda run --no-capture-output -n fenicsx \
  python examples/biofilms/benchmarks/dadu/duddu2009_detachment_1d.py \
  --backend cpp --linear-solver petsc --model shear
```

To run the polynomial approximation or `l^2` law from the paper:

```bash
conda run --no-capture-output -n fenicsx \
  python examples/biofilms/benchmarks/dadu/duddu2009_detachment_1d.py \
  --backend cpp --linear-solver petsc --model poly --poly-order 2
```

```bash
conda run --no-capture-output -n fenicsx \
  python examples/biofilms/benchmarks/dadu/duddu2009_detachment_1d.py \
  --backend cpp --linear-solver petsc --model l2 --k-l2 0.28
```

Outputs are written under:

- `examples/biofilms/benchmarks/dadu/results/duddu2009_detachment_1d/`

## Quick results (sanity check)

With the defaults (`H=0.5 mm`, `l0=0.1 mm`, `eps_alpha=0.01 mm`, `dt=0.05 d`, `ny=240`, `g=0.2 1/d`, `backend=cpp`, PETSc linear solver):

- **PDE vs ODE thickness match (1 day run)**: max `|L_pde - L_ode|` ≈ `3.6e-4 mm` (≈ `0.36 µm`) for `model=shear`.
- **Shear vs polynomial (order 2) agreement** in this regime: max `|L_shear - L_poly|` ≈ `2.2e-4 mm` (≈ `0.22 µm`) over the same 1 day window.

This is consistent with Duddu’s statement that the polynomial expansion approximates the shear law in the 1D/thin regime.

## Stability notes (what to tune first)

If you see overshoots / oscillations in the diffuse interface:

- Reduce `--dt` (default `0.05 d`).
- Increase `--ny` (default `240`) so `eps_alpha` is resolved by multiple elements.
- Keep PDAS bounds on (the script uses PDAS by default).

If the interface becomes too smeared:

- Decrease `--eps-alpha`, but also increase `--ny` to keep it resolved.
- Optionally increase Allen–Cahn regularization via `--ac-M`, `--ac-gamma`.

## Next step (for the paper text)

Once you confirm the time series match (PDE thickness vs ODE thickness) for:

- `model=shear` and
- `model=poly` in the thin-film regime (`l/H` small),

we can lift the same detachment mapping into 2D flow-cell runs and compare against the **2D detachment models** discussed in Duddu’s paper (notably Fig. 6; and the Rittman-type shear model used in Fig. 4).

---

# 2D detachment-model comparison (Duddu 2009, Fig. 6)

The paper compares three continuous **interface recession speed** models in a 2D flow cell (Fig. 6):

- **Shear-based (Rittman-type)**: `F_det = a |tau|^b` (for Fig. 6 they use `b=0.5`, `a=0.1`).
- **Height-based**: `F_det = 0.28 l^2`.
- **Polynomial** (their `b=1/2` 1D expansion for `H=0.5mm`): `F_det = 0.00707 (1 + 2l + 4l^2)`.

In a flat-bottom flow cell, `l` is just the vertical coordinate `y` (distance to substratum).

## How it is implemented in the one-domain model

In `one_domain.py`, detachment enters the `alpha` equation as the surface-localized sink:

- `+ D_det_prev * δ(alpha)` with `δ(alpha)=4 alpha (1-alpha)`

Mapping from a physical recession speed `F_det` (mm/day) to `D_det_prev` (1/day) is:

- `D_det_prev = F_det / (4 eps_alpha)` (same as the 1D mapping)
- with an extra `sqrt(2)` factor when Allen–Cahn is enabled (same reason as the 1D note above)

## Driver added

- `examples/biofilms/benchmarks/dadu/duddu2009_detachment_2d_seq.py` (recommended for reproducible Fig. 6-style runs)
  - Solves a reduced 2D one-domain setup with active unknowns:
    - `v,p,alpha` (+ optionally `S`),
    - with `u,vS,phi` frozen (to avoid poroelastic/mechanics differences).
  - Uses **Stokes–Brinkman** (sets `rho_f=0` inside `build_biofilm_one_domain_forms`) for robustness.
  - Implements the three Fig. 6 detachment models: `--models shear,l2,poly`.
  - Writes a per-model CSV time series with:
    - `A_alpha` (≈ biofilm area), and `A_alpha_window` (optional sub-window, for Fig. 7-style comparisons),
    - `L_max`, `L_mean` (from an `alpha>=0.5` thickness profile),
    - `S_min`, `S_max` (if substrate is solved).
  - Writes a final-time VTK snapshot (`solution_*.vtu`) per model and an `outdir/summary.csv` for quick comparisons.

## How to run (recommended)

### Critical notes for Duddu (2009) Fig. 6 comparisons

- **Disable Allen–Cahn regularization for 2D detachment comparisons**:
  - With the default `--ac-M 1 --ac-gamma 1`, the Allen–Cahn term introduces curvature-driven
    shrinkage (mean-curvature flow) that is *not* part of Duddu’s level-set interface evolution.
  - In practice this makes colonies collapse to ~0 quickly (see `results/duddu2009_fig6_smoke/`).
  - For Fig. 6-style detachment-law comparisons, run with:
    - `--ac-M 0 --ac-gamma 0`
- **Paper units**: the paper specifies `U_avg` in **mm/s** and viscosity in **Pa·s**.
  - The drivers use **days** as the internal time unit, so use the convenience flags:
    - `--U-avg-mm-s 0.83` (converted to mm/day)
    - `--mu-tau-Pa-s 0.001` (converted to Pa·day by dividing by 86400)
  - You can keep `--mu-f` larger (e.g. `1.0`) for numerical convenience since (for constant μ) the
    Stokes velocity field is independent of μ; `--mu-tau-Pa-s` is what matters for detachment τ.

### Coarse smoke test (fast, checks that each model runs)

```bash
conda run --no-capture-output -n fenicsx \
  python examples/biofilms/benchmarks/dadu/duddu2009_detachment_2d_seq.py \
  --backend cpp --linear-solver petsc \
  --models shear,l2,poly \
  --nx 60 --ny 15 --q 2 \
  --dt 0.1 --t-final 0.5 \
  --adaptive-dt --dt-min 1e-3 \
  --ac-M 0 --ac-gamma 0
```

### Paper-like Fig. 6 setup (Duddu 2009: L=2mm, H=0.5mm, r0=25μm, U_avg=0.83mm/s)

```bash
conda run --no-capture-output -n fenicsx \
  python examples/biofilms/benchmarks/dadu/duddu2009_detachment_2d_seq.py \
  --backend cpp --linear-solver petsc \
  --models shear,l2,poly \
  --L 2.0 --H 0.5 --r0 0.025 \
  --k-l2 0.28 --k0-poly 0.00707 --shear-a 0.1 --shear-b 0.5 \
  --dt 0.2 --t-final 7.0 \
  --adaptive-dt --dt-min 0.01 \
  --no-line-search \
  --ac-M 0 --ac-gamma 0 \
  --U-avg-mm-s 0.83 \
  --mu-f 1.0 --mu-tau-Pa-s 0.001 \
  --kappa-inv 1e6 \
  --vtk-every 999999 --flush-csv-every 5 \
  --mu-max 1.0 \
  --outdir examples/biofilms/benchmarks/dadu/results/fig6_noac_units_mu1_t7_ny30
```

Notes:

- `--mu-max 1.0` is a simple calibration so the polynomial detachment case remains present at `t=7d`
  (with the default `mu_max=0.5` it can erode away too aggressively in this reduced setup).
- If Newton struggles with the saddle-point (Stokes) block, try `--no-line-search`.
- The very first run per detachment model compiles C++ kernels and can take ~O(1 min) per model; subsequent runs hit the cache.
- To quickly compare runs, use `examples/biofilms/benchmarks/dadu/plot_duddu2009_detachment_2d.py` (plots `A_alpha` by default).

### Postprocess into Fig. 6-style plots

After the run completes (writes `model=*/solution_*.vtu`), run:

```bash
conda run --no-capture-output -n fenicsx \
  python examples/biofilms/benchmarks/dadu/postprocess_duddu2009_fig6.py \
  --outdir examples/biofilms/benchmarks/dadu/results/fig6_noac_units_mu1_t7_ny30 \
  --models shear,l2,poly \
  --save-prefix fig6_noac_units_mu1_t7_ny30
```

This writes:

- `*_alpha_contours.png` (alpha contours with the `alpha=0.5` interface)
- `*_interface_only.png` (paper-style interface-only panel)
- `*_thickness_profiles.png` (upper-envelope thickness profile from the `alpha=0.5` contour)

and prints per-colony peak heights.

### Reference results (mu_max=1.0 run)

For the run in `examples/biofilms/benchmarks/dadu/results/fig6_noac_units_mu1_t7_ny30/`:

- `summary.csv` at `t=7d`:
  - `A_alpha(l2)=3.679e-02`, `A_alpha(shear)=1.399e-02`, `A_alpha(poly)=4.718e-03`
- Per-colony peak heights at `t=7d` from `postprocess_duddu2009_fig6.py` (mm):
  - `shear`: `0.02593, 0.03369, 0.03555, 0.03366, 0.03970`
  - `l2`:    `0.06432, 0.06382, 0.06432, 0.06382, 0.06432`
  - `poly`:  `0.01774, 0.01528, 0.01774, 0.01528, 0.01774`

Generated plots:

- `fig6_noac_units_mu1_t7_ny30_alpha_contours.png`
- `fig6_noac_units_mu1_t7_ny30_interface_only.png`
- `fig6_noac_units_mu1_t7_ny30_thickness_profiles.png`
- `plot_A_alpha.png` (optional time-series plot)

## Quick sanity result (coarse 2D run)

With a very coarse mesh (only to validate that the three detachment laws run end-to-end):

```bash
conda run --no-capture-output -n fenicsx \
  python examples/biofilms/benchmarks/dadu/duddu2009_detachment_2d.py \
  --backend cpp --linear-solver petsc \
  --models shear,l2,poly \
  --nx 6 --ny 2 --q 2 \
  --dt 0.1 --t-final 0.1 \
  --adaptive-dt --dt-min 1e-3
```

the final biofilm “area” metric satisfies:

- `A_alpha(shear) < A_alpha(l2) ≈ A_alpha(poly)`

which is consistent with Duddu’s qualitative statement that shear-based detachment induces loss even for small colonies, while `l^2` is weaker for short biofilms.

For meaningful `L_max/L_mean` and morphology comparisons you must refine the mesh in `y` (so the initial colonies and `eps_alpha` are resolved).

## Also in the paper: Rittman-type 2D shear detachment (Fig. 4)

Duddu also shows the Rittman-type shear model with `b=0.58` at two different detachment prefactors (`a=0.02` and `a=0.1`).
The same can be explored with the 2D driver by changing:

```bash
conda run --no-capture-output -n fenicsx \
  python examples/biofilms/benchmarks/dadu/duddu2009_detachment_2d.py \
  --backend cpp --linear-solver petsc \
  --models shear --shear-b 0.58 --shear-a 0.02 \
  --adaptive-dt --dt-min 1e-3
```

## What still needs calibration for “publishable” agreement

This 2D benchmark is intended to compare **detachment laws**; it does not yet reproduce Duddu’s full XFEM+level-set algorithm or their exact kinetic model. For a publishable comparison against Fig. 6 / Fig. 7 we likely need to:

- align the **growth kinetics** (their substrate-limited model) by enabling `--solve-substrate` and matching the reaction terms,
- pick a consistent unit scaling for `U_avg`, `mu_f` (and thus `tau`) and the Brinkman permeability `kappa_inv`,
- compare qualitative morphology (pinching/necking vs slab-like) using VTK snapshots and quantitative `A_alpha_window(t)` curves.
