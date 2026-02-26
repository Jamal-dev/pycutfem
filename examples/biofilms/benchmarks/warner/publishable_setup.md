# Warner (1986) publishable verification target for `one_domain.py`

## Which literature example aligns with our model?

From `warner1986.tex`, the most directly alignable numerical examples are **Cases 1–4**:

- **Case 1:** unrestricted growth, fixed surface substrate concentrations.
- **Case 2:** same as case 1, but **step change** in organics at the surface at **t = 6 d**.
- **Case 3:** same as case 1, but continuous biomass **shear removal** with `σ = -λ L^2`, `λ = 750 m^{-1} d^{-1}`.
- **Case 4:** same as case 1, but a short **sloughing event** with `σ = -0.05 m/d` on **[5.984, 5.994] d** (ΔL ≈ −500 µm).

Implementation note for the one-domain driver (`warner1986_one_domain.py`):
- Prefer `--slough-mode shift_window` so the short **[5.984, 5.994] d** sloughing window is represented in the time stepping (and the substrate field is consistent at `t=5.994 d`). `--slough-mode shift` is still available for **thickness-only** checks but applies the drop instantly at `t=5.994 d`.

These cases assume **fixed surface concentrations** (no external mass-transfer resistance), which maps cleanly to the one-domain setup using either:
- a top Dirichlet boundary with a very thin fluid layer, or
- the recommended `--bulk-mode well_mixed` relaxation to emulate a fixed “reservoir” above the film.

Removal/flux comparison note:
- Warner’s reported `jL_1` uses a **fixed 15-point ζ-grid** (UPDATE: `NPOINT=15`) and a one-sided end-stencil at the surface.
  For like-for-like comparisons (especially the **Case 2 spike at t=6 d**), use `--removal-metric warner_stencil` in the one-domain driver.

## Reduced mapping used for the one-domain comparison

The one-domain PDE model currently has **one substrate** field `S`, so we use the reduced mapping:

- `S` ≈ Warner’s `S1` (organics/COD).
- One-domain uses a diffuse interface `α`; thickness is measured either by `α=0.5` (`L_half`) or by `L_eff := ∫α/width`.
- Warner’s 1D kinematics are best approximated (for cases 1–4) with:
  - `--mechanics full` and strong drag (`--kappa-inv 1e10`) so `v ≈ vS` and `div(vS)` matches the intended volumetric source,
  - constant porosity via `--freeze-phi`.

See `investigation.md` for the unit mapping and the “effective density” note (`rho_s_effective` vs `rho_s_star`).

## Reproducible suite runner

Run the reference (FD ζ-model) + one-domain comparisons + summary:

```bash
conda run --no-capture-output -n fenicsx \
  python -u examples/biofilms/benchmarks/warner/run_publishable_suite.py
```

Print the commands without running them:

```bash
conda run --no-capture-output -n fenicsx \
  python -u examples/biofilms/benchmarks/warner/run_publishable_suite.py --dry-run
```

Outputs:
- Reference: `examples/biofilms/results/warner1986/`
- One-domain: `examples/biofilms/results/warner1986_one_domain/`
- Summary tables: `examples/biofilms/results/warner1986_one_domain/publishable_suite_summary_backend=*.{csv,md}`
