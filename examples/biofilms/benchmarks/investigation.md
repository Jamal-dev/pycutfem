# Investigation notes — Warner (1986) benchmark vs one-domain biofilm model

This file is a running log of what was tried, what worked/failed, and the resulting conclusions. The goal is **not** to “fit” the FD Warner implementation, but to verify that the **one-domain diffuse-interface model** can reproduce the Warner benchmark behaviors (growth, step-change, shear detachment, sloughing) under a consistent mapping and consistent units.

## 1) Dimensional analysis / unit consistency

### 1.1 Warner (1986) reference quantities (as used in our FD benchmark)
- Thickness `L` is reported in **µm** in the comparison plots; internally it is a length.
- Substrate concentration `S_i` has units of **mass concentration**, consistent with the paper’s formulation (e.g. COD in `g m^-3`).
- Diffusion `D_i` in the paper has units `m^2 / d` (we use days).
- The reported interfacial flux `j_L` has units of **areal flux** `g m^-2 d^-1`.
- In our `warner1986_benchmark.py` outputs the sign convention is:
  - uptake into the biofilm is stored as **negative** `jL_1 < 0`;
  - “removal” curves should be compared as `removal := -jL_1 > 0`.

### 1.2 One-domain benchmark nondimensionalization (this repo)
We use a *geometric* nondimensionalization of space only:
- `x_phys = L_ref * x`, `y_phys = L_ref * y`
- time is kept in **days** (no `t_ref` scaling)
- substrate `S` is kept as a **physical concentration** `g m^-3`
- kinetic rates (`mu_max`, `k_d`, …) are **physical** `d^-1`

Consequences:
- Gradients are w.r.t. nondimensional coordinates, so `∇ = (1/L_ref) ∇_phys`.
- A physical diffusion coefficient `D_phys [m^2/d]` becomes a nondimensional coefficient
  `D = D_phys / L_ref^2 [1/d]` in the weak form.

### 1.3 Monod term and growth source
In `examples/utils/biofilm/one_domain.py`:
- Monod: `monod(S) = mu_max * S/(S + K_S)`
  - `mu_max [1/d]`, `S,K_S [g m^-3]` ⇒ `monod [1/d]`
- Decay: `k_d [1/d]`
- Solid fraction: `(1-phi) alpha` is dimensionless.
- Therefore:
  - `Pi_over_rho_s := (monod - k_d) * (1-phi) * alpha` has units `1/d`.

This same `Pi_over_rho_s` appears consistently in:
- the porosity equation source,
- the volume-constraint source (via `s_v`, with the outer `alpha` applied in the form),
- the substrate sink (after multiplying by density and dividing by yield).

### 1.4 Substrate equation scaling (important “density division” check)
The one-domain substrate equation is implemented in conservative form for **dissolved substrate in the fluid fraction**:
- storage term: `C(alpha,phi) * S`, where `C=(1-alpha)+alpha*phi` is the local fluid fraction.
- strong form (conceptually): `∂t(CS) + div(CS v) - div(D_S ∇S) + R_S = f_S`.

Sink term:
- The “specific” sink from the growth model is `R_S_specific = (1/Y) * Pi_over_rho_s`.
- If `S` is a **mass concentration** `g m^-3` (as in Warner), the physical sink must be
  `R_S = rho_s_star * R_S_specific = (rho_s_star/Y) * Pi_over_rho_s`,
  where `rho_s_star [g m^-3]` is intrinsic biomass density.
- This is exactly what the form uses:
  `RS_k = rho_s_star * _R_S_consumption(...)`, with `_R_S_consumption = (1/Y) * Pi_over_rho_s`.

So there is **no extra / missing division by density** in the substrate sink: the only density factor is the intended `rho_s_star`.

### 1.5 Effective density inside the biofilm (porosity scaling)
In the one-domain model, the *solid volume fraction* is `B = alpha (1-phi)`. With `alpha≈1` in the biofilm interior, the
effective biomass density driving substrate consumption is

`rho_eff = (1-phi_b) * rho_s_star`.

Therefore, if we set `phi_b=0.3` and `rho_s_star=5000`, we actually get `rho_eff=3500` and we **underpredict** substrate
consumption (and thus transport limitation and later-time growth feedbacks).

For Warner comparisons where the intended effective density is `5000 g/m^3`, use
`rho_s_star = 5000 / (1-phi_b) = 7142.857` when `phi_b=0.3`.

### 1.6 Removal flux units (comparison to Warner `j_L`)
In the one-domain benchmark driver we output two positive “removal” diagnostics:
- top-boundary diffusive flux (through `y=H`): `removal_top_flux`
- domain-integrated consumption divided by strip width: `removal_consumption`

Because the PDE is assembled on a nondimensional domain:
- a physical area element scales as `dA_phys = L_ref^2 dA`
- dividing by a physical width scales as `/ (L_ref * width)`
⇒ overall, to convert an assembled nondimensional integral to `g m^-2 d^-1` we multiply by `L_ref`.

The script therefore computes:
- `removal ≈ L_ref * ( ∫ RS dA / width )`, giving `g m^-2 d^-1`,
consistent with Warner’s `-jL_1`.

## 2) Model-form changes made (per your request)

### 2.1 Alpha advection form (advective vs conservative)
File: `examples/utils/biofilm/one_domain.py`
- α transport is now configurable via `alpha_advection_form`:
  - `advective`: `∂t α + vS·∇α = ...` (indicator / level-set-like transport)
  - `conservative`: `∂t α + div(α vS) = ...` implemented as `vS·∇α + α div(vS)`
- For Warner-style **growth via volumetric expansion** (`k_g=0`), the **advective** form is the relevant one:
  - with the conservative form and `k_g=0`, α is (nearly) conserved under expansion and the thickness tends not to grow;
  - with the advective form, the interface is carried by the skeleton velocity field and thickness grows from `div(vS)>0`.

### 2.2 No explicit alpha diffusion in this benchmark (D_alpha = 0)
File: `examples/biofilms/benchmarks/warner1986_one_domain.py`
- Default `--D-alpha` set to `0.0` (explicit diffusion removed for α).

### 2.3 Conservative Allen–Cahn regularization (mass-conserving)
File: `examples/biofilms/benchmarks/warner1986_one_domain.py`
- Added conservative Allen–Cahn controls:
  - `--alpha-cahn-conservative` (default enabled)
  - `--alpha-cahn-conservative-mode {eliminate,unknown}` (default `eliminate`)
- Adds a global `lambda_alpha` (as `:number:` field) and projects it before each assembly in `eliminate` mode.

### 2.4 Consistent Newton linearization of the volume source `s_v(S,phi)`
The mixture constraint residual is:
- `R_mass = div(C v + B vS) - alpha * s_v`.

If `s_v` depends on unknowns (e.g. Monod kinetics through `S` and/or `phi`), the consistent Newton variation is:
- `δ(alpha*s_v) = (δα) s_v + alpha * (δs_v)`.

Implementation:
- File: `examples/utils/biofilm/one_domain.py`
  - added optional `ds_v` argument (defaults to `0`) and included the extra Jacobian term `-alpha_k * ds_v` in `a_mass`.
- File: `examples/biofilms/benchmarks/warner1986_one_domain.py`
  - for `s_v = (monod(S) - k_d) * (1 - phi)` we now assemble
    - `ds_v = (dmonod) * (1 - phi_k) - (monod - k_d) * dphi`,
    - `dmonod = mu_max * (K_S / (S_k + K_S)^2) * dS`,
    when using `--s-v-jacobian full` (default).
  - `--s-v-jacobian frozen` sets `ds_v=0` (Picard linearization), matching the previous behavior.

Observed effect (cpp backend, PDAS bounds enabled):
- **Converged solutions are unchanged** (residual is identical; only the Jacobian changes).
- Newton/VI iteration counts improve when the coupling is stiff.
  Example (strip, full mechanics, `dt=1 d`, `t_final=2 d`, `s_v_mode=pi`, `s_v_lagged=0`):
  - step 2 VI-Newton iterations: `5` (frozen Jacobian) → `3` (full Jacobian).

### 2.5 Benchmark-driver IC fix: φ should not be blended with α
File: `examples/biofilms/benchmarks/warner1986_one_domain.py`
- The mixture interpolation already uses α via:
  - `C = (1-α) + α φ`, `B = α (1-φ)`.
- Therefore the *biofilm porosity* field `φ` must represent the interior value (≈ constant `φ_b` for the Warner benchmark).
- Earlier, we incorrectly initialized `φ` as a blend like `φ = 1 - (1-φ_b) α`, which double-counts the transition and makes
  `B = α(1-φ) ≈ α^2(1-φ_b)` near the interface, suppressing growth/consumption and distorting the early-time regime.
- The driver now initializes `φ(x,0) = φ_b` and (typically) freezes it for Warner comparisons.

## 3) VI constraints for robustness (PDAS / semismooth Newton)
File: `examples/biofilms/benchmarks/warner1986_one_domain.py`
- Default nonlinear solver for this benchmark is `PdasNewtonSolver` with bounds:
  - `alpha ∈ [0,1]`, `phi ∈ [0,1]`, `S ≥ 0`
- This prevents negative `S` (which can destabilize Monod terms) and keeps α physical.

Notes:
- The PDAS line search can occasionally fail to find a strict semismooth-residual decrease; the benchmark now defaults to soft line-search failure (`PYCUTFEM_LS_FAIL_HARD=0` unless the user overrides it).

## 4) Smoke tests / sanity checks

### 4.1 Frozen mechanics + VI bounds + conservative Allen–Cahn
This configuration converges in VI Newton with bounds enforced (example at `t=0.1 d`):
- `L_eff ~ O(µm)` and positive removal; α, φ, S remain in physical ranges.

### 4.2 Case 1 (strip, cpp): “skeleton” mechanics with corrected density
Parameter set (user-proposed density fix; single-substrate reduced model):
- `mechanics=skeleton`, `freeze_phi=1`, `s_v_mode=pi`, `k_g=0`, `phi_b=0.3`
- `rho_s_star=7142.857` (so `(1-phi_b)*rho_s_star=5000`)
- `eps0=0.003`, `h0=0.005`, `nx=1`, `ny=800`, `dt=0.01 d`, `t_final=0.5 d`
- `alpha_advection_form=advective` (needed for growth via volumetric expansion when `k_g=0`)

Result at `t=0.5 d` (compare to Warner reference in `case1_backend=cpp_timeseries.csv`):
- Warner: `L=9.6653 µm`, `removal=-jL_1=0.15115 g m^{-2} d^{-1}`
- One-domain (this setup): `L_half=13.58 µm`, `removal≈0.3012 g m^{-2} d^{-1}`

So the **pattern is correct** (growth + increasing removal), but the magnitude is off (`L_half` about +40%, removal about 2×).

Additional diagnostics for this run:
- Fitting `log(L_half)` over early times gives an effective exponential rate of about `2.0 d^{-1}` (t∈[0,0.5]d), whereas the sharp-interface
  expectation from Warner case 1 itself is about `1.32 d^{-1}`:
  `r = log(9.6653/5.0)/0.5 ≈ 1.318 d^{-1}`. (A heterotroph-only estimate `mu_net = mu_max*S/(K+S) - k_d = 1.6 d^{-1}` would imply
  `L(0.5)≈11.1 µm`, which is not what Warner case 1 reports because the total biomass growth rate is the mean over species and includes endogenous terms.)
- The integral-based thickness diagnostic `L_eff := (∫ α dA)/width` is significantly larger (`L_eff≈20.74 µm` at `t=0.5 d`), indicating that α is
  not sharply 0/1 and that “partial biofilm” tails (α>0) contribute substantially to the integrated growth/consumption in this diffuse-interface setup.
- With `bulk_mode=well_mixed`, the *top-boundary diffusive flux* diagnostic `removal_top_flux` is tiny (`≈3.5e-3`) because substrate is supplied by the
  internal bulk-relaxation term rather than by diffusion through the top boundary. In this mode, `removal_consumption` is the meaningful “removal from the
  reservoir” metric.

Working hypothesis for the mismatch:
- With `mechanics=skeleton`, the mixture velocity is frozen (`v=0`) so the volume constraint is enforced through `div(B vS)=α s_v`
  rather than through `div(vS)=...` (Warner’s single-velocity kinematics).
- In the diffuse interface, `B=α(1-φ)` is small, so `vS` can become large there (to satisfy the constraint), and the `α=0.5`
  contour can advect faster than the sharp-interface interface speed.
- Quantitatively matching Warner’s kinematics is therefore expected to require either:
  - **full mechanics** with strong drag enforcing `v≈vS` (so the total volume flux is close to `vS` and `div(vS)` matches the intended `s_v`),
    or
  - the “interface-growth mapping” (`k_g>0`) as a numerics-only surrogate benchmark.

### 4.3 Case 1 (strip, cpp): full mechanics + strong drag (recommended)
Parameter set (still single-substrate reduced model, but improved kinematics):
- `mechanics=full`, `freeze_phi=1`, `freeze_u=1`, `freeze_vSx=1`
- `s_v_mode=mu` (so `div(vS)≈mu_net` when `v≈vS`), `s_v_jacobian=full`, `k_g=0`
- `phi_b=0.3`, `rho_s_effective=5000` (so `(1-phi_b)*rho_s_star=5000`)
- `eps0=0.003`, `h0=0.005`, `nx=1`, `ny=240`, `Hy=0.05`, `dt=0.05 d`, `t_final=0.5 d`
- `kappa_inv=1e10`, `bulk_mode=well_mixed`, `bulk_gamma=1000`

Result at `t=0.5 d` (Warner reference: `L=9.6653 µm`, `removal=-jL_1=0.15115 g m^{-2} d^{-1}`):
- One-domain: `L_half=9.2768 µm` (**−4.0%**), `removal≈0.17411 g m^{-2} d^{-1}` (**+15.2%**)

Notes:
- Switching `s_v_mode` from `mu` → `pi` in this *full* mechanics setting underpredicts thickness (because `(1-phi)` is already “built into”
  `B=α(1-phi)` and `C=(1-α)+αφ`; using `pi` double-counts the porosity factor).
- This setting produces an early-time exponential thickness rate fit of about `1.24 d^{-1}` (close to Warner’s `~1.32 d^{-1}`).
- The remaining removal mismatch is expected from the reduced mapping (Warner COD uptake is by heterotrophs only and the total growth rate includes
  autotrophs; the one-domain single-biomass model cannot represent this competition without additional species fields).

### 4.4 Key observation: initial interface width vs initial thickness
To get meaningful comparisons, we need both:
- `eps0 << h0` so the initial “interior” region reaches `α≈1`, and
- a y-mesh that resolves `eps0` (rule of thumb: `Hy/ny ≲ eps0`).

For Warner-style growth comparisons the setup should satisfy `eps0 << h0` **and** the mesh should resolve `eps0`
(`Hy/ny ≲ eps0`) to ensure the intended interior values `alpha≈1` and `phi≈phi_b` are realized.

### 4.5 Current one-domain status vs Warner (cases 1–4)
As of Feb 24, 2026:
- The **finite-difference Warner benchmark** (`warner1986_benchmark.py`) matches the paper curves (thickness and flux patterns).
- The **one-domain PDE model** run on a 2D strip (uniform in x) does **not** reproduce Warner’s thickness growth law:
  - With conservative α advection and `k_g=0`, total α is essentially conserved (no thickness growth).
  - With advective α advection and `k_g=0`, thickness grows via the skeleton kinematics, but the quantitative match depends strongly on the reduced-model mapping (single substrate/species vs Warner’s multispecies/multisubstrate system).
  - With interface-growth (`k_g>0`), we can match thickness more directly (this is useful for testing numerics, but it is no longer a “pure volumetric expansion” benchmark).

These results indicate that, for a publishable Warner benchmark, we need to revisit the **mapping** from Warner’s

## 5) Note on θ=0.5 (Crank–Nicolson) and stiff Monod kinetics

It is tempting to reduce numerical dissipation by switching from backward Euler (`θ=1`) to Crank–Nicolson (`θ=0.5`).
However, for the Warner parameter ranges and the one-domain scaling used here, the **substrate sink is extremely stiff**
in physical units:

- In the biofilm interior (`α≈1`, `φ≈φ_b=0.3`, `S≈3`, `K_S=5`, `μ_max=4.8`, `k_d=0.2`, `Y=0.4`,
  `ρ_s^*≈7142.857` so `(1-φ_b)ρ_s^*=5000`), the linearization of the consumption term has magnitude
  \[
  \frac{dR_S}{dS} \approx \frac{ρ_s^*}{Y}\,(1-φ)\,\frac{d}{dS}\Bigl(μ_{\max}\frac{S}{S+K_S}\Bigr)
  \sim 4.7\times 10^3\ \mathrm{d}^{-1}.
  \]

For a pure decay ODE `S_t = -k S`, Crank–Nicolson updates as
`S_{n+1} = ((1 - kΔt/2)/(1 + kΔt/2)) S_n`, which becomes **sign-oscillatory** once `kΔt > 2`.
With `k≈4700 d^{-1}`, this requires `Δt ≲ 4e-4 d` (tens of seconds) to avoid oscillations.

In practice, with `θ=0.5` and `Δt` in the `10^{-2} d` range, the Newton/PDAS solve can exhibit:
- large transient overshoots in `S` (since only a lower bound `S≥0` is imposed),
- active-set “chattering”, and
- stagnation of the `S` residual even when mechanics blocks converge.

Conclusion: for these stiff kinetics, **`θ=0.5` is not a robust choice** unless `Δt` is extremely small, or the
time integration is changed to an L-stable 2nd-order method (e.g. BDF2 / TR-BDF2) or an IMEX-style treatment where
the stiff reaction is kept fully implicit.

### 5.1 IMEX-style fix implemented (substrate only)
To enable practical `θ=0.5` runs with `Δt=O(10^{-2} d)` we implemented an IMEX-style time discretization **only for the
substrate equation stiff terms**:

- substrate reaction `R_S` can be treated fully implicit (`--substrate-reaction-scheme implicit`)
- substrate diffusion `-div(D∇S)` can be treated fully implicit (`--substrate-diffusion-scheme implicit`)

Implementation: `examples/utils/biofilm/one_domain.py` adds
`substrate_reaction_scheme` and `substrate_diffusion_scheme` (default `"theta"` so existing behavior is unchanged).

With this, a Crank–Nicolson run (`θ=0.5`) can keep second-order accuracy in the smoother transport/kinematics blocks,
while using backward-Euler damping for the stiff substrate operators.

### 5.2 Warner case 1 check (strip, cpp, θ=0.5, IMEX substrate)
Run settings (key items):
- `mechanics=full`, strong drag (`kappa_inv=1e10`), `freeze_phi=1`, `s_v_mode=mu`
- `θ=0.5`, `Δt=0.025 d`, `t_final=0.5 d`
- IMEX substrate: `substrate_reaction_scheme=implicit`, `substrate_diffusion_scheme=implicit`

Outcome at `t=0.5 d` (from `examples/biofilms/results/warner1986_one_domain/one_domain_strip_case1_backend=cpp_timeseries.csv`):
- One-domain: `L_half=9.3648 µm`, `removal=1.7155e-01 g m^{-2} d^{-1}`
- Warner FD reference: `L=9.6653 µm`, `-jL_1=1.5115e-01 g m^{-2} d^{-1}`

So thickness matches closely (≈ −3%), and removal is still high (≈ +14%) but the pattern is correct.

### 4.6 2D check (case 1): x-averaged thickness/removal
Goal: confirm that the 2D implementation reproduces the strip (“1D-like”) benchmark when initialized uniformly and
run on a sufficiently resolved mesh.

Current quick run (not yet a strict 1D equivalence test):
- `mode=both`, `mechanics=full`, `kappa_inv=1e10`, `freeze_phi=1`, `freeze_u=1`, `freeze_vSx=1`
- strip mesh: `nx=4, ny=240`, 2D mesh: `nx_2d=32, ny_2d=32` (coarse in y)
- 2D used a wavy initial thickness (`h0_amp=0.2`), so it is **not** uniform-in-x.

At `t=0.5 d` (Warner reference: `L=9.6653 µm`, `removal=-jL_1=0.15115 g m^{-2} d^{-1}`):
- strip: `L_half=9.2768 µm` (**−4.0%**), `removal≈0.17411` (**+15.2%**)
- 2D: `L_half=8.5813 µm` (**−11.2%**), `removal≈0.15517` (**+2.7%**)

Interpretation:
- The 2D result is **not directly comparable** to the strip because (i) it starts with a wavy free surface and (ii) the
  vertical mesh is much coarser (32 vs 240 elements over the same height).
- To validate 2D against strip/Warner, re-run with `--h0-amp 0` and a y-resolution comparable to the strip (e.g. `ny_2d≈128–256`),
  and compare `L_eff` (area-based) and x-averaged `L_half`.

Uniform-in-x rerun (still moderate y-resolution):
- `h0_amp=0`, 2D mesh: `nx_2d=32, ny_2d=64`, strip: `nx=4, ny=240`
- At `t=0.5 d`:
  - strip: `L_half=9.1931 µm` (**−4.9%**), `removal≈0.16791` (**+11.1%**)
  - 2D: `L_half=8.6433 µm` (**−10.6%**), `removal≈0.14804` (**−2.1%**)

Next: increase `ny_2d` (and possibly reduce `eps0`) to check convergence of the x-averaged thickness.
sharp-interface kinematics to the one-domain diffuse-interface transport (likely involving either a different α transport
strategy in the growth regime or a dedicated Warner-limit reduction of the one-domain model where porosity/capacity scalings
match the constant-density assumptions in Warner & Gujer).

## 5) Open work (still required for the benchmark paper)
- Use `PYCUTFEM_NEWTON_TRACE_RES_FIELDS=1` (and/or VI tracing) on the *mechanics-enabled* runs to identify which block stalls first (often substrate or constraint coupling).
- Calibrate a **strip (“1D-like”)** parameter set that matches Warner case 1–4 thickness and `-jL_1`.
- Run the corresponding **2D** cases and compare:
  - x-averaged thickness/removal vs strip,
  - and (optionally) thickness profile `h(x,t)` from the α=0.5 contour.
- Add the final parameter table + solver settings + stability discussion to the manuscript section.
