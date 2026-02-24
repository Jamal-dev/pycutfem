# Investigation notes — Zhang et al. (2008) cavity example vs one-domain biofilm model

Goal: align a *publishable* verification case for the one-domain diffuse-interface model implemented in
`examples/utils/biofilm/one_domain.py` against a comparable numerical example in the Zhang–Cogan–Wang (2008) phase-field paper
included here as `examples/biofilms/benchmarks/zhang/main.tex`.

## 1) Chosen “comparable” numerical example from the paper

We align with **Section 5.1: “Growth of biofilms in a cavity”** (paper `main.tex`):

- 2D domain: \(\Omega=[0,L]\times[0,1]\) (in the paper, \(L\in\{1,4\}\)).
- **Top-feeding** substrate boundary condition \(c=c^\*\) at \(y=1\).
- No-slip velocity at the cavity walls.
- Initial condition: a bottom-attached biofilm layer with a **single hump extruding into the solvent** that grows into a mushroom shape.

Reason: this case avoids periodic BCs in \(x\) and is the closest match to our current one-domain implementation’s “fixed container + growth” setup.

## 2) Model mapping (paper → this repo)

### 2.1 Paper variables (dimensionless form)
From Eq. (3.3)–(3.5) in `main.tex`, key fields:

- \(v, p\): mixture velocity and pressure, with \(\nabla\cdot v = 0\).
- \(\phi_n\): polymer network (EPS) volume fraction, \(\phi_s=1-\phi_n\).
- \(c\): nutrient concentration.

Constitutive choice used in the paper’s simulations (Eq. (3.4), VA-model):

- \(\tau_n=\frac{2}{Re_n}D\), \(\tau_s=\frac{2}{Re_s}D\),
  so the viscous part is \(\nabla\cdot\left(2(\phi_n/Re_n+\phi_s/Re_s)D\right)\).

### 2.2 One-domain variables (this repo)
From `examples/biofilms/model/model.tex` and the code, the one-domain model uses:

- mixture velocity \(v\), pressure \(p\),
- skeleton velocity \(v^S\) (primary unknown),
- porosity \(\phi\) (fluid fraction inside biofilm),
- diffuse indicator \(\alpha\in[0,1]\) (biofilm presence),
- substrate \(S\) (our nutrient analogue).

Derived mixture weights:

- \(C(\alpha,\phi)=(1-\alpha)+\alpha\phi\) (fluid “capacity”),
- \(B(\alpha,\phi)=\alpha(1-\phi)\) (solid fraction).

### 2.3 Practical mapping used for Zhang comparisons
We map:

- paper polymer fraction \(\phi_n \;\leftrightarrow\; B=\alpha(1-\phi)\),
- paper solvent fraction \(\phi_s=1-\phi_n \;\leftrightarrow\; C=(1-\alpha)+\alpha\phi\),
- paper nutrient \(c \;\leftrightarrow\; S\).

For the Zhang shear cases the paper mentions typical EPS values around **0.09** at the base, e.g.:
“value of \(\phi_n\) … base \(0.09\)” (see the shear-flow discussion around Figs. 8–9).
To match that interior fraction while keeping a sharp biofilm/solvent transition driven by \(\alpha\),
we use a constant biofilm porosity \(\phi_b\) such that:

\[
B_{\mathrm{inside}} = 1-\phi_b \approx 0.09 \quad\Rightarrow\quad \phi_b\approx 0.91.
\]

This is implemented by default in the driver with `--phi-b 0.91 --freeze-phi 1`.

## 3) Parameter alignment knobs

### 3.1 Viscosity / Reynolds numbers (extended-Newtonian “no poroelasticity” regime)
Zhang’s viscous stresses use coefficients \(1/Re_s\) and \(1/Re_n\), weighted by \(\phi_s\) and \(\phi_n\).

In the one-domain model, an “extended Newtonian / non-poroelastic” regime is approximated by:

1) **Disable elasticity**: `--mu-s 0 --lambda-s 0 --freeze-u 1`.
2) **Use strong drag** to keep \(v\approx v^S\): increase `--kappa-inv`.
3) Represent polymer viscosity using the Kelvin–Voigt skeleton viscosity `solid_visco_eta`:
   - set `--solid-visco-eta ≈ (φ_n0/Re_n)` where \(φ_n0\approx 1-φ_b\).
4) Represent solvent viscosity using `mu_f`:
   - set `--mu-f ≈ 1/Re_s`,
   - set `--mu-b-model phi_mu` so the effective viscosity inside the biofilm scales with \(C\approx \phi_s\).

For the cavity parameters in the paper (Section 5, “cavity case”):

- \(Re_s=9.98\times 10^{-4}\) ⇒ \(1/Re_s\approx 1.00\times 10^{3}\),
- \(Re_n=2.33\times 10^{-9}\) ⇒ \(1/Re_n\approx 4.29\times 10^{8}\),
- with \(\phi_b=0.91\) ⇒ \(φ_n0=1-\phi_b=0.09\),
  so `solid_visco_eta ≈ 0.09 / Re_n ≈ 3.86×10^7`.

These values are *very stiff* numerically; use the C++ backend + PETSc in the `fenicsx` environment as you requested.

### 3.2 Growth source in our model (s_v choices)
The one-domain growth/expansion enters through the mixture constraint:

\[
\nabla\cdot(Cv + Bv^S)=\alpha\,s_v.
\]

The benchmark driver supports:

- `--s-v-mode mu`: \(s_v \approx \mu_{\mathrm{net}}=\mu_{\max}\,S/(K_S+S) - k_d\),
- `--s-v-mode pi`: \(s_v \approx \mu_{\mathrm{net}}(1-\phi)\) (biomass-weighted).

Which is “closer” to Zhang depends on what you want to hold fixed:

- If you freeze porosity (`--freeze-phi 1`) and treat the film as expanding at roughly constant \(B\),
  then `mu` is often the better thickness-growth driver.
- If you want the source magnitude to scale with the polymer fraction \(φ_n\), then `pi` is the closer analogue.

In all cases, keep `--s-v-jacobian full` for Newton robustness. (A frozen/Picard Jacobian is supported but generally slower.)

### 3.3 Substrate consumption scaling
Zhang’s substrate sink is \(g_c = A\,\phi_n\,c\) with \(A=100\) in their cavity runs.

Our substrate sink is yield-coupled to the growth model:
\[
R_S \sim (\rho_s^\*/Y)\,\Pi_b/\rho_s^\*,
\]
so you control the magnitude primarily via `--rho-s-star` and `--Y`.
For comparisons we treat `rho_s_star/Y` as a calibration knob to match the paper’s depletion time scale near the top-fed boundary.

## 4) Driver added in this benchmark folder

File: `examples/biofilms/benchmarks/zhang/zhang2008_cavity_one_domain.py`

Key features:

- Cavity geometry + single-hump \(\alpha\) initial condition.
- Top-feeding substrate Dirichlet BC `S=S_top` on `top`.
- Writes a time-series CSV with:
  - \(L_{\mathrm{eff}} := (\int_\Omega \alpha\,\mathrm{d}A)/L_x\) (integral thickness proxy),
  - crude nodal “height” proxies \(H_{\max},H_{\mathrm{mean}}\) from the \(\alpha\ge 0.5\) set,
  - extrema of \(\alpha,S\) and max norms of \(v,p,v^S\).
- Optional `--snapshot-every N` to dump compressed `.npz` snapshots for plotting (coords + nodal values).

## 5) Stability notes / bugs found and fixed

While setting up the Zhang-aligned “extended Newtonian” regime we needed `solid_visco_eta != 0`.
Two Jacobian assembly regressions were discovered and fixed:

1) **`solid_visco_eta` Jacobian assembly bug**
   - Symptom: python backend assembly error `VecOpInfo shapes mismatch ... Roles: trial, other=function`.
   - Fix: in `examples/utils/biofilm/one_domain.py` keep viscous-skeleton Jacobian contributions separated by trial family, and avoid forming `trial + 0*function` terms when damage is disabled.

2) **Frozen/omitted volumetric source Jacobian (`ds_v`) bug**
   - Symptom: same `VecOpInfo` shape-mismatch when `ds_v` is not provided (default) or passed as `Constant(0.0)` (Picard linearization).
   - Fix: treat the “zero derivative” as an explicit zero *trial* expression internally so the mass-block Jacobian stays purely trial-based.

Regression test added:
- `tests/test_biofilm_one_domain_solid_visco_eta_regression.py`

## 6) Suggested reproduction commands (cpp + PETSc)

Fast sanity run (small mesh, short time):
```bash
conda run --no-capture-output -n fenicsx \\
  python examples/biofilms/benchmarks/zhang/zhang2008_cavity_one_domain.py \\
  --backend cpp --nx 32 --ny 32 --dt 1.0 --t-final 10.0 \\
  --phi-b 0.91 --freeze-phi 1 \\
  --mu-s 0 --lambda-s 0 --freeze-u 1 \\
  --mu-b-model phi_mu --kappa-inv 1e8 \\
  --solid-visco-eta 1e3 \\
  --s-v-mode mu --s-v-jacobian full \\
  --snapshot-every 5
```

Zhang-scaled viscosity attempt (stiff; expect to require smaller dt / stronger solver settings):
```bash
conda run --no-capture-output -n fenicsx \\
  python examples/biofilms/benchmarks/zhang/zhang2008_cavity_one_domain.py \\
  --backend cpp --nx 64 --ny 64 --dt 0.5 --t-final 50.0 \\
  --phi-b 0.91 --freeze-phi 1 \\
  --mu-s 0 --lambda-s 0 --freeze-u 1 \\
  --mu-f 1002.004 --mu-b-model phi_mu --kappa-inv 1e10 \\
  --solid-visco-eta 3.863e7 \\
  --s-v-mode pi --s-v-jacobian full \\
  --rho-s-star 800 --Y 1.0
```

## 7) What still needs to be done for the paper write-up

To make the comparison truly publishable, we still need to:

- pick a “reporting metric” that is as close as possible to what Zhang reports (since the paper mostly shows contours),
  e.g. interface height vs time, or a few representative contour snapshots at the same reported times;
- run the Zhang-scaled case to longer times (e.g. up to \(t=200\) and \(t=300\)) and extract:
  - \(\alpha\) (or \(B=\alpha(1-\phi)\)) level curves,
  - velocity and pressure contours,
  - substrate profiles;
- document which of `s_v_mode` (`mu` vs `pi`) yields the closest qualitative match for the hump→mushroom transition.

