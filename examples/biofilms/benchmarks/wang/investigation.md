# Investigation notes — Benchmark 3 redesign around Wang2014

Goal: replace the old Duddu growth comparison with a sharper Benchmark 3 that
answers the reviewer question for Paper 1 directly:

- does the one-domain interface representation converge to a clean two-domain
  sharp-interface reference when the geometry is fixed and the only issue is
  free-fluid / porous coupling across the interface?

## 1) Why Duddu is the wrong benchmark for Paper 1

The Duddu growth problem is a poor Benchmark 3 choice for the reduced
deformation-only paper because:

- it is fundamentally a growth benchmark rather than a representation benchmark,
- the XFEM reference path itself is not a clean paper-faithful target in the
  current repo,
- one-domain runs allow merger of neighboring diffuse colonies while the
  two-domain sharp-interface reference does not,
- and the growth kinetics create ambiguity about whether disagreement is caused
  by the one-domain representation, the transport law, or the biological model.

That means a good `y_top(t)` overlay does not answer the question we need for
Paper 1.

## 2) Wang2014 is the right comparison architecture

Paper:

- `examples/biofilms/deformation_part_numerical_study/69abdda4cb82b99640520d1c/others_papers/wang2014.tex`

External prototypes already available on disk:

- one-domain notebook:
  `/mnt/sda1/PHD/phd-work/task1/one_domain_approach/one_dim_approach.ipynb`
- two-domain notebook:
  `/mnt/sda1/PHD/phd-work/task1/two_domain_approach/two_domain_approach.ipynb`

The Wang comparison is better aligned with the Paper 1 claim because it uses:

- a fixed geometry,
- a flat or stepped interface,
- a direct one-domain vs two-domain comparison,
- and regional `L2` / semi-`H1` errors rather than only contour overlays.

This isolates the representation question cleanly.

## 3) Implemented benchmark in this repo

Added:

- `examples/biofilms/benchmarks/wang/paper1_benchmark3_wang2014_layered.py`
- `examples/biofilms/benchmarks/wang/paper1_benchmark3_wang2014_staircase.py`

Current scope:

- Wang2014 Example 6.1 style layered strip benchmark,
- Wang2014 Example 6.2 style stepped-interface reduced benchmark,
- one-domain transition-layer solve on `y in [-1,1]`,
- closed-form two-domain reference profile for the layered case,
- sharp split two-region reference on the stepped geometry,
- profile overlays for selected `K`,
- regional `L2` / semi-`H1` errors in the fluid and porous subdomains,
- and full-field staircase comparison panels for representative `K`.

Notation note:

- the Wang transition-layer sharpness is now called `transition_sharpness`
  in code and `\vartheta_{\mathrm{tr}}` in the manuscript,
- so the symbol `\theta` remains reserved for the one-step time integrator used
  in the reduced transient benchmarks.

Implementation choices:

- We solve the one-domain profile with a second-order conservative finite
  difference discretization of the 1D transition-layer ODE.
- We keep this benchmark outside `pycutfem` because the goal here is a
  reproducible paper benchmark scaffold, not a new FE backend.
- The driver is intentionally lightweight so it can be rerun inside the
  paper-ready suite with no external notebooks.

## 4) What this benchmark proves

This benchmark family now gives Paper 1 a clean layer-3 baseline:

- the one-domain transition profile is compared to a two-domain sharp-interface
  target on the same geometry,
- the comparison is free of growth and topological-merger ambiguity,
- the layered case supplies the clean flat-interface permeability tables,
- the stepped case shows the same agreement on a non-flat geometry with corners,
- and the reported metrics are the same kind of regional error measures used in
  Wang2014 together with staircase-specific field/profile diagnostics.

## 5) Locked production state

Production run tag:

- `benchmark3_wang_publish_ready_20260307`

Layered case:

- permeability ladder: `1e-2,1e-3,1e-4,1e-5,1e-6`
- strip resolution: `n_y = 8000`
- finest-row metrics:
  - `L2_f = 2.892e-04`
  - `L2_p = 1.145e-05`
  - `H1_f = 2.623e-03`
  - `H1_p = 7.149e-03`

Stepped case:

- interface-resolving grid pairs:
  - `(K,n) = (1e-2,128), (1e-3,256), (1e-4,512)`
- finest-row metrics:
  - `L2_f = 5.022e-04`
  - `L2_p = 5.439e-05`
  - `H1_f = 1.164e-02`
  - `H1_p = 6.598e-03`
  - `profile_inf = 2.392e-05`
  - `max_field_abs_error = 3.268e-03`

Conclusion:

- the layered case closes the flat-interface asymptotic argument,
- the stepped case closes the non-flat-interface objection,
- and Benchmark 3 is now strong enough to keep as the publication baseline for
  Paper 1.

Further work is optional rather than blocking:

1. replace the stepped reduced sharp split model with a full mixed
   Stokes--Darcy reference if we later want a closer reproduction of the exact
   Wang discretization,
2. or add one more geometry after submission if a revision requests it.
