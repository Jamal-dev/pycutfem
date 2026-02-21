# FSI + relaxed wall contact (semi-smooth Newton)

This folder contains the implementation scaffold for the *bouncing elastic ball* benchmark from
`examples/fsi_contact/paper/main.tex` (Frei et al., ACSE 2025), focusing on the **relaxed unilateral
wall contact** term and its **semi-smooth Newton** linearization.

## Contact formulation (paper Problems 5–7)

Let

- `P(U)` be the Alart–Curnier contact reformulation (paper Eq. (21)),
- `⟨·⟩₊ := max(·,0)` be the positive part,
- `H(·)` be the (hard) Heaviside step (`H(x)=1` if `x>0` else `0`).

Then the contact contribution added to the *residual* is

- `R += γ_C k ∫_{Γ_i} ⟨P(U)⟩₊ (φ_s·n_s) ds`

and the **semi-smooth Newton** contribution added to the *Jacobian* is

- `J += γ_C k ∫_{Γ_i} H(P(U)) P'(U)[δU] (φ_s·n_s) ds`.

In the code this pattern is implemented in `examples/utils/fsi/contact.py` via:

- `RelaxedWallContact.residual_term(...)`  (uses `pos_part(P)`),
- `RelaxedWallContact.jacobian_term(...)`  (uses `heaviside(P) * dP`).

## Reusing the contact term

1. Build `P = contact.P_gammaC(...)` and `dP = contact.dP_gammaC(...)`.
2. Add both pieces **explicitly** (do *not* rely on automatic differentiation of `pos_part`):

```python
residual_form += contact.residual_term(P=P, k=Constant(1.0), test_v_s=phi_s) * dGamma
jacobian_form += contact.jacobian_term(P=P, dP=dP, k=Constant(1.0), test_v_s=phi_s) * dGamma
```

## Consistency checks (FD)

These scripts verify `A(U)·δ ≈ (R(U+ε δ)-R(U))/ε` for **inactive** and **active** contact:

- `python examples/debug/semismooth_contact_fd_check.py`
- `python examples/debug/fsi_contact_full_fd_check.py --backends python,jit,cpp`

## Running the bouncing-ball benchmark

Driver: `examples/fsi_contact/bouncing_ball.py`.

Useful defaults:

- Use `--backend cpp` for long runs.
- Keep `--dt 1e-4` to match the paper’s initial time step.
- Use `--adaptive-dt` to enable the paper-style refine/coarsen heuristic (Sec. 6.4).
- The paper reports quantities relative to the event time `t0` (center at height `h0`).
  In Table 2/3, the absolute “contact time” is `t0 + t_cont` and the rebound peak is `t0 + t_jump`.

Example command (paper-like dt, uniform mesh):

```bash
python examples/fsi_contact/bouncing_ball.py --backend cpp --nx 64 --ny 64 --dt 1e-4 --final-time 0.55 --adaptive-dt
```

### Reproducing Table 2 (uniform mesh levels 0–3)

The paper’s uniform refinement levels correspond to `nx=ny`:

- level 0 (DoFs ≈ 2695): `--nx 16  --ny 16`  (h = 5.0 mm)
- level 1 (DoFs ≈ 9928): `--nx 32  --ny 32`  (h = 2.5 mm)
- level 2 (DoFs ≈ 37660): `--nx 64  --ny 64` (h = 1.25 mm)
- level 3 (DoFs ≈ 146648): `--nx 128 --ny 128` (h = 0.625 mm)

Paper-like time stepping:

- initial `--dt 1e-4`
- `--newton-tol 1e-7`
- run at least to `--final-time 0.55` (Table 2 rebound peak is around `t≈0.51`)

After a run, compare against Table 2 via:

```bash
python examples/fsi_contact/compare_to_paper_table2.py output_bouncing_ball/metrics.json
```

Outputs:

- `output_bouncing_ball/metrics.json`: paper-style quantities of interest (to compare to Table 2/3).
- `output_bouncing_ball/min_dist.csv`: time series for the minimal distance to the bottom wall.

You can control expensive energy evaluations via `--metrics-stride` (set `0` to disable).
For long runs, reduce console output with `--log-stride 100` (or similar).
You can also silence the solver’s per-Newton prints via `--newton-print-level 0` (default).

### VTK output (ParaView)

Use either:

- `--save-vtk` (writes every accepted step), or
- `--vtk-every N` (writes every `N` steps; implies `--save-vtk`)

VTK snapshots are written as `solution_0000.vtu`, `solution_0001.vtu`, … under `--out-dir/vtk/`.

## Notes / current gaps vs the paper

The contact residual/Jacobian and the moving-interface plumbing are implemented, including FD checks.
The full paper setup also includes additional stabilization/extension terms (SUPG + extension ghost
penalties) and an adaptive time-step controller; those may be required for quantitative agreement on
coarser meshes or for robustness near impact.
