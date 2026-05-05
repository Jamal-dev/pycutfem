## Duddu et al. (2007) growth-only validation (Fig. 4 + Fig. 6 / Example 2)

This benchmark reproduces the **growth-only** model from:

> R. Duddu, S. Bordas, D. L. Chopp, B. Moran (2007)  
> “A combined extended finite element and level set method for biofilm growth”

All scripts and outputs live under `examples/biofilms/benchmarks/dadu/` (outputs go to `results/`, which is gitignored).

### 0) (Optional) extract paper page PNGs (for comparisons)

The comparison script auto-crops Fig. 6 panels from the rendered paper page image.

```bash
conda run --no-capture-output -n fenicsx python \
  examples/biofilms/benchmarks/dadu/extract_duddu2007_paper_pages.py \
  --pages 17
```

This writes `examples/biofilms/benchmarks/dadu/results/paper2007_pages/page-17.png` (journal page 864, contains Fig. 6 / Example 2).

### 1) Fig. 4 + Table I (1D slab) — quantitative match

Run:

```bash
conda run --no-capture-output -n fenicsx python \
  examples/biofilms/benchmarks/dadu/reproduce_duddu2007_fig4_table1.py \
  --backend cpp --linear-solver petsc
```

Outputs:
- `examples/biofilms/benchmarks/dadu/results/duddu2007_fig4_table1/summary.csv`
- `examples/biofilms/benchmarks/dadu/results/duddu2007_fig4_table1/fig4_compare.png`

Expected (paper vs reproduced, mm/day) in `summary.csv`:
- `XFEM100` ≈ `0.0093`
- `XFEM200` ≈ `0.0097`
- `FD3000` ≈ `0.0103`

Notes:
- `fig4_compare.png` now also overlays the **one-domain** reduction profiles `S(y)` and `Φ(y)` computed by
  `duddu2007_one_domain_slab_speed.py` (same slab geometry, quasi-steady substrate + potential surrogate).

### 2) Fig. 6 (Example 2, 2D three colonies) — qualitative match

Paper setup (see `examples/biofilms/benchmarks/dadu/dadu_2007.tex`):
- Domain: `0.5 mm × 0.5 mm`
- Initial colonies: radii `[0.01, 0.02, 0.01] mm` centered at `x=[0.05, 0.25, 0.45] mm` on `y=0`
- Substrate Dirichlet boundary `Γ_S^d`: **moving**, `0.1 mm` above the top-most biofilm point, `S̄=8.3e-6 mgO2/mm^3`
- Adaptive time step: `dt = 0.8 * min(dx,dy) / max(F)`
- Zero flux on the outer boundary for `S` and `Φ`

Run (fast, smaller XFEM mesh; same level-set grid as paper):

```bash
conda run --no-capture-output -n fenicsx python -u \
  examples/biofilms/benchmarks/dadu/duddu2007_growth_2d_fig6_example2.py \
  --backend cpp --linear-solver petsc \
  --mesh-nx 20 --mesh-ny 20 \
  --grid-nx 200 --grid-ny 200 \
  --q 2 --speed-mode qp \
  --substrate-bc moving --Ls 0.1 --S-penalty 1e6 \
  --t-final 28.6 --reinit-every 1 \
  --outdir examples/biofilms/benchmarks/dadu/results/duddu2007_fig6_example2_paper_match_qp_m20g200
```

Outputs:
- `.../fig6a_interface.png`
- `.../fig6b_S.png`
- `.../fig6c_Phi.png`
- `.../summary.json`

### 3) Side-by-side paper comparison montage (Fig. 6)

```bash
conda run --no-capture-output -n fenicsx python \
  examples/biofilms/benchmarks/dadu/compare_duddu2007_fig6.py \
  --our-dir examples/biofilms/benchmarks/dadu/results/duddu2007_fig6_example2_paper_match_qp_m20g200 \
  --paper-page page-17.png
```

Output:
- `examples/biofilms/benchmarks/dadu/results/duddu2007_fig6_example2_paper_match_qp_m20g200/compare_fig6.png`

Notes:
- The script auto-crops the paper panels from `results/paper2007_pages/page-17.png`. If your page render is named differently, pass `--paper-page ...`.
- `--speed-mode duddu` is available to use the paper’s Fig. 3 “shaded triangle” evaluation; `--speed-mode qp` computes `F=n·∇Φ` directly at interface quadrature points (more robust for long runs).

### 4) One-domain (diffuse-interface) reproduction — qualitative + y_top comparison

This uses our one-domain formulation in `examples/utils/biofilm/one_domain.py` and a Duddu-like splitting in:
`examples/biofilms/benchmarks/dadu/duddu2007_one_domain_growth_2d_fig6_example2.py`.

Run (growth-only, no detachment; recommended “publishable” mesh):

```bash
conda run --no-capture-output -n fenicsx python -u \
  examples/biofilms/benchmarks/dadu/duddu2007_one_domain_growth_2d_fig6_example2.py \
  --backend cpp --linear-solver petsc \
  --substrate-solver newton \
  --substrate-advection off \
  --alpha-advect-with mix \
  --phi-update mix \
  --D-S 120 --gamma-vS 0.1 --vS-ext-mode l2 \
  --D-alpha 0 \
  --ac-M 1 --ac-gamma 5e-4 --ac-mobility degenerate --ac-mobility-floor 0.1 \
  --q 2 --nx 80 --ny 80 \
  --dt 0.2 --t-final 28.6 \
  --newton-tol 1e-8 --max-it 20 \
  --progress-every 20 --write-every 1 --flush-snaps-every 20 \
  --skip-plots \
  --outdir examples/biofilms/benchmarks/dadu/results/_tune_oneDom_phiMix_alphaMix_DS120_vSextL2_gvS0p1_acM1_g5e-4_mobdeg_floor0p1_nx80_dt0p2_q2_t28p6
```

Outputs:
- `.../y_top_timeseries.csv`
- `.../snaps_alpha.npz`
- `.../summary.json`

Export VTK snapshots (for ParaView inspection) from the stored NPZ snapshots:

```bash
conda run --no-capture-output -n fenicsx python -u \
  examples/biofilms/benchmarks/dadu/export_duddu2007_fig6_vtk.py \
  --run-dir examples/biofilms/benchmarks/dadu/results/_tune_oneDom_phiMix_alphaMix_DS120_vSextL2_gvS0p1_acM1_g5e-4_mobdeg_floor0p1_nx80_dt0p2_q2_t28p6 \
  --every 10
```

This writes VTK files under:
- `.../vtk/alpha_series.pvd` (time series of `alpha`)
- `.../vtk/final_fields.vtu` (final `alpha,S,p`)

Regenerate Fig. 6-style PNG panels from the saved α snapshots:

```bash
conda run --no-capture-output -n fenicsx python \
  examples/biofilms/benchmarks/dadu/plot_one_domain_interface_from_snaps.py \
  --results-dir examples/biofilms/benchmarks/dadu/results/_tune_oneDom_phiMix_alphaMix_DS120_vSextL2_gvS0p1_acM1_g5e-4_mobdeg_floor0p1_nx80_dt0p2_q2_t28p6 \
  --paper-times \
  --out examples/biofilms/benchmarks/dadu/results/_tune_oneDom_phiMix_alphaMix_DS120_vSextL2_gvS0p1_acM1_g5e-4_mobdeg_floor0p1_nx80_dt0p2_q2_t28p6/fig6a_interface.png
```

```bash
conda run --no-capture-output -n fenicsx python -u \
  examples/biofilms/benchmarks/dadu/reconstruct_duddu2007_one_domain_final_fields.py \
  --results-dir examples/biofilms/benchmarks/dadu/results/_tune_oneDom_phiMix_alphaMix_DS120_vSextL2_gvS0p1_acM1_g5e-4_mobdeg_floor0p1_nx80_dt0p2_q2_t28p6 \
  --backend cpp --linear-solver petsc
```

```bash
conda run --no-capture-output -n fenicsx python \
  examples/biofilms/benchmarks/dadu/plot_one_domain_fig6_panels_from_npz.py \
  --results-dir examples/biofilms/benchmarks/dadu/results/_tune_oneDom_phiMix_alphaMix_DS120_vSextL2_gvS0p1_acM1_g5e-4_mobdeg_floor0p1_nx80_dt0p2_q2_t28p6
```

Compute a y_top overlay + error metrics against the XFEM reference:

```bash
conda run --no-capture-output -n fenicsx python \
  examples/biofilms/benchmarks/dadu/compare_duddu2007_fig6_y_top.py \
  --a examples/biofilms/benchmarks/dadu/results/duddu2007_fig6_example2_paper_match_qp_m20g200_y_top \
  --b examples/biofilms/benchmarks/dadu/results/_tune_oneDom_phiMix_alphaMix_DS120_vSextL2_gvS0p1_acM1_g5e-4_mobdeg_floor0p1_nx80_dt0p2_q2_t28p6 \
  --label-a XFEM --label-b one-domain \
  --outdir examples/biofilms/benchmarks/dadu/results/_tune_oneDom_phiMix_alphaMix_DS120_vSextL2_gvS0p1_acM1_g5e-4_mobdeg_floor0p1_nx80_dt0p2_q2_t28p6
```

Expected summary (current best full-length one-domain run):
- MAE ≈ `1.84e-02 mm`
- Final (`t=28.6 d`) abs error ≈ `1.95e-02 mm` (one-domain higher; ~8.6% relative)

Create a 3-column montage (paper | XFEM | one-domain):

```bash
python examples/biofilms/benchmarks/dadu/compare_duddu2007_fig6.py \
  --our-dir examples/biofilms/benchmarks/dadu/results/duddu2007_fig6_example2_paper_match_qp_m20g200 \
  --our-label XFEM \
  --extra-dir examples/biofilms/benchmarks/dadu/results/_tune_oneDom_phiMix_alphaMix_DS120_vSextL2_gvS0p1_acM1_g5e-4_mobdeg_floor0p1_nx80_dt0p2_q2_t28p6 \
  --extra-label one-domain \
  --paper-page page-17.png
```

Output:
- `examples/biofilms/benchmarks/dadu/results/duddu2007_fig6_example2_paper_match_qp_m20g200/compare_fig6.png`
