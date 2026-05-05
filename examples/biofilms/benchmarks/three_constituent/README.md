# Three-Constituent Paper Suite

This directory contains the scripts used to regenerate the test cases reported in the three-constituent one-domain paper.

## Environment

Run from the repository root with the `fenicsx` conda environment:

```bash
conda run --no-capture-output -n fenicsx python --version
```

## Smoke Suite

Use this for a fast reproducibility check:

```bash
conda run --no-capture-output -n fenicsx python \
  examples/biofilms/benchmarks/three_constituent/run_paper_benchmarks.py \
  --suite smoke \
  --outdir /tmp/pycutfem_three_constituent_paper_smoke
```

## Full Paper Suite

Use this to regenerate the full paper test-case outputs:

```bash
conda run --no-capture-output -n fenicsx python \
  examples/biofilms/benchmarks/three_constituent/run_paper_benchmarks.py \
  --suite paper \
  --outdir /tmp/pycutfem_three_constituent_paper_full
```

The deformable-layer case is the expensive run. To regenerate tables and figures from an already completed production run, pass its output directory:

```bash
conda run --no-capture-output -n fenicsx python \
  examples/biofilms/benchmarks/three_constituent/run_paper_benchmarks.py \
  --suite paper \
  --reuse-seboldt /tmp/pycutfem_tc_seboldt_T3_outflow_if8_nx32_20260504_160251 \
  --outdir /tmp/pycutfem_three_constituent_paper_full
```

## Main Entrypoints

- `paper1_benchmark1_mms.py`: full nine-field manufactured convergence test.
- `paper1_benchmark3_moving_support_mms.py`: full nine-field moving-support manufactured test.
- `paper1_physical_benchmarks_2_to_5.py`: small physical limit and regression tests.
- `stoter_physical.py`: fixed-bed diffuse Stokes-Darcy interface-profile test.
- `seboldt_physical.py`: flow-driven deformable porous-layer production test.
- `run_paper_benchmarks.py`: runner that collects the paper suite and writes manifests.

## Notes

The manuscript uses the term "test case" for the scientific results. Some script and directory names still use historical "benchmark" wording for compatibility with previous runs.
