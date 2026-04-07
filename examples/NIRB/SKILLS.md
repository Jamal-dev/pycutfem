# Skill: build reusable MOR/NIRB tooling and reproduce the Tiba et al. ROM-FOM paper in pycutfem

## Scope

This skill implements the non-intrusive ROM-FOM workflow from `examples/NIRB/tiba.tex` and turns it into reusable repository-level tooling for future FSI projects.

This is not a one-off example script task. The end state is:
- reusable MOR tools,
- reusable NIRB tools,
- benchmark-specific configs and drivers under `examples/NIRB/`,
- a documented validation pathway.

---

## Environment

Always run code through:

```bash
conda run --no-capture-output -n fenicsx python
````

Examples:

```bash
conda run --no-capture-output -n fenicsx python -m pytest -q
conda run --no-capture-output -n fenicsx python -m pycutfem.nirb.train --config examples/NIRB/configs/example1.yaml
conda run --no-capture-output -n fenicsx python -m pycutfem.nirb.validate --config examples/NIRB/configs/example2_rom4.yaml
```

---

## Paper-to-code map

Implement the following code path directly from the paper.

### Reduced input encoding

Use a POD/PCA encoder for the interface loading:

* Eq. (10)–(13)
* compute SVD of the input snapshot matrix
* keep `r_f` modes
* project to reduced coordinates with a linear orthogonal projection

### Reduced output decoding

Use a quadratic manifold decoder for the displacement field:

* Eq. (14)–(17)
* keep `r_u` displacement modes
* build quadratic features from reduced displacement coordinates
* fit the quadratic correction by least squares

### Interface-only online exchange

Use an interface restriction operator:

* Eq. (19)–(20)
* avoid reconstructing the full displacement at every non-converged subiteration
* reconstruct the full field only when needed for post-processing

### Regression in reduced space

Use a reduced latent-space regressor:

* Eq. (21)
* Example 1: thin-plate-spline RBF
* Example 2: degree-2 polynomial regression with Lasso sparsification and BIC/LARS model selection

### Validation metrics

Implement:

* Eq. (26): Example 1 held-out validation error
* Eq. (30): Example 2 validation error
* Eq. (31): Example 2 regression error in reduced space
* Eq. (34): Example 2 online relative displacement error

---

## Required repository-level modules

The implementation should provide equivalent capabilities to the following.

## `pycutfem.mor`

### `snapshots.py`

Responsibilities:

* snapshot container
* append/store/reload
* save metadata per sample:

  * parameter
  * time
  * subiteration index
  * converged flag
  * timings
  * interface force
  * full displacement

Suggested objects:

* `SnapshotBatch`
* `SnapshotWriter`
* `SnapshotReader`

### `scaling.py`

Responsibilities:

* mean subtraction
* optional standardization
* consistent train/infer transforms

Suggested objects:

* `MeanCenterer`
* `StandardScaler`

### `pod.py`

Responsibilities:

* SVD/POD basis fitting
* energy-based truncation
* fixed-rank truncation
* projection and reconstruction

Suggested objects:

* `PODBasis`
* `fit_pod`
* `project`
* `reconstruct`

### `quadratic_manifold.py`

Responsibilities:

* deterministic quadratic term ordering
* quadratic feature assembly
* least-squares fit of quadratic correction
* linear + quadratic reconstruction

Required convention:
Use a deterministic upper-triangular ordering such as:

```python
[(i, j) for i in range(r_u) for j in range(i, r_u)]
```

Suggested objects:

* `QuadraticFeatureMap`
* `QuadraticManifoldDecoder`
* `fit_quadratic_decoder`

### `interface.py`

Responsibilities:

* assemble the restriction operator `K`
* restrict full-field displacement basis to interface
* support direct online interface reconstruction

Suggested objects:

* `InterfaceRestriction`
* `build_restriction_matrix`

### `regressors.py`

Responsibilities:

* thin-plate-spline RBF
* polynomial feature generation
* Lasso/LARS/BIC selection
* save/load fitted regression models

Suggested objects:

* `ThinPlateSplineRBF`
* `PolynomialLassoRegressor`
* `fit_tps_rbf`
* `fit_poly_lasso`

### `cross_validation.py`

Responsibilities:

* held-out split generation
* mode sweep
* error surface generation
* best-hyperparameter selection

Suggested objects:

* `ModeSweepResult`
* `run_mode_cross_validation`

### `metrics.py`

Responsibilities:

* Eq. (26), (30), (31), (34)
* coupling iteration overhead
* speedup computation
* pass/fail summary

### `timing.py`

Responsibilities:

* timing context managers
* solver timing aggregation
* speedup reports

### `io.py`

Responsibilities:

* save/load trained models
* save/load configs
* save/load result summaries

---

## `pycutfem.nirb`

### `dataset.py`

Responsibilities:

* benchmark config to dataset generation
* offline dataset metadata
* split between train/validation/online-eval

### `offline.py`

Responsibilities:

* fit all offline components
* store trained basis and regressors
* store CV results
* store selected mode counts

### `online.py`

Responsibilities:

* online ROM-FOM evaluation
* interface-only prediction
* optional delayed full-field reconstruction

### `coupling.py`

Responsibilities:

* glue between the fluid FOM and solid ROM
* logging of coupling iterations
* preserve the original partitioned flow where possible

### `validation.py`

Responsibilities:

* generate metric tables
* generate plots
* compute pass/fail
* write machine-readable report

### `cli.py`

Required CLI behavior:

* `python -m pycutfem.nirb.train --config ...`
* `python -m pycutfem.nirb.run --config ...`
* `python -m pycutfem.nirb.validate --config ...`

---

## Benchmark implementation details

## Example 1 — 1D arterial toy model

### Required setup

Reproduce the benchmark with:

* training parameter `mu1 = (2, 6)^T`
* evaluation parameter `muhat = (0.9, 4)^T`
* `T = 18 s`
* `Tf = 120 s`
* `Ti = 0.1 s`
* `m = 737`
* `N = Nu = 101`

### Offline training path

1. Run the FOM-FOM simulation.
2. Collect snapshots from all coupling subiterations.
3. Build the input snapshot matrix and output snapshot matrix.
4. Remove mean fields before fitting POD.
5. Perform 20% held-out validation over mode counts.
6. Recover a best region near:

   * 3 input/pressure modes
   * 9 output/section modes

### Regression

Use thin-plate-spline RBF:

```text
phi(x) = x^2 log(x)
```

### Validation targets

The Example 1 implementation must produce:

* a mode-selection error map,
* a curve like Figure 4,
* future-time prediction at `mu1`,
* prediction at `muhat`,
* nonlinear stress-strain reconstruction,
* coupling-iteration comparison.

Target metrics:

* held-out mean L2 error <= `5e-5`
* iteration overhead <= `15%`
* solid-only speedup >= `80x`

---

## Example 2 — cylinder wake with elastic flaps

### Required setup

Reproduce the benchmark with:

* `rho = 1000`
* `nu_f = 0.001`
* `vmax = 2.5`
* `dt = 0.008`
* training set `P = {178.6, 208.3, 250}`
* domain `D = [178.6, 250]`
* `Tf = 6 s`
* `T = 3 s`
* `Ti = 0.8 s`
* `N = 530`
* `Nu = 3610`
* total offline snapshots = `6890`

### Offline training path

1. Generate or read the three training simulations.
2. Collect snapshots from all coupling subiterations.
3. Use 9 displacement modes.
4. Use 5% held-out data for input-mode cross-validation.
5. The best input-mode counts should land near:

   * ROM 1 = 45
   * ROM 2 = 40
   * ROM 3 = 50
   * ROM 4 = 45

### Regression

Use a degree-2 polynomial regressor:

* include constant term,
* include linear terms,
* include pairwise interaction terms,
* use Lasso sparsification,
* select lambda from a BIC-minimizing point along the LARS path.

Important:

* standardize latent input coordinates before Lasso unless a strong reason not to,
* save the scaler together with the regressor,
* keep the polynomial feature order deterministic.

### Validation targets

The Example 2 implementation must produce:

* singular-value decay plot,
* held-out mode-selection curves,
* sparse polynomial-term visualization,
* left/right tip displacement trajectories,
* phase-space comparison,
* error curves,
* coupling-iteration comparison,
* timing ratio histogram,
* speedup table.

Target metrics:

* `max_t e(t) <= 0.05` for ROM 1, 2, 3
* `max_t e(t) <= 0.05` for ROM 4 at `Re = 192.3`
* `max_t e(t) <= 0.10` for ROM 4 at `Re = 300`
* iteration overhead <= `7%` at `Re = 300`
* iteration overhead <= `10%` for the other Example 2 cases
* average solid speedup >= `180x`
* overall speedup >= `1.6x`

---

## Validation workflow

## Step 1 — Verify offline data integrity

Before training:

* check snapshot counts,
* check shape consistency,
* check parameter labels,
* check time ordering,
* check subiteration ordering,
* verify that interface forces are post-transfer and in the correct space,
* verify full displacement field shape.

Add assertions and tests for all of the above.

## Step 2 — Validate the MOR blocks independently

Before online coupling:

* test POD reconstruction on held-out samples,
* test quadratic manifold reconstruction,
* test regression on training data,
* test interface-only reconstruction against full-field restriction.

## Step 3 — Validate the online coupling

For each benchmark:

* run FOM-FOM reference,
* run ROM-FOM evaluation,
* compute time histories,
* compute error curves,
* compute coupling iteration overhead,
* compute speedups.

## Step 4 — Write pass/fail report

Write:

```text
examples/NIRB/results/<run_name>/metrics.json
examples/NIRB/results/<run_name>/summary.md
```

The report must include:

* configuration,
* mode counts,
* regressor settings,
* all measured metrics,
* pass/fail booleans,
* location of generated plots.

---

## Debugging playbook

When the metrics are not met, use this order.

### 1. Snapshot issues

Symptoms:

* unstable training,
* strange singular values,
* inconsistent shapes,
* poor online accuracy.

Checks:

* all coupling subiterations collected,
* correct interface force,
* correct displacement field,
* correct mapping direction.

### 2. Preprocessing issues

Symptoms:

* wrong optimum mode count,
* bad regression conditioning.

Checks:

* mean subtraction applied consistently,
* scaling saved and reused,
* deterministic train/validation split,
* consistent latent coordinate ordering.

### 3. POD issues

Symptoms:

* poor reduced reconstruction,
* mismatch with singular-value decay.

Checks:

* basis orthonormality,
* retained energy,
* reconstruction residual.

### 4. Quadratic manifold issues

Symptoms:

* good linear POD but poor full-field output,
* mismatch between interface and full field.

Checks:

* quadratic feature ordering,
* orthogonality to linear POD space,
* least-squares solver stability.

### 5. Regression issues

Symptoms:

* reduced coordinates predict poorly,
* online instability.

Checks:

* Example 1 uses TPS RBF, not polynomial Lasso,
* Example 2 uses degree-2 polynomial features,
* lambda chosen by BIC/LARS,
* latent standardization.

### 6. Coupling issues

Symptoms:

* too many fixed-point iterations,
* divergence,
* speedup worse than expected.

Checks:

* interface restriction correctness,
* coupling tolerance,
* quasi-Newton/fixed-point path,
* delayed full-field reconstruction.

### 7. FOM issues

Symptoms:

* reference trajectory wrong,
* impossible validation gap.

Checks:

* BCs,
* parameters,
* time step,
* mesh,
* constitutive law.

Every resolved issue needs a test.

---

## Tests to add

## Unit tests

Add tests for:

* POD orthonormality
* POD reconstruction
* energy criterion truncation
* quadratic feature ordering
* quadratic manifold fit
* interface restriction matrix correctness
* TPS RBF exact interpolation on training points
* polynomial feature builder
* Lasso/LARS/BIC selector
* Eq. (26), (30), (31), (34) metrics
* speedup calculations

## Integration tests

Add tests for:

* snapshot batch -> offline training -> online prediction
* save/load trained offline model
* interface-only online prediction path
* small ROM-FOM coupling mock with deterministic data

## Regression tests

If a bug is found in the main repo:

* add a minimal regression pytest,
* keep it in CI,
* rerun the relevant example.

---

## Recommended artifact structure

Use a structure similar to:

```text
examples/NIRB/
  configs/
    example1.yaml
    example2_rom1.yaml
    example2_rom2.yaml
    example2_rom3.yaml
    example2_rom4.yaml
  data/
  results/
  run_example1_fom.py
  run_example2_fom.py
  train_example1.py
  train_example2.py
  validate_example1.py
  validate_example2.py
  investigation.md
```

The example scripts should call repo-level APIs, not embed the math directly.

---

## Final check before declaring success

Do not stop until all answers below are yes:

* Are MOR tools reusable at the repo level?
* Are NIRB tools reusable at the repo level?
* Does `examples/NIRB/` use those repo-level tools?
* Do Example 1 metrics pass?
* Do Example 2 metrics pass?
* Are tests present and passing?
* Does every fixed bug have a pytest?
* Is there a machine-readable validation report?

````

`examples/NIRB/investigation.md`
```md
# Investigation plan: reproduce the Tiba et al. ROM-FOM/NIRB results in pycutfem

## Objective

Reproduce the paper workflow described in `examples/NIRB/tiba.tex` inside this repository, using reusable repo-level MOR and NIRB tooling rather than one-off scripts.

This investigation has two simultaneous goals:

1. Reproduce the two paper benchmarks.
2. Extend the repository so the same MOR/NIRB machinery can be reused in future FSI projects.

This task is not complete until the acceptance metrics are achieved and written to a machine-readable report.

---

## Source of truth

Use:
- `examples/NIRB/tiba.tex`
- the equations and numerical settings from the paper
- the repo-level agent and skill instructions

---

## What must exist when this investigation is done

## Repo-level reusable capabilities
The repo must contain reusable:
- POD/PCA tools
- quadratic-manifold tools
- interface restriction tools
- non-intrusive regression backends
- offline/online NIRB pipeline tools
- validation and timing tools

## Example-level runnable workflows
`examples/NIRB/` must contain:
- configs
- FOM generation scripts or drivers
- offline training scripts
- online evaluation scripts
- validation scripts
- generated plots and reports

## Tests
There must be:
- unit tests for MOR/NIRB components
- integration tests
- regression tests for any bug found in the main repo

---

## Paper-derived benchmark targets

## Example 1
Training/evaluation settings:
- training parameter `mu1 = (2, 6)^T`
- extra evaluation parameter `muhat = (0.9, 4)^T`
- `T = 18 s`
- `Tf = 120 s`
- `Ti = 0.1 s`
- `m = 737`
- `N = Nu = 101`

Acceptance targets:
- best mode region near 3 input modes and 9 output modes
- held-out mean L2 error <= `5e-5`
- accumulated coupling-iteration overhead <= `15%`
- solid-only speedup >= `80x`

## Example 2
Training/evaluation settings:
- `P = {178.6, 208.3, 250}`
- `D = [178.6, 250]`
- `T = 3 s`
- `Tf = 6 s`
- `Ti = 0.8 s`
- `dt = 0.008 s`
- `N = 530`
- `Nu = 3610`
- total snapshots = `6890`

Acceptance targets:
- 9 displacement modes
- force modes near:
  - ROM 1 = 45
  - ROM 2 = 40
  - ROM 3 = 50
  - ROM 4 = 45
- `max_t e(t) <= 0.05` for ROM 1, 2, 3
- `max_t e(t) <= 0.05` for ROM 4 at `Re = 192.3`
- `max_t e(t) <= 0.10` for ROM 4 at `Re = 300`
- iteration overhead <= `7%` at `Re = 300`
- iteration overhead <= `10%` for the other Example 2 runs
- average solid speedup >= `180x`
- overall speedup >= `1.6x`

---

## Required repo-level design

The reproduction must feed back into the main package.

### Required package expansion
Create or extend repo-level modules equivalent to:

```text
pycutfem/
  mor/
    snapshots.py
    scaling.py
    pod.py
    quadratic_manifold.py
    interface.py
    regressors.py
    cross_validation.py
    metrics.py
    timing.py
    io.py
  nirb/
    dataset.py
    offline.py
    online.py
    coupling.py
    validation.py
    cli.py
````

### Reuse rule

Core math and pipeline code belongs at repo level.
Only benchmark-specific setup belongs under `examples/NIRB/`.

---

## Investigation phases

## Phase 1 — Map the paper to code

* [ ] Read `examples/NIRB/tiba.tex`
* [ ] Create a math-to-code map for Eqs. (10)–(21), (23)–(34)
* [ ] Identify existing repo modules that can be reused
* [ ] Identify missing repo-level capabilities

Deliverable:

* short design note or code comments documenting the mapping

## Phase 2 — Build reusable MOR/NIRB tools

* [ ] Implement snapshot I/O
* [ ] Implement POD/PCA tools
* [ ] Implement quadratic feature builder
* [ ] Implement quadratic manifold decoder
* [ ] Implement interface restriction operator
* [ ] Implement TPS RBF regressor
* [ ] Implement polynomial degree-2 + Lasso/BIC regressor
* [ ] Implement metric functions
* [ ] Implement timing/speedup utilities
* [ ] Implement offline/online pipeline entry points

Deliverable:

* importable repo-level package APIs

## Phase 3 — Example 1 FOM and offline path

* [ ] Create the Example 1 config
* [ ] Generate or load FOM-FOM data
* [ ] Collect all coupling-subiteration snapshots
* [ ] Perform 20% held-out cross-validation
* [ ] Recover near-optimal mode counts
* [ ] Train the TPS RBF reduced regressor

Deliverables:

* saved offline model
* held-out error surface
* selected mode counts

## Phase 4 — Example 1 online validation

* [ ] Run future-time prediction at `mu1`
* [ ] Run evaluation at `muhat`
* [ ] Compute Eq. (26)
* [ ] Compute iteration overhead
* [ ] Compute solid-only speedup
* [ ] Plot pressure trace
* [ ] Plot stress-strain reconstruction

Stop condition:

* Example 1 metrics are green

## Phase 5 — Example 2 FOM and offline path

* [ ] Create configs for ROM 1–4
* [ ] Generate or load training data for `Re = 178.6, 208.3, 250`
* [ ] Collect all coupling-subiteration snapshots
* [ ] Run 5% held-out cross-validation
* [ ] Fit 9 displacement modes
* [ ] Select force modes near 45/40/50/45
* [ ] Fit polynomial degree-2 reduced regressor
* [ ] Select lambda with LARS/BIC
* [ ] Save sparse model and offline artifacts

Deliverables:

* singular-value plots
* CV curves
* sparse-term visualizations

## Phase 6 — Example 2 online validation

* [ ] Run ROM 1, ROM 2, ROM 3 on seen parameters
* [ ] Run ROM 4 at `Re = 192.3`
* [ ] Run ROM 4 at `Re = 300`
* [ ] Compute Eq. (34)
* [ ] Compute iteration overhead
* [ ] Compute solid speedup
* [ ] Compute overall speedup
* [ ] Generate displacement, phase-space, iteration, and speedup plots

Stop condition:

* Example 2 metrics are green

## Phase 7 — Testing and hardening

* [ ] Add unit tests
* [ ] Add integration tests
* [ ] Add regression tests for any bug found
* [ ] Run the full test suite
* [ ] Verify saved-model reload and replay
* [ ] Verify command-line reproducibility

Stop condition:

* tests are green and metrics are green

---

## Commands to standardize

Use these command shapes.

## Tests

```bash
conda run --no-capture-output -n fenicsx python -m pytest -q
```

## Example 1

```bash
conda run --no-capture-output -n fenicsx python -m pycutfem.nirb.train --config examples/NIRB/configs/example1.yaml
conda run --no-capture-output -n fenicsx python -m pycutfem.nirb.run --config examples/NIRB/configs/example1.yaml
conda run --no-capture-output -n fenicsx python -m pycutfem.nirb.validate --config examples/NIRB/configs/example1.yaml
```

## Example 2

```bash
conda run --no-capture-output -n fenicsx python -m pycutfem.nirb.train --config examples/NIRB/configs/example2_rom1.yaml
conda run --no-capture-output -n fenicsx python -m pycutfem.nirb.train --config examples/NIRB/configs/example2_rom2.yaml
conda run --no-capture-output -n fenicsx python -m pycutfem.nirb.train --config examples/NIRB/configs/example2_rom3.yaml
conda run --no-capture-output -n fenicsx python -m pycutfem.nirb.train --config examples/NIRB/configs/example2_rom4.yaml

conda run --no-capture-output -n fenicsx python -m pycutfem.nirb.run --config examples/NIRB/configs/example2_rom4.yaml
conda run --no-capture-output -n fenicsx python -m pycutfem.nirb.validate --config examples/NIRB/configs/example2_rom4.yaml
```

If the CLI does not yet exist, create it at the repo level.

---

## Required artifacts

## Offline artifacts

Store under:

```text
examples/NIRB/results/<run_name>/
```

Expected artifacts:

* `offline_model.*`
* `cv_results.json`
* `singular_values.png`
* `mode_sweep.png`
* `lasso_terms.png`
* `config_used.yaml`

## Online artifacts

Expected artifacts:

* `tip_displacement_left.png`
* `tip_displacement_right.png`
* `phase_space.png`
* `error_curve.png`
* `coupling_iterations.png`
* `speedup_histogram.png`

## Summary artifacts

Expected artifacts:

* `metrics.json`
* `summary.md`

---

## Validation table to fill in during the investigation

## Example 1

| Metric                 |  Target | Achieved | Pass |
| ---------------------- | ------: | -------: | ---- |
| input modes            |      ~3 |          |      |
| output modes           |      ~9 |          |      |
| held-out mean L2 error | <= 5e-5 |          |      |
| iteration overhead     |  <= 15% |          |      |
| solid-only speedup     |  >= 80x |          |      |

## Example 2

| Metric                          |  Target | Achieved | Pass |
| ------------------------------- | ------: | -------: | ---- |
| displacement modes              |       9 |          |      |
| ROM 1 force modes               |     ~45 |          |      |
| ROM 2 force modes               |     ~40 |          |      |
| ROM 3 force modes               |     ~50 |          |      |
| ROM 4 force modes               |     ~45 |          |      |
| ROM 1 max error                 | <= 0.05 |          |      |
| ROM 2 max error                 | <= 0.05 |          |      |
| ROM 3 max error                 | <= 0.05 |          |      |
| ROM 4 max error at 192.3        | <= 0.05 |          |      |
| ROM 4 max error at 300          | <= 0.10 |          |      |
| ROM 4 iteration overhead at 300 |   <= 7% |          |      |
| other iteration overheads       |  <= 10% |          |      |
| average solid speedup           | >= 180x |          |      |
| overall speedup                 | >= 1.6x |          |      |

This table must be mirrored in `metrics.json` and `summary.md`.

---

## Bug and test policy

If the investigation finds a bug in the main repo:

* [ ] isolate the bug
* [ ] write a failing pytest if feasible
* [ ] fix the code
* [ ] add or keep a regression pytest
* [ ] rerun affected experiments

Never close the investigation with an untested bugfix.

---

## Common failure modes to check first

1. Wrong snapshot space:

   * interface force collected before transfer instead of after transfer
   * wrong displacement field shape
2. Missing mean subtraction
3. Wrong quadratic-feature ordering
4. Mismatch between full-field and interface-only reconstruction
5. Wrong regressor for the benchmark
6. No latent standardization before Lasso
7. Wrong train/validation split
8. Wrong `Ti`, `T`, or `Tf`
9. Speed measured inconsistently
10. Coupling iteration count logged incorrectly

Each confirmed failure mode should produce a test.

---

## Completion rule

This investigation is complete only when:

* the repo-level MOR/NIRB tools exist,
* the example-level reproduction path is documented and runnable,
* all quantitative targets pass,
* all new tests pass,
* every discovered repo bug has a regression pytest.

Until then, continue iterating.

```
```
