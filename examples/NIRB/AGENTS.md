# Agent instructions for reproducing the Tiba et al. ROM-FOM/NIRB results in pycutfem

## Mission

Implement, validate, and harden a reusable non-intrusive reduced-order modeling workflow in this repository, using `examples/NIRB/tiba.tex` as the source of truth for the method and targets.

The immediate goal is to reproduce the two benchmarks from the paper inside `examples/NIRB/`, but the implementation must not remain example-local. The repository must gain reusable **MOR** and **NIRB** capabilities at the repo level so the same tools can be reused in other FSI projects.

The agent is **not done** until:
1. the reusable repo-level MOR/NIRB tooling exists,
2. the `examples/NIRB` reproduction path is executable end-to-end,
3. the validation report shows the required error and speed metrics,
4. all new code paths are covered by tests,
5. any bug found in the main repo has a regression pytest.

---

## Non-negotiable working rules

### Execution environment
Always run Python with:

```bash
conda run --no-capture-output -n fenicsx python ....
````

Examples:

```bash
conda run --no-capture-output -n fenicsx python -m pytest -q
conda run --no-capture-output -n fenicsx python examples/NIRB/run_example1_fom.py
conda run --no-capture-output -n fenicsx python -m pycutfem.nirb.train --config examples/NIRB/configs/example2_rom4.yaml
```

### Source of truth

Use the following as authoritative:

1. `examples/NIRB/tiba.tex`
2. the equations and numerical targets in the paper
3. the acceptance criteria in this file

Do not silently replace the method with a different ROM pipeline.

### Persistence rule

Do not stop at the first working script, first plot, or first partial reproduction.

If the metrics are not achieved:

* inspect the data path,
* inspect the interface transfer,
* inspect mean subtraction/scaling,
* inspect basis construction,
* inspect quadratic manifold assembly,
* inspect regressor fitting,
* inspect solver tolerances,
* inspect the partitioned coupling logic,
* rerun the experiments,
* keep iterating until the metrics are met.

Do not declare success while the acceptance thresholds are red.

### Bug rule

If a bug is found in the main repo:

1. isolate it,
2. add a minimal failing pytest when possible,
3. fix it,
4. add or keep the regression test,
5. rerun the affected experiments.

Never leave a bug fixed but untested.

---

## Repo-level extension requirement

The reproduction must extend the repository itself. Do **not** bury MOR/NIRB logic only under `examples/NIRB/`.

### Required outcome

Create reusable repository-level modules for:

* reduced basis / POD / PCA
* quadratic manifold decoding
* non-intrusive regression backends
* interface restriction/lifting operators
* snapshot collection and serialization
* validation metrics
* experiment logging
* FSI ROM-FOM coupling hooks

### Preferred repo-level layout

Adapt to the existing package layout, but the repository must end up with a reusable structure equivalent to:

```text
pycutfem/
  mor/
    __init__.py
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
    __init__.py
    dataset.py
    offline.py
    online.py
    coupling.py
    validation.py
    cli.py
```

If the repository already has related modules, extend them instead of duplicating functionality.

### Reusability requirement

The implementation must work for this paper **and** be reusable for later FSI problems. That means:

* no hard-coded benchmark-specific dimensions inside core modules,
* no benchmark-specific file parsing inside core modules,
* no benchmark-specific regression logic inside core modules,
* use data classes / configs / typed APIs,
* expose clean callable functions or classes.

### Minimum reusable APIs

The repo should expose APIs equivalent in spirit to:

* `fit_pod(X, n_modes=None, energy=None, center=True)`
* `project_to_basis(X, basis, mean=None)`
* `reconstruct_from_basis(coeffs, basis, mean=None)`
* `fit_quadratic_manifold(U, U_red, basis)`
* `build_interface_restriction(interface_dofs, full_dofs)`
* `fit_tps_rbf(F_red, U_red, ...)`
* `fit_poly_lasso(F_red, U_red, degree=2, ...)`
* `run_offline_pipeline(config)`
* `run_online_pipeline(config)`
* `validate_rom(config, results_dir)`

The exact names may differ, but these capabilities must exist.

---

## Mandatory deliverables

The agent must create and maintain:

### 1. Repo-level MOR/NIRB implementation

Reusable code under the repository package, not just scripts.

### 2. Example-level reproduction path

Under `examples/NIRB/`, create:

* configs,
* run scripts,
* validation scripts,
* metric report generators,
* plot generators,
* readme/investigation notes.

### 3. Validation artifacts

For each benchmark, generate:

* mode-selection curves,
* training/validation metrics,
* online trajectory plots,
* coupling iteration counts,
* speedup tables,
* machine-readable metrics (`json` or `yaml`).

### 4. Tests

Add:

* unit tests for reusable MOR/NIRB modules,
* integration tests for small offline/online pipelines,
* regression tests for any bug discovered.

### 5. Final validation report

A single summary artifact that states:

* exact command used,
* git commit,
* configuration,
* achieved metrics,
* pass/fail for each acceptance criterion.

---

## Required implementation path

## Phase 0 — Read the paper and map equations to code

Read `examples/NIRB/tiba.tex` and map the paper equations to concrete code components.

The minimum mapping must include:

* Eq. (10)–(13): POD/PCA encoder for input/interface forces
* Eq. (14)–(17): quadratic manifold decoder for displacement
* Eq. (19)–(20): interface restriction operator
* Eq. (21): regression operator in reduced space
* Eq. (26), (30), (31), (34): validation metrics
* Eq. (23)–(28): Example 1 model and regression choice
* Eq. (29): Example 2 solid constitutive law
* Section 4 + Appendix A: mode counts, validation settings, and targets

Do not implement before this mapping is explicit in code comments, docs, or a design note.

## Phase 1 — Discover the repo capabilities

Before writing major code:

* inspect the repo layout,
* inspect solver entry points,
* inspect existing FSI and finite-element capabilities,
* inspect current testing structure,
* inspect packaging/import conventions.

If the repo already provides some of the needed pieces, reuse them.

## Phase 2 — Build the reusable MOR/NIRB core

Implement reusable repo-level tooling for:

* snapshot storage,
* basis fitting,
* reduced coordinate projection,
* quadratic term generation,
* quadratic manifold fitting,
* interface restriction,
* regression backends,
* metric computation,
* timing and iteration logging,
* offline/online orchestration.

## Phase 3 — Implement Example 1

Implement the 1D toy arterial benchmark reproduction path:

* FOM data generation
* offline snapshot collection
* cross-validation for mode selection
* thin-plate-spline RBF regression
* ROM-FOM online coupling
* metric computation and plot generation

## Phase 4 — Implement Example 2

Implement the cylinder + elastic flaps benchmark reproduction path:

* FOM data generation
* offline snapshot collection for all required Reynolds numbers
* 5% held-out cross-validation
* 9 displacement modes
* polynomial degree-2 regression
* Lasso/LARS/BIC model selection
* ROM-FOM online coupling
* error, stability, and speed evaluation

## Phase 5 — Validate and iterate

Run the full validation suite.

If any acceptance threshold fails:

* keep debugging,
* adjust implementation details,
* improve scaling/regularization/coupling behavior,
* add tests for every discovered failure mode,
* rerun until green.

---

## Acceptance criteria

These are the stopping conditions. The agent must not stop early.

### Benchmark A — Example 1 (1D arterial toy problem)

Required reproduction behavior:

* training case: `mu1 = (2, 6)^T`
* training horizon: `T = 18 s`
* future prediction horizon: `Tf = 120 s`
* second evaluation parameter: `muhat = (0.9, 4)^T`
* use the same offline/online split logic as the paper

Required quantitative outcomes:

1. Cross-validation must recover an optimum consistent with the paper:

   * around **3 input/pressure modes**
   * around **9 output/section modes**
2. Held-out validation error for Eq. (26) must reach:

   * **mean L2 error <= 5e-5**
3. ROM-FOM coupling stability must reach:

   * **accumulated iteration overhead <= 15%**
4. Structural ROM speedup must reach:

   * **solid-only speedup >= 80x**
5. The pipeline must reproduce:

   * inlet pressure trace agreement,
   * nonlinear stress-strain reconstruction,
   * future-time prediction for both parameter settings.

If the exact optimal mode pair differs because of discretization details, the agent may accept a nearby pair only if the error is **equal or lower** than the target and the report explains the difference.

### Benchmark B — Example 2 (cylinder wake + elastic flaps)

Required reproduction behavior:

* training set: `P = {178.6, 208.3, 250}`
* parameter hull: `D = [178.6, 250]`
* training horizon: `T = 3 s`
* online horizon: `Tf = 6 s`
* startup FOM window: `Ti = 0.8 s`
* ROM 1, 2, 3 = time prediction at seen parameters
* ROM 4 = time-parameter prediction at `Re = 192.3` and `Re = 300`

Required quantitative outcomes:

1. Displacement basis size:

   * **9 displacement modes**
2. Force basis sizes must be consistent with Appendix A:

   * **ROM 1 ~= 45**
   * **ROM 2 ~= 40**
   * **ROM 3 ~= 50**
   * **ROM 4 ~= 45**
     A nearby value is acceptable only if the validation metric is equal or better.
3. Online error target for Eq. (34):

   * **max_t e(t) <= 0.05** for ROM 1, 2, 3
   * **max_t e(t) <= 0.05** for ROM 4 at `Re = 192.3`
   * **max_t e(t) <= 0.10** for ROM 4 at `Re = 300`
4. Coupling stability target:

   * **iteration overhead <= 7%** for ROM 4 at `Re = 300`
   * **iteration overhead <= 10%** for the other Example 2 runs
5. Speed targets:

   * **average solid speedup >= 180x**
   * **overall speedup >= 1.6x**
6. Reproduce the evidence class from the paper:

   * mode-selection curves,
   * sparse polynomial term pattern,
   * left/right tip displacement trajectories,
   * phase-space agreement,
   * coupling-iteration comparison,
   * speedup table.

### Global acceptance

The task is complete only when:

* all required metrics are green,
* all required tests pass,
* the example commands are documented,
* the repo-level MOR/NIRB APIs are reusable.

---

## Validation protocol

### Data collection requirements

At a minimum, the offline dataset must capture:

* every coupling subiteration used by the paper,
* interface force after any mesh/interface transfer,
* full solid displacement field,
* converged/non-converged iteration status,
* solver times per subiteration,
* number of fixed-point iterations,
* parameter value,
* time step index.

### Mandatory computed metrics

Compute and store:

* Eq. (26): Example 1 held-out validation error
* Eq. (30): Example 2 ROM validation error
* Eq. (31): Example 2 reduced-space regression error
* Eq. (34): Example 2 online relative displacement error
* accumulated coupling iteration overhead
* solid-only speedup
* end-to-end speedup

### Required machine-readable report

Write a metrics file such as:

```text
examples/NIRB/results/<run_name>/metrics.json
```

It must contain:

* benchmark name,
* config hash,
* git commit hash,
* mode counts,
* regression settings,
* all metrics,
* pass/fail booleans for every acceptance criterion.

---

## Debugging priorities when metrics fail

Use this order:

1. Check that the collected snapshots match the paper:

   * all coupling subiterations included,
   * correct interface quantity,
   * correct mapping direction,
   * correct full displacement field.
2. Check preprocessing:

   * mean subtraction,
   * scaling,
   * deterministic feature ordering,
   * train/test split.
3. Check POD:

   * singular values,
   * retained energy,
   * projection/reconstruction accuracy.
4. Check quadratic manifold:

   * quadratic term ordering,
   * orthogonality to POD space,
   * least-squares fit.
5. Check regression:

   * TPS kernel for Example 1,
   * degree-2 polynomial features for Example 2,
   * Lasso regularization path,
   * BIC selection,
   * latent feature standardization.
6. Check coupling:

   * interface restriction,
   * online reconstruction,
   * convergence criteria,
   * quasi-Newton/fixed-point update path.
7. Check FOM:

   * boundary conditions,
   * material law,
   * time step,
   * mesh,
   * parameter values.

When a failure mode is found, add a test.

---

## Required test policy

### Unit tests

Create unit tests for:

* POD fit/projection/reconstruction
* energy criterion
* quadratic term generation order
* quadratic manifold orthogonality and reconstruction
* interface restriction matrix assembly
* thin-plate-spline RBF interpolation on training points
* polynomial feature construction
* Lasso/LARS/BIC model selection
* metrics formulas
* timing/speedup computation

### Integration tests

Create small integration tests for:

* offline dataset -> POD -> regression -> online prediction
* ROM-FOM coupling hook on a tiny mock problem
* serialization and reload of trained models/configs

### Regression tests

For every bug fixed in the main repo:

* add a regression pytest,
* keep it permanently.

Use lightweight synthetic problems where the full FSI benchmark would be too expensive for CI.

---

## Documentation requirement

Every major reusable module must include:

* purpose,
* math-to-code mapping,
* input/output contract,
* expected tensor shapes,
* failure modes.

`examples/NIRB/` must include:

* runnable commands,
* expected artifacts,
* how to interpret the validation report.

---

## Done criteria

The agent may declare the task done only when all of the following are true:

* repo-level MOR tools exist,
* repo-level NIRB tools exist,
* example scripts use those repo-level tools,
* Example 1 passes all thresholds,
* Example 2 passes all thresholds,
* tests pass,
* any discovered repo bug has a regression pytest,
* the final validation report is present and green.

Until then, keep working.


