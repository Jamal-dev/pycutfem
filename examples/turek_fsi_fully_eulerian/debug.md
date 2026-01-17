# Turek–Hron FSI-2/FSI-3 (Fully Eulerian Solid) — Debug Plan

Goal: run the **FSI-2** case robustly to `t=3.0` (and beyond) without spurious peaks/blow-ups, and track key observables (beam-tip displacement, interface drag/lift, pressure drop).

This file is meant to be **kept up to date** during the investigation (hypotheses, experiments, findings, next steps).

---

## Common “long-run” command (cpp backend, `dFacetPatch`)

This is the current recommended command to validate stability to `t=3.0` (and beyond). It uses:
- cpp backend
- unified precompute
- true ghost stabilization via `dFacetPatch`
- linear solid (default) for robustness; see the nonlinear variant below

```bash
PYCUTFEM_JIT_BACKEND=cpp PYCUTFEM_UNIFIED_PRECOMPUTE=1 USE_FACET_PATCH_GHOST=1 \
PENALTY_VAL=1e-3 PENALTY_GRAD=1e-3 BETA_PENALTY=20 \
PYCUTFEM_LS_FAIL_ACCEPT_FACTOR=20 \
python -u examples/turek_fsi_fully_eulerian.py \
  --turek-case fsi2 --mesh-backend structured --mesh-size 0.05 \
  --dt 0.005 --final-time 3.0 \
  --newton-tol 1e-6 --newton-rtol 0 --max-newton-iter 60 \
  --obs-every 20 --no-save-vtk --no-run-fd-check --no-run-fd-terms
```

Nonlinear solid (StVK) variant (more expensive; best to validate short runs first):

```bash
PYCUTFEM_JIT_BACKEND=cpp PYCUTFEM_UNIFIED_PRECOMPUTE=1 USE_FACET_PATCH_GHOST=1 \
USE_ALIGNED_INTERFACE=1 REFINE_INITIAL=0 \
USE_LINEAR_SOLID=0 PENALTY_VAL=1e-3 PENALTY_GRAD=1e-3 BETA_PENALTY=20 \
PYCUTFEM_LS_FAIL_ACCEPT_FACTOR=20 \
python -u examples/turek_fsi_fully_eulerian.py \
  --turek-case fsi2  --mesh-size 0.05 \
  --dt 0.005 --final-time 3.0 \
  --newton-tol 1e-6 --newton-rtol 0 --max-newton-iter 60 \
  --obs-every 20  --no-run-fd-check --no-run-fd-terms \
  --linear-solver petsc

## Reference (problem + expected behavior)

- Geometry (Turek–Hron channel + cylinder + elastic beam):
  - Channel: `L=2.2`, `H=0.41`
  - Cylinder centered at `CENTER=(0.2,0.2)`, radius `R=0.05`
  - Beam attached behind the cylinder (see `examples/turek_fsi_fully_eulerian.py` for exact parameters)
- FSI presets (from `examples/turek_fsi_fully_eulerian.py`):
  - **FSI-2**: `U_mean=1.0`, `rho_s=1e4`, `dt=0.005`, `theta=0.5` (periodic response)
  - **FSI-3**: `U_mean=2.0`, `rho_s=1e4`, `dt=0.005`, `theta=0.5` (chaotic response)
- Primary diagnostics printed by the example:
  - `tip=(x,y)` and `tip_disp=(dx,dy)`
  - interface traction integrals: `FD` (drag), `FL` (lift)
  - `Δp` between two probe points near the cylinder/beam

---

## Key implementation choice: `dFacetPatch` vs `dGhost`

For CG spaces, `dGhost` can yield **nearly-zero** stabilization because CG jumps vanish across facets. The **facet-patch** integral `dFacetPatch` evaluates polynomial extensions across an interior facet (two-element patch), producing **non-zero** control even for CG and improving robustness.

The example supports toggling via:

- `USE_FACET_PATCH_GHOST=1` → `dFacetPatch`
- `USE_FACET_PATCH_GHOST=0` → `dGhost`

Default is `dFacetPatch` when the method is CG.

---

## Note: “zero residual” on the gmsh backend

If you see `Newton 1: |R|_∞ = 0.00e+00` at every step **and** probe velocities stay ~0, the run is effectively doing “BC updates only” and skipping the solve.

Root cause (fixed in code): the reduced-system pruning logic used a **global max|A| scaled threshold**, so a single very large penalty entry (often from facet-patch/ghost terms) could cause almost the entire *fluid* block to be pruned as “decoupled”, leaving an empty/near-empty reduced system → `|R|∞` prints as zero.

Fix:
- `pycutfem/solvers/nonlinear_solver.py` now treats `PYCUTFEM_DROP_TOL` as an **absolute row/col L1 threshold** (no global max scaling).

Workarounds (older commits):
- Set `PYCUTFEM_DROP_TOL=0` (disable numeric pruning; structural `nnz==0` pruning still happens).
- Prefer `--mesh-backend structured` for long validation runs to `t=3.0` (recommended above).

---

## Run commands (cpp backend)

### Fast debug run (short, coarse)

```bash
PYCUTFEM_JIT_BACKEND=cpp \
PYCUTFEM_UNIFIED_PRECOMPUTE=1 \
USE_FACET_PATCH_GHOST=1 \
PYCUTFEM_PROFILE_PRECOMPUTE=1 \
PENALTY_VAL=1e-3 PENALTY_GRAD=1e-3 \
python -u examples/turek_fsi_fully_eulerian.py \
  --turek-case fsi2 \
  --mesh-backend gmsh --mesh-size 0.05 --no-refine-initial \
  --dt 0.005 --final-time 0.05 \
  --no-save-vtk --obs-every 5 \
  --no-run-fd-check --no-run-fd-terms
```

### Compare vs `dGhost` (expect weaker stabilization for CG)

```bash
PYCUTFEM_JIT_BACKEND=cpp \
PYCUTFEM_UNIFIED_PRECOMPUTE=1 \
USE_FACET_PATCH_GHOST=1 \
PYCUTFEM_PROFILE_PRECOMPUTE=1 \
PENALTY_VAL=1e-3 PENALTY_GRAD=1e-3 \
python -u examples/turek_fsi_fully_eulerian.py \
  --turek-case fsi2 \
  --mesh-backend gmsh --mesh-size 0.05 --no-refine-initial \
  --dt 0.005 --final-time 3.0 \
  --save-vtk --obs-every 5 \
  --no-run-fd-check --no-run-fd-terms
```

---

## Investigation plan (small steps)

1. Establish a **baseline** (`dFacetPatch`, unified precompute on) and record:
   - precompute timings (`PYCUTFEM_PROFILE_PRECOMPUTE=1`)
   - Newton convergence stats and any dt reductions
   - observables trends (FD/FL/Δp/tip)
2. Run the same setup with `USE_FACET_PATCH_GHOST=0` to confirm whether loss of stabilization correlates with:
   - larger peaks in observables
   - Newton stagnation/failure
3. Optimize precompute focusing on `dFacetPatch` factor generation/caching (target: cut refresh dominated by interface/ghost, not facet-patch geometry).
4. Re-run the baseline and compare:
   - peak magnitudes
   - robustness (can we push final time upward?)
   - runtime (precompute and total wall time)
5. If peaks remain:
   - tune stabilization knobs in a controlled sweep (see hypotheses below)
   - isolate whether peaks originate in fluid convection vs solid advection vs interface coupling
6. Validation run: attempt `--final-time 3.0` (FSI-2) with best-known stable settings.

---

## Hypotheses to test (track + resolve)

- [ ] H1: **CG ghost stabilization is ineffective with `dGhost`**; switching to `dFacetPatch` reduces peaks and improves time-step robustness.
- [ ] H2: **Facet-patch precompute dominates refresh time**; caching/reuse cuts precompute wall time significantly.
- [ ] H3: **Near-aligned cuts create extreme slivers** that amplify conditioning issues; sliver weights + nudging (`LEVELSET_NUDGE_EPS`) reduce peaks.
- [ ] H4: **Interface penalty scaling is too weak/strong** (especially for small `dt`); dt-scaled Nitsche penalty (`PYCUTFEM_NITSCHE_PENALTY_DT`) changes stability.
- [ ] H5: **Advection-driven oscillations need CIP damping**; `PYCUTFEM_CIP_BETA_*` reduces peaks without overdamping.
- [ ] H6: **Facet-patch geometry extension becomes ill-conditioned** on curved/high-order geometry; forcing affine patch geometry (`PYCUTFEM_FACET_PATCH_GEO_MODE=affine`) prevents huge `|J^{-1}|`.
- [ ] H7: **Aligned-interface decisions drop/flip active DOFs**; adjust `LEVELSET_EDGE_TOL`, `LEVELSET_SNAP_EPS`, and `USE_ALIGNED_INTERFACE`.

---

## Findings log

### 2026-01-15 (initial state after merging `origin/feature/xfem-agfem-framework`)

- **Implemented facet-patch precompute speedup**: cache facet-patch geometry/basis tables independent of the level-set and refresh only masks/φ when the level-set token changes (`pycutfem/core/dofhandler.py`).
  - Micro-benchmark (Q2 on ~18×18 quads, `q=6`, ~56 ghost edges): first build `~0.54s`, refresh with changed `cache_token` `~0.01s` (≈`55×` faster).
- **Fast simulation comparison (cpp backend, coarse dt)** using structured mesh:
  - `USE_FACET_PATCH_GHOST=1` (`dFacetPatch`): stable Newton (3–4 iters/step), reasonable `FD/FL/Δp` magnitudes; no spurious level-set refresh triggered in this short run.
  - `USE_FACET_PATCH_GHOST=0` (`dGhost`): very large updates (`ΔU` blowing up to `~1e6`) and enormous pressure-drop/forces by `t≈0.8` (clear “peaks”); also triggers level-set refresh each step.
  - Logs:
    - `dFacetPatch`: `/tmp/pycutfem_fsi2_facetpatch_cpp_dt0p05.log`
    - `dGhost`: `/tmp/pycutfem_fsi2_dghost_cpp_dt0p05.log`

### 2026-01-16 (refined structured mesh, dt=0.005) — level-set refresh + Newton issues

**Setup:** `--mesh-backend structured --mesh-size 0.05` with initial anisotropic refinement enabled (default), cpp backend.

This section is now **superseded** by the fixes below (constraint-aware active DOFs + facet-patch precompute reuse). Keeping the notes for historical context.

- **First “real” level-set refresh happens around `t≈0.055`** (step ~11) even with tiny motion:
  - Example log: `/tmp/pycutfem_fsi2_facetpatch_cpp_dt0p005_refined_t0p06_tol1e-6_rtol0.log`
  - At refresh time: `max|Δφ|≈4.3e-08` and `interface_edges=0` (no aligned edges), yet the refresh still triggers a full kernel static-arg rebuild.
  - Refresh cost is currently **~7s** and is dominated by:
    - `volume/+` kernels
    - multiple `facet_patch` kernels (subsetting/copying per-integral remains expensive).

- **Main blocker: Newton stagnates immediately after the first refresh**
  - After the refresh, the next step’s Newton often reaches a floor around `|R|∞≈1.23e-02` and the Armijo line search cannot find a decreasing step (eventually throws “Line search failed: no residual decrease.”).
  - If `NEWTON_RTOL>0`, the solver may incorrectly accept a solution with `|R|∞≈1e-2` because `rtol*|R0|` becomes large when `|R0|` is huge (post-refresh predictor mismatch). This “false convergence” can lead to rapid blow-up in later steps.
  - Setting `NEWTON_RTOL=0` avoids false convergence but exposes the underlying stagnation (step fails instead of silently progressing).

- **Forcing a full rebuild of static args does not fix it**
  - `PYCUTFEM_LEVELSET_KERNEL_REFRESH=rebuild` still exhibits the same `|R|∞≈1e-2` stagnation and line-search failure after refresh, so this is likely **not** just a reuse/merge bug.

**Useful logs from this round**
- “No snap, early steps ok (no refresh)”: `/tmp/pycutfem_fsi2_facetpatch_cpp_dt0p005_refined_t0p02_nosnap.log`
- “First refresh then failure (rtol=0)”: `/tmp/pycutfem_fsi2_facetpatch_cpp_dt0p005_refined_t0p06_tol1e-6_rtol0.log`
- “Rebuild refresh mode (still fails)”: `/tmp/pycutfem_fsi2_facetpatch_cpp_dt0p005_refined_t0p06_rebuildRefresh.log`

### 2026-01-16 (current): root cause + fixes + validation runs

#### Root cause (post-refresh convergence changes)

When hanging-node constraints are present (structured refined mesh), the solver operates in **master DOF space**.
Before this fix, the post-refresh “active DOF recomputation” (and the “extend newly active DOFs” stage) ran in **full DOF space**, which can desynchronize:

- reduced/master indexing (`full_to_red`, `red_to_full`)
- restriction masks from UFL Restrictions
- per-field active subsets

This mismatch manifests as “refresh ⇒ Newton behavior changes / stalls”, often with facet-patch kernels dominating the residual at the stall DOF.

Fixes applied:
- `examples/turek_fsi_fully_eulerian.py`: `recompute_active_dofs()` is now constraint-aware:
  - recomputes active sets in master space when `solver.constraints` exists
  - maps BC/inactive tags to master indices consistently
  - fixes `PYCUTFEM_ACTIVE_DOFS_TRACE` to map master→full for per-field counts
  - makes the “extend newly active DOFs” mapping constraint-aware (master→full before nearest extension)

#### Precompute optimization (`dFacetPatch`)

The facet-patch (true ghost) precompute was the dominant refresh cost. Improvements:

- `pycutfem/core/dofhandler.py`: facet-patch geometry/basis caching + refresh-only φ/masks
- Added **superset reuse**: when the ghost-edge set changes slightly after refresh (often a subset), reuse cached geometry by subsetting rows instead of recomputing
- Cache the derived subset geometry for subsequent refreshes (bounded by `PYCUTFEM_FACET_PATCH_GEO_CACHE_MAX`, default `4`)
- Bound the full facet-patch output cache (`PYCUTFEM_FACET_PATCH_CACHE_MAX`, default `1`) to avoid memory blow-ups in time-dependent runs

#### Validation (cpp backend, forced refresh every step)

Command (restart from the stored reproducer, force refresh with `LEVELSET_UPDATE_TOL=1e-12`):

```bash
PYCUTFEM_JIT_BACKEND=cpp PYCUTFEM_UNIFIED_PRECOMPUTE=1 USE_FACET_PATCH_GHOST=1 \
LEVELSET_UPDATE_TOL=1e-12 PYCUTFEM_PROFILE_PRECOMPUTE=1 \
python -u examples/turek_fsi_fully_eulerian.py \
  --turek-case fsi2 --mesh-backend structured --mesh-size 0.05 \
  --dt 0.005 --final-time 0.06 --max-steps 1 \
  --newton-tol 1e-6 --newton-rtol 0 --max-newton-iter 6 \
  --no-save-vtk --no-compute-observables --no-run-fd-check --no-run-fd-terms \
  --restart-dir _refresh_baseline_dump --restart-step 9 --restart-tag step
```

Key results (see `_facet_patch_refresh_test4.log` in repo root):
- Facet-patch precompute:
  - first build: `~24.5s` (expected: geometry/basis tabulation for ~990 ghost edges)
  - first refresh (ghost edges changed 990→978): **`~1.4s`** (was `~23s` before superset reuse)
  - subsequent refreshes: `~0.75s`
- Kernel refresh wall time:
  - first refresh: `~9.7s` (was `~31s` when facet-patch geometry was recomputed)
  - subsequent refreshes: `~7.1s`
- Newton convergence:
  - converges cleanly after each refresh (3 Newton iterations to `~1e-10` residual in this window)

Solid model in this validation:
- The main run above used the **linear solid** (`USE_LINEAR_SOLID=1`, the example default).
- A short smoke run with **nonlinear StVK** (`USE_LINEAR_SOLID=0`) also converged through the same refresh window (`_nonlinear_solid_smoke.log`), but it has not been validated to `t=3.0` yet.

---

## Next steps (short list)

- [ ] Run longer (target `t=3.0`) with cpp backend + `dFacetPatch` and record:
  - time series of `tip_disp`, `FD/FL`, `Δp`
  - whether refresh triggers remain stable and how often they occur
- [ ] If peaks/nonconvergence reappear at later times:
  - sweep `BETA_PENALTY`, `PENALTY_VAL`, `PENALTY_GRAD` in a controlled range
  - test `SOLID_KINEMATIC_STAB` as a robustness knob for the Eulerian solid advection/kinematic block
- [ ] If refresh frequency is too high (performance): increase `LEVELSET_UPDATE_TOL` and/or use `LEVELSET_SNAP_EPS`/`LEVELSET_NUDGE_EPS` carefully to avoid spurious topology changes.
