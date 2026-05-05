## Purpose

This branch collects the work done to make the fully‑Eulerian CutFEM Turek–Hron FSI2 example
(`examples/turek_fsi_fully_eulerian.py`) more robust and debuggable in the presence of:

- sudden changes in the cut/interface entity sets,
- near‑aligned interfaces (slivers / degenerate intersections),
- JIT kernel refresh overhead and “stale kernel” risks,
- Newton/line‑search stalls that previously triggered repeated `dt` reductions.

It intentionally **does not** commit any run artifacts such as `_turek_*` output folders.

## High‑level changes (what & why)

### 1) Faster and more reliable “precompute/refresh” for moving level sets

Goal: keep each time step practical while the interface moves and tags change.

Key ideas:
- Avoid expensive “owner search in physical coordinates” by evaluating `phi` at reference DOF nodes.
- Unify precompute logic so cut‑volume, interface, and ghost data are refreshed together.
- Provide a C++ geometry precompute backend to reduce Python overhead.

Where:
- `pycutfem/core/dofhandler.py`
- `pycutfem/jit/kernel_args.py`
- `pycutfem/ufl/helpers_geom.py`
- `pycutfem/jit/*` and `pycutfem/jit/cpp_backend/*`

How to use:
- `PYCUTFEM_UNIFIED_PRECOMPUTE=1`
- `PYCUTFEM_PRECOMPUTE_GEOM_BACKEND=cpp` (when using `PYCUTFEM_JIT_BACKEND=cpp`)

### 2) Robust aligned‑interface handling + sliver stabilization consistency

Problem: When `phi=0` aligns with mesh edges/vertices, classification and integration can flip between
“cut” and “aligned edge” logic, creating discontinuities in assembled terms and sometimes singular systems.

Fixes:
- Keep aligned interface edges separate from “cut interface” sets when `USE_ALIGNED_INTERFACE=1`.
- Ensure ghost stabilization edge sets remain non‑empty (add a small interior band around the cut region
  when one side has no fully‑inside/outside elements).
- Make the Hansbo cut ratio floor (`theta_min`) consistent across refresh paths (avoid re‑clipping to `1e-3`
  in one place and `1e-6` in another).

Where:
- `pycutfem/core/mesh.py`
- `examples/utils/fsi/fully_eulerian.py`
- `examples/turek_fsi_fully_eulerian.py`

### 3) More debuggable and robust Newton iteration (without “dt ping‑pong”)

Problem: A stalled step caused repeated `dt` reductions, making debugging difficult and hiding the true
 failure mode. Also, the failure often came from *one term* (interface penalty) or *one kernel*.

Changes:
- Add “abort after one dt reduction” mode for debugging:
  - `PYCUTFEM_ABORT_ON_DT_REDUCTION=1`
- Improve line-search robustness:
  - Default to Armijo (`LS_MODE=armijo`) for this example, and add a fallback if Deal.II‑style line search fails.
- Add residual diagnostics:
  - per‑DOF residual trace and per‑kernel contribution breakdown.

Where:
- `pycutfem/solvers/nonlinear_solver.py`
- `examples/turek_fsi_fully_eulerian.py`

### 4) Interface / cut integration robustness + regression tests

Changes:
- Add a continuity test suite around the alignment switch and additional corner/aligned segment checks.
- Make aligned‑segment tests robust to representing the same polyline as one segment vs two.

Where:
- `tests/ufl/test_alignment_switch_continuity.py`
- `tests/ufl/test_interface_segments_aligned.py`
- `pycutfem/integration/cut_integration.py`

## File‑by‑file notes (what changed)

- `.gitignore`
  - Ignore local run artifacts (notably `_turek_*`) so results/restarts are never accidentally committed.

- `examples/turek_fsi_fully_eulerian.py`
  - Added restart/dump plumbing (level set + full state) so individual failing steps can be reproduced quickly.
  - Added multiple targeted diagnostics (Peclet, interface residual decomposition, residual breakdown).
  - Added stabilization knobs (CIP, optional pressure stabilization, optional dt‑scaled Nitsche penalty).
  - Added multiple “debug safety” knobs (abort on dt reduction, penalty retry, line‑search fallback).
  - Kept experimental extension conditioning hooks (wrong‑side re‑extension / interface stitch) **off by default**.

- `examples/utils/fsi/fully_eulerian.py`
  - Domain‑set construction now separates aligned interface edges from cut interface, and builds a stabilizing
    interior ghost band in pathological “one‑sided” sliver cases.
  - Centralizes sliver weight refresh and DOF tagging helpers used by the example(s).

- `pycutfem/core/mesh.py`
  - Edge/element classification and tagging hardened around aligned‑interface and corner cases.
  - Additional bookkeeping to avoid “empty ghost set” situations that produce singular systems.

- `pycutfem/core/dofhandler.py`
  - Faster cut‑volume and ghost/interface precompute paths (avoid slow Python/owner‑search loops).
  - Extension/mapping helpers to support active DOF changes after retagging.

- `pycutfem/integration/cut_integration.py`
  - Interface extraction and segment bookkeeping made more robust (aligned edges, corner hits, de‑dup).

- `pycutfem/jit/__init__.py`, `pycutfem/jit/cache.py`, `pycutfem/jit/codegen.py`,
  `pycutfem/jit/cpp_backend/cache.py`, `pycutfem/jit/cpp_backend/codegen.py`,
  `pycutfem/ufl/compilers.py`, `pycutfem/ufl/helpers_geom.py`, `pycutfem/jit/kernel_args.py`
  - Runtime‑parameter refresh for `Constant.value` so penalty/dt tuning doesn’t require kernel regeneration.
  - Unified precompute + cached geometry helpers to reduce kernel refresh time after level‑set updates.

- `pycutfem/solvers/nonlinear_solver.py`
  - More robust line search behavior and better failure diagnostics (per‑kernel residual breakdown).
  - Hooks to stop after the first dt reduction for deterministic debugging (`PYCUTFEM_ABORT_ON_DT_REDUCTION=1`).

- `tests/ufl/test_alignment_switch_continuity.py`
  - New focused coverage for “cut ↔ aligned interface” continuity (area/length continuity and degenerate cases).

- `tests/ufl/test_interface_segments_aligned.py`
  - Made geometric expectations robust to segment splitting, and replaced slow symbolic integration with a fast
    analytic reference (keeps CI fast and stable).

## Quick “what to run” checklist

- Run the alignment switch regression suite:

```bash
PYCUTFEM_JIT_BACKEND=cpp conda run -n fenicsx python -m pytest -q \
  tests/ufl/test_alignment_switch_continuity.py
```

- Reproduce a failing step via restart dumps (example):

```bash
PYCUTFEM_ABORT_ON_DT_REDUCTION=1 \
  conda run -n fenicsx python -u examples/turek_fsi_fully_eulerian.py \
  --turek-case fsi2 --restart-dir <dir> --restart-step <n> --final-time <t> --dt 0.005 --no-pin-pressure
```

## Debugging machinery added to the Turek script

### Restart / replay support (for step‑by‑step debugging)

`examples/turek_fsi_fully_eulerian.py` now supports:
- dumping the level set per step,
- dumping full solution vectors per step (large),
- restarting from a prior dump to reproduce a failing step quickly.

Key env vars:
- `PYCUTFEM_DUMP_LEVELSET=1`
- `PYCUTFEM_DUMP_STATE=1`
- `PYCUTFEM_DUMP_STATE_EVERY=1`
- `RESTART_DIR=...`, `RESTART_STEP=...`, `RESTART_TAG=step|fail`

### Term isolation tools

- Peclet diagnostics:
  - `PYCUTFEM_PECLET_TRACE=1`
  - `PYCUTFEM_PECLET_TRACE_NEWTON=1`
- Interface residual decomposition:
  - `PYCUTFEM_INTERFACE_RESIDUAL_TRACE=1`
- Per‑kernel residual breakdown at the worst DOF:
  - `PYCUTFEM_RESIDUAL_BREAKDOWN=1`

### Experimental extension conditioning (default OFF)

These do **not** change the converged solution; they only modify the *initial guess* after re‑tagging.
They are kept as optional tools while diagnosing jump‑driven stalls.

- `PYCUTFEM_REEXTEND_WRONG_SIDE=1`
- `PYCUTFEM_STITCH_INTERFACE_VELOCITY=1`

## Known current blocker (as of the latest log)

Even with the improved integration/refresh and debugging machinery, the run can still stall at the first
large retagging event (currently around **step 11, t≈0.05** for `dt=0.005` in the linear‑solid debug setup),
with very large Peclet spikes and the interface penalty dominating the residual.

See `investigation_log.csv` entries 87–89 for the concrete reproduction and traces.

## Next recommended experiment: sweep `BETA_PENALTY`

The interface penalty uses:
- `beta_N = BETA_PENALTY * k*(k+1)` (and then `* (mu/h + rho*h/dt)` if dt scaling is enabled).

For the current physics scaling, using `BETA_PENALTY=1000` can be numerically brutal. A first sweep
recommended by the project notes is `BETA_PENALTY ∈ [1, 20]`.

Suggested restart command template:

```bash
BETA_PENALTY=10 PYCUTFEM_ABORT_ON_DT_REDUCTION=1 \
  conda run -n fenicsx python -u examples/turek_fsi_fully_eulerian.py \
  --turek-case fsi2 --restart-dir <dir> --restart-step <n> --final-time <t> --dt 0.005 --no-pin-pressure
```
