# Repository Architecture

This repository is now organized around a strict boundary:

- `pycutfem/`: reusable library code (importable API for general FEM/CutFEM usage).
- `examples/`: benchmark/problem-specific scripts and helper modules.
- `tests/`: unit/integration/regression tests.

## Ownership Rules

1. `pycutfem/*` must stay problem-agnostic.
2. `examples/*` can depend on `pycutfem/*`, but not the other way around.
3. Benchmark-specific helpers belong under `examples/utils/*`, not `pycutfem/utils/*`.
4. Debug/prototyping tools belong under `examples/debug` or `examples/utils/debug`.

## Package Layout

### Core library

- `pycutfem/core`: mesh/topology/DOF/level-set core data structures.
- `pycutfem/ufl`: symbolic forms, expressions, compilers.
- `pycutfem/assembly`: local/global assembly and BC application.
- `pycutfem/jit`: backend code generation and kernel compilation.
- `pycutfem/solvers`: linear/nonlinear/time-step solver infrastructure.
- `pycutfem/nonmatching`, `pycutfem/xfem`, `pycutfem/fem`, `pycutfem/integration`, `pycutfem/io`: reusable numerical modules.
- `pycutfem/utils`: only generic, cross-problem helpers (mesh generation, tagging, bitsets, etc.).

### Example-domain helpers

- `examples/utils/biofilm`: biofilm model forms and MMS helpers.
- `examples/utils/fsi`: FSI-specific utilities (fully Eulerian/Turek helper code).
- `examples/utils/fpi`: FPI-specific forms, interface terms, and MMS helpers.
- `examples/utils/shared`: shared helpers used by example-domain modules.
- `examples/utils/debug`: debug-only analytic helpers.

## Naming & Separation Guidelines

1. Prefer feature/domain names over vague names (`fsi`, `biofilm`, `fpi`, `mesh`, `io`).
2. Keep module scope narrow: one coherent responsibility per file.
3. Put reusable numerics in `pycutfem/*`; put paper/benchmark logic in `examples/*`.
4. When a helper starts being reused outside examples, promote it from `examples/utils/*` to `pycutfem/*` with a focused API and tests.

## Dependency Direction

- Allowed:
  - `tests -> pycutfem`
  - `tests -> examples.utils` (for benchmark regression tests)
  - `examples -> pycutfem`
  - `examples.utils -> pycutfem`
- Not allowed:
  - `pycutfem -> examples`

## Core Layering

`pycutfem` is organized as layered subsystems with explicit ownership:

1. Symbolic/UFL layer (`pycutfem/ufl`)
   - Expression tree and algebra: `pycutfem/ufl/expressions.py`
   - Spaces and symbolic field groups: `pycutfem/ufl/spaces.py`
   - Forms/measures and assembly entry points: `pycutfem/ufl/forms.py`, `pycutfem/ufl/measures.py`
2. JIT execution layer (`pycutfem/jit`)
   - IR/codegen/cache: `pycutfem/jit/visitor.py`, `pycutfem/jit/codegen.py`, `pycutfem/jit/cache.py`
   - Kernel argument preparation and scatter helpers: `pycutfem/jit/kernel_args.py`
3. Compiler/assembly bridge (`pycutfem/ufl/compilers.py`, `pycutfem/assembly/*`)
   - Converts symbolic forms to executable kernels and assembles global systems.

### Canonical Module Paths

- `FunctionSpace`: `pycutfem.ufl.spaces`
- JIT kernel-argument helpers: `pycutfem.jit.kernel_args`

Compatibility shims remain in:
- `pycutfem/ufl/functionspace.py`
- `pycutfem/ufl/helpers_jit.py`

These wrappers should not be used for new code.

## Core Design Anti-Patterns

1. JIT/runtime code inside symbolic namespaces.
2. Domain/problem-specific logic inside core modules.
3. Module names that hide ownership (for example, generic `helpers_*` with mixed responsibilities).
4. Cross-layer imports that invert dependency flow.

## Review Checklist For New Core Modules

1. Is the module clearly owned by one layer (symbolic, jit, compiler, solver, io)?
2. Does it avoid importing upward across layers?
3. Are names explicit about responsibility?
4. Is there a compatibility shim only when migration requires it?
