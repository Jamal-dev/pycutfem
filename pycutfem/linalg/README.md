# pycutfem.linalg

This package is the reusable linear-algebra layer for `pycutfem`.

## Layout

- `layout.py`: turns reduced field names into explicit block layouts.
- `block.py`: wraps a sparse matrix and rhs as a cached block linear system.
- `preconditioners.py`: sparse subblock solvers, ILU/direct factorizations,
  block-diagonal and block-triangular preconditioners, and Uzawa-style
  preconditioners.
- `solvers.py`: SciPy Krylov drivers and a standalone Uzawa iteration.

## Design Rules

1. Assembly stays in `pycutfem.solvers.nonlinear_solver`.
2. Block algebra lives here and only depends on sparse matrices, vectors, and
   per-DOF field names.
3. Newton and other high-level solvers should configure this layer through
   field groups instead of hard-coding Schur complements inside solver logic.
4. Saddle-point systems with many Lagrange-multiplier fields are handled by
   arbitrary block layouts, not by special-casing a single pressure block.
