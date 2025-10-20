# pycutfem

`pycutfem` is a research-oriented toolkit for the Cut Finite Element Method
(CutFEM) implemented purely in Python. It provides level-set based geometry
handling, unfitted mesh classifications, a UFL-inspired symbolic layer, and a
Numba-backed code generator that assembles high-order finite element operators
with ghost-penalty stabilisation. The package is designed to make it easy to
prototype immersed boundary discretisations for elliptic and fluid problems,
while keeping the workflow close to classical FEM tooling.

- Level-set aware mesh, topology, and degree-of-freedom management in
  `pycutfem.core`.
- Symbolic form definitions via `pycutfem.ufl` with automatic kernel
  generation in `pycutfem.jit`.
- Ready-to-use Newton and linear solvers with optional PETSc integration in
  `pycutfem.solvers`.
- Example scripts covering Poisson, Stokes, Navier–Stokes, FSI benchmarks, and
  CutFEM verification cases in `examples/`.

## Installation

`pycutfem` targets Python 3.9 or newer. Creating a virtual environment is
recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Optional extras:

- `petsc4py` enables PETSc-based Krylov solvers (`python -m pip install petsc4py`).
- `ngsolve`, `fenics`, or `fenics-dolfinx` are only needed for the comparison
  notebooks in `examples/`.

## What is CutFEM?

The Cut Finite Element Method extends classical FEM to problems where the
physical domain cuts through a background mesh. Instead of conforming the mesh
to the geometry, CutFEM:

1. Embeds the geometry through an implicit level-set description.
2. Classifies elements as fully inside, outside, or intersected (“cut”).
3. Applies stabilisation (e.g., ghost penalties) on the cut interfaces to
   retain coercivity and optimal convergence.
4. Integrates variational forms over the physical sub-cells obtained by
   clipping the background elements.

This approach avoids remeshing and simplifies moving-boundary or fluid–structure
interaction simulations. `pycutfem` provides building blocks for each of the
steps above together with a domain-specific language for expressing CutFEM
forms compactly.

## Running the Poisson CutFEM example

The `examples/poisson_cut.py` script reproduces the deal.II step-85 Poisson
benchmark on a circular domain embedded in a square grid. Run it from the
repository root:

```bash
python examples/poisson_cut.py
```

The script performs four refinement cycles, assembles the CutFEM system,
solves it, prints a convergence table, and writes a VTK file to
`step85_results/step85_solution_cycle1.vtu`. You can inspect the file with
ParaView or any VTK viewer.

## Development notes

- Tests can be run with `pytest`.
- Set `PYCUTFEM_JIT_DEBUG=1` to print kernel argument diagnostics during
  assembly.
- The repository contains Jupyter notebooks under `examples/` for exploratory
  studies and comparisons against external FEM codes.
