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
- Reusable block linear algebra in `pycutfem.linalg` and ready-to-use Newton
  and linear solvers with optional PETSc integration in `pycutfem.solvers`.
- Example scripts covering Poisson, Stokes, Navier–Stokes, FSI benchmarks, and
  CutFEM verification cases in `examples/`.

## Installation

`pycutfem` targets Python 3.9 or newer.  The recommended developer and paper
workflow is the Conda environment in `environment.yml`, because it installs the
Python stack, `pybind11`, Eigen headers, and platform compilers consistently on
Linux, macOS, and Windows.

```bash
conda env create -f environment.yml
conda activate pycutfem
python -m pytest -q tests/test_three_constituent_benchmark_suite.py
```

If you prefer a plain Python virtual environment, install the package in editable
mode.  The package metadata reads `requirements.txt`, so editable installation
now installs the runtime dependencies as well:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
```

Optional extras:

- `petsc4py` enables PETSc-based Krylov solvers.
- `pypardiso` enables the direct PARDISO backend (`python -m pip install -e ".[pardiso]"`).
- `ngsolve`, `fenics`, or `fenics-dolfinx` are only needed for comparison
  notebooks and external-code studies in `examples/`.

### Windows Notes

The supported Windows path is Miniforge or Mambaforge plus the checked-in Conda
environment:

```powershell
conda env create -f environment.yml
conda activate pycutfem
python -m pytest -q tests/test_three_constituent_benchmark_suite.py
```

For C++ backend tests on Windows, install the Microsoft C++ Build Tools if they
are not already present.  The Conda environment provides Eigen under
`%CONDA_PREFIX%\Library\include\eigen3`; the CI pipeline sets
`EIGEN_INCLUDE_DIR` to that path automatically.

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
- Repository structure and ownership rules are documented in `ARCHITECTURE.md`.

### C++ backend

The C++/Eigen code-generation path lives under `pycutfem/jit/cpp_backend`.  Set
`PYCUTFEM_JIT_BACKEND=cpp` to route kernel compilation through the pybind11-based
cache.  Requirements:

- `pybind11`, NumPy headers, and a compiler with C++17 support.
- Eigen headers available through `EIGEN_INCLUDE_DIR`, the Conda environment, or
  `/usr/include/eigen3`.

The Docker image and GitHub Actions workflow both exercise the C++ backend on
the current three-constituent regression tests.

## Three-Constituent Paper Suite

The current paper-suite entry point is:

```bash
python examples/biofilms/benchmarks/three_constituent/run_paper_benchmarks.py \
  --suite smoke \
  --skip-seboldt \
  --outdir out/three_constituent_smoke \
  --backend cpp \
  --linear-backend scipy
```

Use `--suite paper` for the full paper asset generation.  The Seboldt/deformable
layer run is intentionally long; for routine CI and local checks, use
`--skip-seboldt` or `--reuse-seboldt /path/to/completed/run`.

## Docker

The checked-in `Dockerfile` builds a reproducible Linux image with the full
runtime stack, compilers, Eigen, pybind11, and the package installed in editable
mode.

```bash
docker build -t pycutfem:local .
docker run --rm pycutfem:local
```

To run the paper smoke suite and keep outputs on the host:

```bash
mkdir -p out
docker run --rm -v "$PWD/out:/workspace/pycutfem/out" pycutfem:local \
  python examples/biofilms/benchmarks/three_constituent/run_paper_benchmarks.py \
    --suite smoke \
    --skip-seboldt \
    --outdir out/docker_three_constituent_smoke \
    --backend cpp \
    --linear-backend scipy
```

The Docker build context excludes local credentials, nested paper repositories,
and generated numerical output through `.dockerignore`.

## CI/CD

GitHub Actions is configured in `.github/workflows/ci.yml`.

- The test job runs on `ubuntu-latest` and `windows-latest` from
  `environment.yml`.
- The Linux job also runs the three-constituent paper smoke suite without the
  long Seboldt case.
- Core, full, and Docker test jobs write compact pass/fail/skip counts to the
  GitHub Actions job summary and upload the raw JUnit XML report as an artifact.
- The full pytest suite is available through the manual `workflow_dispatch`
  input `test_scope=full`.  By default it targets a self-hosted Linux x64 runner
  so long numerical tests are not constrained by standard hosted-runner wall
  time; use `full_runner=github-hosted` only for shorter diagnostic runs.
- The Docker job builds the image and runs a smoke test inside it.
- On non-PR events, the Docker job pushes `latest` and `${GITHUB_SHA}` tags to
  Docker Hub when these repository secrets are configured:
  `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`.

Do not commit Docker Hub, Overleaf, or other service credentials to the
repository.  Store them as GitHub repository secrets or local untracked files.
