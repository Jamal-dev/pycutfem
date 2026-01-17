# Hanging-Node Verification Benchmarks

This folder collects small, focused checks for the **hanging-node constraint**
machinery (`DofHandler.build_hanging_node_constraints()`) and its use inside the
Newton solver (master-space condensation).

## 1) Polynomial Poisson (exactness / patch-like test)

Runs a Poisson problem on a tiny mesh with a single nonconforming interface and
checks that the solver reproduces an **exact polynomial** solution for `Qp`
(`p=1,2,3`) to near machine precision.

```bash
PYCUTFEM_JIT_BACKEND=cpp python -u examples/hanging_nodes/poisson_patch_test.py
```

## 2) Smooth Poisson (convergence on a hanging mesh)

Solves a smooth manufactured solution `u=sin(pi x) sin(pi y)` on a sequence of
meshes that keep a persistent hanging interface (left half refined one extra
level) and prints L2 errors and observed rates.

```bash
python -u examples/hanging_nodes/poisson_convergence.py
```

## Related examples

- `examples/hanging_node_constraints_demo.py`: minimal constraint detection demo.
- `examples/poisson_hanging_adaptive.py`: MMS + adaptive-like refinement producing hanging nodes.
- `examples/lshape_poisson_amr.py`: L-shape singular benchmark (compare to NGSolve-style runs).

## Related tests

- `tests/test_hanging_constraints_polynomial_reproduction.py`
- `tests/test_hanging_nodes_poisson_verification.py`
