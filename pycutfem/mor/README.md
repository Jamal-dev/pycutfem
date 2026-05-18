# pycutfem.mor

`pycutfem.mor` is the reduced-order modeling (MOR) layer of `pycutfem`.
It is problem-generic: the PDE code supplies residuals, tangents, snapshots,
field layouts, boundary conditions, generated UFL/native kernels, and
quantities of interest.  The MOR layer supplies the reusable algebra: reduced
bases, reduced nonlinear solvers, hyper-reduction, mixed-field stability checks,
artifacts, adjoints, DWR certificates, adaptive enrichment, timing gates, and
readiness checks.

This README is organized like a MATLAB help page.  Each topic is written in the
same order:

1. **Idea:** what problem the tool solves.
2. **First-principles derivation:** where the equations come from.
3. **API mapping:** how the mathematical objects appear in `pycutfem.mor`.
4. **Expected result:** what the example should show.
5. **Example:** a minimal code pattern directly below the explanation.

Run standalone examples from the repository root with:

```bash
conda run --no-capture-output -n fenicsx python your_script.py
```

Most examples below use only `numpy` and the `pycutfem.mor` API.  Native online
examples need generated UFL/C++ kernels from the PDE layer, so those are shown
as call patterns or artifact patterns rather than complete toy PDE solvers.

---

## Contents

1. [Starting point: a finite element residual](#starting-point-a-finite-element-residual)
2. [Recommended MOR workflow](#recommended-mor-workflow)
3. [Topic 1: POD basis construction](#topic-1-pod-basis-construction)
4. [Topic 2: Galerkin and full-row LSPG](#topic-2-galerkin-and-full-row-lspg)
5. [Topic 3: Weighted LSPG and GNAT sampling](#topic-3-weighted-lspg-and-gnat-sampling)
6. [Topic 4: DEIM and QDEIM nonlinear reconstruction](#topic-4-deim-and-qdeim-nonlinear-reconstruction)
7. [Topic 5: Positive empirical cubature](#topic-5-positive-empirical-cubature)
8. [Topic 6: Mixed-field POD, gauge correction, and block weights](#topic-6-mixed-field-pod-gauge-correction-and-block-weights)
9. [Topic 7: Lift and supremizer enrichment](#topic-7-lift-and-supremizer-enrichment)
10. [Topic 8: Decoded bounds and constrained Gauss-Newton](#topic-8-decoded-bounds-and-constrained-gauss-newton)
11. [Topic 9: Gappy POD quantity reconstruction](#topic-9-gappy-pod-quantity-reconstruction)
12. [Topic 10: Quadratic manifold decoders](#topic-10-quadratic-manifold-decoders)
13. [Topic 11: Branch predictors and reference policies](#topic-11-branch-predictors-and-reference-policies)
14. [Topic 12: State updates and sample-local transactions](#topic-12-state-updates-and-sample-local-transactions)
15. [Topic 13: Discrete adjoints and DWR certification](#topic-13-discrete-adjoints-and-dwr-certification)
16. [Topic 14: Feature atlases and local model banks](#topic-14-feature-atlases-and-local-model-banks)
17. [Topic 15: Sparse GNAT lift](#topic-15-sparse-gnat-lift)
18. [Topic 16: Native artifacts and native online solve pattern](#topic-16-native-artifacts-and-native-online-solve-pattern)
19. [Topic 17: Timing and readiness gates](#topic-17-timing-and-readiness-gates)
20. [Acceptance checklist and common mistakes](#acceptance-checklist-and-common-mistakes)
21. [Public API inventory](#module-inventory)

---

## Starting point: a finite element residual

A finite element code usually starts from a weak problem:

$$
F(u;v)=0 \qquad \text{for all test functions } v.
$$

After choosing finite element basis functions
$\varphi_1,\ldots,\varphi_N$, the discrete solution is written as

$$
u_h=\sum_{j=1}^{N}x_j\varphi_j.
$$

Testing with each basis function $v=\varphi_i$ gives one algebraic equation:

$$
R_i(x)=F(u_h;\varphi_i), \qquad i=1,\ldots,N.
$$

So the full-order model (FOM) solves

$$
R(x_n,x_{n-1};\mu)=0.
$$

The symbols are:

| Symbol/code name | Meaning | Size |
| --- | --- | --- |
| $x_n$ | full finite element state at time step $n$ | $N$ |
| $x_{n-1}$ | previous state for transient problems | $N$ |
| $\mu$ | parameters | problem-dependent |
| $R$ / `residual` | algebraic residual vector | $N$ |
| $J=\partial R/\partial x_n$ / `jacobian`, `tangent` | tangent/Jacobian | $N\times N$ |

For nonlinear problems, Newton's method linearizes the residual.  Around an
iterate $x_k$,

$$
R(x_k+\Delta x)
= R(x_k)+J(x_k)\Delta x+\mathcal{O}(\|\Delta x\|^2).
$$

Ignoring the higher-order term and asking the linearized residual to vanish
produces the full Newton system

$$
J(x_k)\Delta x=-R(x_k).
$$

This is the expensive equation MOR tries to avoid solving in dimension $N$ for
every online parameter, time step, or coupling iteration.

---

## Recommended MOR workflow

For a nonlinear production reduced model, use the tools in this order:

1. Run stable FOM trajectories and save states, residuals, tangents, and QoI data.
2. Remove Dirichlet lifts and gauge-correct null-mode fields before POD.
3. Build POD or fieldwise POD bases.
4. Check per-field projection errors and mixed coupling rank.
5. Validate a full-row Galerkin or LSPG reduced model before hyper-reduction.
6. Add lift/supremizer modes if mixed stability fails.
7. Build DEIM/QDEIM, GNAT, gappy POD, or empirical cubature only after the full reduced model is credible.
8. Certify sampled/full residual norm equivalence.
9. Build native sampled kernel bundles and a native reduced artifact.
10. Run online Gauss-Newton in native code.
11. Guard nonlinear branch selection with predictors/reference policies.
12. Enforce decoded physical bounds where needed.
13. Certify quantities of interest with DWR.
14. Adaptively enrich if any gate fails.
15. Accept only if accuracy, bounds, branch, DWR, sampling, and timing gates pass.

---

## Topic 1: POD basis construction

### Idea

POD answers the first MOR question:

> Given many expensive full-order states, what low-dimensional linear space best
> represents them?

Collect snapshots

$$
x_1,x_2,\ldots,x_m\in\mathbb{R}^N
$$

and stack them into a feature-major matrix

$$
X=[x_1,x_2,\ldots,x_m]\in\mathbb{R}^{N\times m}.
$$

Each column is one full state.  Each row is one degree of freedom.

### First-principles derivation

If centering is enabled, compute the mean snapshot

$$
\bar{x}=\frac{1}{m}\sum_{j=1}^{m}x_j
$$

and define

$$
X_c=X-\bar{x}\mathbf{1}^T.
$$

POD asks for the best rank-$r$ approximation to the centered snapshots:

$$
\min_{\operatorname{rank}(X_r)\le r}\|X_c-X_r\|_F.
$$

The singular value decomposition

$$
X_c=U\Sigma Z^T
$$

solves this problem.  By the Eckart--Young theorem, the best rank-$r$ basis is
the first $r$ left singular vectors:

$$
V=U_{[:,1:r]}.
$$

The captured energy is

$$
E_r=\frac{\sum_{i=1}^{r}\sigma_i^2}{\sum_i\sigma_i^2}.
$$

Therefore `fit_pod(snapshots, energy=0.999)` chooses the smallest number of
modes whose singular values capture 99.9 percent of the snapshot energy.

Once $V$ is known, projection and reconstruction are

$$
q=V^T(x-\bar{x}),
\qquad
x_{\mathrm{rec}}=\bar{x}+Vq.
$$

### API mapping

| Math object | API object |
| --- | --- |
| $X$ | `snapshots`, shape `(n_dofs, n_snapshots)` |
| $\bar{x}$ | `pod.mean` |
| $V$ | `pod.basis` |
| $q=V^T(x-\bar{x})$ | `project_to_basis(...)` or `pod.project(...)` |
| $\bar{x}+Vq$ | `reconstruct_from_basis(...)` or `pod.reconstruct(...)` |

POD is a basis-construction tool.  A small POD reconstruction error does not by
itself certify nonlinear online accuracy; it only says the basis can represent
the training snapshots.

### Expected result

The example constructs snapshots from two smooth spatial patterns.  Because the
data are almost low rank, POD should reconstruct them with a relative error near
machine precision.

### Example

```python
import numpy as np
from pycutfem.mor import fit_pod, project_to_basis, reconstruct_from_basis

grid = np.linspace(0.0, 1.0, 80)
times = np.linspace(0.0, 1.0, 30)

snapshots = np.column_stack([
    np.sin(np.pi * grid) * np.exp(-t)
    + 0.2 * np.sin(2.0 * np.pi * grid) * t
    for t in times
])

pod = fit_pod(snapshots, energy=0.999, center=True)
q = project_to_basis(snapshots, pod.basis, pod.mean)
reconstructed = reconstruct_from_basis(q, pod.basis, pod.mean)

relative_error = np.linalg.norm(reconstructed - snapshots) / np.linalg.norm(snapshots)
assert relative_error < 1.0e-12
print(pod.n_modes, relative_error)
```

---

## Topic 2: Galerkin and full-row LSPG

### Idea

After POD gives a reduced trial space, every reduced state has the form

$$
x(q)=x_0+Vq.
$$

Now the question is:

> How do we solve the nonlinear full residual equation $R(x)=0$ using only the
> reduced coordinates $q$?

At Newton iteration $k$,

$$
x_k=x_0+Vq_k.
$$

A reduced correction $\Delta q$ produces the full correction

$$
\Delta x=V\Delta q.
$$

Substitute this into the full Newton equation:

$$
J(x_k)V\Delta q=-R(x_k).
$$

Define

$$
r_k=R(x_k),
\qquad
A_k=J(x_k)V.
$$

Then

$$
A_k\Delta q=-r_k.
$$

Here $A_k\in\mathbb{R}^{N\times r}$ and usually $N\gg r$.  This is an
overdetermined system: it has many more equations than unknown reduced
coordinates.  Galerkin and LSPG are two different ways to choose the reduced
correction.

### Galerkin from first principles

Galerkin requires the residual to be orthogonal to the trial basis:

$$
V^TR(x_0+Vq)=0.
$$

Define the reduced residual

$$
g(q)=V^TR(x_0+Vq).
$$

By the chain rule,

$$
\frac{\partial g}{\partial q}
=V^T\frac{\partial R}{\partial x}\frac{\partial x}{\partial q}
=V^TJV.
$$

Therefore the Galerkin Newton correction solves

$$
(V^TJV)\Delta q=-V^TR.
$$

Galerkin is often a good diagnostic baseline for symmetric, coercive, or mildly
nonsymmetric problems.  For strongly nonlinear or advective problems, residual
minimization may be more robust.

### LSPG from first principles

LSPG starts from the overdetermined reduced Newton equation

$$
A_k\Delta q=-r_k.
$$

Because this equation generally cannot be satisfied exactly, LSPG chooses the
correction that minimizes the linearized residual norm:

$$
\Delta q=
\operatorname*{argmin}_{y\in\mathbb{R}^r}\|r_k+A_ky\|_2^2.
$$

This is the Gauss-Newton step for the nonlinear least-squares problem

$$
q_*=
\operatorname*{argmin}_{q\in\mathbb{R}^r}\|R(x_0+Vq)\|_2^2.
$$

To derive the normal equations, set

$$
\phi(y)=\frac12\|r_k+A_ky\|_2^2.
$$

Then

$$
\nabla_y\phi(y)=A_k^T(r_k+A_ky).
$$

At the minimizer,

$$
A_k^T(r_k+A_k\Delta q)=0,
$$

so

$$
A_k^TA_k\Delta q=-A_k^Tr_k.
$$

Since $A_k=J(x_k)V$, this becomes

$$
V^TJ(x_k)^TJ(x_k)V\Delta q
=-V^TJ(x_k)^TR(x_k).
$$

The nonlinear LSPG optimality condition is

$$
(JV)^TR=0.
$$

This is why LSPG is a Petrov-Galerkin method: the residual is tested against
$JV$, not against $V$.

### API mapping

| Math object | API object |
| --- | --- |
| $r_k=R(x_k)$ | `residual(x)` or native residual kernel output |
| $A_k=J(x_k)V$ | `tangent_times_basis(x, V)` or sampled/native assembly |
| $\min\|r+A\Delta q\|^2$ | `gauss_newton_step(A, r, ...)` |
| $\|A^Tr\|$ | LSPG optimality norm / `step.gradient_norm` |

For LSPG, convergence should normally monitor the optimality norm
$\|A^Tr\|$, not only the raw residual norm $\|r\|$.

### Expected result

The example defines a toy nonlinear residual directly in vector form.  The code
forms $A=JV$ row-by-row and repeatedly calls `gauss_newton_step`.  The loop
should produce a finite reduced state and a reduced residual that decreases
until the LSPG optimality check is satisfied or the iteration limit is reached.

### Example

```python
import numpy as np
from pycutfem.mor import gauss_newton_step

rng = np.random.default_rng(1)
n_dofs = 40
n_modes = 6

V, _ = np.linalg.qr(rng.normal(size=(n_dofs, n_modes)))
x0 = np.zeros(n_dofs)
b = 0.4 * np.sin(np.linspace(0.0, 2.0 * np.pi, n_dofs))

def residual(x):
    return np.sin(x) + 0.15 * x - b

def tangent_times_basis(x, basis):
    diagonal = np.cos(x) + 0.15
    return diagonal[:, None] * basis

q = np.zeros(n_modes)
for _ in range(20):
    x = x0 + V @ q
    r = residual(x)
    A = tangent_times_basis(x, V)
    step = gauss_newton_step(A, r, damping=1.0e-10, backend="python")
    q = q + step.step
    if np.linalg.norm(A.T @ r) < 1.0e-10:
        break

x_rom = x0 + V @ q
print(np.linalg.norm(residual(x_rom)), step.gradient_norm)
```

---

## Topic 3: Weighted LSPG and GNAT sampling

### Idea

Full-row LSPG is robust but still expensive because it evaluates every residual
row.  GNAT is the sampled version of LSPG.  It answers the question:

> Can we solve the LSPG problem using only selected residual rows while still
> seeing the same nonlinear behavior as the full residual?

Two matrices enter the method:

- $S$: a sampling matrix that selects residual rows.
- $W_s$: a sampled weight matrix that scales the selected rows.

### Weighted LSPG from first principles

The Euclidean residual norm

$$
\|R\|_2^2=\sum_{i=1}^{N}R_i^2
$$

treats every residual row equally.  This may be wrong when rows have different
units, magnitudes, or physical importance.  A pressure residual row, an
interface-force row, and a concentration row should not necessarily be measured
with the same scale.

Weighted LSPG uses

$$
\|y\|_W^2=y^TWy,
$$

where $W$ is usually diagonal:

$$
W=\operatorname{diag}(w_1,\ldots,w_N),
\qquad
w_i\ge 0.
$$

The weighted Gauss-Newton correction is

$$
\min_{\Delta q}\|W^{1/2}(r_k+A_k\Delta q)\|_2^2.
$$

The normal equations are

$$
A_k^TWA_k\Delta q=-A_k^TWr_k.
$$

In code, weights are usually stored as a vector.  Applying $W^{1/2}$ means
multiplying residual row `i` and Jacobian row `i` by `sqrt(weights[i])`.

### The sampling matrix $S$

Suppose the sampled residual row ids are

$$
i_1,i_2,\ldots,i_s, \qquad s\ll N.
$$

Let $e_i$ be the $i$-th Euclidean coordinate vector.  The row selector is

$$
S=
\begin{bmatrix}
e_{i_1}^T\\
e_{i_2}^T\\
\vdots\\
e_{i_s}^T
\end{bmatrix}
\in\mathbb{R}^{s\times N}.
$$

Therefore

$$
Sr=
\begin{bmatrix}
r_{i_1}\\
r_{i_2}\\
\vdots\\
r_{i_s}
\end{bmatrix}.
$$

The matrix $S$ is not a reduced basis.  It does not approximate a vector by
linear combinations.  It only extracts selected rows.

### GNAT from first principles

Start with the full-row LSPG step

$$
\min_{\Delta q}\|r_k+A_k\Delta q\|_2^2.
$$

Apply row sampling and sampled weights:

$$
\min_{\Delta q}
\|W_s^{1/2}S(r_k+A_k\Delta q)\|_2^2.
$$

Define

$$
r_s=W_s^{1/2}Sr_k,
\qquad
A_s=W_s^{1/2}SA_k.
$$

Then the sampled problem is just a smaller least-squares problem:

$$
\min_{\Delta q}\|r_s+A_s\Delta q\|_2^2.
$$

Sampling is only safe if the sampled norm tracks the full residual norm on the
validation neighborhood:

$$
\gamma\|R(q)\|_2
\le
\|SR(q)\|_{W_s}
\le
\Gamma\|R(q)\|_2.
$$

This is the norm-equivalence certificate.  If it fails, sampled rows are
missing important residual information.

### API mapping

| Math object | API object |
| --- | --- |
| sampled row ids $i_1,\ldots,i_s$ | `row_dofs` |
| $S$ | implicit row extraction `r[row_dofs]`, `A[row_dofs, :]` |
| sampled weights $w_i$ | `row_weights` |
| $r_s=W_s^{1/2}Sr$ | `sqrt_w * r_full[rows]` |
| $A_s=W_s^{1/2}SA$ | `sqrt_w[:, None] * A_full[rows, :]` |
| norm-equivalence check | `certify_sampled_residual_norm_equivalence` |
| row augmentation | `augment_rows_for_residual_norm_equivalence` |

### Expected result

The example starts with very few sampled rows.  If those rows do not capture the
validation residuals, the augmentation routine adds more rows.  Then the sampled
residual and sampled Jacobian rows are used in the same `gauss_newton_step`
routine as full-row LSPG.

### Example

```python
import numpy as np
from pycutfem.mor import (
    augment_rows_for_residual_norm_equivalence,
    certify_sampled_residual_norm_equivalence,
    gauss_newton_step,
)

rng = np.random.default_rng(2)
n_dofs = 60
n_modes = 5
row_dofs = np.array([0, 3, 10, 22, 37, 51])
row_weights = np.ones(row_dofs.size)

# Offline validation residuals: columns are residuals from states near the
# expected online trajectory.
residual_validation = rng.normal(size=(n_dofs, 25))
row_blocks = (
    {"name": "field_a", "rows": np.arange(0, 20)},
    {"name": "field_b", "rows": np.arange(20, 40)},
    {"name": "field_c", "rows": np.arange(40, 60)},
)

certificate = certify_sampled_residual_norm_equivalence(
    residual_validation,
    row_dofs,
    row_weights=row_weights,
    row_blocks=row_blocks,
    lower_bound=0.95,
    upper_bound=1.05,
)

if not certificate.passed:
    augmented = augment_rows_for_residual_norm_equivalence(
        residual_validation,
        row_dofs,
        row_weights=row_weights,
        row_blocks=row_blocks,
        lower_bound=0.95,
        upper_bound=1.05,
        mandatory_rows=row_dofs,
        max_rows=n_dofs,
    )
    row_dofs = augmented.row_dofs
    row_weights = augmented.row_weights
    certificate = augmented.certificate

assert certificate.passed

r_full = rng.normal(size=n_dofs)
A_full = rng.normal(size=(n_dofs, n_modes))

sqrt_w = np.sqrt(row_weights)
r_s = sqrt_w * r_full[row_dofs]
A_s = sqrt_w[:, None] * A_full[row_dofs, :]

step = gauss_newton_step(A_s, r_s, damping=1.0e-10, backend="python")
print(row_dofs.size, step.step)
```

---

## Topic 4: DEIM and QDEIM nonlinear reconstruction

### Idea

DEIM/QDEIM are used when a nonlinear vector is expensive to evaluate in all
entries but is approximately low rank over the training data.  GNAT samples the
residual equation for a least-squares solve; DEIM/QDEIM reconstruct a nonlinear
vector from selected entries.

### First-principles derivation

Suppose nonlinear feature snapshots are

$$
F=[f_1,f_2,\ldots,f_m]\in\mathbb{R}^{N\times m}.
$$

Fit a collateral basis by POD:

$$
f(\mu)\approx\Phi_f c(\mu),
\qquad
\Phi_f\in\mathbb{R}^{N\times m_f}.
$$

If all entries of $f$ were available, one could compute $c$ by least squares.
DEIM/QDEIM avoid evaluating all entries.  Choose interpolation rows and define a
selector $P^T$.  The sampled interpolation equation is

$$
P^Tf=P^T\Phi_f c.
$$

If $P^T\Phi_f$ is square and nonsingular, then

$$
c=(P^T\Phi_f)^{-1}P^Tf.
$$

With oversampling or rank issues, use the pseudoinverse:

$$
c=(P^T\Phi_f)^\dagger P^Tf.
$$

The reconstructed vector is

$$
f\approx\Phi_f(P^T\Phi_f)^\dagger P^Tf.
$$

DEIM chooses rows greedily from interpolation residuals.  QDEIM chooses rows by
pivoted QR and is often the safer default because it tends to produce a better
conditioned interpolation matrix.

### API mapping

| Math object | API object |
| --- | --- |
| $F$ | `feature_snapshots`, `residual_snapshots` |
| $\Phi_f$ | `collateral.basis` |
| interpolation row ids | `rule.rows` |
| $P^Tf$ | `selected_values = f_new[rule.rows]` |
| $c$ | `interpolation_coefficients(rule, selected_values)` |
| reconstructed $f$ | `reconstruct_from_interpolation(rule, selected_values)` |

### Expected result

The example trains a collateral basis on smooth nonlinear feature snapshots,
selects QDEIM rows, evaluates only those entries for a new parameter, and
reconstructs the full vector.  The relative error should be small for a test
parameter inside the training range.

### Example

```python
import numpy as np
from pycutfem.mor import (
    fit_collateral_basis,
    build_qdeim_interpolation_rule,
    interpolation_coefficients,
    reconstruct_from_interpolation,
)

grid = np.linspace(0.0, 1.0, 100)
mus = np.linspace(0.5, 3.0, 35)

feature_snapshots = np.column_stack([
    np.exp(-mu * grid) * np.sin((1.0 + mu) * np.pi * grid)
    for mu in mus
])

collateral = fit_collateral_basis(feature_snapshots, n_modes=8, center=False)
rule = build_qdeim_interpolation_rule(collateral)

f_new = np.exp(-1.7 * grid) * np.sin(2.7 * np.pi * grid)
selected_values = f_new[rule.rows]
coefficients = interpolation_coefficients(rule, selected_values)
f_approx = reconstruct_from_interpolation(rule, selected_values)

relative_error = np.linalg.norm(f_approx - f_new) / np.linalg.norm(f_new)
assert relative_error < 1.0e-2
print(rule.rows, relative_error)
```

---

## Topic 5: Positive empirical cubature

### Idea

Finite element residuals and reduced operators are often assembled as sums over
cells, facets, cut cells, interface entities, or material blocks:

$$
R(x)=\sum_{e=1}^{n_e}R_e(x).
$$

Empirical cubature asks whether a small weighted subset can reproduce the full
sum on training targets.

### First-principles derivation

Let $a_e\in\mathbb{R}^{n_t}$ be the contribution of entity $e$ to a set of
training targets.  The full target is

$$
b=\sum_e a_e.
$$

Empirical cubature chooses a subset $\mathcal{S}$ and nonnegative weights
$w_e$ such that

$$
\sum_{e\in\mathcal{S}}w_ea_e\approx b,
\qquad
w_e\ge 0.
$$

The nonnegativity condition is important.  Negative weights can create
unphysical cancellation and unstable reduced operators.  Positive weights keep
the approximation closer to a true quadrature rule.

### API mapping

| Math object | API object |
| --- | --- |
| entity contributions $a_e$ | `contributions`, shape `(n_entities, n_targets)` |
| subset $\mathcal{S}$ | `fit.rule.entity_ids` |
| weights $w_e$ | `fit.rule.weights` |
| weighted sum | `apply_empirical_cubature(contributions, fit.rule)` |

### Expected result

The example fits a positive cubature rule to synthetic entity contributions.
Because the allowed number of entities equals the total number of entities, the
rule should pass a tight tolerance and reproduce the full target sum accurately.

### Example

```python
import numpy as np
from pycutfem.mor import apply_empirical_cubature, fit_positive_empirical_cubature

rng = np.random.default_rng(3)
contributions = np.abs(rng.normal(size=(25, 6)))

fit = fit_positive_empirical_cubature(
    contributions,
    max_entities=25,
    tolerance=1.0e-10,
    mandatory_ids=[0],
    entity_kind="cell",
)

approx_target = apply_empirical_cubature(contributions, fit.rule)
exact_target = contributions.sum(axis=0)

assert fit.passed
print(fit.passed, np.linalg.norm(approx_target - exact_target))
```

---

## Topic 6: Mixed-field POD, gauge correction, and block weights

### Idea

Many PDE systems have multiple physical fields:

$$
x=\begin{bmatrix}u\\p\\c\end{bmatrix},
$$

where $u$ may be velocity or displacement, $p$ pressure, and $c$ a scalar
quantity.  A single global Euclidean POD may hide small but important fields
behind large fields.  Fieldwise POD prevents this.

### First-principles derivation

Instead of fitting one basis to all rows, split the snapshot matrix by physical
row blocks:

$$
X=\begin{bmatrix}X_u\\X_p\\X_c\end{bmatrix}.
$$

Fit separate POD bases:

$$
u\approx u_0+V_uq_u,
\qquad
p\approx p_0+V_pq_p,
\qquad
c\approx c_0+V_cq_c.
$$

The global reduced state is block structured:

$$
V=
\begin{bmatrix}
V_u&0&0\\
0&V_p&0\\
0&0&V_c
\end{bmatrix}.
$$

For pressure-like fields, there is another issue: the pressure may be defined
only up to a constant.  Then a snapshot sequence can contain arbitrary pressure
shifts.  Gauge correction removes this null component before POD, for example

$$
p\leftarrow p-\frac{1}{|\Omega|}\int_\Omega p\,dx.
$$

Mixed residuals also need block-aware row weights.  If one residual block has
larger raw magnitude, it can dominate LSPG/GNAT.  Block weights rescale rows so
each field contributes appropriately to the residual norm.

### API mapping

| Math object | API object |
| --- | --- |
| physical row sets | `row_blocks` |
| pressure/null gauge rows | `pressure_gauge_blocks`, `gauge` |
| gauge-corrected snapshots | `gauge_correct_snapshots(...)` |
| block basis $V$ | `fit_fieldwise_pod_basis(...)` |
| per-field errors | `field_projection_errors(...)` |
| stability/gauge certificate | `certify_mixed_stability_basis(...)` |
| block row weights | `build_block_row_weights(...)` |

### Expected result

The example constructs synthetic velocity, pressure, and scalar row blocks,
removes the pressure gauge, fits separate POD bases, checks projection errors,
and builds one residual weight per state row.  The certificate should pass for
this simple synthetic dataset.

### Example

```python
import numpy as np
from pycutfem.mor import (
    fit_fieldwise_pod_basis,
    gauge_correct_snapshots,
    certify_mixed_stability_basis,
    build_block_row_weights,
)

rng = np.random.default_rng(4)
snapshots = rng.normal(size=(12, 30))

velocity_rows = np.arange(0, 4)
pressure_rows = np.arange(4, 7)
scalar_rows = np.arange(7, 12)

row_blocks = (
    {"name": "velocity", "rows": velocity_rows},
    {"name": "pressure", "rows": pressure_rows},
    {"name": "scalar", "rows": scalar_rows},
)
gauge = {"name": "pressure", "rows": pressure_rows}

corrected = gauge_correct_snapshots(snapshots, [gauge])
fieldwise = fit_fieldwise_pod_basis(
    corrected.corrected_snapshots,
    row_blocks,
    n_modes_per_block={"velocity": 4, "pressure": 2, "scalar": 5},
    center=True,
)

certificate = certify_mixed_stability_basis(
    snapshots,
    fieldwise.basis,
    offset=fieldwise.offset,
    row_blocks=row_blocks,
    pressure_gauge_blocks=[gauge],
)
row_weights = build_block_row_weights(corrected.corrected_snapshots, row_blocks)

assert certificate.passed
assert row_weights.shape == (12,)
print(certificate.passed, row_weights.shape)
```

---

## Topic 7: Lift and supremizer enrichment

### Idea

Saddle-point and constrained mixed systems need stable coupling between fields.
For example, incompressible flow couples velocity and pressure.  A reduced
pressure basis can become unstable if the reduced velocity basis cannot respond
to pressure modes.  Lift or supremizer enrichment adds missing primary-field
modes.

### First-principles derivation

A typical mixed linearized block system has the form

$$
\begin{bmatrix}
A & C^T\\
C & 0
\end{bmatrix}
\begin{bmatrix}
u\\p
\end{bmatrix}
=
\begin{bmatrix}
f\\g
\end{bmatrix}.
$$

The coupling operator $C$ connects the primary field to the constraint or
coupled field.  If the reduced bases are $V_u$ and $V_p$, the reduced coupling
contains

$$
V_p^TCV_u.
$$

If this matrix loses rank, some pressure/constraint modes cannot be represented
stably.  A generic lift solves

$$
A_pS=C^T\Phi_c,
$$

where:

- $A_p$ is a primary-field operator,
- $C$ is the coupling operator,
- $\Phi_c$ is the coupled-field basis,
- columns of $S$ are lift/supremizer snapshots.

Adding POD modes of $S$ to the primary basis improves reduced coupling rank.

### API mapping

| Math object | API object |
| --- | --- |
| $A_p$ | `primary_operator` |
| $C$ | `coupling_operator` |
| $\Phi_c$ | `coupled_basis` |
| lift snapshots $S$ | `solve_coupled_lift_snapshots(...)` |
| enriched primary basis | `fit_lift_enriched_basis(...)` |
| rank/stability check | `reduced_coupling_rank_certificate(...)` |

### Expected result

The example solves a tiny algebraic lift problem, enriches the primary basis,
and checks whether the reduced coupling rank is sufficient for the coupled
basis.

### Example

```python
import numpy as np
from pycutfem.mor import (
    fit_lift_enriched_basis,
    reduced_coupling_rank_certificate,
    solve_coupled_lift_snapshots,
)

rng = np.random.default_rng(5)
primary_operator = np.eye(5)
coupling_operator = rng.normal(size=(2, 5))
coupled_basis = np.eye(2)
primary_basis = np.eye(5, 2)

lift_snapshots = solve_coupled_lift_snapshots(
    primary_operator,
    coupling_operator,
    coupled_basis,
)

enriched = fit_lift_enriched_basis(
    primary_basis,
    coupled_basis,
    lift_snapshots,
    n_lift_modes=2,
)

rank = reduced_coupling_rank_certificate(
    coupling_operator,
    enriched.enriched_primary_basis,
    coupled_basis,
    required_rank=coupled_basis.shape[1],
)

assert enriched.enriched_primary_basis.shape[0] == primary_operator.shape[0]
print(rank.passed, enriched.enriched_primary_basis.shape)
```

---

## Topic 8: Decoded bounds and constrained Gauss-Newton

### Idea

Physical constraints apply to the decoded full state, not directly to the
reduced coefficients.  For example, volume fractions, concentrations, damage,
or detachment variables may need

$$
0\le \alpha \le 1.
$$

The reduced vector $q$ has no direct physical units, so bounds should be imposed
on

$$
x(q)=x_0+Vq.
$$

### First-principles derivation

Let $C$ select the full-state rows where bounds apply.  Then decoded bounds are

$$
\ell\le C(x_0+Vq)\le u.
$$

During a Gauss-Newton step, suppose some bounds are active.  For an active row,
we want the next decoded state to lie exactly on the bound:

$$
C_a(x_0+V(q+\Delta q))=b_a.
$$

Rearrange:

$$
C_aV\Delta q=b_a-C_a(x_0+Vq).
$$

So active bounds become equality constraints on the correction.  The
constrained Gauss-Newton problem is

$$
\min_{\Delta q}\|r+A\Delta q\|_2^2
\quad\text{subject to}\quad
C_aV\Delta q=h_a.
$$

The KKT system is

$$
\begin{bmatrix}
A^TA & (C_aV)^T\\
C_aV & 0
\end{bmatrix}
\begin{bmatrix}
\Delta q\\\lambda
\end{bmatrix}
=
\begin{bmatrix}
-A^Tr\\h_a
\end{bmatrix}.
$$

### API mapping

| Math object | API object |
| --- | --- |
| bound rows and limits | `BoundConstraintSpec` |
| reduced bound description | `full_bounds.reduce(trial_basis=V, offset=offset)` |
| feasibility projection | `project_reduced_coefficients_to_bounds(...)` |
| active equations | `reduced_bounds.active_equations(q)` |
| constrained step | `equality_constrained_gauss_newton_step(...)` |

### Expected result

The example starts from a reduced vector whose decoded state violates bounds.
Projection returns a feasible coefficient vector.  Active bound equations are
then passed into a constrained Gauss-Newton step with the correct reduced
step dimension.

### Example

```python
import numpy as np
from pycutfem.mor import (
    BoundConstraintSpec,
    equality_constrained_gauss_newton_step,
    project_reduced_coefficients_to_bounds,
)

V = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
offset = np.array([0.2, 0.2, 0.1])

full_bounds = BoundConstraintSpec(rows=np.array([0, 1, 2]), lower=0.0, upper=1.0)
reduced_bounds = full_bounds.reduce(trial_basis=V, offset=offset)

q_initial = np.array([1.2, 1.2])
q_feasible = project_reduced_coefficients_to_bounds(reduced_bounds, q_initial)
active = reduced_bounds.active_equations(q_feasible)

J = np.array([[2.0, 0.5], [0.2, 1.5], [1.0, -0.3]])
r = np.array([1.0, -0.5, 0.25])

step = equality_constrained_gauss_newton_step(
    J,
    r,
    constraint_matrix=active.constraint_matrix,
    constraint_rhs=active.rhs,
    backend="python",
)

assert reduced_bounds.max_violation(q_feasible) <= 1.0e-8
print(step.step)
```

---

## Topic 9: Gappy POD quantity reconstruction

### Idea

A solver may need a quantity vector, not only the state residual.  Examples are
interface reactions, drag, lift, heat flux, or load vectors.  If evaluating the
full quantity is expensive, gappy POD reconstructs it from selected entries.

### First-principles derivation

Fit a POD basis for quantity snapshots:

$$
y\approx \bar{y}+\Psi c.
$$

Let $P$ select sampled entries.  Then

$$
Py-P\bar{y}\approx P\Psi c.
$$

Solve for the coefficients:

$$
c=(P\Psi)^\dagger(Py-P\bar{y}).
$$

Then reconstruct the full quantity:

$$
y_{\mathrm{rec}}=\bar{y}+\Psi c.
$$

This is different from residual GNAT.  GNAT samples residual rows to solve a
least-squares problem; gappy POD samples quantity rows to reconstruct an output
vector.

### API mapping

| Math object | API object |
| --- | --- |
| quantity snapshots | `snapshots=quantity_snapshots` |
| quantity basis $\Psi$ | stored in `GappyPODQuantityOperator` |
| sample rows | `operator.sample_rows`, `operator.n_samples` |
| sampled entries $Py$ | `operator.sample(y)` |
| reconstructed quantity | `operator.reconstruct_from_samples(samples)` |

### Expected result

The example fits a gappy POD operator for a smooth synthetic quantity.  Sampling
returns only a short vector, while reconstruction returns a full vector with the
same shape as the original quantity.

### Example

```python
import numpy as np
from pycutfem.mor import fit_gappy_pod_quantity_operator

grid = np.linspace(0.0, 1.0, 50)
times = np.linspace(0.0, 1.0, 25)
quantity_snapshots = np.column_stack([
    np.cos(np.pi * grid * (1.0 + t)) + 0.1 * t * grid
    for t in times
])

operator = fit_gappy_pod_quantity_operator(
    snapshots=quantity_snapshots,
    n_modes=6,
    n_sample_rows=12,
    method="qdeim",
    center=True,
    name="interface_reaction",
)

y = quantity_snapshots[:, 3]
samples = operator.sample(y)
y_hat = operator.reconstruct_from_samples(samples)

assert samples.shape == (operator.n_samples,)
assert y_hat.shape == y.shape
print(operator.n_samples, operator.relative_error(y))
```

---

## Topic 10: Quadratic manifold decoders

### Idea

A linear POD decoder assumes the solution manifold is close to a flat affine
subspace:

$$
x(q)=\bar{x}+Vq.
$$

Strongly nonlinear dynamics may lie on a curved manifold.  A quadratic decoder
adds second-order reduced features.

### First-principles derivation

Define the vector of upper-triangular quadratic products

$$
\phi_2(q)=
[q_1^2,\ q_1q_2,\ldots,\ q_iq_j,\ldots,\ q_r^2]^T.
$$

The quadratic decoder is

$$
x(q)=\bar{x}+Vq+H\phi_2(q),
$$

where $H$ is fitted from training snapshots.  If $Q$ contains reduced
coordinates for training snapshots, fit $H$ by least squares:

$$
\min_H\|X-\bar{x}\mathbf{1}^T-VQ-H\Phi_2(Q)\|_F^2.
$$

The quadratic correction can represent curvature that would otherwise require
many more linear POD modes.

### API mapping

| Math object | API object |
| --- | --- |
| $\phi_2(q)$ | `quadratic_feature_matrix(q)` |
| quadratic basis $H$ | `fit_quadratic_manifold(...)` |
| full decoder | `fit_quadratic_decoder(...)` |
| decoded states | `decoder.decode(q)` |

### Expected result

The example fits a POD basis, computes reduced coordinates, fits a quadratic
correction, and verifies that decoding maps reduced coordinates back to the
original snapshot shape.

### Example

```python
import numpy as np
from pycutfem.mor import (
    fit_pod,
    fit_quadratic_decoder,
    fit_quadratic_manifold,
    quadratic_feature_matrix,
)

rng = np.random.default_rng(6)
snapshots = rng.normal(size=(20, 40))

pod = fit_pod(snapshots, n_modes=4, center=True)
q = pod.project(snapshots)
quadratic_basis = fit_quadratic_manifold(snapshots, q, pod.basis, mean=pod.mean)
decoder = fit_quadratic_decoder(snapshots, pod=pod)

decoded = decoder.decode(q)
features = quadratic_feature_matrix(q)

assert decoded.shape == snapshots.shape
print(decoded.shape, features.shape, quadratic_basis.shape, decoder.n_linear_modes)
```

---

## Topic 11: Branch predictors and reference policies

### Idea

Nonlinear residual equations can have several local roots.  In a transient
reduced solve, Newton or Gauss-Newton can jump to a wrong branch even if the
residual decreases.  Predictors and reference policies keep the online solve
near the physically expected trajectory.

### First-principles derivation

A predictor gives a reference reduced state

$$
q_{\mathrm{ref}}.
$$

Examples:

Constant predictor:

$$
q_{\mathrm{ref}}=q_n.
$$

Linear-history predictor:

$$
q_{\mathrm{ref}}=q_n+(q_n-q_{n-1}).
$$

Time-polynomial predictor:

$$
q_{\mathrm{ref}}(t)=\sum_{k=0}^{p}a_kt^k.
$$

The reduced nonlinear solve can include a reference penalty:

$$
\min_{\Delta q}
\|r+A\Delta q\|^2
+
\lambda\|V(q+\Delta q-q_{\mathrm{ref}})\|^2.
$$

The policy can also impose a maximum step norm or maximum reference distance.

### API mapping

| Math object | API object |
| --- | --- |
| constant predictor | `ConstantReducedPredictor` |
| linear-history predictor | `LinearHistoryReducedPredictor` |
| time-polynomial predictor | `fit_time_parameterized_predictor(...)` |
| reference penalty and guards | `ReferencePolicy` |
| native solver options | `result.native_options()` |

### Expected result

The example fits a time-parameterized predictor, creates a linear-history
reference policy, and produces a dictionary of native options that can be passed
into the online Gauss-Newton driver.

### Example

```python
import numpy as np
from pycutfem.mor import (
    ConstantReducedPredictor,
    LinearHistoryReducedPredictor,
    ReferencePolicy,
    fit_time_parameterized_predictor,
    predictor_from_native_dict,
)

times = np.linspace(0.0, 1.0, 20)
q_samples = np.column_stack([np.sin(times), np.cos(times)])

predictor = fit_time_parameterized_predictor(q_samples, times, degree=5)
prediction = predictor.predict(time=0.45)

policy = ReferencePolicy(
    predictor=LinearHistoryReducedPredictor(),
    reference_weight=1.0e-4,
    max_reference_distance=0.5,
    max_step_norm=0.25,
)

result = policy.predict(
    time=1.1,
    q_current=np.array([0.2, 0.3]),
    q_previous=np.array([0.1, 0.25]),
)

roundtrip = predictor_from_native_dict(ConstantReducedPredictor().to_native_dict())
assert prediction.coefficients.shape == (2,)
assert roundtrip.kind == "constant"
print(prediction.coefficients, result.native_options(), roundtrip.kind)
```

---

## Topic 12: State updates and sample-local transactions

### Idea

Native online kernels often need auxiliary arrays that depend on the current
reduced coordinates.  Examples include local coefficient arrays, lifted
Dirichlet values, quadrature history, subscale states, interface caches, and
active-set data.  These arrays must remain consistent during line search.

### First-principles derivation

The simplest state update is affine:

$$
y(q)=y_0+Bq.
$$

If a line-search trial changes $q$, the code must update $y(q)$ before testing
the residual.  If the trial is rejected, mutable state must return to its old
value.  This is a transaction:

1. Save the current mutable arrays.
2. Apply a trial update.
3. Accept the update only if residual, bounds, branch, and state checks pass.
4. Otherwise restore the saved arrays.

### API mapping

| Concept | API object |
| --- | --- |
| affine update $y_0+Bq$ | `AffineStateUpdateSpec` |
| apply updates | `apply_affine_state_updates(...)` |
| mutable-state transaction | `SampleStateTransaction` |
| trial object | `SampleStateTrial` |
| symbolic/native update | `SymbolicStateUpdateKernelSpec`, `NativeStateUpdateKernelCall` |

### Expected result

The example applies an affine state update.  It then modifies a history array
inside a transaction but does not call `trial.accept()`.  Therefore the history
array is restored automatically after leaving the context.

### Example

```python
import numpy as np
from pycutfem.mor import (
    AffineStateUpdateSpec,
    SampleStateTransaction,
    apply_affine_state_updates,
)

update = AffineStateUpdateSpec(
    name="u_local",
    basis=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
    offset=np.array([0.1, 0.2, 0.3]),
)

values = apply_affine_state_updates([update], np.array([2.0, -1.0]))
print(values["u_local"])

state = {"history": np.array([1.0, 2.0, 3.0])}
transaction = SampleStateTransaction(state)

with transaction.trial() as trial:
    state["history"][:] = 0.0
    # Call trial.accept() only if residual, bounds, branch, and state checks pass.

assert np.allclose(state["history"], [1.0, 2.0, 3.0])
print(state["history"])
```

---

## Topic 13: Discrete adjoints and DWR certification

### Idea

State error is not always the quantity of interest.  A simulation may care
about a scalar output such as drag, average concentration, reaction force, or
energy release.  DWR estimates the error in that quantity by weighting the
primal residual with an adjoint sensitivity.

### First-principles derivation

Let $x$ be the exact discrete solution and $x_r$ the reduced solution.  The QoI
error is

$$
Q(x)-Q(x_r).
$$

First-order Taylor expansion gives

$$
Q(x)-Q(x_r)\approx Q_x(x_r)e,
\qquad
e=x-x_r.
$$

Because $R(x)=0$, linearize the residual around $x_r$:

$$
0=R(x)\approx R(x_r)+J(x_r)e.
$$

So

$$
J(x_r)e\approx -R(x_r).
$$

Define the adjoint $z$ by

$$
J(x_r)^Tz=Q_x(x_r)^T.
$$

Then

$$
Q_xe=z^TJe\approx -z^TR(x_r).
$$

Up to the residual sign convention, the DWR estimate is

$$
\eta_Q=z^TR(x_r).
$$

For transient problems, solve the adjoint backward in time:

$$
J_n(x_n)^Tz_n
=
\frac{\partial Q}{\partial x_n}
-
\left(\frac{\partial R_{n+1}}{\partial x_n}\right)^Tz_{n+1},
$$

and sum the residual contributions:

$$
\eta_Q=\sum_n z_n^TR_n(x_r).
$$

DWR is meaningful only if the primal reduced trajectory is branch-correct and
sampling/gauge/bounds checks have passed.

### API mapping

| Math object | API object |
| --- | --- |
| $Q(x)$ | `QoIFunctionalSpec`, `evaluate_qoi_functional` |
| $Q_x$ | `assemble_qoi_gradient`, `check_qoi_gradient` |
| adjoint solve | `solve_discrete_adjoint`, `solve_reduced_discrete_adjoint` |
| $z^TR$ estimate | `dual_weighted_residual_estimate`, `certify_dual_weighted_residual` |
| guard checks | `dwr_certification_guard` |
| dominant terms | `dominant_dwr_contributions` |

### Expected result

The example uses a simple diagonal adjoint system.  The DWR certificate should
pass, the guard should pass, and the analytic QoI gradient should agree with a
finite-difference check.

### Example

```python
import numpy as np
from pycutfem.mor import certify_dual_weighted_residual, dwr_certification_guard

jacobians = [np.eye(3), 2.0 * np.eye(3)]
residuals = [np.array([0.1, 0.0, -0.05]), np.array([0.02, -0.03, 0.01])]
qoi_gradients = [np.array([1.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.0])]

certificate = certify_dual_weighted_residual(
    residuals,
    jacobians,
    qoi_gradients,
    backend="python",
)

guard = dwr_certification_guard(
    certificate.estimate,
    branch_certificate={"passed": True},
)

assert certificate.passed
assert guard.passed
print(certificate.estimate, guard.certified_bound)
```

---

## Topic 14: Nonlinear regime atlases and local model banks

### Idea

A single global reduced basis may be inefficient or inaccurate for strongly
nonlinear trajectories.  Local model banks use different reduced models for
different regimes.  A regime atlas defines those regimes using generic,
online-available features and then validates whether the local model in each
region is accurate enough to deploy.

### First-principles derivation

Build a feature vector for each candidate online stage:

$$
z_i=
\left[
z_{i1},
z_{i2},
\ldots,
z_{id}
\right].
$$

The columns are application-defined: parameters, time, previous reduced state,
residual indicators, sensor values, continuation step data, or any other
quantity available before selecting the local model.

An atlas proposes regions:

$$
z_i\in\Omega_r.
$$

Each region may get its own trial basis, sampling rows, decoder, regression
map, cubature weights, and calibration state.  A deployable region must pass a
validation gate:

$$
E_r^{\mathrm{val}}\le \tau.
$$

During online deployment, select a bank only if the feature lies inside its
certified radius:

$$
\left\|
\frac{z_{\mathrm{online}}-z_{\mathrm{center}}}{z_{\mathrm{scale}}}
\right\|_2
\le d_{\max}.
$$

Use principal angles to decide whether neighboring regions truly need separate
trial bases.  If $\Phi_a^T\Phi_b$ has singular values $\sigma_i$, then

$$
\theta_i=\arccos(\sigma_i).
$$

The chordal distance is

$$
d_c=\sqrt{\sum_i\sin^2\theta_i}.
$$

Small angles mean bases are similar; large angles indicate genuinely different
local spaces.

### API mapping

The full regime-discovery implementation lives in
`pycutfem.mor.regime_atlas`.  See
`pycutfem/mor/regime_atlas/README.md` for the complete help-file treatment.
The older `pycutfem.mor.feature_atlas` and `pycutfem.mor.local_banks` public
modules have been removed; use `pycutfem.mor.regime_atlas` for atlas and bank
imports.

| Concept | API object |
| --- | --- |
| dataset contract | `RegimeDataset`, `RegimeAtlas`, `RegimeRegion` |
| feature scaling | `robust_feature_center_scale`, `scale_feature_matrix` |
| geometry partitioning | `KMedoidsPartitioner`, `EpsilonCoverPartitioner`, `HierarchicalPartitioner` |
| outlier and soft partitioning | `DensityPartitioner`, `MixturePartitioner` |
| error-driven splitting | `ResidualGreedySplitter`, `ResidualGreedyConfig` |
| atlas validation and selection | `RegimeValidationSummary`, `RegimeAtlasSelector` |
| manifest conversion | `build_regime_bank_manifest`, `feature_atlas_to_bank_manifest` |
| online novelty and bank selection | `RegimeOnlineSelector`, `select_local_reduced_model_bank` |
| subspace comparison | `subspace_principal_angles`, `subspace_chordal_distance` |

### Expected result

The example builds two synthetic regimes, validates that the local atlas has
two regions, selects an online feature inside one certified radius, and rejects
an unsupported feature.

### Example

```python
import numpy as np
from pycutfem.mor import (
    KMedoidsPartitioner,
    RegimeOnlineSelector,
    subspace_chordal_distance,
    subspace_principal_angles,
)

features = np.vstack(
    [
        np.linspace(-1.0, -0.8, 12),
        np.linspace(0.8, 1.0, 12),
    ]
).reshape(-1, 1)

atlas = KMedoidsPartitioner(n_regions=2, radius_quantile=1.0).fit(features)
selector = RegimeOnlineSelector(atlas=atlas, fallback_policy={"kind": "global_rom"})

inside = selector.select(feature=np.array([-0.9]))
outside = selector.select(feature=np.array([10.0]))

assert atlas.n_regions == 2
assert inside.selected
assert outside.reason == "outside_certified_region"

rng = np.random.default_rng(7)
Qa, _ = np.linalg.qr(rng.normal(size=(8, 3)))
Qb, _ = np.linalg.qr(rng.normal(size=(8, 3)))
print(atlas.support_counts, inside.region_id, outside.reason)
print(subspace_principal_angles(Qa, Qb), subspace_chordal_distance(Qa, Qb))
```

---

## Topic 15: Sparse GNAT lift

### Idea

Some hyper-reduced operators are represented by sparse lifting maps.  A sparse
GNAT lift maps sampled residual rows into a lifted residual space without
forming dense matrices.

### First-principles derivation

Let $L$ be a sparse lift matrix.  Given sampled residual $r_s$ and sampled trial
Jacobian $A_s$, the lifted objects are

$$
\tilde r=Lr_s,
\qquad
\tilde A=LA_s.
$$

The corresponding normal equations are

$$
\tilde A^T\tilde A\Delta q=-\tilde A^T\tilde r.
$$

Sparse storage matters when the sampled system is still large enough that dense
lifts would dominate memory or runtime.

### API mapping

| Math object | API object |
| --- | --- |
| sparse lift $L$ | `NativeSparseMatrix` |
| $Lr_s$, $LA_s$ | `apply_sparse_gnat_lift(...)` |
| normal equations | `sparse_gnat_normal_equations(...)` |

### Expected result

The example converts a dense toy lift into CSR-like native sparse form, applies
it to a sampled residual and sampled trial matrix, and builds a two-by-two
normal matrix.

### Example

```python
import numpy as np
from pycutfem.mor import NativeSparseMatrix, apply_sparse_gnat_lift, sparse_gnat_normal_equations

lift = NativeSparseMatrix.from_dense(
    np.array([[1.0, 0.0, 0.5], [0.0, 2.0, 0.0]])
)
sampled_residual = np.array([1.0, -1.0, 0.5])
sampled_trial = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

residual, trial = apply_sparse_gnat_lift(lift, sampled_residual, sampled_trial)
normal = sparse_gnat_normal_equations(lift, sampled_residual, sampled_trial)

assert lift.nnz == 3
assert normal["normal_matrix"].shape == (2, 2)
print(lift.nnz, residual, trial.shape, normal["normal_matrix"])
```

---

## Topic 16: Native artifacts and native online solve pattern

### Idea

Offline training happens in Python, but production online nonlinear iterations
should not call Python inside every residual/tangent evaluation.  A native
artifact records the problem-independent reduced model, while live generated
kernel capsules are attached at runtime.

### First-principles structure

A native reduced model is the data needed to evaluate one of these targets:

Galerkin:

$$
g=V^TR,
\qquad
G=V^TJV.
$$

Full-row LSPG:

$$
r=R,
\qquad
A=JV.
$$

Sampled GNAT:

$$
r_s=W_s^{1/2}SR,
\qquad
A_s=W_s^{1/2}SJV.
$$

Therefore the artifact must store at least:

- the trial basis $V$,
- the offset $x_0$,
- residual and tangent kernel references,
- sampled row ids and row weights for GNAT,
- decoded bounds if used,
- state-update metadata,
- reference-policy metadata,
- adjoint/DWR metadata if certification is used,
- solver options and free-form metadata.

The artifact should not store live C++ function pointers.  It stores references
and data.  The runtime attaches compiled kernel capsules.

### API mapping

| Concept | API object |
| --- | --- |
| kernel reference | `NativeKernelReference` |
| sampled GNAT target | `NativeGnatTargetSpec` |
| artifact schema | `NativeReducedArtifact` |
| save/load | `save_native_reduced_artifact`, `load_native_reduced_artifact` |
| online drivers | `solve_native_online_gauss_newton`, variants |
| convergence status | `native_online_convergence_status` |

### Expected result

The artifact example builds a small native reduced artifact, saves it, reloads
it, and verifies that the problem id and sampled rows are preserved.

### Example: artifact save/load

```python
import numpy as np
from pycutfem.mor import (
    NativeGnatTargetSpec,
    NativeKernelReference,
    NativeReducedArtifact,
    save_native_reduced_artifact,
    load_native_reduced_artifact,
)

artifact = NativeReducedArtifact(
    problem_id="toy_problem",
    trial_basis=np.eye(4, 2),
    offset=np.zeros(4),
    residual_kernel=NativeKernelReference(
        kernel_id="residual_kernel",
        abi="pycutfem_native_v1",
        param_order=("mu",),
    ),
    tangent_kernel=NativeKernelReference(
        kernel_id="tangent_kernel",
        abi="pycutfem_native_v1",
        param_order=("mu",),
    ),
    target=NativeGnatTargetSpec(
        row_dofs=np.array([0, 2]),
        row_weights=np.ones(2),
        objective="sampled_lspg",
    ),
    solver_options={"max_iterations": 8, "optimality_tol": 1.0e-10},
)

save_native_reduced_artifact(artifact, "/tmp/native_reduced_artifact.npz")
loaded = load_native_reduced_artifact("/tmp/native_reduced_artifact.npz")
assert loaded.problem_id == "toy_problem"
print(loaded.problem_id, loaded.target.row_dofs)
```

### Native online call pattern

The following is a **non-runnable integration pattern**.  It shows how the
artifact data are used once the PDE/example layer has compiled generated
residual and tangent kernels and has prepared kernel static arguments.

```python
from pycutfem.mor import native_online_convergence_status, solve_native_online_gauss_newton

# These four objects come from the PDE/example layer, not from pycutfem.mor:
# - residual_metadata_capsule: generated residual-kernel metadata/capsule
# - tangent_metadata_capsule: generated tangent-kernel metadata/capsule
# - residual_static_args: static arrays for residual evaluation
# - tangent_static_args: static arrays for tangent evaluation
#
# A typical example builds them from a generated UFL kernel runner:
#
# residual_metadata_capsule = native_kernel_metadata_from_runner(residual_runner)
# tangent_metadata_capsule = native_kernel_metadata_from_runner(tangent_runner)
# residual_static_args = residual_bundle.static_args
# tangent_static_args = tangent_bundle.static_args
#
# The remaining arrays are produced during offline MOR training:
# - trial_basis: POD/fieldwise reduced basis V
# - offset: lifting/mean state x0
# - q0: initial reduced coefficients
# - sampled_rows, sampled_weights: certified GNAT/LSPG sampling data

result = solve_native_online_gauss_newton(
    residual_metadata_capsule=residual_metadata_capsule,
    residual_param_order=("mu",),
    residual_static_args=residual_static_args,
    tangent_metadata_capsule=tangent_metadata_capsule,
    tangent_param_order=("mu",),
    tangent_static_args=tangent_static_args,
    trial_basis=trial_basis,
    offset=offset,
    initial_coefficients=q0,
    row_dofs=sampled_rows,
    row_weights=sampled_weights,
    coefficient_arg_names=("u_k_loc",),
    max_iterations=8,
    residual_tol=1.0e-10,
    optimality_tol=1.0e-10,
    line_search=True,
)

status = native_online_convergence_status(
    result,
    residual_tol=1.0e-10,
    optimality_tol=1.0e-10,
    step_tol=1.0e-12,
)
```

For LSPG/GNAT, prefer an optimality check such as $\|A^Tr\|$ or its sampled
weighted equivalent.  Raw residual norm alone is not always a reliable
convergence indicator for least-squares reduced solves.

---

## Topic 17: Timing and readiness gates

### Idea

A ROM is not useful merely because it is accurate.  It must also be cheaper than
the trusted exact stage and must pass the relevant accuracy/stability/certificate
gates.  Timing and readiness checks prevent reporting misleading speedups.

### First-principles derivation

A measured speedup is

$$
S=\frac{T_{\mathrm{FOM}}}{T_{\mathrm{ROM}}}.
$$

A reduced stage passes a break-even gate only if

$$
S\ge S_{\min}
$$

on enough timing samples.  Readiness is a logical certificate combining many
conditions:

$$
\text{ready}
=
\text{accuracy}
\land
\text{projection}
\land
\text{bounds}
\land
\text{sampling}
\land
\text{DWR}
\land
\text{speedup}
\land
\text{metadata}.
$$

The exact gates depend on the configured criteria.

### API mapping

| Concept | API object |
| --- | --- |
| measured speedup | `speedup`, `build_speedup_report` |
| break-even certificate | `build_stage_break_even_certificate` |
| readiness criteria | `MORReadinessCriteria` |
| readiness certificate | `certify_mor_readiness` |

### Expected result

The example gives reduced timings faster than exact timings, so the cost gate
passes.  The validation summary also satisfies the configured readiness criteria,
so the readiness certificate passes.

### Example

```python
from pycutfem.mor import (
    MORReadinessCriteria,
    build_stage_break_even_certificate,
    build_speedup_report,
    certify_mor_readiness,
)

cost = build_stage_break_even_certificate(
    exact_stage_times=[1.0, 1.1, 0.9],
    reduced_stage_times=[0.35, 0.4, 0.38],
    required_speedup=1.5,
    min_samples=3,
)

summary = {
    "passed": True,
    "errors": {
        "relative_state_vs_fom": 0.01,
        "projection_relative_state_vs_fom": 0.02,
        "max_bound_violation": 0.0,
        "field_errors": {
            "velocity": {"max_relative_error": 0.01, "passed": True},
            "pressure": {"max_relative_error": 0.02, "passed": True},
        },
    },
    "speedup": {
        "predictive_validated_factor": cost.speedup,
        "predictive_validation_passed": True,
        "replay_validation": False,
    },
    "dwr": {"certificate": {"passed": True, "estimate": {"effectivity": 1.0}}},
    "offline": {"sampling": {"interface_complete": True, "missing_mandatory_element_count": 0}},
}
criteria = MORReadinessCriteria(
    max_state_error=0.05,
    max_projection_error=0.05,
    min_validated_speedup=1.5,
    require_dwr=True,
    require_artifact_metadata=False,
)
readiness = certify_mor_readiness(summary, criteria=criteria)
report = build_speedup_report(fom_solid_time=10.0, rom_solid_time=3.0)

assert cost.passed
assert readiness.passed
print(cost.passed, readiness.passed, report)
```

---

## Acceptance checklist and common mistakes

### Production acceptance checklist

A reduced model is ready only when:

- FOM training and validation trajectories are stable and reproducible.
- The reduced trajectory is predictive, not exact trajectory replay.
- Per-field projection errors pass.
- Pressure/null gauges pass.
- Mixed coupling rank gates pass.
- Decoded bounds pass.
- Native nonlinear convergence uses residual and optimality gates.
- GNAT sampling passes residual norm-equivalence.
- DWR effectivity and guard checks pass for selected quantities of interest.
- The native artifact can be saved and loaded.
- Validated online speedup is positive and clears the required threshold.
- `certify_mor_readiness` passes before moving to a harder physical model.

### Common mistakes

- Starting with sparse GNAT before validating full-row LSPG or Galerkin.
- Treating the sampling matrix `S` as a basis.  It is only a row selector.
- Ignoring row weights in mixed systems.
- Reporting POD reconstruction error as if it certified nonlinear online accuracy.
- Fitting pressure POD modes before gauge correction.
- Trusting DWR when the primal solve is on the wrong branch.
- Reporting speedup without including artifact reload, native-kernel, sampling,
  convergence, and certification checks.

---

## Module Inventory

The names below are the public MOR API.  Unless noted, import them from the package root, for example `from pycutfem.mor import fit_pod, gauss_newton_step`.


### `pod.py`

- `PODBasis`
- `fit_pod`
- `project`
- `project_to_basis`
- `reconstruct`
- `reconstruct_from_basis`

### `scaling.py`

- `MeanCenterer`
- `StandardScaler`

### `cross_validation.py`

- `ModeSweepEntry`
- `ModeSweepResult`
- `run_mode_cross_validation`

### `mixed_reduction.py`

- `MixedBasisBlock`
- `FieldwisePODBasis`
- `LiftEnrichment`
- `SupremizerEnrichment`
- `NonAffineReducedDecomposition`
- `PressureGaugeBlock`
- `GaugeCorrectionResult`
- `FieldProjectionError`
- `CouplingRankCertificate`
- `MixedStabilityCertificate`
- `field_dof_indices`
- `build_dirichlet_lifting_vector`
- `remove_lifting_from_snapshots`
- `restore_lifting_to_snapshots`
- `solve_coupled_lift_snapshots`
- `compute_supremizer_snapshots`
- `orthonormalize_columns`
- `fit_lift_enriched_basis`
- `fit_supremizer_enriched_velocity_basis`
- `fit_fieldwise_pod_basis`
- `build_mixed_field_basis`
- `build_mixed_velocity_pressure_basis`
- `pressure_gauge_history`
- `gauge_correct_snapshots`
- `field_projection_errors`
- `reduced_coupling_rank_certificate`
- `certify_mixed_stability_basis`
- `build_block_row_weights`
- `build_nonaffine_reduced_decomposition`

### `decomposition.py`

- `CollateralBasis`
- `InterpolationRule`
- `EmpiricalCubatureRule`
- `ReducedOperatorTerm`
- `NativeReducedEvaluationGraph`
- `fit_collateral_basis`
- `select_deim_rows`
- `select_qdeim_rows`
- `build_interpolation_rule`
- `build_deim_interpolation_rule`
- `build_qdeim_interpolation_rule`
- `interpolation_coefficients`
- `reconstruct_from_interpolation`
- `compose_reduced_operator`

### `sampling.py`

- `SamplingBlock`
- `BlockBalancedGnatSampling`
- `ResidualNormEquivalenceCertificate`
- `AugmentedNormEquivalenceResult`
- `field_row_blocks`
- `support_element_ids_from_rows`
- `rows_supported_on_elements`
- `select_coordinate_band_elements`
- `build_block_balanced_gnat_sampling`
- `certify_sampled_residual_norm_equivalence`
- `augment_rows_for_residual_norm_equivalence`

### `empirical_cubature.py`

- `EmpiricalCubatureFit`
- `fit_positive_empirical_cubature`
- `apply_empirical_cubature`

### `quantity.py`

- `GappyPODQuantityOperator`
- `fit_gappy_pod_quantity_operator`

### `sampled_kernels.py`

- `NativeSampledKernelBundle`
- `build_sampled_static_args`
- `build_sampled_native_kernel_bundle`
- `build_sampled_native_kernel_bundle_from_runner`

### `gauss_newton.py`

- `GaussNewtonNormalEquations`
- `GaussNewtonStepResult`
- `form_normal_equations`
- `gauss_newton_step`

### `constrained_gauss_newton.py`

- `EqualityConstrainedGaussNewtonStepResult`
- `ConstrainedGaussNewtonBackend`
- `ConstrainedGaussNewtonMethod`
- `equality_constrained_gauss_newton_step`

### `constraints.py`

- `BoundActivity`
- `ActiveBoundEquations`
- `ReducedBoundConstraintSpec`
- `BoundConstraintSpec`
- `bound_constraints_from_fields`
- `project_reduced_coefficients_to_bounds`

### `globalization.py`

- `BranchGlobalizationSpec`
- `ContinuationAttempt`
- `ContinuationResult`
- `clip_step_to_trust_region`
- `step_alpha_to_branch_radius`
- `solve_with_branch_backtracking`

### `predictors.py` and `reference.py`

- `ReducedReferencePrediction`
- `ConstantReducedPredictor`
- `LinearHistoryReducedPredictor`
- `TimeParameterizedReducedPredictor`
- `fit_time_parameterized_predictor`
- `predictor_from_native_dict`
- `ReferencePolicy`
- `ReferencePolicyResult`
- `clip_reference_distance`

### `state_updates.py`

- `NativeStateArraySpec`
- `AffineStateUpdateSpec`
- `SymbolicStateUpdateKernelSpec`
- `NativeStateUpdateKernelCall`
- `StateTransactionSpec`
- `coerce_affine_state_update`
- `coerce_affine_state_updates`
- `apply_affine_state_updates`
- `build_dirichlet_lift_state_updates`

### `transactions.py`

- `MutableStateSnapshot`
- `SampleStateTransaction`
- `SampleStateTrial`

### `native_assembly.py`

- `native_kernel_metadata_from_runner`
- `call_native_kernel`
- `reduced_target_from_native_kernel_pair`
- `gnat_system_from_native_kernel`
- `sampled_lspg_rows_from_native_kernel`
- `sampled_galerkin_reduced_system_from_native_kernel`
- `apply_affine_updates_to_static_args`

### `reduced_assembly.py`

- `AffineReducedState`
- `ReducedLocalAssembler`
- `validate_local_blocks`
- `validate_element_weights`
- `decode_element_values`
- `decode_values_on_dofs`
- `sampled_lspg_rows_from_local_blocks`
- `sampled_lspg_element_contributions_from_local_blocks`
- `sampled_galerkin_reduced_system_from_local_blocks`
- `sampled_galerkin_element_contributions_from_local_blocks`
- `sampled_galerkin_reduced_system_from_native_kernel`
- `reduced_reaction_from_local_blocks`
- `constrained_reaction_rows_from_local_blocks`
- `apply_gnat_lift`

### `online_gauss_newton.py`

- `NativeOnlineGaussNewtonResult`
- `NativeOnlineConvergenceStatus`
- `native_online_convergence_status`
- `solve_native_online_gauss_newton`
- `solve_native_deim_online_gauss_newton`
- `solve_native_bound_constrained_online_gauss_newton`
- `solve_native_bound_constrained_deim_online_gauss_newton`
- `solve_native_bound_constrained_galerkin_online_gauss_newton`

### `sparse.py`

- `NativeSparseMatrix`
- `is_sparse_matrix_like`
- `apply_sparse_gnat_lift`
- `sparse_gnat_normal_equations`

### `artifacts.py`

- `NATIVE_REDUCED_ARTIFACT_SCHEMA_VERSION`
- `NativeKernelReference`
- `NativeGnatTargetSpec`
- `NativeAdjointDWRSpec`
- `NativeReducedArtifact`
- `NativeReducedRuntimeProblem`
- `save_native_reduced_artifact`
- `load_native_reduced_artifact`

### `timing.py`

- `Timer`
- `TimingAccumulator`
- `StageBreakEvenCertificate`
- `build_speedup_report`
- `build_stage_break_even_certificate`

### `dwr.py`

- `QoIFunctionalSpec`
- `QoIKernelSpec`
- `QoIStatePolicy`
- `TransientResidualDependencySpec`
- `QoIGradientCheck`
- `DiscreteAdjointResult`
- `DWREstimate`
- `DWRCertificationResult`
- `DWRReducedTrajectory`
- `DWRGuardResult`
- `solve_transpose_system`
- `solve_discrete_adjoint`
- `solve_reduced_discrete_adjoint`
- `finite_difference_gradient`
- `check_qoi_gradient`
- `linearize_qoi_functional`
- `evaluate_qoi_functional`
- `assemble_qoi_gradient`
- `reduced_qoi_gradient_from_full`
- `dual_weighted_residual_estimate`
- `dominant_dwr_contributions`
- `dwr_certification_guard`
- `save_dwr_reduced_trajectory`
- `load_dwr_reduced_trajectory`
- `certify_dual_weighted_residual`
- `certify_dual_weighted_residual_from_artifact_trajectory`

### `adaptive.py`

- `AdaptiveEnrichmentAction`
- `AdaptiveMORDecision`
- `OnlineErrorCalibrationDecision`
- `OnlineErrorCalibrator`
- `select_certified_adaptive_enrichment_actions`
- `augment_rows_from_dwr_localization`

### `regime_atlas/`

- `RegimeDataset`
- `RegimeRegion`
- `RegimeAtlas`
- `RegimeValidationSplit`
- `RegimeValidationReport`
- `RegimeValidationSummary`
- `RegimePartitioner`
- `RegimePartitionerConfig`
- `KMedoidsPartitioner`
- `EpsilonCoverPartitioner`
- `HierarchicalPartitioner`
- `DensityPartitioner`
- `MixturePartitioner`
- `TreeRouter`
- `SubspacePartition`
- `SubspacePartitioner`
- `ResidualGreedyConfig`
- `ResidualGreedySplitEvent`
- `ResidualGreedyResult`
- `ResidualGreedySplitter`
- `RegimeAtlasCandidate`
- `RegimeAtlasSelection`
- `RegimeAtlasSelector`
- `RegimeBankManifest`
- `RegimeNoveltyDecision`
- `RegimeOnlineSelector`
- `LocalReducedModelBankEntry`
- `LocalReducedModelSelection`
- `as_feature_matrix`
- `as_index_vector`
- `make_validation_split`
- `summarize_region_errors`
- `boundary_halo_score`
- `coerce_regime_dataset`
- `normalize_region_labels`
- `labels_to_atlas`
- `make_regime_partitioner`
- `fit_kmedoids_regime_atlas`
- `subspace_distance_matrix`
- `build_regime_bank_manifest`
- `load_regime_bank_manifest`
- `load_local_reduced_model_bank_manifest`
- `select_local_reduced_model_bank`
- `FeatureAtlasDiagnostics`
- `FeatureAtlasFit`
- `FeatureAtlasRegion`
- `FeatureAtlasSizeSelection`
- `KMedoidsResult`
- `diagnose_feature_atlas`
- `feature_atlas_to_bank_manifest`
- `fit_feature_atlas`
- `fit_k_medoids`
- `robust_feature_center_scale`
- `scale_feature_matrix`
- `select_feature_atlas_size`
- `subspace_chordal_distance`
- `subspace_principal_angles`

The old `feature_atlas.py` and `local_banks.py` public shims have been removed.
Import atlas and bank tools from `pycutfem.mor.regime_atlas` or the top-level
`pycutfem.mor` namespace.

### `readiness.py`

- `MORReadinessGate`
- `MORReadinessCriteria`
- `MORReadinessCertificate`
- `certify_mor_readiness`

### `interface.py`

- `InterfaceRestriction`
- `build_interface_restriction`
- `build_restriction_matrix`

### `quadratic_manifold.py`

- `QuadraticFeatureMap`
- `QuadraticManifoldDecoder`
- `quadratic_feature_matrix`
- `fit_quadratic_decoder`
- `fit_quadratic_manifold`

### `regressors.py`

- `PolynomialFeatureMap`
- `PolynomialLeastSquaresRegressor`
- `PolynomialLassoRegressor`
- `ThinPlateSplineRBF`
- `fit_poly_least_squares`
- `fit_poly_lasso`
- `fit_tps_rbf`

### `metrics.py`, `timing.py`, `snapshots.py`, and `io.py`

- `mean_sample_l2_error`
- `snapshot_l2_error`
- `reduced_regression_error`
- `online_relative_displacement_error`
- `max_online_relative_displacement_error`
- `accumulated_iteration_overhead`
- `speedup`
- `Timer`
- `TimingAccumulator`
- `build_speedup_report`
- `NamedSnapshotBatch`
- `NamedSnapshotReader`
- `NamedSnapshotWriter`
- `SnapshotBatch`
- `SnapshotReader`
- `SnapshotWriter`
- `load_config`
- `save_config`
- `load_model`
- `save_model`
- `save_results`

### `nirb/`

The generic non-intrusive reduced-basis workflow lives in `pycutfem.mor.nirb`.
It uses problem-independent input/output names; FSI-specific loaders and
solid/interface adapters live in `examples/utils/nirb`.  See
`pycutfem/mor/nirb/README.md` for the self-contained NIRB help file.

- `NIRBDataset`
- `OfflineConfig`
- `OnlineConfig`
- `RegressionConfig`
- `TrainedNIRBModel`
- `ReducedSpace`
- `ReducedTransfer`
- `ReducedOutputDecoder`
- `ReducedIQNILS`
- `run_offline_pipeline`
- `run_online_pipeline`
- `validate_rom`

### `incompressible.py`

This is a compatibility module that re-exports velocity-pressure helper names
from `mixed_reduction.py`.  New code should import from `pycutfem.mor`
or `pycutfem.mor.mixed_reduction` directly.

### `cpp_backend/`

The C++ backend modules provide native kernels for dense Gauss-Newton,
constrained Gauss-Newton, reduced projection, sparse GNAT, native reduced
assembly, DEIM online evaluation, online Gauss-Newton, and adjoint algebra.
Users normally call the high-level Python wrappers listed above; the wrappers
compile/import the C++ extensions as needed.

## Efficient Usage Rules

- Keep offline training in Python, but keep online nonlinear loops in native C++
  once the artifact is built.
- Use fieldwise POD for mixed systems.
- Gauge-correct pressure-like fields before POD and before QoI certification.
- Use lift/supremizer enrichment when a reduced coupling rank gate fails.
- Use full-row LSPG or true Galerkin as the diagnostic baseline before sparse
  GNAT.
- Do not enable sparse GNAT unless norm-equivalence passes on the validation
  neighborhood.
- Use `ReferencePolicy` for transient nonlinear branch selection.
- Use decoded bounds for physical constraints such as volume fractions,
  concentrations, damage, or detachment variables.
- Treat DWR as a certificate and adaptivity signal, not as a replacement for a
  branch-correct primal solve.
- Report validated speedup only when state, bounds, branch, gauge, DWR, and
  sampling gates pass.

## Production Acceptance Checklist

A reduced model is production-ready only when:

- the FOM trajectory is stable and reproducible,
- the reduced trajectory is predictive, not exact trajectory replay,
- per-field projection errors pass,
- mixed stability and gauge gates pass,
- decoded bounds pass,
- native nonlinear convergence uses the correct residual/optimality gate,
- sampled GNAT passes residual norm-equivalence,
- DWR effectivity and guard checks pass for the selected QoIs,
- the native artifact can be reloaded,
- validated online speedup is positive,
- `certify_mor_readiness` passes before moving to the next harder model.
