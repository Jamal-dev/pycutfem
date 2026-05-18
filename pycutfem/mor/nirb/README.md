# NIRB: Non-Intrusive Reduced-Basis Models

`pycutfem.mor.nirb` contains the generic non-intrusive reduced-basis tools used
by `pycutfem.mor`.  The package is deliberately written in input-output
language.  A NIRB model does not know whether the input is a pressure load, a
material parameter, a boundary condition, a sensor vector, an interface force,
or a reduced coordinate.  It learns a map from one snapshot matrix to another.

Application-specific words such as *interface load*, *solid displacement*,
*fluid reaction*, or *co-simulation data* belong in examples and adapters.  For
FSI examples in this repository, those adapters live under
`examples/utils/nirb`.

This README is written like a help page.  Each topic follows the same pattern:

```text
Idea -> first-principles derivation -> API mapping -> expected result -> example
```

The goal is that a reader who has not seen NIRB, POD, reduced coordinates, or
non-intrusive regression can still follow the complete workflow.

Run standalone examples from the repository root with:

```bash
conda run --no-capture-output -n fenicsx python your_script.py
```

Most examples below use only `numpy`, `pycutfem.mor.snapshots`, and
`pycutfem.mor.nirb`.  Snippets that need example-layer FSI data or a coupled
solver are explicitly marked as **pseudocode**.

---

## Contents

1. [What NIRB solves](#what-nirb-solves)
2. [Topic: Dataset format and the meaning of input-output learning](#topic-dataset-format-and-the-meaning-of-input-output-learning)
3. [Topic: Reduced coordinates for inputs and outputs](#topic-reduced-coordinates-for-inputs-and-outputs)
4. [Topic: Offline NIRB training](#topic-offline-nirb-training)
5. [Topic: Regression maps from first principles](#topic-regression-maps-from-first-principles)
6. [Topic: Quadratic reduced maps and quadratic output decoders](#topic-quadratic-reduced-maps-and-quadratic-output-decoders)
7. [Topic: Context features](#topic-context-features)
8. [Topic: Online prediction](#topic-online-prediction)
9. [Topic: Restricted outputs](#topic-restricted-outputs)
10. [Topic: Reduced spaces and reduced transfers](#topic-reduced-spaces-and-reduced-transfers)
11. [Topic: Reduced output decoders](#topic-reduced-output-decoders)
12. [Topic: Reduced IQN-ILS coupling acceleration](#topic-reduced-iqn-ils-coupling-acceleration)
13. [Topic: Validation and acceptance](#topic-validation-and-acceptance)
14. [Topic: Example-level FSI adapters](#topic-example-level-fsi-adapters)
15. [Recommended workflow](#recommended-workflow)
16. [Common mistakes](#common-mistakes)
17. [Complete API inventory](#complete-api-inventory)

---

## What NIRB Solves

### Idea

A full simulation often gives paired data:

- an input vector `x`, such as a load, parameter vector, sensor vector, boundary
  state, or reduced coupling variable;
- an output vector `y`, such as a displacement, solution field, correction,
  reaction vector, or quantity-related field.

A non-intrusive reduced-basis model learns the map

$$
y \approx F(x,c),
$$

where `c` is an optional context vector such as time, Reynolds number, material
parameter, coupling iteration, or regime feature.

The word **non-intrusive** means that the model does not need residuals,
Jacobians, element kernels, weak forms, or access to the PDE solver internals.
It only needs input-output training pairs.

### First-principles derivation

Assume we have `N` training samples:

$$
x_j \in \mathbb{R}^{n_x},
\qquad
y_j \in \mathbb{R}^{n_y},
\qquad
j=1,\ldots,N.
$$

Collect them in feature-major snapshot matrices:

$$
X = [x_1,x_2,\ldots,x_N] \in \mathbb{R}^{n_x\times N},
\qquad
Y = [y_1,y_2,\ldots,y_N] \in \mathbb{R}^{n_y\times N}.
$$

The direct learning problem would be

$$
y \approx F(x,c),
$$

but this is difficult when `x` and `y` are high-dimensional.  NIRB therefore
uses reduced coordinates.

First compress the input:

$$
x \approx \bar{x} + V_x z_x,
$$

where

- `bar{x}` is the input mean or offset,
- `V_x` is the input reduced basis,
- `z_x` is the low-dimensional input coordinate vector.

Then compress or decode the output:

$$
y \approx D_y(z_y),
$$

where `D_y` may be linear or nonlinear.

Instead of learning the full map `x -> y`, NIRB learns the reduced map

$$
z_y \approx \mathcal{G}(z_x,c).
$$

The complete online prediction is the composition

$$
\hat{y}(x,c)
=
D_y\!\left(
\mathcal{G}\!\left(
V_x^T(x-\bar{x}),c
\right)
\right),
$$

when the input basis is Euclidean-orthonormal.  With a mass matrix, the
projection formula is modified as described later.

This composition explains the whole package:

```text
full input x
    -> project to reduced input coordinates z_x
    -> regress reduced output coordinates z_y
    -> decode to full output y_hat
```

### API mapping

The main NIRB objects are:

- `NIRBDataset`: stores paired input-output snapshots.
- `OfflineConfig`: tells the offline pipeline how many input/output modes to use
  and which regressor to train.
- `RegressionConfig`: selects the reduced map type, for example polynomial,
  radial basis, or nearest-neighbor regression.
- `TrainedNIRBModel`: stores the fitted input basis, output decoder, regressor,
  context statistics, and optional output restriction.
- `run_offline_pipeline`: trains a model from a dataset file.
- `run_online_pipeline` or `model.predict`: predicts outputs for new inputs.

### Expected result

A trained NIRB model should be understood as an approximation of the training
map.  It is reliable when new inputs are inside or close to the training
distribution.  If a new input is far away from the training data, the model may
still return a vector, but that vector is extrapolation and should be treated as
uncertified.

### Example: the complete map in one toy problem

This first example is intentionally small.  It creates a synthetic input-output
map, trains a NIRB model, and predicts the output for the same training inputs.
The error should be small because the data is generated from a smooth polynomial
map and the regressor is allowed to use polynomial features.

```python
from pathlib import Path

import numpy as np

from pycutfem.mor.nirb import OfflineConfig, RegressionConfig, run_offline_pipeline
from pycutfem.mor.snapshots import NamedSnapshotBatch

rng = np.random.default_rng(0)

# Reduced latent variable used only to build a clean synthetic example.
z = rng.normal(size=(2, 80))

# Full input lives in R^4 but has only two active directions.
input_basis = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.0, -0.25],
    ]
)
inputs = input_basis @ z

# Full output lives in R^3 and depends quadratically on z.
outputs = np.vstack(
    [
        1.5 * z[0] - 0.25 * z[1] + 0.75 * z[0] * z[1],
        0.5 * z[1] ** 2,
        -0.2 * z[0] + 0.1 * z[1],
    ]
)

dataset_path = Path("tmp_nirb_basic_dataset.npz")
model_path = Path("tmp_nirb_basic_model.pkl")

NamedSnapshotBatch(fields={"input": inputs, "output": outputs}).save(dataset_path)

model = run_offline_pipeline(
    OfflineConfig(
        dataset_path=str(dataset_path),
        model_path=str(model_path),
        input_modes=2,
        output_modes=3,
        regression=RegressionConfig(kind="poly_ls", degree=2, standardize_inputs=False),
        use_quadratic_decoder=False,
    )
)

prediction = model.predict(inputs)
relative_error = np.linalg.norm(prediction - outputs) / np.linalg.norm(outputs)

print(relative_error)
assert relative_error < 1.0e-10
```

---

## Topic: Dataset Format And The Meaning Of Input-Output Learning

### Idea

A NIRB dataset is a set of paired columns:

```text
input snapshot j  ->  output snapshot j
```

The package does not attach physical meaning to these arrays.  The fields may
be named `input` and `output`, or they may have application names such as
`interface_load` and `solid_displacement`.  The training algorithm only needs to
know which field is the input and which field is the output.

### First-principles derivation

Each sample is a pair `(x_j, y_j)`.  The training problem is empirical:

$$
\min_{\widehat{F}}
\sum_{j=1}^{N}
\left\|y_j - \widehat{F}(x_j,c_j)\right\|^2.
$$

However, `x_j` and `y_j` can be large finite element vectors.  Therefore the
actual loss is usually solved in reduced coordinates:

$$
\min_{\mathcal{G}}
\sum_{j=1}^{N}
\left\|z_{y,j} - \mathcal{G}(z_{x,j},c_j)\right\|^2.
$$

The dataset must therefore preserve column alignment:

$$
X[:,j] \leftrightarrow Y[:,j].
$$

If the columns are shuffled independently, the model learns the wrong map.

The convention is **feature-major**:

```text
shape(input_snapshots)  = (n_input_dofs,  n_samples)
shape(output_snapshots) = (n_output_dofs, n_samples)
```

This is the same convention used by POD snapshot matrices.

### API mapping

Use `NamedSnapshotBatch` from `pycutfem.mor.snapshots` to store fields.  Convert
or load it through the NIRB dataset tools:

- `dataset_from_named_snapshot_batch`
- `load_dataset`
- `NIRBDataset`
- `DatasetSplit`

Important dataset fields are:

- `input_snapshots`
- `output_snapshots`
- `parameters`
- `times`
- `converged`
- `context_features`
- `output_indices`
- `metadata`

### Expected result

The saved file should contain at least two feature-major matrices with the same
number of columns.  The names can be generic or application-specific.  The
offline pipeline will read the selected input and output fields.

### Example: create a generic NIRB dataset

```python
from pathlib import Path

import numpy as np

from pycutfem.mor.snapshots import NamedSnapshotBatch, NamedSnapshotReader

rng = np.random.default_rng(1)

input_snapshots = rng.normal(size=(4, 50))
output_snapshots = np.vstack(
    [
        2.0 * input_snapshots[0] - input_snapshots[1],
        input_snapshots[2] ** 2,
        np.sin(input_snapshots[3]),
    ]
)

path = Path("tmp_nirb_dataset.npz")
NamedSnapshotBatch(
    fields={
        "input": input_snapshots,
        "output": output_snapshots,
    },
    metadata={"case": "synthetic input-output map"},
).save(path)

batch = NamedSnapshotReader(path).load()

assert batch["input"].shape == (4, 50)
assert batch["output"].shape == (3, 50)
assert batch["input"].shape[1] == batch["output"].shape[1]

print(batch.field_names)
```

### Example: use application-specific field names

The fields do not have to be called `input` and `output`.  When the names are
application-specific, pass them to `OfflineConfig`.

```python
from pathlib import Path

import numpy as np

from pycutfem.mor.nirb import OfflineConfig, RegressionConfig, run_offline_pipeline
from pycutfem.mor.snapshots import NamedSnapshotBatch

rng = np.random.default_rng(2)
interface_load = rng.normal(size=(6, 40))
solid_displacement = np.vstack(
    [
        interface_load[0] + 0.1 * interface_load[1],
        interface_load[2] - interface_load[3],
        0.5 * interface_load[4] ** 2,
    ]
)

path = Path("tmp_named_fsi_like_dataset.npz")
NamedSnapshotBatch(
    fields={
        "interface_load": interface_load,
        "solid_displacement": solid_displacement,
    }
).save(path)

model = run_offline_pipeline(
    OfflineConfig(
        dataset_path=str(path),
        model_path="tmp_named_fsi_like_model.pkl",
        dataset_input_field="interface_load",
        dataset_output_field="solid_displacement",
        input_modes=3,
        output_modes=3,
        regression=RegressionConfig(kind="poly_ls", degree=2),
    )
)

prediction = model.predict(interface_load)
assert prediction.shape == solid_displacement.shape
print(prediction.shape)
```

---

## Topic: Reduced Coordinates For Inputs And Outputs

### Idea

NIRB becomes efficient because it does not learn the full map directly.  It
first converts high-dimensional inputs and outputs into small coordinate
vectors.

For the input:

$$
x \mapsto z_x.
$$

For the output:

$$
y \mapsto z_y,
\qquad
z_y \mapsto \hat{y}.
$$

The input reduced space is used for encoding.  The output reduced space is used
for decoding.

### First-principles derivation

Start with the input snapshots

$$
X = [x_1,\ldots,x_N].
$$

Compute the mean

$$
\bar{x} = \frac{1}{N}\sum_{j=1}^{N} x_j.
$$

Subtract the mean:

$$
X_c = X - \bar{x}\mathbf{1}^T.
$$

The POD basis comes from the singular value decomposition:

$$
X_c = U\Sigma Z^T.
$$

The first `r_x` columns of `U` form the input basis:

$$
V_x = U[:,1:r_x].
$$

If the columns are orthonormal in the Euclidean inner product, the reduced
coordinates are

$$
z_x = V_x^T(x-\bar{x}).
$$

The same construction can be applied to the output snapshots:

$$
y \approx \bar{y}+V_y z_y,
\qquad
z_y = V_y^T(y-\bar{y}).
$$

The POD truncation error for a snapshot matrix is

$$
\frac{\|X_c - V_xV_x^TX_c\|_F}{\|X_c\|_F}.
$$

This error measures whether the chosen reduced basis can represent the training
snapshots.  It does not by itself certify that the learned input-output map is
accurate.

### API mapping

The NIRB offline pipeline builds these spaces internally through `input_modes`
and `output_modes`:

```python
OfflineConfig(input_modes=8, output_modes=6)
```

Lower-level related APIs are:

- `ReducedSpace`
- `ReducedOutputDecoder`
- `fit_pod` from `pycutfem.mor`
- `TrainedNIRBModel.encode_input`
- `TrainedNIRBModel.decode_output`

### Expected result

If the synthetic data has rank two and you request two input modes, the input
projection error should be near machine precision.  If you request too few
modes, the projection error increases.

### Example: input and output coordinates by POD

```python
import numpy as np

from pycutfem.mor import fit_pod

rng = np.random.default_rng(3)

# Build low-rank input data in R^20 from two latent coordinates.
latent = rng.normal(size=(2, 60))
input_basis_true, _ = np.linalg.qr(rng.normal(size=(20, 2)))
output_basis_true, _ = np.linalg.qr(rng.normal(size=(12, 3)))

X = input_basis_true @ latent
Y = output_basis_true @ np.vstack(
    [
        latent[0],
        latent[1],
        0.5 * latent[0] - 0.25 * latent[1],
    ]
)

input_pod = fit_pod(X, n_modes=2, center=True)
output_pod = fit_pod(Y, n_modes=3, center=True)

Zx = input_pod.project(X)
Zy = output_pod.project(Y)
X_rec = input_pod.reconstruct(Zx)
Y_rec = output_pod.reconstruct(Zy)

input_error = np.linalg.norm(X_rec - X) / np.linalg.norm(X)
output_error = np.linalg.norm(Y_rec - Y) / np.linalg.norm(Y)

assert input_error < 1.0e-12
assert output_error < 1.0e-12
print(Zx.shape, Zy.shape, input_error, output_error)
```

---

## Topic: Offline NIRB Training

### Idea

Offline training constructs the complete reduced input-output model.  It does
three things:

1. fit an input basis,
2. fit an output decoder,
3. train a regressor from input coordinates to output coordinates.

After offline training, the expensive high-dimensional learning problem has
been replaced by a low-dimensional map.

### First-principles derivation

Given paired snapshots `X` and `Y`, compute

$$
z_{x,j} = V_x^T(x_j-\bar{x}),
\qquad
z_{y,j} = E_y(y_j),
$$

where `E_y` is the output encoder.  For a linear POD decoder,

$$
E_y(y)=V_y^T(y-\bar{y}).
$$

Collect the reduced coordinates as sample-major training data for the
regressor:

$$
Z_x^{\text{train}}
=
\begin{bmatrix}
--- z_{x,1}^T ---\\
--- z_{x,2}^T ---\\
\vdots\\
--- z_{x,N}^T ---
\end{bmatrix}
\in \mathbb{R}^{N\times r_x},
$$

and

$$
Z_y^{\text{train}}
\in \mathbb{R}^{N\times r_y}.
$$

The reduced map is trained by empirical risk minimization:

$$
\min_{\mathcal{G}\in\mathcal{A}}
\sum_{j=1}^{N}
\left\|
 z_{y,j} - \mathcal{G}(z_{x,j},c_j)
\right\|_2^2
+ \text{regularization},
$$

where `A` is the chosen model class: polynomial, ridge, lasso, RBF, nearest
neighbor, and so on.

The trained model stores

$$
\{\bar{x},V_x,D_y,\mathcal{G},\text{context statistics},\text{metadata}\}.
$$

### API mapping

Use:

- `OfflineConfig` to describe the training job,
- `RegressionConfig` to describe the reduced regressor,
- `run_offline_pipeline` to train,
- `TrainedNIRBModel` as the returned trained model.

Important `OfflineConfig` fields:

- `dataset_path`
- `model_path`
- `input_modes`
- `output_modes`
- `regression`
- `center_input`
- `center_output`
- `use_quadratic_decoder`
- `dataset_input_field`
- `dataset_output_field`
- `zero_anchor_weight`
- `output_indices`
- `output_matrix_path`
- `context_feature_names`
- `metadata`

### Expected result

For a constructed polynomial training map, a polynomial regressor of sufficient
degree should reproduce the training outputs to very small error.  For real
simulation data, the training error should be checked together with validation
error on held-out samples.

### Example: complete offline training

```python
from pathlib import Path

import numpy as np

from pycutfem.mor.nirb import OfflineConfig, RegressionConfig, run_offline_pipeline
from pycutfem.mor.snapshots import NamedSnapshotBatch

rng = np.random.default_rng(4)
z = rng.normal(size=(2, 100))

input_basis = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.0, 0.25],
    ]
)
inputs = input_basis @ z

outputs = np.vstack(
    [
        1.5 * z[0] - 0.25 * z[1] + 0.75 * z[0] * z[1],
        0.5 * z[1] ** 2,
        -0.2 * z[0] + 0.1 * z[1],
    ]
)

dataset_path = Path("tmp_nirb_train.npz")
model_path = Path("tmp_nirb_model.pkl")
NamedSnapshotBatch(fields={"input": inputs, "output": outputs}).save(dataset_path)

model = run_offline_pipeline(
    OfflineConfig(
        dataset_path=str(dataset_path),
        model_path=str(model_path),
        input_modes=2,
        output_modes=3,
        regression=RegressionConfig(kind="poly_ls", degree=2, standardize_inputs=False),
        use_quadratic_decoder=False,
        metadata={"example": "offline polynomial NIRB"},
    )
)

prediction = model.predict(inputs)
relative_error = np.linalg.norm(prediction - outputs) / np.linalg.norm(outputs)

assert relative_error < 1.0e-10
print(relative_error, model.metadata)
```

---

## Topic: Regression Maps From First Principles

### Idea

The regressor is the non-intrusive part of NIRB.  It learns the map

$$
z_y \approx \mathcal{G}(z_x,c)
$$

from data.  Once the input and output have been reduced, the regressor only sees
small vectors.

### First-principles derivation

Let each regressor input be

$$
s_j = [z_{x,j},c_j] \in \mathbb{R}^{d},
$$

where `c_j` may be absent.  Let each target be

$$
t_j = z_{y,j} \in \mathbb{R}^{r_y}.
$$

The training set is

$$
\{(s_j,t_j)\}_{j=1}^{N}.
$$

#### Linear least squares

Assume

$$
\mathcal{G}(s)=a+Bs.
$$

Build a design matrix

$$
H =
\begin{bmatrix}
1 & s_1^T\\
1 & s_2^T\\
\vdots & \vdots\\
1 & s_N^T
\end{bmatrix}.
$$

Collect targets in a sample-major matrix

$$
T =
\begin{bmatrix}
--- t_1^T ---\\
--- t_2^T ---\\
\vdots\\
--- t_N^T ---
\end{bmatrix}.
$$

Least squares solves

$$
\min_A \|HA-T\|_F^2.
$$

The fitted coefficient matrix is used to predict

$$
\hat{t}=h(s)^TA.
$$

#### Polynomial regression

Polynomial regression replaces `s` by a feature vector:

$$
\phi_p(s)=
[1,
 s_1,
 \ldots,
 s_d,
 s_1^2,
 s_1s_2,
 \ldots]^T.
$$

Then

$$
\mathcal{G}(s)=A^T\phi_p(s).
$$

Degree 2 gives a quadratic reduced map.  Degree 3 gives cubic terms, and so on.

#### Ridge and lasso

Ridge regression adds an L2 penalty:

$$
\min_A \|HA-T\|_F^2 + \lambda\|A\|_F^2.
$$

Lasso adds an L1 penalty:

$$
\min_A \|HA-T\|_F^2 + \lambda\|A\|_1.
$$

Ridge is useful when features are correlated or the fit is ill-conditioned.
Lasso is useful when you want a sparse polynomial model.

#### Thin-plate spline RBF

A radial basis model predicts from distances to training samples:

$$
\mathcal{G}(s)
=
A^T[1,s]^T
+
\sum_{j=1}^{N} \alpha_j \varphi(\|s-s_j\|),
$$

where the thin-plate spline kernel is commonly of the form

$$
\varphi(r)=r^2\log(r),
$$

with the convention that the limit at `r=0` is zero.  RBFs are flexible for
smooth scattered data, but can be more expensive than polynomial models.

#### Nearest-neighbor models

A nearest-neighbor model predicts by looking at nearby training points.  The
simplest version returns the target of the nearest sample.  A k-nearest version
averages several nearby targets.

This is useful as a robust baseline, but it is piecewise constant or piecewise
smooth rather than a global polynomial law.

### API mapping

Use `RegressionConfig`:

```python
RegressionConfig(kind="poly_ls", degree=2)
```

Supported kinds include:

- `"poly_ls"`
- `"poly_ridge"`
- `"poly_lasso"`
- `"tps_rbf"`
- `"knn"`
- `"knearest"`
- `"nearest"`

Related lower-level APIs from `pycutfem.mor.regressors` are:

- `PolynomialFeatureMap`
- `PolynomialLeastSquaresRegressor`
- `PolynomialLassoRegressor`
- `ThinPlateSplineRBF`
- `KNearestRegressor`

### Expected result

When the true reduced map is quadratic, a degree-2 polynomial least-squares
regressor should reproduce it accurately.  A linear regressor should leave a
larger error because it cannot represent the quadratic term.

### Example: compare linear and quadratic reduced maps

```python
from pathlib import Path

import numpy as np

from pycutfem.mor.nirb import OfflineConfig, RegressionConfig, run_offline_pipeline
from pycutfem.mor.snapshots import NamedSnapshotBatch

rng = np.random.default_rng(5)
z = rng.normal(size=(2, 120))

inputs = np.vstack([z[0], z[1], 0.25 * z[0]])
outputs = np.vstack(
    [
        z[0] + z[1],
        z[0] * z[1],
    ]
)

dataset_path = Path("tmp_regression_compare.npz")
NamedSnapshotBatch(fields={"input": inputs, "output": outputs}).save(dataset_path)

linear_model = run_offline_pipeline(
    OfflineConfig(
        dataset_path=str(dataset_path),
        model_path="tmp_linear_nirb.pkl",
        input_modes=2,
        output_modes=2,
        regression=RegressionConfig(kind="poly_ls", degree=1, standardize_inputs=False),
    )
)

quadratic_model = run_offline_pipeline(
    OfflineConfig(
        dataset_path=str(dataset_path),
        model_path="tmp_quadratic_nirb.pkl",
        input_modes=2,
        output_modes=2,
        regression=RegressionConfig(kind="poly_ls", degree=2, standardize_inputs=False),
    )
)

linear_error = np.linalg.norm(linear_model.predict(inputs) - outputs) / np.linalg.norm(outputs)
quadratic_error = np.linalg.norm(quadratic_model.predict(inputs) - outputs) / np.linalg.norm(outputs)

print(linear_error, quadratic_error)
assert quadratic_error < linear_error
```

---

## Topic: Quadratic Reduced Maps And Quadratic Output Decoders

### Idea

There are two different places where quadratic terms can appear.  They solve
different problems.

A **quadratic reduced map** makes the learned relation nonlinear:

$$
z_x \mapsto z_y.
$$

A **quadratic output decoder** makes the reconstruction nonlinear:

$$
z_y \mapsto y.
$$

They are independent.  A model may use neither, one, or both.

### First-principles derivation

#### Quadratic reduced map

Suppose the output coordinate depends nonlinearly on the input coordinate:

$$
z_y = a + Bz_x + C\phi_2(z_x).
$$

Here

$$
\phi_2(z_x)=
[z_1^2,z_1z_2,\ldots,z_{r_x}^2]^T
$$

contains upper-triangular products.  This is handled by a polynomial regressor
with `degree=2`.

Use this when the coordinates themselves satisfy a nonlinear relation.

#### Quadratic output decoder

A linear output decoder assumes the output field lies close to an affine linear
space:

$$
y \approx \bar{y}+V_yz_y.
$$

Some solution manifolds are curved.  Then a small linear basis may not
reconstruct the field well.  A quadratic decoder uses

$$
D_y(z_y)
=
\bar{y}+V_yz_y+W_y\phi_2(z_y).
$$

The matrix `W_y` is fitted by least squares from output snapshots.  It reduces
reconstruction error when the output manifold is curved in the ambient full
space.

### API mapping

Use a quadratic reduced map with:

```python
RegressionConfig(kind="poly_ls", degree=2)
```

Use a quadratic output decoder with:

```python
OfflineConfig(use_quadratic_decoder=True)
```

Related lower-level APIs are:

- `QuadraticFeatureMap`
- `QuadraticManifoldDecoder`
- `fit_quadratic_decoder`
- `fit_quadratic_manifold`
- `quadratic_feature_matrix`

### Expected result

If the output is a nonlinear function of the reduced coordinates but still
lives in a low-dimensional linear output space, the quadratic reduced map helps.
If the output field itself lies on a curved manifold, the quadratic decoder
helps.  If both effects are present, use both.

### Example: use both quadratic options

```python
from pathlib import Path

import numpy as np

from pycutfem.mor.nirb import OfflineConfig, RegressionConfig, run_offline_pipeline
from pycutfem.mor.snapshots import NamedSnapshotBatch

rng = np.random.default_rng(6)
z = rng.normal(size=(2, 100))

inputs = np.vstack([z[0], z[1], 0.2 * z[0]])

# Output has nonlinear coordinate dependence and a curved full-field component.
outputs = np.vstack(
    [
        z[0],
        z[1],
        z[0] ** 2,
        z[0] * z[1],
        z[1] ** 2,
    ]
)

dataset_path = Path("tmp_quadratic_options.npz")
NamedSnapshotBatch(fields={"input": inputs, "output": outputs}).save(dataset_path)

model = run_offline_pipeline(
    OfflineConfig(
        dataset_path=str(dataset_path),
        model_path="tmp_quadratic_options_model.pkl",
        input_modes=2,
        output_modes=2,
        regression=RegressionConfig(kind="poly_ls", degree=2, standardize_inputs=False),
        use_quadratic_decoder=True,
    )
)

prediction = model.predict(inputs)
relative_error = np.linalg.norm(prediction - outputs) / np.linalg.norm(outputs)

print(relative_error)
assert relative_error < 1.0e-8
```

---

## Topic: Context Features

### Idea

Sometimes the same input vector can lead to different outputs depending on
additional sample-level information.  For example:

- time,
- Reynolds number,
- material parameter,
- coupling iteration,
- regime identifier,
- mesh or loading parameter.

These are context features.  They are appended to the reduced input coordinates
before regression.

### First-principles derivation

Without context, the reduced map is

$$
z_y \approx \mathcal{G}(z_x).
$$

With context,

$$
z_y \approx \mathcal{G}(z_x,c_1,\ldots,c_m).
$$

For each sample, the regressor input becomes

$$
s_j = [z_{x,j}, c_{1,j},\ldots,c_{m,j}].
$$

Because context features may have different scales, the training pipeline stores
context statistics such as mean and scale.  During online prediction, the same
normalization is applied to new context values.

### API mapping

Use:

```python
OfflineConfig(context_feature_names=("time",))
```

During online prediction, pass:

```python
model.predict(new_inputs, context={"time": new_times})
```

Related model fields:

- `context_feature_names`
- `context_feature_stats`

### Expected result

A model trained with a context feature expects that same feature during online
prediction.  The context array should have one value per sample.

### Example: train and predict with time as context

```python
from pathlib import Path

import numpy as np

from pycutfem.mor.nirb import OfflineConfig, RegressionConfig, run_offline_pipeline
from pycutfem.mor.snapshots import NamedSnapshotBatch

times = np.linspace(0.0, 1.0, 40)
inputs = np.vstack([np.ones_like(times), times])
outputs = np.vstack([np.sin(times), np.cos(times)])

path = Path("tmp_nirb_time_dataset.npz")
NamedSnapshotBatch(
    fields={"input": inputs, "output": outputs},
    times=times,
).save(path)

model = run_offline_pipeline(
    OfflineConfig(
        dataset_path=str(path),
        model_path="tmp_nirb_time_model.pkl",
        input_modes=2,
        output_modes=2,
        regression=RegressionConfig(kind="poly_ls", degree=3),
        context_feature_names=("time",),
    )
)

new_times = np.linspace(0.1, 0.9, 5)
new_inputs = np.vstack([np.ones_like(new_times), new_times])
prediction = model.predict(new_inputs, context={"time": new_times})

assert prediction.shape == (2, 5)
print(prediction.shape, model.context_feature_names)
```

> Note: the input already contains time in this toy example so that the data is
> easy to read.  In a real use case, context is useful when important information
> is not already contained in the input vector.

---

## Topic: Online Prediction

### Idea

After offline training, online prediction should be simple.  The trained model
receives a new full input vector or matrix and returns the predicted full
output.

### First-principles derivation

For a new input matrix

$$
X_{\text{new}} = [x_1^{\text{new}},\ldots,x_M^{\text{new}}],
$$

prediction applies four steps:

1. Center and project input:

   $$
   Z_x = V_x^T(X_{\text{new}}-\bar{x}\mathbf{1}^T).
   $$

2. Append context features if used:

   $$
   S = [Z_x^T, C]^T
   $$

   conceptually, with sample-major layout inside the regressor.

3. Predict output coordinates:

   $$
   \widehat{Z}_y = \mathcal{G}(Z_x,C).
   $$

4. Decode:

   $$
   \widehat{Y}=D_y(\widehat{Z}_y).
   $$

A single input vector with shape `(n_input,)` is treated as one sample.  A matrix
with shape `(n_input, n_samples)` is treated as multiple samples.

### API mapping

Use either the trained model directly:

```python
prediction = model.predict(new_input_snapshots)
```

or the online pipeline:

```python
run_online_pipeline(OnlineConfig(...))
```

Important APIs:

- `OnlineConfig`
- `load_input_matrix`
- `run_online_pipeline`
- `TrainedNIRBModel.predict`
- `TrainedNIRBModel.predict_reduced`
- `TrainedNIRBModel.predict_reduced_from_input_coefficients`
- `TrainedNIRBModel.decode_output`

### Expected result

The output shape should be `(n_output, n_samples)` for matrix input.  For a
single vector input, the model may return one output vector or a one-column
matrix depending on the model method; check the package behavior in your code
path and standardize it in application adapters.

### Example: predict from a trained model

```python
from pathlib import Path

import numpy as np

from pycutfem.mor.nirb import OfflineConfig, OnlineConfig, RegressionConfig, run_offline_pipeline, run_online_pipeline
from pycutfem.mor.snapshots import NamedSnapshotBatch

rng = np.random.default_rng(7)
inputs = rng.normal(size=(3, 50))
outputs = np.vstack([inputs[0] + inputs[1], inputs[2] ** 2])

NamedSnapshotBatch(fields={"input": inputs, "output": outputs}).save("tmp_online_train.npz")

model = run_offline_pipeline(
    OfflineConfig(
        dataset_path="tmp_online_train.npz",
        model_path="tmp_online_model.pkl",
        input_modes=3,
        output_modes=2,
        regression=RegressionConfig(kind="poly_ls", degree=2),
    )
)

new_inputs = inputs[:, :4]
prediction_direct = model.predict(new_inputs)
assert prediction_direct.shape == (2, 4)

# Optional file-based online path.
np.savez("tmp_online_inputs.npz", input=new_inputs)
run_online_pipeline(
    OnlineConfig(
        model_path="tmp_online_model.pkl",
        input_path="tmp_online_inputs.npz",
        predictions_path="tmp_online_predictions.npz",
    )
)

print(prediction_direct.shape)
```

---

## Topic: Restricted Outputs

### Idea

Sometimes the full output vector is large, but the online application only needs
part of it:

- interface rows,
- boundary values,
- sensor values,
- a subvector used by a coupled solver,
- rows used by a quantity of interest.

A restricted output model predicts only the requested rows or applies a stored
output restriction.

### First-principles derivation

Let the full output be

$$
y \in \mathbb{R}^{n_y}.
$$

Let `R` be a restriction matrix selecting or combining output rows:

$$
y_R = Ry.
$$

If `R` selects rows, each row of `R` is a coordinate vector.  For example,
selecting rows `[0, 3]` gives

$$
R =
\begin{bmatrix}
1&0&0&0\\
0&0&0&1
\end{bmatrix}.
$$

The restricted prediction is

$$
\hat{y}_R = R D_y(\hat{z}_y).
$$

This can avoid storing or returning full fields when only a subvector is needed.

### API mapping

Offline options:

- `output_indices`
- `output_matrix_path`

Model methods:

- `predict_restricted`
- `output_restriction`

Online option:

- `OnlineConfig(restricted_output=True)`

### Expected result

If the model is trained with output indices, `predict_restricted` returns only
those selected rows.  The number of rows equals the number of selected output
indices or the number of rows in the output restriction matrix.

### Example: train with selected output rows

```python
import numpy as np

from pycutfem.mor.nirb import OfflineConfig, RegressionConfig, run_offline_pipeline
from pycutfem.mor.snapshots import NamedSnapshotBatch

rng = np.random.default_rng(8)
inputs = rng.normal(size=(3, 60))
outputs = np.vstack(
    [
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[0] + inputs[1],
        inputs[1] ** 2,
    ]
)

NamedSnapshotBatch(fields={"input": inputs, "output": outputs}).save("tmp_restricted_dataset.npz")

model = run_offline_pipeline(
    OfflineConfig(
        dataset_path="tmp_restricted_dataset.npz",
        model_path="tmp_restricted_model.pkl",
        input_modes=3,
        output_modes=3,
        output_indices=np.array([0, 3, 4]),
        regression=RegressionConfig(kind="poly_ls", degree=2),
    )
)

restricted = model.predict_restricted(inputs[:, :5])
assert restricted.shape[0] == 3
assert restricted.shape[1] == 5
print(restricted.shape)
```

---

## Topic: Reduced Spaces And Reduced Transfers

### Idea

`ReducedSpace` is a small wrapper around a basis and an optional mass matrix.
It is useful when coupling variables are already represented in reduced
coordinates.

`ReducedTransfer` represents a full transfer operator after projection into
source and target reduced spaces.

### First-principles derivation

Let a full vector be represented by a basis:

$$
u \approx V a.
$$

With Euclidean orthonormal columns, projection is

$$
a = V^T u.
$$

With a mass matrix `M`, the correct projection solves

$$
(V^T M V)a = V^T M u.
$$

The matrix

$$
G = V^T M V
$$

is the Gram matrix.  If the basis is `M`-orthonormal, then `G=I`.

Now suppose a full transfer operator maps source vectors to target vectors:

$$
y = T x.
$$

The source reconstruction is

$$
x \approx V_s a_s.
$$

The target coordinate is obtained by target projection:

$$
a_t
=
(V_t^T M_t V_t)^{-1}V_t^T M_t T V_s a_s.
$$

Therefore the reduced transfer matrix is

$$
T_r
=
G_t^{-1}V_t^T M_t T V_s.
$$

### API mapping

Use:

- `ReducedSpace`
- `ReducedTransfer`
- `ReducedTransfer.from_full_transfer`
- `ReducedSpace.project`
- `ReducedSpace.reconstruct`
- `ReducedSpace.inner`
- `ReducedSpace.norm`
- `ReducedSpace.relative_change`

### Expected result

Applying a reduced transfer to source coefficients should match: reconstruct in
full space, apply full transfer, project to target space.

### Example: reduced transfer from a full matrix

```python
import numpy as np

from pycutfem.mor.nirb import ReducedSpace, ReducedTransfer

source = ReducedSpace(
    basis=np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    ),
    name="source",
)

target = ReducedSpace(
    basis=np.eye(2),
    name="target",
)

full_transfer = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
    ]
)

transfer = ReducedTransfer.from_full_transfer(
    source=source,
    target=target,
    full_transfer=full_transfer,
)

source_coefficients = np.array([0.3, -0.2])
reduced_result = transfer.apply(source_coefficients)

full_source = source.reconstruct(source_coefficients)
manual_target = target.project(full_transfer @ full_source)

assert np.allclose(reduced_result, manual_target)
print(reduced_result)
```

---

## Topic: Reduced Output Decoders

### Idea

A `ReducedOutputDecoder` maps predicted output coefficients back to a physical
or full output space.  It is the last stage of the NIRB prediction pipeline.

### First-principles derivation

A linear decoder has the form

$$
D_y(z)=b+Lz,
$$

where

- `b` is the output mean or bias,
- `L` is the linear reconstruction map.

A quadratic decoder adds nonlinear reduced features:

$$
D_y(z)=b+Lz+Q\phi_2(z).
$$

If the output is already represented in another `ReducedSpace`, then decoding
can mean reconstructing that reduced-space vector or applying an additional map
into another output representation.

### API mapping

Use:

- `ReducedOutputDecoder`
- `ReducedOutputDecoder.from_full_decoder`
- `decode_coefficients`
- `decode_output`

The trained model uses the decoder internally through:

- `TrainedNIRBModel.decode_output`
- `TrainedNIRBModel.predict`

### Expected result

`decode_coefficients` maps one reduced output vector into coefficients of the
`output_space`.  `decode_output` then reconstructs that coefficient vector in
the physical output space.  If the decoder has a zero quadratic map, the
coefficient result equals `bias + linear_map @ coefficients`.

### Example: linear reduced output decoder

```python
import numpy as np

from pycutfem.mor.nirb import ReducedOutputDecoder, ReducedSpace

linear_map = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
)
bias = np.array([0.1, 0.2, 0.3])

# For two reduced input coordinates, the quadratic feature vector has
# [z0*z0, z0*z1, z1*z1], hence three columns.
quadratic_map = np.zeros((3, 3))
output_space = ReducedSpace(basis=np.eye(3), name="decoded-output")

decoder = ReducedOutputDecoder(
    linear_map=linear_map,
    quadratic_map=quadratic_map,
    bias=bias,
    output_space=output_space,
)
coefficients = np.array([2.0, 0.5])

output_coefficients = decoder.decode_coefficients(coefficients)
output = decoder.decode_output(coefficients)
manual = bias + linear_map @ coefficients

assert np.allclose(output_coefficients, manual)
assert np.allclose(output, manual)
print(output)
```

---

## Topic: Reduced IQN-ILS Coupling Acceleration

### Idea

`ReducedIQNILS` is for fixed-point coupling problems.  The acronym means
**Interface Quasi-Newton Inverse Least-Squares**.  The reduced version applies
the same idea to reduced coupling coordinates.

This is useful when a coupled solver repeatedly applies a map

$$
x^k \mapsto g^k = G(x^k)
$$

and wants to find a fixed point

$$
x = G(x).
$$

In FSI, `x` might be an interface displacement, an interface load, or another
coupling vector.  In this generic NIRB package, it is just a vector in a
reduced coupling space.  NIRB does not decide what `x` means; the application
adapter decides that.

### First-principles derivation

Define the fixed-point residual

$$
r^k = g^k - x^k.
$$

At convergence,

$$
r^k = 0.
$$

A relaxed fixed-point iteration would be

$$
x^{k+1}=x^k + \omega r^k.
$$

This is simple but can converge slowly.  IQN-ILS accelerates the update by
building a small inverse secant model from previous coupling iterations.

After iteration `k`, the implementation stores pairs

$$
(x^i,g^i).
$$

From those pairs it forms newest-first residual-difference columns

$$
V_k =
\begin{bmatrix}
r^k-r^{k-1} & r^{k-1}-r^{k-2} & \cdots
\end{bmatrix}
$$

and returned-value-difference columns

$$
W_k =
\begin{bmatrix}
g^k-g^{k-1} & g^{k-1}-g^{k-2} & \cdots
\end{bmatrix}.
$$

The inverse least-squares step finds coefficients `gamma` such that a linear
combination of residual differences approximately cancels the current residual:

$$
\gamma
=
\arg\min_a \|V_k a + r^k\|_2^2.
$$

Then the accelerated next iterate uses the current returned value and the same
linear combination of returned-value differences:

$$
x^{k+1}
=
g^k + W_k\gamma.
$$

This is why the method is called inverse least-squares: it does not build or
factor a Jacobian of the coupled solver.  It only solves a small least-squares
problem in the history space.

If there is not enough history yet, the method falls back to relaxed Picard:

$$
x^{k+1}=x^k+\omega(g^k-x^k).
$$

Regularization adds a Tikhonov term to the least-squares problem:

$$
\gamma
=
\arg\min_a \|V_k a+r^k\|_2^2+\lambda\|a\|_2^2.
$$

The implementation can also reuse secant matrices from previous converged
steps.  `finalize_step()` converts the current step's iteration history into
stored matrices and clears the per-step history.  Call it once after a coupling
step is accepted or abandoned, not after every coupling iteration.

### API mapping

Use:

- `ReducedIQNILS`
- `iqnils_iteration_matrices`
- `iqnils_next_iterate`
- `ReducedIQNILS.next`
- `ReducedIQNILS.finalize_step`

Main configuration fields:

- `omega`: fallback relaxation factor,
- `horizon`: number of previous updates to keep,
- `regularization`: least-squares regularization.

Main call pattern:

```python
# Inside the coupling iteration.
x_next = accelerator.next(current=x_k, returned=g_k)

# After the coupling step has finished.
accelerator.finalize_step()
```

The names mean:

- `current`: the coupling vector sent to the solver, `x^k`,
- `returned`: the coupling vector returned by the solver, `g^k`,
- `next(...)`: appends the pair `(x^k,g^k)` and returns `x^{k+1}`,
- `next(..., converged=True)`: returns `returned` without appending a new
  history pair,
- `finalize_step()`: stores the current step's secant matrices for future
  steps and clears `x_history`/`g_history`.

### Expected result

For a contractive fixed-point map, IQN-ILS should usually reduce the number of
iterations compared with plain relaxation once enough history is available.
The first few iterations may behave like relaxed fixed-point iteration because
there is not yet enough history.  If the history matrices are singular,
ill-conditioned, or empty, the implementation falls back to relaxed Picard.

### Pseudocode: reduced fixed-point acceleration

This is pseudocode because the exact coupled solver is application-specific.
The solver output `solver_apply(x)` could be a fluid-solid coupling step, a
partitioned multiphysics update, or any fixed-point map.

```python
# PSEUDOCODE
from pycutfem.mor.nirb import ReducedIQNILS

accelerator = ReducedIQNILS(omega=0.5, horizon=8, regularization=1.0e-10)

x = initial_reduced_coupling_state.copy()

for k in range(max_coupling_iterations):
    returned = solver_apply(x)       # application-specific fixed-point map
    residual = returned - x

    if norm(residual) < tolerance:
        x = accelerator.next(x, returned, converged=True)
        break

    x = accelerator.next(x, returned)
else:
    raise RuntimeError("coupling did not converge")

accelerator.finalize_step()
```

### Example: standalone fixed-point map

```python
import numpy as np

from pycutfem.mor.nirb import ReducedIQNILS

# A simple linear contractive fixed-point map: returned = A x + b.
A = np.array([[0.6, 0.1], [0.0, 0.4]])
b = np.array([1.0, -0.5])

accelerator = ReducedIQNILS(omega=0.5, horizon=5, regularization=1.0e-12)
x = np.zeros(2)

for k in range(20):
    returned = A @ x + b
    residual_norm = np.linalg.norm(returned - x)

    if residual_norm < 1.0e-10:
        x = accelerator.next(x, returned, converged=True)
        break

    x = accelerator.next(x, returned)
else:
    raise RuntimeError("fixed-point iteration did not converge")

accelerator.finalize_step()

fixed_point = np.linalg.solve(np.eye(2) - A, b)
assert np.linalg.norm(x - fixed_point) < 1.0e-6
print(k + 1, x, fixed_point)
```

---

## Topic: Validation And Acceptance

### Idea

A NIRB model should be accepted only after validation on data not used for
training.  Training error alone can be misleading, especially for high-degree
polynomials or flexible RBF models.

### First-principles derivation

Given reference outputs

$$
Y_{\text{ref}}
$$

and predictions

$$
Y_{\text{pred}},
$$

common errors are

$$
E_F =
\frac{\|Y_{\text{pred}}-Y_{\text{ref}}\|_F}
{\|Y_{\text{ref}}\|_F},
$$

and per-sample relative errors

$$
e_j=
\frac{\|\hat{y}_j-y_j\|_2}{\|y_j\|_2}.
$$

Use thresholds that match the role of the model.  A model used only to produce
an initial guess may tolerate larger error than a model used to replace a solver
stage.

Speedup should also be measured if the NIRB model replaces a costly stage:

$$
\text{speedup} = \frac{T_{\text{FOM}}}{T_{\text{NIRB}}}.
$$

### API mapping

Use:

- `ValidationConfig`
- `validate_rom`

Important fields include:

- `reference_path`
- `prediction_path`
- `metrics_path`
- `thresholds`
- `fom_iterations`
- `rom_iterations`
- `fom_model_time`
- `rom_model_time`
- `fom_total_time`
- `rom_total_time`
- `metadata`

### Expected result

Validation should produce a metrics file and a pass/fail decision based on the
configured thresholds.  If the validation error is much larger than the training
error, the model is overfitting or extrapolating.

### Pseudocode: validate predictions against reference data

The exact file format for references and predictions depends on the application
adapter, so this is pseudocode.

```python
# PSEUDOCODE
from pycutfem.mor.nirb import ValidationConfig, validate_rom

config = ValidationConfig(
    reference_path="validation_reference_outputs.npz",
    prediction_path="validation_nirb_predictions.npz",
    metrics_path="validation_metrics.json",
    thresholds={
        "relative_l2_error": 2.0e-2,
        "max_sample_relative_error": 5.0e-2,
    },
    fom_total_time=100.0,
    rom_total_time=4.0,
    metadata={"case": "held-out validation"},
)

report = validate_rom(config)

if not report["passed"]:
    raise RuntimeError(report["failed_metrics"])
```

### Example: manual validation calculation

```python
import numpy as np

rng = np.random.default_rng(9)
reference = rng.normal(size=(5, 30))
prediction = reference + 1.0e-3 * rng.normal(size=reference.shape)

relative_frobenius_error = np.linalg.norm(prediction - reference) / np.linalg.norm(reference)
per_sample_error = np.linalg.norm(prediction - reference, axis=0) / np.linalg.norm(reference, axis=0)

assert relative_frobenius_error < 1.0e-2
print(relative_frobenius_error, per_sample_error.max())
```

---

## Topic: Example-Level FSI Adapters

### Idea

The core NIRB package is problem-generic.  It does not contain FSI-specific
words such as force, solid, fluid, interface, or co-simulation.  Those meanings
are introduced by example adapters.

An FSI adapter converts application files into a generic NIRB dataset:

```text
interface load snapshots        -> input field
solid displacement snapshots    -> output field
```

The NIRB algorithm still sees only paired matrices.

### First-principles derivation

Suppose an FSI coupling run produces pairs

$$
f_{\Gamma,j} \mapsto d_{\Gamma,j},
$$

where `f_Gamma` is an interface load and `d_Gamma` is a solid displacement or
interface response.  The adapter simply defines

$$
x_j = f_{\Gamma,j},
\qquad
y_j = d_{\Gamma,j}.
$$

Then the generic NIRB training problem becomes

$$
d_{\Gamma} \approx \widehat{F}(f_{\Gamma},c).
$$

No part of the core NIRB method changes.

### API mapping

Example-level tools may provide functions such as:

```python
from examples.utils.nirb import load_cosim_snapshot_batch
```

Core NIRB training still uses:

- `NamedSnapshotBatch`
- `OfflineConfig`
- `run_offline_pipeline`

### Expected result

The adapter should save a `NamedSnapshotBatch`.  The fields may have FSI names,
but once saved, the offline pipeline treats them like any other input-output
pair.

### Pseudocode: convert FSI data to a generic NIRB dataset

```python
# PSEUDOCODE: requires example-layer FSI files.
from examples.utils.nirb import load_cosim_snapshot_batch

batch = load_cosim_snapshot_batch(
    "examples/NIRB/artifacts/some_run",
    force_key="load_guess_data",
    displacement_key="disp_data",
)

batch.save("tmp_fsi_as_generic_nirb_dataset.npz")
```

### Pseudocode: train on FSI-named fields

```python
# PSEUDOCODE: requires the dataset produced by the FSI adapter.
from pycutfem.mor.nirb import OfflineConfig, RegressionConfig, run_offline_pipeline

model = run_offline_pipeline(
    OfflineConfig(
        dataset_path="tmp_fsi_as_generic_nirb_dataset.npz",
        model_path="tmp_fsi_nirb_model.pkl",
        dataset_input_field="interface_load",
        dataset_output_field="solid_displacement",
        input_modes=20,
        output_modes=8,
        regression=RegressionConfig(kind="poly_ridge", degree=2, regularization=1.0e-8),
    )
)
```

---

## Recommended Workflow

Use this sequence for a robust NIRB study:

1. Define what the input vector means and what the output vector means.
2. Collect paired snapshots with matching columns.
3. Store the data in a `NamedSnapshotBatch`.
4. Split the data into training and validation sets.
5. Start with centered linear POD spaces for input and output.
6. Train a simple linear or low-degree polynomial reduced map.
7. Check training error and held-out validation error.
8. Increase input or output modes if projection error is too large.
9. Increase regression complexity only if reduced-coordinate regression error is
   too large.
10. Use `use_quadratic_decoder=True` only if output reconstruction error remains
    high with a linear decoder.
11. Add context features only when the input vector alone is not enough to
    explain the output.
12. Use restricted outputs when the online consumer needs only selected rows.
13. Validate on trajectories, parameters, or regimes not used for fitting.
14. Report error and timing together.
15. Treat far-away inputs as extrapolation unless a separate guard certifies the
    regime.

---

## Common Mistakes

- Confusing the input-output map with the output decoder.  The regressor maps
  `z_x` to `z_y`; the decoder maps `z_y` to `y`.
- Using a quadratic decoder when the real problem is a nonlinear input-output
  relation.  In that case, use a polynomial regressor.
- Using a quadratic reduced map when the output field itself is poorly
  reconstructed by a linear decoder.  In that case, improve the output decoder
  or add output modes.
- Forgetting that snapshot matrices are feature-major.
- Shuffling input and output snapshots independently.
- Reporting POD reconstruction error as if it were validation error for the
  learned map.
- Training and validating on the same samples.
- Ignoring context features that are required to distinguish outputs.
- Using high-degree polynomial regression with too few samples.
- Trusting predictions for inputs far outside the training distribution.
- Putting FSI-specific assumptions into the generic NIRB package instead of an
  example adapter.

---

## Complete API Inventory

### Package exports

These names are available from:

```text
from pycutfem.mor.nirb import <name>
```

- `DatasetSplit`
- `NIRBDataset`
- `OfflineConfig`
- `OnlineConfig`
- `ReducedIQNILS`
- `ReducedOutputDecoder`
- `ReducedSpace`
- `ReducedTransfer`
- `RegressionConfig`
- `TrainedNIRBModel`
- `ValidationConfig`
- `dataset_from_named_snapshot_batch`
- `iqnils_iteration_matrices`
- `iqnils_next_iterate`
- `load_dataset`
- `load_input_matrix`
- `run_offline_pipeline`
- `run_online_pipeline`
- `validate_rom`

### Dataset API

`NIRBDataset` fields:

- `input_snapshots`
- `output_snapshots`
- `parameters`
- `times`
- `converged`
- `context_features`
- `output_indices`
- `metadata`

`NIRBDataset` property:

- `n_snapshots`

`NIRBDataset` methods:

- `context`
- `subset`
- `split`

Dataset functions:

- `dataset_from_named_snapshot_batch`
- `load_dataset`

### Offline API

`RegressionConfig` fields:

- `kind`
- `smoothing`
- `degree`
- `criterion`
- `standardize_inputs`
- `regularization`
- `n_neighbors`
- `power`

`RegressionConfig` class method:

- `from_mapping`

Supported `RegressionConfig.kind` values:

- `"tps_rbf"`
- `"poly_lasso"`
- `"poly_ls"`
- `"poly_ridge"`
- `"knn"`
- `"knearest"`
- `"nearest"`

`OfflineConfig` fields:

- `dataset_path`
- `model_path`
- `input_modes`
- `output_modes`
- `regression`
- `center_input`
- `center_output`
- `use_quadratic_decoder`
- `dataset_input_field`
- `dataset_output_field`
- `zero_anchor_weight`
- `output_indices`
- `output_matrix_path`
- `context_feature_names`
- `metadata`

`OfflineConfig` class method:

- `from_mapping`

`TrainedNIRBModel` fields:

- `input_basis`
- `output_decoder`
- `regressor`
- `output_restriction`
- `context_feature_names`
- `context_feature_stats`
- `metadata`

`TrainedNIRBModel` methods:

- `encode_input`
- `predict_reduced`
- `predict_reduced_from_input_coefficients`
- `decode_output`
- `predict`
- `predict_restricted`

Training function:

- `run_offline_pipeline`

### Online API

`OnlineConfig` fields:

- `model_path`
- `input_path`
- `predictions_path`
- `restricted_output`

`OnlineConfig` class method:

- `from_mapping`

Online functions:

- `load_input_matrix`
- `run_online_pipeline`

### Reduced-space API

`ReducedSpace` fields:

- `basis`
- `mass`
- `name`

`ReducedSpace` properties:

- `n_dofs`
- `n_modes`
- `mass_matrix`
- `gram`

`ReducedSpace` methods:

- `reconstruct`
- `project`
- `inner`
- `norm`
- `relative_change`

`ReducedTransfer` fields:

- `matrix`
- `source`
- `target`

`ReducedTransfer` class method:

- `from_full_transfer`

`ReducedTransfer` method:

- `apply`

`ReducedOutputDecoder` fields:

- `linear_map`
- `quadratic_map`
- `bias`
- `output_space`
- `feature_map`

`ReducedOutputDecoder` class method:

- `from_full_decoder`

`ReducedOutputDecoder` methods:

- `decode_coefficients`
- `decode_output`

`ReducedIQNILS` fields:

- `omega`
- `horizon`
- `regularization`
- `x_history`
- `g_history`
- `old_dr_mats`
- `old_dg_mats`

`ReducedIQNILS` methods:

- `next(current, returned, *, converged=False)`
- `finalize_step()`

Reduced coupling functions:

- `iqnils_iteration_matrices(*, x_history, g_history, iteration_horizon)`
- `iqnils_next_iterate(*, x_curr, g_curr, x_history, g_history, dr_old_mats=None, dg_old_mats=None, omega, horizon, regularization=0.0)`

### Validation API

`ValidationConfig` fields:

- `reference_path`
- `prediction_path`
- `metrics_path`
- `thresholds`
- `fom_iterations`
- `rom_iterations`
- `fom_model_time`
- `rom_model_time`
- `fom_total_time`
- `rom_total_time`
- `metadata`

`ValidationConfig` class method:

- `from_mapping`

Validation function:

- `validate_rom`

### Related MOR APIs used by NIRB

Snapshot storage:

- `NamedSnapshotBatch`
- `NamedSnapshotWriter`
- `NamedSnapshotReader`

POD:

- `PODBasis`
- `fit_pod`
- `project_to_basis`

Regressors:

- `PolynomialFeatureMap`
- `PolynomialLeastSquaresRegressor`
- `PolynomialLassoRegressor`
- `ThinPlateSplineRBF`
- `KNearestRegressor`

Quadratic output manifolds:

- `QuadraticFeatureMap`
- `QuadraticManifoldDecoder`
- `fit_quadratic_decoder`
- `fit_quadratic_manifold`
- `quadratic_feature_matrix`

Model I/O:

- `load_model`
- `save_model`
