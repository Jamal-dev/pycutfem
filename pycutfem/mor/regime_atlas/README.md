# Nonlinear Regime Atlas

`pycutfem.mor.regime_atlas` is the problem-generic layer for discovering
nonlinear regimes and selecting local reduced-model banks.  It owns reusable
MOR logic only.  It does **not** know whether a feature is time, a parameter,
a residual norm, a coupling metric, an interface load, a sensor value, or a
reduced coordinate.  Those meanings belong in the PDE examples and problem
adapters.

This README follows the same help-file style used in the MOR and NIRB guides:

```text
Idea -> first-principles derivation -> API mapping -> expected result -> example
```

The central question is:

> Given many nonlinear training samples, how can we discover regions where one
> local reduced model is trustworthy, and how can the online code select the
> right region without blindly extrapolating?

---

## Contents

1. [Why A Nonlinear Regime Atlas Is Needed](#why-a-nonlinear-regime-atlas-is-needed)
2. [Topic: Regime Datasets And Features](#topic-regime-datasets-and-features)
3. [Topic: Robust Scaling And Feature Geometry](#topic-robust-scaling-and-feature-geometry)
4. [Topic: K-Medoids Regime Discovery](#topic-k-medoids-regime-discovery)
5. [Topic: Epsilon-Cover Atlases](#topic-epsilon-cover-atlases)
6. [Topic: Hierarchical Atlases](#topic-hierarchical-atlases)
7. [Topic: Density-Based Atlases](#topic-density-based-atlases)
8. [Topic: Mixture-Model Atlases](#topic-mixture-model-atlases)
9. [Topic: Validation And Atlas Selection](#topic-validation-and-atlas-selection)
10. [Topic: Residual-Error Greedy Splitting](#topic-residual-error-greedy-splitting)
11. [Topic: Subspace-Based Region Discovery](#topic-subspace-based-region-discovery)
12. [Topic: Tree Routing](#topic-tree-routing)
13. [Topic: Local Bank Manifests And Online Novelty](#topic-local-bank-manifests-and-online-novelty)
14. [Topic: IQN-ILS And Coupled Fixed-Point Acceleration](#topic-iqn-ils-and-coupled-fixed-point-acceleration)
15. [Topic: Choosing The Number Of Regions](#topic-choosing-the-number-of-regions)
16. [Topic: End-To-End Offline/Online Pattern](#topic-end-to-end-offlineonline-pattern)
17. [Common Mistakes](#common-mistakes)
18. [Migration From Old Modules](#migration-from-old-modules)
19. [Complete API Inventory](#complete-api-inventory)

---

## Why A Nonlinear Regime Atlas Is Needed

### Idea

A global reduced basis assumes that the important full-order states live near
one low-dimensional affine space:

$$
x \approx \bar{x}+Vq.
$$

This can fail for strongly nonlinear problems.  A nonlinear trajectory may pass
through qualitatively different phases: small deformation, strong interaction,
detachment, contact, post-transition relaxation, or different parameter-driven
branches.  One basis then has to represent too many different directions and may
need many modes.

A nonlinear regime atlas uses several local reduced models:

$$
\mathcal{M}
\approx
\bigcup_{r=1}^{K}\mathcal{M}_r,
\qquad
x \in \mathcal{M}_r
\Rightarrow
x \approx \bar{x}_r+V_r q_r.
$$

Each local region may own a local trial basis, collateral basis, sampling rule,
non-intrusive map, decoder, hyper-reduction rule, validation report, and
fallback policy.

### First-principles derivation

Start from a training database.  For sample `i`, assume we have:

- a full-order state or snapshot `x_i`,
- optional residual, error, or quantity data,
- an online-available feature vector `z_i`.

The feature vector is small:

$$
z_i \in \mathbb{R}^{d},
\qquad d \ll \dim(x_i).
$$

A regime atlas partitions feature space into regions:

$$
\Omega_1,\Omega_2,\ldots,\Omega_K.
$$

The region assignment is

$$
\rho(i)=r
\quad \Longleftrightarrow \quad
z_i\in\Omega_r.
$$

For each region, collect the associated snapshots:

$$
X_r = [x_i : \rho(i)=r].
$$

Then build a local reduced model, for example by POD:

$$
X_r-\bar{x}_r\mathbf{1}^{T}
= U_r\Sigma_r W_r^{T},
\qquad
V_r = U_r[:,1:m_r].
$$

The atlas is useful only if the local model generalizes.  Therefore region
compactness is not enough.  Each region should also pass a validation gate:

$$
E_r^{\mathrm{val}}
=
\frac{\|X_r^{\mathrm{pred}}-X_r^{\mathrm{ref}}\|}
{\|X_r^{\mathrm{ref}}\|}
\le \tau.
$$

The package therefore separates two ideas:

1. **Partitioning** proposes candidate regions from data.
2. **Validation and selection** decides whether those regions are safe to use.

### API mapping

Use:

- `RegimeDataset` to store row-major features and metadata,
- a partitioner such as `KMedoidsPartitioner`, `EpsilonCoverPartitioner`,
  `HierarchicalPartitioner`, `DensityPartitioner`, `MixturePartitioner`,
  `SubspacePartitioner`, or `ResidualGreedySplitter`,
- `RegimeAtlas` to store discovered regions,
- `RegimeAtlasSelector` to choose among candidate atlases,
- `RegimeOnlineSelector` to select or reject a local bank online.

### Expected result

A good atlas should satisfy all of the following:

- each accepted region has enough support samples,
- each accepted region is compact in the chosen metric,
- local validation error passes,
- online routing is stable near region boundaries,
- unsupported online features trigger fallback instead of silent extrapolation.

### Example

This example creates two separated feature clouds and asks k-medoids to find two
regions.

```python
import numpy as np

from pycutfem.mor.regime_atlas import KMedoidsPartitioner

features = np.vstack(
    [
        np.linspace(-1.0, -0.8, 12),
        np.linspace(0.8, 1.0, 12),
    ]
).reshape(-1, 1)

atlas = KMedoidsPartitioner(n_regions=2).fit(features)

assert atlas.n_regions == 2
assert sorted(atlas.support_counts.tolist()) == [12, 12]
print(atlas.support_counts)
```

The expected result is two regions with twelve samples each.  In a real ROM,
you would then train one local basis or local HROM artifact per region.

---

## Topic: Regime Datasets And Features

### Idea

A regime atlas does not cluster full PDE states directly by default.  Full
states are large, and many of their components are not available during online
selection.  Instead, the atlas clusters **features** that can be computed both
offline and online.

Examples of possible features are:

- parameter values,
- time or load-step index,
- nonlinear residual norm,
- interface load norm,
- interface displacement change,
- contraction factor from a fixed-point iteration,
- reduced coordinates,
- cheap error indicators,
- sensor values,
- local physical indicators.

The core package treats all columns generically.  A feature column only has
physical meaning in the example or problem adapter.

### First-principles derivation

The feature matrix is row-major:

$$
Z=
\begin{bmatrix}
z_1^T\\
z_2^T\\
\vdots\\
z_N^T
\end{bmatrix}
\in\mathbb{R}^{N\times d}.
$$

Each row is one training sample, time step, parameter point, coupling iteration,
or candidate online stage.  Each column is one feature.

A local region should be based on information that will also be known online.
For example, if the online selector will not know the full FOM error, then the
FOM error should not be used as a routing feature.  It may be used for
validation, but not for online routing.

If samples come from trajectories, random sample splits can leak information.
For example, neighboring time steps from the same trajectory may be almost
identical.  A random validation split may therefore look better than true
predictive validation.  Grouped validation avoids this:

$$
G_{\mathrm{train}}\cap G_{\mathrm{val}}=\emptyset.
$$

A group may be a trajectory id, parameter case id, mesh id, initial condition
id, or experiment id.

### API mapping

Use:

- `RegimeDataset` for features and metadata,
- `as_feature_matrix` to coerce arrays or dataset-like inputs,
- `as_index_vector` to validate index arrays,
- `make_validation_split` for random or group-aware splitting,
- `RegimeValidationSplit` to store train/validation indices.

### Expected result

The dataset should preserve feature names, optional sample ids, group ids,
steps, weights, and metadata.  A grouped validation split should keep each group
entirely on one side of the split.

### Example

```python
import numpy as np

from pycutfem.mor.regime_atlas import RegimeDataset, make_validation_split

features = np.array(
    [
        [1.0, 100.0],
        [2.0, 110.0],
        [3.0, 120.0],
        [4.0, 130.0],
    ]
)

dataset = RegimeDataset(
    features=features,
    feature_names=("parameter", "indicator"),
    groups=np.array(["case_a", "case_a", "case_b", "case_b"]),
    steps=np.array([0, 1, 0, 1]),
)

split = make_validation_split(dataset, test_fraction=0.5, random_state=1)

assert split.train_indices.ndim == 1
assert split.validation_indices.ndim == 1
print(split.metadata)
```

The expected result is a validation split object.  Because `dataset.groups` is
present, all samples from the same group are kept together.

---

## Topic: Robust Scaling And Feature Geometry

### Idea

Most atlas methods use distances in feature space.  Distances are meaningless
unless the feature columns have comparable scales.

For example, suppose

$$
z_i=[\|r_i\|,\ \|f_{\Gamma,i}\|].
$$

If residual norms are around `1e-6` and interface loads are around `1e3`, then
plain Euclidean distance is dominated by the load column.  The residual feature
would almost not matter.

Therefore the atlas usually works with scaled features.

### First-principles derivation

A robust scaled feature is

$$
\hat{z}_{ij}
=
\frac{z_{ij}-c_j}{s_j},
$$

where `c_j` is a robust center and `s_j` is a robust scale.  Common choices are

$$
c_j=\operatorname{median}(Z_{:,j}),
\qquad
s_j=\operatorname{IQR}(Z_{:,j}).
$$

The interquartile range is

$$
\operatorname{IQR}(Z_{:,j})
=
Q_{75}(Z_{:,j})-Q_{25}(Z_{:,j}).
$$

Robust scaling is preferred because a few extreme nonlinear failures should not
completely determine the feature scale.

After scaling, the Euclidean distance is

$$
d(i,k)
=
\|\hat{z}_i-\hat{z}_k\|_2.
$$

This distance is now dimensionless and more balanced across features.

### API mapping

Use:

- `robust_feature_center_scale`,
- `scale_feature_matrix`.

Most partitioners internally store or consume scaled features through their
configuration.  When using a saved bank manifest, store the same center and
scale so online features are transformed consistently.

### Expected result

The scaled matrix has the same shape as the original feature matrix.  A feature
with large physical units should no longer dominate distance computations.

### Example

```python
import numpy as np

from pycutfem.mor.regime_atlas import robust_feature_center_scale, scale_feature_matrix

features = np.array(
    [
        [1.0e-6, 1000.0],
        [2.0e-6, 1100.0],
        [3.0e-6, 1200.0],
        [4.0e-6, 1300.0],
    ]
)

center, scale = robust_feature_center_scale(features)
scaled = scale_feature_matrix(features, center=center, scale=scale)

assert scaled.shape == features.shape
print(center)
print(scale)
print(scaled)
```

The expected result is a dimensionless feature matrix.  Later partitioners
should use `scaled`, or should reproduce the same scaling internally.

---

## Topic: K-Medoids Regime Discovery

### Idea

K-medoids discovers `K` regions by choosing `K` representative training samples
called **medoids**.  A medoid is an actual sample from the data, not an average.

This is useful for nonlinear ROMs because the center of a region corresponds to
a real training state that actually occurred.  In contrast, a k-means center may
be an artificial point between regimes.

K-medoids answers:

> Which `K` real training samples best represent the feature cloud, and which
> samples are closest to each representative?

### First-principles derivation

Let the scaled feature vectors be

$$
\hat{z}_1,\ldots,\hat{z}_N.
$$

Choose `K` medoid indices

$$
M=\{m_1,\ldots,m_K\},
\qquad m_r\in\{1,\ldots,N\}.
$$

Each sample is assigned to the nearest medoid:

$$
\rho(i)
=
\arg\min_{r=1,\ldots,K}
\|\hat{z}_i-\hat{z}_{m_r}\|_2.
$$

The k-medoids objective is

$$
\min_{m_1,\ldots,m_K}
\sum_{i=1}^{N} w_i
\min_{r=1,\ldots,K}
\|\hat{z}_i-\hat{z}_{m_r}\|_2^2.
$$

Here `w_i` is an optional sample weight.  A high weight means that sample is
more important to represent well.

The practical algorithm is usually iterative:

1. initialize `K` medoids,
2. assign every sample to the nearest medoid,
3. for each cluster, try replacing the medoid by another sample in that cluster,
4. keep the replacement if it reduces the objective,
5. repeat until no improvement occurs or a maximum iteration count is reached.

The discovered regions are

$$
\Omega_r
=
\{i:\rho(i)=r\}.
$$

Each region can then train a local basis:

$$
V_r = \operatorname{POD}\{x_i:i\in\Omega_r\}.
$$

### How k-medoids finds nonlinear regions

K-medoids does not know the PDE.  It finds regions because the feature vector
was designed so that nearby features mean similar nonlinear behavior.  For
example, if the features are

$$
z_i=
[\log_{10}\|r_i\|,
\log_{10}\|\Delta d_{\Gamma,i}\|,
\|f_{\Gamma,i}\|_{\mathrm{rms}},
\rho_{\mathrm{contract},i}],
$$

then two samples are close when they have similar residual level, similar
interface motion, similar interface load, and similar nonlinear contraction.
The medoids become representative nonlinear regimes.

Therefore k-medoids discovers meaningful regions only when the features encode
meaningful nonlinear state information.

### API mapping

Use:

- `KMedoidsPartitioner`,
- `fit_kmedoids_regime_atlas`,
- legacy function names `fit_k_medoids` and `fit_feature_atlas`, still exported
  from `pycutfem.mor.regime_atlas`,
- `RegimeAtlas.support_counts`,
- `RegimeAtlas.regions`,
- `RegimeAtlas.labels`.

### Expected result

For two separated feature clouds, k-medoids with `n_regions=2` should select
one medoid in each cloud and assign the samples accordingly.  In a ROM pipeline,
one local basis or local HROM artifact is then built per region.

### Example

```python
import numpy as np

from pycutfem.mor.regime_atlas import KMedoidsPartitioner

left = np.column_stack(
    [
        np.linspace(-1.0, -0.8, 12),
        np.linspace(0.0, 0.1, 12),
    ]
)
right = np.column_stack(
    [
        np.linspace(0.8, 1.0, 12),
        np.linspace(1.0, 1.1, 12),
    ]
)
features = np.vstack([left, right])

atlas = KMedoidsPartitioner(n_regions=2).fit(features)

assert atlas.n_regions == 2
assert sorted(atlas.support_counts.tolist()) == [12, 12]
print(atlas.support_counts)
print(atlas.metadata["medoid_indices"])
```

The expected result is two balanced regions.  If your implementation stores
medoid information in metadata, inspect `atlas.metadata` for the selected
representatives.

---

## Topic: Epsilon-Cover Atlases

### Idea

K-medoids requires choosing `K`.  Epsilon-cover starts from a different
question:

> How many regions are needed so every training feature is within distance
> `epsilon` of some representative?

This method is useful when you care about a maximum coverage radius rather than
a fixed number of regions.

### First-principles derivation

Given scaled features

$$
\hat{z}_1,\ldots,\hat{z}_N,
$$

choose centers from the data so that

$$
\max_i \min_r \|\hat{z}_i-c_r\|_2 \le \epsilon.
$$

A common greedy algorithm is farthest-first selection:

1. choose an initial center,
2. compute each sample's distance to its nearest center,
3. add the sample with the largest nearest-center distance,
4. repeat until the maximum distance is below `epsilon` or `max_regions` is
   reached.

The coverage radius of the resulting atlas is

$$
R_{\mathrm{cover}}
=
\max_i\min_r\|\hat{z}_i-c_r\|_2.
$$

If

$$
R_{\mathrm{cover}}\le\epsilon,
$$

then every training sample is covered geometrically.

### How epsilon-cover finds nonlinear regions

Epsilon-cover creates small regions wherever the feature cloud is spread out.
Dense, compact parts of the trajectory may need only one center.  Long or
curved parts need more centers.  It therefore adapts the number of regions to
the geometry of the nonlinear regime path.

### API mapping

Use:

- `EpsilonCoverPartitioner`,
- `RegimeAtlas.coverage`,
- `RegimeAtlas.support_counts`.

### Expected result

If two feature clouds are well separated and each cloud has diameter smaller
than `epsilon`, the method should produce two regions.  If a cloud is long and
curved, it may produce more regions.

### Example

```python
import numpy as np

from pycutfem.mor.regime_atlas import EpsilonCoverPartitioner

features = np.vstack(
    [
        np.linspace(0.0, 0.05, 8),
        np.linspace(2.0, 2.05, 8),
    ]
).reshape(-1, 1)

atlas = EpsilonCoverPartitioner(epsilon=0.1, max_regions=4).fit(features)

assert atlas.n_regions == 2
print(atlas.support_counts)
print(atlas.coverage)
```

The expected result is two regions because each cluster is compact and the two
clusters are far apart.

---

## Topic: Hierarchical Atlases

### Idea

Hierarchical clustering builds a multiscale view of the data.  It starts with
many tiny clusters and repeatedly merges the closest ones.  Cutting the
hierarchy at a chosen level gives the final atlas.

This is useful when the nonlinear regimes may have nested structure, for
example:

```text
all samples
  -> low-load / high-load
      -> early high-load / late high-load
```

### First-principles derivation

Start with each sample as its own cluster:

$$
\mathcal{C}_i=\{i\}.
$$

At each step, merge the two closest clusters.  The definition of cluster
closeness is called the linkage rule.

Single linkage:

$$
d(A,B)=\min_{i\in A,\ j\in B}\|\hat{z}_i-\hat{z}_j\|_2.
$$

Complete linkage:

$$
d(A,B)=\max_{i\in A,\ j\in B}\|\hat{z}_i-\hat{z}_j\|_2.
$$

Average linkage:

$$
d(A,B)=
\frac{1}{|A||B|}
\sum_{i\in A}\sum_{j\in B}\|\hat{z}_i-\hat{z}_j\|_2.
$$

The merge sequence forms a tree, often called a dendrogram.  If the final atlas
needs `K` regions, stop or cut the tree when `K` clusters remain.

### How hierarchical clustering finds nonlinear regions

The method finds regions by repeatedly combining samples or clusters that are
near in feature space.  If the nonlinear trajectory contains nested regimes,
the hierarchy exposes them: coarse cuts give broad regimes, and fine cuts give
specialized local regimes.

### API mapping

Use:

- `HierarchicalPartitioner`,
- `RegimePartitionerConfig`,
- `make_regime_partitioner` with `kind="hierarchical"`.

### Expected result

For two separated clouds, hierarchical clustering with `n_regions=2` should
return the same two main groups.  Different linkage rules may behave
differently for elongated or noisy clouds.

### Example

```python
import numpy as np

from pycutfem.mor.regime_atlas import HierarchicalPartitioner

features = np.vstack(
    [
        np.linspace(0.0, 0.05, 8),
        np.linspace(2.0, 2.05, 8),
    ]
).reshape(-1, 1)

atlas = HierarchicalPartitioner(n_regions=2, linkage="average").fit(features)

assert atlas.n_regions == 2
print(atlas.support_counts)
```

The expected result is two regions with eight samples each.

---

## Topic: Density-Based Atlases

### Idea

Some samples should not be forced into a local reduced model.  They may be rare
failures, transition outliers, unresolved physics, or data points that require
more training.

Density-based partitioning identifies dense groups and labels isolated samples
as outliers.

### First-principles derivation

For each scaled feature point `i`, define its epsilon-neighborhood:

$$
\mathcal{N}_\epsilon(i)
=
\{j:\|\hat{z}_j-\hat{z}_i\|_2\le\epsilon\}.
$$

A point is a core point if

$$
|\mathcal{N}_\epsilon(i)|\ge n_{\min}.
$$

Two core points belong to the same dense region if they are connected through a
chain of neighboring core points.  Points close to a core region can be border
points.  Points not assigned to any dense region are outliers and often receive
label `-1`.

### How density partitioning finds nonlinear regions

The method assumes that a reliable regime has repeated nearby samples.  A point
that appears alone is not treated as a certified regime.  This is appropriate
when a local ROM should be built only where enough training evidence exists.

### API mapping

Use:

- `DensityPartitioner`,
- `RegimeAtlas.outlier_indices`,
- `RegimeAtlas.coverage`.

### Expected result

Dense samples form local regions.  Isolated samples are marked as outliers and
should not automatically get a bankable local model.

### Example

```python
import numpy as np

from pycutfem.mor.regime_atlas import DensityPartitioner

features = np.concatenate(
    [
        np.linspace(0.0, 0.05, 6),
        np.linspace(1.0, 1.05, 6),
        np.array([4.0]),
    ]
).reshape(-1, 1)

atlas = DensityPartitioner(eps=0.08, min_samples=3).fit(features)

assert atlas.n_regions == 2
assert atlas.outlier_indices.tolist() == [12]
print(atlas.coverage)
print(atlas.outlier_indices)
```

The expected result is two dense regions and one outlier.

---

## Topic: Mixture-Model Atlases

### Idea

A mixture model treats regimes probabilistically.  Instead of saying only
"sample `i` belongs to region `r`," it estimates the probability that sample
`i` belongs to each region.

This is useful near region boundaries, where routing may be ambiguous.

### First-principles derivation

A diagonal Gaussian mixture assumes that scaled features are generated by

$$
p(\hat{z})
=
\sum_{r=1}^{K}\pi_r\mathcal{N}(\hat{z};\mu_r,\Sigma_r),
$$

where

$$
\pi_r\ge 0,
\qquad
\sum_{r=1}^{K}\pi_r=1.
$$

The posterior probability of region `r` is

$$
p(r|\hat{z}_i)
=
\frac{\pi_r\mathcal{N}(\hat{z}_i;\mu_r,\Sigma_r)}
{\sum_{s=1}^{K}\pi_s\mathcal{N}(\hat{z}_i;\mu_s,\Sigma_s)}.
$$

The assigned region is

$$
\rho(i)=\arg\max_r p(r|\hat{z}_i).
$$

If

$$
\max_r p(r|\hat{z}_i)<p_{\min},
$$

then the point can be rejected as uncertain or out-of-distribution.

The parameters are usually fitted by expectation-maximization:

1. E-step: estimate posterior probabilities using current parameters.
2. M-step: update weights, means, and variances from posterior-weighted data.
3. Repeat until the likelihood stops improving.

### How mixture models find nonlinear regions

A mixture atlas finds ellipsoidal probability clouds in feature space.  It is
useful when regimes overlap smoothly or when uncertainty in routing is itself
important.  A low maximum probability is a warning that the online point lies
near a boundary or outside the trained regimes.

### API mapping

Use:

- `MixturePartitioner`,
- `RegimeAtlas.metadata` for weights/probability-related information,
- `RegimeOnlineSelector` for rejection/fallback policies.

### Expected result

For two well-separated Gaussian-like clouds, the mixture model returns two
components.  Near boundaries, probability metadata can be used to decide whether
to accept or fallback.

### Example

```python
import numpy as np

from pycutfem.mor.regime_atlas import MixturePartitioner

rng = np.random.default_rng(4)
features = np.vstack(
    [
        rng.normal(loc=-1.0, scale=0.05, size=(20, 1)),
        rng.normal(loc=1.0, scale=0.05, size=(20, 1)),
    ]
)

atlas = MixturePartitioner(n_components=2, min_probability=0.0).fit(features)

assert atlas.n_regions == 2
print(atlas.support_counts)
print(atlas.metadata)
```

The expected result is two probabilistic regions.

---

## Topic: Validation And Atlas Selection

### Idea

A geometric partition is only a proposal.  The atlas is useful only if the local
models trained on its regions actually pass validation.

Therefore, atlas selection should be driven by validation error, support count,
coverage, boundary stability, fallback rate, and complexity.

### First-principles derivation

For region `r`, suppose the local model predicts validation data
`X_pred` and the reference is `X_ref`.  A typical relative error is

$$
E_r^{\mathrm{val}}
=
\frac{\|X_r^{\mathrm{pred}}-X_r^{\mathrm{ref}}\|}
{\|X_r^{\mathrm{ref}}\|}.
$$

The atlas passes if

$$
\max_r E_r^{\mathrm{val}}\le \tau.
$$

But validation error alone is not enough.  Too many regions can overfit.  A
selection score can include penalties:

$$
J(\mathcal{A})
=
\max_r E_r^{\mathrm{val}}
+\lambda C(\mathcal{A})
+\eta K
+\rho E_{\mathrm{boundary}}
+\xi F_{\mathrm{fallback}}.
$$

Here:

- `K` is the number of regions,
- `C(A)` is optional model complexity,
- `E_boundary` measures unstable assignments near boundaries,
- `F_fallback` is the expected fallback rate,
- `lambda`, `eta`, `rho`, and `xi` are penalty weights.

The boundary margin for a sample can be written as

$$
m_i=d_{\mathrm{second}}(z_i)-d_{\mathrm{own}}(z_i).
$$

Small margin means the sample is almost equally close to two regions.  Such
points may switch regions under small perturbations.

### API mapping

Use:

- `RegimeValidationReport`,
- `RegimeValidationSummary`,
- `summarize_region_errors`,
- `boundary_halo_score`,
- `RegimeAtlasCandidate`,
- `RegimeAtlasSelector`,
- `RegimeAtlasSelection`.

### Expected result

The selector should choose the simplest atlas that passes validation.  It
should not select more regions unless the validation improvement is worth the
added complexity.

### Example

```python
import numpy as np

from pycutfem.mor.regime_atlas import (
    KMedoidsPartitioner,
    RegimeAtlasCandidate,
    RegimeAtlasSelector,
    summarize_region_errors,
)

features = np.vstack(
    [np.linspace(-1.0, -0.8, 8), np.linspace(0.8, 1.0, 8)]
).reshape(-1, 1)

coarse = KMedoidsPartitioner(n_regions=1).fit(features)
local = KMedoidsPartitioner(n_regions=2).fit(features)

candidates = (
    RegimeAtlasCandidate(coarse, summarize_region_errors(coarse, [0.25], tolerance=0.1)),
    RegimeAtlasCandidate(local, summarize_region_errors(local, [0.03, 0.04], tolerance=0.1)),
)

selection = RegimeAtlasSelector(max_validation_error=0.1, region_penalty=0.01).select(candidates)

assert selection.selected is candidates[1]
print(selection.scores)
```

The expected result is that the two-region atlas is selected because the
one-region atlas fails the validation tolerance.

---

## Topic: Residual-Error Greedy Splitting

### Idea

Geometry-based clustering can split the data even when a global reduced model
already works.  Residual-error greedy splitting uses validation error to decide
where new regions are actually needed.

It starts with one region.  It trains and validates the local model.  If the
model fails, it splits the worst region.  It accepts the split only if the
validation objective improves enough.

### First-principles derivation

Let an active atlas have regions

$$
\Omega_1,\ldots,\Omega_K.
$$

For each region, train a local model:

$$
M_r = \operatorname{Train}(\Omega_r).
$$

Evaluate the validation error:

$$
E_r=\operatorname{Validate}(M_r,\Omega_r^{\mathrm{val}}).
$$

If

$$
\max_r E_r\le \tau,
$$

stop.  Otherwise choose the worst failing region:

$$
r^*=\arg\max_r E_r.
$$

Propose a split of `Omega_r*` into children, for example by k-medoids with
`K=2` inside that region.  Accept the split if

$$
J_{\mathrm{after}} + \delta < J_{\mathrm{before}}.
$$

The margin `delta` avoids splitting for tiny improvements.  A minimum support
condition avoids single-sample memorization:

$$
|\Omega_{\mathrm{child}}|\ge n_{\min}.
$$

### How residual-greedy splitting finds nonlinear regions

This method finds regions where the **model** fails, not merely where the
feature geometry is spread out.  It is therefore better aligned with ROM/HROM
accuracy.  A region is split only if doing so improves the validation objective.

The core package does not know how to train a PDE-specific ROM.  It receives
callbacks:

```python
model = train_model(indices, dataset)
report = evaluate_model(model, indices, dataset)
```

The callback can train a local POD, a local HROM, a NIRB map, a local GNAT rule,
or any other problem-specific model.

### API mapping

Use:

- `ResidualGreedyConfig`,
- `ResidualGreedySplitter`,
- `ResidualGreedyResult`,
- `ResidualGreedySplitEvent`.

### Expected result

If the global model passes validation, no split is added.  If the mixed region
fails and two child regions pass, one split is accepted.

### Example

```python
import numpy as np

from pycutfem.mor.regime_atlas import RegimeDataset, ResidualGreedyConfig, ResidualGreedySplitter

dataset = RegimeDataset(
    features=np.concatenate([np.linspace(-1.0, -0.1, 20), np.linspace(0.1, 1.0, 20)]).reshape(-1, 1)
)

def train_model(indices, dataset):
    return {"support": int(indices.size)}

def evaluate_model(model, indices, dataset):
    values = dataset.features[indices, 0]
    mixed = np.any(values < 0.0) and np.any(values > 0.0)
    return 1.0 if mixed else 0.01

splitter = ResidualGreedySplitter(
    config=ResidualGreedyConfig(
        max_regions=3,
        min_support=8,
        validation_tolerance=0.1,
        improvement_margin=0.01,
    )
)
result = splitter.fit(dataset, train_model=train_model, evaluate_model=evaluate_model)

assert result.atlas.n_regions == 2
assert result.validation.passed
assert result.accepted_splits == 1
print([event.to_dict() for event in result.events])
```

The expected result is one accepted split.  The toy validation function declares
a region inaccurate if it mixes negative and positive feature values.

---

## Topic: Subspace-Based Region Discovery

### Idea

Sometimes the most meaningful regime information is not in scalar features but
in the local solution spaces themselves.  For example, two parts of a trajectory
may have similar residual norms but require very different POD bases.

Subspace-based discovery compares local bases directly.

### First-principles derivation

Let two local orthonormal bases be

$$
V_a\in\mathbb{R}^{N\times r_a},
\qquad
V_b\in\mathbb{R}^{N\times r_b}.
$$

The principal angles between the two subspaces are defined by the singular
values of

$$
V_a^T V_b.
$$

If

$$
\sigma_j(V_a^T V_b)=\cos(\theta_j),
$$

then

$$
\theta_j=\arccos(\sigma_j).
$$

Small angles mean the spaces are similar.  Large angles mean the spaces point
in different directions.

The chordal distance is

$$
d_c(V_a,V_b)
=
\|\sin(\theta)\|_2
=
\sqrt{\sum_j \sin^2(\theta_j)}.
$$

A subspace partitioner clusters candidate bases using this distance.

### How subspace partitioning finds nonlinear regions

If two local snapshot groups produce nearly the same basis, they probably do
not need separate trial bases.  If their principal angles are large, they
represent genuinely different local linearizations of the nonlinear manifold.

This method is useful after a preliminary segmentation, moving window POD, or
candidate local basis generation.

### API mapping

Use:

- `subspace_principal_angles`,
- `subspace_chordal_distance`,
- `subspace_distance_matrix`,
- `SubspacePartitioner`,
- `SubspacePartition`.

### Expected result

Similar bases are assigned to the same subspace region.  Orthogonal or very
different bases are separated.

### Example

```python
import numpy as np

from pycutfem.mor.regime_atlas import SubspacePartitioner, subspace_chordal_distance, subspace_principal_angles

basis_a = np.eye(4)[:, :2]
basis_b = np.eye(4)[:, :2]
basis_c = np.eye(4)[:, 2:4]
basis_d = np.eye(4)[:, 2:4]

partition = SubspacePartitioner(n_regions=2).fit([basis_a, basis_b, basis_c, basis_d])

assert partition.labels[0] == partition.labels[1]
assert partition.labels[2] == partition.labels[3]
print(subspace_principal_angles(basis_a, basis_c))
print(subspace_chordal_distance(basis_a, basis_c))
print(partition.labels)
```

The expected result is that `basis_a` and `basis_b` share one region, while
`basis_c` and `basis_d` share another.

---

## Topic: Tree Routing

### Idea

Some partitioners are good for offline discovery but inconvenient for online
routing.  A tree router learns simple decision rules from already-discovered
region labels.

For example, after clustering features into region labels, a tree may learn:

```text
if residual_norm < threshold_1:
    use region 0
else if interface_load < threshold_2:
    use region 1
else:
    use region 2
```

This is useful when you want transparent and cheap online routing.

### First-principles derivation

Given features `z_i` and labels `rho_i`, a decision tree recursively splits the
feature space.  At a node, the class proportions are

$$
p_r=\frac{\#\{i:\rho_i=r\}}{\#\{i\}}.
$$

The Gini impurity is

$$
G=1-\sum_r p_r^2.
$$

A candidate split divides the node into left and right children.  The split is
chosen to reduce weighted impurity:

$$
\Delta G
=
G_{\mathrm{parent}}
-
\frac{n_L}{n}G_L
-
\frac{n_R}{n}G_R.
$$

The tree repeats this recursively until depth, support, or purity stopping
criteria are reached.

### How tree routing finds nonlinear regions

The tree does not discover regimes by itself unless labels are supplied.  It
learns a deployable approximation of a region assignment produced by another
method.  Its value is interpretability and cheap online evaluation.

### API mapping

Use:

- `TreeRouter.fit(features, labels)`,
- `TreeRouter.predict(features)`.

### Expected result

For clean labels, the tree should reproduce the training labels and provide a
simple online decision rule.

### Example

```python
import numpy as np

from pycutfem.mor.regime_atlas import TreeRouter

features = np.array([[-1.0], [-0.8], [0.8], [1.0]])
labels = np.array([0, 0, 1, 1])

router = TreeRouter(max_depth=2, min_leaf=1).fit(features, labels)

assert router.predict(features).tolist() == labels.tolist()
print(router.predict(np.array([[-0.9], [0.9]])))
```

The expected result is that negative features route to region `0` and positive
features route to region `1`.

---

## Topic: Local Bank Manifests And Online Novelty

### Idea

A local bank manifest connects each certified region to a saved reduced-model
artifact.  Online selection should not simply choose the nearest artifact.  It
must reject features outside certified radii.

The online selector answers:

> Is this online feature inside a trusted region?  If yes, which model should I
> use?  If no, what fallback reason should I report?

### First-principles derivation

For a bank entry `r`, store:

- feature center `c_r`,
- feature scale `s_r`,
- maximum certified feature distance `rho_r`,
- model path,
- optional step range,
- priority,
- validation certificate.

The normalized distance from an online feature `z` to region `r` is

$$
d_r(z)
=
\left\|\frac{z-c_r}{s_r}\right\|_2.
$$

The feature is accepted by region `r` if

$$
d_r(z)\le \rho_r.
$$

The selected region is usually the closest active accepted region:

$$
r^* = \arg\min_{r:\ d_r(z)\le\rho_r} d_r(z).
$$

If no active region satisfies the radius condition, the selector returns a
fallback decision:

$$
\min_r d_r(z)>\rho_r
\quad\Rightarrow\quad
\text{fallback}.
$$

### How online novelty detection protects the model

Local ROMs are reliable only near their training support.  The radius check is a
cheap novelty detector.  If the feature is outside every certified radius, the
code should run a FOM stage, a global ROM, or an enrichment workflow instead of
committing an untrusted local prediction.

### API mapping

Use:

- `LocalReducedModelBankEntry`,
- `RegimeBankManifest`,
- `build_regime_bank_manifest`,
- `load_regime_bank_manifest`,
- `load_local_reduced_model_bank_manifest`,
- `select_local_reduced_model_bank`,
- `RegimeOnlineSelector`,
- `RegimeNoveltyDecision`,
- `LocalReducedModelSelection`.

### Expected result

A supported feature returns `selected=True`.  An unsupported feature returns
`selected=False` and a reason such as `"outside_certified_region"` or
`"no_active_feature_radius"`.

### Example

```python
import numpy as np

from pycutfem.mor.regime_atlas import (
    LocalReducedModelBankEntry,
    RegimeOnlineSelector,
    build_regime_bank_manifest,
)

entry = LocalReducedModelBankEntry(
    model_id="region_000",
    path="region_000.npz",
    feature_center=np.zeros(1),
    feature_scale=np.ones(1),
    max_feature_distance=0.5,
)

manifest = build_regime_bank_manifest(
    [entry],
    certificates={"region_000": {"max_error": 0.01}},
    fallback_policy={"kind": "global_rom"},
)

selector = RegimeOnlineSelector(entries=manifest.entries, fallback_policy=manifest.fallback_policy)
accepted = selector.select(feature=np.array([0.25]))
rejected = selector.select(feature=np.array([2.0]))

assert accepted.selected
assert rejected.selected is False
print(accepted.to_dict())
print(rejected.to_dict())
```

The expected result is one accepted decision and one fallback decision.

---

## Topic: IQN-ILS And Coupled Fixed-Point Acceleration

### Idea

`IQN-ILS` means **Interface Quasi-Newton with Inverse Least Squares**.  It is a
fixed-point acceleration method often used in partitioned coupled problems such
as FSI, poromechanics coupling, or fluid-biofilm interaction.

The regime atlas does not need to know the physics of IQN-ILS.  However, local
regime selection is often used together with reduced coupling accelerators.
This section gives the theory so the atlas documentation is self-contained.

### First-principles derivation: fixed-point coupling

Suppose a coupled solver updates an interface or coupling variable `x`.  One
partitioned iteration applies a solver map

$$
\tilde{x}_k = H(x_k).
$$

The coupled solution is a fixed point:

$$
x^* = H(x^*).
$$

Define the fixed-point residual

$$
g_k = H(x_k)-x_k = \tilde{x}_k-x_k.
$$

The goal is

$$
g(x^*)=0.
$$

Simple relaxation uses

$$
x_{k+1}=x_k+\omega g_k
=(1-\omega)x_k+\omega\tilde{x}_k,
$$

where `0 < omega <= 1`.  This can converge slowly or fail for strong coupling.

### From Newton to quasi-Newton

Newton's method for the fixed-point residual would solve

$$
J_g(x_k)\Delta x_k = -g_k,
$$

and update

$$
x_{k+1}=x_k+\Delta x_k.
$$

But the Jacobian of the coupled solver map is usually unavailable.  IQN-ILS
builds an approximate inverse action from past coupling iterations.

Store differences of residuals:

$$
\Delta g_j = g_{j+1}-g_j,
$$

and differences of solver outputs or interface iterates:

$$
\Delta \tilde{x}_j = \tilde{x}_{j+1}-\tilde{x}_j.
$$

Collect them into history matrices:

$$
V_k=[\Delta g_{k-m},\ldots,\Delta g_{k-1}],
\qquad
W_k=[\Delta \tilde{x}_{k-m},\ldots,\Delta \tilde{x}_{k-1}].
$$

Here `m` is the history horizon.

IQN-ILS finds coefficients `alpha` such that a linear combination of previous
residual changes cancels the current residual:

$$
\alpha_k
=
\arg\min_\alpha
\|V_k\alpha+g_k\|_2^2.
$$

With Tikhonov regularization, this becomes

$$
\alpha_k
=
\arg\min_\alpha
\|V_k\alpha+g_k\|_2^2
+\lambda\|\alpha\|_2^2.
$$

The normal equations are

$$
(V_k^T V_k+\lambda I)\alpha_k
=
-V_k^T g_k.
$$

Then the next coupling iterate is extrapolated as

$$
x_{k+1}=\tilde{x}_k + W_k\alpha_k.
$$

If there is not enough history, the method falls back to relaxation:

$$
x_{k+1}=x_k+\omega g_k.
$$

### Why this works

The least-squares problem estimates which combination of previous correction
patterns would reduce the current residual.  The same coefficients are then
applied to the output differences to extrapolate a better interface variable.
It is a quasi-Newton method because it approximates the inverse effect of the
fixed-point Jacobian without explicitly forming that Jacobian.

### Reduced IQN-ILS

If the coupling variable is high-dimensional, apply IQN-ILS in reduced
coordinates.  Let

$$
x \approx \bar{x}+V_c a,
$$

where `V_c` is a coupling reduced basis.  Project the residual or coupling
state into reduced coordinates:

$$
a_k = V_c^T M (x_k-\bar{x}),
$$

where `M` is an optional mass matrix.  Then run the same IQN-ILS algebra on the
small vectors `a_k` and reconstruct when needed.

### How IQN-ILS interacts with a regime atlas

A nonlinear coupled problem may need different local coupling bases or
accelerator histories in different regimes.  The atlas can select a local bank
using features such as

$$
z_k=[\log_{10}\|g_k\|,
\log_{10}\|x_k-x_{k-1}\|,
\rho_{\mathrm{contract},k},
\|\tilde{x}_k\|_{\mathrm{rms}}].
$$

Then the online algorithm is:

```text
compute coupling feature z_k
select local regime r using RegimeOnlineSelector
if accepted:
    use local coupling basis V_c^(r)
    run reduced IQN-ILS in that local coordinate system
else:
    use relaxation, global IQN-ILS, FOM coupling, or enrichment fallback
```

### API mapping

The atlas package owns the **region selection** part:

- `RegimeOnlineSelector`,
- `LocalReducedModelBankEntry`,
- `RegimeBankManifest`.

The reduced coupling accelerator may live in a related NIRB/coupling layer, with
names such as:

- `ReducedIQNILS`,
- `iqnils_iteration_matrices`,
- `iqnils_next_iterate`.

Those names are not the region-discovery algorithm itself; they are a common
consumer of the selected local reduced space.

### Expected result

IQN-ILS should reduce the number of fixed-point coupling iterations when the
history is representative.  If the online feature moves outside the certified
regime, the atlas should force fallback rather than reusing an inappropriate
local accelerator.

### Example: pseudocode

The following is pseudocode because the concrete solver map `H` belongs to the
application layer.

```python
# PSEUDOCODE: application-level coupling loop

selector = RegimeOnlineSelector(entries=manifest.entries, fallback_policy=manifest.fallback_policy)
accelerators = load_local_reduced_iqnils_accelerators(manifest)

x = initial_interface_value()

for coupling_iteration in range(max_coupling_iterations):
    x_tilde = coupled_solver_map_H(x)
    g = x_tilde - x

    feature = build_coupling_feature(
        residual_norm=np.linalg.norm(g),
        step_norm=np.linalg.norm(x - x_previous),
        contraction=estimate_contraction(),
        output_norm=np.linalg.norm(x_tilde),
    )

    decision = selector.select(feature=feature)

    if decision.selected:
        accelerator = accelerators[decision.model_id]
        x_next = accelerator.next(x=x, x_tilde=x_tilde, residual=g)
    else:
        # Safe fallback when the local regime is not certified.
        x_next = x + omega * g

    if np.linalg.norm(g) < coupling_tolerance:
        break

    x_previous = x
    x = x_next
```

The expected result is that the atlas chooses an accelerator only inside a
certified local regime.

---

## Topic: Choosing The Number Of Regions

### Idea

The number of regions should not be chosen only by visual inspection.  It should
be selected by a combination of geometry, support, validation, boundary
stability, fallback rate, and cost.

### First-principles derivation

Try candidate region counts

$$
K=1,2,\ldots,K_{\max}.
$$

For each candidate atlas, compute:

1. **Coverage**

   $$
   \mathrm{coverage}
   =
   \frac{\#\{\text{non-outlier samples}\}}{N}.
   $$

2. **Minimum support**

   $$
   \min_r |\Omega_r| \ge n_{\min}.
   $$

3. **Maximum radius**

   $$
   \max_r\max_{i\in\Omega_r}\|\hat{z}_i-c_r\|_2 \le r_{\max}.
   $$

4. **Validation error**

   $$
   \max_r E_r^{\mathrm{val}}\le \tau.
   $$

5. **Boundary stability**

   $$
   \min_i m_i \ge m_{\min}
   $$

   or the halo score is above a minimum margin threshold.

6. **Cost and complexity**

   More regions mean more artifacts, more training, more memory, and possible
   overfitting.

A practical rule is:

> Choose the smallest `K` that passes coverage, support, radius, validation,
> boundary, and timing gates.

### API mapping

Use:

- `RegimeAtlasSelector`,
- `RegimeAtlasCandidate`,
- legacy helper `select_feature_atlas_size`,
- `diagnose_feature_atlas`,
- `boundary_halo_score`.

### Expected result

The selected atlas should be the simplest passing candidate.  The current
selector always returns a best candidate when the input list is nonempty; if no
candidate passes the requested validation tolerance, the caller should inspect
`selection.selected.validation` and decide whether to run a fallback, collect
more data, or broaden the candidate search.

### Example: pseudocode for a K sweep

```python
# PSEUDOCODE: replace train_and_validate_local_models with your application code.

candidates = []

for K in range(1, 9):
    atlas = KMedoidsPartitioner(n_regions=K).fit(training_features)
    region_errors = train_and_validate_local_models(atlas, snapshots, validation_data)
    summary = summarize_region_errors(atlas, region_errors, tolerance=0.02)
    candidates.append(RegimeAtlasCandidate(atlas, summary))

selection = RegimeAtlasSelector(
    max_validation_error=0.02,
    region_penalty=0.005,
).select(candidates)

if selection.selected.validation.max_error <= 0.02:
    train_final_local_banks(selection.selected.atlas)
else:
    run_fom_or_collect_more_training_data(selection.selected.validation)
```

The expected result is not necessarily the largest `K`.  The selected atlas is
the smallest or lowest-score candidate that passes validation.

---

## Topic: End-To-End Offline/Online Pattern

### Idea

The atlas is only one part of a certified local ROM/HROM workflow.  It discovers
regions and routes online features.  The application still trains local models,
validates them, writes artifacts, and defines fallbacks.

### First-principles offline pattern

Offline:

1. Run FOM or trusted high-fidelity simulations.
2. For each sample, store snapshots and online-available features.
3. Scale features.
4. Propose candidate atlases.
5. For each region, train local ROM/HROM/NIRB/IQN-ILS artifacts.
6. Validate each local model on held-out data.
7. Select the simplest passing atlas.
8. Save a bank manifest with centers, scales, radii, certificates, and paths.

Mathematically, the local basis in region `r` is trained from

$$
X_r=[x_i:\rho(i)=r],
$$

and the region is accepted only if

$$
E_r^{\mathrm{val}}\le\tau_r.
$$

### First-principles online pattern

Online:

1. Compute the same feature vector from available online information.
2. Scale it using the offline center and scale.
3. Select a region by distance, probability, tree rule, or manifest radius.
4. If selected, load/use the corresponding local model.
5. If rejected, use fallback.
6. Monitor residuals, error indicators, and cost.
7. If fallback is frequent, collect enrichment data.

### API mapping

Use:

- `RegimeDataset`,
- partitioners,
- `RegimeAtlasSelector`,
- `build_regime_bank_manifest`,
- `RegimeOnlineSelector`,
- `RegimeNoveltyDecision`.

### Expected result

A local model is used only when the online feature lies inside a certified
region.  Otherwise the application receives a clear fallback reason.

### Example: pseudocode

```python
# PSEUDOCODE: offline

dataset = RegimeDataset(features=training_features, groups=case_ids)

candidates = []
for config in atlas_configs:
    atlas = make_regime_partitioner(config).fit(dataset)
    certificates = []
    for region in atlas.regions:
        train_indices = region.sample_indices
        validation_indices = held_out_indices_for_region(region, atlas, validation_split)
        model = train_local_rom(train_indices)
        cert = validate_local_rom(model, validation_indices)
        certificates.append(cert)
    candidates.append(build_candidate(atlas, certificates))

selection = RegimeAtlasSelector(max_validation_error=target_error).select(candidates)
manifest = build_regime_bank_manifest_from_selection(selection)

# PSEUDOCODE: online

selector = RegimeOnlineSelector(entries=manifest.entries, fallback_policy=manifest.fallback_policy)

feature = compute_online_feature()
decision = selector.select(feature=feature)

if decision.selected:
    model = load_model(decision.entry.path)
    result = model.solve_or_predict()
else:
    result = run_fallback(decision.reason)
```

---

## Common Mistakes

- Clustering by time only when time is not the true regime variable.
- Using features offline that are not available online.
- Forgetting to apply the same scaling online as offline.
- Choosing `K` only because the clusters look visually nice.
- Training local bases without validation.
- Treating outliers as bankable regions with too little support.
- Selecting the nearest local model without a certified radius check.
- Ignoring boundary halo samples where region assignment is unstable.
- Using one IQN-ILS history across incompatible nonlinear regimes.
- Reporting local ROM speedup without including fallback frequency.

---

## Migration From Old Modules

### Idea

The implementation lives in `pycutfem.mor.regime_atlas`.  The old public module
paths `pycutfem.mor.feature_atlas` and `pycutfem.mor.local_banks` have been
removed.  Use `pycutfem.mor.regime_atlas` for package-specific imports or
`pycutfem.mor` for the top-level MOR namespace.

### API mapping

Preferred imports:

```python
from pycutfem.mor.regime_atlas import fit_feature_atlas
from pycutfem.mor.regime_atlas import select_local_reduced_model_bank
```

Top-level MOR imports:

```python
from pycutfem.mor import fit_feature_atlas
from pycutfem.mor import select_local_reduced_model_bank
```

### Expected result

Both supported import styles resolve to the same implementation.  Imports from
`pycutfem.mor.feature_atlas` and `pycutfem.mor.local_banks` should fail because
those public shims were removed.

### Example

```python
from pycutfem.mor import fit_feature_atlas as top_level_import
from pycutfem.mor.regime_atlas import fit_feature_atlas as package_import

assert top_level_import is package_import
```

---

## Complete API Inventory

Available from:

```python
from pycutfem.mor.regime_atlas import <name>
```

### Core data and validation

- `RegimeDataset`
- `RegimeRegion`
- `RegimeAtlas`
- `RegimeValidationSplit`
- `RegimeValidationReport`
- `RegimeValidationSummary`
- `as_feature_matrix`
- `as_index_vector`
- `make_validation_split`
- `summarize_region_errors`
- `boundary_halo_score`

### Partitioner interface and strategies

- `RegimePartitioner`
- `RegimePartitionerConfig`
- `coerce_regime_dataset`
- `normalize_region_labels`
- `labels_to_atlas`
- `make_regime_partitioner`
- `KMedoidsPartitioner`
- `fit_kmedoids_regime_atlas`
- `EpsilonCoverPartitioner`
- `HierarchicalPartitioner`
- `DensityPartitioner`
- `MixturePartitioner`
- `TreeRouter`

### Subspace utilities

- `SubspacePartition`
- `SubspacePartitioner`
- `subspace_distance_matrix`
- `subspace_principal_angles`
- `subspace_chordal_distance`

### Residual-error greedy and atlas selection

- `ResidualGreedyConfig`
- `ResidualGreedySplitEvent`
- `ResidualGreedyResult`
- `ResidualGreedySplitter`
- `RegimeAtlasCandidate`
- `RegimeAtlasSelection`
- `RegimeAtlasSelector`

### Banking and online selection

- `LocalReducedModelBankEntry`
- `LocalReducedModelSelection`
- `RegimeBankManifest`
- `build_regime_bank_manifest`
- `load_regime_bank_manifest`
- `load_local_reduced_model_bank_manifest`
- `select_local_reduced_model_bank`
- `RegimeNoveltyDecision`
- `RegimeOnlineSelector`

### Legacy feature-atlas function names

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

### Related coupling/NIRB APIs often used with atlases

These are commonly used by example layers or neighboring MOR/NIRB modules, not
by the atlas partitioner itself:

- `ReducedIQNILS`
- `iqnils_iteration_matrices`
- `iqnils_next_iterate`
- `ReducedSpace`
- `ReducedTransfer`
- `ReducedOutputDecoder`

Use the atlas to decide **which local model or local coupling accelerator is
trusted**.  Use the related coupling/NIRB APIs to perform the actual reduced
prediction or fixed-point acceleration once a region has been selected.
