from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pycutfem.mor.quadratic_manifold import QuadraticFeatureMap


def _as_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values.")
    return vector


def _as_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values.")
    return matrix


def _mass_matrix(mass: np.ndarray | None, n_dofs: int) -> np.ndarray:
    if mass is None:
        return np.eye(int(n_dofs), dtype=float)
    arr = np.asarray(mass, dtype=float)
    if arr.ndim == 1:
        if int(arr.size) != int(n_dofs):
            raise ValueError("diagonal mass size must match the interface dimension.")
        if np.any(arr <= 0.0):
            raise ValueError("diagonal mass entries must be positive.")
        return np.diag(arr)
    if arr.ndim != 2 or arr.shape != (int(n_dofs), int(n_dofs)):
        raise ValueError("mass matrix must have shape (n_dofs, n_dofs).")
    if not np.all(np.isfinite(arr)):
        raise ValueError("mass matrix contains non-finite values.")
    return arr


@dataclass(frozen=True)
class ReducedInterfaceSpace:
    """Mass-weighted reduced basis for interface loads or displacements.

    The basis columns need not be perfectly orthonormal in the supplied mass
    inner product.  Projection solves the small Gram system
    ``(Phi.T M Phi) c = Phi.T M x`` so offline artifacts can use POD bases,
    hand-picked interface modes, or restricted nonlinear-manifold bases.
    """

    basis: np.ndarray
    mass: np.ndarray | None = None
    name: str = "interface"
    _mass_matrix: np.ndarray = field(init=False, repr=False)
    _gram: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        basis = _as_matrix(self.basis, name=f"{self.name}.basis")
        if basis.shape[0] <= 0:
            raise ValueError("interface basis must contain at least one row.")
        object.__setattr__(self, "basis", basis)
        mass = _mass_matrix(self.mass, int(basis.shape[0]))
        gram = basis.T @ (mass @ basis)
        if gram.size:
            try:
                cond = float(np.linalg.cond(gram))
            except np.linalg.LinAlgError:
                cond = float("inf")
            if not np.isfinite(cond) or cond > 1.0e14:
                raise ValueError(f"{self.name} reduced Gram matrix is singular or ill-conditioned.")
        object.__setattr__(self, "_mass_matrix", mass)
        object.__setattr__(self, "_gram", gram)

    @property
    def n_dofs(self) -> int:
        return int(self.basis.shape[0])

    @property
    def n_modes(self) -> int:
        return int(self.basis.shape[1])

    @property
    def mass_matrix(self) -> np.ndarray:
        return np.asarray(self._mass_matrix, dtype=float)

    @property
    def gram(self) -> np.ndarray:
        return np.asarray(self._gram, dtype=float)

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        coeffs = _as_vector(coefficients, name=f"{self.name}.coefficients")
        if int(coeffs.size) != self.n_modes:
            raise ValueError(f"{self.name} expected {self.n_modes} coefficients, got {coeffs.size}.")
        return np.asarray(self.basis @ coeffs, dtype=float).reshape(-1)

    def project(self, values: np.ndarray, *, rcond: float | None = None) -> np.ndarray:
        vector = _as_vector(values, name=f"{self.name}.values")
        if int(vector.size) != self.n_dofs:
            raise ValueError(f"{self.name} expected {self.n_dofs} values, got {vector.size}.")
        if self.n_modes == 0:
            return np.zeros(0, dtype=float)
        rhs = self.basis.T @ (self._mass_matrix @ vector)
        try:
            coeffs = np.linalg.solve(self._gram, rhs)
        except np.linalg.LinAlgError:
            coeffs, *_ = np.linalg.lstsq(self._gram, rhs, rcond=rcond)
        return np.asarray(coeffs, dtype=float).reshape(-1)

    def inner(self, left: np.ndarray, right: np.ndarray) -> float:
        left_vec = _as_vector(left, name=f"{self.name}.left")
        right_vec = _as_vector(right, name=f"{self.name}.right")
        if left_vec.size == self.n_modes:
            left_vec = self.reconstruct(left_vec)
        if right_vec.size == self.n_modes:
            right_vec = self.reconstruct(right_vec)
        if int(left_vec.size) != self.n_dofs or int(right_vec.size) != self.n_dofs:
            raise ValueError("interface inner-product arguments have incompatible sizes.")
        return float(left_vec @ (self._mass_matrix @ right_vec))

    def norm(self, values: np.ndarray) -> float:
        vector = _as_vector(values, name=f"{self.name}.values")
        if vector.size == self.n_modes:
            vector = self.reconstruct(vector)
        return float(np.sqrt(max(self.inner(vector, vector), 0.0)))

    def relative_change(self, new: np.ndarray, old: np.ndarray, *, floor: float = 1.0e-14) -> tuple[float, float]:
        new_vec = _as_vector(new, name=f"{self.name}.new")
        old_vec = _as_vector(old, name=f"{self.name}.old")
        if new_vec.size == self.n_modes:
            new_vec = self.reconstruct(new_vec)
        if old_vec.size == self.n_modes:
            old_vec = self.reconstruct(old_vec)
        diff_norm = self.norm(new_vec - old_vec)
        base = max(self.norm(new_vec), float(floor))
        abs_rms = diff_norm / np.sqrt(max(float(self.n_dofs), 1.0))
        return float(abs_rms), float(diff_norm / base)


@dataclass(frozen=True)
class ReducedTransfer:
    """Small matrix transfer between two reduced interface spaces."""

    matrix: np.ndarray
    source: ReducedInterfaceSpace
    target: ReducedInterfaceSpace

    def __post_init__(self) -> None:
        matrix = _as_matrix(self.matrix, name="transfer.matrix")
        expected = (self.target.n_modes, self.source.n_modes)
        if matrix.shape != expected:
            raise ValueError(f"transfer matrix shape {matrix.shape} does not match {expected}.")
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def from_full_transfer(
        cls,
        *,
        source: ReducedInterfaceSpace,
        target: ReducedInterfaceSpace,
        full_transfer: np.ndarray,
    ) -> "ReducedTransfer":
        transfer = _as_matrix(full_transfer, name="full_transfer")
        if transfer.shape != (target.n_dofs, source.n_dofs):
            raise ValueError("full transfer shape is incompatible with source/target spaces.")
        lifted_source = transfer @ source.basis
        rhs = target.basis.T @ (target.mass_matrix @ lifted_source)
        try:
            matrix = np.linalg.solve(target.gram, rhs)
        except np.linalg.LinAlgError:
            matrix, *_ = np.linalg.lstsq(target.gram, rhs, rcond=None)
        return cls(matrix=np.asarray(matrix, dtype=float), source=source, target=target)

    def apply(self, coefficients: np.ndarray) -> np.ndarray:
        coeffs = _as_vector(coefficients, name="source coefficients")
        if int(coeffs.size) != self.source.n_modes:
            raise ValueError(f"expected {self.source.n_modes} source coefficients, got {coeffs.size}.")
        return np.asarray(self.matrix @ coeffs, dtype=float).reshape(-1)


@dataclass(frozen=True)
class ReducedInterfaceDecoder:
    """Reduced output map for a quadratic solid displacement manifold."""

    linear_map: np.ndarray
    quadratic_map: np.ndarray
    bias: np.ndarray
    output_space: ReducedInterfaceSpace
    feature_map: QuadraticFeatureMap | None = None

    def __post_init__(self) -> None:
        linear = _as_matrix(self.linear_map, name="linear_map")
        quadratic = _as_matrix(self.quadratic_map, name="quadratic_map")
        bias = _as_vector(self.bias, name="bias")
        if int(linear.shape[0]) != self.output_space.n_modes:
            raise ValueError("linear map rows must match output-space modes.")
        if int(bias.size) != self.output_space.n_modes:
            raise ValueError("bias size must match output-space modes.")
        if self.feature_map is None:
            feature_map = QuadraticFeatureMap(rank=int(linear.shape[1]))
        else:
            feature_map = self.feature_map
        if quadratic.shape != (self.output_space.n_modes, feature_map.n_terms):
            raise ValueError("quadratic map shape does not match output-space modes and quadratic features.")
        object.__setattr__(self, "linear_map", linear)
        object.__setattr__(self, "quadratic_map", quadratic)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "feature_map", feature_map)

    @classmethod
    def from_full_decoder(
        cls,
        *,
        linear_basis: np.ndarray,
        quadratic_basis: np.ndarray,
        mean: np.ndarray | None,
        output_space: ReducedInterfaceSpace,
        restriction_matrix: np.ndarray | None = None,
        feature_map: QuadraticFeatureMap | None = None,
    ) -> "ReducedInterfaceDecoder":
        linear_full = _as_matrix(linear_basis, name="linear_basis")
        quadratic_full = _as_matrix(quadratic_basis, name="quadratic_basis")
        if restriction_matrix is None:
            restricted_linear = linear_full
            restricted_quadratic = quadratic_full
            restricted_mean = np.zeros(linear_full.shape[0], dtype=float) if mean is None else _as_vector(mean, name="mean")
        else:
            restriction = _as_matrix(restriction_matrix, name="restriction_matrix")
            restricted_linear = restriction @ linear_full
            restricted_quadratic = restriction @ quadratic_full
            if mean is None:
                restricted_mean = np.zeros(restriction.shape[0], dtype=float)
            else:
                restricted_mean = restriction @ _as_vector(mean, name="mean")
        if int(restricted_linear.shape[0]) != output_space.n_dofs:
            raise ValueError("restricted decoder rows must match output-space dimension.")
        linear_map = np.column_stack(
            [output_space.project(restricted_linear[:, j]) for j in range(restricted_linear.shape[1])]
        )
        quadratic_map = np.column_stack(
            [output_space.project(restricted_quadratic[:, j]) for j in range(restricted_quadratic.shape[1])]
        )
        bias = output_space.project(restricted_mean)
        return cls(
            linear_map=linear_map,
            quadratic_map=quadratic_map,
            bias=bias,
            output_space=output_space,
            feature_map=feature_map,
        )

    def decode_coefficients(self, reduced_displacement: np.ndarray) -> np.ndarray:
        coords = _as_vector(reduced_displacement, name="reduced_displacement")
        if int(coords.size) != int(self.linear_map.shape[1]):
            raise ValueError(f"expected {self.linear_map.shape[1]} displacement coordinates, got {coords.size}.")
        features = self.feature_map.transform(coords.reshape(-1, 1))[:, 0]
        return np.asarray(self.bias + self.linear_map @ coords + self.quadratic_map @ features, dtype=float)

    def decode_interface(self, reduced_displacement: np.ndarray) -> np.ndarray:
        return self.output_space.reconstruct(self.decode_coefficients(reduced_displacement))


def iqnils_iteration_matrices(
    *,
    x_history: list[np.ndarray],
    g_history: list[np.ndarray],
    iteration_horizon: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    keep_count = min(max(int(iteration_horizon), 1), len(x_history), len(g_history))
    if keep_count <= 1:
        return None, None
    x_seq = [_as_vector(values, name="x_history") for values in x_history[-keep_count:]]
    g_seq = [_as_vector(values, name="g_history") for values in g_history[-keep_count:]]
    r_seq = [g_vec - x_vec for x_vec, g_vec in zip(x_seq, g_seq)]
    r_recent = list(reversed(r_seq))
    g_recent = list(reversed(g_seq))
    if len(r_recent) <= 1:
        return None, None
    v_new = np.column_stack(r_recent[:-1]) - np.column_stack(r_recent[1:])
    w_new = np.column_stack(g_recent[:-1]) - np.column_stack(g_recent[1:])
    return np.asarray(v_new, dtype=float), np.asarray(w_new, dtype=float)


def iqnils_next_iterate(
    *,
    x_curr: np.ndarray,
    g_curr: np.ndarray,
    x_history: list[np.ndarray],
    g_history: list[np.ndarray],
    dr_old_mats: list[np.ndarray] | None = None,
    dg_old_mats: list[np.ndarray] | None = None,
    omega: float,
    horizon: int,
    regularization: float = 0.0,
) -> np.ndarray:
    x_curr_arr = np.asarray(x_curr, dtype=float)
    x_curr_vec = _as_vector(x_curr_arr, name="x_curr")
    g_curr_vec = _as_vector(g_curr, name="g_curr")
    if x_curr_vec.shape != g_curr_vec.shape:
        raise ValueError("x_curr and g_curr must have the same size.")
    r_curr = g_curr_vec - x_curr_vec
    alpha = float(np.clip(float(omega), 0.0, 1.0))
    picard = x_curr_vec + alpha * r_curr

    keep_count = min(max(int(horizon), 1), len(x_history), len(g_history))
    if keep_count <= 0:
        return picard.reshape(x_curr_arr.shape)
    x_seq = [_as_vector(values, name="x_history") for values in x_history[-keep_count:]]
    g_seq = [_as_vector(values, name="g_history") for values in g_history[-keep_count:]]
    r_seq = [g_vec - x_vec for x_vec, g_vec in zip(x_seq, g_seq)]
    r_recent = list(reversed(r_seq))
    g_recent = list(reversed(g_seq))
    k = len(r_recent) - 1

    v_old_blocks = [_as_matrix(block, name="dr_old") for block in list(dr_old_mats or []) if np.asarray(block).size]
    w_old_blocks = [_as_matrix(block, name="dg_old") for block in list(dg_old_mats or []) if np.asarray(block).size]
    has_old = bool(v_old_blocks and w_old_blocks)
    if (not has_old) and k == 0:
        return picard.reshape(x_curr_arr.shape)

    if k > 0:
        v_new = np.column_stack(r_recent[:-1]) - np.column_stack(r_recent[1:])
        w_new = np.column_stack(g_recent[:-1]) - np.column_stack(g_recent[1:])
    else:
        v_new = np.empty((r_curr.size, 0), dtype=float)
        w_new = np.empty((g_curr_vec.size, 0), dtype=float)

    if has_old:
        v_old = np.hstack(v_old_blocks)
        w_old = np.hstack(w_old_blocks)
        v = np.hstack((v_new, v_old)) if k > 0 else v_old
        w = np.hstack((w_new, w_old)) if k > 0 else w_old
    else:
        v = v_new
        w = w_new
    if v.size == 0 or w.size == 0:
        return picard.reshape(x_curr_arr.shape)

    delta_r = -r_recent[0]
    reg = max(float(regularization), 0.0)
    try:
        if reg > 0.0:
            n_cols = int(v.shape[1])
            v_aug = np.vstack([v, np.sqrt(reg) * np.eye(n_cols, dtype=float)])
            rhs_aug = np.concatenate([delta_r, np.zeros(n_cols, dtype=float)])
            gamma = np.linalg.lstsq(v_aug, rhs_aug, rcond=None)[0]
        else:
            gamma = np.linalg.lstsq(v, delta_r, rcond=None)[0]
    except np.linalg.LinAlgError:
        return picard.reshape(x_curr_arr.shape)

    delta_x = w @ gamma - delta_r
    if not np.all(np.isfinite(delta_x)):
        return picard.reshape(x_curr_arr.shape)
    return (x_curr_vec + delta_x).reshape(x_curr_arr.shape)


@dataclass
class ReducedIQNILS:
    """Reduced-coordinate IQN-ILS update state for partitioned coupling."""

    omega: float = 0.125
    horizon: int = 50
    regularization: float = 0.0
    x_history: list[np.ndarray] = field(default_factory=list)
    g_history: list[np.ndarray] = field(default_factory=list)
    old_dr_mats: list[np.ndarray] = field(default_factory=list)
    old_dg_mats: list[np.ndarray] = field(default_factory=list)

    def next(self, current: np.ndarray, returned: np.ndarray, *, converged: bool = False) -> np.ndarray:
        x = _as_vector(current, name="current")
        g = _as_vector(returned, name="returned")
        if x.shape != g.shape:
            raise ValueError("current and returned reduced loads must have the same shape.")
        if bool(converged):
            return g.copy()
        self.x_history.append(x.copy())
        self.g_history.append(g.copy())
        return iqnils_next_iterate(
            x_curr=x,
            g_curr=g,
            x_history=self.x_history,
            g_history=self.g_history,
            dr_old_mats=self.old_dr_mats,
            dg_old_mats=self.old_dg_mats,
            omega=float(self.omega),
            horizon=int(self.horizon),
            regularization=float(self.regularization),
        ).reshape(-1)

    def finalize_step(self) -> None:
        v_new, w_new = iqnils_iteration_matrices(
            x_history=self.x_history,
            g_history=self.g_history,
            iteration_horizon=int(self.horizon),
        )
        if v_new is not None and w_new is not None:
            self.old_dr_mats.insert(0, np.asarray(v_new, dtype=float))
            self.old_dg_mats.insert(0, np.asarray(w_new, dtype=float))
        self.x_history.clear()
        self.g_history.clear()


__all__ = [
    "ReducedIQNILS",
    "ReducedInterfaceDecoder",
    "ReducedInterfaceSpace",
    "ReducedTransfer",
    "iqnils_iteration_matrices",
    "iqnils_next_iterate",
]
