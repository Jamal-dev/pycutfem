from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quadratic_manifold import QuadraticManifoldDecoder


def build_restriction_matrix(interface_dofs: list[int] | np.ndarray, full_dofs: int | list[int]) -> np.ndarray:
    indices = np.asarray(interface_dofs, dtype=int)
    if indices.ndim != 1:
        raise ValueError("interface_dofs must be a 1D list of indices")
    full_size = int(full_dofs if np.isscalar(full_dofs) else len(full_dofs))
    if np.any(indices < 0) or np.any(indices >= full_size):
        raise ValueError("interface index out of bounds")
    matrix = np.zeros((indices.size, full_size), dtype=float)
    matrix[np.arange(indices.size), indices] = 1.0
    return matrix


def build_interface_restriction(
    interface_dofs: list[int] | np.ndarray,
    full_dofs: int | list[int],
) -> "InterfaceRestriction":
    return InterfaceRestriction(matrix=build_restriction_matrix(interface_dofs, full_dofs))


@dataclass
class InterfaceRestriction:
    matrix: np.ndarray

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float)
        if self.matrix.ndim != 2:
            raise ValueError("restriction matrix must be 2D")

    @classmethod
    def from_indices(
        cls,
        interface_dofs: list[int] | np.ndarray,
        full_dofs: int | list[int],
    ) -> "InterfaceRestriction":
        return cls(matrix=build_restriction_matrix(interface_dofs, full_dofs))

    def restrict(self, values: np.ndarray) -> np.ndarray:
        return self.matrix @ np.asarray(values, dtype=float)

    def restrict_basis(self, basis: np.ndarray) -> np.ndarray:
        return self.matrix @ np.asarray(basis, dtype=float)

    def restrict_decoder(self, decoder: QuadraticManifoldDecoder) -> QuadraticManifoldDecoder:
        return decoder.restricted(self.matrix)
