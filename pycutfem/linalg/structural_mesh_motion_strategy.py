from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cpp_backend.structural_mesh_motion_strategy import module as _mesh_strategy_module


@dataclass(frozen=True)
class StructuralMeshMotionStrategySettings:
    poisson: float = 0.3
    factor: float = 100.0
    xi: float = 1.5


class StructuralMeshMotionStrategy:
    def __init__(
        self,
        *,
        node_coords,
        connectivity,
        fixed_node_ids,
        interface_node_ids,
        settings: StructuralMeshMotionStrategySettings | None = None,
    ) -> None:
        resolved = settings or StructuralMeshMotionStrategySettings()
        self.node_coords = np.asarray(node_coords, dtype=float)
        self.connectivity = np.asarray(connectivity, dtype=np.int64)
        self.fixed_node_ids = np.asarray(fixed_node_ids, dtype=np.int64).reshape(-1)
        self.interface_node_ids = np.asarray(interface_node_ids, dtype=np.int64).reshape(-1)
        self.settings = resolved
        self._handle = _mesh_strategy_module().StructuralMeshMotionStrategyHandle(
            self.node_coords,
            self.connectivity,
            self.fixed_node_ids,
            self.interface_node_ids,
            float(resolved.poisson),
            float(resolved.factor),
            float(resolved.xi),
        )

    def reset_state(self) -> None:
        self._handle.reset_state()

    def set_state(self, state) -> None:
        self._handle.set_state(np.asarray(state, dtype=float))

    def get_state(self) -> np.ndarray:
        return np.asarray(self._handle.get_state(), dtype=float)

    def solve(
        self,
        *,
        interface_values,
        current_state=None,
        preserve_free_state: bool = True,
    ) -> np.ndarray:
        raw = self._handle.solve(
            np.asarray(interface_values, dtype=float),
            None if current_state is None else np.asarray(current_state, dtype=float),
            bool(preserve_free_state),
        )
        return np.asarray(raw["solution"], dtype=float)
