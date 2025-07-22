"""pycutfem.core.levelset
Enhanced to support multiple level-set functions via *CompositeLevelSet*.
"""
from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np

class LevelSetFunction:
    """Abstract base class"""
    def __call__(self, x: np.ndarray) -> float:
        raise NotImplementedError
    def gradient(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def evaluate_on_nodes(self, mesh) -> np.ndarray:
        return np.apply_along_axis(self, 1, mesh.nodes_x_y_pos)

class CircleLevelSet(LevelSetFunction):
    def __init__(self, center: Tuple[float,float]=(0.,0.), radius: float=1.0):
        self.center=np.asarray(center,dtype=float)
        self.radius=float(radius)
    def __call__(self, x):
        """Signed distance; works for shape (..., 2) or plain (2,)."""
        x = np.asarray(x, dtype=float)
        rel = x - self.center
        # norm along the last axis keeps the leading shape intact
        return np.linalg.norm(rel, axis=-1) - self.radius
    def gradient(self, x):
        d=np.asarray(x-self.center)
        nrm=np.linalg.norm(d)
        return d/nrm if nrm else np.zeros_like(d)

class CompositeLevelSet(LevelSetFunction):
    """Hold several independent level‑set functions.

    Calling the composite returns **an array** of shape (n_levelsets,).
    Gradients stack to (n_levelsets, 2).
    """
    def __init__(self, levelsets: Sequence[LevelSetFunction]):
        self.levelsets=list(levelsets)
    def __call__(self, x):
        return np.array([ls(x) for ls in self.levelsets])
    def gradient(self, x):
        return np.stack([ls.gradient(x) for ls in self.levelsets])
    def evaluate_on_nodes(self, mesh):
        return np.vstack([ls.evaluate_on_nodes(mesh) for ls in self.levelsets])
