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

class AffineLevelSet(LevelSetFunction):
    """
    φ(x, y) = a * x + b * y + c
    Any straight line: choose (a, b, c) so that φ=0 is the line.
    """
    def __init__(self, a: float, b: float, c: float):
        self.a, self.b, self.c = float(a), float(b), float(c)

    # ---- value ------------------------------------------------------
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Works with shape (2,) or (..., 2)
        return self.a * x[..., 0] + self.b * x[..., 1] + self.c

    # ---- gradient ---------------------------------------------------
    def gradient(self, x: np.ndarray) -> np.ndarray:
        g = np.array([self.a, self.b])
        g = g/np.linalg.norm(g) if np.linalg.norm(g) else np.zeros_like(g)
        return g if x.ndim == 1 else np.tile(g, (x.shape[0], 1))

    # ---- optional: signed-distance normalisation --------------------
    def normalised(self):
        """Return a copy scaled so that ‖∇φ‖ = 1 (signed-distance)."""
        norm = np.hypot(self.a, self.b)
        return AffineLevelSet(self.a / norm, self.b / norm, self.c / norm)


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
