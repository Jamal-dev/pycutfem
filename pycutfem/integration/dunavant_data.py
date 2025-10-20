"""Minimal Dunavant data up to degree 10 (inclusive) for reference triangle.
The table is a dict: degree -> (points, weights) where points is a list of
[xi, eta] coordinate pairs in the reference triangle (0,0)-(1,0)-(0,1).
Weights sum to the reference‐triangle area 0.5.

Only the rules actually needed for P2/Q2 tests are included (deg‑5 & 7).
Add more degrees as needed.
"""
import numpy as np
import json
from pathlib import Path
import os


with open(Path(os.path.dirname(__file__))/Path("dunavant_degree_1_to_20.json"), "r") as f:
    dunavant_data = json.load(f)

def get_dunavant_weights(degree):
    """Get the weights for a given degree."""
    return np.array(dunavant_data[str(degree)]["weights"], dtype=np.float64)
def get_dunavant_points(degree):
    """Return Dunavant points in reference-triangle (xi, eta).
    JSON stores barycentric (L1, L2, L3); we map to (xi, eta) = (L2, L3)."""
    raw = dunavant_data[str(degree)]["points"]
    return np.array([[p[1], p[2]] for p in raw], dtype=np.float64)
def get_dunavant_data(degree):
    """Return (points, weights, num_points) with points in (xi, eta).
    JSON file stores barycentric (L1, L2, L3); our ref triangle uses (xi, eta) = (L2, L3)."""
    raw = dunavant_data[str(degree)]
    weights = np.array(raw["weights"], dtype=np.float64)
    points = np.array([[p[1], p[2]] for p in raw["points"]], dtype=np.float64)
    num_points = len(weights)
    return points, weights, num_points

class DunavantData:
    """Container for Dunavant data."""
    def __init__(self, degree):
        self.degree = degree
        self.points, self.weights, self.num_points = get_dunavant_data(degree)
        self.weights = self.weights / 2.0

    def __repr__(self):
        return f"<DunavantData degree={len(self.points)} points={len(self.points)} weights={len(self.weights)}>"

DUNAVANT = {}
for degree in range(1, 21):
    DUNAVANT[degree] = DunavantData(degree)
    if str(degree) not in dunavant_data.keys():
        raise ValueError(f"Dunavant data for degree {degree} not found.")
