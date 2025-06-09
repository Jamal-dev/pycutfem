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
with open(Path("pycutfem/integration/dunavant_degree_1_to_20.json"), "r") as f:
    dunavant_data = json.load(f)

def get_dunavant_weights(degree):
    """Get the weights for a given degree."""
    return np.array(dunavant_data[str(degree)]["weights"], dtype=np.float64)
def get_dunavant_points(degree):
    """Get the points for a given degree."""
    return np.array(dunavant_data[str(degree)]["points"], dtype=np.float64)
def get_dunavant_data(degree):
    weights = np.array(dunavant_data[str(degree)]["weights"], dtype=np.float64)
    num_points = len(weights)
    points = []
    for i in range(num_points):
        x = dunavant_data[str(degree)]["points"][i][0]
        y = dunavant_data[str(degree)]["points"][i][1]
        points.append([x, y])
    points = np.array(points, dtype=np.float64)
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
