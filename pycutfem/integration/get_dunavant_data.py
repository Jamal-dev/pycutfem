import numpy as np
import json
import quadpy

# Set up storage
dunavant_data = {}

# Loop over degrees 1 to 10
for degree in range(1, 21):
    scheme = quadpy.t2.schemes[f'dunavant_{str(degree).zfill(2)}']()
    points = scheme.points.T.astype(np.float64)[:,:3]  # shape (N, 2) in barycentric
    weights = scheme.weights.astype(np.float64)  # shape (N,)
    
    dunavant_data[degree] = {
        "points": points.tolist(),   # JSON serializable
        "weights": weights.tolist()
    }

# Save to JSON file with high precision
with open("dunavant_degree_1_to_20.json", "w") as f:
    json.dump(dunavant_data, f, indent=2)
