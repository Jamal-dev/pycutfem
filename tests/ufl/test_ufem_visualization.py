import pytest
import numpy as np
import matplotlib.pyplot as plt

# --- Imports from your project structure ---
from pycutfem.core import Mesh
from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.utils.tagging import classify_by_levelset # Corrected import path
from pycutfem.io.visualization import plot_mesh_2 # Corrected import path

def test_unfitted_domain_visualization():
    """
    Creates an unfitted mesh, classifies it, and plots the result
    to visually verify the domain tagging.
    """
    # 1. Setup
    nodes_coords, elems, _, corners = structured_quad(2, 2, nx=10, ny=10, poly_order=1)
    # Correctly instantiate the Mesh object based on the updated design
    mesh = Mesh(nodes=nodes_coords,
                element_connectivity=elems, 
                edges_connectivity=None,
                elements_corner_nodes=corners, 
                element_type="quad", 
                poly_order=1)
    
    levelset = CircleLevelSet(center=(1.0, 1.0), radius=0.75)

    # 2. Classify
    classify_by_levelset(mesh, levelset, 
                         neg_tag='inside', pos_tag='outside', cut_tag='cut',
                         ghost_tag='ghost', interface_tag='interface')

    # 3. Assertions
    element_tags = {elem.tag for elem in mesh.elements_list}
    edge_tags = {edge.tag for edge in mesh.edges_list if edge.right is not None}
    
    assert 'inside' in element_tags
    assert 'outside' in element_tags
    assert 'cut' in element_tags
    assert 'interface' in edge_tags
    assert 'ghost' in edge_tags
    
    print("\n--- Unfitted Classification Results ---")
    print(f"Generated Element Tags: {element_tags}")
    print(f"Generated Interior Edge Tags: {edge_tags}")

    # 4. Visualize
    print("\nGenerating visualization...")
    fig, ax = plt.subplots(figsize=(10, 8))
    # This call now works because of the fixes in visualization.py
    plot_mesh_2(mesh, ax=ax, level_set=levelset, show=True, 
              plot_nodes=False, elem_tags=True, edge_colors=True)
              
    print("Test passed: Classification and visualization completed.")

