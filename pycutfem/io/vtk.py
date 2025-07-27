import numpy as np
import meshio
from typing import Dict, Union

from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.expressions import Function, VectorFunction

def export_vtk(
    filename: str,
    mesh: Mesh,
    dof_handler: DofHandler,
    functions: Dict[str, Union[Function, VectorFunction]]
):
    """
    Exports simulation data to a VTK (.vtu) file for visualization. (CORRECTED)

    Args:
        filename: The path to the output file (e.g., 'results/solution_001.vtu').
        mesh: The computational mesh object.
        dof_handler: The DofHandler linking DOFs to nodes.
        functions: A dictionary mapping field names to the Function or
                   VectorFunction objects to be exported.
    """
    # 1. Prepare mesh geometry (this part was correct)
    points_3d = np.pad(mesh.nodes_x_y_pos, ((0, 0), (0, 1)), constant_values=0)
    
    if mesh.element_type == 'quad':
        cell_type = 'quad'
    elif mesh.element_type == 'tri':
        cell_type = 'triangle'
    else:
        raise ValueError(f"Unsupported element type for VTK export: {mesh.element_type}")

    cells = [meshio.CellBlock(cell_type, mesh.corner_connectivity)]

    # 2. Prepare point data from Function objects 
    point_data = {}
    num_nodes = len(mesh.nodes_list)

    for name, func in functions.items():
        if isinstance(func, VectorFunction):
            vector_data = np.zeros((num_nodes, 3))
            
            # Iterate over the VectorFunction's own (g)lobal-to-(l)ocal map.
            for gdof, lidx in func._g2l.items():
                # Find the node and field this global DOF belongs to.
                field, node_id = dof_handler._dof_to_node_map[gdof]
                
                # Find which component (0 for ux, 1 for uy) this field is.
                if field in func.field_names:
                    comp_idx = func.field_names.index(field)
                    # Get the value from the function's data array.
                    value = func.nodal_values[lidx]
                    # Place it in the correct slot in our node-ordered array.
                    vector_data[node_id, comp_idx] = value
            
            point_data[name] = vector_data

        elif isinstance(func, Function):
            scalar_data = np.zeros(num_nodes)

            # Iterate over the scalar Function's own map.
            for gdof, lidx in func._g2l.items():
                # Find the node this global DOF belongs to.
                _field, node_id = dof_handler._dof_to_node_map[gdof]
                
                # Get the value and place it in the node-ordered array.
                scalar_data[node_id] = func.nodal_values[lidx]

            point_data[name] = scalar_data

    # 3. Create a meshio.Mesh object and write to file (this part was correct)
    mesh_out = meshio.Mesh(points_3d, cells, point_data=point_data)
    mesh_out.write(filename)
    print(f"Solution exported to {filename}")