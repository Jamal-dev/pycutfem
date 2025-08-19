import numpy as np
import meshio
from typing import Dict, Union, Callable

from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.expressions import Function, VectorFunction

def export_vtk(
    filename: str,
    mesh: Mesh,
    dof_handler: DofHandler,
    functions: Dict[str, Union[Function, VectorFunction, np.ndarray, Callable[[float, float], float]]]
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
    # 1. Prepare mesh geometry 
    points_3d = np.pad(mesh.nodes_x_y_pos, ((0, 0), (0, 1)), constant_values=0)
    if mesh.element_type == 'quad':
        cell_type = 'quad'
    elif mesh.element_type == 'tri':
        cell_type = 'triangle'
    else:
        raise ValueError(f"Unsupported element type for VTK export: {mesh.element_type}")
    cells = [meshio.CellBlock(cell_type, mesh.corner_connectivity)]

    # 2) point data
    point_data = {}
    num_nodes = len(mesh.nodes_list)

    for name, obj in functions.items():
        # VectorFunction -> 3D vector field
        if isinstance(obj, VectorFunction):
            vec = np.zeros((num_nodes, 3))
            for gdof, lidx in obj._g2l.items():
                field, node_id = dof_handler._dof_to_node_map[gdof]
                if field in obj.field_names:
                    comp = obj.field_names.index(field)
                    vec[node_id, comp] = obj.nodal_values[lidx]
            point_data[name] = vec
            continue

        # Function -> scalar field
        if isinstance(obj, Function):
            scal = np.zeros(num_nodes)
            for gdof, lidx in obj._g2l.items():
                _field, node_id = dof_handler._dof_to_node_map[gdof]
                scal[node_id] = obj.nodal_values[lidx]
            point_data[name] = scal
            continue

        # NEW: numpy array (length = num_nodes)
        if isinstance(obj, np.ndarray):
            arr = np.asarray(obj)
            if arr.ndim == 1 and arr.shape[0] == num_nodes:
                point_data[name] = arr
            elif arr.ndim == 2 and arr.shape[0] == num_nodes and arr.shape[1] in (2, 3):
                # pad 2D vectors to 3D as VTK expects
                v = np.zeros((num_nodes, 3)); v[:, :arr.shape[1]] = arr
                point_data[name] = v
            else:
                raise ValueError(f"{name}: unexpected array shape {arr.shape}")
            continue

        # NEW: callable (x,y) -> value  -> evaluate at nodes
        if callable(obj):
            xy = mesh.nodes_x_y_pos
            vals = np.fromiter((obj(float(x), float(y)) for x, y in xy), count=num_nodes, dtype=float)
            point_data[name] = vals
            continue

        raise TypeError(f"{name}: unsupported data type {type(obj)}")

    # 3) write
    meshio.Mesh(points_3d, cells, point_data=point_data).write(filename)
    print(f"Solution exported to {filename}")