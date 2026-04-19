import numpy as np
import meshio
from typing import Dict, Union, Callable, Optional

from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.expressions import Function, VectorFunction

def export_vtk(
    filename: str,
    mesh: Mesh,
    dof_handler: DofHandler,
    functions: Dict[str, Union[Function, VectorFunction, np.ndarray, Callable[[float, float], float]]],
    *,
    cell_data: Optional[Dict[str, np.ndarray]] = None,
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
                field, node_id = dof_handler._dof_to_node_map.get(int(gdof), (None, None))
                if node_id is None:
                    continue
                if field in obj.field_names:
                    comp = obj.field_names.index(field)
                    vec[node_id, comp] = obj.nodal_values[lidx]
            point_data[name] = vec
            continue

        # Function -> scalar field
        if isinstance(obj, Function):
            scal = np.zeros(num_nodes, dtype=float)
            assigned = np.zeros(num_nodes, dtype=bool)
            for gdof, lidx in obj._g2l.items():
                _field, node_id = dof_handler._dof_to_node_map.get(int(gdof), (None, None))
                if node_id is None:
                    continue
                scal[node_id] = obj.nodal_values[lidx]
                assigned[node_id] = True

            # Visualization quality: when a lower-order CG field (e.g. Q1) lives on a
            # higher-order geometry mesh (e.g. Q2), the DOF-to-node map touches only a
            # subset of mesh nodes. Leaving the remaining nodes at 0 makes ParaView
            # show spurious "holes" and can mislead interpretation (especially when
            # combining fields, e.g. (1-d)*alpha).
            #
            # For quadrilateral Lagrange meshes, we can safely upsample Q1 fields to
            # the mesh nodes by evaluating the bilinear interpolation defined by the
            # element corner values at the high-order node locations.
            if (not np.all(assigned)) and mesh.element_type == "quad" and int(getattr(mesh, "poly_order", 1) or 1) > 1:
                try:
                    p = int(mesh.poly_order)
                    conn_all = np.asarray(getattr(mesh, "elements_connectivity", None))
                    if conn_all.ndim == 2 and conn_all.shape[1] == (p + 1) * (p + 1) and p > 0:
                        for conn in conn_all:
                            # Corner nodes in row-major (j,i) ordering.
                            bl = int(conn[0])
                            br = int(conn[p])
                            tl = int(conn[p * (p + 1)])
                            tr = int(conn[p * (p + 1) + p])
                            if not (assigned[bl] and assigned[br] and assigned[tl] and assigned[tr]):
                                continue
                            f_bl = float(scal[bl])
                            f_br = float(scal[br])
                            f_tr = float(scal[tr])
                            f_tl = float(scal[tl])
                            inv_p = 1.0 / float(p)
                            for j in range(p + 1):
                                t = float(j) * inv_p
                                one_m_t = 1.0 - t
                                for i in range(p + 1):
                                    nid = int(conn[j * (p + 1) + i])
                                    if assigned[nid]:
                                        continue
                                    s = float(i) * inv_p
                                    one_m_s = 1.0 - s
                                    scal[nid] = (
                                        (one_m_s * one_m_t) * f_bl
                                        + (s * one_m_t) * f_br
                                        + (s * t) * f_tr
                                        + (one_m_s * t) * f_tl
                                    )
                        # Do not mark newly filled nodes as "assigned"; corners still remain authoritative.
                except Exception:
                    # Fall back to the sparse mapping (zeros on non-DOF nodes).
                    pass

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

        # NEW: callable defined on coordinates (analytic fields, level sets, ...)
        if callable(obj):
            xy = mesh.nodes_x_y_pos

            def _eval_callable(f, x, y):
                """Try f(x, y); fall back to f([x, y]) if signature differs."""
                try:
                    return f(x, y)
                except TypeError:
                    try:
                        return f(np.array([x, y], dtype=float))
                    except TypeError:
                        return f((x, y))

            first = _eval_callable(obj, float(xy[0, 0]), float(xy[0, 1]))
            first_arr = np.asarray(first, dtype=float)

            if first_arr.ndim == 0:
                vals = np.empty(num_nodes, dtype=float)
                vals[0] = float(first_arr)
                for i, (x, y) in enumerate(xy[1:], start=1):
                    vals[i] = float(_eval_callable(obj, float(x), float(y)))
                point_data[name] = vals
                continue

            if first_arr.ndim == 1 and first_arr.size in (2, 3):
                vec = np.zeros((num_nodes, 3), dtype=float)
                vec[0, : first_arr.size] = first_arr
                for i, (x, y) in enumerate(xy[1:], start=1):
                    res = np.asarray(_eval_callable(obj, float(x), float(y)), dtype=float)
                    if res.ndim != 1 or res.size != first_arr.size:
                        raise ValueError(f"{name}: callable returned inconsistent vector shape {res.shape}")
                    vec[i, : first_arr.size] = res
                point_data[name] = vec
                continue

            raise ValueError(
                f"{name}: callable result with shape {first_arr.shape} is not supported for VTK export"
            )

        raise TypeError(f"{name}: unsupported data type {type(obj)}")

    # 3) write
    cell_data_payload = None
    if cell_data:
        n_cells = len(mesh.corner_connectivity)
        cell_data_payload = {}
        for key, values in cell_data.items():
            arr = np.asarray(values)
            if arr.shape[0] != n_cells:
                raise ValueError(
                    f"cell_data['{key}'] has length {arr.shape[0]}, expected {n_cells} (one value per cell)."
                )
            cell_data_payload[key] = [arr]

    meshio.Mesh(points_3d, cells, point_data=point_data, cell_data=cell_data_payload).write(filename)
    print(f"Solution exported to {filename}")
