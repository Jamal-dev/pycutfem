# pycutfem/core/dofhandler.py

import numpy as np
from typing import Dict, List, Set, Callable, Tuple

# We assume the Mesh class and its components are in a sibling file.
from pycutfem.core.mesh import Mesh

from ufl.forms import BoundaryCondition





class DofHandler:
    """
    Manages degrees of freedom (DOFs) for multi-field finite element problems.
    """

    def __init__(self, fe_map: Dict[str, Mesh], method: str = 'cg'):
        """
        Initializes the DofHandler.
        """
        if method not in ['cg', 'dg']:
            raise ValueError("Method must be either 'cg' or 'dg'")

        self.fe_map = fe_map
        self.method = method
        self.field_names = list(fe_map.keys())
        self.field_offsets: Dict[str, int] = {}
        self.field_num_dofs: Dict[str, int] = {}
        self.element_maps: Dict[str, List[List[int]]] = {name: [] for name in self.field_names}
        self.dof_map: Dict[str, Dict] = {name: {} for name in self.field_names}
        self.total_dofs = 0

        if self.method == 'cg':
            self._build_maps_cg()
        else:
            self._build_maps_dg()

    def _build_maps_cg(self):
        """Builds DOF maps for a Continuous Galerkin formulation."""
        print("Building Continuous Galerkin (CG) DOF maps...")
        current_offset = 0
        for field in self.field_names:
            mesh = self.fe_map[field]
            num_nodes_in_field = len(mesh.nodes_list)
            self.field_offsets[field] = current_offset
            self.field_num_dofs[field] = num_nodes_in_field
            self.dof_map[field] = {node_obj.id: current_offset + i for i, node_obj in enumerate(mesh.nodes_list)}
            self.element_maps[field] = [[self.dof_map[field][node_id] for node_id in element.nodes] for element in mesh.elements_list]
            current_offset += num_nodes_in_field
        self.total_dofs = current_offset
        print(f"Total CG DOFs in the system: {self.total_dofs}")

    def _build_maps_dg(self):
        """Builds DOF maps for a Discontinuous Galerkin formulation."""
        # This implementation remains for future use with DG methods.
        print("Building Discontinuous Galerkin (DG) DOF maps...")
        current_offset = 0
        for field in self.field_names:
            mesh = self.fe_map[field]
            self.field_offsets[field] = current_offset
            self.dof_map[field] = {node_obj.id: {} for node_obj in mesh.nodes_list}
            field_dof_count = 0
            for element in mesh.elements_list:
                num_nodes_in_elem = len(element.nodes)
                element_dofs = list(range(current_offset, current_offset + num_nodes_in_elem))
                self.element_maps[field].append(element_dofs)
                for i, node_id in enumerate(element.nodes):
                    self.dof_map[field][node_id][element.id] = element_dofs[i]
                current_offset += num_nodes_in_elem
                field_dof_count += num_nodes_in_elem
            self.field_num_dofs[field] = field_dof_count
        self.total_dofs = current_offset
        print(f"Total DG DOFs in the system: {self.total_dofs}")

    def get_dirichlet_data(self, bcs: List[BoundaryCondition]) -> Dict[int, float]:
        """
        Generates a dictionary mapping Dirichlet DOFs to their specified values.
        This method now correctly processes a list of BoundaryCondition objects
        by looking for both edge tags and node tags.
        """
        dirichlet_data: Dict[int, float] = {}

        for bc in bcs:
            if bc.method != 'dirichlet':
                continue

            field = bc.field
            tag = bc.domain_tag
            value_func = bc.value

            if self.method != 'cg':
                raise NotImplementedError("get_dirichlet_data for DG is not implemented.")

            mesh = self.fe_map[field]
            nodes_with_tag = set()

            # Robustly check for both edge tags AND node tags that match the bc.domain_tag
            
            # 1. Check edge tags for standard boundary conditions
            for edge in mesh.edges_list:
                if edge.tag == tag:
                    nodes_with_tag.update(edge.nodes)

            # 2. Check node tags for single-point constraints (e.g., pressure pinning)
            for node in mesh.nodes_list:
                # The node must have a 'tag' attribute and it must match
                if hasattr(node, 'tag') and node.tag == tag:
                    nodes_with_tag.add(node.id)

            # 3. Get DOFs and values for all unique nodes found
            for node_id in nodes_with_tag:
                dof = self.dof_map[field].get(node_id)
                if dof is None: continue
                
                x, y = mesh.nodes_x_y_pos[node_id]
                bc_value = value_func(x, y)
                dirichlet_data[dof] = bc_value
                
        return dirichlet_data




# ==============================================================================
#  MAIN BLOCK FOR DEMONSTRATION (Using real Mesh class)
# ==============================================================================
if __name__ == '__main__':
    # This block demonstrates the intended workflow using the actual library components.
    
    # These imports assume the user has pycutfem installed or in their PYTHONPATH
    from pycutfem.utils.meshgen import structured_quad
    from pycutfem.core.topology import Node # Mesh needs this
    
    # 1. Generate mesh data using a library utility
    print("Generating a 2x1 P1 mesh...")
    nodes, elems, _, corners = structured_quad(1, 0.5, nx=2, ny=1, poly_order=1)

    # 2. Instantiate the real Mesh object
    mesh = Mesh(nodes=nodes, 
                element_connectivity=elems,
                elements_corner_nodes=corners, 
                element_type="quad", 
                poly_order=1)

    # 3. Define and apply boundary tags
    bc_dict = {'left': lambda x,y: x==0,
               'bottom': lambda x,y:y==0,
               'top': lambda x,y: y==0.5, 
               'right':lambda x,y:x==1}
    mesh.tag_boundary_edges(bc_dict)
    
    # 4. Define the FE space and create the DofHandlers
    fe_map = {'scalar_field': mesh}

    print("\n" + "="*70)
    print("DEMONSTRATION: CONTINUOUS GALERKIN (CG)")
    print("="*70)
    dof_handler_cg = DofHandler(fe_map, method='cg')
    
    print("\nTotal Unique Nodes:", len(nodes))
    print("Total DOFs (CG):", dof_handler_cg.total_dofs)
    
    print("\nElement-to-DOF Maps (CG):")
    for i, elem_map in enumerate(dof_handler_cg.element_maps['scalar_field']):
        print(f"  Element {i}: {elem_map}")
    print("--> Note: DOFs on the shared edge are the same in both lists.")

    print("\n--- Testing get_dirichlet_data (CG) ---")
    dirichlet_def = {
        'left_wall': {
            'fields': ['scalar_field'],
            'tags': ['left'],
            'value': lambda x, y: y * 100.0 # Value is 100 * y-coordinate
        }
    }
    dirichlet_data_cg = dof_handler_cg.get_dirichlet_data(dirichlet_def)
    print("DOF values on the 'left' boundary:")
    for dof, val in sorted(dirichlet_data_cg.items()):
        print(f"  Global DOF {dof}: {val:.1f}")

    print("\n\n" + "="*70)
    print("DEMONSTRATION: DISCONTINUOUS GALERKIN (DG)")
    print("="*70)
    dof_handler_dg = DofHandler(fe_map, method='dg')

    print("\nNodes per Element:", len(elems[0]))
    print("Total DOFs (DG):", dof_handler_dg.total_dofs, f"({len(elems)} elems * {len(elems[0])} nodes/elem)")
    
    print("\nElement-to-DOF Maps (DG):")
    for i, elem_map in enumerate(dof_handler_dg.element_maps['scalar_field']):
        print(f"  Element {i}: {elem_map}")
    print("--> Note: DOF sets are completely separate for each element.")

    # Find the ID of the interior edge between element 0 and 1
    interior_edge_id = -1
    for edge in mesh.edges_list:
        if edge.left is not None and edge.right is not None:
            interior_edge_id = edge.gid
            break
            
    print("\n--- Testing get_dof_pairs_for_edge (DG) ---")
    left_dofs, right_dofs = dof_handler_dg.get_dof_pairs_for_edge('scalar_field', interior_edge_id)
    print(f"DOF pairs for shared edge {interior_edge_id}:")
    print(f"  DOFs from Left Element (Elem {mesh.edges_list[interior_edge_id].left}): {left_dofs}")
    print(f"  DOFs from Right Element (Elem {mesh.edges_list[interior_edge_id].right}): {right_dofs}")

    print("\n--- Testing get_dirichlet_data (DG) ---")
    dirichlet_data_dg = dof_handler_dg.get_dirichlet_data(dirichlet_def)
    print("DOF values on the 'left' boundary:")
    for dof, val in sorted(dirichlet_data_dg.items()):
        print(f"  Global DOF {dof}: {val:.1f}")

