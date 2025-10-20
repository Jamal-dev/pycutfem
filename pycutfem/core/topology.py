import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional

class Node:
    def __init__(self,id, x, y, tag=None):
        self.x = x
        self.y = y
        self.id = id
        self.tag = tag  

    def __repr__(self):
        return f"Node {self.id}({self.x:.3f}, {self.y:.3f}, tag='{self.tag}')"

    def __eq__(self, other):
        return np.isclose(self.x, other.x) and np.isclose(self.y, other.y)
    def __ne__(self, other):
        return not self.__eq__(other)
    def __lt__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        if np.isclose(self.x, other.x):
            return self.y < other.y
        return self.x < other.x
    def __le__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        if np.isclose(self.x, other.x):
            return self.y <= other.y
        return self.x <= other.x
    def __gt__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        if np.isclose(self.x, other.x):
            return self.y > other.y
        return self.x > other.x
    def __ge__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        if np.isclose(self.x, other.x):
            return self.y >= other.y
        return self.x >= other.x
    def __add__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return Node(self.id, self.x + other.x, self.y + other.y)
    def __sub__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return Node(self.id, self.x - other.x, self.y - other.y)
    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Node(self.id, self.x * scalar, self.y * scalar)
    def __truediv__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        if scalar == 0:
            raise ValueError("Division by zero is not allowed.")
        return Node(self.id, self.x / scalar, self.y / scalar)
    def is_x_equal(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return np.isclose(self.x, other.x)
    def is_y_equal(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return np.isclose(self.y, other.y)

    def __hash__(self):
        return hash((self.x, self.y))
    def __getitem__(self, idx):
        if   idx == 0: return self.x
        elif idx == 1: return self.y
        raise IndexError("Node supports indices 0 (x) and 1 (y)")
    def __iter__(self):
        yield self.x
        yield self.y



@dataclass(slots=True)
class Edge:
    gid: int                   
    nodes: Tuple[int, int]      # Global node indices of the edge's endpoints      
    left: Optional[int]         # Element ID on the left side of the edge 
    right: Optional[int]        # Element ID on the right side of the edge
    normal: np.ndarray          # Normal vector, pointing outward from the left element
    tag: str = ""
    lid: Optional[int] = None      # Local edge index within the left element
    all_nodes: Tuple[int, ...] = ()
    tangent: np.ndarray = field(init=False, default=None)  # Tangent vector of the edge
    def calc_tangent_unit_vector(self, nodes_xy):
        import numpy as np
        i0, i1 = int(self.nodes[0]), int(self.nodes[1])
        x0,y0 = nodes_xy[i0]; x1,y1 = nodes_xy[i1]
        v = np.array([x1 - x0, y1 - y0], float)
        L = float(np.linalg.norm(v))
        if L <= 0: raise ValueError("Zero-length edge.")
        self.tangent = v / L
        return self.tangent

    def calc_normal_unit_vector(self, nodes_xy):
        t = self.calc_tangent_unit_vector(nodes_xy)
        self.normal = np.array([t[1], -t[0]], float)
        return self.normal

    


@dataclass(slots=True)
class Element:
    id: int                     # Element ID
    nodes: Tuple[int, ...]      # Global node indices of the element's vertices
    corner_nodes: Tuple[int, ...] = field(default_factory=tuple)  # Global node indices of the element's corners
    tag: str = ""               # Optional tag for the element
    edges: Tuple[int, ...] = field(default_factory=tuple)
    element_type: str = "quad"
    neighbors: Dict[int, Optional[int]] = field(default_factory=dict)
    poly_order: int = 1
    centroid_x: float = None
    centroid_y: float = None
    interface_pts: List[Tuple[float, float]] = field(default_factory=list)
    def contains_node(self, node_id: int) -> bool:
        """Check if the element contains a specific node."""
        return node_id in self.nodes
    def contains_edge(self, edge_id: int) -> bool:
        """Check if the element contains a specific edge."""
        return edge_id in self.edges
    def centroid(self) -> Tuple[float, float]:
        """Calculate the centroid of the element."""
        if self.centroid_x is None or self.centroid_y is None:
            raise ValueError("Centroid coordinates are not set. Please calculate the centroid first.")
        return self.centroid_x, self.centroid_y
