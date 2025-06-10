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
    def calc_tangent_unit_vector(self):
        """Calculate the unit tangent vector of the edge."""
        dx = self.nodes[1] - self.nodes[0]
        dy = self.nodes[1] - self.nodes[0]
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            raise ValueError("Cannot calculate tangent vector for zero-length edge.")
        self.normal = np.array([dx / length, dy / length]) 
        return self.normal
    def calc_normal_unit_vector(self):
        """Calculate the unit normal vector of the edge."""
        tangent = self.calc_tangent_unit_vector()
        self.tangent = tangent
        return self.tangent


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
