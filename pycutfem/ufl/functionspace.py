import numpy as np
from pycutfem.fem.reference import get_reference
from pycutfem.core.mesh import Mesh


class FunctionSpace:
    def __init__(self, name, field_names: list[str], dim:int=0, side:str = ""):
        self.name = name  # e.g. "velocity"
        self.field_names = field_names  # e.g. ['ux', 'uy']
        self.dim = dim  # 'scalar', 'vector', 'tensor'
        self.side = side  # e.g. '+', '-'


    
