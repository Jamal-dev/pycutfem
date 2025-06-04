"""
Order-agnostic reference-element factory.
"""
from functools import lru_cache
from importlib import import_module

@lru_cache(maxsize=None)
def get_reference(element_type: str, order: int = 1):
    if element_type == "quad":
        shape, grad = import_module("pycutfem.fem.reference.quad_qn").quad_qn(order)
    elif element_type == "tri":
        shape, grad = import_module("pycutfem.fem.reference.tri_pn").tri_pn(order)
    else:
        raise KeyError(element_type)

    class Ref:  # lightweight wrapper
        @staticmethod
        def shape(*args): return shape(*args).astype(float).ravel()
        @staticmethod
        def grad(*args):  return grad(*args).astype(float).T  # (n_loc,2)
    return Ref

