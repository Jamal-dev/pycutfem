"""pycutfem.utils.indicator"""
import numpy as np
from .bitset import BitSet
class IndicatorField:
    def __init__(self, mesh, level_set, tol=1e-12):
        phi_nodes=level_set.evaluate_on_nodes(mesh)
        elem_phi=phi_nodes[mesh.elements]
        min_phi=elem_phi.min(axis=1)
        max_phi=elem_phi.max(axis=1)
        self.values=np.where((min_phi<-tol)&(max_phi>tol),1,np.where(max_phi<-tol,-1,2))
    def _wrap(self, op, val): return BitSet(op(self.values, val))
    def __lt__(self,o): return self._wrap(np.less,o)
    def __le__(self,o): return self._wrap(np.less_equal,o)
    def __eq__(self,o): return self._wrap(np.equal,o)
    def __ne__(self,o): return self._wrap(np.not_equal,o)
    def __gt__(self,o): return self._wrap(np.greater,o)
    def __ge__(self,o): return self._wrap(np.greater_equal,o)
    def to_array(self): return self.values
