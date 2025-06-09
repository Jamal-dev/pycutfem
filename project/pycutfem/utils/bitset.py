"""pycutfem.utils.bitset"""
import numpy as np
class BitSet:
    def __init__(self, mask):
        self.mask=np.asarray(mask,dtype=bool)
    def union(self,other): return BitSet(self.mask|other.mask)
    def intersect(self,other): return BitSet(self.mask&other.mask)
    def diff(self,other): return BitSet(self.mask&~other.mask)
    def xor(self,other): return BitSet(self.mask^other.mask)
    __or__=union
    __and__=intersect
    __sub__=diff
    __xor__=xor
    def cardinality(self): return int(self.mask.sum())
    def to_indices(self): return np.flatnonzero(self.mask)
    def __len__(self): return len(self.mask)
    def __repr__(self): return f'<BitSet {self.cardinality()}/{len(self)}>'
