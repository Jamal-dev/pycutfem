"""pycutfem.utils.bitset"""
from __future__ import annotations

from hashlib import blake2b
import numpy as np


def _bitset_cache_token(mask: np.ndarray) -> str:
    """Stable token for a boolean mask, used for kernel caching."""
    arr = np.asarray(mask, dtype=np.bool_, order="C")
    h = blake2b(digest_size=16)
    shape_arr = np.asarray(arr.shape, dtype=np.int64)
    h.update(shape_arr.tobytes())
    h.update(arr.view(np.uint8).tobytes())
    return h.hexdigest()


class BitSet:
    def __init__(self, mask):
        self.mask = np.asarray(mask, dtype=bool)
        self._cache_token = _bitset_cache_token(self.mask)

    def union(self, other): return BitSet(self.mask | other.mask)
    def intersect(self, other): return BitSet(self.mask & other.mask)
    def diff(self, other): return BitSet(self.mask & ~other.mask)
    def xor(self, other): return BitSet(self.mask ^ other.mask)
    __or__ = union
    __and__ = intersect
    __sub__ = diff
    __xor__ = xor
    def cardinality(self): return int(self.mask.sum())
    def to_indices(self): return np.flatnonzero(self.mask)
    def __len__(self): return len(self.mask)
    def __repr__(self): return f'<BitSet {self.cardinality()}/{len(self)}>'

    @property
    def array(self):
        """Boolean NumPy array used directly inside JIT kernels."""
        return self.mask

    @property
    def cache_token(self) -> str:
        """Stable identifier for kernel caching."""
        return self._cache_token

    # small conveniences that other code paths already assume
    def __getitem__(self, idx):      # BitSet[i] → bool
        return self.mask[idx]

    def __contains__(self, idx):     # idx in BitSet
        return bool(self.mask[idx])


def bitset_cache_token(mask) -> str:
    """Public helper for non-BitSet boolean masks."""
    return _bitset_cache_token(mask)
