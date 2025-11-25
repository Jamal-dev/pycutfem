"""pycutfem.utils.bitset"""
from __future__ import annotations

from hashlib import blake2b
from itertools import count
import numpy as np

# Keep a lightweight, monotonically increasing token source so BitSet IDs stay
# stable even if the underlying mask is mutated in-place between time steps.
_TOKEN_COUNTER = count()


def _bitset_cache_token(mask: np.ndarray) -> str:
    """
    Stable token for a boolean mask, used for kernel caching.

    Previously this hashed the mask contents, which forced a JIT cache miss
    every time the level-set classification changed. We now prefer a stable
    per-object token (when available) and fall back to a lightweight shape/id
    fingerprint to distinguish distinct masks without tying the token to the
    mask values.
    """
    attr_token = getattr(mask, "_cache_token", None)
    if attr_token is not None:
        return attr_token

    arr = np.asarray(mask, dtype=np.bool_, order="C")
    # Distinguish arrays by identity but keep the token short and deterministic
    # within a process. Shape is included to avoid collisions across meshes of
    # different sizes that may be cached simultaneously.
    h = blake2b(digest_size=12)
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    h.update(str(id(arr)).encode())
    return h.hexdigest()


class BitSet:
    def __init__(self, mask):
        self.mask = np.asarray(mask, dtype=bool)
        # Token stays fixed for the lifetime of this BitSet instance.
        self._cache_token = f"bs_{next(_TOKEN_COUNTER)}"

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
