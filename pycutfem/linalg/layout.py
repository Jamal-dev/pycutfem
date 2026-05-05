from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class BlockSpec:
    """Named collection of field names forming one algebraic block."""

    name: str
    fields: tuple[str, ...]

    @classmethod
    def from_value(cls, value, *, index: int) -> "BlockSpec":
        if isinstance(value, cls):
            fields = tuple(dict.fromkeys(str(name) for name in value.fields if str(name)))
            if not fields:
                raise ValueError("BlockSpec.fields must contain at least one field name.")
            return cls(name=str(value.name), fields=fields)

        if isinstance(value, str):
            return cls(name=str(value), fields=(str(value),))

        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
            raw_name = str(value[0])
            raw_fields = value[1]
        else:
            raw_name = f"block_{index}"
            raw_fields = value

        if isinstance(raw_fields, str):
            fields = (str(raw_fields),)
        else:
            fields = tuple(dict.fromkeys(str(name) for name in list(raw_fields or []) if str(name)))
        if not fields:
            raise ValueError(f"Block '{raw_name}' must contain at least one field name.")
        return cls(name=raw_name, fields=fields)


@dataclass(frozen=True)
class FieldBlockLayout:
    """Partition of a reduced system by field names."""

    field_names: np.ndarray
    block_names: tuple[str, ...]
    block_fields: tuple[tuple[str, ...], ...]
    block_indices: tuple[np.ndarray, ...]

    @classmethod
    def from_field_names(
        cls,
        field_names: Sequence[str | None] | np.ndarray,
        blocks: Sequence[BlockSpec | tuple[str, Iterable[str]] | Iterable[str] | str],
        *,
        include_remaining: bool = True,
        remainder_name: str = "remainder",
        allow_empty: bool = False,
    ) -> "FieldBlockLayout":
        names = np.asarray(field_names, dtype=object).reshape(-1)
        used = np.zeros(names.shape, dtype=bool)
        block_names: list[str] = []
        block_fields: list[tuple[str, ...]] = []
        block_indices: list[np.ndarray] = []

        for idx, raw_spec in enumerate(list(blocks or [])):
            spec = BlockSpec.from_value(raw_spec, index=idx)
            if spec.name in block_names:
                raise ValueError(f"Duplicate block name '{spec.name}'.")
            mask = np.isin(names, np.asarray(spec.fields, dtype=object))
            overlap = used & mask
            if np.any(overlap):
                overlap_fields = sorted({str(name) for name in names[overlap].tolist() if str(name)})
                raise ValueError(
                    f"Fields {overlap_fields} were assigned to multiple blocks while building '{spec.name}'."
                )
            block_idx = np.flatnonzero(mask).astype(int, copy=False)
            if block_idx.size == 0 and not allow_empty:
                raise ValueError(
                    f"Block '{spec.name}' did not match any DOFs. Requested fields: {spec.fields}."
                )
            used |= mask
            block_names.append(spec.name)
            block_fields.append(spec.fields)
            block_indices.append(block_idx)

        leftover = np.flatnonzero(~used).astype(int, copy=False)
        if leftover.size:
            if not include_remaining:
                remaining_fields = sorted({str(name) for name in names[leftover].tolist() if str(name)})
                raise ValueError(
                    "Field block layout does not cover the whole vector. "
                    f"Remaining fields: {remaining_fields}."
                )
            remainder_fields = tuple(
                sorted({str(name) for name in names[leftover].tolist() if str(name)})
            )
            block_names.append(str(remainder_name))
            block_fields.append(remainder_fields)
            block_indices.append(leftover)

        return cls(
            field_names=names.copy(),
            block_names=tuple(block_names),
            block_fields=tuple(block_fields),
            block_indices=tuple(np.asarray(idx, dtype=int).copy() for idx in block_indices),
        )

    @property
    def size(self) -> int:
        return int(self.field_names.size)

    @property
    def nblocks(self) -> int:
        return len(self.block_names)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.size, self.size)

    def _normalize_key(self, key: int | str) -> int:
        if isinstance(key, str):
            try:
                return self.block_names.index(str(key))
            except ValueError as exc:
                raise KeyError(f"Unknown block '{key}'.") from exc
        idx = int(key)
        if idx < 0 or idx >= self.nblocks:
            raise IndexError(f"Block index {idx} is out of range for {self.nblocks} blocks.")
        return idx

    def indices(self, key: int | str) -> np.ndarray:
        return np.asarray(self.block_indices[self._normalize_key(key)], dtype=int)

    def fields(self, key: int | str) -> tuple[str, ...]:
        return tuple(self.block_fields[self._normalize_key(key)])

    def split_vector(self, x: Sequence[float] | np.ndarray) -> tuple[np.ndarray, ...]:
        vec = np.asarray(x, dtype=float).reshape(-1)
        if int(vec.size) != self.size:
            raise ValueError(
                f"Vector size {int(vec.size)} does not match block layout size {self.size}."
            )
        return tuple(np.asarray(vec[idx], dtype=float).copy() for idx in self.block_indices)

    def assemble_vector(self, parts: Sequence[Sequence[float] | np.ndarray]) -> np.ndarray:
        if len(parts) != self.nblocks:
            raise ValueError(
                f"Expected {self.nblocks} block vectors but received {len(parts)}."
            )
        out = np.zeros((self.size,), dtype=float)
        for idx, part in zip(self.block_indices, parts):
            part_arr = np.asarray(part, dtype=float).reshape(-1)
            if int(part_arr.size) != int(idx.size):
                raise ValueError(
                    f"Block vector has size {int(part_arr.size)} but expected {int(idx.size)}."
                )
            out[idx] = part_arr
        return out

    def submatrix(
        self,
        matrix: sp.spmatrix | np.ndarray,
        row_block: int | str,
        col_block: int | str,
    ) -> sp.csr_matrix:
        row_idx = self.indices(row_block)
        col_idx = self.indices(col_block)
        if sp.issparse(matrix):
            return matrix.tocsr()[np.ix_(row_idx, col_idx)].tocsr()
        dense = np.asarray(matrix, dtype=float)
        return sp.csr_matrix(dense[np.ix_(row_idx, col_idx)])
