"""Mutable sample-state transactions for certified online ROM stages."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class MutableStateSnapshot:
    """Deep copy of named mutable arrays."""

    arrays: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "arrays",
            {str(name): np.asarray(value).copy() for name, value in dict(self.arrays).items()},
        )


@dataclass
class SampleStateTransaction:
    """Snapshot/restore/commit helper for sample-local online state arrays."""

    arrays: Mapping[str, Any]
    restore_on_reject: bool = True
    commit_on_accept: bool = True

    def __post_init__(self) -> None:
        self.arrays = {str(name): np.asarray(value) for name, value in dict(self.arrays).items()}

    def snapshot(self, names: Any | None = None) -> MutableStateSnapshot:
        selected = tuple(self.arrays) if names is None else tuple(str(name) for name in names)
        missing = [name for name in selected if name not in self.arrays]
        if missing:
            raise KeyError(f"transaction state arrays are missing keys {missing!r}.")
        return MutableStateSnapshot({name: self.arrays[name] for name in selected})

    def restore(self, snapshot: MutableStateSnapshot) -> None:
        for name, values in snapshot.arrays.items():
            if name not in self.arrays:
                raise KeyError(f"cannot restore unknown transaction state array {name!r}.")
            target = self.arrays[name]
            source = np.asarray(values)
            if target.shape != source.shape:
                raise ValueError(f"state array {name!r} changed shape during transaction.")
            np.copyto(target, source, casting="unsafe")

    def update(self, name: str, values: Any) -> None:
        key = str(name)
        if key not in self.arrays:
            raise KeyError(key)
        arr = np.asarray(values)
        if arr.shape != self.arrays[key].shape:
            raise ValueError(f"updated values for {key!r} have incompatible shape.")
        np.copyto(self.arrays[key], arr, casting="unsafe")

    def trial(self, names: Any | None = None) -> "SampleStateTrial":
        return SampleStateTrial(self, self.snapshot(names))


class SampleStateTrial(AbstractContextManager):
    """Context manager that restores state unless explicitly accepted."""

    def __init__(self, transaction: SampleStateTransaction, snapshot: MutableStateSnapshot) -> None:
        self.transaction = transaction
        self.snapshot = snapshot
        self.accepted = False

    def accept(self) -> None:
        self.accepted = True

    def reject(self) -> None:
        self.accepted = False
        self.transaction.restore(self.snapshot)

    def __enter__(self) -> "SampleStateTrial":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            if self.transaction.restore_on_reject:
                self.transaction.restore(self.snapshot)
            return False
        if not self.accepted and self.transaction.restore_on_reject:
            self.transaction.restore(self.snapshot)
        return False


__all__ = [
    "MutableStateSnapshot",
    "SampleStateTransaction",
    "SampleStateTrial",
]
