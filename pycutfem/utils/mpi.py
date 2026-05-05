from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

try:
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    MPI = None


@dataclass(frozen=True)
class MpiContext:
    comm: Any | None = None
    rank: int = 0
    size: int = 1

    @property
    def enabled(self) -> bool:
        return int(self.size) > 1 and self.comm is not None

    @property
    def is_root(self) -> bool:
        return int(self.rank) == 0


def get_mpi_context() -> MpiContext:
    if MPI is None:
        return MpiContext()
    try:
        comm = MPI.COMM_WORLD
        return MpiContext(comm=comm, rank=int(comm.Get_rank()), size=int(comm.Get_size()))
    except Exception:
        return MpiContext()


def barrier(ctx: MpiContext | None) -> None:
    if ctx is None or not ctx.enabled:
        return
    try:
        ctx.comm.Barrier()
    except Exception:
        return


def numba_threads_per_rank(numba_module, *, ctx: MpiContext | None = None, requested: int | None = None) -> int:
    ctx = ctx or get_mpi_context()
    env_requested = os.getenv("PYCUTFEM_NUMBA_THREADS_PER_RANK", "").strip()
    if requested is None and env_requested:
        try:
            requested = int(env_requested)
        except Exception:
            requested = None
    if requested is None:
        ncores = int(os.cpu_count() or 1)
        nranks = max(1, int(ctx.size))
        requested = max(1, int(math.floor(ncores / nranks)))

    max_threads = None
    try:
        max_threads = int(getattr(getattr(numba_module, "config", None), "NUMBA_NUM_THREADS", 0) or 0)
    except Exception:
        max_threads = None
    if not max_threads:
        try:
            max_threads = int(numba_module.get_num_threads())
        except Exception:
            max_threads = None
    if max_threads is not None:
        requested = min(int(requested), max(1, int(max_threads)))
    return max(1, int(requested))


def configure_numba_threads(numba_module, *, ctx: MpiContext | None = None, requested: int | None = None) -> int:
    threads = numba_threads_per_rank(numba_module, ctx=ctx, requested=requested)
    numba_module.set_num_threads(int(threads))
    return int(numba_module.get_num_threads())
