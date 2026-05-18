"""Example-level NIRB adapters and solver-specific data loaders."""

from .fsi import (
    CouplingIterationRecord,
    CouplingTrace,
    NIRBInterfaceTangentCorrector,
    NIRBSolidPrediction,
    NIRBSolidPredictor,
    load_cosim_snapshot_batch,
)

__all__ = [
    "CouplingIterationRecord",
    "CouplingTrace",
    "NIRBInterfaceTangentCorrector",
    "NIRBSolidPrediction",
    "NIRBSolidPredictor",
    "load_cosim_snapshot_batch",
]
