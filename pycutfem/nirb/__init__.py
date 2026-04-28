"""Non-intrusive reduced-basis workflows built on top of :mod:`pycutfem.mor`."""

from .coupling import CouplingIterationRecord, CouplingTrace, NIRBSolidPrediction, NIRBSolidPredictor
from .dataset import DatasetSplit, OfflineDataset, load_cosim_snapshot_batch, load_dataset
from .offline import OfflineConfig, RegressionConfig, TrainedNIRBModel, run_offline_pipeline
from .online import OnlineConfig, load_force_matrix, run_online_pipeline
from .validation import ValidationConfig, validate_rom

__all__ = [
    "CouplingIterationRecord",
    "CouplingTrace",
    "DatasetSplit",
    "NIRBSolidPrediction",
    "NIRBSolidPredictor",
    "OfflineConfig",
    "OfflineDataset",
    "OnlineConfig",
    "RegressionConfig",
    "TrainedNIRBModel",
    "ValidationConfig",
    "load_cosim_snapshot_batch",
    "load_dataset",
    "load_force_matrix",
    "run_offline_pipeline",
    "run_online_pipeline",
    "validate_rom",
]
