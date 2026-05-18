"""Generic non-intrusive reduced-basis tools built on :mod:`pycutfem.mor`."""

from .dataset import DatasetSplit, NIRBDataset, dataset_from_named_snapshot_batch, load_dataset
from .offline import OfflineConfig, RegressionConfig, TrainedNIRBModel, run_offline_pipeline
from .online import OnlineConfig, load_input_matrix, run_online_pipeline
from .reduced_spaces import (
    ReducedIQNILS,
    ReducedOutputDecoder,
    ReducedSpace,
    ReducedTransfer,
    iqnils_iteration_matrices,
    iqnils_next_iterate,
)
from .validation import ValidationConfig, validate_rom

__all__ = [
    "DatasetSplit",
    "NIRBDataset",
    "OfflineConfig",
    "OnlineConfig",
    "ReducedIQNILS",
    "ReducedOutputDecoder",
    "ReducedSpace",
    "ReducedTransfer",
    "RegressionConfig",
    "TrainedNIRBModel",
    "ValidationConfig",
    "dataset_from_named_snapshot_batch",
    "iqnils_iteration_matrices",
    "iqnils_next_iterate",
    "load_dataset",
    "load_input_matrix",
    "run_offline_pipeline",
    "run_online_pipeline",
    "validate_rom",
]
