"""Reusable reduced-order modeling utilities for non-intrusive ROM workflows."""

from .cross_validation import ModeSweepEntry, ModeSweepResult, run_mode_cross_validation
from .interface import (
    InterfaceRestriction,
    build_interface_restriction,
    build_restriction_matrix,
)
from .io import load_config, load_model, save_config, save_model, save_results
from .metrics import (
    accumulated_iteration_overhead,
    max_online_relative_displacement_error,
    mean_sample_l2_error,
    online_relative_displacement_error,
    reduced_regression_error,
    snapshot_l2_error,
    speedup,
)
from .pod import PODBasis, fit_pod, project, project_to_basis, reconstruct, reconstruct_from_basis
from .quadratic_manifold import (
    QuadraticFeatureMap,
    QuadraticManifoldDecoder,
    fit_quadratic_decoder,
    fit_quadratic_manifold,
    quadratic_feature_matrix,
)
from .regressors import (
    PolynomialFeatureMap,
    PolynomialLassoRegressor,
    ThinPlateSplineRBF,
    fit_poly_lasso,
    fit_tps_rbf,
)
from .scaling import MeanCenterer, StandardScaler
from .snapshots import SnapshotBatch, SnapshotReader, SnapshotWriter
from .timing import Timer, TimingAccumulator, build_speedup_report

__all__ = [
    "InterfaceRestriction",
    "MeanCenterer",
    "ModeSweepEntry",
    "ModeSweepResult",
    "PODBasis",
    "PolynomialFeatureMap",
    "PolynomialLassoRegressor",
    "QuadraticFeatureMap",
    "QuadraticManifoldDecoder",
    "SnapshotBatch",
    "SnapshotReader",
    "SnapshotWriter",
    "StandardScaler",
    "ThinPlateSplineRBF",
    "Timer",
    "TimingAccumulator",
    "accumulated_iteration_overhead",
    "build_interface_restriction",
    "build_restriction_matrix",
    "build_speedup_report",
    "fit_pod",
    "fit_poly_lasso",
    "fit_quadratic_decoder",
    "fit_quadratic_manifold",
    "fit_tps_rbf",
    "load_config",
    "load_model",
    "max_online_relative_displacement_error",
    "mean_sample_l2_error",
    "online_relative_displacement_error",
    "project",
    "project_to_basis",
    "quadratic_feature_matrix",
    "reconstruct",
    "reconstruct_from_basis",
    "reduced_regression_error",
    "run_mode_cross_validation",
    "save_config",
    "save_model",
    "save_results",
    "snapshot_l2_error",
    "speedup",
]
