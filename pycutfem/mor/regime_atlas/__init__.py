"""Nonlinear-regime atlas and local reduced-model bank utilities.

The package owns the generic regime-discovery and local-bank layer for MOR/HROM
workflows.  Atlas and bank imports should use ``pycutfem.mor.regime_atlas`` or
the top-level ``pycutfem.mor`` namespace.
"""

from .banking import (
    LocalReducedModelBankEntry,
    LocalReducedModelSelection,
    RegimeBankManifest,
    build_regime_bank_manifest,
    load_local_reduced_model_bank_manifest,
    load_regime_bank_manifest,
    select_local_reduced_model_bank,
)
from .cover import EpsilonCoverPartitioner
from .data import (
    RegimeAtlas,
    RegimeDataset,
    RegimeRegion,
    RegimeValidationSplit,
    as_feature_matrix,
    as_index_vector,
    make_validation_split,
)
from .density import DensityPartitioner
from .features import (
    FeatureAtlasDiagnostics,
    FeatureAtlasFit,
    FeatureAtlasRegion,
    FeatureAtlasSizeSelection,
    KMedoidsResult,
    diagnose_feature_atlas,
    feature_atlas_to_bank_manifest,
    fit_feature_atlas,
    fit_k_medoids,
    robust_feature_center_scale,
    scale_feature_matrix,
    select_feature_atlas_size,
    subspace_chordal_distance,
    subspace_principal_angles,
)
from .hierarchical import HierarchicalPartitioner
from .kmedoids import KMedoidsPartitioner, fit_kmedoids_regime_atlas
from .mixture import MixturePartitioner
from .online import RegimeNoveltyDecision, RegimeOnlineSelector
from .partitioners import (
    RegimePartitioner,
    RegimePartitionerConfig,
    coerce_regime_dataset,
    labels_to_atlas,
    make_regime_partitioner,
    normalize_region_labels,
)
from .residual_greedy import (
    ResidualGreedyConfig,
    ResidualGreedyResult,
    ResidualGreedySplitEvent,
    ResidualGreedySplitter,
)
from .selection import RegimeAtlasCandidate, RegimeAtlasSelection, RegimeAtlasSelector
from .subspace import SubspacePartition, SubspacePartitioner, subspace_distance_matrix
from .tree import TreeRouter
from .validation import (
    RegimeValidationReport,
    RegimeValidationSummary,
    boundary_halo_score,
    summarize_region_errors,
)

__all__ = [
    "DensityPartitioner",
    "EpsilonCoverPartitioner",
    "FeatureAtlasDiagnostics",
    "FeatureAtlasFit",
    "FeatureAtlasRegion",
    "FeatureAtlasSizeSelection",
    "HierarchicalPartitioner",
    "KMedoidsResult",
    "KMedoidsPartitioner",
    "LocalReducedModelBankEntry",
    "LocalReducedModelSelection",
    "MixturePartitioner",
    "RegimeAtlas",
    "RegimeAtlasCandidate",
    "RegimeAtlasSelection",
    "RegimeAtlasSelector",
    "RegimeBankManifest",
    "RegimeDataset",
    "RegimeNoveltyDecision",
    "RegimeOnlineSelector",
    "RegimePartitioner",
    "RegimePartitionerConfig",
    "RegimeRegion",
    "RegimeValidationReport",
    "RegimeValidationSplit",
    "RegimeValidationSummary",
    "ResidualGreedyConfig",
    "ResidualGreedyResult",
    "ResidualGreedySplitEvent",
    "ResidualGreedySplitter",
    "SubspacePartition",
    "SubspacePartitioner",
    "TreeRouter",
    "as_feature_matrix",
    "as_index_vector",
    "boundary_halo_score",
    "build_regime_bank_manifest",
    "coerce_regime_dataset",
    "diagnose_feature_atlas",
    "feature_atlas_to_bank_manifest",
    "fit_feature_atlas",
    "fit_k_medoids",
    "fit_kmedoids_regime_atlas",
    "labels_to_atlas",
    "load_local_reduced_model_bank_manifest",
    "load_regime_bank_manifest",
    "make_regime_partitioner",
    "make_validation_split",
    "normalize_region_labels",
    "robust_feature_center_scale",
    "scale_feature_matrix",
    "select_feature_atlas_size",
    "select_local_reduced_model_bank",
    "subspace_distance_matrix",
    "subspace_chordal_distance",
    "subspace_principal_angles",
    "summarize_region_errors",
]
