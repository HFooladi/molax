"""Molax: Molecular active learning with JAX."""

from ._version import __version__

# Acquisition functions
from .acquisition import (
    bald_sampling,
    batch_bald_sampling,
    combined_acquisition,
    coreset_sampling,
    diversity_sampling,
    dpp_sampling,
    egl_sampling,
    ensemble_bald_sampling,
    ensemble_uncertainty_sampling,
    evidential_uncertainty_sampling,
    uncertainty_sampling,
)

# Metrics
from .metrics import (
    TemperatureScaling,
    evaluate_calibration,
    expected_calibration_error,
    plot_reliability_diagram,
)

# Models
from .models import (
    DeepEnsemble,
    EnsembleConfig,
    EvidentialConfig,
    EvidentialGCN,
    GATConfig,
    GCNConfig,
    GraphTransformerConfig,
    MolecularGCN,
    MPNNConfig,
    UncertaintyGAT,
    UncertaintyGCN,
    UncertaintyGraphTransformer,
    UncertaintyMPNN,
)

# Data utilities
from .utils import (
    ATOM_FEATURIZERS,
    AtomFeaturizer,
    MolecularDataset,
    batch_graphs,
    get_atom_featurizer,
    smiles_to_jraph,
    unbatch_graphs,
)

__all__ = [
    "__version__",
    # Models
    "GCNConfig",
    "UncertaintyGCN",
    "MolecularGCN",
    "MPNNConfig",
    "UncertaintyMPNN",
    "GATConfig",
    "UncertaintyGAT",
    "GraphTransformerConfig",
    "UncertaintyGraphTransformer",
    "EnsembleConfig",
    "DeepEnsemble",
    "EvidentialConfig",
    "EvidentialGCN",
    # Acquisition
    "uncertainty_sampling",
    "ensemble_uncertainty_sampling",
    "evidential_uncertainty_sampling",
    "diversity_sampling",
    "combined_acquisition",
    "bald_sampling",
    "ensemble_bald_sampling",
    "coreset_sampling",
    "batch_bald_sampling",
    "dpp_sampling",
    "egl_sampling",
    # Data utilities
    "MolecularDataset",
    "smiles_to_jraph",
    "batch_graphs",
    "unbatch_graphs",
    "ATOM_FEATURIZERS",
    "AtomFeaturizer",
    "get_atom_featurizer",
    # Metrics
    "expected_calibration_error",
    "evaluate_calibration",
    "TemperatureScaling",
    "plot_reliability_diagram",
]
