"""Acquisition functions for active learning.

This module provides various acquisition functions for selecting informative
samples in active learning pipelines:

Uncertainty-based:
- uncertainty_sampling: MC Dropout variance
- ensemble_uncertainty_sampling: Ensemble disagreement
- evidential_uncertainty_sampling: Evidential uncertainty

BALD (Mutual Information):
- bald_sampling: BALD via MC Dropout
- ensemble_bald_sampling: BALD via ensemble
- evidential_bald_sampling: BALD from evidential parameters

Diversity-based:
- diversity_sampling: Greedy farthest point sampling
- coreset_sampling: K-center greedy in embedding space

Batch-aware:
- batch_bald_sampling: BatchBALD for diverse informative batches
- dpp_sampling: Determinantal Point Process sampling

Gradient-based:
- egl_sampling: Expected Gradient Length
- egl_acquisition: EGL-based sample selection

Combined strategies:
- combined_acquisition: Uncertainty + diversity
- combined_ensemble_acquisition: Ensemble + diversity
- combined_evidential_acquisition: Evidential + diversity
- combined_batch_acquisition: Unified batch-aware interface
"""

from .bald import (
    bald_sampling,
    ensemble_bald_sampling,
    evidential_bald_sampling,
)
from .batch_aware import (
    batch_bald_sampling,
    combined_batch_acquisition,
    dpp_sampling,
)
from .coreset import (
    coreset_sampling,
    coreset_sampling_with_scores,
)
from .expected_change import (
    egl_acquisition,
    egl_sampling,
    egl_sampling_batched,
)
from .uncertainty import (
    combined_acquisition,
    combined_ensemble_acquisition,
    combined_evidential_acquisition,
    diversity_sampling,
    ensemble_uncertainty_sampling,
    evidential_uncertainty_sampling,
    uncertainty_sampling,
)

__all__ = [
    # Uncertainty-based
    "uncertainty_sampling",
    "ensemble_uncertainty_sampling",
    "evidential_uncertainty_sampling",
    # Diversity-based
    "diversity_sampling",
    # Combined strategies
    "combined_acquisition",
    "combined_ensemble_acquisition",
    "combined_evidential_acquisition",
    # BALD (Mutual Information)
    "bald_sampling",
    "ensemble_bald_sampling",
    "evidential_bald_sampling",
    # Core-Set
    "coreset_sampling",
    "coreset_sampling_with_scores",
    # Batch-aware
    "batch_bald_sampling",
    "dpp_sampling",
    "combined_batch_acquisition",
    # Expected Gradient Length
    "egl_sampling",
    "egl_sampling_batched",
    "egl_acquisition",
]
