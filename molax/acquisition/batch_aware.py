"""Batch-aware acquisition functions for active learning.

These methods consider interactions between samples in a batch to select
diverse, informative batches rather than independently selecting top samples.

Includes:
- BatchBALD: Greedy selection maximizing joint mutual information
- DPP: Determinantal Point Process for diversity-quality trade-off

References:
- BatchBALD: Kirsch et al., "BatchBALD: Efficient and Diverse Batch Acquisition
  for Deep Bayesian Active Learning", NeurIPS 2019
- DPP: Biyik et al., "Batch Active Learning Using Determinantal Point Processes",
  arXiv 2019
"""

from typing import List, Optional, Union

import jax.numpy as jnp
import jraph

from molax.models.ensemble import DeepEnsemble
from molax.models.evidential import EvidentialGCN
from molax.models.gcn import UncertaintyGCN
from molax.utils.data import batch_graphs


def batch_bald_sampling(
    model: Union[UncertaintyGCN, DeepEnsemble, EvidentialGCN],
    pool_graphs: List[jraph.GraphsTuple],
    n_select: int,
    n_mc_samples: int = 10,
) -> List[int]:
    """Select a batch maximizing joint mutual information (BatchBALD).

    BatchBALD greedily selects samples that maximize the joint mutual
    information between predictions and model parameters, accounting for
    redundancy between selected samples.

    The greedy approximation selects each sample to maximize the marginal
    gain in joint mutual information given the already-selected samples.

    For efficiency, this implementation uses a simplified approximation that
    penalizes redundancy based on embedding similarity.

    Args:
        model: Model for uncertainty estimation (UncertaintyGCN, DeepEnsemble,
               or EvidentialGCN)
        pool_graphs: List of jraph.GraphsTuple for pool samples
        n_select: Number of samples to select
        n_mc_samples: Number of MC samples for UncertaintyGCN (ignored for
                      DeepEnsemble and EvidentialGCN)

    Returns:
        List of selected indices into pool_graphs
    """
    if not pool_graphs:
        return []

    n_pool = len(pool_graphs)
    n_select = min(n_select, n_pool)

    # Batch all pool graphs
    batched = batch_graphs(pool_graphs)

    # Get individual BALD scores based on model type
    if isinstance(model, DeepEnsemble):
        bald_scores = _compute_ensemble_bald(model, batched, n_pool)
    elif isinstance(model, EvidentialGCN):
        bald_scores = _compute_evidential_bald(model, batched, n_pool)
    else:
        # UncertaintyGCN with MC Dropout
        bald_scores = _compute_mc_bald(model, batched, n_pool, n_mc_samples)

    # Extract embeddings for redundancy penalty
    embeddings = model.extract_embeddings(batched, training=False)[:n_pool]

    # Compute pairwise similarity matrix (RBF kernel)
    # similarity[i,j] = exp(-||e_i - e_j||^2 / (2 * median_dist^2))
    diff = embeddings[:, None, :] - embeddings[None, :, :]
    pairwise_dists = jnp.linalg.norm(diff, axis=2)
    median_dist = jnp.median(pairwise_dists[pairwise_dists > 0])
    bandwidth = jnp.maximum(median_dist, 1e-6)
    similarity = jnp.exp(-(pairwise_dists**2) / (2 * bandwidth**2))

    # Greedy selection with redundancy penalty
    selected: List[int] = []
    remaining_scores = bald_scores.copy()

    for _ in range(n_select):
        # Mask already selected by setting their scores to -inf
        if selected:
            remaining_scores = remaining_scores.at[jnp.array(selected)].set(-jnp.inf)

        # Select best remaining
        best_idx = int(jnp.argmax(remaining_scores))
        selected.append(best_idx)

        # Apply redundancy penalty to similar samples
        # Reduce scores of samples similar to the selected one
        redundancy_penalty = similarity[best_idx] * bald_scores[best_idx]
        remaining_scores = remaining_scores - 0.5 * redundancy_penalty
        remaining_scores = jnp.maximum(remaining_scores, 0)

    return selected


def _compute_mc_bald(
    model: UncertaintyGCN,
    batched: jraph.GraphsTuple,
    n_pool: int,
    n_mc_samples: int,
) -> jnp.ndarray:
    """Compute BALD scores using MC Dropout."""
    means = []
    variances = []

    for _ in range(n_mc_samples):
        mean, var = model(batched, training=True)
        means.append(mean.squeeze(-1))
        variances.append(var.squeeze(-1))

    means = jnp.stack(means, axis=0)
    variances = jnp.stack(variances, axis=0)

    epistemic_var = jnp.var(means, axis=0)
    expected_aleatoric_var = jnp.mean(variances, axis=0)
    total_var = epistemic_var + expected_aleatoric_var

    eps = 1e-8
    bald_scores = 0.5 * jnp.log((total_var + eps) / (expected_aleatoric_var + eps))

    return bald_scores[:n_pool]


def _compute_ensemble_bald(
    ensemble: DeepEnsemble,
    batched: jraph.GraphsTuple,
    n_pool: int,
) -> jnp.ndarray:
    """Compute BALD scores using ensemble disagreement."""
    means = []
    variances = []

    for member in ensemble.members:
        mean, var = member(batched, training=False)
        means.append(mean.squeeze(-1))
        variances.append(var.squeeze(-1))

    means = jnp.stack(means, axis=0)
    variances = jnp.stack(variances, axis=0)

    epistemic_var = jnp.var(means, axis=0)
    expected_aleatoric_var = jnp.mean(variances, axis=0)
    total_var = epistemic_var + expected_aleatoric_var

    eps = 1e-8
    bald_scores = 0.5 * jnp.log((total_var + eps) / (expected_aleatoric_var + eps))

    return bald_scores[:n_pool]


def _compute_evidential_bald(
    model: EvidentialGCN,
    batched: jraph.GraphsTuple,
    n_pool: int,
) -> jnp.ndarray:
    """Compute BALD scores from evidential model."""
    gamma, nu, alpha, beta = model.forward_raw(batched, training=False)

    aleatoric_var = beta / (alpha - 1.0)
    epistemic_var = beta / (nu * (alpha - 1.0))
    total_var = aleatoric_var + epistemic_var

    aleatoric_var = aleatoric_var.squeeze(-1)
    total_var = total_var.squeeze(-1)

    eps = 1e-8
    bald_scores = 0.5 * jnp.log((total_var + eps) / (aleatoric_var + eps))

    return bald_scores[:n_pool]


def dpp_sampling(
    model: Union[UncertaintyGCN, DeepEnsemble, EvidentialGCN],
    pool_graphs: List[jraph.GraphsTuple],
    n_select: int,
    quality_scores: Optional[jnp.ndarray] = None,
    kernel_bandwidth: Optional[float] = None,
) -> List[int]:
    """Select a diverse batch using Determinantal Point Process (DPP).

    DPP models the probability of selecting a subset as proportional to the
    determinant of a kernel matrix restricted to that subset. This naturally
    balances quality (diagonal elements) and diversity (off-diagonal repulsion).

    The L-kernel is constructed as:
        L_ij = q_i * S_ij * q_j

    where q_i are quality scores and S_ij is the similarity kernel.

    This implementation uses greedy MAP (maximum a posteriori) approximation
    for efficient subset selection.

    Args:
        model: Model for embedding extraction (UncertaintyGCN, DeepEnsemble,
               or EvidentialGCN)
        pool_graphs: List of jraph.GraphsTuple for pool samples
        n_select: Number of samples to select
        quality_scores: Optional quality scores for each sample. If None,
                       uses uncertainty scores from the model.
        kernel_bandwidth: Bandwidth for RBF kernel. If None, uses median
                         pairwise distance heuristic.

    Returns:
        List of selected indices into pool_graphs
    """
    if not pool_graphs:
        return []

    n_pool = len(pool_graphs)
    n_select = min(n_select, n_pool)

    # Batch all pool graphs
    batched = batch_graphs(pool_graphs)

    # Extract embeddings
    embeddings = model.extract_embeddings(batched, training=False)[:n_pool]

    # Compute quality scores if not provided
    if quality_scores is None:
        if isinstance(model, DeepEnsemble):
            quality_scores = _compute_ensemble_bald(model, batched, n_pool)
        elif isinstance(model, EvidentialGCN):
            quality_scores = _compute_evidential_bald(model, batched, n_pool)
        else:
            quality_scores = _compute_mc_bald(model, batched, n_pool, n_mc_samples=5)

    # Normalize quality scores to [0, 1]
    q_min = jnp.min(quality_scores)
    q_max = jnp.max(quality_scores)
    if q_max > q_min:
        quality_scores = (quality_scores - q_min) / (q_max - q_min)
    else:
        quality_scores = jnp.ones(n_pool)

    # Add small offset to avoid zero quality
    quality_scores = quality_scores + 0.1

    # Compute pairwise distances
    diff = embeddings[:, None, :] - embeddings[None, :, :]
    pairwise_dists = jnp.linalg.norm(diff, axis=2)

    # Set kernel bandwidth
    if kernel_bandwidth is None:
        # Median heuristic
        nonzero_dists = pairwise_dists[pairwise_dists > 0]
        if nonzero_dists.size > 0:
            kernel_bandwidth = float(jnp.median(nonzero_dists))
        else:
            kernel_bandwidth = 1.0
        kernel_bandwidth = max(kernel_bandwidth, 1e-6)

    # Compute RBF similarity kernel
    similarity = jnp.exp(-(pairwise_dists**2) / (2 * kernel_bandwidth**2))

    # Construct L-kernel: L_ij = q_i * S_ij * q_j
    q = quality_scores[:, None]  # [n_pool, 1]
    L = q * similarity * q.T  # [n_pool, n_pool]

    # Greedy MAP approximation for DPP
    # Select samples that maximize log det(L_S) incrementally
    selected: List[int] = []
    remaining = set(range(n_pool))

    for _ in range(n_select):
        best_idx = -1
        best_gain = -jnp.inf

        for i in remaining:
            # Compute marginal gain of adding i to selected set
            if not selected:
                # First selection: just use diagonal (quality^2)
                gain = L[i, i]
            else:
                # Marginal gain: L_ii - L_iS @ L_SS^{-1} @ L_Si
                # For efficiency, use Cholesky update formulation
                s_indices = jnp.array(selected)
                L_SS = L[jnp.ix_(s_indices, s_indices)]
                L_iS = L[i, s_indices]

                # Add small regularization for numerical stability
                L_SS_reg = L_SS + 1e-6 * jnp.eye(len(selected))

                # Solve L_SS @ x = L_Si
                try:
                    # Use pseudo-inverse for stability
                    L_SS_inv = jnp.linalg.pinv(L_SS_reg)
                    correction = L_iS @ L_SS_inv @ L_iS
                    gain = L[i, i] - correction
                except Exception:
                    gain = L[i, i]

            if gain > best_gain:
                best_gain = gain
                best_idx = i

        if best_idx >= 0:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return selected


def combined_batch_acquisition(
    model: Union[UncertaintyGCN, DeepEnsemble, EvidentialGCN],
    pool_graphs: List[jraph.GraphsTuple],
    n_select: int,
    method: str = "batch_bald",
    n_mc_samples: int = 10,
    quality_scores: Optional[jnp.ndarray] = None,
) -> List[int]:
    """Unified interface for batch-aware acquisition methods.

    Args:
        model: Model for uncertainty/embedding extraction
        pool_graphs: List of jraph.GraphsTuple for pool samples
        n_select: Number of samples to select
        method: Acquisition method - "batch_bald" or "dpp"
        n_mc_samples: Number of MC samples for BALD (UncertaintyGCN only)
        quality_scores: Optional quality scores for DPP

    Returns:
        List of selected indices into pool_graphs
    """
    if method == "batch_bald":
        return batch_bald_sampling(model, pool_graphs, n_select, n_mc_samples)
    elif method == "dpp":
        return dpp_sampling(model, pool_graphs, n_select, quality_scores)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'batch_bald' or 'dpp'.")
