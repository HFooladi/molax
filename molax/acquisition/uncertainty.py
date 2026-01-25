"""Acquisition functions for active learning using jraph graphs."""

from typing import List

import jax.numpy as jnp
import jraph

from molax.models.gcn import UncertaintyGCN
from molax.utils.data import batch_graphs


def uncertainty_sampling(
    model: UncertaintyGCN,
    pool_graphs: List[jraph.GraphsTuple],
    n_samples: int = 10,
) -> jnp.ndarray:
    """Compute uncertainty scores for pool samples using MC dropout.

    Args:
        model: UncertaintyGCN model
        pool_graphs: List of jraph.GraphsTuple for pool samples
        n_samples: Number of MC dropout samples

    Returns:
        Array of uncertainty scores for each pool sample
    """
    if not pool_graphs:
        return jnp.array([])

    n_pool = len(pool_graphs)

    # Batch all pool graphs for efficient processing
    batched = batch_graphs(pool_graphs)

    # Collect MC samples
    predictions = []
    for _ in range(n_samples):
        mean, _ = model(batched, training=True)
        predictions.append(mean.squeeze(-1))

    # Stack predictions: [n_samples, n_graphs_with_padding]
    predictions = jnp.stack(predictions, axis=0)

    # Compute variance across MC samples for each pool item
    uncertainties = jnp.var(predictions, axis=0)

    # Return only scores for actual graphs (exclude padding graph)
    return uncertainties[:n_pool]


def diversity_sampling(
    pool_graphs: List[jraph.GraphsTuple],
    labeled_graphs: List[jraph.GraphsTuple],
    n_select: int,
) -> List[int]:
    """Select diverse samples using greedy farthest point sampling.

    Uses mean node features as molecular fingerprints.

    Args:
        pool_graphs: List of pool sample graphs
        labeled_graphs: List of labeled sample graphs
        n_select: Number of samples to select

    Returns:
        List of selected indices into pool_graphs
    """
    if not pool_graphs:
        return []

    n_select = min(n_select, len(pool_graphs))

    # Compute fingerprints as mean of node features
    def get_fingerprint(graph: jraph.GraphsTuple) -> jnp.ndarray:
        return jnp.mean(graph.nodes, axis=0)

    pool_fps = [get_fingerprint(g) for g in pool_graphs]
    labeled_fps = [get_fingerprint(g) for g in labeled_graphs] if labeled_graphs else []

    selected: List[int] = []

    for _ in range(n_select):
        best_idx = -1
        best_min_dist = -float("inf")

        for i in range(len(pool_graphs)):
            if i in selected:
                continue

            # Compute minimum distance to labeled and selected sets
            min_dist = float("inf")

            for fp in labeled_fps:
                dist = float(jnp.linalg.norm(pool_fps[i] - fp))
                min_dist = min(min_dist, dist)

            for j in selected:
                dist = float(jnp.linalg.norm(pool_fps[i] - pool_fps[j]))
                min_dist = min(min_dist, dist)

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i

        if best_idx >= 0:
            selected.append(best_idx)

    return selected


def combined_acquisition(
    model: UncertaintyGCN,
    pool_graphs: List[jraph.GraphsTuple],
    labeled_graphs: List[jraph.GraphsTuple],
    n_select: int,
    uncertainty_weight: float = 0.7,
    n_mc_samples: int = 10,
) -> List[int]:
    """Combined uncertainty and diversity acquisition.

    Args:
        model: UncertaintyGCN model
        pool_graphs: List of pool sample graphs
        labeled_graphs: List of labeled sample graphs
        n_select: Number of samples to select
        uncertainty_weight: Weight for uncertainty vs diversity (0-1)
        n_mc_samples: Number of MC samples for uncertainty estimation

    Returns:
        List of selected indices into pool_graphs
    """
    if not pool_graphs:
        return []

    n_select = min(n_select, len(pool_graphs))

    # Get uncertainty scores
    uncertainties = uncertainty_sampling(model, pool_graphs, n_mc_samples)

    # Normalize uncertainties to [0, 1]
    if jnp.max(uncertainties) > 0:
        norm_uncertainties = uncertainties / jnp.max(uncertainties)
    else:
        norm_uncertainties = uncertainties

    # Get diversity indices and create diversity scores
    diversity_indices = diversity_sampling(pool_graphs, labeled_graphs, n_select)
    diversity_scores = jnp.zeros(len(pool_graphs))
    diversity_scores = diversity_scores.at[jnp.array(diversity_indices)].set(1.0)

    # Combine scores
    combined = (
        uncertainty_weight * norm_uncertainties
        + (1 - uncertainty_weight) * diversity_scores
    )

    # Select top scoring samples
    top_indices = jnp.argsort(-combined)[:n_select]

    return [int(i) for i in top_indices]
