"""Core-Set selection for active learning.

Core-Set selection uses k-center greedy algorithm in the model's embedding
space to select a diverse and representative subset of samples. This approach
aims to minimize the maximum distance from any point to its nearest selected
point, providing good coverage of the data distribution.

Reference: Sener & Savarese, "Active Learning for Convolutional Neural Networks:
A Core-Set Approach", ICLR 2018.
"""

from typing import List, Union

import jax.numpy as jnp
import jraph

from molax.models.ensemble import DeepEnsemble
from molax.models.evidential import EvidentialGCN
from molax.models.gcn import UncertaintyGCN
from molax.utils.data import batch_graphs


def coreset_sampling(
    model: Union[UncertaintyGCN, DeepEnsemble, EvidentialGCN],
    pool_graphs: List[jraph.GraphsTuple],
    labeled_graphs: List[jraph.GraphsTuple],
    n_select: int,
) -> List[int]:
    """Select samples using k-center greedy algorithm in embedding space.

    The algorithm iteratively selects the point that is furthest from the
    current set of selected/labeled points, ensuring good coverage of the
    embedding space.

    Algorithm:
    1. Extract embeddings for all pool and labeled graphs
    2. Initialize min-distances from pool to labeled set
    3. Greedy loop: select point with maximum min-distance
    4. Update min-distances with the newly selected point
    5. Repeat until n_select points are selected

    Args:
        model: Model with extract_embeddings method (UncertaintyGCN,
               DeepEnsemble, or EvidentialGCN)
        pool_graphs: List of jraph.GraphsTuple for pool samples
        labeled_graphs: List of jraph.GraphsTuple for labeled samples
        n_select: Number of samples to select

    Returns:
        List of selected indices into pool_graphs
    """
    if not pool_graphs:
        return []

    n_pool = len(pool_graphs)
    n_select = min(n_select, n_pool)

    # Extract embeddings for pool graphs
    pool_batched = batch_graphs(pool_graphs)
    pool_embeddings = model.extract_embeddings(pool_batched, training=False)
    # Only keep actual pool embeddings (exclude padding)
    pool_embeddings = pool_embeddings[:n_pool]

    # Extract embeddings for labeled graphs if any
    if labeled_graphs:
        labeled_batched = batch_graphs(labeled_graphs)
        labeled_embeddings = model.extract_embeddings(labeled_batched, training=False)
        # Only keep actual labeled embeddings (exclude padding)
        labeled_embeddings = labeled_embeddings[: len(labeled_graphs)]
    else:
        labeled_embeddings = None

    # Initialize minimum distances to infinity
    min_distances = jnp.full(n_pool, jnp.inf)

    # Compute initial min-distances to labeled set
    if labeled_embeddings is not None:
        for i in range(len(labeled_graphs)):
            # Euclidean distance from each pool point to this labeled point
            dists = jnp.linalg.norm(
                pool_embeddings - labeled_embeddings[i : i + 1], axis=1
            )
            min_distances = jnp.minimum(min_distances, dists)

    selected: List[int] = []

    for _ in range(n_select):
        # Select the point with maximum min-distance
        # Mask out already selected points by setting their distance to -inf
        if selected:
            masked_distances = min_distances.at[jnp.array(selected)].set(-jnp.inf)
        else:
            masked_distances = min_distances
        best_idx = int(jnp.argmax(masked_distances))
        selected.append(best_idx)

        # Update min-distances with the newly selected point
        new_dists = jnp.linalg.norm(
            pool_embeddings - pool_embeddings[best_idx : best_idx + 1], axis=1
        )
        min_distances = jnp.minimum(min_distances, new_dists)

    return selected


def coreset_sampling_with_scores(
    model: Union[UncertaintyGCN, DeepEnsemble, EvidentialGCN],
    pool_graphs: List[jraph.GraphsTuple],
    labeled_graphs: List[jraph.GraphsTuple],
) -> jnp.ndarray:
    """Compute Core-Set scores (min-distance to labeled set) for all pool samples.

    This returns the minimum distance from each pool sample to the labeled set,
    which can be used as diversity scores or combined with uncertainty scores.

    Args:
        model: Model with extract_embeddings method
        pool_graphs: List of jraph.GraphsTuple for pool samples
        labeled_graphs: List of jraph.GraphsTuple for labeled samples

    Returns:
        Array of Core-Set scores (min-distances) for each pool sample
    """
    if not pool_graphs:
        return jnp.array([])

    n_pool = len(pool_graphs)

    # Extract embeddings for pool graphs
    pool_batched = batch_graphs(pool_graphs)
    pool_embeddings = model.extract_embeddings(pool_batched, training=False)
    pool_embeddings = pool_embeddings[:n_pool]

    # If no labeled data, return infinity for all (all points equally far)
    if not labeled_graphs:
        return jnp.full(n_pool, jnp.inf)

    # Extract embeddings for labeled graphs
    labeled_batched = batch_graphs(labeled_graphs)
    labeled_embeddings = model.extract_embeddings(labeled_batched, training=False)
    labeled_embeddings = labeled_embeddings[: len(labeled_graphs)]

    # Compute pairwise distances and take minimum to labeled set
    # pool_embeddings: [n_pool, hidden_dim]
    # labeled_embeddings: [n_labeled, hidden_dim]
    # Using broadcasting: [n_pool, 1, hidden_dim] - [1, n_labeled, hidden_dim]
    diff = pool_embeddings[:, None, :] - labeled_embeddings[None, :, :]
    # distances: [n_pool, n_labeled]
    distances = jnp.linalg.norm(diff, axis=2)
    # min_distances: [n_pool]
    min_distances = jnp.min(distances, axis=1)

    return min_distances
