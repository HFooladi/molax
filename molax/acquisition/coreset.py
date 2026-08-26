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


def coreset_from_embeddings(
    embeddings: jnp.ndarray,
    pool_mask: jnp.ndarray,
    n_select: int,
) -> List[int]:
    """K-center greedy over a precomputed embedding matrix.

    This is the index-based form of :func:`coreset_sampling`. It is the right
    entry point when all molecules already live in one pre-batched
    ``GraphsTuple`` and active learning is driven by a boolean mask, because it
    avoids re-batching the pool every round (re-batching changes array shapes
    and triggers JIT recompilation, which is what the fixed-batch pattern
    exists to avoid).

    Args:
        embeddings: Embeddings for every molecule, shape [n_total, hidden_dim]
        pool_mask: Boolean mask, True for unlabeled candidates. Labeled points
            are treated as already-covered centers.
        n_select: Number of points to select

    Returns:
        List of selected indices into the full ``embeddings`` array (not into
        the pool subset), so they can be applied to the mask directly.
    """
    pool_indices = [int(i) for i in jnp.where(pool_mask)[0]]
    if not pool_indices:
        return []

    n_select = min(n_select, len(pool_indices))

    # Distance from every point to the nearest already-labeled point.
    labeled_mask = ~pool_mask
    if bool(jnp.any(labeled_mask)):
        labeled_embeddings = embeddings[labeled_mask]
        diff = embeddings[:, None, :] - labeled_embeddings[None, :, :]
        min_distances = jnp.min(jnp.linalg.norm(diff, axis=2), axis=1)
    else:
        min_distances = jnp.full(embeddings.shape[0], jnp.inf)

    # Never pick an already-labeled point.
    min_distances = jnp.where(pool_mask, min_distances, -jnp.inf)

    selected: List[int] = []
    for _ in range(n_select):
        best_idx = int(jnp.argmax(min_distances))
        selected.append(best_idx)

        # Cover the newly selected center, and take it out of contention.
        new_dists = jnp.linalg.norm(embeddings - embeddings[best_idx], axis=1)
        min_distances = jnp.minimum(min_distances, new_dists)
        min_distances = min_distances.at[best_idx].set(-jnp.inf)

    return selected


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
    n_labeled = len(labeled_graphs)

    # Embed pool and labeled molecules into one array so the greedy selection
    # can run over a single index space. Pool points come first, so returned
    # indices are already indices into pool_graphs.
    pool_batched = batch_graphs(pool_graphs)
    pool_embeddings = model.extract_embeddings(pool_batched, training=False)[:n_pool]

    if labeled_graphs:
        labeled_batched = batch_graphs(labeled_graphs)
        labeled_embeddings = model.extract_embeddings(labeled_batched, training=False)
        labeled_embeddings = labeled_embeddings[:n_labeled]
        embeddings = jnp.concatenate([pool_embeddings, labeled_embeddings], axis=0)
    else:
        embeddings = pool_embeddings

    pool_mask = jnp.arange(embeddings.shape[0]) < n_pool

    return coreset_from_embeddings(embeddings, pool_mask, n_select)


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
