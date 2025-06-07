from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp


def uncertainty_sampling(
    model: Callable,
    params: dict,
    pool_data: List[Tuple[jnp.ndarray, jnp.ndarray]],
    n_samples: int = 10,
) -> jnp.ndarray:
    """
    Select samples based on predictive uncertainty using Monte Carlo (MC) dropout.
    This function estimates the uncertainty of predictions for each molecule in the pool
    by performing multiple forward passes with dropout enabled.

    Args:
        model: UncertaintyGCN model that supports MC dropout during inference
        params: Dictionary containing the model parameters
        pool_data: List of tuples containing (features, adjacency) matrices for each
                  molecule in the unlabeled pool. Features should be of shape
                  (n_nodes, n_features) and adjacency matrices should be of shape
                  (n_nodes, n_nodes)
        n_samples: Number of MC samples to use for uncertainty estimation. Higher
                  values provide more accurate uncertainty estimates but increase
                  computation time

    Returns:
        jnp.ndarray: Array of shape (n_pool,) containing uncertainty scores for each
                    molecule in the pool. Higher scores indicate higher uncertainty.
    """
    uncertainties = []

    for x, adj in pool_data:
        # Collect MC samples
        predictions = []
        for _ in range(n_samples):
            rng = jax.random.PRNGKey(0)  # You'd want to properly manage seeds
            mean, var = model.apply(
                params, x, adj, training=True, rngs={"dropout": rng}
            )
            predictions.append(mean)

        # Calculate predictive uncertainty
        predictions = jnp.stack(predictions)
        uncertainty = jnp.mean(jnp.var(predictions, axis=0))
        uncertainties.append(uncertainty)

    return jnp.array(uncertainties)


def diversity_sampling(
    pool_data: List[Tuple[jnp.ndarray, jnp.ndarray]],
    labeled_data: List[Tuple[jnp.ndarray, jnp.ndarray]],
    n_select: int,
) -> List[int]:
    """
    Select diverse samples from the unlabeled pool using a maximum distance criterion.
    This function implements a greedy algorithm to select molecules that are maximally
    different from both the labeled set and previously selected molecules.

    Args:
        pool_data: List of tuples containing (features, adjacency) matrices for each
                  molecule in the unlabeled pool
        labeled_data: List of tuples containing (features, adjacency) matrices for each
                     molecule in the labeled set
        n_select: Number of samples to select from the pool

    Returns:
        List[int]: List of indices corresponding to the selected molecules in the
                  pool_data list. The length of the returned list will be n_select.
    """
    # Extract mean node features as molecular fingerprints
    pool_fps = [jnp.mean(x, axis=0) for x, _ in pool_data]
    labeled_fps = [jnp.mean(x, axis=0) for x, _ in labeled_data]

    selected = []

    # Greedily select most distant points
    for _ in range(n_select):
        if not selected:
            # Select point furthest from labeled set
            distances = []
            for i, pool_fp in enumerate(pool_fps):
                min_dist = float("inf")
                for labeled_fp in labeled_fps:
                    dist = jnp.linalg.norm(pool_fp - labeled_fp)
                    min_dist = min(min_dist, dist)
                distances.append(min_dist)
            selected.append(jnp.argmax(jnp.array(distances)))
        else:
            # Select point furthest from both labeled and selected points
            distances = []
            for i, pool_fp in enumerate(pool_fps):
                if i in selected:
                    distances.append(-float("inf"))
                    continue

                min_dist = float("inf")
                # Distance to labeled set
                for labeled_fp in labeled_fps:
                    dist = jnp.linalg.norm(pool_fp - labeled_fp)
                    min_dist = min(min_dist, dist)
                # Distance to selected set
                for j in selected:
                    dist = jnp.linalg.norm(pool_fp - pool_fps[j])
                    min_dist = min(min_dist, dist)
                distances.append(min_dist)
            selected.append(jnp.argmax(jnp.array(distances)))

    return selected


def combined_acquisition(
    model: Callable,
    params: dict,
    pool_data: List[Tuple[jnp.ndarray, jnp.ndarray]],
    labeled_data: List[Tuple[jnp.ndarray, jnp.ndarray]],
    n_select: int,
    uncertainty_weight: float = 0.7,
) -> List[int]:
    """
    Combine uncertainty and diversity sampling strategies for active learning.
    This function implements a hybrid acquisition strategy that balances exploration
    (diversity) and exploitation (uncertainty) when selecting new samples.

    Args:
        model: UncertaintyGCN model that supports MC dropout during inference
        params: Dictionary containing the model parameters
        pool_data: List of tuples containing (features, adjacency) matrices for each
                  molecule in the unlabeled pool
        labeled_data: List of tuples containing (features, adjacency) matrices for each
                     molecule in the labeled set
        n_select: Number of samples to select from the pool
        uncertainty_weight: Float between 0 and 1 indicating the relative importance of
                           uncertainty vs diversity. A value of 1.0 means pure
                           uncertainty sampling, while 0.0 means pure diversity
                           sampling.

    Returns:
        List[int]: List of indices corresponding to the selected molecules in the
                  pool_data list. The length of the returned list will be n_select.
    """
    # Get uncertainty scores
    uncertainties = uncertainty_sampling(model, params, pool_data)

    # Get diversity scores
    diversity_indices = diversity_sampling(pool_data, labeled_data, n_select)
    diversity_scores = jnp.zeros(len(pool_data))
    diversity_scores = diversity_scores.at[diversity_indices].set(1.0)

    # Combine scores
    combined_scores = (
        uncertainty_weight * uncertainties / jnp.max(uncertainties)
        + (1 - uncertainty_weight) * diversity_scores
    )

    # Select top scoring samples
    return jnp.argsort(combined_scores)[-n_select:]
