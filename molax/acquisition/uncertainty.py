import jax
import jax.numpy as jnp
from typing import Callable, List, Tuple

def uncertainty_sampling(
    model: Callable,
    params: dict,
    pool_data: List[Tuple[jnp.ndarray, jnp.ndarray]],
    n_samples: int = 10
) -> jnp.ndarray:
    """
    Select samples based on predictive uncertainty using MC dropout
    
    Args:
        model: UncertaintyGCN model
        params: Model parameters
        pool_data: List of (features, adjacency) tuples for unlabeled pool
        n_samples: Number of MC samples for uncertainty estimation
    
    Returns:
        Array of uncertainty scores for each molecule in the pool
    """
    uncertainties = []
    
    for x, adj in pool_data:
        # Collect MC samples
        predictions = []
        for _ in range(n_samples):
            rng = jax.random.PRNGKey(0)  # You'd want to properly manage seeds
            mean, var = model.apply(
                params, 
                x, 
                adj, 
                training=True, 
                rngs={'dropout': rng}
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
    n_select: int
) -> List[int]:
    """
    Select diverse samples using maximum distance criterion
    
    Args:
        pool_data: List of (features, adjacency) tuples for unlabeled pool
        labeled_data: List of (features, adjacency) tuples for labeled set
        n_select: Number of samples to select
    
    Returns:
        Indices of selected samples
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
                min_dist = float('inf')
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
                    distances.append(-float('inf'))
                    continue
                    
                min_dist = float('inf')
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
    uncertainty_weight: float = 0.7
) -> List[int]:
    """
    Combine uncertainty and diversity for sample selection
    
    Args:
        model: UncertaintyGCN model
        params: Model parameters
        pool_data: List of (features, adjacency) tuples for unlabeled pool
        labeled_data: List of (features, adjacency) tuples for labeled set
        n_select: Number of samples to select
        uncertainty_weight: Weight for uncertainty score (1-weight for diversity)
    
    Returns:
        Indices of selected samples
    """
    # Get uncertainty scores
    uncertainties = uncertainty_sampling(model, params, pool_data)
    
    # Get diversity scores
    diversity_indices = diversity_sampling(pool_data, labeled_data, n_select)
    diversity_scores = jnp.zeros(len(pool_data))
    diversity_scores = diversity_scores.at[diversity_indices].set(1.0)
    
    # Combine scores
    combined_scores = (
        uncertainty_weight * uncertainties / jnp.max(uncertainties) +
        (1 - uncertainty_weight) * diversity_scores
    )
    
    # Select top scoring samples
    return jnp.argsort(combined_scores)[-n_select:]