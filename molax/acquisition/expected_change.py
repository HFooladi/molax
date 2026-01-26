"""Expected Model Change acquisition functions.

Expected Gradient Length (EGL) selects samples that would cause the largest
change in model parameters if labeled. This is approximated by computing the
gradient of the loss with respect to model parameters for each sample, using
the model's prediction as a pseudo-label.

Reference: Settles et al., "Multiple-Instance Active Learning", ICML 2008
"""

from typing import List

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jraph

from molax.models.gcn import UncertaintyGCN
from molax.utils.data import batch_graphs


def egl_sampling(
    model: UncertaintyGCN,
    pool_graphs: List[jraph.GraphsTuple],
) -> jnp.ndarray:
    """Compute Expected Gradient Length scores for pool samples.

    EGL measures the expected change in model parameters if a sample were
    labeled. Samples with high EGL are those where the model would learn
    the most. This is approximated by:

    1. Using the model's prediction as a pseudo-label
    2. Computing the gradient of NLL loss with respect to parameters
    3. Using the L2 norm of the gradient as the EGL score

    This implementation computes gradients for each sample independently.

    Args:
        model: UncertaintyGCN model
        pool_graphs: List of jraph.GraphsTuple for pool samples

    Returns:
        Array of EGL scores for each pool sample
    """
    if not pool_graphs:
        return jnp.array([])

    n_pool = len(pool_graphs)

    # Batch all pool graphs for prediction
    batched = batch_graphs(pool_graphs)

    # Get predictions (use as pseudo-labels)
    mean_pred, _ = model(batched, training=False)
    pseudo_labels = mean_pred.squeeze(-1)[:n_pool]

    # Compute EGL scores for each sample
    egl_scores = []

    for i in range(n_pool):
        # Create a single-sample batch for this graph
        single_graph = pool_graphs[i]
        single_batched = batch_graphs([single_graph])
        single_label = pseudo_labels[i]

        # Define loss function that takes model as input
        def loss_fn(m: UncertaintyGCN) -> jnp.ndarray:
            mean, var = m(single_batched, training=False)
            mean = mean.squeeze(-1)[0]
            var = var.squeeze(-1)[0]

            # NLL loss
            log_term = jnp.log(var + 1e-6)
            sq_term = (single_label - mean) ** 2 / (var + 1e-6)
            return 0.5 * (log_term + sq_term)

        # Get gradients using nnx.grad
        _, grads = nnx.value_and_grad(loss_fn)(model)

        # Compute L2 norm of all gradients
        grad_norm_sq = 0.0
        for leaf in jax.tree_util.tree_leaves(grads):
            if leaf is not None and hasattr(leaf, "dtype"):
                # Only process numeric arrays
                if jnp.issubdtype(leaf.dtype, jnp.floating):
                    grad_norm_sq = grad_norm_sq + jnp.sum(leaf**2)

        egl_scores.append(jnp.sqrt(grad_norm_sq))

    return jnp.array(egl_scores)


def egl_sampling_batched(
    model: UncertaintyGCN,
    pool_graphs: List[jraph.GraphsTuple],
    batch_size: int = 32,
) -> jnp.ndarray:
    """Compute EGL scores with batched gradient computation for efficiency.

    This version processes samples in batches to improve computational
    efficiency, though it uses an approximation by computing average
    gradients over mini-batches.

    Args:
        model: UncertaintyGCN model
        pool_graphs: List of jraph.GraphsTuple for pool samples
        batch_size: Number of samples to process at once

    Returns:
        Array of EGL scores for each pool sample
    """
    if not pool_graphs:
        return jnp.array([])

    n_pool = len(pool_graphs)

    # Batch all pool graphs for prediction
    all_batched = batch_graphs(pool_graphs)

    # Get predictions (use as pseudo-labels)
    mean_pred, _ = model(all_batched, training=False)
    pseudo_labels = mean_pred.squeeze(-1)[:n_pool]

    egl_scores = []

    # Process in batches
    for start in range(0, n_pool, batch_size):
        end = min(start + batch_size, n_pool)
        batch_graphs_list = pool_graphs[start:end]
        batch_labels = pseudo_labels[start:end]
        batch_size_actual = end - start

        # Batch the graphs
        batched = batch_graphs(batch_graphs_list)

        def batch_loss_fn(m: UncertaintyGCN) -> jnp.ndarray:
            mean, var = m(batched, training=False)
            mean = mean.squeeze(-1)
            var = var.squeeze(-1)

            # NLL loss for each sample
            nll = 0.5 * (
                jnp.log(var[:batch_size_actual] + 1e-6)
                + (batch_labels - mean[:batch_size_actual]) ** 2
                / (var[:batch_size_actual] + 1e-6)
            )
            return jnp.mean(nll)

        # Compute gradients using nnx
        _, grads = nnx.value_and_grad(batch_loss_fn)(model)

        # Compute gradient norm (approximation for batch)
        grad_norm_sq = 0.0
        for leaf in jax.tree_util.tree_leaves(grads):
            if leaf is not None and hasattr(leaf, "dtype"):
                if jnp.issubdtype(leaf.dtype, jnp.floating):
                    grad_norm_sq = grad_norm_sq + jnp.sum(leaf**2)

        # Distribute score evenly among batch samples (approximation)
        batch_score = jnp.sqrt(grad_norm_sq) / batch_size_actual
        egl_scores.extend([float(batch_score)] * batch_size_actual)

    return jnp.array(egl_scores)


def egl_acquisition(
    model: UncertaintyGCN,
    pool_graphs: List[jraph.GraphsTuple],
    n_select: int,
    use_batched: bool = False,
    batch_size: int = 32,
) -> List[int]:
    """Select samples with highest Expected Gradient Length.

    Args:
        model: UncertaintyGCN model
        pool_graphs: List of jraph.GraphsTuple for pool samples
        n_select: Number of samples to select
        use_batched: Whether to use batched computation (faster but approximate)
        batch_size: Batch size for batched computation

    Returns:
        List of selected indices into pool_graphs
    """
    if not pool_graphs:
        return []

    n_select = min(n_select, len(pool_graphs))

    # Compute EGL scores
    if use_batched:
        scores = egl_sampling_batched(model, pool_graphs, batch_size)
    else:
        scores = egl_sampling(model, pool_graphs)

    # Select top scoring samples
    top_indices = jnp.argsort(-scores)[:n_select]

    return [int(i) for i in top_indices]
