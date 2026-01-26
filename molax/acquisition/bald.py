"""BALD (Bayesian Active Learning by Disagreement) acquisition functions.

BALD maximizes the mutual information between predictions and model parameters:
    BALD(x) = I(y; θ | x, D) = H[y|x,D] - E_θ[H[y|x,θ]]

For Gaussian outputs, this simplifies to:
    BALD ≈ 0.5 * log(total_var) - 0.5 * log(expected_aleatoric_var)
         = 0.5 * log(total_var / expected_aleatoric_var)

where total_var = epistemic_var + expected_aleatoric_var.

Reference: Houlsby et al., "Bayesian Active Learning for Classification and
Preference Learning", 2011.
"""

from typing import List

import jax.numpy as jnp
import jraph

from molax.models.ensemble import DeepEnsemble
from molax.models.evidential import EvidentialGCN
from molax.models.gcn import UncertaintyGCN
from molax.utils.data import batch_graphs


def bald_sampling(
    model: UncertaintyGCN,
    pool_graphs: List[jraph.GraphsTuple],
    n_mc_samples: int = 10,
) -> jnp.ndarray:
    """Compute BALD scores for pool samples using MC Dropout.

    BALD = H[y|x,D] - E_θ[H[y|x,θ]]
         ≈ 0.5 * log(total_var / expected_aleatoric_var)

    Higher scores indicate samples where the model is uncertain about the
    prediction but confident in its uncertainty estimates (high epistemic
    uncertainty relative to aleatoric).

    Args:
        model: UncertaintyGCN model with MC Dropout
        pool_graphs: List of jraph.GraphsTuple for pool samples
        n_mc_samples: Number of MC dropout samples

    Returns:
        Array of BALD scores for each pool sample
    """
    if not pool_graphs:
        return jnp.array([])

    n_pool = len(pool_graphs)

    # Batch all pool graphs for efficient processing
    batched = batch_graphs(pool_graphs)

    # Collect MC samples of means and variances
    means = []
    variances = []

    for _ in range(n_mc_samples):
        mean, var = model(batched, training=True)
        means.append(mean.squeeze(-1))
        variances.append(var.squeeze(-1))

    # Stack: [n_samples, n_graphs_with_padding]
    means = jnp.stack(means, axis=0)
    variances = jnp.stack(variances, axis=0)

    # Epistemic uncertainty: variance of means across MC samples
    epistemic_var = jnp.var(means, axis=0)

    # Expected aleatoric uncertainty: mean of variances across MC samples
    expected_aleatoric_var = jnp.mean(variances, axis=0)

    # Total variance (law of total variance)
    total_var = epistemic_var + expected_aleatoric_var

    # BALD score: 0.5 * log(total_var / expected_aleatoric_var)
    # Add small epsilon for numerical stability
    eps = 1e-8
    bald_scores = 0.5 * jnp.log((total_var + eps) / (expected_aleatoric_var + eps))

    # Return only scores for actual graphs (exclude padding graph)
    return bald_scores[:n_pool]


def ensemble_bald_sampling(
    ensemble: DeepEnsemble,
    pool_graphs: List[jraph.GraphsTuple],
) -> jnp.ndarray:
    """Compute BALD scores for pool samples using ensemble disagreement.

    Unlike MC Dropout, ensembles provide uncertainty in a single forward pass
    by measuring disagreement between independently trained models.

    BALD = 0.5 * log(total_var / expected_aleatoric_var)

    where:
    - total_var = epistemic_var + aleatoric_var
    - epistemic_var = variance of ensemble means (model disagreement)
    - aleatoric_var = mean of ensemble variances (average predicted noise)

    Args:
        ensemble: DeepEnsemble model
        pool_graphs: List of jraph.GraphsTuple for pool samples

    Returns:
        Array of BALD scores for each pool sample
    """
    if not pool_graphs:
        return jnp.array([])

    n_pool = len(pool_graphs)

    # Batch all pool graphs for efficient processing
    batched = batch_graphs(pool_graphs)

    # Collect predictions from all members
    means = []
    variances = []

    for member in ensemble.members:
        mean, var = member(batched, training=False)
        means.append(mean.squeeze(-1))
        variances.append(var.squeeze(-1))

    # Stack: [n_members, n_graphs_with_padding]
    means = jnp.stack(means, axis=0)
    variances = jnp.stack(variances, axis=0)

    # Epistemic uncertainty: variance of ensemble means
    epistemic_var = jnp.var(means, axis=0)

    # Expected aleatoric uncertainty: mean of ensemble variances
    expected_aleatoric_var = jnp.mean(variances, axis=0)

    # Total variance
    total_var = epistemic_var + expected_aleatoric_var

    # BALD score
    eps = 1e-8
    bald_scores = 0.5 * jnp.log((total_var + eps) / (expected_aleatoric_var + eps))

    # Return only scores for actual graphs (exclude padding graph)
    return bald_scores[:n_pool]


def evidential_bald_sampling(
    model: EvidentialGCN,
    pool_graphs: List[jraph.GraphsTuple],
) -> jnp.ndarray:
    """Compute BALD scores from evidential model's NIG parameters.

    Evidential models provide epistemic and aleatoric uncertainty directly
    from the Normal-Inverse-Gamma distribution parameters:
    - Aleatoric: beta / (alpha - 1)
    - Epistemic: beta / (nu * (alpha - 1))

    BALD = 0.5 * log(total_var / aleatoric_var)

    Args:
        model: EvidentialGCN model
        pool_graphs: List of jraph.GraphsTuple for pool samples

    Returns:
        Array of BALD scores for each pool sample
    """
    if not pool_graphs:
        return jnp.array([])

    n_pool = len(pool_graphs)

    # Batch all pool graphs for efficient processing
    batched = batch_graphs(pool_graphs)

    # Get raw NIG parameters
    gamma, nu, alpha, beta = model.forward_raw(batched, training=False)

    # Compute uncertainties from NIG parameters
    # Aleatoric: expected variance = beta / (alpha - 1)
    # Epistemic: uncertainty in variance = beta / (nu * (alpha - 1))
    aleatoric_var = beta / (alpha - 1.0)
    epistemic_var = beta / (nu * (alpha - 1.0))
    total_var = aleatoric_var + epistemic_var

    # Squeeze to [n_graphs]
    aleatoric_var = aleatoric_var.squeeze(-1)
    total_var = total_var.squeeze(-1)

    # BALD score
    eps = 1e-8
    bald_scores = 0.5 * jnp.log((total_var + eps) / (aleatoric_var + eps))

    # Return only scores for actual graphs (exclude padding graph)
    return bald_scores[:n_pool]
