"""Deep Ensemble models for improved uncertainty quantification.

Deep Ensembles train N independent models with different random initializations.
Uncertainty is estimated via prediction disagreement (epistemic) plus
average predicted variance (aleatoric).

Reference: Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty
Estimation using Deep Ensembles", NeurIPS 2017.
"""

from dataclasses import dataclass
from typing import List, Tuple

import flax.nnx as nnx
import jax.numpy as jnp
import jraph
import optax

from .gcn import GCNConfig, UncertaintyGCN


@dataclass
class EnsembleConfig:
    """Configuration for Deep Ensemble.

    Attributes:
        base_config: Configuration for each ensemble member (GCNConfig)
        n_members: Number of ensemble members (default: 5)
    """

    base_config: GCNConfig
    n_members: int = 5


class DeepEnsemble(nnx.Module):
    """Deep Ensemble of UncertaintyGCN models.

    Trains N independent GCN models with different random initializations.
    Provides improved uncertainty estimation by decomposing into:
    - Epistemic uncertainty: disagreement between models (reducible with more data)
    - Aleatoric uncertainty: average predicted variance (inherent noise)
    """

    def __init__(self, config: EnsembleConfig, rngs: nnx.Rngs):
        """Initialize ensemble with N independent models.

        Args:
            config: EnsembleConfig with base model config and n_members
            rngs: Random number generators for initialization
        """
        self.config = config
        self.n_members = config.n_members

        # Create N independent models with different random seeds
        # Each member gets a unique seed for independent initialization
        self.members = nnx.List(
            [
                UncertaintyGCN(config.base_config, rngs=nnx.Rngs(i))
                for i in range(config.n_members)
            ]
        )

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        training: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass returning mean, total uncertainty, and epistemic uncertainty.

        Args:
            graph: Batched jraph.GraphsTuple
            training: Whether in training mode

        Returns:
            Tuple of:
            - ensemble_mean: Mean prediction across all members
            - total_var: Total uncertainty (epistemic + aleatoric)
            - epistemic_var: Epistemic uncertainty (model disagreement)
            Each has shape [n_graphs, out_features].
        """
        # Collect predictions from all members
        means = []
        variances = []

        for member in self.members:
            mean, var = member(graph, training=training)
            means.append(mean)
            variances.append(var)

        # Stack: [n_members, n_graphs, out_features]
        means = jnp.stack(means, axis=0)
        variances = jnp.stack(variances, axis=0)

        # Ensemble mean prediction
        ensemble_mean = jnp.mean(means, axis=0)

        # Epistemic uncertainty: variance of means (model disagreement)
        epistemic_var = jnp.var(means, axis=0)

        # Aleatoric uncertainty: mean of variances (average predicted noise)
        aleatoric_var = jnp.mean(variances, axis=0)

        # Total uncertainty = epistemic + aleatoric
        total_var = epistemic_var + aleatoric_var

        return ensemble_mean, total_var, epistemic_var

    def predict_member(
        self,
        member_idx: int,
        graph: jraph.GraphsTuple,
        training: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get prediction from a specific ensemble member.

        Args:
            member_idx: Index of the ensemble member (0 to n_members-1)
            graph: Batched jraph.GraphsTuple
            training: Whether in training mode

        Returns:
            Tuple of (mean, variance) from the specified member
        """
        mean, var = self.members[member_idx](graph, training=training)
        return mean, var

    def extract_embeddings(
        self, graph: jraph.GraphsTuple, training: bool = False
    ) -> jnp.ndarray:
        """Extract averaged embeddings from all ensemble members.

        Each member extracts embeddings independently, and the results are
        averaged to produce a single embedding per graph. This can be used
        for Core-Set selection, DPP sampling, or other embedding-based
        acquisition strategies.

        Args:
            graph: Batched jraph.GraphsTuple
            training: Whether in training mode

        Returns:
            Averaged embeddings of shape [n_graphs, hidden_dim]
        """
        # Collect embeddings from all members
        all_embeddings = []
        for member in self.members:
            embeddings = member.extract_embeddings(graph, training=training)
            all_embeddings.append(embeddings)

        # Stack: [n_members, n_graphs, hidden_dim]
        all_embeddings = jnp.stack(all_embeddings, axis=0)

        # Average across members
        return jnp.mean(all_embeddings, axis=0)


def create_ensemble_optimizers(
    ensemble: DeepEnsemble,
    learning_rate: float = 1e-3,
) -> List[nnx.Optimizer]:
    """Create separate optimizers for each ensemble member.

    Args:
        ensemble: DeepEnsemble model
        learning_rate: Learning rate for Adam optimizer

    Returns:
        List of nnx.Optimizer, one per ensemble member
    """
    return [
        nnx.Optimizer(member, optax.adam(learning_rate), wrt=nnx.Param)
        for member in ensemble.members
    ]


def train_ensemble_step(
    ensemble: DeepEnsemble,
    optimizers: List[nnx.Optimizer],
    graph: jraph.GraphsTuple,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Train all ensemble members on the same data.

    Each member is trained independently with its own optimizer.

    Args:
        ensemble: DeepEnsemble model
        optimizers: List of optimizers (one per member)
        graph: Batched input graphs
        labels: Target labels [n_graphs]
        mask: Boolean mask for real graphs

    Returns:
        Mean loss across all ensemble members
    """
    losses = []

    for member, optimizer in zip(ensemble.members, optimizers):
        loss = _train_member_step(member, optimizer, graph, labels, mask)
        losses.append(loss)

    return jnp.mean(jnp.array(losses))


@nnx.jit
def _train_member_step(
    member: UncertaintyGCN,
    optimizer: nnx.Optimizer,
    graph: jraph.GraphsTuple,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled training step for a single ensemble member.

    Args:
        member: Single UncertaintyGCN model
        optimizer: Optimizer for this member
        graph: Batched input graphs
        labels: Target labels [n_graphs]
        mask: Boolean mask for real graphs

    Returns:
        Loss value
    """

    def loss_fn(model: UncertaintyGCN) -> jnp.ndarray:
        mean, var = model(graph, training=True)
        mean = mean.squeeze(-1)
        var = var.squeeze(-1)

        # Negative log-likelihood loss
        nll = 0.5 * (jnp.log(var + 1e-6) + (labels - mean) ** 2 / (var + 1e-6))
        masked_nll = jnp.where(mask, nll, 0.0)
        return jnp.sum(masked_nll) / (jnp.sum(mask) + 1e-6)

    loss, grads = nnx.value_and_grad(loss_fn)(member)
    optimizer.update(member, grads)
    return loss


@nnx.jit
def eval_ensemble_step(
    ensemble: DeepEnsemble,
    graph: jraph.GraphsTuple,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled evaluation step for ensemble.

    Args:
        ensemble: DeepEnsemble model
        graph: Batched input graphs
        labels: Target labels [n_graphs]
        mask: Boolean mask for real graphs

    Returns:
        Tuple of (rmse, mean_epistemic_uncertainty, mean_total_uncertainty)
    """
    mean, total_var, epistemic_var = ensemble(graph, training=False)
    mean = mean.squeeze(-1)
    total_var = total_var.squeeze(-1)
    epistemic_var = epistemic_var.squeeze(-1)

    # Compute RMSE only for real graphs
    se = (mean - labels) ** 2
    masked_se = jnp.where(mask, se, 0.0)
    mse = jnp.sum(masked_se) / jnp.sum(mask)
    rmse = jnp.sqrt(mse)

    # Mean uncertainties over real graphs
    mean_epistemic = jnp.sum(jnp.where(mask, epistemic_var, 0.0)) / jnp.sum(mask)
    mean_total = jnp.sum(jnp.where(mask, total_var, 0.0)) / jnp.sum(mask)

    return rmse, mean_epistemic, mean_total


def get_ensemble_uncertainties(
    ensemble: DeepEnsemble,
    graph: jraph.GraphsTuple,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get uncertainty estimates from ensemble for acquisition.

    Unlike MC Dropout, ensembles provide uncertainty in a single forward pass.

    Args:
        ensemble: DeepEnsemble model
        graph: Batched input graphs

    Returns:
        Tuple of (epistemic_uncertainty, total_uncertainty) per graph
    """
    _, total_var, epistemic_var = ensemble(graph, training=False)
    return epistemic_var.squeeze(-1), total_var.squeeze(-1)
