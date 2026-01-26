"""Evidential Deep Learning for uncertainty quantification.

Predicts uncertainty in a single forward pass by modeling output as a
Normal-Inverse-Gamma (NIG) distribution. This enables separation of
aleatoric (data) and epistemic (model) uncertainty.

Reference: Amini et al., "Deep Evidential Regression", NeurIPS 2020
https://arxiv.org/abs/1910.02600

Note on Uncertainty Calibration:
    Evidential regression can produce increasing/unbounded uncertainty estimates
    during training. This is a known limitation where NIG parameters can drift.
    The `lambda_reg` parameter controls evidence regularization - higher values
    (0.5-1.0) may improve calibration. For active learning, relative uncertainty
    ranking matters more than absolute values, so sample selection still works.
    See examples/evidential_active_learning.py for more details.
"""

from dataclasses import dataclass
from typing import Tuple

import flax.nnx as nnx
import jax.numpy as jnp
import jax.scipy.special as jsp
import jraph
import optax

from .gcn import GCNConfig, GraphConvolution


@dataclass
class EvidentialConfig:
    """Configuration for Evidential GCN.

    Attributes:
        base_config: Configuration for the GCN backbone (GCNConfig)
        lambda_reg: Regularization weight for evidence on errors (default: 0.1)
    """

    base_config: GCNConfig
    lambda_reg: float = 0.1


class EvidentialHead(nnx.Module):
    """Predicts Normal-Inverse-Gamma parameters for evidential regression.

    Outputs (gamma, nu, alpha, beta) where:
    - gamma: mean prediction (unbounded)
    - nu > 0: precision of mean (higher = more confident)
    - alpha > 1: shape parameter
    - beta > 0: scale parameter

    The NIG distribution provides:
    - Aleatoric uncertainty: beta / (alpha - 1)
    - Epistemic uncertainty: beta / (nu * (alpha - 1))
    """

    def __init__(self, in_features: int, rngs: nnx.Rngs):
        """Initialize evidential head.

        Args:
            in_features: Input feature dimension
            rngs: Random number generators
        """
        # Output 4 parameters per output dimension
        self.linear = nnx.Linear(in_features, 4, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        """Forward pass returning NIG parameters.

        Args:
            x: Input features [batch, in_features]

        Returns:
            Tuple of (gamma, nu, alpha, beta) each of shape [batch, 1]
        """
        out = self.linear(x)

        # Split into 4 parameters
        gamma = out[..., 0:1]  # Mean (unbounded)
        nu = nnx.softplus(out[..., 1:2]) + 1e-6  # > 0
        alpha = nnx.softplus(out[..., 2:3]) + 1.0  # > 1
        beta = nnx.softplus(out[..., 3:4]) + 1e-6  # > 0

        return gamma, nu, alpha, beta


class EvidentialGCN(nnx.Module):
    """GCN with Evidential Deep Learning for uncertainty quantification.

    Uses a GCN backbone followed by an evidential head that predicts
    Normal-Inverse-Gamma parameters, enabling single-pass uncertainty
    estimation with separation of aleatoric and epistemic components.
    """

    def __init__(self, config: EvidentialConfig, rngs: nnx.Rngs):
        """Initialize EvidentialGCN.

        Args:
            config: EvidentialConfig with base model config and lambda_reg
            rngs: Random number generators for initialization
        """
        self.config = config
        self.base_config = config.base_config
        self.lambda_reg = config.lambda_reg

        # Build GCN layers
        layers = []
        in_dim = self.base_config.node_features
        for hidden_dim in self.base_config.hidden_features:
            layers.append(GraphConvolution(in_dim, hidden_dim, rngs))
            in_dim = hidden_dim

        self.conv_layers = nnx.List(layers)

        # Evidential head
        self.evidential_head = EvidentialHead(in_dim, rngs)

        # Dropout
        self.dropout = nnx.Dropout(self.base_config.dropout_rate, rngs=rngs)

    def forward_raw(
        self, graph: jraph.GraphsTuple, training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass returning raw NIG parameters.

        Args:
            graph: Batched jraph.GraphsTuple
            training: Whether in training mode

        Returns:
            Tuple of (gamma, nu, alpha, beta) - NIG parameters
            Each has shape [n_graphs, 1]
        """
        # Apply GCN layers
        for conv in self.conv_layers:
            graph = conv(graph)
            graph = graph._replace(nodes=nnx.relu(graph.nodes))
            if training:
                graph = graph._replace(nodes=self.dropout(graph.nodes))

        # Global mean pooling per graph
        n_graphs = graph.n_node.shape[0]
        graph_indices = jnp.repeat(
            jnp.arange(n_graphs),
            graph.n_node,
            total_repeat_length=graph.nodes.shape[0],
        )

        # Sum pooling
        pooled_sum = jraph.segment_sum(
            graph.nodes,
            graph_indices,
            num_segments=n_graphs,
        )

        # Normalize by number of nodes
        n_nodes_per_graph = graph.n_node.astype(jnp.float32)
        n_nodes_per_graph = jnp.maximum(n_nodes_per_graph, 1.0)[:, None]
        pooled = pooled_sum / n_nodes_per_graph

        # Evidential prediction
        return self.evidential_head(pooled)

    def __call__(
        self, graph: jraph.GraphsTuple, training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass returning mean, total uncertainty, and epistemic uncertainty.

        This signature matches DeepEnsemble for drop-in replacement.

        Args:
            graph: Batched jraph.GraphsTuple
            training: Whether in training mode

        Returns:
            Tuple of:
            - mean: Mean prediction [n_graphs, 1]
            - total_var: Total uncertainty (aleatoric + epistemic) [n_graphs, 1]
            - epistemic_var: Epistemic uncertainty [n_graphs, 1]
        """
        gamma, nu, alpha, beta = self.forward_raw(graph, training)

        # Compute uncertainties from NIG parameters
        # Aleatoric: expected variance = beta / (alpha - 1)
        # Epistemic: uncertainty in variance = beta / (nu * (alpha - 1))
        aleatoric_var = beta / (alpha - 1.0)
        epistemic_var = beta / (nu * (alpha - 1.0))
        total_var = aleatoric_var + epistemic_var

        return gamma, total_var, epistemic_var

    def extract_embeddings(
        self, graph: jraph.GraphsTuple, training: bool = False
    ) -> jnp.ndarray:
        """Extract penultimate layer embeddings (graph-level).

        Extracts the pooled graph representations before the evidential head,
        which can be used for Core-Set selection, DPP sampling, or other
        embedding-based acquisition strategies.

        Args:
            graph: Batched jraph.GraphsTuple
            training: Whether in training mode (enables dropout)

        Returns:
            Embeddings of shape [n_graphs, hidden_dim] where hidden_dim
            is the last element of hidden_features in the config.
        """
        # Apply GCN layers
        for conv in self.conv_layers:
            graph = conv(graph)
            graph = graph._replace(nodes=nnx.relu(graph.nodes))
            if training:
                graph = graph._replace(nodes=self.dropout(graph.nodes))

        # Global mean pooling per graph (stop before evidential head)
        n_graphs = graph.n_node.shape[0]
        graph_indices = jnp.repeat(
            jnp.arange(n_graphs),
            graph.n_node,
            total_repeat_length=graph.nodes.shape[0],
        )

        # Sum pooling
        pooled_sum = jraph.segment_sum(
            graph.nodes,
            graph_indices,
            num_segments=n_graphs,
        )

        # Normalize by number of nodes
        n_nodes_per_graph = graph.n_node.astype(jnp.float32)
        n_nodes_per_graph = jnp.maximum(n_nodes_per_graph, 1.0)[:, None]

        return pooled_sum / n_nodes_per_graph


def evidential_loss(
    gamma: jnp.ndarray,
    nu: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    targets: jnp.ndarray,
    lambda_reg: float = 0.1,
) -> jnp.ndarray:
    """Compute NIG negative log-likelihood with regularization.

    The loss consists of:
    1. NIG negative log-likelihood (fits the predicted distribution)
    2. Evidence regularization (penalizes high evidence on errors)

    Args:
        gamma: Predicted mean [batch]
        nu: Precision of mean [batch]
        alpha: Shape parameter [batch]
        beta: Scale parameter [batch]
        targets: Ground truth values [batch]
        lambda_reg: Weight for evidence regularization

    Returns:
        Scalar loss value
    """
    # Compute NIG NLL
    # omega = 2 * beta * (1 + nu)
    omega = 2.0 * beta * (1.0 + nu)

    # NLL terms
    nll = (
        0.5 * jnp.log(jnp.pi / nu)
        - alpha * jnp.log(omega)
        + (alpha + 0.5) * jnp.log((targets - gamma) ** 2 * nu + omega)
        + jsp.gammaln(alpha)
        - jsp.gammaln(alpha + 0.5)
    )

    # Evidence regularization: penalize high evidence on errors
    # Evidence = 2 * nu + alpha
    error = jnp.abs(targets - gamma)
    reg = lambda_reg * error * (2.0 * nu + alpha)

    return nll + reg


def _compute_evidential_loss(
    model: EvidentialGCN,
    graph: jraph.GraphsTuple,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Internal function to compute masked evidential loss.

    Args:
        model: EvidentialGCN model
        graph: Batched input graphs
        labels: Target labels [n_graphs]
        mask: Boolean mask for real graphs

    Returns:
        Scalar loss value
    """
    gamma, nu, alpha, beta = model.forward_raw(graph, training=True)

    # Squeeze to match label shape
    gamma = gamma.squeeze(-1)
    nu = nu.squeeze(-1)
    alpha = alpha.squeeze(-1)
    beta = beta.squeeze(-1)

    # Compute per-sample loss
    per_sample_loss = evidential_loss(gamma, nu, alpha, beta, labels, model.lambda_reg)

    # Apply mask and compute mean
    masked_loss = jnp.where(mask, per_sample_loss, 0.0)
    return jnp.sum(masked_loss) / (jnp.sum(mask) + 1e-6)


def create_evidential_optimizer(
    model: EvidentialGCN, learning_rate: float = 1e-3
) -> nnx.Optimizer:
    """Create optimizer for EvidentialGCN training.

    Args:
        model: The EvidentialGCN model to optimize
        learning_rate: Learning rate

    Returns:
        nnx.Optimizer instance
    """
    return nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)


@nnx.jit
def train_evidential_step(
    model: EvidentialGCN,
    optimizer: nnx.Optimizer,
    graph: jraph.GraphsTuple,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled training step for EvidentialGCN.

    Args:
        model: EvidentialGCN model
        optimizer: Optimizer for the model
        graph: Batched input graphs
        labels: Target labels [n_graphs]
        mask: Boolean mask for real graphs

    Returns:
        Loss value
    """

    def loss_fn(model: EvidentialGCN) -> jnp.ndarray:
        return _compute_evidential_loss(model, graph, labels, mask)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_evidential_step(
    model: EvidentialGCN,
    graph: jraph.GraphsTuple,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled evaluation step for EvidentialGCN.

    Args:
        model: EvidentialGCN model
        graph: Batched input graphs
        labels: Target labels [n_graphs]
        mask: Boolean mask for real graphs

    Returns:
        Tuple of (rmse, mean_epistemic_uncertainty, mean_total_uncertainty)
    """
    mean, total_var, epistemic_var = model(graph, training=False)
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


def get_evidential_uncertainties(
    model: EvidentialGCN,
    graph: jraph.GraphsTuple,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get uncertainty estimates from EvidentialGCN for acquisition.

    Unlike MC Dropout, evidential models provide uncertainty in a single
    forward pass by predicting the parameters of a higher-order distribution.

    Args:
        model: EvidentialGCN model
        graph: Batched input graphs

    Returns:
        Tuple of (epistemic_uncertainty, total_uncertainty) per graph
    """
    _, total_var, epistemic_var = model(graph, training=False)
    return epistemic_var.squeeze(-1), total_var.squeeze(-1)
