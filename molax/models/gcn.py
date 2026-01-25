"""Graph Convolutional Networks using jraph for efficient batched processing."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import flax.nnx as nnx
import jax.numpy as jnp
import jraph


@dataclass
class GCNConfig:
    """Configuration for Graph Convolutional Network.

    Attributes:
        node_features: Input node feature dimension
        hidden_features: List of hidden layer dimensions
        out_features: Output dimension
        dropout_rate: Dropout rate for regularization
    """

    node_features: int
    hidden_features: Sequence[int]
    out_features: int
    dropout_rate: float = 0.1


class GraphConvolution(nnx.Module):
    """Single graph convolution layer using message passing.

    Implements: h_i' = σ(W * aggregate(h_j for j in neighbors(i) ∪ {i}))
    """

    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Apply graph convolution.

        Args:
            graph: Input graph with node features

        Returns:
            Graph with updated node features
        """
        nodes = graph.nodes

        # Message passing: aggregate neighbor features
        # For each edge (sender -> receiver), send the sender's features
        messages = nodes[graph.senders]

        # Aggregate messages at each receiver node (sum aggregation)
        aggregated = jraph.segment_sum(
            messages,
            graph.receivers,
            num_segments=nodes.shape[0],
        )

        # Compute degree for normalization (number of incoming edges per node)
        ones = jnp.ones((graph.senders.shape[0],))
        degree = jraph.segment_sum(ones, graph.receivers, num_segments=nodes.shape[0])
        degree = jnp.maximum(degree, 1.0)  # Avoid division by zero

        # Normalize by degree
        aggregated = aggregated / degree[:, None]

        # Linear transformation
        new_nodes = self.linear(aggregated)

        return graph._replace(nodes=new_nodes)


class MolecularGCN(nnx.Module):
    """Graph Convolutional Network for molecular property prediction.

    Uses jraph for efficient batched processing of variable-sized graphs.
    """

    def __init__(self, config: GCNConfig, rngs: nnx.Rngs):
        self.config = config
        self.rngs = rngs

        # Build GCN layers
        layers = []
        in_dim = config.node_features
        for hidden_dim in config.hidden_features:
            layers.append(GraphConvolution(in_dim, hidden_dim, rngs))
            in_dim = hidden_dim

        # Store as nnx.List for proper parameter tracking
        self.conv_layers = nnx.List(layers)

        # Output projection
        self.output_linear = nnx.Linear(in_dim, config.out_features, rngs=rngs)

        # Dropout
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    def __call__(self, graph: jraph.GraphsTuple, training: bool = False) -> jnp.ndarray:
        """Forward pass through the GCN.

        Args:
            graph: Batched (and possibly padded) jraph.GraphsTuple
            training: Whether in training mode (enables dropout)

        Returns:
            Graph-level predictions of shape [n_graphs, out_features]
        """
        # Apply GCN layers
        for conv in self.conv_layers:
            graph = conv(graph)
            graph = graph._replace(nodes=nnx.relu(graph.nodes))
            if training:
                graph = graph._replace(nodes=self.dropout(graph.nodes))

        # Global mean pooling per graph using jraph utilities
        n_graphs = graph.n_node.shape[0]

        # Create graph indices for each node
        graph_indices = jnp.repeat(
            jnp.arange(n_graphs),
            graph.n_node,
            total_repeat_length=graph.nodes.shape[0],
        )

        # Sum pooling (more stable than mean for padded graphs)
        pooled_sum = jraph.segment_sum(
            graph.nodes,
            graph_indices,
            num_segments=n_graphs,
        )

        # Normalize by number of nodes per graph (avoiding division by zero)
        n_nodes_per_graph = graph.n_node.astype(jnp.float32)
        n_nodes_per_graph = jnp.maximum(n_nodes_per_graph, 1.0)[:, None]
        pooled = pooled_sum / n_nodes_per_graph

        # Output projection
        return self.output_linear(pooled)


class UncertaintyGCN(nnx.Module):
    """GCN with uncertainty estimation via mean and variance heads.

    Outputs both mean prediction and predicted variance for uncertainty quantification.
    """

    def __init__(self, config: GCNConfig, rngs: nnx.Rngs):
        self.config = config
        self.rngs = rngs

        # Build GCN layers
        layers = []
        in_dim = config.node_features
        for hidden_dim in config.hidden_features:
            layers.append(GraphConvolution(in_dim, hidden_dim, rngs))
            in_dim = hidden_dim

        self.conv_layers = nnx.List(layers)

        # Separate heads for mean and variance
        self.mean_head = nnx.Linear(in_dim, config.out_features, rngs=rngs)
        self.var_head = nnx.Linear(in_dim, config.out_features, rngs=rngs)

        # Dropout
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    def __call__(
        self, graph: jraph.GraphsTuple, training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass returning mean and variance.

        Args:
            graph: Batched (and possibly padded) jraph.GraphsTuple
            training: Whether in training mode

        Returns:
            Tuple of (mean, variance) each of shape [n_graphs, out_features]
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

        # Normalize by number of nodes (avoiding division by zero)
        n_nodes_per_graph = graph.n_node.astype(jnp.float32)
        n_nodes_per_graph = jnp.maximum(n_nodes_per_graph, 1.0)[:, None]
        pooled = pooled_sum / n_nodes_per_graph

        # Predict mean and log-variance
        mean = self.mean_head(pooled)
        log_var = self.var_head(pooled)

        # Return mean and variance (exp for positivity)
        return mean, jnp.exp(log_var)


def create_train_state(
    model: UncertaintyGCN, learning_rate: float = 1e-3
) -> nnx.Optimizer:
    """Create optimizer for training.

    Args:
        model: The model to optimize
        learning_rate: Learning rate

    Returns:
        nnx.Optimizer instance
    """
    import optax

    return nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)


@nnx.jit
def train_step(
    model: UncertaintyGCN,
    optimizer: nnx.Optimizer,
    graph: jraph.GraphsTuple,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled training step.

    Args:
        model: The model
        optimizer: The optimizer
        graph: Batched (padded) input graphs
        labels: Target labels of shape [n_graphs] (padded with zeros)
        mask: Boolean mask indicating real graphs (not padding)

    Returns:
        Loss value
    """

    def loss_fn(model: UncertaintyGCN) -> jnp.ndarray:
        mean, var = model(graph, training=True)
        mean = mean.squeeze(-1)
        var = var.squeeze(-1)

        # Compute NLL only for real graphs (masked)
        nll = 0.5 * (jnp.log(var + 1e-6) + (labels - mean) ** 2 / (var + 1e-6))
        # Apply mask and compute mean over real graphs only
        masked_nll = jnp.where(mask, nll, 0.0)
        return jnp.sum(masked_nll) / jnp.sum(mask)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(
    model: UncertaintyGCN,
    graph: jraph.GraphsTuple,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled evaluation step.

    Args:
        model: The model
        graph: Batched (padded) input graphs
        labels: Target labels (padded)
        mask: Boolean mask for real graphs

    Returns:
        Tuple of (mse, mean_predictions)
    """
    mean, _ = model(graph, training=False)
    mean = mean.squeeze(-1)

    # Compute MSE only for real graphs
    se = (mean - labels) ** 2
    masked_se = jnp.where(mask, se, 0.0)
    mse = jnp.sum(masked_se) / jnp.sum(mask)
    return mse, mean


@nnx.jit
def predict_with_uncertainty(
    model: UncertaintyGCN,
    graph: jraph.GraphsTuple,
    n_samples: int = 10,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get predictions with MC dropout uncertainty.

    Args:
        model: The model
        graph: Batched input graphs
        n_samples: Number of MC samples

    Returns:
        Tuple of (mean_prediction, epistemic_uncertainty)
    """
    predictions = []
    for _ in range(n_samples):
        mean, _ = model(graph, training=True)  # Enable dropout
        predictions.append(mean)

    predictions = jnp.stack(predictions, axis=0)
    mean_pred = jnp.mean(predictions, axis=0)
    uncertainty = jnp.var(predictions, axis=0)

    return mean_pred, uncertainty
