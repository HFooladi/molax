"""Message Passing Neural Networks using jraph for efficient batched processing.

MPNN extends GCN by incorporating edge features in the message computation,
enabling the model to leverage bond information for molecular property prediction.
"""

from dataclasses import dataclass
from typing import Literal, Sequence, Tuple

import flax.nnx as nnx
import jax.numpy as jnp
import jraph


@dataclass
class MPNNConfig:
    """Configuration for Message Passing Neural Network.

    Attributes:
        node_features: Input node feature dimension (default: 6 for atom features)
        edge_features: Input edge feature dimension (default: 1 for bond type)
        hidden_features: List of hidden layer dimensions
        out_features: Output dimension
        aggregation: Aggregation method ('sum', 'mean', or 'max')
        dropout_rate: Dropout rate for regularization
    """

    node_features: int = 6
    edge_features: int = 1
    hidden_features: Sequence[int] = (64, 64)
    out_features: int = 1
    aggregation: Literal["sum", "mean", "max"] = "sum"
    dropout_rate: float = 0.1


class MessageFunction(nnx.Module):
    """MLP that computes messages from sender, receiver, and edge features.

    Computes: m_ij = MLP([h_sender || h_receiver || e_ij])
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        out_features: int,
        rngs: nnx.Rngs,
    ):
        """Initialize the message function.

        Args:
            node_features: Dimension of node features
            edge_features: Dimension of edge features
            out_features: Output dimension of message
            rngs: Random number generators
        """
        # Input: sender_features || receiver_features || edge_features
        in_features = 2 * node_features + edge_features
        hidden_features = out_features * 2

        self.linear1 = nnx.Linear(in_features, hidden_features, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_features, out_features, rngs=rngs)

    def __call__(
        self,
        sender_features: jnp.ndarray,
        receiver_features: jnp.ndarray,
        edge_features: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute messages for each edge.

        Args:
            sender_features: Features of sender nodes [n_edges, node_features]
            receiver_features: Features of receiver nodes [n_edges, node_features]
            edge_features: Features of edges [n_edges, edge_features]

        Returns:
            Messages of shape [n_edges, out_features]
        """
        # Concatenate sender, receiver, and edge features
        concat_features = jnp.concatenate(
            [sender_features, receiver_features, edge_features], axis=-1
        )

        # Two-layer MLP
        x = self.linear1(concat_features)
        x = nnx.relu(x)
        x = self.linear2(x)

        return x


class MessagePassingLayer(nnx.Module):
    """Single message passing layer with configurable aggregation.

    Implements:
    1. Gather sender/receiver features using edge indices
    2. Compute messages with edge features using MessageFunction
    3. Aggregate messages using specified method (sum/mean/max)
    4. Update nodes with residual connection
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        out_features: int,
        aggregation: Literal["sum", "mean", "max"],
        rngs: nnx.Rngs,
    ):
        """Initialize the message passing layer.

        Args:
            node_features: Input node feature dimension
            edge_features: Edge feature dimension
            out_features: Output node feature dimension
            aggregation: Aggregation method ('sum', 'mean', or 'max')
            rngs: Random number generators
        """
        self.aggregation = aggregation
        self.out_features = out_features

        # Message function
        self.message_fn = MessageFunction(
            node_features=node_features,
            edge_features=edge_features,
            out_features=out_features,
            rngs=rngs,
        )

        # Node update linear layer (for residual connection when dims match)
        self.node_update = nnx.Linear(node_features, out_features, rngs=rngs)

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Apply message passing.

        Args:
            graph: Input graph with node and edge features

        Returns:
            Graph with updated node features
        """
        nodes = graph.nodes
        edges = graph.edges
        n_nodes = nodes.shape[0]

        # Gather sender and receiver features for each edge
        sender_features = nodes[graph.senders]
        receiver_features = nodes[graph.receivers]

        # Compute messages for each edge
        messages = self.message_fn(sender_features, receiver_features, edges)

        # Aggregate messages at each receiver node
        if self.aggregation == "sum":
            aggregated = jraph.segment_sum(
                messages,
                graph.receivers,
                num_segments=n_nodes,
            )
        elif self.aggregation == "mean":
            # Sum messages
            aggregated = jraph.segment_sum(
                messages,
                graph.receivers,
                num_segments=n_nodes,
            )
            # Count number of messages per node for normalization
            ones = jnp.ones((graph.senders.shape[0],))
            degree = jraph.segment_sum(ones, graph.receivers, num_segments=n_nodes)
            degree = jnp.maximum(degree, 1.0)  # Avoid division by zero
            aggregated = aggregated / degree[:, None]
        elif self.aggregation == "max":
            # Initialize with very negative values
            init_values = jnp.full((n_nodes, self.out_features), -1e9)
            aggregated = jraph.segment_max(
                messages,
                graph.receivers,
                num_segments=n_nodes,
                indices_are_sorted=False,
            )
            # Handle nodes with no incoming edges
            ones = jnp.ones((graph.senders.shape[0],))
            has_edges = jraph.segment_sum(ones, graph.receivers, num_segments=n_nodes)
            aggregated = jnp.where(has_edges[:, None] > 0, aggregated, init_values)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        # Update nodes: project original features and add residual
        node_projection = self.node_update(nodes)
        new_nodes = aggregated + node_projection

        return graph._replace(nodes=new_nodes)


class UncertaintyMPNN(nnx.Module):
    """MPNN with uncertainty estimation via mean and variance heads.

    Outputs both mean prediction and predicted variance for uncertainty quantification.
    Uses edge features (bond information) in message passing.
    """

    def __init__(self, config: MPNNConfig, rngs: nnx.Rngs):
        """Initialize the UncertaintyMPNN.

        Args:
            config: MPNN configuration
            rngs: Random number generators
        """
        self.config = config
        self.rngs = rngs

        # Build message passing layers
        layers = []
        in_dim = config.node_features
        for hidden_dim in config.hidden_features:
            layers.append(
                MessagePassingLayer(
                    node_features=in_dim,
                    edge_features=config.edge_features,
                    out_features=hidden_dim,
                    aggregation=config.aggregation,
                    rngs=rngs,
                )
            )
            in_dim = hidden_dim

        self.mp_layers = nnx.List(layers)

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
        # Apply message passing layers
        for mp_layer in self.mp_layers:
            graph = mp_layer(graph)
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

        # Clip log_var to prevent variance explosion (variance in range [0.01, 100])
        log_var = jnp.clip(log_var, -4.6, 4.6)

        # Return mean and variance (exp for positivity)
        return mean, jnp.exp(log_var)

    def extract_embeddings(
        self, graph: jraph.GraphsTuple, training: bool = False
    ) -> jnp.ndarray:
        """Extract penultimate layer embeddings (graph-level).

        Extracts the pooled graph representations before the output heads,
        which can be used for Core-Set selection, DPP sampling, or other
        embedding-based acquisition strategies.

        Args:
            graph: Batched jraph.GraphsTuple
            training: Whether in training mode (enables dropout)

        Returns:
            Embeddings of shape [n_graphs, hidden_dim] where hidden_dim
            is the last element of hidden_features in the config.
        """
        # Apply message passing layers
        for mp_layer in self.mp_layers:
            graph = mp_layer(graph)
            graph = graph._replace(nodes=nnx.relu(graph.nodes))
            if training:
                graph = graph._replace(nodes=self.dropout(graph.nodes))

        # Global mean pooling per graph (stop before output heads)
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

        return pooled_sum / n_nodes_per_graph


def create_mpnn_optimizer(
    model: UncertaintyMPNN,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
) -> nnx.Optimizer:
    """Create optimizer for MPNN training with regularization.

    Args:
        model: The MPNN model to optimize
        learning_rate: Learning rate (default: 1e-3)
        weight_decay: L2 regularization weight (default: 1e-4)
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)

    Returns:
        nnx.Optimizer instance
    """
    import optax

    # Chain: gradient clipping -> adam with weight decay
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate, weight_decay=weight_decay),
    )

    return nnx.Optimizer(model, optimizer, wrt=nnx.Param)


@nnx.jit
def train_mpnn_step(
    model: UncertaintyMPNN,
    optimizer: nnx.Optimizer,
    graph: jraph.GraphsTuple,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled training step for MPNN.

    Args:
        model: The MPNN model
        optimizer: The optimizer
        graph: Batched (padded) input graphs
        labels: Target labels of shape [n_graphs] (padded with zeros)
        mask: Boolean mask indicating real graphs (not padding)

    Returns:
        Loss value
    """

    def loss_fn(model: UncertaintyMPNN) -> jnp.ndarray:
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
def eval_mpnn_step(
    model: UncertaintyMPNN,
    graph: jraph.GraphsTuple,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled evaluation step for MPNN.

    Args:
        model: The MPNN model
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


def get_mpnn_uncertainties(
    model: UncertaintyMPNN,
    graph: jraph.GraphsTuple,
    n_samples: int = 10,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get uncertainties from MPNN using MC Dropout.

    Args:
        model: The MPNN model
        graph: Batched input graphs
        n_samples: Number of MC dropout samples

    Returns:
        Tuple of (epistemic_variance, predicted_variance)
        - epistemic_variance: Variance from MC dropout (model uncertainty)
        - predicted_variance: Mean of predicted variances (aleatoric uncertainty)
    """
    predictions = []
    variances = []

    for _ in range(n_samples):
        mean, var = model(graph, training=True)  # Enable dropout
        predictions.append(mean.squeeze(-1))
        variances.append(var.squeeze(-1))

    predictions = jnp.stack(predictions, axis=0)
    variances = jnp.stack(variances, axis=0)

    # Epistemic uncertainty: variance across MC samples
    epistemic_var = jnp.var(predictions, axis=0)

    # Aleatoric uncertainty: mean of predicted variances
    aleatoric_var = jnp.mean(variances, axis=0)

    return epistemic_var, aleatoric_var
