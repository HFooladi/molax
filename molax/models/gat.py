"""Graph Attention Networks using jraph for efficient batched processing.

GAT uses learned attention weights to dynamically weight neighbor contributions,
enabling adaptive aggregation based on node relevance.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import flax.nnx as nnx
import jax.numpy as jnp
import jraph


@dataclass
class GATConfig:
    """Configuration for Graph Attention Network.

    Attributes:
        node_features: Input node feature dimension (default: 6 for atom features)
        edge_features: Edge feature dimension (0 = none, >0 = include in attention)
        hidden_features: List of hidden layer dimensions
        out_features: Output dimension
        n_heads: Number of attention heads
        dropout_rate: Dropout rate for node features
        attention_dropout_rate: Dropout rate for attention weights
        negative_slope: Negative slope for LeakyReLU in attention
    """

    node_features: int = 6
    edge_features: int = 0
    hidden_features: Sequence[int] = (64, 64)
    out_features: int = 1
    n_heads: int = 4
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    negative_slope: float = 0.2


class GATAttention(nnx.Module):
    """Single attention head for GAT.

    Computes attention coefficients using concatenation mechanism:
    e_ij = LeakyReLU(a^T [Wh_i || Wh_j || e_ij])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        edge_features: int,
        negative_slope: float,
        rngs: nnx.Rngs,
    ):
        """Initialize the attention head.

        Args:
            in_features: Input node feature dimension
            out_features: Output node feature dimension
            edge_features: Edge feature dimension (0 = no edge features)
            negative_slope: Negative slope for LeakyReLU
            rngs: Random number generators
        """
        self.out_features = out_features
        self.edge_features = edge_features
        self.negative_slope = negative_slope

        # Linear projection for nodes
        self.W = nnx.Linear(in_features, out_features, rngs=rngs)

        # Attention: a^T [Wh_i || Wh_j || e_ij]
        attn_in = 2 * out_features + edge_features
        self.attention = nnx.Linear(attn_in, 1, rngs=rngs)

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        training: bool = False,
    ) -> jnp.ndarray:
        """Compute attention-weighted aggregation.

        Args:
            graph: Input graph with node (and optionally edge) features
            training: Whether in training mode (unused here, for API consistency)

        Returns:
            Updated node features of shape [n_nodes, out_features]
        """
        nodes = graph.nodes
        n_nodes = nodes.shape[0]

        # Linear projection
        h = self.W(nodes)  # [n_nodes, out_features]

        # Gather sender and receiver features for attention computation
        src = h[graph.senders]  # [n_edges, out_features]
        dst = h[graph.receivers]  # [n_edges, out_features]

        # Compute attention logits
        if self.edge_features > 0 and graph.edges is not None:
            attn_input = jnp.concatenate([src, dst, graph.edges], axis=-1)
        else:
            attn_input = jnp.concatenate([src, dst], axis=-1)

        # Apply attention network with LeakyReLU
        e = self.attention(attn_input)  # [n_edges, 1]
        e = jnp.where(
            e > 0,
            e,
            self.negative_slope * e,
        )  # LeakyReLU

        # Softmax over incoming edges per node using segment_softmax
        # Note: segment_softmax normalizes within each segment (receiver node)
        alpha = jraph.segment_softmax(
            e.squeeze(-1),
            graph.receivers,
            num_segments=n_nodes,
        )  # [n_edges]

        # Aggregate with attention weights
        messages = alpha[:, None] * src  # [n_edges, out_features]
        aggregated = jraph.segment_sum(
            messages,
            graph.receivers,
            num_segments=n_nodes,
        )  # [n_nodes, out_features]

        return aggregated


class GATLayer(nnx.Module):
    """Multi-head GAT layer.

    Applies multiple attention heads in parallel and either concatenates
    (for intermediate layers) or averages (for final layer) their outputs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,
        edge_features: int,
        negative_slope: float,
        concat_heads: bool,
        attention_dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        """Initialize the multi-head GAT layer.

        Args:
            in_features: Input node feature dimension
            out_features: Output dimension (split across heads if concat)
            n_heads: Number of attention heads
            edge_features: Edge feature dimension (0 = no edge features)
            negative_slope: Negative slope for LeakyReLU
            concat_heads: Whether to concatenate (True) or average (False) head outputs
            attention_dropout_rate: Dropout rate for attention weights
            rngs: Random number generators
        """
        self.n_heads = n_heads
        self.concat_heads = concat_heads
        self.attention_dropout_rate = attention_dropout_rate

        # Compute per-head dimension
        if concat_heads:
            assert out_features % n_heads == 0, (
                f"out_features ({out_features}) must be divisible by n_heads"
            )
            head_dim = out_features // n_heads
        else:
            head_dim = out_features

        # Create attention heads
        self.heads = nnx.List(
            [
                GATAttention(
                    in_features=in_features,
                    out_features=head_dim,
                    edge_features=edge_features,
                    negative_slope=negative_slope,
                    rngs=rngs,
                )
                for _ in range(n_heads)
            ]
        )

        # Attention dropout
        self.attn_dropout = nnx.Dropout(attention_dropout_rate, rngs=rngs)

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        training: bool = False,
    ) -> jraph.GraphsTuple:
        """Apply multi-head attention.

        Args:
            graph: Input graph with node features
            training: Whether in training mode (enables dropout)

        Returns:
            Graph with updated node features
        """
        # Compute attention for each head
        head_outputs = [head(graph, training) for head in self.heads]

        # Apply attention dropout if training
        if training and self.attention_dropout_rate > 0:
            head_outputs = [self.attn_dropout(h) for h in head_outputs]

        # Combine head outputs
        if self.concat_heads:
            # Concatenate for intermediate layers
            new_nodes = jnp.concatenate(head_outputs, axis=-1)
        else:
            # Average for final layer
            new_nodes = jnp.mean(jnp.stack(head_outputs), axis=0)

        return graph._replace(nodes=new_nodes)


class UncertaintyGAT(nnx.Module):
    """GAT with uncertainty estimation via mean and variance heads.

    Outputs both mean prediction and predicted variance for uncertainty quantification.
    Uses multi-head attention for adaptive neighbor aggregation.
    """

    def __init__(self, config: GATConfig, rngs: nnx.Rngs):
        """Initialize the UncertaintyGAT.

        Args:
            config: GAT configuration
            rngs: Random number generators
        """
        self.config = config
        self.rngs = rngs

        # Build GAT layers
        layers = []
        in_dim = config.node_features

        for hidden_dim in config.hidden_features:
            # Use concat for all layers except the last
            concat_heads = True  # Always concat, final pooling handles output

            layers.append(
                GATLayer(
                    in_features=in_dim,
                    out_features=hidden_dim,
                    n_heads=config.n_heads,
                    edge_features=config.edge_features,
                    negative_slope=config.negative_slope,
                    concat_heads=concat_heads,
                    attention_dropout_rate=config.attention_dropout_rate,
                    rngs=rngs,
                )
            )
            in_dim = hidden_dim

        self.gat_layers = nnx.List(layers)

        # Separate heads for mean and variance
        self.mean_head = nnx.Linear(in_dim, config.out_features, rngs=rngs)
        self.var_head = nnx.Linear(in_dim, config.out_features, rngs=rngs)

        # Feature dropout
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
        # Apply GAT layers
        for gat_layer in self.gat_layers:
            graph = gat_layer(graph, training)
            graph = graph._replace(nodes=nnx.elu(graph.nodes))
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
        # Apply GAT layers
        for gat_layer in self.gat_layers:
            graph = gat_layer(graph, training)
            graph = graph._replace(nodes=nnx.elu(graph.nodes))
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


def create_gat_optimizer(
    model: UncertaintyGAT,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
) -> nnx.Optimizer:
    """Create optimizer for GAT training with regularization.

    Args:
        model: The GAT model to optimize
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
def train_gat_step(
    model: UncertaintyGAT,
    optimizer: nnx.Optimizer,
    graph: jraph.GraphsTuple,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled training step for GAT.

    Args:
        model: The GAT model
        optimizer: The optimizer
        graph: Batched (padded) input graphs
        labels: Target labels of shape [n_graphs] (padded with zeros)
        mask: Boolean mask indicating real graphs (not padding)

    Returns:
        Loss value
    """

    def loss_fn(model: UncertaintyGAT) -> jnp.ndarray:
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
def eval_gat_step(
    model: UncertaintyGAT,
    graph: jraph.GraphsTuple,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled evaluation step for GAT.

    Args:
        model: The GAT model
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


def get_gat_uncertainties(
    model: UncertaintyGAT,
    graph: jraph.GraphsTuple,
    n_samples: int = 10,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get uncertainties from GAT using MC Dropout.

    Args:
        model: The GAT model
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
