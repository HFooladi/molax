"""Graph Transformer using jraph for efficient batched processing.

Graph Transformer applies full self-attention over molecular graphs with
positional encodings to capture long-range dependencies. This enables
state-of-the-art performance on molecular property prediction.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jraph
import optax


@dataclass
class GraphTransformerConfig:
    """Configuration for Graph Transformer.

    Attributes:
        node_features: Input node feature dimension (default: 6 for atom features)
        edge_features: Edge feature dimension for attention bias (default: 1)
        hidden_features: List of hidden layer dimensions (transformer dims)
        out_features: Output dimension
        n_heads: Number of attention heads
        ffn_ratio: Ratio of FFN hidden dim to model dim (default: 4.0)
        dropout_rate: Dropout rate for node features
        attention_dropout_rate: Dropout rate for attention weights
        pe_type: Positional encoding type ('rwpe', 'laplacian', 'none')
        pe_dim: Positional encoding dimension (default: 16)
    """

    node_features: int = 6
    edge_features: int = 1
    hidden_features: Sequence[int] = (64, 64)
    out_features: int = 1
    n_heads: int = 4
    ffn_ratio: float = 4.0
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    pe_type: Literal["rwpe", "laplacian", "none"] = "rwpe"
    pe_dim: int = 16


class RandomWalkPositionalEncoding(nnx.Module):
    """Random Walk Positional Encoding.

    Computes landing probabilities from random walks of different lengths.
    More scalable than Laplacian eigenvectors and captures both local
    and global graph structure.

    For each node i, computes [A^1_ii, A^2_ii, ..., A^k_ii] where A is
    the row-normalized adjacency matrix.
    """

    def __init__(self, pe_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        """Initialize the RWPE module.

        Args:
            pe_dim: Number of random walk steps (landing probabilities)
            hidden_dim: Output dimension after linear projection
            rngs: Random number generators
        """
        self.pe_dim = pe_dim
        self.linear = nnx.Linear(pe_dim, hidden_dim, rngs=rngs)

    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        """Compute random walk positional encodings.

        Args:
            graph: Input graph

        Returns:
            Positional encodings of shape [n_nodes, hidden_dim]
        """
        n_nodes = graph.nodes.shape[0]

        # Build row-normalized adjacency matrix
        # First, compute degree for each node (number of outgoing edges from senders)
        ones = jnp.ones((graph.senders.shape[0],))
        degree = jraph.segment_sum(ones, graph.senders, num_segments=n_nodes)
        degree = jnp.maximum(degree, 1.0)  # Avoid division by zero

        # Create sparse adjacency representation
        # A[receivers, senders] = 1 / degree[senders]
        # We'll compute A^k iteratively using message passing

        # Initialize landing probabilities: p_0 = identity (node is at itself)
        # We'll store diagonal of A^k for k = 1, ..., pe_dim
        pe_features = []

        # Current node probability distribution (initially identity)
        # For each node i, prob[i, j] = probability of being at j starting from i
        # We track the diagonal only by iterating message passing
        current_prob = jnp.eye(n_nodes)

        for k in range(self.pe_dim):
            # Message passing: new_prob[i, :] = sum_j A[i,j] * prob[j, :]
            # Using graph structure: aggregate from receivers to senders
            # normalized by sender degree

            # Gather probabilities from sender nodes
            sender_probs = current_prob[graph.senders]  # [n_edges, n_nodes]
            normalized = sender_probs / degree[graph.senders, None]

            # Aggregate at receiver nodes
            new_prob = jraph.segment_sum(
                normalized,
                graph.receivers,
                num_segments=n_nodes,
            )

            current_prob = new_prob
            # Extract diagonal (landing probability at starting node)
            pe_features.append(jnp.diag(current_prob))

        # Stack and project
        pe = jnp.stack(pe_features, axis=-1)  # [n_nodes, pe_dim]
        return self.linear(pe)


class LaplacianPositionalEncoding(nnx.Module):
    """Laplacian Positional Encoding.

    Computes the k smallest eigenvectors of the graph Laplacian.
    Sign ambiguity is handled by using absolute values.

    Note: This is more expensive than RWPE and may have numerical
    issues for batched graphs. Use RWPE for most applications.
    """

    def __init__(self, pe_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        """Initialize the Laplacian PE module.

        Args:
            pe_dim: Number of eigenvectors to use
            hidden_dim: Output dimension after linear projection
            rngs: Random number generators
        """
        self.pe_dim = pe_dim
        self.linear = nnx.Linear(pe_dim, hidden_dim, rngs=rngs)

    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        """Compute Laplacian positional encodings.

        Args:
            graph: Input graph

        Returns:
            Positional encodings of shape [n_nodes, hidden_dim]
        """
        n_nodes = graph.nodes.shape[0]

        # Build adjacency matrix
        adj = jnp.zeros((n_nodes, n_nodes))
        adj = adj.at[graph.receivers, graph.senders].set(1.0)

        # Make symmetric (in case graph is directed)
        adj = (adj + adj.T) / 2.0

        # Compute degree matrix
        degree = jnp.sum(adj, axis=1)

        # Compute Laplacian: L = D - A
        laplacian = jnp.diag(degree) - adj

        # Compute eigenvalues and eigenvectors
        # Note: eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = jnp.linalg.eigh(laplacian)

        # Take first pe_dim eigenvectors (excluding the constant eigenvector)
        # Skip the first (constant) eigenvector
        k = min(self.pe_dim + 1, n_nodes)
        pe = eigenvectors[:, 1:k]  # [n_nodes, k-1]

        # Pad if we don't have enough eigenvectors
        if pe.shape[1] < self.pe_dim:
            padding = jnp.zeros((n_nodes, self.pe_dim - pe.shape[1]))
            pe = jnp.concatenate([pe, padding], axis=-1)
        else:
            pe = pe[:, : self.pe_dim]

        # Use absolute values to handle sign ambiguity
        pe = jnp.abs(pe)

        return self.linear(pe)


class GraphTransformerAttention(nnx.Module):
    """Multi-head self-attention for graphs.

    Computes full self-attention over all nodes with graph-aware masking
    to prevent attention across different graphs in a batch.
    Optionally incorporates edge features as attention bias.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        edge_features: int,
        attention_dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        """Initialize the attention module.

        Args:
            hidden_dim: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            edge_features: Edge feature dimension (0 = no edge bias)
            attention_dropout_rate: Dropout rate for attention weights
            rngs: Random number generators
        """
        assert hidden_dim % n_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
        )

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim**-0.5
        self.edge_features = edge_features

        # Q, K, V projections
        self.q_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.out_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)

        # Optional edge bias projection
        if edge_features > 0:
            self.edge_proj = nnx.Linear(edge_features, n_heads, rngs=rngs)

        # Attention dropout
        self.attn_dropout = nnx.Dropout(attention_dropout_rate, rngs=rngs)

    def __call__(
        self,
        nodes: jnp.ndarray,
        graph: jraph.GraphsTuple,
        training: bool = False,
    ) -> jnp.ndarray:
        """Apply multi-head self-attention.

        Args:
            nodes: Node features of shape [n_nodes, hidden_dim]
            graph: Graph structure for masking and edge features
            training: Whether in training mode (enables dropout)

        Returns:
            Updated node features of shape [n_nodes, hidden_dim]
        """
        n_nodes = nodes.shape[0]
        n_graphs = graph.n_node.shape[0]

        # Compute Q, K, V
        q = self.q_proj(nodes)  # [n_nodes, hidden_dim]
        k = self.k_proj(nodes)
        v = self.v_proj(nodes)

        # Reshape for multi-head attention
        # [n_nodes, n_heads, head_dim]
        q = q.reshape(n_nodes, self.n_heads, self.head_dim)
        k = k.reshape(n_nodes, self.n_heads, self.head_dim)
        v = v.reshape(n_nodes, self.n_heads, self.head_dim)

        # Compute attention scores: [n_nodes, n_nodes, n_heads]
        # Using einsum: (i, h, d) x (j, h, d) -> (i, j, h)
        attn_scores = jnp.einsum("ihd,jhd->ijh", q, k) * self.scale

        # Create graph-aware attention mask
        # Nodes should only attend to nodes in the same graph
        graph_indices = jnp.repeat(
            jnp.arange(n_graphs),
            graph.n_node,
            total_repeat_length=n_nodes,
        )
        # Mask: same_graph[i, j] = (graph_indices[i] == graph_indices[j])
        same_graph = graph_indices[:, None] == graph_indices[None, :]

        # Apply mask (set cross-graph attention to -inf)
        attn_mask = jnp.where(same_graph[:, :, None], 0.0, -1e9)
        attn_scores = attn_scores + attn_mask

        # Optional: add edge bias
        if self.edge_features > 0 and graph.edges is not None:
            # Project edge features to n_heads biases
            edge_bias = self.edge_proj(graph.edges)  # [n_edges, n_heads]

            # Create edge bias matrix
            edge_bias_matrix = jnp.zeros((n_nodes, n_nodes, self.n_heads))
            edge_bias_matrix = edge_bias_matrix.at[graph.receivers, graph.senders].set(
                edge_bias
            )

            attn_scores = attn_scores + edge_bias_matrix

        # Softmax and dropout
        attn_weights = jax.nn.softmax(attn_scores, axis=1)
        if training:
            attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values: [n_nodes, n_heads, head_dim]
        out = jnp.einsum("ijh,jhd->ihd", attn_weights, v)

        # Reshape and project output
        out = out.reshape(n_nodes, self.hidden_dim)
        out = self.out_proj(out)

        return out


class GraphTransformerLayer(nnx.Module):
    """Single Graph Transformer layer with Pre-LayerNorm.

    Pre-LayerNorm architecture:
    1. LayerNorm -> Attention -> Residual
    2. LayerNorm -> FFN -> Residual

    Pre-LayerNorm is more stable for training than Post-LayerNorm.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        ffn_ratio: float,
        edge_features: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        """Initialize the transformer layer.

        Args:
            hidden_dim: Model dimension
            n_heads: Number of attention heads
            ffn_ratio: Ratio of FFN hidden dim to model dim
            edge_features: Edge feature dimension (0 = no edge bias)
            dropout_rate: Dropout rate for FFN
            attention_dropout_rate: Dropout rate for attention
            rngs: Random number generators
        """
        self.hidden_dim = hidden_dim

        # Pre-LayerNorm
        self.norm1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_dim, rngs=rngs)

        # Self-attention
        self.attention = GraphTransformerAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            edge_features=edge_features,
            attention_dropout_rate=attention_dropout_rate,
            rngs=rngs,
        )

        # FFN
        ffn_hidden = int(hidden_dim * ffn_ratio)
        self.ffn_linear1 = nnx.Linear(hidden_dim, ffn_hidden, rngs=rngs)
        self.ffn_linear2 = nnx.Linear(ffn_hidden, hidden_dim, rngs=rngs)

        # Dropout
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(
        self,
        nodes: jnp.ndarray,
        graph: jraph.GraphsTuple,
        training: bool = False,
    ) -> jnp.ndarray:
        """Apply transformer layer.

        Args:
            nodes: Node features of shape [n_nodes, hidden_dim]
            graph: Graph structure
            training: Whether in training mode

        Returns:
            Updated node features of shape [n_nodes, hidden_dim]
        """
        # Pre-LayerNorm attention block
        residual = nodes
        nodes = self.norm1(nodes)
        nodes = self.attention(nodes, graph, training)
        if training:
            nodes = self.dropout(nodes)
        nodes = residual + nodes

        # Pre-LayerNorm FFN block
        residual = nodes
        nodes = self.norm2(nodes)
        nodes = self.ffn_linear1(nodes)
        nodes = nnx.gelu(nodes)
        if training:
            nodes = self.dropout(nodes)
        nodes = self.ffn_linear2(nodes)
        if training:
            nodes = self.dropout(nodes)
        nodes = residual + nodes

        return nodes


class UncertaintyGraphTransformer(nnx.Module):
    """Graph Transformer with uncertainty estimation.

    Outputs both mean prediction and predicted variance for uncertainty
    quantification. Uses full self-attention with positional encodings
    to capture long-range dependencies in molecular graphs.
    """

    def __init__(self, config: GraphTransformerConfig, rngs: nnx.Rngs):
        """Initialize the UncertaintyGraphTransformer.

        Args:
            config: Graph Transformer configuration
            rngs: Random number generators
        """
        self.config = config
        self.rngs = rngs

        # Get first hidden dimension (used throughout transformer)
        hidden_dim = config.hidden_features[0]

        # Input projection
        input_dim = config.node_features
        if config.pe_type != "none":
            input_dim += hidden_dim  # PE will be projected to hidden_dim

        self.input_proj = nnx.Linear(input_dim, hidden_dim, rngs=rngs)

        # Positional encoding
        self.pe: Optional[
            Union[RandomWalkPositionalEncoding, LaplacianPositionalEncoding]
        ] = None
        if config.pe_type == "rwpe":
            self.pe = RandomWalkPositionalEncoding(
                pe_dim=config.pe_dim,
                hidden_dim=hidden_dim,
                rngs=rngs,
            )
        elif config.pe_type == "laplacian":
            self.pe = LaplacianPositionalEncoding(
                pe_dim=config.pe_dim,
                hidden_dim=hidden_dim,
                rngs=rngs,
            )

        # Build transformer layers
        layers = []
        for i, hidden_features in enumerate(config.hidden_features):
            # Only use edge features in first layer (or all layers)
            edge_feat = config.edge_features if i == 0 else 0

            layers.append(
                GraphTransformerLayer(
                    hidden_dim=hidden_dim,
                    n_heads=config.n_heads,
                    ffn_ratio=config.ffn_ratio,
                    edge_features=edge_feat,
                    dropout_rate=config.dropout_rate,
                    attention_dropout_rate=config.attention_dropout_rate,
                    rngs=rngs,
                )
            )

        self.transformer_layers = nnx.List(layers)

        # Final layer norm
        self.final_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)

        # Output heads
        self.mean_head = nnx.Linear(hidden_dim, config.out_features, rngs=rngs)
        self.var_head = nnx.Linear(hidden_dim, config.out_features, rngs=rngs)

        # Dropout for input
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
        nodes = graph.nodes

        # Add positional encoding
        if self.pe is not None:
            pe = self.pe(graph)
            nodes = jnp.concatenate([nodes, pe], axis=-1)

        # Input projection
        nodes = self.input_proj(nodes)
        if training:
            nodes = self.dropout(nodes)

        # Apply transformer layers
        for layer in self.transformer_layers:
            nodes = layer(nodes, graph, training)

        # Final layer norm
        nodes = self.final_norm(nodes)

        # Global mean pooling per graph
        n_graphs = graph.n_node.shape[0]
        graph_indices = jnp.repeat(
            jnp.arange(n_graphs),
            graph.n_node,
            total_repeat_length=nodes.shape[0],
        )

        # Sum pooling
        pooled_sum = jraph.segment_sum(
            nodes,
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
            is the first element of hidden_features in the config.
        """
        nodes = graph.nodes

        # Add positional encoding
        if self.pe is not None:
            pe = self.pe(graph)
            nodes = jnp.concatenate([nodes, pe], axis=-1)

        # Input projection
        nodes = self.input_proj(nodes)
        if training:
            nodes = self.dropout(nodes)

        # Apply transformer layers
        for layer in self.transformer_layers:
            nodes = layer(nodes, graph, training)

        # Final layer norm
        nodes = self.final_norm(nodes)

        # Global mean pooling per graph (stop before output heads)
        n_graphs = graph.n_node.shape[0]
        graph_indices = jnp.repeat(
            jnp.arange(n_graphs),
            graph.n_node,
            total_repeat_length=nodes.shape[0],
        )

        # Sum pooling
        pooled_sum = jraph.segment_sum(
            nodes,
            graph_indices,
            num_segments=n_graphs,
        )

        # Normalize by number of nodes (avoiding division by zero)
        n_nodes_per_graph = graph.n_node.astype(jnp.float32)
        n_nodes_per_graph = jnp.maximum(n_nodes_per_graph, 1.0)[:, None]

        return pooled_sum / n_nodes_per_graph


def create_graph_transformer_optimizer(
    model: UncertaintyGraphTransformer,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    warmup_steps: int = 100,
    max_grad_norm: float = 1.0,
) -> nnx.Optimizer:
    """Create optimizer for Graph Transformer training with warmup.

    Transformers benefit from learning rate warmup to stabilize early training.

    Args:
        model: The Graph Transformer model to optimize
        learning_rate: Peak learning rate (default: 1e-4)
        weight_decay: L2 regularization weight (default: 1e-4)
        warmup_steps: Number of warmup steps (default: 100)
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)

    Returns:
        nnx.Optimizer instance
    """
    # Learning rate schedule with linear warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=10000,  # Will plateau at peak after warmup if training < 10k steps
        end_value=learning_rate * 0.1,
    )

    # Chain: gradient clipping -> adam with weight decay
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(schedule, weight_decay=weight_decay),
    )

    return nnx.Optimizer(model, optimizer, wrt=nnx.Param)


@nnx.jit
def train_graph_transformer_step(
    model: UncertaintyGraphTransformer,
    optimizer: nnx.Optimizer,
    graph: jraph.GraphsTuple,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled training step for Graph Transformer.

    Args:
        model: The Graph Transformer model
        optimizer: The optimizer
        graph: Batched (padded) input graphs
        labels: Target labels of shape [n_graphs] (padded with zeros)
        mask: Boolean mask indicating real graphs (not padding)

    Returns:
        Loss value
    """

    def loss_fn(model: UncertaintyGraphTransformer) -> jnp.ndarray:
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
def eval_graph_transformer_step(
    model: UncertaintyGraphTransformer,
    graph: jraph.GraphsTuple,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled evaluation step for Graph Transformer.

    Args:
        model: The Graph Transformer model
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


def get_graph_transformer_uncertainties(
    model: UncertaintyGraphTransformer,
    graph: jraph.GraphsTuple,
    n_samples: int = 10,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get uncertainties from Graph Transformer using MC Dropout.

    Args:
        model: The Graph Transformer model
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
