import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Sequence
import flax.nnx as nnx
from dataclasses import dataclass

@dataclass
class GCNLayerConfig:
    """Configuration for Graph Convolutional Network Layer.
    
    Attributes:
        features: Number of output features for the layer
        use_bias: Whether to include bias in the linear transformation
    """
    in_features: int
    out_features: int
    use_bias: bool = True
    rngs: nnx.Rngs = nnx.Rngs(0)

class GCNLayer(nnx.Module):
    """Graph Convolutional Layer implementation following Kipf & Welling (ICLR 2017).
    
    This layer implements the propagation rule:
    H^(l+1) = σ(D^(-1/2) Ã D^(-1/2) H^(l) W^(l))
    where:
    - Ã is adjacency matrix with self-connections
    - D is the degree matrix of Ã
    - H^(l) is the node feature matrix at layer l
    - W^(l) is the weight matrix at layer l
    - σ is a non-linear activation function
    """
    def __init__(self, config: GCNLayerConfig):
        """Initialize GCN layer with the given configuration.
        
        Args:
            config: Configuration specifying output features and bias usage
        """
        super().__init__()
        self.config = config
        self.dense = nnx.Linear(in_features=self.config.in_features, out_features=self.config.out_features, use_bias=self.config.use_bias, rngs=self.config.rngs)
    
    def __call__(self, x: jnp.ndarray, adj: jnp.ndarray) -> jnp.ndarray:
        """Apply graph convolution operation.
        
        Args:
            x: Node feature matrix of shape [num_nodes, in_features]
            adj: Adjacency matrix of shape [num_nodes, num_nodes]
            
        Returns:
            Updated node features of shape [num_nodes, out_features]
        """
        # Add self-connections to adjacency matrix (A + I)
        adj_hat = adj + jnp.eye(adj.shape[0])
        
        # Normalize adjacency matrix (D^(-1/2) * A * D^(-1/2))
        deg = jnp.sum(adj_hat, axis=1)
        deg_inv_sqrt = jnp.power(deg, -0.5)
        # Handle isolated nodes (degree = 0)
        deg_inv_sqrt = jnp.where(jnp.isinf(deg_inv_sqrt), 0., deg_inv_sqrt)
        adj_normalized = deg_inv_sqrt[:, None] * adj_hat * deg_inv_sqrt[None, :]
        
        # Linear transformation and graph convolution
        x = self.dense(x)  # XW
        return jnp.matmul(adj_normalized, x)  # D^(-1/2) * A * D^(-1/2) * XW

@dataclass
class MolecularGCNConfig:
    """Configuration for Molecular Graph Convolutional Network.
    
    Attributes:
        in_features: Dimension of the input features
        hidden_features: Sequence of feature dimensions for each GCN layer
        out_features: Dimension of the final output
        dropout_rate: Rate for dropout regularization
    """
    in_features: int
    hidden_features: Sequence[int]
    out_features: int
    dropout_rate: float = 0.1
    rngs: nnx.Rngs = nnx.Rngs(0, params=1, dropout=2)

class MolecularGCN(nnx.Module):
    """Molecular Graph Convolutional Network for molecular property prediction.
    
    This model consists of multiple GCN layers followed by mean pooling and
    a final prediction layer. It's designed for graph-level tasks on molecular graphs.
    """
    def __init__(self, config: MolecularGCNConfig):
        """Initialize molecular GCN with the given configuration.
        
        Args:
            config: Configuration specifying architecture parameters
        """
        super().__init__()
        self.config = config
        
        # Initialize GCN layers
        # first layer has in_features = config.in_features
        # all other layers have in_features = out_features of previous layer


        self.gcn_layers = []
        in_features = self.config.in_features
        for out_features in self.config.hidden_features:
            self.gcn_layers.append(
                GCNLayer(GCNLayerConfig(
                    in_features=in_features,
                    out_features=out_features,
                    rngs=self.config.rngs
                ))
            )
            in_features = out_features
        
        # Final prediction layer
        self.output = nnx.Linear(
            in_features=self.config.hidden_features[-1],
            out_features=self.config.out_features,
            rngs=self.config.rngs
        )
        
        # Dropout layer for regularization
        self.dropout = nnx.Dropout(rate=self.config.dropout_rate, rngs=self.config.rngs)
    
    def __call__(
        self, 
        x: jnp.ndarray, 
        adj: jnp.ndarray, 
        *, 
        training: bool = False
    ) -> jnp.ndarray:
        """Forward pass through the molecular GCN.
        
        Args:
            x: Node feature matrix of shape [num_nodes, in_features]
            adj: Adjacency matrix of shape [num_nodes, num_nodes]
            training: Whether in training mode (enables dropout)
            
        Returns:
            Predicted molecular property of shape [output_features]
        """
        # Process features through GCN layers
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, adj)
            x = nnx.relu(x)  # Non-linear activation
            if training:
                x = self.dropout(x, deterministic=not training)
        
        # Global pooling: compute mean of node features to get graph representation
        x = jnp.mean(x, axis=0)
        
        # Final prediction
        return self.output(x)

@dataclass
class UncertaintyGCNConfig:
    """Configuration for Uncertainty-aware Graph Convolutional Network.
    
    Attributes:
        in_features: Dimension of the input features
        hidden_features: Sequence of feature dimensions for each GCN layer
        output_features: Dimension of the final output
        dropout_rate: Rate for dropout regularization
        n_heads: Number of output heads for uncertainty (typically 2 for mean/variance)
    """
    in_features: int
    hidden_features: Sequence[int]
    out_features: int
    dropout_rate: float = 0.1
    n_heads: int = 2  # Number of output heads for uncertainty
    rngs: nnx.Rngs = nnx.Rngs(0, params=1, dropout=2)

class UncertaintyGCN(nnx.Module):
    """Uncertainty-aware Graph Convolutional Network for probabilistic predictions.
    
    This model extends MolecularGCN by adding separate prediction heads for
    mean and variance, enabling uncertainty quantification in predictions.
    It implements a Bayesian-by-design approach where aleatoric uncertainty
    is modeled explicitly.
    """
    def __init__(self, config: UncertaintyGCNConfig):
        """Initialize uncertainty-aware GCN with the given configuration.
        
        Args:
            config: Configuration specifying architecture parameters
        """
        super().__init__()
        self.config = config
        
        # Base GCN model for feature extraction
        self.base_model = MolecularGCN(MolecularGCNConfig(
            in_features=self.config.in_features,
            hidden_features=self.config.hidden_features,
            out_features=self.config.out_features,
            dropout_rate=self.config.dropout_rate,
            rngs=self.config.rngs
        ))
        
        # Separate prediction heads for mean and uncertainty
        self.mean_head = nnx.Linear(in_features=self.config.out_features, out_features=self.config.out_features, rngs=self.config.rngs)
        self.var_head = nnx.Linear(in_features=self.config.out_features, out_features=self.config.out_features, rngs=self.config.rngs)
    
    def __call__(
        self, 
        x: jnp.ndarray, 
        adj: jnp.ndarray, 
        *, 
        training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass through the uncertainty-aware GCN.
        
        Args:
            x: Node feature matrix of shape [num_nodes, in_features]
            adj: Adjacency matrix of shape [num_nodes, num_nodes]
            training: Whether in training mode (enables dropout)
            
        Returns:
            Tuple of (mean, variance) for the predicted property
        """
        # Get shared representations from base model
        shared_features = self.base_model(x, adj, training=training)
        
        # Predict mean and log variance from shared features
        mean = self.mean_head(shared_features)
        log_var = self.var_head(shared_features)  # Predict log(variance) for numerical stability
        
        # Return mean and variance (exp of log_var)
        return mean, jnp.exp(log_var)

def create_model(config: UncertaintyGCNConfig) -> Tuple[UncertaintyGCN, Dict]:
    """Create and initialize an uncertainty-aware GCN model.
    
    Args:
        config: Configuration for the model architecture
        
    Returns:
        Tuple of (model, initialized_variables)
    """
    model = UncertaintyGCN(config)
    # Initialize with dummy inputs
    variables = model.init(jax.random.PRNGKey(0), jnp.ones((5, 10)), jnp.ones((5, 5)))
    return model, variables

def create_train_step(model: UncertaintyGCN, learning_rate: float = 1e-3):
    """Create a jitted training step function for the uncertainty-aware GCN.
    
    This function implements training with a negative log-likelihood loss
    that accounts for predictive uncertainty.
    
    Args:
        model: The uncertainty-aware GCN model
        learning_rate: Learning rate for parameter updates
        
    Returns:
        Jitted function for performing a single training step
    """
    
    @jax.jit
    def train_step(
        variables: Dict,
        batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        key: jnp.ndarray
    ) -> Tuple[float, Dict]:
        """Perform a single training step.
        
        Args:
            variables: Model parameters
            batch: Tuple of (node_features, adjacency_matrix, target)
            key: JAX PRNG key for randomness
            
        Returns:
            Tuple of (loss, updated_variables)
        """
        def loss_fn(params):
            x, adj, y = batch
            mean, var = model.apply(params, x, adj, training=True)
            # Negative log-likelihood for a Gaussian: 0.5 * (log(var) + (y-mean)²/var)
            nll = 0.5 * jnp.sum(jnp.log(var) + (y - mean)**2 / var)
            return nll
        
        # Compute loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(variables)
        
        # Update parameters with simple SGD
        # Note: In practice, you might want to use optimizers from optax
        variables = jax.tree_map(
            lambda p, g: p - learning_rate * g,
            variables,
            grads
        )
        return loss, variables
    
    return train_step