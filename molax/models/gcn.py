import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Sequence
import flax.nnx as nnx
from dataclasses import dataclass

@dataclass
class GCNLayerConfig:
    features: int
    use_bias: bool = True

class GCNLayer(nnx.Module):
    """Graph Convolutional Layer"""
    def __init__(self, config: GCNLayerConfig):
        super().__init__()
        self.config = config
        
    def __setup__(self):
        self.dense = nnx.Linear(self.config.features, use_bias=self.config.use_bias)
    
    def __call__(self, x: jnp.ndarray, adj: jnp.ndarray) -> jnp.ndarray:
        # Add self-connections to adjacency matrix
        adj_hat = adj + jnp.eye(adj.shape[0])
        
        # Normalize adjacency matrix
        deg = jnp.sum(adj_hat, axis=1)
        deg_inv_sqrt = jnp.power(deg, -0.5)
        deg_inv_sqrt = jnp.where(jnp.isinf(deg_inv_sqrt), 0., deg_inv_sqrt)
        adj_normalized = deg_inv_sqrt[:, None] * adj_hat * deg_inv_sqrt[None, :]
        
        # Linear transformation and graph convolution
        x = self.dense(x)
        return jnp.matmul(adj_normalized, x)

@dataclass
class MolecularGCNConfig:
    hidden_features: Sequence[int]
    output_features: int
    dropout_rate: float = 0.1

class MolecularGCN(nnx.Module):
    def __init__(self, config: MolecularGCNConfig):
        super().__init__()
        self.config = config
    
    def __setup__(self):
        # Initialize GCN layers
        self.gcn_layers = [
            GCNLayer(GCNLayerConfig(features=features))
            for features in self.config.hidden_features
        ]
        
        # Final prediction layer
        self.output = nnx.Linear(self.config.output_features)
        
        # Dropout layer
        self.dropout = nnx.Dropout(self.config.dropout_rate)
    
    def __call__(
        self, 
        x: jnp.ndarray, 
        adj: jnp.ndarray, 
        *, 
        training: bool = False
    ) -> jnp.ndarray:
        # Initial feature processing through GCN layers
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, adj)
            x = nnx.relu(x)
            if training:
                x = self.dropout(x, deterministic=not training)
        
        # Global pooling (mean of node features)
        x = jnp.mean(x, axis=0)
        
        # Final prediction
        return self.output(x)

@dataclass
class UncertaintyGCNConfig:
    hidden_features: Sequence[int]
    output_features: int
    dropout_rate: float = 0.1
    n_heads: int = 2  # Number of output heads for uncertainty

class UncertaintyGCN(nnx.Module):
    def __init__(self, config: UncertaintyGCNConfig):
        super().__init__()
        self.config = config
    
    def __setup__(self):
        # Base GCN model
        self.base_model = MolecularGCN(MolecularGCNConfig(
            hidden_features=self.config.hidden_features,
            output_features=self.config.hidden_features[-1],
            dropout_rate=self.config.dropout_rate
        ))
        
        # Prediction heads for mean and uncertainty
        self.mean_head = nnx.Linear(self.config.output_features)
        self.var_head = nnx.Linear(self.config.output_features)
    
    def __call__(
        self, 
        x: jnp.ndarray, 
        adj: jnp.ndarray, 
        *, 
        training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Get shared representations
        shared_features = self.base_model(x, adj, training=training)
        
        # Predict mean and log variance
        mean = self.mean_head(shared_features)
        log_var = self.var_head(shared_features)
        
        return mean, jnp.exp(log_var)

def create_model(config: UncertaintyGCNConfig) -> Tuple[UncertaintyGCN, Dict]:
    """Create model and initialize parameters."""
    model = UncertaintyGCN(config)
    variables = model.init(jax.random.PRNGKey(0), jnp.ones((5, 10)), jnp.ones((5, 5)))
    return model, variables

def create_train_step(model: UncertaintyGCN, learning_rate: float = 1e-3):
    """Create jitted training step function."""
    
    @jax.jit
    def train_step(
        variables: Dict,
        batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        key: jnp.ndarray
    ) -> Tuple[float, Dict]:
        def loss_fn(params):
            x, adj, y = batch
            mean, var = model.apply(params, x, adj, training=True)
            nll = 0.5 * jnp.sum(jnp.log(var) + (y - mean)**2 / var)
            return nll
        
        loss, grads = jax.value_and_grad(loss_fn)(variables)
        variables = jax.tree_map(
            lambda p, g: p - learning_rate * g,
            variables,
            grads
        )
        return loss, variables
    
    return train_step