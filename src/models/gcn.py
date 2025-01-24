import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any
from flax import linen as nn

class GCNLayer(nn.Module):
    """Graph Convolutional Network Layer implementation.
    
    This layer implements the graph convolution operation as described in Kipf & Welling (2017).
    It performs feature transformation and message passing between nodes in the graph.
    
    Attributes:
        features: Number of output features for the layer.
    """
    features: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, adj: jnp.ndarray) -> jnp.ndarray:
        """Applies graph convolution operation.
        
        Args:
            x: Node feature matrix of shape [num_nodes, in_features]
            adj: Adjacency matrix of shape [num_nodes, num_nodes]
            
        Returns:
            Updated node features of shape [num_nodes, features] after convolution
        """
        # Add self-connections to adjacency matrix
        adj_hat = adj + jnp.eye(adj.shape[0])
        
        # Normalize adjacency matrix
        deg = jnp.sum(adj_hat, axis=1)
        deg_inv_sqrt = jnp.power(deg, -0.5)
        deg_inv_sqrt = jnp.where(jnp.isinf(deg_inv_sqrt), 0., deg_inv_sqrt)
        adj_normalized = deg_inv_sqrt[:, None] * adj_hat * deg_inv_sqrt[None, :]
        
        # Linear transformation
        dense = nn.Dense(features=self.features)
        x = dense(x)
        
        # Graph convolution
        return jnp.matmul(adj_normalized, x)

class MolecularGCN(nn.Module):
    hidden_features: Tuple[int, ...]
    output_features: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, adj: jnp.ndarray) -> jnp.ndarray:
        # Initial feature processing
        for features in self.hidden_features:
            x = GCNLayer(features=features)(x, adj)
            x = nn.relu(x)
        
        # Global pooling (mean of node features)
        x = jnp.mean(x, axis=0)
        
        # Final prediction layer
        x = nn.Dense(features=self.output_features)(x)
        return x

class UncertaintyGCN(nn.Module):
    """GCN with uncertainty estimation using ensemble or dropout"""
    hidden_features: Tuple[int, ...]
    output_features: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
        adj: jnp.ndarray, 
        training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Apply dropout to input features during training
        if training:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
        
        # Feature processing with dropout between layers
        for features in self.hidden_features:
            x = GCNLayer(features=features)(x, adj)
            x = nn.relu(x)
            if training:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
        
        # Global pooling
        x = jnp.mean(x, axis=0)
        
        # Prediction head
        mean = nn.Dense(features=self.output_features)(x)
        log_var = nn.Dense(features=self.output_features)(x)
        
        return mean, jnp.exp(log_var)

def create_training_step(model: nn.Module, learning_rate: float = 1e-3):
    """Create training step function with negative log likelihood loss"""
    
    @jax.jit
    def training_step(
        params: Dict[str, Any],
        batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        rng: jnp.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        def loss_fn(params):
            x, adj, y = batch
            mean, var = model.apply(params, x, adj, training=True, rngs={'dropout': rng})
            nll = 0.5 * jnp.sum(jnp.log(var) + (y - mean)**2 / var)
            return nll
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        new_params = jax.tree_map(
            lambda p, g: p - learning_rate * g, 
            params, 
            grads
        )
        return loss, new_params
    
    return training_step