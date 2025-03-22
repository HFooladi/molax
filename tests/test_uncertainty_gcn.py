import jax
import jax.numpy as jnp
import numpy as np
import pytest
import flax.nnx as nnx

from molax.models.gcn import UncertaintyGCN, UncertaintyGCNConfig


class TestUncertaintyGCN:
    """Tests for the UncertaintyGCN implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Fixed random seed for reproducibility
        self.key = jax.random.PRNGKey(42)
        self.default_key = jax.random.PRNGKey(42)
        self.params_key = jax.random.PRNGKey(12)
        self.dropout_key = jax.random.PRNGKey(123)
        
        # Default test dimensions
        self.num_nodes = 6
        self.in_features = 4
        self.hidden_features = [8, 16]
        self.out_features = 2
        
        # Create test input data
        self.node_features = jax.random.normal(
            self.key, 
            (self.num_nodes, self.in_features)
        )
        
        # Create different adjacency matrices for testing
        
        # 1. Chain structure
        self.chain_adj = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes - 1):
            self.chain_adj[i, i+1] = self.chain_adj[i+1, i] = 1.0
        self.chain_adj = jnp.array(self.chain_adj)
        
        # 2. Fully connected
        self.full_adj = jnp.ones((self.num_nodes, self.num_nodes)) - jnp.eye(self.num_nodes)
        
        # 3. Star structure (node 0 connected to all others)
        self.star_adj = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(1, self.num_nodes):
            self.star_adj[0, i] = self.star_adj[i, 0] = 1.0
        self.star_adj = jnp.array(self.star_adj)

    def test_init(self):
        """Test UncertaintyGCN initialization."""
        rngs = nnx.Rngs(self.default_key, params=self.params_key, dropout=self.dropout_key)
        config = UncertaintyGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            rngs=rngs
        )
        model = UncertaintyGCN(config)
        
        # Check if base model was properly initialized
        assert hasattr(model, 'base_model')
        
        # Check if output heads were properly initialized
        assert hasattr(model, 'mean_head')
        assert hasattr(model, 'var_head')
        assert isinstance(model.mean_head, nnx.Linear)
        assert isinstance(model.var_head, nnx.Linear)
        assert model.mean_head.out_features == self.out_features
        assert model.var_head.out_features == self.out_features
        
        # Test with custom dropout rate and n_heads
        custom_dropout = 0.3
        config_custom = UncertaintyGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            dropout_rate=custom_dropout,
            n_heads=2,
            rngs=rngs
        )
        model_custom = UncertaintyGCN(config_custom)
        # Check that the base model has the custom dropout rate
        assert model_custom.base_model.dropout.rate == custom_dropout

    def test_forward_shape(self):
        """Test that forward pass produces correct output shapes."""
        rngs = nnx.Rngs(self.default_key, params=self.params_key, dropout=self.dropout_key)
        config = UncertaintyGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            rngs=rngs
        )
        model = UncertaintyGCN(config)
        
        # Test with chain adjacency
        mean, variance = model(self.node_features, self.chain_adj)
        assert mean.shape == (self.out_features,)
        assert variance.shape == (self.out_features,)
        
        # Test with fully connected adjacency
        mean, variance = model(self.node_features, self.full_adj)
        assert mean.shape == (self.out_features,)
        assert variance.shape == (self.out_features,)
        
        # Test with star graph
        mean, variance = model(self.node_features, self.star_adj)
        assert mean.shape == (self.out_features,)
        assert variance.shape == (self.out_features,)

    def test_variance_positivity(self):
        """Test that the predicted variance is always positive."""
        rngs = nnx.Rngs(self.default_key, params=self.params_key, dropout=self.dropout_key)
        config = UncertaintyGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            rngs=rngs
        )
        model = UncertaintyGCN(config)
        
        # Test with different inputs
        for adj in [self.chain_adj, self.full_adj, self.star_adj]:
            _, variance = model(self.node_features, adj)
            # Variance should always be positive
            assert jnp.all(variance > 0)

    def test_training_mode(self):
        """Test that dropout is applied in training mode but not in inference mode."""
        rngs = nnx.Rngs(self.default_key, params=self.params_key, dropout=self.dropout_key)
        config = UncertaintyGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            dropout_rate=0.9,  # Very high dropout for testing
            rngs=rngs
        )
        
        model = UncertaintyGCN(config)
        
        # In inference mode (default)
        mean_inference, var_inference = model(self.node_features, self.chain_adj)
        
        # In training mode
        mean_training, var_training = model(self.node_features, self.chain_adj, training=True)
        
        # Outputs should be different due to dropout
        assert not jnp.allclose(mean_inference, mean_training)
        assert not jnp.allclose(var_inference, var_training)
        
        # Multiple forward passes in inference mode should be identical
        mean_inference2, var_inference2 = model(self.node_features, self.chain_adj)
        np.testing.assert_allclose(mean_inference, mean_inference2)
        np.testing.assert_allclose(var_inference, var_inference2)

    def test_deterministic_output(self):
        """Test that the model produces deterministic outputs with same seed."""
        # Same seed should produce same results
        rngs1 = nnx.Rngs(self.default_key, params=self.params_key, dropout=self.dropout_key)
        config1 = UncertaintyGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            rngs=rngs1
        )
        model1 = UncertaintyGCN(config1)
        
        rngs2 = nnx.Rngs(self.default_key, params=self.params_key, dropout=self.dropout_key)
        config2 = UncertaintyGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            rngs=rngs2
        )
        model2 = UncertaintyGCN(config2)
        
        # Outputs should be identical with same initialization and inputs
        mean1, var1 = model1(self.node_features, self.chain_adj)
        mean2, var2 = model2(self.node_features, self.chain_adj)
        
        np.testing.assert_allclose(mean1, mean2, rtol=1e-5)
        np.testing.assert_allclose(var1, var2, rtol=1e-5)
        
        # Models with different seeds should give different outputs
        params_key = jax.random.PRNGKey(43)
        dropout_key = jax.random.PRNGKey(124)
        rngs3 = nnx.Rngs(self.default_key, params=params_key, dropout=dropout_key)
        config3 = UncertaintyGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            rngs=rngs3
        )
        model3 = UncertaintyGCN(config3)
        
        mean3, var3 = model3(self.node_features, self.chain_adj)
        # Different parameter initialization should produce different results
        assert not jnp.allclose(mean1, mean3)
        assert not jnp.allclose(var1, var3)

    def test_uncertainty_correlation(self):
        """Test that uncertainty increases for out-of-distribution inputs."""
        rngs = nnx.Rngs(self.default_key, params=self.params_key, dropout=self.dropout_key)
        config = UncertaintyGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            rngs=rngs
        )
        model = UncertaintyGCN(config)
        
        # Normal input
        _, var_normal = model(self.node_features, self.chain_adj)
        
        # "Extreme" input (high magnitude values)
        extreme_features = self.node_features * 10.0
        _, var_extreme = model(extreme_features, self.chain_adj)
        
        # Uncertainty should generally increase for out-of-distribution inputs
        # This is a probabilistic test, so we check the mean uncertainty
        assert jnp.mean(var_extreme) >= jnp.mean(var_normal) 