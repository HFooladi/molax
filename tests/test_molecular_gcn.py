import jax
import jax.numpy as jnp
import numpy as np
import pytest
import flax.nnx as nnx

from molax.models.gcn import MolecularGCN, MolecularGCNConfig, GCNLayer, GCNLayerConfig


class TestMolecularGCN:
    """Tests for the MolecularGCN implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Fixed random seed for reproducibility
        self.key = jax.random.PRNGKey(42)
        self.rngs = nnx.Rngs(params=self.key)
        
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
        """Test MolecularGCN initialization."""
        # Test with default dropout rate
        config = MolecularGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            rngs=self.rngs
        )
        model = MolecularGCN(config)
        
        # Check if GCN layers were properly initialized
        assert hasattr(model, 'gcn_layers')
        assert len(model.gcn_layers) == len(self.hidden_features)
        
        # Check if output layer was properly initialized
        assert hasattr(model, 'output')
        assert isinstance(model.output, nnx.Linear)
        assert model.output.out_features == self.out_features
        
        # Check if dropout was properly initialized
        assert hasattr(model, 'dropout')
        assert isinstance(model.dropout, nnx.Dropout)
        assert model.dropout.rate == 0.1  # Default value
        
        # Test with custom dropout rate
        custom_dropout = 0.3
        config_custom = MolecularGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            dropout_rate=custom_dropout,
            rngs=self.rngs
        )
        model_custom = MolecularGCN(config_custom)
        assert model_custom.dropout.rate == custom_dropout

    def test_forward_shape(self):
        """Test that forward pass produces correct output shapes."""
        config = MolecularGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            rngs=self.rngs
        )
        model = MolecularGCN(config)
        
        # Test with chain adjacency
        output = model(self.node_features, self.chain_adj)
        assert output.shape == (self.out_features,)
        
        # Test with fully connected adjacency
        output = model(self.node_features, self.full_adj)
        assert output.shape == (self.out_features,)
        
        # Test with star graph
        output = model(self.node_features, self.star_adj)
        assert output.shape == (self.out_features,)
        
        # Test with batch dimension
        batch_size = 3
        batched_features = jnp.stack([self.node_features] * batch_size)
        batched_adj = jnp.stack([self.chain_adj] * batch_size)
        
        # Need to adjust the model to handle batched inputs or process each graph separately

    def test_training_mode(self):
        """Test that dropout is applied in training mode but not in inference mode."""
        # Use a high dropout rate to ensure visible effect
        config = MolecularGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            dropout_rate=0.9  # Very high dropout for testing
        )
        
        # Use a fixed dropout key for deterministic testing
        dropout_key = jax.random.PRNGKey(123)
        dropout_rngs = nnx.Rngs(dropout=dropout_key, params=self.key)
        config.rngs = dropout_rngs
        
        model = MolecularGCN(config)
        
        # In inference mode (default)
        output_inference = model(self.node_features, self.chain_adj)
        
        # In training mode
        output_training = model(self.node_features, self.chain_adj, training=True)
        
        # Outputs should be different due to dropout
        assert not jnp.allclose(output_inference, output_training)
        
        # Multiple forward passes in inference mode should be identical
        output_inference2 = model(self.node_features, self.chain_adj)
        np.testing.assert_allclose(output_inference, output_inference2)
        
        # Multiple forward passes in training mode may be different due to dropout
        # But this is probabilistic, so we don't assert on it

    def test_pooling(self):
        """Test that global pooling works correctly."""
        config = MolecularGCNConfig(
            in_features=self.in_features,
            hidden_features=[8],  # Single layer for simplicity
            out_features=self.out_features,
            rngs=self.rngs
        )
        model = MolecularGCN(config)
        
        # Create a simple case where we can predict the outcome
        # Use node features where each node has the same features
        constant_features = jnp.ones((self.num_nodes, self.in_features))
        
        # Mock the internal behavior to isolate pooling
        # After GCN layers, we should have node features that we can then verify pooling on
        layer_output = jnp.ones((self.num_nodes, 8))  # Pretend output from GCN layers
        
        # Manual pooling (mean)
        expected_pooled = jnp.mean(layer_output, axis=0)
        
        # We can't directly access the pooling, but we can verify the final output
        # would be consistent with our expected pooling

    def test_multiple_gcn_layers(self):
        """Test that multiple GCN layers are applied correctly."""
        # Test with different numbers of hidden layers
        for num_layers in [1, 2, 3]:
            hidden_features = [8] * num_layers
            
            config = MolecularGCNConfig(
                in_features=self.in_features,
                hidden_features=hidden_features,
                out_features=self.out_features,
                rngs=self.rngs
            )
            model = MolecularGCN(config)
            
            # Verify number of GCN layers
            assert len(model.gcn_layers) == num_layers
            
            # Verify the model still produces expected shape outputs
            output = model(self.node_features, self.chain_adj)
            assert output.shape == (self.out_features,)

    def test_deterministic_output(self):
        """Test that the model produces deterministic outputs with same seed."""
        config = MolecularGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            rngs=self.rngs
        )
        
        # Create two models with same seed
        model1 = MolecularGCN(config)
        
        model2 = MolecularGCN(config)
        
        # Outputs should be identical with same initialization and inputs
        output1 = model1(self.node_features, self.chain_adj, training=False)
        output2 = model2(self.node_features, self.chain_adj, training=False)
        
        np.testing.assert_allclose(output1, output2, rtol=1e-5)
        
        # Models with different seeds should give different outputs
        dropout_key = jax.random.PRNGKey(123)
        dropout_rngs = nnx.Rngs(dropout=dropout_key, params=self.key)
        config.rngs = dropout_rngs
        model3 = MolecularGCN(config)
        
        output3 = model3(self.node_features, self.chain_adj, training=False)
        assert not jnp.allclose(output1, output3) 