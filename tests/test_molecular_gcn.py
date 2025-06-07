import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from molax.models.gcn import MolecularGCN, MolecularGCNConfig


class TestMolecularGCN:
    """Tests for the MolecularGCN implementation."""

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
            self.key, (self.num_nodes, self.in_features)
        )

        # Create different adjacency matrices for testing

        # 1. Chain structure
        self.chain_adj = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes - 1):
            self.chain_adj[i, i + 1] = self.chain_adj[i + 1, i] = 1.0
        self.chain_adj = jnp.array(self.chain_adj)

        # 2. Fully connected
        self.full_adj = jnp.ones((self.num_nodes, self.num_nodes)) - jnp.eye(
            self.num_nodes
        )

        # 3. Star structure (node 0 connected to all others)
        self.star_adj = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(1, self.num_nodes):
            self.star_adj[0, i] = self.star_adj[i, 0] = 1.0
        self.star_adj = jnp.array(self.star_adj)

    def test_init(self):
        """Test MolecularGCN initialization."""
        # Test with default dropout rate
        rngs = nnx.Rngs(
            self.default_key, params=self.params_key, dropout=self.dropout_key
        )
        config = MolecularGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            rngs=rngs,
        )
        model = MolecularGCN(config)

        # Check if GCN layers were properly initialized
        assert hasattr(model, "gcn_layers")
        assert len(model.gcn_layers) == len(self.hidden_features)

        # Check if output layer was properly initialized
        assert hasattr(model, "output")
        assert isinstance(model.output, nnx.Linear)
        assert model.output.out_features == self.out_features

        # Check if dropout was properly initialized
        assert hasattr(model, "dropout")
        assert isinstance(model.dropout, nnx.Dropout)
        assert model.dropout.rate == 0.1  # Default value

        # Test with custom dropout rate
        custom_dropout = 0.3
        config_custom = MolecularGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            dropout_rate=custom_dropout,
            rngs=rngs,
        )
        model_custom = MolecularGCN(config_custom)
        assert model_custom.dropout.rate == custom_dropout

    def test_forward_shape(self):
        """Test that forward pass produces correct output shapes."""
        rngs = nnx.Rngs(
            self.default_key, params=self.params_key, dropout=self.dropout_key
        )
        config = MolecularGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            rngs=rngs,
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

    def test_training_mode(self):
        """Test that dropout is applied in training mode but not in inference mode."""
        rngs = nnx.Rngs(
            self.default_key, params=self.params_key, dropout=self.dropout_key
        )
        # Use a high dropout rate to ensure visible effect
        config = MolecularGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            dropout_rate=0.9,  # Very high dropout for testing
            rngs=rngs,
        )

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

    def test_multiple_gcn_layers(self):
        """Test that multiple GCN layers are applied correctly."""
        # Test with different numbers of hidden layers
        for num_layers in [1, 2, 3]:
            hidden_features = [8] * num_layers

            rngs = nnx.Rngs(
                self.default_key, params=self.params_key, dropout=self.dropout_key
            )
            config = MolecularGCNConfig(
                in_features=self.in_features,
                hidden_features=hidden_features,
                out_features=self.out_features,
                rngs=rngs,
            )
            model = MolecularGCN(config)

            # Verify number of GCN layers
            assert len(model.gcn_layers) == num_layers

            # Verify the model still produces expected shape outputs
            output = model(self.node_features, self.chain_adj)
            assert output.shape == (self.out_features,)

    def test_deterministic_output(self):
        """Test that the model produces deterministic outputs with same seed."""
        # Same seed should produce same results
        rngs1 = nnx.Rngs(
            self.default_key, params=self.params_key, dropout=self.dropout_key
        )
        config1 = MolecularGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            rngs=rngs1,
        )
        model1 = MolecularGCN(config1)

        rngs2 = nnx.Rngs(
            self.default_key, params=self.params_key, dropout=self.dropout_key
        )
        config2 = MolecularGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            rngs=rngs2,
        )
        model2 = MolecularGCN(config2)

        # Outputs should be identical with same initialization and inputs
        output1 = model1(self.node_features, self.chain_adj)
        output2 = model2(self.node_features, self.chain_adj)

        np.testing.assert_allclose(output1, output2, rtol=1e-5)

        # Models with different seeds should give different outputs
        # different seeds for params and dropout
        params_key = jax.random.PRNGKey(43)
        dropout_key = jax.random.PRNGKey(124)
        rngs3 = nnx.Rngs(self.default_key, params=params_key, dropout=dropout_key)
        config3 = MolecularGCNConfig(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            rngs=rngs3,
        )
        model3 = MolecularGCN(config3)

        output3 = model3(self.node_features, self.chain_adj)
        # Different parameter initialization should produce different results
        assert not jnp.allclose(output1, output3)
