import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from molax.models.gcn import GCNLayer, GCNLayerConfig


class TestGCNLayer:
    """Tests for the GCNLayer implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Fixed random seed for reproducibility
        self.key = jax.random.PRNGKey(42)
        self.rngs = nnx.Rngs(params=self.key)

        # Default test dimensions
        self.num_nodes = 5
        self.in_features = 3
        self.out_features = 4

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

        # 3. Isolated node (node 0 is disconnected)
        self.isolated_adj = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(1, self.num_nodes - 1):
            self.isolated_adj[i, i + 1] = self.isolated_adj[i + 1, i] = 1.0
        self.isolated_adj = jnp.array(self.isolated_adj)

    def test_init(self):
        """Test GCNLayer initialization."""
        config = GCNLayerConfig(
            in_features=self.in_features, out_features=self.out_features, rngs=self.rngs
        )
        layer = GCNLayer(config)

        # Check if the dense layer was properly initialized
        assert hasattr(layer, "dense")
        assert isinstance(layer.dense, nnx.Linear)
        assert layer.dense.in_features == self.in_features
        assert layer.dense.out_features == self.out_features

    def test_forward_shape(self):
        """Test that forward pass produces correct output shapes."""
        config = GCNLayerConfig(
            in_features=self.in_features, out_features=self.out_features, rngs=self.rngs
        )
        layer = GCNLayer(config)

        # Test with chain adjacency
        output = layer(self.node_features, self.chain_adj)
        assert output.shape == (self.num_nodes, self.out_features)

        # Test with fully connected adjacency
        output = layer(self.node_features, self.full_adj)
        assert output.shape == (self.num_nodes, self.out_features)

        # Test with isolated node
        output = layer(self.node_features, self.isolated_adj)
        assert output.shape == (self.num_nodes, self.out_features)

    def test_isolated_nodes(self):
        """Test that isolated nodes are handled properly."""
        config = GCNLayerConfig(
            in_features=self.in_features, out_features=self.out_features, rngs=self.rngs
        )
        layer = GCNLayer(config)

        # Get output with isolated node
        output = layer(self.node_features, self.isolated_adj)

        # The first node is isolated, its output should not be NaN or inf
        assert not jnp.any(jnp.isnan(output[0]))
        assert not jnp.any(jnp.isinf(output[0]))

        # The isolated node should receive its own features transformed
        # Test that self-loop is properly added
        dense_output = layer.dense(self.node_features)
        # For isolated node, output should be its own transformed features
        np.testing.assert_allclose(output[0], dense_output[0], rtol=1e-5)

    def test_normalization(self):
        """Test the normalization of adjacency matrix (D^-1/2 A D^-1/2)."""
        # Test with fully connected graph (all nodes have same degree)
        adj = self.full_adj
        adj_hat = adj + jnp.eye(adj.shape[0])
        deg = jnp.sum(adj_hat, axis=1)

        # All nodes should have degree = num_nodes (since fully connected)
        assert jnp.all(deg == self.num_nodes)

        # With normalization, each edge weight should be 1/num_nodes
        deg_inv_sqrt = jnp.power(deg, -0.5)
        adj_normalized = deg_inv_sqrt[:, None] * adj_hat * deg_inv_sqrt[None, :]

        expected_value = 1.0 / self.num_nodes
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                np.testing.assert_almost_equal(
                    adj_normalized[i, j], expected_value, decimal=5
                )

    def test_bias_options(self):
        """Test layer with and without bias."""
        # With bias
        config_with_bias = GCNLayerConfig(
            in_features=self.in_features,
            out_features=self.out_features,
            use_bias=True,
            rngs=self.rngs,
        )
        layer_with_bias = GCNLayer(config_with_bias)
        assert layer_with_bias.dense.use_bias

        # Without bias
        config_no_bias = GCNLayerConfig(
            in_features=self.in_features,
            out_features=self.out_features,
            use_bias=False,
            rngs=self.rngs,
        )
        layer_no_bias = GCNLayer(config_no_bias)
        assert not layer_no_bias.dense.use_bias

        # Outputs should be different
        output_with_bias = layer_with_bias(self.node_features, self.chain_adj)
        output_no_bias = layer_no_bias(self.node_features, self.chain_adj)

        # Assert they're not the same (should differ due to bias terms)
        assert not jnp.allclose(output_with_bias, output_no_bias)

    def test_deterministic_output(self):
        """Test that the layer produces deterministic outputs with same seed."""
        config1 = GCNLayerConfig(
            in_features=self.in_features,
            out_features=self.out_features,
            rngs=nnx.Rngs(params=jax.random.PRNGKey(42)),
        )
        layer1 = GCNLayer(config1)

        config2 = GCNLayerConfig(
            in_features=self.in_features,
            out_features=self.out_features,
            rngs=nnx.Rngs(params=jax.random.PRNGKey(42)),
        )
        layer2 = GCNLayer(config2)

        output1 = layer1(self.node_features, self.chain_adj)
        output2 = layer2(self.node_features, self.chain_adj)

        # With same seed, outputs should be identical
        np.testing.assert_allclose(output1, output2, rtol=1e-5)

        # With different seed, outputs should differ
        config3 = GCNLayerConfig(
            in_features=self.in_features,
            out_features=self.out_features,
            rngs=nnx.Rngs(params=jax.random.PRNGKey(43)),
        )
        layer3 = GCNLayer(config3)
        output3 = layer3(self.node_features, self.chain_adj)

        # Outputs should differ (very unlikely to be the same by chance)
        assert not jnp.allclose(output1, output3)
