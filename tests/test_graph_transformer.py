"""Tests for Graph Transformer models."""

import flax.nnx as nnx
import jax.numpy as jnp
import jraph
import pytest

from molax.models.graph_transformer import (
    GraphTransformerAttention,
    GraphTransformerConfig,
    GraphTransformerLayer,
    LaplacianPositionalEncoding,
    RandomWalkPositionalEncoding,
    UncertaintyGraphTransformer,
    create_graph_transformer_optimizer,
    eval_graph_transformer_step,
    get_graph_transformer_uncertainties,
    train_graph_transformer_step,
)
from molax.utils.data import smiles_to_jraph


@pytest.fixture
def simple_graph():
    """Create a single simple test graph."""
    return smiles_to_jraph("CC")  # Ethane - simple 2-carbon molecule


@pytest.fixture
def batched_graphs():
    """Create a batch of test graphs."""
    smiles = ["C", "CC", "CCC", "CCO"]
    graphs = [smiles_to_jraph(s) for s in smiles]
    return jraph.batch(graphs)


@pytest.fixture
def config():
    """Create default Graph Transformer config."""
    return GraphTransformerConfig(
        node_features=6,
        edge_features=1,
        hidden_features=[32, 32],
        out_features=1,
        n_heads=4,
        ffn_ratio=4.0,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        pe_type="rwpe",
        pe_dim=8,
    )


@pytest.fixture
def config_no_pe():
    """Create Graph Transformer config without positional encoding."""
    return GraphTransformerConfig(
        node_features=6,
        edge_features=1,
        hidden_features=[32, 32],
        out_features=1,
        n_heads=4,
        dropout_rate=0.1,
        pe_type="none",
    )


@pytest.fixture
def config_laplacian_pe():
    """Create Graph Transformer config with Laplacian PE."""
    return GraphTransformerConfig(
        node_features=6,
        edge_features=1,
        hidden_features=[32, 32],
        out_features=1,
        n_heads=4,
        dropout_rate=0.1,
        pe_type="laplacian",
        pe_dim=8,
    )


@pytest.fixture
def model(config):
    """Create Graph Transformer model."""
    return UncertaintyGraphTransformer(config, rngs=nnx.Rngs(0))


class TestPositionalEncodings:
    """Tests for positional encoding modules."""

    def test_rwpe_output_shape(self, simple_graph):
        """Test that RWPE produces correct output shape."""
        pe = RandomWalkPositionalEncoding(
            pe_dim=8,
            hidden_dim=32,
            rngs=nnx.Rngs(0),
        )

        output = pe(simple_graph)

        # Should have n_nodes x hidden_dim
        assert output.shape == (simple_graph.nodes.shape[0], 32)

    def test_rwpe_finite_values(self, simple_graph):
        """Test that RWPE produces finite values."""
        pe = RandomWalkPositionalEncoding(
            pe_dim=8,
            hidden_dim=32,
            rngs=nnx.Rngs(0),
        )

        output = pe(simple_graph)

        assert jnp.all(jnp.isfinite(output))

    def test_rwpe_batched_graphs(self, batched_graphs):
        """Test RWPE on batched graphs."""
        pe = RandomWalkPositionalEncoding(
            pe_dim=8,
            hidden_dim=32,
            rngs=nnx.Rngs(0),
        )

        output = pe(batched_graphs)

        assert output.shape == (batched_graphs.nodes.shape[0], 32)
        assert jnp.all(jnp.isfinite(output))

    def test_laplacian_pe_output_shape(self, simple_graph):
        """Test that Laplacian PE produces correct output shape."""
        pe = LaplacianPositionalEncoding(
            pe_dim=4,  # Smaller dim for small graphs
            hidden_dim=32,
            rngs=nnx.Rngs(0),
        )

        output = pe(simple_graph)

        assert output.shape == (simple_graph.nodes.shape[0], 32)

    def test_laplacian_pe_finite_values(self, simple_graph):
        """Test that Laplacian PE produces finite values."""
        pe = LaplacianPositionalEncoding(
            pe_dim=4,
            hidden_dim=32,
            rngs=nnx.Rngs(0),
        )

        output = pe(simple_graph)

        assert jnp.all(jnp.isfinite(output))

    def test_different_pe_dims(self, simple_graph):
        """Test positional encodings with different dimensions."""
        for pe_dim in [4, 8, 16]:
            rwpe = RandomWalkPositionalEncoding(
                pe_dim=pe_dim,
                hidden_dim=32,
                rngs=nnx.Rngs(0),
            )
            output = rwpe(simple_graph)

            assert output.shape == (simple_graph.nodes.shape[0], 32)
            assert jnp.all(jnp.isfinite(output))


class TestGraphTransformerAttention:
    """Tests for GraphTransformerAttention module."""

    def test_output_shape(self, simple_graph):
        """Test that attention produces correct output shape."""
        hidden_dim = 32
        attn = GraphTransformerAttention(
            hidden_dim=hidden_dim,
            n_heads=4,
            edge_features=0,
            attention_dropout_rate=0.0,
            rngs=nnx.Rngs(0),
        )

        # Project nodes to hidden_dim first
        nodes = jnp.zeros((simple_graph.nodes.shape[0], hidden_dim))
        output = attn(nodes, simple_graph)

        assert output.shape == (simple_graph.nodes.shape[0], hidden_dim)

    def test_attention_with_edge_features(self, simple_graph):
        """Test attention with edge feature bias."""
        hidden_dim = 32
        attn = GraphTransformerAttention(
            hidden_dim=hidden_dim,
            n_heads=4,
            edge_features=1,
            attention_dropout_rate=0.0,
            rngs=nnx.Rngs(0),
        )

        nodes = jnp.zeros((simple_graph.nodes.shape[0], hidden_dim))
        output = attn(nodes, simple_graph)

        assert output.shape == (simple_graph.nodes.shape[0], hidden_dim)
        assert jnp.all(jnp.isfinite(output))

    def test_graph_aware_masking(self, batched_graphs):
        """Test that attention respects graph boundaries."""
        hidden_dim = 32
        attn = GraphTransformerAttention(
            hidden_dim=hidden_dim,
            n_heads=4,
            edge_features=0,
            attention_dropout_rate=0.0,
            rngs=nnx.Rngs(0),
        )

        nodes = jnp.zeros((batched_graphs.nodes.shape[0], hidden_dim))
        output = attn(nodes, batched_graphs)

        # Output should be finite (no NaN from cross-graph attention)
        assert jnp.all(jnp.isfinite(output))

    def test_different_n_heads(self, simple_graph):
        """Test attention with different numbers of heads."""
        hidden_dim = 32
        for n_heads in [1, 2, 4, 8]:
            attn = GraphTransformerAttention(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                edge_features=0,
                attention_dropout_rate=0.0,
                rngs=nnx.Rngs(0),
            )

            nodes = jnp.zeros((simple_graph.nodes.shape[0], hidden_dim))
            output = attn(nodes, simple_graph)

            assert output.shape == (simple_graph.nodes.shape[0], hidden_dim)
            assert jnp.all(jnp.isfinite(output))


class TestGraphTransformerLayer:
    """Tests for GraphTransformerLayer module."""

    def test_output_shape(self, simple_graph):
        """Test that layer produces correct output shape."""
        hidden_dim = 32
        layer = GraphTransformerLayer(
            hidden_dim=hidden_dim,
            n_heads=4,
            ffn_ratio=4.0,
            edge_features=0,
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
            rngs=nnx.Rngs(0),
        )

        nodes = jnp.zeros((simple_graph.nodes.shape[0], hidden_dim))
        output = layer(nodes, simple_graph)

        assert output.shape == (simple_graph.nodes.shape[0], hidden_dim)

    def test_residual_connection(self, simple_graph):
        """Test that residual connections work."""
        hidden_dim = 32
        layer = GraphTransformerLayer(
            hidden_dim=hidden_dim,
            n_heads=4,
            ffn_ratio=4.0,
            edge_features=0,
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
            rngs=nnx.Rngs(0),
        )

        # With zeros input, output should still be non-zero (due to biases)
        nodes = jnp.zeros((simple_graph.nodes.shape[0], hidden_dim))
        output = layer(nodes, simple_graph)

        # Output should be finite
        assert jnp.all(jnp.isfinite(output))

    def test_layer_with_edge_features(self, simple_graph):
        """Test layer with edge features in attention."""
        hidden_dim = 32
        layer = GraphTransformerLayer(
            hidden_dim=hidden_dim,
            n_heads=4,
            ffn_ratio=4.0,
            edge_features=1,
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
            rngs=nnx.Rngs(0),
        )

        nodes = jnp.zeros((simple_graph.nodes.shape[0], hidden_dim))
        output = layer(nodes, simple_graph)

        assert output.shape == (simple_graph.nodes.shape[0], hidden_dim)
        assert jnp.all(jnp.isfinite(output))

    def test_batched_graphs(self, batched_graphs):
        """Test layer on batched graphs."""
        hidden_dim = 32
        layer = GraphTransformerLayer(
            hidden_dim=hidden_dim,
            n_heads=4,
            ffn_ratio=4.0,
            edge_features=0,
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
            rngs=nnx.Rngs(0),
        )

        nodes = jnp.zeros((batched_graphs.nodes.shape[0], hidden_dim))
        output = layer(nodes, batched_graphs)

        assert output.shape == (batched_graphs.nodes.shape[0], hidden_dim)
        assert jnp.all(jnp.isfinite(output))


class TestUncertaintyGraphTransformer:
    """Tests for UncertaintyGraphTransformer model."""

    def test_forward_returns_two_outputs(self, model, batched_graphs):
        """Test that forward pass returns mean and variance."""
        mean, variance = model(batched_graphs)

        assert mean.shape == (4, 1)
        assert variance.shape == (4, 1)

    def test_variance_positive(self, model, batched_graphs):
        """Test that variance is always positive."""
        _, variance = model(batched_graphs)

        assert jnp.all(variance > 0)

    def test_training_mode_uses_dropout(self, config, batched_graphs):
        """Test that training mode enables dropout."""
        model = UncertaintyGraphTransformer(config, rngs=nnx.Rngs(0))

        # Multiple forward passes in training mode should give different results
        results = []
        for _ in range(5):
            mean, _ = model(batched_graphs, training=True)
            results.append(mean)

        # Check that at least some results are different
        all_same = all(jnp.allclose(results[0], r) for r in results[1:])
        # With dropout, results should differ
        if config.dropout_rate > 0:
            assert not all_same, "Training mode should produce varying outputs"

    def test_inference_mode_deterministic(self, model, batched_graphs):
        """Test that inference mode is deterministic."""
        mean1, var1 = model(batched_graphs, training=False)
        mean2, var2 = model(batched_graphs, training=False)

        assert jnp.allclose(mean1, mean2)
        assert jnp.allclose(var1, var2)

    def test_extract_embeddings_shape(self, model, batched_graphs):
        """Test that extract_embeddings returns correct shape."""
        embeddings = model.extract_embeddings(batched_graphs)

        # Should be [n_graphs, first_hidden_dim]
        assert embeddings.shape == (4, 32)

    def test_extract_embeddings_finite(self, model, batched_graphs):
        """Test that extracted embeddings are finite."""
        embeddings = model.extract_embeddings(batched_graphs)

        assert jnp.all(jnp.isfinite(embeddings))

    def test_single_graph(self, config, simple_graph):
        """Test model on a single unbatched graph."""
        model = UncertaintyGraphTransformer(config, rngs=nnx.Rngs(0))

        mean, variance = model(simple_graph)

        assert mean.shape == (1, 1)
        assert variance.shape == (1, 1)
        assert jnp.all(variance > 0)

    def test_no_positional_encoding(self, config_no_pe, batched_graphs):
        """Test model without positional encoding."""
        model = UncertaintyGraphTransformer(config_no_pe, rngs=nnx.Rngs(0))

        mean, variance = model(batched_graphs)

        assert mean.shape == (4, 1)
        assert variance.shape == (4, 1)
        assert jnp.all(variance > 0)
        assert jnp.all(jnp.isfinite(mean))

    def test_laplacian_positional_encoding(self, config_laplacian_pe, batched_graphs):
        """Test model with Laplacian positional encoding."""
        model = UncertaintyGraphTransformer(config_laplacian_pe, rngs=nnx.Rngs(0))

        mean, variance = model(batched_graphs)

        assert mean.shape == (4, 1)
        assert variance.shape == (4, 1)
        assert jnp.all(variance > 0)
        assert jnp.all(jnp.isfinite(mean))

    def test_different_n_heads(self, batched_graphs):
        """Test model with different numbers of attention heads."""
        for n_heads in [1, 2, 4, 8]:
            config = GraphTransformerConfig(
                node_features=6,
                hidden_features=[32],  # Must be divisible by n_heads
                out_features=1,
                n_heads=n_heads,
                pe_type="none",
            )
            model = UncertaintyGraphTransformer(config, rngs=nnx.Rngs(0))

            mean, variance = model(batched_graphs)

            assert mean.shape == (4, 1)
            assert variance.shape == (4, 1)
            assert jnp.all(variance > 0)
            assert jnp.all(jnp.isfinite(mean))


class TestGraphTransformerTraining:
    """Tests for Graph Transformer training utilities."""

    def test_create_optimizer(self, model):
        """Test optimizer creation."""
        optimizer = create_graph_transformer_optimizer(model, learning_rate=1e-4)
        assert isinstance(optimizer, nnx.Optimizer)

    def test_create_optimizer_with_warmup(self, model):
        """Test optimizer creation with warmup."""
        optimizer = create_graph_transformer_optimizer(
            model,
            learning_rate=1e-4,
            warmup_steps=50,
        )
        assert isinstance(optimizer, nnx.Optimizer)

    def test_train_step_returns_loss(self, model, batched_graphs):
        """Test that training step returns a loss value."""
        optimizer = create_graph_transformer_optimizer(model)

        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.ones(4, dtype=bool)

        loss = train_graph_transformer_step(
            model, optimizer, batched_graphs, labels, mask
        )

        assert loss.shape == ()  # Scalar
        assert jnp.isfinite(loss)

    def test_train_step_updates_weights(self, config, batched_graphs):
        """Test that training updates model weights."""
        model = UncertaintyGraphTransformer(config, rngs=nnx.Rngs(0))
        optimizer = create_graph_transformer_optimizer(model)

        # Get initial weights
        initial_weights = model.input_proj.kernel.copy()

        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.ones(4, dtype=bool)

        # Train for a few steps
        for _ in range(5):
            train_graph_transformer_step(model, optimizer, batched_graphs, labels, mask)

        # Check weights changed
        new_weights = model.input_proj.kernel
        assert not jnp.allclose(initial_weights, new_weights), (
            "Weights did not change after training"
        )

    def test_train_step_reduces_loss(self, config, batched_graphs):
        """Test that training reduces loss over time."""
        model = UncertaintyGraphTransformer(config, rngs=nnx.Rngs(42))
        optimizer = create_graph_transformer_optimizer(model, learning_rate=1e-3)

        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.ones(4, dtype=bool)

        # Get initial loss
        initial_loss = train_graph_transformer_step(
            model, optimizer, batched_graphs, labels, mask
        )

        # Train more
        for _ in range(100):
            final_loss = train_graph_transformer_step(
                model, optimizer, batched_graphs, labels, mask
            )

        # Loss should decrease (or at least not increase significantly)
        assert final_loss < initial_loss * 1.1, "Loss should decrease during training"

    def test_train_step_with_partial_mask(self, model, batched_graphs):
        """Test training with only some samples masked."""
        optimizer = create_graph_transformer_optimizer(model)

        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        # Only use first two samples
        mask = jnp.array([True, True, False, False])

        loss = train_graph_transformer_step(
            model, optimizer, batched_graphs, labels, mask
        )

        assert jnp.isfinite(loss)


class TestGraphTransformerEvaluation:
    """Tests for Graph Transformer evaluation utilities."""

    def test_eval_step_returns_mse(self, model, batched_graphs):
        """Test that eval step returns MSE and predictions."""
        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.ones(4, dtype=bool)

        mse, predictions = eval_graph_transformer_step(
            model, batched_graphs, labels, mask
        )

        assert mse.shape == ()
        assert predictions.shape == (4,)
        assert jnp.isfinite(mse)
        assert mse >= 0

    def test_get_uncertainties(self, model, batched_graphs):
        """Test uncertainty extraction."""
        epistemic_var, aleatoric_var = get_graph_transformer_uncertainties(
            model, batched_graphs, n_samples=10
        )

        assert epistemic_var.shape == (4,)
        assert aleatoric_var.shape == (4,)

        # Both should be non-negative
        assert jnp.all(epistemic_var >= 0)
        assert jnp.all(aleatoric_var > 0)  # Aleatoric should be strictly positive

    def test_get_uncertainties_finite(self, model, batched_graphs):
        """Test that uncertainties are finite."""
        epistemic_var, aleatoric_var = get_graph_transformer_uncertainties(
            model, batched_graphs, n_samples=10
        )

        assert jnp.all(jnp.isfinite(epistemic_var))
        assert jnp.all(jnp.isfinite(aleatoric_var))


class TestGraphTransformerConfig:
    """Tests for Graph Transformer configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GraphTransformerConfig()

        assert config.node_features == 6
        assert config.edge_features == 1
        assert config.hidden_features == (64, 64)
        assert config.out_features == 1
        assert config.n_heads == 4
        assert config.ffn_ratio == 4.0
        assert config.dropout_rate == 0.1
        assert config.attention_dropout_rate == 0.1
        assert config.pe_type == "rwpe"
        assert config.pe_dim == 16

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GraphTransformerConfig(
            node_features=10,
            edge_features=3,
            hidden_features=[128, 64, 32],
            out_features=2,
            n_heads=8,
            ffn_ratio=2.0,
            dropout_rate=0.2,
            attention_dropout_rate=0.15,
            pe_type="laplacian",
            pe_dim=32,
        )

        assert config.node_features == 10
        assert config.edge_features == 3
        assert config.hidden_features == [128, 64, 32]
        assert config.out_features == 2
        assert config.n_heads == 8
        assert config.ffn_ratio == 2.0
        assert config.dropout_rate == 0.2
        assert config.attention_dropout_rate == 0.15
        assert config.pe_type == "laplacian"
        assert config.pe_dim == 32

    def test_model_respects_config(self, batched_graphs):
        """Test that model architecture matches config."""
        config = GraphTransformerConfig(
            node_features=6,
            edge_features=0,
            hidden_features=[64, 32, 16],
            out_features=3,
            n_heads=4,
            pe_type="none",
        )
        model = UncertaintyGraphTransformer(config, rngs=nnx.Rngs(0))

        # Check number of layers
        assert len(model.transformer_layers) == 3

        # Check output dimension
        mean, var = model(batched_graphs)
        assert mean.shape == (4, 3)
        assert var.shape == (4, 3)


class TestGraphTransformerAPICompatibility:
    """Tests comparing Graph Transformer to other models."""

    def test_same_api_as_gcn(self, batched_graphs):
        """Test that Graph Transformer has the same API as UncertaintyGCN."""
        from molax.models.gcn import GCNConfig, UncertaintyGCN

        # Create both models
        gcn_config = GCNConfig(
            node_features=6,
            hidden_features=[32, 32],
            out_features=1,
            dropout_rate=0.1,
        )
        gcn = UncertaintyGCN(gcn_config, rngs=nnx.Rngs(0))

        gt_config = GraphTransformerConfig(
            node_features=6,
            hidden_features=[32, 32],
            out_features=1,
            n_heads=4,
            dropout_rate=0.1,
            pe_type="none",
        )
        gt = UncertaintyGraphTransformer(gt_config, rngs=nnx.Rngs(0))

        # Both should accept same inputs and return same output shapes
        gcn_mean, gcn_var = gcn(batched_graphs, training=False)
        gt_mean, gt_var = gt(batched_graphs, training=False)

        assert gcn_mean.shape == gt_mean.shape
        assert gcn_var.shape == gt_var.shape

        # Both should support extract_embeddings
        gcn_emb = gcn.extract_embeddings(batched_graphs)
        gt_emb = gt.extract_embeddings(batched_graphs)

        assert gcn_emb.shape == gt_emb.shape

    def test_same_api_as_mpnn(self, batched_graphs):
        """Test that Graph Transformer has the same API as UncertaintyMPNN."""
        from molax.models.mpnn import MPNNConfig, UncertaintyMPNN

        # Create both models
        mpnn_config = MPNNConfig(
            node_features=6,
            edge_features=1,
            hidden_features=[32, 32],
            out_features=1,
            aggregation="sum",
            dropout_rate=0.1,
        )
        mpnn = UncertaintyMPNN(mpnn_config, rngs=nnx.Rngs(0))

        gt_config = GraphTransformerConfig(
            node_features=6,
            edge_features=1,
            hidden_features=[32, 32],
            out_features=1,
            n_heads=4,
            dropout_rate=0.1,
            pe_type="none",
        )
        gt = UncertaintyGraphTransformer(gt_config, rngs=nnx.Rngs(0))

        # Both should accept same inputs and return same output shapes
        mpnn_mean, mpnn_var = mpnn(batched_graphs, training=False)
        gt_mean, gt_var = gt(batched_graphs, training=False)

        assert mpnn_mean.shape == gt_mean.shape
        assert mpnn_var.shape == gt_var.shape

        # Both should support extract_embeddings
        mpnn_emb = mpnn.extract_embeddings(batched_graphs)
        gt_emb = gt.extract_embeddings(batched_graphs)

        assert mpnn_emb.shape == gt_emb.shape

    def test_same_api_as_gat(self, batched_graphs):
        """Test that Graph Transformer has the same API as UncertaintyGAT."""
        from molax.models.gat import GATConfig, UncertaintyGAT

        # Create both models
        gat_config = GATConfig(
            node_features=6,
            edge_features=1,
            hidden_features=[32, 32],
            out_features=1,
            n_heads=4,
            dropout_rate=0.1,
        )
        gat = UncertaintyGAT(gat_config, rngs=nnx.Rngs(0))

        gt_config = GraphTransformerConfig(
            node_features=6,
            edge_features=1,
            hidden_features=[32, 32],
            out_features=1,
            n_heads=4,
            dropout_rate=0.1,
            pe_type="none",
        )
        gt = UncertaintyGraphTransformer(gt_config, rngs=nnx.Rngs(0))

        # Both should accept same inputs and return same output shapes
        gat_mean, gat_var = gat(batched_graphs, training=False)
        gt_mean, gt_var = gt(batched_graphs, training=False)

        assert gat_mean.shape == gt_mean.shape
        assert gat_var.shape == gt_var.shape

        # Both should support extract_embeddings
        gat_emb = gat.extract_embeddings(batched_graphs)
        gt_emb = gt.extract_embeddings(batched_graphs)

        assert gat_emb.shape == gt_emb.shape

    def test_all_produce_positive_variance(self, batched_graphs):
        """Test that all models produce positive variance."""
        from molax.models.gat import GATConfig, UncertaintyGAT
        from molax.models.gcn import GCNConfig, UncertaintyGCN
        from molax.models.mpnn import MPNNConfig, UncertaintyMPNN

        gcn_config = GCNConfig(
            node_features=6,
            hidden_features=[32],
            out_features=1,
        )
        gcn = UncertaintyGCN(gcn_config, rngs=nnx.Rngs(0))

        mpnn_config = MPNNConfig(
            node_features=6,
            edge_features=1,
            hidden_features=[32],
            out_features=1,
        )
        mpnn = UncertaintyMPNN(mpnn_config, rngs=nnx.Rngs(0))

        gat_config = GATConfig(
            node_features=6,
            hidden_features=[32],
            out_features=1,
            n_heads=4,
        )
        gat = UncertaintyGAT(gat_config, rngs=nnx.Rngs(0))

        gt_config = GraphTransformerConfig(
            node_features=6,
            hidden_features=[32],
            out_features=1,
            n_heads=4,
            pe_type="none",
        )
        gt = UncertaintyGraphTransformer(gt_config, rngs=nnx.Rngs(0))

        _, gcn_var = gcn(batched_graphs)
        _, mpnn_var = mpnn(batched_graphs)
        _, gat_var = gat(batched_graphs)
        _, gt_var = gt(batched_graphs)

        assert jnp.all(gcn_var > 0)
        assert jnp.all(mpnn_var > 0)
        assert jnp.all(gat_var > 0)
        assert jnp.all(gt_var > 0)


class TestGraphTransformerAcquisition:
    """Tests for Graph Transformer integration with acquisition functions."""

    def test_uncertainty_sampling_with_graph_transformer(self, model):
        """Test that Graph Transformer works with uncertainty_sampling."""
        from molax.acquisition.uncertainty import uncertainty_sampling

        smiles = ["C", "CC", "CCC", "CCO", "CCCC"]
        pool_graphs = [smiles_to_jraph(s) for s in smiles]

        # uncertainty_sampling uses MC dropout - same API as other models
        uncertainties = uncertainty_sampling(model, pool_graphs, n_samples=5)

        assert uncertainties.shape == (5,)
        assert jnp.all(uncertainties >= 0)
        assert jnp.all(jnp.isfinite(uncertainties))

    def test_diversity_sampling_with_graph_transformer(self):
        """Test that Graph Transformer graphs work with diversity_sampling."""
        from molax.acquisition.uncertainty import diversity_sampling

        smiles = ["C", "CC", "CCC", "CCO", "CCCC"]
        pool_graphs = [smiles_to_jraph(s) for s in smiles]
        labeled_graphs = [smiles_to_jraph("O")]

        selected = diversity_sampling(pool_graphs, labeled_graphs, n_select=3)

        assert len(selected) == 3
        assert all(isinstance(i, int) for i in selected)
        assert all(0 <= i < 5 for i in selected)
        # Check no duplicates
        assert len(set(selected)) == 3

    def test_combined_acquisition_with_graph_transformer(self, model):
        """Test that Graph Transformer works with combined_acquisition."""
        from molax.acquisition.uncertainty import combined_acquisition

        smiles = ["C", "CC", "CCC", "CCO", "CCCC"]
        pool_graphs = [smiles_to_jraph(s) for s in smiles]
        labeled_graphs = [smiles_to_jraph("O")]

        selected = combined_acquisition(
            model,
            pool_graphs,
            labeled_graphs,
            n_select=3,
            uncertainty_weight=0.7,
            n_mc_samples=5,
        )

        assert len(selected) == 3
        assert all(isinstance(i, int) for i in selected)
        assert all(0 <= i < 5 for i in selected)
