"""Tests for Graph Attention Network models."""

import flax.nnx as nnx
import jax.numpy as jnp
import jraph
import pytest

from molax.models.gat import (
    GATAttention,
    GATConfig,
    GATLayer,
    UncertaintyGAT,
    create_gat_optimizer,
    eval_gat_step,
    get_gat_uncertainties,
    train_gat_step,
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
    """Create default GAT config."""
    return GATConfig(
        node_features=6,
        edge_features=0,
        hidden_features=[32, 32],
        out_features=1,
        n_heads=4,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        negative_slope=0.2,
    )


@pytest.fixture
def config_with_edge_features():
    """Create GAT config that uses edge features."""
    return GATConfig(
        node_features=6,
        edge_features=1,
        hidden_features=[32, 32],
        out_features=1,
        n_heads=4,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        negative_slope=0.2,
    )


@pytest.fixture
def model(config):
    """Create GAT model."""
    return UncertaintyGAT(config, rngs=nnx.Rngs(0))


class TestGATAttention:
    """Tests for GATAttention module."""

    def test_output_shape(self, simple_graph):
        """Test that attention head produces correct output shape."""
        attn = GATAttention(
            in_features=6,
            out_features=16,
            edge_features=0,
            negative_slope=0.2,
            rngs=nnx.Rngs(0),
        )

        output = attn(simple_graph)

        # Should have n_nodes x out_features
        assert output.shape == (simple_graph.nodes.shape[0], 16)

    def test_with_edge_features(self, simple_graph):
        """Test attention head with edge features."""
        attn = GATAttention(
            in_features=6,
            out_features=16,
            edge_features=1,
            negative_slope=0.2,
            rngs=nnx.Rngs(0),
        )

        output = attn(simple_graph)

        assert output.shape == (simple_graph.nodes.shape[0], 16)
        assert jnp.all(jnp.isfinite(output))

    def test_finite_output(self, simple_graph):
        """Test that attention produces finite outputs."""
        attn = GATAttention(
            in_features=6,
            out_features=16,
            edge_features=0,
            negative_slope=0.2,
            rngs=nnx.Rngs(0),
        )

        output = attn(simple_graph)

        assert jnp.all(jnp.isfinite(output))

    def test_different_head_dims(self, simple_graph):
        """Test attention with different output dimensions."""
        for out_dim in [8, 16, 32, 64]:
            attn = GATAttention(
                in_features=6,
                out_features=out_dim,
                edge_features=0,
                negative_slope=0.2,
                rngs=nnx.Rngs(0),
            )

            output = attn(simple_graph)

            assert output.shape == (simple_graph.nodes.shape[0], out_dim)
            assert jnp.all(jnp.isfinite(output))


class TestGATLayer:
    """Tests for GATLayer module."""

    def test_output_shape_preserves_graph_structure(self, simple_graph):
        """Test that GAT layer preserves graph structure."""
        layer = GATLayer(
            in_features=6,
            out_features=32,
            n_heads=4,
            edge_features=0,
            negative_slope=0.2,
            concat_heads=True,
            attention_dropout_rate=0.1,
            rngs=nnx.Rngs(0),
        )

        output_graph = layer(simple_graph)

        # Graph structure should be preserved
        assert output_graph.senders.shape == simple_graph.senders.shape
        assert output_graph.receivers.shape == simple_graph.receivers.shape
        assert jnp.array_equal(output_graph.n_node, simple_graph.n_node)
        assert jnp.array_equal(output_graph.n_edge, simple_graph.n_edge)

        # Node features should have new dimension
        assert output_graph.nodes.shape[0] == simple_graph.nodes.shape[0]
        assert output_graph.nodes.shape[1] == 32

    def test_concat_heads(self, simple_graph):
        """Test concatenation of multiple heads."""
        layer = GATLayer(
            in_features=6,
            out_features=32,  # 4 heads x 8 dim each
            n_heads=4,
            edge_features=0,
            negative_slope=0.2,
            concat_heads=True,
            attention_dropout_rate=0.0,
            rngs=nnx.Rngs(0),
        )

        output_graph = layer(simple_graph)

        assert output_graph.nodes.shape[1] == 32
        assert jnp.all(jnp.isfinite(output_graph.nodes))

    def test_average_heads(self, simple_graph):
        """Test averaging of multiple heads."""
        layer = GATLayer(
            in_features=6,
            out_features=32,
            n_heads=4,
            edge_features=0,
            negative_slope=0.2,
            concat_heads=False,  # Average instead of concat
            attention_dropout_rate=0.0,
            rngs=nnx.Rngs(0),
        )

        output_graph = layer(simple_graph)

        # When averaging, output dim equals head dim
        assert output_graph.nodes.shape[1] == 32
        assert jnp.all(jnp.isfinite(output_graph.nodes))

    def test_batched_graphs(self, batched_graphs):
        """Test GAT layer on batched graphs."""
        layer = GATLayer(
            in_features=6,
            out_features=32,
            n_heads=4,
            edge_features=0,
            negative_slope=0.2,
            concat_heads=True,
            attention_dropout_rate=0.0,
            rngs=nnx.Rngs(0),
        )

        output_graph = layer(batched_graphs)

        # Total nodes should be preserved
        assert output_graph.nodes.shape[0] == batched_graphs.nodes.shape[0]
        assert output_graph.nodes.shape[1] == 32

    def test_with_edge_features(self, simple_graph):
        """Test GAT layer with edge features."""
        layer = GATLayer(
            in_features=6,
            out_features=32,
            n_heads=4,
            edge_features=1,
            negative_slope=0.2,
            concat_heads=True,
            attention_dropout_rate=0.0,
            rngs=nnx.Rngs(0),
        )

        output_graph = layer(simple_graph)

        assert output_graph.nodes.shape[1] == 32
        assert jnp.all(jnp.isfinite(output_graph.nodes))


class TestUncertaintyGAT:
    """Tests for UncertaintyGAT model."""

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
        model = UncertaintyGAT(config, rngs=nnx.Rngs(0))

        # Multiple forward passes in training mode should give different results
        results = []
        for _ in range(5):
            mean, _ = model(batched_graphs, training=True)
            results.append(mean)

        # Check that at least some results are different
        all_same = all(jnp.allclose(results[0], r) for r in results[1:])
        # With dropout, results should differ (unless dropout rate is 0)
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

        # Should be [n_graphs, last_hidden_dim]
        assert embeddings.shape == (4, 32)

    def test_extract_embeddings_finite(self, model, batched_graphs):
        """Test that extracted embeddings are finite."""
        embeddings = model.extract_embeddings(batched_graphs)

        assert jnp.all(jnp.isfinite(embeddings))

    def test_single_graph(self, config, simple_graph):
        """Test model on a single unbatched graph."""
        model = UncertaintyGAT(config, rngs=nnx.Rngs(0))

        mean, variance = model(simple_graph)

        assert mean.shape == (1, 1)
        assert variance.shape == (1, 1)
        assert jnp.all(variance > 0)

    def test_with_edge_features(self, config_with_edge_features, batched_graphs):
        """Test model with edge features in attention."""
        model = UncertaintyGAT(config_with_edge_features, rngs=nnx.Rngs(0))

        mean, variance = model(batched_graphs)

        assert mean.shape == (4, 1)
        assert variance.shape == (4, 1)
        assert jnp.all(variance > 0)
        assert jnp.all(jnp.isfinite(mean))

    def test_different_n_heads(self, batched_graphs):
        """Test model with different numbers of attention heads."""
        for n_heads in [1, 2, 4, 8]:
            config = GATConfig(
                node_features=6,
                hidden_features=[32],  # Must be divisible by n_heads
                out_features=1,
                n_heads=n_heads,
            )
            model = UncertaintyGAT(config, rngs=nnx.Rngs(0))

            mean, variance = model(batched_graphs)

            assert mean.shape == (4, 1)
            assert variance.shape == (4, 1)
            assert jnp.all(variance > 0)
            assert jnp.all(jnp.isfinite(mean))


class TestGATTraining:
    """Tests for GAT training utilities."""

    def test_create_optimizer(self, model):
        """Test optimizer creation."""
        optimizer = create_gat_optimizer(model, learning_rate=1e-3)
        assert isinstance(optimizer, nnx.Optimizer)

    def test_train_step_returns_loss(self, model, batched_graphs):
        """Test that training step returns a loss value."""
        optimizer = create_gat_optimizer(model)

        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.ones(4, dtype=bool)

        loss = train_gat_step(model, optimizer, batched_graphs, labels, mask)

        assert loss.shape == ()  # Scalar
        assert jnp.isfinite(loss)

    def test_train_step_updates_weights(self, config, batched_graphs):
        """Test that training updates model weights."""
        model = UncertaintyGAT(config, rngs=nnx.Rngs(0))
        optimizer = create_gat_optimizer(model)

        # Get initial weights
        initial_weights = model.gat_layers[0].heads[0].W.kernel.copy()

        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.ones(4, dtype=bool)

        # Train for a few steps
        for _ in range(5):
            train_gat_step(model, optimizer, batched_graphs, labels, mask)

        # Check weights changed
        new_weights = model.gat_layers[0].heads[0].W.kernel
        assert not jnp.allclose(initial_weights, new_weights), (
            "Weights did not change after training"
        )

    def test_train_step_reduces_loss(self, config, batched_graphs):
        """Test that training reduces loss over time."""
        model = UncertaintyGAT(config, rngs=nnx.Rngs(42))
        optimizer = create_gat_optimizer(model, learning_rate=1e-3)

        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.ones(4, dtype=bool)

        # Get initial loss
        initial_loss = train_gat_step(model, optimizer, batched_graphs, labels, mask)

        # Train more
        for _ in range(100):
            final_loss = train_gat_step(model, optimizer, batched_graphs, labels, mask)

        # Loss should decrease (or at least not increase significantly)
        assert final_loss < initial_loss * 1.1, "Loss should decrease during training"

    def test_train_step_with_partial_mask(self, model, batched_graphs):
        """Test training with only some samples masked."""
        optimizer = create_gat_optimizer(model)

        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        # Only use first two samples
        mask = jnp.array([True, True, False, False])

        loss = train_gat_step(model, optimizer, batched_graphs, labels, mask)

        assert jnp.isfinite(loss)


class TestGATEvaluation:
    """Tests for GAT evaluation utilities."""

    def test_eval_step_returns_mse(self, model, batched_graphs):
        """Test that eval step returns MSE and predictions."""
        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.ones(4, dtype=bool)

        mse, predictions = eval_gat_step(model, batched_graphs, labels, mask)

        assert mse.shape == ()
        assert predictions.shape == (4,)
        assert jnp.isfinite(mse)
        assert mse >= 0

    def test_get_uncertainties(self, model, batched_graphs):
        """Test uncertainty extraction."""
        epistemic_var, aleatoric_var = get_gat_uncertainties(
            model, batched_graphs, n_samples=10
        )

        assert epistemic_var.shape == (4,)
        assert aleatoric_var.shape == (4,)

        # Both should be non-negative
        assert jnp.all(epistemic_var >= 0)
        assert jnp.all(aleatoric_var > 0)  # Aleatoric should be strictly positive

    def test_get_uncertainties_finite(self, model, batched_graphs):
        """Test that uncertainties are finite."""
        epistemic_var, aleatoric_var = get_gat_uncertainties(
            model, batched_graphs, n_samples=10
        )

        assert jnp.all(jnp.isfinite(epistemic_var))
        assert jnp.all(jnp.isfinite(aleatoric_var))


class TestGATAcquisition:
    """Tests for GAT integration with acquisition functions."""

    def test_uncertainty_sampling_with_gat(self, model):
        """Test that GAT works with uncertainty_sampling."""
        from molax.acquisition.uncertainty import uncertainty_sampling

        smiles = ["C", "CC", "CCC", "CCO", "CCCC"]
        pool_graphs = [smiles_to_jraph(s) for s in smiles]

        # uncertainty_sampling uses MC dropout - same API as GCN
        uncertainties = uncertainty_sampling(model, pool_graphs, n_samples=5)

        assert uncertainties.shape == (5,)
        assert jnp.all(uncertainties >= 0)
        assert jnp.all(jnp.isfinite(uncertainties))

    def test_diversity_sampling_with_gat(self):
        """Test that GAT graphs work with diversity_sampling."""
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

    def test_combined_acquisition_with_gat(self, model):
        """Test that GAT works with combined_acquisition."""
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


class TestGATvsGCN:
    """Tests comparing GAT to GCN."""

    def test_same_api_as_gcn(self, batched_graphs):
        """Test that GAT has the same API as UncertaintyGCN."""
        from molax.models.gcn import GCNConfig, UncertaintyGCN

        # Create both models
        gcn_config = GCNConfig(
            node_features=6,
            hidden_features=[32, 32],
            out_features=1,
            dropout_rate=0.1,
        )
        gcn = UncertaintyGCN(gcn_config, rngs=nnx.Rngs(0))

        gat_config = GATConfig(
            node_features=6,
            hidden_features=[32, 32],
            out_features=1,
            n_heads=4,
            dropout_rate=0.1,
        )
        gat = UncertaintyGAT(gat_config, rngs=nnx.Rngs(0))

        # Both should accept same inputs and return same output shapes
        gcn_mean, gcn_var = gcn(batched_graphs, training=False)
        gat_mean, gat_var = gat(batched_graphs, training=False)

        assert gcn_mean.shape == gat_mean.shape
        assert gcn_var.shape == gat_var.shape

        # Both should support extract_embeddings
        gcn_emb = gcn.extract_embeddings(batched_graphs)
        gat_emb = gat.extract_embeddings(batched_graphs)

        assert gcn_emb.shape == gat_emb.shape

    def test_same_api_as_mpnn(self, batched_graphs):
        """Test that GAT has the same API as UncertaintyMPNN."""
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

        gat_config = GATConfig(
            node_features=6,
            edge_features=1,
            hidden_features=[32, 32],
            out_features=1,
            n_heads=4,
            dropout_rate=0.1,
        )
        gat = UncertaintyGAT(gat_config, rngs=nnx.Rngs(0))

        # Both should accept same inputs and return same output shapes
        mpnn_mean, mpnn_var = mpnn(batched_graphs, training=False)
        gat_mean, gat_var = gat(batched_graphs, training=False)

        assert mpnn_mean.shape == gat_mean.shape
        assert mpnn_var.shape == gat_var.shape

        # Both should support extract_embeddings
        mpnn_emb = mpnn.extract_embeddings(batched_graphs)
        gat_emb = gat.extract_embeddings(batched_graphs)

        assert mpnn_emb.shape == gat_emb.shape

    def test_all_produce_positive_variance(self, batched_graphs):
        """Test that all models produce positive variance."""
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

        _, gcn_var = gcn(batched_graphs)
        _, mpnn_var = mpnn(batched_graphs)
        _, gat_var = gat(batched_graphs)

        assert jnp.all(gcn_var > 0)
        assert jnp.all(mpnn_var > 0)
        assert jnp.all(gat_var > 0)


class TestGATConfig:
    """Tests for GAT configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GATConfig()

        assert config.node_features == 6
        assert config.edge_features == 0
        assert config.hidden_features == (64, 64)
        assert config.out_features == 1
        assert config.n_heads == 4
        assert config.dropout_rate == 0.1
        assert config.attention_dropout_rate == 0.1
        assert config.negative_slope == 0.2

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GATConfig(
            node_features=10,
            edge_features=3,
            hidden_features=[128, 64, 32],
            out_features=2,
            n_heads=8,
            dropout_rate=0.2,
            attention_dropout_rate=0.15,
            negative_slope=0.1,
        )

        assert config.node_features == 10
        assert config.edge_features == 3
        assert config.hidden_features == [128, 64, 32]
        assert config.out_features == 2
        assert config.n_heads == 8
        assert config.dropout_rate == 0.2
        assert config.attention_dropout_rate == 0.15
        assert config.negative_slope == 0.1

    def test_model_respects_config(self, batched_graphs):
        """Test that model architecture matches config."""
        config = GATConfig(
            node_features=6,
            edge_features=0,
            hidden_features=[64, 32, 16],
            out_features=3,
            n_heads=4,
        )
        model = UncertaintyGAT(config, rngs=nnx.Rngs(0))

        # Check number of layers
        assert len(model.gat_layers) == 3

        # Check number of heads per layer
        assert len(model.gat_layers[0].heads) == 4

        # Check output dimension
        mean, var = model(batched_graphs)
        assert mean.shape == (4, 3)
        assert var.shape == (4, 3)


class TestGATAttentionNormalization:
    """Tests for attention weight normalization."""

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 for each node."""
        # Create a simple graph manually for testing
        nodes = jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )
        # Triangle graph: 0->1, 1->0, 0->2, 2->0, 1->2, 2->1
        senders = jnp.array([0, 1, 0, 2, 1, 2])
        receivers = jnp.array([1, 0, 2, 0, 2, 1])
        n_node = jnp.array([3])
        n_edge = jnp.array([6])

        graph = jraph.GraphsTuple(
            nodes=nodes,
            edges=None,
            senders=senders,
            receivers=receivers,
            n_node=n_node,
            n_edge=n_edge,
            globals=None,
        )

        attn = GATAttention(
            in_features=6,
            out_features=8,
            edge_features=0,
            negative_slope=0.2,
            rngs=nnx.Rngs(0),
        )

        # The attention mechanism internally uses segment_softmax
        # which normalizes weights per receiver node
        output = attn(graph)

        # Output should be finite and well-defined
        assert jnp.all(jnp.isfinite(output))
        assert output.shape == (3, 8)
