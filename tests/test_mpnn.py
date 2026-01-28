"""Tests for Message Passing Neural Network models."""

import flax.nnx as nnx
import jax.numpy as jnp
import jraph
import pytest

from molax.models.mpnn import (
    MessageFunction,
    MessagePassingLayer,
    MPNNConfig,
    UncertaintyMPNN,
    create_mpnn_optimizer,
    eval_mpnn_step,
    get_mpnn_uncertainties,
    train_mpnn_step,
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
    """Create default MPNN config."""
    return MPNNConfig(
        node_features=6,
        edge_features=1,
        hidden_features=[32, 32],
        out_features=1,
        aggregation="sum",
        dropout_rate=0.1,
    )


@pytest.fixture
def model(config):
    """Create MPNN model."""
    return UncertaintyMPNN(config, rngs=nnx.Rngs(0))


class TestMessageFunction:
    """Tests for MessageFunction module."""

    def test_output_shape(self):
        """Test that message function produces correct output shape."""
        msg_fn = MessageFunction(
            node_features=6,
            edge_features=1,
            out_features=32,
            rngs=nnx.Rngs(0),
        )

        # Simulate 10 edges
        sender_features = jnp.ones((10, 6))
        receiver_features = jnp.ones((10, 6))
        edge_features = jnp.ones((10, 1))

        messages = msg_fn(sender_features, receiver_features, edge_features)

        assert messages.shape == (10, 32)

    def test_different_edge_dims(self):
        """Test message function with different edge feature dimensions."""
        for edge_dim in [1, 3, 6]:
            msg_fn = MessageFunction(
                node_features=6,
                edge_features=edge_dim,
                out_features=32,
                rngs=nnx.Rngs(0),
            )

            sender_features = jnp.ones((5, 6))
            receiver_features = jnp.ones((5, 6))
            edge_features = jnp.ones((5, edge_dim))

            messages = msg_fn(sender_features, receiver_features, edge_features)

            assert messages.shape == (5, 32)

    def test_finite_output(self):
        """Test that message function produces finite outputs."""
        msg_fn = MessageFunction(
            node_features=6,
            edge_features=1,
            out_features=32,
            rngs=nnx.Rngs(0),
        )

        sender_features = jnp.ones((10, 6)) * 0.5
        receiver_features = jnp.ones((10, 6)) * 0.5
        edge_features = jnp.ones((10, 1)) * 0.5

        messages = msg_fn(sender_features, receiver_features, edge_features)

        assert jnp.all(jnp.isfinite(messages))


class TestMessagePassingLayer:
    """Tests for MessagePassingLayer module."""

    def test_output_shape_preserves_graph_structure(self, simple_graph):
        """Test that MP layer preserves graph structure."""
        mp_layer = MessagePassingLayer(
            node_features=6,
            edge_features=1,
            out_features=32,
            aggregation="sum",
            rngs=nnx.Rngs(0),
        )

        output_graph = mp_layer(simple_graph)

        # Graph structure should be preserved
        assert output_graph.senders.shape == simple_graph.senders.shape
        assert output_graph.receivers.shape == simple_graph.receivers.shape
        assert output_graph.edges.shape == simple_graph.edges.shape
        assert jnp.array_equal(output_graph.n_node, simple_graph.n_node)
        assert jnp.array_equal(output_graph.n_edge, simple_graph.n_edge)

        # Node features should have new dimension
        assert output_graph.nodes.shape[0] == simple_graph.nodes.shape[0]
        assert output_graph.nodes.shape[1] == 32

    def test_sum_aggregation(self, simple_graph):
        """Test sum aggregation method."""
        mp_layer = MessagePassingLayer(
            node_features=6,
            edge_features=1,
            out_features=32,
            aggregation="sum",
            rngs=nnx.Rngs(0),
        )

        output_graph = mp_layer(simple_graph)
        assert output_graph.nodes.shape[1] == 32
        assert jnp.all(jnp.isfinite(output_graph.nodes))

    def test_mean_aggregation(self, simple_graph):
        """Test mean aggregation method."""
        mp_layer = MessagePassingLayer(
            node_features=6,
            edge_features=1,
            out_features=32,
            aggregation="mean",
            rngs=nnx.Rngs(0),
        )

        output_graph = mp_layer(simple_graph)
        assert output_graph.nodes.shape[1] == 32
        assert jnp.all(jnp.isfinite(output_graph.nodes))

    def test_max_aggregation(self, simple_graph):
        """Test max aggregation method."""
        mp_layer = MessagePassingLayer(
            node_features=6,
            edge_features=1,
            out_features=32,
            aggregation="max",
            rngs=nnx.Rngs(0),
        )

        output_graph = mp_layer(simple_graph)
        assert output_graph.nodes.shape[1] == 32
        assert jnp.all(jnp.isfinite(output_graph.nodes))

    def test_batched_graphs(self, batched_graphs):
        """Test MP layer on batched graphs."""
        mp_layer = MessagePassingLayer(
            node_features=6,
            edge_features=1,
            out_features=32,
            aggregation="sum",
            rngs=nnx.Rngs(0),
        )

        output_graph = mp_layer(batched_graphs)

        # Total nodes should be preserved
        assert output_graph.nodes.shape[0] == batched_graphs.nodes.shape[0]
        assert output_graph.nodes.shape[1] == 32


class TestUncertaintyMPNN:
    """Tests for UncertaintyMPNN model."""

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
        model = UncertaintyMPNN(config, rngs=nnx.Rngs(0))

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

    def test_different_aggregations(self, batched_graphs):
        """Test model with different aggregation methods."""
        for agg in ["sum", "mean", "max"]:
            config = MPNNConfig(
                node_features=6,
                edge_features=1,
                hidden_features=[32],
                out_features=1,
                aggregation=agg,
            )
            model = UncertaintyMPNN(config, rngs=nnx.Rngs(0))

            mean, variance = model(batched_graphs)

            assert mean.shape == (4, 1)
            assert variance.shape == (4, 1)
            assert jnp.all(variance > 0)
            assert jnp.all(jnp.isfinite(mean))

    def test_single_graph(self, config, simple_graph):
        """Test model on a single unbatched graph."""
        model = UncertaintyMPNN(config, rngs=nnx.Rngs(0))

        mean, variance = model(simple_graph)

        assert mean.shape == (1, 1)
        assert variance.shape == (1, 1)
        assert jnp.all(variance > 0)


class TestMPNNTraining:
    """Tests for MPNN training utilities."""

    def test_create_optimizer(self, model):
        """Test optimizer creation."""
        optimizer = create_mpnn_optimizer(model, learning_rate=1e-3)
        assert isinstance(optimizer, nnx.Optimizer)

    def test_train_step_returns_loss(self, model, batched_graphs):
        """Test that training step returns a loss value."""
        optimizer = create_mpnn_optimizer(model)

        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.ones(4, dtype=bool)

        loss = train_mpnn_step(model, optimizer, batched_graphs, labels, mask)

        assert loss.shape == ()  # Scalar
        assert jnp.isfinite(loss)

    def test_train_step_updates_weights(self, config, batched_graphs):
        """Test that training updates model weights."""
        model = UncertaintyMPNN(config, rngs=nnx.Rngs(0))
        optimizer = create_mpnn_optimizer(model)

        # Get initial weights
        initial_weights = model.mp_layers[0].message_fn.linear1.kernel.copy()

        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.ones(4, dtype=bool)

        # Train for a few steps
        for _ in range(5):
            train_mpnn_step(model, optimizer, batched_graphs, labels, mask)

        # Check weights changed
        new_weights = model.mp_layers[0].message_fn.linear1.kernel
        assert not jnp.allclose(initial_weights, new_weights), (
            "Weights did not change after training"
        )

    def test_train_step_reduces_loss(self, config, batched_graphs):
        """Test that training reduces loss over time."""
        model = UncertaintyMPNN(config, rngs=nnx.Rngs(42))
        # Use lower learning rate to avoid gradient explosion
        optimizer = create_mpnn_optimizer(model, learning_rate=1e-3)

        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.ones(4, dtype=bool)

        # Get initial loss
        initial_loss = train_mpnn_step(model, optimizer, batched_graphs, labels, mask)

        # Train more
        for _ in range(100):
            final_loss = train_mpnn_step(model, optimizer, batched_graphs, labels, mask)

        # Loss should decrease (or at least not increase significantly)
        assert final_loss < initial_loss * 1.1, "Loss should decrease during training"

    def test_train_step_with_partial_mask(self, model, batched_graphs):
        """Test training with only some samples masked."""
        optimizer = create_mpnn_optimizer(model)

        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        # Only use first two samples
        mask = jnp.array([True, True, False, False])

        loss = train_mpnn_step(model, optimizer, batched_graphs, labels, mask)

        assert jnp.isfinite(loss)


class TestMPNNEvaluation:
    """Tests for MPNN evaluation utilities."""

    def test_eval_step_returns_mse(self, model, batched_graphs):
        """Test that eval step returns MSE and predictions."""
        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.ones(4, dtype=bool)

        mse, predictions = eval_mpnn_step(model, batched_graphs, labels, mask)

        assert mse.shape == ()
        assert predictions.shape == (4,)
        assert jnp.isfinite(mse)
        assert mse >= 0

    def test_get_uncertainties(self, model, batched_graphs):
        """Test uncertainty extraction."""
        epistemic_var, aleatoric_var = get_mpnn_uncertainties(
            model, batched_graphs, n_samples=10
        )

        assert epistemic_var.shape == (4,)
        assert aleatoric_var.shape == (4,)

        # Both should be non-negative
        assert jnp.all(epistemic_var >= 0)
        assert jnp.all(aleatoric_var > 0)  # Aleatoric should be strictly positive

    def test_get_uncertainties_finite(self, model, batched_graphs):
        """Test that uncertainties are finite."""
        epistemic_var, aleatoric_var = get_mpnn_uncertainties(
            model, batched_graphs, n_samples=10
        )

        assert jnp.all(jnp.isfinite(epistemic_var))
        assert jnp.all(jnp.isfinite(aleatoric_var))


class TestMPNNAcquisition:
    """Tests for MPNN integration with acquisition functions."""

    def test_uncertainty_sampling_with_mpnn(self, model):
        """Test that MPNN works with uncertainty_sampling."""
        from molax.acquisition.uncertainty import uncertainty_sampling

        smiles = ["C", "CC", "CCC", "CCO", "CCCC"]
        pool_graphs = [smiles_to_jraph(s) for s in smiles]

        # uncertainty_sampling uses MC dropout - same API as GCN
        uncertainties = uncertainty_sampling(model, pool_graphs, n_samples=5)

        assert uncertainties.shape == (5,)
        assert jnp.all(uncertainties >= 0)
        assert jnp.all(jnp.isfinite(uncertainties))

    def test_diversity_sampling_with_mpnn(self):
        """Test that MPNN graphs work with diversity_sampling."""
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

    def test_combined_acquisition_with_mpnn(self, model):
        """Test that MPNN works with combined_acquisition."""
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


class TestMPNNvsGCN:
    """Tests comparing MPNN to GCN."""

    def test_same_api_as_gcn(self, batched_graphs):
        """Test that MPNN has the same API as UncertaintyGCN."""
        from molax.models.gcn import GCNConfig, UncertaintyGCN

        # Create both models
        gcn_config = GCNConfig(
            node_features=6,
            hidden_features=[32, 32],
            out_features=1,
            dropout_rate=0.1,
        )
        gcn = UncertaintyGCN(gcn_config, rngs=nnx.Rngs(0))

        mpnn_config = MPNNConfig(
            node_features=6,
            edge_features=1,
            hidden_features=[32, 32],
            out_features=1,
            aggregation="sum",
            dropout_rate=0.1,
        )
        mpnn = UncertaintyMPNN(mpnn_config, rngs=nnx.Rngs(0))

        # Both should accept same inputs and return same output shapes
        gcn_mean, gcn_var = gcn(batched_graphs, training=False)
        mpnn_mean, mpnn_var = mpnn(batched_graphs, training=False)

        assert gcn_mean.shape == mpnn_mean.shape
        assert gcn_var.shape == mpnn_var.shape

        # Both should support extract_embeddings
        gcn_emb = gcn.extract_embeddings(batched_graphs)
        mpnn_emb = mpnn.extract_embeddings(batched_graphs)

        assert gcn_emb.shape == mpnn_emb.shape

    def test_both_produce_positive_variance(self, batched_graphs):
        """Test that both models produce positive variance."""
        from molax.models.gcn import GCNConfig, UncertaintyGCN

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

        _, gcn_var = gcn(batched_graphs)
        _, mpnn_var = mpnn(batched_graphs)

        assert jnp.all(gcn_var > 0)
        assert jnp.all(mpnn_var > 0)


class TestMPNNConfig:
    """Tests for MPNN configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MPNNConfig()

        assert config.node_features == 6
        assert config.edge_features == 1
        assert config.hidden_features == (64, 64)
        assert config.out_features == 1
        assert config.aggregation == "sum"
        assert config.dropout_rate == 0.1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MPNNConfig(
            node_features=10,
            edge_features=3,
            hidden_features=[128, 64, 32],
            out_features=2,
            aggregation="mean",
            dropout_rate=0.2,
        )

        assert config.node_features == 10
        assert config.edge_features == 3
        assert config.hidden_features == [128, 64, 32]
        assert config.out_features == 2
        assert config.aggregation == "mean"
        assert config.dropout_rate == 0.2

    def test_model_respects_config(self, batched_graphs):
        """Test that model architecture matches config."""
        config = MPNNConfig(
            node_features=6,
            edge_features=1,
            hidden_features=[64, 32, 16],
            out_features=3,
            aggregation="mean",
        )
        model = UncertaintyMPNN(config, rngs=nnx.Rngs(0))

        # Check number of layers
        assert len(model.mp_layers) == 3

        # Check output dimension
        mean, var = model(batched_graphs)
        assert mean.shape == (4, 3)
        assert var.shape == (4, 3)
