"""Tests for GCN models using jraph."""

import flax.nnx as nnx
import jax.numpy as jnp
import jraph
import pytest

from molax.models.gcn import GCNConfig, GraphConvolution, MolecularGCN, UncertaintyGCN
from molax.utils.data import smiles_to_jraph


@pytest.fixture
def simple_graph():
    """Create a simple test graph."""
    return smiles_to_jraph("CCO")  # ethanol


@pytest.fixture
def batched_graphs():
    """Create a batch of test graphs."""
    smiles = ["C", "CC", "CCC", "CCO"]
    graphs = [smiles_to_jraph(s) for s in smiles]
    return jraph.batch(graphs)


class TestGraphConvolution:
    """Tests for GraphConvolution layer."""

    def test_output_shape(self, simple_graph):
        """Test that output has correct shape."""
        layer = GraphConvolution(6, 32, rngs=nnx.Rngs(0))
        output = layer(simple_graph)
        assert output.nodes.shape == (simple_graph.nodes.shape[0], 32)

    def test_preserves_graph_structure(self, simple_graph):
        """Test that graph structure is preserved."""
        layer = GraphConvolution(6, 32, rngs=nnx.Rngs(0))
        output = layer(simple_graph)
        assert jnp.array_equal(output.senders, simple_graph.senders)
        assert jnp.array_equal(output.receivers, simple_graph.receivers)


class TestMolecularGCN:
    """Tests for MolecularGCN model."""

    def test_single_graph(self, simple_graph):
        """Test forward pass on single graph."""
        config = GCNConfig(node_features=6, hidden_features=[32], out_features=1)
        model = MolecularGCN(config, rngs=nnx.Rngs(0))

        # Batch single graph for model
        batched = jraph.batch([simple_graph])
        output = model(batched)
        assert output.shape == (1, 1)

    def test_batched_graphs(self, batched_graphs):
        """Test forward pass on batched graphs."""
        config = GCNConfig(node_features=6, hidden_features=[32, 32], out_features=1)
        model = MolecularGCN(config, rngs=nnx.Rngs(0))

        output = model(batched_graphs)
        assert output.shape == (4, 1)  # 4 graphs in batch

    def test_training_mode(self, batched_graphs):
        """Test that training mode enables dropout."""
        config = GCNConfig(
            node_features=6, hidden_features=[32], out_features=1, dropout_rate=0.5
        )
        model = MolecularGCN(config, rngs=nnx.Rngs(0))

        out1 = model(batched_graphs, training=True)
        out2 = model(batched_graphs, training=True)
        assert out1.shape == out2.shape


class TestUncertaintyGCN:
    """Tests for UncertaintyGCN model."""

    def test_returns_mean_and_variance(self, batched_graphs):
        """Test that model returns both mean and variance."""
        config = GCNConfig(node_features=6, hidden_features=[32], out_features=1)
        model = UncertaintyGCN(config, rngs=nnx.Rngs(0))

        mean, var = model(batched_graphs)
        assert mean.shape == (4, 1)
        assert var.shape == (4, 1)

    def test_variance_positive(self, batched_graphs):
        """Test that variance is always positive."""
        config = GCNConfig(node_features=6, hidden_features=[32], out_features=1)
        model = UncertaintyGCN(config, rngs=nnx.Rngs(0))

        _, var = model(batched_graphs)
        assert jnp.all(var > 0)

    def test_mc_dropout_variance(self, batched_graphs):
        """Test MC dropout produces variance in predictions."""
        config = GCNConfig(
            node_features=6, hidden_features=[32], out_features=1, dropout_rate=0.3
        )
        model = UncertaintyGCN(config, rngs=nnx.Rngs(0))

        predictions = []
        for _ in range(10):
            mean, _ = model(batched_graphs, training=True)
            predictions.append(mean)

        predictions = jnp.stack(predictions)
        mc_variance = jnp.var(predictions, axis=0)
        assert mc_variance.shape == (4, 1)
