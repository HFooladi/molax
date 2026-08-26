"""Tests for Core-Set acquisition functions."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from molax.acquisition.coreset import (
    coreset_from_embeddings,
    coreset_sampling,
    coreset_sampling_with_scores,
)
from molax.models.ensemble import DeepEnsemble, EnsembleConfig
from molax.models.evidential import EvidentialConfig, EvidentialGCN
from molax.models.gcn import GCNConfig, UncertaintyGCN
from molax.utils.data import batch_graphs, smiles_to_jraph


@pytest.fixture
def gcn_model():
    """Create a test UncertaintyGCN model."""
    config = GCNConfig(
        node_features=6,
        hidden_features=[16, 16],
        out_features=1,
        dropout_rate=0.1,
    )
    return UncertaintyGCN(config, rngs=nnx.Rngs(0))


@pytest.fixture
def ensemble_model():
    """Create a test DeepEnsemble model."""
    base_config = GCNConfig(
        node_features=6,
        hidden_features=[16, 16],
        out_features=1,
        dropout_rate=0.1,
    )
    config = EnsembleConfig(base_config=base_config, n_members=3)
    return DeepEnsemble(config, rngs=nnx.Rngs(0))


@pytest.fixture
def evidential_model():
    """Create a test EvidentialGCN model."""
    base_config = GCNConfig(
        node_features=6,
        hidden_features=[16, 16],
        out_features=1,
        dropout_rate=0.1,
    )
    config = EvidentialConfig(base_config=base_config, lambda_reg=0.1)
    return EvidentialGCN(config, rngs=nnx.Rngs(0))


@pytest.fixture
def sample_graphs():
    """Create sample molecular graphs for testing."""
    smiles_list = [
        "C",  # methane
        "CC",  # ethane
        "CCC",  # propane
        "CCCC",  # butane
        "C=C",  # ethene
        "CCO",  # ethanol
        "CC=O",  # acetaldehyde
        "c1ccccc1",  # benzene
    ]
    graphs = []
    for smiles in smiles_list:
        try:
            graphs.append(smiles_to_jraph(smiles))
        except Exception:
            pass
    return graphs


class TestCoreSetSampling:
    """Tests for coreset_sampling function."""

    def test_returns_correct_count(self, gcn_model, sample_graphs):
        """Test that Core-Set returns the requested number of samples."""
        n_select = 3
        labeled = sample_graphs[:2]
        pool = sample_graphs[2:]

        selected = coreset_sampling(gcn_model, pool, labeled, n_select)
        assert len(selected) == n_select

    def test_no_duplicates(self, gcn_model, sample_graphs):
        """Test that selected indices are unique."""
        n_select = 4
        labeled = sample_graphs[:2]
        pool = sample_graphs[2:]

        selected = coreset_sampling(gcn_model, pool, labeled, n_select)
        assert len(selected) == len(set(selected))

    def test_indices_within_bounds(self, gcn_model, sample_graphs):
        """Test that selected indices are valid pool indices."""
        n_select = 3
        labeled = sample_graphs[:2]
        pool = sample_graphs[2:]

        selected = coreset_sampling(gcn_model, pool, labeled, n_select)
        for idx in selected:
            assert 0 <= idx < len(pool)

    def test_empty_pool(self, gcn_model, sample_graphs):
        """Test with empty pool."""
        labeled = sample_graphs[:2]
        selected = coreset_sampling(gcn_model, [], labeled, 3)
        assert selected == []

    def test_empty_labeled_set(self, gcn_model, sample_graphs):
        """Test with empty labeled set."""
        n_select = 3
        pool = sample_graphs

        selected = coreset_sampling(gcn_model, pool, [], n_select)
        assert len(selected) == n_select
        assert len(selected) == len(set(selected))

    def test_n_select_larger_than_pool(self, gcn_model, sample_graphs):
        """Test when n_select > pool size."""
        labeled = sample_graphs[:2]
        pool = sample_graphs[2:5]  # 3 samples

        selected = coreset_sampling(gcn_model, pool, labeled, n_select=10)
        assert len(selected) == len(pool)

    def test_works_with_ensemble(self, ensemble_model, sample_graphs):
        """Test Core-Set with DeepEnsemble."""
        n_select = 3
        labeled = sample_graphs[:2]
        pool = sample_graphs[2:]

        selected = coreset_sampling(ensemble_model, pool, labeled, n_select)
        assert len(selected) == n_select

    def test_works_with_evidential(self, evidential_model, sample_graphs):
        """Test Core-Set with EvidentialGCN."""
        n_select = 3
        labeled = sample_graphs[:2]
        pool = sample_graphs[2:]

        selected = coreset_sampling(evidential_model, pool, labeled, n_select)
        assert len(selected) == n_select


class TestCoreSetSamplingWithScores:
    """Tests for coreset_sampling_with_scores function."""

    def test_returns_correct_shape(self, gcn_model, sample_graphs):
        """Test that scores have correct shape."""
        labeled = sample_graphs[:2]
        pool = sample_graphs[2:]

        scores = coreset_sampling_with_scores(gcn_model, pool, labeled)
        assert scores.shape == (len(pool),)

    def test_scores_are_non_negative(self, gcn_model, sample_graphs):
        """Test that scores (distances) are non-negative."""
        labeled = sample_graphs[:2]
        pool = sample_graphs[2:]

        scores = coreset_sampling_with_scores(gcn_model, pool, labeled)
        assert jnp.all(scores >= 0)

    def test_empty_pool(self, gcn_model, sample_graphs):
        """Test with empty pool."""
        labeled = sample_graphs[:2]
        scores = coreset_sampling_with_scores(gcn_model, [], labeled)
        assert scores.shape == (0,)

    def test_empty_labeled_returns_inf(self, gcn_model, sample_graphs):
        """Test with empty labeled set returns infinity."""
        pool = sample_graphs

        scores = coreset_sampling_with_scores(gcn_model, pool, [])
        assert jnp.all(jnp.isinf(scores))


class TestCoresetFromEmbeddings:
    """Tests for the index-based k-center greedy helper."""

    def test_returns_indices_into_full_array(self):
        """Selected indices index the full array, not the pool subset."""
        embeddings = jnp.array(
            [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0], [0.1, 0.1]]
        )
        pool_mask = jnp.array([False, True, True, True, True])

        selected = coreset_from_embeddings(embeddings, pool_mask, n_select=2)

        assert len(selected) == 2
        assert all(0 <= i < embeddings.shape[0] for i in selected)
        assert 0 not in selected, "labeled point was selected"

    def test_never_selects_labeled_points(self):
        embeddings = jax.random.normal(jax.random.PRNGKey(0), (20, 8))
        pool_mask = jnp.arange(20) >= 5  # first 5 are labeled

        selected = coreset_from_embeddings(embeddings, pool_mask, n_select=10)

        assert len(selected) == 10
        assert len(set(selected)) == 10, "duplicate selections"
        assert all(i >= 5 for i in selected)

    def test_picks_far_points_first(self):
        """The point furthest from the labeled set must be selected first."""
        embeddings = jnp.array([[0.0, 0.0], [1.0, 0.0], [50.0, 0.0]])
        pool_mask = jnp.array([False, True, True])

        selected = coreset_from_embeddings(embeddings, pool_mask, n_select=1)

        assert selected == [2]

    def test_covers_distinct_clusters(self):
        """With no labeled data, greedy should spread across clusters."""
        embeddings = jnp.array(
            [
                [0.0, 0.0],
                [0.1, 0.1],
                [20.0, 0.0],
                [20.1, 0.1],
                [0.0, 20.0],
                [0.1, 20.1],
            ]
        )
        pool_mask = jnp.ones(6, dtype=bool)

        selected = coreset_from_embeddings(embeddings, pool_mask, n_select=3)

        clusters = {i // 2 for i in selected}
        assert clusters == {0, 1, 2}, f"did not cover all clusters: {selected}"

    def test_empty_pool_returns_empty(self):
        embeddings = jnp.zeros((4, 3))
        assert coreset_from_embeddings(embeddings, jnp.zeros(4, dtype=bool), 2) == []

    def test_clamps_to_pool_size(self):
        embeddings = jax.random.normal(jax.random.PRNGKey(1), (6, 4))
        pool_mask = jnp.array([True, True, False, False, False, False])

        selected = coreset_from_embeddings(embeddings, pool_mask, n_select=99)

        assert len(selected) == 2

    def test_matches_graph_based_coreset(self, gcn_model, sample_graphs):
        """The two entry points must agree, since they share an implementation."""
        pool_graphs, labeled_graphs = sample_graphs[:6], sample_graphs[6:]
        n_pool = len(pool_graphs)

        pool_emb = gcn_model.extract_embeddings(
            batch_graphs(pool_graphs), training=False
        )[:n_pool]
        lab_emb = gcn_model.extract_embeddings(
            batch_graphs(labeled_graphs), training=False
        )[: len(labeled_graphs)]
        embeddings = jnp.concatenate([pool_emb, lab_emb], axis=0)
        pool_mask = jnp.arange(embeddings.shape[0]) < n_pool

        via_graphs = coreset_sampling(
            gcn_model, pool_graphs, labeled_graphs, n_select=3
        )
        via_embeddings = coreset_from_embeddings(embeddings, pool_mask, n_select=3)

        assert via_graphs == via_embeddings
