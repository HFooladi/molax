"""Tests for Core-Set acquisition functions."""

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from molax.acquisition.coreset import coreset_sampling, coreset_sampling_with_scores
from molax.models.ensemble import DeepEnsemble, EnsembleConfig
from molax.models.evidential import EvidentialConfig, EvidentialGCN
from molax.models.gcn import GCNConfig, UncertaintyGCN
from molax.utils.data import smiles_to_jraph


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
