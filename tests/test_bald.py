"""Tests for BALD acquisition functions."""

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from molax.acquisition.bald import (
    bald_sampling,
    ensemble_bald_sampling,
    evidential_bald_sampling,
)
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


class TestBALDSampling:
    """Tests for bald_sampling function."""

    def test_returns_correct_shape(self, gcn_model, sample_graphs):
        """Test that BALD scores have correct shape."""
        scores = bald_sampling(gcn_model, sample_graphs, n_mc_samples=3)
        assert scores.shape == (len(sample_graphs),)

    def test_scores_are_finite(self, gcn_model, sample_graphs):
        """Test that BALD scores are finite (no NaN or inf)."""
        scores = bald_sampling(gcn_model, sample_graphs, n_mc_samples=3)
        assert jnp.all(jnp.isfinite(scores))

    def test_empty_pool(self, gcn_model):
        """Test with empty pool."""
        scores = bald_sampling(gcn_model, [], n_mc_samples=3)
        assert scores.shape == (0,)

    def test_single_sample(self, gcn_model, sample_graphs):
        """Test with single sample."""
        scores = bald_sampling(gcn_model, sample_graphs[:1], n_mc_samples=3)
        assert scores.shape == (1,)


class TestEnsembleBALDSampling:
    """Tests for ensemble_bald_sampling function."""

    def test_returns_correct_shape(self, ensemble_model, sample_graphs):
        """Test that BALD scores have correct shape."""
        scores = ensemble_bald_sampling(ensemble_model, sample_graphs)
        assert scores.shape == (len(sample_graphs),)

    def test_scores_are_finite(self, ensemble_model, sample_graphs):
        """Test that BALD scores are finite."""
        scores = ensemble_bald_sampling(ensemble_model, sample_graphs)
        assert jnp.all(jnp.isfinite(scores))

    def test_empty_pool(self, ensemble_model):
        """Test with empty pool."""
        scores = ensemble_bald_sampling(ensemble_model, [])
        assert scores.shape == (0,)


class TestEvidentialBALDSampling:
    """Tests for evidential_bald_sampling function."""

    def test_returns_correct_shape(self, evidential_model, sample_graphs):
        """Test that BALD scores have correct shape."""
        scores = evidential_bald_sampling(evidential_model, sample_graphs)
        assert scores.shape == (len(sample_graphs),)

    def test_scores_are_finite(self, evidential_model, sample_graphs):
        """Test that BALD scores are finite."""
        scores = evidential_bald_sampling(evidential_model, sample_graphs)
        assert jnp.all(jnp.isfinite(scores))

    def test_empty_pool(self, evidential_model):
        """Test with empty pool."""
        scores = evidential_bald_sampling(evidential_model, [])
        assert scores.shape == (0,)
