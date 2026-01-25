"""Tests for acquisition functions using jraph graphs."""

import jax.numpy as jnp
import pytest

from molax.acquisition import (
    combined_acquisition,
    diversity_sampling,
    uncertainty_sampling,
)
from molax.models.gcn import GCNConfig, UncertaintyGCN
from molax.utils.data import smiles_to_jraph


@pytest.fixture
def model():
    """Create a test model."""
    import flax.nnx as nnx

    config = GCNConfig(
        node_features=6,
        hidden_features=[16, 16],
        out_features=1,
        dropout_rate=0.1,
    )
    return UncertaintyGCN(config, rngs=nnx.Rngs(0))


@pytest.fixture
def sample_graphs():
    """Create sample molecular graphs for testing."""
    # Simple SMILES for testing
    smiles_list = [
        "C",  # methane
        "CC",  # ethane
        "CCC",  # propane
        "CCCC",  # butane
        "C=C",  # ethene
        "C#C",  # ethyne
        "CCO",  # ethanol
        "CC=O",  # acetaldehyde
        "CC(=O)O",  # acetic acid
        "c1ccccc1",  # benzene
    ]
    graphs = []
    for smiles in smiles_list:
        try:
            graphs.append(smiles_to_jraph(smiles))
        except Exception:
            pass
    return graphs


class TestUncertaintySampling:
    """Tests for uncertainty_sampling function."""

    def test_returns_correct_shape(self, model, sample_graphs):
        """Test that uncertainty scores have correct shape."""
        scores = uncertainty_sampling(model, sample_graphs, n_samples=5)
        assert scores.shape == (len(sample_graphs),)

    def test_uncertainty_is_non_negative(self, model, sample_graphs):
        """Test that uncertainty scores are non-negative."""
        scores = uncertainty_sampling(model, sample_graphs, n_samples=5)
        assert jnp.all(scores >= 0)

    def test_empty_pool(self, model):
        """Test with empty pool."""
        scores = uncertainty_sampling(model, [], n_samples=5)
        assert scores.shape == (0,)


class TestDiversitySampling:
    """Tests for diversity_sampling function."""

    def test_returns_correct_count(self, sample_graphs):
        """Test that diversity sampling returns the requested number of samples."""
        n_select = 3
        labeled_data = sample_graphs[:2]
        pool_data = sample_graphs[2:]

        selected = diversity_sampling(pool_data, labeled_data, n_select)
        assert len(selected) == n_select

    def test_no_duplicates(self, sample_graphs):
        """Test that selected indices are unique."""
        n_select = 5
        labeled_data = sample_graphs[:2]
        pool_data = sample_graphs[2:]

        selected = diversity_sampling(pool_data, labeled_data, n_select)
        assert len(selected) == len(set(selected))

    def test_indices_within_bounds(self, sample_graphs):
        """Test that selected indices are valid pool indices."""
        n_select = 3
        labeled_data = sample_graphs[:2]
        pool_data = sample_graphs[2:]

        selected = diversity_sampling(pool_data, labeled_data, n_select)
        for idx in selected:
            assert 0 <= idx < len(pool_data)

    def test_empty_pool(self):
        """Test with empty pool."""
        selected = diversity_sampling([], [], 3)
        assert selected == []


class TestCombinedAcquisition:
    """Tests for combined_acquisition function."""

    def test_returns_correct_count(self, model, sample_graphs):
        """Test that combined acquisition returns correct number of samples."""
        n_select = 3
        labeled_data = sample_graphs[:2]
        pool_data = sample_graphs[2:]

        selected = combined_acquisition(model, pool_data, labeled_data, n_select)
        assert len(selected) == n_select

    def test_no_duplicates(self, model, sample_graphs):
        """Test that selected indices are unique."""
        n_select = 4
        labeled_data = sample_graphs[:2]
        pool_data = sample_graphs[2:]

        selected = combined_acquisition(model, pool_data, labeled_data, n_select)
        assert len(selected) == len(set(selected))

    def test_indices_within_bounds(self, model, sample_graphs):
        """Test that selected indices are valid pool indices."""
        n_select = 3
        labeled_data = sample_graphs[:2]
        pool_data = sample_graphs[2:]

        selected = combined_acquisition(model, pool_data, labeled_data, n_select)
        for idx in selected:
            assert 0 <= idx < len(pool_data)
