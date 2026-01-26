"""Tests for Expected Gradient Length acquisition functions."""

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from molax.acquisition.expected_change import (
    egl_acquisition,
    egl_sampling,
    egl_sampling_batched,
)
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
def sample_graphs():
    """Create sample molecular graphs for testing."""
    smiles_list = [
        "C",  # methane
        "CC",  # ethane
        "CCC",  # propane
        "CCCC",  # butane
        "C=C",  # ethene
        "CCO",  # ethanol
    ]
    graphs = []
    for smiles in smiles_list:
        try:
            graphs.append(smiles_to_jraph(smiles))
        except Exception:
            pass
    return graphs


class TestEGLSampling:
    """Tests for egl_sampling function."""

    def test_returns_correct_shape(self, gcn_model, sample_graphs):
        """Test that EGL scores have correct shape."""
        scores = egl_sampling(gcn_model, sample_graphs)
        assert scores.shape == (len(sample_graphs),)

    def test_scores_are_non_negative(self, gcn_model, sample_graphs):
        """Test that EGL scores (gradient norms) are non-negative."""
        scores = egl_sampling(gcn_model, sample_graphs)
        assert jnp.all(scores >= 0)

    def test_scores_are_finite(self, gcn_model, sample_graphs):
        """Test that EGL scores are finite."""
        scores = egl_sampling(gcn_model, sample_graphs)
        assert jnp.all(jnp.isfinite(scores))

    def test_empty_pool(self, gcn_model):
        """Test with empty pool."""
        scores = egl_sampling(gcn_model, [])
        assert scores.shape == (0,)

    def test_single_sample(self, gcn_model, sample_graphs):
        """Test with single sample."""
        scores = egl_sampling(gcn_model, sample_graphs[:1])
        assert scores.shape == (1,)


class TestEGLSamplingBatched:
    """Tests for egl_sampling_batched function."""

    def test_returns_correct_shape(self, gcn_model, sample_graphs):
        """Test that batched EGL scores have correct shape."""
        scores = egl_sampling_batched(gcn_model, sample_graphs, batch_size=2)
        assert scores.shape == (len(sample_graphs),)

    def test_scores_are_non_negative(self, gcn_model, sample_graphs):
        """Test that batched EGL scores are non-negative."""
        scores = egl_sampling_batched(gcn_model, sample_graphs, batch_size=2)
        assert jnp.all(scores >= 0)

    def test_empty_pool(self, gcn_model):
        """Test with empty pool."""
        scores = egl_sampling_batched(gcn_model, [], batch_size=2)
        assert scores.shape == (0,)


class TestEGLAcquisition:
    """Tests for egl_acquisition function."""

    def test_returns_correct_count(self, gcn_model, sample_graphs):
        """Test that EGL acquisition returns the requested number of samples."""
        n_select = 3
        selected = egl_acquisition(gcn_model, sample_graphs, n_select)
        assert len(selected) == n_select

    def test_no_duplicates(self, gcn_model, sample_graphs):
        """Test that selected indices are unique."""
        n_select = 4
        selected = egl_acquisition(gcn_model, sample_graphs, n_select)
        assert len(selected) == len(set(selected))

    def test_indices_within_bounds(self, gcn_model, sample_graphs):
        """Test that selected indices are valid pool indices."""
        n_select = 3
        selected = egl_acquisition(gcn_model, sample_graphs, n_select)
        for idx in selected:
            assert 0 <= idx < len(sample_graphs)

    def test_empty_pool(self, gcn_model):
        """Test with empty pool."""
        selected = egl_acquisition(gcn_model, [], 3)
        assert selected == []

    def test_n_select_larger_than_pool(self, gcn_model, sample_graphs):
        """Test when n_select > pool size."""
        pool = sample_graphs[:3]
        selected = egl_acquisition(gcn_model, pool, n_select=10)
        assert len(selected) == len(pool)

    def test_batched_mode(self, gcn_model, sample_graphs):
        """Test EGL acquisition with batched computation."""
        n_select = 3
        selected = egl_acquisition(
            gcn_model, sample_graphs, n_select, use_batched=True, batch_size=2
        )
        assert len(selected) == n_select
