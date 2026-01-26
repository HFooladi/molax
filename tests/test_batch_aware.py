"""Tests for batch-aware acquisition functions."""

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from molax.acquisition.batch_aware import (
    batch_bald_sampling,
    combined_batch_acquisition,
    dpp_sampling,
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


class TestBatchBALDSampling:
    """Tests for batch_bald_sampling function."""

    def test_returns_correct_count(self, gcn_model, sample_graphs):
        """Test that BatchBALD returns the requested number of samples."""
        n_select = 3
        selected = batch_bald_sampling(
            gcn_model, sample_graphs, n_select, n_mc_samples=3
        )
        assert len(selected) == n_select

    def test_no_duplicates(self, gcn_model, sample_graphs):
        """Test that selected indices are unique."""
        n_select = 4
        selected = batch_bald_sampling(
            gcn_model, sample_graphs, n_select, n_mc_samples=3
        )
        assert len(selected) == len(set(selected))

    def test_indices_within_bounds(self, gcn_model, sample_graphs):
        """Test that selected indices are valid pool indices."""
        n_select = 3
        selected = batch_bald_sampling(
            gcn_model, sample_graphs, n_select, n_mc_samples=3
        )
        for idx in selected:
            assert 0 <= idx < len(sample_graphs)

    def test_empty_pool(self, gcn_model):
        """Test with empty pool."""
        selected = batch_bald_sampling(gcn_model, [], 3, n_mc_samples=3)
        assert selected == []

    def test_n_select_larger_than_pool(self, gcn_model, sample_graphs):
        """Test when n_select > pool size."""
        pool = sample_graphs[:3]
        selected = batch_bald_sampling(gcn_model, pool, n_select=10, n_mc_samples=3)
        assert len(selected) == len(pool)

    def test_works_with_ensemble(self, ensemble_model, sample_graphs):
        """Test BatchBALD with DeepEnsemble."""
        n_select = 3
        selected = batch_bald_sampling(ensemble_model, sample_graphs, n_select)
        assert len(selected) == n_select

    def test_works_with_evidential(self, evidential_model, sample_graphs):
        """Test BatchBALD with EvidentialGCN."""
        n_select = 3
        selected = batch_bald_sampling(evidential_model, sample_graphs, n_select)
        assert len(selected) == n_select


class TestDPPSampling:
    """Tests for dpp_sampling function."""

    def test_returns_correct_count(self, gcn_model, sample_graphs):
        """Test that DPP returns the requested number of samples."""
        n_select = 3
        selected = dpp_sampling(gcn_model, sample_graphs, n_select)
        assert len(selected) == n_select

    def test_no_duplicates(self, gcn_model, sample_graphs):
        """Test that selected indices are unique."""
        n_select = 4
        selected = dpp_sampling(gcn_model, sample_graphs, n_select)
        assert len(selected) == len(set(selected))

    def test_indices_within_bounds(self, gcn_model, sample_graphs):
        """Test that selected indices are valid pool indices."""
        n_select = 3
        selected = dpp_sampling(gcn_model, sample_graphs, n_select)
        for idx in selected:
            assert 0 <= idx < len(sample_graphs)

    def test_empty_pool(self, gcn_model):
        """Test with empty pool."""
        selected = dpp_sampling(gcn_model, [], 3)
        assert selected == []

    def test_custom_quality_scores(self, gcn_model, sample_graphs):
        """Test DPP with custom quality scores."""
        n_select = 3
        quality_scores = jnp.array([1.0, 0.5, 0.8, 0.2, 0.9, 0.3, 0.7, 0.4])
        quality_scores = quality_scores[: len(sample_graphs)]
        selected = dpp_sampling(
            gcn_model, sample_graphs, n_select, quality_scores=quality_scores
        )
        assert len(selected) == n_select

    def test_custom_bandwidth(self, gcn_model, sample_graphs):
        """Test DPP with custom kernel bandwidth."""
        n_select = 3
        selected = dpp_sampling(
            gcn_model, sample_graphs, n_select, kernel_bandwidth=2.0
        )
        assert len(selected) == n_select

    def test_works_with_ensemble(self, ensemble_model, sample_graphs):
        """Test DPP with DeepEnsemble."""
        n_select = 3
        selected = dpp_sampling(ensemble_model, sample_graphs, n_select)
        assert len(selected) == n_select

    def test_works_with_evidential(self, evidential_model, sample_graphs):
        """Test DPP with EvidentialGCN."""
        n_select = 3
        selected = dpp_sampling(evidential_model, sample_graphs, n_select)
        assert len(selected) == n_select


class TestCombinedBatchAcquisition:
    """Tests for combined_batch_acquisition function."""

    def test_batch_bald_method(self, gcn_model, sample_graphs):
        """Test combined acquisition with batch_bald method."""
        n_select = 3
        selected = combined_batch_acquisition(
            gcn_model, sample_graphs, n_select, method="batch_bald", n_mc_samples=3
        )
        assert len(selected) == n_select

    def test_dpp_method(self, gcn_model, sample_graphs):
        """Test combined acquisition with dpp method."""
        n_select = 3
        selected = combined_batch_acquisition(
            gcn_model, sample_graphs, n_select, method="dpp"
        )
        assert len(selected) == n_select

    def test_invalid_method(self, gcn_model, sample_graphs):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            combined_batch_acquisition(gcn_model, sample_graphs, 3, method="invalid")
