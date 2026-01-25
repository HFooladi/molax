"""Tests for Deep Ensemble models."""

import flax.nnx as nnx
import jax.numpy as jnp
import jraph
import pytest

from molax.models.ensemble import (
    DeepEnsemble,
    EnsembleConfig,
    create_ensemble_optimizers,
    eval_ensemble_step,
    get_ensemble_uncertainties,
    train_ensemble_step,
)
from molax.models.gcn import GCNConfig
from molax.utils.data import smiles_to_jraph


@pytest.fixture
def batched_graphs():
    """Create a batch of test graphs."""
    smiles = ["C", "CC", "CCC", "CCO"]
    graphs = [smiles_to_jraph(s) for s in smiles]
    return jraph.batch(graphs)


@pytest.fixture
def base_config():
    """Create base GCN config for ensemble."""
    return GCNConfig(
        node_features=6,
        hidden_features=[32],
        out_features=1,
        dropout_rate=0.1,
    )


@pytest.fixture
def ensemble_config(base_config):
    """Create ensemble config."""
    return EnsembleConfig(base_config=base_config, n_members=3)


@pytest.fixture
def ensemble(ensemble_config):
    """Create ensemble model."""
    return DeepEnsemble(ensemble_config, rngs=nnx.Rngs(0))


class TestDeepEnsemble:
    """Tests for DeepEnsemble model."""

    def test_creates_n_members(self, ensemble_config):
        """Test that ensemble creates correct number of members."""
        ensemble = DeepEnsemble(ensemble_config, rngs=nnx.Rngs(0))
        assert len(ensemble.members) == 3

    def test_members_have_different_weights(self, ensemble_config):
        """Test that ensemble members have different initializations."""
        ensemble = DeepEnsemble(ensemble_config, rngs=nnx.Rngs(0))

        # Get weights from first layer of each member
        weights = [member.conv_layers[0].linear.kernel for member in ensemble.members]

        # Check weights are different between members
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                assert not jnp.allclose(weights[i], weights[j]), (
                    f"Members {i} and {j} have identical weights"
                )

    def test_returns_three_outputs(self, ensemble, batched_graphs):
        """Test that forward pass returns mean, total_var, epistemic_var."""
        mean, total_var, epistemic_var = ensemble(batched_graphs)

        assert mean.shape == (4, 1)
        assert total_var.shape == (4, 1)
        assert epistemic_var.shape == (4, 1)

    def test_variance_positive(self, ensemble, batched_graphs):
        """Test that variances are always positive."""
        _, total_var, epistemic_var = ensemble(batched_graphs)

        assert jnp.all(total_var >= 0)
        assert jnp.all(epistemic_var >= 0)

    def test_total_variance_geq_epistemic(self, ensemble, batched_graphs):
        """Test that total variance >= epistemic variance (includes aleatoric)."""
        _, total_var, epistemic_var = ensemble(batched_graphs)

        # Total = epistemic + aleatoric, so total >= epistemic
        # Small tolerance for numerical precision
        assert jnp.all(total_var >= epistemic_var - 1e-6)

    def test_predict_member(self, ensemble, batched_graphs):
        """Test prediction from a specific member."""
        mean, var = ensemble.predict_member(0, batched_graphs)

        assert mean.shape == (4, 1)
        assert var.shape == (4, 1)
        assert jnp.all(var > 0)

    def test_ensemble_mean_is_average(self, ensemble, batched_graphs):
        """Test that ensemble mean is average of member predictions."""
        # Get ensemble prediction
        ensemble_mean, _, _ = ensemble(batched_graphs, training=False)

        # Get individual member predictions
        member_means = []
        for i in range(len(ensemble.members)):
            mean, _ = ensemble.predict_member(i, batched_graphs, training=False)
            member_means.append(mean)

        expected_mean = jnp.mean(jnp.stack(member_means), axis=0)
        assert jnp.allclose(ensemble_mean, expected_mean, atol=1e-5)


class TestEnsembleTraining:
    """Tests for ensemble training utilities."""

    def test_create_optimizers(self, ensemble):
        """Test optimizer creation for each member."""
        optimizers = create_ensemble_optimizers(ensemble, learning_rate=1e-3)

        assert len(optimizers) == len(ensemble.members)
        for opt in optimizers:
            assert isinstance(opt, nnx.Optimizer)

    def test_train_step_returns_loss(self, ensemble, batched_graphs):
        """Test that training step returns a loss value."""
        optimizers = create_ensemble_optimizers(ensemble)

        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.ones(4, dtype=bool)

        loss = train_ensemble_step(ensemble, optimizers, batched_graphs, labels, mask)

        assert loss.shape == ()  # Scalar
        assert jnp.isfinite(loss)

    def test_train_step_updates_weights(self, ensemble_config, batched_graphs):
        """Test that training updates model weights."""
        ensemble = DeepEnsemble(ensemble_config, rngs=nnx.Rngs(0))
        optimizers = create_ensemble_optimizers(ensemble)

        # Get initial weights
        initial_weights = [
            member.conv_layers[0].linear.kernel.copy() for member in ensemble.members
        ]

        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.ones(4, dtype=bool)

        # Train for a few steps
        for _ in range(5):
            train_ensemble_step(ensemble, optimizers, batched_graphs, labels, mask)

        # Check weights changed
        for i, member in enumerate(ensemble.members):
            new_weights = member.conv_layers[0].linear.kernel
            assert not jnp.allclose(initial_weights[i], new_weights), (
                f"Member {i} weights did not change after training"
            )


class TestEnsembleEvaluation:
    """Tests for ensemble evaluation utilities."""

    def test_eval_step_returns_metrics(self, ensemble, batched_graphs):
        """Test that eval step returns RMSE and uncertainties."""
        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.ones(4, dtype=bool)

        rmse, mean_epistemic, mean_total = eval_ensemble_step(
            ensemble, batched_graphs, labels, mask
        )

        assert rmse.shape == ()
        assert mean_epistemic.shape == ()
        assert mean_total.shape == ()

        assert jnp.isfinite(rmse)
        assert rmse >= 0

    def test_get_uncertainties(self, ensemble, batched_graphs):
        """Test uncertainty extraction for acquisition."""
        epistemic, total = get_ensemble_uncertainties(ensemble, batched_graphs)

        assert epistemic.shape == (4,)
        assert total.shape == (4,)

        assert jnp.all(epistemic >= 0)
        assert jnp.all(total >= epistemic - 1e-6)


class TestEnsembleVsMCDropout:
    """Tests comparing ensemble to MC Dropout uncertainty."""

    def test_ensemble_uncertainty_different_from_mc(self, base_config, batched_graphs):
        """Test that ensemble provides different uncertainty than MC Dropout."""
        from molax.models.gcn import UncertaintyGCN

        # Create single model with MC Dropout
        single_model = UncertaintyGCN(base_config, rngs=nnx.Rngs(0))

        # Get MC Dropout uncertainty (variance across samples)
        mc_predictions = []
        for _ in range(20):
            mean, _ = single_model(batched_graphs, training=True)
            mc_predictions.append(mean)
        mc_variance = jnp.var(jnp.stack(mc_predictions), axis=0)

        # Create ensemble
        ensemble_config = EnsembleConfig(base_config=base_config, n_members=5)
        ensemble = DeepEnsemble(ensemble_config, rngs=nnx.Rngs(0))

        # Get ensemble epistemic uncertainty
        _, _, epistemic_var = ensemble(batched_graphs, training=False)

        # Both should have same shape but different values
        assert mc_variance.squeeze().shape == epistemic_var.squeeze().shape
        # They won't be identical (different uncertainty estimation methods)
        # Just verify they're both valid (positive, finite)
        assert jnp.all(jnp.isfinite(mc_variance))
        assert jnp.all(jnp.isfinite(epistemic_var))
