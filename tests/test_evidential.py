"""Tests for Evidential Deep Learning models."""

import flax.nnx as nnx
import jax.numpy as jnp
import jraph
import pytest

from molax.models.evidential import (
    EvidentialConfig,
    EvidentialGCN,
    EvidentialHead,
    create_evidential_optimizer,
    eval_evidential_step,
    evidential_loss,
    get_evidential_uncertainties,
    train_evidential_step,
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
    """Create base GCN config for evidential model."""
    return GCNConfig(
        node_features=6,
        hidden_features=[32],
        out_features=1,
        dropout_rate=0.1,
    )


@pytest.fixture
def evidential_config(base_config):
    """Create evidential config."""
    return EvidentialConfig(base_config=base_config, lambda_reg=0.1)


@pytest.fixture
def model(evidential_config):
    """Create evidential model."""
    return EvidentialGCN(evidential_config, rngs=nnx.Rngs(0))


class TestEvidentialHead:
    """Tests for EvidentialHead module."""

    def test_output_shapes(self):
        """Test that evidential head produces correct output shapes."""
        head = EvidentialHead(in_features=32, rngs=nnx.Rngs(0))
        x = jnp.ones((4, 32))

        gamma, nu, alpha, beta = head(x)

        assert gamma.shape == (4, 1)
        assert nu.shape == (4, 1)
        assert alpha.shape == (4, 1)
        assert beta.shape == (4, 1)

    def test_parameter_constraints(self):
        """Test that NIG parameters satisfy constraints."""
        head = EvidentialHead(in_features=32, rngs=nnx.Rngs(0))
        # Test with various input magnitudes
        for scale in [0.1, 1.0, 10.0]:
            x = jnp.ones((4, 32)) * scale
            gamma, nu, alpha, beta = head(x)

            # gamma can be any value (unbounded)
            assert jnp.all(jnp.isfinite(gamma))
            # nu must be > 0
            assert jnp.all(nu > 0)
            # alpha must be > 1
            assert jnp.all(alpha > 1.0)
            # beta must be > 0
            assert jnp.all(beta > 0)

    def test_parameter_constraints_negative_input(self):
        """Test constraints hold for negative inputs."""
        head = EvidentialHead(in_features=32, rngs=nnx.Rngs(0))
        x = -jnp.ones((4, 32)) * 5.0

        gamma, nu, alpha, beta = head(x)

        assert jnp.all(nu > 0)
        assert jnp.all(alpha > 1.0)
        assert jnp.all(beta > 0)


class TestEvidentialGCN:
    """Tests for EvidentialGCN model."""

    def test_forward_returns_three_outputs(self, model, batched_graphs):
        """Test that forward pass returns mean, total_var, epistemic_var."""
        mean, total_var, epistemic_var = model(batched_graphs)

        assert mean.shape == (4, 1)
        assert total_var.shape == (4, 1)
        assert epistemic_var.shape == (4, 1)

    def test_variance_positive(self, model, batched_graphs):
        """Test that variances are always positive."""
        _, total_var, epistemic_var = model(batched_graphs)

        assert jnp.all(total_var > 0)
        assert jnp.all(epistemic_var > 0)

    def test_total_variance_geq_epistemic(self, model, batched_graphs):
        """Test that total variance >= epistemic variance."""
        _, total_var, epistemic_var = model(batched_graphs)

        # Total = aleatoric + epistemic, so total >= epistemic
        assert jnp.all(total_var >= epistemic_var - 1e-6)

    def test_forward_raw_returns_nig_params(self, model, batched_graphs):
        """Test that forward_raw returns valid NIG parameters."""
        gamma, nu, alpha, beta = model.forward_raw(batched_graphs)

        assert gamma.shape == (4, 1)
        assert jnp.all(nu > 0)
        assert jnp.all(alpha > 1.0)
        assert jnp.all(beta > 0)

    def test_training_mode_uses_dropout(self, evidential_config, batched_graphs):
        """Test that training mode enables dropout."""
        model = EvidentialGCN(evidential_config, rngs=nnx.Rngs(0))

        # Multiple forward passes in training mode should give different results
        # (due to dropout randomness)
        results = []
        for _ in range(5):
            mean, _, _ = model(batched_graphs, training=True)
            results.append(mean)

        # Check that at least some results are different
        all_same = all(jnp.allclose(results[0], r) for r in results[1:])
        # With dropout, results should differ (unless dropout rate is 0)
        if evidential_config.base_config.dropout_rate > 0:
            assert not all_same, "Training mode should produce varying outputs"

    def test_inference_mode_deterministic(self, model, batched_graphs):
        """Test that inference mode is deterministic."""
        mean1, var1, _ = model(batched_graphs, training=False)
        mean2, var2, _ = model(batched_graphs, training=False)

        assert jnp.allclose(mean1, mean2)
        assert jnp.allclose(var1, var2)


class TestEvidentialLoss:
    """Tests for evidential loss function."""

    def test_loss_is_finite(self):
        """Test that loss is finite for valid inputs."""
        gamma = jnp.array([1.0, 2.0, 3.0])
        nu = jnp.array([1.0, 2.0, 3.0])
        alpha = jnp.array([2.0, 3.0, 4.0])  # > 1
        beta = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.5, 2.5, 2.8])

        loss = evidential_loss(gamma, nu, alpha, beta, targets, lambda_reg=0.1)

        assert jnp.isfinite(loss).all()

    def test_higher_error_higher_loss(self):
        """Test that larger prediction errors lead to higher loss."""
        nu = jnp.array([1.0])
        alpha = jnp.array([2.0])
        beta = jnp.array([1.0])
        target = jnp.array([5.0])

        # Small error
        gamma_close = jnp.array([4.9])
        loss_close = evidential_loss(gamma_close, nu, alpha, beta, target, 0.1)

        # Large error
        gamma_far = jnp.array([1.0])
        loss_far = evidential_loss(gamma_far, nu, alpha, beta, target, 0.1)

        assert loss_far > loss_close

    def test_regularization_effect(self):
        """Test that regularization affects loss."""
        gamma = jnp.array([1.0])
        nu = jnp.array([2.0])
        alpha = jnp.array([3.0])
        beta = jnp.array([1.0])
        target = jnp.array([2.0])  # Error of 1.0

        loss_low_reg = evidential_loss(gamma, nu, alpha, beta, target, lambda_reg=0.01)
        loss_high_reg = evidential_loss(gamma, nu, alpha, beta, target, lambda_reg=1.0)

        # Higher regularization should increase loss when there's prediction error
        assert loss_high_reg > loss_low_reg


class TestEvidentialTraining:
    """Tests for evidential training utilities."""

    def test_create_optimizer(self, model):
        """Test optimizer creation."""
        optimizer = create_evidential_optimizer(model, learning_rate=1e-3)
        assert isinstance(optimizer, nnx.Optimizer)

    def test_train_step_returns_loss(self, model, batched_graphs):
        """Test that training step returns a loss value."""
        optimizer = create_evidential_optimizer(model)

        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.ones(4, dtype=bool)

        loss = train_evidential_step(model, optimizer, batched_graphs, labels, mask)

        assert loss.shape == ()  # Scalar
        assert jnp.isfinite(loss)

    def test_train_step_updates_weights(self, evidential_config, batched_graphs):
        """Test that training updates model weights."""
        model = EvidentialGCN(evidential_config, rngs=nnx.Rngs(0))
        optimizer = create_evidential_optimizer(model)

        # Get initial weights
        initial_weights = model.conv_layers[0].linear.kernel.copy()

        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.ones(4, dtype=bool)

        # Train for a few steps
        for _ in range(5):
            train_evidential_step(model, optimizer, batched_graphs, labels, mask)

        # Check weights changed
        new_weights = model.conv_layers[0].linear.kernel
        assert not jnp.allclose(initial_weights, new_weights), (
            "Weights did not change after training"
        )

    def test_train_step_reduces_loss(self, evidential_config, batched_graphs):
        """Test that training reduces loss over time."""
        model = EvidentialGCN(evidential_config, rngs=nnx.Rngs(42))
        optimizer = create_evidential_optimizer(model, learning_rate=1e-2)

        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.ones(4, dtype=bool)

        # Get initial loss
        initial_loss = train_evidential_step(
            model, optimizer, batched_graphs, labels, mask
        )

        # Train more
        for _ in range(50):
            final_loss = train_evidential_step(
                model, optimizer, batched_graphs, labels, mask
            )

        # Loss should decrease (or at least not increase significantly)
        assert final_loss < initial_loss * 1.1, "Loss should decrease during training"


class TestEvidentialEvaluation:
    """Tests for evidential evaluation utilities."""

    def test_eval_step_returns_metrics(self, model, batched_graphs):
        """Test that eval step returns RMSE and uncertainties."""
        labels = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.ones(4, dtype=bool)

        rmse, mean_epistemic, mean_total = eval_evidential_step(
            model, batched_graphs, labels, mask
        )

        assert rmse.shape == ()
        assert mean_epistemic.shape == ()
        assert mean_total.shape == ()

        assert jnp.isfinite(rmse)
        assert rmse >= 0

    def test_get_uncertainties(self, model, batched_graphs):
        """Test uncertainty extraction for acquisition."""
        epistemic, total = get_evidential_uncertainties(model, batched_graphs)

        assert epistemic.shape == (4,)
        assert total.shape == (4,)

        assert jnp.all(epistemic > 0)
        assert jnp.all(total >= epistemic - 1e-6)


class TestEvidentialVsMCDropout:
    """Tests comparing evidential model to MC Dropout uncertainty."""

    def test_evidential_single_pass_vs_mc_multiple(self, base_config, batched_graphs):
        """Test that evidential provides uncertainty in single pass unlike MC."""
        from molax.models.gcn import UncertaintyGCN

        # Single model with MC Dropout needs multiple passes
        single_model = UncertaintyGCN(base_config, rngs=nnx.Rngs(0))

        # MC Dropout requires multiple forward passes
        mc_predictions = []
        for _ in range(20):
            mean, _ = single_model(batched_graphs, training=True)
            mc_predictions.append(mean)
        mc_variance = jnp.var(jnp.stack(mc_predictions), axis=0)

        # Evidential model: single forward pass
        evidential_config = EvidentialConfig(base_config=base_config)
        evidential_model = EvidentialGCN(evidential_config, rngs=nnx.Rngs(0))
        _, total_var, epistemic_var = evidential_model(batched_graphs, training=False)

        # Both should produce valid uncertainties
        assert jnp.all(jnp.isfinite(mc_variance))
        assert jnp.all(jnp.isfinite(total_var))
        assert jnp.all(jnp.isfinite(epistemic_var))

        # Both should be positive
        assert jnp.all(mc_variance >= 0)
        assert jnp.all(total_var > 0)
        assert jnp.all(epistemic_var > 0)

    def test_both_methods_produce_same_shape(self, base_config, batched_graphs):
        """Test that evidential and MC Dropout produce same shape outputs."""
        from molax.models.gcn import UncertaintyGCN

        # MC Dropout uncertainty
        single_model = UncertaintyGCN(base_config, rngs=nnx.Rngs(0))
        mc_predictions = []
        for _ in range(10):
            mean, _ = single_model(batched_graphs, training=True)
            mc_predictions.append(mean)
        mc_variance = jnp.var(jnp.stack(mc_predictions), axis=0)

        # Evidential uncertainty
        evidential_config = EvidentialConfig(base_config=base_config)
        evidential_model = EvidentialGCN(evidential_config, rngs=nnx.Rngs(0))
        _, total_var, _ = evidential_model(batched_graphs, training=False)

        # Should have same shape
        assert mc_variance.squeeze().shape == total_var.squeeze().shape


class TestEvidentialAcquisition:
    """Tests for evidential acquisition functions."""

    def test_evidential_uncertainty_sampling(self, model):
        """Test evidential uncertainty sampling for acquisition."""
        from molax.acquisition.uncertainty import evidential_uncertainty_sampling

        smiles = ["C", "CC", "CCC", "CCO", "CCCC"]
        pool_graphs = [smiles_to_jraph(s) for s in smiles]

        # Get uncertainties using epistemic
        uncertainties_epistemic = evidential_uncertainty_sampling(
            model, pool_graphs, use_epistemic=True
        )
        assert uncertainties_epistemic.shape == (5,)
        assert jnp.all(uncertainties_epistemic > 0)

        # Get uncertainties using total
        uncertainties_total = evidential_uncertainty_sampling(
            model, pool_graphs, use_epistemic=False
        )
        assert uncertainties_total.shape == (5,)
        assert jnp.all(uncertainties_total > 0)

        # Total should be >= epistemic
        assert jnp.all(uncertainties_total >= uncertainties_epistemic - 1e-6)

    def test_combined_evidential_acquisition(self, model):
        """Test combined evidential acquisition."""
        from molax.acquisition.uncertainty import combined_evidential_acquisition

        smiles = ["C", "CC", "CCC", "CCO", "CCCC"]
        pool_graphs = [smiles_to_jraph(s) for s in smiles]
        labeled_graphs = [smiles_to_jraph("O")]

        selected = combined_evidential_acquisition(
            model,
            pool_graphs,
            labeled_graphs,
            n_select=3,
            uncertainty_weight=0.7,
        )

        assert len(selected) == 3
        assert all(isinstance(i, int) for i in selected)
        assert all(0 <= i < 5 for i in selected)
        # Check no duplicates
        assert len(set(selected)) == 3
