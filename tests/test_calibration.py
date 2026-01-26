"""Tests for calibration metrics and visualization."""

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from molax.metrics import (
    TemperatureScaling,
    calibration_error_per_sample,
    compute_calibration_curve,
    create_calibration_report,
    evaluate_calibration,
    expected_calibration_error,
    mean_squared_error,
    negative_log_likelihood,
    plot_calibration_comparison,
    plot_confidence_histogram,
    plot_reliability_diagram,
    plot_uncertainty_vs_error,
    plot_z_score_histogram,
    root_mean_squared_error,
    sharpness,
)

# Use non-interactive backend for testing
matplotlib.use("Agg")


@pytest.fixture
def perfect_calibration_data():
    """Generate perfectly calibrated data.

    True values are sampled from N(prediction, variance), so z-scores
    should follow N(0,1) exactly.
    """
    np.random.seed(42)
    n = 500
    predictions = np.random.randn(n) * 2  # Random predictions
    variances = np.abs(np.random.randn(n)) + 0.5  # Random positive variances
    stds = np.sqrt(variances)
    # Sample targets from the predicted distributions
    targets = predictions + stds * np.random.randn(n)

    return (
        jnp.array(predictions),
        jnp.array(variances),
        jnp.array(targets),
    )


@pytest.fixture
def overconfident_data():
    """Generate data where model is overconfident (underestimates uncertainty)."""
    np.random.seed(42)
    n = 500
    predictions = np.random.randn(n) * 2
    # Very small variances (overconfident)
    variances = np.ones(n) * 0.01
    # But actual errors are much larger
    targets = predictions + np.random.randn(n) * 2

    return (
        jnp.array(predictions),
        jnp.array(variances),
        jnp.array(targets),
    )


@pytest.fixture
def underconfident_data():
    """Generate data where model is underconfident (overestimates uncertainty)."""
    np.random.seed(42)
    n = 500
    predictions = np.random.randn(n) * 2
    # Very large variances (underconfident)
    variances = np.ones(n) * 100
    # But actual errors are small
    targets = predictions + np.random.randn(n) * 0.1

    return (
        jnp.array(predictions),
        jnp.array(variances),
        jnp.array(targets),
    )


@pytest.fixture
def sample_data():
    """Simple sample data for basic tests."""
    predictions = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    variances = jnp.array([0.1, 0.2, 0.3, 0.2, 0.1])
    targets = jnp.array([1.1, 1.8, 3.2, 4.1, 4.9])
    return predictions, variances, targets


class TestNegativeLogLikelihood:
    """Tests for negative_log_likelihood function."""

    def test_returns_finite(self, sample_data):
        """Test that NLL returns a finite value."""
        preds, vars, targets = sample_data
        nll = negative_log_likelihood(preds, vars, targets)
        assert jnp.isfinite(nll)

    def test_lower_for_accurate_predictions(self):
        """Test that NLL is lower for more accurate predictions."""
        targets = jnp.array([1.0, 2.0, 3.0])
        variances = jnp.array([0.1, 0.1, 0.1])

        # Accurate predictions
        accurate = jnp.array([1.0, 2.0, 3.0])
        nll_accurate = negative_log_likelihood(accurate, variances, targets)

        # Inaccurate predictions
        inaccurate = jnp.array([2.0, 3.0, 4.0])
        nll_inaccurate = negative_log_likelihood(inaccurate, variances, targets)

        assert nll_accurate < nll_inaccurate

    def test_lower_for_appropriate_variance(self):
        """Test that NLL is lower when variance matches actual error."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.5, 2.5, 3.5])  # Error of 0.5 each

        # Variance matches error: (0.5)^2 = 0.25
        var_match = jnp.array([0.25, 0.25, 0.25])
        nll_match = negative_log_likelihood(predictions, var_match, targets)

        # Variance too small (overconfident)
        var_small = jnp.array([0.01, 0.01, 0.01])
        nll_small = negative_log_likelihood(predictions, var_small, targets)

        # Variance too large (underconfident)
        var_large = jnp.array([10.0, 10.0, 10.0])
        nll_large = negative_log_likelihood(predictions, var_large, targets)

        assert nll_match < nll_small
        assert nll_match < nll_large

    def test_handles_masking(self, sample_data):
        """Test that masking works correctly."""
        preds, vars, targets = sample_data
        mask = jnp.array([True, True, False, False, True])

        nll_masked = negative_log_likelihood(preds, vars, targets, mask)
        assert jnp.isfinite(nll_masked)

        # Compare to manual computation on masked subset
        masked_preds = preds[jnp.where(mask)]
        masked_vars = vars[jnp.where(mask)]
        masked_targets = targets[jnp.where(mask)]
        nll_manual = negative_log_likelihood(masked_preds, masked_vars, masked_targets)

        assert jnp.allclose(nll_masked, nll_manual, rtol=1e-5)


class TestExpectedCalibrationError:
    """Tests for expected_calibration_error function."""

    def test_returns_value_in_range(self, sample_data):
        """Test that ECE returns value in [0, 1]."""
        preds, vars, targets = sample_data
        ece = expected_calibration_error(preds, vars, targets)
        assert 0 <= float(ece) <= 1

    def test_perfect_calibration_low_ece(self, perfect_calibration_data):
        """Test that perfectly calibrated data has low ECE."""
        preds, vars, targets = perfect_calibration_data
        ece = expected_calibration_error(preds, vars, targets, n_bins=10)
        # Should be close to 0 for perfectly calibrated data
        assert float(ece) < 0.1

    def test_overconfident_high_ece(self, overconfident_data):
        """Test that overconfident predictions have high ECE."""
        preds, vars, targets = overconfident_data
        ece = expected_calibration_error(preds, vars, targets)
        # Should be high when model is overconfident
        assert float(ece) > 0.2

    def test_underconfident_high_ece(self, underconfident_data):
        """Test that underconfident predictions have high ECE."""
        preds, vars, targets = underconfident_data
        ece = expected_calibration_error(preds, vars, targets)
        # Should be high when model is underconfident
        assert float(ece) > 0.2

    def test_handles_masking(self, sample_data):
        """Test that masking works correctly."""
        preds, vars, targets = sample_data
        mask = jnp.array([True, True, True, False, False])

        ece = expected_calibration_error(preds, vars, targets, mask=mask)
        assert jnp.isfinite(ece)


class TestCalibrationCurve:
    """Tests for compute_calibration_curve function."""

    def test_returns_correct_keys(self, sample_data):
        """Test that calibration curve returns all expected keys."""
        preds, vars, targets = sample_data
        result = compute_calibration_curve(preds, vars, targets)

        assert "expected_coverage" in result
        assert "observed_coverage" in result
        assert "bin_counts" in result

    def test_coverage_in_range(self, sample_data):
        """Test that coverage values are in [0, 1]."""
        preds, vars, targets = sample_data
        result = compute_calibration_curve(preds, vars, targets, n_bins=5)

        assert jnp.all(result["expected_coverage"] >= 0)
        assert jnp.all(result["expected_coverage"] <= 1)
        assert jnp.all(result["observed_coverage"] >= 0)
        assert jnp.all(result["observed_coverage"] <= 1)

    def test_expected_coverage_monotonic(self, sample_data):
        """Test that expected coverage is monotonically increasing."""
        preds, vars, targets = sample_data
        result = compute_calibration_curve(preds, vars, targets, n_bins=5)

        expected = result["expected_coverage"]
        assert jnp.all(expected[1:] >= expected[:-1])

    def test_perfect_calibration_matches_expected(self, perfect_calibration_data):
        """Test that perfectly calibrated data has observed ~= expected."""
        preds, vars, targets = perfect_calibration_data
        result = compute_calibration_curve(preds, vars, targets, n_bins=10)

        # Observed should be close to expected
        diff = jnp.abs(result["observed_coverage"] - result["expected_coverage"])
        assert jnp.mean(diff) < 0.1


class TestSharpness:
    """Tests for sharpness function."""

    def test_positive_sharpness(self, sample_data):
        """Test that sharpness is always positive."""
        _, vars, _ = sample_data
        sharp = sharpness(vars)
        assert float(sharp) > 0

    def test_lower_for_smaller_variance(self):
        """Test that sharpness is lower for smaller variances."""
        small_vars = jnp.array([0.01, 0.01, 0.01])
        large_vars = jnp.array([10.0, 10.0, 10.0])

        sharp_small = sharpness(small_vars)
        sharp_large = sharpness(large_vars)

        assert sharp_small < sharp_large

    def test_handles_masking(self, sample_data):
        """Test that masking works correctly."""
        _, vars, _ = sample_data
        mask = jnp.array([True, True, False, False, True])

        sharp_masked = sharpness(vars, mask)
        assert jnp.isfinite(sharp_masked)

        # Compare to manual computation
        masked_vars = vars[jnp.where(mask)]
        sharp_manual = sharpness(masked_vars)
        assert jnp.allclose(sharp_masked, sharp_manual, rtol=1e-5)


class TestCalibrationErrorPerSample:
    """Tests for calibration_error_per_sample function."""

    def test_returns_correct_shape(self, sample_data):
        """Test that per-sample calibration returns correct shape."""
        preds, vars, targets = sample_data
        z_scores = calibration_error_per_sample(preds, vars, targets)
        assert z_scores.shape == preds.shape

    def test_z_scores_positive(self, sample_data):
        """Test that z-scores (absolute) are non-negative."""
        preds, vars, targets = sample_data
        z_scores = calibration_error_per_sample(preds, vars, targets)
        assert jnp.all(z_scores >= 0)

    def test_perfect_calibration_mean_zscore(self, perfect_calibration_data):
        """Test that perfectly calibrated data has mean |z| near expected value.

        For N(0,1), E[|Z|] = sqrt(2/pi) ~= 0.798
        """
        preds, vars, targets = perfect_calibration_data
        z_scores = calibration_error_per_sample(preds, vars, targets)
        mean_z = float(jnp.mean(z_scores))

        expected_mean = np.sqrt(2 / np.pi)  # ~0.798
        assert abs(mean_z - expected_mean) < 0.1


class TestMSEandRMSE:
    """Tests for MSE and RMSE functions."""

    def test_mse_correct(self):
        """Test MSE computation."""
        preds = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 2.0, 4.0])  # One error of 1.0

        mse = mean_squared_error(preds, targets)
        expected_mse = (0 + 0 + 1) / 3
        assert jnp.allclose(mse, expected_mse)

    def test_rmse_correct(self):
        """Test RMSE computation."""
        preds = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 2.0, 4.0])

        rmse = root_mean_squared_error(preds, targets)
        expected_rmse = jnp.sqrt((0 + 0 + 1) / 3)
        assert jnp.allclose(rmse, expected_rmse)

    def test_perfect_prediction_zero_error(self):
        """Test that perfect predictions give zero error."""
        preds = jnp.array([1.0, 2.0, 3.0])
        mse = mean_squared_error(preds, preds)
        assert jnp.allclose(mse, 0.0)


class TestTemperatureScaling:
    """Tests for TemperatureScaling class."""

    def test_initial_temperature_is_one(self):
        """Test that initial temperature is 1.0."""
        scaler = TemperatureScaling()
        assert scaler.temperature == 1.0

    def test_not_fitted_raises_error(self):
        """Test that transform raises error if not fitted."""
        scaler = TemperatureScaling()
        with pytest.raises(ValueError, match="must be fitted"):
            scaler.transform(jnp.array([1.0, 2.0]))

    def test_fit_returns_self(self, sample_data):
        """Test that fit returns self for method chaining."""
        preds, vars, targets = sample_data
        scaler = TemperatureScaling()
        result = scaler.fit(preds, vars, targets)
        assert result is scaler

    def test_fit_sets_positive_temperature(self, sample_data):
        """Test that fitting results in positive temperature."""
        preds, vars, targets = sample_data
        scaler = TemperatureScaling()
        scaler.fit(preds, vars, targets)

        assert scaler.temperature > 0

    def test_fit_marks_as_fitted(self, sample_data):
        """Test that fitting marks scaler as fitted."""
        preds, vars, targets = sample_data
        scaler = TemperatureScaling()
        assert not scaler.is_fitted

        scaler.fit(preds, vars, targets)
        assert scaler.is_fitted

    def test_transform_scales_variance(self, sample_data):
        """Test that transform scales variance by temperature."""
        preds, vars, targets = sample_data
        scaler = TemperatureScaling()
        scaler.fit(preds, vars, targets)

        scaled_vars = scaler.transform(vars)
        expected = scaler.temperature * vars

        assert jnp.allclose(scaled_vars, expected)

    def test_overconfident_temperature_increases(self, overconfident_data):
        """Test that temperature > 1 for overconfident model."""
        preds, vars, targets = overconfident_data
        scaler = TemperatureScaling()
        scaler.fit(preds, vars, targets, max_iter=200)

        # For overconfident model, need to increase variance
        assert scaler.temperature > 1.0

    def test_temperature_scaling_improves_nll(self, overconfident_data):
        """Test that temperature scaling improves NLL on validation data."""
        preds, vars, targets = overconfident_data

        # NLL before scaling
        nll_before = float(negative_log_likelihood(preds, vars, targets))

        # Fit and scale
        scaler = TemperatureScaling()
        scaler.fit(preds, vars, targets, max_iter=200)
        scaled_vars = scaler.transform(vars)

        # NLL after scaling
        nll_after = float(negative_log_likelihood(preds, scaled_vars, targets))

        # Should improve (decrease) NLL
        assert nll_after < nll_before


class TestEvaluateCalibration:
    """Tests for evaluate_calibration convenience function."""

    def test_returns_all_metrics(self, sample_data):
        """Test that evaluate_calibration returns all expected metrics."""
        preds, vars, targets = sample_data
        metrics = evaluate_calibration(preds, vars, targets)

        assert "nll" in metrics
        assert "ece" in metrics
        assert "rmse" in metrics
        assert "sharpness" in metrics
        assert "mean_z_score" in metrics

    def test_all_metrics_finite(self, sample_data):
        """Test that all returned metrics are finite."""
        preds, vars, targets = sample_data
        metrics = evaluate_calibration(preds, vars, targets)

        for key, value in metrics.items():
            assert np.isfinite(value), f"{key} is not finite"


class TestVisualization:
    """Tests for visualization functions."""

    def test_reliability_diagram_returns_axes(self, sample_data):
        """Test that reliability diagram returns Axes object."""
        preds, vars, targets = sample_data
        ax = plot_reliability_diagram(preds, vars, targets)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_reliability_diagram_with_custom_ax(self, sample_data):
        """Test reliability diagram with provided axes."""
        preds, vars, targets = sample_data
        fig, ax = plt.subplots()
        result_ax = plot_reliability_diagram(preds, vars, targets, ax=ax)
        assert result_ax is ax
        plt.close()

    def test_calibration_comparison_returns_figure(self, sample_data):
        """Test that calibration comparison returns Figure object."""
        preds, vars, targets = sample_data
        results = {
            "Model A": (preds, vars, targets),
            "Model B": (preds, vars * 2, targets),
        }
        fig = plot_calibration_comparison(results)
        assert isinstance(fig, plt.Figure)
        plt.close()

    def test_uncertainty_vs_error_returns_axes(self, sample_data):
        """Test that uncertainty vs error plot returns Axes object."""
        preds, vars, targets = sample_data
        ax = plot_uncertainty_vs_error(preds, vars, targets)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_confidence_histogram_returns_axes(self, sample_data):
        """Test that confidence histogram returns Axes object."""
        _, vars, _ = sample_data
        ax = plot_confidence_histogram(vars)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_z_score_histogram_returns_axes(self, sample_data):
        """Test that z-score histogram returns Axes object."""
        preds, vars, targets = sample_data
        ax = plot_z_score_histogram(preds, vars, targets)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_calibration_report_returns_figure(self, sample_data):
        """Test that calibration report returns Figure object."""
        preds, vars, targets = sample_data
        fig = create_calibration_report(preds, vars, targets)
        assert isinstance(fig, plt.Figure)
        plt.close()

    def test_visualizations_handle_masking(self, sample_data):
        """Test that visualizations work with masking."""
        preds, vars, targets = sample_data
        mask = jnp.array([True, True, True, False, False])

        # All should work without error
        plot_reliability_diagram(preds, vars, targets, mask=mask)
        plt.close()

        plot_uncertainty_vs_error(preds, vars, targets, mask=mask)
        plt.close()

        plot_confidence_histogram(vars, mask=mask)
        plt.close()

        plot_z_score_histogram(preds, vars, targets, mask=mask)
        plt.close()


class TestIntegrationWithModels:
    """Integration tests with actual model outputs."""

    def test_with_random_model_outputs(self):
        """Test metrics work with typical model output shapes."""
        np.random.seed(42)
        n_samples = 100

        # Simulate model outputs
        mean = jnp.array(np.random.randn(n_samples))
        var = jnp.array(np.abs(np.random.randn(n_samples)) + 0.1)
        targets = jnp.array(np.random.randn(n_samples))

        # All metrics should work
        nll = negative_log_likelihood(mean, var, targets)
        ece = expected_calibration_error(mean, var, targets)
        sharp = sharpness(var)
        metrics = evaluate_calibration(mean, var, targets)

        assert jnp.isfinite(nll)
        assert jnp.isfinite(ece)
        assert jnp.isfinite(sharp)
        assert all(np.isfinite(v) for v in metrics.values())

    def test_with_2d_outputs(self):
        """Test metrics work with 2D model outputs (n_samples, 1)."""
        np.random.seed(42)
        n_samples = 50

        # 2D outputs as from model
        mean = jnp.array(np.random.randn(n_samples, 1))
        var = jnp.array(np.abs(np.random.randn(n_samples, 1)) + 0.1)
        targets = jnp.array(np.random.randn(n_samples, 1))

        # Should handle 2D inputs
        nll = negative_log_likelihood(mean, var, targets)
        ece = expected_calibration_error(mean, var, targets)

        assert jnp.isfinite(nll)
        assert jnp.isfinite(ece)

    def test_empty_mask_handling(self):
        """Test that empty mask (no valid samples) is handled gracefully."""
        mean = jnp.array([1.0, 2.0, 3.0])
        var = jnp.array([0.1, 0.1, 0.1])
        targets = jnp.array([1.0, 2.0, 3.0])
        mask = jnp.array([False, False, False])

        # Should not raise errors
        ece = expected_calibration_error(mean, var, targets, mask=mask)
        assert jnp.isfinite(ece) or ece == 0.0
