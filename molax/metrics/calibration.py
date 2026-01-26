"""Calibration metrics for uncertainty quantification.

This module provides tools to quantify how well predicted uncertainties match
actual error frequencies. Includes ECE, NLL scoring, and temperature scaling
for post-hoc calibration.

Key metrics:
- ECE (Expected Calibration Error): Average gap between confidence and accuracy
- NLL (Negative Log-Likelihood): Proper scoring rule for probabilistic predictions
- Sharpness: Average predicted uncertainty (lower = more confident)

For regression with Gaussian uncertainty, calibration means:
- ~68% of targets fall within +/- 1 sigma of prediction
- ~95% of targets fall within +/- 2 sigma of prediction
"""

from typing import Dict, Optional

import jax
import jax.numpy as jnp
from scipy import stats


def negative_log_likelihood(
    mean: jnp.ndarray,
    var: jnp.ndarray,
    targets: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Compute Gaussian negative log-likelihood (proper scoring rule).

    NLL = 0.5 * (log(2*pi*var) + (y - mean)^2 / var)

    Lower is better. This is the proper scoring rule for probabilistic predictions
    with Gaussian likelihood.

    Args:
        mean: Predicted means of shape [n_samples] or [n_samples, 1]
        var: Predicted variances of shape [n_samples] or [n_samples, 1]
        targets: True values of shape [n_samples] or [n_samples, 1]
        mask: Optional boolean mask for valid samples (True = include)

    Returns:
        Scalar NLL value (mean over valid samples)
    """
    # Flatten inputs
    mean = mean.flatten()
    var = var.flatten()
    targets = targets.flatten()

    # Ensure positive variance with small epsilon
    var = jnp.maximum(var, 1e-6)

    # Compute per-sample NLL
    nll = 0.5 * (jnp.log(2 * jnp.pi * var) + (targets - mean) ** 2 / var)

    if mask is not None:
        mask = mask.flatten()
        masked_nll = jnp.where(mask, nll, 0.0)
        return jnp.sum(masked_nll) / (jnp.sum(mask) + 1e-6)

    return jnp.mean(nll)


def compute_calibration_curve(
    predictions: jnp.ndarray,
    uncertainties: jnp.ndarray,
    targets: jnp.ndarray,
    n_bins: int = 10,
    mask: Optional[jnp.ndarray] = None,
) -> Dict[str, jnp.ndarray]:
    """Compute data for reliability diagrams.

    For regression with Gaussian uncertainty, we compute calibration by checking
    what fraction of targets fall within various confidence intervals.

    For each confidence level p (e.g., 50%, 68%, 90%, 95%):
    - Compute interval: [mean - z_p * std, mean + z_p * std]
    - Count fraction of targets within interval (observed coverage)
    - Perfect calibration: observed coverage = expected coverage

    Args:
        predictions: Predicted means of shape [n_samples]
        uncertainties: Predicted variances of shape [n_samples]
        targets: True values of shape [n_samples]
        n_bins: Number of confidence level bins (default: 10)
        mask: Optional boolean mask for valid samples

    Returns:
        Dictionary with:
        - expected_coverage: Expected confidence levels (bin centers)
        - observed_coverage: Actual fraction of targets within interval
        - bin_counts: Number of samples per bin (all same for regression)
    """
    # Flatten inputs
    predictions = predictions.flatten()
    uncertainties = uncertainties.flatten()
    targets = targets.flatten()

    # Apply mask if provided
    if mask is not None:
        mask = mask.flatten()
        valid_idx = jnp.where(mask)[0]
        predictions = predictions[valid_idx]
        uncertainties = uncertainties[valid_idx]
        targets = targets[valid_idx]

    n_samples = len(predictions)
    if n_samples == 0:
        return {
            "expected_coverage": jnp.array([]),
            "observed_coverage": jnp.array([]),
            "bin_counts": jnp.array([]),
        }

    # Convert variance to std
    std = jnp.sqrt(jnp.maximum(uncertainties, 1e-6))

    # Compute z-scores: how many stds away is each target from prediction?
    z_scores = jnp.abs((targets - predictions) / std)

    # Expected confidence levels (coverage percentiles)
    expected_coverage = jnp.linspace(0.1, 0.99, n_bins)

    # For Gaussian: z_p = Phi^(-1)((1+p)/2) gives the z-value for coverage p
    # Using scipy for ppf (percent point function)
    z_thresholds = jnp.array(
        [float(stats.norm.ppf((1 + p) / 2)) for p in expected_coverage]
    )

    # Compute observed coverage at each confidence level
    observed_coverage = []
    for z_thresh in z_thresholds:
        # Fraction of samples with |z| <= z_thresh
        coverage = jnp.mean(z_scores <= z_thresh)
        observed_coverage.append(float(coverage))

    observed_coverage = jnp.array(observed_coverage)

    return {
        "expected_coverage": expected_coverage,
        "observed_coverage": observed_coverage,
        "bin_counts": jnp.full(n_bins, n_samples),
    }


def expected_calibration_error(
    predictions: jnp.ndarray,
    uncertainties: jnp.ndarray,
    targets: jnp.ndarray,
    n_bins: int = 10,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Compute Expected Calibration Error (ECE) for regression.

    ECE measures the average gap between expected and observed confidence
    across different confidence levels. Perfect calibration = 0.

    For regression: ECE = mean(|observed_coverage - expected_coverage|)

    Args:
        predictions: Predicted means of shape [n_samples]
        uncertainties: Predicted variances of shape [n_samples]
        targets: True values of shape [n_samples]
        n_bins: Number of confidence level bins (default: 10)
        mask: Optional boolean mask for valid samples

    Returns:
        Scalar ECE value in [0, 1]. Lower is better.
    """
    calibration_data = compute_calibration_curve(
        predictions, uncertainties, targets, n_bins, mask
    )

    if len(calibration_data["expected_coverage"]) == 0:
        return jnp.array(0.0)

    # ECE = mean absolute difference between expected and observed coverage
    ece = jnp.mean(
        jnp.abs(
            calibration_data["observed_coverage"]
            - calibration_data["expected_coverage"]
        )
    )
    return ece


def sharpness(
    uncertainties: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Compute sharpness (average predicted uncertainty).

    Sharpness measures how confident the model is on average.
    Lower = sharper/more confident predictions.

    Note: Sharpness alone doesn't indicate quality - a model can be
    overconfidently wrong. Use together with calibration metrics.

    Args:
        uncertainties: Predicted variances of shape [n_samples]
        mask: Optional boolean mask for valid samples

    Returns:
        Scalar mean standard deviation
    """
    uncertainties = uncertainties.flatten()
    std = jnp.sqrt(jnp.maximum(uncertainties, 1e-6))

    if mask is not None:
        mask = mask.flatten()
        masked_std = jnp.where(mask, std, 0.0)
        return jnp.sum(masked_std) / (jnp.sum(mask) + 1e-6)

    return jnp.mean(std)


def calibration_error_per_sample(
    predictions: jnp.ndarray,
    uncertainties: jnp.ndarray,
    targets: jnp.ndarray,
) -> jnp.ndarray:
    """Compute per-sample calibration using z-scores.

    For well-calibrated predictions with Gaussian uncertainty:
    z = (y - mean) / std should follow N(0, 1)

    This means |z| should have mean ~0.8 (half-normal distribution).
    Large |z| indicates underestimated uncertainty.
    Small |z| indicates overestimated uncertainty.

    Args:
        predictions: Predicted means of shape [n_samples]
        uncertainties: Predicted variances of shape [n_samples]
        targets: True values of shape [n_samples]

    Returns:
        Per-sample |z-scores| of shape [n_samples]
    """
    predictions = predictions.flatten()
    uncertainties = uncertainties.flatten()
    targets = targets.flatten()

    std = jnp.sqrt(jnp.maximum(uncertainties, 1e-6))
    z_scores = jnp.abs((targets - predictions) / std)

    return z_scores


def mean_squared_error(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Compute Mean Squared Error.

    Args:
        predictions: Predicted values of shape [n_samples]
        targets: True values of shape [n_samples]
        mask: Optional boolean mask for valid samples

    Returns:
        Scalar MSE value
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    se = (predictions - targets) ** 2

    if mask is not None:
        mask = mask.flatten()
        masked_se = jnp.where(mask, se, 0.0)
        return jnp.sum(masked_se) / (jnp.sum(mask) + 1e-6)

    return jnp.mean(se)


def root_mean_squared_error(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Compute Root Mean Squared Error.

    Args:
        predictions: Predicted values of shape [n_samples]
        targets: True values of shape [n_samples]
        mask: Optional boolean mask for valid samples

    Returns:
        Scalar RMSE value
    """
    return jnp.sqrt(mean_squared_error(predictions, targets, mask))


class TemperatureScaling:
    """Temperature scaling for post-hoc calibration.

    Temperature scaling learns a single parameter T to scale uncertainties:
        calibrated_variance = T * predicted_variance

    T > 1 increases uncertainty (model is overconfident)
    T < 1 decreases uncertainty (model is underconfident)

    The temperature is optimized to minimize NLL on a validation set.

    Usage:
        scaler = TemperatureScaling()
        scaler.fit(val_predictions, val_uncertainties, val_targets)
        calibrated_var = scaler.transform(test_uncertainties)

    Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
    """

    def __init__(self):
        """Initialize temperature scaler with T=1 (no scaling)."""
        self._temperature: float = 1.0
        self._fitted: bool = False

    def fit(
        self,
        val_predictions: jnp.ndarray,
        val_uncertainties: jnp.ndarray,
        val_targets: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        max_iter: int = 100,
        lr: float = 0.1,
    ) -> "TemperatureScaling":
        """Optimize temperature on validation data.

        Uses gradient descent to minimize NLL with respect to temperature.

        Args:
            val_predictions: Validation predictions of shape [n_val]
            val_uncertainties: Validation variances of shape [n_val]
            val_targets: Validation targets of shape [n_val]
            mask: Optional boolean mask for valid samples
            max_iter: Maximum optimization iterations
            lr: Learning rate for gradient descent

        Returns:
            Self for method chaining
        """
        val_predictions = val_predictions.flatten()
        val_uncertainties = val_uncertainties.flatten()
        val_targets = val_targets.flatten()

        if mask is not None:
            mask = mask.flatten()

        # Start with T=1
        log_temp = jnp.array(0.0)  # log(1) = 0

        def nll_fn(log_t):
            """NLL as function of log-temperature."""
            t = jnp.exp(log_t)
            scaled_var = t * val_uncertainties
            return negative_log_likelihood(
                val_predictions, scaled_var, val_targets, mask
            )

        # Gradient descent
        grad_fn = jax.grad(nll_fn)

        for _ in range(max_iter):
            grad = grad_fn(log_temp)
            log_temp = log_temp - lr * grad

            # Clamp to reasonable range: T in [0.01, 100]
            log_temp = jnp.clip(log_temp, jnp.log(0.01), jnp.log(100.0))

        self._temperature = float(jnp.exp(log_temp))
        self._fitted = True

        return self

    def transform(self, uncertainties: jnp.ndarray) -> jnp.ndarray:
        """Apply learned temperature scaling to uncertainties.

        Args:
            uncertainties: Predicted variances of shape [n_samples]

        Returns:
            Scaled variances: T * uncertainties
        """
        if not self._fitted:
            raise ValueError(
                "TemperatureScaling must be fitted before transform. Call fit() first."
            )
        return self._temperature * uncertainties

    @property
    def temperature(self) -> float:
        """Get learned temperature value."""
        return self._temperature

    @property
    def is_fitted(self) -> bool:
        """Check if the scaler has been fitted."""
        return self._fitted


def evaluate_calibration(
    predictions: jnp.ndarray,
    uncertainties: jnp.ndarray,
    targets: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute comprehensive calibration metrics.

    Convenience function that computes all calibration metrics at once.

    Args:
        predictions: Predicted means of shape [n_samples]
        uncertainties: Predicted variances of shape [n_samples]
        targets: True values of shape [n_samples]
        mask: Optional boolean mask for valid samples
        n_bins: Number of bins for ECE computation

    Returns:
        Dictionary with:
        - nll: Negative log-likelihood
        - ece: Expected calibration error
        - rmse: Root mean squared error
        - sharpness: Average predicted std
        - mean_z_score: Mean |z-score| (should be ~0.8 for calibrated)
    """
    predictions = predictions.flatten()
    uncertainties = uncertainties.flatten()
    targets = targets.flatten()

    if mask is not None:
        mask = mask.flatten()
        valid_idx = jnp.where(mask)[0]
        predictions = predictions[valid_idx]
        uncertainties = uncertainties[valid_idx]
        targets = targets[valid_idx]

    nll = float(negative_log_likelihood(predictions, uncertainties, targets))
    ece = float(expected_calibration_error(predictions, uncertainties, targets, n_bins))
    rmse = float(root_mean_squared_error(predictions, targets))
    sharp = float(sharpness(uncertainties))
    z_scores = calibration_error_per_sample(predictions, uncertainties, targets)
    mean_z = float(jnp.mean(z_scores))

    return {
        "nll": nll,
        "ece": ece,
        "rmse": rmse,
        "sharpness": sharp,
        "mean_z_score": mean_z,
    }
