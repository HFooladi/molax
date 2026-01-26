"""Visualization utilities for calibration metrics.

This module provides plotting functions for reliability diagrams and
calibration analysis. All functions return matplotlib Axes objects
for easy customization.
"""

from typing import Dict, Optional, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .calibration import (
    compute_calibration_curve,
    expected_calibration_error,
    sharpness,
)


def plot_reliability_diagram(
    predictions: jnp.ndarray,
    uncertainties: jnp.ndarray,
    targets: jnp.ndarray,
    n_bins: int = 10,
    mask: Optional[jnp.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Reliability Diagram",
    color: str = "steelblue",
    show_ece: bool = True,
) -> plt.Axes:
    """Plot reliability diagram showing calibration quality.

    A reliability diagram visualizes how well-calibrated uncertainty
    estimates are. The x-axis shows expected confidence (coverage),
    and the y-axis shows observed confidence (actual coverage).

    Perfect calibration: points lie on the diagonal (y = x).
    Above diagonal: underconfident (uncertainties too high)
    Below diagonal: overconfident (uncertainties too low)

    Args:
        predictions: Predicted means of shape [n_samples]
        uncertainties: Predicted variances of shape [n_samples]
        targets: True values of shape [n_samples]
        n_bins: Number of confidence level bins
        mask: Optional boolean mask for valid samples
        ax: Optional matplotlib Axes. Creates new figure if None.
        title: Plot title
        color: Color for the calibration curve
        show_ece: Whether to display ECE value in legend

    Returns:
        matplotlib Axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Compute calibration curve
    cal_data = compute_calibration_curve(
        predictions, uncertainties, targets, n_bins, mask
    )

    expected = np.array(cal_data["expected_coverage"])
    observed = np.array(cal_data["observed_coverage"])

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")

    # Plot actual calibration
    label = "Model"
    if show_ece:
        ece = float(
            expected_calibration_error(
                predictions, uncertainties, targets, n_bins, mask
            )
        )
        label = f"Model (ECE={ece:.3f})"

    ax.plot(
        expected, observed, "o-", color=color, linewidth=2, markersize=8, label=label
    )

    # Fill gap between perfect and actual
    ax.fill_between(
        expected,
        expected,
        observed,
        alpha=0.2,
        color=color,
        label="Calibration gap",
    )

    ax.set_xlabel("Expected Confidence (Coverage)", fontsize=11)
    ax.set_ylabel("Observed Confidence (Actual Coverage)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return ax


def plot_calibration_comparison(
    results: Dict[str, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    n_bins: int = 10,
    figsize: Tuple[int, int] = (12, 5),
    colors: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """Compare calibration across multiple models.

    Creates a figure with two subplots:
    1. Reliability diagrams for all models overlaid
    2. Bar chart comparing ECE values

    Args:
        results: Dictionary mapping model names to tuples of
                (predictions, uncertainties, targets)
        n_bins: Number of bins for calibration computation
        figsize: Figure size as (width, height)
        colors: Optional dictionary mapping model names to colors

    Returns:
        matplotlib Figure object
    """
    if colors is None:
        color_cycle = plt.cm.tab10.colors
        colors = {
            name: color_cycle[i % len(color_cycle)]
            for i, name in enumerate(results.keys())
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: Reliability diagrams
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect", zorder=0)

    ece_values = {}
    for name, (preds, uncerts, targets) in results.items():
        cal_data = compute_calibration_curve(preds, uncerts, targets, n_bins)
        expected = np.array(cal_data["expected_coverage"])
        observed = np.array(cal_data["observed_coverage"])

        ece = float(expected_calibration_error(preds, uncerts, targets, n_bins))
        ece_values[name] = ece

        ax1.plot(
            expected,
            observed,
            "o-",
            color=colors[name],
            linewidth=2,
            markersize=6,
            label=f"{name} (ECE={ece:.3f})",
        )

    ax1.set_xlabel("Expected Confidence", fontsize=11)
    ax1.set_ylabel("Observed Confidence", fontsize=11)
    ax1.set_title("Reliability Diagram Comparison", fontsize=12)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_aspect("equal")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: ECE bar chart
    names = list(ece_values.keys())
    eces = [ece_values[n] for n in names]
    bar_colors = [colors[n] for n in names]

    bars = ax2.bar(names, eces, color=bar_colors, edgecolor="black", linewidth=1)

    # Add value labels on bars
    for bar, ece in zip(bars, eces):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{ece:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax2.set_ylabel("Expected Calibration Error", fontsize=11)
    ax2.set_title("Calibration Error Comparison", fontsize=12)
    ax2.set_ylim([0, max(eces) * 1.2 + 0.01])
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_uncertainty_vs_error(
    predictions: jnp.ndarray,
    uncertainties: jnp.ndarray,
    targets: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Uncertainty vs Error",
    color: str = "steelblue",
    show_correlation: bool = True,
) -> plt.Axes:
    """Scatter plot of predicted uncertainty vs actual error.

    For well-calibrated models, higher uncertainty should correlate
    with higher error. Points should cluster around the diagonal.

    Args:
        predictions: Predicted means of shape [n_samples]
        uncertainties: Predicted variances of shape [n_samples]
        targets: True values of shape [n_samples]
        mask: Optional boolean mask for valid samples
        ax: Optional matplotlib Axes. Creates new figure if None.
        title: Plot title
        color: Color for scatter points
        show_correlation: Whether to display Pearson correlation

    Returns:
        matplotlib Axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Flatten and apply mask
    predictions = predictions.flatten()
    uncertainties = uncertainties.flatten()
    targets = targets.flatten()

    if mask is not None:
        mask = mask.flatten()
        valid_idx = jnp.where(mask)[0]
        predictions = predictions[valid_idx]
        uncertainties = uncertainties[valid_idx]
        targets = targets[valid_idx]

    # Compute error and std
    errors = np.abs(np.array(predictions) - np.array(targets))
    stds = np.sqrt(np.array(uncertainties))

    # Scatter plot
    ax.scatter(stds, errors, alpha=0.5, c=color, edgecolors="none", s=30)

    # Diagonal line (perfect calibration: error = std)
    max_val = max(np.max(stds), np.max(errors))
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1.5, label="Perfect")

    # Correlation
    if show_correlation and len(stds) > 1:
        corr = np.corrcoef(stds, errors)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"Pearson r = {corr:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
        )

    ax.set_xlabel("Predicted Std", fontsize=11)
    ax.set_ylabel("Actual Error", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return ax


def plot_confidence_histogram(
    uncertainties: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Uncertainty Distribution",
    color: str = "steelblue",
    n_bins: int = 30,
    show_stats: bool = True,
) -> plt.Axes:
    """Histogram of predicted uncertainties (sharpness visualization).

    Visualizes the distribution of predicted standard deviations.
    A sharp model has a narrow distribution (confident predictions).
    A diffuse model has a wide distribution (varying confidence).

    Args:
        uncertainties: Predicted variances of shape [n_samples]
        mask: Optional boolean mask for valid samples
        ax: Optional matplotlib Axes. Creates new figure if None.
        title: Plot title
        color: Color for histogram bars
        n_bins: Number of histogram bins
        show_stats: Whether to display mean and std

    Returns:
        matplotlib Axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Flatten and apply mask
    uncertainties = uncertainties.flatten()

    if mask is not None:
        mask = mask.flatten()
        valid_idx = jnp.where(mask)[0]
        uncertainties = uncertainties[valid_idx]

    # Convert to std
    stds = np.sqrt(np.array(uncertainties))

    # Plot histogram
    ax.hist(stds, bins=n_bins, color=color, edgecolor="black", alpha=0.7)

    # Stats
    if show_stats:
        mean_std = np.mean(stds)
        std_std = np.std(stds)
        sharp = float(sharpness(jnp.array(uncertainties)))

        stats_text = f"Mean: {mean_std:.3f}\nStd: {std_std:.3f}\nSharpness: {sharp:.3f}"
        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel("Predicted Std", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    return ax


def plot_z_score_histogram(
    predictions: jnp.ndarray,
    uncertainties: jnp.ndarray,
    targets: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Z-Score Distribution",
    color: str = "steelblue",
    n_bins: int = 30,
) -> plt.Axes:
    """Histogram of z-scores with reference normal distribution.

    For well-calibrated Gaussian predictions, z = (y - mean) / std
    should follow N(0, 1). This plot compares the actual z-score
    distribution to the expected standard normal.

    Args:
        predictions: Predicted means of shape [n_samples]
        uncertainties: Predicted variances of shape [n_samples]
        targets: True values of shape [n_samples]
        mask: Optional boolean mask for valid samples
        ax: Optional matplotlib Axes. Creates new figure if None.
        title: Plot title
        color: Color for histogram bars
        n_bins: Number of histogram bins

    Returns:
        matplotlib Axes object
    """
    from scipy import stats as scipy_stats

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Flatten and apply mask
    predictions = predictions.flatten()
    uncertainties = uncertainties.flatten()
    targets = targets.flatten()

    if mask is not None:
        mask = mask.flatten()
        valid_idx = jnp.where(mask)[0]
        predictions = predictions[valid_idx]
        uncertainties = uncertainties[valid_idx]
        targets = targets[valid_idx]

    # Compute z-scores
    stds = np.sqrt(np.array(uncertainties) + 1e-6)
    z_scores = (np.array(targets) - np.array(predictions)) / stds

    # Plot histogram (normalized to density)
    ax.hist(
        z_scores,
        bins=n_bins,
        color=color,
        edgecolor="black",
        alpha=0.7,
        density=True,
        label="Observed",
    )

    # Plot reference N(0,1)
    x = np.linspace(-4, 4, 100)
    ax.plot(
        x,
        scipy_stats.norm.pdf(x),
        "r-",
        linewidth=2,
        label="N(0,1) (expected)",
    )

    # Stats
    mean_z = np.mean(z_scores)
    std_z = np.std(z_scores)
    ax.text(
        0.05,
        0.95,
        f"Mean: {mean_z:.3f}\nStd: {std_z:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Z-Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim([-4, 4])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    return ax


def create_calibration_report(
    predictions: jnp.ndarray,
    uncertainties: jnp.ndarray,
    targets: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Create a comprehensive calibration report with multiple plots.

    Generates a figure with four subplots:
    1. Reliability diagram
    2. Uncertainty vs error scatter
    3. Uncertainty histogram
    4. Z-score histogram

    Args:
        predictions: Predicted means of shape [n_samples]
        uncertainties: Predicted variances of shape [n_samples]
        targets: True values of shape [n_samples]
        mask: Optional boolean mask for valid samples
        model_name: Name of the model for titles
        figsize: Figure size as (width, height)

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Reliability diagram
    plot_reliability_diagram(
        predictions,
        uncertainties,
        targets,
        mask=mask,
        ax=axes[0, 0],
        title=f"{model_name}: Reliability Diagram",
    )

    # Uncertainty vs error
    plot_uncertainty_vs_error(
        predictions,
        uncertainties,
        targets,
        mask=mask,
        ax=axes[0, 1],
        title=f"{model_name}: Uncertainty vs Error",
    )

    # Uncertainty histogram
    plot_confidence_histogram(
        uncertainties,
        mask=mask,
        ax=axes[1, 0],
        title=f"{model_name}: Uncertainty Distribution",
    )

    # Z-score histogram
    plot_z_score_histogram(
        predictions,
        uncertainties,
        targets,
        mask=mask,
        ax=axes[1, 1],
        title=f"{model_name}: Z-Score Distribution",
    )

    plt.tight_layout()
    return fig
