"""Calibration metrics and visualization for uncertainty quantification.

This module provides tools to evaluate and visualize how well predicted
uncertainties match actual errors. Key components:

Metrics:
- negative_log_likelihood: Proper scoring rule for probabilistic predictions
- expected_calibration_error: Average gap between confidence and accuracy
- compute_calibration_curve: Data for reliability diagrams
- sharpness: Average predicted uncertainty
- calibration_error_per_sample: Per-sample z-scores

Calibration:
- TemperatureScaling: Post-hoc calibration via temperature optimization

Visualization:
- plot_reliability_diagram: Calibration quality visualization
- plot_calibration_comparison: Compare multiple models
- plot_uncertainty_vs_error: Scatter of predicted vs actual uncertainty
- plot_confidence_histogram: Distribution of predicted uncertainties
- plot_z_score_histogram: Z-score distribution vs expected N(0,1)
- create_calibration_report: Comprehensive multi-plot report

Example:
    from molax.metrics import (
        expected_calibration_error,
        evaluate_calibration,
        plot_reliability_diagram,
        TemperatureScaling,
    )

    # Get predictions from model
    mean, var = model(graphs, training=False)

    # Evaluate calibration
    metrics = evaluate_calibration(mean, var, targets)
    print(f"ECE: {metrics['ece']:.4f}, NLL: {metrics['nll']:.4f}")

    # Plot reliability diagram
    plot_reliability_diagram(mean, var, targets)

    # Apply temperature scaling
    scaler = TemperatureScaling()
    scaler.fit(val_mean, val_var, val_targets)
    calibrated_var = scaler.transform(var)
"""

from .calibration import (
    TemperatureScaling,
    calibration_error_per_sample,
    compute_calibration_curve,
    evaluate_calibration,
    expected_calibration_error,
    mean_squared_error,
    negative_log_likelihood,
    root_mean_squared_error,
    sharpness,
)
from .visualization import (
    create_calibration_report,
    plot_calibration_comparison,
    plot_confidence_histogram,
    plot_reliability_diagram,
    plot_uncertainty_vs_error,
    plot_z_score_histogram,
)

__all__ = [
    # Metrics
    "negative_log_likelihood",
    "expected_calibration_error",
    "compute_calibration_curve",
    "sharpness",
    "calibration_error_per_sample",
    "mean_squared_error",
    "root_mean_squared_error",
    "evaluate_calibration",
    # Calibration
    "TemperatureScaling",
    # Visualization
    "plot_reliability_diagram",
    "plot_calibration_comparison",
    "plot_uncertainty_vs_error",
    "plot_confidence_histogram",
    "plot_z_score_histogram",
    "create_calibration_report",
]
