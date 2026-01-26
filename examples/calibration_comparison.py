"""Calibration Comparison: MC Dropout vs Ensemble vs Evidential.

This example demonstrates the calibration metrics module by:
1. Training three model types on ESOL dataset
2. Computing calibration metrics for each
3. Visualizing reliability diagrams
4. Applying temperature scaling for post-hoc calibration

The example shows how to use:
- expected_calibration_error, negative_log_likelihood, sharpness
- TemperatureScaling for post-hoc calibration
- plot_reliability_diagram, plot_calibration_comparison
- evaluate_calibration for comprehensive metrics
"""

import time
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jraph
import matplotlib.pyplot as plt
import optax

from molax.metrics import (
    TemperatureScaling,
    create_calibration_report,
    evaluate_calibration,
    plot_calibration_comparison,
)
from molax.models.ensemble import (
    DeepEnsemble,
    EnsembleConfig,
    create_ensemble_optimizers,
    train_ensemble_step,
)
from molax.models.evidential import (
    EvidentialConfig,
    EvidentialGCN,
    create_evidential_optimizer,
    train_evidential_step,
)
from molax.models.gcn import GCNConfig, UncertaintyGCN
from molax.utils.data import MolecularDataset

# Configuration
DATASET_PATH = Path(__file__).parent.parent / "datasets" / "esol.csv"
OUTPUT_DIR = Path(__file__).parent / "assets"
N_EPOCHS = 100
SEED = 42

print("=" * 70)
print("Calibration Comparison: MC Dropout vs Ensemble vs Evidential")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")


def train_mc_dropout_model(
    train_graphs: jraph.GraphsTuple,
    train_labels: jnp.ndarray,
    train_mask: jnp.ndarray,
    n_features: int,
    n_epochs: int,
    seed: int,
) -> UncertaintyGCN:
    """Train a single model with MC Dropout for uncertainty."""
    config = GCNConfig(
        node_features=n_features,
        hidden_features=[64, 64],
        out_features=1,
        dropout_rate=0.2,
    )
    model = UncertaintyGCN(config, rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, optimizer):
        def loss_fn(model):
            mean, var = model(train_graphs, training=True)
            mean, var = mean.squeeze(-1), var.squeeze(-1)
            nll = 0.5 * (
                jnp.log(var + 1e-6) + (train_labels - mean) ** 2 / (var + 1e-6)
            )
            masked_nll = jnp.where(train_mask, nll, 0.0)
            return jnp.sum(masked_nll) / jnp.sum(train_mask)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    for epoch in range(n_epochs):
        loss = train_step(model, optimizer)
        if (epoch + 1) % 25 == 0:
            print(f"    MC Dropout - Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.4f}")

    return model


def get_mc_dropout_predictions(
    model: UncertaintyGCN,
    graphs: jraph.GraphsTuple,
    n_samples: int = 20,
) -> tuple:
    """Get predictions with MC Dropout uncertainty estimation."""
    all_means = []
    for _ in range(n_samples):
        mean, _ = model(graphs, training=True)  # Dropout active
        all_means.append(mean.squeeze(-1))

    all_means = jnp.stack(all_means, axis=0)
    pred_mean = jnp.mean(all_means, axis=0)
    pred_var = jnp.var(all_means, axis=0)

    # Also get aleatoric variance
    _, aleatoric_var = model(graphs, training=False)
    total_var = pred_var + aleatoric_var.squeeze(-1)

    return pred_mean, total_var


def train_ensemble_model(
    train_graphs: jraph.GraphsTuple,
    train_labels: jnp.ndarray,
    train_mask: jnp.ndarray,
    n_features: int,
    n_epochs: int,
    seed: int,
) -> DeepEnsemble:
    """Train a Deep Ensemble."""
    base_config = GCNConfig(
        node_features=n_features,
        hidden_features=[64, 64],
        out_features=1,
        dropout_rate=0.1,
    )
    config = EnsembleConfig(base_config=base_config, n_members=5)
    ensemble = DeepEnsemble(config, rngs=nnx.Rngs(seed))
    optimizers = create_ensemble_optimizers(ensemble, learning_rate=1e-3)

    for epoch in range(n_epochs):
        loss = train_ensemble_step(
            ensemble, optimizers, train_graphs, train_labels, train_mask
        )
        if (epoch + 1) % 25 == 0:
            print(f"    Ensemble - Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.4f}")

    return ensemble


def get_ensemble_predictions(
    ensemble: DeepEnsemble,
    graphs: jraph.GraphsTuple,
) -> tuple:
    """Get predictions from Deep Ensemble."""
    mean, total_var, _ = ensemble(graphs, training=False)
    return mean.squeeze(-1), total_var.squeeze(-1)


def train_evidential_model(
    train_graphs: jraph.GraphsTuple,
    train_labels: jnp.ndarray,
    train_mask: jnp.ndarray,
    n_features: int,
    n_epochs: int,
    seed: int,
) -> EvidentialGCN:
    """Train an Evidential GCN."""
    base_config = GCNConfig(
        node_features=n_features,
        hidden_features=[64, 64],
        out_features=1,
        dropout_rate=0.1,
    )
    config = EvidentialConfig(base_config=base_config, lambda_reg=0.2)
    model = EvidentialGCN(config, rngs=nnx.Rngs(seed))
    optimizer = create_evidential_optimizer(model, learning_rate=1e-3)

    for epoch in range(n_epochs):
        loss = train_evidential_step(
            model, optimizer, train_graphs, train_labels, train_mask
        )
        if (epoch + 1) % 25 == 0:
            print(f"    Evidential - Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.4f}")

    return model


def get_evidential_predictions(
    model: EvidentialGCN,
    graphs: jraph.GraphsTuple,
) -> tuple:
    """Get predictions from Evidential model."""
    mean, total_var, _ = model(graphs, training=False)
    return mean.squeeze(-1), total_var.squeeze(-1)


def main():
    # Load dataset
    print(f"\nLoading dataset from {DATASET_PATH}")
    if not DATASET_PATH.exists():
        print("Dataset not found. Run: python scripts/download_esol.py")
        return

    dataset = MolecularDataset(DATASET_PATH)
    train_data, test_data = dataset.split(test_size=0.2, seed=SEED)
    train_data, val_data = train_data.split(test_size=0.2, seed=SEED)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Batch data
    print("Batching data...")
    train_graphs = jraph.batch(train_data.graphs)
    train_labels = train_data.labels
    train_mask = jnp.ones(len(train_data), dtype=bool)

    val_graphs = jraph.batch(val_data.graphs)
    val_labels = val_data.labels

    test_graphs = jraph.batch(test_data.graphs)
    test_labels = test_data.labels

    n_features = train_data.n_node_features

    # Train models
    print("\n" + "=" * 50)
    print("Training Models")
    print("=" * 50)

    print("\n[1/3] Training MC Dropout model...")
    start = time.time()
    mc_model = train_mc_dropout_model(
        train_graphs, train_labels, train_mask, n_features, N_EPOCHS, SEED
    )
    print(f"    Completed in {time.time() - start:.1f}s")

    print("\n[2/3] Training Deep Ensemble...")
    start = time.time()
    ensemble = train_ensemble_model(
        train_graphs, train_labels, train_mask, n_features, N_EPOCHS, SEED
    )
    print(f"    Completed in {time.time() - start:.1f}s")

    print("\n[3/3] Training Evidential model...")
    start = time.time()
    evidential = train_evidential_model(
        train_graphs, train_labels, train_mask, n_features, N_EPOCHS, SEED
    )
    print(f"    Completed in {time.time() - start:.1f}s")

    # Get test predictions
    print("\n" + "=" * 50)
    print("Generating Predictions")
    print("=" * 50)

    mc_mean, mc_var = get_mc_dropout_predictions(mc_model, test_graphs, n_samples=20)
    ens_mean, ens_var = get_ensemble_predictions(ensemble, test_graphs)
    evd_mean, evd_var = get_evidential_predictions(evidential, test_graphs)

    # Evaluate calibration
    print("\n" + "=" * 50)
    print("Calibration Metrics (Test Set)")
    print("=" * 50)

    models = {
        "MC Dropout": (mc_mean, mc_var),
        "Ensemble": (ens_mean, ens_var),
        "Evidential": (evd_mean, evd_var),
    }

    header = f"{'Model':<15} {'RMSE':>8} {'NLL':>10} {'ECE':>8} "
    header += f"{'Sharpness':>10} {'Mean |z|':>10}"
    print(f"\n{header}")
    print("-" * 65)

    results_for_plot = {}
    for name, (mean, var) in models.items():
        metrics = evaluate_calibration(mean, var, test_labels)
        results_for_plot[name] = (mean, var, test_labels)

        print(
            f"{name:<15} "
            f"{metrics['rmse']:>8.4f} "
            f"{metrics['nll']:>10.4f} "
            f"{metrics['ece']:>8.4f} "
            f"{metrics['sharpness']:>10.4f} "
            f"{metrics['mean_z_score']:>10.4f}"
        )

    # Temperature scaling
    print("\n" + "=" * 50)
    print("Temperature Scaling (Post-hoc Calibration)")
    print("=" * 50)

    # Get validation predictions for fitting temperature
    mc_val_mean, mc_val_var = get_mc_dropout_predictions(
        mc_model, val_graphs, n_samples=20
    )
    ens_val_mean, ens_val_var = get_ensemble_predictions(ensemble, val_graphs)
    evd_val_mean, evd_val_var = get_evidential_predictions(evidential, val_graphs)

    val_models = {
        "MC Dropout": (mc_val_mean, mc_val_var, mc_mean, mc_var),
        "Ensemble": (ens_val_mean, ens_val_var, ens_mean, ens_var),
        "Evidential": (evd_val_mean, evd_val_var, evd_mean, evd_var),
    }

    print(f"\n{'Model':<15} {'Temperature':>12} {'ECE Before':>12} {'ECE After':>12}")
    print("-" * 55)

    calibrated_results = {}
    for name, (val_mean, val_var, test_mean, test_var) in val_models.items():
        # Fit temperature scaler
        scaler = TemperatureScaling()
        scaler.fit(val_mean, val_var, val_labels, max_iter=200, lr=0.05)

        # Evaluate before and after
        ece_before = float(
            evaluate_calibration(test_mean, test_var, test_labels)["ece"]
        )

        calibrated_var = scaler.transform(test_var)
        ece_after = float(
            evaluate_calibration(test_mean, calibrated_var, test_labels)["ece"]
        )

        calibrated_results[f"{name} (calibrated)"] = (
            test_mean,
            calibrated_var,
            test_labels,
        )

        print(
            f"{name:<15} "
            f"{scaler.temperature:>12.4f} "
            f"{ece_before:>12.4f} "
            f"{ece_after:>12.4f}"
        )

    # Create visualizations
    print("\n" + "=" * 50)
    print("Creating Visualizations")
    print("=" * 50)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Calibration comparison before temperature scaling
    fig = plot_calibration_comparison(results_for_plot, figsize=(14, 5))
    fig.suptitle(
        "Calibration Comparison (Before Temperature Scaling)", fontsize=14, y=1.02
    )
    output_path = OUTPUT_DIR / "calibration_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)

    # 2. Calibration comparison after temperature scaling
    fig = plot_calibration_comparison(calibrated_results, figsize=(14, 5))
    fig.suptitle(
        "Calibration Comparison (After Temperature Scaling)", fontsize=14, y=1.02
    )
    output_path = OUTPUT_DIR / "calibration_comparison_calibrated.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)

    # 3. Detailed report for the best model (Ensemble)
    fig = create_calibration_report(
        ens_mean, ens_var, test_labels, model_name="Deep Ensemble"
    )
    output_path = OUTPUT_DIR / "ensemble_calibration_report.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print("""
Key Findings:
- ECE (Expected Calibration Error): Lower is better. Perfect = 0.
- Mean |z|: Should be ~0.8 for perfectly calibrated Gaussian uncertainty.
- Temperature > 1: Model was overconfident (uncertainties too small).
- Temperature < 1: Model was underconfident (uncertainties too large).

For active learning:
- Relative uncertainty ranking matters more than absolute calibration.
- Still, calibrated uncertainties improve sample selection quality.
- Temperature scaling is a simple post-hoc fix with no retraining.
""")

    print("=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
