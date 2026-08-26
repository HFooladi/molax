"""Active Learning Benchmark - Efficient jraph implementation.

Compares acquisition strategies:
- Random sampling (baseline)
- Uncertainty sampling (MC dropout variance)
- Combined uncertainty + Core-Set diversity

Key optimization: Batch all data once, use masking for active learning.
This avoids JIT recompilation and achieves ~400x speedup.

Two settings here are load-bearing, and getting either wrong makes the curve
meaningless:

1.  The model is re-initialized at the start of every acquisition round.
    Warm-starting one model across rounds gives later rounds a larger
    cumulative training budget, so the curve would measure training time as
    much as data efficiency. The cost of doing this correctly is that each
    round must train to convergence from scratch -- hence N_EPOCHS=400, not 50.
    At 50 epochs a fresh model does not even reach the mean-predictor baseline.

2.  The "rich" featurizer is used, not the 6-dim default. With the default
    features this model saturates at a test RMSE of ~2.00 no matter how much
    data it is given (measured at 5%, 25% and 50% of ESOL: 2.02 / 2.01 / 2.03),
    which is *worse* than simply predicting the training mean (2.13). An
    acquisition strategy cannot be evaluated on a model that cannot use data.

Runtime is roughly 20-30 minutes on a GPU.
"""

import time
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jraph
import matplotlib.pyplot as plt
import numpy as np
import optax

from molax.acquisition import coreset_from_embeddings
from molax.models.gcn import GCNConfig, UncertaintyGCN
from molax.utils.data import MolecularDataset

# Configuration
DATASET_PATH = Path(__file__).parent.parent / "datasets" / "esol.csv"
OUTPUT_PATH = Path(__file__).parent / "assets" / "active_learning_benchmark.png"

FEATURES = "rich"  # see note in the module docstring
INITIAL_FRACTION = 0.10  # Start with 10% labeled
BATCH_FRACTION = 0.10  # Add 10% per iteration
MAX_FRACTION = 0.50  # Stop at 50%
N_EPOCHS = 400  # enough for a FRESH model to converge (see docstring)
N_REPEATS = 3
UNCERTAINTY_WEIGHT = 0.7  # weight of uncertainty vs diversity in 'combined'

print("=" * 60)
print("Active Learning Benchmark")
print("=" * 60)
print(f"JAX backend: {jax.default_backend()}")


def create_model_and_optimizer(n_features: int, seed: int):
    """Create fresh model and optimizer."""
    config = GCNConfig(
        node_features=n_features,
        hidden_features=[64, 64],
        out_features=1,
        dropout_rate=0.1,
    )
    model = UncertaintyGCN(config, nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    return model, optimizer


def run_experiment(
    strategy: str,
    train_graphs: jraph.GraphsTuple,
    train_labels: jnp.ndarray,
    test_graphs: jraph.GraphsTuple,
    test_labels: jnp.ndarray,
    n_train: int,
    n_features: int,
    seed: int,
):
    """Run single active learning experiment."""

    # JIT-compiled functions. These close over the fixed batched graphs, so they
    # compile once and are reused across every freshly initialized model.
    @nnx.jit
    def train_step(model, optimizer, mask):
        def loss_fn(model):
            mean, var = model(train_graphs, training=True)
            mean, var = mean.squeeze(-1), var.squeeze(-1)
            nll = 0.5 * (
                jnp.log(var + 1e-6) + (train_labels - mean) ** 2 / (var + 1e-6)
            )
            masked_nll = jnp.where(mask, nll, 0.0)
            return jnp.sum(masked_nll) / (jnp.sum(mask) + 1e-6)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    @nnx.jit
    def evaluate(model):
        mean, _ = model(test_graphs, training=False)
        return jnp.sqrt(jnp.mean((mean.squeeze(-1) - test_labels) ** 2))

    @nnx.jit
    def get_uncertainties(model):
        preds = jnp.stack(
            [model(train_graphs, training=True)[0].squeeze(-1) for _ in range(10)]
        )
        return jnp.var(preds, axis=0)

    @nnx.jit
    def get_embeddings(model):
        return model.extract_embeddings(train_graphs, training=False)

    # Initialize
    rng = np.random.default_rng(seed)
    n_initial = max(10, int(INITIAL_FRACTION * n_train))
    n_per_iter = max(5, int(BATCH_FRACTION * n_train))
    max_labeled = int(MAX_FRACTION * n_train)

    indices = rng.permutation(n_train)
    labeled_mask = jnp.zeros(n_train, dtype=bool).at[indices[:n_initial]].set(True)

    # Warmup JIT once, on a throwaway model
    warmup_model, warmup_opt = create_model_and_optimizer(n_features, seed)
    _ = train_step(warmup_model, warmup_opt, labeled_mask)
    _ = evaluate(warmup_model)
    _ = get_uncertainties(warmup_model)
    _ = get_embeddings(warmup_model)

    results = []

    while int(labeled_mask.sum()) <= max_labeled:
        # Fresh model each round: the learning curve must reflect how much data
        # was acquired, not how many gradient steps have accumulated.
        model, optimizer = create_model_and_optimizer(n_features, seed)
        for _ in range(N_EPOCHS):
            train_step(model, optimizer, labeled_mask)

        # Evaluate
        rmse = float(evaluate(model))
        frac = int(labeled_mask.sum()) / n_train
        results.append((frac, rmse))

        if int(labeled_mask.sum()) >= max_labeled:
            break

        # Select new samples
        pool_mask = ~labeled_mask
        n_select = min(n_per_iter, int(pool_mask.sum()))

        if strategy == "random":
            pool_indices = jnp.where(pool_mask)[0]
            selected = jax.random.permutation(
                jax.random.PRNGKey(seed + len(results)), pool_indices
            )[:n_select]
        elif strategy == "uncertainty":
            uncertainties = get_uncertainties(model)
            uncertainties = jnp.where(pool_mask, uncertainties, -jnp.inf)
            selected = jnp.argsort(-uncertainties)[:n_select]
        else:  # combined: uncertainty + Core-Set diversity
            uncertainties = get_uncertainties(model)

            # Rank-normalize both signals to [0, 1] so they combine on a
            # comparable scale -- raw MC-dropout variance and embedding
            # distance have unrelated units.
            unc_score = uncertainties / (jnp.max(uncertainties) + 1e-12)

            # K-center greedy returns an ordered list; earlier picks cover more
            # of the embedding space, so score them higher.
            embeddings = get_embeddings(model)
            diverse = coreset_from_embeddings(embeddings, pool_mask, n_select)
            div_score = jnp.zeros(n_train)
            if diverse:
                ranks = jnp.linspace(1.0, 0.0, len(diverse))
                div_score = div_score.at[jnp.array(diverse)].set(ranks)

            combined = (
                UNCERTAINTY_WEIGHT * unc_score + (1 - UNCERTAINTY_WEIGHT) * div_score
            )
            combined = jnp.where(pool_mask, combined, -jnp.inf)
            selected = jnp.argsort(-combined)[:n_select]

        labeled_mask = labeled_mask.at[selected].set(True)

    return results


def plot_results(all_results: dict, baseline_rmse: float):
    """Plot benchmark results."""
    plt.figure(figsize=(10, 6))

    colors = {"random": "gray", "uncertainty": "blue", "combined": "green"}
    labels = {
        "random": "Random Sampling",
        "uncertainty": "Uncertainty Sampling",
        "combined": "Uncertainty + Core-Set Diversity",
    }

    for strategy, runs in all_results.items():
        # Aggregate across runs
        all_fracs = sorted(set(f for run in runs for f, _ in run))
        means, stds = [], []

        for frac in all_fracs:
            vals = [rmse for run in runs for f, rmse in run if abs(f - frac) < 0.01]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0)

        fracs = np.array(all_fracs) * 100
        means = np.array(means)
        stds = np.array(stds)

        plt.plot(
            fracs,
            means,
            "-o",
            color=colors[strategy],
            label=labels[strategy],
            linewidth=2,
        )
        plt.fill_between(
            fracs, means - stds, means + stds, color=colors[strategy], alpha=0.2
        )

    plt.axhline(
        baseline_rmse,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Predict training mean",
    )

    plt.xlabel("Training Data Used (%)", fontsize=12)
    plt.ylabel("Test RMSE", fontsize=12)
    plt.title("Active Learning Benchmark: ESOL Dataset", fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nPlot saved to {OUTPUT_PATH}")


def main():
    # Load data
    print(f"\nLoading {DATASET_PATH}")
    dataset = MolecularDataset(DATASET_PATH, features=FEATURES)
    train_data, test_data = dataset.split(test_size=0.2, seed=42)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Batch all data once
    print("Batching data...")
    train_graphs = jraph.batch(train_data.graphs)
    train_labels = train_data.labels
    test_graphs = jraph.batch(test_data.graphs)
    test_labels = test_data.labels

    n_train = len(train_data)
    n_features = train_data.n_node_features

    # The number every curve has to beat. If a strategy sits above this line,
    # the model is not learning and the comparison says nothing about
    # acquisition -- it was this check that exposed the 6-dim featurizer.
    baseline_rmse = float(
        jnp.sqrt(jnp.mean((test_labels - jnp.mean(train_labels)) ** 2))
    )
    print(f"Mean-predictor baseline RMSE: {baseline_rmse:.4f}")

    # Run experiments
    strategies = ["random", "uncertainty", "combined"]
    all_results = {s: [] for s in strategies}

    total_start = time.time()

    for repeat in range(N_REPEATS):
        print(f"\n--- Repeat {repeat + 1}/{N_REPEATS} ---")
        for strategy in strategies:
            start = time.time()
            results = run_experiment(
                strategy,
                train_graphs,
                train_labels,
                test_graphs,
                test_labels,
                n_train,
                n_features,
                seed=42 + repeat,
            )
            elapsed = time.time() - start
            all_results[strategy].append(results)
            final_rmse = results[-1][1]
            print(f"  {strategy:12s}: {elapsed:.1f}s, final RMSE={final_rmse:.4f}")

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.1f}s")

    # Plot
    plot_results(all_results, baseline_rmse)

    # Summary
    print("\n" + "=" * 60)
    print("Summary (final RMSE at 50% data)")
    print("=" * 60)
    print(f"{'baseline':12s}: {baseline_rmse:.4f}  (predict the training mean)")
    for strategy in strategies:
        final_rmses = [run[-1][1] for run in all_results[strategy]]
        mean_rmse = np.mean(final_rmses)
        verdict = (
            "beats baseline" if mean_rmse < baseline_rmse else "WORSE THAN BASELINE"
        )
        print(
            f"{strategy:12s}: {mean_rmse:.4f} +/- {np.std(final_rmses):.4f}  {verdict}"
        )


if __name__ == "__main__":
    main()
