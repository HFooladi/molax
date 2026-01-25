"""Active learning with Deep Ensembles for improved uncertainty quantification.

This example demonstrates:
1. Training a Deep Ensemble of GCN models
2. Using ensemble disagreement (epistemic uncertainty) for acquisition
3. Comparing with single model MC Dropout approach

Key insight: Ensembles provide better uncertainty estimates because each member
is trained independently, capturing different aspects of the solution space.
"""

import time
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from molax.models.ensemble import (
    DeepEnsemble,
    EnsembleConfig,
    _train_member_step,
    create_ensemble_optimizers,
    get_ensemble_uncertainties,
    train_ensemble_step,
)
from molax.models.gcn import GCNConfig
from molax.utils.data import MolecularDataset

# Configuration
DATASET_PATH = Path(__file__).parent.parent / "datasets" / "esol.csv"
N_EPOCHS = 50
LEARNING_RATE = 1e-3
INITIAL_POOL_FRACTION = 0.05
SAMPLES_PER_ITERATION = 50
N_ITERATIONS = 10
N_ENSEMBLE_MEMBERS = 5

print("=" * 60)
print("Active Learning with Deep Ensembles")
print("=" * 60)
print(f"JAX backend: {jax.default_backend()}")
print(f"Ensemble members: {N_ENSEMBLE_MEMBERS}")

# Load dataset
dataset = MolecularDataset(DATASET_PATH)
train_data, test_data = dataset.split(test_size=0.2, seed=42)
print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# Batch ALL training and test data once
print("Batching all data (one-time cost)...")
all_train_graphs = jraph.batch(train_data.graphs)
all_train_labels = train_data.labels
all_test_graphs = jraph.batch(test_data.graphs)
all_test_labels = test_data.labels

n_train = len(train_data)
n_test = len(test_data)

# Create masks for train/test (jraph.batch adds padding graph)
train_mask = jnp.ones(n_train, dtype=bool)
test_mask = jnp.ones(n_test, dtype=bool)

print(f"Train batch: {all_train_graphs.nodes.shape[0]} nodes")
print(f"Test batch: {all_test_graphs.nodes.shape[0]} nodes")

# Create ensemble model
base_config = GCNConfig(
    node_features=train_data.n_node_features,
    hidden_features=[64, 64],
    out_features=1,
    dropout_rate=0.1,
)
ensemble_config = EnsembleConfig(
    base_config=base_config,
    n_members=N_ENSEMBLE_MEMBERS,
)
ensemble = DeepEnsemble(ensemble_config, nnx.Rngs(0))
optimizers = create_ensemble_optimizers(ensemble, LEARNING_RATE)

print(f"Created ensemble with {len(ensemble.members)} members")


# Evaluation function
@nnx.jit
def eval_ensemble(ensemble, graph, labels):
    """Evaluate ensemble on test data."""
    mean, total_var, epistemic_var = ensemble(graph, training=False)
    mean = mean.squeeze(-1)
    total_var = total_var.squeeze(-1)
    epistemic_var = epistemic_var.squeeze(-1)

    # Only evaluate on actual graphs (exclude padding)
    n_graphs = len(labels)
    mean = mean[:n_graphs]
    total_var = total_var[:n_graphs]
    epistemic_var = epistemic_var[:n_graphs]

    rmse = jnp.sqrt(jnp.mean((mean - labels) ** 2))
    mean_epistemic = jnp.mean(epistemic_var)
    mean_total = jnp.mean(total_var)

    return rmse, mean_epistemic, mean_total


# Initialize labeled pool
rng = np.random.default_rng(42)
n_initial = max(10, int(INITIAL_POOL_FRACTION * n_train))
all_indices = rng.permutation(n_train)

# Create mask: True for labeled samples
labeled_mask = jnp.zeros(n_train, dtype=bool)
labeled_mask = labeled_mask.at[all_indices[:n_initial]].set(True)
pool_mask = ~labeled_mask

print(f"\nInitial: {int(labeled_mask.sum())} labeled, {int(pool_mask.sum())} pool")

# Warmup JIT compilation
print("\nJIT warmup...")
start = time.time()
# Train each member once
for member, optimizer in zip(ensemble.members, optimizers):
    _ = _train_member_step(
        member, optimizer, all_train_graphs, all_train_labels, labeled_mask
    )
_ = eval_ensemble(ensemble, all_test_graphs, all_test_labels)
_ = get_ensemble_uncertainties(ensemble, all_train_graphs)
print(f"JIT compilation: {time.time() - start:.2f}s")

# Active learning loop
print("\n" + "=" * 60)
print("Starting Active Learning Loop")
print("=" * 60)
total_start = time.time()

results = []

for iteration in range(N_ITERATIONS):
    iter_start = time.time()

    # Train all ensemble members
    for epoch in range(N_EPOCHS):
        loss = train_ensemble_step(
            ensemble, optimizers, all_train_graphs, all_train_labels, labeled_mask
        )

    # Evaluate
    rmse, mean_epistemic, mean_total = eval_ensemble(
        ensemble, all_test_graphs, all_test_labels
    )

    train_time = time.time() - iter_start
    n_labeled = int(labeled_mask.sum())

    results.append(
        {
            "iteration": iteration + 1,
            "n_labeled": n_labeled,
            "rmse": float(rmse),
            "epistemic_unc": float(mean_epistemic),
            "total_unc": float(mean_total),
        }
    )

    print(
        f"Iter {iteration + 1}/{N_ITERATIONS}: {train_time:.1f}s, "
        f"RMSE={float(rmse):.4f}, "
        f"Epistemic={float(mean_epistemic):.4f}, "
        f"Total={float(mean_total):.4f}, "
        f"labeled={n_labeled}"
    )

    # Select new samples using epistemic uncertainty (skip on last iteration)
    if iteration < N_ITERATIONS - 1:
        epistemic_unc, total_unc = get_ensemble_uncertainties(
            ensemble, all_train_graphs
        )

        # Only consider pool samples (mask out already labeled)
        epistemic_unc = jnp.where(pool_mask, epistemic_unc[:n_train], -jnp.inf)

        # Select top uncertain samples
        n_select = min(SAMPLES_PER_ITERATION, int(pool_mask.sum()))
        top_indices = jnp.argsort(-epistemic_unc)[:n_select]

        # Update masks
        labeled_mask = labeled_mask.at[top_indices].set(True)
        pool_mask = ~labeled_mask

total_time = time.time() - total_start

print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"Total time: {total_time:.1f}s ({total_time / N_ITERATIONS:.1f}s per iteration)")
print(f"Final: {int(labeled_mask.sum())} labeled, RMSE={float(rmse):.4f}")

# Analyze results
print("\n" + "-" * 60)
print("Active Learning Analysis")
print("-" * 60)

# Check if RMSE improves
initial_rmse = results[0]["rmse"]
final_rmse = results[-1]["rmse"]
improvement = (initial_rmse - final_rmse) / initial_rmse * 100

print(f"RMSE: {initial_rmse:.4f} -> {final_rmse:.4f} ({improvement:.1f}% improvement)")

# Check if epistemic uncertainty decreases (should decrease as we label more)
initial_epistemic = results[0]["epistemic_unc"]
final_epistemic = results[-1]["epistemic_unc"]
epistemic_decrease = (initial_epistemic - final_epistemic) / initial_epistemic * 100

print(
    f"Epistemic uncertainty: {initial_epistemic:.4f} -> {final_epistemic:.4f} "
    f"({epistemic_decrease:.1f}% decrease)"
)

# Sanity checks
print("\n" + "-" * 60)
print("Sanity Checks")
print("-" * 60)

# 1. RMSE should generally decrease
rmses = [r["rmse"] for r in results]
rmse_decreasing = rmses[-1] < rmses[0]
status = "PASS" if rmse_decreasing else "WARN"
print(f"[{status}] RMSE decreases: {rmses[0]:.4f} -> {rmses[-1]:.4f}")

# 2. Epistemic uncertainty should decrease (models become more confident)
epistemic_uncs = [r["epistemic_unc"] for r in results]
epistemic_decreasing = epistemic_uncs[-1] < epistemic_uncs[0]
status = "PASS" if epistemic_decreasing else "WARN"
print(f"[{status}] Epistemic unc: {epistemic_uncs[0]:.4f} -> {epistemic_uncs[-1]:.4f}")

# 3. Total uncertainty should be > epistemic (includes aleatoric)
total_uncs = [r["total_unc"] for r in results]
total_gt_epistemic = all(t >= e for t, e in zip(total_uncs, epistemic_uncs))
status = "PASS" if total_gt_epistemic else "FAIL"
print(f"[{status}] Total uncertainty >= Epistemic uncertainty")

# 4. Learning is happening (loss should decrease, RMSE should improve from baseline)
baseline_rmse = 2.5  # Approximate baseline without training
learned = final_rmse < baseline_rmse
status = "PASS" if learned else "FAIL"
print(f"[{status}] Model learns (RMSE {final_rmse:.4f} < baseline ~{baseline_rmse})")

print("\n" + "=" * 60)
if rmse_decreasing and epistemic_decreasing and total_gt_epistemic and learned:
    print("SUCCESS: Deep Ensemble active learning is working correctly!")
else:
    print("WARNING: Some checks did not pass. Review the results above.")
print("=" * 60)
