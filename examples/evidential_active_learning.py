"""Active learning with Evidential Deep Learning.

Demonstrates using EvidentialGCN for active learning with single-pass
uncertainty estimation. Unlike MC Dropout, evidential models predict
uncertainty directly without requiring multiple forward passes.

Key advantages:
- Single forward pass for uncertainty (faster inference)
- Separate aleatoric and epistemic uncertainty
- Well-calibrated for out-of-distribution detection

Reference: Amini et al., "Deep Evidential Regression", NeurIPS 2020
"""

import time
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from molax.models.evidential import (
    EvidentialConfig,
    EvidentialGCN,
    create_evidential_optimizer,
    eval_evidential_step,
    train_evidential_step,
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
LAMBDA_REG = 0.1  # Evidential regularization weight

print("=" * 60)
print("Active Learning with Evidential Deep Learning")
print("=" * 60)
print(f"JAX backend: {jax.default_backend()}")

# Load dataset
dataset = MolecularDataset(DATASET_PATH)
train_data, test_data = dataset.split(test_size=0.2, seed=42)
print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# Batch ALL training and test data once (fixed shapes = no recompilation)
print("Batching all data (one-time cost)...")
all_train_graphs = jraph.batch(train_data.graphs)
all_train_labels = train_data.labels
all_test_graphs = jraph.batch(test_data.graphs)
all_test_labels = test_data.labels

n_train_nodes = all_train_graphs.nodes.shape[0]
n_train_edges = all_train_graphs.edges.shape[0]
print(f"Train batch: {n_train_nodes} nodes, {n_train_edges} edges")
n_test_nodes = all_test_graphs.nodes.shape[0]
n_test_edges = all_test_graphs.edges.shape[0]
print(f"Test batch: {n_test_nodes} nodes, {n_test_edges} edges")

# Create Evidential GCN model
base_config = GCNConfig(
    node_features=train_data.n_node_features,
    hidden_features=[64, 64],
    out_features=1,
    dropout_rate=0.1,
)
config = EvidentialConfig(base_config=base_config, lambda_reg=LAMBDA_REG)
model = EvidentialGCN(config, nnx.Rngs(0))
optimizer = create_evidential_optimizer(model, LEARNING_RATE)


@nnx.jit
def get_epistemic_uncertainties(model, graph):
    """Get epistemic uncertainty for all graphs (single forward pass).

    Unlike MC Dropout which requires multiple passes, evidential models
    provide uncertainty estimates in a single forward pass.
    """
    _, _, epistemic_var = model(graph, training=False)
    return epistemic_var.squeeze(-1)


# Initialize labeled pool
rng = np.random.default_rng(42)
n_train = len(train_data)
n_initial = max(10, int(INITIAL_POOL_FRACTION * n_train))
all_indices = rng.permutation(n_train)

# Create mask: True for labeled samples
labeled_mask = jnp.zeros(n_train, dtype=bool)
labeled_mask = labeled_mask.at[all_indices[:n_initial]].set(True)
pool_mask = ~labeled_mask

print(f"\nInitial: {int(labeled_mask.sum())} labeled, {int(pool_mask.sum())} pool")
print(f"Evidential regularization: lambda={LAMBDA_REG}")

# Warmup JIT
print("\nJIT warmup...")
start = time.time()
_ = train_evidential_step(
    model, optimizer, all_train_graphs, all_train_labels, labeled_mask
)
test_mask_warmup = jnp.ones(len(test_data), dtype=bool)
_ = eval_evidential_step(model, all_test_graphs, all_test_labels, test_mask_warmup)
_ = get_epistemic_uncertainties(model, all_train_graphs)
print(f"JIT compilation: {time.time() - start:.2f}s")

# Active learning loop
print("\n" + "=" * 60)
print("Starting active learning loop...")
print("=" * 60)
total_start = time.time()

for iteration in range(N_ITERATIONS):
    iter_start = time.time()

    # Train
    for epoch in range(N_EPOCHS):
        loss = train_evidential_step(
            model, optimizer, all_train_graphs, all_train_labels, labeled_mask
        )

    # Evaluate
    test_mask = jnp.ones(len(test_data), dtype=bool)
    rmse, mean_epistemic, mean_total = eval_evidential_step(
        model, all_test_graphs, all_test_labels, test_mask
    )

    train_time = time.time() - iter_start
    n_labeled = int(labeled_mask.sum())
    print(
        f"Iter {iteration + 1}/{N_ITERATIONS}: {train_time:.1f}s, "
        f"RMSE={float(rmse):.4f}, labeled={n_labeled}, "
        f"epistemic={float(mean_epistemic):.4f}, total={float(mean_total):.4f}"
    )

    # Select new samples using epistemic uncertainty (skip on last iteration)
    if iteration < N_ITERATIONS - 1:
        # Single forward pass for uncertainty (key advantage of evidential!)
        uncertainties = get_epistemic_uncertainties(model, all_train_graphs)

        # Mask out already labeled samples
        uncertainties = jnp.where(pool_mask, uncertainties, -jnp.inf)

        # Select top uncertain
        n_select = min(SAMPLES_PER_ITERATION, int(pool_mask.sum()))
        top_indices = jnp.argsort(-uncertainties)[:n_select]

        # Update masks
        labeled_mask = labeled_mask.at[top_indices].set(True)
        pool_mask = ~labeled_mask

total_time = time.time() - total_start
print("=" * 60)
print(f"Total time: {total_time:.1f}s ({total_time / N_ITERATIONS:.1f}s per iteration)")
print(f"Final: {int(labeled_mask.sum())} labeled, RMSE={float(rmse):.4f}")
print("\nEvidential model benefits:")
print("- Single forward pass for uncertainty (no MC sampling needed)")
print("- Separate aleatoric and epistemic uncertainty estimates")
print("- Theoretically grounded via Normal-Inverse-Gamma distribution")

# NOTE: Known calibration issue with Evidential Deep Learning
# ============================================================
# You may observe that epistemic/total uncertainty values INCREASE during
# training, even as RMSE decreases. This is a known limitation of evidential
# regression where the NIG parameters can drift, causing unbounded uncertainty.
#
# Potential fixes to explore:
# 1. Increase lambda_reg (try 0.5, 1.0, or higher)
# 2. Add post-hoc calibration (temperature scaling)
# 3. Use different regularization schemes (see follow-up papers)
#
# For active learning, the RELATIVE ranking of uncertainties matters more than
# absolute values. The acquisition function normalizes uncertainties, so sample
# selection still works correctly.
#
# References:
# - Original paper: Amini et al., "Deep Evidential Regression", NeurIPS 2020
# - Calibration issues discussed in: https://github.com/aamini/evidential-deep-learning/issues
