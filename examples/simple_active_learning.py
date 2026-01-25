"""Efficient active learning with jraph - fixed batch approach.

Key insight: Batch ALL training data once, use index masking for active learning.
This avoids JIT recompilation from changing shapes.
"""

import time
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax

from molax.models.gcn import GCNConfig, UncertaintyGCN
from molax.utils.data import MolecularDataset

# Configuration
DATASET_PATH = Path(__file__).parent.parent / "datasets" / "esol.csv"
N_EPOCHS = 50
LEARNING_RATE = 1e-3
INITIAL_POOL_FRACTION = 0.05
SAMPLES_PER_ITERATION = 50
N_ITERATIONS = 10

print("=" * 60)
print("Efficient Active Learning with jraph")
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

# Create model
config = GCNConfig(
    node_features=train_data.n_node_features,
    hidden_features=[64, 64],
    out_features=1,
    dropout_rate=0.1,
)
model = UncertaintyGCN(config, nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)


# JIT-compiled training step with masking
@nnx.jit
def train_step_masked(model, optimizer, graph, labels, mask):
    """Train on subset of graphs indicated by mask."""

    def loss_fn(model):
        mean, var = model(graph, training=True)
        mean = mean.squeeze(-1)
        var = var.squeeze(-1)
        nll = 0.5 * (jnp.log(var + 1e-6) + (labels - mean) ** 2 / (var + 1e-6))
        # Only compute loss for labeled samples
        masked_nll = jnp.where(mask, nll, 0.0)
        return jnp.sum(masked_nll) / (jnp.sum(mask) + 1e-6)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model, graph, labels):
    """Evaluate on all test data."""
    mean, _ = model(graph, training=False)
    mean = mean.squeeze(-1)
    return jnp.sqrt(jnp.mean((mean - labels) ** 2))


@nnx.jit
def get_uncertainties(model, graph):
    """Get uncertainty for all graphs via MC dropout."""
    predictions = []
    for _ in range(10):
        mean, _ = model(graph, training=True)
        predictions.append(mean.squeeze(-1))
    predictions = jnp.stack(predictions)
    return jnp.var(predictions, axis=0)


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

# Warmup JIT
print("\nJIT warmup...")
start = time.time()
_ = train_step_masked(
    model, optimizer, all_train_graphs, all_train_labels, labeled_mask
)
_ = eval_step(model, all_test_graphs, all_test_labels)
_ = get_uncertainties(model, all_train_graphs)
print(f"JIT compilation: {time.time() - start:.2f}s")

# Active learning loop
print("\n" + "=" * 60)
total_start = time.time()

for iteration in range(N_ITERATIONS):
    iter_start = time.time()

    # Train
    for epoch in range(N_EPOCHS):
        loss = train_step_masked(
            model, optimizer, all_train_graphs, all_train_labels, labeled_mask
        )

    # Evaluate
    rmse = eval_step(model, all_test_graphs, all_test_labels)

    train_time = time.time() - iter_start
    n_labeled = int(labeled_mask.sum())
    print(
        f"Iter {iteration + 1}/{N_ITERATIONS}: {train_time:.1f}s, "
        f"RMSE={float(rmse):.4f}, labeled={n_labeled}"
    )

    # Select new samples (skip on last iteration)
    if iteration < N_ITERATIONS - 1:
        uncertainties = get_uncertainties(model, all_train_graphs)
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
