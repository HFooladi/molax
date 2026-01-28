"""MPNN Demo: Edge-aware molecular property prediction.

This script demonstrates how to use the Message Passing Neural Network (MPNN)
for molecular property prediction on the ESOL dataset. Unlike GCN, MPNN
leverages edge features (bond information) in the message passing computation.
"""

from pathlib import Path

import flax.nnx as nnx
import jax.numpy as jnp
import jraph

from molax.models.mpnn import (
    MPNNConfig,
    UncertaintyMPNN,
    create_mpnn_optimizer,
    eval_mpnn_step,
    train_mpnn_step,
)
from molax.utils.data import MolecularDataset

# Configuration
DATASET_PATH = Path(__file__).parent.parent / "datasets" / "esol.csv"
N_EPOCHS = 100
LEARNING_RATE = 5e-4  # Lower learning rate for stability

print("=" * 60)
print("MPNN Demo: Edge-Aware Molecular Property Prediction")
print("=" * 60)

# Load dataset
print("\nLoading ESOL dataset...")
dataset = MolecularDataset(DATASET_PATH)
train_data, test_data = dataset.split(test_size=0.2, seed=42)
print(f"Train: {len(train_data)} molecules, Test: {len(test_data)} molecules")

# Show edge feature info
sample_graph = train_data.graphs[0]
print("\nGraph features:")
print(f"  Node features: {sample_graph.nodes.shape[1]} (atom properties)")
print(f"  Edge features: {sample_graph.edges.shape[1]} (bond type)")

# Batch all data
print("\nBatching data...")
all_train_graphs = jraph.batch(train_data.graphs)
all_train_labels = train_data.labels
all_test_graphs = jraph.batch(test_data.graphs)
all_test_labels = test_data.labels

n_train = len(train_data)
n_test = len(test_data)
train_mask = jnp.ones(n_train, dtype=bool)
test_mask = jnp.ones(n_test, dtype=bool)

# Create MPNN model
print("\nCreating MPNN model...")
config = MPNNConfig(
    node_features=train_data.n_node_features,
    edge_features=1,  # Bond type feature
    hidden_features=[64, 64],
    out_features=1,
    aggregation="sum",
    dropout_rate=0.1,
)
model = UncertaintyMPNN(config, rngs=nnx.Rngs(0))
optimizer = create_mpnn_optimizer(model, learning_rate=LEARNING_RATE)

print(f"  Hidden layers: {config.hidden_features}")
print(f"  Aggregation: {config.aggregation}")
print(f"  Dropout rate: {config.dropout_rate}")

# Training loop
print("\nTraining MPNN...")
print("-" * 40)

for epoch in range(N_EPOCHS):
    # Training step
    train_loss = train_mpnn_step(
        model, optimizer, all_train_graphs, all_train_labels, train_mask
    )

    # Evaluation every 20 epochs
    if (epoch + 1) % 20 == 0:
        test_mse, _ = eval_mpnn_step(model, all_test_graphs, all_test_labels, test_mask)
        test_rmse = jnp.sqrt(test_mse)
        print(
            f"Epoch {epoch + 1:3d}: Train Loss = {float(train_loss):.4f}, "
            f"Test RMSE = {float(test_rmse):.4f}"
        )

# Final evaluation
print("-" * 40)
test_mse, predictions = eval_mpnn_step(
    model, all_test_graphs, all_test_labels, test_mask
)
test_rmse = jnp.sqrt(test_mse)

# Get predictions with uncertainty
mean, variance = model(all_test_graphs, training=False)
mean = mean.squeeze(-1)
variance = variance.squeeze(-1)

print("\nFinal Results:")
print(f"  Test RMSE: {float(test_rmse):.4f}")
print(f"  Mean predicted variance: {float(jnp.mean(variance[:n_test])):.4f}")

# Show some predictions
print("\nSample predictions (first 5 test molecules):")
print(f"{'Actual':>10} {'Predicted':>10} {'Std Dev':>10}")
for i in range(min(5, n_test)):
    actual = float(all_test_labels[i])
    pred = float(mean[i])
    std = float(jnp.sqrt(variance[i]))
    print(f"{actual:>10.3f} {pred:>10.3f} {std:>10.3f}")

print("\n" + "=" * 60)
print("MPNN demo completed successfully!")
print("=" * 60)
