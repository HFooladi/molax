"""GAT Demo: Attention-based molecular property prediction.

This script demonstrates how to use the Graph Attention Network (GAT)
for molecular property prediction on the ESOL dataset. GAT uses learned
attention weights to dynamically weight neighbor contributions.
"""

from pathlib import Path

import flax.nnx as nnx
import jax.numpy as jnp
import jraph

from molax.models.gat import (
    GATConfig,
    UncertaintyGAT,
    create_gat_optimizer,
    eval_gat_step,
    train_gat_step,
)
from molax.models.gcn import GCNConfig, UncertaintyGCN, create_train_state, train_step
from molax.models.mpnn import (
    MPNNConfig,
    UncertaintyMPNN,
    create_mpnn_optimizer,
    train_mpnn_step,
)
from molax.utils.data import MolecularDataset

# Configuration
DATASET_PATH = Path(__file__).parent.parent / "datasets" / "esol.csv"
N_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0

print("=" * 60)
print("GAT Demo: Attention-Based Molecular Property Prediction")
print("=" * 60)

# Load dataset
print("\nLoading ESOL dataset...")
dataset = MolecularDataset(DATASET_PATH)
train_data, test_data = dataset.split(test_size=0.2, seed=42)
print(f"Train: {len(train_data)} molecules, Test: {len(test_data)} molecules")

# Show feature info
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

# Create GAT model
print("\nCreating GAT model...")
config = GATConfig(
    node_features=train_data.n_node_features,
    edge_features=1,  # Use bond type in attention
    hidden_features=[64, 64],
    out_features=1,
    n_heads=4,
    dropout_rate=0.1,
    attention_dropout_rate=0.1,
    negative_slope=0.2,
)
model = UncertaintyGAT(config, rngs=nnx.Rngs(0))
optimizer = create_gat_optimizer(
    model,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,
)

print(f"  Hidden layers: {config.hidden_features}")
print(f"  Attention heads: {config.n_heads}")
print(f"  Edge features in attention: {config.edge_features > 0}")
print(f"  Dropout rate: {config.dropout_rate}")
print(f"  Attention dropout: {config.attention_dropout_rate}")
print(f"  Weight decay: {WEIGHT_DECAY}")

# Training loop
print("\nTraining GAT...")
print("-" * 40)

for epoch in range(N_EPOCHS):
    # Training step
    train_loss = train_gat_step(
        model, optimizer, all_train_graphs, all_train_labels, train_mask
    )

    # Evaluation every 20 epochs
    if (epoch + 1) % 20 == 0:
        test_mse, _ = eval_gat_step(model, all_test_graphs, all_test_labels, test_mask)
        test_rmse = jnp.sqrt(test_mse)
        print(
            f"Epoch {epoch + 1:3d}: Train Loss = {float(train_loss):.4f}, "
            f"Test RMSE = {float(test_rmse):.4f}"
        )

# Final evaluation
print("-" * 40)
test_mse, predictions = eval_gat_step(
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

# Compare with GCN and MPNN
print("\n" + "=" * 60)
print("Comparing GAT with GCN and MPNN...")
print("=" * 60)

# Train GCN
gcn_config = GCNConfig(
    node_features=train_data.n_node_features,
    hidden_features=[64, 64],
    out_features=1,
    dropout_rate=0.1,
)
gcn = UncertaintyGCN(gcn_config, rngs=nnx.Rngs(0))
gcn_optimizer = create_train_state(gcn, learning_rate=LEARNING_RATE)

print("\nTraining GCN for comparison...")
for epoch in range(N_EPOCHS):
    train_step(gcn, gcn_optimizer, all_train_graphs, all_train_labels, train_mask)

gcn_mean, _ = gcn(all_test_graphs, training=False)
gcn_mse = jnp.mean((gcn_mean.squeeze(-1)[:n_test] - all_test_labels[:n_test]) ** 2)
gcn_rmse = jnp.sqrt(gcn_mse)

# Train MPNN
mpnn_config = MPNNConfig(
    node_features=train_data.n_node_features,
    edge_features=1,
    hidden_features=[64, 64],
    out_features=1,
    aggregation="sum",
    dropout_rate=0.1,
)
mpnn = UncertaintyMPNN(mpnn_config, rngs=nnx.Rngs(0))
mpnn_optimizer = create_mpnn_optimizer(mpnn, learning_rate=LEARNING_RATE)

print("Training MPNN for comparison...")
for epoch in range(N_EPOCHS):
    train_mpnn_step(
        mpnn, mpnn_optimizer, all_train_graphs, all_train_labels, train_mask
    )

mpnn_mean, _ = mpnn(all_test_graphs, training=False)
mpnn_mse = jnp.mean((mpnn_mean.squeeze(-1)[:n_test] - all_test_labels[:n_test]) ** 2)
mpnn_rmse = jnp.sqrt(mpnn_mse)

print("\n" + "-" * 40)
print("Model Comparison (Test RMSE):")
print("-" * 40)
print(f"  GCN:  {float(gcn_rmse):.4f}")
print(f"  MPNN: {float(mpnn_rmse):.4f}")
print(f"  GAT:  {float(test_rmse):.4f}")

print("\n" + "=" * 60)
print("GAT demo completed successfully!")
print("=" * 60)
