"""Graph Transformer Demo: State-of-the-art molecular property prediction.

This script demonstrates how to use the Graph Transformer for molecular
property prediction on the ESOL dataset. Graph Transformer uses full
self-attention with positional encodings to capture long-range dependencies.
"""

from pathlib import Path

import flax.nnx as nnx
import jax.numpy as jnp
import jraph

from molax.models.gat import (
    GATConfig,
    UncertaintyGAT,
    create_gat_optimizer,
    train_gat_step,
)
from molax.models.gcn import GCNConfig, UncertaintyGCN, create_train_state, train_step
from molax.models.graph_transformer import (
    GraphTransformerConfig,
    UncertaintyGraphTransformer,
    create_graph_transformer_optimizer,
    eval_graph_transformer_step,
    train_graph_transformer_step,
)
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
LEARNING_RATE = 1e-4  # Lower LR for transformers
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 50
MAX_GRAD_NORM = 1.0

print("=" * 60)
print("Graph Transformer Demo: Molecular Property Prediction")
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

# Create Graph Transformer model
print("\nCreating Graph Transformer model...")
config = GraphTransformerConfig(
    node_features=train_data.n_node_features,
    edge_features=1,  # Use bond type as attention bias
    hidden_features=[64, 64],
    out_features=1,
    n_heads=4,
    ffn_ratio=4.0,
    dropout_rate=0.1,
    attention_dropout_rate=0.1,
    pe_type="rwpe",  # Random Walk Positional Encoding
    pe_dim=16,
)
model = UncertaintyGraphTransformer(config, rngs=nnx.Rngs(0))
optimizer = create_graph_transformer_optimizer(
    model,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    max_grad_norm=MAX_GRAD_NORM,
)

print(f"  Hidden layers: {config.hidden_features}")
print(f"  Attention heads: {config.n_heads}")
print(f"  FFN ratio: {config.ffn_ratio}")
print(f"  Positional encoding: {config.pe_type} (dim={config.pe_dim})")
print(f"  Dropout rate: {config.dropout_rate}")
print(f"  Attention dropout: {config.attention_dropout_rate}")
print(f"  Learning rate: {LEARNING_RATE} (with {WARMUP_STEPS} warmup steps)")
print(f"  Weight decay: {WEIGHT_DECAY}")

# Training loop
print("\nTraining Graph Transformer...")
print("-" * 40)

for epoch in range(N_EPOCHS):
    # Training step
    train_loss = train_graph_transformer_step(
        model, optimizer, all_train_graphs, all_train_labels, train_mask
    )

    # Evaluation every 20 epochs
    if (epoch + 1) % 20 == 0:
        test_mse, _ = eval_graph_transformer_step(
            model, all_test_graphs, all_test_labels, test_mask
        )
        test_rmse = jnp.sqrt(test_mse)
        print(
            f"Epoch {epoch + 1:3d}: Train Loss = {float(train_loss):.4f}, "
            f"Test RMSE = {float(test_rmse):.4f}"
        )

# Final evaluation
print("-" * 40)
test_mse, predictions = eval_graph_transformer_step(
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

# Compare different positional encodings
print("\n" + "=" * 60)
print("Comparing Positional Encoding Types...")
print("=" * 60)

# Train without PE
print("\nTraining Graph Transformer without PE...")
config_no_pe = GraphTransformerConfig(
    node_features=train_data.n_node_features,
    edge_features=1,
    hidden_features=[64, 64],
    out_features=1,
    n_heads=4,
    dropout_rate=0.1,
    pe_type="none",
)
model_no_pe = UncertaintyGraphTransformer(config_no_pe, rngs=nnx.Rngs(0))
optimizer_no_pe = create_graph_transformer_optimizer(
    model_no_pe, learning_rate=LEARNING_RATE, warmup_steps=WARMUP_STEPS
)

for epoch in range(N_EPOCHS):
    train_graph_transformer_step(
        model_no_pe, optimizer_no_pe, all_train_graphs, all_train_labels, train_mask
    )

no_pe_mean, _ = model_no_pe(all_test_graphs, training=False)
no_pe_mse = jnp.mean((no_pe_mean.squeeze(-1)[:n_test] - all_test_labels[:n_test]) ** 2)
no_pe_rmse = jnp.sqrt(no_pe_mse)

# Compare with GCN, MPNN, and GAT
print("\n" + "=" * 60)
print("Comparing with Other Architectures...")
print("=" * 60)

# Train GCN
print("\nTraining GCN for comparison...")
gcn_config = GCNConfig(
    node_features=train_data.n_node_features,
    hidden_features=[64, 64],
    out_features=1,
    dropout_rate=0.1,
)
gcn = UncertaintyGCN(gcn_config, rngs=nnx.Rngs(0))
gcn_optimizer = create_train_state(gcn, learning_rate=1e-3)

for epoch in range(N_EPOCHS):
    train_step(gcn, gcn_optimizer, all_train_graphs, all_train_labels, train_mask)

gcn_mean, _ = gcn(all_test_graphs, training=False)
gcn_mse = jnp.mean((gcn_mean.squeeze(-1)[:n_test] - all_test_labels[:n_test]) ** 2)
gcn_rmse = jnp.sqrt(gcn_mse)

# Train MPNN
print("Training MPNN for comparison...")
mpnn_config = MPNNConfig(
    node_features=train_data.n_node_features,
    edge_features=1,
    hidden_features=[64, 64],
    out_features=1,
    aggregation="sum",
    dropout_rate=0.1,
)
mpnn = UncertaintyMPNN(mpnn_config, rngs=nnx.Rngs(0))
mpnn_optimizer = create_mpnn_optimizer(mpnn, learning_rate=1e-3)

for epoch in range(N_EPOCHS):
    train_mpnn_step(
        mpnn, mpnn_optimizer, all_train_graphs, all_train_labels, train_mask
    )

mpnn_mean, _ = mpnn(all_test_graphs, training=False)
mpnn_mse = jnp.mean((mpnn_mean.squeeze(-1)[:n_test] - all_test_labels[:n_test]) ** 2)
mpnn_rmse = jnp.sqrt(mpnn_mse)

# Train GAT
print("Training GAT for comparison...")
gat_config = GATConfig(
    node_features=train_data.n_node_features,
    edge_features=1,
    hidden_features=[64, 64],
    out_features=1,
    n_heads=4,
    dropout_rate=0.1,
    attention_dropout_rate=0.1,
)
gat = UncertaintyGAT(gat_config, rngs=nnx.Rngs(0))
gat_optimizer = create_gat_optimizer(gat, learning_rate=1e-3)

for epoch in range(N_EPOCHS):
    train_gat_step(gat, gat_optimizer, all_train_graphs, all_train_labels, train_mask)

gat_mean, _ = gat(all_test_graphs, training=False)
gat_mse = jnp.mean((gat_mean.squeeze(-1)[:n_test] - all_test_labels[:n_test]) ** 2)
gat_rmse = jnp.sqrt(gat_mse)

print("\n" + "-" * 40)
print("Model Comparison (Test RMSE):")
print("-" * 40)
print(f"  GCN:                       {float(gcn_rmse):.4f}")
print(f"  MPNN:                      {float(mpnn_rmse):.4f}")
print(f"  GAT:                       {float(gat_rmse):.4f}")
print(f"  Graph Transformer (no PE): {float(no_pe_rmse):.4f}")
print(f"  Graph Transformer (RWPE):  {float(test_rmse):.4f}")

# Test embeddings for acquisition
print("\n" + "=" * 60)
print("Testing Embedding Extraction (for Core-Set Selection)...")
print("=" * 60)

embeddings = model.extract_embeddings(all_test_graphs)
print(f"  Embedding shape: {embeddings.shape}")
print(f"  First embedding: {embeddings[0][:5]}...")  # Show first 5 dims

print("\n" + "=" * 60)
print("Graph Transformer demo completed successfully!")
print("=" * 60)
