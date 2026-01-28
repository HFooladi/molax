"""
Example demonstrating how to use UncertaintyGCN for molecular property prediction.

This script shows how to:
1. Load molecules and convert to jraph graphs
2. Initialize and use an UncertaintyGCN model
3. Interpret uncertainty in predictions
"""

import flax.nnx as nnx
import jax.numpy as jnp
import jraph
import matplotlib.pyplot as plt

from molax.models.gcn import GCNConfig, UncertaintyGCN
from molax.utils.data import smiles_to_jraph

# Create some sample molecules with varying complexity
molecules = [
    ("C", "methane"),
    ("CC", "ethane"),
    ("CCC", "propane"),
    ("CCCC", "butane"),
    ("c1ccccc1", "benzene"),
    ("CCO", "ethanol"),
    ("CC(=O)O", "acetic acid"),
    ("c1ccc(O)cc1", "phenol"),
]

print("=" * 60)
print("UncertaintyGCN Demo: Molecular Property Prediction")
print("=" * 60)

# Convert SMILES to jraph graphs
graphs = [smiles_to_jraph(smi) for smi, _ in molecules]
batched_graphs = jraph.batch(graphs)

print(f"\nLoaded {len(molecules)} molecules")
print(f"Node features: {graphs[0].nodes.shape[1]}")

# Create and initialize the UncertaintyGCN model
config = GCNConfig(
    node_features=graphs[0].nodes.shape[1],
    hidden_features=[32, 16],
    out_features=1,
    dropout_rate=0.1,
)
model = UncertaintyGCN(config, rngs=nnx.Rngs(42))

print("\nModel configuration:")
print(f"  Hidden layers: {config.hidden_features}")
print(f"  Dropout rate: {config.dropout_rate}")

# Make predictions with uncertainty
print("\n" + "-" * 60)
print("Predictions with Uncertainty")
print("-" * 60)

mean, variance = model(batched_graphs, training=False)
mean = mean.squeeze(-1)
variance = variance.squeeze(-1)

print(f"{'Molecule':<15} {'Mean':>10} {'Std Dev':>10} {'95% CI'}")
print("-" * 60)

for i, (_, name) in enumerate(molecules):
    m = float(mean[i])
    std = float(jnp.sqrt(variance[i]))
    ci_low = m - 1.96 * std
    ci_high = m + 1.96 * std
    print(f"{name:<15} {m:>10.4f} {std:>10.4f} [{ci_low:.2f}, {ci_high:.2f}]")

# Demonstrate MC Dropout uncertainty
print("\n" + "-" * 60)
print("MC Dropout Uncertainty (10 samples)")
print("-" * 60)

mc_predictions = []
for _ in range(10):
    pred, _ = model(batched_graphs, training=True)  # Dropout active
    mc_predictions.append(pred.squeeze(-1))

mc_predictions = jnp.stack(mc_predictions)
mc_mean = jnp.mean(mc_predictions, axis=0)
mc_std = jnp.std(mc_predictions, axis=0)

print(f"{'Molecule':<15} {'MC Mean':>10} {'MC Std':>10}")
print("-" * 40)

for i, (_, name) in enumerate(molecules):
    print(f"{name:<15} {float(mc_mean[i]):>10.4f} {float(mc_std[i]):>10.4f}")

# Visualize predictions
print("\n" + "-" * 60)
print("Creating visualization...")

fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(molecules))
names = [name for _, name in molecules]
means = [float(mean[i]) for i in range(len(molecules))]
stds = [float(jnp.sqrt(variance[i])) for i in range(len(molecules))]

ax.bar(x, means, yerr=[1.96 * s for s in stds], capsize=5, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_ylabel("Predicted Value")
ax.set_title("UncertaintyGCN Predictions with 95% Confidence Intervals")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("examples/uncertainty_demo.png", dpi=150)
plt.close()

print("Saved visualization to 'examples/uncertainty_demo.png'")

print("\n" + "=" * 60)
print("Demo completed successfully!")
print("=" * 60)
print("\nKey takeaways:")
print("- UncertaintyGCN outputs both mean prediction and variance")
print("- Variance head predicts aleatoric (data) uncertainty")
print("- MC Dropout provides epistemic (model) uncertainty")
print("- 95% CI = mean ± 1.96 * std")
