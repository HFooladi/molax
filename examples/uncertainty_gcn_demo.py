"""
Example demonstrating how to use UncertaintyGCN for molecular property prediction.

This script shows how to:
1. Create a simple molecule graph
2. Initialize and use an UncertaintyGCN model
3. Interpret uncertainty in predictions
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from molax.models.gcn import UncertaintyGCN, UncertaintyGCNConfig

# Set random seed for reproducibility
key = jax.random.PRNGKey(42)
rngs = nnx.Rngs(0, params=1, dropout=2)

# Create a simple molecule graph (6 atoms with 4 features per atom)
num_atoms = 6
in_features = 4

# 1. Create random atom features
key, subkey = jax.random.split(key)
atom_features = jax.random.normal(subkey, (num_atoms, in_features))

# 2. Create a ring-shaped molecular graph
adjacency_matrix = np.zeros((num_atoms, num_atoms))
for i in range(num_atoms):
    # Connect each atom to its neighbors (creating a ring)
    adjacency_matrix[i, (i + 1) % num_atoms] = 1.0
    adjacency_matrix[i, (i - 1) % num_atoms] = 1.0
adjacency_matrix = jnp.array(adjacency_matrix)

print("atom_features.shape", atom_features.shape)
print("adjacency_matrix.shape", adjacency_matrix.shape)

# 3. Create and initialize the UncertaintyGCN model
config = UncertaintyGCNConfig(
    in_features=in_features,  # Number of input features per atom
    hidden_features=[32, 16, 8],  # GCN layer sizes
    out_features=1,  # Single property prediction
    dropout_rate=0.1,  # Dropout for regularization
    n_heads=2,
    rngs=rngs,
)

model = UncertaintyGCN(config)


# 4. Make a prediction with uncertainty
mean, variance = model(atom_features, adjacency_matrix)

print(f"Predicted property value: {mean[0]:.4f}")
print(f"Prediction uncertainty (variance): {variance[0]:.4f}")
print(
    f"95% confidence interval: ({mean[0] - 1.96 * jnp.sqrt(variance[0]):.4f}, "
    f"{mean[0] + 1.96 * jnp.sqrt(variance[0]):.4f})"
)

# 6. Demonstrate uncertainty behavior with modified inputs
test_points = 50
scaling_factors = jnp.linspace(0.1, 10.0, test_points)
means = []
uncertainties = []

for scale in scaling_factors:
    # Scale the input features to create increasingly out-of-distribution examples
    scaled_features = atom_features * scale
    mean, var = model(scaled_features, adjacency_matrix)
    means.append(mean[0])
    uncertainties.append(
        jnp.sqrt(var[0])
    )  # Use standard deviation for easier interpretation

# Convert to arrays
means = jnp.array(means)
uncertainties = jnp.array(uncertainties)

# 7. Visualize how uncertainty changes with input distribution shift
plt.figure(figsize=(10, 6))
plt.plot(scaling_factors, means, "b-", label="Prediction")
plt.fill_between(
    scaling_factors,
    means - 1.96 * uncertainties,
    means + 1.96 * uncertainties,
    alpha=0.3,
    color="b",
    label="95% Confidence Interval",
)
plt.xlabel("Input Scaling Factor")
plt.ylabel("Predicted Property")
plt.title("Prediction with Uncertainty for Different Input Scales")
plt.legend()
plt.grid(True)
plt.savefig("examples/uncertainty_demo.png")
plt.close()

print("\nGenerating out-of-distribution examples:")
for scale in [0.1, 1.0, 10.0]:
    scaled_features = atom_features * scale
    mean, var = model(scaled_features, adjacency_matrix)
    std = jnp.sqrt(var[0])
    print(
        f"Scale {scale:.1f}: {mean[0]:.4f} ± {std:.4f}  "
        f"(95% CI: {mean[0] - 1.96 * std:.4f} to {mean[0] + 1.96 * std:.4f})"
    )

print("\nDemo completed. Saved visualization to 'uncertainty_demo.png'")
print(
    "This demonstrates how uncertainty increases as inputs become more "
    "out-of-distribution."
)
