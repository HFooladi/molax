# Core Concepts

This page explains the key concepts and architecture patterns in molax.

## Performance: Batch-Once-Then-Mask

The single most important optimization in molax is the **batch-once-then-mask** pattern. This achieves ~400x speedup over naive implementations.

### Why This Matters

JAX compiles functions with `@jit` based on array shapes. If shapes change between calls, JAX recompiles the function—which is slow.

```python
# BAD - Different shapes trigger recompilation every time
for indices in batches:
    batch = jraph.batch([graphs[i] for i in indices])  # Different shapes!
    train_step(model, batch)  # Recompiles every time!
```

### The Solution: Pre-batch + Masking

Batch all data once upfront, then use boolean masks to select which samples contribute to the loss:

```python
import jax.numpy as jnp
import jraph
from flax import nnx

# Batch ALL training data once at the start
all_graphs = jraph.batch(train_data.graphs)
all_labels = jnp.array(train_data.labels)

# Use a mask to track which samples are labeled
labeled_mask = jnp.zeros(len(train_data), dtype=bool)
labeled_mask = labeled_mask.at[:50].set(True)  # Start with 50 labeled

@nnx.jit
def train_step(model, optimizer, mask):
    def loss_fn(model):
        mean, var = model(all_graphs, training=True)
        # Negative log-likelihood loss
        nll = 0.5 * (jnp.log(var) + (all_labels - mean) ** 2 / var)
        # Only count loss for labeled samples
        return jnp.sum(jnp.where(mask, nll, 0.0)) / jnp.sum(mask)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

# Training loop - no recompilation!
for epoch in range(100):
    loss = train_step(model, optimizer, labeled_mask)
```

When you acquire new samples, simply update the mask:

```python
# After acquiring new samples
new_indices = acquisition_function(model, unlabeled_indices)
labeled_mask = labeled_mask.at[new_indices].set(True)
# train_step still uses the same shapes - no recompilation!
```

---

## Data Flow

Understanding how data flows through molax:

```
SMILES string (e.g., "CCO")
    ↓ smiles_to_jraph()
jraph.GraphsTuple (single molecule graph)
    - nodes: atom features [n_atoms, n_features]
    - edges: bond features [n_bonds, n_features]
    - senders/receivers: connectivity
    ↓ jraph.batch()
jraph.GraphsTuple (batched - all molecules as one big graph)
    - nodes: [total_atoms, n_features]
    - n_node: [n_molecules] - atoms per molecule
    - n_edge: [n_molecules] - bonds per molecule
    ↓ UncertaintyGCN / DeepEnsemble / EvidentialGCN
(mean, variance) predictions per molecule
```

### Example: Loading Data

```python
from molax.utils.data import MolecularDataset

# Load dataset
dataset = MolecularDataset('datasets/esol.csv')
train_data, test_data = dataset.split(test_size=0.2, seed=42)

# Batch for training (do this once!)
import jraph
train_graphs = jraph.batch(train_data.graphs)
train_labels = jnp.array(train_data.labels)
```

---

## Uncertainty Types

molax distinguishes between two types of uncertainty:

### Epistemic Uncertainty (Model Uncertainty)

- **What**: Uncertainty due to lack of knowledge/data
- **Behavior**: Decreases with more training data
- **Use case**: Active learning - select samples where model is uncertain
- **Measured by**:
  - MC Dropout variance
  - Ensemble disagreement
  - Evidential epistemic uncertainty

### Aleatoric Uncertainty (Data Uncertainty)

- **What**: Inherent noise in the data
- **Behavior**: Cannot be reduced by more data
- **Use case**: Understanding data quality, heteroscedastic regression
- **Measured by**:
  - Predicted variance head
  - Evidential aleatoric uncertainty

### Why This Matters for Active Learning

For active learning, you typically want to select samples with high **epistemic** uncertainty—these are the samples where acquiring labels will most improve the model. High aleatoric uncertainty indicates noisy data points that won't help much.

```python
# Deep Ensemble separates the two uncertainties
ensemble = DeepEnsemble(config, n_members=5, rngs=nnx.Rngs(0))
mean, epistemic_var, aleatoric_var = ensemble(graphs, training=False)

# Use epistemic uncertainty for acquisition
scores = epistemic_var  # High = model is uncertain
```

---

## Choosing a Model

molax provides three approaches to uncertainty quantification:

### MC Dropout (`UncertaintyGCN`)

**Best for**: Quick prototyping, limited compute

```python
from molax.models.gcn import GCNConfig, UncertaintyGCN

config = GCNConfig(
    node_features=6,
    hidden_features=[64, 64],
    out_features=1,
    dropout_rate=0.1,
)
model = UncertaintyGCN(config, rngs=nnx.Rngs(0))

# Get uncertainty via multiple forward passes
mean, var = model(graphs, training=True)  # training=True enables dropout
```

**Pros**: Single model, fast training, no extra memory
**Cons**: Uncertainty estimates can be poorly calibrated

### Deep Ensembles (`DeepEnsemble`)

**Best for**: Production use, well-calibrated uncertainty

```python
from molax.models.ensemble import EnsembleConfig, DeepEnsemble

config = EnsembleConfig(
    node_features=6,
    hidden_features=[64, 64],
    out_features=1,
    n_members=5,
)
ensemble = DeepEnsemble(config, rngs=nnx.Rngs(0))

mean, epistemic_var, aleatoric_var = ensemble(graphs, training=False)
```

**Pros**: Best calibration, separate epistemic/aleatoric, robust
**Cons**: N× training time and memory

### Evidential Deep Learning (`EvidentialGCN`)

**Best for**: Single-pass uncertainty, out-of-distribution detection

```python
from molax.models.evidential import EvidentialConfig, EvidentialGCN

config = EvidentialConfig(
    node_features=6,
    hidden_features=[64, 64],
    out_features=1,
)
model = EvidentialGCN(config, rngs=nnx.Rngs(0))

mean, aleatoric_var, epistemic_var = model(graphs, training=False)
```

**Pros**: Single forward pass, explicit uncertainty decomposition
**Cons**: Requires careful loss tuning, can be overconfident

---

## Calibration

Well-calibrated uncertainty means the model's confidence matches its accuracy. molax provides tools to measure and visualize calibration:

```python
from molax.metrics import expected_calibration_error, calibration_report
from molax.metrics.visualization import plot_calibration_curve

# Compute ECE
ece = expected_calibration_error(predictions, variances, targets)
print(f"Expected Calibration Error: {ece:.4f}")

# Generate full report
report = calibration_report(predictions, variances, targets)

# Visualize
fig = plot_calibration_curve(predictions, variances, targets)
fig.savefig("calibration.png")
```

A perfectly calibrated model has ECE = 0. In practice, ECE < 0.05 is considered well-calibrated.
