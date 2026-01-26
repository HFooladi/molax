# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

Molax is a high-performance JAX framework for molecular active learning. It uses **jraph** for efficient graph batching, achieving ~400x speedup over naive implementations.

## Quick Commands

```bash
# Setup (using uv)
source env.sh

# Test
pytest tests/

# Format and lint (ruff for both)
ruff format .
ruff check .

# Build docs locally
uv pip install -e .[docs]
mkdocs serve

# Run examples
python examples/simple_active_learning.py
python examples/active_learning_benchmark.py
python examples/ensemble_demo.py
python examples/evidential_demo.py
```

## Architecture

### Performance-Critical Pattern

**IMPORTANT**: The key to performance is batching all data once and using masking:

```python
# GOOD - batch once, use masking (fast)
all_graphs = jraph.batch(train_data.graphs)
labeled_mask = jnp.zeros(n_train, dtype=bool).at[labeled_indices].set(True)

@nnx.jit
def train_step(model, optimizer, mask):
    def loss_fn(model):
        mean, var = model(all_graphs, training=True)
        nll = compute_nll(mean, var, labels)
        return jnp.sum(jnp.where(mask, nll, 0.0)) / jnp.sum(mask)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

# BAD - rebatching causes JIT recompilation (slow)
for indices in batches:
    batch = jraph.batch([graphs[i] for i in indices])  # Different shapes!
    train_step(model, batch)  # Recompiles every time!
```

### Data Flow

```
SMILES string
    ↓ smiles_to_jraph()
jraph.GraphsTuple (single molecule)
    ↓ jraph.batch()
jraph.GraphsTuple (batched - all molecules as one graph)
    ↓ UncertaintyGCN / DeepEnsemble / EvidentialGCN
(mean, variance) predictions
```

### Key Files

| File | Purpose |
|------|---------|
| `molax/models/gcn.py` | `GCNConfig`, `UncertaintyGCN`, `MolecularGCN`, `train_step`, `eval_step` |
| `molax/models/ensemble.py` | `EnsembleConfig`, `DeepEnsemble` for ensemble uncertainty |
| `molax/models/evidential.py` | `EvidentialConfig`, `EvidentialGCN` for evidential uncertainty |
| `molax/utils/data.py` | `MolecularDataset`, `smiles_to_jraph`, `batch_graphs` |
| `molax/acquisition/uncertainty.py` | `uncertainty_sampling`, `ensemble_uncertainty_sampling`, `evidential_uncertainty_sampling` |
| `molax/acquisition/diversity.py` | `diversity_sampling` |
| `molax/metrics/calibration.py` | `expected_calibration_error`, `calibration_report` |
| `molax/metrics/visualization.py` | `plot_calibration_curve`, `plot_reliability_diagram` |

### Model API

```python
from molax.models.gcn import GCNConfig, UncertaintyGCN

config = GCNConfig(
    node_features=6,        # Atom features
    hidden_features=[64, 64],  # GCN layers
    out_features=1,         # Output dim
    dropout_rate=0.1,
)
model = UncertaintyGCN(config, rngs=nnx.Rngs(0))

# Forward pass
mean, variance = model(batched_graphs, training=True)
```

### Ensemble API

```python
from molax.models.ensemble import EnsembleConfig, DeepEnsemble

config = EnsembleConfig(
    node_features=6,
    hidden_features=[64, 64],
    out_features=1,
    n_members=5,
)
ensemble = DeepEnsemble(config, rngs=nnx.Rngs(0))

# Returns separate epistemic and aleatoric uncertainty
mean, epistemic_var, aleatoric_var = ensemble(batched_graphs, training=False)
```

### Evidential API

```python
from molax.models.evidential import EvidentialConfig, EvidentialGCN

config = EvidentialConfig(
    node_features=6,
    hidden_features=[64, 64],
    out_features=1,
)
model = EvidentialGCN(config, rngs=nnx.Rngs(0))

# Single forward pass for both uncertainties
mean, aleatoric_var, epistemic_var = model(batched_graphs, training=False)
```

### Calibration Metrics

```python
from molax.metrics import expected_calibration_error, calibration_report
from molax.metrics.visualization import plot_calibration_curve

# Compute ECE
ece = expected_calibration_error(predictions, variances, targets)

# Generate full report
report = calibration_report(predictions, variances, targets)

# Visualize
fig = plot_calibration_curve(predictions, variances, targets)
```

### Optimizer Pattern (Flax 0.11+)

```python
import optax
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

# In training step:
loss, grads = nnx.value_and_grad(loss_fn)(model)
optimizer.update(model, grads)
```

## Testing

```bash
pytest tests/ -v                    # All tests
pytest tests/test_gcn.py -v         # Model tests
pytest tests/test_ensemble.py -v    # Ensemble tests
pytest tests/test_evidential.py -v  # Evidential tests
pytest tests/test_acquisition.py -v # Acquisition tests
pytest tests/test_calibration.py -v # Calibration tests
```

## Dependencies

Core: `jax`, `flax`, `optax`, `jraph`, `rdkit`, `pandas`, `numpy`, `matplotlib`

## Dataset

ESOL dataset (1,128 molecules): `datasets/esol.csv`

Download: `python scripts/download_esol.py`

## GitHub CLI

Use `gh_cli` command to interact with GitHub:

```bash
# Check workflow runs
gh_cli run list

# View specific run
gh_cli run view <run-id>

# Watch a run in progress
gh_cli run watch <run-id>

# View workflow logs
gh_cli run view <run-id> --log

# Trigger workflow manually
gh_cli workflow run ci.yml
```
