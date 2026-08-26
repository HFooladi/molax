# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

Molax is a high-performance JAX framework for molecular active learning. It uses **jraph** for efficient graph batching, achieving ~400x speedup over naive implementations.

## Quick Commands

```bash
# Setup (using uv)
source setup.sh

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
python examples/mpnn_demo.py
python examples/gat_demo.py
python examples/graph_transformer_demo.py
python examples/ensemble_active_learning.py
python examples/evidential_active_learning.py
python examples/acquisition_strategies_demo.py
python examples/calibration_comparison.py
python examples/uncertainty_gcn_demo.py
python examples/bace_lead_optimization.py   # drug discovery case study
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
    ↓ UncertaintyGCN / UncertaintyMPNN / UncertaintyGAT / UncertaintyGraphTransformer / DeepEnsemble / EvidentialGCN
(mean, variance) predictions
```

### Key Files

| File | Purpose |
|------|---------|
| `molax/models/gcn.py` | `GCNConfig`, `UncertaintyGCN`, `MolecularGCN`, `train_step`, `eval_step` |
| `molax/models/mpnn.py` | `MPNNConfig`, `UncertaintyMPNN` for edge-aware message passing |
| `molax/models/gat.py` | `GATConfig`, `UncertaintyGAT` for attention-based message passing |
| `molax/models/graph_transformer.py` | `GraphTransformerConfig`, `UncertaintyGraphTransformer` for full self-attention with positional encodings |
| `molax/models/ensemble.py` | `EnsembleConfig`, `DeepEnsemble` for ensemble uncertainty |
| `molax/models/evidential.py` | `EvidentialConfig`, `EvidentialGCN` for evidential uncertainty |
| `molax/utils/data.py` | `MolecularDataset`, `smiles_to_jraph`, `batch_graphs` |
| `molax/acquisition/uncertainty.py` | `uncertainty_sampling`, `ensemble_uncertainty_sampling`, `evidential_uncertainty_sampling`, `diversity_sampling`, `combined_*` strategies |
| `molax/acquisition/bald.py` | `bald_sampling`, `ensemble_bald_sampling`, `evidential_bald_sampling` (mutual information) |
| `molax/acquisition/batch_aware.py` | `batch_bald_sampling`, `dpp_sampling`, `combined_batch_acquisition` |
| `molax/acquisition/coreset.py` | `coreset_sampling`, `coreset_sampling_with_scores` (K-center greedy) |
| `molax/acquisition/expected_change.py` | `egl_sampling`, `egl_acquisition` (expected gradient length) |
| `molax/metrics/calibration.py` | `expected_calibration_error`, `evaluate_calibration`, `TemperatureScaling`, `negative_log_likelihood`, `sharpness` |
| `molax/metrics/visualization.py` | `plot_reliability_diagram`, `plot_calibration_comparison`, `plot_uncertainty_vs_error`, `create_calibration_report` |

### Model API

All models share a common pattern: `Config` → `Model(config, rngs)` → `model(graphs, training)` → `(mean, variance)`.

| Model | Config | Extra Config Params | Output |
|-------|--------|-------------------|--------|
| `UncertaintyGCN` | `GCNConfig` | — | `(mean, var)` |
| `UncertaintyMPNN` | `MPNNConfig` | `edge_features`, `aggregation` | `(mean, var)` |
| `UncertaintyGAT` | `GATConfig` | `n_heads`, `attention_dropout_rate`, `negative_slope` | `(mean, var)` |
| `UncertaintyGraphTransformer` | `GraphTransformerConfig` | `n_heads`, `ffn_ratio`, `pe_type`, `pe_dim` | `(mean, var)` |
| `DeepEnsemble` | `EnsembleConfig` | `n_members` | `(mean, epistemic_var, aleatoric_var)` |
| `EvidentialGCN` | `EvidentialConfig` | — | `(mean, aleatoric_var, epistemic_var)` |

All configs share: `node_features`, `hidden_features`, `out_features`, `dropout_rate`. Edge-aware models (MPNN, GAT, GraphTransformer) also take `edge_features`.

```python
# Example: any model follows this pattern
from molax.models import GCNConfig, UncertaintyGCN

config = GCNConfig(node_features=6, hidden_features=[64, 64], out_features=1, dropout_rate=0.1)
model = UncertaintyGCN(config, rngs=nnx.Rngs(0))
mean, variance = model(batched_graphs, training=True)

# Ensemble returns 3 values
from molax.models import EnsembleConfig, DeepEnsemble
ensemble = DeepEnsemble(EnsembleConfig(node_features=6, hidden_features=[64, 64], out_features=1, n_members=5), rngs=nnx.Rngs(0))
mean, epistemic_var, aleatoric_var = ensemble(batched_graphs, training=False)

# GraphTransformer supports embedding extraction for CoreSet
embeddings = model.extract_embeddings(batched_graphs)
```

### Calibration Metrics

```python
from molax.metrics import expected_calibration_error, evaluate_calibration, TemperatureScaling
from molax.metrics import plot_reliability_diagram, create_calibration_report

# Compute ECE
ece = expected_calibration_error(predictions, variances, targets)

# Full evaluation (returns dict with ece, nll, sharpness, mse, rmse)
metrics = evaluate_calibration(predictions, variances, targets)

# Post-hoc calibration
scaler = TemperatureScaling()
scaler.fit(val_mean, val_var, val_targets)
calibrated_var = scaler.transform(test_var)

# Visualize
plot_reliability_diagram(predictions, variances, targets)
create_calibration_report(predictions, variances, targets)  # Multi-plot report
```

### Atom Featurizers

**IMPORTANT**: the default `"basic"` featurizer emits six raw unnormalized
descriptors and trains poorly. Its `chiral_tag` column is constant zero for any
SMILES without explicit stereochemistry -- a dead input dimension. Use
`"rich"` (29-dim one-hot) for anything where accuracy matters.

Measured on ESOL, same architecture and budget for both
(`UncertaintyGCN`, hidden `[128, 128, 128]`, 1500 epochs, 80/20 split, seed 42;
mean-predictor baseline RMSE 2.13):

| Featurizer | Dims | Test RMSE |
|------------|------|-----------|
| `"basic"`  | 6    | 1.33      |
| `"rich"`   | 29   | 0.92      |

**`"basic"` does not scale with data.** With hidden `[64, 64]` and 400 epochs it
plateaus at RMSE ~2.00 whether given 5%, 25% or 50% of ESOL (2.02 / 2.01 / 2.03)
— worse than predicting the training mean. `"rich"` at the same settings goes
1.94 / 1.23 / 1.21. Any active learning experiment run on `"basic"` is measuring
noise, because the model cannot use the data the strategy acquires.

```python
from molax import ATOM_FEATURIZERS, smiles_to_jraph, MolecularDataset

graph = smiles_to_jraph("CCO", features="rich")
dataset = MolecularDataset("datasets/bace.csv", features="rich")

# Read the width from the registry rather than hardcoding it
n_features = ATOM_FEATURIZERS["rich"].dim   # 29
```

The default stays `"basic"` for backward compatibility.

### Optimizer Pattern (Flax 0.11+)

Verified against **flax 0.12.2 / jax 0.9.0 / optax 0.2.6**. `tests/test_flax_compat.py`
guards these patterns; run it first if a Flax upgrade breaks something.

```python
import optax
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

# In training step:
loss, grads = nnx.value_and_grad(loss_fn)(model)
optimizer.update(model, grads)
```

Access variable arrays with `variable[...]`, not `variable.value` -- the latter
is deprecated in Flax 0.12.

### Active Learning Protocol

Two mistakes make an AL curve look better than it is. Both are avoided in
`examples/bace_lead_optimization.py` and `examples/active_learning_benchmark.py`:

- **Re-initialize the model every round.** Warm-starting one model across rounds
  gives later rounds more cumulative gradient steps, so the curve measures
  training budget as much as data efficiency.
- **Standardize labels on the labeled set only**, refitting each round. Pool
  labels are exactly what the campaign has not measured yet; using them leaks
  the answer into the experiment.

Also note that **MC-dropout variance is a weak acquisition signal here** -- on
ESOL it correlates -0.05 with actual error, while the model's own predicted
variance head correlates +0.36. Prefer the variance head or ensemble variance.

## Testing

```bash
pytest tests/ -v                              # All tests
pytest tests/test_gcn.py -v                   # GCN model tests
pytest tests/test_mpnn.py -v                  # MPNN model tests
pytest tests/test_gat.py -v                   # GAT model tests
pytest tests/test_graph_transformer.py -v    # Graph Transformer tests
pytest tests/test_ensemble.py -v              # Ensemble tests
pytest tests/test_evidential.py -v            # Evidential tests
pytest tests/test_acquisition.py -v           # Acquisition tests (uncertainty/diversity)
pytest tests/test_bald.py -v                  # BALD acquisition tests
pytest tests/test_batch_aware.py -v           # BatchBALD/DPP tests
pytest tests/test_coreset.py -v               # CoreSet acquisition tests
pytest tests/test_expected_change.py -v       # Expected gradient length tests
pytest tests/test_calibration.py -v           # Calibration tests
pytest tests/test_featurizers.py -v           # Atom featurizer tests
pytest tests/test_flax_compat.py -v           # Flax NNX API guardrails
```

## Dependencies

Core: `jax`, `flax`, `optax`, `jraph`, `rdkit`, `pandas`, `numpy`, `matplotlib`

## Datasets

| Dataset | Molecules | Endpoint | Path | Download |
|---------|-----------|----------|------|----------|
| ESOL | 1,128 | log solubility | `datasets/esol.csv` | `python scripts/download_esol.py` |
| BACE-1 | 1,513 | pIC50 (beta-secretase) | `datasets/bace.csv` | `python scripts/download_bace.py` |

BACE-1 is a real medicinal chemistry SAR series and is what the drug discovery
case study (`examples/bace_lead_optimization.py`) runs on.

## GitHub CLI

Use `gh` command to interact with GitHub:

```bash
# Check workflow runs
gh run list

# View specific run
gh run view <run-id>

# Watch a run in progress
gh run watch <run-id>

# View workflow logs
gh run view <run-id> --log

# Trigger workflow manually
gh workflow run ci.yml
```
