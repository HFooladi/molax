# molax

[![CI](https://github.com/hfooladi/molax/actions/workflows/ci.yml/badge.svg)](https://github.com/hfooladi/molax/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-%E2%89%A50.4.20-9cf.svg)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-%E2%89%A50.12.0-orange.svg)](https://github.com/google/flax)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Molecular active learning with JAX - a high-performance framework for active learning in molecular property prediction.

## Key Features

- **Fast**: Uses [jraph](https://github.com/deepmind/jraph) for efficient batched graph processing (~400x faster than naive implementations)
- **GPU-accelerated**: Full JAX/Flax NNX integration with JIT compilation
- **Uncertainty quantification**: MC dropout for epistemic uncertainty estimation
- **Multiple acquisition functions**: Random, uncertainty, and combined strategies

## Installation

```bash
git clone https://github.com/HFooladi/molax
cd molax
pip install -e .
```

## Quick Start

```python
import jraph
from molax.utils.data import MolecularDataset
from molax.models.gcn import GCNConfig, UncertaintyGCN

# Load data
dataset = MolecularDataset('datasets/esol.csv')
train_data, test_data = dataset.split(test_size=0.2, seed=42)

# Batch all data once (key for performance!)
train_graphs = jraph.batch(train_data.graphs)
train_labels = train_data.labels

# Create model
config = GCNConfig(
    node_features=6,
    hidden_features=[64, 64],
    out_features=1,
    dropout_rate=0.1,
)
model = UncertaintyGCN(config, rngs=nnx.Rngs(0))

# Training with JIT compilation
@nnx.jit
def train_step(model, optimizer, mask):
    def loss_fn(model):
        mean, var = model(train_graphs, training=True)
        nll = 0.5 * (jnp.log(var) + (train_labels - mean) ** 2 / var)
        return jnp.sum(jnp.where(mask, nll, 0.0)) / jnp.sum(mask)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

# Use masking for active learning (avoids JIT recompilation)
labeled_mask = jnp.zeros(len(train_data), dtype=bool)
labeled_mask = labeled_mask.at[:50].set(True)  # Start with 50 labeled

loss = train_step(model, optimizer, labeled_mask)
```

## Performance

The key to performance is **batching all data once** and using **masking** for active learning:

| Approach | Time (10 iterations, 50 epochs each) |
|----------|--------------------------------------|
| Naive (graph-by-graph) | >10 minutes |
| Optimized (jraph + masking) | ~1.4 seconds |

**~400x speedup** by avoiding JIT recompilation.

## Datasets

- `datasets/esol.csv` - ESOL dataset with 1,128 molecules and aqueous solubility values

Download the ESOL dataset:
```bash
python scripts/download_esol.py
```

## Examples

```bash
# Simple active learning demo
python examples/simple_active_learning.py

# Benchmark comparing acquisition strategies
python examples/active_learning_benchmark.py
```

## Architecture

### Data Flow
```
SMILES → smiles_to_jraph() → jraph.GraphsTuple → jraph.batch() → Model
```

### Key Components

- **`MolecularDataset`**: Loads SMILES, converts to jraph graphs
- **`UncertaintyGCN`**: GCN with mean/variance heads for uncertainty
- **`uncertainty_sampling`**: MC dropout-based acquisition
- **`diversity_sampling`**: Feature-space diversity selection

## Citation

```bibtex
@software{molax2025,
  title={molax: Molecular Active Learning with JAX},
  author={Hosein Fooladi},
  year={2025},
  url={https://github.com/hfooladi/molax}
}
```

## License

MIT License
