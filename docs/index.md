# molax

High-performance molecular active learning with JAX.

molax provides GPU-accelerated active learning for molecular property prediction, using [jraph](https://github.com/deepmind/jraph) for efficient graph batching (~400x speedup over naive implementations).

## Features

- **Multiple uncertainty methods**: MC Dropout, Deep Ensembles, Evidential Deep Learning
- **Calibration metrics**: Expected Calibration Error, calibration curves, reliability diagrams
- **Acquisition functions**: Uncertainty sampling, diversity sampling, combined strategies
- **GPU-accelerated**: Full JAX/Flax NNX integration with JIT compilation

## Installation

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
git clone https://github.com/HFooladi/molax.git
cd molax
uv pip install -e .
```

For development:

```bash
uv pip install -e .[dev]
```

## Quick Start

```python
from molax.utils.data import MolecularDataset
from molax.models.gcn import GCNConfig, UncertaintyGCN
from flax import nnx
import jraph
import jax.numpy as jnp

# Load dataset
dataset = MolecularDataset('datasets/esol.csv')
train_data, test_data = dataset.split(test_size=0.2, seed=42)

# Batch all data once (key for performance!)
train_graphs = jraph.batch(train_data.graphs)
train_labels = jnp.array(train_data.labels)

# Create model with uncertainty quantification
config = GCNConfig(
    node_features=6,
    hidden_features=[64, 64],
    out_features=1,
    dropout_rate=0.1,
)
model = UncertaintyGCN(config, rngs=nnx.Rngs(0))

# Get predictions with uncertainty
mean, variance = model(train_graphs, training=True)
```

## Next Steps

- **[Core Concepts](concepts.md)**: Learn the batch-once-then-mask pattern that enables the 400x speedup
- **[API Reference](api/models.md)**: Detailed documentation of all models and functions
- **[Roadmap](roadmap.md)**: See what's coming next

## Contributing

We welcome contributions! Please see our [GitHub repository](https://github.com/HFooladi/molax) for more information.

## License

This project is licensed under the MIT License.
