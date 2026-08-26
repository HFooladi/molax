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

# Load dataset. Pass features='rich' -- the 6-dim default cannot learn
# from additional data (see the note below).
dataset = MolecularDataset('datasets/esol.csv', features='rich')
train_data, test_data = dataset.split(test_size=0.2, seed=42)

# Batch all data once (key for performance!)
train_graphs = jraph.batch(train_data.graphs)
train_labels = jnp.array(train_data.labels)

# Create model with uncertainty quantification
config = GCNConfig(
    node_features=dataset.n_node_features,
    hidden_features=[64, 64],
    out_features=1,
    dropout_rate=0.1,
)
model = UncertaintyGCN(config, rngs=nnx.Rngs(0))

# Get predictions with uncertainty
mean, variance = model(train_graphs, training=True)
```

!!! warning "Use `features='rich'`"
    The default `"basic"` featurizer emits six raw unnormalized descriptors and
    plateaus at a test RMSE of ~2.00 on ESOL whether given 5%, 25% or 50% of the
    data — worse than simply predicting the training mean (2.13). `"rich"` at the
    same settings goes 1.94 / 1.23 / 1.21. The default is kept only for backward
    compatibility; prefer `"rich"` for new work.

## Next Steps

- **[Core Concepts](concepts.md)**: Learn the batch-once-then-mask pattern that enables the 400x speedup
- **[API Reference](api/models.md)**: Detailed documentation of all models and functions
- **[Roadmap](roadmap.md)**: See what's coming next

## Contributing

We welcome contributions! Please see our [GitHub repository](https://github.com/HFooladi/molax) for more information.

## License

This project is licensed under the MIT License.
