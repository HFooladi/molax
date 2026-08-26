# molax

[![CI](https://github.com/hfooladi/molax/actions/workflows/ci.yml/badge.svg)](https://github.com/hfooladi/molax/actions/workflows/ci.yml)
[![Docs](https://github.com/hfooladi/molax/actions/workflows/docs.yml/badge.svg)](https://hfooladi.github.io/molax/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-%E2%89%A50.4.20-9cf.svg)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-%E2%89%A50.12.0-orange.svg)](https://github.com/google/flax)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

High-performance molecular active learning with JAX. Built with [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html) (the modern Flax API) and [jraph](https://github.com/deepmind/jraph) for efficient graph batching, achieving ~400x speedup over naive implementations.

**[Documentation](https://hfooladi.github.io/molax/)** | **[API Reference](https://hfooladi.github.io/molax/api/models.html)**

## Installation

```bash
# Using uv (recommended)
git clone https://github.com/HFooladi/molax
cd molax
uv pip install -e .

# Or with pip
pip install -e .
```

## Quick Start

```python
from molax.utils.data import MolecularDataset
from molax.models.gcn import GCNConfig, UncertaintyGCN
from flax import nnx
import jraph

# Load and batch data. 'rich' gives 29-dim one-hot atom features and trains
# far better than the legacy 6-dim default (ESOL test RMSE 0.92 vs 1.33).
dataset = MolecularDataset('datasets/esol.csv', features='rich')
train_data, test_data = dataset.split(test_size=0.2, seed=42)
train_graphs = jraph.batch(train_data.graphs)

# Create model with uncertainty
config = GCNConfig(
    node_features=dataset.n_node_features, hidden_features=[64, 64], out_features=1
)
model = UncertaintyGCN(config, rngs=nnx.Rngs(0))

# Get predictions with uncertainty
mean, variance = model(train_graphs, training=True)
```

See the [Core Concepts](https://hfooladi.github.io/molax/concepts.html) guide for the batch-once-then-mask pattern that enables the 400x speedup.

## Features

- **Multiple uncertainty methods**: MC Dropout, Deep Ensembles, Evidential Deep Learning
- **Calibration metrics**: ECE, calibration curves, reliability diagrams
- **Acquisition functions**: Uncertainty sampling, diversity sampling, combined strategies
- **GPU-accelerated**: Full JAX/Flax NNX integration with JIT compilation

## Drug discovery case study

`examples/bace_lead_optimization.py` simulates Design-Make-Test-Analyze rounds
against BACE-1 (beta-secretase 1, an Alzheimer's target) under a fixed assay
budget, using a Bemis-Murcko scaffold split.

```bash
python scripts/download_bace.py
python examples/bace_lead_optimization.py
```

It scores acquisition strategies on **hit enrichment** — how many of the most
potent compounds a campaign recovers per assay — rather than on global RMSE,
because those are different objectives and they do not agree. See the
[case study writeup](https://hfooladi.github.io/molax/case_study_bace.html).

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
