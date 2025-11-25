# molax

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/molax/badge/?version=latest)](https://molax.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/HFooladi/molax/actions/workflows/tests.yml/badge.svg)](https://github.com/HFooladi/molax/actions/workflows/ci.yml)

Molecular active learning with JAX - a lightweight framework for active learning in molecular property prediction.

## Key Features

- 🧠 Graph neural networks implemented in JAX/Flax for molecular representation learning
- 🔄 Complete active learning workflow for molecular property prediction
- 🎯 Multiple acquisition functions for diverse exploration strategies

## Installation

```bash
git clone https://github.com/HFooladi/molax
cd molax
pip install -r requirements.txt
```

Required dependencies:
```
jax
flax
optax
rdkit
pandas
numpy
```

## Usage

Basic usage with SMILES data:

```python
import flax.nnx as nnx
import optax
from molax.utils.data import MolecularDataset
from molax.models.gcn import UncertaintyGCN, UncertaintyGCNConfig

# Load your data
dataset = MolecularDataset('datasets/molecules.csv')

# Split dataset
train_data, test_data = dataset.split_train_test(test_size=0.2)

# Initialize model
config = UncertaintyGCNConfig(
    in_features=train_data.graphs[0][0].shape[1],
    hidden_features=[64, 64],
    out_features=1,
    dropout_rate=0.1,
)
model = UncertaintyGCN(config)

# Initialize optimizer
model_and_opt = nnx.ModelAndOptimizer(model, optax.adam(1e-3))

# Run active learning loop
# See examples/simple_active_learning.py for complete implementation
```

## Features

- Graph neural networks implemented in JAX/Flax
- Uncertainty estimation via MC dropout
- Multiple acquisition functions
- Efficient batch selection
- RDKit-based molecular processing

## Examples

Check `examples/simple_active_learning.py` for a complete active learning pipeline with uncertainty-based acquisition.

For uncertainty quantification demonstration, see `examples/uncertainty_gcn_demo.py`.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Citation

```bibtex
@software{molax2025,
  title={molax: Molecular Active Learning with JAX},
  author={Hosein Fooladi},
  year={2025},
  url={https://github.com/hfooladi/molax},
  description={A lightweight framework for active learning in molecular property prediction}

}
```

## License

MIT License

## Acknowledgements

- This project builds upon the excellent JAX, Flax, and RDKit libraries.
- Thanks to all contributors who have helped improve this project.
