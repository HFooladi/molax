# Overview

`molax` is a python library designed for aiding in drug discovery by providing active learning capabilities. It helps researchers optimize their drug discovery process by intelligently selecting the most informative compounds for testing.

## Features

- Active learning algorithms for molecular property prediction
- Integration with popular molecular representation methods
- Support for various molecular datasets
- Flexible and extensible architecture
- Easy-to-use API for drug discovery workflows

## Installation

You can install `molax` using pip:

```bash
pip install molax
```

For development installation:

```bash
git clone https://github.com/HFooladi/molax.git
cd molax
pip install -e .
```

## Quick Start

Here's a simple example of how to use `molax`:

```python
from molax import ActiveLearner
from molax.datasets import MoleculeDataset
from molax.models import PropertyPredictor

# Initialize your dataset
dataset = MoleculeDataset()

# Create a property predictor model
model = PropertyPredictor()

# Initialize the active learner
learner = ActiveLearner(model=model, dataset=dataset)

# Start the active learning process
learner.run(n_iterations=10)
```

## Documentation

For detailed documentation, please visit our [documentation page](https://molax.readthedocs.io).

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

