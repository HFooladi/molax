# molax

Molecular active learning with JAX - a lightweight framework for active learning in molecular property prediction.

## Installation

```bash
git clone https://github.com/yourusername/molax
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
from molax.utils.data import MolecularDataset
from molax.models import UncertaintyGCN
from molax.acquisition import combined_acquisition

# Load your data
dataset = MolecularDataset('data/molecules.csv', 
                          smiles_col='smiles', 
                          label_col='property')

# Split dataset
train_data, test_data = dataset.split_train_test(test_size=0.2)

# Initialize model
model = UncertaintyGCN(
    hidden_features=(64, 64),
    output_features=1,
    dropout_rate=0.1
)

# Run active learning loop
# See examples/active_learning.py for complete implementation
```

## Features

- Graph neural networks implemented in JAX/Flax
- Uncertainty estimation via MC dropout
- Multiple acquisition functions
- Efficient batch selection
- RDKit-based molecular processing

## Examples

Check `examples/active_learning.py` for a complete active learning pipeline.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License

## Citation

```bibtex
@software{molax2025,
  title={molax: Molecular Active Learning with JAX},
  author={Hosein Fooladi},
  year={2025},
  url={https://github.com/HFooladi/molax}
}
```