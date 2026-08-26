# Data Utilities

This page documents the data loading and processing utilities.

## Dataset Classes

### MolecularDataset

::: molax.utils.data.MolecularDataset

---

## Graph Conversion

Functions for converting molecular representations to graph format.

### smiles_to_jraph

::: molax.utils.data.smiles_to_jraph

### batch_graphs

::: molax.utils.data.batch_graphs

### unbatch_graphs

::: molax.utils.data.unbatch_graphs

---

## Atom Featurizers

Node features are produced by a named atom featurizer. `"basic"` is the default
and emits the original six raw descriptors; `"rich"` emits a 29-dimensional
one-hot encoding that trains substantially better.

On ESOL, with the same architecture and training budget for both
(`UncertaintyGCN`, hidden `[128, 128, 128]`, 1500 epochs, 80/20 split, seed 42):

| Featurizer | Dims | Test RMSE |
|------------|------|-----------|
| `"basic"`  | 6    | 1.33      |
| `"rich"`   | 29   | 0.92      |
| _predict the training mean_ | — | _2.13_ |

```python
from molax import ATOM_FEATURIZERS, MolecularDataset, smiles_to_jraph

graph = smiles_to_jraph("CCO", features="rich")
dataset = MolecularDataset("datasets/bace.csv", features="rich")

n_features = ATOM_FEATURIZERS["rich"].dim  # 29 — read it, don't hardcode it
```

### AtomFeaturizer

::: molax.utils.featurizers.AtomFeaturizer

### get_atom_featurizer

::: molax.utils.featurizers.get_atom_featurizer
