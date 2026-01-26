# Models API

This page documents the neural network models available in molax for molecular property prediction with uncertainty quantification.

## GCN Models

The core Graph Convolutional Network models with MC Dropout uncertainty.

### GCNConfig

::: molax.models.gcn.GCNConfig

### UncertaintyGCN

::: molax.models.gcn.UncertaintyGCN

### MolecularGCN

::: molax.models.gcn.MolecularGCN

---

## Deep Ensembles

Ensemble methods for improved uncertainty quantification through model disagreement.

### EnsembleConfig

::: molax.models.ensemble.EnsembleConfig

### DeepEnsemble

::: molax.models.ensemble.DeepEnsemble

---

## Evidential Deep Learning

Single-pass uncertainty estimation using evidential neural networks.

### EvidentialConfig

::: molax.models.evidential.EvidentialConfig

### EvidentialGCN

::: molax.models.evidential.EvidentialGCN

---

## Training Utilities

::: molax.models.gcn.train_step

::: molax.models.gcn.eval_step
