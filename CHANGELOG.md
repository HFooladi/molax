# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Graph Attention Network (GAT)** (`molax/models/gat.py`)
  - `UncertaintyGAT` model with multi-head attention for adaptive neighbor weighting
  - `GATConfig` with configurable n_heads, edge_features, attention_dropout_rate
  - `GATAttention` and `GATLayer` components
  - Training utilities: `train_gat_step`, `eval_gat_step`, `get_gat_uncertainties`
  - Same API as `UncertaintyGCN`/`UncertaintyMPNN` for drop-in replacement
  - Optional edge feature incorporation in attention computation
  - Comprehensive tests and demo example

- **Message Passing Neural Network (MPNN)** (`molax/models/mpnn.py`)
  - `UncertaintyMPNN` model that leverages edge features (bond information)
  - `MPNNConfig` with configurable aggregation (sum, mean, max)
  - `MessageFunction` and `MessagePassingLayer` components
  - Training utilities: `train_mpnn_step`, `eval_mpnn_step`, `get_mpnn_uncertainties`
  - Same API as `UncertaintyGCN` for drop-in replacement with acquisition functions
  - Comprehensive tests (32 tests) and demo example

### Changed

### Fixed

### Removed

---

## [0.3.0] - 2025-01-26

### Added
- **Documentation Site** deployed to GitHub Pages
  - Core concepts guide explaining batch-once-then-mask pattern
  - Full API reference with mkdocstrings
  - Installation and quick start guides

- **Calibration Metrics** (`molax/metrics/`)
  - `expected_calibration_error` for measuring uncertainty quality
  - `compute_calibration_curve` for calibration analysis
  - `negative_log_likelihood` metric
  - `calibration_report` for comprehensive analysis
  - Visualization tools: `plot_calibration_curve`, `plot_reliability_diagram`

### Changed
- **Migrated to uv** for package management
  - CI/CD workflows now use `astral-sh/setup-uv@v5`
  - Installation instructions updated to uv-first
  - Faster dependency resolution and caching

- Simplified README with links to documentation site

---

## [0.2.0] - 2025-01-26

### Added
- **Evidential Deep Learning** (`molax/models/evidential.py`)
  - `EvidentialGCN` model for single-pass uncertainty estimation
  - Normal-Inverse-Gamma (NIG) loss function with configurable regularization
  - Separate aleatoric and epistemic uncertainty outputs
  - `evidential_uncertainty_sampling` and `combined_evidential_acquisition` functions
  - Comprehensive tests and example script

- **Deep Ensembles** (`molax/models/ensemble.py`)
  - `DeepEnsemble` class with configurable number of members
  - Separate epistemic (model disagreement) and aleatoric (data noise) uncertainty
  - `ensemble_uncertainty_sampling` and `combined_ensemble_acquisition` functions
  - Training utilities for ensemble members

- Feature roadmap documentation (`docs/roadmap.md`)

### Changed
- Migrated to jraph for efficient graph batching (~400x speedup)
- Updated to Flax NNX API (0.12+) with `nnx.List` for layer collections

---

## [0.1.0] - 2025-01-15

### Added
- Initial release
- `MolecularGCN` and `UncertaintyGCN` models
- MC Dropout uncertainty estimation
- Basic acquisition functions: `uncertainty_sampling`, `diversity_sampling`, `combined_acquisition`
- ESOL dataset support
- SMILES to jraph graph conversion utilities

---

[Unreleased]: https://github.com/HFooladi/molax/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/HFooladi/molax/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/HFooladi/molax/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/HFooladi/molax/releases/tag/v0.1.0
