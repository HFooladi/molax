# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

### Removed

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

[Unreleased]: https://github.com/HFooladi/molax/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/HFooladi/molax/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/HFooladi/molax/releases/tag/v0.1.0
