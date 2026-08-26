# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Rich atom featurizer** (`molax/utils/featurizers.py`)
  - `ATOM_FEATURIZERS` registry with named featurizers and declared output widths
  - `"rich"`: 29-dim one-hot encoding (element, degree, total H, hybridization,
    aromatic, in-ring, formal charge)
  - `"basic"`: the original six raw descriptors, unchanged and still the default
  - `smiles_to_jraph(smiles, features=...)` and `MolecularDataset(..., features=...)`
  - `AtomFeaturizer`, `get_atom_featurizer` exported from `molax`
  - Read the width via `ATOM_FEATURIZERS[name].dim` instead of hardcoding it
  - On ESOL (`UncertaintyGCN`, hidden `[128,128,128]`, 1500 epochs, seed 42;
    mean-predictor baseline 2.13): `"basic"` 1.33 vs `"rich"` 0.92

- **BACE-1 drug discovery case study** (`examples/bace_lead_optimization.py`)
  - Simulates Design-Make-Test-Analyze rounds under a fixed assay budget against
    BACE-1 (beta-secretase 1), 1513 inhibitors with pIC50
  - Bemis-Murcko scaffold split; model re-initialized each round; label
    standardization fitted on the labeled set only
  - Scores **hit enrichment** (top-k potent compounds recovered), not just RMSE
  - Compares random / uncertainty / greedy / UCB / uncertainty+diversity
  - `scripts/download_bace.py` and `datasets/bace.csv`
  - Writeup at `docs/case_study_bace.md`

- **`coreset_from_embeddings`** (`molax/acquisition/coreset.py`)
  - Index-based k-center greedy over a precomputed embedding matrix, for use with
    a pre-batched `GraphsTuple` driven by a boolean mask (no re-batching, so no
    JIT recompilation). `coreset_sampling` now delegates to it.

- **Flax API guardrails** (`tests/test_flax_compat.py`)
  - Pins the NNX patterns molax relies on (`nnx.Optimizer(..., wrt=nnx.Param)`,
    two-argument `optimizer.update`, `nnx.List` tracking, `variable[...]` access)
    and fails on any `DeprecationWarning` during a training step.
  - Verified against flax 0.12.2 / jax 0.9.0 / optax 0.2.6.

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
- Atom featurizer is now selectable; the `"basic"` default is byte-for-byte
  unchanged, so existing code and saved models are unaffected.
- Active learning examples re-initialize the model every round. Warm-starting one
  model across rounds gave later rounds a larger cumulative training budget, so
  the curves measured training time as much as data efficiency. This requires
  training to convergence each round (`N_EPOCHS` raised accordingly).
- `examples/active_learning_benchmark.py` and `examples/simple_active_learning.py`
  use the `"rich"` featurizer and print the mean-predictor baseline the curve must
  beat. With `"basic"` the model plateaus at RMSE ~2.00 given 5%, 25% or 50% of
  ESOL (2.02 / 2.01 / 2.03) -- worse than predicting the training mean (2.13) --
  so acquisition strategies could not be meaningfully compared.
- Model demos (`mpnn`, `gat`, `graph_transformer`, `ensemble_active_learning`,
  `evidential_active_learning`, `calibration_comparison`) use `"rich"` features.
- `examples/uncertainty_gcn_demo.py` writes its plot to `examples/assets/`
  (resolved relative to `__file__`) instead of `examples/`.

### Fixed
- **Variance heads could be permanently dead** in `UncertaintyMPNN`,
  `UncertaintyGAT` and `UncertaintyGraphTransformer`. They bounded `log_var` with
  `jnp.clip(log_var, -4.6, 4.6)`, which has exactly zero gradient outside its
  range, making saturation an absorbing state. MPNN fell in completely: every
  molecule pinned at `exp(4.6) = 99.484` with a `var_head` gradient of exactly
  `0.0`, so `mpnn_demo` reported an identical std dev of 9.974 for all molecules.
  Replaced with a leaky clip (`molax/models/bounds.py`) -- identity inside the
  range, small constant slope outside -- so a saturated head always recovers.
  (`tanh` is not sufficient: its derivative underflows to zero in float32 below
  the magnitudes actually observed.)
- **`DeepEnsemble` ignored its `rngs` argument**, hardcoding `nnx.Rngs(i)` per
  member, so two ensembles built from different seeds were bit-identical.
- **`examples/active_learning_benchmark.py` plotted three curves for two
  strategies** -- the `"combined"` branch was a verbatim copy of `"uncertainty"`.
  It now combines uncertainty with Core-Set diversity.
- `examples/ensemble_active_learning.py` hardcoded `baseline_rmse = 2.5` in its
  sanity check, so a model at RMSE 2.4 reported PASS while being worse than
  predicting the mean (2.13). Now computed from the data.

### Removed
- Untracked a scratch notebook and four regenerable example plots that had been
  committed by accident; added `.gitignore` entries. The files remain on disk.

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
