# Molax Feature Roadmap

A prioritized roadmap of killer features for molax, focusing on advancing uncertainty quantification for molecular active learning.

**Target Audiences:**
- Drug discovery researchers seeking reliable predictions with confidence estimates
- ML researchers exploring uncertainty quantification and active learning methods

**Scope:** 2D graph-based molecular representations only (no 3D conformers).

---

## Current Capabilities

| Feature | Status |
|---------|--------|
| Efficient jraph batching (~400x speedup) | ✅ |
| GCN with uncertainty head | ✅ |
| MC Dropout uncertainty | ✅ |
| Uncertainty sampling | ✅ |
| Diversity sampling | ✅ |
| Combined acquisition | ✅ |
| ESOL dataset support | ✅ |
| Flax NNX integration | ✅ |

---

## Phase 1: Uncertainty Excellence (High Priority)

Better uncertainty quantification is the core differentiator for active learning. These features directly improve the quality and reliability of uncertainty estimates.

### 1.1 Deep Ensembles ✅

**Status:** Implemented in `molax/models/ensemble.py`

**What:** Train N independent GCN models with different random initializations; use prediction disagreement as uncertainty.

**Why:** Ensembles consistently outperform single-model uncertainty methods. They capture both aleatoric (data) and epistemic (model) uncertainty.

**Implementation:**

```python
# molax/models/ensemble.py
from dataclasses import dataclass
from flax import nnx
import jax.numpy as jnp
from molax.models.gcn import GCNConfig, UncertaintyGCN

@dataclass
class EnsembleConfig:
    base_config: GCNConfig
    n_members: int = 5

class DeepEnsemble(nnx.Module):
    def __init__(self, config: EnsembleConfig, rngs: nnx.Rngs):
        self.members = [
            UncertaintyGCN(config.base_config, rngs=nnx.Rngs(i))
            for i in range(config.n_members)
        ]

    def __call__(self, graphs, training: bool = False):
        # Collect predictions from all members
        predictions = [m(graphs, training=training) for m in self.members]
        means = jnp.stack([p[0] for p in predictions])  # (N, batch)

        # Ensemble mean and variance
        ensemble_mean = jnp.mean(means, axis=0)
        epistemic_var = jnp.var(means, axis=0)  # Disagreement
        aleatoric_var = jnp.mean(jnp.stack([p[1] for p in predictions]), axis=0)
        total_var = epistemic_var + aleatoric_var

        return ensemble_mean, total_var, epistemic_var
```

**Acceptance Criteria:**
- [x] `DeepEnsemble` class with configurable number of members
- [x] Separate epistemic and aleatoric uncertainty outputs
- [x] Parallel training support for ensemble members
- [x] Tests comparing ensemble vs MC Dropout uncertainty quality

---

### 1.2 Evidential Deep Learning ✅

**Status:** Implemented in `molax/models/evidential.py`

**What:** Directly predict uncertainty without MC sampling by modeling output as a higher-order distribution (Normal-Inverse-Gamma).

**Why:** Single forward pass for uncertainty (faster inference), well-calibrated for out-of-distribution detection.

**Reference:** [Amini et al., NeurIPS 2020](https://arxiv.org/abs/1910.02600)

**Implementation:**

```python
# molax/models/evidential.py
import jax.numpy as jnp
from flax import nnx

class EvidentialHead(nnx.Module):
    """Predicts Normal-Inverse-Gamma parameters for evidential regression."""

    def __init__(self, in_features: int, rngs: nnx.Rngs):
        # Output: (gamma, nu, alpha, beta) - NIG parameters
        self.linear = nnx.Linear(in_features, 4, rngs=rngs)

    def __call__(self, x):
        out = self.linear(x)
        # Ensure valid parameter ranges
        gamma = out[..., 0]  # Mean prediction
        nu = nnx.softplus(out[..., 1]) + 1e-6  # > 0
        alpha = nnx.softplus(out[..., 2]) + 1.0  # > 1
        beta = nnx.softplus(out[..., 3]) + 1e-6  # > 0
        return gamma, nu, alpha, beta

def evidential_loss(gamma, nu, alpha, beta, targets, lambda_reg=0.1):
    """NIG negative log-likelihood with regularization."""
    omega = 2 * beta * (1 + nu)
    nll = (
        0.5 * jnp.log(jnp.pi / nu)
        - alpha * jnp.log(omega)
        + (alpha + 0.5) * jnp.log((targets - gamma)**2 * nu + omega)
        + jnp.lgamma(alpha) - jnp.lgamma(alpha + 0.5)
    )
    # Regularize evidence on errors
    reg = lambda_reg * jnp.abs(targets - gamma) * (2 * nu + alpha)
    return jnp.mean(nll + reg)

def evidential_uncertainty(nu, alpha, beta):
    """Extract aleatoric and epistemic uncertainty from NIG params."""
    aleatoric = beta / (alpha - 1)  # Expected variance
    epistemic = aleatoric / nu      # Uncertainty in the variance
    return aleatoric, epistemic
```

**Acceptance Criteria:**
- [x] `EvidentialGCN` model variant
- [x] NIG loss function with configurable regularization
- [x] Separate aleatoric/epistemic uncertainty outputs
- [x] Comparison with MC Dropout on OOD detection (in tests)

---

### 1.3 Calibration Metrics ✅

**Status:** Implemented in `molax/metrics/`

**What:** Quantify how well predicted uncertainties match actual error frequencies.

**Why:** Raw uncertainties are meaningless without calibration. These metrics let users trust the confidence estimates.

**Implementation:**

```python
# molax/metrics/calibration.py
from molax.metrics import (
    expected_calibration_error,
    negative_log_likelihood,
    compute_calibration_curve,
    sharpness,
    evaluate_calibration,
    TemperatureScaling,
    plot_reliability_diagram,
    plot_calibration_comparison,
    create_calibration_report,
)

# Compute ECE
ece = expected_calibration_error(predictions, uncertainties, targets, n_bins=10)

# Compute NLL (proper scoring rule)
nll = negative_log_likelihood(mean, var, targets)

# Comprehensive evaluation
metrics = evaluate_calibration(mean, var, targets)
# Returns: {'nll': ..., 'ece': ..., 'rmse': ..., 'sharpness': ..., 'mean_z_score': ...}

# Temperature scaling for post-hoc calibration
scaler = TemperatureScaling()
scaler.fit(val_mean, val_var, val_targets)
calibrated_var = scaler.transform(test_var)
print(f"Learned temperature: {scaler.temperature}")

# Visualization
plot_reliability_diagram(predictions, uncertainties, targets)
fig = plot_calibration_comparison({
    "Model A": (preds_a, var_a, targets),
    "Model B": (preds_b, var_b, targets),
})
```

**Acceptance Criteria:**
- [x] ECE computation (Expected Calibration Error)
- [x] Reliability diagram plotting utility
- [x] NLL as proper scoring rule
- [x] Temperature scaling for post-hoc calibration
- [x] Integration into evaluation pipeline

---

## Phase 2: Advanced Acquisition Strategies

Better acquisition functions select more informative samples, improving data efficiency.

### 2.1 BALD (Bayesian Active Learning by Disagreement)

**What:** Maximize mutual information between predictions and model parameters.

**Why:** Theoretically principled; targets samples that maximally reduce model uncertainty.

**Implementation:**

```python
# molax/acquisition/bald.py
import jax.numpy as jnp

def bald_acquisition(
    model,
    graphs,
    n_mc_samples: int = 20,
    rngs: jnp.ndarray = None
) -> jnp.ndarray:
    """
    BALD = H[y|x, D] - E_{theta}[H[y|x, theta]]
    = Total uncertainty - Expected aleatoric uncertainty
    """
    # Collect MC samples
    mc_means = []
    mc_vars = []
    for i in range(n_mc_samples):
        mean, var = model(graphs, training=True)  # Dropout active
        mc_means.append(mean)
        mc_vars.append(var)

    mc_means = jnp.stack(mc_means)  # (n_mc, n_samples)
    mc_vars = jnp.stack(mc_vars)

    # Total uncertainty (entropy of predictive distribution)
    predictive_mean = jnp.mean(mc_means, axis=0)
    predictive_var = jnp.var(mc_means, axis=0) + jnp.mean(mc_vars, axis=0)
    total_entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * predictive_var)

    # Expected aleatoric uncertainty
    expected_entropy = 0.5 * jnp.mean(jnp.log(2 * jnp.pi * jnp.e * mc_vars), axis=0)

    # BALD score = mutual information
    return total_entropy - expected_entropy
```

**Acceptance Criteria:**
- [ ] `bald_acquisition` function
- [ ] Efficient batched MC sampling
- [ ] Comparison benchmark vs uncertainty sampling

---

### 2.2 Core-Set Selection

**What:** Select samples that maximize coverage of the feature space using K-center algorithm.

**Why:** Ensures diversity in learned representations, not just input space.

**Implementation:**

```python
# molax/acquisition/coreset.py
import jax.numpy as jnp

def extract_embeddings(model, graphs) -> jnp.ndarray:
    """Get penultimate layer representations."""
    # Add embedding extraction hook to model
    pass

def k_center_greedy(
    embeddings: jnp.ndarray,
    labeled_mask: jnp.ndarray,
    n_select: int
) -> jnp.ndarray:
    """
    Greedy K-center: iteratively select point furthest from labeled set.
    """
    n_samples = embeddings.shape[0]
    selected = jnp.where(labeled_mask)[0]

    # Compute pairwise distances once
    distances = jnp.linalg.norm(
        embeddings[:, None] - embeddings[None, :], axis=-1
    )

    for _ in range(n_select):
        # Distance from each point to nearest labeled point
        min_dist_to_labeled = jnp.min(distances[:, selected], axis=1)
        min_dist_to_labeled = jnp.where(labeled_mask, -jnp.inf, min_dist_to_labeled)

        # Select furthest point
        new_idx = jnp.argmax(min_dist_to_labeled)
        selected = jnp.append(selected, new_idx)
        labeled_mask = labeled_mask.at[new_idx].set(True)

    return selected[-n_select:]
```

**Acceptance Criteria:**
- [ ] Embedding extraction from any model layer
- [ ] K-center greedy implementation
- [ ] GPU-accelerated distance computations

---

### 2.3 Batch-Aware Acquisition

**What:** When selecting K samples, account for redundancy between them.

**Why:** Naive top-K selection often picks near-duplicates; batch-aware methods improve diversity.

**Implementation:**

```python
# molax/acquisition/batch.py

def batch_bald(
    model, graphs, n_select: int, n_mc_samples: int = 20
) -> jnp.ndarray:
    """
    BatchBALD: Select batch jointly to maximize mutual information.
    Approximated via greedy selection with joint entropy tracking.
    """
    pass

def determinantal_point_process(
    scores: jnp.ndarray,
    similarity_matrix: jnp.ndarray,
    n_select: int
) -> jnp.ndarray:
    """
    DPP sampling: balance high scores with diversity.
    Uses fast greedy MAP inference.
    """
    pass
```

**Acceptance Criteria:**
- [ ] BatchBALD implementation
- [ ] DPP-based diverse selection
- [ ] Configurable diversity-quality tradeoff

---

### 2.4 Expected Model Change

**What:** Select samples that would maximally change model predictions if labeled.

**Why:** Directly targets samples that affect the model most, regardless of current uncertainty.

**Implementation:**

```python
# molax/acquisition/emc.py

def expected_gradient_length(model, graphs, labels_placeholder):
    """
    EGL: Use gradient magnitude as proxy for influence.
    """
    def loss_fn(model, x, y):
        mean, _ = model(x, training=False)
        return jnp.mean((mean - y)**2)

    # Compute gradient for hypothetical labels (use predicted mean)
    predicted_mean, _ = model(graphs, training=False)
    grads = jax.grad(loss_fn)(model, graphs, predicted_mean)

    # Gradient magnitude per sample
    return jnp.linalg.norm(grads, axis=-1)
```

**Acceptance Criteria:**
- [ ] Expected Gradient Length implementation
- [ ] Fisher Information-based variant
- [ ] Efficient gradient computation

---

## Phase 3: Architecture Diversity

Multiple architectures capture different inductive biases about molecular structure.

### 3.1 Message Passing Neural Network (MPNN) ✅

**Status:** Implemented in `molax/models/mpnn.py`

**What:** Generalized framework with explicit edge feature processing.

**Why:** Enables richer molecular representations using bond features.

**Implementation:**

```python
# molax/models/mpnn.py
from molax.models.mpnn import MPNNConfig, UncertaintyMPNN

config = MPNNConfig(
    node_features=6,
    edge_features=1,  # Bond type feature
    hidden_features=[64, 64],
    out_features=1,
    aggregation="sum",  # or "mean", "max"
    dropout_rate=0.1,
)
model = UncertaintyMPNN(config, rngs=nnx.Rngs(0))

# Same API as UncertaintyGCN
mean, variance = model(batched_graphs, training=False)

# Extract embeddings for Core-Set selection
embeddings = model.extract_embeddings(batched_graphs)
```

**Acceptance Criteria:**
- [x] MPNN with edge feature support
- [x] Configurable aggregation (sum, mean, max)
- [x] Same API as UncertaintyGCN for acquisition function compatibility
- [x] MC Dropout uncertainty via `get_mpnn_uncertainties()`

---

### 3.2 Graph Attention Network (GAT) ✅

**Status:** Implemented in `molax/models/gat.py`

**What:** Learn edge importance dynamically via attention mechanism.

**Why:** Adaptively weights neighbor contributions based on learned relevance.

**Implementation:**

```python
# molax/models/gat.py
from molax.models.gat import GATConfig, UncertaintyGAT

config = GATConfig(
    node_features=6,
    edge_features=1,  # Optional: include edge features in attention
    hidden_features=[64, 64],
    out_features=1,
    n_heads=4,
    dropout_rate=0.1,
    attention_dropout_rate=0.1,
    negative_slope=0.2,
)
model = UncertaintyGAT(config, rngs=nnx.Rngs(0))

# Same API as UncertaintyGCN/UncertaintyMPNN
mean, variance = model(batched_graphs, training=False)

# Extract embeddings for Core-Set selection
embeddings = model.extract_embeddings(batched_graphs)
```

**Acceptance Criteria:**
- [x] Multi-head attention implementation
- [x] Edge feature incorporation option
- [x] Dropout on attention weights
- [x] Same API as UncertaintyGCN/UncertaintyMPNN for acquisition function compatibility

---

### 3.3 SchNet (Continuous-Filter Convolutions)

**What:** Distance-based convolutions using RBF expansions (adapted for 2D without coordinates).

**Why:** Smooth distance-aware aggregation; useful when edge weights encode bond lengths.

**Note:** For 2D graphs, use topological distance (shortest path) or bond order as edge weights.

---

### 3.4 Graph Transformer ✅

**Status:** Implemented in `molax/models/graph_transformer.py`

**What:** Full self-attention over molecular graphs with positional encodings.

**Why:** State-of-the-art performance; captures long-range dependencies.

**Implementation:**

```python
# molax/models/graph_transformer.py
from molax.models.graph_transformer import GraphTransformerConfig, UncertaintyGraphTransformer

config = GraphTransformerConfig(
    node_features=6,
    edge_features=1,  # Optional: include edge features as attention bias
    hidden_features=[64, 64],
    out_features=1,
    n_heads=4,
    ffn_ratio=4.0,  # FFN hidden dim = 4 * model dim
    dropout_rate=0.1,
    attention_dropout_rate=0.1,
    pe_type="rwpe",  # Random Walk PE (or "laplacian", "none")
    pe_dim=16,
)
model = UncertaintyGraphTransformer(config, rngs=nnx.Rngs(0))

# Same API as UncertaintyGCN/UncertaintyMPNN/UncertaintyGAT
mean, variance = model(batched_graphs, training=False)

# Extract embeddings for Core-Set selection
embeddings = model.extract_embeddings(batched_graphs)
```

**Acceptance Criteria:**
- [x] Graph-aware attention masking
- [x] Positional encodings (Laplacian eigenvectors, random walk)
- [x] Configurable depth and width
- [x] Same API as UncertaintyGCN/UncertaintyMPNN/UncertaintyGAT for acquisition function compatibility

---

## Phase 4: Rich Molecular Featurization

Better input features directly improve model capacity.

### 4.1 Extended Node Features

**Current:** 6 features (atomic num, degree, charge, chirality, hybridization, aromaticity)

**Proposed:** 20+ features including:

```python
# molax/utils/featurizers.py

def extended_atom_features(atom) -> list:
    """Comprehensive RDKit atom features."""
    return [
        # Current features
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetChiralTag()),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),

        # Ring features
        atom.IsInRing(),
        atom.IsInRingSize(3),
        atom.IsInRingSize(4),
        atom.IsInRingSize(5),
        atom.IsInRingSize(6),

        # Electronic features
        atom.GetNumRadicalElectrons(),
        atom.GetNumImplicitHs(),
        atom.GetNumExplicitHs(),

        # Neighborhood
        atom.GetTotalNumHs(),
        atom.GetTotalDegree(),

        # Pharmacophore-related
        is_hydrogen_donor(atom),
        is_hydrogen_acceptor(atom),

        # Electronegativity (from table)
        ELECTRONEGATIVITY.get(atom.GetAtomicNum(), 0),

        # Atomic mass
        atom.GetMass(),
    ]
```

**Acceptance Criteria:**
- [ ] Configurable feature sets (minimal, standard, extended)
- [ ] Feature normalization utilities
- [ ] Documentation of each feature

---

### 4.2 Edge Feature Support

**What:** Include bond features in message passing.

```python
def bond_features(bond) -> list:
    """RDKit bond features."""
    return [
        int(bond.GetBondType()),  # Single, double, triple, aromatic
        bond.GetIsConjugated(),
        bond.IsInRing(),
        int(bond.GetStereo()),  # Stereochemistry
    ]
```

**Acceptance Criteria:**
- [ ] Edge features in `smiles_to_jraph`
- [ ] Models that consume edge features (MPNN, GAT)

---

### 4.3 Pre-trained Embedding Integration

**What:** Use embeddings from pre-trained molecular language models.

**Why:** Transfer learning from large-scale pre-training.

```python
# molax/utils/pretrained.py

def load_chemberta_embeddings(smiles_list: list[str]) -> jnp.ndarray:
    """Extract ChemBERTa [CLS] token embeddings."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    inputs = tokenizer(smiles_list, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    return jnp.array(outputs.last_hidden_state[:, 0, :])  # [CLS] token
```

**Acceptance Criteria:**
- [ ] ChemBERTa integration
- [ ] Caching for efficiency
- [ ] Option to use as initial node features or auxiliary input

---

## Phase 5: Production Readiness

Features required for real-world deployment.

### 5.1 Model Checkpointing

**What:** Save and load model state.

```python
# molax/utils/checkpointing.py
import orbax.checkpoint as ocp
from flax import nnx

def save_model(model, optimizer, path: str, step: int):
    """Save model and optimizer state."""
    state = nnx.state((model, optimizer))
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(f"{path}/step_{step}", state)

def load_model(model, optimizer, path: str):
    """Restore model and optimizer state."""
    checkpointer = ocp.PyTreeCheckpointer()
    state = checkpointer.restore(path)
    nnx.update((model, optimizer), state)
```

**Acceptance Criteria:**
- [ ] Orbax-based save/load
- [ ] Checkpoint management (keep last N)
- [ ] Resume training from checkpoint

---

### 5.2 Multi-Dataset Support

**What:** Support MoleculeNet benchmarks.

```python
# molax/datasets/moleculenet.py

MOLECULENET_DATASETS = {
    'esol': {'task': 'regression', 'n_tasks': 1, 'metric': 'rmse'},
    'freesolv': {'task': 'regression', 'n_tasks': 1, 'metric': 'rmse'},
    'lipophilicity': {'task': 'regression', 'n_tasks': 1, 'metric': 'rmse'},
    'bbbp': {'task': 'classification', 'n_tasks': 1, 'metric': 'auroc'},
    'tox21': {'task': 'classification', 'n_tasks': 12, 'metric': 'auroc'},
    'sider': {'task': 'classification', 'n_tasks': 27, 'metric': 'auroc'},
    'clintox': {'task': 'classification', 'n_tasks': 2, 'metric': 'auroc'},
    'muv': {'task': 'classification', 'n_tasks': 17, 'metric': 'prc-auc'},
    'hiv': {'task': 'classification', 'n_tasks': 1, 'metric': 'auroc'},
    'bace': {'task': 'classification', 'n_tasks': 1, 'metric': 'auroc'},
}

def load_moleculenet(name: str) -> MolecularDataset:
    """Load a MoleculeNet dataset."""
    pass
```

**Acceptance Criteria:**
- [ ] All MoleculeNet datasets supported
- [ ] Scaffold and random split utilities
- [ ] Classification task support

---

### 5.3 Hyperparameter Optimization

**What:** Automated hyperparameter tuning.

```python
# molax/tuning/optuna_search.py
import optuna

def create_objective(dataset, n_epochs):
    def objective(trial):
        config = GCNConfig(
            hidden_features=[
                trial.suggest_int('hidden_dim', 32, 256),
            ] * trial.suggest_int('n_layers', 1, 4),
            dropout_rate=trial.suggest_float('dropout', 0.0, 0.5),
        )
        # Train and evaluate
        model = UncertaintyGCN(config, rngs=nnx.Rngs(0))
        # ... training loop
        return val_loss
    return objective

def run_hpo(dataset, n_trials=100):
    study = optuna.create_study(direction='minimize')
    study.optimize(create_objective(dataset, n_epochs=50), n_trials=n_trials)
    return study.best_params
```

**Acceptance Criteria:**
- [ ] Optuna integration
- [ ] Search space definitions for each model
- [ ] Pruning for early stopping

---

### 5.4 Experiment Tracking

**What:** Log metrics, hyperparameters, and artifacts.

```python
# molax/tracking/wandb_logger.py
import wandb

class WandbLogger:
    def __init__(self, project: str, config: dict):
        wandb.init(project=project, config=config)

    def log(self, metrics: dict, step: int):
        wandb.log(metrics, step=step)

    def log_model(self, model_path: str):
        wandb.save(model_path)
```

**Acceptance Criteria:**
- [ ] W&B integration
- [ ] MLflow as alternative
- [ ] Automatic logging in training loop

---

## Phase 6: Advanced ML Research

Features for ML researchers pushing the boundaries.

### 6.1 Multi-Task Learning

**What:** Predict multiple properties with shared GNN backbone.

```python
class MultiTaskGCN(nnx.Module):
    def __init__(self, config, n_tasks, rngs):
        self.backbone = GCNBackbone(config, rngs)
        self.heads = [
            UncertaintyHead(config.hidden_features[-1], rngs)
            for _ in range(n_tasks)
        ]

    def __call__(self, graphs, training=False):
        embeddings = self.backbone(graphs, training)
        return [head(embeddings) for head in self.heads]
```

**Acceptance Criteria:**
- [ ] Multi-head architecture
- [ ] Task weighting strategies
- [ ] Uncertainty per task

---

### 6.2 Transfer Learning

**What:** Pre-train on large dataset, fine-tune on small target dataset.

**Acceptance Criteria:**
- [ ] Backbone freezing/unfreezing
- [ ] Learning rate schedules for fine-tuning
- [ ] Pre-trained weights for common backbones

---

### 6.3 Semi-Supervised Learning

**What:** Leverage unlabeled molecules to improve representations.

```python
def consistency_loss(model, unlabeled_graphs, rngs):
    """Encourage consistent predictions under augmentation."""
    # Two forward passes with different dropout
    pred1, _ = model(unlabeled_graphs, training=True)
    pred2, _ = model(unlabeled_graphs, training=True)
    return jnp.mean((pred1 - pred2)**2)
```

**Acceptance Criteria:**
- [ ] Consistency regularization
- [ ] Pseudo-labeling
- [ ] Graph augmentation utilities

---

### 6.4 Meta-Learning (MAML)

**What:** Learn to adapt quickly to new molecular tasks with few examples.

**Why:** Critical for low-data drug discovery scenarios.

```python
def maml_inner_loop(model, support_graphs, support_labels, inner_lr, n_steps):
    """Adapt model to support set."""
    adapted_model = model.clone()  # Create copy

    for _ in range(n_steps):
        loss, grads = nnx.value_and_grad(loss_fn)(adapted_model, support_graphs, support_labels)
        # Manual gradient descent
        adapted_model = jax.tree_map(
            lambda p, g: p - inner_lr * g,
            adapted_model, grads
        )

    return adapted_model
```

**Acceptance Criteria:**
- [ ] MAML implementation for few-shot property prediction
- [ ] Task distribution utilities
- [ ] First-order approximation option

---

## Implementation Priority

| Phase | Timeline | Impact | Effort |
|-------|----------|--------|--------|
| 1. Uncertainty Excellence | High | High | Medium |
| 2. Advanced Acquisition | High | High | Medium |
| 3. Architecture Diversity | Medium | Medium | High |
| 4. Rich Featurization | Medium | Medium | Low |
| 5. Production Readiness | Medium | High | Medium |
| 6. Advanced ML | Low | Medium | High |

---

## Contributing

We welcome contributions! Priority areas:
1. **Uncertainty methods** - Ensembles, evidential learning
2. **Calibration tools** - Metrics and visualization
3. **Acquisition functions** - BALD, batch-aware selection

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
