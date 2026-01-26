"""Demonstration of Advanced Acquisition Strategies.

This script demonstrates that Phase 2 acquisition functions work as intended:
- BALD: Selects samples with high mutual information
- Core-Set: Selects diverse samples covering the embedding space
- BatchBALD/DPP: Selects diverse batches avoiding redundancy
- EGL: Selects samples that would cause largest model change

Run with: python examples/acquisition_strategies_demo.py
"""

import flax.nnx as nnx
import jax.numpy as jnp

from molax.acquisition import (
    bald_sampling,
    ensemble_bald_sampling,
    evidential_bald_sampling,
    uncertainty_sampling,
)
from molax.acquisition.batch_aware import batch_bald_sampling, dpp_sampling
from molax.acquisition.coreset import coreset_sampling, coreset_sampling_with_scores
from molax.acquisition.expected_change import egl_sampling
from molax.models.ensemble import DeepEnsemble, EnsembleConfig
from molax.models.evidential import EvidentialConfig, EvidentialGCN
from molax.models.gcn import GCNConfig, UncertaintyGCN
from molax.utils.data import batch_graphs, smiles_to_jraph


def load_sample_molecules():
    """Load a diverse set of molecules for demonstration."""
    # Diverse molecules with different properties
    molecules = [
        # Simple alkanes (similar to each other)
        ("C", "methane"),
        ("CC", "ethane"),
        ("CCC", "propane"),
        ("CCCC", "butane"),
        ("CCCCC", "pentane"),
        # Alcohols (similar to each other)
        ("CO", "methanol"),
        ("CCO", "ethanol"),
        ("CCCO", "propanol"),
        # Aromatics (similar to each other)
        ("c1ccccc1", "benzene"),
        ("Cc1ccccc1", "toluene"),
        ("c1ccc(O)cc1", "phenol"),
        # Carbonyls
        ("CC=O", "acetaldehyde"),
        ("CC(=O)C", "acetone"),
        ("CC(=O)O", "acetic_acid"),
        # Amines
        ("CN", "methylamine"),
        ("CCN", "ethylamine"),
        # Ethers
        ("COC", "dimethyl_ether"),
        ("CCOC", "ethyl_methyl_ether"),
        # More complex
        ("c1ccc2ccccc2c1", "naphthalene"),
        ("CC(C)C", "isobutane"),
    ]

    graphs = []
    names = []
    for smiles, name in molecules:
        try:
            g = smiles_to_jraph(smiles)
            graphs.append(g)
            names.append(name)
        except Exception as e:
            print(f"Skipping {name}: {e}")

    return graphs, names


def demo_bald_vs_uncertainty():
    """Demonstrate BALD vs simple uncertainty sampling.

    BALD should prefer samples where epistemic uncertainty is high relative
    to aleatoric uncertainty (high mutual information).
    """
    print("\n" + "=" * 70)
    print("DEMO 1: BALD vs Uncertainty Sampling")
    print("=" * 70)
    print("\nBALD = I(y; θ | x) measures mutual information between predictions")
    print("and model parameters. It prefers high epistemic/low aleatoric uncertainty.")

    # Create model
    config = GCNConfig(
        node_features=6,
        hidden_features=[32, 32],
        out_features=1,
        dropout_rate=0.2,
    )
    model = UncertaintyGCN(config, rngs=nnx.Rngs(42))

    graphs, names = load_sample_molecules()

    # Get uncertainty and BALD scores
    uncertainty_scores = uncertainty_sampling(model, graphs, n_samples=20)
    bald_scores = bald_sampling(model, graphs, n_mc_samples=20)

    # Also get the underlying uncertainties
    batched = batch_graphs(graphs)
    means_list = []
    vars_list = []
    for _ in range(20):
        mean, var = model(batched, training=True)
        means_list.append(mean.squeeze(-1))
        vars_list.append(var.squeeze(-1))

    means = jnp.stack(means_list, axis=0)
    variances = jnp.stack(vars_list, axis=0)

    epistemic = jnp.var(means, axis=0)[: len(graphs)]
    aleatoric = jnp.mean(variances, axis=0)[: len(graphs)]

    # Create results table
    print("\nResults (higher BALD = higher epistemic relative to aleatoric):")
    print("-" * 75)
    print(f"{'Molecule':<18} {'Uncert':>10} {'BALD':>10} {'Epist':>10} {'Aleat':>10}")
    print("-" * 75)

    # Sort by BALD score
    sorted_indices = jnp.argsort(-bald_scores)
    for i in sorted_indices[:10]:
        print(
            f"{names[i]:<18} {float(uncertainty_scores[i]):>10.4f} "
            f"{float(bald_scores[i]):>10.4f} {float(epistemic[i]):>10.4f} "
            f"{float(aleatoric[i]):>10.4f}"
        )

    # Compare top selections
    top_uncertainty = jnp.argsort(-uncertainty_scores)[:5]
    top_bald = jnp.argsort(-bald_scores)[:5]

    print("\nTop 5 by Uncertainty:", [names[i] for i in top_uncertainty])
    print("Top 5 by BALD:", [names[i] for i in top_bald])

    overlap = len(set(top_uncertainty.tolist()) & set(top_bald.tolist()))
    print(f"\nOverlap: {overlap}/5 molecules")
    print("(Different selections show BALD uses different criteria than uncertainty)")


def demo_coreset_diversity():
    """Demonstrate Core-Set selects diverse samples.

    Core-Set should select samples that are far apart in embedding space,
    providing good coverage of the data distribution.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Core-Set Diversity Selection")
    print("=" * 70)
    print("\nCore-Set uses k-center greedy to select samples that maximize")
    print("minimum distance to already selected/labeled set.")

    config = GCNConfig(
        node_features=6,
        hidden_features=[32, 32],
        out_features=1,
        dropout_rate=0.1,
    )
    model = UncertaintyGCN(config, rngs=nnx.Rngs(42))

    graphs, names = load_sample_molecules()

    # Simulate labeled set (first few alkanes - similar molecules)
    labeled_indices = [0, 1, 2]  # methane, ethane, propane
    labeled_graphs = [graphs[i] for i in labeled_indices]
    pool_graphs = [g for i, g in enumerate(graphs) if i not in labeled_indices]
    pool_names = [n for i, n in enumerate(names) if i not in labeled_indices]

    print(f"\nLabeled set (similar alkanes): {[names[i] for i in labeled_indices]}")
    print(f"Pool size: {len(pool_graphs)} molecules")

    # Get Core-Set selections
    n_select = 5
    selected = coreset_sampling(model, pool_graphs, labeled_graphs, n_select)

    print(f"\nCore-Set selected {n_select} molecules:")
    for idx in selected:
        print(f"  - {pool_names[idx]}")

    # Get diversity scores (min-distance to labeled set)
    scores = coreset_sampling_with_scores(model, pool_graphs, labeled_graphs)

    print("\nDiversity scores (min-distance to labeled set):")
    print("-" * 50)
    sorted_indices = jnp.argsort(-scores)
    for i in sorted_indices[:8]:
        print(f"  {pool_names[i]:<20} distance: {float(scores[i]):.4f}")

    print("\nExpected: Core-Set should select molecules from different classes")
    print("(aromatics, carbonyls, amines, etc.) rather than more alkanes.")


def demo_batch_diversity():
    """Demonstrate BatchBALD and DPP select diverse batches.

    Naive top-k uncertainty could select redundant samples.
    BatchBALD and DPP should select complementary samples.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Batch-Aware Selection (BatchBALD vs DPP)")
    print("=" * 70)
    print("\nNaive top-k might select similar high-uncertainty samples.")
    print("BatchBALD and DPP consider inter-sample diversity.")

    config = GCNConfig(
        node_features=6,
        hidden_features=[32, 32],
        out_features=1,
        dropout_rate=0.2,
    )
    model = UncertaintyGCN(config, rngs=nnx.Rngs(42))

    graphs, names = load_sample_molecules()

    n_select = 5

    # Naive: top-k by uncertainty
    uncertainty_scores = uncertainty_sampling(model, graphs, n_samples=10)
    naive_selected = jnp.argsort(-uncertainty_scores)[:n_select].tolist()

    # BatchBALD
    batchbald_selected = batch_bald_sampling(model, graphs, n_select, n_mc_samples=10)

    # DPP
    dpp_selected = dpp_sampling(model, graphs, n_select)

    print(f"\nNaive top-{n_select} by uncertainty:")
    print(f"  {[names[i] for i in naive_selected]}")

    print(f"\nBatchBALD top-{n_select}:")
    print(f"  {[names[i] for i in batchbald_selected]}")

    print(f"\nDPP top-{n_select}:")
    print(f"  {[names[i] for i in dpp_selected]}")

    # Measure diversity: compute pairwise distances in embedding space
    batched = batch_graphs(graphs)
    embeddings = model.extract_embeddings(batched, training=False)[: len(graphs)]

    def batch_diversity(indices):
        """Compute average pairwise distance in batch."""
        if len(indices) < 2:
            return 0.0
        embs = embeddings[jnp.array(indices)]
        diff = embs[:, None, :] - embs[None, :, :]
        dists = jnp.linalg.norm(diff, axis=2)
        # Average of upper triangle
        n = len(indices)
        return float(jnp.sum(jnp.triu(dists, k=1)) / (n * (n - 1) / 2))

    print("\nBatch diversity (avg pairwise embedding distance):")
    print(f"  Naive:     {batch_diversity(naive_selected):.4f}")
    print(f"  BatchBALD: {batch_diversity(batchbald_selected):.4f}")
    print(f"  DPP:       {batch_diversity(dpp_selected):.4f}")

    print("\nExpected: BatchBALD and DPP should have higher diversity than naive.")


def demo_egl():
    """Demonstrate Expected Gradient Length.

    EGL selects samples that would cause the largest model change if labeled.
    These are typically samples where the model prediction is uncertain AND
    the loss gradient is large.
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Expected Gradient Length (EGL)")
    print("=" * 70)
    print("\nEGL measures how much the model would change if a sample were labeled.")
    print("High EGL = model would learn a lot from this sample.")

    config = GCNConfig(
        node_features=6,
        hidden_features=[32, 32],
        out_features=1,
        dropout_rate=0.1,
    )
    model = UncertaintyGCN(config, rngs=nnx.Rngs(42))

    graphs, names = load_sample_molecules()

    # Only use a subset for speed
    graphs = graphs[:12]
    names = names[:12]

    # Get EGL scores
    print("\nComputing EGL scores (this may take a moment)...")
    egl_scores = egl_sampling(model, graphs)

    # Compare with uncertainty
    uncertainty_scores = uncertainty_sampling(model, graphs, n_samples=10)

    print("\nResults:")
    print("-" * 60)
    print(f"{'Molecule':<20} {'EGL Score':>15} {'Uncertainty':>15}")
    print("-" * 60)

    sorted_indices = jnp.argsort(-egl_scores)
    for i in sorted_indices:
        print(
            f"{names[i]:<20} {float(egl_scores[i]):>15.4f} "
            f"{float(uncertainty_scores[i]):>15.4f}"
        )

    # Correlation
    corr = jnp.corrcoef(egl_scores, uncertainty_scores)[0, 1]
    print(f"\nCorrelation between EGL and Uncertainty: {float(corr):.3f}")
    print("\n(EGL considers gradient magnitude, not just prediction variance)")


def demo_ensemble_vs_mc_dropout():
    """Compare BALD scores from ensemble vs MC Dropout."""
    print("\n" + "=" * 70)
    print("DEMO 5: Ensemble BALD vs MC Dropout BALD")
    print("=" * 70)
    print("\nBoth methods estimate epistemic uncertainty differently:")
    print("- MC Dropout: Samples from approximate posterior via dropout")
    print("- Ensemble: Uses independent model disagreement")

    # MC Dropout model
    mc_config = GCNConfig(
        node_features=6,
        hidden_features=[32, 32],
        out_features=1,
        dropout_rate=0.2,
    )
    mc_model = UncertaintyGCN(mc_config, rngs=nnx.Rngs(42))

    # Ensemble model
    base_config = GCNConfig(
        node_features=6,
        hidden_features=[32, 32],
        out_features=1,
        dropout_rate=0.1,
    )
    ensemble_config = EnsembleConfig(base_config=base_config, n_members=5)
    ensemble = DeepEnsemble(ensemble_config, rngs=nnx.Rngs(42))

    graphs, names = load_sample_molecules()
    graphs = graphs[:15]
    names = names[:15]

    # Get BALD scores
    mc_bald = bald_sampling(mc_model, graphs, n_mc_samples=20)
    ensemble_bald = ensemble_bald_sampling(ensemble, graphs)

    print("\nBALD scores comparison:")
    print("-" * 60)
    print(f"{'Molecule':<20} {'MC Dropout BALD':>18} {'Ensemble BALD':>18}")
    print("-" * 60)

    for i in range(len(names)):
        print(
            f"{names[i]:<20} {float(mc_bald[i]):>18.4f} "
            f"{float(ensemble_bald[i]):>18.4f}"
        )

    # Top-5 comparison
    top_mc = jnp.argsort(-mc_bald)[:5].tolist()
    top_ensemble = jnp.argsort(-ensemble_bald)[:5].tolist()

    print("\nTop 5 by MC Dropout BALD:", [names[i] for i in top_mc])
    print("Top 5 by Ensemble BALD:", [names[i] for i in top_ensemble])

    overlap = len(set(top_mc) & set(top_ensemble))
    print(f"\nOverlap: {overlap}/5 molecules")


def demo_evidential_bald():
    """Demonstrate Evidential BALD (single forward pass)."""
    print("\n" + "=" * 70)
    print("DEMO 6: Evidential BALD (Single Forward Pass)")
    print("=" * 70)
    print("\nEvidential models provide epistemic/aleatoric uncertainty in one pass.")
    print("No need for MC sampling or multiple ensemble members at inference.")

    base_config = GCNConfig(
        node_features=6,
        hidden_features=[32, 32],
        out_features=1,
        dropout_rate=0.1,
    )
    evidential_config = EvidentialConfig(base_config=base_config, lambda_reg=0.1)
    model = EvidentialGCN(evidential_config, rngs=nnx.Rngs(42))

    graphs, names = load_sample_molecules()
    graphs = graphs[:15]
    names = names[:15]

    # Get BALD scores
    bald_scores = evidential_bald_sampling(model, graphs)

    # Get raw uncertainties
    batched = batch_graphs(graphs)
    mean, total_var, epistemic_var = model(batched, training=False)
    aleatoric_var = total_var - epistemic_var

    print("\nEvidential model uncertainties:")
    print("-" * 75)
    print(f"{'Molecule':<18} {'BALD':>10} {'Epist':>10} {'Aleat':>10} {'Total':>10}")
    print("-" * 75)

    sorted_indices = jnp.argsort(-bald_scores)
    for i in sorted_indices:
        print(
            f"{names[i]:<18} {float(bald_scores[i]):>10.4f} "
            f"{float(epistemic_var[i, 0]):>10.4f} "
            f"{float(aleatoric_var[i, 0]):>10.4f} "
            f"{float(total_var[i, 0]):>10.4f}"
        )


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("Advanced Acquisition Strategies - Conceptual Demonstration")
    print("=" * 70)
    print("\nThis script demonstrates that Phase 2 acquisition functions")
    print("work as intended, selecting samples according to their design goals.")

    demo_bald_vs_uncertainty()
    demo_coreset_diversity()
    demo_batch_diversity()
    demo_egl()
    demo_ensemble_vs_mc_dropout()
    demo_evidential_bald()

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("1. BALD selects based on mutual information, not just uncertainty")
    print("2. Core-Set selects diverse samples covering the embedding space")
    print("3. BatchBALD/DPP select complementary batches, avoiding redundancy")
    print("4. EGL identifies samples that would most change the model")
    print("5. Ensemble and Evidential methods provide alternative BALD estimates")


if __name__ == "__main__":
    main()
