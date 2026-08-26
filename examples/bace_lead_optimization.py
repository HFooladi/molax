"""BACE-1 lead optimization: does active learning actually pay off?

Simulates Design-Make-Test-Analyze rounds against BACE-1 (beta-secretase 1, an
Alzheimer's target). A virtual project starts with a handful of assayed
compounds and, each round, picks which compounds to assay next under a fixed
budget. The question is whether a model-guided choice beats picking at random.

Two things this example does differently from a textbook AL benchmark, both of
which matter for the conclusion:

1.  It reports hit enrichment, not just RMSE. A lead optimization campaign is
    trying to *find potent compounds*, not to build a globally accurate model.
    These are different objectives and they do not agree here -- see the
    summary printed at the end.

2.  It fixes two protocol details that otherwise flatter the results:
    -  the model is re-initialized every round, so the learning curve reflects
       acquired data rather than accumulated gradient steps;
    -  label standardization is fitted on the labeled set only and refitted
       each round. Using pool statistics would leak the answer into the
       experiment, since pool labels are exactly what the campaign has not
       measured yet.

The split is by Bemis-Murcko scaffold, so the test set measures generalization
to chemotypes the model has never seen -- the situation a real project faces.

Usage:
    python scripts/download_bace.py    # once
    python examples/bace_lead_optimization.py
"""

import time
from collections import defaultdict
from pathlib import Path

import flax.nnx as nnx
import jax.numpy as jnp
import jraph
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from molax.acquisition import coreset_from_embeddings
from molax.metrics import evaluate_calibration
from molax.models.gcn import GCNConfig, UncertaintyGCN
from molax.utils.data import smiles_to_jraph
from molax.utils.featurizers import ATOM_FEATURIZERS

RDLogger.DisableLog("rdApp.*")

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "bace.csv"
OUTPUT_PATH = Path(__file__).parent / "assets" / "bace_lead_optimization.png"
# mkdocs only serves files under docs/, so the plot is mirrored there to keep
# the case study page from going stale against the example that generates it.
DOCS_IMAGE_PATH = (
    Path(__file__).parent.parent / "docs" / "assets" / "bace_lead_optimization.png"
)

FEATURES = "rich"
N_TEST_SCAFFOLD_MOLECULES = 300  # held-out chemotypes
N_INITIAL = 60  # compounds assayed before round 1
N_PER_ROUND = 60  # assay budget per round
N_ROUNDS = 7
N_EPOCHS = 300
LEARNING_RATE = 1e-3
HIDDEN = [64, 64]
DROPOUT = 0.1
TOP_K = 100  # "hits" = the K most potent compounds in the pool
UCB_BETA = 1.0
UNCERTAINTY_WEIGHT = 0.7  # uncertainty vs diversity in 'combined'
SEEDS = [0, 1, 2]

STRATEGIES = ["random", "uncertainty", "greedy", "ucb", "combined"]
STRATEGY_LABELS = {
    "random": "Random (baseline)",
    "uncertainty": "Uncertainty",
    "greedy": "Greedy (exploit)",
    "ucb": "UCB (explore+exploit)",
    "combined": "Uncertainty + diversity",
}
STRATEGY_COLORS = {
    "random": "gray",
    "uncertainty": "tab:blue",
    "greedy": "tab:red",
    "ucb": "tab:purple",
    "combined": "tab:green",
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def scaffold_split(smiles_list, n_test):
    """Split by Bemis-Murcko scaffold, largest scaffold groups first.

    Whole scaffold groups go to one side or the other, so no chemotype appears
    in both train and test. Test molecules therefore come from scaffolds the
    model has never seen -- a much harder and more honest evaluation than a
    random split on an SAR series, where near-identical analogs would otherwise
    straddle the split.
    """
    groups = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        groups[scaffold].append(idx)

    # Largest scaffold families fill the pool first; what is left over -- the
    # rare and singleton chemotypes -- becomes the test set. This is the
    # standard (and harder) direction: the campaign explores established analog
    # series while the model is scored on chemistry it has never seen.
    ordered = sorted(groups.values(), key=len, reverse=True)
    n_train_target = len(smiles_list) - n_test

    train_idx: list[int] = []
    test_idx: list[int] = []
    for group in ordered:
        if len(train_idx) < n_train_target:
            train_idx.extend(group)
        else:
            test_idx.extend(group)

    return np.array(train_idx), np.array(test_idx), len(groups)


def load_data():
    if not DATASET_PATH.exists():
        raise SystemExit(
            f"{DATASET_PATH} not found. Run: python scripts/download_bace.py"
        )

    df = pd.read_csv(DATASET_PATH)

    graphs, labels, smiles = [], [], []
    for smi, y in zip(df["smiles"], df["property"]):
        try:
            graphs.append(smiles_to_jraph(smi, features=FEATURES))
            labels.append(float(y))
            smiles.append(smi)
        except ValueError:
            continue

    return graphs, np.array(labels, dtype=np.float32), smiles


# --------------------------------------------------------------------------
# Active learning
# --------------------------------------------------------------------------


@nnx.jit
def train_step(model, optimizer, graphs, y_scaled, mask):
    """One gradient step of masked Gaussian NLL.

    Defined once at module scope: every campaign batches its pool to the same
    shapes, so this compiles a single time for the whole study rather than once
    per (strategy, seed, round).
    """

    def loss_fn(model):
        mean, var = model(graphs, training=True)
        mean, var = mean.squeeze(-1), var.squeeze(-1)
        nll = 0.5 * (jnp.log(var + 1e-6) + (y_scaled - mean) ** 2 / (var + 1e-6))
        return jnp.sum(jnp.where(mask, nll, 0.0)) / (jnp.sum(mask) + 1e-6)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def run_campaign(strategy, seed, pool_graphs, pool_labels, test_graphs, test_labels):
    """Run one simulated campaign and record per-round metrics."""
    n_pool = len(pool_labels)
    n_features = ATOM_FEATURIZERS[FEATURES].dim

    # Batch once; acquisition is expressed as a boolean mask over this fixed
    # batch, so shapes never change and nothing recompiles (see CLAUDE.md).
    batched_pool = jraph.batch(pool_graphs)
    batched_test = jraph.batch(test_graphs)
    test_labels_j = jnp.asarray(test_labels)

    # The K most potent compounds in the pool. The campaign does not know these;
    # they are the ground truth we score hit-finding against.
    true_top_k = set(np.argsort(-pool_labels)[:TOP_K].tolist())

    rng = np.random.default_rng(seed)
    labeled = np.zeros(n_pool, dtype=bool)
    labeled[rng.permutation(n_pool)[:N_INITIAL]] = True

    rows = []
    for round_idx in range(N_ROUNDS):
        mask = jnp.asarray(labeled)

        # Standardize using ONLY assayed labels - pool labels are unmeasured.
        y_mean = float(pool_labels[labeled].mean())
        y_std = float(pool_labels[labeled].std()) or 1.0
        y_scaled = jnp.asarray((pool_labels - y_mean) / y_std)

        # Fresh model each round
        model = UncertaintyGCN(
            GCNConfig(n_features, HIDDEN, 1, DROPOUT),
            nnx.Rngs(seed * 1000 + round_idx),
        )
        optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)

        for _ in range(N_EPOCHS):
            train_step(model, optimizer, batched_pool, y_scaled, mask)

        # --- evaluate on held-out scaffolds (in original pIC50 units) ---
        test_mean, test_var = model(batched_test, training=False)
        test_pred = test_mean.squeeze(-1) * y_std + y_mean
        test_var_orig = test_var.squeeze(-1) * (y_std**2)
        rmse = float(jnp.sqrt(jnp.mean((test_pred - test_labels_j) ** 2)))
        ece = float(
            evaluate_calibration(test_pred, test_var_orig, test_labels_j)["ece"]
        )

        n_assayed = int(labeled.sum())
        hits = len(true_top_k & set(np.where(labeled)[0].tolist()))
        rows.append(
            {
                "strategy": strategy,
                "seed": seed,
                "assayed": n_assayed,
                "rmse": rmse,
                "ece": ece,
                "hits": hits,
                "best_pic50": float(pool_labels[labeled].max()),
            }
        )

        if round_idx == N_ROUNDS - 1:
            break

        # --- acquire the next batch ---
        pool_mask = jnp.asarray(~labeled)
        n_select = min(N_PER_ROUND, int((~labeled).sum()))
        selected = select(
            strategy, model, batched_pool, pool_mask, n_select, rng, n_pool
        )
        labeled[np.asarray(selected)] = True

    return rows


def select(strategy, model, batched_pool, pool_mask, n_select, rng, n_pool):
    """Choose which compounds to assay next."""
    if strategy == "random":
        candidates = np.where(np.asarray(pool_mask))[0]
        return rng.choice(candidates, size=n_select, replace=False)

    mean, var = model(batched_pool, training=False)
    mean, var = mean.squeeze(-1), var.squeeze(-1)

    if strategy == "uncertainty":
        # Predicted variance, not MC dropout: on this codebase MC-dropout
        # variance is uncorrelated with actual error, while the learned
        # variance head carries real signal.
        score = var
    elif strategy == "greedy":
        score = mean  # pure exploitation - assay what looks most potent
    elif strategy == "ucb":
        score = mean + UCB_BETA * jnp.sqrt(var)
    elif strategy == "combined":
        unc = var / (jnp.max(var) + 1e-12)
        embeddings = model.extract_embeddings(batched_pool, training=False)
        diverse = coreset_from_embeddings(embeddings, pool_mask, n_select)
        div = jnp.zeros(n_pool)
        if diverse:
            div = div.at[jnp.array(diverse)].set(jnp.linspace(1.0, 0.0, len(diverse)))
        score = UNCERTAINTY_WEIGHT * unc + (1 - UNCERTAINTY_WEIGHT) * div
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return np.asarray(jnp.argsort(-jnp.where(pool_mask, score, -jnp.inf))[:n_select])


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def plot(df):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    panels = [
        ("hits", f"Top-{TOP_K} potent compounds found", "Hit discovery (the goal)"),
        ("rmse", "Test RMSE (pIC50)", "Global accuracy (not the goal)"),
        ("ece", "Expected calibration error", "Uncertainty calibration"),
    ]

    for ax, (column, ylabel, title) in zip(axes, panels):
        for strategy in STRATEGIES:
            sub = df[df.strategy == strategy].groupby("assayed")[column]
            x = np.array(sorted(sub.groups))
            mean = sub.mean().values
            std = sub.std().fillna(0).values
            ax.plot(
                x,
                mean,
                "-o",
                color=STRATEGY_COLORS[strategy],
                label=STRATEGY_LABELS[strategy],
                linewidth=2,
                markersize=4,
            )
            ax.fill_between(
                x, mean - std, mean + std, color=STRATEGY_COLORS[strategy], alpha=0.15
            )
        ax.set_xlabel("Compounds assayed")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(
        "BACE-1 lead optimization: active learning under an assay budget "
        f"({len(SEEDS)} seeds, scaffold split)",
        fontsize=13,
    )
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nPlot saved to {OUTPUT_PATH}")

    DOCS_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DOCS_IMAGE_PATH, dpi=150)
    print(f"Docs copy saved to {DOCS_IMAGE_PATH}")


def summarize(df):
    final = df[df.assayed == df.assayed.max()]
    budget = int(final.assayed.iloc[0])

    baseline = final[final.strategy == "random"].hits.mean()

    print("\n" + "=" * 74)
    print(f"Results after {budget} assayed compounds ({len(SEEDS)} seeds)")
    print("=" * 74)
    print(
        f"{'strategy':24s} {'top-' + str(TOP_K) + ' found':>14s} "
        f"{'enrichment':>11s} {'RMSE':>13s} {'best pIC50':>11s}"
    )
    print("-" * 74)
    for strategy in STRATEGIES:
        sub = final[final.strategy == strategy]
        hits, hits_sd = sub.hits.mean(), sub.hits.std()
        rmse, rmse_sd = sub.rmse.mean(), sub.rmse.std()
        enrich = hits / baseline if baseline else float("nan")
        print(
            f"{STRATEGY_LABELS[strategy]:24s} {hits:6.1f} +/- {hits_sd:4.1f} "
            f"{enrich:10.2f}x {rmse:6.3f} +/- {rmse_sd:4.3f} "
            f"{sub.best_pic50.mean():10.2f}"
        )

    best_hits = max(STRATEGIES, key=lambda s: final[final.strategy == s].hits.mean())
    best_rmse = min(STRATEGIES, key=lambda s: final[final.strategy == s].rmse.mean())

    print("-" * 74)
    print(f"Best at finding potent compounds : {STRATEGY_LABELS[best_hits]}")
    print(f"Best at global accuracy (RMSE)   : {STRATEGY_LABELS[best_rmse]}")
    print()

    if best_hits == "random":
        print(
            "No strategy beat random at hit discovery. The acquisition signal is\n"
            "not informative on this task -- treat the loop as not yet working."
        )
    else:
        gain = final[final.strategy == best_hits].hits.mean() / baseline
        print(
            f"{STRATEGY_LABELS[best_hits]} found {gain:.2f}x more of the top-{TOP_K}\n"
            f"potent compounds than random for the same assay budget."
        )

    if best_hits == best_rmse:
        print("The same strategy also gave the best global accuracy.")
    else:
        hits_rmse = final[final.strategy == best_hits].rmse.mean()
        rmse_hits = final[final.strategy == best_rmse].hits.mean()
        print(
            f"\nBut a different strategy wins on RMSE: {STRATEGY_LABELS[best_rmse]}\n"
            f"({final[final.strategy == best_rmse].rmse.mean():.3f} vs "
            f"{hits_rmse:.3f}), and it finds only {rmse_hits:.1f} hits vs "
            f"{final[final.strategy == best_hits].hits.mean():.1f}.\n"
            "\n"
            "This is the exploration/exploitation split, and it is the point of\n"
            "this case study. Uncertainty-driven acquisition deliberately assays\n"
            "compounds the model finds confusing, which sharpens the model but\n"
            "spends budget on compounds that are usually not potent. Exploitative\n"
            "acquisition does the opposite. A lead optimization campaign is scored\n"
            "on hits found, so ranking strategies by RMSE would pick the wrong one."
        )
    print("=" * 74)

    csv_path = OUTPUT_PATH.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"Per-round results written to {csv_path}")


def main():
    print("=" * 74)
    print("BACE-1 Lead Optimization Case Study")
    print("=" * 74)

    graphs, labels, smiles = load_data()
    print(
        f"Loaded {len(graphs)} BACE-1 inhibitors ({FEATURES} features, "
        f"{ATOM_FEATURIZERS[FEATURES].dim} dims/atom)"
    )
    print(
        f"pIC50: mean {labels.mean():.2f}, std {labels.std():.2f}, "
        f"range [{labels.min():.2f}, {labels.max():.2f}]"
    )

    train_idx, test_idx, n_scaffolds = scaffold_split(smiles, N_TEST_SCAFFOLD_MOLECULES)
    print(
        f"Scaffold split: {n_scaffolds} Bemis-Murcko scaffolds -> "
        f"{len(train_idx)} pool / {len(test_idx)} held-out"
    )

    pool_graphs = [graphs[i] for i in train_idx]
    pool_labels = labels[train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    test_labels = labels[test_idx]

    print(
        f"\nCampaign: {N_INITIAL} initial + {N_ROUNDS - 1} rounds x "
        f"{N_PER_ROUND} assays, {N_EPOCHS} epochs/round, fresh model each round"
    )
    print(
        f"Hits = the {TOP_K} most potent compounds in the {len(pool_labels)}-"
        f"compound pool\n"
    )

    records = []
    start = time.time()
    for strategy in STRATEGIES:
        strat_start = time.time()
        for seed in SEEDS:
            records.extend(
                run_campaign(
                    strategy,
                    seed,
                    pool_graphs,
                    pool_labels,
                    test_graphs,
                    test_labels,
                )
            )
        done = [
            r
            for r in records
            if r["strategy"] == strategy
            and r["assayed"] == max(x["assayed"] for x in records)
        ]
        print(
            f"  {STRATEGY_LABELS[strategy]:24s} {time.time() - strat_start:6.1f}s  "
            f"hits={np.mean([r['hits'] for r in done]):.1f}"
        )

    print(f"\nTotal: {time.time() - start:.1f}s")

    df = pd.DataFrame(records)
    plot(df)
    summarize(df)


if __name__ == "__main__":
    main()
