"""Tests for atom featurizers and their integration with smiles_to_jraph."""

import jax.numpy as jnp
import pytest
from rdkit import Chem

from molax.utils.data import MolecularDataset, batch_graphs, smiles_to_jraph
from molax.utils.featurizers import (
    ATOM_FEATURIZERS,
    DEFAULT_FEATURIZER,
    get_atom_featurizer,
)

# A spread of chemistry: alcohol, aromatic, drug-like, halogen, charged, ring-fused
SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "ClCCBr",
    "CC(=O)[O-]",
    "C1CC2CCC1CC2",
]


def test_default_is_basic():
    assert DEFAULT_FEATURIZER == "basic"
    assert smiles_to_jraph("CCO").nodes.shape[1] == 6


def test_unknown_featurizer_raises():
    with pytest.raises(ValueError, match="Unknown featurizer"):
        get_atom_featurizer("does-not-exist")
    with pytest.raises(ValueError, match="Unknown featurizer"):
        smiles_to_jraph("CCO", features="does-not-exist")


def test_invalid_smiles_still_raises():
    for features in ATOM_FEATURIZERS:
        with pytest.raises(ValueError, match="Invalid SMILES"):
            smiles_to_jraph("not-a-molecule", features=features)


@pytest.mark.parametrize("features", sorted(ATOM_FEATURIZERS))
def test_declared_dim_matches_output(features):
    """Registry dim must match what the featurizer actually emits."""
    featurizer = get_atom_featurizer(features)
    for smiles in SMILES:
        graph = smiles_to_jraph(smiles, features=features)
        assert graph.nodes.shape[1] == featurizer.dim
        mol = Chem.MolFromSmiles(smiles)
        assert graph.nodes.shape[0] == mol.GetNumAtoms()


@pytest.mark.parametrize("features", sorted(ATOM_FEATURIZERS))
def test_featurizer_is_deterministic(features):
    a = smiles_to_jraph("CC(=O)Oc1ccccc1C(=O)O", features=features)
    b = smiles_to_jraph("CC(=O)Oc1ccccc1C(=O)O", features=features)
    assert jnp.array_equal(a.nodes, b.nodes)


@pytest.mark.parametrize("features", sorted(ATOM_FEATURIZERS))
def test_featurizer_does_not_change_topology(features):
    """Changing features must not change bonds, self-loops or graph sizes."""
    base = smiles_to_jraph("CC(=O)Oc1ccccc1C(=O)O")
    other = smiles_to_jraph("CC(=O)Oc1ccccc1C(=O)O", features=features)
    assert jnp.array_equal(base.senders, other.senders)
    assert jnp.array_equal(base.receivers, other.receivers)
    assert jnp.array_equal(base.n_node, other.n_node)
    assert jnp.array_equal(base.n_edge, other.n_edge)


@pytest.mark.parametrize("features", sorted(ATOM_FEATURIZERS))
def test_batches_cleanly(features):
    graphs = [smiles_to_jraph(s, features=features) for s in SMILES]
    batched = batch_graphs(graphs)
    assert batched.nodes.shape[1] == get_atom_featurizer(features).dim
    assert int(batched.n_node.shape[0]) == len(SMILES) + 1  # + padding graph


def test_basic_features_are_unchanged():
    """Regression guard: the default featurization must not drift.

    These are the exact six raw descriptors molax has always emitted
    (atomic_num, degree, formal_charge, chiral_tag, hybridization, aromatic).
    Downstream users have models trained against them.
    """
    graph = smiles_to_jraph("CCO", features="basic")
    expected = jnp.array(
        [
            [6.0, 1.0, 0.0, 0.0, 4.0, 0.0],  # CH3
            [6.0, 2.0, 0.0, 0.0, 4.0, 0.0],  # CH2
            [8.0, 1.0, 0.0, 0.0, 4.0, 0.0],  # OH
        ],
        dtype=jnp.float32,
    )
    assert jnp.array_equal(graph.nodes, expected)


def test_rich_features_are_one_hot_blocks():
    """The one-hot blocks must be mutually exclusive and complete."""
    graph = smiles_to_jraph("CC(=O)Oc1ccccc1C(=O)O", features="rich")
    nodes = graph.nodes
    # element block (10) and degree block (6) are one-hot over every atom
    assert jnp.all(jnp.sum(nodes[:, 0:10], axis=1) == 1.0)
    assert jnp.all(jnp.sum(nodes[:, 10:16], axis=1) == 1.0)
    # aromatic / in-ring flags are binary
    assert jnp.all((nodes[:, 26] == 0.0) | (nodes[:, 26] == 1.0))
    assert jnp.all((nodes[:, 27] == 0.0) | (nodes[:, 27] == 1.0))


# Rich feature layout, used to name dimensions in the dead-dimension test.
RICH_NAMES = (
    [f"elem_{e}" for e in ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other")]
    + [f"degree_{i}" for i in range(6)]
    + [f"numH_{i}" for i in range(5)]
    + ["hyb_SP", "hyb_SP2", "hyb_SP3", "hyb_SP3D", "hyb_SP3D2"]
    + ["aromatic", "in_ring", "formal_charge"]
)

# One-hot slots for chemistry that simply does not occur in every dataset
# (ESOL has no hypervalent atoms, no degree-5 atoms and no exotic elements).
# These may legitimately be all-zero; nothing else may be.
RICH_RARE_CATEGORY_SLOTS = {
    "elem_other",
    "degree_0",
    "degree_5",
    "hyb_SP3D",
    "hyb_SP3D2",
}


def _dead_dimensions(dataset):
    """Names of feature columns that are constant across the whole dataset."""
    nodes = jnp.concatenate([g.nodes for g in dataset.graphs], axis=0)
    stds = jnp.std(nodes, axis=0)
    return {RICH_NAMES[int(i)] for i in jnp.where(stds == 0.0)[0]}


def test_rich_dead_dimensions_are_only_rare_categories():
    """No rich feature may be dead unless it is a rare one-hot category.

    The 'basic' featurizer carries chiral_tag, which is constant zero for any
    SMILES parsed without explicit stereochemistry -- dead by construction, not
    because the dataset lacks that chemistry (see the test below). This test
    keeps that class of bug out of 'rich': an unused one-hot tail like
    hyb_SP3D2 is fine, a dead 'aromatic' flag is not.
    """
    from pathlib import Path

    esol = Path(__file__).parent.parent / "datasets" / "esol.csv"
    if not esol.exists():
        pytest.skip("ESOL dataset not downloaded")

    dead = _dead_dimensions(MolecularDataset(esol, features="rich"))
    unexpected = dead - RICH_RARE_CATEGORY_SLOTS
    assert unexpected == set(), f"unexpectedly dead rich features: {sorted(unexpected)}"


def test_basic_chiral_tag_is_dead_by_construction():
    """Documents why 'rich' exists: basic's chiral_tag carries no signal.

    RDKit leaves the chiral tag unset for SMILES without stereo annotations,
    so this column is all-zero on ESOL -- one of six input dimensions wasted.
    """
    from pathlib import Path

    esol = Path(__file__).parent.parent / "datasets" / "esol.csv"
    if not esol.exists():
        pytest.skip("ESOL dataset not downloaded")

    dataset = MolecularDataset(esol, features="basic")
    nodes = jnp.concatenate([g.nodes for g in dataset.graphs], axis=0)
    chiral_tag = nodes[:, 3]
    assert float(jnp.std(chiral_tag)) == 0.0


def test_dataset_reports_featurizer_dim():
    import pandas as pd

    df = pd.DataFrame({"smiles": SMILES, "property": [0.0] * len(SMILES)})
    basic = MolecularDataset(df)
    rich = MolecularDataset(df, features="rich")
    assert basic.n_node_features == 6
    assert rich.n_node_features == ATOM_FEATURIZERS["rich"].dim
    assert rich.graphs[0].nodes.shape[1] == rich.n_node_features


def test_split_preserves_featurizer():
    import pandas as pd

    df = pd.DataFrame({"smiles": SMILES, "property": [0.0] * len(SMILES)})
    train, test = MolecularDataset(df, features="rich").split(test_size=0.3, seed=0)
    for part in (train, test):
        assert part.features == "rich"
        assert part.n_node_features == ATOM_FEATURIZERS["rich"].dim
