"""Atom featurizers for converting RDKit atoms to node feature vectors.

Two featurizers are registered:

- ``"basic"`` (default): six raw descriptor values. Kept as the default so that
  existing code and saved configurations keep working unchanged.
- ``"rich"``: 29-dimensional one-hot encoding. Raw descriptors span very
  different scales (atomic number reaches 53 while aromaticity is 0/1), which
  a plain :class:`flax.nnx.Linear` input layer handles poorly. One-hot encoding
  removes the scale mismatch and substantially improves accuracy -- on ESOL it
  takes a two-layer ``UncertaintyGCN`` from a test RMSE of 1.93 to 0.88
  (mean-predictor baseline: 2.13).

Select a featurizer by name through :func:`molax.utils.data.smiles_to_jraph` or
:class:`molax.utils.data.MolecularDataset`::

    graph = smiles_to_jraph("CCO", features="rich")

Read the resulting feature width from the registry rather than hardcoding it::

    n_features = ATOM_FEATURIZERS["rich"].dim
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from rdkit import Chem

# Elements covered explicitly by the "rich" featurizer. Anything outside this
# set falls into a shared "other" slot.
RICH_ELEMENTS: Sequence[object] = [
    6,
    7,
    8,
    9,
    15,
    16,
    17,
    35,
    53,
]  # C N O F P S Cl Br I
RICH_DEGREES: Sequence[object] = [0, 1, 2, 3, 4, 5]
RICH_NUM_HS: Sequence[object] = [0, 1, 2, 3, 4]
RICH_HYBRIDIZATIONS: Sequence[object] = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


@dataclass(frozen=True)
class AtomFeaturizer:
    """A named atom featurizer with a fixed output width.

    Attributes:
        name: Registry key for this featurizer
        dim: Number of features produced per atom
        fn: Callable mapping an RDKit atom to a list of floats of length ``dim``
    """

    name: str
    dim: int
    fn: Callable[[Chem.Atom], List[float]]

    def __call__(self, atom: Chem.Atom) -> List[float]:
        return self.fn(atom)


def _one_hot(value: object, options: Sequence[object]) -> List[float]:
    """One-hot encode ``value`` over ``options`` (all zeros if not present)."""
    return [float(value == option) for option in options]


def _basic_atom_features(atom: Chem.Atom) -> List[float]:
    """Six raw descriptors -- the original molax featurization.

    Note that ``GetChiralTag`` is zero for every atom in datasets parsed from
    SMILES without assigned stereochemistry, so this dimension is often dead.
    Prefer the ``"rich"`` featurizer for new work.
    """
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetChiralTag()),
        float(atom.GetHybridization()),
        float(atom.GetIsAromatic()),
    ]


def _rich_atom_features(atom: Chem.Atom) -> List[float]:
    """29-dimensional one-hot featurization.

    Layout: element (10) + degree (6) + total Hs (5) + hybridization (5)
    + aromatic, in-ring, formal charge (3).
    """
    atomic_num = atom.GetAtomicNum()
    return (
        _one_hot(atomic_num, RICH_ELEMENTS)
        + [float(atomic_num not in RICH_ELEMENTS)]
        + _one_hot(atom.GetDegree(), RICH_DEGREES)
        + _one_hot(int(atom.GetTotalNumHs()), RICH_NUM_HS)
        + _one_hot(atom.GetHybridization(), RICH_HYBRIDIZATIONS)
        + [
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.GetFormalCharge()),
        ]
    )


_RICH_DIM = (
    len(RICH_ELEMENTS)
    + 1
    + len(RICH_DEGREES)
    + len(RICH_NUM_HS)
    + len(RICH_HYBRIDIZATIONS)
    + 3
)

ATOM_FEATURIZERS: Dict[str, AtomFeaturizer] = {
    "basic": AtomFeaturizer(name="basic", dim=6, fn=_basic_atom_features),
    "rich": AtomFeaturizer(name="rich", dim=_RICH_DIM, fn=_rich_atom_features),
}

DEFAULT_FEATURIZER = "basic"


def get_atom_featurizer(features: str = DEFAULT_FEATURIZER) -> AtomFeaturizer:
    """Look up a registered atom featurizer by name.

    Args:
        features: Registry key, one of ``ATOM_FEATURIZERS``

    Returns:
        The matching AtomFeaturizer

    Raises:
        ValueError: If no featurizer is registered under that name
    """
    if features not in ATOM_FEATURIZERS:
        known = ", ".join(sorted(ATOM_FEATURIZERS))
        raise ValueError(f"Unknown featurizer '{features}'. Available: {known}")
    return ATOM_FEATURIZERS[features]
