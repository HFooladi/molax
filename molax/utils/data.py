"""Data utilities for molecular graph datasets using jraph for efficient batching."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import jax.numpy as jnp
import jraph
import numpy as np
import pandas as pd
from rdkit import Chem

from .logger import logger


def smiles_to_jraph(smiles: str) -> jraph.GraphsTuple:
    """Convert SMILES string to jraph GraphsTuple format.

    Args:
        smiles: SMILES string representing the molecule

    Returns:
        jraph.GraphsTuple containing the molecular graph

    Raises:
        ValueError: If the SMILES string is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    atoms = mol.GetAtoms()
    n_atoms = len(atoms)

    # Node features: [atomic_num, degree, formal_charge, chiral_tag,
    #                 hybridization, aromacity]
    node_features = []
    for atom in atoms:
        features = [
            float(atom.GetAtomicNum()),
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetChiralTag()),
            float(atom.GetHybridization()),
            float(atom.GetIsAromatic()),
        ]
        node_features.append(features)

    # Edge features: build sender/receiver lists from bonds
    senders = []
    receivers = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = float(bond.GetBondType())

        # Add both directions for undirected graph
        senders.extend([i, j])
        receivers.extend([j, i])
        edge_features.extend([[bond_type], [bond_type]])

    # Add self-loops for GCN message passing
    for i in range(n_atoms):
        senders.append(i)
        receivers.append(i)
        edge_features.append([1.0])  # Self-loop edge feature

    n_edges = len(senders)

    return jraph.GraphsTuple(
        nodes=jnp.array(node_features, dtype=jnp.float32),
        edges=jnp.array(edge_features, dtype=jnp.float32)
        if edge_features
        else jnp.zeros((0, 1), dtype=jnp.float32),
        senders=jnp.array(senders, dtype=jnp.int32),
        receivers=jnp.array(receivers, dtype=jnp.int32),
        n_node=jnp.array([n_atoms], dtype=jnp.int32),
        n_edge=jnp.array([n_edges], dtype=jnp.int32),
        globals=None,
    )


def batch_graphs(
    graphs: List[jraph.GraphsTuple],
    pad_to_nodes: Optional[int] = None,
    pad_to_edges: Optional[int] = None,
    pad_to_graphs: Optional[int] = None,
) -> jraph.GraphsTuple:
    """Batch multiple graphs into a single padded GraphsTuple.

    Padding ensures consistent shapes for JIT compilation efficiency.

    Args:
        graphs: List of individual GraphsTuple objects
        pad_to_nodes: Pad total nodes to this number (default: auto)
        pad_to_edges: Pad total edges to this number (default: auto)
        pad_to_graphs: Pad to this many graphs (default: len(graphs) + 1)

    Returns:
        Single batched and padded GraphsTuple
    """
    batched = jraph.batch(graphs)

    # Calculate padding sizes if not provided
    n_nodes = int(batched.n_node.sum())
    n_edges = int(batched.n_edge.sum())
    n_graphs = len(graphs)

    if pad_to_nodes is None:
        pad_to_nodes = n_nodes + 1  # +1 for padding graph
    if pad_to_edges is None:
        pad_to_edges = n_edges + 1
    if pad_to_graphs is None:
        pad_to_graphs = n_graphs + 1

    # Pad the batch for consistent shapes
    return jraph.pad_with_graphs(
        batched,
        n_node=pad_to_nodes,
        n_edge=pad_to_edges,
        n_graph=pad_to_graphs,
    )


def unbatch_graphs(batched: jraph.GraphsTuple) -> List[jraph.GraphsTuple]:
    """Unbatch a batched GraphsTuple back to individual graphs.

    Args:
        batched: Batched GraphsTuple

    Returns:
        List of individual GraphsTuple objects
    """
    return jraph.unbatch(batched)  # type: ignore[no-any-return]


class MolecularDataset:
    """Dataset class for molecular graphs using jraph format.

    Attributes:
        graphs: List of jraph.GraphsTuple objects
        labels: Array of property labels
        n_node_features: Number of node features
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path],
        smiles_col: str = "smiles",
        label_col: str = "property",
    ):
        """Initialize dataset from DataFrame or CSV file.

        Args:
            data: DataFrame or path to CSV file
            smiles_col: Column name for SMILES strings
            label_col: Column name for property labels
        """
        logger.info("Initializing MolecularDataset")

        if isinstance(data, (str, Path)):
            logger.info(f"Loading data from file: {data}")
            df = pd.read_csv(data)
        else:
            df = data

        if smiles_col not in df.columns:
            raise ValueError(f"Column '{smiles_col}' not found")
        if label_col not in df.columns:
            raise ValueError(f"Column '{label_col}' not found")

        # Convert SMILES to jraph graphs
        logger.info("Converting SMILES to jraph graphs")
        self.graphs: List[jraph.GraphsTuple] = []
        self.labels: List[float] = []

        for idx, row in df.iterrows():
            try:
                graph = smiles_to_jraph(row[smiles_col])
                self.graphs.append(graph)
                self.labels.append(float(row[label_col]))
            except ValueError as e:
                logger.warning(f"Skipping invalid SMILES at index {idx}: {e}")

        self.labels = jnp.array(self.labels, dtype=jnp.float32)
        self.n_node_features = self.graphs[0].nodes.shape[1] if self.graphs else 6

        logger.info(f"Loaded {len(self.graphs)} molecules")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Tuple[jraph.GraphsTuple, float]:
        return self.graphs[idx], self.labels[idx]

    def get_batched(
        self,
        indices: Optional[List[int]] = None,
        pad_to_nodes: Optional[int] = None,
        pad_to_edges: Optional[int] = None,
        pad_to_graphs: Optional[int] = None,
    ) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        """Get a batched GraphsTuple for the specified indices.

        Args:
            indices: List of indices to include. If None, returns all data.
            pad_to_nodes: Pad to this many nodes for consistent JIT shapes
            pad_to_edges: Pad to this many edges
            pad_to_graphs: Pad to this many graphs

        Returns:
            Tuple of (batched_graphs, labels)
        """
        if indices is None:
            indices = list(range(len(self.graphs)))

        graphs = [self.graphs[i] for i in indices]
        labels = self.labels[jnp.array(indices)]

        batched = batch_graphs(graphs, pad_to_nodes, pad_to_edges, pad_to_graphs)
        return batched, labels

    def compute_padding_sizes(self, batch_size: int) -> Tuple[int, int, int]:
        """Compute fixed padding sizes for efficient JIT compilation.

        Args:
            batch_size: Maximum batch size

        Returns:
            Tuple of (max_nodes, max_edges, n_graphs) for padding
        """
        # Find max nodes and edges across all graphs
        max_nodes_per_graph = max(int(g.n_node[0]) for g in self.graphs)
        max_edges_per_graph = max(int(g.n_edge[0]) for g in self.graphs)

        # Compute totals for a full batch + padding graph
        pad_nodes = max_nodes_per_graph * batch_size + 1
        pad_edges = max_edges_per_graph * batch_size + 1
        pad_graphs = batch_size + 1

        return pad_nodes, pad_edges, pad_graphs

    def split(
        self, test_size: float = 0.2, seed: Optional[int] = None
    ) -> Tuple["MolecularDataset", "MolecularDataset"]:
        """Split dataset into train and test sets.

        Args:
            test_size: Fraction for test set
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        logger.info(f"Splitting dataset: test_size={test_size}, seed={seed}")

        rng = np.random.default_rng(seed)
        n = len(self.graphs)
        indices = rng.permutation(n)

        n_test = int(test_size * n)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        train_dataset = MolecularDataset.__new__(MolecularDataset)
        train_dataset.graphs = [self.graphs[i] for i in train_idx]
        train_dataset.labels = self.labels[train_idx]  # type: ignore[assignment]
        train_dataset.n_node_features = self.n_node_features

        test_dataset = MolecularDataset.__new__(MolecularDataset)
        test_dataset.graphs = [self.graphs[i] for i in test_idx]
        test_dataset.labels = self.labels[test_idx]  # type: ignore[assignment]
        test_dataset.n_node_features = self.n_node_features

        logger.info(f"Split: {len(train_dataset)} train, {len(test_dataset)} test")
        return train_dataset, test_dataset


# Legacy aliases for backward compatibility
def smiles_to_graph(smiles: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Legacy function - converts to node features and adjacency matrix."""
    graph = smiles_to_jraph(smiles)
    n_nodes = int(graph.n_node[0])

    # Reconstruct adjacency matrix from edges
    adj = jnp.zeros((n_nodes, n_nodes))
    for s, r, e in zip(graph.senders, graph.receivers, graph.edges):
        if s != r:  # Skip self-loops for adjacency
            adj = adj.at[int(s), int(r)].set(float(e[0]))

    return graph.nodes, adj
