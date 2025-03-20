import jax.numpy as jnp
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Tuple, List, Optional
import numpy as np

def smiles_to_graph(smiles: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert SMILES string to molecular graph representation.
    
    Args:
        smiles: SMILES string of molecule
    
    Returns:
        Tuple of (node_features, adjacency_matrix)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    # Get atom features
    atoms = mol.GetAtoms()
    n_atoms = len(atoms)
    
    # Atom features: [atomic_num, degree, formal_charge, chiral_tag, hybridization, aromacity]
    node_features = []
    for atom in atoms:
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetChiralTag(),
            atom.GetHybridization(),
            atom.GetIsAromatic(),
        ]
        node_features.append(features)
    
    # Create adjacency matrix
    adjacency = np.zeros((n_atoms, n_atoms))
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        adjacency[i, j] = float(bond_type)
        adjacency[j, i] = float(bond_type)
    
    return jnp.array(node_features), jnp.array(adjacency)

class MolecularDataset:
    def __init__(self, csv_path: str, smiles_col: str = 'smiles', label_col: str = 'labels'):
        """Load molecular dataset from CSV file.
        
        Args:
            csv_path: Path to CSV file
            smiles_col: Name of SMILES column
            label_col: Name of labels column
        """
        self.df = pd.read_csv(csv_path)
        self.smiles_col = smiles_col
        self.label_col = label_col
        
        # Convert SMILES to graphs
        self.graphs = []
        for smiles in self.df[smiles_col]:
            try:
                graph = smiles_to_graph(smiles)
                self.graphs.append(graph)
            except ValueError as e:
                print(f"Skipping invalid SMILES: {e}")
        
        self.labels = jnp.array(self.df[label_col])
        
    def get_batch(self, indices: List[int]) -> Tuple[List[Tuple[jnp.ndarray, jnp.ndarray]], jnp.ndarray]:
        """Get batch of molecular graphs and labels.
        
        Args:
            indices: List of indices to include in batch
            
        Returns:
            Tuple of (list of (node_features, adjacency) tuples, labels)
        """
        batch_graphs = [self.graphs[i] for i in indices]
        batch_labels = self.labels[indices]
        return batch_graphs, batch_labels
    
    def split_train_test(self, test_size: float = 0.2, seed: Optional[int] = None) -> Tuple['MolecularDataset', 'MolecularDataset']:
        """Split dataset into training and test sets.
        
        Args:
            test_size: Fraction of data to use for testing
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if seed is not None:
            np.random.seed(seed)
            
        n_samples = len(self.df)
        n_test = int(test_size * n_samples)
        
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        train_df = self.df.iloc[train_indices].reset_index(drop=True)
        test_df = self.df.iloc[test_indices].reset_index(drop=True)
        
        train_dataset = MolecularDataset(csv_path='', smiles_col=self.smiles_col, label_col=self.label_col)
        train_dataset.df = train_df
        train_dataset.graphs = [self.graphs[i] for i in train_indices]
        train_dataset.labels = self.labels[train_indices]
        
        test_dataset = MolecularDataset(csv_path='', smiles_col=self.smiles_col, label_col=self.label_col)
        test_dataset.df = test_df
        test_dataset.graphs = [self.graphs[i] for i in test_indices]
        test_dataset.labels = self.labels[test_indices]
        
        return train_dataset, test_dataset

def collate_fn(batch_graphs: List[Tuple[jnp.ndarray, jnp.ndarray]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Collate batch of molecular graphs into padded arrays.
    
    Args:
        batch_graphs: List of (node_features, adjacency) tuples
        
    Returns:
        Tuple of (padded_features, padded_adjacency)
    """
    # Find maximum number of nodes
    max_nodes = max(features.shape[0] for features, _ in batch_graphs)
    
    # Pad node features and adjacency matrices
    padded_features = []
    padded_adjacency = []
    
    for features, adjacency in batch_graphs:
        n_nodes = features.shape[0]
        n_features = features.shape[1]
        
        # Pad features
        padded_f = jnp.zeros((max_nodes, n_features))
        padded_f = padded_f.at[:n_nodes].set(features)
        padded_features.append(padded_f)
        
        # Pad adjacency
        padded_a = jnp.zeros((max_nodes, max_nodes))
        padded_a = padded_a.at[:n_nodes, :n_nodes].set(adjacency)
        padded_adjacency.append(padded_a)
    
    return jnp.stack(padded_features), jnp.stack(padded_adjacency)