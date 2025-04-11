import jax.numpy as jnp
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Tuple, List, Optional, Union
import numpy as np
from pathlib import Path

def smiles_to_graph(smiles: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert SMILES string to molecular graph representation.
    
    This function converts a SMILES string into a molecular graph representation suitable
    for graph neural networks. The graph is represented as node features and an adjacency
    matrix. Node features include atomic properties while the adjacency matrix encodes
    the molecular structure.
    
    Args:
        smiles: SMILES string representing the molecule to convert
        
    Returns:
        Tuple containing:
            - node_features: jnp.ndarray of shape (n_atoms, 6) containing atom features
              where each atom is represented by [atomic_num, degree, formal_charge,
              chiral_tag, hybridization, aromacity]
            - adjacency_matrix: jnp.ndarray of shape (n_atoms, n_atoms) where values
              represent bond types (0 for no bond, 1 for single bond, 2 for double bond,
              etc.)
    
    Raises:
        ValueError: If the SMILES string is invalid and cannot be parsed
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    # Get atom features
    atoms = mol.GetAtoms()
    n_atoms = len(atoms)
    
    # Atom features: [atomic_num, degree, formal_charge, chiral_tag, hybridization, aromacity, is_in_ring]
    node_features = []
    for atom in atoms:
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetChiralTag(),
            atom.GetHybridization(),
            atom.GetIsAromatic(),
            atom.IsInRing(),
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
    """A dataset class for handling molecular data in graph format.
    
    This class provides functionality to load molecular data from CSV files or pandas
    DataFrames, convert SMILES strings to graph representations, and manage the dataset
    for training and evaluation. It supports batching, splitting, and random access to
    molecular graphs and their labels.
    
    Attributes:
        df: pandas DataFrame containing the original data
        smiles_col: Name of the column containing SMILES strings
        label_col: Name of the column containing labels
        graphs: List of molecular graphs as (node_features, adjacency_matrix) tuples
        labels: jnp.ndarray containing the labels for all molecules
    """
    
    def __init__(self, df: Union[pd.DataFrame, str], smiles_col: str = 'smiles', label_col: str = 'label'):
        """Initialize a MolecularDataset instance.
        
        Args:
            df: Either a pandas DataFrame or a path to a CSV file containing the data
            smiles_col: Name of the column containing SMILES strings. Defaults to 'smiles'
            label_col: Name of the column containing labels. Defaults to 'label'
            
        Raises:
            ValueError: If the specified column names are not found in the DataFrame
        """
        if isinstance(df, str) or isinstance(df, Path):
            self.df = pd.read_csv(df)
        else:
            self.df = df
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
    
    def __len__(self) -> int:
        """Get the number of molecules in the dataset.
        
        Returns:
            int: Number of molecules in the dataset
        """
        return len(self.graphs)
    
    def __getitem__(self, index: int) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """Get a single molecule and its label by index.
        
        Args:
            index: Integer index of the molecule to retrieve
            
        Returns:
            Tuple containing:
                - graph: Tuple of (node_features, adjacency_matrix)
                - label: jnp.ndarray containing the molecule's label
        """
        return self.graphs[index], self.labels[index]
        
    def get_batch(self, indices: List[int]) -> Tuple[List[Tuple[jnp.ndarray, jnp.ndarray]], jnp.ndarray]:
        """Get a batch of molecular graphs and their labels.
        
        This method is useful for creating mini-batches during training. It returns
        the graphs and labels for the specified indices.
        
        Args:
            indices: List of integer indices specifying which molecules to include
                    in the batch
            
        Returns:
            Tuple containing:
                - batch_graphs: List of (node_features, adjacency_matrix) tuples
                - batch_labels: jnp.ndarray containing the labels for the batch
        """
        batch_graphs = [self.graphs[i] for i in indices]
        batch_labels = self.labels[indices]
        return batch_graphs, batch_labels
    
    def split_train_test(self, test_size: float = 0.2, seed: Optional[int] = None) -> Tuple['MolecularDataset', 'MolecularDataset']:
        """Split the dataset into training and test sets.
        
        This method randomly splits the dataset into training and test sets while
        maintaining the original data distribution. The split is reproducible if
        a seed is provided.
        
        Args:
            test_size: Fraction of the dataset to use for testing (between 0 and 1).
                      Defaults to 0.2 (20% test set)
            seed: Optional random seed for reproducibility. If None, the split will
                 be random
            
        Returns:
            Tuple containing:
                - train_dataset: MolecularDataset instance containing the training data
                - test_dataset: MolecularDataset instance containing the test data
        """
        if seed is not None:
            np.random.seed(seed)
            
        n_samples = len(self)
        n_test = int(test_size * n_samples)
        
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        train_df = self.df.iloc[train_indices].reset_index(drop=True)
        test_df = self.df.iloc[test_indices].reset_index(drop=True)
        
        train_dataset = MolecularDataset(df=train_df, smiles_col=self.smiles_col, label_col=self.label_col)
        test_dataset = MolecularDataset(df=test_df, smiles_col=self.smiles_col, label_col=self.label_col)
        
        return train_dataset, test_dataset

def collate_fn(batch_graphs: List[Tuple[jnp.ndarray, jnp.ndarray]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Collate a batch of molecular graphs into padded arrays.
    
    This function is used to prepare batches of molecular graphs for processing in
    graph neural networks. It handles variable-sized graphs by padding them to the
    size of the largest graph in the batch.
    
    Args:
        batch_graphs: List of (node_features, adjacency_matrix) tuples representing
                     the molecular graphs in the batch
        
    Returns:
        Tuple containing:
            - padded_features: jnp.ndarray of shape (batch_size, max_nodes, n_features)
                              containing the padded node features
            - padded_adjacency: jnp.ndarray of shape (batch_size, max_nodes, max_nodes)
                               containing the padded adjacency matrices
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