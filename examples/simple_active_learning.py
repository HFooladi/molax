from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from molax.models.gcn import UncertaintyGCN, UncertaintyGCNConfig
from molax.utils.data import MolecularDataset

# Path to dataset
DATASET_PATH = Path(__file__).parent.parent / "datasets" / "molecules.csv"

# Load dataset
dataset = MolecularDataset(DATASET_PATH)
print("Dataset loaded")
train_data, test_data = dataset.split_train_test(test_size=0.2, seed=42)
print("Train and test data split")

# Create model configuration
config = UncertaintyGCNConfig(
    in_features=train_data.graphs[0][0].shape[1],
    hidden_features=[64, 64],
    out_features=1,
    dropout_rate=0.1,
)

# Initialize model
model = UncertaintyGCN(config)

# Initialize optimizer with reference sharing (ModelAndOptimizer for simpler API)
model_and_opt = nnx.ModelAndOptimizer(model, optax.adam(1e-3))  # reference sharing

# Initialize random labeled pool
n_initial = 100
key = jax.random.PRNGKey(0)
initial_indices = jnp.arange(len(train_data.graphs))
initial_indices = jax.random.permutation(key, initial_indices)[:n_initial]

labeled_indices = set(initial_indices.tolist())
pool_indices = set(range(len(train_data.graphs))) - labeled_indices

# Active learning loop
n_iterations = 10
batch_size = 5

for iteration in range(n_iterations):
    print(f"Iteration {iteration + 1}/{n_iterations}")

    # Get current labeled and pool data
    labeled_idx_list = list(labeled_indices)
    labeled_data = [train_data.graphs[i] for i in labeled_idx_list]
    labeled_labels = jnp.array([train_data.labels[i] for i in labeled_idx_list])
    pool_idx_list = list(pool_indices)
    pool_data = [train_data.graphs[i] for i in pool_idx_list]

    # Train model
    key, subkey = jax.random.split(key)

    for epoch in range(100):
        # Mini-batch training
        key, subkey = jax.random.split(key)
        batch_indices = jax.random.permutation(subkey, len(labeled_data))[:batch_size]
        batch_graphs = [labeled_data[i] for i in batch_indices]
        batch_labels = labeled_labels[batch_indices]

        # Forward pass and update
        def loss_fn(model: UncertaintyGCN) -> jnp.ndarray:
            total_loss = 0.0
            for i, (features, adj) in enumerate(batch_graphs):
                mean, variance = model(features, adj, training=True)
                # Negative log likelihood loss for Gaussian
                nll = 0.5 * jnp.sum(
                    jnp.log(variance) + ((batch_labels[i] - mean) ** 2) / variance
                )
                total_loss += nll
            return total_loss / len(batch_graphs)

        # Compute gradients and update
        key, subkey = jax.random.split(key)
        loss, grads = nnx.value_and_grad(loss_fn)(model_and_opt.model)
        model_and_opt.update(grads)  # inplace updates

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    # Select new samples with acquisition function
    selected = []

    # Compute acquisition scores
    acquisition_scores = []
    for features, adj in pool_data:
        key, subkey = jax.random.split(key)
        mean, variance = model_and_opt.model(features, adj)
        # Simple uncertainty sampling - choose highest variance
        acquisition_scores.append(variance[0])

    # Select top-k samples
    top_indices = jnp.argsort(-jnp.array(acquisition_scores))[:batch_size]
    selected = top_indices.tolist()

    # Update sets
    for idx in selected:
        labeled_indices.add(pool_idx_list[idx])
        pool_indices.remove(pool_idx_list[idx])

    # Evaluate on test set
    test_loss = 0.0
    test_samples = 0
    for i in range(0, len(test_data.graphs), batch_size):
        batch_end = min(i + batch_size, len(test_data.graphs))
        batch = test_data.graphs[i:batch_end]
        batch_labels = jnp.array([test_data.labels[j] for j in range(i, batch_end)])

        batch_loss = 0
        for j, (features, adj) in enumerate(batch):
            mean, _ = model_and_opt.model(features, adj)
            batch_loss += jnp.mean((mean - batch_labels[j]) ** 2)

        test_loss += batch_loss
        test_samples += batch_end - i

    test_loss /= test_samples
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Labeled pool size: {len(labeled_indices)}")
    print("---")
