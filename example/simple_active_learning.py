import jax
import jax.numpy as jnp
from src.models import UncertaintyGCN
from src.acquisition import combined_acquisition
from src.utils.data import MolecularDataset
import optax

# Load dataset
dataset = MolecularDataset('data/molecules.csv')
train_data, test_data = dataset.split_train_test(test_size=0.2, seed=42)

# Initialize model
model = UncertaintyGCN(
    hidden_features=(64, 64),
    output_features=1,
    dropout_rate=0.1
)

# Initialize optimizer
optimizer = optax.adam(learning_rate=1e-3)

# Initialize random labeled pool
n_initial = 100
initial_indices = jnp.arange(len(train_data.graphs))
initial_indices = jax.random.permutation(jax.random.PRNGKey(0), initial_indices)[:n_initial]

labeled_indices = set(initial_indices.tolist())
pool_indices = set(range(len(train_data.graphs))) - labeled_indices

# Active learning loop
n_iterations = 10
batch_size = 5
key = jax.random.PRNGKey(0)

for iteration in range(n_iterations):
    print(f"Iteration {iteration + 1}/{n_iterations}")
    
    # Get current labeled and pool data
    labeled_data = [train_data.graphs[i] for i in labeled_indices]
    labeled_labels = train_data.labels[list(labeled_indices)]
    pool_data = [train_data.graphs[i] for i in pool_indices]
    
    # Train model
    params = model.init(key, labeled_data[0][0], labeled_data[0][1])
    opt_state = optimizer.init(params)
    
    for epoch in range(100):
        # Mini-batch training
        batch_idx = jax.random.permutation(key, len(labeled_data))[:batch_size]
        batch_graphs = [labeled_data[i] for i in batch_idx]
        batch_labels = labeled_labels[batch_idx]
        
        # Forward pass
        def loss_fn(params):
            preds, _ = model.apply(params, batch_graphs[0][0], batch_graphs[0][1])
            return jnp.mean((preds - batch_labels) ** 2)
        
        # Update
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
    
    # Select new samples
    selected = combined_acquisition(
        model,
        params,
        [train_data.graphs[i] for i in pool_indices],
        labeled_data,
        batch_size
    )
    
    # Update sets
    pool_indices_list = list(pool_indices)
    for idx in selected:
        labeled_indices.add(pool_indices_list[idx])
        pool_indices.remove(pool_indices_list[idx])
    
    # Evaluate on test set
    test_loss = 0
    for i in range(0, len(test_data.graphs), batch_size):
        batch = test_data.graphs[i:i + batch_size]
        batch_labels = test_data.labels[i:i + batch_size]
        preds, _ = model.apply(params, batch[0][0], batch[0][1])
        test_loss += jnp.mean((preds - batch_labels) ** 2)
    
    test_loss /= (len(test_data.graphs) // batch_size)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Labeled pool size: {len(labeled_indices)}")
    print("---")