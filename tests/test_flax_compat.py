"""Guardrails against Flax NNX API drift.

molax targets the Flax NNX API as of flax>=0.12 (verified against flax 0.12.2 /
jax 0.9.0 / optax 0.2.6). The patterns asserted here are the ones that have
already changed once in NNX's history and would otherwise break silently in an
example script rather than loudly in CI.
"""

import warnings

import flax.nnx as nnx
import jax.numpy as jnp
import optax
import pytest

from molax.models.gcn import GCNConfig, UncertaintyGCN
from molax.utils.data import batch_graphs, smiles_to_jraph

SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]


@pytest.fixture
def model():
    config = GCNConfig(
        node_features=6, hidden_features=[16, 16], out_features=1, dropout_rate=0.1
    )
    return UncertaintyGCN(config, rngs=nnx.Rngs(0))


@pytest.fixture
def graphs():
    return batch_graphs([smiles_to_jraph(s) for s in SMILES])


def test_optimizer_requires_wrt_argument(model):
    """nnx.Optimizer gained a required `wrt` argument in Flax 0.11."""
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    assert optimizer is not None


def test_optimizer_update_takes_model_and_grads(model, graphs):
    """`optimizer.update(model, grads)` is the 0.11+ two-argument form.

    Before 0.11 this was `optimizer.update(grads)`. Calling the old form
    against a current Flax raises, and vice versa.
    """
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    before = model.mean_head.kernel[...].copy()

    def loss_fn(model):
        mean, _ = model(graphs, training=True)
        return jnp.mean(mean**2)

    _, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    assert not jnp.allclose(before, model.mean_head.kernel[...]), (
        "optimizer.update did not modify parameters"
    )


def test_variable_indexing_is_supported(model):
    """`variable[...]` is the current accessor; `.value` is deprecated in 0.12.

    molax must not reintroduce `.value` anywhere -- see the no-deprecation test
    below, which is what would actually catch it.
    """
    kernel = model.mean_head.kernel[...]
    assert isinstance(kernel, jnp.ndarray)
    assert kernel.shape == (16, 1)


def test_train_step_emits_no_deprecation_warnings(model, graphs):
    """A full JIT-compiled train step must be free of deprecated API use.

    This is the broad guard: any deprecated NNX call reached during a training
    step -- `.value` access included -- turns into an error here.
    """
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    labels = jnp.zeros(len(SMILES) + 1)
    mask = jnp.arange(len(SMILES) + 1) < len(SMILES)

    @nnx.jit
    def train_step(model, optimizer, graphs, labels, mask):
        def loss_fn(model):
            mean, var = model(graphs, training=True)
            mean, var = mean.squeeze(-1), var.squeeze(-1)
            nll = 0.5 * (jnp.log(var + 1e-6) + (labels - mean) ** 2 / (var + 1e-6))
            return jnp.sum(jnp.where(mask, nll, 0.0)) / jnp.sum(mask)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        warnings.simplefilter("error", FutureWarning)
        loss = train_step(model, optimizer, graphs, labels, mask)

    assert jnp.isfinite(loss)


def test_nnx_list_tracks_parameters_through_jit(model, graphs):
    """Submodules stored in nnx.List must stay visible to grad and jit."""
    assert isinstance(model.conv_layers, nnx.List)
    assert len(model.conv_layers) == 2

    @nnx.jit
    def grad_norm(model):
        def loss_fn(model):
            mean, _ = model(graphs, training=False)
            return jnp.mean(mean**2)

        _, grads = nnx.value_and_grad(loss_fn)(model)
        leaves = nnx.to_flat_state(grads)
        return sum(jnp.sum(v[...] ** 2) for _, v in leaves)

    # Nonzero gradient reaching the list members means they are tracked.
    assert float(grad_norm(model)) > 0.0


def test_dropout_is_active_only_in_training(model, graphs):
    """training=False must be deterministic; training=True must not be."""
    a, _ = model(graphs, training=False)
    b, _ = model(graphs, training=False)
    assert jnp.allclose(a, b), "eval-mode forward pass is not deterministic"

    samples = [model(graphs, training=True)[0] for _ in range(8)]
    spread = jnp.max(jnp.var(jnp.stack(samples), axis=0))
    assert float(spread) > 0.0, "dropout RNG did not advance between calls"
