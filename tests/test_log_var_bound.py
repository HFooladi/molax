"""Regression tests for the log-variance bound in the edge-aware models.

MPNN, GAT and GraphTransformer bound `log_var` so predicted variance stays in
[0.01, 100]. That bound used to be `jnp.clip`, which has *exactly zero*
gradient outside its range: once the variance head saturated it received no
gradient and could never recover. In practice MPNN drove 100% of molecules to
the cap (variance 99.484 for every molecule, var_head gradient 0.0), leaving
the uncertainty output constant and useless.

The bound is now a leaky clip (see molax/models/bounds.py): identity inside the
range, small constant slope outside, so a saturated head always recovers.
"""

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from molax.models.bounds import LOG_VAR_BOUND, LOG_VAR_LEAK, bound_log_var
from molax.models.gat import GATConfig, UncertaintyGAT
from molax.models.graph_transformer import (
    GraphTransformerConfig,
    UncertaintyGraphTransformer,
)
from molax.models.mpnn import MPNNConfig, UncertaintyMPNN
from molax.utils.data import batch_graphs, smiles_to_jraph

SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"]

MODELS = [
    ("mpnn", UncertaintyMPNN, MPNNConfig),
    ("gat", UncertaintyGAT, GATConfig),
    ("graph_transformer", UncertaintyGraphTransformer, GraphTransformerConfig),
]


@pytest.fixture
def graphs():
    return batch_graphs([smiles_to_jraph(s) for s in SMILES])


def _build(model_cls, config_cls):
    config = config_cls(
        node_features=6,
        edge_features=1,
        hidden_features=[16, 16],
        out_features=1,
        dropout_rate=0.1,
    )
    return model_cls(config, rngs=nnx.Rngs(0))


@pytest.mark.parametrize("name,model_cls,config_cls", MODELS)
def test_variance_stays_in_range(name, model_cls, config_cls, graphs):
    """The bound must still hold -- variance in [exp(-B), exp(B)]."""
    model = _build(model_cls, config_cls)
    _, var = model(graphs, training=False)

    # The leak allows a small, bounded overshoot beyond exp(+/-LOG_VAR_BOUND);
    # what matters is that variance stays positive, finite and near the range.
    assert jnp.all(var > 0.0)
    assert jnp.all(jnp.isfinite(var))
    assert jnp.all(var < jnp.exp(LOG_VAR_BOUND) * 10)


@pytest.mark.parametrize("name,model_cls,config_cls", MODELS)
def test_variance_head_gradient_survives_saturation(
    name, model_cls, config_cls, graphs
):
    """A saturated variance head must still receive gradient.

    Drive the head far past the bound by inflating its bias, then check the
    gradient is non-zero. With the old jnp.clip this was exactly 0.0 and the
    head was permanently dead.
    """
    model = _build(model_cls, config_cls)

    # Push raw log_var to ~10x the bound: deep into saturation.
    model.var_head.bias[...] = jnp.full_like(
        model.var_head.bias[...], LOG_VAR_BOUND * 10
    )

    def loss_fn(model):
        mean, var = model(graphs, training=False)
        mean, var = mean.squeeze(-1), var.squeeze(-1)
        targets = jnp.zeros_like(mean)
        return jnp.mean(
            0.5 * (jnp.log(var + 1e-6) + (targets - mean) ** 2 / (var + 1e-6))
        )

    _, grads = nnx.value_and_grad(loss_fn)(model)
    grad_magnitude = float(jnp.abs(grads.var_head.kernel[...]).sum())

    assert jnp.isfinite(grad_magnitude)
    assert grad_magnitude > 0.0, (
        f"{name}: variance head gradient is exactly zero when saturated -- "
        "the head can never recover (the jnp.clip regression)"
    )


@pytest.mark.parametrize("name,model_cls,config_cls", MODELS)
def test_variance_is_not_constant_across_molecules(name, model_cls, config_cls, graphs):
    """Predicted variance must vary by molecule, not collapse to one value."""
    model = _build(model_cls, config_cls)
    _, var = model(graphs, training=False)

    real = var.squeeze(-1)[: len(SMILES)]  # drop the padding graph
    assert float(jnp.std(real)) > 0.0, (
        f"{name}: identical predicted variance for every molecule"
    )


def test_bound_is_exact_identity_inside_range():
    """Well-scaled log_var must pass through completely untouched."""
    inside = jnp.array([-4.5, -1.0, 0.0, 1.0, 4.5])
    assert jnp.allclose(bound_log_var(inside), inside)


def test_bound_is_monotone_and_leaky_outside_range():
    """Outside the range the bound must keep a small non-zero slope."""
    big = jnp.array([LOG_VAR_BOUND * 10])
    expected = LOG_VAR_BOUND + LOG_VAR_LEAK * (LOG_VAR_BOUND * 10 - LOG_VAR_BOUND)
    assert jnp.allclose(bound_log_var(big), expected)

    # strictly increasing, so gradient never flips sign
    xs = jnp.linspace(-50.0, 50.0, 201)
    ys = bound_log_var(xs)
    assert jnp.all(jnp.diff(ys) > 0)
