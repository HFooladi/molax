"""Numerical bounds shared by the uncertainty heads.

Predicted variance is produced as ``exp(log_var)``, so an unbounded ``log_var``
can overflow. The edge-aware models therefore bound it.

The bound must not be a hard clip. ``jnp.clip`` has exactly zero gradient
outside its range, which makes saturation an absorbing state: once the variance
head is pushed past the bound it receives no gradient and can never come back.
This was observed in practice -- training ``UncertaintyMPNN`` on unnormalized
ESOL labels drove raw ``log_var`` to ~48 within 40 steps (initial MSE is ~93, so
the Gaussian NLL correctly wants a large variance, and Adam overshoots). By the
time the fit improved and the correct ``log_var`` had fallen to ~1.7, every
molecule was pinned at ``exp(4.6) = 99.484`` with a gradient of exactly 0.0.

``tanh`` is not sufficient either: its derivative underflows to zero in float32
well before ``log_var`` reaches the magnitudes actually seen above.

A leaky clip fixes both. Inside the bound it is the identity, so well-scaled
models are completely unaffected. Outside, it has a small constant slope, so
gradient always flows and a saturated head recovers.
"""

import jax.numpy as jnp

# Bound on |log_var|: variance in [exp(-4.6), exp(4.6)] ~ [0.01, 100].
LOG_VAR_BOUND = 4.6

# Slope applied beyond the bound. Small enough that variance stays near the
# range above, non-zero so the gradient never dies.
LOG_VAR_LEAK = 0.01


def bound_log_var(log_var: jnp.ndarray) -> jnp.ndarray:
    """Softly bound ``log_var`` while keeping the gradient alive everywhere.

    Identity for ``|log_var| <= LOG_VAR_BOUND``; slope ``LOG_VAR_LEAK`` beyond.

    Args:
        log_var: Raw log-variance predicted by a variance head

    Returns:
        Bounded log-variance, same shape
    """
    clipped = jnp.clip(log_var, -LOG_VAR_BOUND, LOG_VAR_BOUND)
    return clipped + LOG_VAR_LEAK * (log_var - clipped)
