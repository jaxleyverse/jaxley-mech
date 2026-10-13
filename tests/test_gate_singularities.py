"""Rate equations must stay finite and smooth at their removable singularities.

Several rate expressions have the form ``x / (exp(x / y) - 1)``, which is 0 / 0 at
``x == 0``. Guarding that with a small additive constant does not work in float32:
the constant is lost to rounding, and whether it survives at all depends on the
order XLA chooses for the additions, so the bug only appears under ``jax.jit``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxley_mech.channels.fohlmeister97 import KA, Ca, K, Na
from jaxley_mech.channels.l5pc import CaHVA, H, NapEt2, NaTaT, NaTs2T

# (channel, gate, voltage at which that gate's rate expression is 0 / 0)
SINGULARITIES = [
    (NaTaT, "m_gate", -38.0),
    (NaTaT, "h_gate", -66.0),
    (NaTs2T, "m_gate", -32.0),
    (NaTs2T, "h_gate", -60.0),
    (NapEt2, "m_gate", -38.0),
    (NapEt2, "h_gate", -17.0),
    (NapEt2, "h_gate", -64.4),
    (CaHVA, "m_gate", -27.0),
    (H, "m_gate", -154.9),
    (Na, "m_gate", -30.0),
    (K, "n_gate", -40.0),
    (KA, "A_gate", -90.0),
    (Ca, "c_gate", -13.0),
]
IDS = [f"{c.__name__}.{g}@{v}" for c, g, v in SINGULARITIES]


def _neighbourhood(v, n=4):
    """``v`` together with its ``n`` nearest float32 neighbours on either side."""
    out = [np.float32(v)]
    for direction in (-np.inf, np.inf):
        x = np.float32(v)
        for _ in range(n):
            x = np.nextafter(x, np.float32(direction), dtype=np.float32)
            out.append(x)
    return jnp.asarray(sorted(out), dtype=jnp.float32)


def _total(fn):
    """Sum every output of a gate, so no output escapes the gradient check."""
    return lambda v: sum(jnp.sum(o) for o in fn(v))


@pytest.mark.parametrize("channel, gate, v_sing", SINGULARITIES, ids=IDS)
def test_finite_under_jit(channel, gate, v_sing):
    """No output may be NaN or inf at, or adjacent to, the singularity."""
    outputs = jax.jit(getattr(channel, gate))(_neighbourhood(v_sing))
    for out in outputs:
        assert np.all(np.isfinite(np.asarray(out)))


@pytest.mark.parametrize("channel, gate, v_sing", SINGULARITIES, ids=IDS)
def test_continuous_at_singularity(channel, gate, v_sing):
    """The singularity is removable, so the gate must be smooth across it.

    Compares the value at ``v_sing`` against the midpoint of two samples on either
    side. Needs no reference values, and catches a finite-but-wrong limit as well
    as a NaN.
    """
    fn = jax.jit(getattr(channel, gate))
    at = fn(jnp.float32(v_sing))
    lo = fn(jnp.float32(v_sing - 0.01))
    hi = fn(jnp.float32(v_sing + 0.01))
    for out, left, right in zip(at, lo, hi):
        midpoint = 0.5 * (np.asarray(left) + np.asarray(right))
        assert np.allclose(np.asarray(out), midpoint, rtol=1e-3, atol=0.0)


@pytest.mark.parametrize("channel, gate, v_sing", SINGULARITIES, ids=IDS)
def test_gradient_finite_and_bounded(channel, gate, v_sing):
    """The gradient must be finite and must not blow up.

    Cancellation in ``1 - exp(x)`` can leave the value correct while making the
    derivative enormous (observed: 3.5e8 where neighbouring voltages give -8), so
    finiteness alone is too weak. The bound is deliberately loose: the ``where``
    that removes the singularity flattens the derivative of that one term, so the
    gradient at ``v_sing`` legitimately differs from its neighbours by a small
    factor -- it must simply stay the same order of magnitude.
    """
    grad = jax.jit(jax.grad(_total(getattr(channel, gate))))
    at = float(grad(jnp.float32(v_sing)))
    nearby = 0.5 * (
        abs(float(grad(jnp.float32(v_sing - 0.01))))
        + abs(float(grad(jnp.float32(v_sing + 0.01))))
    )
    assert np.isfinite(at)
    assert abs(at) <= 100.0 * max(nearby, 1.0)
