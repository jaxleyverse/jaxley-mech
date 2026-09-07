import jax.numpy as jnp


def efun(x, y):
    """``x / (exp(x / y) - 1)``, with the removable singularity at ``x == 0``.

    ``expm1`` avoids the cancellation that destroys ``1 - exp(t)`` for small ``t`` in
    float32; at ``x == 0`` the analytic limit ``y`` is returned. ``t_safe`` keeps the
    untaken branch finite, so its gradient is not NaN.
    """
    t = x / y
    t_safe = jnp.where(t == 0.0, 1.0, t)
    return jnp.where(t == 0.0, y, x / jnp.expm1(t_safe))


def prettify(outputs_array, rec_states, dt):
    """For single compartment model output only."""
    outputs_length = outputs_array.shape[1]
    outputs_dict = {key: outputs_array[i] for (i, key) in enumerate(rec_states)}
    outputs_dict["time"] = jnp.arange(0, outputs_length * dt, dt)[:outputs_length]
    return outputs_dict
