import jax.numpy as jnp


def efun(x, y):
    x += 1e-9
    return x / (jnp.exp(x / y) - 1.0)
