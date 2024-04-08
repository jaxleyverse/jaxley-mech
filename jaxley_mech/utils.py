import jax.numpy as jnp
from jaxley.solver_gate import save_exp


def efun(x, y):
    x += 1e-9
    return x / (save_exp(x / y) - 1.0)
