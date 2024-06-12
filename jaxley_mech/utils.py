import jax.numpy as jnp
from jaxley.solver_gate import save_exp


def efun(x, y):
    x += 1e-9
    return x / (save_exp(x / y) - 1.0)


def prettify(outputs_array, rec_states, dt):
    """For single compartment model output only."""
    outputs_length = outputs_array.shape[1]
    outputs_dict = {key: outputs_array[i] for (i, key) in enumerate(rec_states)}
    outputs_dict["time"] = jnp.arange(0, outputs_length * dt, dt)
    return outputs_dict
