from typing import Dict, Tuple

import jax.numpy as jnp
from jaxley.synapses.synapse import Synapse


class GapJunction(Synapse):
    """
    Compute the synaptic current for a gap junction. Note that gap junctions are not
    solved with implicit Euler.

    This synapse can also be found in the book:
        L. F. Abbott and E. Marder, "Modeling Small Networks," in Methods in Neuronal
        Modeling, C. Koch and I. Sergev, Eds. Cambridge: MIT Press, 1998.

    synapse_params:
        - gE: the conductance across the gap junction
    """

    synapse_params = {"gE": 0.001}
    synapse_states = {}

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state."""
        return {}

    def compute_current(self, u, pre_voltage, post_voltage, params):
        """Return updated current."""
        return -1 * params["gE"] * (pre_voltage - post_voltage)
