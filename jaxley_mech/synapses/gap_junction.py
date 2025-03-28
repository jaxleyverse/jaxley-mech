from typing import Dict, Optional, Tuple

import jax.numpy as jnp
from jaxley.mechanisms.synapses.synapse import Synapse


class GapJunction(Synapse):
    """
    Compute the synaptic current for a gap junction. Note that gap junctions are not
    solved with implicit Euler.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name = name if name else self.__class__.__name__

        self.params = {f"{name}_gE": 0.001}  # the conductance across the gap junction
        self.states = {}
        self.META = {
            "reference": "Abbott and Marder (1998)",
            "doi": "https://mitpress.mit.edu/books/methods-neuronal-modeling",
        }

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state."""
        return {}

    def compute_current(self, u, pre_voltage, post_voltage, params):
        """Return updated current."""
        return -1 * params[f"{self.name}_gE"] * (pre_voltage - post_voltage)
