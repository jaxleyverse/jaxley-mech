from typing import Dict, Optional, Tuple

import jax.numpy as jnp
from jaxley.synapses.synapse import Synapse
from jax import Array


class GapJunction(Synapse):
    """
    Compute the synaptic current for a gap junction. Note that gap junctions are not
    solved with implicit Euler.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self._name = name = name if name else self.__class__.__name__

        self.synapse_params = {
            f"{name}_gE": 0.001  # the conductance across the gap junction
        }
        self.synapse_states = {}
        self.META = {
            "reference": "Abbott and Marder (1998)",
            "doi": "https://mitpress.mit.edu/books/methods-neuronal-modeling",
        }

    def update_states(
        self,
        synapse_states: dict[str, Array],
        synapse_params: dict[str, Array],
        pre_voltage: Array,
        post_voltage: Array,
        pre_states: dict[str, Array],
        post_states: dict[str, Array],
        pre_params: dict[str, Array],
        post_params: dict[str, Array],
        delta_t: float,
    ):
        """Return updated synapse state."""
        return {}

    def compute_current(
        self,
        synapse_states: dict[str, Array],
        synapse_params: dict[str, Array],
        pre_voltage: Array,
        post_voltage: Array,
        pre_states: dict[str, Array],
        post_states: dict[str, Array],
        pre_params: dict[str, Array],
        post_params: dict[str, Array],
        delta_t: float,
    ):
        """Return updated current."""
        return -1 * synapse_params[f"{self._name}_gE"] * (pre_voltage - post_voltage)
