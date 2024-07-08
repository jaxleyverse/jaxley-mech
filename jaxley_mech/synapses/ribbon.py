from typing import Optional

import jax.numpy as jnp
from jaxley.solver_gate import save_exp
from jaxley.synapses.synapse import Synapse

META = {
    "reference": [
        "Schroeder, C., Klindt, D., Strauss, S., Franke, K., Bethge, M., Euler, T., Berens, P. (2020). System identification with biophysical constraints: a circuit model of the inner retina. NeurIPS 2020.",
        "Dayan, P., & Abbott, L. F. (2001). Theoretical neuroscience: computational and mathematical modeling of neural systems. MIT press.",
    ]
}


class RibbonSynapse(Synapse):
    def __init__(self, name: Optional[str] = None):
        self._name = name = name if name else self.__class__.__name__

        self.synapse_params = {
            f"{name}_gS": 0.1e-4,  # Maximal synaptic conductance (uS)
            f"{name}_tau": 0.5,  # Decay time constant of postsynaptic conductance (s)
            f"{name}_e_syn": 0,  # Reversal potential of postsynaptic membrane at the receptor (mV)
            f"{name}_lam": 0.4,  # Vesicle replenishment rate at the ribbon
            f"{name}_p_r": 0.1,  # Probability of a vesicle at the ribbon moving to the dock
            f"{name}_D_max": 8,  # Maximum number of docked vesicles
            f"{name}_R_max": 50,  # Maximum number of vesicles at the ribbon
        }
        self.synapse_states = {
            f"{name}_released": 0,  # Number of vesicles released
            f"{name}_docked": 4,  # Number of vesicles at the dock
            f"{name}_ribboned": 25,  # Number of vesicles at the ribbon
            f"{name}_P_rel": 0,  # Normalized vesicle release
            f"{name}_P_s": 0,  # Kernel of postsynaptic conductance
        }
        self.META = META

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state."""
        name = self.name
        k = 1.0
        V_half = -35

        # Presynaptic voltage to calcium to release probability
        p_d_t = 1 / (1 + save_exp(-k * (pre_voltage - V_half)))

        # Vesicle release (NOTE: p_d_t is the mean of the beta distribution)
        new_released = p_d_t * u[f"{name}_docked"]

        # Movement to the dock
        new_docked = (
            u[f"{name}_docked"]
            + params[f"{name}_p_r"] * u[f"{name}_ribboned"]
            - new_released
        )
        new_docked = jnp.clip(new_docked, 0, params[f"{name}_D_max"])

        # Movement to the ribbon
        dock_moved = jnp.maximum(0, new_docked - u[f"{name}_docked"])
        new_ribboned = u[f"{name}_ribboned"] + params[f"{name}_lam"] - dock_moved
        new_ribboned = jnp.clip(new_ribboned, 0, params[f"{name}_R_max"])

        # Single exponential decay to model postsynaptic conductance (Dayan & Abbott)
        P_rel = new_released / params[f"{name}_D_max"]
        P_s = save_exp(-delta_t / params[f"{name}_tau"])

        return {
            f"{name}_released": new_released,
            f"{name}_docked": new_docked,
            f"{name}_ribboned": new_ribboned,
            f"{name}_P_rel": P_rel,
            f"{name}_P_s": P_s,
        }

    def compute_current(self, u, pre_voltage, post_voltage, params):
        """Return updated current."""
        name = self.name
        g_syn = params[f"{name}_gS"] * u[f"{name}_P_rel"] * u[f"{name}_P_s"]
        return g_syn * (post_voltage - params[f"{name}_e_syn"])
