from typing import Dict, Optional

import jax.numpy as jnp
from jax.lax import select
from jaxley.solver_gate import save_exp
from jaxley.synapses import Synapse

META = {
    "reference_1": "Nishiyama, S., Hosoki, Y., Koike, C., & Amano, A. (2014). IEEE, 6116-6119.",
    "reference_2": "Witkovsky, P., Schmitz, Y., Akopian, A., Krizaj, D., & Tranchina, D. (1997). Journal of Neuroscience, 17(19), 7297-7306.",
}


class Ribbon_mGluR6(Synapse):
    def __init__(self, name: Optional[str] = None):
        self._name = name = name if name else self.__class__.__name__

        self.synapse_params = {
            f"{name}_V_half": -35,  # Voltage stimulating half-max release (mV)
            f"{name}_lam": 0.4,  # Vesicle replenishment rate at the ribbon
            f"{name}_p_r": 0.1,  # Probability of a vesicle at the ribbon moving to the dock
            f"{name}_D_max": 8,  # Maximum number of docked vesicles
            f"{name}_R_max": 50,  # Maximum number of vesicles at the ribbon
            f"{name}_gTRPM1": 1.65 * 10**-3,  # Maximum conductance (μS)
            f"{name}_eTRPM1": -11.5,  # Reversal potential (mV)
            f"{name}_KGlu": 50,  # Half saturating NT concentration (uM)
        }
        self.synapse_states = {
            f"{name}_released": 0,  # Number of vesicles released
            f"{name}_docked": 4,  # Number of vesicles at the dock
            f"{name}_ribboned": 25,  # Number of vesicles at the ribbon
            f"{name}_Glu": 50,  # Neurotransmitter concentration (mM)
            f"{name}_mTRPM1": 0.5,  # Channel activation
        }
        self.META = META

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state."""
        name = self.name
        # Presynaptic voltage to calcium to release probability
        k = 1.0
        p_d_t = 1 / (1 + save_exp(-k * (pre_voltage - params[f"{name}_V_half"])))

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

        # Get the new concentration of glutamate at the synaptic cleft
        d_Glu = -u[f"{name}_Glu"] + (new_released / params[f"{name}_D_max"]) * 2 * params[f"{name}_KGlu"]
        new_Glu = u[f"{name}_Glu"] + d_Glu * delta_t

        # Start with the receptor model
        Glu_norm = new_Glu**2 / (new_Glu**2 + params[f"{name}_KGlu"] ** 2)
        alpha = 40.0 * 10**-3 * (1 - Glu_norm)
        beta = 40.0 * 10**-3 *  Glu_norm

        dmTRPM1 = alpha * (1 - u[f"{name}_mTRPM1"]) - beta * u[f"{name}_mTRPM1"]
        new_mTRPM1 = u[f"{name}_mTRPM1"] + dmTRPM1 * delta_t

        return {
            f"{name}_released": new_released,
            f"{name}_docked": new_docked,
            f"{name}_ribboned": new_ribboned,
            f"{name}_Glu": new_Glu,  # Neurotransmitter concentration
            f"{name}_mTRPM1": new_mTRPM1,  # Channel activation
        }

    def init_state(self, v, params):
        """Initialize the state."""
        name = self.name
        return {
            f"{name}_released": 0,
            f"{name}_docked": 4,
            f"{name}_ribboned": 25,
            f"{name}_Glu": 25,  # Neurotransmitter concentration
            f"{name}_mTRPM1": 0,  # Channel activation
        }

    def compute_current(self, u, pre_voltage, post_voltage, params):
        """Compute and return synaptic current."""
        name = self.name
        g_syn = (
            params[f"{name}_gTRPM1"] * u[f"{name}_mTRPM1"]
        )  # multiply with 1000 to convert Siemens to milli Siemens.
        return g_syn * (post_voltage - params[f"{name}_eTRPM1"])