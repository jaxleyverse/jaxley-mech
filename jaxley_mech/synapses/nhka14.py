from typing import Dict, Optional

import jax.numpy as jnp
from jax.lax import select
from jaxley.solver_gate import save_exp
from jaxley.synapses import Synapse

META = {
    "reference_1": "Nishiyama, S., Hosoki, Y., Koike, C., & Amano, A. (2014). IEEE, 6116-6119.",
    "reference_2": "Witkovsky, P., Schmitz, Y., Akopian, A., Krizaj, D., & Tranchina, D. (1997). Journal of Neuroscience, 17(19), 7297-7306.",
}


class mGluR6(Synapse):
    def __init__(self, name: Optional[str] = None):
        self._name = name = name if name else self.__class__.__name__

        self.synapse_params = {
            f"{name}_gTRPM1": 1.65 * 10**-2,  # Maximum conductance (μS)
            f"{name}_eTRPM1": -11.5,  # Reversal potential (mV)
            f"{name}_KGlu": 50,  # Max transmitter concentration (mM)
            f"{name}_CGlu": 200,  # Constant of transmitted release
            f"{name}_Vhalf": -22,  # Voltage of half-saturation (mV)
            f"{name}_kGlu": 4.3,  # Slope factor of glutamate release
            f"{name}_r0": 0,  # Baseline glutamate release rate
        }
        self.synapse_states = {
            f"{name}_Glu": 0,  # Neurotransmitter concentration
            f"{name}_mTRPM1": 0,  # Channel activation
        }
        self.META = META

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state."""
        name = self.name

        # Model of glutamate release (Witkovsky et al., 1997)
        new_Glu = (
            params[f"{name}_CGlu"]
            / (
                1
                + save_exp(
                    (params[f"{name}_Vhalf"] - pre_voltage) / params[f"{name}_kGlu"]
                )
            )
            + params[f"{name}_r0"]
        )

        Glu_norm = new_Glu**2 / (new_Glu**2 + params[f"{name}_KGlu"] ** 2)
        alpha = 40.0 * 10**-3 * (1 - Glu_norm)
        beta = 40.0 * 10**-3 *  Glu_norm

        dmTRPM1 = alpha * (1 - u[f"{name}_mTRPM1"]) - beta * u[f"{name}_mTRPM1"]
        new_mTRPM1 = u[f"{name}_mTRPM1"] + dmTRPM1 * delta_t

        return {
            f"{name}_Glu": new_Glu,  # Neurotransmitter concentration
            f"{name}_mTRPM1": new_mTRPM1,  # Channel activation
        }

    def init_state(self, v, params):
        """Initialize the state."""
        name = self.name
        return {
            f"{name}_mTRPM1": 0,  # Channel activation
        }

    def compute_current(self, u, pre_voltage, post_voltage, params):
        """Compute and return synaptic current."""
        name = self.name
        g_syn = (
            params[f"{name}_gTRPM1"] * u[f"{name}_mTRPM1"]
        )  # multiply with 1000 to convert Siemens to milli Siemens.
        return g_syn * (post_voltage - params[f"{name}_eTRPM1"])
