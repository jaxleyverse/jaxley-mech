from typing import Dict, Optional

import jax.numpy as jnp
from jax.lax import select
from jaxley.solver_gate import save_exp
from jaxley.synapses import Synapse

META = {
    "reference_1": "Nishiyama, S., Hosoki, Y., Koike, C., & Amano, A. (2014). IEEE, 6116-6119.",
    "reference_2": "Schroeder, C., Oesterle, J., Berens, P., Yoshimatsu, T., & Baden, T. (2021). eLife.",
}


class Ribbon_mGluR6(Synapse):
    def __init__(self, name: Optional[str] = None):
        self._name = name = name if name else self.__class__.__name__

        self.synapse_params = {
            f"{name}_e_max": 1.5,  # Maximum glutamate release
            f"{name}_r_max": 2.0,  # Rate of RP --> IP, movement to the ribbon
            f"{name}_i_max": 4.0,  # Rate of IP --> RRP, movement to the dock
            f"{name}_d_max": 0.1,  # Rate of RP refilling
            f"{name}_RRP_max": 3.0,  # Maximum number of docked vesicles
            f"{name}_IP_max": 10.0,  # Maximum number of vesicles at the ribbon
            f"{name}_RP_max": 25.0,  # Maximum number of vesicles in the reserve pool
            f"{name}_k": 1.0,  # Slope of calcium conversion nonlinearity
            f"{name}_V_half": -35.0,  # Half the voltage that gives maximum glutamate release
            f"{name}_gTRPM1": 1.65 * 10**-3,  # Maximum conductance (μS)
            f"{name}_eTRPM1": -11.5,  # Reversal potential (mV)
            f"{name}_KGlu": 5.0,  # Half saturating NT concentration (num vesicles)
        }
        self.synapse_states = {
            f"{name}_exo": 0.75,  # Number of vesicles released
            f"{name}_RRP": 1.5,  # Number of vesicles at the dock
            f"{name}_IP": 5.0,  # Number of vesicles at the ribbon
            f"{name}_RP": 12.5,  # Number of vesicles in the reserve pool
            f"{name}_mTRPM1": 0.5,  # Channel activation
        }
        self.META = META

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """
        Return updated synapse state.

        Caution: Synaptic states currently solved with explicit Euler!
        """
        name = self.name

        # Presynaptic voltage to calcium to release probability (switch to Cas dependence?)
        p_d_t = 1.0 / (
            1.0
            + save_exp(
                -1 * params[f"{name}_k"] * (pre_voltage - params[f"{name}_V_half"])
            )
        )

        # Glutamate release
        e_t = (
            params[f"{name}_e_max"]
            * p_d_t
            * u[f"{name}_RRP"]
            / params[f"{name}_RRP_max"]
        )
        # Rate of RP --> IP, movement to the ribbon
        r_t = (
            params[f"{name}_r_max"]
            * (1 - u[f"{name}_IP"] / params[f"{name}_IP_max"])
            * u[f"{name}_RP"]
            / params[f"{name}_RP_max"]
        )
        # Rate of IP --> RRP, movement to the dock
        i_t = (
            params[f"{name}_i_max"]
            * (1 - u[f"{name}_RRP"] / params[f"{name}_RRP_max"])
            * u[f"{name}_IP"]
            / params[f"{name}_IP_max"]
        )
        # Rate of RP refilling
        d_t = params[f"{name}_d_max"] * u[f"{name}_exo"]

        # Calculate the new vesicle numbers
        new_RP = u[f"{name}_RP"] + (d_t - e_t) * delta_t
        new_IP = u[f"{name}_IP"] + (r_t - i_t) * delta_t
        new_RRP = u[f"{name}_RRP"] + (i_t - e_t) * delta_t
        new_exo = u[f"{name}_exo"] + (e_t - d_t) * delta_t

        # Start with the receptor model
        Glu_norm = new_exo**2 / (new_exo**2 + params[f"{name}_KGlu"] ** 2)
        alpha = 40.0 * 10**-3 * (1 - Glu_norm)
        beta = 40.0 * 10**-3 * Glu_norm

        dmTRPM1 = alpha * (1 - u[f"{name}_mTRPM1"]) - beta * u[f"{name}_mTRPM1"]
        new_mTRPM1 = u[f"{name}_mTRPM1"] + dmTRPM1 * delta_t

        return {
            f"{name}_exo": new_exo,
            f"{name}_RRP": new_RRP,
            f"{name}_IP": new_IP,
            f"{name}_RP": new_RP,
            f"{name}_mTRPM1": new_mTRPM1,
        }

    def init_state(self, v, params):
        """Initialize the state."""
        name = self.name
        return {
            f"{name}_exo": 0.75,  # Number of vesicles released
            f"{name}_RRP": 1.5,  # Number of vesicles at the dock
            f"{name}_IP": 5.0,  # Number of vesicles at the ribbon
            f"{name}_RP": 12.5,  # Number of vesicles in the reserve pool
            f"{name}_mTRPM1": 0.5,  # Channel activation
        }

    def compute_current(self, u, pre_voltage, post_voltage, params):
        """Compute and return synaptic current."""
        name = self.name
        g_syn = params[f"{name}_gTRPM1"] * u[f"{name}_mTRPM1"]
        return g_syn * (post_voltage - params[f"{name}_eTRPM1"])
