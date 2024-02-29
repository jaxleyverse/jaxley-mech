from typing import Dict, Optional

import jax.numpy as jnp
from jaxley.synapses import Synapse


class AMPA(Synapse):

    def __init__(self, name: Optional[str] = None):
        self._name = name = name if name else self.__class__.__name__

        self.synapse_params = {
            f"{name}_gAMPA": 0.1e-3,  # Maximum conductance (mS)
            f"{name}_eAMPA": 0.0,  # Reversal potential (mV)
            f"{name}_Cmax": 1,  # Max transmitter concentration (mM)
            f"{name}_Cdur": 1,  # Transmitter duration (ms)
            f"{name}_alpha": 1.1,  # Forward (binding) rate (/ms mM)
            f"{name}_beta": 0.19,  # Backward (unbinding) rate (/ms)
            f"{name}_vt_pre": 0,  # Presynaptic voltage threshold for release (mV)
            f"{name}_deadtime": 1,  # Minimum time between release events (ms)
        }
        self.synapse_states = {
            f"{name}_R": 0,  # Fraction of open receptors
            f"{name}_C": 0,  # Transmitter concentration
            f"{name}_lastrelease": -1000,  # Time since last release (ms)
            f"{name}_timecount": -1,
            f"{name}_R0": 0,  # R at start of release
            f"{name}_R1": 0,  # R at end of release
        }

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state."""
        # Decrement timecount by delta_t, initialize if first run
        name = self.name
        timecount = u[f"{name}_timecount"]
        new_timecount = jnp.where(
            timecount == -1,
            params[f"{name}_Cdur"],
            timecount - delta_t,
        )

        # Determine whether a new release event should start
        new_release_condition = (pre_voltage > params[f"{name}_vt_pre"]) & (
            new_timecount <= -params[f"{name}_deadtime"]
        )
        Cmax = params[f"{name}_Cmax"]
        Cdur = params[f"{name}_Cdur"]
        C = u[f"{name}_C"]
        new_C = jnp.where(
            new_release_condition,
            Cmax,
            jnp.where(new_timecount > 0, C, 0),
        )

        # Update lastrelease time: reset if new release starts, otherwise unchanged
        new_lastrelease = jnp.where(
            new_release_condition, 0, u[f"{name}_lastrelease"] + delta_t
        )

        # Compute Rinf and Rtau for static parameters
        alpha = params[f"{name}_alpha"]
        beta = params[f"{name}_beta"]
        R_inf = Cmax * alpha / (Cmax * alpha + beta)
        R_tau = 1 / (alpha * Cmax + beta)

        # Determine new_R0: should update when new release starts
        R0 = u[f"{name}_R0"]
        R1 = u[f"{name}_R1"]
        R = u[f"{name}_R"]
        new_R0 = jnp.where(
            new_release_condition, R, R0
        )  # Corrected to use previous R0 if no new release

        # Determine new_R1 for use when release ends
        new_R1 = jnp.where(C > 0, R, R1)  # Update R1 if we were releasing

        # Update R based on whether there is a release
        time_since_release = new_lastrelease - Cdur
        new_R = jnp.where(
            new_C > 0,
            R_inf + (new_R0 - R_inf) * exptable(-(new_lastrelease) / R_tau),
            new_R1 * exptable(-time_since_release * beta),
        )

        # Update timecount: reset if new release, keep decrementing otherwise
        new_timecount = jnp.where(new_release_condition, Cdur, new_timecount)

        # Return updated states including R0 and R1 for future use
        return {
            f"{name}_R": new_R,
            f"{name}_C": new_C,
            f"{name}_lastrelease": new_lastrelease,
            f"{name}_timecount": new_timecount,
            f"{name}_R0": new_R0,
            f"{name}_R1": new_R1,
        }

    def init_state(self, v, params):
        """Initialize the state."""
        name = self.name
        return {
            f"{name}_R": 0,
            f"{name}_R0": 0,
            f"{name}_R1": 0,
            f"{name}_C": 0,
            f"{name}_lastrelease": -1000,
        }

    def compute_current(self, u, pre_voltage, post_voltage, params):
        """Compute and return synaptic current."""
        name = self.name
        g_syn = (
            params[f"{name}_gAMPA"] * u[f"{name}_R"] * 1000
        )  # multiply with 1000 to convert Siemens to milli Siemens.
        return g_syn * (post_voltage - params[f"{name}_eAMPA"])


class GABAa(Synapse):

    def __init__(self, name: Optional[str] = None):
        self._name = name = name if name else self.__class__.__name__

        self.synapse_params = {
            f"{name}_gGABAa": 0.1e-3,  # Maximum conductance (mS)
            f"{name}_eGABAa": -80.0,  # Reversal potential (mV)
            f"{name}_Cmax": 1,  # Max transmitter concentration (mM)
            f"{name}_Cdur": 1,  # Transmitter duration (ms)
            f"{name}_alpha": 5,  # Forward (binding) rate (/ms mM)
            f"{name}_beta": 0.18,  # Backward (unbinding) rate (/ms)
            f"{name}_vt_pre": 0,  # Presynaptic voltage threshold for release (mV)
            f"{name}_deadtime": 1,  # Minimum time between release events (ms)
        }
        self.synapse_states = {
            f"{name}_R": 0,  # Fraction of open receptors
            f"{name}_C": 0,  # Transmitter concentration
            f"{name}_lastrelease": -1000,  # Time since last release (ms)
            f"{name}_timecount": -1,
            f"{name}_R0": 0,  # R at start of release
            f"{name}_R1": 0,  # R at end of release
        }

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state."""
        # Decrement timecount by delta_t, initialize if first run
        name = self._name
        timecount = u[f"{name}_timecount"]
        new_timecount = jnp.where(
            timecount == -1,
            params[f"{name}_Cdur"],
            timecount - delta_t,
        )

        # Determine whether a new release event should start
        new_release_condition = (pre_voltage > params[f"{name}_vt_pre"]) & (
            new_timecount <= -params[f"{name}_deadtime"]
        )
        Cmax = params[f"{name}_Cmax"]
        Cdur = params[f"{name}_Cdur"]
        C = u[f"{name}_C"]
        new_C = jnp.where(
            new_release_condition,
            Cmax,
            jnp.where(new_timecount > 0, C, 0),
        )

        # Update lastrelease time: reset if new release starts, otherwise unchanged
        new_lastrelease = jnp.where(
            new_release_condition, 0, u[f"{name}_lastrelease"] + delta_t
        )

        # Compute Rinf and Rtau for static parameters
        alpha = params[f"{name}_alpha"]
        beta = params[f"{name}_beta"]
        R_inf = Cmax * alpha / (Cmax * alpha + beta)
        R_tau = 1 / (alpha * Cmax + beta)

        # Determine new_R0: should update when new release starts
        R0 = u[f"{name}_R0"]
        R1 = u[f"{name}_R1"]
        R = u[f"{name}_R"]
        new_R0 = jnp.where(
            new_release_condition, R, R0
        )  # Corrected to use previous R if no new release

        # Determine new_R1 for use when release ends
        new_R1 = jnp.where(C > 0, R, R1)  # Update R1 if we were releasing

        # Update R based on whether there is a release
        time_since_release = new_lastrelease - Cdur
        new_R = jnp.where(
            new_C > 0,
            R_inf + (new_R0 - R_inf) * exptable(-(new_lastrelease) / R_tau),
            new_R1 * exptable(-beta * time_since_release),
        )

        # Update timecount: reset if new release, keep decrementing otherwise
        new_timecount = jnp.where(new_release_condition, Cdur, new_timecount)

        # Return updated states including R0 and R1 for future use
        return {
            f"{name}_R": new_R,
            f"{name}_C": new_C,
            f"{name}_lastrelease": new_lastrelease,
            f"{name}_timecount": new_timecount,
            f"{name}_R0": new_R0,
            f"{name}_R1": new_R1,
        }

    def init_state(self, v, params):
        """Initialize the state."""
        name = self._name
        return {
            f"{name}_R": 0,
            f"{name}_R0": 0,
            f"{name}_R1": 0,
            f"{name}_C": 0,
            f"{name}_lastrelease": -1000,
        }

    def compute_current(self, u, pre_voltage, post_voltage, params):
        """Compute and return synaptic current."""
        name = self._name
        g_syn = (
            params[f"{name}_gGABAa"] * u[f"{name}_R"] * 1000
        )  # multiply with 1000 to convert Siemens to milli Siemens.
        return g_syn * (post_voltage - params[f"{name}_eGABAa"])


def exptable(x):
    """Approximate exponential function used in NEURON's AMPA model."""
    return jnp.where((x > -10) & (x < 10), jnp.exp(x), 0.0)
