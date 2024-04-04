from typing import Dict, Optional

import jax.numpy as jnp
from jax.lax import select
from jaxley.synapses import Synapse

__all__ = ["AMPA", "GABAa", "GABAb", "NMDA"]

META = {
    "reference": "Destexhe, A., Mainen, Z. F., & Sejnowski, T. J. (1998). Kinetic models of synaptic transmission. Methods in neuronal modeling, 2, 1-25.",
    "link": "https://www.csc.kth.se/utbildning/kth/kurser/DD2435/biomod12/kursbunt/f9/KochCh1Destexhe.pdf",
    "source": "https://modeldb.science/18500?tab=2",
}


class AMPA(Synapse):
    def __init__(self, name: Optional[str] = None):
        self._name = name = name if name else self.__class__.__name__

        self.synapse_params = {
            f"{name}_gAMPA": 0.1e-3,  # Maximum conductance (μS)
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
        self.META = META

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state."""
        # Decrement timecount by delta_t, initialize if first run
        name = self.name
        timecount = u[f"{name}_timecount"]
        new_timecount = select(
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
        new_C = select(
            new_release_condition,
            Cmax,
            select(new_timecount > 0, C, jnp.asarray([0.0])),
        )

        # Update lastrelease time: reset if new release starts, otherwise unchanged
        new_lastrelease = select(
            new_release_condition,
            jnp.asarray([0.0]),
            u[f"{name}_lastrelease"] + delta_t,
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
        new_R0 = select(
            new_release_condition, R, R0
        )  # Corrected to use previous R0 if no new release

        # Determine new_R1 for use when release ends
        new_R1 = select(C > 0, R, R1)  # Update R1 if we were releasing

        # Update R based on whether there is a release
        time_since_release = new_lastrelease - Cdur
        new_R = select(
            new_C > 0,
            R_inf + (new_R0 - R_inf) * exptable(-(new_lastrelease) / R_tau),
            new_R1 * exptable(-time_since_release * beta),
        )

        # Update timecount: reset if new release, keep decrementing otherwise
        new_timecount = select(new_release_condition, Cdur, new_timecount)

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
            params[f"{name}_gAMPA"] * u[f"{name}_R"]
        )  # multiply with 1000 to convert Siemens to milli Siemens.
        return g_syn * (post_voltage - params[f"{name}_eAMPA"])


class GABAa(Synapse):
    def __init__(self, name: Optional[str] = None):
        self._name = name = name if name else self.__class__.__name__

        self.synapse_params = {
            f"{name}_gGABAa": 0.1e-3,  # Maximum conductance (μS)
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
        self.META = META

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state."""
        # Decrement timecount by delta_t, initialize if first run
        name = self._name
        timecount = u[f"{name}_timecount"]
        new_timecount = select(
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
        new_C = select(
            new_release_condition,
            Cmax,
            select(new_timecount > 0, C, jnp.asarray([0.0])),
        )

        # Update lastrelease time: reset if new release starts, otherwise unchanged
        new_lastrelease = select(
            new_release_condition,
            jnp.asarray([0.0]),
            u[f"{name}_lastrelease"] + delta_t,
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
        new_R0 = select(
            new_release_condition, R, R0
        )  # Corrected to use previous R if no new release

        # Determine new_R1 for use when release ends
        new_R1 = select(C > 0, R, R1)  # Update R1 if we were releasing

        # Update R based on whether there is a release
        time_since_release = new_lastrelease - Cdur
        new_R = select(
            new_C > 0,
            R_inf + (new_R0 - R_inf) * exptable(-(new_lastrelease) / R_tau),
            new_R1 * exptable(-beta * time_since_release),
        )

        # Update timecount: reset if new release, keep decrementing otherwise
        new_timecount = select(new_release_condition, Cdur, new_timecount)

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
            params[f"{name}_gGABAa"] * u[f"{name}_R"]
        )  # multiply with 1000 to convert Siemens to milli Siemens.
        return g_syn * (post_voltage - params[f"{name}_eGABAa"])


class GABAb(Synapse):
    def __init__(self, name: Optional[str] = None):
        self._name = name = name if name else self.__class__.__name__

        self.synapse_params = {
            f"{name}_gGABAb": 0.1e-3,  # Maximum conductance (μS)
            f"{name}_eGABAb": -95.0,  # Reversal potential (mV)
            f"{name}_Cmax": 0.5,  # Max transmitter concentration (mM)
            f"{name}_Cdur": 0.3,  # Transmitter duration (ms)
            f"{name}_vt_pre": 0,  # Presynaptic voltage threshold for release (mV)
            f"{name}_deadtime": 1,  # Minimum time between release events (ms)
            f"{name}_K1": 0.09,  # Forward binding rate to receptor (/ms mM)
            f"{name}_K2": 1.2e-3,  # Backward unbinding rate from receptor (/ms)
            f"{name}_K3": 180e-3,  # Rate of G-protein production (/ms)
            f"{name}_K4": 34e-3,  # Rate of G-protein decay (/ms)
            f"{name}_KD": 100,  # Dissociation constant of K+ channel
            f"{name}_n": 4,  # Number of binding sites of G-protein on K+ channel
        }
        self.synapse_states = {
            f"{name}_R": 0,  # Fraction of activated receptors
            f"{name}_G": 0,  # Fraction of activated G-protein
            f"{name}_C": 0,  # Transmitter concentration
            f"{name}_lastrelease": -1000,  # Time since last release (ms)
            f"{name}_timecount": -1,
        }
        self.META = META

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state."""
        # Decrement timecount by delta_t, initialize if first run
        name = self._name
        timecount = u[f"{name}_timecount"]
        new_timecount = select(
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
        new_C = select(
            new_release_condition,
            Cmax,
            select(new_timecount > 0, C, jnp.asarray([0.0])),
        )

        # Update lastrelease time: reset if new release starts, otherwise unchanged
        new_lastrelease = select(
            new_release_condition,
            jnp.asarray([0.0]),
            u[f"{name}_lastrelease"] + delta_t,
        )

        # Update receptor (R) and G-protein (G) fractions
        R = u[f"{name}_R"]
        G = u[f"{name}_G"]
        K1 = params[f"{name}_K1"]
        K2 = params[f"{name}_K2"]
        K3 = params[f"{name}_K3"]
        K4 = params[f"{name}_K4"]
        new_R = R + delta_t * (K1 * C * (1 - R) - K2 * R)
        new_G = G + delta_t * (K3 * R - K4 * G)

        # Update timecount: reset if new release, keep decrementing otherwise
        new_timecount = select(new_release_condition, Cdur, new_timecount)

        # Return updated states including R and G for future use
        return {
            f"{name}_R": new_R,
            f"{name}_G": new_G,
            f"{name}_C": new_C,
            f"{name}_lastrelease": new_lastrelease,
            f"{name}_timecount": new_timecount,
        }

    def init_state(self, v, params):
        """Initialize the state."""
        name = self._name
        return {
            f"{name}_R": 0,
            f"{name}_G": 0,
            f"{name}_lastrelease": -1000,
            f"{name}_C": 0,
            f"{name}_timecount": -1,
        }

    def compute_current(self, u, pre_voltage, post_voltage, params):
        """Compute and return synaptic current."""
        name = self._name
        KD = params[f"{name}_KD"]
        n = params[f"{name}_n"]
        Gn = u[f"{name}_G"] ** n
        g_syn = params[f"{name}_gGABAb"] * Gn / (Gn + KD)  # Convert S to mS
        return g_syn * (post_voltage - params[f"{name}_eGABAb"])


class NMDA(Synapse):
    def __init__(self, name: Optional[str] = None):
        self._name = name = name if name else self.__class__.__name__

        self.synapse_params = {
            f"{name}_gNMDA": 0.1e-3,  # Maximum conductance (μS)
            f"{name}_eNMDA": 0.0,  # Reversal potential (mV)
            f"{name}_Cmax": 1,  # Max transmitter concentration (mM)
            f"{name}_Cdur": 1,  # Transmitter duration (ms)
            f"{name}_alpha": 0.072,  # Forward (binding) rate (/ms mM)
            f"{name}_beta": 0.0066,  # Backward (unbinding) rate (/ms)
            f"{name}_vt_pre": 0,  # Presynaptic voltage threshold for release (mV)
            f"{name}_deadtime": 1,  # Minimum time between release events (ms)
            f"{name}_mg": 1,  # External magnesium concentration (mM)
        }
        self.synapse_states = {
            f"{name}_R": 0,  # Fraction of open receptors
            f"{name}_C": 0,  # Transmitter concentration
            f"{name}_lastrelease": -1000,  # Time since last release (ms)
            f"{name}_timecount": -1,
            f"{name}_R0": 0,  # R at start of release
            f"{name}_R1": 0,  # R at end of release
        }
        self.META = META

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state."""
        # Decrement timecount by delta_t, initialize if first run
        name = self._name
        timecount = u[f"{name}_timecount"]
        new_timecount = select(
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
        new_C = select(
            new_release_condition,
            Cmax,
            select(new_timecount > 0, C, jnp.asarray([0.0])),
        )

        # Update lastrelease time: reset if new release starts, otherwise unchanged
        new_lastrelease = select(
            new_release_condition,
            jnp.asarray([0.0]),
            u[f"{name}_lastrelease"] + delta_t,
        )

        # Compute new_R0 and new_R1 based on the receptor dynamics
        R0 = u[f"{name}_R0"]
        R1 = u[f"{name}_R1"]
        R = u[f"{name}_R"]
        alpha = params[f"{name}_alpha"]
        beta = params[f"{name}_beta"]
        Rinf = Cmax * alpha / (Cmax * alpha + beta)
        Rtau = 1 / (alpha * Cmax + beta)
        new_R0 = select(new_release_condition, R, R0)
        new_R1 = select(C > 0, R, R1)
        time_since_release = new_lastrelease - Cdur

        # Update R based on whether there is a release
        new_R = select(
            new_C > 0,
            Rinf + (new_R0 - Rinf) * exptable(-(new_lastrelease) / Rtau),
            new_R1 * exptable(-beta * time_since_release),
        )

        # Update timecount: reset if new release, keep decrementing otherwise
        new_timecount = select(new_release_condition, Cdur, new_timecount)

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
            f"{name}_timecount": -1,
        }

    def compute_current(self, u, pre_voltage, post_voltage, params):
        """Compute and return synaptic current."""
        name = self._name
        R = u[f"{name}_R"]
        B = self.mgblock(post_voltage, params[f"{name}_mg"])
        g_syn = params[f"{name}_gNMDA"] * R * B
        return g_syn * (post_voltage - params[f"{name}_eNMDA"])

    @staticmethod
    def mgblock(v, mg_concentration):
        """Magnesium block factor."""
        return 1 / (1 + jnp.exp(0.062 * (-v)) * (mg_concentration / 3.57))


def exptable(x):
    """Approximate exponential function used in NEURON's AMPA model."""
    return select((x > -10) & (x < 10), jnp.exp(x), jnp.asarray([0.0]))
