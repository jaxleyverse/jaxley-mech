from typing import Dict, Optional

import jax.numpy as jnp
from jax.debug import print
from jax.lax import select, min
from jaxley.solver_gate import save_exp
from jaxley.synapses import Synapse

__all__ = ["AMPA", "GABAa", "GABAb", "NMDA"]

META = {"notes": "modified (kai-zou, ) syanspses from dms98"}


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
            f"{name}_k": 8,  # Steepness of the sigmoid function
        }
        self.synapse_states = {
            f"{name}_R": 0,  # Fraction of open receptors
            f"{name}_C": 0,  # Transmitter concentration
        }
        self.META = META

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state."""
        name = self.name

        new_release_intensity = 1 / (
            1
            + save_exp(-params[f"{name}_k"] * (pre_voltage - params[f"{name}_vt_pre"]))
        )

        Cmax = params[f"{name}_Cmax"]
        Cdur = params[f"{name}_Cdur"]

        C = u[f"{name}_C"]
        R = u[f"{name}_R"]

        decay_factor = save_exp(-delta_t / Cdur)

        new_C = min(
            Cmax, new_release_intensity * Cmax + C * decay_factor
        )  # smoothed release for gradients

        # Compute Rinf and Rtau for static parameters
        alpha = params[f"{name}_alpha"]
        beta = params[f"{name}_beta"]
        dR_dt = alpha * new_C * (1 - R) - beta * R
        new_R = R + dR_dt * delta_t

        # Return updated states including R0 and R1 for future use
        return {
            f"{name}_R": new_R,
            f"{name}_C": new_C,
        }

    def init_state(self, v, params):
        """Initialize the state."""
        name = self.name
        return {
            f"{name}_R": 0,
            f"{name}_C": 0,
            f"{name}_lastrelease": -1000,
            f"{name}_k": 5,
        }

    def compute_current(self, u, pre_voltage, post_voltage, params):
        """Compute and return synaptic current."""
        name = self.name
        g_syn = params[f"{name}_gAMPA"] * u[f"{name}_R"]
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
            f"{name}_k": 8,  # Steepness of the sigmoid function
        }
        self.synapse_states = {
            f"{name}_R": 0,  # Fraction of open receptors
            f"{name}_C": 0,  # Transmitter concentration
        }
        self.META = META

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state."""
        # Decrement timecount by delta_t, initialize if first run
        name = self._name

        # Determine whether a new release event should start
        new_release_intensity = 1 / (
            1
            + save_exp(-params[f"{name}_k"] * (pre_voltage - params[f"{name}_vt_pre"]))
        )
        Cmax = params[f"{name}_Cmax"]
        Cdur = params[f"{name}_Cdur"]
        C = u[f"{name}_C"]
        # new_C = select(
        #     new_release_intensity > 0.5,
        #     Cmax,
        #     C * save_exp(-delta_t / Cdur),
        # )

        new_C = min(
            Cmax,
            new_release_intensity * Cmax + C * save_exp(-delta_t / Cdur),
        )  # smoothed release for gradients

        # Compute Rinf and Rtau for static parameters
        alpha = params[f"{name}_alpha"]
        beta = params[f"{name}_beta"]
        R = u[f"{name}_R"]
        dR_dt = alpha * new_C * (1 - R) - beta * R
        new_R = R + dR_dt * delta_t

        # Return updated states including R0 and R1 for future use
        return {
            f"{name}_R": new_R,
            f"{name}_C": new_C,
        }

    def init_state(self, v, params):
        """Initialize the state."""
        name = self._name
        return {
            f"{name}_R": 0,
            f"{name}_C": 0,
        }

    def compute_current(self, u, pre_voltage, post_voltage, params):
        """Compute and return synaptic current."""
        name = self._name
        g_syn = params[f"{name}_gGABAa"] * u[f"{name}_R"]
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
            f"{name}_k": 8,  # Steepness of the sigmoid function
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
        }
        self.META = META

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state."""
        # Decrement timecount by delta_t, initialize if first run
        name = self._name

        # Determine whether a new release event should start
        new_release_intensity = 1 / (
            1
            + save_exp(-params[f"{name}_k"] * (pre_voltage - params[f"{name}_vt_pre"]))
        )
        Cmax = params[f"{name}_Cmax"]
        Cdur = params[f"{name}_Cdur"]
        C = u[f"{name}_C"]
        # new_C = select(
        #     new_release_intensity > 0.5,
        #     Cmax,
        #     C * save_exp(-delta_t / Cdur),
        # )

        new_C = min(
            Cmax,
            new_release_intensity * Cmax + C * save_exp(-delta_t / Cdur),
        )  # smoothed release for gradients

        # Update receptor (R) and G-protein (G) fractions
        R = u[f"{name}_R"]
        G = u[f"{name}_G"]
        K1 = params[f"{name}_K1"]
        K2 = params[f"{name}_K2"]
        K3 = params[f"{name}_K3"]
        K4 = params[f"{name}_K4"]
        new_R = R + delta_t * (K1 * C * (1 - R) - K2 * R)
        new_G = G + delta_t * (K3 * R - K4 * G)

        # Return updated states including R and G for future use
        return {
            f"{name}_R": new_R,
            f"{name}_G": new_G,
            f"{name}_C": new_C,
        }

    def init_state(self, v, params):
        """Initialize the state."""
        name = self._name
        return {
            f"{name}_R": 0,
            f"{name}_G": 0,
            f"{name}_C": 0,
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
            f"{name}_mg": 1,  # External magnesium concentration (mM)
            f"{name}_k": 8,  # Steepness of the sigmoid function
        }
        self.synapse_states = {
            f"{name}_R": 0,  # Fraction of open receptors
            f"{name}_C": 0,  # Transmitter concentration
        }
        self.META = META

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state."""
        # Decrement timecount by delta_t, initialize if first run
        name = self._name

        # Determine whether a new release event should start
        new_release_intensity = 1 / (
            1 + save_exp(-params[f"{name}_k"] * (pre_voltage - params[f"{name}_vt_pre"]))
        )
        Cmax = params[f"{name}_Cmax"]
        Cdur = params[f"{name}_Cdur"]
        C = u[f"{name}_C"]
        # new_C = select(
        #     new_release_intensity > 0.5,
        #     Cmax,
        #     C * save_exp(-delta_t / Cdur),
        # )
        new_C = min(
            Cmax,
            new_release_intensity * Cmax + C * save_exp(-delta_t / Cdur),
        )  # smoothed release for gradients

        # Compute new_R0 and new_R1 based on the receptor dynamics
        R = u[f"{name}_R"]
        alpha = params[f"{name}_alpha"]
        beta = params[f"{name}_beta"]

        # Update R based on whether there is a release
        dR_dt = alpha * new_C * (1 - R) - beta * R
        new_R = R + dR_dt * delta_t

        # Return updated states including R0 and R1 for future use
        return {
            f"{name}_R": new_R,
            f"{name}_C": new_C,
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
        R = u[f"{name}_R"]
        B = self.mgblock(post_voltage, params[f"{name}_mg"])
        g_syn = params[f"{name}_gNMDA"] * R * B
        return g_syn * (post_voltage - params[f"{name}_eNMDA"])

    @staticmethod
    def mgblock(v, mg_concentration):
        """Magnesium block factor."""
        return 1 / (1 + save_exp(0.062 * (-v)) * (mg_concentration / 3.57))


def exptable(x):
    """Approximate exponential function used in NEURON's AMPA model."""
    return select((x > -10) & (x < 10), save_exp(x), jnp.zeros_like(x))
