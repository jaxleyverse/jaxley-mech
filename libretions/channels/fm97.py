from typing import Dict, Optional, Union

import jax.numpy as jnp

from jaxley.channels import Channel
from jaxley.solver_gate import solve_gate_exponential

from ..utils import efun

"""
Fohlmeister, J. F., & Miller, R. F. (1997). Impulse Encoding Mechanisms of Ganglion Cells in the Tiger Salamander Retina. Journal of Neurophysiology, 78(4), 1935–1947. https://doi.org/10.1152/jn.1997.78.4.1935

- Species: Tiger salamander;
- Cell type: Retinal ganglion cells;
"""

########################
# extend Channel class #
########################


def clamp(
    self,
    Vh: Union[float, int],  # holding potential
    V: Union[float, int],  # step potential
    T: int,
    dt: float,
    states: Optional[dict] = None,
    params: Optional[Dict[str, jnp.ndarray]] = None,
):
    if states is None:
        states = self.channel_states
    if params is None:
        params = self.channel_params

    # holding potential (should be longer than the step potential duration)
    for _ in range(T * 2):
        states = self.update_states(states, dt=dt, voltages=Vh, params=params)
        self.compute_current(states, voltages=V, params=params)

    amps = []
    for i in range(T):
        states = self.update_states(states, dt=dt, voltages=V, params=params)
        amps.append(self.compute_current(states, voltages=V, params=params))
    return amps


Channel.VClamp = clamp

####################
# define channels  #
####################


class Leak(Channel):
    """Leakage current"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gl": 5e-5,  # S/cm^2
            f"{prefix}_el": -67.0,  # mV
        }
        self.channel_states = {}

    def update_states(
        self, u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """No state to update."""
        return {}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        # Multiply with 1000 to convert Siemens to milli Siemens.
        prefix = self._name
        leak_conds = params[f"{prefix}_gl"] * 1000  # mS/cm^2
        return leak_conds * (voltages - params[f"{prefix}_el"])


class Na(Channel):
    """Sodium channel"""

    channel_params = {
        f"Na_gNa": 50e-3,  # S/cm^2
        f"Na_vNa": 35.0,  # mV
    }
    channel_states = {"Na_m": 0.2, "Na_h": 0.2}

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNa": 50e-3,  # S/cm^2
            f"{prefix}_vNa": 35.0,  # mV
        }
        self.channel_states = {f"{prefix}_m": 0.2, f"{prefix}_h": 0.2}

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        "Update state."
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        m_new = solve_gate_exponential(ms, dt, *Na.m_gate(voltages))
        h_new = solve_gate_exponential(hs, dt, *Na.h_gate(voltages))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        "Return current."
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        na_conds = params[f"{prefix}_gNa"] * (ms**3) * hs * 1000  # mS/cm^2
        current = na_conds * (voltages - params[f"{prefix}_vNa"])
        return current

    @staticmethod
    def m_gate(e):
        alpha = (-0.6 * (e + 30 + 1e-9)) / (jnp.exp(-0.1 * (e + 30 + 1e-9)) - 1)
        beta = 20 * jnp.exp(-(e + 55) / 18)
        return alpha, beta

    @staticmethod
    def h_gate(e):
        alpha = 0.4 * jnp.exp(-0.05 * (e + 50 + 1e-9))
        beta = 6 / (1 + jnp.exp(-0.1 * (e + 20 + 1e-9)))
        return alpha, beta


class K(Channel):
    """Potassium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gK": 12e-3,  # S/cm^2
            f"{prefix}_vK": -75.0,  # mV
        }
        self.channel_states = {f"{prefix}_n": 0.1}

    def update_states(
        self, u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""
        prefix = self._name
        ns = u[f"{prefix}_n"]
        new_n = solve_gate_exponential(ns, dt, *K.n_gate(voltages))
        return {f"{prefix}_n": new_n}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        ns = u[f"{prefix}_n"]

        # Multiply with 1000 to convert Siemens to milli Siemens.
        k_conds = params[f"{prefix}_gK"] * (ns**4) * 1000  # mS/cm^2

        return k_conds * (voltages - params[f"{prefix}_vK"])

    @staticmethod
    def n_gate(e):
        alpha = 0.02 * efun(-(e + 40), 10)
        beta = 0.4 * jnp.exp(-(e + 50) / 80)
        return alpha, beta


class KA(Channel):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gKA": 36e-3,  # S/cm^2
            f"{prefix}_vKA": -75,  # mV
        }
        self.channel_states = {f"{prefix}_A": 0.2, f"{prefix}_hA": 0.2}

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state."""
        prefix = self._name
        As, hAs = u[f"{prefix}_A"], u[f"{prefix}_hA"]
        new_A = solve_gate_exponential(As, dt, *KA.A_gate(voltages))
        new_hA = solve_gate_exponential(hAs, dt, *KA.hA_gate(voltages))
        return {f"{prefix}_A": new_A, f"{prefix}_hA": new_hA}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages: float, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        As, hAs = u[f"{prefix}_A"], u[f"{prefix}_hA"]
        k_conds = params[f"{prefix}_gKA"] * (As**3) * hAs * 1000  # mS/cm^2
        return k_conds * (voltages - params[f"{prefix}_vKA"])

    @staticmethod
    def A_gate(e):
        alpha = 0.006 * efun(-(e + 90), 1)
        beta = 0.1 * jnp.exp(-(e + 30) / 10)
        return alpha, beta

    @staticmethod
    def hA_gate(e):
        alpha = 0.04 * jnp.exp(-(e + 70) / 20)
        beta = 0.6 / (1 + jnp.exp(-(e + 40) / 10))
        return alpha, beta


class Ca(Channel):
    """Calcium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gCa": 22e-4,  # S/cm^2
            f"{prefix}_Ca_e": 2.0,  # mM (external calcium concentration)
            f"{prefix}_Ca_res": 1e-4,  # mM (resting internal calcium concentration)
            f"{prefix}_r": 1.0,  # meters (effective radius of calcium domain near the channel)
        }
        self.channel_states = {f"{prefix}_c": 0.1, f"{prefix}_Ca_i": 1e-4}
        self.CONSTANTS = {
            "F": 96485.3329,  # C/mol (Faraday's constant)
            "T": 295.15,  # Kelvin (temperature)
            "R": 8.314,  # J/(mol K) (gas constant)
            "tau_Ca": 50.0,  # mS (time constant for calcium removal)
        }

    def update_states(self, u, dt, voltages, params):
        """Update state."""
        prefix = self._name
        cs = u[f"{prefix}_c"]
        Ca_in = u[f"{prefix}_Ca_i"]
        new_c = solve_gate_exponential(cs, dt, *Ca.c_gate(voltages))
        # Update internal calcium concentration based on the current calcium current
        iCa = self.compute_current(u, voltages, params)

        dCa_dt = (-3 * iCa / (2 * self.CONSTANTS["F"] * params[f"{prefix}_r"])) - (
            (Ca_in - params[f"{prefix}_Ca_res"]) / self.CONSTANTS[f"tau_Ca"]
        )
        Ca_in += dCa_dt * dt
        return {f"{prefix}_c": new_c, f"{prefix}_Ca_i": Ca_in}

    def compute_voltage(self, u, params):
        """Return voltage."""
        prefix = self._name
        Ca_in = u[f"{prefix}_Ca_i"]
        vCa = (
            self.CONSTANTS["R"] * self.CONSTANTS["T"] / (2 * self.CONSTANTS["F"])
        ) * jnp.log(params[f"{prefix}_Ca_e"] / Ca_in)
        return vCa * 1000  # mV

    def compute_current(self, u, voltages, params):
        """Return current."""
        prefix = self._name
        cs = u[f"{prefix}_c"]
        vCa = self.compute_voltage(u, params)
        # Multiply with 1000 to convert Siemens to milli Siemens.
        ca_conds = params[f"{prefix}_gCa"] * (cs**3) * 1000  # mS/cm^2
        current = ca_conds * (voltages - vCa)
        return current

    @staticmethod
    def c_gate(e):
        alpha = 0.3 * efun(-(e + 13), 10)
        beta = 10 * jnp.exp(-(e + 38) / 18)
        return alpha, beta
