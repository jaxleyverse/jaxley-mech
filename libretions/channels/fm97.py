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

META = {
    "reference": "Fohlmeister, J. F., & Miller, R. F. (1997). Impulse Encoding Mechanisms of Ganglion Cells in the Tiger Salamander Retina. Journal of Neurophysiology, 78(4), 1935–1947. https://doi.org/10.1152/jn.1997.78.4.1935",
    "species": "Tiger salamander",
    "cell_type": "Retinal ganglion cells",
}


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
        self.meta = META

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
            f"{prefix}_gNa": 5e-4,  # S/cm^2
            f"{prefix}_vNa": 35.0,  # mV
        }
        self.channel_states = {f"{prefix}_m": 0.2, f"{prefix}_h": 0.2}
        self.meta = META

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
            "vK": -75.0,  # mV
        }
        self.channel_states = {f"{prefix}_n": 0.1}
        self.meta = META

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

        return k_conds * (voltages - params[f"vK"])

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
            f"vK": -75,  # mV
        }
        self.channel_states = {f"{prefix}_A": 0.2, f"{prefix}_hA": 0.2}
        self.meta = META

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
        return k_conds * (voltages - params[f"vK"])

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
        self.CONSTANTS = {
            "F": 96485.3329,  # C/mol (Faraday's constant)
            "T": 295.15,  # Kelvin (temperature)
            "R": 8.314,  # J/(mol K) (gas constant)
            "tau_Ca": 50.0,  # mS (time constant for calcium removal)
        }
        self.channel_params = {
            f"{prefix}_gCa": 22e-4,  # S/cm^2
            f"{prefix}_r": 1.0,  # meters (effective radius of calcium domain near the channel)
            "CaCon_rest": 1e-4,  # mM (resting internal calcium concentration)
            "CaCon_e": 2.0,  # mM (external calcium concentration)
        }
        self.channel_states = {
            f"{prefix}_c": 0.1,
            f"CaCon_i": 1e-4,
            f"{prefix}_vCa": Ca.compute_voltage(
                1e-4, self.channel_params, self.CONSTANTS
            ),
        }
        self.meta = META

    def update_states(self, u, dt, voltages, params):
        """Update state."""
        prefix = self._name
        cs = u[f"{prefix}_c"]
        Ca_in = u["CaCon_i"]
        new_c = solve_gate_exponential(cs, dt, *Ca.c_gate(voltages))
        # Update internal calcium concentration based on the current calcium current
        iCa = self.compute_current(u, voltages, params)

        dCa_dt = (-3 * iCa / (2 * self.CONSTANTS["F"] * params[f"{prefix}_r"])) - (
            (Ca_in - params["CaCon_rest"]) / self.CONSTANTS[f"tau_Ca"]
        )
        Ca_in += dCa_dt * dt

        vCa = Ca.compute_voltage(Ca_in, params, self.CONSTANTS)

        return {f"{prefix}_c": new_c, "CaCon_i": Ca_in, f"{prefix}_vCa": vCa}

    def compute_current(self, u, voltages, params):
        """Return current."""
        prefix = self._name
        cs = u[f"{prefix}_c"]
        # vCa = self.compute_voltage(u, params)
        # Multiply with 1000 to convert Siemens to milli Siemens.
        ca_conds = params[f"{prefix}_gCa"] * (cs**3) * 1000  # mS/cm^2
        current = ca_conds * (voltages - u[f"{prefix}_vCa"])
        return current

    @staticmethod
    def c_gate(e):
        alpha = 0.3 * efun(-(e + 13), 10)
        beta = 10 * jnp.exp(-(e + 38) / 18)
        return alpha, beta

    @staticmethod
    def compute_voltage(Ca_in, params, CONSTANTS):
        """Return voltage."""
        vCa = (CONSTANTS["R"] * CONSTANTS["T"] / (2 * CONSTANTS["F"])) * jnp.log(
            params["CaCon_e"] / Ca_in
        )
        return vCa * 1000  # mV


class KCa(Channel):
    "Calcium-dependent ligand gated potassium channel"

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gKCa": 36e-3,  # S/cm^2
            "vK": -75.0,  # mV
            "CaDiss": 1e-4,  # mM (calcium concentration for half-maximal activation)
        }

        self.channel_states = {"CaCon_i": 1e-4}
        self.meta = META

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state based on calcium concentration."""
        return {}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages: float, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        # Multiply with 1000 to convert Siemens to milli Siemens.
        x = (u["CaCon_i"] / params["CaDiss"]) ** 2
        k_conds = params[f"{prefix}_gKCa"] * x / (1 + x)  # mS/cm^2
        # print(k_conds)
        return k_conds * (voltages - params["vK"])
