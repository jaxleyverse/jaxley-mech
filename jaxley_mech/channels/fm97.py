from typing import Dict, Optional

import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley.solver_gate import solve_gate_exponential

from ..utils import efun

__all__ = ["Leak", "Na", "K", "KA", "Ca", "KCa"]

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
            f"{prefix}_gLeak": 0.05e-3,  # S/cm^2
            f"{prefix}_eLeak": -67.0,  # mV
        }
        self.channel_states = {}
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """No state to update."""
        return {}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        # Multiply with 1000 to convert Siemens to milli Siemens.
        prefix = self._name
        gLeak = params[f"{prefix}_gLeak"] * 1000  # mS/cm^2
        return gLeak * (v - params[f"{prefix}_eLeak"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        return {}


class Na(Channel):
    """Sodium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNa": 50e-3,  # S/cm^2
            f"{prefix}_eNa": 35.0,  # mV
        }
        self.channel_states = {f"{prefix}_m": 0.2, f"{prefix}_h": 0.2}
        self.META = META

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: float,
        params: Dict[str, jnp.ndarray],
    ):
        "Update state."
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        new_m = solve_gate_exponential(m, dt, *Na.m_gate(v))
        new_h = solve_gate_exponential(h, dt, *Na.h_gate(v))
        return {f"{prefix}_m": new_m, f"{prefix}_h": new_h}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        "Return current."
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        gNa = params[f"{prefix}_gNa"] * (m**3) * h * 1000  # mS/cm^2
        return gNa * (v - params[f"{prefix}_eNa"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = Na.m_gate(v)
        alpha_h, beta_h = Na.h_gate(v)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    @staticmethod
    def m_gate(v):
        alpha = 0.6 * efun(-(v + 30), 10.0)
        beta = 20.0 * jnp.exp(-(v + 55.0) / 18.0)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        alpha = 0.4 * jnp.exp(-(v + 50.0) / 20.0)
        beta = 6.0 / (1.0 + jnp.exp(-0.1 * (v + 20.0)))
        return alpha, beta


class K(Channel):
    """Potassium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gK": 12e-3,  # S/cm^2
            "eK": -75.0,  # mV
        }
        self.channel_states = {f"{prefix}_n": 0.1}
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""
        prefix = self._name
        ns = states[f"{prefix}_n"]
        new_n = solve_gate_exponential(ns, dt, *K.n_gate(v))
        return {f"{prefix}_n": new_n}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        n = states[f"{prefix}_n"]

        # Multiply with 1000 to convert Siemens to milli Siemens.
        gK = params[f"{prefix}_gK"] * (n**4) * 1000  # mS/cm^2

        return gK * (v - params[f"eK"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_n, beta_n = K.n_gate(v)
        return {
            f"{prefix}_n": alpha_n / (alpha_n + beta_n),
        }

    @staticmethod
    def n_gate(v):
        alpha = 0.02 * efun(-(v + 40), 10)
        beta = 0.4 * jnp.exp(-(v + 50) / 80)
        return alpha, beta


class KA(Channel):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gKA": 36e-3,  # S/cm^2
            f"eK": -75,  # mV
        }
        self.channel_states = {f"{prefix}_A": 0.2, f"{prefix}_hA": 0.2}
        self.META = META

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state."""
        prefix = self._name
        A, hA = states[f"{prefix}_A"], states[f"{prefix}_hA"]
        new_A = solve_gate_exponential(A, dt, *KA.A_gate(v))
        new_hA = solve_gate_exponential(hA, dt, *KA.hA_gate(v))
        return {f"{prefix}_A": new_A, f"{prefix}_hA": new_hA}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v: float, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        A, hA = states[f"{prefix}_A"], states[f"{prefix}_hA"]
        gKA = params[f"{prefix}_gKA"] * (A**3) * hA * 1000  # mS/cm^2
        return gKA * (v - params[f"eK"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_a, beta_a = KA.A_gate(v)
        alpha_ha, beta_ha = KA.hA_gate(v)
        return {
            f"{prefix}_A": alpha_a / (alpha_a + beta_a),
            f"{prefix}_hA": alpha_ha / (alpha_ha + beta_ha),
        }

    @staticmethod
    def A_gate(v):
        alpha = 0.006 * efun(-(v + 90), 1)
        beta = 0.1 * jnp.exp(-(v + 30) / 10)
        return alpha, beta

    @staticmethod
    def hA_gate(v):
        alpha = 0.04 * jnp.exp(-(v + 70) / 20)
        beta = 0.6 / (1.0 + jnp.exp(-(v + 40) / 10))
        return alpha, beta


class Ca(Channel):
    """Calcium channel and pump"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_constants = {
            "F": 96485.3329,  # C/mol (Faraday's constant)
            "T": 295.15,  # Kelvin (temperature)
            "R": 8.314,  # J/(mol K) (gas constant)
        }
        self.channel_params = {
            f"{prefix}_gCa": 2.2e-3,  # S/cm^2
            "tau_Ca": 50,  # mS (time constant for calcium removal)
            "CaCon_rest": 1e-4,  # mM (resting internal calcium concentration)
            "CaCon_e": 2.0,  # mM (vxternal calcium concentration)
        }
        self.channel_states = {
            f"{prefix}_c": 0.1,
            f"{prefix}_eCa": self.compute_voltage(1e-4, self.channel_params),
            f"CaCon_i": 1e-4,
        }
        self.META = META

    def update_states(self, states, dt, v, params):
        """Update state."""
        prefix = self._name
        cs = states[f"{prefix}_c"]
        Cai = states["CaCon_i"]
        ca_current = states[f"{prefix}_current"]
        CaRest = params["CaCon_rest"]
        tau_Ca = params["tau_Ca"]
        new_c = solve_gate_exponential(cs, dt, *Ca.c_gate(v))

        driving_channel = (
            -(2 / params[f"length"] + 2 / params[f"radius"])
            * ca_current
            / (2 * self.channel_constants["F"])
        )
        driving_channel = jnp.where(driving_channel <= 0, 0, driving_channel)

        dCa_dt = driving_channel - ((Cai - CaRest) / tau_Ca)
        Cai += dCa_dt * dt

        eCa = self.compute_voltage(Cai, params)

        return {f"{prefix}_c": new_c, "CaCon_i": Cai, f"{prefix}_eCa": eCa}

    def compute_voltage(self, Cai, params):
        """Return voltage."""
        R, T, F = (
            self.channel_constants["R"],
            self.channel_constants["T"],
            self.channel_constants["F"],
        )
        Cao = params["CaCon_e"]
        C = R * T / (2 * F) * 1000  # mV
        eCa = C * jnp.log(Cao / Cai)
        return eCa

    def compute_current(self, states, v, params):
        """Return current."""
        prefix = self._name
        c = states[f"{prefix}_c"]
        # Multiply with 1000 to convert Siemens to milli Siemens.
        gCa = params[f"{prefix}_gCa"] * (c**3) * 1000  # mS/cm^2
        return gCa * (v - states[f"{prefix}_eCa"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_c, beta_c = Ca.c_gate(v)
        return {f"{prefix}_c": alpha_c / (alpha_c + beta_c)}

    @staticmethod
    def c_gate(v):
        alpha = 0.3 * efun(-(v + 13), 10)
        beta = 10 * jnp.exp(-(v + 38) / 18)
        return alpha, beta


class KCa(Channel):
    "Calcium-dependent ligand gated potassium channel"

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gKCa": 0.05e-3,  # S/cm^2
            "eK": -75.0,  # mV
        }
        self.channel_constants = {
            "CaCon_diss": 1e-3,  # mM (calcium concentration for half-maximal activation)
        }
        self.channel_states = {"CaCon_i": 1e-4}
        self.META = META

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state based on calcium concentration."""
        return {}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v: float, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        # Multiply with 1000 to convert Siemens to milli Siemens.
        x = (states["CaCon_i"] / self.channel_constants["CaCon_diss"]) ** 2
        gKCa = params[f"{prefix}_gKCa"] * x / (1 + x) * 1000  # mS/cm^2
        return gKCa * (v - params["eK"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        return {}
