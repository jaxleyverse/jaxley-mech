from typing import Dict, Optional

import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley.solver_gate import solve_gate_exponential

from ..utils import efun

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
            f"{prefix}_gl": 0.05e-3,  # S/cm^2
            f"{prefix}_el": -67.0,  # mV
        }
        self.channel_states = {}
        self.META = META

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

    def init_state(self, voltages, params):
        """Initialize the state such at fixed point of gate dynamics."""
        return {}


class Na(Channel):
    """Sodium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNa": 50e-3,  # S/cm^2
            f"{prefix}_vNa": 35.0,  # mV
        }
        self.channel_states = {f"{prefix}_m": 0.2, f"{prefix}_h": 0.2}
        self.META = META

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

    def init_state(self, voltages, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = Na.m_gate(voltages)
        alpha_h, beta_h = Na.h_gate(voltages)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    @staticmethod
    def m_gate(e):
        alpha = 0.6 * efun(-(e + 30), 10.0)
        beta = 20.0 * jnp.exp(-(e + 55.0) / 18.0)
        return alpha, beta

    @staticmethod
    def h_gate(e):
        alpha = 0.4 * jnp.exp(-(e + 50.0) / 20.0)
        beta = 6.0 / (1 + jnp.exp(-0.1 * (e + 20.0)))
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
        self.META = META

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

    def init_state(self, voltages, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_n, beta_n = K.n_gate(voltages)
        return {
            f"{prefix}_n": alpha_n / (alpha_n + beta_n),
        }

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
        self.META = META

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

    def init_state(self, voltages, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_a, beta_a = KA.A_gate(voltages)
        alpha_ha, beta_ha = KA.hA_gate(voltages)
        return {
            f"{prefix}_a": alpha_a / (alpha_a + beta_a),
            f"{prefix}_ha": alpha_ha / (alpha_ha + beta_ha),
        }

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
            "CaCon_e": 2.0,  # mM (external calcium concentration)
        }
        self.channel_states = {
            f"{prefix}_c": 0.1,
            f"{prefix}_vCa": self.compute_voltage(1e-4, self.channel_params),
            f"CaCon_i": 1e-4,
        }
        self.META = META

    def update_states(self, u, dt, voltages, params):
        """Update state."""
        prefix = self._name
        cs = u[f"{prefix}_c"]
        Cai = u["CaCon_i"]
        ca_current = u[f"{prefix}_current"]
        CaRest = params["CaCon_rest"]
        tau_Ca = params["tau_Ca"]
        new_c = solve_gate_exponential(cs, dt, *Ca.c_gate(voltages))

        dCa_dt = (
            -(2 / params[f"length"] + 2 / params[f"radius"])
            * ca_current
            / (2 * self.channel_constants["F"])
        ) - ((Cai - CaRest) / tau_Ca)
        dCa_dt = jnp.maximum(dCa_dt, 1e-9)  # dCa_dt should not be negative
        Cai += dCa_dt * dt

        vCa = self.compute_voltage(Cai, params)

        return {f"{prefix}_c": new_c, "CaCon_i": Cai, f"{prefix}_vCa": vCa}

    def compute_voltage(self, Cai, params):
        """Return voltage."""
        R, T, F = (
            self.channel_constants["R"],
            self.channel_constants["T"],
            self.channel_constants["F"],
        )
        Cao = params["CaCon_e"]
        C = R * T / (2 * F) * 1000  # mV
        vCa = C * jnp.log(Cao / Cai)
        return vCa

    def compute_current(self, u, voltages, params):
        """Return current."""
        prefix = self._name
        cs = u[f"{prefix}_c"]
        # vCa = self.compute_voltage(u, params)
        # Multiply with 1000 to convert Siemens to milli Siemens.
        ca_conds = params[f"{prefix}_gCa"] * (cs**3) * 1000  # mS/cm^2
        current = ca_conds * (voltages - u[f"{prefix}_vCa"])
        return current

    def init_state(self, voltages, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_c, beta_c = Ca.c_gate(voltages)
        return {f"{prefix}_c": alpha_c / (alpha_c + beta_c)}

    @staticmethod
    def c_gate(e):
        alpha = 0.3 * efun(-(e + 13), 10)
        beta = 10 * jnp.exp(-(e + 38) / 18)
        return alpha, beta


class KCa(Channel):
    "Calcium-dependent ligand gated potassium channel"

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gKCa": 0.05e-3,  # S/cm^2
            "vK": -75.0,  # mV
        }
        self.channel_constants = {
            "CaCon_diss": 1e-3,  # mM (calcium concentration for half-maximal activation)
        }
        self.channel_states = {"CaCon_i": 1e-4}
        self.META = META

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
        x = (u["CaCon_i"] / self.channel_constants["CaCon_diss"]) ** 2
        k_conds = params[f"{prefix}_gKCa"] * x / (1 + x) * 1000  # mS/cm^2
        return k_conds * (voltages - params["vK"])

    def init_state(self, voltages, params):
        """Initialize the state such at fixed point of gate dynamics."""
        return {}
