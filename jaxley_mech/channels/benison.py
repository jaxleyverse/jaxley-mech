from typing import Dict, Optional

import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley.solver_gate import solve_gate_exponential, solve_inf_gate_exponential

from ..utils import efun

META = {
    "reference": "Benison, G., Keizer, J., Chalupa, L. M., & Robinson, D. W. (2001). Modeling Temporal Behavior of Postnatal Cat Retinal Ganglion Cells. Journal of Theoretical Biology, 210(2), 187–199. https://doi.org/10.1006/jtbi.2000.2289",
    "species": "Cat",
    "cell_type": "Retinal ganglion cells",
}


class Leak(Channel):
    """Leakage current"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gl": 0.25e-3,  # S/cm^2
            f"{prefix}_el": -60.0,  # mV
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
            f"{prefix}_gNa": 150e-3,  # S/cm^2
            f"{prefix}_vNa": 75.0,  # mV
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
        alpha = 0.5 * (e + 29.0) / (1 - jnp.exp(-0.18 * (e + 29.0)) + 1e-6)
        beta = 6.0 * jnp.exp(-(e + 45.0) / 15.0)
        return alpha, beta

    @staticmethod
    def h_gate(e):
        alpha = 0.15 * jnp.exp(-(e + 47.0) / 20.0)
        beta = 2.8 / (1.0 + jnp.exp(-0.1 * (e + 20.0)))
        return alpha, beta


class Kdr(Channel):
    """Delayed Rectifier Potassium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gKdr": 75e-3,  # S/cm^2
            "vK": -85.0,  # mV
        }
        self.channel_states = {f"{prefix}_m": 0.1}
        self.META = META

    def update_states(
        self, u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""
        prefix = self._name
        ms = u[f"{prefix}_m"]
        new_m = solve_gate_exponential(ms, dt, *Kdr.m_gate(voltages))
        return {f"{prefix}_m": new_m}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        ms = u[f"{prefix}_m"]

        # Multiply with 1000 to convert Siemens to milli Siemens.
        kdr_conds = params[f"{prefix}_gKdr"] * (ms**3) * 1000  # mS/cm^2

        return kdr_conds * (voltages - params[f"vK"])

    def init_state(self, voltages, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = Kdr.m_gate(voltages)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
        }

    @staticmethod
    def m_gate(e):
        alpha = 0.0065 * (e + 30) / (1.0 - jnp.exp(-0.3 * e))
        beta = 0.083 * jnp.exp((e + 15.0) / 15.0)
        return alpha, beta


class KA(Channel):
    """A type Potassium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gKA": 1.5e-3,  # S/cm^2
            "vK": 45.0,  # mV
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
        m_new = solve_gate_exponential(ms, dt, *KA.m_gate(voltages))
        h_new = solve_inf_gate_exponential(hs, dt, KA.h_gate(voltages), jnp.array(25.0))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        "Return current."
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        KA_conds = params[f"{prefix}_gKA"] * (ms**3) * hs * 1000  # mS/cm^2
        current = KA_conds * (voltages - params[f"vK"])
        return current

    def init_state(self, voltages, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = KA.m_gate(voltages)
        alpha_h, beta_h = KA.h_gate(voltages)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    @staticmethod
    def m_gate(e):
        alpha = 0.02 * (e + 15) / (1 - jnp.exp(-0.12 * (e + 15)) + 1e-6)
        beta = 0.05 * jnp.exp(-(e + 1.0) / 30.0)
        return alpha, beta

    @staticmethod
    def h_gate(e):
        h_inf = 1.0 / (1.0 + jnp.exp((e + 62.0) / 6.35))
        return h_inf


class CaL(Channel):
    """L-type Calcium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gCaL": 2e-3,  # S/cm^2
            "vCa": 45.0,  # mV
        }
        self.channel_states = {f"{prefix}_m": 0.1}
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
        ms = u[f"{prefix}_m"]
        m_new = solve_gate_exponential(ms, dt, *CaL.m_gate(voltages))
        return {
            f"{prefix}_m": m_new,
        }

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        "Return current."
        prefix = self._name
        ms = u[f"{prefix}_m"]
        CaL_conds = params[f"{prefix}_gCaL"] * (ms**2) * 1000  # mS/cm^2
        current = CaL_conds * (voltages - params[f"vCa"])
        return current

    def init_state(self, voltages, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = CaL.m_gate(voltages)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
        }

    @staticmethod
    def m_gate(e):
        alpha = 0.061 * (e - 3.0) / (1.0 - jnp.exp(-(e - 3.0) / 12.5))
        beta = 0.058 * jnp.exp(-(e - 10.0) / 15.0)
        return alpha, beta


class CaN(Channel):
    """N-type Ca channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gCaN": 1.5e-3,  # S/cm^2
            "vCa": 45.0,  # mV
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
        m_new = solve_gate_exponential(ms, dt, *CaN.m_gate(voltages))
        h_new = solve_gate_exponential(hs, dt, *CaN.h_gate(voltages))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        "Return current."
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        CaN_conds = params[f"{prefix}_gCaN"] * (ms**2) * hs * 1000  # mS/cm^2
        current = CaN_conds * (voltages - params[f"vCa"])
        return current

    def init_state(self, voltages, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = CaN.m_gate(voltages)
        alpha_h, beta_h = CaN.h_gate(voltages)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    @staticmethod
    def m_gate(e):
        alpha = 0.1 * (e - 20.0) / (1.0 - jnp.exp(-0.1 * (e - 20.0)) + 1e-6)
        beta = 0.4 * jnp.exp(-(e + 25.0) / 18.0)
        return alpha, beta

    @staticmethod
    def h_gate(e):
        alpha = 0.01 * jnp.exp(-(e + 50.0) / 10.0)
        beta = 0.1 / (1.0 + jnp.exp(-(e + 17.0) / 17.0))
        return alpha, beta


class CaPumpNS(Channel):
    """Non-spatial Calcium pump"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_constants = {
            "F": 96485.3329,  # C/mol (Faraday's constant)
        }
        self.channel_params = {
            "CaCon_diss": 1e-4,  # mM (K_pump, equilibrium calcium value, calcium dissociation constant)
            "CaCon_rest": 1e-4,  # mM (C_eq, resting calcium concentration)
            "tau_store": 12.5,  # ms (characteristic relaxation time)
            "fi": 1.0,  # (dimensionless, fraction of free calcium in the cytoplasm)
            "v_pump": 0.0072e-3,  # mM/ms (pump rate)
        }
        self.channel_states = {
            "CaCon_i": 1e-4  # mM (global internal calcium concentration)
        }
        self.META = META

    def update_states(self, u, dt, voltages, params):
        """Update internal calcium concentration due to the pump action."""
        F = self.channel_constants["F"]

        V_cell = jnp.pi * params["radius"] ** 2 * params["length"]
        fi = params["fi"]
        tau = params["tau_store"]
        C = u["CaCon_i"]
        C_eq = params["CaCon_rest"]

        v_pump = params["v_pump"]
        K_pump = params["CaCon_diss"]
        j_pump = v_pump * (C**2 / (C**2 + K_pump**2))

        CaN_current = u["CaN_current"]
        CaL_current = u["CaL_current"]
        ca_current = CaN_current + CaL_current

        driving_channel = -ca_current / (2 * F * V_cell)
        driving_channel = jnp.maximum(driving_channel, 0.0)
        dCa_dt = driving_channel - (C - C_eq) / tau - j_pump
        new_C = C + fi * dCa_dt * dt

        return {"CaCon_i": new_C}

    def compute_current(self, u, voltages, params):
        """The pump does not directly contribute to the membrane current."""
        return 0


class KCa(Channel):

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gKCa": 2e-3,  # S/cm^2
            "K_KCa": 0.6e-3,  # mM
            "vK": -85.0,  # mV
        }
        self.channel_states = {"CaCon_i": 0.0001}
        self.META = META

    def update_states(
        self, u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""
        return {}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        C = u["CaCon_i"]
        K_KCa = params["K_KCa"]  # mM
        KCa_conds = params[f"{prefix}_gKCa"] * (C**4 / (C**4 + K_KCa**4))  # mS/cm^2
        current = KCa_conds * (voltages - params[f"vK"]) * 1000
        return current

    def init_state(self, voltages, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        return {}
