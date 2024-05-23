from typing import Dict, Optional

import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley.solver_gate import (
    save_exp,
    solve_gate_exponential,
    solve_inf_gate_exponential,
)

from ..utils import efun

__all__ = ["Leak", "Na", "Kdr", "KA", "CaL", "CaN", "CaPumpNS", "KCa"]

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
            f"{prefix}_gLeak": 0.25e-3,  # S/cm^2
            f"{prefix}_eLeak": -60.0,  # mV
        }
        self.channel_states = {}
        self.current_name = f"i_Leak"
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """No state to update."""
        return {}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """""Return the updated states.""" ""
        prefix = self._name
        gLeak = (
            params[f"{prefix}_gLeak"] * 1000
        )  # mS/cm^2, multiply with 1000 to convert Siemens to milli Siemens.
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
            f"{prefix}_gNa": 150e-3,  # S/cm^2
            f"{prefix}_eNa": 75.0,  # mV
        }
        self.channel_states = {f"{prefix}_m": 0.2, f"{prefix}_h": 0.2}
        self.current_name = f"i_Na"
        self.META = META

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Return the updated states."""
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        new_m = solve_gate_exponential(m, dt, *Na.m_gate(v))
        new_h = solve_gate_exponential(h, dt, *Na.h_gate(v))
        return {f"{prefix}_m": new_m, f"{prefix}_h": new_h}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return the updated states."""
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        gNa = (
            params[f"{prefix}_gNa"] * (m**3) * h * 1000
        )  # mS/cm^2, multiply with 1000 to convert Siemens to milli Siemens.
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
        alpha = 0.5 * (v + 29.0) / (1 - save_exp(-0.18 * (v + 29.0)) + 1e-6)
        beta = 6.0 * save_exp(-(v + 45.0) / 15.0)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        alpha = 0.15 * save_exp(-(v + 47.0) / 20.0)
        beta = 2.8 / (1.0 + save_exp(-0.1 * (v + 20.0)))
        return alpha, beta


class Kdr(Channel):
    """Delayed Rectifier Potassium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gKdr": 75e-3,  # S/cm^2
            "eK": -85.0,  # mV
        }
        self.channel_states = {f"{prefix}_m": 0.1}
        self.current_name = f"i_K"
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """""Return the updated states.""" ""
        prefix = self._name
        m = states[f"{prefix}_m"]
        new_m = solve_gate_exponential(m, dt, *Kdr.m_gate(v))
        return {f"{prefix}_m": new_m}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """""Return the updated states.""" ""
        prefix = self._name
        m = states[f"{prefix}_m"]
        gKdr = (
            params[f"{prefix}_gKdr"] * (m**3) * 1000
        )  # mS/cm^2, multiply with 1000 to convert Siemens to milli Siemens.

        return gKdr * (v - params[f"eK"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = Kdr.m_gate(v)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
        }

    @staticmethod
    def m_gate(v):
        alpha = 0.0065 * (v + 30) / (1.0 - save_exp(-0.3 * v))
        beta = 0.083 * save_exp((v + 15.0) / 15.0)
        return alpha, beta


class KA(Channel):
    """A type Potassium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gKA": 1.5e-3,  # S/cm^2
            "eK": -85.0,  # mV
        }
        self.channel_states = {f"{prefix}_m": 0.2, f"{prefix}_h": 0.2}
        self.current_name = f"i_K"
        self.META = META

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Return the updated states."""
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        new_m = solve_gate_exponential(m, dt, *KA.m_gate(v))
        new_h = solve_inf_gate_exponential(h, dt, KA.h_gate(v), jnp.array(25.0))
        return {f"{prefix}_m": new_m, f"{prefix}_h": new_h}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return the updated states."""
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        gKA = (
            params[f"{prefix}_gKA"] * (m**3) * h * 1000
        )  # mS/cm^2, multiply with 1000 to convert Siemens to milli Siemens.
        return gKA * (v - params[f"eK"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = KA.m_gate(v)
        alpha_h, beta_h = KA.h_gate(v)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    @staticmethod
    def m_gate(v):
        alpha = 0.02 * (v + 15) / (1 - save_exp(-0.12 * (v + 15)) + 1e-6)
        beta = 0.05 * save_exp(-(v + 1.0) / 30.0)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        h_inf = 1.0 / (1.0 + save_exp((v + 62.0) / 6.35))
        return h_inf


class CaL(Channel):
    """L-type Calcium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gCaL": 2e-3,  # S/cm^2
            "eCa": 45.0,  # mV
        }
        self.channel_states = {f"{prefix}_m": 0.1}
        self.current_name = f"i_Ca"
        self.META = META

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Return the updated states."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        new_m = solve_gate_exponential(m, dt, *CaL.m_gate(v))
        return {
            f"{prefix}_m": new_m,
        }

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return the updated states."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        gCaL = (
            params[f"{prefix}_gCaL"] * (m**2) * 1000
        )  # mS/cm^2, multiply with 1000 to convert Siemens to milli Siemens.
        current = gCaL * (v - params[f"eCa"])
        return current

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = CaL.m_gate(v)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
        }

    @staticmethod
    def m_gate(v):
        alpha = 0.061 * (v - 3.0) / (1.0 - save_exp(-(v - 3.0) / 12.5))
        beta = 0.058 * save_exp(-(v - 10.0) / 15.0)
        return alpha, beta


class CaN(Channel):
    """N-type Ca channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gCaN": 1.5e-3,  # S/cm^2
            "eCa": 45.0,  # mV
        }
        self.channel_states = {f"{prefix}_m": 0.2, f"{prefix}_h": 0.2}
        self.current_name = f"i_Ca"
        self.META = META

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Return the updated states."""
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        new_m = solve_gate_exponential(m, dt, *CaN.m_gate(v))
        new_h = solve_gate_exponential(h, dt, *CaN.h_gate(v))
        return {f"{prefix}_m": new_m, f"{prefix}_h": new_h}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return the updated states."""
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        gCaN = (
            params[f"{prefix}_gCaN"] * (m**2) * h * 1000
        )  # mS/cm^2, multiply with 1000 to convert Siemens to milli Siemens.
        return gCaN * (v - params[f"eCa"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = CaN.m_gate(v)
        alpha_h, beta_h = CaN.h_gate(v)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    @staticmethod
    def m_gate(v):
        alpha = 0.1 * (v - 20.0) / (1.0 - save_exp(-0.1 * (v - 20.0)) + 1e-6)
        beta = 0.4 * save_exp(-(v + 25.0) / 18.0)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        alpha = 0.01 * save_exp(-(v + 50.0) / 10.0)
        beta = 0.1 / (1.0 + save_exp(-(v + 17.0) / 17.0))
        return alpha, beta


class CaPumpNS(Channel):
    """Non-spatial Calcium pump"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_constants = {
            "F": 96485.3329,  # C/mol (Faraday's constant)
        }
        self.channel_params = {
            "Cad": 1e-4,  # mM (or K_pump, equilibrium calcium value, calcium dissociation constant)
            "Cab": 1e-4,  # mM (or C_eq, resting calcium concentration)
            "tau_store": 12.5,  # m (characteristic relaxation time)
            "fi": 1.0,  # (dimensionless, fraction of free calcium in the cytoplasm)
            "v_pump": 0.0072e-3,  # mM/m (pump rate)
        }
        self.channel_states = {
            "Cai": 1e-4  # mM (global internal calcium concentration)
        }
        self.current_name = f"i_Ca"
        self.META = META

    def update_states(self, states, dt, v, params):
        """Update internal calcium concentration due to the pump action."""
        F = self.channel_constants["F"]

        V_cell = jnp.pi * params["radius"] ** 2 * params["length"]
        fi = params["fi"]
        tau = params["tau_store"]
        Cai = states["Cai"]
        Cab = params["Cab"]

        v_pump = params["v_pump"]
        K_pump = params["Cad"]
        j_pump = v_pump * (Cai**2 / (Cai**2 + K_pump**2))

        ca_current = states["i_Ca"]

        driving_channel = -ca_current / (2 * F * V_cell)
        driving_channel = jnp.maximum(driving_channel, 0.0)
        dCa_dt = driving_channel - (Cai - Cab) / tau - j_pump
        new_Cai = Cai + fi * dCa_dt * dt

        return {"Cai": new_Cai}

    def compute_current(self, states, v, params):
        """The pump does not directly contribute to the membrane current."""
        return 0


class KCa(Channel):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gKCa": 2e-3,  # S/cm^2
            "K_KCa": 0.6e-3,  # mM
            "eK": -85.0,  # mV
        }
        self.channel_states = {"Cai": 0.0001}  # mM, intracellular calcium concentration
        self.current_name = f"i_K"
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """""Return the updated states.""" ""
        return {}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """""Return the updated states.""" ""
        prefix = self._name
        C = states["Cai"]
        K_KCa = params["K_KCa"]  # mM
        gKCa = (
            params[f"{prefix}_gKCa"] * (C**4 / (C**4 + K_KCa**4)) * 1000
        )  # mS/cm^2, multiply with 1000 to convert Siemens to milli Siemens.
        return gKCa * (v - params[f"eK"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        return {}
