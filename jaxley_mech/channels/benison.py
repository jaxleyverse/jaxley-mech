from typing import Dict, Optional

import jax.debug
import jax.numpy as jnp
from jax.lax import select
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
        self.current_name = f"iLeak"
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

    def init_state(self, states, v, params, delta_t):
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
        self.channel_states = {f"{prefix}_m": 0.1, f"{prefix}_h": 0.0}
        self.current_name = f"iNa"
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

    def init_state(self, states, v, params, delta_t):
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
        v += 1e-6
        alpha = 0.5 * (v + 29.0) / (1 - save_exp(-0.18 * (v + 29.0)))
        beta = 6.0 * save_exp(-(v + 45.0) / 15.0)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        v += 1e-6
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
        self.current_name = f"iK"
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """""Return the updated states.""" ""
        prefix = self._name
        m = states[f"{prefix}_m"]
        new_m = solve_gate_exponential(m, dt, *Kdr.m_gate(v))
        # jax.debug.print("new_m={new_m}, v={v}", new_m=new_m, v=v)
        return {f"{prefix}_m": new_m}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """""Return the updated states.""" ""
        prefix = self._name
        m = states[f"{prefix}_m"]
        gKdr = (
            params[f"{prefix}_gKdr"] * (m**4) * 1000
        )  # mS/cm^2, multiply with 1000 to convert Siemens to milli Siemens.

        return gKdr * (v - params[f"eK"])

    def init_state(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = Kdr.m_gate(v)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
        }

    @staticmethod
    def m_gate(v):
        v += 1e-6

        # original rate constants do not work
        # alpha = 0.0065 * (v + 30.0) / (1.0 - jnp.exp(-0.3 * v))
        # beta = 0.083 * jnp.exp(-(v + 15.0) / 15.0)

        # modified rate constant
        alpha = 0.0065 * (v + 30.0) / (1.0 - jnp.exp(-(v + 30.0) / 10.0))
        beta = 0.083 * jnp.exp(-(v + 15.0) / 15.0)
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
        self.current_name = f"iK"
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
        new_h = solve_inf_gate_exponential(h, dt, *KA.h_gate(v))
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

    def init_state(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = KA.m_gate(v)
        h_inf, _ = KA.h_gate(v)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": h_inf,
        }

    @staticmethod
    def m_gate(v):
        v += 1e-6
        alpha = 0.02 * (v + 15) / (1 - save_exp(-0.12 * (v + 15)) + 1e-6)
        beta = 0.05 * save_exp(-(v + 1.0) / 30.0)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        v += 1e-6
        h_inf = 1.0 / (1.0 + save_exp((v + 62.0) / 6.35))
        tau_h = jnp.array(25.0)  # ms, fixed time constant
        return h_inf, tau_h


class CaL(Channel):
    """L-type Calcium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gCaL": 2e-3,  # S/cm^2
        }
        self.channel_states = {f"{prefix}_m": 0.1, "eCa": 125.0}
        self.current_name = f"iCa"
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
        current = gCaL * (v - states["eCa"])
        return current

    def init_state(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = CaL.m_gate(v)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
        }

    @staticmethod
    def m_gate(v):
        v += 1e-6
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
        }
        self.channel_states = {f"{prefix}_m": 1.0, f"{prefix}_h": 0.0, "eCa": 125.0}
        self.current_name = f"iCa"
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
        return gCaN * (v - states[f"eCa"])

    def init_state(self, states, v, params, delta_t):
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
        v += 1e-6
        alpha = 0.1 * (v - 20.0) / (1.0 - save_exp(-0.1 * (v - 20.0)))
        beta = 0.4 * save_exp(-(v + 25.0) / 18.0)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        v += 1e-6
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
            "Ceq": 1e-4,  # mM (or C_eq, resting calcium concentration, 0.1 μM = 1e-4 mM)
            "tau_store": 12.5,  # ms (characteristic relaxation time)
            "K_pump": 1e-4,  # mM (or K_pump, equilibrium calcium value, calcium dissociation constant)
            "v_pump": 0.0072e-3,  # mM/ms (pump rate)
            "fi": 0.025,  # (dimensionless, fraction of free calcium in the cytoplasm)
        }
        self.channel_states = {
            "Cai": 1e-4  # mM (global internal calcium concentration)
        }
        self.current_name = f"iCa"
        self.META = META

    def update_states(self, states, dt, v, params):
        """Update internal calcium concentration due to the pump action."""
        F = self.channel_constants["F"]

        V_cell = (
            jnp.pi * params["radius"] ** 2 * params["length"]
        )  # volume of the cell, assuming cylindrical
        # V_cell = (
        #     4 * jnp.pi * (params["radius"] ** 3) / 3
        # )  # volume of the cell, assuming spherical
        fi = params["fi"]
        tau_store = params["tau_store"]
        Cai = states["Cai"]  # C in eq(6)
        Ceq = params["Ceq"]

        v_pump = params["v_pump"]
        K_pump = params["K_pump"]
        j_pump = v_pump * (Cai**4 / (Cai**4 + K_pump**4))

        iCa = states["iCa"] / 1_000

        driving_channel = -10_000.0 * iCa / (2 * F * V_cell)
        driving_channel = select(
            driving_channel <= 0.0, jnp.zeros_like(driving_channel), driving_channel
        )
        dCa_dt = driving_channel - (Cai - Ceq) / tau_store - j_pump
        new_Cai = Cai + fi * dCa_dt * dt

        return {"Cai": new_Cai}

    def compute_current(self, states, v, params):
        """The pump does not directly contribute to the membrane current."""
        return 0

    def init_state(self, states, v, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        return {}


class KCa(Channel):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gKCa": 2e-3,  # S/cm^2
            "K_KCa": 0.6e-3,  # mM
            "eK": -85.0,  # mV
        }
        self.channel_states = {"Cai": 1e-4}  # mM, intracellular calcium concentration
        self.current_name = f"iK"
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
        Cai = states["Cai"]
        K_KCa = params["K_KCa"]  # mM

        # current resulted from the fourth-power will be too small
        # given the values of Cai and K_KCa provided in the paper
        # gKCa = (
        #     params[f"{prefix}_gKCa"] * (Cai**4 / (Cai**4 + K_KCa**4)) * 1000
        # )  # mS/cm^2, multiply with 1000 to convert Siemens to milli Siemens.

        # changed to the second-power from fm97
        x = (Cai / K_KCa) ** 2
        gKCa = params[f"{prefix}_gKCa"] * (x / (1 + x)) * 1000

        return gKCa * (v - params[f"eK"])

    def init_state(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        return {}


class CaNernstReversal(Channel):
    """Compute Calcium reversal from inner and outer concentration of calcium."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.channel_constants = {
            "F": 96485.3329,  # C/mol (Faraday's constant)
            "T": 279.45,  # Kelvin (temperature)
            "R": 8.314,  # J/(mol K) (gas constant)
        }
        self.channel_params = {"Cao": 2.0}
        self.channel_states = {"eCa": 125.0, "Cai": 5e-05}
        self.current_name = f"iCa"

    def update_states(self, states, dt, v, params):
        """Update internal calcium concentration based on calcium current and decay."""
        R, T, F = (
            self.channel_constants["R"],
            self.channel_constants["T"],
            self.channel_constants["F"],
        )
        Cai = states["Cai"]
        Cao = params["Cao"]
        C = R * T / (2 * F) * 1000  # mV
        eCa = C * jnp.log(Cao / Cai)
        return {"eCa": eCa, "Cai": Cai}

    def compute_current(self, states, v, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(self, states, v, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        return {}
