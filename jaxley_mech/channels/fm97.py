from typing import Dict, Optional

import jax.numpy as jnp
from jax.lax import select
from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler, save_exp, solve_gate_exponential

from ..utils import efun

__all__ = ["Leak", "Na", "K", "KA", "Ca", "CaNernstReversal", "KCa"]

META = {
    "papers": [
        "Fohlmeister, J. F., & Miller, R. F. (1997). Impulse Encoding Mechanisms of Ganglion Cells in the Tiger Salamander Retina. Journal of Neurophysiology, 78(4), 1935–1947. https://doi.org/10.1152/jn.1997.78.4.1935"
    ],
    "species": "Tiger salamander",
    "cell_type": "Retinal ganglion cell",
    "reference": "Fohlmeister & Miller (1997)",
}


class Leak(Channel):
    """Leakage current"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gLeak": 0.05e-3,  # S/cm^2
            f"{prefix}_eLeak": -67.0,  # mV
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
        """Given channel states and voltage, return the current through the channel."""
        prefix = self._name
        gLeak = params[f"{prefix}_gLeak"]  # S/cm^2
        return gLeak * (v - params[f"{prefix}_eLeak"])  # S/cm^2 * mV = mA/cm^2

    def init_state(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        return {}


class Na(Channel):
    """Sodium channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNa": 50e-3,  # S/cm^2
            f"{prefix}_eNa": 35.0,  # mV
        }
        self.channel_states = {f"{prefix}_m": 0.2, f"{prefix}_h": 0.2}
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
        """Given channel states and voltage, return the current through the channel."""
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        gNa = params[f"{prefix}_gNa"] * (m**3) * h  # S/cm^2

        return gNa * (v - params[f"{prefix}_eNa"])  # S/cm^2 * mV = mA/cm^2

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
        alpha = 0.6 * efun(-(v + 30), 10.0)
        beta = 20.0 * save_exp(-(v + 55.0) / 18.0)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        alpha = 0.4 * save_exp(-(v + 50.0) / 20.0)
        beta = 6.0 / (1.0 + save_exp(-0.1 * (v + 20.0)))
        return alpha, beta


class K(Channel):
    """Potassium channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gK": 12e-3,  # S/cm^2
            "eK": -75.0,  # mV
        }
        self.channel_states = {f"{prefix}_n": 0.1}
        self.current_name = f"iK"
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Return the updated states."""
        prefix = self._name
        ns = states[f"{prefix}_n"]
        new_n = solve_gate_exponential(ns, dt, *K.n_gate(v))
        return {f"{prefix}_n": new_n}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Given channel states and voltage, return the current through the channel."""
        prefix = self._name
        n = states[f"{prefix}_n"]
        gK = params[f"{prefix}_gK"] * (n**4)  # S/cm^2
        return gK * (v - params[f"eK"])  # S/cm^2 * mV = mA/cm^2

    def init_state(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_n, beta_n = K.n_gate(v)
        return {
            f"{prefix}_n": alpha_n / (alpha_n + beta_n),
        }

    @staticmethod
    def n_gate(v):
        alpha = 0.02 * efun(-(v + 40), 10)
        beta = 0.4 * save_exp(-(v + 50) / 80)
        return alpha, beta


class KA(Channel):
    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gKA": 36e-3,  # S/cm^2
            f"eK": -75,  # mV
        }
        self.channel_states = {f"{prefix}_A": 0.2, f"{prefix}_hA": 0.2}
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
        A, hA = states[f"{prefix}_A"], states[f"{prefix}_hA"]
        new_A = solve_gate_exponential(A, dt, *KA.A_gate(v))
        new_hA = solve_gate_exponential(hA, dt, *KA.hA_gate(v))
        return {f"{prefix}_A": new_A, f"{prefix}_hA": new_hA}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v: float, params: Dict[str, jnp.ndarray]
    ):
        """Given channel states and voltage, return the current through the channel."""
        prefix = self._name
        A, hA = states[f"{prefix}_A"], states[f"{prefix}_hA"]
        gKA = params[f"{prefix}_gKA"] * (A**3) * hA  # S/cm^2
        return gKA * (v - params[f"eK"])  # S/cm^2 * mV = mA/cm^2

    def init_state(self, states, v, params, delta_t):
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
        beta = 0.1 * save_exp(-(v + 30) / 10)
        return alpha, beta

    @staticmethod
    def hA_gate(v):
        alpha = 0.04 * save_exp(-(v + 70) / 20)
        beta = 0.6 / (1.0 + save_exp(-(v + 40) / 10))
        return alpha, beta


class Ca(Channel):
    """Calcium channel and pump"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_constants = {
            "F": 96485.3329,  # C/mol (Faraday's constant)
            "T": 295.15,  # Kelvin (temperature)
            "R": 8.314,  # J/(mol K) (gas constant)
        }
        self.channel_params = {
            f"{prefix}_gCa": 2.2e-3,  # S/cm^2
        }
        self.channel_states = {
            f"{prefix}_c": 0.1,
            f"eCa": 40.0,  # mV
            f"Cai": 1e-4,  # mM (internal calcium concentration)
        }
        self.current_name = f"iCa"
        self.META = META

    def update_states(self, states, dt, v, params):
        """Return the updated states."""
        prefix = self._name
        cs = states[f"{prefix}_c"]
        new_c = solve_gate_exponential(cs, dt, *Ca.c_gate(v))
        return {f"{prefix}_c": new_c}

    def compute_current(self, states, v, params):
        """Given channel states and voltage, return the current through the channel."""
        prefix = self._name
        c = states[f"{prefix}_c"]
        gCa = params[f"{prefix}_gCa"] * (c**3)  # S/cm^2
        return gCa * (v - states[f"eCa"])  # S/cm^2 * mV = mA/cm^2

    def init_state(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_c, beta_c = Ca.c_gate(v)
        return {f"{prefix}_c": alpha_c / (alpha_c + beta_c)}

    @staticmethod
    def c_gate(v):
        alpha = 0.3 * efun(-(v + 13), 10)
        beta = 10 * save_exp(-(v + 38) / 18)
        return alpha, beta


class CaNernstReversal(Channel):
    """Compute Calcium reversal from inner and outer concentration of calcium."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        self.current_is_in_mA_per_cm2 = True
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


class CaPump(Channel):
    """Calcium ATPase pump modeled after Destexhe et al., 1993/1994."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self.name}_taur": 10,  # Time constant of calcium removal in ms
            f"{self.name}_cainf": 2.4e-4,  # Equilibrium calcium concentration in mM
        }
        self.channel_states = {
            f"Cai": 5e-5,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"iCa"
        self.META = {
            "reference": "Destexhe, A., Babloyantz, A., & Sejnowski, TJ. Ionic mechanisms for intrinsic slow oscillations in thalamic relay neurons. Biophys. J. 65: 1538-1552, 1993.",
            "mechanism": "ATPase pump",
            "source": "https://modeldb.science/3670?tab=2&file=NTW_NEW/capump.mod",
        }

    def update_states(self, states, dt, v, params):
        """Update internal calcium concentration due to pump action and calcium currents."""
        prefix = self._name
        iCa = states[f"iCa"]
        Cai = states[f"Cai"]

        taur = params[f"{prefix}_taur"]
        cainf = params[f"{prefix}_cainf"]

        FARADAY = 96489  # Coulombs per mole

        # Compute inward calcium flow contribution, should not pump inwards
        drive_channel = (
            -10_000.0
            * (2 / params[f"length"] + 2 / params[f"radius"])
            * iCa
            / (2 * FARADAY)
        )

        drive_channel = select(
            drive_channel <= 0, jnp.zeros_like(drive_channel), drive_channel
        )

        # Update internal calcium concentration with exponential euler
        Cai_inf_prime = cainf + drive_channel * taur
        new_Cai = exponential_euler(Cai, dt, Cai_inf_prime, taur)

        return {f"Cai": new_Cai}

    def compute_current(self, states, v, params):
        """The pump does not directly contribute to the membrane current."""
        return 0

    def init_state(self, states, v, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        return {}


class KCa(Channel):
    "Calcium-dependent ligand gated potassium channel"

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gKCa": 0.05e-3,  # S/cm^2
            "eK": -75.0,  # mV
            f"{prefix}_Cad": 1e-3,
        }
        self.channel_states = {"Cai": 1e-4}
        self.current_name = f"iK"
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
        """Given channel states and voltage, return the current through the channel."""
        prefix = self._name
        Cai = states["Cai"]
        Cad = params[f"{prefix}_Cad"]
        x = (Cai / Cad) ** 2
        gKCa = params[f"{prefix}_gKCa"] * x / (1 + x)  # S/cm^2
        return gKCa * (v - params["eK"])  # S/cm^2 * mV = mA/cm^2

    def init_state(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        return {}
