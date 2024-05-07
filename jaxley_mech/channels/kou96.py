from typing import Dict, Optional

import jax.numpy as jnp
from jax.debug import print
from jax.lax import select
from jaxley.channels import Channel
from jaxley.solver_gate import save_exp, solve_gate_exponential

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
            f"{prefix}_gLeak": 0.5e-3,  # S/cm^2
            f"{prefix}_eLeak": -55.0,  # mV
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
        """Given channel states and voltage, return the current through the channel."""
        prefix = self._name
        gLeak = (
            params[f"{prefix}_gLeak"] * 1000
        )  # mS/cm^2, multiply with 1000 to convert Siemens to milli Siemens.
        return gLeak * (v - params[f"{prefix}_eLeak"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        return {}


class Kv(Channel):
    """Delayed Rectifier Potassium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gKv": 2e-3,  # S/cm^2
            "eK": -80.0,  # mV
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
        h = states[f"{prefix}_h"]
        new_m = solve_gate_exponential(m, dt, *Kv.m_gate(v))
        new_h = solve_gate_exponential(h, dt, *Kv.h_gate(v))
        return {f"{prefix}_m": new_m, f"{prefix}_h": new_h}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """""Return the updated states.""" ""
        prefix = self._name
        m = states[f"{prefix}_m"]
        h = states[f"{prefix}_h"]
        gKv = (
            params[f"{prefix}_gKv"] * (m**3) * h * 1000
        )  # mS/cm^2, multiply with 1000 to convert Siemens to milli Siemens.

        return gKv * (v - params[f"eK"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = Kv.m_gate(v)
        alpha_h, beta_h = Kv.h_gate(v)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    @staticmethod
    def m_gate(v):
        v += 1e-6  # Avoid division by zero
        alpha = 5 * (100 - v) / (save_exp((100 - v) / 42) - 1)
        beta = 9 * save_exp(-(v - 20) / 40)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        v += 1e-6  # Avoid division by zero
        alpha = 0.15 * save_exp(-v / 22)
        beta = 0.4125 / (save_exp((10 - v) / 7) + 1)
        return alpha, beta


class Ca(Channel):

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gCa": 1.2e-3,  # S/cm^2
        }
        self.channel_states = {
            f"{prefix}_m": 0.0,  # Initial value for m gating variable
            f"{prefix}_h": 0.0,  # Initial value for h gating variable
            "eCa": 0.0,
        }
        self.current_name = f"i_Ca"
        self.META = META

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables."""
        prefix = self._name
        ms, hs = states[f"{prefix}_m"], states[f"{prefix}_h"]
        m_new = solve_gate_exponential(ms, dt, *self.m_gate(voltages))
        h_new = self.h_gate(voltages)
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new, "eCa": states["eCa"]}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ms, hs = states[f"{prefix}_m"], states[f"{prefix}_h"]
        ca_cond = params[f"{prefix}_gCa"] * (ms**4) * hs * 1000
        current = ca_cond * (voltages - states["eCa"])
        return current

    def init_state(self, voltages, params):
        prefix = self._name
        alpha_m, beta_m = self.m_gate(voltages)
        h = self.h_gate(voltages)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": h,
            "eCa": 0.0,
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable, adjusted for junction potential."""
        v = v + 1e-6
        alpha = 300 * (80 - v) / (save_exp((80 - v) / 25) - 1)
        beta = 1000 / (1 + save_exp((v + 38) / 7))
        return alpha, beta

    @staticmethod
    def h_gate(v):
        v = v + 1e-6
        h = save_exp((40 - v) / 18) / (1 + save_exp((40 - v) / 18))
        return h


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
        self.channel_params = {}
        self.channel_states = {"eCa": 0.0, "CaCon_i": 5e-05, "CaCon_e": 2.0}
        self.current_name = f"i_Ca"

    def update_states(self, states, dt, voltages, params):
        """Update internal calcium concentration based on calcium current and decay."""
        R, T, F = (
            self.channel_constants["R"],
            self.channel_constants["T"],
            self.channel_constants["F"],
        )
        Cai = states["CaCon_i"]
        Cao = states["CaCon_e"]
        C = R * T / (2 * F) * 1000  # mV
        eCa = C * jnp.log(Cao / Cai)
        return {"eCa": eCa, "CaCon_i": Cai, "CaCon_e": Cao}

    def compute_current(self, u, voltages, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(self, voltages, params):
        """Initialize the state at fixed point of gate dynamics."""
        return {}


class CaPump(Channel):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_V_sub": 3.14e-6,  # Volume of submembrane area in liters
            f"{self._name}_V_i": 5.236e-10,  # Volume of deep intracellular area in liters
            f"{self._name}_D_ca": 3e-4,  # Diffusion coefficient for Ca2+ in cm^2/s
            f"{self._name}_Lb1": 0.4,  # Binding rate to low affinity buffer in 1/(s * µM)
            f"{self._name}_Lb2": 0.2,  # Unbinding rate from low affinity buffer in s^-1
            f"{self._name}_Hb1": 100,  # Binding rate to high affinity buffer in 1/(s * µM)
            f"{self._name}_Hb2": 50,  # Unbinding rate from high affinity buffer in s^-1
            f"{self._name}_BL": 500,  # Total concentration of high-affinity buffer in µM
            f"{self._name}_BH": 300,  # Total concentration of low-affinity buffer in µM
            f"{self._name}_I_max": 20,  # Maximum Na-Ca exchanger current in pA
            f"{self._name}_I_max2": 30,  # Maximum Ca-ATPase pump current in pA
            f"CaCon_e": 1.2,  # Extracellular Ca2+ concentration in µM
        }
        self.channel_states = {
            f"CaCon_sub": 0.05,  # Submembrane Ca2+ concentration in µM
            f"CaCon_i": 0.05,  # Intracellular free Ca2+ concentration in µM
            f"CaCon_hs": 0.05,  # Ca2+ bound to high-affinity buffer in submembrane area in µM
            f"CaCon_hf": 0.05,  # Ca2+ bound to high-affinity buffer in deep intracellular area in µM
            f"CaCon_ls": 0.05,  # Ca2+ bound to low-affinity buffer in submembrane area in µM
            f"CaCon_lf": 0.05,  # Ca2+ bound to low-affinity buffer in deep intracellular area in µM
        }
        self.current_name = f"i_Ca"

    def update_states(self, states, dt, voltages, params):
        """Update internal calcium concentration based on complex dynamics."""

        # Constants
        FARADAY = 96485  # Coulombs per mole
        S1 = 2 * FARADAY * params[f"{self._name}_V_sub"]
        S2 = 2 * FARADAY * params[f"{self._name}_V_i"]

        # Retrieve variables from states and inputs
        Ca_sub = states[f"CaCon_sub"]
        Ca_i = states[f"CaCon_i"]
        Ca_e = params["CaCon_e"]
        Ca_hs = states[f"CaCon_hs"]
        Ca_hf = states[f"CaCon_hf"]
        Ca_ls = states[f"CaCon_ls"]
        Ca_lf = states[f"CaCon_lf"]

        I_ca = states["i_Ca"]
        I_max = params[f"{self._name}_I_max"]
        I_max2 = params[f"{self._name}_I_max2"]

        # Calculate i_ex and i_ex2 based on provided formulas
        i_ex = (
            I_max
            * save_exp(-(voltages + 14) / 70)
            * ((Ca_sub - Ca_e) / ((Ca_sub - Ca_e) + 2.3))
        )

        i_ex2 = I_max2 * ((Ca_sub - Ca_e) / ((Ca_sub - Ca_e) + 0.5))

        # Differential equations based on the model
        dCa_sub_dt = (
            (I_ca - i_ex + i_ex2) / S1
            - params[f"{self._name}_D_ca"] * (Ca_sub - Ca_i)
            + params[f"{self._name}_Lb1"] * (params[f"{self._name}_BL"] - Ca_ls) * Ca_i
            - params[f"{self._name}_Lb2"] * Ca_ls
            + params[f"{self._name}_Hb1"] * (params[f"{self._name}_BH"] - Ca_hs) * Ca_i
            - params[f"{self._name}_Hb2"] * Ca_hs
        )

        dCa_i_dt = (
            params[f"{self._name}_D_ca"] * (Ca_sub - Ca_i)
            - params[f"{self._name}_Lb1"] * (params[f"{self._name}_BL"] - Ca_lf) * Ca_i
            + params[f"{self._name}_Lb2"] * Ca_lf
            - params[f"{self._name}_Hb1"] * (params[f"{self._name}_BH"] - Ca_hf) * Ca_i
            + params[f"{self._name}_Hb2"] * Ca_hf
        )

        # Update states
        states[f"CaCon_sub"] += dCa_sub_dt * dt
        states[f"CaCon_i"] += dCa_i_dt * dt
        states[f"CaCon_hs"] += (
            -params[f"{self._name}_Hb2"] * Ca_hs
            + params[f"{self._name}_Hb1"] * (params[f"{self._name}_BH"] - Ca_hs) * Ca_i
        ) * dt
        states[f"CaCon_hf"] += (
            -params[f"{self._name}_Hb2"] * Ca_hf
            + params[f"{self._name}_Hb1"] * (params[f"{self._name}_BH"] - Ca_hf) * Ca_i
        ) * dt
        states[f"CaCon_ls"] += (
            -params[f"{self._name}_Lb2"] * Ca_ls
            + params[f"{self._name}_Lb1"] * (params[f"{self._name}_BL"] - Ca_ls) * Ca_i
        ) * dt
        states[f"CaCon_lf"] += (
            -params[f"{self._name}_Lb2"] * Ca_lf
            + params[f"{self._name}_Lb1"] * (params[f"{self._name}_BL"] - Ca_lf) * Ca_i
        ) * dt

        return states

    def compute_current(self, u, voltages, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(self, voltages, params):
        """Initialize the state at fixed point of gate dynamics."""
        return {}


class KCa(Channel):
    "Calcium-dependent ligand gated potassium channel"

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gKCa": 0.5e-3,  # S/cm^2
            "eK": -80.0,  # mV
        }
        self.channel_constants = {
            "Cad": 1e-3,  # mM (calcium concentration for half-maximal activation; dissociation constant)
        }
        self.channel_states = {"Cai": 1e-4}
        self.current_name = f"i_K"
        self.META = META

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state based on calcium concentration."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        new_m = solve_gate_exponential(m, dt, *KCa.m_gate(v))
        new_n = KCa.n_gate(states["Cai"])
        return {f"{prefix}_m": new_m, f"{prefix}_n": new_n}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v: float, params: Dict[str, jnp.ndarray]
    ):
        """Given channel states and voltage, return the current through the channel."""
        prefix = self._name
        m = states[f"{prefix}_m"]  # mKc
        n = states[f"{prefix}_n"]  # mKc1
        gKCa = (
            params[f"{prefix}_gKCa"] * m**2 * n * 1000
        )  # mS/cm^2, multiply with 1000 to convert Siemens to milli Siemens.
        return gKCa * (v - params["eK"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        return {}

    @staticmethod
    def m_gate(v):
        v += 1e-6  # Avoid division by zero
        alpha = 15 * (80 - v) / (save_exp((80 - v) / 40) - 1)
        beta = 20 * save_exp(-v / 35)
        return alpha, beta

    @staticmethod
    def n_gate(Cai):
        return Cai / (Cai + 0.3)


class ClCa(Channel):
    "Calcium-dependent Chloride channel"

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gClCa": 6.5e-3,  # S/cm^2
            "eCl": -45.0,  # mV
            "Clh": 0.37,  # μM
        }
        self.channel_states = {
            "Cai": 1e-4,  # μM
        }
        self.current_name = f"i_Cl"
        self.META = META

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state based on calcium concentration."""
        prefix = self._name
        Cai = states["Cai"]
        Cah = params["Clh"]
        new_m = ClCa.m_gate(Cah, Cai)
        return {f"{prefix}_m": new_m}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v: float, params: Dict[str, jnp.ndarray]
    ):
        """Given channel states and voltage, return the current through the channel."""
        prefix = self._name
        m = states[f"{prefix}_m"]  # mCl
        gKCa = (
            params[f"{prefix}_gKCa"] * m * 1000
        )  # mS/cm^2, multiply with 1000 to convert Siemens to milli Siemens.
        return gKCa * (v - params["eK"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        return {}

    @staticmethod
    def m_gate(Clh, Cai):
        m = 1 / (1 + save_exp((Clh - Cai) / 0.09))
        return m
