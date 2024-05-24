from typing import Dict, Optional, Union

import jax.numpy as jnp
from jax.debug import print
from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler, save_exp, solve_gate_exponential

META = {
    "cell_type": "rod",
    "reference": "Liu, X.-D., & Kourennyi, D. E. (2004). Effects of Tetraethylammonium on Kx Channels and Simulated Light Response in Rod Photoreceptorss. Annals of Biomedical Engineering, 32(10), 1428–1442. https://doi.org/10.1114/B:ABME.0000042230.99614.8d",
    "code": "https://modeldb.science/64228",
}


class Leak(Channel):
    """Leakage current"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gLeak": 0.52e-3,  # S/cm^2
            f"{prefix}_eLeak": -74.0,  # mV
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
        gLeak = params[f"{prefix}_gLeak"] * 1000  # mS/cm^2
        return gLeak * (v - params[f"{prefix}_eLeak"])  # mS/cm^2 * mV = uA/cm^2

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        return {}


class Kx(Channel):
    """Noninactivating potassium channels"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gKx": 1.04e-3,  # S/cm^2
            "eK": -74,  # mV
        }
        self.channel_states = {
            f"{self._name}_n": 0.1,  # Initial value for m gating variable
        }
        self.current_name = f"iKx"
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self._name
        ns = states[f"{prefix}_n"]
        n_new = solve_gate_exponential(ns, dt, *self.n_gate(v))
        return {f"{prefix}_n": n_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ns = states[f"{prefix}_n"]
        k_cond = params[f"{prefix}_gKx"] * ns * 1000
        return k_cond * (v - params["eK"])  # mS/cm^2 * mV = uA/cm^2

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha, beta = self.n_gate(v)
        return {f"{prefix}_n": alpha / (alpha + beta)}

    @staticmethod
    def n_gate(v):
        """Voltage-dependent dynamics for the n gating variable."""
        v += 1e-6
        alpha = 6.6e-4 * save_exp((v + 50) / 11.4)
        beta = 6.6e-4 * save_exp(-(v + 50) / 11.4)
        return alpha, beta


class Kv(Channel):
    """Delayed Rectifying Potassium Channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gKv": 10e-3,  # S/cm^2
            "eK": -74,  # mV
        }
        self.channel_states = {
            f"{self._name}_n": 0.1,  # Initial value for n gating variable
        }
        self.current_name = f"iKv"
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self._name
        ns = states[f"{prefix}_n"]
        n_new = solve_gate_exponential(ns, dt, *self.n_gate(v))
        return {f"{prefix}_n": n_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ns = states[f"{prefix}_n"]
        k_cond = params[f"{prefix}_gKv"] * ns**4 * 1000
        return k_cond * (v - params["eK"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha, beta = self.n_gate(v)
        return {f"{prefix}_n": alpha / (alpha + beta)}

    @staticmethod
    def n_gate(v):
        """Voltage-dependent dynamics for the n gating variable."""
        v += 1e-6
        alpha = 0.005 * (20 - v) / (save_exp((20 - v) / 22) - 1)
        beta = 0.0625 * save_exp(-v / 80)
        return alpha, beta


class h(Channel):
    """Rod Hyperpolarization-activated h Channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gh": 2.5e-3,  # S/cm^2
            f"{self._name}_eh": -32,  # mV
        }
        self.channel_states = {
            f"{self._name}_n": 0.1,
        }
        self.current_name = f"ih"
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self._name
        ns = states[f"{prefix}_n"]
        n_new = solve_gate_exponential(ns, dt, *self.n_gate(v))
        return {f"{prefix}_n": n_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ns = states[f"{prefix}_n"]
        h_cond = params[f"{prefix}_gh"] * ns * 1000
        return h_cond * (v - params[f"{prefix}_eh"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha, beta = self.n_gate(v)
        return {f"{prefix}_h": alpha / (alpha + beta)}

    @staticmethod
    def n_gate(v):
        """Voltage-dependent dynamics for the h gating variable."""
        alpha = 0.001 * save_exp(-(v + 75) / 10.6)
        beta = 0.001 / save_exp((v + 75) / 10.6)
        return alpha, beta


class Ca(Channel):
    """L-type calcium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gCa": 4e-3,  # S/cm^2
        }
        self.channel_states = {
            f"{self._name}_m": 0.0,  # Initial value for m gating variable
            f"{self._name}_h": 1.0,  # Initial value for h gating variable
            "eCa": 40.0,  # mV, dependent on CaNernstReversal
        }
        self.current_name = f"iCa"
        self.META = META

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables."""
        prefix = self._name
        ms, hs = states[f"{prefix}_m"], states[f"{prefix}_h"]

        m_new = solve_gate_exponential(ms, dt, *self.m_gate(v))
        h_new = solve_gate_exponential(hs, dt, *self.h_gate(v))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new, "eCa": states["eCa"]}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        ca_cond = params[f"{prefix}_gCa"] * m * h * 1000
        current = ca_cond * (v - states["eCa"])
        return current

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name

        alpha_m, beta_m = self.m_gate(v)
        alpha_h, beta_h = self.h_gate(v)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
            "eCa": 40.0,
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        v += 1e-6
        alpha = 0.1 * save_exp((v + 10) / 12.0)
        beta = 0.1 * save_exp(-(v + 10) / 12.0)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable."""
        v += 1e-6
        alpha = 5e-4 * save_exp(-(v - 11) / 18.0)
        beta = 0.01 * save_exp((v - 11) / 18.0)
        return alpha, beta


class CaPump(Channel):
    """Calcium dynamics tracking inside calcium concentration, modeled after Destexhe et al. 1994."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_depth": 10.0,
            f"{self._name}_taur": 20,  # Rate of removal of calcium in ms
            f"{self._name}_cainf": 5e-5,  # mM
        }
        self.channel_states = {
            "Cai": 2e-3,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"iCa"
        self.META = {
            "reference": "Modified from Destexhe et al., 1994",
            "mechanism": "Calcium dynamics",
        }

    def update_states(self, states, dt, v, params):
        """Update internal calcium concentration based on calcium current and decay."""
        prefix = self._name
        iCa = states["iCa"] / 1000  # Calcium current
        Cai = states["Cai"]  # Internal calcium concentration
        Cai_tau = params[f"{prefix}_taur"]
        Cai_inf = params[f"{prefix}_cainf"]
        depth = params[f"{prefix}_depth"]

        FARADAY = 96500  # Coulombs per mole

        # Calculate the contribution of calcium currents to cai change
        drive_channel = -10_000.0 * iCa / (2 * FARADAY * depth)
        drive_channel = jnp.maximum(drive_channel, 0.0)
        dCai_dt = drive_channel / 2 + (Cai_inf - Cai) / Cai_tau
        Cai += dCai_dt * dt
        return {"Cai": Cai}

    def compute_current(self, states, v, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(self, v, params):
        """Initialize the state at fixed point of gate dynamics."""
        return {}


class CaNernstReversal(Channel):
    """Compute Calcium reversal from inner and outer concentration of calcium."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.channel_constants = {
            "F": 96485,  # C/mol (Faraday's constant)
            "T": 279.45,  # Kelvin (temperature)
            "R": 8.314,  # J/(mol K) (gas constant)
        }
        self.channel_params = {"Cao": 2.0}
        self.channel_states = {"eCa": 40.0, "Cai": 2e-3}
        self.current_name = f"iCa"

    def update_states(self, states, dt, v, params):
        """Update internal calcium concentration based on calcium current and decay."""
        R, T, F = (
            self.channel_constants["R"],
            self.channel_constants["T"],
            self.channel_constants["F"],
        )
        Cao = params["Cao"]
        Cai = states["Cai"]
        C = R * T / (2 * F) * 1000  # mV
        eCa = C * jnp.log(Cao / Cai)
        return {"eCa": eCa, "Cai": Cai}

    def compute_current(self, states, v, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(self, v, params):
        """Initialize the state at fixed point of gate dynamics."""
        return {}


class KCa(Channel):
    """Calcium-dependent potassium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gKCa": 5e-3,  # S/cm^2
            f"{self._name}_Khalf": 3.2e-4,  # mM, half-activation concentration
            # with an unfortunate name conflict with potassium K
            "eK": -74,  # mV
        }
        self.channel_states = {
            f"{self._name}_n": 0.1,  # Initial value for n gating variable
            "Cai": 2e-3,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"iKCa"
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self._name
        n_new = self.n_gate(states["Cai"], params[f"{prefix}_Khalf"])
        return {f"{prefix}_n": n_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ns = states[f"{prefix}_n"]
        k_cond = params[f"{prefix}_gKCa"] * ns**4 * 1000
        return k_cond * (v - params["eK"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        Khalf = params[f"{prefix}_Khalf"]
        n = self.n_gate(v, Khalf)
        return {f"{prefix}_n": n}

    @staticmethod
    def n_gate(Cai, Khalf):
        """Calcium-dependent n gating variable."""
        return 1 / (1 + (Khalf / Cai) ** 4)


class ClCa(Channel):
    """Calcium-dependent Chloride channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gClCa": 1.3e-3,  # S/cm^2
            f"{self._name}_Khalf": 1.5e-3,  # mM, half-activation concentration
            # with an unfortunate name conflict with potassium K
            f"{self._name}_eClCa": -20,  # mV
        }
        self.channel_states = {
            f"{self._name}_n": 0.1,  # Initial value for n gating variable
            "Cai": 2e-3,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"iClCa"
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self._name
        n_new = self.n_gate(states["Cai"], params[f"{prefix}_Khalf"])
        return {f"{prefix}_n": n_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ns = states[f"{prefix}_n"]
        k_cond = params[f"{prefix}_gClCa"] * ns * 1000
        return k_cond * (v - params[f"{prefix}_eClCa"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        Khalf = params[f"{prefix}_Khalf"]
        n = self.n_gate(v, Khalf)
        return {f"{prefix}_n": n}

    @staticmethod
    def n_gate(Cai, Khalf):
        """Calcium-dependent n gating variable."""
        return 1 / (1 + (Khalf / Cai) ** 4)
