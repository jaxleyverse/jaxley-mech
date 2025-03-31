from typing import Dict, Optional, Union

import jax.numpy as jnp
from jax.lax import select
from jaxley.mechanisms.channels import Channel
from jaxley.mechanisms.solvers import (
    exponential_euler,
    save_exp,
    solve_gate_exponential,
)

META = {
    "cell_type": "rod photoreceptor",
    "species": "larval tiger salamander",
    "reference": "Liu, et al (2004)",
    "doi": "https://doi.org/10.1114/B:ABME.0000042230.99614.8d",
    "code": "https://modeldb.science/64228",
}


class Leak(Channel):
    """Leakage current"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self.name
        self.params = {
            f"{prefix}_gLeak": 0.52e-3,  # S/cm^2
            f"{prefix}_eLeak": -74.0,  # mV
        }
        self.states = {}
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
        prefix = self.name
        gLeak = params[f"{prefix}_gLeak"]  # S/cm^2 # mS/cm^2
        return gLeak * (v - params[f"{prefix}_eLeak"])  # mS/cm^2 * mV = uA/cm^2

    def init_states(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        return {}


class Kx(Channel):
    """Noninactivating potassium channels"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.params = {
            f"{self.name}_gKx": 1.04e-3,  # S/cm^2
            "eK": -74,  # mV
        }
        self.states = {
            f"{self.name}_n": 0.1,  # Initial value for m gating variable
        }
        self.current_name = f"iKx"
        self.META = META
        self.META.update({"ion": "K"})

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self.name
        ns = states[f"{prefix}_n"]
        n_new = solve_gate_exponential(ns, dt, *self.n_gate(v))
        return {f"{prefix}_n": n_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self.name
        ns = states[f"{prefix}_n"]
        k_cond = params[f"{prefix}_gKx"] * ns  # S/cm^2
        return k_cond * (v - params["eK"])  # S/cm^2 * mV = mA/cm^2

    def init_states(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self.name
        alpha, beta = self.n_gate(v)
        return {f"{prefix}_n": alpha / (alpha + beta)}

    @staticmethod
    def n_gate(v):
        """Voltage-dependent dynamics for the n gating variable."""
        v += 1e-6
        alpha = 6.6e-4 * save_exp((v + 49.9) / 11.4)
        beta = 6.6e-4 * save_exp(-(v + 49.9) / 11.4)
        return alpha, beta


class Kv(Channel):
    """Delayed Rectifying Potassium Channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.params = {
            f"{self.name}_gKv": 10e-3,  # S/cm^2
            "eK": -74,  # mV
        }
        self.states = {
            f"{self.name}_n": 0.1,  # Initial value for n gating variable
        }
        self.current_name = f"iKv"
        self.META = META
        self.META.update({"ion": "K"})

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self.name
        ns = states[f"{prefix}_n"]
        n_new = solve_gate_exponential(ns, dt, *self.n_gate(v))
        return {f"{prefix}_n": n_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self.name
        ns = states[f"{prefix}_n"]
        k_cond = params[f"{prefix}_gKv"] * ns**4  # S/cm^2
        return k_cond * (v - params["eK"])  # S/cm^2 * mV = mA/cm^2

    def init_states(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self.name
        alpha, beta = self.n_gate(v)
        return {f"{prefix}_n": alpha / (alpha + beta)}

    @staticmethod
    def n_gate(v):
        """Voltage-dependent dynamics for the n gating variable."""
        v += 1e-6
        alpha = 0.005 * (20 - v) / (save_exp((20 - v) / 22) - 1)
        beta = 0.0625 * save_exp(-v / 80)
        return alpha, beta


class Hyper(Channel):
    """Rod Hyperpolarization-activated h Channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.params = {
            f"{self.name}_gHyper": 2.5e-3,  # S/cm^2
            f"{self.name}_eHyper": -32,  # mV
        }
        self.states = {
            f"{self.name}_n": 0.000456,
        }
        self.current_name = f"iHyper"
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self.name
        ns = states[f"{prefix}_n"]
        n_new = solve_gate_exponential(ns, dt, *self.n_gate(v))
        return {f"{prefix}_n": n_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self.name
        ns = states[f"{prefix}_n"]
        h_cond = params[f"{prefix}_gHyper"] * ns  # S/cm^2
        return h_cond * (v - params[f"{prefix}_eHyper"])  # S/cm^2 * mV = mA/cm^2

    def init_states(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self.name
        alpha, beta = self.n_gate(v)
        return {f"{prefix}_n": alpha / (alpha + beta)}

    @staticmethod
    def n_gate(v):
        """Voltage-dependent dynamics for the h gating variable."""
        alpha = 0.001 * save_exp(-(v + 75) / 10.6)
        beta = 0.001 * save_exp((v + 75) / 10.6)
        return alpha, beta


class Ca(Channel):
    """L-type calcium channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.params = {
            f"{self.name}_gCa": 4e-3,  # S/cm^2
        }
        self.states = {
            f"{self.name}_m": 0.0,  # Initial value for m gating variable
            f"{self.name}_h": 1.0,  # Initial value for h gating variable
            "eCa": 40.0,  # mV, dependent on CaNernstReversal
        }
        self.current_name = f"iCa"
        self.META = META
        self.META.update({"ion": "Ca"})

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables."""
        prefix = self.name
        ms, hs = states[f"{prefix}_m"], states[f"{prefix}_h"]

        m_new = solve_gate_exponential(ms, dt, *self.m_gate(v))
        h_new = solve_gate_exponential(hs, dt, *self.h_gate(v))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new, "eCa": states["eCa"]}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self.name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        ca_cond = params[f"{prefix}_gCa"] * m * h  # S/cm^2
        current = ca_cond * (v - states["eCa"])  # S/cm^2 * mV = mA/cm^2
        return current

    def init_states(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self.name

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
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.params = {
            f"{self.name}_depth": 0.1,
            f"{self.name}_taur": 20,  # Rate of removal of calcium in ms
            f"{self.name}_cainf": 5e-5,  # mM
        }
        self.states = {
            "Cai": 2e-3,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"iCa"
        self.META = {"reference": "Modified from Destexhe et al., 1994", "ion": "Ca"}

    def update_states(self, states, dt, v, params):
        """Update internal calcium concentration based on calcium current and decay."""
        prefix = self.name
        iCa = states["iCa"]  # Calcium current
        Cai = states["Cai"]  # Internal calcium concentration
        Cai_tau = params[f"{prefix}_taur"]
        Cai_inf = params[f"{prefix}_cainf"]
        depth = params[f"{prefix}_depth"]

        FARADAY = 96485.3329  # Coulombs per mole

        # Calculate the contribution of calcium currents to cai change
        drive_channel = -10_000.0 * iCa / (2 * FARADAY * depth)
        drive_channel = select(
            drive_channel <= 0, jnp.zeros_like(drive_channel), drive_channel
        )

        # Update calcium concentration using exponential_euler
        Cai_inf = Cai_inf + drive_channel / 2 * Cai_tau
        new_Cai = exponential_euler(Cai, dt, Cai_inf, Cai_tau)

        return {"Cai": new_Cai}

    def compute_current(self, states, v, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_states(self, states, v, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        return {"Cai": 2e-3}


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
        self.params = {"Cao": 2.0}
        self.states = {"eCa": 40.0, "Cai": 2e-3}
        self.current_name = f"iCa"
        self.META = META
        self.META.update({"ion": "Ca"})

    def update_states(self, states, dt, v, params):
        """Update internal calcium concentration based on calcium current and decay."""
        R, T, F = (
            self.channel_constants["R"],
            self.channel_constants["T"],
            self.channel_constants["F"],
        )
        Cao = params["Cao"]
        Cai = states["Cai"]
        C = R * T / (2 * F) * 1000
        eCa = C * jnp.log(Cao / Cai)
        return {"eCa": eCa, "Cai": Cai}

    def compute_current(self, states, v, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_states(self, states, v, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        return {"Cai": 2e-3}


class KCa(Channel):
    """Calcium-dependent potassium channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.params = {
            f"{self.name}_gKCa": 5e-3,  # S/cm^2
            f"{self.name}_Khalf": 3.32,  # mM, half-activation concentration
            # with an unfortunate name conflict with potassium K
            "eK": -74,  # mV
        }
        self.states = {
            f"{self.name}_n": 0.1,  # Initial value for n gating variable
            "Cai": 2e-3,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"iKCa"
        self.META = META
        self.META.update({"ion": "K"})

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self.name
        n_new = self.n_gate(states["Cai"], params[f"{prefix}_Khalf"])
        return {f"{prefix}_n": n_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self.name
        ns = states[f"{prefix}_n"]
        k_cond = params[f"{prefix}_gKCa"] * ns**4  # S/cm^2
        return k_cond * (v - params["eK"])  # S/cm^2 * mV = mA/cm^2

    def init_states(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self.name
        Khalf = params[f"{prefix}_Khalf"]
        n = self.n_gate(v, Khalf)
        return {f"{prefix}_n": n, "Cai": 2e-3}

    @staticmethod
    def n_gate(Cai, Khalf):
        """Calcium-dependent n gating variable."""
        return 1 / (1 + (Khalf / Cai) ** 4)


class ClCa(Channel):
    """Calcium-dependent Chloride channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.params = {
            f"{self.name}_gClCa": 1.3e-3,  # S/cm^2
            f"{self.name}_Khalf": 1,  # uM, half-activation concentration
            # with an unfortunate name conflict with potassium K
            f"{self.name}_eClCa": -20,  # mV
        }
        self.states = {
            f"{self.name}_n": 0.1,  # Initial value for n gating variable
            "Cai": 2e-3,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"iClCa"
        self.META = META
        self.META.update({"ion": "Cl"})

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self.name
        n_new = self.n_gate(states["Cai"], params[f"{prefix}_Khalf"])
        return {f"{prefix}_n": n_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self.name
        ns = states[f"{prefix}_n"]
        k_cond = params[f"{prefix}_gClCa"] * ns  # S/cm^2
        return k_cond * (v - params[f"{prefix}_eClCa"])  # S/cm^2 * mV = mA/cm^2

    def init_states(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self.name
        Khalf = params[f"{prefix}_Khalf"]
        n = self.n_gate(v, Khalf)
        return {f"{prefix}_n": n, "Cai": 0.002}

    @staticmethod
    def n_gate(Cai, Khalf):
        """Calcium-dependent n gating variable."""
        return 1 / (1 + (Khalf / Cai) ** 4)
