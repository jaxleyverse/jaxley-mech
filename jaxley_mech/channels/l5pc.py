from typing import Dict, Optional

import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley.solver_gate import solve_gate_exponential, solve_inf_gate_exponential

from ..utils import efun


class NaTaT(Channel):
    """Transient sodium current from Colbert and Pan, 2002."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNaTaT": 0.00001,  # S/cm^2
            "ena": 50.0,  # mV
        }
        self.channel_states = {
            f"{prefix}_m": 0.1,  # Initial value for m gating variable
            f"{prefix}_h": 0.1,  # Initial value for h gating variable
        }
        self.META = {
            "reference": "Colbert and Pan, 2002",
            "species": "unknown",
            "cell_type": "Layer 5 pyramidal cell",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages))
        h_new = solve_inf_gate_exponential(hs, dt, *self.h_gate(voltages))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        gNaTaT = params[f"{prefix}_gNaTaT"] * 1000
        current = gNaTaT * (ms**3) * hs * (voltages - params["ena"])
        return current

    def init_state(self, voltages, params):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages)
        h_inf, _ = self.h_gate(voltages)
        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_h": h_inf,
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        alpha = (0.182 * (v + 38 + 1e-6)) / (1 - jnp.exp(-(v + 38 + 1e-6) / 6))
        beta = (0.124 * (-v - 38 + 1e-6)) / (1 - jnp.exp(-(-v - 38 + 1e-6) / 6))
        m_inf = alpha / (alpha + beta)
        tau_m = 1 / (alpha + beta) / (2.3 ** ((34 - 21) / 10))
        return m_inf, tau_m

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable."""
        alpha = (-0.015 * (v + 66 + 1e-6)) / (1 - jnp.exp((v + 66 + 1e-6) / 6))
        beta = (-0.015 * (-v - 66 + 1e-6)) / (1 - jnp.exp((-v - 66 + 1e-6) / 6))
        h_inf = alpha / (alpha + beta)
        tau_h = 1 / (alpha + beta) / (2.3 ** ((34 - 21) / 10))
        return h_inf, tau_h


class NaTs2T(Channel):
    """Transient sodium current from Colbert and Pan, 2002.
    Almost exactly the same as NaTaT, but with different voltage-dependent dynamics.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNaTs2T": 0.00001,  # S/cm^2
            "ena": 50.0,  # mV
        }
        self.channel_states = {
            f"{prefix}_m": 0.1,  # Initial value for m gating variable
            f"{prefix}_h": 0.1,  # Initial value for h gating variable
        }
        self.META = {
            "reference": "Colbert and Pan, 2002",
            "species": "unknown",
            "cell_type": "Layer 5 pyramidal cell",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages))
        h_new = solve_inf_gate_exponential(hs, dt, *self.h_gate(voltages))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        gNaTaT = params[f"{prefix}_gNaTaT"] * 1000
        current = gNaTaT * (ms**3) * hs * (voltages - params["ena"])
        return current

    def init_state(self, voltages, params):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages)
        h_inf, _ = self.h_gate(voltages)
        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_h": h_inf,
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        alpha = (0.182 * (v + 32 + 1e-6)) / (1 - jnp.exp(-(v + 32 + 1e-6) / 6))
        beta = (0.124 * (-v - 32 + 1e-6)) / (1 - jnp.exp(-(-v - 32 + 1e-6) / 6))
        m_inf = alpha / (alpha + beta)
        tau_m = 1 / (alpha + beta) / (2.3 ** ((34 - 21) / 10))
        return m_inf, tau_m

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable."""
        alpha = (-0.015 * (v + 60 + 1e-6)) / (1 - jnp.exp((v + 60 + 1e-6) / 6))
        beta = (-0.015 * (-v - 60 + 1e-6)) / (1 - jnp.exp((-v - 60 + 1e-6) / 6))
        h_inf = alpha / (alpha + beta)
        tau_h = 1 / (alpha + beta) / (2.3 ** ((34 - 21) / 10))
        return h_inf, tau_h


class NapEt2(Channel):
    """Persistent sodium current from Magistretti & Alonso 1999.

    Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNapEt2": 0.00001,  # S/cm^2
            "ena": 50,  # mV
        }
        self.channel_states = {}
        self.META = {
            "reference": "Magistretti and Alonso 1999",
            "species": "unknown",
            "cell_type": "Layer 5 pyramidal cell",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages))
        h_new = solve_inf_gate_exponential(hs, dt, *self.h_gate(voltages))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        gNaTaT = params[f"{prefix}_gNapEt2"] * 1000
        current = gNaTaT * (ms**3) * hs * (voltages - params["ena"])
        return current

    def init_state(self, voltages, params):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages)
        h_inf, _ = self.h_gate(voltages)
        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_h": h_inf,
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        alpha = (0.182 * (v + 38 + 1e-6)) / (1 - jnp.exp(-(v + 38 + 1e-6) / 6))
        beta = (0.124 * (-v - 38 + 1e-6)) / (1 - jnp.exp(-(-v - 38 + 1e-6) / 6))
        tau_m = 6 / (alpha + beta) / (2.3 ** ((34 - 21) / 10))
        m_inf = 1.0 / (1 + jnp.exp((v + 52.7) / -4.6))

        return m_inf, tau_m

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable."""
        alpha = (-2.88e-6 * (v + 17 + 1e-6)) / (1 - jnp.exp((v + 17 + 1e-6) / 4.63))
        beta = (-6.94e-6 * (v + 64.4 + 1e-6)) / (1 - jnp.exp((-(v + 64.4) + 1e-6) / 6))
        tau_h = 1 / (alpha + beta) / (2.3 ** ((34 - 21) / 10))
        h_inf = 1.0 / (1 + jnp.exp((v + 48.8) / 10))

        return h_inf, tau_h


class CaPump(Channel):
    """Calcium dynamics tracking inside calcium concentration, modeled after Destexhe et al. 1994."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gamma": 0.05,  # Fraction of free calcium (not buffered)
            f"{self._name}_decay": 80,  # Rate of removal of calcium in ms
            f"{self._name}_depth": 0.1,  # Depth of shell in um
            f"{self._name}_minCai": 1e-4,  # Minimum intracellular calcium concentration in mM
        }
        self.channel_states = {
            f"CaCon_i": 1e-4,  # Initial internal calcium concentration in mM
        }
        self.META = {
            "reference": "Modified from Destexhe et al., 1994",
            "mechanism": "Calcium dynamics",
        }

    def update_states(self, u, dt, voltages, params):
        """Update internal calcium concentration based on calcium current and decay."""
        prefix = self._name
        ica = u["CaHVA_current"] + u["CaLVA_current"]
        cai = u["CaCon_i"]
        gamma = params[f"{prefix}_gamma"]
        decay = params[f"{prefix}_decay"]
        depth = params[f"{prefix}_depth"]
        minCai = params[f"{prefix}_minCai"]

        FARADAY = 96485  # Coulombs per mole

        # Calculate the contribution of calcium currents to cai change
        drive_channel = -ica * gamma / (2 * FARADAY * depth)

        # Update cai considering decay towards minCai
        new_cai = cai + dt * (drive_channel - (cai - minCai) / decay)

        # Ensure cai does not go below minCai
        new_cai = jnp.maximum(new_cai, minCai)

        return {f"CaCon_i": new_cai}

    def compute_current(self, u, voltages, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0


class CaHVA(Channel):
    """High-Voltage-Activated (HVA) Ca2+ channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gCaHVA": 0.00001,  # S/cm^2
            "eca": 45.0,  # mV, assuming eca for demonstration
        }
        self.channel_states = {
            f"{self._name}_m": 0.1,  # Initial value for m gating variable
            f"{self._name}_h": 0.1,  # Initial value for h gating variable
        }
        self.META = {
            "reference": "Reuveni, Friedman, Amitai, and Gutnick, J.Neurosci. 1993",
            "mechanism": "HVA Ca2+ channel",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        m_new = solve_gate_exponential(ms, dt, *self.m_gate(voltages))
        h_new = solve_gate_exponential(hs, dt, *self.h_gate(voltages))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        gCaHVA = params[f"{prefix}_gCaHVA"]
        current = gCaHVA * (ms**2) * hs * (voltages - params["eca"])
        return current

    def init_state(self, voltages, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = self.m_gate(voltages)
        alpha_h, beta_h = self.h_gate(voltages)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        alpha = (0.055 * (-27 - v + 1e-6)) / (jnp.exp((-27.0 - v + 1e-6) / 3.8) - 1.0)
        beta = 0.94 * jnp.exp((-75.0 - v + 1e-6) / 17.0)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable."""
        alpha = 0.000457 * jnp.exp((-13.0 - v) / 50.0)
        beta = 0.0065 / (jnp.exp((-v - 15.0) / 28.0) + 1.0)
        return alpha, beta


class CaLVA(Channel):
    """Low-Voltage-Activated (LVA) Ca2+ channel, based on Avery and Johnston 1996 and Randall 1997"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gCaLVA": 0.00001,  # S/cm^2
            "eca": 45.0,  # mV, assuming eca for demonstration
        }
        self.channel_states = {
            f"{self._name}_m": 0.0,  # Initial value for m gating variable
            f"{self._name}_h": 0.0,  # Initial value for h gating variable
        }
        self.META = {
            "reference": "Based on Avery and Johnston 1996 and Randall 1997",
            "mechanism": "LVA Ca2+ channel",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages))
        h_new = solve_inf_gate_exponential(hs, dt, *self.h_gate(voltages))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        gCaLVA = params[f"{prefix}_gCaLVA"]
        current = gCaLVA * (ms**2) * hs * (voltages - params["eca"])
        return current

    def init_state(self, voltages, params):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages)
        h_inf, _ = self.h_gate(voltages)
        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_h": h_inf,
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable, adjusted for junction potential."""
        qt = 2.3 ** ((34 - 21) / 10)  # Q10 temperature correction
        v_shifted = v + 10  # Shift by 10 mV
        m_inf = 1.0 / (1 + jnp.exp((v_shifted + 30) / -6))
        tau_m = (5.0 + 20.0 / (1 + jnp.exp((v_shifted + 25) / 5))) / qt
        return m_inf, tau_m

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable, adjusted for junction potential."""
        qt = 2.3 ** ((34 - 21) / 10)  # Q10 temperature correction
        v_shifted = v + 10  # Shift by 10 mV
        h_inf = 1.0 / (1 + jnp.exp((v_shifted + 80) / 6.4))
        tau_h = (20.0 + 50.0 / (1 + jnp.exp((v_shifted + 40) / 7))) / qt
        return h_inf, tau_h
