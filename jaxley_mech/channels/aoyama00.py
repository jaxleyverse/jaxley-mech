from typing import Dict, Optional, Union

import jax.debug
import jax.numpy as jnp
from jax import Array
from jax.lax import select
from jaxley.channels import Channel
from jaxley.solver_gate import (exponential_euler, save_exp,
                                solve_gate_exponential)

META = {
    "cell_type": "horizontal cell",
    "species": "rabbit",
    "reference": "Aoyama et al. (2000)",
    "doi": "https://doi.org/10.1016/S0168-0102(00)00111-5",
}


class Leak(Channel):
    """Leakage current"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gLeak": 0.5e-3,  # S/cm^2
            f"{prefix}_eLeak": -80.0,  # mV
        }
        self.channel_states = {}
        self.current_name = f"iLeak"
        self.META = META

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """No state to update."""
        return {}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Given channel states and voltage, return the current through the channel."""
        prefix = self._name
        gLeak = params[f"{prefix}_gLeak"]  # S/cm^2
        return gLeak * (voltage - params[f"{prefix}_eLeak"])  # S/cm^2 * mV = mA/cm^2

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the state such at fixed point of gate dynamics."""
        return {}


class Na(Channel):
    """Sodium channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNa": 2.4e-3,  # S/cm^2
            f"{prefix}_eNa": 55.0,  # mV
        }
        self.channel_states = {f"{prefix}_m": 0.026, f"{prefix}_h": 0.922}
        self.current_name = f"iNa"
        self.META = META
        self.META.update({"ion": "Na"})

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Return the updated states."""
        prefix = self._name
        delta_t /= 1000
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        new_m = solve_gate_exponential(m, delta_t, *Na.m_gate(voltage))
        new_h = solve_gate_exponential(h, delta_t, *Na.h_gate(voltage))
        return {f"{prefix}_m": new_m, f"{prefix}_h": new_h}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Given channel states and voltage, return the current through the channel."""
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        gNa = params[f"{prefix}_gNa"] * (m**3) * h  # S/cm^2

        return gNa * (voltage - params[f"{prefix}_eNa"])  # S/cm^2 * mV = mA/cm^2

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = self.m_gate(voltage)
        alpha_h, beta_h = self.h_gate(voltage)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    @staticmethod
    def m_gate(v):
        v += 1e-6
        alpha = 200 * (38 - v) / (save_exp((38 - v) / 25) - 1)
        beta = 2000 * save_exp(-(55 + v) / 18)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        v += 1e-6
        alpha = 1000 * save_exp(-(v + 80.0) / 8.0)
        beta = 800 / (save_exp((80 - v) / 75) + 1)
        return alpha, beta


class Kdr(Channel):
    """Delayed Rectifying Potassium Channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gKdr": 4.5e-3,  # S/cm^2
            "eK": -80,  # mV
        }
        self.channel_states = {
            f"{self._name}_m": 0.139,  # Initial value for m gating variable
            f"{self._name}_h": 0.932,  # Initial value for h gating variable
        }
        self.current_name = f"iKdr"
        self.META = META
        self.META.update({"ion": "K"})

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Update state of gating variables."""
        prefix = self._name
        delta_t /= 1000
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        m_new = solve_gate_exponential(m, delta_t, *self.m_gate(voltage))
        h_new = solve_gate_exponential(h, delta_t, *self.h_gate(voltage))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        h = states[f"{prefix}_h"]
        k_cond = params[f"{prefix}_gKdr"] * m**4 * h  # S/cm^2
        return k_cond * (voltage - params["eK"])  # S/cm^2 * mV = mA/cm^2

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = self.m_gate(voltage)
        alpha_h, beta_h = self.h_gate(voltage)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the n gating variable."""
        v += 1e-6
        alpha = 0.4 * (65 - v) / (save_exp((65 - v) / 50) - 1)
        beta = 4.8 * save_exp((45 - v) / 85)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        v += 1e-6
        alpha = 1500 / (save_exp((v + 92) / 7) + 1)
        beta = 0.02 + 80 / (save_exp((v + 100) / 15) + 1)
        return alpha, beta


class Kto(Channel):
    """Transient Outward Potassium Channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gKto": 15e-3,  # S/cm^2
            "eK": -80,  # mV
        }
        self.channel_states = {
            f"{self._name}_m": 0.139,  # Initial value for m gating variable
            f"{self._name}_h": 0.932,  # Initial value for h gating variable
        }
        self.current_name = f"iKto"
        self.META = META
        self.META.update({"ion": "K"})

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Update state of gating variables."""
        prefix = self._name
        delta_t /= 1000
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        m_new = solve_gate_exponential(m, delta_t, *self.m_gate(voltage))
        h_new = solve_gate_exponential(h, delta_t, *self.h_gate(voltage))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        h = states[f"{prefix}_h"]
        k_cond = params[f"{prefix}_gKto"] * m**3 * h  # S/cm^2
        return k_cond * (voltage - params["eK"])  # S/cm^2 * mV = mA/cm^2

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = self.m_gate(voltage)
        alpha_h, beta_h = self.h_gate(voltage)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the n gating variable."""
        v += 1e-6
        alpha = 2400 / (1 + save_exp(-(v - 50) / 28))
        beta = 80 * save_exp(-v / 36)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        v += 1e-6
        alpha = save_exp(-v / 60)
        beta = 20 / (save_exp(-(v + 40) / 5) + 1)
        return alpha, beta


class Kar(Channel):
    """Anomalous rectifying Potassium Channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gKar": 4.5e-3,  # S/cm^2
            "eK": -80,  # mV
        }
        self.channel_states = {
            f"{self._name}_m": 0.139,  # Initial value for m gating variable
        }
        self.current_name = f"iKar"
        self.META = META
        self.META.update({"ion": "K"})

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Update state of gating variables."""
        prefix = self._name
        delta_t /= 1000
        m_new = self.m_gate(voltage)
        return {f"{prefix}_m": m_new}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        k_cond = params[f"{prefix}_gKar"] * m**5  # S/cm^2
        return k_cond * (voltage - params["eK"])  # S/cm^2 * mV = mA/cm^2

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        m = self.m_gate(voltage)
        return {f"{prefix}_m": m}

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the n gating variable."""
        v += 1e-6
        m = 1 / (1 + save_exp((v + 60) / 12))
        return m


class Ca(Channel):
    """L-type calcium channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gCa": 9e-3,  # S/cm^2
            "eCa": 54.176,  # mV, =12.9 * log(2/0.03)
        }
        self.channel_states = {
            f"{self._name}_m": 0.059,  # Initial value for m gating variable
        }
        self.current_name = f"iCa"
        self.META = META
        self.META.update({"ion": "Ca"})

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Update state of gating variables."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        delta_t /= 1000  # convert to seconds
        m_new = solve_gate_exponential(m, delta_t, *self.m_gate(voltage))
        return {f"{prefix}_m": m_new}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        ca_cond = params[f"{prefix}_gCa"] * m**4  # S/cm^2
        current = ca_cond * (voltage - params["eCa"])  # S/cm^2 * mV = mA/cm^2
        return current

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = self.m_gate(voltage)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        v += 1e-6
        alpha = 240 * (68 - v) / (save_exp((68 - v) / 21) - 1)
        beta = 800 / (save_exp((55 + v) / 55) + 1)
        return alpha, beta
