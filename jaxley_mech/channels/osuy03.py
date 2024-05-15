from typing import Dict, Optional

import jax.debug
import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley.solver_gate import save_exp, solve_gate_exponential

META = {
    "reference": "Ogura, T., Satoh, T.-O., Usui, S., & Yamada, M. (2003). A simulation analysis on mechanisms of damped oscillation in retinal rod photoreceptor cells. Vision Research, 43(19), 2019–2028. https://doi.org/10.1016/S0042-6989(03)00309-2",
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
        self.current_name = f"iLeak"

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


class h(Channel):
    """Hyperpolarization-activated channel in the formulation of Markov model with 5 states"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gh": 5.825e-3,  # S/cm^2
            f"{prefix}_eh": -32.0,  # mV
        }
        self.channel_states = {
            f"{prefix}_C1": 1.0,
            f"{prefix}_C2": 0,
            f"{prefix}_O1": 0,
            f"{prefix}_O2": 0,
            f"{prefix}_O3": 0,
        }
        self.current_name = f"ih"
        self.META = {
            "reference": [
                # Add references if applicable
            ],
            "Species": "generic",
        }

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""
        prefix = self._name
        alpha_h, beta_h = self.h_gate(v)

        # Transition rates according to the matrix K

        C1_to_C2 = 4 * alpha_h * states[f"{prefix}_C1"]
        C2_to_O1 = 3 * alpha_h * states[f"{prefix}_C2"]
        O1_to_O2 = 2 * alpha_h * states[f"{prefix}_O1"]
        O2_to_O3 = alpha_h * states[f"{prefix}_O2"]

        O3_to_O2 = 4 * beta_h * states[f"{prefix}_O3"]
        O2_to_O1 = 3 * beta_h * states[f"{prefix}_O2"]
        O1_to_C2 = 2 * beta_h * states[f"{prefix}_O1"]
        C2_to_C1 = beta_h * states[f"{prefix}_C2"]

        new_C1 = states[f"{prefix}_C1"] + dt * (C2_to_C1 - C1_to_C2)
        new_C2 = states[f"{prefix}_C2"] + dt * (
            C1_to_C2 + O1_to_C2 - C2_to_O1 - C2_to_C1
        )
        new_O1 = states[f"{prefix}_O1"] + dt * (
            C2_to_O1 + O2_to_O1 - O1_to_O2 - O1_to_C2
        )
        new_O2 = states[f"{prefix}_O2"] + dt * (
            O1_to_O2 + O3_to_O2 - O2_to_O3 - O2_to_O1
        )
        new_O3 = states[f"{prefix}_O3"] + dt * (O2_to_O3 - O3_to_O2)

        # jax.debug.print(
        #     "sum={sum:.5f}\tnew_C1={new_C1:.5f}\tnew_C2={new_C2:.5f}\tnew_O1={new_O1:.5f}\tnew_O2={new_O2:.5f}\tnew_O3={new_O3:.5f}",
        #     sum=new_C1 + new_C2 + new_O1 + new_O2 + new_O3,
        #     new_C1=new_C1,
        #     new_C2=new_C2,
        #     new_O1=new_O1,
        #     new_O2=new_O2,
        #     new_O3=new_O3,
        # )

        return {
            f"{prefix}_C1": new_C1,
            f"{prefix}_C2": new_C2,
            f"{prefix}_O1": new_O1,
            f"{prefix}_O2": new_O2,
            f"{prefix}_O3": new_O3,
        }

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        O1 = states[f"{prefix}_O1"]
        O2 = states[f"{prefix}_O2"]
        O3 = states[f"{prefix}_O3"]
        gh = params[f"{prefix}_gh"] * (O1 + O2 + O3) * 1000
        return gh * (v - params[f"{prefix}_eh"])

    @staticmethod
    def h_gate(v):
        v += 1e-6
        alpha = (
            1e-3 * 10.6 / (save_exp((v + 93.62) / 16.47) + 1)
        )  # 1/ms, differ from the paper, which prop used 1/s
        beta = 1e-3 * 15.624 / (save_exp(-(v + 16.243) / 26.81) + 1)
        return alpha, beta


class Kv(Channel):
    """Slow and Fast Delayed Rectifying Potassium Channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gKv1": 9.6e-3,  # S/cm^2
            f"{self._name}_gKv2": 1.0e-3,  # S/cm^2
            "eK": -86.6,  # mV
        }
        self.channel_states = {
            f"{self._name}_m1": 0.1,  # Initial value for n gating variable
            f"{self._name}_m2": 0.1,  # Initial value for n gating variable
            f"{self._name}_h": 0.1,  # Initial value for n gating variable
        }
        self.current_name = f"iKv"
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self._name
        m1s = states[f"{prefix}_m1"]
        hs = states[f"{prefix}_h"]
        m1_new = solve_gate_exponential(m1s, dt, *self.m1_gate(v))
        m2_new = self.m2_gate(v)
        h_new = solve_gate_exponential(hs, dt, *self.h_gate(v))
        return {
            f"{prefix}_m1": m1_new,
            f"{prefix}_m2": m2_new,
            f"{prefix}_h": h_new,
        }

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m1 = states[f"{prefix}_m1"]
        m2 = states[f"{prefix}_m2"]
        h = states[f"{prefix}_h"]
        k_cond = (
            (params[f"{prefix}_gKv1"] * m1**3 + params[f"{prefix}_gKv2"] * m2**3)
            * h
            * 1000
        )
        return k_cond * (v - params["eK"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m1, beta_m1 = self.m1_gate(v)
        alpha_h, beta_h = self.h_gate(v)

        return {
            f"{prefix}_m1": alpha_m1 / (alpha_m1 + beta_m1),
            f"{prefix}_m2": self.m2_gate(v),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    @staticmethod
    def m1_gate(v):
        """Voltage-dependent dynamics for the , gating variable."""
        v += 1e-6
        alpha = 1e-3 * 2 * (210.4 - v) / (save_exp((80.54 - v) / 45) - 1)
        beta = 1e-3 * 14.5 * save_exp(-(v - 36.5) / 37)
        return alpha, beta

    @staticmethod
    def m2_gate(v):
        v += 1e-6
        return 1e-3 / (1 + save_exp(-(v + 15) / 5))

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable."""
        v += 1e-6
        alpha = 1e-3 * 1.2 * save_exp(-(v - 20) / 21.9)
        beta = 1e-3 * 1.316 / (save_exp(-(v - 47.5) / 22) + 1)
        return alpha, beta


class Ca(Channel):
    """L-type calcium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gCa": 1e-3,  # S/cm^2
        }
        self.channel_states = {
            f"{self._name}_m": 0.1,  # Initial value for m gating variable
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
        ms = states[f"{prefix}_m"]
        m_new = solve_gate_exponential(ms, dt, *self.m_gate(v))
        return {f"{prefix}_m": m_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        ca_cond = params[f"{prefix}_gCa"] * m**4 * 1000
        current = ca_cond * (v - states["eCa"])
        return current

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = self.m_gate(v)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            "eCa": 40.0,
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        v += 1e-6
        alpha = 0.3 * (30 - v) / (save_exp((30 - v) / 15) - 1)
        beta = 1 / (save_exp((v + 30) / 12.7) + 1)
        return alpha, beta


class CaPump(Channel):
    """Calcium dynamics tracking inside calcium concentration, modeled after Destexhe et al. 1994."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_depth": 0.1,  # Depth of shell in um
            f"{self._name}_Cai_tau": 200,  # Rate of removal of calcium in ms
            f"{self._name}_Cai_inf": 5e-5,  # mM
            "Cao": 3.0,  # External calcium concentration in mM
        }
        self.channel_states = {
            "Cai": 2e-3,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"iCa"
        self.META = {
            "reference": "Modified from Destexhe et al., 1994",
            "mechanism": "Calcium dynamics",
        }

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update internal calcium concentration based on calcium current and decay."""
        prefix = self._name
        iCa = states["iCa"] / 1_000  # Calcium current

        Cai = states["Cai"]  # Internal calcium concentration

        depth = params[f"{prefix}_depth"]
        Cai_tau = params[f"{prefix}_Cai_tau"]
        Cai_inf = params[f"{prefix}_Cai_inf"]

        FARADAY = 96485  # Coulombs per mole

        # Calculate the contribution of calcium currents to cai change
        drive_channel = -10_000.0 * iCa / (2 * FARADAY * depth)
        dCai_dt = drive_channel / 2 + (Cai_inf - Cai) / Cai_tau
        Cai += dCai_dt * dt
        return {"Cai": Cai}  # Convert to scalar

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
        self.channel_params = {"Cao": 3.0}
        self.channel_states = {"eCa": 40.0, "Cai": 2e-3}
        self.current_name = f"iCa"

    def update_states(self, states, dt, v, params):
        """Update internal calcium concentration based on calcium current and decay."""

        Cao = params["Cao"]
        Cai = states["Cai"]
        eCa = 12.9 * jnp.log(Cao / Cai)
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
            f"{self._name}_gKCa": 13e-3,  # S/cm^2
            # with an unfortunate name conflict with potassium K
            "eK": -86.6,  # mV
        }
        self.channel_states = {
            f"{self._name}_m": 0.1,  # Initial value for m gating variable
            "Cai": 2e-3,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"iKCa"
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self._name
        m_new = self.m_gate(states["Cai"])
        return {f"{prefix}_m": m_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        k_cond = params[f"{prefix}_gKCa"] * m * 1000
        current = k_cond * (v - params["eK"])
        return current

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        m = self.m_gate(2e-3)
        return {f"{prefix}_m": m}

    @staticmethod
    def m_gate(Cai):
        """Calcium-dependent m gating variable."""
        # Cai *= 1e3
        return Cai / (Cai + 12)


class ClCa(Channel):
    """Calcium-dependent Chloride channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gClCa": 16e-3,  # S/cm^2
            f"{self._name}_eCl": -20,  # mV
        }
        self.channel_states = {
            f"{self._name}_m": 0.1,  # Initial value for n gating variable
            "Cai": 2e-3,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"iKCa"
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self._name
        m_new = self.m_gate(states["Cai"])
        return {f"{prefix}_m": m_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        k_cond = params[f"{prefix}_gClCa"] * m * 1000
        return k_cond * (v - params[f"{prefix}_eCl"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        m = self.m_gate(v)
        return {f"{prefix}_m": m}

    @staticmethod
    def m_gate(Cai):
        """Calcium-dependent m gating variable."""
        return Cai / (Cai + 10)
