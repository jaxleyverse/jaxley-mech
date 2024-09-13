from typing import Dict, Optional

import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley.solver_gate import save_exp, solve_gate_exponential

META = {
    "reference": "Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. The Journal of Physiology, 117(4), 500–544. https://doi.org/10.1113/jphysiol.1952.sp004764",
    "type": "squid axon",
    "note": "Unlike the original paper, we adjusted the equations to the modern convention, that is, the resting potential is -65 mV, and signs are flipped.",
}


class Leak(Channel):
    """Leakage current"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gLeak": 0.3e-3,  # S/cm^2
            f"{prefix}_eLeak": -65.0,  # mV
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
        """Return current."""
        # Multiply with 1000 to convert Siemens to milli Siemens.
        prefix = self._name
        leak_conds = params[f"{prefix}_gLeak"] * 1000  # mS/cm^2
        return leak_conds * (v - params[f"{prefix}_eLeak"])

    def init_state(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        return {}


class Na(Channel):
    """Sodium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNa": 40e-3,  # S/cm^2
            f"{prefix}_eNa": 55.0,  # mV
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
        "Update state."
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        new_m = solve_gate_exponential(m, dt, *self.m_gate(v))
        new_h = solve_gate_exponential(h, dt, *self.h_gate(v))

        return {f"{prefix}_m": new_m, f"{prefix}_h": new_h}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        "Return current."
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        gNa = params[f"{prefix}_gNa"] * (m**3) * h * 1000  # mS/cm^2
        current = gNa * (v - params[f"{prefix}_eNa"])
        return current

    def init_state(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = self.m_gate(v)
        alpha_h, beta_h = self.h_gate(v)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    @staticmethod
    def m_gate(v):
        v += 1e-6
        alpha = 0.1 * (v + 40) / (1 - save_exp(-0.1 * (v + 40)))
        beta = 4 * save_exp(-(v + 65) / 18)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        v += 1e-6
        alpha = 0.07 * save_exp(-(v + 65) / 20)
        beta = 1 / (1 + save_exp(-0.1 * (v + 35)))
        return alpha, beta


class K(Channel):
    """Potassium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gK": 35e-3,  # S/cm^2
            f"{prefix}_eK": -77.0,  # mV
        }
        self.channel_states = {f"{prefix}_n": 0.1}
        self.current_name = f"i_K"
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""
        prefix = self._name
        n = states[f"{prefix}_n"]
        new_n = solve_gate_exponential(n, dt, *self.n_gate(v))
        return {f"{prefix}_n": new_n}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        n = states[f"{prefix}_n"]
        gK = params[f"{prefix}_gK"] * (n**4) * 1000  # mS/cm^2
        return gK * (v - params[f"{prefix}_eK"])

    def init_state(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_n, beta_n = self.n_gate(v)
        return {
            f"{prefix}_n": alpha_n / (alpha_n + beta_n),
        }

    @staticmethod
    def n_gate(v):
        v += 1e-6
        alpha = 0.01 * (v + 55) / (1 - save_exp(-(v + 55) / 10))
        beta = 0.125 * save_exp(-(v + 65) / 80)
        return alpha, beta


class Na8States(Na):
    """Sodium channel in the formulation of Markov model with 8 states"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNa": 40e-3,  # S/cm^2
            f"{prefix}_eNa": 55.0,  # mV
        }
        # Initialize all states
        self.channel_states = {
            f"{prefix}_C3": 1.0,
            f"{prefix}_C2": 0.0,
            f"{prefix}_C1": 0.0,
            f"{prefix}_O": 0,
            f"{prefix}_I3": 0.0,
            f"{prefix}_I2": 0.0,
            f"{prefix}_I1": 0.0,
            f"{prefix}_I": 0,
        }
        self.current_name = f"i_Na"
        self.META = {
            "referece": [
                "Fitzhugh, R. (1965). A kinetic model of the conductance changes in nerve membrane. Journal of Cellular and Comparative Physiology, 66(S2), 111–117. https://doi.org/10.1002/jcp.1030660518",
                "Armstrong, C. M. (1981). Sodium channels and gating currents. Physiological Reviews, 61(3), 644–683. https://doi.org/10.1152/physrev.1981.61.3.644",
            ],
            "Species": "squid axon",
        }

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state."""
        prefix = self._name
        alpha_m, beta_m = self.m_gate(v)
        alpha_h, beta_h = self.h_gate(v)

        # jax.debug.print("sum of states {x}", x=jnp.sum(jnp.array([states[f"{prefix}_C3"], states[f"{prefix}_C2"], states[f"{prefix}_C1"], states[f"{prefix}_O"], states[f"{prefix}_I3"], states[f"{prefix}_I2"], states[f"{prefix}_I1"], states[f"{prefix}_I"]])))

        C3 = states[f"{prefix}_C3"]
        C2 = states[f"{prefix}_C2"]
        C1 = states[f"{prefix}_C1"]
        O = states[f"{prefix}_O"]

        I3 = states[f"{prefix}_I3"]
        I2 = states[f"{prefix}_I2"]
        I1 = states[f"{prefix}_I1"]
        I = states[f"{prefix}_I"]

        # Explicitly calculate transitions between states
        C3_to_C2 = 3 * alpha_m * C3
        C2_to_C1 = 2 * alpha_m * C2
        C1_to_O = alpha_m * C1

        O_to_C1 = 3 * beta_m * O
        C1_to_C2 = 2 * beta_m * C1
        C2_to_C3 = beta_m * C2

        I3_to_I2 = 3 * alpha_m * I3
        I2_to_I1 = 2 * alpha_m * I2
        I1_to_I = alpha_m * I1

        I_to_I1 = 3 * beta_m * I
        I1_to_I2 = 2 * beta_m * I1
        I2_to_I3 = beta_m * I2

        # C to I and I to C transitions, explicitly updating each state
        C3_to_I3 = beta_h * C3
        C2_to_I2 = beta_h * C2
        C1_to_I1 = beta_h * C1
        O_to_I = beta_h * O

        I3_to_C3 = alpha_h * I3
        I2_to_C2 = alpha_h * I2
        I1_to_C1 = alpha_h * I1
        I_to_O = alpha_h * I

        # Update states with calculated transitions
        new_C3 = C3 + dt * (C2_to_C3 - C3_to_C2 + I3_to_C3 - C3_to_I3)
        new_C2 = C2 + dt * (
            C3_to_C2 - C2_to_C1 + C1_to_C2 - C2_to_C3 - C2_to_I2 + I2_to_C2
        )
        new_C1 = C1 + dt * (
            C2_to_C1 - C1_to_O + O_to_C1 - C1_to_C2 - C1_to_I1 + I1_to_C1
        )
        new_O = O + dt * (C1_to_O - O_to_C1 - O_to_I + I_to_O)

        new_I3 = I3 + dt * (I2_to_I3 - I3_to_I2 + C3_to_I3 - I3_to_C3)
        new_I2 = I2 + dt * (
            I3_to_I2 - I2_to_I1 + I1_to_I2 - I2_to_I3 + C2_to_I2 - I2_to_C2
        )
        new_I1 = I1 + dt * (
            I2_to_I1 - I1_to_I + I_to_I1 - I1_to_I2 + C1_to_I1 - I1_to_C1
        )
        new_I = I + dt * (I1_to_I - I_to_I1 + O_to_I - I_to_O)

        return {
            f"{prefix}_C3": new_C3,
            f"{prefix}_C2": new_C2,
            f"{prefix}_C1": new_C1,
            f"{prefix}_O": new_O,
            f"{prefix}_I3": new_I3,
            f"{prefix}_I2": new_I2,
            f"{prefix}_I1": new_I1,
            f"{prefix}_I": new_I,
        }

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        O = states[f"{prefix}_O"]
        gNa = params[f"{prefix}_gNa"] * O * 1000  # mS/cm^2
        return gNa * (v - params[f"{prefix}_eNa"])

    def init_state(self, states, v, params, delta_t):
        """Initialize the state."""
        states = self.channel_states
        params = self.channel_params
        for i in range(int(120 / delta_t)):
            states = self.update_states(states, delta_t, v, params)
        return states


class K5States(K):
    """Potassium channel in the formulation of Markov model with 5 states"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gK": 35e-3,  # S/cm^2
            f"{prefix}_eK": -77.0,  # mV
        }
        self.channel_states = {
            f"{prefix}_C4": 1.0,
            f"{prefix}_C3": 0,
            f"{prefix}_C2": 0,
            f"{prefix}_C1": 0,
            f"{prefix}_O": 0.0,
        }
        self.current_name = f"i_K"
        self.META = {
            "referece": [
                "Fitzhugh, R. (1965). A kinetic model of the conductance changes in nerve membrane. Journal of Cellular and Comparative Physiology, 66(S2), 111–117. https://doi.org/10.1002/jcp.1030660518",
                "Armstrong, C. M. (1969). Inactivation of the Potassium Conductance and Related Phenomena Caused by Quaternary Ammonium Ion Injection in Squid Axons. The Journal of General Physiology, 54(5), 553–575. https://doi.org/10.1085/jgp.54.5.553",
            ],
            "Species": "squid axon",
        }

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""
        prefix = self._name
        alpha_n, beta_n = self.n_gate(v)

        # Transitions for activation pathway
        C4_to_C3 = 4 * alpha_n * states[f"{prefix}_C4"]
        C3_to_C2 = 3 * alpha_n * states[f"{prefix}_C3"]
        C2_to_C1 = 2 * alpha_n * states[f"{prefix}_C2"]
        C1_to_O = alpha_n * states[f"{prefix}_C1"]

        O_to_C1 = 4 * beta_n * states[f"{prefix}_O"]
        C1_to_C2 = 3 * beta_n * states[f"{prefix}_C1"]
        C2_to_C3 = 2 * beta_n * states[f"{prefix}_C2"]
        C3_to_C4 = beta_n * states[f"{prefix}_C3"]

        new_C4 = states[f"{prefix}_C4"] + dt * (C3_to_C4 - C4_to_C3)
        new_C3 = states[f"{prefix}_C3"] + dt * (
            C4_to_C3 - C3_to_C2 - C3_to_C4 + C2_to_C3
        )
        new_C2 = states[f"{prefix}_C2"] + dt * (
            C3_to_C2 - C2_to_C1 - C2_to_C3 + C1_to_C2
        )
        new_C1 = states[f"{prefix}_C1"] + dt * (C2_to_C1 - C1_to_O - C1_to_C2 + O_to_C1)
        new_O = states[f"{prefix}_O"] + dt * (C1_to_O - O_to_C1)

        return {
            f"{prefix}_C4": new_C4,
            f"{prefix}_C3": new_C3,
            f"{prefix}_C2": new_C2,
            f"{prefix}_C1": new_C1,
            f"{prefix}_O": new_O,
        }

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        O = states[f"{prefix}_O"]
        gK = params[f"{prefix}_gK"] * O * 1000  # mS/cm^2
        return gK * (v - params[f"{prefix}_eK"])

    def init_state(self, states, v, params, delta_t):
        """Initialize the state with 2 minutes of voltage clamp."""
        states = self.channel_states
        params = self.channel_params
        for i in range(int(120 / delta_t)):
            states = self.update_states(states, delta_t, v, params)
        return states
