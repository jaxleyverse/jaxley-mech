from typing import Dict, Optional

import jax.numpy as jnp
from jaxley.mechanisms.channels import Channel
from jaxley.mechanisms.solvers import save_exp, solve_gate_exponential

from jaxley_mech.mechanisms.solvers import SolverExtension

META = {
    "reference": "Hodgkin & Huxley (1952)",
    "doi": "https://doi.org/10.1113/jphysiol.1952.sp004764",
    "species": "squid",
    "note": "Unlike the original paper, we adjusted the equations to the modern convention, that is, the resting potential is -65 mV, and signs are flipped.",
}


class Leak(Channel):
    """Leakage current"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self.name
        self.params = {
            f"{prefix}_gLeak": 0.3e-3,  # S/cm^2
            f"{prefix}_eLeak": -65.0,  # mV
        }
        self.states = {}
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
        prefix = self.name
        leak_conds = params[f"{prefix}_gLeak"]  # S/cm^2
        return leak_conds * (v - params[f"{prefix}_eLeak"])

    def init_states(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        return {}


class Na(Channel):
    """Sodium channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self.name
        self.params = {
            f"{prefix}_gNa": 40e-3,  # S/cm^2
            f"{prefix}_eNa": 55.0,  # mV
        }
        self.states = {f"{prefix}_m": 0.2, f"{prefix}_h": 0.2}
        self.current_name = f"i_Na"
        self.META = META
        self.META.update({"ion": "Na"})

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: float,
        params: Dict[str, jnp.ndarray],
    ):
        "Update state."
        prefix = self.name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        new_m = solve_gate_exponential(m, dt, *self.m_gate(v))
        new_h = solve_gate_exponential(h, dt, *self.h_gate(v))

        return {f"{prefix}_m": new_m, f"{prefix}_h": new_h}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        "Return current."
        prefix = self.name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        gNa = params[f"{prefix}_gNa"] * (m**3) * h  # S/cm^2
        current = gNa * (v - params[f"{prefix}_eNa"])  # S/cm^2 * mV = mA/cm^2
        return current

    def init_states(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self.name
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
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self.name
        self.params = {
            f"{prefix}_gK": 35e-3,  # S/cm^2
            f"{prefix}_eK": -77.0,  # mV
        }
        self.states = {f"{prefix}_n": 0.1}
        self.current_name = f"i_K"
        self.META = META
        self.META.update({"ion": "K"})

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""
        prefix = self.name
        n = states[f"{prefix}_n"]
        new_n = solve_gate_exponential(n, dt, *self.n_gate(v))
        return {f"{prefix}_n": new_n}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self.name
        n = states[f"{prefix}_n"]
        gK = params[f"{prefix}_gK"] * (n**4)  # S/cm^2
        return gK * (v - params[f"{prefix}_eK"])

    def init_states(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self.name
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


class Na8States(Na, SolverExtension):
    """Sodium channel in the formulation of Markov model with 8 states"""

    def __init__(
        self,
        name: Optional[str] = None,
        solver: Optional[str] = None,
        rtol: float = 1e-8,
        atol: float = 1e-8,
        max_steps: int = 10,
    ):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        SolverExtension.__init__(self, solver, rtol, atol, max_steps)
        prefix = self.name
        self.solver = solver
        self.params = {
            f"{prefix}_gNa": 40e-3,  # S/cm^2
            f"{prefix}_eNa": 55.0,  # mV
        }
        # Initialize all states
        self.states = {
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
            "reference": "Armstrong, C. M. (1981).",
            "doi": "https://doi.org/10.1152/physrev.1981.61.3.644",
            "species": "squid",
            "ion": "Na",
        }

    def derivatives(self, t, states, args):
        """Calculate the derivatives for the Markov states."""
        C3, C2, C1, O, I3, I2, I1, I = states
        v = args[0]
        alpha_m, beta_m = self.m_gate(v)
        alpha_h, beta_h = self.h_gate(v)

        # Transitions for activation pathway (C3 -> O)
        C3_to_C2 = 3 * alpha_m * C3
        C2_to_C1 = 2 * alpha_m * C2
        C1_to_O = alpha_m * C1

        O_to_C1 = 3 * beta_m * O
        C1_to_C2 = 2 * beta_m * C1
        C2_to_C3 = beta_m * C2

        # Transitions for inactivation pathway (I3 -> I)
        I3_to_I2 = 3 * alpha_m * I3
        I2_to_I1 = 2 * alpha_m * I2
        I1_to_I = alpha_m * I1

        I_to_I1 = 3 * beta_m * I
        I1_to_I2 = 2 * beta_m * I1
        I2_to_I3 = beta_m * I2

        # C to I and I to C transitions (C3, C2, C1, O <-> I3, I2, I1, I)
        C3_to_I3 = beta_h * C3
        C2_to_I2 = beta_h * C2
        C1_to_I1 = beta_h * C1
        O_to_I = beta_h * O

        I3_to_C3 = alpha_h * I3
        I2_to_C2 = alpha_h * I2
        I1_to_C1 = alpha_h * I1
        I_to_O = alpha_h * I

        # Derivatives for each state
        dC3_dt = C2_to_C3 - C3_to_C2 + I3_to_C3 - C3_to_I3
        dC2_dt = C3_to_C2 - C2_to_C1 + C1_to_C2 - C2_to_C3 + I2_to_C2 - C2_to_I2
        dC1_dt = C2_to_C1 - C1_to_O + O_to_C1 - C1_to_C2 + I1_to_C1 - C1_to_I1
        dO_dt = C1_to_O - O_to_C1 + I_to_O - O_to_I

        dI3_dt = I2_to_I3 - I3_to_I2 + C3_to_I3 - I3_to_C3
        dI2_dt = I3_to_I2 - I2_to_I1 + I1_to_I2 - I2_to_I3 + C2_to_I2 - I2_to_C2
        dI1_dt = I2_to_I1 - I1_to_I + I_to_I1 - I1_to_I2 + C1_to_I1 - I1_to_C1
        dI_dt = I1_to_I - I_to_I1 + O_to_I - I_to_O

        return jnp.array([dC3_dt, dC2_dt, dC1_dt, dO_dt, dI3_dt, dI2_dt, dI1_dt, dI_dt])

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state."""
        prefix = self.name

        # Retrieve states
        C3 = states[f"{prefix}_C3"]
        C2 = states[f"{prefix}_C2"]
        C1 = states[f"{prefix}_C1"]
        O = states[f"{prefix}_O"]

        I3 = states[f"{prefix}_I3"]
        I2 = states[f"{prefix}_I2"]
        I1 = states[f"{prefix}_I1"]
        I = states[f"{prefix}_I"]

        y0 = jnp.array([C3, C2, C1, O, I3, I2, I1, I])

        # Parameters for dynamics
        args_tuple = (v,)

        y_new = self.solver_func(y0, dt, self.derivatives, args_tuple)

        # Unpack new states
        C3_new, C2_new, C1_new, O_new, I3_new, I2_new, I1_new, I_new = y_new

        return {
            f"{prefix}_C3": C3_new,
            f"{prefix}_C2": C2_new,
            f"{prefix}_C1": C1_new,
            f"{prefix}_O": O_new,
            f"{prefix}_I3": I3_new,
            f"{prefix}_I2": I2_new,
            f"{prefix}_I1": I1_new,
            f"{prefix}_I": I_new,
        }

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self.name
        O = states[f"{prefix}_O"]
        gNa = params[f"{prefix}_gNa"] * O  # S/cm^2
        return gNa * (v - params[f"{prefix}_eNa"])

    def init_states(self, states, v, params, delta_t):
        """Initialize the state to steady-state values."""
        prefix = self.name
        alpha_m, beta_m = self.m_gate(v)
        alpha_h, beta_h = self.h_gate(v)

        m_inf = alpha_m / (alpha_m + beta_m)
        h_inf = alpha_h / (alpha_h + beta_h)

        # Calculate steady-state probabilities
        C3 = (1 - m_inf) ** 3 * h_inf
        C2 = 3 * m_inf * (1 - m_inf) ** 2 * h_inf
        C1 = 3 * m_inf**2 * (1 - m_inf) * h_inf
        O = m_inf**3 * h_inf
        I3 = (1 - m_inf) ** 3 * (1 - h_inf)
        I2 = 3 * m_inf * (1 - m_inf) ** 2 * (1 - h_inf)
        I1 = 3 * m_inf**2 * (1 - m_inf) * (1 - h_inf)
        I = m_inf**3 * (1 - h_inf)

        return {
            f"{prefix}_C3": C3,
            f"{prefix}_C2": C2,
            f"{prefix}_C1": C1,
            f"{prefix}_O": O,
            f"{prefix}_I3": I3,
            f"{prefix}_I2": I2,
            f"{prefix}_I1": I1,
            f"{prefix}_I": I,
        }


class K5States(K, SolverExtension):
    """Potassium channel in the formulation of Markov model with 5 states"""

    def __init__(
        self,
        name: Optional[str] = None,
        solver: Optional[str] = None,
        rtol: float = 1e-8,
        atol: float = 1e-8,
        max_steps: int = 10,
    ):
        super().__init__(name)
        SolverExtension.__init__(self, solver, rtol, atol, max_steps)
        prefix = self.name
        self.solver = solver
        self.params = {
            f"{prefix}_gK": 35e-3,  # S/cm^2
            f"{prefix}_eK": -77.0,  # mV
        }
        self.states = {
            f"{prefix}_C4": 1.0,
            f"{prefix}_C3": 0,
            f"{prefix}_C2": 0,
            f"{prefix}_C1": 0,
            f"{prefix}_O": 0.0,
        }
        self.current_name = f"i_K"
        self.META = {
            "reference": "Armstrong, (1969)",
            "doi": "https://doi.org/10.1085/jgp.54.5.553",
            "species": "squid",
            "ion": "K",
        }

    def derivatives(self, t, states, args):
        """Calculate the derivatives for the Markov states."""
        C4, C3, C2, C1, O = states
        v = args[0]
        alpha_n, beta_n = self.n_gate(
            v
        )  # Use voltage (t) to calculate alpha_n and beta_n

        # Transitions for activation pathway
        C4_to_C3 = 4 * alpha_n * C4
        C3_to_C2 = 3 * alpha_n * C3
        C2_to_C1 = 2 * alpha_n * C2
        C1_to_O = alpha_n * C1

        O_to_C1 = 4 * beta_n * O
        C1_to_C2 = 3 * beta_n * C1
        C2_to_C3 = 2 * beta_n * C2
        C3_to_C4 = beta_n * C3

        # Derivatives of each state
        dC4_dt = C3_to_C4 - C4_to_C3
        dC3_dt = C4_to_C3 - C3_to_C2 - C3_to_C4 + C2_to_C3
        dC2_dt = C3_to_C2 - C2_to_C1 - C2_to_C3 + C1_to_C2
        dC1_dt = C2_to_C1 - C1_to_O - C1_to_C2 + O_to_C1
        dO_dt = C1_to_O - O_to_C1

        return jnp.array([dC4_dt, dC3_dt, dC2_dt, dC1_dt, dO_dt])

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
        **kwargs,
    ):
        """Update state using the specified solver."""
        prefix = self.name

        # Retrieve states
        C4 = states[f"{prefix}_C4"]
        C3 = states[f"{prefix}_C3"]
        C2 = states[f"{prefix}_C2"]
        C1 = states[f"{prefix}_C1"]
        O = states[f"{prefix}_O"]

        y0 = jnp.array([C4, C3, C2, C1, O])

        # Parameters for dynamics
        args_tuple = (v,)

        y_new = self.solver_func(y0, dt, self.derivatives, args_tuple)

        # Unpack new states
        C4_new, C3_new, C2_new, C1_new, O_new = y_new

        return {
            f"{prefix}_C4": C4_new,
            f"{prefix}_C3": C3_new,
            f"{prefix}_C2": C2_new,
            f"{prefix}_C1": C1_new,
            f"{prefix}_O": O_new,
        }

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self.name
        O = states[f"{prefix}_O"]
        gK = params[f"{prefix}_gK"] * O  # S/cm^2
        return gK * (v - params[f"{prefix}_eK"])

    def init_states(self, states, v, params, delta_t):
        """Initialize the state to steady-state values."""
        prefix = self.name
        alpha_n, beta_n = self.n_gate(v)

        n_inf = alpha_n / (alpha_n + beta_n)

        # Calculate steady-state probabilities
        C4 = (1 - n_inf) ** 4
        C3 = 4 * n_inf * (1 - n_inf) ** 3
        C2 = 6 * n_inf**2 * (1 - n_inf) ** 2
        C1 = 4 * n_inf**3 * (1 - n_inf)
        O = n_inf**4

        return {
            f"{prefix}_C4": C4,
            f"{prefix}_C3": C3,
            f"{prefix}_C2": C2,
            f"{prefix}_C1": C1,
            f"{prefix}_O": O,
        }
