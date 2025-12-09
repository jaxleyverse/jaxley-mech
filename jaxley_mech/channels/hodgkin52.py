from typing import Dict, Optional

import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley.solver_gate import save_exp, solve_gate_exponential

from jaxley_mech.channels._base import StatesChannel

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
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gLeak": 0.3e-3,  # S/cm^2
            f"{prefix}_eLeak": -65.0,  # mV
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
        """Return current."""
        # Multiply with 1000 to convert Siemens to milli Siemens.
        prefix = self._name
        leak_conds = params[f"{prefix}_gLeak"]  # S/cm^2
        return leak_conds * (v - params[f"{prefix}_eLeak"])

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
            f"{prefix}_gNa": 40e-3,  # S/cm^2
            f"{prefix}_eNa": 55.0,  # mV
        }
        self.channel_states = {f"{prefix}_m": 0.2, f"{prefix}_h": 0.2}
        self.current_name = "iNa"
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
        gNa = params[f"{prefix}_gNa"] * (m**3) * h  # S/cm^2
        current = gNa * (v - params[f"{prefix}_eNa"])  # S/cm^2 * mV = mA/cm^2
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
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gK": 35e-3,  # S/cm^2
            f"{prefix}_eK": -77.0,  # mV
        }
        self.channel_states = {f"{prefix}_n": 0.1}
        self.current_name = "iK"
        self.META = META
        self.META.update({"ion": "K"})

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
        gK = params[f"{prefix}_gK"] * (n**4)  # S/cm^2
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


class Na8States(StatesChannel, Na):
    """Na channel with m^3 h gating as an auto-generated 8-state chain."""

    def __init__(
        self,
        name: Optional[str] = None,
        solver: str = "sde_implicit",
        rtol: float = 1e-8,
        atol: float = 1e-8,
        max_steps: int = 10,
        shield_mask: Optional[jnp.ndarray] = None,
    ):
        Na.__init__(self, name)
        prefix = self._name

        # Remove single-gate states; Markov states live on the simplex.
        self.channel_states.pop(f"{prefix}_m", None)
        self.channel_states.pop(f"{prefix}_h", None)

        # Channel-count and noise seed defaults.
        self.channel_params.setdefault(f"{prefix}_N_Na", 1e4)
        self.channel_params.setdefault(f"{prefix}_noise_seed", 0)

        StatesChannel.__init__(
            self,
            name=name,
            gate_specs=[("m", 3, Na.m_gate), ("h", 1, Na.h_gate)],
            count_param=f"{prefix}_N_Na",
            solver=solver,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            noise_seed_param=f"{prefix}_noise_seed",
            shield_mask=shield_mask,
        )

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        prefix = self._name
        p_open = self.open_probability(states)
        gNa = params[f"{prefix}_gNa"] * p_open  # S/cm^2
        return gNa * (v - params[f"{prefix}_eNa"])  # S/cm^2 * mV = mA/cm^2


class K5States(StatesChannel, K):
    """K channel with n^4 gating as an auto-generated 5-state chain."""

    def __init__(
        self,
        name: Optional[str] = None,
        solver: str = "sde_implicit",
        rtol: float = 1e-8,
        atol: float = 1e-8,
        max_steps: int = 10,
        shield_mask: Optional[jnp.ndarray] = None,
    ):
        K.__init__(self, name)
        prefix = self._name

        # Remove single-gate state; Markov states live on the simplex.
        self.channel_states.pop(f"{prefix}_n", None)

        # Channel-count and noise seed defaults.
        self.channel_params.setdefault(f"{prefix}_N_K", 1e4)
        self.channel_params.setdefault(f"{prefix}_noise_seed", 0)

        StatesChannel.__init__(
            self,
            name=name,
            gate_specs=[("n", 4, K.n_gate)],
            count_param=f"{prefix}_N_K",
            solver=solver,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            noise_seed_param=f"{prefix}_noise_seed",
            shield_mask=shield_mask,
        )

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        prefix = self._name
        p_open = self.open_probability(states)
        gK = params[f"{prefix}_gK"] * p_open  # S/cm^2
        return gK * (v - params[f"{prefix}_eK"])  # S/cm^2 * mV = mA/cm^2
