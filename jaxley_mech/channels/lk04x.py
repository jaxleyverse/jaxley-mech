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
        gLeak = params[f"{prefix}_gLeak"] * 1000
        return gLeak * (v - params[f"{prefix}_eLeak"])  # mS/cm^2 * mV = uA/cm^2

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        return {}


class Kx(Channel):
    """Noninactivating potassium channels"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gKx": 0.85e-3,  # mS/cm^2
            f"VhalfKx": -50,  # mv
            f"aoKx": 0.66,  # /s
            f"SKx": 5.7,  # mV
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
        n_new = solve_gate_exponential(ns, dt, *self.n_gate(v, params))
        return {f"{prefix}_n": n_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ns = states[f"{prefix}_n"]
        k_cond = params[f"{prefix}_gKx"] * ns * 1000
        return k_cond * (v - params["eK"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha, beta = self.n_gate(v, params)
        return {f"{prefix}_n": alpha / (alpha + beta)}

    @staticmethod
    def n_gate(v, params):
        """Voltage-dependent dynamics for the n gating variable."""
        v += 1e-6
        VhalfKx = params["VhalfKx"]
        aoKx = params["aoKx"]
        SKx = params["SKx"]
        alpha = aoKx * save_exp((v - VhalfKx) / SKx)
        beta = aoKx * save_exp(-(v - VhalfKx) / SKx)
        return alpha, beta


class Kv(Channel):
    """Delayed Rectifying Potassium Channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gKv": 10,  # nS, mS/cm^2
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
        k_cond = params[f"{prefix}_gKv"] * ns**4
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
        alpha = 0.5 * (20 - v) / (save_exp((20 - v) / 22) - 1)
        beta = 6.25 * save_exp(-v / 80)
        return alpha, beta
