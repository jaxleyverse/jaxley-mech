from typing import Dict, Optional

import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler, save_exp, solve_gate_exponential

__all__ = [
    "Leak",
    "hRod",
    "hCone",
    "Kv",
    "Kx",
    "Ca",
    "KCa",
    "ClCa",
    "CaPump",
    "CaNernstReversal",
]

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
        self.current_name = f"i_Kx"
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
        return k_cond * (v - params["eK"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha, beta = self.n_gate(v)
        return {f"{prefix}_n": alpha / (alpha + beta)}

    @staticmethod
    def n_gate(v):
        """Voltage-dependent dynamics for the n gating variable."""
        alpha = 6.6e-4 * save_exp((v + 50) / 11.4)
        beta = 6.6e-4 * save_exp(-(v + 50) / 11.4)
        return alpha, beta


class Kv(Channel):
    """Delayed Rectifying Potassium Channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gKv": 0.52e-3,  # S/cm^2
            "eK": -74,  # mV
        }
        self.channel_states = {
            f"{self._name}_n": 1e-2,  # Initial value for n gating variable
        }
        self.current_name = f"i_Kv"
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
        alpha = 0.005 * (20 - v) / (save_exp((20 - v) / 22) - 1)
        beta = 0.0625 * save_exp(-v / 80)
        return alpha, beta


class Ca(Channel):
    """L-type calcium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gCa": 4e-3,  # S/cm^2
        }
        self.channel_states = {
            f"{self._name}_m": 0.1,  # Initial value for m gating variable
            f"{self._name}_h": 0.1,  # Initial value for h gating variable
            "eCa": 0.0,  # mV, dependent on CaNernstReversal
        }
        self.current_name = f"i_Ca"
        self.META = META

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
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new, "eCa": u["eCa"]}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        ca_cond = params[f"{prefix}_gCa"] * ms * hs * 1000
        current = ca_cond * (voltages - u["eCa"])
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
        alpha = 0.1 * save_exp((v + 10) / 12.0)
        beta = 0.1 * save_exp(-(v + 10) / 12.0)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable."""
        alpha = 0.01 * save_exp((v - 11) / 18.0)
        beta = 0.0005 / (save_exp(-(v - 11) / 18.0))
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
            f"{self._name}_Cai_inf": 2e-3,  # mM
        }
        self.channel_states = {
            f"Cai": 5e-05,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"i_Ca"
        self.META = {
            "reference": "Modified from Destexhe et al., 1994",
            "mechanism": "Calcium dynamics",
        }

    def update_states(self, u, dt, voltages, params):
        """Update internal calcium concentration based on calcium current and decay."""
        prefix = self._name
        iCa = u["i_Ca"] / 1_000.0
        Cai = u["Cai"]
        depth = params[f"{prefix}_depth"]
        Cai_tau = params[f"{prefix}_Cai_tau"]
        Cai_inf = params[f"{prefix}_Cai_inf"]

        FARADAY = 96485.3329  # Coulombs per mole

        # Calculate the contribution of calcium currents to cai change
        drive_channel = -10_000.0 * iCa / (2 * FARADAY * depth)
        drive_channel = jnp.clip(drive_channel, a_min=0.0)
        new_Cai = exponential_euler(Cai, dt, Cai_inf, Cai_tau)

        return {f"Cai": new_Cai}

    def compute_current(self, u, voltages, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(self, voltages, params):
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
            "F": 96485.3329,  # C/mol (Faraday's constant)
            "T": 279.45,  # Kelvin (temperature)
            "R": 8.314,  # J/(mol K) (gas constant)
        }
        self.channel_params = {}
        self.channel_states = {"eCa": 0.0, "Cai": 5e-05, "Cao": 2.0}
        self.current_name = f"i_Ca"

    def update_states(self, u, dt, voltages, params):
        """Update internal calcium concentration based on calcium current and decay."""
        R, T, F = (
            self.channel_constants["R"],
            self.channel_constants["T"],
            self.channel_constants["F"],
        )
        Cai = u["Cai"]
        Cao = u["Cao"]
        C = R * T / (2 * F) * 1000  # mV
        vCa = C * jnp.log(Cao / Cai)
        return {"eCa": vCa, "Cai": Cai, "Cao": Cao}

    def compute_current(self, u, voltages, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(self, voltages, params):
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
            "Cai": 5e-05,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"i_KCa"
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
            f"{self._name}_eCl": -20,  # mV
        }
        self.channel_states = {
            f"{self._name}_n": 0.1,  # Initial value for n gating variable
            "Cai": 5e-05,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"i_KCa"
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
        return k_cond * (v - params[f"{prefix}_eCl"])

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


class hRod(Channel):
    """Rod Hyperpolarization-activated h Channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_ghRod": 2.5e-3,  # S/cm^2
            "ehRod": -32,  # mV
        }
        self.channel_states = {
            f"{self._name}_hRod": 0.1,  # Initial value for h gating variable
        }
        self.current_name = f"i_hRod"
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self._name
        hs = states[f"{prefix}_hRod"]
        h_new = solve_gate_exponential(hs, dt, *self.h_gate(v))
        return {f"{prefix}_hRod": h_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        hs = states[f"{prefix}_hRod"]
        h_cond = params[f"{prefix}_ghRod"] * hs * 1000
        return h_cond * (v - params["ehRod"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha, beta = self.h_gate(v)
        return {f"{prefix}_hRod": alpha / (alpha + beta)}

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable."""
        alpha = 0.001 * save_exp(-(v + 75) / 10.6)
        beta = 0.001 / save_exp((v + 75) / 10.6)
        return alpha, beta


class hCone(Channel):
    """Rod Hyperpolarization-activated h Channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_ghCone": 3.5e-3,  # S/cm^2
            "ehCone": -32.5,  # mV
        }
        self.channel_states = {
            f"{self._name}_hCone": 0.1,  # Initial value for h gating variable
        }
        self.current_name = f"i_hCone"
        self.META = {
            "cell_type": "cone",
            "reference": "Kourennyi, D. E., Liu, X., Hart, J., Mahmud, F., Baldridge, W. H., & Barnes, S. (2004). Reciprocal Modulation of Calcium Dynamics at Rod and Cone Photoreceptor Synapses by Nitric Oxide. Journal of Neurophysiology, 92(1), 477–483. https://doi.org/10.1152/jn.00606.2003",
            "code": "https://modeldb.science/64216?tab=2&file=Kourennyi-etal2004/h.mod",
        }

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self._name
        hs = states[f"{prefix}_hCone"]
        h_new = solve_gate_exponential(hs, dt, *self.h_gate(v))
        return {f"{prefix}_hCone": h_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        hs = states[f"{prefix}_hCone"]
        hs = 1 - (1 + 3 * hs) * (1 - hs) ** 3
        h_cond = params[f"{prefix}_ghCone"] * hs * 1000
        return h_cond * (v - params["ehCone"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha, beta = self.h_gate(v)
        return {f"{prefix}_hCone": alpha / (alpha + beta)}

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable."""
        alpha = 0.001 * 18 / (save_exp((v + 88) / 12) + 1)
        beta = 0.001 * 18 / (save_exp(-(v + 88) / 19) + 1)
        return alpha, beta
