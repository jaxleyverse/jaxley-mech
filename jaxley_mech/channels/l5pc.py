from typing import Dict, Optional

import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley.solver_gate import solve_gate_exponential

from ..utils import efun


class NaTaT(Channel):
    """Transient sodium current from Colbert and Pan, 2002."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNaTa_tbar": 0.00001,  # S/cm^2
            f"{prefix}_ena": None,  # TODO
        }
        self.channel_states = {}
        self.META = {
            "reference": "Colbert and Pan, 2002",
            "species": "unknown",
            "cell_type": "Layer 5 pyramidal cell",
        }

    def update_states(
        self, u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        return {}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        # Multiply with 1000 to convert Siemens to milli Siemens.
        prefix = self._name
        cond = params[f"{prefix}_gNaTa_tbar"] * 1000  # mS/cm^2
        return cond * (voltages - params[f"{prefix}_ena"])

    def init_state(self, voltages, params):
        return {}


class NaTs2T(Channel):
    """Transient sodium current from Colbert and Pan, 2002."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNaTs2_tbar": 0.00001,  # S/cm^2
            f"{prefix}_ena": None,  # TODO
        }
        self.channel_states = {}
        self.META = {
            "reference": "Colbert and Pan, 2002",
            "species": "unknown",
            "cell_type": "Layer 5 pyramidal cell",
        }

    def update_states(
        self, u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        return {}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        # Multiply with 1000 to convert Siemens to milli Siemens.
        prefix = self._name
        cond = params[f"{prefix}_gNaTs2_tbar"] * 1000  # mS/cm^2
        return cond * (voltages - params[f"{prefix}_ena"])

    def init_state(self, voltages, params):
        return {}


class NapEt2(Channel):
    """Persistent sodium current from Magistretti & Alonso 1999.
    
    Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNap_Et2bar": 0.00001,  # S/cm^2
            f"{prefix}_ena": None,  # TODO
        }
        self.channel_states = {}
        self.META = {
            "reference": "Magistretti and Alonso 1999",
            "species": "unknown",
            "cell_type": "Layer 5 pyramidal cell",
        }

    def update_states(
        self, u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        return {}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        # Multiply with 1000 to convert Siemens to milli Siemens.
        prefix = self._name
        cond = params[f"{prefix}_gNap_Et2bar"] * 1000  # mS/cm^2
        return cond * (voltages - params[f"{prefix}_ena"])

    def init_state(self, voltages, params):
        return {}
