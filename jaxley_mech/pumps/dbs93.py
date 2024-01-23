from typing import Dict, Optional

import jax.numpy as jnp

from jaxley.channels import Channel


class CaPump(Channel):
    """Calcium ATPase pump"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_kt": 1e-4,  # mM/ms (time constant of the pump)
            f"CaCon_diss": 1e-4,  # mM (equilibrium calcium value, calcium dissociation constant)
        }
        self.channel_states = {
            "CaCon_i": 1e-4  # mM (global internal calcium concentration)
        }
        self.META = {
            "reference": "Destexhe, A. Babloyantz, A. and Sejnowski, TJ. Ionic mechanisms for intrinsic slow oscillations in thalamic relay neurons. Biophys. J. 65: 1538-1552, 1993.",
            "mechanism": "ATPase pump",
        }

    def update_states(self, u, dt, voltages, params):
        """Update internal calcium concentration due to the pump action."""
        prefix = self._name
        cai = u["CaCon_i"]
        kt = params[f"{prefix}_kt"]
        kd = params[f"CaCon_diss"]

        # Michaelis-Menten dynamics for the pump's action on calcium concentration
        drive_pump = -kt * cai / (cai + kd)
        # Update internal calcium concentration
        new_cai = cai + drive_pump * dt
        return {"CaCon_i": new_cai}

    def compute_current(self, u, voltages, params):
        """The pump does not directly contribute to the membrane current."""
        return 0
