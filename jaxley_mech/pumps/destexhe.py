from typing import Dict, Optional

import jax.numpy as jnp
from jaxley.channels import Channel


class CaPump(Channel):
    """Calcium ATPase pump modeled after Destexhe et al., 1993/1994."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_kt": 1e-4,  # Time constant of the pump in mM/ms
            f"{self._name}_kd": 1e-4,  # Equilibrium calcium value (dissociation constant) in mM
            f"{self.name}_depth": 0.1,  # Depth of shell in um
            f"{self.name}_taur": 1e10,  # Time constant of calcium removal in ms
            f"{self.name}_cainf": 2.4e-4,  # Equilibrium calcium concentration in mM
        }
        self.channel_states = {
            f"CaCon_i": 1e-4,  # Initial internal calcium concentration in mM
        }
        self.META = {
            "reference": "Destexhe, A., Babloyantz, A., & Sejnowski, TJ. Ionic mechanisms for intrinsic slow oscillations in thalamic relay neurons. Biophys. J. 65: 1538-1552, 1993.",
            "mechanism": "ATPase pump",
            "source": "https://modeldb.science/3670?tab=2&file=NTW_NEW/capump.mod",
        }

    def update_states(self, u, dt, voltages, params, ica):
        """Update internal calcium concentration due to pump action and calcium currents."""
        prefix = self._name
        cai = u[f"CaCon_i"]
        kt = params[f"{prefix}_kt"]
        kd = params[f"{prefix}_kd"]
        depth = params[f"{prefix}_depth"]
        taur = params[f"{prefix}_taur"]
        cainf = params[f"{prefix}_cainf"]

        FARADAY = 96489  # Coulombs per mole

        # Compute inward calcium flow contribution, should not pump inwards
        drive_channel = -ica / (
            2 * FARADAY * depth
        )  # why *10000 in the original code? Unit mismatch?
        drive_channel = jnp.where(drive_channel <= 0, 0, drive_channel)

        # Michaelis-Menten dynamics for the pump's action on calcium concentration
        drive_pump = -kt * cai / (cai + kd)

        # Update internal calcium concentration with contributions from channel, pump, and decay to equilibrium
        new_cai = cai + dt * (drive_channel + drive_pump + (cainf - cai) / taur)

        return {f"CaCon_i": new_cai}

    def compute_current(self, u, voltages, params):
        """The pump does not directly contribute to the membrane current."""
        return 0
