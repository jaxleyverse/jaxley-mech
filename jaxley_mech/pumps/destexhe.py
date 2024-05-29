from typing import Dict, Optional

import jax.numpy as jnp
from jax.lax import select
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
            f"Cai": 5e-5,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"iCa"
        self.META = {
            "reference": "Destexhe, A., Babloyantz, A., & Sejnowski, TJ. Ionic mechanisms for intrinsic slow oscillations in thalamic relay neurons. Biophys. J. 65: 1538-1552, 1993.",
            "mechanism": "ATPase pump",
            "source": "https://modeldb.science/3670?tab=2&file=NTW_NEW/capump.mod",
        }

    def update_states(self, states, dt, v, params):
        """Update internal calcium concentration due to pump action and calcium currents."""
        prefix = self._name
        iCa = states[f"iCa"] / 1_000  # Convert from uA/cm^2 to mA/cm^2
        Cai = states[f"Cai"]
        kt = params[f"{prefix}_kt"]
        kd = params[f"{prefix}_kd"]
        depth = params[f"{prefix}_depth"]
        taur = params[f"{prefix}_taur"]
        cainf = params[f"{prefix}_cainf"]

        FARADAY = 96489  # Coulombs per mole

        # Compute inward calcium flow contribution, should not pump inwards
        drive_channel = -10_000.0 * iCa / (2 * FARADAY * depth)
        drive_channel = select(
            drive_channel <= 0, jnp.zeros_like(drive_channel), drive_channel
        )

        # Michaelis-Menten dynamics for the pump's action on calcium concentration
        drive_pump = -kt * Cai / (Cai + kd)

        dCai_dt = drive_channel + drive_pump + (cainf - Cai) / taur

        # Update internal calcium concentration with contributions from channel, pump, and decay to equilibrium
        new_Cai = Cai + dt * dCai_dt

        return {f"Cai": new_Cai}

    def compute_current(self, states, v, params):
        """The pump does not directly contribute to the membrane current."""
        return 0

    def init_state(self, voltages, params):
        """Initialize the internal calcium concentration."""
        return {}
