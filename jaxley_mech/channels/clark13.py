from typing import Dict, Optional
import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley_mech.solvers import SolverExtension

META = {
    "cell_type": ["photoreceptor"],
    "species": ["salamander", "turtle"],
    "reference": "Clark, et al. (2013)",
    "doi": "https://doi.org/10.1371/journal.pcbi.1003289",
    "note": "Default parameters are from salamander"
}


class Phototransduction(Channel, SolverExtension):
    """Abstract phototransduction channel"""
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
        prefix = self._name
        self.channel_params = {
            f"{prefix}_tau_r": 28.0, # relaxation time (ms)
            f"{prefix}_tau_y": 33.0, # timescale of the linear response (ms)
            f"{prefix}_tau_z": 19.0, # timescale of the slow response (ms)
            f"{prefix}_alpha": 0.8, # constant scaling the linear filter (mV*um^2*ms/photon)
            f"{prefix}_beta": 0.16, # constant scaling the nonlinear filter (1/mV)
            f"{prefix}_gamma": 0.23, # weighting factor of the two timescales
            f"{prefix}_ny": 4.0, # rise behavior of the linear response
            f"{prefix}_nz": 10.0,
            f"{prefix}_V_rest": -35.0, # the PR rmp (mV)
        }
        self.channel_states = {
            f"{prefix}_r": -35.0, # difference between instantaneous rp and dark rp
            f"{prefix}_y": 0.0, # filtered light intensity
            f"{prefix}_z": 0.0, # filtered light intensity
            f"{prefix}_Stim": 0.0, # stimulus (TODO: units P*/s?)
            "v": -35.0 # PR voltage (mV)
            }
        self.current_name = f"iPhoto"
        self.META = META

    def derivatives(self, t, states, args):
        pass

    def update_states(self, states, dt, v, params, **kwargs):
        prefix = self._name
        y0 = jnp.array([
            states[f"{prefix}_r"],
            states[f"{prefix}_y"],
            states[f"{prefix}_z"],
        ])
        args_tuple = ()
        r_new, y_new, z_new = self.solver_func(y0, dt, self.derivatives, args_tuple)
        v_new = r_new + params[f"{prefix}_V_rest"]
        return {
            f"{prefix}_r": r_new,
            f"{prefix}_y": y_new,
            f"{prefix}_z": z_new,
            "v": v_new
            }

    def compute_current(self, states, v, params):
        return 0

    def init_state(self, states, v, params, delta_t):
        pass
