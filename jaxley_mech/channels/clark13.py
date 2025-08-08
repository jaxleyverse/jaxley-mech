from typing import Dict, Optional
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
            "tau_r": 28.0, # relaxation time (ms)
            "tau_y": 33.0, # timescale of the linear response (ms)
            "tau_z": 19.0, # timescale of the slow response (ms)
            "alpha": 0.8, # constant scaling the linear filter (mV*um^2*ms/photon)
            "beta": 0.16, # constant scaling the nonlinear filter (1/mV)
            "gamma": 0.23, # weighting factor of the two timescales
            "ny": 4.0, # rise behavior of the linear response
            "nz": 10.0,
        
        }
        self.channel_states = {}
        self.current_name = f"iPhoto"
        self.META = META

    def derivatives(self, t, states, args):
        pass

    def update_states(seld, states, dt, v, params, **kwargs):
        pass

    def compute_current(self, states, v, params):
        pass

    def init_state(self, states, v, params, delta_t):
        pass
