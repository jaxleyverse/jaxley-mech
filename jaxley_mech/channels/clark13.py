from typing import Dict, Optional
import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley_mech.solvers import SolverExtension
import numpy as np

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
            f"{prefix}_Stim": 0.0, # stimulus (TODO: units P*/s?)
            f"{prefix}_y": 0.0, # PROBLEM: vector with size dependent on n_y
            f"{prefix}_z": 0.0, # PROBLEM: vector with size dependent on n_z
            "v": -35.0 # PR voltage (mV)
            }
        self.current_name = f"iPhoto"
        self.META = META

    def get_SS_mats(self, n, tau, delta_t):
        """
        Calculate the state-space matrices for cascaded first-order filters.
        
        For n cascaded filters: x1 -> x2 -> x3 -> ... -> xn -> output
        Each stage: dx_i/dt = -(1/tau)*x_i + (1/tau)*x_(i-1)
        With x_0 = input
        
        Forward Euler discretization: dx/dt ≈ (x[k+1] - x[k])/Ts

        TODO: Was there a way to do this calc once in the init? (prev. discussed)
        """
        alpha = delta_t / tau
        A = np.eye(n) * (1 - alpha)
        if n > 1:
            A[1:, :-1] += np.eye(n-1) * alpha # set the off-diagonal (cascade)
        
        B = np.zeros(n)
        B[0] = alpha # Only the first state gets direct input

        C = np.zeros(n)
        C[-1] = 1.0 # Output comes from the last state

        return A, B, C

    def derivatives(self, t, states, args):
        """Calculate the derivatives for the phototransduction system."""
        tau_r, alpha, beta = args
        r, Stim, y, z, v = states
        dr_dt = 1/tau_r * (alpha*y - (1 + beta*z) * r)
        return jnp.array([dr_dt])

    def update_states(self, states, dt, v, params, **kwargs):
        prefix = self._name
        # Take care of the filtering of the stimulus
        Ay, By, Cy = self.get_SS_mats(params[f"{prefix}_n_y"], params[f"{prefix}_tau_y"], dt)
        Ky_out = Cy @ states[f"{prefix}_y"] 
        new_y = Ay @ states[f"{prefix}_y"] + By * states[f"{prefix}_Stim"]

        Az, Bz, Cz = self.get_SS_mats(params[f"{prefix}_n_z"], params[f"{prefix}_tau_z"], dt)
        Kz_out = Cz @ states[f"{prefix}_z"]
        new_z = Az @ states[f"{prefix}_z"] + Bz * states[f"{prefix}_Stim"]
        new_z = params[f"{prefix}_gamma"] * new_y + (1 - params[f"{prefix}_gamma"]) * new_z

        y0 = jnp.array(states[f"{prefix}_r"])
        args_tuple = (
            states[f"{prefix}_tau_r"],
            params[f"{prefix}_alpha"],
            params[f"{prefix}_beta"]
            )
        r_new = self.solver_func(y0, dt, self.derivatives, args_tuple)
        v_new = r_new + params[f"{prefix}_V_rest"]
        return {
            f"{prefix}_r": r_new,
            f"{prefix}_y": new_y,
            f"{prefix}_z": new_z,
            "v": v_new
            }

    def compute_current(self, states, v, params):
        return 0

    def init_state(self, states, v, params, delta_t):
        pass
