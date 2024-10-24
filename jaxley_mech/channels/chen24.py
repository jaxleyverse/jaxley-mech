from typing import Dict, Optional

import jax.debug
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.lax import select
from jaxley.channels import Channel

from jaxley_mech.solvers import SolverExtension

META = {
    "cell_type": "rod and cones",
    "species": "monkey and mouse",
    "papers": [
        "Chen, Q., Ingram, N. T., Baudin, J., Angueyra, J. M., Sinha, R., & Rieke, F. (2024). Light-adaptation clamp: A tool to predictably manipulate photoreceptor light responses. https://doi.org/10.7554/eLife.93795.1",
    ],
    "code": "https://github.com/chrischen2/photoreceptorLinearization",
}


class Phototransduction(Channel, SolverExtension):
    """Phototransduction channel"""

    def __init__(
        self,
        name: Optional[str] = None,
        solver: Optional[str] = None,
        rtol: float = 1e-8,
        atol: float = 1e-8,
        max_steps: int = 10,
    ):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        SolverExtension.__init__(self, solver, rtol, atol, max_steps)
        prefix = self._name
        self.channel_params = {  # Table 1 / Figure 8
            f"{prefix}_sigma": 22.0,  # σ, /s, Opsin decay rate constant
            f"{prefix}_gamma": 10.0,  # γ, unitless, Opsin gain
            f"{prefix}_phi": 22.0,  # φ, /s, PDE decay rate constant
            f"{prefix}_eta": 2000.0,  # η, /s, PDE dark activate rate
            f"{prefix}_G_dark": 20.0,  # μM, Dark GMP concentration
            f"{prefix}_k": 0.01,  # pA^2μM^-3, cGMP-to_current constant
            f"{prefix}_h": 4.0,  # unitless, Ca2+ GC cooperativity
            f"{prefix}_C_dark": 1.0,  # μM, Dark Ca2+ concentration
            f"{prefix}_beta": 9.0,  # β, /s, Ca2+ extrusion rate constant
            f"{prefix}_n": 3.0,  # unitless, cGMP channel cooperativity
            f"{prefix}_K_GC": 0.5,  # μM, Ca2+ GC affinity
            f"{prefix}_m": 4.0,  # unitless, Ca2+ GC cooperativity
            f"{prefix}_I_dark": 20**3 * 0.01,  # pA, Dark current
        }
        self.channel_states = {
            f"{prefix}_R": 0.0,
            f"{prefix}_P": 90.0,
            f"{prefix}_G": 1.0,
            f"{prefix}_S": 1.0,
            f"{prefix}_C": 0.336,
            f"{prefix}_Stim": 0.0,
        }
        self.current_name = f"iPhoto"
        self.META = META

    def derivatives(self, t, states, args):
        """Calculate the derivatives for the phototransduction system."""
        R, P, G, C = states
        gamma, sigma, phi, eta, beta, k, n, C_dark, I_dark, S, stim = args

        I = k * G**n  # Current through phototransduction channel
        q = beta * C_dark / I_dark

        dR_dt = gamma * stim - sigma * R  # eq(1)
        dP_dt = R - phi * P + eta  # eq(2)
        dC_dt = q * I - beta * C  # eq(5)
        dG_dt = S - P * G  # eq(3)

        return jnp.array([dR_dt, dP_dt, dG_dt, dC_dt])

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
        **kwargs,
    ):
        """Update state of phototransduction variables."""
        prefix = self._name
        dt /= 1000

        # Parameters
        gamma, sigma, phi, eta, beta, K_GC = (
            params[f"{prefix}_gamma"],
            params[f"{prefix}_sigma"],
            params[f"{prefix}_phi"],
            params[f"{prefix}_eta"],
            params[f"{prefix}_beta"],
            params[f"{prefix}_K_GC"],
        )
        k, m, n = params[f"{prefix}_k"], params[f"{prefix}_m"], params[f"{prefix}_n"]
        C_dark, G_dark = (
            params[f"{prefix}_C_dark"],
            params[f"{prefix}_G_dark"],
        )
        I_dark = G_dark**n * k

        # States
        Stim = states[f"{prefix}_Stim"]
        P, R, G, S, C = (
            states[f"{prefix}_P"],
            states[f"{prefix}_R"],
            states[f"{prefix}_G"],
            states[f"{prefix}_S"],
            states[f"{prefix}_C"],
        )
        y0 = jnp.array([R, P, G, C])
        args_tuple = (
            gamma,
            sigma,
            phi,
            eta,
            beta,
            k,
            n,
            C_dark,
            I_dark,
            S,
            Stim,
        )

        y_new = self.solver_func(y0, dt, self.derivatives, args_tuple)
        # Unpack the new states
        R_new, P_new, G_new, C_new = y_new

        S_max = eta / phi * G_dark * (1 + (C_dark / K_GC) ** m)
        S_new = S_max / (1 + (C / K_GC) ** m)  # New state of S, not its derivative

        return {
            f"{prefix}_R": R_new,
            f"{prefix}_P": P_new,
            f"{prefix}_G": G_new,
            f"{prefix}_S": S_new,
            f"{prefix}_C": C_new,
            f"{prefix}_Stim": Stim,
        }

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the phototransduction channel."""
        prefix = self._name
        G = states[f"{prefix}_G"]
        n, k = (
            params[f"{prefix}_n"],
            params[f"{prefix}_k"],
        )
        I = -k * G**n  # eq(4) #pA

        I *= 1e-9
        area = 2 * jnp.pi * params["length"] * params["radius"] * 1e-8  # um^2 to cm^2
        current_density = I / area

        return current_density

    def init_state(self, states, v, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        eta, phi, G_dark, C_dark = (
            params[f"{prefix}_eta"],
            params[f"{prefix}_phi"],
            params[f"{prefix}_G_dark"],
            params[f"{prefix}_C_dark"],
        )
        return {
            f"{prefix}_R": 0.0,
            f"{prefix}_P": eta / phi,
            f"{prefix}_G": G_dark,
            f"{prefix}_S": G_dark * eta / phi,
            f"{prefix}_C": C_dark,
            f"{prefix}_Stim": 0.0,
        }
