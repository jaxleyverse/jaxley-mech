from typing import Dict, Optional

import jax.debug
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.lax import select
from jaxley.channels import Channel
from jaxley.solver_gate import save_exp, solve_gate_exponential


class Phototransduction(Channel):
    """Phototransduction channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {  # Table 1 / Fgirue 8
            f"{prefix}_sigma": 22,  # σ, /s, Opsin decay rate constant
            f"{prefix}_gamma": 10,  # γ, unitless, Opsin gain
            f"{prefix}_phi": 22,  # φ, /s, PDE decay rate constant
            f"{prefix}_eta": 2000,  # η, /s, PDE dark activate rate
            f"{prefix}_G_dark": 20,  # μM, Dark GMP concentration
            f"{prefix}_k": 0.01,  # pA^2μM^-3, cGMP-to_current constant
            f"{prefix}_h": 4,  # unitless, Ca2+ GC cooperativity
            f"{prefix}_Ca_dark": 1,  # μM, Dark Ca2+ concentration
            f"{prefix}_beta": 9,  # β, /s, Ca2+ extrusion rate constant
            f"{prefix}_n": 3,  # unitless, cGMP channel cooperativity
            f"{prefix}_q": 1.0,  # unitless, Fraction of current carried by calcium
            f"{prefix}_S_max": 30909,  # /s, maximal cGMP synthesis rate by GC
            f"{prefix}_K_GC": 0.5,  # μM, Ca2+ GC affinity
            f"{prefix}_m": 4,  # unitless, Ca2+ GC cooperativity
            f"{prefix}_I_dark": 20**3 * 0.01,  # pA, Dark current
        }
        self.channel_states = {
            f"{prefix}_R": 0.0,
            f"{prefix}_P": 90.0,
            f"{prefix}_G": 1,
            f"{prefix}_S": 1,
            f"{prefix}_C": 0.336,
            f"{prefix}_Stim": 0.0,
        }
        self.current_name = f"iPhoto"
        self.META = {
            "cell_type": "rod",
            "species": "monkeys (Macaca fascicularis, nemestrina, mulatta)",
            "reference": [
                "Angueyra, J. M., Baudin, J., Schwartz, G. W., & Rieke, F. (2022). Predicting and Manipulating Cone Responses to Naturalistic Inputs. The Journal of Neuroscience, 42(7), 1254–1274. https://doi.org/10.1523/JNEUROSCI.0793-21.2021"
            ],
        }

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

        gamma, sigma, phi, eta, q, beta, S_max, K_GC, m = (
            params[f"{prefix}_gamma"],
            params[f"{prefix}_sigma"],
            params[f"{prefix}_phi"],
            params[f"{prefix}_eta"],
            params[f"{prefix}_q"],
            params[f"{prefix}_beta"],
            params[f"{prefix}_S_max"],
            params[f"{prefix}_K_GC"],
            params[f"{prefix}_m"],
        )
        k, n = params[f"{prefix}_k"], params[f"{prefix}_n"]
        I_dark, C_dark, G_dark = (
            params[f"{prefix}_I_dark"],
            params[f"{prefix}_Ca_dark"],
            params[f"{prefix}_G_dark"],
        )
        q = beta * C_dark / I_dark
        S_max = eta / phi * G_dark * (1 + (C_dark / K_GC) ** m)

        Stim = states[f"{prefix}_Stim"]
        P, R, G, S, C = (
            states[f"{prefix}_P"],
            states[f"{prefix}_R"],
            states[f"{prefix}_G"],
            states[f"{prefix}_S"],
            states[f"{prefix}_C"],
        )
        I = states[self.current_name]

        # stimulus activates opsin molecules
        dR_dt = gamma * Stim - sigma * R  # eq(1)

        # active opsin molecules activate phosphodiesterase (PDE) molecules through transducin
        dP_dt = R - phi * P + eta  # eq(2)

        # concentration of cGMP in the outer segment depends on the activity of PDE
        dG_dt = S - P * G  # eq(3)

        # ca2+
        I = k * G**n
        dC_dt = q * I - beta * C  # eq(5)
        # jax.debug.print("dC_dt={dC_dt}", dC_dt=dC_dt)

        # S
        S_new = S_max / (1 + (C / K_GC) ** m)

        # Update states
        R_new = R + dR_dt * dt
        P_new = P + dP_dt * dt
        G_new = G + dG_dt * dt
        C_new = C + dC_dt * dt

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
        I = -k * G**n  # eq(4)
        return I

    def init_state(self, v, params):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        eta, phi, G_dark, Ca_dark = (
            params[f"{prefix}_eta"],
            params[f"{prefix}_phi"],
            params[f"{prefix}_G_dark"],
            params[f"{prefix}_Ca_dark"],
        )
        return {
            f"{prefix}_R": 0.0,
            f"{prefix}_P": eta / phi,
            f"{prefix}_G": G_dark,
            f"{prefix}_S": G_dark * eta / phi,
            f"{prefix}_C": Ca_dark,
            f"{prefix}_Stim": 0.0,
        }
