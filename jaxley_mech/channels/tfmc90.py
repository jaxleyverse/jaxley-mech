from typing import Dict, Optional

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
        self.channel_params = {
            "alpha1": 50.0,  # /s, rate constant of Rh* inactivation
            "alpha2": 0.0003,  # /s, rate constant of the reaction Rhi -> Rh*
            "alpha3": 0.03,  # /s, rate constant of the decay of inactive rhodopsin
            "epsilon": 0.5,  # /s * /μM, rate constant of T* activation
            "T_tot": 1000.0,  # μM, total transduction
            "beta1": 2.5,  # /s, rate constant of T* inactivation
            "tau1": 0.2,  # /s * /μM, rate constant of PDE activation
            "tau2": 5.0,  # /s, rate constant of PDE inactivation
            "PDE_tot": 100.0,  # μM, total phosphodiesterase
            "sigma": 1.0,  # /s * /μM, proportionality constant
            "gamma_Ca": 50.0,  # /s, rate constant of Ca2+ extrusion in the absence of Ca2+ buffers mediated by the Na+/Ca2+ exchanger
            "C0": 0.1,  # μM, intracellular Ca2+ concentration at the steady state
            "b": 0.25,  # μM / s * /pA,proportionality constant between Ca2+ influx and photocurrent
            "k1": 0.2,  # /s * /μM, on rate constants for binding of Ca2+ to the buffer
            "k2": 0.8,  # /s, off rate constants for binding of Ca2+ to the buffer
            "eT": 500,  # μM, low affinity Ca2+ buffer concentration
            "V_max": 0.4,  # /s, cGMP hydrolysis in dark
            "A_max": 65.6,  # μM/s, guanylate cyclase activity
            "K_c": 0.1,  # nM, intracellular Ca2+ concentration halving the cyclase activity
            "J_max": 5040.0,  # pA, maximal cGMP-gated current in excised patches
        }
        self.channel_states = {
            "cGMP": 2.0,  # μM, cGMP concentration
            "Ca": 0.3,  # μM, intracellular Ca concentration
            "Cab": 34.9,  # μM, Bound intra Ca concentration
            "Rh": 0.0,  # μM, Rhodopsin concentration
            "Rhi": 0.0,  # μM, Activated rhodopsin concentration
            "Tr": 0.0,  # μM, Activated transducin concentration
            "PDE": 0.0,  # μM, Activated phosphodiesterase concentration
            "Jhv": 0.0,  # Rh*/s, theoretical flux of photoisomerization
        }
        self.current_name = f"iPhoto"
        self.META = {
            "reference": [
                "Torre, V., Forti, S., Menini, A., & Campani, M. (1990). Model of Phototransduction in Retinal Rods. Cold Spring Harbor Symposia on Quantitative Biology, 55(0), 563–573. https://doi.org/10.1101/SQB.1990.055.01.054",
                "Kamiyama, Y., Wu, S. M., & Usui, S. (2009). Simulation analysis of bandpass filtering properties of a rod photoreceptor network. Vision Research, 49(9), 970–978. https://doi.org/10.1016/j.visres.2009.03.003",
            ],
            "note": "The model is from Torre et al. (1990) but variable naming convention and default parameters are from Kamiyama et al. (2009).",
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
        Jhv = kwargs["Jhv"] if "Jhv" in kwargs else states["Jhv"]

        Rh, Rhi, Tr, PDE = states["Rh"], states["Rhi"], states["Tr"], states["PDE"]
        Ca, Cab, cGMP = states["Ca"], states["Cab"], states["cGMP"]
        alpha1, alpha2, alpha3 = params["alpha1"], params["alpha2"], params["alpha3"]
        beta1, tau1, tau2 = params["beta1"], params["tau1"], params["tau2"]
        T_tot, PDE_tot = params["T_tot"], params["PDE_tot"]
        gamma_Ca, k1, k2 = params["gamma_Ca"], params["k1"], params["k2"]
        b, V_max, A_max, K_c = (
            params["b"],
            params["V_max"],
            params["A_max"],
            params["K_c"],
        )
        sigma, epsilon = params["sigma"], params["epsilon"]
        C0 = params["C0"]
        eT = params["eT"]
        J_max = params["J_max"]

        J_Ca = J_max * cGMP**3 / (cGMP**3 + 1000)

        # Update Rhodopsin concentrations
        dRh_dt = Jhv - alpha1 * Rh + alpha2 * Rhi
        dRhi_dt = alpha1 * Rh - (alpha2 + alpha3) * Rhi

        # Update Transducin and PDE concentrations
        dTr_dt = (
            epsilon * Rh * (T_tot - Tr)
            - beta1 * Tr
            + tau2 * PDE
            - tau1 * Tr * (PDE_tot - PDE)
        )
        dPDE_dt = tau1 * Tr * (PDE_tot - PDE) - tau2 * PDE

        # Update cGMP and G-protein concentrations
        dCa_dt = b * J_Ca - gamma_Ca * (Ca - C0) - k1 * (eT - Cab) * Ca + k2 * Cab
        dCab_dt = k1 * (eT - Cab) * Ca - k2 * Cab
        dcGMP_dt = A_max / (1 + (Ca / K_c) ** 4) - cGMP * (V_max + sigma * PDE)

        # Update states
        Rh_new = Rh + dRh_dt * dt
        Rhi_new = Rhi + dRhi_dt * dt
        Tr_new = Tr + dTr_dt * dt
        PDE_new = PDE + dPDE_dt * dt
        Ca_new = Ca + dCa_dt * dt
        Cab_new = Cab + dCab_dt * dt
        cGMP_new = cGMP + dcGMP_dt * dt

        new_states = {
            "Rh": Rh_new,
            "Rhi": Rhi_new,
            "Tr": Tr_new,
            "PDE": PDE_new,
            "Ca": Ca_new,
            "Cab": Cab_new,
            "cGMP": cGMP_new,
        }
        if not "Jhv" in kwargs:
            new_states.update({"Jhv": Jhv})

        return new_states

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the phototransduction channel."""
        cGMP = states["cGMP"]
        J_max = params["J_max"]
        J = J_max * cGMP**3 / (cGMP**3 + 1000)
        current = -J * (1.0 - jnp.exp(v - 8.5) / 17.0)
        return current

    def init_state(self, v, params):
        """Initialize the state at fixed point of gate dynamics."""
        return {
            "cGMP": 2.0,  # μM, cGMP concentration
            "Ca": 0.3,  # μM, intracellular Ca concentration
            "Cab": 34.9,  # μM, Bound intra Ca concentration
            "Rh": 0.0,  # μM, Rhodopsin concentration
            "Rhi": 0.0,  # μM, Activated rhodopsin concentration
            "Tr": 0.0,  # μM, Activated transducin concentration
            "PDE": 0.0,  # μM, Activated phosphodiesterase concentration
            "Jhv": 0.0,  # Rh*/s, theoretical flux of photoisomerization
        }
