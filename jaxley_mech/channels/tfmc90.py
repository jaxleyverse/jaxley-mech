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
        prefix = self._name
        self.channel_params = {
            f"{prefix}_alpha1": 50.0,  # /s, rate constant of Rh* inactivation
            f"{prefix}_alpha2": 0.0003,  # /s, rate constant of the reaction Rhi -> Rh*
            f"{prefix}_alpha3": 0.03,  # /s, rate constant of the decay of inactive rhodopsin
            f"{prefix}_epsilon": 0.5,  # /s * /μM, rate constant of T* activation
            f"{prefix}_T_tot": 1000.0,  # μM, total transduction
            f"{prefix}_beta1": 2.5,  # /s, rate constant of T* inactivation
            f"{prefix}_tau1": 0.2,  # /s * /μM, rate constant of PDE activation
            f"{prefix}_tau2": 5.0,  # /s, rate constant of PDE inactivation
            f"{prefix}_PDE_tot": 100.0,  # μM, total phosphodiesterase
            f"{prefix}_sigma": 1.0,  # /s * /μM, proportionality constant
            f"{prefix}_gamma_Ca": 50.0,  # /s, rate constant of Ca2+ extrusion in the absence of Ca2+ buffers mediated by the Na+/Ca2+ exchanger
            f"{prefix}_C0": 0.1,  # μM, intracellular Ca2+ concentration at the steady state
            f"{prefix}_b": 0.25,  # μM / s * /pA,proportionality constant between Ca2+ influx and photocurrent
            f"{prefix}_k1": 0.2,  # /s * /μM, on rate constants for binding of Ca2+ to the buffer
            f"{prefix}_k2": 0.8,  # /s, off rate constants for binding of Ca2+ to the buffer
            f"{prefix}_eT": 500,  # μM, low affinity Ca2+ buffer concentration
            f"{prefix}_V_max": 0.4,  # /s, cGMP hydrolysis in dark
            f"{prefix}_A_max": 65.6,  # μM/s, guanylate cyclase activity
            f"{prefix}_K": 10,  # # μM, half-saturation constant for cGMP hydrolysis
            f"{prefix}_K_c": 0.1,  # nM, intracellular Ca2+ concentration halving the cyclase activity
            f"{prefix}_J_max": 5040.0,  # pA, maximal cGMP-gated current in excised patches
        }
        self.channel_states = {
            f"{prefix}_cGMP": 2.0,  # μM, cGMP concentration
            f"{prefix}_Ca": 0.3,  # μM, intracellular Ca concentration
            f"{prefix}_Cab": 34.9,  # μM, Bound intra Ca concentration
            f"{prefix}_Rh": 0.0,  # μM, Rhodopsin concentration
            f"{prefix}_Rhi": 0.0,  # μM, Activated rhodopsin concentration
            f"{prefix}_Tr": 0.0,  # μM, Activated transducin concentration
            f"{prefix}_PDE": 0.0,  # μM, Activated phosphodiesterase concentration
            f"{prefix}_Jhv": 0.0,  # Rh*/s, theoretical flux of photoisomerization
        }
        self.current_name = f"iPhoto"
        self.META = {
            "cell_type": "rod",
            "species": "salamander (Ambystoma tigrinum)",
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
        prefix = self._name
        dt /= 1000  # the original implementation is in the time scale of seconds
        # but the solver is in milliseconds
        Jhv = states[f"{prefix}_Jhv"]

        Rh, Rhi, Tr, PDE = (
            states[f"{prefix}_Rh"],
            states[f"{prefix}_Rhi"],
            states[f"{prefix}_Tr"],
            states[f"{prefix}_PDE"],
        )
        Ca, Cab, cGMP = (
            states[f"{prefix}_Ca"],
            states[f"{prefix}_Cab"],
            states[f"{prefix}_cGMP"],
        )
        alpha1, alpha2, alpha3 = (
            params[f"{prefix}_alpha1"],
            params[f"{prefix}_alpha2"],
            params[f"{prefix}_alpha3"],
        )
        beta1, tau1, tau2 = (
            params[f"{prefix}_beta1"],
            params[f"{prefix}_tau1"],
            params[f"{prefix}_tau2"],
        )
        T_tot, PDE_tot = params[f"{prefix}_T_tot"], params[f"{prefix}_PDE_tot"]
        gamma_Ca, k1, k2 = (
            params[f"{prefix}_gamma_Ca"],
            params[f"{prefix}_k1"],
            params[f"{prefix}_k2"],
        )
        b, V_max, A_max, K_c = (
            params[f"{prefix}_b"],
            params[f"{prefix}_V_max"],
            params[f"{prefix}_A_max"],
            params[f"{prefix}_K_c"],
        )
        sigma, epsilon = params[f"{prefix}_sigma"], params[f"{prefix}_epsilon"]
        C0 = params[f"{prefix}_C0"]
        eT = params[f"{prefix}_eT"]

        J_Ca = -states[self.current_name]

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

        return {
            f"{prefix}_Rh": Rh_new,
            f"{prefix}_Rhi": Rhi_new,
            f"{prefix}_Tr": Tr_new,
            f"{prefix}_PDE": PDE_new,
            f"{prefix}_Ca": Ca_new,
            f"{prefix}_Cab": Cab_new,
            f"{prefix}_cGMP": cGMP_new,
            f"{prefix}_Jhv": Jhv,
        }

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the phototransduction channel."""
        prefix = self._name
        cGMP = states[f"{prefix}_cGMP"]
        J_max, K = params[f"{prefix}_J_max"], params[f"{prefix}_K"]
        J = J_max * cGMP**3 / (cGMP**3 + K**3)
        current = -J * (1.0 - jnp.exp(v - 8.5) / 17.0)

        return current

    def init_state(self, v, params):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        return {
            f"{prefix}_cGMP": 2.0,  # μM, cGMP concentration
            f"{prefix}_Ca": 0.3,  # μM, intracellular Ca concentration
            f"{prefix}_Cab": 34.9,  # μM, Bound intra Ca concentration
            f"{prefix}_Rh": 0.0,  # μM, Rhodopsin concentration
            f"{prefix}_Rhi": 0.0,  # μM, Activated rhodopsin concentration
            f"{prefix}_Tr": 0.0,  # μM, Activated transducin concentration
            f"{prefix}_PDE": 0.0,  # μM, Activated phosphodiesterase concentration
            f"{prefix}_Jhv": 0.0,  # Rh*/s, theoretical flux of photoisomerization
        }
