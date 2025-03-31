from typing import Dict, Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.lax import select
from jaxley.mechanisms.channels import Channel
from jaxley.mechanisms.solvers import save_exp, solve_gate_exponential

from jaxley_mech.mechanisms.solvers import SolverExtension


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

        prefix = self.name
        self.params = {
            f"{prefix}_alpha1": 20.0,  # /s, rate constant of Rh* inactivation
            f"{prefix}_alpha2": 0.0005,  # /s, rate constant of the reaction Rhi -> Rh*
            f"{prefix}_alpha3": 0.05,  # /s, rate constant of the decay of inactive rhodopsin
            f"{prefix}_epsilon": 0.5,  # /s * /μM, rate constant of T* activation
            f"{prefix}_T_tot": 1000.0,  # μM, total transduction
            f"{prefix}_beta1": 10.6,  # /s, rate constant of T* inactivation
            f"{prefix}_tau1": 0.1,  # /s * /μM, rate constant of PDE activation
            f"{prefix}_tau2": 10.0,  # /s, rate constant of PDE inactivation
            f"{prefix}_PDE_tot": 100.0,  # μM, total phosphodiesterase
            f"{prefix}_sigma": 1.0,  # /s * /μM, proportionality constant
            f"{prefix}_gamma_Ca": 50.0,  # /s, rate constant of Ca2+ extrusion in the absence of Ca2+ buffers mediated by the Na+/Ca2+ exchanger
            f"{prefix}_C0": 0.1,  # μM, intracellular Ca2+ concentration at the steady state
            f"{prefix}_b": 0.25,  # μM / s * /pA,proportionality constant between Ca2+ influx and photocurrent,
            # b is set to 0.625 in the paper, but only with 0.25 can we reproduce Figure 2B and 2D
            f"{prefix}_k1": 0.2,  # /s * /μM, on rate constants for binding of Ca2+ to the buffer
            f"{prefix}_k2": 0.8,  # /s, off rate constants for binding of Ca2+ to the buffer
            f"{prefix}_eT": 500,  # μM, low affinity Ca2+ buffer concentration
            f"{prefix}_V_max": 0.4,  # /s, cGMP hydrolysis in dark
            f"{prefix}_A_max": 65.6,  # μM/s, guanylate cyclase activity
            f"{prefix}_K": 10,  # # μM, half-saturation constant for cGMP hydrolysis
            f"{prefix}_K_c": 0.1,  # nM, intracellular Ca2+ concentration halving the cyclase activity
            f"{prefix}_J_max": 5040.0,  # pA, maximal cGMP-gated current in excised patches
        }
        self.states = {
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
            "cell_type": "rod photoreceptor",
            "species": "tiger salamander",
            "reference": "Torre, et al. (1990)",
            "doi": "https://doi.org/10.1101/SQB.1990.055.01.054",
            "note": "The model is from Torre et al. (1990) but variable naming convention and default parameters are from Kamiyama et al. (2009).",
        }

    def derivatives(self, t, states, args):
        """Calculate the derivatives for the phototransduction system."""
        Rh, Rhi, Tr, PDE, Ca, Cab, cGMP = states
        (
            alpha1,
            alpha2,
            alpha3,
            epsilon,
            T_tot,
            beta1,
            tau1,
            tau2,
            PDE_tot,
            gamma_Ca,
            k1,
            k2,
            sigma,
            A_max,
            V_max,
            K_c,
            J_max,
            K,
            b,
            C0,
            eT,
            Jhv,
        ) = args

        J_Ca = (
            J_max * cGMP**3 / (cGMP**3 + K**3)
        )  # eq(12) # voltage-independent current

        # Update Rhodopsin concentrations
        dRh_dt = Jhv - alpha1 * Rh + alpha2 * Rhi  # eq(8.1) of Torre et al. (1990)
        dRhi_dt = alpha1 * Rh - (alpha2 + alpha3) * Rhi  # eq(8.2)

        # Update Transducin and PDE concentrations
        dTr_dt = epsilon * Rh * (T_tot - Tr) - beta1 * Tr + tau2 * PDE  # eq(8.3)
        dPDE_dt = tau1 * Tr * (PDE_tot - PDE) - tau2 * PDE  # eq(8.4)

        # Update cGMP and G-protein concentrations
        dCa_dt = (
            b * J_Ca - gamma_Ca * (Ca - C0) - k1 * (eT - Cab) * Ca + k2 * Cab
        )  # eq(9)
        dCab_dt = k1 * (eT - Cab) * Ca - k2 * Cab  # eq(10)
        dcGMP_dt = A_max / (1 + (Ca / K_c) ** 4) - cGMP * (
            V_max + sigma * PDE
        )  # eq(11)

        return jnp.array([dRh_dt, dRhi_dt, dTr_dt, dPDE_dt, dCa_dt, dCab_dt, dcGMP_dt])

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
        **kwargs,
    ):
        """Update state of phototransduction variables."""
        prefix = self.name
        dt /= 1000  # Convert to seconds

        # Retrieve states
        Rh = states[f"{prefix}_Rh"]
        Rhi = states[f"{prefix}_Rhi"]
        Tr = states[f"{prefix}_Tr"]
        PDE = states[f"{prefix}_PDE"]
        Ca = states[f"{prefix}_Ca"]
        Cab = states[f"{prefix}_Cab"]
        cGMP = states[f"{prefix}_cGMP"]

        y0 = jnp.array([Rh, Rhi, Tr, PDE, Ca, Cab, cGMP])

        # Parameters for dynamics
        args_tuple = (
            params[f"{prefix}_alpha1"],
            params[f"{prefix}_alpha2"],
            params[f"{prefix}_alpha3"],
            params[f"{prefix}_epsilon"],
            params[f"{prefix}_T_tot"],
            params[f"{prefix}_beta1"],
            params[f"{prefix}_tau1"],
            params[f"{prefix}_tau2"],
            params[f"{prefix}_PDE_tot"],
            params[f"{prefix}_gamma_Ca"],
            params[f"{prefix}_k1"],
            params[f"{prefix}_k2"],
            params[f"{prefix}_sigma"],
            params[f"{prefix}_A_max"],
            params[f"{prefix}_V_max"],
            params[f"{prefix}_K_c"],
            params[f"{prefix}_J_max"],
            params[f"{prefix}_K"],
            params[f"{prefix}_b"],
            params[f"{prefix}_C0"],
            params[f"{prefix}_eT"],
            states[f"{prefix}_Jhv"],
        )

        y_new = self.solver_func(y0, dt, self.derivatives, args_tuple)

        # Unpack new states
        Rh_new, Rhi_new, Tr_new, PDE_new, Ca_new, Cab_new, cGMP_new = y_new

        return {
            f"{prefix}_Rh": Rh_new,
            f"{prefix}_Rhi": Rhi_new,
            f"{prefix}_Tr": Tr_new,
            f"{prefix}_PDE": PDE_new,
            f"{prefix}_Ca": Ca_new,
            f"{prefix}_Cab": Cab_new,
            f"{prefix}_cGMP": cGMP_new,
        }

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the phototransduction channel."""
        prefix = self.name
        cGMP = states[f"{prefix}_cGMP"]
        J_max, K = params[f"{prefix}_J_max"], params[f"{prefix}_K"]
        J = J_max * cGMP**3 / (cGMP**3 + K**3)  # eq(12)
        current = -J * (1.0 - jnp.exp(v - 8.5) / 17.0)  # from Kamiyama et al. (2009)

        current *= 1e-9
        area = 2 * jnp.pi * params["length"] * params["radius"] * 1e-8  # um^2 to cm^2
        current_density = current / area  # mA/cm^2
        return current_density

    def init_states(self, states, v, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self.name
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
