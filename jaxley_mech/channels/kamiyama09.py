from typing import Dict, Optional, Union

import jax.debug
import jax.numpy as jnp
from jax.lax import select
from jaxley.mechanisms.channels import Channel
from jaxley.mechanisms.solvers import (
    exponential_euler,
    save_exp,
    solve_gate_exponential,
)

from jaxley_mech.solvers import SolverExtension

META = {
    "cell_type": "rod photoreceptor",
    "species": "larval tiger salamander",
    "reference": "Kamiyama, et al. (2009).",
    "doi": "https://doi.org/10.1016/j.visres.2009.03.003",
    "note": "Inner segment of the rod photoreceptor",
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
        prefix = self.name
        self.params = {
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
            f"{prefix}_b": 0.25,  # μM / s * /pA, proportionality constant between Ca2+ influx and photocurrent
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
            "cell_type": "rod",
            "species": "salamander (Ambystoma tigrinum)",
            "reference": [
                "Torre, V., Forti, S., Menini, A., & Campani, M. (1990). Model of Phototransduction in Retinal Rods. Cold Spring Harbor Symposia on Quantitative Biology, 55(0), 563–573. https://doi.org/10.1101/SQB.1990.055.01.054",
                "Kamiyama, Y., Wu, S. M., & Usui, S. (2009). Simulation analysis of bandpass filtering properties of a rod photoreceptor network. Vision Research, 49(9), 970–978. https://doi.org/10.1016/j.visres.2009.03.003",
            ],
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
        dTr_dt = (
            epsilon * Rh * (T_tot - Tr)
            - beta1 * Tr
            + tau2 * PDE
            - tau1 * Tr * (PDE_tot - PDE)  # this term is not in Torre et al. (1990)
        )  # eq(8.3)
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

        # Choose solver
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
        current *= 1e-9  # pA to mA
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


class Leak(Channel):
    """Leakage current"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self.name
        self.params = {
            f"{prefix}_gLeak": 0.35e-3,  # S/cm^2
            f"{prefix}_eLeak": -77.0,  # mV
        }
        self.states = {}
        self.current_name = f"iLeak"
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """No state to update."""
        return {}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Given channel states and voltage, return the current through the channel."""
        prefix = self.name
        gLeak = params[f"{prefix}_gLeak"]
        return gLeak * (v - params[f"{prefix}_eLeak"])  # mS/cm^2 * mV = uA/cm^2

    def init_states(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        return {}


class Kv(Channel):
    """Delayed Rectifying Potassium Channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.params = {
            f"{self.name}_gKv": 2e-3,  # S/cm^2
            "eK": -74,  # mV
        }
        self.states = {
            f"{self.name}_m": 0.43,  # Initial value for n gating variable
            f"{self.name}_h": 0.999,  # Initial value for n gating variable
        }
        self.current_name = f"iKv"
        self.META = META
        self.META.update({"ion": "K"})

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self.name
        dt /= 1000
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        m_new = solve_gate_exponential(m, dt, *self.m_gate(v))
        h_new = solve_gate_exponential(h, dt, *self.h_gate(v))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self.name
        m = states[f"{prefix}_m"]
        h = states[f"{prefix}_h"]
        k_cond = params[f"{prefix}_gKv"] * m**3 * h
        return k_cond * (v - params["eK"])

    def init_states(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self.name
        alpha_m, beta_m = self.m_gate(v)
        alpha_h, beta_h = self.h_gate(v)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the n gating variable."""
        v += 1e-6
        alpha = 5 * (100 - v) / (save_exp((100 - v) / 42) - 1)
        beta = 9 * save_exp(-(v - 20) / 40)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        v += 1e-6
        alpha = 0.15 * save_exp(-v / 22)
        beta = 0.4125 / (save_exp((10 - v) / 7) + 1)
        return alpha, beta


class Hyper(Channel, SolverExtension):
    """Hyperpolarization-activated channel in the formulation of Markov model with 5 states"""

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
            f"{prefix}_gHyper": 3e-3,  # S/cm^2
            f"{prefix}_eHyper": -32.0,  # mV
        }
        self.states = {
            f"{prefix}_C1": 0.646,
            f"{prefix}_C2": 0.298,
            f"{prefix}_O1": 0.0517,
            f"{prefix}_O2": 0.00398,
            f"{prefix}_O3": 0.000115,
        }
        self.current_name = f"iHyper"
        self.META = {
            "species": "generic",
        }

    def derivatives(self, t, states, args):
        """Calculate the derivatives for the Markov states."""
        C1, C2, O1, O2, O3 = states
        v = args[0]
        alpha_h, beta_h = self.h_gate(
            v
        )  # Use the voltage (t here) to calculate alpha and beta

        # Transition rates according to the matrix K
        C1_to_C2 = 4 * alpha_h * C1
        C2_to_O1 = 3 * alpha_h * C2
        O1_to_O2 = 2 * alpha_h * O1
        O2_to_O3 = alpha_h * O2

        O3_to_O2 = 4 * beta_h * O3
        O2_to_O1 = 3 * beta_h * O2
        O1_to_C2 = 2 * beta_h * O1
        C2_to_C1 = beta_h * C2

        # Derivatives of each state
        dC1_dt = C2_to_C1 - C1_to_C2
        dC2_dt = C1_to_C2 + O1_to_C2 - C2_to_O1 - C2_to_C1
        dO1_dt = C2_to_O1 + O2_to_O1 - O1_to_O2 - O1_to_C2
        dO2_dt = O1_to_O2 + O3_to_O2 - O2_to_O3 - O2_to_O1
        dO3_dt = O2_to_O3 - O3_to_O2

        return jnp.array([dC1_dt, dC2_dt, dO1_dt, dO2_dt, dO3_dt])

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
        **kwargs,
    ):
        """Update the states using the specified solver."""
        prefix = self.name
        dt /= 1000  # Convert dt to seconds

        # Retrieve states
        C1 = states[f"{prefix}_C1"]
        C2 = states[f"{prefix}_C2"]
        O1 = states[f"{prefix}_O1"]
        O2 = states[f"{prefix}_O2"]
        O3 = states[f"{prefix}_O3"]

        y0 = jnp.array([C1, C2, O1, O2, O3])

        # Parameters for dynamics
        args_tuple = (v,)

        # Choose solver
        y_new = self.solver_func(y0, dt, self.derivatives, args_tuple)
        # Unpack new states
        C1_new, C2_new, O1_new, O2_new, O3_new = y_new

        return {
            f"{prefix}_C1": C1_new,
            f"{prefix}_C2": C2_new,
            f"{prefix}_O1": O1_new,
            f"{prefix}_O2": O2_new,
            f"{prefix}_O3": O3_new,
        }

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self.name
        O1 = states[f"{prefix}_O1"]
        O2 = states[f"{prefix}_O2"]
        O3 = states[f"{prefix}_O3"]
        gHyper = params[f"{prefix}_gHyper"] * (O1 + O2 + O3)
        return gHyper * (v - params[f"{prefix}_eHyper"])

    @staticmethod
    def h_gate(v):
        v += 1e-6
        alpha = 8 / (save_exp((v + 78) / 14) + 1)
        beta = 18 / (save_exp(-(v + 8) / 19) + 1)
        return alpha, beta

    def init_states(self, states, v, params, delta_t):
        return self.states


class Ca(Channel):
    """L-type calcium channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.params = {
            f"{self.name}_gCa": 0.7e-3,  # S/cm^2
        }
        self.states = {
            f"{self.name}_m": 0.436,  # Initial value for m gating variable
            f"{self.name}_h": 0.5,  # Initial value for h gating variable
            "eCa": 40.0,  # mV, dependent on CaNernstReversal
        }
        self.current_name = f"iCa"
        self.META = META
        self.META.update({"ion": "Ca"})

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables."""
        prefix = self.name
        m = states[f"{prefix}_m"]
        dt /= 1000  # convert to seconds
        m_new = solve_gate_exponential(m, dt, *self.m_gate(v))
        h_new = self.h_gate(v)
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new, "eCa": states["eCa"]}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self.name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        ca_cond = params[f"{prefix}_gCa"] * m**4 * h
        current = ca_cond * (v - states["eCa"])
        return current

    def init_states(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self.name

        alpha_m, beta_m = self.m_gate(v)
        h = self.h_gate(v)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": h,
            "eCa": 40.0,
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        v += 1e-6
        alpha = 3 * (80 - v) / (save_exp((80 - v) / 25.0) - 1)
        beta = 10 / (1 + save_exp((v + 38) / 7.0))
        return alpha, beta

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable."""
        v += 1e-6
        h = save_exp((40 - v) / 18) / (1 + save_exp((40 - v) / 18))
        return h


class CaPump(Channel, SolverExtension):
    """Calcium Pump Channel"""

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
            f"{prefix}_F": 9.648e4,  # Faraday's constant in C/mol
            f"{prefix}_V1": 3.812e-13,  # Compartment volume 1 in dm^3
            f"{prefix}_V2": 5.236e-13,  # Compartment volume 2 in dm^3
            f"{prefix}_D_Ca": 6e-8,  # Diffusion coefficient in dm^2/s
            f"{prefix}_delta": 3e-5,  # Membrane thickness in dm
            f"{prefix}_S1": 3.142e-8,  # Surface area in dm^2
            f"{prefix}_Lb1": 0.4,  # Binding rate constant 1 in s^-1μM^-1
            f"{prefix}_Lb2": 0.2,  # Binding rate constant 2 in s^-1
            f"{prefix}_Hb1": 100,  # Unbinding rate constant 1 in s^-1μM^-1
            f"{prefix}_Hb2": 90,  # Unbinding rate constant 2 in s^-1
            f"{prefix}_Bl": 500,  # Buffer low concentration in μM
            f"{prefix}_Bh": 300,  # Buffer high concentration in μM
            f"{prefix}_Jex": 20,  # External current in pA
            f"{prefix}_Jex2": 20,  # External current 2 in pA
            f"{prefix}_Kex": 2.3,  # External calcium concentration factor in μM
            f"{prefix}_Kex2": 0.5,  # External calcium concentration factor 2 in μM
            f"{prefix}_Cae": 0.01,  # External calcium concentration in μM
        }
        self.states = {
            f"Cas": 0.0966,  # Initial internal calcium concentration in mM
            f"{prefix}_Caf": 0.0966,  # Free intracellular calcium concentration in μM
            f"{prefix}_Cab_ls": 80.929,  # Bound buffer f concentration in μM
            f"{prefix}_Cab_hs": 29.068,  # Bound buffer h concentration in μM
            f"{prefix}_Cab_lf": 80.929,  # Bound buffer h concentration in μM
            f"{prefix}_Cab_hf": 29.068,  # Bound buffer h concentration in μM
        }
        self.current_name = f"iCa"
        self.META = {"reference": "Modified from Destexhe et al., 1994", "ion": "Ca"}

    def derivatives(self, t, states, args):
        """Calculate the derivatives for the calcium pump system."""
        Cas, Caf, Cab_ls, Cab_hs, Cab_lf, Cab_hf = states
        (
            F,
            V1,
            V2,
            D_Ca,
            delta,
            S1,
            Lb1,
            Lb2,
            Hb1,
            Hb2,
            Bl,
            Bh,
            Jex,
            Kex,
            Jex2,
            Kex2,
            Cae,
            iCa,
            v,
        ) = args

        v += 1e-6  # jitter to avoid division by zero
        iEx = Jex * save_exp(-(v + 14) / 70) * (Cas - Cae) / (Cas - Cae + Kex)
        iEx2 = Jex2 * (Cas - Cae) / (Cas - Cae + Kex2)

        # Free intracellular calcium concentration dynamics
        dCas_dt = (
            -1e-6 * (iCa + iEx + iEx2) / (2 * F * V1)
            - D_Ca * S1 * (Cas - Caf) / (delta * V1)
            - Lb1 * Cas * (Bl - Cab_ls)
            + Lb2 * Cab_ls
            - Hb1 * Cas * (Bh - Cab_hs)
            + Hb2 * Cab_hs
        )

        # Bound intracellular calcium concentration dynamics
        dCaf_dt = (
            D_Ca * S1 * (Cas - Caf) / (delta * V2)
            - Lb1 * Caf * (Bl - Cab_lf)
            + Lb2 * Cab_lf
            - Hb1 * Caf * (Bh - Cab_hf)
            + Hb2 * Cab_hf
        )

        dCab_ls_dt = Lb1 * Cas * (Bl - Cab_ls) - Lb2 * Cab_ls
        dCab_hs_dt = Hb1 * Cas * (Bh - Cab_hs) - Hb2 * Cab_hs
        dCab_lf_dt = Lb1 * Caf * (Bl - Cab_lf) - Lb2 * Cab_lf
        dCab_hf_dt = Hb1 * Caf * (Bh - Cab_hf) - Hb2 * Cab_hf

        return jnp.array(
            [dCas_dt, dCaf_dt, dCab_ls_dt, dCab_hs_dt, dCab_lf_dt, dCab_hf_dt]
        )

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
        **kwargs,
    ):
        """Update the state of calcium pump variables."""
        prefix = self.name
        dt /= 1000  # convert to seconds

        # States
        Cas = states[f"Cas"]
        Caf = states[f"{prefix}_Caf"]
        Cab_ls = states[f"{prefix}_Cab_ls"]
        Cab_hs = states[f"{prefix}_Cab_hs"]
        Cab_lf = states[f"{prefix}_Cab_lf"]
        Cab_hf = states[f"{prefix}_Cab_hf"]
        scale_factor = (2 * jnp.pi * params["length"] * params["radius"] * 1e-8) / 1e-9
        iCa = states["iCa"] * scale_factor  # mA/cm^2 to pA
        y0 = jnp.array([Cas, Caf, Cab_ls, Cab_hs, Cab_lf, Cab_hf])

        # Parameters
        args_tuple = (
            params[f"{prefix}_F"],
            params[f"{prefix}_V1"],
            params[f"{prefix}_V2"],
            params[f"{prefix}_D_Ca"],
            params[f"{prefix}_delta"],
            params[f"{prefix}_S1"],
            params[f"{prefix}_Lb1"],
            params[f"{prefix}_Lb2"],
            params[f"{prefix}_Hb1"],
            params[f"{prefix}_Hb2"],
            params[f"{prefix}_Bl"],
            params[f"{prefix}_Bh"],
            params[f"{prefix}_Jex"],
            params[f"{prefix}_Kex"],
            params[f"{prefix}_Jex2"],
            params[f"{prefix}_Kex2"],
            params[f"{prefix}_Cae"],
            iCa,
            v,
        )

        y_new = self.solver_func(y0, dt, self.derivatives, args_tuple)

        # Unpack the new states
        Cas_new, Caf_new, Cab_ls_new, Cab_hs_new, Cab_lf_new, Cab_hf_new = y_new

        return {
            f"Cas": Cas_new,
            f"{prefix}_Caf": Caf_new,
            f"{prefix}_Cab_ls": Cab_ls_new,
            f"{prefix}_Cab_hs": Cab_hs_new,
            f"{prefix}_Cab_lf": Cab_lf_new,
            f"{prefix}_Cab_hf": Cab_hf_new,
        }

    def compute_current(self, states, v, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_states(self, states, v, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self.name
        return {
            f"Cas": 0.0966,  # Initial internal calcium concentration in mM
            f"{prefix}_Caf": 0.0966,
            f"{prefix}_Cab_ls": 80.929,
            f"{prefix}_Cab_hs": 29.068,
            f"{prefix}_Cab_lf": 80.929,
            f"{prefix}_Cab_hf": 29.068,
        }


class CaNernstReversal(Channel):
    """Compute Calcium reversal from inner and outer concentration of calcium."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.params = {"Cao": 1600}  # μM
        self.states = {
            "eCa": 40.0,  # mV
            "Cas": 0.0966,  # μM
        }
        self.current_name = f"iCa"
        self.META = META
        self.META.update({"ion": "Ca"})

    def update_states(self, states, dt, v, params):
        """Update internal calcium concentration based on calcium current and decay."""
        Cao = params["Cao"]
        Cas = states["Cas"]
        eCa = -12.5 * jnp.log(Cas / Cao)
        return {"eCa": eCa, "Cas": Cas}

    def compute_current(self, states, v, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_states(self, states, v, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        return {"Cas": 0.0966}


class KCa(Channel):
    """Calcium-dependent potassium channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.params = {
            f"{self.name}_gKCa": 5e-3,  # S/cm^2
            f"{self.name}_Khalf": 0.3,  # mM, half-activation concentration
            # with an unfortunate name conflict with potassium K
            "eK": -74,  # mV
        }
        self.states = {
            f"{self.name}_m": 0.642,  # Initial value for m gating variable
            f"{self.name}_n": 0.1,  # Initial value for n gating variable
            "Cas": 0.0966,  # Initial internal calcium concentration in μM
        }
        self.current_name = f"iKCa"
        self.META = META
        self.META.update({"ion": "K"})

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self.name
        m = states[f"{prefix}_m"]
        dt /= 1000  # convert to seconds
        m_new = solve_gate_exponential(m, dt, *self.m_gate(v))
        n_new = self.n_gate(states["Cas"], params[f"{prefix}_Khalf"])
        return {f"{prefix}_m": m_new, f"{prefix}_n": n_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self.name
        m = states[f"{prefix}_m"]
        n = states[f"{prefix}_n"]
        k_cond = params[f"{prefix}_gKCa"] * m**2 * n
        return k_cond * (v - params["eK"])

    def init_states(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self.name
        Khalf = params[f"{prefix}_Khalf"]
        alpha_m, beta_m = self.m_gate(v)
        n = self.n_gate(v, Khalf)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_n": n,
            "Cas": 0.0966,
        }

    @staticmethod
    def m_gate(v):
        v += 1e-6
        alpha = 15 * (80 - v) / (save_exp((80 - v) / 40) - 1)
        beta = 20 * save_exp(-v / 35)
        return alpha, beta

    @staticmethod
    def n_gate(Cas, Khalf):
        """Calcium-dependent n gating variable."""
        return Cas / (Cas + Khalf)


class ClCa(Channel):
    """Calcium-dependent Chloride channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.params = {
            f"{self.name}_gClCa": 2e-3,  # S/cm^2
            f"{self.name}_Khalf": 0.37,  # uM, half-activation concentration
            f"{self.name}_eClCa": -20,  # mV
        }
        self.states = {
            f"{self.name}_m": 0.1,  # Initial value for n gating variable
            "Cas": 0.0966,  # Initial internal calcium concentration in μM
        }
        self.current_name = f"iClCa"
        self.META = META
        self.META.update({"ion": "Cl"})

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self.name
        m_new = self.m_gate(states["Cas"], params[f"{prefix}_Khalf"])
        return {f"{prefix}_m": m_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self.name
        m = states[f"{prefix}_m"]
        k_cond = params[f"{prefix}_gClCa"] * m
        return k_cond * (v - params[f"{prefix}_eClCa"])

    def init_states(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self.name
        Khalf = params[f"{prefix}_Khalf"]
        m = self.m_gate(v, Khalf)
        return {f"{prefix}_m": m, "Cas": 0.0966}

    @staticmethod
    def m_gate(Cas, Khalf):
        """Calcium-dependent n gating variable."""
        return 1 / (1 + save_exp((Khalf - Cas) / 0.09))
