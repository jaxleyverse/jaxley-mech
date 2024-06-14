from typing import Dict, Optional, Union

import jax.debug
import jax.numpy as jnp
from jax.lax import select
from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler, save_exp, solve_gate_exponential

META = {
    "cell_type": "rod (inner segment)",
    "species": "Larval tiger salamanders (Ambystoma tigrinum)",
    "reference": "Kamiyama, Y., Wu, S. M., & Usui, S. (2009). Simulation analysis of bandpass filtering properties of a rod photoreceptor network. Vision Research, 49(9), 970–978. https://doi.org/10.1016/j.visres.2009.03.003",
}


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
        J = J_max * cGMP**3 / (cGMP**3 + K**3)  # eq(12)
        current = -J * (1.0 - jnp.exp(v - 8.5) / 17.0)  # from Kamiyama et al. (2009)
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


class Leak(Channel):
    """Leakage current"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gLeak": 0.35e-3,  # S/cm^2
            f"{prefix}_eLeak": -77.0,  # mV
        }
        self.channel_states = {}
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
        prefix = self._name
        gLeak = params[f"{prefix}_gLeak"] * 1000  # mS/cm^2
        return gLeak * (v - params[f"{prefix}_eLeak"])  # mS/cm^2 * mV = uA/cm^2

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        return {}


class Kv(Channel):
    """Delayed Rectifying Potassium Channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gKv": 2e-3,  # S/cm^2
            "eK": -74,  # mV
        }
        self.channel_states = {
            f"{self._name}_m": 0.43,  # Initial value for n gating variable
            f"{self._name}_h": 0.999,  # Initial value for n gating variable
        }
        self.current_name = f"iKv"
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self._name
        dt /= 1000
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        m_new = solve_gate_exponential(m, dt, *self.m_gate(v))
        h_new = solve_gate_exponential(h, dt, *self.h_gate(v))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        h = states[f"{prefix}_h"]
        k_cond = params[f"{prefix}_gKv"] * m**3 * h * 1000
        return k_cond * (v - params["eK"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
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


class Hyper(Channel):
    """Hyperpolarization-activated channel in the formulation of Markov model with 5 states"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gHyper": 3e-3,  # S/cm^2
            f"{prefix}_eHyper": -32.0,  # mV
        }
        self.channel_states = {
            f"{prefix}_C1": 0.646,
            f"{prefix}_C2": 0.298,
            f"{prefix}_O1": 0.0517,
            f"{prefix}_O2": 0.00398,
            f"{prefix}_O3": 0.000115,
        }
        self.current_name = f"iHyper"
        self.META = {
            "reference": [],
            "Species": "generic",
        }

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""
        prefix = self._name
        dt /= 1000
        alpha_h, beta_h = self.h_gate(v)

        # Transition rates according to the matrix K

        C1_to_C2 = 4 * alpha_h * states[f"{prefix}_C1"]
        C2_to_O1 = 3 * alpha_h * states[f"{prefix}_C2"]
        O1_to_O2 = 2 * alpha_h * states[f"{prefix}_O1"]
        O2_to_O3 = alpha_h * states[f"{prefix}_O2"]

        O3_to_O2 = 4 * beta_h * states[f"{prefix}_O3"]
        O2_to_O1 = 3 * beta_h * states[f"{prefix}_O2"]
        O1_to_C2 = 2 * beta_h * states[f"{prefix}_O1"]
        C2_to_C1 = beta_h * states[f"{prefix}_C2"]

        new_C1 = states[f"{prefix}_C1"] + dt * (C2_to_C1 - C1_to_C2)
        new_C2 = states[f"{prefix}_C2"] + dt * (
            C1_to_C2 + O1_to_C2 - C2_to_O1 - C2_to_C1
        )
        new_O1 = states[f"{prefix}_O1"] + dt * (
            C2_to_O1 + O2_to_O1 - O1_to_O2 - O1_to_C2
        )
        new_O2 = states[f"{prefix}_O2"] + dt * (
            O1_to_O2 + O3_to_O2 - O2_to_O3 - O2_to_O1
        )
        new_O3 = states[f"{prefix}_O3"] + dt * (O2_to_O3 - O3_to_O2)

        return {
            f"{prefix}_C1": new_C1,
            f"{prefix}_C2": new_C2,
            f"{prefix}_O1": new_O1,
            f"{prefix}_O2": new_O2,
            f"{prefix}_O3": new_O3,
        }

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        O1 = states[f"{prefix}_O1"]
        O2 = states[f"{prefix}_O2"]
        O3 = states[f"{prefix}_O3"]
        gHyper = params[f"{prefix}_gHyper"] * (O1 + O2 + O3) * 1000
        return gHyper * (v - params[f"{prefix}_eHyper"])

    @staticmethod
    def h_gate(v):
        v += 1e-6
        alpha = 8 / (save_exp((v + 78) / 14) + 1)
        beta = 18 / (save_exp(-(v + 8) / 19) + 1)
        return alpha, beta

    def init_state(self, v, params):
        return self.channel_states


class Ca(Channel):
    """L-type calcium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gCa": 0.7e-3,  # S/cm^2
        }
        self.channel_states = {
            f"{self._name}_m": 0.436,  # Initial value for m gating variable
            f"{self._name}_h": 0.5,  # Initial value for h gating variable
            "eCa": 40.0,  # mV, dependent on CaNernstReversal
        }
        self.current_name = f"iCa"
        self.META = META

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        dt /= 1000  # convert to seconds
        m_new = solve_gate_exponential(m, dt, *self.m_gate(v))
        h_new = self.h_gate(v)
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new, "eCa": states["eCa"]}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        ca_cond = params[f"{prefix}_gCa"] * m**4 * h * 1000
        current = ca_cond * (v - states["eCa"])
        return current

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name

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


class CaPump(Channel):
    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        name = self._name
        self.channel_params = {
            f"{name}_F": 9.648e4,  # Faraday's constant in C/mol
            f"{name}_V1": 3.812e-13,  # Compartment volume 1 in dm^3
            f"{name}_V2": 5.236e-13,  # Compartment volume 2 in dm^3
            f"{name}_D_Ca": 6e-8,  # Diffusion coefficient in dm^2/s
            f"{name}_delta": 3e-5,  # Membrane thickness in dm
            f"{name}_S1": 3.142e-8,  # Surface area in dm^2
            f"{name}_Lb1": 0.4,  # Binding rate constant 1 in s^-1μM^-1
            f"{name}_Lb2": 0.2,  # Binding rate constant 2 in s^-1
            f"{name}_Hb1": 100,  # Unbinding rate constant 1 in s^-1μM^-1
            f"{name}_Hb2": 90,  # Unbinding rate constant 2 in s^-1
            f"{name}_Bl": 500,  # Buffer low concentration in μM
            f"{name}_Bh": 300,  # Buffer high concentration in μM
            f"{name}_Jex": 20,  # External current in pA
            f"{name}_Jex2": 20,  # External current 2 in pA
            f"{name}_Kex": 2.3,  # External calcium concentration factor in μM
            f"{name}_Kex2": 0.5,  # External calcium concentration factor 2 in μM
            f"{name}_Cae": 0.01,  # External calcium concentration in μM
        }
        self.channel_states = {
            f"Cas": 0.0966,  # Initial internal calcium concentration in mM
            f"{name}_Caf": 0.0966,  # Free intracellular calcium concentration in μM
            f"{name}_Cab_ls": 80.929,  # Bound buffer f concentration in μM
            f"{name}_Cab_hs": 29.068,  # Bound buffer h concentration in μM
            f"{name}_Cab_lf": 80.929,  # Bound buffer h concentration in μM
            f"{name}_Cab_hf": 29.068,  # Bound buffer h concentration in μM
        }
        self.current_name = f"iCa"
        self.META = {
            "reference": "Modified from Destexhe et al., 1994",
            "mechanism": "Calcium dynamics",
        }

    def update_states(self, states, dt, v, params):
        """Update the states based on differential equations."""
        prefix = self._name
        v += 1e-6  # jitter to avoid division by zero
        dt /= 1000  # convert to seconds
        F, V1, V2, D_Ca, delta, S1 = (
            params[f"{prefix}_F"],
            params[f"{prefix}_V1"],
            params[f"{prefix}_V2"],
            params[f"{prefix}_D_Ca"],
            params[f"{prefix}_delta"],
            params[f"{prefix}_S1"],
        )
        Lb1, Lb2, Hb1, Hb2, Bl, Bh = (
            params[f"{prefix}_Lb1"],
            params[f"{prefix}_Lb2"],
            params[f"{prefix}_Hb1"],
            params[f"{prefix}_Hb2"],
            params[f"{prefix}_Bl"],
            params[f"{prefix}_Bh"],
        )
        Jex, Kex, Jex2, Kex2, Cae = (
            params[f"{prefix}_Jex"],
            params[f"{prefix}_Kex"],
            params[f"{prefix}_Jex2"],
            params[f"{prefix}_Kex2"],
            params[f"{prefix}_Cae"],
        )

        Cas = states[f"Cas"]
        Caf = states[f"{prefix}_Caf"]
        Cab_ls = states[f"{prefix}_Cab_ls"]
        Cab_hs = states[f"{prefix}_Cab_hs"]
        Cab_lf = states[f"{prefix}_Cab_lf"]
        Cab_hf = states[f"{prefix}_Cab_hf"]

        iCa = states["iCa"]
        iEx = Jex * save_exp(-(v + 14) / 70) * (Cas - Cae) / (Cas - Cae + Kex)
        iEx2 = Jex2 * (Cas - Cae) / (Cas - Cae + Kex2)

        # Free intracellular calcium concentration dynamics
        dCas_dt = (
            -10e-6 * (iCa + iEx + iEx2) / (2 * F * V1)
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

        Cas = Cas + dCas_dt * dt
        Caf = Caf + dCaf_dt * dt
        Cab_ls = Cab_ls + dCab_ls_dt * dt
        Cab_hs = Cab_hs + dCab_hs_dt * dt
        Cab_lf = Cab_lf + dCab_lf_dt * dt
        Cab_hf = Cab_hf + dCab_hf_dt * dt

        # Cas = jnp.maximum(Cas + dCas_dt * dt, 0)
        # Caf = jnp.maximum(Caf + dCaf_dt * dt, 0)
        # Cab_ls = jnp.maximum(Cab_ls + dCab_ls_dt * dt, 0)
        # Cab_hs = jnp.maximum(Cab_hs + dCab_hs_dt * dt, 0)
        # Cab_lf = jnp.maximum(Cab_lf + dCab_lf_dt * dt, 0)
        # Cab_hf = jnp.maximum(Cab_hf + dCab_hf_dt * dt, 0)

        return {
            f"Cas": Cas,
            f"{prefix}_Caf": Caf,
            f"{prefix}_Cab_ls": Cab_ls,
            f"{prefix}_Cab_hs": Cab_hs,
            f"{prefix}_Cab_lf": Cab_lf,
            f"{prefix}_Cab_hf": Cab_hf,
        }

    def compute_current(self, states, v, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(self, v, params):
        """Initialize the state at fixed point of gate dynamics."""
        return {
            f"Cas": 0.0966,  # Initial internal calcium concentration in mM
            f"{self._name}_Caf": 0.0966,
            f"{self._name}_Cab_ls": 80.929,
            f"{self._name}_Cab_hs": 29.068,
            f"{self._name}_Cab_lf": 80.929,
        }


class CaNernstReversal(Channel):
    """Compute Calcium reversal from inner and outer concentration of calcium."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.channel_params = {"Cao": 1600}  # μM
        self.channel_states = {
            "eCa": 40.0,  # mV
            "Cas": 0.0966,  # μM
        }
        self.current_name = f"iCa"

    def update_states(self, states, dt, v, params):
        """Update internal calcium concentration based on calcium current and decay."""
        Cao = params["Cao"]
        Cas = states["Cas"]
        eCa = -12.5 * jnp.log(Cas / Cao)
        return {"eCa": eCa, "Cas": Cas}

    def compute_current(self, states, v, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(self, v, params):
        """Initialize the state at fixed point of gate dynamics."""
        return {"Cas": 0.0966}


class KCa(Channel):
    """Calcium-dependent potassium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gKCa": 5e-3,  # S/cm^2
            f"{self._name}_Khalf": 0.3,  # mM, half-activation concentration
            # with an unfortunate name conflict with potassium K
            "eK": -74,  # mV
        }
        self.channel_states = {
            f"{self._name}_m": 0.642,  # Initial value for m gating variable
            f"{self._name}_n": 0.1,  # Initial value for n gating variable
            "Cas": 0.0966,  # Initial internal calcium concentration in μM
        }
        self.current_name = f"iKCa"
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        dt /= 1000  # convert to seconds
        m_new = solve_gate_exponential(m, dt, *self.m_gate(v))
        n_new = self.n_gate(states["Cas"], params[f"{prefix}_Khalf"])
        return {f"{prefix}_m": m_new, f"{prefix}_n": n_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        n = states[f"{prefix}_n"]
        k_cond = params[f"{prefix}_gKCa"] * m**2 * n * 1000
        return k_cond * (v - params["eK"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
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
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gClCa": 2e-3,  # S/cm^2
            f"{self._name}_Khalf": 0.37,  # uM, half-activation concentration
            f"{self._name}_eClCa": -20,  # mV
        }
        self.channel_states = {
            f"{self._name}_m": 0.1,  # Initial value for n gating variable
            "Cas": 0.0966,  # Initial internal calcium concentration in μM
        }
        self.current_name = f"iClCa"
        self.META = META

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self._name
        m_new = self.m_gate(states["Cas"], params[f"{prefix}_Khalf"])
        return {f"{prefix}_m": m_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        k_cond = params[f"{prefix}_gClCa"] * m * 1000
        return k_cond * (v - params[f"{prefix}_eClCa"])

    def init_state(self, v, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        Khalf = params[f"{prefix}_Khalf"]
        m = self.m_gate(v, Khalf)
        return {f"{prefix}_m": m, "Cas": 0.0966}

    @staticmethod
    def m_gate(Cas, Khalf):
        """Calcium-dependent n gating variable."""
        return 1 / (1 + save_exp((Khalf - Cas) / 0.09))
