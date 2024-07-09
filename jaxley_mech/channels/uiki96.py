from typing import Dict, Optional, Union

import jax.debug
import jax.numpy as jnp
from jax.lax import select
from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler, save_exp, solve_gate_exponential

META = {
    "cell_type": "Bipolar cell",
    "species": "Goldfish; White Bass; Axolotl; Tiger Salamander; Dogfish",
    "reference": [
        "Usui, S., Ishihaiza, A., Kamiyama, Y., & Ishii, H. (1996). Ionic current model of bipolar cells in the lower vertebrate retina. Vision Research, 36(24), 4069–4076. https://doi.org/10.1016/S0042-6989(96)00179-4",
        "Kamiyama, Y., Ishihara, A., Aoyama, T., & Usui, S. (2005). Simulation Analyses of Retinal Cell Responses. In Modeling in the neurosciences.",
    ],
    "notes": "There were various errors in equations in the original paper. All errors are corrected by the 2005 Book chapter from the same authors.",
}


class Leak(Channel):
    """Leakage current"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gLeak": 0.23e-3,  # S/cm^2
            f"{prefix}_eLeak": -21.0,  # mV
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
            "eK": -58,  # mV
        }
        self.channel_states = {
            f"{self._name}_m": 0.0345,  # Initial value for n gating variable
            f"{self._name}_h": 0.8594,  # Initial value for n gating variable
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
        alpha = 400 / (save_exp(-(v - 15) / 36) + 1)
        beta = save_exp(-v / 13)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        v += 1e-6
        alpha = 0.0003 * save_exp(-v / 7)
        beta = 0.02 + 80 / (
            save_exp((v + 115) / 15) + 1
        )  # paper: -(v+115), but should be +(v+115)
        # fix from Kamiyama et al. 1997
        return alpha, beta


class KA(Channel):
    """Delayed Rectifying Potassium Channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gKA": 35e-3,  # S/cm^2
            "eK": -58,  # mV
        }
        self.channel_states = {
            f"{self._name}_m": 0.43,  # Initial value for n gating variable
            f"{self._name}_h": 0.999,  # Initial value for n gating variable
        }
        self.current_name = f"iKA"
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
        k_cond = params[f"{prefix}_gKA"] * m**3 * h * 1000
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
        alpha = 1_200 / (save_exp(-(v - 50) / 28) + 1)
        beta = 6 * save_exp(-v / 10)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        v += 1e-6
        alpha = 0.045 * save_exp(-v / 13)
        beta = 75 / ((save_exp(-v + 50) / 15) + 1)
        return alpha, beta


class Hyper(Channel):
    """Hyperpolarization-activated channel in the formulation of Markov model with 5 states"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gHyper": 0.975e-3,  # S/cm^2
            f"{prefix}_eHyper": -17.7,  # mV
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
        alpha = 3 / (save_exp((v + 110) / 15) + 1)
        beta = 1.5 / (save_exp(-(v + 115) / 15) + 1)
        return alpha, beta

    def init_state(self, v, params):
        return self.channel_states


class Ca(Channel):
    """L-type calcium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gCa": 1.1e-3,  # S/cm^2
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
        alpha = 12_000 * (120 - v) / (save_exp(-(v - 120) / 25) - 1)
        beta = 40_000 / (save_exp((v + 68) / 25) + 1)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable."""
        v += 1e-6
        h = save_exp(-(v - 50) / 11) / (save_exp(-(v - 50) / 11) + 1)
        return h


# class CaPump(Channel):
#     def __init__(
#         self,
#         name: Optional[str] = None,
#     ):
#         super().__init__(name)
#         name = self._name
#         self.channel_params = {
#             f"{name}_F": 9.648e4,  # Faraday's constant in C/mol
#             f"{name}_Vs": 1.692e-13,  # Compartment volume 1 (volume of the submembrance area) in dm^3
#             f"{name}_Vd": 7.356e-13,  # Compartment volume 2 (volume of the deep intracellular area) in dm^3
#             f"{name}_D_Ca": 6e-8,  # Ca Diffusion coefficient in dm^2/s
#             f"{name}_d_sd": 5.8e-5,  # Membrane thickness in dm (distance between submembrance area and the deep intracellular area)
#             f"{name}_S_sd": 4e-8,  # Surface area in dm^2 (surface area of the submembrance and the deep intracellular area shpreical boundary)
#             f"{name}_alpha_bl": 0.4,  # Binding rate constant 1 in s^-1μM^-1 (on rate constant to low-affinity buffer)
#             f"{name}_beta_bl": 0.2,  # Binding rate constant 2 in s^-1 (off rate constant to low-affinity buffer)
#             f"{name}_alpha_bh": 100,  # Unbinding rate constant 1 in s^-1μM^-1 (on rate constant to high-affinity buffer)
#             f"{name}_beta_bh": 90,  # Unbinding rate constant 2 in s^-1 (off rate constant to high-affinity buffer)
#             f"{name}_Cab_l_max": 400,  # total low-affinity buffer concentration in μM
#             f"{name}_Cab_h_max": 300,  # total high-affinity buffer concentration in μM
#             f"{name}_Jex": 150.0,  # External current (maximum Na-Ca exchanger current) in pA
#             f"{name}_Jex2": 150.0,  # External current (maximum Ca-ATPase exchanger current) in pA
#             f"{name}_Kex": 2.3,  # External calcium concentration factor in μM
#             f"{name}_Kex2": 0.5,  # External calcium concentration factor 2 in μM
#             f"{name}_Cae": 0.05,  # External calcium concentration in μM
#         }
#         self.channel_states = {
#             f"Cai": 0.0966,  # Initial internal calcium concentration in μM
#             f"{name}_Cad": 0.0966,  # Free intracellular calcium concentration in μM
#             f"{name}_Cab_ls": 80.929,  # Bound buffer f concentration in μM
#             f"{name}_Cab_hs": 29.068,  # Bound buffer f concentration in μM
#             f"{name}_Cab_ld": 80.929,  # Bound buffer h concentration in μM
#             f"{name}_Cab_hd": 29.068,  # Bound buffer h concentration in μM
#         }
#         self.current_name = f"iCa"
#         self.META = META

#     def update_states(self, states, dt, v, params):
#         """Update the states based on differential equations."""
#         prefix = self._name
#         v += 1e-6  # jitter to avoid division by zero
#         dt /= 1000  # convert to seconds
#         F, Vs, Vd, D_Ca, d_sd, S_sd = (
#             params[f"{prefix}_F"],
#             params[f"{prefix}_Vs"],
#             params[f"{prefix}_Vd"],
#             params[f"{prefix}_D_Ca"],
#             params[f"{prefix}_d_sd"],
#             params[f"{prefix}_S_sd"],
#         )
#         alpha_bl, beta_bl, alpha_bh, beta_bh, Cab_l_max, Cab_h_max = (
#             params[f"{prefix}_alpha_bl"],
#             params[f"{prefix}_beta_bl"],
#             params[f"{prefix}_alpha_bh"],
#             params[f"{prefix}_beta_bh"],
#             params[f"{prefix}_Cab_l_max"],
#             params[f"{prefix}_Cab_h_max"],
#         )
#         Jex, Kex, Jex2, Kex2, Cae = (
#             params[f"{prefix}_Jex"],
#             params[f"{prefix}_Kex"],
#             params[f"{prefix}_Jex2"],
#             params[f"{prefix}_Kex2"],
#             params[f"{prefix}_Cae"],
#         )

#         Cai = states[f"Cai"]
#         Cad = states[f"{prefix}_Cad"]
#         Cab_ls = states[f"{prefix}_Cab_ls"]
#         Cab_hs = states[f"{prefix}_Cab_hs"]
#         Cab_ld = states[f"{prefix}_Cab_ld"]
#         Cab_hd = states[f"{prefix}_Cab_hd"]

#         iCa = states["iCa"]
#         iEx = Jex * (Cai - Cae) / (Cai - Cae + Kex) * save_exp(-(v + 14) / 70)
#         iEx2 = Jex2 * (Cai - Cae) / (Cai - Cae + Kex2)

#         # Free intracellular calcium concentration dynamics
#         dCai_dt = (
#             -10e-6 * (iCa + iEx + iEx2) / (2 * F * Vs)
#             - D_Ca * S_sd * (Cai - Cad) / (d_sd * Vs)
#             + beta_bl * Cab_ls
#             - alpha_bl * Cai * (Cab_l_max - Cab_ls)
#             + beta_bh * Cab_hs
#             - alpha_bh * Cai * (Cab_h_max - Cab_hs)
#         )

#         # Bound intracellular calcium concentration dynamics
#         dCad_dt = (
#             D_Ca * S_sd * (Cai - Cad) / (d_sd * Vd)
#             + beta_bl * Cab_ld
#             - alpha_bl * Cad * (Cab_l_max - Cab_ld)
#             + beta_bh * Cab_hd
#             - alpha_bh * Cad * (Cab_h_max - Cab_hd)
#         )

#         dCab_ls_dt = alpha_bl * Cai * (Cab_l_max - Cab_ls) - beta_bl * Cab_ls
#         dCab_hs_dt = alpha_bh * Cai * (Cab_h_max - Cab_hs) - beta_bh * Cab_hs
#         dCab_ld_dt = alpha_bl * Cad * (Cab_l_max - Cab_ld) - beta_bl * Cab_ld
#         dCab_hd_dt = alpha_bh * Cad * (Cab_h_max - Cab_hd) - beta_bh * Cab_hd

#         Cai = Cai + dCai_dt * dt
#         Cad = Cad + dCad_dt * dt
#         Cab_ls = Cab_ls + dCab_ls_dt * dt
#         Cab_hs = Cab_hs + dCab_hs_dt * dt
#         Cab_ld = Cab_ld + dCab_ld_dt * dt
#         Cab_hd = Cab_hd + dCab_hd_dt * dt

#         return {
#             f"Cai": Cai,
#             f"{prefix}_Cad": Cad,
#             f"{prefix}_Cab_ls": Cab_ls,
#             f"{prefix}_Cab_hs": Cab_hs,
#             f"{prefix}_Cab_ld": Cab_ld,
#             f"{prefix}_Cab_hd": Cab_hd,
#         }

#     def compute_current(self, states, v, params):
#         """This dynamics model does not directly contribute to the membrane current."""
#         return 0

#     def init_state(self, v, params):
#         """Initialize the state at fixed point of gate dynamics."""
#         return {
#             f"Cai": 0.0966,  # Initial internal calcium concentration in μM
#             f"{self._name}_Cad": 0.0966,
#             f"{self._name}_Cab_ls": 80.929,
#             f"{self._name}_Cab_hs": 29.068,
#             f"{self._name}_Cab_lf": 80.929,
#         }


class CaNernstReversal(Channel):
    """Compute Calcium reversal from inner and outer concentration of calcium."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.channel_params = {"Cao": 2.5}  # mM
        self.channel_states = {
            "eCa": 40.0,  # mV
            "Cai": 0.0001,  # mM
        }
        self.current_name = f"iCa"

    def update_states(self, states, dt, v, params):
        """Update internal calcium concentration based on calcium current and decay."""
        Cao = params["Cao"]
        Cai = states["Cai"]
        eCa = 12.9 * jnp.log(Cao / Cai)
        return {"eCa": eCa, "Cai": Cai}

    def compute_current(self, states, v, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(self, v, params):
        """Initialize the state at fixed point of gate dynamics."""
        return {"Cai": 0.0001}


class KCa(Channel):
    """Calcium-dependent potassium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gKCa": 8.5e-3,  # S/cm^2
            f"{self._name}_Khalf": 0.2,  # mM, half-activation concentration
            # with an unfortunate name conflict with potassium K
            "eK": -58,  # mV
        }
        self.channel_states = {
            f"{self._name}_m": 0.642,  # Initial value for m gating variable
            f"{self._name}_n": 0.1,  # Initial value for n gating variable
            "Cai": 0.0106,  # Initial internal calcium concentration in μM
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
        n_new = self.n_gate(states["Cai"], params[f"{prefix}_Khalf"])
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
            "Cai": 0.0001,
        }

    @staticmethod
    def m_gate(v):
        v += 1e-6
        alpha = 100 * (230 - v) / (save_exp((230 - v) / 52) - 1)
        beta = 120 * save_exp(-v / 95)
        return alpha, beta

    @staticmethod
    def n_gate(Cai, Khalf):
        """Calcium-dependent n gating variable."""
        return Cai / (Cai + Khalf)
