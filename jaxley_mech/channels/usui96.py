from typing import Dict, Optional, Union

import jax.numpy as jnp
from jax import Array
from jax.lax import select
from jaxley.channels import Channel
from jaxley.solver_gate import (exponential_euler, save_exp,
                                solve_gate_exponential)

from jaxley_mech.solvers import SolverExtension

META = {
    "cell_type": "bipolar cell",
    "species": ["goldfish", "white bass", "axolotl", "tiger salamander", "dogfish"],
    "reference": "Usui, et al. (1996)",
    "doi": "https://doi.org/10.1016/S0042-6989(96)00179-4",
    "note": "There were various errors in equations in the original paper. All errors are corrected by the 2005 Book chapter from the same authors.",
}


class Leak(Channel):
    """Leakage current"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
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
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """No state to update."""
        return {}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Given channel states and voltage, return the current through the channel."""
        prefix = self._name
        gLeak = params[f"{prefix}_gLeak"]  # S/cm^2
        return gLeak * (voltage - params[f"{prefix}_eLeak"])  # S/cm^2 * mV = mA/cm^2

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the state such at fixed point of gate dynamics."""
        return {}


class Kv(Channel):
    """Delayed Rectifying Potassium Channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gKv": 2e-3,  # S/cm^2
            "eK": -58,  # mV
        }
        self.channel_states = {
            f"{self._name}_m": 0.824374,  # Initial value for n gating variable
            f"{self._name}_h": 0.109794,  # Initial value for n gating variable
        }
        self.current_name = f"iKv"
        self.META = META
        self.META.update({"ion": "K"})

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Update state of gating variables."""
        prefix = self._name
        dt /= 1000
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        m_new = solve_gate_exponential(m, dt, *self.m_gate(voltage))
        h_new = solve_gate_exponential(h, dt, *self.h_gate(voltage))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        h = states[f"{prefix}_h"]
        k_cond = params[f"{prefix}_gKv"] * m**3 * h
        return k_cond * (voltage - params["eK"])

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = self.m_gate(voltage)
        alpha_h, beta_h = self.h_gate(voltage)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    @staticmethod
    def m_gate(voltage):
        """Voltage-dependent dynamics for the n gating variable."""
        v += 1e-6
        alpha = 400 / (save_exp(-(v - 15) / 36) + 1)
        beta = save_exp(-v / 13)
        return alpha, beta

    @staticmethod
    def h_gate(voltage):
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
        self.current_is_in_mA_per_cm2 = True
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
        self.META.update({"ion": "K"})

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Update state of gating variables."""
        prefix = self._name
        dt /= 1000
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        m_new = solve_gate_exponential(m, dt, *self.m_gate(voltage))
        h_new = solve_gate_exponential(h, dt, *self.h_gate(voltage))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        h = states[f"{prefix}_h"]
        k_cond = params[f"{prefix}_gKA"] * m**3 * h
        return k_cond * (voltage - params["eK"])

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = self.m_gate(voltage)
        alpha_h, beta_h = self.h_gate(voltage)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    @staticmethod
    def m_gate(voltage):
        """Voltage-dependent dynamics for the n gating variable."""
        v += 1e-6
        # alpha = 1_200 / (save_exp(-(v - 50) / 28) + 1)
        # beta = 6 * save_exp(-v / 10)
        alpha = 2_400 / (save_exp(-(v - 50) / 28) + 1)
        beta = 12 * save_exp(-v / 10)
        return alpha, beta

    @staticmethod
    def h_gate(voltage):
        v += 1e-6
        alpha = 0.045 * save_exp(-v / 13)
        # beta = 75 / ((save_exp(-v + 50) / 15) + 1)
        beta = 75 / (save_exp(-(v + 30) / 15) + 1)
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

        prefix = self._name
        self.channel_params = {
            f"{prefix}_gHyper": 0.975e-3,  # S/cm^2
            f"{prefix}_eHyper": -17.7,  # mV
        }
        self.channel_states = {
            f"{prefix}_C1": 0.92823,
            f"{prefix}_C2": 0.05490,
            f"{prefix}_O1": 0.00122,
            f"{prefix}_O2": 1.20061e-5,
            f"{prefix}_O3": 4.43854e-8,
        }
        self.current_name = f"iHyper"
        self.META = {
            "reference": [],
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
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
        **kwargs,
    ):
        """Update the states using the specified solver."""
        prefix = self._name
        dt /= 1000  # Convert dt to seconds

        # Retrieve states
        C1 = states[f"{prefix}_C1"]
        C2 = states[f"{prefix}_C2"]
        O1 = states[f"{prefix}_O1"]
        O2 = states[f"{prefix}_O2"]
        O3 = states[f"{prefix}_O3"]

        y0 = jnp.array([C1, C2, O1, O2, O3])

        # Parameters for dynamics
        args_tuple = (voltage,)

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
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Return current."""
        prefix = self._name
        O1 = states[f"{prefix}_O1"]
        O2 = states[f"{prefix}_O2"]
        O3 = states[f"{prefix}_O3"]
        gHyper = params[f"{prefix}_gHyper"] * (O1 + O2 + O3)
        return gHyper * (voltage - params[f"{prefix}_eHyper"])

    @staticmethod
    def h_gate(voltage):
        v += 1e-6
        alpha = 3 / (save_exp((v + 110) / 15) + 1)
        beta = 1.5 / (save_exp(-(v + 115) / 15) + 1)
        return alpha, beta

    def init_state(self, states, v, params, delta_t):
        return self.channel_states


class Ca(Channel):
    """L-type calcium channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gCa": 1.1e-3,  # S/cm^2
        }
        self.channel_states = {
            f"{self._name}_m": 0.290203,  # Initial value for m gating variable
            f"{self._name}_h": 0.5,  # Initial value for h gating variable
            "eCa": 40.0,  # mV, dependent on CaNernstReversal
        }
        self.current_name = f"iCa"
        self.META = META
        self.META.update({"ion": "Ca"})

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Update state of gating variables."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        dt /= 1000  # convert to seconds
        m_new = solve_gate_exponential(m, dt, *self.m_gate(voltage))
        h_new = self.h_gate(voltage)
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new, "eCa": states["eCa"]}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        ca_cond = params[f"{prefix}_gCa"] * m**4 * h
        current = ca_cond * (voltage - states["eCa"])
        return current

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name

        alpha_m, beta_m = self.m_gate(voltage)
        h = self.h_gate(voltage)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": h,
            "eCa": 40.0,
        }

    @staticmethod
    def m_gate(voltage):
        """Voltage-dependent dynamics for the m gating variable."""
        v += 1e-6
        alpha = 12_000 * (120 - v) / (save_exp((120 - v) / 25) - 1)
        beta = 40_000 / (save_exp((v + 68) / 25) + 1)
        return alpha, beta

    @staticmethod
    def h_gate(voltage):
        """Voltage-dependent dynamics for the h gating variable."""
        v += 1e-6
        h = save_exp((50 - v) / 11) / (save_exp((50 - v) / 11) + 1)
        return h


class CaPump(Channel, SolverExtension):
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

        name = self._name
        self.channel_params = {
            f"{name}_F": 9.648e4,  # Faraday's constant in C/mol
            f"{name}_Vs": 1.692e-13,  # Compartment volume 1 (volume of the submembrance area) in dm^3
            f"{name}_Vd": 7.356e-13,  # Compartment volume 2 (volume of the deep intracellular area) in dm^3
            f"{name}_D_Ca": 6e-8,  # Ca Diffusion coefficient in dm^2/s
            f"{name}_d_sd": 5.8e-5,  # Membrane thickness in dm (distance between submembrance area and the deep intracellular area)
            f"{name}_S_sd": 4e-8,  # Surface area in dm^2 (surface area of the submembrance and the deep intracellular area shpreical boundary)
            f"{name}_alpha_bl": 0.4,  # Binding rate constant 1 in s^-1μM^-1 (on rate constant to low-affinity buffer)
            f"{name}_beta_bl": 0.2,  # Binding rate constant 2 in s^-1 (off rate constant to low-affinity buffer)
            f"{name}_alpha_bh": 100,  # Unbinding rate constant 1 in s^-1μM^-1 (on rate constant to high-affinity buffer)
            f"{name}_beta_bh": 90,  # Unbinding rate constant 2 in s^-1 (off rate constant to high-affinity buffer)
            f"{name}_Cab_l_max": 400,  # total low-affinity buffer concentration in μM
            f"{name}_Cab_h_max": 300,  # total high-affinity buffer concentration in μM
            f"{name}_Jex": 360.0,  # External current (maximum Na-Ca exchanger current) in pA
            f"{name}_Jex2": 380.0,  # External current (maximum Ca-ATPase exchanger current) in pA
            f"{name}_Kex": 2.3,  # External calcium concentration factor in μM
            f"{name}_Kex2": 0.5,  # External calcium concentration factor 2 in μM
            f"{name}_Cae": 0.01,  # External calcium concentration in μM
        }
        self.channel_states = {
            f"Cas": 0.01156,  # Initial internal calcium concentration in μM
            f"{name}_Cad": 0.01156,  # Free intracellular calcium concentration in μM
            f"{name}_Cab_ls": 6.78037,  # Bound buffer f concentration in μM
            f"{name}_Cab_hs": 1.26836,  # Bound buffer f concentration in μM
            f"{name}_Cab_ld": 11.30257,  # Bound buffer h concentration in μM
            f"{name}_Cab_hd": 3.80563,  # Bound buffer h concentration in μM
        }
        self.current_name = f"iCa"
        self.META = META
        self.META.update({"ion": "Ca"})

    def derivatives(self, t, states, args):
        """Calculate the derivatives for the calcium pump system."""
        Cas, Cad, Cab_ls, Cab_hs, Cab_ld, Cab_hd = states
        (
            F,
            Vs,
            Vd,
            D_Ca,
            d_sd,
            S_sd,
            alpha_bl,
            beta_bl,
            alpha_bh,
            beta_bh,
            Cab_l_max,
            Cab_h_max,
            Jex,
            Kex,
            Jex2,
            Kex2,
            Cae,
            iCa,
            v,
        ) = args

        # Current terms
        iEx = Jex * (Cas - Cae) / (Cas - Cae + Kex) * save_exp(-(v + 14) / 70)
        iEx2 = Jex2 * (Cas - Cae) / (Cas - Cae + Kex2)

        # Free intracellular calcium concentration dynamics
        dCas_dt = (
            -1e-6 * (iCa + iEx + iEx2) / (2 * F * Vs)
            - D_Ca * S_sd * (Cas - Cad) / (d_sd * Vs)
            + beta_bl * Cab_ls
            - alpha_bl * Cas * (Cab_l_max - Cab_ls)
            + beta_bh * Cab_hs
            - alpha_bh * Cas * (Cab_h_max - Cab_hs)
        )

        # Deep intracellular calcium concentration dynamics
        dCad_dt = (
            D_Ca * S_sd * (Cas - Cad) / (d_sd * Vd)
            + beta_bl * Cab_ld
            - alpha_bl * Cad * (Cab_l_max - Cab_ld)
            + beta_bh * Cab_hd
            - alpha_bh * Cad * (Cab_h_max - Cab_hd)
        )

        # Bound intracellular calcium dynamics
        dCab_ls_dt = alpha_bl * Cas * (Cab_l_max - Cab_ls) - beta_bl * Cab_ls
        dCab_hs_dt = alpha_bh * Cas * (Cab_h_max - Cab_hs) - beta_bh * Cab_hs
        dCab_ld_dt = alpha_bl * Cad * (Cab_l_max - Cab_ld) - beta_bl * Cab_ld
        dCab_hd_dt = alpha_bh * Cad * (Cab_h_max - Cab_hd) - beta_bh * Cab_hd

        return jnp.array(
            [dCas_dt, dCad_dt, dCab_ls_dt, dCab_hs_dt, dCab_ld_dt, dCab_hd_dt]
        )

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Update state of calcium pump variables."""
        prefix = self._name
        dt /= 1000  # Convert to seconds

        # Retrieve states
        Cas = states[f"Cas"]
        Cad = states[f"{prefix}_Cad"]
        Cab_ls = states[f"{prefix}_Cab_ls"]
        Cab_hs = states[f"{prefix}_Cab_hs"]
        Cab_ld = states[f"{prefix}_Cab_ld"]
        Cab_hd = states[f"{prefix}_Cab_hd"]

        scale_factor = (2 * jnp.pi * params["length"] * params["radius"] * 1e-8) / 1e-9
        iCa = states["iCa"] * scale_factor  # mA/cm^2 to pA
        y0 = jnp.array([Cas, Cad, Cab_ls, Cab_hs, Cab_ld, Cab_hd])

        # Parameters for dynamics
        args_tuple = (
            params[f"{prefix}_F"],
            params[f"{prefix}_Vs"],
            params[f"{prefix}_Vd"],
            params[f"{prefix}_D_Ca"],
            params[f"{prefix}_d_sd"],
            params[f"{prefix}_S_sd"],
            params[f"{prefix}_alpha_bl"],
            params[f"{prefix}_beta_bl"],
            params[f"{prefix}_alpha_bh"],
            params[f"{prefix}_beta_bh"],
            params[f"{prefix}_Cab_l_max"],
            params[f"{prefix}_Cab_h_max"],
            params[f"{prefix}_Jex"],
            params[f"{prefix}_Kex"],
            params[f"{prefix}_Jex2"],
            params[f"{prefix}_Kex2"],
            params[f"{prefix}_Cae"],
            iCa,
            voltage,
        )

        y_new = self.solver_func(y0, dt, self.derivatives, args_tuple)

        # Unpack new states
        Cas_new, Cad_new, Cab_ls_new, Cab_hs_new, Cab_ld_new, Cab_hd_new = y_new

        return {
            f"Cas": Cas_new,
            f"{prefix}_Cad": Cad_new,
            f"{prefix}_Cab_ls": Cab_ls_new,
            f"{prefix}_Cab_hs": Cab_hs_new,
            f"{prefix}_Cab_ld": Cab_ld_new,
            f"{prefix}_Cab_hd": Cab_hd_new,
        }

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the state at fixed point of gate dynamics."""
        return self.channel_states


class CaNernstReversal(Channel):
    """Compute Calcium reversal from inner and outer concentration of calcium."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {"Cao": 2500}  # μM
        self.channel_states = {
            "eCa": 40.0,  # mV
            "Cas": 0.0001,  # mM
        }
        self.current_name = f"iCa"
        self.META = META
        self.META.update({"ion": "Ca"})

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Update internal calcium concentration based on calcium current and decay."""
        Cao = params["Cao"]
        Cas = states["Cas"]
        eCa = 12.9 * jnp.log(Cao / Cas)
        return {"eCa": eCa, "Cas": Cas}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the state at fixed point of gate dynamics."""
        return {"Cas": 0.0001}


class KCa(Channel):
    """Calcium-dependent potassium channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
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
            "Cas": 0.0106,  # Initial internal calcium concentration in μM
        }
        self.current_name = f"iKCa"
        self.META = META
        self.META.update({"ion": "K"})

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Update state of gating variables."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        dt /= 1000  # convert to seconds
        m_new = solve_gate_exponential(m, dt, *self.m_gate(voltage))
        n_new = self.n_gate(states["Cas"], params[f"{prefix}_Khalf"])
        return {f"{prefix}_m": m_new, f"{prefix}_n": n_new}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        n = states[f"{prefix}_n"]
        k_cond = params[f"{prefix}_gKCa"] * m**2 * n
        return k_cond * (voltage - params["eK"])

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        Khalf = params[f"{prefix}_Khalf"]
        alpha_m, beta_m = self.m_gate(voltage)
        n = self.n_gate(states["Cas"], Khalf)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_n": n,
            "Cas": 0.0001,
        }

    @staticmethod
    def m_gate(voltage):
        v += 1e-6
        alpha = 100 * (230 - v) / (save_exp((230 - v) / 52) - 1)
        beta = 120 * save_exp(-v / 95)
        return alpha, beta

    @staticmethod
    def n_gate(Cas, Khalf):
        """Calcium-dependent n gating variable."""
        return Cas / (Cas + Khalf)
