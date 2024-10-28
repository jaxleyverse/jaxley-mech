from typing import Dict, Optional

import jax.numpy as jnp
from jax.lax import select
from jaxley.channels import Channel
from jaxley.solver_gate import (
    exponential_euler,
    save_exp,
    solve_gate_exponential,
    solve_inf_gate_exponential,
)

from ..utils import efun

__all__ = [
    "NaTaT",
    "NaTs2T",
    "NapEt2",
    "KPst",
    "KTst",
    "SKE2",
    "SKv3_1",
    "M",
    "CaHVA",
    "CaLVA",
    "CaPump",
    "CaNernstReversal",
    "H",
]

############################
## Sodium channels:       ##
## NaTaT, NaTs2T, NapEt2  ##
############################


class NaTaT(Channel):
    """Transient sodium current from Colbert and Pan, 2002."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNaTaT": 0.00001,  # S/cm^2
            "eNa": 50.0,  # mV
        }
        self.channel_states = {
            f"{prefix}_m": 0.1,  # Initial value for m gating variable
            f"{prefix}_h": 0.1,  # Initial value for h gating variable
        }
        self.current_name = f"i_Na"
        self.META = {
            "reference": "Colbert and Pan (2002)",
            "doi": "https://doi.org/10.1038/nn0602-857",
            "species": "rat",
            "cell_type": "layer 5 pyramidal cell",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/NaTa_t.mod",
            "ion": "Na",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages))
        h_new = solve_inf_gate_exponential(hs, dt, *self.h_gate(voltages))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        na_cond = params[f"{prefix}_gNaTaT"]
        current = na_cond * (ms**3) * hs * (voltages - params["eNa"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages)
        h_inf, _ = self.h_gate(voltages)
        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_h": h_inf,
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        qt = 2.3 ** ((34 - 21) / 10)
        alpha = (0.182 * (v + 38 + 1e-6)) / (1 - save_exp(-(v + 38 + 1e-6) / 6))
        beta = (0.124 * (-v - 38 + 1e-6)) / (1 - save_exp(-(-v - 38 + 1e-6) / 6))
        m_inf = alpha / (alpha + beta)
        tau_m = 1 / (alpha + beta) / qt
        return m_inf, tau_m

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable."""
        qt = 2.3 ** ((34 - 21) / 10)
        alpha = (-0.015 * (v + 66 + 1e-6)) / (1 - save_exp((v + 66 + 1e-6) / 6))
        beta = (-0.015 * (-v - 66 + 1e-6)) / (1 - save_exp((-v - 66 + 1e-6) / 6))
        h_inf = alpha / (alpha + beta)
        tau_h = 1 / (alpha + beta) / qt
        return h_inf, tau_h


class NaTs2T(Channel):
    """Transient sodium current from Colbert and Pan, 2002.
    Almost exactly the same as NaTaT, but with different voltage-dependent dynamics.
    """

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNaTs2T": 0.00001,  # S/cm^2
            "eNa": 50.0,  # mV
        }
        self.channel_states = {
            f"{prefix}_m": 0.1,  # Initial value for m gating variable
            f"{prefix}_h": 0.1,  # Initial value for h gating variable
        }
        self.current_name = f"i_Na"
        self.META = {
            "reference": "Colbert and Pan, 2002",
            "species": "rat",
            "cell_type": "layer 5 pyramidal cell",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/NaTs2_t.mod",
            "note": "Took the NaTa and shifted both activation/inactivation by 6 mv",
            "ion": "Na",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages))
        h_new = solve_inf_gate_exponential(hs, dt, *self.h_gate(voltages))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        na_cond = params[f"{prefix}_gNaTs2T"]
        current = na_cond * (ms**3) * hs * (voltages - params["eNa"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages)
        h_inf, _ = self.h_gate(voltages)
        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_h": h_inf,
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        qt = 2.3 ** ((34 - 21) / 10)
        alpha = (0.182 * (v + 32 + 1e-6)) / (1 - save_exp(-(v + 32 + 1e-6) / 6))
        beta = (0.124 * (-v - 32 + 1e-6)) / (1 - save_exp(-(-v - 32 + 1e-6) / 6))
        m_inf = alpha / (alpha + beta)
        tau_m = 1 / (alpha + beta) / qt
        return m_inf, tau_m

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable."""
        qt = 2.3 ** ((34 - 21) / 10)
        alpha = (-0.015 * (v + 60 + 1e-6)) / (1 - save_exp((v + 60 + 1e-6) / 6))
        beta = (-0.015 * (-v - 60 + 1e-6)) / (1 - save_exp((-v - 60 + 1e-6) / 6))
        h_inf = alpha / (alpha + beta)
        tau_h = 1 / (alpha + beta) / qt
        return h_inf, tau_h


class NapEt2(Channel):
    """Persistent sodium current from Magistretti & Alonso 1999.

    Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21.
    """

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNapEt2": 0.00001,  # S/cm^2
            "eNa": 50,  # mV
        }
        self.channel_states = {
            f"{prefix}_m": 0.1,  # Initial value for m gating variable
            f"{prefix}_h": 0.1,  # Initial value for h gating variable
        }
        self.current_name = f"i_Na"
        self.META = {
            "reference": "Magistretti and Alonso (1999)",
            "doi": "https://doi.org/10.1085/jgp.114.4.491",
            "species": "rat",
            "cell_type": "entorhinal cortex layer-II principal neurons",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/Nap_Et2.mod",
            "ion": "Na",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages))
        h_new = solve_inf_gate_exponential(hs, dt, *self.h_gate(voltages))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        na_cond = params[f"{prefix}_gNapEt2"]
        current = na_cond * (ms**3) * hs * (voltages - params["eNa"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages)
        h_inf, _ = self.h_gate(voltages)
        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_h": h_inf,
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        qt = 2.3 ** ((34 - 21) / 10)  # Q10 temperature correction
        alpha = (0.182 * (v + 38 + 1e-6)) / (1 - save_exp(-(v + 38 + 1e-6) / 6))
        beta = (0.124 * (-v - 38 + 1e-6)) / (1 - save_exp(-(-v - 38 + 1e-6) / 6))
        tau_m = 6 / (alpha + beta) / qt
        m_inf = 1.0 / (1 + save_exp((v + 52.7) / -4.6))
        return m_inf, tau_m

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable."""
        qt = 2.3 ** ((34 - 21) / 10)  # Q10 temperature correction
        alpha = (-2.88e-6 * (v + 17 + 1e-6)) / (1 - save_exp((v + 17 + 1e-6) / 4.63))
        beta = (6.94e-6 * (v + 64.4 + 1e-6)) / (1 - save_exp((-(v + 64.4) + 1e-6) / 6))
        tau_h = 1 / (alpha + beta) / qt
        h_inf = 1.0 / (1 + save_exp((v + 48.8) / 10))

        return h_inf, tau_h


###########################
## Potassium channels    ##
## KPst, KTst            ##
###########################


class KPst(Channel):
    """Persistent component of the K current from Korngreen and Sakmann, 2000, adjusted for junction potential."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gKPst": 0.00001,  # S/cm^2
            "eK": -77.0,  # mV, from l5pc/config/parameters.json
        }
        self.channel_states = {
            f"{prefix}_m": 0.1,  # Initial value for m gating variable
            f"{prefix}_h": 0.1,  # Initial value for h gating variable
        }
        self.current_name = f"i_K"
        self.META = {
            "reference": "Korngreen and Sakmann (2000)",
            "doi": "https://doi.org/10.1111/j.1469-7793.2000.00621.x",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/K_Pst.mod",
            "species": "rat",
            "cell_type": "layer 5 pyramidal cell",
            "note": "Shifted -10 mV to correct for junction potential, rates corrected with Q10",
            "ion": "K",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables using cnexp integration method."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages))
        h_new = solve_inf_gate_exponential(hs, dt, *self.h_gate(voltages))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the potassium current through the channel."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        k_cond = params[f"{prefix}_gKPst"] * (ms**2) * hs
        current = k_cond * (voltages - params["eK"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages)
        h_inf, _ = self.h_gate(voltages)
        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_h": h_inf,
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable, adjusted for junction potential."""
        qt = 2.3 ** ((34 - 21) / 10)  # Q10 temperature correction
        v_adjusted = v + 10  # Adjust for junction potential
        m_inf = 1 / (1 + save_exp(-(v_adjusted + 1) / 12))

        # See here for documentation of `select` vs `cond`:
        # https://github.com/google/jax/issues/7934
        tau_m = select(
            v_adjusted < jnp.asarray(-50.0),
            (1.25 + 175.03 * save_exp(v_adjusted * 0.026)) / qt,
            (1.25 + 13 * save_exp(-v_adjusted * 0.026)) / qt,
        )
        return m_inf, tau_m

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable, adjusted for junction potential."""
        qt = 2.3 ** ((34 - 21) / 10)  # Q10 temperature correction
        v_adjusted = v + 10  # Adjust for junction potential
        h_inf = 1 / (1 + save_exp(-(v_adjusted + 54) / -11))
        tau_h = (
            360
            + (1010 + 24 * (v_adjusted + 55))
            * save_exp(-(((v_adjusted + 75) / 48) ** 2))
        ) / qt
        return h_inf, tau_h


class KTst(Channel):
    """Transient component of the K current from Korngreen and Sakmann, 2000, adjusted for junction potential."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gKTst": 0.00001,  # S/cm^2
            "eK": -77.0,  # mV
        }
        self.channel_states = {
            f"{prefix}_m": 0.1,  # Initial value for m gating variable
            f"{prefix}_h": 0.1,  # Initial value for h gating variable
        }
        self.current_name = f"i_K"
        self.META = {
            "reference": "Korngreen and Sakmann (2000)",
            "doi": "https://doi.org/10.1111/j.1469-7793.2000.00621.x",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/K_Tst.mod",
            "species": "rat",
            "cell_type": "layer 5 pyramidal cell",
            "note": "Shifted -10 mV to correct for junction potential, rates corrected with Q10",
            "ion": "K",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables using cnexp integration method."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages))
        h_new = solve_inf_gate_exponential(hs, dt, *self.h_gate(voltages))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the potassium current through the channel."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        k_cond = params[f"{prefix}_gKTst"] * (ms**4) * hs
        current = k_cond * (voltages - params["eK"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages)
        h_inf, _ = self.h_gate(voltages)
        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_h": h_inf,
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable, adjusted for junction potential."""
        qt = 2.3 ** ((34 - 21) / 10)  # Q10 temperature correction
        v_adjusted = v + 10  # Adjust for junction potential
        m_inf = 1 / (1 + save_exp(-(v_adjusted + 0) / 19))
        tau_m = (0.34 + 0.92 * save_exp(-(((v_adjusted + 71) / 59) ** 2))) / qt
        return m_inf, tau_m

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable, adjusted for junction potential."""
        qt = 2.3 ** ((34 - 21) / 10)  # Q10 temperature correction
        v_adjusted = v + 10  # Adjust for junction potential
        h_inf = 1 / (1 + save_exp(-(v_adjusted + 66) / -10))
        tau_h = (8 + 49 * save_exp(-(((v_adjusted + 73) / 23) ** 2))) / qt
        return h_inf, tau_h


class SKE2(Channel):
    """SK-type calcium-activated potassium current from Kohler et al., 1996."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gSKE2": 0.000001,  # mho/cm^2
            "eK": -77.0,  # mV, assuming eK for potassium
        }
        self.channel_states = {
            f"{prefix}_z": 0.0,  # Initial value for z gating variable
            f"CaCon_i": 5e-05,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"i_K"
        self.META = {
            "reference": "Kohler et al., 1996",
            "doi": "https://doi.org/10.1126/science.273.5282.1709",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/SK_E2.mod",
            "species": ["rat", "human"],
            "cell_type": "unknown",
            "ion": "K",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variable z."""
        prefix = self._name
        zs = u[f"{prefix}_z"]
        cai = u["CaCon_i"]  # intracellular calcium concentration, from CaPump
        z_new = solve_inf_gate_exponential(zs, dt, *self.z_gate(cai))
        return {f"{prefix}_z": z_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the potassium current through the channel."""
        prefix = self._name
        z = u[f"{prefix}_z"]
        k_cond = params[f"{prefix}_gSKE2"] * z
        current = k_cond * (voltages - params["eK"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        cai = 5e-05  # Initial value for intracellular calcium concentration
        z_inf, _ = self.z_gate(cai)
        return {f"{prefix}_z": z_inf}

    @staticmethod
    def z_gate(cai):
        """Dynamics for the z gating variable, dependent on intracellular calcium concentration."""
        cai = select(
            cai < jnp.asarray(1e-7),
            cai + 1e-7,
            cai,
        )
        z_inf = 1 / (1 + (0.00043 / cai) ** 4.8)
        tau_z = 1.0  # tau_z is fixed at 1 ms
        return z_inf, tau_z


class SKv3_1(Channel):
    """Shaw-related potassium channel family SKv3_1 from The EMBO Journal, 1992."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gSKv3_1": 0.00001,  # S/cm^2
            "eK": -77.0,  # mV, assuming eK for potassium
        }
        self.channel_states = {
            f"{prefix}_m": 0.1,  # Initial value for m gating variable
        }
        self.current_name = f"i_K"
        self.META = {
            "reference": "Rettig, et al. (1992)",
            "doi": "https://doi.org/10.1002/j.1460-2075.1992.tb05312.x",
            "species": "rat",
            "cell_type": "unknown",
            "ion": "K",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variable m."""
        prefix = self._name
        ms = u[f"{prefix}_m"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages))
        return {f"{prefix}_m": m_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the potassium current through the channel."""
        prefix = self._name
        m = u[f"{prefix}_m"]
        k_cond = params[f"{prefix}_gSKv3_1"] * m
        current = k_cond * (voltages - params["eK"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages)
        return {f"{prefix}_m": m_inf}

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        m_inf = 1 / (1 + save_exp((v - 18.7) / -9.7))
        tau_m = 0.2 * 20.0 / (1 + save_exp((v + 46.56) / -44.14))
        return m_inf, tau_m


class M(Channel):
    """M-currents and other potassium currents in bullfrog sympathetic neurones from Adams et al., 1982, with temperature corrections."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gM": 0.00001,  # S/cm^2
            "eK": -77.0,  # mV, assuming eK for potassium
        }
        self.channel_states = {
            f"{prefix}_m": 0.0,  # Initial value for m gating variable
        }
        self.current_name = f"i_K"
        self.META = {
            "reference": "Adams et al. (1982)",
            "doi": "https://doi.org/10.1113/jphysiol.1982.sp014357",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/Im.mod",
            "species": "bullfrog",
            "cell_type": "lumbar sympathetic neurones",
            "note": "Corrected rates using Q10 = 2.3, target temperature 34, original 21",
            "ion": "K",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variable m."""
        prefix = self._name
        ms = u[f"{prefix}_m"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages))
        return {f"{prefix}_m": m_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the potassium current through the channel."""
        prefix = self._name
        m = u[f"{prefix}_m"]
        k_cond = params[f"{prefix}_gM"] * m
        current = k_cond * (voltages - params["eK"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages)
        return {f"{prefix}_m": m_inf}

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable, with temperature correction."""
        qt = 2.3 ** ((34 - 21) / 10)  # Q10 temperature correction
        m_alpha = 3.3e-3 * save_exp(2.5 * 0.04 * (v + 35))
        m_beta = 3.3e-3 * save_exp(-2.5 * 0.04 * (v + 35))
        m_inf = m_alpha / (m_alpha + m_beta)
        tau_m = (1 / (m_alpha + m_beta)) / qt
        return m_inf, tau_m


############################
## Calcium channels:      ##
## CaHVA, CaLVA           ##
############################


class CaHVA(Channel):
    """High-Voltage-Activated (HVA) Ca2+ channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gCaHVA": 0.00001,  # S/cm^2
        }
        self.channel_states = {
            f"{self._name}_m": 0.1,  # Initial value for m gating variable
            f"{self._name}_h": 0.1,  # Initial value for h gating variable
            "eCa": 0.0,  # mV, assuming eca for demonstration
        }
        self.current_name = f"i_Ca"
        self.META = {
            "reference": "Reuveni, et al. (1993)",
            "doi": "https://doi.org/10.1523/jneurosci.13-11-04609.1993",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/SK_E2.mod",
            "species": "rat",
            "cell_type": "layer 5 pyramidal cell",
            "ion": "Ca",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        m_new = solve_gate_exponential(ms, dt, *self.m_gate(voltages))
        h_new = solve_gate_exponential(hs, dt, *self.h_gate(voltages))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new, "eCa": u["eCa"]}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        ca_cond = params[f"{prefix}_gCaHVA"] * (ms**2) * hs
        current = ca_cond * (voltages - u["eCa"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = self.m_gate(voltages)
        alpha_h, beta_h = self.h_gate(voltages)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        alpha = (0.055 * (-27 - v + 1e-6)) / (save_exp((-27.0 - v + 1e-6) / 3.8) - 1.0)
        beta = 0.94 * save_exp((-75.0 - v + 1e-6) / 17.0)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable."""
        alpha = 0.000457 * save_exp((-13.0 - v) / 50.0)
        beta = 0.0065 / (save_exp((-v - 15.0) / 28.0) + 1.0)
        return alpha, beta


class CaLVA(Channel):
    """Low-Voltage-Activated (LVA) Ca2+ channel, based on Avery and Johnston 1996 and Randall 1997"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gCaLVA": 0.00001,  # S/cm^2
        }
        self.channel_states = {
            f"{self._name}_m": 0.0,  # Initial value for m gating variable
            f"{self._name}_h": 0.0,  # Initial value for h gating variable
            "eCa": 0.0,  # mV, assuming eCa for demonstration
        }
        self.current_name = f"i_Ca"
        self.META = {
            "reference": "Avery and Johnston (1996)",
            "doi": "https://doi.org/10.1523/jneurosci.16-18-05567.1996",
            "species": "rat",
            "cell_type": "hippocampal CA3 pyramidal cell",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/Ca_LVAst.mod",
            "ion": "Ca",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages))
        h_new = solve_inf_gate_exponential(hs, dt, *self.h_gate(voltages))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new, "eCa": u["eCa"]}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        ca_cond = params[f"{prefix}_gCaLVA"] * (ms**2) * hs
        current = ca_cond * (voltages - u["eCa"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages)
        h_inf, _ = self.h_gate(voltages)
        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_h": h_inf,
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable, adjusted for junction potential."""
        qt = 2.3 ** ((34 - 21) / 10)  # Q10 temperature correction
        v_shifted = v + 10  # Shift by 10 mV
        m_inf = 1.0 / (1 + save_exp((v_shifted + 30) / -6))
        tau_m = (5.0 + 20.0 / (1 + save_exp((v_shifted + 25) / 5))) / qt
        return m_inf, tau_m

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable, adjusted for junction potential."""
        qt = 2.3 ** ((34 - 21) / 10)  # Q10 temperature correction
        v_shifted = v + 10  # Shift by 10 mV
        h_inf = 1.0 / (1 + save_exp((v_shifted + 80) / 6.4))
        tau_h = (20.0 + 50.0 / (1 + save_exp((v_shifted + 40) / 7))) / qt
        return h_inf, tau_h


class CaPump(Channel):
    """Calcium dynamics tracking inside calcium concentration, modeled after Destexhe et al. 1994."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gamma": 0.05,  # Fraction of free calcium (not buffered)
            f"{self._name}_decay": 80,  # Rate of removal of calcium in ms
            f"{self._name}_depth": 0.1,  # Depth of shell in um
            f"{self._name}_minCai": 1e-4,  # Minimum intracellular calcium concentration in mM
        }
        self.channel_states = {
            f"CaCon_i": 5e-05,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"i_Ca"
        self.META = {
            "reference": "Destexhe et al., (1994)",
            "doi": "https://doi.org/10.1152/jn.1994.72.2.803",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/CaDynamics_E2.mod",
            "species": "ferret",
            "cell_type": "thalamic reticular nucleus",
            "ion": "Ca",
        }

    def update_states(self, u, dt, voltages, params):
        """Update internal calcium concentration based on calcium current and decay."""
        prefix = self._name
        ica = u["i_Ca"] / 1_000.0
        cai = u["CaCon_i"]
        gamma = params[f"{prefix}_gamma"]
        decay = params[f"{prefix}_decay"]
        depth = params[f"{prefix}_depth"]
        minCai = params[f"{prefix}_minCai"]

        FARADAY = 96485  # Coulombs per mole

        # Calculate the contribution of calcium currents to cai change
        drive_channel = -10_000.0 * ica * gamma / (2 * FARADAY * depth)

        cai_tau = decay
        cai_inf = minCai + decay * drive_channel
        new_cai = exponential_euler(cai, dt, cai_inf, cai_tau)

        return {f"CaCon_i": new_cai}

    def compute_current(self, u, voltages, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        return {}


class CaNernstReversal(Channel):
    """Compute Calcium reversal from inner and outer concentration of calcium."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_constants = {
            "F": 96485.3329,  # C/mol (Faraday's constant)
            "T": 279.45,  # Kelvin (temperature)
            "R": 8.314,  # J/(mol K) (gas constant)
        }
        self.channel_params = {}
        self.channel_states = {"eCa": 0.0, "CaCon_i": 5e-05, "CaCon_e": 2.0}
        self.current_name = f"i_Ca"
        self.META = {"ion": "Ca"}

    def update_states(self, u, dt, voltages, params):
        """Update internal calcium concentration based on calcium current and decay."""
        R, T, F = (
            self.channel_constants["R"],
            self.channel_constants["T"],
            self.channel_constants["F"],
        )
        Cai = u["CaCon_i"]
        Cao = u["CaCon_e"]
        C = R * T / (2 * F) * 1000  # mV
        vCa = C * jnp.log(Cao / Cai)
        return {"eCa": vCa, "CaCon_i": Cai, "CaCon_e": Cao}

    def compute_current(self, u, voltages, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        return {}


#################################
## hyperpolarization-activated ##
## cation channel              ##
#################################


class H(Channel):
    """H-current (H) from Kole, Hallermann, and Stuart, J. Neurosci., 2006."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gH": 0.00001,  # S/cm^2
            "eH": -45.0,  # mV, reversal potential for H
        }
        self.channel_states = {
            f"{prefix}_m": 0.0,  # Initial value for m gating variable
        }
        self.current_name = f"i_H"
        self.META = {
            "reference": "Kole, et al. (2006)",
            "doi": "https://doi.org/10.1523/JNEUROSCI.3664-05.2006",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/Ih.mod",
            "cell_type": "layer 5 pyramidal cell",
            "species": "rat",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variable m."""
        prefix = self._name
        ms = u[f"{prefix}_m"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages))
        return {f"{prefix}_m": m_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the nonspecific current through the channel."""
        prefix = self._name
        m = u[f"{prefix}_m"]
        h_cond = params[f"{prefix}_gH"] * m
        current = h_cond * (voltages - params["eH"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages)
        return {f"{prefix}_m": m_inf}

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        m_alpha = (
            0.001
            * 6.43
            * (v + 154.9 + 1e-6)
            / (save_exp((v + 154.9 + 1e-6) / 11.9) - 1)
        )
        m_beta = 0.001 * 193 * save_exp(v / 33.1)
        m_inf = m_alpha / (m_alpha + m_beta)
        tau_m = 1 / (m_alpha + m_beta)
        return m_inf, tau_m
        return m_inf, tau_m
