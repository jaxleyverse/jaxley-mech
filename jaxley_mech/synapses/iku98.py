# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional, Tuple

import jax.debug
import jax.numpy as jnp
from jaxley.channels.channel import Channel
from jaxley.solver_gate import save_exp, solve_gate_exponential
from jaxley.synapses.synapse import Synapse


class Glu(Synapse):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gGlu": 240e-3,  # S/cm^2
            f"{prefix}_kGlu": 68,  # μM
            # for the integrated Ca channel
            f"{prefix}_Cm": 90,
            f"{prefix}_kCa": 10,  # mV
            f"{prefix}_kiCa": 0.5,  # 1/μM
            f"{prefix}_nCa": 2,
            f"{prefix}_Cao": 2500,  # μM
            f"{prefix}_Cai": 0.1,  # μM
        }
        self.synapse_states = {
            f"{prefix}_Glu": 0,  # μM
            f"{prefix}_m": 0,
        }
        self.current_name = f"iGlu"

    def update_states(self, states, dt, pre_voltage, post_voltage, params):
        """Return updated synapse state and current."""
        prefix = self._name
        dt /= 1000
        m = states[f"{prefix}_m"]
        kCa, kiCa, Cm = (
            params[f"{prefix}_kCa"],
            params[f"{prefix}_kiCa"],
            params[f"{prefix}_Cm"],
        )
        Cao, Cai = params[f"{prefix}_Cao"], params[f"{prefix}_Cai"]
        j = (
            -1
            * kCa
            * (Cao * save_exp(-80 * pre_voltage) - Cai)
            / (kCa * Cao * save_exp(-80 * pre_voltage) + 1)
        )
        iCa = Cm * m**4 * j
        Glu = (-iCa) ** 2 / ((-iCa) ** 2 + kiCa**2)
        m_new = solve_gate_exponential(m, dt, *self.m_gate(pre_voltage))

        return {f"{prefix}_m": m_new, f"{prefix}_Glu": Glu}

    def compute_current(self, states, pre_voltage, post_voltage, params):
        prefix = self._name
        Glu = states[f"{prefix}_Glu"]
        kGlu = params[f"{prefix}_kGlu"]
        g_syn = params[f"{prefix}_gGlu"] * Glu**2 / (Glu**2 + kGlu**2)
        return g_syn * (save_exp((post_voltage - 5.82) / 115) - 1) * 1000

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        v += 1e-6
        alpha = 100 * (270 - v) / (save_exp((270 - v) / 50) - 1)
        beta = 150 / (1 + save_exp((34 + v) / 6))
        return alpha, beta


# class APB(Synapse):

#     def __init__(self, name: Optional[str] = None):
#         super().__init__(name)
#         prefix = self._name
#         self.synapse_params = {
#             # f"{prefix}_gAPB": 6.7e-3,  # # S/cm^2
#             f"{prefix}_gAPB": 0,  # # S/cm^2
#             f"{prefix}_eAPB": 0.0,  # mV
#             f"{prefix}_k": 1500,  # μM
#             f"{prefix}_n": 3.0,  # unitless
#             f"{prefix}_alpha_p": 0.1,  # 1/s * 1/μM
#             f"{prefix}_beta_p": 542.1,  # 1/s
#             f"{prefix}_alpha_c": 4.5,  # 1/s * 1/μM
#             f"{prefix}_beta_c": 23.9,  # 1/s
#             f"{prefix}_PDE_max": 43.2,  # μM
#             f"{prefix}_cGMP_max": 1500,  # μM
#             # for the integrated Ca channel
#             f"{prefix}_Cm": 90,
#             f"{prefix}_kCa": 10,  # mV
#             f"{prefix}_kiCa": 0.5,  # 1/μM
#             f"{prefix}_nCa": 2,
#             f"{prefix}_Cao": 2500,  # μM
#             f"{prefix}_Cai": 0.1,  # μM
#         }
#         self.synapse_states = {
#             f"{prefix}_PDE": 300,
#             f"{prefix}_cGMP": 300,
#             f"{prefix}_Glu": 0,
#             f"{prefix}_m": 0,  # Initial value for m gating variable
#         }

#     def update_states(self, states, dt, pre_voltage, post_voltage, params):
#         """Return updated synapse state and current."""
#         prefix = self._name
#         dt /= 1000  # Convert ms to s

#         # Update the integrated Ca channel
#         m = states[f"{prefix}_m"]
#         kCa, kiCa, Cm = (
#             params[f"{prefix}_kCa"],
#             params[f"{prefix}_kiCa"],
#             params[f"{prefix}_Cm"],
#         )
#         Cao, Cai = params[f"{prefix}_Cao"], params[f"{prefix}_Cai"]
#         j = (
#             -1
#             * kCa
#             * (Cao * save_exp(-80 * pre_voltage) - Cai)
#             / (kCa * Cao * save_exp(-80 * pre_voltage) + 1)
#         )
#         iCa = Cm * m**4 * j
#         Glu = (-iCa) ** 2 / ((-iCa) ** 2 + kiCa**2)

#         # syanptic dynamics
#         alpha_p, beta_p = params[f"{prefix}_alpha_p"], params[f"{prefix}_beta_p"]
#         alpha_c, beta_c = params[f"{prefix}_alpha_c"], params[f"{prefix}_beta_c"]
#         PDE_max, cGMP_max = params[f"{prefix}_PDE_max"], params[f"{prefix}_cGMP_max"]
#         PDE, cGMP = (
#             states[f"{prefix}_PDE"],
#             states[f"{prefix}_cGMP"],
#         )

#         # Compute the rate of change of PDE and cGMP
#         dPDE = alpha_p * (PDE_max - PDE) * Glu - beta_p * PDE
#         dcGMP = alpha_c * (cGMP_max - cGMP) - beta_c * PDE * cGMP

#         # Update PDE and cGMP
#         new_PDE = PDE + dPDE * dt
#         new_cGMP = cGMP + dcGMP * dt
#         m_new = solve_gate_exponential(m, dt, *self.m_gate(pre_voltage))

#         return {
#             f"{prefix}_PDE": new_PDE,
#             f"{prefix}_cGMP": new_cGMP,
#             f"{prefix}_Glu": Glu,
#             f"{prefix}_m": m_new,
#         }

#     def compute_current(self, states, pre_voltage, post_voltage, params):
#         prefix = self._name
#         cGMP = states[f"{prefix}_cGMP"]
#         k, n = params[f"{prefix}_k"], params[f"{prefix}_n"]
#         g_syn = params[f"{prefix}_gAPB"] * cGMP**n / (cGMP**n + k**n)
#         current = g_syn * (post_voltage - params[f"{prefix}_eAPB"]) * 1000
#         return current

#     @staticmethod
#     def m_gate(v):
#         """Voltage-dependent dynamics for the m gating variable."""
#         v += 1e-6
#         alpha = 100 * (270 - v) / (save_exp((270 - v) / 50) - 1)
#         beta = 150 / (1 + save_exp((34 + v) / 6))
#         return alpha, beta


class APB(Synapse):

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gAPB": 6.7e-3,  # # S/cm^2
            # f"{prefix}_gAPB": 0,  # # S/cm^2
            f"{prefix}_eAPB": 0.0,  # mV
            f"{prefix}_k": 1500,  # μM
            f"{prefix}_n": 3.0,  # unitless
            f"{prefix}_alpha_p": 0.1,  # 1/s * 1/μM
            f"{prefix}_beta_p": 542.1,  # 1/s
            f"{prefix}_alpha_c": 4.5,  # 1/s * 1/μM
            f"{prefix}_beta_c": 23.9,  # 1/s
            f"{prefix}_PDE_max": 43.2,  # μM
            f"{prefix}_cGMP_max": 1500,  # μM
            # for the integrated Ca channel
            f"{prefix}_Cm": 90,
            f"{prefix}_kCa": 10,  # mV
            f"{prefix}_kiCa": 0.5,  # 1/μM
            f"{prefix}_nCa": 2,
            f"{prefix}_Cao": 2500,  # μM
            f"{prefix}_Cai": 0.1,  # μM
        }
        self.synapse_states = {
            f"{prefix}_PDE": 0,
            f"{prefix}_cGMP": 0,
            f"{prefix}_Glu": 0,
            f"{prefix}_m": 0,  # Initial value for m gating variable
        }

    def update_states(self, states, dt, pre_voltage, post_voltage, params):
        """Return updated synapse state and current."""
        prefix = self._name
        dt /= 1000  # Convert ms to s

        # Update the integrated Ca channel
        m = states[f"{prefix}_m"]
        kCa, kiCa, Cm = (
            params[f"{prefix}_kCa"],
            params[f"{prefix}_kiCa"],
            params[f"{prefix}_Cm"],
        )
        Cao, Cai = params[f"{prefix}_Cao"], params[f"{prefix}_Cai"]
        j = (
            -1
            * kCa
            * (Cao * save_exp(-80 * pre_voltage) - Cai)
            / (kCa * Cao * save_exp(-80 * pre_voltage) + 1)
        )
        iCa = Cm * m**4 * j
        Glu = (-iCa) ** 2 / ((-iCa) ** 2 + kiCa**2)

        # # syanptic dynamics
        alpha_p, beta_p = params[f"{prefix}_alpha_p"], params[f"{prefix}_beta_p"]
        alpha_c, beta_c = params[f"{prefix}_alpha_c"], params[f"{prefix}_beta_c"]
        PDE_max, cGMP_max = params[f"{prefix}_PDE_max"], params[f"{prefix}_cGMP_max"]
        PDE, cGMP = (
            states[f"{prefix}_PDE"],
            states[f"{prefix}_cGMP"],
        )

        # # Compute the rate of change of PDE and cGMP
        dPDE = alpha_p * (PDE_max - PDE) * Glu - beta_p * PDE
        dcGMP = alpha_c * (cGMP_max - cGMP) - beta_c * cGMP * PDE

        # jax.debug.print("PDE={PDE}, cGMP={cGMP}", PDE=PDE, cGMP=cGMP)

        # # Update PDE and cGMP
        new_PDE = PDE + dPDE * dt
        new_cGMP = cGMP + dcGMP * dt

        m_new = solve_gate_exponential(m, dt, *self.m_gate(pre_voltage))

        return {
            f"{prefix}_PDE": new_PDE,
            f"{prefix}_cGMP": new_cGMP,
            f"{prefix}_Glu": Glu,
            f"{prefix}_m": m_new,
        }

    def compute_current(self, states, pre_voltage, post_voltage, params):
        prefix = self._name
        cGMP = states[f"{prefix}_cGMP"]

        k, n = params[f"{prefix}_k"], params[f"{prefix}_n"]
        g_syn = params[f"{prefix}_gAPB"] * cGMP**n / (cGMP**n + k**n)
        current = g_syn * (post_voltage - params[f"{prefix}_eAPB"]) * 1000
        return current

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        v += 1e-6
        alpha = 100 * (270 - v) / (save_exp((270 - v) / 50) - 1)
        beta = 150 / (1 + save_exp((34 + v) / 6))
        return alpha, beta
