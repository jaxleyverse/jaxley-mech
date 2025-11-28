from typing import Dict, Optional, Union

import jax.debug
import jax.numpy as jnp
from jax.lax import select
from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler, save_exp, solve_gate_exponential, solve_inf_gate_exponential

from jaxley_mech.solvers import SolverExtension

META = {
    "cell_type": "bipolar cell",
    "species": ["Human Embryonic Kidney Cells"],
    "reference": "Benav, H. (2012)",
    "doi": "http://hdl.handle.net/10900/46043",
    "note": "The model is using the reduced voltage not the membrane potential. Therefore, the resting potential has to be given as a parameter."
}


class Ca_T(Channel):
    """ Transient type of Calcium Channel """

    def __init__(self, 
                 v_rest_global: float,
                 name: Optional[str] = None):
        
        self.current_is_in_mA_per_cm2 = True
                
        super().__init__(name)
        self.channel_params = {
            # To match the experimental data, I had to adapt the conductance
            f"{self._name}_gCa_T": 0.03634,  # S/cm^2
 
            # The facilitate the calculation we treat the resting potential
            # as fixed and set it to v_rest_global
            f"{self._name}_v_r": v_rest_global,  # mV

        }
        self.channel_states = {
            # This is the calcaultion from the dissertation of H. Benav:
            # E_{Ca_T} = Neernst Ca++ * 45/100 + E_K * 55/100
            # E_K is taken from usui96 of Kv channel
            # E_{Ca_T} = 132.65mV ×0.45 + (-58mV) * 0.55 = 27.5mV
            # Equilibrium potential for calcium:
            "eCa": 27.5, # mV  
            
            
            # Experimentally determined values for the gating variables
            # Initial value for m gating variable
            f"{self._name}_m": 0.1, 
            # Initial value for h gating variable  
            f"{self._name}_h": 0.9,  
        }
        self.current_name = f"iCa_T"
        self.META = META


    def update_states(
            self,
            states: Dict[str, jnp.ndarray],
            dt: float,
            v_m: float,
            params: Dict[str, jnp.ndarray],
        ):
            """Update state of gating variables."""
            prefix = self._name
            m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]

            # Since the gating variables are given in the steady state form
            # use solve_inf_gate_exponential to calculate the new values
            m_new = solve_inf_gate_exponential(m, dt, *self.m_gate(v_m, params[f"{self._name}_v_r"]))
            h_new = solve_inf_gate_exponential(h, dt, *self.h_gate(v_m, params[f"{self._name}_v_r"]))

            return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v_m, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        Ca_T_cond = params[f"{prefix}_gCa_T"] * m* h
         
        return  Ca_T_cond * (v_m - params["eCa"]) # mS/cm^2 *mV = mA/cm^2

    def init_state(self, states, v_m, params, delta_t):

        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name


        m_inf, _  = self.m_gate(v_m, params[f"{self._name}_v_r"]) 
        h_inf, _ = self.h_gate(v_m, params[f"{self._name}_v_r"]) 

        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_h": h_inf
        }

    @staticmethod
    def m_gate(v_m, v_rest):
        # Activation
        # The model in the dissertation of H. Benav is based on reduced voltage
        # and not on the membrane potential. Therefore, always subtract the resting potential.
        v = v_m - v_rest
     
        # Calculate the time constant
        tau_m = 1.358 + (21.675 / (1 + save_exp((v_m-39.9596)/4.110392)))

        # Calculate the steady state value
        m_inf = (1 / (1 + save_exp((v  - 37.5456)/-3.073015)))


        # Give m _inf and tau_m back
        return m_inf, tau_m

    @staticmethod
    def h_gate(v_m, v_rest):
        # Inactivation
        
        # The model in the dissertation of H. Benav is based on reduced voltage
        # and not on the membrane potential. Therefore, always subtract the resting potential.   
        v = v_m - v_rest

        # Calculate the time constant
        tau_h = 65.8207 + 0.00223 * save_exp((v-80) / 4.78719)

        # Calculate the steady state value
        h_inf = (1 / (1 + save_exp((v - 8.968)/8.416382)))

        # Give h_inf and tau_h back
        return h_inf, tau_h


class K_IR(Channel):

    """ Inward Rectifying potassium channel """

    def __init__(self, 
                 v_rest_global: float,
                 name: Optional[str] = None):
        
        self.current_is_in_mA_per_cm2 = True
        
        super().__init__(name)
        
        self.channel_params = {
            
            # To match the experimental data, I had to adapt the conductance
            f"{self._name}_gK_IR": 6.27e-4,  # S/cm^2


            # Using the same equilibrium potential as the potassium channel 
            # implemented by the Usui class
            "eK_IR": -58,  # mV

            # The facilitate the calculation we treat the resting potential
            # as fixed and set it to v_rest_global
            f"{self._name}_v_r": v_rest_global,  # mV
        }
        self.channel_states = {
            # Experimentally determined values for the gating variables
            # Initial value for m gating variable           
            f"{self._name}_m": 0.9985,  
        }
        self.current_name = f"iK_IR"
        self.META = META
  

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v_m, params: Dict[str, jnp.ndarray]
    ):
        """Update state of gating variables."""
        prefix = self._name
        m = states[f"{prefix}_m"]

        # Since gating variables are given in a differntial equation,
        # use solve_gate_exponential to solve the gating variable
        m_new = solve_gate_exponential(m, dt, *self.m_gate(v_m, params[f"{self._name}_v_r"]))

        return {f"{prefix}_m": m_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v_m, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        g = params[f"{prefix}_gK_IR"] * m
        return g * (v_m - params["eK_IR"]) # mS/cm^2 *mV = mA/cm^2

    def init_state(self, states, v_m, params, delta_t):

        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = self.m_gate(v_m, params[f"{self._name}_v_r"])
 
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m)
        }

    @staticmethod
    def m_gate(v_m, v_rest):
        """Voltage-dependent dynamics for the n gating variable."""

        # The model in the dissertation of H. Benav is based on reduced voltage
        # and not on the membrane potential. Therefore, always subtract the resting potential.
        v = v_m - v_rest
   
        alpha = 0.13289  * (save_exp((v - 8.94)/ -6.3902))
        beta = 0.16994  * (save_exp((v - 48.94)/ 27.714))


        return alpha, beta       
