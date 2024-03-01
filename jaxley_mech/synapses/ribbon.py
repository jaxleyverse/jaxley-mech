import jax.numpy as jnp
from jaxley.synapses.synapse import Synapse


class RibbonSynapse(Synapse):
    """
    Compute synaptic current and update synapse state for a deterministic ribbon
    synapse.

    Ribbon synapse from Schroeder et al. 2020 supplemented by the full synapse model of
    the Dayan & Abbott 2001 Theoretical Neuroscience textbook. A single exponential
    decay is used to model postsynaptic conductance.

    synapse_params:
        gS: Maximal synaptic conductance (S)
        tau: Decay time constant of postsynaptic conductance (s)
        e_syn: Reversal potential of postsynaptic membrane at the receptor (mV)
        lam: Vesicle replenishment rate at the ribbon
        p_r: Probability of a vesicle at the ribbon moving to the dock
        D_max: Maximum number of docked vesicles
        R_max: Maximum number of vesicles at the ribbon

    synapse_states:
        released: Number of vesicles released
        docked: Number of vesicles at the dock
        ribboned: Number of vesicles at the ribbon
        P_rel: Normalized vesicle release
        P_s: Kernel of postsynaptic conductance
    """

    synapse_params = {
        "gS": 0.5,
        "tau": 0.5,
        "e_syn": 0,
        "lam": 0.4,
        "p_r": 0.1,
        "D_max": 8,
        "R_max": 50,
    }
    synapse_states = {"released": 0, "docked": 4, "ribboned": 25, "P_rel": 0, "P_s": 0}

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state."""

        k = 1.0
        V_half = -35

        # Presynaptic voltage to calcium to release probability
        p_d_t = 1 / (1 + jnp.exp(-k * (pre_voltage - V_half)))

        # Vesicle release (NOTE: p_d_t is the mean of the beta distribution)
        new_released = p_d_t * u["docked"]

        # Movement to the dock
        new_docked = u["docked"] + params["p_r"] * u["ribboned"] - new_released
        new_docked = jnp.maximum(new_docked, params["D_max"])

        # Movement to the ribbon
        new_ribboned = u["ribboned"] + params["lam"] - new_docked
        new_ribboned = jnp.maximum(new_ribboned, params["R_max"])

        P_rel = new_released / params["D_max"]
        P_s = jnp.exp(-delta_t / params["tau"])

        return {
            "released": new_released,
            "docked": new_docked,
            "ribboned": new_ribboned,
            "P_rel": P_rel,
            "P_s": P_s,
        }

    def compute_current(self, u, pre_voltage, post_voltage, params):
        """Return updated current."""
        g_syn = params["gS"] * u["P_rel"] * u["P_s"]
        return g_syn * (post_voltage - params["e_syn"])
