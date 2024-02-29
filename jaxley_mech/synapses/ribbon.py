import jax.numpy as jnp
from jaxley.synapses.synapse import Synapse


class RibbonSynapse(Synapse):
    """
    Compute synaptic current and update synapse state for a deterministic ribbon
    synapse.

    Ribbon synapse from Schroeder et al. 2019 supplemented by the full synapse model of
    the Dayan & Abbott 2001 Theoretical Neuroscience textbook. A single exponential
    decay is used to model the postsynaptic conductance, also referred to as the
    probability that a postsynaptic channel opens given that transmitter was released.
    """

    synapse_params = {"gS": 0.5, "tau": 0.5, "e_syn": 0, "lam": 0.4, "p_r": 0.1}
    synapse_states = {"released": 0, "docked": 4, "ribboned": 25, "P_rel": 0, "P_s": 0}

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state."""

        R_max = 50
        D_max = 8

        k = 1.0
        V_half = -35
        p_r = params["p_r"]
        λ = params["lam"]
        ρ = 0.35

        P_max = 1.0

        # Presynaptic voltage to calcium to release probability
        p_d_t = 1 / (1 + jnp.exp(-k * (pre_voltage - V_half)))

        # Vesicle release (NOTE: p_d_t is the mean of the beta distribution)
        new_released = p_d_t * u["docked"]

        # Movement to the dock
        new_docked = u["docked"] + p_r * u["ribboned"] - new_released
        new_docked = jnp.maximum(new_docked, D_max)

        # Movement to the ribbon
        new_ribboned = u["ribboned"] + λ - new_docked
        new_ribboned = jnp.maximum(new_ribboned, R_max)

        # Vesicle released to "transmitter release probability"
        P_rel = new_released / D_max
        P_s = P_max * jnp.exp(-delta_t / params["tau"])

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
