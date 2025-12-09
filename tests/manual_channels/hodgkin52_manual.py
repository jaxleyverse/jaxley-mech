from typing import Dict, Optional

import jax
import jax.numpy as jnp
from jaxley.solver_gate import save_exp, solve_gate_exponential

from jaxley_mech.channels.hodgkin52 import Na, K
from jaxley_mech.solvers import SolverExtension


class Na8StatesManual(Na, SolverExtension):
    """Sodium channel in the formulation of Markov model with 8 states"""

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
        self.solver = solver
        self._edge_transitions = self._build_edge_transitions()
        self.channel_params = {
            f"{prefix}_gNa": 40e-3,  # S/cm^2
            f"{prefix}_eNa": 55.0,  # mV
            f"{prefix}_N_Na": 1e4,  # number of Na channels
            f"{prefix}_noise_seed": 0,
        }
        # Initialize all states
        self.channel_states = {
            f"{prefix}_C3": 1.0,
            f"{prefix}_C2": 0.0,
            f"{prefix}_C1": 0.0,
            f"{prefix}_O": 0,
            f"{prefix}_I3": 0.0,
            f"{prefix}_I2": 0.0,
            f"{prefix}_I1": 0.0,
            f"{prefix}_I": 0,
            f"{prefix}_noise_ptr": 0,  # index for optional xi sequence in params
        }
        self.current_name = "i_Na"
        self.META = {
            "reference": "Armstrong, C. M. (1981).",
            "doi": "https://doi.org/10.1152/physrev.1981.61.3.644",
            "species": "squid",
            "ion": "Na",
        }

    @staticmethod
    def _build_edge_transitions():
        """Match StatesChannel ordering for m^3 h (20 directed edges)."""
        combos = [
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
        ]
        combo_index = {c: i for i, c in enumerate(combos)}
        powers = [3, 1]
        transitions = []
        for i, combo in enumerate(combos):
            for g_idx, power in enumerate(powers):
                k = combo[g_idx]
                if k < power:
                    forward = list(combo)
                    forward[g_idx] += 1
                    j = combo_index[tuple(forward)]
                    transitions.append(("alpha", i, j, power - k, g_idx))
                if k > 0:
                    back = list(combo)
                    back[g_idx] -= 1
                    j = combo_index[tuple(back)]
                    transitions.append(("beta", i, j, k, g_idx))
        return transitions

    def _sample_xi(self, states, params, size):
        ptr_key = f"{self._name}_noise_ptr"
        seed_key = f"{self._name}_noise_seed"
        ptr = jnp.asarray(states[ptr_key])

        seed_arr = jnp.asarray(params.get(seed_key, 0), dtype=jnp.uint32)
        seed_vec = (
            jnp.broadcast_to(seed_arr, ptr.shape).ravel()
            if seed_arr.ndim == 0
            else seed_arr.ravel()
        )
        ptr_vec = jnp.asarray(ptr, dtype=jnp.uint32).ravel()

        def sample_one(seed_i, ptr_i):
            key = jax.random.PRNGKey(seed_i)
            key = jax.random.fold_in(key, ptr_i)
            return jax.random.normal(key, shape=(size,))

        xi = jax.vmap(sample_one)(seed_vec, ptr_vec).T
        return xi

    def derivatives(self, t, states, args):
        """Calculate the derivatives for the Markov states."""
        C3, C2, C1, O, I3, I2, I1, I = states
        v = args[0]
        alpha_m, beta_m = self.m_gate(v)
        alpha_h, beta_h = self.h_gate(v)

        # Transitions for activation pathway (C3 -> O)
        C3_to_C2 = 3 * alpha_m * C3
        C2_to_C1 = 2 * alpha_m * C2
        C1_to_O = alpha_m * C1

        O_to_C1 = 3 * beta_m * O
        C1_to_C2 = 2 * beta_m * C1
        C2_to_C3 = beta_m * C2

        # Transitions for inactivation pathway (I3 -> I)
        I3_to_I2 = 3 * alpha_m * I3
        I2_to_I1 = 2 * alpha_m * I2
        I1_to_I = alpha_m * I1

        I_to_I1 = 3 * beta_m * I
        I1_to_I2 = 2 * beta_m * I1
        I2_to_I3 = beta_m * I2

        # C to I and I to C transitions (C3, C2, C1, O <-> I3, I2, I1, I)
        C3_to_I3 = beta_h * C3
        C2_to_I2 = beta_h * C2
        C1_to_I1 = beta_h * C1
        O_to_I = beta_h * O

        I3_to_C3 = alpha_h * I3
        I2_to_C2 = alpha_h * I2
        I1_to_C1 = alpha_h * I1
        I_to_O = alpha_h * I

        # Derivatives for each state
        dC3_dt = C2_to_C3 - C3_to_C2 + I3_to_C3 - C3_to_I3
        dC2_dt = C3_to_C2 - C2_to_C1 + C1_to_C2 - C2_to_C3 + I2_to_C2 - C2_to_I2
        dC1_dt = C2_to_C1 - C1_to_O + O_to_C1 - C1_to_C2 + I1_to_C1 - C1_to_I1
        dO_dt = C1_to_O - O_to_C1 + I_to_O - O_to_I

        dI3_dt = I2_to_I3 - I3_to_I2 + C3_to_I3 - I3_to_C3
        dI2_dt = I3_to_I2 - I2_to_I1 + I1_to_I2 - I2_to_I3 + C2_to_I2 - I2_to_C2
        dI1_dt = I2_to_I1 - I1_to_I + I_to_I1 - I1_to_I2 + C1_to_I1 - I1_to_C1
        dI_dt = I1_to_I - I_to_I1 + O_to_I - I_to_O

        return jnp.array([dC3_dt, dC2_dt, dC1_dt, dO_dt, dI3_dt, dI2_dt, dI1_dt, dI_dt])

    def diffusion_matrix(self, x, v, params):
        """Fox-Lu diffusion matrix for the 8-state Na channel."""
        alpha_m, beta_m = self.m_gate(v)
        alpha_h, beta_h = self.h_gate(v)
        C3, C2, C1, O, I3, I2, I1, I = jnp.maximum(x, 0.0)
        N = params[f"{self._name}_N_Na"]

        D = jnp.zeros((8, 8), dtype=x.dtype)

        def add(D, rate, nu):
            return D + rate * jnp.outer(nu, nu)

        D = add(
            D,
            3 * alpha_m * C3 * N,
            jnp.array([-1, +1, 0, 0, 0, 0, 0, 0], dtype=x.dtype),
        )
        D = add(
            D, beta_m * C2 * N, jnp.array([+1, -1, 0, 0, 0, 0, 0, 0], dtype=x.dtype)
        )
        D = add(
            D,
            2 * alpha_m * C2 * N,
            jnp.array([0, -1, +1, 0, 0, 0, 0, 0], dtype=x.dtype),
        )
        D = add(
            D, 2 * beta_m * C1 * N, jnp.array([0, +1, -1, 0, 0, 0, 0, 0], dtype=x.dtype)
        )
        D = add(
            D, alpha_m * C1 * N, jnp.array([0, 0, -1, +1, 0, 0, 0, 0], dtype=x.dtype)
        )
        D = add(
            D, 3 * beta_m * O * N, jnp.array([0, 0, +1, -1, 0, 0, 0, 0], dtype=x.dtype)
        )

        D = add(
            D,
            3 * alpha_m * I3 * N,
            jnp.array([0, 0, 0, 0, -1, +1, 0, 0], dtype=x.dtype),
        )
        D = add(
            D, beta_m * I2 * N, jnp.array([0, 0, 0, 0, +1, -1, 0, 0], dtype=x.dtype)
        )
        D = add(
            D,
            2 * alpha_m * I2 * N,
            jnp.array([0, 0, 0, 0, 0, -1, +1, 0], dtype=x.dtype),
        )
        D = add(
            D, 2 * beta_m * I1 * N, jnp.array([0, 0, 0, 0, 0, +1, -1, 0], dtype=x.dtype)
        )
        D = add(
            D, alpha_m * I1 * N, jnp.array([0, 0, 0, 0, 0, 0, -1, +1], dtype=x.dtype)
        )
        D = add(
            D, 3 * beta_m * I * N, jnp.array([0, 0, 0, 0, 0, 0, +1, -1], dtype=x.dtype)
        )

        D = add(
            D, beta_h * C3 * N, jnp.array([-1, 0, 0, 0, +1, 0, 0, 0], dtype=x.dtype)
        )
        D = add(
            D, alpha_h * I3 * N, jnp.array([+1, 0, 0, 0, -1, 0, 0, 0], dtype=x.dtype)
        )
        D = add(
            D, beta_h * C2 * N, jnp.array([0, -1, 0, 0, 0, +1, 0, 0], dtype=x.dtype)
        )
        D = add(
            D, alpha_h * I2 * N, jnp.array([0, +1, 0, 0, 0, -1, 0, 0], dtype=x.dtype)
        )
        D = add(
            D, beta_h * C1 * N, jnp.array([0, 0, -1, 0, 0, 0, +1, 0], dtype=x.dtype)
        )
        D = add(
            D, alpha_h * I1 * N, jnp.array([0, 0, +1, 0, 0, 0, -1, 0], dtype=x.dtype)
        )
        D = add(D, beta_h * O * N, jnp.array([0, 0, 0, -1, 0, 0, 0, +1], dtype=x.dtype))
        D = add(
            D, alpha_h * I * N, jnp.array([0, 0, 0, +1, 0, 0, 0, -1], dtype=x.dtype)
        )

        return D / jnp.maximum(N**2, 1e-12)

    def edge_noise_increment(self, y, v, params, xi):
        """Edge-based noise increment with one source per directed transition."""
        alpha_m, beta_m = self.m_gate(v)
        alpha_h, beta_h = self.h_gate(v)
        y_mat = jnp.reshape(y, (y.shape[0], -1))
        xi_mat = jnp.reshape(xi, (len(self._edge_transitions), -1))
        N = params[f"{self._name}_N_Na"]
        N_safe = jnp.maximum(N, 1e-12)

        dy_noise = jnp.zeros_like(y_mat)
        for xi_k, (direction, i, j, factor, g_idx) in zip(
            xi_mat, self._edge_transitions
        ):
            alpha = alpha_m if g_idx == 0 else alpha_h
            beta = beta_m if g_idx == 0 else beta_h
            yi = jnp.maximum(y_mat[i], 0.0)
            rate = factor * (alpha if direction == "alpha" else beta) * yi * N
            nu = (
                jnp.zeros((y_mat.shape[0], 1), dtype=y_mat.dtype)
                .at[j, 0]
                .set(1.0)
                .at[i, 0]
                .add(-1.0)
            )
            amp = jnp.sqrt(jnp.maximum(rate, 0.0)) / N_safe
            dy_noise = dy_noise + amp * xi_k[None, :] * nu
        return dy_noise.reshape(y.shape)

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state."""
        prefix = self._name

        # Retrieve states
        C3 = states[f"{prefix}_C3"]
        C2 = states[f"{prefix}_C2"]
        C1 = states[f"{prefix}_C1"]
        O = states[f"{prefix}_O"]

        I3 = states[f"{prefix}_I3"]
        I2 = states[f"{prefix}_I2"]
        I1 = states[f"{prefix}_I1"]
        I = states[f"{prefix}_I"]

        y0 = jnp.array([C3, C2, C1, O, I3, I2, I1, I])

        solver_kind = self.solver_name
        y_in = y0

        uses_noise = solver_kind in ("sde", "sde_implicit", "sde_edges")
        xi_array = None
        if uses_noise:
            xi_size = (
                len(self._edge_transitions)
                if solver_kind == "sde_edges"
                else y0.shape[0]
            )
            xi_array = self._sample_xi(states, params, size=xi_size)

        if solver_kind in ("sde", "sde_implicit"):
            xi_array = jnp.reshape(xi_array, y_in.shape)
            args_tuple = (self.diffusion_matrix, v, params, xi_array)
        elif solver_kind == "sde_edges":
            args_tuple = (self.edge_noise_increment, v, params, xi_array)
        else:
            args_tuple = (v,)

        y_new = self.solver_func(y_in, dt, self.derivatives, args_tuple)

        # Unpack new states
        C3_new, C2_new, C1_new, O_new, I3_new, I2_new, I1_new, I_new = y_new

        return {
            f"{prefix}_C3": C3_new,
            f"{prefix}_C2": C2_new,
            f"{prefix}_C1": C1_new,
            f"{prefix}_O": O_new,
            f"{prefix}_I3": I3_new,
            f"{prefix}_I2": I2_new,
            f"{prefix}_I1": I1_new,
            f"{prefix}_I": I_new,
            f"{prefix}_noise_ptr": states[f"{prefix}_noise_ptr"] + 1,
        }

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        O = states[f"{prefix}_O"]
        gNa = params[f"{prefix}_gNa"] * O  # S/cm^2
        return gNa * (v - params[f"{prefix}_eNa"])

    def init_state(self, states, v, params, delta_t):
        """Initialize the state to steady-state values."""
        prefix = self._name
        alpha_m, beta_m = self.m_gate(v)
        alpha_h, beta_h = self.h_gate(v)

        m_inf = alpha_m / (alpha_m + beta_m)
        h_inf = alpha_h / (alpha_h + beta_h)

        # Calculate steady-state probabilities
        C3 = (1 - m_inf) ** 3 * h_inf
        C2 = 3 * m_inf * (1 - m_inf) ** 2 * h_inf
        C1 = 3 * m_inf**2 * (1 - m_inf) * h_inf
        O = m_inf**3 * h_inf
        I3 = (1 - m_inf) ** 3 * (1 - h_inf)
        I2 = 3 * m_inf * (1 - m_inf) ** 2 * (1 - h_inf)
        I1 = 3 * m_inf**2 * (1 - m_inf) * (1 - h_inf)
        I = m_inf**3 * (1 - h_inf)

        return {
            f"{prefix}_C3": C3,
            f"{prefix}_C2": C2,
            f"{prefix}_C1": C1,
            f"{prefix}_O": O,
            f"{prefix}_I3": I3,
            f"{prefix}_I2": I2,
            f"{prefix}_I1": I1,
            f"{prefix}_I": I,
            f"{prefix}_noise_ptr": 0,
        }




class K5StatesManual(K, SolverExtension):
    """Potassium channel in the formulation of Markov model with 5 states"""

    def __init__(
        self,
        name: Optional[str] = None,
        solver: Optional[str] = None,
        rtol: float = 1e-8,
        atol: float = 1e-8,
        max_steps: int = 10,
    ):
        super().__init__(name)
        SolverExtension.__init__(self, solver, rtol, atol, max_steps)
        prefix = self._name
        self.solver = solver
        self._edge_transitions = self._build_edge_transitions()
        self.channel_params = {
            f"{prefix}_gK": 35e-3,  # S/cm^2
            f"{prefix}_eK": -77.0,  # mV
            f"{prefix}_N_K": 1e4,  # number of K channels
            f"{prefix}_noise_seed": 0,
        }
        self.channel_states = {
            f"{prefix}_C4": 1.0,
            f"{prefix}_C3": 0,
            f"{prefix}_C2": 0,
            f"{prefix}_C1": 0,
            f"{prefix}_O": 0.0,
            f"{prefix}_noise_ptr": 0,  # index for optional xi sequence in params
        }
        self.current_name = f"i_K"
        self.META = {
            "reference": "Armstrong, (1969)",
            "doi": "https://doi.org/10.1085/jgp.54.5.553",
            "species": "squid",
            "ion": "K",
        }

    @staticmethod
    def _build_edge_transitions():
        combos = [(0,), (1,), (2,), (3,), (4,)]
        combo_index = {c: i for i, c in enumerate(combos)}
        power = 4
        transitions = []
        for i, combo in enumerate(combos):
            k = combo[0]
            if k < power:
                forward = (k + 1,)
                j = combo_index[forward]
                transitions.append(("alpha", i, j, power - k, 0))
            if k > 0:
                back = (k - 1,)
                j = combo_index[back]
                transitions.append(("beta", i, j, k, 0))
        return transitions

    def _sample_xi(self, states, params, size):
        ptr_key = f"{self._name}_noise_ptr"
        seed_key = f"{self._name}_noise_seed"
        ptr = jnp.asarray(states[ptr_key])

        seed_arr = jnp.asarray(params.get(seed_key, 0), dtype=jnp.uint32)
        seed_vec = (
            jnp.broadcast_to(seed_arr, ptr.shape).ravel()
            if seed_arr.ndim == 0
            else seed_arr.ravel()
        )
        ptr_vec = jnp.asarray(ptr, dtype=jnp.uint32).ravel()

        def sample_one(seed_i, ptr_i):
            key = jax.random.PRNGKey(seed_i)
            key = jax.random.fold_in(key, ptr_i)
            return jax.random.normal(key, shape=(size,))

        xi = jax.vmap(sample_one)(seed_vec, ptr_vec).T
        return xi

    def derivatives(self, t, states, args):
        """Calculate the derivatives for the Markov states."""
        C4, C3, C2, C1, O = states
        v = args[0]
        alpha_n, beta_n = self.n_gate(
            v
        )  # Use voltage (t) to calculate alpha_n and beta_n

        # Transitions for activation pathway
        C4_to_C3 = 4 * alpha_n * C4
        C3_to_C2 = 3 * alpha_n * C3
        C2_to_C1 = 2 * alpha_n * C2
        C1_to_O = alpha_n * C1

        O_to_C1 = 4 * beta_n * O
        C1_to_C2 = 3 * beta_n * C1
        C2_to_C3 = 2 * beta_n * C2
        C3_to_C4 = beta_n * C3

        # Derivatives of each state
        dC4_dt = C3_to_C4 - C4_to_C3
        dC3_dt = C4_to_C3 - C3_to_C2 - C3_to_C4 + C2_to_C3
        dC2_dt = C3_to_C2 - C2_to_C1 - C2_to_C3 + C1_to_C2
        dC1_dt = C2_to_C1 - C1_to_O - C1_to_C2 + O_to_C1
        dO_dt = C1_to_O - O_to_C1

        return jnp.array([dC4_dt, dC3_dt, dC2_dt, dC1_dt, dO_dt])

    def diffusion_matrix(self, x, v, params):
        """Fox-Lu diffusion matrix for the 5-state K channel."""
        alpha_n, beta_n = self.n_gate(v)
        C4, C3, C2, C1, O = jnp.maximum(x, 0.0)
        N = params[f"{self._name}_N_K"]

        D = jnp.zeros((5, 5), dtype=x.dtype)

        def add(D, rate, nu):
            return D + rate * jnp.outer(nu, nu)

        D = add(D, 4 * alpha_n * C4 * N, jnp.array([-1, +1, 0, 0, 0], dtype=x.dtype))
        D = add(D, beta_n * C3 * N, jnp.array([+1, -1, 0, 0, 0], dtype=x.dtype))
        D = add(D, 3 * alpha_n * C3 * N, jnp.array([0, -1, +1, 0, 0], dtype=x.dtype))
        D = add(D, 2 * beta_n * C2 * N, jnp.array([0, +1, -1, 0, 0], dtype=x.dtype))
        D = add(D, 2 * alpha_n * C2 * N, jnp.array([0, 0, -1, +1, 0], dtype=x.dtype))
        D = add(D, 3 * beta_n * C1 * N, jnp.array([0, 0, +1, -1, 0], dtype=x.dtype))
        D = add(D, alpha_n * C1 * N, jnp.array([0, 0, 0, -1, +1], dtype=x.dtype))
        D = add(D, 4 * beta_n * O * N, jnp.array([0, 0, 0, +1, -1], dtype=x.dtype))

        return D / jnp.maximum(N**2, 1e-12)

    def edge_noise_increment(self, y, v, params, xi):
        """Edge-based noise increment with one source per directed transition."""
        alpha_n, beta_n = self.n_gate(v)
        y_mat = jnp.reshape(y, (y.shape[0], -1))
        xi_mat = jnp.reshape(xi, (len(self._edge_transitions), -1))
        N = params[f"{self._name}_N_K"]
        N_safe = jnp.maximum(N, 1e-12)

        dy_noise = jnp.zeros_like(y_mat)
        for xi_k, (direction, i, j, factor, _g_idx) in zip(
            xi_mat, self._edge_transitions
        ):
            yi = jnp.maximum(y_mat[i], 0.0)
            rate = factor * (alpha_n if direction == "alpha" else beta_n) * yi * N
            nu = (
                jnp.zeros((y_mat.shape[0], 1), dtype=y_mat.dtype)
                .at[j, 0]
                .set(1.0)
                .at[i, 0]
                .add(-1.0)
            )
            amp = jnp.sqrt(jnp.maximum(rate, 0.0)) / N_safe
            dy_noise = dy_noise + amp * xi_k[None, :] * nu
        return dy_noise.reshape(y.shape)

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
        **kwargs,
    ):
        """Update state using the specified solver."""
        prefix = self._name

        # Retrieve states
        C4 = states[f"{prefix}_C4"]
        C3 = states[f"{prefix}_C3"]
        C2 = states[f"{prefix}_C2"]
        C1 = states[f"{prefix}_C1"]
        O = states[f"{prefix}_O"]

        y0 = jnp.array([C4, C3, C2, C1, O])

        solver_kind = self.solver_name
        y_in = y0

        # Parameters for dynamics
        uses_noise = solver_kind in ("sde", "sde_implicit", "sde_edges")
        xi_array = None
        if uses_noise:
            xi_size = (
                len(self._edge_transitions)
                if solver_kind == "sde_edges"
                else y0.shape[0]
            )
            xi_array = self._sample_xi(states, params, size=xi_size)

        if solver_kind in ("sde", "sde_implicit"):
            xi_array = jnp.reshape(xi_array, y_in.shape)
            args_tuple = (self.diffusion_matrix, v, params, xi_array)
        elif solver_kind == "sde_edges":
            args_tuple = (self.edge_noise_increment, v, params, xi_array)
        else:
            args_tuple = (v,)

        y_new = self.solver_func(y_in, dt, self.derivatives, args_tuple)

        # Unpack new states
        C4_new, C3_new, C2_new, C1_new, O_new = y_new

        return {
            f"{prefix}_C4": C4_new,
            f"{prefix}_C3": C3_new,
            f"{prefix}_C2": C2_new,
            f"{prefix}_C1": C1_new,
            f"{prefix}_O": O_new,
            f"{prefix}_noise_ptr": states[f"{prefix}_noise_ptr"] + 1,
        }

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        O = states[f"{prefix}_O"]
        gK = params[f"{prefix}_gK"] * O  # S/cm^2
        return gK * (v - params[f"{prefix}_eK"])

    def init_state(self, states, v, params, delta_t):
        """Initialize the state to steady-state values."""
        prefix = self._name
        alpha_n, beta_n = self.n_gate(v)

        n_inf = alpha_n / (alpha_n + beta_n)

        # Calculate steady-state probabilities
        C4 = (1 - n_inf) ** 4
        C3 = 4 * n_inf * (1 - n_inf) ** 3
        C2 = 6 * n_inf**2 * (1 - n_inf) ** 2
        C1 = 4 * n_inf**3 * (1 - n_inf)
        O = n_inf**4

        return {
            f"{prefix}_C4": C4,
            f"{prefix}_C3": C3,
            f"{prefix}_C2": C2,
            f"{prefix}_C1": C1,
            f"{prefix}_O": O,
            f"{prefix}_noise_ptr": 0,
        }


