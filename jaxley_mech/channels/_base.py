from __future__ import annotations

import math
from itertools import product
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from jaxley.channels import Channel
from jaxley_mech.solvers import SolverExtension


class StatesChannel(Channel, SolverExtension):
    """Mixin for HH-style Markov chains built from independent two-state gates."""

    def __init__(
        self,
        name: Optional[str],
        gate_specs: Sequence[Tuple[str, int, Callable[[float], Tuple[float, float]]]],
        count_param: str,
        solver: str = "sde",
        rtol: float = 1e-8,
        atol: float = 1e-8,
        max_steps: int = 10,
        noise_seed_param: Optional[str] = None,
        xi_param: Optional[str] = None,
        state_key_fn: Optional[Callable[[Tuple[int, ...], int], str]] = None,
    ):
        """
        Args:
            gate_specs: list of (gate_name, power, gate_fn) where gate_fn returns (alpha, beta).
            count_param: parameter name for channel count N (Fox–Lu scaling).
        """
        # Ensure base solver setup happens first so solver_func is available.
        SolverExtension.__init__(self, solver, rtol, atol, max_steps)

        self.gate_specs = gate_specs
        self.count_param = count_param
        self.noise_seed_param = noise_seed_param or f"{self._name}_noise_seed"
        self.xi_param = xi_param or f"{self._name}_xi"
        self.noise_ptr_key = f"{self._name}_noise_ptr"

        self._gate_names = [g for g, _, _ in gate_specs]
        self._powers = [p for _, p, _ in gate_specs]
        combos: List[Tuple[int, ...]] = list(product(*[range(p + 1) for p in self._powers]))
        # For HH-style m^p with a single h-gate, order h-open states first (C*,O), then h-closed (I*).
        if (
            state_key_fn is None
            and len(self._powers) == 2
            and self._powers[1] == 1
            and self._gate_names[1].lower().startswith(("h", "i"))
        ):
            combos = sorted(combos, key=lambda c: (-c[1], c[0]))
        self._state_combos = combos
        self._combo_to_idx = {combo: i for i, combo in enumerate(self._state_combos)}
        self.state_keys = [
            state_key_fn(combo, i)
            if state_key_fn is not None
            else self._default_state_key(combo)
            for i, combo in enumerate(self._state_combos)
        ]
        self.open_state_idx = self._combo_to_idx[tuple(self._powers)]
        closed_idx = self._combo_to_idx[tuple(0 for _ in self._powers)]

        # Precompute binomial coefficients for steady-state distribution.
        self._binom_coeffs = [
            math.prod(math.comb(power, k) for power, k in zip(self._powers, combo))
            for combo in self._state_combos
        ]

        # Precompute transitions (direction, src, dst, factor, gate_idx).
        transitions = []
        for i, combo in enumerate(self._state_combos):
            for g_idx, (_, power, _) in enumerate(gate_specs):
                k = combo[g_idx]
                if k < power:
                    forward = list(combo)
                    forward[g_idx] += 1
                    j = self._combo_to_idx[tuple(forward)]
                    transitions.append(("alpha", i, j, power - k, g_idx))
                if k > 0:
                    back = list(combo)
                    back[g_idx] -= 1
                    j = self._combo_to_idx[tuple(back)]
                    transitions.append(("beta", i, j, k, g_idx))
        self._transitions = transitions

        if self.channel_states is None:
            self.channel_states = {}
        if self.channel_params is None:
            self.channel_params = {}

        for idx, key in enumerate(self.state_keys):
            self.channel_states.setdefault(key, 1.0 if idx == closed_idx else 0.0)
        self.channel_states.setdefault(self.noise_ptr_key, 0)
        self.channel_params.setdefault(self.count_param, 1e4)
        self.channel_params.setdefault(self.noise_seed_param, 0)

    # --- Helpers -----------------------------------------------------
    def _gate_rates(self, v: float) -> List[Tuple[float, float]]:
        return [gate_fn(v) for _, _, gate_fn in self.gate_specs]

    def _default_state_key(self, combo: Tuple[int, ...]) -> str:
        """Human-friendly labels for common HH topologies."""
        if len(combo) == 1:
            k = combo[0]
            power = self._powers[0]
            if k == power:
                return f"{self._name}_O"
            return f"{self._name}_C{power - k}"

        if (
            len(combo) == 2
            and self._powers[1] == 1
            and self._gate_names[1].lower().startswith(("h", "i"))
        ):
            m_open, h_open = combo
            m_power = self._powers[0]
            if h_open == 0:
                label = "I" if m_open == m_power else f"I{m_power - m_open}"
            else:
                if m_open == m_power:
                    label = "O"
                else:
                    label = f"C{m_power - m_open}"
            return f"{self._name}_{label}"

        # Fallback generic
        idx = self._combo_to_idx[combo]
        return f"{self._name}_s{idx}"

    def project_simplex(self, y: jnp.ndarray) -> jnp.ndarray:
        y = jnp.clip(y, 0.0)
        return y / jnp.maximum(y.sum(axis=0, keepdims=True), 1e-12)

    def sample_xi(self, states: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]):
        """Sample xi using noise_seed + noise_ptr or optional precomputed xi sequence."""
        ptr = jnp.asarray(states[self.noise_ptr_key])
        xi_key = self.xi_param
        if xi_key in params:
            idx = ptr % params[xi_key].shape[0]
            return params[xi_key][idx]

        seed_arr = jnp.asarray(params.get(self.noise_seed_param, 0), dtype=jnp.uint32)
        seed_vec = (
            jnp.broadcast_to(seed_arr, ptr.shape).ravel()
            if seed_arr.ndim == 0
            else seed_arr.ravel()
        )
        ptr_vec = jnp.asarray(ptr, dtype=jnp.uint32).ravel()

        def sample_one(seed_i, ptr_i):
            key = jax.random.PRNGKey(seed_i)
            key = jax.random.fold_in(key, ptr_i)
            return jax.random.normal(key, shape=(len(self.state_keys),))

        xi = jax.vmap(sample_one)(seed_vec, ptr_vec).T
        return xi

    def open_probability(self, states: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        return states[self.state_keys[self.open_state_idx]]

    def steady_state_distribution(self, v: float) -> jnp.ndarray:
        gate_ps = []
        for (_, power, _), (alpha, beta) in zip(self.gate_specs, self._gate_rates(v)):
            p = alpha / (alpha + beta + 1e-12)
            gate_ps.append((p, power))

        probs = []
        for coeff, combo in zip(self._binom_coeffs, self._state_combos):
            prob = coeff
            for (p, power), k in zip(gate_ps, combo):
                prob = prob * (p**k) * ((1 - p) ** (power - k))
            probs.append(prob)
        probs = jnp.asarray(probs)
        return probs / jnp.maximum(probs.sum(), 1e-12)

    # --- Dynamics ----------------------------------------------------
    def derivatives(self, t: float, y: jnp.ndarray, args: Tuple) -> jnp.ndarray:
        v = args[0]
        gate_rates = self._gate_rates(v)
        dydt = jnp.zeros_like(y)

        for direction, i, j, factor, g_idx in self._transitions:
            alpha, beta = gate_rates[g_idx]
            rate = factor * (alpha if direction == "alpha" else beta) * y[i]
            dydt = dydt.at[i].add(-rate)
            dydt = dydt.at[j].add(rate)
        return dydt

    def diffusion_matrix(self, y: jnp.ndarray, v: float, params: Dict[str, jnp.ndarray]):
        gate_rates = self._gate_rates(v)
        D = jnp.zeros((len(self.state_keys), len(self.state_keys)), dtype=y.dtype)
        N = params[self.count_param]

        for direction, i, j, factor, g_idx in self._transitions:
            alpha, beta = gate_rates[g_idx]
            rate = factor * (alpha if direction == "alpha" else beta) * y[i] * N
            nu = jnp.zeros_like(y).at[j].set(1.0).at[i].add(-1.0)
            D = D + rate * jnp.outer(nu, nu)

        return D / jnp.maximum(N**2, 1e-12)

    # --- Channel API -------------------------------------------------
    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: float,
        params: Dict[str, jnp.ndarray],
        xi: Optional[jnp.ndarray] = None,
    ):
        y0 = jnp.stack([states[k] for k in self.state_keys])
        y_in = self.project_simplex(y0)

        if xi is None:
            xi = self.sample_xi(states, params)
        xi = jnp.reshape(xi, y_in.shape)

        if self.solver_name in ("sde", "sde_implicit"):
            args_tuple = (self.diffusion_matrix, v, params, xi)
        else:
            args_tuple = (v,)

        y_new = self.solver_func(y_in, dt, self.derivatives, args_tuple)
        y_new = self.project_simplex(y_new)

        out = {k: val for k, val in zip(self.state_keys, y_new)}
        out[self.noise_ptr_key] = jnp.asarray(states[self.noise_ptr_key]) + 1
        return out

    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        delta_t: float,
    ):
        dist = self.steady_state_distribution(v)
        out = {k: val for k, val in zip(self.state_keys, dist)}
        out[self.noise_ptr_key] = 0
        return out
