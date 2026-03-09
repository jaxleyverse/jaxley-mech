from __future__ import annotations

import math
from itertools import product
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jaxley.channels import Channel

from jaxley_mech.solvers import SolverExtension


class StatesChannel(Channel, SolverExtension):
    """Mixin for HH-style Markov chains built from independent two-state gates.

    Mathematically, this is the Fox–Lu / Goldwyn–Shea-Brown Markov SDE
    for channel fractions y(t) (Na or K) (Goldwyn & Shea-Brown 2011, Eq. (6)–(7)):

        dy = A(V) y dt + S(V, y) dW(t),

    where A(V) is the Markov generator (drift from the HH rates) and
    S@S^T = D(V, y) is the diffusion matrix from the system-size expansion.
    """

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
        shield_mask: Optional[jnp.ndarray] = None,
        state_key_fn: Optional[Callable[[Tuple[int, ...], int], str]] = None,
    ):
        """
        Args:
            name: Channel name used to prefix generated state keys.
            gate_specs: list of (gate_name, power, gate_fn) where gate_fn returns (alpha, beta).

                Each gate g has:
                    power p   ~ HH exponent (e.g. 3 for m^3)
                    gate_fn(v) -> (α_g(V), β_g(V))
                These α,β are the HH subunit transition rates; they appear in the
                Markov generator A(V) and diffusion matrix D(V,y).

            count_param: parameter name for channel count N (Fox–Lu system size).
                N is the number of channels of this type in the compartment; the
                diffusion matrix scales like 1/N (Goldwyn Eq. (12)–(13)).

            shield_mask: optional {0,1} mask to drop edge noise sources (stochastic shielding).
                Used in the edge-based noise decomposition (Pu & Thomas, Eqs. (3.4)–(3.6)).

            solver: Solver to use for ``update_states`` (``sde``, ``sde_edges``,
                ``sde_implicit``, ``explicit``, etc.).
            rtol: Relative tolerance passed to implicit and Newton-based solvers.
            atol: Absolute tolerance passed to implicit and Newton-based solvers.
            max_steps: Maximum Newton iterations for implicit solvers.
            noise_seed_param: Parameter name holding the RNG seed for sampling ``xi``.
            xi_param: Parameter name for optional precomputed noise array.
            state_key_fn: Optional function ``(combo, idx) -> str`` to override
                the default state naming convention.

        This constructor:
          • builds the Markov state space (combinations of open subunits),
          • enumerates directed edges (transitions) with appropriate combinatorial factors,
          • initializes fractions in the fully-closed state (y ≈ [1, 0, ..., 0]),
          • sets N and noise_seed defaults.
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

        # Enumerate channel configurations as tuples of (#open subunits per gate).
        # This is Fox–Lu's y(t) or x(t): fractions in each configuration
        # (Goldwyn & Shea-Brown, lines 247–251 on p.4).
        combos: List[Tuple[int, ...]] = list(
            product(*[range(p + 1) for p in self._powers])
        )
        # For Na-type m^p h: reorder so that h-open states (C*,O) come first (C3,C2,C1,O, then I3,...).
        if (
            state_key_fn is None
            and len(self._powers) == 2
            and self._powers[1] == 1
            and self._gate_names[1].lower().startswith(("h", "i"))
        ):
            combos = sorted(combos, key=lambda c: (-c[1], c[0]))
        self._state_combos = combos
        self._combo_to_idx = {combo: i for i, combo in enumerate(self._state_combos)}
        # Human-readable state names (C3,C2,C1,O,I3,... or generic s0,s1,...)
        self.state_keys = [
            state_key_fn(combo, i)
            if state_key_fn is not None
            else self._default_state_key(combo)
            for i, combo in enumerate(self._state_combos)
        ]
        self.open_state_idx = self._combo_to_idx[tuple(self._powers)]
        closed_idx = self._combo_to_idx[tuple(0 for _ in self._powers)]

        # Binomial coefficients for the steady-state distribution on Markov states,
        # matching the combinatorics of independent 2-state gates (cf. Goldwyn Eq. (10)–(13):
        # stationary fractions in each configuration are binomial in m,h,n).
        self._binom_coeffs = [
            math.prod(math.comb(power, k) for power, k in zip(self._powers, combo))
            for combo in self._state_combos
        ]

        # Precompute directed transitions between Markov states.
        # Each entry: (direction, i, j, factor, gate_idx)
        #   direction: "alpha" (opening) or "beta" (closing)
        #   i -> j: source and destination Markov states
        #   factor: number of identically-behaving subunits that can flip
        #   gate_idx: which gate (m,h,n,...) this transition uses
        #
        # These transitions define:
        #   • the generator A(V) (drift) via sums of ν_r k_r(V) y_i,
        #   • the diffusion D(V,y) via sums of ν_r ν_r^T k_r(V) y_i / N,
        # as in Fox & Lu's system-size expansion (see their K_p(Y), D_pq(Y), Eqs. (22),(24),(27)).
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
        # Optional stochastic shielding mask: drop noise on some edges (Pu & Thomas / Schmidt & Galán).
        self._shield_mask = None
        if shield_mask is not None:
            mask_array = jnp.asarray(shield_mask)
            if mask_array.shape[0] != len(transitions):
                raise ValueError(
                    f"shield_mask must have length {len(transitions)}, got {mask_array.shape[0]}"
                )
            self._shield_mask = mask_array.astype(bool)

        if self.channel_states is None:
            self.channel_states = {}
        if self.channel_params is None:
            self.channel_params = {}

        # Initialize y to fully-closed state: y_closed ≈ 1, others 0.
        for idx, key in enumerate(self.state_keys):
            self.channel_states.setdefault(key, 1.0 if idx == closed_idx else 0.0)
        self.channel_states.setdefault(self.noise_ptr_key, 0)
        # N = channel count (system size) used for 1/N diffusion scaling (Fox & Lu).
        self.channel_params.setdefault(self.count_param, 1e4)
        # Base seed for Brownian path W(t) in the SDE dy = A y dt + S dW.
        self.channel_params.setdefault(self.noise_seed_param, 0)

    # --- Helpers -----------------------------------------------------
    def _gate_rates(self, v: float) -> List[Tuple[float, float]]:
        """Evaluate HH subunit rates for each gate at membrane voltage ``v``.

        Args:
            v: Membrane voltage in the same units expected by the gate functions.

        Returns:
            A list of ``(alpha, beta)`` pairs ordered to match ``gate_specs``.
        """
        return [gate_fn(v) for _, _, gate_fn in self.gate_specs]

    def _default_state_key(self, combo: Tuple[int, ...]) -> str:
        """Build a human-readable label for a given Markov configuration.

        Args:
            combo: Tuple containing the number of open subunits for each gate.

        Returns:
            String key that names the state using HH-style conventions
            (e.g., ``C3``/``O``/``I`` for Na or ``Ck``/``O`` for K), or a
            generic ``s{idx}`` label if no pattern fits.
        """
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
        """Project raw state fractions to the probability simplex.

        Clips negative entries to zero and renormalizes so columns sum to one,
        guarding against division by zero when the input sums to ~0.

        Args:
            y: Array of state fractions shaped ``(n_states, ...)``.

        Returns:
            Array with non-negative entries that sum to one along axis 0.
        """
        y = jnp.clip(y, 0.0)
        return y / jnp.maximum(y.sum(axis=0, keepdims=True), 1e-12)

    @staticmethod
    def channels_from_density(density_per_um2: float, area_um2: float) -> float:
        """Convert a surface density into an absolute channel count.

        Args:
            density_per_um2: Channel density in channels/μm².
            area_um2: Membrane area in μm².

        Returns:
            Absolute number of channels for the given area.
        """
        return density_per_um2 * area_um2

    def sample_xi(
        self,
        states: Dict[str, jnp.ndarray],
        params: Dict[str, jnp.ndarray],
        size: Optional[int] = None,
    ):
        """Draw or retrieve Gaussian noise for the stochastic solvers.

        Args:
            states: Current channel states (used for the noise pointer).
            params: Channel parameters containing either a seed (for on-the-fly
                sampling) or a precomputed ``xi`` array.
            size: Optional override for the leading dimension of ``xi``; defaults
                to ``len(state_keys)``.

        Returns:
            Array shaped ``(size, batch?)`` containing standard normal samples.
        """
        xi_size = len(self.state_keys) if size is None else size
        ptr = jnp.asarray(states[self.noise_ptr_key])
        xi_key = self.xi_param
        if xi_key in params:
            idx = ptr % params[xi_key].shape[0]
            xi_val = params[xi_key][idx]
            if xi_val.shape[0] != xi_size:
                raise ValueError(
                    f"Expected xi[{xi_key}] with leading dimension {xi_size}, got {xi_val.shape[0]}"
                )
            return xi_val

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
            return jax.random.normal(key, shape=(xi_size,))

        xi = jax.vmap(sample_one)(seed_vec, ptr_vec).T
        return xi

    def open_probability(self, states: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Return the fraction of channels in the fully open configuration.

        Args:
            states: Mapping from state keys to fractions.

        Returns:
            Fraction corresponding to the all-open Markov state.
        """
        return states[self.state_keys[self.open_state_idx]]

    def steady_state_distribution(self, v: float) -> jnp.ndarray:
        """
        Compute binomial steady-state distribution over Markov states for fixed V.

        This is the Markov-chain analogue of Goldwyn Eq. (10)–(13): at fixed voltage,
        the fraction of channels in each configuration is binomial in the gate open
        probabilities p_g(V) = α_g / (α_g + β_g).  For example, for K (n^4):
            E[fraction open K] = n^4
            Var[fraction open K] = n^4 (1 - n^4) / N_K.

        Args:
            v: Membrane voltage where the steady state is evaluated.

        Returns:
            Probability vector over Markov states summing to one.
        """
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
    def drift(self, t: float, y: jnp.ndarray, args: Tuple) -> jnp.ndarray:
        """
        Deterministic drift: dy/dt = A(V) y.

        This is the A(V) y term in Goldwyn Eq. (6),(7), and Fox & Lu's R(Y)
        (cf. Eq. (22),(26)): for each reaction r: i→j,

            dy/dt += ν_r * k_r(V) * y_i

        where ν_r = e_j - e_i is the stoichiometric vector, and k_r(V) is an
        α or β rate times a combinatorial factor.

        Args:
            t: Time (unused, but kept for solver interface compatibility).
            y: Current state fractions.
            args: Tuple whose first element is the membrane voltage.

        Returns:
            Deterministic increment ``dy/dt`` with the same shape as ``y``.
        """
        v = args[0]
        gate_rates = self._gate_rates(v)
        dydt = jnp.zeros_like(y)

        for direction, i, j, factor, g_idx in self._transitions:
            alpha, beta = gate_rates[g_idx]
            rate = factor * (alpha if direction == "alpha" else beta) * y[i]
            # Stoichiometry ν_r = e_j - e_i
            nu = jnp.zeros_like(y).at[j].set(1.0).at[i].add(-1.0)
            dydt = dydt + rate * nu
        return dydt

    def diffusion_matrix(
        self, y: jnp.ndarray, v: float, params: Dict[str, jnp.ndarray]
    ):
        """
        Diffusion matrix D(y,V) for fractions y (Fox–Lu system-size expansion).

        From the van Kampen / Fox–Lu expansion, for fractions y = n/N:
            D(y,V) = (1/N) Σ_r k_r(V) y_{i_r} ν_r ν_r^T,
        where the sum is over reactions r (edges i→j),
        k_r(V) comes from α/β and combinatorial factors, and ν_r = e_j - e_i.

        This D(y,V) is the diffusion matrix whose square root S(y,V) appears in
        Goldwyn Eq. (6),(7): dy = A(V) y dt + S(V,y) dW(t), with S S^T = D.

        Args:
            y: Current state fractions.
            v: Membrane voltage for evaluating α/β rates.
            params: Channel parameters containing the channel count ``N``.

        Returns:
            Symmetric diffusion matrix shaped ``(n_states, n_states)``.
        """
        gate_rates = self._gate_rates(v)
        D = jnp.zeros((len(self.state_keys), len(self.state_keys)), dtype=y.dtype)
        N = params[self.count_param]

        for direction, i, j, factor, g_idx in self._transitions:
            alpha, beta = gate_rates[g_idx]
            rate = factor * (alpha if direction == "alpha" else beta) * y[i]
            nu = jnp.zeros_like(y).at[j].set(1.0).at[i].add(-1.0)
            # Add contribution rate * ν_r ν_r^T
            D += jnp.maximum(rate, 0.0) * jnp.outer(nu, nu)

        return D / jnp.maximum(N, 1e-12)

    def edge_noise_increment(
        self,
        y: jnp.ndarray,
        v: float,
        params: Dict[str, jnp.ndarray],
        xi: jnp.ndarray,  # shape = (n_transitions,)
    ) -> jnp.ndarray:
        """Compute Σ_k s_k(x) * xi_k with one noise source per directed edge.

        This is the S * xi term in Pu & Thomas eqs. (3.4)–(3.6).

        Args:
            y: Current state fractions.
            v: Membrane voltage for evaluating α/β rates.
            params: Channel parameters containing the channel count ``N``.
            xi: Noise array with one entry per directed transition.

        Returns:
            Stochastic increment matching the shape of ``y``.
        """
        gate_rates = self._gate_rates(v)
        N = params[self.count_param]

        dy_noise = jnp.zeros_like(y)

        for k, (xi_k, tr) in enumerate(zip(xi, self._transitions)):
            if self._shield_mask is not None and not self._shield_mask[k]:
                continue
            direction, i, j, factor, g_idx = tr
            alpha, beta = gate_rates[g_idx]
            rate = factor * (alpha if direction == "alpha" else beta) * y[i]
            amp = jnp.sqrt(jnp.maximum(rate, 1e-12) / N)
            nu = jnp.zeros_like(y).at[j].add(1.0).at[i].add(-1.0)
            dy_noise = dy_noise + amp * xi_k * nu

        return dy_noise

    # --- Channel API -------------------------------------------------
    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: float,
        params: Dict[str, jnp.ndarray],
        xi: Optional[jnp.ndarray] = None,
    ):
        """Advance channel fractions by one time step with chosen solver.

        Dispatches to deterministic or stochastic solvers depending on
        ``solver_name``; optionally consumes externally provided noise. Updates
        the internal noise pointer regardless of solver.

        Args:
            states: Mapping of state keys to current fractions.
            dt: Time step used by the solver.
            v: Membrane voltage during the step.
            params: Channel parameters (rates, counts, seeds, optional ``xi``).
            xi: Optional precomputed noise array; when omitted, samples via
                ``sample_xi`` using ``noise_seed_param`` and the noise pointer.

        Returns:
            New state mapping containing updated fractions and noise pointer.
        """
        y0 = jnp.stack([states[k] for k in self.state_keys])
        y_in = y0

        solver_kind = self.solver_name
        uses_noise = solver_kind in ("sde", "sde_implicit", "sde_edges")
        xi_array: Optional[jnp.ndarray] = None
        if uses_noise:
            if xi is None:
                xi_size = (
                    len(self._transitions)
                    if solver_kind == "sde_edges"
                    else len(self.state_keys)
                )
                xi = self.sample_xi(states, params, size=xi_size)
            xi_array = jnp.asarray(xi)

        if solver_kind in ("sde", "sde_implicit"):
            if xi_array is None:
                raise ValueError("Noise term required for stochastic solver.")
            xi_array = jnp.reshape(xi_array, y_in.shape)
            args_tuple = (self.diffusion_matrix, v, params, xi_array)
        elif solver_kind == "sde_edges":
            if xi_array is None:
                raise ValueError("Noise term required for stochastic solver.")
            args_tuple = (self.edge_noise_increment, v, params, xi_array)
        else:
            args_tuple = (v,)

        y_new = self.solver_func(y_in, dt, self.drift, args_tuple)

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
        """Initialize channel fractions to the voltage-dependent steady state.

        Args:
            states: Unused; included for API compatibility.
            v: Membrane voltage at which to compute the steady-state occupancy.
            params: Channel parameters; used for API compatibility.
            delta_t: Time step (unused).

        Returns:
            State mapping with steady-state fractions and a reset noise pointer.
        """
        dist = self.steady_state_distribution(v)
        out = {k: val for k, val in zip(self.state_keys, dist)}
        out[self.noise_ptr_key] = 0
        return out
