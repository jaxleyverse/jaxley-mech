from typing import Any, Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.debug
import jax.numpy as jnp
import optimistix as optx
from diffrax import ImplicitEuler, ODETerm, diffeqsolve
from jax import lax
from jax.scipy.linalg import solve
from jaxley.mechanisms.solvers import save_exp


### Solver extensions
class SolverExtension:
    def __init__(
        self,
        solver: Optional[str] = None,
        rtol: float = 1e-8,
        atol: float = 1e-8,
        max_steps: int = 10,
        verbose=False,
    ):
        self.solver_name = solver
        self.solver_args = {"rtol": rtol, "atol": atol, "max_steps": max_steps}

        if solver is None:
            raise ValueError(
                "Solver must be specified (`newton`, `explicit`, `rk45` and `diffrax_implicit`)."
            )
        elif solver == "diffrax_implicit":
            self.term = ODETerm(self.derivatives)
            root_finder = optx.Newton(rtol=rtol, atol=atol)
            self.diffrax_solver = ImplicitEuler(root_finder=root_finder)

        self.solver_func = self._get_solver_func(solver)

    def __getstate__(self):
        # Return the state without the solver function reference
        state = self.__dict__.copy()
        state["solver_func"] = (
            self.solver_name
        )  # Store the solver name instead of function reference
        return state

    def __setstate__(self, state):
        # Restore the state and reinitialize the solver function
        self.__dict__.update(state)
        self.solver_func = self._get_solver_func(state["solver_func"])

    def _get_solver_func(self, solver):
        solvers = {
            "newton": self._newton_wrapper,
            "rk45": rk45,
            "explicit": explicit_euler,
            "diffrax_implicit": self._diffrax_implicit_wrapper,
        }
        if solver not in solvers:
            raise ValueError(
                f"Solver {solver} not recognized. Currently supported solvers are: {list(solvers.keys())}"
            )
        return solvers[solver]

    def _newton_wrapper(self, y0, dt, derivatives_func, args):
        return newton(
            y0,
            dt,
            derivatives_func,
            args,
            rtol=self.solver_args["rtol"],
            atol=self.solver_args["atol"],
            max_steps=self.solver_args["max_steps"],
        )

    def _diffrax_implicit_wrapper(self, y0, dt, derivatives_func, args):
        return diffrax_implicit(
            y0=y0,
            dt=dt,
            derivatives_func=derivatives_func,
            args=args,
            term=self.term,
            solver=self.diffrax_solver,
            max_steps=self.solver_args["max_steps"],
        )


### Solvers


def explicit_euler(
    y0: jnp.ndarray, dt: float, derivatives_func: Callable[..., jnp.ndarray], *args: Any
) -> jnp.ndarray:
    """
    Explicit Euler method for solving ODEs.

    Parameters:
    - y0 (jnp.ndarray): Initial state vector.
    - dt (float): Time step size.
    - derivatives_func (Callable): Function that calculates derivatives of the system.
    - *args: Additional arguments for the derivatives function.

    Returns:
    - jnp.ndarray: Updated state vector after one time step.
    """
    dydt = derivatives_func(None, y0, *args)
    return y0 + dydt * dt


def newton(
    y0: jnp.ndarray,
    dt: float,
    derivatives_func: Callable[..., jnp.ndarray],
    *args: Any,
    rtol: float = 1e-8,  # Relative tolerance
    atol: float = 1e-8,  # Absolute tolerance
    max_steps: int = 10,
) -> jnp.ndarray:
    """
    Newton's method for solving implicit equations with early stopping using equinox's while_loop.

    Parameters:
    - y0 (jnp.ndarray): Initial state vector.
    - dt (float): Time step size.
    - derivatives_func (Callable): Function that calculates derivatives of the system.
    - rtol (float): Relative tolerance for convergence.
    - atol (float): Absolute tolerance for convergence.
    - max_steps (int): Maximum number of iterations.

    Returns:
    - jnp.ndarray: Updated state vector after solving the implicit system.
    """

    def _f(y: jnp.ndarray, y_prev: jnp.ndarray) -> jnp.ndarray:
        return y - y_prev - dt * derivatives_func(None, y, *args)

    def body_fun(
        carry: Tuple[jnp.ndarray, jnp.ndarray, int, bool]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int, bool]:
        y, y_prev, i, _ = carry

        F = _f(y, y0)
        J = jax.jacobian(_f)(y, y0)

        # Ensure J and F have compatible dimensions
        J = J.reshape((y.size, y.size))
        F = F.flatten()

        delta = solve(J, -F).reshape(y.shape)
        y = y + delta

        # Check for convergence
        converged = jnp.linalg.norm(delta) < (atol + rtol * jnp.linalg.norm(y))

        # Increment iteration count
        i += 1

        # Debugging
        # jax.debug.print("Iteration {}: Converged = {}", i, converged)

        return y, y_prev, i, converged

    def cond_fun(carry: Tuple[jnp.ndarray, jnp.ndarray, int, bool]) -> bool:
        _, _, i, converged = carry
        return (i < max_steps) & ~converged

    # Initial carry
    init_val = (y0, y0, 0, False)

    # Run the equinox while_loop with early stopping, setting kind to 'checkpointed' to allow reverse-mode differentiation
    final_carry = eqx.internal.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=init_val,
        max_steps=max_steps,
        kind="checkpointed",
    )

    y_new, _, _, _ = final_carry
    return y_new


def rk45(
    y0: jnp.ndarray, dt: float, derivatives_func: Callable[..., jnp.ndarray], *args: Any
) -> jnp.ndarray:
    """
    Non-adaptive Runge-Kutta 4(5) method for solving ODEs.

    Parameters:
    - y0 (jnp.ndarray): Initial state vector.
    - dt (float): Time step size.
    - derivatives_func (Callable): Function that calculates derivatives of the system.
    - *args: Additional arguments for the derivatives function.

    Returns:
    - jnp.ndarray: Updated state vector after one time step using the RK4(5) method.
    """

    def f(y: jnp.ndarray) -> jnp.ndarray:
        return derivatives_func(None, y, *args)

    # Coefficients for the RK45 method
    a2 = 1 / 4
    a3 = [3 / 32, 9 / 32]
    a4 = [1932 / 2197, -7200 / 2197, 7296 / 2197]
    a5 = [439 / 216, -8, 3680 / 513, -845 / 4104]
    a6 = [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40]

    # b4 = [25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5]  # 4th-order
    b5 = [16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]  # 5th-order

    # Compute the RK45 stages
    k1 = f(y0)
    k2 = f(y0 + a2 * dt * k1)
    k3 = f(y0 + dt * (a3[0] * k1 + a3[1] * k2))
    k4 = f(y0 + dt * (a4[0] * k1 + a4[1] * k2 + a4[2] * k3))
    k5 = f(y0 + dt * (a5[0] * k1 + a5[1] * k2 + a5[2] * k3 + a5[3] * k4))
    k6 = f(y0 + dt * (a6[0] * k1 + a6[1] * k2 + a6[2] * k3 + a6[3] * k4 + a6[4] * k5))

    y_new = y0 + dt * (b5[0] * k1 + b5[2] * k3 + b5[3] * k4 + b5[4] * k5 + b5[5] * k6)

    return y_new


def diffrax_implicit(
    y0: jnp.ndarray,
    dt: float,
    derivatives_func: Callable[..., jnp.ndarray],
    args: Tuple,
    term: ODETerm,
    solver: ImplicitEuler,
    max_steps: int,
) -> jnp.ndarray:
    """
    Implicit Euler method using diffrax's Newton root-finder for solving ODEs.

    Parameters:
    - y0 (jnp.ndarray): Initial state vector.
    - dt (float): Time step size.
    - derivatives_func (Callable): Function that calculates derivatives of the system.
    - args (Tuple): Additional arguments for the derivatives function.
    - term (ODETerm): Pre-initialized ODETerm object.
    - solver (ImplicitEuler): Pre-initialized ImplicitEuler solver.
    - max_steps (int): Maximum number of steps.

    Returns:
    - jnp.ndarray: Updated state vector after one time step using the Implicit Euler method.
    """
    y_new = diffeqsolve(
        term, solver, args=args, t0=0.0, t1=dt, dt0=dt, y0=y0, max_steps=max_steps
    )
    return jnp.squeeze(y_new.ys, axis=0)
