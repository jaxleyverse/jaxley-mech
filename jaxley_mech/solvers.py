from typing import Any, Callable, Tuple

import jax
import jax.debug
import jax.numpy as jnp
import optimistix as optx
from diffrax import ImplicitEuler, ODETerm, diffeqsolve
from jax import lax
from jax.scipy.linalg import solve
from jaxley.solver_gate import save_exp


### Solver extensions
class SolverExtension:
    def __init__(
        self,
        solver: str = "newton",
        rtol: float = 1e-8,
        atol: float = 1e-8,
        max_iter: int = 4096,
    ):
        self.solver_name = solver
        self.solver_args = {"rtol": rtol, "atol": atol, "max_iter": max_iter}

        if solver == "diffrax_implicit":
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
            max_iter=self.solver_args["max_iter"],
        )

    def _diffrax_implicit_wrapper(self, y0, dt, derivatives_func, args):
        return diffrax_implicit(
            y0,
            dt,
            derivatives_func,
            args,
            self.term,
            self.diffrax_solver,
            self.solver_args["max_iter"],
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
    max_iter: int = 4096,
) -> jnp.ndarray:
    """
    Newton's method with damping for solving implicit equations.

    Parameters:
    - y0 (jnp.ndarray): Initial state vector.
    - dt (float): Time step size.
    - derivatives_func (Callable): Function that calculates derivatives of the system.
    - rtol (float): Relative tolerance for convergence. Default is 1e-8.
    - atol (float): Absolute tolerance for convergence. Default is 1e-8.
    - max_iter (int): Maximum number of iterations. Default is 4096.
    - *args: Additional arguments for the derivatives function.

    Returns:
    - jnp.ndarray: Updated state vector after solving the implicit system.
    """

    def cond_fun(loop_vars: Tuple[int, jnp.ndarray, jnp.ndarray, bool]) -> bool:
        i, _, _, converged = loop_vars
        return (i < max_iter) & ~converged

    def body_fun(
        loop_vars: Tuple[int, jnp.ndarray, jnp.ndarray, bool]
    ) -> Tuple[int, jnp.ndarray, jnp.ndarray, bool]:
        i, y, delta, converged = loop_vars
        F = _f(y, y0)
        J = jax.jacobian(_f)(y, y0)

        # Ensure J and F have compatible dimensions
        J = J.reshape((y.size, y.size))
        F = F.flatten()

        delta = solve(J, -F).reshape(y.shape)
        y = y + delta
        converged = jnp.linalg.norm(delta) < (atol + rtol * jnp.linalg.norm(y))
        return i + 1, y, delta, converged

    def _f(y: jnp.ndarray, y_prev: jnp.ndarray) -> jnp.ndarray:
        return y - y_prev - dt * derivatives_func(None, y, *args)

    i0 = jnp.array(0)
    delta0 = jnp.zeros_like(y0)
    converged0 = jnp.array(False)
    loop_vars = (i0, y0, delta0, converged0)

    _, y_new, _, _ = lax.while_loop(cond_fun, body_fun, loop_vars)

    return y_new


def rk45(
    y0: jnp.ndarray, dt: float, derivatives_func: Callable[..., jnp.ndarray], *args: Any
) -> jnp.ndarray:
    """
    Runge-Kutta 4(5) method for solving ODEs.

    Parameters:
    - y0 (jnp.ndarray): Initial state vector.
    - dt (float): Time step size.
    - derivatives_func (Callable): Function that calculates derivatives of the system.
    - *args: Additional arguments for the derivatives function.

    Returns:
    - jnp.ndarray: Updated state vector after one time step using the RK4(5) method.
    """

    def f(t: float, y: jnp.ndarray) -> jnp.ndarray:
        return derivatives_func(None, y, *args)

    k1 = f(0, y0)
    k2 = f(dt / 4, y0 + k1 * dt / 4)
    k3 = f(dt / 4, y0 + k2 * dt / 4)
    k4 = f(dt / 2, y0 + k3 * dt / 2)
    k5 = f(dt, y0 + k4 * dt)
    y_new = y0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k5)

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
        term, solver, args=args, t0=0, t1=dt, dt0=dt, y0=y0, max_steps=max_steps
    )
    return jnp.squeeze(y_new.ys, axis=0)
