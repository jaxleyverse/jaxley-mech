import jax
import jax.debug
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import solve

from diffrax import ODETerm, diffeqsolve, ImplicitEuler
import optimistix as optx


def explicit_euler(y0, dt, derivatives_func, *args):
    """Explicit Euler method."""
    dydt = derivatives_func(y0, *args)
    return y0 + dydt * dt


def newton(y0, dt, derivatives_func, *args, tol=1e-6, max_iter=3000):
    """Newton's method with damping for solving implicit equations."""

    def cond_fun(loop_vars):
        i, y, delta, converged = loop_vars
        return (i < max_iter) & ~converged

    def body_fun(loop_vars):
        i, y, delta, converged = loop_vars
        F = _f(y, y0)
        J = jax.jacobian(_f)(y, y0)

        # Ensure J and F have compatible dimensions
        J = J.reshape((y.size, y.size))
        F = F.flatten()

        delta = solve(J, -F).reshape(y.shape)
        y = y + delta
        converged = jnp.linalg.norm(delta) < tol
        return i + 1, y, delta, converged

    def _f(y, y_prev):
        return y - y_prev - dt * derivatives_func(y, *args)

    i0 = jnp.array(0)
    delta0 = jnp.zeros_like(y0)
    converged0 = jnp.array(False)
    loop_vars = (i0, y0, delta0, converged0)

    _, y_final, _, _ = lax.while_loop(cond_fun, body_fun, loop_vars)

    return y_final


def rk45(y0, dt, derivatives_func, *args):
    """Runge-Kutta 4(5) method."""

    def f(t, y):
        return derivatives_func(y, *args)

    k1 = f(0, y0)
    k2 = f(dt / 4, y0 + k1 * dt / 4)
    k3 = f(dt / 4, y0 + k2 * dt / 4)
    k4 = f(dt / 2, y0 + k3 * dt / 2)
    k5 = f(dt, y0 + k4 * dt)
    y_new = y0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k5)

    return y_new


def diffrax_implicit(y0, dt, derivatives_func, args):
        """Implicit Euler method from diffrax using the Newton root-finder."""
        term = ODETerm(derivatives_func)
        root_finder = optx.Newton(rtol=1e-8, atol=1e-8)
        solver = ImplicitEuler(root_finder=root_finder)
        y_new = diffeqsolve(term, solver, args=args, t0=0, t1=dt, dt0=dt, y0=y0)
        y_new = jnp.squeeze(y_new.ys, axis=0)
        return y_new