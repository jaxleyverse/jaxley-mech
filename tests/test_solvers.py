import jax.numpy as jnp
import pytest
from jax import grad

from jaxley_mech.solvers import diffrax_implicit, explicit_euler, newton, rk45

# Just ot make sure all solvers can be run without errors. Precision is not gaurenteed.


# Define a simple system of ODEs (dy/dt = -y) with known solution.
def simple_derivatives(t, y, *args):
    return -y


def test_explicit_euler():
    """Test the explicit Euler method on a simple system."""
    y0 = jnp.array([1.0])  # Initial condition
    dt = 0.01  # Time step
    expected_solution = y0 * jnp.exp(-dt)  # Analytical solution: y(t) = y0 * exp(-t)

    y_new = explicit_euler(y0, dt, simple_derivatives)

    assert jnp.allclose(
        y_new, expected_solution, atol=1e-3
    ), f"Expected {expected_solution}, but got {y_new}"


def test_newton():
    """Test Newton's method on a simple implicit system."""
    y0 = jnp.array([1.0])
    dt = 0.01
    expected_solution = y0 * jnp.exp(-dt)

    # Use Newton's method to solve the implicit system
    y_new = newton(y0, dt, simple_derivatives)

    assert jnp.allclose(
        y_new, expected_solution, atol=1e-3
    ), f"Expected {expected_solution}, but got {y_new}"


def test_rk45():
    """Test the RK45 method on a simple system."""
    y0 = jnp.array([1.0])
    dt = 0.01
    expected_solution = y0 * jnp.exp(-dt)

    # Use RK45 method to solve the system
    y_new = rk45(y0, dt, simple_derivatives)

    assert jnp.allclose(
        y_new, expected_solution, atol=1e-3
    ), f"Expected {expected_solution}, but got {y_new}"


def test_diffrax_implicit():
    """Test the implicit Euler method from diffrax."""

    import optimistix as optx
    from diffrax import ImplicitEuler, ODETerm, diffeqsolve

    y0 = jnp.array([1.0])
    dt = 0.01
    expected_solution = y0 * jnp.exp(-dt)

    args = ()  # No extra arguments needed for this simple test

    term = ODETerm(simple_derivatives)
    root_finder = optx.Newton(rtol=1e-8, atol=1e-8)
    diffrax_solver = ImplicitEuler(root_finder=root_finder)

    # Use the implicit Euler method from diffrax
    y_new = diffrax_implicit(
        y0,
        dt,
        simple_derivatives,
        args,
        term=term,
        solver=diffrax_solver,
        max_steps=300,
    )

    assert jnp.allclose(
        y_new, expected_solution, atol=1e-3
    ), f"Expected {expected_solution}, but got {y_new}"
