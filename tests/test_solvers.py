import jax
import jax.numpy as jnp
import optimistix as optx
import pytest
from diffrax import ImplicitEuler, ODETerm
from jax import grad

from jaxley_mech.mechanisms.solvers import (
    diffrax_implicit,
    explicit_euler,
    newton,
    rk45,
)


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


# Add reverse-mode differentiation tests


def test_newton_reverse_mode():
    """Test reverse-mode differentiation for Newton's method."""
    y0 = jnp.array([1.0])
    dt = 0.01

    def wrapped_newton(y0):
        return jnp.sum(newton(y0, dt, simple_derivatives))

    # Check if reverse-mode differentiation works
    grad_newton = grad(wrapped_newton)(y0)

    assert jnp.isfinite(grad_newton).all(), f"Invalid gradient: {grad_newton}"


def test_explicit_euler_reverse_mode():
    """Test reverse-mode differentiation for the explicit Euler method."""
    y0 = jnp.array([1.0])
    dt = 0.01

    def wrapped_explicit_euler(y0):
        return jnp.sum(explicit_euler(y0, dt, simple_derivatives))

    # Check if reverse-mode differentiation works
    grad_explicit = grad(wrapped_explicit_euler)(y0)

    assert jnp.isfinite(grad_explicit).all(), f"Invalid gradient: {grad_explicit}"


def test_rk45_reverse_mode():
    """Test reverse-mode differentiation for the RK45 method."""
    y0 = jnp.array([1.0])
    dt = 0.01

    def wrapped_rk45(y0):
        return jnp.sum(rk45(y0, dt, simple_derivatives))

    # Check if reverse-mode differentiation works
    grad_rk45 = grad(wrapped_rk45)(y0)

    assert jnp.isfinite(grad_rk45).all(), f"Invalid gradient: {grad_rk45}"


def test_diffrax_implicit_reverse_mode():
    """Test reverse-mode differentiation for the implicit Euler method from diffrax."""
    import optimistix as optx
    from diffrax import ImplicitEuler, ODETerm
    from jax import grad

    y0 = jnp.array([1.0])  # Explicitly cast y0 to float32
    dt = 0.01  # Ensure dt is also float32

    def wrapped_diffrax_implicit(y0):
        args = ()
        term = ODETerm(simple_derivatives)
        root_finder = optx.Newton(rtol=1e-8, atol=1e-8)
        diffrax_solver = ImplicitEuler(root_finder=root_finder)
        return jnp.sum(
            diffrax_implicit(
                y0,
                dt,
                simple_derivatives,
                args,
                term=term,
                solver=diffrax_solver,
                max_steps=300,
            )
        )

    # Compute the gradient using reverse-mode differentiation
    grad_diffrax = grad(wrapped_diffrax_implicit)(y0)

    assert jnp.isfinite(grad_diffrax).all(), f"Invalid gradient: {grad_diffrax}"
