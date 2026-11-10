"""Every mechanism in the package must simulate and differentiate under Jaxley.

The classes are discovered automatically, so a newly added channel or synapse is
tested without registering it here. This guards against API ports that leave a
mechanism unusable (wrong method signatures, renamed arguments that are still used
under their old names, or solvers that assume a different calling convention).
Mechanisms with a ``solver`` argument are tested with every supported solver.
"""

import importlib
import inspect
import pkgutil
import warnings

import jax
import jax.numpy as jnp
import jaxley as jx
import pytest
from jaxley.channels import HH, Channel
from jaxley.pumps import Pump
from jaxley.synapses import Synapse

import jaxley_mech.channels
import jaxley_mech.synapses

SOLVERS = ["explicit", "newton", "rk45", "diffrax_implicit"]
DT = 0.025
T_MAX = 3.0


def _discover(package, base):
    """Return `[(id, class)]` for all subclasses of `base` defined in `package`."""
    found = []
    for module_info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(cls, base)
                and cls.__module__ == module.__name__
                and not inspect.isabstract(cls)
            ):
                found.append((f"{module_info.name}.{name}", cls))
    return sorted(found)


def _with_solvers(classes):
    """Expand each class into one case per solver if it takes a `solver` argument."""
    cases = []
    for name, cls in classes:
        if "solver" in inspect.signature(cls.__init__).parameters:
            for solver in SOLVERS:
                cases.append(pytest.param(cls, solver, id=f"{name}[{solver}]"))
        else:
            cases.append(pytest.param(cls, None, id=name))
    return cases


def _build(cls, solver):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # E.g. warnings about surrogate gradients.
        return cls(solver=solver) if solver is not None else cls()


def _assert_simulates_and_differentiates(module, params_to_train):
    """Simulate `module`, then check that the output and its gradient are finite."""
    for key in params_to_train:
        module.make_trainable(key, verbose=False)
    params = module.get_parameters()

    def loss(params):
        return jnp.sum(jx.integrate(module, params=params, delta_t=DT))

    value, grad = jax.value_and_grad(loss)(params)

    assert jnp.isfinite(value), "Simulated voltage is not finite."
    for leaf in jax.tree_util.tree_leaves(grad):
        assert jnp.all(jnp.isfinite(leaf)), "Gradient is not finite."


@pytest.mark.parametrize(
    "cls, solver",
    _with_solvers(_discover(jaxley_mech.channels, (Channel, Pump))),
)
def test_channel_simulates_and_differentiates(cls, solver):
    channel = _build(cls, solver)
    cell = jx.Cell()
    cell.insert(channel)
    cell.record("v", verbose=False)
    cell.stimulate(jx.step_current(0.5, 1.0, 0.05, DT, T_MAX), verbose=False)
    cell.init_states()
    trainable = list(channel.channel_params)[:1]
    _assert_simulates_and_differentiates(cell, trainable)


@pytest.mark.parametrize(
    "cls, solver",
    _with_solvers(_discover(jaxley_mech.synapses, Synapse)),
)
def test_synapse_simulates_and_differentiates(cls, solver):
    synapse = _build(cls, solver)
    net = jx.Network([jx.Cell() for _ in range(2)])
    net.insert(HH())
    jx.connect(net.cell(0), net.cell(1), synapse)
    net.cell(1).record("v", verbose=False)
    net.cell(0).stimulate(jx.step_current(0.5, 1.0, 0.5, DT, T_MAX), verbose=False)
    net.init_states()
    trainable = list(synapse.synapse_params)[:1]
    _assert_simulates_and_differentiates(net, trainable)
