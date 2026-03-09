import jax.numpy as jnp
import numpy as np
import pytest

from jaxley_mech.channels.hodgkin52 import K5States, Na8States
from tests.manual_channels.hodgkin52_manual import (
    K5StatesManual,
    Na8StatesManual,
)


def _assert_matching_channels(auto_cls, manual_cls, voltage: float, dt: float):
    channel_name = "NaK"
    auto = auto_cls(name=channel_name, solver="explicit")
    manual = manual_cls(name=channel_name, solver="explicit")

    # Ensure both channels see the exact same parameters.
    params = {k: jnp.asarray(v) for k, v in manual.channel_params.items()}

    # Start from steady state for both implementations.
    auto_state = auto.init_state({}, voltage, params, dt)
    manual_state = manual.init_state({}, voltage, params, dt)

    assert set(auto_state) == set(manual_state)
    for key in auto_state:
        np.testing.assert_allclose(auto_state[key], manual_state[key], rtol=1e-7, atol=1e-8)

    # Take one deterministic update step and compare both the states and current.
    auto_next = auto.update_states(auto_state, dt, voltage, params)
    manual_next = manual.update_states(manual_state, dt, voltage, params)

    assert set(auto_next) == set(manual_next)
    for key in auto_next:
        np.testing.assert_allclose(auto_next[key], manual_next[key], rtol=1e-6, atol=1e-6)

    auto_current = auto.compute_current(auto_next, voltage, params)
    manual_current = manual.compute_current(manual_next, voltage, params)
    np.testing.assert_allclose(auto_current, manual_current, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "auto_cls, manual_cls, voltage",
    [
        (Na8States, Na8StatesManual, -20.0),
        (K5States, K5StatesManual, -10.0),
    ],
)
def test_auto_states_match_manual(auto_cls, manual_cls, voltage):
    _assert_matching_channels(auto_cls, manual_cls, voltage=voltage, dt=0.05)
