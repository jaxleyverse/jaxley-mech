import numpy as np
import pytest

from jaxley_mech.channels.fohlmeister97 import KA, Ca, K, KCa, Leak, Na


@pytest.mark.parametrize("channel_class", [Na, K, KA, Leak, Ca, KCa])
def test_init_states(channel_class):
    """Test whether, if the channels are initialized in fixed point, they do not change."""
    voltages = -65.0
    dt = 0.025

    channel = channel_class()
    init_states = channel.init_states(channel.states, voltages, channel.params, dt)

    # Deal with adding potentially missing states (which have no init).
    updated_states = channel.states
    for key, val in init_states.items():
        updated_states[key] = val
    updated_states[f"{channel.current_name}"] = 0.0

    # Add radius and length for those channels that rely on it (e.g. Ca).
    params = channel.params
    params["radius"] = 1.0
    params["length"] = 1.0

    new_states = channel.update_states(updated_states, dt, voltages, params)

    # Channel states should not have changed.
    for key in init_states.keys():
        assert np.max(np.abs(new_states[key] - init_states[key])) < 1e-8
