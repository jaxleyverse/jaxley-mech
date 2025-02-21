import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from typing import List

import jaxley as jx
import numpy as np
import pytest
from jaxley.channels import HH
from jaxley.synapses import IonotropicSynapse

from jaxley_mech.synapses import AMPA, NMDA, GABAa, GABAb, GapJunction, RibbonSynapse


def test_multiparameter_setting():
    """
    Test if the correct parameters are set if one type of synapses is inserted.

    Tests global index dropping: d4daaf019596589b9430219a15f1dda0b1c34d85
    """
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=4)
    cell = jx.Cell(branch, parents=[-1])
    net = jx.Network([cell for _ in range(2)])

    pre = net.cell(0).branch(0).loc(0.0)
    post = net.cell(1).branch(0).loc(0.0)
    jx.connect(pre, post, IonotropicSynapse())

    syn_view = net.IonotropicSynapse
    syn_params = ["IonotropicSynapse_gS", "IonotropicSynapse_e_syn"]

    for p in syn_params:
        syn_view.set(p, 0.32)


def _get_synapse_view(net, synapse_name, single_idx=1, double_idxs=[2, 3]):
    """Access to the synapse view"""
    if synapse_name == "IonotropicSynapse":
        full_syn_view = net.IonotropicSynapse
        single_syn_view = net.IonotropicSynapse.edge(single_idx)
        double_syn_view = net.IonotropicSynapse.edge(double_idxs)
    if synapse_name == "AMPA":
        full_syn_view = net.AMPA
        single_syn_view = net.AMPA.edge(single_idx)
        double_syn_view = net.AMPA.edge(double_idxs)
    if synapse_name == "GABAa":
        full_syn_view = net.GABAa
        single_syn_view = net.GABAa.edge(single_idx)
        double_syn_view = net.GABAa.edge(double_idxs)
    if synapse_name == "GABAb":
        full_syn_view = net.GABAb
        single_syn_view = net.GABAb.edge(single_idx)
        double_syn_view = net.GABAb.edge(double_idxs)
    if synapse_name == "NMDA":
        full_syn_view = net.NMDA
        single_syn_view = net.NMDA.edge(single_idx)
        double_syn_view = net.NMDA.edge(double_idxs)
    if synapse_name == "RibbonSynapse":
        full_syn_view = net.RibbonSynapse
        single_syn_view = net.RibbonSynapse.edge(single_idx)
        double_syn_view = net.RibbonSynapse.edge(double_idxs)
    if synapse_name == "GapJunction":
        full_syn_view = net.GapJunction
        single_syn_view = net.GapJunction.edge(single_idx)
        double_syn_view = net.GapJunction.edge(double_idxs)
    return full_syn_view, single_syn_view, double_syn_view


@pytest.mark.parametrize(
    "synapse_type",
    [IonotropicSynapse, AMPA, GABAa, GABAb, NMDA, RibbonSynapse, GapJunction],
)
def test_set_and_querying_params_one_type(synapse_type):
    """Test if the correct parameters are set if one type of synapses is inserted."""

    synapse_class_init_params = synapse_type.__init__.__code__.co_varnames

    # If the synapse type requires a solver, instantiate it with a solver
    if "solver" in synapse_class_init_params:
        synapse_instance = synapse_type(solver="explicit")  # Specify your solver
    else:
        synapse_instance = synapse_type()

    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=4)
    cell = jx.Cell(branch, parents=[-1])
    net = jx.Network([cell for _ in range(4)])

    for pre_ind in [0, 1]:
        for post_ind in [2, 3]:
            pre = net.cell(pre_ind).branch(0).loc(0.0)
            post = net.cell(post_ind).branch(0).loc(0.0)
            jx.connect(pre, post, synapse_instance)

    # Get the synapse parameters to test setting
    syn_params = list(synapse_instance.synapse_params.keys())
    for p in syn_params:
        net.set(p, 0.15)
        assert np.all(net.edges[p].to_numpy() == 0.15)

    synapse_name = type(synapse_instance).__name__
    full_syn_view, single_syn_view, double_syn_view = _get_synapse_view(
        net, synapse_name
    )

    # There shouldn't be too many synapse_params otherwise this will take a long time
    for p in syn_params:
        full_syn_view.set(p, 0.32)
        assert np.all(net.edges[p].to_numpy() == 0.32)

        single_syn_view.set(p, 0.18)
        assert net.edges[p].to_numpy()[1] == 0.18
        assert np.all(net.edges[p].to_numpy()[np.asarray([0, 2, 3])] == 0.32)

        double_syn_view.set(p, 0.12)
        assert net.edges[p][0] == 0.32
        assert net.edges[p][1] == 0.18
        assert np.all(net.edges[p].to_numpy()[np.asarray([2, 3])] == 0.12)


@pytest.mark.parametrize(
    "synapse_type", [AMPA, GABAa, GABAb, NMDA, RibbonSynapse, GapJunction]
)
def test_set_and_querying_params_two_types(synapse_type):
    """Test whether the correct parameters are set."""

    synapse_class_init_params = synapse_type.__init__.__code__.co_varnames

    # If the synapse type requires a solver, instantiate it with a solver
    if "solver" in synapse_class_init_params:
        synapse_type = synapse_type(solver="explicit")
    else:
        synapse_type = synapse_type()

    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=4)
    cell = jx.Cell(branch, parents=[-1])
    net = jx.Network([cell for _ in range(4)])

    for pre_ind in [0, 1]:
        for post_ind, synapse in zip([2, 3], [IonotropicSynapse(), synapse_type]):
            pre = net.cell(pre_ind).branch(0).loc(0.0)
            post = net.cell(post_ind).branch(0).loc(0.0)
            jx.connect(pre, post, synapse)

    type1_params = list(IonotropicSynapse().synapse_params.keys())
    synapse_type_params = list(synapse_type.synapse_params.keys())

    default_synapse_type = net.edges[synapse_type_params[0]].to_numpy()[[1, 3]]

    net.set(type1_params[0], 0.15)
    assert np.all(net.edges[type1_params[0]].to_numpy()[[0, 2]] == 0.15)
    if synapse_type_params[0] != type1_params[0]:
        assert np.all(
            net.edges[synapse_type_params[0]].to_numpy()[[1, 3]] == default_synapse_type
        )
    else:
        default_synapse_type = 0.15

    synapse_type_name = type(synapse_type).__name__
    synapse_type_full, synapse_type_single, synapse_type_double = _get_synapse_view(
        net, synapse_type_name, double_idxs=[0, 1]
    )

    # Generalize to all parameters
    net.IonotropicSynapse.set(type1_params[0], 0.32)
    assert np.all(net.edges[type1_params[0]].to_numpy()[[0, 2]] == 0.32)
    assert np.all(
        net.edges[synapse_type_params[0]].to_numpy()[[1, 3]] == default_synapse_type
    )

    synapse_type_full.set(synapse_type_params[0], 0.18)
    assert np.all(net.edges[type1_params[0]].to_numpy()[[0, 2]] == 0.32)
    assert np.all(net.edges[synapse_type_params[0]].to_numpy()[[1, 3]] == 0.18)

    net.IonotropicSynapse.edge(1).set(type1_params[0], 0.24)
    assert net.edges[type1_params[0]][0] == 0.32
    assert net.edges[type1_params[0]][2] == 0.24
    assert np.all(net.edges[synapse_type_params[0]].to_numpy()[[1, 3]] == 0.18)

    net.IonotropicSynapse.edge([0, 1]).set(type1_params[0], 0.27)
    assert np.all(net.edges[type1_params[0]].to_numpy()[[0, 2]] == 0.27)
    assert np.all(net.edges[synapse_type_params[0]].to_numpy()[[1, 3]] == 0.18)

    synapse_type_double.set(synapse_type_params[0], 0.21)
    assert np.all(net.edges[type1_params[0]].to_numpy()[[0, 2]] == 0.27)
    assert np.all(net.edges[synapse_type_params[0]].to_numpy()[[1, 3]] == 0.21)


@pytest.mark.parametrize(
    "synapse_type", [AMPA, GABAa, GABAb, NMDA, RibbonSynapse, GapJunction]
)
def test_shuffling_order_of_set(synapse_type):
    """Test whether the result is the same if the order of synapses is changed."""
    synapse_class_init_params = synapse_type.__init__.__code__.co_varnames

    # If the synapse type requires a solver, instantiate it with a solver
    if "solver" in synapse_class_init_params:
        synapse_instance = synapse_type(solver="explicit")  # Specify your solver
    else:
        synapse_instance = synapse_type()

    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=4)
    cell = jx.Cell(branch, parents=[-1])

    net1 = jx.Network([cell for _ in range(4)])
    net2 = jx.Network([cell for _ in range(4)])

    jx.connect(
        net1.cell(0).branch(0).loc(1.0),
        net1.cell(1).branch(0).loc(0.1),
        synapse_instance,
    )
    jx.connect(
        net1.cell(1).branch(0).loc(0.6),
        net1.cell(2).branch(0).loc(0.7),
        synapse_instance,
    )

    jx.connect(
        net1.cell(2).branch(0).loc(0.4),
        net1.cell(3).branch(0).loc(0.3),
        synapse_instance,
    )

    jx.connect(
        net1.cell(3).branch(0).loc(0.1),
        net1.cell(1).branch(0).loc(0.1),
        synapse_instance,
    )

    # Different order as for `net1`.
    jx.connect(
        net2.cell(3).branch(0).loc(0.1),
        net2.cell(1).branch(0).loc(0.1),
        synapse_instance,
    )
    jx.connect(
        net2.cell(1).branch(0).loc(0.6),
        net2.cell(2).branch(0).loc(0.7),
        synapse_instance,
    )
    jx.connect(
        net2.cell(2).branch(0).loc(0.4),
        net2.cell(3).branch(0).loc(0.3),
        synapse_instance,
    )
    jx.connect(
        net2.cell(0).branch(0).loc(1.0),
        net2.cell(1).branch(0).loc(0.1),
        synapse_instance,
    )

    net1.insert(HH())
    net2.insert(HH())

    for i in range(4):
        net1.cell(i).branch(0).loc(0.0).record()
        net2.cell(i).branch(0).loc(0.0).record()

    voltages1 = jx.integrate(net1, t_max=5.0)
    voltages2 = jx.integrate(net2, t_max=5.0)

    assert np.max(np.abs(voltages1 - voltages2)) < 1e-8
