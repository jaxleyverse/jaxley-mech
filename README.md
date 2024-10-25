# jaxley-mech

A [Jaxley](https://github.com/mackelab/jaxley)-based library of ion channels and synapses for biophysical neuron models.

## Installation

`jaxley-mech` is available on [PyPI](https://pypi.org/project/jaxley-mech/):

```bash
pip install jaxley-mech
```

or you can clone the repository and install it via `pip`'s "editable" mode:

```bash
git clone git@github.com:jaxleyverse/jaxley-mech.git
pip install -e jaxley-mech
```

## Usage

See the [notebooks](notebooks) folder for usage examples.

To view available mechanisms and filter them, it is possible to run the following code:
```python
import jaxley_mech as jm
print(jm.find_channel()) # shows metadata of the available channels
print(jm.find_channel(ion="K", species="rat")) # shows metadata of channels with these properties

all_synapses = jm.find_synapse()
print(all_synapses.reference) # shows the references of all synapses
```
