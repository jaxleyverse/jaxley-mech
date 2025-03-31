import importlib
import inspect
import pkgutil

import pandas as pd


def _format_lists(df):
    """Formats lists found in the dataframe."""

    def format_element(x):
        return ", ".join(map(str, x)) if isinstance(x, list) else x

    return df.map(format_element)


def find_channel(
    name=None,
    ion=None,
    species=None,
    cell_type=None,
    reference=None,
    doi=None,
    code=None,
):
    """Return channel metadata in a dataframe filtered by the argument fields."""
    all_meta = []
    jmc = importlib.import_module("jaxley_mech.mechanisms.channels")
    for _, mod_name, _ in pkgutil.walk_packages(jmc.__path__, jmc.__name__ + "."):
        module = importlib.import_module(mod_name)
        classes = inspect.getmembers(module, inspect.isclass)
        for class_name, obj in classes:
            if class_name != "Channel" and class_name != "SolverExtension":
                # Some mechs have a solver arg, some not, this might change
                init_params = obj.__init__.__code__.co_varnames
                if "solver" in init_params:
                    inst = obj(solver="explicit")
                else:
                    inst = obj()
                inst.META.pop("note", None)  # not including extra notes in df returned
                all_meta.append({"name": class_name} | inst.META)
    df = pd.DataFrame(all_meta)

    bound_args = inspect.signature(find_channel).bind(
        name=name,
        ion=ion,
        species=species,
        cell_type=cell_type,
        reference=reference,
        doi=doi,
        code=code,
    )
    filter_values = {k: v for k, v in bound_args.arguments.items() if v is not None}

    if not filter_values:
        return _format_lists(df)
    else:
        mask = pd.Series(True, index=df.index)
        for k, v in filter_values.items():
            mask &= df[k].fillna("").str.contains(v, regex=False, na=False)
        return _format_lists(df[mask])


def find_synapse(
    name=None, species=None, cell_type=None, reference=None, doi=None, code=None
):
    """Return synapse metadata in a dataframe filtered by the argument fields."""
    all_meta = []
    jmc = importlib.import_module("jaxley_mech.synapses")
    for _, mod_name, _ in pkgutil.walk_packages(jmc.__path__, jmc.__name__ + "."):
        module = importlib.import_module(mod_name)
        classes = inspect.getmembers(module, inspect.isclass)
        for class_name, obj in classes:
            if class_name != "Synapse" and class_name != "SolverExtension":
                # Some mechs have a solver arg, some not, this might change
                init_params = obj.__init__.__code__.co_varnames
                if "solver" in init_params:
                    inst = obj(solver="explicit")
                else:
                    inst = obj()
                inst.META.pop("note", None)  # not including extra notes in df returned
                all_meta.append({"name": class_name} | inst.META)
    df = pd.DataFrame(all_meta)

    bound_args = inspect.signature(find_channel).bind(
        name=name,
        species=species,
        cell_type=cell_type,
        reference=reference,
        doi=doi,
        code=code,
    )
    filter_values = {k: v for k, v in bound_args.arguments.items() if v is not None}

    if not filter_values:
        return _format_lists(df)
    else:
        mask = pd.Series(True, index=df.index)
        for k, v in filter_values.items():
            mask &= df[k].fillna("").str.contains(v, regex=False, na=False)
        return _format_lists(df[mask])
