import importlib
import pkgutil
import inspect
import pandas as pd
import numpy as np

def find_channel(name=None, ion=None, species=None, cell_type=None, reference=None, doi=None, code=None):
    # Import all the channel metadata files, or should do the filtering in this step (maybe would take longer)?
    all_meta = []
    jmc = importlib.import_module("jaxley_mech.channels")
    for _, mod_name, _ in pkgutil.walk_packages(jmc.__path__, jmc.__name__ + "."):
        module = importlib.import_module(mod_name)
        classes = inspect.getmembers(module, inspect.isclass)
        for class_name, obj in classes:
            if class_name != "Channel" and class_name != "SolverExtension":
                init_params = obj.__init__.__code__.co_varnames
                if "solver" in init_params:
                    inst = obj(solver="explicit")
                else:
                    inst = obj()
                inst.META.pop("note", None)
                all_meta.append({"name": class_name} | inst.META)
    df = pd.DataFrame(all_meta)
    
    # Scan based on args
    bound_args = inspect.signature(find_channel).bind(
        name=name, 
        ion=ion, 
        species=species, 
        cell_type=cell_type, 
        reference=reference, 
        doi=doi, 
        code=code)
    filter_values = {k: v for k, v in bound_args.arguments.items() if v is not None}

    query_list = [f'{k}=="{v}"' for k, v in filter_values.items()]
    query = ' & '.join(query_list)
    view = df.query(query)

    # Return dataframe with metadata of wanted channels
    return view

def find_synapse():
    # Import all the synapse metadata files
    # Scan based on args
    # Return paths of wanted synapses
    pass


if __name__ == "__main__":
    data = find_channel(ion="K", species="Cat") 
    print(data)

    """
    TODO
    - deal with meta data that can't be queried
    """