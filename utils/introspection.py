import inspect


def wrap_and_override_class_defaults(cfg, lib):
    """
    Build class object and overwride default.
    Parameters
    ----------
    cfg : dict
        name : str
            Extract name of the optimizer.
        `param_name` : obj, Multiple entries
            Name(key) and default value(val) of agrs for the optimizer
            constructor.
    lib : str
        Name of the library to retrieve the class from.
    Returns
    -------
    retrieved_class : obj
        Retrieve class
    arg_str : str
        String under the form "param_name_1=param_name_1, ..., param_name_n=param_name_n"
    wrapper_args : str
        String under the form "param_name_1=param_value_1, ..., param_name_n=param_value_n"
    """
    # Retreive proper optimizer class
    class_name = cfg['name']
    exec(f"from {lib} import {class_name}")
    retrieved_class = eval(class_name)

    # Extract the lr_scheduler_class signature
    class_sig = inspect.signature(retrieved_class)
    class_params = class_sig.parameters.copy()
    param_names = class_params.keys()

    arg_str = ""
    wrapper_args = ""
    for param_name in param_names:
        # Extract class args
        arg_str += f"{param_name}={param_name}, "

        # Update wrapper args with cfg defaults
        if param_name in cfg:
            wrapper_args += f"{param_name}={cfg[param_name]}, "
        else:
            if class_params[param_name].default == inspect._empty:
                wrapper_args = f"{param_name}, {wrapper_args}"
            else:
                wrapper_args += f"{param_name}={class_params[param_name].default}, "

    # Build wrapper function with our defaults values
    exec(f"def class_wrapper({wrapper_args[:-2]}): return retrieved_class({arg_str[:-2]})", locals())
    return eval("class_wrapper")
