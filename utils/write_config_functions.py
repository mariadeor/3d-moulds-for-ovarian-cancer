def dump_vars_to_config(vars_dict, mode="a"):
    with open("config.py", mode) as f:
        for key, value in vars_dict.items():
            if isinstance(value, str):
                f.write(f'{key} = "{value}"\n')
            else:
                f.write(f'{key} = {value}\n')
