#####################################################################
# AUTHOR        Maria Delgado-Ortet
# CONTACT       md863@cam.ac.uk
# INSTITUTION   Department of Radiology, University of Cambridge
# DATE          Feb 2023
#####################################################################
"""Function to copy variables from a dictionary to a Python script."""

#%% -----------------FUNCTIONS--------------
def dump_vars_to_file(vars_dict, filename, mode="a"):
    """
    This function saves the variables in the input dictionary to a Python script.
        INPUTS:
            vars_dict <dict>:   Dictionary with the variable names as keys and the
                                variable values as values.
            filename <str>:     File to save the variables to.
            mode <char>:        "w" to write a new file, "a" to append.
    """

    with open(filename, mode) as f:
        for key, value in vars_dict.items():
            if isinstance(value, str):
                f.write(f'{key} = "{value}"\n')
            else:
                f.write(f"{key} = {value}\n")
    f.close()
