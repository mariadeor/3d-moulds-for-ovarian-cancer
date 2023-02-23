#####################################################################
# AUTHOR        Maria Delgado-Ortet
# CONTACT       md863@cam.ac.uk
# INSTITUTION   Department of Radiology, University of Cambridge
# DATE          Feb 2023
#####################################################################
"""Functions that deal with the input and output variables but do not write anything
on the auxiliary files."""

#%% -----------------LIBRARIES--------------
import os
import shutil


#%% -----------------FUNCTIONS--------------
def save_input_yaml_files():
    """
    This function saves the input yaml files to the output dst_dir.
    """

    from inputs import dicom_info, tunable_parameters
    from outputs import dst_dir

    print(
        "Saving imported yaml files to " + os.path.join(dst_dir, "yaml_inputs") + "...",
        end="",
    )
    os.mkdir(os.path.join(dst_dir, "yaml_inputs"))
    shutil.copyfile(
        tunable_parameters,
        os.path.join(dst_dir, "yaml_inputs", "tunable_parameters.yaml"),
    )
    shutil.copyfile(dicom_info, os.path.join(dst_dir, "yaml_inputs", "dicom_info.yaml"))
    print(" OK")
