#####################################################################
# AUTHOR        Maria Delgado-Ortet
# CONTACT       md863@cam.ac.uk
# INSTITUTION   Department of Radiology, University of Cambridge
# DATE          Feb 2023
#####################################################################
"""Functions for creating the folder with the results and saving those there."""

#%% -----------------LIBRARIES--------------
import os
from datetime import datetime

from utils.dump_vars_modules import dump_vars_to_file
from utils.in_out_modules import save_input_yaml_files


#%% -----------------FUNCTIONS--------------
def create_dst_dir(save_inputs):
    """
    This function creates path_to_results if it does not exits and the corresponding
    subfolder to save the results. The path to the subfolder is saved in a Python
    script "outputs.py".
        INPUTS:
            save_inputs <bool>: Boolean to save input yaml files.
    """

    from inputs import mould_id
    from inputs import results_path as path_to_results

    # Create path_to_results if it does not exist:
    if not os.path.isdir(path_to_results):
        print("Creating " + path_to_results)
        os.mkdir(path_to_results)

    # Create path_to_results/mould_id subfolder:
    dst_dir = os.path.join(path_to_results, mould_id)
    try:
        os.mkdir(dst_dir)
        print("Creating " + dst_dir)

    except FileExistsError:  # In case path_to_results/mould_id already exists, path_to_results/mould_id_Ymd_HMS is created with the system current date and time.
        now = datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M%S")
        mould_id = mould_id + "_" + date_time
        new_dst_dir = os.path.join(path_to_results, mould_id)
        print(
            "WARNING: "
            + dst_dir
            + " already exists. Creating "
            + new_dst_dir
            + " instead."
        )
        dst_dir = new_dst_dir
        os.mkdir(dst_dir)

    dump_vars_to_file({"dst_dir": dst_dir}, "outputs.py", mode="w")
    if save_inputs:
        save_input_yaml_files()
