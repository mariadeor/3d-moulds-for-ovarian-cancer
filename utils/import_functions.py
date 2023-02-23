#####################################################################
# AUTHOR        Maria Delgado-Ortet
# CONTACT       md863@cam.ac.uk
# INSTITUTION   Department of Radiology, University of Cambridge
# DATE          Nov 2022
#####################################################################
"""Functions for importing and checking input yaml files."""

#%% -----------------LIBRARIES--------------
import argparse
import os
from datetime import datetime
import shutil

import yaml

from utils.write_config_functions import dump_vars_to_config

from typing import TYPE_CHECKING

#%% -----------------FUNCTIONS--------------
def build_parser():
    """
    This function builds the parser for the main code to build the 3D moulds.

        OUTPUTS:
            args <argparse.Namespace>:  Object that contains all the data in the parser.
    """

    parser = argparse.ArgumentParser(description="Let's build a mould!")

    parser.add_argument(
        "mould_id",
        type=str,
        help="ID for the mould to be built. The results filenames will contain it",
    )

    parser.add_argument(
        "--tunable_parameters",
        type=str,
        default="tunable_parameters.yaml",
        help="path to the yaml file with the tunable parameters. Specify if different to 'tunable_parameters.yaml'",
    )

    parser.add_argument(
        "--dicom_info",
        type=str,
        default="dicom_info.yaml",
        help="path to the yaml file with the dicom info. Specify if different to 'dicom_info.yaml'",
    )

    parser.add_argument(
        "--results_path",
        type=str,
        default="results",
        help="path to the folder where to save the results. A subfolder under the mould_id name will be created. Specify if different to 'results'",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="if present, the code displays the maks for the ROIs before and after rotation.",
    )

    parser.add_argument(
        "--save_preproc",
        action="store_true",
        help="if present, the code saves the tumour stl mesh before smoothing.",
    )

    parser.add_argument(
        "--save_scad_intermediates",
        action="store_true",
        help="if present, the code saves the scad files of each individual parts of the mould.",
    )

    args = parser.parse_args()

    return args


def import_yaml(path_to_file, check_func):
    """
    This function updates the global variables with those defined in a yaml
    file.
    The suitability of the variables is tested through an inputted function.
        INPUTS:
            path_to_file <str>:     Path to input yaml file.
            check_func <function>:  Function to check that the variables
                                    defined in the file are the expected
                                    and suitable.
    """

    with open(path_to_file) as yaml_file:
        yaml_dict = yaml.safe_load(yaml_file)

    check_func(yaml_dict)

    dump_vars_to_config(yaml_dict)


def check_tunable_parameters(tunable_parameters):
    """
    This function checks the yaml file containing the tunable parameters
    and raises errors if unsuitable.
        INPUTS:
            tunable_parameters <dict>:  Dictionary containing the tunable
                                        parameters defined in the input
                                        yaml file.
    """

    for key, item in tunable_parameters.items():  # Loop through each defined parameter
        if not item:  # Check if the parameter is empty
            raise ValueError(key + " is empty.")

        elif type(item) != int and type(item) != float:  # Check the parameter type
            raise TypeError(key + " is <" + type(item).__name__ + "> and should be *<int>* or *<float>*.")

        elif item < 0:  # Check if positive
            raise ValueError(key + " should be *positive*.")

    # Check if there are missing or unrecognised (extra) parameters:
    ## List with all the *expected* tunable parameters:
    required_params = [
        "slice_thickness",
        "slit_thickness",
        "guides_thickness",
        "cavity_wall_thickness",
        "dist_orguide_baseplate",
        "baseplate_height",
        "baseplate_xy_offset",
        "cavity_height_pct",
        "slguide_height_offset",
        "depth_orslit",
    ]
    ## List with all the the *inputted* parameters:
    input_params = tunable_parameters.keys()

    ## Check if there are any missing parameters
    missing_params = [param for param in required_params if param not in set(input_params)]
    if missing_params:
        raise KeyError("Missing: " + " ".join(missing_params))

    ## Check if there are any unrecognised parameters
    ## ATTN: If so, it prints a warning but it does not raise any error.
    ## In case a parameter had a typo, an error would have already been
    ## raised as it would have been identified as a missing parameter above.
    unrecognised_params = [param for param in input_params if param not in set(required_params)]
    if unrecognised_params:
        print("WARNING: Tunable parameter(s) " + " ".join(unrecognised_params) + " are unrecognised and will be ignored.")


def check_dicom_info(dicom_info):
    """
    This function checks the yaml file containing the DICOM information
    and raises errors if unsuitable.
        INPUTS:
            dicom_info <dict>:  Dictionary containing the DICOM
                                information from the input yaml file.
    """

    for key, item in dicom_info.items():
        if not item:  # Check if the specificity is empty
            raise ValueError(key + " is empty.")

        elif type(item) != str:  # Check the specificity type
            raise TypeError(key + " is <" + type(item).__name__ + "> and should be *<str>*. Adding quotes to " + key + " may resolve this issue.")

    # Check if path_to_dicom is an existing directory.
    if not os.path.isdir(dicom_info["path_to_dicom"]):
        raise OSError("path_to_dicom should be an existing *directory*.")

    # Check if there are missing or unrecognised (extra) info bits:
    ## List with all the *expected* info bits:
    required_info = [
        "path_to_dicom",
        "tumour_roi_name",
        "base_roi_name",
        "ref_point_1_roi_name",
        "ref_point_2_roi_name",
    ]
    ## List with all the the *inputted* info bits:
    input_info = dicom_info.keys()

    ## Check if there are any missing specificities
    missing_info = [info for info in required_info if info not in set(input_info)]
    if missing_info:
        raise KeyError("Missing: " + " ".join(missing_info))

    ## Check if there are any unrecognised info bits
    ## ATTN: If so, it prints a warning but it does not raise any error.
    ## In case an info bit had a typo, an error would have already been
    ## raised as it would have been identified as a missing info bit above.
    unrecognised_info = [info for info in input_info if info not in set(required_info)]
    if unrecognised_info:
        print("WARNING: DICOM information bit(s) " + " ".join(unrecognised_info) + " are unrecognised and will be ignored.")


def create_dst_dir(save_inputs=True):
    """
    This function creates path_to_results if it does not exits and the corresponding subfolder to save the results.
        INPUTS:
            save_inputs <bool>:                 Boolean to save input yaml files.
    """
    
    from config import results_path
    path_to_results = results_path
    # Create path_to_results if it does not exist:
    if not os.path.isdir(path_to_results):
        print("Creating " + path_to_results)
        os.mkdir(path_to_results)

    # Create path_to_results/mould_id subfolder:
    from config import mould_id
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
            "WARNING: " + dst_dir + " already exists. Creating " + new_dst_dir + " instead."
        )
        dst_dir = new_dst_dir
        os.mkdir(dst_dir)
    
    if save_inputs:
        save_input_yaml_files(dst_dir)
    return dst_dir


def save_input_yaml_files(dst_dir):
    """
    This function creates path_to_results/mould_id/yaml_inputs subfolder and saves copies of the yaml inputs used to generate the mould.
        INPUTS:
            dst_dir <str>:                      Path to the subfolder where to store the results.
            parser_args <argparse.Namespace>:   Object that contains all the data in the parser.
            
    """
    from config import tunable_parameters, dicom_info
    print(
        "Saving imported yaml files to " + os.path.join(dst_dir, "yaml_inputs") + "...",
        end="",
    )
    os.mkdir(os.path.join(dst_dir, "yaml_inputs"))
    shutil.copyfile(
        tunable_parameters,
        os.path.join(dst_dir, "yaml_inputs", "tunable_parameters.yaml"),
    )
    shutil.copyfile(
        dicom_info, os.path.join(dst_dir, "yaml_inputs", "dicom_info.yaml")
    )
    print(" OK")
