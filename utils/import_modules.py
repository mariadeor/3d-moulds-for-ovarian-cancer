#####################################################################
# AUTHOR        Maria Delgado-Ortet
# CONTACT       md863@cam.ac.uk
# INSTITUTION   Department of Radiology, University of Cambridge
# DATE          Nov 2022 - Feb 2023
#####################################################################
"""Functions for importing and checking inputs from the command line and yaml files."""

#%% -----------------LIBRARIES--------------
import argparse
import os

import yaml

from utils.dump_vars_modules import dump_vars_to_file


#%% -----------------FUNCTIONS--------------
def build_parser(output_filename="inputs.py"):
    """
    This function builds the parser for the main code to build the 3D moulds and saves
    the variables to a Python script.
        INPUTS:
            output_filename <str>:  Name of the Python filename to save the
                                    imported variables.
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
    dump_vars_to_file(vars(args), "inputs.py", mode="w")
    
    return args


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
            raise TypeError(
                key
                + " is <"
                + type(item).__name__
                + "> and should be *<str>*. Adding quotes to "
                + key
                + " may resolve this issue."
            )

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
        print(
            "WARNING: DICOM information bit(s) "
            + " ".join(unrecognised_info)
            + " are unrecognised and will be ignored."
        )


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
            raise TypeError(
                key
                + " is <"
                + type(item).__name__
                + "> and should be *<int>* or *<float>*."
            )

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
    missing_params = [
        param for param in required_params if param not in set(input_params)
    ]
    if missing_params:
        raise KeyError("Missing: " + " ".join(missing_params))

    ## Check if there are any unrecognised parameters
    ## ATTN: If so, it prints a warning but it does not raise any error.
    ## In case a parameter had a typo, an error would have already been
    ## raised as it would have been identified as a missing parameter above.
    unrecognised_params = [
        param for param in input_params if param not in set(required_params)
    ]
    if unrecognised_params:
        print(
            "WARNING: Tunable parameter(s) "
            + " ".join(unrecognised_params)
            + " are unrecognised and will be ignored."
        )


def import_yaml(path_to_file, check_func, output_filename="inputs.py"):
    """
    This function saves the variables defined in a yaml file to a Python file.
    The suitability of the variables is tested through an inputted function.
        INPUTS:
            path_to_file <str>:     Path to input yaml file.
            check_func <function>:  Function to check that the variables
                                    defined in the file are the expected
                                    and suitable.
            output_filename <str>:  Name of the Python filename to save the
                                    imported variables.
    """

    with open(path_to_file) as yaml_file:
        yaml_dict = yaml.safe_load(yaml_file)

    check_func(yaml_dict)
    dump_vars_to_file(yaml_dict, output_filename)
