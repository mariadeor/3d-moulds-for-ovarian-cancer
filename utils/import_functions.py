#####################################################################
# AUTHOR        Maria Delgado-Ortet
# CONTACT       md863@cam.ac.uk
# INSTITUTION   Department of Radiology, University of Cambridge
# DATE          Nov 2022
#####################################################################

#%% -----------------LIBRARIES--------------
import os

import yaml

#%% -----------------FUNCTIONS--------------
def import_yaml(path_to_file, check_func):
    '''
    This function updates the global variables with those defined in a yaml
    file.
    The suitability of the variables is tested through an inputted function.
        INPUTS:
            path_to_file <str>:     Path to input yaml file.
            check_func <function>:  Function to check that the variables
                                    defined in the file are the expected
                                    and suitable.

        OUTPUTS:
            yaml_dict <dict>:       Dictionary constructed from the input
                                    yaml file.
    '''

    with open(path_to_file) as yaml_file:
        yaml_dict = yaml.safe_load(yaml_file)

    check_func(yaml_dict)

    return yaml_dict


def check_tunable_parameters(tunable_parameters):
    '''
    This function checks the yaml file containing the tunable parameters
    and raises errors if unsuitable.
        INPUTS:
            tunable_parameters <dict>:  Dictionary containing the tunable
                                        parameters defined in the input
                                        yaml file.
    '''

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
                        'slice_thickness',
                        'slit_thickness',
                        'guides_thickness',
                        'cavity_wall_thickness',
                        'dist_orguide_baseplate',
                        'baseplate_height',
                        'baseplate_xy_offset',
                        'cavity_height_pct',
                        'slguide_height_offset',
                        'depth_orslit'
                        ]
    ## List with all the the *inputted* parameters:
    input_params = tunable_parameters.keys()

    ## Check if there are any missing parameters
    missing_params = [param for param in required_params if param not in set(input_params)]
    if missing_params:
        raise KeyError('Missing: ' + ' '.join(missing_params))

    ## Check if there are any unrecognised parameters
    ## ATTN: If so, it prints a warning but it does not raise any error.
    ## In case a parameter had a typo, an error would have already been
    ## raised as it would have been identified as a missing parameters above.
    unrecognised_params = [param for param in input_params if param not in set(required_params)]
    if unrecognised_params:
        print('WARNING: Tunable parameter(s) ' + ' '.join(unrecognised_params) + ' are unrecognised and will be ignored.')


def check_dicom_info(dicom_info):
    ''' 
    This function checks the yaml file containing the DICOM information and raises errors if unsuitable.
            INPUTS:
                dicom_info <dict>: Dictionary containing the DICOM information in the input yaml file.
    '''
    for key, item in dicom_info.items():
        if not item: # Check if the specificity is empty
            raise ValueError(key + " is empty.")
            
        elif type(item) != str: # Check the specificity type
            raise TypeError(key + " is <" + type(item).__name__ +  "> and should be *<str>*")
        
    if not os.path.isdir(dicom_info['path_to_dicom']): # Check if path_to_dicom is an existing directory.
        raise OSError('path_to_dicom should be an existing *directory*.')
        
    # Check if there are missing or unrecognised (extra) specificities:
    required_specs = ['path_to_dicom',
                     'tumour_roi_name', 'base_roi_name', 'ref_point_1_roi_name', 'ref_point_2_roi_name']
    input_specs = dicom_info.keys()
    
    ## Check if there are any missing specificities
    missing_specs = [spec for spec in required_specs if spec not in set(input_specs)]
    if missing_specs:
        raise KeyError('Missing: ' + ' '.join(missing_specs))
    
    ## Check if there are any unrecognised specificities 
    ## ATTN: If so, it prints a warning but it does not raise any error. In case this specificity had a typo, an error would have already been rised during the missing specificities check
    unrecognised_specs = [spec for spec in input_specs if spec not in set(required_specs)]
    if unrecognised_specs:
        print('WARNING: Tunable parameter(s) ' + ' '.join(unrecognised_specs) + ' are unrecognised and will be ignored.')
