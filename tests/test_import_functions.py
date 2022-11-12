#####################################################################
# AUTHOR        Maria Delgado-Ortet
# CONTACT       md863@cam.ac.uk
# INSTITUTION   Department of Radiology, University of Cambridge
# DATE          Nov 2022
#####################################################################
"""Unit tests for import_functions.py"""

#%% -----------------LIBRARIES--------------
import sys
sys.path.append('../') #  Adding the parent directory to PYTHONPATH so modules from subfolder utils can be imported.

from utils.import_functions import *
from pytest import raises


#%% -----------------FUNCTIONS--------------
def test_empty_tunable_parameters():
    with raises(ValueError) as exception:
        import_yaml('empty_tunable_parameters.yaml', check_tunable_parameters)

def test_type_tunable_parameters():
    with raises(TypeError) as exception:
        import_yaml('type_tunable_parameters.yaml', check_tunable_parameters)

def test_negative_tunable_parameters():
    with raises(ValueError) as exception:
        import_yaml('negative_tunable_parameters.yaml', check_tunable_parameters)

def test_missing_tunable_parameters():
    with raises(KeyError) as exception:
        import_yaml('missing_tunable_parameters.yaml', check_tunable_parameters)

def test_typo_tunable_parameters():
    with raises(KeyError) as exception:
        import_yaml('typo_tunable_parameters.yaml', check_tunable_parameters)

def test_empty_dicom_info():
    with raises(ValueError) as exception:
        import_yaml('empty_dicom_info.yaml', check_dicom_info)
