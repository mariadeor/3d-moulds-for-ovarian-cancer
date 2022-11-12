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

import os

from pytest import raises
from utils.import_functions import *


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

def test_type_dicom_info():
    with raises(TypeError) as exception:
        import_yaml('type_dicom_info.yaml', check_dicom_info)

def test_path_dicom_info():
    with raises(OSError) as exception:
        import_yaml('path_dicom_info.yaml', check_dicom_info)

def test_missing_dicom_info():
    os.mkdir('tmp_path') #  To prevent OSError to be raised.
    with raises(KeyError) as exception:
        import_yaml('missing_dicom_info.yaml', check_dicom_info)
    os.rmdir('tmp_path')

def test_typo_dicom_info():
    with raises(KeyError) as exception:
        import_yaml('typo_tunable_parameters.yaml', check_tunable_parameters)
