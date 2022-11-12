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
