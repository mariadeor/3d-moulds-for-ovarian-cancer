#####################################################################
# AUTHOR        Maria Delgado-Ortet
# CONTACT       md863@cam.ac.uk
# INSTITUTION   Department of Radiology, University of Cambridge
# DATE          Nov 2022
#####################################################################

#%% -----------------LIBRARIES--------------
import math
import os

import numpy as np

from utils.array_modules import (
    build_label_mask,
    get_centroid,
    get_mask_slices_dicom_idx,
    rotate_label_mask,
)
from utils.dicom_modules import get_roi_masks
from utils.export_modules import create_dst_dir
from utils.import_modules import (
    build_parser,
    check_dicom_info,
    check_tunable_parameters,
    import_yaml,
)
from utils.mould_modelling_modules import build_mould
from utils.plot_tumour_outlines_modules import plot_tumour_outlines

#%% -----------------MAIN CODE--------------
#%% ----------------------------------------
# ARGUMENT PARSER
# ------------------------------------------
args = build_parser()

#%% ----------------------------------------
# 1. IMPORT INPUTS
# ------------------------------------------
print(
    "# ------------------------------------------ \n# 1. IMPORT INPUTS \n# ------------------------------------------"
)
print("\t## Importing " + args.tunable_parameters + "...", end="")
import_yaml(args.tunable_parameters, check_tunable_parameters)
print(" OK")

print("\t## Importing " + args.dicom_info + "...", end="")
import_yaml(args.dicom_info, check_dicom_info)
print(" OK")

print("\t## Extracting VOIs...", end="")
tumour_mask, _, _, _ = get_roi_masks()
original_tumour_dicom_slices = get_mask_slices_dicom_idx(
    tumour_mask
)  # Used when printing the tumour slices outlines.
print(" OK")

create_dst_dir(
    save_inputs=True
)  # Create folder to save the results and save copies of the yaml inputs used to generate the mould.
print("Import complete.")

#%% ----------------------------------------
# 2. RE-SLICING
# ------------------------------------------
print(
    "\n# ------------------------------------------ \n# 2. RE-SLICING \n# ------------------------------------------"
)
tumour_mask, base_mask, ref_point_1_mask, ref_point_2_mask = get_roi_masks(
    do_reslicing=True
)
print("Re-slicing complete.")

#%% ----------------------------------------
# 3. ROTATION
# ------------------------------------------
# Build a label mask with all the ROIs (mask:label_val –> tumour:1, ref_point_2: 2, ref_point_1: 3, base: 4):
rois_combined = build_label_mask(
    tumour_mask, ref_point_1_mask, ref_point_2_mask, base_mask
)

print(
    "\n# ------------------------------------------ \n# 3. ROTATION \n# ------------------------------------------"
)
# Find the rotation angle (theta) on the xy plane (i.e. the axial plane):
dx      = get_centroid(ref_point_1_mask)[1] - get_centroid(ref_point_2_mask)[1]
dy      = get_centroid(ref_point_1_mask)[0] - get_centroid(ref_point_2_mask)[0]
theta   = math.atan(dy / dx) * 180 / math.pi

tumour_rotated = rotate_label_mask(
    rois_combined, theta, crop_tumour=True
)  # Only the tumour is returned for increased computational speed.
print("Rotation complete.")

#%% ----------------------------------------
# 4. TRANSFORMATION TO WCS
# ------------------------------------------
print(
    "\n# ------------------------------------------ \n# 4. TRANSFORMATION TO WCS \n# ------------------------------------------"
)
# Bring the tumour to the WCS ensuring the cuts are along the cranial-caudal axis:
tumour_wcs = np.rot90(tumour_rotated, k=1, axes=(2, 0))
# Rotate the tumour in the WCS so cranial is at the first cut:
tumour_wcs = np.rot90(tumour_wcs, k=2, axes=(1, 0))
print("Transformation to WCS complete.")

#%% ----------------------------------------
# 5. TUMOUR AND MOULD MODELLING
# ------------------------------------------
print(
    "\n# ------------------------------------------ \n# 5. TUMOUR AND MOULD MODELLING \n# ------------------------------------------"
)
cavity_height, slicing_slits_positions = build_mould(tumour_wcs)
print("Mould modelling complete.")

#%% ----------------------------------------
# 6. PRINT TUMOUR SLICES OUTLINES
# ------------------------------------------
print(
    "\n# ------------------------------------------ \n# 6. PRINTING TUMOUR OUTLINES \n# ------------------------------------------"
)
plot_tumour_outlines(
    tumour_rotated, original_tumour_dicom_slices, slicing_slits_positions, cavity_height
)
print("Printing tumour slices outlines complete.")

print(
    "\n# ------------------------------------------ \n# HAPPY 3D PRINTING! \n# ------------------------------------------"
)

# Remove auxiliary files created during execution:
os.remove("inputs.py")
os.remove("outputs.py")
