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
from solid import scad_render_to_file

from utils.display_functions import (
    plot_slices,
    plot_tumour_outlines,
)
from utils.import_functions import (
    build_parser,
    check_dicom_info,
    check_tunable_parameters,
    create_dst_dir,
    import_yaml,
)
from utils.manipulate_dicom_functions import (
    build_label_mask,
    crop_voi,
    get_box,
    get_centroid,
    get_dicom_slices_idx,
    get_roi_masks,
    rotate_label_mask,
)
from utils.mould_modelling_functions import (
    get_xy_convex_hull_coords,
    make_spiky_tumour,
    build_mould_cavity,
    build_slicing_guide,
    build_orientation_guides,
    carve_slicing_slits
)
from utils.tumour_modelling_function import mesh_and_smooth

from utils.write_config_functions import dump_vars_to_config


#%% -----------------MAIN CODE--------------
#%% ----------------------------------------
# ARGUMENT PARSER
# ------------------------------------------
args = build_parser()
dump_vars_to_config(vars(args), "w")

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
original_tumour_slices = get_dicom_slices_idx(tumour_mask)  # Used when printing the tumour slices outlines.
print(" OK")

dst_dir = create_dst_dir(save_inputs=True)  # Create folder to save the results and save copies of the yaml inputs used to generate the mould.

print("Import complete.")

#%% ----------------------------------------
# 2. RE-SLICING
# ------------------------------------------
print(
    "\n# ------------------------------------------ \n# 2. RE-SLICING \n# ------------------------------------------"
)
tumour_mask, base_mask, ref_point_1_mask, ref_point_2_mask = get_roi_masks(do_reslicing=True)

#%% ----------------------------------------
# 3. ROTATION
# ------------------------------------------
# Save the dimensions of the resliced scan and the slices where the tumour is segmented:
scan_sz = np.shape(tumour_mask)
tumour_slices = np.unique(np.argwhere(tumour_mask)[:, 2])
dump_vars_to_config({
    "scan_sz":scan_sz,
    "tumour_slices":list(tumour_slices),
})

# Create a label mask with all the ROIs (mask:label_val –> tumour:1, ref_point_2: 2, ref_point_1: 3, base: 4):
rois_combined = build_label_mask(tumour_mask, ref_point_1_mask, ref_point_2_mask, base_mask)

print(
    "\n# ------------------------------------------ \n# 3. ROTATION \n# ------------------------------------------"
)
# Find the rotation angle (theta) on the xy plane (i.e. the axial plane):
dx      = get_centroid(ref_point_1_mask)[1] - get_centroid(ref_point_2_mask)[1]
dy      = get_centroid(ref_point_1_mask)[0] - get_centroid(ref_point_2_mask)[0]
theta   = math.atan(dy / dx) * 180 / math.pi

tumour_rotated = rotate_label_mask(rois_combined, theta, crop_tumour=True)  # Only the tumour, not the whole label mask, is returned for increased computational speed.

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

np.save("tumour_wcs.npy", tumour_wcs)

print("Transformation to WCS complete.")

#%% ----------------------------------------
# 5. TUMOUR MODELLING
# ------------------------------------------
print(
    "\n# ------------------------------------------ \n# 5. TUMOUR MODELLING \n# ------------------------------------------"
)
tumour_replica_filename = os.path.join(dst_dir, "tumour_replica_" + args.mould_id + ".stl")
tumour_replica_mesh = mesh_and_smooth(
    tumour_wcs, tumour_replica_filename, save_preproc=args.save_preproc
)  # OPT: Add "--save_preproc" to the command line to save the stl of the preprocessed (prior to smoothing) mesh.

print("Tumour modelling complete.")

#%% ----------------------------------------
# 6. MOULD MODELLING
# ------------------------------------------
print(
    "\n# ------------------------------------------ \n# 6. MOULD MODELLING \n# ------------------------------------------"
)

# ------------------------------------------
# Build the mould cavity
# ------------------------------------------
scad_cavity, cavity_height = build_mould_cavity(dst_dir, tumour_replica_mesh)

# ------------------------------------------
# Build the slicing guide
# ------------------------------------------
scad_slguide = build_slicing_guide(dst_dir, tumour_replica_mesh)

# ------------------------------------------
# Build the perpendicular cutting guides for the slid orientation
# ------------------------------------------
scad_orguides = build_orientation_guides(dst_dir, tumour_replica_mesh)

# ------------------------------------------
# Ensemble the structures
# ------------------------------------------
print("\t## Putting the mould together: cavity + slicing guide + orientation guides...", end="")
scad_mould = scad_cavity + scad_slguide + scad_orguides

if (
    args.save_scad_intermediates
):  # OPT: Add "--save_scad_intermediates" to the command line to save the scad file of the complete mould without slits.
    scad_render_to_file(
        scad_mould,
        os.path.join(dst_dir, "complete_mould_no_slits_" + args.mould_id + ".scad"),
    )
print(" OK")

# ------------------------------------------
# Cut the mould structure
# ------------------------------------------
scad_mould, slicing_slits_positions = carve_slicing_slits(scad_mould, tumour_replica_mesh, cavity_height)

# Save the mould
scad_render_to_file(scad_mould, os.path.join(dst_dir, "mould_" + args.mould_id + ".scad"))

print("Mould modelling complete.")

#%% ----------------------------------------
# 7. PRINT TUMOUR SLICES OUTLINES
# ------------------------------------------
print(
    "\n# ------------------------------------------ \n# 7. PRINTING TUMOUR OUTLINES \n# ------------------------------------------"
)
plot_tumour_outlines(original_tumour_slices, tumour_rotated, dst_dir, cavity_height, slicing_slits_positions)

print("Printing tumour slices outlines complete.")

print(
    "\n# ------------------------------------------ \n# HAPPY 3D PRINTING! \n# ------------------------------------------"
)
