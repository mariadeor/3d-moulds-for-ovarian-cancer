#####################################################################
# AUTHOR        Maria Delgado-Ortet
# CONTACT       md863@cam.ac.uk
# INSTITUTION   Department of Radiology, University of Cambridge
# DATE          Nov 2022
#####################################################################

import argparse
import math
import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon2mask
from solid import (cube, hull, import_stl, linear_extrude, offset, polygon,
                   scad_render_to_file, text, translate)

from utils.import_functions import *
from utils.manipulate_dicom_functions import *
from utils.mould_modelling_functions import *
from utils.tumour_modelling_function import *

if __name__ == '__main__':
    #%% ----------------------------------------
    # ARGUMENT PARSER
    # ------------------------------------------
    parser = argparse.ArgumentParser(description="Let's build a mould!")

    parser.add_argument('mould_id', type=str,
                        help="ID for the mould to be built. The results filenames will contain it")

    parser.add_argument('--tunable_parameters', type=str,
                        default='tunable_parameters.yaml',
                        help="path to the yaml file with the tunable parameters. Specify if different to 'tunable_parameters.yaml'")

    parser.add_argument('--dicom_info', type=str,
                        default='dicom_info.yaml',
                        help="path to the yaml file with the dicom info. Specify if different to 'dicom_info.yaml'")

    parser.add_argument('--results_path', type=str,
                        default='results',
                        help="path to the folder where to save the results. A subfolder under the mould_id name will be created. Specify if different to 'results'")

    parser.add_argument('--display', action='store_true',
                        help="if present, the code displays the maks for the ROIs before and after rotation.")

    parser.add_argument('--save_preproc', action='store_true',
                        help="if present, the code saves the tumour stl mesh before smoothing.")

    parser.add_argument('--save_scad_intermediates', action='store_true',
                        help="if present, the code saves the scad files of each individual parts of the mould.")

    args = parser.parse_args()

    #%% ----------------------------------------
    # 1. IMPORT INPUTS
    # ------------------------------------------
    print("# ------------------------------------------ \n# 1. IMPORT INPUTS \n# ------------------------------------------")

    print("\t## Importing " + args.tunable_parameters + "...", end="")
    globals().update(import_yaml(args.tunable_parameters, check_tunable_parameters))
    print(" OK")

    print("\t## Importing " + args.dicom_info + "...", end="")
    dicom_info_dict = import_yaml(args.dicom_info, check_dicom_info)
    print(" OK")

    print("\t## Extracting VOIs...", end="")
    tumour_mask, base_mask, ref_point_1_mask, ref_point_2_mask = get_roi_masks(dicom_info_dict)
    print(" OK")

    path_to_results = args.results_path
    mould_id = args.mould_id
    if not os.path.isdir(path_to_results):
        os.mkdir(path_to_results)

    dst_dir = os.path.join(path_to_results, mould_id)
    try:
        os.mkdir(dst_dir)

    except FileExistsError:
        now = datetime.now()
        date_time = now.strftime('%Y%m%d_%H%M%S')
        mould_id = mould_id + "_" + date_time
        new_dst_dir = os.path.join(path_to_results, mould_id)
        print("WARNING: " + dst_dir + " already exists. Creating " + new_dst_dir + " instead.")

        dst_dir = new_dst_dir
        os.mkdir(dst_dir)

    print("Saving imported yaml files to " + os.path.join(dst_dir, 'yaml_inputs') + "...", end="")
    os.mkdir(os.path.join(dst_dir, 'yaml_inputs'))
    shutil.copyfile(args.tunable_parameters, os.path.join(dst_dir, 'yaml_inputs', 'tunable_parameters.yaml'))
    shutil.copyfile(args.dicom_info, os.path.join(dst_dir, 'yaml_inputs', 'dicom_info.yaml'))

    print(" OK")

    print("Import complete.")


    #%% ----------------------------------------
    # 2. RE-SLICING
    # ------------------------------------------
    print("\n# ------------------------------------------ \n# 2. RE-SLICING \n# ------------------------------------------")
    print("\t## Re-slicing VOIs to voxel size (1, 1, 1) mm...", end="")
    ds = pydicom.dcmread(os.path.join(dicom_info_dict['path_to_dicom'], os.listdir(dicom_info_dict['path_to_dicom'])[1]))
    scale_x = ds.PixelSpacing[0]
    scale_y = ds.PixelSpacing[1]
    scale_z = ds.SliceThickness

    tumour_mask      = reslice(tumour_mask, scale_x, scale_y, scale_z)
    base_mask        = reslice(base_mask, scale_x, scale_y, scale_z)
    ref_point_1_mask = reslice(ref_point_1_mask, scale_x, scale_y, scale_z)
    ref_point_2_mask = reslice(ref_point_2_mask, scale_x, scale_y, scale_z)
    print(" OK")
    print("\t\tOriginal voxel size: (%f, %f, %f) mm" % (scale_x, scale_y, scale_z))

    print("Re-slicing complete.")


    #%% ----------------------------------------
    # 3. ROTATION
    # ------------------------------------------
    # Create a label mask with all the ROIs
    scan_sz = np.shape(tumour_mask)

    rois_combined                    = np.zeros(scan_sz)
    rois_combined[base_mask]         = 3
    rois_combined[ref_point_1_mask]  = 2
    rois_combined[ref_point_2_mask]  = 4
    rois_combined[tumour_mask]       = 1

    tumour_slices = np.unique(np.argwhere(tumour_mask)[:, 2])
    nbr_tumour_slices = len(tumour_slices)

    if args.display:  # OPT: Display resliced rois_combined
        print("Displaying imported VOIs...")
        for z in tumour_slices:  # z goes from caudal to cranial
            curr_rois_combined_slice = rois_combined[:, :, z]

            plt.matshow(curr_rois_combined_slice)
            plt.axis('off')
            plt.title("Imported and re-sliced VOIs \nSlice " + str(z))

            plt.show(block=False)
            plt.pause(0.001)

    print("\n# ------------------------------------------ \n# 3. ROTATION \n# ------------------------------------------")
    # Find the rotation angle on the xy plane
    dx      = get_centroid(ref_point_1_mask)[1] - get_centroid(ref_point_2_mask)[1]
    dy      = get_centroid(ref_point_1_mask)[0] - get_centroid(ref_point_2_mask)[0]
    theta   = math.atan(dy/dx) * 180 / math.pi

    # Rotate each slice theta degrees
    print("\t## Rotating the tumour VOI %f degrees on the DICOM axial plane..." % theta, end="")
    rois_combined_rotated = np.zeros([scan_sz[0], scan_sz[1], nbr_tumour_slices])
    for idx, z in enumerate(tumour_slices):
        rois_combined_rotated[:, :, idx] = scipy.ndimage.rotate(rois_combined[:, :, z], theta, reshape=False, order=0)
    print(" OK")

    # In case the rotation left the base on top, let's add 180 degrees extra
    if get_centroid(rois_combined_rotated, 1)[0] > get_centroid(rois_combined_rotated, 3)[0]:
        print("\t## Rotating the the tumour VOI extra 180 degrees on the DICOM axial plane...", end="")
        for z in range(nbr_tumour_slices):
            rois_combined_rotated[:, :, z] = scipy.ndimage.rotate(rois_combined_rotated[:, :, z], 180, reshape=False, order=0)
        print(" OK")

    if args.display:  # OPT: Display rois_combined_rotated
        print("\t\tDisplaying rotated VOIs...")
        for z in range(nbr_tumour_slices):
            curr_rois_combined_slice = rois_combined_rotated[:, :, z]

            plt.matshow(curr_rois_combined_slice)
            plt.axis('off')
            plt.title("Rotated VOIs \nSlice " + str(z))

            plt.show(block=False)
            plt.pause(0.001)

    # Keep only the tumour VOI
    tumour_rotated = np.zeros([scan_sz[0], scan_sz[1], nbr_tumour_slices])
    tumour_rotated[rois_combined_rotated == 1] = 1

    # Crop the scan to the tumour VOI bounding box for increased computational speed
    rmin, rmax, cmin, cmax, zmin, zmax = get_box(tumour_rotated)
    tumour_rotated = tumour_rotated[rmin:rmax, cmin:cmax, zmin:zmax]

    print("Rotation complete.")


    #%% ----------------------------------------
    # 4. TRANSFORMATION TO WCS
    # ------------------------------------------
    print("\n# ------------------------------------------ \n# 4. TRANSFORMATION TO WCS \n# ------------------------------------------")
    # Bring the tumour to the WCS ensuring the cuts are along the cranial-caudal axis
    tumour_wcs = np.rot90(tumour_rotated, k=1, axes=(2, 0))

    # Rotate the tumour in the WCS so cranial is at the first cut
    tumour_wcs = np.rot90(tumour_wcs, k=2, axes=(1, 0))

    print("Transformation to WCS complete.")


    #%% ----------------------------------------
    # 5. TUMOUR MODELLING
    # ------------------------------------------
    print("\n# ------------------------------------------ \n# 5. TUMOUR MODELLING \n# ------------------------------------------")
    tumour_replica_filename = os.path.join(dst_dir, 'tumour_replica_' + mould_id + '.stl')
    tumour_replica_mesh = mesh_and_smooth(tumour_wcs, tumour_replica_filename, save_preproc=args.save_preproc)  # OPT: Add input save_preproc = True to save the stl of the preprocessed (prior to smoothing) mesh

    print("Tumour modelling complete.")


    #%% ----------------------------------------
    # 6. MOULD MODELLING
    # ------------------------------------------
    print("\n# ------------------------------------------ \n# 6. MOULD MODELLING \n# ------------------------------------------")

    print("\t## Ensuring the mould base will not close after the slice with the largest area...", end="")

    # Find the slice with the maximum area
    max_area = np.sum(tumour_wcs[:, :, 0])
    max_area_slice_idx = 0
    for z in range(tumour_wcs.shape[2]):
        slice_area = np.sum(tumour_wcs[:, :, z])
        if slice_area > max_area:
            max_area = slice_area
            max_area_slice_idx = z

    # Create a mask of the convex hull projection on the xy plane
    tumour_voxels = np.argwhere(tumour_wcs)
    tumour_xy_coords = tumour_voxels[:, [0, 1]]  # Keep all points x and y coordinates

    tumour_xy_convex_hull_coords = get_xy_convex_hull_coords(tumour_xy_coords)
    tumour_xy_convex_hull_mask = polygon2mask(image_shape=(tumour_wcs.shape[0], tumour_wcs.shape[1]),
                                              polygon=tumour_xy_convex_hull_coords)

    # Replace all the slices above the slice with the maximum area with the mask created above
    tumour_w_spikes = tumour_wcs.copy()  # The output is a tumour with "spiky" appearance
    for z in range(max_area_slice_idx + 1, tumour_w_spikes.shape[2]):
        tumour_w_spikes[:, :, z] = tumour_xy_convex_hull_mask
    print(" OK")

    # Convert the "spiky" tumour to stl, postprocess and save
    tumour_w_spikes_filename = os.path.join(dst_dir, 'tumour_w_spikes_' + mould_id + '.stl')
    tumour_w_spikes_mesh = mesh_and_smooth(tumour_w_spikes, tumour_w_spikes_filename, save_preproc=False)

    # ------------------------------------------
    # Build the mould base
    # ------------------------------------------
    print("\t## Building the mould base...", end="")

    # Import the hull of the "spiky" tumour
    scad_tumour_convex_hull = hull()(import_stl(os.path.split(tumour_w_spikes_filename)[-1]))

    # Find the convex hull projection on the xy plane of the "spiky" tumour
    ## The reason the hull extracted above is not used is because the stl file is centered at (0,0) and this is
    ## the reference while building the mould.
    tumour_xy_coords = np.array([tumour_w_spikes_mesh.vertices[:,0], tumour_w_spikes_mesh.vertices[:,1]]).T
    tumour_xy_convex_hull_coords = get_xy_convex_hull_coords(tumour_xy_coords)

    # Create the mould base: a block of shape of the convex hull projection and of height = mouldHeight
    tumour_sz = tumour_replica_mesh.extents
    cavity_height = cavity_height_pct * tumour_sz[2]
    scad_mould_cavity = linear_extrude(height=cavity_height)(
        offset(r=cavity_wall_thickness)(polygon(tumour_xy_convex_hull_coords))
        )

    # Carve the tumour hull inside the base
    scad_mould = scad_mould_cavity - scad_tumour_convex_hull

    if args.save_scad_intermediates:
        scad_render_to_file(scad_mould, os.path.join(dst_dir, 'mould_cavity_' + mould_id + '.scad'))

    print(" OK")

    # ------------------------------------------
    # Add plate to the mould base
    # ------------------------------------------
    print("\t## Adding plate to the the mould base...", end="")
    baseplate_xy_offset += cavity_wall_thickness
    scad_baseplate = linear_extrude(height=baseplate_height)(
        offset(r=baseplate_xy_offset)(polygon(tumour_xy_convex_hull_coords))
        )

    scad_mould = translate([0, 0, baseplate_height])(scad_mould) + scad_baseplate

    if args.save_scad_intermediates:
        scad_render_to_file(scad_mould, os.path.join(dst_dir, 'mould_cavity_w_baseplate_' + mould_id + '.scad'))

    print(" OK")

    # ------------------------------------------
    # Build the slicing guide
    # ------------------------------------------
    print("\t## Building the slicing guide...", end="")

    # Create the cutting guide main wall - It will be as with as the tumour + the mouldBase
    guides_height = tumour_sz[2] + slguide_height_offset - baseplate_height
    scad_slguide = cube([tumour_sz[0] + 2*cavity_wall_thickness, guides_thickness, guides_height])

    # Translate it: Center with respect to the cavity and upwards as the baseplate_height
    scad_slguide = translate([-(tumour_sz[0] + 2*cavity_wall_thickness)/2, (tumour_sz[1] + 2*baseplate_xy_offset)/2, baseplate_height])(scad_slguide)

    if args.save_scad_intermediates:
        scad_render_to_file(scad_slguide, os.path.join(dst_dir, 'slicing_guide_' + mould_id + '.scad'))

    # Add it to the mould structure
    scad_mould += scad_slguide

    print(" OK")

    # ------------------------------------------
    # Build the perpendicular cutting guides for the slid orientation
    # ------------------------------------------
    print("\t## Building the orientation guides...", end="")
    # Create the perpendicular cutting guide wall
    scad_orguide = cube([guides_thickness, 2*guides_thickness, guides_height])  # As it will be only two "pillars", y = 2*guideSize.

    # Place it on the left of the mould and add translate upwards as baseplate offset
    scad_orguide_left = translate([-(tumour_sz[0] + 2*baseplate_xy_offset)/2 - baseplate_xy_offset - dist_orguide_baseplate, -(2*guides_thickness)/2, baseplate_height])(scad_orguide)

    # Place it also on the right of the mould and translate upwards as the baseplate offset
    scad_orguide_right = translate([(tumour_sz[0] + 2*baseplate_xy_offset)/2 + dist_orguide_baseplate, -(2*guides_thickness)/2, baseplate_height])(scad_orguide)

    if args.save_scad_intermediates:
        scad_render_to_file(scad_orguide_left + scad_orguide_right, os.path.join(dst_dir, 'orientation_guides_' + mould_id + '.scad'))

    # Add it to the mould structure
    scad_mould += scad_orguide_left + scad_orguide_right
    print(" OK")

    # ------------------------------------------
    # Add baseplate to the slicing and orientation guides
    # ------------------------------------------
    print("\t## Adding plate to the slicing and orientation guides...", end="")
    # The baseplate for the slicing guide also need to extend along the x direction to meet the perpendicular guide baseplates.
    scad_slguide_baseplate = cube([tumour_sz[0] + 2*baseplate_xy_offset + dist_orguide_baseplate*2, guides_thickness, baseplate_height])

    # Center it
    scad_slguide_baseplate = translate([-(tumour_sz[0] + 2*baseplate_xy_offset + dist_orguide_baseplate*2)/2, (tumour_sz[1] + 2*baseplate_xy_offset)/2, 0])(scad_slguide_baseplate)

    # Add it to the mould structure
    scad_mould += scad_slguide_baseplate

    # Create the baseplate for the orientation guides
    scad_orguide_baseplate = cube([guides_thickness, tumour_sz[1] + 2*baseplate_xy_offset + guides_thickness, baseplate_height])

    # Place it on the left
    scad_orguide_baseplate_left = translate([-(tumour_sz[0] + 2*baseplate_xy_offset)/2 - guides_thickness - dist_orguide_baseplate, -(tumour_sz[1] + 2*baseplate_xy_offset)/2, 0])(scad_orguide_baseplate)
    scad_orguide_baseplate_right = translate([(tumour_sz[0] + 2*baseplate_xy_offset)/2 + dist_orguide_baseplate, -(tumour_sz[1] + 2*baseplate_xy_offset)/2, 0])(scad_orguide_baseplate)
    print(' OK')

    # Add it to the mould structure
    print("\t## Adding slicing and orientation guides to the mould...", end="")
    scad_mould += scad_orguide_baseplate_left + scad_orguide_baseplate_right

    if args.save_scad_intermediates:
        scad_render_to_file(scad_mould, os.path.join(dst_dir, 'complete_mould_no_slits_' + mould_id + '.scad'))

    print(" OK")

    # ------------------------------------------
    # Cut the mould structure
    # ------------------------------------------
    print("\t## Cutting the mould structure...", end="")

    # Create the slit structure to be carved from the mould to cut along x (slicing guide)
    scad_slicing_slit = cube([slit_thickness, guides_thickness + 2*baseplate_xy_offset + tumour_sz[1], guides_height])

    # Start cutting from the centre
    scad_slicing_slit_central = translate([-slit_thickness/2, -(2*baseplate_xy_offset + tumour_sz[1])/2, baseplate_height])(scad_slicing_slit)
    scad_mould -= scad_slicing_slit_central

    # Make the rest of the cuts
    nbr_cuts_each_half_x = math.floor(tumour_sz[0]/2/slice_thickness)
    for cut in range(nbr_cuts_each_half_x):
        slit_x_position = slice_thickness*(cut + 1)
        scad_mould -= translate([-slit_thickness/2 + slit_x_position, -(2*baseplate_xy_offset + tumour_sz[1])/2, baseplate_height])(scad_slicing_slit)  # Cuts on the left
        scad_mould -= translate([-slit_thickness/2 - slit_x_position, -(2*baseplate_xy_offset + tumour_sz[1])/2, baseplate_height])(scad_slicing_slit)  # Cuts on the right

    # Create the structure to be carved from the mould to cut along y (perpendicular cutting guide)
    scad_orientation_slit = cube([2*guides_thickness + 2*baseplate_xy_offset + tumour_sz[0] + 2*dist_orguide_baseplate, slit_thickness, guides_height])

    # Center it and position to the approrpiate z height
    scad_orientation_slit = translate([-(2*guides_thickness + 2*baseplate_xy_offset + tumour_sz[0] + 2*dist_orguide_baseplate)/2, -slice_thickness/2, cavity_height - depth_orslit])(scad_orientation_slit)

    # Cut it
    scad_mould -= scad_orientation_slit

    # Carve the letters
    font = "Liberation Sans"
    character_depth = 10
    character_size = 0.5*(slice_thickness - slit_thickness)

    start_pos = nbr_cuts_each_half_x * slice_thickness + slice_thickness/2
    nbr_cuts = nbr_cuts_each_half_x * 2
    for nbr in range(1, nbr_cuts + 1):
        scad_char = translate([-start_pos + slice_thickness*nbr, (tumour_sz[1] + 2*baseplate_xy_offset)/2 + guides_thickness/2, guides_height + baseplate_height - character_depth])(
            linear_extrude(height=character_depth)(text(str(nbr), size=character_size, font=font, halign='center', valign='center'))
        )
        scad_mould -= scad_char

    # Save the mould
    scad_render_to_file(scad_mould, os.path.join(dst_dir, 'mould_' + mould_id + '.scad'))

    print(" OK")

    print("Mould modelling complete.")
