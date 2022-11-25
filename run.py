import argparse
import math
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from solid import *

from utils.import_functions import *
from utils.manipulate_dicom_functions import *
from utils.mould_modelling_functions import *
from utils.tumour_modelling_function import *

if __name__ == '__main__':
    #%% ----------------------------------------
    # ARGUMENT PARSER
    # ------------------------------------------
    parser = argparse.ArgumentParser(description = "Let's build a mould!")
    parser.add_argument('mould_id', type = str)
    parser.add_argument('--tunable_parameters', type = str, default = 'tunable_parameters.yaml')
    parser.add_argument('--dicom_info', type = str, default = 'dicom_info.yaml')
    parser.add_argument('--results_path', type = str, default = 'results', help = "Path to the folder where to save the results. A subfolder under the mould_id name will be created.")
    parser.add_argument('--display', action='store_true')
    
    args = parser.parse_args()
    
    #%% ----------------------------------------
    # 1. IMPORT INPUTS
    # ------------------------------------------
    print("# ------------------------------------------ \n# 1. IMPORT INPUTS \n# ------------------------------------------")

    print("\t## Importing " + args.tunable_parameters + "...", end = "")
    globals().update(import_yaml(args.tunable_parameters, check_tunable_parameters))
    print(" OK")

    print("\t## Importing " + args.dicom_info + "...", end = "")
    dicom_info_dict = import_yaml(args.dicom_info, check_dicom_info)
    print(" OK")

    print("\t## Extracting VOIs...", end = "")
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
        
    print("Saving imported yaml files to " + os.path.join(dst_dir, 'yaml_inputs') + "...", end = "")
    os.mkdir(os.path.join(dst_dir, 'yaml_inputs'))
    shutil.copyfile(args.tunable_parameters, os.path.join(dst_dir, 'yaml_inputs', 'tunable_parameters.yaml'))
    shutil.copyfile(args.dicom_info, os.path.join(dst_dir, 'yaml_inputs', 'dicom_info.yaml'))

    print(" OK")

    print("Import complete.")
    
    
    #%% ----------------------------------------
    # 2. RE-SLICING
    # ------------------------------------------
    print("# ------------------------------------------ \n# 2. RE-SLICING \n# ------------------------------------------")
    print("\t## Re-slicing VOIs to voxel size (1, 1, 1) mm...", end = "")
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

    tumour_slices = np.unique(np.argwhere(tumour_mask)[:,2])
    nbr_tumour_slices = len(tumour_slices)

    if args.display: # OPT: Display rois_combined
        print("Displaying imported VOIs...")
        for z in tumour_slices: # z goes from caudal to cranial
            curr_rois_combined_slice = rois_combined[:, :, z]

            plt.matshow(curr_rois_combined_slice)
            plt.axis('off')
            plt.title("Imported and re-sliced VOIs \nSlice " + str(z))
            
            plt.show(block=False)
            plt.pause(0.001)

    print("# ------------------------------------------ \n# 3. ROTATION \n# ------------------------------------------")
    # Find the rotation angle on the xy plane
    dx      = get_centroid(ref_point_1_mask)[1] - get_centroid(ref_point_2_mask)[1]
    dy      = get_centroid(ref_point_1_mask)[0] - get_centroid(ref_point_2_mask)[0]
    theta   = math.atan(dy/dx) * 180 / math.pi

    # Rotate each slice theta degrees
    print("\t## Rotating the tumour VOI %f degrees on the DICOM axial plane..." % theta, end = "")
    rois_combined_rotated   = np.zeros([scan_sz[0], scan_sz[1], nbr_tumour_slices])
    for idx, z in enumerate(tumour_slices):
      rois_combined_rotated[:, :, idx] = scipy.ndimage.rotate(rois_combined[:, :, z], theta, reshape=False, order=0)
    print(" OK")

    # In case the rotation left the base on top, let's add 180 degrees extra
    if get_centroid(rois_combined_rotated, 1)[0] > get_centroid(rois_combined_rotated, 3)[0]:
        print("\t## Rotating the the tumour VOI extra 180 degrees on the DICOM axial plane...", end = "")
        for z in range(nbr_tumour_slices):
            rois_combined_rotated[:, :, z] = scipy.ndimage.rotate(rois_combined_rotated[:, :, z], 180, reshape=False, order=0)
        print(" OK")

    if args.display: # OPT: Display rois_combined_rotated and habitats_rotated
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
    print("# ------------------------------------------ \n# 4. TRANSFORMATION TO WCS \n# ------------------------------------------")
    # Bring the tumour to the WCS ensuring the cuts are along the cranial-caudal axis
    tumour_wcs = np.rot90(tumour_rotated, k = 1, axes = (2, 0))

    # Rotate the tumour in the WCS so cranial is at the first cut
    tumour_wcs = np.rot90(tumour_wcs, k = 2, axes = (1, 0))

    print("Transformation to WCS complete.")
