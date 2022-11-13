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
