#####################################################################
# AUTHOR        Maria Delgado-Ortet
# CONTACT       md863@cam.ac.uk
# INSTITUTION   Department of Radiology, University of Cambridge
# DATE          Nov 2022 - Feb 2023
#####################################################################
"""Functions to get ROIs and metadata information from DICOM files."""

#%% -----------------LIBRARIES--------------
import os

import pydicom
from rt_utils import RTStruct, RTStructBuilder

from utils.array_modules import reslice


#%% -----------------FUNCTIONS--------------
def get_dicom_voxel_size(path_to_dicom):
    """
    This function returns the voxel size of the input DICOM series.
        INPUTS:
            path_to_dicom <str>:    Path to DICOM series.
        OUTPUTS:
            x, y, z <float>:        Voxel size.
    """

    ds = pydicom.dcmread(
        os.path.join(path_to_dicom, os.listdir(path_to_dicom)[1])
    )  # Read DICOM metadata.
    x = ds.PixelSpacing[0]
    y = ds.PixelSpacing[1]
    try:
        z = (
            ds.SpacingBetweenSlices
            if ds.SliceThickness > ds.SpacingBetweenSlices
            else ds.SliceThickness
        )
    except AttributeError:
        z = ds.SliceThickness

    return x, y, z


def get_roi_masks(do_reslicing=False):
    """
    This function returns DICOM ROIs as Numpy boolean arrays.
        INPUTS:
            do_reslicing <bool>:        Boolean flag to reslice the ROIs to (1, 1, 1) mm voxel size.
        OUTPUTS:
            roi_masks <numpy.ndarray>:  Boolean arrays of the ROIs.
    """

    from inputs import (
        base_roi_name,
        path_to_dicom,
        ref_point_1_roi_name,
        ref_point_2_roi_name,
        tumour_roi_name,
    )

    # Find the DICOM-RT
    for dcmfile in os.listdir(path_to_dicom):
        if not dcmfile.startswith("."):
            dcmfile_path = os.path.join(path_to_dicom, dcmfile)
            ds = pydicom.dcmread(dcmfile_path)
            if ds.Modality == "RTSTRUCT":
                rt_struct_path = dcmfile_path
                break

    # Get the masks
    rt_struct = RTStructBuilder.create_from(
        dicom_series_path=path_to_dicom, rt_struct_path=rt_struct_path
    )
    try:
        tumour_mask = rt_struct.get_roi_mask_by_name(tumour_roi_name)
        base_mask = rt_struct.get_roi_mask_by_name(base_roi_name)
        ref_point_1_mask = rt_struct.get_roi_mask_by_name(ref_point_1_roi_name)
        ref_point_2_mask = rt_struct.get_roi_mask_by_name(ref_point_2_roi_name)

    except RTStruct.ROIException:
        print(
            "\nERROR! Specified ROI(s) do not exist in the DICOM-RT file, which contains: ",
            rt_struct.get_roi_names(),
        )
        raise SystemExit(0)

    if do_reslicing:
        print("\t## Re-slicing VOIs to voxel size (1, 1, 1) mm...", end="")
        scale_x, scale_y, scale_z = get_dicom_voxel_size(path_to_dicom)

        tumour_mask = reslice(tumour_mask, scale_x, scale_y, scale_z)
        base_mask = reslice(base_mask, scale_x, scale_y, scale_z)
        ref_point_1_mask = reslice(ref_point_1_mask, scale_x, scale_y, scale_z)
        ref_point_2_mask = reslice(ref_point_2_mask, scale_x, scale_y, scale_z)
        print(" OK")
        print("\t\tOriginal voxel size: (%f, %f, %f) mm" % (scale_x, scale_y, scale_z))

    return tumour_mask, base_mask, ref_point_1_mask, ref_point_2_mask
