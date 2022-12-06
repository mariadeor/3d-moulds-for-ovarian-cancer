#####################################################################
# AUTHOR        Maria Delgado-Ortet
# CONTACT       md863@cam.ac.uk
# INSTITUTION   Department of Radiology, University of Cambridge
# DATE          Nov 2022
#####################################################################
"""Functions for manipulate DICOM files and the masks that derive
from their ROIs."""

#%% -----------------LIBRARIES--------------
import os

import numpy as np
import pydicom
import scipy.ndimage
from rt_utils import RTStruct, RTStructBuilder


#%% -----------------FUNCTIONS--------------
def get_roi_masks(dicom_info_dict):
    """
    This function returns DICOM ROIs as Numpy boolean arrays.
        INPUTS:
            dicom_info_dict <dict>: Dictionary of the DICOM information
                                    from the main function input yaml
                                    file.
        OUTPUTS:
            roi_masks <numpy.ndarray>:  Boolean array of the ROIs.
    """

    path_to_dicom = dicom_info_dict["path_to_dicom"]
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
                  dicom_series_path=path_to_dicom,
                  rt_struct_path=rt_struct_path
                )
    try:
        tumour_mask          = rt_struct.get_roi_mask_by_name(dicom_info_dict["tumour_roi_name"])
        base_mask            = rt_struct.get_roi_mask_by_name(dicom_info_dict["base_roi_name"])
        ref_point_1_mask     = rt_struct.get_roi_mask_by_name(dicom_info_dict["ref_point_1_roi_name"])
        ref_point_2_mask     = rt_struct.get_roi_mask_by_name(dicom_info_dict["ref_point_2_roi_name"])
    
    except RTStruct.ROIException:
        print(
            "\nERROR! Specified ROI(s) do not exist in the DICOM-RT file, which contains: ",
            rt_struct.get_roi_names(),
        )
        raise SystemExit(0)

    return tumour_mask, base_mask, ref_point_1_mask, ref_point_2_mask


def reslice(array, scale_x, scale_y, scale_z):
    """
    This function reslices and interpolates the input array to
    (1, 1, 1) mm voxel size using zero-order spline interpolation.
        INPUTS:
            array <numpy.ndarray>:  Array to reslice and interpolate.
            scale_x, scale_y, scale_z <float>:  Original DICOM images
                                                voxel size in mm.
        OUTPUTS:
            array <numpy.ndarray>:  Resliced and interpolated input
                                    array to (1, 1, 1) mm voxel size.
    """

    return scipy.ndimage.zoom(
        array, (scale_x, scale_y, scale_z), order=0, prefilter=False
    )


def get_centroid(mask, val=True):
    """
    This function returns the centroid of the pixels equal to a specified
    value.
    If a value is not specified, the function assumes by default that it
    is a boolean array and looks for the centroid of the True pixels.
        INPUTS:
            mask <numpy.ndarray>:   Array.
            val (optional): Value of the pixels to find the centroid of.
        OUTPUTS:
            y, x, z <float>: Centroid coordinates.
    """

    idx = np.where(mask == val)
    return np.array([np.mean(idx[0]), np.mean(idx[1]), np.mean(idx[2])])


def get_box(mask):
    """
    This function returns the bounding box of the True pixels of a boolean
    array.
        INPUTS:
            mask <numpy.ndarray>:   Boolean array.
        OUTPUTS:
            rmin, rmax, cmin, cmax, zmin, zmax <int>:   Bounding box limits
                                                        (y, x, z).
    """

    r = np.any(mask, axis=(1, 2))
    c = np.any(mask, axis=(0, 2))
    z = np.any(mask, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax
