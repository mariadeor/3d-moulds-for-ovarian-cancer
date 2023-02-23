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

from utils.display_functions import (
    plot_slices,
)


#%% -----------------FUNCTIONS--------------
def get_roi_masks(do_reslicing=False):
    """
    This function returns DICOM ROIs as Numpy boolean arrays.
        INPUTS:
            do_reslicing <bool>:    Boolean flag to reslice the ROIs to (1, 1, 1) mm voxel size.
        OUTPUTS:
            roi_masks <numpy.ndarray>:  Boolean array of the ROIs.
    """
    from config import path_to_dicom, tumour_roi_name, base_roi_name, ref_point_1_roi_name, ref_point_2_roi_name

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
        tumour_mask          = rt_struct.get_roi_mask_by_name(tumour_roi_name)
        base_mask            = rt_struct.get_roi_mask_by_name(base_roi_name)
        ref_point_1_mask     = rt_struct.get_roi_mask_by_name(ref_point_1_roi_name)
        ref_point_2_mask     = rt_struct.get_roi_mask_by_name(ref_point_2_roi_name)
    
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

        print("Re-slicing complete.")

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
            val (optional):         Value of the pixels to find the centroid of.
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


def get_dicom_slices_idx(mask):
    """
    This function returns the position of the DICOM slices where the input mask is.
        INPUTS:
            mask <numpy.ndarray>:   Boolean array.
        OUTPUTS:
            dicom_slices <numpy.ndarray>:  Array with the DICOM slice indices of the mask.
    """

    dicom_slices = np.unique(np.argwhere(mask)[:, 2])
    dicom_slices = np.shape(mask)[2] - dicom_slices  # DICOM slices are numbered in reverse.
    dicom_slices = np.sort(dicom_slices)
    
    return dicom_slices


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


def build_label_mask(*masks):
    """
    This function combines different masks into a numeric label mask.
    Each mask is labelled according to its position in the input list (starting from 1).
    They are combined in reversed order, i.e. if some voxels have more than two labels, they will keep the lowest one.
        INPUTS:
            masks <numpy.ndarray>:      Boolean arrays to combine.
        OUTPUTS:
            label_mask <numpy.ndarray>: Numeric label mask as float array.
    """
    from  config import display
    
    scan_sz = np.shape(masks[0])
    label_mask = np.zeros(scan_sz)
    for idx, mask in reversed(list(enumerate(masks))):
        label_mask[mask] = idx + 1
    
    if display: # OPT: Add "--display" to the command line to display resliced rois_combined.
        print("Displaying imported VOIs... ", end="")
        plot_slices(rois_combined, tumour_slices, "Imported and re-sliced VOIs")

    return label_mask


def rotate_label_mask(label_mask, theta, **kwargs):
    """
    This function rotates the label mask a number of theta degrees on the xy plane.
        INPUTS:
            label_mask <numpy.ndarray>: Numeric label mask to rotate.
            theta <float>:              Degrees to rotate the label_mask on the xy plane.
            slice_idx_to_rotate <list>: Optional input list with the slices to rotate.
            crop_tumour <bool>:         Optional boolean input to crop the tumour. It assumes is labelled as 1.
        OUTPUTS:
            label_mask_rotated <numpy.ndarray>: Rotated input label mask.
    """
    
    if "slice_idx_to_rotate" in kwargs:
        slice_idx_to_rotate = kwargs["slice_idx_to_rotate"]
    else:
        try:
            from config import tumour_slices
            slice_idx_to_rotate = tumour_slices
        except ImportError:
            slice_idx_to_rotate = list(range(np.shape(label_mask)[2]))
    
    nbr_tumour_slices = len(slice_idx_to_rotate)
    scan_sz = np.shape(label_mask)
    label_mask_rotated = np.zeros([scan_sz[0], scan_sz[1], nbr_tumour_slices])
    
    # Rotate each slice theta degrees:
    print(
        "\t## Rotating the tumour VOI %f degrees on the DICOM axial plane..." % theta,
        end="",
    )
    for idx, z in enumerate(slice_idx_to_rotate):
        label_mask_rotated[:, :, idx] = scipy.ndimage.rotate(
            label_mask[:, :, z], theta, reshape=False, order=0
        )
    print(" OK")

    # In case the rotation resulted in the base being on top (i.e. the y
    # coordinate centroid of the base is greater than the one for the tumour),
    # let's add 180 degrees rotation extra:
    if (get_centroid(label_mask_rotated, 1)[0] > get_centroid(label_mask_rotated, 4)[0]):
        print(
            "\t## Rotating the the tumour VOI extra 180 degrees on the DICOM axial plane...",
            end="",
        )
        for z in range(nbr_tumour_slices):
            label_mask_rotated[:, :, z] = scipy.ndimage.rotate(
                label_mask_rotated[:, :, z], 180, reshape=False, order=0
            )
        print(" OK")
    
    from config import display
    if display: # OPT: Add "--display" to the command line to display the rotated VOIs.
        print("\t\tDisplaying rotated VOIs... ", end="")
        plot_slices(rois_combined_rotated, range(nbr_tumour_slices), "Rotated VOIs")
    
    if "crop_tumour" in kwargs and kwargs["crop_tumour"]:
        print(
            "\t## Cropping the tumour bounding box...",
            end="",
        )
        label_mask_rotated = crop_voi(label_mask_rotated)
        print(" OK")

    return label_mask_rotated


def crop_voi(label_mask, val=1):
    """
    This function crops the VOI of interest of a label mask.
        INPUTS:
            label_mask <numpy.ndarray>: Numeric label mask to rotate.
            val <int>:                  Value of the VOI of interest in the label_mask.
        OUTPUTS:
            voi <numpy.ndarray>:        Boolean mask of the VOI of interest
    """

    # Create a boolean array with VOI of interest only:
    label_mask_sz = np.shape(label_mask)
    voi = np.zeros(label_mask_sz)
    voi[label_mask == val] = 1

    # Crop the VOI:
    rmin, rmax, cmin, cmax, zmin, zmax = get_box(voi)
    voi = voi[rmin : rmax + 1, cmin : cmax + 1, zmin : zmax + 1]

    return voi
