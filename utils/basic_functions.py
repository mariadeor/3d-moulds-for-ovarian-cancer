import os

import pydicom

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
