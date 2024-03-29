#####################################################################
# AUTHOR        Maria Delgado-Ortet
# CONTACT       md863@cam.ac.uk
# INSTITUTION   Department of Radiology, University of Cambridge
# DATE          Nov 2022 - Feb 2023
#####################################################################
"""Function to generate the slices with the tumour outlines."""

#%% -----------------LIBRARIES--------------
import math
import os

import matplotlib as mpl

mpl.use("tkagg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from utils.dicom_modules import get_dicom_voxel_size


#%% -----------------FUNCTIONS--------------
def plot_tumour_outlines(
    tumour_rotated, original_tumour_dicom_slices, slicing_slits_positions, cavity_height
):
    """
    This function is used to generate the slices with the tumour outlines.
    INPUTS:
        tumour_rotated <np.ndarray>:    Tumour rotated in the DICOM space.
        dicom_slices <numpy.ndarray>:   Array with the original DICOM slice indices
                                        of the tumour.
        slicing_slits_positions <list>: List with the positions where the slits have
                                        been placed in the World Coordinate
                                        System (WCS).
    """

    from inputs import depth_orslit, path_to_dicom, slice_thickness
    from outputs import dst_dir

    print("\t## Matching the closest DICOM slice to each mould tumour slice...", end="")
    _, _, scale_z = get_dicom_voxel_size(path_to_dicom)
    original_central_slice_idx = math.floor(
        np.shape(original_tumour_dicom_slices)[0] / 2
    )  # Find the central original DICOM slice.
    slices_sampling = math.floor(
        slice_thickness / scale_z
    )  # Find the relationship between tumour slice thickness and DICOM slice thickness.
    original_tumour_slices_sampled = np.union1d(
        original_tumour_dicom_slices[original_central_slice_idx::slices_sampling],
        original_tumour_dicom_slices[original_central_slice_idx::-slices_sampling],
    )  # Sample the DICOM slices from the center.
    print(" OK")
    print(
        "\t\tWARNING: Please visually assess the DICOM-tissue slice matching as an offset may be expected."
    )

    print("\t## Marking the base and the orientation incision...", end="")
    tumour_outlines = tumour_rotated.copy()

    cmap = ListedColormap(
        ["None", "cyan", "blue", "red"]
    )  # Create a custom colourmap (pixels equal to 0 will have "None" value, pixels equal to 1 will be "cyan"...).

    tumour_outlines[
        tumour_outlines.shape[0] - 1, :, :
    ] = 2  # Marking of the base in blue (2).

    orientation_incision_position = round(tumour_outlines.shape[1] / 2) - 1
    tumour_outlines[
        :, orientation_incision_position, :
    ] = 2  # Marking of the orientation incision position in blue (2).
    tumour_outlines[
        0 : round(tumour_outlines.shape[0] - cavity_height + depth_orslit),
        orientation_incision_position,
        :,
    ] = 3  # Marking of the orientation incision cut depth in red (3).
    print(" OK")

    print("\t## Plotting and saving the tumour slices outlines...", end="")
    # Plot and save the outlines
    outlines_dst_dir = os.path.join(dst_dir, "tumour_slices_outlines")
    os.mkdir(outlines_dst_dir)

    cm = 1 / 2.54  # Centimeters to inches
    figsize = (
        (tumour_outlines.shape[1] * cm / 10),
        (tumour_outlines.shape[0] * cm / 10),
    )  # The outlines are scaled to the expected tumour slices size in the real world.
    for idx, x in enumerate(slicing_slits_positions):
        x += tumour_rotated.shape[2] / 2
        curr_slice = tumour_outlines[:, :, round(x)]

        matfig = plt.figure(figsize=figsize)
        plt.matshow(curr_slice, cmap=cmap, aspect="auto", fignum=matfig.number)
        plt.axis("off")

        plt.savefig(
            os.path.join(
                outlines_dst_dir,
                "Slice_"
                + str(matfig.number)
                + "_DICOM_"
                + str(original_tumour_slices_sampled[idx])
                + ".png",
            ),
            transparent=True,
        )
        plt.show(block=False)
        plt.pause(0.001)
    print(" OK")

    input("INPUT REQUIRED! Please hit enter to close all figures and continue.")
    plt.close("all")
