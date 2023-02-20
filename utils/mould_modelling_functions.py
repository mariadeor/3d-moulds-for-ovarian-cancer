#####################################################################
# AUTHOR        Maria Delgado-Ortet
# CONTACT       md863@cam.ac.uk
# INSTITUTION   Department of Radiology, University of Cambridge
# DATE          Nov 2022
#####################################################################
"""This file contains the function to extract the convex hull coordinates used
during the mould modelling step."""

#%% -----------------LIBRARIES--------------
import os

import numpy as np
from scipy.spatial import ConvexHull
from skimage.draw import polygon2mask

#%% -----------------FUNCTIONS--------------
def get_xy_convex_hull_coords(xy_coords):
    """
    This function returns the coordinates of the convex hull of the input
    points list.
        INPUTS:
            xy_coords <numpy.ndarray>:  (N, 2) array containing the
                                            coordinates of all the points.
        OUTPUTS:
            xy_convex_hull_cords <numpy.ndarray>:   (N, 2) array containing
                                                    the coordinates of the
                                                    points of the convex hull.
    """

    xy_convex_hull = ConvexHull(xy_coords, incremental=True)
    xy_convex_hull_coords_idx = xy_convex_hull.vertices

    return xy_coords[xy_convex_hull_coords_idx]


def make_spiky_tumour(tumour_wcs):
    # Find the slice with the maximum area:
    max_area = np.sum(tumour_wcs[:, :, 0])
    max_area_slice_idx = 0
    for z in range(tumour_wcs.shape[2]):
        slice_area = np.sum(tumour_wcs[:, :, z])
        if slice_area > max_area:
            max_area = slice_area
            max_area_slice_idx = z

    # Create a mask of the convex hull projection on the xy plane:
    tumour_voxels = np.argwhere(tumour_wcs)
    tumour_xy_coords = tumour_voxels[:, [0, 1]]  # Keep all points x and y coordinates
    tumour_xy_convex_hull_coords = get_xy_convex_hull_coords(tumour_xy_coords)
    tumour_xy_convex_hull_mask = polygon2mask(
        image_shape=(tumour_wcs.shape[0], tumour_wcs.shape[1]),
        polygon=tumour_xy_convex_hull_coords,
    )

    # Replace all the slices above the slice with the maximum area with the mask created above:
    tumour_w_spikes = tumour_wcs.copy() # The output is a tumour with "spiky" appearance, here the reason of the variable name.
    for z in range(max_area_slice_idx + 1, tumour_w_spikes.shape[2]):
        tumour_w_spikes[:, :, z] = tumour_xy_convex_hull_mask
    print(" OK")

    return tumour_w_spikes
