#####################################################################
# AUTHOR        Maria Delgado-Ortet
# CONTACT       md863@cam.ac.uk
# INSTITUTION   Department of Radiology, University of Cambridge
# DATE          Nov 2022
#####################################################################

# This file contains the functions defined for mould modelling.

#%% -----------------LIBRARIES--------------
from scipy.spatial import ConvexHull

#%% -----------------FUNCTIONS--------------
def get_xy_convex_hull_coords(xy_coords):
    '''
    This function returns the coordinates of the convex hull of the input points list.
            INPUTS:
                xy_coords <numpy.ndarray>: (N, 2) array containing the coordinates of all the points
            OUTPUTS:
                xy_convex_hull_cords <numpy.ndarray>: (N, 2) array containing the coordinates of the points of the convex hull
    '''
    
    xy_convex_hull = ConvexHull(xy_coords, incremental = True)
    xy_convex_hull_coords_idx = xy_convex_hull.vertices
    return xy_coords[xy_convex_hull_coords_idx]