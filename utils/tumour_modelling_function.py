#####################################################################
# AUTHOR        Maria Delgado-Ortet
# CONTACT       md863@cam.ac.uk
# INSTITUTION   Department of Radiology, University of Cambridge
# DATE          Nov 2022
#####################################################################

# This file contains the function defined for tumour modelling.

#%% -----------------LIBRARIES--------------
import numpy as np
import os
from skimage import measure
from stl import mesh
import trimesh
import shutil

#%% -----------------FUNCTION--------------
def mesh_and_smooth(mask, stl_filename, save_preproc = False):
    '''
    This function reslices and interpolates the input array to (1, 1, 1) mm voxel size using zero-order spline interpolation.
            INPUTS:
                mask <numpy.ndarray>: Binary array
                stl_filename <str>:   Original DICOM images voxel size.
            OUTPUTS:
                array <numpy.ndarray>: Resliced and interpolated input array to (1, 1, 1). 
    '''
    
    # Surface meshing of the input mask
    vertices, faces = measure.marching_cubes(mask, method = '_lorensen')

    # Center the mesh to (mesh_centroid_x, mesh_centroid_y, 0)
    vertices_center = (np.amax(vertices, 0) + np.amin(vertices, 0))/2
    vertices_center[2] = np.amin(vertices, 0)[2]
    vertices -= vertices_center
    
    # Create the new centered mesh
    mask_preproc = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mask_preproc.vectors[i][j] = vertices[f[j],:]
    
    if save_preproc: # OPT: Set save_preproc to True to save the preprocessed mesh
        if stl_filename[-4:] == '.stl': stl_filename = stl_filename[:-4]
        print('Saving ' + stl_filename + '_preproc.stl...', end = "")
        mask_preproc.save(stl_filename + '_preproc.stl')
        print(' OK')
    
    # Apply Laplacian smoothing to the mesh
    mask_preproc = trimesh.Trimesh(vertices, faces)
    mask_postproc = trimesh.smoothing.filter_laplacian(mask_preproc, 1.0)
    
    # Save the processed mesh
    if stl_filename[-4:] != '.stl': stl_filename += ".stl" # Add .stl if not in the input stl_filename
    print('Saving ' + stl_filename + '...', end = "")
    mask_postproc.export(stl_filename)
    print(' OK')
    
    return mask_postproc