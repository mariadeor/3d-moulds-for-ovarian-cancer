#####################################################################
# AUTHOR        Maria Delgado-Ortet
# CONTACT       md863@cam.ac.uk
# INSTITUTION   Department of Radiology, University of Cambridge
# DATE          Nov 2022 - Feb 2023
#####################################################################
"""This file contains the function defined for tumour modelling."""

#%% -----------------LIBRARIES--------------
import os

import numpy as np
import trimesh
from skimage import measure
from stl import mesh


#%% -----------------FUNCTION--------------
def mesh_and_smooth(mask, mesh_name, **kwargs):
    """
    This function extracts 3D surface mesh of the input mask volume
    using the Lorensen and Cline marching cubes algorithm. Next, the
    mesh is smoothed using Laplacian smoothing (λ = 1).
        INPUTS:
            mask <numpy.ndarray>:   Binary array
            mesh_name <str>:        ID to identify this mesh when
                                    saved.
        OUTPUTS:
            mesh <trimesh.Trimesh>: Smoothed mesh.
    """

    from inputs import mould_id, save_preproc
    from outputs import dst_dir
    if "save_preproc" in kwargs:
        save_preproc = kwargs["save_preproc"]
    
    stl_filename = os.path.join(dst_dir, mesh_name + "_" + mould_id + ".stl")

    # Surface meshing of the input mask
    vertices, faces = measure.marching_cubes(mask, method="_lorensen")

    # Center the mesh to (mesh_centroid_x, mesh_centroid_y, 0)
    vertices_center = (np.amax(vertices, 0) + np.amin(vertices, 0)) / 2
    vertices_center[2] = np.amin(vertices, 0)[2]
    vertices -= vertices_center

    # Create the new centered mesh
    mask_preproc = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mask_preproc.vectors[i][j] = vertices[f[j], :]

    if save_preproc:  # OPT: Set save_preproc to True to save the preprocessed mesh
        if stl_filename[-4:] == ".stl":
            stl_filename = stl_filename[:-4]
        print("Saving " + stl_filename + "_preproc.stl...", end="")
        mask_preproc.save(stl_filename + "_preproc.stl")
        print(" OK")

    # Apply Laplacian smoothing to the mesh
    mask_preproc = trimesh.Trimesh(vertices, faces)
    mask_postproc = trimesh.smoothing.filter_laplacian(mask_preproc, 1.0)

    # Save the processed mesh
    if stl_filename[-4:] != ".stl":
        stl_filename += ".stl"  # Add .stl if not in the input stl_filename
    print("Saving " + stl_filename + "...", end="")
    mask_postproc.export(stl_filename)
    print(" OK")

    return mask_postproc
