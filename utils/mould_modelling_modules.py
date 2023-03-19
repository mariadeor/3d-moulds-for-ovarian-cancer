#####################################################################
# AUTHOR        Maria Delgado-Ortet
# CONTACT       md863@cam.ac.uk
# INSTITUTION   Department of Radiology, University of Cambridge
# DATE          Nov 2022 - Feb 2023
#####################################################################
"""This file contains the functions required to build the mould"""

#%% -----------------LIBRARIES--------------
import math
import os

import numpy as np
from scipy.spatial import ConvexHull
from skimage.draw import polygon2mask
from solid import (
    cube,
    hull,
    import_stl,
    linear_extrude,
    offset,
    polygon,
    scad_render_to_file,
    text,
    translate,
)

from utils.tumour_modelling_modules import mesh_and_smooth


#%% -----------------FUNCTIONS--------------
def build_mould(tumour_wcs):
    """
    This is the main function of this script. It calls the other
    functions and models both the tumour and the mould.
        INPUTS:
            tumour_wcs <np.ndarray>:    Boolean array of the tumour
                                        in the World Coordinate
                                        System (WCS).
    """

    from inputs import mould_id, save_scad_intermediates
    from outputs import dst_dir

    print("TUMOUR MODELLING...")
    tumour_replica_mesh = mesh_and_smooth(tumour_wcs, "tumour_replica")
    print("Tumour modelling complete.\n")

    print("MOULD MODELLING...")
    # Build the mould cavity:
    scad_cavity, cavity_height = build_mould_cavity(tumour_wcs, tumour_replica_mesh)

    # Build the slicing guide:
    scad_slguide = build_slicing_guide(tumour_replica_mesh)

    # Build the perpendicular cutting guides for the slid orientation:
    scad_orguides = build_orientation_guides(tumour_replica_mesh)

    # Ensemble the structures:
    print(
        "\t## Putting the mould together: cavity + slicing guide + orientation guides...",
        end="",
    )
    scad_mould = scad_cavity + scad_slguide + scad_orguides

    if (
        save_scad_intermediates
    ):  # OPT: Add "--save_scad_intermediates" to the command line to save the scad file of the complete mould without slits.
        scad_render_to_file(
            scad_mould,
            os.path.join(dst_dir, "complete_mould_no_slits_" + mould_id + ".scad"),
        )
    print(" OK")

    # Cut the mould structure:
    scad_mould, slicing_slits_positions = carve_slicing_slits(
        scad_mould, tumour_replica_mesh
    )

    # Save the mould:
    scad_render_to_file(
        scad_mould, os.path.join(dst_dir, "mould_" + mould_id + ".scad")
    )

    return cavity_height, slicing_slits_positions


def build_mould_cavity(tumour_wcs, tumour_replica_mesh):
    """
    This function builds the mould cavity of the mould.
        INPUTS:
            tumour_wcs <np.ndarray>:    Boolean array of the tumour
                                        in the World Coordinate
                                        System (WCS) to create the
                                        spiky tumour.
            tumour_replica_mesh <trimesh.Trimesh>:  Smoothed tumour
                                                    to fit in the
                                                    mould.
        OUTPUTS:
            scad_mould <py_scad_obj>:   The mould cavity in SCAD.
            cavity_height <float>:      Cavity height in mm.
    """

    from inputs import (
        baseplate_height,
        baseplate_xy_offset,
        cavity_height_pct,
        cavity_wall_thickness,
        mould_id,
        save_scad_intermediates,
    )
    from outputs import dst_dir

    print(
        "\t## Ensuring the mould base will not close after the slice with the largest area...",
    )
    tumour_w_spikes = make_spiky_tumour(tumour_wcs)
    tumour_w_spikes_mesh = mesh_and_smooth(
        tumour_w_spikes, "tumour_w_spikes", save_preproc=False
    )
    print("OK")

    print("\t## Building the mould cavity...", end="")
    # Import the hull of the "spiky" tumour
    tumour_w_spikes_filename = os.path.join(
        dst_dir, "tumour_w_spikes_" + mould_id + ".stl"
    )
    scad_tumour_convex_hull = hull()(
        import_stl(os.path.split(tumour_w_spikes_filename)[-1])
    )

    # Find the convex hull projection on the xy plane of the "spiky" tumour:
    ## The reason the hull extracted above is not used is because the stl file is centered at (0,0) and this is
    ## the reference while building the mould.
    tumour_xy_coords = np.array(
        [tumour_w_spikes_mesh.vertices[:, 0], tumour_w_spikes_mesh.vertices[:, 1]]
    ).T
    tumour_xy_convex_hull_coords = get_xy_convex_hull_coords(tumour_xy_coords)

    # Create the mould base – a block of shape of the convex hull projection and of height = mouldHeight:
    tumour_sz = tumour_replica_mesh.extents
    cavity_height = cavity_height_pct * tumour_sz[2]
    scad_mould_cavity = linear_extrude(height=cavity_height)(
        offset(r=cavity_wall_thickness)(polygon(tumour_xy_convex_hull_coords))
    )

    # Carve the tumour hull inside the base:
    scad_mould = scad_mould_cavity - scad_tumour_convex_hull

    print("\t## Adding the baseplate to the the mould cavity...", end="")
    baseplate_xy_offset += cavity_wall_thickness
    scad_baseplate = linear_extrude(height=baseplate_height)(
        offset(r=baseplate_xy_offset)(polygon(tumour_xy_convex_hull_coords))
    )

    scad_mould = translate([0, 0, baseplate_height])(scad_mould) + scad_baseplate

    if (
        save_scad_intermediates
    ):  # OPT: Add "--save_scad_intermediates" to the command line to save the scad file of the mould cavity with the baseplate.
        scad_render_to_file(
            scad_mould,
            os.path.join(dst_dir, "mould_cavity_w_baseplate_" + mould_id + ".scad"),
        )
    print(" OK")

    return scad_mould, cavity_height


def build_orientation_guides(tumour_replica_mesh):
    """
    This function builds the orientation guides of the mould.
        INPUTS:
            tumour_replica_mesh <trimesh.Trimesh>:  Smoothed tumour
                                                    to fit in the
                                                    mould.
        OUTPUTS:
            scad_orguide_left + scad_orguide_right <py_scad_obj>:   The orientation guides in SCAD.
    """

    from inputs import (
        baseplate_height,
        baseplate_xy_offset,
        cavity_wall_thickness,
        dist_orguide_baseplate,
        guides_thickness,
        mould_id,
        save_scad_intermediates,
        slguide_height_offset,
    )
    from outputs import dst_dir

    print("\t## Building the orientation guides...", end="")
    tumour_sz = tumour_replica_mesh.extents
    guides_height = tumour_sz[2] + slguide_height_offset - baseplate_height

    scad_orguide = cube(
        [
            guides_thickness,
            2 * guides_thickness,  # As it is only two "pillars", y = 2*guide_thickness.
            guides_height,
        ]
    )

    # Place it on the left of the mould and add translate upwards as baseplate offset:
    scad_orguide_left = translate(
        [
            -(tumour_sz[0] / 2 + cavity_wall_thickness + baseplate_xy_offset + dist_orguide_baseplate + guides_thickness),
            -(2 * guides_thickness) / 2,
            baseplate_height,
        ]
    )(scad_orguide)

    # Place it also on the right of the mould and translate upwards as the baseplate offset:
    scad_orguide_right = translate(
        [
            tumour_sz[0] / 2 + cavity_wall_thickness + baseplate_xy_offset + dist_orguide_baseplate,
            -(2 * guides_thickness) / 2,
            baseplate_height,
        ]
    )(scad_orguide)
    print(" OK")

    print("\t## Adding the baseplate to the orientation guides...", end="")
    # Create the baseplate for the orientation guides:
    scad_orguide_baseplate = cube(
        [
            guides_thickness,
            tumour_sz[1] + 2 * (cavity_wall_thickness + baseplate_xy_offset) + guides_thickness,  # It extends on the y axis to meet with the slicing guide baseplate.
            baseplate_height,
        ]
    )

    # (A) Place the baseplate on the left orientation guide:
    scad_orguide_baseplate_left = translate(
        [
            -(tumour_sz[0] / 2 + cavity_wall_thickness + baseplate_xy_offset + dist_orguide_baseplate + guides_thickness),
            -(tumour_sz[1] + 2 * (cavity_wall_thickness + baseplate_xy_offset)) / 2,
            0,
        ]
    )(scad_orguide_baseplate)

    # (B) Add it to the left orientation guide structure:
    scad_orguide_left += scad_orguide_baseplate_left

    # (C) Place the baseplate on the right orientation guide:
    scad_orguide_baseplate_right = translate(
        [
            (tumour_sz[0] / 2 + cavity_wall_thickness + baseplate_xy_offset + dist_orguide_baseplate),
            -(tumour_sz[1] + 2 * (cavity_wall_thickness + baseplate_xy_offset)) / 2,
            0,
        ]
    )(scad_orguide_baseplate)

    # (D) Add it to the right orientation guide structure:
    scad_orguide_right += scad_orguide_baseplate_right
    print(" OK")

    if (
        save_scad_intermediates
    ):  # OPT: Add "--save_scad_intermediates" to the command line to save the scad file of the orientation guides.
        scad_render_to_file(
            scad_orguide_left + scad_orguide_right,
            os.path.join(dst_dir, "orientation_guides_" + mould_id + ".scad"),
        )

    return scad_orguide_left + scad_orguide_right


def build_slicing_guide(tumour_replica_mesh):
    """
    This function builds the slicing guide of the mould.
        INPUTS:
            tumour_replica_mesh <trimesh.Trimesh>:  Smoothed tumour
                                                    to fit in the
                                                    mould.
        OUTPUTS:
            scad_slguide <py_scad_obj>:   The slicing guide in SCAD.
    """
    from inputs import (
        baseplate_height,
        baseplate_xy_offset,
        cavity_wall_thickness,
        dist_orguide_baseplate,
        guides_thickness,
        mould_id,
        save_scad_intermediates,
        slguide_height_offset,
    )
    from outputs import dst_dir

    print("\t## Building the slicing guide...", end="")
    tumour_sz = tumour_replica_mesh.extents
    guides_height = tumour_sz[2] + slguide_height_offset - baseplate_height
    scad_slguide = cube(
        [
            tumour_sz[0] + 2 * cavity_wall_thickness,  # The slicing guide is as wide as the tumour + the mould cavity walls.
            guides_thickness,
            guides_height,
        ]
    )

    # Translate it: Center with respect to the cavity and upwards as the baseplate_height
    scad_slguide = translate(
        [
            -(tumour_sz[0] + 2 * cavity_wall_thickness) / 2,
            tumour_sz[1] / 2 + cavity_wall_thickness + baseplate_xy_offset,
            baseplate_height,
        ]
    )(scad_slguide)
    print(" OK")

    print("\t## Adding the baseplate to the slicing guide...", end="")
    # Create the baseplate for the slicing guide:
    scad_slguide_baseplate = cube(
        [
            tumour_sz[0] + 2 * cavity_wall_thickness + 2 * baseplate_xy_offset + 2 * dist_orguide_baseplate,  # It extends on the x axis to meet with the orientation guides baseplate.
            guides_thickness,
            baseplate_height,
        ]
    )

    # Center the baseplate:
    scad_slguide_baseplate = translate(
        [
            -(tumour_sz[0] + 2 * cavity_wall_thickness + 2 * baseplate_xy_offset + 2 * dist_orguide_baseplate) / 2,
            tumour_sz[1] / 2 + cavity_wall_thickness + baseplate_xy_offset,
            0,
        ]
    )(scad_slguide_baseplate)

    # Add the baseplate to the slicing guide structure:
    scad_slguide += scad_slguide_baseplate

    if (
        save_scad_intermediates
    ):  # OPT: Add "--save_scad_intermediates" to the command line to save the scad file of the slicing guide.
        scad_render_to_file(
            scad_slguide, os.path.join(dst_dir, "slicing_guide_" + mould_id + ".scad")
        )
    print(" OK")

    return scad_slguide


def carve_slicing_slits(scad_mould, tumour_replica_mesh):
    """
    This function cuts the structure of the mould by carving the
    slicing slits.
        INPUTS:
            scad_mould <py_scad_obj>:               Ensembled mould
                                                    structure.
            tumour_replica_mesh <trimesh.Trimesh>:  Smoothed tumour
                                                    to fit in the
                                                    mould.
        OUTPUTS:
            scad_mould <py_scad_obj>:       The complete carved
                                            mould in SCAD.
            slicing_slits_positions <list>: List with the positions
                                            where the slits have
                                            been placed.
    """

    from inputs import (
        baseplate_height,
        baseplate_xy_offset,
        cavity_height_pct,
        cavity_wall_thickness,
        depth_orslit,
        dist_orguide_baseplate,
        guides_thickness,
        slguide_height_offset,
        slice_thickness,
        slit_thickness,
    )

    print("\t## Carving the slicing slits...", end="")
    tumour_sz = tumour_replica_mesh.extents
    guides_height = tumour_sz[2] + slguide_height_offset - baseplate_height
    cavity_height = cavity_height_pct * tumour_sz[2]

    # Create the slit structure to be carved from the mould to cut along x (slicing guide):
    scad_slicing_slit = cube(
        [
            slit_thickness,
            guides_thickness + 2 * (cavity_wall_thickness + baseplate_xy_offset) + tumour_sz[1],
            guides_height,
        ]
    )

    # Add first cut at the centre:
    scad_slicing_slit_central = translate(
        [
            -slit_thickness / 2,
            -(2 * (cavity_wall_thickness + baseplate_xy_offset) + tumour_sz[1]) / 2,
            baseplate_height,
        ]
    )(scad_slicing_slit)
    scad_mould -= scad_slicing_slit_central

    slicing_slits_positions = [0]  # Initialise a list to keep the slicing slits positions for the generation of the tumour outlines.

    # Make the rest of the cuts:
    nbr_cuts_each_half_x = math.floor(tumour_sz[0] / 2 / slice_thickness)
    for cut in range(nbr_cuts_each_half_x):
        slit_x_position = slice_thickness * (cut + 1)
        slicing_slits_positions.extend(
            [slit_x_position, -slit_x_position]
        )  # Append the slicing slits positions.
        scad_mould -= translate(
            [
                -slit_thickness / 2 + slit_x_position,  # Cuts on the left.
                -(2 * (cavity_wall_thickness + baseplate_xy_offset) + tumour_sz[1]) / 2,
                baseplate_height,
            ]
        )(scad_slicing_slit)
        scad_mould -= translate(
            [
                -slit_thickness / 2 - slit_x_position,  # Cuts on the right.
                -(2 * (cavity_wall_thickness + baseplate_xy_offset) + tumour_sz[1]) / 2,
                baseplate_height,
            ]
        )(scad_slicing_slit)
    slicing_slits_positions.sort(
        reverse=True
    )  # Sort the slicing slits positions list. It is reversed so Cranial is first during the tumour outlines printing.

    # Create the slit structure for the orientation slit:
    scad_orientation_slit = cube(
        [
            2 * guides_thickness + 2 * baseplate_xy_offset + 2 * cavity_wall_thickness + tumour_sz[0] + 2 * dist_orguide_baseplate,
            slit_thickness,
            guides_height,
        ]
    )

    # Center it and position to the approrpiate z height:
    scad_orientation_slit = translate(
        [
            -(2 * guides_thickness + 2 * baseplate_xy_offset + 2 * cavity_wall_thickness + tumour_sz[0] + 2 * dist_orguide_baseplate)/ 2,
            -slice_thickness / 2,
            cavity_height - depth_orslit,
        ]
    )(scad_orientation_slit)

    # Cut:
    scad_mould -= scad_orientation_slit

    # Carve the letters
    font = "Liberation Sans"
    character_depth = 10
    character_size = 0.5 * (slice_thickness - slit_thickness)

    start_pos = nbr_cuts_each_half_x * slice_thickness + slice_thickness / 2
    nbr_cuts = nbr_cuts_each_half_x * 2
    for nbr in range(1, nbr_cuts + 1):
        scad_char = translate(
            [
                -start_pos + slice_thickness * nbr,
                (tumour_sz[1] + 2 * baseplate_xy_offset) / 2 + guides_thickness / 2,
                guides_height + baseplate_height - character_depth,
            ]
        )(
            linear_extrude(height=character_depth)(
                text(
                    str(nbr),
                    size=character_size,
                    font=font,
                    halign="center",
                    valign="center",
                )
            )
        )
        scad_mould -= scad_char
        print(" OK")

        return scad_mould, slicing_slits_positions


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
                                                    points of the convex
                                                    hull.
    """

    xy_convex_hull = ConvexHull(xy_coords, incremental=True)
    xy_convex_hull_coords_idx = xy_convex_hull.vertices

    return xy_coords[xy_convex_hull_coords_idx]


def make_spiky_tumour(tumour_wcs):
    """
    This function makes a copy of the tumour that has the slices above the
    widest surface equal to it.
        INPUTS:
            tumour_wcs <np.ndarray>:    Boolean array of the tumour
                                        in the World Coordinate
                                        System (WCS).
        OUTPUTS:
            tumour_w_spikes <np.ndarray>:   Boolean array of the tumour
                                            in the World Coordinate
                                            System (WCS) with the surfaces
                                            above the widest equal to the
                                            widest surface.
    """

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
    tumour_w_spikes = (
        tumour_wcs.copy()
    )  # The output is a tumour with "spiky" appearance, here the reason of the variable name.
    for z in range(max_area_slice_idx + 1, tumour_w_spikes.shape[2]):
        tumour_w_spikes[:, :, z] = tumour_xy_convex_hull_mask

    return tumour_w_spikes
