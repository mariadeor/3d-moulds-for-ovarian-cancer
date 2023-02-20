#####################################################################
# AUTHOR        Maria Delgado-Ortet
# CONTACT       md863@cam.ac.uk
# INSTITUTION   Department of Radiology, University of Cambridge
# DATE          Feb 2023
#####################################################################
"""Functions for manipulate DICOM files and the masks that derive
from their ROIs."""

#%% -----------------LIBRARIES--------------
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt


#%% -----------------FUNCTIONS--------------
def plot_slices(img, slice_idx_list, fig_title):
    print("%d slices" % len(slice_idx_list))
    for z in slice_idx_list:  # z goes from caudal to cranial
        curr_img = img[:, :, z]

        plt.matshow(curr_img)
        plt.axis("off")
        plt.title(fig_title)

        plt.show(block=False)
        plt.pause(0.001)

    input("INPUT REQUIRED! Please hit enter to close all figures and continue.")
    plt.close('all')
