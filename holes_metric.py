import numpy as np
import cv2
import nibabel as nib
import os 
from scipy.ndimage import label

def has_holes(ground_truth):
    """
    Detects if there are holes in a ground truth mask.

    Args:
        ground_truth (numpy.ndarray): A binary ground truth mask with values of 0 or 1.
                                      Can be a 2D or 3D array.

    Returns:
        bool: True if there are holes, False otherwise.
    """
    if ground_truth.ndim not in (2, 3):
        raise ValueError("Input must be a 2D or 3D numpy array")

    # Convert the ground truth mask to a binary mask with values of 0 or 255
    mask = (ground_truth * 255).astype(np.uint8)

    if ground_truth.ndim == 2:
        # If the mask is 2D, find contours in the inverted mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # If the mask is 3D, find contours for each slice and stack the results
        contours = []
        for i in range(mask.shape[2]):
            slice_contours, _ = cv2.findContours(mask[:, :, i], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            contours += slice_contours

    # If there are no contours, then there are no holes
    if len(contours) == 0:
        return False

    # Loop through the contours and check if any of them are nested
    for i, contour in enumerate(contours):
        # Check if this contour is nested in another contour
        if hierarchy[0][i][3] != -1:
            return True

    # If no contours are nested, then there are no holes
    return False

def count_connected_components_3d(segmentation_mask):
    # Ensure that the segmentation mask is a numpy array
    segmentation_mask = np.array(segmentation_mask)
    
    # Label the connected components in the 3D segmentation mask using 26-neighborhood connectivity
    labeled_mask, num_components = label(segmentation_mask, np.ones((3,3,3)))
    
    # Return the number of components
    return num_components

nib_img = nib.load(os.path.join("M:\data\labelsTr", "hippocampus334_outputs.nii.gz"))
data = nib_img.get_fdata()

num_components_3d = count_connected_components_3d(data)

print(data.shape)
print(num_components_3d)