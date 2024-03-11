import os
import nibabel as nib
from PIL import Image
import numpy as np

def convert_pngs_to_nifti(input_folder, output_folder, label = False):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            # Construct the full file path
            file_path = os.path.join(input_folder, filename)
            # Open the image file
            img = Image.open(file_path)
            # Check if the image is RGB and convert to grayscale
            if img.mode != 'L':
                img = img.convert('L').rotate(90, expand=True)
            # Convert the image to a numpy array
            img_array = np.flip(np.array(img), 1)
            # Ensure the image is 2D for NIfTI format (no need for np.newaxis as previously)
            
            # Create a NIfTI image

            img_array = img_array
            if label:
                img_array[img_array > 0.5] = 1
                img_array[img_array <= 0.5] = 0
            nifti_img = nib.Nifti1Image(img_array, affine=np.eye(4))
            
            # Construct the output filename
            output_filename = os.path.splitext(filename)[0] + '.nii'
            output_file_path = os.path.join(output_folder, output_filename)
            
            # Save the NIfTI image
            nib.save(nifti_img, output_file_path)
            print(f"Converted {filename} to NIfTI format and saved as {output_filename}")

# Example usage

if False:
    label = False
    input_folder = '/home/jkalkhof_locale/Downloads/MICCAI_png_Test/images_preprocessed_contrast'
    output_folder = '/home/jkalkhof_locale/Downloads/MICCAI_png_Test/images_preprocessed_contrast_nifti'
elif False:
    label = True
    input_folder = '/home/jkalkhof_locale/Downloads/MICCAI_png_Test/images_preprocessed_labels'
    output_folder = '/home/jkalkhof_locale/Downloads/MICCAI_png_Test/images_preprocessed_nifti_labels'
elif False:
    label = True
    input_folder = '/home/jkalkhof_locale/Documents/MICCAI24_finetuning/Custom_finetuning/images_preprocessed_awful_labels'
    output_folder = '/home/jkalkhof_locale/Documents/MICCAI24_finetuning/Custom_finetuning/images_preprocessed_awful_nii_labels'
elif False:
    label = False
    input_folder = '/home/jkalkhof_locale/Documents/MICCAI24_finetuning/Custom_finetuning/images_preprocessed_awful'
    output_folder = '/home/jkalkhof_locale/Documents/MICCAI24_finetuning/Custom_finetuning/images_preprocessed_awful_nii'
elif False:
    label = True
    input_folder = '/home/jkalkhof_locale/Downloads/MICCAI_png_Test/images_preprocessed_blue_labels/'
    output_folder = '/home/jkalkhof_locale/Downloads/MICCAI_png_Test/images_preprocessed_blue_nii_labels'
elif True:
    label = False
    input_folder = '/home/jkalkhof_locale/Downloads/MICCAI_png_Test/images_preprocessed_blue/'
    output_folder = '/home/jkalkhof_locale/Downloads/MICCAI_png_Test/images_preprocessed_blue_nii'

convert_pngs_to_nifti(input_folder, output_folder, label=label)