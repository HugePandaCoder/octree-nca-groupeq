import os
import nibabel as nib
from PIL import Image
import numpy as np

def normalize_data(data):
    """Normalize the data to 0-1 range."""
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min)

def convert_nifti_to_png(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.nii', '.nii.gz')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_folder)
                base_name = os.path.splitext(os.path.splitext(relative_path)[0])[0]  # Remove .nii/.nii.gz
                output_path = os.path.join(output_folder, base_name + '.png')

                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Load the NIfTI file
                nifti_image = nib.load(input_path)
                data = nifti_image.get_fdata()

                if data.ndim == 3:  # Assuming the 4th dimension is for channels in masks
                    # Merge the first two channels and discard the last one
                    merged_data = np.mean(data[:, :, :2], axis=2)
                    data = merged_data
                
                # Normalize the data
                normalized_data = normalize_data(data)

                # Convert normalized data to uint8
                data_uint8 = (normalized_data * 255).astype(np.uint8)
                
                # Extract the middle slice along the z-axis
                #middle_index = data_uint8.shape[2] // 2
                slice_2d = data_uint8[:, :]

                # Save the slice as a PNG
                image = Image.fromarray(slice_2d)
                image.save(output_path)
                print(f'Converted and saved: {output_path}')

# Example usage
input_folder = '/home/jkalkhof_locale/Documents/MICCAI24_finetuning/_Convert_ChestX8/'  # Update this path
output_folder = '/home/jkalkhof_locale/Documents/MICCAI24_finetuning/_Converted_ChestX8_png/'  # Update this path
convert_nifti_to_png(input_folder, output_folder)





