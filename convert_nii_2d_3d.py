import os
import nibabel as nib
import numpy as np

def generate_new_save_path(original_path, original_base_dir, new_base_dir):
    """
    Generates a new save path for the file in the same relative location within another folder.
    
    Parameters:
    - original_path: The full path of the original file.
    - original_base_dir: The base directory of the original file.
    - new_base_dir: The base directory where the new file should be saved.
    
    Returns:
    - new_file_path: The full path where the new file should be saved.
    """
    # Get the relative path from the original base directory
    relative_path = os.path.relpath(original_path, original_base_dir)
    # Construct the new path by joining the new base directory with the relative path
    new_path = os.path.join(new_base_dir, relative_path)
    # Modify the file name to append "_3D" before the extension
    base, ext = os.path.splitext(new_path)
    new_file_path = f"{base}" #_3D{ext}
    # Create directories if they do not exist
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    return new_file_path

def convert_2d_to_3d_nifti(original_base_dir, new_base_dir):
    for root, dirs, files in os.walk(original_base_dir):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                file_path = os.path.join(root, file)
                # Load the NIfTI file
                img = nib.load(file_path)
                data = img.get_fdata()
                
                # Check if the image is 2D
                if len(data.shape) == 2:
                    # Add a third dimension
                    data_3d = np.expand_dims(data, axis=-1)
                    print(data_3d.shape)
                    # Create a new NIfTI image (it's important to keep the original affine)
                    img_3d = nib.Nifti1Image(data_3d, img.affine)
                    # Generate new save path
                    new_file_path = generate_new_save_path(file_path, original_base_dir, new_base_dir)
                    # Save the new 3D image
                    #nib.save(img_3d, new_file_path)
                    print(f"Converted {file} to 3D and saved as {new_file_path}")

# Specify the original directory and the new directory to save the converted files
original_directory = "/home/jkalkhof_locale/Documents/Data/MICCAI_24_renamed"
new_directory = "/home/jkalkhof_locale/Documents/Data/MICCAI_24_renamed_3D"
convert_2d_to_3d_nifti(original_directory, new_directory)