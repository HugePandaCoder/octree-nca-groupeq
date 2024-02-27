import nibabel as nib
import numpy as np
import os

def merge_labels(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Loop through each file in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
            # Construct the full file path
            file_path = os.path.join(input_dir, file_name)
            
            # Load the NIfTI file
            nifti_img = nib.load(file_path)
            data = nifti_img.get_fdata()
            
            # Check if the z dimension is exactly 3, as expected
            if data.shape[2] == 3:
                # Merge labels 0 and 1 by finding all instances of label 1 and setting them to 0
                #merged_data = np.where(data[:,:,1] == 1, 0, data[:,:,0])
                merged_data = data[:,:,1] + data[:,:,0]
                merged_data[merged_data > 1] = 1

                # The new data will only have the merged labels, dropping the 3rd dimension
                new_data = merged_data
                
                # Create a new NIfTI image from the modified data
                new_nifti_img = nib.Nifti1Image(new_data, affine=nifti_img.affine)
                
                # Construct the output file path
                output_file_path = os.path.join(output_dir, file_name)
                
                # Save the modified NIfTI file
                nib.save(new_nifti_img, output_file_path)
                print(f"Processed and saved: {output_file_path}")
            else:
                print(f"File {file_name} does not have 3 labels in the z dimension. Skipping.")
        else:
            print(f"Skipping non-NIfTI file: {file_name}")

# Example usage
input_dir = '/home/jkalkhof_locale/Documents/Data/MICCAI24_nnUNet/Padchest_50/labels'  # Update this to your input directory path


#output_dir = '/home/jkalkhof_locale/Documents/Data/MICCAI24_nnUNet/ChestX8_1k/labels_testout'  # Update this to your desired output directory path
output_dir = input_dir

merge_labels(input_dir, output_dir)

input_dir = input_dir + "_test"
output_dir = input_dir

merge_labels(input_dir, output_dir)