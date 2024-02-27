import os
import gzip
import shutil

def compress_nii_to_niigz(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.nii'):
                nii_path = os.path.join(root, file)
                niigz_path = nii_path + '.gz'

                # Compress .nii to .nii.gz
                with open(nii_path, 'rb') as f_in:
                    with gzip.open(niigz_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Remove the original .nii file
                os.remove(nii_path)
                print(f"Compressed and removed: {nii_path}")

# Specify the directory to search for .nii files
directory_to_search = "/home/jkalkhof_locale/Documents/Data/MICCAI_24_renamed_3D"
compress_nii_to_niigz(directory_to_search)